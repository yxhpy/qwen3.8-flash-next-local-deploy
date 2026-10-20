"""Conversation-cache soak over HTTP: does a build keep parking and restoring long conversations, and give the
freed memory back, under the live cache settings?  (#3115: from 0.1.45 evicted snapshots can stay in glibc's heap,
and the RAM floor then refuses to park.)

    python3 -I strata_cache_soak.py --workdir ~/strata-soak --rounds 6 \
        A:root=~/Strata,exe=~/Strata/engine/strata \
        B:root=~/Strata-v021,exe=~/Strata-v021/build/strata

Arms are those of strata_ab.py (same test config: 127.0.0.1, own cwd and logs, one frozen profile, nothing written
back to the live install) and run one after the other, each on a fresh engine with the base config's cache flags
and env (MALLOC_MMAP_THRESHOLD_ included).  Every arm gets the same conversations: --tokens long documents, each
followed by short questions, taken in turn for --rounds rounds, so the parked snapshots add up past the cache's
budget and some are evicted.  After each request it reads the engine log's prompt and conversation-cache lines and
samples the engine's [heap] RSS (as tools/strata_cache_stats.sh does), the cache bytes and MemAvailable.

Refuses to run while the strata systemd service is active: two engines do not fit in RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_ab as AB  # noqa: E402  (same folder: the arms, the test config, the server start and stop)

PROMPT = re.compile(r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms')
VOCAB = ('river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember '
         'ledger quartz meadow anchor violet thunder falcon pepper marble island cotton silver bridge').split()
QUESTIONS = (
    'Which three words appear most often in document {k}? Answer in one line.',
    'Name two words from document {k} that start with the same letter.',
    'In one sentence, what kinds of words does document {k} contain?',
    'Give five words from document {k}, separated by commas.',
    'Which word in document {k} is the longest?',
    'Does the word "harbor" appear in document {k}? Answer yes or no.',
)
EVENTS = (('parked', 'conversation cache: parked'), ('restored', 'conversation cache: restored'),
          ('evicted', 'conversation cache: evicted'), ('skip', 'skip parking'),
          ('skip_ram', 'skip parking (physical RAM'))


def document(k: int, tokens: int) -> str:
    rng = random.Random(100 + k)
    words = ' '.join(rng.choice(VOCAB) for _ in range(int(tokens / 1.05)))
    return f'Here is document {k}:\n{words}\n\n'


def post(port: int, messages: list[dict], max_tokens: int) -> dict:
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}/v1/chat/completions',
        data=json.dumps({'messages': messages, 'max_tokens': max_tokens, 'temperature': 0,
                         'chat_template_kwargs': {'enable_thinking': False}}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def tail(path: str, pos: int) -> tuple[list[str], int]:
    try:
        with open(path, errors='replace') as f:
            f.seek(pos)
            data = f.read()
            return data.splitlines(), f.tell()
    except FileNotFoundError:
        return [], pos


def engine_pid(exe: str) -> str | None:
    out = subprocess.run(['pgrep', '-f', '^' + exe.replace('.', r'\.') + ' --serve'],
                         capture_output=True, text=True).stdout.split()
    return out[-1] if out else None


def heap_rss(pid: str | None) -> int | None:
    """The engine's [heap] RSS in bytes, read as tools/strata_cache_stats.sh does."""
    if not pid:
        return None
    heap, in_heap = 0, False
    try:
        with open(f'/proc/{pid}/smaps') as f:
            for line in f:
                if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
                    in_heap = line.rstrip().endswith('[heap]')
                elif in_heap and line.startswith('Rss:'):
                    heap += int(line.split()[1]) * 1024
    except OSError:
        return None
    return heap


def mem_available() -> int:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    return 0


def gib(x: int | None) -> str:
    return f'{x / 2**30:.2f}' if x is not None else 'n/a'


def run_arm(arm: dict, a, base: dict, profile: Path, out) -> list[dict]:
    cfg = AB.test_config(base, arm, 0, a, profile)
    cpath = a.workdir / f'config_{arm["name"]}.json'
    cpath.write_text(json.dumps(cfg, indent=1))
    py = arm['root'] / '.venv' / 'bin' / 'python'
    py = py if py.exists() else AB.HOME / 'Strata' / '.venv' / 'bin' / 'python'
    env0 = {k: v for k, v in os.environ.items() if k != 'STRATA_API_KEY'}
    slog = open(a.workdir / f'server_{arm["name"]}.log', 'w')
    proc = subprocess.Popen([str(py), str(arm['root'] / 'serve' / 'server.py'), '--engine', 'strata',
                             '--config', str(cpath), '--port', str(a.port)],
                            cwd=str(arm['root']), env={**env0, **arm['env']}, stdin=subprocess.DEVNULL,
                            stdout=slog, stderr=subprocess.STDOUT)
    rows: list[dict] = []
    try:
        t0 = time.time()
        AB.wait_ready(a.port, proc)
        print(f'{arm["name"]}: ready in {time.time() - t0:.0f} s', flush=True)
        pid = engine_pid(cfg['exe'])
        _, pos = tail(cfg['log'], 0)                 # the start-up and warm-up lines are not counted
        sizes = [int(x) for x in a.tokens.split(',')]
        convs = [[{'role': 'user', 'content': document(k, n) + QUESTIONS[0].format(k=k)}]
                 for k, n in enumerate(sizes, 1)]
        cache_bytes = None
        deadline = time.time() + a.arm_timeout * 60
        for rnd in range(a.rounds):
            for k, msgs in enumerate(convs, 1):
                if time.time() > deadline:
                    print(f'{arm["name"]}: --arm-timeout reached, stopping this arm', flush=True)
                    return rows
                row = {'arm': arm['name'], 'round': rnd, 'conv': k}
                t1 = time.time()
                try:
                    r = post(a.port, msgs, a.max_new)
                except (urllib.error.URLError, OSError, ValueError) as e:
                    row['error'] = repr(e)
                    print(f'  r{rnd} c{k} ERROR {e!r}', flush=True)
                    out.write(json.dumps(row) + '\n')
                    rows.append(row)
                    if proc.poll() is not None:
                        return rows
                    continue
                row['wall_s'] = round(time.time() - t1, 1)
                msgs.append({'role': 'assistant', 'content': r['choices'][0]['message'].get('content') or ''})
                msgs.append({'role': 'user', 'content': QUESTIONS[(rnd + 1) % len(QUESTIONS)].format(k=k)})
                time.sleep(1)                         # the engine's lines of this request reach its log
                lines, pos = tail(cfg['log'], pos)
                m = None
                for line in lines:
                    m = PROMPT.search(line) or m
                if m:
                    row.update(prompt=int(m.group(1)), reused=int(m.group(2)), read=int(m.group(3)),
                               read_ms=int(m.group(4)))
                cache = [line for line in lines if 'conversation cache:' in line]
                for key, s in EVENTS:
                    row[key] = sum(s in line for line in cache)
                for line in cache:
                    b = re.search(r' bytes=(\d+)', line)
                    cache_bytes = int(b.group(1)) if b else cache_bytes
                row.update(heap=heap_rss(pid), cache_bytes=cache_bytes, mem_avail=mem_available())
                out.write(json.dumps(row) + '\n')
                out.flush()
                rows.append(row)
                print(f'  r{rnd} c{k} prompt {row.get("prompt")} reused {row.get("reused")} read {row.get("read_ms")} ms'
                      f' | parked {row["parked"]} restored {row["restored"]} evicted {row["evicted"]} skip {row["skip"]}'
                      f' | heap {gib(row["heap"])} cache {gib(cache_bytes)} avail {gib(row["mem_avail"])} GiB', flush=True)
    except (RuntimeError, TimeoutError) as e:
        print(f'{arm["name"]}: {e} (see {slog.name})', flush=True)
    finally:
        AB.stop(proc, a.port)
        slog.close()
    return rows


def report(rows: list[dict], names: list[str]) -> str:
    out = ['conversation-cache soak (round 0 reads every document fresh; later rounds should restore):']
    for name in names:
        rs = [r for r in rows if r['arm'] == name]
        later = [r for r in rs if 'reused' in r and r['round'] > 0]
        hits = [r for r in later if r['reused'] > 0]
        tot = {key: sum(r.get(key, 0) for r in rs) for key, _ in EVENTS}
        heaps = [r['heap'] for r in rs if r.get('heap') is not None]
        cbytes = [r['cache_bytes'] for r in rs if r.get('cache_bytes') is not None]
        avail = [r['mem_avail'] for r in rs if r.get('mem_avail')]
        out.append(f'  {name}: {len(rs)} turns, {sum("error" in r for r in rs)} errors; later rounds: '
                   f'{len(hits)}/{len(later)} restored, {len(later) - len(hits)} read again, '
                   f'prompt read {sum(r["read_ms"] for r in later) / 1000:.0f} s')
        out.append(f'      cache lines: parked {tot["parked"]}, restored {tot["restored"]}, evicted {tot["evicted"]}, '
                   f'skip parking {tot["skip"]} (RAM floor {tot["skip_ram"]})')
        out.append(f'      heap RSS max {gib(max(heaps) if heaps else None)} / last {gib(heaps[-1] if heaps else None)} GiB; '
                   f'cache bytes max {gib(max(cbytes) if cbytes else None)} / last {gib(cbytes[-1] if cbytes else None)} GiB; '
                   f'MemAvailable min {gib(min(avail) if avail else None)} GiB')
    out.append("\na build passes if its later-round restores, RAM-floor refusals and heap RSS are no worse than the "
               "reference (the first arm's); a heap far above the cache bytes is #3115's sign")
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=AB.HOME / 'Strata' / 'strata-iq3_xxs.json')
    ap.add_argument('--workdir', type=Path, default=AB.HOME / 'strata-soak')
    ap.add_argument('arms', nargs='+')
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--tokens', default='50000,80000,110000,150000', help='approximate prompt tokens of each conversation')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--arm-timeout', type=float, default=40, help='minutes per arm before it stops early')
    a = ap.parse_args()
    a.identity, a.slots = False, None              # strata_ab's test config with the live flags
    if subprocess.run(['systemctl', 'is-active', '--quiet', 'strata']).returncode == 0:
        sys.exit('the strata service is running: stop it first (sudo systemctl stop strata)')
    if AB.port_open(a.port):
        sys.exit(f'port {a.port} is in use')
    a.workdir = a.workdir.expanduser().resolve()
    a.workdir.mkdir(parents=True, exist_ok=True)
    base = json.loads(a.base.expanduser().read_text())
    profile = AB.freeze_profile(base, a.workdir)
    arms = [AB.parse_arm(s) for s in a.arms]
    rows: list[dict] = []
    with open(a.workdir / 'soak.jsonl', 'a') as out:
        for arm in arms:
            rows += run_arm(arm, a, base, profile, out)
    text = report(rows, [arm['name'] for arm in arms])
    (a.workdir / 'soak-report.txt').write_text(text + '\n')
    print('\n' + text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
