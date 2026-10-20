"""A/B Strata builds on the GPU host: one config, one server per arm, arms alternate over rounds.

    python3 -I strata_ab.py --base ~/Strata/strata-iq3_xxs.json --workdir ~/strata-ab --rounds 5 \
        A:root=~/Strata,exe=~/Strata/engine/strata \
        B:root=~/Strata-v021,exe=~/Strata-v021/build/strata \
        C:root=~/Strata-v021,exe=~/Strata-v021/build/strata,STRATA_ATTN_DAILY=1,...

An arm is NAME:key=value,...  `root` is the Strata checkout whose serve/server.py starts the engine (run by its
.venv, else ~/Strata/.venv), `exe` the engine, `xargs` extra engine arguments separated by ';' (xargs=--spec;0);
every other key goes into the engine's environment.  The first arm is the reference.

Two kinds of run (DETAILS.md "Reproducible greedy output", issues #152 and #410):
  speed (default)   the live flags.  Greedy answers drift from run to run by design: the CPU rounds an expert
                    differently for one token and for several, and the drafts decide which.  So each arm is judged
                    against the reference's own run-to-run spread, and its decode is compared only on the prompts
                    where both arms give the same most common answer.
  --identity --slots N
                    STRATA_IQ_MT_MIN=1 and --prompt-cache 0 --adapt-swaps 0 --pcie-frac 0 on every arm, and
                    --expert-cache N so every arm has the same GPU slots: the answers should repeat byte for byte
                    and can be compared across builds.  --pcie-frac 0 runs every missed expert on the CPU, so the
                    speeds of such a run are not the live ones.

The test config is the base config with host 127.0.0.1, its own port, cwd and logs in --workdir, no
expert_profile_save, and --expert-profile on one frozen copy of the live learned profile: every arm starts from the
same profile and nothing is written back to the live install.  Requests are greedy with thinking off; the long
prompts hide a code number in the middle and ask for it.  Every answer is kept in full (results.jsonl).

Refuses to run while the strata systemd service is active: two engines do not fit in RAM.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path.home()
CODE = "4721"
IDENTITY_ENV = {"STRATA_IQ_MT_MIN": "1"}
IDENTITY_ARGS = (("--prompt-cache", "0"), ("--adapt-swaps", "0"), ("--pcie-frac", "0"))
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
    "用三段话解释为什么天空是蓝色的。",
    "把这句话翻译成英文和日文：今天的会议推迟到下午三点，请大家提前准备材料。",
    "Write a bash script that prints the ten largest files under a directory with human-readable sizes.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the steps.",
    "Write a SQL query that returns each customer's total order amount for 2025, highest first, from tables "
    "customers(id, name) and orders(id, customer_id, amount, created_at).",
    "Summarize the main differences between TCP and UDP in a table.",
    "写一个 JavaScript 防抖函数 debounce(fn, ms)，并给出一个搜索框的使用示例。",
)


def long_prompt(words: int) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = [rng.choice(vocab) for _ in range(words)]
    body.insert(words // 2, f". The secret code is {CODE}. ")
    return ("Here is a list of words:\n" + " ".join(body) +
            "\n\nFirst give the secret code hidden in the list, then summarize what kinds of words appear, in three sentences.")


def parse_arm(s: str) -> dict:
    name, _, kv = s.partition(":")
    opts = dict(x.split("=", 1) for x in kv.split(",") if x)
    root = Path(os.path.expanduser(opts.pop("root", str(HOME / "Strata"))))
    exe = opts.pop("exe", None)
    xargs = [x for x in opts.pop("xargs", "").split(";") if x]
    return {"name": name, "root": root, "exe": os.path.expanduser(exe) if exe else None, "xargs": xargs, "env": opts}


def resolve(cwd: Path, p: str) -> Path:
    return Path(p) if os.path.isabs(p) else cwd / p


def freeze_profile(base: dict, workdir: Path) -> Path:
    """One copy of the live learned profile (else the config's own) for every arm; kept across reruns."""
    dst = workdir / "expert-profile.bin"
    if dst.exists():
        return dst
    cwd = Path(base.get("cwd") or ".")
    cands = []
    if base.get("expert_profile_save"):
        cands.append(resolve(cwd, base["expert_profile_save"]))
    if "--expert-profile" in base["args"]:
        cands.append(resolve(cwd, base["args"][base["args"].index("--expert-profile") + 1]))
    src = next((p for p in cands if p.is_file()), None)
    if src is None:
        sys.exit("no expert profile found in the base config")
    shutil.copy2(src, dst)
    print(f"profile: {src} -> {dst}", flush=True)
    return dst


def set_arg(args: list[str], flag: str, value: str) -> None:
    """Give an engine flag this value: replaced where the config has it, appended where it does not."""
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]


def test_config(base: dict, arm: dict, rnd: int, a, profile: Path) -> dict:
    cfg = json.loads(json.dumps(base))
    args = list(cfg["args"])
    if "--expert-profile-save" in args:
        i = args.index("--expert-profile-save")
        del args[i:i + 2]
    if "--expert-profile" in args:
        args[args.index("--expert-profile") + 1] = str(profile)
    if a.identity:
        for flag, value in IDENTITY_ARGS:
            set_arg(args, flag, value)
        set_arg(args, "--expert-cache", str(a.slots))
    cfg["args"] = args + arm["xargs"]
    for k in ("expert_profile_save", "expert_profile_save_every"):
        cfg.pop(k, None)
    if arm["exe"]:
        cfg["exe"] = arm["exe"]
    cfg.update(host="127.0.0.1", port=a.port, cwd=str(a.workdir), log=str(a.workdir / f"engine_{arm['name']}_r{rnd}.log"))
    return cfg


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def post(port: int, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    post(port, "Say hi.", 8)
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready")


def stop(proc: subprocess.Popen, port: int) -> None:
    proc.terminate()
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    for _ in range(60):                                  # the engine frees ~40 GiB of locked RAM: wait for the port too
        if not port_open(port):
            break
        time.sleep(2)
    time.sleep(10)


def report(rows: list[dict], arms: list[dict], tags: list[str], identity: bool) -> str:
    names = [arm["name"] for arm in arms]
    ref = names[0]
    text_of = {r["sha256"]: r["text"] for r in rows if "sha256" in r}

    def answers(name: str, tag: str) -> list[str]:
        return [r["sha256"] for r in rows if r["arm"] == name and r["req"] == tag and "sha256" in r]

    def common(name: str, tag: str) -> str | None:
        hs = answers(name, tag)
        return collections.Counter(hs).most_common(1)[0][0] if hs else None

    def median(name: str, tag: str, key: str) -> float:
        xs = [r[key] for r in rows if r["arm"] == name and r["req"] == tag and r.get(key)]
        return statistics.median(xs) if xs else float("nan")

    out = ["medians over rounds (decode / prompt tok/s)" +
           (", identity flags: not the live speed (--pcie-frac 0)" if identity else "") + ":"]
    for tag in tags:
        n = next((r["prompt_n"] for r in rows if r["req"] == tag and r.get("prompt_n")), "?")
        cells = [f"{name}: {median(name, tag, 'decode'):6.1f} / {median(name, tag, 'prompt'):7.1f}" for name in names]
        out.append(f"  {tag:7s} ({n} prompt tokens) | " + " | ".join(cells))
    out.append("\nrun-to-run stability (distinct answers per prompt over the rounds; 1 = repeats byte for byte):")
    for name in names:
        counts = {tag: len(set(answers(name, tag))) for tag in tags}
        errs = sum(1 for r in rows if r["arm"] == name and "error" in r)
        stable = sum(1 for c in counts.values() if c == 1)
        out.append(f"  {name}: {stable}/{len(tags)} stable, {errs} errors | " + " ".join(f"{t}={c}" for t, c in counts.items()))
    out.append(f"\nagainst {ref} (judge it by {ref}'s own stability above):")
    for name in names[1:]:
        same = [t for t in tags if common(ref, t) is not None and common(ref, t) == common(name, t)]
        shared = [t for t in tags if set(answers(ref, t)) & set(answers(name, t))]
        line = f"  {name}: same most common answer on {len(same)}/{len(tags)}, an answer in common on {len(shared)}/{len(tags)}"
        ratios = [median(name, t, "decode") / median(ref, t, "decode") for t in same]
        ratios = [x for x in ratios if x == x]
        if ratios and not identity:
            line += f"; decode on those prompts x{statistics.mean(ratios):.3f} (mean of per-prompt median ratios" + \
                    (f", only {len(ratios)} prompts: too few, judge speed by the medians above)" if len(ratios) < 4 else ")")
        out.append(line)
        for t in [t for t in tags if t not in shared][:5]:
            ra, rb = text_of.get(common(ref, t)) or "", text_of.get(common(name, t)) or ""
            at = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
            out.append(f"    {t}: differs at char {at}: {ra[at:at + 40]!r} vs {rb[at:at + 40]!r}")
    longs = [r for r in rows if r["req"].startswith("long") and "text" in r]
    found = sum(1 for r in longs if CODE in r["text"])
    out.append(f"\nlong prompts that give the code {CODE}: {found}/{len(longs)}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=HOME / "Strata" / "strata-iq3_xxs.json")
    ap.add_argument("--workdir", type=Path, default=HOME / "strata-ab")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--long-words", default="24000,75000", help="comma-separated word counts of the long prompts ('' for none)")
    ap.add_argument("--identity", action="store_true", help="the reproducible-greedy flags on every arm (needs --slots)")
    ap.add_argument("--slots", type=int, help="--expert-cache N for every arm with --identity (the smallest auto count of the arms)")
    a = ap.parse_args()
    if a.identity and not a.slots:
        sys.exit("--identity needs --slots N: one --expert-cache size for every arm "
                 "(the smallest 'expert cache N slots' line of the arms' auto starts)")
    if subprocess.run(["systemctl", "is-active", "--quiet", "strata"]).returncode == 0:
        sys.exit("the strata service is running: stop it first (sudo systemctl stop strata)")
    if port_open(a.port):
        sys.exit(f"port {a.port} is in use")
    a.workdir = a.workdir.expanduser().resolve()
    a.workdir.mkdir(parents=True, exist_ok=True)
    base = json.loads(a.base.expanduser().read_text())
    profile = freeze_profile(base, a.workdir)
    arms = [parse_arm(s) for s in a.arms]
    reqs = [(f"chat{i}", p, a.max_new) for i, p in enumerate(PROMPTS)]
    reqs += [(f"long{w // 1000}k", long_prompt(int(w)), 128) for w in (int(x) for x in a.long_words.split(",") if x)]
    env0 = {k: v for k, v in os.environ.items() if k != "STRATA_API_KEY"}
    env_id = IDENTITY_ENV if a.identity else {}
    rows: list[dict] = []
    with open(a.workdir / "results.jsonl", "a") as out:
        for rnd in range(a.rounds):
            for arm in (arms if rnd % 2 == 0 else arms[::-1]):
                cfg = test_config(base, arm, rnd, a, profile)
                cpath = a.workdir / f"config_{arm['name']}.json"
                cpath.write_text(json.dumps(cfg, indent=1))
                py = arm["root"] / ".venv" / "bin" / "python"
                py = py if py.exists() else HOME / "Strata" / ".venv" / "bin" / "python"
                slog = open(a.workdir / f"server_{arm['name']}_r{rnd}.log", "w")
                proc = subprocess.Popen([str(py), str(arm["root"] / "serve" / "server.py"), "--engine", "strata",
                                         "--config", str(cpath), "--port", str(a.port)],
                                        cwd=str(arm["root"]), env={**env0, **env_id, **arm["env"]}, stdin=subprocess.DEVNULL,
                                        stdout=slog, stderr=subprocess.STDOUT)
                try:
                    t0 = time.time()
                    wait_ready(a.port, proc)
                    print(f"round {rnd} {arm['name']}: ready in {time.time() - t0:.0f} s", flush=True)
                    for tag, prompt, max_new in reqs:
                        row = {"round": rnd, "arm": arm["name"], "req": tag, "identity": a.identity}
                        try:
                            r = post(a.port, prompt, max_new)
                            t = r.get("timings", {})
                            text = r["choices"][0]["message"].get("content") or ""
                            row.update(decode=t.get("predicted_per_second"), prompt=t.get("prompt_per_second"),
                                       prompt_n=t.get("prompt_n"), n=r.get("usage", {}).get("completion_tokens"),
                                       finish=r["choices"][0].get("finish_reason"), text=text,
                                       sha256=hashlib.sha256(text.encode()).hexdigest())
                            print(f"  {tag:7s} decode {row['decode']} prompt {row['prompt']} ({row['prompt_n']}) n {row['n']}",
                                  flush=True)
                        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                            row["error"] = repr(e)
                            print(f"  {tag:7s} ERROR {e!r}", flush=True)
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        rows.append(row)
                except (RuntimeError, TimeoutError) as e:
                    print(f"round {rnd} {arm['name']}: {e} (see {slog.name})", flush=True)
                finally:
                    stop(proc, a.port)
                    slog.close()
    text = report(rows, arms, [t for t, _, _ in reqs], a.identity)
    (a.workdir / "report.txt").write_text(text + "\n")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
