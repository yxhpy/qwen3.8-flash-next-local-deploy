#!/usr/bin/env bash
# 统计 GPU 主机上 Strata 某一次启动以来的长 prompt 冷读、会话缓存停放/恢复和引擎堆内存。
# 用法：tools/strata_cache_stats.sh [N]   N=1 当前这次启动（默认），N=2 上一次启动，以此类推
# GPU 主机的 ssh 名称用 STRATA_HOST 指定（默认 gpu-linux）；Strata 装在主机用户的 ~/Strata。
# 只读：不读密钥和对话内容。
set -euo pipefail
N="${1:-1}"
ssh -o BatchMode=yes -o ConnectTimeout=10 "${STRATA_HOST:-gpu-linux}" "python3 -I - $N" <<'PY'
import os, re, subprocess, sys
from datetime import datetime

n = int(sys.argv[1])
log = os.path.expanduser('~/Strata/strata-iq3_xxs.log')
with open(log, errors='replace') as f:
    lines = f.read().splitlines()
starts = [i for i, l in enumerate(lines) if 'engine started:' in l]
if len(starts) < n:
    sys.exit('log has only %d engine starts' % len(starts))
a = starts[-n]
b = starts[-n + 1] if n > 1 else len(lines)
seg = lines[a:b]


def ts(line):
    m = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')


t0 = ts(seg[0])
t1 = ts(lines[b]) if n > 1 else datetime.now()
hours = max((t1 - t0).total_seconds() / 3600, 1e-9)

# 请求行：prompt P tokens = R reused + X read in MS ms (...), G generated in GMS ms
prompt = re.compile(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms.*?, (\d+) generated in (\d+) ms')
req = read_ms = cold = cold_ms = warm = warm_ms = gen = gen_ms = 0
for l in seg:
    m = prompt.match(l)
    if not m:
        continue
    p, r, _, ms, g, gms = map(int, m.groups())
    req += 1
    read_ms += ms
    gen += g
    gen_ms += gms
    if p >= 32768:
        if r == 0:
            cold += 1
            cold_ms += ms
        else:
            warm += 1
            warm_ms += ms

cache = [l for l in seg if 'conversation cache:' in l]


def cnt(s):
    return sum(s in l for l in cache)


last_bytes = None
for l in reversed(cache):
    m = re.search(r' bytes=(\d+)', l)
    if m:
        last_bytes = int(m.group(1))
        break

print('since %s to %s (%.1f h)' % (t0, t1.strftime('%Y-%m-%d %H:%M:%S'), hours))
print('requests %d, prompt read %.1f min' % (req, read_ms / 60000))
print('>=32K cold (0 reused): %d (%.1f/day), %.1f min, avg %.0f s' % (
    cold, cold * 24 / hours, cold_ms / 60000, cold_ms / cold / 1000 if cold else 0))
print('>=32K warm: %d, avg %.1f s' % (warm, warm_ms / warm / 1000 if warm else 0))
print('cold share of prompt read time: %.0f%%' % (100 * cold_ms / read_ms if read_ms else 0))
print('decode avg: %.1f tok/s' % (gen / (gen_ms / 1000) if gen_ms else 0))
print('cache: parked %d, restored %d, skip parking %d (RAM floor %d), evict lines %d' % (
    cnt('conversation cache: parked'), cnt('conversation cache: restored'),
    cnt('skip parking'), cnt('skip parking (physical RAM'), cnt('conversation cache: evicted')))

if n == 1:
    pid = subprocess.run(['pgrep', '-f', '^' + os.path.expanduser('~/Strata/engine/') + '[s]trata --serve'],
                         capture_output=True, text=True).stdout.split()
    if pid:
        heap = 0
        in_heap = False
        with open('/proc/%s/smaps' % pid[0]) as f:
            for line in f:
                if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
                    in_heap = line.rstrip().endswith('[heap]')
                elif in_heap and line.startswith('Rss:'):
                    heap += int(line.split()[1]) * 1024
        print('engine heap RSS: %.2f GiB, cache bytes (last log): %s' % (
            heap / 2**30, '%.2f GiB' % (last_bytes / 2**30) if last_bytes is not None else 'n/a'))
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                print('MemAvailable: %.1f GiB' % (int(line.split()[1]) / 2**20))
PY
