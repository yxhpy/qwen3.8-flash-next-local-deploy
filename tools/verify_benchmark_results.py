"""Recompute the published 2026-10-26 comparisons from synthetic records."""
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/2026-10-26'
s=json.loads((ROOT/'summary.json').read_text())
data=[json.loads(x) for x in (ROOT/'records.jsonl').read_text().splitlines()]
assert len(data)==s['counted_generation_records']==249
def run(name):return [x for x in data if x['run']==name]
pairs=0
for i in [1,2]:
    a,b=run(f'natural-base-{i}'),run(f'natural-combined-{i}')
    assert len(a)==len(b)==9
    for x,y in zip(a,b):
        assert x['request_sha256']==y['request_sha256']
        if not x['warmup']:
            assert y['timings']['predicted_per_second']>x['timings']['predicted_per_second']
            pairs+=1
assert pairs==12
for kind,d in s['production_configuration_comparison'].items():
    for arm,prefix in [('baseline','natural-base-'),('optimized','natural-combined-')]:
        vals=[x['timings']['predicted_per_second'] for x in data if x['run'].startswith(prefix) and x['kind']==kind and not x['warmup']]
        assert len(vals)==4
        assert median(vals)==d[arm]['median_tokens_per_second']
    assert abs(d['ratio_of_median_speed']-d['optimized']['median_tokens_per_second']/d['baseline']['median_tokens_per_second'])<1e-12
parity=0
for x,y in zip(run('fixed-base-1'),run('fixed-combined')):
    for k in ['request_sha256','text_sha256','logits_sha256','state_sha256']:
        assert x[k]==y[k]
    assert x['logits_count']==y['logits_count']==248320
    parity+=1
for x,y in zip(run('cache-completion-baseline-fixed'),run('cache-proof-optimized-fixed')):
    for k in ['request_sha256','text_sha256','logits_sha256','final_state_sha256','restored_sha256']:
        assert x[k]==y[k]
    parity+=1
assert parity==11
for d in s['long_context']:
    assert abs(d['optimized_restore_plus_first_token_seconds']-(d['optimized_restore_seconds']+d['optimized_first_token_after_restore_seconds']))<1e-12
print(f'PASS: {len(data)} records; {pairs}/12 formal pairs faster; {parity}/11 fixed-plan numerical parity cases; median and cache totals agree.')
