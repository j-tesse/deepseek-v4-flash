#!/usr/bin/env bash
# Collapse tune.results into one row per variant; repeated variant names are merged as min-max.
set -euo pipefail
cd "$(dirname "$0")"

# --- configuration ---------------------------------------------------------
RESULTS="tune.results"
# ---------------------------------------------------------------------------

python3 -I - "$RESULTS" <<'EOF'
import re, sys, collections
rows = collections.OrderedDict()
cur = None
for line in open(sys.argv[1]):
    if m := re.match(r'### (\S+) .*?(?:vram=(\d+)MiB)?$', line.strip()):
        cur = rows.setdefault(m[1], collections.defaultdict(list))
        if m[2]: cur['vram'].append(int(m[2]) / 1024)
    elif cur is not None and 'FAILED' in line:
        cur['failed'].append(1)
    elif cur is not None and (m := re.search(r'n=(\d+)\s.*prefill=(\d+).*decode=(\d+)', line)):
        cur[m[1]].append(int(m[2])); cur['dec'].append(int(m[3]))
    elif cur is not None and (m := re.search(r'echo-test: .*decode=(\d+)', line)):
        cur['dec'].append(int(m[1]))

def r(v, f='{:.0f}'):
    if not v: return '-'
    lo, hi = min(v), max(v)
    return f.format(lo) if f.format(lo) == f.format(hi) else f'{f.format(lo)}-{f.format(hi)}'

print('| variant | runs | 0.6k | 4.6k | 18k | decode | VRAM gb |')
print('|---|---|---|---|---|---|---|')
for name, d in rows.items():
    if d['failed'] and not d['512']:
        print(f'| {name} | - | failed to start | | | | |'); continue
    print(f"| {name} | {len(d['512'])} | {r(d['512'])} | {r(d['4096'])} | {r(d['16384'])} | {r(d['dec'])} | {r(d['vram'], '{:.1f}')} |")
EOF
