#!/usr/bin/env bash
# Correctness check for large ubatches + expert offload: hide needles across a long prompt
# (spanning several ubatches) and require all of them back.
set -euo pipefail
cd "$(dirname "$0")"

# --- configuration ---------------------------------------------------------
ENDPOINT="http://localhost:8000"
MODEL="DeepSeek-V4-Flash-0731"
PROMPT_WORDS=20000   # ~20k tokens: 4 ubatches at -ub 6144
# ---------------------------------------------------------------------------

python3 -I - "$PROMPT_WORDS" "$MODEL" <<'EOF' | curl -s "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' -d @- >/tmp/needle.json
import json, random, sys
n = int(sys.argv[1]); random.seed(7)
w = ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta']
words = [random.choice(w) for _ in range(n)]
needles = {0.1: 'The vault code is 4817.', 0.5: 'The ship is named Coriander.', 0.9: 'The secret city is Varnholt.'}
for f, s in sorted(needles.items(), reverse=True): words.insert(int(n*f), s)
q = ' '.join(words) + '\n\nWhat is the vault code, the ship name, and the secret city? Answer in one line.'
print(json.dumps({'model': sys.argv[2], 'messages': [{'role': 'user', 'content': q}], 'max_tokens': 2000, 'cache_prompt': False}))
EOF

answer=$(jq -r '.choices[0].message.content' /tmp/needle.json)
echo "prompt=$(jq -r '.timings.prompt_n' /tmp/needle.json) tokens: $answer"
ok=1
for k in 4817 Coriander Varnholt; do grep -qi "$k" <<<"$answer" || { echo "MISSING $k"; ok=0; }; done
[ "$ok" = 1 ] && echo PASS || { echo FAIL; exit 1; }
