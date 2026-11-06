#!/usr/bin/env bash
# Sweep llama-server flags and benchmark each. Restarts the container per variant,
# so do not run this while agents depend on :8000. Fold the winner into start.sh.
set -euo pipefail
cd "$(dirname "$0")"

# --- configuration ---------------------------------------------------------
IMAGE="deepseek-v4-flash:latest"
CONTAINER="llama_tune"
MODELS_DIR="/home/joran/models/gguf"
QUANT="UD-IQ3_XXS"
MODEL_FILE="DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
ALIAS="DeepSeek-V4-Flash-0731"
PORT=8000
CONTEXT_SIZE=32768
THREADS=12
RESULTS="tune.results"

# name|extra llama-server args
VARIANTS=(
  "current         |-ncmoe 34 -b 6144 -ub 6144"
)
# ---------------------------------------------------------------------------

C='\033[96m'; G='\033[92m'; R='\033[91m'; RS='\033[0m'
say() { echo -e "${C}==>${RS} $*"; }

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
vllm=$(docker ps -q --filter "name=vllm_")
trap 'cleanup; [ -z "$vllm" ] || docker start $vllm >/dev/null' EXIT

[ -z "$vllm" ] || docker stop $vllm >/dev/null
docker ps -q --filter "name=llama_ds_v4_flash" | xargs -r docker stop >/dev/null

: >"$RESULTS"
for v in "${VARIANTS[@]}"; do
  name="${v%%|*}"; args="${v#*|}"; name="${name// /}"
  say "variant ${name}: ${args}"
  cleanup
  sleep 3

  # shellcheck disable=SC2086
  docker run -d --name "$CONTAINER" \
    --runtime nvidia --gpus all --restart no \
    -v "${MODELS_DIR}:/models:ro" -p "${PORT}:${PORT}" --ipc=host \
    --health-cmd "curl -f http://localhost:${PORT}/health || exit 1" \
    --health-interval 30s --health-start-interval 5s --health-timeout 5s \
    --health-retries 120 --health-start-period 30s \
    "$IMAGE" \
    --model "/models/${QUANT}/${MODEL_FILE}" --alias "$ALIAS" \
    --n-gpu-layers 999 --threads "$THREADS" --ctx-size "$CONTEXT_SIZE" \
    --parallel 1 --jinja --flash-attn on \
    --temp 1.0 --top-p 0.95 --min-p 0.0 \
    --host 0.0.0.0 --port "$PORT" $args >/dev/null

  ok=0
  for _ in $(seq 180); do
    s=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo gone)
    [ "$s" = healthy ] && { ok=1; break; }
    docker ps -q -f name="^${CONTAINER}$" | grep -q . || break
    sleep 2
  done
  if [ "$ok" != 1 ]; then
    echo -e "${R}${name}: failed to start${RS}"
    { echo "### $name ($args)"; echo "FAILED TO START"; docker logs --tail 15 "$CONTAINER" 2>&1; } >>"$RESULTS"
    continue
  fi

  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  { echo "### $name ($args)  vram=${vram}MiB"
    ./bench.sh 2>&1 | grep -E "n=|avg sm"
    echo -n "  echo-test: "; ./bench-spec.sh 2>&1 | head -1
    docker stop -t 30 "$CONTAINER" >/dev/null; docker logs "$CONTAINER" 2>&1 | grep -E "MoE cache size|slots|hit rate" || true; } >>"$RESULTS"
  tail -n 8 "$RESULTS"
done

say "results in ${RESULTS}"
cat "$RESULTS"
