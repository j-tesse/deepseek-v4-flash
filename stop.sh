#!/usr/bin/env bash
# Stop DeepSeek V4 Flash and hand the GPU back to the vLLM daily driver.
#   ./stop.sh            stop + restore vLLM
#   ./stop.sh --no-restore   just stop, leave the GPU free
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER="llama_ds_v4_flash"

C='\033[96m'; Y='\033[93m'; RS='\033[0m'
say() { echo -e "${C}==>${RS} $*"; }

if [ "$(docker ps -q -f name="^${CONTAINER}$")" ]; then
  say "stopping $CONTAINER"
  docker stop "$CONTAINER" >/dev/null
else
  say "$CONTAINER not running"
fi

for a in "$@"; do
  [ "$a" = --no-restore ] && { say "leaving GPU free"; exit 0; }
done

running=$(docker ps --filter name=vllm_ --format '{{.Names}}')
[ -z "$running" ] || { say "$running already running"; exit 0; }

# the most recently stopped vllm_* — i.e. the one start.sh stopped
VLLM_CONTAINER=$(docker ps -aq --filter name=vllm_ --filter status=exited \
  | xargs -r docker inspect --format '{{.State.FinishedAt}} {{.Name}}' | sort -r | head -1 | cut -d/ -f2)
if [ -n "$VLLM_CONTAINER" ]; then
  say "restarting $VLLM_CONTAINER"
  docker start "$VLLM_CONTAINER" >/dev/null
  say "waiting for vLLM to come up"
  for _ in $(seq 150); do
    [ "$(docker inspect --format='{{.State.Health.Status}}' "$VLLM_CONTAINER" 2>/dev/null)" = healthy ] && break
    sleep 2
  done

  if [ -x /home/joran/scripts/update-agent-models.sh ]; then
    /home/joran/scripts/update-agent-models.sh || echo -e "${Y}agent sync failed — re-run ~/scripts/update-agent-models.sh once vLLM is healthy${RS}"
  fi
else
  echo -e "${Y}$VLLM_CONTAINER not found — start a model with ~/scripts/vllm.sh${RS}"
fi
