# DeepSeek V4 Flash
Purpose of this repo is to run a quantized version of DeekSeek V4 Flash 0731 using `llama.cpp` in docker.

Serves an OpenAI-compatible API on **port 8000** as a drop-in replacement for the vLLM service.

## Machine
- headless machine running Ubuntu Server
- 96gb system memory - not all of it is available (~91gb total, ~81gb available)
- RTX 5090 with 32gb VRAM - **shared, not exclusive**: `vllm_gemma4-31b-nvfp4-mtp` normally owns
  all of it (`--gpu-memory-utilization 0.982`, `restart: unless-stopped`). One GPU LLM at a time.
  `start.sh` stops vLLM, `stop.sh` restores it.
- docker and nvidia docker tools already installed
- Ryzen 9 9900X (12C/24T), driver 610.43.02 / CUDA 13.3, GPU is Blackwell **sm_120**
- Crucial T705 Gen5 NVMe, ~670gb free

## Architecture
```
deepseek-v4-flash/
├── llama.cpp/          upstream submodule, pinned to a release tag (b10244). Never edited.
├── build.sh            builds upstream's .devops/cuda.Dockerfile (target: server) via build args
├── build-async.sh      start|status [--json]|wait|log|stop — build takes ~10-20min
├── update.sh           move submodule to a newer tag (ff-only, refuses dirty tree) + rebuild
├── download-model.sh   throw-away container pulls the GGUF into ~/models/gguf
├── verify-model.sh     shard count + GGUF magic check
├── start.sh            free GPU → run container → wait healthy → re-point agents
├── stop.sh             stop container → restart vLLM  (`--no-restore` to skip)
├── bench.sh            single-request prefill/decode + PCIe counters against :8000
├── bench-spec.sh       echo-a-file decode bench — the case speculative decoding should win
├── tune.sh             sweep server flags, restarting the container per variant → tune.results
└── build.number/history
```

No custom Dockerfile: upstream's `.devops/cuda.Dockerfile` already does what we need, driven by
`--build-arg CUDA_VERSION` / `CUDA_DOCKER_ARCH`. If a patch ever becomes necessary, add
`<file>.patched.<ext>` here plus a thin Dockerfile that `COPY`s it over the submodule — never edit
`llama.cpp/` in place.

Image tags per build: `deepseek-v4-flash:b<N>`, `:<upstream-describe>`, `:latest`.
`build.number` bumps only on success.

## Model
`unsloth/DeepSeek-V4-Flash-0731-GGUF`, quant **UD-IQ3_XXS** (104gb, 4 shards).
284B total / 13B active MoE, 43 layers, 256 routed experts (6 active), 1M ctx max.

**Routed experts are ~97.5% of the weights** (~277B of 284B); the dense part is only ~6.5B. So the
dense layers + KV cache go on the GPU and the experts are split GPU/RAM with `--n-cpu-moe`.

Fit: 81gb RAM + 32gb VRAM = 113gb usable > 104gb.
- GPU: ~8gb dense + ~4gb KV + ~0.6gb overhead → ~19gb of experts
- experts ≈ 96gb / 43 layers ≈ 2.23gb per layer → ~8 expert layers fit on GPU
- `--n-cpu-moe 35` leaves ~78gb of experts in RAM vs ~85gb free. **Headroom is ~7gb — tight.**

Measured 2026-08-04 at `N_CPU_MOE=36`, `-ub 2048`: **23-24 tok/s decode**, **310 / 715 / 808 tok/s
prefill** at 0.6k / 4.6k / 18k-token prompts. For reference a DGX Spark managed 15.5-17.4 tok/s decode
/ 148 tok/s prompt on a comparable quant. Expect a slower first request after each start — mmap pages
the weights in from NVMe lazily (the server reports healthy in ~16s, long before the weights are
resident).

### Prefill is PCIe-bound, not CPU-bound
llama.cpp's op-offload streams the RAM-resident expert weights to the GPU **once per ubatch** (measured
7-16 GB/s sustained PCIe rx during prefill, ~26gb per 512-token ubatch). The cost is per-ubatch and
fixed, so it amortises over ubatch size — this is the single biggest lever in the whole setup.
Raising `N_CPU_MOE` by one buys the VRAM headroom that makes the big ubatch pay off.

`./tune.sh` sweeps flags and `./bench.sh` measures; measured 2026-08-04, prefill tok/s by prompt size:
| `-ncmoe` / `-ub` | 0.6k | 4.6k | 18k | decode | VRAM |
|---|---|---|---|---|---|
| 35 / 512 (old default) | 172 | 325 | 332 | 22 | 25.6gb |
| 35 / 2048 | 236 | 592 | 759 | 22 | 26.1gb |
| **36 / 2048 (current)** | **310** | **715** | **808** | **23-24** | **24.2gb** |
| 36 / 3072 | 223 | 696 | 903 | 21 | — |
| 36 / 4096 | 224 | 666 | 925 | 22 | — |
| 34 / 2048 | 210 | 579 | 711 | 21 | — |

**2.4x prefill at long context for free.** `-ub` above 2048 keeps helping at 16k+ but regresses at
≤4k, so 2048 is the best all-round pick for agent traffic. Decode depends only on `-ncmoe` (bytes read
from RAM per token), not on `-ub`; the ±1 tok/s spread between same-`-ncmoe` rows is run noise.

Verified at `-ub 2048`: a 3-needle recall test over a 6.5k-token prompt (4 ubatches) returns all three
needles, so the upstream `ubatch >= 32` KV-corruption report does not reproduce here.

### `--no-mmap` is 1.7x faster and still must stay off — it OOM-killed the agent
With mmap the CPU experts sit in pageable page cache, so every per-ubatch copy to the GPU goes through
a bounce buffer (measured 7-16 GB/s on a link good for ~50). `--load-mode none` (= `--no-mmap`) puts
them in **pinned host memory** instead — `make_cpu_buft_list()` prefers the CUDA host buffer type, and
the loader warns about this at startup. Measured 2026-08-04 at `-ncmoe 36 -ub 2048`:

| load mode | 0.6k | 4.6k | 18k | decode | load time |
|---|---|---|---|---|---|
| `mmap` (current) | 310 | 715 | 808 | 23-24 | 16s |
| `none` | 620 | 1292 | 1344 | 23 | 36s |

**4x prefill over the original baseline — and unusable.** It pins ~78gb that the kernel cannot reclaim,
dropping MemAvailable from ~88gb to ~10gb. Four minutes after load the OOM killer fired and took the
`claude` agent process (`oom_score_adj:200` makes it the preferred victim); llama.cpp and the other
containers survived. The mmap safety net is exactly what stops this. Do not re-enable it at 96gb.

This is what a RAM upgrade would buy: at 128gb, `--load-mode none` fits with ~50gb spare and the 4x
prefill is free. It is the single highest-value hardware change for this box — worth far more than
anything left in the flags.

### Decode is near its ceiling — don't expect much
Decode reads ~1.9gb/token of expert weights from RAM (36 layers x 6 of 256 experts). At 23.5 tok/s
that is ~44 GB/s effective against a 96 GB/s theoretical / ~60-70 GB/s practical dual-channel ceiling.
RAM is already at its rated 6000 MT/s (2x48gb CMK96GX5M2B6000Z30), so there is no BIOS win. The only
real lever left is reading fewer bytes: fall back to `UD-IQ2_M` (90.9gb, ~12% fewer bytes) via
`QUANT`/`MODEL_FILE` in `download-model.sh` + `start.sh`, at a quality cost.

**Speculative decoding — including MTP — cannot help here. Do not spend time on it.** Decode is ~100%
expert-weight streaming: 36 layers x 6/256 x 2.23gb ≈ 1.9gb/token, which at 44 GB/s is 42.7ms of the
42.5ms/token budget (the GPU sits at sm=0% during decode). With 6-of-256 routing, adjacent tokens pick
near-disjoint expert sets, so verifying K drafts touches `256*(1-(1-6/256)^K)` experts per layer —
11.9 for K=2 vs 12 sequential, 23.1 for K=4 vs 24. **Batching saves 1-4% even at 100% acceptance**,
so any draft miss is a net loss. Measured on an echo-the-file prompt: off 23 tok/s, `ngram-mod` 21,
`ngram-map-k` 18.

MTP is obtainable — `convert_hf_to_gguf.py --mtp` emits the MTP head as a small standalone draft GGUF
(and `--dspark` a V4 DSpark draft), so it would not require reconverting 284B — but it changes nothing
about the arithmetic above. This quant ships no `nextn` tensors (no `deepseek4.nextn_layer_count`)
regardless. Speculation only pays once the experts are in VRAM, which 32gb cannot do.

mmap stays enabled (the default) so RAM overflow degrades to NVMe paging instead of OOM. Do not add
`--mlock` or `--no-mmap`.

### Models live in `~/models/gguf`, NOT the HF cache
vLLM bind-mounts `~/.cache/huggingface` and pulls by repo id, but `~/scripts/hf-cache-cleanup.sh`
builds its KEEP set by grepping `MODEL_ID=` out of `~/scripts/vllm.sh` and `rm -rf`s everything else
under `hub/`. A GGUF there would be treated as an orphan and deleted. Keeping it at `~/models/gguf`
(the path `~/scripts/llama.sh` already declares) avoids that, and llama.cpp wants a plain file path
anyway rather than the hub's blobs/snapshots symlink layout.

## Flag choices worth remembering
- **f16 KV cache** (no `--cache-type-k/-v`): upstream reports garbage output with quantized KV on this
  architecture. Deliberate deviation from `~/scripts/llama.sh`'s `q8_0` default.
- `--jinja`: required for V4's chat template / tool calling.
- `--restart no`: vLLM's container is `unless-stopped` and also binds 8000 — two auto-restarting
  containers would race for the port after a reboot.
- `--threads 12`: physical cores only; SMT siblings thrash cache.
- Health start period is generous — loading 104gb takes minutes.
- `-b/-ub 2048`: see the prefill tuning table above. Not a default — the default `-ub 512` costs 2.4x
  prefill at long context.
- `start.sh` always recreates the container: the flags are baked in at create time, so reusing a
  stale one would silently ignore edits to the constants.
- Upstream (PR #24162, merged 2026-06-29) had post-merge reports of KV corruption at `ubatch >= 32`
  *with expert offloading*. Did not reproduce at `-ub 2048` (see needle test above).

## Agent config sync
`start.sh` calls `~/scripts/update-agent-models.sh` (same as vllm.sh / colibri) to point pi + OpenClaw
at whatever :8000 serves. Two llama.cpp-specific details:
- `--alias` is what makes `/v1/models` report `DeepSeek-V4-Flash-0731`; without it the id is the raw
  container path of the GGUF, which is what would land in the agent configs.
- That script reads the context window from vLLM's `max_model_len`, **which llama.cpp does not report**.
  It would silently fall back to its own 32768 default, so `start.sh` passes `CONTEXT_WINDOW` explicitly.
  Change `CONTEXT_SIZE` in `start.sh` and both agents follow; verified end-to-end at 40960.

## Ports
8000 = this service / vLLM (mutually exclusive). 8080 = open-webui, do not use. owui already has
`http://host.docker.internal:8000/v1` registered, so it picks this up with no config change.

## Usage
```
./build-async.sh start && ./build-async.sh wait
./download-model.sh
./start.sh
./stop.sh
```

## CLAUDE.md Guidance
- broad architecture of this repo must be recorded in `CLAUDE.md`
- file must be kept in sync with the repo, apply edits to `CLAUDE.md` alongside changes to the repo
- use terse language

## Work Guidance
- scripts are argument-free and configured by the `# --- configuration ---` constants at the top
  - do NOT use env variables to tune our scripts — edit the constant, so there is nothing to memorize
- skip code comments if the code is self-explanatory
- code comments must be extremely terse
- less code is better code
- create and persist small scripts instead of running medium to long bash commands ad-hoc
  - e.g. create `start.sh`, `stop.sh`, `rebuild.sh`, ...
- do not install anything globally
  - example: if a 1-time dependency is required to download a model from hugging-face, just write a script to create and start a throw-away container to get it done
- prefer additive patches over editing
  - example: instead of cloning a repo and applying changes in place, prefer cloning a repo in a subfolder, and "override" files from the subfolder via Dockerfile instructions at build time
