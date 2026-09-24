# MiMo runtime optimization, 2026-09-24

Fresh processes on DGX Spark / GB10. The four-shard
`MiMo-V2.6-Flash-RL-MQ-IQ2-XXS-XS-Q8` artifact and prompt hashes are in
[`swa/workload.json`](swa/workload.json). One VMM weight owner stays resident;
workers run sequentially with a warmup before each measured process. The
user clock range stays 300–2200 MHz; busy samples report 2190 MHz.

## Round 1: share the SWA decode window

The previous walk loaded each KV head once per query head. The shared tile
loads it once for eight query warps, using vector copies. It is now the
default for one-row, window-128, eight-KV-head attention.
`DS4_MIMO2_SWA_DECODE=0` restores the walk;
`DS4_MIMO2_SWA_VEC=0` keeps the shared tile with scalar copies.

8,192 prompt tokens / 128 greedy output tokens, embedded MTP disabled
(`--mtp-draft 1`), no DFlash:

| Metric | Walk | Shared vector tile | Change |
| --- | ---: | ---: | ---: |
| Median prefill, tok/s | 1156.33 | 1154.55 | −0.15% |
| Median decode, tok/s | 21.30 | 23.08 | +8.36% |
| SWA kernels, profiled seconds | 0.599403 | 0.120262 | −79.94% |

Three samples per side; [`compare.json`](swa/compare.json) records the
sample envelope. Profiles are separate runs and are not used as application
throughput samples. The baseline and candidate use the same frozen binary;
[`receipt.json`](swa/receipt.json) records its hash and dirty source base.
Candidate and clean-build measurements are kept separate.

### Numerical contract

Prefill frontier logits are exact. The 128-token Italian continuation
changes 94 token positions starting at index 31; both continuations remain
coherent. This is **not** exact token parity. Three short math, Python and
Korean prompts produce identical correct answers. These are focused checks,
not a statistical quality benchmark.

The attention test compares with independent FP64 softmax using random
nonzero sinks, a 259-row ring, and positions through 262144. Maximum walk
versus tile difference is 1.79e-7; both oracle errors stay below 4.73e-7.
One captured graph is replayed with changed device positions. This bounds
the observed FP32 rounding and checks ring indexing. The attention consumer
does not write KV. The MiMo engine remains eager; no new model-level CUDA
graph capture path is introduced.

Focused tests (use the CUDA architecture appropriate to the test host):

```sh
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121a tests/mimo2_swa_quality.cu -o /tmp/mimo2-swa-quality
/tmp/mimo2-swa-quality
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121a tests/mimo2_decode_rounds.cu -o /tmp/mimo2-decode-rounds
/tmp/mimo2-decode-rounds
cargo test -p ds4-perf
```

## Round 2: parallel decode routing

The old router computes sigmoid in parallel, then scans all 256 experts
eight times on one thread. That serial selection accounts for 5.9% of
decode kernel time after round 1. A warp now reduces the comparisons for
widths 1–8. Ties choose the lowest expert ID; sigmoid, selected-weight
summation and the denominator floor preserve the previous arithmetic.
Wider prefill keeps the previous dispatcher. `DS4_MIMO2_ROUTER_WARP=0`
restores serial selection.

The same 8K/128 protocol, with round 1 enabled on both sides:

| Metric | Serial selection | Warp selection | Change |
| --- | ---: | ---: | ---: |
| Median prefill, tok/s | 1154.40 | 1156.81 | +0.21% |
| Median decode, tok/s | 23.00 | 24.42 | +6.17% |
| Router kernels, profiled seconds | 0.314609 | 0.021011 | −93.32% |

All three 128-token streams match. The focused test checks random scores,
ties, denominator-floor inputs and nonfinite rejection at widths
1/2/8/32/129. IDs and finite weights match exactly; nonfinite inputs keep
the previous rejection values. Typical decode-width kernel time falls
from 52 to 4.3 µs in that test. Full-model timing includes dispatch overhead.

```sh
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121a tests/mimo2_router_warp.cu -o /tmp/mimo2-router
/tmp/mimo2-router
```

## Round 3: keep DFlash attention on the GPU

The old external-drafter path copied Q/K/V to the host, ran RMSNorm, RoPE
and three-pass attention there, then copied attention output back. The
baseline profile spends 6.47 of 14.82 decode seconds outside GPU work
(43.6%, including other host work and synchronization). The device path
normalizes and rotates Q/K in place and shares 32-key K/V tiles across
eight query warps. `DS4_MIMO2_DFLASH_CPU=1` restores the original C loops.

256 prompt / 64 output tokens, external DFlash, draft width 8. Both sides
use the SWA vector tile and serial router in the same frozen binary:

| Metric | Host attention | Device attention | Change |
| --- | ---: | ---: | ---: |
| Median prefill, tok/s | 517.88 | 518.86 | +0.19% |
| Median DFlash decode, tok/s | 4.41 | 6.97 | +58.05% |
| Profiled decode range, seconds | 14.819759 | 9.397574 | −36.59% |
| GPU coverage of decode range | 56.4% | 90.2% | +33.8 points |

Device attention takes 0.018145 seconds, RMSNorm 0.027456 and RoPE 0.074160
in the separate profile. The host baseline has no corresponding attention
CUDA kernel; the application measurement includes the removed host work.
All three 64-token streams and prefill frontier logits match exactly.
The [receipt](dflash/receipt.json) records hashes, controls and scope.

One additional 2K/64 pair exercises the full 1024-row drafter context:
2.25 → 5.15 tok/s, identical tokens. The clean release build with SWA and
router defaults also passes that CPU/GPU pair. The primitive test covers
widths 1/2/4/8, short and full windows, nonzero sinks and positions through
262144. Maximum host/device error is 2.53e-5; the host fallback is exact.
Target verification and accepted-prefix commit logic are unchanged.

**This is a DFlash-mode improvement, not a reason to enable DFlash.** A
same-binary, same-owner 256/64 plain-decode countercheck after fresh warmup
reaches 23.91 tok/s (one sample, versus three DFlash samples). DFlash is
still substantially slower on this workload. Further rounds prioritize
prefill and ordinary decode.

```sh
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121a tests/mimo2_dflash_attn.cu -o /tmp/mimo2-dflash
/tmp/mimo2-dflash
```

## Rejected candidate

A compact IQ2 gate/up activation layout removed eightfold repeated input
quantization and passed byte-exact gate/up tests. Three fresh 8K/128 samples
changed median prefill by +0.48% and decode by −0.84%. The ds4-perf verdict
was [`Pass`, not `Improved`](rejected-compact-q8.json); the gain did not
justify adoption. The compact
layout is absent from this branch.

## Reproduction

Build CUDA for the target device, then `make ds4-bench-perf`. Use the
[weight-owner protocol](../../../misc/proof-harness/README.md) and one
worker at a time. DFlash needs **both base and drafter** in the owner;
base-only import does not provide resident MTP tensors. Record the model,
prompt, binary and manifest hashes in the ds4-perf workload file.

```sh
export DS4_CUDA_WEIGHT_IPC_MANIFEST=/path/to/owner.ipc
./ds4-perf scout --out /path/to/result --collector nsys --proof \
  --repeats 3 --cache-policy warmup-then-fresh \
  --workload /path/to/workload.json -- \
  ./ds4-bench-perf --cuda -m /path/to/model-00001-of-00004.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --gen-tokens 128 --mtp-draft 1
```

Pass each diagnostic control to scout as `--env KEY=VALUE` before `--`.
For DFlash use `--mtp /path/to/DFlash-Q8_0.gguf --mtp-draft 8`, context 256,
and 64 generated tokens. Run a separate 2K case to exercise its full
1024-row context. Wrap the scout in `tools/host_memory_guard.py` with limits
fitted to the resident owner and host reserve. Profiles and unprofiled
throughput are separate processes. Three samples bound this session's
variation; they do not establish a broad workload average.

## Scope and validation

CUDA GB10 only. The model remains eager. No new captured-decode, bank,
media, disk-KV or long-context serving qualification is claimed here.
The 262144-position DFlash primitive test is a position/window check, not
a full-model 262K throughput run. SWA and router do not change prefill
selection; DFlash attention runs only when the external drafter is loaded.

The clean SWA/router source passed formatting, Clippy, all eight host
parity targets, serialized workspace tests and the all-target workspace
check. After adding DFlash, the CUDA production CLI/NVTX benchmark build,
formatting and `cargo test -p ds4-perf` passed again. Standalone CUDA
parity tests cover all three paths. Metal was not tested.

A base-only-owner DFlash attempt lacked imported MTP ranges. A separate
local-MTP pilot had no funded MTP residency and failed with a CUDA memory
access error. Both were excluded. Retained DFlash runs use a replacement
owner exporting base and MTP, verified at 88.94 and 1.46 GiB of device
residency. The old owner exited before the replacement started.
