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
Clean release integration is recorded separately when complete.

### Numerical contract

Prefill frontier logits are exact. The 128-token Italian continuation
changes 94 token positions after index 31; both continuations remain
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
