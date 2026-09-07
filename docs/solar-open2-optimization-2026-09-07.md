# Solar Open2 prefill and decode, 2026-09-07

Campaign scope: two prefill rounds and two decode rounds on one DGX Spark.
Rejected candidates count toward those limits. The starting runtime is
`da153beffbd936c75400e58c70c85fa19ebc1249`; the model weights are unchanged.

## Fixed workload

- Solar Open2 250B `MXQ-v1`: all 11 shards, 95,533,532,160 bytes,
  from [`Baekpica/Solar-Open2-250B-Mixed-Quant-GGUF`](https://huggingface.co/Baekpica/Solar-Open2-250B-Mixed-Quant-GGUF).
  Every shard matched `MXQ-v1-SHA256SUMS` before loading.
- NVIDIA GB10, CUDA 13.3, native `sm_121a` build; one resident VMM/base
  owner with 16 GiB reserve, 80 aligned IQ2 and 421 aligned Q8 artifacts.
- K-FP8/V-FP4 GQA KV, 4,096-token prefill chunks, greedy decoding without
  speculation. No simultaneous production model or build during timing.
- Corpus: `speed-bench/promessi_sposi.txt`, SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- Cold single-shot `ds4-bench`: `--ctx-start N --ctx-max N
  --gen-tokens 64`, N = 8,192 and 65,536. Fresh process for each sample,
  interleaved off/on order A1 B1 B2 A2 A3 B3; median of three per variant.
- Every cold sample dumps all 196,608 frontier logits and all 64 greedy
  token IDs. Adoption requires byte equality to the previous accepted path.
- The graph uses a separate incremental workload: 2,048-token appends on
  one warm session from 2K through 64K, 128 greedy tokens at every frontier.
  Its prefill rate is not the throughput of a cold 64K request.

The older model-card HTTP measurements at 8,222 and 66,761 prompt tokens
used another runtime and request protocol. They are historical serving
evidence, not the baseline for the percentage gains in this campaign.

## Initial profile

Unprofiled opening observations were 1,101.10 / 777.38 tok/s cold prefill
at 8K / 64K, and 17.94 / 13.67 tok/s decode. Each round remeasures its own
baseline; the opening observations are not used as substitute A/B samples.

`nsys --trace=cuda,nvtx` separates the measured prefill from the two boot
warmup chunks and the 32-token decode window:

| GPU kernel group | 8K prefill time | 64K prefill time |
|---|---:|---:|
| Compressed GQA HMMA attention | 0.333 s | 28.696 s |
| Routed MMQ worklist | 1.220 s | 9.724 s |
| IQ2 gate/up pair | 1.105 s | 8.879 s |
| Dense Q8 D2R | 1.072 s | 8.624 s |
| KDA scan / factor / prep | 1.359 s | 10.823 s |
| Expert sum and all residual adds | 0.455 s | 3.659 s |

The 32-token decode profile attributes 1.023 s to aligned dense Q8,
0.534 s to grouped GQA split attention and 0.113 s to its combine pass at
64K. This makes long-context attention a separate target from dense weight
traffic and short-context recurrent-state work.

## Round record

### Prefill 1: finish the MoE block in one pass

Solar batch execution previously materialized the expert sum, added the
shared expert, then added that result to the residual. The new kernel
keeps the finite-value guard and ascending expert-slot accumulation, then
performs the same two separately rounded F32 additions. It removes two
full hidden-buffer passes. `DS4_SOLAR_MOE_RESIDUAL=0` restores the old path.
Decode and other model families retain their prior dispatch.

`tests/test_solar_gates` verifies byte equality against the three original
GPU operations at 1, 7, 257 and 4,096 rows, including an unaligned hidden
width, nonfinite routed values and cancellation. It also rejects null,
short and overlapping buffers before any write. Independent code review
found no blocking issue. All twelve full-model samples preserved every
frontier logit and generated token, including rebuilt-off versus the saved
baseline binary at both contexts.

| Cold prompt | Prefill before → after | Change | Decode before → after |
|---|---:|---:|---:|
| 8,192 tokens | 1,092.85 → 1,108.09 tok/s | +1.39% | 17.96 → 17.96 tok/s |
| 65,536 tokens | 772.18 → 780.02 tok/s | +1.02% | 13.63 → 13.62 tok/s |

In the 8K trace, the expert sum plus the two replaced additions took about
0.379 s; the fused pass takes 0.281 s across 96 layer executions. The
attention residual addition remains separate. The 0.01 tok/s decode
difference at 64K is below 0.1%; the decode execution path is unchanged.
Round 1 is adopted. Raw repetitions are in
[`solar-open2-2026-09-07-rounds.csv`](solar-open2-2026-09-07-rounds.csv).

The remaining prefill round and both decode rounds are pending.
