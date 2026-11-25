# Inkling-2 Small VQ optimization on GB10, November 25

A second campaign after [November 14–15](inkling2-optimization-2026-11-14.md):
four prefill and four decode changes on top of `4242cf82`, and two
questions answered with measurements: whether DSpark's deficit against plain
decoding is the device's, and how fast the native context can be once its
K/V leaves device memory. The [release](https://huggingface.co/Baekpica/Inkling-2-Small-VQ-GGUF)
(`a6db8b8b`) does not change.

| Prefill / decode tok/s | `4242cf82` | This campaign | Gain |
|---|---|---|---|
| 2K–64K sweep | 1,239.9 / 29.72 | 1,428.9 / 31.53 | +15.2% / +6.1% |
| 2K–258K sweep | 917.1 / 25.40 | 1,140.4 / 28.29 | +24.3% / +11.4% |
| 258K frontier of that sweep | 617.3 / 20.66 | 846.2 / 24.63 | +37.1% / +19.2% |
| Cold 8K prompt | 1,377.1 / 31.28 | 1,657.1 / 32.76 | +20.3% / +4.7% |
| Cold 64K prompt | 1,238.4 / 27.77 | 1,574.9 / 30.59 | +27.2% / +10.2% |
| Cold 258K prompt | 866.0 / 20.41 | 1,190.9 / 23.74 | +37.5% / +16.3% |
| 258K with every K/V block on the SSD, prefill | 564.6 | 858.3 | +52.0% |
| 1,040,000 tokens, K/V past 262,144 on the SSD | 284.7 / 0.86 | 518.5 / 0.86 | +82.1% / 0 |

## Protocol

One GB10, 48 SMs, CUDA 13.5.37 (`sm_121a`), driver 620.70.06, kernel
7.0.0-1025-nvidia, boot `fc1f4107-bee6-4ba2-9ba5-13810bca6451`. The enabled,
successful `nvidia-clocks.service` keeps 300–2,200 MHz; busy samples read
2,184 MHz in the record sweeps (2,190 in under 2% of them) and 2,177–2,190
in the hour-long 1M runs. Floors: 240 GB/s streaming, 107.8
TFLOPS BF16, 215.6 FP8 and 431 NVFP4 at the clock cap
([GB10 limits](gb10-hardware.md)).

Speeds are fresh `ds4-bench` processes on `speed-bench/promessi_sposi.txt`,
one model at a time on cores 5–9 and 15–19, caches dropped after each
exit, under the host memory guard (reserve and trip 1 GiB, PSI full limit
40%). Available memory stayed at or above 5.68 GiB in the record sweeps,
4.59 GiB up to 258K and 3.65 GiB with the drafter mapped (2.97 under Nsight
Systems); PSI full (10 s average) peaked at 12.6% during loads and decayed
to 0 once a model was resident.

```sh
python3 tools/host_memory_guard.py --max-gib 128 --high-gib 128 \
  --reserve-gib 1 --trip-gib 1 --psi-full-limit 40 --timeout 3600 \
  --log "$RUN/memory.jsonl" -- taskset -c 5-9,15-19 ./ds4-bench \
  --cuda -m "$MODEL_DIR" --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 \
  --dump-frontier-logits-dir "$RUN/proof" --csv "$RUN/bench.csv"
```

**Fragmented free memory stalls the load.** The host had been up two days.
With every cache dropped, 119.2 GiB were free but only 110.0 GiB of them in
blocks of 2 MiB or more (`/proc/buddyinfo`, orders 9 and up), and the driver
takes device memory in 2 MiB pages: four loads of the ~108 GiB model stalled
in direct compaction near the end (PSI full 40%, guard exit 75). Pinning
device memory up to the stall point and freeing it, three times, left 115.5
GiB in such blocks, and every later load ran clean. Each run below checked
the block total first ([guard notes](host-memory-guard.md#fragmented-free-memory)).

Kernels are attributed with Nsight Systems inside `ds4.prefill` and
`ds4.decode` (decode with `DS4_GLMVQ_PDL=0`); NCU ran only on model-free
harnesses at production shapes, with `--clock-control none`. Rates come from
unprofiled runs.

Numerical gates follow the [output rule](../AGENTS.md#inference-performance-optimization-guidelines):
prefill changes against the chunk-size switch's frontier KL, decode changes
against `DS4_INKLING2_GAIN_DOT=0` on the 1,280 teacher-forced positions of
`tests/inkling2_quality`.

## Starting profile

`4242cf82`, cold prompts ([every class](#kernel-classes-before-and-after)):

- **Prefill.** Routed experts take 43% of a cold 8K prompt (38% at 64K, 26%
  at 258K), Q4_K projections 19%, sliding attention 11%, attention
  residuals 5%, and the q/r rows that also write the sliding bias table 5%.
  Full-layer attention grows with the keys: 4% at 8K, 17% at 64K, 42% at
  258K.
- **Decode.** 32.7 ms a token at 8K: 14.5 ms of Q4_K projections and head
  (3.16 GB at 218 GB/s), 11.5 ms of routed experts (2.30 GB at 200 GB/s)
  and 1.7 ms in 97 one-CTA attention residuals. At 64K the full layers'
  rows add 4.5 ms (0.65 at 8K).
- **A two-draft trial** takes 70 ms for 2.07 kept tokens of prose: 29.6
  tok/s against 31.1 plain.

## Rounds

| Round | Target | Change | Result (fresh processes) | Numerics | Switch |
|---|---|---|---|---|---|
| P4 | Full-layer prefill tiles: eight warps decoded, scored and summed in step (53 TFLOPS at 262K keys); 17% of a cold 64K prefill, 42% at 258K | Ping-pong tile (`ink2a_prefill_pp`, K2-Horizon's), then Q·K on NVFP4 tensor cores (`ink2a_prefill_q4`): the stored 4-bit keys are the `mxf4nvf4` B operand as they are, a query is three NVFP4 rows, only the values decode. Key segments carry the state for the SSD tier | 2,048 rows over 262K keys: 247.2 → 181.7 (ping-pong) → 149.8 ms, 88 TFLOPS. Span rates at 2,048-row chunks: 64K 1,171.9 → 1,274.1, 131K 970.3 → 1,122.9, 258K 723.6 → 908.2 tok/s | 3.9e-3 of the output's RMS (BF16 score roundings: 1.3e-3 to 1.8e-3). Frontier KL to the in-step kernel inside the chunk switch's (table below) | `DS4_INKLING2_ATTN_FP4=0` (BF16 ping-pong), `DS4_INKLING2_ATTN_PP=0` (in step) |
| P5 | Each chunk streams every routed expert once (2.3 GB per layer): 43% of a cold 8K prefill | 4,096-row chunks while every K/V block is resident; the session halves to 2,048 when the graph does not fit, the bank lane before it gives up a checkpoint slot or a bank, and the SSD tier starts there | Cold 64K 1,288.7 → 1,397.6 tok/s for 1.37 GiB (workspace 1.11 → 2.17 GiB, rings and histories 1.04 → 1.35); minimum available memory 7.9 → 6.5 GiB at 64K, 6.0 → 4.4 at 258K | the chunk size is the prefill noise reference | `DS4_INKLING2_PREFILL_CHUNK` |
| P6 | A sliding layer's relative bias, `r · proj[:, d]` over 16 values, was written as a table (201 MB per 2,048 rows) and read once. NCU: 60% of warp cycles waiting on those loads, issue slots 12% busy, 13 TFLOPS | The attention takes the factors: a warp computes its 16 queries' biases over a tile's 79 distances with ten BF16 MMAs into shared memory (rows of 87 entries: aligned pair loads, no bank conflict) | Isolated 2,048 rows 3.75 → 2.15 ms; the q/r kernel writes no table. Same build, cold: 8K 1,514–1,526 → 1,671–1,677 tok/s, 64K 1,447.1 → 1,576.4 | the tensor core's own summation order; the fixture does not move, frontier KL in the chunk switch's band | `DS4_INKLING2_BIAS_RANK=0` |
| P7 | The router GEMM ran cuBLAS' SIMT sgemm on BF16-rounded rows: 1.22 ms per 4,096 rows and layer | TF32 math: products of BF16 values are exact | 0.48 ms per 4,096 rows | summation order | `DS4_INKLING2_ROUTER_TF32=0` |
| D4 | A full layer's decode row walked its keys a lane per key: 137 GB/s at 256K | `ink2a_split`: a KV head's six query heads are the m16 rows of a BF16 MMA over 64-key tiles staged by cp.async; parallel merge and fold; vector partial stores | 256K 2,204.6 → 1,310.0 µs per layer (230 GB/s), 64K 565.0 → 342.7; decode at 64K 27.74 → 29.61 tok/s | BF16 probabilities (prefill's and verify's arithmetic): 7e-4 of the output's RMS | `DS4_INKLING2_ATTN_SPLIT=0` |
| D5 | The attention residual of a decode row ran in one CTA: ~190 KB at one SM's rate, 16–42 µs a call, 97 calls a token | Two launches of 24 CTAs per row: the states' Gram matrix and query dots, then the weighted slice; the output norm is a quadratic form of the Gram matrix | 8–14 µs a call (1.71 → 0.46 ms a token); 8K decode 31.3–31.9 → 33.0–33.2 tok/s | F32 summation order: 300 ppm of fixture outputs move one BF16 step | `DS4_INKLING2_ATTN_RES_SPLIT=0` |
| D6 | A two-draft trial launched 480 microsecond kernels to save, keep and roll back the convolution histories; its verify attention queued eleven waves of CTAs | One tensor of histories, one launch each to save, keep (per layer) and roll back; verify attention in one wave of six splits per KV head | trial 72.1 → 70.6 ms; 2 and 3 drafts and plain decoding emit the same 256 tokens | values unchanged | `DS4_INKLING2_VERIFY_WAVE=0` |
| D7 | Verify-width Q4_K projections (`glmvq_q4m`) staged 16 of a row's 18 blocks, then two for the same 16 warps: 187 GB/s on the 6,144-row projections, 196 on the head | Rows of 17 or 18 blocks take one stage and a warp per block | Isolated, three tokens: 6,144 × 4,608 85.3 → 67.3 µs (237 GB/s), the head 2.65 → 1.95 ms (265). 8K, two drafts, same build: draft 13.3 → 12.6 ms, verify 54.2–54.6 → 52.6–52.8 ms, 30.5–30.6 → 32.0–32.1 tok/s once warm | summation order across blocks; each token within one BF16 step of an F64 dot | `DS4_GLMVQ_Q4M_ROW=0` |

### Numerical gates

Frontier logits after a cold prefill, KL to the in-step kernel at 2,048-row
chunks:

| Frontier | 1,024-row chunks | ping-pong | NVFP4 scores | NVFP4 scores, 4,096 rows | rank bias (to the table) |
|---|---|---|---|---|---|
| 8K | 2.0e-4 | 4.8e-4 | 1.8e-4 | 2.0e-4 | 1.8e-4 |
| 16K | 3.2e-3 | 3.4e-3 | 4.6e-3 | 4.9e-3 | 7.3e-3 |
| 32K | 3.5e-3 | 3.5e-3 | 2.9e-3 | 3.9e-3 | 3.0e-3 |
| 64K | 1.9e-3 | 1.1e-3 | 1.1e-3 | 8.8e-4 | 1.1e-3 |
| 131K | 3.3e-3 | 1.3e-3 | 1.5e-3 | 2.5e-3 | |
| 258K | 4.8e-3 | 3.7e-3 | 4.3e-3 | 6.1e-3 | |

The top-1 token is the same at every frontier but 131K, where the 1,024-row
and ping-pong runs pick the runner-up of a 0.144 / 0.142 pair. 2,048 against
4,096 rows on the final kernels: 8.9e-5 at 8K, 1.4e-3 to 8.6e-3 beyond.

Teacher-forced NLL over 1,280 positions (1,024 after short prefixes, 128
each after 32K and 64K), against `4242cf82`:

| Build or switch | Mean NLL change | Position MAE | Top-1 correct (of 1,280) |
|---|---|---|---|
| `4242cf82` | 1.3056 | | 869 |
| `DS4_INKLING2_GAIN_DOT=0` (accepted switch) | −0.0007 | 0.027 | 865 |
| `496f315e`, every new path switched off | −0.0001 | 0.026 | 869 |
| `496f315e`, BF16 ping-pong scores | −0.0034 | 0.051 | 868 |
| `496f315e` | +0.0013 | 0.053 | 867 |

The standard error of a mean change is 0.003–0.004. Any rounding-level
change moves single positions by about 0.026 (the row with every switch off
differs only by the parallel merge's summation order), a stack of n of them
by about √n as much, and the mean does not move.

The serving checklist (`ds4 --serving-check`, the essay prompt of 6,148
tokens) passes all ten reuse, disk, snapshot and bank items with resident
K/V and with the tier forced (`DS4_INKLING2_KV_RESIDENT_CTX=1024`, 512-row
chunks).

It also found what P5 cost the bank lane. Rerun at the head, its two banks
kept one checkpoint slot of eight (3 in an earlier run with 0.34 GiB more
free) and the partial fork prefilled from zero; the documented two-bank
plan (`--max-seqs 2 -c 131072 --mem-floor-gb 2`) kept 6 slots and the
server refused it after the open (its fitted quote held one bank). The bank
lane now halves the chunk first: the check passes with 8 slots (1.98 GiB
spare) and that plan serves two banks with 8 (2.37 GiB; 2.72 at
`4242cf82`).

## Results

The [2K–64K record](../speed-bench/gb10_spark_inkling2_31efb8d0.meta.txt)
(2,048 new tokens and 128 greedy tokens per frontier, three fresh runs per
build, interleaved): `4242cf82` 1,239.9 / 29.72 → `496f315e` 1,428.9 / 31.53
tok/s, **+15.2% / +6.1%** (medians of run means; base runs
1,238.3–1,241.2 / 29.66–29.99, final 1,428.6–1,435.7 / 31.30–32.04).

| Frontier | 2K | 8K | 16K | 32K | 48K | 64K |
|---|---|---|---|---|---|---|
| `4242cf82` | 1,293.1 / 31.61 | 1,385.6 / 31.37 | 1,334.5 / 30.89 | 1,242.4 / 29.71 | 1,164.2 / 28.77 | 1,083.8 / 27.90 |
| `496f315e` | 1,342.5 / 32.70 | 1,551.3 / 32.65 | 1,506.7 / 32.23 | 1,437.6 / 31.49 | 1,371.1 / 30.94 | 1,299.0 / 30.24 |

A base run whose load a memory-quote probe of this campaign overlapped
(116 s to resident against ~50 s, its first sixteen frontiers 3–20% slow)
is excluded and replaced by a fourth.

The same steps to 258K (32 greedy tokens; one base and two final runs):

| Frontier | 8K | 64K | 131K | 196K | 258K | Mean of 126 |
|---|---|---|---|---|---|---|
| `4242cf82` | 1,391.9 / 31.29 | 1,084.0 / 27.95 | 863.2 / 25.08 | 719.7 / 22.50 | 617.3 / 20.66 | 917.1 / 25.40 |
| `496f315e` | 1,547.8 / 32.52 | 1,309.8 / 30.21 | 1,099.8 / 28.05 | 954.6 / 26.20 | 846.2 / 24.63 | 1,140.4 / 28.29 |

A frontier adds one 2,048-row chunk, so these sweeps do not see P5. Cold
prompts do (means of two runs at 8K and 64K, one at 258K):

| Cold prompt | 8K | 64K | 258K |
|---|---|---|---|
| `4242cf82` | 1,377.1 / 31.28 | 1,238.4 / 27.77 | 866.0 / 20.41 |
| `496f315e` | 1,657.1 / 32.76 | 1,574.9 / 30.59 | 1,190.9 / 23.74 |
| `496f315e`, 2,048-row chunks (prefill, from the sweeps) | 1,494.5 | 1,424.0 | 1,103.7 |

The last row is what a context past the resident threshold gets, the
native 1M plan included: its chunks stay at 2,048 rows, and the sweeps'
tokens over their time match a cold prompt (`4242cf82`: 1,370.6 / 1,232.4 /
865.5 against the measured 1,377.1 / 1,238.4 / 866.0).

### Kernel classes, before and after

Nsight Systems, kernel time inside the NVTX ranges, `4242cf82` → now
(decode with `DS4_GLMVQ_PDL=0`):

| Cold prefill, ms per 1,000 prompt tokens | 8K | 64K | 258K |
|---|---|---|---|
| Whole | 726 → 601 | 805 → 635 | 1,154 → 840 |
| Routed experts | 310 → 244 (43% → 41%) | 304 → 241 (38% → 38%) | 303 → 241 (26% → 29%) |
| Q4_K projections and head | 138 → 139 (19% → 23%) | 137 → 138 (17% → 22%) | 137 → 138 (12% → 16%) |
| Sliding attention | 76 → 41 (11% → 7%) | 79 → 43 (10% → 7%) | 79 → 43 (7% → 5%) |
| Full-layer attention | 32 → 26 (4% → 4%) | 136 → 90 (17% → 14%) | 488 → 299 (42% → 36%) |
| Attention residuals | 38 → 38 (5% → 6%) | 38 → 38 (5% → 6%) | 38 → 37 (3% → 4%) |
| Convolutions and histories | 22 → 22 (3% → 4%) | 22 → 22 (3% → 3%) | 22 → 22 (2% → 3%) |
| Expert sum | 19 → 19 (3% → 3%) | 19 → 19 (2% → 3%) | 19 → 19 (2% → 2%) |
| q/r rows, bias | 37 → 18 (5% → 3%) | 37 → 18 (5% → 3%) | 37 → 18 (3% → 2%) |
| Router and routing | 18 → 9 (2% → 1%) | 17 → 9 (2% → 1%) | 16 → 8 (1% → 1%) |
| K/V writes | 2 → 3 (0% → 0%) | 2 → 3 (0% → 0%) | 3 → 3 (0% → 0%) |
| Merges, conversions, copies | 9 → 9 (1% → 1%) | 8 → 8 (1% → 1%) | 8 → 8 (1% → 1%) |
| No kernel running | 24 → 33 (3% → 6%) | 5 → 6 (1% → 1%) | 4 → 3 (0% → 0%) |

| Decode, ms per token | 8K | 64K |
|---|---|---|
| Whole | 32.67 → 30.95 | 36.89 → 33.29 |
| Routed experts | 11.50 → 11.54 | 11.44 → 11.52 |
| Q4_K projections and head | 14.52 → 14.30 | 14.70 → 14.30 |
| Sliding attention | 0.94 → 0.94 | 0.94 → 0.94 |
| Full-layer attention | 0.65 → 0.38 | 4.50 → 2.57 |
| Attention residuals | 1.71 → 0.47 | 1.74 → 0.49 |
| Convolutions and histories | 0.41 → 0.41 | 0.43 → 0.44 |
| Expert sum | 0.08 → 0.08 | 0.08 → 0.08 |
| q/r rows, bias | 0.47 → 0.49 | 0.50 → 0.51 |
| Router and routing | 1.57 → 1.56 | 1.59 → 1.58 |
| K/V writes | 0.12 → 0.12 | 0.13 → 0.13 |
| Merges, conversions, copies | 0.19 → 0.17 | 0.24 → 0.17 |
| No kernel running | 0.51 → 0.49 | 0.59 → 0.56 |

Experts and Q4_K projections are now 64% of a cold 8K prefill and neither
kernel changed; full-layer attention is still the largest class at 258K.
Decode is 84% weight stream at 199 (experts) and 221 GB/s (Q4_K), against a
240 GB/s floor.

## Is DSpark's deficit the device's?

On chat requests it has none: through the server, two drafts with the rate
gate add 9.5% to the greedy mean of five requests and 4.2% to the sampled
one. Raw prose, which the bench decodes, is where it loses.

| Decode tok/s, up to 512 tokens per request | Code | Explanation | Korean | Arithmetic | Prose continuation | Mean |
|---|---|---|---|---|---|---|
| Plain, greedy | 32.91 | 33.03 | 33.05 | 33.10 | 32.35 | 32.89 |
| 2 drafts, gate | 38.80 | 34.22 | 36.90 | 39.24 | 30.91 | 36.02 |
| 2 drafts, no gate | 42.00 | 32.78 | 36.26 | 39.64 | 30.07 | 36.15 |
| Plain, sampled (T 0.7, top-p 0.95) | 32.73 | 32.73 | 32.63 | 32.90 | 32.02 | 32.60 |
| 2 drafts, gate | 37.72 | 31.64 | 31.21 | 39.69 | 29.65 | 33.98 |
| 2 drafts, no gate | 39.69 | 30.67 | 27.65 | 37.40 | 23.83 | 31.85 |

`ds4-server -c 131072 --mem-floor-gb 2`, one request at a time, rate from
the first to the last streamed token; the drafted and plain servers sample
different continuations.

On the bench's prose (8K frontier, 256 greedy tokens, gate off; rates after
the first call, which pages the drafter part in):

| | `4242cf82` tok/s | Now | Kept per trial | Draft ms | Verify ms |
|---|---|---|---|---|---|
| Plain | 31.14 | 32.75 | | | |
| 2 drafts | 29.61 | 32.03, 32.08 | 2.07 → 2.09 | 13.3 → 12.6 | 56.4 → 52.6, 52.8 |
| 3 drafts | 27.03 | 29.06 | 2.09 | 13.8 → 13.1 | 63.6 → 59.3 |
| 7 drafts | 24.64 | 25.36 | 2.41 → 2.37 | 16.0 → 14.8 | 82.0 → 76.9 |

Why a trial is expensive:

- **Bytes, the device's part.** A plain token reads 5.45 GB of weights
  (2.30 GB of routed experts, 3.16 GB of Q4_K projections and head). A
  two-draft trial reads 9.9 GB: its three verify rows route to 15.4
  distinct experts of 24 (1.93 times one row's), the projections and head
  once, and the drafter's 1.75 GB with the head a second time. At the same
  GB/s a trial costs 1.81 plain tokens: no kernel makes two drafts pay below
  1.81 kept tokens. Prose keeps 2.09 of 3; by their rates the code and
  arithmetic requests keep about 2.6–2.7.
- **Kernels, the engine's part.** A trial took 2.18 plain tokens' time at
  `4242cf82` (70.0 against 32.1 ms) and takes 2.14 now (65.3 against 30.5):
  its kernels move bytes at 85% of the one-row rate, so the break-even sits
  at 2.14 kept tokens and prose misses it by 2%. Per trial, profiled: Q4_K
  projections and both head reads 18.8 ms at 196 GB/s (21.3 before D7; one
  row: 221), verify experts 26.1 ms at 170 GB/s (one row: 199), the
  drafter's graph 8.4 ms, verify attention 2.5 ms, about 5 ms with no kernel
  running.

At the one-row rate the same trial would take about 55 ms, 38 tok/s on this
prose. The verify experts are the largest remaining piece. More drafts do
not help prose: kept tokens rise from 2.09 to 2.37 while the verify grows
from 53 to 77 ms.

Where the bytes are shared, drafts pay regardless: past the resident
threshold a trial reads the K/V file once for all its rows (1.94 against
0.86 tok/s at 1M, [family page](inkling2-vq.md#dspark-drafting)).

## K/V on the SSD

Past the resident threshold (262,144 positions) the full layers' blocks
live in a file, and those layers attend densely: every forward reads the
whole file region.

**Decode there is the drive's.** A token reads (context − threshold) ×
9,216 bytes at ~6 GB/s. With every block on the SSD
(`DS4_INKLING2_KV_RESIDENT_CTX=0`) decode runs 19.2 / 9.9 / 5.0 / 2.6 tok/s
at 32K / 64K / 131K / 258K, as at `4242cf82` (18.8 / 9.8 / 5.0 / 2.6). No
block can be skipped: keys beyond the newest 1,024 carry 71% of a full
layer's attention at 32K and 78–87% from 64K to 258K, and dropping at most
1% of a head's mass still needs 80–92% of those keys' 512-key spans per
head, 97–99% per KV head (the unit a stored block serves).

**Prefill there gained what the kernels gained.** Same setting, prefill
tok/s over the span since the previous frontier:

| Frontier | 32K | 64K | 131K | 258K | File read, time in I/O |
|---|---|---|---|---|---|
| `4242cf82`, 2,048-row chunks | 1,267.9 | 1,037.9 | 809.0 | 564.6 | 207.3 GiB, 38.3 s |
| now, 2,048 rows (the tier's default) | 1,473.3 | 1,315.0 | 1,114.9 | 858.3 | 207.3 GiB, 38.5 s |
| now, `DS4_INKLING2_PREFILL_CHUNK=4096` | 1,613.3 | 1,425.3 | 1,198.2 | 912.8 | 137.5 GiB, 25.4 s |

A chunk reads the file once, so 4,096-row chunks also cut the reads by a
third: 6–10% for 1.5 GiB. The tier defaults to 2,048 rows because a 1M
session at 2,048 rows already dips to 3.7 GiB available; the variable takes
the rest where memory allows.

**The native context.** 1,040,000 tokens at the default threshold prefill
at 518.5 tok/s on average (2,006 s) against 284.7 at `4242cf82` (3,653 s;
the initial release's engine: 268.2), and decode at 0.86 on both: 7.17 GB
of file blocks per token. Each run read 1,372.6 GiB in 250–252 s of I/O, an
eighth of the new prefill: compute bounds it, and at the tile's 88 TFLOPS
the full layers' attention over up to a million keys is about 60% of that.
At least 5.82 GiB stayed available; busy clocks read 2,177–2,190 MHz.

**What changes decode is which bytes leave the device.** The K/V is read
whole by every token; the experts are not. 258K tokens of one session at
2,048-row chunks, replayed through an LRU over (layer, expert):

| Session | Experts a chunk routes to, per layer | Routes on the coldest 6.9 GB (6.25%) | LRU with 6.9 GB paged out | LRU with 13.8 GB paged out |
|---|---|---|---|---|
| Italian prose | 238 of 384 | 0.007% | 0.03 GB read per chunk, 0.07 routes missed per token | 0.08 GB, 0.07 |
| English docs, Rust, C and Italian prose | 302 of 384 | 0.148% | 0.28 GB, 0.17 | 1.71 GB, 0.85 |

A token has 384 routes (8 experts in 48 layers). With 27.6 GB paged out
the mixed session thrashes (a chunk's 302 experts per layer no longer
fit); the prose session still misses 0.36 routes per token.

The native context needs 7.25 GB of K/V past the threshold, 6.6% of the
experts. Paging that many experts out instead and keeping every K/V block
resident, a decode token would read one 6 MB expert every 6 to 14 tokens
where it now reads 7.17 GB of K/V, and a prefill chunk 0.03–0.3 GB of
experts where it now reads the K/V file.

Resident decode is measured to 524K (one 524,288-token prompt, 2,048-row
chunks):

| K/V of the 524,288 positions | Prefill tok/s | Decode tok/s | First token | Available memory, minimum |
|---|---|---|---|---|
| All resident (`DS4_INKLING2_KV_RESIDENT_CTX=524288`) | 839.0 | 19.59 | 0.05 s | 3.88 GiB |
| Default: 262,144 resident, 262,144 on the SSD | 801.0 | 2.54 | 0.40 s | 5.81 GiB |

The resident slope (30.6 ms a token at 8K, 40.6 at 258K, 51.0 at 524K)
extrapolates to about 72 ms at 1M: 14 tok/s against 0.86.

That is a slot pool with one indirection per routed expert, a miss path in
decode and a new memory plan (`--ssd-streaming` does this for GLM-5.3 and
Gemma-4 only): the next change for the native context, not part of this
one. Until then the resident threshold is the lever
(`DS4_INKLING2_KV_RESIDENT_CTX=524288` for 2.36 GB, without drafter or
images), and drafts share one file read per trial.

## Measured and not adopted

- **Two NVFP4 query components.** 143.1 ms against 149.8 at 262K, 4.6e-2
  of the output's RMS against 3.9e-3. Its frontier KL also sat in the band,
  which says the gate cannot tell them apart, not that they are equal.
- **Unrounded far scores on the BF16 ping-pong tile.** 168.2 ms against
  181.7; the NVFP4 tile keeps F32 far scores anyway and is faster.
- **Other CTA counts for the residual split.** 16, 24 and 48 CTAs per row
  ran within 10%: two launches dominate.
- **Other stage shapes for 18-block Q4_K rows.** Two stages of nine blocks
  with two CTAs per SM reached 211 GB/s, three of six 190, 32-row tiles
  201–214: whole rows in one stage won. 24-block rows (the output
  projection, 207 GB/s) gained from none.

## Next levers

What the profiles leave, by recoverable time:

- **Page experts, keep K/V** ([above](#kv-on-the-ssd)): the one change that
  moves decode past the resident threshold by an order of magnitude.
- **Verify experts.** 26 ms of a 65 ms two-draft trial at 170 GB/s, against
  199 for one row. MiniMax's row-group order (a CTA group per verify row
  staging its own input, one task order across groups) is the candidate;
  a union kernel is not, with 15.4 distinct experts in 24 routes.
- **Full-layer attention past 131K.** 36% of a cold 258K prefill at 85–88
  TFLOPS. Scores take three NVFP4 MMAs per key tile (431 TFLOPS peak), the
  values one BF16 MMA (108): the pair's ceiling is 123. Values as an NVFP4
  operand under coded probabilities would raise it to 144 and drop the
  value decode; unprobed.
- **Routed experts and Q4_K projections in prefill.** 41% and 23% of a cold
  8K prompt; neither kernel was profiled in this campaign.
- **The q/r kernel** launches a CTA per row and head: 196,608 per 4,096
  rows, 1.5 ms a layer (3% of prefill).
- **The full layers' near tile** still reads its relative bias from a
  table; the sliding layers' factor form applies.
