# Ling-3.1-flash-1019: follow-up optimization

Follow-up to [the merged campaign](ling31-optimization-2026-10-28.md),
from `f5baa61771f53a72a375feb5ffd7dae2794b2fdc` on
`perf/ling31-1019-rounds`. Counts here belong to this follow-up only.

## Artifact and protocol

The all-12-bit `Baekpica/Ling-3.1-flash-1019-VQ-GGUF` release, manifest
SHA256 `940c92ca2d347e47b929dceb7d25acc852f31a58c443e368a78d1484bd4265d9`.
One GB10, CUDA 13.4.92, 300–2200 MHz operator service; observed busy SM
clocks 2184–2190 MHz. No clock-policy changes. Performance CPU cores
5–9,15–19, one model per fresh process, cache dropped before each load,
`host_memory_guard.py` with reserve/trip 2 GiB and PSI limit 40%.

8K cold prefill after an 8K warmup, 128 forced-length greedy tokens on
`speed-bench/promessi_sposi.txt` (SHA256
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`).
Off/on pairs alternate in order off,on,on,off,off,on. Full frontier logits,
tokens, source diff and binary hashes accompany each local receipt.
These single-frontier rates are distinct from the published incremental sweep.

## P1: fuse SwiGLU with the down-projection preparation

The shared expert's Q4_K down matrix has K=1280. Its old capability check
required a multiple of 512 although the kernel consumes complete 256-value
Q4_K blocks. This kept it on MMQ. The admitted FP16 path plus a fused
SwiGLU/input pack removes the materialized F32 intermediate. No new scratch
allocation; fallback storage remains. `DS4_LING31_MLP_Q4P=0` restores both
MMQ for that shape and the separate SwiGLU pass.

| 8K warmed workload | Off samples | On samples | Median gain |
|---|---|---|---:|
| Prefill, tok/s | 1182.33, 1180.77, 1177.54 | 1194.70, 1180.85, 1189.16 | +0.71% |
| Greedy decode, tok/s | 18.97, 18.86, 19.02 | 18.30, 18.74, 18.92 | -1.21% |

The greedy continuations diverge at token 57. A separate fixed-state gate
loads the same 8K snapshot and evaluates the same 128 token IDs, warming
16 steps then restoring before measurement. Across six pairs, all start/end F32 logits are identical. Off samples are
18.8120, 18.8733, 18.3132, 18.8017, 18.8399, 17.8756 tok/s; on samples
18.8391, 18.2687, 18.2766, 18.8428, 18.1955, 18.5012. Means are
18.5859 → 18.4873 (-0.53%), medians 18.8069 → 18.3889 (-2.22%).
An exploratory bootstrap interval for the mean gain is -2.33% to +1.59%;
this is not proof of speed equivalence. Both controls have fast and slow
processes despite stable SM clocks and zero PSI. The cause remains unresolved.
P1 is retained for the consistent Prefill gain, qualified numerical change
and unchanged decode work; no Decode improvement or neutral-speed claim
is made. All samples remain in the [evidence](benchmarks/ling31-1019-followup-2026-10-28/p1.json).

Nsight Systems, PDL off: the 144 MLP activation/down pipelines total
221.572 → 173.645 ms per 8K prefill (-21.6%). Shared expert calls average
1.167 → 0.906 ms; dense calls 7.849 → 6.305 ms. This saves about 48 ms,
consistent with the unprofiled 0.71% gain. Whole-profile prefill
6942.75 → 6930.06 ms and decode 440.41 → 437.42 ms per eight tokens are
attribution samples, not throughput A/B medians. The 26 in-binary decode
kernels retain all 18,792 SASS instructions after constant-bank relocation
addresses are normalized; their launch counts also match.

The fused pack is byte-identical to the composed production SwiGLU and
FP16 pack on tail rows, K256/1280/10240, overflow scaling and finite-clamp
non-finite inputs. K256/768/1280 Q4_K GEMMs also pass the independent FP64
reference. Isolated K1280 MMQ+quantization 0.955 → 0.771 ms; the fused pack
alone 0.367 → 0.224 ms. No spill or additional allocation.

The admitted Q4_K arithmetic changes outputs. At 8K/32K, frontier KL is
0.001253/0.002954 against chunk-1024 noise 0.000524/0.001110. Because those
single positions exceed the reference, qualification uses the next 128 real
corpus tokens after each of 8K, 16K and 32K prompts (384 targets):

| Against off | Mean ΔNLL | Mean abs ΔNLL | Top-1 agreement |
|---|---:|---:|---:|
| Chunk 1024 reference | +0.011869 | 0.068125 | 96.35% |
| New MLP | +0.006558 | 0.063925 | 96.61% |

Per-prefix candidate mean ΔNLL: +0.017436/+0.006382/-0.004144; reference
+0.013680/+0.024756/-0.002828. The 32K candidate top-1 agreement is lower
(92.19% versus 94.53%); aggregate NLL and top-1 stay within the measured
reference. This qualifies this corpus continuation through 32K, not broad
assistant quality or longer contexts.

The earlier admission-only candidate was rejected: isolated GEMM gain did
not improve whole prefill (median -0.018%), while changing output values.
Only the complete pipeline above is considered for adoption.

## P2: cover the KDA local tail with one more warp

The local KDA preparation spends about 9% of prefill. Its A/Mq work has
544 tasks, so a 512-thread block leaves 480 threads waiting while 32 do a
second dot product. A 544-thread block covers every task once. Only the
original 512 threads own input and output rows; all threads join barriers.
Every output keeps its original reduction order. The 512-thread fallback
is `DS4_GLMVQ_KDA_LOCAL17=0`.

| 8K warmed workload | Off samples | On samples | Median gain |
|---|---|---|---:|
| Prefill, tok/s | 1185.07, 1181.27, 1187.96 | 1203.63, 1189.97, 1191.26 | +0.52% |
| Decode, tok/s | 18.93, 18.97, 18.92 | 18.63, 18.39, 18.92 | -1.58% |

All three prefill pairs improve; all full-vocabulary logits, 128 generated
tokens and cycle receipts are byte-identical. Isolated 4096-row/64-head
kernel medians are 4.848 → 4.638 ms (-4.33%), with all three pairs faster.
NCU reports 4.816 → 4.599 ms, 113 → 95 registers and no spills; shared
memory stays 98,816 B and no allocation is added. The minimum input/output
traffic is about 640 MB, a 2.72 ms floor at 235 GB/s. The post-dot barrier's
share of source stall samples falls from 7.85% to 2.98% (separate replays,
so their absolute sample counts are not comparable).

A paired 128-token NSYS capture with PDL off attributes 645.427 →
597.435 ms to the 126 local calls, saving 47.992 ms. Whole prefill is
6884.096 → 6856.488 ms. Decode is 6860.216 → 6989.452 ms; 124.725 ms of
the 129.236 ms difference is the unchanged grouped Q4 projection kernel.
Its later-layer calls average about 370 or 389 microseconds across process
runs, with matching launch geometry, allocation counts and instructions.
A fresh extra off control is fast. After restoring and rebuilding the clean
P1 source, which contains no 17-warp code, another whole-workload baseline
reproduces the slow pattern: the same later-group medians average 389.127
microseconds versus 388.621 in the candidate. This establishes that the
slow regime exists in the original path. Physical placement remains an
unproven explanation. P2 is retained for its consistent prefill gain,
unchanged values and resource footprint; no Decode gain or speed-equivalence
claim is made. The raw ranges and this limitation remain in the
[evidence](benchmarks/ling31-1019-followup-2026-10-28/p2.json).

The regression compares all local-record bytes and trailing canaries for
BF16 inputs at 1/63/64/65/200 rows and all F32/BF16 combinations at 65 rows.
The existing independent recurrence gate passes through 4096 rows: output
relative RMS ≤0.00138, state ≤0.000213. The four old local kernels retain
all normalized SASS instructions, as do the 26 decode kernels. Full-model
performance is qualified on this Ling artifact; other families sharing the
kernel have fixture coverage only.

Two preceding local probes were rejected: shared input staging was 12.6%
slower, and L1 lookahead did not improve timing. An earlier Q4p wait-suspend
hint also had inconsistent isolated gains and no useful counter improvement.
None is included in the implementation or the adopted-round count.

## P3: retain the residual's BF16 representation inside the trunk

The residual RMS pass is 4.3% of prefill. The residual already has a BF16
rounding boundary, but its F32 buffer makes each 4096 × 4096 pass move at
least 234.9 MB: about a 1.00 ms floor at 235 GB/s, close to the measured
1.05 ms. The new path stores those same BF16 values in the first half of
each original F32 row. The first residual packs in place after a block
barrier; the last plain residual expands back to F32. Intermediate passes
read and write half as many residual bytes. Norm and branch arithmetic,
allocation sizes and the external F32 layout stay unchanged. Decode,
verify and layer-slice calls retain the original path. The fallback is
`DS4_LING31_RESIDUAL_BF16=0`.

| 8K warmed workload | Off samples | On samples | Median gain |
|---|---|---|---:|
| Prefill, tok/s | 1174.98, 1191.08, 1188.83 | 1195.29, 1206.09, 1190.45 | +0.54% |
| Decode, tok/s | 18.38, 18.96, 17.99 | 17.99, 18.85, 18.97 | +2.56% |

All three Prefill pairs improve (+1.73%, +1.26%, +0.14%); every frontier,
generated token and cycle receipt is byte-identical. The apparent Decode
median gain has mixed pair signs and is not counted as an optimization.
Both controls span the previously observed process variation. The 26
Decode kernels retain their 18,792 normalized SASS instructions.

Three isolated 4096-row pairs reduce the median from 1.060 to 0.787 ms
(-25.7%), all faster with identical canonical residual and norm hashes.
NCU measures 1.052544 → 0.761024 ms and L2 requested read/write traffic
235,652,608 → 168,811,840 B (-28.4%). Registers rise 56 → 64, with no spill
or occupancy loss (four blocks per SM); shared memory is unchanged. No
additional allocation is needed. A fresh PDL-off workload profile reduces
the 286 residual RMS calls from 296.347 to 216.820 ms. Whole prefill is
6856.488 → 6870.994 ms in these individual captures: Q4p and KDA times
also vary, so the three unprofiled pairs establish the Prefill result.
The profile's 128-token Decode takes 6989.452 → 7174.564 ms; this is an
attribution sample, not a separate speed qualification.

The focused fixture checks pack, compact update and final expansion
against independent add/round/RMS kernels for 1/3/2048 × 4096 and 65 × 3073
shapes. Both norm types, raw F32 branches, row tails and trailing canaries
pass exactly. This covers the nondefault F32-norm path too. Full-model
proofs cover the default path at 8K; the final MTP and context sweep remain
pending. [Raw evidence](benchmarks/ling31-1019-followup-2026-10-28/p3.json).

## Reproduction and remaining scope

```sh
make tests/test_glm_vq_q4p tests/test_ling31_mlp tests/test_glm_vq_kda tests/test_ling31_residual
./tests/test_glm_vq_q4p
./tests/test_ling31_mlp
./tests/test_glm_vq_kda
./tests/test_ling31_residual
# Set DS4_LING31_MLP_Q4P=0 or 1 before each guarded fresh process:
./ds4-bench --cuda -m "$MODEL" --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --warmup-prefill 8192 --gen-tokens 128 \
  --csv "$RUN.csv" --dump-frontier-logits-dir "$RUN.proof"
```

The local campaign directory is `scratch/ling31-perf-20261028/`: its ledger,
run receipts, guard logs, full proofs, NCU/NSYS captures, fixed-state and
teacher-forcing Rust harnesses preserve the commands and identities.
Native/Rust inference builds passed. MTP, final incremental sweep and
publication are still pending; this change claims no broader serving gate.

## D1: reload KDA vectors instead of spilling the state

The MTP2 profile spends 309.950 ms of 6182.767 ms in KDA (5.0%). Holding
one state column across verify rows saves traffic, but retaining shared
vectors alongside that column costs 255 registers and a 160-byte spill
frame. Reloading the small shared q/k/decay vectors keeps the state in
168 registers without spills. Arithmetic and operation order stay the same.
Only Ling contexts opt in; `DS4_LING31_KDA_RELOAD=0` restores the previous
path. One-row decode and chunked prefill keep their existing kernels.

The additional MTP performance fixture uses ten existing held cases:
Korean/English/Chinese explanation, code, math, JSON, Korean summary and
free writing, tool call and tool follow-up. Each runs in an independent
4096-context session with two drafts, greedy sampling and a 128-token cap,
stopping normally. Their 719 generated tokens are divided by the sum of
their decode times; per-case rates are not averaged. A separate 8K corpus
case excludes EOS and generates 128 tokens. Each fresh process warms an
8K corpus session plus eight speculative calls before the measured cases.
This session lifecycle differs from `ds4-bench`; the published sweep keeps
its original protocol. The [measurement source](benchmarks/ling31-1019-followup-2026-10-28/reproduce/mtp_perf.rs)
and [raw receipts](benchmarks/ling31-1019-followup-2026-10-28/d1.json) record
this bounded protocol.

| MTP2 workload | Off samples, tok/s | On samples, tok/s | Median gain |
|---|---|---|---:|
| Ten assistant cases | 27.6460, 27.6612, 27.5833 | 27.6889, 27.7028, 27.6855 | +0.155% |
| 8K corpus | 21.2106, 21.2881, 21.2240 | 21.3377, 21.1768, 21.1575 | -0.222% |

All three assistant pairs improve (+0.155%, +0.151%, +0.370%); the lowest
on rate exceeds the highest off rate. Corpus pairs have mixed signs
(+0.599%, -0.523%, -0.313%): no corpus speed gain or equivalence is claimed.
All 33 case pairs have identical generated tokens and acceptance cycles;
66 full-vocabulary frontier/end F32 hashes match. The corpus continuation
repeats near its tail, so its speculative acceptance does not represent
assistant workloads. The ten short cases are supplementary performance
evidence, not broad quality or production qualification.

The 8K prefill controls are 1175.24/1188.16/1187.96 versus
1201.75/1197.06/1192.22 tok/s. Their chunked path is unchanged, so that
variation is not counted as another Prefill gain. All 26 ordinary-decode
kernels retain their 18,792 SASS instructions, including the renamed KDA
template. No ordinary Decode gain is claimed.

Isolated 64-head, three-row KDA with rollback snapshots runs at medians
76.676 → 71.039 microseconds (-7.35%); all three pairs improve and full
state/output/snapshot hashes match. Cache-flushed NCU replay is
74.752 → 72.832 microseconds. Local requested traffic falls from 246,784
32-byte sectors (7.90 MB) to zero, while executed warp instructions rise
1,281,536 → 1,540,608 (+20.2%) and shared-load wavefronts increase. The
tradeoff is more shared reads for fewer spills; no allocation, arithmetic,
or shared-storage increase. Native shared storage remains 4096 bytes.
The 16 MiB state read/write/snapshot payload alone has an approximately
8-microsecond L2 floor at 2.0–2.1 TB/s; residency and writeback prevent
assuming that all of it crosses DRAM during the measured kernel.

In the whole 8K/128-token profile with PDL off, KDA falls
309.950 → 292.334 ms, saving 17.616 ms. Whole Decode is
6182.767 → 6134.795 ms; 20.987 ms of that difference comes from unchanged
Q4m kernels. Prefill is 6821.926 → 6918.034 ms, with most variation in
unchanged q4p, local KDA and experts. These attribution captures do not
replace the fresh-process throughput pairs.

The focused fixture compares the held implementations against repeated
one-row recurrence, including every rollback state and conv tail, complete
output and canaries: rows 2/3/4 with snapshots, 64 heads at three rows,
and rows 3/5/63 without snapshots. All pass bytewise. The helper and
kernel results support retaining this small assistant MTP improvement;
the mixed single-corpus result remains disclosed.

## Rejected Decode candidates

One-row absorbed QK tiling reduced its isolated kernel from about 46.3 to
36.6 microseconds, but three full-model pairs gave a -3.12% Decode median
and added 24.7% instructions plus 5.8% requested global sectors. It is not
retained or counted. Q4m probes for a single stage, fewer warps, prepared
input and direct global weight reads also failed to establish a consistent
useful gain. The retained Q4m path is unchanged.
