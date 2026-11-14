# Inkling-2 Small VQ optimization on GB10, November 14–15

This campaign starts at `53043039` with the
[published artifact](https://huggingface.co/Baekpica/Inkling-2-Small-VQ-GGUF)
at revision `b79f2fbd27e981691a7d0c930e0c42b9ece08ec4`.
The local `release.json` SHA256 is
`ff495b121f21e92870db8909f54880da61e02b56a37a9f9af8e60d17691e13ef`;
the loader verifies the model's tensor spans. No weights change.

## Protocol

One GB10, 48 SMs, CUDA 13.4.92 (`sm_121a`), driver user/kernel
610.57.04/615.71.10. The enabled, successful `nvidia-clocks.service` keeps
300–2,200 MHz; busy samples were 2,184 MHz. Boot identity:
`94cfabbf-7a48-4692-b38a-78e43e9be86d`. No profiler changes clocks.
Floors use the [GB10 measurements](gb10-hardware.md): 240 GB/s streaming,
107.8 TFLOPS F16/BF16 and 215.6 TFLOPS FP8 at the clock cap.

Round comparisons use fresh Rust `ds4-bench` processes, one resident model,
2,048-row chunks, default NVFP4 full-layer KV, plain greedy decode, cold
prefill, and `speed-bench/promessi_sposi.txt`. CPU affinity is
`5-9,15-19`. After the previous process exits, caches are cleared. The host
memory guard reserves/trips at 2 GiB, records PSI, and enforces a 1,800 s
deadline. Requested max/high are 128 GiB; admission sets effective limits
from available memory (116.4/115.4 GiB in these runs).

```sh
make -j10 ds4-bench ds4-bench-perf CUDA_ARCH=sm_121
python3 tools/host_memory_guard.py --max-gib 128 --high-gib 128 \
  --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 1800 \
  --log "$RUN/memory.jsonl" -- taskset -c 5-9,15-19 ./ds4-bench \
  --cuda -m "$MODEL_DIR" --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --gen-tokens 64 \
  --dump-frontier-logits-dir "$RUN/proof" --csv "$RUN/bench.csv"
```

Nsight Systems attributes kernels within `ds4.prefill` / `ds4.decode`;
profiled throughput is excluded from A/B rates. Decode rounds disable PDL
only for attribution (`DS4_GLMVQ_PDL=0`): dependent-launch timestamps can
include predecessor waits. A/B throughput keeps default PDL enabled. Earlier
P1–P3 Q4/expert decode timings with PDL are not exclusive compute costs.
NCU profiles isolated
production shapes with `--clock-control none`, never the full model.
Raw traces, counters, full logits, generated IDs, build logs, source diffs,
memory/clocks and process receipts stay in ignored `scratch/inkling2-perf/`.
Cold single-frontier rates are a different protocol from the published
2K–64K suffix sweep. These rounds do not qualify 1M, SSD restore, serving,
DSpark speculation or vision performance.

## P1: keep attention-residual topology in registers (`b3fd2fa6`)

The original 8K trace spends 1,374.457 ms (18.85% of prefill) and
544.178 ms (20.87% of decode) in `ink2_attn_res_kernel`. Runtime stack/query
indices put 592 bytes per thread on the local stack. At 2,048 rows, eight
stacked blocks and one query, NCU reports 2.995 GB of local read/write
requests (79.54% of L1 sectors) and 88.8% long-scoreboard stalls.

Specializing the model's block and query counts removes that stack. The
arithmetic expressions, reduction order and BF16 boundaries stay the same
(playbook class 5.1). The uncommon two-query case retains the runtime path;
`DS4_INKLING2_ATTN_RES_SPEC=0` selects it everywhere. No tensor allocation
or tensor arithmetic is added. Late-layer variants use 80/128 registers
for one/three queries, with no stack or spills.

| Isolated shape (depth 8) | Original median | Specialized median |
|---|---:|---:|
| 2,048 rows, one query | 4.388 ms | 1.077 ms |
| 2,048 rows, three queries | 9.982 ms | 1.450 ms |
| One row, one query | 94.016 µs | 10.923 µs |
| One row, three queries | 156.088 µs | 21.306 µs |

Three fresh isolated pairs, 20 event-timed iterations each, synthetic BF16
states in the production layout. The one-query case has 226.5 MB compulsory
tensor traffic: an optimistic 0.944 ms floor, or 88% of the streaming bound.
Both state passes reaching DRAM would instead cost 1.73 ms; rereads can hit
L2. NCU local read/write requests fall to zero for both query counts.

Full-model comparisons use the archived **original** executable, since
unrolling also changes compilation of the runtime fallback. ABBAAB order:

| Fresh pair | Prefill original → P1 (tok/s) | Decode original → P1 (tok/s) |
|---|---:|---:|
| 8K, 1 | 1120.64 → 1310.37 | 24.70 → 29.54 |
| 8K, 2 | 1121.10 → 1322.47 | 24.69 → 29.76 |
| 8K, 3 | 1126.32 → 1327.52 | 24.46 → 29.66 |
| Mean | 1122.69 → 1320.12 (+17.59%) | 24.617 → 29.653 (+20.46%) |
| 32K, one confirmation pair | 1059.80 → 1241.49 | 23.01 → 27.80 |

All eight runs exit 0; all 200,058 logits are finite and equal between
variants, as are all 64 generated IDs. Minimum available memory is
7.51 GiB, maximum PSI full avg10 is zero. The residual's measured time
falls to 306.083 ms / 108.477 ms in the retained 8K trace (4.95% / 4.98%).
The 162-case `test-inkling2-res` fixture covers row widths 1, 2, 4, 5, 17,
2,048; stack depths 0–8; null, one and three queries; and a nontrivial
block stride. It agrees with the runtime-indexed path. This is one adopted
**Prefill** round; the accompanying decode gain is not a separate round.

Production executable SHA256, original → P1:

```text
bcc68a0162ad24aa9fc517000d2857da3ac041a9f627badf36f455f5ac3efe0d
520a80da8aab9840c2fa0b775a4b9cb906bc2744b8bebf3b20a3389f083fbe9e
```

## P2: read sliding-attention biases ahead of their use

On the retained P1 trace, sliding prefill attention takes 843.569 ms
(13.64%, 160 calls). A late 2,048-row chunk has about 51.5 GFLOP of QK/PV
work, a 0.478 ms BF16 tensor-core floor, but its kernel takes 5.27 ms.
An isolated capture with the same 2,048 rows, position 6,144, 1,024-key
window and 3,072-row ring takes 5.306 ms. Values are synthetic BF16;
the production ring has 3,071 rows, so wrap addresses differ slightly.
The work shape and execution path match; full-model A/B uses the actual ring.

NCU reports 78.8% long-scoreboard stalls and only 8.55% issue-slot use.
SASS line mapping attributes 86.35% of long-scoreboard samples to the
relative-bias conversion immediately consuming a scalar global load.
The new path issues the raw BF16 loads before QK. Volatile PTX loads keep
the compiler from sinking them back to score use. It applies only to
BF16 sliding prefill; `DS4_INKLING2_BIAS_PREFETCH=0` restores the old path.
The score arithmetic, masks and reduction order stay unchanged (class 5.1).

Three isolated pairs: 5.368 / 5.441 / 5.323 ms → 3.861 / 3.861 / 3.972 ms.
NCU: 5.306 → 3.804 ms, 221 → 219 registers, no stack/spills, the same
69.63 KB shared memory and no extra tensor allocation. Global-load sectors
are unchanged; executed instructions increase 0.11%. Ordinary source-level
hoisting reaches only about 4.52 ms. The retained path is still latency
limited, at about 13% of the optimistic tensor-core bound.

| Fresh pair | Prefill P1 → P2 (tok/s) | Decode P1 → P2 (tok/s) |
|---|---:|---:|
| 8K, 1 | 1327.36 → 1382.64 | 29.53 → 29.64 |
| 8K, 2 | 1321.65 → 1383.94 | 29.23 → 29.57 |
| 8K, 3 | 1317.04 → 1384.28 | 29.82 → 29.21 |
| Mean | 1322.02 → 1383.62 (+4.66%) | 29.527 → 29.473 (−0.18%) |
| 32K, one confirmation pair | 1240.30 → 1285.26 (+3.62%) | 28.14 → 28.03 |

The decode ranges overlap (29.23–29.82 / 29.21–29.64); no decode path
changes and no material regression is established by these samples.
All eight runs exit 0 with equal, finite 200,058-logit vectors and equal
64-token continuations. Busy clocks are 2,177–2,184 MHz, mostly 2,184;
minimum available memory is 7.34 GiB and PSI remains zero.
In the new trace, sliding attention takes 618.184 ms (10.45% of prefill).

`test-inkling2-attn` passes 12 cases against the original kernel: widths
9, 63, 64, 65, 129 and 2,048 at position zero and after repeated ring wrap,
with a non-power-of-two capacity. These cover causal masking and query tails.
Production/NVTX builds and independent read-only review pass. This is
**Prefill round 2**. P2 production executable SHA256:

```text
134eca6dc8be145e3e92d2f2e8934dfe1cb1ef6ca1b8bc6cd53aa597c56e802d
```

## P3: skip per-score masks on old full-attention tiles

After P2, resident NVFP4 full attention takes 305.996 ms (5.17% of 8K
prefill). Its last 2,048-row chunk takes 14.128 ms. The matching isolated
shape (position 6,144, 8,192-key capacity) takes 13.56 ms under NCU.
Its useful QK/PV work is 360.8 GFLOP: a 3.35 ms BF16 tensor-core floor.
Synthetic E2M1 bytes and E4M3 scales use the production page layout.

Source-mapped counters attribute 289.70 million instructions (21.9% of
all instructions) to per-score validity and relative-bias handling. Most
full-layer keys are already outside the 1,024-position bias span and
strictly before every query in the tile. A uniform predicate now selects
constant-zero bias and known-valid scores for those complete tiles.
Score formulas and reduction order stay unchanged (class 5.1); causal,
bias-bearing and incomplete tiles retain the original path. The dispatch
is limited to resident NVFP4 prefill. Decode and segmented SSD attention
keep their existing kernels. `DS4_INKLING2_FAR_TILES=0` disables it.

Three isolated pairs: 13.962 / 13.867 / 13.895 ms → 10.927 / 10.881 /
10.870 ms. One 32K-frontier pair: 48.643 → 33.645 ms. NCU reports
13.56 → 10.55 ms, 1,323.18 → 965.14 million instructions, 223 → 209
registers, no spills, and unchanged 53.25 KB shared memory. No tensor
allocation or arithmetic is added. The candidate reaches about 31% of the
optimistic BF16 bound; conversion, score processing and dependencies
still cost time.

| Fresh pair | Prefill P2 → P3 (tok/s) | Decode P2 → P3 (tok/s) |
|---|---:|---:|
| 8K, 1 | 1380.21 → 1392.82 | 29.61 → 29.87 |
| 8K, 2 | 1385.64 → 1392.32 | 29.79 → 29.37 |
| 8K, 3 | 1375.09 → 1396.83 | 29.37 → 29.32 |
| Mean | 1380.31 → 1393.99 (+0.99%) | 29.590 → 29.520 (−0.24%) |
| 32K, one confirmation pair | 1291.28 → 1342.41 (+3.96%) | 27.81 → 28.06 |

Each prefill pair improves; the absolute ranges are 1375.09–1385.64 and
1392.32–1396.83 tok/s. Decode ranges overlap (29.37–29.79 / 29.32–29.87),
with no decode code change. All eight runs exit 0 with equal, finite
200,058-logit vectors and equal 64-token continuations. Minimum available
memory is 7.33 GiB, PSI is zero, and busy clocks are 2,177–2,184 MHz,
mostly 2,184. The retained trace reduces full attention to 251.780 ms;
prefill wall time is 5,850.501 ms. This is the third adopted Prefill round;
The following rounds target plain Decode.

The attention fixture now covers 36 NVFP4 cases in addition to the
12 sliding cases: query widths 9, 63, 64, 65, 129 and 2,048, each at
positions 0, 1,023, 1,087, 1,088, 6,144 and 32,768. This checks the first
eligible old tile, incomplete query tiles and long prefixes against the
original kernel. All outputs are finite and equal.

Two probes did not enter this change. QR already hits L1 on 95.05% of
loads and streams useful data at about 198 GB/s, so the proposed cache
preference did not address its measured limit. Permuting the decoded
NVFP4 tile's stores gave about 2.1% isolated improvement, but added selects
and address work; its end-to-end benefit was not tested or counted.

Production executable SHA256:
`4af7fc639822f85bd6d15781fb309c536d25599cb2f5da6aa7d31f977d28186b`.


## D1: group the one-row input projections

The retained P3 trace with PDL disabled spends 2,465.520 ms (28.53% of
8,642.745 ms decode) in Q/gate/r and K/V projections. Each layer reads
39.15 MB of Q4 weights: a 163.1 µs floor at 240 GB/s, against 200.64 µs
measured. The separate small r/k/v matrices use scalar kernels; the large
Q/gate matrices use staged streams. An isolated r projection takes
20.22 µs, with 52.26% long-scoreboard stalls and about 98 GB/s useful
bandwidth. Its in-model time is 14.13 µs; NCU cache replay and the isolated
input differ and must not be treated as the same measurement.

`inkling2_mm` now reuses the existing `ds4_gpu_gvq_q4_decode_many` helper
at one row. Q/gate/r share one launch, and sliding-layer K/V share another.
Full-layer K and V retain their distinct inputs. Unsupported formats use
the existing fallback; `DS4_INKLING2_Q4_GROUP=0` disables the dispatch.
No CUDA kernel, persistent tensor or allocation is added. The helper stages
small matrices with two splits instead of their scalar eight-split path;
this changes accumulation order (class 5.2).

The isolated replay uses the actual five layer-0 Q4 payloads, verified by
SHA256, synthetic BF16 inputs, and a large-shared-memory primer. Sixteen
rotating copies total 626 MB, with 117 MB of small matrices, exceeding L2;
the initial three-copy experiment was cache-biased and is excluded. The
small BF16 sink operation between projections is omitted. Three pairs:

| Layer type | Separate projections (µs) | Grouped (µs) |
|---|---|---|
| Sliding | 198.474 / 198.203 / 199.436 | 172.348 / 173.276 / 176.423 |
| Full | 199.707 / 198.251 / 205.792 | 180.892 / 182.860 / 183.217 |

NCU's grouped Q/gate/r kernel takes 158.98 µs for 33.84 MB, about 89% of
the 141.0 µs bandwidth floor. It uses 46 registers, 10.368 KB shared memory
and no spills. In the retained whole-model trace, the target falls to
2,141.941 ms (−13.12%) and 26,624 launches from 61,440. Decode wall time
falls to 8,331.216 ms, with 98.49% GPU coverage. These PDL-off traces explain
the target; the following unprofiled rates use default PDL.

### Arithmetic gate

`tests/inkling2_quality` teacher-forces the pinned
`tests/fixtures/inkling2-decode-quality.json`: eight authored multilingual,
code, math and operations answers, plus 32,729- and 65,496-token prefixes,
128 assistant positions each. Official template/tokenizer hashes and all
token IDs are pinned in the fixture (SHA256
`7138b14cf16000ea3c2b28a881680f945721ab8ec40672a269721740c7bb416d`).
This is an arithmetic regression corpus, not a held-out capability score.
The Rust recorder checks every usable vocabulary logit for finiteness and
records stable F64 log-sum-exp NLL and top-1. Its model-free self-test checks
normalization, ties, extreme logits and invalid inputs.

Three fresh guarded processes compare separate projections, the accepted
`DS4_INKLING2_GAIN_DOT=0` arithmetic switch with separate projections, and
the candidate. All 1,280 positions pass finite checks in each process.
Movement from the first configuration:

| Positions | NLL MAE, noise → candidate | NLL RMS, noise → candidate | Top-1 changes, noise → candidate |
|---|---:|---:|---:|
| All 1,280 | 0.030592 → 0.031119 | 0.066852 → 0.071854 | 19 → 19 |
| Short, 1,024 | 0.027977 → 0.029464 | 0.066311 → 0.072515 | 12 → 12 |
| 32K, 128 | 0.040326 → 0.037912 | 0.072290 → 0.075324 | 5 → 3 |
| 64K, 128 | 0.041778 → 0.037572 | 0.065490 → 0.062366 | 2 → 4 |

Baseline / noise / candidate mean NLL is 1.305673 / 1.305274 / 1.303290;
target top-1 correct is 866 / 864 / 868. Not every movement is smaller than
the noise reference: global MAE/RMS is slightly larger, and math NLL rises
0.018277 with the same 91 correct targets. At 32K, candidate and noise both
score 84 correct versus 87 baseline; at 64K, candidate retains 53 versus
52 for noise. This supports the bounded arithmetic gate, with no aggregate
quality loss observed; it does not establish general capability or MTP
quality. Independent read-only review agrees with this interpretation.

### Fresh full-model comparison

Decode rounds extend generation from the Prefill rounds' 64 to 256 tokens
in both arms. The baseline is the archived P3 executable, with ABBAAB order.

| Fresh pair | Prefill P3 → D1 (tok/s) | Decode P3 → D1 (tok/s) |
|---|---:|---:|
| 8K, 1 | 1394.51 → 1390.58 | 29.45 → 31.38 |
| 8K, 2 | 1384.71 → 1380.80 | 30.02 → 30.88 |
| 8K, 3 | 1386.69 → 1396.06 | 29.68 → 31.42 |
| Mean | 1388.64 → 1389.15 (+0.04%) | 29.717 → 31.227 (+5.08%) |
| 32K, one confirmation pair | 1347.70 → 1343.03 (−0.35%) | 27.85 → 29.61 (+6.32%) |

All three decode pairs improve, with nonoverlapping ranges 29.45–30.02 and
30.88–31.42 tok/s. About 2% variation remains within each arm despite the
same 2,184 MHz busy clocks, 56–62 °C temperatures at 8K, zero PSI and equal
256-token 8K continuations; no specific cause for that residual variation
is established. Minimum available memory over the eight speed runs is
7.39 GiB. Prefill logits are finite and equal over all 200,058 vocabulary
entries at both contexts. At 32K, greedy continuations diverge at token 57;
that pair follows different continuations thereafter, so the identical
8K workloads are the repeated performance evidence and teacher-forced
positions are the arithmetic gate.

The compiled dispatch regression passes 144 combinations of counts,
widths, switch state and fallback outcomes. Production, NVTX and quality
builds, NLL self-test and independent code review pass. This is **Decode
round 1**. Production executable SHA256:
`077c5dae4b60afc67da48bd77f26bf5b60ceba1e641e852b90d2fbeb466e3a48`.

## D2: stage sliding decode tiles and expose more splits

A new retained-D1 trace attributes 355.922 ms (4.23% of 8,417.847 ms decode)
to sliding attention, 34.758 µs per call. The isolated replay uses the
production 3,071-row ring and 16 rotating caches (201 MB), with synthetic
BF16 values and hot Q/bias/sink arrays. Attention plus merge takes 36.162 µs,
close to the model's 37.530 µs. Useful KV reads alone are 4.194 MB, a
17.48 µs floor at 240 GB/s. NCU's cold replay takes 39.87 µs; 2,394 of
3,146 long-scoreboard samples wait at stores immediately following KV
global loads. Eight splits expose only 64 CTAs and 16.82% occupancy.

Reusing the existing verify MMA kernel loses: sliding attention plus merge
rises from 38.792 to 49.251 µs, and full attention from 85.891 to 126.188 µs.
It computes 64 query slots for six live heads and requires 177/184 registers
and 69.6/53.2 KB shared memory. That resource and work cost rejects the probe;
it does not enter production. Whole-workload measurement was restarted on
unchanged D1 before selecting the following alternative.

One-row sliding attention now copies 16-byte KV pieces asynchronously into
the existing shared tile, loads raw relative bias before QK, and splits at
64 keys instead of 128 (16 splits at a full window). No buffer is added.
At eight splits, asynchronous copies alone take 28.807 µs; bias prefetch
reduces a separate paired probe from 28.553 to 27.952 µs. Sixteen splits
beat both eight and 32 in the final probe. Three matched isolated pairs,
including merge, take 35.377 / 35.329 / 35.455 µs before and
25.871 / 25.988 / 25.610 µs after.

NCU time falls to 29.31 µs with 33.01% occupancy. Registers remain 51,
shared memory 19.46 KB, spills zero, and issued instructions are nearly
unchanged (2.303 M → 2.301 M). The extra splits reread Q and write more
partials: global read sectors rise 140,288 → 146,432, and store sectors
26,112 → 52,224. These are request sectors, not measured DRAM bytes.
The fixed partial buffer already accommodates 64 splits. Changing the
merge order is class 5.2; copies at matched split geometry are class 5.1.
`DS4_INKLING2_SWA_DECODE=0` restores the old path. Prefill, verify and SSD
segment dispatch remain unchanged.

### Arithmetic and performance gates

Three fresh quality processes compare D2 off, D2 off plus the accepted
gain-dot switch, and D2 on. The off result exactly reproduces D1's retained
quality result. All 1,280 positions have finite full-vocabulary logits.

| Positions | NLL MAE, noise → candidate | NLL RMS, noise → candidate | Top-1 changes, noise → candidate |
|---|---:|---:|---:|
| All 1,280 | 0.029163 → 0.021463 | 0.065354 → 0.055282 | 14 → 17 |
| Short, 1,024 | 0.026489 → 0.018790 | 0.062308 → 0.053242 | 9 → 9 |
| 32K, 128 | 0.041058 → 0.030262 | 0.081322 → 0.067733 | 1 → 4 |
| 64K, 128 | 0.038661 → 0.034049 | 0.070989 → 0.057405 | 4 → 4 |

Baseline / noise / candidate mean NLL is 1.303290 / 1.303950 / 1.305609;
target top-1 correct is 868 / 862 / 869. Candidate NLL rises 0.002319 while
absolute/RMS movement is below the measured reference in each stratum.
At 32K and 64K, correct targets rise from 84/53 to 86/54; Korean falls
98 → 96, as does the noise reference. Cumulative D1+D2 against pre-D1 has
mean NLL change −0.000064 and 17 top-1 changes, with RMS 0.069089 versus
the original noise's 0.066852. This bounded gate does not claim every
metric improves or establish general capability.

The repeated speed comparison lengthens both arms to 1,024 generated
tokens. It uses archived D1 and current D2 in ABBAAB order, default PDL.

| Fresh pair | Prefill D1 → D2 (tok/s) | Decode D1 → D2 (tok/s) |
|---|---:|---:|
| 8K, 1 | 1394.93 → 1394.82 | 30.65 → 31.78 |
| 8K, 2 | 1390.69 → 1392.87 | 30.95 → 31.34 |
| 8K, 3 | 1382.60 → 1380.97 | 30.82 → 31.35 |
| Mean | 1389.41 → 1389.55 (+0.01%) | 30.807 → 31.490 (+2.22%) |
| 32K, one confirmation pair | 1332.30 → 1341.18 (+0.67%) | 29.03 → 30.08 (+3.62%) |

Every repeated pair improves, with nonoverlapping decode ranges
30.65–30.95 / 31.34–31.78 tok/s. All eight processes exit successfully;
minimum available memory is 7.34 GiB and PSI is zero. Busy clocks are
2,184 MHz except two 2,177 MHz samples in the 32K candidate. Temperatures
span 56–67 °C at 8K and 62–72 °C at 32K. Residual variation within each
arm remains unassigned. All prefill logits are equal and finite over
200,058 entries. The 8K continuations match for all 1,024 tokens; the 32K
pair diverges at token 57 and is a confirmation with different later routes.

The retained PDL-off trace reduces sliding attention to 238.668 ms
(23.307 µs per call, −32.94%). Merge rises 34.059 → 39.703 ms; net target
time falls 111.610 ms, 1.33% of the baseline decode wall. Total traced
decode falls 8,417.847 → 8,319.939 ms. Other kernels vary, and PDL-off
attribution is not a prediction of the 2.22% default-PDL throughput gain.

The fixture passes 68 finite/equal cases: 48 previous prefill cases and
20 decode cases covering short tails, split boundaries, ring wraps and
32K positions at matched geometry. Preserved BF16 and NVFP4 fallback SASS
matches D1 after normalizing constant-bank offsets. Production, NVTX and
quality builds and independent code/numerical review pass. This is
**Decode round 2**. Production executable SHA256:
`fea070eaf6543f248b851aaa567357fcf903da4515a2d30a4f610811488ad9fd`.

## D3: batch raw loads in long-context NVFP4 decode

The retained D2 trace at 32K + 256 tokens spends 611.755 ms, 6.99% of
8,750.013 ms decode, in full attention (298.708 µs per call). Its
37,749,888-byte KV payload has a 157.291 µs floor at 240 GB/s, before
conversion, QK, softmax and partials. Sixteen rotating synthetic caches
exceed L2; their isolated attention plus merge takes 313.198 µs. NCU's
cold-cache replay takes 353.82 µs. Of 11,389 long-scoreboard samples,
9,047 wait at the raw FP4 consumer and 1,886 at scale conversion.

The original loop reads and decodes three groups in sequence. The new
path issues the groups' independent raw data and scale loads first, then
converts them. Volatile PTX keeps the third load ahead of conversion;
ordinary C++ batching let the compiler sink it. The path also reuses the
raw relative-bias prefetch. Split count, arithmetic and storage stay the
same (class 5.1). Three isolated 32K pairs, including merge:
311.103 / 311.564 / 311.586 → 294.529 / 293.043 / 293.456 µs. Confirmation
at 8K is 85.294 → 82.633 µs, and at 64K 622.397 → 576.637 µs.

NCU improves to 335.17 µs; long-scoreboard samples fall to 7,607.
Registers rise 54 → 64, with the same 19.456 KB shared memory, zero spills
and five CTAs per SM. Issued instructions rise 76.629 M → 77.683 M
(+1.38%). No persistent tensor allocation is added. A raw shared staging
alternative needs 4,608 extra bytes and reduces residency to four CTAs;
its initial 322.397 µs and simplified-copy 305.709 µs both lose to the
register batch. That additional resource cost is not retained.

### Whole-model evidence and the short-context boundary

The initial comparison uses archived D2 and the candidate, with three
fresh 32K + 1,024-token pairs in ABBAAB order and default PDL:

| Fresh pair | Prefill D2 → D3 (tok/s) | Decode D2 → D3 (tok/s) |
|---|---:|---:|
| 32K, 1 | 1339.51 → 1342.28 | 29.66 → 30.23 |
| 32K, 2 | 1334.51 → 1339.99 | 29.00 → 29.65 |
| 32K, 3 | 1344.59 → 1338.92 | 29.88 → 30.25 |
| Mean | 1339.54 → 1340.40 (+0.06%) | 29.513 → 30.043 (+1.80%) |
| 64K, one confirmation pair | 1253.21 → 1252.35 | 27.52 → 27.74 (+0.80%) |

All three long-context pairs improve, but their absolute decode ranges
overlap: 29.00–29.88 / 29.65–30.25. An 8K confirmation instead falls
31.33 → 30.97 tok/s. To distinguish the path from build differences,
three additional fresh 8K pairs use the same binary with
`DS4_INKLING2_FULL_DECODE=0/1`: 31.62 / 31.52 / 30.86 →
30.98 / 31.54 / 31.03 tok/s. The mean changes −0.48%, with mixed pair
directions and overlapping ranges. Neither a consistent short-context
gain nor a consistent regression is established. A same-binary 32K pair
improves 29.40 → 29.93 tok/s (+1.80%).

The retained path therefore applies only to one-row resident NVFP4
attention with **at least 32,768 actual keys**. This is the current
qualified long-context boundary, not a measured optimal crossover.
Short contexts keep the original kernel; verify and segmented SSD paths
also remain unchanged. `DS4_INKLING2_FULL_DECODE=0` disables the new path.
The extra instructions/registers lack a demonstrated end-to-end benefit
in the short regime. No context limit or cache format changes.

After this dispatch restriction, all four decode kernel bodies retain
the same SASS (constant-bank offsets normalized). A new 8K trace selects
only the original full-attention kernel, and its 200,058 finite logits
and 256 generated IDs match D2. The final same-binary 32K off/on pair
improves **29.51 → 29.67 tok/s (+0.54%)**, with prefill 1344.43 → 1348.65.
Its full logits and 1,024 generated IDs match. Thus the initial +1.80%
mean is a bounded sample, not a fixed expected gain; all five measured
32K pairs improve, including both same-binary confirmations.

The retained 32K PDL-off trace reduces the full-attention target
611.755 → 578.024 ms (298.708 → 282.238 µs, −5.51%), agreeing with the
isolated measurement. Its 33.730 ms saving is 0.39% of the baseline
decode wall; total wall falls 8,750.013 → 8,654.265 ms. Other kernel times
vary. The larger default-PDL throughput gain is not all attributed to
that exclusive PDL-off kernel saving.

All ten initial speed runs have equal, finite full-vocabulary prefill
logits and equal 1,024-token continuations. The eight additional controls
and final long pair do too. PSI is zero; minimum available memory is
7.07 GiB initially, 7.21 GiB in controls and 7.29 GiB in the final pair.
Busy clocks span 2,177–2,190 MHz, mostly 2,184; temperatures reach 71 °C.
Residual absolute variation is not assigned a specific cause.

### Regression evidence

The fixture passes 89 finite/equal cases, including 21 new NVFP4 decode
cases at tails, page/split boundaries, 32K and 64K. Independent review
found that its original periodic synthetic data repeated between batches.
The fixture now mixes high address bits; a deliberate mutant reusing
batch 0's data and scale fails its comparison assertion. The production
code required no corresponding correction.

A fresh quality process exactly reproduces D2's NLL and top-1 records at
all 1,280 teacher-forced positions, with all 200,058 vocabulary logits
finite. The final dispatch selects between those equal arithmetic paths;
the cumulative D1+D2 quality boundary still applies. Production, NVTX and
quality builds and independent code/dispatch reviews pass. This is
**Decode round 3**, qualified at 32K/64K, completing three Prefill and
three Decode rounds. The final round executable SHA256 is
`671bc6615f70343d8f37b249ecbcf5069a48b8081c333c3e24823b544b073dfa`.
