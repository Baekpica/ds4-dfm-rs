# MiniMax VQ: long-context prefill, October 29

**Two new Prefill rounds adopted; the three-round campaign is in progress.**
October 25/27 rounds are historical and do not count here.
[Receipts](benchmarks/minimax-vq-prefill-2026-10-29/README.md) retain raw
samples, numerical costs and workload boundaries.

## Protocol

GB10, CUDA 13.4.92, system cuBLAS 13.8.0.4, KMD 615.71.10. The enabled
clock service keeps 300–2200 MHz; busy samples were 2177–2190 MHz, usually
2184. Main CPU 5 stayed at 3.9 GHz; helper affinity was 5–9,15–19.
The artifact is the unchanged 61-part MiniMax-M3.1-Flash-VQ-GGUF release
(`release.json` SHA256 `2110c1e5cba8d59987e370d50dab6dde5d5f7fc82009cdc27d8131452536f4b0`).
Source starts at `ef41bd6d`; receipts identify each dirty patch and binary.

Each A/B uses fresh processes, cleared page/driver cache, the host guard
with 2 GiB reserve/trip and 20% PSI limit, one GPU job, chunk cap 2048,
context allocation 131072, PDL enabled and lazy CUDA loading. The prompt
is `speed-bench/promessi_sposi.txt`. After a 4096-token warmup, it prefills
8192 tokens, then the 57344-token suffix to 65536, with 256 greedy tokens
at each frontier. Three pairs run off/on/on/off/off/on. **The 64K rate is
a suffix rate, not a cold full-prompt rate or the last 2K chunk's rate.**
Profiles generate 32 tokens and establish attribution, not adoption TPS.

## Why prefill slows with context

The initial profile's equal 2048-row chunks separate context growth from
fixed expert/projection work:

| Frontier | Chunk wall ms | Index scorer ms | Attention ms |
|---:|---:|---:|---:|
| 8192 | 1905.47 | 16.68 | 152.91 |
| 65536 | 2060.18 | 105.92 | 220.52 |

Scoring grows by 89.24 ms and attention by 67.61 ms, explaining the
154.71 ms wall increase. At 64K they occupy 5.14% and 10.70% of the chunk.
This is MiniMax's sparse-attention execution path; Qwen's recurrent GDN/QSA
curve does not establish a transferable kernel or architecture change.

## P1: residual NVFP4 index scoring

The existing index keys are E2M1 values with E4M3 scales per 16 values.
Encode two query components with the existing codec:
`q_hi = encode(BF16(q))`, `q_lo = encode(BF16(q - decode(q_hi)))`.
Score both against the unchanged keys with native block-scaled
[`m16n8k64` MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-block-scaling).
This changes query values and reduction order; model bit identity is not
its numerical contract. No global tensor, KV format or context limit changes.
The existing MXF4 build capability selects this path. Widths up to eight
(including DSpark's supported verify widths) retain the previous path.
`DS4_MINIMAX_SCORE_FP4=0` restores BF16 prefill scoring.

At the 2048-row/64K shape, the BF16 work is approximately 137 GFLOP:
about 1.27 ms at the host's 108 TFLOP/s peak. Two native FP4 components
have a roughly 0.64 ms compute floor at the corresponding FP4 peak.
The isolated warm median falls from 1.740 to 1.272 ms; cold full NCU
replay gives 2.175 versus 1.319 ms. These cache regimes are kept separate.
The candidate uses 80 registers, no spills, 30 KiB user shared plus
1 KiB driver shared (BF16 uses 34 KiB dynamic user shared). It executes
more arithmetic/instructions; native FP4 throughput outweighs that cost.

The actual-model reprofile confirms scorer **105.855 → 75.677 ms** in the
last 64K chunk (−28.5%); chunk wall is 2060.745 → 2036.455 ms (−1.18%).
At 8K the scorer is 16.434 → 11.235 ms.

### Fresh whole-workload results

| Frontier | Prefill off, median [range] | Prefill on, median [range] | Median gain | Paired gains |
|---:|---:|---:|---:|---|
| 8192 | 1089.49 [1087.45, 1093.62] | 1095.24 [1090.47, 1095.65] | +0.53% | −0.29%, +0.53%, +0.75% |
| 65536 | 1025.67 [1025.19, 1025.85] | 1035.52 [1033.01, 1036.78] | **+0.96%** | +0.70%, +1.01%, +1.08% |

All six runs exit 0. Every repeated setting matches 400128 finite frontier
logits and 512 greedy IDs exactly; off also matches the initial baseline.
Minimum available memory is 8.25 GiB and recorded PSI is zero.
The consistent 64K gain, rather than a fixed percentage threshold, supports
adoption. Short-frontier variation remains visible.

### Numerical cost

Seven cases retain held-out English/Korean/Chinese questions and assistant
continuations, with a Promessi prefix inside the user turn: three at 8K,
one at 32K, three at 64K. They provide 1529 assistant positions and 1528
paired next-token NLLs. A same-binary, same-host FP4-off reference is
recorded first; the existing `DS4_MINIMAX_ATTN_SIMT=1` F32-attention path
measures accepted arithmetic variation. Leave this presence-based variable
**unset**, rather than setting it to `0`, for baseline/candidate runs.

| Frontier | F32-attention ΔNLL / RMS / top-1 | P1 ΔNLL / RMS / top-1 |
|---:|---:|---:|
| 8192 | −0.00486 / 0.33221 / 95.60% | −0.00622 / 0.33355 / 96.23% |
| 32768 | −0.00347 / 0.25945 / 95.33% | −0.01560 / 0.21148 / 95.72% |
| 65536 | +0.00817 / 0.38133 / 96.86% | +0.00899 / 0.35693 / 95.91% |

Aggregate NLL is 1.563933 against 1.565392; paired RMS is 0.326846 against
the accepted perturbation's 0.343130. P1 changes six additional top-1
predictions at 64K and one additional prediction overall compared with that
perturbation. The long NLL bias is at the measured reference level; the
remaining top-1 difference is disclosed, not hidden by short cases. The
legacy 2%/95% aggregate check also passes, but is not the measured floor.
These results do not establish broad quality improvement.

### Decode check and variability

Raw-Promessi greedy medians were 21.22 → 21.00 at 8K and 20.86 → 20.59
at 64K. Both paths produced repetitive continuations, with different token
histories; unchanged candidate repeats alone spanned 20.33–21.04 at 64K.
These figures are retained, but do not isolate a Decode execution regression.
CPU/SM clocks and PSI were stable; sampled thermal-slowdown counters were
zero. The 32-token profile's Decode GPU union differs by only 0.16% at 64K.

Six additional fresh processes therefore teacher-force the **same 256
assistant tokens** at each frontier. The timer includes evaluation and the
synchronized logit copy, excluding CPU reference scoring:

| Frontier | Off median [range] | On median [range] | Median change |
|---:|---:|---:|---:|
| 8192 | 20.7425 [20.6320, 20.7975] | 20.7200 [20.6065, 20.7921] | −0.11% |
| 65536 | 20.7170 [20.3371, 20.9171] | 20.9617 [20.8389, 20.9846] | +1.18% |

Both frontiers have mixed paired signs. The greedy median loss does not
reproduce on matched input; no material fixed-tail regression is observed.
Residual process variation is unresolved, so **no Decode speedup is claimed**.

### Checks and rejected trials

- Native score fixtures pass both switches at 9/31/33/128/2048 rows,
  partial causal blocks, zero/tiny queries, 64K and synthetic 1M keys.
- Independent CPU quantization/double scores check the arithmetic. A
  fixture failure exposed fast-math reciprocal division moving a CPU
  nearest-even tie; distance comparisons in input units fix the oracle.
  The same GPU object passed the strict-CPU diagnostic before that fix.
- The reference-capture unit test passes; fixed-tail runs verify positive
  forward time bounded by total Decode time. Seven retained scorer/Decode
  instruction streams match the baseline after masking constant-bank offsets.
- A striped shared-bank mapping improved the scorer 2–3% but increased
  instructions/requests and did not improve repeated whole-workload Prefill.
  It was removed. An XOR variant used more registers or spilled.
- Single-component FP4 passed the legacy gate but had 64K ΔNLL +0.02907
  against +0.00817 from the accepted perturbation. It was not adopted.

## P2: prepare both query components in registers

P1's retained whole profile still attributed 75–78 ms of the last 64K
chunk to scoring. Detailed NCU source attribution put 37.2% of the scorer's
issued instructions in query encoding. Keep the four query values per lane
in registers, convert pairs with CUDA's native nearest-even E2M1 conversion,
and derive the residual directly from the high-component register bytes.
This removes a query reload, shared residual traffic and three warp barriers
per vector. The BF16 boundaries, E4M3 scales, dot order, key writer and
32-query/32-key-block geometry remain the same. Signed-zero encodings may
differ; no effort is spent canonicalizing them.

`DS4_MINIMAX_SCORE_PREP=0` restores P1 preparation inside the native FP4
scorer. Both paths use 80 registers, zero stack/local traffic and 30 KiB
user shared memory. There is no added tensor allocation. At 64K the isolated
warm median is **1.2755 → 0.9926 ms**. Full NCU counts fall from 297.83 to
202.63 million instructions and from 27.64 to 17.68 million requested global
load sectors. The candidate's cold replay is 1.0676 ms; its paired scalar
capture is an unexplained 1.7166 ms outlier (previously 1.3152 ms), so that
inflated cold speedup is not used.

The actual-model last-chunk scorer falls **75.641 → 61.478 ms (−18.7%)**
at 64K and 11.236 → 8.941 ms at 8K. Whole-profile suffix time falls
55408.66 → 55063.20 ms; these profiled times are separate from adoption TPS.

| Frontier | Prefill off, median [range] | Prefill on, median [range] | Median gain | Paired gains |
|---:|---:|---:|---:|---|
| 8192 | 1090.20 [1087.95, 1093.07] | 1093.11 [1091.27, 1094.68] | +0.27% | +0.41%, +0.47%, −0.16% |
| 65536 | 1035.78 [1031.73, 1037.09] | 1040.01 [1038.14, 1041.63] | **+0.41%** | +0.44%, +0.80%, +0.23% |

All six runs exit 0, with 400128 finite frontier logits and 512 generated
IDs exactly equal across both settings and their repeats. The off path also
matches P1. Native score fixtures pass both settings at all six shapes;
the codec check covers 1,114,154 primitive values and 8,704 complete finite
vectors, including ties and the full BF16 range. Ten retained scorer,
tail and Decode instruction streams match P1. Minimum available memory is
8.255 GiB, PSI is zero and busy clocks are 2177–2190 MHz.

Raw greedy Decode medians are 21.13 → 21.04 at 8K and 20.78 → 20.66 at
64K (−0.43%/−0.58%). Paired signs are −/+/+ at both frontiers, and ranges
overlap: 20.94–21.26 versus 20.73–21.39, and 20.53–20.87 versus
20.26–20.98. These now use identical token histories and unchanged Decode
instructions. The separate 32-token profile varies more (GPU union
+2.39%/+3.19%); clock/PSI observations do not explain it. This unresolved
process variation is retained, with no Decode speedup claim. The consistent
64K Prefill gain and reduced work without an additional resource cost support
adoption; no consistent Decode regression is established by these pairs.

The earlier native-converter-only trial is **not** a separate adoption:
64K Prefill median −0.03%, 8K −0.30%, despite a 16.7% scorer improvement.
It also changed the KV writer, whose measured time increased slightly.
Larger key groups lose short-context parallelism; a 64-query tile uses
50 KiB shared/90 registers, permits only one resident CTA, and loses to
the retained 32-query tile. Neither is included.

Quality is qualified here through 64K text prompts. The 1M fixture is
synthetic; this campaign has not rerun full-model 1M, media or HTTP gates.
The final matched 2K–64K DSpark sweep and one further Prefill adoption
remain open.
