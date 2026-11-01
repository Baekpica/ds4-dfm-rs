# MiMo-V2.6-Flash VQ: MTP context and long prefill, 2026-10-31

GB10 (sm_121, CUDA 13.4.92), busy SM clock 2,177–2,190 MHz under the
300–2,200 MHz policy, UNCENSORED release (`f1cd8523`), source from main
`446951cc`. Every run is a fresh process under the memory guard (2 GiB
reserve and trip; minimum available memory 23.3 GiB, PSI 0).
[Family page](mimo2-vq.md); the [October 30 report](mimo2-vq-optimization-2026-10-30.md)
covers the earlier rounds.

## Drafting: MTP blocks see the window

Since #57 each embedded MTP block attended only to its own earlier draft
rows, stored at a cursor from 0 and never the prompt; a restored snapshot
emptied them. The blocks now run as SGLang's multi-layer EAGLE runs
MiMo-V2: block d's row i reads the target's row i and token i+d+1 at
position i, and every depth keeps the target rows (no hidden chain). The
trunk keeps its last 256 rows and their tokens in a ring; a draft step first
writes the block's rows that are still missing (up to its 128-position
window), committed tokens where they are known and this trial's drafts past
them. The next step rewrites the draft rows, so rejects need no undo and the
per-block journal is gone. After a reset or restore a block restarts its
positions there, so its attention never reads slots from another prefix.

Two MTP drafts against the blind blocks of main (`ds4-bench` with a chat
prompt, 512 greedy tokens, one process each):

| Prompt | Plain decode | Blind MTP 2 | Filled MTP 2 | Tokens per cycle |
|---|---:|---:|---:|---:|
| Code: a Rust LRU cache (105 tokens) | 24.96 | 34.26 | **35.96** | 2.667 → 2.813 |
| Rewrite a 156-line C file (2,365 tokens) | 25.05 | 34.00 | **36.29** | 2.723 → 2.926 |
| Korean explanation (152 tokens) | 24.92 | 24.00 | **25.40** | 1.882 → 2.008 |
| Novel continuation, 2K–6K frontiers | – | – | – | 1.959 → 1.979 |

The blocks attend 128 positions, so the prompt adds context only near the
frontier: acceptance rises 5–8% on the chat prompts, far from GLM's
full-attention draft layer (0.63 → 0.97). SGLang hands the blocks the
target's final-norm hidden; that input accepted 0.5–2% fewer drafts on all
four workloads (2.753, 2.876, 1.992, 1.969 tokens per cycle), so the row
before the norm stays, as the blind path fed it.

## R1: E4M3 queries for full-layer prefill scores

The long-context slope is the nine full layers' attention: a 2,048-row chunk
takes 0.675 s longer at 64K than at 2K, and their QK and PV work grows by
47.9 TFLOP, 0.68 s at the kernel's 70 TFLOPS. `mimo2_ga8::prefill` widened
every FP8 K tile to f16 for f16 MMAs; at 32K its tensor pipe was busy 64% of
the elapsed cycles (NCU on the model-free harness, at the profiler's
2.40 GHz), the top stall waiting on that pipe.

`mimo2_ga8::prefill8` keeps the K codes as stored: cp.async copies them into
a shared tile (208-byte rows put ldmatrix rows in distinct banks) and they
are m16n8k32 E4M3 B fragments as they are. Q becomes E4M3 with one scale per
(row, head); scores take the Q and K scales after the dot, each 128-dim group
in a fresh FP32 accumulator. V and P × V are unchanged. QK runs at the FP8
rate with no K widening and half the K tile traffic.
`DS4_MIMO2VQ_GA8_QK=f16` restores `prefill`; `=fp8x2` adds the Q residual as
a second E4M3 component (a diagnostic of the Q rounding).

Model-free harness (`make test-mimo2-ga8`), 2,048 rows, per layer:

| Frontier | f16 Q (`prefill`) | E4M3 Q | E4M3 Q + residual |
|---:|---:|---:|---:|
| 4,096 | 5.87 ms (73 TFLOPS) | 4.66 ms (92) | 5.70 ms (75) |
| 32,768 | 40.9 ms (69) | 29.8 ms (95) | 36.6 ms (78) |
| 63,488 | 79.0 ms (69) | **56.7 ms (95)** | 70.3 ms (77) |
| FP64 error, random rows | 0.002–0.004 | 0.14–0.33 | 0.005–0.013 |

On the harness's unit-variance random rows E4M3 Q moves the output a hundred
times more than f16 Q. The model does not see it: `tests/mimo2_vq_quality
--long` pairs 1,024 teacher-forced positions after a 32K and a 131,072-token
prompt.

| Against the f16-Q default | ΔNLL ± s.e. | mean \|ΔNLL\| | top-1 agreement |
|---|---:|---:|---:|
| 32K: E4M3 Q | +0.0003 ± 0.0041 | 0.0723 | 95.3% |
| 32K: E4M3 Q + residual | −0.0021 ± 0.0039 | 0.0706 | 95.4% |
| 32K: `DS4_MIMO2VQ_GA8=0` (accepted) | −0.0062 ± 0.0043 | 0.0749 | 95.4% |
| 32K: `DS4_MIMO2VQ_GAIN_DOT=0` (accepted) | +0.0026 ± 0.0044 | 0.0663 | 96.6% |
| 128K: E4M3 Q | +0.0030 ± 0.0044 | 0.0774 | 95.5% |
| 128K: E4M3 Q + residual | +0.0030 ± 0.0043 | 0.0765 | 96.1% |
| 128K: f16 full-layer K/V (accepted format) | +0.0053 ± 0.0046 | 0.0789 | 95.0% |
| 128K: `DS4_MIMO2VQ_GAIN_DOT=0` (accepted) | +0.0073 ± 0.0036 | 0.0621 | 97.1% |

E4M3 Q stays inside the accepted switches' band at both frontiers; the
f16-Q control reproduced the October 30 run byte for byte. The 131,072-token
prompt prefilled in 113.1 s instead of 125.3 s.

Whole workload (2K–64K, 32 frontiers of 2,048 tokens, 16 greedy tokens,
off/on/on/off/off/on):

| Prefill tok/s | f16 Q, median [range] | E4M3 Q, median [range] | Gain |
|---|---:|---:|---:|
| Mean of 32 frontiers | 1,230.1 [1,227.4, 1,231.7] | **1,296.5** [1,295.9, 1,297.6] | **+5.40%** (pairs +5.22, +5.72, +5.40%) |
| 2K | 1,489.7 | 1,494.3 | +0.31% |
| 32K | 1,223.5 | 1,293.0 | +5.68% |
| 64K | 1,000.4 | **1,109.1** | **+10.87%** (pairs +10.66, +11.36, +10.79%) |

The 2K→64K decline shrinks from 32.8% to 25.8%.

## Gates and publication sweep

The assistant-turn gate (`tests/mimo2_vq_quality`, 24,785 targets) gives KL
0.0884, top-1 91.77%, NLL 0.7486 against 0.0891 / 91.75% / 0.7463 with f16
Q (bound KL ≤ 0.109, top-1 ≥ 89.7%). The serial MTP lane (`ds4-server`, two
drafts, reasoning off) continued a chat over its live state, 182 of 199
prompt tokens reused, accepted 2.88 tokens per cycle, and repeated the turn
byte for byte after an unrelated prompt; three drafts ran too (code: 35.73
tok/s, 3.26 tokens per cycle against 33.09 / 2.99 blind). The two-bank reuse
gate cannot carry MTP: MiMo VQ drafts only on the serial lane.

[Publication sweep](benchmarks/mimo2-vq-spark-64k-2026-10-31/README.md) at
`2fa8c08e` (three fresh processes per variant, 128 forced tokens):

| 2K–64K, median of run means | October 30 | October 31 |
|---|---:|---:|
| Prefill, no drafting | 1,226.5 | **1,294.9** |
| Decode, no drafting | 23.97 | 23.96 |
| Decode, two MTP drafts | 25.16 | **26.35** |
| Prefill at 64K | 998.0 | **1,106.1** |

MTP decode on the novel follows each frontier's continuation (22.8–32.4
tok/s); its mean gain over plain decode rose from 5% to 10%. A frontier
whose continuation changed with the prefill arithmetic can draft worse (2K:
28.60 → 23.18).

## Rejected

- **Second head one P × V behind.** The two query heads' warps share each
  scheduler and reach their MMAs and softmax together; running the second
  head's P × V at the start of the next tile (a third V tile) was byte-equal
  but 5% slower at 63,488 (58.8 vs 56.0 ms): the per-tile barrier still
  aligns them.
- **One FP32 accumulator for all 192 dims.** 1.6% faster (55.1 ms) with the
  same harness error; not worth a gate against the documented truncation of
  long FP8 sums.

## Next levers

- **P × V in E4M3.** Half the remaining MMAs and no V widening, with V staged
  transposed and the V scales folded into P; P's rounding needs its own gate.
- **Overlapping softmax with MMAs** needs the two heads off the shared
  per-tile barrier (separate rings or producer warps), which the 99 KiB of
  shared memory and 244 registers leave little room for.
- **MTP.** A snapshot carrying the blocks' rows (a restore restarts them
  now), and fewer head passes: the 0.62 GiB head runs once per draft.
