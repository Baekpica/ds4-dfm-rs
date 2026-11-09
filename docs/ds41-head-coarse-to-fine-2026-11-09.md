# DS4.1 coarse-to-fine head: implementation and gates (2026-11-09)

Question: does the gated coarse-to-fine head (L2, k=4096) hold its 100%
candidate recall and top-1 agreement when implemented as the served kernels,
and what does it cost per call?

Gate record for the decision: [ds41-head-gate-2026-11-09.md](ds41-head-gate-2026-11-09.md)
(768 positions over a 2K and an 8K window, 100% recall at k=4096, 372.3 MB ->
144.2 MB per head call). This record covers the implementation.

## What was built

`cuda/ds41_head.cuh`, entered as `ds4_gpu_v41_head_c2f_tensor`
(`ds4_gpu.h`), dispatched from `v41_head_proj` (`ds4_ds41_forward.inc:1176`).
The head is called from two places — the generate forward
(`ds4_ds41_forward.inc:1236`, one row on `xlast` for a block) and the draft
tower (`ds4_ds41_draft.inc:351`, up to eight rows) — and both take the new
path.

Per call, five kernel launches in the current stream (so a captured decode
graph replays them):

1. gather each token row's sketch columns (`v41_head_xsel_kernel`);
2. coarse score `out[t][v] = scale[v] * <q8[v], x[t]|cols>`
   (`v41_head_coarse_kernel`, one warp per vocab row);
3. radix top-3,584 over ids >= 512 plus the 512 anchors
   (`v41_head_topk_kernel`, the indexer's method: 8 bits a round, segmented
   ascending write-out, so the set and its order are deterministic);
4. fill the row with `-INFINITY` (`v41_head_fill_kernel`);
5. exact q4_K rescore of the 4,096 candidates into their own ids
   (`v41_head_rescore_kernel`, one warp per (token, candidate) pair).

Step 5 writes into the logits row the caller already owns, so the truncated
row keeps the logits interface: the argmax kernel, the markov bias add and
the per-row pick are unchanged, and a non-candidate id can never win.

The sketch is built lazily on the first coarse call
(`v41_head_build`): column leverage over a 512-row stride sample, the top
1,024 columns, per-row symmetric Q8. Nothing is written to disk; the
artifact is untouched. Device memory: 126.2 MB of Q8 rows plus 0.5 MB of
scales, and 4.4 MB of per-call scratch.

## Contract

Greedy only. The device sampler and the sampled-speculation verify draw from
the whole logits row, so a truncated row would change the top-p/top-k
support. `st.head_c2f` is set in `ds4_v41_generate_argmax` only when the
request is a bare argmax (`!dev_sample && !penal`) and no `DS41_GEN_LOGITS`
dump is open — the dump is the row a caller diffs, so it keeps the full head.
`DS41_HEAD_FULL=1` forces the full head (the A/B control and the kill
switch). Fallbacks, each of which returns to the full head for the whole
process: a non-q4_K head, a shape outside the path (dim > 8192 or not a
multiple of 256, vocab <= 512 or not a multiple of 4), and a capture that
started before the sketch was built.

## Pre-flight (workstation RTX 4070 SUPER, the exact kernels)

The real head bytes (`output.weight` q4_K, 355.1 MiB) and the gate's own
inputs: `DS41_GEN_HIDDEN` rows and the engine's full-head `DS41_GEN_LOGITS`
rows from the two windows. Harness: `misc/scratch/ds41_head_c2f.cu`.

Sketch build: 3.0-3.2 ms (leverage over 513 sampled rows plus Q8 over
129,280 rows).

| window | positions | width | recall | top-1 | true row first | per call |
|---|---|---|---|---|---|---|
| 2K | 256 | 1 | 256/256 | 256/256 | 256/256 | 0.49 ms |
| 8K | 512 | 1 | 512/512 | 512/512 | 512/512 | 0.48 ms |
| 8K | 512 | 2 | 512/512 | 512/512 | 512/512 | 0.51 ms |
| 8K | 512 | 6 | 512/512 | 512/512 | 512/512 | 0.78 ms |
| 8K | 512 | 8 | 512/512 | 512/512 | 512/512 | 0.97 ms |

The pre-flight reproduces the gate's result on the implementation that
serves: 100% candidate recall and 100% top-1 over 768 positions, and the
true row ranked first inside the candidate set every time. Traffic per call
is 126.2 MB of sketch + 11.2 MB of rescored rows + 0.49 MB of fill = 138.0
MB against 372.3 MB for the full head.

The workstation rates are not GB10 rates; they establish the correctness of
the kernels and that the pass is bandwidth-bound as designed.

## In-model gate (Spark GB10)

Binary `38aa5e9b543f5257e6e0233fc2ddbd8772b5d97140527c1d5aec828780abfdf9`
(`make ds4-server CUDA_ARCH=sm_121` in `~/ds4-check-l6`), one boot per arm,
two boots per arm in opposite order, engram interleaved sidecar on, greedy
requests from the pinned `speed-bench/promessi_sposi.txt` corpus, 128 tokens
per frontier, clocks 2,444-2,476 MHz. `DS41_HEAD_FULL=1` is the control and
the binary is the same in both arms.

| MTP | frontier | metric | full head | coarse-to-fine | delta |
|---|---|---|---|---|---|
| off | 2048 | prefill tok/s | 1,053.5 | 1,023.8 | -2.8% |
| off | 2048 | TTFT s | 2.0425 | 2.1018 | +59 ms |
| off | 2048 | decode tok/s | 30.15 | 30.55 | +1.3% |
| off | 8192 | prefill tok/s | 1,195.0 | 1,196.6 | +0.1% |
| off | 8192 | TTFT s | 7.0794 | 7.0700 | -9 ms |
| off | 8192 | decode tok/s | 30.15 | 30.70 | +1.8% |
| auto | 2048 | decode tok/s | 42.1 | 42.9 | +1.9% |
| auto | 8192 | decode tok/s | 33.75 | 33.6 | -0.4% |

Readings:

- Decode is the target and it moves: +1.3% at 2,048 and +1.8% at 8,192 with
  MTP off (the four arms do not overlap: 30.2/30.1 against 30.7/30.4 and
  30.7/30.7), and +1.9% at 2,048 with MTP on at the same 2.78 tokens per
  step. The 8,192 MTP arm is
  flat inside its own 2.1% spread, and its step got 3-4% more expensive
  while yielding 2.42 tokens per step instead of 2.33 (the scheduler picks
  wider verify rounds when acceptance rises), so that arm is not evidence
  of a regression but it is not evidence of a gain either.
- Prefill is neutral, as it should be: the head is one call per 2,048-token
  block, ~0.5 ms of a ~1.9 s block. The 2,048 frontier looks 2.8% worse
  only because the first request of a process pays the one-time sketch
  build, measured at +59 ms of TTFT (the 8,192 request of the same process
  is neutral).
- Output: all eight arms emitted byte-identical completions at both
  frontiers (`a9933bdc...` at 2,048, `de8aafa3...` at 8,192; the same hashes
  the pre-L3 arms produced). 256 positions per frontier, two MTP modes.
- The sketch builds once per process and logs
  `head sketch: 1024 cols over 129280 rows (126.2 MB), candidates 4096`;
  no arm refused the path and no arm fell back.
- Device memory: 126.2 MB of Q8 rows, 0.5 MB of scales, 0.03 MB of column
  ids and 4.3 MB of per-call scratch (coarse scores, gathered activations,
  candidate ids), 131 MB in total.

Verdict: adopted as the greedy default; `DS41_HEAD_FULL=1` restores the full
head. The measured gain is smaller than the gate's 2.4-3.5% projection
because the full head's own GEMV is cheaper in the served step than the
isolated attribution suggested; +1.3% to +1.9% decode for 131 MB and a 59 ms
one-time build is still a gain with no output change.


## Limits

- k=4096 was gated on 768 positions across two frontiers; it is not a proof
  for every prompt.
- Four final-scale ids can only be reached through the anchors, which is
  why the 512 anchors stay unconditional.
- The draft head's proposals are re-ranked by the markov bias inside the
  candidate set; a proposal outside the set is unreachable. Correctness is
  unaffected (the verify accepts only main-model matches); the 8,192 MTP arm
  shows acceptance rising slightly (2.33 -> 2.42 tokens per step), and the
  emitted tokens are unchanged.
- The 59 ms build lands inside the first request's TTFT. Moving it to the
  weight-promote pass is possible if that ever matters.

## Evidence

Workstation: `misc/scratch/ds41_head_c2f.cu` (harness) and its run over
`misc/scratch/gen.hidden`/`gen2k.hidden` with `ds41_output_q4k.bin`
(scratch, gitignored).
Spark: `/tmp/l6/build.log`, `/tmp/l6/ab.log`, `/tmp/l6/arm.sh`, and per arm
`/tmp/l6m/<name>/{server.log,resp_2048.json,resp_8192.json}` plus
`<name>.csv` and `<name>.guard.jsonl`. The measurement tree is
`~/ds4-check-l6`; the live server was stopped for the window and restored
from the pinned binary `139aef25...` at the recorded argv (pid 209575).
