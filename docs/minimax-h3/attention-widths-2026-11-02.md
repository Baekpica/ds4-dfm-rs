# Three attention widths, measured: the block is not bound by K/V traffic (2026-11-02)

Unit: handoff section 3, steps 0-2. Attention is 64% of an FP8 delivery step -
25.15 s of 39.24 s, 50 calls of 503 ms - so the block is the largest lever left
in the denoise. Section 3's plan was to spend the width: sixteen warps, 256 rows
per CTA on the same 98,304 B ring, halving the K/V traffic per row. The
operator's instruction was to keep every candidate as a **runtime option** rather
than replace one hardcoded shape with another, so the block is now one template
with `H3_DIT_ATTENTION` choosing among four arms. This record is what the four
arms measured, and the answer is that the width is already at its optimum.

**The verdict first.** At the delivery shape (S = 37,722, 56 heads) the arms are
**four warps 653.398 ms, eight 506.392, twelve 511.123, sixteen 660.837** per
call, all four with the same output hash. Twelve is not a tie: three interleaved
rounds of 20 calls put it **+1.0%** behind eight, sign-consistent every round,
because the register ceiling that comes with the width costs more than the K/V
traffic it saves. Sixteen warps spills and loses **30%**. And the two-CTA design
section 3 wanted measured is closed by ptxas rather than by argument: a second
CTA per SM holds the block to 128 registers and produces **exactly** the spill
profile of the sixteen-warp arm (192 B stack, 268 B stores, 516 B loads). Eight
warps therefore stays the default - now one option among four - and the next
lever has to cut work per row, not rows per CTA. Commit: the one landing this
record. Spark `gx10-e1d2`, sm_121a, CUDA 13.0.

## 1. The four arms, one binary

`tests/h3/attn_bm` with the arm in `H3_DIT_ATTENTION` and nothing else changed;
10 calls per figure, 56 heads, head dimension 128, non-causal. `four` is the
older synchronous block the pipelined one replaced, kept as the reference.

| arm | warps | rows/CTA | ring | S = 37,722 | S = 38,900 | `out_fnv` (both shapes) |
|---|---:|---:|---:|---:|---:|---|
| `four` | 4 | 64 | static 32,768 B | 653.398 | 709.405 | `80652b45a36d0166` / `2e63ecb41f0771db` |
| `pipe` | 8 | 128 | 98,304 B | **506.392** | **541.917** | same |
| `pipe12` | 12 | 192 | 98,304 B | 511.123 | 546.529 | same |
| `pipe16` | 16 | 256 | 98,304 B | 660.837 | 705.729 | same |

The hash column is the acceptance: the arithmetic per row does not change with
the width, so a differing `out_fnv` would have been a bug and not a shape. All
eight runs agree at both shapes, which is also the cross-check that the template
conversion did not disturb the eight-warp code path the default uses.

## 2. Twelve warps, interleaved, because the clock drifts

The sweep puts twelve 0.9% behind eight, which is inside this host's run-to-run
spread - the eight-warp arm alone measured 503.081, 506.392 and 509.152 on three
different invocations. So the two arms were alternated in one session, 20 calls
each, three rounds:

| round | `pipe` (8) | `pipe12` | delta |
|---|---:|---:|---:|
| 1 | 509.152 | 514.976 | **+5.824 (+1.14%)** |
| 2 | 513.201 | 519.168 | **+5.967 (+1.16%)** |
| 3 | 516.531 | 520.686 | **+4.155 (+0.80%)** |

Every round lands the same way, and the drift is visible on both arms (8 warps
itself climbs 1.4% across the session), which is why the alternation is the
evidence and the single-shot numbers are not. Twelve warps is not slower by
noise; it is slower by about a percent.

## 3. What the width costs ptxas

The same object compiled with `--ptxas-options=-v` on the Spark, four
instantiations. `MIN_CTA` is the second `__launch_bounds__` argument, and it
exists to be asked rather than shipped:

| instantiation | threads | register ceiling | used | stack | spill stores / loads |
|---|---:|---:|---:|---:|---:|
| `<8, 1>` | 256 | 256 | 208 | 0 B | 0 / 0 |
| `<12, 1>` | 384 | 170 | 168 | 8 B | 4 / 4 |
| `<16, 1>` | 512 | 128 | 128 | 192 B | 268 / 516 |
| `<8, 2>` | 256 | 128 | 128 | 192 B | 268 / 516 |

The last row is the two-CTA question answered. Two CTAs of 256 threads are
65,536 registers, which is the whole register file, so ptxas must fit the block
into 128 per thread - and the block that wants 208 pays with the same spills as
the sixteen-warp arm. Since the sixteen-warp arm, which is the 128-register
ceiling measured, runs 30% slower, a two-CTA build of this body would start
there. That is why the 32-key-tile design section 3 held in reserve was not
written: it is feasible in shared memory (below) and dead in registers, and the
report says so before the benchmark would have.

## 4. The shared-memory arithmetic, corrected and asserted

The device holds **102,400 B per SM**. The 64-key ring is 3 stages x 2 planes x
64 keys x 128 dims x 2 B = 98,304 B, so **two CTAs of it are 196,608 B and do
not fit regardless of registers** - and two stages do not either (65,536 x 2 =
131,072). The handoff's older claim that "128 registers would admit two CTAs"
ignored this. A 32-key tile is 49,152 B per ring, and two of those are 98,304 B,
which does fit - so the tile is the knob that would allow a second CTA, and only
the registers stand in the way. Both facts are now `static_assert`s in the
two-CTA probe rather than prose.

## 5. The default arm, at the delivery canvas, unchanged

The refactor had to leave the shipped arm alone. Same delivery job as
`fp8-delivery-2026-11-02.md` (1344x768, `--frames 121`, 6 steps, seed 42,
`H3_DIT_GEMM=fp8`), rendered twice: once with the templated block, and once more
after the launch label was changed to name the width, so the artifact on disk
comes from the binary this record ships.

| run | `TEXT` fnv | `LOAD` | `DENOISE` | s/step | `FRAMES` | wall | clip md5 |
|---|---|---:|---:|---:|---:|---:|---|
| before (`outputs/fp8-attn`) | `336a227b59fb846a` | 39.640 | 235.419 | 39.24 | 49.202 | 359 | `cffb3fb9...` |
| after, templated | `336a227b59fb846a` | 39.311 | 237.272 | 39.55 | 48.347 | 359 | `cffb3fb9...` |
| after, shipping binary | `336a227b59fb846a` | 45.732 | 237.560 | 39.59 | 48.581 | 367 | **`cffb3fb9...`** |

All three clips are **byte-identical** -
`cffb3fb9bed0cfe5b34e09531f395945` - through prompt encode, six denoise steps,
both VAEs and the mux. The denoise delta, +1.85 s and +2.14 s over six steps, is
0.8-0.9% and inside the spread section 2 measures on this host; the wall's 367 s
on the third run is the `LOAD` column again (39.3 -> 45.7 s, page cache, section
2 of the fp8 record), not the arm. `tests/h3_engine_gate` also passes on the
default arm (residency, `RESULT PASS`), which is the tensor-core path the
scalar-pinning `h3_dit_gate` never covered.

## 6. What this rules out, and what is left

- **Rows per CTA is spent.** Twelve warps cuts the K/V a row re-reads by 33%
  (each CTA reads S x 512 B for 192 rows instead of 128) and gains nothing,
  while paying a percent in registers. The ring is already hiding those loads
  behind the tile's math: the critical path is the per-row work - the k16 MMAs
  and the softmax over 64-key tiles - not the bytes.
- **Sixteen warps is a spill experiment, not a width experiment.** 268 B of
  spill stores per thread cost 30%, an order of magnitude more than the traffic
  could ever return.
- **A second CTA is a register experiment, and it has been run.** `<8, 2>` is
  the same 128-register body that loses 30%; the 32-key tile would buy the
  shared memory for it and change nothing about the ceiling.
- **The lever left is arithmetic per row**, in the same shape as the FP8 GEMM
  arm: E4M3 K/V with the block-scale MMA would cut the same per-row work that
  twelve warps failed to cut. That is a numerics decision with a quality gate,
  not an occupancy decision, and it belongs in its own unit.

## 7. Reproducing

```
make -j8 CUDA_ARCH=sm_121 tests/h3/attn_bm tests/h3_engine_gate
for arm in four pipe pipe12 pipe16; do
    H3_DIT_ATTENTION=$arm tests/h3/attn_bm 37722 56 10
    H3_DIT_ATTENTION=$arm tests/h3/attn_bm 38900 56 10
done
```

The register report is a compile-only pass with `--ptxas-options=-v`; the
two-CTA line comes from `~/ds4-test/pipe-two-cta.cu`, which instantiates
`<8, 2>` and is never launched. The delivery render is
`~/ds4-test/render-widths.sh`, which rebuilds `h3-rs` first so the binary
carries the same object as the benchmark.
