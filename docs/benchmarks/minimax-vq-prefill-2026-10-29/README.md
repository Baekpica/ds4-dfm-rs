# MiniMax VQ October 29 receipts

See the [campaign report](../../minimax-vq-optimization-2026-10-29.md).
P1, P2 and P3 are new Prefill adoptions; this folder does not count historical rounds.

- `p1-performance.json`, six `p1-{off,on}*.csv`: fresh 8K/64K suffix
  Prefill and raw greedy Decode samples, including all negative results.
- `p1-fixed-decode.json`: six fresh matched teacher-forced Decode runs.
- `p1-quality.json`, `quality-noise.json`, `quality-p1.json`: calibrated
  numerical costs, including the unadopted single-component candidate.
- `p1-profiles.json`: equal-chunk attribution and Decode interval unions.
- `p1-conditions.json`, `conditions.json`, `*-clocks.csv`: commands,
  binaries, source-patch hashes, affinity, clocks and guard observations.
- `p1-proofs.json`: hashes of full frontier logits and generated IDs.
- `p1-retained-instructions.json`: retained instruction-stream comparison.
- `quality-seeds.json`, `make-quality-cases.py`: compact source tokens and
  reconstruction of the measured long-prefix corpus.
- `p2-performance.json`, six `p2f-{off,on}*.csv`: P2's fresh pairs.
- `p2-kernel.json`, `p2-profiles.json`: isolated counters, native checks
  and actual-model attribution; the anomalous cold scalar replay is retained.
- `p2-conditions.json`, `p2-proofs.json`, `p2-retained-instructions.json`:
  measured identities, full-output hashes and unchanged instruction streams.
- `p2-rejected-converter.json`: the earlier unadopted converter-only cohort.
- `p3-performance.json`, six `p3-{off,on}*.csv`: P3's fresh pairs.
- `p3-kernel.json`, `p3-profiles.json`, `p3-routes.json`: conversion/mapping
  screens, actual-model attribution and captured block-list identities.
- `p3-conditions.json`, `p3-proofs.json`, `p3-retained-instructions.json`:
  source/binary identities, output hashes and retained instruction streams.
- `card-startup-failures.json`: two invalid final-sweep startup attempts,
  with zero measured frontiers, kernel allocation errors and recovery state.
- `card-performance.json`, `card-summary.json`, `card-conditions.json`:
  six completed fresh sweeps, full-output hashes, raw rates, cycle records,
  repeated-ID differences, commands, clocks and memory/PSI observations.
- `card/{off,on}/dspark3-*.csv`, `*-clocks.txt`: the six raw sweep CSVs and
  clock CSVs. `card/on/throughput.png` shows both settings' observed ranges.
- `card-continuations.json`: decoded 8K/64K examples from both settings.
  These repetitive forced continuations do not qualify Decode speedup.

Large logits, captured references, Nsight reports, guard logs, dirty patches,
failed trials and build logs remain under `scratch/minimax-m31-20261029/`.
They are not bundled as model artifacts. Per-command binary hashes distinguish
the later timing-only quality-helper rebuild from the original calibration.

Build the changed targets with the live `/usr/local/cuda` toolchain:

```sh
make -j6 CUDA_ARCH=sm_121 ds4-bench-perf tests/test_minimax_rows tests/minimax_quality
DS4_MINIMAX_SCORE_FP4=0 tests/test_minimax_rows score
DS4_MINIMAX_SCORE_FP4=1 tests/test_minimax_rows score
DS4_MINIMAX_SCORE_FP4=1 DS4_MINIMAX_SCORE_PREP=0 tests/test_minimax_rows score
DS4_MINIMAX_SCORE_FP4=1 DS4_MINIMAX_SCORE_PREP=1 tests/test_minimax_rows score
DS4_MINIMAX_ATTN_PACK=0 tests/test_minimax_rows attention
DS4_MINIMAX_ATTN_PACK=1 tests/test_minimax_rows attention
```

`make-quality-cases.py MODEL PROMPT OUTPUT` needs the artifact tokenizer and
Python `tokenizers`. The generated cases feed
`tests/minimax_quality --record MODEL CASES REFERENCE 2048 131072` under
FP4=0, then ordinary comparison mode under FP4=0/ATTN_SIMT=1 and FP4=1
with ATTN_SIMT unset. Preserve the guard, affinity and controls in
`p1-conditions.json` when reproducing resident runs; do not run them together.
Fixed Decode uses captured cases 0 and 4 and computes
`256 / decode_forward_s`, separately from total quality-check wall time.

The quality example's capture unit test was run with:
`cargo test -j2 -p ds4-cli --example minimax_quality --release --features native`.
No full workspace, HTTP, media or full-model 1M qualification is claimed.

The final curve uses all three switches (`DS4_MINIMAX_SCORE_FP4`,
`DS4_MINIMAX_SCORE_PREP`, `DS4_MINIMAX_ATTN_PACK`) set to 0 versus 1.
It preserves the card protocol: 524288 allocation, 2048 increments/cap,
128 forced output tokens, DSpark draft 3, no extra warmup. Exact commands
and controls are in `card-conditions.json`. Repeat equality applies to
Prefill logits; generated-ID variation from the forced-EOS/timing-gate
path is retained in `card-performance.json` and explained in the report.

Regenerate the plotted medians and both min–max bands with the existing
plotter and a Python environment containing matplotlib:

```sh
DATA=docs/benchmarks/minimax-vq-prefill-2026-10-29/card
uv run --offline --no-project --with matplotlib python \
  docs/benchmarks/plot-minimax-vq-64k.py "$DATA/on" "$DATA/off" 'All three off'
```

The render was inspected and its values checked against `card-summary.json`.
