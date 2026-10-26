# H3 performance records

`tests/h3_perf_sweep.sh` writes one CSV row per render and keeps the run's
MP4 and engine log beside it. Everything stays in this repository.

    tests/h3/perf/perf.csv          one header, one row per run (tracked)
    tests/h3/perf/runs/*.mp4        the run's output (ignored by git)
    tests/h3/perf/runs/*.log        the engine's own stdout (ignored by git)
    tests/h3/artifacts/             delivered clips and frame sheets (ignored)

## Run one

```sh
# the port
tests/h3_perf_sweep.sh --engine port --width 512 --height 512 \
  --frames 56 --steps 8 --seed 7 --anchor tests/h3/artifacts/keyframe-crocodile.jpg \
  --label port-512

# the reference engine, same shape and seed
H3_REFERENCE=~/h3-oracle/h3 tests/h3_perf_sweep.sh --engine reference \
  --width 512 --height 512 --frames 56 --steps 8 --seed 7 \
  --anchor tests/h3/artifacts/keyframe-crocodile.jpg --label ref-512
```

The script refuses to run without an executable and a checkpoint, writes the
header once, appends nothing on a failed run, and prints the kept path.

## Columns

`ts_utc, engine, label, profile, binary_sha256, git_rev, dirty_files, driver,
clock_before, clock_after, width, height, frames, steps, layers, seed, text_s,
load_s, denoise_s, per_step_s, audio_s, vae_s, mux_s, wall_s, peak_rss_kb,
output_bytes, ckpt_bytes, cache_before_kb, cache_after_kb, prompt_sha256,
anchor, anchor_sha256, cuda_arch`

- `profile` is 1 when the run emitted phase marks. For the reference that
  means `--profile` plus `H3_PROFILE=1`, and those marks synchronize the
  stream (h3_gpu_cuda.cu:658-669), so a profile row and a timing row are not
  the same measurement. Compare `wall_s` only between rows with equal
  `profile`.

- `peak_rss_kb` is empty by construction: the child is reaped before the row
  is written, so its high-water mark is gone. The column stays so the schema
  does not move.
- `label` is sanitized to `[A-Za-z0-9_.-]`, so it can never shift a column.
  `anchor` is a path and is quoted when it contains a comma, which makes that
  one line standard CSV rather than plain comma-separated text.
- Phase cells come from the engine's own lines (`TEXT`, `LOAD`, `DENOISE`,
  `AUDIO`, `FRAMES`, `VIDEO`) and are empty when a line is absent, never zero.
- `denoise_s / steps` is `per_step_s`; compare steps, not totals, across
  shapes.
- `wall_s` is the process wall clock, so it includes load and mux.
- The identity columns exist so a row can be trusted later: same binary hash,
  same checkpoint bytes, same prompt and anchor, and preferably the same
  driver and clock.
- `anchor_sha256` is the anchor's bytes, not its path: two rows with the same
  path are the same workload only if the file has not moved under them.
  Empty when the run carried no anchor.
- `cuda_arch` is the arch the h3 objects were built with, read from the DiT
  object's own packed target strings (`native/h3/h3_dit_gpu.o`, which is what
  the binary in that tree links) and, failing that, from the tree's
  `.ds4-cuda-config.mk`; `unknown` when neither is readable. It exists
  because the 2026-10-26 session could not recover it after the fact: an
  object recompiled with no arch carries compute_75 PTX that the driver JITs
  on the device, which is a different numeric profile, and it was measured as
  a variant of an arch-native build. The object's identity is the Makefile's
  `H3_DIT_LINK_FINGERPRINT`, so a row whose `binary_sha256` came from another
  tree has a pointer here, not a proof.
- The three rows recorded before 2026-10-26 carry `anchor_sha256` computed
  from the anchor file afterwards (its mtime, 19:16, precedes all three runs)
  and an empty `cuda_arch`: that build's arch was never recorded, which is
  the gap the column closes.

## Comparing

Two engines at one shape are a fair A/B only when `width, height, frames,
steps, seed, anchor` and `prompt_sha256` match. Anything else is a different
workload, and the row says so. `tests/h3_perf_sweep.sh --dry-run` prints the
exact command before it runs.

## Artifacts

The delivered clips under `tests/h3/artifacts/` are local evidence and are
ignored by git, as are the run MP4s. The CSV is the durable record; it is
small and safe to commit.
