# DeepSeek V4.1 Flash: the vision tower golden gate (engine and port)

The ds41 vision tower is ported from the YoungAi engine; its correctness gate
is the engine's own: `tools/vision_golden.py` renders a fixture with the
official `inference/{image_processor,vision}.py` code and the BF16 originals
(PIL preprocessing plus the aligner rows), and both engines compare their
preprocessing and their tower against it. This record carries both sides: the
engine's gate (`./ds4 --image ... --vision-golden ...`) and the port's
(`ds4-rs --vision-gguf ... --image ... --vision-golden ...`).

Fixture format (little endian, `src/core/core_v41_vision_gate.c:3-5`):

    "VGLD" u32 ver=1 | u32 w h n_vit_h n_vit_w n_llm_h n_llm_w pp E best_w best_h
    u8 rgb[best_h][best_w][3] (after PIL pad)
    f32 patches[n_vit][pp] | f32 embeds[n_llm][E]

The gate's checks and thresholds (engine `core_v41_vision_gate.c`):

| Check | What it proves | Verdict |
| --- | --- | --- |
| (1) size account | the plan (pad/patch/token grids) against the golden header | fail on mismatch |
| (2) pixels | decoder difference against the golden RGB (PIL-padded) | report only; the engine reads the section and discards it (`(void)grgb`, :76) |
| (3) patches | the production preprocessing against the golden patches | fail over 4 pixel-levels (:73; one level is 2/255) |
| (4) tower (golden patches) | the tower alone: aligner rows against the golden embeds | fail under mean row cosine 0.999 or min row cosine 0.95 (:42-43) |
| (5) end-to-end (our patches) | the whole path, decoder differences included | report only (:87-90) |

Threshold basis (:39-41): the official bf16 run against the same code in f32
has min row cosine 0.926 / mean 0.99927 on `img.jpeg`, so the gate judges mean
and minimum together; the minimum alone would fail the official run against
itself.

## Commands

Engine (on the Spark tree, `gguf-tools/scripts/vision_gate_spark.sh` step 1):

    ./ds4 --cuda -m <base>.gguf --zchain <domain dir> --mem-budget-mb 110000 \
        --no-dspark --vision-gguf <vision.gguf> --image <img> \
        --vision-golden gguf/v41/vision/golden_<name>.bin

Port (this tree; the gate runs instead of generating):

    ds4-rs --cuda -m <base>.gguf --zchain <domain dir> --mem-budget-mb 110000 \
        --no-dspark --vision-gguf <vision.gguf> --image <img> \
        --vision-golden <golden>.bin

Fixture generator (engine tree, outside this repo by rule 14):

    ~/v41env/bin/python tools/vision_golden.py --hf <hf dir> --image <img> \
        --out <golden>.bin

## Fixtures

| Fixture | Source | Golden | State |
| --- | --- | --- | --- |
| review-market-panel.png | engine gate dir, on the Spark | `gguf/v41/vision/golden_panel.bin` (engine tree) | not run yet (Spark window) |
| img.jpeg | engine gate dir, on the Spark | `gguf/v41/vision/golden_img.bin` (engine tree) | not run yet (Spark window) |
| homer.png (190x390) | owner-provided, `misc/vision-test/homer.png` | none yet; mint with the generator | end-to-end picture (5); its plan is 392x784 pad, 56x28 patch, 19x10 token, span 211 (probed against `v41_media::plan` 2026-10-13) |

## Port side, verified on the dev box (2026-10-13)

- Gate code: `crates/ds4-core/src/v41_golden.rs` (VGLD reader, the five
  checks, the two thresholds) with the native tower entry
  (`ds4_bridge_v41_vision_run`, `ds4_bridge_v41_has_vision`) and the CLI
  flags (`--vision-gguf`, `--image`, `--vision-golden`).
- Unit evidence: `cargo test -p ds4-core --lib v41_golden` 4 pass (reader
  round-trip of the engine format, bad magic/version/short reads, the row
  statistics formula, single-row negation); the full
  `cargo test -p ds4-core --lib` run is 394 pass / 0 fail; the CLI parse test
  `v41_vision_gate_flags_parse` passes. The `ssd::tests` and `bench::tests`
  parallel-test env flakes seen in some full runs reproduce at the parent
  commit with this work stashed (pre-existing, not from this unit).
- Native gates at this commit: `make cuda-generic` exit 0; the ds41 generate
  boundary gate PASS (0/36); `tests/test_ds41_bridge_lifetime` pass;
  `tests/parity/bridge_null_oracle` ok (both stub the new tower entries).
- The gate itself needs the model and runs only on the Spark (owner rule
  2026-10-10); everything above is model-free evidence.

## Spark window (pending, owner-gated)

To fill, on the Spark tree, without disturbing the live server:

1. The engine side: its own gate outputs from
   `gguf/v41/vision/gate_<time>/` (`golden_panel.txt`, `golden_img.txt`) and
   the two golden fixtures' hashes.
2. The port side: the same two fixtures through `ds4-rs`, plus `homer.png`
   end-to-end (a golden minted with the generator, or the (5) line only).
3. The sidecar sha256 (`DeepSeek-V4.1-Flash.vision.gguf`, 971,066,016 B) and
   the build shas of both sides (base GGUF sha256 already recorded in
   `docs/deepseek41-flash.md`).
4. Acceptance list: the port gate exits 0 on both fixtures; text byte
   identity with and without `--vision-gguf`; the 400 path with the sidecar
   absent; the admission line when the peak does not fit.

## Limits

- The port gate has not been run against a model yet: this record's port
  section is unit evidence only until the Spark window fills the results.
- The engine's own gate run predates this record; its outputs live in the
  engine tree and are quoted as-is when the window opens.
- The (2) pixel line is a named port addition (report only); the engine has
  no counterpart line, so the two sides' outputs differ there by design.
