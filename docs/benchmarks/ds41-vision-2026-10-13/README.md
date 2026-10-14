# DeepSeek V4.1 Flash: the vision tower golden gate (engine and port)

The ds41 vision tower is ported from the YoungAi engine; its correctness gate
is the engine's own: `tools/vision_golden.py` renders a fixture with the
official `inference/{image_processor,vision}.py` code and the BF16 originals
(PIL preprocessing plus the aligner rows), and both engines compare their
preprocessing and their tower against it. This record carries both sides: the
engine's gate (`./ds4 --image ... --vision-golden ...`) and the port's
(`./ds4 --vision-gguf ... --image ... --vision-golden ...`).

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

    ./ds4 --cuda -m <base>.gguf --zchain <zchain dir> --mem-budget-mb 110000 \
        --no-dspark --vision-gguf <vision.gguf> --image <img> \
        --vision-golden <golden>.bin

Fixture generator (this repo, `tools/vision_golden.py`: the port's copy of
the engine's tool, translated to English; fixture generation only):

    python3 tools/vision_golden.py --hf <hf dir> --image <img> --out <golden>.bin

Gate runner (this repo, `tests/ds41_vision_gate_spark.sh`: the port of the
engine's `gguf-tools/scripts/vision_gate_spark.sh`; golden gate, text
invariance, see-an-image through the port's server, no leftovers):

    bash tests/ds41_vision_gate_spark.sh [main gguf] [vision gguf] [zchain dir]

## Fixtures

| Fixture | Source | Golden | State |
| --- | --- | --- | --- |
| review-market-panel.png | engine tree root; also this repo, `tests/fixtures/vision/` | `tests/fixtures/vision/golden_panel.bin` (Spark-minted) | run 2026-10-13: ok |
| bug.jpeg | engine tree root; also this repo, `tests/fixtures/vision/` | `tests/fixtures/vision/golden_bug.bin` (Spark-minted) | run 2026-10-13: ok; the engine's own gate used img.jpeg (its author's image, not in the engine repo), this repo's jpeg fixture is the engine tree's bug.jpeg |
| homer.png (190x390) | owner-provided, `tests/fixtures/vision/homer.png` | `tests/fixtures/vision/golden_homer.bin` (Spark-minted) | run 2026-10-13: ok; plan 392x784 pad, 56x28 patch, 19x10 token, span 211 |

## Engine side (quoted from the engine's own record, 2026-10-13)

The engine's own gate ran on the engine author's machine; its artifacts
(`gguf/v41/vision/gate_1013_1113` and `gate_1013_1134`) live in the engine
tree, not in this repo. The engine's record (fable5.md, the 10-13 vision
entry) states:

- panel PNG: size account identical (1204x672 / 48x86 / 16x29); patches
  0/2,427,264 elements differ (its PIL transcription is bit-exact); tower
  (golden patches in, its aligner rows out) row cos min 0.990, max diff
  0.0195; after the per-block weight-copy change: mean 0.999804 / min 0.990
  (row 97).
- img.jpeg: size account identical; patches 4% of elements differ by at most
  3.2 levels (stb vs libjpeg IDCT/chroma upsampling, not a transcription
  error); tower row cos min 0.973 (row 747) / mean 0.999696; end-to-end (its
  own patches) row cos min 0.787 (row 584, JPEG decode difference; report
  only).
- Threshold basis from the same record: the official code's own bf16 vs f32
  on img.jpeg, row cos min 0.926 (row 746) / mean 0.99927; panel 0.986 /
  0.99979.
- The port's own run of the same comparison (this repo's generator, torch
  2.14.1+cu130, RTX 4070 SUPER, 2026-10-13): panel bf16 vs f32 row cos min
  0.9688 (row 389) / mean 0.99977, patch max diff 0.0.
- Text invariance gate: byte-identical with and without `--vision-gguf`
  (temp 0, 64 tokens); the see-an-image answers matched the pictures.

Our Spark cannot re-run the engine side: its engine tree sits at fc54d1e
(before the eight vision commits) and no vision-capable engine binary
survives in the model/bin backups (all three checked 2026-10-13).

## Port side, verified on the dev box (2026-10-13)

- Gate code: `crates/ds4-core/src/v41_golden.rs` (VGLD reader, the five
  checks, the two thresholds) with the native tower entry
  (`ds4_bridge_v41_vision_run`, `ds4_bridge_v41_has_vision`) and the CLI
  flags (`--vision-gguf`, `--image`, `--vision-golden`).
- The window's own files are in this tree: the fixtures under
  `tests/fixtures/vision/`, the golden generator `tools/vision_golden.py`
  (the engine's tool, translated) and the gate runner
  `tests/ds41_vision_gate_spark.sh` (the engine's script, ported to the
  port's entries: `--gen-ids` for the text step, the server for the
  see-an-image step).
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

## Port side, verified on the Spark (2026-10-13)

The window ran on the reference Spark (GB10) against the deployed artifacts
(base `vq8sh14-q4k-mtpnative`, the `grrb-code_fit_n15360-engine` directory,
the n-gram shards, sidecar `DeepSeek-V4.1-Flash.vision.gguf` 971,066,016 B,
sha256 `b50974c405aafbd941f8e8767cf90231b522aa4f2fcb506eaea68fbf8b409af5`).
Build `make cuda-spark` at `6c959f82`: `./ds4` sha256 `0d141bae...`, `./ds4-server`
sha256 `cf7f5af0...`. The goldens were minted on that same machine (torch
2.14.1+cu130 on the GB10, `tools/vision_golden.py`), so plan, patches and
tower compare on one device, the engine's own convention. Runner
`bash tests/ds41_vision_gate_spark.sh`; raw outputs under the gate dir it
prints.

| Fixture | plan | pixels | patches | tower (golden patches) | verdict |
| --- | --- | --- | --- | --- | --- |
| review-market-panel.png | 1200x665 -> pad 1204x672, patch 48x86, token 16x29 (matches) | 0/2,427,264 differ | 0/2,427,264 differ, max 0.0 | 464x5120, row cos mean 0.999804 min 0.990113 (row 97), max abs diff 0.01953 | ok |
| bug.jpeg | 1080x2414 -> pad 840x1876, patch 134x60, token 45x20 (matches) | 116,398/4,727,520 differ, max 4 | 116,398/4,727,520 differ, max 0.03125 (4.0 pixel-levels) | 900x5120, mean 0.999881 min 0.993775 (row 159), max abs diff 0.02100 | ok |
| homer.png | 190x390 -> pad 392x784, patch 56x28, token 19x10 (matches) | 0/921,984 differ | 0/921,984 differ, max 0.0 | 190x5120, mean 0.999624 min 0.990381 (row 48), max abs diff 0.02051 | ok |

Both PNGs transcribe bit-exact; the JPEG moves 4.0 pixel-levels at its worst
element, the same class the engine's record gives its img.jpeg. The panel's
numbers land where the engine's own record puts its panel run (mean 0.999804,
min 0.990, row 97, max abs diff 0.0195).

Text invariance (step 2): byte-identical with and without `--vision-gguf`
(temp 0, 64 greedy tokens, the prompt tokenized once with `--dump-tokens`).
The answer as-is begins: "Marginal cost is the additional cost incurred by
producing one more unit of a product or service."

See an image (step 3, the port's server, one greedy request per fixture,
answers printed raw and judged against the pictures):

- review-market-panel.png (502 prompt tokens, 200 completion) lists the
  panel's own headings: `QTTerminal`, `A股交易系统`, `复盘分析`, `大盘数据 ·
  涨跌家数真实判断 · 资金曲线模拟 · 自选股胜率统计 · 偏差归因`, `时间筛选` /
  `全部时间`, matching the engine's recorded reading.
- homer.png (231 prompt, 59 completion): "Homer Simpson ... holding a pink
  donut with sprinkles in his right hand ... There is no text or numbers
  visible in this image." Checked against the fixture: correct.
- bug.jpeg (967 prompt, 200 completion): read as a dense vertical Chinese
  text column, which it is.

The 400 without the sidecar (step 5): HTTP 400, `{"error":{"message":"image
input requires the DeepSeek V4.1 vision sidecar (--vision-gguf)","type":
"invalid_request_error"}}`.

The image surfaces (live, one server, temperature 0): the same image request
through OpenAI `/v1/chat/completions`, Anthropic `/v1/messages` and Responses
`/v1/responses` (`input_image`) answered byte-identical, 231 prompt and 59
completion each (`tests/ds41_api_image_probe.py`: 3/3 surfaces, identical).

Admission (step 6, still open): a 200,000-token prefill was not refused --
the peak fits and the prefill ran on the free box, so the refusal line needs
a longer prompt or memory pressure. The cell stays open.

The dev-box-minted goldens (the run before this one) are kept as the
cross-device finding: the port's tower against them read min 0.942181 on the
panel (row 389, FAIL), while the two official-torch goldens (RTX 4070 SUPER
vs GB10, same code and weights) differ by min 0.938248 on that same row. The
dip is the official implementation's own cross-device bf16 noise, which is
why the engine mints its goldens on the machine that runs the tower.

## Limits

- The admission cell (step 6) is open: a 200,000-token prefill fits the free
  box and runs, so the refusal line has not been captured yet.
- The engine's gate ran on the engine author's machine; its outputs are
  quoted from the engine's record (above), not re-run here.
- The (2) pixel line is a named port addition (report only); the engine has
  no counterpart line, so the two sides' outputs differ there by design.
- The port's CLI one-shot image request is not ported (it answers "not
  ported yet"); the see-an-image step drives the port's server, which is the
  port's production image entry.
