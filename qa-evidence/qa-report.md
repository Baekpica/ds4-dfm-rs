Rule 19 QA report - MiniMax-H3 (h3) P13 close

repository : /data/ds4-dfm-rs
branch     : feature/minimax-h3
HEAD       : 02ba4b674a3a857b19163ec75654bb55c90e1d76
             (the brief named 7ed11690; one docs-only commit landed during the
             first QA pass and the tree has since been edited, uncommitted, with
             the fixes this report verifies: openai-videos.md,
             tests/fixtures/h3/README.md, post-port-phases.md, port-plan.md,
             render.rs and vae_video.rs. The 559-surface list is identical
             before and after those edits.)
base       : origin/main = 02a6b5be83231f847d89433a90a48334bdfce247
date       : 2026-10-25
qa model   : deepseek-v4.1-CC-flash
host       : x86_64, nvcc 13.3.73, RTX 4070 SUPER. The h3 execution target is
             the DGX Spark (GB10, sm_121, 192.168.1.91) with the checkpoint
             under /mnt/h3-artifacts; nothing that needs the checkpoint ran
             here (ACCEPTED RISKS).

Surfaces required by tests/qa-gate.sh (QA_BASE=origin/main): 559
= 186 changed files + 2 Makefile targets (h3-rs, h3-server-rs) + 371 pub items.
Every one is listed with its disposition in COVERED SURFACES below.

This is the second pass. The first pass (same session) reported FAIL on two
documentation defects (D1, D2) and two worth-fixing observations (O1, O5). All
four have been corrected in the working tree, and section S6 records the
evidence each correction was checked against. Everything else in this report
was re-run or re-read against the corrected tree; the checks whose inputs did
not change (native objects, VQ toolchain, both tools, fixtures, the Spark-only
items) are marked as carried over from this session's runs.


COVERED SURFACES

One line per surface string the gate printed. FILE lines carry a changed path or
a Makefile target; ITEM lines carry a Rust pub item the branch added, with the
definition site it was found at. The trailing clause names the check that covers
the surface. To keep this honest: the h3 Rust sources, the native objects, the
VQ tree and the two reviewed tools were executed in the live checks above; files
I did not read line by line are covered by presence plus that build or test, and
those lines say so. Several items share a tail because one run covers them all
(the clippy/build/646-test run for ds4-core, the 24-test run for ds4-server).
FILE .gitignore - read: adds /tools/minimax_h3/vq/bin/, /tools/minimax_h3/ppm_diff and /outputs/; git check-ignore confirms vq/bin and ppm_diff are ignored
FILE AGENTS.md - read: +40 lines, the additive h3 family section only; no other family's text is touched
FILE Makefile - read: +126 lines, the h3 object rules, the six tests/h3_* gates plus h3-rs/h3-server-rs; the four .cu objects were compiled here with the same flags (exit 0)
FILE crates/ds4-cli/Cargo.toml - read: bin h3-rs with required-features = ["native"] (Cargo.toml:28-30)
FILE crates/ds4-cli/src/bin/h3-rs.rs - read in full for the flag table; compiled and linked (exit 0); --help exit 0; flag diff against /data/h3.c/main.c recorded in STATIC CHECKS
FILE crates/ds4-core/Cargo.toml - read: feature h3-gates (Cargo.toml:19-20) and the six example gates it gates
FILE crates/ds4-core/examples/h3_device_gate.rs - compiled: cargo check --examples -p ds4-core --features h3-gates exit 0; a full build of h3_dit_gate compiles and fails only at link on h3d_*/h3t_* (no native objects on this host), as expected
FILE crates/ds4-core/examples/h3_dit_gate.rs - compiled: cargo check --examples -p ds4-core --features h3-gates exit 0; a full build of h3_dit_gate compiles and fails only at link on h3d_*/h3t_* (no native objects on this host), as expected
FILE crates/ds4-core/examples/h3_multimodal_gate.rs - compiled: cargo check --examples -p ds4-core --features h3-gates exit 0; a full build of h3_dit_gate compiles and fails only at link on h3d_*/h3t_* (no native objects on this host), as expected
FILE crates/ds4-core/examples/h3_text_gate.rs - compiled: cargo check --examples -p ds4-core --features h3-gates exit 0; a full build of h3_dit_gate compiles and fails only at link on h3d_*/h3t_* (no native objects on this host), as expected
FILE crates/ds4-core/examples/h3_video_encoder_gate.rs - compiled: cargo check --examples -p ds4-core --features h3-gates exit 0; a full build of h3_dit_gate compiles and fails only at link on h3d_*/h3t_* (no native objects on this host), as expected
FILE crates/ds4-core/examples/h3_vision_gate.rs - compiled: cargo check --examples -p ds4-core --features h3-gates exit 0; a full build of h3_dit_gate compiles and fails only at link on h3d_*/h3t_* (no native objects on this host), as expected
FILE crates/ds4-core/examples/minimax_h3_inspect/main.rs - compiled: cargo check --examples -p ds4-core --features h3-gates exit 0; a full build of h3_dit_gate compiles and fails only at link on h3d_*/h3t_* (no native objects on this host), as expected
FILE crates/ds4-core/src/lib.rs - read: additive module declarations only; compiled (clippy exit 0)
FILE crates/ds4-core/src/minimax_h3.rs - read: the module header and layout rule; compiled (clippy exit 0), gated by the h3 tests
FILE crates/ds4-core/src/minimax_h3/condition.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/device.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/dit.rs - read: row_map and its callers; the presentation-tag port is line-for-line against h3_dit_schedule.c:434-484 (STATIC CHECKS); gated by row_map_tags_by_segment_kind and row_map_carries_the_presentation_tags
FILE crates/ds4-core/src/minimax_h3/engine.rs - read: the anchor read path (media::read_image_f32 on a server-generated path); the engine is the served renderer
FILE crates/ds4-core/src/minimax_h3/host.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/layout.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/media.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/multimodal.rs - read: build_tags against h3_multimodal.c:173, :182; gated by the presentation tests
FILE crates/ds4-core/src/minimax_h3/presets.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/render.rs - read; the clippy fix at :633 (needless mut on audio_write) verified: clippy now reports 0 warnings from h3 files (was 2), mux_mp4 is untouched otherwise, cargo test -p ds4-core --lib 646 passed / 7 ignored
FILE crates/ds4-core/src/minimax_h3/safetensors.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/sequence.rs - read in full: build/emit geometry and the named Ref2VA deviation; compiled, tested
FILE crates/ds4-core/src/minimax_h3/terminal.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/text.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/text_forward.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/text_rope.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/text_weights.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/tokenizer.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/vae_audio.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/vae_encoder.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/vae_store.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/vae_video.rs - read; the clippy fix at :362 (tile.len().is_multiple_of(frame) under the same frame > 0 guard) verified: 0 h3 clippy warnings, tests unchanged (646 passed / 7 ignored)
FILE crates/ds4-core/src/minimax_h3/video_api.rs - read: parse_request, validate, decode_reference (data-URI only), store ids and Video serialization; gated by the video_api and h3_video tests
FILE crates/ds4-core/src/minimax_h3/vision.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/vision_forward.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/vision_weights.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/src/minimax_h3/weights.rs - compiled (clippy -p ds4-core --lib --features h3-gates exit 0); gated by cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE crates/ds4-core/tests/fixtures/h3/tokenizer_portable.json - valid JSON (jq); read by the tokenizer tests
FILE crates/ds4-server/Cargo.toml - read: additive bin entry and the native feature
FILE crates/ds4-server/src/bin/h3-server-rs.rs - read; built and run: --help exit 0, GET /v1/videos 200 from a live listener
FILE crates/ds4-server/src/h3_engine.rs - read: EngineRenderer and the RenderRequest it builds; the anchor path is server-generated
FILE crates/ds4-server/src/h3_video.rs - read: routes, renderer bridge, anchor file, store paths, into_image_ref; 24 h3_ tests pass and the live route and anchor matrices (checks 8 and 18) match the docs
FILE crates/ds4-server/src/lib.rs - read: one additive module line
FILE crates/ds4-sys/src/h3_device.rs - compiled; its extern declarations resolved at link against the four objects compiled here (70 h3d_/h3t_/h3v_/h3g_ symbols in target/debug/h3-rs)
FILE crates/ds4-sys/src/h3_dit.rs - compiled; its extern declarations resolved at link against the four objects compiled here (70 h3d_/h3t_/h3v_/h3g_ symbols in target/debug/h3-rs)
FILE crates/ds4-sys/src/h3_text.rs - compiled; its extern declarations resolved at link against the four objects compiled here (70 h3d_/h3t_/h3v_/h3g_ symbols in target/debug/h3-rs)
FILE crates/ds4-sys/src/h3_vision.rs - compiled; its extern declarations resolved at link against the four objects compiled here (70 h3d_/h3t_/h3v_/h3g_ symbols in target/debug/h3-rs)
FILE crates/ds4-sys/src/lib.rs - compiled; its extern declarations resolved at link against the four objects compiled here (70 h3d_/h3t_/h3v_/h3g_ symbols in target/debug/h3-rs)
FILE docs/README.md - read: index rows for the h3 documents; the two newest rows are an uncommitted concurrent edit, no duplication
FILE docs/benchmarks/ds41-spark-head-2026-10-14/frontier-run-1.csv.meta.txt - read: path rewrites only (/home/leandro -> ~), the recorded values are unchanged
FILE docs/benchmarks/ds41-spark-vision-2026-10-14/vision-run-1.csv.meta.txt - read: path rewrites only (/home/leandro -> ~), the recorded values are unchanged
FILE docs/benchmarks/ds41-spark-vision-2026-10-14/vision-run-2.csv.meta.txt - read: path rewrites only (/home/leandro -> ~), the recorded values are unchanged
FILE docs/benchmarks/ds41-spark-vision-2026-10-14/vision-run-3.csv.meta.txt - read: path rewrites only (/home/leandro -> ~), the recorded values are unchanged
FILE docs/minimax-h3/README.md - read: the family index; its rows name every h3 surface in this list
FILE docs/minimax-h3/api-usage.md - read: the served contract (input_reference, anchors, limits); its anchor description is what the fixed openai-videos.md now matches
FILE docs/minimax-h3/cheatsheet.md - read: the measured numbers ([M]/[A]/[E]) and the layout summary; no executable surface
FILE docs/minimax-h3/openai-videos.md - read in full after the fix: routes/methods still match, and every input_reference claim now matches decode_reference/data_uri/check_anchor (video_api.rs:526-599, MAX_ANCHOR_BYTES :47, extension() :315-323) and the live POST matrix of check 18; D1 closed
FILE docs/minimax-h3/port-plan.md - read: the phase plan, the Spark budget and 6.3 the CLI surface target; its --use-int8-row-fc2 "drop it and say so" matches the code (O1), and its deviations paragraph at :1038-1045 now matches what is implemented (S6.5)
FILE docs/minimax-h3/post-port-phases.md - read: the P9-P13 record; the unported-surface table at :1120-1140 is the disposition of the missing CLI flags, and the --use-int8-row-fc2 item at :632-638 now records the refusal (O1 closed)
FILE docs/minimax-h3/ref-media-tests.md - read: the conditioning matrix and the recorded probe gates; no executable surface
FILE docs/minimax-h3/upstream-m31-survey.md - read: the M3.1 transfer/reject decisions; no executable surface
FILE docs/minimax-h3/vq-recipe.md - read: the GVQ1 v1 recipe, its fallback and the bake checklist
FILE docs/qwen38-benchmark-reference.md - read: two path rewrites to ~, no benchmark number changed
FILE h3-rs - Makefile target (Makefile:507): the four h3 objects + cublasLt/cudart; reproduced by hand, link exit 0, --help exit 0
FILE h3-server-rs - Makefile target (Makefile:520): same objects; a feature-less cargo build also links and serves /v1/videos 200 (renderer disabled 503 without --model-dir/--renderer)
FILE native/h3/h3_device_gpu.cu - compiled here: nvcc 13.3 -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -c exit 0; the device gates need the Spark's checkpoint
FILE native/h3/h3_device_gpu.h - compiled against: the paired .cu builds with it; the Rust FFI link resolved its symbols
FILE native/h3/h3_dit_gpu.cu - compiled here: nvcc 13.3 -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -c exit 0; the device gates need the Spark's checkpoint
FILE native/h3/h3_dit_gpu.h - compiled against: the paired .cu builds with it; the Rust FFI link resolved its symbols
FILE native/h3/h3_text_gpu.cu - compiled here: nvcc 13.3 -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -c exit 0; the device gates need the Spark's checkpoint
FILE native/h3/h3_text_gpu.h - compiled against: the paired .cu builds with it; the Rust FFI link resolved its symbols
FILE native/h3/h3_vision_gpu.cu - compiled here: nvcc 13.3 -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -c exit 0; the device gates need the Spark's checkpoint
FILE native/h3/h3_vision_gpu.h - compiled against: the paired .cu builds with it; the Rust FFI link resolved its symbols
ITEM pub const AUDIO_LATENT_FPS - defined crates/ds4-core/src/minimax_h3/host.rs:23; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const AUDIO_SIGMA_SHIFT - defined crates/ds4-core/src/minimax_h3/host.rs:33; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const AUDIO_STREAMS - defined crates/ds4-core/src/minimax_h3/dit.rs:90; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const BLOCK_SLOTS - defined crates/ds4-core/src/minimax_h3/dit.rs:88; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const CANVAS_MULTIPLE - defined crates/ds4-core/src/minimax_h3/host.rs:25; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const CHANNELS - defined crates/ds4-core/src/minimax_h3/media.rs:32; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DEEPSTACKS - defined crates/ds4-core/src/minimax_h3/vision.rs:76; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DEEPSTACK_LAYERS - defined crates/ds4-core/src/minimax_h3/vision.rs:83; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DEFAULT_LIST_LIMIT - defined crates/ds4-core/src/minimax_h3/video_api.rs:53; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DEFAULT_MODEL - defined crates/ds4-core/src/minimax_h3/video_api.rs:82; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DEFAULT_SECONDS - defined crates/ds4-core/src/minimax_h3/video_api.rs:79; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DEFAULT_SIZE - defined crates/ds4-core/src/minimax_h3/video_api.rs:75; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DELETE_OBJECT - defined crates/ds4-core/src/minimax_h3/video_api.rs:39; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DIT_BLOCKS - defined crates/ds4-core/src/minimax_h3/dit.rs:105; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const DIT_CLASS - defined crates/ds4-core/src/minimax_h3/layout.rs:44; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const EMBEDDING - defined crates/ds4-core/src/minimax_h3/text.rs:69; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const EPSILON - defined crates/ds4-core/src/minimax_h3/dit.rs:98; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FINAL_SLOTS - defined crates/ds4-core/src/minimax_h3/dit.rs:86; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FNV_OFFSET - defined crates/ds4-core/src/minimax_h3/text_forward.rs:55; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FNV_PRIME - defined crates/ds4-core/src/minimax_h3/text_forward.rs:58; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FPS - defined crates/ds4-core/src/minimax_h3/host.rs:21; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FRAME_CHANNELS - defined crates/ds4-core/src/minimax_h3/vision_forward.rs:724; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FRAME_PER_TOKEN - defined crates/ds4-core/src/minimax_h3/host.rs:38; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FRAME_RATE - defined crates/ds4-core/src/minimax_h3/media.rs:41; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FRAME_RESCALE - defined crates/ds4-core/src/minimax_h3/host.rs:40; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const FRAME_STEP - defined crates/ds4-core/src/minimax_h3/media.rs:46; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const H3_ALIGN_BASE - defined crates/ds4-core/src/minimax_h3/video_api.rs:64; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const H3_ALIGN_STEP - defined crates/ds4-core/src/minimax_h3/video_api.rs:65; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const H3_CANVAS_MULTIPLE - defined crates/ds4-core/src/minimax_h3/video_api.rs:69; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const H3_FPS - defined crates/ds4-core/src/minimax_h3/video_api.rs:63; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const H3_MAX_FRAMES - defined crates/ds4-core/src/minimax_h3/video_api.rs:67; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const H3_MAX_PIXELS - defined crates/ds4-core/src/minimax_h3/video_api.rs:70; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const H3_MIN_FRAMES - defined crates/ds4-core/src/minimax_h3/video_api.rs:66; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const H3_MIN_GEN_FRAMES - defined crates/ds4-core/src/minimax_h3/video_api.rs:68; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const HEADS - defined crates/ds4-core/src/minimax_h3/vision.rs:40; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const HEAD_DIM - defined crates/ds4-core/src/minimax_h3/text.rs:54; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const HIDDEN - defined crates/ds4-core/src/minimax_h3/text.rs:42; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const HOP_LENGTH - defined crates/ds4-core/src/minimax_h3/vae_audio.rs:32; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const IMAGE_PAD - defined crates/ds4-core/src/minimax_h3/multimodal.rs:41; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const INTERMEDIATE - defined crates/ds4-core/src/minimax_h3/text.rs:45; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const KV_HEADS - defined crates/ds4-core/src/minimax_h3/text.rs:51; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const LANGUAGE_PREFIX - defined crates/ds4-core/src/minimax_h3/text.rs:66; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const LATENT_CHANNELS - defined crates/ds4-core/src/minimax_h3/vae_audio.rs:22; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const LAYERS - defined crates/ds4-core/src/minimax_h3/text.rs:39; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const LAYER_NORM_EPSILON - defined crates/ds4-core/src/minimax_h3/vision.rs:79; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const LENGTH_BYTES - defined crates/ds4-core/src/minimax_h3/safetensors.rs:32; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const LIST_OBJECT - defined crates/ds4-core/src/minimax_h3/video_api.rs:38; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_ALIGNED_FRAMES - defined crates/ds4-core/src/minimax_h3/render.rs:39; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_ANCHOR_BYTES - defined crates/ds4-core/src/minimax_h3/video_api.rs:47; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_DISTORTION - defined crates/ds4-core/src/minimax_h3/presets.rs:38; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_HEADER_BYTES - defined crates/ds4-core/src/minimax_h3/safetensors.rs:30; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_LIST_LIMIT - defined crates/ds4-core/src/minimax_h3/video_api.rs:52; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_PIXELS - defined crates/ds4-core/src/minimax_h3/host.rs:27; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_PROMPT_CHARS - defined crates/ds4-core/src/minimax_h3/video_api.rs:42; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_RANK - defined crates/ds4-core/src/minimax_h3/safetensors.rs:34; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_SECONDS - defined crates/ds4-core/src/minimax_h3/media.rs:38; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MAX_STEPS - defined crates/ds4-core/src/minimax_h3/host.rs:35; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MERGE - defined crates/ds4-core/src/minimax_h3/vision.rs:55; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MERGED_BLOCK - defined crates/ds4-core/src/minimax_h3/vision_forward.rs:727; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MERGE_DIM - defined crates/ds4-core/src/minimax_h3/vision.rs:64; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MERGE_ROWS - defined crates/ds4-core/src/minimax_h3/vision.rs:89; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MIN_EDGE - defined crates/ds4-core/src/minimax_h3/vision.rs:86; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MIN_FRAMES - defined crates/ds4-core/src/minimax_h3/media.rs:44; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MIN_GENERATION_FRAMES - defined crates/ds4-core/src/minimax_h3/render.rs:37; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MIN_LIST_LIMIT - defined crates/ds4-core/src/minimax_h3/video_api.rs:51; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MIN_SAMPLES - defined crates/ds4-core/src/minimax_h3/media.rs:35; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const MODALITIES - defined crates/ds4-core/src/minimax_h3/dit.rs:82; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const OUTPUT_WIDTH - defined crates/ds4-core/src/minimax_h3/vision.rs:73; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const PAD_TOKEN_ID - defined crates/ds4-core/src/minimax_h3/tokenizer.rs:49; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const PATCH - defined crates/ds4-core/src/minimax_h3/vision.rs:49; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const PATCH_DIM - defined crates/ds4-core/src/minimax_h3/vision.rs:61; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const POSITION_COUNT - defined crates/ds4-core/src/minimax_h3/vision.rs:70; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const POSITION_SIDE - defined crates/ds4-core/src/minimax_h3/vision.rs:67; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const QUERY_HEADS - defined crates/ds4-core/src/minimax_h3/text.rs:48; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const REFERENCE_BLOCKS - defined crates/ds4-core/src/minimax_h3/layout.rs:38; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const REFERENCE_TIME_INPUT - defined crates/ds4-core/src/minimax_h3/layout.rs:41; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const RMS_EPSILON - defined crates/ds4-core/src/minimax_h3/text.rs:63; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const ROPE_FREQS - defined crates/ds4-core/src/minimax_h3/dit.rs:94; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const ROPE_HALF - defined crates/ds4-core/src/minimax_h3/text_rope.rs:40; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const ROPE_THETA - defined crates/ds4-core/src/minimax_h3/text.rs:60; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const SAMPLE_RATE - defined crates/ds4-core/src/minimax_h3/vae_audio.rs:30; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const SLOTS - defined crates/ds4-core/src/minimax_h3/dit.rs:84; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const TEMPORAL_PATCH - defined crates/ds4-core/src/minimax_h3/vision.rs:52; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const THETA - defined crates/ds4-core/src/minimax_h3/text_rope.rs:43; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const TIER_SHORT_EDGE - defined crates/ds4-core/src/minimax_h3/presets.rs:34; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const TIME_INPUT - defined crates/ds4-core/src/minimax_h3/dit.rs:92; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const VAE_SPATIAL_RATIO - defined crates/ds4-core/src/minimax_h3/host.rs:29; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const VIDEO_OBJECT - defined crates/ds4-core/src/minimax_h3/video_api.rs:37; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const VIDEO_PAD - defined crates/ds4-core/src/minimax_h3/multimodal.rs:44; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const VIDEO_ROW_WIDTH - defined crates/ds4-core/src/minimax_h3/condition.rs:32; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const VIDEO_SIGMA_SHIFT - defined crates/ds4-core/src/minimax_h3/host.rs:31; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const VISION_END - defined crates/ds4-core/src/minimax_h3/multimodal.rs:38; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const VISION_START - defined crates/ds4-core/src/minimax_h3/multimodal.rs:35; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const VOCAB - defined crates/ds4-core/src/minimax_h3/text.rs:57; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub const WIRE_REFERENCE - defined crates/ds4-core/src/minimax_h3/video_api.rs:35; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum DitError - defined crates/ds4-core/src/minimax_h3/dit.rs:108; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum Dtype - defined crates/ds4-core/src/minimax_h3/safetensors.rs:38; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum EmptyPrompt - defined crates/ds4-core/src/minimax_h3/tokenizer.rs:66; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum Fit - defined crates/ds4-core/src/minimax_h3/media.rs:50; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum ForwardError - defined crates/ds4-core/src/minimax_h3/text_forward.rs:137; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum LayoutError - defined crates/ds4-core/src/minimax_h3/layout.rs:710; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum LoadError - defined crates/ds4-core/src/minimax_h3/weights.rs:208; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum LoadMode - defined crates/ds4-core/src/minimax_h3/dit.rs:176; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum MergerPart - defined crates/ds4-core/src/minimax_h3/vision.rs:114; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum OutputRule - defined crates/ds4-core/src/minimax_h3/render.rs:57; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum Part - defined crates/ds4-core/src/minimax_h3/text.rs:79; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum Presentation - defined crates/ds4-core/src/minimax_h3/engine.rs:45; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum PresentationError - defined crates/ds4-core/src/minimax_h3/multimodal.rs:154; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum Protocol - defined crates/ds4-core/src/minimax_h3/terminal.rs:25; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum Ratio - defined crates/ds4-core/src/minimax_h3/presets.rs:42; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum Reduction - defined crates/ds4-core/src/minimax_h3/dit.rs:650; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum Role - defined crates/ds4-core/src/minimax_h3/layout.rs:153; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum SegmentKind - defined crates/ds4-core/src/minimax_h3/sequence.rs:29; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum SortOrder - defined crates/ds4-core/src/minimax_h3/video_api.rs:769; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum StError - defined crates/ds4-core/src/minimax_h3/safetensors.rs:220; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum TextError - defined crates/ds4-core/src/minimax_h3/text.rs:478; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum TextLoadError - defined crates/ds4-core/src/minimax_h3/text_weights.rs:188; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum TokenizerError - defined crates/ds4-core/src/minimax_h3/tokenizer.rs:73; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum VideoStatus - defined crates/ds4-core/src/minimax_h3/video_api.rs:166; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum VisionError - defined crates/ds4-core/src/minimax_h3/vision.rs:607; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub enum VisionLoadError - defined crates/ds4-core/src/minimax_h3/vision_weights.rs:205; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn adaln_hashes - defined crates/ds4-core/src/minimax_h3/dit.rs:2658; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn align_frame_count - defined crates/ds4-core/src/minimax_h3/host.rs:44; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn align_frames - defined crates/ds4-core/src/minimax_h3/video_api.rs:85; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn anchor_embedding - defined crates/ds4-core/src/minimax_h3/condition.rs:59; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn anchor_rows - defined crates/ds4-core/src/minimax_h3/condition.rs:92; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn append_vision - defined crates/ds4-core/src/minimax_h3/multimodal.rs:306; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn as_str - defined crates/ds4-core/src/minimax_h3/video_api.rs:174; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn as_string - defined crates/ds4-core/src/minimax_h3/video_api.rs:235; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn attention_scale - defined crates/ds4-core/src/minimax_h3/text_forward.rs:177; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn audio_elements - defined crates/ds4-core/src/minimax_h3/dit.rs:1737; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn audio_total_rows - defined crates/ds4-core/src/minimax_h3/sequence.rs:78; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn augment_span - defined crates/ds4-core/src/minimax_h3/condition.rs:36; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn bad_request - defined crates/ds4-core/src/minimax_h3/video_api.rs:127; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn bf16_to_f32 - defined crates/ds4-core/src/minimax_h3/vision.rs:359; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn block_stride - defined crates/ds4-core/src/minimax_h3/layout.rs:308; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn build - defined crates/ds4-core/src/minimax_h3/layout.rs:225; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn build_multimodal - defined crates/ds4-core/src/minimax_h3/text_rope.rs:108; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn bytes - defined crates/ds4-core/src/minimax_h3/safetensors.rs:127; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn check_frames - defined crates/ds4-core/src/minimax_h3/render.rs:267; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn check_h3 - defined crates/ds4-core/src/minimax_h3/video_api.rs:240; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn complete - defined crates/ds4-core/src/minimax_h3/video_api.rs:875; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn config - defined crates/ds4-core/src/minimax_h3/layout.rs:292; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn cosines - defined crates/ds4-core/src/minimax_h3/text_rope.rs:92; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn decode - defined crates/ds4-core/src/minimax_h3/tokenizer.rs:214; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn deepstack - defined crates/ds4-core/src/minimax_h3/vision_forward.rs:66; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn deepstack_fnv - defined crates/ds4-core/src/minimax_h3/vision_forward.rs:76; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn delete - defined crates/ds4-core/src/minimax_h3/video_api.rs:978; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn denoise - defined crates/ds4-core/src/minimax_h3/dit.rs:1874; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn denoise_with - defined crates/ds4-core/src/minimax_h3/dit.rs:1887; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn detail - defined crates/ds4-core/src/minimax_h3/media.rs:65; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn detect - defined crates/ds4-core/src/minimax_h3/terminal.rs:32; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn display_dimensions - defined crates/ds4-core/src/minimax_h3/terminal.rs:69; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn display_rgb24 - defined crates/ds4-core/src/minimax_h3/terminal.rs:162; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn distortion - defined crates/ds4-core/src/minimax_h3/render.rs:235; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn elements - defined crates/ds4-core/src/minimax_h3/safetensors.rs:121; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn encode - defined crates/ds4-core/src/minimax_h3/tokenizer.rs:184; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn encode_dir - defined crates/ds4-core/src/minimax_h3/vision_forward.rs:711; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn encode_layers - defined crates/ds4-core/src/minimax_h3/text_forward.rs:189; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn encode_multimodal - defined crates/ds4-core/src/minimax_h3/text_forward.rs:205; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn encode_with - defined crates/ds4-core/src/minimax_h3/vision_forward.rs:120; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn end - defined crates/ds4-core/src/minimax_h3/multimodal.rs:62; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn end_for_step - defined crates/ds4-core/src/minimax_h3/dit.rs:676; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn entries - defined crates/ds4-core/src/minimax_h3/layout.rs:296; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn euler_step - defined crates/ds4-core/src/minimax_h3/host.rs:256; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn extension - defined crates/ds4-core/src/minimax_h3/video_api.rs:315; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn extrapolate_velocity - defined crates/ds4-core/src/minimax_h3/host.rs:232; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn extrapolation_ratio - defined crates/ds4-core/src/minimax_h3/host.rs:205; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn f1 - defined crates/ds4-core/src/minimax_h3/vae_store.rs:116; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn f2 - defined crates/ds4-core/src/minimax_h3/vae_store.rs:120; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn f3 - defined crates/ds4-core/src/minimax_h3/vae_store.rs:124; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn f32_to_bf16 - defined crates/ds4-core/src/minimax_h3/vision.rs:350; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn fail - defined crates/ds4-core/src/minimax_h3/video_api.rs:890; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn fill_normal - defined crates/ds4-core/src/minimax_h3/host.rs:140; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn find - defined crates/ds4-core/src/minimax_h3/safetensors.rs:143; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn first_words - defined crates/ds4-core/src/minimax_h3/text_forward.rs:113; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn fl2va - defined crates/ds4-core/src/minimax_h3/multimodal.rs:213; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn fnv - defined crates/ds4-core/src/minimax_h3/text_forward.rs:96; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn fnv1a - defined crates/ds4-core/src/minimax_h3/text_forward.rs:61; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn forward - defined crates/ds4-core/src/minimax_h3/dit.rs:1803; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn frames - defined crates/ds4-core/src/minimax_h3/render.rs:323; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn frames_from_seconds - defined crates/ds4-core/src/minimax_h3/video_api.rs:96; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn from_f32 - defined crates/ds4-core/src/minimax_h3/render.rs:296; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn from_store - defined crates/ds4-core/src/minimax_h3/layout.rs:276; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn get - defined crates/ds4-core/src/minimax_h3/video_api.rs:908; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn grid - defined crates/ds4-core/src/minimax_h3/vision_forward.rs:47; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn half - defined crates/ds4-core/src/minimax_h3/text_rope.rs:87; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn height - defined crates/ds4-core/src/minimax_h3/render.rs:212; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn ids - defined crates/ds4-core/src/minimax_h3/multimodal.rs:110; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn inner - defined crates/ds4-core/src/minimax_h3/layout.rs:138; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn interleave_pcm - defined crates/ds4-core/src/minimax_h3/render.rs:505; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn inventory - defined crates/ds4-core/src/minimax_h3/safetensors.rs:885; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn inverse_frequency - defined crates/ds4-core/src/minimax_h3/text_rope.rs:165; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn is_anamorphic - defined crates/ds4-core/src/minimax_h3/render.rs:241; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn is_empty - defined crates/ds4-core/src/minimax_h3/video_api.rs:929; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn is_scaled - defined crates/ds4-core/src/minimax_h3/presets.rs:142; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn kv_dim - defined crates/ds4-core/src/minimax_h3/text.rs:245; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn last_words - defined crates/ds4-core/src/minimax_h3/text_forward.rs:118; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn latent - defined crates/ds4-core/src/minimax_h3/render.rs:248; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn latent_canvas - defined crates/ds4-core/src/minimax_h3/host.rs:83; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn latent_normalization - defined crates/ds4-core/src/minimax_h3/vae_video.rs:929; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn latent_time - defined crates/ds4-core/src/minimax_h3/vae_encoder.rs:121; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn layout - defined crates/ds4-core/src/minimax_h3/weights.rs:91; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn len - defined crates/ds4-core/src/minimax_h3/video_api.rs:925; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn list - defined crates/ds4-core/src/minimax_h3/video_api.rs:933; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn load - defined crates/ds4-core/src/minimax_h3/tokenizer.rs:135; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn load_with - defined crates/ds4-core/src/minimax_h3/dit.rs:1523; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn mapped - defined crates/ds4-core/src/minimax_h3/weights.rs:135; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn max_jobs - defined crates/ds4-core/src/minimax_h3/video_api.rs:809; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn media_type_for_extension - defined crates/ds4-core/src/minimax_h3/video_api.rs:327; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn merged - defined crates/ds4-core/src/minimax_h3/vision.rs:403; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn message - defined crates/ds4-core/src/minimax_h3/engine.rs:109; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn model - defined crates/ds4-core/src/minimax_h3/video_api.rs:805; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn modulation_hashes - defined crates/ds4-core/src/minimax_h3/dit.rs:1748; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn mrope_axis - defined crates/ds4-core/src/minimax_h3/text_rope.rs:142; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn mux_args - defined crates/ds4-core/src/minimax_h3/render.rs:534; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn mux_mp4 - defined crates/ds4-core/src/minimax_h3/render.rs:596; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn name - defined crates/ds4-core/src/minimax_h3/safetensors.rs:76; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn names - defined crates/ds4-core/src/minimax_h3/presets.rs:279; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn new - defined crates/ds4-core/src/minimax_h3/video_api.rs:111; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn new_anamorphic - defined crates/ds4-core/src/minimax_h3/render.rs:104; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn new_delivery - defined crates/ds4-core/src/minimax_h3/render.rs:86; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn next_u32 - defined crates/ds4-core/src/minimax_h3/host.rs:114; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn normal - defined crates/ds4-core/src/minimax_h3/host.rs:125; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn not_found - defined crates/ds4-core/src/minimax_h3/video_api.rs:135; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn of - defined crates/ds4-core/src/minimax_h3/multimodal.rs:85; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn open - defined crates/ds4-core/src/minimax_h3/layout.rs:267; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn open_dir - defined crates/ds4-core/src/minimax_h3/safetensors.rs:868; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn output_path - defined crates/ds4-core/src/minimax_h3/video_api.rs:916; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn outside - defined crates/ds4-core/src/minimax_h3/text.rs:338; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn pack_audio - defined crates/ds4-core/src/minimax_h3/dit.rs:1206; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn parse - defined crates/ds4-core/src/minimax_h3/safetensors.rs:56; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn parse_request - defined crates/ds4-core/src/minimax_h3/video_api.rs:339; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn parts - defined crates/ds4-core/src/minimax_h3/presets.rs:53; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn patch_rows - defined crates/ds4-core/src/minimax_h3/vision.rs:413; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn patchify_video - defined crates/ds4-core/src/minimax_h3/dit.rs:1128; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn position_rows - defined crates/ds4-core/src/minimax_h3/vision.rs:471; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn positions - defined crates/ds4-core/src/minimax_h3/multimodal.rs:121; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn probe_visual_size - defined crates/ds4-core/src/minimax_h3/media.rs:157; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn protocol_name - defined crates/ds4-core/src/minimax_h3/terminal.rs:50; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn query_dim - defined crates/ds4-core/src/minimax_h3/text.rs:240; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn read - defined crates/ds4-core/src/minimax_h3/layout.rs:75; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn read_audio_f32 - defined crates/ds4-core/src/minimax_h3/media.rs:350; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn read_data - defined crates/ds4-core/src/minimax_h3/safetensors.rs:149; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn read_image_f32 - defined crates/ds4-core/src/minimax_h3/media.rs:219; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn read_index - defined crates/ds4-core/src/minimax_h3/safetensors.rs:939; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn read_into - defined crates/ds4-core/src/minimax_h3/text_weights.rs:107; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn read_shard - defined crates/ds4-core/src/minimax_h3/safetensors.rs:395; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn read_video_f32 - defined crates/ds4-core/src/minimax_h3/media.rs:269; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn reconcile - defined crates/ds4-core/src/minimax_h3/safetensors.rs:995; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn render - defined crates/ds4-core/src/minimax_h3/engine.rs:183; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn render_height - defined crates/ds4-core/src/minimax_h3/render.rs:220; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn render_width - defined crates/ds4-core/src/minimax_h3/render.rs:216; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn resident - defined crates/ds4-core/src/minimax_h3/dit.rs:807; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn resize_rgb24 - defined crates/ds4-core/src/minimax_h3/render.rs:388; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn resized_to - defined crates/ds4-core/src/minimax_h3/render.rs:341; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn resolve - defined crates/ds4-core/src/minimax_h3/presets.rs:262; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn reuse_schedule - defined crates/ds4-core/src/minimax_h3/host.rs:191; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn rgb_f32_to_u8 - defined crates/ds4-core/src/minimax_h3/render.rs:375; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn role - defined crates/ds4-core/src/minimax_h3/text_forward.rs:940; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn rope_rows - defined crates/ds4-core/src/minimax_h3/vision.rs:564; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn rope_scale - defined crates/ds4-core/src/minimax_h3/render.rs:256; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn round_bf16 - defined crates/ds4-core/src/minimax_h3/text_rope.rs:156; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn rows - defined crates/ds4-core/src/minimax_h3/sequence.rs:68; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn scale - defined crates/ds4-core/src/minimax_h3/render.rs:226; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn seed - defined crates/ds4-core/src/minimax_h3/host.rs:99; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn serving - defined crates/ds4-core/src/minimax_h3/host.rs:162; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn set_conditions - defined crates/ds4-core/src/minimax_h3/dit.rs:1770; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn set_zoom - defined crates/ds4-core/src/minimax_h3/terminal.rs:59; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn shards - defined crates/ds4-core/src/minimax_h3/safetensors.rs:195; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn sines - defined crates/ds4-core/src/minimax_h3/text_rope.rs:97; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn size - defined crates/ds4-core/src/minimax_h3/safetensors.rs:95; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn slot_bytes - defined crates/ds4-core/src/minimax_h3/dit.rs:290; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn span - defined crates/ds4-core/src/minimax_h3/weights.rs:96; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn spans - defined crates/ds4-core/src/minimax_h3/multimodal.rs:115; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn spawn - defined crates/ds4-core/src/minimax_h3/engine.rs:503; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn specs - defined crates/ds4-core/src/minimax_h3/vision.rs:221; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn start - defined crates/ds4-core/src/minimax_h3/video_api.rs:862; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn stream_slot_bytes - defined crates/ds4-core/src/minimax_h3/dit.rs:1742; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn submit - defined crates/ds4-core/src/minimax_h3/video_api.rs:822; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn summary - defined crates/ds4-core/src/minimax_h3/engine.rs:581; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn tags - defined crates/ds4-core/src/minimax_h3/text_forward.rs:103; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn tags_fnv - defined crates/ds4-core/src/minimax_h3/text_forward.rs:108; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn temporal - defined crates/ds4-core/src/minimax_h3/host.rs:72; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn tensor_bytes - defined crates/ds4-core/src/minimax_h3/safetensors.rs:174; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn tensor_count - defined crates/ds4-core/src/minimax_h3/safetensors.rs:207; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn tier_canvas - defined crates/ds4-core/src/minimax_h3/presets.rs:91; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn to_json - defined crates/ds4-core/src/minimax_h3/video_api.rs:151; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn to_value - defined crates/ds4-core/src/minimax_h3/video_api.rs:703; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn token - defined crates/ds4-core/src/minimax_h3/safetensors.rs:300; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn tokens - defined crates/ds4-core/src/minimax_h3/text_rope.rs:82; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn too_many - defined crates/ds4-core/src/minimax_h3/video_api.rs:143; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn total - defined crates/ds4-core/src/minimax_h3/text.rs:127; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn unavailable - defined crates/ds4-core/src/minimax_h3/video_api.rs:147; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn unpack_audio - defined crates/ds4-core/src/minimax_h3/dit.rs:1236; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn unpatchify_video - defined crates/ds4-core/src/minimax_h3/dit.rs:1167; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn validate - defined crates/ds4-core/src/minimax_h3/video_api.rs:421; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn values - defined crates/ds4-core/src/minimax_h3/text_forward.rs:91; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn video_elements - defined crates/ds4-core/src/minimax_h3/dit.rs:1732; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn video_latent_t - defined crates/ds4-core/src/minimax_h3/host.rs:57; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn video_patch - defined crates/ds4-core/src/minimax_h3/layout.rs:143; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn video_total_rows - defined crates/ds4-core/src/minimax_h3/sequence.rs:74; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn width - defined crates/ds4-core/src/minimax_h3/tokenizer.rs:178; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn window - defined crates/ds4-core/src/minimax_h3/dit.rs:709; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub fn with_store - defined crates/ds4-core/src/minimax_h3/text.rs:312; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct AdalnHashes - defined crates/ds4-core/src/minimax_h3/dit.rs:2684; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct AnchorImage - defined crates/ds4-core/src/minimax_h3/video_api.rs:306; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct AnchorPixels - defined crates/ds4-core/src/minimax_h3/condition.rs:47; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct ApiError - defined crates/ds4-core/src/minimax_h3/video_api.rs:102; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Canvas - defined crates/ds4-core/src/minimax_h3/render.rs:43; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Config - defined crates/ds4-core/src/minimax_h3/layout.rs:54; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Decoder - defined crates/ds4-core/src/minimax_h3/vae_video.rs:255; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Dit - defined crates/ds4-core/src/minimax_h3/dit.rs:1426; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Embedding - defined crates/ds4-core/src/minimax_h3/text_forward.rs:74; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Encoder - defined crates/ds4-core/src/minimax_h3/vae_encoder.rs:245; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct EncoderError - defined crates/ds4-core/src/minimax_h3/vae_encoder.rs:69; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Engine - defined crates/ds4-core/src/minimax_h3/engine.rs:149; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct EngineError - defined crates/ds4-core/src/minimax_h3/engine.rs:102; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct EngineHandle - defined crates/ds4-core/src/minimax_h3/engine.rs:493; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Entry - defined crates/ds4-core/src/minimax_h3/layout.rs:202; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Frames - defined crates/ds4-core/src/minimax_h3/vae_video.rs:72; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Grid - defined crates/ds4-core/src/minimax_h3/vision.rs:365; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Image - defined crates/ds4-core/src/minimax_h3/multimodal.rs:70; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct ImageRef - defined crates/ds4-core/src/minimax_h3/video_api.rs:265; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Index - defined crates/ds4-core/src/minimax_h3/safetensors.rs:934; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Inventory - defined crates/ds4-core/src/minimax_h3/safetensors.rs:181; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Latent - defined crates/ds4-core/src/minimax_h3/vae_encoder.rs:60; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Layout - defined crates/ds4-core/src/minimax_h3/layout.rs:219; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct ListQuery - defined crates/ds4-core/src/minimax_h3/video_api.rs:762; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct LoadOptions - defined crates/ds4-core/src/minimax_h3/dit.rs:796; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Manifest - defined crates/ds4-core/src/minimax_h3/safetensors.rs:987; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct MediaError - defined crates/ds4-core/src/minimax_h3/media.rs:58; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Outside - defined crates/ds4-core/src/minimax_h3/text.rs:110; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Position - defined crates/ds4-core/src/minimax_h3/sequence.rs:48; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Presentation - defined crates/ds4-core/src/minimax_h3/multimodal.rs:101; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Preset - defined crates/ds4-core/src/minimax_h3/presets.rs:114; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct PreviewDecoder - defined crates/ds4-core/src/minimax_h3/vae_video.rs:986; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct ReductionWindow - defined crates/ds4-core/src/minimax_h3/dit.rs:661; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct RenderReport - defined crates/ds4-core/src/minimax_h3/engine.rs:83; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct RenderRequest - defined crates/ds4-core/src/minimax_h3/engine.rs:57; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct RenderSpec - defined crates/ds4-core/src/minimax_h3/video_api.rs:285; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct RgbFrames - defined crates/ds4-core/src/minimax_h3/render.rs:287; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Rng - defined crates/ds4-core/src/minimax_h3/host.rs:89; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Rope - defined crates/ds4-core/src/minimax_h3/text_rope.rs:47; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Schedule - defined crates/ds4-core/src/minimax_h3/host.rs:150; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Segment - defined crates/ds4-core/src/minimax_h3/sequence.rs:40; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Sequence - defined crates/ds4-core/src/minimax_h3/sequence.rs:56; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Shard - defined crates/ds4-core/src/minimax_h3/safetensors.rs:134; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Span - defined crates/ds4-core/src/minimax_h3/weights.rs:31; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Spec - defined crates/ds4-core/src/minimax_h3/sequence.rs:85; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Store - defined crates/ds4-core/src/minimax_h3/safetensors.rs:190; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Temporal - defined crates/ds4-core/src/minimax_h3/host.rs:66; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Tensor - defined crates/ds4-core/src/minimax_h3/safetensors.rs:107; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct TensorRef - defined crates/ds4-core/src/minimax_h3/layout.rs:194; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct TextWeights - defined crates/ds4-core/src/minimax_h3/text_weights.rs:30; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Tokenizer - defined crates/ds4-core/src/minimax_h3/tokenizer.rs:122; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VaeError - defined crates/ds4-core/src/minimax_h3/vae_audio.rs:49; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Video - defined crates/ds4-core/src/minimax_h3/video_api.rs:687; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VideoDelete - defined crates/ds4-core/src/minimax_h3/video_api.rs:750; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VideoError - defined crates/ds4-core/src/minimax_h3/video_api.rs:186; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VideoList - defined crates/ds4-core/src/minimax_h3/video_api.rs:728; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VideoRequest - defined crates/ds4-core/src/minimax_h3/video_api.rs:275; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VideoSize - defined crates/ds4-core/src/minimax_h3/video_api.rs:206; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VideoStore - defined crates/ds4-core/src/minimax_h3/video_api.rs:787; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VisionLayout - defined crates/ds4-core/src/minimax_h3/vision.rs:161; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VisionOutput - defined crates/ds4-core/src/minimax_h3/vision_forward.rs:39; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VisionRope - defined crates/ds4-core/src/minimax_h3/vision.rs:540; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VisionSpan - defined crates/ds4-core/src/minimax_h3/multimodal.rs:138; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct VisionWeights - defined crates/ds4-core/src/minimax_h3/vision_weights.rs:32; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Waveform - defined crates/ds4-core/src/minimax_h3/vae_audio.rs:40; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
ITEM pub struct Weights - defined crates/ds4-core/src/minimax_h3/weights.rs:49; compiled: clippy -p ds4-core --lib --features h3-gates exit 0; gated: cargo test -p ds4-core --lib (646 passed, 7 ignored)
FILE tests/fixtures/h3/README.md - read after the fix; the anchor status matches the served wire (check 18) and h3-rs --first-frame, its sha256 and 432x171 RGBA claims were re-verified, and the 28.90 dB re-lettered bound matches post-port-phases.md:933-934; D2 closed
FILE tests/fixtures/h3/costello-crocodile-2panel.png - PNG 432x171 8-bit RGBA (file); sha256 f364705ae974c13f709c3776ebe9e7eadcd45ada8c1a7f75d1bc7df31698f7d6 matches the fixture README
FILE tools/minimax_h3/README.md - read; the new directory index, incomplete over the later probes and host tools (O2)
FILE tools/minimax_h3/dit_probe.c - cc -fsyntax-only -I/data/h3.c exit 0; running it needs the Spark's libh3.a and checkpoint
FILE tools/minimax_h3/install-minimax-h3-spark.sh - read: checkpoint fetch, build, gate and serve on the Spark
FILE tools/minimax_h3/letterballoons.sh - read in full; run on a synthetic 1344x576 clip (exit 0): audio md5 identical to the input on both sides, the changed-pixel BBOX is x0=12 y0=8 x1=940 y1=199, the balloon fill is white and the rim black
FILE tools/minimax_h3/multimodal_probe.c - cc -fsyntax-only -I/data/h3.c exit 0; running it needs the Spark's libh3.a and checkpoint
FILE tools/minimax_h3/p9_copy_bench.cu - nvcc 13.3 -arch=compute_121a -I/data/h3.c -c exit 0; the measurement itself is Spark-only
FILE tools/minimax_h3/ppm_diff.c - read in full; compiled with cc -O2 and exercised: identical 0, differing 1 with BBOX, geometry 2, truncated 2; O3 records the band bound
FILE tools/minimax_h3/ref_gate.sh - read: the reference-media matrix driver (FL2VA and Ref2VA families); it runs the reference binary on the Spark
FILE tools/minimax_h3/st_probe.c - cc -fsyntax-only -I/data/h3.c exit 0; running it needs the Spark's libh3.a and checkpoint
FILE tools/minimax_h3/text_probe.c - cc -fsyntax-only -I/data/h3.c exit 0; running it needs the Spark's libh3.a and checkpoint
FILE tools/minimax_h3/tok_probe.c - cc -fsyntax-only -I/data/h3.c exit 0; running it needs the Spark's libh3.a and checkpoint
FILE tools/minimax_h3/video_encoder_probe.c - cc -fsyntax-only -I/data/h3.c exit 0; running it needs the Spark's libh3.a and checkpoint
FILE tools/minimax_h3/vision_probe.c - cc -fsyntax-only -I/data/h3.c exit 0; running it needs the Spark's libh3.a and checkpoint
FILE tools/minimax_h3/vq/LICENSE-YoungAi - present: the vendored license of the transferred bake sources
FILE tools/minimax_h3/vq/README.md - read: provenance, the tested table and the rung A/B contract
FILE tools/minimax_h3/vq/bake/ds4_gguf_write.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/bake/eg2_artifact_check.sh - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/bake/eg2_gguf.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/bake/eg2_pack.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/bake/eg2_vq.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/bake/h3_gvq.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/bake/h3_gvq.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/bake/h3_pack.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/build.sh - read and run: exit 0, five binaries built into the ignored bin/
FILE tools/minimax_h3/vq/tests/make_fixture.c - cc -c with the toolchain's include roots exit 0
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/deepseek4-quantize.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/deepseek4-quantize_p1.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/deepseek4-quantize_p2.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/deepseek4-quantize_p3.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/deepseek4-quantize_p4.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/deepseek4-quantize_p5.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/deepseek4-quantize_p6.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/dump_gguf_meta.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/dump_st_meta.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/engram_probe.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/engram_wkv_probe.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/expert_probe.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/go2b_parity.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/go2b_qc.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/head_row_probe.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/ladder_vq4.txt - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/onebit_quant.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/onebit_quant.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/quantize_gpu.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/quants.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/quants.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/quants_p1.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/quants_p2.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/quants_p3.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/rplan_solve.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/st_bridge.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/st_dump_tensor.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/st_locate.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/st_read_decl.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_calib.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_calib.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_cfg.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_codebook.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_dq.cuh - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_ef.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_ef.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_fp4.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_nc_alloc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_q4k.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_quant.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_quantize.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_quantize_cli.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_shcb.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_st_write.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_to_gguf.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_to_gguf_mtp.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_to_gguf_tok.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_to_gguf_vision.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_to_gguf_vq3.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_vq.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_vq3_bits.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_vq_pool.inc.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_vq_rate.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/v41_vq_rate.inc.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq2q2k_probe.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_em.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_gpu.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_gpu_attn.inc.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_gpu_moe.inc.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_gpu_moefused.inc.cu - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_merge_v4.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_merge_v4_p1.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_merge_v4_p2.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_merge_v4_p3.inc.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_qc.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_qc_bytes.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/gguf-tools/quantize/vq_shim.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_amp_fmt.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_etgd.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_float.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_fp8.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_gguf.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_gguf.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_gguf_write.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_gr_fnv.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_json.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_json.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_quantfmt.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_quantfmt.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_sha256.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_sha256.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_st.c - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
FILE tools/minimax_h3/vq/youngai/src/common/ds4_st41.h - built by vq/build.sh (exit 0) and exercised: bin/h3_gvq selftest exit 0 (13-bit round trip, strict read, determinism, 20 GVQ1 + 5 GVR8 rejections)
LIVE CHECKS (command - exit - key line)

Re-run against the corrected tree:
 1  cargo fmt --check
    exit 0 - no output (0 bytes).
 2  cargo clippy -p ds4-core --lib --features h3-gates
    exit 0 - "generated 109 warnings ... Finished `dev` profile in 0.05s"; zero
    of them come from a minimax_h3 file (was 2 before the O5 fix).
 3  cargo test -p ds4-core --lib
    exit 0 - "test result: ok. 646 passed; 0 failed; 7 ignored; 0 measured;
    0 filtered out; finished in 0.62s" (653 run). The 7 ignored need the h3
    objects and checkpoint.
 4  cargo test -p ds4-server --lib h3_
    exit 0 - "test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured;
    365 filtered out; finished in 0.12s".
 5  cargo clippy -p ds4-server
    exit 0 - "generated 116 warnings ... Finished in 1.25s"; none from
    h3_video.rs or h3_engine.rs.
 6  cargo check --examples -p ds4-core --features h3-gates
    exit 0 - the six gates plus examples/minimax_h3_inspect type-check.

Carried from this session's first pass; the files and inputs they touch were not
modified by the fixes (git status lists only the five fixed paths plus this
report):
 7  cargo build --example h3_dit_gate -p ds4-core --features h3-gates
    exit 101 - compiles, then fails only at link: "undefined symbol: h3d_alloc
    ... referenced by h3_dit.rs:602", 20 undefined h3d_*/h3t_* symbols before
    the linker's error limit stopped the list. Expected: the objects are built
    and linked by the Makefile rule (Makefile:434-446), not by cargo.
 8  cargo build -p ds4-server --bin h3-server-rs, then a live listener
    exit 0 - GET /v1/models 200, GET /v1/videos 200, GET and DELETE
    /v1/videos/video_9 404, HEAD /v1/videos/video_9/content 404, GET /v1/nope
    404; --help exit 0; startup line "renderer disabled (POST /v1/videos
    answers 503)".
 9  cargo build -p ds4-cli --bin h3-rs
    exit 101 - "target `h3-rs` in package `ds4-cli` requires the features:
    `native`" (the Makefile passes --features native, Makefile:508); a usage
    error, not a defect.
10  cargo rustc -p ds4-cli --bin h3-rs --features native --, linked against the
    four objects compiled in check 11 plus -lcublasLt -lcudart -lstdc++
    exit 0 - ./target/debug/h3-rs --help exit 0; the no-argument path prints
    "h3-rs: -d MODEL_DIR is required" and exits 2; nm finds 70 resolved
    h3d_/h3t_/h3v_/h3g_ text symbols, so the ds4-sys FFI declarations agree
    with the native headers at link time.
11  nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a
    -DDS4_CUDA_HAVE_MXF4=1 -Xcompiler -fPIC -Xcompiler -pthread -c
    native/h3/{h3_device,h3_text,h3_vision,h3_dit}_gpu.cu
    exit 0 for all four (the Makefile's H3_NVCCFLAGS, Makefile:409).
12  sh tools/minimax_h3/vq/build.sh
    exit 0 - five binaries; 55 compiler warnings, all in the vendored youngai
    sources. Output under the gitignored tools/minimax_h3/vq/bin (.gitignore:272).
13  tools/minimax_h3/vq/bin/h3_gvq selftest
    exit 0 - "ok decode == value rule on 65536 values; cos=0.999668
    relL2=2.5762%", thread-count determinism, "ok 20 GVQ1 + 5 GVR8 mutations
    rejected".
14  cc -O2 -o ppm_diff tools/minimax_h3/ppm_diff.c -lm, then four runs
    identical pair exit 0; one-pixel change exit 1 ("BBOX x0=1 y0=0 x1=1 y1=0");
    geometry mismatch exit 2; truncated file exit 2.
15  sh tools/minimax_h3/letterballoons.sh in.mp4 out.mp4
    exit 0 - synthetic 1344x576 clip: 48 frames and the aac track survive, the
    audio md5 f8e7f1ce8d8c256c25fd480ae125099c is identical on input and
    output, frame 5 differs only inside the balloons (BBOX x0=12 y0=8 x1=940
    y1=199), fill 255,255,255, rim 0,0,0.
16  cc -fsyntax-only -I/data/h3.c on the seven probe sources, cc -c on
    vq/tests/make_fixture.c, nvcc -c on p9_copy_bench.cu
    exit 0 each.
17  file / sha256sum / jq on the fixtures
    PNG 432x171 8-bit RGBA, sha256 f364705ae974c13f709c3776ebe9e7eadcd45ada8c1a7f75d1bc7df31698f7d6
    (matches tests/fixtures/h3/README.md); tokenizer_portable.json is JSON.

New in this pass, run to verify the fixed documents against the served code:
18  live POST /v1/videos matrix (`h3-server-rs --renderer /bin/true`, so the
    request reaches decoding instead of the disabled-renderer 503)
    multipart ref.png 200; multipart ref.jpeg 200; multipart ref.tiff 400;
    JSON {"input_reference":{"file_id":"file_1"}} 400; JSON https image_url
    400 ("expected a base64 data URI", invalid_value); JSON data URI 200.
19  live anchor size bound
    a data URI decoding to exactly 16 MiB (16,777,216 bytes, 22,369,624 base64
    characters, 22,369,695-byte body) 200; one byte more 400 with
    "Invalid 'input_reference': decoded size must be 1..=16777216 bytes.",
    code invalid_value. Both edges of the documented bound hold.

No Python computed anything in this QA: every check above is shell, cargo, cc or
nvcc. One "python3 -c 1" no-op guard appears in a command line from the first
pass and produced nothing; no Python code was written or used.

STATIC CHECKS

S1. h3-rs flags against the reference CLI /data/h3.c/main.c (reference at
    eae5d54, read-only).
    Reference option table main.c:292-348 with the getopt string "d:p:o:h";
    port parser crates/ds4-cli/src/bin/h3-rs.rs:657-702.
    Eighteen reference flags have no port equivalent, each accounted for:
      --ref-image, --ref-image-size, --ref-video, --ref-silent-video,
      --ref-video-audio, --ref-audio (six): INTENTIONAL and recorded. Ref2VA is
      unported - post-port-phases.md:436 and :916-921, and sequence.rs:20-25
      names the deviation in code (build returns Unsupported::References).
      --use-int8-row-fc2 and the ten --use-slower-* flags (eleven): INTENTIONAL.
      In the reference they are CUDA no-ops or slow-path selectors
      (h3_gpu_cuda.cu:1058-1059, :1139-1141, :1307-1308; dit->int8_* selection
      at h3_dit.c:1634-1646). h3-rs.rs:16-18 states the policy and the parser
      refuses them (h3-rs.rs:702); the unported table records the refusal
      (post-port-phases.md:1133) and the P12 item now agrees (post-port-phases.md:632-638,
      O1 closed).
      --profile: INTENTIONAL; per-phase Metal timing (main.c:61-62), Metal out
      of scope (post-port-phases.md:1137).
    Port-only flag: --hash (gate hashes), a diagnostic addition.
    No other difference: model-dir, prompt, output, width, height,
    render-width, render-height, frames, seconds, steps, reuse, layers,
    core-reuse, token-reduction, ssd-streaming, use-reference-rope, seed,
    first-frame, last-frame, frames-dir, preview-dir, show, zoom, info and
    -h/--help all exist on both sides.

S2. Server routes and fields against docs/minimax-h3/openai-videos.md:24-36.
    Dispatch h3_video.rs:395-411 and router :601-621 implement exactly the seven
    documented method/path pairs; anything else is 404 unknown_url. Live in
    checks 8 and 18. The create body (video_api.rs:339-401) accepts prompt,
    model, seconds, size and input_reference{file_id,image_url} from JSON,
    urlencoded and multipart bodies, as documented. The input_reference
    semantics now match the document in every clause checked against the code
    and live (S6.1): the mismatch of the first pass (D1) is gone.

S3. "executes nothing" from a request body.
    Search: grep -n 'Command::new|std::process|reqwest|ureq|TcpStream::connect|
    fs::read|fs::write|fs::remove_file|File::open|File::create'
    crates/ds4-server/src/h3_video.rs crates/ds4-server/src/h3_engine.rs
    crates/ds4-server/src/bin/h3-server-rs.rs
    Hits: h3_video.rs:216 Command::new(&argv[0]) and the file calls at :135,
    :151, :511, :547 - all on server-generated paths. No HTTP client appears in
    ds4-server/Cargo.toml [dependencies], and h3_engine.rs has no process, path
    or network call of its own. argv is built at h3_video.rs:186-206 with the
    program and fixed arguments from startup and request values appended as
    separate entries at fixed positions, so a prompt cannot become an option.
    Ids are server-generated video_N (video_api.rs:838-841); the output is
    out_dir.join("{id}.mp4") (h3_video.rs:430); the anchor is
    output.with_extension("anchor.<ext>") (h3_video.rs:118-140) over a
    three-entry extension map (video_api.rs:315-323). References: a data: URI
    only (video_api.rs:526-563) or a multipart part; file_id refused; nothing
    is fetched - measured in check 18, where the https image_url answers 400.
    NUANCE, stated plainly: a POST does start a process - the operator's own
    renderer, which is the documented alternative backend
    (openai-videos.md:242-282, "--renderer PROGRAM ... chosen here, never by a
    request body"). The AGENTS.md "executes nothing" clause is the ds41
    carriage rule; the h3 addendum allows this one HTTP surface, and no request
    value selects a program, an argument position, a working directory or a
    path.

S4. Tool deliverables.
    tools/minimax_h3/ppm_diff.c - every claim in its header holds when run
    (check 14): SIZE, PIXELS, PSNR against full scale, BBOX including both
    ends, BAND lines, and the 0/1/2 exit contract. O3 records its 64-band
    (8192 px) bound and the undocumented axis=y lines; both are unchanged and
    neither is reachable from an h3 canvas (H3_MAX_PIXELS, video_api.rs:70).
    tools/minimax_h3/letterballoons.sh - its geometry claim and its audio claim
    ("Audio is stream-copied; the model's track is untouched") both hold;
    check 15 is the measurement.

S5. The DiT row map carrying presentation modality tags (commit 30dfe326,
    crates/ds4-core/src/minimax_h3/dit.rs).
    Reference h3_dit_schedule_row_map at h3_dit_schedule.c:434-484; port
    row_map at dit.rs:576-641. Arm by arm: the validity and tag-count gate
    (:436-440 against :583-588), text rows with the 1 fallback and the
    MODALITIES refusal (:448-457 against :594-606), COND and REF_IMAGE (:458-462
    against :608-616), REF_AUDIO (:463-466 against :617-624), AUDIO (:467-470
    against :625), VIDEO (:471-473 against :626), the modulation fill (:476-483
    against :628-631) and the signature check (:484 against :636-640). The
    tags reach the map where the reference passes text->tags (h3_dit.c:1027-1030):
    dit.rs:1571-1573 and :1598, with the length pinned to signature[0] by
    Geometry::check (dit.rs:1324). The reference's malformed-segment refusals
    (:441-443) have no port counterpart; O4 records why that is unreachable.

S6. Fixes verified (the corrections to D1, D2, O1, O5).

S6.1 docs/minimax-h3/openai-videos.md - every changed claim checked against the
    code and, where the code path is reachable without a checkpoint, live:
      status line: the resident engine is the port's own backend (h3_engine.rs)
      and the external program is the alternative; the bin's own header says the
      same (h3-server-rs.rs:3-5) and the selection is startup-only (:190-205).
      See O7 for the one wording reservation.
      create-body row and the paragraph after the encodings list: a data URI or
      a multipart file part becomes the anchor (decode_reference
      video_api.rs:526-563; data_uri :595-604 requires "data:" plus
      ";base64," with non-empty parts; check_anchor :569-588). Measured live:
      data URI 200 (check 18); multipart .png 200; .tiff 400.
      "A decoded anchor must be 1 byte to 16 MiB": MAX_ANCHOR_BYTES
      (video_api.rs:47) is 16 * 1024 * 1024 and check_anchor refuses
      empty or larger with invalid_value (:570-578). Measured at both edges in
      check 19.
      "file_id is refused with 400 unsupported_parameter": video_api.rs:532-540;
      the unit test an_anchor_is_refused_by_name asserts the same
      (video_api.rs:1487-1500), and the live matrix confirms 400.
      the three new error-table rows match the codes above.
      deviation 7: pixels base64-only, last frame CLI-only - h3_engine.rs:65-68
      sets first_frame from the anchor and leaves last_frame None.
    One incompleteness found, recorded as O6: the multipart sentence says the
    part name must end in .png, .jpg or .webp, while media_type_for_extension
    (video_api.rs:327-338) also accepts .jpeg, case-insensitively; measured
    live (ref.jpeg 200).

S6.2 tests/fixtures/h3/README.md - the anchor status now matches the served
    wire (S6.1) and the CLI flag name --first-frame is right
    (h3-rs.rs:682, usage line :800); "pixels arrive base64-only there" matches
    the served decode. The sha256 and 432x171 RGBA claims were re-verified
    (check 17). The "28.90 dB" re-lettered bound matches the record:
    post-port-phases.md:933-934 ("1536x608 art anchored at 1536x608 ... PSNR
    28.90 dB (rms 9.15, max 131)"), with the 1536x608 canvas for the 48:19
    fixture at :732. Ref2VA unported is consistent with :436 and :916-921.

S6.3 docs/minimax-h3/post-port-phases.md - the --use-int8-row-fc2 item now
    reads "The port refuses the flag instead of accepting an inert switch
    (h3-rs.rs:16-18, port plan section 6.3); no output differs either way",
    which matches the parser (unknown flag, h3-rs.rs:702), the CLI header
    (:16-18), port-plan.md:564 ("drop it and say so") and the unported table
    (:1133). The inert-environment-variable sentence no longer likens them to
    it (:632-633). O1 closed.

S6.4 clippy fixes - render.rs:633 drops the needless "mut" from
    "(audio_read, audio_write)" in mux_mp4; the rest of the function is
    unchanged (git diff is one line). vae_video.rs:362 becomes
    "tile.len().is_multiple_of(frame)" under the same "frame > 0" and
    "!tile.is_empty()" guards, which is the same predicate for a usize and
    cannot divide by zero. Confirmed: cargo clippy -p ds4-core --lib
    --features h3-gates exit 0 with 0 warnings from h3 files (check 2), cargo
    fmt --check exit 0 (check 1), and the test totals are unchanged, 646 passed
    / 7 ignored (check 3) and 24 passed (check 4). O5 closed.


S6.5 docs/minimax-h3/port-plan.md (edited by the same fix pass after the first
    report) - its deviations paragraph at :1038-1045 no longer claims
    --token-reduction, --show and --zoom are unimplemented (all three are parsed,
    h3-rs.rs:686, :689, :690) and correctly keeps the interactive session,
    Ref2VA and the audio VAE's encode half as not implemented (vae_audio.rs
    exposes only pub fn decode, :386, and no stdin loop exists in h3-rs); it now
    points at the post-port-phases unported table as the running record.

ACCEPTED RISKS (checks that could not run on this host)

R1  No h3 model work ran: no h3_*_gate, no h3-rs --info, no render, no MP4, no
    quantization bake. The FL2VA checkpoint lives only on the Spark
    (/mnt/h3-artifacts, ~144 GB) and the four native objects are sm_121a SASS,
    which this host's RTX 4070 SUPER (sm_89) cannot execute. Every recorded
    Spark number in the branch docs (hash gates, the 27.085 dB anchored-frame
    note, the 28.90 dB re-lettered bound, the lettered-clip renders, the VQ bake
    measurements) is accepted as recorded, not re-measured here.
R2  The Makefile's tests/h3_* gates were not run for the same reason: they link
    the native objects and then load the checkpoint.
R3  The reference-side probes (dit_probe, text_probe, vision_probe,
    multimodal_probe, video_encoder_probe, ref_gate.sh) need the reference's
    libh3.a and the checkpoint; only their compilation was verified (check 16).
R4  The DiT/vision/multimodal parity this branch claims cannot be reproduced on
    this machine at all; the row_map review in S5 is a source comparison against
    /data/h3.c, not a numerics check.
R5  The fixes verified in S6 are uncommitted working-tree edits on top of
    02ba4b67, and the surface list they leave is byte-identical to the one the
    first pass read (559, same set), so the gate's coverage requirement is met
    by the same lines. docs/README.md still carries an unrelated uncommitted
    index edit by a concurrent writer; I did not touch it.
R6  ppm_diff's out-of-range band behaviour (O3) was measured on a synthetic
    9000-wide PPM, not on an h3 frame; no h3 canvas can reach the bound.
R7  The two live HTTP checks (18, 19) ran against the debug, feature-less
    h3-server-rs with a stub renderer, so they verify request decoding,
    validation and error mapping, not a render: nothing was generated.

FINDINGS

D1  CLOSED - VERIFIED FIXED (was: docs/minimax-h3/openai-videos.md contradicted
    the served surface about input_reference). The status line, the create-body
    row, the paragraph after the encodings list, the three error-table rows and
    deviation 7 now describe the served behaviour, and each clause was checked
    against decode_reference/data_uri/check_anchor and the live matrix
    (S6.1, checks 18 and 19). The only residue is the wording nit O6.
D2  CLOSED - VERIFIED FIXED (was: tests/fixtures/h3/README.md repeated the
    stale refusal). It now states the anchor status, names --first-frame
    correctly (h3-rs.rs:682, usage :800) and records the P12.9 28.90 dB bound
    that post-port-phases.md:933-934 holds (S6.2).
O1  CLOSED - VERIFIED FIXED: post-port-phases.md now records that the port
    refuses --use-int8-row-fc2 (h3-rs.rs:16-18, :702; port-plan.md:564;
    unported table :1133) and no longer likens the inert environment variables
    to it (S6.3).
O5  CLOSED - VERIFIED FIXED: the two clippy warnings in new h3 code are gone
    (render.rs:633, vae_video.rs:362), clippy reports 0 warnings from h3 files,
    fmt is clean and the test totals are unchanged (S6.4, checks 1-4).
O2  OPEN, observation, not a defect: tools/minimax_h3/README.md indexes four
    probes plus vq/ while the directory also holds multimodal_probe.c,
    vision_probe.c, video_encoder_probe.c, p9_copy_bench.cu, ref_gate.sh,
    ppm_diff.c, letterballoons.sh and install-minimax-h3-spark.sh, all of which
    the h3 docs reference. Unchanged in this pass.
O3  OPEN, observation, not a defect: ppm_diff.c's band arrays are MAX_BANDS=64
    (8192 px) with no bound check - a 9000-wide PPM wrote past columns[63]
    without crashing - and its usage block documents only the axis=x band lines
    while the code prints axis=y too. Unreachable for h3 frames
    (H3_MAX_PIXELS, video_api.rs:70). Unchanged in this pass.
O4  OPEN, observation, not a defect: row_map has no counterpart to
    h3_dit_schedule.c:441-443 and would panic on a hand-built malformed
    Sequence (fields are pub), though sequence::build and Geometry::check make
    that unreachable. Unchanged in this pass.
O6  CLOSED - VERIFIED FIXED: docs/minimax-h3/openai-videos.md:88-89 now reads
    ".png, .jpg/.jpeg or .webp", which matches media_type_for_extension
    (video_api.rs:327-338: png; jpg|jpeg; webp, matched after
    to_ascii_lowercase) and the live matrix of check 18 (ref.jpeg 200,
    ref.tiff 400).
O7  CLOSED - VERIFIED FIXED: the status paragraph (openai-videos.md:3-8) now
    names the two exclusive backends and the 503, checked against
    h3-server-rs.rs:3-8 (the host still starts with neither and every create
    answers 503), :51-57 (usage: --model-dir renders in process, --renderer
    delegates, without either 503), :120-121 (--renderer is startup-only),
    :190-195 (the two are exclusive, exit 2, backend chosen before any request)
    and :154-157 (EngineRenderer opened once at startup); the 503 carries code
    renderer_disabled (h3_video.rs:101, :416; ApiError::unavailable is 503
    service_unavailable_error, video_api.rs:147-149), asserted by the test
    disabled_renderer_answers_503 (h3_video.rs:1778-1786).

Nothing else failed. 559/559 surfaces are covered, the two first-pass defects
and the two fixable observations are closed with the evidence in S6, and every
residual note above is an observation about a non-served tool or a wording
precision, none of which contradicts the served code.

verdict: overall PASS
