# MiniMax-H3 maintainer validation — 2026-10-26

Independent validation for [PR #111](https://github.com/Baekpica/ds4-dfm-rs/pull/111).
This records bounded BF16 execution and reference parity, not general video
quality or production capacity qualification.

## Identities

- Model: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.
  All 102 selected unique files (210,331,771,721 bytes) passed full remote
  digest and size checks. The 65 shared Ref2VA files resolve to their
  byte-identical FL2VA counterparts; the distinct model index is preserved.
- Reference: [h3.c at eae5d54](https://github.com/matrixfede/h3.c/tree/eae5d54e3a0419064b64171dcaebcfd92e9f3f63),
  built without source changes.
- Host: NVIDIA GB10, aarch64 Linux, CUDA 13.4.92, `sm_121a`, Rust 1.98.0.
  Driver UMD 610.57.04 / KMD 615.71.10. The 300–2200 MHz clock policy was
  preserved; busy SM clocks were approximately 2190 MHz. FFmpeg 6.1.1.
- One model process at a time, through `tools/host_memory_guard.py` with
  2 GiB reserve/trip. All completed CLI gates exited 0 without swap.

Binary SHA256:

| Binary / source | SHA256 |
|---|---|
| CLI `882d4ad7` (text-only run) | `771811e64728b56599f90084c400dab58bc78e732fa1a333a8bcc70263a249de` |
| CLI `cae0337b` (conditioning runs) | `09b7f1c217d59756fb6ffff44d1313fe8791e72cb33ab995fb3cecfbe3158862` |
| C reference `eae5d54` | `ee073b6eb578a0fdb18ff5199b0881c43845f53aaaab63fde6388ef98e44fd32` |
| Native server `75bc100d` | `bd33a2f12a28222f1342d1d9e90bde6f8542f99e116228ed9465267ad588515a` |

## Matched CLI outputs

Every pair used a fresh process, the same model, seed 42, reuse 1, the full
50-layer stack and this exact prompt:

```text
A man wearing black coat is walking in the rain. He is listening the music. 'Cyberpunktic' city's night lights his gloomy face.
```

| Input | Canvas / frames / steps | Rust wall time | C wall time | MP4 comparison |
|---|---|---:|---:|---|
| Text | 512×512 / 56 / 20 | 762.36 s | 764.62 s | Byte-identical |
| First + last frame | 256×256 internal → 512×512 output / 22 / 2 | 113.97 s | 118.17 s | Byte-identical |
| Image + video with soundtrack + audio | 256×256 / 56 / 2 | 184.92 s | 181.92 s | Byte-identical |

The text-only pair also passed decoded-frame and decoded-audio hash
comparisons. Its output is 2.333 seconds of H.264 video and stereo 32 kHz AAC.
The conditioning pairs check the repaired internal-canvas path and the
Ref2VA partition with all three media types. Their two-step outputs establish
execution and parity only.

Common MP4 SHA256 for each pair, in table order:

```text
54ffc07656c23b314c405615926f4f05ff5ff7977d2151572622a3737099fde4
bbb653329fa781b7bade3ac6885cf04ea61fe65e7d47c0a36172b6e3eff796a2
ae9e3b34fd4f6d8b34535beb473a62392cfd8225402a2f2e8f4271c3b4c6fdec
```

The Rust 20-step run spent 37.231 s encoding text, 48.156 s loading the DiT,
620.051 s denoising, 1.718 s decoding audio, 53.922 s decoding frames and
0.342 s muxing. Minimum host MemAvailable was 79.419 GiB. These are one pair
of timings, not a repeated performance experiment: the 0.3% wall-time
difference does not establish a speed advantage. Denoising accounts for
about 81% of the current Rust wall time.

### Reproduce the inputs

Set `MODEL_DIR` to the pinned checkpoint, `PROMPT` to the exact text above,
and `BIN` to either the Rust CLI or the pinned C binary. Run each process
under the memory guard; do not run both models concurrently.

```sh
python3 tools/host_memory_guard.py --max-gib 128 --high-gib 128 \
  --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 1800 \
  --log scratch/h3-check.memory.jsonl -- \
  "$BIN" -d "$MODEL_DIR" -p "$PROMPT" \
  --width 512 --height 512 --frames 56 --steps 20 --seed 42 --reuse 1 \
  -o scratch/rain.mp4

ffmpeg -v error -i scratch/rain.mp4 -frames:v 1 scratch/first.png
ffmpeg -v error -i scratch/rain.mp4 -vf 'select=eq(n\,55)' \
  -frames:v 1 scratch/last.png
ffmpeg -v error -i scratch/rain.mp4 -vf scale=128:128 \
  -c:v libx264 -threads 1 -crf 23 -pix_fmt yuv420p \
  -c:a copy scratch/reference.mp4
ffmpeg -v error -i scratch/rain.mp4 -vn -ac 2 -ar 32000 \
  -c:a pcm_f32le scratch/audio.wav
```

For the anchor pair, use `--width 512 --height 512 --render-width 256
--render-height 256 --frames 22 --steps 2 --first-frame scratch/first.png
--last-frame scratch/last.png`. For the reference pair, use `--width 256
--height 256 --frames 56 --steps 2 --ref-image scratch/first.png --ref-video
scratch/reference.mp4 --ref-audio scratch/audio.wav`. Keep seed, reuse and
prompt unchanged; choose distinct output and guard-log paths for each run.
The [reference gate](../../tools/minimax_h3/README.md) compares decoded media
when container metadata differs.

## Native API gate

`h3-server-rs` at `75bc100d` rendered in process with the pinned model,
`--steps 2 --reuse 1 --max-jobs 2`, loopback binding and the memory guard.
The request used the exact rain prompt above:

```json
{"task":"t2va","prompt":"A man wearing black coat is walking in the rain. He is listening the music. 'Cyberpunktic' city's night lights his gloomy face.","conditions":[],"target":{"short_edge":768,"aspect_ratio":"1:1","duration_seconds":4},"seed":42}
```

Create returned an opaque id. Polling reached `completed`; the record had
size 768×768, seed 42 and 107 frames. `HEAD` returned the MP4 content type
and length without a body. Download returned 1,565,090 bytes, identical to
the server's output file. FFprobe found H.264 768×768, 107 frames at 24 fps
(4.458333 s after frame alignment), plus stereo 32 kHz AAC (4.450000 s).

MP4 SHA256:
`252e88ceb94085af5cc6d8d072da6bacf43109157d6f7dc0d829dd31de427f56`.

| Phase | Seconds |
|---|---:|
| Text encoding | 37.774 |
| DiT load | 48.538 |
| Denoise, two evaluations | 975.987 |
| Audio decode | 3.026 |
| Frame decode | 263.217 |
| Mux | 0.731 |
| Client create through observed completion and download | 1,340.264 |

The client polled every 10 seconds; its wall time includes that delay.
The minimum sampled MemAvailable was 67.066 GiB and maximum PSI full avg10
was 6.29%. No memory-guard trip occurred. After successful download the
operator stopped the guard/server (guard exit 75, stop signal); that shutdown
is distinct from the completed render and client exit 0. GPU processes then
cleared and available memory recovered to about 118 GiB.

This qualifies one cold text-to-video API request. It does not qualify a
20-step API workload, repeated resident-cache requests, long clips or all
conditioning combinations through HTTP. The CLI conditioning comparisons
above have their own narrower workload scope.

## Regression checks

- [Host parity CI at 75bc100d](https://github.com/Baekpica/ds4-dfm-rs/actions/runs/40326947577):
  formatting, Clippy, C/Rust parity, serialized workspace tests and all-target
  host checks passed. Existing LLM dispatch, admission, KV and shared kernels
  are unchanged by this PR.
- Focused regressions cover conditioning cache identities, image geometry,
  local-container restrictions, pipe ownership, output preservation, renderer
  descendants, bounded HTTP connections and transient accept failures.
  The final H3 server subset passed all 24 tests.
- The model-free CUDA boundary gate passed all 10 checks. Real FFmpeg mux
  validation, six reference-script regressions and six CLI conditioning
  decoder-boundary cases passed.
- `cargo check -p ds4-core -p ds4-server --target aarch64-apple-darwin
  --all-targets --locked -j2` passed at `cae0337b`. This is compilation only;
  H3 Metal execution is not implemented or qualified.

## Limits

The 20-step sample is one prompt and seed. Longer clips, larger canvases,
sustained queues and warm-cache throughput are not qualified here. There is
no H3 VQ/GGUF runtime, FP8 speed claim or CUDA profiling campaign in this PR.
API admission limits are a separate contract, not proof that every admitted
workload fits a given host. Raw logs and generated media remain in ignored
`scratch/pr111-review/`; the PR thread records the maintainer's results.

## v0.1.4 integration review — 2026-10-27

Reviewed the cumulative PR at `12392a691b24ea574e312c4ee74d210dba9a65f6`
and integrated main/v0.1.4 at `f986d6a598a383ad2ccbda2da350e5cbcfacc9bc`.
H3 remains under **Unreleased**; the published v0.1.4 tag is unchanged.
The following checks use source `8517f0871af45f3b2fa7ce8571f84dfc8bbb0b00`
on the same GB10/CUDA toolchain recorded above, with no model loaded.

The review removed unused preset, delivery-mode and weight-mapping APIs,
avoided unused CLI hashes and a frame-buffer copy, and consolidated Make
rules while preserving linked objects and arithmetic flags. H3 headers now
participate in dependency tracking and H3 outputs in `make clean`.
The existing CLI internal-canvas resizing and all conditioning modes remain.

Correctness fixes reject wrong-rank or wrong-shape VAE tensors before upload,
retain the first shard's tensor when names repeat, return tokenizer errors
for invalid byte symbols or malformed metadata, and validate media classes
for frame anchors as well as references. Existing decoder-supported
AAC/FLAC/AIFF/Matroska MIME types now receive the corresponding extensions.
Each defect was reproduced before its fix with a model-free regression.

Eleven DiT, text and vision launches now place their growing row/plane axis
on CUDA `grid.x`, avoiding the 65,535-block `grid.y`/`grid.z` limit. This
changes scheduling only: block sizes, per-row arithmetic and output indexing
are unchanged. Tiny-width tests reproduced eight long-axis launch failures
before the fix. Short multi-head/column fixtures also check exact outputs.
Long attention launches were inspected against the grid limits; their
quadratic workloads were not run.

| Check | Result |
|---|---|
| `cargo test -p ds4-core --lib minimax_h3 --locked -- --test-threads=1` | 218 passed |
| `cargo test -p ds4-server --lib h3_ --locked -- --test-threads=1` | 24 passed |
| `cargo check -p ds4-core --examples --features h3-gates --locked` | Passed |
| `timeout 30s tests/h3/native_test` | 29 passed; no checkpoint |
| Portable host, tokenizer and real FFmpeg mux executables | 3 passed |
| `python3 tests/h3/test_cli_conditioning.py` | 6 decoder-boundary cases passed |
| `h3-rs --help`, `h3-server-rs --help` | Both passed |

Final build command:

```sh
make -j2 CUDA_ARCH=sm_121 h3-rs h3-server-rs tests/h3/native_test \
  tests/h3-test_host_portable tests/h3-test_tokenizer_portable tests/h3-test_av_mux
```

| Binary at `8517f087` | SHA256 |
|---|---|
| `h3-rs` | `f1d4596ddcc9414b454fb16226521fd07197206621d950a22e49e8f88911f090` |
| `h3-server-rs` | `640c42a173a13ca7f6369b2b8aec15959cfb54ead1bf6ddbda6e821a97520de4` |
| `tests/h3/native_test` | `231b712d10a918e3a63ed3731ee8cb78f49817738382c2c1235e795b0933d70a` |

Raw logs, failing-first fixtures and the digest receipt remain under ignored
`scratch/pr111-v014/` (`validation-receipt.json`). A Make dry-run comparison
verified unchanged object sets for all 15 Rust targets and arithmetic flags
for all four CUDA objects; host-only tests no longer link CUDA libraries.
Header dependency rebuilding was checked with `make -n -W`.

The full-model CLI/API outputs and timings above remain historical evidence
at their named commits. This integration review did not repeat those renders,
Darwin compilation or reference-script checks. It adds no qualification for
long clips, larger canvases, sustained queues, warm caches or performance.
The final PR's Host parity status is recorded in its GitHub checks.
