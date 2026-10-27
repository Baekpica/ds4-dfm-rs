# MiniMax-H3

MiniMax-H3 generates video and audio with a diffusion transformer. Its Rust
host and CUDA kernels are separate from the autoregressive model families;
LLM context, KV-cache, token streaming and MTP options do not apply.

H3 is an unreleased addition after v0.1.4.

The runtime reads the released BF16 safetensors. FL2VA supplies text-to-video
and first/last-frame conditioning. Ref2VA supplies ordered image, video and
audio references. There is no H3 GGUF or VQ loader.

## Build and run

Linux CUDA, the repository's Rust toolchain, FFmpeg and FFprobe are required.
On DGX Spark:

```sh
make CUDA_ARCH=sm_121 h3-rs h3-server-rs
./h3-rs --help
./h3-server-rs --help
```

Use the [host memory guard](../host-memory-guard.md) for model runs. Run one
GPU job at a time. For example, a short 20-step clip with a local model:

```sh
MODEL_DIR=/path/to/MiniMax-H3
python3 tools/host_memory_guard.py --max-gib 128 --high-gib 128 \
  --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 1800 \
  --log scratch/h3.memory.jsonl -- \
  ./h3-rs -d "$MODEL_DIR" \
  -p 'A man in a black coat walks through a rainy neon-lit city.' \
  --width 512 --height 512 --frames 56 --steps 20 --seed 42 --reuse 1 \
  -o scratch/h3-rain.mp4
```

The normal denoise setting is 20 steps. Use `--steps 2` only for an execution
check; two-step output does not establish generation quality.
First/last anchors use `--first-frame` and `--last-frame`.
Ordered references use `--ref-image`, `--ref-video`, `--ref-silent-video`,
`--ref-video-audio` or `--ref-audio`; a request cannot combine references and
frame anchors. Audio alone is insufficient for Ref2VA.

## Measured generation cost

On GB10, one 512×512, 56-frame, 20-step text run took **762.36 s** (12m42s)
for a 2.333-second clip. The matched h3.c run took 764.62 s and produced the
same MP4 bytes. The native API at its smallest square request (768×768,
4 seconds requested, 107 aligned frames) took **1,340.264 s** to observed
completion and download even at **two steps**. That includes 10-second
status polling. Larger canvases and more steps require separate measurement.

The [dated validation](validation-2026-10-26.md) records exact inputs,
phase timings, memory observations and qualification limits. These single
runs are execution receipts, not throughput or broad quality guarantees.
The [v0.1.4 integration review](validation-2026-10-26.md#v014-integration-review--2026-10-27)
records subsequent fixes and model-free checks; full-model renders were not
repeated for that review.

## Model files

The review artifact is [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
at revision `42ed227ee7df40d41602854ae760620d6eb651fe`. The required unique
files total 210,331,771,721 bytes: all of FL2VA, the distinct Ref2VA transformer
and model index, and root metadata. The five remaining Ref2VA components are
byte-identical to their FL2VA counterparts at this revision.

```sh
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --local-dir "$MODEL_DIR" \
  --include 'FL2VA/*' --include 'Ref2VA/transformer/*' \
  --include 'Ref2VA/model_index.json' \
  --include '.gitattributes' --include 'LICENSE' --include 'README.md' \
  --include 'model_index.json' --include 'modular_model_index.json'

for component in audio_vae processor text_encoder tokenizer video_vae; do
  ln -s "../FL2VA/$component" "$MODEL_DIR/Ref2VA/$component"
done
```

Keep the distinct `Ref2VA/model_index.json`. This download size is a disk
requirement, not a peak RAM guarantee. Conversion is unnecessary for BF16.
The model's license is included in the downloaded repository.

## Interfaces and validation

- [Local video API contract](local-v1-contract.md)
- [Starting and using the API](api-usage.md)
- [Reference probes and media comparison](../../tools/minimax_h3/README.md)
- [GB10 maintainer validation, 2026-10-26](validation-2026-10-26.md)
- [Speed baseline against h3.c, 2026-10-27](speed-2026-10-27.md)

Model-free checks include H3 core and server unit tests, the portable
host/tokenizer gates, and `make tests/h3-test_av_mux`. The CUDA boundary gate
uses tiny buffers and no checkpoint:

```sh
make CUDA_ARCH=sm_121 tests/h3/native_test
timeout 30s tests/h3/native_test
```

A passing host or boundary test does not qualify generated media. Full
qualification must record the source revision, model revision, CUDA
architecture, exact prompt/media/seed/shape, decoded output comparison and
memory-guard results. Historical contributor results are not a current
qualification.

## Source and ownership

The port follows [h3.c](https://github.com/JLFN/h3.c/tree/eae5d54e3a0419064b64171dcaebcfd92e9f3f63)
at `eae5d54e3a0419064b64171dcaebcfd92e9f3f63`. Source comments cite files and
lines in that revision; `/data/h3.c` in an inherited citation denotes that
reference checkout, not a runtime path. Its MIT notice is preserved in
[LICENSE-h3.c](../../native/h3/LICENSE-h3.c).

The host pipeline is in `crates/ds4-core/src/minimax_h3/`, device bindings in
`crates/ds4-sys/src/h3_*.rs`, and kernels in `native/h3/`. The CLI is
`crates/ds4-cli/src/bin/h3-rs.rs`; HTTP orchestration is in
`crates/ds4-server/src/h3_video.rs`.
