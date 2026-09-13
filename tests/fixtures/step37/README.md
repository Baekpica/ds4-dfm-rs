# Step 3.7 MQ83 contract fixtures

`metadata.json` records official GGUF metadata at
`stepfun-ai/Step-3.7-Flash-GGUF@0b69336d2fd2adfdef9c66e425f7778196c31482`,
plus the mixed artifact's source revision marker. The test constructs a
metadata-only file with placeholder vocabulary strings; it is not a tokenizer
or inference fixture.

`mq83.tsv` records the independently inventoried 754 tensor names and GGUF
dimensions, with the owner's locked mixed recipe: Q8 critical/shared matrices,
Q4_K routed edges/down, IQ2_XXS interior gate/up, F32 controls. Payload size is
83,001,512,448 bytes. No weight data is included.

Original model revision: `5f6244077ac62e04eec3f320501ff8c2b293373a`.

`mtp.tsv`, `vision.tsv` and their metadata JSON files come from the handoff's
`sidecar-inventory.json`. They preserve the official Q8 MTP and F16 projector
layouts without payload data. Tests use them independently of runtime specs.

`primitives.h` exports FP32 values from the handoff's
`ds4-initial/fixtures/step37-primitives.json` (Torch 2.11.0+cu130, H100 NVL,
seed 3707). It covers clamps 0/7/16 and 288-expert top-8 routing, including
logits near -30 that expose the legacy normalization floor.

## Native gates

`text-inputs.json` preserves the eight text token streams from the Spark
handoff's BF16 probes. Only the inputs are reused; BF16 outputs are not MQ83
goldens. The native driver is an integration gate. The independent oracle is
StepFun llama.cpp at `8f34864def3d351316aaea5ce9b4a06e12198d3c`.

```sh
python3 tests/fixtures/step37/make_text_cases.py scratch/step37 --label native
python3 tests/fixtures/step37/make_text_cases.py scratch/step37 --label oracle
make tests/test_step37_forward CUDA_ARCH=sm_121

# Run each model process serially under tools/host_memory_guard.py.
tests/test_step37_forward "$STEP37_MAIN" @scratch/step37/native.cases - 64 8
step-text-oracle "$STEP37_MAIN" @scratch/step37/oracle.cases - 64 8
python3 tests/step37_logits.py scratch/step37/oracle/arithmetic.f32 scratch/step37/native/arithmetic.f32

# Same execution width; whole-history KV is the independent ring control.
STEP37_VERIFY_KV=1 tests/test_step37_forward "$STEP37_MAIN" scratch/step37/fixtures/ring832.tokens scratch/step37/ring832.f32 64 8
```

`STEP37_MAIN` is the first of the nine MQ83 shards. Both drivers accept a
single input/output pair or `@manifest` with whitespace-separated pairs.
Every case gets a fresh graph/context; the model weights stay resident.
Each output has nine little-endian F32 rows of 128896 logits: one after
prefill, then eight after greedy decode. These short choices do not qualify
long generation or answer quality. Use the production Rust benchmark for
performance: initial native forward includes lazy weight materialization.

Build `text_oracle.cpp` as an executable linked to `llama` in the pinned
checkout. For diagnostics, `STEP37_NO_FA=1` disables FA. The optional
`oracle-f32-attn.patch` (apply with `git apply --unidiff-zero`), together with `STEP37_NO_FA=1 STEP37_F32_ATTN=1
NVIDIA_TF32_OVERRIDE=0`, retains F16 KV but evaluates attention in F32.
Keep the unmodified FA run as a separate control. The strict comparator
still fails cross-engine rows; see [the measured status](../../../docs/step37-initial.md).

`STEP37_TRACE=DIR` records prefill intermediates in either driver.
`STEP37_REPLAY=REFERENCE_DIR` in the native trace driver supplies the same
reference input to each layer. Replay is a local operator check, not a full
model trajectory. The KV proof instead requires byte-identical logits and
all live KV rows across full/ring layouts and a matched-width tail rewrite.

The public session gate keeps one model mapping and compares native session
logits/KV with a separate graph using identical chunk widths:

```sh
make tests/test_step37_session tests/test_step37_state tests/test_cuda_span_lease CUDA_ARCH=sm_121
tests/test_step37_state
tests/test_cuda_span_lease
tests/test_step37_session --policy
# Run under host_memory_guard.py; includes the public generation callback.
tests/test_step37_session "$STEP37_MAIN" scratch/step37/fixtures/ring832.tokens
```

The small span-lease regression needs CUDA but no model. Host-only state
and ledger tests reject stale logits and preserve only untouched checkpoints
on errors. Production Step sessions use `DS4_STEP37_PREFILL_CHUNK` (default
512, valid 1–4096, capped by context); 64 is the locked structural gate width.

Rust server output tests use `cargo test -p ds4-server --test step37_output`
and `--test chat_input`; `make -j1 test-server-parity` covers existing families.
The live `tests/chat_template_live.py` protocol passes all 18 requests with
`--reasoning-effort none --max-tokens 128`, then all 18 with
`--reasoning-effort high --max-tokens 256`, against a single Step server at
context 4096. It exercises all three APIs, buffered/SSE tool calls and result
continuations. Keep MTP and image qualification separate.

The MTP component gate uses the actual Q8 sidecar without a main model.
`make_mtp_vectors.py` needs CPU PyTorch, NumPy and the pinned StepFun
`gguf-py` on `PYTHONPATH`. It writes both weights/file hashes and its selected
activation contract. Keep the two output directories separate:

```sh
python tests/fixtures/step37/make_mtp_vectors.py "$STEP37_MTP" "$REF_FP32"
python tests/fixtures/step37/make_mtp_vectors.py "$STEP37_MTP" "$REF_Q8" --activation q8_1
make tests/test_step37_mtp CUDA_ARCH=sm_121
# Default CUDA small-width Q8 path, identical reference input at each depth:
tests/test_step37_mtp "$STEP37_MTP" "$REF_Q8" --local
# Full predictor chain with dequantized F32 GEMMs (head still uses N=1 Q8):
DS4_CUDA_USE_MMQ=0 DS4_CUDA_Q8_F32_ALL=1 NVIDIA_TF32_OVERRIDE=0   tests/test_step37_mtp "$STEP37_MTP" "$REF_FP32"
```

Run GPU commands serially under the host memory guard (16 GiB suffices).
Both compare all seven hidden rows and the complete final-row vocabulary,
check each head's greedy token, and vary rejected rows for keep counts 0–7
across a wrapped SWA ring. Causally live KV and the next hidden row must be
byte-identical. `STEP37_TRACE=DIR` saves initial per-layer diagnostics.
Without `--local`, inputs propagate from native outputs; the default Q8
synthetic chain still fails the unchanged 3% criterion. It is not an
end-to-end MTP token/KV gate or a throughput measurement.


The production MTP lifecycle gate imports the sidecar from a VMM weight owner:

```sh
make tests/test_step37_spec ds4_weight_server CUDA_ARCH=sm_121
# Inspect the owner's dry-run plan before the guarded owner launch.
./ds4_weight_server --base "$STEP37_MAIN" --mtp "$STEP37_MTP" --backend vmm --scope mtp --manifest "$MANIFEST" --dry-run
./ds4_weight_server --base "$STEP37_MAIN" --mtp "$STEP37_MTP" --backend vmm --scope mtp --manifest "$MANIFEST"
# In another terminal, under the 100 GiB host guard, one worker at a time:
tests/test_step37_spec "$STEP37_MAIN" "$STEP37_MTP" "$MANIFEST" "$ARITHMETIC_TOKENS"
tests/test_step37_spec "$STEP37_MAIN" "$STEP37_MTP" "$MANIFEST" "$RING832_TOKENS"
```

Each run generates 32 tokens. It compares complete vocabulary rows and all
causally live target KV with an independent width-matched graph, checks the
accepted stream against width-one decode, and validates session admission,
pending-call bounds, rewind and reset. `STEP37_TEST_TRUNCATE=1` additionally
requires all four commit lengths by stopping the first cycles early; use the
arithmetic fixture for that check. Stop the owned sidecar after the workers.


## CPU image pixels

`make_pixel_vectors.py` executes the pinned official `GPUToTensor`,
`Step3VisionProcessor` and `ImagePatcher` classes directly. It needs Pillow
12.3.0, Torch 2.11.0 and torchvision 0.26.0. Its JSON records source and
pixel hashes plus library versions. It writes full RGB/CHW float crops to
scratch and four tiny model-free interpolation cases to `resize-vectors.json`.

```sh
python tests/fixtures/step37/make_pixel_vectors.py "$PROCESSING_STEP3" "$PIXEL_REFERENCE"
STEP37_PIXEL_REF="$PIXEL_REFERENCE" cargo test -p ds4-core --lib official_image_pixels -- --ignored --nocapture
```

Use an absolute `PIXEL_REFERENCE` path. Nine RGB inputs (33×27 through
3041×777) cover padding, aspect ratios, the 728-pixel boundary, the 0.2 crop
rounding threshold, out-of-bounds black crop rows and the 3024-pixel cap.
All 34 crops have exact RGB bytes and full normalized/resized CHW floats
within 1e-6 (observed maximum 4.77e-7). Geometry separately matches 28 cases.
The RGB resize follows Pillow's 22-bit coefficients and per-pass rounding;
float resize follows Torch's separable bilinear antialias coefficients.

Token admission (maximum 8192) and each intermediate RGB buffer's 128 MiB
limit precede pixel allocation. This is a per-buffer bound, not a total
process-memory limit. This gate does not exercise encoded-image decoding,
EXIF, GPU vision or generated image answers.


## F16 vision encoder

The native component consumes normalized CHW crops and executes all 47
ViT blocks, two stride-2 padded convolutions and the 4096-wide projector.
`make_vision_vectors.py` uses actual GGUF weights, the original
`EncoderRope2D` methods and independent PyTorch operations. Its default
contract matches the native GEMM: round inputs to F16, accumulate and output
F32. `--contract fp32` is an explicit unrounded-input diagnostic. TF32 is
disabled and reference attention uses the F32 math SDPA backend.

```sh
make tests/test_step37_vision test-step37-vision-ops CUDA_ARCH=sm_121
# Python needs the pinned gguf-py on PYTHONPATH. Run every GPU job serially
# under host_memory_guard.py: 20 GiB for Python, 12 GiB for native.
python tests/fixtures/step37/make_vision_vectors.py "$STEP37_VISION" "$VISION_ENCODER" "$PIXELS_504" "$REF_504" --edge 504
python tests/fixtures/step37/make_vision_vectors.py "$STEP37_VISION" "$VISION_ENCODER" "$PIXELS_728" "$REF_728" --edge 728
tests/test_step37_vision "$STEP37_VISION" "$REF_504" 504
tests/test_step37_vision "$STEP37_VISION" "$REF_728" 728
# Identical reference input at every attention/MLP/conv checkpoint:
tests/test_step37_vision "$STEP37_VISION" "$REF_504" 504 --local
tests/test_step37_vision "$STEP37_VISION" "$REF_728" 728 --local
compute-sanitizer --tool memcheck --error-exitcode 99 tests/test_step37_vision_ops
```

The locked pixel inputs are `case-7-1.f32` (504) and `case-0-0.f32` (728)
from the CPU pixel generator. The native gate compares every attention and
MLP residual, both downsamplers and all final feature values. Relative RMS
limits are 1% for the complete trajectory and 0.1% for matched-input replay.
It also rejects invalid crop shapes and out-of-range native weight spans.
Scratch is 130,460,544 bytes (504) or 272,195,456 bytes (728); these figures
exclude weights and the CUDA backend's shared scratch.

The small operator gate covers CHW patches, HWC downsampling, black padding,
position interpolation, both full coordinate grids, QuickGELU and residual
scales on a non-default stream. Under the production fast-math flags it
exposed `sincosf` substitution error at angle 40. Step now retains the
[documented accurate libdevice call](https://docs.nvidia.com/cuda/archive/12.6.3/libdevice-users-guide/__nv_sincosf.html)
and the source's fixed F32 frequency cache. The unchanged small gate and
compute-sanitizer pass. This encoder component does not yet qualify API
image input, multimodal MTP or generated visual answers.
