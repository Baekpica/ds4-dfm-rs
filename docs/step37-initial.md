# Step 3.7 integration status

The goal includes native Rust-host serving with MTP and still-image input,
Prefill and Decode optimization, documentation/HF model card updates, and a
PR after validation. The September 13 scope extension permits improvements
during implementation. The owner's later numerical policy accepts arithmetic
differences consistent with Mixed Quant when they do not materially affect
generated output. Compare logits, tokens and representative answer quality;
a fixed cross-engine error threshold alone is not a release blocker. Preserve
structural KV, position and media-layout correctness. Report measured fresh-
process prefill/decode and MTP controls separately. Full-model verification
may stop Qwen. The latest owner instruction cancels
automatic restoration; leave Qwen stopped and focus on Step implementation.
Its running binaries and configuration were backed up before shutdown.
The HF update may include required tokenizer, Jinja and processor assets;
preserve upstream revisions and verify uploaded bytes.

`ds4_core::Step37Plan::inspect` validates the MQ83 main artifact and resolves
754 tensor bindings across nine mmap-backed shards. It preserves the per-layer
64/96 query-head schedule, full/SWA RoPE settings and separate routed/shared
clamps. Shared experts require Q8_0; routed precision follows the locked recipe.
Main loading rejects MTP sidecars, wrong source metadata, duplicate or
overlapping tensors, and incompatible dimensions/quantization.

Production open and inspect validate every shard's split index, shard count
and total tensor count. The published source revision appears only in the
first shard; siblings carry three split keys. An explicit sibling revision
must match. Metadata checks do not verify payload identity; use the published
`MQ83/SHA256SUMS` to verify the complete artifact before loading.

`Step37SidecarPlan::inspect` separately validates the Q8 MTP (55 tensors) and
F16 `step3vl` vision projector (667 tensors). It preserves all three distinct
MTP output heads, the unclamped dense predictor blocks, vision layer scales,
convolution layouts and image normalization metadata.

```sh
cargo run -p ds4-core --example step37_inspect -- /path/to/Step-3.7-Flash-MQ83-00001-of-00009.gguf
cargo test -p ds4-core --lib step37 -- --test-threads=1
```

The inspect command also accepts `[MTP_Q8 VISION_F16]` after the first main
shard. On September 13, 2026, the downloaded artifacts passed all bindings:

| Artifact | Tensors | Payload bytes |
|---|---:|---:|
| MQ83 | 754 | 83,001,512,448 |
| MTP Q8 | 55 | 3,702,041,856 |
| Vision F16 | 667 | 3,972,791,296 |

These are tensor payload sizes, excluding GGUF metadata and alignment.

The small `make test-step37-primitives CUDA_ARCH=sm_121` CUDA gate passes on
GB10: post-SiLU clamps, biased sigmoid routing, 64/96-head output gating,
Q/K normalization and split-half RoPE. It replays changed device positions
through 262,143 on a non-default stream. The CUDA backend object also builds.
These component checks do not establish complete model logits or performance.

Rust family selection and native descriptors now bind all 754 main and 55
MTP tensors. Main text execution now uses the public native session behind
the Rust host. Step accepts one full CUDA model; distributed slices, other
backends and DSpark sidecars remain guarded.
The main shape has 45 layers; the three external predictor blocks must not
be subtracted from that count.

The standalone eager CUDA forward executes the MQ83 main model. Eight locked
text fixtures (24–45 tokens, prefill cap 64, eight decode evaluations) produce
finite full-vocabulary logits and the same 72 greedy choices as the pinned
StepFun llama.cpp, with both FA and diagnostic F32 attention controls.
Cross-engine logit parity is **not** qualified: the strict 3% relative-RMS
criterion fails on 56/72 rows against FA; the worst error is 17.60% (code).
Changing attention precision within the oracle also changes logits (up to
13.16% across this suite). Keep these differences explicit.

Investigation fixed the native RMS epsilon (1e-6 → the source's 1e-5). A
45-layer replay with identical reference inputs bounds local attention error
at 3e-5 and FFN output error at 0.82%; its final logits have 0.82% relative
RMS error. The diagnostic oracle keeps F16 KV storage, casts attention inputs
to F32 and disables TF32; its patch is retained beside the test driver.
This isolates local arithmetic from accumulated trajectory differences.

At 832 tokens with 64-token chunks, the SWA ring and full-history KV control
have byte-identical full-vocabulary logits and all 45 layers' live KV rows.
Rewinding and replaying the last chunk preserves those bytes. This is a
structural cache fixture, not a throughput or model-quality workload.
Startup timings include lazy weight materialization and are not performance
comparisons. The test protocol is in
[the fixture README](../tests/fixtures/step37/README.md).

The native session gate compares lazy allocation, prefix extension, decode,
rewind/rebuild and reset against the standalone graph at context 848 with
64-row chunks. Full-vocabulary logits and 45-layer KV match byte for byte;
its session allocation is exactly the estimated 158,151,040 bytes. Repeated
rewinds cannot move the physical retention boundary. Invalid tokens preserve
valid state; failed GPU work and strict rewinds invalidate output readers.
Rust mirrors the checkpoint and generation transitions.

The real Rust CLI also completes the arithmetic raw-token probe with the
same nine greedy choices. This exercises default aligned artifacts and the
enforced memory governor. It exposed and fixed a missing boot-lease update
in raw-span promotion; the regression also preserves the settled census after
boot. Model-source residency at this short context is 83.35 GiB including
additive artifacts. This is not a maximum-context or throughput qualification.

The Rust tokenizer matches the pinned upstream tokenizer on 56 text, Unicode,
tool and media-marker inputs. The unchanged official Jinja matches Python
Jinja on 20 text/history/image/tool/observation cases across four effort
values. The adapter adds the template's `fromjson` filter and rejects malformed
tool JSON. These checks establish input compatibility, not generated output.

Step has an explicit Rust output protocol: its thinking and function XML
share the existing Qwen output parser; input remains the official Step Jinja.
The server passes all host parity tests and 36 live requests at context 4096:
Chat Completions, Messages and Responses, each with thinking disabled/high,
text followups, buffered/streamed tools and tool-result continuation. The
normalization gate also fixed a duplicated Responses tool discriminator.
The live fixture is `tests/chat_template_live.py`, with output caps 128/256.
These are single-request text gates, not batching, vision or MTP qualification.

The pinned upstream tokenizer JSON, configuration, special-token map and
Jinja are available in HF commit
[`9acdcd0`](https://huggingface.co/Baekpica/Step-3.7-Flash-Mixed-Quant-GGUF/commit/9acdcd0e817a029ec486d7fe77fc24be537fdd43).
The MQ83 directory also contains the template/configuration for discovery beside
the shards. All seven uploaded files, including `provenance/input-assets.json`,
passed remote byte verification. This asset upload makes no inference claim.

The native MTP component executes all three Q8 predictor blocks and their
separate heads. Against independent CPU Q8_1 equations with identical input
at each block, hidden relative RMS errors are 0.92%, 0.82%, 0.75%; head errors
are 1.14%, 1.28%, 1.17%, with all three greedy choices matching. A separate
FP32-input control chains all three blocks with at most 0.017% hidden error
(the one-row head still quantizes activations; head error is at most 0.53%).
All kept-prefix lengths 0–7 across a wrapped 512-row window preserve live
KV and next hidden bytes exactly in both controls.

The uncontrolled synthetic Q8 trajectory is not a 3% logit-parity pass:
re-quantization accumulates differences, reaching 6.66% against the CPU Q8_1
chain. The CPU reference does not reproduce MMVQ reduction order. Keep this
diagnostic distinct from matched-input operator checks.

Rust now attaches the validated 55-tensor sidecar and owns greedy acceptance
and EOS handling. Native sessions retain three target hidden rows, warm the
stable prefix in all three predictors, draft up to three tokens and verify
up to four target rows. Commit shortens the widened KV rings and uses the
accepted row's existing hidden output; it does not re-forward accepted tokens.
Failed GPU work invalidates both predictors and the target, with one native
and Rust generation transition.

The arithmetic (24-token) and wrapped-ring (832-token) fixtures each pass
32 generated tokens with an owner-imported MTP Q8 sidecar. Every trial row's
complete vocabulary and the committed 45-layer live KV are byte-identical to
a width-matched independent target graph. All 64 accepted tokens match a
separate width-one, MTP-off control. Together the fixtures exercise commit
lengths 1–4, pending-operation rejection, invalid commit bounds, no-op sync,
rewind/rebuild and reset. Allocations exactly match estimates: 87,308,480
and 208,069,376 bytes at contexts 60 and 868, respectively. This is correctness
evidence, not an MTP throughput claim.
The Rust CLI also loads the owner-backed sidecar with the default memory
governor, uses `--mtp-draft 3` at context 512, and returns
exactly `4` with a normal stop for the arithmetic chat request.

The sidecar-attached Rust server also passes the same 36 API requests at
context 4096, covering buffered/SSE text, tools and continuations with
thinking disabled/high. `DS4_MTP_SPEC_LOG` confirms 42 greedy MTP cycles.
The existing server thinking policy uses nonzero-temperature sampling and
therefore ordinary decode; that portion verifies sidecar coexistence rather
than speculative acceleration. `DS4_MTP_SPEC_DISABLE` retains the server's
ordinary greedy decode fallback.

The three-depth historical warm policy is causally aligned by
`token[t+depth+1]`, with a uniform stable frontier `N-3`. Pinned vLLM warms
only depth 0 on its first pass, then runs depths 1 and 2 on one row each.
This runtime's fuller predictor history is not a claim of vLLM draft-logit
parity. Target verification determines the committed stream.

The initial MTP gates above used raw-layout BASE dispatch: startup previously
disabled local artifacts whenever any IPC manifest was present. The optimized
path now builds BASE artifacts when the manifest supplies only MTP. Its
separate correctness, quality and fresh-process measurements are recorded in
[the first optimization report](step37-optimization-2026-09-13.md).

A later campaign on that landed binary is
[step37-optimization-2026-09-13-r2.md](step37-optimization-2026-09-13-r2.md).
Default Step prefill chunk is 2048. On the same 2048+64 protocol, one GB10
measures median Prefill **1243 tok/s** and MTP Decode **22.43 tok/s**
(ordinary Decode 19.83). 16K ordinary Prefill does not fall off (1269 tok/s).

The image crop planner matches 28 independent official Python cases, including
thin-image padding, the 728/3024 limits, crop order and media token counts.

The Spark handoff also supplies BF16 logits and real image fixtures. These
are distinct from the same-MQ83 runtime comparison above; broad MQ83-versus-BF16
fidelity and long multimodal conversations remain unqualified. The supported
serving limits and commands are below.


CPU image preprocessing now follows the official crop/padding and two
separate interpolation contracts: Pillow RGB8 bilinear before normalization,
then Torch CHW float bilinear antialias. Across nine RGB inputs and 34 crops,
all intermediate RGB bytes match exactly and every final float differs by
at most 4.77e-7. The gate also covers the 3024-pixel cap and black rows from
out-of-bounds crops. This component does not qualify image answers.


The native F16 vision component now executes the complete 47-layer encoder
and projector for both crop sizes. Against independent PyTorch equations
with the same F16-rounded GEMM input contract, complete 504/728 trajectories
have final-feature relative RMS differences of 0.103%/0.101%; same-input
attention/MLP/convolution replay stays below 0.003%. Every residual stage
and all 81/169 final 4096-wide features are checked. The small kernel gate
also passes compute-sanitizer with production fast-math flags after retaining
accurate RoPE trigonometry. API image answers remain a separate live gate.
These are encoder component checks, not generated-image or throughput claims.


Rust now preflights and decodes bounded 8-bit PNG/JPEG data, applies EXIF
orientation and constructs exact official patch/base token replacements.
All image spans and the request-wide 8192-token budget are checked before
pixel allocation. Prepared crops carry validated absolute token offsets in
patch-first/base-last order. The full 34-crop pixel gate passes through this
preparation path. Rust owns bounded encoded-image decoding; the narrow FFI
borrows CHW buffers only for synchronous native encoding/refill. The server
substitutes complete image spans rather than repeating a placeholder token.

The native session shares resident GPU image features between target and MTP
next-token embeddings. Image sync always refills KV; ordinary text sync
rejects image placeholders. The reusable maximum-size encoder workspace uses
272,195,456 bytes, plus up to 8192 x 4096 F32 feature values; admission and
committed allocation include both. The 728-to-504-to-728 shape gate checks
all attention segment bounds. A full 504 encoder replay with reused scratch
retains the same 0.103% projected-feature difference as its fixed-size run.

A 278-token prompt with one 504 and one 728 normalized crop passes 32
MTP-generated tokens against an independent width-one control. Every trial
vocabulary and committed 45-layer live KV match the width-matched reference
byte-for-byte; 22 of 29 proposed draft tokens are accepted. The session's
419,545,728-byte allocation matches its quote. Changed pixels with unchanged
tokens force target/MTP refill and alter logits. A rejected vision GEMM after
input upload poisons both frontiers; malformed spans preserve the old state.
The same gate at 1086 prompt tokens crosses the SWA ring and also passes 32
generated tokens, unchanged-token image replacement and encoder failure.
Draft acceptance is 22/27; allocation is 511,132,288 bytes at context 1122.
The native 128-case splice/shape/admission gate, 95 Rust core tests,
43 server generation tests and affected server/catalog/session parity gates
also pass. These gates do not measure speed.


The Rust server at context 4096 with `--vision` and `--mtp-draft 3` passes
21 image/text requests: red/blue recognition and subsequent text/follow-up
on all three APIs, plus the fixed dashboard, invoice, Earth photograph and
four-image suite in `tests/step37_images_live.py`. The 256/1024/1920-pixel
screenshots retain their original files. Changed Failed=3/9 screenshots
produce the corresponding counts; invoice total/tax are $385/$35. Full-history
image continuations pass Chat Completions and Responses; four-image SSE
passes Messages with the correct image order. Responses retains its existing
full-input replay contract. Greedy image decoding uses MTP. These are bounded
functional/output checks, not a broad vision-quality or throughput benchmark.
Step currently uses the serial lane; multi-sequence graphs are unavailable.

A separate text benchmark completes 16384 prompt tokens and 1024 generated
tokens at context 17416 with MTP draft 3. Default and graph-disabled runs have
identical full-vocabulary frontier logits and all 1024 tokens. This establishes
that bounded workload, not 262144-token capacity or long image conversations.

All [CONTRIBUTING host checks](../CONTRIBUTING.md) pass: formatting, clippy,
eight C/Rust parity targets, serialized workspace tests and all-target checks.
Native operator, artifact-scope and image/MTP state results are recorded in
[the first optimization report](step37-optimization-2026-09-13.md).
Post-landing Prefill/Decode numbers are in
[the follow-up campaign](step37-optimization-2026-09-13-r2.md).

## Running the supported artifact

Download `MQ83/`, `MTP/` and `vision/` from
[the model repository](https://huggingface.co/Baekpica/Step-3.7-Flash-Mixed-Quant-GGUF).
Keep the tokenizer configuration and Jinja files next to the main shards.
The Rust processor reads image geometry from the validated model contract;
it does not require a Python processor at runtime.

After `make cuda-spark`, start the MTP owner in one terminal. Use an absolute
manifest path so the second terminal can resolve the VMM broker:

```sh
STEP_MODEL=/path/to/Step-3.7-Flash-Mixed-Quant-GGUF
python3 tools/host_memory_guard.py --max-gib 6 --high-gib 5 \
  --timeout 0 --log scratch/step-owner-guard.jsonl -- \
  ./ds4_weight_server --backend vmm --scope mtp \
  --base "$STEP_MODEL/MQ83/Step-3.7-Flash-MQ83-00001-of-00009.gguf" \
  --mtp "$STEP_MODEL/MTP/Step3.7-flash-mtp-Q8_0.gguf" \
  --manifest /tmp/ds4-step37.ipc
```

Wait for the owner to report `ready`, then start a bounded worker:

```sh
STEP_MODEL=/path/to/Step-3.7-Flash-Mixed-Quant-GGUF
DS4_CUDA_WEIGHT_IPC_MANIFEST=/tmp/ds4-step37.ipc \
DS4_CUDA_WEIGHT_IPC_SCOPE=mtp \
python3 tools/host_memory_guard.py --max-gib 100 --high-gib 96 \
  --timeout 0 --log scratch/step-worker-guard.jsonl -- \
  ./ds4-server --cuda --host 127.0.0.1 --port 8000 \
  --model-id step-3.7-flash-mq83 -c 4096 --tokens 128 \
  -m "$STEP_MODEL/MQ83/Step-3.7-Flash-MQ83-00001-of-00009.gguf" \
  --mtp "$STEP_MODEL/MTP/Step3.7-flash-mtp-Q8_0.gguf" --mtp-draft 3 \
  --vision "$STEP_MODEL/vision/mmproj-step3.7-flash-f16.gguf"
```

Set `reasoning_effort: "none"` and `temperature: 0` for greedy MTP. Sampled
reasoning and forced protocol prefixes use ordinary decode. Still images
are bounded PNG/JPEG inputs, up to four per request.
Audio, distributed slices, multiple sequence banks and serialized disk KV
are unsupported for Step. Follow-up requests replay their complete history;
changed images refill the session. The benchmark also restores sweep prefixes
by replay outside timing. `kvcache_bytes=0` describes absent serialization,
not zero live KV memory.
