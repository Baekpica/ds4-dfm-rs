# Step 3.7 integration status

The goal includes native Rust-host serving with MTP and still-image input,
Prefill and Decode optimization, documentation/HF model card updates, and a
PR after validation. The September 13 scope extension permits improvements
during implementation. The owner's later numerical policy accepts arithmetic
differences consistent with Mixed Quant when they do not materially affect
generated output. Compare logits, tokens and representative answer quality;
a fixed cross-engine error threshold alone is not a release blocker. Preserve
structural KV, position and media-layout correctness. Report measured fresh-
process prefill/decode and MTP controls separately. Full-model verification may stop Qwen. The latest owner instruction cancels
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

The current `--scope mtp` owner path suppresses in-process base artifacts
because startup checks for any manifest, not which source it covers. These
MTP gates therefore use raw-layout main dispatch. Profile this fallback and
validate source-specific artifact construction during the optimization phase;
do not compare these runs with aligned main-only results as identical paths.

The image crop planner matches 28 independent official Python cases, including
thin-image padding, the 728/3024 limits, crop order and media token counts.
Pixel conversion and vision inference are still pending.

The Spark handoff supplies BF16 logits and real image fixtures. BF16 outputs
are separate from the MQ83 oracle comparison above. Remaining gates:

- Longer Rust continuations and further cross-engine drift investigation.
- Longer MTP continuations and measured MTP-on/off performance controls.
- Vision processing, encoder/projector execution and real document/chart requests.
- Guarded GB10 residency, context/bank admission and prefill/decode measurements.
- Profile and optimize Prefill and Decode; keep before/after throughput and
  numerical evidence, including MTP-on/off and multimodal workloads.
- Final repository documentation, verified HF model card update and GitHub PR.


CPU image preprocessing now follows the official crop/padding and two
separate interpolation contracts: Pillow RGB8 bilinear before normalization,
then Torch CHW float bilinear antialias. Across nine RGB inputs and 34 crops,
all intermediate RGB bytes match exactly and every final float differs by
at most 4.77e-7. The gate also covers the 3024-pixel cap and black rows from
out-of-bounds crops. Encoded-image decoding and native vision integration
remain pending; this component does not qualify image answers.


The native F16 vision component now executes the complete 47-layer encoder
and projector for both crop sizes. Against independent PyTorch equations
with the same F16-rounded GEMM input contract, complete 504/728 trajectories
have final-feature relative RMS differences of 0.103%/0.101%; same-input
attention/MLP/convolution replay stays below 0.003%. Every residual stage
and all 81/169 final 4096-wide features are checked. The small kernel gate
also passes compute-sanitizer with production fast-math flags after retaining
accurate RoPE trigonometry. API image wiring and multimodal MTP are pending.
These are encoder component checks, not generated-image or throughput claims.


Rust now preflights and decodes bounded 8-bit PNG/JPEG data, applies EXIF
orientation and constructs exact official patch/base token replacements.
All image spans and the request-wide 8192-token budget are checked before
pixel allocation. Prepared crops carry validated absolute token offsets in
patch-first/base-last order. The full 34-crop pixel gate passes through this
preparation path. Native session injection and server image routing remain
pending.
