# Step 3.7 integration status

The goal includes native Rust-host serving with MTP and still-image input,
Prefill and Decode optimization, documentation/HF model card updates, and a
PR after validation. The September 13 scope extension permits improvements
during implementation. Retain only measured candidates with logits, greedy
token and KV parity; report fresh-process prefill/decode and MTP controls
separately. Full-model verification may stop Qwen. The latest owner instruction cancels
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
backends and draft sidecars remain guarded until their own implementation.
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

The pinned upstream tokenizer JSON, configuration, special-token map and
Jinja are available in HF commit
[`9acdcd0`](https://huggingface.co/Baekpica/Step-3.7-Flash-Mixed-Quant-GGUF/commit/9acdcd0e817a029ec486d7fe77fc24be537fdd43).
The MQ83 directory also contains the template/configuration for discovery beside
the shards. All seven uploaded files, including `provenance/input-assets.json`,
passed remote byte verification. This asset upload makes no inference claim.

The image crop planner matches 28 independent official Python cases, including
thin-image padding, the 728/3024 limits, crop order and media token counts.
Pixel conversion and vision inference are still pending.

The Spark handoff supplies BF16 logits and real image fixtures. BF16 outputs
are separate from the MQ83 oracle comparison above. Remaining gates:

- Longer Rust continuations and further cross-engine drift investigation.
- Official tokenizer/Jinja, tool output parsing and Rust server API checks.
- External MTP prediction, acceptance and rejected-prefix rollback.
- Vision processing, encoder/projector execution and real document/chart requests.
- Guarded GB10 residency, context/bank admission and prefill/decode measurements.
- Profile and optimize Prefill and Decode; keep before/after throughput and
  numerical evidence, including MTP-on/off and multimodal workloads.
- Final repository documentation, verified HF model card update and GitHub PR.
