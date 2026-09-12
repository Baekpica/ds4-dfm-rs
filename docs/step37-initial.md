# Step 3.7 integration status

The goal includes native Rust-host serving with MTP and still-image input,
Prefill and Decode optimization, documentation/HF model card updates, and a
PR after validation. The September 13 scope extension permits improvements
during implementation. Retain only measured candidates with logits, greedy
token and KV parity; report fresh-process prefill/decode and MTP controls
separately. Full-model verification may temporarily stop Qwen, with its exact
launch configuration and API service restored afterwards (owner approval).
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
MTP tensors. Native loading retains a clear execution guard while the forward
path is unfinished. The main shape has 45 layers; the three external predictor
blocks must not be subtracted from that count.

The Rust tokenizer matches the pinned upstream tokenizer on 56 text, Unicode,
tool and media-marker inputs. The unchanged official Jinja matches Python
Jinja on 20 text/history/image/tool/observation cases across four effort
values. The adapter adds the template's `fromjson` filter and rejects malformed
tool JSON. These checks establish input compatibility, not generated output.

The Spark handoff supplies BF16 full-vocabulary logits and real image fixtures;
its README records that the MQ83 output comparison has not run. Remaining gates:

- Native main forward, same-artifact oracle logits, short greedy decode and KV transitions.
- Official tokenizer/Jinja, tool output parsing and Rust server API checks.
- External MTP prediction, acceptance and rejected-prefix rollback.
- Vision processing, encoder/projector execution and real document/chart requests.
- Guarded GB10 residency, context/bank admission and prefill/decode measurements.
- Profile and optimize Prefill and Decode; keep before/after throughput and
  numerical evidence, including MTP-on/off and multimodal workloads.
- Final repository documentation, verified HF model card update and GitHub PR.
