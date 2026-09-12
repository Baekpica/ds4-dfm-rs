# Step 3.7 integration status

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

This is a host preflight/bind plan, not native model execution. Production
family routing remains unsupported until the native ABI, gated mixed-head
attention, sigmoid MoE routing, RoPE frequency factors, tokenizer, Step vision
and external MTP paths are implemented and compared with the supplied fixtures.
No new CUDA owner, eager weight copy or inference FFI is introduced.

The Spark handoff supplies BF16 full-vocabulary logits and real image fixtures;
its README records that the MQ83 output comparison has not run. Remaining gates:

- Native main forward, same-artifact oracle logits, short greedy decode and KV transitions.
- Official tokenizer/Jinja, tool output parsing and Rust server API checks.
- External MTP prediction, acceptance and rejected-prefix rollback.
- Vision processing, encoder/projector execution and real document/chart requests.
- Guarded GB10 residency, context/bank admission and prefill/decode measurements.
- Final repository documentation, verified HF model card update and GitHub PR.
