# Step 3.7 initial loader

`ds4_core::Step37Plan::inspect` validates the MQ83 main artifact and resolves
754 tensor bindings across nine mmap-backed shards. It preserves the per-layer
64/96 query-head schedule, full/SWA RoPE settings and separate routed/shared
clamps. Shared experts require Q8_0; routed precision follows the locked recipe.
MTP sidecars, wrong source metadata, duplicate or overlapping tensors, and
incompatible dimensions/quantization are rejected.

```sh
cargo run -p ds4-core --example step37_inspect -- /path/to/Step-3.7-Flash-MQ83-00001-of-00009.gguf
cargo test -p ds4-core --lib step37 -- --test-threads=1
```

This is a host preflight/bind plan, not native model execution. Production
family routing remains unsupported until the native ABI, gated mixed-head
attention, sigmoid MoE routing, RoPE frequency factors, tokenizer, Step vision
and external MTP paths are implemented and compared with the supplied fixtures.
No new CUDA owner, eager weight copy or inference FFI is introduced.

The Spark handoff includes BF16/MQ83 full-vocabulary logits and real image
fixtures. Native forward parity, cache transitions, MTP rollback and measured
GB10 memory/performance remain required before enabling production routing.
