# Naive-N0.5-Flash

Integration is in progress. The Rust catalog, MQ87 metadata/layout validator,
NFC/Qwen2 tokenizer and official Jinja input template are implemented.
CUDA execution, serving, cache reuse and DSpark are not qualified yet.
Model opening currently returns an explicit error before native allocation.

All four downloaded shards and their provenance/manifest pass SHA-256 checks.

## Artifact contract

- Main: [NaiveAI/Naive-N0.5-Flash](https://huggingface.co/NaiveAI/Naive-N0.5-Flash),
  revision `0235b3b5ff27422b1f57cdc2acddfaf643e08356`.
- Mixed artifact: [MQ87](https://huggingface.co/Baekpica/Naive-N0.5-Flash-Mixed-Quant-GGUF),
  revision `65b235a48e87acaf7b806b96870378c9c68a654c`.
  Four shards, 613 GGUF tensors, 86,960,547,072 file bytes.
- Architecture: `naive_n05_flash`; 48 layers, hidden 4096, vocabulary 152576,
  one dense FFN followed by 47 top-8 MoE layers with 256 experts each.
- Attention: 39 SWA layers with window 128 and eight KV heads; nine DSA layers
  at `0,5,11,17,23,29,35,41,47` with four KV heads. Q/K width 192, V width 128.
- DSA: direct hidden projections, 16 indexer heads of width 128, stable
  top-2048 selection and native per-row E4M3 rounding. Full GQA and indexer
  histories remain necessary.
- Numerical controls: RMSNorm epsilon `1e-5`, split-half RoPE on 64 dimensions,
  theta 10000 for SWA and 10000000 for DSA, V scale 0.707, SWA-only sinks,
  sigmoid routing with correction bias and normalized unbiased mixing weights.
- Main has no embedded MTP. Its companion DSpark uses five SWA/1024 layers,
  an anchor plus six proposals, eight target hidden taps, Markov correction
  and confidence tensors. Draft source revision:
  `b2b8ee9f5d6b3fd1dfba113d3a363138e37c83b0`.

The validator requires the MQ87 types and split Q/K/V layout. MiMo metadata,
global attention, fused QKV and DeepSeek MLA/compressed KV are different
contracts.

## Current checks

The September 30 host checks cover identification, metadata rejection,
all 613 tensor names/types/dimensions, eight official-template vectors and
their original-tokenizer IDs. The tokenizer check used a 13,014,912-byte
header capture while the main weights were downloading. This is directory
and input-protocol evidence, not full-file integrity or inference evidence.

CUDA primitive checks on GB10 also pass: E4M3 finite codes/scales and midpoint
ties, device-position replay, affine indexer LayerNorm, RMSNorm epsilon,
partial NeoX RoPE, signed index scores, causal stable top-k through a 1M
history, and unbiased sigmoid routing. BF16 attention agrees with an
independent source-equation reference: SWA maximum absolute difference 0;
DSA difference 0.0009765625 for the 2049-key fixture. This tests primitives,
not full-model inference or long-context serving. GPU clocks were set to
300–2200 MHz before these checks.
The native split-projection binder also matches all 613 source-directory
entries. CUDA memcheck reports zero errors for the primitive fixtures.

The DSpark inspector accepts the downloaded Q8 artifact and rejects changed
source/target revisions, tap ordering, mask/block semantics and every tensor
dimension. Its 63-tensor directory and SHA-256
`193b96b39d132656635bc4f6a09ad91c64aed7a52c08f46dabe0e3847cef8a8b`
were checked locally. This validates the sidecar contract, not draft execution.

The independent five-layer CUDA draft now runs the real Q8 file against
synthetic source-equation fixtures (four tap rows, seven noise rows at
positions 17, 1048 and 1048569). Hidden cosine is 0.99882–0.99925 with
relative L2 differences 0.0388–0.0486; Markov bias cosine is 0.9999967.
MMQ activation quantization and F32 Q8 weight reconstruction differ from the
reference's BF16 decoded weights. Learned-mask perturbations leave the
output byte-identical. Raw confidence is evaluated without a sigmoid.
These are isolated graph checks; they do not establish target-token parity,
acceptance or acceleration. Independent local attention checks have maximum
absolute difference 0.00006103515625 at the 1024 and 1M position fixtures.
Device-backtrace CUDA memcheck reports zero errors for the real draft fixture.

The eager main graph and serial session/snapshot paths are implemented behind
the model-open guard. A weight-free GPU test matches allocator bytes to the
quote and restores 2051 rows of DSA K/V, index codes/scales and wrapped SWA
from prefill chunk 32 to chunk 7 byte-for-byte. Truncated snapshots invalidate
the frontier. State tests reject logits after invalidation or a mismatched
frontier. These checks do not establish full-model correctness.

Persistent banks share forward scratch and keep independent DSA/SWA/indexer
histories. The weight-free common bank API passes a three-bank allocation,
full-prefix copy, partial fork, self-restore and snapshot round trip. Eight
lazy SWA checkpoint slots need up to 204,472,320 bytes plus page alignment;
each checkpoint preserves 25,559,040 bytes of displaced SWA rows. Full DSA
KV and indexer histories are copied from the source bank at the chosen cut.
Live serving, partial reuse and disk restart remain to be qualified.
Bank memcheck passes with zero errors using `--show-backtrace device`.
The default host backtrace collector crashes in `libgcc _Unwind_Backtrace`
at CUDA context initialization on this test; device trace and memory checks
remain enabled in the passing run.

The host quote and native geometry tests include full DSA K/V and indexer
history. With prefill chunk 2048 and one bank, the planned 1M allocation is:

| Component | Bytes | GiB |
| --- | ---: | ---: |
| DSA BF16 K/V | 24,159,191,040 | 22.500 |
| SWA BF16 rings | 434,304,000 | 0.404 |
| Indexer E4M3 codes and F32 scales | 1,245,708,288 | 1.160 |
| Activations, scores and top-k workspace | 1,837,994,240 | 1.712 |

The four GGUF files add 80.988 GiB of mapped files; GPU weight residency,
derived tensors, allocator overhead and host headroom need live measurement.
These are planned sizes, not evidence that 1M serving fits. Disk KV stores
checkpoints and does not replace an active bank's GPU history.

```sh
cargo test -p ds4-core --test naive --locked
cargo test -p ds4-core --test naive_draft --locked
make test-naive-memory
make test-naive-bind
make test-naive-draft-bind
make CUDA_ARCH=sm_121 test-naive-draft-ops
make CUDA_ARCH=sm_121 test-naive-primitives
make test-naive-state
make CUDA_ARCH=sm_121 test-naive-graph
make CUDA_ARCH=sm_121 test-naive-banks
# Optional real vocabulary gate; reads only the GGUF header.
NAIVE_TOKENIZER_GGUF=/absolute/path/to/first-shard.gguf \
  cargo test -p ds4-core --test naive --locked
```

## Remaining gates

CUDA forward must match the mixed-weight reference, including indexer
rounding, stable ties and the 2048-to-2049 history transition. Session state
must preserve GQA K/V, SWA rings and indexer history together through partial
reuse, bank changes, snapshots and disk KV. DSpark needs matched off/on
token/state checks, actual acceptance and fresh-process speed evidence.

Qualify 262144, 524288 and 1048576 contexts separately: admission, full prompt
processing, distant-content retrieval, continuation, peak memory and actual
decode. Requested capacity or a listening server alone is insufficient.
Record bank count and draft mode for each result. Only completed DGX gates
may become serving-capability or Hugging Face performance claims.
