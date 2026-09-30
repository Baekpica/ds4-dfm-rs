# Naive-N0.5-Flash

Integration is in progress. The Rust catalog, MQ87 metadata/layout validator,
NFC/Qwen2 tokenizer and official Jinja input template are implemented.
CUDA execution, serving, cache reuse and DSpark are not qualified yet.
Model opening currently returns an explicit error before native allocation.

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

```sh
cargo test -p ds4-core --test naive --locked
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
