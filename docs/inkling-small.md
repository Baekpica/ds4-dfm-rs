# Inkling Small integration

Work in progress; native inference is not implemented yet. The target is
MQ85GB with the separate eight-layer MTP-BF16 draft stack, including text,
image and audio input with text output, matching the
[base model](https://huggingface.co/thinkingmachines/Inkling-Small/blob/8cc5877b44d343f88b92086aa1fb72897950f06a/README.md).
Catalog checks do not qualify serving.

## Artifact contract

- [MQ85GB](https://huggingface.co/Baekpica/Inkling-Small-Mixed-Quant-GGUF/tree/4a5b2db7de3a294af660053284337bd5424cd3e4/MQ85GB):
  six GGUF shards, 888 tensors, 85,704,616,100 payload bytes.
- [MTP-BF16](https://huggingface.co/Baekpica/Inkling-Small-GGUF/tree/01e829c5acf2c9aa8026f5b8157d980e1c20730c/MTP-BF16):
  one GGUF, 160 tensors, 4,463,824,912 payload bytes.
- Source revision: `8cc5877b44d343f88b92086aa1fb72897950f06a`.
- Architecture `inkling`; tensor layout `source-interleaved-v1`.
- [Published math/precision contract](https://huggingface.co/Baekpica/Inkling-Small-Mixed-Quant-GGUF/blob/main/INKLING-CONTRACT.md).

Rust identifies the family, validates embedded source/processor configuration,
and resolves exact MQ85GB and MTP layouts without reading weight payloads.
Both shared experts, image/audio weights, relative projections and causal
convolution tensors are required. MTP metadata must declare the same source
revision, a sidecar role and BF16 recipe. Main loading rejects MTP-only files.

`ModelFamily::Inkling` is 7 and `Variant::InklingSmall` is 9 in the host
catalog. Native enum/shape/graph integration is still pending. Model opening
returns an explicit unimplemented-inference error before entering native code.

The Rust tokenizer reads the embedded source JSON: 199998 BPE entries,
60 fixed special IDs, Unicode segmentation and `ignore_merges=true`.
It preserves literal special strings in ordinary text and recognizes them
in rendered chat. Token output ends at ID 200006; message and thinking
boundaries do not stop generation. Basic role messages and the four thinking
effort levels match independently rendered source-template fixtures. Server
tool/reasoning streams, media insertion and REPL assembly remain pending.
The tokenizer has 200058 entries; native logits must exclude the weight
matrix's padding through row 201023.

## Remaining qualification

1. Connect source chat/reasoning/tool rendering, streamed content markers and
   REPL effort placement to the CLI and server.
2. Bind native weights through the existing Rust → bridge → CUDA boundary.
   Implement relative GQA, interleaved SwiGLU, sigmoid top-6 routing with two
   shared sink weights, and all four residual causal convolutions per layer.
   Preserve BF16 boundaries, FP32 router/reductions and 1/128 attention scale.
3. Prove chunk/decode and captured/eager full-vocabulary logits and greedy
   parity. Cover local-ring wrap, global attention and convolution history.
4. Implement HMLP image and 16-kHz dMel audio preprocessing/encoding, feature
   insertion and API transport. Compare processed inputs and encoder outputs
   to pinned references; run real image/audio requests and text follow-ups.
5. Connect all eight BF16 MTP layers, hidden-state chaining, draft verification
   and accepted-prefix rollback for KV and convolution state. Compare MTP
   off/on tokens and committed state across accept/reject cases.
6. Qualify VMM owner/worker loading, memory admission, session reuse/rewind,
   persistence, concurrent serving, API behavior and end-to-end performance
   on the requested artifacts. Update supported-family docs only after this.

Reference code is pinned to SGLang
`03d06a764e4a83268eefd1bafc676418f7269c89` and Transformers
`cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55`. SGLang `InklingMTPLayer` chains
the raw block hidden state; it applies main embedding norm before draft
embedding norm and concatenates hidden then embedding. Draft global layers
are 1 and 3; main global layers are 5, 11, 17, 23, 29, 35 and 41.

## Checks

```sh
cargo test -p ds4-core --test inkling_catalog --locked
cargo test -p ds4-core --lib inkling --locked
make -j1 test-catalog-parity test-tokenizer-parity
cargo check --workspace --all-targets --locked
```

The artifact tests require files and are explicitly ignored by ordinary
model-free tests. Run them intentionally; they do not allocate GPU weights:

```sh
export INKLING_ARTIFACT_DIR=/home/sunghoon/workspace/ds4-exaone/models/Inkling-Small-Mixed-Quant-GGUF
cargo test -p ds4-core --lib attach_inkling_mtp_artifact --locked -- --ignored
cargo test -p ds4-core --test inkling_catalog checks_downloaded_artifacts --locked -- --ignored
cargo test -p ds4-core --test inkling_tokenizer --locked -- --ignored --test-threads=1
```

September 9: all six MQ85GB shard hashes, MTP SHA-256, real main/MTP
metadata and tensor binding passed. The tokenizer matched 654 source
vectors in both ordinary and rendered-chat modes, including decoded bytes;
all four basic chat-template effort fixtures passed. Numerical and serving
gates remain unverified. Qwen's resident weight owner was stopped with user
authorization before native testing; inspect current ownership before loading.
