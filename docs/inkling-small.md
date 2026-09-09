# Inkling Small integration

Work in progress; Rust text and serial HTTP image/audio paths are implemented
with short artifact smoke tests. MTP and production serving qualification
remain incomplete. The target is
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
catalog and native shape. Native binding resolves all 888 main and 160 draft
tensors, including media weights, without reading their payloads; the real
GGUF descriptor gate checks each name and exactly-once coverage. Rust opening
allows a full CUDA model with the exact MTP-BF16 sidecar. It rejects distributed
slices, steering, DSpark and external media sidecars.

The Rust tokenizer reads the embedded source JSON: 199998 BPE entries,
60 fixed special IDs, Unicode segmentation and `ignore_merges=true`.
It preserves literal special strings in ordinary text and recognizes them
in rendered chat. Token output ends at ID 200006; message and thinking
boundaries do not stop generation. Basic role messages and the four thinking
effort levels match independently rendered source-template fixtures. Server
tool/reasoning streams and media insertion now have targeted checks; REPL
effort assembly remains pending.
The tokenizer has 200058 entries; native logits must exclude the weight
matrix's padding through row 201023.

The CUDA four-tap residual convolution preserves BF16 input/output boundaries
and FP32 accumulation, with a three-row history
in oldest-first order. A separate next-state buffer supports snapshots; an
in-place history update is also supported. On GB10, 18 channel/length pairs
matched the independent FP64 formula and their chunk/decode counterparts;
history also matched through seven CUDA graph replays. Compute Sanitizer
reported zero memory errors, and the existing model-family primitive suite
passed. These are synthetic component gates.

CUDA MoE primitives now implement stable sigmoid-plus-bias top-6 selection,
logsigmoid normalization across six routed and two shared logits, interleaved
SwiGLU and expert-output combination. Routed weights apply after the down
projection; shared weights apply inside SwiGLU before its BF16 cast. Independent
CPU PyTorch fixtures cover ties, selection-only bias, extreme logits and zero
scale, plus both BF16 reduction boundaries. GB10 gates passed at up to 513
router/combine rows and 257 SwiGLU rows, including width 16384. Six graph
replays changed expert IDs and weights correctly; Compute Sanitizer reported
zero errors and convolution regression passed. These component gates do not
qualify full-model numerical parity.

CUDA attention preparation projects each head's 16 relative features into
512 local or 1024 global distance bins. Global Q and relative profiles receive
the source log scale after BF16 rounding, starting beyond position 127999.
Absolute positions stay live on device during captured replay. Eight GB10
shape cases and five replays matched the FP64 formula at BF16 boundaries;
Compute Sanitizer reported zero errors.

CUDA GQA attention now reads the committed BF16 KV prefix and current K/V
without mutating the cache. A separate store commits the accepted prefix,
including chunks wider than the local ring. On GB10, 1105-token local/global
inputs matched across full, decode and 7/63/257/700-token chunks; FP64 probes
covered window and relative-extent boundaries. Fourteen captured replays per
geometry matched output and cache through changing positions and ring wrap.
Rejected K/V suffixes left accepted outputs and committed state unchanged.
Compute Sanitizer reported zero errors. These remain component gates.

BF16 RMSNorm and scale/residual primitives cover the text, attention-head
and HMLP widths. They preserve FP32 normalization/weight arithmetic before
the output cast and invalidate stale producer-Q8 data on overwritten buffers.
GB10 checks passed 18 norm shapes, exact scale/residual boundaries, seven
captured replays and the Q8 reuse regression. Compute Sanitizer reported
zero errors.

The eager graph connects all 42 layers and masks the 966 padded output rows
to negative infinity. On the actual MQ85GB artifact, five-token text and
12-token chat fixtures produced byte-identical valid-vocabulary logits,
greedy output and committed KV/convolution state across full prefill, decode
and 2/3-token chunks; the chat fixture also covered seven-token chunks. The
next greedy token after `The capital of France is` was ` Paris`.
BF16 fixed-row reductions and per-token quantized matmuls preserve this
initial numerical baseline; wider dispatches showed final-logit drift across
token widths and were excluded. Prefill performance is not optimized. This is
internal execution parity; independent full-model source parity, long-context
and captured execution remain unverified.
The five-token graph also passed Compute Sanitizer memory-access checks.
API-error reporting was disabled for that gate after the default run reported
six expected host-registration fallback errors; both logs were retained.

Native CUDA sessions support lazy allocation, exact-prefix reuse, decode,
invalidate and rewind followed by replay. The MQ85GB session gate matched
cold/reused logits and measured exactly 100,306,688 graph bytes at context 32,
matching its memory quote; host session parity also passed. The default
prefill cap is 64 (`DS4_INKLING_PREFILL_CHUNK`, range 1–2048, capped by context).
Batching, snapshots and distributed execution remain unavailable until their
Inkling-specific state paths are implemented.

Rust CLI startup loaded the actual MQ85GB through the host tensor/vocabulary
tables and completed a 28-token source-template chat prefill. All 200058
logits and 168 layer-component arrays matched the native graph byte-for-byte
with the same default aligned-Q8 artifacts. That native gate also matched
decode and 2/3/7-token chunks, including all 9977856 KV/convolution bytes.
The raw-Q8 control matched separately with `DS4_CUDA_NO_Q8_ALIGNED=1`.
Raw and aligned Q8 are different reduction paths: comparing them gave final
relative RMS 0.175 on this fixture, starting with a small layer-0 dense-MLP
BF16 difference. These checks establish matched-path host/native parity,
not independent source parity or equivalence between the two Q8 paths.

The Rust CLI projects Inkling thinking/text boundaries by token ID, preserving
ordinary text that spells those markers and resetting TTY color between
channels. A guarded MQ85GB one-shot run with context 128, temperature zero,
thinking disabled and a 32-token output cap answered `4` to
`What is 2 + 2? Reply with just the number.` without leaking control markers.
This is a short text-generation smoke; broader agent/REPL behavior remains
unqualified.

Serial HTTP text, reasoning and tools passed buffered/SSE checks across
Chat Completions, Messages and Responses at context 1024. Image requests on
all three surfaces passed red/blue identification, repeat and changed-image
requests, two-image order, and text follow-ups. PNG/JPEG preprocessing and
native HMLP stages have separate source-reference checks.

OpenAI Chat accepts `input_audio` content blocks with
`input_audio: {"format":"wav","data":"BASE64"}`. Input is 16 kHz PCM/float
WAV, with mean downmix for up to eight channels. Resampling and compressed
audio codecs are unavailable; Responses and Messages do not accept audio.
The request limit is four combined media inputs, 10 MiB per base64-decoded
payload and 20 MiB total. Media feature rows must fit the session context.
Media requests refill KV and do not store disk checkpoints, so changed media
with identical placeholder tokens cannot reuse stale features.

Rust audio fixtures require exact discrete mel codes against pinned HF
methods, with bounded pre-log energy error; nearly silent FFT bands are not
bit-exact in log space. Native tests cover audio-only and mixed sessions,
repeat/change identity and malformed inputs preserving prior state. Speech
smokes cover buffered/SSE transcription and follow-ups. A mixed audio/image
request answered both questions when they followed the media; an earlier
instruction-first prompt answered only the image question. These are scoped
smokes, not broad ASR or multimodal quality qualification.

## Remaining qualification

1. Finish REPL effort placement and broader agent integration.
2. Extend generation checks beyond short serial smokes; connect session
   persistence and serving lifecycle.
3. Prove chunk/decode and captured/eager full-vocabulary logits and greedy
   parity. Cover local-ring wrap, global attention and convolution history.
4. Broaden media quality/codec coverage beyond the tested PNG/JPEG and 16 kHz
   WAV fixtures, including longer and mixed requests.
5. Connect MTP to serial HTTP generation and qualify stop, streaming and
   continuation behavior. Extend MTP off/on checks beyond short fixtures.
6. Qualify VMM owner/worker loading, memory admission, session reuse/rewind,
   persistence, concurrent serving, API behavior and end-to-end performance
   on the requested artifacts. Update supported-family docs only after this.

Reference code is pinned to SGLang
`03d06a764e4a83268eefd1bafc676418f7269c89` and Transformers
`cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55`. The first draft receives the
target's final-normalized hidden state before division by 16. Subsequent
SGLang `InklingMTPLayer` calls chain raw draft block hidden states. Draft
embedding preparation applies main embedding norm before draft
embedding norm and concatenates hidden then embedding. Draft global layers
are 1 and 3; main global layers are 5, 11, 17, 23, 29, 35 and 41.

All eight BF16 draft blocks have a native execution path with independent
KV and four convolution histories per depth. A seven-row component fixture
uses actual sidecar weights and supplied normalized embeddings/hidden input.
Against a CPU equation oracle, per-depth relative RMS error was at most
0.00438. Full-width and 1/2/3-row chunks produced identical chained hidden
outputs and all 1212416 KV/convolution bytes. These checks exclude shared
embedding/head execution, target verification and speculative commits.
At 529 rows, 7-row chunks and decode also matched all hidden outputs and
17899520 state bytes across local-ring wrap. An extra row beyond context
was rejected without changing any draft state.

An optional journal retains only overwritten KV slots and convolution
inputs from a speculative forward. Restoring each accepted prefix of a
seven-row trial across the local-ring boundary matched all eight layers'
KV/convolution state and next hidden output exactly, even with changed
rejected inputs. This restores carried state; selecting accepted hidden
rows/logits and connecting target verification remain separate work.
The target now retains every row's final-normalized BF16 hidden before
the head's division by 16. A real MQ85GB gate checks those rows against
an independent CPU normalization equation and requires exact seed,
full-vocabulary logits and state parity across prefill/decode/chunks.

Target verification now returns each trial row's greedy token. Accepted-prefix
commit restores all 42 layers and selects the accepted row's logits without
rerunning those layers. On actual MQ85GB with default aligned Q8, accepting
1 through 9 trial tokens matched scalar-prefix argmax, logits, normalized
hidden and all KV/convolution bytes exactly, including changed rejected
tokens. One-row verification and nonmutating input/commit rejection also
passed. This short graph gate does not connect draft generation or Rust
acceptance, and does not qualify long-context MTP.

A separate component gate imports all eight BF16 depths from a VMM owner
and uses the actual MQ85GB Q8 embedding, embedding norm and output head.
With supplied hidden inputs, every depth matches separately assembled
primitives exactly for hidden output and all valid-vocabulary logits.
This uses the raw Q8 head path and does not qualify end-to-end speculation.

The draft cache retains only rows below the target length minus eight. Its
remaining target hidden rows are recomputed for each proposal, then rolled
back before the next depth. An actual-weight component gate with supplied
target hidden matched full-prefix rotation for lengths 1–41, chunks 1/3/13
and 1–8 draft tokens, including all stable KV/convolution state. Repeated
proposals, reset, invalid inputs and context limits passed.

Sessions now load the sidecar, maintain all eight draft states during prefill
and decode, and expose bounded trial/commit operations through the bridge.
Rust chooses the greedy accepted prefix, stopping at the first mismatch or
EOS. A native MQ85GB/MTP test generated 18 greedy tokens in seven trials;
every committed target logit, KV byte and convolution value matched ordinary
decode. Cold replay, invalid commit bounds, injected restore failure and
recovery passed, as did image/audio identity checks with MTP attached.
Context-32 allocation matched its 164265472-byte quote exactly.

Rust CLI MTP on/off produced the same five-capital answer with a shared
base+MTP VMM owner, default aligned Q8, context 128, prefill chunk 1,
temperature zero, thinking disabled and a 32-token output cap. This checks
the application path, not MTP throughput. A sidecar-only owner with a
self-loaded base failed during prefill; its CUDA cause is unresolved. Use
the full base+MTP owner topology for this scoped runtime validation.

## Checks

```sh
cargo test -p ds4-core --test inkling_catalog --locked
cargo test -p ds4-core --lib inkling --locked
make -j1 test-catalog-parity test-tokenizer-parity
cargo check --workspace --all-targets --locked
```

The convolution gate uses small synthetic weights and the normal CUDA
backend, including captured decode with changing input/history:

```sh
make -j2 tests/test_inkling_kernels CUDA_ARCH=sm_121
python3 tools/host_memory_guard.py --max-gib 4 --high-gib 3 \
  --timeout 120 --log scratch/inkling/sconv.memory.jsonl \
  -- ./tests/test_inkling_kernels
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
all four basic chat-template effort fixtures passed. Independent full-model
source parity and serving gates remain unverified. Qwen's resident weight owner was stopped with user
authorization before native testing; inspect current ownership before loading.
