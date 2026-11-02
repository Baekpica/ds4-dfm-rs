---
project: ds4-dfm-rs (clone of JLFN/ds4-dfm-rs)
branch: feat/gemma4-family
base_commit: 8c2068b5 (upstream main)
last_updated_commit: 48546032
sibling_repo: /data/VQ_tools branch dev at ef9464c
handoff_written_at: 2026-11-02T22:00:00+01:00
session_id: 01a19693-09d4-71d2-b0ac-5c87d02208e0
---

Gemma-4 26B A4B VQ — port handoff and plan to OpenAI serving (2026-11-02)

1. What this port must reach

The deliverable is: ds4-server serves the gemma4_vq artifact over the
OpenAI-compatible surface on this machine, which is x86_64 with an RTX 4070
SUPER (compute 8.9), not only on the GB10. "Serves" means a verifiable gate,
not a build:

  GET  /v1/models                 lists the gemma4 model id
  POST /v1/chat/completions       returns a completion for a prompt
  the same request twice          identical ids and text (determinism)
  CPU lane and CUDA lane          the same first N ids

Everything below is the road to that gate, in order, each phase with its own
acceptance test.

2. Current state, verified (2026-11-02)

Committed on feat/gemma4-family, none pushed:

  717a2875  layout transcribed from the artifact (12177 tensors, entry-checked)
  259bd2dc  release reader, config contract, family registration, dispatch
  673463f3  drafter layout (48 tensors) and the per-part revision pin
  f82d0831  the part gate admits tokenizer.* (GGUF's own namespace)
  48546032  ds4-server compiles again (worker hidden width)

What works today, with evidence:

- Identification: identify_gguf(<release dir>) returns Variant::Gemma4A4bVq and
  split_count 2, after the config.json contract matches.
- Verification: release.json, files.json, both part SHA-256s, every GVQ1
  header, every book link and the whole layout are checked when a directory
  opens. Test: a_release_directory_verifies, 9/9 module tests pass against
  /data/vq-artifacts/gguf-family-it.
- Binding names and layouts: bind_names and expected_layouts return the
  family's 12177-tensor set; the drafter's 48 tensors sit under the mtp.
  namespace and join with Feature::Mtp.
- The server crate compiles. Nothing else in the serving path knows the
  family: no tokenizer, no chat renderer, no engine, no lane.

What does NOT work yet: any request. The family refuses at tokenizer load
(tok.rs, ModelFamily::Gemma4Vq => Err "carries no tokenizer metadata" - now
stale, the artifact carries it), and there is no forward pass for the model at
all.

3. The remaining phases, in order

Phase A — tokenizer load (small).

tok.rs must read tokenizer.ggml.tokens (262144), .merges (514906),
.token_type, the four ids, and tokenizer.ggml.special_tokens (22 entries), and
set bos 2, eos 1, unk 3, pad 0, sot 105 (<|turn>), eot 106 (<turn|>), think 98,
the tool_call and tool_response pairs, and the channel tokens.
Acceptance: a unit test builds the special ids from the artifact's own keys,
and `Vocab::load` on the release directory succeeds.

Phase B — the pre-tokenizer (small, must be reference-checked).

tokenizer.json's pipeline is exactly: normalizer Replace(" " -> "▁");
pre_tokenizer Split(" "), behavior MergedWithPrevious; then BPE over the merge
ranks; decoder Replace("▁" -> " "), ByteFallback, Fuse. There is no regex
stage, so it is simpler than the llama3 pre-tokenizer already here, and it must
not be borrowed from another family.
Acceptance: byte-for-byte id agreement against the HF `tokenizers` library (or
llama.cpp's gemma tokenizer) on a corpus that covers ASCII, unicode, digits,
code and the special tokens. No reference is installed on this machine today;
until one is, a round trip is necessary and not sufficient, and the commit must
say which it has.

Phase C — chat rendering (medium).

tokenizer.chat_template holds Google's canonical Gemma 4 template verbatim
(18683 bytes). Either render it in Rust or send the pre-rendered prompt, but do
not paraphrase it, and record which. The family's turn syntax is
<|turn>role\n ... <turn|>\n with <|think|> and <|channel>thought.
Acceptance: a golden test that renders three conversations and compares the
exact bytes against the template's own output for the same input.

Phase D — the engine (the large one; nothing exists).

A CPU reference forward first, from the checkpoint's config: 30 layers, hidden
2816, 16 heads, head_dim 256 sliding on 25 layers and 512 on layers
5/11/17/23/29, 8 KV heads sliding and 2 global, rope_theta 10000 sliding and
1e6 proportional with partial_rotary_factor 0.25 on the full layers,
attention_k_eq_v (full layers have no v_proj), seven norms per layer,
layer_scalar, 128 experts top-8 at FFN 704 plus a dense FFN 2112, gelu
pytorch_tanh activation, final_logit_softcapping 30, tied embeddings, sliding
window 1024.
Acceptance: the CPU forward reproduces the checkpoint's own logits on a fixed
prompt within a stated tolerance. Until that comparison exists, the engine is
unverified by definition; it is the one place where "it runs" and "it is
right" are different claims.

Phase E — binding and KV (medium).

Bind the inventory (trunk, embedding rows read from the embedding part, the
drafter only with Feature::Mtp) into engine buffers; the KV layout is dual
(sliding 1024 on 25 layers, full on 5) and the embedding is host-read row by
row.
Acceptance: a session opens on the release directory, prompt + N tokens
decode, and the memory census matches the plan's model (KV at ctx 262144 f16 is
10.4 GiB, which does not fit the 12 GiB card with the 5.3 GiB trunk - the
qualified context will be far below 262144; measure it, do not assume).

Phase F — the OpenAI lane (medium).

Declare the lane in serving_caps (today: Serial banks, no reuse, no disk,
SpecLane None, MtpKind::Sidecar, HostNeed::Cuda, ctx_max 262144, nothing
qualified), register the model ids, add the Gemma chat splitter, and give
generate.rs the family arm it currently lacks.
Acceptance: the four-part gate in section 1, run with curl, plus
`/v1/models` and a streamed and a non-streamed request.

Phase G — the CUDA lane on sm_89 (medium).

The GVQ1 kernels are shared with the other VQ families and carry no sm_121
dependency (checked: ds4_glmvq_gpu.cuh has neither sm_121 nor MXF4), but
cuda/ds41_vq_decode.cuh:333-337 records that the plain VQ gateup kernel needs
92 registers on sm_89 and cannot launch at 1024 threads, so the persist kernels
are the sm_89 path. Build with `make cuda CUDA_ARCH=sm_89` (or
`make cuda-generic`).
Acceptance: the OpenAI gate passes on the CUDA lane, and CPU and CUDA produce
the same first N ids.

4. Facts a fresh session cannot re-derive

Artifact (canonical): /data/vq-artifacts/gguf-family-it
  trunk      5,730,270,592 B  12176 tensors  revision 4d7ae498...
  embedding  1,476,395,648 B      1 tensor   revision 4d7ae498...
  mtp          621,995,136 B     48 tensors   revision 6e5aaaf4...
  release.json, files.json, config.json; 12225 tensors, 7.291 GiB total.
  The trunk carries the tokenizer and the chat template.
Superseded, do not use as reference: /data/vq-artifacts/gguf-family (no
revision pin, no tokenizer).
VQ payload: /data/vq-artifacts/gemma-4-26b-a4b-it-8x8192x1-family (60
containers, 4.36 GiB, relL2 39.05-39.73 percent).
Checkpoints: /data/models/gemma-4-26B-A4B-it (weights are revision 4d7ae498;
its tokenizer files are the base model's - wrong for this artifact),
/data/models/gemma-4-26B-A4B-it-hf (the instruct revision's config, tokenizer
and chat template), /data/models/gemma-4-26B-A4B-it-assistant-hf (the drafter,
revision 6e5aaaf4, 48 tensors, 419,711,236 parameters).
Tool: /data/VQ_tools, `target/release/vq-gguf {emit,release,check,kv,list}`;
the recipe recipes/gemma-4-26b-a4b-family.json names the family identity.
Build: TMPDIR=/data/tmp/build-tmp for cargo; the crate is large. Tests:
  GEMMA4_VQ_RELEASE=/data/vq-artifacts/gguf-family-it \
  GEMMA4_VQ_INVENTORY=<vq-gguf list output> \
  cargo test -p ds4-core --lib gemma4
One test in the full suite fails for environmental reasons: minimax_h3::render
::tests::mux_pipe_lifetimes asserts fd 3 is closed; with `3>&-` its child
passes. It is not related to this family.
Host: x86_64, i7-14700F, 28 threads, 31 GB RAM, RTX 4070 SUPER 12 GiB
(compute 8.9). NVCC 13.3 at /usr/local/cuda/bin (not on PATH).

5. Risks that must stay visible

- Quality: 13-bit single-stage experts measure 39 percent relL2, and the
  published families pair that geometry with an 8-bit residual (GVR8) which
  this artifact does not have. Nothing here measures model quality: no logits
  or perplexity harness exists in either repository. A working OpenAI endpoint
  will prove the plumbing, not the model.
- Context: the plan's own census puts f16 KV at 262144 tokens at 10.4 GiB
  against a 12 GiB card that also holds a 5.3 GiB trunk. The qualified context
  will be much smaller; measure it before advertising anything.
- The codebook fit is not bit-reproducible (measured spread 0.0015 percentage
  points between two builds of the same stacks). Do not claim byte
  reproducibility for a rebuild.
- The drafter is only wired as a part. Using it needs the drafter's own
  forward (4 layers, hidden 1024, shared KV) and the two projections.

6. Open items outside this port

- Pushes: both branches are ahead (VQ_tools dev 8, ds4 6); each needs the
  owner's word and an OSV scan first.
- docs/recipe-v41-identical.md is deleted in the VQ_tools working tree and was
  not committed by this session.
- The Hugging Face upload (JLFN/gemma-4-26b-a4b-VQ-GGUF) must publish the whole
  release directory, not one file.

7. Staleness check

    cd /data/ds4-gemma4 && git log --oneline 8c2068b5..HEAD
    cd /data/VQ_tools && git log --oneline 434e8b5..HEAD
