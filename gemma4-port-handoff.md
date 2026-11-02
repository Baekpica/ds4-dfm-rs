---
project: ds4-dfm-rs (clone of JLFN/ds4-dfm-rs)
branch: feat/gemma4-family
base_commit: 8c2068b5 (upstream main)
last_updated_commit: 48546032 (W1 and W2 are on top of it, unpushed)
sibling_repo: /data/VQ_tools branch dev
handoff_written_at: 2026-11-02T23:30:00+01:00
session_id: 01a19693-09d4-71d2-b0ac-5c87d02208e0
---

Gemma-4 26B A4B VQ — complete family port, in the tree's own layout (2026-11-02)

1. What the port must reach

ds4-server serves the gemma4_vq artifact over the OpenAI-compatible surface
on this machine (x86_64, RTX 4070 SUPER, compute 8.9) and on the GB10, with
the same contract every other family has:

  GET  /v1/models                lists the gemma4 model id
  POST /v1/chat/completions      returns a completion for a prompt
  the same request twice         identical ids and text
  CPU lane and CUDA lane         the same first N ids
  ds4 --serving-check            runs every declared lane on the artifact

2. The standard this port follows (read, not assumed)

The repository states its own contract in two places, both read for this
plan:

- AGENTS.md "New model-family serving contract": a new MoE family must
  implement every applicable serving option — context length, max_seqs,
  continuous banks, partial prefix reuse, session snapshots, SSD/disk KV
  directory and capacity, and MTP enablement and draft controls when the
  model provides MTP. "Parsing a flag alone is not support." Every family
  runs `ds4 --serving-check`; a declared item that fails is fixed or no
  longer declared.
- docs/ds4-dfm-model-families.md "Design contract": Rust owns
  identification, validation, tensor bind plans, tokenizer/chat behavior and
  lifecycle policy; native code owns weight upload, CUDA/MMQ/VMM, graphs and
  numerical state. Adding a family covers both sides and their state, API and
  correctness gates. No plugin registry, graph framework or broad
  abstraction layer.

The file set is therefore not a choice. Every VQ family in this tree is:

  Rust   crates/ds4-core/src/<family>.rs + {layout.rs, model.rs, release.rs}
         crates/ds4-core/src/tok/<family>.rs, chat_template.rs arm, shape.rs,
         bind.rs, layout.rs, identify.rs, validate.rs, tensors.rs,
         serving.rs, serving_host.rs, serving_check.rs, sibling.rs
         crates/ds4-core/tests/<family>.rs + tests/fixtures/<family>/
         crates/ds4-cli/examples/<family>_gate.rs
  Native ds4_<family>.h, _plan.h, _bind.inc, _payload.inc, _graph.inc,
         _session.inc, _batch.inc, _mtp.inc, _bank_payload.inc,
         _gpu.cuh + cuda/<family>_*.cuh
  Wiring ds4.c include points and arch dispatch, ds4_cuda.cu, Makefile
         (include lists, object deps, test and gate targets), ds4.h,
         ds4_model_catalog.h
  Hosts  crates/ds4-sys FFI, crates/ds4-server (generate.rs family arm,
         model registry, output parsing), docs/<family>-vq.md,
         docs/ds4-dfm-model-families.md row, docs/serving-capabilities.md
  Proof  tokenizer ids vs the reference pipeline, template bytes vs the
         official Jinja, release open, logits/parity gate, serving-check,
         OpenAI gate

3. Work breakdown, in dependency order

W0 — artifact and Rust contract. DONE.

  release.json directory (trunk + embedding + optional mtp), GVQ1/GVBC
  validation, layout of 12177 + 48 tensors, identify/bind/layout/validate/
  serving_caps, family registration.

W1 — Rust tokenizer and chat template. DONE this session (evidence below).

W2 — native family skeleton and binding. DONE this session (evidence below).
  ds4_gemma4_plan.h (the two attention geometries as functions of the
  layer, the KV/memory plan), ds4_gemma4_bind.inc (the pointer adapter over
  the artifact's own HF names), ds4.c family and variant values, the shape
  constant, the layer struct, the weights_bind arm, the host-shape arm, the
  free, and the Makefile include list. The host shape bypasses
  config_validate_model (ds4.c:73369-73378), so the family needs no C-side
  metadata validation: Rust owns it.
  Gate: tests/gemma4_bind.c (make test-gemma4-bind) over `vq-gguf list`.

W3 — the CPU engine.
  ds4_gemma4_graph.inc: 30 layers, sliding-1024 attention on 25 layers and
  full attention on 5/11/17/23/29 (head_dim 512, 2 KV heads,
  attention_k_eq_v so the full layers have no v_proj), partial rotary 0.25
  with rope 1e6 on the full layers and 10000 on the sliding ones, seven
  norms per layer plus layer_scalar, router top-8 of 128 over the GVQ1
  experts with the shared dense FFN 2112, gelu pytorch_tanh, softcap 30,
  tied embedding head.
  Gate: determinism, CPU/CUDA parity later, and the logits gate against the
  reference implementation on a host that can hold the 26B checkpoint (see
  section 6, risk 1). This host cannot: 52 GB of BF16 against 31 GB of RAM.

W4 — the CUDA lane on sm_89.
  ds4_gemma4_gpu.cuh + cuda/gemma4_*.cuh: Q4_K/Q8_0 GEMM/GEMV, the shared
  GVQ1 decode kernels (they carry no sm_121 dependency; the persist kernels
  are the sm_89 path per cuda/ds41_vq_decode.cuh:333-337), sliding/full
  attention, MoE gather, softcap.
  Gate: first-N-id parity with the CPU lane, then decode throughput.

W5 — state: session, payload, banks, snapshots, disk KV.
  ds4_gemma4_session.inc, _payload.inc, _batch.inc, _bank_payload.inc.
  The KV is dual: 25 sliding rings of 1024 and 5 full caches. The
  embedding part stays host-read row by row.
  Gate: `ds4 --serving-check` runs reuse, disk, snapshot and bank items.

W6 — the drafter.
  ds4_gemma4_mtp.inc for the 4-layer checkpoint (hidden 1024, shared KV,
  pre_projection [1024,5632], post_projection [2816,1024], 2048 centroids
  top-32, tied 262144x1024 embedding, 419,711,236 parameters), exposed as
  `--mtp-draft` on the serial lane.
  Gate: drafts never change the greedy output; verify rows equal the trunk.

W7 — serving wiring.
  ds4-sys FFI, ds4-server family arm (Gemma channel/thinking/tool_call
  output parsing, model id, request admission), serving_host limits,
  ds4-cli gate example. Declared caps get qualified or removed by
  --serving-check.
  Gate: the four-part OpenAI gate in section 1.

W8 — docs and release.
  docs/gemma4-vq.md (format, execution, serving, memory table),
  docs/ds4-dfm-model-families.md row, docs/serving-capabilities.md
  regeneration, README, the Hugging Face release directory.

4. What landed this session, with evidence

W1, tokenizer and chat template:

  crates/ds4-core/src/tok/gemma4.rs — the pipeline transcribed from the
  checkpoint's tokenizer.json (normalizer Replace(" " -> U+2581);
  pre_tokenizer Split(" "), MergedWithPrevious; BPE with byte_fallback and
  fuse_unk; post_processor with no BOS), the 22-entry special map, and the
  stop set from generation_config.json (`eos_token_id` [1, 106, 50]).
  tok.rs: load_path release arm (the vocabulary lives in the trunk part),
  specials arm, encode dispatch, rendered-chat scanning, jinja-only refusal,
  is_stop.
  chat_template.rs: the release arm, with GGUF's tokenizer.chat_template as
  the source (Gemma's release ships no tokenizer_config.json sidecar).
  tests/fixtures/gemma4/make_eval_cases.py + eval-cases.json: the reference
  stack (tokenizers 0.23.3 + jinja2 3.1.6) over the pinned checkpoint's own
  tokenizer.json and chat_template.jinja.
  crates/ds4-core/tests/gemma4_vq.rs: four tests.

  Evidence, run against /data/vq-artifacts/gguf-family-it:

    test family_is_its_own_contract ... ok
    test official_template_renders_the_reference_text ... ok
    test source_tokenizer_matches_the_reference_ids ... ok
    GEMMA4_RELEASE_PASS parts=2 tensors=12177
    test release_opens_every_part ... ok

  The tokenizer case set covers ASCII, leading and double spaces, newlines,
  tabs, digits, code, accented unicode, CJK, emoji, the turn/channel
  controls, punctuation and a 60-word run; the three conversations cover a
  single turn, a system turn with thinking off, and a multi-turn chat.
  Ids are compared element for element against the HF pipeline, and chat
  text byte for byte against the official Jinja.

W2, native skeleton and binding:

  ds4_gemma4_plan.h, ds4_gemma4_bind.inc, tests/gemma4_bind.c, the ds4.c
  family/variant/shape/struct/dispatch/free edits and the Makefile target.
  Evidence:

    make ds4_cpu.o                        (12 s, no errors)
    make test-gemma4-bind \
      GEMMA4_INVENTORY=/data/tmp/gemma4-release-inventory.txt
    GEMMA4_BIND_PASS tensors=12177 layers=30 experts=128

  The inventory is `vq-gguf list` over both parts (the embedding part first
  in the file, 12177 lines); the gate parses the stored type and dims, so
  every `tensor_expect_layout` check in the adapter runs against the
  artifact, and it asserts that a full layer's value projection is its own
  key projection (attention_k_eq_v).

5. Artifact and checkpoint facts a fresh session cannot re-derive

Artifact (canonical): /data/vq-artifacts/gguf-family-it
  trunk      5,730,270,592 B  12176 tensors  revision 4d7ae498...
  embedding  1,476,395,648 B      1 tensor   revision 4d7ae498...
  mtp          621,995,136 B     48 tensors   revision 6e5aaaf4...
  release.json, files.json, config.json; 12225 tensors, 7.291 GiB total.
  The trunk carries the tokenizer and the chat template.
  config.json is byte-identical to the embedded contract
  (crates/ds4-core/src/gemma4_vq/config.json): 19 keys, architectures
  ["Gemma4ForConditionalGeneration"].
Superseded, do not use as reference: /data/vq-artifacts/gguf-family (no
revision pin, no tokenizer).
VQ payload: /data/vq-artifacts/gemma-4-26b-a4b-it-8x8192x1-family (60
containers, 4.36 GiB, relL2 39.05-39.73 percent).
Checkpoints: /data/models/gemma-4-26B-A4B-it (weights are revision 4d7ae498;
its tokenizer files are the base model's), /data/models/gemma-4-26B-A4B-it-hf
(the instruct revision's config, tokenizer.json, tokenizer_config.json,
chat_template.jinja and generation_config.json),
/data/models/gemma-4-26B-A4B-it-assistant-hf (the drafter, revision
6e5aaaf4, 48 tensors, 419,711,236 parameters).
Tool: /data/VQ_tools, `target/release/vq-gguf {emit,release,check,kv,list}`;
the recipe recipes/gemma-4-26b-a4b-family.json names the family identity.
Reference stack for tokenizer/chat fixtures:
  python3 -m venv /data/tmp/gemma4-ref-venv
  /data/tmp/gemma4-ref-venv/bin/pip install tokenizers jinja2
  /data/tmp/gemma4-ref-venv/bin/python tests/fixtures/gemma4/make_eval_cases.py
Build: TMPDIR=/data/tmp/build-tmp for cargo. Family tests:
  DS4_GEMMA4_MODEL=/data/vq-artifacts/gguf-family-it \
  cargo test -p ds4-core --test gemma4_vq -- --include-ignored
One test in the full suite fails for environmental reasons:
minimax_h3::render::tests::mux_pipe_lifetimes asserts fd 3 is closed; with
`3>&-` its child passes. Not related to this family.
Host: x86_64, i7-14700F, 28 threads, 31 GB RAM, RTX 4070 SUPER 12 GiB
(compute 8.9). NVCC 13.3 at /usr/local/cuda/bin (not on PATH).

6. Risks that must stay visible

- Quality is unmeasured and this host cannot measure it. The 26B checkpoint
  is ~52 GB in BF16 against 31 GB of RAM, so the logits gate needs the GB10
  box (128 GB) or an equivalent host. Until it runs, "it serves" is a
  plumbing claim, not a model claim. The artifact's own experts measure
  39.05-39.73 percent relL2, and the published families pair that geometry
  with an 8-bit residual this artifact does not have.
- Context: the model's own maximum is 262144. f16 KV at that length does not
  fit the 12 GiB card beside the trunk; the qualified context must be
  measured by --serving-check, never advertised from metadata.
- The codebook fit is not bit-reproducible (0.0015 percentage points between
  two builds of the same stacks). Do not claim byte reproducibility.
- Residency: the model is MoE with 8 of 128 experts per layer, so a partial
  residency plan is legitimate, but this port does not declare one until the
  engine's measured working set exists. `serving_caps` states what is
  qualified, and today nothing is.
- The drafter is only wired as a part. Using it needs its own forward and
  the two projections (W6).

7. Open items outside the port

- Pushes: both branches are ahead; each needs the owner's word and an OSV
  scan first.
- docs/recipe-v41-identical.md is deleted in the VQ_tools working tree and
  was not committed by this session.
- The Hugging Face upload (JLFN/gemma-4-26b-a4b-VQ-GGUF) must publish the
  whole release directory, not one file.

8. Staleness check

    cd /data/ds4-gemma4 && git log --oneline 8c2068b5..HEAD
    cd /data/VQ_tools && git log --oneline 434e8b5..HEAD
