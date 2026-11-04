---
project: ds4-dfm-rs (clone of JLFN/ds4-dfm-rs)
branch: feat/gemma4-family
base_commit: 8c2068b5 (upstream main)
last_updated_commit: baca41b1 (docs(handoff): Phase 12 and Phase 13), plus
  this handoff update. 4aa records the long-context attribution: the MTP
  drafter's oracle attention was the O(ctx) cost, the guard now lets the
  fast kernel take keys=start (bit-identical), the launcher sets all eight
  measured fast options, and the 13.4K prefill profile puts 72 percent of
  GPU time in the attention tile kernel. 4y records the Spark serving
  stall: the serving default prefill chunk was 1, so an 8.5K open-grok
  prompt was 8,500 serial forwards and produced no chunk inside the
  client's 600 s idle timeout. The default is now 512 (the native bind's
  own), the launcher sets the measured fast paths, and the same prompt
  prefills in 22 s. 4v records the P8
  trial gate, the offload matrix and the discrete SSD path; 4w records the
  fused verify head, the PLE verdict and the owner's reordering; 4x
  records the 4070 serving lane and the CPU-expert verdict. The phase plan
  now carries Phase 12 (--moe-cache-mib, the SLRU expert cache, next) and
  Phase 13 (--n-cpu-moe, gated by the AVX2 microbenchmark). The commits on
  top of d6ad9774 are
  f0c2b92a (serving-check prefix reference), 19f5b75b (the Gemma VQ header
  split), 659a152b (optional FP8 full-layer KV and fused expert gate/up),
  bf6b3989 (optional fast decode attention), c09cfbf9 (the prefill-first
  plan), 9e23197f (the direct VQ prefill round, pushed to the fork),
  2faef6c3 (the attention and dense-down round), 14f27345 (the token-tiled
  dense down), b84e1bb0 (Phase 3), 781bf3cb (the Phase 3 handoff),
  b87664cb (the Phase 7 P8 review) and 7e67cf1b (the six-draft ceiling),
  followed by a7922215, 1bd4afe5, d77cbc97, 27221680, c1330085, 951f6322,
  cdbdf200, 767720d7, 341481b2, b09744a6, 92ad0679, 23af6003, e5b96d61,
  993ddd08, 78e96e02, 882bf8f8, 85af4cb6, 5905f7e7 and baca41b1.
  No GLM file is modified.
  Sibling VQ_tools: branch feature/gvq1-gemv-profile at be8c7e6; dev stays
  65ef74d; the owner edits that tree from other sessions, so leave its
  uncommitted files alone.
  Spark: leandro@192.168.1.91, ~/ds4-gemma4 HEAD 341481b2 with the
  verify-head files rsynced and built (make cuda-spark, 475 s, 0 errors;
  the head test and the long gate pass there); the serving quote fix and
  the CPU test are not on it. It serves the MTP lane on 0.0.0.0:8889
  (ctx 262144, bf16 KV, run-gemma4.sh) with the fixed launcher: prefill
  chunk 2,048 and the eight fast options; a diverse 13,515-token prompt
  prefills in 47.8 s (9.66K open-grok prompt ~35 s). The MoE workspace fix
  and the drafter fast-attention fix are built there (ds4, ds4-bench and
  ds4-server 12:43:34-36); the script defaults 2,048 on aarch64 and 512 on
  x86_64. Model at ~/models/gguf-family-it-r8.
plan: docs/gemma4-port-plan.md
sibling_repo: /data/VQ_tools branch dev; the profile unit is on
  feature/gvq1-gemv-profile (be8c7e6)
handoff_written_at: 2026-11-04T13:02:18+01:00
session_id: 01a19693-09d4-71d2-b0ac-5c87d02208e0
---

Gemma-4 26B A4B VQ — complete family port, in the tree's own layout (2026-11-03)

1. What the port must reach

ds4-server serves the gemma4_vq artifact over the OpenAI-compatible surface
on this machine (x86_64, RTX 4070 SUPER, compute 8.9) and on the GB10, with
the same contract every other family has:

  GET  /v1/models                lists the gemma4 model id
  POST /v1/chat/completions      returns a completion for a prompt
  the same request twice         identical ids and text
  sm_89 CUDA (compute 8.9)       this family declares HostNeed::Cuda, as
                                 every VQ family here does; there is no
                                 CPU execution lane for a VQ payload
  ds4 --serving-check            runs every declared lane on the artifact

State at this handoff: the family answers through both doors.

The serial session binds the plan, allocates the graph (KV 0.23 GiB at ctx
2048 with one-row prefill), runs all 30 layers, returns 262,144 of 262,144
finite logits, and two identical runs are byte-identical. The CLI one-shot
renders the artifact's own chat template: asked for Sweden's biggest city it
reasoned in the thought channel and answered Stockholm. The HTTP server
serves the same artifact and returned finish_reason stop with the same answer
in 55 s, reasoning_content free of control markers. The template is pinned to
the checkpoint file with ten oracle-rendered vectors (9b6f4a67), and the stop
set 1/106/50 from generation_config.json is asserted. P6a is closed
(19f25d46): a row-grouped prefill now equals the one-row path bit for bit
(41-token logits byte-identical at chunks 1, 2, 8, 9 and 22), and
--serving-check reports prefix, rewind and continue exact with
"result: pass (works 3, FAIL 0, WRONG 0, undeclared 0)". P6b is closed
(7cd25a50): the reference gate ran on the GB10 against the 48.07 GiB
checkpoint, returned the reference argmax on all four probes, and found and
closed the port's rotary pair counts (section 4e). The branch is pushed to
origin/feat/gemma4-family through 4ba1523c; 1d86327d and 7cd25a50 are local
and await the owner's word for the push.

A second, better artifact now serves through the same code: the R13+8 build
(4f) at /data/vq-artifacts/gguf-family-it-r8, 9.95 GiB, expert weight error
23.38 percent against the original 39.29, and the serving checklist passes
on it unchanged. It is the artifact every remaining gate runs on: P7's
payload, snapshots and disk KV, P8's drafter and P9's OpenAI check are all
measured against this directory, and the checklist must pass on it end to
end. The 13-bit artifact stays as the comparison point, not a second target.

P7 is closed (4g, 4h). The session payload, snapshots, the disk session
payload and the banked multi-sequence lane all pass the serving checklist on
the R13+8 artifact: result: pass (works 10, FAIL 0, WRONG 0, undeclared 0).
The batching refusal is gone; serving.rs declares banks and disk Qualified.

What is still open, in order:

  0. speed  the port was functionally complete at 2.9 decode / 3.9 prefill
          tok/s zero-copy. The measured fixes are committed in 8fae06b0
          (4l): device residency 3.9x, batched projections 2.15x, grouped
          expert prefill 22x. The direct VQ decode GEMV is d4f6d4f9 (4o):
          decode 10.86 -> 13.42 tok/s (+23.6 percent) and prefill
          352 -> 374, device-resident on the 4070 (DS4_CUDA_COPY_MODEL=1,
          R13+8, ctx 256, eight greedy tokens). Decode is still below the
          sibling 15-25 tok/s. The measured profile of the remaining time
          is attention 53.1 percent, direct GEMV 10.8, RoPE 6.5, restore
          6.4, router 3.4, gelu 1.4; section 9 phases 1-3 own that work.
          Phase 3 closed in b84e1bb0 (4t): the same artifact now measures
          32.76 tok/s decode median and 605.0 prefill, the rope, the router
          and the head are the family's own kernels, and the remaining
          measured costs are the VQ expert GEMVs (9.3 ms/step), the decode
          attention (5.9) and the Q4_K projections (3.3).
  0b. SSD  streaming is built and opt-in in 8fae06b0 (4m). The owner's rule
          is that it stays an option, never a family default. Gemma-only
          module and engine arms, no GLM edit; split shards supported; the
          4070 refuses with the coherent-access message. Remaining: a GB10
          run (slot hit rate, the 4 GiB default, the decode gain) before
          calling the option measured. A RAM-resident expert option is
          possible through the same slot/LRU module and is a capacity
          lever on this card, not a speed one (4o, section 9 phase 6).
  0c. profiles  the VQ_tools half is built on branch feature/gvq1-gemv-profile
          (42974fc): vqg_gvq1_* in c/gpu/decode-gpu.cu, the `gres`
          directive, the residual-aware fingerprint, `--gguf` /
          `--fingerprint-from`, and 3 kernel-parity plus 8 profile tests.
          The profile itself is NOT run yet: no profiles/ directory exists
          and docs/profile-optimizer.md now says so (be8c7e6). The ds4 half
          is not started: no consumer, no fingerprint bridge, no loader.
          Section 9 phase 4 owns it; the recommended first slice is the
          decode GEMV profile, not the 4070 stream map.
  1. P8   the drafter is committed in 8fae06b0 and passes gates A, B and C
          on a short prompt. The exact wrapped-ring gate passed on the
          Spark (4u: 1081 tokens, ring 1029, off 126.36 s, on 128.85 s,
          byte-identical); the default chunk-16 pair is self-consistent
          but differs from the exact path (the known grouped-prefill
          trade). The serving checklist passes (works 10, FAIL 0, WRONG 0,
          undeclared 0), but its mtp row is still "no check yet": no trial
          runs. The 406-token prefix failure is closed by Phase 0
          (f0c2b92a) and the exact-prefill reference is the gate. The draft
          ceiling is 6 (7e67cf1b); the measured matrix at ctx 1024 is
          no-MTP 6.75, draft 2 12.87, draft 4 15.99, draft 6 18.42 tok/s.
          Open: the real MTP trial gate, the Gemma-owned verify path, and
          the serving-default speed number.
  2. P9   the four-part OpenAI gate is scripted (tests/gemma4_serving_http_gate.sh)
          and not yet run; the live tool-tail continuation is done
          (be901c46); the serving-host limits are documented.
  3. P10  docs/gemma4-vq.md, the families row, README and the capability
          table are done (4a8d038b, f47d494c); left: the HF release
          directory, the OSV scan (already clean on 2026-11-03) and pushes
          (owner's word required).
  4. P11  artifact quality comparison, after P9: R13+8 (the artifact under
          test, still serving) against the two-stage 8x4096x2 build, on the
          P6b probes and greedy text through the OpenAI path. The served arm
          needs two-stage support in vq-gguf and in the ds4 GVQ reader first;
          the weight-space arm is recorded (4j): the two-stage directory's
          60-stack mean is 18.7203 against R13+8's 23.3772.

Operational facts this host needs:

  - ds4-server needs --mem-floor-gb 2 here: 6.71 GiB of weights plus the
    default 4 GiB floor leaves no room for the context quote, and the first
    launch was rejected with quote_overflow.
  - DS4_LOCK_FILE must point away from /tmp/ds4.lock, which the
    EmbeddingGemma server on 8098 holds; the gate used
    /tmp/ds4-gemma4-gate.lock.
  - Build: make cuda CUDA_ARCH=sm_89, then make ds4-server CUDA_ARCH=sm_89.
  - Serve: DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock ./ds4-server \
        -m /data/vq-artifacts/gguf-family-it --cuda -c 2048 \
        --host 127.0.0.1 --port 8099 --mem-floor-gb 2

  25eeaf9d  tokenizer + chat template, 15 reference cases + 3 conversations
  c6419b0f  native shape, dispatch, pointer bind, bind gate (12177 tensors)
  015287d7  handoff: the logits-gate risk sized from the checkpoint itself
  7aa7077a  handoff: state that nothing runs yet, engine corrected to CUDA
  ba2ef0bb  docs: the pinned reference semantics, every kernel cites it
  1a2dbc17  primitives: norm, router input/selection, rope, gelu, softcap,
            embedding scale, layer scalar - all gated
  272d9118  attention: both geometries, sliding window, gated
            (make test-gemma4-primitives; 0 failures)
  081bcbba  the device engine (ds4_gemma4_gpu.cuh, ds4_gemma4.h) and the
            30-layer graph (ds4_gemma4_graph.inc); both units compile
  3530614f  handoff: engine state and the remaining sequence
Commits on this branch, oldest first (all pushed to origin/feat/gemma4-family):

  15a6d9d0  docs: the phase plan P6..P10
  1daaab87  fix(cuda): build for sm_89 again, guarding the Blackwell-only
            instructions (e2m1x2, try_wait.parity, bulk prefetch)
  04a2679b  feat(gemma4-vq): the serial session reaches the graph
            (P6, gate open: graph line PASS, logits nan)
  7dd5b2d8  fix(gemma4-vq): read the norm and router weights as BF16
            (the P6 nan; primitives gate 15/15, all finite, deterministic)
  c85a975b  fix(gemma4-vq): run the tokenizer.json decoder when a piece is
            printed (U+2581 -> space, ByteFallback), both engines
  2dca7709  fix(gemma4-vq): refuse the multi-sequence graph instead of
            segfaulting (the serving check no longer core-dumps)
  aa0c5829  docs(gemma4): P6 state - the nan is fixed, the prefill spread is
            the open item
  33e30d05  feat(gemma4-vq): a full-row hash probe, and the stage it names
  911fe68f  docs(handoff): the ordered work queue from P6a to P10
  a576baf6  fix(gemma4-vq): serial prefill by default, and declare the reuse
            the session has
  9b6f4a67  feat(gemma4-vq): the official chat template, pinned and served
  57b25db0  docs(handoff): the chat template is pinned and served
  c6e7f72e  docs(handoff): state at 57b25db0, with the exact resume point
  e6a40976  docs(handoff): name this file's own commit beside the code state
  19f25d46  P6a: make a grouped prefill repeat the one-row calls
            (Q4_K projections, Q8_0 dense down, router and expert routes;
            logits byte-identical at chunks 1..22; serving check passes)
  55b34a71  docs(handoff): close P6a, record the P7 reconnaissance
  4ba1523c  docs(handoff): record the P7 payload plan
  1d86327d  fix(gemma4-vq): correct the rotary pair counts (128 sliding,
            64 full; found by the reference gate; the test pins them now)
  7cd25a50  fix(gemma4-vq): price the KV ring the graph uses (ring golden in
            the primitive test; kv rows clamped to ctx; plan helper agrees)
  faa2489e  feat(gemma4-vq): bind the R13+8 residual stage (layout, part and
            release validation, record links; 23.38 percent artifact loads)
  151dc611  docs(handoff): note the two-stage build in flight for P11
  e5cf90e3  docs(handoff): P7 slice 1 is written, its gate is the next action
  c42e8b31  feat(gemma4-vq): session payload carries both KV layouts
            (snapshot session qualified works; pass works 4, undeclared 0)
  dd6772dc  feat(gemma4-vq): the banked lane and the disk payload
            (disk session and all five bank items work; pass works 10)
  05761686  docs(handoff): P7 is closed, the gate prints works 10
  4a8d038b  docs(gemma4-vq): the family page, the families row and the
            README row
  be901c46  feat(gemma4-vq): the live tool tail renders the canonical
            responses (three unit tests pin the bytes)
  50e46a88  docs(gemma4-port-plan): P7 closed, the live tool tail done
  f47d494c  docs(serving): the Gemma row in the generated capability table
  b33f67c7  docs(handoff): P9 progress beside P8, and the OSV result
  453fd123  test(gemma4-vq): the four-part OpenAI serving gate
  e96d09d8  docs(handoff): the two-stage build's own weight-space number
  89fa33b7  docs(gemma4-vq): point the residual paragraph at what still
            exists
  (P8 is uncommitted: 4k)

Build note: this host has CUDA 13.3 and no recorded arch, and the default
nvcc target emits sm_121-only instructions. .ds4-cuda-config.mk records
CUDA_ARCH := sm_89; use `make cuda CUDA_ARCH=sm_89`. Two things this host
needs and did not have before 1daaab87: the sm_89 build itself (three
Blackwell-only instruction families were unguarded in the GLM/Beam/MiniMax
paths) and, for any run, DS4_LOCK_FILE — the default /tmp/ds4.lock is held
by a live EmbeddingGemma ds4-server (pid 3321996, port 8098), which must
not be disturbed.

What the engine does, reachable now through the serial session (P6):
embedding gather scaled by bf16 53.0, the 30 layers in reference order with
the seven norms, q/k/v (v copies k on the five full layers), per-head q/k
norms, the weightless v norm, rope, KV store into a ring of window+chunk
rows for sliding layers and the full context for full layers, attention,
o_proj, the dense FFN whose Q8_0 down reads the map through
ds4_gpu_matmul_q8_0_tensor, the router off the post-attention residual, the
128-expert block accumulating in bf16, the three post-FFN norms, the two
residual adds, layer_scalar, the final norm, the tied head and the tanh-30
cap.

THE PLAN IS docs/gemma4-port-plan.md (P6..P10, with gates). The summary
below is the same sequence; the plan file carries the detail.

  1. ds4.c session hooks. Mirror ling31's touch points: the family
     dispatch in ds4_model_apply_host_shape and the engine load (the bind
     call at the same place ling31_gpu_bind is called), then a graph in the
     serial session and the banked paths, then ds4_context_memory for the
     family, then payload save/restore and the bank payload, then MTP.
     Grep for DS4_MODEL_FAMILY_LING31_VQ in ds4.c (about 40 sites) and give
     Gemma-4 the same treatment; anything not applicable (KDA, mHC, DSA)
     drops out.
  2. The first real run: ds4 --serving-check on the artifact is the gate;
     before that, a first-N-token determinism run.
  3. crates/ds4-sys FFI for the new entry points, crates/ds4-server family
     arm (model id, admission, chat rendering, stop handling), and the
     OpenAI gate.
  4. The drafter (W5) and the docs (W7).

  Status at 7cd25a50: item 1 landed for the serial session (engine load,
  graph, context memory, prefill cap); item 2's checks all pass, including
  the reference logits gate (4e); item 3's family arm, chat rendering and
  stops landed and the serving checklist passes its reuse items; item 4 is
  untouched.

Known gaps in the graph to close with the session work: decode currently
recomputes the full row per token (no fused decode kernels), the KV ring
lives wholly in VRAM (ds4_context_memory must report it), and the expert
grouping does one host readback per layer per step (fine for decode, to be
measured for prefill).

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

W3 — the engine, on CUDA. This is the standard for a VQ family and the
  plan's earlier "CPU engine first" was a deviation: every VQ family in
  this tree declares HostNeed::Cuda in serving_caps (Glm53Vq, Ling31Vq,
  BeamVq, MinimaxVq, Axk2Vq, Mimo2Vq, NaiveN05, IQuestQ1 - checked in
  crates/ds4-core/src/serving.rs), and there is no CPU GVQ decoder anywhere
  in the tree: GVQ payloads execute through the shared ds4_gvq_record plan
  (ds4_glmvq.h) and its kernels. Only Qwen35 and DeepSeek4 are HostNeed::Any,
  and neither is a VQ family.
  Files: ds4_gemma4_gpu.cuh + cuda/gemma4_*.cuh (what the shared runtime
  lacks: two-geometry attention with per-layer head_dim, partial rotary,
  softcap 30, the 128-expert router with its two scale vectors, layer
  scaling), then ds4_gemma4_graph.inc (the 30-layer walk over the bound
  plan), then ds4_gemma4_session.inc and its KV.
  FIRST STEP, before any kernel: recover the reference semantics. DONE, in
  docs/gemma4-reference-semantics.md. The checkpoint's HF repo carries no
  modeling code, so the source of truth is transformers at the pinned commit
  cc1451df3b5b218295bc16803569256c53c9d6e2 (2026-11-02), and that file
  records the router order, where layer_scalar multiplies, the RMSNorm
  placement, the two rotary geometries with their exact inv_freq formulas,
  the softcap, the activation, and the drafter's real behaviour - with the
  file hashes, the config values, the measured constants and the list of
  things the engine must not do. Read it before writing a kernel; the
  kernel comments cite it by section.
  Gate: first-N-id determinism, then the logits gate against the reference
  (section 6, risk 1: 48.07 GiB checkpoint, 8.19 GiB active path, paged run
  untested on this host).

W4 — state and serving surfaces.
  Session, payload, banks, snapshots, disk KV (ds4_gemma4_session.inc,
  _payload.inc, _batch.inc, _bank_payload.inc). The KV is dual: 25 sliding
  rings of 1024 and 5 full caches; the embedding part stays host-read row by
  row. Gate: `ds4 --serving-check` runs every declared item.

W5 — the drafter.
  ds4_gemma4_mtp.inc for the 4-layer checkpoint (hidden 1024, FF 8192,
  every layer consumes trunk KV, pre_projection [1024,5632],
  post_projection [2816,1024], tied 262144x1024 embedding, 419,711,236
  parameters), exposed as `--mtp-draft` on the serial lane. The centroid
  masked-embedding path is NOT part of this checkpoint:
  use_ordered_embeddings is false, so the drafter's logits are a plain head
  over its own table (docs/gemma4-reference-semantics.md section 4). The
  trunk must publish the K/V of layers 28 (last sliding) and 29 (last full)
  at full length when drafting is on.
  Gate: drafts never change the greedy output; verify rows equal the trunk.

W6 — serving wiring. Chat rendering, the family arm, the model id and the
  stops are DONE (9b6f4a67): ModelSyntax::Gemma4 (variant 23) covers the chat
  format, tool markers, streaming thought close, generated-output parsing,
  the retry policy and the canonical fallback renderer; the template path
  renders the artifact's own Jinja; think_start strips the thought-channel
  opener. ds4-sys needed no new FFI: engine open already binds this family,
  proven by the server run. Remaining: the four-part OpenAI gate, the live
  tool-tail continuation, and the serving-host limits in docs.
  Gate: the four-part OpenAI gate in section 1.

W7 — docs and release.
  docs/gemma4-vq.md (format, execution, serving, memory table),
  docs/ds4-dfm-model-families.md row, docs/serving-capabilities.md
  regeneration, README, the Hugging Face release directory.

Scope note: the checkpoint is image-text-to-text, but the VQ artifact
carries no vision tensor (12177 text tensors + the drafter's 48); vision is
a sidecar part in this tree's terms and is out of scope until a vision part
exists. Nothing may encode the tower into the VQ payload.

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

4b. P6, the serial session: the nan is fixed, the gate state, the spread

What is in the tree now:

  ds4.c               the family memory quote, prefill cap, session fit, the
                      engine-load bind, context-memory arms, session create /
                      lazy alloc / free / invalidate / rewind, sync and eval
                      dispatch, the hidden-width arm, prewarm skip, eager
                      residency, payload refusals, and the batching refusal.
  ds4_gemma4_session.inc   sync (prefix resume, chunked prefill, head on the
                      last chunk) and decode (one row, head every step),
                      logits readback with a non-finite count.
  ds4_gemma4_graph.inc     named step failures plus the DS4_GEMMA4_DEBUG
                      probe set: gemma4_probe_at (four leading values),
                      gemma4_probe_row (the same at any row), gemma4_scan
                      (whole-buffer non-finite count and range) and the layer-0
                      weight dump from the map. All silent without the env.
  ds4_gemma4_plan.h        gemma4_activation_bytes, shared with the graph's
                      own size, so graph and estimate cannot drift.
  ds4_gemma4_bind.inc      gemma4_prefill_cap, gemma4_session_memory.

The nan (7dd5b2d8) — root cause, with the evidence that pinned it:

  The artifact stores every RMSNorm weight, router.scale and
  router.per_expert_scale as GGUF type 30 (BF16). `ds4 --inventory` reports
  model.layers.0.input_layernorm.weight as 2816 x bf16 at abs 4692465280, and
  an od dump of that offset decodes to 1.70..30.6, all sane. The three Gemma-4
  primitives that take a weight index (g4_rms_kernel,
  g4_router_input_kernel, g4_router_kernel) took `const float *` and indexed
  per element, so each read two BF16 values as one float32. That is why the
  failure was partial, not total: a float whose high half is the BF16 weight
  reads as that weight, so the first 1408 elements of a 2816-wide norm looked
  plausible and the rest read past the tensor. The scan shows the signature:
  58 non-finite of 8448, first at index 1572, range [-2.5e38, 3.3e38].

  The unit test carried the same wrong shape (float32 weight arrays), which is
  why it passed while the real path failed; it now rounds the weights to BF16
  before the copy, so the reference and the kernel see the artifact's values.

Gate state, in the plan's order:

1. graph line with KV bytes — PASS.
   ds4: Gemma-4 graph ctx=2048 chunk=2048 KV=0.62 GiB activation=132.0 MiB
   (that run predates the serial default; at chunk=1 the same line now reads
   KV=0.23 GiB, and serving_host prices exactly that number)
2. first-token determinism — PASS.
   Two identical prompt runs with --dump-logits: byte-identical files.
3. numeric bring-up — PASS for finiteness and shape sanity.
   Every stage probe is finite; 0 of 262144 logits non-finite; exit 0.
   Single-row forwards are exact: DS4_GEMMA4_PREFILL_CHUNK=1 --serving-check
   reports prefix, rewind and continue "works ... exact" (kept 21, top-1 17/17).
   The layer-0 stage values for a given token are identical between a 3-row
   batch and the same token computed alone through attention (q proj, k proj,
   attn out all bit-equal; the block output differs by 2 bf16 ULPs).
4. ds4 --serving-check — RUNS, verdict FAIL, 1 WRONG. Exact output:

   reuse     prefix    works, undeclared  top-1 17/17
   reuse     rewind    works, undeclared  top-1 17/17
   reuse     continue  WRONG              top-1 16/17; row 0 picked 236772
                                         (lead 2.438) where both references
                                         picked 1595 (leads 3.438, 1.812)
   disk/snapshot     no  (P7; the payload refusal is explicit)
   banks x5          no  "this model does not yet implement the
                         multi-sequence graph" (P7; 2dca7709)
   mtp/context/prompt/media: no check yet

The closed item: prefill row-grouping used to move the logits (P6a)

  Before 19f25d46, measured with the same prompt, same tokens, same positions,
  only the prefill chunk size differing (--dump-logits, 262144 values each):

    chunk 2    vs chunk 1   max |delta| 2.375   at index 14067
    chunk 8    vs chunk 1   max |delta| 2.625   at index 101
    chunk 9    vs chunk 1   max |delta| 3.125   at index 148583
    chunk 22   vs chunk 1   max |delta| 4.8125  at index 238348
    chunk 8    vs chunk 9   max |delta| 2.9375
  The argmax is the same token (100, <|channel>) in every configuration; the
  spread is in the logit magnitudes, which grows with the grouped width. Per
  stage it is ULP-level at layer 0, so this is amplification through the
  top-8 router (a discontinuous selection) rather than one stage being wrong.

  Ruled out as the source, each with a measured run:
    - the Q8_0 dense down: DS4_CUDA_PREFILL_PATH=cublas at chunk 2 gives the
      same 2.375 at 14067 as the default mmq path.
    - the Q4_K 2..8-token MMA (q4m): DS4_GLMVQ_Q4_MMA=0 at chunk 2 still
      differs (4.125 at 237033).
    - the VQ expert tensor-core paths: DS4_GLMVQ_MMA=0 DS4_GLMVQ_MMA8=0 at
      chunk 2 gives the same 2.375 at 14067 as the defaults.
  Bisect result (gemma4_row_hash, full-row FNV-1a, layer 0, token 818 at
  position 1, 2-row batch against the same token computed alone):

    q proj             9c136b29 == 9c136b29   identical
    attn out           94840fc0 == 94840fc0   identical
    cur after attn     80bd2527 != cfc77d3c   FIRST DIVERGENCE
    dense down normed  f345fe48 != e76e25e2
    router input       cb2a3235 != 2ad81edc
    expert acc normed  f31e0919 != 8cfdb122
    layer out          f5b9cd17 != f3d6ea31

  So the first stage that moved was the o projection: attention matched, and
  the residual after o_proj did not. o_proj is Q4_K with cols 4096 (16 blocks)
  and q_proj is Q4_K with cols 2816 (11 blocks); both were computed by q4m at
  two rows and by the SIMT dot at one row, and only the 16-block one landed on
  a different sum order. That was ULP-level, not a wrong result - the layer-0
  block output differs by 2 bf16 ULPs - amplified through the top-8 router.

  What closed it (19f25d46). Three paths order their sums by the row count, and
  each now repeats the one-row call per row when the chunk is wider:

    - the Q4_K projections: q4m has its own FMA order from two rows up, and
      glmvq_q4_select picks the scalar or stream kernel per call, so
      ds4_gpu_gvq_q4_decode_rows is exact only for the staged shapes its
      comment names;
    - the Q8_0 dense down: K=2112 skips the multiple-of-256 and
      multiple-of-128 tiers, so n_tok=1 runs matmul_q8_0_preq_warp8_kernel
      while n_tok>1 runs cuBLAS or matmul_q8_0_preq_kernel (ds4_cuda.cu);
    - the routed experts (N grows when two routes group into one expert) and
      the BF16 router (it leaves its one-token kernel above four rows).

  The shape is the one plain_graph_matmul_tensor already uses in ds4.c when a
  kernel's reduction geometry moves with the row count. The probe was extended
  for the hunt: DS4_GEMMA4_HASH_LAYER=N adds the stage set at layer N, and
  DS4_GEMMA4_HASH_LAYERS=all hashes every layer's block output.

  Evidence after the fix:
    - six tokens (2,818,7488,563,6641,684), chunks 1 vs 2, all 30 layers,
      positions 0..5: zero differing hashes (before: position 0 first
      diverged at layer 11, position 5 at layer 7; the layer-11 stage probe
      named the dense down, and the same per-row fix closed both);
    - a 41-token prompt's logits are byte-identical at chunks 1, 2, 8, 9 and
      22 (sha256 aad4d287777ed9761aef280f005fb759baa63cf0cf223d8f74a54d0f156b84cf,
      argmax 100 in every run); the chunk-1 dump repeated after the fix has
      the same sha256;
    - --serving-check now reports:

        reuse     prefix    qualified  works  kept 21, top-1 17/17, exact
        reuse     rewind    qualified  works  kept 21, top-1 17/17, exact
        reuse     continue  qualified  works  kept 38, top-1 17/17
                                             (prefill 17, decode-fed 17), exact
        result: pass (works 3, FAIL 0, WRONG 0, undeclared 0)
      disk/snapshot/banks/mtp/context/prompt/media keep their P7/P9 status.

  The family still declares a one-row prefill: a wider chunk is now exact but
  repeats the same launches, so one row remains the measured default. The
  declare-at-one comment in serving_host.rs says so.

Next step, in order:

  a. P6b closed at 7cd25a50: the reference gate ran on the GB10 against the
     48.07 GiB checkpoint (section 4e). The four fixes it or its sibling
     audit produced are in 1d86327d and 7cd25a50; the ring-resume guard it
     named belongs to P7.
  b. Then P7 (banks, payloads, snapshots, disk KV), P8 (the drafter), the P9
     remainder and P10, in the order of the open list at the top.

Commands this host needs:

  make cuda CUDA_ARCH=sm_89
  DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock ./ds4 -m /data/vq-artifacts/gguf-family-it \
      --cuda -c 2048 --serving-check -p "Hello"
  DS4_GEMMA4_DEBUG=1 ... --token-ids 2,818,5279 --predict 2 --temp 0
  DS4_GEMMA4_PREFILL_CHUNK=1 ... (the single-row control)
  --dump-logits FILE with -p (22 tokens) for the chunk comparisons above

Measured facts from these runs:

  - The engine binds the plan: 12177 tensors, 6.71 GiB mapped, residency
    eager; the graph allocates lazily at the session's ctx.
  - The run needs DS4_LOCK_FILE because the default lock is held by the
    EmbeddingGemma server; 11.3 GiB of the 12 GiB card was free.
  - A 22-token prefill plus one token costs about 50 s wall, most of it
    weight registration; the graph free is clean (exit 0).

4c. The official chat template served end to end (2026-11-03, commit 9b6f4a67)

The template was already read from the artifact (chat_template.rs's release
arm), but nothing pinned it and the server had no family arm, so no chat
request could be rendered or split. What landed:

  tests/fixtures/gemma4/chat_template.jinja, provenance.json (sha256
  ae53464b..., revision 4d7ae498, eos [1,106,50]), generation_config.json,
  make_chat_vectors.py and server-vectors.json: ten vectors rendered by the
  independent Python oracle (jinja2 3.1.6) covering thinking off/on, a system
  turn, tool declarations, tool results, preserved tool thinking, media
  placeholders, no generation prompt, and the template's rejection of string
  tool arguments.
  crates/ds4-core/tests/chat_template.rs: render_official_gemma4 reproduces
  all ten byte for byte.
  crates/ds4-core/tests/gemma4_vq.rs: artifact_carries_the_official_chat_
  template reads the template out of the GGUF, renders the same vectors and
  asserts the three stops.
  ds4-server: ModelSyntax::Gemma4 (variant 23) with chat format, tool markers,
  streaming thought close, generated-output parser, retry policy and the
  canonical fallback renderer; think_start now strips the
  <|channel>thought opener from reasoning_content.
  serving_host.rs: the family declares its one-row prefill cap (same env the
  engine reads) so the scheduler's verified-width fence no longer refuses the
  exact serial plan, and bank_kv_bytes prices the hybrid layout the way
  ds4_gemma4_plan.h allocates it (251,863,040 bytes at ctx 2048, the 0.23 GiB
  the graph line reported, against 0.47 GiB generic).

Evidence:

  cargo test -p ds4-core --test chat_template            8 passed
  cargo test -p ds4-core --lib serving_host::tests::gemma4  2 passed
  cargo test -p ds4-server gemma4                         9 passed
  DS4_GEMMA4_MODEL=... cargo test --test gemma4_vq -- --ignored artifact_carries
                                                          1 passed
  POST /v1/chat/completions, ctx 2048, --mem-floor-gb 2, temp 0:
    finish_reason stop, content "The largest city in Sweden is **Stockholm**.",
    144 completion tokens, 55 s, reasoning_content clean of control markers.
  The CLI one-shot path renders the same template: -p "What is the name of the
  biggest city in Sweden?" began in <|channel>thought and answered correctly.

Two family defects were fixed on the way, both found by trying to serve:
the chunk fence, and the generic KV charge. Two remain visible:

  - The live tool-tail continuation refuses this family (it needs the
    canonical template); a tool-result follow-up on a cached continuation
    fails rather than silently mis-rendering.
  - The server needs --mem-floor-gb 2 on this 12 GiB card: 6.71 GiB of
    weights plus the default 4 GiB floor leaves no room for the quote.
    Not a family defect, a host limit to state in the serving docs.

Open at that commit were the row-invariance work and the reference logits
gate; the first closed in 4b (19f25d46), the second in 4e (7cd25a50).

4d. P7 reconnaissance: what the payload slice needs (2026-11-03)

The serial session already provides sync, eval and prefix reuse
(ds4_gemma4_session.inc). What P7 adds, in dependency order:

  1. payload save/restore (ds4_gemma4_payload.inc). The contract is
     ds4.h:1598-1600: magic "DSV4", version 3, 13 u32 header fields, then
     tokens, logits, then the state. The dispatch arms to extend are
     ds4_session_payload_bytes (ds4.c, the is_* chain), ds4_session_save_payload
     and ds4_session_load_payload; ds4_ling31_payload.inc (154 lines) is the
     model, including the header checks and the failed-restore rule (a partial
     load never leaves a usable checkpoint).
  2. snapshots need no new mechanism: ds4_session_save_snapshot /
     load_snapshot go through the payload, so the serving check's snapshot
     item follows from (1).
  3. disk KV plugs into ds4_kvstore (ds4_kvstore.h) once (1) names the bytes.
  4. banks need the real multi-sequence graph (ds4_gemma4_batch.inc) and the
     ds4_engine_supports_batching refusal removal; that is the large piece and
     stays last.

Gemma-4's payload layout, stated explicitly as the plan requires. A layer's
cache is (ds4_gemma4.h:57-69):

  full    (layers 5, 11, 17, 23, 29):  ctx rows
  sliding (the other 25):              G4_WINDOW + cap rows = 1024 + cap
  row bytes: kv_heads * head_dim * 2 * sizeof(uint16_t)
             sliding 8 * 256, full 2 * 512; kcache and vcache each that many.

The first slice saves each layer's whole cache at its fixed row count (0.23
GiB at ctx 2048, cap 1), which needs no dead-row reasoning; a later trim can
save only the live ring rows and must keep the same validation. The header
carries ctx, cap, G4_LAYERS, the two row counts, the vocab and n, so a
restore can reject a payload from a different ctx or cap; kv_bytes is
recomputed, not trusted (it does not fit u32 at large ctx).

Follow the ling31 set rather than invent a shape: ds4_ling31_payload.inc
(with its header checks and its failed-restore rule) is the model for the new
ds4_gemma4_payload.inc, and ds4_ling31_batch.inc / _bank_payload.inc are the
models for the banked lane. The payload also needs its three dispatch arms in
ds4.c and the ds4_kvstore wiring.

Next concrete step: run the P7 slice 1 gate (4g) and commit it; disk KV and
the banked lane follow.

4e. The reference gate: what ran, what it found (P6b closed, 2026-11-03)

Host and stack. The GB10 (gx10-e1d2, 192.168.1.91, 121 GB unified, driver
580.178.04) runs the reference; the 48.07 GiB checkpoint is at
/mnt/models/gemma-4-26B-A4B-it (51,644,289,748 B after the rsync, including
the instruct tokenizer and chat template overlay). Python venv
/home/leandro/gemma4-ref-venv: torch 2.14.1+cu130 (aarch64, sm_121) and
transformers 5.23.0.dev0 built from the pinned commit
cc1451df3b5b218295bc16803569256c53c9d6e2. Key access from this host is
~/.ssh/id_ed25519_gb10; the gate scripts are tools/gemma4-ref/, also copied to
/mnt/models/gemma4-ref (ref_logits.py, compare_logits.py,
ref_rope_variant.py).

Input. Both stacks are fed the same ids: the artifact's chat template with
the CLI's default system prompt ("You are a helpful assistant"), thinking
off, which is 28 ids for "The capital of France is" and 1243 for the repeated
harbour paragraph; the port reports prompt_tokens 28 and 1243, equal to the
reference render, so no tokenizer or template difference is in play. All
numbers are post-softcap, full vocabulary, last position.

  probe               ids   ref argmax   port argmax   cosine   KL ref||port   top10
  A short              28   818          818           0.9879   0.0024         6/10
  B long             1243   1509         1509          0.9873   0.0037         8/10
  C window edge      1024   1509         1509          0.9801   0.00038        8/10
  D first eviction   1025   1509         1509          0.9800   0.00038        8/10

The defect the gate found. Before 1d86327d the graph rotated 64 pairs on
sliding layers and 128 on full ones; the reference rotates 128 of 128 and 64
of 256 (modeling_rope_utils.py:193-265) and its own buffers measure 128 and
64 nonzero. The bisect that separated that from the artifact's error is
tools/gemma4-ref/ref_rope_variant.py: the reference patched to the port's
counts still picked 1509 at 1243 tokens, so the artifact did not explain the
flip; the fixed port returns 1509 at 25.875 against the reference's 25.125.
Pre-fix numbers, for the record: A cosine 0.9846 and KL 0.0010 (the short
prompt hid the phase error); B argmax 818 at 15.8125 with the reference token
at port rank 1, cosine 0.9692, KL ref||port 4.45.

What the gate does not prove. The distributions are close, not equal: the
artifact is a lossy 8x8192x1 VQ encoding, so the residual (cosine 0.98,
top-10 overlap 6-8 of 10, max |delta| 8-11 at the top) is the encoding's
error, not a port defect. No greedy text, perplexity or benchmark was
compared, and nothing here is a quality claim.

Two findings recorded for P7. The port has no ring-resume guard:
gemma4_sync_keep returns session_keep_prefix without the ring_holds check that
Beam, IQuest, Naive and MiMo2 use (ds4.c:77320), so a divergent continuation
more than two positions behind the newest row can read overwritten slots; a
fresh single prefill cannot trigger it. And the attention golden had no
wrapped-ring case until 7cd25a50, so the slot mapping every sliding layer
uses had no unit gate either.

4f. The R13+8 artifact and the VQ toolkit fix (2026-11-03)

Why. The audit in 4e showed the serving artifact is the family recipe without
its residual stage: 39.29 percent weight relL2 on the expert stacks, against
the 22.9 percent the sibling R13+8 layers reach. VQ_tools had no residual
encoder, no GVR8 tensor and no recipe key, so the loss was not a conversion
defect but a missing mechanism.

What the toolkit gained (VQ_tools c859d1f, recipe 0fd5f9d):
- build-vq-gpu --residual-bits 8 (or residual_stage.bits in the recipe)
  trains a 256-entry E4M3 book on the error the stages leave, writes
  <name>.gvr8.bin (GVR1 header plus one byte per 8-weight vector) and
  <name>.gvr8_book.bin, and reports the manifest relL2 with the consumer's
  arithmetic, BF16(C1 + C2) at gain 1.
- vq-gguf gvr1.rs builds and validates the header (canonical offsets, zero
  padding, payload digest); emit writes one GVR8 tensor per expert matrix,
  one 256-entry book per stack, links them in gemma4_vq.residuals, and marks
  the release format version 2; check re-verifies headers, digests, books and
  links in both directions.
- recipes/gemma-4-26b-a4b-family-r8.json is the recipe: the family shape
  (13-bit, one stage, E4M3, experts only) plus residual_stage bits 8 on all
  30 layers, id gemma-4-26b-a4b-family-r8. VQ_tools' working tree still
  carries another line's edits to VQ_tools-handoff.md and
  docs/recipe-v41-identical.md; neither is part of these commits.

The artifact: /data/vq-artifacts/gguf-family-it-r8, 9.95 GiB, 23805 tensors,
format version 2. Trunk 8.00 GiB (11520 GVQ1, 11520 GVR8, 120 books, 175
Q4_K, 30 Q8_0, 391 BF16), embedding 1.38 GiB, mtp 0.58 GiB. The payload
containers are 7.01 GiB (4.35 stage plus 2.66 residual) at 23.3772 percent
mean relL2 over the 60 stacks (39.2868 before); 0.59 more bits per weight.

Build and emit, for reproduction:
  ./bin/build-vq-gpu /data/models/gemma-4-26B-A4B-it OUT \
      --recipe recipes/gemma-4-26b-a4b-family-r8.json        (819 s)
  ./target/release/vq-gguf emit --out DIR/gemma4-vq8-trunk.gguf \
      --recipe recipes/gemma-4-26b-a4b-family-r8.json \
      --artifact OUT --source /data/models/gemma-4-26B-A4B-it --part trunk \
      --tokenizer /data/models/gemma-4-26B-A4B-it-hf \
      --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52
  the embedding and mtp parts as in the 13-bit release; mtp from
  /data/models/gemma-4-26B-A4B-it-assistant, revision 6e5aaaf4.
  ./target/release/vq-gguf release --dir DIR --out DIR/release.json \
      --recipe recipes/gemma-4-26b-a4b-family-r8.json \
      --config /data/models/gemma-4-26B-A4B-it-hf/config.json

What the port gained (faa2489e): the layout has two contracts chosen by the
inventory (12177 plain, 23757 with residuals); the part validator parses GVR1
headers and enforces the all-or-none residuals map; release.json may be
version 2 but only with the -r8 recipe id; the bind looks up the residual
books and planes optionally, all-or-none per layer, and links each expert
record to its plane and the plane to its book (the link gvq_link_res checks).

Measured against the reference, the same probes as 4e:

  probe          artifact   cosine    KL ref||port   top10   argmax
  A 28 tokens    13-bit     0.9879    0.002371       6/10    818
  A 28 tokens    R13+8      0.9906    0.000548       6/10    818
  B 1243 tokens  13-bit     0.9873    0.003730       8/10    1509
  B 1243 tokens  R13+8      0.9901    0.419508       8/10    1509

Read the B KL carefully: it is one token. 1509 contributes 0.425 of the
0.4195, because this artifact puts the top logit 0.6 nats below the reference
(24.5 against 25.125) where the 13-bit artifact put it 0.75 above (25.875).
Every shape metric improved: cosine, rms delta (2.44 to 2.38 on A, 1.72 to
1.36 on B) and the runner-up order. Weight error 39 to 23 percent is a better
approximation of the same model, not a different model; no claim beyond these
numbers, and no perplexity or benchmark was run.

The serving checklist on this artifact, the full list, so the next session
knows exactly what still fails and why:

  reuse     prefix     qualified  works   kept 17, top-1 17/17, exact
  reuse     rewind     qualified  works   kept 21, top-1 17/17, exact
  reuse     continue   qualified  works   kept 38, top-1 17/17 (prefill 17,
                                           decode-fed 17), exact
  disk      session    -          no      session payload was written for a
                                           different model family        (P7)
  snapshot  session    -          no      Gemma-4 disk KV restore is not
                                           implemented yet                (P7)
  banks     warm/fork/partial/disk/width  no   batch_ctx_create: this model
                                           does not implement the
                                           multi-sequence graph           (P7)
  mtp       lossless              no check yet  declared Sidecar - None   (P8)
  result: pass (works 3, FAIL 0, WRONG 0, undeclared 0)

Everything marked P7 or P8 is the remaining work; the three reuse items and
the result line are the checklist passing on this artifact today. The
snapshot row above has already moved: P7 slice 1 (4g) turned it into
"works, kept 38, top-1 17/17, exact" on a build whose declaration was still
None, which is why that run ended with "undeclared 1" rather than a pass.

4g. P7 slice 1: the session payload, written and awaiting its gate (2026-11-03)

The code is on disk and builds; the gate did not finish, so this section
states precisely what is proven and what is not.

Proven, from the earlier run of the same build: the snapshot item reported
"works, kept 38, top-1 17/17, exact", i.e. a checkpoint was written, read
back and reproduced the next 17 tokens, and the rewind/continue neighbours
stayed exact. Not proven: the declaration flip. That run ended "undeclared
1" because serving.rs still said Support::None while the mechanism worked;
the flip to Qualified is written but the rebuild-plus-rerun that would print
"result: pass (works 3, FAIL 0, WRONG 0, undeclared 0)" was killed at the
build step.

What ds4_gemma4_payload.inc contains (172 lines, the model is
ds4_ling31_payload.inc):

  - gemma4_payload_rows/_row_bytes/_body: the body is n token ids, the
    DS4_N_VOCAB logits, then every layer's k and v whole, at
    ds4_gemma4_kv_rows(il) rows of kv_heads*head_dim*2 bytes. Sliding
    layers price at G4_WINDOW + cap, full layers at ctx, so the byte count
    is derived from the same helper the graph uses, never assumed.
  - gemma4_payload_bytes refuses a graph that is not ready, failed, at a
    different position than n, or at n above ctx - the same guard shape as
    the other families.
  - save: 13-u32 header {magic, version, ctx, cap, G4_LAYERS, tag "GVK4",
    0, n, G4_WINDOW, rows(0), rows(5), DS4_N_VOCAB, 0}, then tokens, logits,
    caches; tokens are range-checked and logits finite-checked before any
    byte is written.
  - restore: sets failed and position 0 first, validates magic, version,
    ctx, cap, layers, tag, window, both row counts, vocab, the reserved
    words, and that the remaining length equals the recomputed body; then
    reads, writes g->logits, synchronizes, and only on full success clears
    failed and sets position = n. A partial load is never a usable prefix.
  - Both caches are copied whole with payload_write_tensor_span /
    payload_read_tensor_span at DS4_SESSION_IO_CHUNK, the existing helpers,
    so nothing here re-implements the I/O.

Wiring: #include "ds4_gemma4_payload.inc" after the ling31 include in ds4.c;
one arm each in ds4_session_payload_bytes, ds4_session_save_payload, the
load chain's restore ternary, and ds4_session_is_gemma4 added to the guard
list that calls ds4_session_ensure_graph. The old refusal arm ("Gemma-4 disk
KV restore is not implemented yet") is gone.

The gate, exactly:

  make cuda CUDA_ARCH=sm_89 && \
    DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock \
    ./ds4 -m /data/vq-artifacts/gguf-family-it-r8 --cuda -c 2048 \
    --serving-check -p "Hello"

  expect: snapshot session qualified works kept 38, top-1 17/17, exact
          result: pass (works 3, FAIL 0, WRONG 0, undeclared 0)

Then commit ds4_gemma4_payload.inc, ds4.c and serving.rs together with the
checklist line and this section. P7 continues at disk KV (ds4_kvstore, the
row-count derivation above is what it needs), then the banked lane.

4h. P7 closed: the disk layout and the banked lane (2026-11-03)

Both remaining items landed in dd6772dc and the gate printed, on the R13+8
artifact at ctx 2048 with prompt "Hello":

  reuse     prefix/rewind/continue  qualified  works  kept 21/21/38, exact
  disk      session                 qualified  works  kept 38, top-1 17/17, exact
  snapshot  session                 qualified  works  kept 38, top-1 17/17, exact
  banks     warm/fork/partial/disk  qualified  works  kept 21, tokens 16/16
  banks     width                   qualified  works  tokens 16/16 + 16/16
  result: pass (works 10, FAIL 0, WRONG 0, undeclared 0)

Two mechanisms, no checkpoint slab and no ring-resume guard needed:

  - The host payload validator (crates/ds4-core/src/payload.rs) now knows
    the "GVK4" layout: word 8 is the sliding-ring size, words 9 and 10 the
    per-layout row counts and word 12 stays zero, so it validates the
    serialized timeline (version, context, token count, vocabulary) and
    leaves the caches to the native reader. Before this the disk item failed
    at save with "written for a different model family" even though the
    native side was correct.
  - ds4_gemma4_batch.inc is the multi-sequence graph. Each bank owns its KV
    caches of both layouts; graph 0's GVQ context, expert scratch and
    working rows are borrowed, because the scheduler serializes bank forwards
    under the engine generation lock (A.X-K2's shape). Every cache row is
    position-indexed (sliding slot = position % rows, full slot = position),
    so copying the first min(pos, rows) rows copies exactly the slots the
    positions below pos occupy. That makes any prefix of a bank exact: fork
    and partial reuse are copies (or an in-place rewind) and no checkpoint
    slots are needed. ds4_gemma4_bank_payload.inc serializes one bank through
    the same DSV4 v3 writer the session uses.
  - gemma4_graph_release split into clear (what the graph owns) plus free,
    with a borrowed flag, so bank graphs free only their KV.
  - ds4.c gained the batch create/fit (base = gemma4_session_memory, each
    extra bank = ds4_gemma4_kv_bytes), the three bank payload arms, and the
    driver dispatch for prefill, decode, copy, fork, reset, logits and trim;
    the ds4_engine_supports_batching refusal is gone. serving.rs declares
    banks Persistent/Qualified and disk Qualified.
  - At ctx 2048 the fit chose 2 banks with 4.23 GiB spare on the 12 GiB
    card; the batch fit headroom line reads floor 4 GiB + burst 2 GiB.

Not done in this unit, and why it does not block the checklist: the
ring-resume guard noted in 4e (gemma4_sync_keep returns
session_keep_prefix without the ring_holds check the other ring families
use). The banked lane is exact, but the serial lane can still accept a
divergent continuation that reaches back past the rows the ring still holds
and read overwritten slots. It is a real defect, it is not exercised by any
checklist item, and it is recorded here as the next serial-lane fix.

4i. P9 and P10 progress beside P8 (2026-11-03)

Done and committed while P8 owns the working tree:

  - be901c46: the live tool-tail continuation renders the canonical template
    instead of refusing. Each result is one <|tool_response> block inside the
    model turn that emitted the call, the name resolves from the assistant's
    call id then the message name (chat_template.jinja:286-293), the Anthropic
    shape reads one part per call, and with thinking off nothing follows the
    responses because generation continues the same turn (382-389); with
    thinking on only the thought channel opens. Three unit tests pin the
    bytes. Stop handling was already correct: Vocab::is_stop covers ids 1
    (eos), 106 (<turn|>) and 50 (<|tool_response>).
  - f47d494c: the generated capability table now carries the Gemma row
    (banks persistent/qualified, partial reuse, disk and snapshot qualified,
    MTP sidecar/none/none until P8, cap 262144, qualified context unset) and
    DS4_GEMMA4_PREFILL_CHUNK; the drift test passes.
  - 4a8d038b: docs/gemma4-vq.md, the families row and the README row.
  - 50e46a88: docs/gemma4-port-plan.md records P7 closed and the P9 progress.
  - OSV scan (rule 10) on 2026-11-03: ds4-gemma4 1 manifest, 146 packages,
    0 vulnerable; VQ_tools 1 manifest, 34 packages, 0 vulnerable.

Still open in P9: the four-part OpenAI gate (GET /v1/models, a chat
completion, the same request twice byte-identical, and sm_89 CUDA), which
needs the GPU and the tree, both held by P8 while it works.

4j. P11 weight-space arm: the two-stage build is on disk (2026-11-03)

/data/tmp/vq-2stage finished at 11:48 (8.1 GB, 60 expert stacks, layers
0-29). Its own manifest (the build's report, `relL2` column) gives a 60-stack
mean of 18.7203 percent, min 18.31, max 19.48. That is the comparable figure
for this directory; the 18.7685 recorded earlier came from a different run
and should not be quoted for it. The R13+8 build's 60-stack mean is 23.3772,
so the weight-space gap is 4.66 points for the extra bits.

The served arm still needs two-stage support in vq-gguf and in the ds4 GVQ
reader; nothing in this directory can be served yet.

4k. P8: the drafter, taken over and in flight (2026-11-03)

The subagent's work was stopped and taken over. The tree carries, uncommitted:
ds4_gemma4_mtp.inc (new, the runtime and the draft round) plus 19 modified
files (Makefile, ds4.c, the Gemma includes and headers, the CUDA attention,
the primitive test, and the Rust bind/layout/serving files). It builds clean
with make cuda CUDA_ARCH=sm_89.

What the drafter does, every step cited in the file: the assistant's forward
(pre_projection over cat[trunk token row, trunk final-norm row], four shared
layers with q_proj + q_norm + rope and the trunk's K/V, the dense MLP, the
tied 262144x1024 head, post_projection), and the reference's draft round
(candidate_generator.py:1327-1351): position held at input_ids.shape[1]-1 for
every draft step, argmax between steps, up to num_assistant_tokens. The trunk
publishes the final-norm row on every forward (gemma4_mtp_publish) and drops
it on reset/restore; the K/V the drafter reads is layers 28 (sliding) and 29
(full), as docs/gemma4-reference-semantics.md section 4 settles it.

Gates on the short prompt (48 tokens, 24 generated, temp 0):
  A  drafting on equals drafting off: 122 bytes identical.
  B  every verify round matched: keep=5 five times, draft[i] == target[i-1]
     for each accepted row (e.g. draft 100,45518,108,236829,139 vs target
     45518,108,236829,139,4661).
  C  two drafting runs byte-identical.

The defect found and fixed in the take-over: the batched verify wrote five
rows into a sliding ring of window + chunk (1025 rows at chunk 1), and two
slots the block's own first row reads (positions start-1023, start-1022) were
overwritten by rows 2 and 3 of the same block. A short prompt never wraps the
ring, so gates A/B/C above could not see it. The ring is now window + the
widest block one forward writes, max(chunk, drafts + 1), in ds4_gemma4.h
(ds4_gemma4_kv_rows) and ds4_gemma4_plan.h (gemma4_kv_capacity), with the
Rust quote (serving_host::bank_kv_bytes) and its test updated. Session KV at
ctx 2048 went from 251,863,040 to 252,682,240 bytes. A payload written by the
old row count is rejected by the body-length check, which is the intended
failure mode for an unreleased format.

The gate in flight when this handoff was written (started 13:16, ~1300-token
prompt so the ring has wrapped):

  cd /data/ds4-gemma4 && export DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock
  P=$(cat /tmp/g4-long.txt)
  ./ds4 -m /data/vq-artifacts/gguf-family-it-r8 --cuda -c 2048 -p "$P" -n 12 --temp 0
  ./ds4 -m /data/vq-artifacts/gguf-family-it-r8 --cuda -c 2048 \
    --mtp /data/vq-artifacts/gguf-family-it-r8/gemma4-vq8-mtp.gguf \
    --mtp-draft 4 -p "$P" -n 12 --temp 0
  cmp the two outputs.

The no-draft run finished rc=0 (70 bytes, 12.7 minutes); the drafting run was
terminated by the environment about 7 minutes in, before its prefill finished,
so the wrapped-ring comparison is UNPROVEN and must be re-run from scratch.
If the two differ, the first place to look is the verify forward's ring
interaction (the block rows) and the drafter's keys=start window, not the
drafter arithmetic: gates A/B/C already pin the arithmetic on a short prompt.
Expect each long run to take 12 to 15 minutes on this card, so run the pair
with no other GPU work and a long timeout.

Remaining steps, in order:

  1. Finish the long gate; compare byte for byte.
  2. Run `--serving-check` with `--mtp ... --mtp-draft 4` and confirm
     result: pass (works 10, undeclared 0) with the mtp row declared
     Sidecar/serial/qualified.
  3. Decide G4_MTP_DRAFTS: it is 4 today with the comment "keeps the verify
     row block at the artifact's own layer count", but the checkpoint's
     generation_config says num_assistant_tokens = 6. Raising it to 6 is a
     one-line change (G4_MTP_DRAFTS, the Rust GEMMA4_MTP_DRAFTS, the
     serving_host constant) plus the same gates; the reference supports it.
  4. Measure speed with ds4-bench, which reports prefill_tps, gen_tps and
     first_token_sec (TTFT) per frontier:
       ./ds4-bench -m /data/vq-artifacts/gguf-family-it-r8 --cuda \
         --prompt-file <file> --ctx-start 2048 --ctx-max 2048 --gen-tokens 32
     and the same with --mtp ... --mtp-draft 4. No speed number has been
     recorded yet; the owner asked for it.
  5. Commit P8 (the include and the 19 files) with the gate lines, then
     update docs/gemma4-vq.md (the drafter is no longer "pending"), the
     plan's P8 section and the capability row if the caps change.

Host notes from this unit: /tmp is a 16 GiB tmpfs and was full at 99 percent
(two 4.4 GB vqtest emit outputs and a 594 MB assistant repack left by the
subagent); the two emit outputs were removed, the repack at
/tmp/gemma4-assistant-mtp.gguf was kept. A CUDA build needs several GB free
there or nvcc fails with "error writing to ...: No space left on device".

4l. The speed unit: measured, fixed, and the SSD contract (2026-11-03)

The owner rejected the first speed reading ("this speed are not correct")
and required the comparison to the sibling VQ families, not a guess. The
measurements below are ds4-bench on the R13+8 artifact, a 256-token prompt,
ctx 256, 8 greedy tokens, one run per configuration:

  configuration                                      prefill   decode
  zero-copy (the engine's default on this card)         3.94      2.91
  DS4_CUDA_COPY_MODEL=1, per-route prefill             15.41     11.22
  + batched projections (grouped GEMM per matrix)      33.22     10.99
  + grouped expert prefill (expert-grouped rows)      337.86     11.12

The 13-bit artifact under the same copy + per-route path measures 15.45 /
11.23, so the residual stage is not the cost.

The zero-copy finding: peak VRAM during the 3.94/2.91 run was 1083 MiB and
the engine logged "CUDA registered 9.37 GiB model mapping for device
access" and "map 9.37 GiB, device 0.00 GiB, host-pinned 9.37 GiB". The GPU
was 100 percent busy at about 60 W: every weight was read over PCIe. On a
discrete card ds4_gpu_materialize_model_plan returns before the eager pass
(it gates on cudaDevAttrIntegrated) and ds4_gpu_set_model_map registers the
whole mmap. DS4_CUDA_COPY_MODEL=1 is the existing lever: it copied the
9.37 GiB in 2.96 s, set g_model_device_owned, and the copy run peaked at
10,663 MiB of VRAM at 100 percent utilization.

The profile that found the prefill defect: nsys on one 256-token chunk
showed glmvq_restore at 72.7 percent of GPU time, 185,040 launches, which
is 720 per token (8 experts x 3 matrices x 30 layers). The P6a exactness
decision had put the per-route one-row loop on rows > 1 and left the
expert-grouped loop only for rows == 1, so a chunk restored every expert
matrix once per route. The siblings do the opposite: Ling-3.1's
l31_mm_many and A.X-K2's axk2_mm_many pass all rows to one grouped linear,
and their MoE groups by expert.

The changes, all Gemma-only. cuda/glm_vq.cuh and cuda/glm_vq_ops.cuh were
briefly edited and are now byte-identical to HEAD (git diff empty); no GLM
file is modified.

  ds4_gemma4_graph.inc  gemma4_projection_rows calls
    ds4_gpu_gvq_linear_many over all rows; the Q8_0 dense down takes the
    chunk in one ds4_gpu_matmul_q8_0_tensor call; the per-row loop remains
    only as the DS4_GEMMA4_EXACT_PREFILL gate.
  ds4_gemma4_bind.inc   gemma4_prefill_cap defaults to min(ctx, 512), not
    1. At ctx 2048 the sliding ring is window + 512 = 1536 rows, 0.63 GiB
    of KV, which leaves room for the 9.37 GiB artifact on the 12 GiB card.
  ds4_gemma4_gpu.cuh    g4_moe_run uses the expert-grouped loop for
    rows > 1 by default (one restore per expert per chunk); the per-route
    one-row loop is behind DS4_GEMMA4_EXACT_PREFILL. This is the 22x.

The exactness gate is preserved, not deleted: run the chunk-size checks
with DS4_GEMMA4_EXACT_PREFILL=1. The default prefill now has the sibling
GEMM accumulation order, so it is not byte-identical to the one-row path;
that is the same trade the siblings make.

Decode is the remaining gap. Every token still calls glmvq_restore 720
times (8 experts x 3 x 30), each materializing a bf16 matrix and then
running cuBLAS. The shared VQ decode kernels (glmvq_decode_gu/down, the
Ling-3.1 l31_decode_run path) cannot take this geometry: their loaders
static_assert whole 256-input k-steps and, with a residual stage, whole
2048-input blocks. Gemma's gate/up rows are 2816 wide (2048 + 768) and the
down rows are 704 (2.75 steps), so the templates refuse at compile time.
Generalizing them means editing cuda/glm_vq_decode.cuh, a GLM file and
off-limits. The next decode step is a Gemma-only kernel or a Gemma-only
vector restore, not a shared-header change.

SSD streaming contract for the Gemma-only module (not built yet):

  - The VQ1 header stores gain/main/plane as offsets relative to the
    record's data start (ds4_glmvq_gpu.cuh gvq_link: data + main, data +
    gain). A streamed record can be copied byte-for-byte into a device
    slot and its glmvq_source rebased by setting s.data = slot. The
    per-matrix book and residual book stay resident (separate records,
    64 KiB and 2 KiB).
  - The linear path (ds4_gpu_gvq_linear -> glmvq_gemm) reads the host
    glmvq_source, so updating model->source[index] after a load is enough
    for the current MoE path; the device source table only matters to the
    shared route/decode kernels, which this port does not use.
  - The GLM53 stream is the contract to mirror, not to edit: slot pool,
    LRU, pread from the model fd, stats. The Gemma sizes differ: the
    expert set is about 5.9 GiB, one record about 550 KiB, and one token
    touches 8 experts x 3 x 30 = 720 records (about 387 MiB) with no cache
    hit. A slot pool that holds most of the 5.9 GiB is the only
    configuration that can beat the resident path on this card; the
    option is still useful for the 13-bit artifact, larger contexts and
    banks.
  - The family gate in ds4.c refuses ssd_streaming for every non-GLM53
    family (the DS4_MODEL_FAMILY != GLM53 arm). Adding a Gemma arm there
    and a gemma4_stream member/arm is engine work, not a GLM edit; do not
    change glm53_stream_* or ds4_glm53_cache.h.
  - The boundary found in reconnaissance: ds4_gpu_set_stream_map
    (ds4_glm53_map.inc, a GLM file to call, not edit) refuses unless
    cuda_model_coherent_host_access(device) is true, which is the
    integrated GB10 shape. On the RTX 4070 SUPER the GLM53-style stream
    map cannot be set, so a Gemma streaming option must target the GB10
    and print the same clear refusal on this card. The pread source is
    ds4_model.fd; the slot/LRU contract to mirror is ds4_glm53_cache.h
    (layer/expert key, pin-before-miss, LRU victim).

Commands used (no other GPU work; DS4_LOCK_FILE away from /tmp/ds4.lock):

  cd /data/ds4-gemma4
  export DS4_LOCK_FILE=/tmp/ds4-gemma4-bench.lock DS4_CUDA_COPY_MODEL=1
  ./ds4-bench -m /data/vq-artifacts/gguf-family-it-r8 --cuda \
    --prompt-file /tmp/g4-bench-256.txt --ctx-start 256 --ctx-max 256 \
    --gen-tokens 8 --csv /tmp/g4-bench-grouped.csv

The profile command was:

  /usr/local/cuda/bin/nsys profile -o /tmp/g4-prof --force-overwrite true \
    --stats=true ./ds4-bench -m /data/vq-artifacts/gguf-family-it-r8 \
    --cuda --prompt-file /tmp/g4-bench-256.txt --ctx-start 256 --ctx-max 256 \
    --gen-tokens 1

The kernel summary is in /tmp/g4-prof.log (scratch, not in the repo).

Host facts: the RTX 4070 SUPER has 12 GiB; the 9.37 GiB copy plus the CUDA
context, 0.06 GiB of KV at ctx 265 and 17 MiB of activations peaked at
10,663 MiB. At ctx 2048 the 512-row cap costs 0.63 GiB of KV, so the copy
is tight but fits; ds4-server still needs --mem-floor-gb 2. A CUDA build
needs several GiB free on /tmp (16 GiB tmpfs).

4m. SSD streaming is built and opt-in (2026-11-03)

The owner's constraint is explicit: SSD streaming is an option, never a
family default. Nothing in the Gemma launcher, session path or graph turns
it on; `--ssd-streaming` is the only trigger, and the default remains the
resident path (zero-copy unless DS4_CUDA_COPY_MODEL=1 is set).

Native:

  - `ds4_gemma4_stream.inc` is the Gemma-only cache: a bounded device slot
    pool, pin-before-miss admission, LRU victim, pread from the model fd,
    and the same stats vocabulary as ds4_glm53_stream.inc. The default
    budget is 4 GiB; --ssd-streaming-cache-experts N and
    --ssd-streaming-cache-experts GB override it. The minimum is one route
    set (8 experts).
  - `ds4_gemma4_gpu.cuh` adds `ds4_gpu_g4_stream_rebind`: it parses the
    streamed VQ1/GVR1 headers with the same checks as gvq_link and
    gvq_link_res, points the plan's host `glmvq_source` at the device slot,
    and leaves the per-layer book and residual book resident. The device
    source table is intentionally not updated: only the shared GLM route/
    decode kernels read it, and this port does not use them.
  - `ds4_gemma4_gpu.cuh` also adds `ds4_gpu_g4_moe_run_ids`, so the stage
    reads the route ids once and the grouped MoE reuses them instead of a
    second D2H read per layer.
  - The graph stages the largest row prefix whose expert union fits and
    runs the existing grouped kernel over that prefix. With a cache of at
    least 128 experts the whole chunk runs in one call.
  - Split shards are supported. The R13+8 release is two GGUF shards
    (embedding 1.38 GiB, trunk 8.00 GiB); model_open_split maps them into
    one page-aligned virtual range. The stream reconstructs the shard bases
    with fstat and preads from the shard that owns the record. GLM53 still
    requires one shard.
  - The family gate admits Gemma-4 and keeps the GLM53 one-shard rule. On
    a card without coherent CUDA host page tables (the RTX 4070 SUPER)
    ds4_gpu_set_stream_map refuses; the engine prints the Gemma-specific
    refusal and closes. The GB10 is the target.

Rust:

  - ssd.rs admits Gemma4Vq on CUDA, allows the native default without the
    GLM ServingBudget, and passes a fixed expert/GiB cache through. The
    GLM auto quote (prefill rows, staging banks, 12 GiB reserve) is not
    applied to Gemma.
  - serving_host.rs charges the Gemma cache in the host quote (default
    4 GiB, R13+8 record-size estimate, 1 MiB staging/metadata), and uses
    the same cgroup-aware available-memory path.
  - The CLI help now names both families; the flag remains opt-in.

Measured on the 4070 SUPER (R13+8, ctx 256, 8 greedy tokens, resident):

  DS4_CUDA_COPY_MODEL=1, no SSD: 342.36 prefill / 10.64 decode tok/s
  --ssd-streaming: refused before allocation with
    "SSD streaming requires coherent CUDA host page tables"
    "Gemma-4 SSD streaming requires coherent CUDA host page tables (the
     GB10 shape); this card keeps the resident path"
  The refusal is the expected result here; the option never changes the
  resident run's behavior.

Still open: no GB10 run of the Gemma stream (slot hit rate, 4 GiB default
vs the 5.9 GiB expert set, and the decode gain are unmeasured); the decode
gap in 4l remains. The 4 GiB default is a policy choice, not a measurement.

4n. Per-GPU profiles: feasibility and the missing consumer (2026-11-03)

The owner asked whether the kernel work can move from a GB10-only shape to
per-GPU profiles: tune Gemma-4 on the RTX 4070 with the VQ_tools kernel
tools, write a second profile, and load the profile for the card in use.
The answer is yes in principle, but not with the current runtime.

What already exists:

  - VQ_tools `vq-tune` writes a per-artifact, per-GPU profile
    (`docs/profile-optimizer.md`, `c/gpu/vq-profile.h`). It records the
    exact CUDA device name and an FNV-1a artifact fingerprint; the reader
    refuses a profile from another artifact or GPU.
  - `vq-tune --profile PATH` writes anywhere, so
    `vq-tune.profile.rtx4070` and `vq-tune.profile.gb10` can coexist. The
    format does not need a new version for that.
  - The tuner binary exists on this host (`bin/vq-tune`) and the 4070 is
    the CUDA device it would measure; a full sweep was not run in this
    investigation. The Gemma 13-bit VQ payload
    `/data/vq-artifacts/gemma-4-26b-a4b-it-8x8192x1-family` has a
    manifest (`dim=8 nc=8192 stages=1`) and grouped geometries the generic
    readiness functions accept: down_proj 360448/128 = 2816 rows x 704
    cols (generic warp variant; the 88 indices per row are not a whole
    32-index block, so the DQV3 block kernel is not ready), gate_up_proj
    180224/128 = 1408 rows x 2816 cols (352 indices per row, so the DQV3
    block kernel is ready too).

The two gaps found by reading the code:

  1. No runtime consumer. A search of ds4-gemma4 for `VQPROF`,
     `vq-tune.profile` or `vq_profile` returns nothing. The only
     `DS4_*PROFILE*` environment variables are stage-timing diagnostics.
     The model repository is the documented owner of the decision to
     consume a profile, and it has not made that decision yet.
  2. The tuner measures different kernels. `vq-tune` calls
     `vqg_gemv_t` and `vqg_prefill_tile` in `c/gpu/decode-gpu.cu`.
     Gemma-4 runs `ds4_gpu_gvq_linear` -> `glmvq_gemm`, which restores the
     VQ matrix to bf16 (`glmvq_restore_bf16`) and calls cuBLAS, plus the
     grouped expert prefill. The profile's variant, threads and tile
     entries do not map to those kernels. The tuner's promise that a win
     there is a win in the model holds for the standalone decoder, not for
     this runtime.

Two more boundaries:

  - The artifact identity differs. The profile fingerprint is over the VQ
    payload `manifest.txt` and `.bin` containers. The served runtime opens
    the GGUF release (`release.json` plus shards); its release metadata
    does not carry the VQ payload FNV. A profile must either be loaded
    against the VQ payload directory, or the FNV must be embedded in the
    GGUF release at emit time, or a GGUF-based profile identity must be
    added.
  - The R13+8 residual stage is a copied GVR8 sidecar. `vq-tune` skips
    copied sidecars by design, so a main-stage profile does not cover the
    residual restore. Covering it needs a tuner and consumer for GVR8.

The GB10 hardcoding is not the profile system. The SSD stream calls
`ds4_gpu_set_stream_map`, which requires coherent CUDA host page tables
(a GB10 property); no profile can change that. The resident path is not
GB10-only: it runs on the 4070 now (4m). Making SSD streaming work on the
4070 needs a discrete-card resident-span map and stream design, a separate
unit from profiles.

Recommended order if the owner wants the profile work:

  A. Decode GEMV profile slice (first). Add a direct VQ GEMV decode kernel
     for batch 1 adapted from `vqg_gemv` (dim 8, any cols), make its
     variant/threads selectable per geometry, add the runtime loader with a
     path from `DS4_VQ_PROFILE` or a flag, and bridge the fingerprint.
     This targets the decode gap (10.7 tok/s on the 4070 against the
     sibling 15-25) and the tuner already sweeps the design.
  B. Prefill profile after A. Either make the existing restore mode
     (scalar vs vector) profile-driven per geometry, or port the
     `vqg_prefill` tiles into the grouped path and gate them at 1e-4.
  C. GVR8 residual tuning and the 4070 stream map are separate units; do
     not fold them into A.

Evidence commands used (no code changed in this investigation):

  cd /data/ds4-gemma4
  grep -rn "vq-tune.profile\|VQPROF\|vq_profile" --include=*.c \
    --include=*.h --include=*.cuh --include=*.inc --include=*.rs .
  cd /data/VQ_tools
  grep -n "vqg_gemv_ready\|vqg_prefill_ready" c/gpu/decode-gpu.cu
  ls /data/vq-artifacts/gemma-4-26b-a4b-it-8x8192x1-family
  head -5 /data/vq-artifacts/gemma-4-26b-a4b-it-8x8192x1-family/manifest.txt

4o. The Gemma-only decode GEMV, and the VQ_tools profile unit (2026-11-03)

The owner asked for two things that landed together: a decode path that is
not the shared GLM one, and a per-GPU profile system that can tune Gemma-4
on the RTX 4070 instead of only the GB10 shape.

The runtime half is d4f6d4f9: cuda/gemma_vq_decode.cuh, a batch-1 GVQ1
GEMV for dim 8 and any column count that is a multiple of 64, and
ds4_gemma4_gpu.cuh routes the n == 1 projection through it for gate, up and
down. ds4_gpu_gvq_linear stays the fallback for every other geometry.

Why the shared loader cannot take it (read, not assumed):
  - cuda/glm_vq_decode.cuh:154 requires every row to be whole 256-input
    k-steps and, with a residual stage, whole 2048-input blocks.
  - cuda/glm_vq_decode.cuh:157 requires a partial row under 2048 inputs to
    be a multiple of 256 and to have no residual.
  - Gemma's gate/up rows are 2816 wide (G4_EMBED, ds4_gemma4_plan.h:22) and
    its down rows 704 wide (G4_FF, ds4_gemma4_plan.h:40). 704 = 2*256 + 192
    fails the k-step rule; 2816 = 2048 + 3*256 fails the block rule, so the
    R13+8 residual stage is refused as well.
  - The shared GLM headers are untouched. This is the same per-family
    pattern the tree already uses (cuda/minimax_decode.cuh,
    cuda/iquest_decode.cuh, cuda/ds41_vq_decode.cuh).

The arithmetic is the restore's own, cited in the header: index packing,
plane bit and gain lookup at cuda/glm_vq.cuh:120-129; the GVR8 residual
added before the gain at cuda/glm_vq.cuh:112-118; the bf16 weight and
activation boundary of the CUBLAS_COMPUTE_32F GEMM at
cuda/glm_vq_linear.cuh:97-110. Variant 1 is one warp per row, variant 2 one
block per row, 128/256/512 threads, and the launcher takes the variant and
width so a profile can choose them. DS4_GEMMA4_VQ_GEMV=0 disables the
direct path; DS4_GEMMA4_VQ_GEMV_VARIANT and DS4_GEMMA4_VQ_GEMV_THREADS
select the knobs until the profile loader replaces the environment.

Measured, device-resident, R13+8, ctx 256, eight greedy tokens,
DS4_CUDA_COPY_MODEL=1 on the RTX 4070 SUPER:
  - decode 10.86 -> 13.42 tok/s (+23.6 percent);
  - prefill 352 -> 374 tok/s;
  - the eight greedy tokens are byte-identical with the direct path on and
    off;
  - nsys on the new path: g4_attn_kernel 53.1 percent (1.52 s, 990
    launches), the direct GEMV 10.8 percent, g4_rope_kernel 6.5,
    glmvq_restore 6.4 (5,973 launches, the vocabulary head in chunks),
    g4_router_kernel 3.4, g4_gelu 1.4.

The VQ_tools half is branch feature/gvq1-gemv-profile (worktree
/home/leandro/.opengrok/worktrees/data-vq-tools/subagent-01a1a1f4-...):
  - 42974fc, 21 files, 2271 insertions: vqg_gvq1_ready / vqg_gvq1_t /
    vqg_gvq1_launch_t in c/gpu/decode-gpu.cu (variant 1 warp, variant 2
    whole 256-index blocks, legal at 2816 but not at 704); the `gres`
    directive and the residual-aware FNV in c/gpu/vq-profile.h and
    crates/vq-tune/src/profile.rs; `vq-tune --gguf DIR
    --fingerprint-from DIR`, which times the served release's trunk
    records and gates every candidate against a host reference decoded
    from the same GVQ1 bytes; 3 GVQ1 parity tests and 8 profile
    round-trip/fingerprint tests.
  - be8c7e6 corrects docs/profile-optimizer.md: the page claimed a
    committed profile under profiles/gemma-4-26b-a4b-it-r8/ and rates
    measured against it. Neither exists. The page now states that the
    sweep has not been run and gives the command that produces it.
  - The profile is therefore NOT produced yet. The identity the tuner
    computes for the R13+8 payload is recorded by the run, not by this
    handoff.

Two boundaries to keep visible:
  - The tuner prices its own vqg_gvq1_* kernel; the runtime runs
    cuda/gemma_vq_decode.cuh. They implement the same arithmetic but are
    two copies, so a kernel change must be made in both or the profile's
    choice stops describing the runtime. The parity tests gate the tuner's
    copy against a host reference; the runtime's copy is gated by the
    byte-identical eight-token run.
  - No runtime consumer exists yet: no VQPROF parser, no DS4_VQ_PROFILE
    path, no fingerprint bridge. The profile format keys on the RQVQ
    payload fingerprint, while the served runtime opens the GGUF release,
    whose release.json carries no VQ FNV. Phase 4 in section 9 is the
    unit that closes both.

4p. The 2026-11-03 optimization session: the prefix gate closed, four optional paths

Four commits landed on feat/gemma4-family on top of d6ad9774. No GLM file
was touched; the only untracked file is review_agy.md (the external review,
not part of the tree).

f0c2b92a fix(serving-check): compare an off-grid kept prefix without reuse.
  The decisive run was DS4_GEMMA4_EXACT_PREFILL=1 on the 406-token prompt:
  every item passed (prefix and rewind exact, banks 16/16) in 28 minutes
  against 5 batched. That confirmed the batched prefill's chunk-width
  arithmetic, not prefix bookkeeping, caused the row-16 flip. The prefix
  item now gets a reuse-free reference at the same split (T[..k] cold, then
  T[k..] one token at a time) when the kept split is off the prefill grid;
  rewind keeps the two-sync reference. On the synthetic 406 prompt the
  batched run now has prefix and rewind work (prefix top-1 15/17, near-tie
  2 <= 2.750); the banks still report 10/16 because they expose only token
  streams and have no noise floor, while the exact run proves the bank path
  is correct. The original 406 prompt used for the first report is not on
  disk; /tmp/g4-bench-256.txt is the reconstruction that reproduces the
  structure (prompt 406, split 203, grid 512) and is more sensitive in the
  banks.

19f5b75b refactor(gemma4): give the VQ device runtime its own header.
  ds4_gemma4_vq.h now owns the Gemma VQ entry points; ds4_gemma4.h keeps
  the plan geometry and KV row helpers and no longer includes ds4_glmvq.h
  (the shared model-agnostic GVQ runtime header). This is the owner's
  "its own file" requirement; every sibling family includes the shared
  header from its own family header.

659a152b feat(gemma4): optional FP8 full-layer KV and fused expert gate/up.
  - DS4_GEMMA4_FULL_KV=fp8: the five full layers store E4M3 K, E4M3 V and
    one f16 scale per KV head (2,056 B a row against 4,096 bf16), following
    the Solar FP8 rows Beam and MiMo ship (ds4_cuda.cu:44407-44480,
    ds4_beam_gpu.cuh:795-880). At ctx 262144 the full layers fall from
    5.37 GiB to 2.69 GiB; the sliding rings stay bf16. Payload h[6] records
    the format, the host quote prices the same rows, and the option refuses
    --mtp (the drafter reads the full layer's bf16 KV). A three-token smoke
    run matched bf16 exactly; the long-context gate is still open.
  - DS4_GEMMA4_MOE_FUSED=1: one GEMV over an expert's gate and up, GELU in
    the epilogue (llama.cpp commit c49edf096's fused gate/up write-back,
    adapted to GVQ1). The arithmetic reuses g4_vq_dot_vector and g4_gelu;
    an eight-token probe matched the separate path exactly. On the 4070 it
    is neutral: prefill 336.53 -> 370.69, decode 13.29 -> 13.24 tok/s
    (ctx 256, copy resident). Decode is bandwidth-bound on the VQ bytes, so
    the saved launches do not move it. Keep the option for the GB10.
  - Makefile deps cover ds4_gemma4_vq.h, cuda/gemma4_attn_fp8.cuh and
    cuda/gemma4_moe_fused.cuh.

bf6b3989 feat(gemma4): optional grouped-KV fast decode attention.
  cuda/gemma4_attn_fast.cuh: one CTA per KV head, the group's query heads
  (2 sliding, 8 full) sharing one shared-memory K/V tile. The per-lane dot
  order, warp reduction, online softmax and float32 accumulators are the
  oracle's; only the K/V load path changes. Used only when rows == 1 and
  keys == 0; prefill and the MTP drafter keep cuda/gemma4_attn.cuh.
  DS4_GEMMA4_ATTN_FAST=1, default off. Verified, 8 greedy tokens identical
  off/on on each machine:
  - 4070 ctx 256 copy: decode 11.33 -> 18.09 tok/s (1.60x), first token
    0.0863 -> 0.0522 s.
  - 4070 ctx 2048 zero-copy: decode 1.59 -> 2.42 tok/s (1.52x), first
    token 0.6445 -> 0.4114 s.
  - GB10 ctx 256: decode 12.46 -> 13.73 tok/s (1.10x), first token
    0.0817 -> 0.0743 s.
  The two machines produce different token sequences from each other; the
  within-machine parity is the gate.

The Spark (GB10) now carries the work: host leandro@192.168.1.91, worktree
~/ds4-gemma4 (a git worktree of ~/ds4-dfm-rs, branch feat/gemma4-family at
d6ad9774 with the four commits' file content rsynced), artifact at
~/models/gguf-family-it-r8 (10 GB, copied 2026-11-03), and the VQ_tools
tree at ~/VQ_tools (dev at 65ef74d plus the uncommitted owner files; .git
carries feature/gvq1-gemv-profile at be8c7e6). Build with
  export PATH=$HOME/.cargo/bin:$PATH && cd ~/ds4-gemma4 && make cuda-spark
(cargo is not on PATH in a non-interactive ssh shell; the C/CUDA objects
compile for sm_121a and the first build failed only at the cargo step for
that reason). The 406 checklist, the FP8 long-context gate and the
fast-attention bench at ctx 2048 are the open Spark runs.

4s. The 2026-11-03 decode measurement: the VQ GEMM is latency-bound on the
    codebook, not conversion-bound (no commit; the tree is clean)

- What is measured, and how. The VQ expert GEMM (g4_vq_gemm_items_kernel)
  is 142.2 ms of the 288.0 ms of GPU time at ctx 256: 100.0 gate/up plus
  42.3 down. A temporary DS4_GEMMA4_VQ_SPLIT knob (since reverted) skipped
  the compute or the decode, and each arm was profiled with nsys. Same
  binary, same prompt, ctx 256 copy-resident, 30 layers:
    gate/up  normal 95.94  decode only 50.04  compute only 34.62
    down     normal 41.87  decode only 24.73  compute only 16.00
  The phases are additive (50.04 + 34.62 = 84.66 against 95.94), so the
  decode is 52 percent of gate/up and 59 percent of down, and it does not
  overlap the compute at two blocks per SM.
- The compute is at the hardware limit: 16.6 G padded MACs in 34.6 ms over
  48 SMs at about 2.4 GHz is 128 FMA/cycle/SM, the full scalar FP32 issue
  rate. Its only waste is the token tile: 2064 routes over 128 experts is
  16.1 tokens per expert against BN = 32, so about half of the compute is
  padding. The row tile is exact (704 = 11 x 64, 2816 = 44 x 64).
- The decode is not conversion-bound. A second arm (DS4_GEMMA4_VQ_DEC=1)
  pre-converted the residual book to f16 in shared and added in f16: four
  conversions per two weights against seven. Result: gate/up 98.33 ->
  96.11 ms, down 42.92 -> 42.68, inside the run-to-run spread and not
  worth its complexity; it was reverted. Its frontier logits were
  byte-identical to the reference arm (relL2 0.0, same eight tokens), which
  does say the f16 sum is exact on this artifact's codeword pairs.
- What the numbers fit: the book gather. Each thread issues four random
  8-byte reads into the matrix's 64 KB codebook per k tile, and the kernel
  holds 2 blocks x 8 warps x 4 loads = 64 in flight. At about 250 cycles of
  L2 latency that is 0.26 loads/cycle/SM = 2.1 weights/cycle; the measured
  rate is 2.7. The 64 KB book cannot be L1-resident today: two blocks of
  43.2 KB force a 100 KB shared carve-out on a 128 KB L1/shared SM, leaving
  28 KB.
- The next lever, with the mechanism named: hold one weight tile at a time
  (decode gate, compute gate, decode up, compute up), which drops the block
  to 25.8 KB, and set the preferred shared carve-out to 64 KB so two blocks
  leave 64 KB of L1 for the book; mark the activation and index streams
  .cg (L2 only) so they do not evict it. Expected: the gathers hit L1 and
  the decode moves toward its instruction bound. Verify by re-adding the
  split knob and profiling, then the usual tokens plus frontier gate.
- A row-tile experiment is neutral: BM 128 -> 64 (704 = 11 x 64 exactly, no
  row padding, four blocks per SM by registers) gave 94.26 gate/up and
  41.41 down against 95.94 and 41.87, inside the spread, and the
  decode-only arm was worse (down 26.70 against 24.73). More blocks did not
  create overlap; the serialization is the barrier pair inside a block.
- Not done: the tensor-core arm. The sibling's cuda/ds41_vq_prefill_mma.cuh
  is the reference (VQM 128x128 tiles, ldmatrix plus mma.m16n8k16, the
  token loop breaks at n8 granularity so padding is not paid, the row gain
  applied at the exit). It would take the compute from 50.6 ms to a few ms
  and leave the decode as the floor, so the decode's own ceiling with the
  L1 fix has to be measured first.

4t. Phase 3 closed: the vocabulary head, RoPE and the router (2026-11-03, late)

- Round start f64d3d66. The nsys decode window (eight steps, 320.8 ms,
  40.1 ms/step) put the cost here: rope 5.50 ms/step, the router top-8
  2.88, the head 3.23 (1.47 GB in 3.10-3.16 ms is 466-474 GB/s, already
  92-94 percent of the card's 504 GB/s), gate/up 6.24, decode attention
  5.87, down 3.05, Q4_K projections 3.29, all norms 1.30.
- Four Gemma-only units, all default-off and all byte-identical on the
  frontier logits (relL2 0.0, same eight tokens [6113, 1698, 236761,
  28333, 506, 877, 6910, 3136]):
  A DS4_GEMMA4_FUSE_ROPE, the fused norm+rope in
    cuda/gemma4_primitives.cuh (one block per (row, head), the trig spread
    over the pairs, the two kernels' arithmetic unchanged): 26.22 -> 32.10
    decode, 557.56 -> 613.63 prefill.
  B DS4_GEMMA4_ROUTER_PAR, the parallel top-k in g4_router_kernel (k block
    max passes, the same tie rule and pick-order sum): 27.16 -> 33.60
    decode.
  C DS4_GEMMA4_HEAD_GEMV, cuda/gemma4_head.cuh (persistent BF16 GEMV, the
    input row staged in shared, __ldcs, the softcap fused, the shared
    kernel's per-lane FMA order and XOR reduction): 34.11 -> 34.82 decode.
  D DS4_GEMMA4_ROUTER_DOT, cuda/gemma4_router.cuh (the shared BF16 dot's
    split==8 mapping and partial-sum order): 33.10 -> 35.30 decode.
- Interleaved timing, same binary, four fresh processes: off 578.36 / 24.40
  and 575.49 / 24.81, on 579.94 / 31.17 and 630.10 / 34.35. Median decode
  24.61 -> 32.76 tok/s (+33.1 percent), prefill 576.9 -> 605.0 (+4.9),
  TTFT 39.5 -> 31.6 ms.
- The round's record is docs/gemma4-phase3-2026-11-03.md; the design review
  that shaped it is agy_review_phase3.md (Gemini 4 Argon, session 14).
- Still sibling code: the Q4_K projections (glmvq_q4_stream and q4_small,
  3.29 ms/step) and the dense Q8_0 down (0.48 ms/step). They are the next
  ownership unit, not part of Phase 3's list.
- Phase 3 is closed. The next phase in the owner's order is Phase 7 (P8).

4u. Phase 7 P8 in flight: the Spark gates and the six-draft ceiling (2026-11-04)

- Round start b87664cb, after the Phase 3 handoff. The branch is pushed to
  JLFN/ds4-dfm-rs; the Spark worktree was nine commits behind and is now
  at 7e67cf1b. Its pre-existing content (the 2faef6c3 tree) is in
  stash@{0}; the aarch64 test fixtures remain locally modified.
- Spark build: make cuda-spark, 469 s at b87664cb and 482 s at 7e67cf1b,
  0 errors. The 4070 cannot hold the trunk plus the 0.58 GiB MTP sidecar
  copied at ctx 1536: the lazy graph alloc fails and then
  ds4_session_create fails even with DS4_SESSION_LAZY_GRAPH=0. The Spark
  (128 GB unified) runs it.
- Wrapped-ring gate, exact path: prompt /tmp/g4-long.txt (1081 tokens),
  chunk 4, ring 1029 at four drafts, ctx 1536, n=12, temp 0,
  DS4_GEMMA4_EXACT_PREFILL=1. Off 126.36 s, on 128.85 s, both 64 bytes,
  byte-identical. The gate is closed in the exact path.
- Default batched path, same prompt at chunk 16 (ring 1040): off and on
  both 66 bytes and identical, but the output differs from the exact path
  ("paragraph." against "block of"). The 2x2 matrix: exact chunk 4 and 16
  and default chunk 4 all say "block of"; only default chunk 16 says
  "paragraph.". Frontier logits: default 4 against 16 relL2 5.86e-2, max
  7.375, token flip; exact 4 against 16 relL2 4.43e-3, max 0.5, no flip.
  This is the known grouped-prefill accumulation trade recorded in 4b and
  Phase 0, confirmed at 1081 tokens; the chunk checks stay on
  DS4_GEMMA4_EXACT_PREFILL=1, and the P8 on/off pairs are lossless in both
  paths.
- Serving check, Spark, ctx 2048, --mtp --mtp-draft 4: pass (works 10,
  FAIL 0, WRONG 0, undeclared 0). The mtp row still reads "no check yet"
  (declared Sidecar qualified Serial): the checklist binds and declares
  the sidecar but does not run a speculative trial. That remains open.
- Draft matrix, Spark ctx 1024, 1081-token prompt, copy-resident, 32
  generated tokens: no MTP 6.75 tok/s (first token 0.153 s), draft 2
  12.87 (0.234), draft 4 15.99 (0.305), draft 6 18.42 (0.372). The spec
  log shows full acceptance (keep 3/3, 5/5, 7/7) in most rounds, partial
  at the end. Six drafts is the measured winner.
- Draft ceiling 6 (7e67cf1b): G4_MTP_DRAFTS is decoupled from
  G4_MTP_LAYERS (still 4) and raised to the checkpoint's
  num_assistant_tokens. The Rust validator, the serving quote and its test
  follow (253,091,840 bytes at ctx 2048, was 252,682,240). The exact gate
  at six drafts is byte-identical; the ring resize 1029 -> 1031 leaves the
  no-draft output unchanged (SAME_AS_DRAFT4_OFF).
- Phase 8 review: Agy_phase8_review.md (40,626 bytes, Gemini 4 Argon,
  session 14). The GLM acd8aa49 kernel is an absorbed-latent MLA kernel,
  not a GQA kernel; agy keeps four changes over the subagent's plan: both
  D=256/GROUP=2 and D=512/GROUP=8 from the first slice (full layers are
  61.5 percent of 8K attention FLOPs), pack the GQA group into the M=16
  tile, no score exchange on D=256 and a two-way split on D=512, and fuse
  the f32/bf16 conversion with one reused KV buffer. Q and K are already
  bf16-rounded, so the operand conversion is 0 ULP; chunk invariance comes
  from starting the key loop at (begin_all/32)*32; the gate is max_abs
  <= 2.5e-3, relL2 <= 1.5e-3, 0.0 chunk diff, the ctx-256 tokens, the
  four probes, the 8K KL <= 2e-2 and a sha256 repeat. Land after P8/P9 with
  G4_ATTN_MMA_MIN_ROWS = 16; the standalone header and test are the
  zero-risk slice.
- MiniMax/Beam correction: in this tree neither drafter is R13+8.
  MiniMax's drafter is Q8_0 + BF16, Beam's DFlash2 is Q8_0 + F32/BF16, and
  MiniMax's R13+8 is the trunk-expert recipe on 20 of 57 layers; Beam's
  trunk is R13 only. A VQ Gemma drafter would copy MiniMax's trunk format,
  not its MTP; no sibling has Gemma's separate 262144x1024 head (86
  percent of the 622 MB sidecar).
- P8 remaining: a real MTP trial gate (prefix reuse, snapshot, a stop
  inside an accepted verify block, temp>0 spec_sample_rows), the
  Gemma-owned verify path (the verify still calls the shared Q4_K/GVQ
  linear; the multi-row head and device argmax are unmeasured), and the
  speed number at the serving default. The Phase 3 and P8 options stay
  default-off.
- Next unit (P8 trial gate). The serving checklist's mtp row calls
  unchecked() and prints "no check yet"; --mtp only binds and declares the
  sidecar. Build a real trial: run the same session with MTP off and on,
  compare the emitted token ids byte for byte, and cover (a) prefix reuse
  (a second prompt sharing the first 11 tokens), (b) snapshot restore,
  (c) a stop inside an accepted verify block, and (d) temp > 0 sampling,
  where ds4_gemma4_mtp.inc still hardcodes sample_argmax and never calls
  spec_sample_rows. The target's tokens must be identical to the no-MTP
  run; the drafter may only change acceptance and speed. The gate belongs
  in crates/ds4-core/src/serving_check.rs and the serving_host tests; the
  Spark is the run host.
- Reproduce: /tmp/g4-long.txt (1081 tokens), /tmp/g4-spark-gate.sh (exact
  pair), /tmp/g4-spark-gate2.sh (default chunk 16), /tmp/g4-spark-gate3.sh
  (2x2), /tmp/g4-spark-logits.sh (four dumps) and /tmp/g4-spark-bench.sh
  (draft matrix). The four logits JSONs are /tmp/g4-l-*.json on this host;
  the Spark has the scripts under /tmp and the artifact at
  ~/models/gguf-family-it-r8.

4v. P8 trial gate, the offload matrix and the discrete SSD path (2026-11-04)

- P8 trial gate: built in crates/ds4-core/src/serving_check.rs
  (gemma4_mtp_gate, MtpReport/MtpBlock) and the thin example
  crates/ds4-cli/examples/gemma4_mtp_gate.rs; the Makefile target is
  tests/gemma4_mtp_gate. It runs greedy off/on, the state after the first
  multi-token block against a one-row replay, prefix reuse on the prefill
  grid, snapshot restore, a stop inside an accepted block, and
  sample-and-match at temp 0.8. ds4_gemma4_mtp.inc now calls
  spec_sample_rows when the host sampler is armed (the temp>0 fix).
  Measured on the 4070, long prompt, ctx 1280: PASS, 24 tokens, 4.80
  tokens/call, accepted [1,2,7,7,7], prefix/snapshot/stop/sampling equal.
  The default batched verify state is not bit-identical to one-row decode:
  block max_abs 1.297, relL2 2.467e-2, payload_equal false; the stop round
  max_abs 1.188, relL2 6.47e-2. The exact-path byte check remains to run
  on the Spark (DS4_GEMMA4_EXACT_PREFILL=1).
- Offload matrix (Spark GB10, ctx 1024/8192, draft 6): resident 424.08 /
  18.53 at 1024 and 289.90 / 8.53 at 8192; SSD streaming 182.65 / 17.88
  and 247.55 / 8.38. SSD costs 57% and 14.6% of prefill and 3.5% / 1.8%
  of decode; it is a capacity lever, not a speed one. FP8 full-layer KV
  cannot pair with --mtp at all ("the drafter reads the full layer's bf16
  KV", ds4.c:74110-74117).
- 4070 envelope: the resident path is host-pinned zero-copy (9.37 GiB
  registered, device 0.00); a full DS4_CUDA_COPY_MODEL=1 copy succeeds
  (9.95 GiB) but the session graph fails at ctx 1057 and at ctx 545 with
  chunk 64, so the 3.81-4.34 tok/s baseline is PCIe, not device compute.
- Discrete SSD streaming: ds4_gemma4_map.inc (new, Gemma-owned; the shared
  ds4_glm53_map.inc is untouched) promotes the 2.35 GiB resident spans
  through the unit materializer, publishes one 128-byte device range per
  cold expert record for the bind-time header gather, and restores the
  truthful record before the first routed read. Measured on the 4070, ctx
  1024, 4 GiB pool, draft 6: resident 74.75 prefill / 4.34 gen / 1.543 s
  first token; streamed 51.47 / 13.11 / 0.527; with
  DS4_WEIGHT_RESIDENCY_MTP=eager 51.54 / 13.28 / 0.515. The frontier
  logits and greedy ids are bit-identical between the two modes. Cache over
  the run: 250,746 hits, 2,694 misses, 4.92 GiB read, 0.60 GiB reread,
  17.5 s I/O. A 3,840-slot pool (7.01 GiB) fails the graph budget at ctx
  1024 even with the drafter host-pinned; 4 GiB is the operating point.
- The llama.cpp-style CPU-expert path is not available: there is no CPU
  GVQ decoder (every expert GEMV is CUDA), and the Rust vq3 oracle is a
  different container and not wired. The active-expert VRAM cache is the
  applicable mechanism, and it is now measured on the 4070.
- Phase 8 / verify-head review: phase8_att_agy.md (agy, 19,131 bytes).
  Advice: the multi-row verify head + device argmax before Phase 8
  (roofline estimate +12.5-15% on GB10 draft-6); Phase 8 MIN_ROWS=16 gives
  0% to the 7-row verify; the six Phase 3 kernels are default-off; ds41's
  verify head uses argmax_rows; gvq_linear rows>4 takes the cuBLAS GEMM.
  Verified points and corrections are in the session; the estimate is not
  measured.
- Working tree at handoff: the SSD unit (ds4_gemma4_map.inc, ds4.c,
  ds4_cuda.cu, ds4_gemma4.h, ds4_gemma4_graph.inc, ds4_gemma4_stream.inc,
  ds4_gpu.h, Makefile, docs/gemma4-vq.md), the P8 gate (serving_check.rs,
  lib.rs, the example, ds4_gemma4_mtp.inc, Makefile), the reworked
  tests/gemma4_serving_http_gate.sh, phase8_att_agy.md and
  offload_review.md. Nothing is committed or pushed.

4w. The fused verify head, the PLE verdict and the owner's reordering (2026-11-04)

- Verify head (b09744a6). g4_head_gemv_rows_kernel in cuda/gemma4_head.cuh
  stages 1..8 input rows as bf16 in shared (8 * 2816 * 2 = 45,056 B, under
  the 48 KiB static limit) and accumulates every row in the one-row
  kernel's exact eight-FMA sequence and XOR reduction, so the 1.48 GB tied
  table is read once per verify block instead of once per row.
  ds4_gpu_g4_head_rows dispatches 1..8 rows; the verify keeps the per-row
  shared linear as the fallback. The ids come from
  ds4_gpu_argmax_rows_tensor (n int32, 28 B) and the commit reads only the
  kept frontier row instead of the 7.34 MB block.
- The byte-identity check is a test, not a claim: tests/gemma4_head_rows
  compares the fused output against the one-row kernel on the same weights
  and inputs for every row count 1..8; 4,096 of 4,096 values are exact on
  sm_89 and on sm_121 (make test-gemma4-head-rows).
- Measured, GB10, ctx 1024, draft 6, device-resident (DS4_CUDA_COPY_MODEL=1):
  18.38 -> 20.21 gen tok/s (+10.0%), prefill 419.98 -> 419.66 (unchanged),
  first token 0.3725 -> 0.3374 s. Plain decode unchanged: 6.76 -> 6.76
  with DS4_GEMMA4_HEAD_GEMV on vs off (the single-row head is not a decode
  lever). The long default gate passes on both machines with the same
  stream and acceptance as before (4.80 tokens/call, [1,2,7,7,7]).
  agy's roofline estimate was +12.5-15% (20.8-21.2); the measured +10.0%
  is below it and is the number to quote.
- The exact-path byte gate (DS4_GEMMA4_EXACT_PREFILL=1) fails, but it fails
  identically on the Spark baseline at 341481b2, which does not contain the
  verify head. The exact switch covers the projections and the dense path,
  not every batched piece of the verify forward, so the block KV or norm
  still drifts. Known limitation, not a regression; closing it means
  extending the exact coverage to the verify forward, or gating on the
  frontier logits instead of the whole payload.
- PLE verdict (checked, not applicable). Gemma-4 26B A4B has
  hidden_size_per_layer_input = 0 (config.json:33 of both checkpoints) and
  no PLE tensors (the safetensors index has zero per_layer/ple entries;
  docs/gemma4-reference-semantics.md:63, 75-79). Qwen3.8's PLE is a
  different feature: a 320,001,446-row n-gram hash sidecar, 16 rows of 320
  B per token = 5,120 B BF16 (ds4_ple.h:15-21, ds4_ple.c:200-224,
  1019-1023; 95.37 GiB BF16 / 47.7 GiB FP8), read through a 4 KiB-page
  cache. Gemma's routed experts are about 450 MiB per token
  (8 x 1,965,568 x 30), so the PLE model reads about 92,000 times less per
  token; that is the architecture, not the loader. On the GB10 SSD costs
  only 1.8-3.5% of decode against resident, so PLE cannot be the missing
  speed. The only PLE-like headroom found is overlapping the six serial
  preads and uploads per expert and the per-layer id readback
  (ds4_gemma4_stream.inc:410-459, 478-480, 500-517); unmeasured, and not
  PLE. The Qwen model directory lives on the Spark at
  ~/models/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF, not on
  this host.
- Owner's reordering (2026-11-04): no more MTP trials on the 4070 until
  the llama.cpp-style --cpu-moe CPU-expert path is implemented. The order
  is therefore: close P8 (done except the serving-default speed number and
  the exact-path limitation), P9 (run tests/gemma4_serving_http_gate.sh),
  then the CPU-expert phase with its deciding standalone CPU GEMV over one
  408 KB expert record, then Phase 8 (the tensor-core attention kernel,
  MIN_ROWS=16, a prefill lever with no draft-6 value).
- Working tree at handoff: 993ddd08 is committed and not pushed (the
  verify head b09744a6, the handoffs 92ad0679/23af6003, the serving quote
  e5b96d61 and the CPU measurement 993ddd08). Untracked: review_agy.md and
  the built tests/gemma4_mtp_gate, tests/gemma4_head_rows and
  tests/gemma4_cpu_expert_gemv binaries. The Spark is synced to b09744a6,
  built (475 s, 0 errors), and both the head test and the long gate pass
  there; its git HEAD is still 341481b2 and the verify-head files arrived
  by rsync, so "synced" means the working tree, not a pulled commit. It
  does not yet have the serving quote fix or the CPU test.

4x. The 4070 serving lane and the CPU-expert verdict (2026-11-04)

- The 4070 serving quote now prices only what a no-MTP boot maps: the
  Gemma branch in gguf_span_bytes subtracts the release's embedded MTP
  part (622 MB), the rule Ling-3.1 already uses (e5b96d61). The card boots
  device-resident at ctx 16384 with DS4_GEMMA4_FULL_KV=fp8: 9.37 GiB
  copied in 2.8 s, 10.16 GiB of 12 GiB used, 20.95 tok/s on a short chat
  against 2.66 tok/s host-mapped. The banked lane is refused at that size
  (free=1.45 GiB after the copy), so the server is serial, reuse=exact.
  The quote admits ctx 32768 with about 0.2 GiB of margin.
- run-gemma4.sh (repo root) is the launcher: serve start/stop/status/logs,
  chat, plan, bench; defaults to the GPU copy, FP8 full-layer KV and fast
  attention; DS4_GEMMA4_RESIDENT=0 is the host-mapped fallback. It uses
  /tmp/ds4-gemma4-gate.lock because the default /tmp/ds4.lock belongs to
  the embeddinggemma server on port 8098 on this host. The open-grok entry
  is [model."gemma4-vq-4070"], base_url http://127.0.0.1:8099/v1,
  context_window 16384, api_key "local" (added to ~/.opengrok/config.toml;
  the backup is config.toml.bak-*).
- Known serving issue: a request killed mid-flight (the open-grok headless
  run) left the serial session graph allocated, and later requests got
  HTTP 503 "no session graph fits beside the batch banks" until the
  server was restarted. Not fixed; the restart is the workaround.
- open-grok's headless prompt did not return an answer in 339 s (killed):
  the server log showed prefill_cap=1, so its long system prompt was still
  prefilling. The direct chat through the same server answered at 20.95
  tok/s, so the model and wire work. 4y is the measurement and the fix.
- CPU-expert verdict (993ddd08): the deciding GEMV over the real 408 KB
  record measures 278.4 ms per token at 28 threads against the device
  path's ~94 ms, and 1.70 GB/s means it is compute-bound. The measurement
  does not clear the bar; do not port the host decoder unless the owner
  overrides. The active-expert VRAM cache remains the 4070 mechanism.
  agy's review (cpu_moe_agy.md, 2026-11-04) agrees with STOP and corrects
  the baseline: the ~94 ms is the uncached PCIe streaming number, while
  the active-expert VRAM cache runs all 240 experts in 9.29 ms/token
  (6.243 ms block + 3.050 ms warp, docs/gemma4-phase3-2026-11-03.md:26-30),
  so even its corrected AVX2 estimate (38-55 ms/token after fixing the
  benchmark's thread-spawn tax, per-weight bf16 and serial FMA) is 4-6x
  slower than the GPU. It also finds CPU experts cannot overlap with GPU
  attention (layer l MoE feeds layer l+1 attention) and that MTP verify at
  7 rows routes 45-56 unique experts per layer, 2.65-3.30 GiB per step.
  llama.cpp itself now ships the same pattern as ds4's slot cache:
  --moe-cache-mib, an SLRU active-expert VRAM cache (src/llama-moe-cache.cpp).
- Spark serving lane (2026-11-04): run-gemma4.sh now takes DS4_GEMMA4_MTP,
  DS4_GEMMA4_MTP_DRAFT (6), DS4_GEMMA4_MTP_MODE (on) and DS4_GEMMA4_HOST,
  and when MTP is set it also sets DS4_SERVER_CONTINUOUS=0. That last one
  is required: the plan refuses Gemma MTP on the continuous lane
  ("gemma4_vq MTP runs only on the serial lane", serving.rs:2559-2567),
  and DS4_SERVER_CONTINUOUS=0 is the switch that forces the serial driver
  (serving.rs:581-587). The Spark runs 0.0.0.0:8889, ctx 262144, bf16
  full-layer KV, weights=true, mtp=on draft=6, no SSD streaming; the
  effective plan is reuse=exact mtp=on. A short chat with reasoning_effort
  none answered in 1.00 s. The open-grok entry is [model."gemma4-vq-spark"]
  at http://192.168.1.91:8889/v1 with context_window 262144.

4y. The Spark serving stall: the prefill chunk was one row (2026-11-04)

- Symptom: open-grok on tmux 14 (model gemma4-vq-spark) waited 10m13s and
  failed with "inference idle timeout after 600s with no chunks". The
  server was up, /v1/models answered, the GPU sat at 94-96%: it was
  prefilling, not deadlocked.
- Root cause: serving_host.rs passed GEMMA4_NATIVE_DEFAULT = 1 as the
  session's prefill cap (introduced with 9b6f4a67). The native bind's own
  default is min(ctx, 512) (gemma4_prefill_cap, ds4_gemma4_bind.inc:267),
  and the sibling VQ families serve 2,048-4,096-row chunks (GLMVQ 4096,
  Ling-3.1 4096, Beam 2048, MiniMax 4096, A.X-K2 2048). At cap 1 the
  graph's grouped projections and expert routes never see more than one
  row: an 8.5K prompt became 8,500 serial forwards.
- Measured on the Spark, same binary, same flags (MTP on, serial lane,
  bf16 KV, ctx 262144, device-resident):
    295 tokens, chunk 1      13.03 s   22.6 tok/s
    295 tokens, chunk 512     0.447 s 660 tok/s
    2,115 tokens, chunk 512   3.57 s  592 tok/s
    8,092 tokens, chunk 512  21.97 s  368 tok/s
  The 8,092-token run repeated on a fresh boot as the first request:
  22.16 s, cached_tokens 0. The earlier 6-minute 8.4K request was a
  one-off; it did not reproduce.
- The launcher also enabled only one of the four measured fast paths
  (ATTN_FAST). The attention round's "on" arm is all four (VQ_PREFILL,
  ATTN_PREFILL, ATTN_FAST, Q8_BATCH; docs/gemma4-attention-round-2026-
  11-03.md), so run-gemma4.sh now sets all four and passes
  DS4_GEMMA4_PREFILL_CHUNK=512 (DS4_GEMMA4_PREFILL overrides).
- Code and docs: GEMMA4_NATIVE_DEFAULT is 512 in serving_host.rs (the
  stale comment that called the one-row path "the measured path" is
  replaced); docs/gemma4-vq.md's KV example (ctx 2,048, chunk 512:
  356,515,840 bytes) and its limits bullet are corrected. The launcher's
  fast_envs sets the four envs together; FAST=0 restores the oracle path.
- open-grok: [model."gemma4-vq-spark"] already had the right endpoint,
  model id, 262144 context and api_key; it now carries
  inference_idle_timeout_secs = 1800 and a description that names the
  512-row prefill. [models] default is gemma4-vq-spark (the owner's
  change; not reverted). The config backup is
  config.toml.bak-gemma4-spark-20261104-102642.
- Verification on the running Spark server: "Say hello in one short
  sentence." with reasoning_effort none answered "Hello!" in 0.22 s;
  a 32-token greedy request returned finish=length with empty content,
  which is the thought channel consuming the whole budget, not a fault.
- The 512 cap was also the MoE workspace cap, not only the default: the
  graph created the expert workspace with min(rows, 512)
  (ds4_gemma4_graph.inc:405) and the session memory priced the same
  (ds4_gemma4_bind.inc:293). A 2,048-row chunk therefore failed in the
  experts (items_cap overflow) and 8,192 failed earlier in the router
  (cuBLAS status 7). The workspace now follows the session chunk in all
  three places; the Spark rebuild (native objects + ds4-server at
  11:19:44) makes 2,048 work. Measured on the Spark, chunk 2,048:
  13,515 diverse tokens 47.8 s (vs over 4 minutes at 512); the first SSE
  role chunk arrives in 4.8 s. 4,096 is untested; 8,192 still fails in
  the router. The launcher defaults 2,048 on aarch64 and 512 on x86_64.
- Follow-up: the Spark binary is now built from the fixed sources but its
  git HEAD is still 341481b2; the source changes (serving_host.rs 512,
  the MoE workspace, the no-think default) are uncommitted on that
  worktree. The 4070 quote was not re-measured: its own Gemma server
  (pid 3118283) held 10.4 GiB of the card during this check.

4z. The thought channel was a server default, not the Gemma standard
(2026-11-04)

- The server defaulted every non-DeepSeek family to thinking on with the
  port's low effort (`parse.rs`: `thinking_enabled = !env.engine_defaults`,
  `default_effort = Low`). Gemma-4's canonical template opens
  `<|channel>thought` only for an explicit thinking request, so an omitted
  `reasoning_effort` should mean no thought channel. At open-grok's high
  effort a "1+2" turn spent minutes in the thought block before any
  visible content.
- Fix: `ServerConfig.no_think_default` (serve.rs), set by the bin for
  `ModelSyntax::Gemma4`; `default_effort` returns `None` for it; the four
  parse surfaces now compute
  `thinking_enabled = !engine_defaults && default_effort != None`. An
  explicit `reasoning_effort` still sets thinking and its level, which is
  the OpenAI parameter contract: omitted means the provider's default,
  named means honored. Siblings keep their own defaults.
- Verified on the Spark (ds4-server 11:49:53): a request with no
  `reasoning_effort` returned `{"content":"3","reasoning":null,
  "finish":"stop"}` in 0.66 s at a 16-token prompt. The open-grok entry
  no longer carries `supports_reasoning_effort`/`reasoning_effort`, so its
  requests omit the parameter; the config backup is
  config.toml.bak-gemma4-reason-20261104-113843.
- The remaining open-grok latency is not the thought channel: its system
  prompt is 9.66K, the chunk-2048 prefill takes about 65 s, and the
  long-context decode is slow. That is the prefill/decode performance
  work, not the reasoning default.
- The docs bullet claiming the live tool tail refuses this family was
  stale: be901c46 (2026-11-03) renders the canonical Gemma tool
  responses inside the model turn; only a non-tool suffix is refused.
  docs/gemma4-vq.md is corrected.

4aa. Long-context attribution and the drafter's fast attention (2026-11-04)

- Controlled bench matrix on the Spark, 13,447-token prompt from the repo
  docs, ds4-bench, one run per arm, 17 greedy tokens, bf16 KV, chunk 2,048:
    arm                                              prefill   gen   TTFT
    no MTP, four fast options                         312.0    4.55  0.23
    no MTP, eight fast options                        315.5    4.83  0.21
    MTP draft 6, four fast options, before fix        314.1    2.36  0.76
    MTP draft 6, four fast options, after fix         312.1    3.69  0.47
    MTP draft 6, eight fast options, after fix        316.2    3.84  0.45
    MTP draft 2, eight fast options, after fix        314.7    4.83  0.28
  The pre-fix arms also measured: MTP draft 2 312.9/3.52, draft 4
  313.1/2.85, draft 6 with ATTN_FAST=0 309.0/2.27, draft 6 with
  ATTN_PREFILL=0 (the oracle verify) 288.4/1.69. The no-MTP decode with
  the direct VQ GEMV off is 3.93 (the direct path is +16 percent); with
  VQ_PREFILL off the prefill is 310.5 (neutral on the Spark, unlike the
  4070); chunks 1,024/2,048/4,096 measure 298.3/312.0/311.7 prefill.
- The drafter's attention was the O(ctx) cost: gemma4_mtp_step called
  ds4_gpu_g4_attn (one warp per query head, its own global K/V reads) for
  six sequential drafts, while ds4_gpu_g4_attn_fast (one CTA per KV head,
  one shared 16-key tile, the same per-key arithmetic) refused keys != 0.
  The drafter passes keys=start. The guard now allows keys with the
  oracle's bound, and the step takes the fast kernel under
  DS4_GEMMA4_ATTN_FAST. The 1,081-token gate (MTP draft 6, 24 greedy
  tokens, eight fast options) is byte-identical: 115 bytes, cmp clean.
- The launcher now sets all eight measured fast options, not four. The
  four Phase-3 options (FUSE_ROPE, ROUTER_PAR, HEAD_GEMV, ROUTER_DOT) were
  missing from fast_envs; they are bit-identical on their Phase-3 gates.
  The no-MTP arm moves 4.55 -> 4.83 gen tok/s and 312.0 -> 315.5 prefill.
- MTP is still a net loss at 13.4K even after the fix. The speculative log
  keeps 1-2 tokens per round there (keep=1,2,1,1,2,2,2,4,1 over nine
  rounds), so six draft steps cost more than they save; draft 2 matches
  plain decode. At 2,048 tokens the same prompt keeps
  7/1/1/1/2/3/1/4/3 and reaches 15.06 gen tok/s. The acceptance is
  prompt- and context-dependent; do not quote the 4070 gate's
  [1,2,7,7,7] as a long-context number.
- nsys kernel summary of the 13,447-token prefill (chunk 2,048, eight fast
  options): g4_attn_tile_kernel<8> (full layers) 22.95 s, 53.1 percent;
  g4_attn_tile_kernel<16> (sliding) 8.25 s, 19.1 percent; the two
  g4_vq_gemm_items kernels 4.26 + 2.11 s, 14.7 percent; the batched dense
  Q8 down 3.04 s, 7.0 percent; everything else under 4 percent. The
  full-layer tile averages 656 ms per chunk at this length (about
  171 G MAC/s); the sliding tile 47 ms. Attention, not the expert GEMM,
  is the next prefill lever.
- The cheap attention experiment was measured and rejected: widening the
  tile block to 16 warps (8 rows x 2 heads sliding, 2 rows x 8 heads
  full, 512 threads) shares one staged K/V range across twice the rows
  but halves the CTA count. It regressed prefill 315.5 -> 312.9 tok/s,
  MTP draft 6 3.84 -> 3.60 and no-MTP 4.83 -> 4.80, so it was reverted.
  The block is back to 8 warps (4 rows x 2 heads sliding, 1 row x 8 heads
  full). The next attention step is the tensor-core kernel (Phase 8,
  cuda/gemma4_attn_mma.cuh), which is the only change that removes the
  scalar per-key dot and the per-key softmax serial chain.
- The Spark binaries are rebuilt from these sources (ds4, ds4-bench and
  ds4-server at 12:43:34-36) and the worktree carries the same two header
  edits; its git HEAD is still 341481b2. The server is stopped while the
  benches run and is restarted with the eight-option launcher after them.

4ab. Phase 8: the tensor-core prefill attention (2026-11-04)

- cuda/gemma4_attn_mma.cuh is new: one template per geometry, <256, 2>
  sliding (64 tokens per CTA, 8 tokens x 2 heads per warp, full D = 256,
  32 KiB shared) and <512, 8> full (8 tokens per CTA, 2 tokens x 8 heads
  per warp, two warps per M group split D = 512 into 256-dim halves,
  32 KiB K/V plus 16 KiB score exchange). 32-key tiles, Q in registers,
  f32 online softmax once per tile, P rounded to bf16 for P*V, key loop
  aligned to (begin_all / 32) * 32 for chunk invariance. The dispatch
  ds4_gpu_g4_attn_mma takes rows >= 16 and keys == 0 only; the MTP verify
  widths stay on the bit-exact tile kernel. DS4_GEMMA4_ATTN_MMA=1 enables
  it (default off: parity path, not bit identity).
- Synthetic gate (tests/test_gemma4_primitives, both geometries, rows
  16/17/64, first 0/1000/1025/4096/8192, ring, window, shared-prefix and
  poisoned-slot cases): max abs error 1.4e-3 against g4_ref_attn, inside
  the Phase 8 2.5e-3 tolerance. The test also needed the Phase-3 router
  template argument fixed (g4_router_kernel<false>).
- One-layer model probe (DS4_GEMMA4_ATTN_CMP=1, ctx 256, layer 0 sliding
  and layer 5 full): the attention output against the tile kernel is
  relL2 4.05e-4 and 3.94e-4, max 0.021 and 0.0073. The kernel is correct
  for real model data.
- Pinned 256-token gate (/tmp/g4-bench-256.txt, 8 greedy tokens): both
  paths return exactly [6113, 1698, 236761, 28333, 506, 877, 6910, 3136];
  KL(tile || mma) 8.6e-5 (gate 5e-3), relL2 0.101 (gate 6e-2, 1.7x over
  on the tanh-30 logits). On the repo-docs prompt the greedy text diverges
  after the first sentence and the frontier KL is 0.026 at 256 tokens and
  0.10 at 2,048: routing propagation through 30 MoE layers, not a kernel
  error (per-layer attention is 4e-4 away). Recorded as a measured parity
  trade; the tile kernel is the exact reference and stays the default.
- Speed, Spark, ds4-bench, eight fast options, 17 greedy tokens:
    frontier 2,048: tile 576.1 prefill, MMA 977.1 (1.70x), decode 12.6
    frontier 13,447: tile 309.5 prefill, MMA 1,061.4 (3.43x), decode 4.82
  The 13.4K prefill drops from 43.5 s to 12.7 s of GPU work; the 9.66K
  open-grok prompt from about 34 s to about 10 s. Decode is unchanged
  (rows == 1 uses the decode kernel). The 13.4K prefill clears the 1,000
  tok/s floor the owner set.
- Spark binaries rebuilt (ds4, ds4-bench, ds4-server 13:44, and the probe
  build at 13:59); the server is restarted on the committed build with the
  eight-option launcher and draft 2. The worktree HEAD is still 341481b2.

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
containers, 4.36 GiB, relL2 39.05-39.73 percent) for the 13-bit artifact,
and /data/vq-artifacts/gemma-4-26b-a4b-it-8x8192x1-family with the residual
sidecars (7.01 GiB, mean 23.3772 percent) behind
/data/vq-artifacts/gguf-family-it-r8. VQ_tools dev 0fd5f9d carries the recipe
(recipes/gemma-4-26b-a4b-family-r8.json) and the encoder/emitter (c859d1f).
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
Reference host: GB10 at leandro@192.168.1.91 (gx10-e1d2, aarch64, 121 GB
unified, driver 580.178.04, 10 GbE), checkpoint at
/mnt/models/gemma-4-26B-A4B-it, venv /home/leandro/gemma4-ref-venv (torch
2.14.1+cu130, transformers at cc1451df). Load of the 48.07 GiB bf16 model
takes about 9 minutes; one reference process per gate run, and never two at
once on the 121 GB of unified memory.

6. Risks that must stay visible

- Quality is measured only at the gate's resolution. The reference checkpoint
  is 48.07 GiB on disk (51.6 GB), not the 7.29 GiB artifact: 42.54 GiB of it
  is experts, 5.53 GiB everything else (attention 2.33, dense FFN 1.75,
  embedding 1.38, norms and routers 0.07; measured from the two safetensors
  headers, 1013 tensors). The gate ran on the GB10 (section 4e): four probes,
  the reference argmax on all four, cosine 0.980-0.988, KL ref||port
  0.0004-0.0037. That bounds the artifact against the checkpoint at four
  sequence lengths and nothing more: no greedy-text comparison, no
  perplexity, no benchmark, and no claim about the answers the model gives.
  The artifact those numbers describe measures 39.29 percent on its experts;
  the R13+8 build in section 4f measures 23.38 and was compared on two of the
  same probes (A: cosine 0.9906, KL 0.00055; B: 0.9901 with the reference
  argmax). Both are approximations of the checkpoint, so the residual in
  every cosine and KL number is the encoding's error, not the port's.
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

4q. The 2026-11-03 evening round: the direct VQ prefill landed (+30.8
    percent), the Q8_0 dense down tail is in progress

- Commit 9e23197f, pushed to JLFN/ds4-gemma4 (d6ad9774..9e23197f). New
  cuda/gemma_vq_prefill.cuh plus the one-pass runner g4_moe_run_prefill in
  ds4_gemma4_gpu.cuh; DS4_GEMMA4_VQ_PREFILL=1, default off. The container
  format is unchanged; the records are the existing R13+8 ones.
- Design: one work item per (expert, 128-row tile, 32-token tile), one
  launch per layer over every routed expert (about 600 gate/up items and
  2200 down items at 256 tokens). A block decodes its weight tile from
  the GVQ1 bytes into shared bf16, stages 32 tokens x 64 columns of bf16
  activations, and each thread keeps a 4-row x 4-token f32 micro-tile.
  Gate/up are fused with the GELU epilogue; the output is bf16-rounded at
  the boundary glmvq_gemm applies. The gather and scatter keep the
  reference's grouped order (m->order), so each output row sees the same
  sequence of expert contributions.
- Measured, 4070 ctx 256 copy-resident, three fresh-process runs each:
  off 338.14 / 329.50 / 351.22 tok/s (median 338.14), on 425.49 / 450.61 /
  442.21 (median 442.21). Median +30.8 percent, mean +29.4 percent, the
  ranges do not overlap. Kernel attribution: the gate/up item kernel fell
  from 7.50 to 3.24 ms per launch and the down from 3.66 to 1.38 ms after
  the load fixes (per-layer expert cost 11.16 -> 4.62 ms).
- ncu on the first version (606 blocks, 256 threads): occupancy 33.3
  percent theoretical / 32.2 achieved (111 registers and 39 KB shared,
  2 blocks/SM), warp cycles per issued instruction 13.54 with 8.3 (61.3
  percent) L1TEX scoreboard stalls, DRAM 3.78 percent, L2 60.54, L1/TEX
  53.65, SM compute 28.0. The codebook lookups missed L1 behind the two
  shared tiles; the fixes were uint2 codeword loads, the residual book in
  shared (2 KB, or 4 KB for the pair), two threads per weight row with
  four contiguous vectors, and the gain hoisted out of the K loop.
  Registers after: 128 fused, 80 single; shared 43.3 / 23.8 KB.
- Numerics: DS4_GEMMA4_VQ_VERIFY=1 compares against restore + cuBLAS on
  the live records. Gate/up relL2 1.9e-3 to 2.9e-3 (max 7.8e-3 to 1.6e-2),
  down 1e-5 to 1e-9. The same harness runs the reference against itself at
  half the token count: relL2 0.0 to 3.3e-3. The direct kernel therefore
  sits inside the reference's own chunk-width band; the residual
  difference is cuBLAS's shape-dependent split-K, not a decode error.
  Eight greedy tokens are identical off/on; the frontier logits differ by
  relL2 2.14e-1 (the accumulated gate/up order difference), which is the
  accepted class for this port.
- Evidence record: docs/gemma4-vq-prefill-2026-11-03.md. Advisor review:
  agy_opus_review.md (Claude Opus 5.5, tmux session 15); the Gemini 4
  Argon review (session 14) is in that session's transcript and agrees.
  Both rank the next units: (1) the Q8_0 dense down to a tiled MMQ, about
  135-140 ms per chunk; (2) the prefill attention (double exp first, then
  a group-shared tile kernel), about 120-130 ms; (3) the remaining VQ
  levers (one weight tile at a time to free L1, double buffering); (4) an
  MMA rewrite of the VQ prefill, days of work with a smaller ceiling.
- Q8_0 dense down, measured and REVERTED: the Gemma dense down is
  2112x2816 at 256 tokens, K = 8*256 + 64, and ran the naive
  one-block-per-row-token preq kernel at 0.63 TFLOP/s (4.8 ms per layer,
  21 percent of prefill). The fused MMQ tail was extended to a 64-value
  tail (tail_k 64/128 through the loader, tile processor, worklist kernel
  and launcher; the dense entry accepts K % 256 == 64; the dispatch gate
  relaxed to in_dim % 64 == 0). It worked and was numerically right: the
  eight greedy tokens were identical to the baseline. But it was 1.6x
  SLOWER: prefill 211.46 tok/s against 338.14 with the preq kernel, the
  worklist path logging "dense q8 K=2112 batched on the fused MMQ tail
  path". The two files (cuda/mmq/ds4_mmq.cu, ds4_cuda.cu) were reverted;
  the preq kernel stays. Do not re-attempt without a profile of the tail
  worklist at this shape: the per-call cost, not the arithmetic, is the
  suspect (30 calls, about 20 ms each).
- Spark: the tree is synced at 9e23197f and builds (make cuda-spark, 239 s,
  rc 0). The verify + A/B + parity protocol was handed to a monitor
  subagent; results pending at this handoff. Note: ctx 1024 does not fit
  with the model copied (lazy session graph alloc failed), so a longer
  prefill on the 4070 needs zero-copy; the Spark has 128 GB and can take a
  larger ctx.
- The bus question to the GPT 6 Astra session (01a1a2bf, dashboard session
  5) was accepted; its reply had not arrived at this handoff.

4r. The 2026-11-03 second round: attention tiled, dense down batched,
    the direct-prefill scatter made deterministic

- Commits 2faef6c3 (the round), 14f27345 (the token tile) and the two
  handoff commits, pushed to JLFN/ds4-dfm-rs (9e23197f..4829b404). Evidence:
  docs/gemma4-attention-round-2026-11-03.md. No GLM or Ling-3.1 file is
  touched; the container format is unchanged.
- Correctness first, and it was a real defect: the GB10 run of 4q found
  the direct VQ prefill was not run-to-run reproducible. g4_moe_scatter_kernel
  does dst[target] = bf16(dst[target] + src*w); the direct path ran it once
  over all pairs, so the eight experts of one output row were read-modify-
  written by eight threads in the same launch. Three pairs of identical
  runs differed by relL2 0.085 to 0.125 on the frontier logits while the
  per-expert path was bit-reproducible (off vs off2 0.0). The new
  g4_moe_scatter_ordered_kernel uses one thread per output element and
  walks the token's eight grouped rows in ascending expert order (the
  reference's per-expert loop order), rounding after each add, so the
  accumulation is the reference's own sequence. The 4q claim that the
  frontier difference was only cuBLAS accumulation order was wrong for
  the scatter; the direct path now differs from the reference only in the
  GEMM itself.
- expf: the oracle's two exp(double) per key became expf, which is the
  reference's precision class (modeling_gemma4.py:839, softmax with
  dtype=torch.float32). The FP64 pipe is 1/64 rate on these cards.
  Attention 139.8 -> 70.2 ms of GPU time at ctx 256.
- cuda/gemma4_attn_tile.cuh, DS4_GEMMA4_ATTN_PREFILL=1: one CTA per (row
  tile, KV head), 8 warps sharing one staged K/V tile (4 rows x 2 heads on
  the sliding layers, 1 row x 8 heads on the full ones), uint4 staging,
  16 KiB of shared. Prefill attention 33.6 -> 9.5 ms (25 x 0.28 + 5 x
  0.50-0.70 ms). BIT-IDENTICAL to the oracle on the live model: the whole
  262144-logit frontier dump matches byte for byte.
- cuda/gemma4_attn_fast.cuh, DS4_GEMMA4_ATTN_FAST=1: the decode kernel's
  staging loop now copies uint4 instead of one bf16 per thread. Decode
  attention 36.7 -> 5.8 ms (0.18 ms sliding, 0.24 ms full). Also
  bit-identical to the oracle on the live model.
- ds4_gpu_matmul_q8_0_dense_batch, DS4_GEMMA4_Q8_BATCH=1: the dense down
  is Q8_0 with K = 8*256 + 64, which the shared dispatcher's tile path
  refuses, so it fell to matmul_q8_0_preq_kernel (one block per (row,
  token), only K/32 of 256 threads active). The new entry runs the
  existing batch warp8 kernel for that shape (16 rows per block, 16
  threads per row, one token per block). 138.2 -> 65.1 ms. The fp32 sum
  order changes: eight tokens identical, frontier logits relL2 5.10e-2
  with the same argmax and the same top-5 set. A second kernel in the same
  entry, matmul_q8_0_preq_batch_warp8_tok_kernel<4>, then tiles four
  tokens per weight read: 65.1 -> 51.8 ms (TOK 8 measured 1.88 ms per
  layer, worse, so 4 stays). Against the batch kernel the frontier logits
  move relL2 3.89e-2, tokens identical.
- Measured, ctx 256 copy-resident, three fresh-process runs per arm, all
  four options against none: prefill 395.51 -> 543.19 tok/s median (+37.3
  percent, mean +39.6), decode 13.12 -> 24.76 (+88.7 percent). Total GPU
  kernel time 511.5 -> 305.0 ms (288.0 with the token tile). The off arm
  is itself faster than 4q's 338.14 because expf serves the oracle path
  too; the like-for-like number is the same-binary A/B. A second A/B after
  the token tile ran on a host with load average about 4.4 and gave
  362.94 -> 613.20 (+69 percent) with a wide spread; its off median moved
  8 percent with no code change, so treat the end-to-end medians as noisy
  and the kernel attribution as the evidence.
- GB10 (same round, R5, same protocol, monitor subagent): build clean,
  two full-stack gen-8 runs byte-identical, same eight-token array;
  prefill 274.44 -> 422.55 tok/s median (+54.0 percent), decode 13.47 ->
  16.89 (+25.4 percent), first token 0.074 -> 0.058 s. The GB10's decode
  gain is smaller than the 4070's because the attention was a smaller
  share of its decode step.
- Still the largest kernels at ctx 256 with the whole stack on: the VQ
  gate/up item kernel 100.0 ms, the dense down 51.8, the VQ down 42.3,
  rope 18.1, the Q4_K projections 15.6, the tile attention 9.5.
- Not done: the GB10 has not been re-measured with this round's kernels;
  the token-tiled dense down (the batch kernel re-reads each weight row
  once per token, L2-served) is the next unit for that shape; the exact-
  prefill gate (28 minutes) has not been re-run with the new kernels, only
  the frontier byte-identity check.

7. Open items outside the port

- Pushes: feat/gemma4-family is pushed through 4829b404 (9e23197f..4829b404,
  ls-remote confirmed, OSV scan 146 packages / 0 vulnerable). Any further
  push still needs the owner's word and a fresh scan.
- docs/recipe-v41-identical.md is deleted in the VQ_tools working tree and
  was not committed by this session.
- The Hugging Face upload (JLFN/gemma-4-26b-a4b-VQ-GGUF) must publish the
  whole release directory, not one file.

8. Staleness check

    cd /data/ds4-gemma4 && git log --oneline 8c2068b5..HEAD
    git status --short          # expect review_agy.md untracked plus the
                                # handoff commit. If ds4_gemma4_mtp.inc or
                                # the speed files are untracked again, read
                                # 4k and 4l.
    git ls-remote origin feat/gemma4-family   # expect 4829b404 (pushed)
    ssh leandro@192.168.1.91 'cd ~/ds4-gemma4 && git log --oneline -1 && \
      ls ~/models/gguf-family-it-r8/release.json'
                                # the Spark worktree content is at 2faef6c3
                                # (this round's build); 14f27345, the token
                                # tile, is not synced there yet. Its git HEAD
                                # still reads d6ad9774 until a pull
    cd /data/VQ_tools && git log --oneline -2    # expect 65ef74d, b47dae6 on
                                                 # dev; the profile unit is on
                                                 # feature/gvq1-gemv-profile
    cd /data/VQ_tools && git log --oneline dev..feature/gvq1-gemv-profile
                                # expect be8c7e6, 42974fc
    cd /data/VQ_tools && git status --short      # the owner's other sessions
                                                 # may hold files here; do not
                                                 # commit or revert them
    cd /data/VQ_tools && git log --oneline 434e8b5..HEAD
    # the state blocks, from the working tree
    sed -n '1,60p' gemma4-port-handoff.md
    # the gate, one command (R13+8 is the target; the copy is the measured
    # residency lever, section 4l)
    make cuda CUDA_ARCH=sm_89 && \
      DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock DS4_GEMMA4_DEBUG=1 \
      DS4_CUDA_COPY_MODEL=1 \
      ./ds4 -m /data/vq-artifacts/gguf-family-it-r8 --cuda -c 2048 \
      --token-ids 2,818,5279 --predict 2 --temp 0

9. The remaining phases, in order, and the files that carry each

Order change (owner, 2026-11-03): Phase 7 runs before Phase 4. The phase
numbers and their content are unchanged; only the execution order moves.

Phase 0 is closed (f0c2b92a). The measured speed order after the second
round (2faef6c3) is: the dense down and the VQ expert GEMM first (65 and
143 ms of the 305 ms GPU time at ctx 256), then the remaining decode
primitives (the 704 block variant, the vocabulary head, RoPE, the
router), then the per-GPU profile consumer. Phases 5-6 are the residency
options. Phase 7 closes the port. R13+8v2 is dropped: the next prefill
kernel works on the existing R13+8 records and must not introduce a new
container version.

Phase 0 — CLOSED: the 406-token prefix gate (f0c2b92a)

  DS4_GEMMA4_EXACT_PREFILL=1 passed every checklist item (prefix and rewind
  exact, banks 16/16, 28 minutes against 5 batched). The batched prefill's
  chunk-width arithmetic, not prefix bookkeeping, caused the row-16 flip.
  The prefix item now has a reuse-free reference at the same split; rewind
  keeps the two-sync reference. The synthetic prompt's banks still report
  10/16 because they expose only token streams; the exact run proves the
  bank path. The original 406 prompt is lost; /tmp/g4-bench-256.txt is the
  reconstruction (prompt 406, split 203, grid 512). The remaining Phase 0
  work is only the wrapped-ring gate from 4k, which is now Phase 7's P8.

Phase 1 — decode attention: DONE as an option (bf6b3989); the prefill
  side landed in the second round (2faef6c3, see 4r)

  cuda/gemma4_attn_fast.cuh is a CTA-per-KV-head kernel: the group's query
  heads (2 sliding, 8 full) share one shared-memory K/V tile, with the
  oracle's arithmetic. It is decode-only (rows == 1, keys == 0) and
  DS4_GEMMA4_ATTN_FAST=1, default off. Measured: 4070 ctx 256 copy 11.33 ->
  18.09 tok/s (1.60x), 4070 ctx 2048 zero-copy 1.59 -> 2.42 (1.52x), GB10
  ctx 256 12.46 -> 13.73 (1.10x); 8 greedy tokens identical off/on on each
  machine. The uint4 staging change (2faef6c3) took the decode attention
  from 36.7 to 5.8 ms at ctx 256 and the served decode from 13.12 to 24.76
  tok/s in the same-binary A/B. The prefill side is
  cuda/gemma4_attn_tile.cuh (DS4_GEMMA4_ATTN_PREFILL=1): one CTA per (row
  tile, KV head), 8 warps sharing one staged tile, 33.6 -> 9.5 ms and
  bit-identical to the oracle. Both remain default-off; folding them into
  one option and the ctx 8192 bench on both cards is the remaining
  decision.

Phase 2 — fused MoE: the decode gate/up option is in (659a152b, neutral);
  the direct VQ prefill is DONE (9e23197f); the attention and the dense
  down are DONE as options (2faef6c3); the token-tiled dense down is next

  The fused decode gate/up + GELU is bit-identical but neutral on the 4070
  (decode 13.29 -> 13.24 tok/s): the decode is bandwidth-bound on the VQ
  bytes, so the saved launches do not move it. The 704 block variant (two
  256-index blocks plus a 192 tail) and the fused scatter are still open.
  The prefill side is done as an option: cuda/gemma_vq_prefill.cuh with the
  one-pass runner measured +30.8 percent median prefill at ctx 256 on the
  4070 (338.14 -> 442.21 tok/s), eight tokens identical, and the per-GEMM
  difference inside the reference's own chunk-width band. See 4q and
  docs/gemma4-vq-prefill-2026-11-03.md. The Q8_0 dense down through the
  fused MMQ tail was tried and reverted: numerically right (identical
  tokens) but 1.6x slower at K=2112 (211.46 against 338.14 tok/s prefill),
  so the preq kernel stays.
  The second round (2faef6c3, see 4r and
  docs/gemma4-attention-round-2026-11-03.md) closed the prefill attention
  (tile kernel, 33.6 -> 9.5 ms, bit-identical), the decode attention
  (uint4 staging, 36.7 -> 5.8 ms, bit-identical), the oracle's double exp
  (expf, 139.8 -> 70.2 ms) and the dense down (batch warp8, 138.2 -> 65.1
  ms), and fixed the direct-prefill scatter race. Whole stack against
  none, ctx 256: prefill +37.3 percent, decode +88.7 percent.
  Next, in order: (1) DONE (14f27345): the dense down is token-tiled
  (2.17 -> 1.72 ms per layer); (2) the VQ expert GEMM, measured in 4s: the
  decode is 52-59 percent of it and is latency-bound on the 64 KB
  codebook, not conversion-bound. The next unit is one weight tile at a
  time plus a 64 KB shared carve-out (64 KB of L1 for the book) with the
  activation and index streams marked .cg; the tensor-core arm
  (cuda/ds41_vq_prefill_mma.cuh is the reference) follows it; (3) the GB10
  re-run of this round; (4) the exact-prefill gate with the new kernels.
  Do not change the container format.

Phase 3 — CLOSED (4t): the vocabulary head, RoPE and the router

  Four Gemma-only units, all default-off and byte-identical on the frontier
  logits: the fused norm+rope (DS4_GEMMA4_FUSE_ROPE), the parallel router
  top-k (DS4_GEMMA4_ROUTER_PAR), the family's BF16 head with the fused
  softcap (DS4_GEMMA4_HEAD_GEMV) and the family's router dot
  (DS4_GEMMA4_ROUTER_DOT). Interleaved median: decode 24.61 -> 32.76 tok/s
  (+33.1 percent), prefill 576.9 -> 605.0 (+4.9), TTFT 39.5 -> 31.6 ms.
  Record: docs/gemma4-phase3-2026-11-03.md. The Q4_K projections and the
  dense Q8_0 down are still sibling kernels; that is the next ownership
  unit, separate from this phase.

Phase 7 — the port gates that remain (runs before Phase 4, owner's order)

  P8 IN FLIGHT (4u, 4v). Closed: the exact wrapped-ring gate on the Spark
  (1081 tokens, ring 1029, off/on identical), the default chunk-16 pair
  (self-consistent; the chunk dependence is the known grouped-prefill
  trade), the serving checklist (pass, works 10), the draft matrix
  (no MTP 6.75, draft 2 12.87, draft 4 15.99, draft 6 18.42 tok/s at ctx
  1024; six drafts is the ceiling now, 7e67cf1b), and the real MTP trial
  gate (d77cbc97; 24 tokens, 4.80 tokens/call, prefix/snapshot/stop/
  sampling equal, default-path state drift 1.297).
  DONE (b09744a6, 4w): the Gemma-owned verify head plus device argmax.
  cuda/gemma4_head.cuh takes 1..8 rows with the fused softcap and bf16
  staging, ds4_gpu_g4_head_rows dispatches them, gemma4_mtp_verify makes
  one fused call (per-row shared linear kept as the fallback), the ids come
  from ds4_gpu_argmax_rows_tensor over r->verify (28 B), and the commit
  reads only the kept row. tests/gemma4_head_rows pins 0 ULP for 1..8 rows
  on sm_89 and sm_121. Measured GB10 draft 6: 18.38 -> 20.21 tok/s
  (+10.0%); plain decode unchanged. The DS4_GEMMA4_HEAD_GEMV arm alone is
  6.76 -> 6.76 (no decode lever).
  Remaining P8: the serving-default speed number (the bench above runs
  DS4_CUDA_COPY_MODEL=1), and the exact-path byte gate
  (DS4_GEMMA4_EXACT_PREFILL=1), which fails identically on the Spark
  baseline at 341481b2: the exact switch does not cover every batched
  piece of the verify forward, so the block KV or norm still drifts. That
  is a known limitation; close it by extending the exact coverage to the
  verify forward or by gating on the frontier logits instead of the whole
  payload.
  P9 DONE on the GB10 (2026-11-04): tests/gemma4_serving_http_gate.sh
  passes every check there (boot listening + CUDA backend on NVIDIA GB10
  sm_121, /v1/models, non-stream completion, field-by-field determinism,
  SSE stream through [DONE], max_tokens -1 -> 400 invalid_request_error);
  the MTP arm was skipped. On the 4070 the server refuses to boot:
  quote_overflow at ctx 2048 and 1024 with --mem-floor-gb 2 and 1, and
  --ssd-streaming takes the shared GLM-5.3 branch ("requires one GLM-5.3
  GGUF shard") instead of the Gemma discrete map. That is a real
  discrete-serving gap to record, not a gate failure: the quote accounting
  for the host-mapped 9.37 GiB weights, or the Gemma map behind the server
  flag.
  P10: the HF release directory, the OSV scan (clean 2026-11-04: 1
  manifest, 146 packages, 0 vulnerable), the pushes (owner's word).
  P11: the served-arm comparison and two-stage support in vq-gguf and the
  ds4 GVQ reader.
  Files: tests/gemma4_serving_http_gate.sh, docs/gemma4-vq.md, README.md,
  the families row, the VQ_tools vq-gguf and the ds4 GVQ reader for the
  two-stage payload.

Phase 4 — the per-GPU profile consumer (the owner's request)

  VQ_tools half: run the sweep and commit profiles/gemma-4-26b-a4b-it-r8/
  with the profile and a README holding the command and the measured
  rates. The command is in docs/profile-optimizer.md on
  feature/gvq1-gemv-profile. Then a GB10 sweep for the second profile; the
  Spark already has the VQ_tools tree at ~/VQ_tools, the Gemma worktree at
  ~/ds4-gemma4 and the artifact at ~/models/gguf-family-it-r8.
  ds4 half: new cuda/gemma_vq_profile.h (the VQPROF 1 parser including
  gres), a loader fed by DS4_VQ_PROFILE or a --vq-profile flag, the
  fingerprint bridge (load against the VQ payload directory, or embed the
  payload FNV in release.json at emit time), and geometry -> variant/width
  selection in g4_moe_linear_one. The device name in the profile is the
  final check: a 4070 profile must not load on a GB10 or the reverse.
  Files: cuda/gemma_vq_profile.h (new), ds4_gemma4_gpu.cuh, ds4.c (engine
  option and flag), crates/ds4-core (option validation, quote),
  docs/gemma4-vq.md.
  Gate: the 4070 profile selects the sweep's variant, the bench matches
  the sweep's ordering, and the wrong-GPU profile is refused by name.

Phase 5 — SSD streaming: measured on both cards (4v)

  GB10, draft 6, 4 GiB pool: resident 424.08 prefill / 18.53 decode at ctx
  1024 and 289.90 / 8.53 at ctx 8192; streamed 182.65 / 17.88 and 247.55 /
  8.38. Streaming costs 57% and 14.6% of prefill and 3.5% / 1.8% of
  decode, so it is a capacity lever, not a speed lever. The expert set is
  7.03 GiB (the old 5.9 GiB comment is stale).
  The 4070 discrete-card map is DONE (27221680, ds4_gemma4_map.inc):
  resident spans device-resident, experts host-cold, one 128-byte device
  range per cold record for the bind. Measured at ctx 1024, 4 GiB pool,
  draft 6: resident 74.75 / 4.34 / 1.543 s; streamed 51.47 / 13.11 /
  0.527; with DS4_WEIGHT_RESIDENCY_MTP=eager 51.54 / 13.28 / 0.515; the
  frontier logits and greedy ids are bit-identical. A 3,840-slot pool
  (7.01 GiB) fails the graph budget at ctx 1024 even with the drafter
  host-pinned; 4 GiB is the operating point. The open follow-ups are in
  offload_review.md: pinned staging, the mmap fast path, the 48-slot
  verify-width pool, admission accounting, and the --ssd-streaming plus
  --mtp map clobber.
  Files: ds4_gemma4_stream.inc, ds4_gemma4_map.inc, ds4.c, ds4_cuda.cu,
  ds4_gemma4_graph.inc, ds4_gpu.h, crates/ds4-core/src/ssd.rs and
  serving_host.rs.

Phase 12 — --moe-cache-mib, the SLRU active-expert VRAM cache (next, owner 2026-11-04)

  The owner's instruction: add llama.cpp's --moe-cache-mib as a phase.
  What exists already: the Gemma SSD map has a bounded device slot pool,
  an LRU with pin-before-miss admission, pread from the model fd and the
  stats vocabulary (ds4_gemma4_stream.inc:2-13, 41-44, 112-128, 486-517).
  The 4 GiB / 2,189-slot default measured 250,746 hits, 2,694 misses,
  4.92 GiB read, 0.60 GiB reread, 17.5 s I/O on the 4070 (Phase 5).
  What it lacks against llama.cpp: the --moe-cache-mib flag itself, the
  SLRU split (probationary + protected segments), and overlap of the
  preads with the uploads.
  Unit 12.1 — the flag. Add --moe-cache-mib N (MiB) to ds4-server and
  ds4, mapped onto the existing ssd_streaming_cache_bytes; precedence
  over --ssd-streaming-cache-bytes, default 4 GiB on the 4070, and the
  resolved pool printed in the plan. Files:
  crates/ds4-server/src/bin/ds4-server-rs.rs (parse),
  crates/ds4-core/src/serving.rs (ServingRequest), serving_host.rs (plan
  and quote), ds4.c (native option), ds4_gemma4_stream.inc (sizing).
  Gate: the flag sizes the pool in the plan JSON, a 4070 SSD run boots,
  and the frontier logits and greedy ids stay bit-identical to the
  current cache.
  Unit 12.2 — the SLRU. Split the slots into probation and protected
  (a target protected share, start at 80/20), admit a miss to probation,
  promote on a hit, evict probation LRU first then protected LRU; keep
  the current LRU behind --moe-cache-policy lru for the A/B. Files:
  ds4_gemma4_stream.inc (slot struct, gemma4_stream_find/victim, the
  admission loop), serving.rs (the policy field). Gate: same prompt,
  byte-identical logits/ids, hit rate not worse, decode not slower; A/B
  the reread bytes and the I/O seconds.
  Unit 12.3 — overlap. A pinned staging buffer and a second stream for
  the preads (offload_review.md follow-ups 1 and 2); measure the 17.5 s
  I/O. This is where llama.cpp's prefetch stream maps onto our pread.
  Risk: the current cache is measured, and a wrong protected share can
  evict the hot set. Keep the policy selectable, default the new one only
  after the A/B, and keep the frontier byte gate as the regression.

Phase 13 background — the --cpu-moe feasibility and measurement (2026-11-04)

  The owner's 2026-11-04 instruction: do what llama.cpp does with
  --cpu-moe, i.e. keep the always-active parts on the GPU and compute the
  routed experts on the CPU. The feasibility check (2026-11-04) found no
  CPU GVQ decoder in this tree: every expert GEMV is CUDA
  (cuda/gemma_vq_decode.cuh:74-93, cuda/gemma4_moe_fused.cuh:26-77), the
  router is CUDA, and the family declares HostNeed::Cuda. The Rust vq3
  oracle (crates/ds4-core/src/vq.rs:346-400) is a different container with
  a per-row index window and is not wired to a forward.
  The port would need: a host GVQ1/GVR1 header parse, the flat 12-bit plus
  one-plane index decode, the 8192-word E4M3 book (64 KiB) and the
  256-word residual book (2 KiB) per layer host-side, a GEMV that
  reproduces the reference bf16 rounding and warp reduction order, and one
  D2H/H2D of the 2816-wide row plus a sync per layer. That is a medium
  port and it would lose byte identity, which the project's gates rely on.
  The deciding measurement before committing to it: a standalone CPU GEMV
  over one 408 KB expert record, to see whether 1.43 G weights per token
  (240 experts x 3 matrices x 1.98 M) lands under the device path's ~94 ms.
  MEASURED 2026-11-04 (tests/gemma4_cpu_expert_gemv.c, the real layer-5
  expert-0 down_proj record: 408,448 B GVQ1 + 247,936 B GVR1, 1,982,464
  weights; the program verifies both headers and shapes and reads the
  books from the artifact): 28 threads 0.387 ms/matrix, 1.160 ms/expert
  (three matrices), 278.4 ms per token for 240 experts; 16 threads 285.3,
  8 threads 409.9, 1 thread 2273. Effective 1.70 GB/s, so the scalar
  decode is compute-bound, not memory-bound. The device path's ~94 ms is
  3x faster: the deciding measurement does NOT clear the bar. It would
  take a roughly 3x faster AVX2 decoder just to break even, and the CPU
  arm would still lose byte identity and pay the D2H/H2D per layer.
  Recommendation: stop before the port; the active-expert VRAM cache
  remains the mechanism. If the owner overrides, the next unit is an AVX2
  decode microbenchmark (four vectors per iteration, no E4M3 table), not
  the forward port.
  The active-expert VRAM cache (Phase 5) is the mechanism already measured
  on the 4070.
  REORDERED by the owner on 2026-11-04: no more MTP trials on the 4070
  until this path is implemented, so this phase moves ahead of Phase 8.
  Order after the measurement: P9 is done (GB10 green; the 4070 lane now
  boots after the quote fix, 16k FP8 device-resident, 20.95 tok/s), and
  the CPU GEMV measurement above fails its bar, so the host decoder port
  is not the next unit. The next phase is Phase 8 (the tensor-core
  attention kernel, MIN_ROWS=16, a prefill lever with no draft-6 value).
  The owner's "no 4070 MTP until CPU-MoE" rule has no implementation to
  wait for; the active-expert VRAM cache is the measured 4070 mechanism,
  and MTP on the 4070 can be revisited on its own terms.

Phase 13 — --n-cpu-moe, static CPU placement for the first N layers (after 12)

  llama.cpp's --n-cpu-moe N keeps the first N layers' MoE weights in the
  CPU and computes them there (common/arg.cpp:2778-2786,
  ggml/src/ggml-cpu/ggml-cpu.c:1483-1741); --cpu-moe is the same for all
  layers. It is a fit mechanism, not a speed one. The feasibility and the
  deciding measurement are in the Phase 13 background block above: no
  host GVQ decoder exists, the scalar GEMV is 278.4 ms/token, and agy's
  review (cpu_moe_agy.md) puts a corrected AVX2 decoder at 38-55 ms/token
  against the VRAM cache's 9.29 ms/token, with MTP verify at 7 rows at
  180-250 ms/step. So Phase 13 starts with its gate, not the port.
  Unit 13.0 — the AVX2 microbenchmark (the gate). Persistent threadpool
  outside the timed loop, pre-expanded FP16 books, 4-8 vectors per
  iteration with independent accumulators, and a cold 472 MiB DRAM ring
  (agy's four fixes to tests/gemma4_cpu_expert_gemv.c). Decision: if it
  lands at 38-55 ms/token, the port is defensible as a default-off
  capacity option; if not, Phase 13 stops and the record says so.
  Unit 13.1 — the host decoder. GVQ1/GVR1 header parse, the 13-bit index
  plus plane plus residual byte, the 8192-word and 256-word books
  host-side, a GEMV with the reference bf16 rounding (or tolerance
  gates), and one D2H/H2D of the row per layer.
  Unit 13.2 — the option. --n-cpu-moe N places the first N layers'
  experts on the CPU and the rest in the VRAM cache; default off; the
  plan prints the split; a tolerance-parity gate and a per-layer timing
  gate. Risk: slower than the GPU by construction, and MTP verify
  multiplies the CPU work; only the VRAM freed justifies it.

Phase 6 — RAM-resident experts (optional, capacity not speed)

  The same slot/LRU module with the host mapping as the source instead of
  pread. On this card the model fits (9.37 GiB mapped, about 10.66 GiB peak
  of 12.28), so this buys room for a longer KV, not tok/s. Do not fold it
  into phase 5.
  Files: ds4_gemma4_stream.inc (a host-mapping source), ds4.c (option),
  crates/ds4-core (admission and quote).
