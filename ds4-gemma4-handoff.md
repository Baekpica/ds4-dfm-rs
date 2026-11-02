---
project: ds4-gemma4
plan_start_commit: 6d5121ad
last_updated_commit: 385114c7
branch: feat/gemma4-a4b
remote: fork https://github.com/JLFN/ds4-dfm-rs.git
handoff_written_at_context_usage: 24%
handoff_written_at: 2026-11-02T17:22:52+01:00
session_id: 01a19693-09d4-71d2-b0ac-5c87d02208e0
---

Handoff + executable plan — ds4-gemma4 (2026-11-02)

Note on the frontmatter: it MUST be the first thing in the file, with
bare values (no inline comments after the value) — the handoff-hooks
parser reads it verbatim, and a commented or misplaced frontmatter
silently breaks the SessionStart probe readout.

1. Session protocol

This project is a sequence of units; each unit ends with a verified commit
set. On completing a unit: fold its outcome into section 2, replace section 3
with the next unit's plan, append the project memory entry, then call
compact_session as the last action of the turn. The auto-compact threshold is
the failsafe, not the trigger. Do not checkpoint mid-work: only write or
replace this file at a clean, verified, committed boundary (a push is a
separate owner decision, rule 10).

2. Current state (2026-11-02 — COMPLETE, verified)

- Newest unit: the block path's ring crossing is fixed and gated (385114c7).
  A ring block now stages its rows as f16, attends from staging plus ring, and
  commits after attention (native/gemma4/gemma4_gpu.cu:1359-1404, :6134-6142);
  the reference shape is /data/ds4-dfm-rs/cuda/ds41_attn.cuh:39-46, :377-401.
- Gate: --block-cross, two plans, the walk as oracle. Before: 1280 1.609e-1,
  1536 plans 2.689e-1 vs 2.188e-1, 2048 1.841e-1 top-1 537 vs 15981. After:
  1280 8.560e-3, 1536 both 2.976e-1 and A vs B 0.000e0, 2048 3.066e-2 top-1
  537 vs 537. Inside the window 25 tokens still give 6.859e-5.
- Recipe finding: the YoungAi recipe VQs the weights only (one 12/13-bit
  index per 8 weights, one shared codebook per layer, experts only, q4_K
  skeleton) and does NOT VQ the KV (FP8 SWA ring + FP4 main + KVC store).
  Our artifact is 2 stages (3.10 bpw), per-tensor codebooks, every 2-D matrix;
  our KV is dense VQ at 14.30/14.34% relL2, 17.9 vs 14.2 ms/token.
- State: three disjoint lanes (gate step + VQ KV; gate block + f16 KV; server
  + f16 KV walk). Commits 385114c7 and 6d5121ad are local, not pushed.

3. Next unit plan — reach the house serving path (2026-11-02)

Goal: Gemma-4 A4B reaches the shape MiMo VQ and GLM VQ already have: an
OpenAI host that prefills a prompt as blocks over a house KV cache, reuses
the common prefix across requests, and is registered in serving_caps. The
VQ weight artifact stays; the VQ KV is not assumed to be the recipe.

Live ground truth, verified this session:
- The block path is ring-safe and gated (385114c7); it has no production
  caller. The server's sync walks the prompt one token at a time
  (crates/ds4-server/src/bin/gemma4-server-rs/gemma4_engine.rs:607-621).
- The sibling families have seven layers: catalog/identity, artifact contract,
  native bind, timeline (block prefill + decode + KV + snapshot), serving
  surface, KV persistence (KVC store, prefix reuse), live qualification.
  GLM VQ and MiMo VQ pass all seven; MiniMax VQ has five, one unqualified.
  Gemma-4 has artifact + kernels + a partial serving surface.
- No sibling family uses VQ codebooks for KV: GLM is a BF16 latent (DSA),
  MiniMax is NVFP4 pages decoded to BF16 in shared memory, MiMo VQ is FP8
  E4M3 full rows plus f16 SWA rings. The VQ KV is local to this repo.

Phase A — block prefill in the server (first, it is mechanical now)
Files: gemma4_engine.rs (declare gm4_gpu_model_prefill after :146, add a
prefill method to GpuModel, rewrite sync :607-621), maybe main.rs for one log
line. No trait change: DecodeIo::sync is enough.
Implementation: reset the caches; chunk the prompt at GM4_PREFILL_MAX = 1024;
call gm4_gpu_model_prefill(&tokens[at..at+take], at, logits) per chunk; on
any -1 fall back to the token walk and log one line (the geometry refusal at
gemma4_gpu.cu:6144-6146 is the expected error). Keep the last chunk's logits;
they are the sample point, exactly as the walk's last step is today.
Acceptance: make tests/gemma4_server builds; a 2048-token chat through
/v1/chat/completions gives the same first token and greedy continuation as a
forced-walk run (add GM4_SERVER_WALK=1 to force the walk); the 25-token gate
still gives 6.859e-5; TTFT recorded with and without the block path. Note the
per-layer host readback (moe_items_build, gemma4_gpu.cu:6000-6090) still
serializes each layer; measure before claiming a TTFT win.

Phase B — the KV recipe decision (f16 now; house codecs later)
Facts: the engine's KV is FP8 SWA ring 128 rows and FP4 E2M1 + E4M3/16 main
KV, quantized after RoPE, with a persistent store of token history + global
KV (890 B/token) and the SWA rebuilt by replaying 128 tokens
(/data/YoungAi/docs/decode.md:42-44, :176-180). Our VQ KV is 14.30% sliding
and 14.34% full relL2 from one 512-token prompt, uncalibrated (gain 1.0), and
3.8 ms slower per decode token.
Option 1 (recommended): keep f16 KV for the server now; port FP8 SWA and FP4
main only when the capacity gate needs it. Files: native/gemma4/gemma4_gpu.cu
(kv_put/attn), .h.
Option 2: keep the VQ KV as an experiment, calibrate it (activation-weighted
codebook, per-channel scales, more than one 512-token prompt), free the f16
cache, and add the block VQ path. Do not make it the default before a gate.
Acceptance: logits and continuation NLL against the f16 walk on the prompt
set; decode ms/token; VRAM.

Phase C — KV store and prefix reuse
Implement SerialKvIo for the engine: sync_start, live_len, sync_with_prefill_
checkpoints, save_payload, load_payload_range (crates/ds4-server/src/
generate.rs:425-466); use the KVC format and store in /data/ds4-dfm-rs/
crates/ds4-kv/src/format.rs:12-19 and store.rs; the server plumbing reference
is generate.rs:4407-4516. Acceptance: the live gate pattern of
tests/serving_reuse_live.py (seed, append, edit, branch, restart, cold) over
HTTP; the reused prefix is the common prefix; a fresh-process restart restores
and continues; a wrong-identity store is refused.

Phase D — VQ caches in the block path (only if Phase B chooses VQ)
The block layer is f16-only today: layer_prefill_dev takes gm4_kv*
(gemma4_gpu.cu:6113) and the VQ branch lives only in layer_forward_dev
(:5046-5057). Add kvq_k/kvq_v to gm4_model_spec, an encode+attention_vq
branch over the batch in layer_prefill_dev, and free the f16 cache when the
pair is present (today the layer still requires a valid f16 cache,
:4993-5002). Acceptance: block prefill with VQ KV matches f16 KV logits
within the model band; VRAM shows the f16 cache gone; the decode cost is
recorded.

Phase E — registration
Add the family to serving_caps (Variant, reuse/disk/MTP as Support::None
until gated), the render syntax (the Gemma-4 thought channel), and the model
ids; crates/ds4-core/src/serving.rs, crates/ds4-server/src/render.rs,
docs/gemma4-a4b-registration-2026-11-02.md. Acceptance: the capability
generator test passes; /v1/models lists the family; a chat round trip.

Phase F — the weight recipe decision
Measure the current 2-stage all-matrix artifact (3.10 bpw, 9.57 GB) against
a YoungAi-style build: one stage, one codebook per layer shared by the
layer's matrices, routed experts only, with a q4_K or single-stage VQ skeleton
for attention/head. The toolkit needs a per-layer sharing mode and an
experts-only selection (VQ_tools plan/build; today the converter selects every
2-D matrix, /data/VQ_tools/AGENTS.md). Acceptance: same-top/KLD/NLL on the
held-out set plus size, recorded as the product decision.

Phase G — end-to-end gate
A live OpenAI chat on the VQ artifact with the chosen KV and Phase A's block
prefill; compare the first token and greedy continuation with the gate; run
the reuse/restart receipts; OSV scan before any push.

Verify, package, handoff:
- Build: TMPDIR=/data/tmp/build-tmp make tests/gemma4_gpu_gate and
  TMPDIR=/data/tmp/build-tmp make tests/gemma4_server.
- Gates: the 25-token block bisect (6.859e-5); --block-cross at 2048 with
  GM4_PF_LEGACY=1 (both plans 0.000e0, top-1 all matching); the server chat
  vs the forced walk.
- Commit per phase with the repo's seven message rules; the last commit of
  the unit carries "Unit: <name> complete".

4. Project facts a fresh session cannot re-derive

- Artifact: /data/vq-artifacts/gemma-4-26b-a4b-8x4096, 657 entries, 9.57 GB,
  dim 8 / nc 4096 / 2 stages / DQV3 weights; the router is the only dim-4,
  8-bit matrix. Tokenizer: /data/models/gemma-4-26B-A4B-it/tokenizer.json.
- Model config: crates/ds4-core/src/gemma4_a4b/config.json (30 layers, 25
  sliding window 1024, 5 full; hidden 2816; 16 heads, 8/2 KV heads of
  256/512; vocab 262144; softcap 30; embed scale 53).
- Gate binary: tests/gemma4_gpu_gate (make target above). Server binary:
  tests/gemma4_server.
- KV captures and containers (machine-local, do not commit paths):
  /data/tmp/kv-capture/{sliding,full} and /data/tmp/kv-vq-{sliding,full};
  layer scope, 2 stages, 12-bit, relL2 14.30% and 14.34%.
- Reference: /data/YoungAi (weight recipe TECHNICAL.md:124-196; KV recipe
  docs/decode.md:42-44, :176-180); /data/ds4-dfm-rs (serving path:
  crates/ds4-server, crates/ds4-kv; block prefills: ds4_mimo2_session.inc,
  ds4_glm53_graph.inc).
- GPU: RTX 4070 SUPER, 12 GB, ~1.5 GiB free after the residency; nvcc at
  /usr/local/cuda/bin; TMPDIR must point at /data/tmp/build-tmp.
- Branch/remotes: feat/gemma4-a4b; fork JLFN/ds4-dfm-rs (push target);
  upstream Baekpica/ds4-dfm-rs (never push).

5. Guardrails and open items

- Push pending: 385114c7 and 6d5121ad are local; a push needs the owner's
  word and an OSV scan (rule 10). Branch feat/gemma4-a4b goes to the fork,
  never the Baekpica upstream.
- Owner decisions: keep the 2-stage all-matrix weight artifact or build the
  YoungAi recipe; adopt FP8/FP4 KV or calibrate the VQ KV; default the block
  path now.
- Open: the block layer is f16-only; the VQ cache does not free the f16
  cache; the server has no container flags; no store/prefix reuse; the family
  is not in serving_caps.
- Risk: the 1536 block-vs-walk relL2 (2.976e-1) is the block's accumulation
  order, identical across plans, not the ring; do not use it as a quality
  verdict without the NLL gate.
