# Gemma-4 26B A4B VQ: the phase plan for the complete port

This is the execution plan for finishing the family port. The state carrier is
`gemma4-port-handoff.md`; this file is the sequence. Each phase states what it
touches, the gate that proves it, and what a fresh agent must read first. A
phase is done only when its gate has been run and its output pasted into the
handoff.

Branch: `feat/gemma4-family` in /data/ds4-gemma4 (fork JLFN/ds4-dfm-rs,
base 8c2068b5). Sibling repo: /data/VQ_tools (branch dev).

## Standing facts a fresh session must not re-derive

Build: `make cuda CUDA_ARCH=sm_89` (CUDA 13.3 on this host; without the arch
the tree does not compile, and `.ds4-cuda-config.mk` now records sm_89).
Primitive gate: `make test-gemma4-primitives`. Bind gate:
`make test-gemma4-bind GEMMA4_INVENTORY=/data/tmp/gemma4-release-inventory.txt`
(the inventory is produced by `vq-gguf list` over the release directory).

Paths: artifact `/data/vq-artifacts/gguf-family-it` (release.json, trunk +
embedding + mtp); checkpoint `/data/models/gemma-4-26B-A4B-it`;
config/tokenizer copy `/data/models/gemma-4-26B-A4B-it-hf`; drafter
`/data/models/gemma-4-26B-A4B-it-assistant-hf`. Reference sources are pinned
by hash in `docs/gemma4-reference-semantics.md` and were read from
transformers cc1451df3b5b218295bc16803569256c53c9d6e2; the scratch copy under
/data/tmp/gemma4-ref may be gone — re-fetch with the hashes in section 1.

Runtime facts, verified: every VQ family declares HostNeed::Cuda and there is
no CPU GVQ decoder, so this family is CUDA-only by the tree's own standard. The
shared linear handles BF16/Q4_K/FP8/GLM_VQ; the dense down_proj's Q8_0 goes
through `ds4_gpu_matmul_q8_0_tensor` straight from the map (no dequantisation,
no extra memory).

## P0-P5: done

- P0 artifact and Rust contract (identify, validate, layout, serving_caps).
- P1 tokenizer and chat template, 15 reference id cases and 3 conversations.
- P2 native shape, dispatch, pointer bind, 12177-tensor bind gate.
- P3 reference semantics pinned and recorded with hashes and citations.
- P4 primitives and attention, gated against a double-precision reference.
- P5 device engine (ds4_gemma4_gpu.cuh, ds4_gemma4.h) and the 30-layer graph
  (ds4_gemma4_graph.inc); both units compile, no session reaches them yet.

## P6: the serial session (closed; the reference gate passed)

State at 2026-11-03: the engine binds the plan, the session allocates the
graph (KV 0.23 GiB at ctx 2048 with one-row forwards), runs all 30 layers and
returns 262,144 of 262,144 finite logits; two identical runs are
byte-identical. The nan of the previous state was three primitives reading the
artifact's BF16 weights as float32 (7dd5b2d8). The prefill spread is closed
(19f25d46): three paths ordered their sums by the row count (Q4_K projections,
the Q8_0 dense down, the expert route group and the router); each now repeats
the one-row call per row. The 41-token prompt's logits are byte-identical at
chunks 1, 2, 8, 9 and 22, and the 1243-token prompt is byte-identical between
chunk 1 (ring 1025, 218 wraps) and chunk 256 (ring 1280, no wrap), sha256
6f2f12688c8bd0127003fceadd1d216f7c5cf847ea1b0e1204b3932e83a4b0eb, so the
sliding ring and the grouped prefill are both exact past the window.

The reference gate ran on the GB10 against the 48.07 GiB checkpoint with the
pinned transformers (cc1451df, modeling_gemma4.py sha256 257dc8a7). Four
probes, all with the reference argmax: 28 tokens (cosine 0.9879, KL ref||port
0.0024), 1024 tokens (0.9801, 0.00038), 1025 tokens (0.9800, 0.00038) and
1243 tokens (0.9873, 0.0037); top-10 overlap 6/10, 8/10, 8/10 and 8/10. The
gate found the port rotating the wrong number of pairs, 64 on sliding layers
and 128 on full ones where the reference rotates 128 and 64
(modeling_rope_utils.py:193-265; the reference's own buffers measure 128 and
64 nonzero). That was a position-proportional phase error: invisible at 28
tokens, it put the port's 1243-token argmax on 818 instead of 1509. Fixed,
with a ring golden in the primitive test and a check that pins
`gemma4_rot(0)`/`gemma4_rot(5)` to the reference pair counts. Section 4e of
gemma4-port-handoff.md holds the numbers.

The serving checklist's prefix, rewind and continue items are exact, "result:
pass (works 3, FAIL 0, WRONG 0, undeclared 0)". Section 4b of
gemma4-port-handoff.md holds the earlier measurements and the bisect. All five
gate items PASS.

Goal: one prompt in, one token out, deterministically, through the C engine.

Touch points in ds4.c, mirrored from `DS4_MODEL_FAMILY_LING31_VQ` (grep it;
about 40 sites, far fewer apply here):
1. the engine load path where `ling31_gpu_bind` is called (~74513) — call
   `gemma4_gpu_bind` next to it, after the weights bind;
2. `ds4_context_memory` for the family (two sites: ~37285 and ~45716) —
   return `gemma4_graph_bytes(ctx, cap)` plus the plan's KV;
3. session create/step: give the session a `ds4_gemma4_graph`, call
   `gemma4_forward` for prefill chunks and for decode, and expose the logits
   through the existing sampling path;
4. `ds4_model_apply_host_shape` already selects `DS4_SHAPE_GEMMA4_A4B_VQ`;
   confirm the host shape reaches `config_validate_model` bypass (~73369).

Gate, in this order:
- `ds4 --inspect` or the load path prints the graph line with KV bytes — PASS;
- a first-token run on a fixed prompt twice gives identical ids (determinism)
  — PASS (--dump-logits byte-identical);
- numeric bring-up: every stage finite, and the prefill row-grouping spread
  measured and either removed or recorded — finite PASS, spread removed
  (4b, 19f25d46), and the 1243-token prompt is chunk-invariant;
- `ds4 --serving-check` on the artifact reports the declared items — PASS for
  the reuse items with single-row forwards; the bank and disk items are P7;
- the logits gate against the reference — PASS on the GB10 against the
  48.07 GiB checkpoint (four probes, reference argmax on all four, cosine
  0.980-0.988, KL ref||port 0.0004-0.0037), after fixing the rotary pair
  counts it found. No quality claim beyond those numbers: the artifact is a
  lossy VQ encoding, so the distributions are close, not equal.

## P7: banks, payloads, snapshots, disk KV

Add `ds4_gemma4_session.inc`, `_payload.inc`, `_batch.inc`,
`_bank_payload.inc` following ling31's shape: per-bank graph with its own KV,
the shared scratch borrowed from the owner graph, payload save/restore of the
KV and the RNG/sampling state, snapshot capture/drop, and the disk-KV
directory. Gemma's KV is two kinds (ring 1024+chunk for 25 layers, full
context for 5), so the payload writer must state the per-layer layout
explicitly and the restore must validate it. `ds4_engine_supports_batching`
returns false for this family today (2dca7709 stopped the serving check
crashing in the DeepSeek slab body); the batch include removes that refusal
and replaces it with the real lane.

Gate: `ds4 --serving-check` with banks > 1; a snapshot restore reproducing the
next token; a disk-KV reload reproducing it again; `ds4_context_memory`
agreeing with the allocator's measured use.

## P8: the drafter

`ds4_gemma4_mtp.inc`: the 4-layer assistant (hidden 1024, FF 8192, every layer
consuming trunk KV), pre_projection [1024,5632], post_projection [2816,1024],
its own tied 262144x1024 head (the centroid path is not used:
`use_ordered_embeddings` is false), fixed position per draft round, argmax
chain. The trunk must publish layers 28 and 29's K/V when drafting is on.

Gate: `--mtp-draft` never changes the greedy output; draft rows equal the
trunk's verify rows; a determinism run with drafts on and off.

## P9: serving

Rust side: `crates/ds4-sys` FFI for the new entry points, the family arm in
`crates/ds4-server` (model id, admission, chat rendering through the existing
Jinja path, stop handling with ids 1/106/50, output parsing), `serving_host`
limits and `docs/serving-capabilities.md` regeneration.

Gate: the four-part OpenAI check in the handoff section 1 — GET /v1/models,
POST /v1/chat/completions returns a completion, the same request twice returns
identical ids and text, and sm_89 CUDA runs it.

## P10: docs and release

`docs/gemma4-vq.md` (format, execution, serving, memory table), the
`docs/ds4-dfm-model-families.md` row, README, the Hugging Face release
directory upload, an OSV scan of both repos, and only then pushes (owner's
word required; never upstream to Baekpica).

## Phase discipline

Every phase ends with: the gate command and its output pasted into the
handoff, a Conventional Commit with the body explaining the mechanism and its
citation, and the handoff's "state at this handoff" block rewritten. A phase
with an open gate is recorded as open, never as done.

Risk carried throughout: this family's correctness ultimately rests on the
logits gate, and this host cannot run the 48 GiB reference checkpoint. Until
that gate runs on a host that can, every quality claim stays unverified, and
the engine's evidence is the primitive gates plus structural checks.
