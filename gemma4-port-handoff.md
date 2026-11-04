---
project: ds4-gemma4 (Gemma-4 26B A4B dense-VQ R13+8 port into ds4)
plan_start_commit: 9dd5137e
last_updated_commit: 99775f2c
branch: feat/gemma4-family
remote: https://github.com/JLFN/ds4-dfm-rs.git (public fork; upstream untouched)
handoff_written_at_context_usage: 21%
handoff_written_at: 2026-11-04T20:42:00+01:00
session_id: 01a19693-09d4-71d2-b0ac-5c87d02208e0
---

Handoff + executable plan — ds4-gemma4 (2026-11-04)

This file was reset to the template structure on 2026-11-04; the
previous accumulating body (sections 4a-4ai, 187 KB) is in git history
at 9dd5137e. History is not replayed here: git log and the project
memory entry are the record.

0. Rules for writing THIS document

- Only write or replace this file at a clean, verified, committed
  boundary. Never checkpoint mid-work.
- Every claim in section 2 must be re-verified in the writing session by
  running the relevant command, not copied from the previous handoff. If
  it cannot be re-run, mark it unverified.
- Hard caps: section 2 max 15 lines, section 5 max 15 lines. Section 3
  has no cap; it is the work spec.
- Do not replay prior units' logs, re-explain global rules, or record
  acceptance evidence not observed in this session.
- When a unit completes, replace section 3 wholesale with the next
  unit's plan and fold the outcome into section 2. Rotate, do not append.
- The handoff-hooks parser reads the frontmatter verbatim: bare values,
  first thing in the file, no inline comments.

1. Session protocol

This project is a sequence of units; each unit ends with a verified,
committed, pushed commit set. Benchmarks run on the Spark only (section
5). At the end of a unit: write the next unit's plan into section 3,
execute, verify, commit, offer the push, then compact this session in
place with the compact_session tool. No new open-grok session is opened.
The auto-compact threshold is the failsafe, not the trigger.

2. Current state (2026-11-04 — COMPLETE, verified)

- Newest unit 99775f2c: host experts (CPU_MOE=1), host-sourced slots
  (MOE_HOST=1), and the cache-miss fallback (MOE_MISS=cpu) with the
  ordered add. At ctx 131072 on the 4070, decode tok/s cold/warm: CPU
  13.6; 4 GB slots from disk 25.8/37.8; from host pages 21.8/34.8; a
  128-expert cache 10.8/13.0 (evict) vs 10.5/12.6 (miss->CPU). Cache
  size, not the miss policy, is the lever. One-token decode is
  bit-identical to the GPU; longer contexts differ by 1-2 bf16 ULPs
  (route flips, the GPU's own prefill band).
  docs/gemma4-cpu-moe-runtime-2026-11-04.md.
- Prior: 788cd338 the same unit's first half; b6ca88c2 fair CPU ceiling
  (15.31 ms cold vs 9.37 ms floor); 5bda5aac split-K tensor cores.
- 4070 endpoint UP: 127.0.0.1:8099 pid 3325649, ctx 131072, 4 GB
  host-sourced slots. Spark 192.168.1.91:8889 runs the old rsync tree;
  ~/ds4-gemma4-git builds from git at 6affa69f.
- Metal: no Gemma-4 support (0 VQ refs); the host path is x86 AVX2 plus
  CUDA copies, not portable as written.
- Bench rule: Spark only; 4070 serving/manual. feat/gemma4-family;
  origin 6affa69f; local commits unpushed (owner's word + OSV scan).

3. Next unit plan — typed option surface for the Gemma knobs (2026-11-04)

Status: not started. Owner priority: every knob added this session is a
hand-written getenv string read in C.

The problem (owner, this session): DS4_GEMMA4_ATTN_SPLIT_SPAN,
DS4_GEMMA4_ATTN_SPLIT_MMA, DS4_GEMMA4_CPU_MOE, DS4_GEMMA4_MOE_HOST,
DS4_GEMMA4_MOE_MISS and DS4_GEMMA4_CPU_MOE_THREADS are read by hand in the
Gemma C/CUDA headers. None is validated, none is listed by --check-config,
and a typo reads as zero or false instead of failing. Rust already has a
typed option surface with proper errors and a plan JSON; these belong there.

Scope: those six, plus the older knobs in the same style that the inventory
finds (ATTN_FAST, ATTN_PREFILL, ATTN_MMA, MOE_PERSIST, ROUTER_PAR,
ROUTER_DOT, HEAD_GEMV, Q8_BATCH, VQ_PREFILL, EXACT_PREFILL, FUSE_ROPE,
PREFILL_CHUNK). Leave genuine diagnostics (DEBUG, VERIFY, DUMP, HASH_*)
as env vars.

Work:
1. Inventory: grep -rn "DS4_GEMMA4_" over the C/CUDA tree; one line per
   knob with file:line, default, accepted range and what it changes.
2. Rust: add typed fields to the serving and bench request
   (crates/ds4-core) and the native bridge. Validation: span 16..4096
   (reject rather than clamp), threads 1..64, modes as enums
   (split_mma off/on, miss evict|cpu, host_slots off|on), booleans parsed
   strictly ("1"/"0"/"true"/"false"; anything else is an error that names
   the knob).
3. Engine: the C side reads the validated request fields instead of
   getenv; the getenv helpers stay only for the diagnostics.
4. Plan JSON and --check-config: echo every knob with its effective value
   and list it in the config check, so the quote shows what will run.
5. Gate: a typo in any knob makes --check-config fail with a named error;
   the plan JSON shows the effective values; a bench run driven by the
   typed surface is byte-identical (tokens and frontier logits) to the
   env-driven run it replaces.

Then (unchanged plan, after this unit): the span 64/32/16 sweep at ctx
13,447 on the Spark, using the 6affa69f build, decides whether the span
128 default moves. The measured table, the decision rule and the Phase A
commands are in git at 74c44b3e (previous section 3).

4. Project facts a fresh session cannot re-derive

- Repository: /data/ds4-gemma4, branch feat/gemma4-family, remote
  https://github.com/JLFN/ds4-dfm-rs.git (public fork). Upstream is not
  touched. The Spark worktree ~/ds4-gemma4 carries the sources by rsync;
  its git HEAD is still 341481b2 and is not the state carrier.
- Artifact: canonical /data/vq-artifacts/gguf-family-it-r8; fast-disk
  copy /home/leandro/models/gguf-family-it-r8 (10.69 GB). The launcher
  prefers the fast copy. Artifact totals: 12,225 tensors, 7.291 GiB
  (trunk 5,730,270,592 B, embedding 1,476,395,648 B, mtp 621,995,136 B).
- Geometry: 30 layers, 25 sliding (window 1024, 16 q heads, 8 KV heads,
  head_dim 256) and 5 full at 5/11/17/23/29 (16 q, 2 KV, head_dim 512);
  hidden 2816, vocab 262144 tied, 128 experts top-8 at FF 704,
  tanh-GELU, logit softcap 30.
- Build: local make ds4 ds4-bench ds4-server (sm_89); Spark the same
  with CUDA_ARCH=sm_121 and PATH=$HOME/.cargo/bin:$PATH. Binaries in the
  repo root.
- Launcher: run-gemma4.sh serve start|stop|status|logs, plan, check.
  Knobs: DS4_GEMMA4_CTX (16384 default), DS4_GEMMA4_KV (fp8 default),
  DS4_GEMMA4_HOST/PORT (127.0.0.1:8099), DS4_GEMMA4_SSD and
  _SSD_CACHE (4GB), DS4_GEMMA4_SPLIT (1), DS4_GEMMA4_MMA (1),
  DS4_GEMMA4_FAST_ATTN (1), DS4_GEMMA4_PERSIST (1), DS4_GEMMA4_MODEL.
  SPLIT=1 adds DS4_GEMMA4_ATTN_SPLIT=1 and
  DS4_GEMMA4_ATTN_SPLIT_MMA=1; the latter can be overridden to 0 in the
  environment.
- Endpoints: Spark 0.0.0.0:8889 (pid 163167, ctx 262144, bf16 KV) and
  local 4070 127.0.0.1:8099 (pid 3290680, ctx 262144, E4M3 full-layer
  KV, SSD 4GB). Open-grok entries: [model."gemma4-vq-spark"] and
  [model."gemma4-vq-4070"] in ~/.opengrok/config.toml, both
  context_window 262144. The launcher prints the 4070 entry for both
  (display only; the listen address is what matters).
- Bench protocol: one GPU job at a time; the Spark server holds
  /tmp/ds4-gemma4-gate.lock, so stop it before ds4-bench and restart it
  after. Benchmarks on the Spark only.
- Key sources: cuda/gemma4_attn_split.cuh (scalar split-K),
  cuda/gemma4_attn_split_mma.cuh (tensor-core full-layer split),
  cuda/gemma4_attn_mma.cuh (prefill MMA), cuda/gemma4_attn_fast.cuh,
  ds4_gemma4_gpu.cuh (dispatch and knobs), ds4_gemma4_graph.inc (trunk
  attention call), ds4_gemma4_mtp.inc (drafter call, keys=start),
  tests/test_gemma4_primitives.cu (synthetic gates).
- Reference reviews: agy_review_decode_20261104.md (rank 1/2/3),
  Agy_phase8_review.md (prefill MMA). Family doc: docs/gemma4-vq.md.
- Memory: ~/.opengrok/memory/vq-tools-7e3aff66/MEMORY.md (the session's
  workspace key; search "ds4-gemma4" there). The project handoff is this
  file.

5. Guardrails and open items

- Global guardrails: ~/.opengrok/AGENTS.md rules 0, 0.1, 1, 2, 4, 8, 15,
  16, 17, 18, 20, 21, 22. Project-workflow rules 3, 5, 6, 7, 9, 10, 12,
  13, 14, 19, 23 in ~/.opengrok/project-work-rules.md.
- Project rules (AGENTS.md): model-specific code stays out of VQ_tools;
  the vision tower is a sidecar; no local absolute paths in commits; do
  not copy tuning profiles across artifacts/GPUs; no quality claims from
  structural checks; Conventional Commits; OSV scan before a push.
- Benchmarks run on the Spark only; the 4070 is the serving lane.
- Leave the five untracked owner files and /data/VQ_tools untouched.
- Open: the span 64/32/16 points at 13,447 (this unit); agy 2.2/2.3 and
  Priority 4 serving; the decode re-profile; SSD lane qualification.
- Open: the push of 5bda5aac and the handoff reset needs the owner's
  word and a fresh OSV scan (origin is at 3c89d43c).
- The 4070's FP8 full-layer KV lane never calls the split function, so
  4ai is inert there; the Spark's bf16 lane is the one that uses it.

6. Progress and staleness check (run first, every session)

    cd /data/ds4-gemma4
    git log --oneline 9dd5137e..HEAD
    git diff --stat 9dd5137e..HEAD
    git log --grep "Unit: .* complete" 9dd5137e..HEAD
The diff shows what changed since this plan began; the grep lists unit
boundaries (each unit's final commit carries the trailer). Trust git
over the plan headings: continue with the first unit whose work is not
in the diff. This document's own reset commit is expected in the log.

Staleness check:
    cd /data/ds4-gemma4 && git log --oneline 9dd5137e..HEAD
Any output means this handoff predates newer commits. Read the log and
diffs before trusting sections 2-5, then update last_updated_commit.
If branch or remote in the frontmatter differs from the repo, stop and
reconcile before continuing.
