---
project: ds4-gemma4 (Gemma-4 26B A4B dense-VQ R13+8 port into ds4)
plan_start_commit: 9dd5137e
last_updated_commit: b6ca88c2
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

- Newest unit b6ca88c2: fair CPU-expert ceiling (review 5.A). 15.31 ms/
  token cold at 20t vs a 9.37 ms streaming floor (1.63x); scalar cold
  117.3 ms; max |diff| 4.8e-7. agy's 8-wide lever regressed 4.8%, 2-row
  blocking was noise; STOP stands, corrected (CPU experts 1.65x the
  GPU's 9.29 ms, not 4-6x). docs/gemma4-cpu-moe-2026-11-04.md.
- Prior unit 4ai (5bda5aac): split-K on tensor cores, gates 4.8e-6,
  32 tokens identical at 256/1,024/13,447, span-128 A/B up to 1.095x.
- Spark builds from git at 6affa69f (~/ds4-gemma4-git, all four
  binaries, 0 errors, split-MMA kernel present); the live server still
  runs the old rsync tree ~/ds4-gemma4. 192.168.1.91:8889 live; local
  4070 127.0.0.1:8099 DOWN (no listener).
- Bench rule: Spark only; 4070 is the serving/manual lane. Branch
  feat/gemma4-family; origin 6affa69f; b6ca88c2 local, unpushed; owner
  files untouched.

3. Next unit plan — span 64/32/16 at ctx 13,447 (2026-11-04)

Status: not yet started. This closes the last unmeasured point of rank
2 and decides whether the split-span default moves below 128.

Ground truth (verified this session):
- g4_split_span() in ds4_gemma4_gpu.cuh: DS4_GEMMA4_ATTN_SPLIT_SPAN,
  default 128, clamped [16, 4096], no fixed ceiling.
- The host computes splits = ceil(visible / span); the split kernel and
  the split-MMA kernel re-derive span = (last - begin + splits) /
  splits and mask [k0, k1]. The span changes only the split count and
  the combine; the arithmetic is the per-split kernel's.
- 4ag measured the scalar split path on the GB10, 64 gen tokens, gen
  tok/s by span:
    ctx 256    512 29.14  256 33.72  128 35.81  64 37.32  32 38.67  16 39.37
    ctx 1024   512 24.73  256 30.39  128 34.16  64 36.42
    ctx 13447  512 22.95  256 28.20  128 30.66
  The 13,447 points for 64/32/16 are the gap. With the 4ai MMA path the
  baseline is 26.38 tok/s at span 128 (24.09 scalar).
- At 13,447 the full layers ask for splits = ceil(13448 / span) per KV
  head, 2 KV heads: span 128 -> 212 CTAs, 64 -> 422, 32 -> 842,
  16 -> 1,682. The MMA kernel is bandwidth-bound and each CTA stages Q
  once, so the optimum may move up or down; measure, do not assume.

Phase A — prep the Spark (bench window)
1. The git clone ~/ds4-gemma4-git is at 6affa69f with a full build
   (ds4, ds4-server, ds4-bench, ds4-eval, 0 errors, split-MMA kernel
   present, built 21:06). Use ~/ds4-gemma4-git/ds4-bench for the sweep;
   do not rsync and do not rebuild. The clone's run-gemma4.sh has not
   been exercised, so keep the serving lane on the old tree for now.
2. Stop the server (it holds /tmp/ds4-gemma4-gate.lock):
     cd ~/ds4-gemma4 && ./run-gemma4.sh serve stop
   Verify pgrep -f ds4-server is empty and port 8889 is closed.
3. Keep the exact restart command for the end:
     DS4_GEMMA4_CTX=262144 DS4_GEMMA4_KV=bf16 DS4_GEMMA4_HOST=0.0.0.0 \
       DS4_GEMMA4_PORT=8889 ./run-gemma4.sh serve start

Phase B — the sweep
- Prompt /tmp/g4-prof-14k.txt (13,447 tokens) and the model at
  $HOME/models/gguf-family-it-r8.
- One arm per span, the eight fast options plus the persistent MoE, the
  split-MMA path on, 32 gen tokens:
    DS4_GEMMA4_ATTN_FAST=1 DS4_GEMMA4_ATTN_PREFILL=1 \
    DS4_GEMMA4_VQ_PREFILL=1 DS4_GEMMA4_Q8_BATCH=1 \
    DS4_GEMMA4_FUSE_ROPE=1 DS4_GEMMA4_ROUTER_PAR=1 \
    DS4_GEMMA4_HEAD_GEMV=1 DS4_GEMMA4_ROUTER_DOT=1 \
    DS4_GEMMA4_ATTN_MMA=1 DS4_GEMMA4_ATTN_SPLIT=1 \
    DS4_GEMMA4_ATTN_SPLIT_MMA=1 DS4_GEMMA4_ATTN_SPLIT_SPAN=$span \
    DS4_CUDA_COPY_MODEL=1 DS4_LOCK_FILE=/tmp/ds4-gemma4-gate.lock \
    ./ds4-bench -m $HOME/models/gguf-family-it-r8 --cuda \
      --prompt-file /tmp/g4-prof-14k.txt --ctx-start 13447 \
      --ctx-max 13447 --gen-tokens 32
  for span in 128 64 32 16.
- Record the ctx,prefill,gen line of each arm. Repeat the winner once to
  bound noise; the AGENTS.md adoption rule wants a repeated gain.
- If the winner is not 128, also run the scalar arm
  (DS4_GEMMA4_ATTN_SPLIT_MMA=0) at the winner and at 128 for the pair.

Phase C — acceptance and decision
- Expected: the finer spans add CTAs and Q staging per CTA. Pick the
  highest gen tok/s at 13,447 that does not regress the 256 and 1,024
  points; re-run 64/32/16 at 256 and 1,024 only if the 13,447 winner
  differs from 128 (the 4ag table is the scalar baseline there).
- If 64 (or 32/16) wins at 13,447 and holds 256/1,024, change the
  default in g4_split_span and the launcher/doc text. If 128 remains
  best at 13,447, keep 128 and record that the question is closed.
- If the default moves, extend the synthetic gate with the winning span
  (attn_split_mma_case already takes a span; add 64/32/16 at 8,192 keys)
  and run make test-gemma4-primitives on sm_89 and sm_121.
- Commit the measured table and the decision:
    perf(gemma4-vq): span <N> default, the 13,447 points
  with the raw ctx,prefill,gen lines in the body. Update the table in
  docs/gemma4-vq.md and this handoff's section 3 replacement.
- Restart the Spark server with the Phase A command and verify
  /v1/models plus one greedy completion.

Phase D — following units (not this one)
1. agy 2.2/2.3 (the MTP drafter) and Priority 4 serving.
2. Re-profile the decode after 4ag/4ai: gen8 minus gen0 nsys at ctx 256
   and 13,447; split what is left between the MoE and the rest.
3. SSD lane qualification: the plan warns ssd_unqualified and there is no
   sustained or long-context decode measurement (4v is ctx 1,024). Use
   an owner-approved method; do not bench on the 4070.

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
