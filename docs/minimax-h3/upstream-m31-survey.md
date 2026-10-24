# The upstream M3.1 VQ branch, surveyed for H3

Phase P8.5 of [post-port-phases.md](post-port-phases.md). This page answers
two questions and records the verdicts so they are not re-litigated:

1. Does H3 need a K/V cache, and is the port using one correctly?
2. What in `origin/feat/add-minimax-vq` (MiniMax-M3.1-Flash) transfers to
   MiniMax-H3, and what is rejected by name?

Method: read-only. The branch is checked out at `/data/ds4-minimax-vq-ref`
(clean, `fd7192f4`), like `/data/YoungAi` it is never modified, never merged,
and no file from it is cherry-picked. A mechanism may be re-implemented in h3
code, and every such copy cites `file:line` here and in the code comment.

## 1. What the branch is

`origin/feat/add-minimax-vq` at `fd7192f4` ("Let the MiniMax SSD-tier pool
grow on request"), 31 commits over the merge base `600e465e`. It carries no
H3 file: `git ls-tree -r --name-only origin/feat/add-minimax-vq | grep h3` is
empty.

It is MiniMax-M3.1-Flash, a text MoE: 60 layers, 64 query / 4 KV heads x 128,
per-head Gemma q/k norms, NeoX RoPE on 64 of 128 dims, an MSA block indexer
(block 128, top 16 blocks per KV head), dense FFN on layers 0-2 and 128
routed experts top-4 on the rest, NVFP4 K/V pages, and a DSpark drafter
(`ds4_minimax.h:18-40`, `docs/minimax-vq.md:14-46`).

H3 is a 50-block diffusion transformer over one packed text+video+audio
sequence (`port-plan.md`, E4-E6). The two models share a repository, a CUDA
backend and a host pattern, and nothing else. Autoregressive decode, an
indexer, routed experts and an argmax sampler do not exist in H3.

## 2. The KV audit

H3 has no attention K/V cache, so there is no KV usage to be wrong. What the
reference calls a cache is three resident objects:

| Reference fact | Evidence |
|---|---|
| `h3_cache_info` holds `embedding_entries`, `embedding_bytes`, `prepared_dit`, `video_decoder` | `h3.h:21-27` |
| `h3_cache_clear` frees exactly the conditioning memo, the DiT and the video decoder | `h3.c:53-64` |
| The decoder is a hit only under the same conditioning key and shape | `h3.c:697-723` |
| The conditioning key includes render geometry, frame count, image size, and the first/last/reference files with their sizes and mtimes | `h3.c:143-166` |
| The prepared-DiT key adds shape, steps, layers, core reuse, token reduction, reference RoPE, SSD streaming and the slow-path switches | `h3.c:168-190` |

The DiT recomputes attention over the whole packed sequence at every
evaluation step. `qkv_capacity` (`h3_dit.c:1505-1516`) bounds a scratch
allocation; it is not a cache, and nothing is carried between steps.

The port matches that structure:

- The text tower's `key` and `value` tensors are projection weights
  (`text_forward.rs:458-465`); its activation scratch holds every prompt
  token (`:521-529`), is allocated inside `run` (`:199-200`) and is released
  on return. Encoding is one full-sequence pass.
- The DiT streams block matrices through two slots (`dit.rs` `Streamer`) and
  keeps nothing across steps.

So "correct KV usage" for H3 is these three checks, and none is a cache:

| Check | State today | Owner |
|---|---|---|
| Scratch sized to the sequence and freed per call | holds | E4-E6 |
| A reuse key built from every input that changes the output | missing; the server spawns a renderer per request, so there is no reuse at all | P10.3 |
| The VAE temporal-overlap blend on multi-chunk paths | every gate so far is 22 frames = one chunk, one tile | P9.5 |

A request to "continue a video with its KV cache" would be a new mechanism,
not a port. Editing in H3 is conditioned generation (first/last anchors,
Ref2VA), as P12 states.

## 3. Candidates and verdicts

| Candidate | Where | Verdict | What would reverse it |
|---|---|---|---|
| In-process session owning the engine, with first-use module loads prewarmed at boot | `ds4_minimax_session.inc:1-2`; `d1eff63e`; the 1.3 s of module loads once and the 0.2 s it added to a first request are recorded at `docs/minimax-vq.md:262-264` | Adopt in P10.1. The reference keeps the same three objects resident (`h3.c:697-723`), so this is mirroring, not inventing | nothing likely; the gate is P10.6's two-request timing |
| A reuse key built from every input, media pixels included | `ds4_minimax_session.inc:26-32`: a prompt extends the checkpoint only under the same media tag, so identical pads over other pixels prefill again | Adopt in P10.3 and P12.4. Mirror `h3_conditioning_key` (`h3.c:143-166`), never a prompt-only key | if the measured memo hit rate on the operator's workload is zero, keep the memo but stop advertising it |
| Capacity as a measured knob, swept with miss counts and an identical-output check | `fd7192f4` and `docs/minimax-vq.md:147-150`: at 256K, 128 tokens, 300/107/68 misses, 10.9/4.9/3.3 ms of reads per token, 14.74/15.75/16.71 tok/s, identical tokens | Adopt the method for P9's residency and prefetch-depth sweeps | not applicable; this is a measurement discipline |
| Disk K/V tier: unlinked `O_TMPFILE` file opened with `O_DIRECT`, pinned LRU pool, slot table, Linux AIO for selective reads | `ds4_minimax_kv.inc:1-25`, `:366`; AIO measurement at `:223-231` and `docs/minimax-vq.md:139-143` (0.94 ms on threads, 0.24 ms in one `io_submit`) | Reject for H3. H3 streams 735 MiB per block in a strict ascending ring with no random misses; the transferable contract (reserve at open, 512-byte `O_DIRECT` alignment, event-driven prefetch) is already recorded at `port-plan.md:376-412` | a measured H3 workload that needs random access to weights or state, which does not exist |
| Pinned host memory aliased into the device | `ds4_minimax.h:148-155` (`ds4_gpu_minimax_host_alloc/free`) | Candidate for P9.1/P9.7 only | P9.1 shows the gap is in the staging or copy path, and a probe confirms `pageableMemoryAccess` / `canMapHostMemory` on the Spark. Measure mmap, malloc staging and pinned staging with the same copy API |
| Row count as a kernel template argument | `344dee7d`, `docs/minimax-vq.md:58-60` | Candidate for the text encode, not the DiT. The DiT repeats one shape per render, so a runtime row count costs it nothing; the text encode's rows equal the prompt length and it is 20-30 s of a render (P10.3) | the text encode's attribution shows predicated-row waste at the prompt lengths the server sees |
| PDL on short kernels | `7d9adeba`, `docs/minimax-vq.md:60-63` | Candidate for P9.7 | an `nsys` trace of the DiT step shows exposed launch gaps that PDL closes. Attribute with PDL off, as `AGENTS.md` requires, and keep copies inside the launch chain as kernels |
| Media input plumbing: base64 only, at most four per request, ffmpeg frame sampling, page-cache drop after streaming a tower | `6fd141ff`, `docs/minimax-vq.md:290-328`, `crates/ds4-core/src/minimax_vq/media.rs` | Plumbing only, in P12 | the plumbing is needed at all only once first/last-frame lands. Semantics stay with `/data/h3.c` (`h3_multimodal.c`, `h3_vision_encoder.c`, `h3_video_encoder.c`) |
| Session snapshot payload over a bank graph | `ds4_minimax_bank_payload.inc:1-44` | Reject. H3's only reuse state is the conditioning memo plus residency, both cheaper in RAM than on disk | a measured cold-start cost that the memo cannot avoid |
| Gate record layout: one directory per run, one CSV per sweep, `tables.md`, `summary.json` | `docs/benchmarks/minimax-vq-spark-64k-2026-10-24/` | Adopt for P9 and P11 records. Every number there names its commit and its command | not applicable |
| Kernel-local wins: epilogue gate loads, merged indexer GEMV grid, two-pass host argmax, warp-specialized experts, tensor-core block scoring | `85086da5`, `152fa899`, `2f89a938`, `4d47f9f6`, `0a1f2696` | Reject by name. Each is bound to an M3.1 structure H3 does not have: the indexer, 128 routed experts, an argmax sampler | H3 grows one of those structures, which it will not |
| VQ weights for the DiT or the text tower | `vq-recipe.md`, `docs/minimax-vq.md:14-46` | Out of scope here. They change bytes; that is `port-plan.md` section 5's T2 rung with its own quality gate | that rung's gate, not this survey |

## 4. What this survey does not prove

- It does not measure the branch: every number quoted is the branch's own, on
  its own model and artifact. None transfers as an H3 expectation.
- It does not decide P9.1. The mmap-versus-`pread` question is a hypothesis
  until `tools/minimax_h3/p9_copy_bench.cu` runs on the Spark and an A/B
  follows.
- It does not order P9's targets beyond what `post-port-phases.md` already
  fixes; the three live candidates (pinned staging, row templating, PDL) are
  each behind the measurement named above.
- It says nothing about H3 quality. No weight format or arithmetic changes
  here.
