# DeepSeek V4.1 (ds41) ③ post-train port plan — branch `ds41_post_train`

Owner request (2026-10-13): "posttrain complete, no web GUI and no web chat —
I want to be able to send an HTTP POST to it instead", and on scope: "it must
be fully ported except there is no webchat".

This document is the plan. `ds4-dfm-rs-handoff.md` carries the same plan in its
executable form (section 3) and is the session carrier.

---

## 1. Source of truth and pins

| tree | pin read this session | state |
| --- | --- | --- |
| `/data/YoungAi` (branch `restructure`) | `fc54d1e1e4feb056a3dc4fd7204876bc435c8e4b` | the READ-ONLY reference; HEAD of the engine repo |
| `/data/YoungAi-restructure` (detached worktree of the same repo) | `221b9e2ab07aa2012ca0efc96c423eeb4ea7dc9e` | an ancestor of `fc54d1e`, 26 commits behind; NO two-step trainer |
| port working tree | branch `ds41_post_train` at `33a97ce` | this work |

Pin drift that matters, verified by `git -C /data/YoungAi log --oneline
7fb26cc..HEAD -- src/core/core_ptrain.c ...`:

- `AGENTS.md` and port plan §6.25 name `7fb26cc` as the pin; the tree has moved
  26 commits forward to `fc54d1e`. The three commits since `7fb26cc` that touch
  `src/train/` are `ad40a89` (models page: delete local models), `2602853`
  (tunnel status), `10eaf71` (public access: login, API keys, settings, frp,
  HTTPS cert, PWA). They are studio/web work.
- `src/core/core_ptrain*` is UNCHANGED between `7fb26cc` and HEAD (the pathspec
  log is empty), so the trainer bodies at HEAD are the ones §6.24 planned
  against.
- The 10-11 two-step work (`core_ptrain_expand.c`, `core_ptrain_onpolicy.c`,
  `eval=`, the three readouts, `sample_n`) exists ONLY in `/data/YoungAi`. The
  restructure worktree (`221b9e2`) carries the earlier one-step trainer:
  `core_ptrain.h` 166 lines there against 203 at HEAD, no `_expand`/`_onpolicy`
  files. **`misc/posttrain-research-2026-10-11.md` §2 was written against the
  restructure in places and must be re-read against `/data/YoungAi` HEAD.**
- `ds4_posttrain.c/.h` are byte-identical across the trees.
- The engine's own design doc for the two-step round is `docs/posttrain.md`
  (145 lines, 2026-10-11): one round, two steps, three metrics.

Doctrine that governs every unit here: the engine is the authority, every ported
function mirrors it with `file:line` citations, every deviation is named and
measured (`AGENTS.md`, global rule 0.2).

## 2. What the port has today, and what is missing

Verified in this tree:

| piece | state | evidence |
| --- | --- | --- |
| ②③ reader / merge / `base.fnv` gate | DONE | `crates/ds4-core/src/sidecar.rs`, `zchain.rs` (`n_amp_layers` `:148`, `k_range` `:154`) |
| ③ amp APPLY at the forward tail | NOT PORTED, refused by name | `crates/ds4-core/src/v41_run.rs` (`zchain_load_for_run`, the refusal string "carries amp_Lnn.bin; the low-rank amplifier apply is not ported") |
| `--posttrain` flag plumbing (CLI + server) | DONE | `crates/ds4-cli/src/lib.rs:227`, `crates/ds4-server/src/bin/ds4-server-rs.rs:262` |
| native V4.1 forward + score + greedy generate | DONE | `ds41_forward.h` (`ds4_v41_score_ids`, `ds4_v41_generate_argmax`), `ds4_ds41_forward.inc`, `ds4_ds41_*.inc` |
| native V4.1 GPU kernel aggregate | DONE for the inference families | `ds4_ds41_gpu.cuh:39-60` (include-order contract) |
| forward archive (`v41_tsave`: per-layer bf16 `hc_in`, selections, compressor) | ABSENT by design | `ds4_ds41_forward.inc:153` "Dropped from the engine's struct: tsave, graph, snap_*, mainh/draft" |
| backward kernel family (`ds4_gpu_bwd_*`) | ABSENT | no `cuda/*bwd*` file, no `ds4_gpu_bwd_*` declaration in `ds4_gpu.h` |
| trainer (`core_ptrain*.c`) | ABSENT | no Rust or native counterpart anywhere |
| `--score-ids` on the Rust host | ABSENT (native entry exists) | `ds41_forward.h:94` declares it; no `crates/` route |
| studio (`ds4-train`, `/api/train/*`, `/api/models/*`) | ABSENT | no `ds4-train` binary, no train routes in `crates/ds4-server` |
| `ds4_loss`, `ds4_zfinetune` (DQZ2 ③ merge), `ds4_posttrain` module | ABSENT | grep over the tree |
| `anchor_metrics`, `posttrain_reward` tools | ABSENT | not in the tree |

Engine sizes to be mirrored (lines at `/data/YoungAi` `fc54d1e`):

| engine file(s) | lines | port target |
| --- | --- | --- |
| `src/core/ds4_gpu_bwd.h` | the 24-entry contract | declarations in `ds4_gpu.h` |
| `src/cuda/cuda_bwd_{dense,comp,attn,hc,vq,moe}.inc.cu` | 438 / 80 / 296 / 119 / 366 / 228 = 1527 | `cuda/ds41_bwd_*.cuh` |
| `src/core/core_ptrain.h` + `core_ptrain{,_bwd,_data,_diag,_expand,_layer,_onpolicy,_pack,_probe,_teacher}.c` | 203 + 389 + 203 + 367 + 278 + 124 + 393 + 153 + 290 + 136 + 349 = 2885 | `ds4_ds41_ptrain*.inc` |
| `src/core/core_v41_amp.c` (apply/load) | 331 | apply in the native forward tail; the reader is already ported |
| `src/train/*` (25 files incl. `train_internal.h`) | 4103 | new crate `ds4-train` (Rust), minus the HTML |
| `gguf-tools/bench/anchor_metrics.c` + 2 `.inc.c` | 476 + 109 | native tool kept in-tree |
| `ds4_posttrain.c/.h` | 63 + 49 | `crates/ds4-core/src/posttrain.rs` |
| `ds4_loss.c/.h`, `ds4_zfinetune.c/.h` | ~152 + small | `crates/ds4-core` |

## 3. Target architecture

The port's split is fixed by doctrine: **native C + CUDA for the engine work,
Rust for the host and the control plane** (`AGENTS.md`; `docs/rust-migration/ARCHITECTURE.md`).

```
                          /api/train/*  POST      /api/models/*  POST
   operator/script  ────────────────►  ds4-train (Rust, new bin)  ──────┐
                                       owns the port, supervises       │ /v1,/monitor,/metrics
                                       (no HTML, no chat page)         ▼  proxied bytes
                                                              ds4-server --cuda -m ① ...
                                       job thread: stop model → train → gate → restart
                                                              │
                                        ds4 (Rust CLI, new --ptrain <cfg>)   │ ./ds4 --score-ids
                                                              ▼              ▼
                                     native trainer (ds4_ds41_ptrain*.inc)  native score (exists)
                                     + cuda/ds41_bwd_*.cuh                  + anchor_metrics
                                                              │
                                     ③ amp_Lnn.bin + base.fnv  ──► mounted by --posttrain
                                                              ▼
                                     native forward tail: y += x·(B·A), round_bf16
```

Named decisions (each needs the owner's nod because it changes a boundary):

1. **The trainer is native, inside the `ds4.c` TU.** One `.inc` per engine file,
   `ds4_ds41_ptrain*.inc`, included by `ds4.c` after `ds4_ds41_kvreuse.inc`
   (precedent: `ds4_ds41_kvreuse.inc` and `ds4_ds41_forward.inc`). Reason: the
   trainer reads `ds4_v41_state` fields, the per-layer archive and the device
   tensors that the forward owns; the engine gets the same reach through
   `core_internal.h`, and a same-TU include is this port's established seam. The
   alternative (new TUs + a new internal header) adds an internal ABI that the
   port does not have today; rejected unless a compile-time reason appears.
2. **The Rust host drives the trainer through a new native surface header**
   `ds41_ptrain.h`, mirroring the `ds41_forward.h` precedent: slim entries
   (`ds4_v41_ptrain_run`, a progress callback, abort) and no engine layouts in
   Rust. `native/bridge/ds4_bridge.h` is FROZEN ("do not add new
   `ds4_bridge_*` except create/load/session/prefill/decode/KV/destroy") and is
   not touched: the trainer entry belongs on the ds41 surface, exactly as
   `ds4_v41_score_ids` and `ds4_v41_generate_argmax` do.
3. **The ③ apply lands in the native forward tail** (`ds4_ds41_forward.inc`,
   the MoE tail), not in Rust: the engine's apply site is
   `core_v41_forward.c:148-155` (`v41_amp_hook` then `v41_amp_apply`). The Rust
   half already merges and validates the amp rows (`zchain.rs`); it only has to
   stop refusing and hand the rows to the device store.
4. **The HTTP API is a new Rust binary `ds4-train`** owning page-port semantics:
   it supervises the model child and the training child exactly as
   `train_internal.h:1-22` describes (the model is 100+ GB, so training stops
   the model first). `ds4-server` exposes the same routes with `serving=1`
   refusals for spawn operations, as the engine does. The HTML pages are the
   ONLY thing dropped: `/`, `/index.html`, `/studio`, `/train`, `chat.html`,
   `monitor.html`, `logo.jpg` and the page-serving half of the static route.

## 4. Units

Order matters only where a dependency forces it: PT-0 → PT-1 → PT-3 → PT-4 →
PT-7's job thread. PT-2, PT-5, PT-6, PT-7a (the model-free HTTP surface) and
PT-8 are independent and can be taken in any order.

| unit | scope | gate |
| --- | --- | --- |
| PT-0 | forward archive (`tsave`) back in the native state + the bwd declarations + the C harness skeleton | model-free: struct/offset self-check + archive-vs-dump parity (the port already writes `x_Lnn`/`y_Lnn`/`hce_Lnn` for n ≤ 64) |
| PT-1 | the backward kernel family: `cuda_bwd_{dense,comp,attn,hc,vq,moe}.inc.cu` → `cuda/ds41_bwd_*.cuh`, in the engine's include order, plus the 24 `ds4_gpu_bwd_*` entries declared in `ds4_gpu.h` | model-free: per-entry finite-difference self-test (`tests/test_ds41_bwd.cu`); then the trainer's own `gradcheck`/`hccheck`/`rccheck`/`packcheck` on the Spark |
| PT-2 | the ③ apply: device store for `ampA/ampB/ampT`, the apply kernel (engine `cuda_v41_4.inc.cu:321`, small-batch arm `cuda_draft_attn.inc.cu:418-443`), `round_bf16`, the forward-tail hook; lift the Rust refusal | model-free: device apply vs a CPU reference on a synthetic `amp_Lnn.bin` (+ the existing `tests/zchain.rs`); live: zero-A run bit-identical to no-amp, then a directory carrying an amp loads and serves |
| PT-3 | the trainer (`core_ptrain*.c`, both steps) as `ds4_ds41_ptrain*.inc` + `--ptrain <cfg>` in the Rust CLI | model-free: `pt_cfg_load`/`pt_parse_line` parity on the 14-case corpus (research note §9); live: the engine's four built-in checks, then a 10-minute probe material run |
| PT-4 | the judge: `--score-ids`/`--score-out` host route, `anchor_metrics`, the gate rule, `pick.txt` | model-free: `anchor_metrics` vs the engine's binary on fixed inputs, and the gate rule vs the engine's revised readout (`train_gate.c`); live: two-arm scoring on the wt2 512-token material |
| PT-5 | `ds4_zfinetune` (DQZ2 ③ merge) + `ds4_loss` (the four calibration losses) | model-free: merge parity on a fixture pair; loss parity vs the C module on captured arrays |
| PT-6 | `ds4_posttrain` closed-form 1-bit module (preserved formula; zero callers in the engine, slated for deletion there — `docs/clear.md:113`) | model-free: bit/ULP parity on the engine's own anisotropic fixture (MSE 3697 vs the 8106 baseline, `docs/log.md:672`) |
| PT-7 | the supervisor + HTTP API: crate `ds4-train` (bin `ds4-train`) mirroring `src/train/*` minus HTML; the shared routes on `ds4-server` (`serving=1`); the job thread, child control, watchdog, `/proc` instance lock, `serve_pick.txt`, the live plugin hook | model-free: route/status-code/JSON/error-string parity vs the engine's `ds4-train` (the 14-case upload corpus + every route's shape); live: the full cycle on the Spark |
| PT-8 | legacy completeness the owner asked for ("fully ported"): `posttrain_reward` + the advantage path (`v41_amp_run --adv-list`) and the nightly stages that consume it | the tool's own self-test (`-DPOSTTRAIN_REWARD_TEST`) + a parity run on the recorded fixture. No studio consumer; recorded as legacy |
| out of scope | the ①/② producers (`gguf-tools/amp/zlayer*`, `ds4quant_*`, `amp_solve*`, `v41_*_solve*`). The ② sidecar READER is already ported; these are the quantizer/solver tools, and the request was ③ | named, not silently dropped |

### PT-0 detail

`v41_tsave` is the engine's forward archive (`core_v41.h:62-66`), consumed at
`core_v41_attn.c:132-133` for the frozen-selection replay. Port scope: the
per-layer bf16 `hc_in` (the recompute entry), the engram gate input, the
compressor remainder for the boundary layers, and the per-sample selections
(`topk`/`idx`) used when replay is on. The port's state struct explicitly drops
it today (`ds4_ds41_forward.inc:153`), so this is a forward change, not just a
new buffer: it must be allocated only when the trainer asks for it, so the
serving path's memory profile does not move.

Gate: the port already writes `<out>.x_Lnn.bin` / `.y_Lnn.bin` / `.hce_Lnn.bin`
for n ≤ 64 (`ds41_forward.h:4-8`, the engine's own capture protocol). Archived
`hc_in` for the same ids must equal the dumped `hce`/entry rows bit for bit on
the same run.

### PT-1 detail

Include order is a contract: the engine includes the six files at
`ds4_cuda.cu:104-109` in the order dense, comp, attn, hc, vq, moe, with the
stated constraints (`bwd_dense` after `v41_q4k`/`gemv_highprec`; `bwd_comp`
before `bwd_attn`; `bwd_moe` after `vq_row`/`vq_prefill`/`bwd_vq`). The port's
aggregate root is `ds4_ds41_gpu.cuh`; the six includes go after
`cuda/ds41_sample.cuh` (line 60) in the engine's relative order.

The 24 entries are enumerated in the research note §2 table with their
declaration line in `src/core/ds4_gpu_bwd.h` and their definition line in
`src/cuda/*.inc.cu`; that table is the port's work list. The numerics contract
is engine-stated: f32 row-major `[n][dim]`, bf16 boundary rounding treated as
straight-through (`ds4_gpu_bwd.h:21-22`).

## 5. The HTTP surface (full mirror, minus the HTML)

Routes mirrored from `train_api.c` / `train_models.c` (research note §3 has the
per-route table with params, response and status). Nothing about parameters,
status codes, JSON shapes or error strings changes; only the page routes go.

| method | path | notes |
| --- | --- | --- |
| GET | `/api/train/status` | `{time,mem_avail_mb,port,mode,busy,downloading,phase,run,proc{...},root}` |
| GET | `/api/train/runs`, `/api/train/run?name=` | records, newest-config-first |
| GET | `/api/train/data`, `/api/train/preview?path=&n=` | `n ≤ 50`, ≤ 4 KB per row |
| POST | `/api/train/upload?kind=jsonl\|doc&name=&dir=` | raw body, 64 MB cap, store byte-exact |
| POST | `/api/train/start`, `/api/train/stop`, `/api/train/serve` | job control |
| GET | `/api/models/list?hub=0` | host GGUF + sidecars + loaded trio + fnv + posttrains + pick + hub |
| POST | `/api/models/download`, `/cancel`, `/load`, `/plugins` | plugins = the live ②③ hot-swap |
| any | `/api/auth/*`, `/api/keys/*`, `/api/net/*` | login, API keys, cloud/frp/tunnel/HTTPS cert |
| proxy | `/v1/*`, `/monitor`, `/metrics`, `/api/models/plugins`, `/api/models/list` | byte-proxied to the model child, streaming both directions |
| dropped | `/`, `/index.html`, `/studio`, `/train`, `chat.html`, `monitor.html`, `logo.jpg`, page half of the static route | the only removal |

Two engine-comment facts to preserve rather than unify: the studio's upload
check is deliberately LOOSER than the trainer's (substring test on
`"messages"`/`"text"` vs the trainer's full field whitelist), and the doc route
rewrites `<dir>/<dir>.text.jsonl` whole
(`train_api.c:155-158` vs `core_ptrain_data.c:209-241`).

## 6. Assets and environment (externals)

Research result (2026-10-13, verified by reading both engine worktrees and
searching this machine). None of the training material is in either engine
tree, in git, or on this box; `/data/YoungAi/gguf-tools/data/` holds only
`corpus/`, `expert-masks/`, `layer-tables/`, `prompts/`. A `find` over `/data`
(depth 6) and `$HOME` (depth 4) for `hold.jsonl`, `kvc_train*`, `youngai_facts*`
returns nothing except the probe SCRIPT in the two worktrees. The material
lives on the Spark's engine tree (`fable5.md:10760`: "只在 spark(不进 git)").

| asset | kind | where it must live (engine path) | provenance |
| --- | --- | --- | --- |
| hold material | material | `gguf-tools/data/posttrain/hold.jsonl` | 63 general questions (`hold/questions.txt`) + the deployed model's own answers, converted from `review_v2`'s `hold_general.*.qa` (`fable5.md:9328`, `:10637`). The studio HARD-FAILS without it (`train_job.c:71-72`). 63 questions / 47 rows appear in different eras; the file is authoritative. |
| probe material | material (hand-written) | `gguf-tools/data/posttrain/xlang/` | `kvc_train` 26 rows = the KV file's 48-byte `KVC` header rule written in Python/Go/Rust (4 questions each) + 14 Chinese/English "explain the rule" rows; `kvc_eval_js` 5 rows = the same four things in JS/TS, the held-out language (`fable5.md:10896`). The engine's own verdict on it: "KVC 是我手写的单个 26 行案例…样本太单薄" (`fable5.md:10907`-era). |
| facts material | material | `gguf-tools/data/posttrain/youngai_facts*.jsonl` | 12 project facts × 3 phrasings (36 rows) + a 13-question rephrased probe; the 2-form training file (24) and the eval holdout (12) are its derived forms (`fable5.md:10800`, `:10979`). |
| large corpus | material | `gguf-tools/data/posttrain/review_v2*` | the finance replay: 1326 questions + 63 hold, blocks named by trading day (`fable5.md:10648`). |
| smoke material | material | any small `*.jsonl` | the bit-identical regression uses `jsonl_smoke` with `sample_n=0` and no `eval=` (`fable5.md`, 10-11 entry). |
| wt2 ids | gate asset | `gguf/go-onebit/g7/wt2.ids` | `gguf-tools/scripts/regen_ids.sh:34` over `gguf-tools/data/corpus` |
| FP teacher anchor (265 MB) | gate asset | `gguf/v41judge/teacher_g7_wt2_n512.bin` | needs the HF factory weights (`v41_judge.sh` / `v41_teacher.py`); `release_pack.sh:13` states a pack consumer cannot produce it |
| judge binary | tool | `gguf-tools/bench/anchor_metrics` | in-tree source (476 + 109 lines), built by the port |
| run output | state | `gguf/v41/posttrain/kd-<name>-L<layers>-lr<lr>-e<epochs>-MMDDHHMM` | `train_job.c:79` |
| model pick | state | `gguf/serve_pick.txt` | line 1 = ① absolute, line 2 = ② or `none`, then extra engine args |

The fourth material path is generative: upload documents → `chunks/<base>.txt`
→ the model writes Q&A → `gensplit` (nightly) or the studio's doc route
regenerates `<dir>/<dir>.text.jsonl` (`train_api.c:113-134`). It needs a served
model, so it cannot be the first run's material.

The port today has `gguf-tools/` (the pre-restructure tree: quants, imatrix,
quality-testing) and NO `gguf/` and no `gguf-tools/data/`. PT-4/PT-7 must
create the layout above in the port's install directory; `train_ctl.c:14-20`
gives the three root-relative constants (`tr_root`, `tr_ftd`, `tr_datad`).

The gate harness to copy is `gguf-tools/scripts/studio_posttrain_probe.sh`
(71 lines): it drives ONLY the main process's HTTP surface — `POST
/api/train/upload`, `POST /api/train/start` (its `extra` argument carries
`eval=`), `GET /api/train/status|runs|run`, `GET /api/models/list`,
`POST /api/models/plugins`, `POST /v1/chat/completions` through the proxy — and
records the three answer groups (before, ③ mounted, ③ unmounted). It is exactly
the "HTTP POST, no GUI" flow of this plan, and it is the natural acceptance
harness for PT-7.

Memory numbers in the engine are the engine's: trainer floor
`PT_MEM_FLOOR_MB = 3000`, studio watchdog `TR_WD_KILL_MB = 2500`, training
budget `--mem-budget-mb 110000`, model start floor 100000, post-load floor
10000. The port must re-measure on its own host before adopting any of them
(global rule 20; the port's `install.sh` already carries the start/run/kill
floors for serving).

## 7. Gate ladder

1. Model-free, on the dev box: the parsers, the merge/apply math, the backward
   kernels' finite differences, the HTTP route parity. No GPU, no model.
2. Spark, model-free: build (`make cuda-spark`, `make ds4-server`) and the
   native kernel self-tests.
3. Spark, live: the trainer's own checks (`packcheck`, `rccheck`, `hccheck`,
   `gradcheck`), the 10-minute probe material, then a real run with the three
   readouts, then the wt2 two-arm gate.
4. Cycle: `POST /api/train/start` end to end (stop model → train → gate → pick
   → restart), with the model child serving afterwards on the same port.
5. CI: `.github/workflows/host-parity.yml` stays green; the local gate must
   include a FULL-workspace compile (the recorded lesson: the campaign's local
   checks missed `ds4-cli` and CI's Format failure masked the compile error).

## 8. Risk analysis

Decision: add a training subsystem (native kernels + trainer + a supervising
HTTP service) to a serving-focused port.

External risks (not controllable):
- The gate's assets are not in the repo and two of them cannot be produced
  without the HF factory weights (teacher anchor) or an unverified external
  step (hold.jsonl). Evidence: `release_pack.sh:20-24`, research note §7. If a
  Spark window opens without them, only units PT-0/1/2/5/6/7a can be gated.
- The engine tree moves (26 commits in two days) and the trainer's doc
  (`docs/posttrain.md`) calls itself a 方案稿 (proposal). Evidence: `7fb26cc` →
  `fc54d1e` with `src/train` changes. Mitigation: re-read the trainer at every
  unit start; pin the commit in each commit message.

Decision risks:
- The backward family is 1,527 lines of CUDA with no port counterpart and no
  reference in this tree. Unverified until PT-1's finite-difference gates run.
- `v41_tsave` is a forward change in a route that is currently gated and fast;
  a careless allocation strategy can move the serving memory profile or the
  decode graphs. Evidence: `ds4_ds41_forward.inc:153` drops it deliberately.
- A supervising service that stops the model child touches the operator's LIVE
  deployment. Evidence: the operator's Spark server is live; `install.sh` is
  the only sanctioned lifecycle.
- The studio mirror is ~4,100 lines of C including auth, keys, frp tunnels and
  TLS certificates — the largest single unit, and the least related to training.

Measurement before adoption:
- PT-1 and PT-3 are adopted only on the engine's own checks (finite differences
  and the four built-in checks), not on "it converges".
- The memory floors are adopted only after a measured MemAvailable curve on the
  Spark with the trainer's real buffers allocated.
- The HTTP surface is adopted on route/JSON parity against the engine's binary,
  not on "it returns 200".

Prevention: one unit per commit with its gate in the message; kill switches for
anything new in the serving path (the apply must be inert when no amp is
mounted); the training child runs under the same watchdog and instance lock as
the engine's studio.

Rollback: the branch is additive; every unit is a commit, and the revert is
`git revert <sha>` on this branch. The one irreversible action is a live Spark
deployment; `install.sh stop` plus the backed-up binary is the reverse.

Residual risk: the trained ③ can only be judged by the wt2 materials; if those
are missing, a trained artifact exists with no acceptance evidence and must not
be mounted. That signal is `pick.txt` being empty.

## 9. Open questions for the owner

1. PT-8 (legacy reward): port it for completeness, or record it as dead code?
   The 10-10 trainer deliberately dropped the reward loop (`core_ptrain.h:10`).
2. PT-7 auth/net: the studio's login, API keys, frp tunnel and HTTPS cert exist
   to publish a web workbench. Mirror them (full port), or drop them together
   with the pages they served?
3. RESOLVED (research 2026-10-13, §6): where the trainer's material comes from
   for the first live run. Answer: nothing is in either tree or on this box;
   the material lives on the Spark, and the plan recommends (a) reuse the
   Spark's `hold.jsonl` (a material artifact, not code; re-deriving it means
   re-asking the same 63 questions to a served model and changes nothing),
   (b) author the port's own probe material per `docs/posttrain.md` §4.1 rather
   than inherit the engine's 26-row hand-written `kvc_train`, which the engine
   itself judged too thin to set defaults on, (c) keep a tiny hand-written
   jsonl for the `sample_n=0`, no-`eval=` bit-identical smoke, and (d) verify
   the 265 MB FP teacher anchor's presence before scheduling PT-4's live gate.
   The one thing the research cannot settle is ownership: is the port's
   training corpus the operator's existing material (finance `review_v2`), or
   does the port ship its own?
