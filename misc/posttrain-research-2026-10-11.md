# Post-training subsystem research report — DeepSeek V4.1 Flash (③ post-train)

Sources: engine `/data/YoungAi-restructure` (JLFN/YoungAi @ 221b9e2, git worktree), older pin `/data/YoungAi` @ 3946dbc, port `/data/ds4-dfm-rs`. Everything below is read from those trees; no file was modified.

## 0. Pin drift between the two engine trees (read this first)

The two engine trees are NOT the same commit, and the difference lands squarely on post-training.

- `/data/YoungAi-restructure` (221b9e2) carries the 10-10 "one JSONL, one command" post-train subsystem: `src/core/core_ptrain{,.h,_bwd,_data,_diag,_layer,_pack,_probe,_teacher}.c`, the studio `src/train/*` (12 files), `web/studio.html`, and the `ds4-train` binary target (`Makefile:85-86,241-242`).
- `/data/YoungAi` (3946dbc) has the SAME file names but an older body (`docs/log.md`-era). Its `core_ptrain_data.c` header describes the 10-01 list/`.qa` format, not JSONL: `core_ptrain_data.c:2` "清单每行 `<块.txt> <问答.qa> <train|eval> [hard|rft|kl] [权重]`" and defaults `c->layer_lo = c->layer_hi = DS4_N_LAYER - 1u` (last layer only), `epochs=2`, `lr=1e-3`, plus `hard_num/hard_txt/rft_scale/eval0/probe0/sample_batch` config keys that no longer exist. Its graphify cache still lists `pt_userq_load`, `pt_chunk_id`, `pt_tag`, `pt_load_qa`, `pt_rft_group` — functions absent from the restructure.
- `/data/YoungAi` has NO `src/train/` directory and no `web/studio.html`; the studio is new in the restructure.
- `ds4_posttrain.c/.h` are byte-identical between the two trees (verified by reading both).
- The restructure also moved root docs into `docs/` (e.g. `docs/log.md`, `docs/clear.md`).

Consequence for the port plan: cite the restructure (221b9e2); the port's own `zchain.rs:3` still cites `commit 3946dbc`, which is stale for the trainer but correct for `core_v41_amp.c` (see §5).

---

## 1. `ds4_posttrain.c` / `.h` — the standalone closed-form module

### Math (the header formula)

`ds4_posttrain.h:12-18` states the lever: output-optimal, activation-aware per-row 1-bit scale. Generic 1-bit uses `s_i = mean|w_i|` (weight-space optimal); this solves the scale optimal in OUTPUT space over calibration activations X:

```
b_i  = sign(w_i)
s_i  = sum_x (w_i . x)(b_i . x) / sum_x (b_i . x)^2
```

i.e. the least-squares regression of the true row output onto the 1-bit row output. Row-major: W is rows×d, X is n×d (`ds4_posttrain.h:20`). The module is declared "closed-form — no gradients, no training" (`ds4_posttrain.h:16`).

Implementation (`ds4_posttrain.c`):

- `ds4_posttrain_row_scale_meanabs` (`:8-12`): the weight-space fallback `s = Σ|w_i| / d`.
- `ds4_posttrain_row_scale` (`:15-33`): the regression. `sign(0) := +1` (`:18-19`: "so the bit pattern is deterministic and matches `ds4_posttrain_quant_row`"). Degenerate guard `if (den <= 1e-30) return 0.0f` (`:34`) — caller must fall back.
- `ds4_posttrain_quant_row` (`:36-42`): `bits[j] = w_row[j] < 0 ? -1 : 1`; scale via `row_scale` when `X && n`, else `row_scale_meanabs` (`:40-41`). Packing is the file format's job, not the math module's (`ds4_posttrain.h:37-39`).
- `ds4_posttrain_row_cosine` (`:45-62`): `cos(Xw, s·Xb)`; note the comment "scale choice cancels in cosine; it reports the DIRECTION recovery" (`ds4_posttrain.h:44-46`). Degenerate den → 0.

### The reference doc (path is stale)

The header cites "`gguf-tools/go-onebit/ALGORITHM.md §6.1`" (`ds4_posttrain.h:2`, `ds4_posttrain.c:2`). No `go-onebit/ALGORITHM.md` exists. The real file is `gguf-tools/docs/ALGORITHM.md`, and §6.1 = section 6, item 1 ("修正方向 → 1. 定制化 Go-aware 1-bit 量化"), at `gguf-tools/docs/ALGORITHM.md:57`:

> `s_go_i = Σ_x(w_i·x)(b_i·x)/Σ_x(b_i·x)²` (用 Go 激活协方差，b=sign(w)；w1/w3 用 x、w2 用 SwiGLU 中间 h)。实测 per-expert base 一致 +0.10（0.55→~0.70，诚实 ≥2-sample 值），纯量化、合规。

The same doc records the surrounding result set: generic 1-bit expert output cos ~0.55 (`ALGORITHM.md:16`), the §6 work order `closed-form first` (`:53-61`), and the tool `validate_fwd.c --goaware` (`:65`). `docs/log.md:672` records the unit test: "输出域 MSE 3697 vs 通用基线 8106" (anisotropic activations, closed-form optimality).

### Callers — none

Grep for `ds4_posttrain_row_scale|ds4_posttrain_quant_row|ds4_posttrain_row_cosine` across the whole restructure returns matches only inside `ds4_posttrain.{c,h}` itself. There are no callers in `src/` or `gguf-tools/`. `docs/clear.md:113` (C0 "dead-code clearance") confirms the intent: root modules and `ds4_posttrain` are scheduled for deletion "按引用零" (zero references). `validate_fwd.c --goaware` is a *separate* implementation in `gguf-tools/calib`, not a caller.

### Where it links

`Makefile:250` `MODULE_OBJS = ds4_z.o ds4_loss.o ds4_posttrain.o`; `Makefile:252` `modules: $(MODULE_OBJS)`; rule at `:260-261`. The comment `:245-249` says these are built with `all` (so they "stay green") but linked only by the tools that consume them — and no such tool exists.

### What a 1:1 port must preserve

- Exact arithmetic order: `double` accumulation, `wx` then `bx` per calibration sample (`ds4_posttrain.c:22-29`), `if (den <= 1e-30) return 0` (`:34`).
- `sign(0) = +1` and the same comparison (`< 0.0f`) in both `bits` fill (`:39`) and the `bx` accumulation (`:27`), so the bit pattern is deterministic and matched between scale and cosine.
- The fallback chain: activation-aware → 0 → `mean|w|` (`:40-41`).
- The cosine convention where scale cancels (`:45-62`).
- No packing, no file format in the module.
- Then mirror the header's citations to the real `gguf-tools/docs/ALGORITHM.md:57`.

---

## 2. The training core `src/core/core_ptrain*.c`

### One `--ptrain <cfg>` run, end to end

Entry `ds4_engine_ptrain` (`core_ptrain.c:178-348`), invoked from the CLI/server `--ptrain` flag (`ds4_v41_api.h:53-56`). Flow:

1. Require a V4.1 model + GPU; refuse if `--ptrain` is combined with `--posttrain` (`core_ptrain.c:179,183`).
2. `pt_cfg_load` (`core_ptrain_data.c:26-91`), `mkdir(out)`.
3. `pt_data_load` (`core_ptrain_data.c:290-317`) reads the JSONL `data=` and `hold=` files, builds chunks, tokenizes, splits hold-out.
4. `pt_teacher` (`core_ptrain_teacher.c:189-230`) computes/caches teacher top-K (skinless for `sft`/one-hot samples).
5. Pre-warm (`core_ptrain.c:209-223`): build the fullest pack, run one full forward+backward to size the on-demand scratch; abort if `MemAvailable < PT_MEM_FLOOR_MB`.
6. Optional checks run serially (`core_ptrain.c:227-241`): `packcheck` → `rccheck` → `hccheck` → `gradcheck`; all but `gradcheck=1` exit after checking.
7. Step-0 evaluation (`pt_eval` for eval/train64/hold), held-out TSV dump, greedy probe (`:242-257`).
8. Epochs loop (`:268-334`): Fisher-Yates shuffle (`:270-277`), batch = `c.batch` samples, `pt_run_list` packs them, gradient norm + global clip, Adam per training layer (`:292-298`), log every 10 steps / last, `max_steps` early-stop.
9. Per-epoch: re-eval, `pt_paired_gain`, `heldout_eNN.tsv`, `ckpt_eNN` via `pt_save_amp`, greedy probes (`:314-333`).
10. After `epochs`, final `pt_save_amp` into `out` and print the mount line `--zchain ② --posttrain ③` (`:336-338`).

### `pt_cfg_load` keys and defaults

Defaults (`core_ptrain_data.c:29-30`): `layer_lo=0, layer_hi=DS4_N_LAYER-1` (all layers), `rank=64, topk=64, epochs=3, batch=4, maxlen=1024, probe_n=6, probe_tok=96, seed=1, dbg_layer=-1, lr=2e-4, clip=1.0, init_std=0`.

Keys accepted (`:40-81`): `data`, `out`, `hold`, `init`, `layers` (`a-z` range or single), `rank`, `topk`, `epochs`, `batch`, `maxlen`, `probe_n`, `probe_tok`, `seed`, `gradcheck`, `prof`, `rccheck`, `diag`, `max_steps`, `epoch_tok`, `packcheck`, `nopack`, `dbglayer`, `poison`, `gclayers` (`a/b/..`), `hccheck` (`a/b/..`, `m` suffix = mid), `lr`, `clip`, `init_std`. Unknown key = hard error (`:81`). `#` comments, blank lines skipped (`:36-37`); every non-comment line must have `=` (`:38`).

Validation (`:86-90`): `data` and `out` required; `layers` in range; `layer_hi` MUST be the last layer `DS4_N_LAYER-1` (gradient path requires it); `rank>0`, `1<=topk<=128`, `batch>0`, `maxlen>=16`. The `pt_cfg` struct is in `core_ptrain.h:22-43`.

### JSONL data contract (`core_ptrain_data.c`)

Accepted line shapes (header `core_ptrain.h:5-7`, parser `pt_parse_line:193-260)`):

- `{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}], "context":"材料(可省)"}` — conversational; optional leading `{"role":"system",…}`.
- `{"text":"…"}` — continued-pretraining; rendered `BOS + 原文 + EOS`, every token is a target (`:141-161`).

Fields: only `context`, `text`, `messages` at top level (`:209-241`). Anything else errors: "只认 messages / context / text 三个字段(reward/mode/split 10-10 起不收)" (`:241`). Inside a message object only `role`/`content` (`:226-229`); `role` ∈ {system,user,assistant} (`:236`); duplicate role errors "只收单轮" (`:238`); missing role/content errors (`:234`). A `text` line may not also carry `messages`/`context` (`:251-253`); empty text errors (`:254`). `messages` must have both user and assistant (`:258`); empty q or a errors (`:260`). Trailing bytes after the closing `}` error "行尾有多余内容" (`:245-246`).

Error handling: one bad line stops the whole run with file+line+reason (`:274-278`); the comment banner says silent skipping would "look like the material was halved" (`:9`). The `hold=` file additionally rejects `context` or `text` lines (`:277`).

Hold-out FNV rule (`pt_is_eval:120-126`): FNV-1a 64 over the *question* text with all whitespace (space, `\n`, `\r`, `\t`) removed; `h % 5 == 0` → eval. Seed `1469598103934665603`, prime `1099511628211`. `text` rows use the whole original text (`:142`). Deterministic across files/runs.

Chunking (`pt_chunk_of:93-111`): one chunk per context (name = `c_%016llx` FNV of the context), `hold`, `sft` (no context), `text`; chunk LCP is the maximal common teacher-prefix computed at load (`:296-307`).

Student/teacher render alignment (`pt_add_sample:164-189`):

- Student ids = `ds4_encode_chat_prompt(sys, q, DS4_THINK_NONE)` (`:169`).
- Teacher ids = `ds4_encode_chat_prompt(sys, ctx + "\n\n" + q, THINK_NONE)` when context present, else identical to student (`:170-176`).
- Answer tokenized ONCE (`ds4_tokenize_text(a)` + EOS) and appended to both (`:177-178,182-183`); `sids`/`tids` share the same answer ids, `sa0`/`ta0` = prompt lengths (`:182-184`), so answer positions align 1:1. Samples whose `sp.len+at.len > maxlen` are dropped and counted (`:179`).
- No-context and `text` samples get an in-place one-hot teacher table (`pt_fill_onehot:128-135`, `sft=1`), which the KL kernel reduces to cross-entropy.

`text` segmentation (`pt_add_text:141-161`): BOS + ≤`maxlen-1` tokens per segment, EOS only on the last segment; segments are independent samples, all on the same hold-out side (FNV over whole text).

### Teacher top-K machinery (`core_ptrain_teacher.c`)

- `PT_TAIL=256`, `PT_TROWS=1024`, `PT_TEACHER_VER=2` (`:13-16`). The teacher runs the deployed ①+② model (not ③) forward on `教师序列 = context + blank line + question + answer` and takes top-K at the answer rows via `ds4_gpu_bwd_topk_tensor(tid,tp,tr, logits, r0, m, V, K)` (`:169`).
- Per-context prefix is pre-computed once and snapshotted (window ring, compressor remainder, counters) — `pt_snap_io:28-42`; the continuation block re-runs the last `PT_TAIL` tokens + question + answer under the full decoder (`:159-171`).
- Per-sample cache `<data>.teacher.bin` and `<hold>.teacher.bin` (`:200-201`), magic `PT_CACHE_MAGIC = 0x324b5450` "PTK2" (`:58`), format `header{K}+records{key u64, m u32, top_id[m·K] i32, top_p[m·K] f32, top_rest[m] f32}`, append-only (`:109-131`). Cache key = FNV over `(PT_TEACHER_VER, K, tn, ta0, base_fnv_lo, base_fnv_hi, all teacher ids)` (`pt_sample_key:68-76`) — importantly including the ② directory fingerprint (`g_pt_base_fnv`, `:62-67`) so swapping the sidecar invalidates teacher tables (the 10-10 fix).
- Top-K = softmax over full vocab, take max K rounds, remainder `1 - Σp`; **the call overwrites the logits rows it reads** (`ds4_gpu_bwd.h:48-50`), so it is called after the loss.

### The backward path (`core_ptrain_bwd.c` + `core_ptrain_layer.c`) and the CUDA entries

Numerics contract: activations and gradients are f32 row-major `[n][dim]`; bf16 module-boundary rounding is treated as straight-through (derivative 1) — `ds4_gpu_bwd.h:21-22`.

CUDA entries called, with the header declaration and the concrete `.inc.cu` definition:

| entry | decl | definition |
|---|---|---|
| `ds4_gpu_bwd_matmul_t_tensor` | `ds4_gpu_bwd.h:27-30` | `src/cuda/cuda_bwd_dense.inc.cu:49` |
| `ds4_gpu_bwd_grouped_matmul_t_tensor` | `:31-34` | `cuda_bwd_dense.inc.cu:89` |
| `ds4_gpu_bwd_kl_topk_tensor` | `:37-45` | `cuda_bwd_dense.inc.cu:200` |
| `ds4_gpu_bwd_topk_tensor` | `:47-50` | `cuda_bwd_dense.inc.cu:259` |
| `ds4_gpu_bwd_rms_norm_tensor` | `:52-55` | `cuda_bwd_dense.inc.cu:282` |
| `ds4_gpu_bwd_hc_pre_tensor` | `:57-61` | `cuda_bwd_dense.inc.cu:307` |
| `ds4_gpu_bwd_hc_post_tensor` | `:62-68` | `cuda_bwd_dense.inc.cu:340` |
| `ds4_gpu_bwd_amp_tensor` | `:69-73` | `cuda_bwd_dense.inc.cu:351` |
| `ds4_gpu_bwd_sumsq_tensor` | `:158` | `cuda_bwd_dense.inc.cu:383` |
| `ds4_gpu_bwd_adam_tensor` | `:165-168` | `cuda_bwd_dense.inc.cu:403` |
| `ds4_gpu_bwd_pack_bf16_tensor` | `:127` | `cuda_bwd_dense.inc.cu:420` |
| `ds4_gpu_bwd_unpack_bf16_tensor` | `:128` | `cuda_bwd_dense.inc.cu:425` |
| `ds4_gpu_bwd_axpy_tensor` | `:157` | `cuda_bwd_dense.inc.cu:434` |
| `ds4_gpu_bwd_sparse_attn_tensor` | `:78-83` | `cuda_bwd_attn.inc.cu:252` |
| `ds4_gpu_bwd_rope_tensor` | `:84-86` | `cuda_bwd_attn.inc.cu:290` |
| `ds4_gpu_bwd_hc_mix_tensor` | `:87-92` | `cuda_bwd_hc.inc.cu:102` |
| `ds4_gpu_bwd_swiglu_tensor` | `:93-95` | `cuda_bwd_moe.inc.cu:19` |
| `ds4_gpu_bwd_router_tensor` | `:96-98` | `cuda_bwd_moe.inc.cu:54` |
| `ds4_gpu_bwd_router_fixed_tensor` | `:113-115` | `cuda_bwd_moe.inc.cu:77` |
| `ds4_gpu_bwd_routed_moe_tensor` | `:99-104` | `cuda_bwd_moe.inc.cu:132` |
| `ds4_gpu_bwd_moe_capture` | `:105-110` | `cuda_bwd_moe.inc.cu:126` (capture body in `cuda_vq_prefill_mma.inc.cu`) |
| `ds4_gpu_bwd_wcache` | `:111-112` | `cuda_v41_q4k.inc.cu:288` |
| `ds4_gpu_bwd_compress_pool_tensor` | `:116-120` | `cuda_bwd_comp.inc.cu:33` |
| `ds4_gpu_bwd_engram_gate_tensor` | `:121-124` | `cuda_bwd_comp.inc.cu:70` |

Exit segment for the last-layer-only case (`core_ptrain_bwd.c:149-165`): `matmul_t(W_headᵀ)` → `rms_norm` → `hc_pre` → `hc_post` → `amp`. Multi-layer exit segment (`:136-148`): same head/norm/hc_pre then `pt_layers_bwd`.

`pt_layers_bwd` (`core_ptrain_layer.c:349-372`) walks `hi → lo`. Per layer `pt_layer_bwd` (`:201-321`):

1. `pt_layer_recompute` (`:102-146`): re-runs the layer from the archived bf16 `hc_in` (or from tokens at L0, `:109-110`), replaying engram gate, `v41_hc_half`, `v41_attn_in`, sparse attention (`ds4_gpu_v41_sparse_attn_tensor`), `v41_attn_out`, `hc_post`, `v41_moe` with capture on (`ds4_gpu_bwd_moe_capture(1)`, `:144`).
2. ffn half backward: `hc_post` → `amp` → shared-expert `matmul_t` ×2 + `swiglu` → routed MoE (`ds4_gpu_bwd_routed_moe_tensor`) + `router` → `rms_norm` → `hc_pre` → `hc_mix` (`:230-258`).
3. attention half backward: `hc_post` → `grouped_matmul_t(wo_a)` → inverse RoPE → `sparse_attn` per segment → RoPE → `q_b`/`q_a` → RoPE/`kv` → `attn_norm` → `hc_pre` → `hc_mix`, then the engram gate backward (`:260-314`).
4. Compressor backward for kv-source layers: `pt_compress_bwd` (`:149-172`) `matmul_bf16` (kv+gate), `compress_pool` fwd, inverse RoPE, `rms_norm`, `compress_pool` bwd, `matmul_t` ×2.

Per-layer weight bf16 cache is toggled `ds4_gpu_bwd_wcache(1/0)` around recompute+backward (`:355-358`).

### Batch pack + packcheck (`core_ptrain_pack.c`)

- `PT_PACK_MAX = 8` (`core_ptrain.h:102`). A pack = up to 8 samples concatenated into the batch state `r->st` (rows ≤ `maxlen`), each with its own request slot `rq[i]` whose cache fields are *views* onto the batch buffers (`core_ptrain_pack.c:9-14`).
- `pt_run_list` (`:222-260`): sort by length descending, first-fit into bins (`rows ≤ cap_tok`, `≤ PT_PACK_MAX`); a 1-sample bin runs the single-sample path (`:252`), else `pt_pack_step` (`:171-221`). Token-parallel segments run once for the whole batch; cache segments (window/compressor/indexer/sparse) run per sample via `pt_segs` (`core_ptrain_layer.c:87-95`).
- `pt_pack_warm` (`:266-288`): memory floor probe, runs the densest and the widest packs one forward+backward.
- `pt_pack_check` (`core_ptrain_diag.c:200-273`): `packcheck=1` = structural gate (single sample in a 1-pack must match the single path: archived entry `hc` bitwise, KL equal, grad-norm within `max(1e-2, 3×noise floor)`); `packcheck=2` = STE-spike survey across sample lengths at L8 (finite-difference ratio). Rationale (`:159-162`): cuBLAS switches algorithm with row count, so bitwise cross-path comparison is meaningless; the finite-difference ratio is the only oracle.

### Diag (`core_ptrain_diag.c`)

- `pt_diag` (`:75-156`, `diag=1`): per answer-bit, prints teacher / ③-mounted / deployed (A=0) distributions side by side to `out/diag.txt`, first `PT_DIAG_SHOW=5` names + reference-token probability; summary of "forked" positions (teacher top ≠ deployed top) and whether ③ followed. A=0 zeroing is irreversible, so the run exits after diag.
- `packcheck` lives here too (above).

### Probe (`core_ptrain_probe.c`)

Greedy probes on hold-out questions, 1-way batch via `ds4_v41_gen_run` (byte-identical to the single-request path, `:5-7`): `pt_probes` (`:54-86`) picks ≤`probe_n` eval samples spread across chunks plus ≤`PT_PROBE_HOLD=2` hold samples (`:33-52`), runs deployed (①+②, cached in `base[]` across epochs) and ③-mounted, and writes `out/probe_<tag>.txt`.

### ③ output: `pt_save_amp` (`core_ptrain_bwd.c:173-200`)

For each layer `il` in `[layer_lo, layer_hi]`, writes `amp_L%02u.bin`:

- Header `<i32 D=E><i32 K=rank><i32 1>` (`:184`), `1 = f32` (`:184` comment: "训练产物原值落盘; 要省盘再走 fp4x32").
- Body `A[K][D]` then `B[K][D]`, both f32 row-major, read back from `st->ampA[il]` / `st->ampB[il]` (`:186-188`). `B` memory `[K][D]` row-major = mathematical D×K column-major; apply is `T = x·Bᵀ`, `y += T·A` (`core_v41_amp.c:5-7`).
- Then `base.fnv`: `"%016llx %u\n"` of `ds4_gr_dir_fnv(amp_dir, DS4_N_LAYER)` — FNV-1a 64 over the ② directory's `gr_Lnn.bin` in layer order plus the file count (`core_ptrain_bwd.c:191-198`; algorithm `src/common/ds4_gr_fnv.h`; when no ② is mounted the seed/count is written). The engine re-checks this on load (`core_v41_amp.c:165-185`).
- Init/overlay rule (`pt_run_alloc`, `core_ptrain_bwd.c:28-79`): `A=0`, `B ~ N(0, std²)` with `std = init_std>0 ? init_std : 1/√E` (`:35-39`), so the initial ③ is the identity. With `init=<dir>`, layer files are read first — shape must be `{E, K, 1}` and f32 (`:52-59`) — and then `base.fnv` is verified against the currently mounted ② (`:62-74`): mismatch = stop ("换了底座不许叠加"). The trainer refuses to train on top of an existing ② amp (`:37-38`, "本版不接按秩拼接训练").

`ckpt_eNN` directories hold the same per-epoch, and `pt_gate` picks a round from them. The final `out` is written from `r.st.amp*` after the loop (`core_ptrain.c:336-338`).

### Memory watchdog and budgets

- Trainer (`core_ptrain.c`): `PT_MEM_FLOOR_MB = 3000` (`:27`); after pre-warm, if `MemAvailable < 3000` it prints "撞看门狗 —— 停车" and exits (`:209-222`). `pt_release` calls `ds4_gpu_v41_scratch_release()` between phases (`:22-25`). Per-step memory is logged (`:306`).
- Studio watchdog: `TR_WD_KILL_MB 2500` (`train_internal.h:88`), enforced in `tr_child_wait` — every 5s, two consecutive readings below the line → SIGTERM then SIGKILL, return `-2` (`src/train/train_child.c:63-81`). Same line for the model watchdog (`train_model.c:60-74`).
- Budgets: training `--mem-budget-mb 110000` (`train_job.c:15,87`), model serve `110000` with start floor 100000 and post-load floor 10000 (`train_model.c:16-18`), scoring `--mem-budget-mb 40000 --weight-cache-mb 88000` (`train_gate.c:12-13`).

---

## 3. The studio `src/train/*`

Layout (`train_internal.h:1-22`): `ds4-train` is the always-up main process holding the page port; the model (`ds4-server`) and training (`./ds4 --ptrain`) are its child processes. Two binaries share the API: `ds4-server` exposes the same page/status routes but refuses "spawn children" operations (`serving=1`). Physical constraint: the model is 100+ GB, so training stops the model child first.

### HTTP routes (method, params, response, status)

Main router `train_api.c:197-241`; models router `train_models.c:400-412`. All JSON, `application/json; charset=utf-8` (`train_api.c:200`).

| method | path | params | response / status |
|---|---|---|---|
| GET | `/`, `/index.html`, `/studio`, `/train` | — | `web/studio.html` (200) or 404 (`train_api.c:201`, `serve_page:67-75`) |
| GET | `/logo.jpg`, `/favicon.ico` | — | `web/logo.jpg` image/jpeg (`:202,188-195`) |
| GET | `/api/train/status` | — | `{time,mem_avail_mb,port,mode,busy,downloading,phase,run,proc{scanned,script,ds4,judge,cycle,server},root}` (`api_status:78-93`) |
| GET | `/api/train/runs` | — | `[{name,…summary}]` newest-config-first (`:205`; `tr_json_runs`) |
| GET | `/api/train/run` | `?name=` | full run JSON, else 404 (`:206-210`; `tr_json_run`) |
| GET | `/api/train/data` | — | `{jsonl:[…]}` (`:211`; `tr_json_data`) |
| GET | `/api/train/preview` | `?path=&n=` (n≤50) | `{rows:[…]}` first n lines, ≤4 KB each; 400 bad path, 404 missing (`:212-231`) |
| POST | `/api/train/start` | body `{data,epochs,layers,lr,extra}` | `{ok,error}`; 200, 400 bad params, 409 already running; refused 409 if `serving` (`api_launch:96-110`) |
| POST | `/api/train/stop` | — | `{ok,error}`; 409 if none; 409 if `serving` (`:235`) |
| POST | `/api/train/upload` | `?kind=jsonl\|doc&name=&dir=` + raw body | see below (`:236`) |
| POST | `/api/train/serve` | — | `{ok,error}`; starts model child from `serve_pick.txt`; 409 if `serving` or already up (`:237-240`) |
| GET | `/api/models/list` | `?hub=0` suppresses the HF probe (`api_list:278-305`) | host GGUF + sidecars + loaded trio + fnv map + posttrains + pick + hub + logs |
| POST | `/api/models/download` | body `{endpoint,engram_dir,no_xet}` | spawns `hf_install.sh download`; 200/400/409 (`api_download:312-330`) |
| POST | `/api/models/cancel` | — | kills download; 200/409 (`:407`) |
| POST | `/api/models/load` | body `{gguf,sidecar}` | validates, writes `serve_pick.txt`, restarts child; 200/400/409 (`api_load:333-362`) |
| POST | `/api/models/plugins` | body `{zchain,posttrain}` | hot-swap ②/③ on the live server; requires `serving && tr_live.request` else 409 (`api_plugins:370-398`) |
| * | unknown `/api/models/*` | — | 405 if not POST, else 404 (`:402-411`) |

Proxied verbatim to the model child (`train_proxy.c:15-18`): `/v1/…`, `/monitor`, `/monitor.html`, `/metrics`, `/api/models/plugins`, `/api/models/list`. The proxy streams both directions, one thread per request (`proxy_run:41-66`), 502 if the child is unreachable (`:45-48`); main is a thin blocking HTTP server (`train_main.c:11-42`).

### The upload contract, exactly (`api_upload`, `train_api.c:138-185`)

- Body: raw bytes (not multipart). Empty body → 400 "空文件" (`:141`). `body_len > TR_UPLOAD_MAX` → 400 "超过 64 MB"; `TR_UPLOAD_MAX = 64*1024*1024` and the comment says it equals the server's `max_body` (`train_internal.h:34`, `train_api.c:142`).
- `kind=jsonl`: `name` must match `[A-Za-z0-9._-]`, no `..`, and end `.jsonl` (`:145-148`). Per-line validation (`:149-161`): each non-empty line must parse as a JSON value AND start with `{` AND contain the literal `"messages"` or `"text"` (`:155-158`); the first failing line → 400 `第 N 行不是带 messages 或 text 的 JSON 对象`. Note this is a *looser* check than the trainer: it uses `json_skip_value` + substring test, not the trainer's full field whitelist. Stored bytes = the raw body exactly, at `$tr_datad/<name>` = `gguf-tools/data/posttrain/<name>` (`:162-166`, `train_ctl.c:20`). Response 200: `{"ok":true,"path":"gguf-tools/data/posttrain/<name>","rows":N}` (`:168`). Write failure → 500 (`:166`).
- `kind=doc`: `dir` must match `[A-Za-z0-9._-]` (`:171`); creates `<dir>/chunks/` and writes `<dir>/chunks/<base>.txt` (base = name minus extension, `:174-179`). Then `rebuild_text_jsonl(dir)` (`:113-134`) rewrites `<dir>/<dir>.text.jsonl`, one `{"text": full}` line per `.txt` in `chunks/`, trailing whitespace stripped. Response 200: `{"ok":true,"path":…,"bytes":B,"text_jsonl":"…","rows":R}` (`:180`). Write failure → 500.
- Other `kind` → 400 "kind 只认 jsonl / doc" (`:184`).

UI: `web/studio.html` (single-file, no CDN, `:6`). The data tab drop zone posts to `/api/train/upload` (`studio.html:709-728`): `kind=jsonl&name=` for `.jsonl`, `kind=doc&dir=<docdir>&name=` for `.txt`/`.md`; client-side extension check rejects others. Train button → `/api/train/start` with `{data,epochs,layers,lr,extra}` (`:550`); `serve` (`:548`); records (`:525,572`); model tab (`:848-1013`).

### The job thread (`train_job.c`)

`tr_job_train` (`:132-139`) → `launch` (`:119-130`) → detached `job_thread` (`:96-117`). Phases `TR_JOB_STOP_MODEL → TRAIN → GATE → RESTART` (`train_internal.h:105`).

- Reads ①②③ + extra args from `gguf/serve_pick.txt` (`:101`), stops the model child (`tr_model_stop_sync`, `:105`), enforces the instance lock by scanning the busy binary list (`:108`): `ds4, ds4-bench, ds4-server, ds4quant_run, zlayer, v41_amp_run` — any present aborts.
- `do_train` (`:69-92`): hold file is fixed `gguf-tools/data/posttrain/hold.jsonl` (`:71-72`), required. Output dir `gguf/v41/posttrain/kd-<name>-L<layers>-lr<lr>-e<epochs>-MMDDHHMM` (`:79`). Writes `ptrain.cfg` (`write_cfg:48-59`) with keys `data,out,hold,layers,rank=64,lr,epochs,batch=4,topk=64,maxlen=1024,probe_n=6,probe_tok=96,epoch_tok=0` plus the comma-split `extra` (`k=v` per line). `data=`/`out=`/`hold=` are repo-relative (engine runs at repo root). Child command (`:86-87`): `./ds4 --cuda -m ① [--zchain ②] [--engram-dir …] --mem-budget-mb 110000 --no-dspark --ptrain <cfg>`, log to `<out>/train.out`.
- After `rc==0`, `TR_JOB_GATE` then `tr_gate_pick` (`:90-92`); logs `TRAIN_DONE <out> 选中 <pick> (关联侧车 <zch>; 没挂…)`.
- Always restarts the model child from `serve_pick.txt` (`:111-112`) — "训完只产出，不挂" (`:3-4`).

### The gate (`train_gate.c`)

Constants (`:12-15`): `TR_WT2_IDS="gguf/go-onebit/g7/wt2.ids"`, `TR_WT2_TEACHER="gguf/v41judge/teacher_g7_wt2_n512.bin"`, `TR_ANCHOR_METRICS="gguf-tools/bench/anchor_metrics"`, `TR_GATE_NTOK=512`, `TR_GATE_SMIN_PP=-0.5`, `TR_GATE_KLD_REL=0.03`.

`arm_read` (`:20-27`) extracts `Σmin` and `Mean KLD` from the anchor_metrics text. `score_arm` (`:30-57`): runs `./ds4 … --score-ids <wt2_n512.ids> --score-out <out>/eval/stu_<tag>.bin [--posttrain <ckpt>]`, then `anchor_metrics --ref-raw <teacher> --ids <wt2.ids> --student <stu>`; deletes the 265 MB student table after recording metrics.

`tr_gate_pick` (`:73-121`): requires the teacher anchor and the judge binary (`:79-82`); builds `eval/wt2_n512.ids` from the first 512 lines of `wt2.ids` (`:83-88`); runs the ② arm once (`:92`); per `ckpt_eNN` with `amp_L39.bin` and a matching `epoch N eval_kl` line (`last_run_epochs:57-71`) runs the ②+③ arm, writes `eval/gate_ckpt_eNN.txt` (② then ②+③), and applies the pass rule (`:109`): `dpp = 100·(smin_stu − smin_base) ≥ -0.5` AND `rel = kld_stu/kld_base − 1 ≤ 0.03`. Pick rule (`:111`): among passing rounds, lowest held-out `eval_kl` (`:111`); writes `pick.txt` (`:115-117`). None passing → no pick, returns true (":不进开关").

`train_runs.c` `gate_json` (`:66-84`) reads the same two-arm file and re-derives `pass` with the same `-0.5` / `0.03`.

### `train_models.c` — ③ list, fingerprint matching, plugins

- `emit_posttrains` (`:126-159`): lists `<tr_ftd>/<run>/<pick.txt first line>` directories that also have `base.fnv`; sorts by pick mtime descending; up to `MAX_POSTTRAINS=8`; emits `{path,run,fnv,time}`. A dir without `base.fnv` is dropped (can't be matched to a ②).
- `sidecar_fnv` (`:106-116`): recomputes a ② directory FNV via `ds4_gr_dir_fnv(dir, FNV_SCAN_LAYERS=64)`, cached by directory mtime.
- `read_fnv_file` (`:118-124`): reads ③ `base.fnv` `"%016llx %u"`.
- `emit_hub` (`:196-230`): runs `bash hf_install.sh probe` and parses `repo/shards/gguf/engram/sidecar` lines; scans `hf/` for the n-gram shard hint.
- `tr_pick_write` (`:265-275`): writes `gguf/serve_pick.txt` atomically (tmp+rename): line 1 = ① absolute, line 2 = ② or `none`, then optional `--engram-dir <dir>` and `--posttrain <dir>`.
- `api_plugins` (`:370-398`): validates ② is a sidecar of the loaded GGUF, ③ under `gguf/` and a dir with `base.fnv`, and **matches the ③ base.fnv against the loaded ②'s fingerprint** (`:389-392`) before handing the swap to `tr_live.request`.
- `api_load` (`:333-362`): validates GGUF under `gguf/`, `.gguf`, regular; sidecar same-dir + same-stem-prefix + `is_sidecar_dir`; keeps any existing ③ only when ①② are unchanged (`:359-361`).

### `train_child.c` — process control

`serve_pick.txt` is read by `tr_pick_read` (`train_model.c:31-51`) and written by `tr_pick_write`. Instance lock = `/proc` scan for the busy binary list (`train_model.c:82`, `train_job.c:108`); `tr_proc_kill` matches by program name (argv[0], or `bash/sh` running a script of that name) only — never substring in the whole command line (`train_ctl.c:50-86`, 10-10 real bug). `spawn` (`:29-45`) does `setsid`, redirects logs, closes fd≥3 (so the listen socket never leaks into the child tree, `:6`), `chdir(tr_root)`, `execvp`. `tr_child_wait` (`:60-81`) is the watchdog + cancel loop. `tr_http_local` (`:100-134`) is the one-shot local HTTP client used to wait for `/v1/models` and to smoke `/v1/chat/completions`. Model lifecycle in `train_model.c:78-149`: busy lock → start floor → `serve_pick` → `has_gr` → run with `--trace` → poll `/v1/models` up to 120×5s → post-load floor ≥ 10000 → smoke prompt → watchdog thread; `tr_model_stop_sync` (`:137-149`) SIGTERM then wait for `/proc` disappearance (100+ GB unmap).

---

## 4. The reward `gguf-tools/bench/posttrain_reward.c`

What it computes (header `:1-20`):

- `stock <report.txt> <ohlc.txt> <YYYYMMDD>`: reads the report's trailing JSON `entry_price`, `target_price`, `stop_loss` (`pr_json_num:44-63`, last occurrence wins); next bar = first date after report day (`pr_stock:101-125`). Trade iff `rr = (target−entry)/(entry−stop) + 1e-9 ≥ 3` AND `target>entry>stop` (`:110-112`). Fill iff entry within next-day `[low,high]` (`:115`); exit via `pr_exit:91-99` (both target and stop hit → treat as stop, bin −1; target → +1; stop → −1; else close → 0), minus `PR_COST=0.002` round-trip (`:113,26`). Also emits a T+1 (`*_t1`) column reported but not judged (`:118-124`). rc 0 = scored, 2 = INVALID (no JSON) not 0 (`:105-108`), 3 = no next bar yet (`:109`).
- `market <text.txt> <上涨|下跌>`: first non-listed `大盘上涨/大盘下跌` occurrence = conclusion (`pr_market_pred:148-159`, list-option detector `pr_listing:128-146`); match → +1 else −1; no conclusion → INVALID (`:162-166`).
- Self-test mode: `-DPOSTTRAIN_REWARD_TEST`, `:189-222`, synthetic fixture with hand-computed answers (7 stock cases + 3 market cases).

Who calls it:

- `gguf-tools/scripts/z_nightly_spark.sh:1598-1631` `stage_reward` runs it per `s*.content.txt`; `stage_sample` (`:1516-1593`) is the N-sample collector (stop server → restart with/without ③ → N seeded `/v1/chat/completions` → pull real prompt/gen ids from `--trace`); `stage_solve3` (`:1633-…`) consumes `reward.txt` → within-group mean-subtracted advantages → `v41_amp_run --adv-list`. This is the OLD seventh-version REINFORCE/reward pipeline.
- Built by `make -C gguf-tools posttrain_reward` (`gguf-tools/Makefile:168-171`).

Does the studio use it? No. `grep posttrain_reward src/` finds nothing; `src/train/*` has no reference. The 10-10 trainer dropped the reward loop entirely: `core_ptrain.h:10` lists what was cut ("账本硬目标 / 自采样 REINFORCE / 锚 KL / 锚行合并 / 采样器 / ……") and `core_ptrain_data.c:5,241` reject `reward/mode/split` fields. The studio gate is the wt2 five-metric gate, not the reward. So `posttrain_reward` is legacy alongside `v41_amp_run --adv-list`; the current ③ producer is `./ds4 --ptrain`.

---

## 5. The ③ APPLY at inference (what the port lacks)

### Engine side

Reference: `core_v41_amp.c` (three-file deployment).

- `amp_Lnn.bin` reader `amp_read_layer` (`:194-223`): header `<i32 D><i32 K><i32 ty>`; requires `D == DS4_N_EMBD`, `0 < K ≤ 8192`, `ty ∈ {1=f32, 43=fp4x32}` (`:198-210`); reads `A[K][D]` then `B[K][D]` (`:212-215`); `--zchain-scale β` multiplies the ② A only (`:215-216`); unknown/broken = hard fail, never skip (`:199-210`, header `:187-188`).
- `v41_amp_load` (`:223-270`): first `v41_pt_base_ok(pt_dir, dir)` (`:226`); `v41_rb_load(dir)` router biases, ② only, clears all first (`v41_rb_load:110-121`); `v41_gr_load(dir, pt_dir)` per-layer gains product `(1+r₂)(1+r₃)` (`:130-152`); then per layer `k2 = amp_read_layer(dir…)`, `k3 = amp_read_layer(pt_dir…)`, concatenate rows ② then ③, cap `K ≤ 8192` (`:231-256`); upload to `st->ampA/ampB`, `ampK=K`; allocate `ampT` sized to max K (`:265`). Gains/biases without any amp are legal (`:257-258`); nothing at all in either dir is a refusal (`:260-263`).
- `v41_pt_base_ok` (`:165-185`): ③ `base.fnv` absent → warn and pass; present but mismatched → stop ("停车, 不出假读数"); equals `ds4_gr_dir_fnv(amp_dir, DS4_N_LAYER)`.
- Apply site `core_v41_forward.c:148-155`: after the MoE tail + `round_bf16`, non-draft path stores the MoE input (`tsave->moe_in`), calls `v41_amp_hook` (solve hook, `:152`, can end the forward), then `return v41_amp_apply(st, il)` (`:155`). Draft towers take their own `v41_amp_apply(st, il)` at `:149`. `v41_amp_apply` (`core_v41_amp.c:322-329`): if `ampA[il]` non-NULL, `y += x·(B·A)` via `ds4_gpu_v41_amp_apply_tensor` (or the small-batch draft kernel), then `round_bf16`. `g_ds4_v41_pt_dir` from `--posttrain` (`core_v41_api.c:80`); `v41_state_plugins` per state (`core_v41_state.c:76-87`); flags `cli_opts.c:265-268`, `server_config.c:316-318`.

The engine's own install.sh contract notes ③ is finance-only by the reference's own constraint (port `install.sh:33-34,53`).

### Port side

- The reader/merger is DONE and tested: `crates/ds4-core/src/sidecar.rs` `AmpSidecar::read` (`:252-299`, header `{D,K,ty}`, f32/fp4x32, `AMP_MAX_RANK=8192` `:39`), `check_base_fnv` (`:348-381`), `gr_dir_fnv` (`:306-330`); `crates/ds4-core/src/zchain.rs` `V41Zchain::load` (`:92-137`), `merge_amp` (`:226-278`, ② read first, broken ② stops before ③, rank cap), `merge_gr` (`:179-222`), `k_range` (`:154-157`).
- The APPLY is NOT ported. `v41_run.rs` `v41_mount_zchain` (`:939-995`) mounts rb + gr to the device stores, prints `zchain mount: gr=… rb=… amp=0`, and `zchain_load_for_run` (`:1003-1019`) refuses by name: if `merged.n_amp_layers() > 0` → `"V4.1 zchain carries amp_Lnn.bin; the low-rank amplifier apply is not ported"` (`:1013-1017`). Unit test asserts the refusal names `amp_Lnn.bin` (`v41_run.rs:1223-1227`).
- `--posttrain` flags exist in the port: server `crates/ds4-server/src/bin/ds4-server-rs.rs:261-262`, CLI `crates/ds4-cli/src/lib.rs:227-228`, wired through `generate.rs:3797-4005` into `V41RunOptions.posttrain` (`v41_run.rs:118-121`). So the port can *load* ③ only in the sense of reading/merging and refusing if it carries an amp.
- `crates/ds4-core/examples/ds41_inspect.rs` prints the posttrain evidence: `zchain-merge: gr=… rb=… amp=… k=… base=…` (`:315-321`), the merged layer bits (`:325-337`), first amp `k2/k3/typ` (`:349-359`), and `posttrain: base.fnv ok/ABSENT/MISMATCH` plus `gr_L39 type/factor[0]` (`:369-399`). This is the tool that produced §6.6.4's `base=ok f4a3fd3988135e75/66`.

---

## 6. The port's recorded state

- `docs/deepseek41-port-plan.md` §6.6.4 (`:409-436`): records the ②/③ merge mirror and its model-free gates (`tests/zchain.rs`, 8) and live evidence on the Spark. The amp half: "no directory on the Spark carries `amp_Lnn.bin`, so the rank concatenation and β are covered model-free only; their live gate arrives with P4's device upload" (`:430-435`).
- §6.14 (`:986-1050`): "Recorded limits. The amp half of the sidecar is not ported: the real directory carries no `amp_Lnn.bin` (39 gr + 27 rb + manifest.txt, amp=0) and no Spark directory does, so there is nothing to gate. Its apply point is the engine's `v41_layer` tail (`core_v41_forward.c:148-155` … the towers take their own ampA at `:149`; the solve hook `:152` …) and its loader rides the same amp-directory wiring … recorded for that future unit together with the base.fnv fingerprint gate (`v41_pt_base_ok`, `core_v41_amp.c:163` …; missing warns and allows, mismatch stops)." (`:1038-1048`).
- The open item "no Spark directory carries amp_Lnn.bin" recurs at §6.14 (`:1040`), and §6.18's unit-Z record (`:1503-1504`): "a dir carrying `amp_Lnn.bin` is refused by name (no Spark dir carries one; the §6.14 limit stands)".
- §6.21 (`:1576-1619`) is the qwen38-derived serving-candidate section (P2 engram overlap, Q-1 MTP compact draft vocab, Q-2 MTP prefix cache). It does not touch post-train.
- Gap table: G6 `:125` (sidecars incl. `amp_Lnn.bin`, `base.fnv`, `--zchain/--posttrain`), G8 `:127` (serving surface), both marked "(a)" = C reference not yet in the Rust tree.

---

## 7. Gate assets

Where the scripts expect them vs where they come from:

- `gguf/go-onebit/g7/wt2.ids` — the wt2 (WikiText-2-for-Go) judge ids. Referenced by `train_gate.c:14` and `z_nightly_spark.sh:77`. Produced by `gguf-tools/scripts/regen_ids.sh:34` (tokenizes `gguf-tools/data/corpus`); the corpus head is `gguf-tools/data/corpus/wikitext2_test_head.txt` and there is an older FP anchor `gguf/go-onebit/r30/anchor_wt2_s2653.bin` used by the campaign scripts. The g7 tree is NOT in the repo (no `gguf/` dir at the root); it ships in the release pack (`release_pack.sh:23-24`).
- `gguf/v41judge/teacher_g7_wt2_n512.bin` — the FP teacher top-K anchor (265 MB) for the first 512 wt2 ids. Referenced `train_gate.c:15`, `TR_WT2_TEACHER`. `release_pack.sh:13` explains it "要 HF 出厂权重才能出, 拿包的人出不了": produced on a machine with the HF factory weights, via `v41_judge.sh` / `v41_teacher.py` (`v41_teacher.py:21` "ids 文件: 一行一个 token id(与本仓 wt2.ids 同格式)"). Not in the repo; shipped in the release pack.
- `gguf-tools/data/posttrain/hold.jsonl` — the fixed hold-out material (generic questions + deployed-state answers). Referenced `train_job.c:70-72` and `z_nightly_spark.sh:1994,2095`. It is shipped in the release pack (`release_pack.sh:24`); it is NOT present in the `gguf-tools/data/` tree, and no in-tree producer was found. Treat "produced by which step" as unverified.
- `gguf-tools/bench/anchor_metrics` — the five-metric judge (Σmin / Mean KLD / median KLD / Same top token / PPL). Referenced `train_gate.c:16` `TR_ANCHOR_METRICS`; built by `make -C gguf-tools anchor_metrics` (`gguf-tools/Makefile:173-176`); shipped (`release_pack.sh:20-21`). Source `gguf-tools/bench/anchor_metrics.c` + the two `.inc.c` tables.

Release packing (`gguf-tools/scripts/release_pack.sh:20-24`): `FILES=(ds4-server ds4-train ds4 gguf-tools/bench/anchor_metrics web/*.html web/*.jpg LICENSE … gguf-tools/scripts/hf_install.sh gguf-tools/data/posttrain/hold.jsonl gguf/go-onebit/g7/wt2.ids gguf/v41judge/teacher_g7_wt2_n512.bin)`. So the training-critical assets ship with the release; the large model/engram/sidecar come from the model page download.

---

## 8. Feasibility matrix

Dependencies key: **model stack** = V4.1 forward; **GPU backward** = the `ds4_gpu_bwd_*` family + the CUDA backward `.inc.cu` files in §2; **assets** = §7.

| piece | depends on | portable now? | port touch points |
|---|---|---|---|
| `ds4_posttrain` module (math) | nothing (pure C, no engine coupling) | YES — pure closed-form, no GPU, no model | new module (e.g. `crates/ds4-core/src/posttrain.rs` or a small crate); no kernels |
| Upload API + studio routing (JSONL validation, store, list/preview) | nothing model-side; SSH/HTTP layer only | YES — pure host code | `crates/ds4-server` (axum route + body limit 64 MB), a data-dir policy module; no `ds4-core`/kernels |
| Training core (`core_ptrain*.c`) | model stack (full V4.1 forward, state plugin attach) AND GPU backward (all 24 entries in §2, `moe_capture`, `wcache`) AND assets (hold.jsonl, wt2 anchor for the gate) | NO — must wait for the inference port to close; the port explicitly has no backward pass (`cuda/ds41_vq_prefill_mma.cuh:595-597`) | would need the whole backward kernel family under `cuda/` (new `ds41_bwd_*.cuh`) + a Rust trainer driver in a new crate; the 3 GB floor risk is the same as the engine |
| ③ APPLY (amp concat + `y += x·(B·A)` + bf16 round) | reader/merge (already ported, §5); device apply = one GEMM pair on `st->y`/`st->xn` + bf16 round; needs the ⑤-⑦ stores and the V4.1 forward tail | PARTIAL — the merge and base.fnv gate are done; the apply needs the P4-7 stores + forward-tail hook and currently refuses by name | `crates/ds4-core/src/zchain.rs` (have), `v41_run.rs` (`v41_mount_zchain` `:939-995`, refusal `:1013-1017`), device side the bridge + a `cuda/ds41_*.cuh` amp-apply kernel (mirror `ds4_gpu_v41_amp_apply_tensor`) |
| Studio job/gate/models (orchestration, wt2 gate, `serve_pick`) | the training core (for job/gate) + assets; models/download/plugins need the model child lifecycle | PARTIAL — job/gate must wait for the trainer; models/download/plugins can be built now against the port server | new server crate module; `anchor_metrics` would need a Rust port or the C binary shipped; `serve_pick.txt` + `/proc` scan |
| Reward (`posttrain_reward`) | nothing model-side (pure C, text+OHLC in) | YES, but LEGACY — the studio path does not use it and the 10-10 trainer dropped the reward loop (`core_ptrain.h:10`, `core_ptrain_data.c:241`) | optional; a `ds4-core`/tool function if ever needed |

Bottom line per the owner's three questions:

- **(a) standalone module `ds4_posttrain`**: portable now (pure math) — but note it has ZERO callers in the engine and is slated for deletion (`docs/clear.md:113`); porting it is preserving a validated formula, not wiring a live path.
- **(b) tests that verify uploading training data as JSON**: portable now (host-only); the engine oracle is `ds4-train`'s `api_upload`, testable without a model.
- **(c) post-train LOADED and TRAINED**: LOAD is half-done in the port (reader/merge/gate yes; apply refused); TRAINED is blocked on the GPU backward family, i.e. on the inference port closing (P4/P5). The reward path is legacy and not part of the current product flow.

---

## 9. Test design for "upload training data as JSON"

Characterization corpus (each case = one POST to `/api/train/upload`, compared both to the engine's `ds4-train` and to the trainer `./ds4 --ptrain` acceptance where applicable):

| # | case | expected studio result (`train_api.c`) | trainer result (`core_ptrain_data.c`) |
|---|---|---|---|
| 1 | good text-only: `{"text":"…"}\n` | 200 `{ok:true,path,rows:1}`; bytes = body | accepted; BOS+text+EOS, one-hot targets |
| 2 | messages-only user+assistant | 200 `{ok:true,…}` | accepted; hold-out by FNV(question) |
| 3 | system+user+assistant | 200 | accepted; system rendered on both sides (`:168,173`) |
| 4 | extra field `{"messages":[…],"reward":1}` | 200 (loose check: contains `"messages"`) | error "只认 messages / context / text 三个字段" (`:241`) |
| 5 | one bad line among good | 400 `第 N 行不是带 messages 或 text 的 JSON 对象`; nothing written | run stops at that line (`:274-278`) |
| 6 | wrong extension (`data.txt` as jsonl, or `.json`) | 400 "料文件要以 .jsonl 结尾" (`:147`) | n/a |
| 7 | oversize (>64 MB) | 400 "超过 64 MB" (`:142`) | n/a (trainer has no cap) |
| 8 | empty | 400 "空文件" (`:141`) | n/a |
| 9 | non-object line (`[1]`, `"x"`, `123`) | 400 (line does not start `{`) | error "不是 JSON 对象" (`:196`) |
| 10 | CRLF line endings | 200; leading `\r` stripped per line (`:152`); raw body stored byte-exact | `pt_load_file` skips leading ` \t\r` (`:274`) |
| 11 | trailing newline / no trailing newline | 200 both; last line handled (`:160` `p = nl ? nl+1 : end`) | handled (`:270-283`) |
| 12 | `context` + messages | 200 | accepted; teacher prefix = context |
| 13 | `text` + messages/context mixed | 200 (loose) | error "text 行不能再带 messages / context" (`:251`) |
| 14 | hold file with context/text | n/a | error (`:277`) |

Oracle and acceptance:

- Oracle A (upload API): the engine's `ds4-train` on the Spark (release pack), same raw body. Acceptance = same status code, same response JSON shape, and byte-identical stored file at `gguf-tools/data/posttrain/<name>` (or `<dir>/chunks/<base>.txt` + regenerated `<dir>.text.jsonl`).
- Oracle B (trainer): where the training core exists, feed the same JSONL to `./ds4 --ptrain` and assert the accept/reject verdict matches; specifically cases 4, 5, 9, 13, 14 must line up (the studio's loose check differs from the trainer on case 4 and 13 — a port must preserve the engine's *exact* split of responsibilities, not unify them).
- The doc-tab path: assert `rebuild_text_jsonl` semantics (one `{"text":…}` line per `chunks/*.txt`, trailing whitespace stripped, whole-dir rewrite) and the response keys `bytes`, `text_jsonl`, `rows`.

---

## 10. Risks / unknowns

- Path citation is stale in the reference itself: `ds4_posttrain.h:2` cites `gguf-tools/go-onebit/ALGORITHM.md`; the file is `gguf-tools/docs/ALGORITHM.md` and §6.1 = section 6 item 1 (`:57`). Verified by grep (no `go-onebit/ALGORITHM.md` anywhere).
- `hold.jsonl` producer UNVERIFIED: it is shipped by `release_pack.sh:24` but has no in-tree generator; the external step that writes it was not confirmed.
- The port's `zchain.rs:3` cites `commit 3946dbc` (the OLD pin). For `core_v41_amp.c` behavior that pin matches (the amp loader is stable across the two trees); for the *trainer* it is stale (see §0).
- The port has NO backward pass (`cuda/ds41_vq_prefill_mma.cuh:595-597`), so the training core cannot be ported before the inference port closes; this is a hard dependency, not a scheduling preference.
- `anchor_metrics` and the wt2 anchor/`hold.jsonl` are external release assets (not in the repo). A Rust port of the gate needs either the shipped C `anchor_metrics` binary (as the engine does) or a from-scratch Rust port of its five metrics; the engine's gate shells out to the C binary (`train_gate.c:30-53`), so the simplest faithful path is to keep shelling out.
- The studio upload validation is intentionally looser than the trainer's (`train_api.c:155-158` substring test vs `core_ptrain_data.c:209-241` full parse). A port must reproduce both, not one.
- `posttrain_reward` is legacy: it belongs to the pre-10-10 seventh-version REINFORCE pipeline (`z_nightly_spark.sh:1598-1631` → `stage_solve3` → `v41_amp_run --adv-list`); it is absent from the studio and from the 10-10 trainer. Porting it has no current consumer.
- Anything above that touches GPU memory floors (3000 MB trainer, 2500 MB watchdog) is the engine's numbers on the DGX Spark; the port must re-measure on its own host before adopting them (rule 20: measure, do not inherit an unverified budget).
