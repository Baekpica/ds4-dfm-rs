# DeepSeek V4.1 port — function inventory from the C engine

[Port plan](deepseek41-port-plan.md) | [Repository README](../README.md)

Companion evidence file for `docs/deepseek41-port-plan.md`. Every line cites the
C engine at `/data/YoungAi` (commit 3946dbc), which is the port's source of
truth. Collected by read-only inventory passes on 2026-10-08; a citation is a
pointer to re-verify, not a substitute for reading the file.

Legend: HOST = plain C host code, DEVICE = CUDA kernel, LAUNCHER = host-side
wrapper that launches kernels. "public" = declared in `ds4_v41_api.h` or
`ds4_gpu_v41.h`.

## 1. Metadata, bind and shape (host)

`src/core/core_validate_v41.c`

- `:14 ds4_engine_v41_set_engram_dir` — stores `--engram-dir`. public
- `:16 v41_arr_i32` / `:28 v41_arr_u64` / `:40 v41_req_u64` — required GGUF
  array/scalar readers, hard-exit when absent. static
- `:46 v41_load_metadata` — fills `g_ds4_v41` from the `deepseek4.*` keys,
  builds the per-layer source tables and the engram tables. called from
  `core_validate.c:216`

Metadata keys read there (required unless noted): `deepseek4.context_length`;
`attention.kv_source_layers`; `attention.index_source_layers`;
`attention.candidate.source_layer|topk_blocks|block_size`; `engram.layer_ids`,
`engram.num_embeddings`, `engram.<i>.table_path|weight_offset|scale_offset`,
`engram.max_ngram_size|head_count|head_dim|vocab_size|compressed_vocab_size|pad_id_compressed`;
`mtp.tower_count|expert_count` and, read as a group only when all are present,
`mtp.block_size|expert_used_count|noise_token_id|markov_rank|target_layers`.
Cross-checks: every compressing layer needs a kv source and an index source
before it; a kv source must also be an index source.

`src/core/core_bind_v41.c`

- `:11 need` / `:13 required_tensor_name` / `:19 expect` / `:39 expect_skel` —
  required-tensor and shape/type assertions. static
- `:41 weights_bind_v41` — binds every V4.1 tensor, then validates shapes and
  types. called from `core_bind.c`

Bound names (the artifact's contract):

- Global: `token_embd.weight` (FP4X32 or Q4_K), `output_norm.weight` (F32),
  `output.weight`, `engram.token_map` (I32), `engram.multipliers` (I64),
  `engram.primes` (I64), `engram.offsets` (I64), and the shared tower heads
  `mtp.main_proj/main_norm/markov_embd/markov_head/confidence/out_norm.weight`.
- Per layer: `hc_attn_fn|scale|base`, `attn_norm`, `attn_q_a`, `attn_q_a_norm`,
  `attn_q_b`, `attn_kv`, `attn_kv_a_norm`, `attn_sinks`, `attn_output_a`,
  `attn_output_b`; when the layer is a kv source also `attn_compressor_kv`,
  `attn_compressor_norm`, `attn_compressor_gate` (ratio > 1), `indexer.wk`,
  `indexer.k_norm`; when it is an index source also `indexer.attn_q_b`,
  `indexer.proj`; `hc_ffn_fn|scale|base`, `ffn_norm`, `ffn_gate_inp`,
  `exp_probs_b.bias`, **`ffn_exps_vq.blob` (required; there are no
  gate/up/down expert tensors)**, `ffn_gate_shexp`, `ffn_up_shexp`,
  `ffn_down_shexp`; engram layers add `engram_wkv` (FP8_32X32), `engram_q`,
  `engram_k` (F32).
- Per tower `mtp.T.*`: the same attention/hc/ffn name set, plus either
  `mtp.T.ffn_exps_vq.blob` or per-expert
  `mtp.T.ffn_exp.E.{gate,up,down}.weight`.
- No `output_hc_*` tensors: the head reuses the last `ffn_pre` as `hc_pre`.

`src/core/core_shape_select.c`

- `:81 DS4_SHAPE_V41_FLASH` — 40 layers, 5120 embd, 384 experts top-6,
  q_lora 1280, ff 2304, indexer 32 heads, rms eps 1e-20, route scale 1.5.
- `:166 ds4_select_shape_from_metadata` — picks FLASH / PRO / V41_FLASH
  (V4.1 branch at `:211`). Only the V4.1 FLASH table exists; there is no V41
  PRO.

## 2. V4.1 forward and state (host)

`src/core/core_v41_api.c` — public control surface: `:7 ds4_engine_is_v41`,
`:10 ds4_engine_set_decode_sampling`, `:18 set_decoder_full`, `:22
set_vq_group`, `:23 set_prof`, `:38 set_dspark`, `:39 dspark`,
`:42 last_spec_stats`, `:55 set_emit_trace`, `:62 set_block`, `:64
set_verify_k`, `:66 set_draft_amp`, `:70 set_draft_amp_scale`, `:75
set_amp_dir`, `:80 set_posttrain_dir`, `:83 set_amp_scale`, `:86
set_moe_hook`, `:91 set_moe_hook_layer`, `:96 set_progress`, `:97 ctx`,
`:103 set_chunk`, and `:426 ds4_engine_v41_generate_argmax` (the full generate:
chunked prefill with CED, per-token decode with the CUDA graph, sampling,
DSpark speculation; `:424` is the no-GPU stub).

`src/core/core_v41_forward.c`

- `:14 v41_hc_mixes` — split `hc_mix` into pre/post/comb via sinkhorn. static
- `:23 v41_hc_half` — half-layer entry: mix GEMM + fused split/hc_pre/norm.
- `:75 v41_moe` — router + routed VQ MoE + shared fp4 expert + tail bf16,
  applies the amplifier and tsave.
- `:182 v41_layer` — one block: hc mixes, attention, hc_post, ffn half, MoE.
- `:241 v41_spec_snapshot` — back up the ring cells a verify batch overwrites.
- `:267 v41_spec_rollback` — restore ring cells, revert comp counts and cpre
  rows, fix mainh.
- `:300 v41_forward_body` — embed, expand_hc, 40 layers, exit head, CED stop.
- `:362 v41_prefill_chunk` / `:372 v41_forward` — prefill chunking and append.
- Diagnostics: `:45 v41_moe_uniq_probe`, `:159 v41_nan_scan`, `:171
  v41_dump_rows`.

`src/core/core_v41_attn.c` — `:11 v41_theta`, `:12 v41_osl`, `:15 v41_rope`
(static); `:30 v41_compress_source_graph`, `:61 v41_compress_source`, `:126
v41_index_source` (static); `:196 v41_attention_kv_only` (CED boundary),
`:206 v41_tproj` (matmul dispatch by on-disk type FP8/Q4_K/FP4X32), `:216
v41_embed`, `:222 v41_tproj_grouped`, `:237 v41_draft_attention`, `:263
v41_draft_push_main`, `:286 v41_attn_in`, `:312 v41_attn_cache`, `:351
v41_attn_out`, `:368 v41_attention`.

`src/core/core_v41_state.c` — `:12 v41_alloc`, `:90 v41_state_alloc`,
`:139 v41_batch_rows_alloc`, `:154 v41_state_shrink`, `:198
v41_index_scratch_prepare`, `:232 v41_state_free` (allocators static).

`src/core/core_v41_sample.c` — `:17 v41_sample_pick` (assemble `want[]` from
device slots: accept uses the draft, reject the residual), `:24
v41_device_next`, `:42 v41_next_token` (device path or host sampling with
penalties).

`src/core/core_v41_multi.c` / `core_v41_req.c` — `:19 v41_batch_alloc`, `:48
v41_attach` (8 row views), `:67 v41_detach`, `:80 v41_multi_body`, `:152
v41_multi_advance`, `:165 v41_multi_step`, `:195 v41_multi_pick`, `:202
v41_multi_pick_rows`; and the public request surface `ds4_v41_batch_open/close`,
`ds4_v41_req_open/close`, `:94 ds4_v41_req_prefill_step`, `:110
ds4_v41_req_take`, `:117 ds4_v41_multi_step`, `:139 ds4_v41_multi_round`
(draft, verify, accept, rollback, outq), `:235 ds4_v41_req_prefill_bytes`,
`:249 ds4_v41_req_resident_bytes` (the server's admission numbers).

`src/core/core_v41_mgraph.c` — the batched whole-step CUDA graph: `:41 mg_alloc`,
`:54 mg_bucket_cap`, `:60 mg_find`, `:74 v41_multi_graph_ready`, `:87
mg_capture`, `:154 v41_multi_graph_round`, `:31 v41_mgraph_free`.

## 3. Serving surface (host)

`src/server/server_generate_v41.c` — `:23 v41_progress_cb`, `:85 v41_emit`,
`:317 v41_gen_begin`, `:375 v41_gen_end`, `:406 generate_job_v41`;
statics `:18 v41_kind`, `:37 v41_stream_begin`, `:73 v41_prefill_done`, `:211
v41_finish`.

`src/server/server_sched_v41.c` — `:22 mem_available_mb`, `:31 lane_finish`,
`:40 lane_admit`, `:52 sched_prefill`, `:86 v41_sched_run` (admit, time-share
prefill and decode, `ds4_v41_multi_round`, emit, teardown). Admission is
memory-based: a prefill needs `MemAvailable >= 2 x
ds4_v41_req_prefill_bytes(prompt)`; it waits while decode lanes exist, and
fails the request when it still does not fit. Prefill is one chunk at a time on
the oldest lane; decode accumulates until `t_dec >= t_chunk`. Cross-request
speculation only for n <= 2 lanes.

`src/server/server_batch.c:196 worker_main` — V4.1 dispatch at `:198` when
`batch_max >= 2`. `src/server/server_main.c:29` — V4.1 context from metadata,
disk KV dropped, session creation skipped. `src/server/server_config.c` —
`--posttrain` `:315`, `--engram-dir` `:319`, `--no-dspark` `:329`, `--ctx`
rejected `:353`, `--batch` `:343`.

Sampling decision (`core_v41_api.c:139-153`, `core_v41_sample.c`): temperature
<= 0 and no penalty -> device argmax; temperature > 0 and no penalty -> device
sampling kernel with rejection sampling (speculation continues); any penalty ->
full logits row read to host, penalties applied, host sampler (speculation
refused for that request; explicit `--dspark` with penalties is a hard error).

Flags and defaults: `--zchain <dir>`, `--zchain-scale` (1.0), `--posttrain`,
`--engram-dir`, `--draft-amp`, `--draft-amp-scale` (1.0), `--v41-no-engram`,
`--v41-chunk` (0 = 2048), `--decoder-full` (CED on by default), `--no-dspark`
(speculation on by default), `--dspark` (mode 2), `--dspark-block`,
`--dspark-verify`, `--no-graph` (graphs on), `--no-vq-group` (grouped VQ on),
`--no-lanes` (lanes on), `--idx-mma` (off), `--v41-prof`, `--emit-trace`,
`--score-*`, `--ptrain`, `--draft-train`, `--multi-probe`, `--ctx` rejected.

## 4. Device surface (stays native)

`cuda_v41_1.inc.cu` — dense fp4x32 GEMM and grouped matmul, embed, bf16
round/cast, rms norm, add, expand_hc; scratch helpers
`ds4_gpu_v41_scratch_generation|bytes|release`.

`cuda_v41_2.inc.cu` — RoPE with the YaRN ramp, compressor pool
(`v41_compress_pool_kernel`, `v41_compress_step_n_kernel` with the pre-shift
snapshot), main sparse attention `v41_sparse_attn_kernel` and its dispatcher
(full / split-K / mma decode).

`cuda_v41_3.inc.cu` — `v41_router_kernel` (sqrtsoftplus gate + bias topk),
`ds4_gpu_v41_set_rb_override`, `v41_swiglu_kernel` (clamp), `v41_engram_gate_kernel`,
`ds4_gpu_v41_routed_moe_tensor` (the VQ-blob routed MoE).

`cuda_v41_4.inc.cu` — fp4x32 GEMV (NT-specialized, planar/bf16 variants),
engram row decode, `ds4_gpu_v41_amp_apply_tensor` (y += x·(B·A)), scale_round,
argmax.

`cuda_v41_attn_split.inc.cu` / `cuda_v41_attn_mma_decode.inc.cu` — decode
split-K segments and merge with the sink in the denominator; tensor-core
segment variant.

`cuda_v41_hc.inc.cu` — row rsqrt, scale_rows, fused hc_mix GEMM, sinkhorn
split (20 iterations), hc_pre, fused split+pre+norm, hc_post.

`cuda_v41_indexer.inc.cu` / `cuda_v41_indexer_mma.inc.cu` — indexer score
(scalar and s8 tensor-core), candidate-block selection, radix topk, and the
int-mantissa q prep for the mma variant.

`cuda_v41_q4k.inc.cu` — q4_K decode, GEMV (staged and pipelined), bf16
staging + cuBLAS GEMM for prefill, embed, output-head column norm, and the bf16
weight cache.

`cuda_v41_fp4_planar.inc.cu` — planar fp4 copy (4 GiB cache).

`cuda_v41_nvfp4.inc.cu` — fp4x32 -> NVFP4 conversion, activation -> NVFP4, and
the cuBLASLt prefill GEMM.

`cuda_v41_gemv_highprec.inc.cu` — f32 / bf16 / fp8-block GEMV, grouped
variants, head column norm.

`cuda_v41_sample.inc.cu` — splitmix64 uniform, radix select, sample gate, and
the per-row sampling kernel with speculative rejection sampling.

Files this inventory adds beyond the plan's first list, and which a port must
not miss: `cuda_kv_pack.inc.cu` (`v41_kv_pack_kernel`, `v41_act_quant_kernel`,
`ds4_gpu_v41_ckv_pack_tensor`, `_idxk_pack_tensor`, `_act_quant_fp8_tensor`,
`_act_quant_fp4_tensor`), `cuda_kv_ring.inc.cu` (`v41_win_commit_kernel`,
`v41_win_ring_snap_kernel` and their launchers — the window commit and
rollback), and `cuda_v41_draft.inc.cu` (tower MoE, hc_mean, ring_rows,
row_gather, row_add, mkcache lookup/add).

## 5. Default path versus diagnostics

Default: context from metadata only; prefill chunked at 2048; CED on; CUDA
graphs on; DSpark speculation on (greedy requests are bit-identical, sampled
requests fall back to pure decode); grouped VQ on; lanes on; concurrent
scheduler when `batch_max >= 2`.

Diagnostics: `--v41-prof`, `--emit-trace`, `--v41-chunk`, `--decoder-full`,
`--no-graph`, `--no-dspark`/`--dspark`/`--dspark-block`/`--dspark-verify`,
`--dspark-capture`, `--v41-no-engram`, `--no-vq-group`, `--no-lanes`,
`--idx-mma`, `--zchain-scale`, `--draft-amp-scale`, the `--score-*` family,
`--ptrain`, `--draft-train`, `--multi-probe`.

## 6. VQ subsystem (the expert blob)

Host side (`vq_fmt.h`, all static inline): `:22 ds4vq_f16`, `:30
ds4vq_blob_nexp`, `:34 ds4vq_blob_ver`, `:41 ds4vq_blob_ok` (magic, version
whitelist 2..=3, `nexp` in 1..=4096, size), `:50 ds4vq_slot` (payload offset
from the table at blob+16), `:57 ds4vq_dequant_f32`, `:92 ds4vq_dequant_f16`.

Host side (`src/core/core_sidecar.c`): `:33 vq_dir_load` (`--vq-dir`, per-layer
`dql_vq_L%02u.bin` mmap), `:60 vq_model_load` (binds `blk.%u.ffn_exps_vq.blob`,
type 42, straight into the model mmap), `:89 residual_load` (`--residual`,
also binds the blob when present), `:233 residual_set_for` (returns the layer's
blob pointer, `vq` flag and byte count). `core_bind_v41.c:87` requires the blob
per layer and leaves `ffn_gate_exps`/`ffn_up_exps` NULL.

Device side, the layout consumer: `cuda_vq_row.inc.cu:21 v41_vq_open<V3>`
(parses v2/v3, requires dim 8), `:74 v41_e4m3x2_to_half2` (hardware
`cvt.rn.f16x2.e4m3x2`), `:84 v41_vq_cw<FP8>`, `:103 v41_vq_dot8_cw`, `:111
v41_vq_dot8<FP8>`, `:117 v41_vq_xpack_kernel` (activation to packed bf16), the
shfl bitstream readers `:138 v41_vq_sel3`, `:141 v41_vq_blk`, `:143
v41_vq_mbit`, `:145 v41_vq_row_ptr`, `:149 v41_vq_ext_ptr`, `:155
v41_vq_blk_load`, `:177 v41_vq_row_first_blk`, `:188 v41_vq_blk_rounds`, `:213
v41_vq_row_dot`.

Device side, decode: `cuda_vq_decode.inc.cu` (`v41_vq_cb_to_shared`,
`v41_vq_swiglu`, `v41_vq_gateup_kernel`, `v41_vq_down_kernel`,
`v41_vq_reduce_kernel`, `v41_vq_order_kernel`, the dispatcher
`v41_vq_fused_moe_n`, `v41_vq_tail_kernel`); `cuda_vq_group.inc.cu` (the
M-token group kernels); `cuda_vq_persist.inc.cu` (the persistent n=1 and
n>=2 verify-batch kernels plus `v41_vq_stream`); `cuda_vq_decode_launch.inc.cu`
(`:12 v41_vq_fused_moe`, which picks the instance by version and bit width:
v3 12-bit, v3 13-bit with plane, v2 12-bit, v2 11-bit).

Device side, prefill: `cuda_vq_prefill.inc.cu` (scheduler, per-layer head
table `vqp_hdr_build`, `cuda_vq_moe_prefill_gemm`, `vqp_reduce_kernel`,
`ds4_gpu_v41_set_gr_override`, `ds4_gpu_v41_vq_capture_expert_out`),
`cuda_vq_prefill_fused.inc.cu` (v2 fused), `cuda_vq_prefill_mma.inc.cu` (v3
bf16 tensor-core, `vqm_kernel`), `cuda_vq_reg_mma.inc.cu` (`vqs_kernel`,
register direct-decode), `cuda_vq_align.inc.cu` (`cuda_vq_blob_populate_aligned`,
128 B bitstream alignment with a slot-table rewrite).

Retired or probe-only, which the port must not resurrect: the whole
`cuda_vq_prefill_nvfp4.inc.cu` path has no call site; `vq2_probe_layer` and
`vq_load_cb_sh` have none either; the device globals `g_vq_exp_mode`,
`g_vq_cyc`, `g_vq_cyc_on` have no setter. `cuda_vq_probe.inc.cu` runs only
under `--v41-prof`.

Include order in the single CUDA TU is a hard contract (`ds4_cuda.cu:65-98`),
and `v41_vq_fused_moe` has a duplicate prototype at `cuda_v41_1.inc.cu:47-50`
that must stay in sync with its definition. Two entry families must not be
conflated: `cuda_vq_moe_forward` (v2-only, the V4 merged path) and
`v41_vq_fused_moe` (version-dispatched, the V4.1 path); both funnel large
batches into `cuda_vq_moe_prefill_gemm`.

No environment variable reaches the VQ path: a grep of the whole `src` tree
finds `getenv` only for HOME, PATH and LINENOISE_ASSUME_TTY. The switches are
`--v41-prof` (probes, and it disables decode graphs), `--no-vq-group` (pins
back to the per-pair kernels), `--vq-dir`, `--residual`, `--zchain` (gr
overrides). `DS4_V41_GEMV_MAX_TOK = 8` (`ds4_gpu_v41.h:265`) is the
decode-versus-prefill threshold.

## 7. Engram and sidecars

Engram runtime, `src/core/core_v41_engram.c`: `:15 v41_tensor_host`, `:21
v41_edio_pread` (4 KB-aligned O_DIRECT read through a 2x4096 bounce),
`:30 v41_engram_open_shard` (O_DIRECT, else buffered with POSIX_FADV_RANDOM
and a warning that output may become non-reproducible), `:65 v41_engram_hash`,
`:128/:236 v41_ejob_submit_ring`, `:128 v41_ejob_io_thread`, `:151
v41_ejob_io_idle`, `:159 v41_eworker_run`, `:186 v41_ejob_free`, `:206
v41_ejob_create`, `:287 v41_ejob_submit_round`, `:261 v41_engram_graph_arm`,
`:268/:346 v41_ejob_wait`, `:271 v41_engram_graph_serve`, `:281
v41_engram_graph_err`, `:311 v41_engram_prefetch`, `:381 v41_engram_close`,
`:398 v41_fnv1a`, `:403 v41_engram_fingerprint`, `:419 v41_engram_rows`,
`:458 v41_engram_apply`, `:480 v41_engram`.

`core_v41_ering.c`: `:39 v41_ering_open`, `:70 close`, `:79 cap`, `:88 push`,
`:113 submit`, `:121 wait` (raw io_uring syscalls, per-slot bounce, free-slot
stack). `core_v41_epool.c`: `:35 v41_epool_worker`, `:54 init`, `:69
threads`, `:73 submit`, `:88 wait` — a process-global pool of
`V41_EGATHER_THREADS = 48`.

Row layout: 256 B e4m3 weight plus 8 B ue8m0 scale = 264 B per row; weight at
`weight_offset + r*256`, scale at `scale_offset + r*8`; dequant is `e4m3 *
e8m0(row[256 + d/32])`, rounded bf16 (`cuda_v41_4.inc.cu:302`).

The EGRC constants file (`gguf-tools/scripts/v41_engram_consts.py`): magic
'EGRC', version, vocab, compressed vocab, layer count, max n-gram, head count,
pad id, then `token_map` i32, `multipliers` i64, `primes` i64, `offsets` i64.
The converter embeds them as GGUF tensors `engram.token_map|multipliers|primes|offsets`
and writes the per-layer `deepseek4.engram.<i>.table_path|weight_offset|scale_offset`
as absolute byte offsets into the official shards (model-00047/48-of-00048,
~203 GB for the two tables).

Rolling n-gram hash (`core_v41_engram.c:65-89`): for k in 0..G take
`cid = token_map[hist[p-k]]` (or the pad id), `prod[k] = cid * multipliers[ei*G+k]`;
`rolling = xor of prod[1..G]`; for each head h, `r = rolling % primes[...]`,
`rows[...] = r + offsets[...]`. History comes from the whole-token ring, so a
4-gram crossing a chunk boundary resolves correctly. The official reference is
`NgramHashState.forward` in HF `inference/engram.py`, reproduced bit-exactly by
the consts script.

Read path: units are (position, column, half); one unit is one pread; a decode
step is 2 engram layers x 24 rows = 48 rows, 12,672 logical bytes, about 96
aligned 4 KB reads. The next layer's round is submitted as soon as the current
one is received; io_uring depth is 512 (a 4096 experiment was rejected and
recorded). O_DIRECT is what keeps the 203 GB table from flushing the 103 GiB
model mapping, which is the documented temperature-0 nondeterminism failure.

Sidecars, `src/core/core_v41_amp.c`: `:20 amp_read_mat`, `:40
v41_gr_accum_layer` (`gr_Lnn.bin`: header `<i32 n_expert><i32 D><i32 type>`,
type 1 f32, 2 f16, 43 fp4x32 storing `s-1`), `:88 v41_rb_load_layer` /
`:111 v41_rb_load` / `:121 v41_rb_clear_all` (`rb_Lnn.bin`: header
`<i32 n_expert><i32 1=f32>` plus the router-bias delta, pushed to the device
override), `:130 v41_gr_load` (the three-file multiplier: start at 1, multiply
the gr tables of (2) and (3) elementwise, upload per layer), `:166
v41_pt_base_ok` (fingerprint gate), `:192 amp_read_layer` (`amp_Lnn.bin`:
`<i32 D><i32 K><i32 stype>`, A then B, stype 1 f32 or 43 fp4x32), `:222
v41_amp_load`, `:274 v41_amp_hook`, `:311 v41_amp_free`, `:322 v41_amp_apply`.

Fingerprint, `src/common/ds4_gr_fnv.h:20 ds4_gr_dir_fnv`: FNV-1a 64 over
`gr_Lnn.bin` and `rb_Lnn.bin` of the sidecar directory in layer order, plus a
file count; `base.fnv` stores that plus the count. A missing file warns and
proceeds; a mismatch is a hard stop (a stale post-train file would otherwise
run and quietly produce wrong numbers).

Post-training lives in `core_ptrain*.c` (there is no `core_posttrain*` file):
the config/data loaders, the teacher pass with its cache, the per-layer
recompute and backward, the packed batch runner, the diagnostics, and
`pt_save_amp` writing `amp_Lnn.bin` plus `base.fnv`. The zchain `DQZ2` format
and its loader are `ds4_zchain.c:44`, with types 1-10 covering the gain chain,
V8, GE, frozen and dynamic `z^L`, the route sidecar and the four-loss record.

Flags: `--engram-dir`, `--v41-no-engram`, `--zchain`, `--zchain-scale`,
`--posttrain`, `--ptrain`, `--score-*`, `--draft-amp`, `--draft-amp-scale`.
No environment variables (grep of `src/core` and `src/cuda`).

## 8. MTP draft towers and speculative decode

Two drafters exist and must not be conflated: the V4-era DSpark drafter
(`dspark.*` tensors, `core_gpu_dspark.c`, `cuda_dspark.inc.cu`) and the V4.1
three-tower MTP drafter (`mtp.*` tensors, `core_v41_draft.c`,
`cuda_v41_draft.inc.cu`). The port needs the second.

Host, `src/core/core_v41_draft.c`: `:25 v41_small_matmul`, `:36
v41_draft_exp_off` (per-tower expert offset table, or the VQ blob and no
table), `:66 v41_draft_byte_report`, `:94 v41_draft_alloc`, `:179
v41_draft_free`, `:202 v41_draft_main_x`, `:213 v41_draft_put`, `:221
v41_draft_slots`, `:232 v41_draft_block`, `:301 v41_draft_fill`, `:317
v41_draft_gpu_round`, `:326 v41_draft_graph_round`, `:361 v41_draft_wait`,
`:373 v41_draft_launch` (0 cannot draft, 1 graph in flight, 2 direct dispatch
done), `:418 v41_draft_step`.

`core_v41_draft_amp.c` (exit alignment and distillation kit), `core_v41_dcap.c`
(`:59 v41_dcap_jury`, `:75 ds4_engine_v41_dspark_capture` — offline capture),
`core_draft_sched.c` (`:33 observe`, `:38 cost`, `:41 draft_cost`, `:47 fit`,
`:68 v41_draft_pick_k` — the confidence scheduler that chooses k from measured
costs), and the offline trainer `core_draft_kd*.c` (`--draft-train`; it
requires VQ-blob towers for backprop).

Device, `cuda_v41_draft.inc.cu`: `:16 v41_fp4_row_dot`, `:38
v41_mtp_gateup_kernel`, `:54 v41_mtp_down_kernel`, `:75 v41_mtp_bind`, `:101
ds4_gpu_v41_mtp_moe_tensor`, `:143 v41_hc_mean_kernel` + `:155` launcher,
`:169 v41_ring_rows_kernel` + `:177`, `:191 v41_row_gather_kernel` + `:205`,
`:218 v41_row_add_kernel` + `:222`, `:232 v41_mkcache_lookup_kernel` + `:246`,
`:255 v41_mkcache_add_kernel` + `:261` (the 256-slot markov bias cache).
`cuda_draft_attn.inc.cu` holds the KD-training primitives (attention forward
and backward, jury, total-variation loss, low-rank apply).

Tower contract: metadata `deepseek4.mtp.tower_count|expert_count|block_size|expert_used_count|noise_token_id|markov_rank|target_layers`;
the runtime group must be all present or speculation is not armed (the engine
does not guess defaults). Per tower, the same 21 tensor names as a normal
layer, plus experts in exactly one of two forms: one `mtp.T.ffn_exps_vq.blob`,
or per-expert `mtp.T.ffn_exp.E.{gate,up,down}.weight`. Five shared heads
(`mtp.main_proj`, `mtp.main_norm`, `mtp.markov_embd`, `mtp.markov_head`,
`mtp.confidence`) and the shared `mtp.out_norm`. `main_proj` takes the hc
four-route mean of the main model's target layers (official 37/38/39) and
projects it to `main_x`, the KV source of block attention; the code warns that
taking the wrong location does not error, it just drops acceptance to about 1.

Draft forward: the tower window holds `main_x` rather than the tower's own KV;
no compressed KV and no indexer; sliding window 128 with in-block full
visibility; block layout `[real token, noise x (B-1)]`; per position, the
markov head adds a bias to the logits row and greedy picks the token, while the
confidence head emits the per-position conditional acceptance logit.
`DS4_MTP_MAX_BLOCK = 8`.

Verify and rollback, from `core_v41_forward.c:241` and `:267` (the engine's own
explanation, translated): "Why back up: the verify batch computes 1+k positions
into the cache, and the ones not accepted must be undone. Window: this batch's
rows first sit in the buffered block region; after the layer they are written
into the ring by win_commit, which overwrites the cell from 128 steps ago, still
inside the next step's visible window. So only the n cells about to be
overwritten are saved: 40 layers x n (<=6) rows x 512 x 4 B, about 0.5 MB,
against 10.5 MB for the whole 128 rows, 22x. Compressor pending rows: that is a
genuine left shift, and the snapshot point is before the shift. comp_kv and
index_k are written by absolute group number and the counters are overwritten
next round, so no backup is needed." Acceptance: longest prefix where the main
model's own pick equals the draft; at temperature 0 that is bit-identical to
pure decode, so a mismatch is a rollback bug and not numerical drift. The two
things a port forgets: the main-hidden ring trim in rollback, and the
compressor pending-row restore point taken before the shift.

Flags and defaults: speculation is on by default; `--no-dspark` off, `--dspark`
explicit mode 2 (refused together with penalties), `--dspark-block`,
`--dspark-verify` (diagnostics, not speed knobs), `--draft-amp`,
`--draft-amp-scale`, `--draft-train`, `--dspark-capture`. No environment
variables reach this subsystem.

## 9. Status

All four enumerations are complete: core and serving (section 1-5), VQ
(section 6), engram and sidecars (section 7), MTP draft and speculative decode
(section 8). The next step is folding these into the plan's phase work list.

