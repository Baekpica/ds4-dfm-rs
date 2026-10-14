/* ds4_ds41_draft.inc — DeepSeek V4.1 (ds41) DSpark draft towers, host side.
 * Ported from the C engine at /data/YoungAi (commit 3946dbc), files
 * src/core/core_v41_draft.c (every function cites its engine line) and the
 * window/attention halves in src/core/core_v41_attn.c:225-286.  Included by
 * ds4.c right after ds4_ds41_forward.inc (same TU: it calls the forward's
 * statics).
 *
 * One draft round: draft a block of `block` positions in one forward (three
 * towers, 128 experts top-3, window attention only), one main-model verify of
 * 1+k positions, accept the longest matching prefix, roll the rest back — the
 * engine's rule is that temperature-0 greedy output is byte-identical with
 * drafting on and off (pure time saving, not approximation).
 *
 *   main_x      = main_norm(main_proj(the main model's target-layer attention
 *                 inputs, captured by v41_layer into the main-hidden ring))
 *   block input = [last real token, noise x (B-1)] embeddings (noise id from
 *                 the metadata)
 *   towers      = the same v41_layer body with the tower weights (draft=1)
 *   per position= head(out_norm(h)) row i + markov_head(token_i) bias ->
 *                 argmax -> token_{i+1}; confidence = proj([h_i; markov_embed_i])
 *
 * Failure modes (the engine's, kept): 1. capturing main_hidden at the layer
 * OUTPUT instead of the attention input does not error, acceptance drops to
 * about 1; 2. letting the draft block's KV into the history window does not
 * error, acceptance rots over the session; 3. a GGUF without the draft
 * parameters leaves ready=0 and the caller decodes one token at a time.
 *
 * Not ported here (named): the amp/distillation sidecars
 * (core_v41_draft_amp.c), --dspark-block, and the sampling draft arm
 * (dev_sample).  The dense per-expert MoE exists for the per-expert artifact
 * form; this artifact's towers carry VQ blobs.  The per-round CUDA graph is
 * ported below (E3; core_v41_draft.c:326-370). */
#ifndef DS4_DS41_DRAFT_INC
#define DS4_DS41_DRAFT_INC

#define DS4_V41_MKCACHE_SLOTS 256u   /* core_v41.h:168 (64 slots 50% hit / 256 slots 59%) */
/* core_v41.h:167: draft graphs are indexed by the window top-up width; 1+block
 * (6) is the steady width, a round that sat out a few steps tops up more. */
#define DS4_V41_DRAFT_GROWS 8u

typedef struct {
    ds4_v41_state st;          /* the towers' own buffers (draft=1, layer id = tower index) */
    ds4_gpu_tensor *mainx_raw; /* main_proj output (pre-norm) */
    ds4_gpu_tensor *mk_embed;  /* [cap][rank] per-position markov embed (confidence reads it) */
    ds4_gpu_tensor *mk_cur;    /* [rank] this position's row into markov_head */
    ds4_gpu_tensor *mk_bias;   /* [V] this position's logit bias */
    ds4_gpu_tensor *mk_cache, *mk_cache_ids, *mk_cache_next, *mk_hit;   /* [slots][V] f32 / [slots] id (-1 empty) / [1] rotation / [1] hit */
    ds4_gpu_tensor *conf_in, *conf;   /* [cap][E+rank] / [cap] */
    ds4_gpu_tensor *ids, *ids_next;   /* i32 [cap]: [0] = the real token, [1..block] = drafts; argmax landing slot */
    ds4_gpu_tensor *h;         /* [cap][E] the head's hc_pre output (confidence reads it too) */
    uint64_t *exp_off[DS4_MTP_MAX_TOWERS];   /* per-expert form: [3][n_expert] offsets; NULL for blobs */
    ds4_gpu_tensor *mainh_lin; /* [SWA][n_target][E] contiguous rows out of the main ring (main_proj's batch) */
    int64_t win_end;           /* newest absolute position in the tower windows (-1 = empty); the fill tops up (win_end, pos_main] */
    uint32_t block, ready;
    /* Sampling under speculation (core_v41_api.c:222): dev_sample = 1 makes
     * the per-row draft pick draw from the tower's own distribution (the same
     * device sample kernel, samp.stream = 1 so the coins are independent of
     * the verify's stream 0); the tower's logits become the verify kernel's q
     * (st.spec_q) for the rejection sampling.  0 = point-mass draft (argmax). */
    int dev_sample;
    ds4_gpu_sample_params samp;
    int32_t host_ids[DS4_MTP_MAX_BLOCK + 1];
    float host_conf[DS4_MTP_MAX_BLOCK];
    int32_t *p_tok, *p_bpos, *p_wpos, *p_first, *p_ids;   /* pinned slots the puts upload from */
    float *p_onehot, *p_conf;
    ds4_gpu_tensor *firstd;    /* device int: the ring-rows start (graph path; the eager path passes the value directly) */
    int cap_mode;              /* capture mode: 1 changes the puts to zero-copy kernels (E3) */
    /* Draft graph state (core_v41.h:199-210, core_v41_draft.c:326-370): one
     * exec per rows bucket, its scratch generation, and the direct warm-up
     * count that gates capture.  graph_off = a capture failed, drafts walk
     * direct from then on (loud).  pending = a graph was launched and not yet
     * waited (v41_draft_wait).  h_* = the host-gap account. */
    void *gexec[DS4_V41_DRAFT_GROWS + 1u];
    uint64_t ggen[DS4_V41_DRAFT_GROWS + 1u];
    uint32_t gwarm[DS4_V41_DRAFT_GROWS + 1u];
    int graph_off, pending;
    uint32_t gsteps, gcaps;    /* rounds walked on graphs / captures */
    double h_slots, h_launch, h_capture, t_launched;   /* slot-write seconds / cudaGraphLaunch seconds / capture seconds / this round's launch time */
    int last_warm;             /* the last round was a warm-up (request's first round, or a bucket's first direct round): the scheduler excludes its wall time */
    uint32_t rounds;           /* draft rounds produced */
} ds4_v41_draft;

/* core_v41_draft.c:25-33 — f32 vs bf16 BY THE REGISTERED GGUF TYPE, never
 * assumed.  The 2026-09-15 crash: the converter's plan_small stores
 * gate/markov/confidence as f32 while the main path's same-named matrices are
 * bf16; reading f32 bytes as bf16 does not error, it feeds garbage (acceptance
 * 0.14/5). */
static bool v41_small_matmul(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
                             uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n) {
    if (w->type == DS4_TENSOR_BF16)
        return ds4_gpu_v41_matmul_bf16_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n) != 0;
    if (w->type == DS4_TENSOR_F32)
        return ds4_gpu_v41_matmul_f32_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n) != 0;
    fprintf(stderr, "ds4: [v41] drafter: %.*s type %u is neither f32 nor bf16\n", (int)w->name.len, w->name.ptr, w->type);
    return false;
}

/* core_v41_draft.c:36-57 — the per-tower expert offset table [tower][3 x
 * n_expert], built once at bind time.  Why not in the kernel: a name lookup +
 * 128 range parses per layer is pure host waste. */
static bool v41_draft_exp_off(ds4_engine *e, ds4_v41_draft *dr) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    for (uint32_t T = 0; T < v->mtp_towers; T++) {
        /* Blob towers (the 100 GB recipe): no table; the state carries the blob
         * and the MoE takes the main fused kernel (v41_moe). */
        if (e->weights.v41_towers.exps_vq[T]) {
            dr->st.tower_exps_vq[T] = e->weights.v41_towers.exps_vq[T];
            dr->exp_off[T] = NULL; dr->st.tower_exp_off[T] = NULL;
            continue;
        }
        uint64_t *off = xmalloc((size_t)3u * v->mtp_experts * 8u);
        for (uint32_t i = 0; i < v->mtp_experts; i++) {
            const ds4_tensor *g = e->weights.v41_towers.exp_gate[T][i];
            const ds4_tensor *u = e->weights.v41_towers.exp_up[T][i];
            const ds4_tensor *d = e->weights.v41_towers.exp_down[T][i];
            if (!g || !u || !d) { free(off); fprintf(stderr, "ds4: [v41] draft tower %u expert %u tensors missing\n", T, i); return false; }
            off[i] = g->abs_offset; off[v->mtp_experts + i] = u->abs_offset; off[2u * v->mtp_experts + i] = d->abs_offset;
        }
        dr->exp_off[T] = off;
        dr->st.tower_exp_off[T] = off;
    }
    return true;
}

/* core_v41_draft.c:103-150 — the draft state's buffer list, minus amp /
 * distillation / graph tensors (named above). */
static bool v41_draft_state_alloc(ds4_v41_draft *dr, uint32_t cap, uint32_t towers) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint32_t FF = DS4_N_FF_EXP, R = v->mtp_markov_rank;
    const uint32_t pcap = cap > SWA ? cap : SWA;
    const uint64_t low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, mix = 2u * HC + HC * HC;
    ds4_v41_state *st = &dr->st;
    bool ok = true;
    st->draft = 1; st->cap_tok = cap; st->ctx = cap; st->idx_owner = -1; st->cand_owner = -1;
    st->tok = v41_alloc(cap * 4, &ok);          st->pos = v41_alloc((uint64_t)pcap * 4, &ok);
    st->hc = v41_alloc((uint64_t)cap * HC * E * 4, &ok);   st->hc2 = v41_alloc((uint64_t)cap * HC * E * 4, &ok);
    st->mix = v41_alloc((uint64_t)cap * mix * 4, &ok);      st->pre = v41_alloc((uint64_t)cap * HC * 4, &ok);
    st->post = v41_alloc((uint64_t)cap * HC * 4, &ok);      st->comb = v41_alloc((uint64_t)cap * HC * HC * 4, &ok);
    st->pre_mix = v41_alloc((uint64_t)cap * HC * 4, &ok);
    st->x = v41_alloc((uint64_t)cap * E * 4, &ok);          st->xn = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->qr = v41_alloc((uint64_t)cap * Q * 4, &ok);         st->qrn = v41_alloc((uint64_t)cap * Q * 4, &ok);
    st->q = v41_alloc((uint64_t)cap * NH * HD * 4, &ok);
    st->kv = v41_alloc((uint64_t)pcap * HD * 4, &ok);       st->kvn = v41_alloc((uint64_t)pcap * HD * 4, &ok);
    st->wintmp = v41_alloc((uint64_t)SWA * HD * 4, &ok);
    st->o = v41_alloc((uint64_t)cap * NH * HD * 4, &ok);    st->low = v41_alloc((uint64_t)cap * low * 4, &ok);
    st->attn_out = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->glog = v41_alloc((uint64_t)cap * v->mtp_experts * 4, &ok);
    st->sel = v41_alloc((uint64_t)cap * v->mtp_used * 4, &ok);
    st->rw = v41_alloc((uint64_t)cap * v->mtp_used * 4, &ok);
    st->routed = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->sg = v41_alloc((uint64_t)cap * FF * 4, &ok);        st->su = v41_alloc((uint64_t)cap * FF * 4, &ok);
    st->sh = v41_alloc((uint64_t)cap * FF * 4, &ok);        st->so = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->y = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->logits = v41_alloc((uint64_t)cap * DS4_N_VOCAB * 4, &ok);
    st->main_x = v41_alloc((uint64_t)pcap * E * 4, &ok);
    /* The tower windows must hold one full window top-up (up to SWA rows land
     * at once after a prefill); sizing them block+1 was the 2026-09-18 crash
     * that silently disabled speculation (the fill failed quietly and "spec ==
     * pure decode" stayed green because no round ever drafted). */
    for (uint32_t T = 0; T < towers; T++) st->win[T] = v41_alloc((uint64_t)(SWA + pcap) * HD * 4, &ok);
    dr->mainx_raw = v41_alloc((uint64_t)pcap * E * 4, &ok);
    dr->mk_embed = v41_alloc((uint64_t)cap * R * 4, &ok);
    dr->mk_cur = v41_alloc((uint64_t)R * 4, &ok);
    dr->mk_bias = v41_alloc((uint64_t)DS4_N_VOCAB * 4, &ok);
    dr->mk_cache = v41_alloc((uint64_t)DS4_V41_MKCACHE_SLOTS * DS4_N_VOCAB * 4, &ok);
    dr->mk_cache_ids = v41_alloc((uint64_t)DS4_V41_MKCACHE_SLOTS * 4, &ok);
    dr->mk_cache_next = v41_alloc(16, &ok);
    dr->mk_hit = v41_alloc(16, &ok);
    if (ok) {
        int32_t neg[DS4_V41_MKCACHE_SLOTS];
        for (uint32_t i = 0; i < DS4_V41_MKCACHE_SLOTS; i++) neg[i] = -1;
        const uint32_t zero4[4] = { 0u, 0u, 0u, 0u };
        if (!ds4_gpu_tensor_write(dr->mk_cache_ids, 0, neg, sizeof neg) || !ds4_gpu_tensor_write(dr->mk_cache_next, 0, zero4, 16)) ok = false;
    }
    dr->conf_in = v41_alloc((uint64_t)cap * (E + R) * 4, &ok);
    dr->conf = v41_alloc((uint64_t)cap * 4, &ok);
    dr->ids = v41_alloc((uint64_t)cap * 4, &ok);
    dr->ids_next = v41_alloc(16, &ok);
    dr->h = v41_alloc((uint64_t)cap * E * 4, &ok);
    dr->mainh_lin = v41_alloc((uint64_t)SWA * v->n_mtp_target * E * 4, &ok);
    dr->win_end = -1;
    dr->p_tok = ds4_gpu_host_alloc((uint64_t)cap * 4);
    dr->p_bpos = ds4_gpu_host_alloc((uint64_t)cap * 4);
    /* p_wpos sizes by the window, not the graph's rows bucket: the first
     * post-prefill round tops up up to SWA rows (core_v41_draft.c:148-151). */
    dr->p_wpos = ds4_gpu_host_alloc((uint64_t)SWA * 4);
    dr->p_first = ds4_gpu_host_alloc(4);
    dr->p_ids = ds4_gpu_host_alloc((uint64_t)cap * 4);
    dr->p_conf = ds4_gpu_host_alloc((uint64_t)cap * 4);
    dr->p_onehot = ds4_gpu_host_alloc((uint64_t)cap * DS4_N_HC * 4);
    dr->firstd = v41_alloc(16, &ok);
    if (!dr->p_tok || !dr->p_bpos || !dr->p_wpos || !dr->p_first || !dr->p_ids || !dr->p_conf || !dr->p_onehot) ok = false;
    if (ok) for (uint32_t i = 0; i < cap * DS4_N_HC; i++) dr->p_onehot[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    return ok;
}

/* core_v41_draft.c:94-175 — arm the drafter; ready=0 means the caller decodes
 * one token at a time (no draft parameters in the GGUF, no towers). */
static void v41_draft_free(ds4_v41_draft *dr);
static bool v41_draft_alloc(ds4_engine *e, ds4_v41_draft *dr) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    memset(dr, 0, sizeof *dr);
    if (!v->mtp_towers || !v->mtp_block || !v->n_mtp_target) return false;
    if (!e->weights.v41_towers.main_proj || !e->weights.v41_towers.markov_embd || !e->weights.v41_towers.confidence) return false;
    const uint32_t B = v->mtp_block, cap = B + 1u;   /* +1: the verify batch runs up to 1+block rows; main_x sizes by it too */
    if (!v41_draft_state_alloc(dr, cap, v->mtp_towers)) { v41_draft_free(dr); return false; }
    if (!v41_draft_exp_off(e, dr)) { v41_draft_free(dr); return false; }
    dr->block = B;   /* the engine's --dspark-block clamp is not ported (named) */
    dr->ready = 1;
    fprintf(stderr, "ds4: [v41] DSpark drafter armed: %u towers x %u experts top-%u (%s), block %u, target layers",
            v->mtp_towers, v->mtp_experts, v->mtp_used, e->weights.v41_towers.exps_vq[0] ? "VQ blob" : "per-expert fp4x32", B);
    for (uint32_t i = 0; i < v->n_mtp_target; i++) fprintf(stderr, " L%02d", (int)v->mtp_target[i]);
    fprintf(stderr, "\n");
    return true;
}

/* core_v41_draft.c:179-200, minus the amp teardown. */
static void v41_draft_free(ds4_v41_draft *dr) {
    ds4_v41_state *st = &dr->st;
    for (uint32_t r = 0; r <= DS4_V41_DRAFT_GROWS; r++) { if (dr->gexec[r]) ds4_gpu_decode_graph_free(dr->gexec[r]); dr->gexec[r] = NULL; }
    ds4_gpu_host_free(dr->p_tok); ds4_gpu_host_free(dr->p_bpos); ds4_gpu_host_free(dr->p_wpos); ds4_gpu_host_free(dr->p_first);
    ds4_gpu_host_free(dr->p_ids); ds4_gpu_host_free(dr->p_conf); ds4_gpu_host_free(dr->p_onehot);
    dr->p_tok = dr->p_bpos = dr->p_wpos = dr->p_first = dr->p_ids = NULL; dr->p_conf = dr->p_onehot = NULL;
    ds4_gpu_tensor **all[] = { &st->tok, &st->pos, &st->hc, &st->hc2, &st->mix, &st->pre, &st->post, &st->comb, &st->pre_mix,
        &st->x, &st->xn, &st->qr, &st->qrn, &st->q, &st->kv, &st->kvn, &st->wintmp, &st->o, &st->low, &st->attn_out,
        &st->glog, &st->sel, &st->rw, &st->routed, &st->sg, &st->su, &st->sh, &st->so, &st->y, &st->logits, &st->main_x,
        &dr->mainx_raw, &dr->mk_embed, &dr->mk_cur, &dr->mk_bias, &dr->conf_in, &dr->conf, &dr->ids, &dr->ids_next, &dr->h,
        &dr->mainh_lin, &dr->firstd, &dr->mk_cache, &dr->mk_cache_ids, &dr->mk_cache_next, &dr->mk_hit };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) { if (*all[i]) ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
    for (uint32_t T = 0; T < DS4_MTP_MAX_TOWERS; T++) {
        if (st->win[T]) { ds4_gpu_tensor_free(st->win[T]); st->win[T] = NULL; }
        free(dr->exp_off[T]); dr->exp_off[T] = NULL; st->tower_exp_off[T] = NULL; st->tower_exps_vq[T] = NULL;
    }
    dr->ready = 0;
}

/* core_v41_draft.c:202-211 — main_x = main_norm(main_proj(mainh_lin rows)),
 * the official DSparkBlock.forward_embed.  main_proj dispatches on the
 * registered type (FP8 / q4_K / fp4x32) via v41_tproj. */
static bool v41_draft_main_x(ds4_engine *e, ds4_v41_draft *dr, uint32_t rows) {
    const ds4_model *m = &e->model;
    const uint64_t in = (uint64_t)DS4_N_EMBD * g_ds4_v41.n_mtp_target;
    if (!v41_tproj(m, dr->mainx_raw, e->weights.v41_towers.main_proj, in, DS4_N_EMBD, dr->mainh_lin, rows, 1)) return false;
    return ds4_gpu_v41_rms_norm_tensor(dr->st.main_x, dr->mainx_raw, m->map, m->size,
                                       e->weights.v41_towers.main_norm->abs_offset, DS4_N_EMBD, rows, DS4_RMS_EPS) != 0;
}

/* core_v41_draft.c:213-219 — host -> device; capture mode would switch to the
 * zero-copy kernel (E3), the eager path writes synchronously. */
static bool v41_draft_put(const ds4_v41_draft *dr, ds4_gpu_tensor *t, uint64_t off, const void *pinned, uint64_t bytes) {
    return dr->cap_mode ? ds4_gpu_tensor_write_zerocopy(t, off, pinned, bytes) != 0 : ds4_gpu_tensor_write(t, off, pinned, bytes) != 0;
}

/* core_v41_draft.c:221-230 — the round's host slots: the window top-up
 * positions / ring start row + the block input [real token, noise x (B-1)]
 * (official draft_input_ids) and block positions pos0..pos0+B-1.  Every path
 * (direct, capture, replay) must call it: the graph bakes only the slot
 * ADDRESSES, values are read at replay time (the 2026-09-24 conviction:
 * replay read capture-time positions, the window rotted, acceptance fell
 * 0.47 -> 0.03 with no error). */
static void v41_draft_slots(const ds4_v41_state *main_st, ds4_v41_draft *dr, int64_t first, uint32_t rows, uint32_t pos0) {
    for (uint32_t i = 0; i < rows; i++) dr->p_wpos[i] = (int32_t)(first + (int64_t)i);   /* absolute positions (rope) */
    if (rows) dr->p_first[0] = (int32_t)(first % (int64_t)main_st->mainh_cap);
    for (uint32_t i = 0; i < dr->block; i++) {
        dr->p_tok[i] = i ? (int32_t)g_ds4_v41.mtp_noise_id : dr->host_ids[0];
        dr->p_bpos[i] = (int32_t)(pos0 + i);
    }
    __sync_synchronize();
}

/* core_v41_attn.c:239-261 — the draft tower's attention (official
 * DSparkAttention).  Three differences from the main route: no compressed
 * KV/indexer (compress_ratio == 0), the window holds the main model's
 * per-position main_x projections (pushed by v41_draft_push_main), and all n
 * positions of the block see each other (full_block).
 * The block's KV must NOT enter the history window: the draft is a trial, and
 * only main-confirmed positions belong in the window (else the next round
 * drafts from a history containing rejected drafts — no error, acceptance
 * rots). */
static bool v41_draft_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model;
    const ds4_layer_weights *l = &e->weights.v41_towers.tower[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint64_t rowb = (uint64_t)HD * 4;
    if (!v41_tproj(m, st->qr, l->attn_q_a, E, Q, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->qrn, st->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS)) return false;
    if (!v41_tproj(m, st->q, l->attn_q_b, Q, (uint64_t)NH * HD, st->qrn, n, 1)) return false;
    if (!v41_rope(st->q, st->pos, n, NH, HD, 0, false)) return false;
    if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->kvn, st->pos, n, 1, HD, 0, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, n, HD, 32)) return false;
    if (!ds4_gpu_tensor_copy(st->win[il], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)n * rowb)) return false;
    /* ring=0: the tower window is a linear segment maintained by
     * v41_draft_push_main, not the main route's ring. */
    if (!ds4_gpu_v41_sparse_attn_tensor(st->o, st->q, st->win[il], NULL, NULL, m->map, m->size,
                                        l->attn_sinks->abs_offset, n, st->pos0, SWA, 0, 0, 0, NH, HD,
                                        (float)(1.0 / sqrt((double)HD)), 1, 0, 0u, NULL, 0u)) return false;   /* full window, no dirty slots */
    if (!v41_rope(st->o, st->pos, n, NH, HD, 0, true)) return false;
    const uint32_t grp = NH / DS4_N_OUT_GROUP;
    if (!v41_tproj_grouped(m, st->low, l->attn_output_a, DS4_N_OUT_GROUP, (uint64_t)grp * HD, DS4_N_LORA_O, st->o, n, 1)) return false;
    return v41_tproj(m, st->attn_out, l->attn_output_b, (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, E, st->low, n, 1);
}

/* core_v41_attn.c:263-280 — project `rows` CONFIRMED positions' main_x into
 * every tower's window (official self.window_kv_cache[start_pos % win] =
 * main_kv).  The window is a linear buffer, not a ring: new rows go to the
 * block region, then the whole thing shifts left by rows — same effect, one
 * modulo less.  st->pos holds the rows' absolute positions (rope). */
static bool v41_draft_push_main(ds4_engine *e, ds4_v41_state *st, uint32_t rows) {
    const ds4_model *m = &e->model;
    const uint32_t E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA;
    const uint64_t rowb = (uint64_t)HD * 4;
    if (!rows) return true;
    for (uint32_t T = 0; T < g_ds4_v41.mtp_towers; T++) {
        const ds4_layer_weights *l = &e->weights.v41_towers.tower[T];
        if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->main_x, rows, 1)) return false;
        if (!ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, rows, DS4_RMS_EPS)) return false;
        if (!v41_rope(st->kvn, st->pos, rows, 1, HD, 0, false)) return false;
        if (!ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, rows, HD, 32)) return false;
        if (!ds4_gpu_tensor_copy(st->win[T], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)rows * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->wintmp, 0, st->win[T], (uint64_t)rows * rowb, (uint64_t)SWA * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->win[T], 0, st->wintmp, 0, (uint64_t)SWA * rowb)) return false;
    }
    return true;
}

/* core_v41_draft.c:232-299 — one draft block forward: towers -> head ->
 * per-position markov greedy -> confidence.  Slots are filled by
 * v41_draft_slots (ids[0] = the last real token). */
static bool v41_draft_block(ds4_engine *e, ds4_v41_draft *dr, uint32_t pos0) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const ds4_model *m = &e->model;
    ds4_v41_state *st = &dr->st;
    const uint32_t B = dr->block, E = DS4_N_EMBD, R = v->mtp_markov_rank;
    st->n = B; st->pos0 = pos0;
    if (!v41_draft_put(dr, st->tok, 0, dr->p_tok, (uint64_t)B * 4) || !v41_draft_put(dr, st->pos, 0, dr->p_bpos, (uint64_t)B * 4)) return false;
    if (!v41_draft_put(dr, st->pre_mix, 0, dr->p_onehot, (uint64_t)B * DS4_N_HC * 4)) return false;
    if (!v41_embed(m, st->x, st->tok, e->weights.token_embd, DS4_N_VOCAB, B, E)) return false;
    if (!ds4_gpu_v41_expand_hc_tensor(st->hc, st->x, E, DS4_N_HC, B)) return false;
    for (uint32_t T = 0; T < v->mtp_towers; T++) if (!v41_layer(e, st, T)) return false;
    /* Exit: the official mtp[-1] borrows the main model's head; the norm is
     * mtp.2.norm (out_norm).  The amp correction (dr->ampK) is not ported:
     * without the sidecar it is the identity anyway (core_v41_draft.c:252). */
    if (!ds4_gpu_v41_hc_pre_tensor(dr->h, st->hc, st->pre_mix, E, DS4_N_HC, B)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->xn, dr->h, m->map, m->size, e->weights.v41_towers.out_norm->abs_offset, E, B, DS4_RMS_EPS)) return false;
    if (!v41_tproj(m, st->logits, e->weights.output, E, DS4_N_VOCAB, st->xn, B, 0)) return false;
    /* Per position, all on device (one D2H per position would cost the whole
     * speculation gain): logits[i] += markov_head(token_i) -> argmax ->
     * token_{i+1}. */
    const uint64_t nrow = e->weights.v41_towers.markov_embd->ndim > 1 ? e->weights.v41_towers.markov_embd->dim[1] : DS4_N_VOCAB;
    const uint32_t eb = e->weights.v41_towers.markov_embd->type == DS4_TENSOR_BF16 ? 2u : 4u;
    /* Bias cache: only the original bf16 markov_head path caches (a hit
     * returns the whole GEMV grid; the bias comes from the slot). */
    const bool mkc = dr->mk_cache && e->weights.v41_towers.markov_head->type == DS4_TENSOR_BF16;
    for (uint32_t i = 0; i < B; i++) {
        if (mkc && !ds4_gpu_v41_mkcache_lookup_tensor(dr->mk_hit, dr->mk_cache_ids, dr->mk_cache_next, dr->ids, i, DS4_V41_MKCACHE_SLOTS)) return false;
        if (!ds4_gpu_v41_row_gather_tensor(dr->mk_embed, m->map, m->size, e->weights.v41_towers.markov_embd->abs_offset,
                                           nrow, R, eb, dr->ids, i, i)) return false;   /* the confidence head reads the original row */
        if (!ds4_gpu_v41_row_gather_tensor(dr->mk_cur, m->map, m->size, e->weights.v41_towers.markov_embd->abs_offset,
                                           nrow, R, eb, dr->ids, i, 0)) return false;
        if (mkc) {
            if (!ds4_gpu_v41_matmul_bf16_skip_tensor(dr->mk_bias, m->map, m->size, e->weights.v41_towers.markov_head->abs_offset,
                                                     R, DS4_N_VOCAB, dr->mk_cur, 1, dr->mk_hit)) return false;
        } else if (!v41_small_matmul(m, dr->mk_bias, e->weights.v41_towers.markov_head, R, DS4_N_VOCAB, dr->mk_cur, 1)) return false;
        if (mkc ? !ds4_gpu_v41_mkcache_add_tensor(st->logits, i, dr->mk_bias, dr->mk_cache, dr->mk_hit, DS4_N_VOCAB)
                : !ds4_gpu_v41_row_add_tensor(st->logits, i, dr->mk_bias, DS4_N_VOCAB)) return false;
        /* argmax writes only the first int of its own row slot: land it in
         * ids_next, then copy to ids[i+1] (official output_ids[:, i+1]).
         * Under sampling the draft also draws from the tower's distribution
         * (2026-09-29, core_v41_draft.c:285-287): the accept-rate ceiling
         * becomes sum min(p,q) instead of p(argmax q); the sample lands in the
         * same slot's first int. */
        if (dr->dev_sample) {
            if (!ds4_gpu_v41_sample_tensor(dr->ids_next, st->logits, i, 1u, DS4_N_VOCAB, st->pos, st->tok, &dr->samp, NULL)) return false;
        } else if (!ds4_gpu_v41_argmax_tensor(dr->ids_next, st->logits, i, DS4_N_VOCAB)) return false;
        if (!ds4_gpu_tensor_copy(dr->ids, (uint64_t)(i + 1u) * 4, dr->ids_next, 0, 4)) return false;
    }
    /* confidence = proj([h_i; markov_embed_i]) */
    for (uint32_t i = 0; i < B; i++) {
        if (!ds4_gpu_tensor_copy(dr->conf_in, (uint64_t)i * (E + R) * 4, dr->h, (uint64_t)i * E * 4, (uint64_t)E * 4)) return false;
        if (!ds4_gpu_tensor_copy(dr->conf_in, ((uint64_t)i * (E + R) + E) * 4, dr->mk_embed, (uint64_t)i * R * 4, (uint64_t)R * 4)) return false;
    }
    return v41_small_matmul(m, dr->conf, e->weights.v41_towers.confidence, (uint64_t)E + R, 1, dr->conf_in, B);
}

/* core_v41_draft.c:301-315 — top the window up: rows [first, first+rows) out
 * of the main ring -> main_proj/main_norm -> push into the tower windows.
 * These three steps MUST speak up on failure (the 2026-09-18 conviction: a
 * silent return false made the caller read "no draft this round", an
 * undersized buffer disabled the whole speculation path and the "spec ==
 * pure decode" gate stayed green). */
static bool v41_draft_fill(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, uint32_t rows) {
    if (!v41_draft_put(dr, dr->st.pos, 0, dr->p_wpos, (uint64_t)rows * 4)) { fprintf(stderr, "ds4: [v41] drafter: writing %u window positions failed\n", rows); return false; }
    if (dr->cap_mode && !ds4_gpu_tensor_write_zerocopy(dr->firstd, 0, dr->p_first, 4)) { fprintf(stderr, "ds4: [v41] drafter: writing the start row failed\n"); return false; }
    if (!ds4_gpu_v41_ring_rows_tensor(dr->mainh_lin, main_st->mainh, g_ds4_v41.n_mtp_target * DS4_N_EMBD, main_st->mainh_cap,
                                      (uint32_t)dr->p_first[0], rows, dr->cap_mode ? dr->firstd : NULL)) {
        fprintf(stderr, "ds4: [v41] drafter: taking %u rows from the main_hidden ring failed\n", rows); return false;
    }
    if (!v41_draft_main_x(e, dr, rows)) { fprintf(stderr, "ds4: [v41] drafter: main_proj/main_norm over %u rows failed\n", rows); return false; }
    if (!v41_draft_push_main(e, &dr->st, rows)) { fprintf(stderr, "ds4: [v41] drafter: pushing %u rows into the tower windows failed\n", rows); return false; }
    /* the block attention's main_x is only the last row (the official
     * forward_spec's main_hidden is the current position) */
    if (rows > 1 && !ds4_gpu_tensor_copy(dr->st.main_x, 0, dr->st.main_x, (uint64_t)(rows - 1u) * DS4_N_EMBD * 4,
                                         (uint64_t)DS4_N_EMBD * 4)) return false;
    return true;
}

/* core_v41_draft.c:317-324 — one round's GPU part (fill + ids[0] + block),
 * shared by the direct and (future) capture paths; the only difference is
 * whether the host writes go through sync copies or zero-copy kernels. */
static bool v41_draft_gpu_round(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int64_t first, uint32_t rows, uint32_t pos0) {
    v41_draft_slots(main_st, dr, first, rows, pos0);
    if (rows && !v41_draft_fill(e, main_st, dr, rows)) return false;
    if (!v41_draft_put(dr, dr->ids, 0, dr->p_tok, 4)) return false;   /* p_tok[0] = the real token (caller-filled) */
    return v41_draft_block(e, dr, pos0);
}

/* ---- the draft round's CUDA graph (core_v41_draft.c:326-360) ----
 * One graph per rows bucket (the window top-up width): the fill's shapes move
 * with rows, so each width captures its own, warmed direct first (gwarm).
 * Capture failure turns every draft graph off, loudly -- a silent fallback
 * would leave "spec == pure decode" vacuously green while every draft walked
 * direct (the engine's 2026-09-18 conviction). */
static bool v41_draft_graph_round(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int64_t first, uint32_t rows, uint32_t pos0) {
    const uint64_t gen = ds4_gpu_v41_scratch_generation();
    if (dr->gexec[rows] && dr->ggen[rows] != gen) {   /* a scratch pointer moved: every draft graph baked the old ones */
        for (uint32_t r = 0; r <= DS4_V41_DRAFT_GROWS; r++) { if (dr->gexec[r]) ds4_gpu_decode_graph_free(dr->gexec[r]); dr->gexec[r] = NULL; }
        fprintf(stderr, "ds4: [graph] draft graphs: backend scratch moved, all re-capturing\n");
    }
    if (!dr->gexec[rows]) {
        const double tc0 = now_sec();
        dr->cap_mode = 1;
        if (!ds4_gpu_decode_graph_capture_begin()) { dr->cap_mode = 0; dr->graph_off = 1; return false; }
        bool ok = v41_draft_gpu_round(e, main_st, dr, first, rows, pos0);
        if (ok) ok = ds4_gpu_tensor_read_zerocopy(dr->p_ids, dr->ids, 0, (uint64_t)(dr->block + 1u) * 4) != 0 &&
                     ds4_gpu_tensor_read_zerocopy(dr->p_conf, dr->conf, 0, (uint64_t)dr->block * 4) != 0;
        dr->cap_mode = 0;
        void *exec = ds4_gpu_decode_graph_capture_end();   /* end the capture on every path or the stream stays in capture mode */
        if (!ok || !exec) {
            if (exec) ds4_gpu_decode_graph_free(exec);
            fprintf(stderr, "ds4: *[graph] draft graph (top-up %u rows) capture failed: drafts walk direct*\n", rows);
            dr->graph_off = 1;
            return false;
        }
        dr->gexec[rows] = exec; dr->ggen[rows] = gen; dr->gcaps++; dr->h_capture += now_sec() - tc0;
        fprintf(stderr, "ds4: [graph] draft graph (top-up %u rows) captured\n", rows);
    }
    /* Replay reads the slots at replay time: refill them for this round (the
     * capture pass filled the same values; harmless).  Launch only (the
     * 2026-10-07 split): the wait and readback live in v41_draft_wait so the
     * caller can emit while the draft runs. */
    const double tl0 = now_sec();
    v41_draft_slots(main_st, dr, first, rows, pos0);
    const double tl1 = now_sec();
    if (!ds4_gpu_decode_graph_launch(dr->gexec[rows])) return false;
    dr->t_launched = now_sec(); dr->h_slots += tl1 - tl0; dr->h_launch += dr->t_launched - tl1;
    dr->pending = 1;
    return true;
}
/* The other half of a launched draft graph: wait, then read the pinned
 * landing (v41_draft_launch returned 1).  A direct round already has its
 * results; its wait is a no-op. */
static bool v41_draft_wait(ds4_v41_draft *dr) {
    if (!dr->pending) return true;
    dr->pending = 0;
    if (!ds4_gpu_synchronize()) return false;
    __sync_synchronize();
    memcpy(dr->host_ids, dr->p_ids, (size_t)(dr->block + 1u) * 4);
    memcpy(dr->host_conf, dr->p_conf, (size_t)dr->block * 4);
    dr->gsteps++;
    return true;
}

/* core_v41_draft.c:373-420.  Returns 0 = cannot draft (inputs missing),
 * 1 = a draft graph is in flight (host_ids/host_conf need v41_draft_wait),
 * 2 = the direct dispatch ran synchronously (results ready; wait is a no-op). */
static int v41_draft_launch(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main) {
    if (!dr->ready || !main_st->mainh) return 0;
    if (main_st->mainh_end != (int64_t)pos_main) {   /* the main forward did not write this position into the ring (CED skip / early stop) = inputs missing */
        fprintf(stderr, "ds4: [v41] drafter: main_hidden ring ends at %lld != pos_main %u, no draft this round\n", (long long)main_st->mainh_end, pos_main);
        return 0;
    }
    /* Top up (win_end, pos_main] — rounds that sat out and graph-warmed steps
     * are all caught here; a gap wider than the window rebuilds it. */
    const uint32_t SWA = DS4_N_SWA;
    int64_t first = dr->win_end + 1;
    if ((int64_t)pos_main - first + 1 > (int64_t)SWA) first = (int64_t)pos_main - (int64_t)SWA + 1;
    if (first < 0) first = 0;
    const uint32_t rows = (uint32_t)((int64_t)pos_main - first + 1);
    if (rows) {
        const int64_t avail_first = main_st->mainh_end - (int64_t)main_st->mainh_n + 1;
        if (first < avail_first) {
            fprintf(stderr, "ds4: [v41] drafter: window needs %lld..%u, the ring's contiguous run starts at %lld (%u rows), no draft this round\n",
                    (long long)first, pos_main, (long long)avail_first, main_st->mainh_n);
            return 0;
        }
    }
    dr->host_ids[0] = tok; dr->p_tok[0] = tok;
    const uint32_t pos0 = pos_main + 1u;
    /* Graph eligibility (core_v41_draft.c:395-399): the main state allows
     * graphs, no capture failed before, this rows width warmed direct (its
     * lazy allocations are built), and pos0 >= the window -- the block
     * attention bakes pos0; below the window the key count varies with the
     * position and no one graph fits. */
    if (v41_graph_allowed(main_st) && !dr->graph_off && rows >= 1u && rows <= DS4_V41_DRAFT_GROWS && pos0 >= SWA && dr->gwarm[rows] > 0 &&
        v41_draft_graph_round(e, main_st, dr, first, rows, pos0)) {
        dr->win_end = pos_main;
        dr->rounds++; dr->last_warm = 0;
        return 1;
    }
    /* Warm-up round = the request's first round (all lazy allocations and
     * first touches pay here) or a rows bucket's first direct round that could
     * capture next time.  A direct round that can never graph (position below
     * the window / rows out of range / graphs off) is NOT warm: its wall time
     * is its true cost and the scheduler counts it (core_v41_draft.c:403-405). */
    const bool in_grows = rows >= 1u && rows <= DS4_V41_DRAFT_GROWS;
    dr->last_warm = dr->rounds == 0u || (in_grows && dr->gwarm[rows] == 0u && pos0 >= SWA && !dr->graph_off);
    if (!v41_draft_gpu_round(e, main_st, dr, first, rows, pos0)) return 0;
    if (rows) dr->win_end = pos_main;
    if (!ds4_gpu_synchronize()) return 0;
    if (!ds4_gpu_tensor_read(dr->ids, 0, dr->host_ids, (uint64_t)(dr->block + 1u) * 4)) return 0;
    if (!ds4_gpu_tensor_read(dr->conf, 0, dr->host_conf, (uint64_t)dr->block * 4)) return 0;
    if (in_grows) dr->gwarm[rows]++;
    dr->rounds++;
    dr->pending = 0;
    return 2;
}
__attribute__((unused))   /* the ported engine step (core_v41_draft.c:418-420); the run loop calls launch/wait directly */
static bool v41_draft_step(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main) {
    const int r = v41_draft_launch(e, main_st, dr, tok, pos_main);
    return r == 2 || (r == 1 && v41_draft_wait(dr));   /* core_v41_draft.c:418-420 */
}

/* ---- the confidence scheduler (core_draft_sched.c:31-97) ----
 * How many positions to verify each round: the drafter's own confidence says
 * how likely position j survives, and the engine measures the costs per
 * request (walked-graph step / one draft round / a verify of n rows) instead
 * of baking constants — a kernel that gets faster leaves stale constants
 * behind and the scheduler then blocks speculation entirely (the 2026-09-17
 * conviction: zero draft rounds in 40 tokens after a new kernel landed).
 *   round yield = 1 + sum_{j<=k} prod_{i<=j} rho*sigmoid(conf_i)
 *   round cost  = draft + verify(1+k rows)
 * pick k = argmax yield/cost; > 1 (vs one pure-decode step) means worth it.
 * k only affects speed, never the output: temperature-0 spec == pure decode
 * byte for byte (the d1 gate re-runs on every change).  rho (the acceptance
 * calibration for sampling) is history-only, so it is reproducible at temp 0;
 * the measured costs are not, which is why the trace gate pins k
 * (--dspark-verify / verify_k). */
typedef struct {
    double pred1, real1;
    double ms[DS4_MTP_MAX_BLOCK + 2u]; uint32_t cnt[DS4_MTP_MAX_BLOCK + 2u];
    double draft_ms; uint32_t draft_n;
} v41_sched;

static float v41_sched_sigmoid(float c) { return isfinite(c) ? 1.0f / (1.0f + expf(-c)) : 0.0f; }   /* non-finite conf = no confidence */

/* Sampling calibration (core_draft_sched.c:33-36): the conf head learned
 * "does greedy pick the same token", which is systematically optimistic under
 * sampling.  rho = first-position hits / sum of first sigmoids over this
 * request's drafted rounds, multiplied into every position's sigma (capped
 * at 1).  Observed every round, k=0 included, or a first-round rejection
 * deadlocks rho at 0. */
static void v41_sched_observe(v41_sched *s, const float *conf, int hit) {
    if (!s) return;
    s->pred1 += (double)v41_sched_sigmoid(conf[0]);
    s->real1 += hit ? 1.0 : 0.0;
}
/* Cost samples: n_rows = 1 is a single-token step, >= 2 a verify of 1+k rows. */
static void v41_sched_cost(v41_sched *s, uint32_t n_rows, double ms) {
    if (!s || n_rows == 0u || n_rows > DS4_MTP_MAX_BLOCK + 1u || !(ms > 0.0)) return;
    s->ms[n_rows] += ms; s->cnt[n_rows]++;
}
static void v41_sched_draft_cost(v41_sched *s, double ms) { if (s && ms > 0.0) { s->draft_ms += ms; s->draft_n++; } }

/* Weighted least squares ms(n) = v1 + tok*(n-1) over the per-n means (counts
 * as weights); one distinct x degenerates to a constant.  false = no samples. */
static bool v41_sched_fit(const v41_sched *s, double *v1, double *tok) {
    double W = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    uint32_t nx = 0; double x0 = 0;
    for (uint32_t n = 1; n <= DS4_MTP_MAX_BLOCK + 1u; n++) {
        if (!s->cnt[n]) continue;
        const double w = (double)s->cnt[n], x = (double)(n - 1u), y = s->ms[n] / w;
        if (nx == 0) x0 = x;
        if (nx == 0 || x != x0) nx++;
        W += w; Sx += w * x; Sy += w * y; Sxx += w * x * x; Sxy += w * x * y;
    }
    if (W <= 0.0) return false;
    const double den = W * Sxx - Sx * Sx;
    if (nx < 2u || den <= 0.0) { *v1 = Sy / W; *tok = 0.0; return true; }
    *tok = (W * Sxy - Sx * Sy) / den;
    *v1 = (Sy - *tok * Sx) / W;
    return true;
}

/* conf[0..block-1] = per-position confidence logits.  Returns how many
 * positions to verify (0 = not even the draft is worth its cost);
 * *value_out = the predicted yield/cost ratio (1.0 = break-even with one
 * pure-decode step). */
static uint32_t v41_draft_pick_k(const float *conf, uint32_t block, float *value_out, const v41_sched *s) {
    if (value_out) *value_out = 1.0f;
    if (!s || block == 0u) return block;
    /* Startup: no verify samples => measure the longest block once; one n>=2
     * point => add an n=2 (or the longest) point so the fit has a slope. */
    uint32_t nmeas = 0, first = 0;
    for (uint32_t n = 2; n <= block + 1u; n++) if (s->cnt[n]) { if (!nmeas) first = n; nmeas++; }
    if (nmeas == 0u) return block;
    if (nmeas == 1u && !s->cnt[1]) return first == 2u ? block : 1u;
    double v1, tok;
    if (!v41_sched_fit(s, &v1, &tok)) return block;
    /* One pure-decode step's ms: the real single-token samples when present,
     * else the fit at n=1 (verify 1 row ~ one step). */
    const double step = s->cnt[1] ? s->ms[1] / (double)s->cnt[1] : v1;
    const double draft = s->draft_n ? s->draft_ms / (double)s->draft_n : 0.0;
    if (!(step > 0.0)) return block;
    const float rho = s->pred1 > 0.0 ? (float)(s->real1 / s->pred1) : 1.0f;
    float surv = 1.0f, sum = 0.0f;
    double best = 1.0 / (draft + v1);   /* k=0: the draft is wasted, one single-token step runs */
    uint32_t bestk = 0;
    for (uint32_t j = 0; j < block; j++) {
        const float pj = rho * v41_sched_sigmoid(conf[j]);
        surv *= pj < 1.0f ? pj : 1.0f;
        sum += surv;
        const double cost = draft + v1 + tok * (double)(j + 1u);   /* verify 1+(j+1) rows */
        const double val = (1.0 + (double)sum) / (cost > 0.0 ? cost : step);
        if (val > best) { best = val; bestk = j + 1u; }
    }
    if (value_out) *value_out = (float)(best * step);   /* 1.0 = break-even with one pure-decode step */
    return bestk;
}

/* The DS41_GEN_LOGITS dump (see the generate entry): one full logits row per
 * emitted position, in emission order. */
static void v41_gen_dump_row(ds4_v41_state *st, FILE *fd) {
    if (!fd) return;
    float *row = xmalloc((size_t)DS4_N_VOCAB * 4);
    if (row && ds4_gpu_tensor_read(st->logits, (uint64_t)st->last_logit_row * DS4_N_VOCAB * 4, row, (uint64_t)DS4_N_VOCAB * 4)) {
        fwrite(row, 4, DS4_N_VOCAB, fd);
    }
    free(row);
}

/* The engine's per-run spec account for the server monitor
 * (core_v41_api.c:41-46): the generate entry clears them and the spec
 * summary writes them; ds4_engine_v41_last_spec_stats reads them.  The
 * server reports the round/accept rates, so the counts must outlive the
 * call. */
static int g_v41_last_spec_rounds, g_v41_last_spec_off, g_v41_last_spec_acc;
void ds4_engine_v41_last_spec_stats(int *rounds, int *offered, int *accepted) {
    if (rounds) *rounds = g_v41_last_spec_rounds;
    if (offered) *offered = g_v41_last_spec_off;
    if (accepted) *accepted = g_v41_last_spec_acc;
}

/* ---- the greedy generate entry (core_v41_api.c:426-470 + the round drive
 * :175-425, the eager subset) ----
 * Prompt prefill in chunks (the shared v41_prefill_chunk), then decode steps:
 * device argmax on the last logits row -> emit -> next token.  The engine's
 * generate path sets head_last_only so a prefill block computes ONE logits
 * row (the head GEMV on xlast, core_v41_forward.c:334-342).
 * Spec decode runs by default when the GGUF carries the draft towers and
 * parameters (core_v41_api.c:219-221): each round drafts a block, verifies
 * 1+k positions in one batch, accepts the longest matching prefix and rolls
 * the rest back.  Temperature-0 output is byte-identical with drafting on and
 * off — pure time saving.  verify_k pins the round's k (the engine's
 * --dspark-verify; the deterministic trace gate uses it), 0 = the confidence
 * scheduler.  Named limits: sampling and the penalties are not ported (their
 * unit); no_engram mirrors the engine's --v41-no-engram. */
int ds4_v41_generate_argmax(void *engine, const int *prompt, int n_prompt, int n_predict, int no_engram, int verify_k,
                            struct ds4_v41_images *imgs, const ds41_engram_feed *feed, int (*emit)(int token, void *ud), void *ud) {
    ds4_engine *e = (ds4_engine *)engine;
    if (!e || !prompt || n_prompt < 1 || DS4_MODEL_VARIANT != DS4_VARIANT_V41) return 1;
    const uint32_t np = (uint32_t)n_prompt;
    uint32_t ctx = g_ds4_v41.ctx;
    if (ctx == 0u) { fprintf(stderr, "ds4: [v41] model metadata has no deepseek4.context_length\n"); return 1; }
    if (np + 1u > ctx) { fprintf(stderr, "ds4: [v41] prompt %u tokens over context %u\n", np, ctx); return 1; }
    /* allocate only what this run can reach (core_v41_api.c:120-135) */
    const uint64_t need = (uint64_t)np + (uint64_t)(n_predict > 0 ? n_predict : 0) + DS4_MTP_MAX_BLOCK + 2u;
    if ((uint64_t)ctx > need) ctx = (uint32_t)need;
    const int room = ctx > np ? (int)(ctx - np) : 0;
    if (n_predict > room) n_predict = room;
    const uint32_t cap = v41_gen_cap(e, np);
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, ctx, DS4_MTP_MAX_BLOCK + 2u)) return 1;
    st.head_last_only = 1;
    st.no_engram = no_engram;   /* the gate's compare-only mode (the engine's --v41-no-engram) */
    st.efeed = feed;            /* rows: the feed's prepare callback runs per block (spec), or the caller pre-fills and sets pos0 */
    /* Diagnostic (kept): DS41_GEN_LOGITS=<path> appends the full logits row of
     * every emitted position (f32, DS4_N_VOCAB per row) so the step path can
     * be diffed against the score path on the same sequence. */
    const char *gen_lg = getenv("DS41_GEN_LOGITS");
    FILE *gen_fd = gen_lg ? fopen(gen_lg, "wb") : NULL;
    ds4_gpu_tensor *am = ds4_gpu_tensor_alloc((uint64_t)(DS4_MTP_MAX_BLOCK + 2u) * 16u);   /* per-row 16 B slots (sampling layout; argmax uses the first int) */
    /* Three routes (core_v41_api.c:139-167, executed by ds4_ds41_sample.inc):
     * temperature > 0 and no penalties = the device sample kernel (speculation
     * runs, the kernel does the rejection sampling); any penalty non-zero =
     * read the whole row back, penalize on the host and sample (penalties look
     * at the token history, speculation does not take them: an explicit
     * --dspark is refused by name, a default-on dspark turns this request into
     * pure decode, loudly -- a silent route change would leave the user
     * thinking speculation was on); otherwise the device argmax. */
    const bool penal = g_decode_sampling.dry_multiplier > 0.f || g_decode_sampling.freq_penalty != 0.f || g_decode_sampling.presence_penalty != 0.f;
    const bool dev_sample = g_decode_sampling.temperature > 0.f && !penal;
    int dspark = v41_dspark_mode();
    if (penal && dspark == 2) {
        fprintf(stderr, "ds4: *repetition penalties (dry %.2f freq %.2f presence %.2f) and --dspark speculation cannot be combined* "
                        "(penalties rewrite logits from the token history, the verify does not take them; drop --dspark or the penalties)\n",
                (double)g_decode_sampling.dry_multiplier, (double)g_decode_sampling.freq_penalty, (double)g_decode_sampling.presence_penalty);
        if (gen_fd) fclose(gen_fd);
        if (am) ds4_gpu_tensor_free(am);
        v41_state_free(&st);
        return 1;
    }
    if (penal && dspark) {
        fprintf(stderr, "ds4: [v41] this request turns on repetition penalties (dry %.2f freq %.2f presence %.2f), "
                        "speculation does not take penalties => this request runs pure decode\n",
                (double)g_decode_sampling.dry_multiplier, (double)g_decode_sampling.freq_penalty, (double)g_decode_sampling.presence_penalty);
        dspark = 0;
    }
    float *rowbuf = penal ? xmalloc((size_t)DS4_N_VOCAB * 4u) : NULL;
    v41_hist hist = {NULL, 0, 0, NULL};
    if (penal) {
        hist.cap = (uint32_t)(n_predict > 0 ? n_predict : 0) + 2u;
        hist.tok = xmalloc((size_t)hist.cap * sizeof(int32_t));
        if (g_decode_sampling.dry_multiplier > 0.f) hist.brk = ds4_decode_breakers(e, DS4_N_VOCAB);
    }
    uint64_t rng = g_decode_sampling.seed ? g_decode_sampling.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());   /* the V4 CLI's rule (core_v41_api.c:163-164) */
    const bool sampling = dev_sample || penal;   /* log only: this request is not bare argmax */
    st.dev_sample = dev_sample ? 1 : 0;
    st.samp = (ds4_gpu_sample_params){ .temperature = g_decode_sampling.temperature, .top_p = g_decode_sampling.top_p,
                                       .min_p = g_decode_sampling.min_p, .top_k = g_decode_sampling.top_k, .seed = rng };
    /* Spec decode arms when the GGUF carries the towers + draft parameters
     * (core_v41_api.c:219-221); ready=0 keeps the single-token loop.  The
     * three-way switch (--no-dspark / default / --dspark) is
     * ds4_engine_v41_set_dspark (ds41_forward.h); DS41_NO_DSPARK remains the
     * env fallback until a setter is called. */
    ds4_v41_draft dr;
    const bool spec = dspark && v41_draft_alloc(e, &dr);
    /* Sampling + speculation (core_v41_api.c:216-222): the draft draws from
     * the tower's own distribution (coin stream 1) and its logits become the
     * verify kernel's q for the rejection sampling (accept <=> u < min(1,p/q),
     * reject draws from max(0,p-q)).  The engine's accept-rate jury (the same
     * text parsed position by position, d1 accjury): first-position expected
     * accept 0.693 -> 0.744 (temperature 1.0, CFO), 0.634 -> 0.757 (the
     * broad set), 0.834 -> 0.842 (temperature 0.6, CFO) -- point mass
     * E[p(argmax q)] vs distribution draft E[sum min(p,q)].  Greedy is
     * untouched (temperature 0 stays argmax, the byte-for-byte gate). */
    if (spec && dev_sample) { dr.dev_sample = 1; dr.samp = st.samp; dr.samp.stream = 1u; st.spec_q = dr.st.logits; }
    v41_sched cal; memset(&cal, 0, sizeof cal);
    uint32_t spec_rounds = 0, spec_acc = 0, spec_off = 0, spec_skipped = 0, spec_hist[DS4_MTP_MAX_BLOCK + 1];
    g_v41_last_spec_rounds = g_v41_last_spec_off = g_v41_last_spec_acc = 0;   /* the monitor account clears per run (core_v41_api.c:42) */
    for (uint32_t i = 0; i <= DS4_MTP_MAX_BLOCK; i++) spec_hist[i] = 0;
    double ms_draft = 0, ms_snap = 0, ms_verify = 0, ms_argmax = 0;
    /* Host-gap account (core_v41_api.c:233-237, the engine's 2026-10-07 split):
     * h_vd = verify wait return -> the next draft graph's launch return
     * (accept/rollback/emit/slot write/launch); h_dv = draft step return ->
     * the verify graph's launch (pick_k/start/launch). */
    double h_vd = 0, h_dv = 0, t_prev_end = 0; uint32_t h_vdn = 0, h_dvn = 0;
    /* The next round's draft goes out at the END of a round, before the emit
     * (core_v41_api.c:237): the emit's fwrite+fflush then rides on the draft
     * graph's execution instead of stalling the GPU.  draft_r = the launch's
     * return (0 cannot draft / 1 in flight / 2 direct done). */
    bool draft_inflight = false; int draft_r = 0; double t_dl0 = 0;
    int rc = 1;
    int32_t tok = 0;
    int produced = 0;
    do {
        if (!am) break;
        /* --emit-trace prints the prompt too (`[ptok] <pos> <id>`): the prompt
         * is the chat template's render, whose re-tokenization cannot be
         * reconstructed from text afterwards (core_v41_api.c:171-173). */
        if (g_ds4_v41_emit_trace) for (uint32_t i = 0; i < np; i++) fprintf(stderr, "[ptok] %u %d\n", i, prompt[i]);
        const double t0 = now_sec();
        bool ok = true, aborted = false;
        /* The KV prefix reuse (core_v41_api.c:179): the fresh state takes the
         * previous request's prompt-end checkpoint when this prompt starts
         * with it, and the chunk loop then starts at that position. */
        const uint32_t reused = v41_kvreuse_restore(&st, prompt, np, imgs ? imgs->sig_of : NULL);
        /* Vision (core_v41_api.c:181): the prefill blocks overlay the visual
         * span embeddings by position (v41_forward_body); the tower runs once,
         * in the first block.  Cleared right after the prefill
         * (core_v41_api.c:193) so decode / draft clones never see it. */
        st.vimg = imgs;
        for (uint32_t c0 = reused; ok && c0 < np;) {
            ok = v41_prefill_chunk(e, &st, prompt, np, &c0, cap);
            /* The engine's per-chunk progress hook (core_v41_api.c:187-194):
             * nonzero = the caller wants out (the server's client-gone check),
             * stop before the remaining chunks. */
            if (ok && g_v41_progress &&
                g_v41_progress(g_v41_progress_ud, "prefill_chunk", (int)c0, (int)np) != 0) {
                fprintf(stderr, "[v41] prefill aborted by the caller at %u/%u\n", c0, np);
                aborted = true;
                break;
            }
        }
        st.vimg = NULL;
        if (!ok || aborted) break;
        /* Decode mutates the ring, the remainder rows and the group counts, so
         * the checkpoint is taken here (core_v41_api.c:192), after a clean
         * prefill and before the first token.  An identical prompt just
         * restored its own checkpoint and is not re-saved. */
        if (reused < np) v41_kvreuse_save(&st, prompt, np, imgs ? imgs->sig_of : NULL);
        const double t1 = now_sec();
        fprintf(stderr, "[v41] prefill %u token %.1fs (%.1f t/s)\n", np - reused, t1 - t0, (double)(np - reused) / (t1 - t0 + 1e-9));
        /* The sampling account (core_v41_api.c:192-196), one line per request. */
        if (sampling) fprintf(stderr, "[v41] decode sampling (%s) temp %.2f top_p %.2f min_p %.2f top_k %d seed %llu%s dry %.2f/%.2f/%d freq %.2f presence %.2f\n",
                              dev_sample ? "device kernel" : "host penalty route",
                              (double)g_decode_sampling.temperature, (double)g_decode_sampling.top_p, (double)g_decode_sampling.min_p,
                              g_decode_sampling.top_k, (unsigned long long)rng, penal ? " + penalties" : "", (double)g_decode_sampling.dry_multiplier,
                              (double)g_decode_sampling.dry_base, g_decode_sampling.dry_allowed_length,
                              (double)g_decode_sampling.freq_penalty, (double)g_decode_sampling.presence_penalty);
        if (getenv("DS41_ADDR_DUMP")) {   /* the A/B layout discriminator (see the harness) */
            fprintf(stderr, "addr am %p logits %p x %p xn %p tok %p pos %p\n", ds4_gpu_tensor_ptr(am), ds4_gpu_tensor_ptr(st.logits),
                    ds4_gpu_tensor_ptr(st.x), ds4_gpu_tensor_ptr(st.xn), ds4_gpu_tensor_ptr(st.tok), ds4_gpu_tensor_ptr(st.pos));
            for (uint32_t k = 0; k < g_ds4_v41.n_engram; k++)
                if (st.eraw[k]) fprintf(stderr, "addr eraw[%u] %p\n", k, ds4_gpu_tensor_ptr(st.eraw[k]));
            if (st.erows) fprintf(stderr, "addr erows %p ekv %p\n", ds4_gpu_tensor_ptr(st.erows), st.ekv ? ds4_gpu_tensor_ptr(st.ekv) : NULL);
        }
        /* the last prefill position's logits -> the first token (last_logit_row
         * says which row the head wrote; row 0 under head_last_only).  The
         * pick goes through v41_next_token (core_v41_api.c:198): rowbuf = the
         * host penalty route, else the device arm (sample kernel / argmax). */
        int32_t tok32 = 0;
        if (!v41_next_token(&st, am, st.last_logit_row, rowbuf, &rng, &hist, &tok32)) break;
        tok = (int)tok32;
        v41_gen_dump_row(&st, gen_fd);
        if (!rowbuf && !dev_sample) {   /* the engine's one-shot argmax self-check (core_v41_api.c:193-200):
             * a wrong kernel would silently emit wrong tokens, so compare the
             * device index against a host scan of the same row once.  The
             * sampling arm has no such reference (the kernel is the sampler). */
            float *row = xmalloc((size_t)DS4_N_VOCAB * 4);
            if (row && ds4_gpu_tensor_read(st.logits, (uint64_t)st.last_logit_row * DS4_N_VOCAB * 4, row, (uint64_t)DS4_N_VOCAB * 4)) {
                uint32_t best = 0;
                for (uint32_t v = 1; v < DS4_N_VOCAB; v++) if (row[v] > row[best]) best = v;
                if ((int32_t)best != tok) fprintf(stderr, "ds4: *[v41] argmax kernel %d != host %u*\n", tok, best);
            }
            free(row);
        }
        rc = 0;
        bool stop = false;
        while (!stop && produced < n_predict) {
            produced++;
            /* The engine's id-level instrument (core_v41_api.c:246): one line
             * per emitted token, absolute position + id, in emission order.
             * The position is the live state's n_past, never a counter -- a
             * rollback that pushed the state wrong shows as a position
             * mismatch here, which is why the engine prints it at all
             * (core_v41_api.c:244-245). */
            if (g_ds4_v41_emit_trace) fprintf(stderr, "[emit] %u %d\n", st.n_past, tok);
            const bool draft_now = spec;   /* speculation on: every step drafts (the scheduler picks k, may be 0) */
            /* Pure decode step on the graph (core_v41_api.c:257-267): no
             * speculation, or a round the scheduler sat out.  The first step
             * of a shape stays direct (it builds every lazy allocation;
             * capture must not allocate), then one launch per step.  Emit
             * rides on the graph: launch, emit, wait. */
            if (produced < n_predict && !draft_now && v41_graph_ready(&st) && tok != ds4_token_eos(e) && st.n_past + 1u <= st.ctx) {
                int32_t nt = 0;
                const double ts0 = now_sec();
                if (!v41_graph_launch(e, &st, (int32_t)tok, 0)) { rc = 1; break; }
                if (emit && emit((int)tok, ud) != 0) { (void)v41_graph_wait(e, &st, &nt); break; }
                if (!v41_graph_wait(e, &st, &nt)) { rc = 1; break; }
                if (spec) v41_sched_cost(&cal, 1u, (now_sec() - ts0) * 1e3);   /* one pure step's wall time: the scheduler's price basis */
                /* The penalty route (core_v41_api.c:264-266): the token in the
                 * graph's tail slot is not used; read this step's logits row
                 * (row 0, n=1) back and sample on the host after the
                 * penalties.  The device routes (sampling / argmax): nt is the
                 * slot's pick. */
                if (rowbuf && !v41_next_token(&st, am, 0, rowbuf, &rng, &hist, &nt)) { rc = 1; break; }
                tok = (int)nt;
                v41_gen_dump_row(&st, gen_fd);
                continue;
            }
            if (emit && emit((int)tok, ud) != 0) break;   /* the CLI's EOS convention: nonzero = stop */
            if (produced >= n_predict || (int)tok == ds4_token_eos(e)) { break; }
            if (st.n_past + 1u > st.ctx) { fprintf(stderr, "[v41] context full %u\n", st.ctx); break; }
            /* ---- one round (core_v41_api.c:272-390) ---- */
            uint32_t k = 0;
            int32_t batch[DS4_MTP_MAX_BLOCK + 1];
            batch[0] = tok;
            double tr0 = now_sec();
            float sched_val = 0.f;   /* the scheduler's predicted yield/cost (printed for k=0 rounds too) */
            bool drafted = false, have_draft = false;
            const uint32_t pos_round = st.n_past;   /* the round's first position (diagnostic) */
            if (draft_inflight) {
                /* The previous round's end already sent this draft (core_v41_api.c:279-283): wait only. */
                draft_inflight = false; tr0 = t_dl0;
                if (draft_r == 1 && !v41_draft_wait(&dr)) { rc = 1; break; }
                have_draft = draft_r != 0;
            } else if (draft_now) {
                dr.t_launched = 0.0;
                const int r = v41_draft_launch(e, &st, &dr, (int32_t)tok, st.n_past - 1u);
                if (r == 1 && t_prev_end > 0.0 && dr.t_launched > t_prev_end) { h_vd += dr.t_launched - t_prev_end; h_vdn++; }
                if (r == 1 && !v41_draft_wait(&dr)) { rc = 1; break; }
                have_draft = r != 0;
            }
            if (have_draft) {
                drafted = true;
                const uint32_t use = verify_k ? (uint32_t)verify_k
                                              : v41_draft_pick_k(dr.host_conf, dr.block, &sched_val, &cal);
                if (!verify_k && sched_val < 1.0f) spec_skipped++;   /* predicted not to repay even the draft */
                k = dr.block < use ? dr.block : use;
                /* Only positions inside the real context can be verified. */
                const uint32_t room_k = st.ctx - st.n_past - 1u;
                if (k > room_k) { k = room_k; }
                for (uint32_t i = 0; i < k; i++) batch[i + 1u] = dr.host_ids[i + 1u];
            }
            const uint32_t nb = 1u + k;
            const double tr1 = now_sec();
            double tr2 = tr1, tr3;
            int32_t want[DS4_MTP_MAX_BLOCK + 1];
            bool walked_graph = false;   /* this dispatch walked a graph (steady cost) or direct (warm-up / capture failure) */
            /* Verify batch on the graph (core_v41_api.c:310-323): once this n
             * warmed direct, snapshot + forward + n argmax + readback are one
             * launch; the snapshot bytes are inside the graph.  Otherwise the
             * direct route below (snapshot -> forward -> per-row argmax) --
             * the two routes are byte-identical. */
            if (k && v41_graph_batch_ready(&st, nb) && v41_graph_batch_launch(e, &st, batch, nb)) {
                h_dv += now_sec() - tr1; h_dvn++;
                if (!v41_graph_batch_wait(e, &st, want)) { rc = 1; break; }
                tr3 = now_sec(); walked_graph = true; t_prev_end = tr3;
            } else if (!k && v41_graph_ready(&st)) {
                /* A k=0 round (the draft ran, the scheduler sat it out): this
                 * step is a plain single-token step on the n=1 graph
                 * (core_v41_api.c:314-320; not in the steady account). */
                int32_t nt = 0;
                if (!v41_graph_launch(e, &st, (int32_t)tok, 1) || !v41_graph_wait(e, &st, &nt)) { rc = 1; break; }
                want[0] = nt;
                tr3 = now_sec(); walked_graph = true;
            } else {
                if (k && !v41_spec_snapshot(&st, nb)) { rc = 1; break; }
                tr2 = now_sec();
                if (!v41_forward(e, &st, batch, nb)) { rc = 1; break; }
                if (k && !ds4_gpu_synchronize()) { rc = 1; break; }   /* the split account wants real wall time */
                tr3 = now_sec();
                if (!v41_device_next(&st, am, 0u, nb, batch, want)) { rc = 1; break; }   /* sample kernel / per-row argmax, assembled into want[] */
            }
            /* The penalty route (core_v41_api.c:329-330): with penalties
             * speculation is off, so nb is always 1 and only want[0] matters;
             * the direct step (warm-up / capture-failure re-run) also takes the
             * host route here. */
            if (rowbuf && !v41_next_token(&st, am, 0, rowbuf, &rng, &hist, &want[0])) { rc = 1; break; }
            if (!k) v41_gen_dump_row(&st, gen_fd);   /* the DS41_GEN_LOGITS dump: the logits row behind a single-token decision */
            if (k) { ms_draft += (tr1 - tr0) * 1e3; ms_snap += (tr2 - tr1) * 1e3;
                     ms_verify += (tr3 - tr2) * 1e3; ms_argmax += (now_sec() - tr3) * 1e3; }
            if (drafted) {
                /* The scheduler records STEADY costs only (core_v41_api.c:335-339):
                 * a shape's first direct dispatch is its warm-up (lazy
                 * allocations, first touches); recording it would make that
                 * shape look expensive, the scheduler would avoid it and it
                 * would never collect graph samples -- self-fulfilling.
                 * Draft cost likewise excluded on a warm-up round. */
                const bool steady = walked_graph || !(k ? v41_graph_batch_ready(&st, nb) : v41_graph_ready(&st));
                if (!dr.last_warm) v41_sched_draft_cost(&cal, (tr1 - tr0) * 1e3);
                if (steady) v41_sched_cost(&cal, nb, (tr3 - tr2) * 1e3);
            }
            uint32_t a = 0;
            while (a < k && want[a] == batch[a + 1u]) a++;   /* accept the longest matching prefix */
            /* Observe the first position every drafted round (k=0 included), or
             * the calibration deadlocks at rho=0 (core_draft_sched.c:33). */
            if (drafted && !verify_k) v41_sched_observe(&cal, dr.host_conf, k ? (a >= 1u) : (want[0] == dr.host_ids[1]));
            /* The engine's per-round account (core_v41_api.c:346-357), the
             * spec trace gate's instrument. */
            if (drafted && (g_ds4_v41_prof || g_ds4_v41_emit_trace)) {
                fprintf(stderr, "[dspark] pos %u k %u val %.3f acc %u conf", pos_round, k, (double)sched_val, a);
                for (uint32_t i = 0; i < dr.block; i++) {
                    const float c = dr.host_conf[i];
                    fprintf(stderr, " %.3f", isfinite(c) ? 1.0 / (1.0 + exp(-(double)c)) : 0.0);
                }
                if (g_ds4_v41_prof) {   /* the draft vs main-model ids (core_v41_api.c:352-356) */
                    fprintf(stderr, " | draft");
                    for (uint32_t i = 0; i < k; i++) fprintf(stderr, " %d", batch[i + 1u]);
                    fprintf(stderr, " | main");
                    for (uint32_t i = 0; i < nb; i++) fprintf(stderr, " %d", want[i]);
                }
                fprintf(stderr, "\n");
            }
            if (k) {
                spec_rounds++; spec_acc += a; spec_off += k; spec_hist[a]++;
                const double tb0 = now_sec();
                if (!v41_spec_rollback(&st, 1u + a)) { rc = 1; break; }
                ms_snap += (now_sec() - tb0) * 1e3;   /* the rollback belongs to the snapshot/rollback term (core_v41_api.c:363-366) */
            }
            const int tok_next = (int)want[a];
            /* Send the next round's draft BEFORE the emit (core_v41_api.c:368-378):
             * the draft only needs the rolled-back state and tok_next, both
             * ready here; the emit then rides on the draft's execution instead
             * of stalling the GPU (nsys measured 1.26 ms idle there).  A
             * stopping round does not send.  A launch that returns 0 is marked
             * too: the round top must not retry it. */
            bool cont = spec && produced + a < (uint32_t)n_predict && tok_next != ds4_token_eos(e) && st.n_past + 1 <= st.ctx;
            for (uint32_t i = 0; cont && i < a; i++) if (batch[i + 1u] == ds4_token_eos(e)) cont = false;
            if (cont) {
                t_dl0 = now_sec(); dr.t_launched = 0.0;
                draft_r = v41_draft_launch(e, &st, &dr, (int32_t)tok_next, st.n_past - 1u);
                if (draft_r == 1 && t_prev_end > 0.0 && dr.t_launched > t_prev_end) { h_vd += dr.t_launched - t_prev_end; h_vdn++; }
                draft_inflight = true;
            }
            /* The earned tokens: the draft matched the main model, emit them
             * straight (core_v41_api.c:384-390; the n_predict cap holds). */
            for (uint32_t i = 0; i < a; i++) {
                if (produced >= n_predict) break;
                produced++;
                const int t = (int)batch[i + 1u];
                /* Position = the post-rollback n_past minus the not-yet-emitted
                 * accepted tokens (core_v41_api.c:387, the same absolute-
                 * position convention as the loop-top site). */
                if (g_ds4_v41_emit_trace) fprintf(stderr, "[emit] %u %d\n", st.n_past - a + i, t);
                if ((emit && emit(t, ud) != 0) || t == ds4_token_eos(e)) { stop = true; break; }
            }
            tok = tok_next;
        }
        if (draft_inflight && draft_r == 1) (void)v41_draft_wait(&dr);   /* the loop stopped with a draft in flight (emit stop / cap): drain before free */
        /* The engine's summary (core_v41_api.c:393-411). */
        if (spec_rounds) {
            g_v41_last_spec_rounds = (int)spec_rounds;
            g_v41_last_spec_off = (int)spec_off;
            g_v41_last_spec_acc = (int)spec_acc;
            fprintf(stderr, "\n[v41] DSpark: %u rounds (scheduler skipped %u), average accept %.2f/%u; histogram",
                    spec_rounds, spec_skipped, (double)spec_acc / (double)spec_rounds, dr.block);
            for (uint32_t i = 0; i <= dr.block; i++) fprintf(stderr, " %u:%u", i, spec_hist[i]);
            const double R = (double)spec_rounds, ea = 1.0 + (double)spec_acc / R;
            const double per = (ms_draft + ms_snap + ms_verify + ms_argmax) / R;
            fprintf(stderr, "\n[v41] one round %.1f ms = draft %.1f + snapshot/rollback %.1f + verify %.1f + argmax %.1f"
                            "; yield %.2f token => %.1f ms/token\n",
                    per, ms_draft / R, ms_snap / R, ms_verify / R, ms_argmax / R, ea, per / ea);
            {   /* printed whenever spec rounds ran (not prof-only: prof's per-stage syncs skew the gaps) */
                double bp = 0, bl = 0, pb = 0, pe = 0, pc = 0, pa = 0, tc = 0; uint32_t bn = 0, nc = 0;
                v41_graph_batch_host(&st, &bp, &bl, &bn); v41_graph_batch_prep(&st, &pb, &pe, &pc, &pa); v41_graph_capture_cost(&st, &tc, &nc);
                const double gs = dr.gsteps ? (double)dr.gsteps : 1.0, bd = bn ? (double)bn : 1.0;
                fprintf(stderr, "[v41] host gaps/round: verify end->draft graph out %.2f ms (slots %.3f + cudaGraphLaunch(draft, %u graph rounds) %.2f; rest = accept/rollback/emit)"
                                " | draft end->verify graph out %.2f ms (start %.2f = slots %.3f + engram prep %.3f + graph checks %.3f + arm %.3f; cudaGraphLaunch %.2f); samples %u/%u rounds"
                                " | capture (one-off, amortized above): verify graphs %u for %.0f ms, draft graphs %u for %.0f ms\n",
                        h_vdn ? h_vd / h_vdn * 1e3 : 0.0, dr.h_slots / gs * 1e3, dr.gsteps, dr.h_launch / gs * 1e3,
                        h_dvn ? h_dv / h_dvn * 1e3 : 0.0, bp / bd * 1e3, pb / bd * 1e3, pe / bd * 1e3, pc / bd * 1e3, pa / bd * 1e3, bl / bd * 1e3, h_dvn, h_dvn,
                        nc, tc * 1e3, dr.gcaps, dr.h_capture * 1e3);
            }
        }
    } while (0);
    free(rowbuf); free(hist.tok); free((void *)hist.brk);   /* the penalty route's buffers (core_v41_api.c:419) */
    if (gen_fd) fclose(gen_fd);
    if (spec) v41_draft_free(&dr);
    if (am) ds4_gpu_tensor_free(am);
    v41_state_free(&st);
    return rc;
}

#endif /* DS4_DS41_DRAFT_INC */
