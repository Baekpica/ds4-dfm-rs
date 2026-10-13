/* ds4_ds41_forward.inc — DeepSeek V4.1 (ds41) native forward, the eager
 * single-state path, ported from the C engine at /data/YoungAi (commit
 * 3946dbc):
 *   - the state allocation: src/core/core_v41_state.c (v41_state_alloc /
 *     v41_state_free / v41_index_scratch_prepare, :13-231);
 *   - the forward driver: src/core/core_v41_forward.c (v41_hc_half :24-35,
 *     v41_moe :74-157, v41_layer :182-232, v41_forward_body :299-355,
 *     v41_prefill_chunk :361-370, v41_forward :372-407);
 *   - the attention block: src/core/core_v41_attn.c (v41_rope :12-17,
 *     v41_compress_source :61-124, v41_index_source :126-190,
 *     v41_attention_kv_only :197-201, v41_tproj/v41_embed/v41_tproj_grouped
 *     :206-228, v41_attn_in :286-310, v41_attn_cache :312-349, v41_attn_out
 *     :351-364, v41_attention :366-371);
 *   - the engram layer: src/core/core_v41_engram.c (v41_engram_rows :405-446,
 *     v41_engram_apply :448-471, the direct path only);
 *   - the score entry: src/core/core_v41_score.c (:27-110).
 *
 * ★Scope and named deviations (P4-4)★ — the engine's graph path (captured
 * decode steps, device positions), the post-training store (tsave), the
 * DSpark draft towers, the zchain/amp plugins, spec decode's snapshot and
 * rollback, the MoE/attention side streams, sampling and the mainh ring are
 * NOT ported; every posd argument is NULL and every fork is 0.  The engine's
 * engram background pool + io_uring pread are host work that the port keeps
 * on the Rust side: the forward consumes a ds41_engram_feed (pinned raw rows
 * + host row ids) supplied by the caller.  The fp4x32 skeleton matmul arm is
 * not ported (the artifact is q4_K); it refuses by name.
 *
 * Placement contract: this file is included by ds4.c inside the GPU-only
 * region, after the weights bind, ds4_layer_weights and struct ds4_engine. */
#include "ds41_kvfmt.h"   /* DS4_V41_CKV_BYTES / DS4_V41_IDXK_BYTES, shared with the CUDA side */

/* --v41-chunk (core_v41.h:25): the prefill block size; 0 in the CLI = this. */
#define DS4_V41_CHUNK 2048u
/* ds4_gpu_v41.h:265: the largest token count the fused GEMV decode path
 * takes; the same value as V41_GEMV_MAX_TOK in the CUDA TU. */
#define DS4_V41_GEMV_MAX_TOK 8u

/* The CUDA TU owns the --v41-prof flag (ds4_ds41_gpu.cuh); this TU reads it. */
extern int g_ds4_v41_prof;
/* --decoder-full (core_v41_api.c): CED blocks also run the decoder segment.
 * The port has no CLI flag yet; the score path never sets ced_skip, so this
 * only matters to the generate-path prefill slicing kept below. */
static int g_ds4_v41_decoder_full = 0;

/* ---- P5: the serving switches (ds41_forward.h) -----------------------------
 * The engine's process globals, one setter each; the env fallbacks are read
 * once at first use so the E-gates' DS41_NO_DSPARK / DS41_NO_GRAPH keep
 * working until a setter overrides them. */
static int g_ds4_v41_dspark = -1;   /* core_v41_api.c:37: 0 off / 1 default on / 2 explicit --dspark */
void ds4_engine_v41_set_dspark(int mode) { g_ds4_v41_dspark = mode; }
static int v41_dspark_mode(void) {
    if (g_ds4_v41_dspark < 0) g_ds4_v41_dspark = getenv("DS41_NO_DSPARK") ? 0 : 1;
    return g_ds4_v41_dspark;
}
void ds4_engine_v41_set_emit_trace(int on) { g_ds4_v41_emit_trace = on; }
void ds4_engine_v41_set_prof(int on) { g_ds4_v41_prof = on; }

/* The engine's prefill progress hook (core_v41_api.c:187-194). */
static int (*g_v41_progress)(void *ud, const char *event, int current, int total);
static void *g_v41_progress_ud;
void ds4_engine_v41_set_progress(int (*cb)(void *ud, const char *event, int current, int total), void *ud) {
    g_v41_progress = cb;
    g_v41_progress_ud = ud;
}

static uint32_t v41_gen_cap(const ds4_engine *e, uint32_t np) {
    uint32_t cap = DS4_V41_CHUNK < np ? DS4_V41_CHUNK : np;
    /* A short prompt still needs the armed tower's full verification batch.
     * The main state and pinned feed share this bound; real context wins. */
    if (v41_dspark_mode() && g_ds4_v41.mtp_towers && g_ds4_v41.mtp_block && g_ds4_v41.n_mtp_target &&
        e->weights.v41_towers.main_proj && e->weights.v41_towers.markov_embd && e->weights.v41_towers.confidence) {
        const uint32_t width = 1u + g_ds4_v41.mtp_block;
        if (cap < width) { cap = width; }
    }
    if (cap > g_ds4_v41.ctx) { cap = g_ds4_v41.ctx; }
    return cap;
}

/* The Rust host's rows provider (ds41_forward.h): pinned raw buffers owned
 * here, the caller fills them through the trampoline.  The cap matches the
 * generate entry's prefill cap; a larger block is refused (a cap drift must
 * fail loud, not read past the buffer). */
struct ds4_v41_feed {
    ds41_engram_feed feed;
    ds4_v41_rows_fn fn;
    void *ud;
    ds4_v41_wait_fn wait_fn;
    void *wait_ud;
    ds4_v41_next_fn next_fn;
    void *next_ud;
    void *raw[DS4_V41_MAX_ENGRAM];
    uint32_t n_layers, rows_cap;
};
static int v41_feed_tramp(uint32_t pos0, const int *tokens, int n, void *ud) {
    ds4_v41_feed *f = (ds4_v41_feed *)ud;
    if (n < 0 || (uint32_t)n > f->rows_cap) {
        fprintf(stderr, "ds4: [v41] engram feed block %d over its capacity %u\n", n, f->rows_cap);
        return 1;
    }
    const int rc = f->fn ? f->fn(pos0, tokens, n, (uint8_t *const *)f->raw, f->ud) : 0;
    f->feed.pos0 = pos0;   /* declare the exact block; the forward asserts it at the point of use */
    return rc;
}
/* P2: the wait hook's trampoline — the provider collects layer k's round. */
static int v41_feed_wait_tramp(int k, void *ud) {
    ds4_v41_feed *f = (ds4_v41_feed *)ud;
    return f->wait_fn ? f->wait_fn(k, f->wait_ud) : 0;
}
/* The cross-block prefetch's trampoline (ds41_forward.h): the chunk loop
 * declares the next prefill block; the provider may read its first round
 * during the current block's tail. */
static int v41_feed_next_tramp(uint32_t pos0, const int *tokens, int n, void *ud) {
    ds4_v41_feed *f = (ds4_v41_feed *)ud;
    return f->next_fn ? f->next_fn(pos0, tokens, n, f->next_ud) : 0;
}
void ds4_v41_feed_close(ds4_v41_feed *f) {
    if (!f) return;
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) if (f->raw[k]) ds4_gpu_host_free(f->raw[k]);
    if (f->feed.flags) ds4_gpu_host_free(f->feed.flags);
    free(f);
}
ds4_v41_feed *ds4_v41_feed_open(void *engine, int n_prompt, ds4_v41_rows_fn fn, void *ud,
                                ds4_v41_wait_fn wait, void *wait_ud,
                                ds4_v41_next_fn next, void *next_ud) {
    if (!engine || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || g_ds4_v41.n_engram == 0) return NULL;
    const uint32_t cols = (g_ds4_v41.engram_max_ngram - 1u) * g_ds4_v41.engram_heads;
    const uint32_t stride = g_ds4_v41.engram_head_dim + g_ds4_v41.engram_head_dim / 32u;
    const uint32_t np = n_prompt > 0 ? (uint32_t)n_prompt : 1u;
    const uint32_t cap = v41_gen_cap((const ds4_engine *)engine, np);
    ds4_v41_feed *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    f->fn = fn; f->ud = ud; f->rows_cap = cap; f->n_layers = g_ds4_v41.n_engram;
    f->wait_fn = wait; f->wait_ud = wait_ud;
    f->next_fn = next; f->next_ud = next_ud;
    for (uint32_t k = 0; k < f->n_layers; k++) {
        f->raw[k] = ds4_gpu_host_alloc((uint64_t)cap * cols * stride);
        if (!f->raw[k]) { ds4_v41_feed_close(f); return NULL; }
        f->feed.raw[k] = f->raw[k];
    }
    f->feed.prepare = v41_feed_tramp;
    f->feed.ud = f;
    if (next) f->feed.prefetch_next = v41_feed_next_tramp;
    /* P2: the graph route's flag slots exist only for the async protocol
     * (the engine's J->flags, core_v41_engram.c:222-225). */
    if (wait) {
        f->feed.flags = ds4_gpu_host_alloc((uint64_t)(DS4_V41_MAX_ENGRAM + 2) * sizeof(int32_t));
        if (!f->feed.flags) { ds4_v41_feed_close(f); return NULL; }
        memset(f->feed.flags, 0, (DS4_V41_MAX_ENGRAM + 2) * sizeof(int32_t));
        f->feed.wait = v41_feed_wait_tramp;
    }
    return f;
}
const ds41_engram_feed *ds4_v41_feed_handle(const ds4_v41_feed *f) {
    return f ? &f->feed : NULL;
}

/* The engram row feed (ds41_forward.h): pinned raw bytes + host row ids per
 * engram layer.  In production the Rust host hashes and preads; the C harness
 * may pread the golden row ids directly. */
typedef ds41_engram_feed v41_engram_feed;

/* The eager single-state V4.1 state (core_v41.h:64-167, the inference
 * subset).  Dropped from the engine's struct: tsave, graph, snap_*, mainh/draft,
 * amp*, samp/spec_q, eg_* counters, eshard/ejob (host pread is outside),
 * n_direct*. */
/* One image's preprocessing + span (core_v41_vision.c / core_v41_api.h).  The
 * patches are host memory, freed after the tower forward; row0 is the first
 * embed row in the request's embd; sig is fnv1a64 over the image bytes (the KV
 * identity). */
/* The per-layer image-row router bias (uploaded by the vision file, included
 * later); the router in v41_moe reads it when the block carries image rows. */
extern ds4_gpu_tensor *g_v41_bias_vl[DS4_MAX_LAYER];

typedef struct {
    float   *patches;
    uint32_t n_vit_h, n_vit_w, n_llm_h, n_llm_w;
    uint32_t n_tok;              /* span length = n_llm_h*(n_llm_w+1)+2 */
    uint32_t row0, start;
    uint64_t sig;
} v41_image;

/* A request's whole image set (core_v41_api.h ds4_v41_images): the expanded
 * prompt's per-position row map/signature, and the span embeddings built by the
 * tower.  Attached to the prefill state and dropped after it. */
struct ds4_v41_images {
    uint32_t n;
    v41_image *img;
    uint32_t n_rows;             /* total embd rows */
    uint32_t np;                 /* expanded prompt positions */
    int32_t *row_of;             /* [np] embd row per position, -1 = text */
    uint64_t *sig_of;            /* [np] image signature per position, 0 = text */
    ds4_gpu_tensor *embd;        /* [n_rows][E] built span embeddings */
    int encoded;
};

/* Defined in ds4_ds41_vision.inc (included after this file): run the tower once
 * for the request and build the span embeddings into imgs->embd. */
bool v41_vision_encode(ds4_engine *e, struct ds4_v41_images *imgs);

typedef struct {
    uint32_t cap_tok, ctx;      /* max tokens per block / position capacity */
    uint32_t n_past;            /* positions already committed */
    uint32_t n, pos0;           /* this block: token count, first absolute position */
    ds4_gpu_tensor *tok, *pos, *posg;                    /* i32 [cap]; i32 [cap] new group positions g*ratio */
    int32_t *posg_pin[DS4_MAX_LAYER];   /* per kv-source layer pinned slot the zero-copy write uploads from */
    ds4_gpu_tensor *hc, *hc2;   /* [cap][hc][E] residual lanes, ping-pong */
    ds4_gpu_tensor *mix, *pre, *post, *comb, *pre_mix;   /* [cap][24] [cap][4] [cap][4] [cap][16] [cap][4] */
    ds4_gpu_tensor *x, *xn;     /* [cap][E] sublayer input / normalized */
    ds4_gpu_tensor *qr, *qrn, *q, *kv, *kvn;             /* [cap][1280] x2, [cap][64*512], [cap][512] x2 */
    ds4_gpu_tensor *ckv, *csc, *pooled, *latent, *ktmp;  /* [cap][512] x4, [cap][128] */
    ds4_gpu_tensor *win[DS4_MAX_LAYER], *wintmp;         /* window buffers / shift scratch [SWA][512] */
    ds4_gpu_tensor *comp_kv[DS4_MAX_LAYER], *index_k[DS4_MAX_LAYER];
    ds4_gpu_tensor *cpre_kv[DS4_MAX_LAYER], *cpre_sc[DS4_MAX_LAYER];
    uint32_t cpend[DS4_MAX_LAYER];    /* source layer's pending rows (= pos0 % ratio) */
    uint32_t ng_src[DS4_MAX_LAYER];   /* groups in the source layer's cache (incl. this block's new ones) */
    uint32_t win_from[DS4_MAX_LAYER]; /* first valid absolute position in the window ring (CED advances it) */
    ds4_gpu_tensor *iq, *iw, *iscore, *cand, *idx;   /* [cap][32*128], [cap][32], f32 [isrows][iscap], i32 [icrows][1+topk_blocks], i32 [cap][512] */
    uint32_t iscap, isrows, icrows;   /* scratch extents (grown by group count / row count, not ctx) */
    ds4_gpu_tensor *o, *low, *attn_out;                  /* [cap][64*512], [cap][8192], [cap][E] */
    ds4_gpu_tensor *glog, *sel, *rw, *routed;            /* [cap][384], i32 [cap][6], [cap][6], [cap][E] */
    ds4_gpu_tensor *sg, *su, *sh, *so, *y;               /* [cap][2304] x3, [cap][E] x2 */
    ds4_gpu_tensor *logits;     /* [logits_rows][V]: the score path opens cap rows */
    uint32_t logits_rows;
    uint32_t last_logit_row;
    /* Generate path (core_v41.h:104-110): prefill blocks compute ONE logits
     * row — the last position's normed input goes to xlast and the head runs
     * the decode-style GEMV on it; last_logit_row then points at row 0. */
    int head_last_only;
    ds4_gpu_tensor *xlast;      /* [1][E] the last position's normed head input */
    ds4_gpu_tensor *eraw[DS4_V41_MAX_ENGRAM], *erows, *ekv;   /* engram: raw row bytes [cap][cols][stride], decoded rows, wkv output */
    const v41_engram_feed *efeed;    /* caller-supplied rows (hash + pread are host work) */
    int egraph_uploaded;        /* capture pass: bitmask — layer ei's flag wait + upload are in the graph (core_v41.h:126) */
    int no_engram;              /* compare-only mode: skip the engram layers */
    /* Vision (2026-10-13): himg[ctx] flags image-span positions (the engram hash
     * lookback stops at them, official NgramHashState DEAD); vimg is the request's
     * image set (read only inside one prefill block; decode / draft clones always
     * see NULL); vrows[i] is this block's row index into vimg->embd (-1 = text),
     * vrows_on = this block has an image row (cleared at block end,
     * core_v41_forward.c:400-414). */
    uint8_t *himg;
    struct ds4_v41_images *vimg;
    ds4_gpu_tensor *vrows;
    int vrows_on;
    /* Draft towers (P5/E1, core_v41.h:64-167): the same struct doubles as the
     * tower state when draft is set — layer ids are tower indices, no engram,
     * no compressed KV/indexer, and the window holds main_x projections pushed
     * by v41_draft_push_main instead of the tower's own KV. */
    int draft;
    ds4_gpu_tensor *main_x;      /* [pcap][E] the current position's main-hidden projection (block attention KV source) */
    const ds4_tensor *tower_exps_vq[DS4_MTP_MAX_TOWERS];   /* blob form: the tower's VQ blob (main fused decode kernel) */
    const uint64_t *tower_exp_off[DS4_MTP_MAX_TOWERS];   /* per-expert form: [3][n_expert] offset table */
    /* The main-hidden ring (core_v41.h:144-151): [mainh_cap][n_target][E],
     * cell = position % cap; only the contiguous run ending at mainh_end is
     * real.  mainh_wrote is set by v41_layer, accounted by v41_forward. */
    ds4_gpu_tensor *mainh;
    uint32_t mainh_cap, mainh_wrote;
    int64_t mainh_end;
    uint32_t mainh_n;
    /* Spec verify snapshot (core_v41.h:83-85, core_v41_forward.c:241/267):
     * the ring cells a verify batch's commit overwrites (snap_win, per layer,
     * lazily allocated) and the compressor pending rows taken BEFORE the shift
     * (snap_cpre_*, allocated when snap_on is first seen). */
    ds4_gpu_tensor *snap_win[DS4_MAX_LAYER], *snap_cpre_kv[DS4_MAX_LAYER], *snap_cpre_sc[DS4_MAX_LAYER];
    uint32_t snap_cpend[DS4_MAX_LAYER], snap_past, snap_on;
    uint32_t snap_n;            /* rows in the snapshotted batch (1+k); the rollback restores [keep, snap_n) */
    /* Decode-step CUDA graph (core_v41.h:114-119, E3).  graph = 1 while the
     * step runs under the "device position" convention (position-dependent
     * kernels read st->pos; grid/shared sized by the bucket cap, never the
     * true position).  n_direct1 / n_direct_n count the direct warm-up
     * forwards a graph shape needs before capture (lazy allocations and
     * kernel attributes must all be built outside a capture).  iscap_gen
     * bumps whenever the index scratch moves: a captured graph bakes
     * pointers.  dgraph is the graph module's private state
     * (ds4_ds41_graph.inc; core_decode_graph.c:29-55). */
    int graph;
    uint32_t graph_pos_lo, graph_pos_cap;   /* the graph's valid position window [lo, cap] (bucket), last row cap+n-1 */
    uint32_t n_direct1;
    uint32_t n_direct_n[DS4_MTP_MAX_BLOCK + 2u];
    uint32_t iscap_gen;
    void *dgraph;
    /* Device sampling (2026-09-28, core_v41_sample.c; core_v41.h:122-124):
     * dev_sample = 1 makes the graph/direct tail use the sampling kernel
     * instead of argmax (temperature > 0 and no penalties); speculation runs
     * on this route too (the kernel does the rejection sampling).  The
     * penalty route still reads the whole row back and samples on the host,
     * dev_sample stays 0 there.  samp is the per-request face (constant,
     * baked into the captured graph).  spec_q = the draft tower's logits
     * [block][V] (row i = the draft distribution q for verify row i) while
     * speculating under sampling; NULL = point-mass draft.  The engine's
     * psamp (the batch path's per-request face, core_v41.h:123) has no port
     * counterpart: there is no batch path. */
    int dev_sample;
    ds4_gpu_sample_params samp;
    const ds4_gpu_tensor *spec_q;
    /* The block's host token ids, set by v41_forward: the engram feed's
     * prepare callback hashes the block's rows from them (the engine hashes
     * from st->hist, core_v41_forward.c:377). */
    const int32_t *blk_ids;
    int ced_done;               /* v41_forward_body: this block stopped at the CED edge (no logits) */
    int ced_skip;               /* this block runs only the encoder segment + the edge layer's KV */
    const char *dump_prefix;    /* non-NULL: per-layer MoE in/out dumps (single block, n <= 64) */
    uint32_t idx_topk;          /* topk width the latest indexer source produced */
    uint32_t idx_ratio;         /* compress ratio of the layer that produced it */
    int16_t idx_owner;          /* latest indexer source layer, -1 = none */
    int16_t cand_owner;         /* candidate-list producer layer, -1 = none */
} ds4_v41_state;

static ds4_gpu_tensor *v41_alloc(uint64_t bytes, bool *ok) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes ? bytes : 16);
    if (!t) *ok = false;
    return t;
}

/* core_v41_state.c:19-39 — dense row buffers (embed/hc/norm/MoE/head). */
static void v41_alloc_dense(ds4_v41_state *st, uint32_t cap, uint32_t logits_rows, bool *ok) {
    const uint64_t E = DS4_N_EMBD, HC = DS4_N_HC, FF = DS4_N_FF_EXP, NE = DS4_N_EXPERT, K = DS4_N_EXPERT_USED;
    const uint64_t mix = 2 * HC + HC * HC;
    st->tok = v41_alloc((uint64_t)cap * 4, ok);          st->pos = v41_alloc((uint64_t)cap * 4, ok);
    st->vrows = v41_alloc((uint64_t)cap * 4, ok);        /* vision: per-block row map (unread without images) */
    st->hc = v41_alloc((uint64_t)cap * HC * E * 4, ok);  st->hc2 = v41_alloc((uint64_t)cap * HC * E * 4, ok);
    st->mix = v41_alloc((uint64_t)cap * mix * 4, ok);    st->pre = v41_alloc((uint64_t)cap * HC * 4, ok);
    st->post = v41_alloc((uint64_t)cap * HC * 4, ok);    st->comb = v41_alloc((uint64_t)cap * HC * HC * 4, ok);
    st->pre_mix = v41_alloc((uint64_t)cap * HC * 4, ok);
    st->x = v41_alloc((uint64_t)cap * E * 4, ok);        st->xn = v41_alloc((uint64_t)cap * E * 4, ok);
    st->attn_out = v41_alloc((uint64_t)cap * E * 4, ok);
    st->glog = v41_alloc((uint64_t)cap * NE * 4, ok);    st->sel = v41_alloc((uint64_t)cap * K * 4, ok);
    st->rw = v41_alloc((uint64_t)cap * K * 4, ok);       st->routed = v41_alloc((uint64_t)cap * E * 4, ok);
    st->sg = v41_alloc((uint64_t)cap * FF * 4, ok);      st->su = v41_alloc((uint64_t)cap * FF * 4, ok);
    st->sh = v41_alloc((uint64_t)cap * FF * 4, ok);      st->so = v41_alloc((uint64_t)cap * E * 4, ok);
    st->y = v41_alloc((uint64_t)cap * E * 4, ok);
    /* logits only for the rows the caller reads (0 = every position, the score path) */
    st->logits_rows = (logits_rows && logits_rows < cap) ? logits_rows : cap;
    st->logits = v41_alloc((uint64_t)st->logits_rows * DS4_N_VOCAB * 4, ok);
    st->xlast = v41_alloc((uint64_t)E * 4, ok);   /* core_v41_state.c:37 */
}

/* core_v41_state.c:42-47 — attention projection rows (q path / kv / o / low). */
static void v41_alloc_attn_rows(ds4_v41_state *st, uint32_t cap, bool *ok) {
    const uint64_t HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    st->qr = v41_alloc((uint64_t)cap * Q * 4, ok);       st->qrn = v41_alloc((uint64_t)cap * Q * 4, ok);
    st->q = v41_alloc((uint64_t)cap * NH * HD * 4, ok);
    st->kv = v41_alloc((uint64_t)cap * HD * 4, ok);      st->kvn = v41_alloc((uint64_t)cap * HD * 4, ok);
    st->o = v41_alloc((uint64_t)cap * NH * HD * 4, ok);  st->low = v41_alloc((uint64_t)cap * low * 4, ok);
}

/* core_v41_state.c:49-57 — attention cache scratch (compressor / indexer). */
static void v41_alloc_attn_cache(ds4_v41_state *st, uint32_t cap, bool *ok) {
    const uint64_t HD = DS4_N_HEAD_DIM, IH = DS4_N_INDEXER_HEAD, IK = DS4_N_INDEXER_HEAD_DIM;
    st->posg = v41_alloc((uint64_t)cap * 4, ok);
    st->ckv = v41_alloc((uint64_t)cap * HD * 4, ok);     st->csc = v41_alloc((uint64_t)cap * HD * 4, ok);
    st->pooled = v41_alloc((uint64_t)cap * HD * 4, ok);  st->latent = v41_alloc((uint64_t)cap * HD * 4, ok);
    st->ktmp = v41_alloc((uint64_t)cap * IK * 4, ok);
    st->iq = v41_alloc((uint64_t)cap * IH * IK * 4, ok); st->iw = v41_alloc((uint64_t)cap * IH * 4, ok);
    st->idx = v41_alloc((uint64_t)cap * DS4_N_INDEXER_TOP_K * 4, ok);
}

static void v41_free_list(ds4_gpu_tensor ***list, size_t n) {
    for (size_t i = 0; i < n; i++) { if (*list[i]) ds4_gpu_tensor_free(*list[i]); *list[i] = NULL; }
}
#define V41_FREE(...) do { ds4_gpu_tensor **l_[] = { __VA_ARGS__ }; v41_free_list(l_, sizeof(l_) / sizeof(l_[0])); } while (0)
static void v41_free_dense(ds4_v41_state *st) {
    V41_FREE(&st->tok, &st->pos, &st->vrows, &st->hc, &st->hc2, &st->mix, &st->pre, &st->post, &st->comb, &st->pre_mix, &st->x, &st->xn, &st->attn_out,
             &st->glog, &st->sel, &st->rw, &st->routed, &st->sg, &st->su, &st->sh, &st->so, &st->y, &st->logits, &st->xlast);
}
static void v41_free_attn(ds4_v41_state *st) {
    V41_FREE(&st->posg, &st->qr, &st->qrn, &st->q, &st->kv, &st->kvn, &st->ckv, &st->csc, &st->pooled, &st->latent, &st->ktmp,
             &st->iq, &st->iw, &st->idx, &st->o, &st->low);
}

/* core_v41_state.c:92-135.  Windows are zeroed on allocation: the CED path
 * never writes the decoder segment's ring slots, so a recycled cudaMalloc
 * block would otherwise serve another request's KV as history (bug.md 6.2,
 * the win_from clamp is the fix, this is the safety net). */
static void v41_state_free(ds4_v41_state *st);
static void v41_graph_free(ds4_v41_state *st);   /* ds4_ds41_graph.inc (core_v41_state.c:233 calls it first, as here) */
static bool v41_state_alloc(ds4_v41_state *st, uint32_t cap, uint32_t ctx, uint32_t logits_rows) {
    memset(st, 0, sizeof *st);
    st->cap_tok = cap; st->ctx = ctx; st->idx_owner = -1; st->cand_owner = -1;
    if (cap == 0 || ctx < cap) { fprintf(stderr, "ds4: [v41] bad state params (cap %u ctx %u)\n", cap, ctx); return false; }
    if (ctx > g_ds4_v41.ctx) { fprintf(stderr, "ds4: [v41] context %u (model metadata deepseek4.context_length), state asked %u\n", g_ds4_v41.ctx, ctx); return false; }
    const uint64_t HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA;
    bool ok = true;
    v41_alloc_dense(st, cap, logits_rows, &ok);
    v41_alloc_attn_rows(st, cap, &ok);
    v41_alloc_attn_cache(st, cap, &ok);
    st->wintmp = v41_alloc(SWA * HD * 4, &ok);
    /* Vision: himg[ctx] flags image-span positions for the engram lookback
     * (core_v41_state.c:127).  Allocated unconditionally (1 byte/position). */
    st->himg = xmalloc_zeroed((size_t)ctx, 1);
    if (!st->himg) ok = false;
    /* core_v41_state.c:104-108: the main-hidden ring exists only when the
     * model carries draft towers (and its target layers are known). */
    if (g_ds4_v41.mtp_towers && g_ds4_v41.n_mtp_target) {
        st->mainh_cap = (uint32_t)SWA;
        st->mainh = v41_alloc((uint64_t)st->mainh_cap * g_ds4_v41.n_mtp_target * DS4_N_EMBD * 4, &ok);
        st->mainh_end = -1; st->mainh_n = 0;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        st->win[il] = v41_alloc((SWA + cap) * HD * 4, &ok);
        if (ok && !ds4_gpu_tensor_fill_f32(st->win[il], 0.f, (uint64_t)(SWA + cap) * HD)) ok = false;
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) { fprintf(stderr, "ds4: [v41] kv source layer L%u has ratio 0\n", il); ok = false; break; }
        /* two extra groups: the graph path's trash slot lives at ctx/ratio; the
         * eager path never writes it, but the allocation formula stays the
         * engine's one copy (core_v41_state.c:119-121). */
        const uint64_t ngcap = (uint64_t)ctx / ratio + 2;
        st->comp_kv[il] = v41_alloc(ngcap * DS4_V41_CKV_BYTES, &ok);
        st->index_k[il] = v41_alloc(ngcap * DS4_V41_IDXK_BYTES, &ok);
        st->posg_pin[il] = ds4_gpu_host_alloc((uint64_t)cap * 4);
        if (!st->posg_pin[il]) ok = false;
        if (ratio > 1) {
            st->cpre_kv[il] = v41_alloc((ratio + (uint64_t)cap) * HD * 4, &ok);
            st->cpre_sc[il] = v41_alloc((ratio + (uint64_t)cap) * HD * 4, &ok);
        }
    }
    if (!ok) { fprintf(stderr, "ds4: [v41] state buffer allocation failed (cap=%u ctx=%u)\n", cap, ctx); v41_state_free(st); }
    return ok;
}

/* core_v41_state.c:233-247, minus the engine's amp/job teardown. */
static void v41_state_free(ds4_v41_state *st) {
    v41_graph_free(st);
    v41_free_dense(st);
    v41_free_attn(st);
    V41_FREE(&st->wintmp, &st->iscore, &st->cand, &st->erows, &st->ekv, &st->mainh, &st->main_x);
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) V41_FREE(&st->eraw[k]);
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        V41_FREE(&st->win[il], &st->comp_kv[il], &st->index_k[il], &st->cpre_kv[il], &st->cpre_sc[il]);
        V41_FREE(&st->snap_win[il], &st->snap_cpre_kv[il], &st->snap_cpre_sc[il]);
        if (st->posg_pin[il]) { ds4_gpu_host_free(st->posg_pin[il]); st->posg_pin[il] = NULL; }
    }
    free(st->himg); st->himg = NULL;
}

/* core_v41_state.c:195-229, minus the graph-capture branch: the score scratch
 * grows to what this pass actually touches (group count x row count), never
 * to ctx — the 1M-context scratch illusion the engine documents there. */
static bool v41_index_scratch_prepare(ds4_v41_state *st, uint32_t ng_need, uint32_t rows_score, uint32_t rows_cand) {
    const uint32_t kcap = g_ds4_v41.candidate_source_layer >= 0 && g_ds4_v41.candidate_topk_blocks > 0 ? (uint32_t)g_ds4_v41.candidate_topk_blocks : 0u;
    if (ng_need <= st->iscap && rows_score <= st->isrows && rows_cand <= st->icrows && st->iscore && (st->cand || !kcap)) return true;
    /* The graph route grows this before capture (dg_capture, core_v41_state.c:199-204);
     * meeting a growth need here means the capture sized it short. */
    if (st->graph) {
        fprintf(stderr, "ds4: [v41] index scratch growth asked under capture (needs %u/%u rows x %u groups, has %u/%u x %u)\n",
                rows_score, rows_cand, ng_need, st->isrows, st->icrows, st->iscap);
        return false;
    }
    uint32_t want = st->iscap ? st->iscap * 2u : ng_need;
    if (want < ng_need) want = ng_need;
    if (want > st->ctx) want = st->ctx;
    const uint32_t dec_rows = DS4_MTP_MAX_BLOCK + 2u;
    uint32_t rs = rows_score <= dec_rows ? dec_rows : rows_score, rc = rows_cand <= dec_rows ? dec_rows : rows_cand;
    if (rs > st->cap_tok) rs = st->cap_tok < rows_score ? rows_score : st->cap_tok;
    if (rc > st->cap_tok) rc = st->cap_tok < rows_cand ? rows_cand : st->cap_tok;
    bool ok = true;
    ds4_gpu_tensor *is = v41_alloc((uint64_t)rs * want * 4, &ok);
    ds4_gpu_tensor *cd = kcap ? v41_alloc((uint64_t)rc * (1u + kcap) * 4, &ok) : NULL;
    if (!ok) {
        if (is) ds4_gpu_tensor_free(is);
        if (cd) ds4_gpu_tensor_free(cd);
        fprintf(stderr, "ds4: [v41] index scratch allocation failed (%u groups x %u/%u rows)\n", want, rs, rc);
        return false;
    }
    if (st->iscore) ds4_gpu_tensor_free(st->iscore);
    if (st->cand) ds4_gpu_tensor_free(st->cand);
    st->iscore = is; st->cand = cd; st->iscap = want; st->isrows = rs; st->icrows = rc;
    st->iscap_gen++;   /* pointers moved: every captured graph baked the old ones (dg_launch checks) */
    return true;
}

/* ---- dispatchers (core_v41_attn.c:206-228) ----
 * Skeleton tensors are q4_K or fp8_32x32 in this artifact; the engine's
 * third arm (fp4x32, old GGUFs) is not ported and refuses by name. */
static bool v41_tproj(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
                      uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n, int round_out) {
    if (w->type == DS4_TENSOR_FP8_32X32)
        return ds4_gpu_v41_matmul_fp8blk_round_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n, round_out) != 0;
    if (w->type == DS4_TENSOR_Q4_K)
        return ds4_gpu_v41_matmul_q4k_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n, round_out) != 0;
    fprintf(stderr, "ds4: [v41] skeleton matmul: fp4x32 weights are not ported yet (P4-5); this artifact is q4_K\n");
    return false;
}
static bool v41_embed(const ds4_model *m, ds4_gpu_tensor *out, const ds4_gpu_tensor *tok,
                      const ds4_tensor *w, uint64_t n_vocab, uint32_t n, uint64_t dim) {
    if (w->type == DS4_TENSOR_Q4_K)
        return ds4_gpu_v41_embed_q4k_tensor(out, tok, m->map, m->size, w->abs_offset, n_vocab, n, dim) != 0;
    return ds4_gpu_v41_embed_fp4x32_tensor(out, tok, m->map, m->size, w->abs_offset, (uint32_t)n_vocab, n, (uint32_t)dim) != 0;
}
static bool v41_tproj_grouped(const ds4_model *m, ds4_gpu_tensor *low, const ds4_tensor *w, uint32_t n_groups,
                              uint64_t group_dim, uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n, int round_out) {
    if (w->type == DS4_TENSOR_FP8_32X32)
        return ds4_gpu_v41_grouped_matmul_fp8blk_tensor(low, m->map, m->size, w->abs_offset, n_groups, group_dim, rank, heads, n, round_out) != 0;
    if (w->type == DS4_TENSOR_Q4_K)
        return ds4_gpu_v41_grouped_matmul_q4k_tensor(low, m->map, m->size, w->abs_offset, n_groups, group_dim, rank, heads, n, round_out) != 0;
    fprintf(stderr, "ds4: [v41] grouped skeleton matmul: fp4x32 weights are not ported yet (P4-5); this artifact is q4_K\n");
    return false;
}

/* RoPE per layer kind (core_v41_attn.c:10-17): compressor layers use
 * theta 160000 + YaRN (original 65536), window layers theta 10000, no YaRN. */
static float v41_theta(uint32_t ratio) { return ratio ? DS4_COMPRESS_ROPE_FREQ_BASE : DS4_ROPE_FREQ_BASE; }
static uint32_t v41_osl(uint32_t ratio) { return ratio ? (uint32_t)DS4_ROPE_ORIG_CTX : 0u; }
static bool v41_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t rows, uint32_t n_head, uint32_t hd, uint32_t ratio, bool inverse) {
    return ds4_gpu_v41_rope_tensor(x, pos, rows, n_head, hd, DS4_N_ROT, v41_theta(ratio), v41_osl(ratio), DS4_ROPE_SCALE_FACTOR,
                                   DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, inverse) != 0;
}

/* ---- engram (core_v41_engram.c:405-471, the direct path) ----
 * The hash + pread are host work outside this file; here the rows are already
 * in the caller's pinned feed: upload -> decode rows -> wkv (fp8) -> gate. */
static bool v41_engram_rows(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    (void)e;
    const ds4_v41_cfg *v = &g_ds4_v41;
    const int16_t ei = v->engram_index_of[il];
    if (ei < 0) return true;
    const uint32_t n = st->n;
    const uint32_t HD = v->engram_head_dim, cols = (v->engram_max_ngram - 1u) * v->engram_heads;
    const uint32_t stride = HD + HD / 32u;
    const uint64_t in_dim = (uint64_t)cols * HD, out_dim = (uint64_t)(DS4_N_HC + 1u) * DS4_N_EMBD;
    if (!st->erows) {   /* three buffers sized to cap once; addresses stay stable for the session */
        /* No allocation under capture (ThreadLocal voids it): the direct warm
         * step builds these before any graph captures. */
        if (st->graph) { fprintf(stderr, "ds4: [v41] engram buffers not built before capture (a direct step must warm first)\n"); return false; }
        for (uint32_t k = 0; k < v->n_engram; k++) st->eraw[k] = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * cols * stride);
        st->erows = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * in_dim * 4);
        st->ekv = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * out_dim * 4);
        if (!st->erows || !st->ekv) return false;
    }
    for (uint32_t k = 0; k < v->n_engram; k++) if (!st->eraw[k]) return false;
    const v41_engram_feed *fd = st->efeed;
    if (!fd || !fd->raw[ei]) { fprintf(stderr, "ds4: [v41] engram layer L%u has no row feed (host hash + pread missing)\n", il); return false; }
    /* The feed declares which block its slot 0 holds; the state is the
     * authority (the engine hashes rows from st->hist at st->pos0, never from
     * a caller counter).  A drifted declaration must refuse here — the E0
     * class fed rows for shifted positions and no device instrument could see
     * it.  The caller refills slot 0 per decode step and updates pos0 with it;
     * a multi-chunk score refills per chunk. */
    if (fd->pos0 != st->pos0) {
        fprintf(stderr, "ds4: [v41] engram feed pos0 %u != block pos0 %u (rows must be for the block the forward is about to run)\n", fd->pos0, st->pos0);
        return false;
    }
    if (st->graph) {
        /* ★capture★ (core_v41_engram.c:433-446): record the layer's flag-wait
         * spin kernel then the zero-copy upload, once per capture pass; the
         * host must not wait here (a host wait or a sync copy would void the
         * capture).  No wait hook = the synchronous protocol: the upload
         * alone is recorded, exactly the old graph. */
        if (!(st->egraph_uploaded & (1 << ei))) {
            if (fd->wait && fd->flags &&
                !ds4_gpu_host_flag_wait(&fd->flags[ei], &fd->flags[v->n_engram], &fd->flags[v->n_engram + 1])) return false;
            if (!ds4_gpu_tensor_write_zerocopy(st->eraw[ei], 0, fd->raw[ei], (uint64_t)n * cols * stride)) return false;
            st->egraph_uploaded |= 1 << ei;
        }
    } else {
        /* The engine's v41_ejob_wait (core_v41_engram.c:441): this layer's
         * rows must be on the host before the upload — waiting on the disk,
         * never on the GPU. */
        if (fd->wait && fd->wait((int)ei, fd->ud) != 0) {
            fprintf(stderr, "ds4: [v41] engram layer L%u rows not ready (feed wait failed)\n", il);
            return false;
        }
        if (!ds4_gpu_tensor_write_zerocopy(st->eraw[ei], 0, fd->raw[ei], (uint64_t)n * cols * stride)) return false;
    }
    /* Diagnostic (kept): DS41_FEED_DIGEST=<path> appends the FNV-1a of the
     * exact bytes this upload put on the device.  The row-id self-check
     * cannot see a payload error (right ids, wrong bytes); the caller prints
     * the host-side digest of the same range and the two must match. */
    {
        const char *fdg = st->graph ? NULL : getenv("DS41_FEED_DIGEST");   /* a device readback would void a capture; the predicate excludes the env in graph mode anyway */
        if (fdg) {
            static FILE *dfp = NULL;
            const uint64_t bytes = (uint64_t)n * cols * stride;
            uint8_t *tmp = (uint8_t *)xmalloc(bytes);
            if (!dfp) dfp = fopen(fdg, "a");
            if (dfp && tmp && ds4_gpu_tensor_read(st->eraw[ei], 0, tmp, bytes)) {
                uint64_t h = 0xcbf29ce484222325ull;
                for (uint64_t q = 0; q < bytes; q++) { h ^= tmp[q]; h *= 0x100000001b3ull; }
                fprintf(dfp, "dev pos0 %u n %u k %u fnv %016llx\n", st->pos0, n, ei, (unsigned long long)h);
                fflush(dfp);
            }
            free(tmp);
        }
    }
    if (st->dump_prefix && fd->rows[ei]) {   /* compare fixture: row ids to <prefix>.erows_Lnn.txt */
        char p[4400]; snprintf(p, sizeof p, "%s.erows_L%02u.txt", st->dump_prefix, il);
        FILE *df = fopen(p, "w");
        if (df) { for (uint32_t q = 0; q < n; q++) for (uint32_t c = 0; c < cols; c++) fprintf(df, "%lld%c", (long long)fd->rows[ei][(size_t)q * cols + c], c + 1 == cols ? '\n' : ' '); fclose(df); }
    }
    if (!ds4_gpu_v41_engram_rows_tensor(st->erows, st->eraw[ei], n * cols, HD)) return false;
    /* Image positions get no engram contribution (official engram_mask;
     * core_v41_engram.c:460): zeroing the rows makes wkv 0, so the gate adds 0. */
    if (st->vrows_on) return ds4_gpu_v41_rows_zero_tensor(st->erows, st->vrows, n, (uint32_t)in_dim) != 0;
    return true;
}

static bool v41_engram_apply(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const int16_t ei = v->engram_index_of[il];
    if (ei < 0) return true;
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HC = DS4_N_HC;
    const uint64_t in_dim = (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads * v->engram_head_dim;
    const uint64_t out_dim = (uint64_t)(HC + 1) * E;
    if (!st->ekv) st->ekv = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * out_dim * 4);
    if (!st->erows || !st->ekv) return false;
    /* wkv is official FP8 on disk (e4m3 + 32x32 block scales) -> bf16 -> gate -> hc in place */
    if (!ds4_gpu_v41_matmul_fp8blk_tensor(st->ekv, m->map, m->size, l->engram_wkv->abs_offset, in_dim, out_dim, st->erows, n)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->ekv, (uint64_t)n * out_dim)) return false;
    if (!ds4_gpu_v41_engram_gate_tensor(st->hc, st->ekv, m->map, m->size, l->engram_q->abs_offset, l->engram_k->abs_offset, E, HC, n, DS4_RMS_EPS)) return false;
    if (st->dump_prefix) {   /* compare fixture: hc after the engram to <prefix>.hce_Lnn.bin */
        char p[4400];
        float *buf = xmalloc((size_t)n * HC * E * 4);
        ds4_gpu_synchronize();
        snprintf(p, sizeof p, "%s.hce_L%02u.bin", st->dump_prefix, il);
        if (ds4_gpu_tensor_read(st->hc, 0, buf, (uint64_t)n * HC * E * 4)) { FILE *f = fopen(p, "wb"); if (f) { fwrite(buf, 4, (size_t)n * HC * E, f); fclose(f); } }
        free(buf);
    }
    return true;
}

static bool v41_engram(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    return v41_engram_rows(e, st, il) && v41_engram_apply(e, st, il);
}

/* ---- the compressor source layer (core_v41_attn.c:61-124, direct path) ----
 * New compressor inputs append after the pending rows; completed groups pool
 * (ratio > 1) or project (ratio 1), then the shared chain runs: latent norm
 * -> wk -> k_norm -> rope -> fp4 pack into index_k; latent -> rope -> fp4
 * pack into comp_kv.  The spec snapshot point is BEFORE the destructive
 * remainder-to-head shift. */
/* Graph route (core_v41_attn.c:20-58): the append/pool step is one
 * position-driven kernel (ds4_gpu_v41_compress_step_n_tensor) and the
 * downstream chain runs UNCONDITIONALLY every step -- the pack kernels
 * compute the true group number from the device position, and rows of an
 * incomplete group land in the trash slot past the group cap (never read).
 * A host "is the group complete" branch would have to become "issue every
 * step, decide on device" anyway: one capture, hundreds of replays.  The
 * host counters are NOT advanced here -- the graph's launch advances them in
 * closed form (dg_advance). */
static bool v41_compress_source_graph(ds4_engine *e, ds4_v41_state *st, uint32_t il, uint32_t ratio) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, IK = DS4_N_INDEXER_HEAD_DIM;
    const uint64_t rowb = (uint64_t)HD * 4;
    const uint32_t g_trash = st->ctx / ratio;   /* same source as the ngcap = ctx/ratio + 2 allocation: this cell is past the group cap */
    const ds4_gpu_tensor *pg = st->posg;        /* new group positions: ratio>1 written by the append kernel; ratio 1 the position itself */
    uint32_t np;
    if (ratio > 1) {
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->ckv, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->csc, m->map, m->size, l->attn_compressor_gate->abs_offset, E, HD, st->xn, n)) return false;
        /* The snapshot is only for the verify batch (n>1), and its buffers are
         * built by the direct round (no allocation under capture). */
        ds4_gpu_tensor *sk = n > 1u ? st->snap_cpre_kv[il] : NULL, *ss = n > 1u ? st->snap_cpre_sc[il] : NULL;
        if (n > 1u && (!sk || !ss)) { fprintf(stderr, "ds4: [graph] L%u compressor snapshot buffer not built (a verify batch must warm direct first)\n", il); return false; }
        if (!ds4_gpu_v41_compress_step_n_tensor(st->pooled, st->posg, st->cpre_kv[il], st->cpre_sc[il], sk, ss, st->ckv, st->csc, st->pos, ratio, HD, n)) return false;
        np = (ratio - 1u + n) / ratio;
    } else {           /* ratio 1: pure projection (bf16 weights) -> bf16 */
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->pooled, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_round_bf16_tensor(st->pooled, (uint64_t)n * HD)) return false;
        pg = st->pos; np = n;
    }
    if (!ds4_gpu_v41_rms_norm_tensor(st->latent, st->pooled, m->map, m->size, l->attn_compressor_norm->abs_offset, HD, np, DS4_RMS_EPS)) return false;
    if (!ds4_gpu_v41_matmul_bf16_tensor(st->ktmp, m->map, m->size, l->indexer_wk->abs_offset, HD, IK, st->latent, np)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->ktmp, (uint64_t)np * IK)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->ckv, st->ktmp, m->map, m->size, l->indexer_k_norm->abs_offset, IK, np, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->ckv, pg, np, 1, IK, ratio, false)) return false;
    if (!ds4_gpu_v41_idxk_pack_tensor(st->index_k[il], 0, st->ckv, np, st->pos, ratio, g_trash, n)) return false;
    if (!ds4_gpu_tensor_copy(st->pooled, 0, st->latent, 0, (uint64_t)np * rowb)) return false;
    if (!v41_rope(st->pooled, pg, np, 1, HD, ratio, false)) return false;
    return ds4_gpu_v41_ckv_pack_tensor(st->comp_kv[il], 0, st->pooled, np, st->pos, ratio, g_trash, n) != 0;
}
static bool v41_compress_source(ds4_engine *e, ds4_v41_state *st, uint32_t il, uint32_t ratio) {
    if (st->graph) return v41_compress_source_graph(e, st, il, ratio);
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, IK = DS4_N_INDEXER_HEAD_DIM;
    const uint64_t rowb = (uint64_t)HD * 4;
    const uint32_t g0 = st->pos0 / ratio;     /* groups already in the cache */
    uint32_t ng_new;
    if (ratio > 1) {   /* official Compressor ratio>1: in-group softmax pool -> bf16 */
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->ckv, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->csc, m->map, m->size, l->attn_compressor_gate->abs_offset, E, HD, st->xn, n)) return false;
        const uint32_t pend = st->cpend[il];
        if (!ds4_gpu_tensor_copy(st->cpre_kv[il], (uint64_t)pend * rowb, st->ckv, 0, (uint64_t)n * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->cpre_sc[il], (uint64_t)pend * rowb, st->csc, 0, (uint64_t)n * rowb)) return false;
        const uint32_t tot = pend + n, rem = tot % ratio;
        ng_new = tot / ratio;
        /* Spec rollback point (core_v41_attn.c:76-85, speed.md 6 D1): the
         * remainder-to-the-head move below is destructive — after it the
         * batch's compressor inputs are gone.  Save the whole block first;
         * the rollback takes the accepted prefix's rows back out of it. */
        if (st->snap_on) {
            bool sok = true;
            const uint64_t nb = ((uint64_t)ratio + st->cap_tok) * rowb;   /* same formula as the cpre_* allocation */
            if (!st->snap_cpre_kv[il]) { st->snap_cpre_kv[il] = v41_alloc(nb, &sok); st->snap_cpre_sc[il] = v41_alloc(nb, &sok); }
            if (!sok) return false;
            if (!ds4_gpu_tensor_copy(st->snap_cpre_kv[il], 0, st->cpre_kv[il], 0, (uint64_t)tot * rowb)) return false;
            if (!ds4_gpu_tensor_copy(st->snap_cpre_sc[il], 0, st->cpre_sc[il], 0, (uint64_t)tot * rowb)) return false;
            st->snap_cpend[il] = pend;
        }
        if (ng_new) {
            if (!ds4_gpu_v41_compress_pool_tensor(st->pooled, st->cpre_kv[il], st->cpre_sc[il], tot, ratio, HD)) return false;
            if (rem) {   /* move the remainder to the head (source rows >= ratio > rem, no overlap) */
                if (!ds4_gpu_tensor_copy(st->cpre_kv[il], 0, st->cpre_kv[il], (uint64_t)ng_new * ratio * rowb, (uint64_t)rem * rowb)) return false;
                if (!ds4_gpu_tensor_copy(st->cpre_sc[il], 0, st->cpre_sc[il], (uint64_t)ng_new * ratio * rowb, (uint64_t)rem * rowb)) return false;
            }
        }
        st->cpend[il] = rem;
    } else {           /* ratio 1: pure projection (bf16 weights) -> bf16 */
        if (!ds4_gpu_v41_matmul_bf16_tensor(st->pooled, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, st->xn, n)) return false;
        if (!ds4_gpu_v41_round_bf16_tensor(st->pooled, (uint64_t)n * HD)) return false;
        ng_new = n;
    }
    st->ng_src[il] = g0 + ng_new;
    if (!ng_new) return true;
    {   /* new group positions g*ratio through the pinned slot (a sync memcpy would serialize concurrent lanes) */
        int32_t *pg = st->posg_pin[il];
        if (!pg) return false;
        for (uint32_t g = 0; g < ng_new; g++) pg[g] = (int32_t)((g0 + g) * ratio);
        if (!ds4_gpu_tensor_write_zerocopy(st->posg, 0, pg, (uint64_t)ng_new * 4)) return false;
    }
    if (!ds4_gpu_v41_rms_norm_tensor(st->latent, st->pooled, m->map, m->size, l->attn_compressor_norm->abs_offset, HD, ng_new, DS4_RMS_EPS)) return false;
    /* indexer key: k = k_norm(wk(latent)) -> rope(group pos) -> fp4(ue8m0/32); ckv borrows as the [ng_new][IK] output */
    if (!ds4_gpu_v41_matmul_bf16_tensor(st->ktmp, m->map, m->size, l->indexer_wk->abs_offset, HD, IK, st->latent, ng_new)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->ktmp, (uint64_t)ng_new * IK)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->ckv, st->ktmp, m->map, m->size, l->indexer_k_norm->abs_offset, IK, ng_new, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->ckv, st->posg, ng_new, 1, IK, ratio, false)) return false;
    /* quantize + pack in one launch into the 72 B/group cache (the packed bytes ARE the quantized values) */
    if (!ds4_gpu_v41_idxk_pack_tensor(st->index_k[il], g0, st->ckv, ng_new, NULL, 0, 0, 0)) return false;
    /* compressed KV: latent -> rope(group pos) -> fp4(e4m3 scale/16) -> 288 B/group cache */
    if (!ds4_gpu_tensor_copy(st->pooled, 0, st->latent, 0, (uint64_t)ng_new * rowb)) return false;
    if (!v41_rope(st->pooled, st->posg, ng_new, 1, HD, ratio, false)) return false;
    return ds4_gpu_v41_ckv_pack_tensor(st->comp_kv[il], g0, st->pooled, ng_new, NULL, 0, 0, 0) != 0;
}

/* ---- the indexer source layer (core_v41_attn.c:126-190, direct path) ----
 * q = wq_b(qr_norm) -> rope -> fp4; weights = proj(x) * softmax_scale scaled
 * on bf16; score the source layer's whole key cache -> [candidate blocks] ->
 * topk.  The row-block loop (C1) degenerates to one block at n <= 10. */
static bool v41_index_source(ds4_engine *e, ds4_v41_state *st, uint32_t il, uint32_t ratio) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t n = st->n, E = DS4_N_EMBD, IH = DS4_N_INDEXER_HEAD, IK = DS4_N_INDEXER_HEAD_DIM;
    const int16_t src = v->kv_source_of[il];
    if (src < 0 || !st->index_k[src]) return false;
    /* Graph route: ng/topk are the bucket CAPS and the position comes from the
     * device slot (ds4_gpu_v41.h "device position"); the "no completed groups
     * yet" case is handled by the kernels themselves (device ng = 0: score and
     * topk run empty, attention reads the window only) -- the host does not
     * branch.  Direct route: the host's true values. */
    const ds4_gpu_tensor *posd = st->graph ? st->pos : NULL;
    const uint32_t ng = st->graph ? (st->graph_pos_cap + n) / ratio : st->ng_src[src];
    st->idx_owner = (int16_t)il;
    if (!ng) { st->idx_topk = 0; st->idx_ratio = 0; return true; }   /* no completed groups yet: window only */
    const uint32_t dec_rows = DS4_MTP_MAX_BLOCK + 2u;
    uint32_t Rb = n <= dec_rows ? n : n / 32u;
    if (Rb < dec_rows) Rb = dec_rows < n ? dec_rows : n;
    if (st->graph && Rb < n) { fprintf(stderr, "ds4: [v41] graph route's index scoring does not block (n %u)\n", n); return false; }
    if (!v41_index_scratch_prepare(st, ng, Rb, n)) return false;
    if (!v41_tproj(m, st->iq, l->indexer_attn_q_b, DS4_N_LORA_Q, (uint64_t)IH * IK, st->qrn, n, 1)) return false;
    if (!v41_rope(st->iq, st->pos, n, IH, IK, ratio, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp4_tensor(st->iq, (uint64_t)n * IH, IK, 32, false)) return false;
    if (!ds4_gpu_v41_matmul_bf16_tensor(st->iw, m->map, m->size, l->indexer_proj->abs_offset, E, IH, st->xn, n)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->iw, (uint64_t)n * IH)) return false;
    /* weights = proj(x) * (softmax_scale * n_heads^-0.5), multiplied on bf16 as the official model does */
    if (!ds4_gpu_v41_scale_round_tensor(st->iw, (uint64_t)n * IH, (float)(1.0 / sqrt((double)IK) / sqrt((double)IH)))) return false;
    /* C2 compact candidates: the source layer (L20) writes ascending block lists; later indexer sources score only the list */
    const uint32_t kcap = v->candidate_source_layer >= 0 && v->candidate_topk_blocks > 0 ? (uint32_t)v->candidate_topk_blocks : 0u;
    const bool uses_cand = kcap > 0 && (int32_t)il > v->candidate_source_layer && st->cand_owner >= 0 && st->cand;
    const bool is_cand = kcap > 0 && (int32_t)il == v->candidate_source_layer && st->cand;
    const uint32_t bs = v->candidate_block_size > 0 ? (uint32_t)v->candidate_block_size : 1u, lrow = (1u + kcap) * 4u;
    const uint32_t topk = DS4_N_INDEXER_TOP_K < ng ? DS4_N_INDEXER_TOP_K : ng;
    for (uint32_t b0 = 0; b0 < n; b0 += Rb) {
        const uint32_t nr = n - b0 < Rb ? n - b0 : Rb;
        const bool whole = b0 == 0 && nr == n;
        ds4_gpu_tensor *iq = whole ? st->iq : ds4_gpu_tensor_view(st->iq, (uint64_t)b0 * IH * IK * 4, (uint64_t)nr * IH * IK * 4);
        ds4_gpu_tensor *iw = whole ? st->iw : ds4_gpu_tensor_view(st->iw, (uint64_t)b0 * IH * 4, (uint64_t)nr * IH * 4);
        ds4_gpu_tensor *cd = (whole || !st->cand) ? st->cand : ds4_gpu_tensor_view(st->cand, (uint64_t)b0 * lrow, (uint64_t)nr * lrow);
        /* idx's row stride is topk (min(index_topk, ng)), not the configured
         * constant: views offset by 512 while the kernels stride by topk
         * misplace every row block when ng < 512 (the 2026-10-01 conviction). */
        ds4_gpu_tensor *ix = whole ? st->idx : ds4_gpu_tensor_view(st->idx, (uint64_t)b0 * topk * 4, (uint64_t)nr * topk * 4);
        const ds4_gpu_tensor *cl = uses_cand ? cd : NULL;
        bool ok = iq && iw && ix && (cd || !st->cand);
        if (ok) ok = ds4_gpu_v41_indexer_score_tensor(st->iscore, iq, st->index_k[src], iw, cl, bs, kcap, nr, st->pos0 + b0, ng, IH, IK, ratio, posd) != 0;
        if (ok && is_cand)
            ok = ds4_gpu_v41_candidate_blocks_tensor(cd, st->iscore, nr, st->pos0 + b0, ng, ratio, kcap, bs, posd) != 0;
        if (ok) ok = ds4_gpu_v41_indexer_topk_tensor(ix, st->iscore, nr, ng, topk, ratio, posd, cl, bs, kcap) != 0;
        if (!whole) { ds4_gpu_tensor_free(iq); ds4_gpu_tensor_free(iw); if (cd) ds4_gpu_tensor_free(cd); ds4_gpu_tensor_free(ix); }
        if (!ok) return false;
    }
    if (is_cand) st->cand_owner = (int16_t)il;
    st->idx_topk = topk;
    st->idx_ratio = ratio;   /* the attention computes each query's segment length from its own position (see ds4_gpu_v41.h) */
    return true;
}

/* CED's edge layer: only this layer's compressed KV + index key go into the
 * source cache; the decoder segment reads this cache and its own attention
 * output is nobody's input (core_v41_attn.c:197-201). */
static bool v41_attention_kv_only(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (!ratio || !g_ds4_v41.is_kv_source[il]) return true;
    return v41_compress_source(e, st, il, ratio);
}

/* ---- attention, three segments (core_v41_attn.c:286-364) ----
 * in: projections (q path + window kv); cache: window / compressor / indexer
 * / sparse attention / ring commit; out: output projections.  The kv branch
 * forks onto the side stream at n <= 8 (core_v41_attn.c:290-308): both
 * branches only share-read xn/pos and write disjoint tensors, so the kv
 * GEMV + rms + rope + act_quant hide behind q_b; the ring copy in
 * v41_attn_cache lands on the main stream after the join. */
static bool v41_attn_in(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    /* the towers run v41_draft_attention, so the engine's condition has no
     * draft term (core_v41_attn.c:296); n > 8 prefill shares scratch and
     * stays serial */
    const int fork = n <= DS4_V41_GEMV_MAX_TOK && ds4_gpu_side_mark() && ds4_gpu_side_begin();
    /* window kv: wkv -> bf16 -> kv_norm -> rope(last 64 dims) -> fp8 act_quant(32 blocks) -> window rows [SWA, SWA+n) */
    if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->xn, n, 1) ||
        !ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS) ||
        !v41_rope(st->kvn, st->pos, n, 1, HD, ratio, false) ||
        !ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, n, HD, 32)) { if (fork) (void)ds4_gpu_side_join(); return false; }
    if (fork) (void)ds4_gpu_side_main();
    /* q path: q_a -> bf16 -> q_norm -> q_b -> bf16 -> rope(absolute position) */
    if (!v41_tproj(m, st->qr, l->attn_q_a, E, Q, st->xn, n, 1) ||
        !ds4_gpu_v41_rms_norm_tensor(st->qrn, st->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS) ||
        !v41_tproj(m, st->q, l->attn_q_b, Q, (uint64_t)NH * HD, st->qrn, n, 1) ||
        !v41_rope(st->q, st->pos, n, NH, HD, ratio, false)) { if (fork) (void)ds4_gpu_side_join(); return false; }
    if (fork && !ds4_gpu_side_join()) return false;
    return true;
}

static bool v41_attn_cache(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t n = st->n, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, SWA = DS4_N_SWA;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint64_t rowb = (uint64_t)HD * 4;
    if (!ds4_gpu_tensor_copy(st->win[il], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)n * rowb)) return false;
    uint32_t ng = 0, topk = 0, iratio = 0; const ds4_gpu_tensor *comp = NULL;
    const ds4_gpu_tensor *posd = st->graph ? st->pos : NULL;   /* graph: the position is the device slot, ng/topk are the bucket caps */
    if (ratio) {
        if (v->is_kv_source[il] && !v41_compress_source(e, st, il, ratio)) return false;
        if (v->is_index_source[il] && !v41_index_source(e, st, il, ratio)) return false;
        const int16_t src = v->kv_source_of[il];
        if (src < 0 || !st->comp_kv[src]) { fprintf(stderr, "ds4: [v41] L%u compressed layer has no source\n", il); return false; }
        ng = st->graph ? (st->graph_pos_cap + n) / ratio : st->ng_src[src];
        if (ng) {
            if (st->idx_owner < 0) { fprintf(stderr, "ds4: [v41] L%u compressed layer has no topk\n", il); return false; }
            comp = st->comp_kv[src]; topk = st->idx_topk; iratio = st->idx_ratio;
            /* Graph kernels derive ng and the segment length from ONE ratio, so
             * the topk source layer's ratio must equal this layer's (the
             * wiring table guarantees it; refuse rather than misread). */
            if (st->graph && iratio != ratio) { fprintf(stderr, "ds4: [v41] L%u graph route: topk source ratio %u != layer ratio %u\n", il, iratio, ratio); return false; }
        }
    }
    /* sparse attention (window 128 + topk compressed rows, sink in the denominator only) -> bf16 -> inverse rope */
    if (!ds4_gpu_v41_sparse_attn_tensor(st->o, st->q, st->win[il], (ng && topk) ? comp : NULL, (ng && topk) ? st->idx : NULL, m->map, m->size,
                                        l->attn_sinks->abs_offset, n, st->pos0, SWA, ng, topk, iratio, NH, HD,
                                        (float)(1.0 / sqrt((double)HD)), 0, 1, st->win_from[il], posd, st->graph_pos_cap)) return false;   /* win_from clamps CED-unwritten ring slots */
    /* Graph route's verify batch: save the ring cells the commit below is
     * about to overwrite (same cells the direct route's v41_spec_snapshot
     * saves; the snapshot buffer was built by the direct warm round -- no
     * allocation under capture). */
    if (st->graph && n > 1u) {
        if (!st->snap_win[il]) { fprintf(stderr, "ds4: [graph] L%u window-ring snapshot buffer not built (a verify batch must warm direct first)\n", il); return false; }
        if (!ds4_gpu_v41_win_ring_snap_tensor(st->win[il], st->snap_win[il], 0u, 0u, n, SWA, HD, 0, posd)) return false;
    }
    /* commit this block's rows into the ring AFTER the attention read the history */
    return ds4_gpu_v41_win_commit_tensor(st->win[il], st->pos0, n, SWA, HD, posd) != 0;
}

static bool v41_attn_out(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (!v41_rope(st->o, st->pos, n, NH, HD, ratio, true)) return false;
    const uint32_t grp = NH / DS4_N_OUT_GROUP;
    if (!v41_tproj_grouped(m, st->low, l->attn_output_a, DS4_N_OUT_GROUP,
                           (uint64_t)grp * HD, DS4_N_LORA_O, st->o, n, 1)) return false;
    /* wo_b must go through the dispatcher: a direct fp4x32 call would read a
     * q4_K tensor with the wrong decoder (the 2026-09-20 conviction: NaN from
     * L00 on, no error). */
    if (!v41_tproj(m, st->attn_out, l->attn_output_b, (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, E, st->low, n, 1)) return false;
    return true;
}

static bool v41_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    return v41_attn_in(e, st, il) && v41_attn_cache(e, st, il) && v41_attn_out(e, st, il);
}

/* ---- the forward driver (core_v41_forward.c) ---- */

/* Half-layer entry (single.md S3): mix -> fused kernel (split + hc_pre + norm).
 * Order matters: hc_pre inside the fused kernel uses the incoming pre_mix
 * (the previous layer's); the swap happens after the kernel. */
static bool v41_hc_half(ds4_engine *e, ds4_v41_state *st, const ds4_layer_weights *l, bool attn_half) {
    const ds4_model *m = &e->model;
    const ds4_tensor *fn = attn_half ? l->hc_attn_fn : l->hc_ffn_fn;
    const ds4_tensor *sc = attn_half ? l->hc_attn_scale : l->hc_ffn_scale;
    const ds4_tensor *bs = attn_half ? l->hc_attn_base : l->hc_ffn_base;
    const ds4_tensor *nm = attn_half ? l->attn_norm : l->ffn_norm;
    if (!ds4_gpu_v41_hc_mix_tensor(st->mix, st->hc, m->map, m->size, fn->abs_offset, DS4_N_EMBD, DS4_N_HC, st->n, DS4_RMS_EPS)) return false;
    if (!ds4_gpu_v41_hc_fused_tensor(st->pre, st->post, st->comb, st->x, st->xn, st->mix, st->hc, st->pre_mix,
                                     m->map, m->size, sc->abs_offset, bs->abs_offset, nm->abs_offset,
                                     DS4_N_EMBD, DS4_N_HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS, st->n)) return false;
    { ds4_gpu_tensor *t = st->pre; st->pre = st->pre_mix; st->pre_mix = t; }
    return true;
}

/* MoE (official MoE.forward): router f32/bf16 by tensor type -> routed VQ
 * experts -> shared expert (fp8/fp4) -> bf16.  At n <= 8 the routed partials
 * stay in the worker scratch and the tail entry folds them with the shared
 * expert in one launch (the engine's tail path).  The shared expert forks
 * onto the side stream there (core_v41_forward.c:83-92): it only share-reads
 * xn with the router + routed segment, whose latency then hides behind the
 * shared GEMVs; the towers share scratch and never fork, and n > 8 prefill
 * GEMM shares bf16 scratch, so both stay serial (the engine's conditions). */
static bool v41_moe(const ds4_model *m, const ds4_layer_weights *l, ds4_v41_state *st, uint32_t il) {
    const uint32_t n = st->n, E = DS4_N_EMBD, FF = DS4_N_FF_EXP;
    /* Decode small batches fold the routed partials into the shared expert in
     * one launch; the towers never take that arm (core_v41_forward.c:77). */
    const bool tail = !st->draft && n <= DS4_V41_GEMV_MAX_TOK;
    /* Draft towers: 128 experts top-3, their own router shape
     * (core_v41_forward.c:78-80). */
    const uint32_t NE = st->draft ? g_ds4_v41.mtp_experts : DS4_N_EXPERT;
    const uint32_t KU = st->draft ? g_ds4_v41.mtp_used : DS4_N_EXPERT_USED;
    const int fork = tail && ds4_gpu_side_mark() && ds4_gpu_side_begin();
    if (fork) {
        if (!v41_tproj(m, st->sg, l->ffn_gate_shexp, E, FF, st->xn, n, 1) ||
            !v41_tproj(m, st->su, l->ffn_up_shexp, E, FF, st->xn, n, 1) ||
            !ds4_gpu_v41_swiglu_tensor(st->sh, st->sg, st->su, n, FF, DS4_SWIGLU_CLAMP_EXP) ||
            !v41_tproj(m, st->so, l->ffn_down_shexp, FF, E, st->sh, n, 1)) { (void)ds4_gpu_side_join(); return false; }
        (void)ds4_gpu_side_main();
    }
    /* router gate: bf16 on the main layers, f32 on the towers (two converter plans) */
    if (l->ffn_gate_inp->type == DS4_TENSOR_F32) {
        if (!ds4_gpu_v41_matmul_f32_tensor(st->glog, m->map, m->size, l->ffn_gate_inp->abs_offset, E, NE, st->xn, n)) { goto fork_fail; }
    } else if (!ds4_gpu_v41_matmul_bf16_tensor(st->glog, m->map, m->size, l->ffn_gate_inp->abs_offset, E, NE, st->xn, n)) { goto fork_fail; }
    /* Image positions pick experts with bias_vl (official Gate.bias_vl); the
     * draft tower and any block without images pass NULL = all-text. */
    const bool vl = st->vrows_on && !st->draft;
    if (!ds4_gpu_v41_router_tensor(st->sel, st->rw, st->glog, m->map, m->size, l->ffn_exp_probs_b->abs_offset,
                                   vl ? g_v41_bias_vl[il] : NULL, vl ? st->vrows : NULL, n, NE,
                                   KU, DS4_EXPERT_WEIGHT_SCALE)) { goto fork_fail; }
    if (st->draft) {
        /* The two on-disk tower expert forms (core_v41_forward.c:108-120).
         * Blob: the main fused decode kernel with layer = DS4_N_LAYER + tower
         * (that id only picks the gr sidecar / head-cache slot; towers have no
         * gr).  Per-expert: the dense fp4x32 MoE.  Reading a blob with the
         * per-expert kernel yields a full set of fake drafts and shows up only
         * as low acceptance — the two forms must never be mixed. */
        const ds4_tensor *tb = st->tower_exps_vq[il];
        if (tb) {
            if (!ds4_gpu_v41_routed_moe_tensor(st->routed, m->map, m->size, tb->abs_offset, tb->bytes, E, FF, E, st->sel, st->rw,
                                               NE, KU, DS4_SWIGLU_CLAMP_EXP, st->xn, (uint32_t)DS4_N_LAYER + il, n)) return false;
        } else if (!ds4_gpu_v41_mtp_moe_tensor(st->routed, m->map, il, st->tower_exp_off[il], E, FF, E, st->sel, st->rw,
                                               NE, KU, DS4_SWIGLU_CLAMP_EXP, st->xn, n)) return false;
    } else if (!ds4_gpu_v41_routed_moe_tensor(tail ? NULL : st->routed, m->map, m->size, l->ffn_exps_vq->abs_offset, l->ffn_exps_vq->bytes,
                                       E, FF, E, st->sel, st->rw, DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, st->xn, il, n)) { goto fork_fail; }
    /* shared expert: w1/w3 -> bf16 -> swiglu(clamp) -> bf16 -> w2 -> bf16 */
    if (fork) { if (!ds4_gpu_side_join()) return false; }
    else if (!v41_tproj(m, st->sg, l->ffn_gate_shexp, E, FF, st->xn, n, 1) ||
             !v41_tproj(m, st->su, l->ffn_up_shexp, E, FF, st->xn, n, 1) ||
             !ds4_gpu_v41_swiglu_tensor(st->sh, st->sg, st->su, n, FF, DS4_SWIGLU_CLAMP_EXP) ||
             !v41_tproj(m, st->so, l->ffn_down_shexp, FF, E, st->sh, n, 1)) return false;
    if (tail) {   /* decode batch: reduce + add + bf16 round in one launch (same order, bit-identical) */
        if (!ds4_gpu_v41_moe_tail_tensor(st->y, st->so, st->rw, n, DS4_N_EXPERT_USED, E)) return false;
    } else {
        if (!ds4_gpu_tensor_copy(st->y, 0, st->routed, 0, (uint64_t)n * E * 4)) return false;
        if (!ds4_gpu_v41_add_tensor(st->y, st->so, (uint64_t)n * E)) return false;
        if (!ds4_gpu_v41_round_bf16_tensor(st->y, (uint64_t)n * E)) return false;
    }
    return true;

fork_fail:
    /* Main submission failed while the independent shared branch still owns
     * request buffers. Join before the caller releases or retries the state. */
    if (fork) { (void)ds4_gpu_side_join(); }
    return false;
}

static void v41_dump_rows(const ds4_v41_state *st, const ds4_gpu_tensor *t, const char *kind, uint32_t il, uint64_t rowf) {
    char p[4400]; float *buf = xmalloc((size_t)st->n * rowf * 4);
    ds4_gpu_synchronize();
    if (ds4_gpu_tensor_read(t, 0, buf, (uint64_t)st->n * rowf * 4)) {
        snprintf(p, sizeof p, "%s.%s_L%02u.bin", st->dump_prefix, kind, il);
        FILE *f = fopen(p, "wb"); if (f) { fwrite(buf, 4, (size_t)st->n * rowf, f); fclose(f); }
    }
    free(buf);
}

/* Diagnostic probe (DS41_FWD_TRACE=1): per-stage max magnitude and non-finite
 * count for the late layers, to localize a divergence.  Readbacks force a
 * sync; debug only, never on the release path. */
static void v41_trace_probe(uint32_t il, const char *tag, const ds4_gpu_tensor *t, uint64_t n) {
    float *buf = xmalloc((size_t)n * 4);
    ds4_gpu_synchronize();
    if (ds4_gpu_tensor_read(t, 0, buf, n * 4)) {
        float mx = 0.f; uint64_t nf = 0;
        for (uint64_t i = 0; i < n; i++) { if (!isfinite(buf[i])) nf++; if (fabsf(buf[i]) > mx) mx = fabsf(buf[i]); }
        fprintf(stderr, "[trace] L%02u %s max %.6g nonfinite %llu\n", il, tag, (double)mx, (unsigned long long)nf);
    }
    free(buf);
}

/* One layer (official Block.forward): engram -> attn half -> attention ->
 * hc_post -> ffn half -> MoE -> hc_post; pre_mix carries ffn_pre forward. */
/* The draft towers' attention arm (ds4_ds41_draft.inc, core_v41_attn.c:239-261). */
static bool v41_draft_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il);
static bool v41_layer(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model;
    /* Draft towers reuse this layer body with the tower weights (layer id =
     * tower index; core_v41_forward.c:184). */
    const ds4_layer_weights *l = st->draft ? &e->weights.v41_towers.tower[il] : &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD;
    static int tr_on = -1;
    if (tr_on < 0) tr_on = getenv("DS41_FWD_TRACE") ? 1 : 0;
    static int l0_on = -1;   /* DS41_DUMP_L0=1: layer-0 stage dumps (the n>=17 divergence instrument, kept) */
    if (l0_on < 0) l0_on = getenv("DS41_DUMP_L0") ? 1 : 0;
    const int l0 = l0_on && il == 0u && st->dump_prefix != NULL;
    const int tr = tr_on && il >= 27u;
    if (tr) v41_trace_probe(il, "hc_in", st->hc, (uint64_t)n * DS4_N_HC * E);
    if (!st->draft && !st->no_engram && g_ds4_v41.engram_index_of[il] >= 0) {
        if (!v41_engram(e, st, il)) return false;
    }
    /* DSpark: capture the target layers' attention input (the hc four-route
     * mean, after the engram and before the layer; core_v41_forward.c:197-207)
     * into the main-hidden ring.  Only the last mainh_cap rows of the block
     * move; the graph route passes the device position slot so the ring cell
     * follows the replayed step (the eager path passes it directly).  A wrong
     * capture point does not error — acceptance collapses to about 1. */
    if (!st->draft && st->mainh && g_ds4_v41.mtp_target_slot[il] >= 0) {
        const uint32_t rows = n < st->mainh_cap ? n : st->mainh_cap, src0 = n - rows;
        if (!ds4_gpu_v41_hc_mean_tensor(st->mainh, st->hc, E, DS4_N_HC, rows, src0,
                                        (uint32_t)g_ds4_v41.mtp_target_slot[il], g_ds4_v41.n_mtp_target,
                                        st->pos0 + src0, st->mainh_cap, st->graph ? st->pos : NULL)) return false;
        st->mainh_wrote = rows;
    }
    if (!v41_hc_half(e, st, l, true)) return false;
    if (l0) v41_dump_rows(st, st->xn, "l0axn", il, E);
    if (tr) {
        v41_trace_probe(il, "mix_attn", st->mix, (uint64_t)n * (2u * DS4_N_HC + DS4_N_HC * DS4_N_HC));
        v41_trace_probe(il, "comb_attn", st->comb, (uint64_t)n * DS4_N_HC * DS4_N_HC);
        v41_trace_probe(il, "x_attn", st->x, (uint64_t)n * E);
        v41_trace_probe(il, "xn_attn", st->xn, (uint64_t)n * E);
    }
    if (st->draft) {
        if (!v41_draft_attention(e, st, il)) return false;   /* the tower window arm */
    } else if (!v41_attention(e, st, il)) return false;
    if (l0) { v41_dump_rows(st, st->o, "l0ao", il, (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM); v41_dump_rows(st, st->attn_out, "l0aout", il, E); }
    if (tr) v41_trace_probe(il, "o", st->o, (uint64_t)n * DS4_N_HEAD * DS4_N_HEAD_DIM);
    if (!ds4_gpu_v41_hc_post_tensor(st->hc2, st->attn_out, st->hc, st->post, st->comb, E, DS4_N_HC, n)) return false;
    { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }
    if (l0) v41_dump_rows(st, st->hc, "l0ahc", il, (uint64_t)DS4_N_HC * E);
    if (tr) v41_trace_probe(il, "hc_post_attn", st->hc, (uint64_t)n * DS4_N_HC * E);
    if (!v41_hc_half(e, st, l, false)) return false;
    if (tr) {
        v41_trace_probe(il, "comb_ffn", st->comb, (uint64_t)n * DS4_N_HC * DS4_N_HC);
        v41_trace_probe(il, "xn_ffn", st->xn, (uint64_t)n * E);
    }
    if (!v41_moe(m, l, st, il)) return false;
    if (tr) v41_trace_probe(il, "y", st->y, (uint64_t)n * E);
    if (st->dump_prefix) { v41_dump_rows(st, st->xn, "x", il, E); v41_dump_rows(st, st->y, "y", il, E); }   /* fixture: MoE in/out */
    if (!ds4_gpu_v41_hc_post_tensor(st->hc2, st->y, st->hc, st->post, st->comb, E, DS4_N_HC, n)) return false;
    { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }
    return true;
}

/* Forward body: embed -> expand to hc lanes -> layers -> head.  The direct
 * path owns the position advance and the sync (v41_forward below). */
static bool v41_forward_body(ds4_engine *e, ds4_v41_state *st) {
    const uint32_t n = st->n, E = DS4_N_EMBD;
    st->ced_done = 0;
    bool ok = v41_embed(&e->model, st->x, st->tok, e->weights.token_embd, DS4_N_VOCAB, n, E) != 0;
    /* Image span positions take the visual embeddings over the word embeddings
     * (official merge_image_embeddings: after embed, before the hc expansion);
     * only an image-bearing prefill block turns this on. */
    if (ok && st->vrows_on) ok = ds4_gpu_v41_row_select_tensor(st->x, st->vimg->embd, st->vrows, n, E) != 0;
    if (ok) ok = ds4_gpu_v41_expand_hc_tensor(st->hc, st->x, E, DS4_N_HC, n) != 0;
    const double t0 = now_sec();
    double lt[DS4_N_LAYER + 1], tprev = t0;
    /* CED edge = the last kv source layer (metadata-driven; the decoder segment reads that layer's cache) */
    uint32_t ced_edge = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) if (g_ds4_v41.is_kv_source[il]) ced_edge = il;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (st->ced_skip && il == ced_edge) {   /* only the global KV; no downstream consumer of this block's decoder segment */
            ok = v41_hc_half(e, st, &e->weights.layer[il], true) && v41_attention_kv_only(e, st, il);
            if (ok && ds4_gpu_flush_commands() == 0) ok = false;
            if (n >= 64u) fputc('\n', stderr);
            /* U-A body-end marker: the CED return never reaches the [v41-prof] emitter below,
             * so this block's exact end (edge included) prints here instead (probe only). */
            if (g_ds4_v41_prof) fprintf(stderr, "[v41-prof] pos %u+%u total %.1f ms | ced edge %u\n", st->pos0, n, (now_sec() - t0) * 1e3, il);
            st->ced_done = 1;
            /* the edge layer and everything after it wrote no window ring this block: their earliest valid
             * position moves to the next block's start (the attention kernel clamps the window floor to it) */
            for (uint32_t l = ced_edge; l < DS4_N_LAYER; l++) st->win_from[l] = st->pos0 + n;
            return ok;
        }
        ok = v41_layer(e, st, il);
        if (ok && (g_ds4_v41_prof || n >= 64u) && ds4_gpu_flush_commands() == 0) ok = false;   /* decode does not sync per layer */
        if (g_ds4_v41_prof) { const double tn = now_sec(); lt[il] = tn - tprev; tprev = now_sec(); }
        if (n >= 64u) fprintf(stderr, "[v41] pos %u+%u L%02u %s %.1fs\r", st->pos0, n, il, ok ? "ok" : "FAIL", now_sec() - t0);
    }
    if (n >= 64u) fputc('\n', stderr);
    if (!ok) { fprintf(stderr, "ds4: [v41] forward failed (pos0 %u n %u)\n", st->pos0, n); return false; }
    /* head: h = hc_pre(hc, pre_mix) -> norm -> output projection (f32 logits) */
    if (!ds4_gpu_v41_hc_pre_tensor(st->x, st->hc, st->pre_mix, E, DS4_N_HC, n)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->xn, st->x, e->model.map, e->model.size, e->weights.output_norm->abs_offset, E, n, DS4_RMS_EPS)) return false;
    if (getenv("DS41_DUMP_HEAD")) {
        /* P4-4 head investigation: dump the head weights as the resolve serves
         * them and the raw mapping bytes for the same spans (see ds41_forward.h). */
        const char *pref = getenv("DS41_DUMP_HEAD");
        char pa[512], pb[512];
        snprintf(pa, sizeof pa, "%s.norm_dev.bin", pref);
        snprintf(pb, sizeof pb, "%s.norm_raw.bin", pref);
        ds4_gpu_v41_debug_read_weight(e->model.map, e->model.size, e->weights.output_norm->abs_offset, 20480u, pa);
        ds4_gpu_v41_debug_dump_raw(e->model.map, e->model.size, e->weights.output_norm->abs_offset, 20480u, pb);
        snprintf(pa, sizeof pa, "%s.w_dev.bin", pref);
        snprintf(pb, sizeof pb, "%s.w_raw.bin", pref);
        ds4_gpu_v41_debug_read_weight(e->model.map, e->model.size, e->weights.output->abs_offset, 1u << 20, pa);
        ds4_gpu_v41_debug_dump_raw(e->model.map, e->model.size, e->weights.output->abs_offset, 1u << 20, pb);
        fprintf(stderr, "ds4: [v41] head weights dumped to %s.*\n", pref);
    }
    if (st->head_last_only && n > DS4_V41_GEMV_MAX_TOK) {
        /* core_v41_forward.c:334-342: the generate path reads only the last
         * position, so a prefill block runs the decode-style GEMV head on
         * xlast instead of the full GEMM.  Deliberately NOT bit-identical to
         * the GEMM head (the engine's own note, 2026-09-29): the GEMV keeps
         * the q4_K head weights unrounded — this IS the generate path's
         * reference, and the port mirrors it for exactly that reason.  The
         * logits buffer only needs the rows the caller reads (logits_rows),
         * so the n > logits_rows guard stays on the full-GEMM arm only. */
        if (!ds4_gpu_tensor_copy(st->xlast, 0, st->xn, (uint64_t)(n - 1u) * E * 4, (uint64_t)E * 4)) return false;
        if (!v41_tproj(&e->model, st->logits, e->weights.output, E, DS4_N_VOCAB, st->xlast, 1, 0)) return false;
        st->last_logit_row = 0;
    } else {
        if (n > st->logits_rows) { fprintf(stderr, "ds4: [v41] head needs %u logits rows, buffer has %u\n", n, st->logits_rows); return false; }
        if (!v41_tproj(&e->model, st->logits, e->weights.output, E, DS4_N_VOCAB, st->xn, n, 0)) return false;
        st->last_logit_row = n - 1u;
    }
    if (g_ds4_v41_prof) {   /* per-layer ms: one glance shows which layer eats the time (engram L1/L14, sources L2/8/14/20) */
        if (ds4_gpu_flush_commands() == 0) return false;
        lt[DS4_N_LAYER] = now_sec() - tprev;
        fprintf(stderr, "[v41-prof] pos %u+%u total %.1f ms | full | layers(ms):", st->pos0, n, (now_sec() - t0) * 1e3);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) fprintf(stderr, " %.1f", lt[il] * 1e3);
        fprintf(stderr, " | head %.1f\n", lt[DS4_N_LAYER] * 1e3);
    }
    return true;
}

/* One prefill block (core_v41_forward.c:361-370): the last block keeps at
 * least a window of positions (official SWA bounded replay); CED skips the
 * decoder segment of every block but the last. */
static bool v41_forward(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n);
/* The chunk loop's slice rule: the cap, then keep at least a window for the
 * next block.  One rule, two users: the current chunk and the cross-block
 * prefetch's declaration for the next one. */
static uint32_t v41_chunk_nc(uint32_t np, uint32_t c0, uint32_t cap) {
    uint32_t nc = np - c0 < cap ? np - c0 : cap;
    const uint32_t rest = np - c0 - nc;
    if (rest > 0u && rest < DS4_N_SWA && nc > DS4_N_SWA) nc -= DS4_N_SWA - rest;
    return nc;
}
static bool v41_prefill_chunk(ds4_engine *e, ds4_v41_state *st, const int32_t *prompt, uint32_t np, uint32_t *c0, uint32_t cap) {
    const uint32_t nc = v41_chunk_nc(np, *c0, cap);
    st->ced_skip = (!g_ds4_v41_decoder_full && *c0 + nc < np) ? 1 : 0;
    /* Cross-block prefetch (ds41_forward.h): declare the next chunk while
     * this one still has compute to hide its first round.  A nonzero return
     * is a provider error and refuses the run; "no prefetch right now" is
     * 0, not a refusal. */
    if (st->efeed && st->efeed->prefetch_next && *c0 + nc < np) {
        const uint32_t c1 = *c0 + nc;
        const uint32_t n1 = v41_chunk_nc(np, c1, cap);
        if (st->efeed->prefetch_next(c1, prompt + c1, (int)n1, st->efeed->ud) != 0) {
            fprintf(stderr, "ds4: [v41] engram prefetch declaration refused (pos0 %u n %u)\n", c1, n1);
            return false;
        }
    }
    if (!v41_forward(e, st, prompt + *c0, nc)) return false;
    *c0 += nc;
    return true;
}

static bool v41_forward(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n) {
    if (n == 0 || n > st->cap_tok) { fprintf(stderr, "ds4: [v41] forward block %u exceeds cap %u\n", n, st->cap_tok); return false; }
    if (st->n_past + n > st->ctx) { fprintf(stderr, "ds4: [v41] context full (%u+%u > %u)\n", st->n_past, n, st->ctx); return false; }
    st->n = n; st->pos0 = st->n_past; st->idx_owner = -1; st->cand_owner = -1; st->idx_topk = 0; st->idx_ratio = 0;
    st->graph = 0;   /* direct dispatch: position-dependent kernels take the host's true values (core_v41_forward.c:374) */
    {
        int32_t *pos = xmalloc((size_t)n * 4);
        for (uint32_t i = 0; i < n; i++) pos[i] = (int32_t)(st->pos0 + i);
        const bool ok = ds4_gpu_tensor_write(st->tok, 0, ids, (uint64_t)n * 4) && ds4_gpu_tensor_write(st->pos, 0, pos, (uint64_t)n * 4);
        free(pos);
        if (!ok) return false;
    }
    /* Vision: map this block's positions to vimg->embd rows and run the tower
     * once (all images together) before the first block; afterwards each block
     * only fetches rows by position.  himg records image positions like hist
     * does, so the engram hash lookback can stop at them; vrows_on is cleared at
     * block end so the decode step and clones never see it
     * (core_v41_forward.c:388-404). */
    st->vrows_on = 0;
    if (st->vimg && !st->draft) {
        if (!v41_vision_encode(e, st->vimg)) return false;
        int32_t *rows = xmalloc((size_t)n * 4);
        int any = 0;
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t p = st->pos0 + i;
            rows[i] = p < st->vimg->np ? st->vimg->row_of[p] : -1;
            st->himg[p] = rows[i] >= 0;
            any |= rows[i] >= 0;
        }
        const bool okr = !any || ds4_gpu_tensor_write(st->vrows, 0, rows, (uint64_t)n * 4);
        free(rows);
        if (!okr) return false;
        st->vrows_on = any;
    } else if (st->himg) {
        memset(st->himg + st->pos0, 0, n);
    }
    float *pm = xmalloc((size_t)n * DS4_N_HC * 4);   /* pre_mix = one-hot on lane 0 */
    for (uint32_t i = 0; i < n * DS4_N_HC; i++) pm[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    const bool okpm = ds4_gpu_tensor_write(st->pre_mix, 0, pm, (uint64_t)n * DS4_N_HC * 4) != 0;
    free(pm);
    if (!okpm) return false;
    st->mainh_wrote = 0;
    st->blk_ids = ids;   /* the engram feed's prepare hashes this block's rows from these (the engine's hist memcpy, core_v41_forward.c:377) */
    /* The feed's block provider: a spec verify batch's tokens exist only
     * inside the forward, so the provider is told the exact block.  Return
     * convention like emit: 0 = filled, nonzero = refuse.  pos0 is still
     * asserted at the point of use (v41_engram_rows). */
    if (st->efeed && st->efeed->prepare && st->efeed->prepare(st->pos0, ids, (int)n, st->efeed->ud) != 0) {
        fprintf(stderr, "ds4: [v41] engram feed prepare failed (pos0 %u n %u)\n", st->pos0, n);
        return false;
    }
    if (ds4_gpu_begin_commands() == 0) return false;
    const bool body_ok = v41_forward_body(e, st);
    st->vrows_on = 0;   /* image rows belong to this block only */
    if (!body_ok) return false;
    if (ds4_gpu_end_commands() == 0 || ds4_gpu_synchronize() == 0) return false;
    /* The main-hidden ring's accounting (core_v41_forward.c:391-400): continue
     * the run when this block touches the previous one, restart it otherwise
     * (a CED block writes no ring rows).  Only the contiguous run ending at
     * mainh_end is real — the draft window fill may read nothing else. */
    if (st->mainh && st->mainh_wrote) {
        const int64_t first = (int64_t)st->pos0 + (int64_t)(n - st->mainh_wrote);
        const bool contig = st->mainh_n > 0 && st->mainh_end == first - 1;
        const uint32_t nn = contig ? st->mainh_n + st->mainh_wrote : st->mainh_wrote;
        st->mainh_n = nn > st->mainh_cap ? st->mainh_cap : nn;
        st->mainh_end = (int64_t)st->pos0 + (int64_t)n - 1;
    }
    st->n_past += n;
    /* Warm counters for the decode graph (core_v41_forward.c:402-405): a
     * direct forward of this shape has built every lazy allocation and kernel
     * attribute, so capture is allowed for it afterwards.  CED blocks never
     * count (they skip whole layers).  (The engine also excludes its hook
     * early-stop; the port has no hooks.) */
    if (n == 1u && !st->draft && !st->ced_done) st->n_direct1++;
    if (n < DS4_MTP_MAX_BLOCK + 2u && !st->draft && !st->ced_done) st->n_direct_n[n]++;
    return true;
}

/* ---- spec verify snapshot / rollback (core_v41_forward.c:241/267) ----
 * Why back up: a verify batch computes 1+k positions into the cache; the ones
 * not accepted must be undone.  Window: the batch's rows first sit in the
 * buffered block region; after each layer win_commit writes them into the
 * ring, overwriting the cell from 128 steps ago — still inside the next
 * step's visible window.  So only the n cells about to be overwritten are
 * saved (40 layers x n(<=6) rows x 512 x 4 B ~ 0.5 MB, against 10.5 MB for
 * the whole 128-row window).  Compressor pending rows: that is a genuine left
 * shift and the snapshot point is BEFORE the shift (taken in
 * v41_compress_source).  comp_kv/index_k are written by absolute group number
 * and the counters are overwritten next round — no backup needed. */
static bool v41_spec_snapshot(ds4_v41_state *st, uint32_t n) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    const uint32_t nmax = DS4_MTP_MAX_BLOCK + 1u;
    bool ok = true;
    if (!n || n > nmax) { fprintf(stderr, "ds4: [v41] spec snapshot batch %u over limit %u\n", n, nmax); return false; }
    /* snap stores the ring cell batch row i is about to overwrite, indexed by
     * row, so the restore interval is exactly [keep, n). */
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        if (!st->snap_win[il]) st->snap_win[il] = v41_alloc((uint64_t)nmax * rowb, &ok);
        if (ok && !ds4_gpu_v41_win_ring_snap_tensor(st->win[il], st->snap_win[il], st->n_past,
                                                    0u, n, DS4_N_SWA, DS4_N_HEAD_DIM, 0, NULL)) ok = false;
    }
    st->snap_n = n;
    st->snap_past = st->n_past;
    st->snap_on = ok ? 1u : 0u;
    return ok;
}

static bool v41_restore_match(const ds4_gpu_tensor *got, uint64_t goff,
                               const ds4_gpu_tensor *snap, uint64_t soff,
                               uint8_t *buf, uint64_t rowb, const char *lane,
                               uint32_t il, uint32_t row) {
    const uint64_t gb = got ? ds4_gpu_tensor_bytes(got) : 0u;
    const uint64_t sb = snap ? ds4_gpu_tensor_bytes(snap) : 0u;
    if (!got || !snap || goff > gb || rowb > gb - goff || soff > sb || rowb > sb - soff ||
        !ds4_gpu_tensor_read(got, goff, buf, rowb) ||
        !ds4_gpu_tensor_read(snap, soff, buf + rowb, rowb)) {
        fprintf(stderr, "ds4: [v41] rollback check FAIL: %s L%u row %u readback (%llu bytes)\n",
                lane, il, row, (unsigned long long)rowb);
        return false;
    }
    if (memcmp(buf, buf + rowb, (size_t)rowb) != 0) {
        fprintf(stderr, "ds4: [v41] rollback check FAIL: %s L%u row %u byte mismatch\n", lane, il, row);
        return false;
    }
    return true;
}

/* Check restore plumbing against this verify's own snapshots. N>1 accepted
 * values may depend on rejected drafts through MoE FP; that is not the
 * reference here. Run after verify wait, outside capture, and read exactly
 * one full tensor row into each bounded lane. */
static bool v41_rollback_check(const ds4_v41_state *st, uint32_t keep, uint32_t mainh_before) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    const uint64_t np = (uint64_t)st->snap_past + keep;
    if (!ds4_gpu_synchronize()) {
        fprintf(stderr, "ds4: [v41] rollback check FAIL: synchronization\n");
        return false;
    }
    if (!keep || keep > st->snap_n || st->snap_n > DS4_MTP_MAX_BLOCK + 1u ||
        (uint64_t)st->snap_past + st->snap_n > st->ctx || st->n_past != np || st->snap_on) {
        fprintf(stderr, "ds4: [v41] rollback check FAIL: frontier past %u expected %llu, keep %u/%u\n",
                st->n_past, (unsigned long long)np, keep, st->snap_n);
        return false;
    }
    if (st->mainh) {
        /* Verify wrote snap_n new rows. Rejected cells also overwrote old
         * history, so rollback shortens the run instead of restoring a full ring. */
        const uint32_t dropped = st->snap_n - keep;
        const uint32_t expected = mainh_before > dropped ? mainh_before - dropped : 0u;
        if (!st->mainh_cap || st->mainh_cap > DS4_N_SWA || mainh_before > st->mainh_cap ||
            mainh_before < st->snap_n || st->mainh_n != expected || st->mainh_n > np ||
            st->mainh_end != (int64_t)np - 1) {
            fprintf(stderr, "ds4: [v41] rollback check FAIL: mainh count %u expected %u, end %lld expected %lld\n",
                    st->mainh_n, expected, (long long)st->mainh_end, (long long)np - 1);
            return false;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_kv_source[il]) { continue; }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio || st->ng_src[il] != np / ratio || st->cpend[il] != np % ratio ||
            (ratio > 1u && st->snap_cpend[il] != st->snap_past % ratio)) {
            fprintf(stderr, "ds4: [v41] rollback check FAIL: source L%u groups %u pending %u ratio %u at %llu\n",
                    il, st->ng_src[il], st->cpend[il], ratio, (unsigned long long)np);
            return false;
        }
    }
    uint8_t *buf = malloc((size_t)(2u * rowb));
    if (!buf) {
        fprintf(stderr, "ds4: [v41] rollback check FAIL: readback allocation\n");
        return false;
    }
    uint32_t windows = 0, pending = 0;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        for (uint32_t i = keep; i < st->snap_n && ok; i++) {
            const uint64_t cell = ((uint64_t)st->snap_past + i) % DS4_N_SWA;
            ok = v41_restore_match(st->win[il], cell * rowb, st->snap_win[il], (uint64_t)i * rowb,
                                    buf, rowb, "window", il, i);
            windows += ok;
        }
        if (!ok || !g_ds4_v41.is_kv_source[il]) { continue; }
        const uint32_t pend = st->cpend[il];
        const uint64_t first = (uint64_t)st->snap_cpend[il] + keep - pend;
        for (uint32_t i = 0; i < pend && ok; i++) {
            const uint64_t src = (first + i) * rowb, dst = (uint64_t)i * rowb;
            ok = v41_restore_match(st->cpre_kv[il], dst, st->snap_cpre_kv[il], src,
                                    buf, rowb, "cpre_kv", il, i) &&
                 v41_restore_match(st->cpre_sc[il], dst, st->snap_cpre_sc[il], src,
                                    buf, rowb, "cpre_sc", il, i);
            pending += ok;
        }
    }
    free(buf);
    if (ok) {
        fprintf(stderr, "ds4: [v41] rollback check PASS: pos %llu keep %u/%u, %u window rows, %u pending rows, mainh %u\n",
                (unsigned long long)np, keep, st->snap_n, windows, pending, st->mainh_n);
    }
    return ok;
}

/* keep = accepted positions in this batch (including the confirmed head).
 * Three things (core_v41_forward.c:267-296): (1) the window ring — batch rows
 * [keep, n) were committed by the layers and overwrote real rows from 128
 * steps back; write the snapshot's cells back (the accepted head rows stay).
 * (2) compressed KV/index keys: rewind only the group COUNT to
 * (pos0+keep)/ratio; contents are by absolute group number and get
 * overwritten next round.  (3) compressor pending rows: take the last pend'
 * rows of the pre-shift snapshot — those are the accepted tokens' inputs.
 * Miss (3) and the next pool mixes a rejected draft token into a group: no
 * error, the group's global KV just stops matching pure decode (the byte
 * gate catches it).  Miss (1) and the next step reads "future drafts" as
 * history. */
static bool v41_spec_rollback(ds4_v41_state *st, uint32_t keep) {
    if (!st->snap_on) return false;
    const uint32_t mainh_before = st->mainh_n;
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    const uint32_t pos0 = st->snap_past, np = pos0 + keep;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (keep < st->snap_n &&
            !ds4_gpu_v41_win_ring_snap_tensor(st->win[il], st->snap_win[il], pos0,
                                              keep, st->snap_n, DS4_N_SWA, DS4_N_HEAD_DIM, 1, NULL)) return false;
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        st->ng_src[il] = np / ratio;
        if (ratio == 1u || !st->cpre_kv[il]) { st->cpend[il] = 0; continue; }
        const uint32_t pend2 = np % ratio;
        if (pend2) {   /* snapshot rows (snap_cpend + keep - pend2) onward = the last accepted tokens' inputs */
            const uint64_t src = (uint64_t)(st->snap_cpend[il] + keep - pend2) * rowb;
            if (!ds4_gpu_tensor_copy(st->cpre_kv[il], 0, st->snap_cpre_kv[il], src, (uint64_t)pend2 * rowb)) return false;
            if (!ds4_gpu_tensor_copy(st->cpre_sc[il], 0, st->snap_cpre_sc[il], src, (uint64_t)pend2 * rowb)) return false;
        }
        st->cpend[il] = pend2;
    }
    st->n_past = np;
    st->snap_on = 0;
    if (st->mainh) {   /* rejected rows leave the ring's contiguous run (their cells get overwritten next step; until then they must not be read) */
        const uint32_t dropped = st->snap_n - keep;
        st->mainh_n = st->mainh_n > dropped ? st->mainh_n - dropped : 0;
        st->mainh_end = (int64_t)np - 1;
    }
    const char *check = getenv("DS41_ROLLBACK_CHECK");
    if (check && strcmp(check, "0") != 0) { return v41_rollback_check(st, keep, mainh_before); }
    return true;
}

/* The greedy per-row next-token pick (v41_device_next) and the sampling
 * routes moved to ds4_ds41_sample.inc (unit H): the engine keeps them in
 * src/core/core_v41_sample.c, and the graph module calls v41_sample_pick from
 * there.  The old argmax-only helper that lived here was its greedy arm. */

/* Engram table metadata for the feed builder (ds41_forward.h): the C loader
 * filled g_ds4_v41 from the GGUF (v41_load_wiring), so the harness never
 * re-parses the metadata. */
int ds4_v41_engram_count(void *engine) {
    (void)engine;
    return (int)g_ds4_v41.n_engram;
}
int ds4_v41_engram_meta(void *engine, int k, uint32_t *il, char *path, int path_cap,
                        uint64_t *rows, uint64_t *weight_off, uint64_t *scale_off,
                        uint32_t *head_dim, uint32_t *cols) {
    (void)engine;
    if (k < 0 || (uint32_t)k >= g_ds4_v41.n_engram) return 0;
    if (il) *il = (uint32_t)g_ds4_v41.engram_layer[k];
    if (path && path_cap > 0) snprintf(path, (size_t)path_cap, "%s", g_ds4_v41.engram_table_path[k]);
    if (rows) *rows = g_ds4_v41.engram_rows[k];
    if (weight_off) *weight_off = g_ds4_v41.engram_weight_off[k];
    if (scale_off) *scale_off = g_ds4_v41.engram_scale_off[k];
    if (head_dim) *head_dim = g_ds4_v41.engram_head_dim;
    if (cols) *cols = (g_ds4_v41.engram_max_ngram - 1u) * g_ds4_v41.engram_heads;
    return 1;
}

/* Zchain gate support (unit D): the shape the sidecar headers must match and
 * the per-layer router-bias reference (name -> tensor -> abs_offset, the
 * engine's model_find_tensor(m, "blk.%u.exp_probs_b.bias") at
 * core_v41_amp.c:104). */
int ds4_v41_shape(void *engine, uint32_t *n_layer, uint32_t *n_expert, uint32_t *n_embd) {
    if (!engine) return 0;
    if (n_layer) *n_layer = DS4_N_LAYER;
    if (n_expert) *n_expert = DS4_N_EXPERT;
    if (n_embd) *n_embd = DS4_N_EMBD;
    return 1;
}

int ds4_v41_router_bias_ref(void *engine, uint32_t il, const void **map, uint64_t *size, uint64_t *offset) {
    ds4_engine *e = (ds4_engine *)engine;
    if (!e || !map || !size || !offset || il >= DS4_N_LAYER) return 0;
    const ds4_tensor *t = e->weights.layer[il].ffn_exp_probs_b;
    if (!t) return 0;
    *map = e->model.map;
    *size = e->model.size;
    *offset = t->abs_offset;
    return 1;
}

/* The zchain device-side mount (unit Z, ds41_forward.h): the three device
 * writes the Rust reader drives, 1:1 with the engine's v41_amp_load loop
 * (core_v41_amp.c:227-270).  The engine reads the files in C; production
 * reads them on the Rust host (zchain.rs, P2/P3) and lands here. */
int ds4_v41_zchain_gr_set(void *engine, uint32_t il, const float *data, uint32_t n_expert, uint32_t n_embd) {
    (void)engine;
    if (DS4_MODEL_VARIANT != DS4_VARIANT_V41 || il >= DS4_N_LAYER) { return 0; }
    /* NULL clears a previous mount's gains through the same backend setter. */
    return ds4_gpu_v41_set_gr_override(il, data, n_expert, n_embd);
}

int ds4_v41_zchain_rb_set(void *engine, uint32_t il, const float *delta, uint32_t n_expert) {
    const void *map = NULL;
    uint64_t size = 0, off = 0;
    if (!delta || !ds4_v41_router_bias_ref(engine, il, &map, &size, &off)) return 0;
    return ds4_gpu_v41_set_rb_override(map, size, off, delta, n_expert);
}

void ds4_v41_zchain_rb_clear(void) {
    (void)ds4_gpu_v41_set_rb_override(NULL, 0, 0, NULL, 0);
}

int ds4_v41_engram_hash_ref(void *engine, ds41_engram_hash_ref *ref) {
    ds4_engine *e = (ds4_engine *)engine;
    if (!e || !ref) return 0;
    const ds4_tensor *tm = e->weights.engram_token_map, *mu = e->weights.engram_multipliers;
    const ds4_tensor *pr = e->weights.engram_primes, *of = e->weights.engram_offsets;
    if (!tm || !mu || !pr || !of) return 0;
    ref->map = e->model.map;
    ref->size = e->model.size;
    ref->token_map_off = tm->abs_offset; ref->token_map_bytes = tm->bytes;
    ref->mult_off = mu->abs_offset;      ref->mult_bytes = mu->bytes;
    ref->prim_off = pr->abs_offset;      ref->prim_bytes = pr->bytes;
    ref->off_off = of->abs_offset;       ref->off_bytes = of->bytes;
    ref->max_ngram = g_ds4_v41.engram_max_ngram;
    ref->heads = g_ds4_v41.engram_heads;
    ref->n_engram = g_ds4_v41.n_engram;
    ref->n_vocab = DS4_N_VOCAB;
    ref->pad = (int32_t)g_ds4_v41.engram_pad;
    return 1;
}

/* ---- the score entry (core_v41_score.c:27-110) ----
 * Teacher-forced scoring; the aux outputs (--score-nll/--score-topk/--score-rms)
 * and the deployment-split mode are not ported (the golden capture used the
 * plain path).  PPL is exp(mean NLL) over n-1 targets, the engine's formula. */
int ds4_v41_score_ids(void *engine, const int *ids, int n_ids, const char *out_path,
                      int no_engram, int chunk, const ds41_engram_feed *feed) {
    ds4_engine *e = (ds4_engine *)engine;
    if (!e || !ids || n_ids < 1 || !out_path) return 1;
    if (DS4_MODEL_VARIANT != DS4_VARIANT_V41) { fprintf(stderr, "ds4: [v41] score entry on a non-V4.1 model\n"); return 1; }
    const uint32_t n = (uint32_t)n_ids;
    const uint32_t ck = chunk > 0 ? (uint32_t)chunk : DS4_V41_CHUNK;
    const uint32_t cap = ck < n ? ck : n;
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, n, 0)) return 1;   /* the score path reads every position's logits */
    st.dump_prefix = (n <= 64u && cap == n) ? out_path : NULL;   /* single-block small runs dump per-layer x/y for the compare */
    st.no_engram = no_engram;
    st.efeed = feed;
    if (no_engram) fprintf(stderr, "[v41] no-engram compare mode: the engram layers are skipped\n");
    if (ck < n) fprintf(stderr, "[v41] chunked %u (%u blocks)\n", ck, (n + ck - 1) / ck);
    FILE *fo = fopen(out_path, "wb");
    if (!fo) { fprintf(stderr, "ds4: cannot write %s\n", out_path); v41_state_free(&st); return 1; }
    int hd[2] = { (int)n, (int)DS4_N_VOCAB };
    fwrite(hd, 4, 2, fo);
    float *lg = xmalloc((size_t)cap * DS4_N_VOCAB * 4);
    double nll = 0.0; const double t0 = now_sec();
    bool ok = true;
    for (uint32_t c0 = 0, nc = 0; ok && c0 < n; c0 += nc) {
        nc = n - c0 < cap ? n - c0 : cap;
        st.ced_skip = 0;
        ok = v41_forward(e, &st, ids + c0, nc);
        if (ok) ok = ds4_gpu_tensor_read(st.logits, 0, lg, (uint64_t)nc * DS4_N_VOCAB * 4) != 0;
        if (!ok) break;
        fwrite(lg, 4, (size_t)nc * DS4_N_VOCAB, fo);
        for (uint32_t i = 0; i < nc; i++) {   /* teacher forcing: position c0+i predicts ids[c0+i+1] */
            const uint32_t row_i = c0 + i;
            const float *row = lg + (size_t)i * DS4_N_VOCAB;
            const int tgt = row_i + 1u < n ? ids[row_i + 1u] : -1;
            if (tgt < 0) continue;
            float mx = row[0]; for (uint32_t v = 1; v < DS4_N_VOCAB; v++) if (row[v] > mx) mx = row[v];
            double se = 0.0; for (uint32_t v = 0; v < DS4_N_VOCAB; v++) se += exp((double)row[v] - mx);
            nll += -((double)row[tgt] - mx - log(se));
        }
    }
    if (fo) fclose(fo);
    free(lg);
    if (ok) fprintf(stderr, "[v41] score done S=%u V=%u -> %s  PPL(segment %u tokens) = %.4f  %.1fs\n",
                    n, DS4_N_VOCAB, out_path, n, n > 1 ? exp(nll / (double)(n - 1)) : 0.0, now_sec() - t0);
    v41_state_free(&st);
    return ok ? 0 : 1;
}

