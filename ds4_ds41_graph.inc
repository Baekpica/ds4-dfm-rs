/* ds4_ds41_graph.inc — the decode-step CUDA graph orchestration, ported from
 * the C engine at /data/YoungAi (commit 3946dbc): src/core/core_decode_graph.c
 * (every function cites its engine lines) and the host halves the engine
 * keeps in core_v41_api.c (the round drive) and core_v41_draft.c (the draft
 * graph).  Included by ds4.c after ds4_ds41_forward.inc and before
 * ds4_ds41_draft.inc (same TU: it calls the forward's statics).
 *
 * What one captured step contains: slot uploads (zero-copy kernels, not
 * memcpy nodes), the whole forward body, the argmax tail, and the readback --
 * with every position-dependent value read from the device slot st->pos (the
 * "device position" convention, ds4_gpu_v41.h): the host passes the bucket
 * CAPS for grid/shared sizes and the kernels compute the true ng/topk from
 * posd[0].  That is what lets one capture replay hundreds of steps.  Position
 * buckets of 1024: crossing re-captures (tens of ms, amortized).
 *
 * Capture failure policy (the engine's, kept): any unsafe call from the
 * capturing thread voids the capture (ThreadLocal) and capture_end reports
 * NULL -- the step's kernels never ran, so the caller re-runs that step by
 * direct dispatch.  n=1 failure switches THIS state to direct for good
 * (n1_off); a batch failure switches every batch graph off (batch_off, ONE
 * field for all n).  Never a process-wide switch: the engine's 09-22 night
 * lost a whole serving night to one global flag.
 *
 * Port adaptations, named:
 *  - the engram provider is synchronous (P4-4): dg_launch calls the feed's
 *    prepare before the capture/launch (the engine's v41_engram_prefetch slot)
 *    and the graph records only the zero-copy upload -- no flag-wait kernel,
 *    no arm/serve (nothing is in flight to wait for).  Provider success, the
 *    declared pos0 and the captured pinned pointers are validated on EVERY
 *    launch (the captured upload bakes the address; a moved buffer invalidates
 *    the graphs and re-captures).
 *  - no hist buffer: the provider owns the history the engine keeps in
 *    st->hist (its prepare gets the block's tokens directly).
 *  - the tail follows st->dev_sample (unit H): sampling launches the device
 *    sample kernel over the n rows (per-request params baked into the graph;
 *    spec_q feeds the rejection sampling for n > 1) and the wait assembles
 *    the rows with v41_sample_pick; the greedy arm keeps one argmax per row
 *    and picks next[i] = slot[4i].  The state is per request, so dev_sample
 *    and spec_q are constants for the life of its graphs.
 *  - the --idx-mma indexer scratch prepare (core_decode_graph.c:179) has no
 *    counterpart (the indexer mma is not ported).
 *  - DS41_NO_GRAPH=1 disables the graph (the engine's
 *    ds4_engine_v41_set_graph(0); the CLI flags land with the serving unit).
 */
#ifndef DS4_DS41_GRAPH_INC
#define DS4_DS41_GRAPH_INC

#define DGRAPH_BUCKET 1024u   /* position bucket width: inside a bucket every kernel's grid cap is fixed; crossing re-captures */
#define DGRAPH_NMAX (DS4_MTP_MAX_BLOCK + 2u)   /* graphs indexed by batch size n: [1] pure decode; [2..1+block] verify batches */

typedef struct {
    void *exec;                /* this n's graph instance; NULL = not captured yet */
    uint32_t lo, cap;          /* valid for a batch whose first row is in [lo, cap] (last row reaches cap+n-1) */
    uint64_t gen;              /* backend scratch generation at capture (ds4_gpu_v41_scratch_generation) */
    uint32_t idx_gen;          /* index scratch generation at capture (st->iscap_gen) */
} dg_inst;
typedef struct {
    dg_inst g[DGRAPH_NMAX];
    int32_t *tokv, *posv;      /* pinned [NMAX] = this batch's tokens / positions: the zero-copy upload's source */
    float *onehot;             /* pinned [NMAX][HC]: the pre_mix one-hot (lane 0), uploaded at the step's head */
    int32_t *next;             /* pinned [NMAX*4]: row i's argmax lands in next[4i] (each device slot is 16 B, read back whole) */
    ds4_gpu_tensor *am;        /* device argmax slots [NMAX][16 B] */
    ds4_gpu_tensor *amv[DGRAPH_NMAX];   /* per-row views (each argmax writes its own slot's first int) */
    uint32_t steps, bsteps, captures, regrow;   /* pure-decode steps walked / verify batches walked / captures / re-captures after a scratch move */
    uint32_t k0steps;          /* k=0 rounds' steps on the n=1 graph (2026-09-28): not in the steady account (a draft ran before them) */
    int cur_pure;              /* this step is a pure decode step (counted) or a k=0 round's (counted separately) */
    double t_prep, t_launch, t_sync, t_gap;   /* step-boundary account (seconds, cumulative; n=1 only): start+fetch submit / cudaGraphLaunch host time / wait / last sync -> this step */
    double b_prep, b_launch; uint32_t b_n;    /* same two, verify batches (n>=2) and k=0 rounds (the 2026-10-07 host-gap split) */
    double b_begin, b_engram, b_check, b_arm; /* start split four ways: slot write / engram prepare / graph checks (+recapture) / arm */
    double t_capture; uint32_t n_capture;     /* capture host time (incl. n=1) / count */
    double t_last_sync, t_launched;   /* last step's sync return time / this step's launch return time */
    int direct_pending; int32_t pending_tok;   /* n=1 capture failed this step: the launch was not sent, wait re-runs it direct */
    uint32_t cur_n;            /* this launch's row count (wait reads back / advances by it) */
    int batch_off;             /* a verify-batch capture failed: every batch graph is off for this state (ONE field, the engine's) */
    int n1_off;                /* an n=1 capture failed: this state walks direct afterwards.  Request-local, never the process */
    /* Port addition: the feed's pinned raw pointers at capture.  The captured
     * upload kernel baked these ADDRESSES; a provider that swaps a buffer must
     * invalidate the graphs (dg_launch checks every launch). */
    const void *feed_raw[DS4_V41_MAX_ENGRAM];
} decode_graph;

/* The engine's ds4_engine_v41_set_graph + the port's diagnostics.  A run with
 * a readback diagnostic set (FWD_TRACE / DUMP_HEAD / FEED_DIGEST) stays
 * direct: those sync inside the forward and would void every capture. */
static int g_ds4_v41_graph = -1;
void ds4_engine_v41_set_graph(int on) { g_ds4_v41_graph = on; }
static int v41_graph_enabled(void) {
    if (g_ds4_v41_graph < 0) g_ds4_v41_graph = getenv("DS41_NO_GRAPH") ? 0 : 1;
    return g_ds4_v41_graph;
}
static int v41_graph_diag(void) {
    static int on = -1;
    if (on < 0) on = (getenv("DS41_FWD_TRACE") || getenv("DS41_DUMP_HEAD") || getenv("DS41_FEED_DIGEST")) ? 1 : 0;
    return on;
}

static void v41_graph_free(ds4_v41_state *st) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!g) return;
    for (uint32_t n = 0; n < DGRAPH_NMAX; n++) {
        if (g->g[n].exec) ds4_gpu_decode_graph_free(g->g[n].exec);
        if (g->amv[n]) ds4_gpu_tensor_free(g->amv[n]);
    }
    ds4_gpu_host_free(g->tokv); ds4_gpu_host_free(g->posv); ds4_gpu_host_free(g->onehot); ds4_gpu_host_free(g->next);
    if (g->am) ds4_gpu_tensor_free(g->am);
    if (g->steps) {
        const double wall = (g->t_gap + g->t_prep + g->t_launch + g->t_sync) / g->steps;   /* steady per-step wall (first direct step and captures excluded) */
        fprintf(stderr, "ds4: [graph] walked %u pure-decode steps, %u captures (%u re-captures after scratch moves); per-step host: last sync->enter %.0f us, "
                        "start+fetch submit %.0f us, cudaGraphLaunch %.0f us, wait %.2f ms => steady %.2f ms/step = %.2f t/s\n",
                g->steps, g->captures, g->regrow,
                g->t_gap / g->steps * 1e6, g->t_prep / g->steps * 1e6, g->t_launch / g->steps * 1e6, g->t_sync / g->steps * 1e3,
                wall * 1e3, 1.0 / wall);
    }
    if (g->bsteps || g->k0steps) fprintf(stderr, "ds4: [graph] verify batches walked %u rounds, k=0 rounds walked %u steps on the n=1 graph (not in the steady account)\n", g->bsteps, g->k0steps);
    free(g); st->dgraph = NULL;
}

static bool dg_alloc(ds4_v41_state *st) {
    if (st->dgraph) return true;
    decode_graph *g = xmalloc(sizeof *g); memset(g, 0, sizeof *g);
    g->tokv = ds4_gpu_host_alloc((uint64_t)DGRAPH_NMAX * sizeof(int32_t));
    g->posv = ds4_gpu_host_alloc((uint64_t)DGRAPH_NMAX * sizeof(int32_t));
    g->onehot = ds4_gpu_host_alloc((uint64_t)DGRAPH_NMAX * DS4_N_HC * sizeof(float));
    g->next = ds4_gpu_host_alloc((uint64_t)DGRAPH_NMAX * 4u * sizeof(int32_t));
    g->am = ds4_gpu_tensor_alloc((uint64_t)DGRAPH_NMAX * 16u);
    st->dgraph = g;
    if (!g->tokv || !g->posv || !g->onehot || !g->next || !g->am) { v41_graph_free(st); return false; }
    for (uint32_t n = 0; n < DGRAPH_NMAX; n++) {
        g->amv[n] = ds4_gpu_tensor_view(g->am, (uint64_t)n * 16u, 16u);
        if (!g->amv[n]) { v41_graph_free(st); return false; }
    }
    for (uint32_t i = 0; i < DGRAPH_NMAX * DS4_N_HC; i++) g->onehot[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    return true;
}

/* Graph eligibility (core_decode_graph.c:98-116).  ready: flag on, main route
 * (not a draft tower), one direct n=1 step warmed (every lazy allocation
 * built), no host-readback probe/hook/fixture.  allowed: same minus the n=1
 * warm (the draft graph has its own warm counter -- the 2026-10-07 finding: a
 * k>=1-only run never walks n=1, so requiring it disabled the draft graph
 * forever).  batch_ready: this n warmed direct and the snapshot buffers built
 * (the direct round builds them; capture must not allocate). */
static bool v41_graph_ready(const ds4_v41_state *st) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    return v41_graph_enabled() && !st->draft && st->n_direct1 > 0 && !g_ds4_v41_prof && !v41_graph_diag() && !st->dump_prefix &&
           !(g && g->n1_off);
}
static bool v41_graph_allowed(const ds4_v41_state *st) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    return v41_graph_enabled() && !st->draft && !g_ds4_v41_prof && !v41_graph_diag() && !st->dump_prefix && !(g && g->n1_off);
}
static bool v41_graph_batch_ready(const ds4_v41_state *st, uint32_t n) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    /* No n=1 warm required: a pinned-k run (--dspark-verify) walks verify
     * batches only, n=1 never runs (core_decode_graph.c:110-115). */
    return v41_graph_enabled() && !st->draft && !g_ds4_v41_prof && !v41_graph_diag() && !st->dump_prefix &&
           n >= 2u && n < DGRAPH_NMAX && st->n_direct_n[n] > 0 && st->snap_win[0] && !(g && g->batch_off);
}

/* Advance the host counters by n steps in closed form (core_decode_graph.c:117-138).
 * The direct route computes ng_src/cpend per layer in v41_compress_source; the
 * closed form is n_past/ratio and n_past%ratio -- the graph's kernels compute
 * the same values from the device position.  The mainh ring accounting is the
 * same expression v41_forward's tail uses. */
static void dg_advance(ds4_v41_state *st, uint32_t n) {
    const uint32_t pos0 = st->n_past;
    st->n_past += n;
    if (st->mainh) {   /* the captured hc_mean kernel wrote this step's rows into the ring; the host only accounts */
        const uint32_t rows = n < st->mainh_cap ? n : st->mainh_cap;
        const int64_t first = (int64_t)pos0 + (int64_t)(n - rows);
        const bool contig = st->mainh_n > 0 && st->mainh_end == first - 1;
        const uint32_t nn = contig ? st->mainh_n + rows : rows;
        st->mainh_n = nn > st->mainh_cap ? st->mainh_cap : nn;
        st->mainh_end = (int64_t)pos0 + (int64_t)n - 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        st->ng_src[il] = st->n_past / ratio;
        st->cpend[il] = ratio > 1u ? st->n_past % ratio : 0u;
    }
}

/* The step's host-side start, shared by direct and capture (core_decode_graph.c:139-145):
 * the state fields as v41_forward writes them.  The engine also memcpys the
 * block's ids into st->hist (the engram hash source); the port's provider owns
 * that history and gets the ids through the feed's prepare instead. */
static void dg_begin_step(ds4_v41_state *st, const int32_t *ids, uint32_t n) {
    (void)ids;
    st->n = n; st->pos0 = st->n_past; st->idx_owner = -1; st->cand_owner = -1; st->idx_topk = 0; st->idx_ratio = 0;
    st->ced_skip = 0;
}

/* Capture this n's graph for the current bucket (core_decode_graph.c:146-201).
 * During a capture the kernels do not run -- nodes are recorded -- and no host
 * state advances (n_past and the counters move in dg_wait). */
static bool dg_capture(ds4_engine *e, ds4_v41_state *st, uint32_t n) {
    decode_graph *g = (decode_graph *)st->dgraph;
    dg_inst *in = &g->g[n];
    if (in->exec) { ds4_gpu_decode_graph_free(in->exec); in->exec = NULL; }
    in->lo = st->n_past;
    uint32_t cap = (st->n_past / DGRAPH_BUCKET + 1u) * DGRAPH_BUCKET - 1u;   /* bucket cap = first row's upper bound; last row reaches cap+n-1 */
    if (cap > st->ctx - n) cap = st->ctx - n;
    in->cap = cap;
    /* Grow every scratch the body can need BEFORE st->graph=1 (no allocation
     * under capture; the engine's 2026-09-23 finding: with the flag set first,
     * the index scratch prepare refused its own capture-time growth and every
     * new bucket voided its capture -- the server ran direct all night). */
    if (!ds4_gpu_v41_attn_scratch_prepare(n, DS4_N_HEAD, DS4_N_HEAD_DIM)) return false;
    /* Index scoring scratch: the most groups any indexer source can show at
     * the bucket's last batch.  The traversal must match the consumer
     * (v41_index_source keys off is_index_source with that layer's own ratio);
     * kv sources alone would miss a pure indexer source with a smaller ratio. */
    uint32_t ng_max = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_index_source[il]) continue;
        const uint32_t r = ds4_layer_compress_ratio(il);
        if (r && (cap + n) / r > ng_max) ng_max = (cap + n) / r;
    }
    if (ng_max && !v41_index_scratch_prepare(st, ng_max, n, n)) return false;
    /* Candidate-block scratch: block count from the bucket's last batch (more
     * groups later in the bucket = more blocks), one list per row. */
    if (g_ds4_v41.candidate_source_layer >= 0 && g_ds4_v41.candidate_block_size > 0) {
        const uint32_t cr = ds4_layer_compress_ratio((uint32_t)g_ds4_v41.candidate_source_layer);
        const uint32_t cbs = (uint32_t)g_ds4_v41.candidate_block_size;
        const uint32_t cng = cr ? (cap + n) / cr : 0u;
        if (cng && !ds4_gpu_v41_candidate_scratch_prepare(n, (cng + cbs - 1u) / cbs)) return false;
    }
    /* (The engine also prepares the --idx-mma indexer scratch here,
     * core_decode_graph.c:179; the port's indexer mma is not ported.) */
    st->graph = 1; st->graph_pos_lo = in->lo; st->graph_pos_cap = cap;
    st->egraph_uploaded = 0;   /* P2: a fresh capture pass records every engram layer's flag wait + upload (core_decode_graph.c:180) */
    if (!ds4_gpu_decode_graph_capture_begin()) { st->graph = 0; return false; }
    /* Slot uploads are zero-copy kernels, not memcpy nodes (nsys on GB10: a
     * memcpy node costs ~170 us, the four of them 0.69 ms/step). */
    bool ok = ds4_gpu_tensor_write_zerocopy(st->tok, 0, g->tokv, (uint64_t)n * sizeof(int32_t)) &&
              ds4_gpu_tensor_write_zerocopy(st->pos, 0, g->posv, (uint64_t)n * sizeof(int32_t)) &&
              ds4_gpu_tensor_write_zerocopy(st->pre_mix, 0, g->onehot, (uint64_t)n * DS4_N_HC * sizeof(float));
    if (ok) ok = v41_forward_body(e, st);
    /* Tail: sampling on => ONE sample-kernel launch over the n rows (full
     * samples / accept flags / residual samples, 16 bytes each; the params are
     * per-request constants baked into the graph); otherwise one argmax per
     * row (each writes its own 16-byte slot's first int).  Either way, ONE
     * zero-copy readback of the whole block (core_decode_graph.c:186-192). */
    if (ok && st->dev_sample) ok = ds4_gpu_v41_sample_tensor(g->am, st->logits, 0u, n, DS4_N_VOCAB, st->pos, st->tok, &st->samp,
                                                             n > 1u ? st->spec_q : NULL) != 0;
    else for (uint32_t i = 0; ok && i < n; i++) ok = ds4_gpu_v41_argmax_tensor(g->amv[i], st->logits, i, DS4_N_VOCAB) != 0;
    if (ok) ok = ds4_gpu_tensor_read_zerocopy(g->next, g->am, 0, (uint64_t)n * 16u) != 0;
    st->graph = 0;
    void *exec = ds4_gpu_decode_graph_capture_end();   /* end the capture on every path or the stream stays in capture mode */
    if (!ok || !exec) { if (exec) ds4_gpu_decode_graph_free(exec); return false; }
    in->exec = exec;
    g->captures++;
    in->gen = ds4_gpu_v41_scratch_generation();   /* stamped after capture_end: the prepares above may have grown (and bumped) it */
    in->idx_gen = st->iscap_gen;
    for (uint32_t k = 0; k < g_ds4_v41.n_engram && k < DS4_V41_MAX_ENGRAM; k++) g->feed_raw[k] = st->efeed ? st->efeed->raw[k] : NULL;
    fprintf(stderr, "ds4: [graph] %u-row graph captured: position bucket [%u, %u]\n", n, in->lo, in->cap);
    return true;
}

/* Direct dispatch of one step (n=1 capture failure's re-run, core_decode_graph.c:203-208). */
static bool dg_direct_step(ds4_engine *e, ds4_v41_state *st, int32_t tok, int32_t *next_tok) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!v41_forward(e, st, &tok, 1u)) return false;
    return v41_device_next(st, g->am, 0u, 1u, NULL, next_tok);
}

/* A scratch pointer moved: every graph baked the old ones (core_decode_graph.c:210-215). */
static void dg_invalidate(decode_graph *g, const char *why) {
    int any = 0;
    for (uint32_t n = 0; n < DGRAPH_NMAX; n++) if (g->g[n].exec) { ds4_gpu_decode_graph_free(g->g[n].exec); g->g[n].exec = NULL; any = 1; }
    if (any) { fprintf(stderr, "ds4: [graph] %s: graphs invalidated, re-capturing\n", why); g->regrow++; }
}

/* ---- the engram graph three-step (core_v41.h:344-347, core_v41_engram.c:261-286)
 * arm: before the launch, publish this step's sequence into the want slot —
 * each captured spin kernel waits flag[ei] >= want — and clear the timeout
 * slot.  serve: after the launch, before the sync — collect each layer's
 * round on the host and publish its flag; the GPU resumes within ~1 us of the
 * publish (ds41_engram.cuh:74-97).  err: after the sync — a host read failure
 * leaves a flag unset, the spin kernel times out and writes the err slot; the
 * step's output is unusable and the caller refuses it.  All three are no-ops
 * for the synchronous protocol (no wait hook) and for engram-less runs. */
static bool v41_engram_graph_live(const ds4_v41_state *st) {
    return !st->no_engram && g_ds4_v41.n_engram > 0 && st->efeed && st->efeed->wait && st->efeed->flags;
}
static void v41_engram_graph_arm(ds4_v41_state *st) {
    if (!v41_engram_graph_live(st)) return;
    /* the flags/seq slots are native-owned mutable state inside the caller's
     * struct; the provider never touches them. */
    v41_engram_feed *fd = (v41_engram_feed *)st->efeed;
    fd->graph_seq++;
    fd->flags[g_ds4_v41.n_engram + 1] = 0;
    fd->flags[g_ds4_v41.n_engram] = (int32_t)fd->graph_seq;
    __sync_synchronize();
}
static void v41_engram_graph_serve(ds4_v41_state *st) {
    if (!v41_engram_graph_live(st)) return;
    const v41_engram_feed *fd = st->efeed;
    for (uint32_t ei = 0; ei < g_ds4_v41.n_engram; ei++) {
        if (fd->wait((int)ei, fd->ud) != 0) return;   /* no publish: the GPU times out, the err check stops the step */
        __sync_synchronize();                          /* rows in memory before the flag (core_v41_engram.c:277-279) */
        fd->flags[ei] = (int32_t)fd->graph_seq;
        __sync_synchronize();
    }
}
static int v41_engram_graph_err(const ds4_v41_state *st) {
    if (!v41_engram_graph_live(st)) return 0;
    return st->efeed->flags[g_ds4_v41.n_engram + 1];
}

/* Launch half of one step (core_decode_graph.c:219-271): write the pinned
 * slots, prepare the engram feed, check/re-capture, launch.  The caller may
 * emit the current token between launch and wait.  Position state must not be
 * touched between them (the graph is reading the slots). */
static bool dg_launch(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n, int pure) {
    if (!dg_alloc(st)) return false;
    decode_graph *g = (decode_graph *)st->dgraph;
    dg_inst *in = &g->g[n];
    if (st->n_past + n > st->ctx) { fprintf(stderr, "ds4: [v41] context full (%u+%u > %u)\n", st->n_past, n, st->ctx); return false; }
    const double t0 = now_sec();
    g->cur_pure = n == 1u && pure;
    /* The gap account only bridges pure decode steps; a batch/k0 step clears it (core_decode_graph.c:227-229). */
    if (g->cur_pure && g->t_last_sync > 0.0) g->t_gap += t0 - g->t_last_sync;
    dg_begin_step(st, ids, n);
    for (uint32_t i = 0; i < n; i++) { g->tokv[i] = ids[i]; g->posv[i] = (int32_t)(st->pos0 + i); }
    __sync_synchronize();   /* slots in memory before the graph's head kernel reads them */
    const double tp0 = now_sec();
    /* The engram feed (the engine's v41_engram_prefetch slot, core_decode_graph.c:233-235):
     * the provider fills the pinned rows for THIS block, before the capture or
     * launch, and its declared position must match the state -- at every
     * launch, not only during capture (the captured upload reads the pinned
     * buffer at replay). */
    if (!st->no_engram && g_ds4_v41.n_engram > 0 && st->efeed) {
        if (st->efeed->prepare && st->efeed->prepare(st->pos0, ids, (int)n, st->efeed->ud) != 0) {
            fprintf(stderr, "ds4: [v41] engram feed prepare failed (pos0 %u n %u)\n", st->pos0, n);
            return false;
        }
        if (st->efeed->pos0 != st->pos0) {
            fprintf(stderr, "ds4: [v41] engram feed pos0 %u != block pos0 %u\n", st->efeed->pos0, st->pos0);
            return false;
        }
    }
    const double tp1 = now_sec();
    /* A scratch pointer moved => the graphs baked stale addresses (the
     * 2026-09-19 illegal-access conviction: a direct verify batch grew the
     * backend scratch and a captured graph read a freed page).  The
     * generation is global: one move invalidates every n. */
    if (in->exec && in->gen != ds4_gpu_v41_scratch_generation()) dg_invalidate(g, "backend scratch moved");
    if (in->exec && in->idx_gen != st->iscap_gen) dg_invalidate(g, "index scratch grew");
    /* Port addition: the captured engram upload baked the pinned addresses. */
    if (in->exec && g_ds4_v41.n_engram > 0) {
        for (uint32_t k = 0; k < g_ds4_v41.n_engram && k < DS4_V41_MAX_ENGRAM; k++) {
            if (g->feed_raw[k] != (st->efeed ? st->efeed->raw[k] : NULL)) { dg_invalidate(g, "engram feed pointers moved"); break; }
        }
    }
    if (!in->exec || st->pos0 < in->lo || st->pos0 + n - 1u > in->cap) {
        const double tc0 = now_sec();
        const bool cap_ok = dg_capture(e, st, n);
        g->t_capture += now_sec() - tc0; g->n_capture++;   /* capture host time is accounted separately (it inflates the start mean in short runs) */
        if (!cap_ok) {
            if (n == 1u) {
                fprintf(stderr, "ds4: *[graph] capture failed: this request's step and all later ones walk direct*\n");
                g->n1_off = 1;
                g->direct_pending = 1;   /* wait re-runs this step by direct dispatch */
                g->pending_tok = ids[0]; g->cur_n = 1u;
                return true;
            }
            fprintf(stderr, "ds4: *[graph] %u-row verify batch capture failed: this state's batches walk direct (the pure-decode graph keeps running)*\n", n);
            g->batch_off = 1;
            return false;   /* nothing was sent; the caller's direct route re-starts the step */
        }
    }
    const double t1 = now_sec();
    /* arm (core_v41.h:344-346, core_v41_engram.c:261-269): the step's sequence
     * must be published before the launch — the captured spin kernels compare
     * flag[ei] against it.  No-op for the synchronous protocol. */
    v41_engram_graph_arm(st);
    const double tp2 = now_sec();
    if (!ds4_gpu_decode_graph_launch(in->exec)) return false;
    g->t_launched = now_sec();
    if (g->cur_pure) { g->t_prep += t1 - t0; g->t_launch += g->t_launched - t1; }
    else { g->b_prep += tp2 - t0; g->b_launch += g->t_launched - tp2; g->b_n++;
           g->b_begin += tp0 - t0; g->b_engram += tp1 - tp0; g->b_check += t1 - tp1; g->b_arm += tp2 - t1; }
    g->cur_n = n;
    return true;
}
/* Verify-batch / k=0 launch host account (cumulative seconds): start (slot write + feed + graph checks) / cudaGraphLaunch itself / count. */
static void v41_graph_batch_host(const ds4_v41_state *st, double *prep, double *launch, uint32_t *n) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    *prep = g ? g->b_prep : 0.0; *launch = g ? g->b_launch : 0.0; *n = g ? g->b_n : 0u;
}
static void v41_graph_batch_prep(const ds4_v41_state *st, double *t_begin, double *t_engram, double *t_check, double *t_arm) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    *t_begin = g ? g->b_begin : 0.0; *t_engram = g ? g->b_engram : 0.0; *t_check = g ? g->b_check : 0.0; *t_arm = g ? g->b_arm : 0.0;
}
static void v41_graph_capture_cost(const ds4_v41_state *st, double *t_capture, uint32_t *n_capture) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    *t_capture = g ? g->t_capture : 0.0; *n_capture = g ? g->n_capture : 0u;
}

/* Wait half (core_decode_graph.c:285-302): sync, pick each row's token from
 * the pinned readback, advance the host counters. */
static bool dg_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!g) return false;
    if (g->direct_pending) { g->direct_pending = 0; return dg_direct_step(e, st, g->pending_tok, next); }
    const uint32_t n = g->cur_n;
    /* serve (core_v41_engram.c:271-280): between the launch and the sync,
     * collect each engram layer's round and publish its flag — the graph's
     * spin kernels resume as each layer's rows land.  A synchronous provider
     * finished in prepare; the call is then a no-op. */
    v41_engram_graph_serve(st);
    if (!ds4_gpu_synchronize()) return false;
    /* err (core_v41_engram.c:282-286): a host read failure leaves a flag
     * unset, the spin kernel times out and writes the err slot — refuse the
     * step before its output is sampled or its position advanced. */
    if (v41_engram_graph_err(st)) {
        fprintf(stderr, "ds4: [v41] engram graph wait timed out; the step's output is unusable\n");
        return false;
    }
    const double t3 = now_sec();
    if (g->cur_pure) { g->t_sync += t3 - g->t_launched; g->t_last_sync = t3; }
    else g->t_last_sync = 0.0;   /* after a batch/k0 step the next pure step's gap is not counted */
    __sync_synchronize();
    /* Per-row pick (core_decode_graph.c:300): sampling assembles "accepted =>
     * the draft / rejected => the residual" from the four ints per row, argmax
     * takes each slot's first int (core_v41_sample.c:17-22). */
    v41_sample_pick(g->next, g->tokv, n, st->dev_sample, next);
    dg_advance(st, n);
    if (n != 1u) g->bsteps++; else if (g->cur_pure) g->steps++; else g->k0steps++;
    return true;
}

static bool v41_graph_launch(ds4_engine *e, ds4_v41_state *st, int32_t tok, int after_draft) { return dg_launch(e, st, &tok, 1u, !after_draft); }
static bool v41_graph_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next_tok) { return dg_wait(e, st, next_tok); }
static bool v41_graph_step(ds4_engine *e, ds4_v41_state *st, int32_t tok, int32_t *next_tok) {
    return v41_graph_launch(e, st, tok, 0) && v41_graph_wait(e, st, next_tok);
}

/* Verify batch: launch + snapshot accounting (core_decode_graph.c:312-323).
 * The snapshot BYTES are in the graph (the per-layer win-ring save in
 * v41_attn_cache and the compressor append kernel's snap_cpre); the host only
 * records where this batch starts, how many rows, and each source layer's
 * pending count at that moment -- the same fields the direct route's
 * v41_spec_snapshot + v41_compress_source set, so v41_spec_rollback runs
 * unchanged. */
static bool v41_graph_batch_launch(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n) {
    if (n < 2u || n >= DGRAPH_NMAX) return false;
    if (!dg_launch(e, st, ids, n, 0)) return false;
    st->snap_n = n; st->snap_past = st->pos0; st->snap_on = 1u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio > 1u) st->snap_cpend[il] = st->pos0 % ratio;
    }
    return true;
}
static bool v41_graph_batch_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next) { return dg_wait(e, st, next); }

#endif /* DS4_DS41_GRAPH_INC */
