/* ds4_ds41_kvreuse.inc -- V4.1 KV prefix reuse, the port of
 * src/core/core_v41_kvreuse.c (/data/YoungAi @ 7fb26cc, commit a8e878a
 * 2026-10-11; the file itself is the reference, line for line below).
 *
 * Why (the engine's own account, core_v41_kvreuse.c:1-6): every request used
 * to prefill from token 0, while a chat or agent step sends "the previous
 * prompt + what the model just said + a new piece" -- the long head is
 * identical to the previous request's and was re-read every time (measured
 * 10-11: 31 tool calls adding 5-9k tokens a step, 40k-190k prefilled a step,
 * 40 s to 3 min a step at a normal 1000 t/s prefill / 40 t/s decode).
 *
 * How: after prefill, before decode, save a prompt-end checkpoint of the
 * buffers decode mutates or later reads -- compressed KV and index keys up to
 * the used group count (ng_src, :69), the window ring's first SWA rows (a
 * position p always lives in slot p % SWA, cuda_kv_ring.inc.cu), the
 * compressor remainder rows (cpre_kv/cpre_sc head cpend rows), the draft
 * main-hidden ring + counters, per-layer ng_src/cpend/win_from, the prompt
 * token sequence, and the last logits row (:83-84, 2026-10-12).  The next
 * request whose prompt starts with those tokens (memcmp, :98) opens a fresh
 * state, copies the checkpoint in and prefills only the suffix, starting at
 * c0 = reused (core_v41_api.c:179-180); an identical prompt restores the
 * logits row to row 0 and prefills nothing (:109).
 *
 * The named difference from a cold start (:16-19): the suffix chunks from the
 * checkpoint position, so block boundaries differ -- FP-level, the same class
 * as changing --v41-chunk.  The CED "last block runs the full decoder" rule
 * still applies to the suffix (v41_prefill_chunk counts from c0).
 *
 * Invalidation (:22): the KV was computed with the mounted amplifier, so the
 * engine drops the checkpoint on ds4_engine_v41_switch_plugins
 * (core_v41_state.c:95/:111).  The port has no switch event -- the Rust host
 * mounts the sidecar per run (v41_run.rs, v41_mount_zchain) -- so the mount
 * notes its identity here and the native drops when it changes
 * (v41_kvreuse_note_mount).
 *
 * Failure mode (:24): a copy failure means this request prefills from token 0
 * (never a half checkpoint), with one log line.
 *
 * Port seam (named deviation): the engine memcpy's K.tok into st->hist
 * (:118) because its engram hash reads the token history from the state.  The
 * port's history is host-side (V41Feed/FeedCore.hist in v41_run.rs), so the
 * host seeds it from v41_kvreuse_peek before the run: same content, different
 * seam.  The restore re-runs the memcmp, so the native decision stays the
 * authority. */

static int g_v41_kv_reuse = 1;   /* the engine's g_ds4_v41_kv_reuse (core_v41_kvreuse.c:27) */
void ds4_engine_v41_set_kv_reuse(int on) { g_v41_kv_reuse = on; }

static int g_v41_kvreuse_last_reused;
int ds4_engine_v41_last_reused(void) { return g_v41_kvreuse_last_reused; }

static struct {
    int valid;
    uint32_t n;                 /* checkpoint position = that prompt's token count (core_v41_kvreuse.c:32) */
    int32_t *tok; uint32_t tok_cap;
    ds4_gpu_tensor *ckv[DS4_MAX_LAYER], *idk[DS4_MAX_LAYER], *win[DS4_MAX_LAYER],
                   *cpk[DS4_MAX_LAYER], *cps[DS4_MAX_LAYER], *mainh;
    ds4_gpu_tensor *logits;     /* the prompt-end logits row [V] f32: an identical prompt prefills nothing and samples from it */
    uint32_t ng_src[DS4_MAX_LAYER], cpend[DS4_MAX_LAYER], win_from[DS4_MAX_LAYER];
    int64_t mainh_end; uint32_t mainh_n;
    uint64_t bytes;             /* checkpoint VRAM, the log line's number */
    int mount_has; uint64_t mount_hash;   /* the amplifier identity that computed the KV (port addition, see the header) */
} K;

static void v41_kvreuse_drop(void) { K.valid = 0; }

/* Grow a checkpoint buffer when it is too small (1.5x; content not kept --
 * the caller rewrites the whole span right after).  core_v41_kvreuse.c:45-52. */
static bool v41_kvreuse_ensure(ds4_gpu_tensor **t, uint64_t need) {
    if (!need) return true;
    if (*t && ds4_gpu_tensor_bytes(*t) >= need) return true;
    if (*t) ds4_gpu_tensor_free(*t);
    bool ok = true;
    *t = v41_alloc(need + need / 2, &ok);
    return ok && *t;
}

static bool v41_kvreuse_cp(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bytes) {
    return !bytes || (dst && src && ds4_gpu_tensor_copy(dst, 0, src, 0, bytes));
}

/* Called with the state exactly at the prompt end, after the prefill and
 * before decode (core_v41_api.c:192).  core_v41_kvreuse.c:59. */
static void v41_kvreuse_save(const ds4_v41_state *st, const int32_t *prompt, uint32_t np) {
    if (!g_v41_kv_reuse) return;
    K.valid = 0;
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    bool ok = true; uint64_t total = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ok = v41_kvreuse_ensure(&K.win[il], (uint64_t)DS4_N_SWA * rowb) &&
             v41_kvreuse_cp(K.win[il], st->win[il], (uint64_t)DS4_N_SWA * rowb);
        total += (uint64_t)DS4_N_SWA * rowb;
        K.ng_src[il] = st->ng_src[il]; K.cpend[il] = st->cpend[il]; K.win_from[il] = st->win_from[il];
        if (!ok || !st->comp_kv[il]) continue;
        const uint64_t kb = (uint64_t)st->ng_src[il] * DS4_V41_CKV_BYTES;
        const uint64_t ib = (uint64_t)st->ng_src[il] * DS4_V41_IDXK_BYTES;
        ok = v41_kvreuse_ensure(&K.ckv[il], kb) && v41_kvreuse_cp(K.ckv[il], st->comp_kv[il], kb) &&
             v41_kvreuse_ensure(&K.idk[il], ib) && v41_kvreuse_cp(K.idk[il], st->index_k[il], ib);
        total += kb + ib;
        if (ok && st->cpre_kv[il] && st->cpend[il]) {
            const uint64_t pb = (uint64_t)st->cpend[il] * rowb;
            ok = v41_kvreuse_ensure(&K.cpk[il], pb) && v41_kvreuse_cp(K.cpk[il], st->cpre_kv[il], pb) &&
                 v41_kvreuse_ensure(&K.cps[il], pb) && v41_kvreuse_cp(K.cps[il], st->cpre_sc[il], pb);
            total += 2 * pb;
        }
    }
    if (ok && st->mainh) {
        const uint64_t mb = ds4_gpu_tensor_bytes(st->mainh);
        ok = v41_kvreuse_ensure(&K.mainh, mb) && v41_kvreuse_cp(K.mainh, st->mainh, mb);
        total += mb;
    }
    K.mainh_end = st->mainh_end; K.mainh_n = st->mainh_n;
    const uint64_t vb = (uint64_t)DS4_N_VOCAB * 4;
    if (ok) ok = v41_kvreuse_ensure(&K.logits, vb) && st->logits &&
                 ds4_gpu_tensor_copy(K.logits, 0, st->logits, (uint64_t)st->last_logit_row * vb, vb);
    total += vb;
    if (ok && !ds4_gpu_synchronize()) ok = false;
    if (!ok) { fprintf(stderr, "ds4: [v41] KV checkpoint does not fit (VRAM); the next request prefills whole\n"); return; }
    if (K.tok_cap < np) { free(K.tok); K.tok_cap = np + np / 2; K.tok = xmalloc((size_t)K.tok_cap * 4); }
    memcpy(K.tok, prompt, (size_t)np * 4);
    K.n = np; K.bytes = total; K.valid = 1;
}

/* Called on a freshly allocated state, before the chunk loop: when the
 * checkpoint connects, copy it in and return the prefill start (= the
 * checkpoint position; == np means nothing to prefill, the prompt-end logits
 * row is already at row 0).  0 = prefill from token 0.
 * core_v41_kvreuse.c:96. */
static uint32_t v41_kvreuse_restore(ds4_v41_state *st, const int32_t *prompt, uint32_t np) {
    g_v41_kvreuse_last_reused = 0;
    if (!g_v41_kv_reuse || !K.valid || K.n > np || K.n > st->ctx ||
        memcmp(K.tok, prompt, (size_t)K.n * 4)) return 0;
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ok = v41_kvreuse_cp(st->win[il], K.win[il], (uint64_t)DS4_N_SWA * rowb);
        if (!ok || !st->comp_kv[il]) continue;
        ok = v41_kvreuse_cp(st->comp_kv[il], K.ckv[il], (uint64_t)K.ng_src[il] * DS4_V41_CKV_BYTES) &&
             v41_kvreuse_cp(st->index_k[il], K.idk[il], (uint64_t)K.ng_src[il] * DS4_V41_IDXK_BYTES);
        if (ok && st->cpre_kv[il] && K.cpend[il])
            ok = v41_kvreuse_cp(st->cpre_kv[il], K.cpk[il], (uint64_t)K.cpend[il] * rowb) &&
                 v41_kvreuse_cp(st->cpre_sc[il], K.cps[il], (uint64_t)K.cpend[il] * rowb);
    }
    if (ok && st->mainh && K.mainh) ok = v41_kvreuse_cp(st->mainh, K.mainh, ds4_gpu_tensor_bytes(st->mainh));
    if (ok && K.n == np) {   /* identical prompt: the logits row lands at row 0, nothing is prefilled */
        ok = v41_kvreuse_cp(st->logits, K.logits, (uint64_t)DS4_N_VOCAB * 4);
        st->last_logit_row = 0;
    }
    if (ok && !ds4_gpu_synchronize()) ok = false;
    if (!ok) {   /* a half copy leaves nothing behind: a cold start rewrites ring, remainder rows and group counts from position 0 */
        fprintf(stderr, "ds4: [v41] KV checkpoint does not copy into the fresh state; this request prefills whole\n");
        return 0;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        st->ng_src[il] = K.ng_src[il]; st->cpend[il] = K.cpend[il]; st->win_from[il] = K.win_from[il];
    }
    if (st->mainh) { st->mainh_end = K.mainh_end; st->mainh_n = K.mainh_n; }
    st->n_past = K.n;
    g_v41_kvreuse_last_reused = (int)K.n;
    fprintf(stderr, "[v41] KV reuse %u/%u token (checkpoint %.1f MB), prefill suffix %u\n",
            K.n, np, (double)K.bytes / 1048576.0, np - K.n);
    return K.n;
}

/* The restore's read-only guard minus the copies: the prefill start this
 * prompt would get.  The Rust host seeds the engram feed's history with
 * prompt[0..reused) before the run (the engine's memcpy at
 * core_v41_kvreuse.c:118, a port seam).  A stale peek is harmless: the
 * restore re-runs the memcmp and a cold start rewrites the same ranges; the
 * state's ctx fence is not repeated here (it can only make the peek
 * optimistic, and an optimistic seed is an exact prefix of the prompt). */
uint32_t v41_kvreuse_peek(const int32_t *prompt, uint32_t np) {
    if (!g_v41_kv_reuse || !K.valid || K.n > np || memcmp(K.tok, prompt, (size_t)K.n * 4)) return 0;
    return K.n;
}

/* The port's stand-in for the engine's ds4_engine_v41_switch_plugins drop
 * (core_v41_state.c:111): the host reports the mounted amplifier's identity
 * once per run and the checkpoint dies when it differs. */
void v41_kvreuse_note_mount(int has_mount, uint64_t hash) {
    if (K.mount_has == has_mount && K.mount_hash == hash) return;
    K.mount_has = has_mount; K.mount_hash = hash;
    if (K.valid) fprintf(stderr, "ds4: [v41] amplifier changed; the KV checkpoint is dropped\n");
    v41_kvreuse_drop();
}
