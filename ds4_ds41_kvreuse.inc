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
 * authority.
 *
 * The disk half (item 4): export/import below move the same span set through
 * the Rust host's KVC store (crates/ds4-kv, the port of ds4_kvstore.c).  The
 * store owns the file (header, token-keyed name, budget, eviction); the
 * native owns the payload: a meta block + the spans in one fixed order.  The
 * tokens stay out of the payload -- the store's text block carries them (LE
 * i32) and the restore re-runs the memcmp, so the native decision stays the
 * only authority.  The mount identity and the payload ABI ride the store's
 * trailer (Rust side, v41_disk.rs), never the payload. */

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
    uint64_t mainh_bytes;       /* the saved mainh span; the disk meta carries it */
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
    K.mainh_bytes = 0;
    if (ok && st->mainh) {
        const uint64_t mb = ds4_gpu_tensor_bytes(st->mainh);
        ok = v41_kvreuse_ensure(&K.mainh, mb) && v41_kvreuse_cp(K.mainh, st->mainh, mb);
        K.mainh_bytes = mb;
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
    if (ok && st->mainh) {
        /* Capacity can include an older checkpoint's tail. Only the saved
         * span belongs to this checkpoint, even when the allocation is larger. */
        ok = K.mainh && K.mainh_bytes >= ds4_gpu_tensor_bytes(st->mainh) &&
             ds4_gpu_tensor_bytes(K.mainh) >= ds4_gpu_tensor_bytes(st->mainh) &&
             v41_kvreuse_cp(st->mainh, K.mainh, ds4_gpu_tensor_bytes(st->mainh));
    }
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

/* ---- The disk half (item 4): the payload export/import -------------------
 *
 * The file is the Rust store's (crates/ds4-kv: header, token-keyed name,
 * atomic rename, budget, eviction); the native writes and reads only the
 * payload after the store's text block:
 *
 *   meta:  per layer (ng_src, cpend, win_from) u32                12 x L
 *          mainh_end i64, mainh_n u32,
 *          mainh_bytes u64, checkpoint bytes u64                      28
 *   spans: per layer win SWA rows, ckv ng_src x 288, idk ng_src x 72,
 *          cpk/cps cpend rows each -- the v41_kvreuse_save set, one fixed
 *          order (v41_kvreuse_walk)
 *   tail:  mainh (mainh_bytes), logits (V x 4)
 *
 * The tokens stay out of the payload: the store's text block carries them
 * (LE i32) and the import takes them from the caller, so the restore's
 * memcmp stays the only authority.  A failed import leaves K invalid (never
 * half): the request prefills cold, like a failed save. */

#define V41_KVREUSE_META_ROW_BYTES  12u
#define V41_KVREUSE_META_TAIL_BYTES 28u

static void v41_kv_put_le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static uint32_t v41_kv_get_le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void v41_kv_put_le64(uint8_t *p, uint64_t v) {
    v41_kv_put_le32(p, (uint32_t)v);
    v41_kv_put_le32(p + 4, (uint32_t)(v >> 32));
}

static uint64_t v41_kv_get_le64(const uint8_t *p) {
    return (uint64_t)v41_kv_get_le32(p) | ((uint64_t)v41_kv_get_le32(p + 4) << 32);
}

/* One growable read/write buffer for the spans (never freed, like K). */
static uint8_t *g_v41_kv_span_buf;
static uint64_t g_v41_kv_span_cap;

static uint8_t *v41_kv_span_buf_get(uint64_t need) {
    if (g_v41_kv_span_cap >= need) return g_v41_kv_span_buf;
    free(g_v41_kv_span_buf);
    g_v41_kv_span_buf = xmalloc((size_t)need);
    g_v41_kv_span_cap = need;
    return g_v41_kv_span_buf;
}

/* Walk the checkpoint spans in the file's fixed order so export and import
 * cannot drift; `fn` reads (export) or ensures+reads (import) one span and
 * returns false to abort.  Sizes come from the live K (export) or the
 * just-parsed meta (import). */
static bool v41_kvreuse_walk(const uint32_t *ng_src, const uint32_t *cpend, uint64_t mainh_bytes,
                             bool (*fn)(void *ud, ds4_gpu_tensor **slot, uint64_t bytes), void *ud) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!fn(ud, &K.win[il], (uint64_t)DS4_N_SWA * rowb)) return false;
        if (ng_src[il] &&
            (!fn(ud, &K.ckv[il], (uint64_t)ng_src[il] * DS4_V41_CKV_BYTES) ||
             !fn(ud, &K.idk[il], (uint64_t)ng_src[il] * DS4_V41_IDXK_BYTES))) return false;
        if (cpend[il] &&
            (!fn(ud, &K.cpk[il], (uint64_t)cpend[il] * rowb) ||
             !fn(ud, &K.cps[il], (uint64_t)cpend[il] * rowb))) return false;
    }
    if (!fn(ud, &K.mainh, mainh_bytes)) return false;
    return fn(ud, &K.logits, (uint64_t)DS4_N_VOCAB * 4);
}

static bool v41_kvreuse_export_span(void *ud, ds4_gpu_tensor **slot, uint64_t bytes) {
    FILE *fp = ud;
    if (!bytes) return true;
    if (!*slot) return false;
    uint8_t *buf = v41_kv_span_buf_get(bytes);
    return ds4_gpu_tensor_read(*slot, 0, buf, bytes) != 0 &&
           fwrite(buf, 1, (size_t)bytes, fp) == (size_t)bytes;
}

/* The payload size the meta describes (the import's expected total). */
static uint64_t v41_kvreuse_payload_bytes(const uint32_t *ng_src, const uint32_t *cpend, uint64_t mainh_bytes) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    uint64_t total = (uint64_t)DS4_N_LAYER * V41_KVREUSE_META_ROW_BYTES + V41_KVREUSE_META_TAIL_BYTES;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        total += (uint64_t)DS4_N_SWA * rowb;
        total += (uint64_t)ng_src[il] * (DS4_V41_CKV_BYTES + DS4_V41_IDXK_BYTES);
        if (cpend[il]) total += 2 * (uint64_t)cpend[il] * rowb;
    }
    return total + mainh_bytes + (uint64_t)DS4_N_VOCAB * 4;
}

/* Write the checkpoint payload to a host-named path (the store stages it
 * into the final file).  1 = written; 0 = nothing to write or a failure (the
 * file is removed).  The host calls this only when the in-memory checkpoint
 * is exactly the prompt being served. */
int v41_kvreuse_export(const char *path) {
    if (!g_v41_kv_reuse || !K.valid || !K.n || !path) return 0;
    if (!ds4_gpu_synchronize()) return 0;
    const uint64_t meta = (uint64_t)DS4_N_LAYER * V41_KVREUSE_META_ROW_BYTES + V41_KVREUSE_META_TAIL_BYTES;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    uint8_t *mb = xmalloc((size_t)meta);
    uint8_t *p = mb;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        v41_kv_put_le32(p, K.ng_src[il]); p += 4;
        v41_kv_put_le32(p, K.cpend[il]); p += 4;
        v41_kv_put_le32(p, K.win_from[il]); p += 4;
    }
    v41_kv_put_le64(p, (uint64_t)K.mainh_end); p += 8;
    v41_kv_put_le32(p, K.mainh_n); p += 4;
    v41_kv_put_le64(p, K.mainh_bytes); p += 8;
    v41_kv_put_le64(p, K.bytes); p += 8;
    bool ok = fwrite(mb, 1, (size_t)meta, fp) == (size_t)meta;
    free(mb);
    if (ok) ok = v41_kvreuse_walk(K.ng_src, K.cpend, K.mainh_bytes, v41_kvreuse_export_span, fp);
    if (fclose(fp) != 0) ok = false;
    if (!ok) remove(path);
    return ok ? 1 : 0;
}

static bool v41_kvreuse_import_span(void *ud, ds4_gpu_tensor **slot, uint64_t bytes) {
    FILE *fp = ud;
    if (!bytes) return true;
    if (!v41_kvreuse_ensure(slot, bytes)) return false;
    uint8_t *buf = v41_kv_span_buf_get(bytes);
    if (fread(buf, 1, (size_t)bytes, fp) != (size_t)bytes) return false;
    return ds4_gpu_tensor_write(*slot, 0, buf, bytes) != 0;
}

/* Load a checkpoint payload from (path, offset, bytes) -- the store's
 * payload range -- plus the token block the caller matched (the store's text
 * block).  1 = K now holds the checkpoint; 0 = a miss (K invalid, the
 * request prefills cold).  The mount identity was the caller's gate (the
 * trailer); this re-validates every size against the live configuration. */
int v41_kvreuse_import(const char *path, uint64_t offset, uint64_t bytes,
                       const int32_t *tokens, uint32_t n) {
    if (!g_v41_kv_reuse || !path || !tokens || !n) return 0;
    if (n > g_ds4_v41.ctx) { return 0; }
    const uint64_t meta = (uint64_t)DS4_N_LAYER * V41_KVREUSE_META_ROW_BYTES + V41_KVREUSE_META_TAIL_BYTES;
    if (bytes < meta) return 0;
    uint32_t ng[DS4_MAX_LAYER] = {0}, cp[DS4_MAX_LAYER] = {0}, wf[DS4_MAX_LAYER] = {0};
    uint64_t mainh_end = 0, mainh_bytes = 0, ck_bytes = 0;
    uint32_t mainh_n = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    uint8_t *mb = xmalloc((size_t)meta);
    bool ok = fseeko(fp, (off_t)offset, SEEK_SET) == 0 && fread(mb, 1, (size_t)meta, fp) == (size_t)meta;
    if (ok) {
        const uint8_t *p = mb;
        const uint64_t mainh_bytes_expected = g_ds4_v41.mtp_towers && g_ds4_v41.n_mtp_target
            ? (uint64_t)DS4_N_SWA * g_ds4_v41.n_mtp_target * DS4_N_EMBD * 4u : 0u;
        for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
            ng[il] = v41_kv_get_le32(p); p += 4;
            cp[il] = v41_kv_get_le32(p); p += 4;
            wf[il] = v41_kv_get_le32(p); p += 4;
            /* Source counters cover the whole prompt: CED still compresses
             * its final KV source. Reused layers own no groups or remainder,
             * even when their attention uses a positive compression ratio. */
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            const bool source = g_ds4_v41.is_kv_source[il] != 0;
            const uint32_t ng_expected = source && ratio ? n / ratio : 0u;
            const uint32_t cp_expected = source && ratio > 1u ? n % ratio : 0u;
            if (source && !ratio) { ok = false; }
            if (ng[il] != ng_expected || cp[il] != cp_expected || wf[il] > n) { ok = false; }
        }
        mainh_end = v41_kv_get_le64(p); p += 8;
        mainh_n = v41_kv_get_le32(p); p += 4;
        mainh_bytes = v41_kv_get_le64(p); p += 8;
        ck_bytes = v41_kv_get_le64(p); p += 8;
        /* A saved ring is always complete. Accepting a shorter span can
         * expose stale bytes when K reuses a larger allocation. Its live
         * rows must also describe a run ending at this prompt's frontier. */
        if (mainh_bytes != mainh_bytes_expected || mainh_n > DS4_N_SWA || mainh_n > n) { ok = false; }
        if (!mainh_bytes_expected && mainh_n) { ok = false; }
        if (mainh_bytes_expected &&
            (mainh_n ? mainh_end != (uint64_t)n - 1u : mainh_end != UINT64_MAX)) { ok = false; }
    }
    free(mb);
    if (ok && v41_kvreuse_payload_bytes(ng, cp, mainh_bytes) != bytes) ok = false;
    if (ok) {
        K.valid = 0;   /* a half import must never be adopted */
        ok = v41_kvreuse_walk(ng, cp, mainh_bytes, v41_kvreuse_import_span, fp);
        if (ok && !ds4_gpu_synchronize()) ok = false;
    }
    fclose(fp);
    if (!ok) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        K.ng_src[il] = ng[il]; K.cpend[il] = cp[il]; K.win_from[il] = wf[il];
    }
    K.mainh_end = (int64_t)mainh_end; K.mainh_n = mainh_n;
    K.mainh_bytes = mainh_bytes; K.bytes = ck_bytes;
    if (K.tok_cap < n) { free(K.tok); K.tok_cap = n + n / 2; K.tok = xmalloc((size_t)K.tok_cap * 4); }
    memcpy(K.tok, tokens, (size_t)n * 4);
    K.n = n;
    K.valid = 1;
    fprintf(stderr, "[v41] KV disk checkpoint loaded (%u token, %.1f MB)\n", n, (double)bytes / 1048576.0);
    return 1;
}
