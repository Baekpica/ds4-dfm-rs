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
 * Port addition (the post-generation checkpoint).  The reference keeps ONE
 * checkpoint, taken after the prefill and before decode (:16-19), so the turn
 * that extends the previous prompt by what the model itself said can only ever
 * reuse the previous prompt.  This port adds T: the same state, saved when the
 * run ends (v41_kvreuse_save_turn), so the next turn reuses that block too.
 * T carries the reference's span set minus the logits row (its position is
 * never the prompt's end: the restore requires T.n < np), and its token key is
 * the STATE's positions -- the prompt plus what the state committed, never the
 * emitted ids (a stopping round leaves accepted rows unemitted).  A prompt
 * that does not start with T's key misses it and falls back to K.
 *
 * Port seam (named deviation): the engine memcpy's K.tok into st->hist
 * (:118) because its engram hash reads the token history from the state.  The
 * port's history is host-side (V41Feed/FeedCore.hist in v41_run.rs), so the
 * host seeds it from v41_kvreuse_peek before the run: same content, different
 * seam.  The restore re-runs the memcmp, so the native decision stays the
 * authority.  v41_kvreuse_peek answers with the same checkpoint the restore
 * will take, so the seed covers the reuse.
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

typedef struct {
    int valid;
    uint32_t n;                 /* checkpoint position = that prompt's token count (core_v41_kvreuse.c:32) */
    int32_t *tok; uint32_t tok_cap;
    uint64_t *sig;              /* Image identity per prompt position; zero is text. */
    ds4_gpu_tensor *ckv[DS4_MAX_LAYER], *idk[DS4_MAX_LAYER], *win[DS4_MAX_LAYER],
                   *cpk[DS4_MAX_LAYER], *cps[DS4_MAX_LAYER], *mainh;
    ds4_gpu_tensor *logits;     /* the prompt-end logits row [V] f32: an identical prompt prefills nothing and samples from it.  K's alone -- a turn checkpoint has no prompt end to serve. */
    uint32_t ng_src[DS4_MAX_LAYER], cpend[DS4_MAX_LAYER], win_from[DS4_MAX_LAYER];
    int64_t mainh_end; uint32_t mainh_n;
    uint64_t mainh_bytes;       /* the saved mainh span; the disk meta carries it */
    uint64_t bytes;             /* checkpoint VRAM, the log line's number */
} v41_kv_slot;

/* K: the reference's checkpoint, written with the state exactly at the prompt
 * end and before decode (core_v41_kvreuse.c:16-19).  T: the port addition,
 * written when the run ends (v41_kvreuse_save_turn). */
static v41_kv_slot K, T;

/* Which checkpoint a disk payload carries.  K is the prompt class: the
 * reference's boundary checkpoint, stored under the reasons "cold" and
 * "continued".  T is the turn class: the live frontier the reference stores
 * under "evict" and "shutdown" (server_generate_body1.inc:110,
 * server_kv_cache2.c:151, server_main.c:233).  Only the prompt class carries
 * a logits row -- a turn has no prompt end to serve -- so the class rides the
 * payload meta and the import refuses a payload of the other class. */
typedef enum { V41_KV_CLASS_PROMPT = 0, V41_KV_CLASS_TURN = 1 } v41_kv_class;

/* The slot a class names, or NULL for a class this build does not know: the
 * ABI is an int, so the caller's value is never trusted. */
static v41_kv_slot *v41_kv_class_slot(v41_kv_class cls) {
    switch (cls) {
        case V41_KV_CLASS_PROMPT: return &K;
        case V41_KV_CLASS_TURN: return &T;
    }
    return NULL;
}

static bool v41_kv_class_has_logits(v41_kv_class cls) { return cls == V41_KV_CLASS_PROMPT; }

/* The amplifier identity that computed both checkpoints (port addition, see
 * the header): one per process, not per slot -- the same mounted sidecar
 * computed both, so a change drops both. */
static int g_v41_kv_mount_has; static uint64_t g_v41_kv_mount_hash;

/* The ids the state committed after the prompt, for T.  The state keeps no
 * token history of its own (ds4_v41_state), and the emitted ids are NOT the
 * state's: an earned-token break leaves accepted rows unemitted.  The run
 * appends here where the state advances (ds4_ds41_draft.inc). */
typedef struct { int32_t *ids; uint32_t n, cap; } v41_turn_ids;

static void v41_turn_ids_add(v41_turn_ids *t, const int32_t *ids, uint32_t n) {
    if (!n) return;
    if (t->n + n > t->cap) {   /* speculation commits up to a block a round, and n_predict is no bound */
        const uint32_t cap = t->n + n + 64u;
        int32_t *grown = xmalloc((size_t)cap * 4);
        if (t->n) { memcpy(grown, t->ids, (size_t)t->n * 4); }
        free(t->ids);
        t->ids = grown; t->cap = cap;
    }
    memcpy(t->ids + t->n, ids, (size_t)n * 4);
    t->n += n;
}

static void v41_turn_ids_free(v41_turn_ids *t) { free(t->ids); memset(t, 0, sizeof *t); }

static void v41_kvreuse_drop(void) { K.valid = 0; T.valid = 0; }

/* T's key: the prompt the turn checkpoint was saved with, then the ids the
 * state committed after it (v41_kvreuse_save_turn).  Answers that count and
 * fills `out` when `out` is non-NULL and `cap` covers it; a count above `cap`
 * writes nothing, so the caller sizes the buffer with (NULL, 0) first.  0 =
 * no turn checkpoint (no run left one, or the amplifier changed).  The host
 * needs the key because the store is its own: it writes the disk entry the
 * next process finds, and that key is the store's text. */
uint32_t v41_kvreuse_turn_key(int32_t *out, uint32_t cap) {
    if (!g_v41_kv_reuse || !T.valid || !T.n) return 0;
    if (out && cap >= T.n) { memcpy(out, T.tok, (size_t)T.n * 4); }
    return T.n;
}

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

/* The checkpoint's span set in one place: the state -> the slot (a save) and
 * the slot -> the state (a restore).  It is the reference's own list
 * (core_v41_kvreuse.c:59-84), so K and T carry the same spans by construction
 * -- a span added to one is added to both.  The logits row is not part of it
 * (only a prompt end has one) and each caller does its own counters. */
static uint64_t v41_kv_save_spans(v41_kv_slot *s, const ds4_v41_state *st, bool *ok) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    uint64_t total = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER && *ok; il++) {
        *ok = v41_kvreuse_ensure(&s->win[il], (uint64_t)DS4_N_SWA * rowb) &&
              v41_kvreuse_cp(s->win[il], st->win[il], (uint64_t)DS4_N_SWA * rowb);
        total += (uint64_t)DS4_N_SWA * rowb;
        s->ng_src[il] = st->ng_src[il]; s->cpend[il] = st->cpend[il]; s->win_from[il] = st->win_from[il];
        if (!*ok || !st->comp_kv[il]) continue;
        const uint64_t kb = (uint64_t)st->ng_src[il] * DS4_V41_CKV_BYTES;
        const uint64_t ib = (uint64_t)st->ng_src[il] * DS4_V41_IDXK_BYTES;
        *ok = v41_kvreuse_ensure(&s->ckv[il], kb) && v41_kvreuse_cp(s->ckv[il], st->comp_kv[il], kb) &&
              v41_kvreuse_ensure(&s->idk[il], ib) && v41_kvreuse_cp(s->idk[il], st->index_k[il], ib);
        total += kb + ib;
        if (*ok && st->cpre_kv[il] && st->cpend[il]) {
            const uint64_t pb = (uint64_t)st->cpend[il] * rowb;
            *ok = v41_kvreuse_ensure(&s->cpk[il], pb) && v41_kvreuse_cp(s->cpk[il], st->cpre_kv[il], pb) &&
                  v41_kvreuse_ensure(&s->cps[il], pb) && v41_kvreuse_cp(s->cps[il], st->cpre_sc[il], pb);
            total += 2 * pb;
        }
    }
    s->mainh_bytes = 0;
    if (*ok && st->mainh) {
        const uint64_t mb = ds4_gpu_tensor_bytes(st->mainh);
        *ok = v41_kvreuse_ensure(&s->mainh, mb) && v41_kvreuse_cp(s->mainh, st->mainh, mb);
        s->mainh_bytes = mb;
        total += mb;
    }
    s->mainh_end = st->mainh_end; s->mainh_n = st->mainh_n;
    return total;
}

static bool v41_kv_load_spans(const v41_kv_slot *s, ds4_v41_state *st) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ok = v41_kvreuse_cp(st->win[il], s->win[il], (uint64_t)DS4_N_SWA * rowb);
        if (!ok || !st->comp_kv[il]) continue;
        ok = v41_kvreuse_cp(st->comp_kv[il], s->ckv[il], (uint64_t)s->ng_src[il] * DS4_V41_CKV_BYTES) &&
             v41_kvreuse_cp(st->index_k[il], s->idk[il], (uint64_t)s->ng_src[il] * DS4_V41_IDXK_BYTES);
        if (ok && st->cpre_kv[il] && s->cpend[il])
            ok = v41_kvreuse_cp(st->cpre_kv[il], s->cpk[il], (uint64_t)s->cpend[il] * rowb) &&
                 v41_kvreuse_cp(st->cpre_sc[il], s->cps[il], (uint64_t)s->cpend[il] * rowb);
    }
    if (ok && st->mainh) {
        /* Capacity can include an older checkpoint's tail. Only the saved
         * span belongs to this checkpoint, even when the allocation is larger. */
        ok = s->mainh && s->mainh_bytes >= ds4_gpu_tensor_bytes(st->mainh) &&
             ds4_gpu_tensor_bytes(s->mainh) >= ds4_gpu_tensor_bytes(st->mainh) &&
             v41_kvreuse_cp(st->mainh, s->mainh, ds4_gpu_tensor_bytes(st->mainh));
    }
    return ok;
}

/* The slot -> the state scalars both checkpoints share, in the reference's
 * order (core_v41_kvreuse.c:118-131).  `tag` is empty for K, so K's line stays
 * byte-identical for the harness; T prints `turn `. */
static void v41_kvreuse_adopt(ds4_v41_state *st, const v41_kv_slot *s, uint32_t np, const char *tag) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        st->ng_src[il] = s->ng_src[il]; st->cpend[il] = s->cpend[il]; st->win_from[il] = s->win_from[il];
    }
    if (st->mainh) { st->mainh_end = s->mainh_end; st->mainh_n = s->mainh_n; }
    st->n_past = s->n;
    for (uint32_t i = 0; i < s->n; i++) { st->himg[i] = s->sig[i] != 0; }
    g_v41_kvreuse_last_reused = (int)s->n;
    fprintf(stderr, "[v41] KV reuse %u/%u token (%scheckpoint %.1f MB), prefill suffix %u\n",
            s->n, np, tag, (double)s->bytes / 1048576.0, np - s->n);
}

/* Called with the state exactly at the prompt end, after the prefill and
 * before decode (core_v41_api.c:192).  core_v41_kvreuse.c:59. */
static void v41_kvreuse_save(const ds4_v41_state *st, const int32_t *prompt, uint32_t np,
                            const uint64_t *sig) {
    if (!g_v41_kv_reuse) return;
    K.valid = 0;
    bool ok = true;
    uint64_t total = v41_kv_save_spans(&K, st, &ok);
    const uint64_t vb = (uint64_t)DS4_N_VOCAB * 4;
    if (ok) ok = v41_kvreuse_ensure(&K.logits, vb) && st->logits &&
                 ds4_gpu_tensor_copy(K.logits, 0, st->logits, (uint64_t)st->last_logit_row * vb, vb);
    total += vb;
    if (ok && !ds4_gpu_synchronize()) ok = false;
    if (!ok) { fprintf(stderr, "ds4: [v41] KV checkpoint does not fit (VRAM); the next request prefills whole\n"); return; }
    if (K.tok_cap < np) {
        free(K.tok); free(K.sig);
        K.tok_cap = np + np / 2;
        K.tok = xmalloc((size_t)K.tok_cap * 4);
        K.sig = xmalloc((size_t)K.tok_cap * 8);
    }
    memcpy(K.tok, prompt, (size_t)np * 4);
    if (sig) { memcpy(K.sig, sig, (size_t)np * 8); }
    else { memset(K.sig, 0, (size_t)np * 8); }
    K.n = np; K.bytes = total; K.valid = 1;
}

/* The port addition: the checkpoint the reference does not have.  Called when
 * the run ends, with the state (st) at its last committed position, so the
 * next turn reuses the block the model itself sampled instead of re-prefilling
 * it.  `gen` are the ids the state committed after the prompt (v41_turn_ids, in
 * order); the key is the prompt followed by them.  No logits row: the restore
 * requires T.n < np, so the suffix prefill always writes the head's row. */
static void v41_kvreuse_save_turn(const ds4_v41_state *st, const int32_t *prompt, uint32_t np,
                                  const int32_t *gen, uint32_t ng, const uint64_t *sig) {
    if (!g_v41_kv_reuse || !ng) return;
    T.valid = 0;
    bool ok = true;
    const uint64_t total = v41_kv_save_spans(&T, st, &ok);
    if (ok && !ds4_gpu_synchronize()) ok = false;
    const uint32_t n = np + ng;
    if (!ok) { fprintf(stderr, "ds4: [v41] KV turn checkpoint does not fit (VRAM); the next turn starts at the prompt end\n"); return; }
    if (T.tok_cap < n) {
        free(T.tok); free(T.sig);
        T.tok_cap = n + n / 2;
        T.tok = xmalloc((size_t)T.tok_cap * 4);
        T.sig = xmalloc((size_t)T.tok_cap * 8);
    }
    memcpy(T.tok, prompt, (size_t)np * 4);
    memcpy(T.tok + np, gen, (size_t)ng * 4);
    if (sig) { memcpy(T.sig, sig, (size_t)np * 8); }
    else { memset(T.sig, 0, (size_t)np * 8); }
    memset(T.sig + np, 0, (size_t)ng * 8);   /* a sampled position is text: no image identity */
    T.n = n; T.bytes = total; T.valid = 1;
    fprintf(stderr, "[v41] KV turn checkpoint %u token (%.1f MB)\n", n, (double)total / 1048576.0);
}

/* YoungAi 846c929c core_v41_kvreuse.c:97-100. Compare only the saved
 * prefix: images appended after it must not invalidate otherwise exact KV. */
static bool v41_kv_sig_match(const v41_kv_slot *s, const uint64_t *sig) {
    if (sig) { return memcmp(s->sig, sig, (size_t)s->n * 8) == 0; }
    for (uint32_t i = 0; i < s->n; i++) {
        if (s->sig[i]) { return false; }
    }
    return true;
}

/* Which checkpoint describes this prompt: T first (its position is the previous
 * prompt plus that turn's committed tokens, so its reuse is longer), then K.
 * Both apply the reference's guard minus the copies (core_v41_kvreuse.c:96):
 * the saved prefix must match token for token and carry the same image
 * identity.  T is excluded at equal length -- it has no logits row, and an
 * identical prompt is K's case. */
static const v41_kv_slot *v41_kvreuse_pick(const int32_t *prompt, uint32_t np, const uint64_t *sig) {
    if (T.valid && T.n < np && memcmp(T.tok, prompt, (size_t)T.n * 4) == 0 && v41_kv_sig_match(&T, sig)) return &T;
    if (K.valid && K.n <= np && memcmp(K.tok, prompt, (size_t)K.n * 4) == 0 && v41_kv_sig_match(&K, sig)) return &K;
    return NULL;
}

/* Called on a freshly allocated state, before the chunk loop: when a
 * checkpoint connects, copy it in and return the prefill start (= the
 * checkpoint position; == np means nothing to prefill, the prompt-end logits
 * row is already at row 0).  0 = prefill from token 0.
 * core_v41_kvreuse.c:96. */
static uint32_t v41_kvreuse_restore(ds4_v41_state *st, const int32_t *prompt, uint32_t np,
                                   const uint64_t *sig) {
    g_v41_kvreuse_last_reused = 0;
    if (!g_v41_kv_reuse) return 0;
    const v41_kv_slot *s = v41_kvreuse_pick(prompt, np, sig);
    if (!s || s->n > st->ctx) return 0;
    /* Named deviation from core_v41_kvreuse.c:109 at 7fb26cc: its full hit
     * restores logits but leaves the sampler's position slot uninitialized
     * (cuda_v41_sample.inc.cu:158-160). Refill sampled full hits until the
     * checkpoint carries that state; a suffix writes it through forward.
     * Only K can hit at full length (the pick excludes T there). */
    if (st->dev_sample && s->n == np) {
        fprintf(stderr, "[v41] sampled full-prefix hit: cold prefill for sampler state\n");
        return 0;
    }
    bool ok = v41_kv_load_spans(s, st);
    if (ok && s->n == np) {   /* identical prompt: the logits row lands at row 0, nothing is prefilled */
        ok = v41_kvreuse_cp(st->logits, s->logits, (uint64_t)DS4_N_VOCAB * 4);
        st->last_logit_row = 0;
    }
    if (ok && !ds4_gpu_synchronize()) ok = false;
    if (!ok) {   /* a half copy leaves nothing behind: a cold start rewrites ring, remainder rows and group counts from position 0 */
        fprintf(stderr, "ds4: [v41] KV checkpoint does not copy into the fresh state; this request prefills whole\n");
        return 0;
    }
    v41_kvreuse_adopt(st, s, np, s == &T ? "turn " : "");
    return s->n;
}

/* The restore's read-only guard minus the copies: the prefill start this
 * prompt would get.  The Rust host seeds the engram feed's history with
 * prompt[0..reused) before the run (the engine's memcpy at
 * core_v41_kvreuse.c:118, a port seam).  A stale peek is harmless: the
 * restore re-runs the memcmp and a cold start rewrites the same ranges; the
 * state's ctx fence is not repeated here (it can only make the peek
 * optimistic, and an optimistic seed is an exact prefix of the prompt). */
uint32_t v41_kvreuse_peek(const int32_t *prompt, uint32_t np, const uint64_t *sig) {
    if (!g_v41_kv_reuse) return 0;
    const v41_kv_slot *s = v41_kvreuse_pick(prompt, np, sig);
    return s ? s->n : 0;
}

/* The port's stand-in for the engine's ds4_engine_v41_switch_plugins drop
 * (core_v41_state.c:111): the host reports the mounted amplifier's identity
 * once per run and both checkpoints die when it differs. */
void v41_kvreuse_note_mount(int has_mount, uint64_t hash) {
    if (g_v41_kv_mount_has == has_mount && g_v41_kv_mount_hash == hash) return;
    g_v41_kv_mount_has = has_mount; g_v41_kv_mount_hash = hash;
    if (K.valid || T.valid) fprintf(stderr, "ds4: [v41] amplifier changed; the KV checkpoint is dropped\n");
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
 *          class u8 + 3 reserved                                       4
 *   spans: per layer win SWA rows, ckv ng_src x 288, idk ng_src x 72,
 *          cpk/cps cpend rows each -- the v41_kvreuse_save set, one fixed
 *          order (v41_kvreuse_walk) -- then mainh (mainh_bytes) and, for the
 *          prompt class alone, the logits row (V x 4)
 *   tail:  the image signatures (n x LE u64)
 *
 * One payload carries one class (v41_kv_class).  The class rides the meta, so
 * the import validates the shape it read before it copies anything; the store
 * cannot tell the two apart, and its trailer carries the mount identity.
 *
 * The tokens stay out of the payload: the store's text block carries them
 * (LE i32 + LE u64 image signature) and the import takes tokens from the caller, so the restore's
 * memcmp stays the only authority.  A failed import leaves the class's slot
 * invalid (never half): the request prefills cold, like a failed save. */

#define V41_KVREUSE_META_ROW_BYTES  12u
#define V41_KVREUSE_META_TAIL_BYTES 28u
#define V41_KVREUSE_META_CLASS_BYTES 4u

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
 * returns false to abort.  Sizes come from the class's live slot (export) or
 * the just-parsed meta (import). */
static bool v41_kvreuse_walk(v41_kv_slot *s, const uint32_t *ng_src, const uint32_t *cpend,
                             uint64_t mainh_bytes, v41_kv_class cls,
                             bool (*fn)(void *ud, ds4_gpu_tensor **slot, uint64_t bytes), void *ud) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!fn(ud, &s->win[il], (uint64_t)DS4_N_SWA * rowb)) return false;
        if (ng_src[il] &&
            (!fn(ud, &s->ckv[il], (uint64_t)ng_src[il] * DS4_V41_CKV_BYTES) ||
             !fn(ud, &s->idk[il], (uint64_t)ng_src[il] * DS4_V41_IDXK_BYTES))) return false;
        if (cpend[il] &&
            (!fn(ud, &s->cpk[il], (uint64_t)cpend[il] * rowb) ||
             !fn(ud, &s->cps[il], (uint64_t)cpend[il] * rowb))) return false;
    }
    if (!fn(ud, &s->mainh, mainh_bytes)) return false;
    return !v41_kv_class_has_logits(cls) || fn(ud, &s->logits, (uint64_t)DS4_N_VOCAB * 4);
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
static uint64_t v41_kvreuse_meta_bytes(void) {
    return (uint64_t)DS4_N_LAYER * V41_KVREUSE_META_ROW_BYTES + V41_KVREUSE_META_TAIL_BYTES +
           V41_KVREUSE_META_CLASS_BYTES;
}

static uint64_t v41_kvreuse_payload_bytes(const uint32_t *ng_src, const uint32_t *cpend,
                                          uint64_t mainh_bytes, v41_kv_class cls) {
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4;
    uint64_t total = v41_kvreuse_meta_bytes();
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        total += (uint64_t)DS4_N_SWA * rowb;
        total += (uint64_t)ng_src[il] * (DS4_V41_CKV_BYTES + DS4_V41_IDXK_BYTES);
        if (cpend[il]) total += 2 * (uint64_t)cpend[il] * rowb;
    }
    total += mainh_bytes;
    if (v41_kv_class_has_logits(cls)) total += (uint64_t)DS4_N_VOCAB * 4;
    return total;
}

/* Write the named class's checkpoint payload to a host-named path (the store
 * stages it into the final file).  1 = written; 0 = nothing to write, an
 * unknown class, or a failure (the file is removed).  The host calls this
 * only for a checkpoint it wants on disk: the prompt class when the
 * in-memory checkpoint is exactly the prompt being served, the turn class at
 * the end of a run. */
int v41_kvreuse_export(const char *path, int cls) {
    const v41_kv_class kind = (v41_kv_class)cls;
    v41_kv_slot *s = v41_kv_class_slot(kind);
    if (!g_v41_kv_reuse || !s || !s->valid || !s->n || !path) return 0;
    if (!ds4_gpu_synchronize()) return 0;
    const uint64_t meta = v41_kvreuse_meta_bytes();
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    uint8_t *mb = xmalloc((size_t)meta);
    uint8_t *p = mb;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        v41_kv_put_le32(p, s->ng_src[il]); p += 4;
        v41_kv_put_le32(p, s->cpend[il]); p += 4;
        v41_kv_put_le32(p, s->win_from[il]); p += 4;
    }
    v41_kv_put_le64(p, (uint64_t)s->mainh_end); p += 8;
    v41_kv_put_le32(p, s->mainh_n); p += 4;
    v41_kv_put_le64(p, s->mainh_bytes); p += 8;
    v41_kv_put_le64(p, s->bytes); p += 8;
    v41_kv_put_le32(p, (uint32_t)kind); p += 4;   /* the 3 reserved bytes write as zero */
    bool ok = fwrite(mb, 1, (size_t)meta, fp) == (size_t)meta;
    free(mb);
    if (ok) ok = v41_kvreuse_walk(s, s->ng_src, s->cpend, s->mainh_bytes, kind,
                                  v41_kvreuse_export_span, fp);
    /* Persist the signatures with the checkpoint, never reconstruct them
     * from the incoming request. The Rust trailer marks payload ABI 3. */
    for (uint32_t i = 0; ok && i < s->n; i++) {
        uint8_t raw[8];
        v41_kv_put_le64(raw, s->sig[i]);
        ok = fwrite(raw, 1, sizeof raw, fp) == sizeof raw;
    }
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
 * block).  The payload names its own class, so the import lands it in that
 * class's slot: the store selects a candidate by its token prefix alone and
 * does not know which shape it holds.  1 = that slot now holds the
 * checkpoint; 0 = a miss (nothing written, the request prefills cold).  The
 * mount identity was the caller's gate (the trailer); this re-validates the
 * class and every size against the live configuration. */
int v41_kvreuse_import(const char *path, uint64_t offset, uint64_t bytes,
                       const int32_t *tokens, uint32_t n) {
    if (!g_v41_kv_reuse || !path || !tokens || !n) return 0;
    if (n > g_ds4_v41.ctx) { return 0; }
    const uint64_t meta = v41_kvreuse_meta_bytes();
    if (bytes < meta) return 0;
    uint32_t ng[DS4_MAX_LAYER] = {0}, cp[DS4_MAX_LAYER] = {0}, wf[DS4_MAX_LAYER] = {0};
    uint64_t mainh_end = 0, mainh_bytes = 0, ck_bytes = 0;
    uint32_t mainh_n = 0;
    v41_kv_class kind = V41_KV_CLASS_PROMPT;   /* replaced by the payload's own */
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
        /* Only a class this build knows may land: a turn's KV belongs in the
         * turn slot (its restore has no logits row to serve) and a prompt's in
         * the prompt slot, and the two have different span sets. */
        kind = (v41_kv_class)v41_kv_get_le32(p);
        /* A saved ring is always complete. Accepting a shorter span can
         * expose stale bytes when the slot reuses a larger allocation. Its
         * live rows must also describe a run ending at this prompt's frontier. */
        if (mainh_bytes != mainh_bytes_expected || mainh_n > DS4_N_SWA || mainh_n > n) { ok = false; }
        if (!mainh_bytes_expected && mainh_n) { ok = false; }
        if (mainh_bytes_expected &&
            (mainh_n ? mainh_end != (uint64_t)n - 1u : mainh_end != UINT64_MAX)) { ok = false; }
    }
    free(mb);
    v41_kv_slot *s = v41_kv_class_slot(kind);
    if (ok && !s) { ok = false; }
    if (ok && v41_kvreuse_payload_bytes(ng, cp, mainh_bytes, kind) + (uint64_t)n * 8 != bytes) ok = false;
    if (ok) {
        s->valid = 0;   /* a half import must never be adopted */
        ok = v41_kvreuse_walk(s, ng, cp, mainh_bytes, kind, v41_kvreuse_import_span, fp);
        if (ok && !ds4_gpu_synchronize()) ok = false;
    }
    uint64_t *sig = NULL;
    if (ok) {
        sig = xmalloc((size_t)n * 8);
        for (uint32_t i = 0; ok && i < n; i++) {
            uint8_t raw[8];
            ok = fread(raw, 1, sizeof raw, fp) == sizeof raw;
            if (ok) { sig[i] = v41_kv_get_le64(raw); }
        }
    }
    fclose(fp);
    if (!ok) { free(sig); return 0; }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        s->ng_src[il] = ng[il]; s->cpend[il] = cp[il]; s->win_from[il] = wf[il];
    }
    s->mainh_end = (int64_t)mainh_end; s->mainh_n = mainh_n;
    s->mainh_bytes = mainh_bytes; s->bytes = ck_bytes;
    if (s->tok_cap < n) {
        free(s->tok); free(s->sig);
        s->tok_cap = n + n / 2;
        s->tok = xmalloc((size_t)s->tok_cap * 4);
        s->sig = xmalloc((size_t)s->tok_cap * 8);
    }
    memcpy(s->tok, tokens, (size_t)n * 4);
    memcpy(s->sig, sig, (size_t)n * 8);
    free(sig);
    s->n = n;
    s->valid = 1;
    fprintf(stderr, "[v41] KV disk %scheckpoint loaded (%u token, %.1f MB)\n",
            v41_kv_class_has_logits(kind) ? "" : "turn ", n, (double)bytes / 1048576.0);
    return 1;
}

/* Port lifetime addition to YoungAi core_v41_kvreuse.c:31-43: dropping a
 * checkpoint only invalidates it. Model close must also release its buffers
 * before the GPU backend closes, then forget the previous model identity. */
static void v41_kv_release(v41_kv_slot *s) {
    s->valid = 0;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor *slots[] = {s->ckv[il], s->idk[il], s->win[il], s->cpk[il], s->cps[il]};
        for (size_t i = 0; i < sizeof slots / sizeof slots[0]; i++) {
            if (slots[i]) { ds4_gpu_tensor_free(slots[i]); }
        }
    }
    if (s->mainh) { ds4_gpu_tensor_free(s->mainh); }
    if (s->logits) { ds4_gpu_tensor_free(s->logits); }
    free(s->tok);
    free(s->sig);
    memset(s, 0, sizeof *s);
}

static void v41_kvreuse_cleanup(void) {
    v41_kv_release(&K);
    v41_kv_release(&T);
    g_v41_kv_mount_has = 0; g_v41_kv_mount_hash = 0;   /* forget the previous model identity */
    free(g_v41_kv_span_buf);
    g_v41_kv_span_buf = NULL;
    g_v41_kv_span_cap = 0;
    g_v41_kvreuse_last_reused = 0;
}
