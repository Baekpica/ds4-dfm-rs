/* ds4_ds41_sample.inc — the V4.1 "pick the next token" fork (unit H), ported
 * from the C engine at /data/YoungAi (commit 3946dbc): src/core/core_v41_sample.c
 * (device sample / device argmax / host penalty route) and
 * src/core/core_decode_penalty.c (freq / presence / DRY), plus the sampling
 * face from src/core/core_v41_api.c:10-13.  Included by ds4.c between
 * ds4_ds41_forward.inc and ds4_ds41_graph.inc (the graph's wait and the
 * generate loop call these).
 *
 * Why its own file (engine header comment): after 2026-09-28 made sampling the
 * default (temperature 1.0, the model-card recipe), a request without a
 * temperature takes the sampling route; the engine's measured split on a real
 * request: greedy pure decode 29.24 t/s / sampling pure decode 28.92 /
 * greedy + speculation 42.49 -- host sampling itself costs only 0.32 ms/step
 * (1%), what is really lost is speculation (-32%).  Speculation under
 * sampling requires rejection sampling (accept iff uniform < the probability
 * the target distribution gives the draft; reject draws from the residual),
 * and that step's numerator and denominator both live on the device, so the
 * pick moved into the device kernel (cuda/ds41_sample.cuh), the host reads
 * 16 bytes/row -- the same slot, the same readback loop as the greedy argmax.
 *
 * The three routes (decided in ds4_ds41_draft.inc's generate entry, executed
 * here):
 *   temperature <= 0 and no penalties -> device argmax (the old path, byte for
 *     byte; the gate is "temperature 0 output is byte-identical");
 *   temperature > 0 and no penalties -> the device sampling kernel
 *     (st->dev_sample), speculation runs;
 *   any penalty non-zero -> read the whole row back, penalize on the host,
 *     hand it to the V4 route's own sampler (ds4_sample_logits; at
 *     temperature 0 it yields argmax).  Penalties read the token history, so
 *     speculation does not take them.
 * What happens on error: a mis-wired want (accept flag filled with the
 * residual or vice versa) forks the speculative and pure-decode
 * distributions; under sampling that is not visible byte-for-byte, so the
 * gate is the kernel's marginal-distribution selftest
 * (tests/test_ds41_sample.cu, the engine's tests/cuda_sample_selftest.c) and
 * the greedy route's byte-for-byte gate. */
#ifndef DS4_DS41_SAMPLE_INC
#define DS4_DS41_SAMPLE_INC

/* The engine's per-request sampling face (core_v41_api.c:10-13).  The engine
 * calls this once per request before the generate (server_generate_v41.c:409,
 * cli_diag.c:50); NULL resets to the all-zero face = bare argmax. */
static ds4_decode_sampling g_decode_sampling;
void ds4_engine_set_decode_sampling(const ds4_decode_sampling *sp) {
    if (sp) g_decode_sampling = *sp;
    else memset(&g_decode_sampling, 0, sizeof g_decode_sampling);
}

/* ---- core_decode_penalty.c (2026-09-21, 113-1.md §4.2) ---------------------
 * Only entered when the request explicitly carries a penalty parameter; the
 * default routes (bare argmax / pure sampling) never come through here -- the
 * iron rule "the engine must not alter the model's output".
 * Scope = the generated segment (prompt excluded): quoted source text is
 * never penalized (the V4 route's repeat_penalize_buf uses the same scope).
 *
 * (1) Frequency / presence penalty (OpenAI semantics, byte-for-byte the same
 *     formula as the V4 route): logits[t] -= freq*count(t) + (count(t)>0 ?
 *     presence : 0).  It counts tokens, so it stops "the same word repeating"
 *     but not "a whole passage copied verbatim" (each copied token alone
 *     appears only once or twice).
 * (2) DRY (Don't Repeat Yourself, llama.cpp's): for every candidate t, find
 *     the longest already-seen sequence L that "the current suffix + t"
 *     continues (= the length ending at an earlier t whose window matches the
 *     current suffix token for token); when L >= allowed, logits[t] -=
 *     mult*base^(L-allowed).  The longer the copy, the harder the exponential
 *     penalty; short legitimate repeats (numbers / names / terms, L 1~2) are
 *     untouched; sequence breakers (token text containing newline, colon,
 *     quote, star) cut the match, so structural list/paragraph repetition is
 *     not treated as copying.  WORKS AT TEMPERATURE 0 TOO (it changes
 *     logits, argmax sees it) => it is the only way to have "deterministic
 *     direct output + no loops".
 *
 * Why this algorithm and not z-algorithm: n <= a few thousand, normal text
 * has single-digit L, a step's tens of thousands of comparisons are
 * negligible; a looping region's L grows into the hundreds, which is exactly
 * when it must be killed, and the DRY_MAX_L cap bounds the worst case at
 * n * 512 comparisons (~0.5 ms) -- the penalty long since pushed that token
 * to -1e15.  What happens on error: a wrong breaker table (all 0) = matches
 * continue across paragraph boundaries, list formatting gets over-penalized
 * and the output format collapses; all 1 = no matches at all, DRY silently
 * dead. */
#define DRY_MAX_L 512u          /* backwards match cap; base^(512) is astronomical already */
#define DRY_MAX_EXP 64          /* exponent cap: 1.75^64 ~ 3e15, finite but equal to disabling the token; uncapped it becomes inf */

static void ds4_decode_penalize(float *logits, uint32_t n_vocab, const int32_t *gen, uint32_t n, const uint8_t *breaker,
                                float freq, float presence, float dry_mult, float dry_base, int dry_allowed) {
    if (!logits || !gen || n == 0) return;
    if (freq != 0.f || presence != 0.f) {
        /* The V4 route's same two-pass walk: the first pass subtracts freq per
         * occurrence and presence once at the first occurrence; the second
         * clears the markers (O(n), never touches the whole vocab). */
        static uint8_t *seen = NULL;
        if (!seen) seen = xcalloc((size_t)n_vocab, 1);
        for (uint32_t i = 0; i < n; i++) {
            const int32_t t = gen[i];
            if (t < 0 || (uint32_t)t >= n_vocab) continue;
            logits[t] -= freq;
            if (!seen[t]) { seen[t] = 1; logits[t] -= presence; }
        }
        for (uint32_t i = 0; i < n; i++) if (gen[i] >= 0 && (uint32_t)gen[i] < n_vocab) seen[gen[i]] = 0;
    }
    if (dry_mult <= 0.f || dry_base <= 1.f || dry_allowed < 1 || n < 2 || !breaker) return;
    /* best[t] = candidate t's longest match L; only touched entries are
     * non-zero, and the touched list clears them afterwards (no memset of the
     * whole vocab). */
    static uint32_t *best = NULL; static int32_t *touched = NULL; static uint32_t ntouched = 0;
    if (!best) { best = xcalloc((size_t)n_vocab, sizeof(uint32_t)); touched = xmalloc((size_t)n_vocab * sizeof(int32_t)); }
    ntouched = 0;
    const int32_t last = gen[n - 1];
    if (last < 0 || (uint32_t)last >= n_vocab || breaker[last]) return;   /* the current suffix ends on a breaker: match length is 0, nothing to penalize */
    /* Candidate t = gen[j] (j may reach n-1: "the same token repeating"); the
     * gen[j-1] before it must equal the current last token for L >= 1; then
     * compare backwards gen[j-1-L] vs gen[n-1-L] (the two windows may overlap,
     * which is what periodic text is), stopping at a breaker or the start. */
    for (uint32_t j = 1; j < n; j++) {
        if (gen[j - 1] != last) continue;
        const int32_t t = gen[j];
        if (t < 0 || (uint32_t)t >= n_vocab || breaker[t]) continue;   /* the candidate itself is a breaker: not penalized (llama.cpp's convention) */
        uint32_t L = 1;
        while (L < DRY_MAX_L && j - 1 >= L && gen[j - 1 - L] == gen[n - 1 - L] && !breaker[gen[n - 1 - L]]) L++;
        if (best[t] == 0) touched[ntouched++] = t;
        if (L > best[t]) best[t] = L;
    }
    for (uint32_t k = 0; k < ntouched; k++) {
        const int32_t t = touched[k];
        const uint32_t L = best[t]; best[t] = 0;
        if (L < (uint32_t)dry_allowed) continue;
        int e = (int)(L - (uint32_t)dry_allowed); if (e > DRY_MAX_EXP) e = DRY_MAX_EXP;
        logits[t] -= dry_mult * powf(dry_base, (float)e);
    }
}

/* The breaker table: per token, whether its text contains newline / quote /
 * star.  llama.cpp's default set also has the colon; deliberately dropped
 * here -- the engine's 2026-09-21 third probe batch: with the colon (fullwidth
 * ":" included) as a breaker the model found the loophole -- copying
 * "Kweichow Moutai H1 2024: revenue? net profit?" line by line, the colon cut
 * every line into two short matches (<= 6 tokens, under 8 nat of penalty),
 * which could not beat the looping region's 9~10 nat of confidence, and the
 * copy continued at 100%.  Without the colon the whole line is one match
 * (~12 tokens), penalized 1.75^10 ~ 200, and the copy flips at the 7th token.
 * Newline stays a breaker (or fixed list/table formats would be
 * over-penalized), so <= 6-token short-line loops are still not stopped --
 * that is what the frequency penalty is for, a separate knob.
 * Built once (129k tokens, each decoded once, tens of ms), reused in-process. */
static uint8_t *ds4_decode_breakers(ds4_engine *e, uint32_t n_vocab) {
    uint8_t *b = xcalloc((size_t)n_vocab, 1);
    for (uint32_t t = 0; t < n_vocab; t++) {
        size_t len = 0;
        char *s = ds4_token_text(e, (int)t, &len);
        if (!s) continue;
        for (size_t i = 0; i < len; i++) {
            const unsigned char c = (unsigned char)s[i];
            if (c == '\n' || c == '"' || c == '*') { b[t] = 1; break; }
        }
        free(s);
    }
    return b;
}

/* ---- core_v41_sample.c: the pick and the two device readback arms ---------- */

/* Assemble want[] from the sampled slots: with dev_sample and a draft present
 * (i+1 < n), slot[1] is the accept flag -- accepted takes the draft (batch
 * [i+1], the verify batch's next row's input), rejected takes the residual
 * sample slot[2]; otherwise (pure decode, or the greedy route) slot[0] is the
 * pick.  The engine's expression, core_v41_sample.c:17-22.
 * The tool-syntax greedy hook (2026-10-11, the engine's c6e2706): under
 * sampling, ask per position "is the next position tool-call syntax?" -- yes
 * takes the raw argmax (the kernel's slot [3]); the question walks only the
 * accepted prefix (the caller's text + batch[1..i]) and stops asking once the
 * draft is not accepted (or at the last row).  argmax (temperature 0) never
 * asks: its pick is already [0]. */
static void v41_sample_pick(ds4_v41_state *st, const int32_t *slot, const int32_t *batch, uint32_t n, int32_t *want) {
    const ds4_decode_sampling *sp = &g_decode_sampling;
    bool ask = st->dev_sample && sp->greedy_fn;
    for (uint32_t i = 0; i < n; i++) {
        const int32_t *s = slot + 4u * i;
        if (ask && sp->greedy_fn(sp->greedy_ud, batch ? batch + 1 : NULL, i)) want[i] = s[3];
        else want[i] = (st->dev_sample && batch && i + 1u < n) ? (s[1] ? batch[i + 1u] : s[2]) : s[0];
        if (ask && !(batch && i + 1u < n && want[i] == batch[i + 1u])) ask = false;
    }
}

/* Device pick for a verify batch or a single row (core_v41_sample.c:24-37):
 * sampling on => one kernel launch over the n rows (full samples, accept
 * flags, residuals, 16 bytes each) then one readback; argmax => the argmax
 * kernel writes only its slot's first int, so the rows are argmaxed and read
 * back one at a time (the direct path only pays this on verify batches). */
static bool v41_device_next(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row0, uint32_t n, const int32_t *batch, int32_t *want) {
    int32_t slot[(DS4_MTP_MAX_BLOCK + 2u) * 4u];
    if (!am || n == 0u || n > DS4_MTP_MAX_BLOCK + 2u) return false;   /* the kernel's own entry checks 16*n bytes against out->bytes */
    if (st->dev_sample) {
        if (!ds4_gpu_v41_sample_tensor(am, st->logits, row0, n, DS4_N_VOCAB, st->pos, st->tok, &st->samp, n > 1u ? st->spec_q : NULL) ||
            !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(am, 0, slot, (uint64_t)n * 16u)) return false;
    } else {
        for (uint32_t i = 0; i < n; i++) {
            slot[4u * i + 1u] = slot[4u * i + 2u] = slot[4u * i + 3u] = 0;
            if (!ds4_gpu_v41_argmax_tensor(am, st->logits, row0 + i, DS4_N_VOCAB) || !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(am, 0, &slot[4u * i], 4)) return false;
        }
    }
    v41_sample_pick(st, slot, batch, n, want);
    return true;
}

/* The host penalty route (core_v41_sample.c:39-54): read the row back, apply
 * the history penalties, hand it to the V4 sampler.  rowbuf non-NULL = this
 * request takes the route (any penalty non-zero).  The engine reads the
 * per-request face from st->psamp when the batch path set one
 * (core_v41.h:251); the port has no batch path, so the global face is the
 * only source.  At temperature 0 the sampler yields argmax -- the "read back,
 * penalize, argmax" route the header describes. */
static bool v41_next_token(ds4_v41_state *st, ds4_gpu_tensor *am, uint32_t row, float *rowbuf, uint64_t *rng, v41_hist *h, int32_t *out) {
    if (!rowbuf) return v41_device_next(st, am, row, 1u, NULL, out);
    const ds4_decode_sampling *sp = &g_decode_sampling;
    if (!ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(st->logits, (uint64_t)row * DS4_N_VOCAB * 4u, rowbuf, (uint64_t)DS4_N_VOCAB * 4u)) return false;
    if (h && h->n && (sp->dry_multiplier > 0.f || sp->freq_penalty != 0.f || sp->presence_penalty != 0.f))
        ds4_decode_penalize(rowbuf, DS4_N_VOCAB, h->tok, h->n, h->brk, sp->freq_penalty, sp->presence_penalty,
                            sp->dry_multiplier, sp->dry_base, sp->dry_allowed_length);
    /* The penalty route asks the same question as the pick (the engine's
     * v41_next_token, c6e2706): at a tool-syntax position the sampler is asked
     * with temperature 0, so the post-penalty argmax is the token. */
    const float temp = (sp->greedy_fn && sp->greedy_fn(sp->greedy_ud, NULL, 0)) ? 0.f : sp->temperature;
    *out = (int32_t)ds4_sample_logits(rowbuf, (int)DS4_N_VOCAB, temp, sp->top_k, sp->top_p, sp->min_p, rng);
    if (h && h->n < h->cap) h->tok[h->n++] = *out;
    return true;
}

#endif /* DS4_DS41_SAMPLE_INC */
