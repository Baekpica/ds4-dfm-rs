/* Full-model dots3 predictor, accepted-state and snapshot parity. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "dots3 MTP FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
enum { DEFAULT_PROMPT_ROWS = 600, VERIFY_ROWS = 4, MAX_TEST_CONTEXT = 4096, TAIL_CONTEXT = 32 };

static void logits_equal(ds4_session *a, ds4_session *b) {
    double largest = 0;
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        CHECK(isfinite(a->logits[i]) && isfinite(b->logits[i]));
        const double d = fabs((double)a->logits[i] - b->logits[i]);
        if (d > largest) { largest = d; }
    }
    fprintf(stderr, "dots3 MTP target full-vocab max_abs=%.9g\n", largest);
    CHECK(largest == 0);
}

static void tensor_equal(ds4_gpu_tensor *a, uint64_t ao, ds4_gpu_tensor *b,
                         uint64_t bo, uint64_t bytes) {
    uint8_t *left = xmalloc(bytes), *right = xmalloc(bytes);
    CHECK(ds4_gpu_tensor_read(a, ao, left, bytes));
    CHECK(ds4_gpu_tensor_read(b, bo, right, bytes));
    CHECK(!memcmp(left, right, bytes));
    free(left); free(right);
}

static void cache_equal(ds4_dots3_gpu_graph *a, ds4_dots3_gpu_graph *b) {
    CHECK(a->cache_len == b->cache_len);
    for (unsigned il = 0; il < DS4_N_LAYER; il++) {
        if (!a->layer_kv_latent[il]) { CHECK(!b->layer_kv_latent[il]); continue; }
        CHECK(a->layer_cache_cap[il] == b->layer_cache_cap[il]);
        const unsigned n = a->cache_len < a->layer_cache_cap[il]
            ? a->cache_len : a->layer_cache_cap[il];
        tensor_equal(a->layer_kv_latent[il], 0, b->layer_kv_latent[il], 0,
                     (uint64_t)n * ds4_dots3_layer_kv_lora(il) * sizeof(uint16_t));
        tensor_equal(a->layer_k_pe[il], 0, b->layer_k_pe[il], 0,
                     (uint64_t)n * DS4_N_ROT * sizeof(uint16_t));
        if (a->layer_idx_k[il]) {
            tensor_equal(a->layer_idx_k[il], 0, b->layer_idx_k[il], 0,
                         (uint64_t)n * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        }
    }
}

static void state_equal(ds4_session *a, ds4_session *b) {
    logits_equal(a, b);
    CHECK(a->dots3_spec->target_pos == b->dots3_spec->target_pos);
    CHECK(!a->dots3_spec->trial_n && !b->dots3_spec->trial_n);
    cache_equal(&a->dots3_graph, &b->dots3_graph);
    cache_equal(&a->dots3_spec->draft, &b->dots3_spec->draft);
    tensor_equal(a->dots3_spec->carry, 0, b->dots3_spec->carry, 0,
                 (uint64_t)DS4_N_EMBD * sizeof(float));
}

/* Compare discarded physical writes against the pre-trial journal. This also
 * covers full-attention/DSA rows outside the live prefix. */
static void rejected_equal(ds4_session *s, unsigned base, unsigned keep, unsigned trial_n) {
    uint64_t off = 0;
    for (unsigned il = 0; il < DS4_N_LAYER; il++) {
        ds4_dots3_gpu_graph *g = il + 1 == DS4_N_LAYER ? &s->dots3_spec->draft : &s->dots3_graph;
        const unsigned pos = il + 1 == DS4_N_LAYER ? base - 1 : base;
        ds4_gpu_tensor *parts[] = {g->layer_kv_latent[il], g->layer_k_pe[il], g->layer_idx_k[il]};
        const uint64_t widths[] = {ds4_dots3_layer_kv_lora(il) * sizeof(uint16_t),
            DS4_N_ROT * sizeof(uint16_t), DS4_N_INDEXER_HEAD_DIM * sizeof(float)};
        for (unsigned part = 0; part < 3; part++) {
            if (!parts[part]) { continue; }
            for (unsigned i = keep; i < trial_n; i++) {
                tensor_equal(parts[part], (uint64_t)((pos + i) % g->layer_cache_cap[il]) * widths[part],
                             s->dots3_spec->journal, off + i * widths[part], widths[part]);
            }
            off += VERIFY_ROWS * widths[part];
        }
    }
}

static void short_trial_equal(ds4_session *plain, ds4_session *spec,
                              ds4_session *teacher, int budget, int expected, int keep) {
    char err[256] = {0};
    int tokens[VERIFY_ROWS], targets[VERIFY_ROWS];
    for (unsigned i = 0; i < VERIFY_ROWS; i++) { tokens[i] = targets[i] = -1; }
    const unsigned base = (unsigned)ds4_session_pos(spec);
    const uint64_t generation = spec->generation;
    const int first = ds4_session_argmax(plain);
    CHECK(first >= 0 && ds4_session_argmax(spec) == first);
    const int n = ds4_session_dots3_trial(spec, first, budget, tokens, targets,
                                         VERIFY_ROWS, err, sizeof(err));
    CHECK(n == expected && keep > 0 && keep <= n && tokens[0] == first);
    CHECK(ds4_session_pos(spec) == (int)base);
    for (int i = n; i < VERIFY_ROWS; i++) { CHECK(tokens[i] == -1 && targets[i] == -1); }
    for (int i = 0; i < keep; i++) {
        CHECK(ds4_session_eval(plain, tokens[i], err, sizeof(err)) == 0);
        CHECK(ds4_session_eval(teacher, tokens[i], err, sizeof(err)) == 0);
        CHECK(ds4_session_argmax(plain) == targets[i]);
    }
    CHECK(ds4_session_dots3_commit(spec, keep, err, sizeof(err)) == 0);
    CHECK(ds4_session_pos(spec) == (int)base + keep && spec->generation == generation);
    logits_equal(plain, spec);
    cache_equal(&plain->dots3_graph, &spec->dots3_graph);
    state_equal(teacher, spec);
    rejected_equal(spec, base, (unsigned)keep, (unsigned)n);
    fprintf(stderr, "dots3 MTP short trial: budget=%d rows=%d keep=%d position=%d PASS\n",
            budget, n, keep, ds4_session_pos(spec));
}

static void context_tail(ds4_engine *engine, const ds4_tokens *words) {
    /* Real small-context graphs exercise the end fence without another long
     * prefill. Existing 600/2304-row cases cover wrapped SWA and DSA state. */
    ds4_session *plain = NULL, *spec = NULL, *teacher = NULL;
    setenv("DS4_SESSION_LAZY_GRAPH", "0", 1);
    setenv("DS4_DOTS3_MTP", "0", 1);
    CHECK(ds4_session_create(&plain, engine, TAIL_CONTEXT) == 0);
    setenv("DS4_DOTS3_MTP", "1", 1);
    CHECK(ds4_session_create(&spec, engine, TAIL_CONTEXT) == 0);
    CHECK(ds4_session_create(&teacher, engine, TAIL_CONTEXT) == 0);
    CHECK(plain->dots3_graph.ctx_cap == TAIL_CONTEXT);
    CHECK(spec->dots3_graph.ctx_cap == TAIL_CONTEXT && spec->dots3_spec->draft.ctx_cap == TAIL_CONTEXT);
    CHECK(teacher->dots3_graph.ctx_cap == TAIL_CONTEXT && teacher->dots3_spec->draft.ctx_cap == TAIL_CONTEXT);
    ds4_tokens prompt = {0};
    for (unsigned i = 0; i < TAIL_CONTEXT - (VERIFY_ROWS - 1); i++) {
        token_vec_push(&prompt, words->v[i % (unsigned)words->len]);
    }
    char err[256] = {0};
    CHECK(ds4_session_sync(plain, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_sync(spec, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_sync(teacher, &prompt, err, sizeof(err)) == 0);
    for (int remaining = VERIFY_ROWS - 1; remaining > 0; remaining--) {
        CHECK(ds4_session_pos(spec) == TAIL_CONTEXT - remaining);
        short_trial_equal(plain, spec, teacher, VERIFY_ROWS, remaining, 1);
    }
    const uint64_t generation = spec->generation;
    const int first = ds4_session_argmax(spec);
    int tokens[VERIFY_ROWS], targets[VERIFY_ROWS];
    CHECK(first >= 0 && ds4_session_pos(spec) == TAIL_CONTEXT);
    CHECK(ds4_session_dots3_trial(spec, first, VERIFY_ROWS, tokens, targets,
                                  VERIFY_ROWS, err, sizeof(err)) < 0);
    CHECK(ds4_session_eval(spec, first, err, sizeof(err)) != 0);
    CHECK(spec->generation == generation && spec->checkpoint_valid);
    CHECK(ds4_session_pos(spec) == TAIL_CONTEXT);
    logits_equal(plain, spec);
    cache_equal(&plain->dots3_graph, &spec->dots3_graph);
    state_equal(teacher, spec);
    ds4_session_free(teacher); ds4_session_free(spec); ds4_session_free(plain);
    ds4_tokens_free(&prompt);
    fprintf(stderr, "dots3 MTP real context=%d exhausted without mutation PASS\n", TAIL_CONTEXT);
}

/* Independent one-row predictor oracle. At position zero softmax has one key,
 * so the expanded V projection is the attention result. This isolates the
 * new own-embedding/concat/projection/norm/dense/shared-head contract from
 * the existing, separately qualified target attention implementation. */
static void predictor_reference(ds4_engine *e, ds4_dots3_spec *sp, int token) {
    const ds4_model *m = &e->model;
    const ds4_layer_weights *l = &e->weights.layer[DS4_N_LAYER - 1];
    const unsigned h = DS4_N_EMBD, heads = DS4_N_SWA_HEAD;
    const unsigned nope = DS4_N_SWA_KEY_MLA - DS4_N_ROT;
    const unsigned values = heads * DS4_N_VALUE_MLA;
    const unsigned scratch_n = DS4_N_FF_DENSE > 2 * h ? DS4_N_FF_DENSE : 2 * h;
    float *previous = xmalloc(h * sizeof(float)), *embed = xmalloc(h * sizeof(float));
    float *joined = xmalloc(2 * h * sizeof(float)), *x = xmalloc(h * sizeof(float));
    float *norm = xmalloc(h * sizeof(float)), *nw = xmalloc(h * sizeof(float));
    float *scratch = xmalloc(scratch_n * sizeof(float));
    float *kv = xmalloc((DS4_N_SWA_KV_LORA + DS4_N_ROT) * sizeof(float));
    float *expanded = xmalloc(heads * (nope + DS4_N_VALUE_MLA) * sizeof(float));
    float *attn = xmalloc(values * sizeof(float)), *gate = xmalloc(heads * sizeof(float));
    float *out = xmalloc(h * sizeof(float));
    float *a = xmalloc(DS4_N_FF_DENSE * sizeof(float)), *b = xmalloc(DS4_N_FF_DENSE * sizeof(float));
    float *expected = xmalloc(DS4_N_VOCAB * sizeof(float));
    for (unsigned i = 0; i < h; i++) { previous[i] = 0.3f * sinf((float)i * 0.017f); }
    CHECK(ds4_gpu_tensor_write(sp->carry, 0, previous, h * sizeof(float)));
    CHECK(dots3_draft_row(sp, e, token, sp->carry, sp->draft_logits));
    dots3_ref_dequant_row(m, e->weights.dots3_token_embd_mtp, 0, (unsigned)token, embed, h);
    dots3_ref_norm_weight(m, l->nextn_enorm, nw, h);
    dots3_ref_rms(embed, nw, joined, h, DS4_RMS_EPS);
    dots3_ref_norm_weight(m, l->nextn_hnorm, nw, h);
    dots3_ref_rms(previous, nw, joined + h, h, DS4_RMS_EPS);
    dots3_ref_matvec(m, l->nextn_eh_proj, 0, joined, x, scratch, scratch_n);
    dots3_ref_norm_weight(m, l->attn_norm, nw, h);
    dots3_ref_rms(x, nw, norm, h, DS4_RMS_EPS);
    dots3_ref_matvec(m, l->attn_kv_a_mqa, 0, norm, kv, scratch, scratch_n);
    dots3_ref_norm_weight(m, l->attn_kv_a_norm, nw, h);
    const float scale = sqrtf((float)h / DS4_N_SWA_KV_LORA);
    for (unsigned i = 0; i < DS4_N_SWA_KV_LORA; i++) { nw[i] *= scale; }
    dots3_ref_rms(kv, nw, kv, DS4_N_SWA_KV_LORA, DS4_RMS_EPS);
    for (unsigned i = 0; i < DS4_N_SWA_KV_LORA; i++) { kv[i] = motif3_bf16_round_reference(kv[i]); }
    dots3_ref_matvec(m, l->attn_kv_b, 0, kv, expanded, scratch, scratch_n);
    dots3_ref_matvec(m, l->attn_gate, 0, norm, gate, scratch, scratch_n);
    for (unsigned hh = 0; hh < heads; hh++) {
        const float weight = 1 / (1 + expf(-gate[hh]));
        for (unsigned j = 0; j < DS4_N_VALUE_MLA; j++) {
            attn[hh * DS4_N_VALUE_MLA + j] = expanded[hh * (nope + DS4_N_VALUE_MLA) + nope + j] * weight;
        }
    }
    dots3_ref_matvec(m, l->attn_output, 0, attn, out, scratch, scratch_n);
    for (unsigned i = 0; i < h; i++) { x[i] += out[i]; }
    dots3_ref_norm_weight(m, l->ffn_norm, nw, h);
    dots3_ref_rms(x, nw, norm, h, DS4_RMS_EPS);
    dots3_ref_matvec(m, l->ffn_gate, 0, norm, a, scratch, scratch_n);
    dots3_ref_matvec(m, l->ffn_up, 0, norm, b, scratch, scratch_n);
    for (unsigned i = 0; i < DS4_N_FF_DENSE; i++) { a[i] = a[i] / (1 + expf(-a[i])) * b[i]; }
    dots3_ref_matvec(m, l->ffn_down, 0, a, out, scratch, scratch_n);
    for (unsigned i = 0; i < h; i++) { x[i] += out[i]; }
    dots3_ref_norm_weight(m, l->nextn_shared_head_norm, nw, h);
    dots3_ref_rms(x, nw, norm, h, DS4_RMS_EPS);
    dots3_ref_matvec(m, e->weights.output, 0, norm, expected, scratch, scratch_n);
    double aa = 0, bb = 0, ab = 0, dd = 0;
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        const double actual = sp->draft_logits[i], ref = expected[i], diff = actual - ref;
        CHECK(isfinite(actual) && isfinite(ref));
        aa += actual * actual; bb += ref * ref; ab += actual * ref; dd += diff * diff;
    }
    const double cosine = ab / sqrt(aa * bb), relative = sqrt(dd / bb);
    fprintf(stderr, "dots3 MTP independent full-vocab cosine=%.9g relative_l2=%.9g top=%d/%d\n",
            cosine, relative, (int)argmax_f32(sp->draft_logits, DS4_N_VOCAB), (int)argmax_f32(expected, DS4_N_VOCAB));
    CHECK(cosine > 0.9999 && relative < 0.02);
    CHECK(argmax_f32(sp->draft_logits, DS4_N_VOCAB) == argmax_f32(expected, DS4_N_VOCAB));
    dots3_spec_reset(sp);
    free(expected); free(b); free(a); free(out); free(gate); free(attn); free(expanded);
    free(kv); free(scratch); free(nw); free(norm); free(x); free(joined); free(embed); free(previous);
}

static void expect_invalid(ds4_session *s, const ds4_session_snapshot *snap) {
    char err[256] = {0};
    const uint64_t generation = s->generation;
    CHECK(ds4_session_load_snapshot(s, snap, err, sizeof(err)) != 0);
    CHECK(s->generation == generation + 1 && !s->checkpoint_valid && !s->checkpoint.len);
    CHECK(!s->dots3_graph.cache_len && !s->dots3_spec->target_pos && !s->dots3_spec->draft.cache_len);
    CHECK(!ds4_session_payload_bytes(s));
    CHECK(ds4_session_argmax(s) == -1);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <dots3-MQ87-first-shard.gguf> [prompt_rows=600, long=2304]\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    const unsigned long parsed = argc == 3 ? strtoul(argv[2], &end, 10) : DEFAULT_PROMPT_ROWS;
    if (!parsed || parsed > MAX_TEST_CONTEXT - 16 || (end && *end)) { return 2; }
    const unsigned prompt_rows = (unsigned)parsed;
    const unsigned context = prompt_rows + 16 <= 1024 ? 1024 : MAX_TEST_CONTEXT;
    fprintf(stderr, "dots3 MTP fixture: rows=%u ctx=%u native_cap=32\n", prompt_rows, context);
    setenv("DS4_NO_BOOT_PREWARM", "1", 1);
    setenv("DS4_SESSION_LAZY_GRAPH", "0", 1);
    setenv("DS4_DOTS3_PREFILL_CHUNK", "32", 1);
    unsetenv("DS4_MTP_SPEC_DISABLE");
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_CUDA,
        .n_threads = 8, .mtp_draft_tokens = 3, .defer_boot_prewarm = true};
    ds4_engine *engine = NULL;
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    CHECK(DS4_MODEL_VARIANT == DS4_VARIANT_DOTS3_NOTE_PREV);
    CHECK(dots3_spec_bytes(4096, 128) == UINT64_C(26068040));
    CHECK(dots3_spec_bytes(1024, 32) == UINT64_C(24101960));
    CHECK(dots3_spec_bytes(262144, 4096) == UINT64_C(107332680));
    ds4_session *plain = NULL, *spec = NULL, *teacher = NULL;
    setenv("DS4_DOTS3_MTP", "0", 1);
    CHECK(ds4_session_create(&plain, engine, (int)context) == 0);
    setenv("DS4_DOTS3_MTP", "1", 1);
    CHECK(ds4_session_create(&spec, engine, (int)context) == 0);
    CHECK(!ds4_session_dots3_mtp(plain) && ds4_session_dots3_mtp(spec));
    setenv("DS4_SESSION_LAZY_GRAPH", "1", 1);
    CHECK(ds4_session_create(&teacher, engine, (int)context) == 0);
    CHECK(teacher->graph_pending && ds4_session_dots3_mtp(teacher));
    setenv("DS4_DOTS3_MTP", "0", 1);
    char alloc_err[128] = {0};
    CHECK(ds4_session_ensure_graph(teacher, alloc_err, sizeof(alloc_err)) == 0);
    CHECK(teacher->dots3_spec && ds4_session_dots3_mtp(teacher));
    setenv("DS4_DOTS3_MTP", "1", 1);
    ds4_tokens words = {0}, prompt = {0};
    ds4_tokenize_text(engine, "The capital of France is Paris. Explain local attention and counting from one to ten.\n", &words);
    CHECK(words.len > 0);
    predictor_reference(engine, teacher->dots3_spec, words.v[0]);
    for (unsigned i = 0; i < prompt_rows; i++) { token_vec_push(&prompt, words.v[i % (unsigned)words.len]); }
    char err[256] = {0};
    CHECK(ds4_session_sync(plain, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_sync(spec, &prompt, err, sizeof(err)) == 0);
    logits_equal(plain, spec);
    ds4_session_snapshot base = {0}, saved = {0};
    CHECK(ds4_session_save_snapshot(plain, &base, err, sizeof(err)) == 0);
    CHECK(ds4_session_save_snapshot(spec, &saved, err, sizeof(err)) == 0);
    CHECK(saved.len == base.len + dots3_spec_payload_bytes(spec->dots3_spec, prompt_rows));
    for (int keep = 1; keep <= VERIFY_ROWS; keep++) {
        CHECK(ds4_session_load_snapshot(plain, &base, err, sizeof(err)) == 0);
        CHECK(ds4_session_load_snapshot(spec, &saved, err, sizeof(err)) == 0);
        CHECK(ds4_session_load_snapshot(teacher, &saved, err, sizeof(err)) == 0);
        int tokens[VERIFY_ROWS], targets[VERIFY_ROWS];
        const uint64_t generation = spec->generation;
        const int first = ds4_session_argmax(plain);
        CHECK(ds4_session_dots3_trial(spec, -1, VERIFY_ROWS, tokens, targets, VERIFY_ROWS, err, sizeof(err)) < 0);
        CHECK(spec->generation == generation);
        const int n = ds4_session_dots3_trial(spec, first, VERIFY_ROWS, tokens, targets, VERIFY_ROWS, err, sizeof(err));
        CHECK(n == VERIFY_ROWS && tokens[0] == first);
        CHECK(ds4_session_sync(spec, &prompt, err, sizeof(err)) != 0);
        CHECK(ds4_session_eval(spec, first, err, sizeof(err)) != 0);
        CHECK(ds4_session_dots3_commit(spec, 0, err, sizeof(err)) != 0);
        CHECK(spec->generation == generation && spec->checkpoint_valid && spec->checkpoint.len == (int)prompt_rows);
        CHECK(spec->dots3_spec->trial_n == VERIFY_ROWS && !ds4_session_payload_bytes(spec));
        CHECK(ds4_session_argmax(spec) == -1);
        ds4_session_snapshot pending = {0};
        CHECK(ds4_session_save_snapshot(spec, &pending, err, sizeof(err)) != 0);
        CHECK(!pending.ptr);
        for (int i = 0; i < keep; i++) {
            CHECK(ds4_session_eval(plain, tokens[i], err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(teacher, tokens[i], err, sizeof(err)) == 0);
            CHECK(ds4_session_argmax(plain) == targets[i]);
        }
        CHECK(ds4_session_dots3_commit(spec, keep, err, sizeof(err)) == 0);
        CHECK(ds4_session_pos(spec) == (int)prompt_rows + keep && spec->generation == generation);
        logits_equal(plain, spec);
        state_equal(teacher, spec);
        rejected_equal(spec, prompt_rows, (unsigned)keep, VERIFY_ROWS);
        const int next = ds4_session_argmax(plain);
        CHECK(ds4_session_eval(plain, next, err, sizeof(err)) == 0);
        CHECK(ds4_session_eval(teacher, next, err, sizeof(err)) == 0);
        CHECK(ds4_session_eval(spec, next, err, sizeof(err)) == 0);
        logits_equal(plain, spec);
        ds4_session_snapshot roundtrip = {0};
        CHECK(ds4_session_save_snapshot(spec, &roundtrip, err, sizeof(err)) == 0);
        CHECK(ds4_session_load_snapshot(spec, &roundtrip, err, sizeof(err)) == 0);
        state_equal(teacher, spec);
        ds4_session_snapshot_free(&roundtrip);
    }
    CHECK(ds4_session_load_snapshot(plain, &base, err, sizeof(err)) == 0);
    CHECK(ds4_session_load_snapshot(spec, &saved, err, sizeof(err)) == 0);
    CHECK(ds4_session_load_snapshot(teacher, &saved, err, sizeof(err)) == 0);
    for (int budget = 1; budget < VERIFY_ROWS; budget++) {
        short_trial_equal(plain, spec, teacher, budget, budget, budget);
    }
    /* Malformed and incompatible payloads must retire both native frontiers. */
    expect_invalid(spec, &base);
    CHECK(ds4_session_load_snapshot(spec, &saved, err, sizeof(err)) == 0);
    ds4_session_snapshot broken = saved;
    broken.len--;
    expect_invalid(spec, &broken);
    CHECK(ds4_session_load_snapshot(spec, &saved, err, sizeof(err)) == 0);
    const uint8_t position_byte = saved.ptr[base.len];
    saved.ptr[base.len] ^= 1;
    expect_invalid(spec, &saved);
    saved.ptr[base.len] = position_byte;
    CHECK(ds4_session_load_snapshot(spec, &saved, err, sizeof(err)) == 0);
    /* Cancelled trials and strict rewind require replay before any sampling. */
    int tokens[VERIFY_ROWS], targets[VERIFY_ROWS];
    CHECK(ds4_session_dots3_trial(spec, ds4_session_argmax(spec), VERIFY_ROWS, tokens, targets,
                                  VERIFY_ROWS, err, sizeof(err)) == VERIFY_ROWS);
    ds4_session_rewind(spec, prompt_rows - 1);
    CHECK(!spec->checkpoint_valid && !spec->dots3_spec->trial_n);
    CHECK(ds4_session_eval(spec, words.v[0], err, sizeof(err)) != 0);
    CHECK(ds4_session_sync(spec, &prompt, err, sizeof(err)) == 0);
    CHECK(ds4_session_load_snapshot(teacher, &saved, err, sizeof(err)) == 0);
    state_equal(teacher, spec);
    ds4_session_snapshot_free(&saved); ds4_session_snapshot_free(&base);
    ds4_session_free(teacher); ds4_session_free(spec); ds4_session_free(plain);
    if (prompt_rows == DEFAULT_PROMPT_ROWS) { context_tail(engine, &words); }
    ds4_tokens_free(&prompt); ds4_tokens_free(&words); ds4_engine_close(engine);
    puts("dots3 MTP: independent predictor, full-vocabulary/cache parity, accepts 1/2/3/4, short trials, wrapped rollback, snapshot/malformed/cancel PASS");
    return 0;
}
