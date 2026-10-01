/* Compare diagnostic paths, widths and rejected values using one shared model. */
#include "../ds4.c"
#include <assert.h>

static void diff(const char *name, const float *a, const float *b, size_t n) {
    double delta = 0, norm = 0, maximum = 0;
    size_t changed = 0;
    for (size_t i = 0; i < n; i++) {
        assert(isfinite(a[i]) && isfinite(b[i]));
        const double d = (double)a[i] - b[i];
        delta += d * d; norm += (double)a[i] * a[i];
        if (fabs(d) > maximum) { maximum = fabs(d); }
        if (memcmp(a + i, b + i, sizeof(float))) { changed++; }
    }
    printf("%s changed=%zu max_abs=%.9g relative_l2=%.9g\n",
        name, changed, maximum, sqrt(delta / (norm ? norm : 1)));
    assert(!memcmp(a, b, n * sizeof(float)));
}

enum { MOE_NORM, MOE_GATE, MOE_UP, MOE_DOWN, MOE_STAGES };
enum { MOE_TRACE_WIDTH = N05_USED * N05_EMBED };
enum live_case { OFF_ONE, ON_ONE, ON_WIDE, ON_ALTERED, LIVE_CASES };

static void control_path(enum live_case which) {
    const char *knob = getenv("DS4_NAIVE_TEST_CONTROL");
    if (!knob || !*knob) {
        if (!getenv("DS4_NAIVE_TEST_INDEX_PACK")) { return; }
        knob = "DS4_NAIVE_INDEX_PACK";
    }
    // Read the diagnostic switch per call so one owner can serve both paths.
    assert(!setenv(knob, which == OFF_ONE ? "0" : "1", 1));
}

static void trace_rows(ds4_session *s, const int *tokens, unsigned n, float *trace, float *moe) {
    ds4_naive_graph *g = &s->naive_graph;
    ds4_engine *e = s->engine;
    const unsigned pos = g->position;
    unsigned positions[N05_DF_BLOCK];
    int target[N05_DF_BLOCK];
    char err[256];
    for (unsigned i = 0; i < n; i++) { positions[i] = pos + i; }
    assert(ds4_gpu_tensor_write(g->tokens, 0, tokens, n * sizeof(int)));
    assert(ds4_gpu_tensor_write(g->positions, 0, positions, n * sizeof(unsigned)));
    assert(ds4_gpu_step37_rope(g->rope_full, g->freq_full, g->positions, N05_ROT, n));
    assert(ds4_gpu_step37_rope(g->rope_swa, g->freq_swa, g->positions, N05_ROT, n));
    assert(ds4_gpu_embed_tokens_quant_tensor(g->ws.b_cur, g->tokens, e->model.map, e->model.size,
        e->weights.token_embd->abs_offset, e->weights.token_embd->type, N05_VOCAB, n, N05_EMBED));
    assert(ds4_gpu_naive_round(g->ws.b_cur, (uint64_t)n * N05_EMBED));
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        assert(naive_layer(g, &e->model, &e->weights, il, n));
        assert(ds4_gpu_tensor_read(g->ws.b_cur, 0, trace + il * N05_EMBED, N05_EMBED * sizeof(float)));
        if (il == 1) {
            ds4_gpu_tensor *stages[] = {g->ws.b_norm, g->ws.b_routed_gate,
                g->ws.b_routed_up, g->ws.b_routed_down};
            const unsigned widths[] = {N05_EMBED, N05_USED * N05_FF,
                N05_USED * N05_FF, MOE_TRACE_WIDTH};
            for (unsigned j = 0; j < MOE_STAGES; j++) {
                assert(ds4_gpu_tensor_read(stages[j], 0,
                    moe + j * MOE_TRACE_WIDTH, widths[j] * sizeof(float)));
            }
        }
        const int tap = naive_tap(il);
        if (tap >= 0) { assert(ds4_gpu_naive_df_tap(g->draft.ws->tap, g->ws.b_cur, 0, n, (unsigned)tap)); }
    }
    assert(naive_draft_prepare(&g->draft, pos, n));
    assert(naive_verify_rows(g, &e->model, &e->weights, n, target));
    g->position = pos + n;
    g->draft.trial_pos = pos; g->draft.trial_n = n;
    memcpy(g->draft.trial, tokens, n * sizeof(int));
    assert(!ds4_session_naive_commit(s, 1, err, sizeof(err)));
    assert(g->position == pos + 1 && g->draft.position == pos + 1);
}

static void span_equal(const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
                         uint64_t offset, uint64_t bytes) {
    assert(bytes);
    const size_t cap = bytes < DS4_SESSION_IO_CHUNK ? (size_t)bytes : DS4_SESSION_IO_CHUNK;
    uint8_t *x = xmalloc(cap), *y = xmalloc(cap);
    while (bytes) {
        const size_t count = bytes < cap ? (size_t)bytes : cap;
        assert(ds4_gpu_tensor_read(a, offset, x, count));
        assert(ds4_gpu_tensor_read(b, offset, y, count));
        assert(!memcmp(x, y, count));
        offset += count; bytes -= count;
    }
    free(x); free(y);
}

static void ring_equal(const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
                         unsigned capacity, unsigned first, unsigned end, uint64_t row) {
    assert(capacity && row && first <= end && end - first <= capacity);
    while (first < end) {
        const unsigned slot = first % capacity;
        unsigned count = capacity - slot;
        if (count > end - first) { count = end - first; }
        span_equal(a, b, (uint64_t)slot * row, (uint64_t)count * row);
        first += count;
    }
}

static void state_equal(ds4_session *a, ds4_session *b) {
    ds4_naive_graph *ga = &a->naive_graph, *gb = &b->naive_graph;
    const unsigned end = ga->position;
    assert(!ga->failed && !gb->failed && ga->position == gb->position);
    assert(a->checkpoint_valid && b->checkpoint_valid);
    assert(a->checkpoint.len == (int)end && b->checkpoint.len == (int)end);
    assert(!memcmp(a->checkpoint.v, b->checkpoint.v, (size_t)end * sizeof(int)));
    diff("committed logits", a->logits, b->logits, N05_VOCAB);
    span_equal(ga->logits, gb->logits, 0, N05_VOCAB * sizeof(float));

    // DSA retains all history; only the causal SWA window remains live.
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        const unsigned rows = naive_saved_rows(il, end);
        const uint64_t row = naive_kv_heads(il) * N05_KV_WIDTH * sizeof(uint16_t);
        assert(ga->kv_cap[il] == gb->kv_cap[il]);
        ring_equal(ga->kv[il], gb->kv[il], ga->kv_cap[il], end - rows, end, row);
        if (naive_is_dsa(il)) {
            span_equal(ga->codes[il], gb->codes[il], 0, (uint64_t)end * N05_INDEX_DIM);
            span_equal(ga->scales[il], gb->scales[il], 0, (uint64_t)end * sizeof(float));
        }
    }

    // Compare every retained draft row, excluding unused scratch and trial lanes.
    assert(ga->draft.ws && gb->draft.ws);
    assert(!ga->draft.trial_n && !gb->draft.trial_n);
    assert(ga->draft.position == end && gb->draft.position == end);
    assert(ga->draft.first == gb->draft.first);
    const uint64_t row = 2 * N05_DF_KV * N05_DF_DIM * sizeof(uint16_t);
    for (unsigned il = 0; il < N05_DF_LAYERS; il++) {
        ring_equal(ga->draft.kv[il], gb->draft.kv[il], N05_DF_CAP, ga->draft.first, end, row);
    }
    printf("committed state rows=%u draft_first=%u exact\n", end, ga->draft.first);
}

static void cache_diff(ds4_naive_graph *a, ds4_naive_graph *b, unsigned pos) {
    // Wide verification can overwrite old draft slots and advance first.
    // Its accepted row is valid; future rows and retired slots are not comparable.
    assert(a->position == pos + 1 && b->position == pos + 1);
    assert(a->draft.position == pos + 1 && b->draft.position == pos + 1);
    assert(!a->draft.trial_n && !b->draft.trial_n);
    uint8_t x[8 * (N05_KEY + N05_VALUE) * sizeof(uint16_t)], y[sizeof(x)];
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        const size_t bytes = naive_kv_heads(il) * (N05_KEY + N05_VALUE) * sizeof(uint16_t);
        const uint64_t offset = (uint64_t)(pos % a->kv_cap[il]) * bytes;
        assert(ds4_gpu_tensor_read(a->kv[il], offset, x, bytes));
        assert(ds4_gpu_tensor_read(b->kv[il], offset, y, bytes));
        size_t changed = 0;
        for (size_t j = 0; j < bytes; j++) { changed += x[j] != y[j]; }
        printf("cache layer=%u changed_bytes=%zu\n", il, changed);
        assert(!changed);
        if (naive_is_dsa(il)) {
            assert(ds4_gpu_tensor_read(a->codes[il], (uint64_t)pos * N05_INDEX_DIM, x, N05_INDEX_DIM));
            assert(ds4_gpu_tensor_read(b->codes[il], (uint64_t)pos * N05_INDEX_DIM, y, N05_INDEX_DIM));
            assert(!memcmp(x, y, N05_INDEX_DIM));
            assert(ds4_gpu_tensor_read(a->scales[il], (uint64_t)pos * sizeof(float), x, sizeof(float)));
            assert(ds4_gpu_tensor_read(b->scales[il], (uint64_t)pos * sizeof(float), y, sizeof(float)));
            assert(!memcmp(x, y, sizeof(float)));
        }
    }
    for (unsigned il = 0; il < N05_DF_LAYERS; il++) {
        const size_t bytes = 2 * N05_DF_KV * N05_DF_DIM * sizeof(uint16_t);
        const uint64_t offset = (uint64_t)(pos % N05_DF_CAP) * bytes;
        assert(ds4_gpu_tensor_read(a->draft.kv[il], offset, x, bytes));
        assert(ds4_gpu_tensor_read(b->draft.kv[il], offset, y, bytes));
        size_t changed = 0;
        for (size_t j = 0; j < bytes; j++) { changed += x[j] != y[j]; }
        printf("draft_cache layer=%u changed_bytes=%zu\n", il, changed);
        assert(!changed);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s main.gguf draft.gguf prompt.i32\n", argv[0]); return 2; }
    ds4_host_shape shape = {.variant = DS4_VARIANT_NAIVE_N05_FLASH};
    ds4_host_shape_install(&shape);
    ds4_engine_options options = {.model_path = argv[1], .dspark_path = argv[2],
        .backend = DS4_BACKEND_CUDA, .mtp_draft_tokens = N05_DF_BLOCK - 1,
        .quality = true, .defer_boot_prewarm = true, .power_percent = 100};
    ds4_engine *e = NULL;
    assert(!ds4_engine_open(&e, &options) && e);
    ds4_host_shape_clear();
    FILE *fp = fopen(argv[3], "rb");
    assert(fp);
    ds4_tokens input = {0};
    int token;
    while (fread(&token, sizeof(token), 1, fp)) { ds4_tokens_push(&input, token); }
    assert(feof(fp) && input.len > 0); fclose(fp);
    ds4_session *s[LIVE_CASES];
    float *hidden[LIVE_CASES];
    float *moe[LIVE_CASES];
    char err[256];
    for (unsigned i = 0; i < LIVE_CASES; i++) {
        control_path((enum live_case)i);
        assert(!ds4_session_create(&s[i], e, input.len + 16));
        assert(!ds4_session_sync(s[i], &input, err, sizeof(err)));
        hidden[i] = xmalloc(N05_LAYERS * N05_EMBED * sizeof(float));
        moe[i] = xmalloc(MOE_STAGES * MOE_TRACE_WIDTH * sizeof(float));
    }
    // Qualify the complete prefix before verification can crop the draft ring.
    for (unsigned i = ON_ONE; i < LIVE_CASES; i++) { state_equal(s[OFF_ONE], s[i]); }
    int trial[N05_DF_BLOCK], altered[N05_DF_BLOCK];
    const int anchor = ds4_session_argmax(s[OFF_ONE]);
    assert(anchor >= 0);
    control_path(ON_ONE);
    assert(naive_draft_block(&s[ON_ONE]->naive_graph.draft, &e->model, &e->weights,
        anchor, 0, N05_DF_BLOCK, trial));
    memcpy(altered, trial, sizeof(trial));
    for (unsigned i = 1; i < N05_DF_BLOCK; i++) { altered[i] = (trial[i] + 7919 * i) % N05_VOCAB; }
    for (unsigned i = 0; i < LIVE_CASES; i++) {
        control_path((enum live_case)i);
        const unsigned rows = i <= ON_ONE ? 1 : N05_DF_BLOCK;
        const int *tokens = i == ON_ALTERED ? altered : trial;
        trace_rows(s[i], tokens, rows, hidden[i], moe[i]);
    }
    const char *names[] = {"first MoE input", "first MoE gate", "first MoE up", "first MoE down"};
    const unsigned widths[] = {N05_EMBED, N05_USED * N05_FF,
        N05_USED * N05_FF, MOE_TRACE_WIDTH};
    const char *pairs[] = {"control", "width", "rejected"};
    for (unsigned pair = 0; pair < LIVE_CASES - 1; pair++) {
        char name[64];
        for (unsigned i = 0; i < MOE_STAGES; i++) {
            snprintf(name, sizeof(name), "%s %s", pairs[pair], names[i]);
            diff(name, moe[pair] + i * MOE_TRACE_WIDTH, moe[pair + 1] + i * MOE_TRACE_WIDTH, widths[i]);
        }
        for (unsigned il = 0; il < N05_LAYERS; il++) {
            snprintf(name, sizeof(name), "%s layer=%u", pairs[pair], il);
            diff(name, hidden[pair] + il * N05_EMBED, hidden[pair + 1] + il * N05_EMBED, N05_EMBED);
        }
        snprintf(name, sizeof(name), "%s logits", pairs[pair]);
        diff(name, s[pair]->logits, s[pair + 1]->logits, N05_VOCAB);
        assert(ds4_session_argmax(s[pair]) == ds4_session_argmax(s[pair + 1]));
    }
    printf("argmax off1=%d on1=%d width7=%d altered=%d\n", ds4_session_argmax(s[OFF_ONE]),
        ds4_session_argmax(s[ON_ONE]), ds4_session_argmax(s[ON_WIDE]), ds4_session_argmax(s[ON_ALTERED]));
    state_equal(s[OFF_ONE], s[ON_ONE]);
    cache_diff(&s[ON_ONE]->naive_graph, &s[ON_WIDE]->naive_graph, input.len);
    cache_diff(&s[ON_WIDE]->naive_graph, &s[ON_ALTERED]->naive_graph, input.len);
    for (unsigned i = 0; i < LIVE_CASES; i++) { free(hidden[i]); free(moe[i]); ds4_session_free(s[i]); }
    ds4_tokens_free(&input); ds4_engine_close(e);
    fflush(stdout);
    return 0;
}
