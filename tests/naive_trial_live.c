/* Isolate width arithmetic from rejected-token values using one shared model. */
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
}

enum { MOE_NORM, MOE_GATE, MOE_UP, MOE_DOWN, MOE_STAGES };
enum { MOE_TRACE_WIDTH = N05_USED * N05_EMBED };

static void index_path(unsigned session) {
    if (!getenv("DS4_NAIVE_TEST_INDEX_PACK")) { return; }
    // Compare the scalar query layout against packing in the same owner.
    assert(!setenv("DS4_NAIVE_INDEX_PACK", session ? "1" : "0", 1));
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

static void cache_diff(ds4_naive_graph *a, ds4_naive_graph *b, unsigned pos) {
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
    ds4_session *s[3];
    float *hidden[3];
    float *moe[3];
    char err[256];
    for (unsigned i = 0; i < 3; i++) {
        index_path(i);
        assert(!ds4_session_create(&s[i], e, input.len + 16));
        assert(!ds4_session_sync(s[i], &input, err, sizeof(err)));
        hidden[i] = xmalloc(N05_LAYERS * N05_EMBED * sizeof(float));
        moe[i] = xmalloc(MOE_STAGES * MOE_TRACE_WIDTH * sizeof(float));
    }
    int trial[N05_DF_BLOCK], altered[N05_DF_BLOCK];
    const int anchor = ds4_session_argmax(s[0]);
    assert(anchor >= 0);
    assert(naive_draft_block(&s[1]->naive_graph.draft, &e->model, &e->weights,
        anchor, 0, N05_DF_BLOCK, trial));
    memcpy(altered, trial, sizeof(trial));
    for (unsigned i = 1; i < N05_DF_BLOCK; i++) { altered[i] = (trial[i] + 7919 * i) % N05_VOCAB; }
    index_path(0);
    trace_rows(s[0], trial, 1, hidden[0], moe[0]);
    index_path(1);
    trace_rows(s[1], trial, N05_DF_BLOCK, hidden[1], moe[1]);
    trace_rows(s[2], altered, N05_DF_BLOCK, hidden[2], moe[2]);
    const char *names[] = {"first MoE input", "first MoE gate", "first MoE up", "first MoE down"};
    const unsigned widths[] = {N05_EMBED, N05_USED * N05_FF,
        N05_USED * N05_FF, MOE_TRACE_WIDTH};
    for (unsigned i = 0; i < MOE_STAGES; i++) {
        diff(names[i], moe[0] + i * MOE_TRACE_WIDTH, moe[1] + i * MOE_TRACE_WIDTH, widths[i]);
    }
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        char name[64];
        snprintf(name, sizeof(name), "width layer=%u", il);
        diff(name, hidden[0] + il * N05_EMBED, hidden[1] + il * N05_EMBED, N05_EMBED);
        snprintf(name, sizeof(name), "rejected layer=%u", il);
        diff(name, hidden[1] + il * N05_EMBED, hidden[2] + il * N05_EMBED, N05_EMBED);
        assert(!memcmp(hidden[0] + il * N05_EMBED, hidden[1] + il * N05_EMBED, N05_EMBED * sizeof(float)));
        assert(!memcmp(hidden[1] + il * N05_EMBED, hidden[2] + il * N05_EMBED, N05_EMBED * sizeof(float)));
    }
    diff("width logits", s[0]->logits, s[1]->logits, N05_VOCAB);
    diff("rejected logits", s[1]->logits, s[2]->logits, N05_VOCAB);
    assert(!memcmp(s[0]->logits, s[1]->logits, N05_VOCAB * sizeof(float)));
    assert(!memcmp(s[1]->logits, s[2]->logits, N05_VOCAB * sizeof(float)));
    printf("argmax width1=%d width7=%d altered=%d\n",
        ds4_session_argmax(s[0]), ds4_session_argmax(s[1]), ds4_session_argmax(s[2]));
    cache_diff(&s[0]->naive_graph, &s[1]->naive_graph, input.len);
    cache_diff(&s[1]->naive_graph, &s[2]->naive_graph, input.len);
    assert(ds4_session_argmax(s[1]) == ds4_session_argmax(s[2]));
    const int width1 = ds4_session_argmax(s[0]), width7 = ds4_session_argmax(s[1]);
    for (unsigned i = 0; i < 3; i++) { free(hidden[i]); free(moe[i]); ds4_session_free(s[i]); }
    ds4_tokens_free(&input); ds4_engine_close(e);
    fflush(stdout);
    assert(width1 == width7);
    return 0;
}
