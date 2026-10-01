/* Real Q8 draft, isolated from the much larger target model. */
#include "../ds4.c"
#include <assert.h>

static void read_fixture(const char *dir, const char *name, float *out, size_t count) {
    char path[PATH_MAX]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *fp = fopen(path, "rb");
    assert(fp && fread(out, sizeof(float), count, fp) == count && fgetc(fp) == EOF);
    fclose(fp);
}

static void compare(const char *dir, const char *name, ds4_gpu_tensor *tensor, size_t count) {
    float *want = xmalloc(count * sizeof(float)), *got = xmalloc(count * sizeof(float));
    read_fixture(dir, name, want, count);
    assert(ds4_gpu_tensor_read(tensor, 0, got, count * sizeof(float)));
    double delta = 0, norm = 0, candidate = 0, dot = 0;
    for (size_t i = 0; i < count; i++) {
        assert(isfinite(got[i]));
        delta += (double)(got[i] - want[i]) * (got[i] - want[i]);
        norm += (double)want[i] * want[i]; candidate += (double)got[i] * got[i]; dot += (double)want[i] * got[i];
    }
    printf("%s relative_l2=%.6g cosine=%.9g\n", name, sqrt(delta / norm), dot / sqrt(norm * candidate));
    /* Structural guard; mixed MMQ arithmetic still needs actual acceptance
     * and output checks. This correlation alone does not qualify DSpark. */
    assert(dot / sqrt(norm * candidate) > .98);
    free(want); free(got);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s draft.gguf fixture_dir\n", argv[0]); return 2; }
    ds4_model m; model_open(&m, argv[1], false, false);
    ds4_naive_draft w; assert(naive_draft_bind(&w, &m));
    assert(ds4_gpu_init() && ds4_gpu_set_model_map(m.map, m.size));
    ds4_naive_df_runtime d = {0}; assert(naive_draft_alloc(&d, &w));
    float *tap = xmalloc(4 * N05_DF_SLOT * sizeof(float)), *anchor = xmalloc(N05_EMBED * sizeof(float));
    read_fixture(argv[2], "tap.f32", tap, 4 * N05_DF_SLOT);
    read_fixture(argv[2], "anchor.f32", anchor, N05_EMBED);
    const unsigned cases[] = {17, 1048, 1048569};
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const unsigned pos = cases[i];
        d.position = pos - 4; d.first = pos - 4;
        assert(ds4_gpu_tensor_write(d.ws->tap, 0, tap, 4 * N05_DF_SLOT * sizeof(float)));
        assert(naive_draft_prepare(&d, pos - 4, 4));
        assert(ds4_gpu_tensor_write(d.ws->cur, 0, anchor, N05_EMBED * sizeof(float)));
        assert(naive_draft_layers(&d, N05_DF_BLOCK));
        char name[64]; snprintf(name, sizeof(name), "%u.hidden.f32", pos);
        compare(argv[2], name, d.ws->hidden, N05_DF_BLOCK * N05_EMBED);
        float *saved = xmalloc(N05_DF_BLOCK * N05_EMBED * sizeof(float));
        assert(ds4_gpu_tensor_read(d.ws->hidden, 0, saved, N05_DF_BLOCK * N05_EMBED * sizeof(float)));
        /* Learned mask must erase arbitrary non-anchor proposal embeddings. */
        for (unsigned r = 1; r < N05_DF_BLOCK; r++) {
            assert(ds4_gpu_tensor_write(d.ws->cur, (uint64_t)r * N05_EMBED * sizeof(float), anchor, N05_EMBED * sizeof(float)));
        }
        assert(ds4_gpu_tensor_write(d.ws->cur, 0, anchor, N05_EMBED * sizeof(float)));
        assert(naive_draft_layers(&d, N05_DF_BLOCK));
        float *again = xmalloc(N05_DF_BLOCK * N05_EMBED * sizeof(float));
        assert(ds4_gpu_tensor_read(d.ws->hidden, 0, again, N05_DF_BLOCK * N05_EMBED * sizeof(float)));
        assert(!memcmp(saved, again, N05_DF_BLOCK * N05_EMBED * sizeof(float)));
        free(saved); free(again);
        const int previous = 198;
        assert(ds4_gpu_tensor_write(d.ws->token, 0, &previous, sizeof(previous)));
        assert(ds4_gpu_embed_tokens_quant_tensor(d.ws->markov, d.ws->token, m.map, m.size,
            w.markov1->abs_offset, w.markov1->type, N05_VOCAB, 1, N05_DF_RANK));
        assert(ds4_gpu_naive_round(d.ws->markov, N05_DF_RANK));
        assert(plain_graph_matmul_tensor(d.ws->bias, &m, w.markov2, N05_DF_RANK, N05_VOCAB, d.ws->markov, 1));
        assert(ds4_gpu_naive_round(d.ws->bias, N05_VOCAB));
        snprintf(name, sizeof(name), "%u.markov.f32", pos);
        compare(argv[2], name, d.ws->bias, N05_VOCAB);
        ds4_gpu_tensor *hidden = ds4_gpu_tensor_view(d.ws->hidden, N05_EMBED * sizeof(float), N05_EMBED * sizeof(float));
        assert(ds4_gpu_naive_df_conf(d.ws->confidence, hidden, d.ws->markov, m.map, m.size, w.conf->abs_offset, w.bias->abs_offset));
        float want, got; snprintf(name, sizeof(name), "%u.confidence.f32", pos);
        read_fixture(argv[2], name, &want, 1); assert(ds4_gpu_tensor_read(d.ws->confidence, 0, &got, sizeof(got)));
        printf("%s source=%g native=%g\n", name, want, got);
        assert(isfinite(got)); ds4_gpu_tensor_free(hidden);
    }
    naive_draft_free(&d); model_close(&m); free(tap); free(anchor);
    puts("real draft isolated forward and learned-mask invariance pass");
    return 0;
}
