/* Deterministic cache bytes identify every layer and absolute row. */
typedef enum { SEED, CHECK } cache_op;

static void draft_rows(ds4_naive_df_runtime *d, unsigned n, cache_op op) {
    if (!d->ws) { return; }
    const unsigned rows = n < N05_DF_WINDOW ? n : N05_DF_WINDOW;
    const unsigned first = n - rows;
    const size_t width = 2 * N05_DF_KV * N05_DF_DIM * sizeof(uint16_t);
    uint8_t *want = malloc(width), *got = malloc(width);
    assert(want && got);
    for (unsigned il = 0; il < N05_DF_LAYERS; il++) {
        for (unsigned p = first; p < n; p++) {
            for (size_t j = 0; j < width; j++) { want[j] = (uint8_t)(il * 29 + p * 31 + j * 7); }
            const uint64_t offset = (uint64_t)(p % N05_DF_CAP) * width;
            if (op == SEED) { assert(ds4_gpu_tensor_write(d->kv[il], offset, want, width)); }
            else {
                assert(ds4_gpu_tensor_read(d->kv[il], offset, got, width));
                assert(!memcmp(want, got, width));
            }
        }
    }
    if (op == SEED) { d->position = n; d->first = first; d->trial_n = 0; }
    else { assert(d->position == n && d->first <= first && !d->trial_n); }
    free(want); free(got);
}

static void cache_rows(ds4_naive_graph *g, unsigned n, cache_op op) {
    draft_rows(&g->draft, n, op);
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        const unsigned rows = naive_saved_rows(il, n), first = n - rows;
        const size_t width = naive_kv_heads(il) * N05_KV_WIDTH * sizeof(uint16_t);
        uint8_t *want = malloc(width), *got = malloc(width);
        assert(want && got);
        for (unsigned pos = first; pos < n; pos++) {
            for (size_t j = 0; j < width; j++) { want[j] = (uint8_t)(il * 17 + pos * 13 + j * 3); }
            const uint64_t offset = (uint64_t)(pos % g->kv_cap[il]) * width;
            if (op == SEED) { assert(ds4_gpu_tensor_write(g->kv[il], offset, want, width)); }
            else {
                assert(ds4_gpu_tensor_read(g->kv[il], offset, got, width));
                assert(!memcmp(want, got, width));
            }
        }
        free(want); free(got);
        if (!naive_is_dsa(il)) { continue; }
        const size_t bytes = (size_t)n * N05_INDEX_DIM;
        want = malloc(bytes); got = malloc(bytes);
        float *scales = malloc(n * sizeof(float)), *read = malloc(n * sizeof(float));
        assert(want && got && scales && read);
        for (size_t j = 0; j < bytes; j++) { want[j] = (uint8_t)(j * 7 + il); }
        for (unsigned j = 0; j < n; j++) { scales[j] = (j + il + 1) * .000125f; }
        if (op == SEED) {
            assert(ds4_gpu_tensor_write(g->codes[il], 0, want, bytes));
            assert(ds4_gpu_tensor_write(g->scales[il], 0, scales, n * sizeof(float)));
        } else {
            assert(ds4_gpu_tensor_read(g->codes[il], 0, got, bytes));
            assert(ds4_gpu_tensor_read(g->scales[il], 0, read, n * sizeof(float)));
            assert(!memcmp(want, got, bytes) && !memcmp(scales, read, n * sizeof(float)));
        }
        free(want); free(got); free(scales); free(read);
    }
}
