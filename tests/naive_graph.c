/* Allocator/admission agreement and per-bank history ownership, no weights. */
#include "../ds4.c"
#include <assert.h>

#include "naive_state_fixture.h"

static void snapshot_test(ds4_naive_graph *a) {
    const unsigned n = 2051;
    ds4_naive_graph b = {0};
    assert(naive_graph_alloc(&b, a->context, 7));
    cache_rows(a, n, SEED);
    a->position = n;
    int *tokens = malloc(n * sizeof(int)), *restored = NULL;
    float *logits = malloc(N05_VOCAB * sizeof(float)), *read = malloc(N05_VOCAB * sizeof(float));
    assert(tokens && logits && read);
    for (unsigned i = 0; i < n; i++) { tokens[i] = i % N05_VOCAB; }
    for (unsigned i = 0; i < N05_VOCAB; i++) { logits[i] = (float)(i % 113) * .125f; }
    FILE *fp = tmpfile();
    char err[256];
    assert(fp && !naive_payload_save(a, tokens, n, logits, fp, err, sizeof(err)));
    const uint64_t bytes = naive_payload_bytes(a, n);
    assert((uint64_t)ftello(fp) == bytes);
    rewind(fp);
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    uint64_t left = bytes;
    for (unsigned i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        assert(!payload_read_u32(fp, &h[i], &left, err, sizeof(err)));
    }
    assert(!naive_payload_restore(&b, fp, &left, h, &restored, read, err, sizeof(err)));
    assert(!left && !b.failed && b.position == n);
    assert(!memcmp(tokens, restored, n * sizeof(int)));
    assert(!memcmp(logits, read, N05_VOCAB * sizeof(float)));
    cache_rows(&b, n, CHECK);
    free(restored); restored = NULL;
    assert(ds4_gpu_tensor_read(b.logits, 0, read, N05_VOCAB * sizeof(float)));
    assert(!memcmp(logits, read, N05_VOCAB * sizeof(float)));

    // A short body must invalidate the old frontier before any read.
    left = naive_payload_body(n) - 1;
    assert(naive_payload_restore(&b, fp, &left, h, &restored, read, err, sizeof(err)));
    assert(b.failed && !b.position && !restored);
    left++;
    h[8]++;
    assert(naive_payload_restore(&b, fp, &left, h, &restored, read, err, sizeof(err)));
    assert(b.failed && !b.position && !restored);
    h[8]--;
    assert(!ftruncate(fileno(fp), (off_t)(bytes - 1)));
    assert(!fseeko(fp, sizeof(h), SEEK_SET));
    left = naive_payload_body(n);
    assert(naive_payload_restore(&b, fp, &left, h, &restored, read, err, sizeof(err)));
    assert(b.failed && !b.position && !restored);
    assert(naive_reset(&b) && !b.position && !b.failed);
    fclose(fp);
    naive_graph_free(&b);
    free(tokens); free(logits); free(read);
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_NAIVE_N05_FLASH;
    assert(ds4_gpu_init());
    const unsigned ctx = 4097, cap = 32;
    const ds4_context_memory plan = naive_context_memory(ctx, cap);
    ds4_naive_graph a = {0}, b = {0};
    assert(naive_graph_alloc(&a, ctx, cap));
    assert(a.bytes == plan.total_bytes);
    assert(naive_graph_clone_bank(&b, &a));
    assert(b.bytes == plan.raw_bytes);
    assert(a.scores == b.scores && a.ws.b_cur == b.ws.b_cur);
    for (unsigned il = 0; il < N05_LAYERS; il++) {
        assert(a.kv[il] != b.kv[il]);
        if (!naive_is_dsa(il)) { assert(!a.codes[il] && !a.scales[il]); continue; }
        assert(a.codes[il] != b.codes[il] && a.scales[il] != b.scales[il]);
        const uint8_t code[] = {0, 126, 128, 254};
        const float scale = .03125f;
        uint8_t got[sizeof(code)];
        float restored;
        assert(ds4_gpu_tensor_write(a.codes[il], 2048ull * N05_INDEX_DIM, code, sizeof(code)));
        assert(ds4_gpu_tensor_copy(b.codes[il], 2048ull * N05_INDEX_DIM,
                                  a.codes[il], 2048ull * N05_INDEX_DIM, sizeof(code)));
        assert(ds4_gpu_tensor_write(a.scales[il], 2048ull * sizeof(float), &scale, sizeof(scale)));
        assert(ds4_gpu_tensor_copy(b.scales[il], 2048ull * sizeof(float),
                                  a.scales[il], 2048ull * sizeof(float), sizeof(scale)));
        assert(ds4_gpu_tensor_read(b.codes[il], 2048ull * N05_INDEX_DIM, got, sizeof(got)));
        assert(ds4_gpu_tensor_read(b.scales[il], 2048ull * sizeof(float), &restored, sizeof(restored)));
        assert(!memcmp(got, code, sizeof(got)) && restored == scale);
    }
    const int invalid = -1;
    assert(!naive_forward(&a, NULL, NULL, &invalid, 1, 0));
    assert(!a.failed && a.position == 0);
    naive_graph_free(&b);
    assert(a.owns_scratch && a.bytes == plan.total_bytes);
    snapshot_test(&a);
    naive_graph_free(&a);
    assert(a.bytes == 0 && !a.ws.b_cur);
    ds4_gpu_cleanup();
    puts("Naive allocation and 2051-row snapshot pass across prefill caps 32/7");
    return 0;
}
