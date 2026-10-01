/* Every rejection boundary commits one prefix and preserves both windows. */
#include "../ds4.c"
#include <assert.h>
#include "naive_state_fixture.h"

int main(void) {
    g_ds4_shape = DS4_SHAPE_NAIVE_N05_FLASH;
    assert(ds4_gpu_init());
    ds4_engine e = {0};
    e.backend = DS4_BACKEND_CUDA; e.dspark_ready = true;
    ds4_model model = {0};
    ds4_naive_draft weights = {.model = &model};
    ds4_session s = {.engine = &e, .naive_graph_ready = true, .checkpoint_valid = true};
    const unsigned start = 2051;
    s.logits = malloc(N05_VOCAB * sizeof(float));
    s.checkpoint.v = malloc(4097 * sizeof(int)); s.checkpoint.cap = 4097;
    assert(s.logits && s.checkpoint.v);
    ds4_naive_graph *g = &s.naive_graph;
    assert(naive_graph_alloc(g, 4097, 32));
    assert(naive_draft_alloc(&g->draft, &weights));
    ds4_naive_df_runtime *d = &g->draft;
    float *rows = malloc(N05_DF_BLOCK * N05_VOCAB * sizeof(float));
    assert(rows);
    for (unsigned i = 0; i < N05_DF_BLOCK; i++) {
        for (unsigned j = 0; j < N05_VOCAB; j++) { rows[i * N05_VOCAB + j] = i + (j % 71) * .25f; }
    }
    char err[256];
    for (unsigned keep = 1; keep <= N05_DF_BLOCK; keep++) {
        cache_rows(g, start, SEED);
        cache_rows(g, start + N05_DF_BLOCK, SEED);
        g->position = start + N05_DF_BLOCK;
        d->first = g->position - N05_DF_CAP;
        d->trial_pos = start; d->trial_n = N05_DF_BLOCK;
        for (unsigned i = 0; i < N05_DF_BLOCK; i++) { d->trial[i] = (int)(100 + i); }
        s.checkpoint.len = start;
        assert(ds4_gpu_tensor_write(d->ws->verify, 0, rows, N05_DF_BLOCK * N05_VOCAB * sizeof(float)));
        assert(ds4_session_argmax(&s) == -1);
        assert(!naive_payload_bytes(g, g->position));
        assert(ds4_session_naive_commit(&s, 0, err, sizeof(err)));
        assert(ds4_session_naive_commit(&s, N05_DF_BLOCK + 1, err, sizeof(err)));
        assert(d->trial_n == N05_DF_BLOCK && s.checkpoint.len == (int)start);
        int tokens[N05_DF_BLOCK], target[N05_DF_BLOCK];
        assert(ds4_session_naive_trial(&s, 100, N05_DF_BLOCK, tokens, target,
            N05_DF_BLOCK, err, sizeof(err)) == -1);
        assert(!ds4_session_naive_commit(&s, (int)keep, err, sizeof(err)));
        assert(g->position == start + keep && s.checkpoint.len == (int)(start + keep));
        assert(!memcmp(s.logits, rows + (keep - 1) * N05_VOCAB, N05_VOCAB * sizeof(float)));
        for (unsigned i = 0; i < keep; i++) { assert(s.checkpoint.v[start + i] == (int)(100 + i)); }
        cache_rows(g, start + keep, CHECK);
        assert(naive_payload_bytes(g, start + keep));
        assert(ds4_session_argmax(&s) >= 0);
    }
    free(rows); free(s.logits); free(s.checkpoint.v);
    naive_graph_free(g);
    ds4_gpu_cleanup();
    puts("Naive commits 1..7 verified rows with byte-exact target and draft windows");
    return 0;
}
