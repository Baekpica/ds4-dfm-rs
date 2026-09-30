/* Prefix forks must preserve full index history and displaced SWA rows. */
#include "../ds4.c"
#include <assert.h>
#include "naive_state_fixture.h"

int main(void) {
    g_ds4_shape = DS4_SHAPE_NAIVE_N05_FLASH;
    assert(ds4_gpu_init());
    assert(!setenv("DS4_NAIVE_PREFILL_CHUNK", "32", 1));
    ds4_engine e = {0};
    e.backend = DS4_BACKEND_CUDA; e.metal_ready = true;
    ds4_batch_ctx *ctx = NULL;
    char err[256];
    assert(!ds4_batch_ctx_create(&e, 4097, 3, 32, &ctx, err, sizeof(err)));
    assert(ctx && ds4_batch_ctx_max_seq(ctx) == 3);
    assert(ds4_batch_ctx_supports_partial_reuse(ctx));
    ds4_naive_batch_runtime *rt = ctx->naive;
    assert(rt && rt->checkpoint_slab);
    const unsigned saved = 2051, appended = 2251;
    ds4_naive_graph *g = &rt->graph[0];
    cache_rows(g, saved, SEED);
    g->position = saved;
    rt->bank_logits_valid[0] = 1;
    for (unsigned i = 0; i < N05_VOCAB; i++) { rt->bank_logits[i] = (i % 71) * .25f; }
    assert(naive_ckpt_capture(rt, 0, saved, N05_HAS_LOGITS, 0));
    const int slot = naive_ckpt_find(rt, 0, saved, saved);
    assert(slot >= 0);
    cache_rows(g, appended, SEED);
    g->position = appended;
    uint32_t cut = 0;
    assert(naive_ckpt_restore(rt, 0, 1, slot, saved, &cut));
    assert(cut == saved && rt->graph[1].position == saved);
    assert(rt->bank_logits_valid[1]);
    assert(!memcmp(rt->bank_logits, rt->bank_logits + N05_VOCAB, N05_VOCAB * sizeof(float)));
    cache_rows(&rt->graph[1], saved, CHECK);

    assert(naive_ckpt_restore(rt, 0, 0, slot, saved, &cut));
    cache_rows(g, saved, CHECK);
    cache_rows(&rt->graph[1], appended, SEED);
    rt->graph[1].position = appended;
    cache_rows(g, saved, CHECK);
    assert(naive_bank_copy(rt, 1, 2, appended));
    cache_rows(&rt->graph[2], appended, CHECK);
    assert(rt->graph[2].position == appended);
    assert(rt->graph[2].codes[0] != rt->graph[1].codes[0]);
    assert(rt->graph[2].scales[0] != rt->graph[1].scales[0]);
    assert(rt->graph[2].ws.b_cur == g->ws.b_cur);

    // The same bank payload API backs SSD KV and server snapshots.
    for (unsigned i = 0; i < appended; i++) { ctx->bank_hist[2 * ctx->seq_cap + i] = i; }
    ctx->bank_hist_len[2] = appended; ctx->bank_hist_valid[2] = 1;
    const uint64_t bytes = ds4_cont_bank_payload_bytes(ctx, 2);
    FILE *fp = tmpfile();
    assert(bytes && fp && !ds4_cont_bank_save_payload(ctx, 2, fp, err, sizeof(err)));
    assert((uint64_t)ftello(fp) == bytes);
    rewind(fp);
    assert(!ds4_cont_bank_restore_payload(ctx, 0, fp, bytes, err, sizeof(err)));
    assert(ctx->bank_hist_valid[0] && ctx->bank_hist_len[0] == appended);
    assert(!memcmp(ctx->bank_hist, ctx->bank_hist + 2 * ctx->seq_cap, appended * sizeof(int)));
    assert(rt->bank_logits_valid[0] && rt->graph[0].position == appended);
    cache_rows(g, appended, CHECK);
    rewind(fp);
    assert(ds4_cont_bank_restore_payload(ctx, 0, fp, bytes - 1, err, sizeof(err)));
    assert(!ctx->bank_hist_valid[0] && !rt->bank_logits_valid[0]);
    assert(g->failed && !g->position);
    fclose(fp);

    naive_ckpt_drop(rt, 0);
    assert(!partial_checkpoint_ref(&rt->checkpoint[slot], 0));
    assert(partial_checkpoint_ref(&rt->checkpoint[slot], 1));
    assert(!naive_ckpt_restore(rt, 0, 2, slot, saved, &cut));
    const int invalid = -1;
    assert(!naive_bank_prefill(rt, &e, 2, &invalid, 1, appended, N05_FINAL_ROWS));
    assert(!rt->bank_logits_valid[2] && rt->graph[2].position == appended);
    ds4_batch_ctx_destroy(ctx);
    ds4_gpu_cleanup();
    puts("Naive three-bank copy, partial fork and self-restore preserve KV/index bytes");
    return 0;
}
