/* Model-free CUDA gate for EXAONE LLLG checkpoint copy and lineage. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "EXAONE checkpoint FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

enum { BANKS = 2, CONTEXT = 512, CHUNK = 32, FRONTIER = 200, ADVANCED = 480 };

static uint16_t kv_value(unsigned layer, unsigned pos, unsigned lane) {
    return (uint16_t)(1 + layer * CONTEXT + pos + lane * 7);
}

static void fill_bank(ds4_exaone_batch_runtime *rt, unsigned bank, unsigned frontier) {
    ds4_exaone_gpu_graph *g = &rt->graph[bank];
    for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        const unsigned cap = g->layer_kv_cap[il];
        uint16_t *rows = xcalloc((size_t)cap * 2 * g->kv_dim, sizeof(*rows));
        for (unsigned p = 0; p < frontier; p++) {
            for (unsigned j = 0; j < 2 * g->kv_dim; j++) {
                rows[(size_t)(p % cap) * 2 * g->kv_dim + j] = kv_value(il, p, j);
            }
        }
        CHECK(ds4_gpu_tensor_write(g->layer_kv[il], 0, rows, (uint64_t)cap * 2 * g->kv_dim * sizeof(*rows)));
        free(rows);
    }
    rt->cache_len[bank] = frontier;
}

static void check_bank(ds4_exaone_gpu_graph *g, unsigned frontier) {
    for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        const unsigned cap = g->layer_kv_cap[il];
        uint16_t *rows = xmalloc((size_t)cap * 2 * g->kv_dim * sizeof(*rows));
        CHECK(ds4_gpu_tensor_read(g->layer_kv[il], 0, rows, (uint64_t)cap * 2 * g->kv_dim * sizeof(*rows)));
        unsigned start = exaone_graph_layer_is_sliding(il) && frontier > DS4_N_SWA
            ? frontier - DS4_N_SWA : 0;
        for (unsigned p = start; p < frontier; p++) {
            for (unsigned j = 0; j < 2 * g->kv_dim; j++) {
                CHECK(rows[(size_t)(p % cap) * 2 * g->kv_dim + j] == kv_value(il, p, j));
            }
        }
        free(rows);
    }
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_KEXAONE_236B;
    CHECK(ds4_gpu_init());
    ds4_exaone_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->max_seq = BANKS;
    rt->ctx_size = CONTEXT;
    rt->graph = xcalloc(BANKS, sizeof(*rt->graph));
    rt->cache_len = xcalloc(BANKS, sizeof(*rt->cache_len));
    rt->bank_logits = xcalloc((size_t)BANKS * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(BANKS, 1);
    for (unsigned b = 0; b < BANKS; b++) {
        ds4_exaone_gpu_graph *g = &rt->graph[b];
        g->ctx_size = CONTEXT;
        g->kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
        for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
            g->layer_kv_cap[il] = exaone_graph_layer_kv_cap(il, CONTEXT, CHUNK);
            g->layer_kv[il] = ds4_gpu_tensor_alloc((uint64_t)g->layer_kv_cap[il] * 2 * g->kv_dim * sizeof(uint16_t));
            CHECK(g->layer_kv[il]);
        }
    }
    exaone_ckpt_init(rt);
    CHECK(rt->checkpoint_slab);
    /* 36 local layers, GQA K and V; no Step predictor or global-layer snapshot. */
    CHECK(rt->checkpoint_slot_bytes == 36u * DS4_N_SWA * 2u * rt->graph[0].kv_dim * sizeof(uint16_t));
    fill_bank(rt, 0, FRONTIER);
    rt->bank_logits[17] = 7.0f;
    rt->bank_logits_valid[0] = 1;
    CHECK(exaone_ckpt_capture(rt, 0, FRONTIER, true, 0));
    int slot = exaone_ckpt_find(rt, 0, FRONTIER, FRONTIER);
    CHECK(slot >= 0 && exaone_ckpt_find(rt, 0, FRONTIER - 1, FRONTIER) < 0);
    fill_bank(rt, 0, ADVANCED);
    rt->bank_logits[17] = -1.0f;
    unsigned restored = 0;
    CHECK(exaone_ckpt_restore(rt, 0, 1, (unsigned)slot, FRONTIER, &restored));
    CHECK(restored == FRONTIER && rt->cache_len[1] == FRONTIER);
    check_bank(&rt->graph[1], FRONTIER);
    check_bank(&rt->graph[0], ADVANCED);
    CHECK(rt->bank_logits_valid[1]);
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        CHECK(rt->bank_logits[DS4_N_VOCAB + i] == (i == 17 ? 7.0f : 0.0f));
    }
    CHECK(exaone_ckpt_find(rt, 1, FRONTIER, FRONTIER) == slot);
    CHECK(exaone_ckpt_capture(rt, 0, ADVANCED, true, 0));
    CHECK(exaone_ckpt_restore(rt, 0, 0, (unsigned)slot, FRONTIER, &restored));
    CHECK(exaone_ckpt_find(rt, 0, ADVANCED, ADVANCED) == slot);
    check_bank(&rt->graph[0], FRONTIER);
    exaone_ckpt_drop(rt, 0);
    CHECK(exaone_ckpt_find(rt, 0, FRONTIER, FRONTIER) < 0);
    CHECK(exaone_ckpt_find(rt, 1, FRONTIER, FRONTIER) == slot);
    CHECK(!exaone_ckpt_restore(rt, 0, 1, (unsigned)slot, FRONTIER, &restored));

    enum { SHORT_PREFIX = 8 };
    fill_bank(rt, 0, SHORT_PREFIX);
    CHECK(exaone_ckpt_capture(rt, 0, SHORT_PREFIX, false, 0));
    CHECK(exaone_ckpt_find(rt, 0, SHORT_PREFIX, SHORT_PREFIX) < 0);
    int short_slot = exaone_ckpt_find(rt, 0, SHORT_PREFIX, SHORT_PREFIX + 1);
    CHECK(short_slot >= 0);
    CHECK(exaone_ckpt_restore(rt, 0, 0, (unsigned)short_slot, SHORT_PREFIX, &restored));
    CHECK(!rt->bank_logits_valid[0]);
    check_bank(&rt->graph[0], SHORT_PREFIX);

    ds4_batch_ctx ctx = {.exaone = rt};
    exaone_ckpt_drop(rt, 0);
    const uint64_t bytes = ds4_gpu_tensor_bytes(rt->checkpoint_slab);
    uint64_t before = ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, bytes);
    uint64_t freed = ds4_batch_ctx_trim_free(&ctx, UINT64_MAX);
    uint64_t after = ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, bytes);
    CHECK(freed > 0 && after > 0 && before - after == freed);
    CHECK(exaone_ckpt_restore(rt, 1, 0, (unsigned)slot, FRONTIER, &restored));
    check_bank(&rt->graph[0], FRONTIER);
    exaone_ckpt_drop(rt, 0);
    exaone_ckpt_drop(rt, 1);
    before = ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, bytes);
    CHECK(ds4_batch_ctx_trim_free(&ctx, UINT64_MAX) == before);
    CHECK(ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, bytes) == 0);
    fill_bank(rt, 0, FRONTIER);
    rt->bank_logits_valid[0] = 1;
    CHECK(exaone_ckpt_capture(rt, 0, FRONTIER, true, 0));
    slot = exaone_ckpt_find(rt, 0, FRONTIER, FRONTIER);
    CHECK(slot >= 0 && exaone_ckpt_restore(rt, 0, 1, (unsigned)slot, FRONTIER, &restored));
    check_bank(&rt->graph[1], FRONTIER);
    exaone_batch_runtime_free(rt);
    ds4_gpu_cleanup();
    puts("EXAONE checkpoint: LLLG wrap, global prefix, logits, fork, truncate, lineage, reclaim/remap PASS");
    return 0;
}
