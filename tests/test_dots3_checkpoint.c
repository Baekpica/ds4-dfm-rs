/* No weights: dots3 local latent/RoPE rings and full DSA cache ownership. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "dots3 checkpoint FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
enum { BANKS = 2, CONTEXT = 2048, CHUNK = 64, FRONTIER = 700, ADVANCED = 1600 };

static uint16_t value(unsigned layer, unsigned pos, unsigned lane) {
    return (uint16_t)(1 + layer * CONTEXT + pos + lane * 7);
}

static void fill_bank(ds4_dots3_gpu_graph *g, unsigned frontier) {
    for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        const unsigned cap = g->layer_cache_cap[il];
        const unsigned width[] = {ds4_dots3_layer_kv_lora(il), DS4_N_ROT};
        ds4_gpu_tensor *tensor[] = {g->layer_kv_latent[il], g->layer_k_pe[il]};
        for (unsigned part = 0; part < 2; part++) {
            uint16_t *rows = xcalloc((size_t)cap * width[part], sizeof(*rows));
            for (unsigned pos = 0; pos < frontier; pos++) {
                for (unsigned j = 0; j < width[part]; j++) {
                    rows[(size_t)(pos % cap) * width[part] + j] = value(il, pos, j + part);
                }
            }
            CHECK(ds4_gpu_tensor_write(tensor[part], 0, rows, (uint64_t)cap * width[part] * sizeof(*rows)));
            free(rows);
        }
        if (g->layer_idx_k[il]) {
            float *idx = xcalloc((size_t)cap * DS4_N_INDEXER_HEAD_DIM, sizeof(*idx));
            for (unsigned pos = 0; pos < frontier; pos++) {
                for (unsigned j = 0; j < DS4_N_INDEXER_HEAD_DIM; j++) {
                    idx[(size_t)pos * DS4_N_INDEXER_HEAD_DIM + j] = (float)value(il, pos, j);
                }
            }
            CHECK(ds4_gpu_tensor_write(g->layer_idx_k[il], 0, idx, (uint64_t)cap * DS4_N_INDEXER_HEAD_DIM * sizeof(*idx)));
            free(idx);
        }
    }
    g->cache_len = frontier;
}

static void check_bank(ds4_dots3_gpu_graph *g, unsigned frontier) {
    for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        const unsigned cap = g->layer_cache_cap[il];
        const unsigned start = !ds4_dots3_layer_is_full_attention(il) && frontier > DS4_N_SWA
            ? frontier - DS4_N_SWA : 0;
        const unsigned width[] = {ds4_dots3_layer_kv_lora(il), DS4_N_ROT};
        ds4_gpu_tensor *tensor[] = {g->layer_kv_latent[il], g->layer_k_pe[il]};
        for (unsigned part = 0; part < 2; part++) {
            uint16_t *rows = xmalloc((size_t)cap * width[part] * sizeof(*rows));
            CHECK(ds4_gpu_tensor_read(tensor[part], 0, rows, (uint64_t)cap * width[part] * sizeof(*rows)));
            for (unsigned pos = start; pos < frontier; pos++) {
                for (unsigned j = 0; j < width[part]; j++) {
                    CHECK(rows[(size_t)(pos % cap) * width[part] + j] == value(il, pos, j + part));
                }
            }
            free(rows);
        }
        if (g->layer_idx_k[il]) {
            float *idx = xmalloc((size_t)frontier * DS4_N_INDEXER_HEAD_DIM * sizeof(*idx));
            CHECK(ds4_gpu_tensor_read(g->layer_idx_k[il], 0, idx, (uint64_t)frontier * DS4_N_INDEXER_HEAD_DIM * sizeof(*idx)));
            for (unsigned pos = 0; pos < frontier; pos++) {
                for (unsigned j = 0; j < DS4_N_INDEXER_HEAD_DIM; j++) {
                    CHECK(idx[(size_t)pos * DS4_N_INDEXER_HEAD_DIM + j] == (float)value(il, pos, j));
                }
            }
            free(idx);
        }
    }
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_DOTS3_NOTE_PREV;
    setenv("DS4_DOTS3_PREFILL_CHUNK", "4096", 1);
    CHECK(dots3_graph_memory_estimate(8192) == UINT64_C(6112078720));
    CHECK(dots3_graph_memory_estimate(262144) == UINT64_C(11735591808));
    CHECK(ds4_gpu_init());
    ds4_dots3_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->max_seq = BANKS;
    rt->ctx_size = CONTEXT;
    rt->prefill_cap = CHUNK;
    rt->graph = xcalloc(BANKS, sizeof(*rt->graph));
    rt->bank_logits = xcalloc((size_t)BANKS * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(BANKS, 1);
    for (unsigned b = 0; b < BANKS; b++) {
        ds4_dots3_gpu_graph *g = &rt->graph[b];
        g->ctx_cap = CONTEXT;
        g->cap = CHUNK;
        g->ready = g->cache_ready = true;
        for (unsigned il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
            const unsigned cap = dots3_graph_layer_cache_cap(CONTEXT, CHUNK, il);
            g->layer_cache_cap[il] = cap;
            g->layer_kv_latent[il] = ds4_gpu_tensor_alloc((uint64_t)cap * ds4_dots3_layer_kv_lora(il) * sizeof(uint16_t));
            g->layer_k_pe[il] = ds4_gpu_tensor_alloc((uint64_t)cap * DS4_N_ROT * sizeof(uint16_t));
            CHECK(g->layer_kv_latent[il] && g->layer_k_pe[il]);
            if (ds4_dots3_layer_is_full_attention(il)) {
                g->layer_idx_k[il] = ds4_gpu_tensor_alloc((uint64_t)cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                CHECK(g->layer_idx_k[il]);
            }
        }
    }
    dots3_ckpt_init(rt);
    CHECK(rt->checkpoint_slab);
    CHECK(rt->checkpoint_slot_bytes == 33u * DS4_N_SWA * (DS4_N_SWA_KV_LORA + DS4_N_ROT) * sizeof(uint16_t));
    fill_bank(&rt->graph[0], FRONTIER);
    rt->bank_logits[17] = 7;
    rt->bank_logits_valid[0] = 1;
    CHECK(dots3_ckpt_capture(rt, 0, FRONTIER, true, 0));
    int slot = dots3_ckpt_find(rt, 0, FRONTIER, FRONTIER);
    CHECK(slot >= 0 && dots3_ckpt_find(rt, 0, FRONTIER - 1, FRONTIER) < 0);
    fill_bank(&rt->graph[0], ADVANCED);
    unsigned restored = 0;
    CHECK(dots3_ckpt_restore(rt, 0, 1, (unsigned)slot, FRONTIER, &restored));
    CHECK(restored == FRONTIER && rt->graph[1].cache_len == FRONTIER);
    check_bank(&rt->graph[0], ADVANCED);
    check_bank(&rt->graph[1], FRONTIER);
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        CHECK(rt->bank_logits[DS4_N_VOCAB + i] == (i == 17 ? 7 : 0));
    }
    CHECK(dots3_ckpt_restore(rt, 0, 0, (unsigned)slot, FRONTIER, &restored));
    check_bank(&rt->graph[0], FRONTIER);
    dots3_bank_reset(rt, 0);
    CHECK(dots3_ckpt_find(rt, 0, FRONTIER, FRONTIER) < 0);
    CHECK(dots3_ckpt_find(rt, 1, FRONTIER, FRONTIER) == slot);
    CHECK(!dots3_ckpt_restore(rt, 0, 1, (unsigned)slot, FRONTIER, &restored));
    CHECK(dots3_bank_copy(rt, 1, 0, FRONTIER));
    check_bank(&rt->graph[0], FRONTIER);
    dots3_bank_reset(rt, 0);
    dots3_bank_reset(rt, 1);
    CHECK(dots3_ckpt_trim(rt, UINT64_MAX) > 0);
    CHECK(ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, ds4_gpu_tensor_bytes(rt->checkpoint_slab)) == 0);
    fill_bank(&rt->graph[0], 8);
    CHECK(dots3_ckpt_capture(rt, 0, 8, false, 0));
    CHECK(dots3_ckpt_find(rt, 0, 8, 8) < 0);
    slot = dots3_ckpt_find(rt, 0, 8, 9);
    CHECK(slot >= 0 && dots3_ckpt_restore(rt, 0, 1, (unsigned)slot, 8, &restored));
    CHECK(!rt->bank_logits_valid[1]);
    check_bank(&rt->graph[1], 8);
    dots3_batch_free(rt);
    ds4_gpu_cleanup();
    puts("dots3 checkpoint: local latent/RoPE wrap, full DSA prefix, fork/truncate, logits, lineage, remap PASS");
    return 0;
}
