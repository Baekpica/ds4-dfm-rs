/* Model-free GPU copy gate: overwrite a wrapped SWA ring, then restore its
 * checkpoint and full-attention prefix without changing the source bank. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "checkpoint FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

enum { BANKS = 2, CONTEXT = 1024, CAP = 640, FRONTIER = 700, ADVANCED = 1000 };

static uint16_t row_value(unsigned layer, unsigned pos) {
    return (uint16_t)(1 + layer * CONTEXT + pos);
}

static void fill_bank(ds4_step37_graph *g, unsigned frontier) {
    uint16_t row[2 * S37_KV];
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        for (unsigned p = 0; p < frontier; p++) {
            for (unsigned j = 0; j < 2 * S37_KV; j++) { row[j] = row_value(il, p); }
            CHECK(ds4_gpu_tensor_write(g->kv[il], (uint64_t)(p % g->kv_cap[il]) * sizeof(row), row, sizeof(row)));
        }
    }
    g->position = g->high_water = frontier;
}

static void check_bank(ds4_step37_graph *g, unsigned frontier) {
    uint16_t row[2 * S37_KV];
    for (unsigned il = 0; il < STEP37_LAYERS; il++) {
        unsigned start = step37_sliding(il) && frontier > S37_WINDOW ? frontier - S37_WINDOW : 0;
        for (unsigned p = start; p < frontier; p++) {
            CHECK(ds4_gpu_tensor_read(g->kv[il], (uint64_t)(p % g->kv_cap[il]) * sizeof(row), row, sizeof(row)));
            for (unsigned j = 0; j < 2 * S37_KV; j++) { CHECK(row[j] == row_value(il, p)); }
        }
    }
}

static void fill_predictors(ds4_step37_spec *s, unsigned frontier) {
    s->position = frontier;
    s->tail_rows = frontier < STEP37_DRAFT_LAYERS ? frontier : STEP37_DRAFT_LAYERS;
    unsigned base = frontier - s->tail_rows;
    uint16_t row[2 * S37_KV];
    for (unsigned d = 0; d < STEP37_DRAFT_LAYERS; d++) {
        const unsigned il = STEP37_LAYERS + d;
        for (unsigned p = 0; p < base; p++) {
            for (unsigned j = 0; j < 2 * S37_KV; j++) { row[j] = row_value(il, p); }
            CHECK(ds4_gpu_tensor_write(s->draft.graph.kv[il], (uint64_t)(p % CAP) * sizeof(row), row, sizeof(row)));
        }
        s->draft.position[d] = s->draft.high_water[d] = base;
    }
    CHECK(ds4_gpu_tensor_fill_f32(s->tail, (float)frontier, STEP37_DRAFT_LAYERS * S37_HIDDEN));
}

static void check_predictors(ds4_step37_spec *s, unsigned frontier) {
    CHECK(step37_spec_valid(s) && s->position == frontier);
    const unsigned base = frontier - s->tail_rows;
    uint16_t row[2 * S37_KV];
    for (unsigned d = 0; d < STEP37_DRAFT_LAYERS; d++) {
        const unsigned il = STEP37_LAYERS + d;
        unsigned start = base > S37_WINDOW ? base - S37_WINDOW : 0;
        for (unsigned p = start; p < base; p++) {
            CHECK(ds4_gpu_tensor_read(s->draft.graph.kv[il], (uint64_t)(p % CAP) * sizeof(row), row, sizeof(row)));
            for (unsigned j = 0; j < 2 * S37_KV; j++) { CHECK(row[j] == row_value(il, p)); }
        }
    }
    float tail[STEP37_DRAFT_LAYERS * S37_HIDDEN];
    CHECK(ds4_gpu_tensor_read(s->tail, 0, tail, s->tail_rows * S37_HIDDEN * sizeof(float)));
    for (unsigned j = 0; j < s->tail_rows * S37_HIDDEN; j++) { CHECK(tail[j] == (float)frontier); }
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_STEP37_FLASH;
    CHECK(ds4_gpu_init());
    ds4_step37_batch_runtime *rt = xcalloc(1, sizeof(*rt));
    rt->max_seq = BANKS;
    rt->ctx_size = CONTEXT;
    rt->graph = xcalloc(BANKS, sizeof(*rt->graph));
    rt->spec = xcalloc(BANKS, sizeof(*rt->spec));
    rt->bank_logits = xcalloc(BANKS * DS4_N_VOCAB, sizeof(float));
    rt->bank_logits_valid = xcalloc(BANKS, 1);
    for (unsigned b = 0; b < BANKS; b++) {
        ds4_step37_graph *g = &rt->graph[b];
        g->context = CONTEXT;
        g->cap = CAP - S37_WINDOW;
        for (unsigned il = 0; il < STEP37_LAYERS; il++) {
            g->kv_cap[il] = step37_sliding(il) ? CAP : CONTEXT;
            g->kv[il] = ds4_gpu_tensor_alloc((uint64_t)g->kv_cap[il] * 2 * S37_KV * sizeof(uint16_t));
            CHECK(g->kv[il]);
        }
        ds4_step37_spec *s = &rt->spec[b];
        s->draft.graph.cap = CAP - S37_WINDOW;
        s->draft.graph.context = CONTEXT;
        for (unsigned il = STEP37_LAYERS; il < STEP37_LAYERS + STEP37_DRAFT_LAYERS; il++) {
            s->draft.graph.kv_cap[il] = CAP;
            s->draft.graph.kv[il] = ds4_gpu_tensor_alloc((uint64_t)CAP * 2 * S37_KV * sizeof(uint16_t));
            CHECK(s->draft.graph.kv[il]);
        }
        s->tail = ds4_gpu_tensor_alloc(STEP37_DRAFT_LAYERS * S37_HIDDEN * sizeof(float));
        CHECK(s->tail);
    }
    step37_ckpt_init(rt);
    CHECK(rt->checkpoint_slab);
    fill_bank(&rt->graph[0], FRONTIER);
    fill_predictors(&rt->spec[0], FRONTIER);
    rt->bank_logits[17] = 7.0f;
    rt->bank_logits_valid[0] = 1;
    CHECK(step37_ckpt_capture(rt, 0, FRONTIER, true, 0));
    int slot = step37_ckpt_find(rt, 0, FRONTIER, FRONTIER);
    CHECK(slot >= 0);
    CHECK(step37_ckpt_find(rt, 0, FRONTIER - 1, FRONTIER) == -1);
    fill_bank(&rt->graph[0], ADVANCED);
    fill_predictors(&rt->spec[0], ADVANCED);
    rt->bank_logits[17] = -1.0f;
    unsigned restored = 0;
    CHECK(step37_ckpt_restore(rt, 0, 1, (unsigned)slot, FRONTIER, &restored));
    CHECK(restored == FRONTIER && rt->graph[1].position == FRONTIER);
    CHECK(rt->graph[1].high_water == FRONTIER + rt->graph[1].cap && rt->bank_logits_valid[1]);
    CHECK(rt->bank_logits[DS4_N_VOCAB + 17] == 7.0f);
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        CHECK(rt->bank_logits[DS4_N_VOCAB + i] == (i == 17 ? 7.0f : 0.0f));
    }
    check_bank(&rt->graph[1], FRONTIER);
    check_bank(&rt->graph[0], ADVANCED);
    check_predictors(&rt->spec[1], FRONTIER);
    check_predictors(&rt->spec[0], ADVANCED);
    CHECK(!step37_draft_rewind(&rt->spec[1].draft, 0, FRONTIER - STEP37_DRAFT_LAYERS - 1));
    /* A compact checkpoint restores only the live window. The chunk-sized
     * older ring slack is not present, so rewind below it must be refused. */
    CHECK(!step37_rewind(&rt->graph[1], FRONTIER - 1));
    CHECK(step37_ckpt_find(rt, 1, FRONTIER, FRONTIER) == slot);
    const unsigned shared_slot = (unsigned)slot;

    /* In-place rollback keeps no future lineage, and resetting a bank must
     * not invalidate a checkpoint still referenced by its fork. */
    CHECK(step37_ckpt_restore(rt, 0, 0, (unsigned)slot, FRONTIER, &restored));
    check_bank(&rt->graph[0], FRONTIER);
    check_predictors(&rt->spec[0], FRONTIER);
    CHECK(step37_batch_runtime_reset_bank(rt, 0));
    CHECK(step37_ckpt_find(rt, 0, FRONTIER, FRONTIER) == -1);
    CHECK(step37_ckpt_find(rt, 1, FRONTIER, FRONTIER) == slot);
    CHECK(!step37_ckpt_restore(rt, 0, 1, (unsigned)slot, FRONTIER, &restored));

    /* A mid-prefill snapshot has KV but no frontier distribution. It can
     * serve a suffix replay, never an exact zero-suffix request. */
    enum { SHORT_PREFIX = 8 };
    fill_bank(&rt->graph[0], SHORT_PREFIX);
    fill_predictors(&rt->spec[0], SHORT_PREFIX);
    CHECK(step37_ckpt_capture(rt, 0, SHORT_PREFIX, false, 0));
    CHECK(step37_ckpt_find(rt, 0, SHORT_PREFIX, SHORT_PREFIX) < 0);
    slot = step37_ckpt_find(rt, 0, SHORT_PREFIX, SHORT_PREFIX + 1);
    CHECK(slot >= 0);
    CHECK(step37_ckpt_restore(rt, 0, 0, (unsigned)slot, SHORT_PREFIX, &restored));
    CHECK(!rt->bank_logits_valid[0]);
    check_bank(&rt->graph[0], SHORT_PREFIX);
    check_predictors(&rt->spec[0], SHORT_PREFIX);

    /* Reclaim dead slots through the serial-lane API, retaining shared
     * checkpoints even when their unaligned VMM pages border dead slots. */
    ds4_batch_ctx ctx = {.step37 = rt};
    const uint64_t slab_bytes = ds4_gpu_tensor_bytes(rt->checkpoint_slab);
    uint64_t before = ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, slab_bytes);
    CHECK(before > 0 && ds4_batch_ctx_trim_free(&ctx, 0) == 0);
    CHECK(step37_batch_runtime_reset_bank(rt, 0));
    uint64_t freed = ds4_batch_ctx_trim_free(&ctx, UINT64_MAX);
    uint64_t after = ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, slab_bytes);
    CHECK(freed > 0 && after > 0 && before - after == freed);
    CHECK(step37_ckpt_restore(rt, 1, 0, shared_slot, FRONTIER, &restored));
    check_bank(&rt->graph[0], FRONTIER);
    check_predictors(&rt->spec[0], FRONTIER);
    CHECK(rt->bank_logits[17] == 7.0f);

    CHECK(step37_batch_runtime_reset_bank(rt, 0));
    CHECK(step37_batch_runtime_reset_bank(rt, 1));
    before = ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, slab_bytes);
    CHECK(ds4_batch_ctx_trim_free(&ctx, UINT64_MAX) == before);
    CHECK(ds4_gpu_tensor_resident(rt->checkpoint_slab, 0, slab_bytes) == 0);
    CHECK(ds4_batch_ctx_trim_free(&ctx, UINT64_MAX) == 0);
    /* A reclaimed slot can be mapped and used again. */
    fill_bank(&rt->graph[0], FRONTIER);
    fill_predictors(&rt->spec[0], FRONTIER);
    rt->bank_logits_valid[0] = 1;
    CHECK(step37_ckpt_capture(rt, 0, FRONTIER, true, 0));
    slot = step37_ckpt_find(rt, 0, FRONTIER, FRONTIER);
    CHECK(slot >= 0 && step37_ckpt_restore(rt, 0, 1, (unsigned)slot, FRONTIER, &restored));
    check_bank(&rt->graph[1], FRONTIER);
    check_predictors(&rt->spec[1], FRONTIER);
    step37_batch_runtime_free(rt);
    ds4_gpu_cleanup();
    puts("Step checkpoint: wrapped SWA, full KV, logits, fork, lineage, reclaim/remap PASS");
    return 0;
}
