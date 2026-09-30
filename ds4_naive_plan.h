#ifndef DS4_NAIVE_PLAN_H
#define DS4_NAIVE_PLAN_H

#include <stdbool.h>
#include <stdint.h>

/* Naive's explicit GQA/DSA geometry; no MLA or compressed history. */
enum {
    N05_LAYERS = 48, N05_DSA_LAYERS = 9, N05_EMBED = 4096,
    N05_VOCAB = 152576, N05_HEADS = 64, N05_KEY = 192, N05_VALUE = 128,
    N05_ROT = 64, N05_WINDOW = 128, N05_DENSE = 16384,
    N05_FF = 2048, N05_EXPERTS = 256, N05_USED = 8,
    N05_INDEX_HEADS = 16, N05_INDEX_DIM = 128, N05_TOP_K = 2048,
    N05_QUERY_TILE = 32, N05_HISTORY_TILE = 4096,
    N05_CONTEXT = 1048576, N05_PREFILL = 2048, N05_PREFILL_MAX = 8192
};

typedef struct {
    uint64_t dsa, swa, index, scratch;
} ds4_naive_memory;

static inline bool naive_is_dsa(unsigned layer) {
    return layer < N05_LAYERS && (layer == 0 || layer % 6 == 5);
}

static inline unsigned naive_kv_heads(unsigned layer) {
    return naive_is_dsa(layer) ? 4 : 8;
}

static inline unsigned naive_kv_capacity(unsigned layer, unsigned ctx, unsigned cap) {
    if (naive_is_dsa(layer)) { return ctx; }
    const unsigned needed = N05_WINDOW - 1 + cap;
    return ctx < needed ? ctx : needed;
}

static inline ds4_naive_memory naive_memory(unsigned ctx, unsigned cap) {
    ds4_naive_memory m = {0};
    if (!ctx || ctx > N05_CONTEXT || !cap || cap > ctx || cap > N05_PREFILL_MAX) { return m; }
    m.dsa = (uint64_t)N05_DSA_LAYERS * ctx * 4 * (N05_KEY + N05_VALUE) * sizeof(uint16_t);
    m.swa = (uint64_t)(N05_LAYERS - N05_DSA_LAYERS) *
        naive_kv_capacity(1, ctx, cap) * 8 * (N05_KEY + N05_VALUE) * sizeof(uint16_t);
    /* Native E4M3 codes plus the original per-row F32 reconstruction scale. */
    m.index = (uint64_t)N05_DSA_LAYERS * ctx * (N05_INDEX_DIM + sizeof(float));
    const uint64_t row = 4 * N05_EMBED + N05_HEADS * N05_KEY + 8 * (N05_KEY + N05_VALUE) +
        N05_HEADS * N05_VALUE + 3 * N05_DENSE + N05_EXPERTS + 2 * N05_USED +
        3 * N05_USED * N05_FF + N05_USED * N05_EMBED + 2 + 2 * N05_ROT +
        N05_INDEX_HEADS * N05_INDEX_DIM + N05_INDEX_DIM + N05_INDEX_HEADS + N05_TOP_K;
    m.scratch = ((uint64_t)cap * row + N05_ROT + N05_VOCAB) * sizeof(float);
    const uint64_t queries = cap < N05_QUERY_TILE ? cap : N05_QUERY_TILE;
    const uint64_t tiles = ((uint64_t)ctx + N05_HISTORY_TILE - 1) / N05_HISTORY_TILE;
    const uint64_t selected = ctx < N05_TOP_K ? ctx : N05_TOP_K;
    m.scratch += queries * ctx * sizeof(float) +
        2 * queries * tiles * selected * (sizeof(float) + sizeof(uint32_t));
    return m;
}

#endif
