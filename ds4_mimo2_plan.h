#ifndef DS4_MIMO2_PLAN_H
#define DS4_MIMO2_PLAN_H

/* Native admission geometry; included after the host shape is defined. */
enum { M2_EMBED = 4096, M2_Q = 64 * 192, M2_K = 8 * 192, M2_V = 8 * 128,
       M2_HEADS = 64 * 128, M2_QKV = M2_Q + M2_K + M2_V, M2_ROT = 64,
       M2_DENSE = 16384, M2_FF = 2048, M2_EXPERTS = 256, M2_USED = 8,
       M2_WINDOW = 128, M2_CONTEXT = 1048576, M2_MAX_BATCH = 8192 };

static bool mimo2_is_full(unsigned layer) {
    return layer == 0 || (layer < MIMO2_LAYERS && layer % 6 == 5);
}

static unsigned mimo2_prefill_cap(unsigned ctx) {
    enum { DEFAULT_BATCH = 4096, MAX_BATCH = 8192 };
    unsigned cap = DEFAULT_BATCH;
    const char *env = getenv("DS4_MIMO2_PREFILL_CHUNK");
    if (env && env[0]) {
        unsigned parsed = 0;
        if (sscanf(env, "%u", &parsed) == 1 && parsed > 0) { cap = parsed; }
    }
    if (cap > MAX_BATCH) { cap = MAX_BATCH; }
    return ctx < cap ? ctx : cap;
}

static unsigned mimo2_kv_capacity(unsigned layer, unsigned ctx, unsigned cap) {
    if (mimo2_is_full(layer)) { return ctx; }
    const unsigned needed = M2_WINDOW + cap - 1;
    return needed < ctx ? needed : ctx;
}

static ds4_context_memory mimo2_memory(unsigned ctx, unsigned cap) {
    ds4_context_memory m = {0};
    if (!ctx || ctx > M2_CONTEXT || !cap || cap > ctx || cap > M2_MAX_BATCH) { return m; }
    m.raw_cap = ctx;
    m.prefill_cap = cap;
    for (unsigned il = 0; il < MIMO2_LAYERS; il++) {
        const unsigned heads = mimo2_is_full(il) ? 4 : 8;
        m.raw_bytes += (uint64_t)mimo2_kv_capacity(il, ctx, cap) * heads * (192 + 128) * sizeof(uint16_t);
    }
    const uint64_t rows = 4 * M2_EMBED + M2_Q + M2_K + M2_V + M2_HEADS +
        3 * M2_DENSE + M2_EXPERTS + 2 * M2_USED +
        3 * M2_USED * M2_FF + M2_USED * M2_EMBED + M2_QKV + 2 + 2 * M2_ROT;
    m.scratch_bytes = ((uint64_t)cap * rows + M2_ROT + DS4_N_VOCAB) * sizeof(float);
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#endif
