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

/* The VQ release runs this graph with its own expert runtime and scratch
 * (the engine's variant; standalone plan tests have the mixed artifact). */
#ifdef DS4_MODEL_VARIANT
static bool mimo2_vq(void) { return DS4_MODEL_VARIANT == DS4_VARIANT_MIMO26_FLASH_VQ; }

/* GVQ linear scratch: MiMo's projections run the plain Q8_0 paths, so the
 * context only borrows its routing vectors and cuBLAS handle. */
enum { M2VQ_GVQ_SCRATCH = 256 };

/* Device bytes the VQ release adds per graph owner: BF16 embedding rows,
 * F32 sinks, the GVQ context and the routed-expert runtime. */
static uint64_t mimo2vq_scratch_bytes(unsigned cap) {
    uint64_t bytes = (uint64_t)cap * M2_EMBED * sizeof(uint16_t) +
        (uint64_t)(MIMO2_LAYERS + MIMO2_DRAFT_LAYERS) * 64 * sizeof(float);
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    bytes += M2VQ_GVQ_SCRATCH + ds4_gpu_gvq_workspace_linear(cap) + ds4_gpu_mimo2vq_moe_bytes(cap);
#endif
    return bytes;
}

/* Full-attention rows of the VQ release: E4M3 K and V heads, then one f16
 * scale per head each (1,296 B, the release's 1M-context cache contract).
 * DS4_MIMO2VQ_FULL_KV=f16 keeps the mixed artifact's f16 rows, a diagnostic
 * of the FP8 rounding. */
static bool mimo2_fp8_rows(unsigned layer) {
    static int fp8 = -1;
    if (!mimo2_vq() || !mimo2_is_full(layer)) { return false; }
    if (fp8 < 0) {
        const char *value = getenv("DS4_MIMO2VQ_FULL_KV");
        fp8 = !(value && !strcmp(value, "f16"));
    }
    return fp8 != 0;
}
#else
static bool mimo2_vq(void) { return false; }
static uint64_t mimo2vq_scratch_bytes(unsigned cap) { (void)cap; return 0; }
static bool mimo2_fp8_rows(unsigned layer) { (void)layer; return false; }
#endif

/* Bytes of one KV row: f16 [K heads | V heads], or the VQ release's FP8 row. */
static uint64_t mimo2_kv_row(unsigned layer) {
    if (mimo2_fp8_rows(layer)) { return 4 * (192 + 128) + 2 * 4 * 2; }
    return (uint64_t)(mimo2_is_full(layer) ? 4 : 8) * (192 + 128) * sizeof(uint16_t);
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
        m.raw_bytes += (uint64_t)mimo2_kv_capacity(il, ctx, cap) * mimo2_kv_row(il);
    }
    // The VQ release replaces the routed rows with its own expert runtime.
    const uint64_t routed = mimo2_vq() ? 0 : 3 * M2_USED * M2_FF + M2_USED * M2_EMBED;
    const uint64_t rows = 4 * M2_EMBED + M2_Q + M2_K + M2_V + M2_HEADS +
        3 * M2_DENSE + M2_EXPERTS + 2 * M2_USED + routed + M2_QKV + 2 + 2 * M2_ROT;
    m.scratch_bytes = ((uint64_t)cap * rows + M2_ROT + DS4_N_VOCAB) * sizeof(float);
    if (mimo2_vq()) { m.scratch_bytes += mimo2vq_scratch_bytes(cap); }
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}

#endif
