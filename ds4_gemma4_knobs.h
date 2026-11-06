#ifndef DS4_GEMMA4_KNOBS_H
#define DS4_GEMMA4_KNOBS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Typed tuning surface for the Gemma-4 family.
 *
 * Every knob here used to be a hand-parsed DS4_GEMMA4_* getenv string read at
 * the point of use: no validation, no listing in the host's config check, and
 * a typo silently took the zero branch (a span of "12x8" became 12 and was
 * then clamped; a miss policy of "cpux" meant evict).  The Rust host now owns
 * the surface: it parses the environment and its own flags into this struct,
 * validates every value and names the knob in the error, and carries the
 * fully resolved struct through ds4_engine_options.  When `present` is set the
 * fields below are authoritative and no getenv call is made; when it is clear
 * (the C test hosts and the deprecated C oracles) the legacy getenv fallbacks
 * below apply unchanged.
 *
 * Keep the field order in sync with ds4_bridge_model_open_options (native/
 * bridge/ds4_bridge.h) and ds4_bridge_model_open_options in crates/ds4-sys. */
typedef struct {
    /* 1 = every field below is authoritative.  The Rust host always fills
     * the whole struct, defaults included, so a zero field is a value, not
     * an absence. */
    int32_t present;

    /* Attention.  span is the key count per split CTA (host-validated
     * 16..4096); mma selects the tensor-core full-layer split kernel. */
    int32_t attn_split_mma;
    int32_t attn_split;
    int32_t attn_fast;
    int32_t attn_prefill;
    int32_t attn_mma;

    /* Graph fusions and per-family kernels. */
    int32_t fuse_rope;
    int32_t head_gemv;
    int32_t router_dot;
    int32_t router_par;
    int32_t q8_batch;
    int32_t moe_persist;
    int32_t moe_fused;
    int32_t vq_prefill;
    int32_t exact_prefill;
    int32_t q4_decode;

    /* Host routed experts and the expert-slot cache.  `moe_miss_cpu` is the
     * miss policy: 0 evicts a resident slot, 1 computes the route on the
     * host.  `moe_host` sources slot fills from the model mmap instead of
     * the shard fds.  `moe_cpu_layers` is the per-layer split: layers below
     * it keep their routed experts host-cold and run top-8 on the CPU, the
     * layers at or above it keep theirs device-resident (llama.cpp's
     * -ncmoe shape).  0 is the all-GPU default and G4_LAYERS is the whole
     * model on the host, which is what `cpu_moe` has always meant. */
    int32_t cpu_moe;
    int32_t moe_host;
    int32_t moe_miss_cpu;
    uint32_t moe_cpu_layers;
    /* The routing-driven GPU slot cache over host-cold experts, llama.cpp's
     * --moe-cache-mib: 0 keeps the lane off, N reserves N MiB of VRAM for
     * expert slots that fill from the mmap (ds4_gemma4_stream.inc). */
    uint32_t moe_cache_mib;
    /* Pinned host ring the slot fills are staged through, in MiB: 0 means
     * the 256 MiB default, otherwise 64..2048.  The ring is what turns the
     * pool's six blocking cudaMemcpy calls per expert into queued copies on
     * the upload stream (ds4_gemma4_stream.inc). */
    uint32_t moe_staging_mib;

    /* Full-layer KV format (0 bf16, 1 E4M3) and the direct VQ GEMV. */
    int32_t full_kv_fp8;
    int32_t vq_gemv;

    uint32_t attn_split_span;
    uint32_t cpu_moe_threads;
    /* 0 keeps the bind's min(ctx, 512) default. */
    uint32_t prefill_chunk;
    /* g4_vq_gemv: 0 auto, 1 warp, 2 block; threads 32..512 in steps of 32. */
    uint32_t vq_gemv_variant;
    uint32_t vq_gemv_threads;
} ds4_gemma4_options;

#ifdef __cplusplus
extern "C" {
#endif

/* The process-wide resolved surface, installed by ds4_engine_open from
 * ds4_engine_options.gemma4.  Defined in ds4.c. */
ds4_gemma4_options *ds4_gemma4_knobs(void);

#ifdef __cplusplus
}
#endif

/* Legacy boolean: the old helpers accepted exactly "1".  The typed path
 * accepts whatever the host validated (0/1/true/false normalized by Rust). */
static inline int ds4_gemma4_knob_bool(int32_t typed, const char *env) {
    if (ds4_gemma4_knobs()->present) { return typed != 0; }
    const char *v = getenv(env);
    return v && (strcmp(v, "1") == 0 || strcmp(v, "true") == 0);
}

/* Requirement switches that also accepted "0" as an explicit off. */
static inline int ds4_gemma4_knob_tristate(int32_t typed, const char *env) {
    if (ds4_gemma4_knobs()->present) { return typed != 0; }
    const char *v = getenv(env);
    if (!v) { return 0; }
    if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0) { return 1; }
    if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0) { return 0; }
    /* A value the legacy parser did not understand took the zero branch. */
    return 0;
}

/* Key span per split CTA.  The host rejects anything outside 16..4096; the
 * legacy fallback keeps its old clamp so the C oracle behaves as before. */
static inline uint32_t ds4_gemma4_knob_split_span(void) {
    if (ds4_gemma4_knobs()->present) { return ds4_gemma4_knobs()->attn_split_span; }
    const char *v = getenv("DS4_GEMMA4_ATTN_SPLIT_SPAN");
    long span = v ? strtol(v, NULL, 10) : 0;
    if (span <= 0) { span = 128; }
    if (span < 16) { span = 16; }
    if (span > 4096) { span = 4096; }
    return (uint32_t)span;
}

/* Host expert pool width: the runtime's 20-thread default, clamped 1..64. */
static inline uint32_t ds4_gemma4_knob_cpu_threads(void) {
    if (ds4_gemma4_knobs()->present) { return ds4_gemma4_knobs()->cpu_moe_threads; }
    int threads = 20;
    const char *v = getenv("DS4_GEMMA4_CPU_MOE_THREADS");
    if (v && atoi(v) > 0) { threads = atoi(v); }
    if (threads < 1) { threads = 1; }
    if (threads > 64) { threads = 64; }
    return (uint32_t)threads;
}

/* Cache-miss policy: 0 evict a resident slot, 1 compute on the host. */
static inline int ds4_gemma4_knob_miss_cpu(void) {
    if (ds4_gemma4_knobs()->present) { return ds4_gemma4_knobs()->moe_miss_cpu != 0; }
    const char *v = getenv("DS4_GEMMA4_MOE_MISS");
    return v && strcmp(v, "cpu") == 0;
}

/* Full-layer KV: 0 bf16, 1 E4M3. */
static inline int ds4_gemma4_knob_full_kv_fp8(void) {
    if (ds4_gemma4_knobs()->present) { return ds4_gemma4_knobs()->full_kv_fp8 != 0; }
    const char *v = getenv("DS4_GEMMA4_FULL_KV");
    return v && strcmp(v, "fp8") == 0;
}

/* Prefill rows per forward; 0 keeps the bind's min(ctx, 512) default. */
static inline uint32_t ds4_gemma4_knob_prefill_chunk(void) {
    if (ds4_gemma4_knobs()->present) { return ds4_gemma4_knobs()->prefill_chunk; }
    const char *v = getenv("DS4_GEMMA4_PREFILL_CHUNK");
    if (!v || !*v) { return 0; }
    char *end = NULL;
    const unsigned long parsed = strtoul(v, &end, 10);
    if (end == v || *end || parsed == 0ul) { return 0; }
    return (uint32_t)parsed;
}

#endif
