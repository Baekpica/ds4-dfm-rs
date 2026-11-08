#ifndef DS4_GEMMA4_KNOBS_H
#define DS4_GEMMA4_KNOBS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Gemma-4 switches are DS4_GEMMA4_* environment variables read where they
 * are used, as every family's kernel switches are (docs/gemma4-vq.md,
 * Diagnostic switches). A switch is on for 1, true or on. */
static inline int ds4_gemma4_knob_bool(const char *env) {
    const char *v = getenv(env);
    return v && (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "on") == 0);
}

/* A default-on path is off only for 0, false or off. */
static inline int ds4_gemma4_knob_off(const char *env) {
    const char *v = getenv(env);
    return v && (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "off") == 0);
}

/* Keys per split-K CTA: 64 by default (the GB10 span sweep), 16..4096. */
static inline uint32_t ds4_gemma4_knob_split_span(void) {
    const char *v = getenv("DS4_GEMMA4_ATTN_SPLIT_SPAN");
    long span = v ? strtol(v, NULL, 10) : 0;
    if (span <= 0) { span = 64; }
    if (span < 16) { span = 16; }
    if (span > 4096) { span = 4096; }
    return (uint32_t)span;
}

/* Host expert pool width: 20 threads by default, never more than the online
 * CPUs (the workers spin between sweeps) and at most 64. */
static inline uint32_t ds4_gemma4_knob_cpu_threads(void) {
    const char *v = getenv("DS4_GEMMA4_CPU_MOE_THREADS");
    long threads = v ? strtol(v, NULL, 10) : 0;
    if (threads <= 0) { threads = 20; }
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus > 0 && threads > cpus) { threads = cpus; }
    if (threads > 64) { threads = 64; }
    return (uint32_t)threads;
}

/* Cache-miss policy: default ON computes non-active (cache-miss) routes on
 * the host CPU while active resident FP8 experts run on Tensor Core GEMM /
 * persistent GPU kernels. Set DS4_GEMMA4_MOE_MISS=evict|gpu|0 to evict a
 * resident slot instead. */
static inline int ds4_gemma4_knob_miss_cpu(void) {
    const char *v = getenv("DS4_GEMMA4_MOE_MISS");
    if (!v || !*v) { return 1; }
    if (strcmp(v, "0") == 0 || strcmp(v, "off") == 0 || strcmp(v, "false") == 0 ||
        strcmp(v, "gpu") == 0 || strcmp(v, "evict") == 0) {
        return 0;
    }
    return 1;
}

/* Full-layer KV: default ON keeps E4M3 FP8 rows (with per-head FP16 scale);
 * set DS4_GEMMA4_FULL_KV=bf16|0 or DS4_CUDA_FP8_KV=0 for BF16. */
static inline int ds4_gemma4_knob_full_kv_fp8(void) {
    if (ds4_gemma4_knob_off("DS4_CUDA_FP8_KV")) { return 0; }
    const char *v = getenv("DS4_GEMMA4_FULL_KV");
    if (!v || !*v) { return 1; }
    if (strcmp(v, "0") == 0 || strcmp(v, "off") == 0 ||
        strcmp(v, "false") == 0 || strcmp(v, "bf16") == 0) {
        return 0;
    }
    return 1;
}

/* Prefill rows per forward; 0 keeps the bind's min(ctx, 512) default. */
static inline uint32_t ds4_gemma4_knob_prefill_chunk(void) {
    const char *v = getenv("DS4_GEMMA4_PREFILL_CHUNK");
    if (!v || !*v) { return 0; }
    char *end = NULL;
    const unsigned long parsed = strtoul(v, &end, 10);
    if (end == v || *end || parsed == 0ul || parsed > UINT32_MAX) { return 0; }
    return (uint32_t)parsed;
}

#endif
