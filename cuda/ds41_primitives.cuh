/* ds41_primitives.cuh — DeepSeek V4.1 (ds41) decode-side primitives, ported
 * from the C engine at /data/YoungAi (commit 3946dbc), file
 * `src/cuda/cuda_v41_1.inc.cu` (scratch/grow lines 7-88, v41_bf16r :120) and
 * `src/cuda/cuda_internal.cuh` (PDL, :204-218).
 *
 * Scope note: only what the VQ decode family consumes is ported here. The
 * engine's scratch registry and `ds4_gpu_v41_scratch_release` stay behind
 * until the ds41 state wiring lands (P4-2); the grow-only discipline and the
 * generation counter are kept because captured decode graphs must be
 * re-captured when a scratch pointer moves (the engine's 2026-09-19
 * conviction: a graph baked the pre-growth pointer and read a freed page).
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* `V41_GEMV_MAX_TOK` = `DS4_V41_GEMV_MAX_TOK` (`ds4_gpu_v41.h:265`): the largest
 * token count the fused GEMV decode path takes; above it the prefill family
 * runs. The Rust host splits on the same value. */
#define V41_GEMV_MAX_TOK 8u

/* Grow-only device scratch slot. One slot per buffer; a grow frees the old
 * block and bumps the generation so a captured decode graph is re-captured. */
typedef struct { void *p; uint64_t cap; } v41_scratch;

static uint64_t g_v41_scratch_gen = 0;
extern "C" uint64_t ds4_gpu_v41_scratch_generation(void) { return g_v41_scratch_gen; }   /* the host TU reads it (dg_launch's stale-pointer check) */

static void *v41_grow(v41_scratch *s, uint64_t bytes, const char *what) {
    if (bytes <= s->cap) return s->p;
    (void)cudaDeviceSynchronize();
    if (s->p) (void)cudaFree(s->p);
    s->p = NULL; s->cap = 0;
    if (cudaMalloc(&s->p, (size_t)bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [ds41] %s scratch alloc failed (%.1f MB)\n", what, (double)bytes / 1048576.0);
        return NULL;
    }
    s->cap = bytes;
    g_v41_scratch_gen++;
    /* A grow mid-run re-resolves every consumer's pointer; print it so a
     * reallocation can never hide inside a quiet gate log (a grow landing
     * between decode steps is a determinism lead, not noise). */
    fprintf(stderr, "ds4: [ds41] grow %s -> %.1f MB (gen %llu)\n", what, (double)bytes / 1048576.0,
            (unsigned long long)g_v41_scratch_gen);
    return s->p;
}

/* RNE round f32 -> bf16 -> f32 (`cuda_v41_1.inc.cu:120`). NaN/Inf pass
 * through. Identity on values already on the bf16 grid, which is why the
 * VQ path's "pack activations to bf16" step is lossless by contract. */
__device__ __forceinline__ static float v41_bf16r(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}

/* Programmatic dependent launch (`cuda_internal.cuh:204-218`). A registered
 * kernel waits for its upstream's completion and memory visibility before
 * touching upstream outputs; when the kernel is not launched through PDL the
 * wait returns immediately, so one kernel serves both paths.
 *
 * Portability deviation, named: the engine compiles sm_121 only and writes the
 * bare asm; `griddepcontrol` is sm_90+, so this port guards it. On sm_121 the
 * compiled code is identical; below sm_90 no PDL graph can exist and the wait
 * is a no-op, which is exactly the engine's "not launched through PDL"
 * behavior. */
__device__ __forceinline__ static void v41_pdl_wait(void) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

#define V41_PDL_MAX 64
static const void *g_v41_pdl_ready[V41_PDL_MAX];
static int g_v41_pdl_n = 0;
/* Launch side: declares that this kernel calls v41_pdl_wait before reading
 * upstream output. Duplicate registration is harmless. */
static void v41_pdl_register(const void *fn) {
    for (int i = 0; i < g_v41_pdl_n; i++) if (g_v41_pdl_ready[i] == fn) return;
    if (g_v41_pdl_n < V41_PDL_MAX) g_v41_pdl_ready[g_v41_pdl_n++] = fn;
}
static int v41_pdl_is_ready(const void *fn) {
    for (int i = 0; i < g_v41_pdl_n; i++) if (g_v41_pdl_ready[i] == fn) return 1;
    return 0;
}
