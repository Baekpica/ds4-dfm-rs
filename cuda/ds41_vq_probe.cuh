/* ds41_vq_probe.cuh — the two VQ decode watchdogs, ported from the C engine at
 * /data/YoungAi (commit 3946dbc), file `src/cuda/cuda_vq_probe.inc.cu`.
 *
 * They run only under --v41-prof and change no values: (1) whether activations
 * really sit on the bf16 grid (the premise of packing them as bf16) and (2)
 * whether activations / intermediates exceed f16's exponent range (which would
 * decide f16 vs TF32 if the expert kernels ever move to tensor cores).
 * Counting "elements whose low 16 bits are nonzero" is a harder proof of the
 * grid invariant than comparing output hashes.
 *
 * Order: after ds41_primitives.cuh (v41_grow) and before ds41_vq_decode.cuh
 * (which calls the probe under the prof flag).
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ds41_primitives.cuh"

/* bf16-grid watchdog: any element whose low 16 mantissa bits are nonzero
 * breaks the "packing is identity" invariant. */
__global__ static void v41_vq_offgrid_kernel(uint32_t *cnt, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (__float_as_uint(src[i]) & 0xffffu) != 0u) atomicAdd(cnt, 1u);
}

/* f16-range watchdog: |x| > 65504 overflows f16 to inf (and 0*inf = NaN would
 * silently poison a row); 0<|x|<2^-14 lands in f16 subnormals; below 2^-24 it
 * becomes 0. src is f32 activation, b16 packed bf16 (pass one, NULL the
 * other). */
__global__ static void v41_f16range_kernel(unsigned long long *cnt, const float *src, const uint16_t *b16, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v;
    if (src) v = src[i];
    else v = __uint_as_float((uint32_t)b16[i] << 16);
    const float a = fabsf(v);
    atomicAdd(cnt + 3, 1ull);
    if (!(a <= 65504.f)) atomicAdd(cnt + 0, 1ull);
    else if (a > 0.f && a < 6.103515625e-05f) {
        atomicAdd(cnt + 1, 1ull);
        if (a < 5.9604645e-08f) atomicAdd(cnt + 2, 1ull);
    }
}

static v41_scratch g_v41_f16rng;

/* Cumulative counters, one set per probed quantity (a shared accumulator once
 * printed a label from one call and numbers from both — a lying probe is worse
 * than none). slot: 0 = activation, 1 = intermediate. */
static void v41_f16range_probe(const float *src, const uint16_t *b16, uint64_t n, int slot, const char *what) {
    unsigned long long *c = (unsigned long long *)v41_grow(&g_v41_f16rng, 32, "v41 f16 range");
    unsigned long long h[4] = { 0, 0, 0, 0 };
    static unsigned long long tot[2][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
    static uint64_t calls[2] = { 0, 0 };
    if (!c || slot < 0 || slot > 1) return;
    if (cudaMemsetAsync(c, 0, 32, ds4_current_stream()) != cudaSuccess) return;
    v41_f16range_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(c, src, b16, n);
    if (cudaStreamSynchronize(ds4_current_stream()) != cudaSuccess) return;
    if (cudaMemcpy(h, c, 32, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    for (int i = 0; i < 4; i++) tot[slot][i] += h[i];
    if ((++calls[slot] % 200u) == 0u)
        fprintf(stderr, "[f16-range] cumulative %s: overflow %llu / subnormal %llu / flushed %llu / %llu elements\n",
                what, tot[slot][0], tot[slot][1], tot[slot][2], tot[slot][3]);
}
