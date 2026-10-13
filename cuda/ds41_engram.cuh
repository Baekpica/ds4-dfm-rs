/* ds41_engram.cuh — DeepSeek V4.1 (ds41) engram device pieces, ported from the
 * C engine at /data/YoungAi (commit 3946dbc):
 *   - the gate kernel + entry: src/cuda/cuda_v41_3.inc.cu:109-141;
 *   - the read-path primitives: src/cuda/cuda_decode_graph.inc.cu:118-171
 *     (the host-flag spin kernel, the pinned->device zero-copy upload) and
 *     src/cuda/cuda_graphcap.inc.cu:254-259 (pinned mapped host memory).
 *
 * The host half of the read path (shard open, O_DIRECT pread, the per-layer
 * rounds) lives in the Rust host (crates/ds4-core EngramShard/EngramHash, P2,
 * §6.6.2); these entries are the device side the forward wires: host allocates
 * pinned rows, uploads them zero-copy, the graph's spin kernel waits on the
 * per-layer flag, and the rows dequant (ds41_fp8blk.cuh) turns them into the
 * bf16 grid the wkv matmul eats.
 *
 * Numerics of the gate (official Engram.forward): per (token, hc row) the
 * block computes sh = sum h^2, sk = sum key^2, sd = sum h*(qw*kw)*key in f32,
 * then rstd = rsqrt(sh/E + eps) * rsqrt(sk/E + eps), dot = sd * rstd *
 * rsqrt(E), mag = sqrt(max(|dot|, 1e-6)), z = copysign(mag, dot),
 * gate = sigmoid(z), h[d] = bf16r(h[d] + gate*val[d]).  The f32 reduction
 * order (per-thread stride + shared-memory tree) and the fast-math expf
 * differ from a host emulation; the output is bf16-quantized, so the compare
 * is at bf16-ulp distance. */
#pragma once

/* engram gate: one block per (token, hc row), 256 threads, three tree
 * reductions in shared memory (cuda_v41_3.inc.cu:109-134). */
__global__ static void v41_engram_gate_kernel(float *hc, const float *kv, const float *qw, const float *kw,
                                              uint32_t E, uint32_t n_hc, float eps) {
    v41_pdl_wait();   /* PDL: first instruction waits for the upstream; no-op when not launched through PDL */
    const uint32_t t = blockIdx.y, c = blockIdx.x;
    float *h = hc + ((uint64_t)t * n_hc + c) * E;
    const float *key = kv + (uint64_t)t * (n_hc + 1u) * E + (uint64_t)c * E;
    const float *val = kv + (uint64_t)t * (n_hc + 1u) * E + (uint64_t)n_hc * E;
    float sh = 0.f, sk = 0.f, sd = 0.f;
    for (uint32_t d = threadIdx.x; d < E; d += blockDim.x) {
        const float hv = h[d], kv_ = key[d], w = qw[c * E + d] * kw[c * E + d];
        sh += hv * hv; sk += kv_ * kv_; sd += hv * w * kv_;
    }
    __shared__ float r[3][256];
    r[0][threadIdx.x] = sh; r[1][threadIdx.x] = sk; r[2][threadIdx.x] = sd; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) { r[0][threadIdx.x] += r[0][threadIdx.x + k]; r[1][threadIdx.x] += r[1][threadIdx.x + k]; r[2][threadIdx.x] += r[2][threadIdx.x + k]; }
        __syncthreads();
    }
    const float rstd = rsqrtf(r[0][0] / (float)E + eps) * rsqrtf(r[1][0] / (float)E + eps);
    const float dot = r[2][0] * rstd * rsqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float z = copysignf(mag, dot);
    const float gate = 1.0f / (1.0f + expf(-z));
    for (uint32_t d = threadIdx.x; d < E; d += blockDim.x) h[d] = v41_bf16r(h[d] + gate * val[d]);
}
extern "C" int ds4_gpu_v41_engram_gate_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *kv, const void *model_map, uint64_t model_size,
                                   uint64_t q_w_offset, uint64_t k_w_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    if (!hc || !kv) return 0;
    const float *qw = (const float *)cuda_model_range_ptr(model_map, q_w_offset, (uint64_t)n_hc * n_embd * 4, "v41 engram q");
    const float *kw = (const float *)cuda_model_range_ptr(model_map, k_w_offset, (uint64_t)n_hc * n_embd * 4, "v41 engram k");
    if (!qw || !kw) return 0;
    v41_engram_gate_kernel<<<dim3(n_hc, n_tok), 256, 0, ds4_current_stream()>>>((float *)hc->ptr, (const float *)kv->ptr, qw, kw, n_embd, n_hc, eps);
    return cuda_ok(cudaGetLastError(), "v41 engram gate");
}

/* ---- the read path's device primitives (cuda_decode_graph.inc.cu) ---- */

/* Pinned, device-mapped host memory (cuda_graphcap.inc.cu:254-259): the
 * upload kernels address it directly, and a pageable cudaMemcpyAsync would
 * degrade to a sync copy (and void a capture). */
extern "C" void *ds4_gpu_host_alloc(uint64_t bytes) {
    void *p = NULL;
    if (cudaHostAlloc(&p, (size_t)bytes, cudaHostAllocMapped) != cudaSuccess) { (void)cudaGetLastError(); return NULL; }
    return p;
}
extern "C" void ds4_gpu_host_free(void *p) { if (p) (void)cudaFreeHost(p); }

/* GPU spin on a host flag, replacing graph host nodes: on GB10 every host
 * node leaves a 0.6-0.9 ms hole (the driver callback wakeup), while the
 * 1-thread spin kernel resumes in ~1 us.  Semantics: return once flag >= want
 * (both pinned; want is this step's sequence, written by the host before the
 * launch; the sequence only grows, so no ABA).  If the host's read failed and
 * never sets the flag, the ~5 s timeout writes err and lets the kernel
 * through; the host checks err after the sync and stops.  __ldcv
 * (ld.global.cv) re-fetches at each coherence point — plain reads saw stale
 * L2 lines on GB10 (a whole token of lag, hit in the V4 era). */
__global__ static void decode_flagwait_kernel(const int32_t *flag, const int32_t *want, int32_t *err) {
    const int32_t w = __ldcv(want);
    for (uint32_t i = 0; ; i++) {
        if (__ldcv(flag) >= w) return;
        if (i >= 5000000u) { *err = 1; return; }
        __nanosleep(1000);
    }
}
extern "C" int ds4_gpu_host_flag_wait(const void *flag_pinned, const void *want_pinned, void *err_pinned) {
    void *f = NULL, *w = NULL, *e = NULL;
    if (cudaHostGetDevicePointer(&f, (void *)flag_pinned, 0) != cudaSuccess || cudaHostGetDevicePointer(&w, (void *)want_pinned, 0) != cudaSuccess ||
        cudaHostGetDevicePointer(&e, err_pinned, 0) != cudaSuccess || !f || !w || !e) { (void)cudaGetLastError(); return 0; }
    decode_flagwait_kernel<<<1, 1, 0, ds4_current_stream()>>>((const int32_t *)f, (const int32_t *)w, (int32_t *)e);
    return cuda_ok(cudaGetLastError(), "host flag wait");
}

/* Zero-copy upload from pinned host memory (cuda_decode_graph.inc.cu:147-171):
 * a graph memcpy node costs ~170 us on GB10; a 256-thread kernel reading the
 * pinned memory through its device address costs 2-3 us.  __ldcv is mandatory
 * — GB10's L2 caches host-memory lines and plain reads return the previous
 * step's values.  The launch stream is ds4_current_stream() so an eager call
 * orders with the eager neighbors and a captured one records the capture
 * stream (the engine's 2026-09-30 fix for concurrent lanes). */
__global__ static void decode_hostcopy_kernel(uint32_t *dst, const uint32_t *src_host, uint32_t nwords) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) dst[i] = __ldcv(src_host + i);
}
extern "C" int ds4_gpu_tensor_write_zerocopy(ds4_gpu_tensor *t, uint64_t offset, const void *pinned, uint64_t bytes) {
    if (!t || !pinned || offset > t->bytes || bytes > t->bytes - offset || (bytes & 3u) || (offset & 3u)) return 0;
    if (bytes == 0) return 1;
    void *dev = NULL;
    if (cudaHostGetDevicePointer(&dev, (void *)pinned, 0) != cudaSuccess || !dev) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [ds41] pinned memory has no device address (needs cudaHostAllocMapped)\n");
        return 0;
    }
    const uint32_t nw = (uint32_t)(bytes / 4u);
    decode_hostcopy_kernel<<<(nw + 255u) / 256u, 256, 0, ds4_current_stream()>>>((uint32_t *)((char *)t->ptr + offset),
                                                                                  (const uint32_t *)dev, nw);
    return cuda_ok(cudaGetLastError(), "tensor write zerocopy");
}

/* The reverse direction (cuda_decode_graph.inc.cu:172-183): a kernel writes a
 * few device words into the mapped pinned buffer (the graph's per-row argmax
 * landing), saving the ~170 us D2H memcpy node; the host reads the pinned
 * buffer after the step's sync.  Same stream rule as the upload: the capture
 * stream during capture, so the node lands in the graph. */
__global__ static void decode_hoststore_kernel(uint32_t *dst_host, const uint32_t *src, uint32_t nwords) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) dst_host[i] = src[i];
}
extern "C" int ds4_gpu_tensor_read_zerocopy(void *pinned, const ds4_gpu_tensor *t, uint64_t offset, uint64_t bytes) {
    if (!t || !pinned || offset > t->bytes || bytes > t->bytes - offset || (bytes & 3u) || (offset & 3u)) return 0;
    if (bytes == 0) return 1;
    void *dev = NULL;
    if (cudaHostGetDevicePointer(&dev, pinned, 0) != cudaSuccess || !dev) { (void)cudaGetLastError(); return 0; }
    const uint32_t nw = (uint32_t)(bytes / 4u);
    decode_hoststore_kernel<<<(nw + 255u) / 256u, 256, 0, ds4_current_stream()>>>((uint32_t *)dev,
                                                                                  (const uint32_t *)((const char *)t->ptr + offset), nw);
    return cuda_ok(cudaGetLastError(), "tensor read zerocopy");
}
