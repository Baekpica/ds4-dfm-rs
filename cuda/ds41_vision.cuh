/* cuda/ds41_vision.cuh — DeepSeek V4.1 vision tower (DeepSeek-ViT + aligner)
 * device primitives.  1:1 port of the engine's
 * src/cuda/cuda_v41_vision.inc.cu (2026-10-13).
 *
 * Only a handful of primitives are new: the ViT matmuls (patch_embed, wqkv,
 * wo, w1/w2, both aligner layers) are BF16 and take the existing bf16 path;
 * rms_norm / residual add / bf16 rounding exist.  Missing: the Linear bias,
 * 2D RoPE, unmasked dense attention, untruncated SwiGLU, GELU, the aligner's
 * 3x3 pixel re-arrange, and the two row helpers that place the visual rows.
 * Discipline: f32 compute, bf16-rounded at the official module boundaries
 * (v41_bf16r), the same as the trunk.
 *
 * ★Weights do NOT live in the model mapping★ (core_v41_vision.c): an image
 * streams them off the sidecar mmap into a temporary device tensor per block,
 * so the two "dev tensor" entries below exist (ds4_gpu_v41_matmul_bf16_dev_tensor
 * and ds4_gpu_v41_rms_norm_dev_tensor).  The engine split the same two out of
 * its model-map variants (cuda_v41_1.inc.cu:302).
 *
 * Getting any of this wrong does not error - the model still generates, it
 * just answers about the image wrongly.  The gate is tools/vision_golden.py's
 * official aligner rows compared per-row by cosine (core_v41_vision_gate.c). */

/* ---- the two model-map entries, retargeted at a device tensor ---- */
/* CUDA v41_bf16_matmul_ptr (cuda_v41_1.inc.cu:302-316): W is already a
 * device-readable pointer.  The GEMV/GEMM dispatch is preserved exactly so the
 * tower's numerics match the trunk's (and the engine's golden). */
extern "C" int ds4_gpu_v41_matmul_bf16_dev_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *W,
                                                  uint64_t in_dim, uint64_t out_dim,
                                                  const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !W || !x || n_tok == 0 || W->bytes < in_dim * out_dim * 2) return 0;
    return v41_bf16_matmul_ptr((const __nv_bfloat16 *)W->ptr, in_dim, out_dim, x, out, n_tok);
}
extern "C" int ds4_gpu_v41_rms_norm_dev_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *w,
                                               uint32_t dim, uint32_t n_tok, float eps) {
    if (!out || !x || !w || w->bytes < (uint64_t)dim * 4) return 0;
    v41_rms_norm_kernel<<<n_tok, 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)x->ptr,
                                                                 (const float *)w->ptr, dim, eps);
    return cuda_ok(cudaGetLastError(), "vision rms norm");
}

/* ---- the ten new primitives (cuda_v41_vision.inc.cu) ---- */
static v41_scratch g_vis_qkv16, g_vis_s, g_vis_p;

/* x[i] = bf16(x[i] + bias[i % dim]) — the official Linear's bias, rounded to
 * bf16 in the same step (cuda_v41_vision.inc.cu:12). */
__global__ static void v41_vision_bias_add_kernel(float *x, const float *b, uint64_t n, uint32_t dim) {
    v41_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = v41_bf16r(x[i] + b[i % dim]);
}
extern "C" int ds4_gpu_v41_vision_bias_add_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *bias, uint32_t n_tok, uint32_t dim) {
    if (!x || !bias || !n_tok || !dim || bias->bytes < (uint64_t)dim * 4) return 0;
    const uint64_t n = (uint64_t)n_tok * dim;
    v41_vision_bias_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const float *)bias->ptr, n, dim);
    return cuda_ok(cudaGetLastError(), "vision bias add");
}

/* 2D RoPE (cuda_v41_vision.inc.cu:38-57): one thread per (token, head, j<rd)
 * rotates q and k in place.  inv_freq[i] = theta^(-2i/rd), i < rd/2; the first
 * half of the rope dims take the row index (t/n_w), the second the column
 * (t%n_w); half-split rotation, result bf16. */
__global__ static void v41_vision_rope_kernel(float *qkv, uint32_t n_tok, uint32_t n_w, uint32_t n_head,
                                              uint32_t head_dim, float theta) {
    v41_pdl_wait();
    const uint32_t rd = head_dim / 2u, nf = rd / 2u;
    const uint64_t total = (uint64_t)n_tok * n_head * rd;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t j = (uint32_t)(i % rd), h = (uint32_t)((i / rd) % n_head), t = (uint32_t)(i / ((uint64_t)rd * n_head));
    const uint32_t fi = j < nf ? j : j - nf;
    const float pos = j < nf ? (float)(t / n_w) : (float)(t % n_w);
    const float inv = 1.0f / powf(theta, (float)(2u * fi) / (float)rd);
    const float f = pos * inv, c = cosf(f), s = sinf(f);
    const uint64_t row = (uint64_t)t * 3u * n_head * head_dim;
    #pragma unroll
    for (uint32_t part = 0; part < 2u; part++) {   /* 0 = q, 1 = k */
        float *b = qkv + row + (uint64_t)part * n_head * head_dim + (uint64_t)h * head_dim;
        const float x1 = b[j], x2 = b[j + rd];
        b[j] = v41_bf16r(x1 * c - x2 * s);
        b[j + rd] = v41_bf16r(x2 * c + x1 * s);
    }
}
extern "C" int ds4_gpu_v41_vision_rope_tensor(ds4_gpu_tensor *qkv, uint32_t n_tok, uint32_t n_w, uint32_t n_head,
                                              uint32_t head_dim, float theta) {
    if (!qkv || !n_tok || !n_w || !n_head || (head_dim % 4u) != 0u) return 0;
    const uint64_t total = (uint64_t)n_tok * n_head * (head_dim / 2u);
    v41_vision_rope_kernel<<<(unsigned)((total + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)qkv->ptr, n_tok, n_w, n_head, head_dim, theta);
    return cuda_ok(cudaGetLastError(), "vision rope");
}

/* Row softmax: one block per row, scale then max-subtract then exp then
 * normalize, output bf16 for the P·V tensor-core product
 * (cuda_v41_vision.inc.cu:66-81). */
__global__ static void v41_vision_softmax_kernel(__nv_bfloat16 *P, const float *S, uint32_t nk, float scale) {
    v41_pdl_wait();
    __shared__ float red[256];
    const float *s = S + (uint64_t)blockIdx.x * nk;
    __nv_bfloat16 *p = P + (uint64_t)blockIdx.x * nk;
    float m = -INFINITY;
    for (uint32_t k = threadIdx.x; k < nk; k += blockDim.x) m = fmaxf(m, s[k] * scale);
    red[threadIdx.x] = m; __syncthreads();
    for (uint32_t o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    m = red[0]; __syncthreads();
    float sum = 0.f;
    for (uint32_t k = threadIdx.x; k < nk; k += blockDim.x) sum += expf(s[k] * scale - m);
    red[threadIdx.x] = sum; __syncthreads();
    for (uint32_t o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    const float inv = 1.0f / red[0];
    for (uint32_t k = threadIdx.x; k < nk; k += blockDim.x) p[k] = __float2bfloat16(expf(s[k] * scale - m) * inv);
}
/* Query-tile byte cap: S(f32)+P(bf16) is H*qc*n*6 B; a 1024-token image has
 * n ~ 8600 and H 16 (0.8 MB/row), so 192 MB caps qc ~ 230 and ~38 tiles per
 * layer - the same order as the trunk prefill scratch
 * (cuda_v41_vision.inc.cu:82-84). */
#define V41_VIS_ATTN_BYTES (192ull << 20)

/* Dense unmasked attention, scale = head_dim^-1/2 (official
 * F.scaled_dot_product_attention; cuda_v41_vision.inc.cu:86-121).  Per-head
 * batched cuBLAS: S = Q·K^T (f32) -> row softmax -> bf16 P -> P·V; queries are
 * tiled so the f32 scratch stays under V41_VIS_ATTN_BYTES. */
extern "C" int ds4_gpu_v41_dense_attn_tensor(ds4_gpu_tensor *o, const ds4_gpu_tensor *qkv, uint32_t n_tok, uint32_t n_head, uint32_t head_dim) {
    if (!o || !qkv || !g_cublas_ready || !n_tok || !n_head || !head_dim) return 0;
    const uint32_t HD = n_head * head_dim, W = 3u * HD;
    const uint64_t xn = (uint64_t)n_tok * W;
    __nv_bfloat16 *q16 = (__nv_bfloat16 *)v41_grow(&g_vis_qkv16, xn * 2, "vision qkv bf16");
    if (!q16) return 0;
    v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, ds4_current_stream()>>>(q16, (const float *)qkv->ptr, xn);
    if (!cuda_ok(cudaGetLastError(), "vision qkv->bf16")) return 0;
    uint64_t qc = V41_VIS_ATTN_BYTES / ((uint64_t)n_head * n_tok * 6u);
    if (qc < 1) qc = 1;
    if (qc > n_tok) qc = n_tok;
    float *S = (float *)v41_grow(&g_vis_s, (uint64_t)n_head * qc * n_tok * 4, "vision attn S");
    __nv_bfloat16 *P = (__nv_bfloat16 *)v41_grow(&g_vis_p, (uint64_t)n_head * qc * n_tok * 2, "vision attn P");
    if (!S || !P) return 0;
    const float alpha = 1.0f, beta = 0.0f, scale = 1.0f / sqrtf((float)head_dim);
    (void)cublasSetStream(g_cublas, ds4_current_stream());
    cuda_cublas_ws_prep(ds4_current_stream());
    for (uint32_t q0 = 0; q0 < n_tok; q0 += (uint32_t)qc) {
        const uint32_t nq = (uint32_t)(n_tok - q0 < qc ? n_tok - q0 : qc);
        /* S[h][q][k] = Σ_d K[k][d]·Q[q][d]: column-major this is C(nk × nq) =
         * K^T(nk × D) · Q(D × nq), both strides W (the whole row). */
        cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n_tok, (int)nq, (int)head_dim, &alpha,
                                                       q16 + HD, CUDA_R_16BF, (int)W, (long long)head_dim,
                                                       q16 + (uint64_t)q0 * W, CUDA_R_16BF, (int)W, (long long)head_dim, &beta,
                                                       S, CUDA_R_32F, (int)n_tok, (long long)n_tok * nq, (int)n_head,
                                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "vision attn QK^T")) return 0;
        v41_vision_softmax_kernel<<<(unsigned)(n_head * nq), 256, 0, ds4_current_stream()>>>(P, S, n_tok, scale);
        if (!cuda_ok(cudaGetLastError(), "vision attn softmax")) return 0;
        /* O[h][q][d] = Σ_k V[k][d]·P[q][k]: C(D × nq) = V(D × nk) · P(nk × nq);
         * O lands directly in o's [q0.., h·D..] span (row stride HD). */
        st = cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)head_dim, (int)nq, (int)n_tok, &alpha,
                                        q16 + 2u * HD, CUDA_R_16BF, (int)W, (long long)head_dim,
                                        P, CUDA_R_16BF, (int)n_tok, (long long)n_tok * nq, &beta,
                                        (float *)o->ptr + (uint64_t)q0 * HD, CUDA_R_32F, (int)HD, (long long)head_dim, (int)n_head,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "vision attn PV")) return 0;
    }
    return ds4_gpu_v41_round_bf16_tensor(o, (uint64_t)n_tok * HD);
}

/* Untruncated SwiGLU on the packed [gate|up] halves:
 * h = bf16(silu(gate)·up) (cuda_v41_vision.inc.cu:124-135). */
__global__ static void v41_vision_swiglu_pair_kernel(float *h, const float *gu, uint64_t n, uint32_t mid) {
    v41_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t t = i / mid, j = i % mid;
    const float g = gu[t * 2u * mid + j], u = gu[t * 2u * mid + mid + j];
    h[i] = v41_bf16r((g / (1.0f + expf(-g))) * u);
}
extern "C" int ds4_gpu_v41_swiglu_pair_tensor(ds4_gpu_tensor *h, const ds4_gpu_tensor *gu, uint32_t n_tok, uint32_t mid) {
    if (!h || !gu || !n_tok || !mid) return 0;
    const uint64_t n = (uint64_t)n_tok * mid;
    v41_vision_swiglu_pair_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)h->ptr, (const float *)gu->ptr, n, mid);
    return cuda_ok(cudaGetLastError(), "vision swiglu");
}

/* In-place exact-erf GELU (official F.gelu default; cuda_v41_vision.inc.cu:139-149). */
__global__ static void v41_vision_gelu_kernel(float *x, uint64_t n) {
    v41_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = v41_bf16r(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
}
extern "C" int ds4_gpu_v41_gelu_tensor(ds4_gpu_tensor *x, uint64_t n) {
    if (!x || !n) return 0;
    v41_vision_gelu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)x->ptr, n);
    return cuda_ok(cudaGetLastError(), "vision gelu");
}

/* Aligner r×r pixel re-arrange (official F.pad(0 to a multiple of r) +
 * F.unfold(r, stride=r) + transpose; cuda_v41_vision.inc.cu:152-169):
 * x[n_h·n_w][dim] (patch grid, row-major) -> out[⌈n_h/r⌉·⌈n_w/r⌉][dim·r·r],
 * column order c·r·r + kh·r + kw; outside the grid is 0. */
__global__ static void v41_vision_unfold_kernel(float *out, const float *x, uint32_t n_h, uint32_t n_w, uint32_t dim, uint32_t r, uint32_t ow, uint64_t total) {
    v41_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t cols = dim * r * r;
    const uint32_t col = (uint32_t)(i % cols), cell = (uint32_t)(i / cols);
    const uint32_t oi = cell / ow, oj = cell % ow;
    const uint32_t c = col / (r * r), rem = col % (r * r), kh = rem / r, kw = rem % r;
    const uint32_t pi = oi * r + kh, pj = oj * r + kw;
    out[i] = (pi < n_h && pj < n_w) ? x[((uint64_t)pi * n_w + pj) * dim + c] : 0.0f;
}
extern "C" int ds4_gpu_v41_vision_unfold_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n_h, uint32_t n_w, uint32_t dim, uint32_t r) {
    if (!out || !x || !n_h || !n_w || !dim || !r) return 0;
    const uint32_t oh = (n_h + r - 1u) / r, ow = (n_w + r - 1u) / r;
    const uint64_t total = (uint64_t)oh * ow * dim * r * r;
    v41_vision_unfold_kernel<<<(unsigned)((total + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, n_h, n_w, dim, r, ow, total);
    return cuda_ok(cudaGetLastError(), "vision unfold");
}

/* Row replace: rows[i] >= 0 => x row i = src row rows[i], else untouched - the
 * visual embeddings overwrite the word embeddings at the image positions
 * (official merge_image_embeddings; cuda_v41_vision.inc.cu:171-181). */
__global__ static void v41_row_select_kernel(float *x, const float *src, const int32_t *rows, uint64_t n, uint32_t dim) {
    v41_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t r = rows[i / dim];
    if (r >= 0) x[i] = src[(uint64_t)r * dim + i % dim];
}
extern "C" int ds4_gpu_v41_row_select_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *src, const ds4_gpu_tensor *rows, uint32_t n_tok, uint32_t dim) {
    if (!x || !src || !rows || !n_tok || !dim) return 0;
    const uint64_t n = (uint64_t)n_tok * dim;
    v41_row_select_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const float *)src->ptr, (const int32_t *)rows->ptr, n, dim);
    return cuda_ok(cudaGetLastError(), "vision row select");
}

/* Row zero: rows[i] >= 0 => x row i = 0 - the engram contributes nothing at
 * the image positions (official engram_mask; cuda_v41_vision.inc.cu:183-195). */
__global__ static void v41_rows_zero_kernel(float *x, const int32_t *rows, uint64_t n, uint32_t dim) {
    v41_pdl_wait();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (rows[i / dim] >= 0) x[i] = 0.0f;
}
extern "C" int ds4_gpu_v41_rows_zero_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *rows, uint32_t n_tok, uint32_t dim) {
    if (!x || !rows || !n_tok || !dim) return 0;
    const uint64_t n = (uint64_t)n_tok * dim;
    v41_rows_zero_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>(
        (float *)x->ptr, (const int32_t *)rows->ptr, n, dim);
    return cuda_ok(cudaGetLastError(), "vision rows zero");
}
