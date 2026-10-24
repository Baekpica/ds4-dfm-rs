/* v41_dq.cuh — V4.1 HF 权重格式 → f32 的 device 核, v41_vq.cu / v41_fp4.cu 共用(2026-09-12)。
 *
 * 【口径】与 st_locate.h(C 探针读器)和 v41_hf_io.py(教师端, 逐位对拍过)完全同式:
 *   - E4M3 + ue8m0 块 scale: 值 × 2^(b−127); b=255 是 NaN 槽, 当 0。
 *   - FP4 E2M1 两两打包: 低 nibble = 偶数列, 高 nibble = 奇数列; scale 每行每 32 列一个。
 * 解错不报错只出假数, 所以三处只准有一种写法 —— 标量基元全部来自 src/common/ds4_fp8.h。
 *
 * 【NaN 计数】E4M3 的 0x7f/0xff 是 NaN。出厂权重不该有, 但一旦有, VQ 与 FP4 重量化都会
 * 把它扩散成整块 NaN 而不报错。所以 dequant 顺手数一下, 调用方 >0 就硬停。 */
#ifndef V41_DQ_CUH
#define V41_DQ_CUH
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "../../src/common/ds4_fp8.h"
#include "../../src/common/ds4_float.h"

#define V41_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "★CUDA %s @%s:%d: %s★\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)

/* E4M3 [rows×cols] + ue8m0 scale [rows/sbr × cols/sbc] → f32 */
__global__ static void v41_dq_fp8_kernel(const uint8_t *w, const uint8_t *s, int rows, int cols,
                                         int sbr, int sbc, float *out, int *nan_cnt) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)rows * cols;
    if (i >= n) return;
    int r = (int)(i / cols), c = (int)(i % cols);
    uint8_t sb = s[(long long)(r / sbr) * (cols / sbc) + c / sbc];
    float sc = sb == 255 ? 0.f : ds4_e8m0_to_f32(sb);
    float v = ds4_e4m3fn_to_f32(w[i]);
    if (v != v) { atomicAdd(nan_cnt, 1); v = 0.f; }
    out[i] = v * sc;
}

/* FP4 打包 [rows × cols/2] + ue8m0 scale [rows × cols/32] → f32 [rows × cols](cols = 逻辑列数) */
__global__ static void v41_dq_fp4_kernel(const uint8_t *w, const uint8_t *s, int rows, int cols, float *out) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int packed = cols / 2;
    long long np = (long long)rows * packed;
    if (j >= np) return;
    int r = (int)(j / packed), c0 = (int)(j % packed) * 2;
    uint8_t q = w[j];
    uint8_t sb = s[(long long)r * (cols / 32) + c0 / 32];      /* c0 偶数 ⇒ c0, c0+1 同块 */
    float sc = sb == 255 ? 0.f : ds4_e8m0_to_f32(sb);
    float *o = out + (long long)r * cols + c0;
    o[0] = ds4_fp4_nibble_to_f32(q & 0x0f) * sc;
    o[1] = ds4_fp4_nibble_to_f32((q >> 4) & 0x0f) * sc;
}

__global__ static void v41_dq_bf16_kernel(const uint16_t *w, long long n, float *out, int *nan_cnt) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = ds4_bf16_to_f32(w[i]);
    if (v != v) { atomicAdd(nan_cnt, 1); v = 0.f; }
    out[i] = v;
}

/* 可增长的 device 暂存区: 15360 个专家逐个过, 每次 cudaMalloc/Free 比算还慢。 */
typedef struct { void *p; size_t cap; } v41_dbuf;
static int v41_dbuf_need(v41_dbuf *b, size_t n) {
    if (b->cap >= n) return 0;
    if (b->p) cudaFree(b->p);
    b->p = NULL; b->cap = 0;
    V41_CK(cudaMalloc(&b->p, n));
    b->cap = n;
    return 0;
}

/* 主机上的一张 HF 张量 → device f32。dtype: "F8_E4M3"(要 s/sbr/sbc) / "I8"(FP4, 要 s) / "BF16"。
 * cols 一律传逻辑列数(FP4 = 打包形状 ×2)。dW 由调用方保证 ≥ rows*cols*4 字节。 */
static int v41_upload_dequant(const uint8_t *w, const uint8_t *s, const char *dtype, int rows, int cols,
                              int sbr, int sbc, float *dW, v41_dbuf *scratch, int *nan_out) {
    long long n = (long long)rows * cols;
    int *dnan; static int *g_nan = NULL;
    if (!g_nan) { V41_CK(cudaMalloc(&g_nan, sizeof(int))); }
    dnan = g_nan;
    V41_CK(cudaMemset(dnan, 0, sizeof(int)));
    unsigned thr = 256;
    if (!strcmp(dtype, "F8_E4M3")) {
        size_t wb = (size_t)n, sb = (size_t)(rows / sbr) * (cols / sbc);
        if (v41_dbuf_need(scratch, wb + sb)) return -1;
        uint8_t *dw = (uint8_t *)scratch->p, *ds = dw + wb;
        V41_CK(cudaMemcpy(dw, w, wb, cudaMemcpyHostToDevice));
        V41_CK(cudaMemcpy(ds, s, sb, cudaMemcpyHostToDevice));
        v41_dq_fp8_kernel<<<(unsigned)((n + thr - 1) / thr), thr>>>(dw, ds, rows, cols, sbr, sbc, dW, dnan);
    } else if (!strcmp(dtype, "I8")) {
        size_t wb = (size_t)n / 2, sb = (size_t)rows * (cols / 32);
        if (v41_dbuf_need(scratch, wb + sb)) return -1;
        uint8_t *dw = (uint8_t *)scratch->p, *ds = dw + wb;
        V41_CK(cudaMemcpy(dw, w, wb, cudaMemcpyHostToDevice));
        V41_CK(cudaMemcpy(ds, s, sb, cudaMemcpyHostToDevice));
        v41_dq_fp4_kernel<<<(unsigned)((n / 2 + thr - 1) / thr), thr>>>(dw, ds, rows, cols, dW);
    } else if (!strcmp(dtype, "BF16")) {
        size_t wb = (size_t)n * 2;
        if (v41_dbuf_need(scratch, wb)) return -1;
        V41_CK(cudaMemcpy(scratch->p, w, wb, cudaMemcpyHostToDevice));
        v41_dq_bf16_kernel<<<(unsigned)((n + thr - 1) / thr), thr>>>((const uint16_t *)scratch->p, n, dW, dnan);
    } else {
        fprintf(stderr, "★v41_upload_dequant: 不认识的 dtype %s★\n", dtype);
        return -1;
    }
    V41_CK(cudaGetLastError());
    V41_CK(cudaDeviceSynchronize());
    int nn = 0;
    V41_CK(cudaMemcpy(&nn, dnan, sizeof(int), cudaMemcpyDeviceToHost));
    if (nan_out) *nan_out = nn;
    return 0;
}

/* 统计 Σ(a−b)² 与 Σa²(double 原子累加; 只作报数, 顺序无所谓) */
__global__ static void v41_sse_kernel(const float *a, const float *b, long long n, double *acc) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double se[256], sn[256];
    double d = 0, e = 0;
    if (i < n) { float x = a[i], y = b[i]; d = (double)(x - y) * (x - y); e = (double)x * x; }
    se[threadIdx.x] = d; sn[threadIdx.x] = e;
    __syncthreads();
    for (int o = blockDim.x / 2; o; o >>= 1) {
        if (threadIdx.x < o) { se[threadIdx.x] += se[threadIdx.x + o]; sn[threadIdx.x] += sn[threadIdx.x + o]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(&acc[0], se[0]); atomicAdd(&acc[1], sn[0]); }
}

#endif
