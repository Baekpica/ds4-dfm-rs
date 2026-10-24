/* v41_quant.cu — V4.1 专家权重量化, GPU 原地量化-反量化(2026-09-11)。
 *
 * 【定位与分工】算法在这里(C/CUDA), 教师/学生前向脚本只负责把 GPU 指针递进来 ——
 * 全仓零 Python 数值链的铁律: Python 不许另写一套量化。本库导出的函数直接吃 torch
 * cuda tensor 的 data_ptr(), 原地把权重改成"量化后再解回来"的值, 所以调用方零解包逻辑,
 * 也不需要 85 GB 的中间文件。
 *
 * 【为什么是量化-反量化而不是产出紧凑格式】现在要回答的问题是"这个位宽的质量是多少",
 * 走的是教师同一条前向路(差异只来自权重)。紧凑格式是部署时的事, 那一版会另出。
 * ★体积不由这里决定★ —— 由 (nbit, blk) 决定, 账在 st_volume_audit.sh 里算:
 *   落地 bpw = nbit + 8/blk   (每 blk 个元素一个 ue8m0 scale)
 *   nbit=1, blk=32 ⇒ 1.25 bpw
 *
 * 【码本】高斯 Lloyd-Max 最优标量码本 × 每块 σ。engram/专家两针都实测过: 32 元素块内
 * 是各向同性高斯形状, 所以这就是该位宽下标量量化的理论最优 —— 测出来的是率失真下界,
 * 不是"量化器没调好"。1bit 的 0.7979 恰是 √(2/π)。
 *
 * 编译: nvcc -O3 -fmad=false --compiler-options -fPIC -shared -o libv41quant.so v41_quant.cu
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

/* ★码本来自 v41_codebook(按真实权重分布 DP 求的全局最优)★, 不再假设高斯。
 * 首版用高斯 Lloyd-Max 撞过坑: V4.1 专家出厂是 FP4(16 个离散值), 高斯最优点套不上,
 * 4.25 bpw 白白吃到 KLD 0.41。DP 解出来的 4bit 码本几乎精确还原 FP4 的 16 个值
 * (cos 0.9995) —— 码本必须匹配源分布。
 * 通用 K 点(不强制对称): FP4 里 ±0 两槽合占 11.65%, 强制"符号+幅度"表示不了 0。 */
__constant__ float CBK[256];
static int g_K = 0;

extern "C" int v41_set_codebook(const float *cb, int k) {
    if (k < 2 || k > 256) { fprintf(stderr, "★码本大小 %d 不合法★\n", k); return 1; }
    cudaError_t e = cudaMemcpyToSymbol(CBK, cb, sizeof(float) * k);
    if (e != cudaSuccess) { fprintf(stderr, "★码本上传失败: %s★\n", cudaGetErrorString(e)); return 2; }
    g_K = k;
    return 0;
}

/* 一个 block 处理一行里的一个 blk 元素段。blk 固定 32 ⇒ 用一个 warp 正好。
 * 【为什么用 ue8m0 而不是 f16 存 scale】跟 V4.1 出厂格式同构(专家的 scale 就是每行每 32 列
 * 一个 ue8m0), 位宽账干净: 8/32 = 0.25 bpw。f16 会变成 0.5 bpw, 白吃一倍。 */
__global__ void quant_dequant_kernel(float *w, long long nblk, int blk, int nl, int pow2) {
    long long b = (long long)blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    float *p = w + b * blk;
    int lane = threadIdx.x;

    /* σ = sqrt(mean(w²)), warp 归约 */
    float s2 = 0.f;
    for (int i = lane; i < blk; i += 32) s2 += p[i] * p[i];
    for (int o = 16; o; o >>= 1) s2 += __shfl_down_sync(0xffffffff, s2, o);
    float sigma = __shfl_sync(0xffffffff, sqrtf(s2 / blk), 0);

    /* ★scale 走 ue8m0(2 的幂)★: 真实存储只能表示 2^k, 这里必须同样取整, 否则测出来的
     * 质量比真部署好一截 —— 那种"内部好看、落地劣化"的账本仓吃过亏。 */
    if (sigma <= 0.f) {
        for (int i = lane; i < blk; i += 32) p[i] = 0.f;
        return;
    }
    float q;
    if (pow2) {
        /* ★取最近的 2 的幂, 不是向下取★: floor 会让 q 落在 [σ/2, σ], 系统性低估 σ,
         * 重建值整体偏大。 */
        int e = (int)rintf(log2f(sigma));
        if (e < -127) e = -127;
        if (e > 127) e = 127;
        q = ldexpf(1.0f, e);
    } else {
        q = sigma;   /* 仅用于与 expert_probe.c(精确 σ)对拍, 不是可落地的格式 */
    }

    for (int i = lane; i < blk; i += 32) {
        float a = p[i] / q;            /* 通用码本 ⇒ 不再拆符号 */
        int best = 0;
        float bd = fabsf(a - CBK[0]);
        for (int k = 1; k < nl; k++) {
            float d = fabsf(a - CBK[k]);
            if (d < bd) { bd = d; best = k; }
        }
        p[i] = CBK[best] * q;
    }
}

extern "C" int v41_quant_dequant_gpu(void *w, long long numel, int nbit, int blk, int pow2) {
    if (nbit < 1 || nbit > 4 || blk <= 0 || numel % blk) {
        fprintf(stderr, "★v41_quant: 参数不合法 nbit=%d blk=%d numel=%lld★\n", nbit, blk, numel);
        return 1;
    }
    if (g_K != (1 << nbit)) {
        fprintf(stderr, "★码本未设或大小不符: 已设 %d, 需要 %d ——"
                        " 先调 v41_set_codebook(v41_codebook 求的那份)★\n", g_K, 1 << nbit);
        return 4;
    }
    long long nblk = numel / blk;
    dim3 thr(32, 8);
    long long grid = (nblk + thr.y - 1) / thr.y;
    quant_dequant_kernel<<<(unsigned)grid, thr>>>((float *)w, nblk, blk, g_K, pow2);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "★kernel 失败: %s★\n", cudaGetErrorString(e)); return 3; }
    return 0;
}
