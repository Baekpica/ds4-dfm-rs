/* v41_q4k.cu — 骨架张量重量化到 q4_K(4.5 bpw, 144 B/256 元素), 编码 + GPU 解码(2026-09-19)。
 *
 * 【为什么有这个文件】用户 09-19 令"骨架使用 q4 量化"。骨架此前走 FP4 E2M1 1×32(4.25 bpw,
 * v41_fp4.cu)。q4_K 每 256 元素一个 f16 主 scale + 一个 f16 主 min + 8 组 6-bit 子 scale/min,
 * 表达力比"一个 2 的幂 × 32 列"强得多(它有 min 项 ⇒ 能表示非零均值的块), 代价是 +0.25 bpw。
 *
 * 【零语料】用户 09-19 令"按权重的直接量化, 不要语料的那种" ⇒ 调 ds4q_quantize_chunk 时
 * quant_weights 传 NULL。那条路的逐元素权重是 `av_x + |x|`(av_x = 块内 RMS), **全部从权重
 * 自身算**, 不读任何激活/语料 —— 就是 llama.cpp 不带 imatrix 时的默认 q4_K。
 *
 * 【编码为什么走既有 CPU 实现(而不是另写 GPU 版)】铁律"只写 GPU 路"针对的是新原语与推理路。
 * q4_K 的编码在本仓已有唯一生产实现(gguf-tools/quantize/quants.c 的 qkx3 路, V4 时代的 Q4_K
 * 骨架档就是它量的), 且与 src/common/ds4_deq_q4_K 是逐式同源的一对(ds4_unit 有逐字节金标)。
 * 再写一份 GPU 编码器 = 同一个格式基元出现两份实现, 数值分叉不报错只出假数。量级上也不是
 * 那条铁律的场景: 骨架 7.842 B 参数 = routed 专家的 1/69(铁律的实撞代价是 VQ 在 543 B 上
 * CPU 慢 40 倍)。★耗时按分片实测上报, 若成为瓶颈再 GPU 化★。
 * **解码一律 GPU**(下面这个核): 教师端与残差统计都走它, 不碰 CPU 解码。
 *
 * 【落盘】dtype 写 "Q4_K", shape 记【逻辑形状】[rows, cols], 字节数 = rows×(cols/256)×144。
 * safetensors 的字节数由 data_offsets 决定、不由 dtype 推, 所以自定义 dtype 名是安全的
 * (F8_E8M0 早就是这么用的)。读它的两处: 教师端 v41_hf_io.load_q4k / 转换器 v41_to_gguf。 */
#include "v41_dq.cuh"
extern "C" {
#include "quants.h"
}

/* q4_K 块 → f32, 与 src/common/ds4_quantfmt.c 的 ds4_deq_q4_K 逐式同(那份是金标, 改这里必须对拍)。
 * 一个 warp 一块: lane l 负责第 l 与第 l+32 … 共 8 个元素, 每次取一个 64 元素组的两个半边。 */
__global__ static void v41_q4k_decode_kernel(const uint8_t *src, long long nblk, float *out) {
    long long b = (long long)blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    const uint8_t *blk = src + b * 144;
    int lane = threadIdx.x;                       /* 0..31 = 子块内列 */
    uint16_t hd = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    uint16_t hm = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
    float d = ds4_f16_to_f32(hd), dmin = ds4_f16_to_f32(hm);
    const uint8_t *sc = blk + 4, *q = blk + 16;
    float *y = out + b * 256;
    for (int j = 0; j < 8; j++) {                 /* 8 个 32 元素子块 */
        uint8_t s, m;
        if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
        else { s = (uint8_t)((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
               m = (uint8_t)((sc[j + 4] >> 4)  | ((sc[j - 0] >> 6) << 4)); }
        /* qs 每 32 字节存相邻两个子块: 偶数子块取低 nibble, 奇数子块取高 nibble */
        uint8_t byte = q[(j / 2) * 32 + lane];
        uint8_t v = (j & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
        y[j * 32 + lane] = d * (float)s * (float)v - dmin * (float)m;
    }
}

/* 残差累加: Σ(a−b)², Σa² */
__global__ static void v41_q4k_sse_kernel(const float *a, const float *b, long long n, double *acc) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double d = 0, e = 0;
    if (i < n) { double x = a[i], y = b[i]; d = (x - y) * (x - y); e = x * x; }
    for (int o = 16; o; o >>= 1) { d += __shfl_xor_sync(0xffffffffu, d, o); e += __shfl_xor_sync(0xffffffffu, e, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&acc[0], d); atomicAdd(&acc[1], e); }
}

/* ---- 设备缓冲(与 v41_fp4.cu 同法: 全程复用, 不每张量 malloc/free) ---- */
typedef struct { void *p; size_t n; } q4k_buf;
static int q4k_need(q4k_buf *b, size_t n) {
    if (b->n >= n) return 0;
    if (b->p) cudaFree(b->p);
    V41_CK(cudaMalloc(&b->p, n));
    b->n = n;
    return 0;
}

/* HF 张量(F8_E4M3+scale / BF16) → f32(device)。与 v41_fp4.cu 走同一批 dequant 核。 */
static int q4k_to_f32(const uint8_t *w, const uint8_t *s, const char *dtype, int rows, int cols,
                      int sbr, int sbc, q4k_buf *dW, q4k_buf *dRaw, q4k_buf *dSc, int *nan_dev) {
    long long n = (long long)rows * cols;
    if (q4k_need(dW, (size_t)n * sizeof(float))) return -1;
    V41_CK(cudaMemset(nan_dev, 0, sizeof(int)));
    if (!strcmp(dtype, "F8_E4M3")) {
        if (q4k_need(dRaw, (size_t)n) || q4k_need(dSc, (size_t)(rows / sbr) * (cols / sbc))) return -1;
        V41_CK(cudaMemcpy(dRaw->p, w, (size_t)n, cudaMemcpyHostToDevice));
        V41_CK(cudaMemcpy(dSc->p, s, (size_t)(rows / sbr) * (cols / sbc), cudaMemcpyHostToDevice));
        v41_dq_fp8_kernel<<<(unsigned)((n + 255) / 256), 256>>>((const uint8_t *)dRaw->p, (const uint8_t *)dSc->p,
                                                                rows, cols, sbr, sbc, (float *)dW->p, nan_dev);
    } else if (!strcmp(dtype, "BF16")) {
        if (q4k_need(dRaw, (size_t)n * 2)) return -1;
        V41_CK(cudaMemcpy(dRaw->p, w, (size_t)n * 2, cudaMemcpyHostToDevice));
        v41_dq_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>((const uint16_t *)dRaw->p, n, (float *)dW->p, nan_dev);
    } else {
        fprintf(stderr, "★q4_K 不认的源 dtype %s★\n", dtype);
        return -2;
    }
    V41_CK(cudaDeviceSynchronize());
    int nan_h = 0;
    V41_CK(cudaMemcpy(&nan_h, nan_dev, sizeof(int), cudaMemcpyDeviceToHost));
    if (nan_h) { fprintf(stderr, "★源权重有 %d 个 NaN, 停车★\n", nan_h); return -3; }
    return 0;
}

/* 主机接口: 一张 HF 骨架张量 → q4_K 块流(主机缓冲 blocks_out, rows×(cols/256)×144 字节)。
 * cols 必须是 256 的倍数(调用方保证; 不满足的张量由调用方改走 FP4)。
 * stats[0]=Σ(w−ŵ)², stats[1]=Σw²  —— ŵ 是【从落盘字节 GPU 解回来的值】, 报的就是读回来的残差。 */
extern "C" int v41_q4k_requant_host(const uint8_t *w, const uint8_t *s, const char *dtype, int rows, int cols,
                                    int sbr, int sbc, uint8_t *blocks_out, double *stats) {
    static q4k_buf dW = {NULL, 0}, dRaw = {NULL, 0}, dSc = {NULL, 0}, dBlk = {NULL, 0}, dRec = {NULL, 0};
    static int *nan_dev = NULL;
    static double *acc = NULL;
    static float *hostf = NULL;
    static size_t hostf_n = 0;
    if (cols % 256) { fprintf(stderr, "★q4_K 要 256 的倍数列, 收到 %d★\n", cols); return -2; }
    if (!nan_dev) V41_CK(cudaMalloc(&nan_dev, sizeof(int)));
    if (!acc) V41_CK(cudaMalloc(&acc, 2 * sizeof(double)));

    long long n = (long long)rows * cols;
    if (q4k_to_f32(w, s, dtype, rows, cols, sbr, sbc, &dW, &dRaw, &dSc, nan_dev)) return -1;

    /* 编码在主机: 把 f32 拷回来, 逐行交给 ds4q_quantize_chunk(零语料, quant_weights=NULL) */
    if (hostf_n < (size_t)n) {
        free(hostf);
        hostf = (float *)malloc((size_t)n * sizeof(float));
        if (!hostf) { hostf_n = 0; fprintf(stderr, "★q4_K 主机缓冲 %.2f GB 要不到★\n", n * 4.0 / 1e9); return -4; }
        hostf_n = (size_t)n;
    }
    V41_CK(cudaMemcpy(hostf, dW.p, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    ds4q_quantize_init(DS4Q_TYPE_Q4_K);
    /* start=0: 该参数只给 imatrix 定位行偏移, 这里 imatrix=NULL(零语料) ⇒ 恒 0 */
    size_t nb = ds4q_quantize_chunk(DS4Q_TYPE_Q4_K, hostf, blocks_out, 0, rows, cols, NULL);
    size_t want = (size_t)rows * (size_t)(cols / 256) * 144u;
    if (nb != want) { fprintf(stderr, "★q4_K 写出 %zu 字节, 应为 %zu★\n", nb, want); return -5; }

    /* 残差: 解码走 GPU 核(= 部署/教师读回来的那条路) */
    long long nblk = (long long)rows * (cols / 256);
    if (q4k_need(&dBlk, want) || q4k_need(&dRec, (size_t)n * sizeof(float))) return -1;
    V41_CK(cudaMemcpy(dBlk.p, blocks_out, want, cudaMemcpyHostToDevice));
    V41_CK(cudaMemset(acc, 0, 2 * sizeof(double)));
    dim3 blk(32, 8);
    v41_q4k_decode_kernel<<<(unsigned)((nblk + 7) / 8), blk>>>((const uint8_t *)dBlk.p, nblk, (float *)dRec.p);
    v41_q4k_sse_kernel<<<(unsigned)((n + 255) / 256), 256>>>((const float *)dW.p, (const float *)dRec.p, n, acc);
    V41_CK(cudaDeviceSynchronize());
    V41_CK(cudaMemcpy(stats, acc, 2 * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

/* 教师端(ctypes)与转换器用: 设备上的 q4_K 块流 → f32 [rows, cols]。指针都是 device 指针。 */
extern "C" int v41_q4k_decode_gpu(const void *blocks, int rows, int cols, void *out_f32) {
    if (cols % 256) { fprintf(stderr, "★q4_K decode: 列数 %d 不是 256 倍数★\n", cols); return -2; }
    long long nblk = (long long)rows * (cols / 256);
    dim3 blk(32, 8);
    v41_q4k_decode_kernel<<<(unsigned)((nblk + 7) / 8), blk>>>((const uint8_t *)blocks, nblk, (float *)out_f32);
    V41_CK(cudaDeviceSynchronize());
    return 0;
}
