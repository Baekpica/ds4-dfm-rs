/* v41_fp4.cu — 骨架张量重量化到 FP4(E2M1 + 每行每 32 列一个 ue8m0 scale), GPU(2026-09-12)。
 *
 * 【落盘格式 = V4.1 出厂 routed 专家的格式】weight I8 [rows, cols/2](低 nibble 偶数列,
 * 高 nibble 奇数列), scale F8_E8M0 [rows, cols/32]。选这个而不是自造格式的原因: 教师端
 * v41_hf_io.dq_fp4 / C 读器 st_read_fp4_block 都已经会读它(逐位对拍过), 骨架 FP4 化后
 * 一行读器代码都不用改 —— 读器少一份实现就少一处"解错不报错只出假数"的口子。
 *
 * 【scale 怎么选】ue8m0 只能是 2 的幂。候选 e ∈ {emin−1, emin, emin+1}, emin = 最小的
 * 不裁剪指数(amax/2^e ≤ 6), 逐块算三次重建 SSE 取最小 —— MXFP4 惯例是 floor(log2 amax)−2
 * 定死一个, 这里多试两个邻居是因为块内分布常常让"允许把最大值裁到 6"的更细 scale 总误差
 * 更小。平局取先遇到的(小 e), 规则固定 ⇒ 确定性。
 *
 * 【什么进这里】F8_E4M3(32×32 块 scale)的注意力/shared 专家/indexer 投影, 以及 BF16 的
 * embed/head/vision 大矩阵。不进: 路由 gate、norm、indexer/compressor 的小 BF16 投影
 * (官方特意留 BF16 的, 精度决定 top-k 选择, 体积又可忽略)。 */
#include "v41_dq.cuh"

/* 一个 warp 处理一个 1×32 块; lane = 块内列 */
__global__ static void fp4_requant_kernel(const float *w, long long nblk, uint8_t *packed, uint8_t *scale, double *acc) {
    long long b = (long long)blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= nblk) return;
    int lane = threadIdx.x;
    const unsigned FULL = 0xffffffffu;
    float v = w[b * 32 + lane];
    float m = fabsf(v);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
    float q = 0.f; int be = 0;
    if (m > 0.f) {
        int emin = (int)ceilf(log2f(m / 6.0f));
        float best = 3.0e38f; be = emin;
        for (int e = emin - 1; e <= emin + 1; e++) {
            if (e < -127 || e > 127) continue;
            float s = ldexpf(1.0f, e);
            float r = ds4_e2m1fn_round(v / s) * s;
            float d = (r - v) * (r - v);
            for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(FULL, d, o);
            if (d < best) { best = d; be = e; }
        }
        q = ds4_e2m1fn_round(v / ldexpf(1.0f, be));
    }
    int mag = 0; float aq = fabsf(q);
    for (int i = 1; i < 8; i++) if (ds4_e2m1fn_value(i) == aq) mag = i;
    unsigned nib = (unsigned)mag | (q < 0.f ? 8u : 0u);
    unsigned other = __shfl_xor_sync(FULL, nib, 1);
    if ((lane & 1) == 0) packed[b * 16 + lane / 2] = (uint8_t)(nib | (other << 4));
    uint8_t sb = (uint8_t)(m > 0.f ? be + 127 : 127);
    if (lane == 0) scale[b] = sb;
    /* 残差统计(部署解码口径: nibble 表 × 2^(sb−127)) */
    float rv = ds4_fp4_nibble_to_f32((uint8_t)nib) * ds4_e8m0_to_f32(sb);
    double d = (double)(rv - v) * (rv - v), en = (double)v * v;
    for (int o = 16; o; o >>= 1) { d += __shfl_xor_sync(FULL, d, o); en += __shfl_xor_sync(FULL, en, o); }
    if (lane == 0) { atomicAdd(&acc[0], d); atomicAdd(&acc[1], en); }
}

/* 主机接口: 一张 HF 张量(F8_E4M3+scale / BF16) → FP4 打包 + scale(主机缓冲)。
 * cols 传逻辑列数; sbr/sbc 是 F8 的 scale 块高/宽(32×32; engram 类 1×32 也走得通)。
 * stats[0]=Σ(w−ŵ)², stats[1]=Σw²。 */
extern "C" int v41_fp4_requant_host(const uint8_t *w, const uint8_t *s, const char *dtype, int rows, int cols,
                                    int sbr, int sbc, uint8_t *packed_out, uint8_t *scale_out, double *stats) {
    static v41_dbuf scratch = {NULL, 0}, dW = {NULL, 0}, dP = {NULL, 0}, dS = {NULL, 0};
    static double *acc = NULL;
    if (cols % 32 || cols <= 0 || rows <= 0) { fprintf(stderr, "★FP4 重量化: cols=%d 不是 32 的倍数★\n", cols); return -1; }
    long long n = (long long)rows * cols, nblk = n / 32;
    if (v41_dbuf_need(&dW, sizeof(float) * n)) return -1;
    if (v41_dbuf_need(&dP, (size_t)n / 2)) return -1;
    if (v41_dbuf_need(&dS, (size_t)nblk)) return -1;
    if (!acc) V41_CK(cudaMalloc(&acc, sizeof(double) * 2));
    int nan = 0;
    if (v41_upload_dequant(w, s, dtype, rows, cols, sbr, sbc, (float *)dW.p, &scratch, &nan)) return -1;
    if (nan) { fprintf(stderr, "★张量含 %d 个 NaN, 拒绝重量化(出厂权重不该有 NaN)★\n", nan); return -2; }
    V41_CK(cudaMemset(acc, 0, sizeof(double) * 2));
    dim3 thr(32, 8);
    fp4_requant_kernel<<<(unsigned)((nblk + thr.y - 1) / thr.y), thr>>>((const float *)dW.p, nblk, (uint8_t *)dP.p, (uint8_t *)dS.p, acc);
    V41_CK(cudaGetLastError());
    V41_CK(cudaMemcpy(packed_out, dP.p, (size_t)n / 2, cudaMemcpyDeviceToHost));
    V41_CK(cudaMemcpy(scale_out, dS.p, (size_t)nblk, cudaMemcpyDeviceToHost));
    V41_CK(cudaMemcpy(stats, acc, sizeof(double) * 2, cudaMemcpyDeviceToHost));
    return 0;
}
