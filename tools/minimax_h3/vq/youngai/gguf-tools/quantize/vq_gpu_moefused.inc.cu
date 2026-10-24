/* vq_gpu_moefused.inc.cu — VQ-fused 批量专家 GEMM(dequant 融进 kernel)。只被 vq_gpu.cu include。
 *
 * ★为什么(2026-08-30 用户"理论上一个单元 20s")★ BFLT 定谳: 深单元 bdq 58s + gemm 62s ——
 * 每次层前向把整层 256 专家×3 矩阵 dequant 成 26GB fp32(写带宽墙), 再喂 n≈12 的瘦 GEMM
 * (0.4 TFLOPS=峰值 2%)。权重整个 sweep 不变, 物化是纯重复劳动。
 * 融合: 码本(nc≤1024×dim4 fp16→smem float ≤16KB)+行增益+紧凑索引即读即解码即 FMA,
 * 权重流量 = 索引 ~2.6MB/矩阵(vs 33.5MB fp32), X/H 分块进 smem 复用。
 * 数值: fp32 逐段顺序累加(确定性), 与 cublas 归约序不同属既有 GPU 语义容差;
 * 宿主侧首 2 调用与旧路(dequant+cublas)全量对拍, relerr>1e-4 → 大声永久回退, 不静默。 */

#define VQF_TT 16   /* token 瓦片: acc[16] 全展开进寄存器(动态下标会退局部内存) */
typedef struct { uint64_t o1,o3,o2; int nc13,nb13,nc2,nb2,nt; } vqg_fj;

/* z=0/1: G/U = X[nt,DIM]·W{1,3}[MOEI,DIM]^T ; z=2: Y = H[nt,MOEI]·W2[DIM,MOEI]^T
 * 统一参数: rowsW=输出维(MOEI 或 DIM), K=归约维(DIM 或 MOEI)。 */
__global__ static void vqg_moef_kernel(const uint8_t *__restrict__ base, const vqg_fj *__restrict__ jobs,
                                       const float *__restrict__ Xin, float *__restrict__ Out,
                                       int ntmax, int rowsW, int K, int z, int mBlocks) {
    const vqg_fj j = jobs[blockIdx.y];
    const int tt = blockIdx.x / mBlocks, mb = blockIdx.x % mBlocks;
    const int t0 = tt * VQF_TT;
    if (t0 >= j.nt) return;
    const int ntc = (j.nt - t0) < VQF_TT ? (j.nt - t0) : VQF_TT;
    const int m = mb * blockDim.x + threadIdx.x;
    const uint64_t off = z == 0 ? j.o1 : z == 1 ? j.o3 : j.o2;
    const int nc = z == 2 ? j.nc2 : j.nc13, nbit = z == 2 ? j.nb2 : j.nb13;
    const uint8_t *pay = base + off;
    const uint8_t *cb_h = pay + 16;
    const uint8_t *gr_h = cb_h + (size_t)nc * 4 * 2;
    const uint8_t *ix = gr_h + (size_t)rowsW * 2;
    extern __shared__ float sh[];
    float *Cs = sh;                     /* 码本 nc*4 float ≤16KB */
    float *Xs = sh + (size_t)nc * 4;    /* X 瓦片 [VQF_TT][VQF_CH] */
    for (int i = threadIdx.x; i < nc * 4; i += blockDim.x) {
        uint16_t h = (uint16_t)cb_h[i * 2] | ((uint16_t)cb_h[i * 2 + 1] << 8);
        Cs[i] = __half2float(*(const __half *)&h);
    }
    float acc[VQF_TT];
#pragma unroll
    for (int t = 0; t < VQF_TT; t++) acc[t] = 0.0f;
    float g = 0.0f;
    if (m < rowsW) {
        uint16_t gh = (uint16_t)gr_h[m * 2] | ((uint16_t)gr_h[m * 2 + 1] << 8);
        g = __half2float(*(const __half *)&gh);
    }
    const int VQF_CH = 128;
    const size_t xbase = (size_t)blockIdx.y * ntmax * K;   /* 本专家的 X/H 切片 */
    const int nidx_row = K / 4;
    for (int k0 = 0; k0 < K; k0 += VQF_CH) {
        __syncthreads();
        for (int i = threadIdx.x; i < ntc * VQF_CH; i += blockDim.x) {
            const int t = i / VQF_CH, d = i % VQF_CH;
            Xs[i] = Xin[xbase + (size_t)(t0 + t) * K + k0 + d];
        }
        __syncthreads();
        if (m < rowsW) {
            const size_t i0 = (size_t)m * nidx_row + k0 / 4;
            for (int c = 0; c < VQF_CH / 4; c++) {
                uint32_t v;
                if (nbit == 8) v = ix[i0 + c];
                else {
                    const size_t bit = (i0 + c) * (size_t)nbit;
                    uint32_t w0 = ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8) | ((uint32_t)ix[(bit >> 3) + 2] << 16);
                    v = (w0 >> (bit & 7)) & ((1u << nbit) - 1u);
                }
                const float *ce = Cs + (size_t)v * 4;
#pragma unroll
                for (int d = 0; d < 4; d++) {
                    const float wv = g * ce[d];
                    const int xo = c * 4 + d;
#pragma unroll
                    for (int t = 0; t < VQF_TT; t++) acc[t] += wv * Xs[t * VQF_CH + xo];
                }
            }
        }
    }
    if (m < rowsW)
        for (int t = 0; t < ntc; t++)
            Out[((size_t)blockIdx.y * ntmax + t0 + t) * rowsW + m] = acc[t];
}

/* 宿主: 融合三 GEMM+swiglu。复用 g_moe_* 缓冲池与 g_moe_s 流。返回 1=走通, 0=回退旧路。 */
extern "C" int vqg_moe_batch_fused(const uint8_t *base, const vqg_fj *jobs_h, const float *Xpad,
                                   const float *Wt, float *Ypad, int nE, int ntmax,
                                   int DIM, int MOEI, float swlim)
{
    if (nE < 1 || ntmax < 1 || ntmax > 512) return 0;
    if (!g_moe_h) {
        if (cublasCreate(&g_moe_h) != CUBLAS_STATUS_SUCCESS) { g_moe_h = NULL; return 0; }
        cudaStreamCreateWithFlags(&g_moe_s, cudaStreamNonBlocking);
        cublasSetStream(g_moe_h, g_moe_s);
    }
    static vqg_fj *jd = NULL; static int jcap = 0;
    if (jcap < nE) { if (jd) cudaFree(jd); if (cudaMalloc((void **)&jd, (size_t)nE * sizeof(vqg_fj)) != cudaSuccess) { jd = NULL; jcap = 0; return 0; } jcap = nE; }
    const size_t nx = (size_t)nE * ntmax * DIM, nh = (size_t)nE * ntmax * MOEI;
    if (!moe_need(&g_moe_X, &g_moe_cap_x, nx)) return 0;
    if (!moe_need(&g_moe_Y, &g_moe_cap_y, nx)) return 0;
    if (!moe_need(&g_moe_G, &g_moe_cap_g, nh)) return 0;
    if (!moe_need(&g_moe_U, &g_moe_cap_u, nh)) return 0;
    if (cudaMemcpyAsync(jd, jobs_h, (size_t)nE * sizeof(vqg_fj), cudaMemcpyHostToDevice, g_moe_s) != cudaSuccess) return 0;
    if (cudaMemcpyAsync(g_moe_X, Xpad, nx * sizeof(float), cudaMemcpyHostToDevice, g_moe_s) != cudaSuccess) return 0;
    int nc13max = 0, nc2max = 0, ntM = 0;
    for (int e = 0; e < nE; e++) { if (jobs_h[e].nc13 > nc13max) nc13max = jobs_h[e].nc13;
        if (jobs_h[e].nc2 > nc2max) nc2max = jobs_h[e].nc2; if (jobs_h[e].nt > ntM) ntM = jobs_h[e].nt; }
    const int TPB = 128, CH = 128;
    const int tt = (ntM + VQF_TT - 1) / VQF_TT;
    {   /* G/U: rowsW=MOEI, K=DIM */
        const int mB = (MOEI + TPB - 1) / TPB;
        size_t shm = ((size_t)nc13max * 4 + (size_t)VQF_TT * CH) * sizeof(float);
        dim3 gr((unsigned)(mB * tt), (unsigned)nE);
        vqg_moef_kernel<<<gr, TPB, shm, g_moe_s>>>(base, jd, g_moe_X, g_moe_G, ntmax, MOEI, DIM, 0, mB);
        vqg_moef_kernel<<<gr, TPB, shm, g_moe_s>>>(base, jd, g_moe_X, g_moe_U, ntmax, MOEI, DIM, 1, mB);
    }
    {   /* swiglu+路由权重(复用原 kernel, 逐式同) */
        float *dWt = NULL;
        if (Wt) { if (!moe_need(&g_moe_W, &g_moe_cap_w, (size_t)nE * ntmax)) return 0;
            if (cudaMemcpyAsync(g_moe_W, Wt, (size_t)nE * ntmax * sizeof(float), cudaMemcpyHostToDevice, g_moe_s) != cudaSuccess) return 0;
            dWt = g_moe_W; }
        const size_t nblk = (nh + 255) / 256;
        moe_swiglu_kernel<<<(unsigned)nblk, 256, 0, g_moe_s>>>(g_moe_G, g_moe_U, dWt, ntmax, MOEI, nh, swlim);
    }
    {   /* Y: rowsW=DIM, K=MOEI, 输入=swiglu 后的 G */
        const int mB = (DIM + TPB - 1) / TPB;
        size_t shm = ((size_t)nc2max * 4 + (size_t)VQF_TT * CH) * sizeof(float);
        dim3 gr((unsigned)(mB * tt), (unsigned)nE);
        vqg_moef_kernel<<<gr, TPB, shm, g_moe_s>>>(base, jd, g_moe_G, g_moe_Y, ntmax, DIM, MOEI, 2, mB);
    }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { static int _d = 0;
        if (!_d++) fprintf(stderr, "[moe-fused] ★launch 失败 %s → 回退旧路★\n", cudaGetErrorString(e)); return 0; } }
    if (cudaMemcpyAsync(Ypad, g_moe_Y, nx * sizeof(float), cudaMemcpyDeviceToHost, g_moe_s) != cudaSuccess) return 0;
    if (cudaStreamSynchronize(g_moe_s) != cudaSuccess) { static int _d2 = 0;
        if (!_d2++) fprintf(stderr, "[moe-fused] ★sync 失败 → 回退旧路★\n"); return 0; }
    return 1;
}
