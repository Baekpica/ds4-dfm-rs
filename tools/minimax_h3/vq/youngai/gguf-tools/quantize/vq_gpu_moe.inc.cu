/* vq_gpu_moe.inc.cu — 批量专家前向(GPU)。物理分片, 只被 vq_gpu.cu include。
 *
 * ★为什么要它(2026-08-29 用户令"先把 sweep 改成 gpu, 速度太慢了")★
 * 现状: bytes_moe 是 20 个 CPU 线程各自跑【单专家】GEMM。sweep 抽格到 ~512 行后, 每个专家
 * 只分到 nt≈12 个 token ⇒ 单次 GEMM 12×4096×2048×2 ≈ 2.0e8 FLOP, 正好卡在 dq_matmul 的
 * GPU 门槛(>=2e8)上, 分到 token 少的专家全部掉回 CPU。实测一层前向 354.7s CPU 累计 /
 * 20 线程 = 17.7s 墙钟, GPU 完全没吃到活; sweep 每层试 6 种形态 ⇒ ~105s/层, 43 层 1.3 小时。
 *
 * ★为什么不是"降门槛"★ dq_matmul 的注释里已有实测: 阈降到 5e6 是【负收益】(259s vs 181s/层)
 * —— 20 线程高频提交小 GEMM, launch+sync 队列争用吃掉全部收益。同一份注释给了正解:
 * "高频小矩阵的正确姿势是【批量结构改造】"。这个文件就是那个改造。
 *
 * 做法: 专家权重批 dequant 后本就躺在 managed 的 g_bmw_buf 里, 排布是 [nE][3][DIM*MOEI]
 * 等步长连续 —— 正好是 cublasSgemmStridedBatched 要的形状。把 nE 个专家的 token 补齐到
 * ntmax 后一次批量提交: 3 次 batched GEMM + 1 个 SwiGLU kernel, 取代 nE×3 次小 GEMM。
 * 补齐浪费 ntmax/ntavg(sweep 场景约 3×), 相对 GPU 与 CPU 的差距可以忽略。
 *
 * 数值: fp32 全程, 与 CPU 路同精度同式(clip→silu→乘)。任何一步失败返回 0, 调用方落回
 * 原 CPU 路径 —— 那条路逐字节不变, 不是"兜底近似"。 */

#include <cublas_v2.h>
extern "C" void vqg_last_err(const char *tag);   /* 文件作用域声明(函数体内放 extern"C" 是编译错) */

static cublasHandle_t g_moe_h = NULL;
static cudaStream_t   g_moe_s = NULL;
/* 补齐缓冲(按需增长, 进程内复用): Xg/Hg/Ug/Yg */
static float *g_moe_X = NULL, *g_moe_G = NULL, *g_moe_U = NULL, *g_moe_Y = NULL, *g_moe_W = NULL;
static size_t g_moe_cap_x = 0, g_moe_cap_g = 0, g_moe_cap_u = 0, g_moe_cap_y = 0, g_moe_cap_w = 0;

static int moe_need(float **p, size_t *cap, size_t n) {
    if (*cap >= n) return 1;
    if (*p) cudaFree(*p);
    *p = NULL; *cap = 0;
    if (cudaMalloc((void **)p, n * sizeof(float)) != cudaSuccess) { *p = NULL; return 0; }
    *cap = n; return 1;
}

/* SwiGLU + 路由权重, 与 dq_expert_fp 逐式对齐(fwd_p1:286-296):
 *     if (swlim>0) { uu 夹到 [-lim,lim];  ★gg 只夹【上界】, 下界不夹★ }
 *     h = silu(gg) * uu ;  if (weight) h[s,:] *= weight[s]
 * ★gg 单边夹是原式如此★ —— 我第一版写成上下都夹, 那是数值错误, 已改。
 * 路由权重必须在【第三个 GEMM 之前】乘进 h(原式同), 不能挪到 scatter 后。
 * 补齐行(i>=nt[e])算出来是垃圾, 但 scatter 只取前 nt 行, 不污染结果。
 * Wt[nE][ntmax] = 每 (专家,槽) 的 ge[e]*rw; 补齐槽填 0 即可。 */
__global__ static void moe_swiglu_kernel(float *G, const float *U, const float *Wt,
                                         int ntmax, int MOEI, size_t n, float lim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = G[i], u = U[i];
    if (lim > 0.0f) { u = fminf(fmaxf(u, -lim), lim); if (g > lim) g = lim; }
    /* ★逐式对齐 dq_silu(fwd_p1:48) = z/(1+expf(-z))★: 原式【没有 ±60 夹持】, 也用的是
     * expf 而非快速内建 __expf。我第一版两样都加了 —— 那是自作主张的偏差, 已去掉。
     * 极负 g 时 expf(-g) 溢出成 inf, z/inf=0, 与 CPU 路同行为。 */
    float h = (g / (1.0f + expf(-g))) * u;
    if (Wt) h *= Wt[i / (size_t)MOEI];          /* 行号 = i/MOEI, 直接索引 [nE*ntmax] */
    G[i] = h;
}

/* 一块 nE 个专家的批量前向。
 *   Wbuf   managed [nE][3][DIM*MOEI]  (w1,w3,w2 顺序与 bytes_moe 的 sl0+0/1/2 一致)
 *   Xpad   host    [nE][ntmax][DIM]   调用方已按专家补齐好的输入(补齐行内容任意)
 *   Ypad   host    [nE][ntmax][DIM]   输出
 * 返回 1=GPU 走通, 0=失败(调用方落回 CPU 路)。 */
extern "C" int vqg_moe_batch(const float *Wbuf, const float *Xpad, const float *Wt, float *Ypad,
                             int nE, int ntmax, int DIM, int MOEI, float swlim)
{
    /* ★失败必须说清在哪一步(2026-08-29)★: 第一版全是裸 return 0, 一失败就静默落回
     * 单线程 CPU(571s/层), 完全查不出是分配失败、拷贝失败还是 GEMM 参数错。 */
    #define MOE_FAIL(tag, code) do{ static int _d=0; if(!_d++) \
        fprintf(stderr,"[moe-gpu] ★%s 失败 code=%d★ nE=%d ntmax=%d DIM=%d MOEI=%d\n", \
                (tag),(int)(code),nE,ntmax,DIM,MOEI); return 0; }while(0)
    if (nE < 1 || ntmax < 1) return 0;
    if (!g_moe_h) {
        cublasStatus_t cs = cublasCreate(&g_moe_h);
        if (cs != CUBLAS_STATUS_SUCCESS) { g_moe_h = NULL; MOE_FAIL("cublasCreate", cs); }
        cudaStreamCreateWithFlags(&g_moe_s, cudaStreamNonBlocking);
        cublasSetStream(g_moe_h, g_moe_s);
    }
    const size_t nx = (size_t)nE * ntmax * DIM, nh = (size_t)nE * ntmax * MOEI;
    /* ★两个缓冲共用一个 cap 变量是 bug(第一版)★: moe_need(G,&cap_h,nh) 先把 cap_h 设成 nh,
     * 紧接着 moe_need(U,&cap_h,nh) 看见 cap_h>=nh 就【直接返回成功】—— U 根本没分配, 保持
     * NULL, 后面 GEMM 拿 NULL 指针必然失败。各给各的 cap。 */
    if (!moe_need(&g_moe_X, &g_moe_cap_x, nx)) MOE_FAIL("alloc X", 0);
    if (!moe_need(&g_moe_Y, &g_moe_cap_y, nx)) MOE_FAIL("alloc Y", 0);
    if (!moe_need(&g_moe_G, &g_moe_cap_g, nh)) MOE_FAIL("alloc G", 0);
    if (!moe_need(&g_moe_U, &g_moe_cap_u, nh)) MOE_FAIL("alloc U", 0);
    { cudaError_t e = cudaMemcpyAsync(g_moe_X, Xpad, nx * sizeof(float), cudaMemcpyHostToDevice, g_moe_s);
      if (e != cudaSuccess) MOE_FAIL("H2D X", e); }

    const float one = 1.0f, zero = 0.0f;
    const long long sW = (long long)3 * DIM * MOEI;          /* 专家间权重步长 */
    const long long sX = (long long)ntmax * DIM, sH = (long long)ntmax * MOEI;
    /* 行主序 C[n,m]=A[n,k]·B[m,k]^T ⇒ cuBLAS 列主序等价 (OP_T, OP_N, m, n, k, B, A, C) */
    #define MOE_GEMM(Bp, Ap, Cp, m, n, k, sb, sa, sc) \
        cublasSgemmStridedBatched(g_moe_h, CUBLAS_OP_T, CUBLAS_OP_N, (m), (n), (k), \
            &one, (Bp), (k), (sb), (Ap), (k), (sa), &zero, (Cp), (m), (sc), nE)
    { cublasStatus_t st = MOE_GEMM(Wbuf,                     g_moe_X, g_moe_G, MOEI, ntmax, DIM, sW, sX, sH); if (st != CUBLAS_STATUS_SUCCESS) MOE_FAIL("gemm", st); }
    { cublasStatus_t st = MOE_GEMM(Wbuf + (size_t)DIM*MOEI,  g_moe_X, g_moe_U, MOEI, ntmax, DIM, sW, sX, sH); if (st != CUBLAS_STATUS_SUCCESS) MOE_FAIL("gemm", st); }
    {   /* 路由权重上设备(小: nE*ntmax 个 float) */
        float *dWt = NULL;
        if (Wt) { if (!moe_need(&g_moe_W, &g_moe_cap_w, (size_t)nE*ntmax)) MOE_FAIL("alloc Wt", 0);
                  if (cudaMemcpyAsync(g_moe_W, Wt, (size_t)nE*ntmax*sizeof(float),
                                      cudaMemcpyHostToDevice, g_moe_s) != cudaSuccess) MOE_FAIL("H2D Wt", 0);
                  dWt = g_moe_W; }
        const int T = 256; const size_t nblk = (nh + T - 1) / T;
        moe_swiglu_kernel<<<(unsigned)nblk, T, 0, g_moe_s>>>(g_moe_G, g_moe_U, dWt, ntmax, MOEI, nh, swlim);
        vqg_last_err("moe_swiglu后"); }
    { cublasStatus_t st = MOE_GEMM(Wbuf + (size_t)2*DIM*MOEI, g_moe_G, g_moe_Y, DIM, ntmax, MOEI, sW, sH, sX); if (st != CUBLAS_STATUS_SUCCESS) MOE_FAIL("gemm", st); }
    #undef MOE_GEMM
    { cudaError_t e = cudaMemcpyAsync(Ypad, g_moe_Y, nx * sizeof(float), cudaMemcpyDeviceToHost, g_moe_s);
      if (e != cudaSuccess) MOE_FAIL("D2H Y", e); }
    { cudaError_t e = cudaStreamSynchronize(g_moe_s); if (e != cudaSuccess) MOE_FAIL("sync", e); }
    return 1;
}

extern "C" void vqg_moe_release(void) {
    if (g_moe_X) { cudaFree(g_moe_X); g_moe_X = NULL; g_moe_cap_x = 0; }
    if (g_moe_Y) { cudaFree(g_moe_Y); g_moe_Y = NULL; g_moe_cap_y = 0; }
    if (g_moe_G) { cudaFree(g_moe_G); g_moe_G = NULL; g_moe_cap_g = 0; }
    if (g_moe_U) { cudaFree(g_moe_U); g_moe_U = NULL; g_moe_cap_u = 0; }
    if (g_moe_W) { cudaFree(g_moe_W); g_moe_W = NULL; g_moe_cap_w = 0; }
    if (g_moe_h) { cublasDestroy(g_moe_h); g_moe_h = NULL; }
    if (g_moe_s) { cudaStreamDestroy(g_moe_s); g_moe_s = NULL; }
}
