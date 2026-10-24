/* ================= FP 锚 attention 整层驻留 GPU (2026-08-22) =================
 * 起因: S=8192 时 dq_attention 吃掉 100% 的层时间(实测 56.6/57.4/61.0s 每层)。
 * 逐层排查:
 *   ① mask+softmax 是 O(S·N·NH)=43 亿次标量循环且单线程 → 已按行 pthread 并行(见 ds4quant_fwd.c)
 *   ② GEMM 确实走 cuBLAS(探针 cublas=0 cuda=0 -> GPU), 但只跑出 704 GFLOPS
 * 真因: 操作数全留在主机内存, 靠 GB10 的 ATS 让 GPU 直访。每头 SC[S,N]=268MB 写回主机,
 *      下一个 GEMM 再读回来 ⇒ 每层 ~40GB 走一致性链路, 实测等效 3.2 GB/s(页粒度访问, 非流式)。
 * 改法: 每层只把 q(1GB)/kva(17MB) 传一次进显存, SC 全程留在显存(mask+softmax 用核就地做),
 *      o 攒完一次性回主机。每层流量 40GB → 2GB。
 * 数值: softmax 与 CPU 版逐式一致(减 max、denom 从 exp(sink-max) 起、-inf 项置 0、double 累加),
 *      仅归约顺序不同 —— 与本管线固有的多线程求和抖动同量级(旧二进制自比 87% 元素差, p99 8.8e-5)。 */
#define VQG_ATT_TPB 256

__global__ static void vqg_attn_softmax_kernel(float *SC, float sinkh, int S, int N, int WIN, int ratio) {
    const int s = blockIdx.x;
    float *scr = SC + (size_t)s * N;
    __shared__ float shm[VQG_ATT_TPB];
    __shared__ double shd[VQG_ATT_TPB];
    const int tid = threadIdx.x;

    float m = -INFINITY;
    for (int n = tid; n < N; n += VQG_ATT_TPB) {
        int ok;
        if (n < S) ok = (n <= s) && (n > s - WIN);        /* sliding window causal */
        else       ok = ((n - S) < (s + 1) / ratio);       /* comp_ok */
        if (!ok) { scr[n] = -INFINITY; }
        else { float v = scr[n]; if (v > m) m = v; }
    }
    shm[tid] = m; __syncthreads();
    for (int st = VQG_ATT_TPB >> 1; st > 0; st >>= 1) {
        if (tid < st) shm[tid] = fmaxf(shm[tid], shm[tid + st]);
        __syncthreads();
    }
    const float mx = shm[0]; __syncthreads();

    double d = 0.0;
    for (int n = tid; n < N; n += VQG_ATT_TPB) {
        float v = scr[n];
        if (v == -INFINITY) { scr[n] = 0.0f; continue; }
        float e = expf(v - mx);
        scr[n] = e; d += (double)e;
    }
    shd[tid] = d; __syncthreads();
    for (int st = VQG_ATT_TPB >> 1; st > 0; st >>= 1) {
        if (tid < st) shd[tid] += shd[tid + st];
        __syncthreads();
    }
    const float inv = (float)(1.0 / (shd[0] + exp((double)sinkh - (double)mx)));
    __syncthreads();
    for (int n = tid; n < N; n += VQG_ATT_TPB) scr[n] *= inv;
}


/* ═══ 带状融合 attention(2026-08-28 速度)═══
 * 起因: 上面那条"两个 cuBLAS gemm + softmax 核"的路子, 对每行 s 都把 N=8192 列的分数
 * 算全, 可掩码只放行 WIN=128 列的滑窗(加上压缩段的 (s+1)/ratio 列)。实测每层 0.75s,
 * 其中 98% 的乘加算完立刻被 softmax 核置成 -inf/0 扔掉。
 * 改法: 一个 block 负责一个 (行 s, 头 h), 只遍历该行真正有效的列。
 *   有效列 = 滑窗 (s-WIN, s]  ∪  压缩段 [S, S+min(Sc,(s+1)/ratio))
 * 数值: 被掩的列在原路里也是先算完再置 0, 对 max / denom / 输出的贡献恒为 0 —— 跳过它
 * 不改变任何一项结果; softmax 逐式照抄上面的核(减 max、denom 从 exp(sink-max) 起、
 * double 累加、最后 scr*=inv 再乘 kva), 只有点积的归约顺序与 cuBLAS 不同。
 * 退路: 共享内存装不下(WIN+Sc 太大)或 HD 超过 4×TPB 时返回 0, 调用方走上面的老路。 */
#define VQG_BAT_TPB 256

__global__ static void vqg_attn_band_kernel(const float *__restrict__ q,
        const float *__restrict__ kva, const float *__restrict__ sink,
        float *__restrict__ o, int S, int N, int NH, int HD, int WIN, int ratio, float scale) {
    const int s = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int lo = (s - WIN + 1) > 0 ? (s - WIN + 1) : 0;
    const int nw = s - lo + 1;                       /* 滑窗有效列数 */
    const int Sc = N - S;
    int nc = 0;
    if (Sc > 0 && ratio > 0) { nc = (s + 1) / ratio; if (nc > Sc) nc = Sc; }
    const int tot = nw + nc;

    extern __shared__ float bsh[];
    float *shq = bsh;                 /* [HD]  本行本头的 q */
    float *shs = shq + HD;            /* [tot] 分数/概率 */
    float *shm = shs + (WIN + Sc);    /* [TPB] 归约暂存(float) */
    /* ★shd 必须 8 字节对齐(2026-08-30 misaligned-address 实锤修)★
     * 偏移=(HD+WIN+Sc+TPB)*4 字节; Sc=S/ratio+2 为【奇数】时(实撞: 复核前缀 S=1920→Sc=17)
     * 偏移≡4 (mod 8) → double 写错位 → misaligned address, 且该错误是 sticky(context 报废,
     * cudaGetLastError 清不掉) → 此后同进程所有 CUDA 调用永久失败(batched 永久翻 0,
     * sweep 掉纯 CPU 慢 9 倍的元凶)。S=8192(Sc=66)/512(Sc=6) 恰为偶数所以从未暴露。 */
    size_t _off = ((size_t)((shm + VQG_BAT_TPB) - bsh) * 4 + 7) & ~(size_t)7;
    double *shd = (double *)((char *)bsh + _off);   /* [TPB] 归约暂存(double, 8 对齐) */

    for (int d = tid; d < HD; d += VQG_BAT_TPB) shq[d] = q[((size_t)s * NH + h) * HD + d];
    __syncthreads();

    /* ① 分数: 一个 warp 干一列, lane 沿 d 连续读 kva ⇒ 合并访存 */
    const int wid = tid >> 5, lane = tid & 31, NWARP = VQG_BAT_TPB >> 5;
    for (int j = wid; j < tot; j += NWARP) {
        const int n = (j < nw) ? (lo + j) : (S + (j - nw));
        const float *kn = kva + (size_t)n * HD;
        float a = 0.0f;
        for (int d = lane; d < HD; d += 32) a += shq[d] * kn[d];
        for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) shs[j] = a * scale;
    }
    __syncthreads();

    /* ② softmax(与老核逐式一致) */
    float m = -INFINITY;
    for (int j = tid; j < tot; j += VQG_BAT_TPB) if (shs[j] > m) m = shs[j];
    shm[tid] = m; __syncthreads();
    for (int st = VQG_BAT_TPB >> 1; st > 0; st >>= 1) {
        if (tid < st) shm[tid] = fmaxf(shm[tid], shm[tid + st]);
        __syncthreads();
    }
    const float mx = shm[0]; __syncthreads();
    double dsum = 0.0;
    for (int j = tid; j < tot; j += VQG_BAT_TPB) { float e = expf(shs[j] - mx); shs[j] = e; dsum += (double)e; }
    shd[tid] = dsum; __syncthreads();
    for (int st = VQG_BAT_TPB >> 1; st > 0; st >>= 1) {
        if (tid < st) shd[tid] += shd[tid + st];
        __syncthreads();
    }
    const float inv = (float)(1.0 / (shd[0] + exp((double)sink[h] - (double)mx)));
    __syncthreads();
    for (int j = tid; j < tot; j += VQG_BAT_TPB) shs[j] *= inv;
    __syncthreads();

    /* ③ 输出: 每线程认领固定的几个 d, 沿 j 累加(kva 行内连续读 ⇒ 合并) */
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < tot; j++) {
        const int n = (j < nw) ? (lo + j) : (S + (j - nw));
        const float *kn = kva + (size_t)n * HD;
        const float p = shs[j];
#pragma unroll
        for (int r = 0; r < 4; r++) { int d = tid + r * VQG_BAT_TPB; if (d < HD) acc[r] += p * kn[d]; }
    }
#pragma unroll
    for (int r = 0; r < 4; r++) { int d = tid + r * VQG_BAT_TPB;
        if (d < HD) o[((size_t)s * NH + h) * HD + d] = acc[r]; }
}

static float *g_at_sink = NULL;
static float *g_at_q = NULL, *g_at_kva = NULL, *g_at_sc = NULL, *g_at_o = NULL;
static size_t g_at_nq = 0, g_at_nkva = 0, g_at_nsc = 0, g_at_no = 0;
static cublasHandle_t g_at_h = NULL;

static int vqg_at_need(float **p, size_t *have, size_t want) {
    if (*have >= want) return 1;
    if (*p) cudaFree(*p);
    *p = NULL; *have = 0;
    if (cudaMalloc((void **)p, want * sizeof(float)) != cudaSuccess) { *p = NULL; return 0; }
    *have = want; return 1;
}

/* 返回 1 = GPU 路已完成(o 已填); 0 = 调用方回落 CPU 路径。 */
extern "C" int vqg_attention(const float *q, const float *kva, const float *sink, float *o,
                             int S, int N, int NH, int HD, int WIN, int ratio, float scale) {
    if (!vqg_ready()) return 0;
    if (S <= 0 || N <= 0 || NH <= 0 || HD <= 0) return 0;
    if (!g_at_h && cublasCreate(&g_at_h) != CUBLAS_STATUS_SUCCESS) { g_at_h = NULL; return 0; }
    const size_t nq = (size_t)S * NH * HD, nkva = (size_t)N * HD, nsc = (size_t)S * N;
    if (!vqg_at_need(&g_at_q, &g_at_nq, nq) || !vqg_at_need(&g_at_kva, &g_at_nkva, nkva) ||
        !vqg_at_need(&g_at_sc, &g_at_nsc, nsc) || !vqg_at_need(&g_at_o, &g_at_no, nq)) return 0;
    struct timespec _t0,_t1,_t2,_t3; clock_gettime(CLOCK_MONOTONIC,&_t0);
    if (cudaMemcpy(g_at_q, q, nq * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(g_at_kva, kva, nkva * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    clock_gettime(CLOCK_MONOTONIC,&_t1);

    /* 带状路优先(见 vqg_attn_band_kernel 注释); 条件不满足静默走下面的两 gemm 老路 */
    {
        const int Sc = N - S;
        const size_t _fpart = (size_t)HD * 4 + (size_t)(WIN + (Sc > 0 ? Sc : 0)) * 4
                             + (size_t)VQG_BAT_TPB * 4;
        const size_t shbytes = ((_fpart + 7) & ~(size_t)7) + (size_t)VQG_BAT_TPB * 8;   /* double 段 8 对齐(与 kernel 内 _off 同式) */
        int shmax = 0, dev = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&shmax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (HD <= 4 * VQG_BAT_TPB && Sc >= 0 && shbytes <= (size_t)shmax && NH <= 65535) {
            static size_t g_at_nsink = 0;
            if (vqg_at_need(&g_at_sink, &g_at_nsink, (size_t)NH) &&
                cudaMemcpy(g_at_sink, sink, (size_t)NH * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess) {
                cudaFuncSetAttribute(vqg_attn_band_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes);
                dim3 gr(S, NH);
                vqg_attn_band_kernel<<<gr, VQG_BAT_TPB, shbytes>>>(
                    g_at_q, g_at_kva, g_at_sink, g_at_o, S, N, NH, HD, WIN, ratio, scale);
    { void vqg_last_err(const char*); vqg_last_err("vqg_attn_band_kernel后"); }
                if (cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess) {
                    clock_gettime(CLOCK_MONOTONIC, &_t2);
                    if (cudaMemcpy(o, g_at_o, nq * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
                    clock_gettime(CLOCK_MONOTONIC, &_t3);
                    /* 只报一次: 用来确认带状路真的走上了(静默回落两 gemm 老路 = 慢 5 倍
                     * 却没有任何症状), 不是逐层刷屏。 */
                    { static int said = 0; if (!said) { said = 1;
                        fprintf(stderr,"[gatt] 带状路已启用 S=%d N=%d NH=%d HD=%d WIN=%d\n",S,N,NH,HD,WIN); } }
                    return 1;
                }
                cudaGetLastError();   /* 清错, 回落老路 */
            }
        }
    }
    const float one = 1.0f, zero = 0.0f;
    for (int h = 0; h < NH; h++) {
        /* RowMajor SC[S,N] = q_h[S,HD](lda=NH*HD) · kva[N,HD](ldb=HD)^T * scale */
        if (cublasSgemm(g_at_h, CUBLAS_OP_T, CUBLAS_OP_N, N, S, HD,
                        &scale, g_at_kva, HD, g_at_q + (size_t)h * HD, NH * HD,
                        &zero, g_at_sc, N) != CUBLAS_STATUS_SUCCESS) return 0;
        vqg_attn_softmax_kernel<<<S, VQG_ATT_TPB>>>(g_at_sc, sink[h], S, N, WIN, ratio);
    { void vqg_last_err(const char*); vqg_last_err("vqg_attn_softmax_kernel后"); }
        /* RowMajor o_h[S,HD] = SC[S,N](lda=N) · kva[N,HD](ldb=HD), o 行距 NH*HD */
        if (cublasSgemm(g_at_h, CUBLAS_OP_N, CUBLAS_OP_N, HD, S, N,
                        &one, g_at_kva, HD, g_at_sc, N,
                        &zero, g_at_o + (size_t)h * HD, NH * HD) != CUBLAS_STATUS_SUCCESS) return 0;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return 0;
    clock_gettime(CLOCK_MONOTONIC,&_t2);
    if (cudaMemcpy(o, g_at_o, nq * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    clock_gettime(CLOCK_MONOTONIC,&_t3);
    { static int said2 = 0; if (!said2) { said2 = 1;
        fprintf(stderr,"[gatt] ★回落两-gemm 老路★ S=%d N=%d NH=%d HD=%d WIN=%d\n",S,N,NH,HD,WIN); } }
    return 1;
}
