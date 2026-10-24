/* engram_wkv_probe.cu — engram 量化误差过完 wkv 投影还剩多少(2026-09-10)。
 *
 * 【回答什么】engram_probe 测出单行 signref 1bit 的重建余弦 = 0.7983(= sqrt(2/π), 标量
 * 1bit 的率失真下界), 对应 60.3% 向量误差。但模型不直接用这些行 —— 官方 model.py 的
 * Engram.forward 是: 查 24 行(3 种 n-gram × 8 heads)拼成 6144 维 → wkv 投影成 4 份 key
 * + 1 份 value → gate=sigmoid(残差流·key 归一化点积) → h + gate·value。
 * 所以真正决定成败的是【投影输出】的误差, 不是单行误差。这一针直接量它。
 *
 * 【为什么不用跑模型】wkv 权重就 157 MB, 查表行按真实分段采样即可。gate 那一步需要真实
 * 残差流 h, 这里量不了 —— 但 key 的余弦就是 gate 稳不稳的上界, 先看它。
 *
 * 【采样怎么对上真实语义】engram.py: 每个 (n-gram 长度, head) 对占表里一段互不重叠的
 * 素数长区间, 24 段累加 = 384,006,168 行(= config 的 engram_num_embeddings, 实测对上)。
 * 所以一次查表 = 从 24 个段各取一行。均分近似段边界(素数都在 16,000,000 附近)。
 *
 * 【GPU】GEMM 走 cublas 生产路(铁律: 只写 GPU 版本)。量化是闭式 signref, 无迭代。
 *
 * 编译: nvcc -O3 -o engram_wkv_probe engram_wkv_probe.cu -lcublas
 * 用法: engram_wkv_probe <hf-dir> <层号 1|14> [组数]
 */
#include "st_locate.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"★CUDA %s @%d: %s★\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"★cuBLAS %s @%d: %d★\n",#x,__LINE__,(int)s_); exit(1);} } while(0)

static int cmpf(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}
/* 打分布并返回中位余弦。相对误差 = sqrt(1-cos²)/cos 的常见近似在 cos 不接近 1 时会失真,
 * 这里直接报 sqrt(1-cos²) —— 即最优缩放后残差占原向量的比例, 与 cos 一一对应不含近似。 */
static double report(const char *tag, float *v, long long n) {
    qsort(v, n, 4, cmpf);
    double m = 0; for (long long i = 0; i < n; i++) m += v[i];
    double med = v[n/2];
    printf("%-22s 最差 %.4f  p1 %.4f  p10 %.4f  p50 %.4f  p90 %.4f  均值 %.4f   (中位残差 %.1f%%)\n",
           tag, v[0], v[n/100], v[n/10], v[n/2], v[n*9/10], m/(double)n,
           100.0 * sqrt(1.0 - med*med));
    return med;
}

/* 逐 256 维段做标量量化。engram 的行实测就是高斯形状(σ=13.34/√256=0.834, E|w| 实测 0.678
 * = σ·√(2/π) 的 0.665, 差 2%), 所以用高斯 Lloyd-Max 的最优码本 × 每行 σ —— 闭式, 无迭代,
 * 且是【该位宽下标量量化的理论最优】, 保证测到的是率失真下界而不是"量化器没调好"。
 * 表来自标准 Lloyd-Max 解(对称, 只列正半轴): 1bit 的 0.7979 恰是 √(2/π), 与 mean|w| 同解。 */
static const float LM1[] = {0.7978846f};
static const float LM2[] = {0.4527800f, 1.5104300f};
static const float LM3[] = {0.2451700f, 0.7560000f, 1.3439000f, 2.1520000f};
static const float LM4[] = {0.1284000f, 0.3881000f, 0.6568000f, 0.9424000f,
                            1.2562000f, 1.6181000f, 2.0690000f, 2.7326000f};

static void quant_row(const float *src, float *dst, long long dim, int nbit) {
    const float *lm = nbit == 1 ? LM1 : nbit == 2 ? LM2 : nbit == 3 ? LM3 : LM4;
    int nl = 1 << (nbit - 1);                     /* 正半轴层数 */
    double s2 = 0;
    for (long long i = 0; i < dim; i++) s2 += (double)src[i] * src[i];
    float sig = (float)sqrt(s2 / (double)dim);
    if (sig <= 0) { for (long long i = 0; i < dim; i++) dst[i] = 0; return; }
    for (long long i = 0; i < dim; i++) {
        float a = fabsf(src[i]) / sig;
        /* 最近重建点(层数 ≤8, 线性扫比二分更快也更不容易写错) */
        int best = 0; float bd = fabsf(a - lm[0]);
        for (int k = 1; k < nl; k++) { float d = fabsf(a - lm[k]); if (d < bd) { bd = d; best = k; } }
        float q = lm[best] * sig;
        dst[i] = src[i] < 0 ? -q : q;
    }
}

/* 两个向量的余弦 */
static float cosine(const float *a, const float *b, long long n) {
    double d = 0, na = 0, nb = 0;
    for (long long i = 0; i < n; i++) { d += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return (na > 0 && nb > 0) ? (float)(d / (sqrt(na) * sqrt(nb))) : 0.0f;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: engram_wkv_probe <hf-dir> <层号 1|14> [组数]\n"); return 2; }
    const char *dir = argv[1];
    int LID = atoi(argv[2]);
    int NG = argc > 3 ? atoi(argv[3]) : 4096;      /* 组数 = 模拟多少个 token 位置的查表 */
    int NBIT = argc > 4 ? atoi(argv[4]) : 1;       /* 每元素位宽; 落地 bpw 还要 + 行增益 2B/256 = 0.0625 */
    if (NBIT < 1 || NBIT > 4) { fprintf(stderr, "★位宽只支持 1..4★\n"); return 2; }

    char n1[512], n2[512], n3[512], n4[512];
    snprintf(n1, sizeof n1, "layers.%d.engram.embed.weight", LID);
    snprintf(n2, sizeof n2, "layers.%d.engram.embed.scale",  LID);
    snprintf(n3, sizeof n3, "layers.%d.engram.wkv.weight",   LID);
    snprintf(n4, sizeof n4, "layers.%d.engram.wkv.scale",    LID);
    st_tref E, ES, W, WS;
    if (st_locate(dir, n1, &E) || st_locate(dir, n2, &ES) ||
        st_locate(dir, n3, &W) || st_locate(dir, n4, &WS)) { fprintf(stderr, "★张量缺★\n"); return 2; }

    const long long HD = E.shape[1];               /* head_dim = 256 */
    const long long NROW = E.shape[0];
    const long long OUT = W.shape[0], IN = W.shape[1];   /* 25600 × 6144 */
    const long long NCOL = IN / HD;                /* 24 = 3 n-gram × 8 heads */
    const long long DIM = 5120, HC = 4;            /* hc_mult=4 ⇒ key 4×5120, value 5120 */
    if (OUT != DIM * (HC + 1) || IN != NCOL * HD) {
        fprintf(stderr, "★形状不符预期: wkv %lldx%lld, 期望 %lldx%lld★\n", OUT, IN, DIM*(HC+1), NCOL*HD); return 2;
    }
    printf("=== engram wkv 投影探针: layer %d, %d bit/元素 (落地 %.4f bpw) ===\n",
           LID, NBIT, NBIT + 2*8.0/256);
    printf("表 %lld 行×%lld, 一次查表 %lld 行 → 拼 %lld 维 → wkv → key %lld×%lld + value %lld\n",
           NROW, HD, NCOL, IN, HC, DIM, DIM);

    /* ---- wkv 权重: 32×32 块 scale, 一次性 dequant ---- */
    long long sblk_r = OUT / WS.shape[0], sblk_c = IN / WS.shape[1];
    printf("wkv scale 块 = %lld×%lld\n", sblk_r, sblk_c);
    float *Wh = (float *)malloc((size_t)OUT * IN * 4);
    if (!Wh) { fprintf(stderr, "内存不足(wkv %.1f MB)\n", OUT*IN*4/1e6); return 1; }
    if (st_read_fp8_block(&W, &WS, 0, OUT, IN, sblk_r, sblk_c, Wh)) return 1;

    /* ---- 采样: 24 段, 每段 sqrt(NG) 个随机起点 × sqrt(NG) 连续行 ----
     * 纯随机单行会打成 NG×24 次 264 字节随机 I/O; 分段起点既避免"只看相邻 n-gram"的偏差,
     * 又把 I/O 变成上千次 16 KB 顺序读。 */
    int SQ = 1; while (SQ * SQ < NG) SQ++;
    long long NGa = (long long)SQ * SQ;            /* 实际组数(向上取到平方数) */
    long long seg = NROW / NCOL;
    printf("采样 %lld 组 × %lld 行 (每段 %d 起点 × %d 行, 段长 %lld)\n", NGa, NCOL, SQ, SQ, seg);

    float *Vo = (float *)malloc((size_t)NGa * IN * 4);   /* 原始拼接向量 */
    float *Vq = (float *)malloc((size_t)NGa * IN * 4);   /* signref 量化后 */
    float *rowbuf = (float *)malloc((size_t)SQ * HD * 4);
    if (!Vo || !Vq || !rowbuf) { fprintf(stderr, "内存不足\n"); return 1; }

    srand(1);
    for (long long c = 0; c < NCOL; c++) {
        long long base = c * seg;
        for (int b = 0; b < SQ; b++) {
            long long r0 = base + (long long)((double)rand() / ((double)RAND_MAX + 1) * (double)(seg - SQ));
            /* embed 每行自带 scale ⇒ 行块高 = 1, 列块 = HD/scale列数 */
            if (st_read_fp8_block(&E, &ES, r0, SQ, HD, 1, HD / ES.shape[1], rowbuf)) return 1;
            for (int k = 0; k < SQ; k++) {
                long long g = (long long)b * SQ + k;            /* 组号 */
                float *dst = Vo + g * IN + c * HD;
                memcpy(dst, rowbuf + (size_t)k * HD, (size_t)HD * 4);
                quant_row(dst, Vq + g * IN + c * HD, HD, NBIT);
            }
        }
    }

    /* 输入侧对照: 拼接后 6144 维的余弦(应 ≈ 单行的 0.798, 因为 24 段同分布) */
    float *cin = (float *)malloc((size_t)NGa * 4);
    for (long long g = 0; g < NGa; g++) cin[g] = cosine(Vo + g*IN, Vq + g*IN, IN);

    /* ---- GPU GEMM: out[NGa,OUT] = V[NGa,IN] @ Wh[OUT,IN]^T ---- */
    float *dW, *dV, *dO;
    CK(cudaMalloc(&dW, (size_t)OUT * IN * 4));
    CK(cudaMalloc(&dV, (size_t)NGa * IN * 4));
    CK(cudaMalloc(&dO, (size_t)NGa * OUT * 4));
    CK(cudaMemcpy(dW, Wh, (size_t)OUT * IN * 4, cudaMemcpyHostToDevice));
    cublasHandle_t h; CB(cublasCreate(&h));
    const float alpha = 1.0f, beta = 0.0f;
    float *Oo = (float *)malloc((size_t)NGa * OUT * 4);
    float *Oq = (float *)malloc((size_t)NGa * OUT * 4);
    if (!Oo || !Oq) { fprintf(stderr, "内存不足(输出 %.1f MB ×2)\n", NGa*OUT*4/1e6); return 1; }

    for (int pass = 0; pass < 2; pass++) {
        CK(cudaMemcpy(dV, pass ? Vq : Vo, (size_t)NGa * IN * 4, cudaMemcpyHostToDevice));
        /* 行主序 out = V @ W^T ⇔ 列主序 out^T[OUT,NGa] = W^T_op × V^T; W 行主序[OUT,IN]
         * 在列主序里是 [IN,OUT], 故取 OP_T 还原成 [OUT,IN]。 */
        CB(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                       (int)OUT, (int)NGa, (int)IN,
                       &alpha, dW, (int)IN, dV, (int)IN, &beta, dO, (int)OUT));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(pass ? Oq : Oo, dO, (size_t)NGa * OUT * 4, cudaMemcpyDeviceToHost));
    }

    /* ---- 统计 ---- */
    float *cval = (float *)malloc((size_t)NGa * 4);
    float *ckey = (float *)malloc((size_t)NGa * HC * 4);
    for (long long g = 0; g < NGa; g++) {
        /* model.py: key, value = kv.split([hc_mult*dim, dim]) ⇒ key 在前, value 在后 */
        for (int j = 0; j < HC; j++)
            ckey[g*HC + j] = cosine(Oo + g*OUT + (long long)j*DIM, Oq + g*OUT + (long long)j*DIM, DIM);
        cval[g] = cosine(Oo + g*OUT + HC*DIM, Oq + g*OUT + HC*DIM, DIM);
    }
    printf("\n");
    double cI = report("输入 6144 维拼接", cin, NGa);
    double cV = report("★value(写入残差)", cval, NGa);
    double cK = report("★key(决定 gate)", ckey, NGa * HC);
    /* 投影是否放大/衰减误差: >1 表示 wkv 把量化噪声压掉了一部分(投影不是各向同性时会发生) */
    printf("\n投影前后: 输入残差 %.1f%% → value 残差 %.1f%%(×%.2f) / key 残差 %.1f%%(×%.2f)\n",
           100*sqrt(1-cI*cI), 100*sqrt(1-cV*cV), sqrt(1-cV*cV)/sqrt(1-cI*cI),
           100*sqrt(1-cK*cK), sqrt(1-cK*cK)/sqrt(1-cI*cI));
    free(Wh); free(Vo); free(Vq); free(rowbuf); free(cin); free(cval); free(ckey);
    free(Oo); free(Oq);
    cudaFree(dW); cudaFree(dV); cudaFree(dO); cublasDestroy(h);
    close(E.fd); close(ES.fd); close(W.fd); close(WS.fd);
    return 0;
}
