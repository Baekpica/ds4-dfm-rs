/* v41_codebook.c — 按真实权重分布求最优标量码本(2026-09-11)。
 *
 * 【为什么要它】首版量化用高斯 Lloyd-Max 码本, 撞了一个隐蔽的坑: V4.1 的专家出厂就是
 * FP4 —— 16 个离散值 × 每 32 列一个 ue8m0 scale。拿高斯最优点去套这种离散源, 即使位宽
 * 相同也重现不了原值, 误差凭空产生(实测 4.25 bpw 竟有 KLD 0.41)。
 *
 * 【解法, ★不是训练也不是迭代★】量化块(32 元素)与 FP4 的 scale 块(32 列)对齐 ⇒ 块内
 * 所有元素共用同一个出厂 scale ⇒ 归一化值 w/σ 只取决于块内那 16 个 FP4 值的构成, 是个
 * 离散分布。统计它的直方图, 然后用**动态规划**求"k 个重建点使加权 MSE 最小"的精确解:
 *     opt[i][j] = min_{m<i} opt[m][j-1] + cost(m+1..i)
 * 一维最优量化的标准 DP, O(bins²·k), 毫秒级, 全局最优 —— 不是 k-means 那种局部收敛,
 * 也不涉及任何梯度。
 *
 * 【为什么不强制对称】首版是"符号位 + 2^(nbit-1) 个幅度", 表示不了 0。而实测 FP4 里
 * ±0 两个槽合占 **11.65%** —— 强制对称等于把这 11.65% 全推到 ±c 上。改成通用 2^nbit 点,
 * DP 自己决定要不要留 0, 体积一个字节不变(还是 nbit 位索引)。
 *
 * 编译: cc -O3 -std=c99 -o v41_codebook v41_codebook.c -lm
 * 用法: v41_codebook <hf-dir> <层号> <nbit> <出码本.bin> [专家数=16 每专家行数=256]
 */
#include "st_locate.h"

#define NB 2048                 /* 直方图 bin 数 */
#define RNG 4.0                 /* 归一化值范围 [-RNG, RNG]; FP4/RMS 实际约 ±2.6, 留余量 */

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: v41_codebook <hf-dir> <层号> <nbit> <出码本.bin> [专家数 每专家行数]\n");
        return 2;
    }
    const char *dir = argv[1];
    int LID = atoi(argv[2]), NBIT = atoi(argv[3]);
    const char *out = argv[4];
    int NE = argc > 5 ? atoi(argv[5]) : 16;
    int NR = argc > 6 ? atoi(argv[6]) : 256;
    if (NBIT < 1 || NBIT > 8) { fprintf(stderr, "★nbit 只支持 1..8★\n"); return 2; }
    const long long BLK = 32;

    /* ---- 统计归一化值的带权直方图 ---- */
    static double hist[NB];
    long long ntot = 0;
    for (int e = 0; e < NE; e++) {
        for (int wi = 1; wi <= 3; wi++) {
            char nw[512], ns[512];
            snprintf(nw, sizeof nw, "layers.%d.ffn.experts.%d.w%d.weight", LID, e, wi);
            snprintf(ns, sizeof ns, "layers.%d.ffn.experts.%d.w%d.scale",  LID, e, wi);
            st_tref W, S;
            if (st_locate(dir, nw, &W) || st_locate(dir, ns, &S)) { fprintf(stderr, "★缺 %s★\n", nw); return 2; }
            long long rows = W.shape[0], cols = W.shape[1] * 2;
            int nr = (int)(NR < rows ? NR : rows);
            long long r0 = (rows - nr) / 2;
            float *w = (float *)malloc((size_t)nr * cols * 4);
            if (!w) { fprintf(stderr, "内存不足\n"); return 1; }
            if (st_read_fp4_block(&W, &S, r0, nr, cols, BLK, w)) return 1;
            for (long long r = 0; r < nr; r++) {
                for (long long c = 0; c < cols; c += BLK) {
                    const float *p = w + r * cols + c;
                    double s2 = 0;
                    for (int i = 0; i < BLK; i++) s2 += (double)p[i] * p[i];
                    double sig = sqrt(s2 / BLK);
                    if (sig <= 0) continue;
                    /* ★用与 kernel 完全相同的 scale 口径★: 最近的 2 的幂(ue8m0)。
                     * 码本按 σ 统计而 kernel 按 2^k 量化 = 两套口径, 码本立刻失配。 */
                    int ex; frexp(sig, &ex);
                    double q = ldexp(1.0, (int)lrint(log2(sig)));
                    (void)ex;
                    for (int i = 0; i < BLK; i++) {
                        double v = p[i] / q;
                        int b = (int)((v + RNG) / (2 * RNG) * NB);
                        if (b < 0) b = 0;
                        if (b >= NB) b = NB - 1;
                        hist[b] += 1.0;
                        ntot++;
                    }
                }
            }
            free(w); close(W.fd); close(S.fd);
        }
    }
    fprintf(stderr, "[统计] %lld 个归一化值 (层 %d, %d 专家 × %d 行 × 3 矩阵)\n", ntot, LID, NE, NR);

    /* ---- 一维最优量化 DP ---- */
    /* 前缀和: 个数 / Σx / Σx²  ⇒ 任意区间的最优重建点(加权均值)与代价 O(1) */
    static double c0[NB + 1], c1[NB + 1], c2[NB + 1];
    for (int i = 0; i < NB; i++) {
        double x = -RNG + (i + 0.5) * (2 * RNG) / NB;
        c0[i + 1] = c0[i] + hist[i];
        c1[i + 1] = c1[i] + hist[i] * x;
        c2[i + 1] = c2[i] + hist[i] * x * x;
    }
    int K = 1 << NBIT;
    static double opt[NB + 1][257];
    static int arg[NB + 1][257];
    for (int i = 0; i <= NB; i++) for (int j = 0; j <= K; j++) opt[i][j] = (i == 0 && j == 0) ? 0 : 1e300;
    for (int j = 1; j <= K; j++) {
        for (int i = 1; i <= NB; i++) {
            double best = 1e300; int bm = 0;
            for (int m = j - 1; m < i; m++) {
                if (opt[m][j - 1] >= 1e299) continue;
                double n = c0[i] - c0[m], s = c1[i] - c1[m], q = c2[i] - c2[m];
                /* 区间代价 = Σw(x−μ)² = Σwx² − (Σwx)²/Σw */
                double cost = n > 0 ? q - s * s / n : 0;
                double v = opt[m][j - 1] + cost;
                if (v < best) { best = v; bm = m; }
            }
            opt[i][j] = best; arg[i][j] = bm;
        }
    }
    /* 回溯出 K 个重建点 */
    float cb[256];
    int i = NB, j = K;
    while (j > 0) {
        int m = arg[i][j];
        double n = c0[i] - c0[m], s = c1[i] - c1[m];
        cb[j - 1] = (float)(n > 0 ? s / n : 0.0);
        i = m; j--;
    }
    double mse = opt[NB][K] / (ntot > 0 ? ntot : 1);
    /* cos ≈ sqrt(1 − MSE/E[x²]); E[x²] 由直方图给 */
    double ex2 = c2[NB] / (ntot > 0 ? ntot : 1);
    fprintf(stderr, "[DP] %d 个重建点, 加权 MSE %.6g, E[x²] %.6g ⇒ 重建 cos ≈ %.4f\n",
            K, mse, ex2, sqrt(1.0 - mse / ex2));
    fprintf(stderr, "[码本]");
    for (int k = 0; k < K; k++) fprintf(stderr, " %.4f", cb[k]);
    fprintf(stderr, "\n");

    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "★写不了 %s★\n", out); return 1; }
    int32_t kk = K;
    fwrite(&kk, 4, 1, f);
    fwrite(cb, 4, K, f);
    fclose(f);
    fprintf(stderr, "→ %s\n", out);
    return 0;
}
