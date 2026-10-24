/* expert_probe.c — V4.1 主干专家再压到 ~1.3 bpw 的可行性探针(2026-09-10)。
 *
 * 【回答什么】100 GB 方案里主干专家(543.582 B 参数, 40 层×384)只能拿 1.2963 bpw。而它的
 * 出厂态【已经是 FP4】(每 32 列一个 ue8m0 scale, 落地 4.25 bpw) —— 这是从"已经量化过一轮"
 * 的源再压 3.3 倍, 跟 V4 时代从 BF16 压到 2.25 bpw 完全不是一回事, V4 的档位表一个数都借不了。
 *
 * 【为什么先测熵】源只有 16 个离散值。如果 nibble 的经验熵已经远低于 4 bit, 说明 FP4 本身
 * 有冗余, 无损/近无损就能省一截; 如果接近满 4 bit, 那到 1.3 bpw 的每一位都得从失真里买。
 * 不先看这个就选方案 = 猜。
 *
 * 【位宽口径】跟 FP4 同块结构(每行每 32 列一个 ue8m0 scale, 0.25 bpw 开销):
 *   1 bit + scale = 1.25 bpw   2 bit + scale = 2.25   3 bit + scale = 3.25   (FP4 原态 = 4.25)
 * 目标 1.2963 落在 1.25 档上 —— 所以 1 bit 那一行就是判决行。
 *
 * 【局限, 必须写在脸上】这里只量【权重重建余弦】, 不是端到端质量。本仓有前科: 层内指标与
 * 端到端有 40× 错位税(fable5 放大器战役)。所以这一针的用法是【排除】而不是【判准】——
 * 权重余弦若在 1.25 bpw 崩到 0.6 级, 端到端不可能好; 若还不错, 也仍要等真尺。
 *
 * 编译: cc -O3 -std=c99 -o expert_probe expert_probe.c -lm
 * 用法: expert_probe <hf-dir> <层号> [专家数 每专家采样行数]
 */
#include "st_locate.h"

/* 高斯 Lloyd-Max 最优码本(正半轴), 与 engram_wkv_probe 同一套 —— 两针可横比。 */
static const float LM1[] = {0.7978846f};
static const float LM2[] = {0.4527800f, 1.5104300f};
static const float LM3[] = {0.2451700f, 0.7560000f, 1.3439000f, 2.1520000f};

/* 对一个 32 元素块做 nbit 标量量化(块内一个 σ), 返回重建值写回 dst。 */
static void quant_blk(const float *src, float *dst, int n, int nbit) {
    const float *lm = nbit == 1 ? LM1 : nbit == 2 ? LM2 : LM3;
    int nl = 1 << (nbit - 1);
    double s2 = 0;
    for (int i = 0; i < n; i++) s2 += (double)src[i] * src[i];
    float sig = (float)sqrt(s2 / n);
    if (sig <= 0) { for (int i = 0; i < n; i++) dst[i] = 0; return; }
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]) / sig;
        int best = 0; float bd = fabsf(a - lm[0]);
        for (int k = 1; k < nl; k++) { float d = fabsf(a - lm[k]); if (d < bd) { bd = d; best = k; } }
        dst[i] = (src[i] < 0 ? -1.0f : 1.0f) * lm[best] * sig;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: expert_probe <hf-dir> <层号> [专家数 每专家行数]\n"); return 2; }
    const char *dir = argv[1];
    int LID = atoi(argv[2]);
    int NE = argc > 3 ? atoi(argv[3]) : 8;         /* 采样几个专家 */
    int NR = argc > 4 ? atoi(argv[4]) : 256;       /* 每专家采样多少行 */
    const long long BLK = 32;                       /* FP4 的 scale 块宽, 量化块与之对齐 */

    printf("=== 专家探针: layer %d, %d 个专家 × %d 行 ===\n", LID, NE, NR);
    /* nibble 直方图: 16 个槽 */
    long long hist[16] = {0}, ntot = 0;
    /* 各位宽的余弦累计: [1..3] bit */
    double csum[4] = {0}; long long ccnt = 0;
    float cmin[4] = {2, 2, 2, 2};

    for (int e = 0; e < NE; e++) {
        for (int wi = 1; wi <= 3; wi++) {
            char nw[512], ns[512];
            snprintf(nw, sizeof nw, "layers.%d.ffn.experts.%d.w%d.weight", LID, e, wi);
            snprintf(ns, sizeof ns, "layers.%d.ffn.experts.%d.w%d.scale",  LID, e, wi);
            st_tref W, S;
            if (st_locate(dir, nw, &W) || st_locate(dir, ns, &S)) { fprintf(stderr, "★缺 %s★\n", nw); return 2; }
            if (strcmp(W.dtype, "I8") || strcmp(S.dtype, "F8_E8M0")) {
                fprintf(stderr, "★dtype 不是 I8/E8M0 (%s/%s), 解码会错★\n", W.dtype, S.dtype); return 2;
            }
            long long rows = W.shape[0], cols = W.shape[1] * 2;   /* I8 打包 ⇒ 逻辑列数 ×2 */
            long long sblk_c = cols / S.shape[1];
            if (S.shape[0] != rows || sblk_c != BLK) {
                fprintf(stderr, "★scale 形状意外: [%lld,%lld] vs 权重 %lldx%lld(块宽 %lld)★\n",
                        S.shape[0], S.shape[1], rows, cols, sblk_c); return 2;
            }
            int nr = (int)(NR < rows ? NR : rows);
            /* 采样连续 nr 行(专家矩阵行之间无序, 连续块即可代表) */
            long long r0 = (rows - nr) / 2;
            float *w = (float *)malloc((size_t)nr * cols * 4);
            float *q = (float *)malloc((size_t)nr * cols * 4);
            if (!w || !q) { fprintf(stderr, "内存不足\n"); return 1; }
            if (st_read_fp4_block(&W, &S, r0, nr, cols, BLK, w)) return 1;

            /* nibble 直方图: 直接重读打包字节, 免得从浮点反推 */
            {
                long long packed = cols / 2;
                uint8_t *raw = (uint8_t *)malloc((size_t)nr * packed);
                if (pread(W.fd, raw, (size_t)nr * packed, W.off + r0 * packed) == (ssize_t)((size_t)nr * packed)) {
                    for (long long i = 0; i < (long long)nr * packed; i++) {
                        hist[raw[i] & 0x0f]++; hist[(raw[i] >> 4) & 0x0f]++; ntot += 2;
                    }
                }
                free(raw);
            }

            for (int nb = 1; nb <= 3; nb++) {
                for (long long r = 0; r < nr; r++)
                    for (long long c = 0; c < cols; c += BLK)
                        quant_blk(w + r*cols + c, q + r*cols + c, (int)BLK, nb);
                /* 逐行余弦(行 = 一个输出通道, 与 FFN 的 GEMV 语义对齐) */
                for (long long r = 0; r < nr; r++) {
                    double d = 0, na = 0, nbn = 0;
                    for (long long c = 0; c < cols; c++) {
                        d += (double)w[r*cols+c] * q[r*cols+c];
                        na += (double)w[r*cols+c] * w[r*cols+c];
                        nbn += (double)q[r*cols+c] * q[r*cols+c];
                    }
                    float cs = (na > 0 && nbn > 0) ? (float)(d / (sqrt(na)*sqrt(nbn))) : 0.0f;
                    csum[nb] += cs;
                    if (cs < cmin[nb]) cmin[nb] = cs;
                    if (nb == 1) ccnt++;
                }
            }
            free(w); free(q);
            close(W.fd); close(S.fd);
        }
    }

    /* ---- 行内 scale 跨度: 决定 VQ 的"每行一个增益"够不够 ----
     * V4.1 的出厂 scale 是每行每 32 列一个(一行 160 个)。VQ 的行增益只有每行一个,
     * 粗 160 倍。若行内 scale 跨好几个量级, 单增益覆盖不住, 码本要被迫表示跨量级的值 ——
     * 那 nc 再大也是浪费。这一格必须在选格式之前量。 */
    {
        char nw[512], ns[512];
        snprintf(ns, sizeof ns, "layers.%d.ffn.experts.0.w1.scale", LID);
        snprintf(nw, sizeof nw, "layers.%d.ffn.experts.0.w1.weight", LID);
        st_tref S, W;
        if (!st_locate(dir, ns, &S) && !st_locate(dir, nw, &W)) {
            long long rows = S.shape[0], nsc = S.shape[1];
            int nr = (int)(NR < rows ? NR : rows);
            uint8_t *sb = (uint8_t *)malloc((size_t)nr * nsc);
            if (sb && pread(S.fd, sb, (size_t)nr * nsc, S.off) == (ssize_t)((size_t)nr * nsc)) {
                double sp_sum = 0; int sp_max = 0; long long nrow = 0;
                for (int r = 0; r < nr; r++) {
                    int lo = 255, hi = 0;
                    for (long long c = 0; c < nsc; c++) {
                        int e = sb[(size_t)r * nsc + c];
                        if (e == 255) continue;          /* NaN 槽 */
                        if (e < lo) lo = e;
                        if (e > hi) hi = e;
                    }
                    if (hi >= lo) { int sp = hi - lo; sp_sum += sp; if (sp > sp_max) sp_max = sp; nrow++; }
                }
                printf("\n[行内 scale 跨度] %lld 行 × %lld 个 scale/行: 平均 %.1f 个 2 的幂"
                       "(=%.0f×), 最大 %d 个(=%.3g×)\n",
                       nrow, nsc, sp_sum / nrow, pow(2, sp_sum / nrow), sp_max, pow(2, sp_max));
                printf("  ★VQ 的每行一个增益要覆盖这个跨度; 跨度大 = 行增益不够, 得保留细粒度 scale★\n");
            }
            free(sb); close(S.fd); close(W.fd);
        }
    }

    printf("\n[FP4 nibble 分布] (%lld 个)\n", ntot);
    const char *lbl[16] = {"0","+.5","+1","+1.5","+2","+3","+4","+6","-0","-.5","-1","-1.5","-2","-3","-4","-6"};
    double H = 0;
    for (int i = 0; i < 16; i++) {
        double p = (double)hist[i] / ntot;
        if (p > 0) H -= p * log2(p);
        printf("  %-5s %6.3f%%", lbl[i], 100*p);
        if (i % 4 == 3) printf("\n");
    }
    printf("★经验熵 %.4f bit/元素★ (满 4 bit 说明 FP4 无冗余可捡; 越低说明存在无损压缩空间)\n", H);

    printf("\n[重建余弦] 逐行, %lld 行样本\n", ccnt);
    const double bpw[4] = {0, 1.25, 2.25, 3.25};
    for (int nb = 1; nb <= 3; nb++)
        printf("  %d bit + 每32元素 ue8m0 scale = %.2f bpw:  均值 %.4f  最差 %.4f   (中位残差 %.1f%%)\n",
               nb, bpw[nb], csum[nb]/ccnt, cmin[nb], 100*sqrt(1 - (csum[nb]/ccnt)*(csum[nb]/ccnt)));
    printf("\n★目标 1.2963 bpw 落在 1.25 档 ⇒ 判决看第一行★\n");
    return 0;
}
