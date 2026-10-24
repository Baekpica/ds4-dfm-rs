/* engram_probe.c — V4.1 engram 查表的量化可行性探针(2026-09-10)。
 *
 * 【要回答什么】engram 两张 3.84 亿行 × 256 的 FP8 表共 196.9 B 参数, 占全模型 25.8%。
 * 落地 100 GB 的方案里它只能拿到 1 bpw(24.6 GB)。问题就一个: 1 bpw 之后还剩多少。
 *
 * 【为什么能不带引擎测】engram 是 embedding 查表, 不参与矩阵乘 —— 查出来的向量准不准
 * 就是它的全部。所以离线读几十万行、量化、比余弦, 就能给出判断, 不必等引擎支持 V4.1。
 *
 * 【采样怎么取】3.84 亿行不可能全读(98 GB)。取 NBLK 个随机起点各读 RPB 连续行:
 * 纯随机单行 pread 会打成几十万次 264 字节的随机 I/O(盘上是灾难), 而分散起点足以避免
 * "只看到相邻 n-gram"的偏差。种子固定 ⇒ 可复现(铁律: 捕获必须可复现)。
 *
 * 【这里为什么没有 GPU】本针只做 dequant + 闭式 signref(gain×sign, 无迭代)+ 统计直方图,
 * 全是 O(n) 一遍过, 瓶颈在读盘不在算。铁律禁的是"为了能编过补一个标量参考版顶替生产
 * GPU kernel", 以及拿 CPU 路出速度读数 —— 本针不产速度读数, signref 闭式与 GPU 逐位同解。
 * 需要 k-means 的 VQ 档若要测, 走 vq_gpu.cu 的 vqg_assign 生产路, 不在这里补 CPU 版。
 *
 * 编译: cc -O3 -o engram_probe engram_probe.c -lm
 * 用法: engram_probe <hf-dir> <张量前缀, 如 layers.1.engram.embed> [块数 每块行数 种子]
 */
/* -std=c99 下 unistd.h 默认不暴露 pread ⇒ 隐式声明会按 int 收返回值(ssize_t 被截),
 * 现在块小看不出来, 换大块就是静默错读。显式开 POSIX. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>

/* ---- FP8 解码。E4M3 用全仓唯一那份(src/common/ds4_fp8.h); E8M0 是 V4.1 新的 scale 格式:
 * 无符号 8 位纯指数, 值 = 2^(b-127), b=255 为 NaN。V4 时代的 scale 是 F32, 所以这份是新的。 */
#include "../../src/common/ds4_fp8.h"

static float e8m0(uint8_t b) { return b == 255 ? NAN : ldexpf(1.0f, (int)b - 127); }

/* ---- safetensors 头里定位一个张量: 返回所在文件 fd、数据区起点、形状 ---- */
typedef struct { int fd; long long off, nbytes; long long shape[4]; int nd; char dtype[32]; } tref_t;

static int find_tensor(const char *dir, const char *name, tref_t *T) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "目录打不开 %s\n", dir); return -1; }
    struct dirent *e; char path[4096]; int found = 0;
    while ((e = readdir(d)) && !found) {
        const char *s = strstr(e->d_name, ".safetensors");
        if (!s || s[12]) continue;
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        FILE *f = fopen(path, "rb"); if (!f) continue;
        uint64_t hlen = 0;
        if (fread(&hlen, 8, 1, f) != 1 || hlen < 2 || hlen > (1u << 30)) { fclose(f); continue; }
        char *hdr = malloc(hlen + 1);
        if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); continue; }
        hdr[hlen] = 0; fclose(f);
        char key[512]; snprintf(key, sizeof key, "\"%s\"", name);
        char *p = strstr(hdr, key);
        if (p) {
            char *dt = strstr(p, "\"dtype\":\"");
            char *sh = strstr(p, "\"shape\":[");
            char *of = strstr(p, "\"data_offsets\":[");
            if (dt && sh && of) {
                dt += 9; int i = 0; while (dt[i] && dt[i] != '"' && i < 31) { T->dtype[i] = dt[i]; i++; }
                T->dtype[i] = 0;
                T->nd = 0; char *q = sh + 9;
                while (*q && *q != ']' && T->nd < 4) { T->shape[T->nd++] = strtoll(q, &q, 10); if (*q == ',') q++; }
                q = of + 16;
                long long s0 = strtoll(q, &q, 10); if (*q == ',') q++;
                long long s1 = strtoll(q, &q, 10);
                /* 数据区起点 = 8(头长字段) + header_len; 张量在其中的偏移是 s0 */
                T->off = 8 + (long long)hlen + s0;
                T->nbytes = s1 - s0;
                T->fd = open(path, O_RDONLY);
                if (T->fd < 0) { fprintf(stderr, "打不开 %s\n", path); free(hdr); closedir(d); return -1; }
                found = 1;
                fprintf(stderr, "[定位] %s → %s dtype=%s shape=%lldx%lld 字节=%lld\n",
                        name, e->d_name, T->dtype, T->shape[0], T->nd > 1 ? T->shape[1] : 1, T->nbytes);
            }
        }
        free(hdr);
    }
    closedir(d);
    return found ? 0 : -1;
}

static int cmpf(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: engram_probe <hf-dir> <前缀 layers.1.engram.embed> [块数 每块行数 种子]\n"); return 2; }
    const char *dir = argv[1], *pre = argv[2];
    int NBLK = argc > 3 ? atoi(argv[3]) : 1024;
    int RPB  = argc > 4 ? atoi(argv[4]) : 256;
    unsigned seed = argc > 5 ? (unsigned)atoi(argv[5]) : 1;

    char nw[512], ns[512];
    snprintf(nw, sizeof nw, "%s.weight", pre);
    snprintf(ns, sizeof ns, "%s.scale", pre);
    tref_t W, S;
    if (find_tensor(dir, nw, &W) || find_tensor(dir, ns, &S)) { fprintf(stderr, "★张量找不到★\n"); return 2; }

    long long NROW = W.shape[0], DIM = W.shape[1], NSC = S.shape[1];
    if (strcmp(W.dtype, "F8_E4M3") || strcmp(S.dtype, "F8_E8M0")) {
        fprintf(stderr, "★dtype 不是预期的 E4M3/E8M0 (%s/%s), 解码会错, 硬拒★\n", W.dtype, S.dtype); return 2;
    }
    if (S.shape[0] != NROW || DIM % NSC) { fprintf(stderr, "★scale 形状对不上★\n"); return 2; }
    long long BLK = DIM / NSC;          /* 每个 scale 管多少列 */
    long long M = (long long)NBLK * RPB;
    fprintf(stderr, "[采样] %d 块 × %d 行 = %lld 行 / 共 %lld 行 (%.4f%%), 每 scale 管 %lld 列\n",
            NBLK, RPB, M, NROW, 100.0 * M / NROW, BLK);

    uint8_t *wb = malloc((size_t)RPB * DIM), *sb = malloc((size_t)RPB * NSC);
    float *row = malloc((size_t)DIM * 4);
    float *cosv = malloc((size_t)M * 4);     /* signref 每行重建余弦 */
    float *nrm  = malloc((size_t)M * 4);     /* 每行 L2 范数 */
    if (!wb || !sb || !row || !cosv || !nrm) { fprintf(stderr, "内存不足\n"); return 1; }

    long long nz_rows = 0, kept = 0;
    double abs_sum = 0; long long abs_n = 0;
    long long hist[8] = {0};                  /* |w| 量级直方图: 0, <1e-4, <1e-3, <1e-2, <0.1, <1, <10, ≥10 */
    srand(seed);
    for (int b = 0; b < NBLK; b++) {
        long long r0 = (long long)((double)rand() / ((double)RAND_MAX + 1) * (double)(NROW - RPB));
        if (pread(W.fd, wb, (size_t)RPB * DIM, W.off + r0 * DIM) != (ssize_t)((size_t)RPB * DIM)) {
            fprintf(stderr, "★weight 读不满 @行%lld★\n", r0); return 1;
        }
        if (pread(S.fd, sb, (size_t)RPB * NSC, S.off + r0 * NSC) != (ssize_t)((size_t)RPB * NSC)) {
            fprintf(stderr, "★scale 读不满 @行%lld★\n", r0); return 1;
        }
        for (int r = 0; r < RPB; r++) {
            const uint8_t *wr = wb + (size_t)r * DIM, *sr = sb + (size_t)r * NSC;
            double n2 = 0, l1 = 0;
            for (long long c = 0; c < DIM; c++) {
                float sc = e8m0(sr[c / BLK]);
                float v = ds4_e4m3fn_to_f32(wr[c]) * (isnan(sc) ? 0.0f : sc);
                row[c] = v; n2 += (double)v * v; l1 += fabs(v);
                float a = fabsf(v);
                int k = a == 0 ? 0 : a < 1e-4f ? 1 : a < 1e-3f ? 2 : a < 1e-2f ? 3 : a < 0.1f ? 4 : a < 1.0f ? 5 : a < 10.0f ? 6 : 7;
                hist[k]++;
                abs_sum += a; abs_n++;
            }
            nrm[kept] = (float)sqrt(n2);
            if (n2 == 0) { nz_rows++; cosv[kept] = 1.0f; kept++; continue; }  /* 全零行: 1bit 也无损 */
            /* signref: 每行一个增益 g = L1/DIM, 重建 = g·sign(w)。闭式最小化 L2 的解就是 L1 均值。 */
            double g = l1 / (double)DIM, dot = 0, q2 = 0;
            for (long long c = 0; c < DIM; c++) {
                double q = row[c] > 0 ? g : row[c] < 0 ? -g : 0.0;
                dot += q * row[c]; q2 += q * q;
            }
            cosv[kept] = (float)(q2 > 0 ? dot / (sqrt(q2) * sqrt(n2)) : 0.0);
            kept++;
        }
    }

    qsort(cosv, kept, 4, cmpf);
    qsort(nrm,  kept, 4, cmpf);
    printf("=== engram 探针: %s ===\n", pre);
    printf("采样 %lld 行, 全零行 %lld (%.2f%%)\n", kept, nz_rows, 100.0 * nz_rows / kept);
    printf("|w| 量级分布: 零 %.2f%% | <1e-4 %.2f%% | <1e-3 %.2f%% | <1e-2 %.2f%% | <0.1 %.2f%% | <1 %.2f%% | <10 %.2f%% | ≥10 %.2f%%\n",
           100.0*hist[0]/abs_n, 100.0*hist[1]/abs_n, 100.0*hist[2]/abs_n, 100.0*hist[3]/abs_n,
           100.0*hist[4]/abs_n, 100.0*hist[5]/abs_n, 100.0*hist[6]/abs_n, 100.0*hist[7]/abs_n);
    printf("平均 |w| = %.6g\n", abs_sum / abs_n);
    printf("行 L2 范数: p1 %.4g  p10 %.4g  p50 %.4g  p90 %.4g  p99 %.4g\n",
           nrm[kept/100], nrm[kept/10], nrm[kept/2], nrm[kept*9/10], nrm[kept*99/100]);
    double cos_mean = 0;
    for (long long i = 0; i < kept; i++) cos_mean += cosv[i];
    cos_mean /= (double)kept;
    printf("★signref(1.0625 bpw) 重建余弦: 最差 %.4f  p1 %.4f  p10 %.4f  p50 %.4f  p90 %.4f  均值 %.4f★\n",
           cosv[0], cosv[kept/100], cosv[kept/10], cosv[kept/2], cosv[kept*9/10], cos_mean);
    close(W.fd); close(S.fd);
    return 0;
}
