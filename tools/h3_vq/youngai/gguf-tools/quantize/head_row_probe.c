/* head_row_probe.c — lm_head(output.weight, Q4_K)逐行量化误差: 特殊 token 行 vs 普通行(2026-10-18)。
 *
 * 为什么: 10-17 FP 锚量到收口位 `</think>` 的概率被量化定向压 2.3~4.5 倍, 10-18 鹈鹕思考 43k 不收口。五指标尺按 token 平权,
 * 一个 `</think>` 位在 15360 里看不见; 要判"是不是结构病"得直接看 lm_head 里那一行被 Q4_K 压成了什么样 —— 特殊 token 行
 * (EOS / <think> / </think> / DSML 标记)要是系统性地比普通行误差大, 修法就是把这几行留高精度, 不用碰语料。
 * 用法: head_row_probe <gguf> <hf 目录> [行号…]   默认行 = 0 1 2 与 128790..128830(V4.1 特殊 token 段)。
 *   对照 = 1024 个随机普通行(seed 1)的相对 L2 误差分位。bf16 原件从 HF safetensors 的 head.weight 读(与 v41_to_gguf 同一来源)。
 * 编: cc -O2 -Isrc/common gguf-tools/quantize/head_row_probe.c src/common/ds4_gguf.c src/common/ds4_quantfmt.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ds4_gguf.h"
#include "ds4_quantfmt.h"
#include "ds4_st41.h"

typedef struct { double rel, maxabs, norm; } rowerr;

static rowerr row_err(const uint8_t *q, uint64_t nblk, const uint8_t *bf, uint64_t n, float *tmp) {
    ds4_deq_q4_K(q, nblk, tmp);
    double e2 = 0, r2 = 0, ma = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t u = (uint32_t)(bf[2 * i] | (bf[2 * i + 1] << 8)) << 16; float ref; memcpy(&ref, &u, 4);
        const double d = (double)tmp[i] - ref;
        e2 += d * d; r2 += (double)ref * ref; if (fabs(d) > ma) ma = fabs(d);
    }
    return (rowerr){ sqrt(e2 / (r2 > 0 ? r2 : 1)), ma, sqrt(r2) };
}
static int dcmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: %s <gguf> <hf目录> [行号…]\n", argv[0]); return 2; }
    ds4_gguf g; char err[256];
    if (ds4_gguf_open(&g, argv[1], err, sizeof err)) { fprintf(stderr, "gguf: %s\n", err); return 1; }
    const ds4_gguf_tensor *t = ds4_gguf_find(&g, "output.weight");
    if (!t || t->type != DS4_GGT_Q4_K) { fprintf(stderr, "output.weight 不在或不是 Q4_K(type %u)\n", t ? t->type : 0); return 1; }
    uint64_t blk, tsz; ds4_ggt_geom(t->type, &blk, &tsz);
    const uint64_t n = t->ne[0], rows = t->ne[1], nblk = n / blk, rowb = nblk * tsz;
    uint64_t nb; const uint8_t *qd = ds4_gguf_tensor_data(&g, t, &nb);
    v41_st S; if (v41_st_open(&S, argv[2])) { fprintf(stderr, "safetensors 打不开 %s\n", argv[2]); return 1; }
    const v41_st_ent *E = v41_st_find(&S, "head.weight");
    if (!E || strcmp(E->dtype, "BF16") || E->shape[0] != (int64_t)rows || E->shape[1] != (int64_t)n) { fprintf(stderr, "head.weight 形状/类型不对\n"); return 1; }
    const uint8_t *bf = v41_st_data(&S, E);
    float *tmp = malloc(n * sizeof(float));
    printf("output.weight Q4_K %llu 行 × %llu, 每行 %llu 块 %llu B\n", (unsigned long long)rows, (unsigned long long)n, (unsigned long long)nblk, (unsigned long long)rowb);

    /* 对照: 1024 个随机行 */
    enum { NR = 1024 }; double rel[NR], ma[NR], nm[NR]; srand(1);
    for (int i = 0; i < NR; i++) { uint64_t r = (uint64_t)rand() % rows; rowerr e = row_err(qd + r * rowb, nblk, bf + r * n * 2, n, tmp); rel[i] = e.rel; ma[i] = e.maxabs; nm[i] = e.norm; }
    qsort(rel, NR, sizeof rel[0], dcmp); qsort(ma, NR, sizeof ma[0], dcmp); qsort(nm, NR, sizeof nm[0], dcmp);
    printf("随机 %d 行 相对 L2 误差 p50 %.4f p90 %.4f p99 %.4f max %.4f | 最大绝对误差 p50 %.4f p99 %.4f | 行范数 p50 %.3f p1 %.3f p99 %.3f\n",
           NR, rel[NR / 2], rel[NR * 9 / 10], rel[NR * 99 / 100], rel[NR - 1], ma[NR / 2], ma[NR * 99 / 100], nm[NR / 2], nm[NR / 100], nm[NR * 99 / 100]);

    printf("%8s %10s %10s %10s  %s\n", "行", "相对L2", "最大绝对", "行范数", "在随机行分位");
    int argrows = argc - 3;
    for (int k = 0; k < (argrows ? argrows : 3 + 41); k++) {
        uint64_t r;
        if (argrows) r = strtoull(argv[3 + k], NULL, 10);
        else r = k < 3 ? (uint64_t)k : 128790 + (uint64_t)(k - 3);
        if (r >= rows) continue;
        rowerr e = row_err(qd + r * rowb, nblk, bf + r * n * 2, n, tmp);
        int pct = 0; while (pct < NR && rel[pct] < e.rel) pct++;
        printf("%8llu %10.4f %10.4f %10.3f  p%d%s\n", (unsigned long long)r, e.rel, e.maxabs, e.norm, pct * 100 / NR, pct >= NR * 99 / 100 ? "  ★" : "");
    }
    v41_st_close(&S); ds4_gguf_close(&g); free(tmp);
    return 0;
}
