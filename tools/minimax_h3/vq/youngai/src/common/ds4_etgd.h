/* ds4_etgd.h — 引擎 --score-topk / --eval-topk 产物("ETGD")的唯一读取实现(2026-09-13)。
 *
 * 谁写: core_score_aux.c(引擎, 两条打分路共用)。谁读: 后训练解算器(靶要每行的 p)与判决器
 * anchor_metrics(--ref-topk 量决策点 argmax)。三方都按这一份字节约定走 —— 抄第二份迟早对不上,
 * 而对不上不会报错, 只会给出一个安静的错数。
 *
 * 布局: 头 <u32 'ETGD'><u32 K><u32 S><u32 VOCAB>, 之后 S 条定长记录, 第 i 条:
 *   <row u32 = i><tgt i32><tgt_p f32><mass f32><ids i32[K]><ps f32[K]>
 * tgt = 位置 i 该预测的 token(= ids[i+1]), 末位为 -1; ps 是归一化后的概率;
 * mass = top-K 覆盖的概率质量(这条近似有多糙的自证, 别只信"应该够了")。
 *
 * 纯头文件 + static: 读一个几 MB 的表不值得再加一个编译单元。 */
#ifndef DS4_ETGD_H
#define DS4_ETGD_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DS4_ETGD_MAGIC 0x44475445u

typedef struct {
    int   *tgt;      /* [n] 目标 token, <0 = 末位没有下一个 token */
    float *tgt_p;    /* [n] 目标 token 的概率(不在 top-K 里时引擎仍按全词表算的真值) */
    float *mass;     /* [n] top-K 覆盖的概率质量 */
    int   *ids;      /* [n*K] */
    float *ps;       /* [n*K] */
    int    n, K, S, vocab;
} ds4_etgd;

/* 读整份。失败返回非 0 并打印原因(o 保持全 0)。 */
static int ds4_etgd_read(const char *path, ds4_etgd *o) {
    memset(o, 0, sizeof *o);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "★top-K 表打不开: %s★\n", path); return -1; }
    uint32_t hd[4];
    if (fread(hd, 4, 4, f) != 4 || hd[0] != DS4_ETGD_MAGIC) {
        fprintf(stderr, "★%s 不是 --score-topk 产物★\n", path); fclose(f); return -1; }
    o->K = (int)hd[1]; o->S = (int)hd[2]; o->vocab = (int)hd[3];
    if (o->K <= 0 || o->K > 4096) { fprintf(stderr, "★%s K=%d 不合理★\n", path, o->K); fclose(f); return -1; }
    const size_t rec = 16u + (size_t)o->K * 8u;
    const long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f) - pos;
    fseek(f, pos, SEEK_SET);
    if (sz <= 0 || (size_t)sz % rec) {
        fprintf(stderr, "★%s 记录区 %ld 不是 %zu 的整数倍★\n", path, sz, rec); fclose(f); return -1; }
    o->n = (int)((size_t)sz / rec);
    o->tgt = (int *)malloc((size_t)o->n * sizeof(int));
    o->tgt_p = (float *)malloc((size_t)o->n * sizeof(float));
    o->mass = (float *)malloc((size_t)o->n * sizeof(float));
    o->ids = (int *)malloc((size_t)o->n * (size_t)o->K * sizeof(int));
    o->ps = (float *)malloc((size_t)o->n * (size_t)o->K * sizeof(float));
    if (!o->tgt || !o->tgt_p || !o->mass || !o->ids || !o->ps) {
        fprintf(stderr, "★top-K 表内存不足★\n"); fclose(f); return -1; }
    for (int i = 0; i < o->n; i++) {
        uint32_t row = 0;
        if (fread(&row, 4, 1, f) != 1 || fread(&o->tgt[i], 4, 1, f) != 1 ||
            fread(&o->tgt_p[i], 4, 1, f) != 1 || fread(&o->mass[i], 4, 1, f) != 1 ||
            fread(o->ids + (size_t)i * o->K, 4, (size_t)o->K, f) != (size_t)o->K ||
            fread(o->ps + (size_t)i * o->K, 4, (size_t)o->K, f) != (size_t)o->K) {
            fprintf(stderr, "★%s 在第 %d 行截断★\n", path, i); fclose(f); return -1; }
        /* 行号自证: 记录里写的就是绝对行号, 与 ids 文件同一口径。对不上说明文件不是连续行
         * (比如钩子提前停车那趟的半成品), 那种错在下游只会表现为"判错了行"。 */
        if ((int)row != i) { fprintf(stderr, "★%s 第 %d 条记的行号是 %u★\n", path, i, row); fclose(f); return -1; }
    }
    fclose(f);
    return 0;
}

static void ds4_etgd_free(ds4_etgd *o) {
    free(o->tgt); free(o->tgt_p); free(o->mass); free(o->ids); free(o->ps);
    memset(o, 0, sizeof *o);
}

/* 行 i 上某个 token 的概率: 在 top-K 里就返回它的 p, 不在返回 0(它已经很小了)。 */
static float ds4_etgd_p_of(const ds4_etgd *o, int i, int token) {
    const int *ids = o->ids + (size_t)i * o->K;
    const float *ps = o->ps + (size_t)i * o->K;
    for (int k = 0; k < o->K; k++) if (ids[k] == token) return ps[k];
    return 0.f;
}

/* 行 i 的 top-1(argmax)。p_out 非空时回填它的概率。 */
static int ds4_etgd_top1(const ds4_etgd *o, int i, float *p_out) {
    const int *ids = o->ids + (size_t)i * o->K;
    const float *ps = o->ps + (size_t)i * o->K;
    int b = 0;
    for (int k = 1; k < o->K; k++) if (ps[k] > ps[b]) b = k;
    if (p_out) *p_out = ps[b];
    return ids[b];
}

#endif /* DS4_ETGD_H */
