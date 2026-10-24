/* v41_quantize.c — DeepSeek-V4.1-Flash HF → 量化模型目录, 纯权重量化不需要前向(2026-09-12)。
 *
 * 【配方(用户 09-12 定, 09-19 加 100 GB 档)】
 *   routed 专家(40 层 × 384)  VQ dim8×nc<N>: 索引 log2(N) bpw + f16 码本(每专家) + f16 行增益。
 *                              nc4096 = 1.519 bpw(09-12 现役) / nc2048 = 1.387(100 GB 档)。
 *                              ★全程零语料★: 码本用 Lloyd 解在权重向量自身上, 行增益取自权重。
 *   骨架(attn/shared/indexer 投影, embed/head, vision 大矩阵)
 *                              --skel fp4 : E2M1 + 每 32 列 ue8m0 = 4.25 bpw(09-12 现役)
 *                              --skel q4k : q4_K 144 B/256 = 4.5 bpw(09-19; 零语料, 见 v41_q4k.cu)
 *   MTP 三塔自带专家(3×128)   --mtp-vq 时与主干同款 VQ(7.22 → 2.36 GB); 否则出厂 FP4 原样透传
 *   engram 查表(203 GB)        不动: 留在 HF 第 47/48 分片, 输出目录软链过去(量化器不写它)
 *   其余(norm/gate/hc/sink/小 BF16 投影)  原样透传
 *
 * 【输出】<out>/model-layerNN.safetensors × 40 + model-common.safetensors + model.safetensors.index.json。
 * 每层一个分片: 断点续跑按层跳过已完成的(最终名存在才算完成, 半成品是 .part)。
 * 【读它】v41_hf_io.py(教师端)认 index.json: 骨架 FP4 与出厂 routed 专家同格式零改动;
 * VQ 三件(idx/cb/gain)走 libv41vq.so 的 v41_vq_decode_gpu —— 与量化器内部算残差的是同一个核。
 *
 * 用法: v41_quantize <hf-dir> <out-dir> [--vq-dim 8] [--vq-nc 4096] [--vq-iters 8] [--vq-stride 8]
 *                    [--skel fp4|q4k] [--mtp-vq] [--layers a:b] [--no-common] [--force]
 *                    [--vq-nc-table FILE]   逐专家 nc 表("L e nc", v41_nc_alloc 产物; 2026-09-21 动态位宽针; 缺项走全局 --vq-nc)
 * 编译: make -C gguf-tools v41_quantize (Linux+CUDA) */
#include "../../src/common/ds4_st41.h"
#include "v41_st_write.h"
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

#include "v41_calib.h"
#include "v41_ef.h"
#include "v41_vq_rate.h"
typedef int (*v41_vq_ef_fn)(void *ud, int m, float *V, int rows, int cols, const float *C, int nc, int dim, const float *g, int *idx);
extern int v41_vq_expert_from_fp4(const uint8_t *const *w, const uint8_t *const *s, const int *rows, const int *cols,
                                  int nmat, int dim, int nc, int iters, int stride, const float *cwx, const float *cwh, int cw_stats_only,
                                  v41_vq_ef_fn ef, void *ef_ud, const v41_vq_opt *opt,
                                  uint8_t **idx_out, uint16_t *cb_out, uint16_t **gain_out, double *stats);
/* 共享码本探针(2026-09-21, v41_vq_pool.inc.cu): 每专家归一化向量取样进池 / 池上训一本 */
extern int v41_vq_pool_sample_from_fp4(const uint8_t *const *w, const uint8_t *const *s, const int *rows, const int *cols,
                                       int nmat, int dim, int stride, float *out_host, long long cap, long long *n_io);
extern int v41_vq_train_codebook(const float *V_host, long long nv, int dim, int nc, int iters, float lam, int cb_fp8, uint16_t *cb_out);
extern int v41_fp4_requant_host(const uint8_t *w, const uint8_t *s, const char *dtype, int rows, int cols,
                                int sbr, int sbc, uint8_t *packed_out, uint8_t *scale_out, double *stats);
extern int v41_q4k_requant_host(const uint8_t *w, const uint8_t *s, const char *dtype, int rows, int cols,
                                int sbr, int sbc, uint8_t *blocks_out, double *stats);

/* calib: 金融域校准料目录(v41_amp_run --dump-calib 落的 calib_Lnn.bin; 2026-09-20 深夜, 用户令"按金融域量化"), NULL = 零语料。
 * calib_ab: 机制审计 —— 每个专家再走一遍平权指派只记账不落盘, 同一把加权尺上 校准 必须 < 平权(冒烟用, 量化时间翻倍) */
typedef struct { int dim, nc, iters, stride, l0, l1, common, force, skel_q4k, mtp_vq; const char *calib; int calib_ab; float ef_betas[4]; int ef_nb; float alpha;
                 float lam; int shared_cb, cb_fp8; } cfg_t;   /* ef_nb>0 = 级 2; alpha = 列权指数; lam/shared_cb/cb_fp8 = 率侧探针(v41_vq_rate.h) */
enum { J_PASS, J_FP4, J_VQ, J_Q4K };
typedef struct {
    int kind;
    const v41_st_ent *w, *s;              /* PASS/FP4 */
    const v41_st_ent *ew[3], *es[3];      /* VQ: w1, w3, w2 */
    int t[8];                             /* 输出张量序号 */
    int e;                                /* VQ: 专家号(校准料按它取名下行) */
    int nc;                               /* VQ: 本专家码本大小(--vq-nc-table / --vq-nc-layers 逐层, 否则全局 C->nc) */
    int grp;                              /* VQ: 共享码本的组号(主干层 = 0, common 里按 MTP 塔号; 见 v41_shcb.inc.c) */
} job_t;
typedef struct { double sse, en, wsse, wen; uint64_t bytes; long long params; } acc_t;   /* wsse/wen = 列权加权的残差账(平权时 = sse/en) */
static acc_t g_vq, g_fp4, g_q4k, g_pass; /* 全程账 */
static acc_t g_ab;                       /* --calib-ab 的平权对照账(本分片) */
static v41_calib *g_cal;                 /* 本层校准料(common/三塔无校准 ⇒ NULL = 平权, 与今天同) */
static int g_cal_nh, g_cal_fb;           /* 本层 w2 列权: 逐专家算出的 / 名下 <8 行退回平权的 */
static v41_ef *g_ef;                     /* 本层级 2(误差反馈)上下文, NULL = 不做 */
static double g_q4k_sec;                 /* q4_K 编码在主机的累计秒数(铁律: CPU 路必须报耗时) */
/* --vq-nc-table(2026-09-21 动态位宽): [nlayers][nexp] 逐专家 nc, NULL = 全局 C->nc。只管主干 routed 专家, 三塔仍走全局 */
static int *g_nct; static const char *g_nct_path;
/* --vq-nc-layers a:b=NC(2026-09-21 方案 v3, 113.md): 整层一档位宽。比逐专家表好写好核(14 层 13 位 = 一个参数 vs 5376 行),
 * 而且共享码本要求"一层一个 nc"(一层一本码本喂不了两种码本大小) —— 逐专家表在共享模式下会被 shcb_build_group 拒掉。 */
#define NCL_MAX 8
static struct { int a, b, nc; } g_ncl[NCL_MAX]; static int g_nncl;
/* --stats-out(2026-09-21): 逐专家落 "L e nc sse energy bytes" —— sse 就是这个专家在本位宽下的【真实量化误差】,
 * 分配器拿它当重要性(纯权重, 零语料), 比只按能量分更准(能量 × 好不好编 都在里面了) */
static FILE *g_stats; static int g_stats_layer;
/* 率侧探针(2026-09-21): 共享码本按组放在 g_shcb[](v41_shcb.inc.c); g_H = 本分片索引熵累计(分片行报均值) */
static double g_H_sum; static int g_H_n, g_H_nc;   /* g_H_nc = 本分片最后一个专家的 nc(报"等概率几位"要按它, 按全局 nc 报会在逐层混位宽时误导) */
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static int bits_of(int nc) { int b = 1; while ((1 << b) < nc) b++; return b; }
static int ends_with(const char *s, const char *suf) { size_t a = strlen(s), b = strlen(suf); return a >= b && !strcmp(s + a - b, suf); }
static int is_routed_expert(const char *n) { return !strncmp(n, "layers.", 7) && strstr(n, ".ffn.experts.") != NULL; }
static int is_mtp_expert(const char *n) { return !strncmp(n, "mtp.", 4) && strstr(n, ".ffn.experts.") != NULL; }

/* 哪些张量重量化到 FP4(见文件头"配方") */
static int wants_fp4(const v41_st_ent *E) {
    if (E->nd != 2 || !ends_with(E->name, ".weight")) return 0;
    if (strstr(E->name, ".gate.") || strstr(E->name, ".compressor.") || strstr(E->name, ".indexer.w") ||
        strstr(E->name, ".indexer.k_norm") || strstr(E->name, ".indexer.weights_proj")) {
        if (strcmp(E->dtype, "F8_E4M3")) return 0;        /* 官方留 BF16 的小投影不动; F8 的照常 */
    }
    if (!strcmp(E->dtype, "F8_E4M3")) return E->shape[1] % 32 == 0;
    if (!strcmp(E->dtype, "BF16")) {
        int big = (!strcmp(E->name, "embed.weight") || !strcmp(E->name, "head.weight") || !strncmp(E->name, "vision.", 7));
        return big && E->shape[0] * E->shape[1] >= 1000000 && E->shape[1] % 32 == 0;
    }
    return 0;
}

static const v41_st_ent *sibling_scale(const v41_st *S, const v41_st_ent *W) {
    char n[256]; size_t L = strlen(W->name);
    if (L < 7 || L + 1 > sizeof n) return NULL;
    memcpy(n, W->name, L - 6); strcpy(n + L - 6, "scale");
    return v41_st_find(S, n);
}

/* 共享码本(一组一本)的取样与训练: 要 cfg_t/job_t/now_s, 所以在它们之后 include */
#include "v41_shcb.inc.c"

/* ---- 计划: 给一个分片登记全部输出张量 ---- */
static int plan_pass(v41_stw *W, job_t *J, const v41_st *S, const v41_st_ent *E) {
    J->kind = J_PASS; J->w = E; J->s = ends_with(E->name, ".weight") ? sibling_scale(S, E) : NULL;
    J->t[0] = v41_stw_plan(W, E->name, E->dtype, E->nd, E->shape, E->nbytes);
    if (J->s) J->t[1] = v41_stw_plan(W, J->s->name, J->s->dtype, J->s->nd, J->s->shape, J->s->nbytes);
    return 0;
}
static int plan_fp4(v41_stw *W, job_t *J, const v41_st *S, const v41_st_ent *E) {
    J->kind = J_FP4; J->w = E; J->s = sibling_scale(S, E);
    if (!strcmp(E->dtype, "F8_E4M3") && !J->s) { fprintf(stderr, "★%s 是 F8 却没有 scale★\n", E->name); return -1; }
    int64_t rows = E->shape[0], cols = E->shape[1];
    int64_t shw[2] = {rows, cols / 2}, shs[2] = {rows, cols / 32};
    char sn[256]; snprintf(sn, sizeof sn, "%.*sscale", (int)strlen(E->name) - 6, E->name);
    J->t[0] = v41_stw_plan(W, E->name, "I8", 2, shw, (uint64_t)rows * cols / 2);
    J->t[1] = v41_stw_plan(W, sn, "F8_E8M0", 2, shs, (uint64_t)rows * (cols / 32));
    return 0;
}
/* q4_K: 一张张量一个输出, dtype "Q4_K" + 【逻辑形状】; 字节数 = rows×(cols/256)×144。
 * safetensors 的字节数由 data_offsets 定、不由 dtype 推 ⇒ 自定义 dtype 名安全(F8_E8M0 同例)。 */
static int plan_q4k(v41_stw *W, job_t *J, const v41_st *S, const v41_st_ent *E) {
    J->kind = J_Q4K; J->w = E; J->s = sibling_scale(S, E);
    if (!strcmp(E->dtype, "F8_E4M3") && !J->s) { fprintf(stderr, "★%s 是 F8 却没有 scale★\n", E->name); return -1; }
    int64_t rows = E->shape[0], cols = E->shape[1], sh[2] = {rows, cols};
    J->t[0] = v41_stw_plan(W, E->name, "Q4_K", 2, sh, (uint64_t)rows * (cols / 256) * 144u);
    return 0;
}

/* pre = "layers.7" 或 "mtp.1" —— 主干 routed 与 MTP 三塔共用这一份计划器(格式完全相同) */
static int plan_vq(v41_stw *W, job_t *J, const v41_st *S, const char *pre, int e, const cfg_t *C, int nc, int grp) {
    static const char *mat[3] = {"w1", "w3", "w2"};
    J->kind = J_VQ; J->e = e; J->nc = nc; J->grp = grp;
    char n[256];
    snprintf(n, sizeof n, "%s.ffn.experts.%d.vq.cb", pre, e);
    int64_t shc[2] = {nc, C->dim};
    J->t[0] = v41_stw_plan(W, n, "F16", 2, shc, (uint64_t)nc * C->dim * 2);
    int bits = bits_of(nc);
    for (int m = 0; m < 3; m++) {
        snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.weight", pre, e, mat[m]);
        J->ew[m] = v41_st_find(S, n);
        snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.scale", pre, e, mat[m]);
        J->es[m] = v41_st_find(S, n);
        if (!J->ew[m] || !J->es[m] || strcmp(J->ew[m]->dtype, "I8")) { fprintf(stderr, "★缺专家张量或非 FP4: %s★\n", n); return -1; }
        int64_t rows = J->ew[m]->shape[0], cols = J->ew[m]->shape[1] * 2;
        if (cols % C->dim) { fprintf(stderr, "★%s 列数 %lld 不是 dim 倍数★\n", n, (long long)cols); return -1; }
        int64_t nidx_row = cols / C->dim, bytes_row = (nidx_row * bits + 7) / 8;
        int64_t shi[2] = {rows, bytes_row}, shg[1] = {rows};
        snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.idx", pre, e, mat[m]);
        J->t[1 + m * 2] = v41_stw_plan(W, n, "U8", 2, shi, (uint64_t)rows * bytes_row);
        snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.gain", pre, e, mat[m]);
        J->t[2 + m * 2] = v41_stw_plan(W, n, "F16", 1, shg, (uint64_t)rows * 2);
    }
    return 0;
}

/* ---- 执行一个 job: 算 + 写 ---- */
static int run_job(v41_stw *W, v41_st *S, const job_t *J, const cfg_t *C, acc_t *avq, acc_t *afp4, acc_t *aq4k) {
    double st[4] = {0, 0, 0, 0};
    if (J->kind == J_PASS) {
        const uint8_t *d = v41_st_data(S, J->w); if (!d) return -1;
        if (v41_stw_write(W, J->t[0], d, J->w->nbytes)) return -1;
        g_pass.bytes += J->w->nbytes; g_pass.params += v41_st_numel(J->w) * (!strcmp(J->w->dtype, "I8") ? 2 : 1);
        if (J->s) { d = v41_st_data(S, J->s); if (!d || v41_stw_write(W, J->t[1], d, J->s->nbytes)) return -1; g_pass.bytes += J->s->nbytes; }
        return 0;
    }
    if (J->kind == J_FP4) {
        int rows = (int)J->w->shape[0], cols = (int)J->w->shape[1];
        const uint8_t *w = v41_st_data(S, J->w), *s = J->s ? v41_st_data(S, J->s) : NULL;
        if (!w || (J->s && !s)) return -1;
        int sbr = J->s ? (int)(rows / J->s->shape[0]) : 1, sbc = J->s ? (int)(cols / J->s->shape[1]) : 1;
        uint8_t *pk = (uint8_t *)malloc((size_t)rows * cols / 2), *sc = (uint8_t *)malloc((size_t)rows * (cols / 32));
        int rc = v41_fp4_requant_host(w, s, J->w->dtype, rows, cols, sbr, sbc, pk, sc, st);
        if (rc) { free(pk); free(sc); fprintf(stderr, "★FP4 失败 %s rc=%d★\n", J->w->name, rc); return -1; }
        if (v41_stw_write(W, J->t[0], pk, (uint64_t)rows * cols / 2) || v41_stw_write(W, J->t[1], sc, (uint64_t)rows * (cols / 32))) return -1;
        free(pk); free(sc);
        afp4->sse += st[0]; afp4->en += st[1]; afp4->bytes += (uint64_t)rows * cols / 2 + (uint64_t)rows * (cols / 32); afp4->params += (long long)rows * cols;
        return 0;
    }
    if (J->kind == J_Q4K) {
        int rows = (int)J->w->shape[0], cols = (int)J->w->shape[1];
        const uint8_t *w = v41_st_data(S, J->w), *s = J->s ? v41_st_data(S, J->s) : NULL;
        if (!w || (J->s && !s)) return -1;
        int sbr = J->s ? (int)(rows / J->s->shape[0]) : 1, sbc = J->s ? (int)(cols / J->s->shape[1]) : 1;
        uint64_t nb = (uint64_t)rows * (cols / 256) * 144u;
        uint8_t *blk = (uint8_t *)malloc(nb);
        if (!blk) { fprintf(stderr, "★q4_K 缓冲 %.2f GB 要不到★\n", nb / 1e9); return -1; }
        double tq = now_s();
        int rc = v41_q4k_requant_host(w, s, J->w->dtype, rows, cols, sbr, sbc, blk, st);
        g_q4k_sec += now_s() - tq;
        if (rc) { free(blk); fprintf(stderr, "★q4_K 失败 %s rc=%d★\n", J->w->name, rc); return -1; }
        if (v41_stw_write(W, J->t[0], blk, nb)) { free(blk); return -1; }
        free(blk);
        aq4k->sse += st[0]; aq4k->en += st[1]; aq4k->bytes += nb; aq4k->params += (long long)rows * cols;
        return 0;
    }
    /* VQ: 三矩阵一份码本; nc 逐专家(--vq-nc-table)或全局 */
    const uint8_t *w[3], *s[3]; int rows[3], cols[3]; uint8_t *idx[3]; uint16_t *gain[3];
    const int nc = J->nc; int bits = bits_of(nc);
    /* 码本比 4096 大时训练采样步长按比例缩小, 保住"每个码字约 135 个训练向量"(stride 8 是按 nc4096 定的, 不缩的话
     * nc16384 每码字只剩 34 个, 码本训不实)。只对 nc>4096 的专家生效 ⇒ 12 位及以下的产物逐字节不变。 */
    const int stride = nc > 4096 ? (C->stride * 4096 / nc > 0 ? C->stride * 4096 / nc : 1) : C->stride;
    for (int m = 0; m < 3; m++) {
        w[m] = v41_st_data(S, J->ew[m]); s[m] = v41_st_data(S, J->es[m]);
        if (!w[m] || !s[m]) return -1;
        rows[m] = (int)J->ew[m]->shape[0]; cols[m] = (int)J->ew[m]->shape[1] * 2;
        idx[m] = (uint8_t *)malloc((size_t)rows[m] * ((cols[m] / C->dim * bits + 7) / 8));
        gain[m] = (uint16_t *)malloc(sizeof(uint16_t) * rows[m]);
    }
    uint16_t *cb = (uint16_t *)malloc(sizeof(uint16_t) * nc * C->dim);
    /* 校准路(2026-09-20): w1/w3 吃层级 E[x²] 列权, w2 吃本专家 E[h_e²](名下 <8 行退回平权, 分片行报计数) */
    const float *cwx = NULL, *cwh = NULL; static float cwh_buf[65536];
    if (g_cal) {
        cwx = v41_calib_colw_x(g_cal);
        if (cols[2] > 65536) { fprintf(stderr, "★w2 列数 %d 超过列权缓冲★\n", cols[2]); return -1; }
        const int nh = v41_calib_colw_h(g_cal, J->e, w[0], s[0], w[1], s[1], rows[0], cols[0], cwh_buf);
        if (nh < 0) { fprintf(stderr, "★专家 %d 的 h 列权失败★\n", J->e); return -1; }
        if (nh >= 8) { cwh = cwh_buf; g_cal_nh++; } else g_cal_fb++;
    }
    if (g_ef) v41_ef_set_expert(g_ef, J->e);
    /* 率侧探针选项: 任一开关(或 --stats-out)在才传, 否则 NULL = 现役路一个字不变 */
    v41_vq_rate rate = {0, 0, 0, 0, 0};
    uint16_t *const shcb = (J->grp >= 0 && J->grp < SHCB_MAX_GRP) ? g_shcb[J->grp] : NULL;
    const v41_vq_opt opt = {C->lam, shcb, C->cb_fp8, &rate};
    const int want = g_stats || C->lam > 0.f || shcb || C->cb_fp8;
    int rc = v41_vq_expert_from_fp4(w, s, rows, cols, 3, C->dim, nc, C->iters, stride, cwx, cwh, 0,
                                    g_ef ? v41_ef_callback : NULL, g_ef, want ? &opt : NULL, idx, cb, gain, st);
    if (rc) { fprintf(stderr, "★VQ 失败 %s rc=%d★\n", J->ew[0]->name, rc); return -1; }
    if (want) { g_H_sum += rate.H; g_H_n++; g_H_nc = nc; }
    if (g_cal && C->calib_ab) {   /* 机制审计对照: 同专家平权指派, 同一把加权尺记账, 不落盘 */
        uint8_t *idx2[3]; uint16_t *gain2[3]; double st2[4] = {0, 0, 0, 0};
        uint16_t *cb2 = (uint16_t *)malloc(sizeof(uint16_t) * nc * C->dim);
        for (int m = 0; m < 3; m++) { idx2[m] = (uint8_t *)malloc((size_t)rows[m] * ((cols[m] / C->dim * bits + 7) / 8)); gain2[m] = (uint16_t *)malloc(sizeof(uint16_t) * rows[m]); }
        const int rc2 = v41_vq_expert_from_fp4(w, s, rows, cols, 3, C->dim, nc, C->iters, stride, cwx, cwh, 1, NULL, NULL, NULL, idx2, cb2, gain2, st2);
        for (int m = 0; m < 3; m++) { free(idx2[m]); free(gain2[m]); }
        free(cb2);
        if (rc2) { fprintf(stderr, "★VQ 对照失败 %s rc=%d★\n", J->ew[0]->name, rc2); return -1; }
        g_ab.sse += st2[0]; g_ab.en += st2[1]; g_ab.wsse += st2[2]; g_ab.wen += st2[3];
    }
    if (v41_stw_write(W, J->t[0], cb, (uint64_t)nc * C->dim * 2)) return -1;
    for (int m = 0; m < 3; m++) {
        uint64_t nb = (uint64_t)rows[m] * ((cols[m] / C->dim * bits + 7) / 8);
        if (v41_stw_write(W, J->t[1 + m * 2], idx[m], nb) || v41_stw_write(W, J->t[2 + m * 2], gain[m], (uint64_t)rows[m] * 2)) return -1;
        avq->bytes += nb + (uint64_t)rows[m] * 2; avq->params += (long long)rows[m] * cols[m];
        free(idx[m]); free(gain[m]);
    }
    avq->bytes += (uint64_t)nc * C->dim * 2;
    avq->sse += st[0]; avq->en += st[1]; avq->wsse += st[2]; avq->wen += st[3];
    if (g_stats) { fprintf(g_stats, "%d %d %d %.9g %.9g %.4f %.0f %.4f %.3f %.3f\n", g_stats_layer, J->e, nc, st[0], st[1],
                           rate.H, rate.used, rate.mass_half, rate.w16, rate.w32); fflush(g_stats); }
    free(cb);
    return 0;
}

static const char *base_of(const char *p) { const char *s = strrchr(p, '/'); return s ? s + 1 : p; }
static const char *meta_json(const cfg_t *C) {
    static char m[720]; char al[32] = "", nt[160] = "", rt[64] = "";
    if (C->calib && C->alpha != 1.f) snprintf(al, sizeof al, " alpha=%g", C->alpha);
    /* 位宽配方随产物走(事后辨认这份模型是哪张表/哪几段量的) */
    if (g_nct_path) snprintf(nt, sizeof nt, " nctab=%s", base_of(g_nct_path));
    else if (g_nncl) { int p = 0; for (int k = 0; k < g_nncl && p < (int)sizeof nt - 24; k++)
        p += snprintf(nt + p, sizeof nt - p, " nc%d=%d:%d", g_ncl[k].nc, g_ncl[k].a, g_ncl[k].b); }
    if (C->lam > 0.f || C->shared_cb || C->cb_fp8)   /* 率侧探针配方随产物走 */
        snprintf(rt, sizeof rt, "%s%s%s", C->lam > 0.f ? " ecvq" : "", C->shared_cb ? " sharedcb" : "", C->cb_fp8 ? " cbfp8" : "");
    snprintf(m, sizeof m, "\"format\":\"pt\",\"ds4_v41_recipe\":\"routed=vq%dx%d skeleton=%s mtp=%s engram=hf_sidecar %s%s%s%s%s%s\","
             "\"vq_dim\":\"%d\",\"vq_nc\":\"%d\",\"vq_iters\":\"%d\",\"vq_stride\":\"%d\",\"vq_ecvq\":\"%g\"",
             C->dim, C->nc, C->skel_q4k ? "q4_K" : "fp4_1x32", C->mtp_vq ? "vq" : "fp4_passthrough",
             C->calib ? "cal=" : "nocal", C->calib ? base_of(C->calib) : "", al, C->ef_nb ? " ef" : "", nt, rt,
             C->dim, C->nc, C->iters, C->stride, C->lam);
    return m;
}
static double cosv(const acc_t *a) { return a->en > 0 ? sqrt(fmax(0.0, 1.0 - a->sse / a->en)) : 1.0; }
static double cosw(const acc_t *a) { return a->wen > 0 ? sqrt(fmax(0.0, 1.0 - a->wsse / a->wen)) : 1.0; }

/* 一个分片(某一层, 或 L=-1 的 common) */
static int do_shard(v41_st *S, const cfg_t *C, const char *out, int L, int nexp) {
    char path[4200];
    if (L >= 0) snprintf(path, sizeof path, "%s/model-layer%02d.safetensors", out, L);
    else snprintf(path, sizeof path, "%s/model-common.safetensors", out);
    struct stat sb;
    if (!C->force && !stat(path, &sb)) { printf("[跳过] %s 已完成 (%.2f GB)\n", path, sb.st_size / 1e9); return 0; }
    double t0 = now_s();
    g_stats_layer = L;
    v41_stw W; if (v41_stw_begin(&W, path)) return -1;
    /* 校准料只对主干层(common 里的三塔专家有自己的路由, 没取过料 ⇒ 平权, 与今天同) */
    g_cal = NULL; g_cal_nh = g_cal_fb = 0; memset(&g_ab, 0, sizeof g_ab);
    g_ef = NULL;
    if (C->calib && L >= 0) {
        char err[256];
        g_cal = v41_calib_open(C->calib, L, nexp, err, sizeof err);
        if (!g_cal) { fprintf(stderr, "★校准料 L%02d: %s★\n", L, err); v41_stw_abort(&W); return -1; }
        if (C->ef_nb > 0 && !(g_ef = v41_ef_open(g_cal, C->ef_betas, C->ef_nb))) { fprintf(stderr, "★级 2 L%02d 初始化失败★\n", L); v41_stw_abort(&W); return -1; }
    }
    /* job 上限 = 本分片最多的专家数(主干 384; common 里三塔合计 3×128) + 其余张量余量 */
    int nvq_plan = 0;
    job_t *jobs = (job_t *)calloc((size_t)nexp + 8192, sizeof(job_t)); int nj = 0;
    char pre[32]; int plen = 0;
    if (L >= 0) {
        plen = snprintf(pre, sizeof pre, "layers.%d.", L);
        char p2[32]; snprintf(p2, sizeof p2, "layers.%d", L);
        for (int e = 0; e < nexp; e++) {
            const int nc = g_nct ? g_nct[(size_t)L * nexp + e] : C->nc;
            if (plan_vq(&W, &jobs[nj++], S, p2, e, C, nc, 0)) return -1;
            nvq_plan++;
        }
    } else if (C->mtp_vq) {
        /* MTP 三塔的专家与主干同款 VQ(格式完全相同, 只是前缀不同)。塔数/专家数从张量名探得。
         * 共享码本按塔分组(三塔路由与主干不同, 混池没依据; 代价只是多两本码本)。 */
        int ntower = 0;
        for (int t = 0; t < 64; t++) {
            char p2[32]; snprintf(p2, sizeof p2, "mtp.%d", t);
            char probe[128]; snprintf(probe, sizeof probe, "%s.ffn.experts.0.w1.weight", p2);
            if (!v41_st_find(S, probe)) continue;
            int ne = 0;
            for (;; ne++) { snprintf(probe, sizeof probe, "%s.ffn.experts.%d.w1.weight", p2, ne); if (!v41_st_find(S, probe)) break; }
            printf("  [mtp] 塔 %d: %d 个专家 → VQ dim%d×nc%d\n", t, ne, C->dim, C->nc);
            if (ntower >= SHCB_MAX_GRP) { fprintf(stderr, "★塔数超过共享码本组上限 %d★\n", SHCB_MAX_GRP); return -1; }
            for (int e = 0; e < ne; e++) {
                if (nj >= nexp + 8192) { fprintf(stderr, "★job 表满★\n"); return -1; }
                if (plan_vq(&W, &jobs[nj++], S, p2, e, C, C->nc, ntower)) return -1;
                nvq_plan++;
            }
            ntower++;
        }
    }
    for (int i = 0; i < S->ne; i++) {
        const v41_st_ent *E = &S->e[i];
        if (L >= 0 ? strncmp(E->name, pre, (size_t)plen) != 0 : !strncmp(E->name, "layers.", 7)) continue;
        if (strstr(E->name, ".engram.") || ends_with(E->name, ".scale")) continue;
        if (L >= 0 && is_routed_expert(E->name)) continue;           /* 已在 VQ 计划里 */
        if (L < 0 && C->mtp_vq && is_mtp_expert(E->name)) continue;  /* 同上(三塔) */
        if (nj >= nexp + 8192) { fprintf(stderr, "★job 表满★\n"); return -1; }
        /* 三塔专家在 --mtp-vq 关掉时按出厂 FP4 原样透传 —— 不重量化(它已经是 FP4) */
        if (wants_fp4(E) && !is_mtp_expert(E->name)) {
            /* q4_K 要 256 的倍数列; 不满足的那张退回 FP4 而不是悄悄透传(体积账里看得见) */
            int q4k_ok = C->skel_q4k && E->shape[1] % 256 == 0;
            if (q4k_ok ? plan_q4k(&W, &jobs[nj++], S, E) : plan_fp4(&W, &jobs[nj++], S, E)) return -1;
        } else plan_pass(&W, &jobs[nj++], S, E);
    }
    if (v41_stw_write_header(&W, meta_json(C))) return -1;
    g_H_sum = 0; g_H_n = 0;
    if (C->shared_cb) {   /* 一组一本: 主干层一组, common 里 MTP 三塔各一组(见 v41_shcb.inc.c) */
        int ng = 0;
        for (int j = 0; j < nj; j++) if (jobs[j].kind == J_VQ && jobs[j].grp + 1 > ng) ng = jobs[j].grp + 1;
        for (int g = 0; g < ng; g++) {
            char tag[48];
            if (L >= 0) snprintf(tag, sizeof tag, "L%02d", L); else snprintf(tag, sizeof tag, "mtp.%d", g);
            if (shcb_build_group(S, C, jobs, nj, g, tag, t0)) { v41_stw_abort(&W); return -1; }
        }
        v41_st_release_idle(S);
    }
    acc_t avq = {0}, afp4 = {0}, aq4k = {0};
    int nvq = 0;
    for (int j = 0; j < nj; j++) {
        if (run_job(&W, S, &jobs[j], C, &avq, &afp4, &aq4k)) { v41_stw_abort(&W); return -1; }
        if (jobs[j].kind == J_VQ && (++nvq % 64 == 0 || nvq == nvq_plan))
            printf("  %s 专家 %3d/%d  %6.0fs  VQ 残差 %.2f%% (cos %.4f) 加权 cos %.4f\n", L >= 0 ? pre : "common", nvq, nvq_plan,
                   now_s() - t0, 100.0 * avq.sse / fmax(avq.en, 1e-30), cosv(&avq), cosw(&avq));
    }
    if (v41_stw_end(&W)) return -1;
    shcb_free();
    if (g_H_n) printf("  [率] %s 索引熵均值 %.4f bit(等概率 %d)%s%s\n", L >= 0 ? pre : "common", g_H_sum / g_H_n, bits_of(g_H_nc),
                      C->lam > 0.f ? " ECVQ" : "", C->cb_fp8 ? " 码本E4M3" : "");
    if (g_cal) {
        printf("  [校准] %s w2 列权 逐专家 %d / 退回平权 %d; 校准指派 cos %.4f 加权 cos %.4f", pre, g_cal_nh, g_cal_fb, cosv(&avq), cosw(&avq));
        if (C->calib_ab) printf(" | 平权指派对照 cos %.4f 加权 cos %.4f ⇒ 加权尺上校准%s平权", cosv(&g_ab), cosw(&g_ab), cosw(&avq) > cosw(&g_ab) ? "赢" : "★没赢★");
        if (g_ef) { char rb[256]; v41_ef_report(g_ef, rb, sizeof rb); printf(" | %s", rb); v41_ef_close(g_ef); g_ef = NULL; }
        printf("\n");
        v41_calib_close(g_cal); g_cal = NULL;
    }
    g_vq.sse += avq.sse; g_vq.en += avq.en; g_vq.wsse += avq.wsse; g_vq.wen += avq.wen; g_vq.bytes += avq.bytes; g_vq.params += avq.params;
    g_fp4.sse += afp4.sse; g_fp4.en += afp4.en; g_fp4.bytes += afp4.bytes; g_fp4.params += afp4.params;
    g_q4k.sse += aq4k.sse; g_q4k.en += aq4k.en; g_q4k.bytes += aq4k.bytes; g_q4k.params += aq4k.params;
    printf("[分片] %s  %.3f GB  %.0fs  | VQ %.3f B 参数 cos %.4f 加权 cos %.4f %.4f bpw | FP4 %.3f B cos %.4f | q4_K %.3f B cos %.4f\n",
           path + strlen(out) + 1, W.total / 1e9, now_s() - t0, avq.params / 1e9, cosv(&avq), cosw(&avq),
           avq.params ? avq.bytes * 8.0 / avq.params : 0.0, afp4.params / 1e9, cosv(&afp4),
           aq4k.params / 1e9, cosv(&aq4k));
    if (g_nct && L >= 0) {   /* 动态位宽针: 本层位宽直方图, 与 v41_nc_alloc 打的账对得上才算表真进了产物 */
        int hist[17] = {0}; for (int e = 0; e < nexp; e++) hist[bits_of(g_nct[(size_t)L * nexp + e])]++;
        printf("  [位宽表] L%02d:", L); for (int k = 1; k <= 16; k++) if (hist[k]) printf(" %d位×%d", k, hist[k]); printf("\n");
    }
    fflush(stdout);
    free(jobs);
    v41_st_release_idle(S);
    return 0;
}

/* 命令行解析与位宽表装填(要 cfg_t 与 bits_of/g_nct, 所以在它们之后 include) */
#include "v41_quantize_cli.inc.c"

/* index.json 按【盘上实际文件】生成(含软链进来的 engram 分片), 续跑/重跑都对得上 */
static int write_index(const char *out) {
    v41_st O; if (v41_st_open(&O, out)) return -1;
    char p[4200]; snprintf(p, sizeof p, "%s/model.safetensors.index.json", out);
    FILE *f = fopen(p, "w"); if (!f) return -1;
    uint64_t tot = 0; for (int i = 0; i < O.ne; i++) tot += O.e[i].nbytes;
    fprintf(f, "{\"metadata\":{\"total_size\":%llu},\"weight_map\":{", (unsigned long long)tot);
    for (int i = 0; i < O.ne; i++) fprintf(f, "%s\"%s\":\"%s\"", i ? "," : "", O.e[i].name, O.sh[O.e[i].shard].file);
    fprintf(f, "}}\n"); fclose(f);
    printf("[index] %d 个张量 / %d 个文件 / 总字节 %.3f GB → %s\n", O.ne, O.nsh, tot / 1e9, p);
    v41_st_close(&O);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { qcli_usage(); return 2; }
    cfg_t C = {8, 4096, 8, 8, 0, -1, 1, 0, 0, 0, NULL, 0, {0, 0, 0, 0}, 0, 1.f, 0.f, 0, 0};
    { const int rc = qcli_parse(argc, argv, &C); if (rc) return rc; }
    v41_calib_set_alpha(C.alpha);
    setvbuf(stdout, NULL, _IOLBF, 0);
    v41_st S; double t0 = now_s();
    if (v41_st_open(&S, argv[1])) return 1;
    int nlayers = 0, nexp = 0;
    for (int i = 0; i < S.ne; i++) {
        int L, e;
        if (sscanf(S.e[i].name, "layers.%d.ffn.experts.%d.w1.weight", &L, &e) == 2) { if (L + 1 > nlayers) nlayers = L + 1; if (L == 0 && e + 1 > nexp) nexp = e + 1; }
        else if (sscanf(S.e[i].name, "layers.%d.", &L) == 1 && L + 1 > nlayers) nlayers = L + 1;
    }
    if (C.l1 < 0) C.l1 = nlayers;
    { const int rc = qcli_nctab(&C, nlayers, nexp); if (rc) return rc; }
    printf("[索引] %d 个张量 / %d 分片, %.1fs; %d 层 × %d 专家; 配方 VQ dim%d×nc%d(%d bit) iters%d stride%d, 骨架 %s, 三塔 %s, 层 [%d,%d)%s, 校准 %s%s\n",
           S.ne, S.nsh, now_s() - t0, nlayers, nexp, C.dim, C.nc, bits_of(C.nc), C.iters, C.stride,
           C.skel_q4k ? "q4_K 4.5bpw" : "FP4 4.25bpw", C.mtp_vq ? "VQ" : "FP4 透传", C.l0, C.l1, C.common ? " + common" : "",
           C.calib ? C.calib : "无(零语料)", C.calib_ab ? " + 平权对照审计" : "");
    if (mkdir(argv[2], 0775) && errno != EEXIST) { fprintf(stderr, "★建不了 %s★\n", argv[2]); return 1; }
    for (int L = C.l0; L < C.l1; L++) if (do_shard(&S, &C, argv[2], L, nexp)) { fprintf(stderr, "★第 %d 层失败, 停车★\n", L); return 1; }
    if (C.common && do_shard(&S, &C, argv[2], -1, nexp)) return 1;
    v41_st_close(&S);
    if (write_index(argv[2])) return 1;
    if (C.calib) {   /* 校准来源随产物走(事后辨认这份模型用了哪份料): 取料程序的 calib.txt 一行 + 目录 */
        char src[4300], dst[4300], line[1024] = "";
        snprintf(src, sizeof src, "%s/calib.txt", C.calib); snprintf(dst, sizeof dst, "%s/calib.txt", argv[2]);
        FILE *fi = fopen(src, "r"); if (fi) { if (!fgets(line, sizeof line, fi)) line[0] = 0; fclose(fi); }
        FILE *fo = fopen(dst, "w"); if (fo) { fprintf(fo, "calib_dir=%s\n%salpha=%g ef=%d\n", C.calib, line, C.alpha, C.ef_nb); fclose(fo); }
    }
    printf("[总账] routed VQ: %.3f B 参数 → %.3f GB (%.4f bpw) cos %.4f 加权 cos %.4f\n", g_vq.params / 1e9, g_vq.bytes / 1e9,
           g_vq.params ? g_vq.bytes * 8.0 / g_vq.params : 0.0, cosv(&g_vq), cosw(&g_vq));
    if (g_fp4.params) printf("[总账] 骨架 FP4: %.3f B 参数 → %.3f GB (4.25 bpw) cos %.4f\n", g_fp4.params / 1e9, g_fp4.bytes / 1e9, cosv(&g_fp4));
    if (g_q4k.params) printf("[总账] 骨架 q4_K: %.3f B 参数 → %.3f GB (4.50 bpw) cos %.4f; 编码在主机累计 %.0fs\n",
                             g_q4k.params / 1e9, g_q4k.bytes / 1e9, cosv(&g_q4k), g_q4k_sec);
    printf("[总账] 透传: %.3f B 参数 → %.3f GB\n", g_pass.params / 1e9, g_pass.bytes / 1e9);
    printf("[总账] 本程序写出 %.3f GB (engram 203 GB 软链不计), 用时 %.0fs\n",
           (g_vq.bytes + g_fp4.bytes + g_q4k.bytes + g_pass.bytes) / 1e9, now_s() - t0);
    return 0;
}
