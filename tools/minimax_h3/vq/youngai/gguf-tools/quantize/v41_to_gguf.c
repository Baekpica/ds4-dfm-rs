/* v41_to_gguf.c — V4.1 量化目录(v41_quantize 产出的 safetensors) → 引擎单 GGUF(2026-09-12, 战役 P0)。
 *
 * 【干什么】把 09-12 落盘的 1.5 bpw 文件(routed 专家 VQ dim8×nc4096 三件 + 骨架 FP4 1×32 + bf16 小张量)
 * 按引擎的 blk.N.* 命名与 GGUF 类型写成一个文件, 引擎才能 mmap 跑它。纯搬字节: 不重新量化,
 * 数值与 Python 判决过的文件逐位同 —— 这是后面"引擎 logits 对拍 Python 学生"判决成立的前提。
 *
 * 【映射】
 *   routed 专家 → blk.L.ffn_exps_vq.blob(type 42, DQVL ver 2: nexp 真值 384; 每矿阵 DQVQ 载荷 =
 *                 [头 16][码本 nc×dim f16][行增益 rows f16][12bit 索引位流(行字节对齐 = 连续流)][8 B 尾垫])
 *                 码本三矩阵共用一份, 但 DQVQ 每载荷自带 ⇒ 每专家写三份(+50 MB/层, 先换实现简单; 去重后置)
 *   FP4 1×32 权重(weight I8 + scale ue8m0) → type 43 fp4x32(16 B nibble + 1 B scale 交织块)
 *   bf16 小张量(norm/hc/gate/indexer wk/k_norm/proj) → f32;  f32 照抄(compressor wkv/wgate, sink, hc)
 *   engram wkv(FP8 E4M3 + 32×32 块 ue8m0) → f16(幅值可表示, 逐元素检查不溢出)
 *   engram 表(203 GB)不进文件: 元数据记真实分片路径 + 字节偏移, 引擎 mmap 按行读
 *   engram 哈希常量 ← v41_engram_consts.py 产物(EGRC), 作 I32/I64 张量嵌入
 *   vision / aligner / image_start|end|newline / gate.bias_vl 不要; mtp 三塔留 P5
 * 用法: v41_to_gguf <量化目录> <engram_consts.bin> <out.gguf> */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include "../../src/common/ds4_st41.h"
#include "../../src/common/ds4_gguf_write.h"
#include "../../src/common/ds4_quantfmt.h"
#include "../../src/common/ds4_fp8.h"
#include "../../src/common/ds4_float.h"
#include "../../vq_fmt.h"   /* DQVL/DQVQ 的魔数与 ver 白名单: 格式常量全仓只有这一份(引擎也读它) */
#include "v41_cfg.h"

enum { J_F32_BF16, J_F32_COPY, J_BF16_COPY, J_FP8BLK, J_FP4X32, J_Q4K, J_VQBLOB, J_VQBLOB3, J_RAW };
typedef struct { int kind; char src[192]; int layer; const void *raw; int s2; int nexp; } job_t;
static job_t *g_jobs; static int g_nj, g_cj;
static v41_st S; static v41_cfg C; static ds4gw W;
/* ★第二个源: 原始 HF 目录(2026-09-17, mtp-1.md 的 M6 真因)★
 * DSpark 三塔在**原始 HF 里是 FP8**(E4M3 + 32×32 块缩放), 而我们吃的量化目录里它们已经被量化器
 * 连同主干骨架一起压成了 FP4(I8 + 1×32) —— 于是"三塔 FP4 全透传"实际透传的是一份被压过的草稿器。
 * 实撞的代价: 草稿器对**它本来就该对齐的 FP 原模型**的首位一致率只有 0.5023(gguf-tools/bench/dspark_agree),
 * 最容易的那一档(FP 锚 top1−top2 ≥ 6)也只有 0.814, 而底座是 0.977 —— 信息就在它的输入里, 它却答不出来。
 * 三塔骨架总共才 ~260 MB, 还原成 FP8 只多 ~240 MB, 这笔精度不该省。
 * ★只有 mtp.* 走这个源★: 主干是有意量化的, 那是产品本身。 */
static v41_st S2; static int g_has_s2 = 0, g_plan_s2 = 0;
static uint64_t g_out_bytes;
static const char *g_out;   /* 输出路径(写出循环打日志用; 两种模式共用) */

static void die(const char *m) { fprintf(stderr, "★%s★\n", m); exit(1); }
static const v41_st_ent *need_in(const char *name, int s2) {
    const v41_st_ent *e = v41_st_find(s2 ? &S2 : &S, name);
    if (!e) { fprintf(stderr, "★缺张量 %s(源: %s)★\n", name, s2 ? "原始 HF" : "量化目录"); exit(1); }
    return e;
}
static const v41_st_ent *need(const char *name) { return need_in(name, g_plan_s2); }
static const uint8_t *sdata(const v41_st_ent *e, int s2) { return v41_st_data(s2 ? &S2 : &S, e); }
static int add_job(int kind, const char *gname, uint32_t type, uint32_t nd, const uint64_t *ne, uint64_t nbytes, const char *src, int layer, const void *raw) {
    if (g_nj == g_cj) { g_cj = g_cj ? g_cj * 2 : 4096; g_jobs = (job_t *)realloc(g_jobs, sizeof(job_t) * g_cj); }
    job_t *j = &g_jobs[g_nj++];
    memset(j, 0, sizeof *j); j->kind = kind; j->layer = layer; j->raw = raw;
    if (src) snprintf(j->src, sizeof j->src, "%s", src);
    j->s2 = g_plan_s2;
    return ds4gw_tensor(&W, gname, type, nd, ne, nbytes);
}

/* ---- 登记: 按源张量 dtype 决定目标类型 ---- */
static void plan_small(const char *gname, const char *src) {            /* bf16/f32 小张量 → f32 */
    const v41_st_ent *e = need(src);
    uint64_t ne[4]; for (int i = 0; i < e->nd; i++) ne[i] = (uint64_t)e->shape[e->nd - 1 - i];
    int kind = !strcmp(e->dtype, "BF16") ? J_F32_BF16 : !strcmp(e->dtype, "F32") ? J_F32_COPY : -1;
    if (kind < 0) { fprintf(stderr, "★%s dtype %s 不是 BF16/F32★\n", src, e->dtype); exit(1); }
    add_job(kind, gname, DS4_GGT_F32, (uint32_t)e->nd, ne, (uint64_t)v41_st_numel(e) * 4, src, -1, NULL);
}
/* ★2026-09-15 clear.md C1: BF16 原样存, 不再展开成 f32★
 * 这几个矩阵(路由 gate / compressor / indexer 投影)在 HF 原件里本来就是 BF16, 是我们的转换器
 * 把它们展成 f32 才让每 token 多读了 0.20 GB。存回 BF16 = **拿回官方精度**, 逐位无损, 字节减半。
 * 为什么不顺手压 FP4: 这几个矩阵决定"选哪 6 个专家"和"哪些位置进候选池", 位宽一动行为就跳 ——
 * 要压得单独过五指标门(clear.md C1 的 A/B 项), 不许搭这一刀的便车。 */
static void plan_bf16(const char *gname, const char *src) {
    const v41_st_ent *e = need(src);
    if (strcmp(e->dtype, "BF16")) { fprintf(stderr, "★%s dtype %s 不是 BF16★\n", src, e->dtype); exit(1); }
    uint64_t ne[4]; for (int i = 0; i < e->nd; i++) ne[i] = (uint64_t)e->shape[e->nd - 1 - i];
    add_job(J_BF16_COPY, gname, DS4_GGT_BF16, (uint32_t)e->nd, ne, (uint64_t)v41_st_numel(e) * 2, src, -1, NULL);
}
/* 骨架一张矩阵 → GGUF。★按【盘上登记的 dtype】分流, 不按配方分流★ ——
 * 量化器 --skel fp4 出 I8+scale(FP4 1×32), --skel q4k 出 dtype "Q4_K"(144 B/256 块, 自带
 * f16 主 scale/min, 没有 sibling .scale)。转换器不需要知道跑的是哪个配方, 也就不会出现
 * "配方换了、转换器没跟着换"这种错配(它只会缺张量或类型不认, 两种都是硬停车, 不是假数)。 */
static void plan_fp4(const char *gname, const char *src) {
    const v41_st_ent *e = need(src);
    if (e->nd != 2) { fprintf(stderr, "★%s 不是二维★\n", src); exit(1); }
    uint64_t rows = (uint64_t)e->shape[0];
    if (!strcmp(e->dtype, "Q4_K")) {                                   /* q4_K 块流, 原样搬 */
        uint64_t cols = (uint64_t)e->shape[1];                          /* Q4_K 的 shape 记逻辑形状 */
        if (cols % 256) { fprintf(stderr, "★%s 列 %llu 非 256 倍★\n", src, (unsigned long long)cols); exit(1); }
        uint64_t ne[2] = {cols, rows};
        add_job(J_Q4K, gname, DS4_GGT_Q4_K, 2, ne, rows * (cols / 256) * 144, src, -1, NULL);
        return;
    }
    char sn[192]; snprintf(sn, sizeof sn, "%s", src); char *dot = strrchr(sn, '.'); strcpy(dot, ".scale"); need(sn);
    if (strcmp(e->dtype, "I8")) { fprintf(stderr, "★%s dtype %s 既不是 Q4_K 也不是 FP4 打包 I8★\n", src, e->dtype); exit(1); }
    uint64_t cols = (uint64_t)e->shape[1] * 2;
    if (cols % 32) { fprintf(stderr, "★%s 列 %llu 非 32 倍★\n", src, (unsigned long long)cols); exit(1); }
    uint64_t ne[2] = {cols, rows};
    add_job(J_FP4X32, gname, DS4_GGT_FP4X32, 2, ne, rows * cols / 32 * 17, src, -1, NULL);
}
/* ★2026-09-15 clear.md C1: engram wkv 原样存 FP8, 不再展开成 f16★
 * 这张表 6144×25600, 两层共 0.315 GB。原来转换器把 e4m3×块缩放算成 f16 落盘 —— 字节翻倍(0.63 GB),
 * **而且是有损的**: f16 只有 10 位尾数, e4m3 的 3 位尾数乘上 2 的幂之后不一定落在 f16 格点上,
 * 大值还可能溢出(老代码专门数了 over/nan 就是防这个)。原样存反而更准, 且每 token 少读 0.31 GB。
 * 布局: 先 rows×cols 个 e4m3 字节, 紧跟 ceil(rows/32)×ceil(cols/32) 个 ue8m0 缩放字节。 */
static void plan_fp8blk(const char *gname, const char *src) {
    const v41_st_ent *e = need(src);
    if (e->nd != 2 || strcmp(e->dtype, "F8_E4M3")) { fprintf(stderr, "★%s 不是 F8_E4M3 二维★\n", src); exit(1); }
    char sn[192]; snprintf(sn, sizeof sn, "%s", src); strcpy(strrchr(sn, '.'), ".scale");
    const v41_st_ent *se = need(sn);
    uint64_t rows = (uint64_t)e->shape[0], cols = (uint64_t)e->shape[1];
    if ((uint64_t)se->shape[0] != (rows + 31) / 32 || (uint64_t)se->shape[1] != (cols + 31) / 32)
        { fprintf(stderr, "★%s 32×32 scale 形状不配★\n", src); exit(1); }
    uint64_t ne[2] = {cols, rows};
    add_job(J_FP8BLK, gname, DS4_GGT_FP8_32X32, 2, ne, rows * cols + ((rows + 31) / 32) * ((cols + 31) / 32), src, -1, NULL);
}
/* pre = "layers.7" 或 "mtp.1" —— 主干 routed 与 DSpark 三塔的 VQ 产物格式完全相同(量化器那边也是
 * 一份 plan_vq 吃两处), 所以这里只把前缀参数化, 不复制第二份 blob 打包逻辑。
 * 索引位宽由码本大小定(nc=4096 ⇒ 12 bit, nc=2048 ⇒ 11): 写死位宽会在换配方时静默错位。 */
static uint64_t vq_payload_bytes(const char *pre, int e, const char *m, uint64_t *rows, uint64_t *cols) {
    char n[192];
    snprintf(n, sizeof n, "%s.ffn.experts.%d.vq.cb", pre, e); const v41_st_ent *cb = need(n);
    snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.idx", pre, e, m); const v41_st_ent *ix = need(n);
    snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.gain", pre, e, m); const v41_st_ent *g = need(n);
    uint64_t nc = (uint64_t)cb->shape[0], dim = (uint64_t)cb->shape[1];
    int nbit = 0; while ((1ull << nbit) < nc) nbit++;
    *rows = (uint64_t)ix->shape[0];
    *cols = (uint64_t)ix->shape[1] * 8 / (uint64_t)nbit * dim;
    if ((uint64_t)g->shape[0] != *rows) die("vq.gain 行数与 idx 不符");
    return 16 + nc * dim * 2 + *rows * 2 + (uint64_t)ix->shape[0] * (uint64_t)ix->shape[1] + 8;
}
static void plan_vqblob(const char *pre, int L, int nexp, const char *gname) {
    uint64_t total = 16 + (uint64_t)nexp * 3 * 8;
    static const char *mats[3] = {"w1", "w3", "w2"};        /* which: 0=w1 1=w3 2=w2 (vq_fmt.h) */
    for (int e = 0; e < nexp; e++) for (int w = 0; w < 3; w++) { uint64_t r, c; total += vq_payload_bytes(pre, e, mats[w], &r, &c); }
    uint64_t ne[1] = {total};
    int ti = add_job(J_VQBLOB, gname, DS4_GGT_VQBLOB, 1, ne, total, pre, L, NULL);
    (void)ti;
    g_jobs[g_nj - 1].nexp = nexp;
}
/* DQVL v3(码本一层一本 + E4M3 + 13 位拆位平面)的登记器, 定义在 v41_to_gguf_vq3.inc.c;
 * 目录里 384 份码本不逐字节同(老目录)时它自己回退到上面的 v2 —— 老目录转出来一个字节不变。 */
/* v3 的码本探测结果(384 份是否逐字节同 / 能否无损存 E4M3), recipe KV 与登记器共用同一份探测 —— 一件事只写一处 */
typedef struct { int shared, fp8, nc, dim; uint64_t cb_bytes; } vq3_cb;
static vq3_cb vq3_probe(const char *pre, int nexp);
static void plan_vqblob_v3(const char *pre, int L, int nexp, const char *gname);
/* 三塔专家在量化目录里是 VQ 三件(--mtp-vq 配方)还是不在(出厂 FP4, 09-17 起从原件取)? 按盘上实情分流。
 * ★探测名必须是 .vq.cb★: 码本是【每专家一份】所以名字里没有矩阵名, 而 idx/gain 是每矩阵一份, 叫
 * `...experts.0.w1.vq.idx`。09-19 实撞: 探 `experts.0.vq.idx`(少了 w1)永远找不到 ⇒ 静默走回原件 FP4 分支,
 * 不报错, 只是转出来的文件多 4.9 GB —— 是体积不对才暴露的。 */
static int mtp_has_vq(int T) {
    char n[192]; snprintf(n, sizeof n, "mtp.%d.ffn.experts.0.vq.cb", T);
    return v41_st_find(&S, n) != NULL;
}
/* deepseek4.recipe 按【盘上实情】写(2026-09-20): 原来写死 "vq8x4096 skeleton=fp4x32", 对 q4_K 骨架 / 11 位那几份
 * 就是假话, 而这个 KV 是事后辨认一份 GGUF 配方的唯一凭证。三个事实各读一张登记张量: 主干码本形状定 dim×nc,
 * embed 的 dtype 定骨架格式(与 plan_fp4 同一判据), 三塔码本在不在定三塔格式(与 plan_mtp 同一判据)。 */
static const char *g_qdir;   /* 量化目录(main 设): 校准来源看目录里有没有量化器落的 calib.txt */
static void recipe_str(char *buf, size_t n) {
    const v41_st_ent *cb = need_in("layers.0.ffn.experts.0.vq.cb", 0), *em = need_in("embed.weight", 0);
    char mtp[32] = "fp4_native", cal[160] = "nocal";
    if (mtp_has_vq(0)) { const v41_st_ent *mc = need_in("mtp.0.ffn.experts.0.vq.cb", 0); snprintf(mtp, sizeof mtp, "vq%dx%d", (int)mc->shape[1], (int)mc->shape[0]); }
    /* 校准量化(2026-09-20): 量化器把取料目录写进 <目录>/calib.txt 首行 "calib_dir=…", 这里只抄目录名 —— 没有就是零语料 */
    if (g_qdir) {
        char p[4300], line[1024]; snprintf(p, sizeof p, "%s/calib.txt", g_qdir);
        FILE *f = fopen(p, "r");
        if (f) {
            if (fgets(line, sizeof line, f) && !strncmp(line, "calib_dir=", 10)) {
                char *s = line + 10, *e = s + strcspn(s, "\n"); *e = 0;
                const char *b = strrchr(s, '/'); snprintf(cal, sizeof cal, "cal=%s", b ? b + 1 : s);
            }
            fclose(f);
        }
    }
    /* ★逐层混位宽(2026-09-21 方案 v3)★: L00 与末层的码本大小不同 ⇒ 写成 "vq8x8192/4096", 不然 recipe 又是假话。
     * 末层号从 config 的 n_layers 取(转换器本来就按它建层)。 */
    char rt[64] = "";
    { char n2[96]; snprintf(n2, sizeof n2, "layers.%d.ffn.experts.0.vq.cb", C.n_layers - 1);
      const v41_st_ent *cbl = v41_st_find(&S, n2);
      if (cbl && cbl->shape[0] != cb->shape[0]) snprintf(rt, sizeof rt, "/%d", (int)cbl->shape[0]); }
    /* 共享码本与 E4M3 是 blob 的事(目录里码本仍每专家一份), 按盘上实情探一次 */
    const vq3_cb ci = vq3_probe("layers.0", C.n_routed_experts);
    snprintf(buf, n, "routed=vq%dx%d%s%s%s skeleton=%s mtp=%s engram=sidecar %s (v41_quantize)",
             (int)cb->shape[1], (int)cb->shape[0], rt, ci.shared ? " sharedcb" : "", ci.shared && ci.fp8 ? " cbfp8" : "",
             strcmp(em->dtype, "Q4_K") ? "fp4x32" : "q4_K", mtp, cal);
}
static int in_list(const int *v, int n, int x) { for (int i = 0; i < n; i++) if (v[i] == x) return 1; return 0; }

static void plan_layer(int L) {
    char g[96], s[192];
#define SMALL(gf, sf) do { snprintf(g, sizeof g, gf, L); snprintf(s, sizeof s, sf, L); plan_small(g, s); } while (0)
#define BF16(gf, sf)  do { snprintf(g, sizeof g, gf, L); snprintf(s, sizeof s, sf, L); plan_bf16(g, s); } while (0)
#define FP4(gf, sf)   do { snprintf(g, sizeof g, gf, L); snprintf(s, sizeof s, sf, L); plan_fp4(g, s); } while (0)
    SMALL("blk.%d.hc_attn_fn.weight", "layers.%d.hc_attn_fn");
    SMALL("blk.%d.hc_attn_scale.weight", "layers.%d.hc_attn_scale");
    SMALL("blk.%d.hc_attn_base.weight", "layers.%d.hc_attn_base");
    SMALL("blk.%d.attn_norm.weight", "layers.%d.attn_norm.weight");
    FP4("blk.%d.attn_q_a.weight", "layers.%d.attn.wq_a.weight");
    SMALL("blk.%d.attn_q_a_norm.weight", "layers.%d.attn.q_norm.weight");
    FP4("blk.%d.attn_q_b.weight", "layers.%d.attn.wq_b.weight");
    FP4("blk.%d.attn_kv.weight", "layers.%d.attn.wkv.weight");
    SMALL("blk.%d.attn_kv_a_norm.weight", "layers.%d.attn.kv_norm.weight");
    SMALL("blk.%d.attn_sinks.weight", "layers.%d.attn.attn_sink");
    FP4("blk.%d.attn_output_a.weight", "layers.%d.attn.wo_a.weight");
    FP4("blk.%d.attn_output_b.weight", "layers.%d.attn.wo_b.weight");
    const int ratio = C.compress_ratios[L];
    if (in_list(C.kv_source_layers, C.n_kv_source, L)) {
        BF16("blk.%d.attn_compressor_kv.weight", "layers.%d.attn.compressor.wkv.weight");
        if (ratio > 1) BF16("blk.%d.attn_compressor_gate.weight", "layers.%d.attn.compressor.wgate.weight");
        SMALL("blk.%d.attn_compressor_norm.weight", "layers.%d.attn.compressor.norm.weight");
        BF16("blk.%d.indexer.wk.weight", "layers.%d.attn.indexer.wk.weight");
        SMALL("blk.%d.indexer.k_norm.weight", "layers.%d.attn.indexer.k_norm.weight");
    }
    if (in_list(C.index_source_layers, C.n_index_source, L)) {
        FP4("blk.%d.indexer.attn_q_b.weight", "layers.%d.attn.indexer.wq_b.weight");
        BF16("blk.%d.indexer.proj.weight", "layers.%d.attn.indexer.weights_proj.weight");
    }
    SMALL("blk.%d.hc_ffn_fn.weight", "layers.%d.hc_ffn_fn");
    SMALL("blk.%d.hc_ffn_scale.weight", "layers.%d.hc_ffn_scale");
    SMALL("blk.%d.hc_ffn_base.weight", "layers.%d.hc_ffn_base");
    SMALL("blk.%d.ffn_norm.weight", "layers.%d.ffn_norm.weight");
    BF16("blk.%d.ffn_gate_inp.weight", "layers.%d.ffn.gate.weight");
    SMALL("blk.%d.exp_probs_b.bias", "layers.%d.ffn.gate.bias");
    { char gn[96], pre[64]; snprintf(gn, sizeof gn, "blk.%d.ffn_exps_vq.blob", L); snprintf(pre, sizeof pre, "layers.%d", L); plan_vqblob_v3(pre, L, C.n_routed_experts, gn); }
    FP4("blk.%d.ffn_gate_shexp.weight", "layers.%d.ffn.shared_experts.w1.weight");
    FP4("blk.%d.ffn_up_shexp.weight", "layers.%d.ffn.shared_experts.w3.weight");
    FP4("blk.%d.ffn_down_shexp.weight", "layers.%d.ffn.shared_experts.w2.weight");
    if (in_list(C.engram_layer_ids, C.n_engram, L)) {
        snprintf(g, sizeof g, "blk.%d.engram_wkv.weight", L); snprintf(s, sizeof s, "layers.%d.engram.wkv.weight", L); plan_fp8blk(g, s);
        SMALL("blk.%d.engram_q.weight", "layers.%d.engram.q_weight");
        SMALL("blk.%d.engram_k.weight", "layers.%d.engram.k_weight");
    }
#undef SMALL
#undef BF16
#undef FP4
}

#include "v41_to_gguf_mtp.inc.c"   /* DSpark 三塔登记(拆出去只为守 500 行, 单 TU 语义不变) */

/* ---- 生成字节 ---- */
static uint8_t *g_buf; static uint64_t g_cap;
static uint8_t *buf(uint64_t n) { if (n > g_cap) { g_cap = n + (n >> 3); g_buf = (uint8_t *)realloc(g_buf, g_cap); if (!g_buf) die("内存不够"); } return g_buf; }

static uint64_t gen_small(const job_t *j) {
    const v41_st_ent *e = need_in(j->src, j->s2); const uint8_t *p = sdata(e, j->s2); if (!p) exit(1);
    uint64_t n = (uint64_t)v41_st_numel(e); float *o = (float *)buf(n * 4);
    if (j->kind == J_BF16_COPY) { memcpy(o, p, n * 2); return n * 2; }   /* 原样搬 2 字节, 一位都不动 */
    if (j->kind == J_F32_COPY) memcpy(o, p, n * 4);
    else for (uint64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, p + 2 * i, 2); o[i] = ds4_bf16_to_f32(h); }
    return n * 4;
}
static uint64_t gen_fp4(const job_t *j) {
    const v41_st_ent *e = need_in(j->src, j->s2); const uint8_t *w = sdata(e, j->s2);
    char sn[192]; snprintf(sn, sizeof sn, "%s", j->src); strcpy(strrchr(sn, '.'), ".scale");
    const v41_st_ent *se = need_in(sn, j->s2); const uint8_t *sc = sdata(se, j->s2);
    if (!w || !sc) exit(1);
    uint64_t rows = (uint64_t)e->shape[0], cb = (uint64_t)e->shape[1] / 16;   /* 每行块数 = cols/32 */
    if ((uint64_t)se->shape[0] != rows || (uint64_t)se->shape[1] != cb) { fprintf(stderr, "★%s scale 形状不配★\n", j->src); exit(1); }
    uint8_t *o = buf(rows * cb * 17);
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t b = 0; b < cb; b++) {
            uint8_t *d = o + (r * cb + b) * 17;
            memcpy(d, w + r * (cb * 16) + b * 16, 16);
            d[16] = sc[r * cb + b];
        }
    return rows * cb * 17;
}
static uint64_t gen_fp8blk(const job_t *j) {
    const v41_st_ent *e = need_in(j->src, j->s2); const uint8_t *w = sdata(e, j->s2);
    char sn[192]; snprintf(sn, sizeof sn, "%s", j->src); strcpy(strrchr(sn, '.'), ".scale");
    const v41_st_ent *se = need_in(sn, j->s2); const uint8_t *sc = sdata(se, j->s2);
    if (!w || !sc) exit(1);
    uint64_t rows = (uint64_t)e->shape[0], cols = (uint64_t)e->shape[1];
    uint64_t sr = (rows + 31) / 32, scn = (cols + 31) / 32, n = rows * cols + sr * scn;
    uint8_t *o = buf(n);
    memcpy(o, w, rows * cols);                 /* e4m3 平面, 一位不动 */
    memcpy(o + rows * cols, sc, sr * scn);     /* ue8m0 缩放平面 */
    return n;
}
static uint64_t gen_vqblob(const job_t *j) {
    const int L = j->layer, nexp = j->nexp;
    const char *pre = j->src;                       /* "layers.7" / "mtp.1", plan 期存下来的 */
    static const char *mats[3] = {"w1", "w3", "w2"};
    uint64_t total = 16 + (uint64_t)nexp * 3 * 8;
    for (int e = 0; e < nexp; e++) for (int w = 0; w < 3; w++) { uint64_t r, c; total += vq_payload_bytes(pre, e, mats[w], &r, &c); }
    uint8_t *o = buf(total); memset(o, 0, 16 + (uint64_t)nexp * 3 * 8);
    uint32_t hdr[4] = {DS4VQ_BLOB_MAGIC, 2u, (uint32_t)L, (uint32_t)nexp}; memcpy(o, hdr, 16);
    uint64_t *tab = (uint64_t *)(o + 16); uint64_t off = 16 + (uint64_t)nexp * 3 * 8;
    char n[192];
    for (int e = 0; e < nexp; e++) {
        snprintf(n, sizeof n, "%s.ffn.experts.%d.vq.cb", pre, e); const v41_st_ent *cb = need(n); const uint8_t *cbp = v41_st_data(&S, cb);
        for (int w = 0; w < 3; w++) {
            snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.idx", pre, e, mats[w]); const v41_st_ent *ix = need(n); const uint8_t *ixp = v41_st_data(&S, ix);
            snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.gain", pre, e, mats[w]); const v41_st_ent *g = need(n); const uint8_t *gp = v41_st_data(&S, g);
            if (!cbp || !ixp || !gp) exit(1);
            uint64_t rows, cols; vq_payload_bytes(pre, e, mats[w], &rows, &cols);
            uint16_t dim = (uint16_t)cb->shape[1], nc = (uint16_t)cb->shape[0];
            uint8_t *p = o + off; tab[e * 3 + w] = off;
            memcpy(p, &(uint32_t){DS4VQ_MAT_MAGIC}, 4); memcpy(p + 4, &dim, 2); memcpy(p + 6, &nc, 2);
            uint32_t r32 = (uint32_t)rows, c32 = (uint32_t)cols; memcpy(p + 8, &r32, 4); memcpy(p + 12, &c32, 4);
            p += 16; memcpy(p, cbp, (size_t)nc * dim * 2); p += (size_t)nc * dim * 2;
            memcpy(p, gp, rows * 2); p += rows * 2;
            uint64_t ixb = (uint64_t)ix->shape[0] * (uint64_t)ix->shape[1]; memcpy(p, ixp, ixb); p += ixb;
            memset(p, 0, 8); p += 8;
            off = (uint64_t)(p - o);
        }
    }
    if (off != total) die("vq blob 字节账不平");
    return total;
}
#include "v41_to_gguf_vq3.inc.c"   /* DQVL v3 的登记与字节生成(要 buf()/add_job/need_in, 所以在它们之后) */


#include "v41_to_gguf_tok.inc.c"      /* tokenizer 常量 → KV(拆出去只为守 500 行) */
#include "v41_to_gguf_vision.inc.c"   /* 视觉塔 + aligner 登记与元数据(2026-10-13; 取原始 HF, 要 plan_bf16/plan_small/g_plan_s2) */

int main(int argc, char **argv) {
    if (argc >= 5 && !strcmp(argv[1], "vision")) return main_vision(argv[2], argv[3], argv[4]);   /* 视觉侧车模式(v41_to_gguf_vision.inc.c) */
    if (argc < 5) {
        fprintf(stderr, "用法: v41_to_gguf <量化目录> <engram_consts.bin> <tokenizer_consts.bin> <out.gguf> [原始HF目录]\n"
                        "  第 5 个参数 = 原始 HF 目录, **带 DSpark 三塔时必给**: 三塔在原件里是 FP8/BF16,\n"
                        "  而量化目录里已经被量化器压成 FP4 —— 那是草稿器的精度, 不该跟主干一起压(见 plan_mtp 注释)。\n");
        return 2;
    }
    const char *qdir = argv[1], *egp = argv[2], *tkp = argv[3], *out = argv[4];
    g_qdir = qdir; g_out = out;
    const char *hfdir = argc > 5 ? argv[5] : NULL;
    if (v41_st_open(&S, qdir)) return 1;
    if (hfdir) {
        if (v41_st_open(&S2, hfdir)) { fprintf(stderr, "★打不开原始 HF 目录 %s★\n", hfdir); return 1; }
        g_has_s2 = 1;
        fprintf(stderr, "[源] 主干 ← %s; ★三塔 ← %s(原生精度)★\n", qdir, hfdir);
    }
    char cfgp[4200]; snprintf(cfgp, sizeof cfgp, "%s/inference/config.json", qdir); v41_cfg_load(&C, cfgp);
    v41_cfg_vision_load(&C, qdir);   /* 视觉塔超参在顶层 config.json 的 vision_config(2026-10-13) */
    /* ★主文件不带视觉塔★(2026-10-17): 10-13 用户定的是视觉塔走侧车(`v41_to_gguf vision …` 出 *.vision.gguf, 引擎 --vision-gguf 第二映射),
     * 主 GGUF 与 09-22 现役同口径(1000 张量)。这里不清零的话 plan_vision/emit_vision_kv 会把 306 个 BF16 张量(0.971 GB)连元数据一起塞进主文件
     * —— 10-17 量新底座时实撞: 同配方多出 0.97 GB, 查了半天才发现是这一步。清零后两个函数各自 return, 侧车模式 main_vision 不走这里不受影响。 */
    if (C.vis_layers) { fprintf(stderr, "[视觉] 主文件不带视觉塔(%d 块): 走侧车 `v41_to_gguf vision <量化目录> <原始HF> <out>.vision.gguf`\n", C.vis_layers); C.vis_layers = 0; }
    fprintf(stderr, "[配置] %d 层 dim %d 专家 %d top-%d 路由 %s ×%.2f; 压缩比 %d 项; kv源 %d 索引源 %d engram %d 层\n",
            C.n_layers, C.dim, C.n_routed_experts, C.n_activated_experts, C.score_func, C.route_scale, C.n_compress_ratios, C.n_kv_source, C.n_index_source, C.n_engram);
    /* engram 常量 */
    FILE *ef = fopen(egp, "rb"); if (!ef) die("打不开 engram_consts.bin");
    uint32_t eh[8]; if (fread(eh, 4, 8, ef) != 8 || eh[0] != 0x43524745u) die("engram_consts.bin 魔数不对(EGRC)");
    uint32_t ev = eh[1], evocab = eh[2], ecv = eh[3], enl = eh[4], eng = eh[5], enh = eh[6], epad = eh[7];
    if (ev != 1 || (int)evocab != C.vocab_size || (int)ecv != C.engram_compressed_vocab_size || (int)enl != C.n_engram || (int)eng != C.engram_max_ngram_size || (int)enh != C.engram_n_heads) die("engram_consts.bin 与 config 不配");
    uint64_t n_tm = evocab, n_mul = (uint64_t)enl * eng, n_pr = (uint64_t)enl * (eng - 1) * enh, n_off = (uint64_t)enl * (eng - 1) * enh;
    int32_t *tm = (int32_t *)malloc(n_tm * 4); int64_t *mul = (int64_t *)malloc(n_mul * 8), *pr = (int64_t *)malloc(n_pr * 8), *offs = (int64_t *)malloc(n_off * 8);
    if (fread(tm, 4, n_tm, ef) != n_tm || fread(mul, 8, n_mul, ef) != n_mul || fread(pr, 8, n_pr, ef) != n_pr || fread(offs, 8, n_off, ef) != n_off) die("engram_consts.bin 读不满");
    fclose(ef);

    if (ds4gw_begin(&W, out)) return 1;
    /* ---- 元数据: 引擎 core_validate 要的全套 deepseek4.* + V4.1 新键 ---- */
    ds4gw_kv_str(&W, "general.architecture", "deepseek4");
    ds4gw_kv_str(&W, "general.name", "DeepSeek V4.1 Flash");
    ds4gw_kv_u32(&W, "general.alignment", 32);
    ds4gw_kv_str(&W, "deepseek4.variant", "v41_flash");
    { char rc[128]; recipe_str(rc, sizeof rc); ds4gw_kv_str(&W, "deepseek4.recipe", rc); fprintf(stderr, "[配方] %s\n", rc); }
    emit_tokenizer(tkp);
    ds4gw_kv_u32(&W, "deepseek4.block_count", (uint32_t)C.n_layers);
    emit_vision_kv();   /* deepseek4.vision.* + image_token_id(没视觉塔不写) */
    ds4gw_kv_u64(&W, "deepseek4.context_length", v41_cfg_context_length(qdir));   /* 模型自己声明的, 见 v41_cfg.h */
    ds4gw_kv_u32(&W, "deepseek4.embedding_length", (uint32_t)C.dim);
    ds4gw_kv_u32(&W, "deepseek4.vocab_size", (uint32_t)C.vocab_size);
    ds4gw_kv_u32(&W, "deepseek4.attention.head_count", (uint32_t)C.n_heads);
    ds4gw_kv_u32(&W, "deepseek4.attention.head_count_kv", 1);
    ds4gw_kv_u32(&W, "deepseek4.attention.key_length", (uint32_t)C.head_dim);
    ds4gw_kv_u32(&W, "deepseek4.attention.value_length", (uint32_t)C.head_dim);
    ds4gw_kv_u32(&W, "deepseek4.rope.dimension_count", (uint32_t)C.rope_head_dim);
    ds4gw_kv_u32(&W, "deepseek4.attention.q_lora_rank", (uint32_t)C.q_lora_rank);
    ds4gw_kv_u32(&W, "deepseek4.attention.output_lora_rank", (uint32_t)C.o_lora_rank);
    ds4gw_kv_u32(&W, "deepseek4.attention.output_group_count", (uint32_t)C.o_groups);
    ds4gw_kv_u32(&W, "deepseek4.expert_count", (uint32_t)C.n_routed_experts);
    /* 引擎按元数据认三塔, 不写死数字(铁律: 引擎禁版本名/魔数硬编码) */
    ds4gw_kv_u32(&W, "deepseek4.mtp.tower_count", (uint32_t)C.n_mtp_layers);
    ds4gw_kv_u32(&W, "deepseek4.mtp.expert_count", (uint32_t)C.dspark_n_routed_experts);
    /* 草稿器跑起来还要这五件(speed.md 段 6 D1): 每塔选几个专家 / 一块出几位 / 占位 token 的 id /
     * markov 头的秩 / 主模型哪几层的注意力输入拼成 main_x。少一件引擎就只能猜, 所以全部落元数据。 */
    ds4gw_kv_u32(&W, "deepseek4.mtp.expert_used_count", (uint32_t)C.dspark_n_activated_experts);
    ds4gw_kv_u32(&W, "deepseek4.mtp.block_size", (uint32_t)C.dspark_block_size);
    ds4gw_kv_u32(&W, "deepseek4.mtp.noise_token_id", (uint32_t)C.dspark_noise_token_id);
    ds4gw_kv_u32(&W, "deepseek4.mtp.markov_rank", (uint32_t)C.dspark_markov_rank);
    ds4gw_kv_arr_i32(&W, "deepseek4.mtp.target_layers", C.dspark_target_layer_ids, (uint64_t)C.n_dspark_target);
    ds4gw_kv_u32(&W, "deepseek4.expert_used_count", (uint32_t)C.n_activated_experts);
    ds4gw_kv_u32(&W, "deepseek4.expert_feed_forward_length", (uint32_t)C.moe_inter_dim);
    ds4gw_kv_u32(&W, "deepseek4.expert_shared_count", (uint32_t)C.n_shared_experts);
    ds4gw_kv_u32(&W, "deepseek4.hash_layer_count", 0);
    ds4gw_kv_u32(&W, "deepseek4.attention.sliding_window", (uint32_t)C.window_size);
    ds4gw_kv_u32(&W, "deepseek4.attention.indexer.head_count", (uint32_t)C.index_n_heads);
    ds4gw_kv_u32(&W, "deepseek4.attention.indexer.key_length", (uint32_t)C.index_head_dim);
    ds4gw_kv_u32(&W, "deepseek4.attention.indexer.top_k", (uint32_t)C.index_topk);
    ds4gw_kv_u32(&W, "deepseek4.hyper_connection.count", (uint32_t)C.hc_mult);
    ds4gw_kv_u32(&W, "deepseek4.hyper_connection.sinkhorn_iterations", (uint32_t)C.hc_sinkhorn_iters);
    ds4gw_kv_f32(&W, "deepseek4.hyper_connection.epsilon", C.hc_eps);
    ds4gw_kv_f32(&W, "deepseek4.attention.layer_norm_rms_epsilon", C.norm_eps);
    ds4gw_kv_f32(&W, "deepseek4.expert_weights_scale", C.route_scale);
    ds4gw_kv_bool(&W, "deepseek4.expert_weights_norm", 1);
    ds4gw_kv_str(&W, "deepseek4.expert_gating_func", C.score_func);
    ds4gw_kv_f32(&W, "deepseek4.rope.freq_base", C.rope_theta);
    ds4gw_kv_f32(&W, "deepseek4.rope.scaling.factor", C.rope_factor);
    ds4gw_kv_f32(&W, "deepseek4.rope.scaling.yarn_beta_fast", (float)C.beta_fast);
    ds4gw_kv_f32(&W, "deepseek4.rope.scaling.yarn_beta_slow", (float)C.beta_slow);
    ds4gw_kv_u64(&W, "deepseek4.rope.scaling.original_context_length", (uint64_t)C.original_seq_len);
    ds4gw_kv_f32(&W, "deepseek4.attention.compress_rope_freq_base", C.compress_rope_theta);
    { int32_t cr[V41_MAX_LAYERS]; for (int i = 0; i < C.n_compress_ratios; i++) cr[i] = C.compress_ratios[i];
      ds4gw_kv_arr_i32(&W, "deepseek4.attention.compress_ratios", cr, (uint64_t)C.n_compress_ratios); }
    { float sw[V41_MAX_LAYERS]; for (int i = 0; i < C.n_layers; i++) sw[i] = C.swiglu_limit;
      ds4gw_kv_arr_f32(&W, "deepseek4.swiglu_clamp_exp", sw, (uint64_t)C.n_layers); }
    { int32_t v[V41_MAX_LIST]; for (int i = 0; i < C.n_kv_source; i++) v[i] = C.kv_source_layers[i];
      ds4gw_kv_arr_i32(&W, "deepseek4.attention.kv_source_layers", v, (uint64_t)C.n_kv_source);
      for (int i = 0; i < C.n_index_source; i++) v[i] = C.index_source_layers[i];
      ds4gw_kv_arr_i32(&W, "deepseek4.attention.index_source_layers", v, (uint64_t)C.n_index_source); }
    ds4gw_kv_u32(&W, "deepseek4.attention.candidate.source_layer", (uint32_t)C.candidate_source_layer);
    ds4gw_kv_u32(&W, "deepseek4.attention.candidate.topk_blocks", (uint32_t)C.candidate_topk_blocks);
    ds4gw_kv_u32(&W, "deepseek4.attention.candidate.block_size", (uint32_t)C.candidate_block_size);
    { int32_t v[V41_MAX_LIST]; for (int i = 0; i < C.n_engram; i++) v[i] = C.engram_layer_ids[i];
      ds4gw_kv_arr_i32(&W, "deepseek4.engram.layer_ids", v, (uint64_t)C.n_engram);
      uint64_t rows[V41_MAX_LIST]; for (int i = 0; i < C.n_engram; i++) rows[i] = (uint64_t)C.engram_num_embeddings[i];
      ds4gw_kv_arr_u64(&W, "deepseek4.engram.num_embeddings", rows, (uint64_t)C.n_engram); }
    ds4gw_kv_u32(&W, "deepseek4.engram.max_ngram_size", (uint32_t)C.engram_max_ngram_size);
    ds4gw_kv_u32(&W, "deepseek4.engram.head_count", (uint32_t)C.engram_n_heads);
    ds4gw_kv_u32(&W, "deepseek4.engram.head_dim", (uint32_t)C.engram_head_dim);
    ds4gw_kv_u32(&W, "deepseek4.engram.vocab_size", (uint32_t)C.engram_vocab_size);
    ds4gw_kv_u32(&W, "deepseek4.engram.compressed_vocab_size", (uint32_t)C.engram_compressed_vocab_size);
    ds4gw_kv_u32(&W, "deepseek4.engram.pad_id_compressed", epad);
    /* engram 表: 真实分片路径 + 绝对字节偏移(引擎 mmap 该文件, 行 i 的 256 B 权重在 weight_offset+i*256) */
    for (int i = 0; i < C.n_engram; i++) {
        char n[128], k[128]; int L = C.engram_layer_ids[i];
        snprintf(n, sizeof n, "layers.%d.engram.embed.weight", L); const v41_st_ent *ew = need(n);
        snprintf(n, sizeof n, "layers.%d.engram.embed.scale", L); const v41_st_ent *es = need(n);
        if (ew->shard != es->shard) die("engram 表 weight/scale 不在同一分片");
        char rp[PATH_MAX]; if (!realpath(S.sh[ew->shard].path, rp)) die("engram 分片 realpath 失败");
        if (ew->shape[0] != C.engram_num_embeddings[i] || ew->shape[1] != C.engram_head_dim) die("engram 表形状与 config 不配");
        snprintf(k, sizeof k, "deepseek4.engram.%d.table_path", i); ds4gw_kv_str(&W, k, rp);
        snprintf(k, sizeof k, "deepseek4.engram.%d.weight_offset", i); ds4gw_kv_u64(&W, k, S.sh[ew->shard].data0 + ew->off);
        snprintf(k, sizeof k, "deepseek4.engram.%d.scale_offset", i); ds4gw_kv_u64(&W, k, S.sh[es->shard].data0 + es->off);
        fprintf(stderr, "[engram] L%d 表 %lld×%lld @ %s +%llu\n", L, (long long)ew->shape[0], (long long)ew->shape[1], rp, (unsigned long long)(S.sh[ew->shard].data0 + ew->off));
    }
    /* ---- 张量登记 ---- */
    plan_fp4("token_embd.weight", "embed.weight");
    for (int L = 0; L < C.n_layers; L++) plan_layer(L);
    for (int T = 0; T < C.n_mtp_layers; T++) plan_mtp(T);   /* DSpark 三塔(段 6 S1) */
    plan_vision();   /* 视觉塔 + aligner + 各层 bias_vl(2026-10-13, 原件 BF16) */
    plan_small("output_norm.weight", "norm.weight");
    plan_fp4("output.weight", "head.weight");
    { uint64_t ne[3];
      ne[0] = n_tm; add_job(J_RAW, "engram.token_map", 26, 1, ne, n_tm * 4, NULL, -1, tm);
      ne[0] = eng; ne[1] = enl; add_job(J_RAW, "engram.multipliers", 27, 2, ne, n_mul * 8, NULL, -1, mul);
      ne[0] = enh; ne[1] = eng - 1; ne[2] = enl; add_job(J_RAW, "engram.primes", 27, 3, ne, n_pr * 8, NULL, -1, pr);
      ne[0] = (uint64_t)(eng - 1) * enh; ne[1] = enl; add_job(J_RAW, "engram.offsets", 27, 2, ne, n_off * 8, NULL, -1, offs); }
    fprintf(stderr, "[登记] %d 张量, 数据区 %.3f GB\n", W.nt, (double)W.total / 1e9);
    if (ds4gw_header(&W)) return 1;
    const int rc = write_jobs();
    v41_st_close(&S);
    return rc;
}

/* ---- 逐张量生成+写(主文件与视觉侧车两种模式共用) ---- */
static int write_jobs(void) {
    int last_layer = -1;
    for (int i = 0; i < g_nj; i++) {
        const job_t *j = &g_jobs[i]; uint64_t n = 0; const void *d = g_buf;
        switch (j->kind) {
            case J_F32_BF16: case J_F32_COPY: case J_BF16_COPY: n = gen_small(j); d = g_buf; break;
            case J_FP4X32: n = gen_fp4(j); d = g_buf; break;
            /* q4_K: 盘上块布局(144 B/256, 连续)与 GGUF 的 Q4_K 逐字节相同 ⇒ 原样搬, 不解不压。
             * 字节数由 plan 期按逻辑形状算死, 这里只核对源张量的实际长度对不对得上。 */
            case J_Q4K: {
                const v41_st_ent *e = need_in(j->src, j->s2);
                d = sdata(e, j->s2); if (!d) exit(1);
                n = e->nbytes;
                if (n != W.t[i].nbytes) die("q4_K 源字节数与登记不符");
                break;
            }
            case J_FP8BLK: n = gen_fp8blk(j); d = g_buf; break;
            case J_VQBLOB: n = gen_vqblob(j); d = g_buf; break;
            case J_VQBLOB3: n = gen_vqblob_v3(j); d = g_buf; break;
            case J_RAW: n = W.t[i].nbytes; d = j->raw; break;
        }
        if (ds4gw_write(&W, i, d, n)) return 1;
        g_out_bytes += n;
        if (j->kind == J_VQBLOB || j->kind == J_VQBLOB3) { fprintf(stderr, "[L%02d] 专家 blob %.3f GB, 累计 %.2f GB\n", j->layer, (double)n / 1e9, (double)g_out_bytes / 1e9); fflush(stderr); }
        if (j->layer >= 0 && j->layer != last_layer) { last_layer = j->layer; v41_st_release_idle(&S); }
    }
    if (ds4gw_end(&W)) return 1;
    fprintf(stderr, "[完成] %s: %d 张量, %.3f GB 数据(十进制)\n", g_out, W.nt, (double)g_out_bytes / 1e9);
    return 0;
}
