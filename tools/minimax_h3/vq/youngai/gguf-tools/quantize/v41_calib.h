/* v41_calib.h — 量化校准料的文件格式与量化器接口(2026-09-20 深夜, 用户令"按金融域量化")。
 * 写方 gguf-tools/amp/v41_calib_dump.inc.c(引擎在现役量化模型上一趟取料), 读方 quantize/v41_calib.cu(量化器)。
 * 字节口径只此一份: 两边都拿这里的偏移函数算位置, 改布局只改这里。
 *
 * 【文件】<dir>/calib_Lnn.bin = 头 32 B + x[ntok][D] bf16 + sel[ntok][n_used] i32 + rw[ntok][n_used] f32
 *   x   = 该层 MoE 入口(ffn_norm 出口, 引擎给的就是 bf16 格点 ⇒ 存 bf16 零损, 84 MB/层@8192)
 *   sel = 路由选中的专家号, rw = 对应路由权重(引擎 y = Σ_k rw_k·expert_k(x) + shared)
 * 【量化器怎么用】级 1 列权: w1/w3 吃层级 E[x²](全部 token 共用), w2 吃逐专家 E[h_e²](h 在出厂 FP 权重上算);
 * 级 2 误差反馈另从同一份料取逐专家 Gram。 */
#ifndef V41_CALIB_H
#define V41_CALIB_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { char magic[4]; int32_t ntok, D, n_used; float clamp; int32_t reserved[3]; } v41_calib_hdr;
#define V41_CALIB_MAGIC "V41C"
static inline size_t v41_calib_off_x(const v41_calib_hdr *h) { (void)h; return sizeof(v41_calib_hdr); }
static inline size_t v41_calib_off_sel(const v41_calib_hdr *h) { return sizeof(v41_calib_hdr) + (size_t)h->ntok * (size_t)h->D * 2u; }
static inline size_t v41_calib_off_rw(const v41_calib_hdr *h) { return v41_calib_off_sel(h) + (size_t)h->ntok * (size_t)h->n_used * 4u; }
static inline size_t v41_calib_size(const v41_calib_hdr *h) { return v41_calib_off_rw(h) + (size_t)h->ntok * (size_t)h->n_used * 4u; }

/* 量化器侧(v41_calib.cu, 全 GPU): 一层一个对象 */
typedef struct v41_calib v41_calib;
/* 列权指数(2026-09-21 01:41 实撞): 金融 E[x²] 全额当列权, 金融 j 只 +0.6 pp 而 wt2 −2.15 pp(域过拟合)。w ← w^α 压动态范围,
 * α=1 全额(默认), 0.5 = 开方(AWQ 那族的经验最优), 0 = 平权。在归一化之前作用, 对 x 侧与 h 侧列权同时生效。 */
void v41_calib_set_alpha(float alpha);
/* 读 <dir>/calib_L<L>.bin, 建逐专家行表, 算层级 E[x²] 列权(rw² 加权, 均值归一到 1); 顺手打一行自检。失败返回 NULL 并写 err */
v41_calib *v41_calib_open(const char *dir, int L, int n_expert, char *err, size_t errn);
const float *v41_calib_colw_x(const v41_calib *c);      /* 主机 [D] */
int v41_calib_rows(const v41_calib *c, int e);           /* 专家 e 名下的 (token,pick) 行数 */
/* 逐专家 w2 列权: 名下行在出厂 FP4 的 w1/w3(主机字节 + 每 32 列 ue8m0 scale)上算 h = silu(clamp(x·W1ᵀ))·clamp(x·W3ᵀ),
 * out[MID] = Σ rw²·h² 按列, 均值归一到 1。返回用到的行数; < 8 行不算(返回行数, out 不写 ⇒ 调用方退回平权); <0 = 出错 */
int v41_calib_colw_h(v41_calib *c, int e, const uint8_t *w1, const uint8_t *s1, const uint8_t *w3, const uint8_t *s3,
                     int MID, int D, float *out);
void v41_calib_close(v41_calib *c);

/* ---- 级 2(误差反馈, v41_ef.cu)要的料 ---- */
int v41_calib_D(const v41_calib *c);
float v41_calib_clamp(const v41_calib *c);
/* 专家 e 名下行上设备: *dx = [m][D] f32(bf16 原值), *drw2 = [m] rw², *dval = [m] 1=val 行(取料目录里有 calib.layout 才有 val,
 * 否则全 0); 指针到下一次调用前有效。返回 m(<0 出错) */
int v41_calib_expert_x(v41_calib *c, int e, const float **dx, const float **drw2, const float **dval);
/* 层级 Gram(x 侧, 全部 token, 权 Σ_k rw²): *dH = 设备 [D][D] f32(对称, 已除以权和 ⇒ 是 E[x xᵀ]), 首次调用现算并缓存。返回 0 */
int v41_calib_layer_gram(v41_calib *c, const float **dH);

#ifdef __cplusplus
}
#endif
#endif
