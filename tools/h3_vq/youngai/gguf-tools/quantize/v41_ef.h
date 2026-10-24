/* v41_ef.h — 级 2: 误差反馈重指派(GPTQ/GPTVQ 同式)的量化器接口(2026-09-21 凌晨)。实现 v41_ef.cu, 只进 v41_quantize。
 * 用法(v41_quantize.c): 每层 v41_ef_open(校准料, β 表) → 每专家 v41_ef_set_expert(e) → 把 v41_ef_callback 递给
 * v41_vq_expert_from_fp4 的 ef 钩子(编码器在【码本与行增益定死之后、打包之前】按矩阵回调, 钩子只改索引) → 层末 v41_ef_close。 */
#ifndef V41_EF_H
#define V41_EF_H
#include <stdint.h>
#include "v41_calib.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct v41_ef v41_ef;
/* betas: x 侧 H 里层级 Gram 的权 β 候选(≤4 个), 每专家按 val 行输出误差二选一; nval_min = val 行不足时退回 betas[0] */
v41_ef *v41_ef_open(v41_calib *cal, const float *betas, int nbeta);
void v41_ef_set_expert(v41_ef *ef, int e);
/* 编码器钩子: m = 0/1/2(w1/w3/w2); V = 该矩阵【归一化后】的权重 [rows][cols] 行主序(设备, 可被改写); C = 码本 [nc][dim](设备 f32, 已舍到 f16 格点);
 * g = 行增益 [rows](设备); idx = 索引 [rows][cols/dim](设备, 进来是级 1 的指派, 出去是重指派)。返回 0 = 已重指派, 1 = 本专家跳过(料太薄), <0 = 出错 */
int v41_ef_callback(void *ud, int m, float *V, int rows, int cols, const float *C, int nc, int dim, const float *g, int *idx);
/* 层账: 做了几个专家 / 跳过几个 / 各 β 被选次数 / val 输出误差 级2÷级1 的平均 */
void v41_ef_report(const v41_ef *ef, char *buf, size_t n);
void v41_ef_close(v41_ef *ef);
#ifdef __cplusplus
}
#endif
#endif
