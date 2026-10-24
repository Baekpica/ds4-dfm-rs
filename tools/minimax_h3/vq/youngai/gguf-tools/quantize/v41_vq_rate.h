/* v41_vq_rate.h — VQ 编码器的"率侧"选项(2026-09-21, 113 GB 方案探针: 索引熵 / ECVQ / 共享码本 / 码本 E4M3)。
 *
 * 【为什么有这个头】09-21 上午判定"当前格式下纯权重动态位宽 = 均匀 12 位", 剩下的钱只在三处:
 *   ①索引不是等概率的 —— 熵编码能省的比特数只有量了真实索引分布才知道;
 *   ②ECVQ(率约束指派: 距离 + λ×码长)在同样的平均码长下失真更低 —— 值几位也要量;
 *   ③码本跨专家共享(每层一本)省 3 GB, 亏多少质量要量。
 * 这里只定义选项结构, 三件事的核在 v41_vq_rate.inc.cu / v41_vq_pool.inc.cu, 入口在 v41_vq.cu。
 * 量化器(v41_quantize.c)与 CUDA 侧共用这一份, 字段顺序即契约。 */
#ifndef V41_VQ_RATE_H
#define V41_VQ_RATE_H
#include <stdint.h>

/* 率统计(对【最终全量指派】算, 即落盘那份索引): 单位见字段注释。 */
typedef struct {
    double H;          /* 索引经验熵, bit/索引(nc4096 等概率 = 12.0) */
    double used;       /* 被用到的码字数 */
    double mass_half;  /* 最常用的一半码字占的索引比例 */
    double w16, w32;   /* 嵌套码本分块定宽: 按频次排名后, 每 16/32 个连续索引一块取块内最大排名所需位宽的平均(bit/索引, 不含块标签) */
} v41_vq_rate;

typedef struct {
    float lam;                 /* ECVQ 率惩罚 λ(每 bit 换多少失真, 失真单位 = 单位 RMS 行上的 8 维平方误差); 0 = 纯 Lloyd(现役, 逐字节不变) */
    const uint16_t *cb_fixed;  /* 非 NULL: 用这本给定码本(f16 位型 [nc×dim]), 不训练只指派 —— 共享码本探针 */
    int cb_fp8;                /* 1: 码本落盘前舍到 E4M3 格点(仍以 f16 位型存, E4M3 ⊂ f16)。为的是 8192 词码本能塞进 64 KB shared */
    v41_vq_rate *rate;         /* 非 NULL: 回填率统计 */
} v41_vq_opt;
#endif
