/* ds4_quantfmt.h — GGUF 量化块格式: 类型几何 + 标量 dequant 的全仓唯一实现。
 *
 * 来源: zlayer.c 的 gg_* 家族(2026-08-25 对 gguf-py 逐位过闸), 其自身谱系是
 * ds4.c deq_q2K_row_f32 / llama.cpp 标准式 / metal moe.metal 的 iq2xxs 解码表。
 * 金标: tests/fixtures/quantfmt/ 下每类型固定输入块 + 过闸版输出, 由
 * tests/unit/test_common.c 逐字节回归 —— 改动这里必须先过那道闸。
 *
 * ★iq2_xxs 双表陷阱(踩过一次, max|Δ|=15 不报错)★
 *   - ds4_iq2xxs_kgrid[256]: 2bit 打包网格(编码/解码共用同一张打包表)。
 *   - ds4_iq2xxs_val[4] = {0x08,0x19,0x2b,--}: 【解码】码→值映射
 *     (= llama.cpp iq2xxs_grid = gguf-py IQ2_XXS.grid_map = metal moe.metal:23)。
 *   - 编码器的搜索空间 2*l+1 ∈ {1,3,5,7}(gguf-tools/quants.c)【不是】解码表,
 *     两者同名不同数。消费方一律引用这里, 别再抄表。 */
#ifndef DS4_COMMON_QUANTFMT_H
#define DS4_COMMON_QUANTFMT_H

#include <stdint.h>
#include <stddef.h>

/* GGUF/GGML 张量类型号(线上格式, 数值不可改) */
enum {
    DS4_GGT_F32     = 0,
    DS4_GGT_F16     = 1,
    DS4_GGT_Q8_0    = 8,
    DS4_GGT_Q2_K    = 10,
    DS4_GGT_Q4_K    = 12,
    DS4_GGT_IQ2_XXS = 16,
    DS4_GGT_BF16    = 30,
    DS4_GGT_VQBLOB  = 42,   /* 本仓自定: VQ 层 blob(DQVL 表+DQVQ 载荷), 不透明字节, shape=[nbytes] */
    /* 本仓自定(2026-09-12, V4.1 骨架): FP4 E2M1 + 每 32 元素一个 ue8m0 指数 scale。
     * 块 = 16 B nibble(字节 j 低 4 位 = 元素 2j, 高 4 位 = 元素 2j+1, 与 HF/官方 convert.py 同序)
     * + 1 B scale(e8m0, 值 = 2^(e-127)) = 17 B / 32 元素 = 4.25 bpw。值 = nibble 表 × scale。
     * 与 HF 出厂专家/我们量化器的 FP4 1×32 块语义逐位同, 只是把 weight/scale 两张量交织成 GGUF 块。 */
    DS4_GGT_FP4X32  = 43,
    /* 本仓自定: FP8 E4M3 + 32×32 块 ue8m0 缩放(engram wkv 的官方盘上格式, 2026-09-15 clear.md C1)。
     * 字节布局 = 先 rows×cols 个 e4m3, 紧跟 ceil(rows/32)×ceil(cols/32) 个 ue8m0 缩放字节。
     * 不是逐行块(所以进不了 gguf_types 的 block/size 表), shape 记 [cols][rows], 字节数显式给。 */
    DS4_GGT_FP8_32X32 = 44,
};

/* 类型几何: 每块元素数 blk 与每块字节数 tsz。认识返回 1, 不认识返回 0。 */
int ds4_ggt_geom(uint32_t ty, uint64_t *blk, uint64_t *tsz);

/* 单类型标量 dequant: src = nblk 个原始块, out = nblk*块元素数 个 f32。 */
void ds4_deq_q8_0(const uint8_t *src, uint64_t nblk, float *out);
void ds4_deq_q2_K(const uint8_t *src, uint64_t nblk, float *out);
void ds4_deq_q4_K(const uint8_t *src, uint64_t nblk, float *out);
void ds4_deq_iq2_xxs(const uint8_t *src, uint64_t nblk, float *out);
void ds4_deq_fp4x32(const uint8_t *src, uint64_t nblk, float *out);

/* fp4x32 编码(2026-09-13, 反修放大器产物落 fp4 用): f32[nblk*32] → nblk 个 17 B 块。
 * 与 ds4_deq_fp4x32 配对: 解码端只做 表[nibble]×2^(e−127), 所以 e 怎么选是编码端的自由度,
 * 本函数逐块搜最优(判据 = 块内平方误差)。为什么要搜见 .c 里的注释。 */
void ds4_quant_fp4x32(const float *src, uint64_t nblk, uint8_t *out);

/* 分派入口: 一段字节 → f32[nelem]。nelem 必须是块大小整数倍。
 * 成功返回 0; 类型不认识/不整除返回 -1(不打印不退出, 调用方决定停车口径)。 */
int ds4_deq_bytes(uint32_t ty, const uint8_t *src, uint64_t nelem, float *out);

/* iq2_xxs 网格常量(解码表), 供 GPU 后端上传/对拍引用 */
extern const uint8_t  ds4_iq2xxs_val[4];
extern const uint16_t ds4_iq2xxs_kgrid[256];

#endif /* DS4_COMMON_QUANTFMT_H */
