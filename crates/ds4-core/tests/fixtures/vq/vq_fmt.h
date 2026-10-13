/* vq_fmt.h — v2.2 VQ 码本侧车 DQVQ/DQVL 只读解析(量化器与引擎共用, 零依赖纯解码)。
 * 层 blob(dql_vq_L%02d.bin 原字节 / overlay GGUF blk.L.ffn_exps_vq.blob):
 *   [u32 'DQVL'][u32 ver][u32 L][u32 nexp=256][表 256×3 u64 载荷偏移(0=无)] [DQVQ 载荷...]
 * 每矩阵载荷: [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
 *   [码本 nc*dim f16][g_r rows f16][索引流: dim8→1B/idx; dim4→9bit 位流 LE(+1 安全字节)]
 * 值 = 码本[idx][d] * g_r[row]。表内 which: 0=w1 1=w3 2=w2(仅热非零)。 */
#ifndef DS4_VQ_FMT_H
#define DS4_VQ_FMT_H
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define DS4VQ_BLOB_MAGIC 0x4C565144u
#define DS4VQ_MAT_MAGIC  0x51565144u   /* v2 载荷: 头 + 自带码本 + 增益 + 位流 */
/* ★v3(2026-09-21, 113.md 方案 v3)★: 码本一层一本(提到 blob 头之后, 载荷用 cb_off 指过去, 只存 E4M3)
 * + 13 位索引拆成"12 位主流 + 1 位平面"。载荷换了魔数, 所以老二进制读 v3 会在这里认不出来而不是错位解。
 * ★槽表位置(offset 16)两版相同★ —— 挪了它老二进制就会把码本字节当偏移用, 那是随机垃圾而不是可诊断的失败。 */
#define DS4VQ_MAT3_MAGIC 0x33565144u   /* 'DQV3' */
#define DS4VQ_BLOB_VER_MIN 2u
#define DS4VQ_BLOB_VER_MAX 3u

static inline float ds4vq_f16(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) { if (!m) f = s; else { e = 112; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; f = s | (e << 23) | (m << 13); } }
    else if (e == 31) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float out; memcpy(&out, &f, 4); return out;
}

/* blob 校验 + 取矿阵载荷偏移(0=无)。
 * ★nexp 读头里的字段(偏移 12), 不再写死 256★(2026-09-12): V4.1 一层 384 专家, 表是 384×3;
 * 旧 V4 blob 的头同样写着 256, 所以老文件语义不变。 */
static inline uint32_t ds4vq_blob_nexp(const uint8_t *blob) {
    uint32_t n; memcpy(&n, blob + 12, 4); return n;
}
static inline uint32_t ds4vq_blob_ver(const uint8_t *blob) {
    uint32_t v; memcpy(&v, blob + 4, 4); return v;
}
/* ★ver 白名单(2026-09-21)★: 以前这里只查魔数, 于是一个将来版本的 blob 会被判"合法"然后按本版布局解 ——
 * 不报错, 只出假权重。加白名单之后, 读到不认识的版本是**加载期硬停**(调用方 exit), 那是能一眼定位的失败。 */
static inline int ds4vq_blob_ok(const uint8_t *blob, size_t sz) {
    if (!blob || sz < 16) return 0;
    uint32_t mg; memcpy(&mg, blob, 4);
    if (mg != DS4VQ_BLOB_MAGIC) return 0;
    const uint32_t ver = ds4vq_blob_ver(blob);
    if (ver < DS4VQ_BLOB_VER_MIN || ver > DS4VQ_BLOB_VER_MAX) return 0;
    const uint32_t nexp = ds4vq_blob_nexp(blob);
    return nexp >= 1 && nexp <= 4096 && sz >= 16 + (size_t)nexp * 3 * 8;
}
static inline uint64_t ds4vq_slot(const uint8_t *blob, int e, int which) {
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    return off;
}

/* 载荷 dequant → f32(引擎 CPU MoE fallback 用)。out[rows*cols] row-major。0=成功 */
static inline int ds4vq_dequant_f32(const uint8_t *pay, float *out, int exp_rows, int exp_cols) {
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return -1;
    uint16_t dim, nc; memcpy(&dim, pay + 4, 2); memcpy(&nc, pay + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
    if ((int)rows != exp_rows || (int)cols != exp_cols) return -2;
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)nc * dim * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    /* 堆分配 + 通用位宽(2026-08-08: f16 版早修过"硬编码 9bit 只认 nc512", f32 版漏修 —
     * r86 冷 vq4×256(8bit 索引)在 CPU MoE 生产路被按 9bit 错位解 ⇒ 107 冷专家全垃圾。
     * 与 f16 版/量化器 vq_nbits() 同口径。 */
    float *cbf = (float *)malloc((size_t)nc * dim * sizeof(float));
    if (!cbf) return -3;
    for (int i = 0; i < (int)nc * dim; i++) { uint16_t h; memcpy(&h, cb + 2 * (size_t)i, 2); cbf[i] = ds4vq_f16(h); }
    int nbit = 0; while ((1 << nbit) < (int)nc) nbit++; if (nbit < 1) nbit = 1;
    const uint32_t imsk = (nbit >= 32) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    for (uint32_t r = 0; r < rows; r++) {
        uint16_t gh; memcpy(&gh, gr + 2 * (size_t)r, 2);
        float g = ds4vq_f16(gh);
        size_t i0 = (size_t)r * cols / dim, i1 = i0 + cols / dim;
        float *orow = out + (size_t)r * cols;
        for (size_t i = i0; i < i1; i++) {
            uint32_t v;
            if (nbit == 8) v = ix[i];
            else { size_t bit = i * (size_t)nbit; uint32_t w = (uint32_t)ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8) | ((uint32_t)ix[(bit >> 3) + 2] << 16); v = (w >> (bit & 7)) & imsk; }
            const float *c = cbf + (size_t)v * dim;
            float *o = orow + (i - i0) * dim;
            for (int d = 0; d < dim; d++) o[d] = c[d] * g;
        }
    }
    free(cbf);
    return 0;
}

/* 载荷 dequant → f16 半精度输出(引擎 scratch 口径); rows/cols 校验用。0=成功 */
static inline int ds4vq_dequant_f16(const uint8_t *pay, uint16_t *out,
                                    int exp_rows, int exp_cols) {
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return -1;
    uint16_t dim, nc; memcpy(&dim, pay + 4, 2); memcpy(&nc, pay + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
    if ((int)rows != exp_rows || (int)cols != exp_cols) return -2;
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)nc * dim * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    /* 码本先转 f32, 逐行乘 g_r 后转 f16 输出。
     * ★堆分配(2026-08-01 修 SIGBUS)★: 原 `float cbf[512*8]` 是 16 KiB 栈数组, 只够
     * 冠军的 vq4×512 / vq8×256。R28 计划表用到 nc≤1024 × dim≤32 = 32768 float(128 KiB),
     * 直接踩穿 gather 线程栈 → KERN_PROTECTION_FAILURE。码本相对整矩阵 dequant
     * (rows×cols 元素)是小量, 每次 malloc 的开销可忽略。 */
    float *cbf = (float *)malloc((size_t)nc * dim * sizeof(float));
    if (!cbf) return -3;
    for (int i = 0; i < (int)nc * dim; i++) { uint16_t h; memcpy(&h, cb + 2 * (size_t)i, 2); cbf[i] = ds4vq_f16(h); }
    /* 索引位宽由码本大小定, 与量化器 vq_nbits() 同口径 —— 原实现硬编码 9bit,
     * 对 nc=256(8bit 字节流)/nc=1024(10bit)全部错位。 */
    int nbit = 0; while ((1 << nbit) < (int)nc) nbit++; if (nbit < 1) nbit = 1;
    const uint32_t imsk = (nbit >= 32) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    size_t nidx = (size_t)rows * cols / dim;
    for (uint32_t r = 0; r < rows; r++) {
        uint16_t gh; memcpy(&gh, gr + 2 * (size_t)r, 2);
        float g = ds4vq_f16(gh);
        size_t i0 = (size_t)r * cols / dim, i1 = i0 + cols / dim;
        uint16_t *orow = out + (size_t)r * cols;
        for (size_t i = i0; i < i1; i++) {
            uint32_t v;
            if (nbit == 8) v = ix[i];
            else { size_t bit = i * (size_t)nbit; uint32_t w = (uint32_t)ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8) | ((uint32_t)ix[(bit >> 3) + 2] << 16); v = (w >> (bit & 7)) & imsk; }
            const float *c = cbf + (size_t)v * dim;
            uint16_t *o = orow + (i - i0) * dim;
            for (int d = 0; d < dim; d++) {
                /* f32→f16: ARM 硬件 FCVT(round-to-nearest-even), 比软件位操作快~10×,
                 * 是 VQ dequant decode ~21% 提速; 已验非质量因(回退测 bare 仍 echo=无关)。 */
                __fp16 hh = (__fp16)(c[d] * g);
                memcpy(&o[d], &hh, 2);
            }
        }
        (void)nidx;
    }
    free(cbf);
    return 0;
}
#endif
