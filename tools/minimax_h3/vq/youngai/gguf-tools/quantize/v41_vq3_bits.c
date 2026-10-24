/* v41_vq3_bits.c — DQVL v3 的"13 位直排 → 12 位主流 + 1 位平面"位搬运, 与它的金标自测(2026-09-21)。
 *
 * 【为什么单独一个文件】这段是纯位运算, 错了**不报错只出假权重**(整层专家值错位, 出口尺掉几个点, 查半天),
 * 而它又是离线可完全验证的: 拆开再合回来必须逐位还原原始索引。所以做成能单独编译自测的一份代码,
 * 转换器 include 它, `make -C gguf-tools tools-test` 跑它的金标。
 *
 * 【布局】直排: 第 k 个索引占 [k*nbit, k*nbit+nbit), LSB 先, 逐行字节对齐(行首恒字节对齐)。
 * 拆后: 主流每索引 12 位(同上规则), ext 平面每索引 1 位(字节 k/8 的第 k%8 位, LSB 先)。
 *
 * 【两处边界, 都是踩过才写下来的】
 *   ① 读: 13 位最多跨 3 字节, 行尾那几个索引的 b0+3 越过本行 —— 目录里各行连续, 越界读到的是下一行的字节;
 *      最后一行更是越出张量, 有踩 mmap 尾的风险。所以按 srow 截断, 越界字节当 0(那些位不属于任何索引)。
 *   ② 写: 12 位在字节内的相位只有 0 和 4(12 mod 8 = 4), 两种都跨 2 个字节 —— 只写一个字节会把高 4 位丢掉,
 *      解出来的权重就是"每隔一个索引错一次", 而字节数完全正确、任何体积核对都发现不了。 */
#ifndef V41_VQ3_BITS_C
#define V41_VQ3_BITS_C
#include <stdint.h>
#include <string.h>

#define VQ3_MAIN_NBIT  12u                  /* 主位流位宽(几何锁死 = 09-18 那套"3 条整线一块"的读法) */

/* 一行: 直排 nbit 位流 → 主流(12 位) + ext 平面(nbit 超出 12 的那 1 位; nbit == 12 时传 NULL)。 */
static void vq3_split_row(const uint8_t *src, uint64_t srow, uint64_t nidx, int nbit,
                          uint8_t *main_out, uint8_t *ext_out) {
    const uint64_t mr = (nidx * VQ3_MAIN_NBIT + 7) / 8;
    memset(main_out, 0, mr);
    if (ext_out) memset(ext_out, 0, (nidx + 7) / 8);
    for (uint64_t k = 0; k < nidx; k++) {
        const uint64_t bit = k * (uint64_t)nbit, b0 = bit >> 3;
        uint32_t w = 0;
        for (int t = 0; t < 3 && b0 + (uint64_t)t < srow; t++) w |= (uint32_t)src[b0 + (uint64_t)t] << (8 * t);
        const uint32_t v = (w >> (bit & 7)) & ((1u << nbit) - 1u);
        const uint64_t mb = k * VQ3_MAIN_NBIT, mb0 = mb >> 3;
        const uint32_t sh = (uint32_t)(mb & 7), cur = (v & 0xFFFu) << sh;
        main_out[mb0] |= (uint8_t)(cur & 0xFFu);
        if (mb0 + 1 < mr) main_out[mb0 + 1] |= (uint8_t)((cur >> 8) & 0xFFu);
        if (sh > 4 && mb0 + 2 < mr) main_out[mb0 + 2] |= (uint8_t)((cur >> 16) & 0xFFu);
        if (ext_out && (v >> VQ3_MAIN_NBIT)) ext_out[k >> 3] |= (uint8_t)(1u << (k & 7));
    }
}

/* 反向(自测与引擎侧参照): 主流 + ext 平面 → 第 k 个索引。引擎 CUDA 核做的是同一件事, 这里是它的标量金标。
 * static inline: 转换器只用正向, 不写 inline 会在它那边报 unused-function。 */
static inline uint32_t vq3_get_index(const uint8_t *main_in, const uint8_t *ext_in, uint64_t k) {
    const uint64_t mb = k * VQ3_MAIN_NBIT, mb0 = mb >> 3;
    const uint32_t sh = (uint32_t)(mb & 7);
    uint32_t w = (uint32_t)main_in[mb0] | ((uint32_t)main_in[mb0 + 1] << 8);
    if (sh > 4) w |= (uint32_t)main_in[mb0 + 2] << 16;
    uint32_t v = (w >> sh) & 0xFFFu;
    if (ext_in && ((ext_in[k >> 3] >> (k & 7)) & 1u)) v |= 1u << VQ3_MAIN_NBIT;
    return v;
}

#ifdef V41_VQ3_TEST
#include <stdio.h>
#include <stdlib.h>
/* 金标: 造一行随机索引 → 直排打包(与量化器 pack_kernel 同式) → 拆 → 逐个取回, 必须逐位还原。
 * 覆盖 13 位与 12 位、V4.1 真实两种行长(gateup 640 / down 288)、以及 1..40 的零散长度(尾巴). */
static void pack_direct(const uint32_t *idx, uint64_t nidx, int nbit, uint8_t *out, uint64_t srow) {
    memset(out, 0, srow);
    for (uint64_t k = 0; k < nidx; k++)
        for (int b = 0; b < nbit; b++)
            if ((idx[k] >> b) & 1u) { const uint64_t p = k * (uint64_t)nbit + (uint64_t)b; out[p >> 3] |= (uint8_t)(1u << (p & 7)); }
}
static int one_case(uint64_t nidx, int nbit, unsigned seed) {
    const uint64_t srow = (nidx * (uint64_t)nbit + 7) / 8, mr = (nidx * VQ3_MAIN_NBIT + 7) / 8, er = (nidx + 7) / 8;
    uint32_t *idx = (uint32_t *)malloc(sizeof(uint32_t) * nidx);
    uint8_t *src = (uint8_t *)malloc(srow), *mo = (uint8_t *)malloc(mr + 4), *eo = (uint8_t *)malloc(er + 4);
    if (!idx || !src || !mo || !eo) { printf("v41_vq3_bits: 内存不够\n"); return 1; }
    srand(seed);
    const uint32_t msk = (1u << nbit) - 1u;
    for (uint64_t k = 0; k < nidx; k++) idx[k] = (uint32_t)rand() & msk;
    idx[0] = msk; if (nidx > 1) idx[nidx - 1] = msk;          /* 边界: 首末都取满位(最容易暴露丢高位) */
    if (nidx > 2) idx[1] = 0;
    pack_direct(idx, nidx, nbit, src, srow);
    memset(mo, 0xAA, mr + 4); memset(eo, 0x55, er + 4);        /* 毒化: 越界写会被下面的哨兵抓到 */
    vq3_split_row(src, srow, nidx, nbit, mo, nbit > (int)VQ3_MAIN_NBIT ? eo : NULL);
    int bad = 0;
    for (uint64_t k = 0; k < nidx && bad < 4; k++) {
        const uint32_t got = vq3_get_index(mo, nbit > (int)VQ3_MAIN_NBIT ? eo : NULL, k);
        if (got != idx[k]) { printf("  ★错位 nidx=%llu nbit=%d k=%llu: 期望 %u 得到 %u★\n",
                                    (unsigned long long)nidx, nbit, (unsigned long long)k, idx[k], got); bad++; }
    }
    for (int t = 0; t < 4; t++) if (mo[mr + t] != 0xAA) { printf("  ★主流越界写 nidx=%llu nbit=%d★\n", (unsigned long long)nidx, nbit); bad++; break; }
    if (nbit > (int)VQ3_MAIN_NBIT) for (int t = 0; t < 4; t++) if (eo[er + t] != 0x55) { printf("  ★ext 越界写 nidx=%llu★\n", (unsigned long long)nidx); bad++; break; }
    free(idx); free(src); free(mo); free(eo);
    return bad;
}
int main(void) {
    int bad = 0, n = 0;
    const uint64_t lens[] = {640, 288, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 39, 40};
    for (unsigned s = 1; s <= 3; s++)
        for (size_t i = 0; i < sizeof lens / sizeof lens[0]; i++)
            for (int nb = 12; nb <= 13; nb++) { bad += one_case(lens[i], nb, s * 1000u + (unsigned)i); n++; }
    /* 12 位必须与直排逐字节相同(转换器就是靠这一条对 12 位层走 memcpy 而不是逐位重排) */
    {
        const uint64_t nidx = 640, srow = nidx * 12 / 8;
        uint32_t *idx = (uint32_t *)malloc(sizeof(uint32_t) * nidx);
        uint8_t *src = (uint8_t *)malloc(srow), *mo = (uint8_t *)malloc(srow);
        srand(7); for (uint64_t k = 0; k < nidx; k++) idx[k] = (uint32_t)rand() & 0xFFFu;
        pack_direct(idx, nidx, 12, src, srow);
        vq3_split_row(src, srow, nidx, 12, mo, NULL);
        if (memcmp(src, mo, srow)) { printf("  ★12 位拆分不是恒等(转换器的 memcpy 快路就错了)★\n"); bad++; }
        n++;
        free(idx); free(src); free(mo);
    }
    printf("v41_vq3_bits 自测: %d 例, %s\n", n, bad ? "★失败★" : "全过");
    return bad ? 1 : 0;
}
#endif
#endif
