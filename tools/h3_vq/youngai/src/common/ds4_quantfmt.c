/* ds4_quantfmt.c — GGUF 量化块标量 dequant。数值契约见 ds4_quantfmt.h 头注释;
 * 一切改动先过 tests/unit/test_common.c 的金标夹具逐字节闸。 */
#include "ds4_quantfmt.h"
#include "ds4_float.h"
#include "ds4_fp8.h"

#include <assert.h>
#include <string.h>

int ds4_ggt_geom(uint32_t ty, uint64_t *blk, uint64_t *tsz) {
    switch (ty) {
        case DS4_GGT_F32:     *blk = 1;   *tsz = 4;   return 1;
        case DS4_GGT_F16:     *blk = 1;   *tsz = 2;   return 1;
        case DS4_GGT_BF16:    *blk = 1;   *tsz = 2;   return 1;
        /* e4m3 平面 + 32×32 块缩放: 1024 个元素配 1 个缩放字节(见 core_gguf.c 类型表的注释) */
        case DS4_GGT_FP8_32X32: *blk = 1024; *tsz = 1025; return 1;
        case DS4_GGT_Q8_0:    *blk = 32;  *tsz = 34;  return 1;
        case DS4_GGT_Q2_K:    *blk = 256; *tsz = 84;  return 1;
        case DS4_GGT_Q4_K:    *blk = 256; *tsz = 144; return 1;
        case DS4_GGT_IQ2_XXS: *blk = 256; *tsz = 66;  return 1;
        case DS4_GGT_VQBLOB:  *blk = 1;   *tsz = 1;   return 1;
        case DS4_GGT_FP4X32:  *blk = 32;  *tsz = 17;  return 1;
        default: return 0;
    }
}

/* fp4x32: 16 B nibble + 1 B e8m0。nibble 表与 scale 解码都走 ds4_fp8.h(全仓唯一基元)。 */
void ds4_deq_fp4x32(const uint8_t *src, uint64_t nblk, float *out) {
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = src + b * 17u;
        const float s = ds4_e8m0_to_f32(blk[16]);
        float *o = out + b * 32u;
        for (int j = 0; j < 16; j++) {
            o[2 * j]     = ds4_fp4_nibble_to_f32(blk[j] & 0x0F) * s;
            o[2 * j + 1] = ds4_fp4_nibble_to_f32(blk[j] >> 4) * s;
        }
    }
}

/* f32 → fp4x32。
 * 【scale 为什么要逐块搜】E2M1 的幅值格点是 {0,.5,1,1.5,2,3,4,6}, 一块 32 个元素共用一个
 * 2 的幂 scale。选大了格点太粗, 选小了大值被饱和裁到 ±6·scale。哪种更亏取决于这一块的分布:
 * 反修放大器的 A/B 是解算出来的低秩因子, 行与行的动态范围能差一个量级, 固定口径(只按 RMS
 * 或只按 amax)在另一半块上就是系统性误差。scale 本来就逐块存在文件里, 搜它不要钱 ——
 * 6 档 × 32 元素, 整份 A/B 编一次也就几百毫秒。
 * 【搜哪 6 档】e0 = ceil(log2(amax/6)) 是"一个都不裁"的最小指数; 从 e0−1(允许少量裁剪, 重尾
 * 块上反而更准) 到 e0+4 各试一次, 取块内平方误差最小的。
 * 【全零块】scale 存 127(=2^0), nibble 全 0 —— 解出来就是 0, 且位型唯一(不留随机残字节)。 */
void ds4_quant_fp4x32(const float *src, uint64_t nblk, uint8_t *out) {
    for (uint64_t b = 0; b < nblk; b++) {
        const float *in = src + b * 32u;
        uint8_t *blk = out + b * 17u;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            const float a = fabsf(in[j]);
            if (a > amax) amax = a;
        }
        if (!(amax > 0.0f)) { memset(blk, 0, 16); blk[16] = 127; continue; }
        const int e0 = (int)ceilf(log2f(amax / 6.0f));
        int ebest = 0;
        float errbest = -1.0f;
        for (int d = -1; d <= 4; d++) {
            int e = e0 + d;
            if (e < -126) e = -126;          /* e8m0 字节 = e+127, 0 号是次正规特例, 255 是 NaN 槽 */
            if (e > 127) e = 127;
            const float s = ldexpf(1.0f, e), inv = 1.0f / s;
            float err = 0.0f;
            for (int j = 0; j < 32; j++) {
                const float r = ds4_e2m1fn_round(in[j] * inv) * s - in[j];
                err += r * r;
            }
            if (errbest < 0.0f || err < errbest) { errbest = err; ebest = e; }
        }
        const float inv = 1.0f / ldexpf(1.0f, ebest);
        for (int j = 0; j < 16; j++) {
            const uint8_t lo = ds4_fp4_f32_to_nibble(in[2 * j] * inv);
            const uint8_t hi = ds4_fp4_f32_to_nibble(in[2 * j + 1] * inv);
            blk[j] = (uint8_t)(lo | (uint8_t)(hi << 4));
        }
        blk[16] = (uint8_t)(ebest + 127);
    }
}

/* q2_K: 与 ds4.c deq_q2K_row_f32 / CUDA host_deq_q2k_block 同式(已对拍);
 * 索引式 qpos/shift 一字未改。 */
void ds4_deq_q2_K(const uint8_t *src, uint64_t nblk, float *out) {
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = src + b * 84u;
        const uint8_t *sc = blk, *qs = blk + 16;
        uint16_t hd, hm;
        memcpy(&hd, blk + 80, 2); memcpy(&hm, blk + 82, 2);
        const float d = ds4_f16_to_f32(hd), dm = ds4_f16_to_f32(hm);
        float *o = out + b * 256u;
        for (int j = 0; j < 16; j++) {
            const float dj = d * (float)(sc[j] & 0xF), mj = dm * (float)(sc[j] >> 4);
            for (int ii = 0; ii < 16; ii++) {
                const int idx = j * 16 + ii;
                const int qpos = (idx / 128) * 32 + (idx % 32);
                const int q = (qs[qpos] >> ((idx % 128) / 32 * 2)) & 3;
                o[idx] = dj * (float)q - mj;
            }
        }
    }
}

/* q4_K: llama.cpp get_scale_min_k4 + dequantize_row_q4_K(= gguf-py Q4_K.get_scale_min) */
static void q4k_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
           *m = (uint8_t)((q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4)); }
}

void ds4_deq_q4_K(const uint8_t *src, uint64_t nblk, float *out) {
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = src + b * 144u;
        uint16_t hd, hm;
        memcpy(&hd, blk, 2); memcpy(&hm, blk + 2, 2);
        const float d = ds4_f16_to_f32(hd), dmin = ds4_f16_to_f32(hm);
        const uint8_t *scales = blk + 4, *q = blk + 16;
        float *y = out + b * 256u;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            q4k_scale_min(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = dmin * (float)m;
            q4k_scale_min(is + 1, scales, &sc, &m);
            const float d2 = d * (float)sc, m2 = dmin * (float)m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4)  - m2;
            q += 32; is += 2;
        }
    }
}

void ds4_deq_q8_0(const uint8_t *src, uint64_t nblk, float *out) {
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = src + b * 34u;
        uint16_t hd; memcpy(&hd, blk, 2);
        const float d = ds4_f16_to_f32(hd);
        const int8_t *qs = (const int8_t *)(blk + 2);
        float *o = out + b * 32u;
        for (int j = 0; j < 32; j++) o[j] = d * (float)qs[j];
    }
}

/* iq2_xxs 双表(陷阱说明见头文件): val = 解码值 {0x08,0x19,0x2b}, kgrid = 2bit 打包网格。
 * 符号表 ksigns_iq2xs 不抄: 它就是"popcount 为奇数则置 bit7", 现场算。 */
const uint8_t ds4_iq2xxs_val[4] = { 0x08, 0x19, 0x2b, 0x00 };
const uint16_t ds4_iq2xxs_kgrid[256] = {
        0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
      100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
     1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
     1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
     2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
     4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
     5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
     8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
};

void ds4_deq_iq2_xxs(const uint8_t *src, uint64_t nblk, float *out) {
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = src + b * 66u;
        uint16_t hd; memcpy(&hd, blk, 2);
        const float d = ds4_f16_to_f32(hd);
        float *y = out + b * 256u;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint16_t q2[4];
            memcpy(q2, blk + 2 + ib32 * 8, 8);
            const uint32_t a_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t a_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            const float db = d * (0.5f + (float)(a_s >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint32_t gi = (a_g >> (8 * l)) & 0xFFu;         /* aux8[l] */
                const uint16_t kg = ds4_iq2xxs_kgrid[gi];
                uint32_t s7 = (a_s >> (7 * l)) & 127u;
                int par = 0;
                for (int t = 0; t < 7; t++) par ^= (int)((s7 >> t) & 1u);
                const uint32_t signs = par ? (s7 | 0x80u) : s7;       /* = ksigns_iq2xs[s7] */
                for (int j = 0; j < 8; j++) {
                    const int code = (kg >> (2 * j)) & 3;
                    assert(code != 3 && "iq2_xxs 打包网格不含码 3 — 表被改坏");
                    const float gv = (float)ds4_iq2xxs_val[code];
                    *y++ = (signs & (1u << j)) ? -(db * gv) : (db * gv);
                }
            }
        }
    }
}

int ds4_deq_bytes(uint32_t ty, const uint8_t *src, uint64_t nelem, float *out) {
    uint64_t blk, tsz;
    if (!ds4_ggt_geom(ty, &blk, &tsz)) return -1;
    if (nelem % blk) return -1;
    const uint64_t nb = nelem / blk;
    switch (ty) {
        case DS4_GGT_F32:  memcpy(out, src, (size_t)nelem * 4); return 0;
        case DS4_GGT_F16:
            for (uint64_t i = 0; i < nelem; i++) { uint16_t h; memcpy(&h, src + i * 2, 2); out[i] = ds4_f16_to_f32(h); }
            return 0;
        case DS4_GGT_BF16:
            for (uint64_t i = 0; i < nelem; i++) { uint16_t h; memcpy(&h, src + i * 2, 2); out[i] = ds4_bf16_to_f32(h); }
            return 0;
        case DS4_GGT_Q8_0:    ds4_deq_q8_0(src, nb, out);    return 0;
        case DS4_GGT_Q2_K:    ds4_deq_q2_K(src, nb, out);    return 0;
        case DS4_GGT_Q4_K:    ds4_deq_q4_K(src, nb, out);    return 0;
        case DS4_GGT_IQ2_XXS: ds4_deq_iq2_xxs(src, nb, out); return 0;
        case DS4_GGT_FP4X32:  ds4_deq_fp4x32(src, nb, out);  return 0;
        default: return -1;
    }
}
