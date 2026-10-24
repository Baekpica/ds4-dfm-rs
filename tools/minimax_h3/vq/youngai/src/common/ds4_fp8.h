/* ds4_fp8.h — E4M3FN / E8M0 / E2M1(FP4) 标量基元的全仓唯一实现。
 *
 * 此前 E4M3 解码在仓里有 6 份人肉副本(ds4.c / ds4_cuda.cu / ds4_distributed.c /
 * deepseek4-quantize.c / hf_read.c / st_read.c), 其中 hf_read.c 文件头自认是
 * deepseek4-quantize.c 的复制。本头收敛为一份, host/device 共用。
 *
 * 数值契约:
 *   - ds4_e4m3fn_to_f32: 位型解码。0x7f/0xff(全 1 绝对值) = NaN —— 这是 E4M3FN
 *     的 IEEE 语义, 与 st_read.c 的 ST_LUT 一致。注意 deepseek4-quantize.c 的
 *     旧实现把 0x7f 解成 0.0f; 它迁移到本头时必须在调用点显式保留旧语义
 *     (否则含 NaN 权重的量化输出会变字节), 见重构阶段7。
 *   - ds4_e4m3fn_value/ds4_e4m3fn_round: 引擎 KV FP8 往返用的幅值表+最近舍入
 *     (ties 偶数尾数优先), 逐式转录 ds4.c dsv4_e4m3fn_value_cpu/dequant_cpu。
 *   - ds4_e8m0_to_f32: e==0 按 0x00400000 位型(2^-127 次正规), 其余 2^(e-127)。
 *   - ds4_e2m1fn_*: FP4 幅值表 {0,.5,1,1.5,2,3,4,6} + 最近舍入(同 ties 规则)。
 */
#ifndef DS4_COMMON_FP8_H
#define DS4_COMMON_FP8_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#ifdef __CUDACC__
#define DS4_HOSTDEV_FP8 __host__ __device__ static inline
#else
#define DS4_HOSTDEV_FP8 static inline
#endif

/* E4M3FN 位型 → f32。abs==0x7f 是 NaN(E4M3FN 无 Inf)。 */
DS4_HOSTDEV_FP8 float ds4_e4m3fn_to_f32(uint8_t x) {
    const uint8_t abs = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (abs == 0) return sign ? -0.0f : 0.0f;
    if (abs == 0x7f) return NAN;
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    float value = exp == 0 ? ldexpf((float)man, -9)
                           : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

/* E8M0(1 字节纯指数 scale) → f32。e=0 按 0x00400000 位型。 */
DS4_HOSTDEV_FP8 float ds4_e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* E4M3FN 幅值表(下标 0..126, 不含 NaN 槽), 引擎 KV FP8 往返的二分搜索用。 */
DS4_HOSTDEV_FP8 float ds4_e4m3fn_value(int i) {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

/* x → 最近的 E4M3FN 可表示值(饱和 ±448, ties 偶数尾数优先)。
 * 即引擎的"量化再解量化"一步到位: 用于 KV FP8 存储路的数值等价参考。 */
DS4_HOSTDEV_FP8 float ds4_e4m3fn_round(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ds4_e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - ds4_e4m3fn_value(best));
        const float next_diff = fabsf(ax - ds4_e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }
    return sign * ds4_e4m3fn_value(best);
}

/* E2M1(FP4) 幅值表与最近舍入(饱和 ±6, ties 偶数尾数优先)。 */
DS4_HOSTDEV_FP8 float ds4_e2m1fn_value(int i) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    return values[i & 7];
}

DS4_HOSTDEV_FP8 float ds4_e2m1fn_round(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - ds4_e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - ds4_e2m1fn_value(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * ds4_e2m1fn_value(best);
}

/* FP4 nibble(带符号位, 0..15) → f32。与 deepseek4-quantize.c fp4_table /
 * st_read.c FP4T 逐值同(注意 8 号槽是 -0.0f)。 */
/* ★2026-09-15 判负存档: 把这张表换成纯位运算(算 f32 位型, 零访存)★
 * 假设: 这个函数在 GPU 解码核里每个权重元素都调一次, 下标逐 lane 不同, 非一致索引的常量内存读
 * 会按不同下标串行重放(一个 warp 最多 16 次)。实测 **解码 60.7 → 65.6 ms/token(慢 8%)**, 已回退。
 * 真因大概是: 这个核是访存延迟受限的, LDC 的重放被后续 warp 盖住了, 而位运算那 6 条 ALU 指令
 * 是实打实多出来的。位型版逐值与本表 16/16 逐位同(核过), 不是数值问题, 纯是代价问题。 */
DS4_HOSTDEV_FP8 float ds4_fp4_nibble_to_f32(uint8_t n) {
    static const float t[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return t[n & 15];
}

/* f32 → FP4 nibble(0..15), ds4_fp4_nibble_to_f32 的逆向那一半(2026-09-13, 反修放大器落 fp4 用)。
 * 舍入与饱和全走 ds4_e2m1fn_round —— 编码端自己再写一遍最近搜索, 迟早与它漂开。
 * 【−0.0 走 0 号槽】8 号槽解出来是 −0.0f, 数值上与 +0.0 相同; 统一编到 0 号槽, "编码→解码→
 * 再编码"才稳定到同一个位型(否则同一份权重存两次出两种字节, 逐字节对拍全废)。 */
DS4_HOSTDEV_FP8 uint8_t ds4_fp4_f32_to_nibble(float x) {
    const float v = ds4_e2m1fn_round(x);
    const float av = fabsf(v);
    uint8_t n = 0;
    for (int i = 1; i < 8; i++) {
        if (av == ds4_e2m1fn_value(i)) { n = (uint8_t)i; break; }   /* 表里的精确值, 相等比较安全 */
    }
    return n == 0 ? (uint8_t)0 : (uint8_t)(v < 0.0f ? (n | 8u) : n);
}

/* f32 → E8M0 字节(ds4_e8m0_to_f32 的逆)。入参必须已经是 2 的整数幂 —— 缩放因子本来就是
 * 这么算出来的(ceil(log2) 取幂), 所以这里只做位型搬运, 不做舍入。
 * 次正规那一格(位型 0x00400000 = 2^-127)编回 0, 与解码端的特例对上, 往返才闭合。 */
DS4_HOSTDEV_FP8 uint8_t ds4_e8m0_f32_to_byte(float s) {
    uint32_t bits;
    memcpy(&bits, &s, sizeof(bits));
    if (bits == 0x00400000u) return 0;
    return (uint8_t)((bits >> 23) & 0xffu);
}

/* f32 → E4M3FN 字节。先过 ds4_e4m3fn_round(编码端不另写一套最近搜索 —— 同 ds4_fp4_f32_to_nibble
 * 那条注释的理由: 写两份迟早漂开), 再在幅值表上二分找回下标。幅值表对下标单调递增, 二分安全。 */
DS4_HOSTDEV_FP8 uint8_t ds4_e4m3fn_f32_to_byte(float x) {
    const float v = ds4_e4m3fn_round(x);
    const float av = fabsf(v);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ds4_e4m3fn_value(mid) <= av) lo = mid;
        else hi = mid - 1;
    }
    return (uint8_t)((v < 0.0f && av != 0.0f) ? (lo | 0x80) : lo);
}

#endif /* DS4_COMMON_FP8_H */
