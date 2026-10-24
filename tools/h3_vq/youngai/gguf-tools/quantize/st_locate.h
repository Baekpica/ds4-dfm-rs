/* st_locate.h — safetensors 张量定位 + V4.1 的 FP8 解码, 探针共用一份(2026-09-10)。
 *
 * 【为什么单独成头】engram_probe 与 engram_wkv_probe 都要"在 48 个分片里找到某张量、
 * 拿到 fd 和绝对偏移、按 E4M3+E8M0 解出浮点"。抄两份 = 迟早一边改一边不改, 而解码错了
 * 不会报错, 只会静默出一组好看的假数。
 *
 * 【与 src/common/ds4_st.c 的分工】那份是 V4 的读器(FP8 E4M3 + 128×128 F32 块 scale),
 * V4.1 换成了 32×32 块 + ue8m0(纯指数)scale + 专家 FP4 打包进 I8, 不通用。等 V4.1 进
 * 生产链时读器要在 src/common 收口成一份, 这里只是量化前的探针用。
 */
#ifndef DS4_ST_LOCATE_H
#define DS4_ST_LOCATE_H
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include "../../src/common/ds4_fp8.h"

/* ue8m0: 无符号 8 位纯指数 scale, 值 = 2^(b-127), 255 保留给 NaN。V4.1 新增(V4 是 F32)。 */
static float st_e8m0(uint8_t b) { return b == 255 ? NAN : ldexpf(1.0f, (int)b - 127); }

typedef struct {
    int fd;
    long long off;        /* 张量数据在文件里的绝对偏移 */
    long long nbytes;
    long long shape[4];
    int nd;
    char dtype[32];
} st_tref;

/* 在 dir 下逐个分片扫头, 找到 name。找到返回 0(fd 已打开, 调用方负责 close)。 */
static int st_locate(const char *dir, const char *name, st_tref *T) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "目录打不开 %s\n", dir); return -1; }
    struct dirent *e; char path[4096]; int found = 0;
    while ((e = readdir(d)) && !found) {
        const char *s = strstr(e->d_name, ".safetensors");
        if (!s || s[12]) continue;
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        FILE *f = fopen(path, "rb"); if (!f) continue;
        uint64_t hlen = 0;
        if (fread(&hlen, 8, 1, f) != 1 || hlen < 2 || hlen > (1u << 30)) { fclose(f); continue; }
        char *hdr = (char *)malloc(hlen + 1);
        if (!hdr) { fclose(f); continue; }
        if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); continue; }
        hdr[hlen] = 0; fclose(f);
        char key[512]; snprintf(key, sizeof key, "\"%s\"", name);
        char *p = strstr(hdr, key);
        /* 前缀撞名防护: 键名后必须紧跟 '"'(完整键), 否则 "a.w1" 会命中 "a.w10" */
        if (p) {
            char *dt = strstr(p, "\"dtype\":\"");
            char *sh = strstr(p, "\"shape\":[");
            char *of = strstr(p, "\"data_offsets\":[");
            if (dt && sh && of) {
                dt += 9; int i = 0;
                while (dt[i] && dt[i] != '"' && i < 31) { T->dtype[i] = dt[i]; i++; }
                T->dtype[i] = 0;
                T->nd = 0; char *q = sh + 9;
                while (*q && *q != ']' && T->nd < 4) { T->shape[T->nd++] = strtoll(q, &q, 10); if (*q == ',') q++; }
                q = of + 16;
                long long s0 = strtoll(q, &q, 10); if (*q == ',') q++;
                long long s1 = strtoll(q, &q, 10);
                T->off = 8 + (long long)hlen + s0;
                T->nbytes = s1 - s0;
                T->fd = open(path, O_RDONLY);
                if (T->fd < 0) { fprintf(stderr, "打不开 %s\n", path); free(hdr); closedir(d); return -1; }
                found = 1;
                fprintf(stderr, "[定位] %s → %s %s %lldx%lld (%lld 字节)\n", name, e->d_name, T->dtype,
                        T->shape[0], T->nd > 1 ? T->shape[1] : 1, T->nbytes);
            }
        }
        free(hdr);
    }
    closedir(d);
    return found ? 0 : -1;
}

/* 读一块 [r0,r0+nr) × 全列 的 E4M3 权重, 按 32×32 块的 ue8m0 scale 解成 f32。
 * scale 张量形状 [rows/32, cols/32]; engram 的 embed 是每行独立 scale([N,8]), 也走同一条路
 * (它的"行块高"=1) —— 由调用方传 sblk_r。返回 0 成功。 */
static int st_read_fp8_block(const st_tref *W, const st_tref *S, long long r0, long long nr,
                             long long cols, long long sblk_r, long long sblk_c, float *out) {
    long long nsc = cols / sblk_c;
    uint8_t *wb = (uint8_t *)malloc((size_t)nr * cols);
    long long sr0 = r0 / sblk_r, snr = (r0 + nr - 1) / sblk_r - sr0 + 1;
    uint8_t *sb = (uint8_t *)malloc((size_t)snr * nsc);
    if (!wb || !sb) { free(wb); free(sb); return -1; }
    if (pread(W->fd, wb, (size_t)nr * cols, W->off + r0 * cols) != (ssize_t)((size_t)nr * cols)) {
        fprintf(stderr, "★weight 读不满 @行%lld★\n", r0); free(wb); free(sb); return -1;
    }
    if (pread(S->fd, sb, (size_t)snr * nsc, S->off + sr0 * nsc) != (ssize_t)((size_t)snr * nsc)) {
        fprintf(stderr, "★scale 读不满 @行%lld★\n", r0); free(wb); free(sb); return -1;
    }
    for (long long r = 0; r < nr; r++) {
        const uint8_t *wr = wb + r * cols;
        const uint8_t *sc = sb + ((r0 + r) / sblk_r - sr0) * nsc;
        float *o = out + r * cols;
        for (long long c = 0; c < cols; c++) {
            float s = st_e8m0(sc[c / sblk_c]);
            o[c] = ds4_e4m3fn_to_f32(wr[c]) * (isnan(s) ? 0.0f : s);
        }
    }
    free(wb); free(sb);
    return 0;
}

/* 读一块 FP4(E2M1) 权重: 存储是 [rows, cols/2] 的 I8, 两个 FP4 打包进一个字节;
 * scale 是 ue8m0, 形状 [rows, cols/32] —— ★注意 routed 专家的 scale 块是 1 行 × 32 列★
 * (比 attn/shared 的 32×32 更细), 官方 convert.py 的 assert 写死了这个形状。
 * 打包顺序: 低 nibble = 偶数列, 高 nibble = 奇数列(官方 convert.py: low 先 high 后,
 * 与 deepseek4-quantize_p2.inc.c:315 逐字同)。搞反了不会报错, 只会静默出一组假数。
 * cols 传【逻辑列数】(解包后的), 即 shape[1]*2。 */
static int st_read_fp4_block(const st_tref *W, const st_tref *S, long long r0, long long nr,
                             long long cols, long long sblk_c, float *out) {
    long long packed = cols / 2, nsc = cols / sblk_c;
    uint8_t *wb = (uint8_t *)malloc((size_t)nr * packed);
    uint8_t *sb = (uint8_t *)malloc((size_t)nr * nsc);
    if (!wb || !sb) { free(wb); free(sb); return -1; }
    if (pread(W->fd, wb, (size_t)nr * packed, W->off + r0 * packed) != (ssize_t)((size_t)nr * packed)) {
        fprintf(stderr, "★FP4 weight 读不满 @行%lld★\n", r0); free(wb); free(sb); return -1;
    }
    if (pread(S->fd, sb, (size_t)nr * nsc, S->off + r0 * nsc) != (ssize_t)((size_t)nr * nsc)) {
        fprintf(stderr, "★FP4 scale 读不满 @行%lld★\n", r0); free(wb); free(sb); return -1;
    }
    for (long long r = 0; r < nr; r++) {
        const uint8_t *wr = wb + r * packed, *sc = sb + r * nsc;
        float *o = out + r * cols;
        for (long long j = 0; j < packed; j++) {
            uint8_t q = wr[j];
            long long c0 = 2 * j;
            float s0 = st_e8m0(sc[c0 / sblk_c]), s1 = st_e8m0(sc[(c0 + 1) / sblk_c]);
            o[c0]     = ds4_fp4_nibble_to_f32(q & 0x0f)        * (isnan(s0) ? 0.0f : s0);
            o[c0 + 1] = ds4_fp4_nibble_to_f32((q >> 4) & 0x0f) * (isnan(s1) ? 0.0f : s1);
        }
    }
    free(wb); free(sb);
    return 0;
}

#endif
