/* ds4_gr_fnv.h — 增益插件目录的指纹(2026-09-13)。引擎(加载时核对)与解算器(落盘时写入)
 * 共用这一份, 免得两边各算各的、对不上还查不出为什么。
 *
 * 干什么用: 三文件部署里, ③后训练件是对"①+②这个态"解出来的。② 换了版, ③ 就打在错的基线上,
 * 照样能跑、照样出一个像模像样的读数 —— 那是最难查的一类错。所以 ③ 落盘时把 ② 目录的指纹写进
 * base.fnv, 引擎加载 ③ 时重算比对, 不符就停车。
 *
 * 指纹 = 按层号顺序对目录里 gr_Lnn.bin 的全字节做 FNV-1a 64。防的是"拿错了文件", 不是防篡改,
 * 所以不引哈希库; 文件数一起记, 少一层也能看出来。 */
#ifndef DS4_GR_FNV_H
#define DS4_GR_FNV_H

#include <stdint.h>
#include <stdio.h>

#define DS4_GR_FNV_SEED 1469598103934665603ull
#define DS4_GR_FNV_PRIME 1099511628211ull

/* 扫 dir/gr_L00.bin .. gr_L(nl-1).bin(缺的层跳过), 回填文件数。 */
static uint64_t ds4_gr_dir_fnv(const char *dir, unsigned nl, unsigned *n_file) {
    uint64_t h = DS4_GR_FNV_SEED;
    unsigned cnt = 0;
    unsigned char buf[65536];
    static const char *const kinds[2] = { "gr", "rb" };   /* 增益表 + 路由偏置(2026-09-20): ② 的任一件换版, ③ 都得重解 */
    for (unsigned il = 0; il < nl; il++) {
        for (int kd = 0; kd < 2; kd++) {
            char p[4200];
            snprintf(p, sizeof p, "%s/%s_L%02u.bin", dir, kinds[kd], il);
            FILE *f = fopen(p, "rb");
            if (!f) continue;
            size_t got;
            while ((got = fread(buf, 1, sizeof buf, f)) > 0)
                for (size_t t = 0; t < got; t++) { h ^= buf[t]; h *= DS4_GR_FNV_PRIME; }
            fclose(f);
            cnt++;
        }
    }
    if (n_file) *n_file = cnt;
    return h;
}

#endif /* DS4_GR_FNV_H */
