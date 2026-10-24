/* st_dump_tensor.c — 用本仓 C 读器解一块权重, 写成裸 f32 供对拍(2026-09-11)。
 *
 * 【为什么要它】V4.1 的权重格式解错不会报错, 只会静默出假数(FP4 两两打包的高低 nibble
 * 顺序、ue8m0 的指数偏置、routed 专家 1×32 与其余 32×32 两种块形状)。所以 C 读器必须
 * 跟一个独立实现逐位对拍 —— 对面是 torch 的官方 dtype(v41_dequant_parity.py)。
 *
 * 输出: <out>/<张量名>.f32 (nr×cols 个 f32) + 同名 .json(r0/nr/cols, 给对拍方切同一块)
 *
 * 编译: cc -O3 -std=c99 -o st_dump_tensor st_dump_tensor.c -lm
 * 用法: st_dump_tensor <hf-dir> <out-dir> <张量名> [起始行 行数]
 */
#include "st_locate.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: st_dump_tensor <hf-dir> <out-dir> <张量名> [起始行 行数]\n"); return 2; }
    const char *dir = argv[1], *out = argv[2], *name = argv[3];
    long long r0 = argc > 4 ? atoll(argv[4]) : 0;
    long long nr = argc > 5 ? atoll(argv[5]) : 8;

    st_tref W, S;
    char sname[512];
    snprintf(sname, sizeof sname, "%.*s.scale", (int)(strrchr(name, '.') - name), name);
    if (st_locate(dir, name, &W)) { fprintf(stderr, "★找不到 %s★\n", name); return 2; }
    if (st_locate(dir, sname, &S)) { fprintf(stderr, "★找不到 %s★\n", sname); return 2; }

    int is_fp4 = (strcmp(W.dtype, "I8") == 0);
    long long cols = is_fp4 ? W.shape[1] * 2 : W.shape[1];
    long long rows = W.shape[0];
    if (r0 + nr > rows) { fprintf(stderr, "★越界: r0+nr=%lld > rows=%lld★\n", r0 + nr, rows); return 2; }

    float *buf = (float *)malloc((size_t)nr * cols * 4);
    if (!buf) { fprintf(stderr, "内存不足\n"); return 1; }
    int rc;
    if (is_fp4) {
        /* 专家 FP4: scale 每行一份, 每份管 cols/S.shape[1] 列 */
        rc = st_read_fp4_block(&W, &S, r0, nr, cols, cols / S.shape[1], buf);
    } else {
        /* E4M3: 块高 = rows/S.shape[0](专家外的矩阵是 32, engram 的 embed 是 1) */
        rc = st_read_fp8_block(&W, &S, r0, nr, cols, rows / S.shape[0], cols / S.shape[1], buf);
    }
    if (rc) return 1;

    char path[4096];
    snprintf(path, sizeof path, "%s/%s.f32", out, name);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "★写不了 %s★\n", path); return 1; }
    if (fwrite(buf, 4, (size_t)nr * cols, f) != (size_t)nr * cols) { fprintf(stderr, "★写不满★\n"); return 1; }
    fclose(f);
    snprintf(path, sizeof path, "%s/%s.json", out, name);
    f = fopen(path, "w");
    fprintf(f, "{\"r0\":%lld,\"nr\":%lld,\"cols\":%lld,\"dtype\":\"%s\",\"rows\":%lld}\n",
            r0, nr, cols, W.dtype, rows);
    fclose(f);
    fprintf(stderr, "[dump] %s 行[%lld,%lld) × %lld 列, dtype=%s\n", name, r0, r0 + nr, cols, W.dtype);
    free(buf); close(W.fd); close(S.fd);
    return 0;
}
