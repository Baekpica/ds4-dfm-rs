/* st_bridge.c — st_read.c 的薄桥接(2026-08-23): st_read 全 static, CUDA 工具经此
 * 以 opaque 指针使用(分离编译, nvcc 不吃 C 方言)。 */
#include "../calib/st_read.c"
void *stb_open(const char *hf) {
    st_ctx *c = malloc(sizeof(st_ctx));
    st_open(c, hf);
    return c;
}
float *stb_read(void *ctx, const char *name, long *R, long *C) {
    return st_read_weight((st_ctx *)ctx, name, R, C);
}
