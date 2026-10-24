/* st_read_decl.h — st_bridge 的 opaque 声明(CUDA 侧用) */
#ifndef ST_READ_DECL_H
#define ST_READ_DECL_H
void *stb_open(const char *hf);
float *stb_read(void *ctx, const char *name, long *R, long *C);
#endif
