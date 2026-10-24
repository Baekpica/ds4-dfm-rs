/* ds4_gguf_write.h — GGUF v3 容器【写盘器】的全仓唯一实现(2026-09-12, V4.1 转换器首用)。
 *
 * 【用法】两段式: ①先 kv_*() 把元数据攒在内存, tensor() 登记每个张量(名/类型/形状/字节数);
 * ②header() 一次写出 magic/计数/KV/张量表并补齐到 alignment; 之后按登记顺序 write() 逐个张量
 * 的数据(每个张量尾部自动补 alignment 填充); end() 关文件。
 * 为什么两段: GGUF 张量表里每个张量带"相对数据区偏移", 数据区起点又取决于头的总长, 所以
 * 必须先知道全部张量的字节数才能写头 —— 而张量字节数只由类型×形状决定, 登记时就算得出,
 * 数据可以边算边追加, 不需要攒在内存里。
 * 【与引擎读器的契约】版本 3, alignment 默认 32(引擎 core_model_open 读 general.alignment,
 * 缺省 32); 张量数据区起点 = 头长向上对齐; 每个张量偏移相对数据区起点且对齐。
 * 类型编号用 gguf ggml 号(core_gguf.c gguf_types 表): 0 f32 / 1 f16 / 24 i8 / 26 i32 / 27 i64 /
 * 42 vqblob / 43 fp4x32(本仓新增, 见 ds4_quantfmt.h)。字节数由调用方按类型算好传入。
 * 【出错】返回 -1 并 stderr 一句人话, 不 exit —— 调用方定生死。 */
#ifndef DS4_COMMON_GGUF_WRITE_H
#define DS4_COMMON_GGUF_WRITE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

enum { DS4GW_U8 = 0, DS4GW_I8 = 1, DS4GW_U16 = 2, DS4GW_I16 = 3, DS4GW_U32 = 4, DS4GW_I32 = 5,
       DS4GW_F32 = 6, DS4GW_BOOL = 7, DS4GW_STR = 8, DS4GW_ARR = 9, DS4GW_U64 = 10, DS4GW_I64 = 11, DS4GW_F64 = 12 };

typedef struct { char name[160]; uint32_t nd, type; uint64_t ne[4]; uint64_t nbytes, off; } ds4gw_tinfo;

typedef struct {
    FILE *f; char path[4300];
    uint8_t *kv; size_t kvlen, kvcap; uint64_t nkv;
    ds4gw_tinfo *t; int nt, tcap, next;
    uint64_t align, data0, total;
} ds4gw;

static inline int ds4gw_begin(ds4gw *W, const char *path) {
    memset(W, 0, sizeof *W);
    snprintf(W->path, sizeof W->path, "%s", path);
    W->f = fopen(path, "wb");
    if (!W->f) { fprintf(stderr, "★写不了 %s★\n", path); return -1; }
    W->align = 32;
    return 0;
}
static inline void ds4gw_put(ds4gw *W, const void *p, size_t n) {
    if (W->kvlen + n > W->kvcap) { W->kvcap = (W->kvlen + n) * 2 + 4096; W->kv = (uint8_t *)realloc(W->kv, W->kvcap); }
    memcpy(W->kv + W->kvlen, p, n); W->kvlen += n;
}
static inline void ds4gw_put_u32(ds4gw *W, uint32_t v) { ds4gw_put(W, &v, 4); }
static inline void ds4gw_put_u64(ds4gw *W, uint64_t v) { ds4gw_put(W, &v, 8); }
static inline void ds4gw_put_str(ds4gw *W, const char *s) { uint64_t n = strlen(s); ds4gw_put_u64(W, n); ds4gw_put(W, s, n); }

static inline void ds4gw_kv_u32(ds4gw *W, const char *k, uint32_t v) { ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_U32); ds4gw_put_u32(W, v); W->nkv++; }
static inline void ds4gw_kv_u64(ds4gw *W, const char *k, uint64_t v) { ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_U64); ds4gw_put_u64(W, v); W->nkv++; }
static inline void ds4gw_kv_f32(ds4gw *W, const char *k, float v) { ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_F32); ds4gw_put(W, &v, 4); W->nkv++; }
static inline void ds4gw_kv_bool(ds4gw *W, const char *k, int v) { uint8_t b = v ? 1 : 0; ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_BOOL); ds4gw_put(W, &b, 1); W->nkv++; }
static inline void ds4gw_kv_str(ds4gw *W, const char *k, const char *v) { ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_STR); ds4gw_put_str(W, v); W->nkv++; }
static inline void ds4gw_kv_arr_i32(ds4gw *W, const char *k, const int32_t *v, uint64_t n) {
    ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_ARR); ds4gw_put_u32(W, DS4GW_I32); ds4gw_put_u64(W, n); ds4gw_put(W, v, n * 4); W->nkv++;
}
static inline void ds4gw_kv_arr_u32(ds4gw *W, const char *k, const uint32_t *v, uint64_t n) {
    ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_ARR); ds4gw_put_u32(W, DS4GW_U32); ds4gw_put_u64(W, n); ds4gw_put(W, v, n * 4); W->nkv++;
}
static inline void ds4gw_kv_arr_f32(ds4gw *W, const char *k, const float *v, uint64_t n) {
    ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_ARR); ds4gw_put_u32(W, DS4GW_F32); ds4gw_put_u64(W, n); ds4gw_put(W, v, n * 4); W->nkv++;
}
static inline void ds4gw_kv_arr_u64(ds4gw *W, const char *k, const uint64_t *v, uint64_t n) {
    ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_ARR); ds4gw_put_u32(W, DS4GW_U64); ds4gw_put_u64(W, n); ds4gw_put(W, v, n * 8); W->nkv++;
}
static inline void ds4gw_kv_arr_str(ds4gw *W, const char *k, const char *const *v, uint64_t n) {
    ds4gw_put_str(W, k); ds4gw_put_u32(W, DS4GW_ARR); ds4gw_put_u32(W, DS4GW_STR); ds4gw_put_u64(W, n);
    for (uint64_t i = 0; i < n; i++) ds4gw_put_str(W, v[i]);
    W->nkv++;
}

static inline uint64_t ds4gw_pad(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

/* 登记张量(按最终写出顺序); ne 按 GGUF 习惯【内维在前】。返回序号。 */
static inline int ds4gw_tensor(ds4gw *W, const char *name, uint32_t type, uint32_t nd, const uint64_t *ne, uint64_t nbytes) {
    if (W->nt == W->tcap) { W->tcap = W->tcap ? W->tcap * 2 : 1024; W->t = (ds4gw_tinfo *)realloc(W->t, sizeof(ds4gw_tinfo) * W->tcap); }
    ds4gw_tinfo *T = &W->t[W->nt];
    memset(T, 0, sizeof *T);
    snprintf(T->name, sizeof T->name, "%s", name);
    T->type = type; T->nd = nd;
    for (uint32_t i = 0; i < nd && i < 4; i++) T->ne[i] = ne[i];
    T->nbytes = nbytes;
    T->off = W->total;
    W->total += ds4gw_pad(nbytes, W->align);
    return W->nt++;
}

static inline int ds4gw_header(ds4gw *W) {
    /* 头 = magic(4) version(4) n_tensors(8) n_kv(8) + KV 段 + 张量表 */
    uint64_t tbl = 0;
    for (int i = 0; i < W->nt; i++) tbl += 8 + strlen(W->t[i].name) + 4 + (uint64_t)W->t[i].nd * 8 + 4 + 8;
    uint64_t hl = 4 + 4 + 8 + 8 + W->kvlen + tbl;
    W->data0 = ds4gw_pad(hl, W->align);
    FILE *f = W->f;
    uint32_t ver = 3; uint64_t nt = (uint64_t)W->nt;
    if (fwrite("GGUF", 1, 4, f) != 4 || fwrite(&ver, 4, 1, f) != 1 || fwrite(&nt, 8, 1, f) != 1 || fwrite(&W->nkv, 8, 1, f) != 1) goto bad;
    if (W->kvlen && fwrite(W->kv, 1, W->kvlen, f) != W->kvlen) goto bad;
    for (int i = 0; i < W->nt; i++) {
        ds4gw_tinfo *T = &W->t[i];
        uint64_t n = strlen(T->name);
        if (fwrite(&n, 8, 1, f) != 1 || fwrite(T->name, 1, n, f) != n || fwrite(&T->nd, 4, 1, f) != 1) goto bad;
        for (uint32_t d = 0; d < T->nd; d++) if (fwrite(&T->ne[d], 8, 1, f) != 1) goto bad;
        if (fwrite(&T->type, 4, 1, f) != 1 || fwrite(&T->off, 8, 1, f) != 1) goto bad;
    }
    for (uint64_t p = hl; p < W->data0; p++) if (fputc(0, f) == EOF) goto bad;
    return 0;
bad:
    fprintf(stderr, "★GGUF 头写失败 %s★\n", W->path); return -1;
}

/* 必须按登记顺序写; 字节数与登记不符就硬停(偏一个字节整个文件全错位) */
static inline int ds4gw_write(ds4gw *W, int ti, const void *data, uint64_t nbytes) {
    if (ti != W->next) { fprintf(stderr, "★GGUF 写序错: 期望 #%d 来了 #%d(%s)★\n", W->next, ti, W->t[ti].name); return -1; }
    if (nbytes != W->t[ti].nbytes) { fprintf(stderr, "★%s 字节数 %llu != 登记 %llu★\n", W->t[ti].name, (unsigned long long)nbytes, (unsigned long long)W->t[ti].nbytes); return -1; }
    if (nbytes && fwrite(data, 1, nbytes, W->f) != nbytes) { fprintf(stderr, "★写失败 %s(盘满?)★\n", W->path); return -1; }
    for (uint64_t p = nbytes; p < ds4gw_pad(nbytes, W->align); p++) if (fputc(0, W->f) == EOF) return -1;
    W->next++;
    return 0;
}

static inline int ds4gw_end(ds4gw *W) {
    if (W->next != W->nt) { fprintf(stderr, "★GGUF 未写全: %d/%d★\n", W->next, W->nt); return -1; }
    int rc = (fflush(W->f) || fclose(W->f)) ? -1 : 0;
    W->f = NULL; free(W->kv); free(W->t); W->kv = NULL; W->t = NULL;
    return rc;
}
#endif
