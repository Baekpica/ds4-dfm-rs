/* v41_st_write.h — safetensors 单分片【写盘器】(2026-09-12, V4.1 量化器用)。
 *
 * 【用法】先 plan 全部张量(名/类型/形状/字节数), 再 write_header, 再按 plan 顺序逐个 write
 * 数据, 最后 end。两段式的原因: safetensors 头在文件最前且含每个张量的 data_offsets,
 * 而量化产物的字节数只由形状决定(索引流 = rows × bytes_row 等), 所以量化前就能把头算死,
 * 数据边算边追加写, 不需要 2.7 GB 的分片在内存里攒齐再落盘, 也不用写完再搬一遍。
 *
 * 【.part 改名】写到 <path>.part, end 时 rename。断点续跑只认最终名 —— 写一半被杀的分片
 * 永远不会被当成完整产物(铁律: 里程碑须有盘上证据, 半成品不算)。
 *
 * 头 JSON 8 字节对齐用空格填(safetensors 允许尾随 0x20)。 */
#ifndef V41_ST_WRITE_H
#define V41_ST_WRITE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    char name[192];
    char dtype[12];
    int nd;
    int64_t shape[4];
    uint64_t nbytes;
} v41_out_t;

typedef struct {
    FILE *f;
    char path[4300], tmp[4310];
    v41_out_t *t; int nt, cap;
    uint64_t data0;       /* 数据区起点 = 8 + 头长 */
    int next;             /* 下一个该写的张量序号 */
    uint64_t total;       /* 数据区总字节 */
} v41_stw;

static int v41_stw_begin(v41_stw *W, const char *path) {
    memset(W, 0, sizeof *W);
    snprintf(W->path, sizeof W->path, "%s", path);
    snprintf(W->tmp, sizeof W->tmp, "%s.part", path);
    W->f = fopen(W->tmp, "wb");
    if (!W->f) { fprintf(stderr, "★写不了 %s★\n", W->tmp); return -1; }
    return 0;
}

static int v41_stw_plan(v41_stw *W, const char *name, const char *dtype, int nd, const int64_t *shape, uint64_t nbytes) {
    if (W->nt == W->cap) { W->cap = W->cap ? W->cap * 2 : 1024; W->t = (v41_out_t *)realloc(W->t, sizeof(v41_out_t) * W->cap); }
    v41_out_t *T = &W->t[W->nt++];
    memset(T, 0, sizeof *T);
    snprintf(T->name, sizeof T->name, "%s", name);
    snprintf(T->dtype, sizeof T->dtype, "%s", dtype);
    T->nd = nd;
    for (int i = 0; i < nd && i < 4; i++) T->shape[i] = shape[i];
    T->nbytes = nbytes;
    W->total += nbytes;
    return W->nt - 1;
}

/* meta: 已经拼好的 JSON 对象体(不含外层花括号), 可 NULL */
static int v41_stw_write_header(v41_stw *W, const char *meta) {
    size_t cap = (size_t)W->nt * 256 + (meta ? strlen(meta) : 0) + 64;
    char *h = (char *)malloc(cap);
    size_t L = 0;
    L += (size_t)snprintf(h + L, cap - L, "{");
    if (meta) L += (size_t)snprintf(h + L, cap - L, "\"__metadata__\":{%s},", meta);
    uint64_t off = 0;
    for (int i = 0; i < W->nt; i++) {
        v41_out_t *T = &W->t[i];
        L += (size_t)snprintf(h + L, cap - L, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", i ? "," : "", T->name, T->dtype);
        for (int d = 0; d < T->nd; d++) L += (size_t)snprintf(h + L, cap - L, "%s%lld", d ? "," : "", (long long)T->shape[d]);
        L += (size_t)snprintf(h + L, cap - L, "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)off, (unsigned long long)(off + T->nbytes));
        off += T->nbytes;
        if (L + 300 > cap) { cap *= 2; h = (char *)realloc(h, cap); }
    }
    L += (size_t)snprintf(h + L, cap - L, "}");
    while (L % 8) h[L++] = ' ';
    uint64_t hl = L;
    if (fwrite(&hl, 8, 1, W->f) != 1 || fwrite(h, 1, L, W->f) != L) { free(h); fprintf(stderr, "★头写失败 %s★\n", W->tmp); return -1; }
    free(h);
    W->data0 = 8 + hl;
    return 0;
}

/* 必须按 plan 顺序写; 字节数与 plan 不符就硬停 —— 偏一个字节整个分片全错位 */
static int v41_stw_write(v41_stw *W, int ti, const void *data, uint64_t nbytes) {
    if (ti != W->next) { fprintf(stderr, "★写盘顺序错: 期望 #%d 来了 #%d (%s)★\n", W->next, ti, W->t[ti].name); return -1; }
    if (nbytes != W->t[ti].nbytes) { fprintf(stderr, "★%s 字节数 %llu != plan %llu★\n", W->t[ti].name, (unsigned long long)nbytes, (unsigned long long)W->t[ti].nbytes); return -1; }
    if (nbytes && fwrite(data, 1, nbytes, W->f) != nbytes) { fprintf(stderr, "★写失败 %s(盘满?)★\n", W->tmp); return -1; }
    W->next++;
    return 0;
}

static int v41_stw_end(v41_stw *W) {
    if (W->next != W->nt) { fprintf(stderr, "★分片未写全: %d/%d★\n", W->next, W->nt); return -1; }
    if (fflush(W->f) || fclose(W->f)) { fprintf(stderr, "★关闭失败 %s★\n", W->tmp); return -1; }
    W->f = NULL;
    if (rename(W->tmp, W->path)) { fprintf(stderr, "★改名失败 %s★\n", W->path); return -1; }
    free(W->t); W->t = NULL;
    return 0;
}

static void v41_stw_abort(v41_stw *W) {
    if (W->f) fclose(W->f);
    remove(W->tmp);
    free(W->t); W->t = NULL;
}

#endif
