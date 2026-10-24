/* ds4_gguf.c — GGUF v3 只读解析。谱系与金标见 ds4_gguf.h; 逐式转录 zlayer.c
 * gg_open(die 口径改为返回错误, 数值/字节布局逻辑不变)。 */
#include "ds4_gguf.h"
#include "ds4_quantfmt.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GG_FAIL(...) do { \
        if (err && errlen) snprintf(err, errlen, __VA_ARGS__); \
        ds4_gguf_close(g); \
        return -1; \
    } while (0)

static uint64_t gg_align_up(uint64_t x) { return (x + 31u) & ~31ull; }   /* GGUF 默认 alignment=32 */

/* 光标式读取(全在 mmap 上); 越界置 fail 位, 调用方统一检查 */
typedef struct { const uint8_t *p; const uint8_t *end; int fail; } gg_cur;

static void gg_rd(gg_cur *c, void *dst, size_t n) {
    if (c->fail || (size_t)(c->end - c->p) < n) { c->fail = 1; return; }
    memcpy(dst, c->p, n); c->p += n;
}
static uint32_t gg_u32(gg_cur *c) { uint32_t v = 0; gg_rd(c, &v, 4); return v; }
static uint64_t gg_u64(gg_cur *c) { uint64_t v = 0; gg_rd(c, &v, 8); return v; }
static void gg_skipstr(gg_cur *c) {
    uint64_t n = gg_u64(c);
    if (c->fail || (uint64_t)(c->end - c->p) < n) { c->fail = 1; return; }
    c->p += n;
}
/* KV 值按类型跳过(与 gguf-py 同一 13 类型表; 8=string, 9=array 递归) */
static void gg_skipval(gg_cur *c, uint32_t t) {
    static const int sz[13] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
    if (c->fail) return;
    if (t == 8) { gg_skipstr(c); return; }
    if (t == 9) {
        uint32_t et = gg_u32(c); uint64_t n = gg_u64(c);
        for (uint64_t i = 0; i < n && !c->fail; i++) gg_skipval(c, et);
        return;
    }
    if (t > 12 || sz[t] == 0) { c->fail = 1; return; }
    uint8_t tmp[8]; gg_rd(c, tmp, (size_t)sz[t]);
}

int ds4_gguf_open(ds4_gguf *g, const char *path, char *err, size_t errlen) {
    memset(g, 0, sizeof(*g));
    int fd = open(path, O_RDONLY);
    if (fd < 0) { if (err && errlen) snprintf(err, errlen, "打不开: %s", path); return -1; }
    struct stat st;
    if (fstat(fd, &st) || st.st_size <= 0) {
        close(fd);
        if (err && errlen) snprintf(err, errlen, "空文件: %s", path);
        return -1;
    }
    g->msz = (size_t)st.st_size;
    g->map = (const uint8_t *)mmap(NULL, g->msz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (g->map == MAP_FAILED) {
        g->map = NULL;
        if (err && errlen) snprintf(err, errlen, "mmap 失败: %s", path);
        return -1;
    }

    gg_cur c = { g->map, g->map + g->msz, 0 };
    uint32_t magic = gg_u32(&c), ver = gg_u32(&c);
    uint64_t n_t = gg_u64(&c), n_kv = gg_u64(&c);
    if (c.fail || !(magic == 0x46554747u && ver == 3))
        GG_FAIL("非 GGUF v3(magic=%08x ver=%u)", magic, ver);
    g->kv0 = (uint64_t)(c.p - g->map); g->nkv = n_kv;
    for (uint64_t i = 0; i < n_kv && !c.fail; i++) { gg_skipstr(&c); gg_skipval(&c, gg_u32(&c)); }
    if (c.fail) GG_FAIL("GGUF KV 段截断/类型不认识");

    g->nt = (int)n_t;
    g->t = (ds4_gguf_tensor *)calloc((size_t)(n_t ? n_t : 1), sizeof(ds4_gguf_tensor));
    if (!g->t) GG_FAIL("calloc 张量目录失败");
    for (uint64_t i = 0; i < n_t; i++) {
        ds4_gguf_tensor *t = &g->t[i];
        uint64_t nl = gg_u64(&c);
        if (c.fail || (uint64_t)(c.end - c.p) < nl) GG_FAIL("GGUF 张量名截断");
        t->name = (char *)malloc((size_t)nl + 1);
        if (!t->name) GG_FAIL("malloc 张量名失败");
        memcpy(t->name, c.p, (size_t)nl); t->name[nl] = 0; c.p += nl;
        t->nd = gg_u32(&c);
        if (c.fail || t->nd < 1 || t->nd > 4) GG_FAIL("GGUF 张量 %s 维数超范围", t->name);
        t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
        for (uint32_t d = 0; d < t->nd; d++) t->ne[d] = gg_u64(&c);
        t->type = gg_u32(&c);
        t->off = gg_u64(&c);
        if (c.fail) GG_FAIL("GGUF 张量目录截断");
    }
    g->data0 = gg_align_up((uint64_t)(c.p - g->map));
    return 0;
}

void ds4_gguf_close(ds4_gguf *g) {
    if (g->t) {
        for (int i = 0; i < g->nt; i++) free(g->t[i].name);
        free(g->t);
    }
    if (g->map) munmap((void *)g->map, g->msz);
    memset(g, 0, sizeof(*g));
}

const uint8_t *ds4_gguf_kv(const ds4_gguf *g, const char *key, uint32_t *type) {
    gg_cur c = { g->map + g->kv0, g->map + g->msz, 0 };
    const size_t kl = strlen(key);
    for (uint64_t i = 0; i < g->nkv && !c.fail; i++) {
        uint64_t n = gg_u64(&c);
        if (c.fail || (uint64_t)(c.end - c.p) < n) return NULL;
        const uint8_t *k = c.p; c.p += n;
        uint32_t t = gg_u32(&c);
        if (c.fail) return NULL;
        if (n == kl && !memcmp(k, key, kl)) { *type = t; return c.p; }
        gg_skipval(&c, t);
    }
    return NULL;
}

const ds4_gguf_tensor *ds4_gguf_find(const ds4_gguf *g, const char *name) {
    for (int i = 0; i < g->nt; i++)
        if (!strcmp(g->t[i].name, name)) return &g->t[i];
    return NULL;
}

const uint8_t *ds4_gguf_tensor_data(const ds4_gguf *g, const ds4_gguf_tensor *t,
                                    uint64_t *nbytes_out) {
    uint64_t blk, tsz;
    if (!ds4_ggt_geom(t->type, &blk, &tsz)) return NULL;
    uint64_t nelem = 1;
    for (int d = 0; d < 4; d++) nelem *= t->ne[d];
    if (nelem % blk) return NULL;
    const uint64_t nbytes = nelem / blk * tsz;
    if (g->data0 + t->off + nbytes > g->msz) return NULL;
    if (nbytes_out) *nbytes_out = nbytes;
    return g->map + g->data0 + t->off;
}
