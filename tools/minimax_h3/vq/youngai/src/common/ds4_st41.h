/* ds4_st41.h — V4.1 safetensors 多分片【只读索引】(2026-09-12; 原 quantize/v41_st_index.h, 进生产链收口到 src/common —— 格式基元全仓只此一份, 量化器与 GGUF 转换器共用)。
 *
 * 【干什么】扫一个 HF 目录下全部 *.safetensors 的 JSON 头, 建"张量名 → (分片, 偏移, dtype,
 * 形状)"的有序表, 之后按名 bsearch 取数据指针(分片按需 mmap, 用完可整片卸掉)。
 *
 * 【为什么不用 st_locate.h】那份是探针用的: 每找一个张量重扫 48 个分片头。量化器要取
 * 96085 个张量, 重扫 = 48×96085 次头解析, 光这一步就要几十分钟。索引只建一次(0.1 s)。
 * 【为什么不用 deepseek4-quantize_p2 的 st_db】它与 V4 量化器的 json 解析器/die/imatrix
 * 绑成一个 TU, 且 db_read 带 V4 的 dequant 语义(FP8 128×128)。这里只要"定位", 不要解码。
 *
 * 【分片卸载】整模 510 GB 走 mmap, 扫过的页都算进 RSS/页缓存。按层推进时只需当前几个
 * 分片, 所以提供 v41_st_release_idle(): 把本层没碰过的分片 munmap —— 不这么做, 跑到后半
 * 段 RSS 看着 300 GB+, 看门狗会误杀(fable5 09-12: RSS 对 mmap 是虚高的)。 */
#ifndef DS4_COMMON_ST41_H
#define DS4_COMMON_ST41_H
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct {
    char *name;
    int shard;
    uint64_t off;          /* 相对该分片数据区起点 */
    uint64_t nbytes;
    char dtype[12];
    int nd;
    int64_t shape[4];
} v41_st_ent;

typedef struct {
    char *file;            /* 文件名(不含目录) */
    char *path;
    uint8_t *map;          /* NULL = 未 mmap */
    size_t msz;
    uint64_t data0;        /* 8 + header_len */
    int touched;           /* 本轮被取过数据 */
} v41_st_shard;

typedef struct {
    v41_st_shard *sh; int nsh;
    v41_st_ent *e; int ne, cap;
} v41_st;

static const char *v41_ws(const char *p) { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++; return p; }

/* 解析一个张量的 {...} 段: "dtype":"X","shape":[...],"data_offsets":[a,b] */
static int v41_parse_ent(const char *seg, const char *end, v41_st_ent *E) {
    const char *p = strstr(seg, "\"dtype\"");
    if (!p || p > end) return -1;
    p = v41_ws(strchr(p + 7, ':') + 1);
    if (*p != '"') return -1;
    p++; int i = 0;
    while (*p && *p != '"' && i < 11) E->dtype[i++] = *p++;
    E->dtype[i] = 0;
    p = strstr(seg, "\"shape\"");
    if (!p || p > end) return -1;
    p = v41_ws(strchr(p + 7, ':') + 1);
    if (*p != '[') return -1;
    p++; E->nd = 0;
    for (;;) {
        p = v41_ws(p);
        if (*p == ']') { p++; break; }
        char *q; long long v = strtoll(p, &q, 10);
        if (q == p) return -1;
        if (E->nd < 4) E->shape[E->nd] = v;
        E->nd++; p = v41_ws(q);
        if (*p == ',') p++;
    }
    if (E->nd > 4) return -1;
    p = strstr(seg, "\"data_offsets\"");
    if (!p || p > end) return -1;
    p = v41_ws(strchr(p + 14, ':') + 1);
    if (*p != '[') return -1;
    char *q; long long a = strtoll(p + 1, &q, 10);
    if (*q != ',') return -1;
    long long b = strtoll(q + 1, &q, 10);
    E->off = (uint64_t)a; E->nbytes = (uint64_t)(b - a);
    return 0;
}

static int v41_ent_cmp(const void *a, const void *b) {
    return strcmp(((const v41_st_ent *)a)->name, ((const v41_st_ent *)b)->name);
}
static int v41_str_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int v41_st_scan_shard(v41_st *S, int si) {
    v41_st_shard *sh = &S->sh[si];
    FILE *f = fopen(sh->path, "rb");
    if (!f) { fprintf(stderr, "★打不开 %s★\n", sh->path); return -1; }
    uint64_t hlen = 0;
    if (fread(&hlen, 8, 1, f) != 1 || hlen < 2 || hlen > (1u << 30)) { fclose(f); fprintf(stderr, "★%s 头长不合理★\n", sh->file); return -1; }
    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr || fread(hdr, 1, hlen, f) != hlen) { fclose(f); free(hdr); fprintf(stderr, "★%s 头读不满★\n", sh->file); return -1; }
    hdr[hlen] = 0; fclose(f);
    sh->data0 = 8 + hlen;
    /* 顶层对象: 逐键扫描。值要么是张量对象 {..}, 要么是 __metadata__ 的 {..}(跳过)。 */
    const char *p = v41_ws(hdr);
    if (*p == '{') p++;
    while (*p) {
        p = v41_ws(p);
        if (*p != '"') { if (!*p || *p == '}') break; p++; continue; }
        const char *k0 = ++p;
        while (*p && *p != '"') p++;
        size_t klen = (size_t)(p - k0);
        if (*p == '"') p++;
        p = v41_ws(p); if (*p != ':') continue; p = v41_ws(p + 1);
        if (*p != '{') { while (*p && *p != ',') p++; if (*p == ',') p++; continue; }
        /* 找匹配的 '}'(张量段内无嵌套对象, __metadata__ 也是平的字符串表) */
        const char *seg = p; int depth = 0;
        while (*p) { if (*p == '{') depth++; else if (*p == '}') { depth--; if (!depth) break; } p++; }
        const char *end = p; if (*p == '}') p++;
        if (klen == 12 && !strncmp(k0, "__metadata__", 12)) continue;
        if (S->ne == S->cap) { S->cap = S->cap ? S->cap * 2 : 4096; S->e = (v41_st_ent *)realloc(S->e, sizeof(v41_st_ent) * S->cap); }
        v41_st_ent *E = &S->e[S->ne];
        memset(E, 0, sizeof *E);
        E->name = (char *)malloc(klen + 1); memcpy(E->name, k0, klen); E->name[klen] = 0;
        E->shard = si;
        if (v41_parse_ent(seg, end, E)) { fprintf(stderr, "★%s 里 %s 的头解析失败★\n", sh->file, E->name); free(hdr); return -1; }
        S->ne++;
    }
    free(hdr);
    return 0;
}

/* 打开目录: 扫全部分片头。成功 0。 */
static int v41_st_open(v41_st *S, const char *dir) {
    memset(S, 0, sizeof *S);
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "★目录打不开 %s★\n", dir); return -1; }
    char **names = NULL; int n = 0, cap = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        const char *s = strstr(de->d_name, ".safetensors");
        if (!s || s[12]) continue;
        if (n == cap) { cap = cap ? cap * 2 : 64; names = (char **)realloc(names, sizeof(char *) * cap); }
        names[n++] = strdup(de->d_name);
    }
    closedir(d);
    if (!n) { fprintf(stderr, "★%s 下没有 .safetensors★\n", dir); return -1; }
    qsort(names, n, sizeof(char *), v41_str_cmp);      /* 分片号有序, 日志可读 */
    S->sh = (v41_st_shard *)calloc(n, sizeof(v41_st_shard)); S->nsh = n;
    for (int i = 0; i < n; i++) {
        S->sh[i].file = names[i];
        size_t L = strlen(dir) + strlen(names[i]) + 2;
        S->sh[i].path = (char *)malloc(L);
        snprintf(S->sh[i].path, L, "%s/%s", dir, names[i]);
        if (v41_st_scan_shard(S, i)) return -1;
    }
    free(names);
    qsort(S->e, S->ne, sizeof(v41_st_ent), v41_ent_cmp);
    for (int i = 1; i < S->ne; i++)
        if (!strcmp(S->e[i].name, S->e[i - 1].name)) { fprintf(stderr, "★张量名重复: %s★\n", S->e[i].name); return -1; }
    return 0;
}

static const v41_st_ent *v41_st_find(const v41_st *S, const char *name) {
    v41_st_ent key; key.name = (char *)name;
    return (const v41_st_ent *)bsearch(&key, S->e, S->ne, sizeof(v41_st_ent), v41_ent_cmp);
}

/* 取数据指针(分片按需 mmap)。失败 NULL。 */
static const uint8_t *v41_st_data(v41_st *S, const v41_st_ent *E) {
    v41_st_shard *sh = &S->sh[E->shard];
    if (!sh->map) {
        int fd = open(sh->path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "★打不开 %s★\n", sh->path); return NULL; }
        struct stat st; fstat(fd, &st);
        sh->msz = (size_t)st.st_size;
        sh->map = (uint8_t *)mmap(NULL, sh->msz, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (sh->map == MAP_FAILED) { sh->map = NULL; fprintf(stderr, "★mmap 失败 %s★\n", sh->path); return NULL; }
    }
    if (sh->data0 + E->off + E->nbytes > sh->msz) { fprintf(stderr, "★%s 越界 %s★\n", E->name, sh->file); return NULL; }
    sh->touched = 1;
    return sh->map + sh->data0 + E->off;
}

/* 卸掉本轮没碰过的分片, 并把"碰过"标记清零供下一轮。 */
static void v41_st_release_idle(v41_st *S) {
    for (int i = 0; i < S->nsh; i++) {
        v41_st_shard *sh = &S->sh[i];
        if (sh->map && !sh->touched) { munmap(sh->map, sh->msz); sh->map = NULL; }
        sh->touched = 0;
    }
}

static void v41_st_close(v41_st *S) {
    for (int i = 0; i < S->nsh; i++) {
        if (S->sh[i].map) munmap(S->sh[i].map, S->sh[i].msz);
        free(S->sh[i].file); free(S->sh[i].path);
    }
    for (int i = 0; i < S->ne; i++) free(S->e[i].name);
    free(S->sh); free(S->e);
    memset(S, 0, sizeof *S);
}

static int64_t v41_st_numel(const v41_st_ent *E) {
    int64_t n = 1;
    for (int i = 0; i < E->nd; i++) n *= E->shape[i];
    return n;
}

#endif
