/*
 * DeepSeek V4 Flash/Pro HF -> GGUF quantizer.
 *
 * This is a plain C, model-specific version of the DS4 quantization pipeline.
 * It deliberately keeps only the pieces needed by the DeepSeek V4 Flash and
 * Pro GGUF recipes used by this repository:
 *
 * - safetensors index/header loading;
 * - FP8 E4M3 + (F32 | E8M0) 128x128 block-scale dequantization, for dense
 *   tensors AND for routed experts on the Base checkpoint (whose experts are
 *   F8_E4M3 with an F32 block scale, not packed FP4);
 * - packed FP4 + E8M0 dequantization for routed experts on the packed release;
 * - local Q8_0, Q4_K, Q2_K, and IQ2_XXS quantization;
 * - GGUF metadata/tensor-order reuse from an existing template GGUF.
 *
 * The optional imatrix is the legacy llama.cpp binary .dat format emitted by
 * ds4's collector.  DS4 stores one packed vector per routed tensor, laid out as
 * n_experts consecutive per-expert importance vectors.  When no external
 * imatrix is supplied and IQ2_XXS requires one, this tool falls back to the
 * same synthetic weight-energy heuristic used by the old generator:
 * each column importance is sum(row[column]^2) over the dequantized weight.
 */

#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "quants.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#if defined(_WIN32)
#error "deepseek4-quantize.c currently targets POSIX systems"
#endif

#define DS4_KV_QUANTIZE_IMATRIX_FILE      "quantize.imatrix.file"
#define DS4_KV_QUANTIZE_IMATRIX_DATASET   "quantize.imatrix.dataset"
#define DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES "quantize.imatrix.entries_count"
#define DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS  "quantize.imatrix.chunks_count"
#define DS4_GGUF_DEFAULT_ALIGNMENT 32

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type;

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "error: %s %s: %s\n", what, path ? path : "", strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static char *path_join(const char *a, const char *b) {
    const size_t na = strlen(a);
    const size_t nb = strlen(b);
    const bool slash = na && a[na - 1] == '/';
    char *out = xmalloc(na + (slash ? 0 : 1) + nb + 1);
    memcpy(out, a, na);
    size_t pos = na;
    if (!slash) out[pos++] = '/';
    memcpy(out + pos, b, nb + 1);
    return out;
}

static bool str_starts(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool str_ends(const char *s, const char *suffix) {
    const size_t ns = strlen(s);
    const size_t nf = strlen(suffix);
    return ns >= nf && memcmp(s + ns - nf, suffix, nf) == 0;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open", path);
    if (fseeko(fp, 0, SEEK_END) != 0) die_errno("seek", path);
    off_t n = ftello(fp);
    if (n < 0) die_errno("tell", path);
    if (fseeko(fp, 0, SEEK_SET) != 0) die_errno("seek", path);
    char *buf = xmalloc((size_t)n + 1);
    if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) die_errno("read", path);
    buf[n] = '\0';
    fclose(fp);
    if (len_out) *len_out = (size_t)n;
    return buf;
}

static uint64_t read_u64_le_fp(FILE *fp, const char *what) {
    uint8_t b[8];
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

static uint32_t read_u32_le_fp(FILE *fp, const char *what) {
    uint32_t v;
    if (fread(&v, 1, sizeof(v), fp) != sizeof(v)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return v;
}

static int32_t read_i32_fp(FILE *fp, const char *what) {
    int32_t v;
    if (fread(&v, 1, sizeof(v), fp) != sizeof(v)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return v;
}

static uint16_t load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int64_t load_i64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return (int64_t)v;
}

static float load_f32_le(const uint8_t *p) {
    uint32_t b = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

/* =====
 * Minimal JSON tokenizer
 *
 * Safetensors uses ordinary JSON for the model index and per-shard headers.
 * We only need objects, arrays, strings, and primitive numbers; escaped tensor
 * names do not occur in the files produced by Hugging Face, so strings are
 * copied as raw UTF-8 slices after locating the closing quote.
 */

typedef enum {
    JT_OBJECT,
    JT_ARRAY,
    JT_STRING,
    JT_PRIMITIVE,
} json_type;

typedef struct {
    json_type type;
    int start;
    int end;
    int parent;
    int size;
} json_tok;

typedef struct {
    json_tok *v;
    int len;
    int cap;
    const char *js;
    int js_len;
} json_doc;

static int json_add(json_doc *d, json_type type, int start, int end, int parent) {
    if (d->len == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 4096;
        d->v = xrealloc(d->v, (size_t)d->cap * sizeof(d->v[0]));
    }
    int id = d->len++;
    d->v[id] = (json_tok){ .type = type, .start = start, .end = end, .parent = parent, .size = 0 };
    if (parent >= 0) d->v[parent].size++;
    return id;
}

static json_doc json_parse_text(const char *js, size_t len) {
    json_doc d = { .js = js, .js_len = (int)len };
    int parent = -1;
    for (int i = 0; i < (int)len; i++) {
        unsigned char c = (unsigned char)js[i];
        if (isspace(c) || c == ':' || c == ',') continue;
        if (c == '{' || c == '[') {
            parent = json_add(&d, c == '{' ? JT_OBJECT : JT_ARRAY, i, -1, parent);
            continue;
        }
        if (c == '}' || c == ']') {
            if (parent < 0) die("bad JSON: unmatched close");
            d.v[parent].end = i + 1;
            parent = d.v[parent].parent;
            continue;
        }
        if (c == '"') {
            int start = i + 1;
            i++;
            bool esc = false;
            for (; i < (int)len; i++) {
                if (esc) {
                    esc = false;
                } else if (js[i] == '\\') {
                    esc = true;
                } else if (js[i] == '"') {
                    break;
                }
            }
            if (i >= (int)len) die("bad JSON: unterminated string");
            json_add(&d, JT_STRING, start, i, parent);
            continue;
        }
        int start = i;
        while (i < (int)len && !isspace((unsigned char)js[i]) &&
               js[i] != ',' && js[i] != ']' && js[i] != '}') {
            i++;
        }
        json_add(&d, JT_PRIMITIVE, start, i, parent);
        i--;
    }
    if (parent != -1) die("bad JSON: unterminated object/array");
    return d;
}

static void json_free(json_doc *d) {
    free(d->v);
    memset(d, 0, sizeof(*d));
}

static bool json_tok_eq(const json_doc *d, int tok, const char *s) {
    const json_tok *t = &d->v[tok];
    const int n = t->end - t->start;
    return t->type == JT_STRING && (int)strlen(s) == n && memcmp(d->js + t->start, s, (size_t)n) == 0;
}

static char *json_strdup_tok(const json_doc *d, int tok) {
    const json_tok *t = &d->v[tok];
    return xstrndup(d->js + t->start, (size_t)(t->end - t->start));
}

static bool json_is_descendant(const json_doc *d, int tok, int parent) {
    for (int p = d->v[tok].parent; p >= 0; p = d->v[p].parent) {
        if (p == parent) return true;
    }
    return false;
}

static int json_skip(const json_doc *d, int tok) {
    int i = tok + 1;
    while (i < d->len && json_is_descendant(d, i, tok)) i++;
    return i;
}

static int json_obj_get(const json_doc *d, int obj, const char *key) {
    if (obj < 0 || d->v[obj].type != JT_OBJECT) return -1;
    for (int i = obj + 1; i < d->len && d->v[i].parent == obj;) {
        int k = i;
        int v = i + 1;
        if (v >= d->len || d->v[v].parent != obj) return -1;
        if (json_tok_eq(d, k, key)) return v;
        i = json_skip(d, v);
    }
    return -1;
}

static int64_t json_i64(const json_doc *d, int tok) {
    char tmp[64];
    const int n = d->v[tok].end - d->v[tok].start;
    if (n <= 0 || n >= (int)sizeof(tmp)) die("bad JSON integer");
    memcpy(tmp, d->js + d->v[tok].start, (size_t)n);
    tmp[n] = '\0';
    return strtoll(tmp, NULL, 10);
}

/* =====
 * Small string hash map
 */

typedef struct {
    char *key;
    int value;
} hslot;

typedef struct {
    hslot *slots;
    int cap;
} hmap;

static uint64_t fnv1a_str(const char *s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static void hmap_build(hmap *m, char **keys, int n) {
    int cap = 1;
    while (cap < n * 3) cap <<= 1;
    m->cap = cap ? cap : 2;
    m->slots = xcalloc((size_t)m->cap, sizeof(m->slots[0]));
    for (int i = 0; i < n; i++) {
        uint64_t h = fnv1a_str(keys[i]);
        int p = (int)(h & (uint64_t)(m->cap - 1));
        while (m->slots[p].key) p = (p + 1) & (m->cap - 1);
        m->slots[p].key = keys[i];
        m->slots[p].value = i;
    }
}

static int hmap_get(const hmap *m, const char *key) {
    if (!m->slots) return -1;
    uint64_t h = fnv1a_str(key);
    int p = (int)(h & (uint64_t)(m->cap - 1));
    while (m->slots[p].key) {
        if (strcmp(m->slots[p].key, key) == 0) return m->slots[p].value;
        p = (p + 1) & (m->cap - 1);
    }
    return -1;
}

static void hmap_free(hmap *m) {
    free(m->slots);
    memset(m, 0, sizeof(*m));
}

/* =====
 * safetensors database
 */

#define MAX_DIMS 8

typedef struct {
    char *dtype;
    int n_dims;
    int64_t shape[MAX_DIMS];
    uint64_t begin;
    uint64_t end;
} st_info;

typedef struct {
    char *name;
    char *file;
} weight_map_entry;

typedef struct {
    char *name;
    st_info info;
} tensor_entry;

typedef struct {
    char *file;
    char *path;
    uint64_t data_base;
    tensor_entry *tensors;
    int n_tensors;
    int cap_tensors;
    hmap tensor_map;
    FILE *fp;
    pthread_mutex_t lock;
    bool loaded;
} shard;

typedef struct {
    char *hf_dir;
    weight_map_entry *weights;
    int n_weights;
    hmap weight_map;
    shard *shards;
    int n_shards;
    int cap_shards;
    pthread_mutex_t lock;
} st_db;

typedef struct {
    char *dtype;
    int n_dims;
    int64_t shape[MAX_DIMS];
    uint8_t *data;
    size_t nbytes;
} st_value;

