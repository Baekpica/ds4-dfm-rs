/* mmap GGUF v3 metadata + identify dump. Copied from ds4.c at v0.6.5-dfm. */

#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define DS4_GGUF_MAGIC 0x46554747u

enum {
    GGUF_VALUE_UINT8 = 0,
    GGUF_VALUE_INT8 = 1,
    GGUF_VALUE_UINT16 = 2,
    GGUF_VALUE_INT16 = 3,
    GGUF_VALUE_UINT32 = 4,
    GGUF_VALUE_INT32 = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL = 7,
    GGUF_VALUE_STRING = 8,
    GGUF_VALUE_ARRAY = 9,
    GGUF_VALUE_UINT64 = 10,
    GGUF_VALUE_INT64 = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *ptr;
    uint64_t len;
} ds4_str;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    int err; /* 1 truncated 2 nest 3 type 4 array-too-large */
} ds4_cursor;

typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;

typedef struct {
    const uint8_t *map;
    uint64_t size;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    ds4_kv *kv;
} ds4_model;

static bool ds4_streq(ds4_str s, const char *z)
{
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static bool cursor_has(ds4_cursor *c, uint64_t n)
{
    if (n > c->size || c->pos > c->size - n) {
        if (!c->err) c->err = 1;
        return false;
    }
    return true;
}

static bool cursor_read(ds4_cursor *c, void *dst, uint64_t n)
{
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(ds4_cursor *c, uint64_t n)
{
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(ds4_cursor *c, uint32_t *v)
{
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(ds4_cursor *c, uint64_t *v)
{
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(ds4_cursor *c, ds4_str *s)
{
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t scalar_value_size(uint32_t type)
{
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(ds4_cursor *c, uint32_t type, int depth)
{
    if (depth > 8) {
        if (!c->err) c->err = 2;
        return false;
    }
    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);
    if (type == GGUF_VALUE_STRING) {
        ds4_str ignored;
        return cursor_string(c, &ignored);
    }
    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;
        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;
        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                if (!c->err) c->err = 4;
                return false;
            }
            return cursor_skip(c, len * item_size);
        }
        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }
    if (!c->err) c->err = 3;
    return false;
}

static const char *err_token(int err)
{
    switch (err) {
    case 2: return "nest";
    case 3: return "type";
    case 4: return "array-too-large";
    default: return "truncated";
    }
}

static const char *type_name(uint32_t t)
{
    switch (t) {
    case GGUF_VALUE_UINT8: return "UINT8";
    case GGUF_VALUE_INT8: return "INT8";
    case GGUF_VALUE_UINT16: return "UINT16";
    case GGUF_VALUE_INT16: return "INT16";
    case GGUF_VALUE_UINT32: return "UINT32";
    case GGUF_VALUE_INT32: return "INT32";
    case GGUF_VALUE_FLOAT32: return "FLOAT32";
    case GGUF_VALUE_BOOL: return "BOOL";
    case GGUF_VALUE_STRING: return "STRING";
    case GGUF_VALUE_ARRAY: return "ARRAY";
    case GGUF_VALUE_UINT64: return "UINT64";
    case GGUF_VALUE_INT64: return "INT64";
    case GGUF_VALUE_FLOAT64: return "FLOAT64";
    default: return "UNKNOWN";
    }
}

static ds4_cursor cursor_at(const ds4_model *m, uint64_t pos)
{
    ds4_cursor c;
    c.base = m->map;
    c.size = m->size;
    c.pos = pos;
    c.err = 0;
    return c;
}

static ds4_kv *model_find_kv(const ds4_model *m, const char *key)
{
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const ds4_model *m, const char *key, ds4_str *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u16(const ds4_model *m, const char *key, uint16_t *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT16) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_read(&c, out, sizeof(*out));
}

static bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static uint32_t split_count_of(const ds4_model *m)
{
    uint16_t s16 = 0;
    uint32_t s32 = 0;
    if (model_get_u16(m, "split.count", &s16)) return s16;
    if (model_get_u32(m, "split.count", &s32)) return s32;
    return 0;
}

static bool parse_metadata(ds4_model *m, ds4_cursor *c)
{
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) return false;
    m->alignment = 32;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];
        if (!cursor_string(c, &kv->key)) return false;
        if (!cursor_u32(c, &kv->type)) return false;
        kv->value_pos = c->pos;
        if (ds4_streq(kv->key, "general.alignment") && kv->type == GGUF_VALUE_UINT32) {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment = 0;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }
        if (!skip_value(c, kv->type, 0)) return false;
    }
    return true;
}

typedef struct {
    uint32_t n_layer, n_embd, n_vocab, n_head, n_head_kv, n_head_dim, n_value_dim;
    uint32_t n_rot, n_lora_q, n_lora_o, n_out_group, n_expert, n_expert_used;
    uint32_t n_ff_exp, n_expert_shared, n_hash_layer, n_swa, n_indexer_head;
    uint32_t n_indexer_head_dim, n_indexer_top_k, n_hc, n_hc_sinkhorn_iter;
} ds_dims;

static const char *const ds_keys[] = {
    "deepseek4.block_count",
    "deepseek4.embedding_length",
    "deepseek4.vocab_size",
    "deepseek4.attention.head_count",
    "deepseek4.attention.head_count_kv",
    "deepseek4.attention.key_length",
    "deepseek4.attention.value_length",
    "deepseek4.rope.dimension_count",
    "deepseek4.attention.q_lora_rank",
    "deepseek4.attention.output_lora_rank",
    "deepseek4.attention.output_group_count",
    "deepseek4.expert_count",
    "deepseek4.expert_used_count",
    "deepseek4.expert_feed_forward_length",
    "deepseek4.expert_shared_count",
    "deepseek4.hash_layer_count",
    "deepseek4.attention.sliding_window",
    "deepseek4.attention.indexer.head_count",
    "deepseek4.attention.indexer.key_length",
    "deepseek4.attention.indexer.top_k",
    "deepseek4.hyper_connection.count",
    "deepseek4.hyper_connection.sinkhorn_iterations",
};

static int read_ds_dims(const ds4_model *m, ds_dims *d, const char **missing)
{
    uint32_t *fields[] = {
        &d->n_layer, &d->n_embd, &d->n_vocab, &d->n_head, &d->n_head_kv,
        &d->n_head_dim, &d->n_value_dim, &d->n_rot, &d->n_lora_q, &d->n_lora_o,
        &d->n_out_group, &d->n_expert, &d->n_expert_used, &d->n_ff_exp,
        &d->n_expert_shared, &d->n_hash_layer, &d->n_swa, &d->n_indexer_head,
        &d->n_indexer_head_dim, &d->n_indexer_top_k, &d->n_hc,
        &d->n_hc_sinkhorn_iter,
    };
    memset(d, 0, sizeof(*d));
    for (size_t i = 0; i < sizeof(ds_keys) / sizeof(ds_keys[0]); i++) {
        if (!model_get_u32(m, ds_keys[i], fields[i])) {
            *missing = ds_keys[i];
            return 0;
        }
    }
    return 1;
}

static int dims_eq(const ds_dims *d,
                   uint32_t n_layer, uint32_t n_embd, uint32_t n_vocab,
                   uint32_t n_head, uint32_t n_head_kv, uint32_t n_head_dim,
                   uint32_t n_value_dim, uint32_t n_rot, uint32_t n_lora_q,
                   uint32_t n_lora_o, uint32_t n_out_group, uint32_t n_expert,
                   uint32_t n_expert_used, uint32_t n_ff_exp,
                   uint32_t n_expert_shared, uint32_t n_hash_layer, uint32_t n_swa,
                   uint32_t n_indexer_head, uint32_t n_indexer_head_dim,
                   uint32_t n_indexer_top_k, uint32_t n_hc,
                   uint32_t n_hc_sinkhorn_iter)
{
    return d->n_layer == n_layer && d->n_embd == n_embd && d->n_vocab == n_vocab &&
           d->n_head == n_head && d->n_head_kv == n_head_kv &&
           d->n_head_dim == n_head_dim && d->n_value_dim == n_value_dim &&
           d->n_rot == n_rot && d->n_lora_q == n_lora_q && d->n_lora_o == n_lora_o &&
           d->n_out_group == n_out_group && d->n_expert == n_expert &&
           d->n_expert_used == n_expert_used && d->n_ff_exp == n_ff_exp &&
           d->n_expert_shared == n_expert_shared && d->n_hash_layer == n_hash_layer &&
           d->n_swa == n_swa && d->n_indexer_head == n_indexer_head &&
           d->n_indexer_head_dim == n_indexer_head_dim &&
           d->n_indexer_top_k == n_indexer_top_k && d->n_hc == n_hc &&
           d->n_hc_sinkhorn_iter == n_hc_sinkhorn_iter;
}

static void identify(const ds4_model *m)
{
    ds4_str arch = {0};
    int have_arch = model_get_string(m, "general.architecture", &arch);
    if (!have_arch || ds4_streq(arch, "deepseek4")) {
        ds_dims d;
        const char *missing = NULL;
        if (!read_ds_dims(m, &d, &missing)) {
            printf("ERROR missing-key %s\n", missing);
            return;
        }
        if (dims_eq(&d, 43, 4096, 129280, 64, 1, 512, 512, 64, 1024, 1024, 8,
                    256, 6, 2048, 1, 3, 128, 64, 128, 512, 4, 20)) {
            printf("IDENTIFY DeepSeek V4 Flash family=0 variant=0\n");
            return;
        }
        if (dims_eq(&d, 61, 7168, 129280, 128, 1, 512, 512, 64, 1536, 1024, 16,
                    384, 6, 3072, 1, 3, 128, 64, 128, 1024, 4, 20)) {
            printf("IDENTIFY DeepSeek V4 Pro family=0 variant=1\n");
            return;
        }
        printf("IDENTIFY unsupported\n");
        return;
    }
    if (ds4_streq(arch, "exaone-moe")) {
        printf("IDENTIFY K-EXAONE 236B A23B family=3 variant=4\n");
        return;
    }
    if (ds4_streq(arch, "solar-open2")) {
        printf("IDENTIFY Solar Open2 250B family=1 variant=2\n");
        return;
    }
    if (ds4_streq(arch, "motif3")) {
        printf("IDENTIFY Motif-3 family=2 variant=3\n");
        return;
    }
    if (ds4_streq(arch, "dots3-note")) {
        printf("IDENTIFY dots3-note-prev family=4 variant=5\n");
        return;
    }
    if (ds4_streq(arch, "qwen4exp")) {
        printf("IDENTIFY Qwen3.8-Flash-Next family=5 variant=6\n");
        return;
    }
    if (ds4_streq(arch, "glm5-next")) {
        printf("IDENTIFY GLM 5.3 Flash family=6 variant=7\n");
        return;
    }
    printf("ERROR unsupported-arch %.*s\n", (int)arch.len, arch.ptr);
}

static void dump_and_identify(const ds4_model *m)
{
    printf("HEADER version=%u n_tensors=%llu n_kv=%llu alignment=%llu\n",
           m->version,
           (unsigned long long)m->n_tensors,
           (unsigned long long)m->n_kv,
           (unsigned long long)m->alignment);
    for (uint64_t i = 0; i < m->n_kv; i++) {
        printf("KV %.*s %s\n", (int)m->kv[i].key.len, m->kv[i].key.ptr,
               type_name(m->kv[i].type));
    }
    printf("SPLIT %u\n", split_count_of(m));
    ds4_str arch = {0};
    if (!model_get_string(m, "general.architecture", &arch)) {
        printf("ARCH missing\n");
    } else {
        printf("ARCH %.*s\n", (int)arch.len, arch.ptr);
    }
    identify(m);
}

int main(int argc, char **argv)
{
    ds4_model m;
    ds4_cursor c;
    int fd;
    struct stat st;
    void *map;
    uint32_t magic;

    if (argc != 2) {
        fprintf(stderr, "usage: catalog_c_oracle PATH\n");
        return 2;
    }
    memset(&m, 0, sizeof(m));
    fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        printf("ERROR io\n");
        return 0;
    }
    if (fstat(fd, &st) < 0) {
        close(fd);
        printf("ERROR io\n");
        return 0;
    }
    if (st.st_size < 32) {
        close(fd);
        printf("ERROR too-small\n");
        return 0;
    }
    map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        printf("ERROR io\n");
        return 0;
    }
    m.map = map;
    m.size = (uint64_t)st.st_size;
    c.base = m.map;
    c.size = m.size;
    c.pos = 0;
    c.err = 0;
    if (!cursor_u32(&c, &magic)) {
        printf("ERROR %s\n", err_token(c.err));
        goto done;
    }
    if (magic != DS4_GGUF_MAGIC) {
        printf("ERROR not-gguf\n");
        goto done;
    }
    if (!cursor_u32(&c, &m.version) || !cursor_u64(&c, &m.n_tensors) ||
        !cursor_u64(&c, &m.n_kv)) {
        printf("ERROR %s\n", err_token(c.err));
        goto done;
    }
    if (m.version != 3) {
        printf("ERROR unsupported-version %u\n", m.version);
        goto done;
    }
    if (!parse_metadata(&m, &c)) {
        printf("ERROR %s\n", err_token(c.err));
        goto done;
    }
    dump_and_identify(&m);
done:
    free(m.kv);
    munmap(map, (size_t)st.st_size);
    close(fd);
    return 0;
}
