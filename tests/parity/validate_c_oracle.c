/* Host config_validate oracle. Do not include ds4.c.
 * Token lines match crates/ds4-core/src/validate.rs dump_validate. */

#include <fcntl.h>
#include <math.h>
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
    int err;
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

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array;

static char g_tok[256];

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

static bool cursor_u32(ds4_cursor *c, uint32_t *v) { return cursor_read(c, v, 4); }
static bool cursor_u64(ds4_cursor *c, uint64_t *v) { return cursor_read(c, v, 8); }

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
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}

static bool skip_value(ds4_cursor *c, uint32_t type, int depth)
{
    if (depth > 8) { if (!c->err) c->err = 2; return false; }
    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);
    if (type == GGUF_VALUE_STRING) { ds4_str ign; return cursor_string(c, &ign); }
    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type; uint64_t len;
        if (!cursor_u32(c, &item_type) || !cursor_u64(c, &len)) return false;
        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) { if (!c->err) c->err = 4; return false; }
            return cursor_skip(c, len * item_size);
        }
        for (uint64_t i = 0; i < len; i++)
            if (!skip_value(c, item_type, depth + 1)) return false;
        return true;
    }
    if (!c->err) c->err = 3;
    return false;
}

static const char *err_token(int err)
{
    if (err == 2) return "nest";
    if (err == 3) return "type";
    if (err == 4) return "array-too-large";
    return "truncated";
}

static ds4_cursor cursor_at(const ds4_model *m, uint64_t pos)
{
    ds4_cursor c;
    c.base = m->map; c.size = m->size; c.pos = pos; c.err = 0;
    return c;
}

static ds4_kv *model_find_kv(const ds4_model *m, const char *key)
{
    for (uint64_t i = 0; i < m->n_kv; i++)
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    return NULL;
}

static bool model_get_string(const ds4_model *m, const char *key, ds4_str *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_u64_compat(const ds4_model *m, const char *key, uint64_t *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) return cursor_u64(&c, out);
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

static bool model_get_f32_compat(const ds4_model *m, const char *key, float *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) return cursor_read(&c, out, 4);
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0;
        if (!cursor_read(&c, &v, 8)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, 4)) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

static bool model_get_bool(const ds4_model *m, const char *key, bool *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    uint8_t v = 0;
    if (!cursor_read(&c, &v, 1)) return false;
    *out = v != 0;
    return true;
}

static bool model_get_array(const ds4_model *m, const char *key, ds4_array *out)
{
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type) || !cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

static bool parse_metadata(ds4_model *m, ds4_cursor *c)
{
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) return false;
    m->alignment = 32;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];
        if (!cursor_string(c, &kv->key) || !cursor_u32(c, &kv->type)) return false;
        kv->value_pos = c->pos;
        if (!skip_value(c, kv->type, 0)) return false;
    }
    return true;
}

static int failk(const char *kind, const char *k)
{
    snprintf(g_tok, sizeof g_tok, "%s %s", kind, k);
    return 1;
}

static int faill(const char *kind, uint32_t n)
{
    snprintf(g_tok, sizeof g_tok, "%s %u", kind, n);
    return 1;
}

static int req_u32(const ds4_model *m, const char *key, uint32_t *o)
{
    if (!model_get_u32(m, key, o)) return failk("missing-key", key);
    return 0;
}

static int req_u64c(const ds4_model *m, const char *key, uint64_t *o)
{
    if (!model_get_u64_compat(m, key, o)) return failk("missing-key", key);
    return 0;
}

static int req_f32(const ds4_model *m, const char *key, float *o)
{
    if (!model_get_f32_compat(m, key, o)) return failk("missing-key", key);
    return 0;
}

static int req_bool(const ds4_model *m, const char *key, bool *o)
{
    if (!model_get_bool(m, key, o)) return failk("missing-key", key);
    return 0;
}

static int exp_u32(const char *name, uint32_t got, uint32_t want)
{
    return got == want ? 0 : failk("mismatch", name);
}

static int exp_u64(const char *name, uint64_t got, uint64_t want)
{
    return got == want ? 0 : failk("mismatch-u64", name);
}

static int exp_f32(const char *name, float got, float want)
{
    float scale = fabsf(want) > 1.0f ? fabsf(want) : 1.0f;
    return fabsf(got - want) <= scale * 1.0e-6f ? 0 : failk("mismatch-f32", name);
}

static int exp_bool(const char *name, bool got, bool want)
{
    return got == want ? 0 : failk("mismatch-bool", name);
}

static uint32_t flash_compress(uint32_t il)
{
    if (il < 2) return 0;
    return (il & 1u) == 0 ? 4u : 128u;
}

static uint32_t pro_compress(uint32_t il)
{
    if (il < 2) return 128u;
    return (il & 1u) == 0 ? 4u : 128u;
}

static int validate_compress(const ds4_model *m, uint32_t n_layer, int pro)
{
    const char *key = "deepseek4.attention.compress_ratios";
    ds4_array arr;
    if (!model_get_array(m, key, &arr)) return failk("missing-array", key);
    if (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)
        return failk("array-type", key);
    if (arr.len < n_layer) return failk("array-short", key);
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < n_layer; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) { snprintf(g_tok, sizeof g_tok, "truncated"); return 1; }
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, 4)) { snprintf(g_tok, sizeof g_tok, "truncated"); return 1; }
            if (v < 0) { snprintf(g_tok, sizeof g_tok, "negative-array"); return 1; }
            got = (uint32_t)v;
        }
        uint32_t want = pro ? pro_compress(il) : flash_compress(il);
        if (got != want) return faill("compress-ratio", il);
    }
    return 0;
}

static int validate_swiglu(const ds4_model *m, uint32_t n_layer, float want)
{
    const char *key = "deepseek4.swiglu_clamp_exp";
    ds4_array arr;
    if (!model_get_array(m, key, &arr)) return failk("missing-array", key);
    if (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)
        return failk("array-type", key);
    if (arr.len < n_layer) return failk("array-short", key);
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < n_layer; i++) {
        float got = 0;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, 4)) { snprintf(g_tok, sizeof g_tok, "truncated"); return 1; }
        } else {
            double v = 0;
            if (!cursor_read(&c, &v, 8)) { snprintf(g_tok, sizeof g_tok, "truncated"); return 1; }
            got = (float)v;
        }
        if (exp_f32("swiglu_clamp_exp", got, want)) return 1;
    }
    return 0;
}

static int validate_deepseek(const ds4_model *m, int pro)
{
    const uint32_t n_layer = pro ? 61u : 43u;
    const uint32_t n_embd = pro ? 7168u : 4096u;
    const uint32_t n_head = pro ? 128u : 64u;
    const uint32_t n_lora_q = pro ? 1536u : 1024u;
    const uint32_t n_out = pro ? 16u : 8u;
    const uint32_t n_expert = pro ? 384u : 256u;
    const uint32_t n_ff = pro ? 3072u : 2048u;
    const uint32_t n_topk = pro ? 1024u : 512u;
    const float ews = pro ? 2.5f : 1.5f;
    uint32_t v;
    if (req_u32(m, "deepseek4.block_count", &v) ||
        req_u32(m, "deepseek4.embedding_length", &v) ||
        req_u32(m, "deepseek4.vocab_size", &v)) return 1;
    {
        uint32_t n_embd_g, n_vocab, n_h, n_hkv, n_hd, n_vd, n_rot, n_lq, n_lo, n_og;
        uint32_t n_ex, n_eu, n_ffg, n_es, n_hl, n_swa, n_ih, n_ihd, n_itk, n_hc, n_hcs, n_ly;
        uint32_t grp = 0, gused = 0;
        if (req_u32(m, "deepseek4.block_count", &n_ly) ||
            req_u32(m, "deepseek4.embedding_length", &n_embd_g) ||
            req_u32(m, "deepseek4.vocab_size", &n_vocab) ||
            req_u32(m, "deepseek4.attention.head_count", &n_h) ||
            req_u32(m, "deepseek4.attention.head_count_kv", &n_hkv) ||
            req_u32(m, "deepseek4.attention.key_length", &n_hd) ||
            req_u32(m, "deepseek4.attention.value_length", &n_vd) ||
            req_u32(m, "deepseek4.rope.dimension_count", &n_rot) ||
            req_u32(m, "deepseek4.attention.q_lora_rank", &n_lq) ||
            req_u32(m, "deepseek4.attention.output_lora_rank", &n_lo) ||
            req_u32(m, "deepseek4.attention.output_group_count", &n_og) ||
            req_u32(m, "deepseek4.expert_count", &n_ex) ||
            req_u32(m, "deepseek4.expert_used_count", &n_eu) ||
            req_u32(m, "deepseek4.expert_feed_forward_length", &n_ffg) ||
            req_u32(m, "deepseek4.expert_shared_count", &n_es) ||
            req_u32(m, "deepseek4.hash_layer_count", &n_hl) ||
            req_u32(m, "deepseek4.attention.sliding_window", &n_swa) ||
            req_u32(m, "deepseek4.attention.indexer.head_count", &n_ih) ||
            req_u32(m, "deepseek4.attention.indexer.key_length", &n_ihd) ||
            req_u32(m, "deepseek4.attention.indexer.top_k", &n_itk) ||
            req_u32(m, "deepseek4.hyper_connection.count", &n_hc) ||
            req_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations", &n_hcs))
            return 1;
        (void)model_get_u32(m, "deepseek4.expert_group_count", &grp);
        (void)model_get_u32(m, "deepseek4.expert_group_used_count", &gused);
        if (exp_u32("embedding_length", n_embd_g, n_embd) ||
            exp_u32("vocab_size", n_vocab, 129280) ||
            exp_u32("attention.head_count", n_h, n_head) ||
            exp_u32("attention.key_length", n_hd, 512) ||
            exp_u32("attention.head_count_kv", n_hkv, 1) ||
            exp_u32("attention.value_length", n_vd, 512) ||
            exp_u32("rope.dimension_count", n_rot, 64) ||
            exp_u32("attention.output_group_count", n_og, n_out) ||
            exp_u32("attention.q_lora_rank", n_lq, n_lora_q) ||
            exp_u32("attention.output_lora_rank", n_lo, 1024) ||
            exp_u32("expert_count", n_ex, n_expert) ||
            exp_u32("expert_used_count", n_eu, 6) ||
            exp_u32("expert_feed_forward_length", n_ffg, n_ff) ||
            exp_u32("expert_shared_count", n_es, 1) ||
            exp_u32("hash_layer_count", n_hl, 3) ||
            exp_u32("expert_group_count", grp, 0) ||
            exp_u32("expert_group_used_count", gused, 0) ||
            exp_u32("attention.sliding_window", n_swa, 128) ||
            exp_u32("attention.indexer.head_count", n_ih, 64) ||
            exp_u32("attention.indexer.key_length", n_ihd, 128) ||
            exp_u32("attention.indexer.top_k", n_itk, n_topk) ||
            exp_u32("hyper_connection.count", n_hc, 4) ||
            exp_u32("hyper_connection.sinkhorn_iterations", n_hcs, 20) ||
            exp_u32("block_count", n_ly, n_layer))
            return 1;
        if (validate_compress(m, n_layer, pro) || validate_swiglu(m, n_layer, 10.0f))
            return 1;
        {
            uint64_t rope_orig = 65536;
            float f, scale = 16.0f, yf = 32.0f, ys = 1.0f;
            bool b;
            (void)model_get_u64_compat(m, "deepseek4.rope.scaling.original_context_length", &rope_orig);
            if (exp_u64("rope.scaling.original_context_length", rope_orig, 65536)) return 1;
            if (req_f32(m, "deepseek4.rope.freq_base", &f) || exp_f32("rope.freq_base", f, 10000.0f))
                return 1;
            (void)model_get_f32_compat(m, "deepseek4.rope.scaling.factor", &scale);
            if (exp_f32("rope.scaling.factor", scale, 16.0f)) return 1;
            (void)model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_fast", &yf);
            if (exp_f32("rope.scaling.yarn_beta_fast", yf, 32.0f)) return 1;
            (void)model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_slow", &ys);
            if (exp_f32("rope.scaling.yarn_beta_slow", ys, 1.0f)) return 1;
            if (req_f32(m, "deepseek4.attention.compress_rope_freq_base", &f) ||
                exp_f32("attention.compress_rope_freq_base", f, 160000.0f)) return 1;
            if (req_f32(m, "deepseek4.expert_weights_scale", &f) ||
                exp_f32("expert_weights_scale", f, ews)) return 1;
            if (req_f32(m, "deepseek4.attention.layer_norm_rms_epsilon", &f) ||
                exp_f32("attention.layer_norm_rms_epsilon", f, 1.0e-6f)) return 1;
            if (req_f32(m, "deepseek4.hyper_connection.epsilon", &f) ||
                exp_f32("hyper_connection.epsilon", f, 1.0e-6f)) return 1;
            if (req_bool(m, "deepseek4.expert_weights_norm", &b) ||
                exp_bool("expert_weights_norm", b, true)) return 1;
        }
    }
    return 0;
}

/* Flash/Pro identify dims — same as catalog_c_oracle. */
static const char *const ds_keys[] = {
    "deepseek4.block_count", "deepseek4.embedding_length", "deepseek4.vocab_size",
    "deepseek4.attention.head_count", "deepseek4.attention.head_count_kv",
    "deepseek4.attention.key_length", "deepseek4.attention.value_length",
    "deepseek4.rope.dimension_count", "deepseek4.attention.q_lora_rank",
    "deepseek4.attention.output_lora_rank", "deepseek4.attention.output_group_count",
    "deepseek4.expert_count", "deepseek4.expert_used_count",
    "deepseek4.expert_feed_forward_length", "deepseek4.expert_shared_count",
    "deepseek4.hash_layer_count", "deepseek4.attention.sliding_window",
    "deepseek4.attention.indexer.head_count", "deepseek4.attention.indexer.key_length",
    "deepseek4.attention.indexer.top_k", "deepseek4.hyper_connection.count",
    "deepseek4.hyper_connection.sinkhorn_iterations",
};

static int read_ds_select(const ds4_model *m, int *pro)
{
    uint32_t d[22];
    for (int i = 0; i < 22; i++) {
        if (!model_get_u32(m, ds_keys[i], &d[i]))
            return failk("missing-key", ds_keys[i]);
    }
    if (d[0] == 43 && d[1] == 4096 && d[2] == 129280 && d[3] == 64 && d[4] == 1 &&
        d[5] == 512 && d[6] == 512 && d[7] == 64 && d[8] == 1024 && d[9] == 1024 &&
        d[10] == 8 && d[11] == 256 && d[12] == 6 && d[13] == 2048 && d[14] == 1 &&
        d[15] == 3 && d[16] == 128 && d[17] == 64 && d[18] == 128 && d[19] == 512 &&
        d[20] == 4 && d[21] == 20) {
        *pro = 0;
        return 0;
    }
    if (d[0] == 61 && d[1] == 7168 && d[2] == 129280 && d[3] == 128 && d[4] == 1 &&
        d[5] == 512 && d[6] == 512 && d[7] == 64 && d[8] == 1536 && d[9] == 1024 &&
        d[10] == 16 && d[11] == 384 && d[12] == 6 && d[13] == 3072 && d[14] == 1 &&
        d[15] == 3 && d[16] == 128 && d[17] == 64 && d[18] == 128 && d[19] == 1024 &&
        d[20] == 4 && d[21] == 20) {
        *pro = 1;
        return 0;
    }
    snprintf(g_tok, sizeof g_tok, "unsupported");
    return 1;
}

static int streq_str(ds4_str s, const char *z)
{
    return ds4_streq(s, z);
}

static int validate_motif3(const ds4_model *m)
{
    uint32_t u; uint64_t u64; float f; bool b; ds4_str s;
    if (req_u32(m, "motif3.block_count", &u) || exp_u32("block_count", u, 53)) return 1;
    if (req_u64c(m, "motif3.context_length", &u64) || exp_u64("context_length", u64, 262144)) return 1;
    if (req_u32(m, "motif3.embedding_length", &u) || exp_u32("embedding_length", u, 4096)) return 1;
    if (req_u32(m, "motif3.vocab_size", &u) || exp_u32("vocab_size", u, 220160)) return 1;
    if (req_u32(m, "motif3.feed_forward_length", &u) || exp_u32("feed_forward_length", u, 12288)) return 1;
    if (req_u32(m, "motif3.leading_dense_block_count", &u) || exp_u32("leading_dense_block_count", u, 2)) return 1;
    if (req_u32(m, "motif3.expert_count", &u) || exp_u32("expert_count", u, 384)) return 1;
    if (req_u32(m, "motif3.expert_used_count", &u) || exp_u32("expert_used_count", u, 8)) return 1;
    if (req_u32(m, "motif3.expert_feed_forward_length", &u) || exp_u32("expert_feed_forward_length", u, 1280)) return 1;
    if (req_u32(m, "motif3.expert_shared_count", &u) || exp_u32("expert_shared_count", u, 1)) return 1;
    if (req_u32(m, "motif3.expert_gating_func", &u) || exp_u32("expert_gating_func", u, 1)) return 1;
    if (req_u32(m, "motif3.attention.head_count", &u) || exp_u32("attention.head_count", u, 80)) return 1;
    if (req_u32(m, "motif3.attention.head_count_kv", &u) || exp_u32("attention.head_count_kv", u, 16)) return 1;
    if (req_u32(m, "motif3.attention.noise_head_count", &u) || exp_u32("attention.noise_head_count", u, 16)) return 1;
    if (req_u32(m, "motif3.attention.key_length", &u) || exp_u32("attention.key_length", u, 192)) return 1;
    if (req_u32(m, "motif3.attention.value_length", &u) || exp_u32("attention.value_length", u, 128)) return 1;
    if (req_u32(m, "motif3.attention.q_lora_rank", &u) || exp_u32("attention.q_lora_rank", u, 1024)) return 1;
    if (req_u32(m, "motif3.attention.kv_lora_rank", &u) || exp_u32("attention.kv_lora_rank", u, 512)) return 1;
    if (req_u32(m, "motif3.attention.rope_dimension_count", &u) || exp_u32("attention.rope_dimension_count", u, 64)) return 1;
    if (req_u32(m, "motif3.attention.sliding_window", &u) || exp_u32("attention.sliding_window", u, 128)) return 1;
    if (req_u32(m, "motif3.attention.sliding_window_period", &u) || exp_u32("attention.sliding_window_period", u, 4)) return 1;
    if (req_u32(m, "motif3.mhc.expansion_rate", &u) || exp_u32("mhc.expansion_rate", u, 4)) return 1;
    if (req_u32(m, "motif3.mhc.sinkhorn_iterations", &u) || exp_u32("mhc.sinkhorn_iterations", u, 20)) return 1;
    if (req_u32(m, "motif3.mtp.block_count", &u) || exp_u32("mtp.block_count", u, 1)) return 1;
    if (req_bool(m, "motif3.expert_weights_norm", &b) || exp_bool("expert_weights_norm", b, true)) return 1;
    if (req_bool(m, "motif3.attention.elementwise_output_gate", &b) || exp_bool("attention.elementwise_output_gate", b, true)) return 1;
    if (req_bool(m, "motif3.mhc.enabled", &b) || exp_bool("mhc.enabled", b, true)) return 1;
    if (req_bool(m, "motif3.polynorm.sigmoid_weight", &b) || exp_bool("polynorm.sigmoid_weight", b, true)) return 1;
    if (req_bool(m, "motif3.rope.scaling.apply_mscale", &b) || exp_bool("rope.scaling.apply_mscale", b, false)) return 1;
    if (req_f32(m, "motif3.expert_weights_scale", &f) || exp_f32("expert_weights_scale", f, 2.0f)) return 1;
    if (req_f32(m, "motif3.expert_score_correction", &f) || exp_f32("expert_score_correction", f, 1.0e-4f)) return 1;
    if (req_f32(m, "motif3.attention.layer_norm_rms_epsilon", &f) || exp_f32("attention.layer_norm_rms_epsilon", f, 1.0e-5f)) return 1;
    if (req_f32(m, "motif3.rope.freq_base", &f) || exp_f32("rope.freq_base", f, 10000.0f)) return 1;
    if (req_f32(m, "motif3.rope.freq_base_swa", &f) || exp_f32("rope.freq_base_swa", f, 10000.0f)) return 1;
    if (req_f32(m, "motif3.rope.scaling.factor", &f) || exp_f32("rope.scaling.factor", f, 64.0f)) return 1;
    if (req_f32(m, "motif3.rope.scaling.beta_fast", &f) || exp_f32("rope.scaling.beta_fast", f, 32.0f)) return 1;
    if (req_f32(m, "motif3.rope.scaling.beta_slow", &f) || exp_f32("rope.scaling.beta_slow", f, 1.0f)) return 1;
    if (req_f32(m, "motif3.rope.scaling.mscale", &f) || exp_f32("rope.scaling.mscale", f, 1.0f)) return 1;
    if (req_f32(m, "motif3.mhc.h_post_coefficient", &f) || exp_f32("mhc.h_post_coefficient", f, 1.0f)) return 1;
    if (req_f32(m, "motif3.polynorm.output_scale", &f) || exp_f32("polynorm.output_scale", f, 0.5f)) return 1;
    if (req_f32(m, "motif3.polynorm.bias_clamp", &f) || exp_f32("polynorm.bias_clamp", f, 0.5f)) return 1;
    if (req_f32(m, "motif3.hidden_clamp", &f) || exp_f32("hidden_clamp", f, 1000000.0f)) return 1;
    if (!model_get_string(m, "motif3.attention.sliding_window_pattern", &s) || !streq_str(s, "interleave"))
        return failk("mismatch-string", "motif3.attention.sliding_window_pattern");
    if (!model_get_string(m, "motif3.rope.scaling.type", &s) || !streq_str(s, "yarn"))
        return failk("mismatch-string", "motif3.rope.scaling.type");
    if (!model_get_string(m, "motif3.activation", &s) || !streq_str(s, "poly_norm"))
        return failk("mismatch-string", "motif3.activation");
    if (!model_get_string(m, "motif3.source.config_sha256", &s) ||
        !streq_str(s, "30f14b635d3258a18c3ff7e69829f8fbfa775e87477ffabb59a79115bba820a5"))
        return failk("mismatch-string", "motif3.source.config_sha256");
    {
        uint32_t full = 0, il;
        for (il = 0; il < 53; il++)
            if ((il % 4u) == 0) full++;
        if (exp_u32("full_attention_layer_count", full, 14)) return 1;
    }
    return 0;
}

static int validate_dots3(const ds4_model *m)
{
    uint32_t u; uint64_t u64; float f; bool b; ds4_str s;
    if (req_u32(m, "dots3-note.block_count", &u) || exp_u32("block_count", u, 47)) return 1;
    if (req_u64c(m, "dots3-note.context_length", &u64) || exp_u64("context_length", u64, 524288)) return 1;
    if (req_u32(m, "dots3-note.embedding_length", &u) || exp_u32("embedding_length", u, 5120)) return 1;
    if (req_u32(m, "dots3-note.vocab_size", &u) || exp_u32("vocab_size", u, 152064)) return 1;
    if (req_u32(m, "dots3-note.feed_forward_length", &u) || exp_u32("feed_forward_length", u, 13824)) return 1;
    if (req_u32(m, "dots3-note.leading_dense_block_count", &u) || exp_u32("leading_dense_block_count", u, 1)) return 1;
    if (req_u32(m, "dots3-note.expert_count", &u) || exp_u32("expert_count", u, 256)) return 1;
    if (req_u32(m, "dots3-note.expert_used_count", &u) || exp_u32("expert_used_count", u, 8)) return 1;
    if (req_u32(m, "dots3-note.expert_feed_forward_length", &u) || exp_u32("expert_feed_forward_length", u, 1536)) return 1;
    if (req_u32(m, "dots3-note.expert_shared_count", &u) || exp_u32("expert_shared_count", u, 1)) return 1;
    if (req_u32(m, "dots3-note.attention.head_count", &u) || exp_u32("attention.head_count", u, 128)) return 1;
    if (req_u32(m, "dots3-note.attention.head_count_kv", &u) || exp_u32("attention.head_count_kv", u, 128)) return 1;
    if (req_u32(m, "dots3-note.attention.key_length", &u) || exp_u32("attention.key_length", u, 192)) return 1;
    if (req_u32(m, "dots3-note.attention.value_length", &u) || exp_u32("attention.value_length", u, 128)) return 1;
    if (req_u32(m, "dots3-note.sliding_window", &u) || exp_u32("sliding_window", u, 513)) return 1;
    if (req_u32(m, "dots3-note.index_topk", &u) || exp_u32("index_topk", u, 2048)) return 1;
    if (req_u32(m, "dots3-note.q_lora_rank", &u) || exp_u32("q_lora_rank", u, 1024)) return 1;
    if (req_u32(m, "dots3-note.kv_lora_rank", &u) || exp_u32("kv_lora_rank", u, 512)) return 1;
    if (req_u32(m, "dots3-note.swa_kv_lora_rank", &u) || exp_u32("swa_kv_lora_rank", u, 1024)) return 1;
    if (req_u32(m, "dots3-note.full_attention_count", &u) || exp_u32("full_attention_count", u, 13)) return 1;
    if (req_bool(m, "dots3-note.language_only", &b) || exp_bool("language_only", b, true)) return 1;
    if (req_bool(m, "dots3-note.mtp.present", &b) || exp_bool("mtp.present", b, true)) return 1;
    if (req_f32(m, "dots3-note.rope.freq_base", &f) || exp_f32("rope.freq_base", f, 80000000.0f)) return 1;
    if (req_f32(m, "dots3-note.rope.freq_base_swa", &f) || exp_f32("rope.freq_base_swa", f, 50000.0f)) return 1;
    if (req_f32(m, "dots3-note.attention.layer_norm_rms_epsilon", &f) ||
        exp_f32("attention.layer_norm_rms_epsilon", f, 1.0e-5f)) return 1;
    if (!model_get_string(m, "dots3-note.source.config_sha256", &s) ||
        !streq_str(s, "99b7de680dd456111c36efb8749f8ae7177328e97b65a3e39a6700cbc1173833"))
        return failk("mismatch-string", "dots3-note.source.config_sha256");
    {
        uint32_t full = 0, il;
        for (il = 0; il < 47; il++) {
            if (il + 1 >= 47) continue;
            if (il == 0 || (il % 4u) == 1u) full++;
        }
        if (exp_u32("full_attention_layer_count", full, 13)) return 1;
    }
    return 0;
}

static int validate_solar(const ds4_model *m)
{
    uint32_t n_layer, u; uint64_t u64; float f; bool b;
    ds4_array arr;
    if (req_u32(m, "solar-open2.block_count", &n_layer) || exp_u32("block_count", n_layer, 48)) return 1;
    if (req_u64c(m, "solar-open2.context_length", &u64) || exp_u64("context_length", u64, 1048576)) return 1;
    if (req_u32(m, "solar-open2.embedding_length", &u) || exp_u32("embedding_length", u, 4096)) return 1;
    if (req_u32(m, "solar-open2.vocab_size", &u) || exp_u32("vocab_size", u, 196608)) return 1;
    if (req_u32(m, "solar-open2.feed_forward_length", &u) || exp_u32("feed_forward_length", u, 10240)) return 1;
    if (req_u32(m, "solar-open2.attention.head_count", &u) || exp_u32("attention.head_count", u, 64)) return 1;
    if (req_u32(m, "solar-open2.attention.key_length", &u) || exp_u32("attention.key_length", u, 128)) return 1;
    if (req_u32(m, "solar-open2.attention.value_length", &u) || exp_u32("attention.value_length", u, 128)) return 1;
    if (req_u32(m, "solar-open2.expert_count", &u) || exp_u32("expert_count", u, 320)) return 1;
    if (req_u32(m, "solar-open2.expert_used_count", &u) || exp_u32("expert_used_count", u, 8)) return 1;
    if (req_u32(m, "solar-open2.expert_feed_forward_length", &u) || exp_u32("expert_feed_forward_length", u, 1280)) return 1;
    if (req_u32(m, "solar-open2.expert_shared_count", &u) || exp_u32("expert_shared_count", u, 1)) return 1;
    if (req_u32(m, "solar-open2.leading_dense_block_count", &u) || exp_u32("leading_dense_block_count", u, 0)) return 1;
    if (req_u32(m, "solar-open2.ssm.conv_kernel", &u) || exp_u32("ssm.conv_kernel", u, 4)) return 1;
    if (req_u32(m, "solar-open2.kda.head_dim", &u) || exp_u32("kda.head_dim", u, 128)) return 1;
    if (req_u32(m, "solar-open2.expert_gating_func", &u) || exp_u32("expert_gating_func", u, 2)) return 1;
    if (req_f32(m, "solar-open2.attention.layer_norm_rms_epsilon", &f) ||
        exp_f32("attention.layer_norm_rms_epsilon", f, 1.0e-5f)) return 1;
    if (req_f32(m, "solar-open2.expert_weights_scale", &f) || exp_f32("expert_weights_scale", f, 1.0f)) return 1;
    if (req_bool(m, "solar-open2.expert_weights_norm", &b) || exp_bool("expert_weights_norm", b, true)) return 1;
    if (req_f32(m, "solar-open2.rope.freq_base", &f) || exp_f32("rope.freq_base (vestigial)", f, 10000.0f)) return 1;
    if (model_get_u32(m, "solar-open2.rope.dimension_count", &u) &&
        exp_u32("rope.dimension_count (NoPE)", u, 0)) return 1;
    if (exp_bool("internal use_rope", false, false)) return 1;
    if (!model_get_array(m, "solar-open2.attention.head_count_kv", &arr))
        return failk("missing-array", "solar-open2.attention.head_count_kv");
    if (arr.type != GGUF_VALUE_INT32 && arr.type != GGUF_VALUE_UINT32)
        return failk("array-type", "solar-open2.attention.head_count_kv");
    if (arr.len != n_layer) return failk("array-short", "solar-open2.attention.head_count_kv");
    {
        ds4_cursor c = cursor_at(m, arr.data_pos);
        uint32_t il;
        for (il = 0; il < n_layer; il++) {
            uint32_t got = 0, want = ((il % 4u) == 0u) ? 8u : 0u;
            if (arr.type == GGUF_VALUE_UINT32) {
                if (!cursor_u32(&c, &got)) { snprintf(g_tok, sizeof g_tok, "truncated"); return 1; }
            } else {
                int32_t v = 0;
                if (!cursor_read(&c, &v, 4)) { snprintf(g_tok, sizeof g_tok, "truncated"); return 1; }
                if (v < 0) { snprintf(g_tok, sizeof g_tok, "negative-array"); return 1; }
                got = (uint32_t)v;
            }
            if (got != want) return faill("schedule", il);
        }
    }
    if (exp_f32("internal KDA q/k l2 epsilon", 1.0e-6f, 1.0e-6f)) return 1;
    if (exp_f32("internal KDA gate clamp minimum", -5.0f, -5.0f)) return 1;
    return 0;
}

static int validate_exaone(const ds4_model *m)
{
    uint32_t n_layer, u; uint64_t u64; float f; bool b;
    ds4_array arr;
    if (req_u32(m, "exaone-moe.block_count", &n_layer) || exp_u32("block_count", n_layer, 49)) return 1;
    if (req_u64c(m, "exaone-moe.context_length", &u64) || exp_u64("context_length", u64, 262144)) return 1;
    if (req_u32(m, "exaone-moe.embedding_length", &u) || exp_u32("embedding_length", u, 6144)) return 1;
    if (req_u32(m, "exaone-moe.vocab_size", &u) || exp_u32("vocab_size", u, 153600)) return 1;
    if (req_u32(m, "exaone-moe.feed_forward_length", &u) || exp_u32("feed_forward_length", u, 18432)) return 1;
    if (req_u32(m, "exaone-moe.attention.head_count", &u) || exp_u32("attention.head_count", u, 64)) return 1;
    if (req_u32(m, "exaone-moe.attention.head_count_kv", &u) || exp_u32("attention.head_count_kv", u, 8)) return 1;
    if (req_u32(m, "exaone-moe.attention.key_length", &u) || exp_u32("attention.key_length", u, 128)) return 1;
    if (req_u32(m, "exaone-moe.attention.value_length", &u) || exp_u32("attention.value_length", u, 128)) return 1;
    if (req_u32(m, "exaone-moe.expert_count", &u) || exp_u32("expert_count", u, 128)) return 1;
    if (req_u32(m, "exaone-moe.expert_used_count", &u) || exp_u32("expert_used_count", u, 8)) return 1;
    if (req_u32(m, "exaone-moe.expert_feed_forward_length", &u) || exp_u32("expert_feed_forward_length", u, 2048)) return 1;
    if (req_u32(m, "exaone-moe.expert_shared_feed_forward_length", &u) ||
        exp_u32("expert_shared_feed_forward_length", u, 2048)) return 1;
    if (req_u32(m, "exaone-moe.expert_shared_count", &u) || exp_u32("expert_shared_count", u, 1)) return 1;
    if (req_u32(m, "exaone-moe.expert_group_count", &u) || exp_u32("expert_group_count", u, 1)) return 1;
    if (req_u32(m, "exaone-moe.expert_group_used_count", &u) || exp_u32("expert_group_used_count", u, 1)) return 1;
    if (req_u32(m, "exaone-moe.expert_gating_func", &u) || exp_u32("expert_gating_func", u, 2)) return 1;
    if (req_u32(m, "exaone-moe.leading_dense_block_count", &u) || exp_u32("leading_dense_block_count", u, 1)) return 1;
    if (req_u32(m, "exaone-moe.nextn_predict_layers", &u) || exp_u32("nextn_predict_layers", u, 1)) return 1;
    if (req_u32(m, "exaone-moe.attention.sliding_window", &u) || exp_u32("attention.sliding_window", u, 128)) return 1;
    if (req_f32(m, "exaone-moe.rope.freq_base", &f) || exp_f32("rope.freq_base", f, 1000000.0f)) return 1;
    if (req_f32(m, "exaone-moe.attention.layer_norm_rms_epsilon", &f) ||
        exp_f32("attention.layer_norm_rms_epsilon", f, 1.0e-5f)) return 1;
    if (req_f32(m, "exaone-moe.expert_weights_scale", &f) || exp_f32("expert_weights_scale", f, 2.5f)) return 1;
    if (req_bool(m, "exaone-moe.expert_weights_norm", &b) || exp_bool("expert_weights_norm", b, true)) return 1;
    if (!model_get_array(m, "exaone-moe.attention.sliding_window_pattern", &arr))
        return failk("missing-array", "exaone-moe.attention.sliding_window_pattern");
    if (arr.type != GGUF_VALUE_BOOL)
        return failk("array-type", "exaone-moe.attention.sliding_window_pattern");
    if (arr.len != n_layer) return failk("array-short", "exaone-moe.attention.sliding_window_pattern");
    {
        ds4_cursor c = cursor_at(m, arr.data_pos);
        uint32_t il;
        for (il = 0; il < n_layer; il++) {
            uint8_t got = 0;
            bool want = ((il % 4u) != 3u);
            if (!cursor_read(&c, &got, 1)) { snprintf(g_tok, sizeof g_tok, "truncated"); return 1; }
            if ((got != 0) != want) return faill("swa-pattern", il);
        }
    }
    return 0;
}

static int validate_k2(const ds4_model *m)
{
    uint32_t u = 0;
    uint64_t u64 = 0;
    float f = 0;
    bool b = false;

    if (req_u32(m, "k2-horizon.block_count", &u) || exp_u32("block_count", u, 61)) return 1;
    if (req_u64c(m, "k2-horizon.context_length", &u64) ||
        exp_u64("context_length", u64, 524288)) return 1;
    if (req_u32(m, "k2-horizon.embedding_length", &u) ||
        exp_u32("embedding_length", u, 6144)) return 1;
    if (req_u32(m, "k2-horizon.feed_forward_length", &u) ||
        exp_u32("feed_forward_length", u, 16384)) return 1;
    if (req_u32(m, "k2-horizon.attention.head_count", &u) ||
        exp_u32("attention.head_count", u, 48)) return 1;
    if (req_u32(m, "k2-horizon.attention.head_count_kv", &u) ||
        exp_u32("attention.head_count_kv", u, 8)) return 1;
    if (req_u32(m, "k2-horizon.attention.key_length", &u) ||
        exp_u32("attention.key_length", u, 128)) return 1;
    if (req_u32(m, "k2-horizon.attention.value_length", &u) ||
        exp_u32("attention.value_length", u, 128)) return 1;
    if (req_u32(m, "k2-horizon.attention.group_norm_groups", &u) ||
        exp_u32("attention.group_norm_groups", u, 1)) return 1;
    if (req_u32(m, "k2-horizon.rope.dimension_count", &u) ||
        exp_u32("rope.dimension_count", u, 64)) return 1;
    if (req_u32(m, "k2-horizon.expert_count", &u) || exp_u32("expert_count", u, 192)) return 1;
    if (req_u32(m, "k2-horizon.expert_used_count", &u) ||
        exp_u32("expert_used_count", u, 8)) return 1;
    if (req_u32(m, "k2-horizon.expert_feed_forward_length", &u) ||
        exp_u32("expert_feed_forward_length", u, 1792)) return 1;
    if (req_u32(m, "k2-horizon.leading_dense_block_count", &u) ||
        exp_u32("leading_dense_block_count", u, 3)) return 1;
    if (req_u32(m, "k2-horizon.moe_every_n_layers", &u) ||
        exp_u32("moe_every_n_layers", u, 1)) return 1;
    if (req_u32(m, "k2-horizon.expert_shared_count", &u) ||
        exp_u32("expert_shared_count", u, 1)) return 1;
    if (req_u32(m, "k2-horizon.expert_shared_feed_forward_length", &u) ||
        exp_u32("expert_shared_feed_forward_length", u, 1792)) return 1;
    if (req_u32(m, "k2-horizon.expert_gating_func", &u) ||
        exp_u32("expert_gating_func", u, 2)) return 1;
    if (req_f32(m, "k2-horizon.rope.freq_base", &f) ||
        exp_f32("rope.freq_base", f, 10000000.0f)) return 1;
    if (req_f32(m, "k2-horizon.attention.layer_norm_rms_epsilon", &f) ||
        exp_f32("attention.layer_norm_rms_epsilon", f, 1.0e-6f)) return 1;
    if (req_f32(m, "k2-horizon.expert_weights_scale", &f) ||
        exp_f32("expert_weights_scale", f, 2.5f)) return 1;
    if (req_bool(m, "k2-horizon.expert_weights_norm", &b) ||
        exp_bool("expert_weights_norm", b, true)) return 1;
    return 0;
}

static void dump_validate(const ds4_model *m)
{
    ds4_str arch = {0};
    int have = model_get_string(m, "general.architecture", &arch);
    if (!have || ds4_streq(arch, "deepseek4")) {
        int pro = 0;
        if (read_ds_select(m, &pro)) { printf("%s\n", g_tok); return; }
        if (validate_deepseek(m, pro)) { printf("%s\n", g_tok); return; }
        printf("ok\n");
        return;
    }
    if (ds4_streq(arch, "exaone-moe")) {
        if (validate_exaone(m)) { printf("%s\n", g_tok); return; }
        printf("ok\n");
        return;
    }
    if (ds4_streq(arch, "solar-open2")) {
        if (validate_solar(m)) { printf("%s\n", g_tok); return; }
        printf("ok\n");
        return;
    }
    if (ds4_streq(arch, "motif3")) {
        if (validate_motif3(m)) { printf("%s\n", g_tok); return; }
        printf("ok\n");
        return;
    }
    if (ds4_streq(arch, "dots3-note")) {
        if (validate_dots3(m)) { printf("%s\n", g_tok); return; }
        printf("ok\n");
        return;
    }
    if (ds4_streq(arch, "k2-horizon")) {
        if (validate_k2(m)) { printf("%s\n", g_tok); return; }
        printf("ok\n");
        return;
    }
    printf("unsupported-arch %.*s\n", (int)arch.len, arch.ptr);
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
        fprintf(stderr, "usage: validate_c_oracle PATH\n");
        return 2;
    }
    memset(&m, 0, sizeof(m));
    fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("io\n"); return 0; }
    if (fstat(fd, &st) < 0) { close(fd); printf("io\n"); return 0; }
    if (st.st_size < 32) { close(fd); printf("too-small\n"); return 0; }
    map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { close(fd); printf("io\n"); return 0; }
    m.map = map;
    m.size = (uint64_t)st.st_size;
    c.base = m.map; c.size = m.size; c.pos = 0; c.err = 0;
    if (!cursor_u32(&c, &magic)) { printf("%s\n", err_token(c.err)); goto done; }
    if (magic != DS4_GGUF_MAGIC) { printf("not-gguf\n"); goto done; }
    if (!cursor_u32(&c, &m.version) || !cursor_u64(&c, &m.n_tensors) ||
        !cursor_u64(&c, &m.n_kv)) {
        printf("%s\n", err_token(c.err));
        goto done;
    }
    if (m.version != 3) { printf("unsupported-version %u\n", m.version); goto done; }
    if (!parse_metadata(&m, &c)) { printf("%s\n", err_token(c.err)); goto done; }
    dump_validate(&m);
done:
    free(m.kv);
    munmap(map, (size_t)st.st_size);
    close(fd);
    return 0;
}
