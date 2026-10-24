static void st_value_free(st_value *v) {
    free(v->dtype);
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void parse_shape(const json_doc *d, int arr_tok, st_info *info, const char *name) {
    if (d->v[arr_tok].type != JT_ARRAY) {
        fprintf(stderr, "error: bad shape for %s\n", name);
        exit(1);
    }
    int nd = 0;
    for (int i = arr_tok + 1; i < d->len && d->v[i].parent == arr_tok; i = json_skip(d, i)) {
        if (nd >= MAX_DIMS) die("too many safetensors dimensions");
        info->shape[nd++] = json_i64(d, i);
    }
    info->n_dims = nd;
}

static int db_find_shard(st_db *db, const char *file) {
    for (int i = 0; i < db->n_shards; i++) {
        if (strcmp(db->shards[i].file, file) == 0) return i;
    }
    if (db->n_shards == db->cap_shards) {
        db->cap_shards = db->cap_shards ? db->cap_shards * 2 : 32;
        db->shards = xrealloc(db->shards, (size_t)db->cap_shards * sizeof(db->shards[0]));
    }
    shard *s = &db->shards[db->n_shards];
    memset(s, 0, sizeof(*s));
    s->file = xstrdup(file);
    s->path = path_join(db->hf_dir, file);
    pthread_mutex_init(&s->lock, NULL);
    return db->n_shards++;
}

static void shard_add_tensor(shard *s, char *name, st_info info) {
    if (s->n_tensors == s->cap_tensors) {
        s->cap_tensors = s->cap_tensors ? s->cap_tensors * 2 : 256;
        s->tensors = xrealloc(s->tensors, (size_t)s->cap_tensors * sizeof(s->tensors[0]));
    }
    s->tensors[s->n_tensors++] = (tensor_entry){ .name = name, .info = info };
}

static void shard_load(shard *s) {
    if (s->loaded) return;
    FILE *fp = fopen(s->path, "rb");
    if (!fp) die_errno("open", s->path);
    uint64_t header_len = read_u64_le_fp(fp, "safetensors header length");
    char *header = xmalloc((size_t)header_len + 1);
    if (fread(header, 1, (size_t)header_len, fp) != (size_t)header_len) die_errno("read header", s->path);
    header[header_len] = '\0';
    s->data_base = 8 + header_len;

    json_doc d = json_parse_text(header, (size_t)header_len);
    if (d.len < 1 || d.v[0].type != JT_OBJECT) die("bad safetensors header");
    for (int i = 1; i < d.len && d.v[i].parent == 0;) {
        int k = i;
        int v = i + 1;
        if (v >= d.len || d.v[v].parent != 0) die("bad safetensors header object");
        if (!json_tok_eq(&d, k, "__metadata__")) {
            char *name = json_strdup_tok(&d, k);
            st_info info = {0};
            int dtype = json_obj_get(&d, v, "dtype");
            int shape = json_obj_get(&d, v, "shape");
            int offsets = json_obj_get(&d, v, "data_offsets");
            if (dtype < 0 || shape < 0 || offsets < 0) die("bad safetensors tensor entry");
            info.dtype = json_strdup_tok(&d, dtype);
            parse_shape(&d, shape, &info, name);
            int n_off = 0;
            for (int j = offsets + 1; j < d.len && d.v[j].parent == offsets; j = json_skip(&d, j)) {
                int64_t x = json_i64(&d, j);
                if (n_off == 0) info.begin = (uint64_t)x;
                else if (n_off == 1) info.end = (uint64_t)x;
                n_off++;
            }
            if (n_off != 2) die("bad safetensors data_offsets");
            shard_add_tensor(s, name, info);
        }
        i = json_skip(&d, v);
    }
    char **keys = xmalloc((size_t)s->n_tensors * sizeof(keys[0]));
    for (int i = 0; i < s->n_tensors; i++) keys[i] = s->tensors[i].name;
    hmap_build(&s->tensor_map, keys, s->n_tensors);
    free(keys);
    json_free(&d);
    free(header);
    s->fp = fp;
    s->loaded = true;
}

static void db_open(st_db *db, const char *hf_dir) {
    memset(db, 0, sizeof(*db));
    pthread_mutex_init(&db->lock, NULL);
    db->hf_dir = xstrdup(hf_dir);
    char *index_path = path_join(hf_dir, "model.safetensors.index.json");
    size_t len = 0;
    char *text = read_file(index_path, &len);
    json_doc d = json_parse_text(text, len);
    int weight_map = json_obj_get(&d, 0, "weight_map");
    if (weight_map < 0 || d.v[weight_map].type != JT_OBJECT) die("safetensors index has no weight_map");

    int cap = 4096;
    db->weights = xmalloc((size_t)cap * sizeof(db->weights[0]));
    for (int i = weight_map + 1; i < d.len && d.v[i].parent == weight_map;) {
        int k = i;
        int v = i + 1;
        if (db->n_weights == cap) {
            cap *= 2;
            db->weights = xrealloc(db->weights, (size_t)cap * sizeof(db->weights[0]));
        }
        db->weights[db->n_weights].name = json_strdup_tok(&d, k);
        db->weights[db->n_weights].file = json_strdup_tok(&d, v);
        db->n_weights++;
        i = json_skip(&d, v);
    }
    char **keys = xmalloc((size_t)db->n_weights * sizeof(keys[0]));
    for (int i = 0; i < db->n_weights; i++) {
        keys[i] = db->weights[i].name;
        db_find_shard(db, db->weights[i].file);
    }
    hmap_build(&db->weight_map, keys, db->n_weights);
    free(keys);
    json_free(&d);
    free(text);
    free(index_path);
}

static void db_close(st_db *db) {
    for (int i = 0; i < db->n_weights; i++) {
        free(db->weights[i].name);
        free(db->weights[i].file);
    }
    for (int i = 0; i < db->n_shards; i++) {
        shard *s = &db->shards[i];
        if (s->fp) fclose(s->fp);
        for (int j = 0; j < s->n_tensors; j++) {
            free(s->tensors[j].name);
            free(s->tensors[j].info.dtype);
        }
        free(s->tensors);
        hmap_free(&s->tensor_map);
        pthread_mutex_destroy(&s->lock);
        free(s->file);
        free(s->path);
    }
    hmap_free(&db->weight_map);
    pthread_mutex_destroy(&db->lock);
    free(db->weights);
    free(db->shards);
    free(db->hf_dir);
    memset(db, 0, sizeof(*db));
}

static bool db_has(const st_db *db, const char *name) {
    return hmap_get(&db->weight_map, name) >= 0;
}

static tensor_entry *db_tensor(st_db *db, const char *name, shard **shard_out) {
    pthread_mutex_lock(&db->lock);
    int wi = hmap_get(&db->weight_map, name);
    if (wi < 0) {
        fprintf(stderr, "error: HF tensor not found: %s\n", name);
        exit(1);
    }
    const char *file = db->weights[wi].file;
    int si = db_find_shard(db, file);
    shard *s = &db->shards[si];
    shard_load(s);
    int ti = hmap_get(&s->tensor_map, name);
    if (ti < 0) {
        fprintf(stderr, "error: HF tensor %s missing from shard %s\n", name, file);
        exit(1);
    }
    if (shard_out) *shard_out = s;
    tensor_entry *te = &s->tensors[ti];
    pthread_mutex_unlock(&db->lock);
    return te;
}

static st_value db_read(st_db *db, const char *name) {
    shard *s = NULL;
    tensor_entry *te = db_tensor(db, name, &s);
    const size_t nbytes = (size_t)(te->info.end - te->info.begin);
    st_value v = {0};
    v.dtype = xstrdup(te->info.dtype);
    v.n_dims = te->info.n_dims;
    memcpy(v.shape, te->info.shape, sizeof(v.shape));
    v.nbytes = nbytes;
    v.data = xmalloc(nbytes);
    /* ★无锁并发读★: 原来是 fseeko+fread —— FILE* 的文件位置是共享状态, 必须靠
     * per-shard 锁串行化。而同一层的 256 个专家几乎都落在同一个 shard, 于是 16 个
     * worker 全部堵在这一把锁上排队读, 20 核机器实测只跑出 3 核(300% CPU), 每层 61s。
     * pread 自带 offset、是原子的, 同一个 fd 可被多线程并发调用, 锁可以整个去掉。 */
    const off_t base = (off_t)(s->data_base + te->info.begin);
    size_t done = 0;
    const int fd = fileno(s->fp);
    while (done < nbytes) {
        const ssize_t got = pread(fd, (char *)v.data + done, nbytes - done, base + (off_t)done);
        if (got < 0) { if (errno == EINTR) continue; die_errno("read tensor", s->path); }
        if (got == 0) die("short read on tensor");
        done += (size_t)got;
    }
    return v;
}

/* =====
 * DeepSeek V4 data conversion
 */

/* e4m3/e8m0: 换 src/common/ds4_fp8.h 共享实现(批2 收敛)。
 * ★E4M3 语义陷阱★共享库把 0x7f/0xff(NaN 槽)解成 NaN(IEEE 语义, ds4_fp8.h 头注有言);
 * 本量化器旧实现解成 0.0f, 含 NaN 权重的量化输出字节依赖这个口径 —— 显式包一层保旧语义。 */
#include "../../src/common/ds4_fp8.h"

static float e8m0_to_f32(uint8_t e) { return ds4_e8m0_to_f32(e); }

static float e4m3fn_to_f32(uint8_t x) {
    float v = ds4_e4m3fn_to_f32(x);
    return isnan(v) ? 0.0f : v;
}

static float bf16_to_f32_bits(uint16_t bits) {
    return ds4q_bf16_to_f32(bits);
}

static int64_t value_nelements(const st_value *v) {
    int64_t n = 1;
    for (int i = 0; i < v->n_dims; i++) n *= v->shape[i];
    return n;
}

static float *tensor_to_f32(const st_value *t, int64_t *n_out) {
    const int64_t n = value_nelements(t);
    float *out = xmalloc((size_t)n * sizeof(float));
    if (strcmp(t->dtype, "F32") == 0) {
        if (t->nbytes != (size_t)n * sizeof(float)) die("bad F32 byte size");
        memcpy(out, t->data, t->nbytes);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        if (t->nbytes != (size_t)n * sizeof(uint16_t)) die("bad BF16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32_bits(load_u16_le(t->data + (size_t)i * 2));
    } else if (strcmp(t->dtype, "F16") == 0) {
        if (t->nbytes != (size_t)n * sizeof(uint16_t)) die("bad F16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = ds4q_f16_to_f32(load_u16_le(t->data + (size_t)i * 2));
    } else if (strcmp(t->dtype, "F8_E4M3") == 0) {
        if (t->nbytes != (size_t)n) die("bad F8_E4M3 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = e4m3fn_to_f32(t->data[i]);
    } else {
        fprintf(stderr, "error: cannot convert HF dtype directly: %s\n", t->dtype);
        exit(1);
    }
    if (n_out) *n_out = n;
    return out;
}

/* F8_E4M3 weights with a 128x128 block scale.  The DeepSeek-V4-Flash *Base*
 * checkpoint stores that scale as F32 on disk (config.json claims "ue8m0", but
 * the safetensors dtype is F32); the packed quant release stores it as F8_E8M0.
 * Accept both — geometry is identical, only the per-block scale read differs. */
static float *dequant_fp8_weight(const st_value *w, const st_value *scale, int64_t *n_out) {
    if (strcmp(w->dtype, "F8_E4M3") != 0) die("bad FP8 weight dtype (want F8_E4M3)");
    const bool scale_f32 = (strcmp(scale->dtype, "F32") == 0);
    if (!scale_f32 && strcmp(scale->dtype, "F8_E8M0") != 0)
        die("FP8 block scale must be F32 (Base checkpoint) or F8_E8M0 (packed release)");
    if (w->n_dims != 2 || scale->n_dims != 2) die("FP8 tensor must be 2D");
    const int64_t out_dim = w->shape[0];
    const int64_t in_dim = w->shape[1];
    const int64_t block_out = 128;
    const int64_t block_in = 128;
    if (out_dim % block_out || in_dim % block_in) die("FP8 dims are not divisible by 128");
    const int64_t scale_rows = out_dim / block_out;
    const int64_t scale_cols = in_dim / block_in;
    if (scale->shape[0] != scale_rows || scale->shape[1] != scale_cols) die("FP8 scale shape mismatch");
    if (scale->nbytes < (size_t)scale_rows * (size_t)scale_cols * (scale_f32 ? 4u : 1u))
        die("FP8 scale byte size too small");
    float *out = xmalloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
    for (int64_t ob = 0; ob < scale_rows; ob++) {
        for (int64_t ib = 0; ib < scale_cols; ib++) {
            const size_t sidx = (size_t)ob * (size_t)scale_cols + (size_t)ib;
            const float s = scale_f32 ? load_f32_le(scale->data + sidx * 4)
                                      : e8m0_to_f32(scale->data[sidx]);
            for (int64_t r = 0; r < block_out; r++) {
                const int64_t row = ob * block_out + r;
                const size_t base = (size_t)row * (size_t)in_dim + (size_t)ib * (size_t)block_in;
                for (int64_t c = 0; c < block_in; c++) {
                    out[base + (size_t)c] = e4m3fn_to_f32(w->data[base + (size_t)c]) * s;
                }
            }
        }
    }
    if (n_out) *n_out = out_dim * in_dim;
    return out;
}

static float *dequant_fp4_weight(const st_value *w, const st_value *scale, int64_t *n_out) {
    static const float fp4_table[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    if (strcmp(w->dtype, "I8") != 0 || strcmp(scale->dtype, "F8_E8M0") != 0) die("bad FP4 weight/scale dtype");
    if (w->n_dims != 2 || scale->n_dims != 2) die("FP4 tensor must be 2D");
    const int64_t out_dim = w->shape[0];
    const int64_t packed_in = w->shape[1];
    const int64_t in_dim = packed_in * 2;
    if (in_dim % 32) die("FP4 in_dim is not divisible by 32");
    const int64_t n_blocks = in_dim / 32;
    if (scale->shape[0] != out_dim || scale->shape[1] != n_blocks) die("FP4 scale shape mismatch");
    float *out = xmalloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t b = 0; b < n_blocks; b++) {
            const float s = e8m0_to_f32(scale->data[(size_t)r * (size_t)n_blocks + (size_t)b]);
            const size_t wbase = ((size_t)r * (size_t)n_blocks + (size_t)b) * 16;
            const size_t obase = (size_t)r * (size_t)in_dim + (size_t)b * 32;
            for (int64_t j = 0; j < 16; j++) {
                const uint8_t q = w->data[wbase + (size_t)j];
                out[obase + (size_t)(2*j + 0)] = fp4_table[q & 0x0f] * s;
                out[obase + (size_t)(2*j + 1)] = fp4_table[(q >> 4) & 0x0f] * s;
            }
        }
    }
    if (n_out) *n_out = out_dim * in_dim;
    return out;
}

/* =====
 * Imatrix
 */

typedef struct {
    char *name;
    float *values;
    int n_values;
} imatrix_entry;

typedef struct {
    char *file;
    char *dataset;
    imatrix_entry *entries;
    int n_entries;
    hmap map;
    int chunks;
    bool strict;
} imatrix_store;

static void imatrix_load(imatrix_store *im, const char *path, bool strict) {
    memset(im, 0, sizeof(*im));
    im->file = xstrdup(path);
    im->strict = strict;
    im->chunks = -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open imatrix", path);
    int32_t n_entries = read_i32_fp(fp, "imatrix entry count");
    if (n_entries < 1) die("imatrix has no entries");
    im->entries = xcalloc((size_t)n_entries, sizeof(im->entries[0]));
    im->n_entries = n_entries;
    for (int i = 0; i < n_entries; i++) {
        int32_t len = read_i32_fp(fp, "imatrix name length");
        if (len <= 0 || len > 4096) die("bad imatrix name length");
        char *name = xmalloc((size_t)len + 1);
        if (fread(name, 1, (size_t)len, fp) != (size_t)len) die("short imatrix name read");
        name[len] = '\0';
        int32_t ncall = read_i32_fp(fp, "imatrix calls");
        int32_t nval = read_i32_fp(fp, "imatrix values");
        if (nval < 1) die("bad imatrix value count");
        float *values = xmalloc((size_t)nval * sizeof(float));
        if (fread(values, sizeof(float), (size_t)nval, fp) != (size_t)nval) die("short imatrix value read");
        if (ncall > 0) {
            for (int j = 0; j < nval; j++) values[j] /= (float)ncall;
        }
        for (int j = 0; j < nval; j++) {
            if (!isfinite(values[j])) die("non-finite imatrix value");
        }
        im->entries[i] = (imatrix_entry){ .name = name, .values = values, .n_values = nval };
    }
    if (fgetc(fp) != EOF) {
        if (fseeko(fp, -1, SEEK_CUR) == 0) {
            im->chunks = read_i32_fp(fp, "imatrix chunks");
            int32_t dlen = read_i32_fp(fp, "imatrix dataset length");
            if (dlen > 0 && dlen < (1 << 20)) {
                im->dataset = xmalloc((size_t)dlen + 1);
                if (fread(im->dataset, 1, (size_t)dlen, fp) == (size_t)dlen) {
                    im->dataset[dlen] = '\0';
                } else {
                    free(im->dataset);
                    im->dataset = NULL;
                }
            }
        }
    }
    fclose(fp);
    char **keys = xmalloc((size_t)n_entries * sizeof(keys[0]));
    for (int i = 0; i < n_entries; i++) keys[i] = im->entries[i].name;
    hmap_build(&im->map, keys, n_entries);
    free(keys);
    fprintf(stderr, "loaded imatrix %s: %d entries%s%s\n",
            path, n_entries, im->dataset ? ", dataset=" : "", im->dataset ? im->dataset : "");
}

static bool imatrix_enabled(const imatrix_store *im) {
    return im && im->n_entries > 0;
}

static const float *imatrix_find(
        const imatrix_store *im,
        const char **names,
        int n_names,
        int64_t ncols,
        int expert_id,
        int n_experts) {
    if (!imatrix_enabled(im)) return NULL;
    char tmp[4096];
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 0; i < n_names; i++) {
            if (!names[i]) continue;
            const char *candidate = names[i];
            if (expert_id >= 0 && pass < 2) {
                snprintf(tmp, sizeof(tmp), "%s.expert%s%d", names[i], pass == 0 ? "." : "_", expert_id);
                candidate = tmp;
            } else if (pass < 2) {
                continue;
            }
            int idx = hmap_get(&im->map, candidate);
            if (idx < 0) continue;
            const imatrix_entry *e = &im->entries[idx];
            if ((int64_t)e->n_values == ncols) return e->values;
            if (expert_id >= 0 && n_experts > 0 && (int64_t)e->n_values == ncols * (int64_t)n_experts) {
                return e->values + (size_t)expert_id * (size_t)ncols;
            }
            fprintf(stderr, "error: imatrix size mismatch for %s: got %d expected %" PRId64 "\n",
                    candidate, e->n_values, ncols);
            exit(1);
        }
    }
    if (im->strict) {
        fprintf(stderr, "error: missing imatrix entry for %s\n", names[0] ? names[0] : "(unnamed)");
        exit(1);
    }
    return NULL;
}

static void imatrix_free(imatrix_store *im) {
    for (int i = 0; i < im->n_entries; i++) {
        free(im->entries[i].name);
        free(im->entries[i].values);
    }
    free(im->entries);
    free(im->file);
    free(im->dataset);
    hmap_free(&im->map);
    memset(im, 0, sizeof(*im));
}

/* =====
 * Optional routed-expert hot mask (precision-inversion / hot-resident build)
 *
 * A DSXM mask (gguf-tools/make_expert_mask.py) marks, per layer, which routed
 * experts to KEEP.  With --experts-hot-mask the quantizer emits each main-block
 * ffn_{gate,up,down}_exps tensor containing ONLY the kept (hot) experts,
 * compacted to contiguous slots at the requested --experts type, and records
 *   ds4.expert_keep_map.kept_counts   array<i32> length n_layers
 *   ds4.expert_keep_map.original_ids  array<i32> length sum(kept_counts)
 * so ds4.c:load_expert_keep_map() maps router expert id -> compact slot at run
 * time.  Dropped (cold) experts are not written into this GGUF; in the hybrid
 * layout they are streamed from the HF safetensors.  The byte format matches
 * shrink_gguf.py exactly, so the runtime consumes either producer identically.
 * MTP draft experts (mtp.N.*) are never shrunk here, matching shrink_gguf's
 * blk-only regex.
 */
typedef struct {
    bool active;
    int n_layers;
    int n_expert;        /* mask experts-per-layer; must equal routed expert count */
    int *kept_counts;    /* [n_layers] */
    int **kept_ids;      /* [n_layers][kept_counts[l]] kept expert ids, ascending */
    int total_kept;
} hot_mask;

