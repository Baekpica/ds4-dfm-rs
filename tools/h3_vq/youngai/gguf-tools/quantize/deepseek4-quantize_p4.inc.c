static byte_buf i64_to_i32(const st_value *src) {
    if (strcmp(src->dtype, "I64") != 0) die("expected I64 source for I32 tensor");
    const int64_t n = value_nelements(src);
    if (src->nbytes != (size_t)n * sizeof(int64_t)) die("bad I64 byte size");
    byte_buf out = { .size = (size_t)n * sizeof(int32_t), .data = xmalloc((size_t)n * sizeof(int32_t)) };
    int32_t *dst = (int32_t *)out.data;
    for (int64_t i = 0; i < n; i++) {
        int64_t v = load_i64_le(src->data + (size_t)i * 8);
        if (v < INT32_MIN || v > INT32_MAX) die("I64 value out of I32 range");
        dst[i] = (int32_t)v;
    }
    return out;
}

static size_t tensor_nbytes(ds4q_type type, const int64_t *ne, int n_dims) {
    size_t nbytes = ds4q_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) nbytes *= (size_t)ne[i];
    return nbytes;
}

static void check_reversed_shape(const char *gguf_name, const st_info *info, const tensor_meta *tmpl) {
    int nd = tensor_n_dims(tmpl);
    /* HF 前导 1 维剔除(2026-08-18): DSpark confidence proj 形如 [1, 4352], GGUF 侧
     * ne 规约后 rank=1 — 元素等价, 按剔除后的形状比对。 */
    int ind = info->n_dims;
    const int64_t *ish = info->shape;
    while (ind > 1 && ish[0] == 1) { ish++; ind--; }
    if (ind != nd) {
        fprintf(stderr, "error: rank mismatch for %s\n", gguf_name);
        exit(1);
    }
    for (int i = 0; i < nd; i++) {
        if (tmpl->ne[i] != ish[nd - 1 - i]) {
            fprintf(stderr, "error: shape mismatch for %s\n", gguf_name);
            exit(1);
        }
    }
}

static byte_buf generate_regular(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                                 ds4q_type target, const imatrix_store *imatrix) {
    char *hf_name = hf_name_for_regular(gguf_name);
    tensor_entry *te = db_tensor(db, hf_name, NULL);
    check_reversed_shape(gguf_name, &te->info, tmpl);
    if (target == DS4Q_TYPE_I32) {
        st_value sv = db_read(db, hf_name);
        byte_buf b = i64_to_i32(&sv);
        st_value_free(&sv);
        free(hf_name);
        return b;
    }
    if (!is_quantizable_target(target)) die("unsupported regular target type");
    int64_t n = 0;
    float *f32 = NULL;
    if (strcmp(te->info.dtype, "F8_E4M3") == 0) {
        if (!str_ends(hf_name, ".weight")) die("FP8 tensor without .weight suffix");
        char *scale_name = xstrdup(hf_name);
        strcpy(scale_name + strlen(scale_name) - strlen(".weight"), ".scale");
        if (!db_has(db, scale_name)) die("missing FP8 scale tensor");
        st_value w = db_read(db, hf_name);
        st_value s = db_read(db, scale_name);
        f32 = dequant_fp8_weight(&w, &s, &n);
        st_value_free(&w);
        st_value_free(&s);
        free(scale_name);
    } else {
        st_value w = db_read(db, hf_name);
        f32 = tensor_to_f32(&w, &n);
        st_value_free(&w);
    }
    const char *names[2] = { gguf_name, hf_name };
    const float *imat = imatrix_find(imatrix, names, 2, tmpl->ne[0], -1, 0);
    byte_buf b = f32_to_type(f32, n, target, tmpl->ne[0], imat);
    free(f32);
    free(hf_name);
    return b;
}

typedef struct {
    st_db *db;
    const char *gguf_name;
    const tensor_meta *tmpl;
    ds4q_type target;
    int n_experts;         /* full routed expert count == imatrix segment count */
    int n_emit;            /* experts to emit (== n_experts, or hot kept count) */
    const int *emit_src;   /* [n_emit] source expert id per output slot, NULL = identity */
    const imatrix_store *imatrix;
    expert_tensor expert;
    const char *wid;
    int64_t ncols;
    int64_t nrows;
    size_t per_expert;
    byte_buf *out;
    int next;
    int done;
    pthread_mutex_t lock;
} expert_job;

/* slot = output position in the (possibly shrunken) tensor; src = the HF /
 * imatrix expert id feeding it.  Without a hot mask they are identical. */
static void generate_one_expert(expert_job *j, int slot) {
    int src = j->emit_src ? j->emit_src[slot] : slot;
    char prefix[256];
    if (j->expert.is_mtp)
        snprintf(prefix, sizeof(prefix), "mtp.%d.ffn.experts.%d.%s", j->expert.layer, src, j->wid);
    else
        snprintf(prefix, sizeof(prefix), "layers.%d.ffn.experts.%d.%s", j->expert.layer, src, j->wid);
    char weight_name[320];
    char scale_name[320];
    snprintf(weight_name, sizeof(weight_name), "%s.weight", prefix);
    snprintf(scale_name, sizeof(scale_name), "%s.scale", prefix);
    st_value w = db_read(j->db, weight_name);
    st_value s = db_read(j->db, scale_name);
    int64_t n = 0;
    float *f32;
    if (strcmp(w.dtype, "F8_E4M3") == 0) {
        /* Base checkpoint: experts are F8_E4M3 (1 byte/elem) + 128x128 block scale. */
        if (w.n_dims != 2 || w.shape[0] != j->nrows || w.shape[1] != j->ncols)
            die("expert shape mismatch (FP8)");
        f32 = dequant_fp8_weight(&w, &s, &n);
    } else {
        /* Packed quant release: experts are I8-packed FP4 (2 per byte) + E8M0 scale. */
        if (w.n_dims != 2 || w.shape[0] != j->nrows || w.shape[1] * 2 != j->ncols)
            die("expert shape mismatch (FP4)");
        f32 = dequant_fp4_weight(&w, &s, &n);
    }
    const char *names[3] = { j->gguf_name, weight_name, NULL };
    const float *imat = imatrix_find(j->imatrix, names, 2, j->ncols, src, j->n_experts);
    byte_buf q = f32_to_type(f32, n, j->target, j->ncols, imat);
    if (q.size != j->per_expert) die("expert quantized size mismatch");
    memcpy(j->out->data + (size_t)slot * j->per_expert, q.data, q.size);
    free(q.data);
    free(f32);
    st_value_free(&w);
    st_value_free(&s);
}

static void *expert_worker(void *arg) {
    expert_job *j = arg;
    for (;;) {
        pthread_mutex_lock(&j->lock);
        int slot = j->next++;
        pthread_mutex_unlock(&j->lock);
        if (slot >= j->n_emit) break;
        generate_one_expert(j, slot);
        pthread_mutex_lock(&j->lock);
        int done = ++j->done;
        if (done % 32 == 0 || done == j->n_emit) {
            fprintf(stderr, "generate_expert_tensor: layer %d %s %d/%d experts\n",
                    j->expert.layer, j->wid, done, j->n_emit);
        }
        pthread_mutex_unlock(&j->lock);
    }
    return NULL;
}

static byte_buf generate_expert(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix, const hot_mask *hm) {
    expert_tensor e = parse_expert_tensor(gguf_name);
    if (!e.is_expert) die("not an expert tensor");
    if (!is_quantizable_target(target)) die("unsupported expert target type");
    const char *wid = expert_part_name(e.part);
    const int64_t ncols = tmpl->ne[0];
    const int64_t nrows = tmpl->ne[1];
    const size_t per_expert = (size_t)nrows * ds4q_row_size(target, ncols);
    /* Hot mask shrinks routed experts. With a matched MTP mask it ALSO shrinks
     * the MTP draft block (mtp.0) so the reduced drafter fits GPU-resident for the
     * 本机-MTP loopback (acceptance comes from MTP+copy-spec hybrid; this is for fit). */
    int n_emit = n_experts;
    const int *emit_src = NULL;
    if (hm && hm->active) {
        if (e.layer < 0 || e.layer >= hm->n_layers) die("hot mask: expert layer out of range");
        n_emit = hm->kept_counts[e.layer];
        emit_src = hm->kept_ids[e.layer];
        if (n_emit < 1) die("hot mask: layer keeps zero experts");
    }
    byte_buf out = { .size = per_expert * (size_t)n_emit,
                     .data = xmalloc(per_expert * (size_t)(n_emit > 0 ? n_emit : 1)) };
    ds4q_quantize_init(target);
    int worker_count = n_threads > 0 ? n_threads : 8;
    if (worker_count < 1) worker_count = 1;
    if (worker_count > n_emit) worker_count = n_emit;
    fprintf(stderr, "generate_expert_tensor: layer %d %s emitting %d/%d experts using %d worker%s\n",
            e.layer, wid, n_emit, n_experts, worker_count, worker_count == 1 ? "" : "s");
    expert_job job = {
        .db = db, .gguf_name = gguf_name, .tmpl = tmpl, .target = target,
        .n_experts = n_experts, .n_emit = n_emit, .emit_src = emit_src,
        .imatrix = imatrix, .expert = e, .wid = wid,
        .ncols = ncols, .nrows = nrows, .per_expert = per_expert, .out = &out,
    };
    pthread_mutex_init(&job.lock, NULL);
    pthread_t *threads = xcalloc((size_t)worker_count, sizeof(threads[0]));
    for (int i = 1; i < worker_count; i++) pthread_create(&threads[i], NULL, expert_worker, &job);
    expert_worker(&job);
    for (int i = 1; i < worker_count; i++) pthread_join(threads[i], NULL);
    pthread_mutex_destroy(&job.lock);
    free(threads);
    return out;
}

static byte_buf generate_tensor(st_db *db, const char *name, const tensor_meta *tmpl,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix, const hot_mask *hm) {
    if (parse_expert_tensor(name).is_expert) {
        return generate_expert(db, name, tmpl, target, n_experts, n_threads, imatrix, hm);
    }
    return generate_regular(db, name, tmpl, target, imatrix);
}

/* =====
 * Minimal GGUF reader/writer
 *
 * GGUF metadata is copied as raw KV records from the template.  Tensor infos
 * are rewritten with the new target types and offsets.  This keeps the tool C
 * only and independent from general-purpose GGUF libraries.
 */

typedef struct {
    size_t start;
    size_t end;
} byte_span;

typedef struct {
    char *path;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint8_t *kv_raw;
    size_t kv_raw_len;
    size_t alignment;
    int n_experts;
    size_t data_offset;
    tensor_meta *tensors;
    hmap tensor_map;
} gguf_file;

typedef struct {
    tensor_meta *tensors;
    uint64_t n_tensors;
    uint64_t n_kv_extra;
    size_t meta_size;
    size_t data_offset;
    size_t tensor_bytes;
    size_t alignment;
} output_context;

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static char *read_gguf_string_fp(FILE *fp) {
    uint64_t n = read_u64_le_fp(fp, "GGUF string length");
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) die("short GGUF string read");
    s[n] = '\0';
    return s;
}

static void skip_bytes_fp(FILE *fp, uint64_t n) {
    if (fseeko(fp, (off_t)n, SEEK_CUR) != 0) die("GGUF seek failed");
}

static void skip_gguf_value_fp(FILE *fp, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = read_u64_le_fp(fp, "GGUF string length");
        skip_bytes_fp(fp, n);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = read_u32_le_fp(fp, "GGUF array type");
        uint64_t n = read_u64_le_fp(fp, "GGUF array count");
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len = read_u64_le_fp(fp, "GGUF array string length");
                skip_bytes_fp(fp, len);
            }
        } else {
            size_t sz = gguf_scalar_size(elem_type);
            if (!sz) die("unsupported GGUF array type");
            skip_bytes_fp(fp, n * sz);
        }
        return;
    }
    size_t sz = gguf_scalar_size(type);
    if (!sz) die("unsupported GGUF value type");
    skip_bytes_fp(fp, sz);
}

static size_t gguf_string_size(const char *s) {
    return sizeof(uint64_t) + strlen(s);
}

static void write_u32(FILE *fp, uint32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u32 failed");
}

static void write_u64(FILE *fp, uint64_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u64 failed");
}

static void write_gguf_string(FILE *fp, const char *s) {
    uint64_t n = strlen(s);
    write_u64(fp, n);
    if (n && fwrite(s, 1, (size_t)n, fp) != (size_t)n) die("write string failed");
}

static bool is_imatrix_kv_key(const char *key) {
    return str_starts(key, "quantize.imatrix.");
}

static size_t extra_imatrix_kv_size(const imatrix_store *im) {
    if (!imatrix_enabled(im)) return 0;
    size_t n = 0;
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_FILE) + 4 + gguf_string_size(im->file);
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES) + 4 + 8;
    if (im->dataset) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_DATASET) + 4 + gguf_string_size(im->dataset);
    if (im->chunks > 0) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS) + 4 + 8;
    return n;
}

static uint64_t extra_imatrix_kv_count(const imatrix_store *im) {
    if (!imatrix_enabled(im)) return 0;
    return 2 + (im->dataset ? 1 : 0) + (im->chunks > 0 ? 1 : 0);
}

static void write_imatrix_kvs(FILE *fp, const imatrix_store *im) {
    if (!imatrix_enabled(im)) return;
    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_FILE);
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, im->file);

    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES);
    write_u32(fp, GGUF_TYPE_UINT64);
    write_u64(fp, (uint64_t)im->n_entries);

    if (im->dataset) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_DATASET);
        write_u32(fp, GGUF_TYPE_STRING);
        write_gguf_string(fp, im->dataset);
    }
    if (im->chunks > 0) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS);
        write_u32(fp, GGUF_TYPE_UINT64);
        write_u64(fp, (uint64_t)im->chunks);
    }
}

/* GGUF array<i32> KV: key string, outer-type ARRAY, elem-type INT32, count, values. */
static size_t kv_array_i32_size(const char *key, size_t n) {
    return gguf_string_size(key) + 4 /*outer type*/ + 4 /*elem type*/ + 8 /*count*/ + n * 4;
}

static void write_kv_array_i32(FILE *fp, const char *key, const int *values, size_t n) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_ARRAY);
    write_u32(fp, GGUF_TYPE_INT32);
    write_u64(fp, (uint64_t)n);
    for (size_t i = 0; i < n; i++) write_u32(fp, (uint32_t)values[i]);
}

static uint64_t extra_keepmap_kv_count(const hot_mask *hm) {
    return (hm && hm->active) ? 2 : 0;
}

static size_t extra_keepmap_kv_size(const hot_mask *hm) {
    if (!hm || !hm->active) return 0;
    return kv_array_i32_size("ds4.expert_keep_map.kept_counts", (size_t)hm->n_layers)
         + kv_array_i32_size("ds4.expert_keep_map.original_ids", (size_t)hm->total_kept);
}

static void write_keepmap_kvs(FILE *fp, const hot_mask *hm) {
    if (!hm || !hm->active) return;
    write_kv_array_i32(fp, "ds4.expert_keep_map.kept_counts", hm->kept_counts, (size_t)hm->n_layers);
    /* original_ids: kept ids concatenated in layer order, ascending within a layer. */
    int *flat = xmalloc((size_t)(hm->total_kept > 0 ? hm->total_kept : 1) * sizeof(int));
    size_t k = 0;
    for (int l = 0; l < hm->n_layers; l++)
        for (int s = 0; s < hm->kept_counts[l]; s++) flat[k++] = hm->kept_ids[l][s];
    write_kv_array_i32(fp, "ds4.expert_keep_map.original_ids", flat, k);
    free(flat);
}

