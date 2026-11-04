/* Gemma-4 SSD expert cache. This is the Gemma-only counterpart of
 * ds4_glm53_stream.inc: a bounded device slot pool, an LRU with pin-before-
 * miss admission, pread from the model fd and the same stats vocabulary.
 * The GLM module is read as the contract, never edited.
 *
 * The option is opt-in: ds4_engine_open only reaches this file when
 * --ssd-streaming was given and the family gate admitted Gemma-4. Nothing in
 * the Gemma launcher or session path turns it on by itself.
 *
 * A streamed VQ1 record is copied byte for byte into a device slot. Its
 * header stores gain/main/plane as offsets relative to the record's data
 * start (ds4_glmvq_gpu.cuh gvq_link: data + main, data + gain), so the
 * source entry is rebased by setting s.data = slot and re-deriving the
 * three pointers. The per-layer book and residual book stay resident
 * records. The linear path (ds4_gpu_gvq_linear -> glmvq_gemm) reads the
 * host glmvq_source table, so the rebind is visible to the current MoE
 * path; the device source table belongs to the shared route/decode kernels,
 * which this port does not use.
 *
 * ds4_gpu_set_stream_map refuses without coherent CUDA host page tables
 * (the GB10 shape); ds4_gemma4_map.inc is the discrete arm of that call, and
 * ds4_engine_open takes it for Gemma-4 (ds4.c). On a discrete card the
 * resident spans are device copies published as device ranges and the experts
 * stay host-cold; the default (resident) path never reaches either. */

#ifndef DS4_GEMMA4_STREAM_INC
#define DS4_GEMMA4_STREAM_INC

enum {
    /* A bounded default, not the full 5.9 GiB expert set: a cache that holds
     * every expert defeats the option's purpose. --ssd-streaming-cache-
     * experts and --ssd-streaming-cache-experts GB override it. */
    G4_STREAM_DEFAULT_GIB = 4u,
    G4_STREAM_ALIGN = 256u,
    G4_STREAM_HEADER = 128u,
};

typedef struct {
    uint32_t layer, expert;
    uint64_t used, pinned;
} ds4_gemma4_stream_slot;

typedef struct {
    uint64_t hits, misses, read_bytes, reread_bytes;
    uint64_t route_unique, routed_rows;
    uint64_t io_ns, upload_ns;
} ds4_gemma4_stream_stats;

struct ds4_gemma4_stream {
    const ds4_model *model;
    ds4_gvq_model *plan;
    ds4_gpu_tensor *gate, *up, *down;
    ds4_gpu_tensor *gate_res, *up_res, *down_res;
    uint64_t gate_stride, up_stride, down_stride, res_stride;
    uint32_t count;
    uint64_t bytes;
    ds4_gemma4_stream_slot *slots;
    uint8_t *seen;
    uint8_t *staging;
    uint64_t staging_bytes;
    int32_t *ids;
    uint32_t ids_cap;
    uint32_t epoch;
    uint32_t read_layer;
    /* The artifact may be split across GGUF shards. model_open_split maps
     * them into one page-aligned virtual range, so a record's absolute
     * offset is rebased by the shard's mapping base before pread. */
    uint32_t shard_count;
    int *shard_fds;
    uint64_t *shard_base;
    uint64_t *shard_end;
    bool cold;
    bool residual;
    ds4_gemma4_stream_stats stats;
};

static uint64_t gemma4_stream_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) { return 0; }
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static uint64_t gemma4_stream_round(uint64_t bytes, uint64_t align) {
    if (!align) { return bytes; }
    return (bytes + align - 1u) / align * align;
}

static uint32_t gemma4_stream_index(const ds4_model *m, const ds4_tensor *t) {
    return t ? (uint32_t)(t - m->tensors) : DS4_GVQ_NO_LINK;
}

/* Return a whole-row prefix whose route union fits. Repeated top-k IDs
 * consume one slot, exactly like glm53_cache_rows. */
static uint32_t gemma4_stream_rows(const int32_t *ids, uint32_t rows,
        uint32_t used, uint32_t experts, uint32_t capacity) {
    if (!ids || !used || used > experts || experts > G4_EXPERTS || !capacity ||
        rows > UINT32_MAX / used) { return 0u; }
    uint8_t seen[G4_EXPERTS] = {0};
    uint32_t unique = 0u;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t i = 0; i < used; i++) {
            const int32_t id = ids[(uint64_t)row * used + i];
            if (id < 0 || (uint32_t)id >= experts) { return 0u; }
            if (seen[id]) { continue; }
            seen[id] = 1u;
            if (++unique > capacity) { return row; }
        }
    }
    return rows;
}

static int gemma4_stream_find(const ds4_gemma4_stream_slot *slots,
        uint32_t count, uint32_t layer, uint32_t expert) {
    for (uint32_t i = 0; i < count; i++) {
        if (slots[i].used && slots[i].layer == layer && slots[i].expert == expert) {
            return (int)i;
        }
    }
    return -1;
}

static int gemma4_stream_victim(const ds4_gemma4_stream_slot *slots,
        uint32_t count, uint64_t epoch) {
    int victim = -1;
    for (uint32_t i = 0; i < count; i++) {
        if (slots[i].pinned == epoch) { continue; }
        if (!slots[i].used) { return (int)i; }
        if (victim < 0 || slots[i].used < slots[victim].used) { victim = (int)i; }
    }
    return victim;
}

static void gemma4_stream_report(const ds4_gemma4_stream *s) {
    if (!s || (!s->stats.hits && !s->stats.misses)) { return; }
    fprintf(stderr,
        "ds4: Gemma-4 SSD cache: hits=%llu misses=%llu unique=%llu rows=%llu "
        "read=%.2f GiB reread=%.2f GiB io=%.1f ms upload=%.1f ms\n",
        (unsigned long long)s->stats.hits,
        (unsigned long long)s->stats.misses,
        (unsigned long long)s->stats.route_unique,
        (unsigned long long)s->stats.routed_rows,
        (double)s->stats.read_bytes / 1073741824.0,
        (double)s->stats.reread_bytes / 1073741824.0,
        (double)s->stats.io_ns / 1.0e6,
        (double)s->stats.upload_ns / 1.0e6);
}

static void gemma4_stream_free(ds4_gemma4_stream *s) {
    /* The discrete map's cold-header device ranges; a no-op unless
     * --ssd-streaming installed the Gemma discrete plan. */
    ds4_gpu_g4_stream_plan_release();
    if (!s) { return; }
    gemma4_stream_report(s);
    ds4_gpu_tensor_free(s->gate);
    ds4_gpu_tensor_free(s->up);
    ds4_gpu_tensor_free(s->down);
    ds4_gpu_tensor_free(s->gate_res);
    ds4_gpu_tensor_free(s->up_res);
    ds4_gpu_tensor_free(s->down_res);
    free(s->slots);
    free(s->seen);
    free(s->staging);
    free(s->ids);
    free(s->shard_fds);
    free(s->shard_base);
    free(s->shard_end);
    free(s);
}

static bool gemma4_stream_alloc(ds4_gemma4_stream *s, const ds4_model *m,
        const ds4_weights *w, const ds4_engine_options *opt) {
    if (!s || !m || !w || !w->gemma4 || !opt || m->fd < 0) {
        return false;
    }
    memset(s, 0, sizeof(*s));
    s->model = m;
    s->cold = opt->ssd_streaming_cold;
    /* Reconstruct model_open_split's page-aligned shard bases. The combined
     * map is one reserved range; a tensor's abs_offset is relative to it, so
     * pread needs the shard base and the shard's own fd. */
    s->shard_count = m->split_count ? m->split_count : 1u;
    s->shard_fds = xmalloc((size_t)s->shard_count * sizeof(*s->shard_fds));
    s->shard_base = xmalloc((size_t)s->shard_count * sizeof(*s->shard_base));
    s->shard_end = xmalloc((size_t)s->shard_count * sizeof(*s->shard_end));
    const long page_l = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_l > 0 ? (uint64_t)page_l : 4096u;
    uint64_t shard_at = 0;
    for (uint32_t k = 0; k < s->shard_count; k++) {
        s->shard_fds[k] = k == 0 ? m->fd : m->split_fds[k - 1u];
        struct stat st;
        if (fstat(s->shard_fds[k], &st) != 0 || st.st_size < 32) { return false; }
        s->shard_base[k] = shard_at;
        s->shard_end[k] = shard_at + (uint64_t)st.st_size;
        if (s->shard_end[k] > m->size) { return false; }
        const uint64_t span = gemma4_stream_round((uint64_t)st.st_size, page);
        if (span > UINT64_MAX - shard_at) { return false; }
        shard_at += span;
    }
    if (s->shard_count > 1u && shard_at != m->size) { return false; }
    uint64_t gate_max = 0, up_max = 0, down_max = 0, res_max = 0;
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        const ds4_gemma4_layer *l = &w->gemma4[il];
        for (uint32_t e = 0; e < G4_EXPERTS; e++) {
            if (!l->gate[e] || !l->up[e] || !l->down[e]) { return false; }
            if (l->gate[e]->bytes > gate_max) { gate_max = l->gate[e]->bytes; }
            if (l->up[e]->bytes > up_max) { up_max = l->up[e]->bytes; }
            if (l->down[e]->bytes > down_max) { down_max = l->down[e]->bytes; }
            const bool any_res = l->gate_res[e] || l->up_res[e] || l->down_res[e];
            if (any_res && (!l->gate_res[e] || !l->up_res[e] || !l->down_res[e])) {
                return false;
            }
            if (any_res) {
                s->residual = true;
                if (l->gate_res[e]->bytes > res_max) { res_max = l->gate_res[e]->bytes; }
                if (l->up_res[e]->bytes > res_max) { res_max = l->up_res[e]->bytes; }
                if (l->down_res[e]->bytes > res_max) { res_max = l->down_res[e]->bytes; }
            }
        }
    }
    if (!gate_max || !up_max || !down_max) { return false; }
    s->gate_stride = gemma4_stream_round(gate_max, G4_STREAM_ALIGN);
    s->up_stride = gemma4_stream_round(up_max, G4_STREAM_ALIGN);
    s->down_stride = gemma4_stream_round(down_max, G4_STREAM_ALIGN);
    s->res_stride = s->residual ? gemma4_stream_round(res_max, G4_STREAM_ALIGN) : 0u;
    const uint64_t per = s->gate_stride + s->up_stride + s->down_stride +
        (s->residual ? 3u * s->res_stride : 0u);
    if (!per) { return false; }
    const uint64_t all = (uint64_t)G4_LAYERS * G4_EXPERTS;
    uint64_t capacity;
    if (opt->ssd_streaming_cache_experts) {
        capacity = opt->ssd_streaming_cache_experts;
    } else if (opt->ssd_streaming_cache_bytes) {
        capacity = opt->ssd_streaming_cache_bytes / per;
    } else {
        capacity = (uint64_t)G4_STREAM_DEFAULT_GIB * 1073741824u / per;
    }
    if (capacity > all) { capacity = all; }
    if (capacity < G4_USED || capacity > UINT32_MAX) {
        fprintf(stderr, "ds4: Gemma-4 SSD expert cache must hold at least %u experts\n",
                (unsigned)G4_USED);
        return false;
    }
    s->count = (uint32_t)capacity;
    s->bytes = capacity * per;
    s->staging_bytes = gate_max;
    if (up_max > s->staging_bytes) { s->staging_bytes = up_max; }
    if (down_max > s->staging_bytes) { s->staging_bytes = down_max; }
    if (res_max > s->staging_bytes) { s->staging_bytes = res_max; }
    s->staging = xmalloc(s->staging_bytes);
    s->slots = xcalloc(s->count, sizeof(*s->slots));
    s->seen = xcalloc(all, sizeof(*s->seen));
    s->gate = ds4_gpu_weight_alloc(m->map, capacity * s->gate_stride);
    s->up = ds4_gpu_weight_alloc(m->map, capacity * s->up_stride);
    s->down = ds4_gpu_weight_alloc(m->map, capacity * s->down_stride);
    if (s->residual) {
        s->gate_res = ds4_gpu_weight_alloc(m->map, capacity * s->res_stride);
        s->up_res = ds4_gpu_weight_alloc(m->map, capacity * s->res_stride);
        s->down_res = ds4_gpu_weight_alloc(m->map, capacity * s->res_stride);
    }
    if (!s->gate || !s->up || !s->down ||
        (s->residual && (!s->gate_res || !s->up_res || !s->down_res))) {
        return false;
    }
    fprintf(stderr,
        "ds4: Gemma-4 SSD expert cache: %u/%llu slots, %.2f GiB, %.2f MiB staging%s\n",
        s->count, (unsigned long long)all, (double)s->bytes / 1073741824.0,
        (double)s->staging_bytes / 1048576.0, s->residual ? ", R13+8" : "");
    return true;
}

static bool gemma4_stream_attach(ds4_gemma4_stream *s, ds4_gvq_model *plan) {
    if (!s || !plan) { return false; }
    s->plan = plan;
    return true;
}

/* The resident set: every tensor except the routed expert matrices and their
 * residual planes. Books, norms, attention, dense FFN, the token table and
 * the router stay directly addressable; the expert records arrive through
 * the slot pool. */
static bool gemma4_stream_spans(const ds4_weights *w, const ds4_model *m,
        ds4_model_map_span_vec *spans) {
    if (!w || !w->gemma4 || !m || !m->tensors || !m->n_tensors || !spans) {
        return false;
    }
    memset(spans, 0, sizeof(*spans));
    uint8_t *streamed = xcalloc(m->n_tensors, sizeof(*streamed));
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        const ds4_gemma4_layer *l = &w->gemma4[il];
        for (uint32_t e = 0; e < G4_EXPERTS; e++) {
            const uint32_t indices[] = {
                gemma4_stream_index(m, l->gate[e]),
                gemma4_stream_index(m, l->up[e]),
                gemma4_stream_index(m, l->down[e]),
                gemma4_stream_index(m, l->gate_res[e]),
                gemma4_stream_index(m, l->up_res[e]),
                gemma4_stream_index(m, l->down_res[e]),
            };
            for (uint32_t i = 0; i < sizeof(indices) / sizeof(indices[0]); i++) {
                if (indices[i] != DS4_GVQ_NO_LINK && indices[i] < m->n_tensors) {
                    streamed[indices[i]] = 1u;
                }
            }
        }
    }
    for (uint32_t i = 0; i < m->n_tensors; i++) {
        if (!streamed[i]) { model_map_span_vec_include_one(spans, &m->tensors[i]); }
    }
    free(streamed);
    qsort(spans->v, spans->len, sizeof(spans->v[0]), model_map_span_cmp);
    uint32_t out = 0;
    for (uint32_t i = 0; i < spans->len; i++) {
        if (out == 0 || spans->v[i].off > spans->v[out - 1u].end) {
            spans->v[out++] = spans->v[i];
        } else if (spans->v[i].end > spans->v[out - 1u].end) {
            spans->v[out - 1u].end = spans->v[i].end;
        }
    }
    spans->len = out;
    return spans->len != 0 && spans->max_tensor_bytes != 0;
}

/* The cold record set: every routed expert VQ record and, on an R13+8
 * artifact, its residual plane -- exactly the records gemma4_stream_spans
 * leaves out of the resident plan. `indices` holds 6 * G4_LAYERS * G4_EXPERTS
 * entries; the returned count is ascending and duplicate-free. The discrete
 * map (ds4_gemma4_map.inc) publishes one GVQ_HEADER-sized device range per
 * entry so the bind-time header gather never reads the host map. */
static int gemma4_stream_index_cmp(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static uint32_t gemma4_stream_cold_records(const ds4_weights *w,
        const ds4_model *m, uint32_t *indices) {
    if (!w || !w->gemma4 || !m || !indices) { return 0u; }
    uint32_t n = 0u;
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        const ds4_gemma4_layer *l = &w->gemma4[il];
        for (uint32_t e = 0; e < G4_EXPERTS; e++) {
            const ds4_tensor *const ts[] = {
                l->gate[e], l->up[e], l->down[e],
                l->gate_res[e], l->up_res[e], l->down_res[e],
            };
            for (uint32_t k = 0; k < sizeof(ts) / sizeof(ts[0]); k++) {
                const uint32_t i = gemma4_stream_index(m, ts[k]);
                if (i != DS4_GVQ_NO_LINK) { indices[n++] = i; }
            }
        }
    }
    qsort(indices, n, sizeof(indices[0]), gemma4_stream_index_cmp);
    return n;
}

static bool gemma4_stream_read(ds4_gemma4_stream *s, const ds4_tensor *t,
        ds4_gpu_tensor *dst, uint64_t offset, uint8_t *header) {
    if (!s || !t || !dst || !s->model || s->model->fd < 0 ||
        t->abs_offset > s->model->size || t->bytes > s->model->size - t->abs_offset ||
        t->bytes > s->staging_bytes || t->bytes < G4_STREAM_HEADER) {
        return false;
    }
    /* The slot was sized as stride >= t->bytes at allocation, and `offset` is
     * slot * stride, so the upload cannot leave the tensor. */
    uint32_t shard = s->shard_count;
    for (uint32_t k = 0; k < s->shard_count; k++) {
        if (t->abs_offset >= s->shard_base[k] &&
            t->bytes <= s->shard_end[k] - t->abs_offset) {
            shard = k;
            break;
        }
    }
    if (shard == s->shard_count) {
        fprintf(stderr, "ds4: Gemma-4 SSD record crosses a shard boundary at %llu\n",
            (unsigned long long)t->abs_offset);
        return false;
    }
    const uint64_t local = t->abs_offset - s->shard_base[shard];
    uint64_t done = 0;
    const uint64_t read_start = gemma4_stream_ns();
    while (done < t->bytes) {
        const ssize_t got = pread(s->shard_fds[shard], s->staging + done,
            (size_t)(t->bytes - done), (off_t)(local + done));
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) {
            fprintf(stderr, "ds4: Gemma-4 SSD read failed at %llu: %s\n",
                (unsigned long long)(t->abs_offset + done),
                got ? strerror(errno) : "short file");
            return false;
        }
        done += (uint64_t)got;
        s->stats.read_bytes += (uint64_t)got;
    }
    s->stats.io_ns += gemma4_stream_ns() - read_start;
    if (header) { memcpy(header, s->staging, G4_STREAM_HEADER); }
    const uint64_t upload_start = gemma4_stream_ns();
    if (!ds4_gpu_tensor_write(dst, offset, s->staging, t->bytes)) { return false; }
    s->stats.upload_ns += gemma4_stream_ns() - upload_start;
#ifdef POSIX_FADV_DONTNEED
    if (s->cold) {
        (void)posix_fadvise(s->shard_fds[shard], (off_t)local, (off_t)t->bytes,
            POSIX_FADV_DONTNEED);
    }
#endif
    return true;
}

static bool gemma4_stream_load(ds4_gemma4_stream *s, const ds4_gemma4_layer *l,
        uint32_t il, uint32_t e, uint32_t slot) {
    if (!s || !s->plan || !l || e >= G4_EXPERTS || slot >= s->count) { return false; }
    const uint64_t gate_off = (uint64_t)slot * s->gate_stride;
    const uint64_t up_off = (uint64_t)slot * s->up_stride;
    const uint64_t down_off = (uint64_t)slot * s->down_stride;
    const uint64_t res_off = (uint64_t)slot * s->res_stride;
    uint8_t mh[G4_STREAM_HEADER], rh[G4_STREAM_HEADER];
    const bool reread = s->seen[(uint64_t)il * G4_EXPERTS + e] != 0u;
    uint64_t bytes = 0;

    if (!gemma4_stream_read(s, l->gate[e], s->gate, gate_off, mh)) { return false; }
    bytes += l->gate[e]->bytes;
    if (s->residual && !gemma4_stream_read(s, l->gate_res[e], s->gate_res, res_off, rh)) {
        return false;
    }
    if (s->residual) { bytes += l->gate_res[e]->bytes; }
    if (!ds4_gpu_g4_stream_rebind(s->plan, gemma4_stream_index(s->model, l->gate[e]),
            s->gate, gate_off, l->gate[e]->bytes, mh,
            s->residual ? gemma4_stream_index(s->model, l->gate_res[e]) : DS4_GVQ_NO_LINK,
            s->residual ? s->gate_res : NULL, res_off,
            s->residual ? l->gate_res[e]->bytes : 0u,
            s->residual ? rh : NULL)) { return false; }

    if (!gemma4_stream_read(s, l->up[e], s->up, up_off, mh)) { return false; }
    bytes += l->up[e]->bytes;
    if (s->residual && !gemma4_stream_read(s, l->up_res[e], s->up_res, res_off, rh)) {
        return false;
    }
    if (s->residual) { bytes += l->up_res[e]->bytes; }
    if (!ds4_gpu_g4_stream_rebind(s->plan, gemma4_stream_index(s->model, l->up[e]),
            s->up, up_off, l->up[e]->bytes, mh,
            s->residual ? gemma4_stream_index(s->model, l->up_res[e]) : DS4_GVQ_NO_LINK,
            s->residual ? s->up_res : NULL, res_off,
            s->residual ? l->up_res[e]->bytes : 0u,
            s->residual ? rh : NULL)) { return false; }

    if (!gemma4_stream_read(s, l->down[e], s->down, down_off, mh)) { return false; }
    bytes += l->down[e]->bytes;
    if (s->residual && !gemma4_stream_read(s, l->down_res[e], s->down_res, res_off, rh)) {
        return false;
    }
    if (s->residual) { bytes += l->down_res[e]->bytes; }
    if (!ds4_gpu_g4_stream_rebind(s->plan, gemma4_stream_index(s->model, l->down[e]),
            s->down, down_off, l->down[e]->bytes, mh,
            s->residual ? gemma4_stream_index(s->model, l->down_res[e]) : DS4_GVQ_NO_LINK,
            s->residual ? s->down_res : NULL, res_off,
            s->residual ? l->down_res[e]->bytes : 0u,
            s->residual ? rh : NULL)) { return false; }

    if (reread) { s->stats.reread_bytes += bytes; }
    s->seen[(uint64_t)il * G4_EXPERTS + e] = 1u;
    return true;
}

/* Read the next `rows` routes, stage the largest prefix whose expert union
 * fits the cache, and return that prefix. The caller runs the MoE over the
 * same prefix with the ids left in s->ids. */
static uint32_t gemma4_stream_stage(ds4_gemma4_stream *s, const ds4_model *m,
        const ds4_weights *w, uint32_t il, const ds4_gpu_tensor *selected,
        uint32_t first, uint32_t rows) {
    if (!s || !s->plan || !m || !w || !w->gemma4 || il >= G4_LAYERS ||
        !selected || !rows) { return 0u; }
    const uint64_t count = (uint64_t)rows * G4_USED;
    if (count > s->ids_cap) {
        int32_t *ids = realloc(s->ids, (size_t)count * sizeof(*ids));
        if (!ids) { return 0u; }
        s->ids = ids;
        s->ids_cap = (uint32_t)count;
    }
    if (!ds4_gpu_tensor_read(selected,
            (uint64_t)first * G4_USED * sizeof(int32_t), s->ids,
            count * sizeof(int32_t))) { return 0u; }
    const uint32_t chunk = gemma4_stream_rows(s->ids, rows, G4_USED, G4_EXPERTS, s->count);
    if (!chunk) { return 0u; }
    const ds4_gemma4_layer *l = &w->gemma4[il];
    s->read_layer = il;
    s->stats.routed_rows += chunk;
    if (++s->epoch == 0u) {
        for (uint32_t i = 0; i < s->count; i++) { s->slots[i].pinned = 0u; }
        s->epoch = 1u;
    }
    uint8_t unique[G4_EXPERTS] = {0};
    for (uint64_t i = 0; i < (uint64_t)chunk * G4_USED; i++) {
        const int32_t id = s->ids[i];
        if (id < 0 || (uint32_t)id >= G4_EXPERTS) { return 0u; }
        if (!unique[id]) { s->stats.route_unique++; unique[id] = 1u; }
    }
    for (uint64_t i = 0; i < (uint64_t)chunk * G4_USED; i++) {
        const int slot = gemma4_stream_find(s->slots, s->count, il,
            (uint32_t)s->ids[i]);
        if (slot >= 0) { s->slots[slot].pinned = s->epoch; }
    }
    for (uint64_t i = 0; i < (uint64_t)chunk * G4_USED; i++) {
        const uint32_t expert = (uint32_t)s->ids[i];
        int slot = gemma4_stream_find(s->slots, s->count, il, expert);
        if (slot >= 0) {
            s->stats.hits++;
        } else {
            slot = gemma4_stream_victim(s->slots, s->count, s->epoch);
            if (slot < 0) {
                fprintf(stderr, "ds4: Gemma-4 SSD routing batch exceeds %u cache slots\n",
                        s->count);
                return 0u;
            }
            /* A failed partial upload must not remain a cache hit. */
            s->slots[slot].used = 0u;
            if (!gemma4_stream_load(s, l, il, expert, (uint32_t)slot)) { return 0u; }
            s->slots[slot].layer = il;
            s->slots[slot].expert = expert;
            s->stats.misses++;
        }
        s->slots[slot].used = s->epoch;
        s->slots[slot].pinned = s->epoch;
    }
    return chunk;
}

#endif
