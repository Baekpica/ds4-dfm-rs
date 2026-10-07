/* Routed expert I/O stays behind the inference boundary. The GGUF remains
 * mmap-backed; only mandatory tensors and this bounded slot pool are resident. */
#include "ds4_glm53_cache.h"

enum { GLM53_STREAM_DEFAULT_GIB = 24u };

typedef struct {
    ds4_glm53_cache_slot *slots;
    ds4_gpu_tensor *gate, *up, *down, *selected;
    unsigned char *staging;
    int32_t *ids;
    uint32_t count, ids_cap;
    uint64_t gate_stride, down_stride, staging_bytes, bytes, epoch;
    uint64_t hits, misses, read_bytes;
    bool cold;
} ds4_glm53_stream;

static uint64_t glm53_stream_gcd(uint64_t a, uint64_t b) {
    while (b) {
        const uint64_t next = a % b;
        a = b;
        b = next;
    }
    return a;
}

static uint64_t glm53_stream_round(uint64_t bytes, uint64_t align) {
    if (!align || bytes > UINT64_MAX - align + 1u) { return 0; }
    return ((bytes + align - 1u) / align) * align;
}

static bool glm53_stream_unit(const ds4_tensor *t, uint64_t *unit) {
    if (!t || !unit || t->ndim != 3u || !DS4_N_EXPERT ||
        t->dim[2] != DS4_N_EXPERT || !t->dim[0] || !t->dim[1] ||
        t->dim[0] % QK_K || !tensor_is_routed_expert_type(t->type) ||
        t->dim[0] > UINT64_MAX / t->dim[1] ||
        t->dim[0] * t->dim[1] > UINT64_MAX / DS4_N_EXPERT) { return false; }
    uint64_t bytes = 0u;
    if (!tensor_nbytes(t->type, t->dim[0] * t->dim[1] * DS4_N_EXPERT, &bytes) ||
        !bytes || t->bytes != bytes || bytes % DS4_N_EXPERT) { return false; }
    *unit = bytes / DS4_N_EXPERT;
    return true;
}

static void glm53_stream_free(ds4_glm53_stream *s) {
    if (!s) { return; }
    ds4_gpu_tensor_free(s->gate);
    ds4_gpu_tensor_free(s->up);
    ds4_gpu_tensor_free(s->down);
    ds4_gpu_tensor_free(s->selected);
    free(s->slots);
    free(s->staging);
    free(s->ids);
    memset(s, 0, sizeof(*s));
}

static void glm53_stream_report(const ds4_glm53_stream *s) {
    if (!s || !s->count) { return; }
    /* Read counters after inference ends; live readers would need locking. */
    fprintf(stderr, "ds4: SSD expert totals: hits=%llu misses=%llu read_bytes=%llu cache_bytes=%llu\n",
            (unsigned long long)s->hits, (unsigned long long)s->misses,
            (unsigned long long)s->read_bytes, (unsigned long long)s->bytes);
}

static bool glm53_stream_alloc(ds4_glm53_stream *s,
                               const ds4_model *m,
                               const ds4_weights *w,
                               const ds4_engine_options *opt) {
    if (!s || !m || !w || !opt || m->fd < 0 || m->split_count > 1u) {
        return false;
    }
    uint64_t gate_max = 0, down_max = 0, gate_align = 1, down_align = 1;
    for (uint32_t il = DS4_N_LEADING_DENSE; il < DS4_N_LAYER; il++) {
        const ds4_tensor *g = w->layer[il].ffn_gate_exps;
        const ds4_tensor *u = w->layer[il].ffn_up_exps;
        const ds4_tensor *d = w->layer[il].ffn_down_exps;
        uint64_t gb = 0u, ub = 0u, db = 0u;
        if (!glm53_stream_unit(g, &gb) || !glm53_stream_unit(u, &ub) ||
            !glm53_stream_unit(d, &db) || g->type != u->type || gb != ub ||
            g->dim[0] != DS4_N_EMBD || g->dim[1] != DS4_N_FF_EXP ||
            u->dim[0] != g->dim[0] || u->dim[1] != g->dim[1] ||
            d->dim[0] != DS4_N_FF_EXP || d->dim[1] != DS4_N_EMBD ||
            g->abs_offset > m->size || g->bytes > m->size - g->abs_offset ||
            u->abs_offset > m->size || u->bytes > m->size - u->abs_offset ||
            d->abs_offset > m->size || d->bytes > m->size - d->abs_offset) { return false; }
        if (gb > gate_max) { gate_max = gb; }
        if (db > down_max) { down_max = db; }
        uint64_t block = 0;
        if (!tensor_nbytes(g->type, QK_K, &block)) { return false; }
        gate_align = gate_align / glm53_stream_gcd(gate_align, block) * block;
        if (!tensor_nbytes(d->type, QK_K, &block)) { return false; }
        down_align = down_align / glm53_stream_gcd(down_align, block) * block;
    }
    s->gate_stride = glm53_stream_round(gate_max, gate_align);
    s->down_stride = glm53_stream_round(down_max, down_align);
    const uint64_t per = 2u * s->gate_stride + s->down_stride;
    if (!s->gate_stride || !s->down_stride || !per) { return false; }
    const uint64_t budget = opt->ssd_streaming_cache_bytes
        ? opt->ssd_streaming_cache_bytes
        : (uint64_t)GLM53_STREAM_DEFAULT_GIB * 1073741824u;
    uint64_t capacity = opt->ssd_streaming_cache_experts
        ? opt->ssd_streaming_cache_experts : budget / per;
    const uint64_t all = (uint64_t)(DS4_N_LAYER - DS4_N_LEADING_DENSE) * DS4_N_EXPERT;
    if (!opt->ssd_streaming_cache_experts && capacity > all) { capacity = all; }
    if (!per || capacity < DS4_N_EXPERT_USED || capacity > all ||
        capacity > INT_MAX || per > UINT64_MAX / capacity) {
        fprintf(stderr, "ds4: SSD expert cache must hold at least %u experts and at most %llu\n",
                DS4_N_EXPERT_USED, (unsigned long long)all);
        return false;
    }
    s->count = (uint32_t)capacity;
    s->bytes = capacity * per;
    s->cold = opt->ssd_streaming_cold;
    s->staging_bytes = gate_max > down_max ? gate_max : down_max;
    s->slots = xcalloc(s->count, sizeof(*s->slots));
    s->staging = xmalloc(s->staging_bytes);
    s->gate = ds4_gpu_weight_alloc(m->map, capacity * s->gate_stride);
    s->up = ds4_gpu_weight_alloc(m->map, capacity * s->gate_stride);
    s->down = ds4_gpu_weight_alloc(m->map, capacity * s->down_stride);
    if (!s->gate || !s->up || !s->down) {
        glm53_stream_free(s);
        return false;
    }
    fprintf(stderr, "ds4: SSD expert cache: %u global slots, %.2f GiB, %.2f MiB staging\n",
            s->count, (double)s->bytes / 1073741824.0,
            (double)s->staging_bytes / 1048576.0);
    return true;
}

static bool glm53_stream_read(ds4_glm53_stream *s, const ds4_model *m,
                              const ds4_tensor *t, uint32_t expert,
                              ds4_gpu_tensor *dst, uint64_t offset) {
    uint64_t bytes = 0u;
    if (!s || !m || m->fd < 0 || !dst || !glm53_stream_unit(t, &bytes) ||
        expert >= DS4_N_EXPERT || t->abs_offset > m->size ||
        t->bytes > m->size - t->abs_offset) { return false; }
    /* Validate the full canonical range before adding an expert offset. */
    const uint64_t start = t->abs_offset + (uint64_t)expert * bytes;
    if (bytes > s->staging_bytes || start > m->size || bytes > m->size - start) {
        return false;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t got = pread(m->fd, s->staging + done,
                                   (size_t)(bytes - done), (off_t)(start + done));
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) {
            fprintf(stderr, "ds4: SSD expert read failed at %llu: %s\n",
                    (unsigned long long)(start + done), got ? strerror(errno) : "short file");
            return false;
        }
        done += (uint64_t)got;
    }
    if (!ds4_gpu_tensor_write(dst, offset, s->staging, bytes)) { return false; }
    s->read_bytes += bytes;
#ifdef POSIX_FADV_DONTNEED
    if (s->cold) {
        /* Evict only the expert bytes just consumed; unrelated file ranges
         * and the mandatory resident tensors retain their own cache policy. */
        (void)posix_fadvise(m->fd, (off_t)start, (off_t)bytes, POSIX_FADV_DONTNEED);
    }
#endif
    return true;
}

static bool glm53_stream_select(ds4_glm53_stream *s, const ds4_model *m,
                                const ds4_layer_weights *l, uint32_t il,
                                const ds4_gpu_tensor *selected, uint32_t rows) {
    if (!s || !s->count || !rows || rows > UINT32_MAX / DS4_N_EXPERT_USED) {
        return false;
    }
    const uint32_t count = rows * DS4_N_EXPERT_USED;
    if (count > s->ids_cap) {
        int32_t *ids = realloc(s->ids, (size_t)count * sizeof(*ids));
        if (!ids) { return false; }
        s->ids = ids;
        ds4_gpu_tensor_free(s->selected);
        s->selected = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(*ids));
        if (!s->selected) { s->ids_cap = 0; return false; }
        s->ids_cap = count;
    }
    if (!s->selected || !ds4_gpu_tensor_read(selected, 0, s->ids,
                                              (uint64_t)count * sizeof(*s->ids))) {
        return false;
    }
    if (++s->epoch == 0) {
        for (uint32_t i = 0; i < s->count; i++) { s->slots[i].pinned = 0; }
        s->epoch = 1;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (s->ids[i] < 0 || (uint32_t)s->ids[i] >= DS4_N_EXPERT) { return false; }
        const int slot = glm53_cache_find(s->slots, s->count, il, s->ids[i]);
        if (slot >= 0) { s->slots[slot].pinned = s->epoch; }
    }
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t expert = (uint32_t)s->ids[i];
        int slot = glm53_cache_find(s->slots, s->count, il, expert);
        if (slot >= 0) {
            s->hits++;
        } else {
            slot = glm53_cache_victim(s->slots, s->count, s->epoch);
            if (slot < 0) {
                fprintf(stderr, "ds4: SSD routing batch exceeds %u cache slots\n", s->count);
                return false;
            }
            /* A failed partial upload must not remain a cache hit. */
            s->slots[slot].used = 0;
            if (!glm53_stream_read(s, m, l->ffn_gate_exps, expert, s->gate,
                                   (uint64_t)slot * s->gate_stride) ||
                !glm53_stream_read(s, m, l->ffn_up_exps, expert, s->up,
                                   (uint64_t)slot * s->gate_stride) ||
                !glm53_stream_read(s, m, l->ffn_down_exps, expert, s->down,
                                   (uint64_t)slot * s->down_stride)) { return false; }
            s->slots[slot].layer = il;
            s->slots[slot].expert = expert;
            s->misses++;
        }
        s->slots[slot].used = s->epoch;
        s->slots[slot].pinned = s->epoch;
        s->ids[i] = slot;
    }
    return ds4_gpu_tensor_write(s->selected, 0, s->ids,
                                (uint64_t)count * sizeof(*s->ids)) != 0;
}

static bool glm53_stream_spans(const ds4_weights *w, ds4_model_map_span_vec *spans) {
    memset(spans, 0, sizeof(*spans));
    model_map_span_vec_include_one(spans, w->token_embd);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights layer = w->layer[il];
        /* Routed tensors alone stream. Dense/shared experts, MTP controls,
         * embedding and output stay on the validated resident GPU paths. */
        layer.ffn_gate_exps = NULL;
        layer.ffn_up_exps = NULL;
        layer.ffn_down_exps = NULL;
        model_map_span_vec_include_layer(spans, &layer);
    }
    model_map_span_vec_include_output(spans, w);
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
