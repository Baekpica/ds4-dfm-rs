/* Routed expert I/O stays behind the inference boundary. The GGUF remains
 * mmap-backed; only mandatory tensors and this bounded slot pool are resident. */
#include "ds4_glm53_cache.h"

enum { GLM53_STREAM_DEFAULT_GIB = 24u };
enum { GLM53_STAGE_BANKS = 2u, GLM53_HOT_ROWS = 8u };
enum glm53_hot_copy { GLM53_HOT_KEEP, GLM53_HOT_SKIP };

typedef struct {
    ds4_glm53_cache_slot *slots;
    ds4_gpu_tensor *gate, *up, *down, *selected;
    unsigned char *staging;
    int32_t *ids;
    uint32_t count, ids_cap;
    uint64_t gate_stride, down_stride, staging_bytes, bytes, epoch;
    uint64_t hits, misses, read_bytes;
    uint64_t route_unique, routed_rows, io_ns, upload_ns, wait_ns, full_bytes, hot_kept, reread_bytes;
    ds4_gpu_upload *upload;
    unsigned char *ahead;
    uint8_t *seen;
    pthread_t reader;
    pthread_mutex_t lock;
    const ds4_model *job_model;
    const ds4_layer_weights *job_weights;
    uint32_t job_layer, job_base, active_layer, active_base, last_count, read_layer;
    uint32_t recent[GLM53_CACHE_EXPERTS_MAX], route_clock;
    uint64_t job_bytes, job_io_ns, job_upload_ns, job_reread;
    bool lock_ready, reading, stop, job_ok, grouped;
    bool cold, hot_active;
} ds4_glm53_stream;

typedef struct {
    uint64_t hits, misses, bytes, unique, reread, io, upload, wait, full;
} glm53_stream_stats;

static glm53_stream_stats glm53_stream_snap(const ds4_glm53_stream *s) {
    if (!s) { return (glm53_stream_stats){0}; }
    return (glm53_stream_stats){s->hits, s->misses, s->read_bytes,
        s->route_unique, s->reread_bytes, s->io_ns, s->upload_ns, s->wait_ns, s->full_bytes};
}

static void glm53_stream_delta(const ds4_glm53_stream *s,
        glm53_stream_stats before, uint32_t pos, uint32_t rows) {
    if (!s || !s->count || rows <= 1u) { return; }
    const glm53_stream_stats after = glm53_stream_snap(s);
    fprintf(stderr, "ds4: SSD prefill: pos=%u rows=%u hits=%llu misses=%llu unique=%llu pread_bytes=%llu reread_bytes=%llu io_ms=%.3f upload_ms=%.3f wait_ms=%.3f full_tensor_bytes=%llu\n",
        pos, rows, (unsigned long long)(after.hits - before.hits),
        (unsigned long long)(after.misses - before.misses),
        (unsigned long long)(after.unique - before.unique),
        (unsigned long long)(after.bytes - before.bytes),
        (unsigned long long)(after.reread - before.reread),
        (double)(after.io - before.io) / 1.0e6,
        (double)(after.upload - before.upload) / 1.0e6,
        (double)(after.wait - before.wait) / 1.0e6,
        (unsigned long long)(after.full - before.full));
}

static uint64_t glm53_stream_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) { return 0u; }
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static void glm53_stream_account(ds4_glm53_stream *s) {
    s->read_bytes += s->job_bytes;
    s->full_bytes += s->job_bytes;
    s->io_ns += s->job_io_ns;
    s->upload_ns += s->job_upload_ns;
    s->reread_bytes += s->job_reread;
    s->job_bytes = s->job_io_ns = s->job_upload_ns = 0u;
    s->job_reread = 0u;
}

static void glm53_stream_cancel(ds4_glm53_stream *s) {
    if (!s || !s->lock_ready) { return; }
    pthread_mutex_lock(&s->lock);
    s->stop = true;
    pthread_mutex_unlock(&s->lock);
    if (s->reading) {
        pthread_join(s->reader, NULL);
        s->reading = false;
        glm53_stream_account(s);
        // Queued groups have no committed metadata, including after a
        // complete upload which cancellation prevented us from committing.
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++) { s->slots[s->job_base + i].used = 0; }
    }
    s->active_layer = UINT32_MAX;
}

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

static void glm53_stream_hot_end(ds4_glm53_stream *s) {
    if (!s || !s->hot_active) { return; }
    for (uint32_t i = 0u; i < s->count; i++) {
        if (s->slots[i].pinned == GLM53_CACHE_HOT_PIN) { s->slots[i].pinned = 0u; }
    }
    s->hot_active = false;
}

static void glm53_stream_hot_begin(ds4_glm53_stream *s, uint32_t rows) {
    if (!s || !s->count) { return; }
    glm53_stream_hot_end(s);
    const uint32_t layers = DS4_N_LAYER - DS4_N_LEADING_DENSE;
    const uint32_t tail = GLM53_HOT_ROWS * DS4_N_EXPERT_USED;
    const uint32_t hot = tail < DS4_N_EXPERT ? tail : DS4_N_EXPERT;
    const char *value = getenv("DS4_GLM53_HOT_CACHE");
    // Leave one whole layer available while protecting recent routes from
    // earlier layers. A smaller cache uses ordinary submission pins/LRU.
    s->hot_active = rows > GLM53_HOT_ROWS && (!value || strcmp(value, "0") != 0) &&
        s->count >= DS4_N_EXPERT + (layers - 1u) * hot;
}

/* The GPU slots become the decode cache; only transient I/O state is released. */
static void glm53_stream_finish(ds4_glm53_stream *s) {
    if (!s) { return; }
    glm53_stream_cancel(s);
    const uint32_t all = (DS4_N_LAYER - DS4_N_LEADING_DENSE) * DS4_N_EXPERT;
    if (s->grouped && s->count < all) {
        for (uint32_t i = 0u; i < GLM53_STAGE_BANKS * DS4_N_EXPERT; i++) {
            s->slots[i].used = 0u;
        }
    }
    glm53_stream_hot_end(s);
    ds4_gpu_upload_free(s->upload);
    if (s->lock_ready) { pthread_mutex_destroy(&s->lock); }
    free(s->ahead);
    s->upload = NULL;
    s->ahead = NULL;
    s->lock_ready = false;
    s->grouped = false;
    s->job_model = NULL;
    s->job_weights = NULL;
}

static void glm53_stream_free(ds4_glm53_stream *s) {
    if (!s) { return; }
    glm53_stream_finish(s);
    ds4_gpu_tensor_free(s->gate);
    ds4_gpu_tensor_free(s->up);
    ds4_gpu_tensor_free(s->down);
    ds4_gpu_tensor_free(s->selected);
    free(s->slots);
    free(s->staging);
    free(s->ids);
    free(s->seen);
    memset(s, 0, sizeof(*s));
}

static void glm53_stream_report(const ds4_glm53_stream *s) {
    if (!s || !s->count) { return; }
    /* Read counters after inference ends; live readers would need locking. */
    fprintf(stderr, "ds4: SSD expert totals: hits=%llu misses=%llu pread_bytes=%llu cache_bytes=%llu unique_routes=%llu routed_rows=%llu io_ms=%.3f upload_ms=%.3f wait_ms=%.3f full_tensor_bytes=%llu hot_kept=%llu reread_bytes=%llu\n",
            (unsigned long long)s->hits, (unsigned long long)s->misses,
            (unsigned long long)s->read_bytes, (unsigned long long)s->bytes,
            (unsigned long long)s->route_unique, (unsigned long long)s->routed_rows,
            (double)s->io_ns / 1.0e6, (double)s->upload_ns / 1.0e6,
            (double)s->wait_ns / 1.0e6, (unsigned long long)s->full_bytes,
            (unsigned long long)s->hot_kept, (unsigned long long)s->reread_bytes);
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
    s->seen = xcalloc((uint64_t)DS4_N_LAYER * DS4_N_EXPERT, sizeof(*s->seen));
    s->staging = xmalloc(s->staging_bytes);
    s->active_layer = UINT32_MAX;
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
    const uint8_t bit = dst == s->gate ? 1u : dst == s->up ? 2u : 4u;
    uint8_t *seen = &s->seen[(uint64_t)s->read_layer * DS4_N_EXPERT + expert];
    const bool reread = (*seen & bit) != 0u;
    const uint64_t read_start = glm53_stream_ns();
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
        s->read_bytes += (uint64_t)got;
        if (reread) { s->reread_bytes += (uint64_t)got; }
    }
    *seen |= bit;
    s->io_ns += glm53_stream_ns() - read_start;
    const uint64_t upload_start = glm53_stream_ns();
    if (!ds4_gpu_tensor_write(dst, offset, s->staging, bytes)) { return false; }
    s->upload_ns += glm53_stream_ns() - upload_start;
#ifdef POSIX_FADV_DONTNEED
    if (s->cold) {
        /* Evict only the expert bytes just consumed; unrelated file ranges
         * and the mandatory resident tensors retain their own cache policy. */
        (void)posix_fadvise(m->fd, (off_t)start, (off_t)bytes, POSIX_FADV_DONTNEED);
    }
#endif
    return true;
}

static bool glm53_stream_stopped(ds4_glm53_stream *s) {
    pthread_mutex_lock(&s->lock);
    const bool stop = s->stop;
    pthread_mutex_unlock(&s->lock);
    return stop;
}

static bool glm53_stream_ahead(ds4_glm53_stream *s, const ds4_tensor *t,
        uint32_t expert, ds4_gpu_tensor *dst, uint64_t offset) {
    uint64_t bytes;
    const ds4_model *m = s->job_model;
    if (!glm53_stream_unit(t, &bytes) || bytes > s->staging_bytes ||
        t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) { return false; }
    const uint64_t start = t->abs_offset + (uint64_t)expert * bytes;
    uint64_t done = 0u;
    const uint8_t bit = dst == s->gate ? 1u : dst == s->up ? 2u : 4u;
    uint8_t *seen = &s->seen[(uint64_t)s->job_layer * DS4_N_EXPERT + expert];
    const bool reread = (*seen & bit) != 0u;
    const uint64_t io_start = glm53_stream_ns();
    while (done < bytes) {
        if (glm53_stream_stopped(s)) { return false; }
        const ssize_t got = pread(m->fd, s->ahead + done, bytes - done, (off_t)(start + done));
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) { return false; }
        done += (uint64_t)got;
        s->job_bytes += (uint64_t)got;
        if (reread) { s->job_reread += (uint64_t)got; }
    }
    *seen |= bit;
    s->job_io_ns += glm53_stream_ns() - io_start;
    if (glm53_stream_stopped(s)) { return false; }
    const uint64_t upload_start = glm53_stream_ns();
    const bool ok = ds4_gpu_upload_write(s->upload, dst, offset, s->ahead, bytes) != 0;
    s->job_upload_ns += glm53_stream_ns() - upload_start;
#ifdef POSIX_FADV_DONTNEED
    if (ok && s->cold) { (void)posix_fadvise(m->fd, start, bytes, POSIX_FADV_DONTNEED); }
#endif
    return ok;
}

static void *glm53_stream_worker(void *arg) {
    ds4_glm53_stream *s = arg;
    const ds4_layer_weights *l = s->job_weights;
    bool ok = true;
    for (uint32_t e = 0; ok && e < DS4_N_EXPERT; e++) {
        const uint64_t slot = s->job_base + e;
        ok = glm53_stream_ahead(s, l->ffn_gate_exps, e, s->gate, slot * s->gate_stride) &&
             glm53_stream_ahead(s, l->ffn_up_exps, e, s->up, slot * s->gate_stride) &&
             glm53_stream_ahead(s, l->ffn_down_exps, e, s->down, slot * s->down_stride);
    }
    s->job_ok = ok;
    return NULL;
}

static bool glm53_stream_join(ds4_glm53_stream *s) {
    if (!s->reading) { return true; }
    const uint64_t wait_start = glm53_stream_ns();
    pthread_join(s->reader, NULL);
    s->reading = false;
    s->wait_ns += glm53_stream_ns() - wait_start;
    glm53_stream_account(s);
    if (!s->job_ok || glm53_stream_stopped(s)) { return false; }
    if (++s->epoch == 0u) { s->epoch = 1u; }
    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
        s->slots[s->job_base + e] = (ds4_glm53_cache_slot){s->job_layer, e, s->epoch, 0u};
    }
    s->misses += DS4_N_EXPERT;
    return true;
}

static int glm53_stream_group(const ds4_glm53_stream *s, uint32_t il) {
    const uint32_t all = (DS4_N_LAYER - DS4_N_LEADING_DENSE) * DS4_N_EXPERT;
    const uint32_t groups = s->count >= all ? all / DS4_N_EXPERT : GLM53_STAGE_BANKS;
    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t base = g * DS4_N_EXPERT;
        bool ready = true;
        for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
            const ds4_glm53_cache_slot *slot = &s->slots[base + e];
            if (!slot->used || slot->layer != il || slot->expert != e) { ready = false; break; }
        }
        if (ready) { return (int)base; }
    }
    return -1;
}

static bool glm53_stream_queue(ds4_glm53_stream *s, const ds4_model *m,
        const ds4_weights *w, uint32_t il) {
    const int ready = glm53_stream_group(s, il);
    if (ready >= 0) {
        if (++s->epoch == 0u) { s->epoch = 1u; }
        for (uint32_t e = 0; e < DS4_N_EXPERT; e++) { s->slots[ready + e].used = s->epoch; }
        return true;
    }
    const uint32_t all = (DS4_N_LAYER - DS4_N_LEADING_DENSE) * DS4_N_EXPERT;
    uint32_t base = (il - DS4_N_LEADING_DENSE) * DS4_N_EXPERT;
    if (s->count < all) {
        base = UINT32_MAX;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t g = 0; g < GLM53_STAGE_BANKS; g++) {
            const uint32_t candidate = g * DS4_N_EXPERT;
            if (s->active_layer != UINT32_MAX && candidate == s->active_base) { continue; }
            const uint64_t age = s->slots[candidate].used;
            if (base == UINT32_MAX || age < oldest) { base = candidate; oldest = age; }
        }
        if (base == UINT32_MAX) { return false; }
    }
    // No group may be overwritten while earlier inference still reads it.
    if (!ds4_gpu_stream_synchronize()) { return false; }
    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) { s->slots[base + e].used = 0u; }
    s->job_model = m;
    s->job_weights = &w->layer[il];
    s->job_layer = il;
    s->job_base = base;
    s->job_bytes = s->job_io_ns = s->job_upload_ns = s->job_reread = 0u;
    s->job_ok = false;
    pthread_mutex_lock(&s->lock);
    s->stop = false;
    pthread_mutex_unlock(&s->lock);
    if (pthread_create(&s->reader, NULL, glm53_stream_worker, s) != 0) { return false; }
    s->reading = true;
    return true;
}

static bool glm53_stream_begin(ds4_glm53_stream *s, const ds4_model *m,
        const ds4_weights *w, uint32_t il) {
    if (!s || il < DS4_N_LEADING_DENSE) { return true; }
    const char *value = getenv("DS4_GLM53_PREFETCH");
    if ((value && strcmp(value, "0") == 0) || s->count < GLM53_STAGE_BANKS * DS4_N_EXPERT) { return true; }
    if (!s->lock_ready) {
        if (pthread_mutex_init(&s->lock, NULL) != 0) { return false; }
        s->upload = ds4_gpu_upload_new();
        s->ahead = malloc(s->staging_bytes);
        if (!s->upload || !s->ahead) {
            ds4_gpu_upload_free(s->upload);
            free(s->ahead);
            s->upload = NULL;
            s->ahead = NULL;
            pthread_mutex_destroy(&s->lock);
            return false;
        }
        s->lock_ready = true;
    }
    if (!glm53_stream_join(s) || !glm53_stream_queue(s, m, w, il) ||
        !glm53_stream_join(s)) { return false; }
    s->grouped = true;
    s->active_layer = il;
    memset(s->recent, 0, sizeof(s->recent));
    s->route_clock = 0u;
    const int base = glm53_stream_group(s, il);
    if (base < 0) { return false; }
    s->active_base = (uint32_t)base;
    const uint32_t end = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    return il + 1u >= end || glm53_stream_queue(s, m, w, il + 1u);
}

static int glm53_stream_recent(const ds4_glm53_stream *s, const uint8_t *seen) {
    uint32_t newest = 0u;
    int expert = -1;
    for (uint32_t e = 0u; e < DS4_N_EXPERT; e++) {
        if (!seen[e] && s->recent[e] > newest) {
            newest = s->recent[e];
            expert = (int)e;
        }
    }
    return expert;
}

static bool glm53_stream_end(ds4_glm53_stream *s, uint32_t il, enum glm53_hot_copy keep) {
    if (!s || !s->grouped || s->active_layer != il) { return true; }
    if (keep == GLM53_HOT_SKIP) {
        s->active_layer = UINT32_MAX;
        return true;
    }
    const uint32_t all = (DS4_N_LAYER - DS4_N_LEADING_DENSE) * DS4_N_EXPERT;
    const uint32_t first = GLM53_STAGE_BANKS * DS4_N_EXPERT;
    const uint32_t layers = DS4_N_LAYER - DS4_N_LEADING_DENSE;
    const uint32_t per = s->count >= all ? 0u : (s->count - first) / layers;
    const uint32_t base = first + (il - DS4_N_LEADING_DENSE) * per;
    const char *value = getenv("DS4_GLM53_HOT_COPY");
    const bool budget = !value || strcmp(value, "0") != 0;
    const uint32_t limit = budget ? DS4_N_EXPERT : GLM53_HOT_ROWS * DS4_N_EXPERT_USED;
    uint8_t seen[GLM53_CACHE_EXPERTS_MAX] = {0};
    uint32_t kept = 0u;
    // Retain recency across physical chunks: a three-row tail must not
    // discard the rest of this layer's already-funded decode hot set.
    for (uint32_t n = 0u; n < limit && kept < per; n++) {
        int expert = budget ? glm53_stream_recent(s, seen) : -1;
        if (!budget) {
            if (n >= s->last_count) { break; }
            expert = s->ids[s->last_count - 1u - n] - (int32_t)s->active_base;
        }
        if (expert < 0 && budget) { break; }
        const uint32_t src = s->active_base + (uint32_t)expert;
        if (src < s->active_base || src >= s->active_base + DS4_N_EXPERT) { return false; }
        if (seen[expert]) { continue; }
        seen[expert] = 1u;
        const uint32_t dst = base + kept++;
        s->slots[dst].used = 0u;
        if (!ds4_gpu_tensor_copy(s->gate, (uint64_t)dst * s->gate_stride,
                s->gate, (uint64_t)src * s->gate_stride, s->gate_stride) ||
            !ds4_gpu_tensor_copy(s->up, (uint64_t)dst * s->gate_stride,
                s->up, (uint64_t)src * s->gate_stride, s->gate_stride) ||
            !ds4_gpu_tensor_copy(s->down, (uint64_t)dst * s->down_stride,
                s->down, (uint64_t)src * s->down_stride, s->down_stride)) { return false; }
        s->slots[dst] = (ds4_glm53_cache_slot){il, expert, s->epoch, 0u};
    }
    s->hot_kept += kept;
    s->active_layer = UINT32_MAX;
    return true;
}

static bool glm53_stream_routes(ds4_glm53_stream *s,
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
    return true;
}

static bool glm53_stream_stage(ds4_glm53_stream *s, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il, int32_t *ids, uint32_t rows) {
    if (!s || !ids || !rows || il >= DS4_N_LAYER || rows > s->ids_cap / DS4_N_EXPERT_USED ||
        glm53_cache_rows(ids, rows, DS4_N_EXPERT_USED, DS4_N_EXPERT, s->count) != rows) {
        return false;
    }
    /* A second routed sub-batch uses host IDs already read. Drain its
     * predecessor before reusing any slot, just as the first route read did. */
    if (!ds4_gpu_stream_synchronize()) { return false; }
    const uint32_t count = rows * DS4_N_EXPERT_USED;
    uint8_t unique[GLM53_CACHE_EXPERTS_MAX] = {0};
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_EXPERT) { return false; }
        if (!unique[ids[i]]) { s->route_unique++; unique[ids[i]] = 1u; }
    }
    s->routed_rows += rows;
    s->read_layer = il;
    s->last_count = count;
    if (s->grouped && s->active_layer == il) {
        if (count > UINT32_MAX - s->route_clock) { return false; }
        if (++s->epoch == 0u) { s->epoch = 1u; }
        for (uint32_t e = 0; e < DS4_N_EXPERT; e++) { s->slots[s->active_base + e].used = s->epoch; }
        for (uint32_t i = 0; i < count; i++) {
            s->recent[ids[i]] = ++s->route_clock;
            ids[i] += (int32_t)s->active_base;
        }
        s->hits += count;
        return ds4_gpu_tensor_write(s->selected, 0u, ids, (uint64_t)count * sizeof(*ids)) != 0;
    }
    if (s->hot_active) {
        for (uint32_t i = 0u; i < s->count; i++) {
            if (s->slots[i].layer == il && s->slots[i].pinned == GLM53_CACHE_HOT_PIN) {
                s->slots[i].pinned = 0u;
            }
        }
    }
    if (++s->epoch == 0) {
        for (uint32_t i = 0; i < s->count; i++) { s->slots[i].pinned = 0; }
        s->epoch = 1;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_EXPERT) { return false; }
        const int slot = glm53_cache_find(s->slots, s->count, il, ids[i]);
        if (slot >= 0) { s->slots[slot].pinned = s->epoch; }
    }
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t expert = (uint32_t)ids[i];
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
        ids[i] = slot;
    }
    if (s->hot_active) {
        const uint32_t tail = GLM53_HOT_ROWS * DS4_N_EXPERT_USED;
        const uint32_t start = count > tail ? count - tail : 0u;
        for (uint32_t i = start; i < count; i++) {
            ds4_glm53_cache_slot *slot = &s->slots[ids[i]];
            if (slot->pinned != GLM53_CACHE_HOT_PIN) { s->hot_kept++; }
            slot->pinned = GLM53_CACHE_HOT_PIN;
        }
    }
    return ds4_gpu_tensor_write(s->selected, 0, ids,
                                (uint64_t)count * sizeof(*ids)) != 0;
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
