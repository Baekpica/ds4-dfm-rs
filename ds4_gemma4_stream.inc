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
    /* Records one layer's fill can hand out: gate, up and down plus their
     * three residual planes, for every expert of the layer. The staging ring
     * must hold at least this many slices, or a fill wraps mid-layer and the
     * wrap has to drain behind the previous layer's compute (measured: 30
     * wraps cost 4.9 s of a 5.0 s fill, 2026-11-05). */
    G4_STREAM_LAYER_SLICES = G4_EXPERTS * 6u,
};

typedef struct {
    uint32_t layer, expert;
    uint64_t used, pinned;
} ds4_gemma4_stream_slot;

typedef struct {
    uint64_t hits, misses, read_bytes, reread_bytes;
    uint64_t route_unique, routed_rows;
    uint64_t io_ns, upload_ns, wait_ns, scan_ns;
    uint64_t stage_copy_ns, submit_ns;
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
    /* Per-layer mapping (layer, expert) -> slot, or -1 when the record is
     * not resident. It replaces the linear walk over `slots` that used to
     * resolve every route id: at a 2,048-row chunk that walk cost 2.2 s of a
     * 12.3 s prefill (measured 2026-11-05). `free_slots` counts slots never
     * used yet, so has_free stays O(1). */
    int32_t *slot_map;
    uint32_t free_slots;
    uint8_t *seen;
    /* The staging ring: pinned host memory that a record is copied into
     * before it is submitted to the upload stream. `staging_slot` is the
     * largest record rounded to G4_STREAM_ALIGN (0.39 MiB today) and
     * `staging_used` is the next free byte; a batch drains (publish + sync)
     * before the ring wraps or before the caller launches the MoE kernel
     * over the slots it just filled. */
    uint8_t *staging;
    uint64_t staging_bytes;
    uint64_t staging_slot;
    uint32_t staging_slots;
    uint64_t staging_used;
    bool staging_pinned;
    /* Queued copies are only usable outside a CUDA graph capture; `queued`
     * is set per staged batch by gemma4_stream_stage. */
    ds4_gpu_upload *upload;
    bool queued;
    bool ordered;
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
    /* Fill from the model mmap instead of the shard fds. The cache lane
     * (DS4_GEMMA4_MOE_CACHE_MIB) always sets this: its experts are cold in
     * the mmap by construction and there is no --ssd-streaming to give the
     * slots a file to read. */
    bool host_source;
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

/* Diagnostic switch for the queued path (DS4_GEMMA4_MOE_STAGING_QUEUED=0
 * restores the blocking ds4_gpu_tensor_write fill). Measured against each
 * other: from a file-backed source the driver's blocking copy
 * is already as fast as an explicit memcpy plus a queued transfer, so the
 * ring only pays once the copies overlap compute. Like the other
 * diagnostics this stays an environment variable; it changes no arithmetic. */
static bool gemma4_stream_queued_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_GEMMA4_MOE_STAGING_QUEUED");
        on = (v && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Drain the copies queued on the upload stream and recycle the ring. The
 * publish step orders the caller's stream after the copies (so the next
 * kernel reads what was just written) and the sync is what makes it safe to
 * overwrite a ring slice; the wait is charged to upload_ns so the report
 * line keeps meaning "time spent moving records", not "time spent
 * submitting them". */
static bool gemma4_stream_ring_flush(ds4_gemma4_stream *s) {
    if (!s) { return false; }
    const bool pending = s->staging_used != 0u;
    s->staging_used = 0u;
    if (!s->upload || !pending) { return true; }
    const uint64_t start = gemma4_stream_ns();
    const bool ok = ds4_gpu_upload_publish(s->upload) &&
                    ds4_gpu_upload_sync(s->upload);
    s->stats.wait_ns += gemma4_stream_ns() - start;
    return ok;
}

/* The next ring slice for `bytes`, or NULL when the record is larger than a
 * slice. Wrapping drains first: a slice is never handed out while a copy
 * into it may still be in flight. */
static uint8_t *gemma4_stream_ring_take(ds4_gemma4_stream *s, uint64_t bytes) {
    if (!s || !s->staging || !bytes || bytes > s->staging_slot) { return NULL; }
    const uint64_t need = gemma4_stream_round(bytes, G4_STREAM_ALIGN);
    if (s->staging_used + need > s->staging_bytes) {
        if (!gemma4_stream_ring_flush(s)) { return NULL; }
    }
    uint8_t *slice = s->staging + s->staging_used;
    s->staging_used += need;
    return slice;
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

/* The resident slot for (layer, expert), or -1. The per-layer map holds one
 * entry per expert, so this is the same answer the linear walk gave, in one
 * load (llama.cpp publishes the same shape as its per-layer slot map,
 * src/llama-moe-cache.cpp:614). */
static int gemma4_stream_find(const ds4_gemma4_stream *s, uint32_t layer,
        uint32_t expert) {
    if (!s || !s->slot_map || layer >= G4_LAYERS || expert >= G4_EXPERTS) {
        return -1;
    }
    return s->slot_map[(uint64_t)layer * G4_EXPERTS + expert];
}

/* Drop whatever (layer, expert) `slot` held from the map. The slot itself is
 * left alone; the caller decides whether it becomes free or refilled. */
static void gemma4_stream_map_drop(ds4_gemma4_stream *s, uint32_t slot) {
    if (!s || !s->slot_map || slot >= s->count) { return; }
    const ds4_gemma4_stream_slot *old = &s->slots[slot];
    if (!old->used) { return; }
    if (old->layer < G4_LAYERS && old->expert < G4_EXPERTS) {
        s->slot_map[(uint64_t)old->layer * G4_EXPERTS + old->expert] = -1;
    }
}

/* Record (layer, expert) as living in `slot`. Called on every fill, so the
 * map can never disagree with `slots`. */
static void gemma4_stream_map_set(ds4_gemma4_stream *s, uint32_t slot,
        uint32_t layer, uint32_t expert) {
    if (!s || !s->slot_map || slot >= s->count || layer >= G4_LAYERS ||
        expert >= G4_EXPERTS) {
        return;
    }
    s->slots[slot].layer = layer;
    s->slots[slot].expert = expert;
    s->slot_map[(uint64_t)layer * G4_EXPERTS + expert] = (int32_t)slot;
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

/* Cache-miss fallback (DS4_GEMMA4_MOE_MISS=cpu): is the expert resident in
 * this layer, does the pool still have a free slot, and which expert of the
 * layer is resident (the zero-weight stand-in a host-computed route is
 * mapped to in the GPU chunk)? */
static int gemma4_stream_slot_of(const ds4_gemma4_stream *s, uint32_t layer,
        uint32_t expert) {
    return gemma4_stream_find(s, layer, expert);
}

static bool gemma4_stream_has_free(const ds4_gemma4_stream *s) {
    return s && s->free_slots > 0u;
}

static int32_t gemma4_stream_any_resident(const ds4_gemma4_stream *s, uint32_t layer) {
    if (!s || !s->slot_map || layer >= G4_LAYERS) { return -1; }
    for (uint32_t e = 0; e < G4_EXPERTS; e++) {
        if (s->slot_map[(uint64_t)layer * G4_EXPERTS + e] >= 0) {
            return (int32_t)e;
        }
    }
    return -1;
}

static void gemma4_stream_report(const ds4_gemma4_stream *s) {
    if (!s || (!s->stats.hits && !s->stats.misses)) { return; }
    fprintf(stderr,
        "ds4: Gemma-4 SSD cache: hits=%llu misses=%llu unique=%llu rows=%llu "
        "read=%.2f GiB reread=%.2f GiB io=%.1f ms scan=%.1f ms "
        "enqueue=%.1f ms wait=%.1f ms copy=%.1f ms submit=%.1f ms\n",
        (unsigned long long)s->stats.hits,
        (unsigned long long)s->stats.misses,
        (unsigned long long)s->stats.route_unique,
        (unsigned long long)s->stats.routed_rows,
        (double)s->stats.read_bytes / 1073741824.0,
        (double)s->stats.reread_bytes / 1073741824.0,
        (double)s->stats.io_ns / 1.0e6,
        (double)s->stats.scan_ns / 1.0e6,
        (double)s->stats.upload_ns / 1.0e6,
        (double)s->stats.wait_ns / 1.0e6,
        (double)s->stats.stage_copy_ns / 1.0e6,
        (double)s->stats.submit_ns / 1.0e6);
}

static void gemma4_stream_free(ds4_gemma4_stream *s) {
    /* The discrete map's cold-header device ranges; a no-op unless
     * --ssd-streaming installed the Gemma discrete plan. */
    ds4_gpu_g4_stream_plan_release();
    if (!s) { return; }
    /* Copies still on the upload stream land in the tensors freed below. */
    if (s->upload) { (void)ds4_gpu_upload_sync(s->upload); }
    gemma4_stream_report(s);
    ds4_gpu_tensor_free(s->gate);
    ds4_gpu_tensor_free(s->up);
    ds4_gpu_tensor_free(s->down);
    ds4_gpu_tensor_free(s->gate_res);
    ds4_gpu_tensor_free(s->up_res);
    ds4_gpu_tensor_free(s->down_res);
    free(s->slots);
    free(s->slot_map);
    free(s->seen);
    if (s->staging_pinned) {
        ds4_gpu_host_pinned_free(s->staging);
    } else {
        free(s->staging);
    }
    ds4_gpu_upload_free(s->upload);
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
    /* Fill from the model mapping, which is the page cache: once the pages
     * are touched (gemma4_stream_warm) the copy runs at 8.26 GiB/s, the rate
     * tests/bench_gemma4_staging measures for the same records, and the
     * device side needs 43 ms to submit and 1.5 ms to drain 6.5 GiB. Without
     * the warm the same copy measured 1.36 GiB/s (pread 1.88), because the
     * first touch of every page faults (2026-11-05).
     * DS4_GEMMA4_MOE_STAGING_PREAD=1 keeps pread as the alternative source. */
    s->host_source = true;
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
    /* One slice holds the largest record; the ring is the staging knob's
     * size (256 MiB by default), which at 0.39 MiB a record is 655 slices
     * and therefore five layers' worth of in-flight fills. */
    s->staging_slot = gemma4_stream_round(s->staging_bytes, G4_STREAM_ALIGN);
    uint64_t ring = gemma4_moe_staging_bytes();
    /* Hold a whole layer's fill, whatever the knob says: a shorter ring
     * forces a mid-layer drain, and that drain waits for the compute the
     * fill was supposed to overlap. */
    const uint64_t layer_ring = (uint64_t)G4_STREAM_LAYER_SLICES * s->staging_slot;
    if (ring < layer_ring) { ring = layer_ring; }
    s->staging_bytes = ring;
    s->staging = ds4_gpu_host_pinned_alloc(ring);
    s->staging_pinned = s->staging != NULL;
    if (!s->staging) {
        /* No pinned host memory (a driver refusal, or a platform without
         * it): the ring still works, the copies are just not overlapped. */
        s->staging = xmalloc(ring);
    }
    s->staging_slots = s->staging ? (uint32_t)(ring / s->staging_slot) : 0u;
    /* The upload stream and its ordering events. Without them the fill
     * stays on the blocking ds4_gpu_tensor_write path. */
    s->upload = ds4_gpu_upload_new();

    /* Diagnostic (DS4_GEMMA4_MOE_STAGING_TRACE=1): how fast this process can
     * read its own model mapping. The fill copies every expert record out of
     * it, so the rate here bounds the whole upload path; measured against the
     * standalone tests/bench_gemma4_staging, which reads the same file from a
     * small process. */
    if (s->model->map && s->model->size > (128u << 20) &&
        getenv("DS4_GEMMA4_MOE_STAGING_TRACE")) {
        const uint64_t probe = 64u << 20;
        const uint8_t *from = (const uint8_t *)s->model->map +
            (s->model->size / 2u);
        void *buf = xmalloc(probe);
        for (int round = 0; round < 2; round++) {
            const uint64_t t0 = gemma4_stream_ns();
            memcpy(buf, from, probe);
            /* Consume the copy so the compiler cannot elide it; the sum also
             * forces the bytes to be materialized. */
            uint64_t sink = 0;
            for (uint64_t i = 0; i < probe; i += 4096u) {
                sink += ((const uint8_t *)buf)[i];
            }
            const double sec = (double)(gemma4_stream_ns() - t0) / 1.0e9;
            fprintf(stderr, "ds4: Gemma-4 map read probe %d: %.2f GiB/s "
                    "(64 MiB, sum %llu)%s\n", round,
                    (double)probe / sec / 1073741824.0,
                    (unsigned long long)sink, round ? "" : " [cold]");
        }
        free(buf);

        /* The same read in the fill's own pattern: every record of layer 0
         * (128 experts x six records), scattered over the artifact the way
         * the router's misses are. */
        uint64_t pattern_bytes = 0;
        for (int pass = 0; pass < 2; pass++) {
        const uint64_t p0 = gemma4_stream_ns();
        for (uint32_t e = 0; e < G4_EXPERTS; e++) {
            const ds4_tensor *const ts[] = {
                w->gemma4[0].gate[e], w->gemma4[0].up[e], w->gemma4[0].down[e],
                w->gemma4[0].gate_res[e], w->gemma4[0].up_res[e],
                w->gemma4[0].down_res[e],
            };
            for (uint32_t k = 0; k < sizeof(ts) / sizeof(ts[0]); k++) {
                const ds4_tensor *t = ts[k];
                if (!t || t->bytes > s->staging_slot) { continue; }
                uint8_t *slice = gemma4_stream_ring_take(s, t->bytes);
                if (!slice) { continue; }
                memcpy(slice, (const uint8_t *)s->model->map + t->abs_offset,
                       t->bytes);
                pattern_bytes += t->bytes;
            }
        }
        {
            const double sec = (double)(gemma4_stream_ns() - p0) / 1.0e9;
            fprintf(stderr, "ds4: Gemma-4 map read probe pattern %d: %.2f GiB/s "
                    "(%.2f MiB, layer 0 records)%s\n", pass,
                    (double)pattern_bytes / sec / 1073741824.0,
                    (double)pattern_bytes / 1048576.0,
                    pass ? "" : " [cold]");
        }
        s->staging_used = 0u;
        }
    }
    s->slots = xcalloc(s->count, sizeof(*s->slots));
    s->slot_map = xmalloc((size_t)G4_LAYERS * G4_EXPERTS * sizeof(*s->slot_map));
    for (uint32_t i = 0; i < (uint32_t)G4_LAYERS * G4_EXPERTS; i++) {
        s->slot_map[i] = -1;
    }
    s->free_slots = s->count;
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
        "ds4: Gemma-4 SSD expert cache: %u/%llu slots, %.2f GiB, "
        "%.2f MiB staging ring (%u x %.2f MiB)%s%s\n",
        s->count, (unsigned long long)all, (double)s->bytes / 1073741824.0,
        (double)s->staging_bytes / 1048576.0, s->staging_slots,
        (double)s->staging_slot / 1048576.0,
        s->residual ? ", R13+8" : "",
        s->staging_pinned ? (s->upload ? ", pinned/queued" : ", pinned") : "");
    return true;
}

/* First touch of every expert record, once per process.
 *
 * The fill reads these records from the model mapping (or by pread), and the
 * first read of a page costs a fault: measured 1.38 to 1.9 GiB/s on the first
 * pass against 10 to 19.8 GiB/s on the second, over the same layer-0 records
 * inside one process (2026-11-05). That difference is the whole upload cost -
 * the device side takes 56 ms to submit and 1.5 ms to drain 6.5 GiB - so the
 * mapping is touched here, at open, and every request then reads warm pages.
 *
 * Touching does not pin anything: the pages stay file-backed page cache and
 * the kernel may still reclaim them, so this is not a pin and needs no
 * extra RAM beyond the page cache that the fill would pull in anyway.
 * DS4_GEMMA4_MOE_WARM=0 skips the warm. */
static void gemma4_stream_warm(ds4_gemma4_stream *s, const ds4_weights *w) {
    if (!s || !s->model || !s->model->map || !w || !w->gemma4) { return; }
    const char *off_env = getenv("DS4_GEMMA4_MOE_WARM");
    if (off_env && off_env[0] == '0') { return; }
    const uint64_t start = gemma4_stream_ns();
    uint64_t bytes = 0;
    volatile uint8_t sink = 0;
    const long page_l = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_l > 0 ? (uint64_t)page_l : 4096u;
    for (uint32_t il = 0; il < G4_LAYERS; il++) {
        const ds4_gemma4_layer *l = &w->gemma4[il];
        for (uint32_t e = 0; e < G4_EXPERTS; e++) {
            const ds4_tensor *const ts[] = {
                l->gate[e], l->up[e], l->down[e],
                l->gate_res[e], l->up_res[e], l->down_res[e],
            };
            for (uint32_t k = 0; k < sizeof(ts) / sizeof(ts[0]); k++) {
                const ds4_tensor *t = ts[k];
                if (!t || t->abs_offset + t->bytes > s->model->size) { continue; }
                const uint8_t *p = (const uint8_t *)s->model->map + t->abs_offset;
                for (uint64_t o = 0; o < t->bytes; o += page) { sink ^= p[o]; }
                bytes += t->bytes;
            }
        }
    }
    const double sec = (double)(gemma4_stream_ns() - start) / 1.0e9;
    fprintf(stderr, "ds4: Gemma-4 expert warm: %.2f GiB touched in %.2fs "
            "(%.2f GiB/s, checksum %u)\n", (double)bytes / 1073741824.0, sec,
            sec > 0 ? (double)bytes / sec / 1073741824.0 : 0.0,
            (unsigned)sink);
}

static bool gemma4_stream_attach(ds4_gemma4_stream *s, ds4_gvq_model *plan) {    if (!s || !plan) { return false; }
    s->plan = plan;
    return true;
}

/* The resident set: every tensor except the routed expert matrices and their
 * residual planes. Books, norms, attention, dense FFN, the token table and
 * the router stay directly addressable; the expert records arrive through
 * the slot pool. `cold_layers` bounds which layers pay that: the first N
 * layers' experts stay host-cold (the CPU MoE lane, DS4_GEMMA4_MOE_CPU_LAYERS
 * and the SSD stream's -ncmoe shape), and every layer at or above N keeps its
 * expert records in this resident plan, so the GPU path reads them exactly
 * as it does with no host or stream arm at all. G4_LAYERS is today's
 * all-cold behaviour; 0 is the all-resident default. */
static bool gemma4_stream_spans(const ds4_weights *w, const ds4_model *m,
        uint32_t cold_layers, ds4_model_map_span_vec *spans) {
    if (!w || !w->gemma4 || !m || !m->tensors || !m->n_tensors || !spans) {
        return false;
    }
    memset(spans, 0, sizeof(*spans));
    uint8_t *streamed = xcalloc(m->n_tensors, sizeof(*streamed));
    if (cold_layers > (uint32_t)G4_LAYERS) { cold_layers = (uint32_t)G4_LAYERS; }
    for (uint32_t il = 0; il < cold_layers; il++) {
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

/* The cold record set: the routed expert VQ records, and on an R13+8
 * artifact their residual planes, of the first `cold_layers` layers -- the
 * records gemma4_stream_spans leaves out of the resident plan. `indices`
 * holds 6 * G4_LAYERS * G4_EXPERTS entries; the returned count is ascending
 * and duplicate-free. The discrete
 * map (ds4_gemma4_map.inc) publishes one GVQ_HEADER-sized device range per
 * entry so the bind-time header gather never reads the host map. */
static int gemma4_stream_index_cmp(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static uint32_t gemma4_stream_cold_records(const ds4_weights *w,
        const ds4_model *m, uint32_t cold_layers, uint32_t *indices) {
    if (!w || !w->gemma4 || !m || !indices) { return 0u; }
    uint32_t n = 0u;
    if (cold_layers > (uint32_t)G4_LAYERS) { cold_layers = (uint32_t)G4_LAYERS; }
    for (uint32_t il = 0; il < cold_layers; il++) {
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
        ds4_gpu_tensor *dst, uint64_t offset, uint8_t *header, bool hazard) {
    if (!s || !t || !dst || !s->model || s->model->fd < 0 ||
        t->abs_offset > s->model->size || t->bytes > s->model->size - t->abs_offset ||
        t->bytes > s->staging_slot || t->bytes < G4_STREAM_HEADER) {
        return false;
    }
    /* Diagnostic (DS4_GEMMA4_MOE_STAGING_PREAD=1): fill through pread from
     * the shard fds instead of the mapping. Reading the mapping costs a page
     * fault per 4 KiB page (measured 1.4 GiB/s, cold and warm alike), while
     * pread copies out of the page cache without touching this process's page
     * tables. */
    static int force_pread = -1;
    if (force_pread < 0) {
        const char *v = getenv("DS4_GEMMA4_MOE_STAGING_PREAD");
        force_pread = (v && v[0] != '0') ? 1 : 0;
    }
    if (!force_pread && s->host_source && s->model->map) {
        const uint8_t *src = (const uint8_t *)s->model->map + t->abs_offset;
        if (header) { memcpy(header, src, G4_STREAM_HEADER); }
        const uint64_t upload_start = gemma4_stream_ns();
        if (s->queued) {
            /* A slot that something may still be reading must not be
             * overwritten until that work retires; a never-used slot has no
             * such reader, so its fill needs no ordering at all and can run
             * under the previous layer's compute. */
            if (hazard && !s->ordered) {
                if (ds4_gpu_upload_begin(s->upload)) {
                    s->ordered = true;
                } else {
                    s->queued = false;
                }
            }
        }
        if (s->queued) {
            /* Copy into the pinned ring and hand the transfer to the upload
             * stream. The host is free to stage the next record while this
             * one is in flight; gemma4_stream_stage drains the batch before
             * the kernel that reads the slots. */
            uint8_t *slice = gemma4_stream_ring_take(s, t->bytes);
            if (!slice) { return false; }
            const uint64_t copy_start = gemma4_stream_ns();
            memcpy(slice, src, t->bytes);
            const uint64_t submit_start = gemma4_stream_ns();
            const bool queued_ok =
                ds4_gpu_upload_enqueue(s->upload, dst, offset, slice, t->bytes);
            s->stats.stage_copy_ns += submit_start - copy_start;
            s->stats.submit_ns += gemma4_stream_ns() - submit_start;
            if (!queued_ok) { return false; }
        } else if (!ds4_gpu_tensor_write(dst, offset, src, t->bytes)) {
            return false;
        }
        s->stats.upload_ns += gemma4_stream_ns() - upload_start;
        return true;
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
    if (s->queued && hazard && !s->ordered) {
        if (ds4_gpu_upload_begin(s->upload)) {
            s->ordered = true;
        } else {
            s->queued = false;
        }
    }
    uint8_t *stage = gemma4_stream_ring_take(s, t->bytes);
    if (!stage) { return false; }
    uint64_t done = 0;
    const uint64_t read_start = gemma4_stream_ns();
    while (done < t->bytes) {
        const ssize_t got = pread(s->shard_fds[shard], stage + done,
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
    if (header) { memcpy(header, stage, G4_STREAM_HEADER); }
    const uint64_t upload_start = gemma4_stream_ns();
    if (s->queued) {
        if (!ds4_gpu_upload_enqueue(s->upload, dst, offset, stage, t->bytes)) {
            return false;
        }
    } else if (!ds4_gpu_tensor_write(dst, offset, stage, t->bytes)) {
        return false;
    }
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
        uint32_t il, uint32_t e, uint32_t slot, bool hazard) {
    if (!s || !s->plan || !l || e >= G4_EXPERTS || slot >= s->count) { return false; }
    const uint64_t gate_off = (uint64_t)slot * s->gate_stride;
    const uint64_t up_off = (uint64_t)slot * s->up_stride;
    const uint64_t down_off = (uint64_t)slot * s->down_stride;
    const uint64_t res_off = (uint64_t)slot * s->res_stride;
    uint8_t mh[G4_STREAM_HEADER], rh[G4_STREAM_HEADER];
    const bool reread = s->seen[(uint64_t)il * G4_EXPERTS + e] != 0u;
    uint64_t bytes = 0;

    if (!gemma4_stream_read(s, l->gate[e], s->gate, gate_off, mh, hazard)) { return false; }
    bytes += l->gate[e]->bytes;
    if (s->residual && !gemma4_stream_read(s, l->gate_res[e], s->gate_res, res_off, rh, hazard)) {
        return false;
    }
    if (s->residual) { bytes += l->gate_res[e]->bytes; }
    if (!ds4_gpu_g4_stream_rebind(s->plan, gemma4_stream_index(s->model, l->gate[e]),
            s->gate, gate_off, l->gate[e]->bytes, mh,
            s->residual ? gemma4_stream_index(s->model, l->gate_res[e]) : DS4_GVQ_NO_LINK,
            s->residual ? s->gate_res : NULL, res_off,
            s->residual ? l->gate_res[e]->bytes : 0u,
            s->residual ? rh : NULL)) { return false; }

    if (!gemma4_stream_read(s, l->up[e], s->up, up_off, mh, hazard)) { return false; }
    bytes += l->up[e]->bytes;
    if (s->residual && !gemma4_stream_read(s, l->up_res[e], s->up_res, res_off, rh, hazard)) {
        return false;
    }
    if (s->residual) { bytes += l->up_res[e]->bytes; }
    if (!ds4_gpu_g4_stream_rebind(s->plan, gemma4_stream_index(s->model, l->up[e]),
            s->up, up_off, l->up[e]->bytes, mh,
            s->residual ? gemma4_stream_index(s->model, l->up_res[e]) : DS4_GVQ_NO_LINK,
            s->residual ? s->up_res : NULL, res_off,
            s->residual ? l->up_res[e]->bytes : 0u,
            s->residual ? rh : NULL)) { return false; }

    if (!gemma4_stream_read(s, l->down[e], s->down, down_off, mh, hazard)) { return false; }
    bytes += l->down[e]->bytes;
    if (s->residual && !gemma4_stream_read(s, l->down_res[e], s->down_res, res_off, rh, hazard)) {
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
    /* The two admission walks below resolve each route id through the
     * per-layer slot map; their time is charged to scan_ns. */
    const uint64_t scan_start = gemma4_stream_ns();
    const uint64_t upload_before = s->stats.upload_ns;
    const uint64_t io_before = s->stats.io_ns;
    for (uint64_t i = 0; i < (uint64_t)chunk * G4_USED; i++) {
        const int slot = gemma4_stream_find(s, il, (uint32_t)s->ids[i]);
        if (slot >= 0) { s->slots[slot].pinned = s->epoch; }
    }
    /* Open the batch's queued window: the upload stream waits for the
     * compute already running, so no slice it is about to overwrite can still
     * be read by a kernel. Inside a CUDA graph capture the event pair is
     * illegal and begin() refuses, which leaves the blocking copies below. */
    s->queued = gemma4_stream_queued_on() && s->upload != NULL &&
                !ds4_gpu_capture_active();
    s->ordered = false;
    for (uint64_t i = 0; i < (uint64_t)chunk * G4_USED; i++) {
        const uint32_t expert = (uint32_t)s->ids[i];
        int slot = gemma4_stream_find(s, il, expert);
        if (slot >= 0) {
            s->stats.hits++;
        } else {
            slot = gemma4_stream_victim(s->slots, s->count, s->epoch);
            if (slot < 0) {
                fprintf(stderr, "ds4: Gemma-4 SSD routing batch exceeds %u cache slots\n",
                        s->count);
                s->queued = false;
                return 0u;
            }
            /* A failed partial upload must not remain a cache hit. */
            const bool hazard = s->slots[slot].used != 0u;
            if (hazard) {
                gemma4_stream_map_drop(s, (uint32_t)slot);
            } else if (s->free_slots) {
                s->free_slots--;
            }
            s->slots[slot].used = 0u;
            if (!gemma4_stream_load(s, l, il, expert, (uint32_t)slot, hazard)) {
                s->queued = false;
                return 0u;
            }
            gemma4_stream_map_set(s, (uint32_t)slot, il, expert);
            s->stats.misses++;
        }
        s->slots[slot].used = s->epoch;
        s->slots[slot].pinned = s->epoch;
    }
    /* Scan time is the walk minus the work it hosted: the pread reads
     * (io_ns) and the copies (enqueue time), both already accounted. */
    {
        const uint64_t walked = gemma4_stream_ns() - scan_start;
        const uint64_t hosted = (s->stats.upload_ns - upload_before) +
                                (s->stats.io_ns - io_before);
        s->stats.scan_ns += walked > hosted ? walked - hosted : 0u;
    }
    /* One drain per staged batch: the caller launches the MoE kernel over
     * these slots immediately after this returns. */
    s->queued = false;
    if (!gemma4_stream_ring_flush(s)) { return 0u; }
    return chunk;
}

#endif
