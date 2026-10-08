/* Actual SSD reads with a model-free tensor backend; no CUDA allocation. */
#define DS4_NO_GPU
#include "../ds4.c"
#include "../ds4_gpu.h"
#include <sched.h>

struct ds4_gpu_tensor {
    unsigned char *data;
    uint64_t bytes;
    int memc;
    const void *source;
};
struct ds4_gpu_upload { uint64_t bytes; };
static uint64_t gpu_bytes, pread_calls, pread_bytes, advise_calls, sync_calls;
static uint64_t upload_calls, upload_bytes;
static uint64_t copy_calls, copy_bytes;
static uint32_t upload_live;
static ds4_mem_cell gpu_census[DS4_MEMC__COUNT], source_census[DS4_MEMC__COUNT];
static uint64_t census_faults;
static const void *bound_map;
static int upload_fail, alloc_fail, upload_new_fail;
static int copy_fail;
static int pread_intr, pread_fail, sync_fail;
static size_t pread_limit;
enum read_gate { READ_IDLE, READ_ARMED, READ_BLOCKED, READ_RELEASED };
static enum read_gate read_gate;
static off_t read_offset;
static pthread_mutex_t read_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t read_ready = PTHREAD_COND_INITIALIZER;
static const ds4_model *close_model_watch;
static const ds4_layer_weights *close_weights_watch;
static bool close_weights_cleared, close_read_seen, close_read_valid;
enum upload_fault { UPLOAD_CLEAN, UPLOAD_PARTIAL };
static enum upload_fault upload_fault;
static ds4_gpu_tensor *busy_tensor;
static unsigned char busy_byte;
enum { BUSY_TENSORS = 3 };
typedef struct {
    ds4_gpu_tensor *tensor;
    unsigned char *copy;
    uint64_t off, bytes;
} busy_span;
static busy_span busy_guard[BUSY_TENSORS];
static pthread_mutex_t gpu_mutex = PTHREAD_MUTEX_INITIALIZER;

static int busy_intact_locked(void) {
    if (busy_tensor && busy_tensor->data[0] != busy_byte) { return 0; }
    for (unsigned i = 0; i < BUSY_TENSORS; i++) {
        const busy_span *g = &busy_guard[i];
        if (g->tensor && memcmp(g->tensor->data + g->off, g->copy, g->bytes)) {
            return 0;
        }
    }
    return 1;
}

static int busy_intact(void) {
    pthread_mutex_lock(&gpu_mutex);
    const int intact = busy_intact_locked();
    pthread_mutex_unlock(&gpu_mutex);
    return intact;
}

static void busy_release(void) {
    pthread_mutex_lock(&gpu_mutex);
    for (unsigned i = 0; i < BUSY_TENSORS; i++) {
        free(busy_guard[i].copy);
        busy_guard[i] = (busy_span){0};
    }
    busy_tensor = NULL;
    pthread_mutex_unlock(&gpu_mutex);
}

static int busy_conflicts(ds4_gpu_tensor *t, uint64_t off, uint64_t bytes) {
    for (unsigned i = 0; i < BUSY_TENSORS; i++) {
        const busy_span *g = &busy_guard[i];
        if (g->tensor == t && off < g->off + g->bytes && g->off < off + bytes) {
            return 1;
        }
    }
    return 0;
}

int ds4_gpu_stream_synchronize(void) {
    sync_calls++;
    if (sync_fail || !busy_intact()) { return 0; }
    busy_release();
    return 1;
}

static ds4_gpu_tensor *fixture_alloc(uint64_t bytes, int memc,
                                     const void *source) {
    if (alloc_fail && --alloc_fail == 0) { return NULL; }
    ds4_gpu_tensor *t = calloc(1u, sizeof(*t));
    if (!t) { return NULL; }
    t->data = calloc(1u, bytes);
    if (!t->data) { free(t); return NULL; }
    t->bytes = bytes;
    t->memc = memc;
    t->source = source;
    gpu_bytes += bytes;
    ds4_mem_cell_note_alloc(&gpu_census[memc], bytes, bytes, &census_faults);
    if (source) {
        ds4_mem_cell_note_alloc(&source_census[memc], bytes, bytes, &census_faults);
    }
    return t;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    return fixture_alloc(bytes, DS4_MEMC_ENGINE_OTHER, NULL);
}

ds4_gpu_tensor *ds4_gpu_weight_alloc(const void *model_map, uint64_t bytes) {
    if (!model_map || model_map != bound_map || !bytes) { return NULL; }
    return fixture_alloc(bytes, DS4_MEMC_WEIGHT_SPAN, model_map);
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) { return; }
    gpu_bytes -= t->bytes;
    ds4_mem_cell_note_free(&gpu_census[t->memc], t->bytes, t->bytes, &census_faults);
    if (t->source) {
        ds4_mem_cell_note_free(&source_census[t->memc], t->bytes, t->bytes,
                              &census_faults);
    }
    free(t->data);
    free(t);
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off,
                         const void *data, uint64_t bytes) {
    if (!t || off > t->bytes || bytes > t->bytes - off) { return 0; }
    pthread_mutex_lock(&gpu_mutex);
    if (busy_tensor || busy_conflicts(t, off, bytes)) {
        pthread_mutex_unlock(&gpu_mutex);
        return 0;
    }
    if (upload_fail && --upload_fail == 0) {
        if (upload_fault == UPLOAD_PARTIAL) { memcpy(t->data + off, data, bytes / 2u); }
        pthread_mutex_unlock(&gpu_mutex);
        return 0;
    }
    memcpy(t->data + off, data, bytes);
    pthread_mutex_unlock(&gpu_mutex);
    return 1;
}

struct ds4_gpu_upload *ds4_gpu_upload_new(void) {
    if (upload_new_fail && --upload_new_fail == 0) { return NULL; }
    struct ds4_gpu_upload *driver = calloc(1u, sizeof(*driver));
    if (driver) { upload_live++; }
    return driver;
}

int ds4_gpu_upload_write(struct ds4_gpu_upload *driver, ds4_gpu_tensor *dst,
        uint64_t off, const void *src, uint64_t bytes) {
    if (!driver) { return 0; }
    upload_calls++;
    if (!ds4_gpu_tensor_write(dst, off, src, bytes)) { return 0; }
    /* The opaque driver returns after its upload stream completes. Real CUDA
     * buffer reuse is a separate driver gate; this mock checks slot overlap. */
    driver->bytes += bytes;
    upload_bytes += bytes;
    return 1;
}

void ds4_gpu_upload_free(struct ds4_gpu_upload *driver) {
    if (!driver) { return; }
    upload_live--;
    free(driver);
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_off,
        const ds4_gpu_tensor *src, uint64_t src_off, uint64_t bytes) {
    if (!dst || !src || dst_off > dst->bytes || bytes > dst->bytes - dst_off ||
        src_off > src->bytes || bytes > src->bytes - src_off) { return 0; }
    /* D2D copies follow compute on the inference stream. */
    if (!ds4_gpu_stream_synchronize()) { return 0; }
    pthread_mutex_lock(&gpu_mutex);
    copy_calls++;
    if (copy_fail && --copy_fail == 0) {
        memmove(dst->data + dst_off, src->data + src_off, bytes / 2u);
        pthread_mutex_unlock(&gpu_mutex);
        return 0;
    }
    memmove(dst->data + dst_off, src->data + src_off, bytes);
    copy_bytes += bytes;
    pthread_mutex_unlock(&gpu_mutex);
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off,
                        void *data, uint64_t bytes) {
    if (!t || off > t->bytes || bytes > t->bytes - off) { return 0; }
    /* The blocking route read drains the previous submission before eviction. */
    if (!busy_intact()) { return 0; }
    busy_release();
    memcpy(data, t->data + off, bytes);
    return 1;
}

static ssize_t fixture_pread(int fd, void *dst, size_t bytes, off_t off) {
    pread_calls++;
    pthread_mutex_lock(&read_mutex);
    if (read_gate == READ_ARMED && off == read_offset) {
        read_gate = READ_BLOCKED;
        pthread_cond_broadcast(&read_ready);
        while (read_gate == READ_BLOCKED) {
            pthread_cond_wait(&read_ready, &read_mutex);
        }
        if (close_weights_watch) {
            close_read_valid = close_weights_watch->ffn_gate_exps &&
                close_weights_watch->ffn_up_exps && close_weights_watch->ffn_down_exps &&
                close_model_watch->map && close_model_watch->size && close_model_watch->fd >= 0;
            close_read_seen = true;
            pthread_cond_broadcast(&read_ready);
        }
        read_gate = READ_IDLE;
    }
    pthread_mutex_unlock(&read_mutex);
    if (pread_intr) { pread_intr--; errno = EINTR; return -1; }
    if (pread_fail && --pread_fail == 0) { errno = EIO; return -1; }
    if (pread_limit && bytes > pread_limit) { bytes = pread_limit; }
    const ssize_t got = pread(fd, dst, bytes, off);
    if (got > 0) { pread_bytes += (uint64_t)got; }
    return got;
}

static int fixture_advise(int fd, off_t off, off_t bytes, int policy) {
    (void)fd;
    if (off < 0 || bytes <= 0 || policy != POSIX_FADV_DONTNEED) { return EINVAL; }
    advise_calls++;
    return 0;
}

#define pread fixture_pread
#define posix_fadvise fixture_advise
#include "../ds4_glm53_stream.inc"
#undef pread
#undef posix_fadvise

enum { LAYERS = 6, FIRST = 3, EXPERTS = 12, USED = 3, WIDTH = 256 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM stream FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

typedef struct {
    ds4_model model;
    ds4_weights weights;
    ds4_tensor routed[LAYERS][3], resident[3];
    unsigned char *bytes;
} fixture;

static void fixture_init(fixture *f) {
    memset(f, 0, sizeof(*f));
    g_ds4_shape = DS4_SHAPE_GLM53_FLASH;
    g_ds4_shape.n_layer = LAYERS;
    g_ds4_shape.n_leading_dense = FIRST;
    g_ds4_shape.n_expert = EXPERTS;
    g_ds4_shape.n_expert_used = USED;
    g_ds4_shape.n_embd = WIDTH;
    g_ds4_shape.n_ff_exp = WIDTH;
    g_ds4_shape.n_nextn_predict = 0u;
    uint64_t cursor = 4096u;
    for (unsigned il = FIRST; il < LAYERS; il++) {
        const uint32_t types[] = {il == 4u ? 16u : 17u,
                                  il == 4u ? 16u : 17u,
                                  il == 4u ? 17u : 10u};
        for (unsigned j = 0u; j < 3u; j++) {
            ds4_tensor *t = &f->routed[il][j];
            *t = (ds4_tensor){.type = types[j], .ndim = 3u,
                .dim = {WIDTH, WIDTH, EXPERTS}, .abs_offset = cursor};
            CHECK(tensor_nbytes(t->type, WIDTH * WIDTH * EXPERTS, &t->bytes));
            cursor += t->bytes;
        }
        f->weights.layer[il].ffn_gate_exps = &f->routed[il][0];
        f->weights.layer[il].ffn_up_exps = &f->routed[il][1];
        f->weights.layer[il].ffn_down_exps = &f->routed[il][2];
    }
    for (unsigned i = 0u; i < 3u; i++) {
        f->resident[i] = (ds4_tensor){.abs_offset = i * 1024u, .bytes = 1024u};
    }
    f->weights.token_embd = &f->resident[0];
    f->weights.layer[0].attn_norm = &f->resident[1];
    f->weights.output = &f->resident[2];
    f->bytes = calloc(1u, cursor);
    CHECK(f->bytes);
    for (unsigned il = FIRST; il < LAYERS; il++) {
        for (unsigned j = 0u; j < 3u; j++) {
            ds4_tensor *t = &f->routed[il][j];
            const uint64_t unit = t->bytes / EXPERTS;
            for (unsigned e = 0u; e < EXPERTS; e++) {
                /* A constant expert payload would hide a wrong partial-read
                 * offset. Preserve the expert seed while varying each byte. */
                for (uint64_t i = 0u; i < unit; i++) {
                    f->bytes[t->abs_offset + e * unit + i] =
                        (unsigned char)(il * 32u + j * 12u + e + i * 13u + (i >> 8u));
                }
            }
        }
    }
    char path[] = "/tmp/ds4-glm53-stream-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0 && unlink(path) == 0);
    CHECK(write(fd, f->bytes, cursor) == (ssize_t)cursor);
    f->model.fd = fd;
    f->model.size = cursor;
    f->model.map = f->bytes;
    f->model.split_count = 1u;
}

static void check_slots(const ds4_glm53_stream *s, const fixture *f,
                        unsigned il, unsigned rows) {
    const int32_t *ids = (const int32_t *)s->selected->data;
    ds4_gpu_tensor *all[] = {s->gate, s->up, s->down};
    for (unsigned i = 0u; i < rows * USED; i++) {
        CHECK(ids[i] >= 0 && (uint32_t)ids[i] < s->count);
        const ds4_glm53_cache_slot *slot = &s->slots[ids[i]];
        CHECK(slot->used && slot->layer == il &&
            (slot->pinned == s->epoch || (s->hot_active && slot->pinned == GLM53_CACHE_HOT_PIN)));
        for (unsigned j = 0u; j < 3u; j++) {
            const ds4_tensor *t = &f->routed[il][j];
            const uint64_t unit = t->bytes / EXPERTS;
            const uint64_t stride = j == 2u ? s->down_stride : s->gate_stride;
            CHECK(memcmp(all[j]->data + ids[i] * stride,
                f->bytes + t->abs_offset + slot->expert * unit, unit) == 0);
        }
    }
}

static void check_accounting(fixture *f) {
    ds4_glm53_stream s = {0};
    const ds4_engine_options opt = {.ssd_streaming = true,
                                    .ssd_streaming_cache_experts = USED};
    CHECK(glm53_stream_alloc(&s, &f->model, &f->weights, &opt));
    CHECK(ds4_mem_cell_live(&gpu_census[DS4_MEMC_WEIGHT_SPAN]) == s.bytes);
    CHECK(ds4_mem_cell_live(&source_census[DS4_MEMC_WEIGHT_SPAN]) == s.bytes);
    CHECK(!ds4_mem_cell_live(&gpu_census[DS4_MEMC_ENGINE_OTHER]));
    CHECK(gpu_census[DS4_MEMC_WEIGHT_SPAN].alloc_calls == 3u);

    ds4_gpu_tensor *scratch = ds4_gpu_tensor_alloc(sizeof(int32_t));
    CHECK(scratch);
    CHECK(ds4_mem_cell_live(&gpu_census[DS4_MEMC_ENGINE_OTHER]) == sizeof(int32_t));
    ds4_gpu_tensor_free(scratch);
    /* Free must use allocation-time source identity, even after map teardown. */
    bound_map = NULL;
    glm53_stream_free(&s);
    CHECK(!gpu_bytes && !census_faults);
    CHECK(!ds4_mem_cell_live(&gpu_census[DS4_MEMC_WEIGHT_SPAN]));
    CHECK(!ds4_mem_cell_live(&source_census[DS4_MEMC_WEIGHT_SPAN]));

    bound_map = f->model.map;
    alloc_fail = 2;
    CHECK(!glm53_stream_alloc(&s, &f->model, &f->weights, &opt));
    CHECK(!gpu_bytes && !census_faults);
    CHECK(!ds4_mem_cell_live(&gpu_census[DS4_MEMC_WEIGHT_SPAN]));
    CHECK(!ds4_mem_cell_live(&source_census[DS4_MEMC_WEIGHT_SPAN]));
    CHECK(!ds4_mem_cell_live(&gpu_census[DS4_MEMC_ENGINE_OTHER]));
    puts("GLM stream: weight/source accounting and partial cleanup passed");
}

static bool fixture_select(ds4_glm53_stream *s, const ds4_model *m,
        const ds4_layer_weights *l, uint32_t il,
        const ds4_gpu_tensor *selected, uint32_t rows) {
    return glm53_stream_routes(s, selected, rows) &&
        glm53_stream_stage(s, m, l, il, s->ids, rows);
}

static void fixture_restore(fixture *f) {
    CHECK(ftruncate(f->model.fd, (off_t)f->model.size) == 0);
    CHECK(pwrite(f->model.fd, f->bytes, f->model.size, 0) == (ssize_t)f->model.size);
}

static void fault_reset(void) {
    upload_fail = alloc_fail = upload_new_fail = copy_fail = pread_intr = pread_fail = sync_fail = 0;
    pread_limit = 0u;
    upload_fault = UPLOAD_CLEAN;
    CHECK(busy_intact());
    busy_release();
    CHECK(read_gate == READ_IDLE);
}

static void read_arm(off_t off) {
    CHECK(pthread_mutex_lock(&read_mutex) == 0);
    CHECK(read_gate == READ_IDLE);
    read_offset = off;
    read_gate = READ_ARMED;
    CHECK(pthread_mutex_unlock(&read_mutex) == 0);
}

static void read_await(void) {
    struct timespec deadline;
    CHECK(clock_gettime(CLOCK_REALTIME, &deadline) == 0);
    deadline.tv_sec += 5;
    CHECK(pthread_mutex_lock(&read_mutex) == 0);
    while (read_gate == READ_ARMED) {
        CHECK(pthread_cond_timedwait(&read_ready, &read_mutex, &deadline) == 0);
    }
    CHECK(read_gate == READ_BLOCKED);
    CHECK(pthread_mutex_unlock(&read_mutex) == 0);
}

static void read_open(void) {
    CHECK(pthread_mutex_lock(&read_mutex) == 0);
    CHECK(read_gate == READ_BLOCKED);
    read_gate = READ_RELEASED;
    CHECK(pthread_cond_broadcast(&read_ready) == 0);
    CHECK(pthread_mutex_unlock(&read_mutex) == 0);
}

static void stream_new(ds4_glm53_stream *s, fixture *f, uint32_t count) {
    *s = (ds4_glm53_stream){0};
    const ds4_engine_options opt = {.ssd_streaming = true,
                                    .ssd_streaming_cache_experts = count};
    CHECK(glm53_stream_alloc(s, &f->model, &f->weights, &opt));
}

static void check_routes(const ds4_glm53_stream *s, const fixture *f,
        uint32_t il, uint32_t rows, const int32_t *want) {
    check_slots(s, f, il, rows);
    const int32_t *slots = (const int32_t *)s->selected->data;
    for (uint32_t i = 0u; i < rows * USED; i++) {
        CHECK(s->slots[slots[i]].expert == (uint32_t)want[i]);
    }
}

static uint64_t layer_bytes(const fixture *f, uint32_t il) {
    return f->routed[il][0].bytes + f->routed[il][1].bytes + f->routed[il][2].bytes;
}

static void check_payload(const ds4_glm53_stream *s, const fixture *f,
        uint32_t il, uint32_t slot, uint32_t expert) {
    CHECK(slot < s->count && s->slots[slot].used);
    CHECK(s->slots[slot].layer == il && s->slots[slot].expert == expert);
    ds4_gpu_tensor *all[] = {s->gate, s->up, s->down};
    for (uint32_t j = 0u; j < BUSY_TENSORS; j++) {
        const ds4_tensor *t = &f->routed[il][j];
        const uint64_t unit = t->bytes / EXPERTS;
        const uint64_t stride = j == 2u ? s->down_stride : s->gate_stride;
        CHECK(!memcmp(all[j]->data + slot * stride,
            f->bytes + t->abs_offset + expert * unit, unit));
    }
}

static uint32_t check_layer(const ds4_glm53_stream *s, const fixture *f,
        uint32_t il) {
    const int base = glm53_cache_find(s->slots, s->count, il, 0u);
    CHECK(base >= 0 && (uint32_t)base + EXPERTS <= s->count);
    for (uint32_t e = 0u; e < EXPERTS; e++) {
        CHECK(glm53_cache_find(s->slots, s->count, il, e) == base + (int)e);
        check_payload(s, f, il, (uint32_t)base + e, e);
    }
    return (uint32_t)base;
}

static void check_absent(const ds4_glm53_stream *s, uint32_t il) {
    for (uint32_t e = 0u; e < EXPERTS; e++) {
        CHECK(glm53_cache_find(s->slots, s->count, il, e) < 0);
    }
}

/* Preserve every weight byte the pending GPU submission can consume. The
 * same guard can later allow uploads to an independent staging group. */
static void fixture_busy(ds4_glm53_stream *s, uint32_t first, uint32_t count) {
    CHECK(first <= s->count && count <= s->count - first);
    ds4_gpu_tensor *tensors[] = {s->gate, s->up, s->down};
    CHECK(pthread_mutex_lock(&gpu_mutex) == 0);
    for (unsigned i = 0u; i < BUSY_TENSORS; i++) {
        CHECK(!busy_guard[i].tensor);
        const uint64_t stride = i == 2u ? s->down_stride : s->gate_stride;
        busy_guard[i] = (busy_span){.tensor = tensors[i],
            .off = first * stride, .bytes = count * stride};
        busy_guard[i].copy = malloc(busy_guard[i].bytes);
        CHECK(busy_guard[i].copy);
        memcpy(busy_guard[i].copy, tensors[i]->data + busy_guard[i].off,
               busy_guard[i].bytes);
    }
    CHECK(pthread_mutex_unlock(&gpu_mutex) == 0);
}

static void check_eintr(fixture *f) {
    fault_reset();
    ds4_glm53_stream s;
    stream_new(&s, f, USED);
    const int32_t ids[] = {0, 1, 2};
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(ids));
    CHECK(routes && ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    enum { INTERRUPTS = 3, READ_LIMIT = 257 };
    const uint64_t calls = pread_calls, bytes = pread_bytes;
    const uint64_t gate = f->routed[FIRST][0].bytes / EXPERTS;
    const uint64_t down = f->routed[FIRST][2].bytes / EXPERTS;
    pread_intr = INTERRUPTS;
    pread_limit = READ_LIMIT;
    CHECK(fixture_select(&s, &f->model, &f->weights.layer[FIRST], FIRST, routes, 1u));
    CHECK(!pread_intr && s.misses == USED);
    CHECK(pread_calls - calls == INTERRUPTS + USED *
        (2u * ((gate + READ_LIMIT - 1u) / READ_LIMIT) +
         (down + READ_LIMIT - 1u) / READ_LIMIT));
    CHECK(pread_bytes - bytes == USED * (2u * gate + down));
    CHECK(s.read_bytes == pread_bytes - bytes);
    check_routes(&s, f, FIRST, 1u, ids);
    fault_reset();
    glm53_stream_free(&s);
    ds4_gpu_tensor_free(routes);
    CHECK(!gpu_bytes);
    puts("GLM stream: EINTR retries and partial pread offsets passed");
}

enum read_fault { READ_ERROR, PARTIAL_UPLOAD, TRUNCATED_FILE };

static void check_retry(fixture *f, enum read_fault fault) {
    fault_reset();
    fixture_restore(f);
    ds4_glm53_stream s;
    stream_new(&s, f, USED);
    const int32_t warm[] = {0, 1, 2}, next[] = {3, 1, 2};
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(warm));
    CHECK(routes && ds4_gpu_tensor_write(routes, 0u, warm, sizeof(warm)));
    CHECK(fixture_select(&s, &f->model, &f->weights.layer[FIRST], FIRST, routes, 1u));
    CHECK(ds4_gpu_tensor_write(routes, 0u, next, sizeof(next)));
    if (fault == READ_ERROR) { pread_fail = 1; }
    if (fault == PARTIAL_UPLOAD) { upload_fail = 2; upload_fault = UPLOAD_PARTIAL; }
    if (fault == TRUNCATED_FILE) {
        const ds4_tensor *gate = f->weights.layer[FIRST].ffn_gate_exps;
        CHECK(ftruncate(f->model.fd,
            (off_t)(gate->abs_offset + 4u * (gate->bytes / EXPERTS) - 1u)) == 0);
    }
    CHECK(!fixture_select(&s, &f->model, &f->weights.layer[FIRST], FIRST, routes, 1u));
    CHECK(glm53_cache_find(s.slots, s.count, FIRST, 3u) < 0);
    CHECK(glm53_cache_find(s.slots, s.count, FIRST, 0u) < 0);
    CHECK(s.misses == USED);
    for (uint32_t expert = 1u; expert < USED; expert++) {
        const int slot = glm53_cache_find(s.slots, s.count, FIRST, expert);
        CHECK(slot >= 0);
        ds4_gpu_tensor *all[] = {s.gate, s.up, s.down};
        for (uint32_t j = 0u; j < BUSY_TENSORS; j++) {
            const ds4_tensor *t = &f->routed[FIRST][j];
            const uint64_t unit = t->bytes / EXPERTS;
            const uint64_t stride = j == 2u ? s.down_stride : s.gate_stride;
            CHECK(!memcmp(all[j]->data + slot * stride,
                f->bytes + t->abs_offset + expert * unit, unit));
        }
    }
    fault_reset();
    fixture_restore(f);
    const uint64_t calls = pread_calls, hits = s.hits;
    CHECK(fixture_select(&s, &f->model, &f->weights.layer[FIRST], FIRST, routes, 1u));
    CHECK(pread_calls - calls == BUSY_TENSORS && s.misses == USED + 1u);
    CHECK(s.hits - hits == USED - 1u);
    check_routes(&s, f, FIRST, 1u, next);
    glm53_stream_free(&s);
    ds4_gpu_tensor_free(routes);
    CHECK(!gpu_bytes);
    printf("GLM stream: fault=%u invalidation and full retry passed\n", (unsigned)fault);
}

static void check_fence(fixture *f) {
    fault_reset();
    ds4_glm53_stream s;
    stream_new(&s, f, USED);
    const int32_t warm[] = {0, 1, 2}, next[] = {3, 4, 5};
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(warm));
    CHECK(routes && ds4_gpu_tensor_write(routes, 0u, warm, sizeof(warm)));
    CHECK(fixture_select(&s, &f->model, &f->weights.layer[FIRST], FIRST, routes, 1u));
    CHECK(ds4_gpu_tensor_write(routes, 0u, next, sizeof(next)));
    CHECK(glm53_stream_routes(&s, routes, 1u));
    ds4_glm53_cache_slot metadata[USED];
    memcpy(metadata, s.slots, sizeof(metadata));
    const uint64_t calls = pread_calls, epoch = s.epoch, fences = sync_calls;
    fixture_busy(&s, 0u, USED);
    sync_fail = 1;
    CHECK(!glm53_stream_stage(&s, &f->model, &f->weights.layer[FIRST], FIRST, s.ids, 1u));
    CHECK(sync_calls == fences + 1u && pread_calls == calls && s.epoch == epoch);
    CHECK(!memcmp(metadata, s.slots, sizeof(metadata)) && busy_intact());
    CHECK(!memcmp(next, s.ids, sizeof(next)));
    sync_fail = 0;
    CHECK(glm53_stream_stage(&s, &f->model, &f->weights.layer[FIRST], FIRST, s.ids, 1u));
    CHECK(!busy_guard[0].tensor && s.misses == 2u * USED);
    check_routes(&s, f, FIRST, 1u, next);
    glm53_stream_free(&s);
    ds4_gpu_tensor_free(routes);
    CHECK(!gpu_bytes);
    puts("GLM stream: all-slot GPU completion guard and retry passed");
}

static void check_chunks(fixture *f) {
    fault_reset();
    enum { MAX_ROWS = 128 };
    const uint32_t widths[] = {MAX_ROWS, 1u, 16u, 32u, MAX_ROWS, 1u};
    ds4_glm53_stream s;
    stream_new(&s, f, USED);
    int32_t ids[MAX_ROWS * USED];
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(ids));
    CHECK(routes);
    for (uint32_t phase = 0u; phase < LAYERS - FIRST; phase++) {
        const uint32_t il = FIRST + phase;
        for (uint32_t i = 0u; i < MAX_ROWS * USED; i++) {
            ids[i] = (int32_t)(phase * USED + (i + phase) % USED);
        }
        CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
        for (uint32_t step = 0u; step < sizeof(widths) / sizeof(widths[0]); step++) {
            const uint64_t bytes = s.read_bytes, hits = s.hits;
            fixture_busy(&s, 0u, USED);
            CHECK(fixture_select(&s, &f->model, &f->weights.layer[il], il, routes, widths[step]));
            CHECK(!busy_guard[0].tensor);
            if (step) {
                CHECK(s.read_bytes == bytes && s.hits - hits == widths[step] * USED);
            }
            check_routes(&s, f, il, widths[step], ids);
        }
    }
    CHECK(s.misses == (LAYERS - FIRST) * USED);
    glm53_stream_free(&s);
    ds4_gpu_tensor_free(routes);
    CHECK(!gpu_bytes);
    puts("GLM stream: repeated prefill/decode/append chunks preserve warm experts");
}

static void check_faults(fixture *f) {
    fixture_restore(f);
    check_eintr(f);
    check_retry(f, READ_ERROR);
    check_retry(f, PARTIAL_UPLOAD);
    check_retry(f, TRUNCATED_FILE);
    check_fence(f);
    check_chunks(f);
    CHECK(!census_faults);
}

static void group_routes(ds4_glm53_stream *s, const fixture *f, uint32_t il,
        uint32_t rows, const int32_t *ids) {
    const uint64_t bytes = (uint64_t)rows * USED * sizeof(*ids);
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(bytes);
    CHECK(routes && ds4_gpu_tensor_write(routes, 0u, ids, bytes));
    CHECK(fixture_select(s, &f->model, &f->weights.layer[il], il, routes, rows));
    const int32_t *slots = (const int32_t *)s->selected->data;
    for (uint32_t i = 0u; i < rows * USED; i++) {
        CHECK(slots[i] == (int32_t)s->active_base + ids[i]);
    }
    check_layer(s, f, il);
    ds4_gpu_tensor_free(routes);
}

static void group_new(ds4_glm53_stream *s, fixture *f, uint32_t count) {
    fault_reset();
    fixture_restore(f);
    CHECK(setenv("DS4_GLM53_PREFETCH", "1", 1) == 0);
    stream_new(s, f, count);
}

static void group_free(ds4_glm53_stream *s) {
    CHECK(busy_intact());
    busy_release();
    glm53_stream_free(s);
    CHECK(!gpu_bytes && !upload_live && !census_faults);
    CHECK(unsetenv("DS4_GLM53_PREFETCH") == 0);
}

static void check_group_ctor(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    upload_new_fail = 1;
    CHECK(!glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    CHECK(!upload_new_fail && !upload_live && !s.reading);

    /* Graph reset preserves its engine's cache. The same stream must recover
     * when a transient upload-stream constructor failure clears. */
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    CHECK(s.upload && s.ahead && s.grouped);
    check_layer(&s, f, FIRST);
    CHECK(glm53_stream_join(&s));
    group_free(&s);
    puts("GLM stream: transient upload constructor failure permits same-stream retry");
}

#ifdef GLM53_CLOSE_FIXTURE
/* The generator includes the actual engine-close body. Unrelated family,
 * vocabulary and device cleanup are mocked; expert I/O/free remain real. */
typedef struct {
    void *qwen_ple_cuda, *qwen_ple_store;
    ds4_weights weights;
    bool mtp_ready, dspark_ready;
    ds4_model mtp_model, dspark_model, vision_model, model;
    ds4_glm53_stream glm53_stream;
    float *directional_steering_dirs;
    char *directional_steering_file;
} close_engine;

static void close_weights(ds4_weights *w) {
    CHECK(pthread_mutex_lock(&read_mutex) == 0);
    weights_free(w);
    close_weights_cleared = true;
    CHECK(pthread_cond_broadcast(&read_ready) == 0);
    /* Hold the failing close at its invalidation boundary until the blocked
     * reader checks its borrowed table. This forces the unsafe schedule. */
    while (close_weights_watch && !close_read_seen) {
        CHECK(pthread_cond_wait(&read_ready, &read_mutex) == 0);
    }
    CHECK(pthread_mutex_unlock(&read_mutex) == 0);
}

static void close_model(ds4_model *m) {
    /* The fake map is borrowed from the file fixture, so skip munmap/free. */
    if (m->fd >= 0) { CHECK(close(m->fd) == 0); }
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

#undef DS4_NO_GPU
#define ds4_engine close_engine
#define ds4_engine_close fixture_close
#define weights_free close_weights
#define model_close close_model
#define qwen4exp_report_ple_stats(...) ((void)0)
#define ds4_qwen38_ple_cuda_destroy(...) ((void)0)
#define ds4_ple_store_close(...) ((void)0)
#define vocab_free(...) ((void)0)
#define ds4_threads_shutdown(...) ((void)0)
#define mimo2_media_free(...) ((void)0)
#define ds4_gpu_cleanup(...) ((void)0)
#define ds4_release_instance_lock(...) ((void)0)
#include GLM53_CLOSE_FIXTURE
#undef ds4_release_instance_lock
#undef ds4_gpu_cleanup
#undef mimo2_media_free
#undef ds4_threads_shutdown
#undef vocab_free
#undef ds4_ple_store_close
#undef ds4_qwen38_ple_cuda_destroy
#undef qwen4exp_report_ple_stats
#undef model_close
#undef weights_free
#undef ds4_engine_close
#undef ds4_engine
#define DS4_NO_GPU

static void *close_run(void *arg) {
    fixture_close(arg);
    return NULL;
}

static void close_await(ds4_glm53_stream *s) {
    const uint64_t deadline = glm53_stream_ns() + 5000000000u;
    for (;;) {
        CHECK(pthread_mutex_lock(&read_mutex) == 0);
        const bool cleared = close_weights_cleared;
        CHECK(pthread_mutex_unlock(&read_mutex) == 0);
        CHECK(pthread_mutex_lock(&s->lock) == 0);
        const bool stopped = s->stop;
        CHECK(pthread_mutex_unlock(&s->lock) == 0);
        if (cleared || stopped) { return; }
        CHECK(glm53_stream_ns() < deadline);
        sched_yield();
    }
}

static void check_group_close(fixture *f) {
    close_engine *e = calloc(1u, sizeof(*e));
    CHECK(e);
    e->weights = f->weights;
    e->model = f->model;
    e->model.fd = dup(f->model.fd);
    CHECK(e->model.fd >= 0);
    group_new(&e->glm53_stream, f, 2u * EXPERTS);
    read_arm((off_t)f->routed[FIRST + 1u][0].abs_offset);
    CHECK(glm53_stream_begin(&e->glm53_stream, &e->model, &e->weights, FIRST));
    read_await();

    CHECK(pthread_mutex_lock(&read_mutex) == 0);
    close_model_watch = &e->model;
    close_weights_watch = &e->weights.layer[FIRST + 1u];
    close_weights_cleared = close_read_seen = false;
    close_read_valid = true;
    CHECK(pthread_mutex_unlock(&read_mutex) == 0);
    pthread_t closing;
    CHECK(pthread_create(&closing, NULL, close_run, e) == 0);
    close_await(&e->glm53_stream);
    read_open();
    CHECK(pthread_join(closing, NULL) == 0);
    close_model_watch = NULL;
    close_weights_watch = NULL;
    CHECK(close_read_seen && close_read_valid);
    CHECK(!gpu_bytes && !upload_live && !census_faults);
    CHECK(unsetenv("DS4_GLM53_PREFETCH") == 0);
    puts("GLM stream: engine close preserves borrowed weights/model until reader join");
}
#endif

static void check_group_chain(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    const uint64_t reads = pread_bytes, uploads = upload_bytes;
    read_arm((off_t)f->routed[FIRST + 1u][0].abs_offset);
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    read_await();
    CHECK(s.reading && s.active_layer == FIRST);
    const uint32_t base = check_layer(&s, f, FIRST);
    check_absent(&s, FIRST + 1u);
    const int32_t ids[] = {0, 1, 2};
    group_routes(&s, f, FIRST, 1u, ids);
    fixture_busy(&s, base, EXPERTS);
    read_open();
    CHECK(glm53_stream_join(&s));
    CHECK(busy_intact() && busy_guard[0].tensor);
    check_layer(&s, f, FIRST);
    check_layer(&s, f, FIRST + 1u);
    CHECK(glm53_stream_end(&s, FIRST, GLM53_HOT_KEEP));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST + 1u));
    group_routes(&s, f, FIRST + 1u, 1u, ids);
    CHECK(glm53_stream_end(&s, FIRST + 1u, GLM53_HOT_KEEP));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, LAYERS - 1u));
    group_routes(&s, f, LAYERS - 1u, 1u, ids);
    CHECK(!s.reading && !busy_guard[0].tensor);
    check_layer(&s, f, FIRST + 1u);
    check_layer(&s, f, LAYERS - 1u);
    check_absent(&s, FIRST);
    uint64_t total = 0u;
    for (uint32_t il = FIRST; il < LAYERS; il++) { total += layer_bytes(f, il); }
    CHECK(s.read_bytes == total && s.full_bytes == total);
    CHECK(pread_bytes - reads == total && upload_bytes - uploads == total);
    CHECK(s.misses == (LAYERS - FIRST) * EXPERTS);
    const uint64_t calls = pread_calls;
    CHECK(glm53_stream_end(&s, LAYERS - 1u, GLM53_HOT_KEEP));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, LAYERS - 1u));
    CHECK(pread_calls == calls && s.read_bytes == total);
    group_free(&s);
    puts("GLM stream: independent groups, atomic commit and full-layer reuse passed");
}

static void check_group_eintr(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    const uint64_t reads = pread_bytes;
    read_arm((off_t)f->routed[FIRST + 1u][0].abs_offset);
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    read_await();
    fixture_busy(&s, check_layer(&s, f, FIRST), EXPERTS);
    pread_intr = 3;
    pread_limit = 257u;
    read_open();
    CHECK(glm53_stream_join(&s));
    CHECK(!pread_intr && busy_intact());
    check_layer(&s, f, FIRST + 1u);
    CHECK(s.read_bytes == layer_bytes(f, FIRST) + layer_bytes(f, FIRST + 1u));
    CHECK(s.read_bytes == pread_bytes - reads);
    group_free(&s);
    puts("GLM stream: background EINTR and partial-read offsets passed");
}

static void check_group_retry(fixture *f, enum read_fault fault) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    const uint32_t next = FIRST + 1u;
    const ds4_tensor *gate = &f->routed[next][0];
    const uint64_t unit = gate->bytes / EXPERTS;
    const uint64_t reads = pread_bytes;
    read_arm((off_t)(gate->abs_offset + unit));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    read_await();
    CHECK(s.job_bytes > 0u);
    fixture_busy(&s, check_layer(&s, f, FIRST), EXPERTS);
    check_absent(&s, next);
    if (fault == READ_ERROR) { pread_fail = 1; }
    if (fault == PARTIAL_UPLOAD) { upload_fail = 2; upload_fault = UPLOAD_PARTIAL; }
    if (fault == TRUNCATED_FILE) {
        CHECK(ftruncate(f->model.fd, (off_t)(gate->abs_offset + 2u * unit - 1u)) == 0);
    }
    read_open();
    CHECK(!glm53_stream_join(&s));
    CHECK(!s.reading && s.misses == EXPERTS && busy_intact());
    check_absent(&s, next);
    check_layer(&s, f, FIRST);
    CHECK(s.read_bytes == pread_bytes - reads);
    fault_reset();
    fixture_restore(f);
    const uint64_t calls = pread_calls, before = s.read_bytes;
    CHECK(glm53_stream_queue(&s, &f->model, &f->weights, next));
    CHECK(glm53_stream_join(&s));
    CHECK(pread_calls - calls == BUSY_TENSORS * EXPERTS);
    CHECK(s.read_bytes - before == layer_bytes(f, next));
    CHECK(s.misses == 2u * EXPERTS);
    check_layer(&s, f, next);
    group_free(&s);
    printf("GLM stream: background fault=%u invalidation and retry passed\n", (unsigned)fault);
}

static void check_group_faults(fixture *f) {
    check_group_retry(f, READ_ERROR);
    check_group_retry(f, PARTIAL_UPLOAD);
    check_group_retry(f, TRUNCATED_FILE);
}

static void *cancel_run(void *arg) {
    glm53_stream_cancel(arg);
    return NULL;
}

static void *free_run(void *arg) {
    glm53_stream_free(arg);
    return NULL;
}

static void *finish_run(void *arg) {
    glm53_stream_finish(arg);
    return NULL;
}

static void stop_await(ds4_glm53_stream *s) {
    const uint64_t deadline = glm53_stream_ns() + 5000000000u;
    for (;;) {
        CHECK(pthread_mutex_lock(&s->lock) == 0);
        const bool stopped = s->stop;
        CHECK(pthread_mutex_unlock(&s->lock) == 0);
        if (stopped) { return; }
        CHECK(glm53_stream_ns() < deadline);
        sched_yield();
    }
}

static void check_group_cancel(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    const uint32_t next = FIRST + 1u;
    const ds4_tensor *gate = &f->routed[next][0];
    const uint64_t reads = pread_bytes;
    read_arm((off_t)(gate->abs_offset + gate->bytes / EXPERTS));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    read_await();
    CHECK(s.job_bytes > 0u);
    fixture_busy(&s, check_layer(&s, f, FIRST), EXPERTS);
    pthread_t cancel;
    CHECK(pthread_create(&cancel, NULL, cancel_run, &s) == 0);
    stop_await(&s);
    read_open();
    CHECK(pthread_join(cancel, NULL) == 0);
    CHECK(!s.reading && s.active_layer == UINT32_MAX && busy_intact());
    check_absent(&s, next);
    check_layer(&s, f, FIRST);
    CHECK(s.read_bytes == pread_bytes - reads);
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, next));
    check_layer(&s, f, next);
    CHECK(glm53_stream_join(&s));
    group_free(&s);
    puts("GLM stream: blocked-read cancellation, byte accounting and retry passed");
}

static void check_group_free(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    read_arm((off_t)f->routed[FIRST + 1u][0].abs_offset);
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    read_await();
    pthread_t freeing;
    CHECK(pthread_create(&freeing, NULL, free_run, &s) == 0);
    stop_await(&s);
    read_open();
    CHECK(pthread_join(freeing, NULL) == 0);
    CHECK(!s.count && !s.reading && !s.lock_ready && !s.ahead && !s.upload);
    CHECK(!gpu_bytes && !upload_live && !census_faults);
    CHECK(unsetenv("DS4_GLM53_PREFETCH") == 0);
    puts("GLM stream: free joins a blocked reader before releasing storage");
}

static void check_finished(const ds4_glm53_stream *s) {
    CHECK(!s->ahead && !s->upload && !s->lock_ready && !s->reading && !s->grouped);
    CHECK(!s->job_model && !s->job_weights && s->active_layer == UINT32_MAX);
    CHECK(!upload_live && s->count && s->gate && s->up && s->down);
    CHECK(s->staging && s->slots && s->seen);
    CHECK(!s->ids_cap || (s->selected && s->ids));
}

static void check_finish_keep(ds4_glm53_stream *s, const fixture *f) {
    const uint64_t bytes = gpu_bytes, reads = s->read_bytes;
    const uint32_t ids_cap = s->ids_cap;
    const ds4_gpu_tensor *selected = s->selected;
    const int32_t *ids = s->ids;
    const size_t meta_bytes = s->count * sizeof(*s->slots);
    ds4_glm53_cache_slot *meta = malloc(meta_bytes);
    CHECK(meta && !s->reading);
    memcpy(meta, s->slots, meta_bytes);
    const uint32_t all = (DS4_N_LAYER - DS4_N_LEADING_DENSE) * DS4_N_EXPERT;
    if (s->grouped && s->count < all) {
        /* Staging becomes vacant decode slots; copied hot experts remain. */
        for (uint32_t i = 0u; i < GLM53_STAGE_BANKS * DS4_N_EXPERT; i++) {
            meta[i].used = 0u;
        }
    }
    CHECK(busy_intact());
    busy_release();
    fixture_busy(s, 0u, s->count);
    glm53_stream_finish(s);
    check_finished(s);
    CHECK(gpu_bytes == bytes && s->read_bytes == reads && busy_intact());
    CHECK(s->ids_cap == ids_cap && s->selected == selected && s->ids == ids);
    CHECK(!memcmp(meta, s->slots, meta_bytes));
    for (uint32_t i = 0u; i < s->count; i++) {
        if (s->slots[i].used) { check_payload(s, f, s->slots[i].layer, i, s->slots[i].expert); }
    }
    /* Repeated completion must not double-count a joined job or free cache. */
    glm53_stream_finish(s);
    CHECK(gpu_bytes == bytes && s->read_bytes == reads && busy_intact());
    free(meta);
    busy_release();
}

static void check_group_finish(fixture *f) {
    enum { ROWS = 128, HOT = USED, PASSES = 2 };
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS + (LAYERS - FIRST) * HOT);
    int32_t ids[ROWS * USED], append[ROWS * USED];
    const uint32_t widths[] = {1u, 16u, 32u, ROWS};
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(append));
    CHECK(routes);
    for (uint32_t pass = 0u; pass < PASSES; pass++) {
        for (uint32_t i = 0u; i < ROWS * USED; i++) { ids[i] = (int32_t)((i + pass) % EXPERTS); }
        for (uint32_t il = FIRST; il < LAYERS; il++) {
            CHECK(glm53_stream_begin(&s, &f->model, &f->weights, il));
            group_routes(&s, f, il, ROWS, ids);
            CHECK(glm53_stream_end(&s, il, GLM53_HOT_KEEP));
        }
        CHECK(s.lock_ready && s.upload && s.ahead);
        check_finish_keep(&s, f);
        for (uint32_t i = 0u; i < ROWS * USED; i++) { append[i] = ids[ROWS * USED - 1u - i % USED]; }
        CHECK(ds4_gpu_tensor_write(routes, 0u, append, sizeof(append)));
        for (uint32_t step = 0u; step < sizeof(widths) / sizeof(*widths); step++) {
            for (uint32_t il = FIRST; il < LAYERS; il++) {
                const uint64_t reads = s.read_bytes, hits = s.hits;
                CHECK(fixture_select(&s, &f->model, &f->weights.layer[il], il, routes, widths[step]));
                check_routes(&s, f, il, widths[step], append);
                CHECK(s.read_bytes == reads && s.hits - hits == widths[step] * USED);
                check_finished(&s);
            }
        }
    }
    ds4_gpu_tensor_free(routes);
    group_free(&s);
    puts("GLM stream: finish retains hot bytes across short appends and later full passes");
}

static void check_hot_cache(fixture *f) {
    enum { ROWS = 16, ALL = (LAYERS - FIRST) * EXPERTS };
    ds4_engine_options opt = {0};
    opt.ssd_streaming_cache_experts = ALL;
    ds4_glm53_stream s = {0};
    CHECK(glm53_stream_alloc(&s, &f->model, &f->weights, &opt));
    CHECK(unsetenv("DS4_GLM53_HOT_CACHE") == 0);
    glm53_stream_hot_begin(&s, ROWS);
    CHECK(s.hot_active);
    int32_t ids[ROWS * USED];
    for (uint32_t i = 0u; i < ROWS * USED; i++) { ids[i] = (int32_t)(i % EXPERTS); }
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(ids));
    CHECK(routes && ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    for (uint32_t il = FIRST; il < LAYERS; il++) {
        CHECK(fixture_select(&s, &f->model, &f->weights.layer[il], il, routes, ROWS));
        check_routes(&s, f, il, ROWS, ids);
    }
    CHECK(s.hot_kept == ALL && glm53_cache_victim(s.slots, s.count, s.epoch + 1u) < 0);
    glm53_stream_hot_end(&s);
    CHECK(!s.hot_active && glm53_cache_victim(s.slots, s.count, s.epoch + 1u) >= 0);
    const uint64_t reads = s.read_bytes;
    for (uint32_t il = FIRST; il < LAYERS; il++) {
        CHECK(fixture_select(&s, &f->model, &f->weights.layer[il], il, routes, 1u));
    }
    CHECK(s.read_bytes == reads);
    CHECK(setenv("DS4_GLM53_HOT_CACHE", "0", 1) == 0);
    glm53_stream_hot_begin(&s, ROWS);
    CHECK(!s.hot_active);
    CHECK(unsetenv("DS4_GLM53_HOT_CACHE") == 0);
    ds4_gpu_tensor_free(routes);
    glm53_stream_free(&s);
    opt.ssd_streaming_cache_experts = USED;
    CHECK(glm53_stream_alloc(&s, &f->model, &f->weights, &opt));
    glm53_stream_hot_begin(&s, ROWS);
    CHECK(!s.hot_active);
    glm53_stream_free(&s);
    CHECK(unsetenv("DS4_GLM53_HOT_CACHE") == 0);
    CHECK(!gpu_bytes);
    puts("GLM stream: recent routes survive all layers, small cache falls back");
}

static void finish_retry(fixture *f, enum read_fault fault) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    const uint32_t next = FIRST + 1u;
    const ds4_tensor *gate = &f->routed[next][0];
    const uint64_t unit = gate->bytes / EXPERTS, reads = pread_bytes;
    read_arm((off_t)(gate->abs_offset + unit));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    read_await();
    if (fault == READ_ERROR) { pread_fail = 1; }
    if (fault == PARTIAL_UPLOAD) { upload_fail = 2; upload_fault = UPLOAD_PARTIAL; }
    if (fault == TRUNCATED_FILE) {
        CHECK(ftruncate(f->model.fd, (off_t)(gate->abs_offset + 2u * unit - 1u)) == 0);
    }
    read_open();
    CHECK(!glm53_stream_join(&s));
    check_absent(&s, next);
    check_layer(&s, f, FIRST);
    CHECK(s.read_bytes == pread_bytes - reads);
    check_finish_keep(&s, f);
    check_absent(&s, next);
    fault_reset();
    fixture_restore(f);
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, next));
    check_layer(&s, f, next);
    CHECK(glm53_stream_join(&s));
    check_layer(&s, f, next + 1u);
    CHECK(s.read_bytes == pread_bytes - reads);
    check_finish_keep(&s, f);
    group_free(&s);
}

static void check_finish_retry(fixture *f) {
    finish_retry(f, READ_ERROR);
    finish_retry(f, PARTIAL_UPLOAD);
    finish_retry(f, TRUNCATED_FILE);
    puts("GLM stream: failed reads/uploads finish cleanly and retry with fresh host staging");
}

static void check_finish_cancel(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    const uint32_t next = FIRST + 1u;
    const ds4_tensor *gate = &f->routed[next][0];
    const uint64_t reads = pread_bytes;
    read_arm((off_t)(gate->abs_offset + gate->bytes / EXPERTS));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    read_await();
    fixture_busy(&s, check_layer(&s, f, FIRST), EXPERTS);
    pthread_t finishing;
    CHECK(pthread_create(&finishing, NULL, finish_run, &s) == 0);
    stop_await(&s);
    read_open();
    CHECK(pthread_join(finishing, NULL) == 0);
    check_finished(&s);
    CHECK(busy_intact() && s.read_bytes == pread_bytes - reads);
    check_absent(&s, next);
    check_absent(&s, FIRST);
    check_finish_keep(&s, f);
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    check_layer(&s, f, FIRST);
    CHECK(glm53_stream_join(&s));
    check_layer(&s, f, next);
    CHECK(s.read_bytes == pread_bytes - reads);
    check_finish_keep(&s, f);
    group_free(&s);
    puts("GLM stream: finish joins cancellation and returns staging slots for retry");
}

static void check_group_fence(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    for (uint32_t il = FIRST; il < LAYERS; il++) {
        CHECK(glm53_stream_begin(&s, &f->model, &f->weights, il));
    }
    CHECK(!s.reading);
    ds4_glm53_cache_slot before[2u * EXPERTS];
    memcpy(before, s.slots, sizeof(before));
    const uint64_t calls = pread_calls, epoch = s.epoch;
    fixture_busy(&s, 0u, s.count);
    sync_fail = 1;
    CHECK(!glm53_stream_queue(&s, &f->model, &f->weights, FIRST));
    CHECK(!s.reading && pread_calls == calls && s.epoch == epoch);
    CHECK(!memcmp(before, s.slots, sizeof(before)) && busy_intact());
    sync_fail = 0;
    CHECK(glm53_stream_queue(&s, &f->model, &f->weights, FIRST));
    CHECK(glm53_stream_join(&s));
    CHECK(!busy_guard[0].tensor);
    check_layer(&s, f, FIRST);
    group_free(&s);
    puts("GLM stream: whole-group GPU fence failure and retry passed");
}

static void check_group_lru(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS);
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, LAYERS - 1u));
    CHECK(glm53_stream_end(&s, LAYERS - 1u, GLM53_HOT_KEEP));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    CHECK(glm53_stream_join(&s));
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    const int32_t ids[] = {0, 1, 2};
    group_routes(&s, f, FIRST, 1u, ids);
    CHECK(glm53_stream_end(&s, FIRST, GLM53_HOT_KEEP));
    const uint64_t calls = pread_calls;
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, LAYERS - 1u));
    CHECK(pread_calls - calls == BUSY_TENSORS * EXPERTS);
    check_layer(&s, f, FIRST);
    check_layer(&s, f, LAYERS - 1u);
    check_absent(&s, FIRST + 1u);
    group_free(&s);
    puts("GLM stream: active reuse refreshes whole-group LRU passed");
}

static void check_group_hot(fixture *f) {
    enum { ROWS = 16, HOT = 2 };
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS + (LAYERS - FIRST) * HOT);
    int32_t ids[ROWS * USED];
    for (uint32_t i = 0u; i < ROWS * USED; i++) {
        ids[i] = (int32_t)(i % USED + (i >= (ROWS - GLM53_HOT_ROWS) * USED ? 9u : 0u));
    }
    const uint64_t copies = copy_calls;
    for (uint32_t il = FIRST; il < LAYERS; il++) {
        CHECK(glm53_stream_begin(&s, &f->model, &f->weights, il));
        group_routes(&s, f, il, ROWS, ids);
        CHECK(glm53_stream_end(&s, il, GLM53_HOT_KEEP));
    }
    CHECK(!s.reading && s.hot_kept == (LAYERS - FIRST) * HOT);
    CHECK(copy_calls - copies == (LAYERS - FIRST) * HOT * BUSY_TENSORS);
    const uint32_t first = 2u * EXPERTS;
    CHECK(s.slots[first].layer == FIRST && s.slots[first].expert == 11u);
    CHECK(s.slots[first + 1u].layer == FIRST && s.slots[first + 1u].expert == 10u);
    const int32_t hot[] = {11, 10, 11};
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(hot));
    CHECK(routes && ds4_gpu_tensor_write(routes, 0u, hot, sizeof(hot)));
    const uint64_t reads = s.read_bytes, hits = s.hits;
    CHECK(fixture_select(&s, &f->model, &f->weights.layer[FIRST], FIRST, routes, 1u));
    CHECK(s.read_bytes == reads && s.hits - hits == USED);
    check_routes(&s, f, FIRST, 1u, hot);
    ds4_gpu_tensor_free(routes);
    group_free(&s);
    puts("GLM stream: final eight rows preserve decode hot experts without rereads");
}

static void check_group_copy(fixture *f) {
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS + LAYERS - FIRST);
    const uint32_t il = LAYERS - 1u, dst = 2u * EXPERTS + il - FIRST;
    const int32_t warm[] = {0, 1, 2}, next[] = {9, 10, 11};
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, il));
    group_routes(&s, f, il, 1u, warm);
    CHECK(glm53_stream_end(&s, il, GLM53_HOT_KEEP));
    check_payload(&s, f, il, dst, 2u);
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, il));
    group_routes(&s, f, il, 1u, next);
    fixture_busy(&s, check_layer(&s, f, il), EXPERTS);
    copy_fail = 2;
    CHECK(!glm53_stream_end(&s, il, GLM53_HOT_KEEP));
    CHECK(!s.slots[dst].used && s.active_layer == il && s.hot_kept == 1u);
    check_layer(&s, f, il);
    CHECK(!copy_fail && glm53_stream_end(&s, il, GLM53_HOT_KEEP));
    CHECK(s.hot_kept == 2u);
    check_payload(&s, f, il, dst, 11u);
    group_free(&s);
    puts("GLM stream: partial hot D2D copy invalidates its destination and retries");
}

static void check_hot_budget(fixture *f) {
    enum { ROWS = 16, HOT = 3 };
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS + (LAYERS - FIRST) * HOT);
    int32_t ids[ROWS * USED];
    for (uint32_t i = 0u; i < ROWS * USED; i++) {
        ids[i] = i < (ROWS - GLM53_HOT_ROWS) * USED ? (int32_t)(i % 2u) : 11;
    }
    const uint64_t copies = copy_calls;
    for (uint32_t il = FIRST; il < LAYERS; il++) {
        CHECK(glm53_stream_begin(&s, &f->model, &f->weights, il));
        group_routes(&s, f, il, ROWS, ids);
        CHECK(glm53_stream_end(&s, il, GLM53_HOT_KEEP));
    }
    // Duplicate final routes must not leave the already-budgeted cache empty.
    // Keep newest unique experts, then serve them after staging is returned.
    CHECK(s.hot_kept == (LAYERS - FIRST) * HOT);
    CHECK(copy_calls - copies == (LAYERS - FIRST) * HOT * BUSY_TENSORS);
    const uint32_t first = 2u * EXPERTS;
    const uint32_t recent[HOT] = {11u, 1u, 0u};
    for (uint32_t il = FIRST; il < LAYERS; il++) {
        for (uint32_t n = 0u; n < HOT; n++) {
            check_payload(&s, f, il, first + (il - FIRST) * HOT + n, recent[n]);
        }
    }
    glm53_stream_finish(&s);
    const int32_t decode[] = {11, 1, 0};
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(decode));
    CHECK(routes && ds4_gpu_tensor_write(routes, 0u, decode, sizeof(decode)));
    const uint64_t reads = s.read_bytes;
    for (uint32_t il = FIRST; il < LAYERS; il++) {
        CHECK(fixture_select(&s, &f->model, &f->weights.layer[il], il, routes, 1u));
        check_routes(&s, f, il, 1u, decode);
    }
    CHECK(s.read_bytes == reads);
    ds4_gpu_tensor_free(routes);
    group_free(&s);

    const char *hot_copy_env = "DS4_GLM53_HOT_COPY";
    CHECK(setenv(hot_copy_env, "0", 1) == 0);
    group_new(&s, f, 2u * EXPERTS + (LAYERS - FIRST) * HOT);
    const uint32_t last = LAYERS - 1u;
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, last));
    group_routes(&s, f, last, ROWS, ids);
    CHECK(glm53_stream_end(&s, last, GLM53_HOT_KEEP) && s.hot_kept == 1u);
    glm53_stream_finish(&s);
    CHECK(glm53_cache_find(s.slots, s.count, last, 11u) >= 0);
    CHECK(glm53_cache_find(s.slots, s.count, last, 0u) < 0);
    CHECK(glm53_cache_find(s.slots, s.count, last, 1u) < 0);
    group_free(&s);
    CHECK(unsetenv(hot_copy_env) == 0);
    puts("GLM stream: recent unique routes fill the hot budget without rereads");
}

static void check_hot_tail(fixture *f) {
    enum { ROWS = 16, HOT = 3 };
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS + (LAYERS - FIRST) * HOT);
    int32_t ids[ROWS * USED];
    for (uint32_t i = 0u; i < ROWS * USED; i++) { ids[i] = (int32_t)(i % HOT); }
    const int32_t tail[USED] = {11, 11, 11};
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    group_routes(&s, f, FIRST, ROWS, ids);
    group_routes(&s, f, FIRST, 1u, tail);
    CHECK(glm53_stream_end(&s, FIRST, GLM53_HOT_KEEP) && s.hot_kept == HOT);
    // A short final chunk must retain recent routes from the same layer's
    // preceding chunk, rather than leaving its funded hot slots empty.
    glm53_stream_finish(&s);
    const uint32_t recent[HOT] = {11u, 2u, 1u};
    for (uint32_t n = 0u; n < HOT; n++) {
        check_payload(&s, f, FIRST, 2u * EXPERTS + n, recent[n]);
    }
    group_free(&s);
    puts("GLM stream: short tails retain recent routes from the full layer window");
}

static void check_hot_last(fixture *f) {
    enum { ROWS = 16, HOT = 3 };
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS + (LAYERS - FIRST) * HOT);
    int32_t ids[ROWS * USED];
    for (uint32_t i = 0u; i < ROWS * USED; i++) { ids[i] = (int32_t)(i % HOT); }
    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    group_routes(&s, f, FIRST, ROWS, ids);
    const uint64_t before = copy_calls;
    CHECK(glm53_stream_end(&s, FIRST, GLM53_HOT_SKIP));
    CHECK(copy_calls == before && s.hot_kept == 0u && s.active_layer == UINT32_MAX);
    for (uint32_t i = 2u * EXPERTS; i < s.count; i++) { CHECK(!s.slots[i].used); }
    glm53_stream_finish(&s);

    CHECK(glm53_stream_begin(&s, &f->model, &f->weights, FIRST));
    group_routes(&s, f, FIRST, ROWS, ids);
    CHECK(glm53_stream_end(&s, FIRST, GLM53_HOT_KEEP));
    CHECK(s.hot_kept == HOT && copy_calls - before == HOT * BUSY_TENSORS);
    glm53_stream_finish(&s);
    group_free(&s);
    puts("GLM stream: intermediate windows avoid unused hot copies");
}

static void check_group_full(fixture *f) {
    enum { MAX_ROWS = 2048 };
    ds4_glm53_stream s;
    group_new(&s, f, (LAYERS - FIRST) * EXPERTS);
    const uint64_t reads = pread_bytes;
    const uint32_t widths[] = {1u, 128u, MAX_ROWS, 16u};
    int32_t ids[MAX_ROWS * USED];
    for (uint32_t i = 0u; i < MAX_ROWS * USED; i++) { ids[i] = (int32_t)(i % EXPERTS); }
    uint64_t total = 0u;
    for (uint32_t il = FIRST; il < LAYERS; il++) { total += layer_bytes(f, il); }
    for (uint32_t pass = 0u; pass < 4u; pass++) {
        for (uint32_t il = FIRST; il < LAYERS; il++) {
            CHECK(glm53_stream_begin(&s, &f->model, &f->weights, il));
            group_routes(&s, f, il, widths[pass], ids);
            CHECK(glm53_stream_end(&s, il, GLM53_HOT_KEEP));
        }
        CHECK(!s.reading && s.read_bytes == total && s.full_bytes == total);
        CHECK(pread_bytes - reads == total);
        for (uint32_t il = FIRST; il < LAYERS; il++) { check_layer(&s, f, il); }
    }
    group_free(&s);
    puts("GLM stream: full cache retains three complete layers across repeated prefill");
}

static void check_group_rounds(fixture *f) {
    enum { ROWS = 128, ROUNDS = 16, HOT = 2 };
    ds4_glm53_stream s;
    group_new(&s, f, 2u * EXPERTS + (LAYERS - FIRST) * HOT);
    const uint64_t reads = pread_bytes;
    int32_t ids[ROWS * USED];
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(USED * sizeof(*ids));
    CHECK(routes);
    for (uint32_t pass = 0u; pass < ROUNDS; pass++) {
        for (uint32_t i = 0u; i < ROWS * USED; i++) { ids[i] = (int32_t)((i + pass) % EXPERTS); }
        for (uint32_t il = FIRST; il < LAYERS; il++) {
            CHECK(glm53_stream_begin(&s, &f->model, &f->weights, il));
            group_routes(&s, f, il, ROWS, ids);
            fixture_busy(&s, s.active_base, EXPERTS);
            CHECK(glm53_stream_end(&s, il, GLM53_HOT_KEEP));
        }
        const int32_t hot[] = {ids[ROWS * USED - 1u], ids[ROWS * USED - 2u], ids[ROWS * USED - 1u]};
        CHECK(!s.reading && !busy_guard[0].tensor);
        CHECK(ds4_gpu_tensor_write(routes, 0u, hot, sizeof(hot)));
        const uint64_t before = s.read_bytes, hits = s.hits;
        CHECK(fixture_select(&s, &f->model, &f->weights.layer[FIRST], FIRST, routes, 1u));
        CHECK(s.read_bytes == before && s.hits - hits == USED);
        check_routes(&s, f, FIRST, 1u, hot);
        CHECK(s.read_bytes == pread_bytes - reads);
    }
    ds4_gpu_tensor_free(routes);
    group_free(&s);
    puts("GLM stream: sixteen prefill/decode transitions preserve latest hot payloads");
}

typedef struct { const char *name; void (*run)(fixture *); } group_case;
static const group_case group_cases[] = {
    {"ctor", check_group_ctor},
#ifdef GLM53_CLOSE_FIXTURE
    {"close", check_group_close},
#endif
    {"chain", check_group_chain}, {"eintr", check_group_eintr},
    {"retry", check_group_faults}, {"cancel", check_group_cancel},
    {"free", check_group_free}, {"fence", check_group_fence},
    {"finish", check_group_finish}, {"finish_retry", check_finish_retry},
    {"finish_cancel", check_finish_cancel},
    {"lru", check_group_lru}, {"hot", check_group_hot},
    {"copy", check_group_copy}, {"hot_budget", check_hot_budget}, {"hot_tail", check_hot_tail},
    {"hot_last", check_hot_last},
    {"full", check_group_full},
    {"rounds", check_group_rounds},
};

static void check_prefetch(fixture *f, const char *name) {
    uint32_t ran = 0u;
    for (uint32_t i = 0u; i < sizeof(group_cases) / sizeof(group_cases[0]); i++) {
        if (name && strcmp(name, group_cases[i].name)) { continue; }
        group_cases[i].run(f);
        ran++;
    }
    CHECK(ran);
}

int main(int argc, char **argv) {
    fixture f;
    fixture_init(&f);
    bound_map = f.model.map;
    if (argc >= 2 && strcmp(argv[1], "prefetch") == 0) {
        check_prefetch(&f, argc >= 3 ? argv[2] : NULL);
        CHECK(close(f.model.fd) == 0);
        free(f.bytes);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "accounting") == 0) {
        check_accounting(&f);
        CHECK(close(f.model.fd) == 0);
        free(f.bytes);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "faults") == 0) {
        check_faults(&f);
        CHECK(close(f.model.fd) == 0);
        free(f.bytes);
        return 0;
    }
    ds4_glm53_stream s = {0};
    const ds4_engine_options opt = {.ssd_streaming = true,
                                    .ssd_streaming_cache_experts = USED};
    CHECK(glm53_stream_alloc(&s, &f.model, &f.weights, &opt));
    CHECK(s.count == USED && gpu_bytes == s.bytes);
    CHECK(s.gate_stride % 66u == 0u && s.gate_stride % 74u == 0u);
    CHECK(s.down_stride % 74u == 0u && s.down_stride % 84u == 0u);
    CHECK(s.staging_bytes == WIDTH * 84u);
    int32_t ids[2u * USED] = {0, 1, 2, 0, 1, 2};
    ds4_gpu_tensor *routes = ds4_gpu_tensor_alloc(sizeof(ids));
    CHECK(routes && ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    CHECK(fixture_select(&s, &f.model, &f.weights.layer[3], 3u, routes, 1u));
    CHECK(s.misses == USED && !s.hits && pread_calls == 3u * USED && !advise_calls);
    const uint64_t first_bytes = s.read_bytes;
    check_slots(&s, &f, 3u, 1u);

    ids[0] = 3;
    ids[1] = 2;
    ids[2] = 1;
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    busy_tensor = s.gate;
    busy_byte = s.gate->data[0];
    CHECK(fixture_select(&s, &f.model, &f.weights.layer[3], 3u, routes, 1u));
    CHECK(!busy_tensor && s.misses == USED + 1u && s.hits == 2u);
    CHECK(glm53_cache_find(s.slots, s.count, 3u, 0u) < 0);
    CHECK(glm53_cache_find(s.slots, s.count, 3u, 2u) >= 0);
    check_slots(&s, &f, 3u, 1u);

    ids[3] = ids[0]; ids[4] = ids[1]; ids[5] = ids[2];
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    const uint64_t warm_bytes = s.read_bytes;
    CHECK(fixture_select(&s, &f.model, &f.weights.layer[3], 3u, routes, 2u));
    CHECK(s.read_bytes == warm_bytes && s.hits == 8u);
    check_slots(&s, &f, 3u, 2u);

    s.cold = true;
    CHECK(fixture_select(&s, &f.model, &f.weights.layer[4], 4u, routes, 1u));
    CHECK(s.misses == 2u * USED + 1u && advise_calls == 3u * USED);
    CHECK(s.read_bytes > first_bytes);
    CHECK(glm53_cache_find(s.slots, s.count, 3u, 3u) < 0);
    check_slots(&s, &f, 4u, 1u);

    for (unsigned i = 0u; i < 2u * USED; i++) { ids[i] = (int32_t)i; }
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    CHECK(!fixture_select(&s, &f.model, &f.weights.layer[5], 5u, routes, 2u));
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    upload_fail = 2;
    CHECK(!fixture_select(&s, &f.model, &f.weights.layer[4], 4u, routes, 1u));
    CHECK(glm53_cache_find(s.slots, s.count, 4u, 0u) < 0);
    upload_fail = 0;
    ids[0] = 11; ids[1] = 10; ids[2] = 9;
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    const ds4_tensor *g = f.weights.layer[5].ffn_gate_exps;
    CHECK(ftruncate(f.model.fd, (off_t)(g->abs_offset + g->bytes - 1u)) == 0);
    CHECK(!fixture_select(&s, &f.model, &f.weights.layer[5], 5u, routes, 1u));
    CHECK(glm53_cache_find(s.slots, s.count, 5u, 11u) < 0);

    ds4_model_map_span_vec spans;
    CHECK(glm53_stream_spans(&f.weights, &spans));
    CHECK(spans.len == 1u && spans.v[0].off == 0u && spans.v[0].end == 3072u);
    CHECK(spans.max_tensor_bytes == 1024u);
    free(spans.v);

    /* Malformed offsets must fail before an overflow wraps into another tensor. */
    ds4_tensor malformed = *f.weights.layer[3].ffn_gate_exps;
    malformed.abs_offset = UINT64_MAX - 10u;
    CHECK(!glm53_stream_read(&s, &f.model, &malformed, 1u, s.gate, 0u));
    malformed = *f.weights.layer[3].ffn_gate_exps;
    CHECK(!glm53_stream_read(&s, &f.model, &malformed, EXPERTS, s.gate, 0u));
    malformed.bytes++;
    CHECK(!glm53_stream_read(&s, &f.model, &malformed, 0u, s.gate, 0u));
    malformed.bytes = 0u;
    CHECK(!glm53_stream_read(&s, &f.model, &malformed, 0u, s.gate, 0u));
    glm53_stream_free(&s);
    ds4_gpu_tensor_free(routes);
    CHECK(!gpu_bytes);

    ds4_tensor *gate_weight = f.weights.layer[3].ffn_gate_exps;
    const uint64_t gate_bytes = gate_weight->bytes;
    gate_weight->bytes = 0u;
    CHECK(!glm53_stream_alloc(&s, &f.model, &f.weights, &opt));
    CHECK(!gpu_bytes);
    gate_weight->bytes = gate_bytes;
    f.weights.layer[3].ffn_up_exps = NULL;
    CHECK(!glm53_stream_alloc(&s, &f.model, &f.weights, &opt));
    CHECK(!gpu_bytes);

    f.weights.layer[3].ffn_up_exps = &f.routed[3][1];
    ds4_engine_options small = opt;
    small.ssd_streaming_cache_experts = USED - 1u;
    CHECK(!glm53_stream_alloc(&s, &f.model, &f.weights, &small));
    CHECK(!gpu_bytes);
    check_faults(&f);
    check_hot_cache(&f);
    check_prefetch(&f, NULL);
    CHECK(close(f.model.fd) == 0);
    free(f.bytes);
    puts("GLM stream: reads, pinned eviction, strides, failures, accounting and spans passed");
    return 0;
}
