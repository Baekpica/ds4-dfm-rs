/* Actual SSD reads with a model-free tensor backend; no CUDA allocation. */
#define DS4_NO_GPU
#include "../ds4.c"
#include "../ds4_gpu.h"

struct ds4_gpu_tensor {
    unsigned char *data;
    uint64_t bytes;
    int memc;
    const void *source;
};
static uint64_t gpu_bytes, pread_calls, advise_calls;
static ds4_mem_cell gpu_census[DS4_MEMC__COUNT], source_census[DS4_MEMC__COUNT];
static uint64_t census_faults;
static const void *bound_map;
static int upload_fail, alloc_fail;
static ds4_gpu_tensor *busy_tensor;
static unsigned char busy_byte;

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
    if (busy_tensor || !t || off > t->bytes || bytes > t->bytes - off) { return 0; }
    if (upload_fail && --upload_fail == 0) { return 0; }
    memcpy(t->data + off, data, bytes);
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off,
                        void *data, uint64_t bytes) {
    if (!t || off > t->bytes || bytes > t->bytes - off) { return 0; }
    /* The blocking route read drains the previous submission before eviction. */
    if (busy_tensor) {
        if (busy_tensor->data[0] != busy_byte) { return 0; }
        busy_tensor = NULL;
    }
    memcpy(data, t->data + off, bytes);
    return 1;
}

static ssize_t fixture_pread(int fd, void *dst, size_t bytes, off_t off) {
    pread_calls++;
    return pread(fd, dst, bytes, off);
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
                memset(f->bytes + t->abs_offset + e * unit,
                       (int)(il * 32u + j * 12u + e), unit);
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
        CHECK(slot->used && slot->layer == il && slot->pinned == s->epoch);
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

int main(int argc, char **argv) {
    fixture f;
    fixture_init(&f);
    bound_map = f.model.map;
    if (argc == 2 && strcmp(argv[1], "accounting") == 0) {
        check_accounting(&f);
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
    CHECK(glm53_stream_select(&s, &f.model, &f.weights.layer[3], 3u, routes, 1u));
    CHECK(s.misses == USED && !s.hits && pread_calls == 3u * USED && !advise_calls);
    const uint64_t first_bytes = s.read_bytes;
    check_slots(&s, &f, 3u, 1u);

    ids[0] = 3;
    ids[1] = 2;
    ids[2] = 1;
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    busy_tensor = s.gate;
    busy_byte = s.gate->data[0];
    CHECK(glm53_stream_select(&s, &f.model, &f.weights.layer[3], 3u, routes, 1u));
    CHECK(!busy_tensor && s.misses == USED + 1u && s.hits == 2u);
    CHECK(glm53_cache_find(s.slots, s.count, 3u, 0u) < 0);
    CHECK(glm53_cache_find(s.slots, s.count, 3u, 2u) >= 0);
    check_slots(&s, &f, 3u, 1u);

    ids[3] = ids[0]; ids[4] = ids[1]; ids[5] = ids[2];
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    const uint64_t warm_bytes = s.read_bytes;
    CHECK(glm53_stream_select(&s, &f.model, &f.weights.layer[3], 3u, routes, 2u));
    CHECK(s.read_bytes == warm_bytes && s.hits == 8u);
    check_slots(&s, &f, 3u, 2u);

    s.cold = true;
    CHECK(glm53_stream_select(&s, &f.model, &f.weights.layer[4], 4u, routes, 1u));
    CHECK(s.misses == 2u * USED + 1u && advise_calls == 3u * USED);
    CHECK(s.read_bytes > first_bytes);
    CHECK(glm53_cache_find(s.slots, s.count, 3u, 3u) < 0);
    check_slots(&s, &f, 4u, 1u);

    for (unsigned i = 0u; i < 2u * USED; i++) { ids[i] = (int32_t)i; }
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    CHECK(!glm53_stream_select(&s, &f.model, &f.weights.layer[5], 5u, routes, 2u));
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    upload_fail = 2;
    CHECK(!glm53_stream_select(&s, &f.model, &f.weights.layer[4], 4u, routes, 1u));
    CHECK(glm53_cache_find(s.slots, s.count, 4u, 0u) < 0);
    upload_fail = 0;
    ids[0] = 11; ids[1] = 10; ids[2] = 9;
    CHECK(ds4_gpu_tensor_write(routes, 0u, ids, sizeof(ids)));
    const ds4_tensor *g = f.weights.layer[5].ffn_gate_exps;
    CHECK(ftruncate(f.model.fd, (off_t)(g->abs_offset + g->bytes - 1u)) == 0);
    CHECK(!glm53_stream_select(&s, &f.model, &f.weights.layer[5], 5u, routes, 1u));
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
    CHECK(close(f.model.fd) == 0);
    free(f.bytes);
    puts("GLM stream: reads, pinned eviction, strides, failures, accounting and spans passed");
    return 0;
}
