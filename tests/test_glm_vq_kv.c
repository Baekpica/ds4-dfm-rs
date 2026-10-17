/* Actual KV_B fixture: compare the new handoff with the previously qualified
 * projection/attention sequence, including layout, positions and cache bytes. */
#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_glmvq.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { HEADS = 64, DIM = 256, LATENT = 512, BF16_TYPE = 30,
       MAX_KEYS = 2048, MAX_ROWS = 1024 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GVQ KV FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static float sample(uint32_t index, uint32_t salt) {
    /* Distinct rows/heads, nonzero signs and exact BF16 fractions. */
    uint32_t value = index ^ salt;
    value ^= value >> 16; value *= 0x7feb352du;
    value ^= value >> 15; value *= 0x846ca68bu;
    value ^= value >> 16;
    return ((int)(value % 255) - 127) / 128.0f;
}

static void check_paths(const ds4_gvq_model *model) {
    const uint64_t cb = (uint64_t)MAX_KEYS * LATENT * sizeof(uint16_t);
    const uint64_t qb = (uint64_t)MAX_ROWS * HEADS * DIM * sizeof(float);
    const uint64_t kb = (uint64_t)MAX_KEYS * HEADS * 2 * DIM * sizeof(float);
    uint16_t *cache = malloc(cb), *after = malloc(cb);
    float *query = malloc(qb), *reference = malloc(qb), *actual = malloc(qb);
    CHECK(cache && after && query && reference && actual);
    for (uint32_t i = 0; i < MAX_KEYS * LATENT; i++) {
        const float value = sample(i, 0x5353u);
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        cache[i] = bits >> 16;
    }
    for (uint32_t i = 0; i < MAX_ROWS * HEADS * DIM; i++) { query[i] = sample(i, 0x3219u); }

    ds4_gpu_tensor *owner = ds4_gpu_tensor_alloc(cb + sizeof(uint16_t));
    CHECK(owner);
    ds4_gpu_tensor *aligned = ds4_gpu_tensor_view(owner, 0, cb);
    ds4_gpu_tensor *unaligned = ds4_gpu_tensor_view(owner, sizeof(uint16_t), cb);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(qb), *out = ds4_gpu_tensor_alloc(qb);
    ds4_gpu_tensor *latent = ds4_gpu_tensor_alloc(cb * 2), *kv = ds4_gpu_tensor_alloc(kb);
    ds4_gpu_tensor *pos = ds4_gpu_tensor_alloc(MAX_ROWS * sizeof(uint32_t));
    ds4_gpu_tensor *short_cache = ds4_gpu_tensor_view(owner, 0, sizeof(uint16_t));
    CHECK(aligned && unaligned && q && out && latent && kv && pos && short_cache);
    CHECK(ds4_gpu_tensor_write(q, 0, query, qb));

    uint64_t scratch = ds4_gpu_gvq_attn_bytes(MAX_ROWS, MAX_KEYS, HEADS, DIM);
    const uint64_t linear = ds4_gpu_gvq_linear_bytes(HEADS * 2 * DIM, LATENT,
        MAX_KEYS, BF16_TYPE, DS4_GVQ_BF16);
    if (linear > scratch) { scratch = linear; }
    /* Expert kernels are unrelated to this isolated attention contract. */
    CHECK(!setenv("DS4_GLMVQ_MMA", "0", 1) && !setenv("DS4_GLMVQ_DECODE", "0", 1));
    CHECK(!setenv("DS4_GLMVQ_LATENT_ATTN", "0", 1));
    CHECK(!setenv("DS4_GLMVQ_KV_BF16", "0", 1));
    ds4_gvq_ctx *old = ds4_gpu_gvq_create(model, scratch, MAX_ROWS);
    CHECK(!setenv("DS4_GLMVQ_KV_BF16", "1", 1));
    ds4_gvq_ctx *fast = ds4_gpu_gvq_create(model, scratch, MAX_ROWS);
    CHECK(old && fast);
    const struct { uint32_t rows, first; } cases[] = {
        {1, 0}, {1, 1984}, {7, 26}, {64, 129}, {1024, 0}, {960, 1024}, {1, 2015}
    };
    const unsigned count = sizeof(cases) / sizeof(cases[0]);
    for (unsigned c = 0; c <= count; c++) {
        const unsigned index = c < count ? c : 2;
        const uint32_t rows = cases[index].rows, first = cases[index].first;
        ds4_gpu_tensor *input = c == count ? unaligned : aligned;
        const uint64_t bytes = (uint64_t)rows * HEADS * DIM * sizeof(float);
        CHECK(ds4_gpu_tensor_write(input, 0, cache, cb));
        /* Spell out the old production sequence independently of the wrapper. */
        CHECK(ds4_gpu_gvq_unpack(latent, input, (uint64_t)(first + rows) * LATENT));
        CHECK(ds4_gpu_gvq_linear(old, kv, 0, latent, first + rows, DS4_GVQ_BF16));
        CHECK(ds4_gpu_gvq_positions(pos, first, rows));
        CHECK(ds4_gpu_gvq_attn(old, out, q, kv, pos, NULL, 0, rows, first + rows, HEADS, DIM));
        CHECK(ds4_gpu_tensor_read(out, 0, reference, bytes));
        for (unsigned mode = 0; mode < 2; mode++) {
            CHECK(ds4_gpu_gvq_dsa(mode ? fast : old, out, q, input, 0, latent, kv, pos,
                first, rows, HEADS, DIM, NULL, 0));
            CHECK(ds4_gpu_tensor_read(out, 0, actual, bytes));
            for (uint64_t i = 0; i < bytes / sizeof(float); i++) {
                CHECK(isfinite(actual[i]) && actual[i] == reference[i]);
            }
            uint32_t positions[MAX_ROWS];
            CHECK(ds4_gpu_tensor_read(pos, 0, positions, rows * sizeof(uint32_t)));
            for (uint32_t i = 0; i < rows; i++) { CHECK(positions[i] == first + i); }
            CHECK(ds4_gpu_tensor_read(input, 0, after, cb) && !memcmp(cache, after, cb));
        }
        printf("GVQ KV rows=%u first=%u unaligned=%u exact=%llu cache/positions=PASS\n",
            rows, first, input == unaligned, (unsigned long long)(bytes / sizeof(float)));
    }
    /* Invalid geometry/capacity must fail before enqueueing work. */
    CHECK(!ds4_gpu_gvq_dsa(fast, out, q, short_cache, 0, latent, kv, pos, 0, 1, HEADS, DIM, NULL, 0));
    CHECK(!ds4_gpu_gvq_dsa(fast, out, q, aligned, 1, latent, kv, pos, 0, 1, HEADS, DIM, NULL, 0));
    CHECK(!ds4_gpu_gvq_dsa(fast, out, q, aligned, 0, latent, kv, pos, UINT32_MAX, 1, HEADS, DIM, NULL, 0));
    CHECK(!ds4_gpu_gvq_dsa(fast, out, q, aligned, 0, latent, kv, pos, 0, 1, HEADS - 1, DIM, NULL, 0));
    CHECK(!ds4_gpu_gvq_dsa(fast, out, q, aligned, 0, latent, kv, pos, 0, 1, UINT32_MAX, UINT32_MAX, NULL, 0));
    CHECK(!ds4_gpu_gvq_dsa(fast, out, q, aligned, 0, latent, kv, pos, 0, 0, HEADS, DIM, NULL, 0));
    ds4_gpu_tensor_free(short_cache); ds4_gpu_tensor_free(pos);
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(latent);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(unaligned); ds4_gpu_tensor_free(aligned); ds4_gpu_tensor_free(owner);
    ds4_gpu_gvq_free(fast); ds4_gpu_gvq_free(old);
    free(actual); free(reference); free(query); free(after); free(cache);
}

/* The optional fixture comes from the independently checked per-operation
 * FP64 probe, not from the native adapter being tested here. */
static void check_latent(const ds4_gvq_model *model, const char *path) {
    enum { MAGIC = 0x4c415451, KEYS = 1985, QUERY_ROWS = 2, FAST_KEYS = 512 };
    const uint64_t cb = (uint64_t)KEYS * LATENT * sizeof(uint16_t);
    const uint64_t qb = (uint64_t)HEADS * DIM * sizeof(float);
    const uint64_t kb = (uint64_t)KEYS * HEADS * 2 * DIM * sizeof(float);
    uint32_t header[5];
    FILE *file = fopen(path, "rb");
    uint16_t *cache = malloc(cb), *after = malloc(cb);
    float *query = malloc(qb), *expected = malloc(qb * 2);
    float *actual = malloc(qb * QUERY_ROWS), *control = malloc(qb * QUERY_ROWS);
    CHECK(file && cache && after && query && expected && actual && control);
    CHECK(fread(header, sizeof(header), 1, file) == 1 && header[0] == MAGIC &&
        header[1] == KEYS && header[2] == HEADS && header[3] == DIM && header[4] == LATENT);
    CHECK(fread(cache, cb, 1, file) == 1 && fread(query, qb, 1, file) == 1 &&
        fread(expected, qb * 2, 1, file) == 1 && fgetc(file) == EOF && !fclose(file));

    ds4_gpu_tensor *owner = ds4_gpu_tensor_alloc(cb + sizeof(uint16_t));
    CHECK(owner);
    ds4_gpu_tensor *aligned = ds4_gpu_tensor_view(owner, 0, cb);
    ds4_gpu_tensor *unaligned = ds4_gpu_tensor_view(owner, sizeof(uint16_t), cb);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(qb * QUERY_ROWS);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(qb * QUERY_ROWS);
    ds4_gpu_tensor *latent = ds4_gpu_tensor_alloc(cb * 2), *kv = ds4_gpu_tensor_alloc(kb);
    ds4_gpu_tensor *pos = ds4_gpu_tensor_alloc(QUERY_ROWS * sizeof(uint32_t));
    CHECK(aligned && unaligned && q && out && latent && kv && pos);
    for (uint32_t i = 0; i < QUERY_ROWS; i++) { CHECK(ds4_gpu_tensor_write(q, i * qb, query, qb)); }
    CHECK(!setenv("DS4_GLMVQ_MMA", "0", 1) && !setenv("DS4_GLMVQ_DECODE", "0", 1));
    CHECK(!setenv("DS4_GLMVQ_KV_BF16", "1", 1) && !setenv("DS4_GLMVQ_LATENT_ATTN", "0", 1));
    uint64_t scratch = ds4_gpu_gvq_attn_bytes(QUERY_ROWS, KEYS, HEADS, DIM);
    const uint64_t linear = ds4_gpu_gvq_linear_bytes(HEADS * 2 * DIM, LATENT,
        KEYS, BF16_TYPE, DS4_GVQ_BF16);
    if (linear > scratch) { scratch = linear; }
    ds4_gvq_ctx *old = ds4_gpu_gvq_create(model, scratch, QUERY_ROWS);
    CHECK(!setenv("DS4_GLMVQ_LATENT_ATTN", "1", 1));
    ds4_gvq_ctx *fast = ds4_gpu_gvq_create(model, scratch, QUERY_ROWS);
    CHECK(old && fast);
    const struct { uint32_t rows, keys; int fixture; } cases[] = {
        {1, FAST_KEYS - 1, -1}, {1, FAST_KEYS, 0}, {1, KEYS, 1},
        {1, KEYS, -1}, {QUERY_ROWS, KEYS, -1}
    };
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const uint32_t rows = cases[c].rows, first = cases[c].keys - rows;
        const uint64_t bytes = rows * qb;
        ds4_gpu_tensor *input = c == 3 ? unaligned : aligned;
        CHECK(ds4_gpu_tensor_write(input, 0, cache, cb));
        const float *reference = control;
        if (cases[c].fixture >= 0) {
            reference = expected + (uint64_t)cases[c].fixture * HEADS * DIM;
        } else {
            CHECK(ds4_gpu_gvq_dsa(old, out, q, input, 0, latent, kv, pos, first, rows, HEADS, DIM, NULL, 0));
            CHECK(ds4_gpu_tensor_read(out, 0, control, bytes));
        }
        CHECK(ds4_gpu_gvq_dsa(fast, out, q, input, 0, latent, kv, pos, first, rows, HEADS, DIM, NULL, 0));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, bytes));
        for (uint64_t i = 0; i < bytes / sizeof(float); i++) {
            CHECK(isfinite(actual[i]) && actual[i] == reference[i]);
        }
        uint32_t positions[QUERY_ROWS];
        CHECK(ds4_gpu_tensor_read(pos, 0, positions, rows * sizeof(uint32_t)));
        for (uint32_t i = 0; i < rows; i++) { CHECK(positions[i] == first + i); }
        CHECK(ds4_gpu_tensor_read(input, 0, after, cb) && !memcmp(cache, after, cb));
        printf("GVQ latent rows=%u keys=%u fixture=%d unaligned=%u cache/positions=PASS\n",
            rows, cases[c].keys, cases[c].fixture, input == unaligned);
    }
    ds4_gpu_gvq_free(fast); ds4_gpu_gvq_free(old);
    ds4_gpu_tensor_free(pos); ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(latent);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(unaligned); ds4_gpu_tensor_free(aligned); ds4_gpu_tensor_free(owner);
    free(control); free(actual); free(expected); free(query); free(after); free(cache);
}

int main(int argc, char **argv) {
    CHECK(argc == 2 || argc == 3);
    // References follow the expanded/latent adapters; flash has its own test.
    CHECK(!setenv("DS4_GLMVQ_FLASH", "0", 1));
    const uint64_t bytes = (uint64_t)HEADS * 2 * DIM * LATENT * sizeof(uint16_t);
    FILE *file = fopen(argv[1], "rb");
    CHECK(file && !fseek(file, 0, SEEK_END) && ftell(file) == (long)bytes);
    void *map = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    CHECK(map != MAP_FAILED && ds4_gpu_init());
    CHECK(ds4_gpu_model_source_bind(map, bytes, DS4_MSRC_ROLE_PRIMARY, -1,
        DS4_RESIDENCY_EAGER_DEVICE, "base", "GVQ KV fixture") >= 0);
    CHECK(ds4_gpu_set_model_map_range(map, bytes, 0, bytes, bytes));
    ds4_unit_tensor_in tensor = {0, bytes, DS4_TCAT_ROUTED_EXPERT, 1};
    ds4_unit_compile_params params = {0};
    params.max_unit_bytes = bytes; params.vmm_granularity = 256;
    params.device_promote = 1; params.promote_experts = 1;
    ds4_phys_unit unit;
    uint64_t populated = 0;
    CHECK(ds4_units_compile(&tensor, 1, &params, &unit) == 1);
    CHECK(!ds4_gpu_model_units_bind(map, &unit, 1));
    CHECK(ds4_gpu_materialize_model_plan(map, bytes, &populated));
    const ds4_gvq_record record = {0, bytes, BF16_TYPE, HEADS * 2 * DIM, LATENT,
        DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK};
    ds4_gvq_model *model = ds4_gpu_gvq_bind(map, bytes, &record, 1);
    CHECK(model);
    if (argc == 3) { check_latent(model, argv[2]); }
    else { check_paths(model); }
    ds4_gpu_gvq_unbind(model);
    ds4_gpu_unregister_model_map(map);
    ds4_gpu_cleanup();
    CHECK(!munmap(map, bytes) && !fclose(file));
    puts(argc == 3 ? "GLMVQ_LATENT_ADAPTER_PASS" : "GLMVQ_KV_HANDOFF_PASS");
    return 0;
}
