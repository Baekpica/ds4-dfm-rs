/* Native indexer binding, causal top-k expansion and masked DSA handoff.
 * Fixture arithmetic is checked independently before this adapter test. */
#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_glmvq.h"
#include "ds4_glm53_compact.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

enum { KEYS = 8192, WIDTH = 128, RATIO = 4, INDEX_HEADS = 32, ROWS = 32,
       POOLS = KEYS / RATIO, TOP = 512, STRIDE = DS4_GLM53_MAX_SELECTED,
       HEADS = 64, DIM = 256, LATENT = 512, BF16_TYPE = 30, MAGIC = 0x53505651 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GVQ sparse FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static void *read_block(FILE *f, size_t bytes) {
    void *p = malloc(bytes);
    CHECK(p && fread(p, bytes, 1, f) == 1);
    return p;
}

static ds4_gpu_tensor *upload(const void *p, uint64_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t && ds4_gpu_tensor_write(t, 0, p, bytes));
    return t;
}

static float unpack(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void check_index(ds4_gvq_ctx *ctx, FILE *file) {
    const size_t xb = (size_t)KEYS * WIDTH * sizeof(float), pb = POOLS * WIDTH * sizeof(uint16_t);
    const size_t qb = ROWS * INDEX_HEADS * WIDTH * sizeof(float), wb = ROWS * INDEX_HEADS * sizeof(float);
    float *raw = read_block(file, xb), *gate = read_block(file, xb);
    float *expected = read_block(file, pb * 2), *bound = read_block(file, pb * 2);
    float *q = read_block(file, qb), *weights = read_block(file, wb);
    uint16_t *actual = calloc(1, pb);
    CHECK(actual);
    ds4_gpu_tensor *x = upload(raw, xb), *g = upload(gate, xb), *cache = upload(actual, pb);
    float tail[RATIO * WIDTH] = {0};
    ds4_gpu_tensor *tk = upload(tail, sizeof(tail)), *tg = upload(tail, sizeof(tail));
    const uint32_t chunks[] = {1, 2, 2048, 2048, 2048, 2044, 1};
    uint32_t first = 0;
    for (unsigned i = 0; i < sizeof(chunks) / sizeof(chunks[0]); i++) {
        const uint64_t offset = (uint64_t)first * WIDTH * sizeof(float);
        const uint64_t bytes = (uint64_t)chunks[i] * WIDTH * sizeof(float);
        ds4_gpu_tensor *vx = ds4_gpu_tensor_view(x, offset, bytes), *vg = ds4_gpu_tensor_view(g, offset, bytes);
        CHECK(vx && vg && ds4_gpu_gvq_pool(ctx, cache, tk, tg, vx, vg, 1, 2, 3, first, chunks[i], KEYS));
        first += chunks[i];
        if (first == KEYS - 1) {
            CHECK(ds4_gpu_tensor_read(tk, 0, tail, sizeof(tail)));
            CHECK(!memcmp(tail, raw + (KEYS - RATIO) * WIDTH, (RATIO - 1) * WIDTH * sizeof(float)));
            CHECK(ds4_gpu_tensor_read(tg, 0, tail, sizeof(tail)));
            CHECK(!memcmp(tail, gate + (KEYS - RATIO) * WIDTH, (RATIO - 1) * WIDTH * sizeof(float)));
        }
        ds4_gpu_tensor_free(vg); ds4_gpu_tensor_free(vx);
    }
    CHECK(first == KEYS && ds4_gpu_tensor_read(cache, 0, actual, pb));
    for (unsigned i = 0; i < POOLS * WIDTH; i++) {
        CHECK(isfinite(unpack(actual[i])) && fabsf(unpack(actual[i]) - expected[i]) <= bound[i]);
    }
    ds4_gpu_tensor *query = upload(q, qb), *w = upload(weights, wb);
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(ROWS * POOLS * sizeof(float));
    ds4_gpu_tensor *pools = ds4_gpu_tensor_alloc(ROWS * TOP * sizeof(uint32_t));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(ROWS * STRIDE * sizeof(uint32_t));
    CHECK(scores && pools && selected);
    CHECK(ds4_gpu_gvq_index(ctx, scores, query, w, cache, POOLS, ROWS, KEYS - ROWS));
    CHECK(ds4_gpu_indexer_topk_tensor(pools, scores, POOLS, ROWS, TOP, 0, UINT32_MAX));
    CHECK(ds4_gpu_glm53_pool_expand(selected, pools, ROWS, KEYS - ROWS, TOP, TOP * RATIO, STRIDE));
    float *s = malloc(ROWS * POOLS * sizeof(float));
    uint32_t *ids = malloc(ROWS * TOP * sizeof(uint32_t)), *tokens = malloc(ROWS * STRIDE * sizeof(uint32_t));
    CHECK(s && ids && tokens && ds4_gpu_tensor_read(scores, 0, s, ROWS * POOLS * sizeof(float)) &&
        ds4_gpu_tensor_read(pools, 0, ids, ROWS * TOP * sizeof(uint32_t)) &&
        ds4_gpu_tensor_read(selected, 0, tokens, ROWS * STRIDE * sizeof(uint32_t)));
    const double qs = 1.0f / sqrtf(WIDTH), gamma = 192.0 * 0x1p-24 / (1 - 192.0 * 0x1p-24);
    for (unsigned r = 0; r < ROWS; r++) {
        const uint32_t visible = KEYS - ROWS + r + 1;
        unsigned char seen[POOLS] = {0};
        float floor = FLT_MAX;
        for (unsigned k = 0; k < TOP; k++) {
            const uint32_t pool = ids[r * TOP + k];
            CHECK(pool < visible / RATIO && !seen[pool]);
            seen[pool] = 1;
            floor = fminf(floor, s[r * POOLS + pool]);
            for (unsigned d = 0; d < RATIO; d++) { CHECK(tokens[r * STRIDE + k * RATIO + d] == pool * RATIO + d); }
        }
        for (unsigned k = 0; k < POOLS; k++) {
            CHECK(isfinite(s[r * POOLS + k]));
            if (k >= visible / RATIO) { CHECK(s[r * POOLS + k] == -FLT_MAX); }
            if (!seen[k]) { CHECK(s[r * POOLS + k] <= floor); }
        }
        for (unsigned k = 0; k < RATIO - 1; k++) {
            CHECK(tokens[r * STRIDE + TOP * RATIO + k] ==
                (k < visible % RATIO ? visible - visible % RATIO + k : UINT32_MAX));
        }
        // Sample score rows/channels independently; the full arithmetic gate
        // already covers the kernel, while this checks adapter layout/binding.
        for (unsigned k = 0; k < visible / RATIO; k += 509) {
            double sum = 0, magnitude = 0;
            for (unsigned h = 0; h < INDEX_HEADS; h++) {
                double dot = 0, absolute = 0;
                for (unsigned d = 0; d < WIDTH; d++) {
                    const double product = q[(r * INDEX_HEADS + h) * WIDTH + d] * (double)unpack(actual[k * WIDTH + d]);
                    dot += product; absolute += fabs(product);
                }
                const float scale = weights[r * INDEX_HEADS + h] * (1.0f / sqrtf(INDEX_HEADS));
                sum += fmax(dot * qs, 0) * scale;
                magnitude += absolute * qs * fabs(scale);
            }
            CHECK(fabs(s[r * POOLS + k] - sum) <= gamma * magnitude + FLT_MIN);
        }
    }
    CHECK(!ds4_gpu_gvq_pool(ctx, cache, tk, tg, x, g, 1, 2, 3, KEYS, 1, KEYS));
    CHECK(!ds4_gpu_gvq_index(ctx, scores, query, w, cache, POOLS - 1, ROWS, KEYS - ROWS));
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(pools); ds4_gpu_tensor_free(scores);
    ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(query); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tk);
    ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(g); ds4_gpu_tensor_free(x);
    free(tokens); free(ids); free(s); free(actual); free(weights); free(q);
    free(bound); free(expected); free(gate); free(raw);
    puts("GVQ index binding, tail, causal top512/expansion PASS");
}

static void check_attention(ds4_gvq_ctx *ctx, FILE *file) {
    const uint64_t cb = (uint64_t)KEYS * LATENT * sizeof(uint16_t), rb = HEADS * DIM * sizeof(float);
    uint16_t *cache = read_block(file, cb), *after = malloc(cb);
    CHECK(after);
    ds4_gpu_tensor *c = upload(cache, cb), *latent = ds4_gpu_tensor_alloc(cb * 2);
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)KEYS * HEADS * 2 * DIM * sizeof(float));
    ds4_gpu_tensor *pos = ds4_gpu_tensor_alloc(2 * sizeof(uint32_t));
    CHECK(latent && kv && pos);
    for (unsigned test = 0; test < 3; test++) {
        uint32_t header[2];
        CHECK(fread(header, sizeof(header), 1, file) == 1);
        const uint32_t rows = header[0], keys = header[1];
        CHECK((rows == 1 && (keys == 2049 || keys == KEYS)) || (rows == 2 && keys == KEYS));
        const uint64_t qb = rows * rb, sb = (uint64_t)rows * STRIDE * sizeof(int32_t);
        float *query = read_block(file, qb), *expected = read_block(file, qb), *actual = malloc(qb);
        int32_t *selection = read_block(file, sb);
        CHECK(actual);
        ds4_gpu_tensor *q = upload(query, qb), *selected = upload(selection, sb), *out = ds4_gpu_tensor_alloc(qb);
        CHECK(out && ds4_gpu_gvq_dsa(ctx, out, q, c, 0, latent, kv, pos,
            keys - rows, rows, HEADS, DIM, selected, STRIDE));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, qb));
        for (uint64_t i = 0; i < qb / sizeof(float); i++) { CHECK(isfinite(actual[i]) && actual[i] == expected[i]); }
        uint32_t positions[2];
        CHECK(ds4_gpu_tensor_read(pos, 0, positions, rows * sizeof(uint32_t)));
        for (unsigned r = 0; r < rows; r++) { CHECK(positions[r] == keys - rows + r); }
        CHECK(ds4_gpu_tensor_read(c, 0, after, cb) && !memcmp(cache, after, cb));
        CHECK(!ds4_gpu_gvq_dsa(ctx, out, q, c, 0, latent, kv, pos, keys - rows, rows, HEADS, DIM, selected, STRIDE + 1));
        printf("GVQ masked DSA keys=%u rows=%u cache/positions PASS\n", keys, rows);
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(q);
        free(selection); free(actual); free(expected); free(query);
    }
    ds4_gpu_tensor_free(pos); ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(latent); ds4_gpu_tensor_free(c);
    free(after); free(cache);
}

int main(int argc, char **argv) {
    CHECK(argc == 3);
    const uint64_t kb = (uint64_t)HEADS * 2 * DIM * LATENT * sizeof(uint16_t);
    const uint64_t bytes = kb + (2 + RATIO) * WIDTH * sizeof(uint16_t);
    FILE *weights = fopen(argv[1], "rb"), *fixture = fopen(argv[2], "rb");
    uint32_t header[4];
    CHECK(weights && fixture && !fseek(weights, 0, SEEK_END) && ftell(weights) == (long)bytes &&
        fread(header, sizeof(header), 1, fixture) == 1 && header[0] == MAGIC &&
        header[1] == KEYS && header[2] == ROWS && header[3] == STRIDE);
    void *map = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fileno(weights), 0);
    CHECK(map != MAP_FAILED && ds4_gpu_init());
    CHECK(ds4_gpu_model_source_bind(map, bytes, DS4_MSRC_ROLE_PRIMARY, -1,
        DS4_RESIDENCY_EAGER_DEVICE, "base", "GVQ sparse fixture") >= 0);
    CHECK(ds4_gpu_set_model_map_range(map, bytes, 0, bytes, bytes));
    ds4_unit_tensor_in tensor = {0, bytes, DS4_TCAT_ROUTED_EXPERT, 1};
    ds4_unit_compile_params params = {0};
    params.max_unit_bytes = bytes; params.vmm_granularity = 256;
    params.device_promote = 1; params.promote_experts = 1;
    ds4_phys_unit unit;
    uint64_t populated = 0;
    CHECK(ds4_units_compile(&tensor, 1, &params, &unit) == 1 && !ds4_gpu_model_units_bind(map, &unit, 1));
    CHECK(ds4_gpu_materialize_model_plan(map, bytes, &populated));
    const ds4_gvq_record records[] = {
        {0, kb, BF16_TYPE, HEADS * 2 * DIM, LATENT, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK},
        {kb, WIDTH * 2, BF16_TYPE, WIDTH, 1, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK},
        {kb + WIDTH * 2, WIDTH * 2, BF16_TYPE, WIDTH, 1, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK},
        {kb + WIDTH * 4, RATIO * WIDTH * 2, BF16_TYPE, RATIO, WIDTH, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK}
    };
    ds4_gvq_model *model = ds4_gpu_gvq_bind(map, bytes, records, 4);
    CHECK(model && !setenv("DS4_GLMVQ_MMA", "0", 1) && !setenv("DS4_GLMVQ_DECODE", "0", 1) &&
        !setenv("DS4_GLMVQ_KV_BF16", "1", 1) && !setenv("DS4_GLMVQ_LATENT_ATTN", "1", 1));
    uint64_t scratch = ds4_gpu_gvq_attn_bytes(2, KEYS, HEADS, DIM);
    const uint64_t index = ds4_gpu_gvq_index_bytes(ROWS, POOLS);
    if (index > scratch) { scratch = index; }
    ds4_gvq_ctx *ctx = ds4_gpu_gvq_create(model, scratch, ROWS);
    CHECK(ctx);
    check_index(ctx, fixture);
    check_attention(ctx, fixture);
    CHECK(fgetc(fixture) == EOF && !fclose(fixture));
    ds4_gpu_gvq_free(ctx); ds4_gpu_gvq_unbind(model);
    ds4_gpu_unregister_model_map(map); ds4_gpu_cleanup();
    CHECK(!munmap(map, bytes) && !fclose(weights));
    puts("GLMVQ_SPARSE_ADAPTER_PASS");
    return 0;
}
