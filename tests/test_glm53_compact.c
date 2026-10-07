#include "ds4_gpu.h"
#include "ds4_glm53_compact.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr, label) do { \
    if (!(expr)) { fprintf(stderr, "FAIL: %s\n", label); return 0; } \
} while (0)

enum { Q8_BLOCK = 32, Q8_BYTES = 34, TEST_ROWS = 3, CACHE_ROWS = 12,
       TEST_HEADS = 2, POOL_ROWS = 13, POOL_CAP = 16 };

static float half_value(uint16_t bits) {
    const uint32_t sign = (uint32_t)(bits >> 15);
    const uint32_t exponent = (bits >> 10) & 31u;
    const uint32_t fraction = bits & 1023u;
    float value = exponent ? ldexpf(1.0f + (float)fraction / 1024.0f,
                                   (int)exponent - 15)
                           : ldexpf((float)fraction, -24);
    return sign ? -value : value;
}

static uint16_t bf16_value(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static int close_values(const float *got, const double *expected,
                        size_t count, float tolerance, const char *label) {
    float error = 0.0f;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "FAIL: %s nonfinite at %zu\n", label, i);
            return 0;
        }
        error = fmaxf(error, (float)fabs((double)got[i] - expected[i]));
    }
    printf("%s max_abs %.9g\n", label, error);
    return error <= tolerance;
}

static int weight_quant(uint32_t h, uint32_t row, uint32_t col,
                        uint32_t salt) {
    return (int)((h * 11u + row * 7u + col * 3u + salt) % 13u) - 6;
}

static void make_q8(unsigned char *dst, uint32_t rows, uint32_t cols,
                    uint32_t salt) {
    for (uint32_t h = 0; h < TEST_HEADS; h++) {
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t b = 0; b < cols / Q8_BLOCK; b++) {
                unsigned char *block = dst +
                    (((uint64_t)h * rows + r) * (cols / Q8_BLOCK) + b) * Q8_BYTES;
                const uint16_t scale = 0x2000u; /* Exactly 1/128. */
                memcpy(block, &scale, sizeof(scale));
                for (uint32_t d = 0; d < Q8_BLOCK; d++) {
                    ((int8_t *)(block + 2))[d] =
                        (int8_t)weight_quant(h, r, b * Q8_BLOCK + d, salt);
                }
            }
        }
    }
}

static int check_attention(uint32_t dim, uint32_t latent_dim) {
    const uint64_t weight_bytes = (uint64_t)TEST_HEADS * latent_dim *
                                 dim / Q8_BLOCK * Q8_BYTES;
    const uint64_t map_size = 2u * weight_bytes;
    unsigned char *map = NULL;
    CHECK(posix_memalign((void **)&map, 4096u, map_size) == 0, "compact map");
    make_q8(map, latent_dim, dim, 1u);
    make_q8(map + weight_bytes, dim, latent_dim, 5u);

    const size_t n_low = TEST_ROWS * TEST_HEADS * latent_dim;
    const size_t n_q = TEST_ROWS * TEST_HEADS * dim;
    float *latent = calloc(CACHE_ROWS * latent_dim, sizeof(float));
    float *q = calloc(n_q, sizeof(float));
    float *got = calloc(n_q, sizeof(float));
    float *low_got = calloc(n_low, sizeof(float));
    double *low_ref = calloc(n_low, sizeof(double));
    double *expected = calloc(n_q, sizeof(double));
    uint16_t *cache_bits = calloc(CACHE_ROWS * latent_dim, sizeof(uint16_t));
    CHECK(latent && q && got && low_got && low_ref && expected && cache_bits,
          "compact host buffers");
    for (uint32_t r = 0; r < CACHE_ROWS; r++) {
        for (uint32_t j = 0; j < latent_dim; j++) {
            /* All values are exactly representable in FP16; preserve row identity. */
            latent[r * latent_dim + j] =
                (float)((int)((r * 29u + j * 5u) % 31u) - 15) / 64.0f;
        }
    }
    for (size_t i = 0; i < n_q; i++) {
        q[i] = (float)((int)((i * 3u) % 23u) - 11) / 128.0f;
    }
    const uint32_t selected[TEST_ROWS * 5] = {
        0u, 2u, 4u, UINT32_MAX, 11u,
        1u, 3u, 5u, 6u, UINT32_MAX,
        2u, 4u, 6u, 7u, 8u
    };
    CHECK(ds4_gpu_set_model_map(map, map_size), "compact register map");
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(CACHE_ROWS * latent_dim * sizeof(float));
    ds4_gpu_tensor *query = ds4_gpu_tensor_alloc(n_q * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc(CACHE_ROWS * latent_dim * sizeof(uint16_t));
    ds4_gpu_tensor *low_q = ds4_gpu_tensor_alloc(n_low * sizeof(float));
    ds4_gpu_tensor *low_out = ds4_gpu_tensor_alloc(n_low * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_q * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(sizeof(selected));
    CHECK(x && query && cache && low_q && low_out && out && sel, "compact device buffers");
    CHECK(ds4_gpu_tensor_write(x, 0u, latent, CACHE_ROWS * latent_dim * sizeof(float)), "latent input");
    CHECK(ds4_gpu_tensor_write(query, 0u, q, n_q * sizeof(float)), "query input");
    CHECK(ds4_gpu_tensor_write(sel, 0u, selected, sizeof(selected)), "selection input");
    CHECK(ds4_gpu_glm53_store_low(cache, x, CACHE_ROWS, 0u, CACHE_ROWS, latent_dim), "store compact latent");
    CHECK(ds4_gpu_tensor_read(cache, 0u, cache_bits, CACHE_ROWS * latent_dim * sizeof(uint16_t)), "read compact cache");
    for (size_t i = 0; i < CACHE_ROWS * latent_dim; i++) {
        CHECK(half_value(cache_bits[i]) == latent[i], "FP16 row storage");
    }
    CHECK(ds4_gpu_glm53_absorb_q(low_q, query, map, map_size, 0u,
                               TEST_ROWS, TEST_HEADS, latent_dim, dim), "absorb compact Q");
    CHECK(ds4_gpu_tensor_read(low_q, 0u, low_got, n_low * sizeof(float)), "read absorbed Q");
    for (uint32_t t = 0; t < TEST_ROWS; t++) {
        for (uint32_t h = 0; h < TEST_HEADS; h++) {
            for (uint32_t j = 0; j < latent_dim; j++) {
                double value = 0.0;
                for (uint32_t d = 0; d < dim; d++) {
                    value += (double)q[(t * TEST_HEADS + h) * dim + d] *
                             weight_quant(h, j, d, 1u) / 128.0;
                }
                low_ref[(t * TEST_HEADS + h) * latent_dim + j] = value;
            }
        }
    }
    CHECK(close_values(low_got, low_ref, n_low, 2.0e-5f, "absorbed Q"), "absorbed CPU reference");

    for (uint32_t mode = 0; mode < 2u; mode++) {
        const uint32_t pos0 = 5u;
        CHECK(ds4_gpu_glm53_attn_low(low_out, low_q, cache, mode ? NULL : sel,
                                   mode ? 0u : 5u, TEST_ROWS, pos0,
                                   CACHE_ROWS, TEST_HEADS, latent_dim, dim), "compact attention");
        CHECK(ds4_gpu_glm53_proj_v(out, low_out, map, map_size, weight_bytes,
                                 TEST_ROWS, TEST_HEADS, latent_dim, dim), "compact V projection");
        CHECK(ds4_gpu_tensor_read(out, 0u, got, n_q * sizeof(float)), "read compact heads");
        memset(expected, 0, n_q * sizeof(double));
        /* Expand K and V independently, then apply conventional attention. */
        for (uint32_t t = 0; t < TEST_ROWS; t++) {
            const uint32_t visible = pos0 + t + 1u;
            for (uint32_t h = 0; h < TEST_HEADS; h++) {
                double score[CACHE_ROWS], maximum = -INFINITY, denom = 0.0;
                for (uint32_t r = 0; r < CACHE_ROWS; r++) {
                    int included = mode ? r < visible : 0;
                    for (uint32_t s = 0; !mode && s < 5u; s++) {
                        included |= selected[t * 5u + s] == r && r < visible;
                    }
                    score[r] = -INFINITY;
                    if (!included) { continue; }
                    double dot = 0.0;
                    for (uint32_t d = 0; d < dim; d++) {
                        double key = 0.0;
                        for (uint32_t j = 0; j < latent_dim; j++) {
                            key += half_value(cache_bits[r * latent_dim + j]) *
                                   weight_quant(h, j, d, 1u) / 128.0;
                        }
                        dot += q[(t * TEST_HEADS + h) * dim + d] * key;
                    }
                    score[r] = dot / sqrt((double)dim);
                    maximum = fmax(maximum, score[r]);
                }
                for (uint32_t r = 0; r < CACHE_ROWS; r++) {
                    score[r] = isfinite(score[r]) ? exp(score[r] - maximum) : 0.0;
                    denom += score[r];
                }
                for (uint32_t d = 0; d < dim; d++) {
                    double value = 0.0;
                    for (uint32_t r = 0; r < CACHE_ROWS; r++) {
                        double expanded = 0.0;
                        for (uint32_t j = 0; j < latent_dim; j++) {
                            expanded += half_value(cache_bits[r * latent_dim + j]) *
                                        weight_quant(h, d, j, 5u) / 128.0;
                        }
                        value += score[r] / denom * expanded;
                    }
                    expected[(t * TEST_HEADS + h) * dim + d] = value;
                }
            }
        }
        CHECK(close_values(got, expected, n_q, 3.0e-5f,
                           mode ? "dense expanded attention" : "selected expanded attention"),
              "expanded attention parity");
    }
    CHECK(!ds4_gpu_glm53_store_low(cache, x, 2u, CACHE_ROWS - 1u, CACHE_ROWS, latent_dim), "reject cache overrun");
    CHECK(!ds4_gpu_glm53_absorb_q(low_q, query, map, map_size - 1u,
                                weight_bytes, TEST_ROWS, TEST_HEADS, latent_dim, dim), "reject weight overrun");
    CHECK(!ds4_gpu_glm53_proj_v(out, low_out, map, map_size, weight_bytes,
                              TEST_ROWS + 1u, TEST_HEADS, latent_dim, dim), "reject projection buffers");
    ds4_gpu_tensor_free(sel); ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low_out); ds4_gpu_tensor_free(low_q);
    ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(query); ds4_gpu_tensor_free(x);
    ds4_gpu_unregister_model_map(map);
    free(cache_bits); free(expected); free(low_ref); free(low_got);
    free(got); free(q); free(latent); free(map);
    return 1;
}

static int check_pool(void) {
    const uint32_t width = DS4_GLM53_POOL_DIM;
    enum { MAP_BYTES = 4096 };
    unsigned char *map = NULL;
    CHECK(posix_memalign((void **)&map, MAP_BYTES, MAP_BYTES) == 0, "pool map");
    memset(map, 0, MAP_BYTES);
    float *norm = (float *)map;
    float *bias = norm + width;
    uint16_t *ape = (uint16_t *)(bias + width);
    for (uint32_t d = 0; d < width; d++) {
        norm[d] = 0.75f + (float)(d % 7u) / 32.0f;
        bias[d] = (float)((int)(d % 5u) - 2) / 16.0f;
        for (uint32_t r = 0; r < DS4_GLM53_POOL_SIZE; r++) {
            ape[r * width + d] = bf16_value((float)((int)((d + 3u * r) % 9u) - 4) / 8.0f);
        }
    }
    float raw[POOL_ROWS * DS4_GLM53_POOL_DIM];
    float gates[POOL_ROWS * DS4_GLM53_POOL_DIM];
    for (uint32_t r = 0; r < POOL_ROWS; r++) {
        for (uint32_t d = 0; d < width; d++) {
            raw[r * width + d] = (float)((int)((r * 17u + d * 3u) % 29u) - 14) / 16.0f;
            gates[r * width + d] = (float)((int)((r * 7u + d) % 11u) - 5) / 4.0f;
        }
    }
    CHECK(ds4_gpu_set_model_map(map, MAP_BYTES), "pool register map");
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(sizeof(raw));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(sizeof(gates));
    ds4_gpu_tensor *tail_k = ds4_gpu_tensor_alloc(4u * width * sizeof(float));
    ds4_gpu_tensor *tail_gate = ds4_gpu_tensor_alloc(4u * width * sizeof(float));
    ds4_gpu_tensor *pool = ds4_gpu_tensor_alloc(4u * width * sizeof(uint16_t));
    CHECK(input && gate && tail_k && tail_gate && pool, "pool tensors");
    const uint32_t chunks[] = {3u, 2u, 7u, 1u};
    uint32_t pos = 0u;
    for (uint32_t c = 0; c < sizeof(chunks) / sizeof(chunks[0]); c++) {
        const uint32_t rows = chunks[c];
        CHECK(ds4_gpu_tensor_write(input, 0u, raw + pos * width, rows * width * sizeof(float)), "pool raw rows");
        CHECK(ds4_gpu_tensor_write(gate, 0u, gates + pos * width, rows * width * sizeof(float)), "pool gate rows");
        CHECK(ds4_gpu_glm53_pool_key(pool, tail_k, tail_gate, input, gate,
                                    map, MAP_BYTES, 0u, width * sizeof(float),
                                    2u * width * sizeof(float), pos, rows, POOL_CAP,
                                    1.0e-6f), "pool boundary update");
        pos += rows;
    }
    uint16_t got[4 * DS4_GLM53_POOL_DIM];
    CHECK(ds4_gpu_tensor_read(pool, 0u, got, sizeof(got)), "read pools");
    for (uint32_t p = 0; p < POOL_ROWS / 4u; p++) {
        double mean[4] = {0}, inv[4], normalized[4 * DS4_GLM53_POOL_DIM];
        for (uint32_t r = 0; r < 4u; r++) {
            for (uint32_t d = 0; d < width; d++) { mean[r] += raw[(p * 4u + r) * width + d]; }
            mean[r] /= width;
            double variance = 0.0;
            for (uint32_t d = 0; d < width; d++) {
                const double delta = raw[(p * 4u + r) * width + d] - mean[r];
                variance += delta * delta;
            }
            inv[r] = 1.0 / sqrt(variance / width + 1.0e-6);
            for (uint32_t d = 0; d < width; d++) {
                normalized[r * width + d] = (raw[(p * 4u + r) * width + d] - mean[r]) * inv[r] * norm[d] + bias[d];
            }
        }
        for (uint32_t d = 0; d < width; d++) {
            double sum = 0.0, denom = 0.0;
            for (uint32_t r = 0; r < 4u; r++) {
                uint32_t ape_bits = (uint32_t)ape[r * width + d] << 16;
                float ape_f;
                memcpy(&ape_f, &ape_bits, sizeof(ape_f));
                const double weight = exp((double)gates[(p * 4u + r) * width + d] + ape_f);
                sum += weight * normalized[r * width + d];
                denom += weight;
            }
            CHECK(fabs(half_value(got[p * width + d]) - sum / denom) < 1.2e-3,
                  "independent pool LayerNorm/APE/softmax");
        }
    }
    float tail[4 * DS4_GLM53_POOL_DIM];
    CHECK(ds4_gpu_tensor_read(tail_k, 0u, tail, sizeof(tail)), "read pool tail");
    for (uint32_t d = 0; d < width; d++) { CHECK(tail[d] == raw[12u * width + d], "pos modulo4 tail state"); }

    enum { SCORE_ROWS = 3, POOLS = 4, SCORE_HEADS = 2 };
    float query[SCORE_ROWS * SCORE_HEADS * DS4_GLM53_POOL_DIM];
    float weights[SCORE_ROWS * SCORE_HEADS];
    for (size_t i = 0; i < sizeof(query) / sizeof(query[0]); i++) { query[i] = (float)((int)(i % 13u) - 6) / 32.0f; }
    for (size_t i = 0; i < sizeof(weights) / sizeof(weights[0]); i++) { weights[i] = (float)(i + 1u) / 8.0f; }
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(sizeof(query));
    ds4_gpu_tensor *w = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(SCORE_ROWS * POOLS * sizeof(float));
    CHECK(q && w && scores, "score tensors");
    CHECK(ds4_gpu_tensor_write(q, 0, query, sizeof(query)), "score queries");
    CHECK(ds4_gpu_tensor_write(w, 0, weights, sizeof(weights)), "score weights");
    CHECK(ds4_gpu_glm53_pool_score(scores, q, w, pool, POOLS,
                                  SCORE_ROWS, 6u, SCORE_HEADS, 0.125f), "pooled scores");
    float score_got[SCORE_ROWS * POOLS];
    CHECK(ds4_gpu_tensor_read(scores, 0, score_got, sizeof(score_got)), "read pooled scores");
    for (uint32_t t = 0; t < SCORE_ROWS; t++) {
        for (uint32_t p = 0; p < POOLS; p++) {
            if (p >= (6u + t + 1u) / 4u) {
                CHECK(isinf(score_got[t * POOLS + p]) && score_got[t * POOLS + p] < 0.0f, "causal pooled mask");
                continue;
            }
            double value = 0.0;
            for (uint32_t h = 0; h < SCORE_HEADS; h++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < width; d++) { dot += query[(t * SCORE_HEADS + h) * width + d] * half_value(got[p * width + d]); }
                value += fmax(dot, 0.0) * weights[t * SCORE_HEADS + h] * 0.125;
            }
            CHECK(fabs(score_got[t * POOLS + p] - value) < 2.0e-6, "pooled score reference");
        }
    }
    enum { SELECT_ROWS = 4, SELECT_POOLS = 2, INDEX_TOPK = 8, OUT_WIDTH = 11 };
    const uint32_t pools[SELECT_ROWS * SELECT_POOLS] = {1u, 0u, 0u, 1u, UINT32_MAX, 1u, 1u, 0u};
    ds4_gpu_tensor *picked = ds4_gpu_tensor_alloc(sizeof(pools));
    ds4_gpu_tensor *expanded = ds4_gpu_tensor_alloc(SELECT_ROWS * OUT_WIDTH * sizeof(uint32_t));
    CHECK(picked && expanded, "selection tensors");
    CHECK(ds4_gpu_tensor_write(picked, 0, pools, sizeof(pools)), "selected pools");
    CHECK(ds4_gpu_glm53_pool_expand(expanded, picked, SELECT_ROWS, 8u,
                                   SELECT_POOLS, INDEX_TOPK, OUT_WIDTH), "expanded pooled selection");
    uint32_t expanded_got[SELECT_ROWS * OUT_WIDTH];
    CHECK(ds4_gpu_tensor_read(expanded, 0, expanded_got, sizeof(expanded_got)), "read expanded selection");
    for (uint32_t t = 0; t < SELECT_ROWS; t++) {
        const uint32_t visible = 8u + t + 1u;
        for (uint32_t s = 0; s < OUT_WIDTH; s++) {
            uint32_t expected = UINT32_MAX;
            if (s < INDEX_TOPK) {
                const uint32_t p = pools[t * SELECT_POOLS + s / 4u];
                if (p < visible / 4u) { expected = p * 4u + s % 4u; }
            } else if (s - INDEX_TOPK < visible % 4u) {
                expected = visible - visible % 4u + s - INDEX_TOPK;
            }
            CHECK(expanded_got[t * OUT_WIDTH + s] == expected, "selected pools plus partial tail");
        }
    }
    CHECK(!ds4_gpu_glm53_pool_key(pool, tail_k, tail_gate, input, gate,
                                map, MAP_BYTES, 0u, width * sizeof(float), MAP_BYTES - 1u,
                                0u, 1u, POOL_CAP, 1.0e-6f), "reject APE overrun");
    ds4_gpu_tensor_free(expanded); ds4_gpu_tensor_free(picked);
    ds4_gpu_tensor_free(scores); ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(pool); ds4_gpu_tensor_free(tail_gate); ds4_gpu_tensor_free(tail_k);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(input);
    ds4_gpu_unregister_model_map(map);
    free(map);
    return 1;
}

int main(void) {
    if (!ds4_gpu_init()) { return 1; }
    if (!check_attention(32u, 64u) || !check_attention(256u, 512u) || !check_pool()) { return 1; }
    puts("GLM-5.3 compact DSA: valid");
    return 0;
}
