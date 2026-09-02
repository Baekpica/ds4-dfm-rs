/* CUDA parity for Qwen3.8-Flash-Next MoE semantics: softmax top-10 routing,
 * sigmoid-gated shared expert output, and the recipe's Q6_K[512] +
 * Q5_0[128] expert-down split without an assignment-by-hidden tail scratch. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ROUTER_ROWS = 4,
    ROUTER_EXPERTS = 512,
    ROUTER_USED = 10,
    HIDDEN = 2560,
    SHARED_ROWS = 3,
    DOWN_ASSIGNMENTS = 20,
    DOWN_EXPERTS = 8,
    DOWN_MID = 640,
    DOWN_MAIN = 512,
    DOWN_TAIL = 128,
    QK_K = 256,
    QK_5_0 = 32,
};

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
} test_block_q6_k;

typedef struct {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[QK_5_0 / 2];
} test_block_q5_0;

typedef struct {
    uint16_t d;
    int8_t qs[QK_5_0];
} test_block_q8_0;

_Static_assert(sizeof(test_block_q6_k) == 210, "Q6_K block layout");
_Static_assert(sizeof(test_block_q5_0) == 22, "Q5_0 block layout");
_Static_assert(sizeof(test_block_q8_0) == 34, "Q8_0 block layout");

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint32_t next_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16u;
    const uint32_t exp = (h >> 10u) & 0x1fu;
    const uint32_t frac = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0u) {
        if (frac == 0u) {
            bits = sign;
        } else {
            uint32_t f = frac;
            uint32_t shift = 0u;
            while ((f & 0x0400u) == 0u) {
                f <<= 1u;
                shift++;
            }
            f &= 0x03ffu;
            bits = sign | ((127u - 15u - shift) << 23u) | (f << 13u);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (frac << 13u);
    } else {
        bits = sign | ((exp + 112u) << 23u) | (frac << 13u);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float sigmoid_ref(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x))
                     : expf(x) / (1.0f + expf(x));
}

static void compare_f32(const char *name, const float *got, const float *want,
                        uint64_t count, float atol, float rtol) {
    double worst_ratio = 0.0;
    float worst_abs = 0.0f;
    uint64_t worst_i = 0u;
    for (uint64_t i = 0; i < count; i++) {
        const float abs_error = fabsf(got[i] - want[i]);
        const float limit = atol + rtol * fabsf(want[i]);
        const double ratio = limit > 0.0f
            ? (double)abs_error / limit
            : (abs_error == 0.0f ? 0.0 : INFINITY);
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            worst_abs = abs_error;
            worst_i = i;
        }
    }
    if (worst_ratio > 1.0) {
        fprintf(stderr,
                "FAIL: %s at %llu got %.9g want %.9g abs %.9g (%.3fx limit)\n",
                name, (unsigned long long)worst_i, got[worst_i], want[worst_i],
                worst_abs, worst_ratio);
        exit(1);
    }
    printf("%-45s pass (worst abs %.3g, %.3fx limit)\n",
           name, worst_abs, worst_ratio);
}

static void test_softmax_topk_router(void) {
    const uint64_t logits_count = (uint64_t)ROUTER_ROWS * ROUTER_EXPERTS;
    const uint64_t selected_count = (uint64_t)ROUTER_ROWS * ROUTER_USED;
    float *logits = (float *)malloc(logits_count * sizeof(*logits));
    int32_t *ids_want = (int32_t *)malloc(selected_count * sizeof(*ids_want));
    int32_t *ids_got = (int32_t *)malloc(selected_count * sizeof(*ids_got));
    float *weights_want = (float *)malloc(selected_count * sizeof(*weights_want));
    float *weights_got = (float *)malloc(selected_count * sizeof(*weights_got));
    float *scratch = (float *)malloc(ROUTER_EXPERTS * sizeof(*scratch));
    REQUIRE(logits && ids_want && ids_got && weights_want && weights_got &&
            scratch, "router host allocation");

    for (uint32_t row = 0; row < ROUTER_ROWS; row++) {
        for (uint32_t expert = 0; expert < ROUTER_EXPERTS; expert++) {
            logits[(uint64_t)row * ROUTER_EXPERTS + expert] =
                1.7f * sinf((float)(expert + 3u * row + 1u) * 0.031f) +
                0.4f * cosf((float)(expert + 11u) * 0.013f) +
                (float)((int)(expert % 19u) - 9) * 0.007f;
        }
        /* Exercise the explicit deterministic tie policy at the top. */
        logits[(uint64_t)row * ROUTER_EXPERTS + 17u + row] = 9.0f;
        logits[(uint64_t)row * ROUTER_EXPERTS + 41u + row] = 9.0f;

        memcpy(scratch, logits + (uint64_t)row * ROUTER_EXPERTS,
               ROUTER_EXPERTS * sizeof(*scratch));
        float max_selected = -INFINITY;
        for (uint32_t slot = 0; slot < ROUTER_USED; slot++) {
            int32_t best = -1;
            float best_value = -INFINITY;
            for (uint32_t expert = 0; expert < ROUTER_EXPERTS; expert++) {
                if (scratch[expert] > best_value) {
                    best_value = scratch[expert];
                    best = (int32_t)expert;
                }
            }
            ids_want[(uint64_t)row * ROUTER_USED + slot] = best;
            weights_want[(uint64_t)row * ROUTER_USED + slot] = best_value;
            if (best_value > max_selected) max_selected = best_value;
            scratch[(uint32_t)best] = -INFINITY;
        }
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < ROUTER_USED; slot++) {
            const uint64_t at = (uint64_t)row * ROUTER_USED + slot;
            weights_want[at] = expf(weights_want[at] - max_selected);
            sum += weights_want[at];
        }
        for (uint32_t slot = 0; slot < ROUTER_USED; slot++)
            weights_want[(uint64_t)row * ROUTER_USED + slot] /= sum;
    }

    ds4_gpu_tensor *d_logits = ds4_gpu_tensor_alloc(
        logits_count * sizeof(float));
    ds4_gpu_tensor *d_ids = ds4_gpu_tensor_alloc(
        selected_count * sizeof(int32_t));
    ds4_gpu_tensor *d_weights = ds4_gpu_tensor_alloc(
        selected_count * sizeof(float));
    REQUIRE(d_logits && d_ids && d_weights, "router GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_logits, 0, logits,
                                 logits_count * sizeof(float)),
            "router logits upload");
    REQUIRE(ds4_gpu_qwen4exp_softmax_topk_router_tensor(
                d_ids, d_weights, d_logits, ROUTER_EXPERTS, ROUTER_USED,
                ROUTER_ROWS),
            "Qwen softmax top-k launch");
    REQUIRE(ds4_gpu_tensor_read(d_ids, 0, ids_got,
                                selected_count * sizeof(int32_t)),
            "router ids download");
    REQUIRE(ds4_gpu_tensor_read(d_weights, 0, weights_got,
                                selected_count * sizeof(float)),
            "router weights download");
    for (uint64_t i = 0; i < selected_count; i++)
        REQUIRE(ids_got[i] == ids_want[i], "router selected expert exact");
    compare_f32("Qwen softmax top-10 normalized weights", weights_got,
                weights_want, selected_count, 2.0e-6f, 2.0e-6f);
    for (uint32_t row = 0; row < ROUTER_ROWS; row++) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < ROUTER_USED; slot++)
            sum += weights_got[(uint64_t)row * ROUTER_USED + slot];
        REQUIRE(fabsf(sum - 1.0f) <= 2.0e-6f,
                "router selected weights sum to one");

        int32_t ids_one[ROUTER_USED];
        float weights_one[ROUTER_USED];
        ds4_gpu_tensor *logits_one = ds4_gpu_tensor_view(
            d_logits, (uint64_t)row * ROUTER_EXPERTS * sizeof(float),
            ROUTER_EXPERTS * sizeof(float));
        REQUIRE(logits_one &&
                ds4_gpu_qwen4exp_softmax_topk_router_tensor(
                    d_ids, d_weights, logits_one,
                    ROUTER_EXPERTS, ROUTER_USED, 1u) &&
                ds4_gpu_tensor_read(
                    d_ids, 0u, ids_one, sizeof(ids_one)) &&
                ds4_gpu_tensor_read(
                    d_weights, 0u, weights_one, sizeof(weights_one)),
                "router scalar-row parity launch");
        ds4_gpu_tensor_free(logits_one);
        REQUIRE(memcmp(
                    ids_one,
                    ids_got + (uint64_t)row * ROUTER_USED,
                    sizeof(ids_one)) == 0 &&
                memcmp(
                    weights_one,
                    weights_got + (uint64_t)row * ROUTER_USED,
                    sizeof(weights_one)) == 0,
                "router scalar-row bit parity");
    }
    puts("Qwen softmax top-10 scalar/multi-row parity  bit-exact");
    REQUIRE(!ds4_gpu_qwen4exp_softmax_topk_router_tensor(
                d_ids, d_weights, d_logits, ROUTER_EXPERTS, ROUTER_USED, 0u),
            "router rejects zero tokens");

    ds4_gpu_tensor_free(d_weights);
    ds4_gpu_tensor_free(d_ids);
    ds4_gpu_tensor_free(d_logits);
    free(scratch);
    free(weights_got);
    free(weights_want);
    free(ids_got);
    free(ids_want);
    free(logits);
}

static void test_shared_expert_gate(void) {
    const uint64_t count = (uint64_t)SHARED_ROWS * HIDDEN;
    float *shared = (float *)malloc(count * sizeof(*shared));
    float *want = (float *)malloc(count * sizeof(*want));
    float *got = (float *)malloc(count * sizeof(*got));
    const float gate[SHARED_ROWS] = {-12.0f, 0.37f, 11.0f};
    REQUIRE(shared && want && got, "shared expert host allocation");
    for (uint64_t i = 0; i < count; i++) {
        shared[i] = 0.8f * sinf((float)(i + 5u) * 0.017f) +
                    0.13f * cosf((float)(i + 19u) * 0.007f);
        want[i] = shared[i] * sigmoid_ref(gate[i / HIDDEN]);
    }

    ds4_gpu_tensor *d_shared = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *d_out = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *d_gate = ds4_gpu_tensor_alloc(sizeof(gate));
    REQUIRE(d_shared && d_out && d_gate, "shared expert GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_shared, 0, shared, count * sizeof(float)),
            "shared expert upload");
    REQUIRE(ds4_gpu_tensor_write(d_gate, 0, gate, sizeof(gate)),
            "shared gate upload");
    REQUIRE(ds4_gpu_qwen4exp_shared_expert_gate_tensor(
                d_out, d_shared, d_gate, SHARED_ROWS, HIDDEN),
            "shared expert gate launch");
    REQUIRE(ds4_gpu_tensor_read(d_out, 0, got, count * sizeof(float)),
            "shared expert output download");
    compare_f32("Qwen sigmoid-gated shared expert", got, want, count,
                2.0e-6f, 2.0e-6f);

    REQUIRE(ds4_gpu_qwen4exp_shared_expert_gate_tensor(
                d_shared, d_shared, d_gate, SHARED_ROWS, HIDDEN),
            "shared expert gate in-place launch");
    REQUIRE(ds4_gpu_tensor_read(d_shared, 0, got, count * sizeof(float)),
            "shared expert in-place download");
    compare_f32("Qwen shared expert gate in-place", got, want, count,
                2.0e-6f, 2.0e-6f);

    ds4_gpu_tensor_free(d_gate);
    ds4_gpu_tensor_free(d_out);
    ds4_gpu_tensor_free(d_shared);
    free(got);
    free(want);
    free(shared);
}

static void fill_q6(test_block_q6_k *blocks, uint64_t count) {
    uint32_t state = 0x83c47ab1u;
    for (uint64_t block = 0; block < count; block++) {
        for (uint32_t i = 0; i < QK_K / 2; i++)
            blocks[block].ql[i] = (uint8_t)(next_u32(&state) >> 24u);
        for (uint32_t i = 0; i < QK_K / 4; i++)
            blocks[block].qh[i] = (uint8_t)(next_u32(&state) >> 24u);
        for (uint32_t i = 0; i < QK_K / 16; i++)
            blocks[block].scales[i] =
                (int8_t)((int)(next_u32(&state) % 17u) - 8);
        blocks[block].d = 0x2000u; /* exactly 0.0078125 */
    }
}

static void fill_q5_0(test_block_q5_0 *blocks, uint64_t count) {
    uint32_t state = 0x4da21937u;
    for (uint64_t block = 0; block < count; block++) {
        blocks[block].d = 0x2800u; /* exactly 0.03125 */
        for (uint32_t i = 0; i < 4u; i++)
            blocks[block].qh[i] = (uint8_t)(next_u32(&state) >> 24u);
        for (uint32_t i = 0; i < QK_5_0 / 2; i++)
            blocks[block].qs[i] = (uint8_t)(next_u32(&state) >> 24u);
    }
}

static void fill_q8_0(test_block_q8_0 *blocks, uint64_t count) {
    uint32_t state = 0x7b392451u;
    for (uint64_t block = 0; block < count; block++) {
        blocks[block].d = 0x2800u; /* exactly 0.03125 */
        for (uint32_t i = 0; i < QK_5_0; i++)
            blocks[block].qs[i] =
                (int8_t)((int)(next_u32(&state) % 255u) - 127);
    }
}

static void dequant_q6(const test_block_q6_k *blocks, float *out) {
    for (uint32_t block_i = 0; block_i < DOWN_MAIN / QK_K; block_i++) {
        const test_block_q6_k *block = blocks + block_i;
        const float d = f16_to_f32(block->d);
        const uint8_t *ql = block->ql;
        const uint8_t *qh = block->qh;
        const int8_t *scales = block->scales;
        float *dst = out + block_i * QK_K;
        for (uint32_t half = 0; half < 2u; half++) {
            for (uint32_t lane = 0; lane < 32u; lane++) {
                const int q1 = (ql[lane] & 0x0f) |
                               (((qh[lane] >> 0u) & 3u) << 4u);
                const int q2 = (ql[lane + 32u] & 0x0f) |
                               (((qh[lane] >> 2u) & 3u) << 4u);
                const int q3 = (ql[lane] >> 4u) |
                               (((qh[lane] >> 4u) & 3u) << 4u);
                const int q4 = (ql[lane + 32u] >> 4u) |
                               (((qh[lane] >> 6u) & 3u) << 4u);
                dst[lane] = d * scales[lane / 16u] * (float)(q1 - 32);
                dst[lane + 32u] =
                    d * scales[lane / 16u + 2u] * (float)(q2 - 32);
                dst[lane + 64u] =
                    d * scales[lane / 16u + 4u] * (float)(q3 - 32);
                dst[lane + 96u] =
                    d * scales[lane / 16u + 6u] * (float)(q4 - 32);
            }
            dst += 128u;
            ql += 64u;
            qh += 32u;
            scales += 8u;
        }
    }
}

static void dequant_q5_0(const test_block_q5_0 *blocks, float *out) {
    for (uint32_t block_i = 0; block_i < DOWN_TAIL / QK_5_0; block_i++) {
        const test_block_q5_0 *block = blocks + block_i;
        const float d = f16_to_f32(block->d);
        const uint32_t qh = (uint32_t)block->qh[0] |
                            ((uint32_t)block->qh[1] << 8u) |
                            ((uint32_t)block->qh[2] << 16u) |
                            ((uint32_t)block->qh[3] << 24u);
        for (uint32_t lane = 0; lane < 16u; lane++) {
            const int q0 = (block->qs[lane] & 0x0f) |
                           (int)(((qh >> lane) & 1u) << 4u);
            const int q1 = (block->qs[lane] >> 4u) |
                           (int)(((qh >> (lane + 16u)) & 1u) << 4u);
            out[block_i * QK_5_0 + lane] = d * (float)(q0 - 16);
            out[block_i * QK_5_0 + lane + 16u] = d * (float)(q1 - 16);
        }
    }
}

static void dequant_q8_0(const test_block_q8_0 *blocks, float *out) {
    for (uint32_t block_i = 0; block_i < DOWN_TAIL / QK_5_0; block_i++) {
        const test_block_q8_0 *block = blocks + block_i;
        const float d = f16_to_f32(block->d);
        for (uint32_t lane = 0; lane < QK_5_0; lane++)
            out[block_i * QK_5_0 + lane] = d * (float)block->qs[lane];
    }
}

static void compare_mmq(const char *name, const float *got,
                        const float *want, uint64_t count) {
    double error2 = 0.0;
    double ref2 = 0.0;
    double got2 = 0.0;
    double dot = 0.0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < count; i++) {
        const double error = (double)got[i] - want[i];
        error2 += error * error;
        ref2 += (double)want[i] * want[i];
        got2 += (double)got[i] * got[i];
        dot += (double)got[i] * want[i];
        const float abs_error = fabsf(got[i] - want[i]);
        if (abs_error > max_abs) max_abs = abs_error;
    }
    const double rel_rms = sqrt(error2 / ref2);
    const double one_minus_cos = 1.0 - dot / sqrt(got2 * ref2);
    printf("%-45s rel_rms %.3e 1-cos %.3e max %.4g\n",
           name, rel_rms, one_minus_cos, max_abs);
    REQUIRE(rel_rms <= 2.0e-2, "split down relative RMS");
    REQUIRE(one_minus_cos <= 2.0e-3, "split down cosine parity");
    REQUIRE(max_abs <= 2.5f, "split down maximum error");
}

static void test_split_down(void *model_map, uint64_t model_size,
                            uint64_t main_offset, uint64_t main_bytes,
                            uint64_t tail_offset, uint64_t tail_bytes,
                            uint64_t q8_tail_offset,
                            uint64_t q8_tail_bytes) {
    const uint64_t mid_count = (uint64_t)DOWN_ASSIGNMENTS * DOWN_MID;
    const uint64_t main_count = (uint64_t)DOWN_ASSIGNMENTS * DOWN_MAIN;
    const uint64_t down_count = (uint64_t)DOWN_ASSIGNMENTS * HIDDEN;
    float *mid = (float *)malloc(mid_count * sizeof(*mid));
    float *packed = (float *)malloc(main_count * sizeof(*packed));
    float *packed_got = (float *)malloc(main_count * sizeof(*packed_got));
    float *main_want = (float *)malloc(down_count * sizeof(*main_want));
    float *tail_want = (float *)malloc(down_count * sizeof(*tail_want));
    float *q8_tail_want = (float *)malloc(down_count * sizeof(*q8_tail_want));
    float *combined_want = (float *)malloc(down_count * sizeof(*combined_want));
    float *main_got = (float *)malloc(down_count * sizeof(*main_got));
    float *combined_got = (float *)malloc(down_count * sizeof(*combined_got));
    float *tail_got = (float *)malloc(down_count * sizeof(*tail_got));
    int32_t *ids = (int32_t *)malloc(DOWN_ASSIGNMENTS * sizeof(*ids));
    REQUIRE(mid && packed && packed_got && main_want && tail_want &&
            q8_tail_want &&
            combined_want && main_got && combined_got && tail_got && ids,
            "split down host allocation");

    for (uint64_t i = 0; i < mid_count; i++)
        mid[i] = 0.31f * sinf((float)(i + 7u) * 0.019f) +
                 0.08f * cosf((float)(i + 23u) * 0.007f);
    for (uint32_t assignment = 0; assignment < DOWN_ASSIGNMENTS; assignment++) {
        ids[assignment] = (int32_t)((assignment * 5u + assignment / 3u) %
                                    DOWN_EXPERTS);
        memcpy(packed + (uint64_t)assignment * DOWN_MAIN,
               mid + (uint64_t)assignment * DOWN_MID,
               DOWN_MAIN * sizeof(float));
    }

    const test_block_q6_k *main_weights = (const test_block_q6_k *)(
        (const unsigned char *)model_map + main_offset);
    const test_block_q5_0 *tail_weights = (const test_block_q5_0 *)(
        (const unsigned char *)model_map + tail_offset);
    const test_block_q8_0 *q8_tail_weights = (const test_block_q8_0 *)(
        (const unsigned char *)model_map + q8_tail_offset);
    const uint64_t main_blocks_per_row = DOWN_MAIN / QK_K;
    const uint64_t tail_blocks_per_row = DOWN_TAIL / QK_5_0;
    float main_row[DOWN_MAIN];
    float tail_row[DOWN_TAIL];
    for (uint32_t assignment = 0; assignment < DOWN_ASSIGNMENTS; assignment++) {
        const uint32_t expert = (uint32_t)ids[assignment];
        const float *input = mid + (uint64_t)assignment * DOWN_MID;
        for (uint32_t out = 0; out < HIDDEN; out++) {
            dequant_q6(main_weights +
                ((uint64_t)expert * HIDDEN + out) * main_blocks_per_row,
                main_row);
            dequant_q5_0(tail_weights +
                ((uint64_t)expert * HIDDEN + out) * tail_blocks_per_row,
                tail_row);
            float main_sum = 0.0f;
            float tail_sum = 0.0f;
            for (uint32_t k = 0; k < DOWN_MAIN; k++)
                main_sum += input[k] * main_row[k];
            for (uint32_t k = 0; k < DOWN_TAIL; k++)
                tail_sum += input[DOWN_MAIN + k] * tail_row[k];
            const uint64_t at = (uint64_t)assignment * HIDDEN + out;
            main_want[at] = main_sum;
            tail_want[at] = tail_sum;
            combined_want[at] = main_sum + tail_sum;
            dequant_q8_0(q8_tail_weights +
                ((uint64_t)expert * HIDDEN + out) * tail_blocks_per_row,
                tail_row);
            q8_tail_want[at] = 0.0f;
            for (uint32_t k = 0; k < DOWN_TAIL; k++)
                q8_tail_want[at] +=
                    input[DOWN_MAIN + k] * tail_row[k];
        }
    }

    ds4_gpu_tensor *d_mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *d_packed = ds4_gpu_tensor_alloc(main_count * sizeof(float));
    ds4_gpu_tensor *d_ids = ds4_gpu_tensor_alloc(
        DOWN_ASSIGNMENTS * sizeof(int32_t));
    ds4_gpu_tensor *d_down = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    REQUIRE(d_mid && d_packed && d_ids && d_down, "split down GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_mid, 0, mid, mid_count * sizeof(float)),
            "split down mid upload");
    REQUIRE(ds4_gpu_tensor_write(d_ids, 0, ids,
                                 DOWN_ASSIGNMENTS * sizeof(int32_t)),
            "split down ids upload");
    REQUIRE(ds4_gpu_qwen4exp_pack_expert_down_main_tensor(
                d_packed, d_mid, DOWN_ASSIGNMENTS, DOWN_MID, DOWN_MAIN),
            "expert-down main pack launch");
    REQUIRE(ds4_gpu_tensor_read(d_packed, 0, packed_got,
                                main_count * sizeof(float)),
            "expert-down main pack download");
    for (uint64_t i = 0; i < main_count; i++) {
        uint32_t got_bits, want_bits;
        memcpy(&got_bits, &packed_got[i], sizeof(got_bits));
        memcpy(&want_bits, &packed[i], sizeof(want_bits));
        REQUIRE(got_bits == want_bits, "expert-down main pack exact");
    }
    printf("%-45s pass (%llu exact values)\n",
           "Qwen expert-down 640 -> contiguous 512 pack",
           (unsigned long long)main_count);

    REQUIRE(ds4_gpu_routed_matmul_tensor(
                d_down, d_packed, d_ids, model_map, model_size,
                main_offset, main_bytes, 14u, DOWN_MAIN, HIDDEN,
                DOWN_EXPERTS, DOWN_ASSIGNMENTS, 1u),
            "Q6_K expert-down main launch");
    REQUIRE(ds4_gpu_tensor_read(d_down, 0, main_got,
                                down_count * sizeof(float)),
            "Q6_K expert-down main download");
    compare_mmq("Qwen Q6_K 512-column expert-down main", main_got,
                main_want, down_count);

    REQUIRE(ds4_gpu_qwen4exp_q5_0_tail_accum_tensor(
                d_down, d_mid, d_ids, model_map, model_size,
                tail_offset, tail_bytes, DOWN_ASSIGNMENTS, DOWN_MID,
                DOWN_MAIN, DOWN_TAIL, HIDDEN, DOWN_EXPERTS),
            "Q5_0 expert-down tail accumulate launch");
    REQUIRE(ds4_gpu_tensor_read(d_down, 0, combined_got,
                                down_count * sizeof(float)),
            "combined split down download");
    for (uint64_t i = 0; i < down_count; i++)
        tail_got[i] = combined_got[i] - main_got[i];
    compare_f32("Qwen Q5_0 128-column tail accumulation", tail_got,
                tail_want, down_count, 8.0e-5f, 8.0e-5f);
    compare_mmq("Qwen combined Q6_K[512] + Q5_0[128]", combined_got,
                combined_want, down_count);

    REQUIRE(ds4_gpu_tensor_write(
                d_down, 0, main_got, down_count * sizeof(*main_got)),
            "reset expert-down main output");
    REQUIRE(ds4_gpu_qwen4exp_q8_0_tail_accum_tensor(
                d_down, d_mid, d_ids, model_map, model_size,
                q8_tail_offset, q8_tail_bytes, DOWN_ASSIGNMENTS, DOWN_MID,
                DOWN_MAIN, DOWN_TAIL, HIDDEN, DOWN_EXPERTS),
            "Q8_0 expert-down tail accumulate launch");
    REQUIRE(ds4_gpu_tensor_read(d_down, 0, combined_got,
                                down_count * sizeof(float)),
            "Q8_0 split down download");
    for (uint64_t i = 0; i < down_count; i++)
        tail_got[i] = combined_got[i] - main_got[i];
    compare_f32("Qwen MTP Q8_0 128-column tail accumulation", tail_got,
                q8_tail_want, down_count, 1.5e-4f, 1.5e-4f);

    REQUIRE(!ds4_gpu_qwen4exp_pack_expert_down_main_tensor(
                d_mid, d_mid, DOWN_ASSIGNMENTS, DOWN_MID, DOWN_MAIN),
            "main pack rejects input/output alias");
    REQUIRE(!ds4_gpu_qwen4exp_q5_0_tail_accum_tensor(
                d_down, d_mid, d_ids, model_map, model_size,
                tail_offset, tail_bytes, DOWN_ASSIGNMENTS, DOWN_MID,
                DOWN_MAIN, DOWN_TAIL - 1u, HIDDEN, DOWN_EXPERTS),
            "tail accumulation rejects non-block width");
    REQUIRE(!ds4_gpu_qwen4exp_q8_0_tail_accum_tensor(
                d_down, d_mid, d_ids, model_map, model_size,
                q8_tail_offset, q8_tail_bytes, DOWN_ASSIGNMENTS, DOWN_MID,
                DOWN_MAIN, DOWN_TAIL - 1u, HIDDEN, DOWN_EXPERTS),
            "Q8 tail accumulation rejects non-block width");

    ds4_gpu_tensor_free(d_down);
    ds4_gpu_tensor_free(d_ids);
    ds4_gpu_tensor_free(d_packed);
    ds4_gpu_tensor_free(d_mid);
    free(ids);
    free(tail_got);
    free(combined_got);
    free(main_got);
    free(combined_want);
    free(tail_want);
    free(main_want);
    free(q8_tail_want);
    free(packed_got);
    free(packed);
    free(mid);
}

static void test_q5_tail_bank2(void *model_map, uint64_t model_size,
                               uint64_t tail_offset, uint64_t tail_bytes) {
    const uint64_t mid_count = (uint64_t)DOWN_ASSIGNMENTS * DOWN_MID;
    const uint64_t down_count = (uint64_t)DOWN_ASSIGNMENTS * HIDDEN;
    float *mid0 = (float *)malloc(mid_count * sizeof(*mid0));
    float *mid1 = (float *)malloc(mid_count * sizeof(*mid1));
    float *seed0 = (float *)malloc(down_count * sizeof(*seed0));
    float *seed1 = (float *)malloc(down_count * sizeof(*seed1));
    float *scalar0 = (float *)malloc(down_count * sizeof(*scalar0));
    float *scalar1 = (float *)malloc(down_count * sizeof(*scalar1));
    float *paired0 = (float *)malloc(down_count * sizeof(*paired0));
    float *paired1 = (float *)malloc(down_count * sizeof(*paired1));
    int32_t ids0[DOWN_ASSIGNMENTS];
    int32_t ids1[DOWN_ASSIGNMENTS];
    REQUIRE(mid0 && mid1 && seed0 && seed1 && scalar0 && scalar1 &&
            paired0 && paired1, "paired Q5 tail host allocation");
    for (uint64_t i = 0; i < mid_count; i++) {
        mid0[i] = 0.27f * sinf((float)(i + 3u) * 0.021f);
        mid1[i] = 0.19f * cosf((float)(i + 17u) * 0.015f) - 0.03f;
    }
    for (uint64_t i = 0; i < down_count; i++) {
        seed0[i] = 0.11f * sinf((float)(i + 1u) * 0.009f);
        seed1[i] = 0.07f * cosf((float)(i + 5u) * 0.013f);
    }
    for (uint32_t i = 0; i < DOWN_ASSIGNMENTS; i++) {
        ids0[i] = (int32_t)((i * 3u + 1u) % DOWN_EXPERTS);
        ids1[i] = (int32_t)((i * 5u + 2u) % DOWN_EXPERTS);
    }

    ds4_gpu_tensor *d_mid0 = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *d_mid1 = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *d_ids0 = ds4_gpu_tensor_alloc(sizeof(ids0));
    ds4_gpu_tensor *d_ids1 = ds4_gpu_tensor_alloc(sizeof(ids1));
    ds4_gpu_tensor *d_down0 = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    ds4_gpu_tensor *d_down1 = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    REQUIRE(d_mid0 && d_mid1 && d_ids0 && d_ids1 && d_down0 && d_down1,
            "paired Q5 tail GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_mid0, 0, mid0, mid_count * sizeof(float)) &&
            ds4_gpu_tensor_write(d_mid1, 0, mid1, mid_count * sizeof(float)) &&
            ds4_gpu_tensor_write(d_ids0, 0, ids0, sizeof(ids0)) &&
            ds4_gpu_tensor_write(d_ids1, 0, ids1, sizeof(ids1)),
            "paired Q5 tail input upload");
    REQUIRE(setenv("DS4_QWEN_Q5_TAIL_EXPERT_MAJOR", "0", 1) == 0,
            "select assignment-major Q5 tail path");

    REQUIRE(ds4_gpu_tensor_write(
                d_down0, 0, seed0, down_count * sizeof(float)) &&
            ds4_gpu_tensor_write(
                d_down1, 0, seed1, down_count * sizeof(float)),
            "scalar Q5 tail seed upload");
    REQUIRE(ds4_gpu_qwen4exp_q5_0_tail_accum_tensor(
                d_down0, d_mid0, d_ids0, model_map, model_size,
                tail_offset, tail_bytes, DOWN_ASSIGNMENTS, DOWN_MID,
                DOWN_MAIN, DOWN_TAIL, HIDDEN, DOWN_EXPERTS) &&
            ds4_gpu_qwen4exp_q5_0_tail_accum_tensor(
                d_down1, d_mid1, d_ids1, model_map, model_size,
                tail_offset, tail_bytes, DOWN_ASSIGNMENTS, DOWN_MID,
                DOWN_MAIN, DOWN_TAIL, HIDDEN, DOWN_EXPERTS),
            "scalar Q5 tail bank launches");
    REQUIRE(ds4_gpu_tensor_read(
                d_down0, 0, scalar0, down_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                d_down1, 0, scalar1, down_count * sizeof(float)),
            "scalar Q5 tail bank download");

    REQUIRE(ds4_gpu_tensor_write(
                d_down0, 0, seed0, down_count * sizeof(float)) &&
            ds4_gpu_tensor_write(
                d_down1, 0, seed1, down_count * sizeof(float)),
            "paired Q5 tail seed upload");
    REQUIRE(ds4_gpu_qwen4exp_q5_0_tail_accum_bank2_tensor(
                d_down0, d_mid0, d_ids0, d_down1, d_mid1, d_ids1,
                model_map, model_size, tail_offset, tail_bytes,
                DOWN_ASSIGNMENTS, DOWN_MID, DOWN_MAIN, DOWN_TAIL,
                HIDDEN, DOWN_EXPERTS),
            "paired Q5 tail launch");
    REQUIRE(ds4_gpu_tensor_read(
                d_down0, 0, paired0, down_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                d_down1, 0, paired1, down_count * sizeof(float)),
            "paired Q5 tail download");
    REQUIRE(memcmp(paired0, scalar0, down_count * sizeof(float)) == 0,
            "paired Q5 tail bank0 bit parity");
    REQUIRE(memcmp(paired1, scalar1, down_count * sizeof(float)) == 0,
            "paired Q5 tail bank1 bit parity");
    REQUIRE(!ds4_gpu_qwen4exp_q5_0_tail_accum_bank2_tensor(
                d_down0, d_mid0, d_ids0, d_down0, d_mid1, d_ids1,
                model_map, model_size, tail_offset, tail_bytes,
                DOWN_ASSIGNMENTS, DOWN_MID, DOWN_MAIN, DOWN_TAIL,
                HIDDEN, DOWN_EXPERTS),
            "paired Q5 tail rejects shared output");
    printf("%-45s pass (%llu exact values per bank)\n",
           "Qwen paired Q5_0 tail bit parity",
           (unsigned long long)down_count);
    REQUIRE(setenv("DS4_QWEN_Q5_TAIL_EXPERT_MAJOR", "1", 1) == 0,
            "restore expert-major Q5 tail test path");

    ds4_gpu_tensor_free(d_down1);
    ds4_gpu_tensor_free(d_down0);
    ds4_gpu_tensor_free(d_ids1);
    ds4_gpu_tensor_free(d_ids0);
    ds4_gpu_tensor_free(d_mid1);
    ds4_gpu_tensor_free(d_mid0);
    free(paired1);
    free(paired0);
    free(scalar1);
    free(scalar0);
    free(seed1);
    free(seed0);
    free(mid1);
    free(mid0);
}

enum {
    FUSED_ASSIGNMENTS = 1000,
    Q8_MAIN_BLOCKS_PER_ROW = DOWN_MAIN / QK_5_0,
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} test_block_q5_k;

_Static_assert(sizeof(test_block_q5_k) == 176, "Q5_K block layout");

static void fill_q5_k(test_block_q5_k *blocks, uint64_t count) {
    uint32_t state = 0x5e1c93a7u;
    for (uint64_t block = 0; block < count; block++) {
        blocks[block].d = 0x2000u;    /* exactly 0.0078125 */
        blocks[block].dmin = 0x1c00u; /* exactly 0.00390625 */
        for (uint32_t i = 0; i < 12u; i++)
            blocks[block].scales[i] = (uint8_t)(next_u32(&state) >> 24u);
        for (uint32_t i = 0; i < QK_K / 8; i++)
            blocks[block].qh[i] = (uint8_t)(next_u32(&state) >> 24u);
        for (uint32_t i = 0; i < QK_K / 2; i++)
            blocks[block].qs[i] = (uint8_t)(next_u32(&state) >> 24u);
    }
}

static void dequant_q8_0_blocks(const test_block_q8_0 *blocks,
                                uint32_t n_blocks, float *out) {
    for (uint32_t block_i = 0; block_i < n_blocks; block_i++) {
        const test_block_q8_0 *block = blocks + block_i;
        const float d = f16_to_f32(block->d);
        for (uint32_t lane = 0; lane < QK_5_0; lane++)
            out[block_i * QK_5_0 + lane] = d * (float)block->qs[lane];
    }
}

/* Dequantize every expert-down row once: [DOWN_EXPERTS][HIDDEN][DOWN_MID],
 * main columns first, then the 128-column tail. */
static float *fused_reference_rows(const void *model_map,
                                   uint64_t main_offset, uint32_t main_type,
                                   uint64_t tail_offset, uint32_t tail_type) {
    const uint64_t rows = (uint64_t)DOWN_EXPERTS * HIDDEN;
    float *w = (float *)malloc(rows * DOWN_MID * sizeof(*w));
    REQUIRE(w, "fused reference row allocation");
    const unsigned char *base = (const unsigned char *)model_map;
    for (uint64_t row = 0; row < rows; row++) {
        float *dst = w + row * DOWN_MID;
        if (main_type == 14u) {
            dequant_q6((const test_block_q6_k *)(base + main_offset) +
                           row * (DOWN_MAIN / QK_K), dst);
        } else {
            dequant_q8_0_blocks((const test_block_q8_0 *)(base + main_offset) +
                                    row * Q8_MAIN_BLOCKS_PER_ROW,
                                Q8_MAIN_BLOCKS_PER_ROW, dst);
        }
        if (tail_type == 6u) {
            dequant_q5_0((const test_block_q5_0 *)(base + tail_offset) +
                             row * (DOWN_TAIL / QK_5_0), dst + DOWN_MAIN);
        } else {
            dequant_q8_0((const test_block_q8_0 *)(base + tail_offset) +
                             row * (DOWN_TAIL / QK_5_0), dst + DOWN_MAIN);
        }
    }
    return w;
}

/* The fused main+tail MMQ entry against an F32 reference and against the
 * separate main GEMM + tail accumulate it replaces.  1000 assignments over
 * eight experts give ~125-row buckets, so both the 128-wide worklist tiles
 * and the narrow ragged tails run. */
static void test_fused_down(void *model_map, uint64_t model_size,
                            uint64_t main_offset, uint64_t main_bytes,
                            uint32_t main_type, uint64_t tail_offset,
                            uint64_t tail_bytes, uint32_t tail_type,
                            const char *name) {
    const uint64_t mid_count = (uint64_t)FUSED_ASSIGNMENTS * DOWN_MID;
    const uint64_t main_count = (uint64_t)FUSED_ASSIGNMENTS * DOWN_MAIN;
    const uint64_t down_count = (uint64_t)FUSED_ASSIGNMENTS * HIDDEN;
    float *mid = (float *)malloc(mid_count * sizeof(*mid));
    float *want = (float *)malloc(down_count * sizeof(*want));
    float *fused = (float *)malloc(down_count * sizeof(*fused));
    float *separate = (float *)malloc(down_count * sizeof(*separate));
    int32_t *ids = (int32_t *)malloc(FUSED_ASSIGNMENTS * sizeof(*ids));
    REQUIRE(mid && want && fused && separate && ids,
            "fused down host allocation");
    for (uint64_t i = 0; i < mid_count; i++)
        mid[i] = 0.29f * sinf((float)(i + 11u) * 0.017f) +
                 0.06f * cosf((float)(i + 5u) * 0.011f);
    for (uint32_t a = 0; a < FUSED_ASSIGNMENTS; a++)
        ids[a] = (int32_t)((a * 7u + a / 5u) % DOWN_EXPERTS);

    float *w = fused_reference_rows(model_map, main_offset, main_type,
                                    tail_offset, tail_type);
    for (uint32_t a = 0; a < FUSED_ASSIGNMENTS; a++) {
        const float *x = mid + (uint64_t)a * DOWN_MID;
        const float *rows = w + (uint64_t)ids[a] * HIDDEN * DOWN_MID;
        for (uint32_t out = 0; out < HIDDEN; out++) {
            const float *row = rows + (uint64_t)out * DOWN_MID;
            float sum = 0.0f;
            for (uint32_t k = 0; k < DOWN_MID; k++) sum += x[k] * row[k];
            want[(uint64_t)a * HIDDEN + out] = sum;
        }
    }
    free(w);

    ds4_gpu_tensor *d_mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *d_packed = ds4_gpu_tensor_alloc(main_count * sizeof(float));
    ds4_gpu_tensor *d_ids = ds4_gpu_tensor_alloc(
        FUSED_ASSIGNMENTS * sizeof(int32_t));
    ds4_gpu_tensor *d_down = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    REQUIRE(d_mid && d_packed && d_ids && d_down, "fused down GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_mid, 0, mid, mid_count * sizeof(float)) &&
            ds4_gpu_tensor_write(d_ids, 0, ids,
                                 FUSED_ASSIGNMENTS * sizeof(int32_t)),
            "fused down input upload");
    REQUIRE(ds4_gpu_qwen4exp_pack_expert_down_main_tensor(
                d_packed, d_mid, FUSED_ASSIGNMENTS, DOWN_MID, DOWN_MAIN),
            "fused down main pack launch");
    REQUIRE(ds4_gpu_qwen4exp_routed_down_fused_tensor(
                d_down, d_packed, d_mid, d_ids, model_map, model_size,
                main_offset, main_bytes, main_type, tail_offset, tail_bytes,
                tail_type, FUSED_ASSIGNMENTS, DOWN_MID, DOWN_MAIN, DOWN_TAIL,
                HIDDEN, DOWN_EXPERTS, FUSED_ASSIGNMENTS),
            "fused expert-down launch");
    REQUIRE(ds4_gpu_tensor_read(d_down, 0, fused, down_count * sizeof(float)),
            "fused expert-down download");
    compare_mmq(name, fused, want, down_count);

    REQUIRE(ds4_gpu_routed_matmul_tensor(
                d_down, d_packed, d_ids, model_map, model_size,
                main_offset, main_bytes, main_type, DOWN_MAIN, HIDDEN,
                DOWN_EXPERTS, FUSED_ASSIGNMENTS, 1u),
            "separate expert-down main launch");
    REQUIRE(tail_type == 6u
                ? ds4_gpu_qwen4exp_q5_0_tail_accum_tensor(
                      d_down, d_mid, d_ids, model_map, model_size,
                      tail_offset, tail_bytes, FUSED_ASSIGNMENTS, DOWN_MID,
                      DOWN_MAIN, DOWN_TAIL, HIDDEN, DOWN_EXPERTS)
                : ds4_gpu_qwen4exp_q8_0_tail_accum_tensor(
                      d_down, d_mid, d_ids, model_map, model_size,
                      tail_offset, tail_bytes, FUSED_ASSIGNMENTS, DOWN_MID,
                      DOWN_MAIN, DOWN_TAIL, HIDDEN, DOWN_EXPERTS),
            "separate expert-down tail launch");
    REQUIRE(ds4_gpu_tensor_read(d_down, 0, separate,
                                down_count * sizeof(float)),
            "separate expert-down download");
    compare_mmq("  fused against separate main + tail", fused, separate,
                down_count);

    REQUIRE(!ds4_gpu_qwen4exp_routed_down_fused_tensor(
                d_down, d_packed, d_mid, d_ids, model_map, model_size,
                main_offset, main_bytes, main_type, tail_offset, tail_bytes,
                tail_type, FUSED_ASSIGNMENTS, DOWN_MID, DOWN_MAIN,
                DOWN_TAIL - QK_5_0, HIDDEN, DOWN_EXPERTS, FUSED_ASSIGNMENTS),
            "fused entry rejects a non-128 tail");
    REQUIRE(!ds4_gpu_qwen4exp_routed_down_fused_tensor(
                d_down, d_packed, d_mid, d_ids, model_map, model_size,
                main_offset, main_bytes, main_type, tail_offset, tail_bytes,
                tail_type, FUSED_ASSIGNMENTS, DOWN_MID, DOWN_MAIN, DOWN_TAIL,
                HIDDEN, DOWN_EXPERTS, 0u),
            "fused entry rejects an empty bucket bound");

    ds4_gpu_tensor_free(d_down);
    ds4_gpu_tensor_free(d_ids);
    ds4_gpu_tensor_free(d_packed);
    ds4_gpu_tensor_free(d_mid);
    free(ids);
    free(separate);
    free(fused);
    free(want);
    free(mid);
}

/* Production-shape launch for NCU: Q5_K[512] main + Q5_0[128] tail over
 * 8,025 tokens x top-10 through 512 experts, nothing else resident. */
enum {
    DENSE_TAIL_K = 640,
    DENSE_TAIL_ROWS = 300,
    DENSE_TAIL_SMALL_ROWS = 40,
    DENSE_TAIL_BLOCKS_PER_ROW = DENSE_TAIL_K / QK_5_0,
    SHEXP_PAIR_ROWS = 100,
    SHEXP_PAIR_SMALL_ROWS = 32,
    SHEXP_FF = 640,
    SHEXP_K_BLOCKS = HIDDEN / QK_5_0,
};

/* Dense Q8_0 [K=640 x M=2560] against an F32 reference: prefill width takes
 * the fused MMQ tail path, the narrow width stays on the legacy batch kernel,
 * and the two must agree on the rows they share. */
static void test_dense_tail(void *model_map, uint64_t model_size,
                            uint64_t weight_offset, uint64_t weight_bytes) {
    const uint64_t x_count = (uint64_t)DENSE_TAIL_ROWS * DENSE_TAIL_K;
    const uint64_t out_count = (uint64_t)DENSE_TAIL_ROWS * HIDDEN;
    const uint64_t small_count = (uint64_t)DENSE_TAIL_SMALL_ROWS * HIDDEN;
    float *x = (float *)malloc(x_count * sizeof(*x));
    float *want = (float *)malloc(out_count * sizeof(*want));
    float *got = (float *)malloc(out_count * sizeof(*got));
    float *small = (float *)malloc(small_count * sizeof(*small));
    float *row = (float *)malloc(DENSE_TAIL_K * sizeof(*row));
    REQUIRE(x && want && got && small && row, "dense tail host allocation");
    for (uint64_t i = 0; i < x_count; i++)
        x[i] = 0.33f * sinf((float)(i + 3u) * 0.013f) -
               0.05f * cosf((float)(i + 19u) * 0.021f);
    const test_block_q8_0 *w = (const test_block_q8_0 *)(
        (const unsigned char *)model_map + weight_offset);
    for (uint32_t out = 0; out < HIDDEN; out++) {
        dequant_q8_0_blocks(w + (uint64_t)out * DENSE_TAIL_BLOCKS_PER_ROW,
                            DENSE_TAIL_BLOCKS_PER_ROW, row);
        for (uint32_t r = 0; r < DENSE_TAIL_ROWS; r++) {
            const float *xr = x + (uint64_t)r * DENSE_TAIL_K;
            float sum = 0.0f;
            for (uint32_t k = 0; k < DENSE_TAIL_K; k++) sum += xr[k] * row[k];
            want[(uint64_t)r * HIDDEN + out] = sum;
        }
    }

    ds4_gpu_tensor *d_x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *d_out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    REQUIRE(d_x && d_out, "dense tail GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_x, 0, x, x_count * sizeof(float)),
            "dense tail input upload");
    REQUIRE(ds4_gpu_matmul_q8_0_tensor(
                d_out, model_map, model_size, weight_offset, DENSE_TAIL_K,
                HIDDEN, d_x, DENSE_TAIL_ROWS),
            "dense tail prefill-width launch");
    REQUIRE(ds4_gpu_tensor_read(d_out, 0, got, out_count * sizeof(float)),
            "dense tail prefill-width download");
    compare_mmq("Qwen dense Q8_0 K=640 fused MMQ tail", got, want, out_count);
    REQUIRE(ds4_gpu_matmul_q8_0_tensor(
                d_out, model_map, model_size, weight_offset, DENSE_TAIL_K,
                HIDDEN, d_x, DENSE_TAIL_SMALL_ROWS),
            "dense tail narrow-width launch");
    REQUIRE(ds4_gpu_tensor_read(d_out, 0, small, small_count * sizeof(float)),
            "dense tail narrow-width download");
    compare_mmq("  narrow width legacy kernel", small, want, small_count);
    compare_mmq("  fused against legacy shared rows", got, small, small_count);

    ds4_gpu_tensor_free(d_out);
    ds4_gpu_tensor_free(d_x);
    free(row);
    free(small);
    free(got);
    free(want);
    free(x);
}

/* Shared-expert gate/up pair (2560 -> 640/640): prefill width splits across
 * the common MMQ kernels, the narrow width keeps the pair kernel. */
static void test_shexp_pair(void *model_map, uint64_t model_size,
                            uint64_t gate_offset, uint64_t up_offset) {
    const uint64_t x_count = (uint64_t)SHEXP_PAIR_ROWS * HIDDEN;
    const uint64_t out_count = (uint64_t)SHEXP_PAIR_ROWS * SHEXP_FF;
    const uint64_t small_count = (uint64_t)SHEXP_PAIR_SMALL_ROWS * SHEXP_FF;
    float *x = (float *)malloc(x_count * sizeof(*x));
    float *want_gate = (float *)malloc(out_count * sizeof(*want_gate));
    float *want_up = (float *)malloc(out_count * sizeof(*want_up));
    float *gate = (float *)malloc(out_count * sizeof(*gate));
    float *up = (float *)malloc(out_count * sizeof(*up));
    float *row = (float *)malloc(HIDDEN * sizeof(*row));
    REQUIRE(x && want_gate && want_up && gate && up && row,
            "shared pair host allocation");
    for (uint64_t i = 0; i < x_count; i++)
        x[i] = 0.21f * sinf((float)(i + 7u) * 0.009f) +
               0.04f * cosf((float)(i + 2u) * 0.031f);
    const uint64_t offsets[2] = {gate_offset, up_offset};
    float *wants[2] = {want_gate, want_up};
    for (uint32_t which = 0; which < 2u; which++) {
        const test_block_q8_0 *w = (const test_block_q8_0 *)(
            (const unsigned char *)model_map + offsets[which]);
        for (uint32_t out = 0; out < SHEXP_FF; out++) {
            dequant_q8_0_blocks(w + (uint64_t)out * SHEXP_K_BLOCKS,
                                SHEXP_K_BLOCKS, row);
            for (uint32_t r = 0; r < SHEXP_PAIR_ROWS; r++) {
                const float *xr = x + (uint64_t)r * HIDDEN;
                float sum = 0.0f;
                for (uint32_t k = 0; k < HIDDEN; k++) sum += xr[k] * row[k];
                wants[which][(uint64_t)r * SHEXP_FF + out] = sum;
            }
        }
    }

    ds4_gpu_tensor *d_x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *d_gate = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *d_up = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    REQUIRE(d_x && d_gate && d_up, "shared pair GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_x, 0, x, x_count * sizeof(float)),
            "shared pair input upload");
    REQUIRE(ds4_gpu_matmul_q8_0_pair_tensor(
                d_gate, d_up, model_map, model_size, gate_offset, up_offset,
                HIDDEN, SHEXP_FF, SHEXP_FF, d_x, SHEXP_PAIR_ROWS),
            "shared pair prefill-width launch");
    REQUIRE(ds4_gpu_tensor_read(d_gate, 0, gate, out_count * sizeof(float)) &&
            ds4_gpu_tensor_read(d_up, 0, up, out_count * sizeof(float)),
            "shared pair prefill-width download");
    compare_mmq("Qwen shared gate via MMQ split", gate, want_gate, out_count);
    compare_mmq("Qwen shared up via MMQ split", up, want_up, out_count);
    REQUIRE(ds4_gpu_matmul_q8_0_pair_tensor(
                d_gate, d_up, model_map, model_size, gate_offset, up_offset,
                HIDDEN, SHEXP_FF, SHEXP_FF, d_x, SHEXP_PAIR_SMALL_ROWS),
            "shared pair narrow-width launch");
    REQUIRE(ds4_gpu_tensor_read(d_gate, 0, gate, small_count * sizeof(float)) &&
            ds4_gpu_tensor_read(d_up, 0, up, small_count * sizeof(float)),
            "shared pair narrow-width download");
    compare_mmq("  narrow width pair kernel gate", gate, want_gate, small_count);
    compare_mmq("  narrow width pair kernel up", up, want_up, small_count);

    ds4_gpu_tensor_free(d_up);
    ds4_gpu_tensor_free(d_gate);
    ds4_gpu_tensor_free(d_x);
    free(row);
    free(up);
    free(gate);
    free(want_up);
    free(want_gate);
    free(x);
}

enum {
    PAIR_TOKENS = 300,
    PAIR_USED = 4,
    PAIR_FF = 640,
    PAIR_EXPERTS = DOWN_EXPERTS,
};

/* Routed gate/up pair through the compact worklist (Q5_K edge layers, Q8_0
 * MTP block) against the generic per-tensor routed matmul over the same
 * weights and routing: both are MMQ over identical Q8_1 activations, so they
 * must agree to accumulation order. */
static void test_routed_pair(void *model_map, uint64_t model_size,
                             uint32_t weight_type, uint64_t gate_offset,
                             uint64_t up_offset, uint64_t weight_bytes,
                             const char *name) {
    const uint64_t x_count = (uint64_t)PAIR_TOKENS * HIDDEN;
    const uint64_t assignments = (uint64_t)PAIR_TOKENS * PAIR_USED;
    const uint64_t out_count = assignments * PAIR_FF;
    float *x = (float *)malloc(x_count * sizeof(*x));
    int32_t *ids = (int32_t *)malloc(assignments * sizeof(*ids));
    float *pair_gate = (float *)malloc(out_count * sizeof(*pair_gate));
    float *pair_up = (float *)malloc(out_count * sizeof(*pair_up));
    float *ref_gate = (float *)malloc(out_count * sizeof(*ref_gate));
    float *ref_up = (float *)malloc(out_count * sizeof(*ref_up));
    REQUIRE(x && ids && pair_gate && pair_up && ref_gate && ref_up,
            "routed pair host allocation");
    for (uint64_t i = 0; i < x_count; i++)
        x[i] = 0.24f * sinf((float)(i + 13u) * 0.012f) +
               0.05f * cosf((float)(i + 3u) * 0.027f);
    for (uint32_t t = 0; t < PAIR_TOKENS; t++)
        for (uint32_t k = 0; k < PAIR_USED; k++)
            ids[(uint64_t)t * PAIR_USED + k] =
                (int32_t)((t * 3u + k * 5u + t / 7u) % PAIR_EXPERTS);
    /* Router top-k is without replacement: keep the slots distinct. */
    for (uint32_t t = 0; t < PAIR_TOKENS; t++)
        for (uint32_t k = 1; k < PAIR_USED; k++)
            for (uint32_t j = 0; j < k; j++)
                if (ids[(uint64_t)t * PAIR_USED + k] ==
                    ids[(uint64_t)t * PAIR_USED + j])
                    ids[(uint64_t)t * PAIR_USED + k] =
                        (ids[(uint64_t)t * PAIR_USED + k] + 1) % PAIR_EXPERTS;

    ds4_gpu_tensor *d_x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *d_ids = ds4_gpu_tensor_alloc(assignments * sizeof(int32_t));
    ds4_gpu_tensor *d_gate = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *d_up = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    REQUIRE(d_x && d_ids && d_gate && d_up, "routed pair GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_x, 0, x, x_count * sizeof(float)) &&
            ds4_gpu_tensor_write(d_ids, 0, ids, assignments * sizeof(int32_t)),
            "routed pair input upload");
    REQUIRE(ds4_gpu_routed_gate_up_tensor(
                d_gate, d_up, d_x, d_ids, model_map, model_size,
                gate_offset, weight_bytes, up_offset, weight_bytes,
                weight_type, HIDDEN, PAIR_FF, PAIR_EXPERTS, PAIR_TOKENS,
                PAIR_USED),
            "routed pair launch");
    REQUIRE(ds4_gpu_tensor_read(d_gate, 0, pair_gate, out_count * sizeof(float)) &&
            ds4_gpu_tensor_read(d_up, 0, pair_up, out_count * sizeof(float)),
            "routed pair download");
    REQUIRE(ds4_gpu_routed_matmul_tensor(
                d_gate, d_x, d_ids, model_map, model_size, gate_offset,
                weight_bytes, weight_type, HIDDEN, PAIR_FF, PAIR_EXPERTS,
                PAIR_TOKENS, PAIR_USED) &&
            ds4_gpu_routed_matmul_tensor(
                d_up, d_x, d_ids, model_map, model_size, up_offset,
                weight_bytes, weight_type, HIDDEN, PAIR_FF, PAIR_EXPERTS,
                PAIR_TOKENS, PAIR_USED),
            "routed reference launch");
    REQUIRE(ds4_gpu_tensor_read(d_gate, 0, ref_gate, out_count * sizeof(float)) &&
            ds4_gpu_tensor_read(d_up, 0, ref_up, out_count * sizeof(float)),
            "routed reference download");
    char label[96];
    snprintf(label, sizeof(label), "%s gate", name);
    compare_mmq(label, pair_gate, ref_gate, out_count);
    snprintf(label, sizeof(label), "%s up", name);
    compare_mmq(label, pair_up, ref_up, out_count);

    ds4_gpu_tensor_free(d_up);
    ds4_gpu_tensor_free(d_gate);
    ds4_gpu_tensor_free(d_ids);
    ds4_gpu_tensor_free(d_x);
    free(ref_up);
    free(ref_gate);
    free(pair_up);
    free(pair_gate);
    free(ids);
    free(x);
}

enum {
    DECODE_EXPERTS = 16,
    DECODE_USED = 10,
};

/* Decode-width routed projections (one token, top-10) through the public
 * entries: gate/up over Q8_0 experts and the Q6_K[512] + Q5_0[128] down,
 * against F32 references.  With 10 assignments both now ride the vector
 * tier (the MMQ tiles served every shape above eight before). */
static void test_decode_vec(void *model_map, uint64_t model_size,
                            uint64_t gate_offset, uint64_t up_offset,
                            uint64_t gu_bytes, uint64_t main_offset,
                            uint64_t main_bytes, uint64_t tail_offset,
                            uint64_t tail_bytes) {
    const uint64_t gu_count = (uint64_t)DECODE_USED * PAIR_FF;
    const uint64_t mid_count = (uint64_t)DECODE_USED * DOWN_MID;
    const uint64_t main_count = (uint64_t)DECODE_USED * DOWN_MAIN;
    const uint64_t down_count = (uint64_t)DECODE_USED * HIDDEN;
    float *x = (float *)malloc(HIDDEN * sizeof(*x));
    float *gate_want = (float *)malloc(gu_count * sizeof(*gate_want));
    float *up_want = (float *)malloc(gu_count * sizeof(*up_want));
    float *gate_got = (float *)malloc(gu_count * sizeof(*gate_got));
    float *up_got = (float *)malloc(gu_count * sizeof(*up_got));
    float *mid = (float *)malloc(mid_count * sizeof(*mid));
    float *down_want = (float *)malloc(down_count * sizeof(*down_want));
    float *down_got = (float *)malloc(down_count * sizeof(*down_got));
    /* Scratch row: a dequantized gate/up row (HIDDEN) or down row (DOWN_MID). */
    float *row = (float *)malloc((HIDDEN > DOWN_MID ? HIDDEN : DOWN_MID) * sizeof(*row));
    int32_t ids[DECODE_USED];
    REQUIRE(x && gate_want && up_want && gate_got && up_got && mid &&
            down_want && down_got && row, "decode vec host allocation");
    for (uint32_t i = 0; i < HIDDEN; i++)
        x[i] = 0.27f * sinf((float)(i + 5u) * 0.011f) +
               0.04f * cosf((float)(i + 9u) * 0.023f);
    for (uint32_t k = 0; k < DECODE_USED; k++)
        ids[k] = (int32_t)((k * 3u + 1u) % DECODE_EXPERTS);
    for (uint64_t i = 0; i < mid_count; i++)
        mid[i] = 0.31f * sinf((float)(i + 2u) * 0.017f) -
                 0.05f * cosf((float)(i + 7u) * 0.009f);

    const unsigned char *base = (const unsigned char *)model_map;
    for (uint32_t k = 0; k < DECODE_USED; k++) {
        const uint64_t expert = (uint64_t)ids[k];
        for (uint32_t o = 0; o < PAIR_FF; o++) {
            const uint64_t wrow = (expert * PAIR_FF + o) * SHEXP_K_BLOCKS;
            float g = 0.0f, u = 0.0f;
            dequant_q8_0_blocks((const test_block_q8_0 *)(base + gate_offset) + wrow,
                                SHEXP_K_BLOCKS, row);
            for (uint32_t i = 0; i < HIDDEN; i++) g += x[i] * row[i];
            dequant_q8_0_blocks((const test_block_q8_0 *)(base + up_offset) + wrow,
                                SHEXP_K_BLOCKS, row);
            for (uint32_t i = 0; i < HIDDEN; i++) u += x[i] * row[i];
            gate_want[(uint64_t)k * PAIR_FF + o] = g;
            up_want[(uint64_t)k * PAIR_FF + o] = u;
        }
        const float *xr = mid + (uint64_t)k * DOWN_MID;
        for (uint32_t o = 0; o < HIDDEN; o++) {
            dequant_q6((const test_block_q6_k *)(base + main_offset) +
                           (expert * HIDDEN + o) * (DOWN_MAIN / QK_K), row);
            dequant_q5_0((const test_block_q5_0 *)(base + tail_offset) +
                             (expert * HIDDEN + o) * (DOWN_TAIL / QK_5_0),
                         row + DOWN_MAIN);
            float sum = 0.0f;
            for (uint32_t i = 0; i < DOWN_MID; i++) sum += xr[i] * row[i];
            down_want[(uint64_t)k * HIDDEN + o] = sum;
        }
    }

    ds4_gpu_tensor *d_x = ds4_gpu_tensor_alloc(HIDDEN * sizeof(float));
    ds4_gpu_tensor *d_ids = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *d_gate = ds4_gpu_tensor_alloc(gu_count * sizeof(float));
    ds4_gpu_tensor *d_up = ds4_gpu_tensor_alloc(gu_count * sizeof(float));
    ds4_gpu_tensor *d_mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *d_packed = ds4_gpu_tensor_alloc(main_count * sizeof(float));
    ds4_gpu_tensor *d_down = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    REQUIRE(d_x && d_ids && d_gate && d_up && d_mid && d_packed && d_down,
            "decode vec GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(d_x, 0, x, HIDDEN * sizeof(float)) &&
            ds4_gpu_tensor_write(d_ids, 0, ids, sizeof(ids)) &&
            ds4_gpu_tensor_write(d_mid, 0, mid, mid_count * sizeof(float)),
            "decode vec input upload");
    REQUIRE(ds4_gpu_routed_gate_up_tensor(
                d_gate, d_up, d_x, d_ids, model_map, model_size,
                gate_offset, gu_bytes, up_offset, gu_bytes, 8u, HIDDEN,
                PAIR_FF, DECODE_EXPERTS, 1u, DECODE_USED),
            "decode gate/up launch");
    REQUIRE(ds4_gpu_tensor_read(d_gate, 0, gate_got, gu_count * sizeof(float)) &&
            ds4_gpu_tensor_read(d_up, 0, up_got, gu_count * sizeof(float)),
            "decode gate/up download");
    compare_mmq("Qwen decode top-10 Q8_0 gate (vec tier)", gate_got, gate_want, gu_count);
    compare_mmq("Qwen decode top-10 Q8_0 up (vec tier)", up_got, up_want, gu_count);

    REQUIRE(ds4_gpu_qwen4exp_pack_expert_down_main_tensor(
                d_packed, d_mid, DECODE_USED, DOWN_MID, DOWN_MAIN),
            "decode down pack launch");
    REQUIRE(ds4_gpu_routed_matmul_bounded_tensor(
                d_down, d_packed, d_ids, model_map, model_size,
                main_offset, main_bytes, 14u, DOWN_MAIN, HIDDEN,
                DECODE_EXPERTS, DECODE_USED, 1u, 1u),
            "decode down main launch");
    REQUIRE(ds4_gpu_qwen4exp_q5_0_tail_accum_tensor(
                d_down, d_mid, d_ids, model_map, model_size,
                tail_offset, tail_bytes, DECODE_USED, DOWN_MID,
                DOWN_MAIN, DOWN_TAIL, HIDDEN, DECODE_EXPERTS),
            "decode down tail launch");
    REQUIRE(ds4_gpu_tensor_read(d_down, 0, down_got, down_count * sizeof(float)),
            "decode down download");
    compare_mmq("Qwen decode top-10 Q6_K[512]+Q5_0[128] down", down_got, down_want, down_count);

    ds4_gpu_tensor_free(d_down);
    ds4_gpu_tensor_free(d_packed);
    ds4_gpu_tensor_free(d_mid);
    ds4_gpu_tensor_free(d_up);
    ds4_gpu_tensor_free(d_gate);
    ds4_gpu_tensor_free(d_ids);
    ds4_gpu_tensor_free(d_x);
    free(row);
    free(down_got);
    free(down_want);
    free(mid);
    free(up_got);
    free(gate_got);
    free(up_want);
    free(gate_want);
    free(x);
}

static void profile_fused_down(void) {
    enum {
        tokens = 8025,
        experts = 512,
        used = 10,
        assignments = tokens * used,
    };
    const uint64_t main_offset = 4096u;
    const uint64_t main_blocks =
        (uint64_t)experts * HIDDEN * (DOWN_MAIN / QK_K);
    const uint64_t main_bytes = main_blocks * sizeof(test_block_q5_k);
    const uint64_t tail_offset = (main_offset + main_bytes + 4095u) & ~4095ull;
    const uint64_t tail_blocks =
        (uint64_t)experts * HIDDEN * (DOWN_TAIL / QK_5_0);
    const uint64_t tail_bytes = tail_blocks * sizeof(test_block_q5_0);
    const uint64_t model_size = (tail_offset + tail_bytes + 4095u) & ~4095ull;
    void *model_map = NULL;
    REQUIRE(posix_memalign(&model_map, 4096u, (size_t)model_size) == 0,
            "fused down profile model allocation");
    memset(model_map, 0, (size_t)model_size);
    fill_q5_k((test_block_q5_k *)((unsigned char *)model_map + main_offset),
              main_blocks);
    fill_q5_0((test_block_q5_0 *)((unsigned char *)model_map + tail_offset),
              tail_blocks);
    REQUIRE(ds4_gpu_set_model_map(model_map, model_size),
            "fused down profile model registration");

    const uint64_t mid_count = (uint64_t)assignments * DOWN_MID;
    const uint64_t main_count = (uint64_t)assignments * DOWN_MAIN;
    const uint64_t down_count = (uint64_t)assignments * HIDDEN;
    int32_t *ids = (int32_t *)malloc((size_t)assignments * sizeof(*ids));
    REQUIRE(ids, "fused down profile ids allocation");
    for (uint32_t i = 0; i < assignments; i++)
        ids[i] = (int32_t)((i * 37u + i / used) % experts);

    ds4_gpu_tensor *d_mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *d_packed = ds4_gpu_tensor_alloc(main_count * sizeof(float));
    ds4_gpu_tensor *d_ids = ds4_gpu_tensor_alloc(
        (uint64_t)assignments * sizeof(int32_t));
    ds4_gpu_tensor *d_down = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    REQUIRE(d_mid && d_packed && d_ids && d_down,
            "fused down profile GPU allocation");
    REQUIRE(ds4_gpu_tensor_fill_f32(d_mid, 0.25f, mid_count) &&
            ds4_gpu_tensor_write(
                d_ids, 0, ids, (uint64_t)assignments * sizeof(int32_t)) &&
            ds4_gpu_tensor_fill_f32(d_down, 0.0f, down_count),
            "fused down profile input initialization");
    REQUIRE(ds4_gpu_qwen4exp_pack_expert_down_main_tensor(
                d_packed, d_mid, assignments, DOWN_MID, DOWN_MAIN),
            "fused down profile pack launch");
    REQUIRE(ds4_gpu_qwen4exp_routed_down_fused_tensor(
                d_down, d_packed, d_mid, d_ids, model_map, model_size,
                main_offset, main_bytes, 13u, tail_offset, tail_bytes, 6u,
                assignments, DOWN_MID, DOWN_MAIN, DOWN_TAIL, HIDDEN, experts,
                tokens),
            "fused down production profile launch");
    REQUIRE(ds4_gpu_synchronize(), "fused down production profile sync");
    float edge[2] = {0.0f, 0.0f};
    REQUIRE(ds4_gpu_tensor_read(d_down, 0, &edge[0], sizeof(float)) &&
            ds4_gpu_tensor_read(
                d_down, (down_count - 1u) * sizeof(float),
                &edge[1], sizeof(float)) &&
            isfinite(edge[0]) && isfinite(edge[1]),
            "fused down production profile output");
    printf("NCU Qwen fused expert-down: tokens=%u assignments=%u experts=%u "
           "edge=%.6g/%.6g\n",
           tokens, assignments, experts, edge[0], edge[1]);

    ds4_gpu_tensor_free(d_down);
    ds4_gpu_tensor_free(d_ids);
    ds4_gpu_tensor_free(d_packed);
    ds4_gpu_tensor_free(d_mid);
    free(ids);
    ds4_gpu_unregister_model_map(model_map);
    free(model_map);
}

static void profile_q5_tail(void) {
    enum {
        tokens = 8025,
        experts = 512,
        used = 10,
        assignments = tokens * used,
    };
    const uint64_t weight_offset = 4096u;
    const uint64_t weight_blocks =
        (uint64_t)experts * HIDDEN * (DOWN_TAIL / QK_5_0);
    const uint64_t weight_bytes = weight_blocks * sizeof(test_block_q5_0);
    const uint64_t model_size =
        (weight_offset + weight_bytes + 4095u) & ~4095ull;
    void *model_map = NULL;
    REQUIRE(posix_memalign(&model_map, 4096u, (size_t)model_size) == 0,
            "Q5 tail profile model allocation");
    memset(model_map, 0, (size_t)model_size);
    fill_q5_0((test_block_q5_0 *)((unsigned char *)model_map + weight_offset),
              weight_blocks);
    REQUIRE(ds4_gpu_set_model_map(model_map, model_size),
            "Q5 tail profile model registration");

    const uint64_t mid_count = (uint64_t)assignments * DOWN_MID;
    const uint64_t down_count = (uint64_t)assignments * HIDDEN;
    int32_t *ids = (int32_t *)malloc((size_t)assignments * sizeof(*ids));
    REQUIRE(ids, "Q5 tail profile ids allocation");
    for (uint32_t i = 0; i < assignments; i++)
        ids[i] = (int32_t)((i * 37u + i / used) % experts);

    ds4_gpu_tensor *d_mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *d_ids = ds4_gpu_tensor_alloc(
        (uint64_t)assignments * sizeof(int32_t));
    ds4_gpu_tensor *d_down = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    REQUIRE(d_mid && d_ids && d_down, "Q5 tail profile GPU allocation");
    REQUIRE(ds4_gpu_tensor_fill_f32(d_mid, 0.25f, mid_count) &&
            ds4_gpu_tensor_write(
                d_ids, 0, ids, (uint64_t)assignments * sizeof(int32_t)) &&
            ds4_gpu_tensor_fill_f32(d_down, 0.0f, down_count),
            "Q5 tail profile input initialization");
    REQUIRE(setenv("DS4_QWEN_Q5_TAIL_EXPERT_MAJOR", "1", 1) == 0,
            "Q5 tail profile dispatch selection");
    REQUIRE(ds4_gpu_qwen4exp_q5_0_tail_accum_tensor(
                d_down, d_mid, d_ids, model_map, model_size,
                weight_offset, weight_bytes, assignments, DOWN_MID,
                DOWN_MAIN, DOWN_TAIL, HIDDEN, experts),
            "Q5 tail production profile launch");
    REQUIRE(ds4_gpu_synchronize(), "Q5 tail production profile sync");
    float edge[2] = {0.0f, 0.0f};
    REQUIRE(ds4_gpu_tensor_read(d_down, 0, &edge[0], sizeof(float)) &&
            ds4_gpu_tensor_read(
                d_down, (down_count - 1u) * sizeof(float),
                &edge[1], sizeof(float)) &&
            isfinite(edge[0]) && isfinite(edge[1]),
            "Q5 tail production profile output");
    printf("NCU Qwen Q5 tail: tokens=%u assignments=%u experts=%u "
           "edge=%.6g/%.6g\n",
           tokens, assignments, experts, edge[0], edge[1]);

    ds4_gpu_tensor_free(d_down);
    ds4_gpu_tensor_free(d_ids);
    ds4_gpu_tensor_free(d_mid);
    free(ids);
    ds4_gpu_unregister_model_map(model_map);
    free(model_map);
}

int main(void) {
    REQUIRE(ds4_gpu_init(), "CUDA init");
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable whole-map test copy");
    REQUIRE(setenv("DS4_QWEN_Q5_TAIL_EXPERT_MAJOR", "1", 1) == 0,
            "force expert-major Q5_0 tail test path");
    if (getenv("DS4_QWEN_PROFILE_Q5_TAIL")) {
        profile_q5_tail();
        ds4_gpu_cleanup();
        return 0;
    }
    if (getenv("DS4_QWEN_PROFILE_FUSED_DOWN")) {
        profile_fused_down();
        ds4_gpu_cleanup();
        return 0;
    }

    const uint64_t main_offset = 4096u;
    const uint64_t main_blocks =
        (uint64_t)DOWN_EXPERTS * HIDDEN * (DOWN_MAIN / QK_K);
    const uint64_t main_bytes = main_blocks * sizeof(test_block_q6_k);
    const uint64_t tail_offset = (main_offset + main_bytes + 4095u) & ~4095ull;
    const uint64_t tail_blocks =
        (uint64_t)DOWN_EXPERTS * HIDDEN * (DOWN_TAIL / QK_5_0);
    const uint64_t tail_bytes = tail_blocks * sizeof(test_block_q5_0);
    const uint64_t q8_tail_offset =
        (tail_offset + tail_bytes + 4095u) & ~4095ull;
    const uint64_t q8_tail_bytes = tail_blocks * sizeof(test_block_q8_0);
    const uint64_t q8_main_offset =
        (q8_tail_offset + q8_tail_bytes + 4095u) & ~4095ull;
    const uint64_t q8_main_blocks =
        (uint64_t)DOWN_EXPERTS * HIDDEN * Q8_MAIN_BLOCKS_PER_ROW;
    const uint64_t q8_main_bytes = q8_main_blocks * sizeof(test_block_q8_0);
    const uint64_t dense_tail_offset =
        (q8_main_offset + q8_main_bytes + 4095u) & ~4095ull;
    const uint64_t dense_tail_blocks =
        (uint64_t)HIDDEN * DENSE_TAIL_BLOCKS_PER_ROW;
    const uint64_t dense_tail_bytes =
        dense_tail_blocks * sizeof(test_block_q8_0);
    const uint64_t shexp_gate_offset =
        (dense_tail_offset + dense_tail_bytes + 4095u) & ~4095ull;
    const uint64_t shexp_blocks = (uint64_t)SHEXP_FF * SHEXP_K_BLOCKS;
    const uint64_t shexp_bytes = shexp_blocks * sizeof(test_block_q8_0);
    const uint64_t shexp_up_offset =
        (shexp_gate_offset + shexp_bytes + 4095u) & ~4095ull;
    const uint64_t q5k_pair_blocks =
        (uint64_t)PAIR_EXPERTS * PAIR_FF * (HIDDEN / QK_K);
    const uint64_t q5k_pair_bytes = q5k_pair_blocks * sizeof(test_block_q5_k);
    const uint64_t q5k_gate_offset =
        (shexp_up_offset + shexp_bytes + 4095u) & ~4095ull;
    const uint64_t q5k_up_offset =
        (q5k_gate_offset + q5k_pair_bytes + 4095u) & ~4095ull;
    const uint64_t q8_pair_blocks =
        (uint64_t)PAIR_EXPERTS * PAIR_FF * SHEXP_K_BLOCKS;
    const uint64_t q8_pair_bytes = q8_pair_blocks * sizeof(test_block_q8_0);
    const uint64_t q8_gate_offset =
        (q5k_up_offset + q5k_pair_bytes + 4095u) & ~4095ull;
    const uint64_t q8_up_offset =
        (q8_gate_offset + q8_pair_bytes + 4095u) & ~4095ull;
    /* Decode fixture: 16 experts so a top-10 routing exists. */
    const uint64_t dec_gu_blocks =
        (uint64_t)DECODE_EXPERTS * PAIR_FF * SHEXP_K_BLOCKS;
    const uint64_t dec_gu_bytes = dec_gu_blocks * sizeof(test_block_q8_0);
    const uint64_t dec_gate_offset =
        (q8_up_offset + q8_pair_bytes + 4095u) & ~4095ull;
    const uint64_t dec_up_offset =
        (dec_gate_offset + dec_gu_bytes + 4095u) & ~4095ull;
    const uint64_t dec_main_blocks =
        (uint64_t)DECODE_EXPERTS * HIDDEN * (DOWN_MAIN / QK_K);
    const uint64_t dec_main_bytes = dec_main_blocks * sizeof(test_block_q6_k);
    const uint64_t dec_main_offset =
        (dec_up_offset + dec_gu_bytes + 4095u) & ~4095ull;
    const uint64_t dec_tail_blocks =
        (uint64_t)DECODE_EXPERTS * HIDDEN * (DOWN_TAIL / QK_5_0);
    const uint64_t dec_tail_bytes = dec_tail_blocks * sizeof(test_block_q5_0);
    const uint64_t dec_tail_offset =
        (dec_main_offset + dec_main_bytes + 4095u) & ~4095ull;
    const uint64_t model_size =
        (dec_tail_offset + dec_tail_bytes + 4095u) & ~4095ull;
    void *model_map = NULL;
    REQUIRE(posix_memalign(&model_map, 4096u, (size_t)model_size) == 0,
            "aligned split-down model map");
    memset(model_map, 0, (size_t)model_size);
    fill_q6((test_block_q6_k *)((unsigned char *)model_map + main_offset),
            main_blocks);
    fill_q5_0((test_block_q5_0 *)((unsigned char *)model_map + tail_offset),
              tail_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + q8_tail_offset), tail_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + q8_main_offset),
              q8_main_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + dense_tail_offset),
              dense_tail_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + shexp_gate_offset),
              shexp_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + shexp_up_offset),
              shexp_blocks);
    fill_q5_k((test_block_q5_k *)(
                  (unsigned char *)model_map + q5k_gate_offset),
              q5k_pair_blocks);
    fill_q5_k((test_block_q5_k *)(
                  (unsigned char *)model_map + q5k_up_offset),
              q5k_pair_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + q8_gate_offset),
              q8_pair_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + q8_up_offset),
              q8_pair_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + dec_gate_offset),
              dec_gu_blocks);
    fill_q8_0((test_block_q8_0 *)(
                  (unsigned char *)model_map + dec_up_offset),
              dec_gu_blocks);
    fill_q6((test_block_q6_k *)((unsigned char *)model_map + dec_main_offset),
            dec_main_blocks);
    fill_q5_0((test_block_q5_0 *)((unsigned char *)model_map + dec_tail_offset),
              dec_tail_blocks);
    REQUIRE(ds4_gpu_set_model_map(model_map, model_size),
            "register split-down model map");

    puts("== Qwen3.8-Flash-Next MoE semantics ==");
    test_softmax_topk_router();
    test_shared_expert_gate();
    test_split_down(model_map, model_size, main_offset, main_bytes,
                    tail_offset, tail_bytes, q8_tail_offset, q8_tail_bytes);
    test_q5_tail_bank2(model_map, model_size, tail_offset, tail_bytes);
    test_fused_down(model_map, model_size, main_offset, main_bytes, 14u,
                    tail_offset, tail_bytes, 6u,
                    "Qwen fused Q6_K[512] + Q5_0[128] MMQ");
    test_fused_down(model_map, model_size, q8_main_offset, q8_main_bytes,
                    8u, q8_tail_offset, q8_tail_bytes, 8u,
                    "Qwen MTP fused Q8_0[512] + Q8_0[128] MMQ");
    test_dense_tail(model_map, model_size, dense_tail_offset,
                    dense_tail_bytes);
    test_shexp_pair(model_map, model_size, shexp_gate_offset,
                    shexp_up_offset);
    test_routed_pair(model_map, model_size, 13u, q5k_gate_offset,
                     q5k_up_offset, q5k_pair_bytes,
                     "Qwen edge-layer Q5_K gate/up worklist pair");
    test_routed_pair(model_map, model_size, 8u, q8_gate_offset,
                     q8_up_offset, q8_pair_bytes,
                     "Qwen MTP Q8_0 gate/up worklist pair");
    test_decode_vec(model_map, model_size, dec_gate_offset, dec_up_offset,
                    dec_gu_bytes, dec_main_offset, dec_main_bytes,
                    dec_tail_offset, dec_tail_bytes);

    REQUIRE(ds4_gpu_synchronize(), "final CUDA synchronization");
    ds4_gpu_unregister_model_map(model_map);
    free(model_map);
    ds4_gpu_cleanup();
    puts("all Qwen3.8 MoE semantic checks passed");
    return 0;
}
