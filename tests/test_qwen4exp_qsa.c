/* Production-geometry Qwen Sparse Attention parity. The context crosses the
 * 2,048-token budget so block scoring/top-512 selection is exercised rather
 * than degenerating to dense causal attention. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ROWS = 7,
    POS0 = 2065,
    CACHE_CAP = POS0 + ROWS,
    RATIO = 4,
    BLOCK_CAP = CACHE_CAP / RATIO,
    TOKEN_BUDGET = 2048,
    SELECTED_CAP = TOKEN_BUDGET + RATIO - 1,
    INDEX_HEADS = 4,
    INDEX_DIM = 128,
    INDEX_Q_DIM = INDEX_HEADS * INDEX_DIM,
    INDEX_QK_DIM = INDEX_Q_DIM + INDEX_DIM,
    HEADS = 24,
    KV_HEADS = 2,
    HEAD_DIM = 256,
    ROTARY_DIM = 64,
    YARN_ORIG_CTX = 262144,
    YARN_POS0 = 1000000 - ROWS,
    Q_DIM = HEADS * HEAD_DIM,
    Q_PROJ_DIM = Q_DIM * 2,
    KV_DIM = KV_HEADS * HEAD_DIM,
};

#define QWEN_ROPE_BASE 10000000.0f
#define QWEN_YARN_FACTOR 4.0f
#define QWEN_YARN_BETA_FAST 32.0f
#define QWEN_YARN_BETA_SLOW 1.0f

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

typedef struct {
    float score;
    uint32_t id;
} score_id;

static int score_desc(const void *pa, const void *pb) {
    const score_id *a = (const score_id *)pa;
    const score_id *b = (const score_id *)pb;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return a->id < b->id ? -1 : a->id > b->id;
}

static int u32_asc(const void *pa, const void *pb) {
    const uint32_t a = *(const uint32_t *)pa;
    const uint32_t b = *(const uint32_t *)pb;
    return a < b ? -1 : a > b;
}

static float sigmoid_ref(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x))
                     : expf(x) / (1.0f + expf(x));
}

static void upload_f32(ds4_gpu_tensor *dst, const float *src,
                       uint64_t count, const char *what) {
    REQUIRE(ds4_gpu_tensor_write(dst, 0, src, count * sizeof(*src)), what);
}

static void read_f32(const ds4_gpu_tensor *src, float *dst,
                     uint64_t count, const char *what) {
    REQUIRE(ds4_gpu_tensor_read(src, 0, dst, count * sizeof(*dst)), what);
}

static void compare_f32(const char *name, const float *got, const float *want,
                        uint64_t count, float max_abs_limit,
                        double rel_rms_limit) {
    double error2 = 0.0, ref2 = 0.0;
    float max_abs = 0.0f;
    uint64_t worst = 0u;
    for (uint64_t i = 0; i < count; i++) {
        const float error = fabsf(got[i] - want[i]);
        if (error > max_abs) { max_abs = error; worst = i; }
        error2 += (double)(got[i] - want[i]) * (got[i] - want[i]);
        ref2 += (double)want[i] * want[i];
    }
    const double rel_rms = sqrt(error2 / fmax(ref2, 1.0e-30));
    if (max_abs > max_abs_limit || rel_rms > rel_rms_limit) {
        fprintf(stderr,
                "FAIL: %s max %.9g @%llu got %.9g want %.9g rel_rms %.9g\n",
                name, max_abs, (unsigned long long)worst, got[worst],
                want[worst], rel_rms);
        exit(1);
    }
    printf("%-48s pass (max %.3g, rel RMS %.3g)\n",
           name, max_abs, rel_rms);
}

static void zero_rms_norm(float *x, const float *weight, uint32_t rows,
                          uint32_t width, uint32_t group) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t base = 0; base < width; base += group) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < group; d++) {
                const float v = x[(uint64_t)row * width + base + d];
                sum += v * v;
            }
            const float scale = 1.0f / sqrtf(sum / (float)group + 1.0e-6f);
            for (uint32_t d = 0; d < group; d++)
                x[(uint64_t)row * width + base + d] *=
                    scale * (1.0f + weight[d]);
        }
    }
}

static void rope_freqs(float *inv_freq, float *attn_factor, float factor) {
    const uint32_t half = ROTARY_DIM / 2u;
    const float low = floorf((float)ROTARY_DIM *
        logf((float)YARN_ORIG_CTX /
             (QWEN_YARN_BETA_FAST * 2.0f * (float)M_PI)) /
        (2.0f * logf(QWEN_ROPE_BASE)));
    const float high = ceilf((float)ROTARY_DIM *
        logf((float)YARN_ORIG_CTX /
             (QWEN_YARN_BETA_SLOW * 2.0f * (float)M_PI)) /
        (2.0f * logf(QWEN_ROPE_BASE)));
    *attn_factor = factor > 1.0f ? 0.1f * logf(factor) + 1.0f : 1.0f;
    for (uint32_t d = 0; d < half; d++) {
        const float base = 1.0f / powf(QWEN_ROPE_BASE,
            (2.0f * (float)d) / (float)ROTARY_DIM);
        if (factor <= 1.0f) {
            inv_freq[d] = base;
            continue;
        }
        const float ramp = fminf(1.0f, fmaxf(0.0f,
            ((float)d - fmaxf(low, 0.0f)) /
            fmaxf(0.001f, fminf(high, ROTARY_DIM - 1.0f) -
                             fmaxf(low, 0.0f))));
        inv_freq[d] = base * (1.0f - ramp) + base / factor * ramp;
    }
}

static void rope_first(float *x, uint32_t rows, uint32_t heads,
                       uint32_t head_dim, uint32_t pos0, float factor) {
    const uint32_t half = ROTARY_DIM / 2u;
    float inv_freq[ROTARY_DIM / 2u], attn_factor;
    rope_freqs(inv_freq, &attn_factor, factor);
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            float *v = x + ((uint64_t)row * heads + head) * head_dim;
            for (uint32_t d = 0; d < half; d++) {
                const float theta = (float)(pos0 + row) * inv_freq[d];
                const float c = cosf(theta) * attn_factor;
                const float s = sinf(theta) * attn_factor;
                const float x0 = v[d], x1 = v[d + half];
                v[d] = x0 * c - x1 * s;
                v[d + half] = x1 * c + x0 * s;
            }
        }
    }
}

static void mrope_first(float *x, uint32_t rows, uint32_t heads,
                        uint32_t head_dim, const int32_t *positions,
                        uint32_t pos0, float factor) {
    const uint32_t half = ROTARY_DIM / 2u;
    float inv_freq[ROTARY_DIM / 2u], attn_factor;
    rope_freqs(inv_freq, &attn_factor, factor);
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            float *v = x + ((uint64_t)row * heads + head) * head_dim;
            for (uint32_t d = 0; d < half; d++) {
                const float theta =
                    (float)positions[((uint64_t)pos0 + row) * 3u + d % 3u] *
                    inv_freq[d];
                const float c = cosf(theta) * attn_factor;
                const float s = sinf(theta) * attn_factor;
                const float x0 = v[d], x1 = v[d + half];
                v[d] = x0 * c - x1 * s;
                v[d + half] = x1 * c + x0 * s;
            }
        }
    }
}

static void pool_blocks(float *pooled, const float *raw,
                        const float *weight, const int32_t *positions,
                        float factor) {
    for (uint32_t block = 0; block < BLOCK_CAP; block++) {
        float *out = pooled + (uint64_t)block * INDEX_DIM;
        for (uint32_t d = 0; d < INDEX_DIM; d++) {
            float sum = 0.0f;
            for (uint32_t r = 0; r < RATIO; r++)
                sum += raw[((uint64_t)block * RATIO + r) * INDEX_DIM + d];
            out[d] = sum / (float)RATIO;
        }
        zero_rms_norm(out, weight, 1u, INDEX_DIM, INDEX_DIM);
        if (positions)
            mrope_first(out, 1u, 1u, INDEX_DIM,
                        positions, block * RATIO, factor);
        else
            rope_first(out, 1u, 1u, INDEX_DIM, block * RATIO, factor);
    }
}

static void check_yarn_rope(void) {
    float input[HEAD_DIM], want[HEAD_DIM], got[HEAD_DIM];
    for (uint32_t d = 0; d < HEAD_DIM; d++)
        input[d] = 0.31f * sinf((float)(d + 7u) * 0.037f);
    memcpy(want, input, sizeof(want));
    rope_first(want, 1u, 1u, HEAD_DIM, YARN_POS0, QWEN_YARN_FACTOR);

    ds4_gpu_tensor *device = ds4_gpu_tensor_alloc(sizeof(input));
    REQUIRE(device, "YaRN tensor allocation");
    upload_f32(device, input, HEAD_DIM, "YaRN input upload");
    REQUIRE(ds4_gpu_qwen4exp_rope_tensor(
                device, 1u, 1u, HEAD_DIM, ROTARY_DIM, YARN_POS0,
                QWEN_ROPE_BASE, QWEN_YARN_FACTOR, YARN_ORIG_CTX),
            "QSA 1M YaRN RoPE");
    read_f32(device, got, HEAD_DIM, "YaRN output download");
    compare_f32("QSA factor-4 YaRN at position 1M", got, want,
                HEAD_DIM, 5.0e-6f, 5.0e-6);
    ds4_gpu_tensor_free(device);
}

static void block_scores(float *scores, const float *query,
                         const float *pooled) {
    for (uint32_t row = 0; row < ROWS; row++) {
        const uint32_t visible_blocks = (POS0 + row + 1u) / RATIO;
        for (uint32_t block = 0; block < BLOCK_CAP; block++) {
            if (block >= visible_blocks) {
                scores[(uint64_t)row * BLOCK_CAP + block] = -INFINITY;
                continue;
            }
            float score = 0.0f;
            for (uint32_t head = 0; head < INDEX_HEADS; head++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < INDEX_DIM; d++)
                    dot += query[((uint64_t)row * INDEX_HEADS + head) *
                                 INDEX_DIM + d] *
                           pooled[(uint64_t)block * INDEX_DIM + d];
                score += fmaxf(dot, 0.0f);
            }
            scores[(uint64_t)row * BLOCK_CAP + block] =
                score / sqrtf((float)INDEX_DIM);
        }
    }
}

static void select_tokens(int32_t *tokens, uint32_t *counts,
                          const float *scores) {
    for (uint32_t row = 0; row < ROWS; row++) {
        int32_t *out = tokens + (uint64_t)row * SELECTED_CAP;
        for (uint32_t i = 0; i < SELECTED_CAP; i++) out[i] = -1;
        const uint32_t visible = POS0 + row + 1u;
        const uint32_t visible_blocks = visible / RATIO;
        score_id ranked[BLOCK_CAP];
        for (uint32_t block = 0; block < visible_blocks; block++) {
            ranked[block].score = scores[(uint64_t)row * BLOCK_CAP + block];
            ranked[block].id = block;
        }
        qsort(ranked, visible_blocks, sizeof(ranked[0]), score_desc);
        const uint32_t chosen = visible_blocks < 512u ? visible_blocks : 512u;
        uint32_t ids[512];
        for (uint32_t i = 0; i < chosen; i++) ids[i] = ranked[i].id;
        qsort(ids, chosen, sizeof(ids[0]), u32_asc);
        uint32_t count = 0u;
        for (uint32_t i = 0; i < chosen; i++)
            for (uint32_t r = 0; r < RATIO; r++)
                out[count++] = (int32_t)(ids[i] * RATIO + r);
        for (uint32_t token = visible_blocks * RATIO;
             token < visible; token++) out[count++] = (int32_t)token;
        counts[row] = count;
    }
}

static void attention_reference(float *out, const float *query,
                                const float *gate, const float *k_cache,
                                const float *v_cache, const int32_t *selected,
                                const uint32_t *counts) {
    float *weights = malloc(SELECTED_CAP * sizeof(float));
    REQUIRE(weights, "attention weight scratch");
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t head = 0; head < HEADS; head++) {
            const uint32_t kv_head = head / (HEADS / KV_HEADS);
            const float *q = query +
                ((uint64_t)row * HEADS + head) * HEAD_DIM;
            float maximum = -INFINITY;
            for (uint32_t slot = 0; slot < counts[row]; slot++) {
                const uint32_t token = (uint32_t)selected[
                    (uint64_t)row * SELECTED_CAP + slot];
                float dot = 0.0f;
                for (uint32_t d = 0; d < HEAD_DIM; d++)
                    dot += q[d] * k_cache[(uint64_t)token * KV_DIM +
                                           (uint64_t)kv_head * HEAD_DIM + d];
                weights[slot] = dot / sqrtf((float)HEAD_DIM);
                if (weights[slot] > maximum) maximum = weights[slot];
            }
            float sum = 0.0f;
            for (uint32_t slot = 0; slot < counts[row]; slot++) {
                weights[slot] = expf(weights[slot] - maximum);
                sum += weights[slot];
            }
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                float value = 0.0f;
                for (uint32_t slot = 0; slot < counts[row]; slot++) {
                    const uint32_t token = (uint32_t)selected[
                        (uint64_t)row * SELECTED_CAP + slot];
                    value += weights[slot] / sum *
                        v_cache[(uint64_t)token * KV_DIM +
                                (uint64_t)kv_head * HEAD_DIM + d];
                }
                const uint64_t at =
                    ((uint64_t)row * HEADS + head) * HEAD_DIM + d;
                out[at] = value * sigmoid_ref(gate[at]);
            }
        }
    }
    free(weights);
}

static void check_vision_attention(void) {
    enum { VROWS = 67, VHEADS = 16, VDIM = 72, VWIDTH = VHEADS * VDIM };
    float qkv[VROWS * 3 * VWIDTH];
    float want[VROWS * VWIDTH], got[VROWS * VWIDTH], naive[VROWS * VWIDTH];
    int32_t segment_start[VROWS], segment_end[VROWS];

    for (uint32_t row = 0; row < VROWS; row++) {
        segment_start[row] = row < 36u ? 0 : 36;
        segment_end[row] = row < 36u ? 36 : VROWS;
    }

    for (uint32_t row = 0; row < VROWS; row++) {
        for (uint32_t head = 0; head < VHEADS; head++) {
            float *q = qkv + (uint64_t)row * 3u * VWIDTH +
                       (uint64_t)head * VDIM;
            float *k = q + VWIDTH;
            float *v = k + VWIDTH;
            for (uint32_t d = 0; d < VDIM; d++) {
                q[d] = 0.31f * sinf((float)(row * 137u + head * 73u + d + 1u) * 0.071f);
                k[d] = 0.27f * cosf((float)(row * 109u + head * 59u + d + 3u) * 0.053f);
                v[d] = 0.23f * sinf((float)(row * 97u + head * 47u + d + 5u) * 0.037f);
            }
        }
    }
    for (uint32_t row = 0; row < VROWS; row++) {
        for (uint32_t head = 0; head < VHEADS; head++) {
            const float *q = qkv + (uint64_t)row * 3u * VWIDTH +
                             (uint64_t)head * VDIM;
            float score[VROWS], maximum = -INFINITY, sum = 0.0f;
            const uint32_t begin = (uint32_t)segment_start[row];
            const uint32_t end = (uint32_t)segment_end[row];
            for (uint32_t key_row = begin; key_row < end; key_row++) {
                const float *k = qkv + (uint64_t)key_row * 3u * VWIDTH +
                                 VWIDTH + (uint64_t)head * VDIM;
                float dot = 0.0f;
                for (uint32_t d = 0; d < VDIM; d++) dot += q[d] * k[d];
                score[key_row] = dot / sqrtf((float)VDIM);
                maximum = fmaxf(maximum, score[key_row]);
            }
            for (uint32_t key_row = begin; key_row < end; key_row++) {
                score[key_row] = expf(score[key_row] - maximum);
                sum += score[key_row];
            }
            for (uint32_t d = 0; d < VDIM; d++) {
                float value = 0.0f;
                for (uint32_t key_row = begin; key_row < end; key_row++) {
                    const float *v = qkv + (uint64_t)key_row * 3u * VWIDTH +
                                     2u * VWIDTH + (uint64_t)head * VDIM;
                    value += score[key_row] * v[d];
                }
                want[((uint64_t)row * VHEADS + head) * VDIM + d] = value / sum;
            }
        }
    }

    ds4_gpu_tensor *dqkv = ds4_gpu_tensor_alloc(sizeof(qkv));
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(sizeof(got));
    ds4_gpu_tensor *dstart = ds4_gpu_tensor_alloc(sizeof(segment_start));
    ds4_gpu_tensor *dend = ds4_gpu_tensor_alloc(sizeof(segment_end));
    REQUIRE(dqkv && dout && dstart && dend, "vision attention GPU allocation");
    REQUIRE(ds4_gpu_tensor_write(dqkv, 0, qkv, sizeof(qkv)),
            "vision attention QKV upload");
    REQUIRE(ds4_gpu_tensor_write(dstart, 0, segment_start, sizeof(segment_start)) &&
            ds4_gpu_tensor_write(dend, 0, segment_end, sizeof(segment_end)),
            "vision attention segment upload");
    unsetenv("DS4_CUDA_NO_QWEN_VISION_TILE");
    REQUIRE(ds4_gpu_qwen4exp_vision_attention_tensor(
                dout, dqkv, dstart, dend, VROWS, VHEADS, VDIM),
            "tiled vision attention launch");
    read_f32(dout, got, VROWS * VWIDTH, "vision attention download");
    compare_f32("Qwen vision full-attention softmax", got, want,
                VROWS * VWIDTH, 5.0e-5f, 5.0e-5);
    setenv("DS4_CUDA_NO_QWEN_VISION_TILE", "1", 1);
    REQUIRE(ds4_gpu_qwen4exp_vision_attention_tensor(
                dout, dqkv, dstart, dend, VROWS, VHEADS, VDIM),
            "naive vision attention launch");
    read_f32(dout, naive, VROWS * VWIDTH, "naive vision attention download");
    REQUIRE(memcmp(got, naive, sizeof(got)) == 0,
            "tiled/naive vision attention bit parity");
    unsetenv("DS4_CUDA_NO_QWEN_VISION_TILE");
    ds4_gpu_tensor_free(dend);
    ds4_gpu_tensor_free(dstart);
    ds4_gpu_tensor_free(dout);
    ds4_gpu_tensor_free(dqkv);
}

static void profile_qsa_attention(void) {
    enum {
        profile_rows = 8025,
        profile_selected = 2051,
        profile_cache = 8192,
    };
    const uint64_t q_count =
        (uint64_t)profile_rows * HEADS * HEAD_DIM;
    const uint64_t score_count =
        (uint64_t)profile_rows * HEADS * profile_selected;
    const uint64_t selected_count =
        (uint64_t)profile_rows * profile_selected;
    const uint64_t cache_count =
        (uint64_t)profile_cache * KV_HEADS * HEAD_DIM;
    int32_t *selected = malloc(selected_count * sizeof(*selected));
    uint32_t *counts = malloc(profile_rows * sizeof(*counts));
    REQUIRE(selected && counts, "QSA profile host allocation");
    for (uint32_t row = 0; row < profile_rows; row++) {
        counts[row] = profile_selected;
        for (uint32_t slot = 0; slot < profile_selected; slot++)
            selected[(uint64_t)row * profile_selected + slot] =
                (int32_t)slot;
    }

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *scores =
        ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *query = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *k_cache =
        ds4_gpu_tensor_alloc(cache_count * sizeof(float));
    ds4_gpu_tensor *v_cache =
        ds4_gpu_tensor_alloc(cache_count * sizeof(float));
    ds4_gpu_tensor *tokens =
        ds4_gpu_tensor_alloc(selected_count * sizeof(int32_t));
    ds4_gpu_tensor *dcounts =
        ds4_gpu_tensor_alloc(profile_rows * sizeof(uint32_t));
    REQUIRE(out && scores && query && gate && k_cache && v_cache && tokens &&
                dcounts,
            "QSA profile GPU allocation");
    REQUIRE(ds4_gpu_tensor_fill_f32(query, 0.03125f, q_count) &&
                ds4_gpu_tensor_fill_f32(gate, -0.25f, q_count) &&
                ds4_gpu_tensor_fill_f32(k_cache, 0.0625f, cache_count) &&
                ds4_gpu_tensor_fill_f32(v_cache, 0.125f, cache_count) &&
                ds4_gpu_tensor_write(tokens, 0, selected,
                                     selected_count * sizeof(*selected)) &&
                ds4_gpu_tensor_write(dcounts, 0, counts,
                                     profile_rows * sizeof(*counts)),
            "QSA profile input initialization");
    REQUIRE(ds4_gpu_qwen4exp_qsa_attention_tensor(
                out, scores, query, gate, k_cache, v_cache, tokens, dcounts,
                profile_rows, HEADS, KV_HEADS, HEAD_DIM, profile_selected,
                profile_cache),
            "QSA production profile launch");
    REQUIRE(ds4_gpu_synchronize(), "QSA production profile sync");
    float edge[2] = {0.0f, 0.0f};
    REQUIRE(ds4_gpu_tensor_read(out, 0, &edge[0], sizeof(float)) &&
                ds4_gpu_tensor_read(out, (q_count - 1u) * sizeof(float),
                                    &edge[1], sizeof(float)) &&
                isfinite(edge[0]) && isfinite(edge[1]),
            "QSA production profile output");
    printf("NCU Qwen QSA: rows=%u selected=%u edge=%.6g/%.6g\n",
           profile_rows, profile_selected, edge[0], edge[1]);

    ds4_gpu_tensor_free(dcounts);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(v_cache);
    ds4_gpu_tensor_free(k_cache);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(query);
    ds4_gpu_tensor_free(scores);
    ds4_gpu_tensor_free(out);
    free(counts);
    free(selected);
}

int main(void) {
    if (getenv("DS4_QWEN_PROFILE_QSA")) {
        REQUIRE(ds4_gpu_init(), "CUDA init");
        profile_qsa_attention();
        ds4_gpu_cleanup();
        return 0;
    }
    const uint64_t raw_count = (uint64_t)CACHE_CAP * INDEX_DIM;
    const uint64_t pooled_count = (uint64_t)BLOCK_CAP * INDEX_DIM;
    const uint64_t index_qk_count = (uint64_t)ROWS * INDEX_QK_DIM;
    const uint64_t index_q_count = (uint64_t)ROWS * INDEX_Q_DIM;
    const uint64_t score_count = (uint64_t)ROWS * BLOCK_CAP;
    const uint64_t token_count = (uint64_t)ROWS * SELECTED_CAP;
    const uint64_t qproj_count = (uint64_t)ROWS * Q_PROJ_DIM;
    const uint64_t q_count = (uint64_t)ROWS * Q_DIM;
    const uint64_t kv_rows_count = (uint64_t)ROWS * KV_DIM;
    const uint64_t kv_cache_count = (uint64_t)CACHE_CAP * KV_DIM;
    const uint64_t attn_score_count =
        (uint64_t)ROWS * HEADS * SELECTED_CAP;

    const uint64_t index_q_norm_offset = 4096u;
    const uint64_t index_k_norm_offset = index_q_norm_offset +
        INDEX_DIM * sizeof(float);
    const uint64_t q_norm_offset = index_k_norm_offset +
        INDEX_DIM * sizeof(float);
    const uint64_t k_norm_offset = q_norm_offset +
        HEAD_DIM * sizeof(float);
    const uint64_t model_size =
        (k_norm_offset + HEAD_DIM * sizeof(float) + 4095u) & ~4095ull;
    void *model_map = NULL;
    REQUIRE(posix_memalign(&model_map, 4096u, (size_t)model_size) == 0,
            "QSA aligned model fixture");
    memset(model_map, 0, (size_t)model_size);
    float *index_q_weight =
        (float *)((unsigned char *)model_map + index_q_norm_offset);
    float *index_k_weight =
        (float *)((unsigned char *)model_map + index_k_norm_offset);
    float *q_weight = (float *)((unsigned char *)model_map + q_norm_offset);
    float *k_weight = (float *)((unsigned char *)model_map + k_norm_offset);
    for (uint32_t d = 0; d < INDEX_DIM; d++)
        index_q_weight[d] =
            0.018f * sinf((float)(d + 3u) * 0.083f);
    for (uint32_t d = 0; d < INDEX_DIM; d++)
        index_k_weight[d] =
            0.022f * cosf((float)(d + 11u) * 0.067f);
    for (uint32_t d = 0; d < HEAD_DIM; d++)
        q_weight[d] = 0.025f * sinf((float)(d + 1u) * 0.071f);
    for (uint32_t d = 0; d < HEAD_DIM; d++)
        k_weight[d] = 0.021f * cosf((float)(d + 7u) * 0.053f);

#define HOST_ALLOC(name, count) float *name = malloc((count) * sizeof(float)); \
    REQUIRE(name, #name " allocation")
    HOST_ALLOC(raw, raw_count);
    HOST_ALLOC(raw_want, raw_count);
    HOST_ALLOC(pooled_want, pooled_count);
    HOST_ALLOC(pooled_got, pooled_count);
    HOST_ALLOC(index_qk, index_qk_count);
    HOST_ALLOC(index_query_want, index_q_count);
    HOST_ALLOC(index_query_got, index_q_count);
    HOST_ALLOC(scores_want, score_count);
    HOST_ALLOC(scores_got, score_count);
    HOST_ALLOC(q_projected, qproj_count);
    HOST_ALLOC(query_want, q_count);
    HOST_ALLOC(query_got, q_count);
    HOST_ALLOC(gate_want, q_count);
    HOST_ALLOC(gate_got, q_count);
    HOST_ALLOC(key, kv_rows_count);
    HOST_ALLOC(value, kv_rows_count);
    HOST_ALLOC(key_want, kv_rows_count);
    HOST_ALLOC(k_cache, kv_cache_count);
    HOST_ALLOC(v_cache, kv_cache_count);
    HOST_ALLOC(k_cache_want, kv_cache_count);
    HOST_ALLOC(v_cache_want, kv_cache_count);
    HOST_ALLOC(attn_want, q_count);
    HOST_ALLOC(attn_got, q_count);
#undef HOST_ALLOC
    int32_t *tokens_want = malloc(token_count * sizeof(int32_t));
    int32_t *tokens_got = malloc(token_count * sizeof(int32_t));
    int32_t *mrope_positions =
        malloc((uint64_t)CACHE_CAP * 3u * sizeof(int32_t));
    uint32_t counts_want[ROWS], counts_got[ROWS];
    REQUIRE(tokens_want && tokens_got && mrope_positions,
            "QSA token/position allocation");
    for (uint32_t i = 0; i < CACHE_CAP; i++) {
        mrope_positions[3u * i] = (int32_t)(i % 97u);
        mrope_positions[3u * i + 1u] = (int32_t)((2u * i + 7u) % 83u);
        mrope_positions[3u * i + 2u] = (int32_t)((3u * i + 11u) % 71u);
    }

    for (uint64_t i = 0; i < raw_count; i++)
        raw[i] = 0.31f * sinf((float)(i + 5u) * 0.0031f) +
                 0.13f * cosf((float)(i + 17u) * 0.0097f) +
                 (float)((int)(i % 23u) - 11) * 0.001f;
    memcpy(raw_want, raw, raw_count * sizeof(float));
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t d = 0; d < INDEX_Q_DIM; d++) {
            const float v = 0.27f * sinf((float)((uint64_t)row *
                INDEX_Q_DIM + d + 29u) * 0.0081f) +
                0.09f * cosf((float)(d + 13u) * 0.037f);
            index_qk[(uint64_t)row * INDEX_QK_DIM + d] = v;
            index_query_want[(uint64_t)row * INDEX_Q_DIM + d] = v;
        }
        for (uint32_t d = 0; d < INDEX_DIM; d++) {
            const float v = 0.33f * cosf((float)((uint64_t)row *
                INDEX_DIM + d + 7u) * 0.011f) +
                0.05f * sinf((float)(d + 41u) * 0.043f);
            index_qk[(uint64_t)row * INDEX_QK_DIM + INDEX_Q_DIM + d] = v;
            raw_want[((uint64_t)POS0 + row) * INDEX_DIM + d] = v;
        }
    }
    zero_rms_norm(index_query_want, index_q_weight,
                  ROWS, INDEX_Q_DIM, INDEX_DIM);
    rope_first(index_query_want, ROWS, INDEX_HEADS, INDEX_DIM, POS0, 1.0f);
    pool_blocks(pooled_want, raw_want, index_k_weight, NULL, 1.0f);
    block_scores(scores_want, index_query_want, pooled_want);
    select_tokens(tokens_want, counts_want, scores_want);

    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t head = 0; head < HEADS; head++) {
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                const uint64_t at =
                    ((uint64_t)row * HEADS + head) * HEAD_DIM + d;
                const uint64_t src =
                    ((uint64_t)row * HEADS + head) * 2u * HEAD_DIM + d;
                query_want[at] = 0.24f * sinf((float)(at + 3u) * 0.0047f) +
                    0.08f * cosf((float)(d + 37u) * 0.019f);
                gate_want[at] = 0.7f * cosf((float)(at + 11u) * 0.0039f);
                q_projected[src] = query_want[at];
                q_projected[src + HEAD_DIM] = gate_want[at];
            }
        }
    }
    for (uint64_t i = 0; i < kv_rows_count; i++) {
        key[i] = 0.29f * sinf((float)(i + 7u) * 0.0067f) +
                 0.11f * cosf((float)(i + 31u) * 0.017f);
        value[i] = 0.26f * cosf((float)(i + 5u) * 0.0053f) -
                   0.07f * sinf((float)(i + 23u) * 0.021f);
    }
    memcpy(key_want, key, kv_rows_count * sizeof(float));
    zero_rms_norm(query_want, q_weight, ROWS, Q_DIM, HEAD_DIM);
    zero_rms_norm(key_want, k_weight, ROWS, KV_DIM, HEAD_DIM);
    rope_first(query_want, ROWS, HEADS, HEAD_DIM, POS0, 1.0f);
    rope_first(key_want, ROWS, KV_HEADS, HEAD_DIM, POS0, 1.0f);
    for (uint64_t i = 0; i < kv_cache_count; i++) {
        k_cache[i] = 0.18f * sinf((float)(i + 3u) * 0.0019f);
        v_cache[i] = 0.22f * cosf((float)(i + 19u) * 0.0023f);
    }
    memcpy(k_cache_want, k_cache, kv_cache_count * sizeof(float));
    memcpy(v_cache_want, v_cache, kv_cache_count * sizeof(float));
    for (uint32_t row = 0; row < ROWS; row++) {
        memcpy(k_cache_want + ((uint64_t)POS0 + row) * KV_DIM,
               key_want + (uint64_t)row * KV_DIM,
               KV_DIM * sizeof(float));
        memcpy(v_cache_want + ((uint64_t)POS0 + row) * KV_DIM,
               value + (uint64_t)row * KV_DIM,
               KV_DIM * sizeof(float));
    }
    attention_reference(attn_want, query_want, gate_want,
                        k_cache_want, v_cache_want,
                        tokens_want, counts_want);

    REQUIRE(ds4_gpu_init(), "CUDA init");
    check_yarn_rope();
    check_vision_attention();
    REQUIRE(unsetenv("DS4_CUDA_COPY_MODEL") == 0,
            "disable QSA fixture whole copy");
    REQUIRE(ds4_gpu_set_model_map(model_map, model_size),
            "register QSA model fixture");
#define DEV_ALLOC(name, count, type) \
    ds4_gpu_tensor *name = ds4_gpu_tensor_alloc((uint64_t)(count) * sizeof(type)); \
    REQUIRE(name, #name " GPU allocation")
    DEV_ALLOC(draw, raw_count, float);
    DEV_ALLOC(dpool, pooled_count, float);
    DEV_ALLOC(dindex_qk, index_qk_count, float);
    DEV_ALLOC(dindex_q, index_q_count, float);
    DEV_ALLOC(dmrope, (uint64_t)CACHE_CAP * 3u, int32_t);
    DEV_ALLOC(dscores, score_count, float);
    DEV_ALLOC(dblocks, (uint64_t)ROWS * 512u, uint32_t);
    DEV_ALLOC(dtokens, token_count, int32_t);
    DEV_ALLOC(dcounts, ROWS, uint32_t);
    DEV_ALLOC(dqproj, qproj_count, float);
    DEV_ALLOC(dquery, q_count, float);
    DEV_ALLOC(dgate, q_count, float);
    DEV_ALLOC(dkey, kv_rows_count, float);
    DEV_ALLOC(dvalue, kv_rows_count, float);
    DEV_ALLOC(dkcache, kv_cache_count, float);
    DEV_ALLOC(dvcache, kv_cache_count, float);
    DEV_ALLOC(dattn_scores, attn_score_count, float);
    DEV_ALLOC(dout, q_count, float);
#undef DEV_ALLOC
    upload_f32(draw, raw, raw_count, "raw index cache upload");
    upload_f32(dindex_qk, index_qk, index_qk_count, "index qk upload");
    REQUIRE(ds4_gpu_tensor_write(
                dmrope, 0, mrope_positions,
                (uint64_t)CACHE_CAP * 3u * sizeof(int32_t)),
            "M-RoPE positions upload");
    REQUIRE(ds4_gpu_qwen4exp_qsa_split_index_tensor(
                dindex_q, draw, dindex_qk, ROWS, POS0, CACHE_CAP,
                INDEX_HEADS, INDEX_DIM),
            "QSA index split/store");
    REQUIRE(ds4_gpu_qwen4exp_shared_group_rms_norm_rows_tensor(
                dindex_q, dindex_q, model_map, model_size,
                index_q_norm_offset, INDEX_Q_DIM, INDEX_DIM, ROWS, 1.0e-6f),
            "QSA index query norm");
    REQUIRE(ds4_gpu_qwen4exp_rope_tensor(
                dindex_q, ROWS, INDEX_HEADS, INDEX_DIM,
                ROTARY_DIM, POS0, QWEN_ROPE_BASE, 1.0f, YARN_ORIG_CTX),
            "QSA index query RoPE");
    REQUIRE(ds4_gpu_qwen4exp_qsa_pool_blocks_tensor(
                dpool, draw, NULL, model_map, model_size, index_k_norm_offset,
                0u, BLOCK_CAP, BLOCK_CAP, RATIO, INDEX_DIM,
                ROTARY_DIM, QWEN_ROPE_BASE, 1.0f, YARN_ORIG_CTX, 1.0e-6f),
            "QSA pooled index blocks");
    REQUIRE(ds4_gpu_qwen4exp_qsa_block_scores_tensor(
                dscores, dindex_q, dpool, ROWS, POS0, BLOCK_CAP,
                INDEX_HEADS, INDEX_DIM, RATIO),
            "QSA block scores");
    REQUIRE(ds4_gpu_indexer_topk_tensor(
                dblocks, dscores, BLOCK_CAP, ROWS, 512u, 0u, UINT32_MAX),
            "QSA exact top-512 blocks");
    REQUIRE(ds4_gpu_qwen4exp_qsa_expand_selection_tensor(
                dtokens, dcounts, dblocks, ROWS, POS0, BLOCK_CAP,
                512u, RATIO, TOKEN_BUDGET),
            "QSA block-to-token expansion");

    read_f32(dindex_q, index_query_got, index_q_count,
             "QSA index query download");
    compare_f32("QSA index query norm + first-64 RoPE",
                index_query_got, index_query_want, index_q_count,
                4.0e-6f, 4.0e-6);
    read_f32(dpool, pooled_got, pooled_count, "QSA pool download");
    compare_f32("QSA four-token pooled key cache", pooled_got, pooled_want,
                pooled_count, 4.0e-6f, 4.0e-6);

    pool_blocks(pooled_want, raw_want, index_k_weight, mrope_positions,
                QWEN_YARN_FACTOR);
    REQUIRE(ds4_gpu_qwen4exp_qsa_pool_blocks_tensor(
                dpool, draw, dmrope, model_map, model_size,
                index_k_norm_offset, 0u, BLOCK_CAP, BLOCK_CAP, RATIO,
                INDEX_DIM, ROTARY_DIM, QWEN_ROPE_BASE, QWEN_YARN_FACTOR,
                YARN_ORIG_CTX, 1.0e-6f),
            "QSA M-RoPE pooled index blocks");
    read_f32(dpool, pooled_got, pooled_count, "QSA M-RoPE pool download");
    compare_f32("QSA M-RoPE pooled key cache", pooled_got, pooled_want,
                pooled_count, 4.0e-6f, 4.0e-6);

    for (uint32_t row = 0; row < ROWS; row++)
        memcpy(index_query_want + (uint64_t)row * INDEX_Q_DIM,
               index_qk + (uint64_t)row * INDEX_QK_DIM,
               INDEX_Q_DIM * sizeof(float));
    memcpy(index_query_got, index_query_want,
           index_q_count * sizeof(float));
    zero_rms_norm(index_query_want, index_q_weight,
                  ROWS, INDEX_Q_DIM, INDEX_DIM);
    zero_rms_norm(index_query_got, index_q_weight,
                  ROWS, INDEX_Q_DIM, INDEX_DIM);
    mrope_first(index_query_want, ROWS, INDEX_HEADS, INDEX_DIM,
                mrope_positions, POS0, QWEN_YARN_FACTOR);
    upload_f32(dindex_q, index_query_got, index_q_count,
               "QSA M-RoPE input upload");
    REQUIRE(ds4_gpu_qwen4exp_mrope_tensor(
                dindex_q, dmrope, ROWS, INDEX_HEADS, INDEX_DIM,
                ROTARY_DIM, POS0, QWEN_ROPE_BASE, QWEN_YARN_FACTOR,
                YARN_ORIG_CTX),
            "QSA interleaved M-RoPE");
    read_f32(dindex_q, index_query_got, index_q_count,
             "QSA M-RoPE query download");
    compare_f32("QSA interleaved THW M-RoPE", index_query_got,
                index_query_want, index_q_count, 4.0e-6f, 4.0e-6);
    read_f32(dscores, scores_got, score_count, "QSA scores download");
    for (uint32_t row = 0; row < ROWS; row++) {
        const uint32_t visible_blocks = (POS0 + row + 1u) / RATIO;
        compare_f32("QSA visible block scores",
                    scores_got + (uint64_t)row * BLOCK_CAP,
                    scores_want + (uint64_t)row * BLOCK_CAP,
                    visible_blocks, 3.0e-5f, 4.0e-6);
    }
    REQUIRE(ds4_gpu_tensor_read(dtokens, 0, tokens_got,
                                token_count * sizeof(int32_t)),
            "QSA selected token download");
    REQUIRE(ds4_gpu_tensor_read(dcounts, 0, counts_got,
                                sizeof(counts_got)),
            "QSA selected counts download");
    for (uint32_t row = 0; row < ROWS; row++) {
        REQUIRE(counts_got[row] == counts_want[row],
                "QSA selected count exact");
        REQUIRE(memcmp(tokens_got + (uint64_t)row * SELECTED_CAP,
                       tokens_want + (uint64_t)row * SELECTED_CAP,
                       SELECTED_CAP * sizeof(int32_t)) == 0,
                "QSA selected token ids exact");
    }
    printf("%-48s pass (7 rows, counts %u..%u)\n",
           "QSA top-512 blocks -> chronological tokens",
           counts_got[0], counts_got[ROWS - 1u]);

    upload_f32(dqproj, q_projected, qproj_count, "QSA q projection upload");
    upload_f32(dkey, key, kv_rows_count, "QSA key upload");
    upload_f32(dvalue, value, kv_rows_count, "QSA value upload");
    upload_f32(dkcache, k_cache, kv_cache_count, "QSA K cache upload");
    upload_f32(dvcache, v_cache, kv_cache_count, "QSA V cache upload");
    REQUIRE(ds4_gpu_qwen4exp_qsa_split_q_gate_tensor(
                dquery, dgate, dqproj, ROWS, HEADS, HEAD_DIM),
            "QSA main query/gate split");
    REQUIRE(ds4_gpu_qwen4exp_shared_group_rms_norm_rows_tensor(
                dquery, dquery, model_map, model_size, q_norm_offset,
                Q_DIM, HEAD_DIM, ROWS, 1.0e-6f),
            "QSA main query norm");
    REQUIRE(ds4_gpu_qwen4exp_shared_group_rms_norm_rows_tensor(
                dkey, dkey, model_map, model_size, k_norm_offset,
                KV_DIM, HEAD_DIM, ROWS, 1.0e-6f),
            "QSA main key norm");
    REQUIRE(ds4_gpu_qwen4exp_rope_tensor(
                dquery, ROWS, HEADS, HEAD_DIM, ROTARY_DIM,
                POS0, QWEN_ROPE_BASE, 1.0f, YARN_ORIG_CTX),
            "QSA main query RoPE");
    REQUIRE(ds4_gpu_qwen4exp_rope_tensor(
                dkey, ROWS, KV_HEADS, HEAD_DIM, ROTARY_DIM,
                POS0, QWEN_ROPE_BASE, 1.0f, YARN_ORIG_CTX),
            "QSA main key RoPE");
    REQUIRE(ds4_gpu_qwen4exp_qsa_store_kv_tensor(
                dkcache, dvcache, dkey, dvalue, ROWS, POS0, CACHE_CAP,
                KV_HEADS, HEAD_DIM),
            "QSA KV store");
    REQUIRE(ds4_gpu_qwen4exp_qsa_attention_tensor(
                dout, dattn_scores, dquery, dgate, dkcache, dvcache,
                dtokens, dcounts, ROWS, HEADS, KV_HEADS, HEAD_DIM,
                SELECTED_CAP, CACHE_CAP),
            "QSA selected attention");
    read_f32(dquery, query_got, q_count, "QSA query download");
    read_f32(dgate, gate_got, q_count, "QSA gate download");
    compare_f32("QSA main query norm + first-64 RoPE",
                query_got, query_want, q_count, 5.0e-6f, 5.0e-6);
    compare_f32("QSA interleaved output-gate split",
                gate_got, gate_want, q_count, 0.0f, 0.0);
    read_f32(dout, attn_got, q_count, "QSA attention download");
    compare_f32("QSA selected GQA softmax + sigmoid gate",
                attn_got, attn_want, q_count, 2.0e-5f, 2.0e-5);
    /* Seven rows take the decode-width split reduce (17 chunks of the
     * 2,051-slot cap); the serial gqa12 kernel behind the kill switch must
     * agree to fp32 reordering noise. */
    {
        float *serial_got = malloc(q_count * sizeof(*serial_got));
        REQUIRE(serial_got, "QSA serial reduce allocation");
        REQUIRE(setenv("DS4_QWEN_QSA_NO_SPLIT_REDUCE", "1", 1) == 0,
                "QSA split reduce kill switch");
        REQUIRE(ds4_gpu_qwen4exp_qsa_attention_tensor(
                    dout, dattn_scores, dquery, dgate, dkcache, dvcache,
                    dtokens, dcounts, ROWS, HEADS, KV_HEADS, HEAD_DIM,
                    SELECTED_CAP, CACHE_CAP),
                "QSA selected attention (serial reduce)");
        unsetenv("DS4_QWEN_QSA_NO_SPLIT_REDUCE");
        read_f32(dout, serial_got, q_count, "QSA serial attention download");
        compare_f32("QSA split reduce vs serial gqa12 reduce",
                    attn_got, serial_got, q_count, 4.0e-6f, 2.0e-6);
        free(serial_got);
    }

    REQUIRE(!ds4_gpu_qwen4exp_qsa_split_q_gate_tensor(
                dquery, dquery, dqproj, ROWS, HEADS, HEAD_DIM),
            "QSA split rejects aliased outputs");
    REQUIRE(!ds4_gpu_qwen4exp_qsa_store_kv_tensor(
                dkcache, dkcache, dkey, dvalue, ROWS, POS0, CACHE_CAP,
                KV_HEADS, HEAD_DIM),
            "QSA KV store rejects aliased caches");
    REQUIRE(ds4_gpu_synchronize(), "final QSA synchronization");

    ds4_gpu_tensor *all[] = {
        dout, dattn_scores, dvcache, dkcache, dvalue, dkey, dgate, dquery,
        dqproj, dcounts, dtokens, dblocks, dscores, dindex_q, dindex_qk,
        dmrope, dpool, draw,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++)
        ds4_gpu_tensor_free(all[i]);
    ds4_gpu_unregister_model_map(model_map);
    ds4_gpu_cleanup();
    free(attn_got); free(attn_want); free(v_cache_want); free(k_cache_want);
    free(v_cache); free(k_cache); free(key_want); free(value); free(key);
    free(gate_got); free(gate_want); free(query_got); free(query_want);
    free(q_projected); free(mrope_positions); free(tokens_got); free(tokens_want);
    free(scores_got); free(scores_want); free(index_query_got);
    free(index_query_want); free(index_qk); free(pooled_got);
    free(pooled_want); free(raw_want); free(raw); free(model_map);
    puts("all Qwen3.8 QSA checks passed");
    return 0;
}
