/* Small native gates; no model weights, owner, or inference context. */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/ling3vl_primitives.cuh"
#include "../ds4_ling3vl_rope.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)

template <class T> static T *upload(const T *data, size_t n) {
    T *p;
    CUDA(cudaMalloc(&p, n * sizeof(T)));
    CUDA(cudaMemcpy(p, data, n * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

static void close(float got, double want, double tol = 1e-6) {
    if (!std::isfinite(got) || fabs(got - want) > tol * (1 + fabs(want))) {
        fprintf(stderr, "got %.9g expected %.12g\n", got, want);
        exit(1);
    }
}

/* Host mirror of the published routing rule: sigmoid, bias-corrected group
 * and expert selection, unbiased weights, renormalized and scaled. */
static void reference_router(const float *logits, const float *bias,
                             int *ids, float *weights, float scale) {
    std::vector<double> prob(LING_EXPERTS), score(LING_EXPERTS);
    for (unsigned e = 0; e < LING_EXPERTS; e++) {
        prob[e] = 1.0 / (1.0 + exp(-(double)logits[e]));
        score[e] = prob[e] + bias[e];
    }
    std::vector<double> group(LING_GROUPS);
    for (unsigned g = 0; g < LING_GROUPS; g++) {
        double best = -INFINITY, second = -INFINITY;
        for (unsigned i = 0; i < LING_PER_GROUP; i++) {
            const double v = score[g * LING_PER_GROUP + i];
            if (v > best) { second = best; best = v; }
            else if (v > second) { second = v; }
        }
        group[g] = best + second;
    }
    std::vector<bool> keep(LING_GROUPS, false);
    std::vector<double> left = group;
    for (unsigned k = 0; k < LING_GROUPS_USED; k++) {
        unsigned best = 0;
        for (unsigned g = 1; g < LING_GROUPS; g++) {
            if (left[g] > left[best]) { best = g; }
        }
        keep[best] = true;
        left[best] = -INFINITY;
    }
    std::vector<double> masked(LING_EXPERTS);
    for (unsigned e = 0; e < LING_EXPERTS; e++) {
        masked[e] = keep[e / LING_PER_GROUP] ? score[e] : -INFINITY;
    }
    double sum = 0.0;
    for (unsigned k = 0; k < LING_USED; k++) {
        unsigned best = 0;
        for (unsigned e = 1; e < LING_EXPERTS; e++) {
            if (masked[e] > masked[best]) { best = e; }
        }
        ids[k] = (int)best;
        weights[k] = (float)prob[best];
        sum += prob[best];
        masked[best] = -INFINITY;
    }
    if (sum < 6.103515625e-5) { sum = 6.103515625e-5; }
    for (unsigned k = 0; k < LING_USED; k++) {
        weights[k] = (float)(prob[ids[k]] / sum * (double)scale);
    }
}

static void router(cudaStream_t stream) {
    enum { ROWS = 3u };
    std::vector<float> logits(ROWS * LING_EXPERTS), bias(LING_EXPERTS);
    unsigned seed = 12345u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (float)((double)(seed >> 8) / 16777216.0);
    };
    for (unsigned i = 0; i < logits.size(); i++) { logits[i] = next() * 8.0f - 4.0f; }
    /* A strong bias on one low group proves selection uses the biased score
     * while the emitted weight stays unbiased. */
    for (unsigned e = 0; e < LING_EXPERTS; e++) { bias[e] = next() * 0.2f - 0.1f; }
    for (unsigned e = 0; e < LING_PER_GROUP; e++) { bias[e] = 5.0f; }

    float *d_logits = upload(logits.data(), logits.size());
    float *d_bias = upload(bias.data(), bias.size());
    int *d_ids;
    float *d_w;
    CUDA(cudaMalloc(&d_ids, ROWS * LING_USED * sizeof(int)));
    CUDA(cudaMalloc(&d_w, ROWS * LING_USED * sizeof(float)));
    ling3vl_router<<<ROWS, 256, 0, stream>>>(d_ids, d_w, d_logits, d_bias, 2.5f);
    CUDA(cudaStreamSynchronize(stream));

    std::vector<int> ids(ROWS * LING_USED);
    std::vector<float> w(ROWS * LING_USED);
    CUDA(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(w.data(), d_w, w.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned r = 0; r < ROWS; r++) {
        int want_ids[LING_USED];
        float want_w[LING_USED];
        reference_router(logits.data() + (size_t)r * LING_EXPERTS, bias.data(),
                         want_ids, want_w, 2.5f);
        double sum = 0.0;
        for (unsigned k = 0; k < LING_USED; k++) {
            CHECK(ids[r * LING_USED + k] == want_ids[k]);
            close(w[r * LING_USED + k], want_w[k]);
            sum += w[r * LING_USED + k];
        }
        /* norm_topk_prob then routed_scaling_factor. */
        close((float)sum, 2.5, 1e-5);
        /* The biased group has to win, and its experts come first. */
        CHECK(ids[r * LING_USED] < (int)LING_PER_GROUP);
    }
    CUDA(cudaFree(d_ids));
    CUDA(cudaFree(d_w));
    CUDA(cudaFree(d_bias));
    CUDA(cudaFree(d_logits));
    printf("router OK\n");
}

/* Independent double-precision YaRN values from the Transformers v4.57.1
 * equations: correction dimensions [13, 21], factor 2, beta 32/1. */
static void rope_table(void) {
    const float expected[] = {
        1.0f, 0.614020362104f, 0.377021005078f, 0.231498574059f,
        0.142144838270f, 0.0872798250656f, 0.0535915897911f, 0.0329063273693f,
        0.0202051550468f, 0.0124063766182f, 0.00761776786348f, 0.00467746458196f,
        0.00287205849634f, 0.00176350239791f, 0.00101514973212f, 0.000581767765701f,
        0.000331701736002f, 0.000188004572354f, 0.000105818749294f, 5.90680606901e-5f,
        3.26420928123e-5f, 1.78159196875e-5f, 1.09393374577e-5f, 6.71697594697e-6f,
        4.12436000320e-6f, 2.53244102261e-6f, 1.55497035371e-6f, 9.54783459645e-7f,
        5.86256485622e-7f, 3.59973419587e-7f, 2.21031009442e-7f, 1.35717540454e-7f,
    };
    float inv[32], attn;
    for (unsigned ctx : {1u, 65536u, 131072u}) {
        CHECK(ling3vl_rope_table(inv, &attn, ctx));
        CHECK(attn == 1.0f);
        for (unsigned j = 0; j < 32u; j++) {
            CHECK(inv[j] == (float)pow(6000000.0, -2.0 * j / 64.0));
        }
    }
    for (unsigned ctx : {131073u, 200000u, 262144u}) {
        CHECK(ling3vl_rope_table(inv, &attn, ctx));
        close(attn, 1.0693147180559945, 1e-7);
        for (unsigned j = 0; j < 32u; j++) {
            CHECK(fabs(inv[j] / expected[j] - 1.0f) < 2e-7f);
        }
    }
    CHECK(!ling3vl_rope_table(inv, &attn, 0u));
    CHECK(!ling3vl_rope_table(inv, &attn, 262145u));
    printf("rope_table OK\n");
}

/* Text rows set all three axes equal, so M-RoPE must equal 1-D RoPE; an image
 * row separates them. Pairs are adjacent, not half-offset. */
static void mrope(cudaStream_t stream, unsigned ctx, unsigned stride) {
    enum { HEADS = 2u, ROWS = 4u, ROTARY = 64u, HALF = ROTARY / 2u };
    const unsigned STRIDE = stride, OFFSET = stride - ROTARY;
    std::vector<float> x(ROWS * HEADS * STRIDE, 0.0f);
    for (unsigned i = 0; i < x.size(); i++) { x[i] = (float)((i % 7) + 1) * 0.125f; }
    std::vector<float> inv(HALF);
    float attn;
    CHECK(ling3vl_rope_table(inv.data(), &attn, ctx));
    /* Text and distinct image axes, including the native and YaRN edges. */
    const int32_t pos[ROWS * 3] = {11, 11, 11, 20, 23, 29,
        131071, 131071, 131071, 262143, 200000, 131073};
    std::vector<float> before = x;

    float *d_x = upload(x.data(), x.size());
    float *d_inv = upload(inv.data(), inv.size());
    int32_t *d_pos = upload(pos, ROWS * 3);
    const uint64_t pairs = (uint64_t)ROWS * HEADS * HALF;
    ling3vl_mrope<<<(pairs + 255u) / 256u, 256, 0, stream>>>(
        d_x, d_pos, d_inv, HEADS, STRIDE, OFFSET, HALF, 8u, 8u + 12u, pairs, attn);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(x.data(), d_x, x.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned h = 0; h < HEADS; h++) {
            const size_t base = ((size_t)r * HEADS + h) * STRIDE + OFFSET;
            for (unsigned j = 0; j < HALF; j++) {
                const unsigned axis = j < 8u ? 0u : (j < 20u ? 1u : 2u);
                const double theta = (double)pos[r * 3 + axis] * (double)inv[j];
                const double c = cos(theta) * attn, s = sin(theta) * attn;
                const double x0 = before[base + 2 * j], x1 = before[base + 2 * j + 1];
                close(x[base + 2 * j], x0 * c - x1 * s, 1e-5);
                close(x[base + 2 * j + 1], x0 * s + x1 * c, 1e-5);
            }
            /* The nope half is untouched. */
            for (unsigned d = 0; d < OFFSET; d++) {
                const size_t i = ((size_t)r * HEADS + h) * STRIDE + d;
                close(x[i], before[i]);
            }
        }
    }
    CUDA(cudaFree(d_pos));
    CUDA(cudaFree(d_inv));
    CUDA(cudaFree(d_x));
    printf("mrope ctx=%u stride=%u OK\n", ctx, stride);
}

/* The fused kv_a_mqa row is 576 wide; the latent RMSNorm covers only the
 * leading 512.  Advancing the input by 512 mixes the previous row's 64-value
 * RoPE tail into the next latent. */
static void rms_norm_strided(cudaStream_t stream) {
    enum { DIM = 512u, STRIDE = 576u, ROWS = 2u };
    const float eps = 1e-5f;
    std::vector<float> in((size_t)ROWS * STRIDE, 0.0f);
    std::vector<float> weight(DIM);
    for (unsigned d = 0; d < DIM; d++) { weight[d] = 0.5f + (d % 17) / 16.0f; }
    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned d = 0; d < DIM; d++) {
            in[(size_t)r * STRIDE + d] = ((int)(r * 64 + d % 31) - 15) / 8.0f;
        }
        /* Distinct RoPE tail so a width-as-stride bug cannot hide. */
        for (unsigned d = DIM; d < STRIDE; d++) {
            in[(size_t)r * STRIDE + d] = 100.0f + (float)r * 50.0f + (float)(d - DIM);
        }
    }

    float *d_in = upload(in.data(), in.size());
    float *d_w = upload(weight.data(), weight.size());
    float *d_out;
    CUDA(cudaMalloc(&d_out, (size_t)ROWS * DIM * sizeof(float)));
    ling3vl_rms_norm<<<ROWS, 256, 0, stream>>>(
        d_out, d_in, d_w, DIM, STRIDE, ROWS, eps);
    CUDA(cudaStreamSynchronize(stream));

    std::vector<float> out((size_t)ROWS * DIM);
    CUDA(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
    for (unsigned r = 0; r < ROWS; r++) {
        double sumsq = 0.0;
        for (unsigned d = 0; d < DIM; d++) {
            const double v = in[(size_t)r * STRIDE + d];
            sumsq += v * v;
        }
        const double inv = 1.0 / sqrt(sumsq / (double)DIM + eps);
        for (unsigned d = 0; d < DIM; d++) {
            close(out[(size_t)r * DIM + d],
                  in[(size_t)r * STRIDE + d] * inv * weight[d], 1e-5);
        }
    }
    CUDA(cudaFree(d_out));
    CUDA(cudaFree(d_w));
    CUDA(cudaFree(d_in));
    printf("rms_norm_strided OK\n");
}

int main(void) {
    cudaStream_t stream;
    CUDA(cudaStreamCreate(&stream));
    router(stream);
    rope_table();
    for (unsigned ctx : {131072u, 131073u, 262144u}) {
        mrope(stream, ctx, 192u);
        mrope(stream, ctx, 576u);
    }
    rms_norm_strided(stream);
    CUDA(cudaStreamDestroy(stream));
    printf("Ling primitives OK\n");
    return 0;
}
