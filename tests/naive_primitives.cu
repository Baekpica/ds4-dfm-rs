/* Independent source equations; no model weights or inference context. */
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include "../ds4_naive_plan.h"
#include "../cuda/naive_primitives.cuh"
#include "../cuda/naive_sparse_tile.cuh"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)
#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); exit(1); \
} } while (0)

template <class T> static T *device(size_t n) {
    T *p;
    CUDA(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

template <class T> static T *upload(const std::vector<T> &v) {
    T *p = device<T>(v.size());
    CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

static float decode(unsigned code) {
    const unsigned exp = (code >> 3) & 15, mant = code & 7;
    const float value = exp ? std::ldexp(1.0f + mant / 8.0f, (int)exp - 7)
                            : std::ldexp((float)mant, -9);
    return code & 128 ? -value : value;
}

/* Enumerate the finite E4M3 table instead of using the CUDA conversion. */
static unsigned encode(float value) {
    const float magnitude = fabsf(value);
    unsigned best = 0;
    double distance = INFINITY;
    for (unsigned c = 0; c <= 126; c++) {
        const double d = fabs((double)magnitude - decode(c));
        if (d < distance || (d == distance && !(c & 1))) { best = c; distance = d; }
    }
    return best | (std::signbit(value) ? 128 : 0);
}

static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(bits));
    return value;
}

static void fp8(cudaStream_t stream) {
    std::vector<float> x(6 * N05_INDEX_DIM);
    for (unsigned d = 0; d < N05_INDEX_DIM; d++) {
        x[d] = decode(d < 127 ? d : 126);
        x[N05_INDEX_DIM + d] = -x[d];
        x[2 * N05_INDEX_DIM + d] = d < 126 ? (decode(d) + decode(d + 1)) * .5f : 448.0f;
        x[3 * N05_INDEX_DIM + d] = d ? (d - 64.0f) * 1e-7f : -0.0f;
        x[4 * N05_INDEX_DIM + d] = 0.0f;
        x[5 * N05_INDEX_DIM + d] = sinf((float)d) * 37.0f;
    }
    x[0] = 448.0f;
    x[N05_INDEX_DIM] = -448.0f;
    x[2 * N05_INDEX_DIM] = 448.0f;
    float *dx = upload(x), *scale = device<float>(12);
    const std::vector<unsigned> positions = {0, 1, 2, 3, 4, 5};
    unsigned *dp = upload(positions);
    uint8_t *codes = device<uint8_t>(2 * x.size());
    naive_fp8_pack<<<6, 128, 0, stream>>>(codes, scale, dx, dp, 6);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<uint8_t> got(x.size());
    std::vector<float> scales(6);
    CUDA(cudaMemcpy(got.data(), codes, got.size(), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(scales.data(), scale, scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned r = 0; r < 6; r++) {
        float maximum = 1e-4f;
        for (unsigned d = 0; d < N05_INDEX_DIM; d++) { maximum = std::max(maximum, fabsf(x[r * N05_INDEX_DIM + d])); }
        const float expected_scale = maximum / 448.0f;
        CHECK(scales[r] == expected_scale);
        for (unsigned d = 0; d < N05_INDEX_DIM; d++) {
            CHECK(got[r * N05_INDEX_DIM + d] == encode(x[r * N05_INDEX_DIM + d] / expected_scale));
        }
    }
    naive_fp8_query<<<6, 128, 0, stream>>>(dx, 6);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<float> query(x.size());
    CUDA(cudaMemcpy(query.data(), dx, query.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < query.size(); i++) { CHECK(query[i] == decode(got[i]) * scales[i / N05_INDEX_DIM]); }

    // Replay changes positions without changing the queued kernel arguments.
    CUDA(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    naive_fp8_pack<<<6, 128, 0, stream>>>(codes, scale, dx, dp, 6);
    CUDA(cudaStreamEndCapture(stream, &graph));
    CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    const unsigned moved[] = {6, 7, 8, 9, 10, 11};
    CUDA(cudaMemcpyAsync(dp, moved, sizeof(moved), cudaMemcpyHostToDevice, stream));
    CUDA(cudaGraphLaunch(exec, stream));
    CUDA(cudaStreamSynchronize(stream));
    std::vector<uint8_t> tail(x.size());
    CUDA(cudaMemcpy(tail.data(), codes + x.size(), tail.size(), cudaMemcpyDeviceToHost));
    CHECK(tail == got);
    CUDA(cudaGraphExecDestroy(exec)); CUDA(cudaGraphDestroy(graph));
    CUDA(cudaFree(dx)); CUDA(cudaFree(scale)); CUDA(cudaFree(codes)); CUDA(cudaFree(dp));
    printf("E4M3 table, midpoint ties, signs, zero floor, live-position replay OK\n");
}

static void scores(cudaStream_t stream) {
    enum { HISTORY = 17, ROWS = 2 };
    std::vector<uint8_t> codes(HISTORY * N05_INDEX_DIM);
    std::vector<float> scales(HISTORY), q(ROWS * N05_INDEX_HEADS * N05_INDEX_DIM), weights(ROWS * N05_INDEX_HEADS);
    const std::vector<unsigned> positions = {9, 16};
    for (unsigned k = 0; k < HISTORY; k++) {
        scales[k] = (k + 1) * .125f;
        for (unsigned d = 0; d < N05_INDEX_DIM; d++) { codes[k * N05_INDEX_DIM + d] = ((k + d) % 40) | ((d & 1) ? 128 : 0); }
    }
    for (unsigned i = 0; i < q.size(); i++) { q[i] = ((int)(i % 15) - 7) * .0625f; }
    for (unsigned i = 0; i < weights.size(); i++) { weights[i] = ((int)(i % 7) - 3) * .125f; }
    uint8_t *dc = upload(codes);
    float *ds = upload(scales), *dq = upload(q), *dw = upload(weights), *out = device<float>(ROWS * HISTORY);
    unsigned *dp = upload(positions);
    naive_index_scores<<<dim3((HISTORY + 3) / 4, ROWS), 128, 0, stream>>>(out, dq, dc, ds, dw, dp, HISTORY);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<float> got(ROWS * HISTORY);
    CUDA(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned k = 0; k < HISTORY; k++) {
            if (k > positions[r]) { CHECK(got[r * HISTORY + k] == -INFINITY); continue; }
            double score = 0;
            for (unsigned h = 0; h < N05_INDEX_HEADS; h++) {
                double dot = 0;
                for (unsigned d = 0; d < N05_INDEX_DIM; d++) {
                    dot += (double)q[(r * N05_INDEX_HEADS + h) * N05_INDEX_DIM + d] * decode(codes[k * N05_INDEX_DIM + d]) * scales[k];
                }
                score += std::max(0.0, dot) * weights[r * N05_INDEX_HEADS + h];
            }
            CHECK(fabs(got[r * HISTORY + k] - score) < 1e-6 * (1 + fabs(score)));
        }
    }
    CUDA(cudaFree(dc)); CUDA(cudaFree(ds)); CUDA(cudaFree(dq)); CUDA(cudaFree(dw)); CUDA(cudaFree(dp)); CUDA(cudaFree(out));
    printf("F32 reconstructed dot, per-head ReLU, signed weights, causal scores OK\n");
}

static void norms(cudaStream_t stream) {
    enum { ROWS = 2 };
    std::vector<float> x(ROWS * N05_EMBED), w(N05_EMBED);
    for (unsigned i = 0; i < x.size(); i++) { x[i] = ((int)(i % 31) - 15) * .0625f; }
    for (unsigned d = 0; d < w.size(); d++) { w[d] = 1 + (d % 7) * .0078125f; }
    float *dx = upload(x), *dw = upload(w), *out = device<float>(x.size());
    naive_rms<<<ROWS, 256, 0, stream>>>(out, dx, dw, N05_EMBED);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<float> got(x.size());
    CUDA(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned r = 0; r < ROWS; r++) {
        double sum = 0;
        for (unsigned d = 0; d < N05_EMBED; d++) { sum += (double)x[r * N05_EMBED + d] * x[r * N05_EMBED + d]; }
        const float inv = 1 / sqrtf((float)(sum / N05_EMBED) + 1e-5f);
        for (unsigned d = 0; d < N05_EMBED; d++) { CHECK(got[r * N05_EMBED + d] == bf16(bf16(x[r * N05_EMBED + d] * inv) * w[d])); }
    }
    std::vector<float> key(ROWS * N05_INDEX_DIM), weight(N05_INDEX_DIM), bias(N05_INDEX_DIM);
    for (unsigned d = 0; d < N05_INDEX_DIM; d++) {
        key[d] = 3;
        key[N05_INDEX_DIM + d] = 3 + ((int)(d % 17) - 8) * .125f;
        weight[d] = 1 + (d % 3) * .125f; bias[d] = ((int)(d % 9) - 4) * .0625f;
    }
    float *dk = upload(key), *wk = upload(weight), *bk = upload(bias);
    naive_key_norm<<<ROWS, N05_INDEX_DIM, 0, stream>>>(dk, wk, bk);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<float> normalized(key.size());
    CUDA(cudaMemcpy(normalized.data(), dk, normalized.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned r = 0; r < ROWS; r++) {
        double mean = 0, variance = 0;
        for (unsigned d = 0; d < N05_INDEX_DIM; d++) { mean += key[r * N05_INDEX_DIM + d] / (double)N05_INDEX_DIM; }
        for (unsigned d = 0; d < N05_INDEX_DIM; d++) { variance += pow(key[r * N05_INDEX_DIM + d] - mean, 2) / N05_INDEX_DIM; }
        for (unsigned d = 0; d < N05_INDEX_DIM; d++) {
            const float want = bf16((float)((key[r * N05_INDEX_DIM + d] - mean) / sqrt(variance + 1e-5) * weight[d] + bias[d]));
            CHECK(normalized[r * N05_INDEX_DIM + d] == want);
        }
    }
    CUDA(cudaFree(dx)); CUDA(cudaFree(dw)); CUDA(cudaFree(out)); CUDA(cudaFree(dk)); CUDA(cudaFree(wk)); CUDA(cudaFree(bk));
    printf("BF16 RMS epsilon 1e-5, affine index LayerNorm OK\n");
}

static void rope(cudaStream_t stream, unsigned width, float theta) {
    enum { ROWS = 3, HEADS = 2 };
    std::vector<float> x(ROWS * HEADS * width), table(ROWS * N05_ROT);
    const unsigned positions[] = {0, 2048, 1048575};
    for (unsigned i = 0; i < x.size(); i++) { x[i] = ((int)(i % 19) - 9) * .123f; }
    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned d = 0; d < N05_ROT / 2; d++) {
            const float angle = positions[r] * (float)pow(theta, -2.0 * d / N05_ROT);
            table[r * N05_ROT + 2 * d] = cosf(angle); table[r * N05_ROT + 2 * d + 1] = sinf(angle);
        }
    }
    float *dx = upload(x), *dt = upload(table);
    naive_rope<<<dim3(HEADS, ROWS), 256, 0, stream>>>(dx, (const float2 *)dt, HEADS, width);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<float> got(x.size());
    CUDA(cudaMemcpy(got.data(), dx, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned r = 0; r < ROWS; r++) {
        for (unsigned h = 0; h < HEADS; h++) {
            const unsigned base = (r * HEADS + h) * width;
            for (unsigned d = 0; d < width; d++) {
                float want = bf16(x[base + d]);
                if (d < N05_ROT) {
                    const unsigned pair = d % (N05_ROT / 2);
                    const float a = bf16(x[base + pair]), b = bf16(x[base + pair + N05_ROT / 2]);
                    const float c = bf16(table[r * N05_ROT + 2 * pair]), s = bf16(table[r * N05_ROT + 2 * pair + 1]);
                    want = d < N05_ROT / 2 ? bf16(bf16(a * c) - bf16(b * s)) : bf16(bf16(b * c) + bf16(a * s));
                }
                CHECK(got[base + d] == want);
            }
        }
    }
    CUDA(cudaFree(dx)); CUDA(cudaFree(dt));
    printf("split-half partial RoPE width=%u theta=%.0f OK\n", width, theta);
}

static void router(cudaStream_t stream) {
    std::vector<float> logits(N05_EXPERTS), bias(N05_EXPERTS);
    for (unsigned e = 0; e < N05_EXPERTS; e++) {
        logits[e] = ((int)e - 128) * .03125f; bias[e] = e % 11 ? 0 : 3;
    }
    float *dl = upload(logits), *db = upload(bias), *dw = device<float>(N05_USED);
    int *di = device<int>(N05_USED);
    naive_router<<<1, N05_EXPERTS, 0, stream>>>(di, dw, dl, db);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<int> got(N05_USED), expected(N05_EXPERTS);
    std::vector<float> weights(N05_USED);
    CUDA(cudaMemcpy(got.data(), di, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(weights.data(), dw, weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::iota(expected.begin(), expected.end(), 0);
    const auto prob = [&](int e) { return 1 / (1 + exp(-(double)logits[e])); };
    std::stable_sort(expected.begin(), expected.end(), [&](int a, int b) { return prob(a) + bias[a] > prob(b) + bias[b]; });
    expected.resize(N05_USED); std::sort(expected.begin(), expected.end());
    CHECK(got == expected);
    double sum = 0;
    for (int e : expected) { sum += prob(e); }
    for (unsigned i = 0; i < N05_USED; i++) { CHECK(fabs(weights[i] - prob(got[i]) / sum) < 1e-7); }
    CUDA(cudaFree(dl)); CUDA(cudaFree(db)); CUDA(cudaFree(di)); CUDA(cudaFree(dw));
    printf("sigmoid selection bias, unbiased weights, numeric expert order OK\n");
}

static void ffn(cudaStream_t stream) {
    std::vector<float> gate(N05_EMBED), up(N05_EMBED), down(N05_USED * N05_EMBED), weights(N05_USED);
    for (unsigned i = 0; i < N05_EMBED; i++) { gate[i] = ((int)(i % 31) - 15) * .127f; up[i] = (i % 19) * .213f; }
    for (unsigned e = 0; e < N05_USED; e++) {
        weights[e] = (e + 1) / 36.0f;
        for (unsigned d = 0; d < N05_EMBED; d++) { down[e * N05_EMBED + d] = ((int)((d + e * 5) % 23) - 11) * .159f; }
    }
    float *dg = upload(gate), *du = upload(up), *dd = upload(down), *dw = upload(weights), *out = device<float>(N05_EMBED);
    naive_swiglu<<<N05_EMBED / 256, 256, 0, stream>>>(out, dg, du, N05_EMBED);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<float> got(N05_EMBED);
    CUDA(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < got.size(); i++) {
        const float g = bf16(gate[i]), u = bf16(up[i]);
        CHECK(got[i] == bf16(bf16(g / (1 + expf(-g))) * u));
    }
    naive_sum<<<N05_EMBED / 256, 256, 0, stream>>>(out, dd, dw, N05_EMBED);
    naive_round<<<N05_EMBED / 256, 256, 0, stream>>>(dg, N05_EMBED);
    naive_add<<<N05_EMBED / 256, 256, 0, stream>>>(dg, out, N05_EMBED);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(got.data(), dg, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned d = 0; d < N05_EMBED; d++) {
        float sum = 0;
        for (unsigned e = 0; e < N05_USED; e++) { sum = bf16(sum + bf16(bf16(down[e * N05_EMBED + d]) * weights[e])); }
        // cur is a source BF16 state in the graph.
        CHECK(got[d] == bf16(bf16(gate[d]) + sum));
    }
    CUDA(cudaFree(dg)); CUDA(cudaFree(du)); CUDA(cudaFree(dd)); CUDA(cudaFree(dw)); CUDA(cudaFree(out));
    printf("BF16 SwiGLU, weighted expert accumulation, residual OK\n");
}

enum class AttentionData { Simple, Varied };
enum class AttentionSpan { Full, Early };

static void attention(cudaStream_t stream, unsigned kv_heads, unsigned window,
                      AttentionData pattern = AttentionData::Simple,
                      AttentionSpan span = AttentionSpan::Full) {
    const unsigned sparse = span == AttentionSpan::Early ? N05_DF_BLOCK : 2049;
    const unsigned history = window ? 146 : sparse, rows = 3;
    const unsigned capacity = window ? 130 : history;
    const unsigned kw = kv_heads * N05_KEY, vw = kv_heads * N05_VALUE, stride = kw + vw;
    std::vector<float> k((size_t)history * kw), v((size_t)history * vw), q(rows * N05_HEADS * N05_KEY), sinks(N05_HEADS);
    std::vector<unsigned> all(history), pos = {history - 3, history - 2, history - 1}, ids(rows * N05_TOP_K, UINT32_MAX);
    std::iota(all.begin(), all.end(), 0);
    for (unsigned t = 0; t < history; t++) {
        for (unsigned h = 0; h < kv_heads; h++) {
            for (unsigned d = 0; d < N05_KEY; d++) { k[((size_t)t * kv_heads + h) * N05_KEY + d] = ((int)((t + h + d) % 13) - 6) * .0625f; }
            for (unsigned d = 0; d < N05_VALUE; d++) { v[((size_t)t * kv_heads + h) * N05_VALUE + d] = ((t * 3 + h + d) % 17) * .03125f; }
        }
    }
    for (unsigned r = 0; r < rows; r++) {
        for (unsigned h = 0; h < N05_HEADS; h++) {
            sinks[h] = bf16((h % 7) * .125f);
            for (unsigned d = 0; d < 4; d++) { q[((r * N05_HEADS + h) * N05_KEY) + d] = (r + h % 3 + 1) * .125f; }
        }
        for (unsigned i = 0; i < std::min(pos[r] + 1, (unsigned)N05_TOP_K); i++) { ids[r * N05_TOP_K + i] = i + (r == 2 && !window ? 1 : 0); }
    }
    if (!window) {
        // An ascending future ID must be ignored even within the live count.
        ids[std::min(pos[0], (unsigned)N05_TOP_K - 1)] = pos[0] + 1;
        if (span == AttentionSpan::Early) { ids[(rows - 1) * N05_TOP_K + pos[rows - 1]] = UINT32_MAX; }
    }
    // Excluded/future values must not contaminate the first queries.
    for (unsigned d = 0; d < vw; d++) { v[d] = 16; v[(size_t)(history - 1) * vw + d] = 32; }
    if (pattern == AttentionData::Varied) {
        // BF16 products with different exponents exercise rounding boundaries.
        uint32_t random = 0x9e3779b9u;
        const auto next = [&] {
            constexpr uint32_t multiplier = 1664525u, increment = 1013904223u;
            random = random * multiplier + increment;
            const float value = ((int)(random >> 8) - 8388608) * 0x1p-23f;
            return bf16(std::ldexp(value, -(int)(random & 7)));
        };
        for (float &value : k) { value = next(); }
        for (float &value : q) { value = next(); }
        for (float &value : v) { value = next(); }
    }
    float *dk = upload(k), *dv = upload(v), *dq = upload(q), *sink = upload(sinks), *out = device<float>(rows * N05_HEADS * N05_VALUE);
    unsigned *da = upload(all), *dp = upload(pos), *di = upload(ids);
    uint16_t *cache = device<uint16_t>((size_t)capacity * stride);
    for (unsigned first = 0; first < history; first += 17) {
        const unsigned n = std::min(17u, history - first), count = n * stride;
        naive_kv_store<<<(count + 255) / 256, 256, 0, stream>>>((__nv_bfloat16 *)cache,
            dk + (size_t)first * kw, dv + (size_t)first * vw, da + first, kv_heads, n, capacity);
    }
    naive_attention<<<dim3(N05_HEADS / 4, rows), 128, 0, stream>>>(out, dq, (const __nv_bfloat16 *)cache,
        window ? sink : nullptr, dp, window ? nullptr : di, kv_heads, capacity, window);
    CUDA(cudaStreamSynchronize(stream));
    std::vector<float> got(rows * N05_HEADS * N05_VALUE);
    CUDA(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    // Reusing BF16 scores changes neither the serial softmax nor the V sum.
    if (window) {
        naive_attention<4, N05_WINDOW><<<dim3(N05_HEADS / 4, rows), 128, 0, stream>>>(out, dq,
            (const __nv_bfloat16 *)cache, sink, dp, nullptr, kv_heads, capacity, window);
    } else {
        naive_attention<4, N05_TOP_K><<<dim3(N05_HEADS / 4, rows), 128, 0, stream>>>(out, dq,
            (const __nv_bfloat16 *)cache, nullptr, dp, di, kv_heads, capacity, window);
    }
    CUDA(cudaStreamSynchronize(stream));
    std::vector<float> cached(got.size());
    CUDA(cudaMemcpy(cached.data(), out, cached.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(!memcmp(got.data(), cached.data(), got.size() * sizeof(float)));
    if (!window) {
        naive_sparse_tile<<<dim3(N05_HEADS, rows), 128, 0, stream>>>(out, dq,
            (const __nv_bfloat16 *)cache, dp, di, capacity);
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(cached.data(), out, cached.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(!memcmp(got.data(), cached.data(), got.size() * sizeof(float)));
        naive_sparse_tile<NaiveCache::Full><<<dim3(N05_HEADS, rows), 128, 0, stream>>>(out, dq,
            (const __nv_bfloat16 *)cache, dp, di, capacity);
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(cached.data(), out, cached.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(!memcmp(got.data(), cached.data(), got.size() * sizeof(float)));
        naive_attention<4, 0, NaiveCache::Full><<<dim3(N05_HEADS / 4, rows), 128, 0, stream>>>(out, dq,
            (const __nv_bfloat16 *)cache, nullptr, dp, di, kv_heads, capacity, 0);
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(cached.data(), out, cached.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(!memcmp(got.data(), cached.data(), got.size() * sizeof(float)));
    }
    double max_error = 0;
    for (unsigned r = 0; r < rows; r++) {
        for (unsigned h = 0; h < N05_HEADS; h++) {
            const unsigned kh = h / (N05_HEADS / kv_heads), first = window ? pos[r] + 1 - window : 0;
            const unsigned count = window ? window : std::min(pos[r] + 1, (unsigned)N05_TOP_K);
            std::vector<double> logits(count);
            double maximum = window ? sinks[h] : -INFINITY;
            for (unsigned i = 0; i < count; i++) {
                const unsigned t = window ? first + i : ids[r * N05_TOP_K + i];
                if (t > pos[r]) { logits[i] = -INFINITY; continue; }
                double dot = 0;
                for (unsigned d = 0; d < N05_KEY; d++) { dot += (double)q[(r * N05_HEADS + h) * N05_KEY + d] * bf16(k[((size_t)t * kv_heads + kh) * N05_KEY + d]); }
                logits[i] = bf16(bf16((float)dot) * (float)(1 / sqrt(192.0)));
                maximum = std::max(maximum, logits[i]);
            }
            double denominator = window ? exp(sinks[h] - maximum) : 0;
            for (double x : logits) { denominator += exp(x - maximum); }
            for (unsigned d = 0; d < N05_VALUE; d++) {
                double sum = 0;
                for (unsigned i = 0; i < count; i++) {
                    const unsigned t = window ? first + i : ids[r * N05_TOP_K + i];
                    if (t > pos[r]) { continue; }
                    const float probability = bf16((float)(exp(logits[i] - maximum) / denominator));
                    sum += (double)probability * bf16(bf16(v[((size_t)t * kv_heads + kh) * N05_VALUE + d]) * .707f);
                }
                const double want = bf16((float)sum), error = fabs(got[(r * N05_HEADS + h) * N05_VALUE + d] - want);
                CHECK(std::isfinite(got[(r * N05_HEADS + h) * N05_VALUE + d]));
                CHECK(error <= .002 * (1 + fabs(want)));
                max_error = std::max(max_error, error);
            }
        }
    }
    CUDA(cudaFree(dk)); CUDA(cudaFree(dv)); CUDA(cudaFree(dq)); CUDA(cudaFree(sink)); CUDA(cudaFree(out));
    CUDA(cudaFree(da)); CUDA(cudaFree(dp)); CUDA(cudaFree(di)); CUDA(cudaFree(cache));
    printf("BF16 GQA heads=%u window=%u ring=%u max_abs=%.6g OK\n", kv_heads, window, capacity, max_error);
}

static void topk(cudaStream_t stream, unsigned history, unsigned rows) {
    std::vector<float> scores((size_t)rows * history);
    std::vector<unsigned> positions(rows);
    for (unsigned r = 0; r < rows; r++) {
        positions[r] = r == 0 ? history - 1 : history - 1 - r * 7;
        for (unsigned k = 0; k < history; k++) {
            /* Equal scores straddle leaf/merge boundaries. Future keys win
             * unless selection itself applies the causal boundary. */
            scores[(size_t)r * history + k] = (float)((k * 31u + r * 13u) % 257u) - 129.0f;
            if (r == 1) { scores[(size_t)r * history + k] = k & 1 ? -0.0f : 0.0f; }
            if (k > positions[r]) { scores[(size_t)r * history + k] = 10000.0f; }
        }
    }
    float *ds = upload(scores);
    unsigned *dp = upload(positions), *ids = device<unsigned>((size_t)rows * N05_TOP_K);
    const unsigned tiles = (history + N05_HISTORY_TILE - 1) / N05_HISTORY_TILE;
    uint64_t *a = device<uint64_t>((size_t)rows * tiles * N05_TOP_K);
    uint64_t *b = device<uint64_t>((size_t)rows * tiles * N05_TOP_K);
    CUDA(naive_topk_launch(ids, ds, a, b, dp, history, rows, stream));
    CUDA(cudaStreamSynchronize(stream));
    std::vector<unsigned> got((size_t)rows * N05_TOP_K);
    CUDA(cudaMemcpy(got.data(), ids, got.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (unsigned r = 0; r < rows; r++) {
        std::vector<unsigned> want(positions[r] + 1);
        std::iota(want.begin(), want.end(), 0);
        std::stable_sort(want.begin(), want.end(), [&](unsigned i, unsigned j) {
            return scores[(size_t)r * history + i] > scores[(size_t)r * history + j];
        });
        want.resize(std::min((unsigned)want.size(), (unsigned)N05_TOP_K));
        std::sort(want.begin(), want.end());
        want.resize(N05_TOP_K, UINT32_MAX);
        for (unsigned k = 0; k < N05_TOP_K; k++) { CHECK(got[(size_t)r * N05_TOP_K + k] == want[k]); }
    }
    CUDA(cudaFree(ds)); CUDA(cudaFree(dp)); CUDA(cudaFree(ids)); CUDA(cudaFree(a)); CUDA(cudaFree(b));
    printf("stable causal top-k history=%u rows=%u OK\n", history, rows);
}

int main(void) {
    cudaStream_t stream;
    CUDA(cudaStreamCreate(&stream));
    fp8(stream);
    scores(stream);
    norms(stream);
    rope(stream, N05_KEY, 10000);
    rope(stream, N05_INDEX_DIM, 10000000);
    router(stream);
    ffn(stream);
    for (unsigned h : {1u, 127u, 2048u, 2049u, 4096u, 4097u, 12345u, 1048576u}) {
        topk(stream, h, h > 127 ? 3 : 1);
    }
    attention(stream, 8, N05_WINDOW);
    attention(stream, 4, 0);
    attention(stream, 8, N05_WINDOW, AttentionData::Varied);
    attention(stream, 4, 0, AttentionData::Varied);
    attention(stream, 4, 0, AttentionData::Varied, AttentionSpan::Early);
    CUDA(cudaStreamDestroy(stream));
    return 0;
}
