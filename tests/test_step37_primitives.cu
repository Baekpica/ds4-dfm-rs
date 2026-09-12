/* Small native gates; no model weights, owner, or inference context. */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "fixtures/step37/primitives.h"
#include "../cuda/step37_primitives.cuh"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)

template<class T> static T *upload(const T *data, size_t n) {
    T *p;
    CUDA(cudaMalloc(&p, n * sizeof(T)));
    CUDA(cudaMemcpy(p, data, n * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

static void close(float got, double want, double tol = 3e-6) {
    if (!std::isfinite(got) || fabs(got - want) > tol * (1 + fabs(want))) {
        fprintf(stderr, "got %.9g expected %.12g\n", got, want);
        exit(1);
    }
}

static void activation(cudaStream_t stream) {
    enum { WIDTH = 11, ROWS = 2 };
    float gate[WIDTH * ROWS], up[WIDTH * ROWS], got[WIDTH * ROWS];
    for (unsigned i = 0; i < WIDTH * ROWS; i++) {
        gate[i] = fixture_gate[i % WIDTH];
        up[i] = fixture_up[i % WIDTH];
    }
    const float weights[] = {0.25f, 3.0f};
    float *g = upload(gate, WIDTH * ROWS), *u = upload(up, WIDTH * ROWS);
    float *w = upload(weights, ROWS), *y = upload(gate, WIDTH * ROWS);
    const float limits[] = {0, 7, 16};
    for (unsigned c = 0; c < 3; c++) {
        step37_swiglu<<<1, 64, 0, stream>>>(y, g, u, w, WIDTH, WIDTH * ROWS, limits[c]);
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(got, y, sizeof(got), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < WIDTH * ROWS; i++) {
            close(got[i], fixture_swiglu[c * WIDTH + i % WIDTH] * weights[i / WIDTH]);
        }
        // The shared expert uses the same post-SiLU clamp without route weights.
        step37_swiglu<<<1, 64, 0, stream>>>(y, g, u, nullptr, WIDTH, WIDTH * ROWS, limits[c]);
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(got, y, sizeof(got), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < WIDTH * ROWS; i++) {
            close(got[i], fixture_swiglu[c * WIDTH + i % WIDTH]);
        }
    }
    CUDA(cudaFree(g)); CUDA(cudaFree(u)); CUDA(cudaFree(w)); CUDA(cudaFree(y));
}

static void routing(cudaStream_t stream) {
    enum { ROWS = 3, EXPERTS = 288, USED = 8 };
    float *x = upload(fixture_logits, ROWS * EXPERTS);
    float *b = upload(fixture_bias, EXPERTS);
    float *w = upload(fixture_weights, ROWS * USED);
    int *ids = upload(fixture_ids, ROWS * USED);
    int got_ids[ROWS * USED]; float got_weights[ROWS * USED];
    step37_router<<<ROWS, 128, 2 * EXPERTS * sizeof(float), stream>>>(ids, w, x, b);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(got_ids, ids, sizeof(got_ids), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(got_weights, w, sizeof(got_weights), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < ROWS * USED; i++) {
        CHECK(got_ids[i] == fixture_ids[i]);
        close(got_weights[i], fixture_weights[i]);
    }
    // Equal scores select ascending expert IDs. A nonfinite row must not
    // become a plausible finite routing distribution or an invalid gather ID.
    float equal[ROWS * EXPERTS] = {};
    equal[EXPERTS] = NAN;
    CUDA(cudaMemcpy(x, equal, sizeof(equal), cudaMemcpyHostToDevice));
    step37_router<<<ROWS, 128, 2 * EXPERTS * sizeof(float), stream>>>(ids, w, x, nullptr);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(got_ids, ids, sizeof(got_ids), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(got_weights, w, sizeof(got_weights), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < USED; i++) {
        CHECK(got_ids[i] == (int)i);
        close(got_weights[i], 3.0 / USED);
        CHECK(got_ids[USED + i] >= 0 && got_ids[USED + i] < EXPERTS);
        CHECK(std::isnan(got_weights[USED + i]));
    }
    CUDA(cudaFree(x)); CUDA(cudaFree(b)); CUDA(cudaFree(w)); CUDA(cudaFree(ids));
}

static void head_gate(cudaStream_t stream, unsigned heads) {
    const unsigned count = 3 * heads * 128;
    std::vector<float> x(count), gate(3 * heads), got(count);
    for (unsigned i = 0; i < count; i++) { x[i] = sin(i * 0.7); }
    for (unsigned i = 0; i < 3 * heads; i++) { gate[i] = ((int)(i % 41) - 20) * 0.7f; }
    float *dx = upload(x.data(), count), *dg = upload(gate.data(), gate.size());
    step37_attn_gate<<<(count + 255) / 256, 256, 0, stream>>>(dx, dg, count);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(got.data(), dx, count * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < count; i++) {
        close(got[i], x[i] / (1 + exp(-double(gate[i / 128]))));
    }
    CUDA(cudaFree(dx)); CUDA(cudaFree(dg));
}

static void rope(cudaStream_t stream, unsigned heads, unsigned rotary, float theta) {
    enum { ROWS = 3, DIM = 128 };
    const size_t count = ROWS * heads * DIM;
    std::vector<float> x(count), got(count);
    float norm[DIM], frequency[DIM / 2];
    for (size_t i = 0; i < count; i++) { x[i] = sin(i * 0.17) * 3; }
    for (unsigned i = 0; i < DIM; i++) { norm[i] = 0.8f + i * 0.002f; }
    for (unsigned i = 0; i < rotary / 2; i++) {
        const float factor = rotary == DIM ? 1 : 1 + i * 0.11f;
        frequency[i] = pow(double(theta), -2.0 * i / rotary) / factor;
    }
    unsigned positions[ROWS] = {0, 511, 262143};
    float *dx = upload(x.data(), count), *dy = upload(x.data(), count);
    float *dn = upload(norm, DIM), *df = upload(frequency, rotary / 2);
    float2 *table;
    CUDA(cudaMalloc(&table, ROWS * rotary / 2 * sizeof(float2)));
    unsigned *dp = upload(positions, ROWS);
    // Replaying this graph with changed device positions catches baked RoPE
    // scalars. Full attention alone uses the frequency-factor tensor.
    cudaGraph_t graph; cudaGraphExec_t executable;
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    step37_rope_table<<<ROWS, 64, 0, stream>>>(table, df, dp, rotary / 2, ROWS);
    step37_qk_rope<<<ROWS * heads, DIM, 0, stream>>>(dy, dx, dn, table, heads, rotary);
    CUDA(cudaStreamEndCapture(stream, &graph));
    CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (unsigned replay = 0; replay < 2; replay++) {
        if (replay) { positions[0] = 4097; positions[1] = 512; positions[2] = 131072; }
        CUDA(cudaMemcpyAsync(dp, positions, sizeof(positions), cudaMemcpyHostToDevice, stream));
        CUDA(cudaGraphLaunch(executable, stream));
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(got.data(), dy, count * sizeof(float), cudaMemcpyDeviceToHost));
        for (unsigned row = 0; row < ROWS * heads; row++) {
            double sum = 0;
            for (unsigned d = 0; d < DIM; d++) { sum += double(x[row * DIM + d]) * x[row * DIM + d]; }
            const double scale = 1 / sqrt(sum / DIM + 1e-5);
            for (unsigned d = 0; d < DIM; d++) {
                double want = x[row * DIM + d] * scale * norm[d];
                if (d < rotary) {
                    const unsigned half = rotary / 2, j = d % half;
                    // The reference uses the same FP32 phase contract, with
                    // independent FP64 norm and trigonometric arithmetic.
                    const float angle = positions[row / heads] * frequency[j];
                    const unsigned mate = d < half ? d + half : d - half;
                    const double other = x[row * DIM + mate] * scale * norm[mate];
                    want = want * cos(angle) + (d < half ? -other : other) * sin(angle);
                }
                close(got[row * DIM + d], want, 2e-5);
            }
        }
    }
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
    CUDA(cudaFree(dx)); CUDA(cudaFree(dy)); CUDA(cudaFree(dn)); CUDA(cudaFree(df)); CUDA(cudaFree(dp));
    CUDA(cudaFree(table));
}

int main() {
    setbuf(stdout, nullptr);
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    activation(stream);
    puts("post-SiLU clamp PASS");
    routing(stream);
    puts("router PASS");
    head_gate(stream, 64);
    head_gate(stream, 96);
    rope(stream, 64, 64, 5000000);
    rope(stream, 96, 128, 10000);
    rope(stream, 8, 64, 5000000);
    rope(stream, 8, 128, 10000);
    CUDA(cudaStreamDestroy(stream));
    puts("Step37 CUDA: post-SiLU clamp, biased router, QK norm/RoPE replay PASS");
}
