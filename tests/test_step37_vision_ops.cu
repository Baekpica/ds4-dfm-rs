/* Step vision layout/arithmetic checks without model weights. */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/step37_vision.cuh"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "vision FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)
static void close(float got, double want) {
    if (!std::isfinite(got) || fabs(got - want) > 8e-6 * (1 + fabs(want))) {
        fprintf(stderr, "vision difference: got %.12g expected %.12g, error %.12g\n", got, want, fabs(got - want));
        exit(1);
    }
}
static float *upload(const std::vector<float> &v) {
    float *p; CUDA(cudaMalloc(&p, v.size() * 4));
    CUDA(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return p;
}
static void read(float *p, std::vector<float> &v, cudaStream_t s) {
    CUDA(cudaStreamSynchronize(s));
    CUDA(cudaMemcpy(v.data(), p, v.size() * 4, cudaMemcpyDeviceToHost));
}
static void columns(cudaStream_t s, unsigned edge, unsigned channels, unsigned kernel) {
    const unsigned stride = kernel == 14 ? 14 : 2, pad = kernel == 14 ? 0 : 1;
    const unsigned output = (edge + 2 * pad - kernel) / stride + 1;
    const unsigned inner = channels * kernel * kernel;
    std::vector<float> x(edge * edge * channels), y(output * output * inner);
    for (unsigned i = 0; i < x.size(); i++) { x[i] = (int)(i % 109) - 54; }
    float *d = upload(x), *out = upload(y);
    s37v_im2col<<<(y.size() + 255) / 256, 256, 0, s>>>(out, d, edge, channels, kernel);
    read(out, y, s);
    for (unsigned oy = 0; oy < output; oy++) {
        for (unsigned ox = 0; ox < output; ox++) {
            for (unsigned c = 0; c < channels; c++) {
                for (unsigned ky = 0; ky < kernel; ky++) {
                    for (unsigned kx = 0; kx < kernel; kx++) {
                        int ix = ox * stride + kx - pad, iy = oy * stride + ky - pad;
                        float want = 0;
                        if (ix >= 0 && iy >= 0 && ix < (int)edge && iy < (int)edge) {
                            size_t at = kernel == 14 ? (c * edge + iy) * edge + ix :
                                (iy * edge + ix) * channels + c;
                            want = x[at];
                        }
                        CHECK(y[(oy * output + ox) * inner + (c * kernel + ky) * kernel + kx] == want);
                    }
                }
            }
        }
    }
    CUDA(cudaFree(d)); CUDA(cudaFree(out));
}
static void position(cudaStream_t s, unsigned edge) {
    enum { DIM = 1536, SOURCE = 52 };
    std::vector<float> p(SOURCE * SOURCE * DIM), x(edge * edge * DIM), y(x.size());
    for (unsigned i = 0; i < p.size(); i++) { p[i] = sin(i * 0.37); }
    for (unsigned i = 0; i < x.size(); i++) { x[i] = sin(i * 0.19); }
    float *dp = upload(p), *dx = upload(x);
    s37v_position<<<(x.size() + 255) / 256, 256, 0, s>>>(dx, dp, edge);
    read(dx, y, s);
    for (unsigned i = 0; i < y.size(); i++) {
        const unsigned row = i / DIM, ch = i % DIM;
        // Independent scalar align_corners=False interpolation, no AA.
        const double sx = fmax(0, (row % edge + 0.5) * SOURCE / edge - 0.5);
        const double sy = fmax(0, (row / edge + 0.5) * SOURCE / edge - 0.5);
        const unsigned x0 = (unsigned)sx, y0 = (unsigned)sy;
        const unsigned x1 = x0 + 1 < SOURCE ? x0 + 1 : x0;
        const unsigned y1 = y0 + 1 < SOURCE ? y0 + 1 : y0;
        const double wx = sx - x0, wy = sy - y0;
        const double top = p[(y0 * SOURCE + x0) * DIM + ch] * (1 - wx) + p[(y0 * SOURCE + x1) * DIM + ch] * wx;
        const double bottom = p[(y1 * SOURCE + x0) * DIM + ch] * (1 - wx) + p[(y1 * SOURCE + x1) * DIM + ch] * wx;
        close(y[i], x[i] + top * (1 - wy) + bottom * wy);
    }
    CUDA(cudaFree(dp)); CUDA(cudaFree(dx));
}
static void rope(cudaStream_t s, unsigned edge) {
    enum { DIM = 1536, HEAD = 96, HEADS = 16 };
    std::vector<float> x(edge * edge * 3 * DIM), bias(3 * DIM), y(x.size());
    for (unsigned i = 0; i < x.size(); i++) { x[i] = sin(i * 0.013); }
    for (unsigned i = 0; i < bias.size(); i++) { bias[i] = cos(i * 0.037); }
    float *dx = upload(x), *db = upload(bias);
    s37v_qkv<<<(x.size() / 2 + 255) / 256, 256, 0, s>>>(dx, db, edge);
    read(dx, y, s);
    for (unsigned row = 0; row < edge * edge; row++) {
        for (unsigned h = 0; h < 3 * HEADS; h++) {
            for (unsigned pair = 0; pair < HEAD / 2; pair++) {
                const size_t at = (row * 3 * HEADS + h) * HEAD + pair * 2;
                const double a = x[at] + bias[h * HEAD + pair * 2];
                const double b = x[at + 1] + bias[h * HEAD + pair * 2 + 1];
                const double theta = h >= 2 * HEADS ? 0 :
                    (pair < HEAD / 4 ? row % edge : row / edge) * pow(10000., -2. * (pair % (HEAD / 4)) / (HEAD / 2));
                close(y[at], a * cos(theta) - b * sin(theta));
                close(y[at + 1], b * cos(theta) + a * sin(theta));
            }
        }
    }
    CUDA(cudaFree(dx)); CUDA(cudaFree(db));
}
static void elementwise(cudaStream_t s) {
    enum { DIM = 1536, ROWS = 3 };
    std::vector<float> x(DIM * ROWS), residual(x.size()), bias(DIM), scale(DIM), y(x.size());
    for (unsigned i = 0; i < x.size(); i++) { x[i] = (int)(i % 79) / 4.f - 10; residual[i] = sin(i); }
    for (unsigned i = 0; i < DIM; i++) { bias[i] = cos(i); scale[i] = sin(i * 0.31) * 2; }
    float *dx = upload(x), *db = upload(bias), *ds = upload(scale), *dr = upload(residual);
    s37v_quick_gelu<<<(x.size() + 255) / 256, 256, 0, s>>>(dx, db, x.size(), DIM);
    read(dx, y, s);
    for (unsigned i = 0; i < y.size(); i++) { double v = x[i] + bias[i % DIM]; close(y[i], v / (1 + exp(-1.702 * v))); }
    CUDA(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    s37v_residual<<<(x.size() + 255) / 256, 256, 0, s>>>(dr, dx, db, ds, x.size());
    read(dr, y, s);
    for (unsigned i = 0; i < y.size(); i++) { close(y[i], residual[i] + (x[i] + bias[i % DIM]) * double(scale[i % DIM])); }
    CUDA(cudaFree(dx)); CUDA(cudaFree(db)); CUDA(cudaFree(ds)); CUDA(cudaFree(dr));
}
int main() {
    cudaStream_t s; CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    columns(s, 28, 3, 14); columns(s, 5, 7, 3); columns(s, 26, 3, 3);
    for (unsigned edge : {36u, 52u}) {
        fprintf(stderr, "position %u\n", edge); position(s, edge);
        fprintf(stderr, "rope %u\n", edge); rope(s, edge);
    }
    fprintf(stderr, "elementwise\n");
    elementwise(s); CUDA(cudaStreamDestroy(s));
    puts("Step vision patch, 2D position/RoPE, QuickGELU and residual PASS");
}
