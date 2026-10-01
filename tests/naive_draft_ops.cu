/* Source equations for the bounded, noncausal DSpark block. */
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../cuda/naive_primitives.cuh"
#include "../cuda/naive_draft.cuh"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)
template<class T> static T *upload(const std::vector<T> &v) {
    T *p; CUDA(cudaMalloc(&p, v.size() * sizeof(T)));
    CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
static float bf(float x) {
    uint32_t u; memcpy(&u, &x, 4); u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
    memcpy(&x, &u, 4); return x;
}
static void taps(void) {
    const unsigned rows = 1031, keep = 1024, tap = 7;
    std::vector<float> x((size_t)rows * N05_EMBED), y((size_t)keep * N05_DF_SLOT, -19);
    for (size_t i = 0; i < x.size(); i++) { x[i] = (float)i; }
    float *a = upload(x), *b = upload(y);
    naive_df_tap<<<(keep * N05_EMBED + 255) / 256, 256>>>(b, a, rows - keep, keep, tap);
    CUDA(cudaMemcpy(y.data(), b, y.size() * 4, cudaMemcpyDeviceToHost));
    for (unsigned r = 0; r < keep; r++) {
        for (unsigned t = 0; t < N05_DF_TAPS; t++) {
            for (unsigned d = 0; d < N05_EMBED; d++) {
                CHECK(y[(size_t)r * N05_DF_SLOT + t * N05_EMBED + d] ==
                    (t == tap ? x[(size_t)(r + rows - keep) * N05_EMBED + d] : -19));
            }
        }
    }
    CUDA(cudaFree(a)); CUDA(cudaFree(b));
}
static void attention(unsigned start) {
    const unsigned B = N05_DF_BLOCK, D = N05_DF_DIM, H = N05_DF_HEADS, KH = N05_DF_KV;
    const unsigned first = start > N05_DF_WINDOW ? start - N05_DF_WINDOW : 0;
    std::vector<float> q(B * H * D), nk(B * KH * D), nv(nk.size()), out(q.size());
    std::vector<__nv_bfloat16> cache(N05_DF_CAP * KH * D * 2);
    for (size_t i = 0; i < q.size(); i++) { q[i] = bf(.09f * sinf(i * .13f)); }
    for (size_t i = 0; i < nk.size(); i++) {
        nk[i] = bf(.17f * cosf(i * .07f)); nv[i] = bf(.23f * sinf(i * .09f));
    }
    for (unsigned p = first; p < start; p++) {
        for (unsigned d = 0; d < KH * D; d++) {
            const size_t slot = (size_t)(p % N05_DF_CAP) * KH * D * 2;
            cache[slot + d] = __float2bfloat16(bf(.19f * sinf((p % 4096) * .1f + d * .3f)));
            cache[slot + KH * D + d] = __float2bfloat16(bf(.29f * cosf((p % 4096) * .03f + d * .2f)));
        }
    }
    float *dq = upload(q), *dk = upload(nk), *dv = upload(nv), *dy = upload(out);
    auto *dc = upload(cache);
    naive_df_attn<<<dim3(H / 4, B), 128>>>(dy, dq, dc, dk, dv, first, start, B);
    CUDA(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost));
    float max_error = 0;
    for (unsigned r = 0; r < B; r++) {
        const unsigned begin = start + r >= N05_DF_WINDOW ? start + r - N05_DF_WINDOW + 1 : 0;
        for (unsigned h = 0; h < H; h++) {
            const unsigned kh = h / (H / KH), count = start - begin + B;
            std::vector<float> scores(count), probs(count);
            float maximum = -INFINITY, sum = 0;
            for (unsigned i = 0; i < count; i++) {
                const unsigned p = begin + i;
                float dot = 0;
                for (unsigned d = 0; d < D; d++) {
                    const float key = p < start
                        ? __bfloat162float(cache[(size_t)(p % N05_DF_CAP) * KH * D * 2 + kh * D + d])
                        : nk[((size_t)(p - start) * KH + kh) * D + d];
                    dot += q[((size_t)r * H + h) * D + d] * key;
                }
                scores[i] = bf(bf(dot) * (1.0f / sqrtf((float)D)));
                maximum = std::max(maximum, scores[i]);
            }
            for (unsigned i = 0; i < count; i++) { probs[i] = expf(scores[i] - maximum); sum += probs[i]; }
            for (unsigned d = 0; d < D; d++) {
                float value = 0;
                for (unsigned i = 0; i < count; i++) {
                    const unsigned p = begin + i;
                    const float v = p < start
                        ? __bfloat162float(cache[(size_t)(p % N05_DF_CAP) * KH * D * 2 + KH * D + kh * D + d])
                        : nv[((size_t)(p - start) * KH + kh) * D + d];
                    value += bf(probs[i] / sum) * v;
                }
                max_error = std::max(max_error, fabsf(out[((size_t)r * H + h) * D + d] - bf(value)));
            }
        }
    }
    printf("draft attention start=%u max_abs=%g\n", start, max_error);
    CHECK(max_error <= 0.0009765625f);
    CUDA(cudaFree(dq)); CUDA(cudaFree(dk)); CUDA(cudaFree(dv)); CUDA(cudaFree(dy)); CUDA(cudaFree(dc));
}
static void top2(void) {
    std::vector<float> logits(N05_VOCAB, -3);
    logits[300] = logits[400] = 9;
    float *x = upload(logits);
    std::vector<unsigned> bits(4);
    unsigned *y = upload(bits);
    naive_df_top2<<<1, 256>>>(y, x);
    CUDA(cudaMemcpy(bits.data(), y, 16, cudaMemcpyDeviceToHost));
    float a, b; memcpy(&a, &bits[2], 4); memcpy(&b, &bits[3], 4);
    CHECK(bits[0] == 300 && bits[1] == 400 && a == 9 && b == 9);
    logits[700] = NAN;
    CUDA(cudaMemcpy(x, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
    naive_df_top2<<<1, 256>>>(y, x);
    CUDA(cudaMemcpy(bits.data(), y, 16, cudaMemcpyDeviceToHost));
    memcpy(&a, &bits[2], 4); CHECK(std::isnan(a));
    CUDA(cudaFree(x)); CUDA(cudaFree(y));
}
int main(void) {
    CUDA(cudaSetDevice(0)); taps(); top2();
    attention(4); attention(1024); attention(1048569);
    puts("DSpark tap layout, local window, future noise visibility and wrapped absolute positions pass");
    return 0;
}
