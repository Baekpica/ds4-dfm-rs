// Group selection, learned sinks, causal DFly tail and the three rotary axes.
#include "../ds4_gpu.h"
#include "../ds4_hy4.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool ok, const char *why) {
    if (!ok) {
        std::fprintf(stderr, "Hy4: %s\n", why);
        std::exit(1);
    }
}
static ds4_gpu_tensor *tensor(size_t n) {
    auto *t = ds4_gpu_tensor_alloc(n);
    check(t, "allocate");
    return t;
}
template <class T> static void upload(ds4_gpu_tensor *t, const std::vector<T> &v) {
    check(ds4_gpu_tensor_write(t, 0, v.data(), v.size() * sizeof(T)), "upload");
}
static void attention(bool fp8, unsigned heads, unsigned pos, bool draft) {
    constexpr unsigned rows = 16;
    const unsigned cap = heads == 64 ? std::min(pos + rows, 511 + rows) : pos + rows;
    ds4_hy4_cache cache = {};
    cache.cap = cap;
    cache.fp8 = fp8;
    cache.latent = tensor(cap * (fp8 ? 513 : 1024));
    cache.rope = tensor(cap * 128);
    auto *out = tensor(rows * heads * 512 * 4), *a = tensor(rows * heads * 512 * 4),
         *q = tensor(rows * heads * 256 * 4), *ids = tensor(rows * 512 * 4);
    std::vector<float> sinks(heads), qa(rows * heads * 512), qr(rows * heads * 256);
    for (unsigned h = 0; h < heads; h++) {
        sinks[h] = 3.f + h % 8;
        for (unsigned r = 0; r < rows; r++) {
            qa[(r * heads + h) * 512 + 259] = 1.f + h % 7;
            qr[(r * heads + h) * 256 + 249] = 0.5f + r % 5;
        }
    }
    float *sink = nullptr;
    check(cudaMalloc(&sink, heads * 4) == cudaSuccess &&
              cudaMemcpy(sink, sinks.data(), heads * 4, cudaMemcpyHostToDevice) == cudaSuccess,
          "sink");
    upload(a, qa);
    upload(q, qr);
    std::vector<__nv_bfloat16> rot(cap * 64), bf(cap * 512);
    std::vector<unsigned char> bytes(cap * 513);
    const unsigned char codes[] = {0x38, 0x3a, 0x34, 0x30};
    const float mantissa[] = {1.f, 1.25f, .75f, .5f};
    auto value = [&](unsigned t) { return std::ldexp(mantissa[(t / 11) % 4], int(t % 3) - 1); };
    auto rope = [](unsigned t) { return (int(t % 17) - 8) * .25f; };
    for (unsigned t = pos + rows - cap; t < pos + rows; t++) {
        const unsigned slot = t % cap;
        std::fill_n(bytes.begin() + slot * 512, 512, codes[(t / 11) % 4]);
        std::fill_n(bf.begin() + slot * 512, 512, __float2bfloat16(value(t)));
        rot[slot * 64 + 57] = __float2bfloat16(rope(t));
        bytes[cap * 512 + slot] = static_cast<unsigned char>(int(t % 3) - 1);
    }
    upload(cache.rope, rot);
    if (fp8) {
        upload(cache.latent, bytes);
    } else {
        upload(cache.latent, bf);
    }
    std::vector<int> indices(rows * 512);
    for (unsigned r = 0; r < rows; r++) {
        const unsigned groups = (draft ? pos - 1 : pos + r) / 4;
        for (unsigned g = 0; g < 511 && groups; g++) {
            indices[r * 512 + g] = (groups - 1 - (g % groups) + r) % groups;
        }
    }
    upload(ids, indices);
    check(ds4_gpu_hy4_attention(out, a, q, &cache, ids, sink, pos, rows, heads, draft), "MLA");
    std::vector<float> result(rows * heads * 512);
    check(ds4_gpu_tensor_read(out, 0, result.data(), result.size() * 4), "read");
    for (unsigned r = 0; r < rows; r++) {
        const unsigned frontier = draft ? pos - 1 : pos + r;
        std::vector<unsigned> tokens;
        if (heads == 32 && frontier >= 2048) {
            for (unsigned g = 0; g < 511; g++) {
                for (unsigned j = 0; j < 4; j++) {
                    tokens.push_back(indices[(draft ? 0 : r * 512) + g] * 4 + j);
                }
            }
            for (unsigned j = frontier / 4 * 4; j <= frontier; j++) {
                tokens.push_back(j);
            }
        } else {
            const unsigned count = heads == 64 ? std::min(512u, frontier + 1) : frontier + 1;
            for (unsigned j = frontier + 1 - count; j <= frontier; j++) {
                tokens.push_back(j);
            }
        }
        if (draft) {
            for (unsigned j = pos; j <= pos + r; j++) {
                tokens.push_back(j);
            }
        }
        for (unsigned h = 0; h < heads; h++) {
            double num = 0, den = std::exp(double(sinks[h]));
            for (unsigned t : tokens) {
                const double score = ((1.f + h % 7) * value(t) + (.5f + r % 5) * rope(t)) / 16.;
                const double w = std::exp(score);
                num += w * value(t);
                den += w;
            }
            for (unsigned k = 0; k < 512; k++) {
                check(std::fabs(result[(r * heads + h) * 512 + k] - num / den) < 0.003,
                      "causal MLA, learned sink and rotary cache");
            }
        }
    }
    cudaFree(sink);
    for (auto *t : {out, a, q, ids, cache.latent, cache.rope}) {
        ds4_gpu_tensor_free(t);
    }
}
static void index_top() {
    constexpr unsigned n = 19, pos = 4087, count = (pos + n - 1) / 4;
    auto *q = tensor(n * 16 * 128 * 4), *w = tensor(n * 16 * 4), *tile = tensor(n * 1024 * 4);
    auto *ids = tensor(n * 512 * 4), *best = tensor(n * 512 * 4);
    ds4_hy4_cache cache = {};
    cache.groups = tensor(count * 128 * 2);
    std::vector<__nv_bfloat16> keys(count * 128);
    std::vector<float> queries(n * 16 * 128), weights(n * 16, 1.f);
    for (unsigned j = 0; j < count; ++j) {
        keys[j * 128] = __float2bfloat16(j / 128);
        keys[j * 128 + 1] = __float2bfloat16(j % 128);
    }
    for (unsigned i = 0; i < n; ++i) {
        queries[i * 16 * 128] = 128;
        queries[i * 16 * 128 + 128 + 1] = 1;
        queries[(i * 16 + 2) * 128] = -1;
    }
    upload(q, queries);
    upload(w, weights);
    upload(cache.groups, keys);
    check(ds4_gpu_hy4_index(ids, tile, best, q, w, &cache, pos, n), "pooled index");
    std::vector<int> selected(n * 512);
    check(ds4_gpu_tensor_read(ids, 0, selected.data(), selected.size() * 4), "read index");
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < 511; ++j) {
            check(selected[i * 512 + j] == int((pos + i) / 4 - j - 1), "top 511 complete groups");
        }
    }
    for (auto *t : {q, w, tile, ids, best, cache.groups}) { ds4_gpu_tensor_free(t); }
}

int main() {
    check(ds4_gpu_init(), "CUDA initialization");
    index_top();
    attention(false, 64, 536, false);
    for (bool fp8 : {false, true}) {
        attention(fp8, 32, 2039, false);
        attention(fp8, 32, 2053, true);
    }
    auto *x = tensor(256 * 4), *p = tensor(3 * 4);
    std::vector<float> v(256);
    for (unsigned f = 0; f < 32; f++) {
        v[192 + 2 * f] = 1.f;
    }
    upload(x, v);
    upload(p, std::vector<int>{31, 47, 63});
    check(ds4_gpu_hy4_rope(x, p, 1, 1, 256, 192, 1e4f) && ds4_gpu_tensor_read(x, 0, v.data(), v.size() * 4),
          "RoPE");
    for (unsigned f = 0; f < 32; f++) {
        const int axes[3] = {31, 47, 63};
        const float angle = axes[f < 30 ? f % 3 : 0] * std::pow(1e4f, -2.f * f / 64.f);
        check(std::fabs(v[192 + 2 * f] - std::cos(angle)) < 0.006f &&
                  std::fabs(v[193 + 2 * f] - std::sin(angle)) < 0.006f,
              "M-RoPE");
    }
    ds4_gpu_tensor_free(p);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    std::puts("Hy4 CUDA arithmetic passed");
}
