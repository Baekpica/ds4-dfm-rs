// Group selection, learned sinks, causal DFly tail, the three rotary axes,
// verify-width BF16 rows against one-row decode and MQ-Q5-Q8's re-typed
// router and per-head projections.
#include "../ds4_gpu.h"
#include "../ds4_hy4.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Prefill rows take one CTA each; fewer than 16 rows split their keys.
static void attention(bool fp8, unsigned heads, unsigned pos, bool draft, unsigned rows) {
    const unsigned cap = heads == 64 ? std::min(pos + rows, 511 + rows) : pos + rows;
    ds4_hy4_cache cache = {};
    cache.cap = cap;
    cache.fp8 = fp8;
    cache.latent = tensor(cap * (fp8 ? 513 : 1024));
    cache.rope = tensor(cap * 128);
    auto *out = tensor(rows * heads * 512 * 4), *pack = tensor(rows * heads * 512 * 2), *a = tensor(rows * heads * 512 * 4),
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
    // The tiled kernel reads q_abs as BF16, the scalar one as F32.
    const bool tiled = ds4_gpu_hy4_flash(rows, draft);
    if (tiled) {
        std::vector<__nv_bfloat16> half(qa.size());
        std::transform(qa.begin(), qa.end(), half.begin(), [](float v) { return __float2bfloat16(v); });
        upload(a, half);
    } else {
        upload(a, qa);
    }
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
    const int kind = ds4_gpu_hy4_attention(out, pack, a, q, &cache, ids, sink, pos, rows, heads, draft);
    check(kind == (tiled ? DS4_HY4_HEAD_PACKED : DS4_HY4_HEAD_F32), "MLA");
    std::vector<float> result(rows * heads * 512);
    if (kind == DS4_HY4_HEAD_PACKED) {
        std::vector<__nv_bfloat16> half(result.size());
        check(ds4_gpu_tensor_read(pack, 0, half.data(), half.size() * 2), "read");
        std::transform(half.begin(), half.end(), result.begin(), [](__nv_bfloat16 v) { return __bfloat162float(v); });
    } else {
        check(ds4_gpu_tensor_read(out, 0, result.data(), result.size() * 4), "read");
    }
    // BF16 softmax weights and rows: ~2^-8 relative.
    const double slack = kind == DS4_HY4_HEAD_PACKED ? 0.008 : 0.0;
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
                check(std::fabs(result[(r * heads + h) * 512 + k] - num / den) < 0.003 + slack * num / den,
                      "causal MLA, learned sink and rotary cache");
            }
        }
    }
    cudaFree(sink);
    for (auto *t : {out, pack, a, q, ids, cache.latent, cache.rope}) {
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

// Two to four verify rows of a BF16 projection equal their one-row decode
// results bit for bit, from F32 or BF16 input rows.
static void verify_rows() {
    constexpr unsigned in = 3072, outs = 1536;
    std::vector<__nv_bfloat16> w((size_t)in * outs);
    for (size_t i = 0; i < w.size(); i++) {
        w[i] = __float2bfloat16(float(int(i * 2654435761u % 2001) - 1000) / 1024.f);
    }
    check(ds4_gpu_set_model_map(w.data(), w.size() * 2), "weight map");
    for (unsigned n = 2; n <= 4; n++) {
        std::vector<float> x((size_t)n * in);
        std::vector<__nv_bfloat16> xb(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = __bfloat162float(__float2bfloat16(float(int(i * 40503u % 997) - 498) / 256.f));
            xb[i] = __float2bfloat16(x[i]);
        }
        auto *xf = tensor(x.size() * 4), *xh = tensor(xb.size() * 2), *got = tensor(n * outs * 4),
             *one = tensor(outs * 4);
        upload(xf, x);
        upload(xh, xb);
        for (ds4_hy4_rows input : {DS4_HY4_ROWS_F32, DS4_HY4_ROWS_BF16}) {
            check(ds4_gpu_hy4_rows_bf16(got, w.data(), w.size() * 2, 0, in, outs,
                                        input == DS4_HY4_ROWS_F32 ? xf : xh, input, n) == 1,
                  "verify rows");
            std::vector<float> rows(n * outs), ref(outs);
            check(ds4_gpu_tensor_read(got, 0, rows.data(), rows.size() * 4), "read rows");
            for (unsigned t = 0; t < n; t++) {
                auto *xt = ds4_gpu_tensor_view(xf, (uint64_t)t * in * 4, in * 4);
                check(xt && ds4_gpu_matmul_bf16_stable_rows_tensor(one, w.data(), w.size() * 2, 0, in, outs, xt, 1) &&
                          ds4_gpu_tensor_read(one, 0, ref.data(), outs * 4),
                      "one-row reference");
                ds4_gpu_tensor_free(xt);
                check(std::equal(ref.begin(), ref.end(), rows.begin() + (size_t)t * outs), "verify row = decode row");
            }
        }
        for (auto *t : {xf, xh, got, one}) {
            ds4_gpu_tensor_free(t);
        }
    }
}

// MQ-Q5-Q8's BF16 router equals the F32 router bit for bit: one row on the
// F32 GEMV, prefill rows on its SGEMM, verify rows on one-row arithmetic.
static void bf16_router() {
    constexpr unsigned in = 3072, outs = 512;
    const size_t values = (size_t)in * outs;
    std::vector<uint8_t> map(values * 6);
    auto *wb = reinterpret_cast<__nv_bfloat16 *>(map.data());
    auto *wf = reinterpret_cast<float *>(map.data() + values * 2);
    for (size_t i = 0; i < values; i++) {
        wb[i] = __float2bfloat16(float(int(i * 2654435761u % 4001) - 2000) / 8192.f);
        wf[i] = __bfloat162float(wb[i]);
    }
    check(ds4_gpu_set_model_map(map.data(), map.size()), "router map");
    for (unsigned n : {1u, 3u, 9u}) {
        std::vector<float> x((size_t)n * in), got((size_t)n * outs), ref((size_t)n * outs);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = float(int(i * 40503u % 1999) - 999) / 777.f;
        }
        auto *xt = tensor(x.size() * 4), *a = tensor(got.size() * 4), *b = tensor(ref.size() * 4);
        upload(xt, x);
        check(ds4_gpu_hy4_router(a, map.data(), map.size(), 0, xt, n) &&
                  ds4_gpu_tensor_read(a, 0, got.data(), got.size() * 4),
              "BF16 router");
        for (unsigned r = 0; r < n; r++) {
            auto *row = ds4_gpu_tensor_view(xt, (uint64_t)r * in * 4, in * 4);
            const bool one = n != 9;
            check(row &&
                      (one ? ds4_gpu_ling3vl_matmul_f32(b, map.data(), map.size(), values * 2, in, outs, row, 1)
                           : r || ds4_gpu_matmul_f32_tensor(b, map.data(), map.size(), values * 2, in, outs, xt, n)) &&
                      ds4_gpu_tensor_read(b, 0, ref.data() + (one ? (size_t)r * outs : 0), (one ? outs : n * outs) * 4),
                  "F32 router reference");
            ds4_gpu_tensor_free(row);
        }
        check(std::equal(got.begin(), got.end(), ref.begin()), "BF16 router = F32 router");
        for (auto *t : {xt, a, b}) {
            ds4_gpu_tensor_free(t);
        }
    }
}

// MQ-Q5-Q8's Q8_0 k_b/v_b: staged decode rows and the widened prefill GEMM
// both match a host dot over BF16-rounded weights.
static void q8_heads() {
    constexpr unsigned h = 4, in = 192, width = 512, stride = 256;
    const size_t blocks = (size_t)h * width * in / 32;
    std::vector<uint8_t> codes(blocks * 34);
    std::vector<float> w(blocks * 32);
    for (size_t b = 0; b < blocks; b++) {
        const __half d = __float2half(float(1 + b % 13) / 512.f);
        memcpy(codes.data() + b * 34, &d, 2);
        for (unsigned k = 0; k < 32; k++) {
            const int8_t q = int8_t(int((b * 31 + k * 7) % 255) - 127);
            codes[b * 34 + 2 + k] = uint8_t(q);
            w[b * 32 + k] = __bfloat162float(__float2bfloat16(__half2float(d) * q));
        }
    }
    check(ds4_gpu_set_model_map(codes.data(), codes.size()), "Q8_0 head map");
    for (unsigned n : {3u, 16u}) {
        std::vector<__nv_bfloat16> x((size_t)n * h * stride);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = __float2bfloat16(float(int(i * 2246822519u % 1001) - 500) / 256.f);
        }
        auto *pack = tensor(x.size() * 2), *out = tensor((size_t)n * h * width * 4);
        upload(pack, x);
        check(ds4_gpu_hy4_head_mm(out, pack, pack, codes.data(), codes.size(), 0, n, h, in, stride, width,
                                  DS4_HY4_HEAD_PACKED, DS4_HY4_HEAD_OUT_F32, 8),
              "Q8_0 head projection");
        std::vector<float> got((size_t)n * h * width);
        check(ds4_gpu_tensor_read(out, 0, got.data(), got.size() * 4), "read Q8_0 heads");
        for (unsigned r = 0; r < n; r++) {
            for (unsigned head = 0; head < h; head++) {
                for (unsigned j = 0; j < width; j++) {
                    double ref = 0, mag = 0;
                    for (unsigned k = 0; k < in; k++) {
                        const double v = w[((size_t)head * width + j) * in + k] *
                                         __bfloat162float(x[((size_t)r * h + head) * stride + k]);
                        ref += v;
                        mag += std::fabs(v);
                    }
                    check(std::fabs(got[((size_t)r * h + head) * width + j] - ref) <= 1e-5 * mag + 1e-6,
                          "Q8_0 head projection = host dot");
                }
            }
        }
        ds4_gpu_tensor_free(pack);
        ds4_gpu_tensor_free(out);
    }
}

// Prefill rows of MQ-Q5-Q8's Q8_0 dense matrices on widened BF16 weights,
// from F32 and from BF16 input rows, match a host dot.
static void q8_wide() {
    constexpr unsigned in = 512, outs = 96, n = 64;
    const size_t blocks = (size_t)in * outs / 32;
    std::vector<uint8_t> codes(blocks * 34);
    std::vector<float> w(blocks * 32);
    for (size_t b = 0; b < blocks; b++) {
        const __half d = __float2half(float(1 + b % 11) / 256.f);
        memcpy(codes.data() + b * 34, &d, 2);
        for (unsigned k = 0; k < 32; k++) {
            const int8_t q = int8_t(int((b * 17 + k * 29) % 255) - 127);
            codes[b * 34 + 2 + k] = uint8_t(q);
            w[b * 32 + k] = __bfloat162float(__float2bfloat16(__half2float(d) * q));
        }
    }
    check(ds4_gpu_set_model_map(codes.data(), codes.size()), "Q8_0 dense map");
    std::vector<float> x((size_t)n * in);
    std::vector<__nv_bfloat16> xb(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        xb[i] = __float2bfloat16(float(int(i * 2654435761u % 2001) - 1000) / 512.f);
        x[i] = __bfloat162float(xb[i]);
    }
    auto *xf = tensor(x.size() * 4), *xh = tensor(xb.size() * 2), *out = tensor((size_t)n * outs * 4);
    upload(xf, x);
    upload(xh, xb);
    for (ds4_hy4_rows input : {DS4_HY4_ROWS_F32, DS4_HY4_ROWS_BF16}) {
        check(ds4_gpu_hy4_q8_wide(out, codes.data(), codes.size(), 0, in, outs, input == DS4_HY4_ROWS_F32 ? xf : xh,
                                  input, n) == 1,
              "widened Q8_0 dense");
        std::vector<float> got((size_t)n * outs);
        check(ds4_gpu_tensor_read(out, 0, got.data(), got.size() * 4), "read widened Q8_0 dense");
        for (unsigned r = 0; r < n; r++) {
            for (unsigned j = 0; j < outs; j++) {
                double ref = 0, mag = 0;
                for (unsigned k = 0; k < in; k++) {
                    ref += (double)w[(size_t)j * in + k] * x[(size_t)r * in + k];
                    mag += std::fabs((double)w[(size_t)j * in + k] * x[(size_t)r * in + k]);
                }
                check(std::fabs(got[(size_t)r * outs + j] - ref) <= 1e-5 * mag + 1e-6, "widened Q8_0 dense = host dot");
            }
        }
    }
    for (auto *t : {xf, xh, out}) {
        ds4_gpu_tensor_free(t);
    }
}

int main() {
    // The rows check maps a host weight buffer; copy it like a resident model.
    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    check(ds4_gpu_init(), "CUDA initialization");
    index_top();
    verify_rows();
    bf16_router();
    q8_heads();
    q8_wide();
    for (unsigned rows : {16u, 4u}) {
        attention(false, 64, 536, false, rows);
        for (bool fp8 : {false, true}) {
            attention(fp8, 32, 2039, false, rows);
            attention(fp8, 32, 2053, true, rows);
        }
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
