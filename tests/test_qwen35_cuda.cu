/* CUDA parity test for Prism Bonsai (qwen35 / PQ2_0) on the CUDA backend.
 *
 * The oracle is ds4's own CPU reference code, reached through the
 * DS4_TEST_HOOKS entry points below (pq2_0_row_f32 and the double-precision
 * row dot in ds4.c), so nothing here re-implements the format:
 *
 *   1. Row dequant (token embeddings) must match the reference bit-exactly.
 *   2. The decode matvec (MMVQ) and the prefill tile (MMQ) must match the
 *      reference within a documented tolerance, at the model's real shapes.
 *   3. With activations that are exactly representable in the Q8_1 form the
 *      kernels use, the matmul must match the reference to float rounding -
 *      the check that catches a wrong weight-tile layout.
 *
 * Build: make test-qwen35-cuda */

#include "ds4_gpu.h"
#include "cuda/mmq/ds4_mmq.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/mman.h>
#include <vector>

/* Defined in ds4.c under DS4_TEST_HOOKS (built as ds4_cuda_test_hooks.o). */
extern "C" int ds4_test_pq2_0_ref_row(const void *blocks, uint64_t row,
                                      uint64_t in_dim, float *out);
extern "C" int ds4_test_pq2_0_ref_matvec(const void *blocks, uint64_t out_dim,
                                         uint64_t in_dim, const float *x,
                                         float *out);
extern "C" int ds4_test_hadamard_fold(int op, uint32_t block_size, float *x,
                                      uint32_t n, const float *signs,
                                      uint32_t hd, uint32_t nk, uint32_t rep,
                                      float *scratch);

namespace {

constexpr int QK = 128;         // values per PQ2_0 block
constexpr int BLOCK_BYTES = 34; // fp16 scale + 32 code bytes

struct block_pq2_0_test {
    uint16_t d;
    uint8_t qs[QK / 4];
};

static_assert(sizeof(block_pq2_0_test) == BLOCK_BYTES, "unexpected PQ2_0 layout");

/* Model shapes: (out_dim, in_dim) pairs the Bonsai trunk actually matmuls.
 * kEmbd is the embedding lookup width (qwen35.embedding_length). */
constexpr int kEmbd = 5120;

struct shape { int M; int K; };

const shape kShapes[] = {
    {17408,  5120},  // ffn_gate / ffn_up
    { 5120, 17408},  // ffn_down
    { 5120,  6144},  // attn_output / lin_out
    { 6144,  5120},  // q/k/v projections (narrow)
};

uint32_t g_rng = 0x9e3779b9u;

static uint32_t next_rand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/* Uniform in [-1, 1]. */
static float frand(void) {
    return ((float)(next_rand() & 0xffffffu) / 8388608.0f) - 1.0f;
}

static uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        return (uint16_t)sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Random blocks with realistic positive scales.  Codes span the full 0..3
 * alphabet so the +2d level and the negative level are both exercised. */
static void fill_blocks(std::vector<block_pq2_0_test> &blocks) {
    for (auto &b : blocks) {
        b.d = float_to_half(0.002f + 0.02f * std::fabs(frand()));
        for (uint8_t &byte : b.qs) byte = (uint8_t)(next_rand() & 0xffu);
    }
}

static bool cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

static bool close_enough(const std::vector<float> &got,
                         const std::vector<float> &expected,
                         float abs_tol, float rel_tol, const char *label) {
    float worst = 0.0f;
    size_t worst_i = 0;
    int failures = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float limit = abs_tol + rel_tol * std::fabs(expected[i]);
        if (diff > worst) {
            worst = diff;
            worst_i = i;
        }
        if (!std::isfinite(got[i]) || diff > limit) failures++;
    }
    std::fprintf(stderr,
                 "%s: max_abs=%g (at %zu) failures=%d/%zu: %s\n",
                 label, worst, worst_i, failures, got.size(),
                 failures == 0 ? "PASS" : "FAIL");
    return failures == 0;
}

/* Relative L2 error over the whole output.  For the random-activation cases
 * this is the honest criterion: the kernels quantize the activation to the
 * Q8_1 form (one fp16 scale plus int8 codes per 32 values), so individual
 * outputs near zero carry a large *relative* error while the aggregate error
 * stays a small fraction of the signal.  A wrong weight-tile layout, by
 * contrast, puts ~100% of the energy into the error. */
static bool relative_l2_ok(const std::vector<float> &got,
                           const std::vector<float> &expected, float tol,
                           const char *label) {
    double err2 = 0.0, ref2 = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        const double d = (double)got[i] - (double)expected[i];
        err2 += d * d;
        ref2 += (double)expected[i] * (double)expected[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    const double rel = std::sqrt(err2) / std::sqrt(ref2 > 0.0 ? ref2 : 1.0);
    std::fprintf(stderr,
                 "%s: rel_l2=%g (tol %g) max_abs=%g rms_ref=%g: %s\n",
                 label, rel, tol, max_abs,
                 std::sqrt(ref2 / (double)(got.empty() ? 1 : got.size())),
                 rel <= tol ? "PASS" : "FAIL");
    return rel <= tol;
}

/* 1. Row lookup: the embedding path dequantizes rows; it must do so
 * bit-exactly, since it feeds the folded inverse transform. */
bool test_row_lookup() {
    constexpr int n_rows = 64;
    constexpr int in_dim = kEmbd;
    std::vector<block_pq2_0_test> blocks(
        (size_t)n_rows * (in_dim / QK));
    fill_blocks(blocks);

    void *d_blocks = nullptr;
    float *d_out = nullptr;
    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "create stream") ||
        !cuda_ok(cudaMalloc(&d_blocks, blocks.size() * sizeof(blocks[0])),
                 "alloc blocks") ||
        !cuda_ok(cudaMalloc(&d_out, (size_t)in_dim * sizeof(float)),
                 "alloc row out")) {
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(d_blocks, blocks.data(),
                                 blocks.size() * sizeof(blocks[0]),
                                 cudaMemcpyHostToDevice, stream),
                 "copy blocks")) {
        return false;
    }

    std::vector<float> ref(in_dim), got(in_dim);
    bool ok = true;
    for (int row = 0; row < n_rows; row += 19) {
        if (ds4_test_pq2_0_ref_row(blocks.data(), (uint64_t)row, in_dim,
                                   ref.data()) != 0) {
            std::fprintf(stderr, "reference row %d failed\n", row);
            ok = false;
            break;
        }
        const int rc = ds4_mmq_pq2_0_rows_f32(d_out, d_blocks, nullptr,
                                              (uint64_t)row, 1u, in_dim,
                                              stream);
        if (!cuda_ok(cudaMemcpyAsync(got.data(), d_out,
                                     got.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy row out") ||
            !cuda_ok(cudaStreamSynchronize(stream), "sync row")) {
            ok = false;
            break;
        }
        if (rc != 0) {
            std::fprintf(stderr, "ds4_mmq_pq2_0_rows_f32 returned %d\n", rc);
            ok = false;
            break;
        }
        for (int i = 0; i < in_dim; i++) {
            if (got[i] != ref[i]) {
                std::fprintf(stderr,
                             "row %d element %d: got %g expected %g (bit-exact "
                             "match required)\n",
                             row, i, got[i], ref[i]);
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }

    /* Multi-row (prefill) lookup driven by a device token array. */
    if (ok) {
        const int n_tok = 8;
        std::vector<int32_t> tokens = {3, 17, 0, 63, 42, 5, 60, 31};
        std::vector<float> multi_ref((size_t)n_tok * in_dim);
        for (int t = 0; t < n_tok; t++) {
            ds4_test_pq2_0_ref_row(blocks.data(), (uint64_t)tokens[t], in_dim,
                                   multi_ref.data() + (size_t)t * in_dim);
        }
        int32_t *d_tokens = nullptr;
        float *d_multi = nullptr;
        if (!cuda_ok(cudaMalloc(&d_tokens, tokens.size() * sizeof(int32_t)),
                     "alloc tokens") ||
            !cuda_ok(cudaMalloc(&d_multi, multi_ref.size() * sizeof(float)),
                     "alloc multi out") ||
            !cuda_ok(cudaMemcpyAsync(d_tokens, tokens.data(),
                                     tokens.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice, stream),
                     "copy tokens")) {
            ok = false;
        } else {
            const int rc = ds4_mmq_pq2_0_rows_f32(d_multi, d_blocks, d_tokens,
                                                  0u, (uint32_t)n_tok,
                                                  (uint32_t)in_dim, stream);
            std::vector<float> multi_got(multi_ref.size());
            if (!cuda_ok(cudaMemcpyAsync(multi_got.data(), d_multi,
                                         multi_got.size() * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream),
                         "copy multi out") ||
                !cuda_ok(cudaStreamSynchronize(stream), "sync multi")) {
                ok = false;
            } else if (rc != 0) {
                std::fprintf(stderr, "multi-row lookup returned %d\n", rc);
                ok = false;
            } else {
                int bad = 0;
                for (size_t i = 0; i < multi_got.size(); i++) {
                    if (multi_got[i] != multi_ref[i]) bad++;
                }
                std::fprintf(stderr,
                             "PQ2_0 CUDA row lookup (%d tokens x %d): "
                             "mismatches=%d: %s\n",
                             n_tok, in_dim, bad, bad == 0 ? "PASS" : "FAIL");
                ok = ok && bad == 0;
            }
            cudaFree(d_multi);
            cudaFree(d_tokens);
        }
    }

    cudaFree(d_blocks);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
}

/* Runs one shape through the decode vec kernel and the prefill MMQ kernel,
 * comparing both against the ds4 CPU reference. */
bool run_shape(const shape &s, int n_tok, const char *label,
               bool exact_activations) {
    const int M = s.M;
    const int K = s.K;
    std::vector<block_pq2_0_test> blocks((size_t)M * (K / QK));
    fill_blocks(blocks);

    std::vector<float> x((size_t)n_tok * K);
    if (exact_activations) {
        /* Constant per 32-value block: Q8_1 quantization is then exact (the
         * block amax is its own value, so every element codes to +-127), which
         * leaves the weight-tile layout as the only thing under test. */
        for (int t = 0; t < n_tok; t++) {
            for (int k = 0; k < K; k += 32) {
                const float v = 0.05f + 0.5f * std::fabs(frand());
                for (int j = 0; j < 32; j++) x[(size_t)t * K + k + j] = v;
            }
        }
    } else {
        for (float &v : x) v = 0.5f * frand();
    }

    std::vector<float> ref((size_t)n_tok * M, 0.0f);
    for (int t = 0; t < n_tok; t++) {
        if (ds4_test_pq2_0_ref_matvec(blocks.data(), (uint64_t)M, (uint64_t)K,
                                      x.data() + (size_t)t * K,
                                      ref.data() + (size_t)t * M) != 0) {
            std::fprintf(stderr, "%s: reference matvec failed\n", label);
            return false;
        }
    }

    void *d_blocks = nullptr;
    float *d_x = nullptr;
    float *d_out = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = false;
    if (!cuda_ok(cudaStreamCreate(&stream), "create stream") ||
        !cuda_ok(cudaMalloc(&d_blocks, blocks.size() * sizeof(blocks[0])),
                 "alloc weights") ||
        !cuda_ok(cudaMalloc(&d_x, x.size() * sizeof(float)), "alloc x") ||
        !cuda_ok(cudaMalloc(&d_out, ref.size() * sizeof(float)), "alloc out")) {
        return false;
    }

    do {
        if (!cuda_ok(cudaMemcpyAsync(d_blocks, blocks.data(),
                                     blocks.size() * sizeof(blocks[0]),
                                     cudaMemcpyHostToDevice, stream),
                     "copy weights") ||
            !cuda_ok(cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream),
                     "copy x")) {
            break;
        }

        /* MMQ output is column-major per column: out[col*M + row]. */
        std::vector<float> mmq(ref.size(), 0.0f);
        const int rc_mmq = ds4_mmq_pq2_0_dense(d_blocks, d_x, d_out, M, n_tok,
                                               K, stream);
        if (!cuda_ok(cudaMemcpyAsync(mmq.data(), d_out,
                                     mmq.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream),
                     "copy mmq out") ||
            !cuda_ok(cudaStreamSynchronize(stream), "sync mmq")) {
            break;
        }
        if (rc_mmq != 0) {
            std::fprintf(stderr, "%s: MMQ returned %d\n", label, rc_mmq);
            break;
        }
        char mmq_label[160];
        std::snprintf(mmq_label, sizeof(mmq_label), "%s MMQ (M=%d K=%d N=%d)",
                      label, M, K, n_tok);
        const bool mmq_ok = exact_activations
            ? close_enough(mmq, ref, 1e-5f, 1e-3f, mmq_label)
            : relative_l2_ok(mmq, ref, 0.05f, mmq_label);

        bool vec_ok = true;
        if (n_tok == 1) {
            std::vector<float> vec(M, 0.0f);
            std::vector<float> vec_ref(ref.begin(), ref.end());
            const int rc_vec = ds4_mmq_pq2_0_dense_vec(d_blocks, d_x, d_out,
                                                       M, 1, K, stream);
            if (!cuda_ok(cudaMemcpyAsync(vec.data(), d_out,
                                         vec.size() * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream),
                         "copy vec out") ||
                !cuda_ok(cudaStreamSynchronize(stream), "sync vec")) {
                break;
            }
            if (rc_vec != 0) {
                std::fprintf(stderr, "%s: MMVQ returned %d\n", label, rc_vec);
                break;
            }
            char vec_label[160];
            std::snprintf(vec_label, sizeof(vec_label), "%s MMVQ (M=%d K=%d)",
                          label, M, K);
            vec_ok = exact_activations
                ? close_enough(vec, vec_ref, 1e-5f, 1e-3f, vec_label)
                : relative_l2_ok(vec, vec_ref, 0.05f, vec_label);
        }
        ok = mmq_ok && vec_ok;
    } while (false);

    cudaFree(d_blocks);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
}

/* 2. Guards: shapes the kernels cannot serve must be refused, not misread. */
bool test_shape_guards() {
    std::vector<block_pq2_0_test> blocks(64 * (512 / QK));
    fill_blocks(blocks);
    std::vector<float> x(512, 0.1f);
    std::vector<float> out(64, 0.0f);

    void *d_blocks = nullptr;
    float *d_x = nullptr;
    float *d_out = nullptr;
    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "create stream") ||
        !cuda_ok(cudaMalloc(&d_blocks, blocks.size() * sizeof(blocks[0])),
                 "alloc weights") ||
        !cuda_ok(cudaMalloc(&d_x, x.size() * sizeof(float)), "alloc x") ||
        !cuda_ok(cudaMalloc(&d_out, out.size() * sizeof(float)),
                 "alloc out")) {
        return false;
    }

    const int bad_k = ds4_mmq_pq2_0_dense(d_blocks, d_x, d_out, 64, 1, 128,
                                          stream);
    const int bad_rows = ds4_mmq_pq2_0_rows_f32(d_out, d_blocks, nullptr, 0u,
                                                1u, 100u, stream);
    const bool ok = bad_k != 0 && bad_rows != 0;
    std::fprintf(stderr, "PQ2_0 shape guards (K%%256, in_dim%%128): %s\n",
                 ok ? "PASS" : "FAIL");

    cudaFree(d_blocks);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
}

/* 4. Folded activation transform.  Three oracles, each catching a different
 * mistake: the explicit normalized Sylvester matrix (the fold selftest's own
 * comparison, which depends on no ds4 code), the reference ds4_hadamard_*
 * functions through their test hook, and the forward/inverse round trip.  The
 * gdn reorder goes through the same hook's permute-then-forward path. */

/* (row, col) of the normalized Sylvester Hadamard matrix. */
static float hadamard_entry(uint32_t row, uint32_t col, uint32_t n) {
    const uint32_t parity = (uint32_t)__builtin_popcount(row & col) & 1u;
    return (parity ? -1.0f : 1.0f) / std::sqrt((float)n);
}

struct fold_case {
    const char *name;
    uint32_t    n;
    uint32_t    n_tok;
    int         op;        /* 0 rotate, 1 forward, 2 inverse, 3 gdn reorder + forward */
    bool        use_signs;
};

/* Runs one fold op on the device and against the reference. */
static bool compare_fold(const fold_case &c, uint32_t bs, uint32_t hd,
                         uint32_t nk, uint32_t rep,
                         const std::vector<float> &signs,
                         const std::vector<float> &x0, ds4_gpu_tensor *gx,
                         ds4_gpu_tensor *gs) {
    const uint64_t count = (uint64_t)c.n * c.n_tok;
    std::vector<float> ref = x0, got(count), scratch(c.n);
    const float *sr = c.use_signs ? signs.data() : nullptr;

    /* The reference hook works on one row; the kernel covers n_tok rows. */
    for (uint32_t t = 0; t < c.n_tok; t++) {
        if (ds4_test_hadamard_fold(c.op, bs, ref.data() + (size_t)t * c.n, c.n,
                                   sr, hd, nk, rep, scratch.data()) != 0) {
            std::fprintf(stderr, "fold %s: reference failed\n", c.name);
            return false;
        }
    }

    bool ran = ds4_gpu_tensor_write(gx, 0, x0.data(), count * sizeof(float));
    if (ran) {
        switch (c.op) {
        case 0:   /* bare rotation: the inverse entry with no signs */
            ran = ds4_gpu_qwen35_fold_inverse_tensor(gx, c.n, c.n_tok, bs,
                                                     nullptr) != 0;
            break;
        case 1:
            ran = ds4_gpu_qwen35_fold_forward_tensor(gx, c.n, c.n_tok, bs, gs,
                                                     0u, 0u, 0u, 0u) != 0;
            break;
        case 2:
            ran = ds4_gpu_qwen35_fold_inverse_tensor(gx, c.n, c.n_tok, bs,
                                                     gs) != 0;
            break;
        default:
            ran = ds4_gpu_qwen35_fold_forward_tensor(gx, c.n, c.n_tok, bs, gs,
                                                     1u, hd, nk, rep) != 0;
            break;
        }
    }
    ran = ran && ds4_gpu_tensor_read(gx, 0, got.data(), count * sizeof(float));
    if (!ran) {
        std::fprintf(stderr, "fold %s: CUDA run failed\n", c.name);
        return false;
    }
    return close_enough(got, ref, 1e-4f, 1e-4f, c.name);
}

static bool test_fold_transform() {
    /* The model's folded input widths, plus the single-block case. */
    struct width { uint32_t n; uint32_t n_tok; };
    const width widths[] = {
        { 5120, 1}, { 5120, 5}, { 6144, 4}, {17408, 2}, { 1024, 1},
    };
    constexpr uint32_t bs = 1024;                    /* prism.hadamard block */
    constexpr uint32_t hd = 128, nk = 16, rep = 3;   /* gdn v-grouped geometry */

    bool ok = true;
    for (const width &w : widths) {
        const uint32_t n = w.n;
        const uint32_t n_tok = w.n_tok;
        const uint64_t count = (uint64_t)n * n_tok;

        std::vector<float> signs(n);
        for (uint32_t i = 0; i < n; i++) {
            signs[i] = ((i * 7u) % 3u == 0u) ? -1.0f : 1.0f;
        }
        std::vector<float> x0(count);
        for (uint64_t i = 0; i < count; i++) {
            x0[i] = std::sin((float)i * 0.7f) + 0.3f * std::cos((float)i * 1.3f);
        }

        ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc(count * sizeof(float));
        ds4_gpu_tensor *gs = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        if (!gx || !gs ||
            !ds4_gpu_tensor_write(gs, 0, signs.data(), n * sizeof(float))) {
            std::fprintf(stderr, "fold n=%u: tensor setup failed\n", n);
            ds4_gpu_tensor_free(gx);
            ds4_gpu_tensor_free(gs);
            return false;
        }

        const fold_case cases[] = {
            {"fold rotate",  n, n_tok, 0, false},
            {"fold forward", n, n_tok, 1, true},
            {"fold inverse", n, n_tok, 2, true},
        };
        for (const fold_case &c : cases) {
            ok = compare_fold(c, bs, hd, nk, rep, signs, x0, gx, gs) && ok;
        }
        /* The gdn reorder only exists where the head geometry tiles the row. */
        if ((uint64_t)hd * nk * rep == n) {
            const fold_case gdn_case = {"fold gdn reorder + forward", n, n_tok, 3, true};
            ok = compare_fold(gdn_case, bs, hd, nk, rep, signs, x0, gx, gs) && ok;
        }

        /* Round trip: the inverse must undo the forward exactly enough that a
         * second forward reproduces the first (the transform's H*H = I). */
        {
            std::vector<float> once(count), twice(count);
            bool ran = ds4_gpu_tensor_write(gx, 0, x0.data(), count * sizeof(float)) &&
                ds4_gpu_qwen35_fold_forward_tensor(gx, n, n_tok, bs, gs, 0u, 0u, 0u, 0u) &&
                ds4_gpu_tensor_read(gx, 0, once.data(), count * sizeof(float)) &&
                ds4_gpu_qwen35_fold_forward_tensor(gx, n, n_tok, bs, gs, 0u, 0u, 0u, 0u) &&
                ds4_gpu_qwen35_fold_inverse_tensor(gx, n, n_tok, bs, gs) &&
                ds4_gpu_qwen35_fold_inverse_tensor(gx, n, n_tok, bs, gs) &&
                ds4_gpu_tensor_read(gx, 0, twice.data(), count * sizeof(float));
            if (!ran) {
                std::fprintf(stderr, "fold n=%u: round trip run failed\n", n);
                ok = false;
            } else {
                char label[160];
                std::snprintf(label, sizeof(label),
                              "fold forward/inverse round trip n=%u rows=%u",
                              n, n_tok);
                ok = close_enough(twice, x0, 1e-3f, 1e-3f, label) && ok;
            }
        }

        /* Explicit Sylvester matrix on one block: the fold selftest's oracle,
         * which depends on none of ds4's transform code. */
        if (n == bs) {
            std::vector<float> want(n), got(n);
            for (uint32_t row = 0; row < n; row++) {
                double acc = 0.0;
                for (uint32_t col = 0; col < n; col++) {
                    acc += (double)hadamard_entry(row, col, n) * (double)x0[col];
                }
                want[row] = (float)acc;
            }
            const bool ran =
                ds4_gpu_tensor_write(gx, 0, x0.data(), count * sizeof(float)) &&
                ds4_gpu_qwen35_fold_forward_tensor(gx, n, n_tok, bs, nullptr,
                                                   0u, 0u, 0u, 0u) &&
                ds4_gpu_tensor_read(gx, 0, got.data(), count * sizeof(float));
            if (!ran) {
                std::fprintf(stderr, "fold n=%u: explicit-matrix run failed\n", n);
                ok = false;
            } else {
                ok = close_enough(got, want, 1e-4f, 1e-4f,
                                  "fold vs explicit Hadamard matrix n=1024") && ok;
            }
        }

        ds4_gpu_tensor_free(gx);
        ds4_gpu_tensor_free(gs);
    }
    return ok;
}

/* 3. Host wiring: the entries the qwen35 CUDA graph will call
 * (ds4_gpu_embed_token(s)_quant_tensor and ds4_gpu_matmul_pq2_0_tensor) must
 * resolve a PQ2_0 weight out of the model map and dispatch to the kernels
 * above.  This drives the real host path, not the raw kernel entries. */
bool test_host_wiring() {
    constexpr int n_vocab = 8;
    constexpr int in_dim = kEmbd;
    constexpr uint64_t arena_bytes = (uint64_t)64 << 20;

    std::vector<block_pq2_0_test> embd((size_t)n_vocab * (in_dim / QK));
    fill_blocks(embd);

    void *arena = mmap(nullptr, arena_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) {
        std::fprintf(stderr, "host wiring: mmap failed\n");
        return false;
    }
    std::memcpy(arena, embd.data(), embd.size() * sizeof(embd[0]));

    bool ok = true;
    if (ds4_gpu_init() != 1 || ds4_gpu_set_model_map(arena, arena_bytes) != 1) {
        std::fprintf(stderr, "host wiring: backend/model-map setup failed\n");
        munmap(arena, arena_bytes);
        return false;
    }

    /* Single-token embedding lookup, bit-exact against the reference. */
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    std::vector<float> ref(in_dim), got(in_dim);
    if (!out) {
        std::fprintf(stderr, "host wiring: tensor alloc failed\n");
        ok = false;
    }
    for (int token = 0; ok && token < n_vocab; token += 3) {
        ds4_test_pq2_0_ref_row(embd.data(), (uint64_t)token, in_dim, ref.data());
        if (!ds4_gpu_embed_token_quant_tensor(out, arena, arena_bytes, 0,
                                              142u, n_vocab, (uint32_t)token,
                                              in_dim) ||
            !ds4_gpu_tensor_read(out, 0, got.data(),
                                 got.size() * sizeof(float))) {
            std::fprintf(stderr, "host wiring: embed_token(%d) failed\n", token);
            ok = false;
            break;
        }
        for (int i = 0; i < in_dim; i++) {
            if (got[i] != ref[i]) {
                std::fprintf(stderr,
                             "host wiring: embed_token(%d) element %d got %g "
                             "expected %g\n",
                             token, i, got[i], ref[i]);
                ok = false;
                break;
            }
        }
    }
    std::fprintf(stderr, "PQ2_0 host embed_token (bit-exact): %s\n",
                 ok ? "PASS" : "FAIL");

    /* Multi-token embedding lookup through the device token array. */
    if (ok) {
        const int n_tok = 4;
        std::vector<int32_t> tokens = {7, 0, 5, 2};
        ds4_gpu_tensor *tok = ds4_gpu_tensor_alloc(
            (uint64_t)n_tok * sizeof(int32_t));
        ds4_gpu_tensor *multi = ds4_gpu_tensor_alloc(
            (uint64_t)n_tok * in_dim * sizeof(float));
        std::vector<float> multi_ref((size_t)n_tok * in_dim), multi_got(multi_ref.size());
        for (int t = 0; t < n_tok; t++) {
            ds4_test_pq2_0_ref_row(embd.data(), (uint64_t)tokens[t], in_dim,
                                   multi_ref.data() + (size_t)t * in_dim);
        }
        const bool wrote = tok && multi &&
            ds4_gpu_tensor_write(tok, 0, tokens.data(),
                                 tokens.size() * sizeof(int32_t)) &&
            ds4_gpu_embed_tokens_quant_tensor(multi, tok, arena, arena_bytes, 0,
                                              142u, n_vocab, (uint32_t)n_tok,
                                              in_dim) &&
            ds4_gpu_tensor_read(multi, 0, multi_got.data(),
                                multi_got.size() * sizeof(float));
        int bad = wrote ? 0 : 1;
        if (wrote) {
            for (size_t i = 0; i < multi_got.size(); i++) {
                if (multi_got[i] != multi_ref[i]) bad++;
            }
        }
        std::fprintf(stderr,
                     "PQ2_0 host embed_tokens (%d tokens, bit-exact): %s\n",
                     n_tok, bad == 0 ? "PASS" : "FAIL");
        ok = ok && bad == 0;
        ds4_gpu_tensor_free(multi);
        ds4_gpu_tensor_free(tok);
    }
    ds4_gpu_tensor_free(out);

    /* Dense matmul through the graph entry, decode and prefill batch. */
    const shape s = {5120, 6144};
    const uint64_t w_bytes = (uint64_t)s.M * (s.K / QK) * sizeof(block_pq2_0_test);
    if (ok && w_bytes > arena_bytes) {
        std::fprintf(stderr, "host wiring: arena too small for matmul\n");
        ok = false;
    }
    if (ok) {
        std::vector<block_pq2_0_test> w(s.M * (s.K / QK));
        fill_blocks(w);
        std::memcpy(arena, w.data(), w.size() * sizeof(w[0]));

        for (const int n_tok : {1, 4}) {
            std::vector<float> x((size_t)n_tok * s.K);
            for (float &v : x) v = 0.5f * frand();
            std::vector<float> ref_mm((size_t)n_tok * s.M, 0.0f);
            for (int t = 0; t < n_tok; t++) {
                ds4_test_pq2_0_ref_matvec(w.data(), (uint64_t)s.M, (uint64_t)s.K,
                                          x.data() + (size_t)t * s.K,
                                          ref_mm.data() + (size_t)t * s.M);
            }
            ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc(
                (uint64_t)n_tok * s.K * sizeof(float));
            ds4_gpu_tensor *go = ds4_gpu_tensor_alloc(
                (uint64_t)n_tok * s.M * sizeof(float));
            std::vector<float> got_mm(ref_mm.size());
            const bool ran = gx && go &&
                ds4_gpu_tensor_write(gx, 0, x.data(), x.size() * sizeof(float)) &&
                ds4_gpu_matmul_pq2_0_tensor(go, arena, arena_bytes, 0,
                                            (uint64_t)s.K, (uint64_t)s.M, gx,
                                            (uint64_t)n_tok) &&
                ds4_gpu_tensor_read(go, 0, got_mm.data(),
                                    got_mm.size() * sizeof(float));
            char label[128];
            std::snprintf(label, sizeof(label),
                          "host matmul_quant M=%d K=%d N=%d", s.M, s.K, n_tok);
            const bool this_ok = ran && relative_l2_ok(got_mm, ref_mm, 0.05f, label);
            if (!ran) {
                std::fprintf(stderr, "%s: host dispatch failed\n", label);
            }
            ok = ok && this_ok;
            ds4_gpu_tensor_free(go);
            ds4_gpu_tensor_free(gx);
        }
    }

    munmap(arena, arena_bytes);
    return ok;
}

/* 5. GDN output norm gates.  The kernel is shared with the qwen4 family, so
 * both gates are checked against a double-precision reference: sigmoid (the
 * qwen4exp call) and silu (this family).  A kernel that ignored the selector
 * would fail one of the two. */
bool test_gdn_out_gates() {
    constexpr uint32_t T = 3, H = 48, D = 128;
    constexpr float eps = 1e-6f;
    const uint64_t n = (uint64_t)T * H * D;
    const uint64_t arena_bytes = (uint64_t)1 << 20;

    void *arena = mmap(nullptr, arena_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) return false;
    std::vector<float> gamma(D);
    for (uint32_t i = 0; i < D; i++) gamma[i] = 0.5f + 0.5f * std::fabs(frand());
    std::memcpy(arena, gamma.data(), D * sizeof(float));

    std::vector<float> z(n), base(n);
    for (uint64_t i = 0; i < n; i++) {
        z[i] = 4.0f * frand();
        base[i] = frand();
    }

    ds4_gpu_tensor *gout = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *gz = ds4_gpu_tensor_alloc(n * sizeof(float));
    bool ok = gout && gz &&
        ds4_gpu_tensor_write(gz, 0, z.data(), n * sizeof(float));
    if (!ok) {
        std::fprintf(stderr, "gdn out: tensor setup failed\n");
        ds4_gpu_tensor_free(gout);
        ds4_gpu_tensor_free(gz);
        munmap(arena, arena_bytes);
        return false;
    }

    for (int silu = 0; silu <= 1; silu++) {
        std::vector<float> want(n), got(n);
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t h = 0; h < H; h++) {
                const uint64_t base_i = ((uint64_t)t * H + h) * D;
                double ss = 0.0;
                for (uint32_t i = 0; i < D; i++) {
                    ss += (double)base[base_i + i] * (double)base[base_i + i];
                }
                const double r = 1.0 / std::sqrt(ss / D + eps);
                for (uint32_t i = 0; i < D; i++) {
                    const double zv = z[base_i + i];
                    const double sig = 1.0 / (1.0 + std::exp(-zv));
                    const double gate = silu ? zv * sig : sig;
                    want[base_i + i] = (float)(base[base_i + i] * r * gamma[i] * gate);
                }
            }
        }
        const bool ran =
            ds4_gpu_tensor_write(gout, 0, base.data(), n * sizeof(float)) &&
            (silu ? ds4_gpu_qwen35_gdn_out_tensor(gout, gz, arena, arena_bytes, 0,
                                                  T, H, D, eps) != 0
                  : ds4_gpu_qwen4_gdn_out_tensor(gout, gz, arena, arena_bytes, 0,
                                                 T, H, D, eps) != 0) &&
            ds4_gpu_tensor_read(gout, 0, got.data(), n * sizeof(float));
        if (!ran) {
            std::fprintf(stderr, "gdn out: %s run failed\n", silu ? "silu" : "sigmoid");
            ok = false;
            continue;
        }
        char label[128];
        std::snprintf(label, sizeof(label), "gdn out norm, %s gate",
                      silu ? "silu (Bonsai)" : "sigmoid (qwen4exp)");
        ok = close_enough(got, want, 1e-5f, 1e-4f, label) && ok;
    }

    ds4_gpu_tensor_free(gout);
    ds4_gpu_tensor_free(gz);
    munmap(arena, arena_bytes);
    return ok;
}

/* Bonsai attention core: the row-exact kernel run with the split-K partial
 * buffer the CUDA graph hands it (splits up to 64) against the same kernel run
 * without one (a single split), both measured against the double-precision
 * definition of the attention this family computes, at its own geometry
 * (D = 256, H = 24, Hkv = 4, six query heads per kv head).
 *
 * The key ranges are long on purpose.  The dispatcher splits only past 32 keys
 * per split (splits = min(64, (keys + 31) / 32)), so a short case would
 * silently test the single-split path again; pos0 = 32768 reaches the split
 * ceiling of 64 that the graph sizes its partial buffer for.  T = 1 is the
 * per-token decode shape, T = 16 the row batch the chunked prefill hands the
 * kernel.
 *
 * The two orders are compared to each other as well, and the split order is
 * the accurate one at long range: one online-softmax chain over 32K keys
 * drifts roughly 1e-6 absolute while the same sum in 64 pieces reduces to
 * roughly 1e-8 from the double-precision value. */
static bool test_attention_split() {
    constexpr uint32_t H = 24, Hkv = 4, D = 256;
    constexpr uint32_t splits_max = 64;                 /* dispatcher ceiling */
    constexpr float scale = 1.0f / 16.0f;               /* 1/sqrt(D) */
    constexpr float tol = 3e-5f;                        /* against the oracle */
    /* tile == true marks the shapes where a NULL partial with T >= 32 selects
     * the token-tile MMA kernel (warp score tile + one expf per (column, key)),
     * which is the path the graph hands a prefilled chunk to; tile == false
     * keeps the row-exact kernel, single-split or split-K. */
    struct attn_case { uint32_t T, pos0; const char *label; bool tile; };
    const attn_case cases[] = {
        { 1,  2048, "attention split: decode, ctx 2048", false},
        { 1, 32768, "attention split: decode, ctx 32768", false},
        {16,  2048, "attention split: chunk rows, ctx 2048", false},
        {16, 32768, "attention split: chunk rows, ctx 32768", false},
        {32,  2048, "attention token-tile: chunk 32, ctx 2048", true},
        {32, 32768, "attention token-tile: chunk 32, ctx 32768", true},
        {33,  4096, "attention token-tile: odd chunk 33, ctx 4096", true},
        {64, 32768, "attention token-tile: chunk 64, ctx 32768", true},
    };

    bool ok = true;
    for (const attn_case &c : cases) {
        const uint32_t T = c.T, pos0 = c.pos0;
        const uint32_t keys = pos0 + T;
        const uint64_t qn = (uint64_t)T * H * D, kvn = (uint64_t)keys * Hkv * D;

        std::vector<float> q(qn), gate(qn);
        std::vector<__half> kc(kvn), vc(kvn);
        for (uint64_t i = 0; i < qn; i++) {
            /* Row amplitudes span the softmax's two regimes: peaked, flat and
             * ordinary, so the split reduction is compared on both tails. */
            const float amp = ((i / D) % 3u == 0u) ? 4.0f
                            : ((i / D) % 3u == 1u) ? 0.001f : 1.0f;
            q[i] = amp * frand();
            gate[i] = 4.0f * frand();
        }
        for (uint64_t i = 0; i < kvn; i++) {
            kc[i] = __float2half_rn(frand());
            vc[i] = __float2half_rn(frand());
        }

        ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(qn * 4);
        ds4_gpu_tensor *ggate = ds4_gpu_tensor_alloc(qn * 4);
        ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(kvn * 2);
        ds4_gpu_tensor *gv = ds4_gpu_tensor_alloc(kvn * 2);
        ds4_gpu_tensor *gref = ds4_gpu_tensor_alloc(qn * 4);
        ds4_gpu_tensor *gsplit = ds4_gpu_tensor_alloc(qn * 4);
        ds4_gpu_tensor *partial =
            ds4_gpu_tensor_alloc((uint64_t)T * H * splits_max * (D + 2) * 4);
        if (!gq || !ggate || !gk || !gv || !gref || !gsplit || !partial ||
            !ds4_gpu_tensor_write(gq, 0, q.data(), qn * 4) ||
            !ds4_gpu_tensor_write(ggate, 0, gate.data(), qn * 4) ||
            !ds4_gpu_tensor_write(gk, 0, kc.data(), kvn * 2) ||
            !ds4_gpu_tensor_write(gv, 0, vc.data(), kvn * 2)) {
            std::fprintf(stderr, "%s: tensor setup failed\n", c.label);
            ok = false;
            continue;
        }

        /* gref is the NULL-partial call, i.e. the single-split row-exact kernel
         * for T < 32 and the token-tile MMA kernel for T >= 32; gsplit is the
         * split-K row-exact one.  Which of the two is binding depends on the
         * shape: see the comparison below. */
        const bool ran =
            ds4_gpu_qwen4_attn_decode_tensor(gref, gq, ggate, gk, gv, NULL, NULL, NULL,
                                             T, H, Hkv, D, pos0, false, 0u, scale) &&
            ds4_gpu_qwen4_attn_decode_tensor(gsplit, gq, ggate, gk, gv, NULL, NULL, partial,
                                             T, H, Hkv, D, pos0, false, 0u, scale);
        if (!ran) {
            std::fprintf(stderr, "%s: run failed\n", c.label);
            ok = false;
            goto cleanup;
        }

        {
            std::vector<float> ref(qn), got(qn);
            if (!ds4_gpu_tensor_read(gref, 0, ref.data(), qn * 4) ||
                !ds4_gpu_tensor_read(gsplit, 0, got.data(), qn * 4)) {
                std::fprintf(stderr, "%s: readback failed\n", c.label);
                ok = false;
                goto cleanup;
            }
            char label[128];
            std::snprintf(label, sizeof(label), "%s (T=%u pos0=%u)", c.label, T, pos0);

            /* The two orders differ by fp32 accumulation alone, and the gap
             * grows with the key range: the single-split kernel runs one
             * online-softmax chain over every key, the split kernel one chain
             * per 512-key range plus a 64-way merge.  Measured, the split
             * order is the closer of the two to the double-precision value
             * (see the oracle below), so this bound only has to catch a
             * mis-wired split - a wrong range or a dropped split moves the
             * output by a fraction of the scale, not by a rounding step. */
            double split_vs_single = 0.0, ref_scale = 0.0;
            for (uint64_t i = 0; i < qn; i++) {
                split_vs_single = std::fmax(split_vs_single,
                                            std::fabs((double)got[i] - (double)ref[i]));
                ref_scale = std::fmax(ref_scale, std::fabs((double)ref[i]));
            }
            /* Below the token-tile gate the two kernels differ by reduction
             * order alone, and this bound is what catches a mis-wired split
             * (a wrong range or a dropped split moves the output by a fraction
             * of the scale, not by a rounding step).  Above the gate the NULL
             * partial selects a different kernel entirely, so the same
             * comparison is a cross-kernel check with the looser bound the
             * MMA score tile needs. */
            const double order_limit = (c.tile ? 1e-3 : 1e-4) * std::fmax(ref_scale, 1e-8);
            std::fprintf(stderr, "%s: %s max|d|=%.2e (order limit %.2e): %s\n",
                         label, c.tile ? "token-tile vs split" : "split vs single",
                         split_vs_single, order_limit,
                         split_vs_single <= order_limit ? "PASS" : "FAIL");
            ok = split_vs_single <= order_limit && ok;

            /* Independent check of the values themselves against the same
             * definition the kernels implement, in double precision: four
             * rows spanning the batch and six heads spanning the kv groups. */
            std::vector<double> acc(D);
            double worst_split = 0.0, worst_ref = 0.0, peak_scale = 0.0;
            const uint32_t rows[4] = { 0, T / 3, 2 * T / 3, T - 1 };
            const uint32_t heads[6] = { 0, 4, 9, 14, 19, H - 1 };
            for (uint32_t ti = 0; ti < 4; ti++) {
                const uint32_t t = rows[ti];
                for (uint32_t hi = 0; hi < 6; hi++) {
                    const uint32_t h = heads[hi];
                    const uint32_t kh = h / (H / Hkv);
                    const uint32_t n = pos0 + t + 1;
                    double peak = -INFINITY, denom = 0.0;
                    std::vector<double> scores(n);
                    for (uint32_t j = 0; j < n; j++) {
                        double dot = 0.0;
                        for (uint32_t d = 0; d < D; d++) {
                            dot += (double)q[((uint64_t)t * H + h) * D + d] *
                                   (double)__half2float(kc[((uint64_t)j * Hkv + kh) * D + d]);
                        }
                        scores[j] = dot * (double)scale;
                        peak = std::fmax(peak, scores[j]);
                    }
                    for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
                    for (uint32_t j = 0; j < n; j++) {
                        const double w = std::exp(scores[j] - peak);
                        denom += w;
                        for (uint32_t d = 0; d < D; d++) {
                            acc[d] += w * (double)__half2float(vc[((uint64_t)j * Hkv + kh) * D + d]);
                        }
                    }
                    for (uint32_t d = 0; d < D; d++) {
                        const uint64_t i = ((uint64_t)t * H + h) * D + d;
                        const double want = denom > 0.0
                            ? acc[d] / denom / (1.0 + std::exp(-(double)gate[i])) : 0.0;
                        peak_scale = std::fmax(peak_scale, std::fabs(want));
                        worst_split = std::fmax(worst_split, std::fabs((double)got[i] - want));
                        worst_ref = std::fmax(worst_ref, std::fabs((double)ref[i] - want));
                    }
                }
            }
            /* The path that ships for this shape is the binding check and the
             * other one is its control:
             *  - tile cases run on the token-tile MMA kernel (gref), bound to
             *    the oracle at tol, with the split-K order as the control;
             *  - every other case runs on the split-K kernel (gsplit), bound
             *    to the oracle, with the single-split chain as the control.
             * The control's own distance from the oracle grows with the key
             * range because it runs one fp32 online-softmax chain over every
             * key: measured 4.7e-6 of the scale at pos0 = 2048 and 9.7e-5 at
             * pos0 = 32768, i.e. ~20x and ~200x worse than the order it is
             * compared against, which is why the long cases get the looser
             * bound. */
            const double split_limit = tol * std::fmax(peak_scale, 1e-8);
            const double ref_limit = (keys >= 8192u ? 3e-4 : tol) * std::fmax(peak_scale, 1e-8);
            const double bind   = c.tile ? worst_ref   : worst_split;
            const double ctrl   = c.tile ? worst_split : worst_ref;
            const double bind_lim = c.tile ? tol * std::fmax(peak_scale, 1e-8) : split_limit;
            const double ctrl_lim = c.tile ? split_limit : ref_limit;
            const bool bind_ok = bind <= bind_lim;
            const bool ctrl_ok = ctrl <= ctrl_lim;
            std::fprintf(stderr,
                         "%s: %s max|d|=%.2e (limit %.2e) %s; %s max|d|=%.2e (limit %.2e) %s\n",
                         label, c.tile ? "token-tile" : "split", bind, bind_lim,
                         bind_ok ? "PASS" : "FAIL", c.tile ? "split" : "single", ctrl,
                         ctrl_lim, ctrl_ok ? "PASS" : "FAIL");
            ok = bind_ok && ctrl_ok && ok;
        }

cleanup:
        ds4_gpu_tensor_free(partial);
        ds4_gpu_tensor_free(gsplit);
        ds4_gpu_tensor_free(gref);
        ds4_gpu_tensor_free(gv);
        ds4_gpu_tensor_free(gk);
        ds4_gpu_tensor_free(ggate);
        ds4_gpu_tensor_free(gq);
    }
    return ok;
}



/* 6. bf16 gated delta-net scalars.  ssm_alpha and ssm_beta are the only bf16
 * weights in the artifact, 96 of them at the real (out_dim 48, in_dim 5120).
 * A prefill chunk runs them through the token-tiled kernel, a decode step
 * through the untiled one.  The tiled form only interleaves the loops, so the
 * two must agree bit-for-bit at every width, and both must sit inside the
 * float-rounding band of a double-precision dot.  T=8 is the tile boundary the
 * dispatcher uses, T=9 the first tiled width, 486 the chunk this host really
 * prefills. */
bool test_bf16_matvec_tile() {
    constexpr int out_dim = 48;   // ssm_alpha / ssm_beta rows
    constexpr int in_dim = 5120;  // qwen35.embedding_length
    constexpr uint64_t arena_bytes = (uint64_t)out_dim * in_dim * 2;

    std::mt19937 rng(20260930u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> w((size_t)out_dim * in_dim);
    for (__nv_bfloat16 &v : w) v = __float2bfloat16(dist(rng));

    void *arena = mmap(nullptr, arena_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) {
        std::fprintf(stderr, "bf16 tile: mmap failed\n");
        return false;
    }
    std::memcpy(arena, w.data(), w.size() * sizeof(w[0]));

    bool ok = true;
    if (ds4_gpu_init() != 1 || ds4_gpu_set_model_map(arena, arena_bytes) != 1) {
        std::fprintf(stderr, "bf16 tile: backend/model-map setup failed\n");
        munmap(arena, arena_bytes);
        return false;
    }

    for (const int T : {1, 8, 9, 11, 486}) {
        const uint64_t x_count = (uint64_t)in_dim * T;
        const uint64_t o_count = (uint64_t)out_dim * T;
        std::vector<float> x(x_count), base(o_count), tiled(o_count), ref(o_count);
        for (float &v : x) v = dist(rng);

        ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc(x_count * sizeof(float));
        ds4_gpu_tensor *gout = ds4_gpu_tensor_alloc(o_count * sizeof(float));
        const bool wrote = gx && gout &&
            ds4_gpu_tensor_write(gx, 0, x.data(), x_count * sizeof(float));

        bool this_ok = wrote;
        if (wrote) {
            setenv("DS4_QWEN35_BF16_MATVEC_TILED", "0", 1);
            this_ok = ds4_gpu_qwen35_matvec_bf16_tensor(
                          gout, gx, arena, arena_bytes, 0, in_dim, out_dim,
                          (uint32_t)T) &&
                      ds4_gpu_tensor_read(gout, 0, base.data(),
                                          o_count * sizeof(float));
            unsetenv("DS4_QWEN35_BF16_MATVEC_TILED");
            this_ok = this_ok &&
                      ds4_gpu_qwen35_matvec_bf16_tensor(
                          gout, gx, arena, arena_bytes, 0, in_dim, out_dim,
                          (uint32_t)T) &&
                      ds4_gpu_tensor_read(gout, 0, tiled.data(),
                                          o_count * sizeof(float));
        }

        /* Reference: exact bf16 weights against the float activation, in
         * double, so only the kernel's f32 accumulation separates them. */
        for (int t = 0; t < T; t++) {
            for (int row = 0; row < out_dim; row++) {
                double acc = 0.0;
                for (int i = 0; i < in_dim; i++) {
                    acc += (double)__bfloat162float(w[(size_t)row * in_dim + i]) *
                           (double)x[(size_t)t * in_dim + i];
                }
                ref[(size_t)t * out_dim + row] = (float)acc;
            }
        }

        if (this_ok && std::memcmp(base.data(), tiled.data(),
                                   o_count * sizeof(float)) != 0) {
            for (uint64_t i = 0; i < o_count; i++) {
                if (base[i] != tiled[i]) {
                    std::fprintf(stderr,
                                 "bf16 tile T=%d: tiled %g != untiled %g at %llu\n",
                                 T, tiled[i], base[i],
                                 (unsigned long long)i);
                    break;
                }
            }
            this_ok = false;
        }

        double num = 0.0, den = 0.0, max_abs = 0.0;
        if (this_ok) {
            for (uint64_t i = 0; i < o_count; i++) {
                const double d = (double)tiled[i] - (double)ref[i];
                num += d * d;
                den += (double)ref[i] * (double)ref[i];
                max_abs = std::max(max_abs, std::fabs(d));
            }
            const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
            if (rel > 1e-5) {
                std::fprintf(stderr, "bf16 tile T=%d: relative L2 %.3g over the "
                                     "float-rounding band\n", T, rel);
                this_ok = false;
            } else {
                std::fprintf(stderr, "bf16 tile T=%d: bit-identical to the "
                                     "untiled kernel, max abs %.3g, rel L2 %.3g\n",
                             T, max_abs, rel);
            }
        }

        if (gx) ds4_gpu_tensor_free(gx);
        if (gout) ds4_gpu_tensor_free(gout);
        ok = this_ok && ok;
    }

    munmap(arena, arena_bytes);
    std::fprintf(stderr, "bf16 matvec tile parity: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main() {
    if (ds4_mmq_init(0) != 0) {
        std::fprintf(stderr, "ds4_mmq_init failed\n");
        return 1;
    }

    const bool rows_ok = test_row_lookup();
    const bool guards_ok = test_shape_guards();
    const bool host_ok = test_host_wiring();
    const bool fold_ok = test_fold_transform();
    const bool gdn_ok = test_gdn_out_gates();
    const bool attn_ok = test_attention_split();
    const bool bf16_ok = test_bf16_matvec_tile();

    /* Random activations: the kernels quantize the activation to the Q8_1
     * form, so the outputs are compared by relative L2 error against the
     * double-precision reference (criterion documented above). */
    bool shapes_ok = true;
    for (const shape &s : kShapes) {
        shapes_ok = run_shape(s, 1, "random1", false) && shapes_ok;
    }

    /* Prefill batches: one column tile boundary (8), a MMQ_DP4A-sized batch
     * (64) and a 256-column batch. */
    shapes_ok = run_shape(kShapes[0], 8, "prefill8", false) && shapes_ok;
    shapes_ok = run_shape(kShapes[1], 64, "prefill64", false) && shapes_ok;
    shapes_ok = run_shape(kShapes[2], 256, "prefill256", false) && shapes_ok;

    /* Exact-activation pass: near-float-rounding agreement is required, which
     * is what makes a wrong tile index visible. */
    bool exact_ok = true;
    for (const shape &s : kShapes) {
        exact_ok = run_shape(s, 1, "exact1", true) && exact_ok;
    }
    exact_ok = run_shape(kShapes[0], 64, "exact64", true) && exact_ok;

    const bool ok = rows_ok && guards_ok && host_ok && fold_ok && gdn_ok &&
                    attn_ok && bf16_ok && shapes_ok && exact_ok;
    std::fprintf(stderr, "PQ2_0 CUDA parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
