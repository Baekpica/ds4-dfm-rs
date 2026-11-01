/*
 * The refiner's dense attention, scalar arm against tensor-core arm, on the
 * same operands at the two shapes stage 2 runs it at.
 *
 *   tests/h3/ltx_attn_bm            # 16,128 x 16,128 and 16,128 x 1,024
 *   tests/h3/ltx_attn_bm 16128 3    # three passes for the mean
 *
 * `h3d_ltx_attention` is the token-major scalar walk the port shipped
 * (`h3d_ltx_attention_kernel`), and it is the kernel `h3_ltx_refiner_gate`
 * holds bit-identical to the reference. `h3d_ltx_attention_mma` is the
 * token-major port of the tower's `mma` kernel. They are two arithmetic
 * classes, so this reports both what each costs and how far apart they are:
 * `rel_rms` is the number the golden's `--compare` would show, not a bit count,
 * and `argmax` agreement is the check that matters for an attention - a
 * difference that moves no row's largest attention weight is a rounding
 * difference and not a different function.
 *
 * Build: `make tests/h3/ltx_attn_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum { HEADS = 32, HEAD_DIM = 128, CONTEXT_ROWS = 1024 };

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

/* Deterministic values in [-1, 1) as BF16, token-major `[rows, heads, dim]`. */
static void fill(std::vector<__nv_bfloat16> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = __float2bfloat16((float)(state >> 8) / 8388608.0f - 1.0f);
    }
}

static double flops(uint32_t q_rows, uint32_t kv_rows) {
    /* Scores and the weighted sum, both `2 * rows * keys * dim` per head. */
    return 4.0 * (double)q_rows * (double)kv_rows * (double)HEAD_DIM *
           (double)HEADS;
}

int main(int argc, char **argv) {
    const uint32_t rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 16128;
    const int passes = argc > 2 ? std::atoi(argv[2]) : 3;
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    /* The two dense shapes: the video self-attention (square) and the text
     * cross-attention (16,128 queries against the prompt's 1,024 rows). */
    const uint32_t pairs[2][2] = {{rows, rows}, {rows, CONTEXT_ROWS}};
    const char *names[2] = {"square", "cross"};

    for (int shape = 0; shape < 2; shape++) {
        const uint32_t q_rows = pairs[shape][0], kv_rows = pairs[shape][1];
        const size_t q_elements = (size_t)q_rows * HEADS * HEAD_DIM;
        const size_t kv_elements = (size_t)kv_rows * HEADS * HEAD_DIM;

        std::vector<__nv_bfloat16> host;
        auto upload = [&](size_t elements, uint32_t seed) {
            h3d_tensor *tensor = h3d_alloc(gpu, H3D_BF16, elements, error,
                                           sizeof(error));
            if (!tensor) fail("alloc", error);
            host.resize(elements);
            fill(host, seed);
            if (!h3d_upload(gpu, tensor, host.data(), error, sizeof(error))) {
                fail("upload", error);
            }
            return tensor;
        };
        h3d_tensor *query = upload(q_elements, 11u);
        h3d_tensor *key = upload(kv_elements, 13u);
        h3d_tensor *value = upload(kv_elements, 17u);
        h3d_tensor *scalar = h3d_alloc(gpu, H3D_BF16, q_elements, error,
                                       sizeof(error));
        h3d_tensor *mma = h3d_alloc(gpu, H3D_BF16, q_elements, error,
                                    sizeof(error));
        if (!scalar || !mma) fail("alloc", error);
        const float scale = 1.0f / sqrtf((float)HEAD_DIM);

        auto run_scalar = [&]() {
            return h3d_ltx_attention(gpu, scalar, query, key, value, q_rows,
                                     kv_rows, HEADS, HEAD_DIM, scale, error,
                                     sizeof(error));
        };
        auto run_mma = [&]() {
            return h3d_ltx_attention_mma(gpu, mma, query, key, value, q_rows,
                                         kv_rows, HEADS, HEAD_DIM, scale, error,
                                         sizeof(error));
        };

        /* One warm-up pass each, then the timed arms. */
        if (!run_scalar()) fail("scalar warm-up", error);
        if (!run_mma()) fail("mma warm-up", error);
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);

        double ms[2] = {};
        for (int arm = 0; arm < 2; arm++) {
            for (int pass = 0; pass < passes; pass++) {
                auto start = std::chrono::steady_clock::now();
                if (arm == 0 ? !run_scalar() : !run_mma()) fail("arm", error);
                if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
                auto finish = std::chrono::steady_clock::now();
                ms[arm] +=
                        std::chrono::duration<double, std::milli>(finish - start)
                                .count();
            }
            ms[arm] /= passes;
        }

        std::vector<__nv_bfloat16> got_scalar(q_elements), got_mma(q_elements);
        if (!h3d_read(gpu, got_scalar.data(), scalar, error, sizeof(error))) {
            fail("read", error);
        }
        if (!h3d_read(gpu, got_mma.data(), mma, error, sizeof(error))) {
            fail("read", error);
        }

        /* Difference, and whether it moves a row's argmax weight. */
        double square = 0.0, scale_max = 0.0, worst = 0.0;
        size_t differing = 0;
        for (size_t index = 0; index < q_elements; index++) {
            const double a = __bfloat162float(got_scalar[index]);
            const double b = __bfloat162float(got_mma[index]);
            if (a != b) differing++;
            const double delta = a - b;
            square += delta * delta;
            scale_max = std::fmax(scale_max, std::fabs(a));
            worst = std::fmax(worst, std::fabs(delta));
        }
        const double rms = sqrt(square / (double)q_elements);
        uint64_t sign_flips = 0;
        for (size_t row = 0; row < (size_t)q_rows * HEADS; row++) {
            double sum_a = 0.0, sum_b = 0.0;
            for (uint32_t dim = 0; dim < HEAD_DIM; dim++) {
                sum_a += __bfloat162float(got_scalar[row * HEAD_DIM + dim]);
                sum_b += __bfloat162float(got_mma[row * HEAD_DIM + dim]);
            }
            if ((sum_a < 0.0) != (sum_b < 0.0)) sign_flips++;
        }

        std::printf(
                "shape=%-6s q=%-5u kv=%-5u scalar_ms=%9.3f scalar_tflops=%6.2f "
                "mma_ms=%9.3f mma_tflops=%6.2f gain=%.2fx\n",
                names[shape], q_rows, kv_rows, ms[0],
                flops(q_rows, kv_rows) / (ms[0] * 1e9), ms[1],
                flops(q_rows, kv_rows) / (ms[1] * 1e9), ms[0] / ms[1]);
        std::printf(
                "  compare differing=%.4f%% rel_rms=%.3e max_abs=%.4g "
                "scale=%.4g row_sign_flips=%llu of %u\n",
                100.0 * (double)differing / (double)q_elements, rms / scale_max,
                worst, scale_max, (unsigned long long)sign_flips,
                (unsigned)((size_t)q_rows * HEADS));
    }
    return 0;
}
