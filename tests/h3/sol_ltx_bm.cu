/*
 * The refiner's Sol-Attn sparse forward, scalar arm against tensor-core arm,
 * on one plan at the shape stage 2 runs it at, for each of the release's three
 * taus (`sol_attention.py:18` STAGE2_TAUS).
 *
 *   tests/h3/sol_ltx_bm            # 16,128 x 32 heads, tau 1.0 / 1.25 / 1.5
 *   tests/h3/sol_ltx_bm 16128 3    # three timed passes per arm
 *
 * Both arms read the same `struct h3d_sol_plan`, built by the same
 * `h3d_sol_ltx_search` the refiner calls, so the mask - and therefore the work
 * - is identical between them. `h3d_sol_ltx_forward` is the scalar walk the
 * port shipped (`h3d_sol_ltx_forward_kernel`, 430.8 ms of a 638.0 ms block);
 * `h3d_sol_ltx_forward_mma` is the tower's sparse walk re-expressed for this
 * port's layout.
 *
 * The FLOPs are counted from the mask, not from the dense shape: a folded
 * block costs one 32-wide column of the fold, an exact block costs its 64 keys
 * in both passes. `exact` is reported next to the rate so the rate can be read
 * against the work it was bought with.
 *
 * Build: `make tests/h3/sol_ltx_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum { HEADS = 32, HEAD_DIM = 128 };

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

/* One exact 64-key block in both passes, plus one folded 32-block group in
 * both passes: `2 * q * k * dim` twice for the block, `2 * q * 32 * dim` twice
 * for the fold, at 64 query rows per CTA. */
static double sparse_flops(uint64_t exact_blocks, uint64_t q_blocks,
                           uint64_t words) {
    const double block = 4.0 * 64.0 * 64.0 * (double)HEAD_DIM;
    const double fold = 4.0 * 64.0 * 32.0 * (double)HEAD_DIM;
    return (double)exact_blocks * block +
           (double)q_blocks * (double)HEADS * (double)words * fold;
}

int main(int argc, char **argv) {
    const uint32_t rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 16128;
    const int passes = argc > 2 ? std::atoi(argv[2]) : 3;
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    const uint32_t blocks = (rows + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    const uint32_t words = h3d_sol_words(rows);
    const size_t elements = (size_t)rows * HEADS * HEAD_DIM;
    const size_t summary = (size_t)blocks * HEADS * HEAD_DIM;

    auto alloc = [&](int dtype, size_t count) {
        h3d_tensor *tensor = h3d_alloc(gpu, dtype, count, error, sizeof(error));
        if (!tensor) fail("alloc", error);
        return tensor;
    };
    std::vector<__nv_bfloat16> host;
    auto upload = [&](size_t count, uint32_t seed) {
        h3d_tensor *tensor = alloc(H3D_BF16, count);
        host.resize(count);
        fill(host, seed);
        if (!h3d_upload(gpu, tensor, host.data(), error, sizeof(error))) {
            fail("upload", error);
        }
        return tensor;
    };

    h3d_tensor *query = upload(elements, 11u);
    h3d_tensor *key = upload(elements, 13u);
    h3d_tensor *value = upload(elements, 17u);
    h3d_tensor *scalar = alloc(H3D_BF16, elements);
    h3d_tensor *mma = alloc(H3D_BF16, elements);
    struct h3d_sol_plan plan = {};
    plan.kc = alloc(H3D_BF16, summary);
    plan.vc = alloc(H3D_BF16, summary);
    plan.mask = alloc(H3D_U32, (size_t)blocks * HEADS * words);
    plan.count = alloc(H3D_U32, (size_t)blocks * HEADS);
    plan.threshold = alloc(H3D_F32, (size_t)blocks * HEADS);
    plan.query_bar = alloc(H3D_F32, summary);
    plan.kc_mean = alloc(H3D_F32, (size_t)HEADS * HEAD_DIM);
    plan.kc_var = alloc(H3D_F32, (size_t)HEADS * HEAD_DIM);
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);

    std::vector<uint32_t> counts((size_t)blocks * HEADS);
    const float taus[3] = {1.0f, 1.25f, 1.5f};
    for (int which = 0; which < 3; which++) {
        const float tau = taus[which];
        if (!h3d_sol_ltx_search(gpu, &plan, query, key, value, rows, HEADS,
                                scale, tau, error, sizeof(error))) {
            fail("search", error);
        }
        if (!h3d_read(gpu, counts.data(), plan.count, error, sizeof(error))) {
            fail("read", error);
        }
        uint64_t exact = 0;
        for (uint32_t item : counts) exact += item;

        auto run_scalar = [&]() {
            return h3d_sol_ltx_forward(gpu, scalar, &plan, query, key, value,
                                       rows, HEADS, scale, error,
                                       sizeof(error));
        };
        auto run_mma = [&]() {
            return h3d_sol_ltx_forward_mma(gpu, mma, &plan, query, key, value,
                                           rows, HEADS, scale, error,
                                           sizeof(error));
        };
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
                ms[arm] += std::chrono::duration<double, std::milli>(finish -
                                                                    start)
                                   .count();
            }
            ms[arm] /= passes;
        }

        std::vector<__nv_bfloat16> got_scalar(elements), got_mma(elements);
        if (!h3d_read(gpu, got_scalar.data(), scalar, error, sizeof(error))) {
            fail("read", error);
        }
        if (!h3d_read(gpu, got_mma.data(), mma, error, sizeof(error))) {
            fail("read", error);
        }

        double square = 0.0, scale_max = 0.0, worst = 0.0;
        size_t differing = 0;
        for (size_t index = 0; index < elements; index++) {
            const double a = __bfloat162float(got_scalar[index]);
            const double b = __bfloat162float(got_mma[index]);
            if (a != b) differing++;
            const double delta = a - b;
            square += delta * delta;
            scale_max = std::fmax(scale_max, std::fabs(a));
            worst = std::fmax(worst, std::fabs(delta));
        }
        const double rms = sqrt(square / (double)elements);
        uint64_t sign_flips = 0;
        for (size_t row = 0; row < (size_t)rows * HEADS; row++) {
            double sum_a = 0.0, sum_b = 0.0;
            for (uint32_t dim = 0; dim < HEAD_DIM; dim++) {
                sum_a += __bfloat162float(got_scalar[row * HEAD_DIM + dim]);
                sum_b += __bfloat162float(got_mma[row * HEAD_DIM + dim]);
            }
            if ((sum_a < 0.0) != (sum_b < 0.0)) sign_flips++;
        }

        const double flops = sparse_flops(exact, blocks, words);
        std::printf(
                "tau=%.2f exact=%5.2f%% scalar_ms=%9.3f scalar_tflops=%6.2f "
                "mma_ms=%9.3f mma_tflops=%6.2f gain=%5.2fx\n",
                tau, 100.0 * (double)exact / ((double)blocks * (double)HEADS *
                                              (double)blocks),
                ms[0], flops / (ms[0] * 1e9), ms[1], flops / (ms[1] * 1e9),
                ms[0] / ms[1]);
        std::printf(
                "  compare differing=%.4f%% rel_rms=%.3e max_abs=%.4g "
                "scale=%.4g row_sign_flips=%llu of %u\n",
                100.0 * (double)differing / (double)elements, rms / scale_max,
                worst, scale_max, (unsigned long long)sign_flips,
                (unsigned)((size_t)rows * HEADS));
    }
    return 0;
}
