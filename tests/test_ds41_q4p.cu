/* test_ds41_q4p.cu — the V4.1 prefill q4_K projection entry
 * (ds4_gpu_v41_matmul_q4k_tensor, n > 8) against an FP64 dot of the exact
 * Q4_K weights (model-free).
 *
 * Inputs are BF16-valued, as every projection input of the model is.  The
 * entry takes glmvq_q4p by default and cuBLAS under DS4_DS41_Q4P=0 (make
 * test-ds41-q4p runs both): either way every rounded output must be a BF16
 * value within a relative RMS of the reference that covers its weight
 * rounding and the output rounding (2^-9 relative, about 1.1e-3 RMS).
 * Shapes are the model's (attn_kv, ffn_down_shexp) and one that q4p does not
 * take (rows not a multiple of 128); widths include a 9-row and a tail tile.
 * An unrounded call (the head) and a tensor whose scales leave q4p's F16
 * range must give the same accuracy on cuBLAS.  The prefill pair (two
 * projections of one input) and the attention output pair (block-diagonal
 * wo_a into wo_b) must give the bits of their projections run one by one.
 *
 * Build and run: make test-ds41-q4p */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <vector>

#include "ds4_gpu.h"

enum { BLK = 256, BYTES = 144 };
static const double BOUND = 3.5e-3;
static std::mt19937 rng(41);
enum { PAGE = 4096 };

static float half_to_float(const uint8_t *p) {
    const uint16_t h = (uint16_t)(p[0] | (p[1] << 8));
    const int e = (h >> 10) & 31, m = h & 1023;
    const float v = e ? ldexpf(1.0f + m / 1024.0f, e - 15) : ldexpf((float)m, -24);
    return h & 0x8000 ? -v : v;
}

static uint16_t float_to_half(float f) {
    int e;
    const float m = frexpf(fabsf(f), &e);   /* f = m 2^e, m in [0.5, 1) */
    const uint16_t sign = f < 0 ? 0x8000 : 0;
    if (e - 1 < -14) { return sign | (uint16_t)lrintf(fabsf(f) * 16777216.0f); }
    return sign | (uint16_t)((e - 1 + 15) << 10) | (uint16_t)lrintf((m * 2.0f - 1.0f) * 1024.0f);
}

static float bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&v, &u, 4);
    return v;
}

/* Element j of a Q4_K block: d sc q - dmin m (GGUF order). */
static double weight(const uint8_t *block, uint32_t j) {
    const uint8_t *q = block + 4;
    const uint32_t p = j / 64, within = j % 64, sub = 2 * p + (within >= 32), l = within % 32;
    const uint8_t byte = block[16 + 32 * p + l];
    const int value = within < 32 ? byte & 15 : byte >> 4;
    const int sc = sub < 4 ? q[sub] & 63 : (q[sub + 4] & 15) | ((q[sub - 4] >> 6) << 4);
    const int m = sub < 4 ? q[sub + 4] & 63 : (q[sub + 4] >> 4) | ((q[sub] >> 6) << 4);
    return (double)half_to_float(block) * sc * value - (double)half_to_float(block + 2) * m;
}

/* d and dmin in the hVQ skeleton's range (max 5.6e-3 and 6e-2). */
static std::vector<uint8_t> make_weights(uint32_t rows, uint32_t cols, bool out_of_range) {
    std::vector<uint8_t> w((size_t)rows * (cols / BLK) * BYTES);
    for (size_t b = 0; b < w.size() / BYTES; b++) {
        uint8_t *block = w.data() + b * BYTES;
        const uint16_t d = float_to_half(1e-4f + (rng() % 1000) * 2e-6f), dmin = float_to_half(1e-4f + (rng() % 1000) * 5e-6f);
        memcpy(block, &d, 2);
        memcpy(block + 2, &dmin, 2);
        for (uint32_t i = 4; i < BYTES; i++) { block[i] = (uint8_t)rng(); }
    }
    if (out_of_range) {
        const uint16_t one = float_to_half(1.0f);
        memcpy(w.data(), &one, 2);
        w[4] = 63;
    }
    return w;
}

/* Each weight maps its own pages, as a tensor of the model map would: the
 * range resolver registers whole pages and keys ranges by host address, so
 * buffers neither share a page nor are freed (an address could return). */
static uint8_t *pages(const std::vector<uint8_t> &host) {
    uint8_t *w = (uint8_t *)aligned_alloc(PAGE, (host.size() + PAGE - 1) / PAGE * PAGE);
    if (w) { memcpy(w, host.data(), host.size()); }
    return w;
}

static std::vector<float> bf16_rows(uint32_t tokens, uint32_t cols) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t)tokens * cols);
    for (auto &v : x) { v = bf16(nd(rng)); }
    return x;
}

/* y = W x through the single-projection entry (rounded). */
static bool project(const uint8_t *w, uint64_t wbytes, uint32_t rows, uint32_t cols, const std::vector<float> &x,
                    uint32_t tokens, std::vector<float> &y) {
    y.assign((size_t)tokens * rows, 0.0f);
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(x.size() * 4), *yt = ds4_gpu_tensor_alloc(y.size() * 4);
    const bool ok = xt && yt && ds4_gpu_tensor_write(xt, 0, x.data(), x.size() * 4) &&
        ds4_gpu_v41_matmul_q4k_tensor(yt, w, wbytes, 0, cols, rows, xt, tokens, 1) &&
        ds4_gpu_synchronize() && ds4_gpu_tensor_read(yt, 0, y.data(), y.size() * 4);
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(yt);
    return ok;
}

static int run_case(const char *name, uint32_t rows, uint32_t cols, uint32_t tokens, int round_out, bool out_of_range) {
    const std::vector<uint8_t> host = make_weights(rows, cols, out_of_range);
    const size_t wbytes = host.size();
    uint8_t *w = pages(host);
    if (!w) { printf("%-24s FAIL (alloc)\n", name); return 1; }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t)tokens * cols), y((size_t)tokens * rows);
    for (auto &v : x) { v = bf16(nd(rng)); }

    const uint64_t xb = x.size() * 4, yb = y.size() * 4;
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xb), *yt = ds4_gpu_tensor_alloc(yb);
    int ok = xt && yt && ds4_gpu_tensor_write(xt, 0, x.data(), xb) &&
        ds4_gpu_v41_matmul_q4k_tensor(yt, w, wbytes, 0, cols, rows, xt, tokens, round_out) &&
        ds4_gpu_synchronize() && ds4_gpu_tensor_read(yt, 0, y.data(), yb);
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(yt);
    if (!ok) { printf("%-24s FAIL (launch)\n", name); return 1; }

    /* Sampled outputs: error against the exact Q4_K dot, and the BF16 grid. */
    const uint32_t blocks = cols / BLK;
    double err = 0, ref2 = 0;
    uint32_t off_grid = 0;
    for (uint32_t i = 0; i < 2048; i++) {
        const uint32_t t = rng() % tokens, r = rng() % rows;
        double ref = 0;
        for (uint32_t k = 0; k < cols; k++) {
            ref += weight(w + ((size_t)r * blocks + k / BLK) * BYTES, k % BLK) * x[(size_t)t * cols + k];
        }
        const float got = y[(size_t)t * rows + r];
        err += (got - ref) * (got - ref);
        ref2 += ref * ref;
        off_grid += round_out && bf16(got) != got;
    }
    const double rel = sqrt(err / ref2);
    const int pass = rel < BOUND && !off_grid;
    printf("%-24s %s: rows=%u cols=%u tokens=%u relRMS %.3g, %u outputs off the BF16 grid\n", name, pass ? "PASS" : "FAIL",
           rows, cols, tokens, rel, off_grid);
    return !pass;
}

/* kv and q_a (or gate and up) through the pair entry: the bits of the two
 * single projections.  -1 (not taken) is the cuBLAS arm's answer. */
static int pair_case(uint32_t rows_a, uint32_t rows_b, uint32_t cols, uint32_t tokens) {
    const std::vector<uint8_t> ha = make_weights(rows_a, cols, false), hb = make_weights(rows_b, cols, false);
    const uint64_t off_b = (ha.size() + PAGE - 1) / PAGE * PAGE;
    std::vector<uint8_t> map(off_b + hb.size());
    memcpy(map.data(), ha.data(), ha.size());
    memcpy(map.data() + off_b, hb.data(), hb.size());
    uint8_t *m = pages(map), *wa = pages(ha), *wb = pages(hb);
    const std::vector<float> x = bf16_rows(tokens, cols);
    std::vector<float> ya((size_t)tokens * rows_a), yb((size_t)tokens * rows_b), ra, rb;
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(x.size() * 4);
    ds4_gpu_tensor *at = ds4_gpu_tensor_alloc(ya.size() * 4), *bt = ds4_gpu_tensor_alloc(yb.size() * 4);
    int got = xt && at && bt && m && wa && wb && ds4_gpu_tensor_write(xt, 0, x.data(), x.size() * 4) ?
        ds4_gpu_v41_matmul_q4k_pair_tensor(at, 0, rows_a, bt, off_b, rows_b, m, map.size(), cols, xt, tokens) : 0;
    const bool read = got > 0 && ds4_gpu_synchronize() && ds4_gpu_tensor_read(at, 0, ya.data(), ya.size() * 4) &&
        ds4_gpu_tensor_read(bt, 0, yb.data(), yb.size() * 4);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(at); ds4_gpu_tensor_free(bt);
    if (got < 0) { printf("%-24s PASS: not taken (cuBLAS arm)\n", "pair"); return 0; }
    const bool ok = read && project(wa, ha.size(), rows_a, cols, x, tokens, ra) && project(wb, hb.size(), rows_b, cols, x, tokens, rb);
    const bool same = ok && !memcmp(ya.data(), ra.data(), ya.size() * 4) && !memcmp(yb.data(), rb.data(), yb.size() * 4);
    printf("%-24s %s: %u and %u rows of %u, %u tokens, %s\n", "pair", same ? "PASS" : "FAIL", rows_a, rows_b, cols, tokens,
           ok ? (same ? "bit-identical to one by one" : "differs from one by one") : "launch failed");
    return !same;
}

/* wo_a (groups of group_dim -> rank) into wo_b through the output pair:
 * the bits of each group's projection, then wo_b over their rows. */
static int out_case(uint32_t groups, uint32_t group_dim, uint32_t rank, uint32_t out_dim, uint32_t tokens) {
    std::vector<std::vector<uint8_t>> ga;
    std::vector<uint8_t> ha;
    for (uint32_t g = 0; g < groups; g++) {
        ga.push_back(make_weights(rank, group_dim, false));
        ha.insert(ha.end(), ga.back().begin(), ga.back().end());
    }
    const std::vector<uint8_t> hb = make_weights(out_dim, groups * rank, false);
    const uint64_t off_b = (ha.size() + PAGE - 1) / PAGE * PAGE;
    std::vector<uint8_t> map(off_b + hb.size());
    memcpy(map.data(), ha.data(), ha.size());
    memcpy(map.data() + off_b, hb.data(), hb.size());
    uint8_t *m = pages(map), *wb = pages(hb);
    const std::vector<float> heads = bf16_rows(tokens, groups * group_dim);
    std::vector<float> y((size_t)tokens * out_dim);
    ds4_gpu_tensor *ht = ds4_gpu_tensor_alloc(heads.size() * 4), *yt = ds4_gpu_tensor_alloc(y.size() * 4);
    const int got = ht && yt && m && wb && ds4_gpu_tensor_write(ht, 0, heads.data(), heads.size() * 4) ?
        ds4_gpu_v41_out_proj_q4k_tensor(yt, m, map.size(), 0, off_b, groups, group_dim, rank, out_dim, ht, tokens) : 0;
    const bool read = got > 0 && ds4_gpu_synchronize() && ds4_gpu_tensor_read(yt, 0, y.data(), y.size() * 4);
    ds4_gpu_tensor_free(ht); ds4_gpu_tensor_free(yt);
    if (got < 0) { printf("%-24s PASS: not taken (cuBLAS arm)\n", "output pair"); return 0; }

    /* One by one: each group's heads as their own rows, then wo_b. */
    bool ok = read;
    std::vector<float> low((size_t)tokens * groups * rank), part, ref;
    for (uint32_t g = 0; ok && g < groups; g++) {
        std::vector<float> xg((size_t)tokens * group_dim);
        for (uint32_t t = 0; t < tokens; t++) {
            memcpy(&xg[(size_t)t * group_dim], &heads[((size_t)t * groups + g) * group_dim], group_dim * 4);
        }
        uint8_t *w = pages(ga[g]);
        ok = w && project(w, ga[g].size(), rank, group_dim, xg, tokens, part);
        for (uint32_t t = 0; ok && t < tokens; t++) {
            memcpy(&low[((size_t)t * groups + g) * rank], &part[(size_t)t * rank], rank * 4);
        }
    }
    ok = ok && project(wb, hb.size(), out_dim, groups * rank, low, tokens, ref);
    const bool same = ok && !memcmp(y.data(), ref.data(), y.size() * 4);
    printf("%-24s %s: %u groups of %u -> %u, then %u, %u tokens, %s\n", "output pair", same ? "PASS" : "FAIL",
           groups, group_dim, rank, out_dim, tokens, ok ? (same ? "bit-identical to one by one" : "differs from one by one") : "launch failed");
    return !same;
}

int main(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "test_ds41_q4p: ds4_gpu_init failed\n"); return 2; }
    int fail = 0;
    fail |= run_case("attn_kv shape", 512, 5120, 300, 1, false);
    fail |= run_case("ffn_down_shexp shape", 5120, 2304, 256, 1, false);
    fail |= run_case("nine rows", 1280, 5120, 9, 1, false);
    fail |= run_case("rows not 128-aligned", 136, 1280, 200, 1, false);
    fail |= run_case("unrounded (head)", 512, 1280, 129, 0, false);
    fail |= run_case("scales out of range", 256, 1280, 64, 1, true);
    fail |= pair_case(512, 1280, 5120, 130);
    fail |= out_case(4, 512, 256, 512, 100);
    printf("DS41 q4_K prefill projections: %s\n", fail ? "FAIL" : "PASS");
    ds4_gpu_cleanup();
    return fail;
}
