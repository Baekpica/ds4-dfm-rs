/* MiniMax's warp-specialized prefill experts (cuda/minimax_mma8w.cuh)
 * against glmvq_mma8_kernel on one 2,048-row chunk at production shapes:
 * 128 experts x (6144 -> 3072 -> 6144), top-4, 13-bit books with and
 * without the GVR8 residual. Random books, indices and routes; no model.
 * Gate rows, up scales, E4M3 down inputs (un-permuted) and down rows must
 * agree bit for bit; each kernel's time and MMA rate are printed.
 * DS4_MW_ITERS=N repeats the timed launches (default 5); DS4_MW_ONLY=gate|
 * up|down|old runs one kernel once (for ncu). Exit status 0 = every check passed. */
#include "cuda/glm_vq_linear.cuh"
#include "cuda/minimax_mma8w.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

enum { H = 6144, F = 3072, E = 128, USED = 4, ROWS = 2048, OLD_BM = 256, OLD_BT = 64 };
// Distinct weight sets behind the 128 experts (DS4_MW_SETS; 1 keeps every
// expert's index stream in L2, which separates DRAM from kernel limits).
static int SETS = 32;
static constexpr float LIMIT = 7.0f;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static void *dev(uint64_t bytes, const void *host = nullptr) {
    void *p = nullptr;
    CK(cudaMalloc(&p, bytes));
    if (host) { CK(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice)); }
    return p;
}

template <typename T> static std::vector<T> down(const void *p, uint64_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

static uint64_t g_state = 0x9e3779b97f4a7c15ull;
static uint64_t next_u64() {
    g_state ^= g_state << 13; g_state ^= g_state >> 7; g_state ^= g_state << 17;
    return g_state;
}

static void fill_bytes(std::vector<uint8_t> &v) {
    for (size_t i = 0; i < v.size(); i += 8) {
        const uint64_t r = next_u64();
        memcpy(v.data() + i, &r, std::min<size_t>(8, v.size() - i));
    }
}

// Finite E4M3 codewords of moderate size (no NaN bytes 0x7f / 0xff).
static std::vector<uint8_t> make_book(uint32_t words) {
    std::vector<uint8_t> b((size_t)words * GLMVQ_VECTOR);
    for (auto &x : b) {
        uint8_t v = (uint8_t)(next_u64() & 0xffu);
        v = (uint8_t)((v & 0x80u) | (0x20u + (v & 0x3fu)));   // exponents 4..11: |x| in [2^-3, 2^4]
        x = v;
    }
    return b;
}

struct matrix_set { uint8_t *main[3], *plane[3], *res[3]; __half *gain[3]; };

static matrix_set make_set() {
    matrix_set s{};
    for (int p = 0; p < 3; p++) {
        const uint32_t rows = p == 2 ? H : F, cols = p == 2 ? F : H;
        std::vector<uint8_t> main((size_t)rows * cols * 3 / 16), plane((size_t)rows * cols / 64), res((size_t)rows * cols / 8);
        fill_bytes(main); fill_bytes(plane); fill_bytes(res);
        std::vector<__half> gain(rows);
        for (auto &g : gain) { g = __float2half(0.01f + 0.04f * (float)(next_u64() % 1000) / 1000.0f); }
        s.main[p] = (uint8_t *)dev(main.size(), main.data());
        s.plane[p] = (uint8_t *)dev(plane.size(), plane.data());
        s.res[p] = (uint8_t *)dev(res.size(), res.data());
        s.gain[p] = (__half *)dev(gain.size() * sizeof(__half), gain.data());
    }
    return s;
}

static float elapsed(cudaEvent_t a, cudaEvent_t b) { float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }

template <bool RES>
static int run_case(const std::vector<matrix_set> &sets, const uint8_t *book, const uint8_t *res_book,
        const float *x, const uint32_t *grouped_d, const std::vector<uint32_t> &grouped, const std::vector<uint32_t> &offsets,
        uint32_t pairs, int iters, const char *only) {
    // Sources for glmvq_mma8_kernel: [projection][expert].
    std::vector<glmvq_source> src(3 * E);
    for (int p = 0; p < 3; p++) {
        for (int e = 0; e < E; e++) {
            const matrix_set &s = sets[e % SETS];
            const uint32_t rows = p == 2 ? H : F, cols = p == 2 ? F : H;
            src[p * E + e] = {nullptr, nullptr, {s.main[p], s.plane[p], book, s.gain[p], rows, cols, GLMVQ_E4M3,
                RES ? s.res[p] : nullptr, RES ? res_book : nullptr}, rows, cols, GLMVQ_WEIGHT_VQ};
        }
    }
    auto *src_d = (glmvq_source *)dev(src.size() * sizeof(glmvq_source), src.data());
    std::vector<glmvq_mma_item> old_items;
    std::vector<mw_item> ws_items;
    for (uint32_t e = 0; e < E; e++) {
        for (uint32_t s = offsets[e]; s < offsets[e + 1]; s += OLD_BT) {
            old_items.push_back({{e, E + e, 2 * E + e}, s, std::min((uint32_t)OLD_BT, offsets[e + 1] - s)});
        }
    }
    for (int p = 0; p < 3; p++) {
        for (uint32_t e = 0; e < E; e++) {
            const glmvq_matrix &vq = src[p * E + e].vq;
            for (uint32_t s = offsets[e]; s < offsets[e + 1]; s += MW_BT) {
                ws_items.push_back({s, std::min((uint32_t)MW_BT, offsets[e + 1] - s), vq.gain, vq.main, vq.plane, vq.res, 0});
            }
        }
    }
    auto *old_d = (glmvq_mma_item *)dev(old_items.size() * sizeof(glmvq_mma_item), old_items.data());
    auto *ws_d = (mw_item *)dev(ws_items.size() * sizeof(mw_item), ws_items.data());
    const uint32_t nold = (uint32_t)old_items.size(), nws = (uint32_t)ws_items.size() / 3;
    std::vector<uint32_t> token(pairs);
    for (uint32_t i = 0; i < pairs; i++) { token[i] = grouped[i] / USED; }
    auto *token_d = (uint32_t *)dev(pairs * 4, token.data());
    const uint64_t xb = (uint64_t)ROWS * H / GLMVQ_MMA8_K;
    auto *x8o = (uint8_t *)dev((uint64_t)ROWS * H), *x8w = (uint8_t *)dev((uint64_t)ROWS * H);
    auto *xso = (float *)dev(xb * 4), *xsw = (float *)dev(xb * 4);
    auto *gate_o = (__nv_bfloat16 *)dev((uint64_t)pairs * F * 2), *gate_w = (__nv_bfloat16 *)dev((uint64_t)pairs * F * 2);
    auto *mid_o = (uint8_t *)dev((uint64_t)pairs * F), *mid_w = (uint8_t *)dev((uint64_t)pairs * F);
    auto *ms_o = (float *)dev((uint64_t)pairs * F / 128 * 4), *ms_w = (float *)dev((uint64_t)pairs * F / 128 * 4);
    auto *down_o = (__nv_bfloat16 *)dev((uint64_t)pairs * H * 2), *down_w = (__nv_bfloat16 *)dev((uint64_t)pairs * H * 2);
    glmvq_mma8_quant<float><<<(uint32_t)((xb * 32 + 255) / 256), 256>>>(x8o, xso, x, xb);
    mw_quant<<<(uint32_t)((xb * 32 + 255) / 256), 256>>>(x8w, xsw, x, xb);
    CK(cudaGetLastError());
    int blocks[3] = {};
    CK((glmvq_mma8_prepare<OLD_BM, OLD_BT, 1, GLMVQ_MMA_GATE, 13, GLMVQ_CLAMP_OAI, RES>(&blocks[0])));
    CK((glmvq_mma8_prepare<OLD_BM, OLD_BT, 1, GLMVQ_MMA_UP, 13, GLMVQ_CLAMP_OAI, RES>(&blocks[1])));
    CK((glmvq_mma8_prepare<OLD_BM, OLD_BT, 1, GLMVQ_MMA_DOWN, 13, GLMVQ_CLAMP_OAI, RES>(&blocks[2])));
    constexpr uint32_t shared8 = glmvq_mma8_bytes<OLD_BM, OLD_BT, 13, RES>(), threads8 = glmvq_mma8_threads<OLD_BM, 1>();
    int device = 0, sms = 0;
    CK(cudaGetDevice(&device));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    auto old_gate = [&] {
        glmvq_mma8_kernel<OLD_BM, OLD_BT, 1, GLMVQ_MMA_GATE, 13, GLMVQ_CLAMP_OAI, RES><<<
            std::min(nold * (F / OLD_BM), (uint32_t)blocks[0]), threads8, shared8>>>(src_d, old_d, nold, book, x8o, xso,
            grouped_d, USED, gate_o, nullptr, down_o, F, H, LIMIT, nullptr, nullptr, res_book);
    };
    auto old_up = [&] {
        glmvq_mma8_kernel<OLD_BM, OLD_BT, 1, GLMVQ_MMA_UP, 13, GLMVQ_CLAMP_OAI, RES><<<
            std::min(nold * (F / OLD_BM), (uint32_t)blocks[1]), threads8, shared8>>>(src_d, old_d, nold, book, x8o, xso,
            grouped_d, USED, gate_o, nullptr, down_o, F, H, LIMIT, mid_o, ms_o, res_book);
    };
    auto old_down = [&] {
        glmvq_mma8_kernel<OLD_BM, OLD_BT, 1, GLMVQ_MMA_DOWN, 13, GLMVQ_CLAMP_OAI, RES><<<
            std::min(nold * (H / OLD_BM), (uint32_t)blocks[2]), threads8, shared8>>>(src_d, old_d, nold, book, mid_o, ms_o,
            grouped_d, USED, gate_o, nullptr, down_o, H, F, LIMIT, nullptr, nullptr, res_book);
    };
    auto ws_gate = [&] {
        CK((mw_launch<GLMVQ_MMA_GATE, RES>(ws_d, nws, book, res_book, x8w, xsw, token_d, gate_w, nullptr, F, H, LIMIT,
            nullptr, nullptr, (uint32_t)sms, 0)));
    };
    auto ws_up = [&] {
        CK((mw_launch<GLMVQ_MMA_UP, RES>(ws_d + nws, nws, book, res_book, x8w, xsw, token_d, gate_w, nullptr, F, H, LIMIT,
            mid_w, ms_w, (uint32_t)sms, 0)));
    };
    auto ws_down = [&] {
        CK((mw_launch<GLMVQ_MMA_DOWN, RES>(ws_d + 2 * nws, nws, book, res_book, mid_w, ms_w, token_d, nullptr, down_w, H, F,
            LIMIT, nullptr, nullptr, (uint32_t)sms, 0)));
    };
    if (only) {
        // One launch of one kernel (the others run first for valid inputs).
        const bool old = !strcmp(only, "old");
        old_gate(); old_up();
        if (!old) { ws_gate(); ws_up(); }
        CK(cudaDeviceSynchronize());
        if (old) { old_down(); }
        else if (!strcmp(only, "gate")) { ws_gate(); }
        else if (!strcmp(only, "up")) { ws_up(); }
        else { ws_down(); }
        CK(cudaDeviceSynchronize());
        return 0;
    }
    old_gate(); old_up(); old_down();
    ws_gate(); ws_up(); ws_down();
    CK(cudaDeviceSynchronize());
    int fail = 0;
    auto same16 = [&](const char *what, const void *a, const void *b, uint64_t n) {
        const auto x = down<uint16_t>(a, n), y = down<uint16_t>(b, n);
        uint64_t d = 0;
        for (uint64_t i = 0; i < n; i++) { d += x[i] != y[i]; }
        printf("  %-34s %s (%llu of %llu differ)\n", what, d ? "FAIL" : "ok", (unsigned long long)d, (unsigned long long)n);
        fail |= d != 0;
    };
    same16("gate rows (BF16)", gate_o, gate_w, (uint64_t)pairs * F);
    {
        const auto a = down<float>(ms_o, (uint64_t)pairs * F / 128), b = down<float>(ms_w, (uint64_t)pairs * F / 128);
        uint64_t d = 0;
        for (size_t i = 0; i < a.size(); i++) { d += memcmp(&a[i], &b[i], 4) != 0; }
        printf("  %-34s %s (%llu differ)\n", "down input scales", d ? "FAIL" : "ok", (unsigned long long)d);
        fail |= d != 0;
        const auto x = down<uint8_t>(mid_o, (uint64_t)pairs * F), y = down<uint8_t>(mid_w, (uint64_t)pairs * F);
        d = 0;
        for (uint64_t p = 0; p < pairs; p++) {
            for (uint32_t k = 0; k < F; k++) { d += x[p * F + k] != y[p * F + beam_w_perm(k)]; }
        }
        printf("  %-34s %s (%llu differ)\n", "down input rows (E4M3, k order)", d ? "FAIL" : "ok", (unsigned long long)d);
        fail |= d != 0;
    }
    same16("down rows (BF16)", down_o, down_w, (uint64_t)pairs * H);
    // MMA work: 2 * pairs * rows * cols per projection, twice on residual layers.
    const double flops = 2.0 * pairs * (double)F * H * (RES ? 2 : 1);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto timed = [&](const char *what, auto fn) {
        fn();
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++) { fn(); }
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        const float ms = elapsed(a, b) / iters;
        printf("  %-12s %8.3f ms  %6.1f TFLOPS (MMA)\n", what, ms, flops / (ms * 1e-3) / 1e12);
        return ms;
    };
    const float og = timed("old gate", old_gate), ou = timed("old up", old_up), od = timed("old down", old_down);
    const float wg = timed("ws gate", ws_gate), wu = timed("ws up", ws_up), wd = timed("ws down", ws_down);
    printf("  sum old %.3f ms, ws %.3f ms (%.2fx)\n", og + ou + od, wg + wu + wd, (og + ou + od) / (wg + wu + wd));
    return fail;
}

int main() {
    const int iters = getenv("DS4_MW_ITERS") ? atoi(getenv("DS4_MW_ITERS")) : 5;
    if (getenv("DS4_MW_SETS")) { SETS = atoi(getenv("DS4_MW_SETS")); }
    const char *only = getenv("DS4_MW_ONLY");
    const auto book_h = make_book(8192), res_h = make_book(GLMVQ_RES_WORDS);
    const auto *book = (const uint8_t *)dev(book_h.size(), book_h.data());
    const auto *res_book = (const uint8_t *)dev(res_h.size(), res_h.data());
    std::vector<matrix_set> sets;
    for (int i = 0; i < SETS; i++) { sets.push_back(make_set()); }
    // Activations: normal rows with a few large channels per token.
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t)ROWS * H);
    for (auto &v : x) { v = nd(rng); }
    for (uint32_t t = 0; t < ROWS; t++) {
        for (int k = 0; k < 4; k++) { x[(size_t)t * H + (k * 1543 + t * 7) % H] *= 40.0f; }
    }
    auto *x_d = (float *)dev(x.size() * 4, x.data());
    // Routes: four distinct experts per token, grouped by expert (ascending).
    const uint32_t pairs = ROWS * USED;
    std::vector<uint32_t> route(pairs), offsets(E + 1, 0), grouped(pairs);
    std::uniform_int_distribution<uint32_t> pick(0, E - 1);
    for (uint32_t t = 0; t < ROWS; t++) {
        for (uint32_t k = 0; k < USED; k++) {
            uint32_t e;
            do { e = pick(rng); } while (std::find(&route[t * USED], &route[t * USED] + k, e) != &route[t * USED] + k);
            route[t * USED + k] = e;
            offsets[e + 1]++;
        }
    }
    for (uint32_t e = 0; e < E; e++) { offsets[e + 1] += offsets[e]; }
    std::vector<uint32_t> cursor(offsets.begin(), offsets.end() - 1);
    for (uint32_t i = 0; i < pairs; i++) { grouped[cursor[route[i]]++] = i; }
    auto *grouped_d = (uint32_t *)dev(pairs * 4, grouped.data());
    int fail = 0;
    printf("R13 (no residual):\n");
    fail |= run_case<false>(sets, book, res_book, x_d, grouped_d, grouped, offsets, pairs, iters, only);
    printf("R13+8 (residual):\n");
    fail |= run_case<true>(sets, book, res_book, x_d, grouped_d, grouped, offsets, pairs, iters, only);
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail;
}
