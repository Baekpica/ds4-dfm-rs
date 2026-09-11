// Canonical activation bytes and fixed-reduction MoE batch parity.
#include "ds4_mmq.h"
#include "ds4_repack.h"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "../../../ds4_gpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

enum { HIDDEN = 4096, MIDDLE = 2048, USED = 6, SHARED = 2,
       MAX_EXPERTS = 256, OFFSET = 4096, REPEATS = 4 };
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); exit(1); \
} } while (0)
#define CUDA(expr) CHECK((expr) == cudaSuccess)

using Vec = int (*)(const void *, const float *, const int32_t *, float *,
                    int, int, int, int, int, cudaStream_t);
static Vec oracle(ggml_type type) {
    switch (type) {
    case GGML_TYPE_Q8_0: return ds4_mmq_q8_0_moe_vec;
    case GGML_TYPE_Q3_K: return ds4_mmq_q3_K_moe_vec;
    case GGML_TYPE_Q4_K: return ds4_mmq_q4_K_moe_vec;
    case GGML_TYPE_IQ2_XXS: return ds4_mmq_iq2_xxs_moe_vec;
    case GGML_TYPE_IQ2_XS: return ds4_mmq_iq2_xs_moe_vec;
    default: CHECK(false); return nullptr;
    }
}

static uint32_t random_bits() {
    static uint32_t state = 976;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

static std::vector<unsigned char> weights(ggml_type type, int m, int k, int ne) {
    std::vector<unsigned char> data(ds4_mmq_inkling_wbytes(type, m, k, ne));
    CHECK(!data.empty());
    for (auto &byte : data) { byte = random_bits(); }
    const size_t blocks = data.size() / ggml_type_size(type);
    for (size_t b = 0; b < blocks; b++) {
        const __half scale = __float2half(b % 100003 == 3
            ? std::numeric_limits<float>::infinity() : 0.003f + (b % 31) * 0.0007f);
        switch (type) {
        case GGML_TYPE_Q8_0: ((block_q8_0 *)data.data())[b].d = scale; break;
        case GGML_TYPE_Q3_K: ((block_q3_K *)data.data())[b].d = scale; break;
        case GGML_TYPE_Q4_K:
            ((block_q4_K *)data.data())[b].data.d = scale;
            ((block_q4_K *)data.data())[b].data.dmin = scale;
            break;
        case GGML_TYPE_IQ2_XXS: ((block_iq2_xxs *)data.data())[b].d = scale; break;
        case GGML_TYPE_IQ2_XS: ((block_iq2_xs *)data.data())[b].d = scale; break;
        default: CHECK(false);
        }
    }
    return data;
}

static void *device_copy(const void *data, size_t bytes) {
    void *ptr = nullptr; CUDA(cudaMalloc(&ptr, bytes));
    if (data) { CUDA(cudaMemcpy(ptr, data, bytes, cudaMemcpyHostToDevice)); }
    return ptr;
}

static void exact(const void *a, const void *b, size_t bytes) {
    std::vector<unsigned char> x(bytes), y(bytes);
    CUDA(cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost));
    if (memcmp(x.data(), y.data(), bytes) == 0) { return; }
    for (size_t i = 0; i < bytes / sizeof(float); i++) {
        float xa, ya;
        memcpy(&xa, x.data() + i * sizeof(float), sizeof(float));
        memcpy(&ya, y.data() + i * sizeof(float), sizeof(float));
        if (memcmp(&xa, &ya, sizeof(float))) {
            fprintf(stderr, "first difference [%zu]: %.9g / %.9g\n", i, xa, ya);
            break;
        }
    }
    CHECK(false);
}

enum Routes { SPREAD, REPEATED, INVALID, RANDOM };
static void batch_case(ggml_type type, int m, int tokens, int ne, int used,
                       Routes routing) {
    const int group = used > 1 ? 1 : ne == SHARED ? SHARED : USED;
    const int rows = tokens * group, assignments = rows * used;
    const int k = used > 1 ? HIDDEN : MIDDLE;
    auto w = weights(type, m, k, ne);
    std::vector<float> x((size_t)rows * k);
    std::vector<int32_t> ids(assignments);
    for (auto &v : x) { v = ((int)(random_bits() % 2001) - 1000) / 417.0f; }
    for (int a = 0; a < assignments; a++) {
        ids[a] = routing == REPEATED ? ne - 1 :
            routing == RANDOM ? random_bits() % ne : (a * 13 + a / used) % ne;
        if (routing == INVALID && a % 3 == 0) { ids[a] = -1; }
    }
    void *dw = device_copy(w.data(), w.size());
    auto *dx = (float *)device_copy(x.data(), x.size() * sizeof(float));
    auto *di = (int32_t *)device_copy(ids.data(), ids.size() * sizeof(int32_t));
    const size_t out_bytes = (size_t)assignments * m * sizeof(float);
    auto *got = (float *)device_copy(nullptr, out_bytes);
    auto *want = (float *)device_copy(nullptr, out_bytes);
    const size_t q8row = k / QK8_1 * sizeof(block_q8_1), q8bytes = rows * q8row;
    auto *q8 = (char *)device_copy(nullptr, q8bytes);
    auto *q8rows = (char *)device_copy(nullptr, q8bytes);
    quantize_row_q8_1_cuda(dx, nullptr, q8, type, k, k, k, (int64_t)k * rows,
                          k, 1, rows, 1, nullptr);
    for (int row = 0; row < rows; row++) {
        quantize_row_q8_1_cuda(dx + (size_t)row * k, nullptr, q8rows + row * q8row,
                              type, k, k, k, k, k, 1, 1, 1, nullptr);
    }
    exact(q8, q8rows, q8bytes);
    const Vec vec = oracle(type);
    auto reference = [&]() {
        for (int row = 0; row < rows; row += group) {
            CHECK(vec(dw, dx + (size_t)row * k, di + row * used,
                      want + (size_t)row * used * m,
                      m, k, group, ne, used, nullptr) == 0);
        }
    };
    reference();
    CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
    exact(got, want, out_bytes);
    // The warp-tile kill switch restores the four-column kernel exactly.
    CUDA(cudaMemset(got, 0xff, out_bytes));
    CHECK(setenv("DS4_INKLING_NO_MOE_TILE", "1", 1) == 0);
    CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
    CHECK(unsetenv("DS4_INKLING_NO_MOE_TILE") == 0);
    exact(got, want, out_bytes);

    if (type == GGML_TYPE_Q8_0 && ne == SHARED && used == SHARED) {
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(setenv("DS4_INKLING_NO_SHARED_Q8", "1", 1) == 0);
        CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
        CHECK(unsetenv("DS4_INKLING_NO_SHARED_Q8") == 0);
        exact(got, want, out_bytes);
    }

    if (type == GGML_TYPE_Q8_0 && ne == SHARED && used == SHARED) {
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(setenv("DS4_INKLING_NO_SHARED_TILE", "1", 1) == 0);
        CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
        CHECK(unsetenv("DS4_INKLING_NO_SHARED_TILE") == 0);
        exact(got, want, out_bytes);
    }

    if (type == GGML_TYPE_Q8_0 && ne == SHARED && used == 1) {
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(setenv("DS4_INKLING_NO_SHARED_DOWN_TILE", "1", 1) == 0);
        CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
        CHECK(unsetenv("DS4_INKLING_NO_SHARED_DOWN_TILE") == 0);
        exact(got, want, out_bytes);
    }

    // The SoA slab switch restores canonical-row slabs on every Q8 tile path.
    if (type == GGML_TYPE_Q8_0) {
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(setenv("DS4_INKLING_NO_SHARED_SOA", "1", 1) == 0);
        CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
        CHECK(unsetenv("DS4_INKLING_NO_SHARED_SOA") == 0);
        exact(got, want, out_bytes);
    }

    if (type == GGML_TYPE_Q8_0 && ne != SHARED) {
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(setenv("DS4_INKLING_NO_Q8_ROUTED_TILE", "1", 1) == 0);
        CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
        CHECK(unsetenv("DS4_INKLING_NO_Q8_ROUTED_TILE") == 0);
        exact(got, want, out_bytes);
    }

    // Branched columns and table signs must reproduce the lean IQ2 tiles.
    if (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS) {
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(setenv("DS4_INKLING_NO_IQ2_LEAN", "1", 1) == 0);
        CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
        CHECK(unsetenv("DS4_INKLING_NO_IQ2_LEAN") == 0);
        exact(got, want, out_bytes);
    }

    if (type == GGML_TYPE_Q3_K) {
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(setenv("DS4_INKLING_NO_Q3_TILE", "1", 1) == 0);
        CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
        CHECK(unsetenv("DS4_INKLING_NO_Q3_TILE") == 0);
        exact(got, want, out_bytes);
    }

    if (type == GGML_TYPE_Q4_K) {
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(setenv("DS4_INKLING_NO_Q4_TILE", "1", 1) == 0);
        CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
        CHECK(unsetenv("DS4_INKLING_NO_Q4_TILE") == 0);
        exact(got, want, out_bytes);
    }

    const size_t work_bytes = ds4_mmvq_inkling_bytes(rows, ne, used);
    void *work = device_copy(nullptr, work_bytes);
    if ((type == GGML_TYPE_Q8_0 && ne == SHARED && tokens == 65) ||
        (type == GGML_TYPE_Q8_0 && ne != SHARED && tokens == 65) ||
        (type == GGML_TYPE_Q4_K && tokens == 513)) {
        cudaStream_t stream;
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA(cudaMemsetAsync(got, 0xff, out_bytes, stream));
        CHECK(ds4_mmvq_inkling(dw, type, q8, di, got, work, work_bytes,
                              m, k, rows, ne, used, stream) == 0);
        CUDA(cudaStreamSynchronize(stream));
        exact(got, want, out_bytes);
        CUDA(cudaStreamDestroy(stream));

        // Exercise canonical four-byte alignment too. Shared-Q8 vector
        // staging falls back; Q4_K keeps scalar-aligned loads.
        auto *unaligned = (char *)device_copy(nullptr, q8bytes + sizeof(float));
        CUDA(cudaMemcpy(unaligned + sizeof(float), q8, q8bytes, cudaMemcpyDeviceToDevice));
        CUDA(cudaMemset(got, 0xff, out_bytes));
        CHECK(ds4_mmvq_inkling(dw, type, unaligned + sizeof(float), di, got, work,
                              work_bytes, m, k, rows, ne, used, nullptr) == 0);
        exact(got, want, out_bytes);
        CUDA(cudaFree(unaligned));
    }
    CHECK(ds4_mmvq_inkling(dw, type, q8, di, got, work, work_bytes - 1,
                          m, k, rows, ne, used, nullptr) == -1);
    exact(got, want, out_bytes);
    // A malformed high id has no valid legacy read; require a clean zero.
    CUDA(cudaMemset(di, 0x7f, ids.size() * sizeof(int32_t)));
    CHECK(ds4_mmvq_inkling(dw, type, q8, di, got, work, work_bytes,
                          m, k, rows, ne, used, nullptr) == 0);
    CUDA(cudaMemset(q8rows, 0, q8bytes));
    std::vector<float> zero(assignments * m);
    CUDA(cudaMemcpy(zero.data(), got, out_bytes, cudaMemcpyDeviceToHost));
    for (float v : zero) { CHECK(v == 0); }
    CUDA(cudaMemcpy(di, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    float elapsed[3] = {};
    if (m == HIDDEN && tokens >= 64) {
        cudaEvent_t start, stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
        // Modes: per-token legacy, four-column batch (kill switch), warp tiles.
        for (int mode = 0; mode < 3; mode++) {
            if (mode == 1) { CHECK(setenv("DS4_INKLING_NO_MOE_TILE", "1", 1) == 0); }
            else { CHECK(unsetenv("DS4_INKLING_NO_MOE_TILE") == 0); }
            CUDA(cudaEventRecord(start));
            for (int repeat = 0; repeat < REPEATS; repeat++) {
                if (mode == 0) { reference(); }
                else { CHECK(ds4_mmq_inkling_moe(dw, type, dx, di, got,
                                    m, k, rows, ne, used, nullptr) == 0); }
            }
            CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
            CUDA(cudaEventElapsedTime(elapsed + mode, start, stop));
        }
        CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
        exact(got, want, out_bytes);
    }
    printf("type=%d M=%d tokens=%d experts=%d used=%d routes=%d exact; %.3f -> %.3f -> %.3f ms\n",
           type, m, tokens, ne, used, routing, elapsed[0] / REPEATS, elapsed[1] / REPEATS,
           elapsed[2] / REPEATS);
    CUDA(cudaFree(work)); CUDA(cudaFree(q8)); CUDA(cudaFree(q8rows));
    CUDA(cudaFree(dw)); CUDA(cudaFree(dx)); CUDA(cudaFree(di));
    CUDA(cudaFree(got)); CUDA(cudaFree(want));
}

static std::vector<unsigned char> pack_iq2_aligned(
        const std::vector<unsigned char> &raw, int m, int k, int ne) {
    const uint64_t nblk = (uint64_t)ne * m * (k / 256);
    CHECK(raw.size() == nblk * 66u);
    const uint64_t bytes = ds4_mmq_iq2_xxs_aligned_bytes(m, k, ne);
    CHECK(bytes != 0);
    std::vector<unsigned char> art(bytes);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    for (uint64_t b = 0; b < nblk; b++) {
        memcpy(art.data() + b * 2, raw.data() + b * 66, 2);
        memcpy(art.data() + dq_bytes + b * 64, raw.data() + b * 66 + 2, 64);
    }
    return art;
}

static void candidate_case() {
    ds4_repack_tensor t;
    t.type = 16;
    t.ndim = 3;
    t.dims[0] = 4096;
    t.dims[1] = 4096;
    t.dims[2] = 256;
    t.bytes = 66ull * 16ull * 4096ull * 256ull;
    t.name = "model.llm.layers.3.mlp.experts.w13_weight";
    CHECK(ds4_repack_iq2_candidate(t));
    t.name = "blk.7.ffn_gate_exps.weight";
    t.dims[0] = 2048;
    t.dims[1] = 1024;
    t.bytes = 66ull * 1000ull;
    CHECK(ds4_repack_iq2_candidate(t));
    t.name = "model.llm.layers.3.mlp.experts.w2_weight";
    CHECK(!ds4_repack_iq2_candidate(t));
    t.name = "model.llm.layers.3.mlp.shared_experts.shared_w13_weight";
    t.dims[0] = 4096;
    t.dims[1] = 4096;
    t.dims[2] = 2;
    t.bytes = 66ull * 16ull * 4096ull * 2ull;
    CHECK(!ds4_repack_iq2_candidate(t));
    t.name = "model.llm.layers.3.mlp.experts.w13_weight";
    t.dims[0] = 1000;
    t.bytes = 66ull * 1000ull;
    CHECK(!ds4_repack_iq2_candidate(t));
    t.type = 17;
    t.dims[0] = 2048;
    t.dims[1] = 4096;
    t.dims[2] = 256;
    t.bytes = 74ull * 8ull * 4096ull * 256ull;
    t.name = "model.llm.layers.3.mlp.experts.w2_weight";
    CHECK(ds4_repack_iq2_xs_candidate(t));
    t.name = "model.llm.layers.3.mlp.experts.w13_weight";
    CHECK(!ds4_repack_iq2_xs_candidate(t));
}

static std::vector<unsigned char> pack_iq2_xs_aligned(
        const std::vector<unsigned char> &raw, int m, int k, int ne) {
    const uint64_t nblk = (uint64_t)ne * m * (k / 256);
    CHECK(raw.size() == nblk * 74u);
    const uint64_t bytes = ds4_mmq_iq2_xs_aligned_bytes(m, k, ne);
    CHECK(bytes != 0);
    std::vector<unsigned char> art(bytes);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const uint64_t sc_bytes = (nblk * 8u + 63u) & ~63ull;
    for (uint64_t b = 0; b < nblk; b++) {
        memcpy(art.data() + b * 2, raw.data() + b * 74, 2);
        memcpy(art.data() + dq_bytes + b * 8, raw.data() + b * 74 + 66, 8);
        memcpy(art.data() + dq_bytes + sc_bytes + b * 64, raw.data() + b * 74 + 2, 64);
    }
    return art;
}

static void aligned_xs_case(int m, int tokens, int ne, Routes routing) {
    const int used = 1, rows = tokens, assignments = rows * used, k = MIDDLE;
    auto w = weights(GGML_TYPE_IQ2_XS, m, k, ne);
    auto art = pack_iq2_xs_aligned(w, m, k, ne);
    std::vector<float> x((size_t)rows * k);
    std::vector<int32_t> ids(assignments);
    for (auto &v : x) { v = ((int)(random_bits() % 2001) - 1000) / 417.0f; }
    for (int a = 0; a < assignments; a++) {
        ids[a] = routing == REPEATED ? ne - 1 :
            routing == RANDOM ? random_bits() % ne : (a * 13 + a / used) % ne;
        if (routing == INVALID && a % 3 == 0) { ids[a] = -1; }
    }
    void *dw = device_copy(w.data(), w.size());
    void *da = device_copy(art.data(), art.size());
    auto *dx = (float *)device_copy(x.data(), x.size() * sizeof(float));
    auto *di = (int32_t *)device_copy(ids.data(), ids.size() * sizeof(int32_t));
    const size_t out_bytes = (size_t)assignments * m * sizeof(float);
    auto *got = (float *)device_copy(nullptr, out_bytes);
    auto *want = (float *)device_copy(nullptr, out_bytes);
    CHECK(ds4_mmq_inkling_moe(dw, GGML_TYPE_IQ2_XS, dx, di, want, m, k, rows, ne, used, nullptr) == 0);
    CHECK(ds4_mmq_inkling_moe_iq2_xs_aligned(da, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
    exact(got, want, out_bytes);
    CUDA(cudaMemset(got, 0xff, out_bytes));
    CHECK(setenv("DS4_INKLING_NO_IQ2_LEAN", "1", 1) == 0);
    CHECK(ds4_mmq_inkling_moe_iq2_xs_aligned(da, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
    CHECK(unsetenv("DS4_INKLING_NO_IQ2_LEAN") == 0);
    exact(got, want, out_bytes);
    printf("aligned-xs M=%d tokens=%d experts=%d routes=%d exact\n", m, tokens, ne, routing);
    CUDA(cudaFree(dw)); CUDA(cudaFree(da)); CUDA(cudaFree(dx)); CUDA(cudaFree(di));
    CUDA(cudaFree(got)); CUDA(cudaFree(want));
}

static void aligned_case(int m, int tokens, int ne, Routes routing) {
    const int used = USED, rows = tokens, assignments = rows * used, k = HIDDEN;
    auto w = weights(GGML_TYPE_IQ2_XXS, m, k, ne);
    auto art = pack_iq2_aligned(w, m, k, ne);
    std::vector<float> x((size_t)rows * k);
    std::vector<int32_t> ids(assignments);
    for (auto &v : x) { v = ((int)(random_bits() % 2001) - 1000) / 417.0f; }
    for (int a = 0; a < assignments; a++) {
        ids[a] = routing == REPEATED ? ne - 1 :
            routing == RANDOM ? random_bits() % ne : (a * 13 + a / used) % ne;
        if (routing == INVALID && a % 3 == 0) { ids[a] = -1; }
    }
    void *dw = device_copy(w.data(), w.size());
    void *da = device_copy(art.data(), art.size());
    auto *dx = (float *)device_copy(x.data(), x.size() * sizeof(float));
    auto *di = (int32_t *)device_copy(ids.data(), ids.size() * sizeof(int32_t));
    const size_t out_bytes = (size_t)assignments * m * sizeof(float);
    auto *got = (float *)device_copy(nullptr, out_bytes);
    auto *want = (float *)device_copy(nullptr, out_bytes);
    CHECK(ds4_mmq_inkling_moe(dw, GGML_TYPE_IQ2_XXS, dx, di, want, m, k, rows, ne, used, nullptr) == 0);
    CHECK(ds4_mmq_inkling_moe_iq2_aligned(da, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
    exact(got, want, out_bytes);
    CUDA(cudaMemset(got, 0xff, out_bytes));
    CHECK(setenv("DS4_INKLING_NO_IQ2_LEAN", "1", 1) == 0);
    CHECK(ds4_mmq_inkling_moe_iq2_aligned(da, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
    CHECK(unsetenv("DS4_INKLING_NO_IQ2_LEAN") == 0);
    exact(got, want, out_bytes);
    CUDA(cudaMemset(got, 0xff, out_bytes));
    CHECK(setenv("DS4_INKLING_NO_IQ2_ALIGNED", "1", 1) == 0);
    CHECK(ds4_mmq_inkling_moe_iq2_aligned(da, dx, di, got, m, k, rows, ne, used, nullptr) == 0);
    CHECK(unsetenv("DS4_INKLING_NO_IQ2_ALIGNED") == 0);
    exact(got, want, out_bytes);
    const size_t work_bytes = ds4_mmvq_inkling_bytes(rows, ne, used);
    void *work = device_copy(nullptr, work_bytes);
    const size_t q8row = k / QK8_1 * sizeof(block_q8_1), q8bytes = rows * q8row;
    auto *q8 = (char *)device_copy(nullptr, q8bytes);
    quantize_row_q8_1_cuda(dx, nullptr, q8, GGML_TYPE_IQ2_XXS, k, k, k,
                          (int64_t)k * rows, k, 1, rows, 1, nullptr);
    CUDA(cudaMemset(got, 0xff, out_bytes));
    CHECK(ds4_mmvq_inkling_iq2_aligned(da, q8, di, got, work, work_bytes,
                                      m, k, rows, ne, used, nullptr) == 0);
    exact(got, want, out_bytes);
    printf("aligned M=%d tokens=%d experts=%d routes=%d exact\n", m, tokens, ne, routing);
    CUDA(cudaFree(work)); CUDA(cudaFree(q8));
    CUDA(cudaFree(dw)); CUDA(cudaFree(da)); CUDA(cudaFree(dx)); CUDA(cudaFree(di));
    CUDA(cudaFree(got)); CUDA(cudaFree(want));
}

static ds4_gpu_tensor *tensor(const void *data, size_t bytes) {
    auto *t = ds4_gpu_tensor_alloc(bytes); CHECK(t);
    if (data) { CHECK(ds4_gpu_tensor_write(t, 0, data, bytes)); }
    return t;
}

static void wrapper_case() {
    auto w = weights(GGML_TYPE_Q8_0, HIDDEN, HIDDEN, SHARED);
    const size_t map_bytes = OFFSET + w.size();
    void *map = nullptr; CHECK(posix_memalign(&map, OFFSET, map_bytes) == 0);
    memcpy((char *)map + OFFSET, w.data(), w.size());
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    enum { ROWS = 3 };
    std::vector<float> x(ROWS * HIDDEN, 0.5f), want(ROWS * SHARED * HIDDEN), got(want.size());
    const int32_t ids[ROWS * SHARED] = {1, 0, 1, -1, 0, 1};
    auto *dx = tensor(x.data(), x.size() * sizeof(float));
    auto *di = tensor(ids, sizeof(ids));
    auto *out = tensor(nullptr, got.size() * sizeof(float));
    for (unsigned row = 0; row < ROWS; row++) {
        auto *in = ds4_gpu_tensor_view(dx, row * HIDDEN * sizeof(float), HIDDEN * sizeof(float));
        auto *route = ds4_gpu_tensor_view(di, row * SHARED * sizeof(int32_t), SHARED * sizeof(int32_t));
        auto *dest = ds4_gpu_tensor_view(out, row * SHARED * HIDDEN * sizeof(float), SHARED * HIDDEN * sizeof(float));
        CHECK(ds4_gpu_routed_matmul_tensor(dest, in, route, map, map_bytes,
                OFFSET, w.size(), GGML_TYPE_Q8_0, HIDDEN, HIDDEN, SHARED, 1, SHARED));
        ds4_gpu_tensor_free(in); ds4_gpu_tensor_free(route); ds4_gpu_tensor_free(dest);
    }
    CHECK(ds4_gpu_tensor_read(out, 0, want.data(), want.size() * sizeof(float)));
    auto call = [&](ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t bytes,
                    uint32_t rows, uint32_t type) {
        return ds4_gpu_inkling_routed(a, b, di, map, map_bytes, OFFSET,
            bytes, type, HIDDEN, HIDDEN, SHARED, rows, SHARED);
    };
    CHECK(call(out, dx, w.size(), ROWS, GGML_TYPE_Q8_0) == 1);
    CHECK(ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)));
    CHECK(memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
    CHECK(call(out, dx, w.size(), 1, GGML_TYPE_Q8_0) == 0);
    CHECK(call(out, dx, w.size(), ROWS, GGML_TYPE_F16) == 0);
    CHECK(call(out, dx, w.size() - 1, ROWS, GGML_TYPE_Q8_0) == -1);
    CHECK(call(out, dx, w.size(), ROWS + 1, GGML_TYPE_Q8_0) == -1);
    CHECK(call(out, out, w.size(), ROWS, GGML_TYPE_Q8_0) == -1);
    CHECK(setenv("DS4_INKLING_NO_MOE_BATCH", "1", 1) == 0);
    CHECK(call(out, dx, w.size(), ROWS, GGML_TYPE_Q8_0) == 0);
    CHECK(unsetenv("DS4_INKLING_NO_MOE_BATCH") == 0);
    CHECK(ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)));
    CHECK(memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(out);
    CHECK(ds4_gpu_synchronize()); ds4_gpu_cleanup(); free(map);
    puts("native wrapper parity/bounds/alias/kill checks passed");
}

int main() {
    CHECK(unsetenv("DS4_INKLING_NO_MOE_BATCH") == 0);
    CHECK(unsetenv("DS4_INKLING_NO_MOE_TILE") == 0);
    CHECK(unsetenv("DS4_INKLING_NO_IQ2_ALIGNED") == 0);
    CHECK(unsetenv("DS4_INKLING_NO_IQ2_XS_ALIGNED") == 0);
    CHECK(unsetenv("DS4_INKLING_NO_SHARED_SOA") == 0);
    CHECK(unsetenv("DS4_INKLING_NO_IQ2_LEAN") == 0);
    CHECK(unsetenv("DS4_INKLING_NO_Q3_TILE") == 0);
    CHECK(ds4_gpu_init()); CHECK(ds4_mmq_init(0) == 0);
    candidate_case();
    CHECK(ds4_mmvq_inkling_bytes(8192 * USED + 1, MAX_EXPERTS, 1) == 0);
    CHECK(ds4_mmvq_inkling_bytes(8193, MAX_EXPERTS, USED) == 0);
    const ggml_type types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
                               GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS};
    for (auto type : types) {
        for (int rows : {1, 3, 4, 5, 64, 65}) {
            batch_case(type, 128, rows, MAX_EXPERTS, USED, SPREAD);
            batch_case(type, 128, rows, MAX_EXPERTS, 1, REPEATED);
        }
        batch_case(type, 128, 9, MAX_EXPERTS, USED, INVALID);
        batch_case(type, 128, 9, MAX_EXPERTS, 1, INVALID);
        batch_case(type, HIDDEN, 64, 17, USED, SPREAD);
        batch_case(type, HIDDEN, 64, 17, 1, SPREAD);
        batch_case(type, HIDDEN, 9, SHARED, SHARED, REPEATED);
        batch_case(type, HIDDEN, 9, SHARED, 1, REPEATED);
    }
    for (int tokens : {8191, 8192}) {
        batch_case(GGML_TYPE_IQ2_XXS, 2, tokens, MAX_EXPERTS, USED, REPEATED);
        batch_case(GGML_TYPE_IQ2_XS, 2, tokens, MAX_EXPERTS, 1, SPREAD);
    }
    // Production routed/shared geometry at the default and wider prefill chunks.
    for (int tokens : {64, 65, 256, 512}) {
        batch_case(GGML_TYPE_IQ2_XXS, HIDDEN, tokens, MAX_EXPERTS, USED, RANDOM);
        batch_case(GGML_TYPE_IQ2_XS, HIDDEN, tokens, MAX_EXPERTS, 1, RANDOM);
    }
    batch_case(GGML_TYPE_IQ2_XXS, HIDDEN, 64, MAX_EXPERTS, USED, INVALID);
    aligned_case(128, 64, MAX_EXPERTS, SPREAD);
    aligned_case(HIDDEN, 1, MAX_EXPERTS, RANDOM);
    aligned_case(HIDDEN, 64, MAX_EXPERTS, RANDOM);
    aligned_case(HIDDEN, 512, MAX_EXPERTS, RANDOM);
    aligned_case(HIDDEN, 64, MAX_EXPERTS, INVALID);
    aligned_xs_case(HIDDEN, 1, MAX_EXPERTS, RANDOM);
    aligned_xs_case(HIDDEN, 64, MAX_EXPERTS, RANDOM);
    aligned_xs_case(HIDDEN, 512, MAX_EXPERTS, SPREAD);
    batch_case(GGML_TYPE_IQ2_XS, HIDDEN, 64, MAX_EXPERTS, 1, INVALID);
    for (int tokens : {64, 512}) {
        batch_case(GGML_TYPE_Q8_0, HIDDEN, tokens, SHARED, SHARED, SPREAD);
        batch_case(GGML_TYPE_Q8_0, HIDDEN, tokens, SHARED, 1, SPREAD);
        batch_case(GGML_TYPE_Q8_0, HIDDEN, tokens, MAX_EXPERTS, USED, RANDOM);
    }
    for (int tokens : {15, 16, 17}) {
        for (Routes route : {SPREAD, REPEATED, INVALID}) {
            batch_case(GGML_TYPE_Q8_0, 128, tokens, SHARED, SHARED, route);
        }
    }
    // Persistent shared-expert tiles: ragged rows/columns, repeated and invalid
    // routes, maximum worklists, and explicit nonblocking stream execution.
    for (int tokens : {63, 64, 65, 8192}) {
        for (Routes route : {REPEATED, INVALID}) {
            batch_case(GGML_TYPE_Q8_0, 126, tokens, SHARED, SHARED, route);
            batch_case(GGML_TYPE_Q8_0, 126, tokens, SHARED, 1, route);
        }
    }
    // Decode-once Q3_K up tiles: threshold edges, ragged 124/126 rows (126
    // falls back to four columns), full width, random/repeated/invalid routes.
    for (int tokens : {255, 256, 257, 8192}) {
        batch_case(GGML_TYPE_Q3_K, 124, tokens, MAX_EXPERTS, USED, RANDOM);
        batch_case(GGML_TYPE_Q3_K, 124, tokens, MAX_EXPERTS, USED, INVALID);
    }
    batch_case(GGML_TYPE_Q3_K, 126, 512, MAX_EXPERTS, USED, RANDOM);
    batch_case(GGML_TYPE_Q3_K, 124, 257, MAX_EXPERTS, USED, REPEATED);
    batch_case(GGML_TYPE_Q3_K, HIDDEN, 512, MAX_EXPERTS, USED, RANDOM);
    for (int used : {int(USED), 1}) {
        for (int tokens : {511, 512, 513, 8192}) {
            batch_case(GGML_TYPE_Q4_K, 126, tokens, MAX_EXPERTS, used, RANDOM);
            batch_case(GGML_TYPE_Q4_K, 126, tokens, MAX_EXPERTS, used, INVALID);
        }
        batch_case(GGML_TYPE_Q4_K, 126, 513, MAX_EXPERTS, used, REPEATED);
        batch_case(GGML_TYPE_Q4_K, HIDDEN, 512, MAX_EXPERTS, used, RANDOM);
    }
    for (int used : {int(USED), 1}) {
        for (int tokens : {63, 64, 65, 8192}) {
            batch_case(GGML_TYPE_Q8_0, 126, tokens, MAX_EXPERTS, used, RANDOM);
            batch_case(GGML_TYPE_Q8_0, 126, tokens, MAX_EXPERTS, used, INVALID);
        }
        batch_case(GGML_TYPE_Q8_0, 126, 65, MAX_EXPERTS, used, REPEATED);
        batch_case(GGML_TYPE_Q8_0, HIDDEN, 512, MAX_EXPERTS, used, RANDOM);
    }
    wrapper_case(); puts("Inkling expert batch checks passed"); return 0;
}
