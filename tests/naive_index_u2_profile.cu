/* Original kernel/GPU bytes are the full oracle. The finite CPU equation
 * samples bounded keys; exceptional fixtures use original GPU bytes only. */
#include "../cuda/naive_primitives.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CUDA(call) do { const cudaError_t rc = (call); if (rc != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); exit(1); } } while (0)

enum class Path : unsigned { Original, U2, Guarded };
enum class Fixture : unsigned { Normal, Ties, KeyEdges, QueryEdges };
static constexpr unsigned PROFILE_REPEATS = 20;
static constexpr unsigned PROFILE_THREADS = 128;
static constexpr unsigned CPU_SAMPLES = 64;

template<class T> static T *upload(const std::vector<T> &values) {
    T *out;
    CUDA(cudaMalloc(&out, values.size() * sizeof(T)));
    CUDA(cudaMemcpy(out, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return out;
}

static float from_bits(uint32_t bits) {
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint32_t to_bits(float value) {
    uint32_t out;
    memcpy(&out, &value, sizeof(out));
    return out;
}

static float cpu_mul(float a, float b) {
    volatile float out = a * b;
    return out;
}

static float cpu_add(float a, float b) {
    volatile float out = a + b;
    return out;
}

static float cpu_e4m3(uint8_t code) {
    const unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
    const float magnitude = exponent ? std::ldexp((float)(8 + mantissa), (int)exponent - 10)
                                     : std::ldexp((float)mantissa, -9);
    return code & 128 ? -magnitude : magnitude;
}

static float cpu_score(const std::vector<float> &query, const std::vector<uint8_t> &codes,
        const std::vector<float> &scales, const std::vector<float> &weights,
        unsigned row, unsigned key) {
    float k[N05_INDEX_DIM];
    for (unsigned d = 0; d < N05_INDEX_DIM; d++) {
        k[d] = cpu_mul(cpu_e4m3(codes[(uint64_t)key * N05_INDEX_DIM + d]), scales[key]);
    }
    float score = 0;
    for (unsigned h = 0; h < N05_INDEX_HEADS; h++) {
        float dots[N05_INDEX_WARP] = {}, next[N05_INDEX_WARP];
        const float *head = query.data() + ((uint64_t)row * N05_INDEX_HEADS + h) * N05_INDEX_DIM;
        for (unsigned lane = 0; lane < N05_INDEX_WARP; lane++) {
            for (unsigned part = 0; part < N05_INDEX_PARTS; part++) {
                const unsigned d = lane + part * N05_INDEX_WARP;
                dots[lane] = std::fma(head[d], k[d], dots[lane]);
            }
        }
        for (unsigned step = N05_INDEX_WARP / 2; step; step /= 2) {
            for (unsigned lane = 0; lane < N05_INDEX_WARP; lane++) {
                next[lane] = cpu_add(dots[lane], dots[lane ^ step]);
            }
            memcpy(dots, next, sizeof(dots));
        }
        score = cpu_add(score, cpu_mul(fmaxf(0, dots[0]), weights[(uint64_t)row * N05_INDEX_HEADS + h]));
    }
    return score;
}

static void fill_inputs(std::vector<uint8_t> &codes, std::vector<float> &scales,
        std::vector<float> &query, std::vector<float> &weights, Fixture fixture) {
    static const uint32_t edges[] = {
        0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
        0x00800000, 0x80800000, 0x00000001, 0x80000001, 0x7f800000, 0xff800000,
        0x7fc00001, 0xffc00031, 0x7f800001, 0x3b800000,
    };
    constexpr unsigned EDGE_COUNT = sizeof(edges) / sizeof(edges[0]);
    for (unsigned key = 0; key < scales.size(); key++) {
        scales[key] = .001234567f * (1 + key % 17);
        if (fixture == Fixture::Ties) { scales[key] = .125f; }
        if (fixture == Fixture::KeyEdges) { scales[key] = from_bits(edges[key % EDGE_COUNT]); }
        for (unsigned d = 0; d < N05_INDEX_DIM; d++) {
            uint8_t code = ((key * 13 + d * 19) % 127) | ((key + d) % 2 ? 128 : 0);
            if (fixture == Fixture::Ties) { code = (d & 1) ? 0xb8 : 0x38; }
            if (fixture == Fixture::KeyEdges) { code = (uint8_t)((uint64_t)key * 37 + d); }
            codes[(uint64_t)key * N05_INDEX_DIM + d] = code;
        }
    }
    for (size_t i = 0; i < query.size(); i++) {
        query[i] = ((int)((i * 7) % 113) - 56) * .0137f;
        if (fixture == Fixture::Ties) {
            query[i] = i / (N05_INDEX_HEADS * N05_INDEX_DIM) % 2 ? .25f : from_bits(i & 1 ? 0x80000000 : 0);
        }
        if (fixture == Fixture::QueryEdges) {
            const unsigned head = (i / N05_INDEX_DIM) % N05_INDEX_HEADS;
            if (head % 4 == 0) { query[i] = from_bits(edges[(i % N05_INDEX_DIM) % EDGE_COUNT]); }
            if (head % 4 == 1) { query[i] = from_bits(i & 1 ? 0x80000000 : 0); }
            if (head % 4 == 2) { query[i] = from_bits(i & 1 ? 0x00000001 : 0x80800000); }
            if (head % 4 == 3) { query[i] = from_bits(i & 1 ? 0x7f7fffff : 0xff7fffff); }
        }
    }
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = ((int)((i * 11) % 31) - 15) * .007123f;
        if (fixture == Fixture::Ties) { weights[i] = i & 1 ? -.25f : .25f; }
        if (fixture == Fixture::KeyEdges) { weights[i] = i & 1 ? -.125f : .125f; }
        if (fixture == Fixture::QueryEdges) { weights[i] = from_bits(edges[(i * 7) % EDGE_COUNT]); }
    }
}

static void check_bytes(const void *got, const void *want, size_t bytes, const char *label) {
    if (!memcmp(got, want, bytes)) { return; }
    const auto *a = (const unsigned char *)got, *b = (const unsigned char *)want;
    size_t i = 0;
    while (i < bytes && a[i] == b[i]) { i++; }
    fprintf(stderr, "%s differs at byte %zu: %02x != %02x\n", label, i, a[i], b[i]);
    exit(3);
}

int main(int argc, char **argv) {
    const unsigned history = argc >= 2 ? (unsigned)atoi(argv[1]) : 8192;
    const unsigned rows = argc >= 3 ? (unsigned)atoi(argv[2]) : N05_QUERY_TILE;
    const auto path = argc >= 4 ? (Path)(unsigned)atoi(argv[3]) : Path::Original;
    const auto fixture = argc >= 5 ? (Fixture)(unsigned)atoi(argv[4]) : Fixture::Normal;
    const unsigned repeats = argc >= 6 ? (unsigned)atoi(argv[5]) : PROFILE_REPEATS;
    const unsigned last = argc >= 7 ? (unsigned)atoi(argv[6]) : history - 1;
    if (!rows || rows > N05_QUERY_TILE || history < rows || history > N05_CONTEXT ||
        path > Path::Guarded || fixture > Fixture::QueryEdges || !repeats || repeats > PROFILE_REPEATS ||
        last >= history || rows > last + 1) { return 2; }
    const bool use_u2 = path == Path::U2 || (path == Path::Guarded && rows == N05_QUERY_TILE && history > N05_TOP_K);

    std::vector<uint8_t> codes((size_t)history * N05_INDEX_DIM);
    std::vector<float> scales(history), query((size_t)rows * N05_INDEX_HEADS * N05_INDEX_DIM);
    std::vector<float> weights((size_t)rows * N05_INDEX_HEADS);
    std::vector<unsigned> positions(rows);
    fill_inputs(codes, scales, query, weights, fixture);
    for (unsigned r = 0; r < rows; r++) { positions[r] = last - rows + 1 + r; }
    auto *dc = upload(codes);
    auto *dp = upload(positions);
    auto *ds = upload(scales), *dq = upload(query), *packed = upload(query), *dw = upload(weights);
    float *out, *reference;
    CUDA(cudaMalloc(&out, (size_t)rows * history * sizeof(float)));
    CUDA(cudaMalloc(&reference, (size_t)rows * history * sizeof(float)));
    unsigned *ids, *reference_ids;
    CUDA(cudaMalloc(&ids, (size_t)rows * N05_TOP_K * sizeof(unsigned)));
    CUDA(cudaMalloc(&reference_ids, (size_t)rows * N05_TOP_K * sizeof(unsigned)));
    const unsigned tiles = (history + N05_HISTORY_TILE - 1) / N05_HISTORY_TILE;
    uint64_t *a, *b;
    CUDA(cudaMalloc(&a, (size_t)rows * tiles * N05_TOP_K * sizeof(uint64_t)));
    CUDA(cudaMalloc(&b, (size_t)rows * tiles * N05_TOP_K * sizeof(uint64_t)));
    naive_fp8_query<<<rows * N05_INDEX_HEADS, N05_INDEX_DIM>>>(dq, rows * N05_INDEX_HEADS);
    naive_fp8_query<NaiveIndexLayout::Warp><<<rows * N05_INDEX_HEADS, N05_INDEX_DIM>>>(packed, rows * N05_INDEX_HEADS);
    naive_index_scores<NaiveIndexLayout::Warp><<<dim3((history + 3) / 4, rows), PROFILE_THREADS>>>(
        reference, packed, dc, ds, dw, dp, history);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());

    const auto launch = [&]() {
        if (use_u2) {
            naive_index_u2<<<dim3((history + 7) / 8, rows), PROFILE_THREADS>>>(out, packed, dc, ds, dw, dp, history);
        } else {
            naive_index_scores<NaiveIndexLayout::Warp><<<dim3((history + 3) / 4, rows), PROFILE_THREADS>>>(
                out, packed, dc, ds, dw, dp, history);
        }
        CUDA(cudaGetLastError());
    };
    std::vector<float> expected((size_t)rows * history), result(expected.size());
    CUDA(cudaMemcpy(expected.data(), reference, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(naive_topk_launch(reference_ids, reference, a, b, dp, history, rows, nullptr));
    std::vector<unsigned> expected_ids((size_t)rows * N05_TOP_K), result_ids(expected_ids.size());
    CUDA(cudaMemcpy(expected_ids.data(), reference_ids, expected_ids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    const auto verify = [&]() {
        CUDA(cudaMemcpy(result.data(), out, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check_bytes(result.data(), expected.data(), result.size() * sizeof(float), "score GPU bytes");
        CUDA(naive_topk_launch(ids, out, a, b, dp, history, rows, nullptr));
        CUDA(cudaMemcpy(result_ids.data(), ids, result_ids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        check_bytes(result_ids.data(), expected_ids.data(), result_ids.size() * sizeof(unsigned), "stable top-k IDs");
        for (unsigned r = 0; r < rows; r++) {
            for (unsigned key = positions[r] + 1; key < history; key++) {
                if (to_bits(result[(uint64_t)r * history + key]) != 0xff800000u) { exit(4); }
            }
        }
    };
    launch();
    verify();

    size_t cpu_samples = 0;
    const bool cpu_finite = fixture == Fixture::Normal || fixture == Fixture::Ties;
    if (cpu_finite) {
        CUDA(cudaMemcpy(query.data(), dq, query.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (unsigned r = 0; r < rows; r++) {
            std::vector<unsigned> keys;
            for (unsigned i = 0; i < std::min(history, CPU_SAMPLES); i++) {
                keys.push_back(i < CPU_SAMPLES / 2 ? i : (unsigned)((uint64_t)(i - CPU_SAMPLES / 2) * history / (CPU_SAMPLES / 2)));
            }
            for (unsigned i = 0; i < 8; i++) {
                if (history > i) { keys.push_back(history - i - 1); }
                if (positions[r] >= i) { keys.push_back(positions[r] - i); }
                if (positions[r] + i + 1 < history) { keys.push_back(positions[r] + i + 1); }
            }
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            for (unsigned key : keys) {
                const float want = key <= positions[r] ? cpu_score(query, codes, scales, weights, r, key) : -INFINITY;
                const float got = expected[(uint64_t)r * history + key];
                if (to_bits(got) != to_bits(want)) {
                    fprintf(stderr, "finite CPU sample row=%u key=%u got=%08x want=%08x\n", r, key, to_bits(got), to_bits(want));
                    return 5;
                }
                cpu_samples++;
            }
        }
    }

    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    launch();
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaEventRecord(start));
    for (unsigned repeat = 0; repeat < repeats; repeat++) { launch(); }
    CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
    float elapsed;
    CUDA(cudaEventElapsedTime(&elapsed, start, end));
    verify();
    const size_t device_bytes = codes.size() + (scales.size() + 2 * query.size() + weights.size() + 2 * result.size()) * sizeof(float) +
        positions.size() * sizeof(unsigned) + 2 * result_ids.size() * sizeof(unsigned) + (size_t)2 * rows * tiles * N05_TOP_K * sizeof(uint64_t);
    printf("index history=%u rows=%u path=%u effective=%s fixture=%u last_pos=%u repeats=%u milliseconds=%.6f "
        "gpu_before=1 gpu_after=1 ids_before=1 ids_after=1 cpu_sample_exact=%u cpu_samples=%zu cpu_full=0 device_bytes=%zu\n",
        history, rows, (unsigned)path, use_u2 ? "u2" : "original", (unsigned)fixture, last, repeats, elapsed / repeats,
        (unsigned)cpu_finite, cpu_samples, device_bytes);
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
    CUDA(cudaFree(dc)); CUDA(cudaFree(dp)); CUDA(cudaFree(ds)); CUDA(cudaFree(dq)); CUDA(cudaFree(packed)); CUDA(cudaFree(dw));
    CUDA(cudaFree(out)); CUDA(cudaFree(reference)); CUDA(cudaFree(ids)); CUDA(cudaFree(reference_ids)); CUDA(cudaFree(a)); CUDA(cudaFree(b));
    return 0;
}
