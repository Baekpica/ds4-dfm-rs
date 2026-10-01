/* Production router versus the stable-argmax fast path.
 * Resident synthetic inputs, rotating output spans, and no model projection
 * differ from whole inference. Captures are F32 logits[rows,256], bias[256]. */
#include <cuda_runtime.h>
#include "../cuda/naive_primitives.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#define CUDA(call) do { const cudaError_t rc = (call); if (rc != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); exit(1); } } while (0)

enum class Mode { Base, Warp };
enum class Fixture { Seeded, Ties, SignedZero, Extremes, ScoreZero, Negative,
                     MinusInf, AllMinusInf, PlusInf, Nonfinite, ExtremeBias,
                     UlpTies, TinySum, NanBiasZero, NanBiasTail, NanBiasAll,
                     MixedInfBias, FewFinite, PlusInfLogits, NanLogitsAll, Captured };
enum { REPEATS = 20, ROUTER_SEED = 0x4e303546, LOGIT_STEPS = 128, BIAS_STEPS = 32 };
static constexpr float WEIGHT_EPS = 8 * std::numeric_limits<float>::epsilon();

template<class T> static T *upload(const std::vector<T> &values) {
    T *out;
    CUDA(cudaMalloc(&out, values.size() * sizeof(T)));
    CUDA(cudaMemcpy(out, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return out;
}

static void read_fixture(const char *path, std::vector<float> &logits, std::vector<float> &bias) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    assert(fread(logits.data(), sizeof(float), logits.size(), fp) == logits.size());
    assert(fread(bias.data(), sizeof(float), bias.size(), fp) == bias.size());
    assert(fgetc(fp) == EOF && !ferror(fp));
    assert(!fclose(fp));
}

static void make_fixture(std::vector<float> &logits, std::vector<float> &bias, Fixture fixture) {
    std::mt19937 rng(ROUTER_SEED);
    const float maximum = std::numeric_limits<float>::max();
    const float inf = std::numeric_limits<float>::infinity();
    const uint32_t nan_bits = 0x7fc00011u;
    float nan;
    memcpy(&nan, &nan_bits, sizeof(nan));
    const float extreme[] = {-maximum, -100, -20, -0.0f, 0.0f, 20, 100, maximum};
    for (unsigned e = 0; e < N05_EXPERTS; e++) {
        if (fixture == Fixture::Seeded) {
            bias[e] = ((int)(rng() % (2 * BIAS_STEPS + 1)) - BIAS_STEPS) / 32.0f;
        } else if (fixture == Fixture::AllMinusInf) { bias[e] = -inf; }
        else if (fixture == Fixture::PlusInf) { bias[e] = inf; }
        else if (fixture == Fixture::ExtremeBias) { bias[e] = e % 2 ? -maximum : maximum; }
        else if (fixture == Fixture::NanBiasZero) { bias[e] = e ? 0 : nan; }
        else if (fixture == Fixture::NanBiasTail) { bias[e] = e == N05_EXPERTS - 1 ? nan : 0; }
        else if (fixture == Fixture::NanBiasAll) { bias[e] = nan; }
        else if (fixture == Fixture::MixedInfBias) { bias[e] = e % 3 == 0 ? -inf : (e % 3 == 1 ? inf : nan); }
        else if (fixture == Fixture::FewFinite) { bias[e] = e < N05_USED - 1 ? 0 : -inf; }
        else if (fixture == Fixture::Negative) { bias[e] = -1; }
        else { bias[e] = (fixture == Fixture::SignedZero || fixture == Fixture::ScoreZero) && e % 2 ? -0.0f : 0.0f; }
    }
    for (size_t i = 0; i < logits.size(); i++) {
        if (fixture == Fixture::Seeded || fixture == Fixture::Negative || fixture == Fixture::ExtremeBias) {
            logits[i] = ((int)(rng() % (2 * LOGIT_STEPS + 1)) - LOGIT_STEPS) / 16.0f;
        } else if (fixture == Fixture::Ties) { logits[i] = (i % 4) * .5f; }
        else if (fixture == Fixture::SignedZero) { logits[i] = i % 2 ? -0.0f : 0.0f; }
        else if (fixture == Fixture::Extremes) { logits[i] = extreme[i % (sizeof(extreme) / sizeof(extreme[0]))]; }
        else if (fixture == Fixture::ScoreZero) { logits[i] = -maximum; }
        else if (fixture == Fixture::MinusInf) { logits[i] = -inf; }
        else if (fixture == Fixture::PlusInfLogits) { logits[i] = inf; }
        else if (fixture == Fixture::NanLogitsAll) { logits[i] = nan; }
        else { logits[i] = 0; }
    }
    if (fixture == Fixture::UlpTies) {
        const unsigned edges[] = {31, 32, 63, 64, 95, 96, 127, 128, 191, 192, 223, 224, 254, 255};
        for (unsigned e = 0; e < N05_EXPERTS; e++) { bias[e] = -1; }
        for (size_t i = 0; i < logits.size(); i++) { logits[i] = -maximum; }
        for (unsigned i = 0; i < sizeof(edges) / sizeof(edges[0]); i++) {
            const float p = i % 3 == 0 ? 0 : (i % 3 == 1 ? .5f : 1);
            const float target = i % 3 == 0 ? std::nextafter(.5f, inf) : .5f;
            bias[edges[i]] = target - p;
            for (size_t row = 0; row < logits.size() / N05_EXPERTS; row++) {
                logits[row * N05_EXPERTS + edges[i]] = p == 0 ? -maximum : (p == 1 ? maximum : 0);
            }
        }
        return;
    }
    if (fixture == Fixture::TinySum) {
        const unsigned chosen[] = {255, 224, 127, 96, 65, 63, 32, 0};
        for (unsigned e = 0; e < N05_EXPERTS; e++) { bias[e] = -2; }
        for (size_t i = 0; i < logits.size(); i++) { logits[i] = 8; }
        for (unsigned k = 0; k < N05_USED; k++) {
            bias[chosen[k]] = 1 + .25f * (N05_USED - k);
            for (size_t row = 0; row < logits.size() / N05_EXPERTS; row++) {
                // Half are zero; the others make sum+1e-20 materially significant.
                logits[row * N05_EXPERTS + chosen[k]] = k % 2 ? -46.5f - .5f * ((k + row) % N05_USED) : -maximum;
            }
        }
        return;
    }
    if (fixture != Fixture::Nonfinite) { return; }
    for (size_t row = 0; row < logits.size() / N05_EXPERTS; row++) {
        float *values = logits.data() + row * N05_EXPERTS;
        if (row % 4 == 0) { values[0] = nan; }
        else if (row % 4 == 1) { values[N05_USED - 1] = nan; }
        else if (row % 4 == 2) { values[0] = inf; values[N05_USED - 1] = -inf; }
        else { for (unsigned e = 0; e < N05_EXPERTS; e++) { values[e] = nan; } }
    }
}

__global__ static void router_operands(float *prob, float *score, const float *logits, const float *bias) {
    const unsigned e = threadIdx.x;
    const size_t i = (size_t)blockIdx.x * N05_EXPERTS + e;
    prob[i] = 1.0f / (1.0f + expf(-logits[i]));
    score[i] = prob[i] + bias[e];
}

static void launch(Mode mode, unsigned rows, int *ids, float *weights, const float *logits, const float *bias) {
    if (mode == Mode::Base) { naive_router<<<rows, N05_EXPERTS>>>(ids, weights, logits, bias); }
    else { naive_router_warp<NaiveRouterTrace::Off><<<rows, N05_ROUTER_WARP>>>(ids, weights, logits, bias); }
}

static void trace_operands(Mode mode, unsigned rows, int *ids, float *weights,
                           const float *logits, const float *bias, float *prob, float *score) {
    if (mode == Mode::Base) { router_operands<<<rows, N05_EXPERTS>>>(prob, score, logits, bias); }
    else { naive_router_warp<NaiveRouterTrace::On><<<rows, N05_ROUTER_WARP>>>(ids, weights, logits, bias, prob, score); }
}

static void check_operands(const char *what, const std::vector<float> &actual,
                            const std::vector<float> &expected) {
    for (size_t i = 0; i < expected.size(); i++) {
        if (std::isfinite(expected[i])) {
            if (!memcmp(actual.data() + i, expected.data() + i, sizeof(float))) { continue; }
        } else if (std::isnan(expected[i]) ? std::isnan(actual[i]) : actual[i] == expected[i]) { continue; }
        fprintf(stderr, "%s operand mismatch row=%zu expert=%zu\n", what, i / N05_EXPERTS, i % N05_EXPERTS);
        exit(1);
    }
}

static void check_cpu(const std::vector<float> &prob, const std::vector<float> &score,
                      const std::vector<int> &ids, const std::vector<float> &weights, unsigned rows) {
    for (unsigned row = 0; row < rows; row++) {
        const size_t base = (size_t)row * N05_EXPERTS;
        float work[N05_EXPERTS];
        memcpy(work, score.data() + base, sizeof(work));
        int chosen[N05_USED];
        float sum = 0;
        bool finite = true;
        for (unsigned e = 0; e < N05_EXPERTS; e++) { finite &= std::isfinite(prob[base + e]) && std::isfinite(work[e]); }
        // Mirror the original scan, including nonfinite and duplicate-ID cases.
        for (unsigned k = 0; k < N05_USED; k++) {
            unsigned best = 0;
            for (unsigned e = 1; e < N05_EXPERTS; e++) { if (work[e] > work[best]) { best = e; } }
            chosen[k] = best; sum += prob[base + best]; work[best] = -INFINITY;
        }
        for (unsigned k = 1; k < N05_USED; k++) {
            const int id = chosen[k];
            unsigned j = k;
            while (j && chosen[j - 1] > id) { chosen[j] = chosen[j - 1]; j--; }
            chosen[j] = id;
        }
        for (unsigned k = 0; k < N05_USED; k++) {
            const size_t at = (size_t)row * N05_USED + k;
            assert(ids[at] == chosen[k]);
            if (!finite) { continue; }
            const float want = prob[base + chosen[k]] / (sum + 1e-20f);
            assert(std::isfinite(weights[at]) && weights[at] >= 0);
            assert(std::fabs(weights[at] - want) <= WEIGHT_EPS);
        }
    }
}

template<class T> static void check_exact(const char *what, const std::vector<T> &actual,
                                         const std::vector<T> &expected) {
    for (unsigned repeat = 0; repeat < REPEATS; repeat++) {
        const T *values = actual.data() + repeat * expected.size();
        if (!memcmp(values, expected.data(), expected.size() * sizeof(T))) { continue; }
        for (size_t i = 0; i < expected.size(); i++) {
            if (!memcmp(values + i, expected.data() + i, sizeof(T))) { continue; }
            fprintf(stderr, "%s mismatch repeat=%u row=%zu rank=%zu\n", what, repeat, i / N05_USED, i % N05_USED);
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s ROWS base|warp FIXTURE [capture.f32]\n", argv[0]); return 2; }
    const unsigned rows = (unsigned)strtoul(argv[1], NULL, 10);
    Mode mode;
    if (!strcmp(argv[2], "base")) { mode = Mode::Base; }
    else if (!strcmp(argv[2], "warp")) { mode = Mode::Warp; }
    else { return 2; }
    const char *name = argv[3];
    Fixture fixture;
    if (!strcmp(name, "seeded")) { fixture = Fixture::Seeded; }
    else if (!strcmp(name, "ties")) { fixture = Fixture::Ties; }
    else if (!strcmp(name, "signed-zero")) { fixture = Fixture::SignedZero; }
    else if (!strcmp(name, "extremes")) { fixture = Fixture::Extremes; }
    else if (!strcmp(name, "score-zero")) { fixture = Fixture::ScoreZero; }
    else if (!strcmp(name, "negative")) { fixture = Fixture::Negative; }
    else if (!strcmp(name, "minus-inf")) { fixture = Fixture::MinusInf; }
    else if (!strcmp(name, "all-minus-inf")) { fixture = Fixture::AllMinusInf; }
    else if (!strcmp(name, "plus-inf")) { fixture = Fixture::PlusInf; }
    else if (!strcmp(name, "nonfinite")) { fixture = Fixture::Nonfinite; }
    else if (!strcmp(name, "extreme-bias")) { fixture = Fixture::ExtremeBias; }
    else if (!strcmp(name, "ulp-ties")) { fixture = Fixture::UlpTies; }
    else if (!strcmp(name, "tiny-sum")) { fixture = Fixture::TinySum; }
    else if (!strcmp(name, "nan-bias-zero")) { fixture = Fixture::NanBiasZero; }
    else if (!strcmp(name, "nan-bias-tail")) { fixture = Fixture::NanBiasTail; }
    else if (!strcmp(name, "nan-bias-all")) { fixture = Fixture::NanBiasAll; }
    else if (!strcmp(name, "mixed-inf-bias")) { fixture = Fixture::MixedInfBias; }
    else if (!strcmp(name, "few-finite")) { fixture = Fixture::FewFinite; }
    else if (!strcmp(name, "plus-inf-logits")) { fixture = Fixture::PlusInfLogits; }
    else if (!strcmp(name, "nan-logits-all")) { fixture = Fixture::NanLogitsAll; }
    else if (!strcmp(name, "captured")) { fixture = Fixture::Captured; }
    else { return 2; }
    if ((rows != 1 && rows != N05_PREFILL) || (fixture == Fixture::Captured ? argc != 5 : argc != 4)) { return 2; }

    std::vector<float> logits((size_t)rows * N05_EXPERTS), bias(N05_EXPERTS);
    if (fixture == Fixture::Captured) { read_fixture(argv[4], logits, bias); }
    else { make_fixture(logits, bias, fixture); }
    size_t bad_rows = 0;
    for (unsigned row = 0; row < rows; row++) {
        bool bad = false;
        for (unsigned e = 0; e < N05_EXPERTS; e++) { bad |= !std::isfinite(logits[(size_t)row * N05_EXPERTS + e]) || !std::isfinite(bias[e]); }
        bad_rows += bad;
    }
    float *dl = upload(logits), *db = upload(bias), *dp, *ds, *tp, *ts, *dw, *rw;
    int *di, *ri;
    const size_t count = (size_t)rows * N05_USED;
    CUDA(cudaMalloc(&dp, logits.size() * sizeof(float)));
    CUDA(cudaMalloc(&ds, logits.size() * sizeof(float)));
    CUDA(cudaMalloc(&tp, logits.size() * sizeof(float)));
    CUDA(cudaMalloc(&ts, logits.size() * sizeof(float)));
    CUDA(cudaMalloc(&ri, count * sizeof(int)));
    CUDA(cudaMalloc(&rw, count * sizeof(float)));
    CUDA(cudaMalloc(&di, REPEATS * count * sizeof(int)));
    CUDA(cudaMalloc(&dw, REPEATS * count * sizeof(float)));
    router_operands<<<rows, N05_EXPERTS>>>(dp, ds, dl, db);
    naive_router<<<rows, N05_EXPERTS>>>(ri, rw, dl, db);
    trace_operands(mode, rows, di, dw, dl, db, tp, ts);
    launch(mode, rows, di, dw, dl, db);
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());

    std::vector<float> prob(logits.size()), score(logits.size()), traced_prob(logits.size()), traced_score(logits.size()), expected_weights(count);
    std::vector<int> expected_ids(count);
    CUDA(cudaMemcpy(prob.data(), dp, prob.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(score.data(), ds, score.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(traced_prob.data(), tp, traced_prob.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(traced_score.data(), ts, traced_score.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_operands("probability", traced_prob, prob); check_operands("score", traced_score, score);
    CUDA(cudaMemcpy(expected_ids.data(), ri, count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(expected_weights.data(), rw, count * sizeof(float), cudaMemcpyDeviceToHost));
    check_cpu(prob, score, expected_ids, expected_weights, rows);

    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    CUDA(cudaEventRecord(start));
    for (unsigned repeat = 0; repeat < REPEATS; repeat++) {
        launch(mode, rows, di + repeat * count, dw + repeat * count, dl, db);
    }
    CUDA(cudaGetLastError()); CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
    float elapsed;
    CUDA(cudaEventElapsedTime(&elapsed, start, end));
    std::vector<int> actual_ids(REPEATS * count);
    std::vector<float> actual_weights(REPEATS * count);
    CUDA(cudaMemcpy(actual_ids.data(), di, actual_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(actual_weights.data(), dw, actual_weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_exact("IDs", actual_ids, expected_ids); check_exact("weights", actual_weights, expected_weights);
    printf("router rows=%u mode=%s fixture=%s seed=%u repeats=%u us=%.6f nonfinite_source_rows=%zu cpu_ids_exact=1 finite_operands_exact=1 gpu_ids_weights_exact=1\n",
        rows, argv[2], name, fixture == Fixture::Seeded ? (unsigned)ROUTER_SEED : 0u,
        REPEATS, elapsed * 1000 / REPEATS, bad_rows);
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
    CUDA(cudaFree(dl)); CUDA(cudaFree(db)); CUDA(cudaFree(dp)); CUDA(cudaFree(ds)); CUDA(cudaFree(tp)); CUDA(cudaFree(ts));
    CUDA(cudaFree(di)); CUDA(cudaFree(dw)); CUDA(cudaFree(ri)); CUDA(cudaFree(rw));
    return 0;
}
