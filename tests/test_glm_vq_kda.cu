// Tensor-core chunked KDA (glmvq_kda_local + glmvq_kda_scan) against the F32
// recurrence followed by the gated output norm, on random GLM-shaped heads.
// The chunked path changes F32 order and rounds MMA operands to F16, so it is
// judged by relative error of the normed output and the final state.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>
#include "../cuda/glm_vq_kda.cuh"
#define CK(...) do { cudaError_t e_ = (__VA_ARGS__); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
static std::mt19937 rng(11);
template <typename T> static T *up(const std::vector<T> &v) { T *p; CK(cudaMalloc(&p, v.size() * sizeof(T))); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <typename T> static std::vector<T> down(const T *p, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
static std::vector<float> normal(size_t n, float mean, float sd) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<float> v(n);
    for (auto &x : v) { x = nd(rng); }
    return v;
}

template <typename T> static T *input(size_t n, float mean, float sd) {
    const auto values = normal(n, mean, sd);
    return up(std::vector<T>(values.begin(), values.end()));
}

// The extra warp must neither own input rows nor leave record/tail holes.
template <typename Q, typename F> static int local_parity(uint32_t rows) {
    const uint32_t heads = 3, width = heads * GLMVQ_KDA_DIM;
    const size_t count = (size_t)rows * width, tail = width * GLMVQ_CONV;
    auto *q = input<Q>(count, 0, 1), *k = input<Q>(count, 0, 1), *v = input<Q>(count, 0, 1);
    auto *f = input<F>(count, -2, 2);
    auto *b = input<float>(rows * heads, 0, 2), *qw = input<float>(tail, 0, .5f);
    auto *kw = input<float>(tail, 0, .5f), *vw = input<float>(tail, 0, .5f);
    auto *qt = input<float>(tail, 0, 1), *kt = input<float>(tail, 0, 1), *vt = input<float>(tail, 0, 1);
    auto *a = input<float>(heads, 0, .5f), *bias = input<float>(width, -2, 1);
    const uint32_t chunks = (rows + GLMVQ_KC - 1) / GLMVQ_KC;
    const size_t bytes = (size_t)chunks * heads * GLMVQ_KR_BYTES;
    uint8_t *old, *next;
    CK(cudaMalloc(&old, bytes + 16)); CK(cudaMalloc(&next, bytes + 16));
    CK(cudaMemset(old, 0x3c, bytes + 16)); CK(cudaMemset(next, 0xa5, bytes + 16));
    CK(cudaFuncSetAttribute(glmvq_kda_local<Q, F, GLMVQ_KINPUT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, GLMVQ_KLOCAL_SMEM));
    CK(cudaFuncSetAttribute(glmvq_kda_local<Q, F>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, GLMVQ_KLOCAL_SMEM));
    glmvq_kda_local<Q, F, GLMVQ_KINPUT><<<dim3(chunks, heads), GLMVQ_KINPUT, GLMVQ_KLOCAL_SMEM>>>(
        old, qt, kt, vt, q, k, v, f, b, qw, kw, vw, a, bias, rows, heads, -5.0f);
    glmvq_kda_local<Q, F><<<dim3(chunks, heads), GLMVQ_KLOCAL, GLMVQ_KLOCAL_SMEM>>>(
        next, qt, kt, vt, q, k, v, f, b, qw, kw, vw, a, bias, rows, heads, -5.0f);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const auto before = down(old, bytes + 16), after = down(next, bytes + 16);
    bool ok = std::equal(before.begin(), before.begin() + bytes, after.begin());
    for (size_t i = bytes; i < bytes + 16; i++) { ok &= before[i] == 0x3c && after[i] == 0xa5; }
    printf("kda local rows=%u q=%zu f=%zu: records and canary %s\n", rows, sizeof(Q), sizeof(F), ok ? "ok" : "FAIL");
    for (void *p : {(void *)q, (void *)k, (void *)v, (void *)f, (void *)b, (void *)qw, (void *)kw,
            (void *)vw, (void *)qt, (void *)kt, (void *)vt, (void *)a, (void *)bias, (void *)old, (void *)next}) { cudaFree(p); }
    return !ok;
}

struct Diff { double rel, worst; };
static Diff compare(const std::vector<float> &a, const std::vector<float> &b) {
    double num = 0, den = 0, worst = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - b[i];
        num += d * d; den += (double)b[i] * b[i]; worst = fmax(worst, fabs(d));
    }
    return {sqrt(num / fmax(den, 1e-30)), worst};
}

static int run(uint32_t rows, float state_sd, bool timing) {
    const uint32_t heads = 64, D = GLMVQ_KDA_DIM;
    const uint64_t width = (uint64_t)heads * D, count = (uint64_t)rows * width, tail = width * GLMVQ_CONV;
    auto q = normal(count, 0, 1), k = normal(count, 0, 1), v = normal(count, 0, 1);
    auto forget = normal(count, -2, 2), beta = normal((uint64_t)rows * heads, 0, 2);
    auto qw = normal(tail, 0, 0.5f), kw = normal(tail, 0, 0.5f), vw = normal(tail, 0, 0.5f);
    auto a_log = normal(heads, 0, 0.5f), dt = normal(width, -2, 1), gate = normal(count, 0, 2);
    auto state = normal(width * D, 0, state_sd), qt = normal(tail, 0, 1), kt = normal(tail, 0, 1), vt = normal(tail, 0, 1);
    std::vector<__nv_bfloat16> norm(D);
    for (uint32_t i = 0; i < D; i++) { norm[i] = __float2bfloat16(0.5f + (float)(rng() % 1000) / 1000.0f); }
    float *dq = up(q), *dk = up(k), *dv = up(v), *df = up(forget), *db = up(beta), *dqw = up(qw), *dkw = up(kw);
    float *dvw = up(vw), *da = up(a_log), *ddt = up(dt), *dg = up(gate);
    __nv_bfloat16 *dn = up(norm);
    // Reference: recurrence, then the warp gated norm.
    float *rs = up(state), *rqt = up(qt), *rkt = up(kt), *rvt = up(vt), *rout, *rheads;
    CK(cudaMalloc(&rout, count * 4)); CK(cudaMalloc(&rheads, count * 4));
    glmvq_kda<<<heads, D>>>(rout, rs, rqt, rkt, rvt, dq, dk, dv, df, db, dqw, dkw, dvw, da, ddt, rows, heads, -5.0f, {});
    glmvq_rms_warp<<<(rows * heads + GLMVQ_RMS_ROWS - 1) / GLMVQ_RMS_ROWS, GLMVQ_RMS_ROWS * 32>>>(
        rout, dn, dg, rheads, rows * heads, D, 1e-6f, GLMVQ_RMS_GATED);
    CK(cudaDeviceSynchronize());
    // Chunked.
    const uint32_t chunks = (rows + GLMVQ_KC - 1) / GLMVQ_KC;
    float *cs = up(state), *cqt = up(qt), *ckt = up(kt), *cvt = up(vt), *cheads;
    uint8_t *records;
    CK(cudaMalloc(&records, (uint64_t)chunks * heads * GLMVQ_KR_BYTES)); CK(cudaMalloc(&cheads, count * 4));
    CK(cudaFuncSetAttribute(glmvq_kda_local<float, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_KLOCAL_SMEM));
    CK(cudaFuncSetAttribute(glmvq_kda_scan<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_KSCAN_SMEM));
    auto chunked = [&]() {
        glmvq_kda_local<<<dim3(chunks, heads), GLMVQ_KLOCAL, GLMVQ_KLOCAL_SMEM>>>(records, cqt, ckt, cvt,
            dq, dk, dv, df, db, dqw, dkw, dvw, da, ddt, rows, heads, -5.0f);
        glmvq_kda_scan<<<heads, GLMVQ_KSCAN, GLMVQ_KSCAN_SMEM>>>(cheads, cs, records, dg, dn, rows, heads, 1e-6f);
    };
    chunked();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const Diff out = compare(down(cheads, count), down(rheads, count));
    const Diff st = compare(down(cs, width * D), down(rs, width * D));
    // F16 operands: ~1 BF16 ulp on a few percent of outputs, state ~2e-4 relative.
    const bool ok = out.rel < 3e-3 && st.rel < 5e-4 && std::isfinite(out.worst) && std::isfinite(st.worst);
    printf("kda rows=%u state_sd=%.1f: output relRMS %.3g max %.3g, state relRMS %.3g max %.3g %s\n",
        rows, state_sd, out.rel, out.worst, st.rel, st.worst, ok ? "ok" : "FAIL");
    if (timing) {
        cudaEvent_t e0, e1, e2;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
        const int reps = 10;
        float local = 0, scan = 0, recur = 0;
        for (int r = 0; r < reps; r++) {
            CK(cudaEventRecord(e0));
            glmvq_kda_local<<<dim3(chunks, heads), GLMVQ_KLOCAL, GLMVQ_KLOCAL_SMEM>>>(records, cqt, ckt, cvt,
                dq, dk, dv, df, db, dqw, dkw, dvw, da, ddt, rows, heads, -5.0f);
            CK(cudaEventRecord(e1));
            glmvq_kda_scan<<<heads, GLMVQ_KSCAN, GLMVQ_KSCAN_SMEM>>>(cheads, cs, records, dg, dn, rows, heads, 1e-6f);
            CK(cudaEventRecord(e2));
            CK(cudaEventSynchronize(e2));
            float a, b;
            CK(cudaEventElapsedTime(&a, e0, e1)); CK(cudaEventElapsedTime(&b, e1, e2));
            local += a; scan += b;
        }
        CK(cudaEventRecord(e0));
        glmvq_kda<<<heads, D>>>(rout, rs, rqt, rkt, rvt, dq, dk, dv, df, db, dqw, dkw, dvw, da, ddt, rows, heads, -5.0f, {});
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&recur, e0, e1));
        printf("  timing rows=%u: local %.3f ms, scan %.3f ms (recurrence %.2f ms)\n", rows, local / reps, scan / reps, recur);
    }
    for (void *p : {(void *)dq, (void *)dk, (void *)dv, (void *)df, (void *)db, (void *)dqw, (void *)dkw, (void *)dvw,
            (void *)da, (void *)ddt, (void *)dg, (void *)dn, (void *)rs, (void *)rqt, (void *)rkt, (void *)rvt,
            (void *)rout, (void *)rheads, (void *)cs, (void *)cqt, (void *)ckt, (void *)cvt, (void *)cheads,
            (void *)records}) { cudaFree(p); }
    return !ok;
}

enum class Snapshots { Omit, Keep };

// Each verify frontier must equal an ordinary one-row recurrence, including
// the conv tails and the state restored after any rejected draft.
static int verify_parity(uint32_t rows, uint32_t heads, Snapshots snapshots) {
    const size_t width = heads * GLMVQ_KDA_DIM, count = rows * width;
    const size_t state = width * GLMVQ_KDA_DIM, tail = width * GLMVQ_CONV;
    const size_t stride = state + 3 * tail + 16, shots = rows * stride;
    auto *q = input<float>(count, 0, 1), *k = input<float>(count, 0, 1), *v = input<float>(count, 0, 1);
    auto *f = input<float>(count, -2, 2), *b = input<float>(rows * heads, 0, 2);
    auto *qw = input<float>(tail, 0, .5f), *kw = input<float>(tail, 0, .5f), *vw = input<float>(tail, 0, .5f);
    auto *a = input<float>(heads, 0, .5f), *bias = input<float>(width, -2, 1);
    auto initial = normal(stride, 0, .3f);
    std::fill(initial.end() - 16, initial.end(), -1234.0f);
    auto *reference = up(initial), *reference_out = up(std::vector<float>(count + 16, -1234));
    auto *reference_shots = up(std::vector<float>(shots + 16, -1234));
    for (uint32_t t = 0; t < rows; t++) {
        glmvq_kda<<<heads, GLMVQ_KDA_DIM>>>(reference_out + t * width, reference,
            reference + state, reference + state + tail, reference + state + 2 * tail,
            q + t * width, k + t * width, v + t * width, f + t * width, b + t * heads,
            qw, kw, vw, a, bias, 1, heads, -5.0f, {});
        CK(cudaGetLastError());
        if (snapshots == Snapshots::Keep && t + 1 < rows) {
            CK(cudaMemcpy(reference_shots + t * stride, reference, stride * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    const auto expected = down(reference, stride), expected_out = down(reference_out, count + 16);
    const auto expected_shots = down(reference_shots, shots + 16);
    int failed = 0;
    for (auto reads : {GLMVQ_KDA_CACHED, GLMVQ_KDA_RELOAD}) {
        auto *pool = up(initial), *out = up(std::vector<float>(count + 16, -1234));
        auto *shot = up(std::vector<float>(shots + 16, -1234));
        const glmvq_kda_snap snap = snapshots == Snapshots::Keep
            ? glmvq_kda_snap{shot, shot + state, shot + state + tail, shot + state + 2 * tail, stride}
            : glmvq_kda_snap{};
        auto kernel = reads == GLMVQ_KDA_RELOAD ? glmvq_kda<true, GLMVQ_KDA_RELOAD> : glmvq_kda<true>;
        kernel<<<heads, GLMVQ_KDA_DIM>>>(out, pool, pool + state, pool + state + tail, pool + state + 2 * tail,
            q, k, v, f, b, qw, kw, vw, a, bias, rows, heads, -5.0f, snap);
        CK(cudaGetLastError());
        const auto actual = down(pool, stride), actual_out = down(out, count + 16);
        const auto actual_shots = down(shot, shots + 16);
        const bool ok = !memcmp(actual.data(), expected.data(), stride * sizeof(float)) &&
            !memcmp(actual_out.data(), expected_out.data(), (count + 16) * sizeof(float)) &&
            !memcmp(actual_shots.data(), expected_shots.data(), (shots + 16) * sizeof(float));
        printf("kda verify rows=%u heads=%u snapshots=%d reads=%d: state, output, rollback, canaries %s\n",
            rows, heads, (int)snapshots, (int)reads, ok ? "ok" : "FAIL");
        failed += !ok;
        cudaFree(pool); cudaFree(out); cudaFree(shot);
    }
    for (void *p : {(void *)q, (void *)k, (void *)v, (void *)f, (void *)b, (void *)qw, (void *)kw,
            (void *)vw, (void *)a, (void *)bias, (void *)reference, (void *)reference_out, (void *)reference_shots}) {
        cudaFree(p);
    }
    return failed;
}

int main(int argc, char **argv) {
    int failed = 0;
    for (uint32_t rows : {2u, 3u, 4u}) { failed += verify_parity(rows, 3, Snapshots::Keep); }
    failed += verify_parity(3, 64, Snapshots::Keep);
    for (uint32_t rows : {3u, 5u, 63u}) { failed += verify_parity(rows, 3, Snapshots::Omit); }
    if (argc == 2 && !strcmp(argv[1], "--verify")) {
        printf("%s\n", failed ? "FAILED" : "all ok");
        return failed;
    }
    rng.seed(11);
    for (uint32_t rows : {1u, 63u, 64u, 65u, 200u}) {
        failed += local_parity<__nv_bfloat16, __nv_bfloat16>(rows);
    }
    failed += local_parity<float, float>(65);
    failed += local_parity<float, __nv_bfloat16>(65);
    failed += local_parity<__nv_bfloat16, float>(65);
    failed += run(64, 0.0f, false);
    failed += run(200, 0.3f, false);
    failed += run(2048, 0.3f, true);
    failed += run(4096, 0.3f, true);
    printf("%s\n", failed ? "FAILED" : "all ok");
    return failed;
}
