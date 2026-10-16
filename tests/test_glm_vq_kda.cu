// Tensor-core chunked KDA (glmvq_kda_local + glmvq_kda_scan) against the F32
// recurrence followed by the gated output norm, on random GLM-shaped heads.
// The chunked path changes F32 order and rounds MMA operands to F16, so it is
// judged by relative error of the normed output and the final state.
#include <cstdio>
#include <cmath>
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
    CK(cudaFuncSetAttribute(glmvq_kda_local<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GLMVQ_KLOCAL_SMEM));
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

int main() {
    int failed = 0;
    failed += run(64, 0.0f, false);
    failed += run(200, 0.3f, false);
    failed += run(2048, 0.3f, true);
    failed += run(4096, 0.3f, true);
    printf("%s\n", failed ? "FAILED" : "all ok");
    return failed;
}
