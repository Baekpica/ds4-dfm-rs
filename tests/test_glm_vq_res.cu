/* R13+8 (GVR8 residual) kernels against the restore reference: restore bits,
 * decode and verify experts, FP8 prefill MMA. Random E4M3 books and indices;
 * no model files. Exit status 0 means every check passed. */
#include "cuda/glm_vq_linear.cuh"
#include "cuda/glm_vq_decode.cuh"
#include "cuda/glm_vq_mma8.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

enum { H = GLMVQ_DEC_HIDDEN, F = GLMVQ_DEC_FFN, E = GLMVQ_DEC_USED, K1 = 8192, K2 = 256, T = 24 };
static constexpr float CLAMP = 10.0f;
static constexpr uint64_t WORKSPACE = 32ull << 20;

static std::vector<void *> g_alloc;
static void *dev(uint64_t bytes, const void *host = nullptr) {
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { fprintf(stderr, "cudaMalloc\n"); exit(2); }
    g_alloc.push_back(p);
    if (host && cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { exit(2); }
    return p;
}

static float e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return b & 128 ? -v : v;
}

// Nearest finite E4M3 byte to x.
static uint8_t e4m3_byte(float x) {
    uint8_t best = 0;
    float err = INFINITY;
    for (int b = 0; b < 256; b++) {
        if ((b & 127) == 127) { continue; }
        const float d = std::fabs(e4m3_value((uint8_t)b) - x);
        if (d < err) { err = d; best = (uint8_t)b; }
    }
    return best;
}

static uint16_t f16_bits(float x) {
    const __half h = __float2half_rn(x);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

static float bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float y;
    memcpy(&y, &u, 4);
    return y;
}

struct host_matrix {
    uint32_t rows, cols;
    std::vector<uint8_t> main, plane, res;
    std::vector<uint16_t> gain;
    std::vector<uint32_t> i1;
};

static host_matrix make_matrix(std::mt19937 &rng, uint32_t rows, uint32_t cols) {
    host_matrix m{rows, cols};
    const uint64_t n = (uint64_t)rows * cols / 8;
    m.i1.resize(n);
    m.main.assign(n * 3 / 2, 0);
    m.plane.assign(n / 8, 0);
    m.res.resize(n);
    m.gain.resize(rows);
    std::uniform_int_distribution<uint32_t> idx1(0, K1 - 1), idx2(0, K2 - 1);
    std::uniform_real_distribution<float> g(0.01f, 0.05f);
    for (uint64_t j = 0; j < n; j++) {
        m.i1[j] = idx1(rng);
        m.res[j] = (uint8_t)idx2(rng);
    }
    for (uint64_t p = 0; p < n / 2; p++) {
        const uint32_t a = m.i1[2 * p] & 4095u, b = m.i1[2 * p + 1] & 4095u;
        m.main[3 * p] = a & 255u;
        m.main[3 * p + 1] = (uint8_t)((a >> 8) | ((b & 15u) << 4));
        m.main[3 * p + 2] = (uint8_t)(b >> 4);
    }
    for (uint64_t j = 0; j < n; j++) { m.plane[j / 8] |= (uint8_t)(((m.i1[j] >> 12) & 1u) << (j % 8)); }
    for (uint32_t r = 0; r < rows; r++) { m.gain[r] = f16_bits(g(rng)); }
    return m;
}

struct dev_matrix {
    glmvq_matrix vq;
};

static glmvq_matrix upload(const host_matrix &m, const void *book, const void *book2, bool res) {
    // 128-byte aligned streams, as the GVQ1/GVR8 payload layout guarantees.
    const auto *main = (const uint8_t *)dev(m.main.size(), m.main.data());
    const auto *plane = (const uint8_t *)dev(m.plane.size(), m.plane.data());
    const auto *gain = (const __half *)dev(m.gain.size() * 2, m.gain.data());
    const auto *r = (const uint8_t *)dev(m.res.size(), m.res.data());
    return {main, plane, book, gain, m.rows, m.cols, GLMVQ_E4M3, res ? r : nullptr,
        res ? (const uint8_t *)book2 : nullptr};
}

static int g_fail = 0;
static void check(bool ok, const char *what, double value) {
    printf("%-58s %s (%.3g)\n", what, ok ? "ok" : "FAIL", value);
    if (!ok) { g_fail = 1; }
}

// Relative RMS of a - b over b.
static double relrms(const std::vector<float> &a, const std::vector<float> &b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) { num += (double)(a[i] - b[i]) * (a[i] - b[i]); den += (double)b[i] * b[i]; }
    return std::sqrt(num / (den > 0 ? den : 1));
}

static std::vector<float> fetch(const float *p, uint64_t n) {
    std::vector<float> out(n);
    if (cudaMemcpy(out.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) { exit(2); }
    return out;
}

int main() {
    std::mt19937 rng(5353008);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint8_t> book1(K1 * 8), book2(K2 * 8);
    for (auto &b : book1) { b = e4m3_byte(normal(rng)); }
    for (auto &b : book2) { b = e4m3_byte(0.35f * normal(rng)); }
    const void *dbook1 = dev(book1.size(), book1.data());
    const void *dbook2 = dev(book2.size(), book2.data());

    std::vector<host_matrix> hm;
    for (uint32_t e = 0; e < E; e++) {
        hm.push_back(make_matrix(rng, F, H));
        hm.push_back(make_matrix(rng, F, H));
        hm.push_back(make_matrix(rng, H, F));
    }
    std::vector<glmvq_matrix> res_m, plain_m;
    for (const auto &m : hm) {
        res_m.push_back(upload(m, dbook1, dbook2, true));
        plain_m.push_back(res_m.back());
        plain_m.back().res = nullptr;
        plain_m.back().res_book = nullptr;
    }

    // (A) Restore: bits of BF16(gain * (C1 + C2)) against the host formula.
    {
        const host_matrix &m = hm[2];
        auto *out = (__nv_bfloat16 *)dev((uint64_t)m.rows * m.cols * 2);
        if (glmvq_restore_bf16(res_m[2], out, nullptr, GLMVQ_RESTORE_VECTOR) != cudaSuccess) { return 2; }
        std::vector<uint16_t> got((uint64_t)m.rows * m.cols);
        cudaMemcpy(got.data(), out, got.size() * 2, cudaMemcpyDeviceToHost);
        uint64_t bad = 0;
        for (uint64_t j = 0; j < got.size() / 8; j++) {
            const uint32_t r = (uint32_t)(j / (m.cols / 8));
            const float g = __half2float(*(const __half *)&m.gain[r]);
            for (uint32_t d = 0; d < 8; d++) {
                const float w = g * (e4m3_value(book1[m.i1[j] * 8 + d]) + e4m3_value(book2[m.res[j] * 8 + d]));
                uint32_t u;
                const float wb = bf16_round(w);
                memcpy(&u, &wb, 4);
                bad += got[j * 8 + d] != (uint16_t)(u >> 16);
            }
        }
        check(bad == 0, "restore BF16(gain*(C1+C2)) bit-exact", (double)bad);
    }

    // Reference experts: restore + cuBLAS BF16 GEMM, as the fallback path.
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetWorkspace(handle, dev(WORKSPACE), WORKSPACE);
    const uint64_t scratch_bytes = ((uint64_t)H * F + 4 * (H + F)) * 2 * 4;
    void *scratch = dev(scratch_bytes);
    std::vector<float> x(4 * H), weights(4 * E);
    for (auto &v : x) { v = bf16_round(normal(rng)); }
    for (auto &w : weights) { w = std::uniform_real_distribution<float>(0.05f, 0.4f)(rng); }
    auto *dx = (float *)dev(x.size() * 4, x.data());
    auto *dw = (float *)dev(weights.size() * 4, weights.data());
    auto expert_ref = [&](const std::vector<glmvq_matrix> &ms, uint32_t row, std::vector<float> &gate,
            std::vector<float> &up, std::vector<float> &out) {
        auto *g = (float *)dev(E * F * 4), *u = (float *)dev(E * F * 4), *mid = (float *)dev(E * F * 4);
        auto *part = (float *)dev(E * H * 4);
        for (uint32_t e = 0; e < E; e++) {
            const glmvq_matrix *m = &ms[3 * e];
            const glmvq_source s0 = {nullptr, nullptr, m[0], F, H, GLMVQ_WEIGHT_VQ};
            const glmvq_source s1 = {nullptr, nullptr, m[1], F, H, GLMVQ_WEIGHT_VQ};
            const glmvq_source s2 = {nullptr, nullptr, m[2], H, F, GLMVQ_WEIGHT_VQ};
            glmvq_gemm(handle, s0, dx + row * H, g + e * F, 1, GLMVQ_COMPUTE_BF16, scratch, scratch_bytes, nullptr,
                GLMVQ_RESTORE_SCALAR);
            glmvq_gemm(handle, s1, dx + row * H, u + e * F, 1, GLMVQ_COMPUTE_BF16, scratch, scratch_bytes, nullptr,
                GLMVQ_RESTORE_SCALAR);
            glmvq_swiglu<<<F / GLMVQ_THREADS, GLMVQ_THREADS>>>(g + e * F, u + e * F, mid + e * F, F, CLAMP);
            glmvq_gemm(handle, s2, mid + e * F, part + e * H, 1, GLMVQ_COMPUTE_BF16, scratch, scratch_bytes, nullptr,
                GLMVQ_RESTORE_SCALAR);
        }
        gate = fetch(g, E * F);
        up = fetch(u, E * F);
        std::vector<float> p = fetch(part, E * H);
        out.assign(H, 0.0f);
        for (uint32_t h = 0; h < H; h++) {
            double sum = 0;
            for (uint32_t e = 0; e < E; e++) { sum += (double)p[e * H + h] * weights[row * E + e]; }
            out[h] = (float)sum;
        }
    };

    // (B) One-row decode, both gain modes, against the reference and plain R13.
    std::vector<glmvq_decode_item> items(E);
    for (uint32_t e = 0; e < E; e++) { items[e] = {res_m[3 * e], res_m[3 * e + 1], res_m[3 * e + 2], e}; }
    auto *ditems = (glmvq_decode_item *)dev(E * sizeof(glmvq_decode_item), items.data());
    auto *mid = (__nv_bfloat16 *)dev(4 * E * F * 2);
    auto *partial = (float *)dev(4 * E * H * 4);
    auto *out = (float *)dev(4 * H * 4);
    auto *pgate = (float *)dev(E * F * 4), *pup = (float *)dev(E * F * 4);
    uint32_t blocks = 0;
    if (glmvq_decode_prepare<13, 32, GLMVQ_DEC_PROOF, GLMVQ_GAIN_WEIGHT, glmvq_glm_form, true>(&blocks) != cudaSuccess ||
        glmvq_decode_prepare<13, 32, GLMVQ_DEC_PROOF, GLMVQ_GAIN_DOT, glmvq_glm_form, true>(&blocks) != cudaSuccess ||
        glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, glmvq_glm_form, true>(&blocks) != cudaSuccess) {
        check(false, "decode prepare (R13+8 shared memory)", 0);
        return 1;
    }
    std::vector<float> rgate, rup, rout, pg, pu, po;
    expert_ref(res_m, 0, rgate, rup, rout);
    expert_ref(plain_m, 0, pg, pu, po);
    std::vector<float> one_row;
    for (int mode = 0; mode < 2; mode++) {
        cudaError_t err = mode == 0
            ? glmvq_decode_launch<13, 32, GLMVQ_DEC_PROOF, GLMVQ_GAIN_WEIGHT, glmvq_glm_form, true>(ditems, dbook1, dx,
                mid, partial, out, dw, CLAMP, blocks, nullptr, pgate, pup, nullptr, GLMVQ_LAUNCH_PLAIN, dbook2)
            : glmvq_decode_launch<13, 32, GLMVQ_DEC_PROOF, GLMVQ_GAIN_DOT, glmvq_glm_form, true>(ditems, dbook1, dx,
                mid, partial, out, dw, CLAMP, blocks, nullptr, pgate, pup, nullptr, GLMVQ_LAUNCH_PLAIN, dbook2);
        if (err != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { check(false, "decode launch", 0); return 1; }
        // Items are in ascending expert order; the reduce weights by pair.
        std::vector<float> got = fetch(out, H), g = fetch(pgate, E * F);
        std::vector<float> ref(H);
        for (uint32_t h = 0; h < H; h++) { ref[h] = rout[h]; }
        const double gate_err = relrms(g, rgate), out_err = relrms(got, ref);
        const double plain = relrms(po, ref);
        char what[96];
        snprintf(what, sizeof(what), "decode %s gate vs restore", mode ? "GAIN_DOT" : "GAIN_WEIGHT");
        // BF16-rounded dots in a different order flip a few last bits (WEIGHT);
        // DOT skips the per-weight BF16 rounding (~2^-9 relative).
        check(gate_err < (mode ? 1e-2 : 3e-3), what, gate_err);
        snprintf(what, sizeof(what), "decode %s output vs restore (R13-only gap %.3g)", mode ? "GAIN_DOT" : "GAIN_WEIGHT", plain);
        check(out_err < (mode ? 1e-2 : 5e-3) && plain > 10 * out_err, what, out_err);
        if (mode == 1) { one_row = got; }
    }

    // (C) Verify rows 1-3 (MTP widths 2-4): row r equals that row's decode.
    {
        uint32_t vblocks = 0;
        if (glmvq_decode_prepare<13, 32, GLMVQ_DEC_NORMAL, GLMVQ_GAIN_DOT, glmvq_glm_form, true>(&vblocks) != cudaSuccess) {
            return 1;
        }
        for (uint32_t rows = 2; rows <= 4; rows++) {
            // Every row routes to the same eight experts (worst overlap).
            glmvq_verify_plan plan = {};
            std::vector<glmvq_verify_item> vitems(E);
            for (uint32_t e = 0; e < E; e++) {
                vitems[e].gate = res_m[3 * e];
                vitems[e].up = res_m[3 * e + 1];
                vitems[e].down = res_m[3 * e + 2];
                vitems[e].count = rows;
                for (uint32_t r = 0; r < rows; r++) { vitems[e].pair[r] = r * E + e; }
                for (uint32_t r = 0; r < rows; r++) { plan.order[r * E + e] = e; }
            }
            plan.items = E;
            uint32_t groups = 0, used = GLMVQ_VER_GROUP;
            for (uint32_t i = 0; i < E; i++) {
                if (used + rows > GLMVQ_VER_GROUP) { plan.group[groups++] = i; used = 0; }
                vitems[i].slot = used;
                used += rows;
            }
            plan.group[groups] = E;
            plan.groups = groups;
            auto *dv = (glmvq_verify_item *)dev(E * sizeof(glmvq_verify_item), vitems.data());
            auto *dp = (glmvq_verify_plan *)dev(sizeof(plan), &plan);
            if (glmvq_verify_launch<13, 32, GLMVQ_GAIN_DOT, glmvq_glm_form, true>(dv, dp, dbook1, dx, rows, mid,
                    partial, out, dw, CLAMP, vblocks, nullptr, nullptr, GLMVQ_LAUNCH_PLAIN, dbook2) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess) { check(false, "verify launch", 0); return 1; }
            std::vector<float> got = fetch(out, (uint64_t)rows * H);
            // Row 0 against the one-row decode: bit-identical per row.
            bool same = true;
            for (uint32_t h = 0; h < H; h++) { same = same && got[h] == one_row[h]; }
            char what[96];
            snprintf(what, sizeof(what), "verify %u rows: row 0 bit-identical to decode", rows);
            check(same, what, 0);
        }
    }

    // (D) FP8 prefill MMA (gate and down stages) against restore + GEMM.
    {
        int mblocks = 0;
        if (glmvq_mma8_prepare<256, 32, 1, GLMVQ_MMA_GATE, 13, GLMVQ_CLAMP_GATE, true>(&mblocks) != cudaSuccess ||
            glmvq_mma8_prepare<256, 32, 1, GLMVQ_MMA_DOWN, 13, GLMVQ_CLAMP_GATE, true>(&mblocks) != cudaSuccess) {
            check(false, "mma8 prepare (R13+8 shared memory)", 0);
            return 1;
        }
        printf("mma8 R13+8 shared memory %u B\n", glmvq_mma8_bytes<256, 32, 13, true>());
        std::vector<float> xt((uint64_t)T * H), xd((uint64_t)T * F);
        for (auto &v : xt) { v = bf16_round(normal(rng)); }
        for (auto &v : xd) { v = bf16_round(0.3f * normal(rng)); }
        std::vector<glmvq_source> sources(3);
        sources[0] = {nullptr, nullptr, res_m[0], F, H, GLMVQ_WEIGHT_VQ};
        sources[1] = {nullptr, nullptr, res_m[1], F, H, GLMVQ_WEIGHT_VQ};
        sources[2] = {nullptr, nullptr, res_m[2], H, F, GLMVQ_WEIGHT_VQ};
        auto *dsrc = (glmvq_source *)dev(3 * sizeof(glmvq_source), sources.data());
        glmvq_mma_item item = {{0, 1, 2}, 0, T};
        auto *ditem = (glmvq_mma_item *)dev(sizeof(item), &item);
        std::vector<uint32_t> map(T);
        for (uint32_t t = 0; t < T; t++) { map[t] = t; }
        auto *dmap = (uint32_t *)dev(T * 4, map.data());
        for (int stage = 0; stage < 2; stage++) {
            const uint32_t cols = stage ? F : H, rows = stage ? H : F;
            const std::vector<float> &in = stage ? xd : xt;
            auto *din = (float *)dev(in.size() * 4, in.data());
            auto *q8 = (uint8_t *)dev((uint64_t)T * cols);
            auto *qs = (float *)dev((uint64_t)T * cols / GLMVQ_MMA8_K * 4);
            const uint64_t qb = (uint64_t)T * cols / GLMVQ_MMA8_K;
            glmvq_mma8_quant<float><<<(uint32_t)((qb * 32 + 255) / 256), 256>>>(q8, qs, din, qb);
            auto *res_out = (__nv_bfloat16 *)dev((uint64_t)T * rows * 2);
            if (stage == 0) {
                glmvq_mma8_kernel<256, 32, 1, GLMVQ_MMA_GATE, 13, GLMVQ_CLAMP_GATE, true>
                    <<<mblocks, glmvq_mma8_threads<256, 1>(), glmvq_mma8_bytes<256, 32, 13, true>()>>>(
                    dsrc, ditem, 1, (const uint8_t *)dbook1, q8, qs, dmap, 1, res_out, nullptr, nullptr,
                    rows, cols, CLAMP, nullptr, nullptr, (const uint8_t *)dbook2);
            } else {
                item.matrix[2] = 2;
                glmvq_mma8_kernel<256, 32, 1, GLMVQ_MMA_DOWN, 13, GLMVQ_CLAMP_GATE, true>
                    <<<mblocks, glmvq_mma8_threads<256, 1>(), glmvq_mma8_bytes<256, 32, 13, true>()>>>(
                    dsrc, ditem, 1, (const uint8_t *)dbook1, q8, qs, dmap, 1, nullptr, nullptr, res_out,
                    rows, cols, CLAMP, nullptr, nullptr, (const uint8_t *)dbook2);
            }
            if (cudaDeviceSynchronize() != cudaSuccess) { check(false, "mma8 launch", 0); return 1; }
            std::vector<uint16_t> raw((uint64_t)T * rows);
            cudaMemcpy(raw.data(), res_out, raw.size() * 2, cudaMemcpyDeviceToHost);
            std::vector<float> got(raw.size());
            for (size_t i = 0; i < raw.size(); i++) { uint32_t u = (uint32_t)raw[i] << 16; memcpy(&got[i], &u, 4); }
            // Reference: restored BF16 weights times the (unquantized) inputs.
            auto *ref = (float *)dev((uint64_t)T * rows * 4);
            const glmvq_source &s = sources[stage ? 2 : 0];
            glmvq_gemm(handle, s, din, ref, T, GLMVQ_COMPUTE_BF16, scratch, scratch_bytes, nullptr, GLMVQ_RESTORE_SCALAR);
            glmvq_source plain = s;
            plain.vq.res = nullptr;
            plain.vq.res_book = nullptr;
            auto *pref = (float *)dev((uint64_t)T * rows * 4);
            glmvq_gemm(handle, plain, din, pref, T, GLMVQ_COMPUTE_BF16, scratch, scratch_bytes, nullptr, GLMVQ_RESTORE_SCALAR);
            const std::vector<float> r = fetch(ref, (uint64_t)T * rows), p = fetch(pref, (uint64_t)T * rows);
            const double err = relrms(got, r), gap = relrms(p, r);
            char what[96];
            snprintf(what, sizeof(what), "mma8 %s vs restore (E4M3 activations; R13-only gap %.3g)",
                stage ? "down" : "gate", gap);
            check(err < 4e-2 && gap > 5 * err, what, err);
        }
    }

    cublasDestroy(handle);
    for (void *p : g_alloc) { cudaFree(p); }
    printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail;
}
