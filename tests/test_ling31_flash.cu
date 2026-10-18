/* Ling-3.1 tensor-core latent attention against an FP64 reference of the
 * absorbed MLA: s = (q_abs·c + q_pe·k_pe) / sqrt(192) over keys 0..pos0+row,
 * softmax, O = Σ p c. Q and P round to BF16 in the kernel, so the gate is a
 * relative error bound, at single- and multi-split, partial-tile, decode and
 * verify shapes. FP8 latent rows must store the host-computed E4M3 values,
 * dequantize to the simulation's BF16 rows and attend bit for bit as the
 * BF16 kernel does on them. */
#include "cuda/l31_flash.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool check(cudaError_t err, const char *what) {
    if (err == cudaSuccess) { return true; }
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

static bool run(uint32_t rows, uint32_t pos0, std::mt19937 &rng) {
    constexpr uint32_t H = GLMVQ_FA_HEADS, L = GLMVQ_FA_LATENT, R = L31_FA_ROPE, K = L31_FA_KEY;
    const uint32_t keys = pos0 + rows;
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> q((uint64_t)rows * H * K), q_abs((uint64_t)rows * H * L);
    std::vector<__nv_bfloat16> latent((uint64_t)keys * L), k_pe((uint64_t)keys * R);
    for (auto &v : q) { v = unit(rng); }
    for (auto &v : q_abs) { v = 0.5f * unit(rng); }
    for (auto &v : latent) { v = __float2bfloat16_rn(0.5f * unit(rng)); }
    for (auto &v : k_pe) { v = __float2bfloat16_rn(unit(rng)); }
    const float scale = 1.0f / sqrtf((float)K);

    float *d_q = nullptr, *d_abs = nullptr, *d_out = nullptr;
    __nv_bfloat16 *d_lat = nullptr, *d_pe = nullptr;
    void *d_scratch = nullptr;
    const uint64_t out_n = (uint64_t)rows * H * L;
    bool ok = check(cudaMalloc(&d_q, q.size() * 4), "alloc") && check(cudaMalloc(&d_abs, q_abs.size() * 4), "alloc") &&
        check(cudaMalloc(&d_out, out_n * 4), "alloc") && check(cudaMalloc(&d_lat, latent.size() * 2), "alloc") &&
        check(cudaMalloc(&d_pe, k_pe.size() * 2), "alloc") && check(cudaMalloc(&d_scratch, l31_fa_scratch_bytes()), "alloc") &&
        check(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_abs, q_abs.data(), q_abs.size() * 4, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_lat, latent.data(), latent.size() * 2, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_pe, k_pe.data(), k_pe.size() * 2, cudaMemcpyHostToDevice), "copy") &&
        check(l31_fa_launch(d_out, d_scratch, d_q, d_abs, d_lat, nullptr, d_pe, rows, pos0, scale, GLMVQ_LAUNCH_PLAIN, 0), "launch") &&
        check(cudaDeviceSynchronize(), "kernel");
    std::vector<float> out(out_n);
    ok = ok && check(cudaMemcpy(out.data(), d_out, out_n * 4, cudaMemcpyDeviceToHost), "read");

    double worst = 0.0;
    for (uint32_t r = 0; ok && r < rows; r++) {
        const uint32_t visible = pos0 + r + 1;
        for (uint32_t h = 0; h < H; h++) {
            const float *qa = &q_abs[((uint64_t)r * H + h) * L], *qp = &q[((uint64_t)r * H + h) * K + L31_FA_NOPE];
            std::vector<double> s(visible);
            double peak = -INFINITY;
            for (uint32_t k = 0; k < visible; k++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < L; d++) { dot += (double)qa[d] * __bfloat162float(latent[(uint64_t)k * L + d]); }
                for (uint32_t d = 0; d < R; d++) { dot += (double)qp[d] * __bfloat162float(k_pe[(uint64_t)k * R + d]); }
                s[k] = dot * scale;
                peak = std::max(peak, s[k]);
            }
            double sum = 0.0;
            for (auto &v : s) { v = std::exp(v - peak); sum += v; }
            double norm = 0.0, err = 0.0;
            for (uint32_t d = 0; d < L; d++) {
                double ref = 0.0;
                for (uint32_t k = 0; k < visible; k++) { ref += s[k] * __bfloat162float(latent[(uint64_t)k * L + d]); }
                ref /= sum;
                const double got = out[((uint64_t)r * H + h) * L + d];
                norm += ref * ref;
                err += (got - ref) * (got - ref);
            }
            worst = std::max(worst, std::sqrt(err / std::max(norm, 1e-30)));
        }
    }
    // BF16 Q and P carry ~2^-9 relative rounding each.
    const bool pass = ok && worst < 1.0e-2;
    printf("rows=%u pos0=%u keys=%u rel_rms %.3e %s\n", rows, pos0, keys, worst, pass ? "ok" : "FAIL");
    cudaFree(d_q); cudaFree(d_abs); cudaFree(d_out); cudaFree(d_lat); cudaFree(d_pe); cudaFree(d_scratch);
    return pass;
}

// Host E4M3 value of one stored row element, exactly as the store kernel
// derives it (row max into (224, 448], one power-of-two scale).
static float fp8_value(const float *row, uint32_t d) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < GLMVQ_FA_LATENT; i++) { amax = std::max(amax, std::fabs(row[i])); }
    if (amax == 0.0f) { return 0.0f; }
    int k = 0;
    std::frexp(amax, &k);
    const int e = amax > std::ldexp(448.0f, k - 9) ? k - 8 : k - 9;
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(std::ldexp(row[d], -e), __NV_SATFINITE, __NV_E4M3);
    return std::ldexp(__half2float(__half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3))), e);
}

// l31_fa_launch's grid and splits for either kernel at any row count (the
// BF16 launcher stops at L31_FA_ROWS; a CTA's work does not depend on it).
static cudaError_t launch_rows(bool fp8, float *out, void *scratch, const float *q, const float *q_abs,
        const void *latent, const int8_t *exps, const __nv_bfloat16 *k_pe, uint32_t rows, uint32_t pos0, float scale) {
    const uint32_t tiles = (pos0 + rows + GLMVQ_FA_KEYS - 1) / GLMVQ_FA_KEYS;
    const uint32_t splits = std::max(1u, std::min(tiles, (uint32_t)L31_FA_SPLITS / rows));
    const uint32_t per = (tiles + splits - 1) / splits;
    float *partial = (float *)scratch;
    float *stats = partial + (uint64_t)rows * splits * GLMVQ_FA_HEADS * GLMVQ_FA_LATENT;
    const dim3 grid(rows, splits);
    if (fp8) {
        l31_fa_kernel<true><<<grid, GLMVQ_FA_THREADS, L31_FA8_SMEM>>>(out, splits > 1 ? partial : nullptr, stats, q, q_abs,
            latent, exps, k_pe, pos0, splits > 1 ? per : UINT32_MAX, scale);
    } else {
        l31_fa_kernel<false><<<grid, GLMVQ_FA_THREADS, L31_FA_SMEM>>>(out, splits > 1 ? partial : nullptr, stats, q, q_abs,
            latent, nullptr, k_pe, pos0, splits > 1 ? per : UINT32_MAX, scale);
    }
    if (splits > 1) {
        l31_fa_combine<<<dim3(rows, GLMVQ_FA_HEADS), GLMVQ_FA_LATENT / 2>>>(out, partial, stats, splits);
    }
    return cudaGetLastError();
}

/* FP8 rows against their simulation: rows span 2^-12..2^12 in scale, one is
 * zero and one has a single outlier; the cache has an odd capacity so the
 * exponent region is padded. */
static bool run_fp8(uint32_t rows, uint32_t pos0, std::mt19937 &rng) {
    constexpr uint32_t H = GLMVQ_FA_HEADS, L = GLMVQ_FA_LATENT, R = L31_FA_ROPE, K = L31_FA_KEY;
    const uint32_t keys = pos0 + rows, cap = keys + 13;
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::uniform_int_distribution<int> octave(-12, 12);
    std::vector<float> q((uint64_t)rows * H * K), q_abs((uint64_t)rows * H * L);
    std::vector<float> norm((uint64_t)keys * L), raw((uint64_t)keys * (L + R));
    for (auto &v : q) { v = unit(rng); }
    for (auto &v : q_abs) { v = 0.5f * unit(rng); }
    for (uint32_t k = 0; k < keys; k++) {
        const float s = std::ldexp(1.0f, octave(rng));
        for (uint32_t d = 0; d < L; d++) { norm[(uint64_t)k * L + d] = k == 1 ? 0.0f : s * unit(rng); }
        for (uint32_t d = 0; d < L + R; d++) { raw[(uint64_t)k * (L + R) + d] = unit(rng); }
    }
    if (keys > 2) { norm[2 * L + 7] = 3.0e4f; }
    const float scale = 1.0f / sqrtf((float)K);

    float *d_q = nullptr, *d_abs = nullptr, *d_out[2] = {}, *d_norm = nullptr, *d_raw = nullptr;
    __nv_bfloat16 *d_sim = nullptr, *d_pe = nullptr, *d_back = nullptr;
    uint8_t *d_fp8 = nullptr;
    void *d_scratch = nullptr;
    const uint64_t out_n = (uint64_t)rows * H * L, fp8_bytes = ds4_l31_latent_bytes(cap, DS4_L31_LATENT_FP8);
    bool ok = check(cudaMalloc(&d_q, q.size() * 4), "alloc") && check(cudaMalloc(&d_abs, q_abs.size() * 4), "alloc") &&
        check(cudaMalloc(&d_out[0], out_n * 4), "alloc") && check(cudaMalloc(&d_out[1], out_n * 4), "alloc") &&
        check(cudaMalloc(&d_norm, norm.size() * 4), "alloc") && check(cudaMalloc(&d_raw, raw.size() * 4), "alloc") &&
        check(cudaMalloc(&d_sim, (uint64_t)cap * L * 2), "alloc") && check(cudaMalloc(&d_back, (uint64_t)keys * L * 2), "alloc") &&
        check(cudaMalloc(&d_pe, (uint64_t)cap * R * 2), "alloc") && check(cudaMalloc(&d_fp8, fp8_bytes), "alloc") &&
        check(cudaMalloc(&d_scratch, l31_fa_scratch_bytes()), "alloc") &&
        check(cudaMemset(d_fp8, 0, fp8_bytes), "zero") &&
        check(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_abs, q_abs.data(), q_abs.size() * 4, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_norm, norm.data(), norm.size() * 4, cudaMemcpyHostToDevice), "copy") &&
        check(cudaMemcpy(d_raw, raw.data(), raw.size() * 4, cudaMemcpyHostToDevice), "copy");
    if (ok) {
        l31_store_latent<DS4_L31_LATENT_SIM><<<keys, L31_LATENT_THREADS>>>(d_sim, d_pe, d_norm, d_raw, 0, cap);
        l31_store_latent<DS4_L31_LATENT_FP8><<<keys, L31_LATENT_THREADS>>>(d_fp8, d_pe, d_norm, d_raw, 0, cap);
        l31_latent_dequant<<<keys, L / 16>>>(d_back, d_fp8, 0, cap);
        const int8_t *exps = (const int8_t *)(d_fp8 + (uint64_t)cap * L);
        ok = check(cudaGetLastError(), "store") &&
            check(launch_rows(false, d_out[0], d_scratch, d_q, d_abs, d_sim, nullptr, d_pe, rows, pos0, scale), "bf16") &&
            check(cudaDeviceSynchronize(), "bf16 kernel") &&
            check(l31_fa_launch(d_out[1], d_scratch, d_q, d_abs, d_fp8, exps, d_pe, rows, pos0, scale, GLMVQ_LAUNCH_PLAIN, 0), "fp8") &&
            check(cudaDeviceSynchronize(), "kernel");
    }
    std::vector<__nv_bfloat16> sim((uint64_t)keys * L), back((uint64_t)keys * L);
    std::vector<float> out[2] = {std::vector<float>(out_n), std::vector<float>(out_n)};
    ok = ok && check(cudaMemcpy(sim.data(), d_sim, sim.size() * 2, cudaMemcpyDeviceToHost), "read") &&
        check(cudaMemcpy(back.data(), d_back, back.size() * 2, cudaMemcpyDeviceToHost), "read") &&
        check(cudaMemcpy(out[0].data(), d_out[0], out_n * 4, cudaMemcpyDeviceToHost), "read") &&
        check(cudaMemcpy(out[1].data(), d_out[1], out_n * 4, cudaMemcpyDeviceToHost), "read");
    uint64_t host_mismatch = 0;
    double worst = 0.0;
    for (uint32_t k = 0; ok && k < keys; k++) {
        const float *row = &norm[(uint64_t)k * L];
        float amax = 0.0f;
        for (uint32_t d = 0; d < L; d++) { amax = std::max(amax, std::fabs(row[d])); }
        for (uint32_t d = 0; d < L; d++) {
            const float v = fp8_value(row, d);
            host_mismatch += __bfloat162float(sim[(uint64_t)k * L + d]) != v;
            // Normal E4M3 (scaled value at least 2^-6) keeps three mantissa
            // bits: half an ulp is at most 2^-4 of the value.
            int k = 0;
            std::frexp(amax, &k);
            const int e = amax > std::ldexp(448.0f, k - 9) ? k - 8 : k - 9;
            if (amax > 0.0f && std::ldexp(std::fabs(row[d]), -e) >= 0.015625f) {
                worst = std::max(worst, (double)std::fabs(v - row[d]) / std::fabs(row[d]));
            }
        }
    }
    const bool rows_same = ok && !memcmp(sim.data(), back.data(), sim.size() * 2);
    const bool attn_same = ok && !memcmp(out[0].data(), out[1].data(), out_n * 4);
    const bool pass = ok && !host_mismatch && worst <= 0.0625 && rows_same && attn_same;
    printf("fp8 rows=%u pos0=%u keys=%u host mismatches %llu max rel %.4f rows %s attention %s %s\n", rows, pos0, keys,
        (unsigned long long)host_mismatch, worst, rows_same ? "same" : "DIFFER", attn_same ? "bit-identical" : "DIFFER",
        pass ? "ok" : "FAIL");
    cudaFree(d_q); cudaFree(d_abs); cudaFree(d_out[0]); cudaFree(d_out[1]); cudaFree(d_norm); cudaFree(d_raw);
    cudaFree(d_sim); cudaFree(d_back); cudaFree(d_pe); cudaFree(d_fp8); cudaFree(d_scratch);
    return pass;
}

// `bench`: decode-row time per call against the bytes of every cached key.
static void bench(uint32_t rows, uint32_t pos0, bool fp8) {
    constexpr uint32_t H = GLMVQ_FA_HEADS, L = GLMVQ_FA_LATENT, R = L31_FA_ROPE, K = L31_FA_KEY;
    const uint64_t keys = pos0 + rows, lat = fp8 ? ds4_l31_latent_bytes(keys, DS4_L31_LATENT_FP8) : keys * L * 2;
    float *d_q = nullptr, *d_abs = nullptr, *d_out = nullptr;
    uint8_t *d_lat = nullptr;
    __nv_bfloat16 *d_pe = nullptr;
    void *d_scratch = nullptr;
    cudaMalloc(&d_q, (uint64_t)rows * H * K * 4); cudaMalloc(&d_abs, (uint64_t)rows * H * L * 4);
    cudaMalloc(&d_out, (uint64_t)rows * H * L * 4); cudaMalloc(&d_lat, lat);
    cudaMalloc(&d_pe, keys * R * 2); cudaMalloc(&d_scratch, l31_fa_scratch_bytes());
    cudaMemset(d_q, 0, (uint64_t)rows * H * K * 4); cudaMemset(d_abs, 0, (uint64_t)rows * H * L * 4);
    cudaMemset(d_lat, 0, lat); cudaMemset(d_pe, 0, keys * R * 2);
    const int8_t *exps = fp8 ? (const int8_t *)(d_lat + keys * L) : nullptr;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const float scale = 1.0f / sqrtf((float)K);
    for (int i = 0; i < 3; i++) { l31_fa_launch(d_out, d_scratch, d_q, d_abs, d_lat, exps, d_pe, rows, pos0, scale, GLMVQ_LAUNCH_PLAIN, 0); }
    cudaEventRecord(a);
    const int iters = 20;
    for (int i = 0; i < iters; i++) { l31_fa_launch(d_out, d_scratch, d_q, d_abs, d_lat, exps, d_pe, rows, pos0, scale, GLMVQ_LAUNCH_PLAIN, 0); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    const double us = 1000.0 * ms / iters, gb = (double)(lat + keys * R * 2) / 1e9;
    printf("bench %s rows=%u keys=%llu %.1f us %.1f GB/s\n", fp8 ? "fp8" : "bf16", rows, (unsigned long long)keys, us,
        gb / (us * 1e-6));
    cudaFree(d_q); cudaFree(d_abs); cudaFree(d_out); cudaFree(d_lat); cudaFree(d_pe); cudaFree(d_scratch);
}

int main(int argc, char **argv) {
    if (!check(cudaFuncSetAttribute(l31_fa_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L31_FA_SMEM),
            "shared memory") ||
        !check(cudaFuncSetAttribute(l31_fa_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L31_FA8_SMEM),
            "shared memory")) { return 1; }
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        for (bool fp8 : {false, true}) {
            for (uint32_t pos : {8191u, 32767u, 65535u, 131071u, 262143u}) { bench(1, pos, fp8); }
            bench(3, 32765u, fp8);
        }
        for (uint32_t rows : {8u, 32u, 63u}) { bench(rows, 65536u, true); }
        return 0;
    }
    std::mt19937 rng(31);
    bool ok = true;
    const uint32_t cases[][2] = {{1, 0}, {1, 30}, {1, 31}, {1, 1000}, {1, 4099}, {3, 4097}, {4, 70}, {2, 9000}};
    for (const auto &c : cases) { ok = run(c[0], c[1], rng) && ok; }
    for (const auto &c : cases) { ok = run_fp8(c[0], c[1], rng) && ok; }
    ok = run_fp8(37, 200, rng) && run_fp8(63, 3000, rng) && ok;
    return ok ? 0 : 1;
}
