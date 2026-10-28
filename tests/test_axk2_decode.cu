// V2 gate/up scheduling: real geometry, independent sampled dots, and canaries.
#include "cuda/glm_vq_decode.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

enum { H = 7168, F = 2048, E = 8 };
static void check(cudaError_t err) {
    if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); std::exit(1); }
}
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float code(uint8_t x) {
    const uint32_t exponent = (x >> 3) & 15, fraction = x & 7;
    const float value = exponent ? std::ldexp(1.0f + fraction / 8.0f, (int)exponent - 7)
                                 : std::ldexp((float)fraction, -9);
    return x & 128 ? -value : value;
}
struct packed {
    std::vector<uint8_t> main, plane;
    std::vector<__half> gain;
};

template <uint32_t BITS, glmvq_decode_gain GAIN>
static bool run() {
    std::mt19937 rng(821 + BITS);
    std::vector<void *> allocations;
    auto upload = [&](const void *src, size_t bytes) {
        void *p = nullptr; check(cudaMalloc(&p, bytes)); allocations.push_back(p);
        if (src) { check(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice)); }
        return p;
    };
    std::vector<uint8_t> books[2];
    const uint4 *device_books[2];
    for (uint32_t j = 0; j < 2; j++) {
        books[j].resize((1u << BITS) * 8);
        for (auto &v : books[j]) { v = (uint8_t)rng(); if ((v & 127) == 127) { v ^= 1; } }
        device_books[j] = (const uint4 *)upload(books[j].data(), books[j].size());
    }
    packed matrices[E * 2];
    glmvq_decode_item host[E] = {};
    for (uint32_t e = 0; e < E; e++) {
        for (uint32_t j = 0; j < 2; j++) {
            auto &m = matrices[2 * e + j];
            m.main.resize(F * H / 8); m.plane.resize(F * H / 64 * (BITS - 8)); m.gain.resize(F);
            for (auto &v : m.main) { v = (uint8_t)rng(); }
            for (auto &v : m.plane) { v = (uint8_t)rng(); }
            for (auto &v : m.gain) { v = __float2half_rn(((int)(rng() % 257) - 128) / 2048.0f); }
            auto &dst = j ? host[e].up : host[e].gate;
            dst = {(const uint8_t *)upload(m.main.data(), m.main.size()),
                (const uint8_t *)upload(m.plane.data(), m.plane.size()), device_books[j],
                (const __half *)upload(m.gain.data(), F * 2), F, H, GLMVQ_E4M3, nullptr, nullptr};
        }
        host[e].pair = E - 1 - e;
    }
    std::vector<float> input(H);
    for (auto &v : input) { v = bf(((int)(rng() % 1025) - 512) / 4096.0f); }
    auto *x = (const float *)upload(input.data(), H * 4);
    auto *items = (const glmvq_decode_item *)upload(host, sizeof(host));
    const size_t sizes[3] = {E * F * 2, E * F * 4, E * F * 4};
    uint8_t *guarded[3];
    for (uint32_t j = 0; j < 3; j++) { guarded[j] = (uint8_t *)upload(nullptr, sizes[j] + 32); }
    std::vector<uint8_t> reference[3];
    bool ok = true;
    for (uint32_t mode = 0; mode < 2; mode++) {
        for (uint32_t j = 0; j < 3; j++) { check(cudaMemset(guarded[j], 0xa5, sizes[j] + 32)); }
        const auto kernel = mode
            ? glmvq_decode_gu<BITS, 32, GLMVQ_DEC_PROOF, GAIN, F, GLMVQ_CLAMP_GATE, false, H, GLMVQ_DEC_COMPACT>
            : glmvq_decode_gu<BITS, 32, GLMVQ_DEC_PROOF, GAIN, F, GLMVQ_CLAMP_GATE, false, H>;
        const uint32_t shared = 2 * (1u << BITS) * 8 + H * 2;
        kernel<<<48, 1024, shared>>>(items, device_books[0], x, (__nv_bfloat16 *)(guarded[0] + 16),
            INFINITY, (float *)(guarded[1] + 16), (float *)(guarded[2] + 16), device_books[1]);
        std::vector<float> projections[2];
        for (uint32_t j = 0; j < 3; j++) {
            std::vector<uint8_t> result(sizes[j] + 32);
            check(cudaMemcpy(result.data(), guarded[j], result.size(), cudaMemcpyDeviceToHost));
            for (uint32_t i = 0; i < 16; i++) {
                ok = ok && result[i] == 0xa5 && result[result.size() - 16 + i] == 0xa5;
            }
            if (!mode) { reference[j] = result; } else { ok = ok && result == reference[j]; }
            if (j) { projections[j - 1].resize(E * F); memcpy(projections[j - 1].data(), result.data() + 16, sizes[j]); }
            else {
                for (uint32_t i = 0; i < E * F; i++) {
                    __nv_bfloat16 v; memcpy(&v, result.data() + 16 + i * 2, 2);
                    ok = ok && std::isfinite(__bfloat162float(v));
                }
            }
        }
        for (uint32_t e = 0; e < E; e++) {
            for (uint32_t j = 0; j < 2; j++) {
                const auto &m = matrices[2 * e + j];
                for (uint32_t row : {0u, 1u, 1023u, 2047u}) {
                    const float gain = __half2float(m.gain[row]);
                    float lanes[32] = {};
                    for (uint32_t step = 0; step < H / 256; step++) {
                        for (uint32_t lane = 0; lane < 32; lane++) {
                            const uint32_t at = row * (H / 8) + step * 32 + lane, pack = 8 / (BITS - 8);
                            const uint32_t index = m.main[at] | ((m.plane[at / pack] >> ((at % pack) * (BITS - 8))) & ((1u << (BITS - 8)) - 1)) << 8;
                            for (uint32_t d = 0; d < 8; d++) {
                                float w = code(books[j][index * 8 + d]);
                                if constexpr (GAIN == GLMVQ_GAIN_WEIGHT) { w = bf(w * gain); }
                                lanes[lane] = std::fma(w, input[step * 256 + lane * 8 + d], lanes[lane]);
                            }
                        }
                    }
                    for (uint32_t stride = 16; stride; stride >>= 1) {
                        float next[32];
                        for (uint32_t lane = 0; lane < 32; lane++) { next[lane] = lanes[lane] + lanes[lane ^ stride]; }
                        memcpy(lanes, next, sizeof(lanes));
                    }
                    const float want = bf(GAIN == GLMVQ_GAIN_DOT ? lanes[0] * gain : lanes[0]);
                    ok = ok && projections[j][e * F + row] == want && std::isfinite(want);
                }
            }
        }
    }
    for (void *p : allocations) { check(cudaFree(p)); }
    printf("bits=%u gain=%u CPU dots/full parity/canaries %s\n", BITS, (uint32_t)GAIN, ok ? "PASS" : "FAIL");
    return ok;
}
int main() {
    return run<9, GLMVQ_GAIN_DOT>() && run<10, GLMVQ_GAIN_DOT>() &&
        run<9, GLMVQ_GAIN_WEIGHT>() && run<10, GLMVQ_GAIN_WEIGHT>() ? 0 : 1;
}
