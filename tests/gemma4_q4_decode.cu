/* Gemma Q4_K decode gate (cuda/gemma4_q4_decode.cuh).
 *
 * Synthetic Q4_K matrices at the artifact's shapes: q 4096x2816, k/v
 * 2048x2816, o 2816x4096, dense gate/up 2112x2816. The reference is the
 * shared dispatcher the per-expert loop uses today (glmvq_q4_decode with the
 * AUTO config: the scalar kernel under 4 MiB, the staged stream above it),
 * so the numbers are the parity of the Gemma pass against the path it
 * replaces. The GLU reference runs the same shared decode for gate and up
 * and the existing g4_gelu_kernel over the two outputs.
 *
 * build: make tests/gemma4_q4_decode
 * run:   ./tests/gemma4_q4_decode
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../ds4_gemma4_plan.h"
#include "../cuda/gemma4_q4_decode.cuh"
#include "../cuda/glm_vq.cuh"
#include "../cuda/glm_vq_q4.cuh"

#define CUDA_OK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e_)); std::exit(2); } } while(0)
static unsigned failures = 0;

struct rng {
    uint32_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
};

/* One Q4_K matrix: 144-byte blocks, 256 weights each, with finite f16
 * scales. The byte layout is cuda/q4_k.cuh's (GGML Q4_K). */
static uint8_t *make_q4(uint32_t rows, uint32_t cols, rng &r, uint8_t **device) {
    const uint64_t blocks = (uint64_t)rows * cols / DS4_Q4_K_BLOCK;
    std::vector<uint8_t> host(blocks * DS4_Q4_K_BYTES);
    for (uint64_t b = 0; b < blocks; b++) {
        uint8_t *block = host.data() + b * DS4_Q4_K_BYTES;
        const __half d = __float2half_rn(0.0005f + (r.next() % 1000u) * 1e-6f);
        const __half dm = __float2half_rn(0.0002f + (r.next() % 1000u) * 5e-7f);
        std::memcpy(block, &d, sizeof(d));
        std::memcpy(block + 2, &dm, sizeof(dm));
        for (uint32_t i = 4; i < 16; i++) { block[i] = (uint8_t)r.next(); }
        for (uint32_t i = 16; i < DS4_Q4_K_BYTES; i++) { block[i] = (uint8_t)r.next(); }
    }
    uint8_t *dev = nullptr;
    CUDA_OK(cudaMalloc(&dev, host.size()));
    CUDA_OK(cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice));
    *device = dev;
    return dev;
}

static void report(const char *name, const float *got, const float *want, uint32_t count,
        double rel_limit) {
    double num = 0.0, den = 0.0, mx = 0.0;
    uint32_t same = 0;
    for (uint32_t i = 0; i < count; i++) {
        const double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * want[i];
        if (std::fabs(d) > mx) { mx = std::fabs(d); }
        same += got[i] == want[i];
    }
    const double rel = std::sqrt(num / (den > 0.0 ? den : 1.0));
    std::printf("{\"test\":\"%s\",\"relL2\":%.9g,\"max_error\":%.9g,\"exact\":%u,"
        "\"count\":%u,\"limit\":%.3g}\n", name, rel, mx, same, count, rel_limit);
    if (!(rel <= rel_limit)) {
        std::fprintf(stderr, "%s: relL2 %.3e above %.3e\n", name, rel, rel_limit);
        failures++;
    }
}

int main(void) {
    rng r{0x51ed270bu};
    const uint32_t shapes[][2] = {
        {4096, 2816},   /* q */
        {2048, 2816},   /* k */
        {2048, 2816},   /* v */
        {2816, 4096},   /* o */
        {2112, 2816},   /* dense gate */
        {2112, 2816},   /* dense up */
    };
    const uint32_t count = sizeof(shapes) / sizeof(shapes[0]);
    std::vector<uint8_t *> weights(count);
    for (uint32_t c = 0; c < count; c++) {
        make_q4(shapes[c][0], shapes[c][1], r, &weights[c]);
    }
    const uint32_t cols = 2816;
    std::vector<float> input(cols);
    for (uint32_t i = 0; i < cols; i++) {
        input[i] = std::sin((i + 1) * 0.117f) + 0.3f * std::cos((i + 3) * 0.031f);
    }
    float *d_input = nullptr;
    CUDA_OK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    glmvq_q4_config config = {};
    CUDA_OK(glmvq_q4_prepare(&config, GLMVQ_Q4_AUTO));

    std::vector<float *> d_ref(count), d_got(count);
    for (uint32_t c = 0; c < count; c++) {
        CUDA_OK(cudaMalloc(&d_ref[c], (size_t)shapes[c][0] * sizeof(float)));
        CUDA_OK(cudaMalloc(&d_got[c], (size_t)shapes[c][0] * sizeof(float)));
        CUDA_OK(glmvq_q4_decode(weights[c], d_input, d_ref[c], shapes[c][0], shapes[c][1],
            1u, 0, &config));
        const uint64_t shared = (uint64_t)G4_Q4_WARPS * (shapes[c][1] / G4_Q4_BLOCK) *
            G4_Q4_BYTES;
        g4_q4_many many = {};
        many.weight[0] = weights[c];
        many.out[0] = d_got[c];
        many.rows[0] = shapes[c][0];
        for (uint32_t k = 1; k <= G4_Q4_MANY; k++) {
            many.first[k] = 1u + (shapes[c][0] - 1u) / G4_Q4_WARPS;
        }
        g4_q4_many_kernel<<<many.first[G4_Q4_MANY], G4_Q4_WARPS * 32u, (size_t)shared>>>(
            many, d_input, shapes[c][1]);
        CUDA_OK(cudaGetLastError());
    }
    CUDA_OK(cudaDeviceSynchronize());

    /* A: each matrix alone against the shared dispatcher. */
    const char *names[count] = {"q", "k", "v", "o", "dense_gate", "dense_up"};
    for (uint32_t c = 0; c < count; c++) {
        std::vector<float> got(shapes[c][0]), want(shapes[c][0]);
        CUDA_OK(cudaMemcpy(got.data(), d_got[c], got.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(want.data(), d_ref[c], want.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        report(names[c], got.data(), want.data(), shapes[c][0], 1e-5);
    }

    /* B: the q/k/v group in one launch, the way the graph asks for it. */
    {
        g4_q4_many many = {};
        uint32_t groups = 0;
        for (uint32_t c = 0; c < 3; c++) {
            many.weight[c] = weights[c];
            many.out[c] = d_got[c];
            many.rows[c] = shapes[c][0];
            many.first[c] = groups;
            groups += 1u + (shapes[c][0] - 1u) / G4_Q4_WARPS;
        }
        many.first[3] = groups;
        many.first[4] = groups;
        many.first[G4_Q4_MANY] = groups;
        const uint64_t shared = (uint64_t)G4_Q4_WARPS * (cols / G4_Q4_BLOCK) * G4_Q4_BYTES;
        g4_q4_many_kernel<<<groups, G4_Q4_WARPS * 32u, (size_t)shared>>>(many, d_input, cols);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        for (uint32_t c = 0; c < 3; c++) {
            std::vector<float> got(shapes[c][0]), want(shapes[c][0]);
            CUDA_OK(cudaMemcpy(got.data(), d_got[c], got.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(want.data(), d_ref[c], want.size() * sizeof(float),
                cudaMemcpyDeviceToHost));
            char name[32];
            std::snprintf(name, sizeof(name), "group_%s", names[c]);
            report(name, got.data(), want.data(), shapes[c][0], 1e-5);
        }
    }

    /* C: the fused gate/up pair against the shared decode plus g4_gelu. */
    {
        const uint32_t rows = shapes[4][0];
        float *d_glu = nullptr;
        CUDA_OK(cudaMalloc(&d_glu, (size_t)rows * sizeof(float)));
        const uint64_t shared = 2ull * G4_Q4_WARPS * (cols / G4_Q4_BLOCK) * G4_Q4_BYTES;
        g4_q4_glu_kernel<<<1u + (rows - 1u) / G4_Q4_WARPS, G4_Q4_WARPS * 32u, (size_t)shared>>>(
            weights[4], weights[5], d_glu, d_input, cols, rows);
        CUDA_OK(cudaGetLastError());
        g4_gelu_kernel<<<(rows + 255u) / 256u, 256u>>>(d_ref[4], d_ref[4], d_ref[5], rows);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> got(rows), want(rows);
        CUDA_OK(cudaMemcpy(got.data(), d_glu, got.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(want.data(), d_ref[4], want.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        report("glu", got.data(), want.data(), rows, 1e-5);
        CUDA_OK(cudaFree(d_glu));
    }

    /* Indicative timing: the shared path launches one kernel per matrix; the
     * Gemma pass one for q/k/v, one for o and one for the dense pair. */
    {
        const int iters = 20;
        cudaEvent_t a, b;
        CUDA_OK(cudaEventCreate(&a));
        CUDA_OK(cudaEventCreate(&b));
        CUDA_OK(cudaEventRecord(a));
        for (int it = 0; it < iters; it++) {
            for (uint32_t c = 0; c < count; c++) {
                glmvq_q4_decode(weights[c], d_input, d_ref[c], shapes[c][0], shapes[c][1],
                    1u, 0, &config);
            }
        }
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float ref_ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ref_ms, a, b));
        CUDA_OK(cudaEventRecord(a));
        for (int it = 0; it < iters; it++) {
            g4_q4_many many = {};
            uint32_t groups = 0;
            for (uint32_t c = 0; c < 3; c++) {
                many.weight[c] = weights[c];
                many.out[c] = d_got[c];
                many.rows[c] = shapes[c][0];
                many.first[c] = groups;
                groups += 1u + (shapes[c][0] - 1u) / G4_Q4_WARPS;
            }
            many.first[3] = groups;
            many.first[4] = groups;
            many.first[G4_Q4_MANY] = groups;
            const uint64_t shared = (uint64_t)G4_Q4_WARPS * (cols / G4_Q4_BLOCK) * G4_Q4_BYTES;
            g4_q4_many_kernel<<<groups, G4_Q4_WARPS * 32u, (size_t)shared>>>(many, d_input, cols);
            g4_q4_many one = {};
            one.weight[0] = weights[3];
            one.out[0] = d_got[3];
            one.rows[0] = shapes[3][0];
            for (uint32_t k = 1; k <= G4_Q4_MANY; k++) {
                one.first[k] = 1u + (shapes[3][0] - 1u) / G4_Q4_WARPS;
            }
            const uint64_t shared_o = (uint64_t)G4_Q4_WARPS * (shapes[3][1] / G4_Q4_BLOCK) *
                G4_Q4_BYTES;
            g4_q4_many_kernel<<<one.first[G4_Q4_MANY], G4_Q4_WARPS * 32u, (size_t)shared_o>>>(
                one, d_input, shapes[3][1]);
            const uint64_t shared_glu = 2ull * G4_Q4_WARPS * (cols / G4_Q4_BLOCK) * G4_Q4_BYTES;
            g4_q4_glu_kernel<<<1u + (shapes[4][0] - 1u) / G4_Q4_WARPS, G4_Q4_WARPS * 32u,
                (size_t)shared_glu>>>(weights[4], weights[5], d_got[4], d_input, cols,
                shapes[4][0]);
        }
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float gemma_ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&gemma_ms, a, b));
        std::printf("{\"test\":\"timing\",\"shared_ms\":%.4f,\"gemma_ms\":%.4f,"
            "\"speedup\":%.3f}\n", ref_ms / iters, gemma_ms / iters, ref_ms / gemma_ms);
        CUDA_OK(cudaEventDestroy(a));
        CUDA_OK(cudaEventDestroy(b));
    }

    std::printf("{\"failures\":%u}\n", failures);
    return failures ? 1 : 0;
}
