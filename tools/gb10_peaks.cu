/* GB10 peak probes for docs/gb10-hardware.md: mma.sync throughput per input
 * type (eight independent accumulators per warp, 32 warps per SM), the E4M3
 * rate of one warp per scheduler against its independent accumulator chains
 * (a kernel with one MMA warp per SMSP), streaming DRAM reads, a device copy
 * and L2 reads. Run with nothing else on the GPU:
 *
 *   make tools/gb10_peaks && ./tools/gb10_peaks
 *
 * Each line is the median of five timed launches after a warm-up. */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

enum { ILP = 8, ITERS = 4096 };

enum mma_kind { BF16_F32, F16_F32, F16_F16, TF32_F32, E4M3_F32, S8_S32, E2M1_NVF4 };

template <mma_kind K>
__device__ __forceinline__ void mma_once(uint32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    if constexpr (K == BF16_F32) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (K == F16_F32) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (K == F16_F16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
            : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (K == TF32_F32) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (K == E4M3_F32) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (K == S8_S32) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
#if defined(DS4_CUDA_HAVE_MXF4)
        const uint32_t scale = 0x38383838u;   // ue4m3 1.0
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(scale), "r"(scale));
#endif
    }
}

// One warp per scheduler (128 threads per SM), CHAINS dependent E4M3 chains.
template <int CHAINS>
__global__ void __launch_bounds__(128) chain_probe(uint32_t *out, uint32_t seed, int iters) {
    uint32_t a[4], b[2], d[CHAINS][4];
    for (int i = 0; i < 4; i++) { a[i] = (seed + threadIdx.x * 4 + i) & 0x3c3c3c3cu; }
    for (int i = 0; i < 2; i++) { b[i] = (seed * 3 + threadIdx.x + i) & 0x3c3c3c3cu; }
    for (int j = 0; j < CHAINS; j++) { d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0u; }
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int j = 0; j < CHAINS; j++) { mma_once<E4M3_F32>(d[j], a, b); }
    }
    uint32_t sum = 0;
    for (int j = 0; j < CHAINS; j++) { sum ^= d[j][0] ^ d[j][3]; }
    if (sum == seed) { out[blockIdx.x * blockDim.x + threadIdx.x] = sum; }
}

template <mma_kind K>
__global__ void __launch_bounds__(256) mma_probe(uint32_t *out, uint32_t seed) {
    uint32_t a[4], b[2], d[ILP][4];
    for (int i = 0; i < 4; i++) { a[i] = (seed + threadIdx.x * 4 + i) & 0x3c3c3c3cu; }
    for (int i = 0; i < 2; i++) { b[i] = (seed * 3 + threadIdx.x + i) & 0x3c3c3c3cu; }
    for (int j = 0; j < ILP; j++) { d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0u; }
    for (int it = 0; it < ITERS; it++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) { mma_once<K>(d[j], a, b); }
    }
    uint32_t sum = 0;
    for (int j = 0; j < ILP; j++) { sum ^= d[j][0] ^ d[j][1] ^ d[j][2] ^ d[j][3]; }
    if (sum == seed) { out[blockIdx.x * blockDim.x + threadIdx.x] = sum; }
}

// Four independent 16-byte loads in flight per thread.
__global__ void read_probe(const float4 *p, uint64_t n, float *out) {
    float acc = 0.0f;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < n; i += 4 * stride) {
        const float4 v0 = __ldcs(p + i), v1 = __ldcs(p + i + stride), v2 = __ldcs(p + i + 2 * stride),
            v3 = __ldcs(p + i + 3 * stride);
        acc += v0.x + v1.y + v2.z + v3.w;
    }
    if (acc == 12345.0f) { out[0] = acc; }
}

__global__ void l2_probe(const float4 *p, uint64_t n, int passes, float *out) {
    float acc = 0.0f;
    for (int pass = 0; pass < passes; pass++) {
        for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
            const float4 v = __ldcg(p + i);
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == 12345.0f) { out[0] = acc; }
}

template <typename F>
static float median_ms(F launch) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch();
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < 5; i++) {
        CK(cudaEventRecord(a));
        launch();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float t = 0;
        CK(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    return ms[2];
}

template <int CHAINS>
static void run_chains(int sms, uint32_t *out) {
    const int iters = 131072 / CHAINS;
    const float ms = median_ms([&] { chain_probe<CHAINS><<<sms, 128>>>(out, 7u, iters); CK(cudaGetLastError()); });
    const double mmas = (double)iters * CHAINS;   // per warp = per scheduler
    printf("e4m3, 1 warp/SMSP, %2d chains   %6.1f ns per MMA %8.1f TFLOPS\n", CHAINS, ms * 1e6 / mmas,
        (double)sms * 4 * mmas * 2.0 * 16 * 8 * 32 / (ms * 1e-3) / 1e12);
}

// Read the SM clock beside the run (nvidia-smi): rates scale with it.
template <mma_kind K>
static void run_mma(const char *name, double flops_per_mma, int sms, uint32_t *out) {
    const int blocks = sms * 4;   // 32 warps per SM
    const float ms = median_ms([&] { mma_probe<K><<<blocks, 256>>>(out, 7u); CK(cudaGetLastError()); });
    const double total = (double)blocks * 8 * ITERS * ILP * flops_per_mma;
    printf("%-30s %8.1f T(FL)OPS\n", name, total / (ms * 1e-3) / 1e12);
}

int main() {
    int device = 0, sms = 0, clock = 0;
    CK(cudaGetDevice(&device));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CK(cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, device));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, device));
    printf("%s: %d SMs, sm_%d%d, max clock %.0f MHz\n", prop.name, sms, prop.major, prop.minor, clock / 1000.0);
    uint32_t *out;
    CK(cudaMalloc(&out, (size_t)sms * 4 * 256 * sizeof(uint32_t)));
    run_mma<BF16_F32>("mma bf16 -> f32 (m16n8k16)", 2.0 * 16 * 8 * 16, sms, out);
    run_mma<F16_F32>("mma f16 -> f32 (m16n8k16)", 2.0 * 16 * 8 * 16, sms, out);
    run_mma<F16_F16>("mma f16 -> f16 (m16n8k16)", 2.0 * 16 * 8 * 16, sms, out);
    run_mma<TF32_F32>("mma tf32 -> f32 (m16n8k8)", 2.0 * 16 * 8 * 8, sms, out);
    run_mma<E4M3_F32>("mma e4m3 -> f32 (m16n8k32)", 2.0 * 16 * 8 * 32, sms, out);
    run_mma<S8_S32>("mma s8 -> s32 (m16n8k32)", 2.0 * 16 * 8 * 32, sms, out);
#if defined(DS4_CUDA_HAVE_MXF4)
    run_mma<E2M1_NVF4>("mma nvfp4 block-scaled (k64)", 2.0 * 16 * 8 * 64, sms, out);
#endif
    run_chains<1>(sms, out); run_chains<2>(sms, out); run_chains<4>(sms, out);
    float *sink;
    CK(cudaMalloc(&sink, sizeof(float)));
    const uint64_t big = 4ull << 30;   // 4 GiB: far beyond L2
    float4 *buf;
    CK(cudaMalloc(&buf, big));
    CK(cudaMemset(buf, 0, big));
    const uint64_t n = big / sizeof(float4);
    float ms = median_ms([&] { read_probe<<<sms * 8, 512>>>(buf, n, sink); CK(cudaGetLastError()); });
    printf("%-30s %8.1f GB/s\n", "DRAM streaming read", big / (ms * 1e-3) / 1e9);
    ms = median_ms([&] { CK(cudaMemcpyAsync(buf, buf + n / 2, big / 2, cudaMemcpyDeviceToDevice)); });
    printf("%-30s %8.1f GB/s (read + write)\n", "DRAM copy (cudaMemcpy D2D)", big / (ms * 1e-3) / 1e9);
    const uint64_t small = 8ull << 20;   // 8 MiB: resident in the 24 MiB L2
    const int passes = 64;
    ms = median_ms([&] { l2_probe<<<sms * 8, 512>>>(buf, small / sizeof(float4), passes, sink); CK(cudaGetLastError()); });
    printf("%-30s %8.1f GB/s\n", "L2 read (8 MiB, 64 passes)", (double)small * passes / (ms * 1e-3) / 1e9);
    return 0;
}
