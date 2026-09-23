// Drives ds4_gpu_mimo2_attention, not the split kernel by itself.
// FATTN=0 and ATTN_SPLIT=0 must each keep the walk and skip the scratch.
// A failed scratch alloc must still return the walk with a clear last error.
// Cleanup must drop the scratch so the next init can allocate it again.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool m2_hmma_available = false;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) { return 1; }
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static inline cudaStream_t ds4_current_stream(void) { return 0; }
static inline int ds4_capture_active(void) { return 0; }

static const char *cuda_model_range_ptr(
        const void *, uint64_t, uint64_t, const char *) {
    return nullptr;
}

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
    int memc;
};

// Same byte count as m2_split_ready. The hook fails only that allocation.
enum { M2_SPLIT_BYTES = 32 * 64 * (2 + 128) * (int)sizeof(float) };

static int m2_fail_split_alloc = 0;
static cudaError_t (*m2_real_malloc)(void **, size_t) = cudaMalloc;

template <typename T>
static cudaError_t m2_hook_malloc(T **ptr, size_t bytes) {
    if (m2_fail_split_alloc && bytes == (size_t)M2_SPLIT_BYTES) {
        m2_fail_split_alloc = 0;
        void *discard = nullptr;
        // A real failure leaves the sticky last-error the fallback must clear.
        return m2_real_malloc(&discard, (size_t)1 << 62);
    }
    void *raw = nullptr;
    cudaError_t status = m2_real_malloc(&raw, bytes);
    if (status == cudaSuccess) { *ptr = static_cast<T *>(raw); }
    return status;
}

#define cudaMalloc m2_hook_malloc
#include "../ds4_mimo2_gpu.cuh"
#undef cudaMalloc

static int check_close(const std::vector<float> &got, const std::vector<float> &walk) {
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i])) { return 0; }
        if (std::fabs(got[i] - walk[i]) > 1e-4) { return 0; }
    }
    return 1;
}

static int launch_walk(
        float *out, float *q, __half *cache, unsigned *pos,
        std::vector<float> &host) {
    mimo2_attention<<<dim3(64 / 4, 1), 128>>>(out, q, cache, nullptr, pos, 4, 18, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) { return 0; }
    if (cudaMemcpy(host.data(), out, host.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 0;
    }
    return 1;
}

int main() {
    enum { HEADS = 64, KEY = 192, VALUE = 128, KV = 4, CAP = 18 };
    const size_t stride = KV * (KEY + VALUE);
    std::vector<__half> cache(CAP * stride);
    std::vector<float> q(HEADS * KEY), host(HEADS * VALUE), walk(HEADS * VALUE);
    unsigned position = 17;
    for (size_t i = 0; i < q.size(); i++) { q[i] = sinf((float)i * 0.013f); }
    for (size_t i = 0; i < cache.size(); i++) {
        cache[i] = __float2half_rn(sinf((float)i * 0.01f));
    }

    float *dq = nullptr, *dout = nullptr;
    __half *dc = nullptr;
    unsigned *dp = nullptr;
    if (cudaMalloc(&dq, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dout, host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dc, cache.size() * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&dp, sizeof(position)) != cudaSuccess) {
        fprintf(stderr, "fixture alloc failed\n");
        return 1;
    }
    cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, cache.data(), cache.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dp, &position, sizeof(position), cudaMemcpyHostToDevice);
    if (!launch_walk(dout, dq, dc, dp, walk)) { return 1; }

    ds4_gpu_tensor out{dout, host.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor query{dq, q.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor kv{dc, cache.size() * sizeof(__half), 0, 0};
    ds4_gpu_tensor pos{dp, sizeof(position), 0, 0};

    auto run = [&](const char *fattn, const char *split) {
        if (fattn) { setenv("DS4_MIMO2_FATTN", fattn, 1); }
        else { unsetenv("DS4_MIMO2_FATTN"); }
        if (split) { setenv("DS4_MIMO2_ATTN_SPLIT", split, 1); }
        else { unsetenv("DS4_MIMO2_ATTN_SPLIT"); }
        unsetenv("DS4_MIMO2_PREFILL_HMMA");
        unsetenv("DS4_MIMO2_PREFILL_ASYNC");
        unsetenv("DS4_MIMO2_SWA_HMMA");
        return ds4_gpu_mimo2_attention(
            &out, &query, &kv, &pos, nullptr, 0, 0, KV, CAP, 1, 0, position);
    };

    // Alloc failure is first, while the scratch is still null.
    m2_fail_split_alloc = 1;
    cudaGetLastError();
    int rc = run(nullptr, nullptr);
    cudaError_t stuck = cudaGetLastError();
    if (rc != 1 || stuck != cudaSuccess || m2_split_buf != nullptr) {
        printf("alloc_fallback rc=%d last=%s buf=%p\n", rc, cudaGetErrorString(stuck), (void *)m2_split_buf);
        return 2;
    }
    if (cudaMemcpy(host.data(), dout, host.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        !check_close(host, walk)) {
        printf("alloc_fallback output drifted\n");
        return 2;
    }
    puts("alloc_fallback=walk_clear");

    if (run("1", "0") != 1 || m2_split_buf != nullptr) {
        printf("attn_split=0 still allocated buf=%p\n", (void *)m2_split_buf);
        return 3;
    }
    puts("attn_split=0 walk");

    if (run("0", "1") != 1 || m2_split_buf != nullptr) {
        printf("fattn=0 still allocated buf=%p\n", (void *)m2_split_buf);
        return 4;
    }
    puts("fattn=0 walk");

    if (run("1", "1") != 1 || m2_split_buf == nullptr) {
        printf("split path did not allocate\n");
        return 5;
    }
    if (cudaMemcpy(host.data(), dout, host.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        !check_close(host, walk)) {
        printf("split output drifted\n");
        return 5;
    }
    puts("split=on");

    // Release is the cleanup path ds4_gpu_cleanup calls. Missing or no-op
    // leaves the scratch so the next init cannot prove a fresh alloc.
    m2_split_release();
    if (m2_split_buf != nullptr) {
        printf("cleanup left buf=%p\n", (void *)m2_split_buf);
        return 6;
    }
    if (run("1", "1") != 1 || m2_split_buf == nullptr) {
        printf("second init failed buf=%p\n", (void *)m2_split_buf);
        return 6;
    }
    puts("cleanup_reinit=ok");
    m2_split_release();

    // Verify width is 2..8 rows. That batch must use the split scratch.
    // Nine rows stays off it. ATTN_SPLIT=0 keeps the walk and does not alloc.
    enum { VROWS = 8, VCAP = 9 };
    std::vector<float> vq(VCAP * HEADS * KEY), vhost(VCAP * HEADS * VALUE), vwalk(VROWS * HEADS * VALUE);
    std::vector<unsigned> vpos(VCAP, position);
    for (unsigned r = 0; r < VCAP; r++) {
        for (size_t i = 0; i < HEADS * KEY; i++) {
            vq[(size_t)r * HEADS * KEY + i] = sinf((float)(r + 1) * (float)i * 0.013f);
        }
    }
    float *dvq = nullptr, *dvout = nullptr;
    unsigned *dvp = nullptr;
    if (cudaMalloc(&dvq, vq.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dvout, vhost.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dvp, vpos.size() * sizeof(unsigned)) != cudaSuccess) {
        return 7;
    }
    cudaMemcpy(dvq, vq.data(), vq.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvp, vpos.data(), vpos.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
    for (unsigned r = 0; r < VROWS; r++) {
        mimo2_attention<<<dim3(HEADS / 4, 1), 128>>>(
            dvout + (size_t)r * HEADS * VALUE, dvq + (size_t)r * HEADS * KEY,
            dc, nullptr, dvp + r, KV, CAP, 0);
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(vwalk.data(), dvout, vwalk.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 7;
    }
    ds4_gpu_tensor vout{dvout, vhost.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor vquery{dvq, vq.size() * sizeof(float), 0, 0};
    ds4_gpu_tensor vpositions{dvp, vpos.size() * sizeof(unsigned), 0, 0};
    auto verify = [&](uint32_t nrows) {
        setenv("DS4_MIMO2_FATTN", "1", 1);
        setenv("DS4_MIMO2_ATTN_SPLIT", "1", 1);
        return ds4_gpu_mimo2_attention(
            &vout, &vquery, &kv, &vpositions, nullptr, 0, 0, KV, CAP, nrows, 0, position);
    };
    if (verify(2) != 1 || m2_split_buf == nullptr) {
        printf("rows=2 stayed on the walk buf=%p\n", (void *)m2_split_buf);
        return 7;
    }
    if (cudaMemcpy(vhost.data(), dvout, 2 * HEADS * VALUE * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        !check_close({vhost.begin(), vhost.begin() + 2 * HEADS * VALUE},
                     {vwalk.begin(), vwalk.begin() + 2 * HEADS * VALUE})) {
        printf("rows=2 output drifted\n");
        return 7;
    }
    if (verify(8) != 1) { return 7; }
    if (cudaMemcpy(vhost.data(), dvout, vwalk.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        !check_close({vhost.begin(), vhost.begin() + vwalk.size()}, vwalk)) {
        printf("rows=8 output drifted\n");
        return 7;
    }
    puts("verify_2_8=split");
    m2_split_release();
    setenv("DS4_MIMO2_ATTN_SPLIT", "0", 1);
    if (ds4_gpu_mimo2_attention(
            &vout, &vquery, &kv, &vpositions, nullptr, 0, 0, KV, CAP, 4, 0, position) != 1 ||
        m2_split_buf != nullptr) {
        printf("verify kill switch allocated buf=%p\n", (void *)m2_split_buf);
        return 8;
    }
    setenv("DS4_MIMO2_ATTN_SPLIT", "1", 1);
    if (verify(9) != 1 || m2_split_buf != nullptr) {
        printf("rows=9 used the verify split buf=%p\n", (void *)m2_split_buf);
        return 8;
    }
    puts("verify_bounds=ok");
    return 0;
}
