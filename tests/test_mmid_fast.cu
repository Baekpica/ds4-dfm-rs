/* Parity harness for the counting-sort mm_ids map (cuda/mmq/mmid.cu).
 *
 * For every shape below the map is built twice through the production
 * launcher, once with the counting sort disabled (the warp scans) and once
 * enabled, over the same pre-zeroed output buffers.  ids_src1, ids_dst and
 * expert_bounds must be byte-identical: the kernels are only faster, never
 * a different routing.  Shapes cover Qwen's 8K gate/up and down maps, the
 * Solar/DeepSeek top-6/top-8 widths, dropped (-1) router rows, ids at or
 * past n_experts, and a decode width that stays on the warp scans. */
#include <cuda_runtime.h>

#include <cstdint>

#include "../cuda/mmq/mmid.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

#define CUDA_REQUIRE(call) do {                                               \
    const cudaError_t err_ = (call);                                          \
    if (err_ != cudaSuccess) {                                                \
        fprintf(stderr, "FAIL: %s -> %s (%s:%d)\n", #call,                    \
                cudaGetErrorString(err_), __FILE__, __LINE__);                \
        exit(1);                                                              \
    }                                                                         \
} while (0)

struct shape {
    const char *name;
    int n_tokens;
    int n_expert_used;
    int n_experts;
    int drop_percent;   /* rows whose ids are all -1 */
    int overflow_percent; /* slots holding an id == n_experts (dropped) */
};

static uint32_t next_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

/* Distinct experts per token (router top-k is without replacement), with a
 * skew so some buckets are far larger than others. */
static void fill_ids(std::vector<int32_t> &ids, const shape &s, uint32_t seed) {
    uint32_t state = seed;
    std::vector<int> used;
    for (int it = 0; it < s.n_tokens; it++) {
        int32_t *row = ids.data() + (size_t)it * s.n_expert_used;
        if ((int)(next_u32(&state) % 100u) < s.drop_percent) {
            for (int k = 0; k < s.n_expert_used; k++) row[k] = -1;
            continue;
        }
        used.clear();
        for (int k = 0; k < s.n_expert_used; k++) {
            int e;
            do {
                const uint32_t r = next_u32(&state);
                e = (r & 1u) ? (int)(r % (uint32_t)(s.n_experts / 8 + 1))
                             : (int)((r >> 8) % (uint32_t)s.n_experts);
                bool dup = false;
                for (int d : used) if (d == e) dup = true;
                if (!dup) break;
            } while (true);
            used.push_back(e);
            row[k] = (int)(next_u32(&state) % 100u) < s.overflow_percent
                ? s.n_experts : e;
        }
    }
}

static void build(const shape &s, const int32_t *d_ids, int32_t *d_src1,
                  int32_t *d_dst, int32_t *d_bounds, int enable_fast,
                  float *ms_out) {
    const size_t rows = (size_t)s.n_tokens * s.n_expert_used;
    ds4_mmid_fast_set_enabled(enable_fast);
    cudaEvent_t t0, t1;
    CUDA_REQUIRE(cudaEventCreate(&t0));
    CUDA_REQUIRE(cudaEventCreate(&t1));
    CUDA_REQUIRE(cudaMemsetAsync(d_src1, 0, rows * sizeof(int32_t), 0));
    CUDA_REQUIRE(cudaMemsetAsync(d_dst, 0, rows * sizeof(int32_t), 0));
    CUDA_REQUIRE(cudaMemsetAsync(d_bounds, 0xff, (size_t)(s.n_experts + 1) * sizeof(int32_t), 0));
    const size_t scratch_bytes = ds4_mmid_fast_scratch_bytes(
        s.n_experts, s.n_tokens, s.n_expert_used);
    void *scratch = NULL;
    if (scratch_bytes) CUDA_REQUIRE(cudaMalloc(&scratch, scratch_bytes));
    CUDA_REQUIRE(cudaEventRecord(t0, 0));
    ggml_cuda_launch_mm_ids_helper_scratch(
        d_ids, d_src1, d_dst, d_bounds, s.n_experts, s.n_tokens,
        s.n_expert_used, /*nchannels_y=*/1, /*si1=*/s.n_expert_used,
        /*sis1=*/1, scratch, scratch_bytes, 0);
    CUDA_REQUIRE(cudaEventRecord(t1, 0));
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaStreamSynchronize(0));
    if (scratch) CUDA_REQUIRE(cudaFree(scratch));
    CUDA_REQUIRE(cudaEventElapsedTime(ms_out, t0, t1));
    CUDA_REQUIRE(cudaEventDestroy(t0));
    CUDA_REQUIRE(cudaEventDestroy(t1));
}

static void run_shape(const shape &s) {
    const size_t rows = (size_t)s.n_tokens * s.n_expert_used;
    std::vector<int32_t> ids(rows);
    fill_ids(ids, s, 0x9e3779b9u ^ (uint32_t)s.n_tokens);
    int32_t *d_ids = NULL, *d_src1 = NULL, *d_dst = NULL, *d_bounds = NULL;
    CUDA_REQUIRE(cudaMalloc(&d_ids, rows * sizeof(int32_t)));
    CUDA_REQUIRE(cudaMalloc(&d_src1, rows * sizeof(int32_t)));
    CUDA_REQUIRE(cudaMalloc(&d_dst, rows * sizeof(int32_t)));
    CUDA_REQUIRE(cudaMalloc(&d_bounds, (size_t)(s.n_experts + 1) * sizeof(int32_t)));
    CUDA_REQUIRE(cudaMemcpy(d_ids, ids.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice));

    std::vector<int32_t> src1[2], dst[2], bounds[2];
    float ms[2] = {0.0f, 0.0f};
    for (int fast = 0; fast < 2; fast++) {
        /* Warm once so the timing excludes module load and pool growth. */
        build(s, d_ids, d_src1, d_dst, d_bounds, fast, &ms[fast]);
        build(s, d_ids, d_src1, d_dst, d_bounds, fast, &ms[fast]);
        src1[fast].resize(rows);
        dst[fast].resize(rows);
        bounds[fast].resize((size_t)s.n_experts + 1);
        CUDA_REQUIRE(cudaMemcpy(src1[fast].data(), d_src1, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_REQUIRE(cudaMemcpy(dst[fast].data(), d_dst, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_REQUIRE(cudaMemcpy(bounds[fast].data(), d_bounds, (size_t)(s.n_experts + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
    REQUIRE(memcmp(bounds[0].data(), bounds[1].data(), bounds[0].size() * sizeof(int32_t)) == 0,
            "expert_bounds parity");
    REQUIRE(memcmp(src1[0].data(), src1[1].data(), rows * sizeof(int32_t)) == 0,
            "ids_src1 parity");
    REQUIRE(memcmp(dst[0].data(), dst[1].data(), rows * sizeof(int32_t)) == 0,
            "ids_dst parity");
    /* Structural sanity independent of the reference: bounds are monotone
     * and end at the number of in-range ids. */
    int in_range = 0;
    for (size_t i = 0; i < rows; i++) {
        if (ids[i] >= 0 && ids[i] < s.n_experts) in_range++;
    }
    int negatives = 0;
    for (size_t i = 0; i < rows; i++) if (ids[i] < 0) negatives++;
    REQUIRE(bounds[1][0] == negatives, "bucket 0 starts after dropped ids");
    for (int e = 0; e < s.n_experts; e++)
        REQUIRE(bounds[1][e] <= bounds[1][e + 1], "expert_bounds monotone");
    REQUIRE(bounds[1][s.n_experts] == negatives + in_range, "expert_bounds total");
    printf("%-34s tokens=%6d used=%2d experts=%4d  scan %.3f ms  sort %.3f ms  bit-identical\n",
           s.name, s.n_tokens, s.n_expert_used, s.n_experts, ms[0], ms[1]);

    CUDA_REQUIRE(cudaFree(d_bounds));
    CUDA_REQUIRE(cudaFree(d_dst));
    CUDA_REQUIRE(cudaFree(d_src1));
    CUDA_REQUIRE(cudaFree(d_ids));
}

int main(void) {
    const shape shapes[] = {
        {"Qwen gate/up 8K prefill",       8025, 10, 512, 0, 0},
        {"Qwen down 8K prefill (rows)",  80250,  1, 512, 0, 0},
        {"Qwen gate/up 256 boot chunk",    256, 10, 512, 0, 0},
        {"Qwen with dropped rows",        8025, 10, 512, 3, 0},
        {"Qwen with overflow ids",        8025, 10, 512, 0, 2},
        {"top-6 over 256 experts",        4096,  6, 256, 1, 0},
        {"top-8 over 384 experts",        3000,  8, 384, 0, 1},
        {"top-16 over 64 experts (scans)", 2048, 16,  64, 0, 0},
        {"ragged chunk tail",             8001, 10, 512, 0, 0},
        {"decode width stays on scans",     16, 10, 512, 0, 0},
    };
    puts("== counting-sort mm_ids map parity ==");
    for (const shape &s : shapes) run_shape(s);
    puts("all mm_ids map parity checks passed");
    return 0;
}
