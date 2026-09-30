// SPDX-License-Identifier: MIT
// Production Down dispatch and worklist bounds. Link as
// mimo2_gateup_schedule.cu, adding -ldl. TEST_WEIGHTS supplies raw IQ2_XS
// weights (592 MiB); EXPECT_PIPE64/EXPECT_WORKLIST are 0 or 1. Compare
// PROBE_DUMP files from fresh OFF/ON processes; timings are not qualified.
#include "ds4_mmq.h"

#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int ds4_cuda_q8_fold_take_q81(const void *, uint64_t, void *) {
    return 0;
}

namespace {
constexpr int M = 4096, K = 2048, EXPERTS = 256, USED = 8;
constexpr int MAX_TOKENS = 8192, QUANT_BLOCK = 256, QUANT_BYTES = 74;
constexpr int TILE_Y = 128, WIDTH_SHIFT = 29;
constexpr uint32_t COL_MASK = 0x1fffffffu;
constexpr size_t CHUNK = 1u << 20;
enum class Input { Uniform, Concentrated, Invalid, Generic };
int worklist_calls = 0, pipe64_calls = 0;

void fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

#define require(condition, message) do { if (!(condition)) { fail(message); } } while (0)

void check(cudaError_t rc) {
    if (rc == cudaSuccess) { return; }
    std::fprintf(stderr, "%s\n", cudaGetErrorString(rc));
    std::exit(1);
}

template<class T> T argument(void **args, int index) {
    T value;
    std::memcpy(&value, args[index], sizeof(value));
    return value;
}

template<class T> T *upload(const std::vector<T> &host) {
    T *device = nullptr;
    check(cudaMalloc(&device, host.size() * sizeof(T)));
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

int setting(const char *key) {
    const char *value = std::getenv(key);
    require(value && (!std::strcmp(value, "0") || !std::strcmp(value, "1")), key);
    return value[0] - '0';
}

void *weights() {
    const char *path = std::getenv("TEST_WEIGHTS");
    require(path, "set TEST_WEIGHTS to raw IQ2_XS weights");
    FILE *file = std::fopen(path, "rb");
    require(file, "weights open failed");
    const size_t bytes = (size_t)EXPERTS * M * K / QUANT_BLOCK * QUANT_BYTES;
    void *device = nullptr;
    check(cudaMalloc(&device, bytes));
    std::vector<char> host(CHUNK);
    for (size_t offset = 0; offset < bytes; offset += CHUNK) {
        const size_t size = bytes - offset < CHUNK ? bytes - offset : CHUNK;
        require(std::fread(host.data(), 1, size, file) == size, "weights truncated");
        check(cudaMemcpy((char *)device + offset, host.data(), size, cudaMemcpyHostToDevice));
    }
    require(std::fgetc(file) == EOF, "weights size mismatch");
    require(std::fclose(file) == 0, "weights close failed");
    return device;
}

void inspect(void **args, int max_width) {
    const int columns = argument<int>(args, 8);
    uint32_t count = 0;
    check(cudaMemcpy(&count, argument<const void *>(args, 6), sizeof(count), cudaMemcpyDeviceToHost));
    const size_t capacity = ((size_t)columns + EXPERTS * (max_width - 1)) / max_width * (M / TILE_Y);
    require(count > 0 && count <= capacity, "worklist count exceeds proven capacity");
    std::vector<uint3> work(count);
    check(cudaMemcpy(work.data(), argument<const void *>(args, 5), count * sizeof(uint3), cudaMemcpyDeviceToHost));
    std::vector<int32_t> bounds(EXPERTS + 1);
    check(cudaMemcpy(bounds.data(), argument<const void *>(args, 3), bounds.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    require(bounds[0] >= 0 && bounds.back() <= columns, "invalid expert bounds");
    size_t expected = 0;
    for (int expert = 0; expert < EXPERTS; expert++) {
        const int rows = bounds[expert + 1] - bounds[expert];
        require(rows >= 0, "expert bounds are not monotonic");
        expected += (size_t)((rows + max_width - 1) / max_width) * (M / TILE_Y);
    }
    require(count == expected, "worklist count does not match routed buckets");
    // Track valid column/output-row tiles, not just item counts: duplicate
    // writes can still produce finite and numerically identical output.
    // Negative IDs reserve a prefix in the sorted map; only expert buckets
    // are valid work, so leave that prefix outside the coverage proof.
    std::vector<unsigned char> seen((size_t)(bounds.back() - bounds[0]) * (M / TILE_Y));
    for (const uint3 item : work) {
        const uint32_t code = item.y >> WIDTH_SHIFT;
        require(code <= 4 && (max_width != 64 || code != 0), "worklist exceeds pipeline width");
        require(item.x < EXPERTS && item.z < M / TILE_Y, "worklist expert/row out of bounds");
        const int offset = (int)(item.y & COL_MASK);
        const int rows = bounds[item.x + 1] - bounds[item.x];
        require(offset % max_width == 0 && offset < rows, "worklist column out of bucket");
        const int width = 128 >> code;
        const int columns_left = rows - offset < max_width ? rows - offset : max_width;
        require(width >= columns_left, "worklist tail misses valid columns");
        for (int column = 0; column < columns_left; column++) {
            const size_t index = (size_t)(bounds[item.x] + offset + column - bounds[0]) * (M / TILE_Y) + item.z;
            require(!seen[index], "duplicate worklist output tile");
            seen[index] = 1;
        }
    }
    for (unsigned char covered : seen) { require(covered == 1, "missing worklist output tile"); }
}

void dump(const std::vector<float> &host) {
    const char *path = std::getenv("PROBE_DUMP");
    if (!path) { return; }
    FILE *file = std::fopen(path, "wb");
    require(file, "output open failed");
    require(std::fwrite(host.data(), sizeof(float), host.size(), file) == host.size(), "output write failed");
    require(std::fclose(file) == 0, "output close failed");
}
}

extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid,
        dim3 block, void **args, size_t shared, cudaStream_t stream) {
    using Launch = cudaError_t (*)(cudaKernel_t, dim3, dim3, void **, size_t, cudaStream_t);
    static Launch launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    require(launch, "CUDA launch symbol missing");
    const char *name = nullptr;
    check(cudaFuncGetName(&name, reinterpret_cast<const void *>(kernel)));
    const cudaError_t rc = launch(kernel, grid, block, args, shared, stream);
    if (!name || !std::strstr(name, "ds4_moe_worklist_mmq_kernel")) { return rc; }

    check(rc);
    check(cudaStreamSynchronize(stream));
    const int width = std::strstr(name, "Li64E") ? 64 : 128;
    worklist_calls++;
    pipe64_calls += width == 64;
    inspect(args, width);
    return rc;
}

int main(int argc, char **argv) {
    require(argc == 3, "usage: mimo2-down-pipe TOKENS uniform|concentrated|invalid|generic");
    char *end = nullptr;
    const long tokens = std::strtol(argv[1], &end, 10);
    require(end != argv[1] && !*end && tokens > 0 && tokens <= MAX_TOKENS, "invalid token count");
    Input input = Input::Uniform;
    if (!std::strcmp(argv[2], "concentrated")) { input = Input::Concentrated; }
    else if (!std::strcmp(argv[2], "invalid")) { input = Input::Invalid; }
    else if (!std::strcmp(argv[2], "generic")) { input = Input::Generic; }
    else { require(!std::strcmp(argv[2], "uniform"), "invalid input mode"); }

    const int want_pipe = setting("EXPECT_PIPE64"), want_work = setting("EXPECT_WORKLIST");
    const int rows = (int)tokens * USED;
    require(input == Input::Generic || tokens >= 32, "fused entry requires at least 32 tokens");
    require(ds4_mmq_init(0) == 0, "MMQ init failed");
    void *w = weights();
    std::mt19937 rng(0x5a0au);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> host((size_t)rows * K);
    for (float &x : host) { x = value(rng); }
    float *gate = upload(host);
    for (float &x : host) { x = value(rng); }
    float *up = upload(host);
    host.clear();
    host.shrink_to_fit();
    std::vector<int32_t> routes(rows);
    for (int token = 0; token < tokens; token++) {
        for (int slot = 0; slot < USED; slot++) {
            const int row = token * USED + slot;
            routes[row] = input == Input::Concentrated ? slot : (token + slot * 37) % EXPERTS;
            if (input == Input::Invalid && row % 113 == 0) { routes[row] = -1; }
        }
    }
    int32_t *ids = upload(routes);
    float *out = nullptr;
    const size_t count = (size_t)rows * M;
    check(cudaMalloc(&out, count * sizeof(float)));
    // Missing router IDs keep a sentinel; valid rows must all be written.
    check(cudaMemset(out, 0xff, count * sizeof(float)));
    const int rc = input == Input::Generic
        ? ds4_mmq_iq2_xs_moe(w, gate, ids, out, M, K, rows, EXPERTS, 1, 0)
        : ds4_mmq_mimo2_down(w, gate, up, ids, out, rows, 0);
    require(rc == 0, "Down entry failed");
    check(cudaDeviceSynchronize());
    require(worklist_calls == want_work && pipe64_calls == want_pipe, "Down dispatch mismatch");
    std::vector<float> output(count);
    check(cudaMemcpy(output.data(), out, count * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) {
        if (routes[i / M] < 0) { continue; }
        require(std::isfinite(output[i]), "valid output row is non-finite or unwritten");
    }
    dump(output);
    std::printf("dispatch_exact=true worklist=%d pipe64=%d tokens=%ld mode=%s\n",
                worklist_calls, pipe64_calls, tokens, argv[2]);
    check(cudaFree(out)); check(cudaFree(ids)); check(cudaFree(up));
    check(cudaFree(gate)); check(cudaFree(w));
    return 0;
}
