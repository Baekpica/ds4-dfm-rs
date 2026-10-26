// Include the DiT implementation only to queue work on its private stream.
#include "../../native/h3/h3_dit_gpu.cu"
#include "../../native/h3/h3_text_gpu.h"
#include "../../native/h3/h3_vision_gpu.h"

#include <chrono>
#include <future>
#include <limits>

static int failures = 0;

static void check(bool passed, const char *name) {
    printf("%s %s\n", passed ? "PASS" : "FAIL", name);
    failures += !passed;
}

static void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(2);
    }
}

struct StreamBlock {
    std::promise<void> entered;
    std::promise<void> release;
    std::future<void> released = release.get_future();
};

static void CUDART_CB hold_stream(void *data) {
    auto *block = static_cast<StreamBlock *>(data);
    block->entered.set_value();
    block->released.wait();
}

static void read_waits(h3d_context *gpu) {
    char error[512] = {};
    h3d_tensor *tensor = h3d_alloc(gpu, H3D_U32, 1, error, sizeof(error));
    if (!tensor) {
        fprintf(stderr, "%s\n", error);
        exit(2);
    }
    uint32_t source = 42;
    uint32_t *result = nullptr;
    cuda_check(cudaMallocHost(&result, sizeof(*result)));
    cuda_check(cudaMemcpy(tensor->data, &source, sizeof(source), cudaMemcpyHostToDevice));
    *result = 0;

    StreamBlock block;
    cuda_check(cudaLaunchHostFunc(gpu->stream, hold_stream, &block));
    block.entered.get_future().wait();
    std::promise<void> started;
    auto reader = std::async(std::launch::async, [&] {
        cuda_check(cudaSetDevice(0));
        started.set_value();
        return h3d_read(gpu, result, tensor, error, sizeof(error));
    });
    started.get_future().wait();
    bool returned_early = reader.wait_for(std::chrono::milliseconds(500)) ==
                          std::future_status::ready;
    // Always release and join before inspecting or freeing the host buffer.
    block.release.set_value();
    int ok = reader.get();
    cuda_check(cudaStreamSynchronize(gpu->stream));
    check(ok && !returned_early && *result == source, "read waits for host bytes");
    cuda_check(cudaFreeHost(result));
    h3d_free(tensor);
}

static void allocation_sizes(h3d_context *dit, h3t_context *text, h3v_context *vision) {
    char error[512] = {};
    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    for (int dtype : {H3D_BF16, H3D_F32, H3D_U32}) {
        size_t item = dtype == H3D_BF16 ? 2 : 4;
        auto *tensor = h3d_alloc(dit, dtype, maximum / item + 2, error, sizeof(error));
        check(!tensor, "DiT allocation rejects byte overflow");
        h3d_free(tensor);
    }
    for (int dtype : {H3T_BF16, H3T_F32}) {
        size_t item = dtype == H3T_BF16 ? 2 : 4;
        auto *tensor = h3t_alloc(text, dtype, maximum / item + 2, error, sizeof(error));
        check(!tensor, "text allocation rejects byte overflow");
        h3t_free(tensor);
    }
    auto *tensor = h3v_alloc(vision, maximum / 2 + 2, error, sizeof(error));
    check(!tensor, "vision allocation rejects byte overflow");
    h3v_free(tensor);
}

static void gate_dtype(h3d_context *gpu) {
    char error[512] = {};
    auto *values = h3d_alloc(gpu, H3D_BF16, 1, error, sizeof(error));
    if (!values) {
        exit(2);
    }
    cuda_check(cudaMemset(values->data, 0, 2));
    for (int dtype : {H3D_BF16, H3D_F32, H3D_U32}) {
        // Even invalid types have four initialized bytes and a valid row 0.
        auto *map = h3d_alloc(gpu, dtype, dtype == H3D_BF16 ? 2 : 1, error, sizeof(error));
        if (!map) {
            exit(2);
        }
        cuda_check(cudaMemset(map->data, 0, 4));
        int ok = h3d_gate(gpu, values, values, values, values, map,
                          1, 1, 1, 0, error, sizeof(error));
        check(bool(ok) == (dtype == H3D_U32), "gate requires U32 row indices");
        cuda_check(cudaStreamSynchronize(gpu->stream));
        h3d_free(map);
    }
    h3d_free(values);
}

int main() {
    char error[512] = {};
    auto *dit = h3d_create(error, sizeof(error));
    auto *text = h3t_create(error, sizeof(error));
    auto *vision = h3v_create(error, sizeof(error));
    if (!dit || !text || !vision) {
        fprintf(stderr, "%s\n", error);
        return 2;
    }
    read_waits(dit);
    allocation_sizes(dit, text, vision);
    gate_dtype(dit);
    h3v_destroy(vision);
    h3t_destroy(text);
    h3d_destroy(dit);
    return failures ? 1 : 0;
}
