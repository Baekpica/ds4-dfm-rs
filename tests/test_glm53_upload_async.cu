/* The production upload driver must finish its copy without draining compute. */
#include "../ds4_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "upload FAIL %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
enum { BYTES = 65536, CYCLES = 220000000, PASSES = 8 };

__global__ static void compute_wait(void) {
    const unsigned long long start = clock64();
    while (clock64() - start < CYCLES) { __nanosleep(1000); }
}

int main(void) {
    CHECK(ds4_gpu_init());
    ds4_gpu_tensor *dst = ds4_gpu_tensor_alloc(BYTES);
    ds4_gpu_upload *upload = ds4_gpu_upload_new();
    CHECK(dst && upload);
    std::vector<unsigned char> source(BYTES), result(BYTES);
    cudaEvent_t done;
    CHECK(cudaEventCreate(&done) == cudaSuccess);
    for (unsigned pass = 0; pass < PASSES; pass++) {
        for (unsigned i = 0; i < BYTES; i++) { source[i] = (unsigned char)(i * 31u + pass); }
        compute_wait<<<1, 32>>>();
        CHECK(cudaGetLastError() == cudaSuccess);
        CHECK(cudaEventRecord(done) == cudaSuccess);
        int copied = 0;
        std::thread worker([&] { copied = ds4_gpu_upload_write(upload, dst, 0, source.data(), BYTES); });
        worker.join();
        CHECK(copied);
        CHECK(cudaEventQuery(done) == cudaErrorNotReady);
        // Poison immediately: returning from write releases the source bytes.
        std::fill(source.begin(), source.end(), 0xa5u);
        CHECK(ds4_gpu_tensor_read(dst, 0, result.data(), BYTES));
        for (unsigned i = 0; i < BYTES; i++) { CHECK(result[i] == (unsigned char)(i * 31u + pass)); }
    }
    CHECK(!ds4_gpu_upload_write(upload, dst, BYTES, source.data(), 1));
    CHECK(cudaEventDestroy(done) == cudaSuccess);
    ds4_gpu_upload_free(upload);
    ds4_gpu_tensor_free(dst);
    ds4_gpu_cleanup();
    std::puts("SSD upload: independent compute, source reuse, bounds PASS");
}
