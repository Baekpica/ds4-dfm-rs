#include "../cuda/mmq/quantize.cuh"
#include "../cuda/mmq/ds4_mimo2_swiglu.cuh"
#include "../cuda/step37_primitives.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(call) do { cudaError_t rc = (call); if (rc != cudaSuccess) { \
    fprintf(stderr, "%s at %d\n", cudaGetErrorString(rc), __LINE__); exit(1); } } while (0)

__global__ static void inputs(float *gate, float *up, int *ids, int width, int rows) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)width * rows) {
        gate[i] = (int)(i % 127) * 0.07f - 4.0f;
        up[i] = (int)(i % 91) * 0.09f - 3.0f;
        if ((i / 128) % 7 == 0) { gate[i] = 0.0f; up[i] = 0.0f; }
    }
    if (i < rows) { ids[i] = rows - 1 - i; }
}

static void run(int rows) {
    constexpr int width = 2048, repeats = 20;
    const size_t count = (size_t)rows * width;
    const size_t bytes = count * sizeof(float);
    const size_t qbytes = count / 128 * sizeof(block_q8_1_mmq);
    float *gate, *up, *mid;
    int *ids;
    block_q8_1_mmq *ref, *got;
    CK(cudaMalloc(&gate, bytes)); CK(cudaMalloc(&up, bytes)); CK(cudaMalloc(&mid, bytes));
    CK(cudaMalloc(&ids, rows * sizeof(int)));
    CK(cudaMalloc(&ref, qbytes + 65536)); CK(cudaMalloc(&got, qbytes + 65536));
    inputs<<<(count + 255) / 256, 256>>>(gate, up, ids, width, rows);
    CK(cudaMemset(ref, 0, qbytes + 65536)); CK(cudaMemset(got, 0, qbytes + 65536));
    auto reference = [&]() {
        step37_swiglu<<<(count + 255) / 256, 256>>>(mid, gate, up, nullptr, width, count, 0);
        quantize_mmq_q8_1_cuda(mid, ids, ref, GGML_TYPE_IQ2_XS, width,
                width, width, count, width, rows, 1, 1, 0);
    };
    auto candidate = [&]() {
        mimo2_swiglu_q8<<<dim3(rows, width / 512), 128>>>(gate, up, ids, got, width, rows);
    };
    reference(); candidate(); CK(cudaDeviceSynchronize());
    std::vector<unsigned char> a(qbytes), b(qbytes);
    CK(cudaMemcpy(a.data(), ref, qbytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), got, qbytes, cudaMemcpyDeviceToHost));
    if (a != b) {
        size_t count_bad = 0;
        for (size_t i = 0; i < qbytes; i++) { count_bad += a[i] != b[i]; }
        fprintf(stderr, "rows=%d bad_bytes=%zu\n", rows, count_bad); exit(2);
    }
    cudaEvent_t start, end; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
    float times[2];
    for (int mode = 0; mode < 2; mode++) {
        CK(cudaEventRecord(start));
        for (int i = 0; i < repeats; i++) { if (mode) { candidate(); } else { reference(); } }
        CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
        CK(cudaEventElapsedTime(&times[mode], start, end));
    }
    printf("rows=%d exact=1 classic_ms=%.6f fused_ms=%.6f\n", rows, times[0]/repeats, times[1]/repeats);
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(end));
    CK(cudaFree(gate)); CK(cudaFree(up)); CK(cudaFree(mid)); CK(cudaFree(ids));
    CK(cudaFree(ref)); CK(cudaFree(got));
}
int main() { for (int rows : {1, 8, 256, 1032, 32768}) { run(rows); } }
