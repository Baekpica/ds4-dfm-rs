/* Test the real map entry without registering or allocating GPU memory. */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "../ds4_mem_census.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "map registration FAIL %d: %s\n", __LINE__, #x); \
    std::exit(1); \
} } while (0)

enum { cudaSuccess, cudaMemcpyHostToDevice, cudaDevAttrIntegrated };
using cudaError_t = int;
static char mapping[4096], device_copy[4096];
static const void *g_model_host_base, *g_model_coherent_direct_map, *g_model_fd_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size, g_mem_device_owned_bytes, g_model_range_bytes;
static int g_model_registered, g_model_hmm_direct, g_model_range_mapping_supported;
static int g_model_cache_full, g_model_device_owned, g_model_fd;
static uint8_t g_model_needs_device_copy[DS4_MSRC_MAX];
static struct { struct { int residency; } v[DS4_MSRC_MAX]; } g_model_srcs;
static struct {
    int source, integrated, coherent, complete, unified_limit;
    uint64_t limit;
    unsigned registrations, allocations, pinned;
} mock;

static int cuda_mem_src_index(const void *) { return mock.source; }
static int cuda_model_source_host_mapped(const void *) {
    return mock.source >= 0 &&
           g_model_srcs.v[mock.source].residency == DS4_RESIDENCY_HOST_MAPPED;
}
static int cuda_model_current_direct_available(const void *) { return g_model_registered; }
static void cuda_model_preserve_current_direct_mapping() { g_model_registered = 0; }
static int cuda_model_map_replaces_complete(const void *) { return mock.complete; }
static int cuda_model_coherent_host_access(int) { return mock.coherent; }
static unsigned cuda_model_host_register_flags() { return 0; }
struct weight_env { int cache_limit_unified; };
static weight_env cuda_weight_env() { return {mock.unified_limit}; }
static uint64_t cuda_model_cache_limit_bytes() { return mock.limit; }
static cudaError_t cudaGetDevice(int *value) { *value = 0; return cudaSuccess; }
static cudaError_t cudaDeviceGetAttribute(int *value, int, int) {
    *value = mock.integrated;
    return cudaSuccess;
}
static cudaError_t cudaHostRegister(void *, size_t, unsigned) {
    ++mock.registrations;
    return cudaSuccess;
}
static cudaError_t cudaHostGetDevicePointer(void **value, void *ptr, unsigned) {
    *value = ptr;
    return cudaSuccess;
}
static cudaError_t cudaHostUnregister(void *) { return cudaSuccess; }
static void cuda_pinned_file_note_register(const void *, uint64_t) { ++mock.pinned; }
static void cuda_pinned_file_note_unregister(const void *) { --mock.pinned; }
static cudaError_t cudaMalloc(void **value, size_t bytes) {
    CHECK(bytes == sizeof(mapping));
    ++mock.allocations;
    *value = device_copy;
    return cudaSuccess;
}
static cudaError_t cudaFree(void *) { return cudaSuccess; }
static cudaError_t cudaMemcpy(void *out, const void *in, size_t bytes, int) {
    std::memcpy(out, in, bytes);
    return cudaSuccess;
}
static cudaError_t cudaGetLastError() { return cudaSuccess; }
static const char *cudaGetErrorString(cudaError_t) { return "injected failure"; }
static void cuda_mem_note_alloc_src(int, int, uint64_t, uint64_t, const void *) {}

#include "model_registration.inc"

static void reset() {
    std::memset(&mock, 0, sizeof(mock));
    mock.integrated = mock.coherent = mock.unified_limit = 1;
    mock.limit = sizeof(mapping);
    g_model_srcs.v[0].residency = DS4_RESIDENCY_EAGER_DEVICE;
    g_model_host_base = g_model_coherent_direct_map = g_model_fd_host_base = nullptr;
    g_model_device_base = nullptr;
    g_model_registered_size = g_mem_device_owned_bytes = g_model_range_bytes = 0;
    g_model_registered = g_model_hmm_direct = g_model_range_mapping_supported = 0;
    g_model_cache_full = g_model_device_owned = 0;
    g_model_fd = 3;
    std::memset(g_model_needs_device_copy, 0, sizeof(g_model_needs_device_copy));
    unsetenv("DS4_CUDA_COPY_MODEL");
    unsetenv("DS4_CUDA_WEIGHT_IPC_MANIFEST");
}

static void run() { CHECK(ds4_gpu_set_model_map(mapping, sizeof(mapping)) == 1); }

int main() {
    reset();
    run();
    CHECK(mock.registrations == 0 && mock.allocations == 0 && mock.pinned == 0);
    CHECK(g_model_device_base == mapping && g_model_fd_host_base == mapping);
    CHECK(g_model_registered == 0 && g_model_hmm_direct == 0);
    CHECK(g_model_needs_device_copy[0] == 0);  /* Keep the existing budget. */
    run();
    CHECK(mock.registrations == 0);  /* Repeated binding is inert. */

    reset(); mock.limit--;
    run(); CHECK(mock.registrations == 1);  /* A partial cache still needs a map. */
    reset(); g_model_range_bytes = 1;
    run(); CHECK(mock.registrations == 1);  /* Other sources share the budget. */
    reset(); g_model_range_bytes = mock.limit + 1;
    run(); CHECK(mock.registrations == 1);  /* Avoid unsigned remaining-budget wrap. */
    reset(); mock.unified_limit = 0;
    run(); CHECK(mock.registrations == 1);  /* Explicit caps retain their policy. */
    reset(); mock.coherent = 0;
    run(); CHECK(mock.registrations == 1);
    reset(); mock.integrated = mock.coherent = mock.unified_limit = 0;
    run(); CHECK(mock.registrations == 1);
    reset(); mock.integrated = 0;
    run(); CHECK(mock.registrations == 1);  /* Eager materialization is integrated-only. */
    reset(); mock.source = -1;
    run(); CHECK(mock.registrations == 1);
    reset(); g_model_srcs.v[0].residency = DS4_RESIDENCY_LAZY_COPY_DEVICE;
    run(); CHECK(mock.registrations == 1);
    reset(); g_model_srcs.v[0].residency = DS4_RESIDENCY_HOST_MAPPED;
    run(); CHECK(mock.registrations == 1);

    reset(); setenv("DS4_CUDA_WEIGHT_IPC_MANIFEST", "fixture", 1);
    run(); CHECK(mock.registrations == 0 && g_model_hmm_direct == 1);
    reset(); mock.complete = 1;
    run(); CHECK(mock.registrations == 0 && g_model_coherent_direct_map == mapping);
    reset(); setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    run(); CHECK(mock.registrations == 0 && mock.allocations == 1 && g_model_device_owned == 1);
    CHECK(g_mem_device_owned_bytes == sizeof(mapping));
    reset(); setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    g_model_srcs.v[0].residency = DS4_RESIDENCY_HOST_MAPPED;
    CHECK(ds4_gpu_set_model_map(mapping, sizeof(mapping)) == 0);
    CHECK(mock.registrations == 0 && mock.allocations == 0);
    reset();
    CHECK(ds4_gpu_set_model_map(nullptr, sizeof(mapping)) == 0);
    CHECK(ds4_gpu_set_model_map(mapping, 0) == 0);
    CHECK(mock.registrations == 0);
    std::puts("model registration: PASS (eager fit, fallback policies, imports, explicit copy)");
}
