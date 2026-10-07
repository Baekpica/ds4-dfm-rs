/* Actual range-copy transaction, model-free CUDA mocks. No device allocation. */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "GLM upload FAIL %d: %s\n", __LINE__, #x); \
    std::exit(1); \
} } while (0)

enum { DS4_GOVC_ENGINE_BOOT, DS4_MEMC_WEIGHT_SPAN, DS4_MEMD_UNIFIED_DEVICE,
       DS4_GOV_ADMIT, DS4_GOV_REFUSE };
enum { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorInvalidValue,
       cudaMemcpyHostToDevice };
using cudaError_t = int;
struct ds4_gov_claim {
    int requester, memc, domain;
    uint64_t proposed_outstanding, operation_transient;
};
struct cuda_model_range {
    const void *host_base;
    uint64_t offset, bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered, arena_allocated, imported_ipc, imported_vmm;
    uint64_t vmm_handle, vmm_va, vmm_alloc_bytes;
};

static uint64_t g_model_range_bytes;
static int g_model_plan_frozen, g_in_governed_materialize;
static struct {
    bool host_mapped, complete, reject_claim, fail_alloc, fail_copy, fail_publish;
    unsigned allocations, frees, copies, staged, publications, charges, covers, leases;
    uint64_t charged_bytes, leased_bytes;
    ds4_gov_claim claim;
    char *device;
    const void *source;
    uint64_t offset, bytes;
} mock;

static bool cuda_model_source_host_mapped(const void *) { return mock.host_mapped; }
static int cuda_model_map_replaces_complete(const void *) { return mock.complete; }
static const char *cuda_model_direct_fallback_ptr(const void *map, uint64_t off) {
    return static_cast<const char *>(map) + off;
}
static int ds4_gov_governed_check(const char *, const ds4_gov_claim *claim, int) {
    mock.claim = *claim;
    return mock.reject_claim ? DS4_GOV_REFUSE : DS4_GOV_ADMIT;
}
static cudaError_t cudaMalloc(void **out, size_t bytes) {
    ++mock.allocations;
    if (mock.fail_alloc) { return cudaErrorMemoryAllocation; }
    *out = std::malloc(bytes);
    CHECK(*out);
    mock.device = static_cast<char *>(*out);
    return cudaSuccess;
}
static cudaError_t cudaFree(void *ptr) {
    ++mock.frees;
    std::free(ptr);
    mock.device = nullptr;
    return cudaSuccess;
}
static cudaError_t cudaGetLastError() { return cudaSuccess; }
static const char *cudaGetErrorString(cudaError_t) { return "injected fixture fault"; }
static cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, int) {
    ++mock.copies;
    if (mock.fail_copy) { return cudaErrorInvalidValue; }
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}
static int cuda_stage_copy_to_dev(const void *map, uint64_t offset,
                                   uint64_t bytes, char *dev,
                                   int direct_discard, const char *) {
    ++mock.staged;
    CHECK(direct_discard == 0);
    CHECK(map == mock.source && offset == mock.offset && bytes == mock.bytes);
    if (mock.fail_copy) { return 0; }
    std::memcpy(dev, static_cast<const char *>(map) + offset, bytes);
    return 1;
}
static int cuda_model_range_publish(cuda_model_range range) {
    ++mock.publications;
    CHECK(g_in_governed_materialize == 1);
    CHECK(range.host_base == mock.source && range.offset == mock.offset);
    CHECK(range.bytes == mock.bytes && range.device_ptr == mock.device);
    CHECK(range.host_registered == 0 && range.arena_allocated == 0);
    CHECK(std::memcmp(range.device_ptr,
                      static_cast<const char *>(mock.source) + mock.offset,
                      mock.bytes) == 0);
    return !mock.fail_publish;
}
static void cuda_mem_note_alloc_src(int cls, int domain, uint64_t alloc,
                                     uint64_t commit, const void *source) {
    CHECK(cls == DS4_MEMC_WEIGHT_SPAN && domain == DS4_MEMD_UNIFIED_DEVICE);
    CHECK(alloc == mock.bytes && commit == mock.bytes && source == mock.source);
    ++mock.charges;
    mock.charged_bytes += commit;
}
static void ds4_gov_publish_use(int cls, uint64_t alloc, uint64_t commit) {
    CHECK(cls == DS4_GOVC_ENGINE_BOOT && alloc == commit);
    ++mock.leases;
    mock.leased_bytes = commit;
}
static void cuda_substrate_cover(const void *source, uint64_t bytes) {
    CHECK(source == mock.source && bytes == mock.bytes);
    ++mock.covers;
}
struct weight_env { int verbose; };
static weight_env cuda_weight_env() { return {0}; }

#include "glm53_upload_range.inc"

static char source[384];
static constexpr uint64_t OFFSET = 17, BYTES = 257, EXISTING = 19;

static void reset(bool complete) {
    std::memset(&mock, 0, sizeof(mock));
    mock.complete = complete;
    mock.source = source;
    mock.offset = OFFSET;
    mock.bytes = BYTES;
    g_model_range_bytes = EXISTING;
    g_model_plan_frozen = g_in_governed_materialize = 0;
    for (size_t i = 0; i < sizeof(source); ++i) { source[i] = char((i * 17u) ^ (i >> 2)); }
}

static const char *run() {
    return cuda_model_range_populate_device_copy(source, OFFSET, BYTES, "fixture");
}

static void no_publication() {
    CHECK(mock.publications == 0 && mock.charges == 0 && mock.covers == 0);
    CHECK(mock.leases == 0 && g_model_range_bytes == EXISTING);
    CHECK(g_in_governed_materialize == 0);
}

static void transactions(bool complete) {
    reset(complete);
    const char *dev = run();
    CHECK(dev && dev == mock.device && dev != source + OFFSET);
    CHECK(mock.claim.proposed_outstanding == EXISTING + BYTES);
    CHECK(mock.claim.operation_transient == 0);
    CHECK(mock.publications == 1 && mock.charges == 1 && mock.covers == 1);
    CHECK(mock.charged_bytes == BYTES && mock.leased_bytes == EXISTING + BYTES);
    CHECK(g_model_range_bytes == EXISTING + BYTES && g_in_governed_materialize == 0);
    if (complete) {
        CHECK(mock.staged == 1 && mock.copies == 0);
    } else {
        CHECK(mock.staged == 0 && mock.copies == 1);
    }
    cudaFree(mock.device);

    reset(complete);
    mock.host_mapped = true;
    CHECK(run() == source + OFFSET && mock.allocations == 0);
    no_publication();
    reset(complete);
    mock.reject_claim = true;
    CHECK(run() == source + OFFSET && mock.allocations == 0);
    no_publication();
    reset(complete);
    mock.fail_alloc = true;
    CHECK(run() == source + OFFSET && mock.frees == 0);
    no_publication();
    reset(complete);
    mock.fail_copy = true;
    CHECK(run() == source + OFFSET && mock.frees == 1);
    no_publication();
    reset(complete);
    mock.fail_publish = true;
    CHECK(!run() && mock.frees == 1 && mock.publications == 1);
    CHECK(mock.charges == 0 && mock.covers == 0 && mock.leases == 0);
    CHECK(g_model_range_bytes == EXISTING && g_in_governed_materialize == 0);
    reset(complete);
    g_model_plan_frozen = 1;
    CHECK(run() == mock.device);
    CHECK(mock.claim.operation_transient == BYTES && mock.leases == 0);
    cudaFree(mock.device);
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    const bool complete = std::strcmp(argv[1], "replacement") == 0;
    CHECK(complete || std::strcmp(argv[1], "transaction") == 0);
    transactions(complete);
    std::puts(complete ? "GLM replacement upload: PASS" : "GLM upload transaction: PASS");
}
