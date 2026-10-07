/* CUDA mapping policy with mocked device facts; no model or GPU allocations. */
#include "../ds4_gpu.h"
#include "../ds4_model_catalog.h"
#include "../ds4_mem_census.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { MAP_BYTES = 32768, PAGE_BYTES = 4096, cudaSuccess = 0 };
static const void *g_model_host_base, *g_model_coherent_direct_map, *g_model_fd_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered, g_model_device_owned, g_model_range_mapping_supported;
static int g_model_hmm_direct, g_model_cache_full, g_model_fd = -1;
static uint8_t g_model_needs_device_copy[DS4_MSRC_MAX];
static int coherent = 1, source = 0, device_error;
static uint64_t registered, copied;
static unsigned preserved;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM map FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static int cudaGetDevice(int *device) { *device = 0; return device_error; }
static int cuda_model_coherent_host_access(int device) { (void)device; return coherent; }
static int cuda_mem_src_index(const void *map) { (void)map; return source; }
static void cuda_model_preserve_current_direct_mapping(void) {
    preserved++;
    g_model_registered = g_model_device_owned = 0;
}
int ds4_gpu_set_model_map_spans(const void *map, uint64_t size,
        const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max) {
    (void)offsets; (void)sizes; (void)count; (void)max;
    g_model_host_base = map;
    registered += size;
    return 1;
}

#include "../ds4_glm53_map.inc"

int main(void) {
    unsigned char map[MAP_BYTES] = {0};
    uint64_t offsets[] = {0u, MAP_BYTES - PAGE_BYTES};
    uint64_t sizes[] = {PAGE_BYTES, PAGE_BYTES};
    g_model_fd = 17;
    g_model_cache_full = 1;
    g_model_needs_device_copy[0] = 1u;
    CHECK(ds4_gpu_set_stream_map(map, sizeof(map), offsets, sizes, 2u));
    CHECK(!registered && !copied);
    CHECK(g_model_host_base == map && g_model_device_base == (const char *)map);
    CHECK(g_model_registered_size == sizeof(map) && !g_model_registered && !g_model_device_owned);
    CHECK(g_model_coherent_direct_map == map && g_model_hmm_direct);
    CHECK(!g_model_range_mapping_supported && !g_model_cache_full && !g_model_needs_device_copy[0]);
    CHECK(g_model_fd_host_base == map && preserved == 1u);

    /* The eager boot plan promotes mandatory weights and leaves experts cold. */
    const ds4_unit_tensor_in tensors[] = {
        {0u, PAGE_BYTES, 0u, 1u},
        {PAGE_BYTES, MAP_BYTES - 2u * PAGE_BYTES, DS4_TCAT_ROUTED_EXPERT, 1u},
        {MAP_BYTES - PAGE_BYTES, PAGE_BYTES, 0u, 1u}
    };
    ds4_unit_compile_params params = {0};
    params.merge_gap = PAGE_BYTES;
    params.device_promote = 1u;
    params.promote_experts = g_model_needs_device_copy[0];
    ds4_phys_unit units[3];
    CHECK(ds4_units_compile(tensors, 3u, &params, units) == 3);
    CHECK(units[0].policy == DS4_UPOL_DEVICE_PROMOTE && units[2].policy == DS4_UPOL_DEVICE_PROMOTE);
    CHECK(units[1].policy == DS4_UPOL_EXPERT_COLD && units[1].allocator == DS4_UALLOC_NONE);

    coherent = 0;
    CHECK(!ds4_gpu_set_stream_map(map, sizeof(map), offsets, sizes, 2u));
    coherent = 1; device_error = 1;
    CHECK(!ds4_gpu_set_stream_map(map, sizeof(map), offsets, sizes, 2u));
    device_error = 0; source = DS4_MSRC_MAX;
    CHECK(!ds4_gpu_set_stream_map(map, sizeof(map), offsets, sizes, 2u));
    source = 0;
    sizes[1] = PAGE_BYTES + 1u;
    CHECK(!ds4_gpu_set_stream_map(map, sizeof(map), offsets, sizes, 2u));
    sizes[1] = 0u;
    CHECK(!ds4_gpu_set_stream_map(map, sizeof(map), offsets, sizes, 2u));
    sizes[1] = PAGE_BYTES; offsets[1] = 1u;
    CHECK(!ds4_gpu_set_stream_map(map, sizeof(map), offsets, sizes, 2u));
    CHECK(!ds4_gpu_set_stream_map(map, sizeof(map), offsets, sizes, 0u));
    CHECK(!ds4_gpu_set_stream_map(NULL, sizeof(map), offsets, sizes, 2u));
    CHECK(preserved == 1u && !registered && !copied);
    puts("GLM map: bounded coherent mapping, eager expert-cold policy and rejection passed");
    return 0;
}
