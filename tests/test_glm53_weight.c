/* Model-free CUDA census gate: at most 4608 bytes of cache allocations. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAP_BYTES = 256, BASE_BYTES = 1024, AUX_BYTES = 2048, EXTRA_BYTES = 1536 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM weight FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static ds4_mem_cell read_cell(int cls, int src) {
    ds4_mem_cell cell = {0};
    const int rc = src < 0
        ? ds4_gpu_mem_census_read(cls, DS4_MEMD_UNIFIED_DEVICE, &cell)
        : ds4_gpu_mem_src_census_read(src, cls, DS4_MEMD_UNIFIED_DEVICE, &cell);
    CHECK(rc == 0);
    return cell;
}

static uint64_t live(int cls, int src) {
    const ds4_mem_cell cell = read_cell(cls, src);
    return ds4_mem_cell_live(&cell);
}

static void check_freeze(void) {
    unsigned char map[MAP_BYTES];
    CHECK(ds4_gpu_init());
    const int src = ds4_gpu_model_source_bind(map, MAP_BYTES,
        DS4_MSRC_ROLE_PRIMARY, -1, DS4_RESIDENCY_HOST_MAPPED, "cache", "fixture");
    CHECK(src >= 0);
    const uint64_t faults = ds4_metrics_get()->memgov_faults;
    const uint64_t initial = live(DS4_MEMC_WEIGHT_SPAN, src);
    ds4_gpu_tensor *a = ds4_gpu_weight_alloc(map, BASE_BYTES);
    ds4_gpu_tensor *b = ds4_gpu_weight_alloc(map + MAP_BYTES / 2, EXTRA_BYTES);
    CHECK(a && b);
    ds4_gpu_model_plan_freeze();
    CHECK(ds4_metrics_get()->memgov_faults == faults);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, src) == initial + BASE_BYTES + EXTRA_BYTES);

    ds4_gpu_tensor_free(b);
    CHECK(ds4_gpu_model_source_bind(map, 1u, DS4_MSRC_ROLE_PRIMARY, -1,
        DS4_RESIDENCY_HOST_MAPPED, "cache", "fixture") == src);
    ds4_gpu_model_plan_freeze();
    CHECK(ds4_metrics_get()->memgov_faults == faults);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, src) == initial + BASE_BYTES);
    ds4_gpu_tensor_free(a);
    ds4_gpu_model_plan_freeze();
    CHECK(ds4_metrics_get()->memgov_faults == faults);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, src) == initial);
    ds4_gpu_cleanup();
    puts("GLM weight: live mapped cache freeze and source credit release passed");
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "freeze") == 0) {
        check_freeze();
        return 0;
    }
    unsigned char maps[2][MAP_BYTES], unknown[MAP_BYTES];
    CHECK(ds4_gpu_init());
    const int base = ds4_gpu_model_source_bind(maps[0], MAP_BYTES,
        DS4_MSRC_ROLE_PRIMARY, -1, DS4_RESIDENCY_HOST_MAPPED, "base", "fixture");
    const int aux = ds4_gpu_model_source_bind(maps[1], MAP_BYTES,
        DS4_MSRC_ROLE_AUXILIARY, -1, DS4_RESIDENCY_EAGER_DEVICE, "mtp", "fixture");
    CHECK(base >= 0 && aux >= 0 && base != aux);
    const uint64_t faults = ds4_gpu_mem_census_faults();
    const uint64_t all0 = live(DS4_MEMC_WEIGHT_SPAN, -1);
    const uint64_t base0 = live(DS4_MEMC_WEIGHT_SPAN, base);
    const uint64_t aux0 = live(DS4_MEMC_WEIGHT_SPAN, aux);
    const uint64_t other0 = live(DS4_MEMC_ENGINE_OTHER, -1);
    const uint64_t spare0 = live(DS4_MEMC_WEIGHT_SPAN, DS4_MSRC_MAX);
    const uint64_t diag0 = live(DS4_MEMC_DIAG, -1);

    CHECK(!ds4_gpu_weight_alloc(unknown, BASE_BYTES));
    CHECK(!ds4_gpu_weight_alloc(maps[0], 0u));
    ds4_gpu_mem_scope_begin(DS4_MEMC_DIAG);
    ds4_gpu_tensor *a = ds4_gpu_weight_alloc(maps[0], BASE_BYTES);
    ds4_gpu_tensor *b = ds4_gpu_weight_alloc(maps[1], AUX_BYTES);
    ds4_gpu_tensor *c = ds4_gpu_weight_alloc(maps[0] + MAP_BYTES / 2, EXTRA_BYTES);
    ds4_gpu_tensor *ordinary = ds4_gpu_tensor_alloc(sizeof(int32_t));
    ds4_gpu_mem_scope_end();
    CHECK(a && b && c && ordinary);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, -1) == all0 + BASE_BYTES + AUX_BYTES + EXTRA_BYTES);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, base) == base0 + BASE_BYTES + EXTRA_BYTES);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, aux) == aux0 + AUX_BYTES);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, DS4_MSRC_MAX) == spare0);
    CHECK(live(DS4_MEMC_ENGINE_OTHER, -1) == other0);
    CHECK(live(DS4_MEMC_DIAG, -1) == diag0 + sizeof(int32_t));

    ds4_gpu_tensor *view = ds4_gpu_tensor_view(a, 0u, sizeof(int32_t));
    CHECK(view);
    ds4_gpu_tensor_free(view);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, base) == base0 + BASE_BYTES + EXTRA_BYTES);
    /* The interior allocation's source must survive a later extent rebind. */
    CHECK(ds4_gpu_model_source_bind(maps[0], 1u, DS4_MSRC_ROLE_PRIMARY, -1,
        DS4_RESIDENCY_HOST_MAPPED, "base", "fixture") == base);
    ds4_gpu_tensor_free(c);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(ordinary);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, -1) == all0);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, base) == base0);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, aux) == aux0);
    CHECK(live(DS4_MEMC_WEIGHT_SPAN, DS4_MSRC_MAX) == spare0);
    CHECK(live(DS4_MEMC_ENGINE_OTHER, -1) == other0);
    CHECK(live(DS4_MEMC_DIAG, -1) == diag0);
    CHECK(ds4_gpu_mem_census_faults() == faults);
    ds4_gpu_cleanup();
    puts("GLM weight: global/source census, view, scope and free passed");
    return 0;
}
