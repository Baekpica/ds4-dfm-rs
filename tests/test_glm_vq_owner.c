/* Adjacent independent matrices share a host page. Resolve them from the
 * funded resident plan before serving, without per-matrix registration. */
#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_glmvq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { MATRIX_DIM = 128, MATRICES = 2, ALIGNMENT = 128, BF16_TYPE = 30 };

int main(int argc, char **argv) {
    const int cold = argc == 2 && strcmp(argv[1], "--cold") == 0;
    const uint64_t matrix = MATRIX_DIM * MATRIX_DIM * sizeof(uint16_t);
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t bytes = (ALIGNMENT + MATRICES * matrix + page - 1u) / page * page;
    uint8_t *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED || !ds4_gpu_init()) { return 1; }
    for (unsigned i = 0; i < MATRICES * MATRIX_DIM * MATRIX_DIM; i++) {
        ((uint16_t *)(map + ALIGNMENT))[i] = 0x3f80u;
    }
    if (mprotect(map, bytes, PROT_READ) ||
        ds4_gpu_model_source_bind(map, bytes, DS4_MSRC_ROLE_PRIMARY, -1,
            DS4_RESIDENCY_EAGER_DEVICE, "base", "GVQ adjacent matrices") < 0 ||
        !ds4_gpu_set_model_map_range(map, bytes, 0u, bytes, matrix)) { return 1; }

    ds4_gvq_record records[MATRICES];
    ds4_unit_tensor_in tensors[MATRICES];
    for (unsigned i = 0; i < MATRICES; i++) {
        const uint64_t offset = ALIGNMENT + i * matrix;
        records[i] = (ds4_gvq_record){offset, matrix, BF16_TYPE, MATRIX_DIM, MATRIX_DIM,
            DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK, DS4_GVQ_NO_LINK};
        tensors[i] = (ds4_unit_tensor_in){offset, matrix, DS4_TCAT_ROUTED_EXPERT, 1u};
    }
    ds4_unit_compile_params params = {0};
    params.max_unit_bytes = bytes;
    params.vmm_granularity = 256u;
    params.device_promote = 1u;
    params.promote_experts = !cold;
    ds4_phys_unit units[MATRICES];
    const int count = ds4_units_compile(tensors, MATRICES, &params, units);
    uint64_t populated = 0u;
    if (count <= 0 || ds4_units_verify(tensors, MATRICES, &params, units, count) ||
        ds4_gpu_model_units_bind(map, units, (uint32_t)count) ||
        !ds4_gpu_materialize_model_plan(map, bytes, &populated)) { return 1; }
    ds4_gvq_model *model = ds4_gpu_gvq_bind(map, bytes, records, MATRICES);
    const int ok = model && populated >= MATRICES * matrix;
    ds4_gpu_gvq_unbind(model);
    ds4_gpu_unregister_model_map(map);
    ds4_gpu_cleanup();
    munmap(map, bytes);
    printf("GLM VQ adjacent owner: bound=%d, populated=%llu, %s\n",
        model != NULL, (unsigned long long)populated, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
