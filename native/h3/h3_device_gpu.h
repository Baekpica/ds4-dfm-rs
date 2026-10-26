/*
 * h3dev_probe — the CUDA device identity the h3 host reports.
 *
 * Additive MiniMax-H3 family: a new name, no existing family's device query is
 * touched. The reference is `/data/h3.c` at `eae5d54`: `h3_device_probe`
 * (`h3_device_cuda.cu:17-50`) filling `h3_device_info` (`h3.h:130-141`), the
 * two-sided contract `h3_device.h:10` and the shape `h3.c:456-463` uses when a
 * context is opened.
 *
 * Every field is the reference's, in the reference's units:
 *   name, architecture        snprintf("%.127s") of `properties.name` and
 *                             "CUDA sm_%d%d" (`h3_device_cuda.cu:38-40`)
 *   physical_memory           sysinfo(2) totalram * mem_unit, 0 if the call
 *                             fails (`:41-45`)
 *   recommended_working_set   properties.totalGlobalMem (`:46`)
 *   max_buffer_length         properties.totalGlobalMem (`:47`)
 *   unified_memory            properties.unifiedAddressing (`:48`)
 *
 * Deviations, named: `apple_gpu_family` and `metal4` are Metal-only fields of
 * the shared struct, so the CUDA probe leaves them zero exactly as the
 * reference does (`h3_device_cuda.cu:23` memsets the struct). `sysinfo(2)` is
 * Linux-only; on another host `physical_memory` stays 0 rather than being
 * approximated from a different call.
 */
#ifndef H3D_DEVICE_GPU_H
#define H3D_DEVICE_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { H3D_DEVICE_NAME_SIZE = 128 };

typedef struct {
    char name[H3D_DEVICE_NAME_SIZE];
    char architecture[H3D_DEVICE_NAME_SIZE];
    uint64_t physical_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    int apple_gpu_family;
    int metal4;
    int unified_memory;
} h3dev_device_info;

/* Same contract as `h3_device_probe`: 1 on success, 0 with `error` filled. */
int h3dev_probe(h3dev_device_info *info, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
