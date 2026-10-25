/*
 * The CUDA device probe. Mirrors `h3_device_probe`
 * (`/data/h3.c` at `eae5d54`, `h3_device_cuda.cu:17-50`) call for call: select
 * the current device, read its properties, format the name and the compute
 * capability, take host physical memory from sysinfo(2) and GPU capacity from
 * totalGlobalMem twice. The field-by-field mapping is in `h3_device_gpu.h`.
 */
#include "h3_device_gpu.h"

#include <cuda_runtime_api.h>

#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

/* `h3_cuda_error` (`h3_device_cuda.cu:9-15`): "operation: cudaGetErrorString". */
static void h3dev_cuda_error(char *error, size_t error_size,
                             const char *operation, cudaError_t status) {
    if (!error || !error_size) {
        return;
    }
    snprintf(error, error_size, "%s: %s", operation,
             cudaGetErrorString(status));
}

extern "C" int h3dev_probe(h3dev_device_info *info, char *error,
                           size_t error_size) {
    if (!info) {
        if (error && error_size) {
            snprintf(error, error_size, "device info is required");
        }
        return 0;
    }
    memset(info, 0, sizeof(*info));

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        h3dev_cuda_error(error, error_size, "cannot select CUDA device", status);
        return 0;
    }

    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        h3dev_cuda_error(error, error_size, "cannot inspect CUDA device", status);
        return 0;
    }

    snprintf(info->name, sizeof(info->name), "%.127s", properties.name);
    snprintf(info->architecture, sizeof(info->architecture), "CUDA sm_%d%d",
             properties.major, properties.minor);

#if defined(__linux__)
    /* Whole-machine RAM, not device memory: the reference reports the host
     * figure a unified-memory machine would share (`h3_device_cuda.cu:41-45`). */
    struct sysinfo system;
    if (sysinfo(&system) == 0) {
        info->physical_memory =
            (uint64_t)system.totalram * (uint64_t)system.mem_unit;
    }
#endif

    info->recommended_working_set = (uint64_t)properties.totalGlobalMem;
    info->max_buffer_length = (uint64_t)properties.totalGlobalMem;
    info->unified_memory = properties.unifiedAddressing ? 1 : 0;
    return 1;
}
