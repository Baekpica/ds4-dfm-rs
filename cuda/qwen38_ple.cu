#include "qwen38_ple.h"

#include <cuda_runtime.h>

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const uint8_t *first;
    const uint8_t *second;
    uint32_t first_bytes;
    uint32_t reserved;
} ds4_ple_cuda_row;

typedef struct {
    ds4_ple_store *store;
    size_t row_count;
    void *host_descriptors;
    ds4_ple_row_view views[1];
} ds4_ple_cuda_leases;

#define DS4_PLE_CUDA_TILE_ROWS 256u

struct ds4_qwen38_ple_cuda {
    ds4_ple_store *store;
    uint8_t *host_base;
    uint8_t *device_base;
    size_t cache_bytes;
    bool registered;
    ds4_qwen38_ple_cuda_stats stats;
};

static bool cuda_error(
        char *error,
        size_t error_size,
        cudaError_t status,
        const char *operation) {
    if (error && error_size)
        snprintf(error, error_size, "%s: %s",
                 operation, cudaGetErrorString(status));
    return false;
}

static bool ple_cuda_error(
        char *error,
        size_t error_size,
        const char *format,
        ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, error_size, format, args);
        va_end(args);
    }
    return false;
}

static uint64_t ple_cuda_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) +
           (uint64_t)ts.tv_nsec;
}

__global__ static void qwen38_ple_gather_kernel(
        const ds4_ple_cuda_row *rows,
        uint16_t *output,
        size_t row_count) {
    const size_t row = (size_t)blockIdx.x;
    if (row >= row_count) return;
    const ds4_ple_cuda_row descriptor = rows[row];
    for (uint32_t column = threadIdx.x;
         column < DS4_PLE_ROW_DIM;
         column += blockDim.x) {
        const uint32_t byte = column * sizeof(uint16_t);
        const uint8_t *source =
            byte < descriptor.first_bytes
                ? descriptor.first + byte
                : descriptor.second +
                      (byte - descriptor.first_bytes);
        output[row * DS4_PLE_ROW_DIM + column] =
            *(const uint16_t *)source;
    }
}

static void release_leases(ds4_ple_cuda_leases *leases) {
    if (!leases) return;
    for (size_t i = 0; i < leases->row_count; i++)
        ds4_ple_store_release_row(
            leases->store, &leases->views[i]);
    free(leases->host_descriptors);
    free(leases);
}

static void CUDART_CB release_leases_callback(void *opaque) {
    release_leases((ds4_ple_cuda_leases *)opaque);
}

static ds4_ple_cuda_leases *allocate_leases(
        size_t row_capacity) {
    if (row_capacity == 0u ||
        row_capacity >
            (SIZE_MAX - sizeof(ds4_ple_cuda_leases)) /
                sizeof(ds4_ple_row_view))
        return NULL;
    const size_t bytes =
        sizeof(ds4_ple_cuda_leases) +
        (row_capacity - 1u) * sizeof(ds4_ple_row_view);
    ds4_ple_cuda_leases *leases =
        (ds4_ple_cuda_leases *)calloc(1, bytes);
    if (!leases) return NULL;
    leases->host_descriptors = calloc(
        row_capacity, sizeof(ds4_ple_cuda_row));
    if (!leases->host_descriptors) {
        free(leases);
        return NULL;
    }
    return leases;
}

static bool describe_row(
        const ds4_qwen38_ple_cuda *context,
        const ds4_ple_row_view *view,
        ds4_ple_cuda_row *descriptor,
        char *error,
        size_t error_size) {
    if (view->segment_count < 1u ||
        view->segment_count > 2u ||
        view->segment_bytes[0] == 0u ||
        (view->segment_bytes[0] & 1u) != 0u ||
        (view->segment_count == 2u &&
         (view->segment_bytes[1] == 0u ||
          (view->segment_bytes[1] & 1u) != 0u)) ||
        view->segment_bytes[0] +
                (view->segment_count == 2u
                     ? view->segment_bytes[1]
                     : 0u) !=
            DS4_PLE_ROW_BYTES)
        return ple_cuda_error(
            error, error_size,
            "PLE CUDA row view is malformed");

    const uintptr_t host_base =
        (uintptr_t)context->host_base;
    const uintptr_t first =
        (uintptr_t)view->segments[0];
    if (view->segment_bytes[0] > context->cache_bytes ||
        first < host_base ||
        first - host_base >
            context->cache_bytes - view->segment_bytes[0])
        return ple_cuda_error(
            error, error_size,
            "PLE CUDA row view is outside the registered cache");
    descriptor->first =
        context->device_base + (first - host_base);
    descriptor->first_bytes = view->segment_bytes[0];

    if (view->segment_count == 2u) {
        const uintptr_t second =
            (uintptr_t)view->segments[1];
        if (view->segment_bytes[1] > context->cache_bytes ||
            second < host_base ||
            second - host_base >
                context->cache_bytes - view->segment_bytes[1])
            return ple_cuda_error(
                error, error_size,
                "PLE CUDA second row segment is outside the cache");
        descriptor->second =
            context->device_base + (second - host_base);
    } else {
        descriptor->second =
            descriptor->first + descriptor->first_bytes;
    }
    return true;
}

static void abort_gather(
        cudaStream_t stream,
        ds4_ple_cuda_row *device_descriptors,
        ds4_ple_cuda_leases *current_leases) {
    /* Complete callbacks from earlier tiles before releasing the store-side
     * objects owned by this failed call. */
    (void)cudaStreamSynchronize(stream);
    release_leases(current_leases);
    if (device_descriptors)
        (void)cudaFree(device_descriptors);
}

ds4_qwen38_ple_cuda *ds4_qwen38_ple_cuda_create(
        ds4_ple_store *store,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!store) {
        ple_cuda_error(error, error_size,
                       "PLE CUDA store is null");
        return NULL;
    }
    void *host_base = NULL;
    size_t cache_bytes = 0;
    if (!ds4_ple_store_cache_span(
            store, &host_base, &cache_bytes)) {
        ple_cuda_error(error, error_size,
                       "PLE cache span is unavailable");
        return NULL;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        cuda_error(error, error_size, status,
                   "cudaGetDevice");
        return NULL;
    }
    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        cuda_error(error, error_size, status,
                   "cudaGetDeviceProperties");
        return NULL;
    }
    if (!properties.canMapHostMemory) {
        ple_cuda_error(error, error_size,
                       "CUDA device cannot map pinned host memory");
        return NULL;
    }

    ds4_qwen38_ple_cuda *context =
        (ds4_qwen38_ple_cuda *)calloc(1, sizeof(*context));
    if (!context) {
        ple_cuda_error(error, error_size,
                       "cannot allocate PLE CUDA context");
        return NULL;
    }
    status = cudaHostRegister(
        host_base, cache_bytes,
        cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (status != cudaSuccess) {
        cuda_error(error, error_size, status,
                   "cudaHostRegister PLE cache");
        free(context);
        return NULL;
    }
    context->registered = true;
    void *device_base = NULL;
    status = cudaHostGetDevicePointer(
        &device_base, host_base, 0);
    if (status != cudaSuccess) {
        cuda_error(error, error_size, status,
                   "cudaHostGetDevicePointer PLE cache");
        cudaHostUnregister(host_base);
        free(context);
        return NULL;
    }
    context->store = store;
    context->host_base = (uint8_t *)host_base;
    context->device_base = (uint8_t *)device_base;
    context->cache_bytes = cache_bytes;
    return context;
}

void ds4_qwen38_ple_cuda_destroy(
        ds4_qwen38_ple_cuda *context) {
    if (!context) return;
    /* Stream callbacks hold row leases and refer to the store. Destruction is
     * rare, so a device-wide drain is the unambiguous lifetime boundary. */
    (void)cudaDeviceSynchronize();
    if (context->registered)
        (void)cudaHostUnregister(context->host_base);
    free(context);
}

bool ds4_qwen38_ple_cuda_gather(
        ds4_qwen38_ple_cuda *context,
        const uint64_t *row_ids,
        size_t token_count,
        void *device_output,
        void *stream_handle,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!context || !row_ids || !device_output ||
        token_count == 0)
        return ple_cuda_error(
            error, error_size,
            "PLE CUDA gather received an invalid argument");
    if (token_count > SIZE_MAX / DS4_PLE_N_HEADS)
        return ple_cuda_error(
            error, error_size,
            "PLE CUDA row count overflows size_t");
    const size_t row_count =
        token_count * DS4_PLE_N_HEADS;
    if (row_count > UINT32_MAX)
        return ple_cuda_error(
            error, error_size,
            "PLE CUDA row count exceeds the grid limit");
    cudaStream_t stream = (cudaStream_t)stream_handle;
    const size_t descriptor_capacity =
        row_count < DS4_PLE_CUDA_TILE_ROWS
            ? row_count
            : DS4_PLE_CUDA_TILE_ROWS;
    ds4_ple_cuda_row *device_descriptors = NULL;
    cudaError_t status = cudaMallocAsync(
        (void **)&device_descriptors,
        descriptor_capacity * sizeof(*device_descriptors),
        stream);
    if (status != cudaSuccess)
        return cuda_error(
            error, error_size, status,
            "cudaMallocAsync PLE descriptors");

    const uint64_t acquire_started =
        ple_cuda_now_ns();
    uint64_t enqueue_elapsed = 0;
    size_t emitted = 0;
    while (emitted < row_count) {
        const size_t remaining = row_count - emitted;
        const size_t capacity =
            remaining < descriptor_capacity
                ? remaining
                : descriptor_capacity;
        ds4_ple_cuda_leases *leases =
            allocate_leases(capacity);
        if (!leases) {
            abort_gather(stream, device_descriptors, NULL);
            return ple_cuda_error(
                error, error_size,
                "cannot allocate PLE CUDA tile descriptors");
        }
        leases->store = context->store;
        ds4_ple_cuda_row *descriptors =
            (ds4_ple_cuda_row *)leases->host_descriptors;

        size_t tile_rows = 0;
        while (tile_rows < capacity) {
            ds4_ple_row_view *view =
                &leases->views[tile_rows];
            bool acquired = false;
            bool ok;
            if (tile_rows == 0u) {
                ok = ds4_ple_store_acquire_row(
                    context->store,
                    row_ids[emitted + tile_rows], view,
                    error, error_size);
                acquired = ok;
            } else {
                ok = ds4_ple_store_try_acquire_row(
                    context->store,
                    row_ids[emitted + tile_rows], view,
                    &acquired, error, error_size);
            }
            if (!ok) {
                abort_gather(stream, device_descriptors, leases);
                return false;
            }
            if (!acquired) break;
            leases->row_count = tile_rows + 1u;
            if (!describe_row(
                    context, view, &descriptors[tile_rows],
                    error, error_size)) {
                abort_gather(stream, device_descriptors, leases);
                return false;
            }
            tile_rows++;
        }
        if (tile_rows == 0u) {
            abort_gather(stream, device_descriptors, leases);
            return ple_cuda_error(
                error, error_size,
                "PLE CUDA tile acquisition made no progress");
        }

        /* Page workers completed their CPU writes before leases were
         * returned. Publish those writes before a mapped-host read. */
        __sync_synchronize();
        const uint64_t enqueue_started = ple_cuda_now_ns();
        status = cudaMemcpyAsync(
            device_descriptors, descriptors,
            tile_rows * sizeof(*descriptors),
            cudaMemcpyHostToDevice, stream);
        if (status == cudaSuccess) {
            qwen38_ple_gather_kernel<<<
                (uint32_t)tile_rows, 128u, 0, stream>>>(
                device_descriptors,
                (uint16_t *)device_output +
                    emitted * DS4_PLE_ROW_DIM,
                tile_rows);
            status = cudaGetLastError();
        }
        if (status == cudaSuccess)
            status = cudaLaunchHostFunc(
                stream, release_leases_callback, leases);
        if (status != cudaSuccess) {
            abort_gather(stream, device_descriptors, leases);
            return cuda_error(
                error, error_size, status,
                "enqueue PLE CUDA gather tile");
        }
        enqueue_elapsed += ple_cuda_now_ns() - enqueue_started;
        emitted += tile_rows;
    }
    const uint64_t acquire_finished =
        ple_cuda_now_ns();
    const uint64_t acquire_elapsed =
        acquire_finished >= acquire_started
            ? acquire_finished - acquire_started
            : 0;

    status = cudaFreeAsync(device_descriptors, stream);
    if (status != cudaSuccess) {
        abort_gather(stream, device_descriptors, NULL);
        return cuda_error(
            error, error_size, status,
            "cudaFreeAsync PLE descriptors");
    }

    __atomic_fetch_add(
        &context->stats.gather_calls,
        UINT64_C(1), __ATOMIC_RELAXED);
    __atomic_fetch_add(
        &context->stats.gathered_rows,
        (uint64_t)row_count, __ATOMIC_RELAXED);
    __atomic_fetch_add(
        &context->stats.output_bytes,
        (uint64_t)row_count * DS4_PLE_ROW_BYTES,
        __ATOMIC_RELAXED);
    __atomic_fetch_add(
        &context->stats.acquire_nanoseconds_total,
        acquire_elapsed, __ATOMIC_RELAXED);
    __atomic_fetch_add(
        &context->stats.enqueue_nanoseconds_total,
        enqueue_elapsed, __ATOMIC_RELAXED);
    __atomic_fetch_add(
        &context->stats.acquire_latency_histogram[
            ds4_ple_latency_bucket(acquire_elapsed)],
        UINT64_C(1), __ATOMIC_RELAXED);
    uint64_t old = __atomic_load_n(
        &context->stats.acquire_nanoseconds_max,
        __ATOMIC_RELAXED);
    while (old < acquire_elapsed &&
           !__atomic_compare_exchange_n(
               &context->stats.acquire_nanoseconds_max,
               &old, acquire_elapsed, false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    }
    return true;
}

void ds4_qwen38_ple_cuda_get_stats(
        const ds4_qwen38_ple_cuda *context,
        ds4_qwen38_ple_cuda_stats *stats) {
    if (!context || !stats) return;
    stats->gather_calls = __atomic_load_n(
        &context->stats.gather_calls, __ATOMIC_RELAXED);
    stats->gathered_rows = __atomic_load_n(
        &context->stats.gathered_rows, __ATOMIC_RELAXED);
    stats->output_bytes = __atomic_load_n(
        &context->stats.output_bytes, __ATOMIC_RELAXED);
    stats->acquire_nanoseconds_total = __atomic_load_n(
        &context->stats.acquire_nanoseconds_total,
        __ATOMIC_RELAXED);
    stats->acquire_nanoseconds_max = __atomic_load_n(
        &context->stats.acquire_nanoseconds_max,
        __ATOMIC_RELAXED);
    stats->enqueue_nanoseconds_total = __atomic_load_n(
        &context->stats.enqueue_nanoseconds_total,
        __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < DS4_PLE_LATENCY_BUCKETS; i++)
        stats->acquire_latency_histogram[i] = __atomic_load_n(
            &context->stats.acquire_latency_histogram[i],
            __ATOMIC_RELAXED);
}
