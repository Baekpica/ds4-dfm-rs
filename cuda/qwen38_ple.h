#ifndef DS4_QWEN38_PLE_CUDA_H
#define DS4_QWEN38_PLE_CUDA_H

#include "../ds4_ple.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_qwen38_ple_cuda ds4_qwen38_ple_cuda;

typedef struct {
    uint64_t gather_calls;
    uint64_t gathered_rows;
    uint64_t output_bytes;
    uint64_t acquire_nanoseconds_total;
    uint64_t acquire_nanoseconds_max;
    uint64_t acquire_latency_histogram[DS4_PLE_LATENCY_BUCKETS];
    /* Part of acquire_nanoseconds_total spent enqueueing tiles (descriptor
     * copy, gather launch, lease-release host function). */
    uint64_t enqueue_nanoseconds_total;
} ds4_qwen38_ple_cuda_stats;

/* Register exactly the store's bounded cache allocation as mapped pinned host
 * memory. No sidecar bytes outside that span are registered or made resident. */
ds4_qwen38_ple_cuda *ds4_qwen38_ple_cuda_create(
    ds4_ple_store *store,
    char *error,
    size_t error_size);

void ds4_qwen38_ple_cuda_destroy(ds4_qwen38_ple_cuda *context);

/* Gather token_count * 16 BF16 rows into a device output laid out as
 * [token_count, 2560]. stream_handle is a cudaStream_t passed as void * so the
 * public C header does not expose CUDA headers. Cache leases are released by a
 * stream callback only after the gather has completed. */
bool ds4_qwen38_ple_cuda_gather(
    ds4_qwen38_ple_cuda *context,
    const uint64_t *row_ids,
    size_t token_count,
    void *device_output,
    void *stream_handle,
    char *error,
    size_t error_size);

void ds4_qwen38_ple_cuda_get_stats(
    const ds4_qwen38_ple_cuda *context,
    ds4_qwen38_ple_cuda_stats *stats);

#ifdef __cplusplus
}
#endif

#endif
