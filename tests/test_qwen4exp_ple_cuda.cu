#include "../cuda/qwen38_ple.h"

#include <cuda_runtime.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) +
           (uint64_t)ts.tv_nsec;
}

static void fail_cuda(cudaError_t status, const char *operation) {
    fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    exit(1);
}

static void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) fail_cuda(status, operation);
}

static void fail_ple(const char *operation, const char *error) {
    fprintf(stderr, "%s: %s\n", operation, error[0] ? error : "unknown error");
    exit(1);
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s ARTIFACT_ROOT [MANIFEST_RELATIVE_PATH]\n", argv[0]);
        return 2;
    }

    char error[512] = {0};
    const char *manifest = argc == 3 ? argv[2] : "ple/ple-manifest.json";
    const size_t cache_bytes = 32u * 1024u * 1024u;
    ds4_ple_store *store = ds4_ple_store_open(
        argv[1], manifest, cache_bytes, 8u, true,
        error, sizeof(error));
    if (!store) fail_ple("ds4_ple_store_open", error);

    ds4_qwen38_ple_cuda *cuda_context =
        ds4_qwen38_ple_cuda_create(store, error, sizeof(error));
    if (!cuda_context)
        fail_ple("ds4_qwen38_ple_cuda_create", error);

    int device = 0;
    cudaDeviceProp properties;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");

    const size_t token_count = 257u;
    const size_t row_count = token_count * DS4_PLE_N_HEADS;
    const size_t output_bytes = row_count * DS4_PLE_ROW_BYTES;
    int64_t *tokens = (int64_t *)calloc(token_count, sizeof(*tokens));
    uint64_t *row_ids = (uint64_t *)calloc(row_count, sizeof(*row_ids));
    uint8_t *reference = (uint8_t *)malloc(output_bytes);
    uint8_t *actual_first = (uint8_t *)malloc(output_bytes);
    uint8_t *actual_second = (uint8_t *)malloc(output_bytes);
    if (!tokens || !row_ids || !reference || !actual_first || !actual_second) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    const ds4_ple_hash_config *hash_config =
        ds4_ple_store_hash_config(store);
    uint64_t generator = UINT64_C(0x9368e53c2f6af274);
    for (size_t token = 0; token < token_count; token++) {
        generator = generator * UINT64_C(6364136223846793005) + 1u;
        tokens[token] = (int64_t)(generator % hash_config->unigram_vocab_size);
        if (token != 0u && token % 43u == 0u)
            tokens[token] = hash_config->eos_token_id;
    }

    ds4_ple_hash_state hash_state;
    ds4_ple_hash_state_reset(&hash_state, hash_config);
    size_t hashed = 0;
    const size_t chunk_pattern[] = {1u, 7u, 31u, 2u, 64u, 11u};
    size_t chunk_index = 0;
    while (hashed < token_count) {
        size_t chunk = chunk_pattern[
            chunk_index++ % (sizeof(chunk_pattern) / sizeof(chunk_pattern[0]))];
        if (chunk > token_count - hashed) chunk = token_count - hashed;
        if (!ds4_ple_hash_rows(
                hash_config, &hash_state, tokens + hashed, chunk,
                row_ids + hashed * DS4_PLE_N_HEADS,
                error, sizeof(error)))
            fail_ple("ds4_ple_hash_rows", error);
        hashed += chunk;
    }

    if (!ds4_ple_store_prefetch_rows(
            store, row_ids, row_count, error, sizeof(error)))
        fail_ple("ds4_ple_store_prefetch_rows", error);
    for (size_t row = 0; row < row_count; row++) {
        if (!ds4_ple_store_read_row(
                store, row_ids[row],
                reference + row * DS4_PLE_ROW_BYTES,
                DS4_PLE_ROW_BYTES, error, sizeof(error)))
            fail_ple("ds4_ple_store_read_row", error);
    }

    size_t crossing_rows = 0;
    for (size_t row = 0; row < row_count; row++) {
        ds4_ple_row_view view;
        if (!ds4_ple_store_acquire_row(
                store, row_ids[row], &view, error, sizeof(error)))
            fail_ple("ds4_ple_store_acquire_row", error);
        crossing_rows += view.segment_count == 2u;
        ds4_ple_store_release_row(store, &view);
    }

    void *device_first = NULL;
    void *device_second = NULL;
    cudaStream_t first_stream;
    cudaStream_t second_stream;
    cudaEvent_t started;
    cudaEvent_t finished;
    check_cuda(cudaMalloc(&device_first, output_bytes), "cudaMalloc first output");
    check_cuda(cudaMalloc(&device_second, output_bytes), "cudaMalloc second output");
    check_cuda(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking),
               "cudaStreamCreate first");
    check_cuda(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking),
               "cudaStreamCreate second");
    check_cuda(cudaEventCreate(&started), "cudaEventCreate started");
    check_cuda(cudaEventCreate(&finished), "cudaEventCreate finished");

    check_cuda(cudaEventRecord(started, first_stream), "cudaEventRecord started");
    const uint64_t host_started = now_ns();
    if (!ds4_qwen38_ple_cuda_gather(
            cuda_context, row_ids, token_count, device_first,
            (void *)first_stream, error, sizeof(error)))
        fail_ple("first ds4_qwen38_ple_cuda_gather", error);
    if (!ds4_qwen38_ple_cuda_gather(
            cuda_context, row_ids, token_count, device_second,
            (void *)second_stream, error, sizeof(error)))
        fail_ple("second ds4_qwen38_ple_cuda_gather", error);
    check_cuda(cudaEventRecord(finished, first_stream), "cudaEventRecord finished");
    check_cuda(cudaStreamSynchronize(first_stream), "cudaStreamSynchronize first");
    check_cuda(cudaStreamSynchronize(second_stream), "cudaStreamSynchronize second");
    const uint64_t host_finished = now_ns();

    float first_stream_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&first_stream_ms, started, finished),
               "cudaEventElapsedTime");
    check_cuda(cudaMemcpy(actual_first, device_first, output_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy first output");
    check_cuda(cudaMemcpy(actual_second, device_second, output_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy second output");

    if (memcmp(reference, actual_first, output_bytes) != 0 ||
        memcmp(reference, actual_second, output_bytes) != 0) {
        for (size_t byte = 0; byte < output_bytes; byte++) {
            if (reference[byte] != actual_first[byte] ||
                reference[byte] != actual_second[byte]) {
                fprintf(stderr,
                        "CUDA PLE mismatch at byte %zu: ref=%02x first=%02x second=%02x\n",
                        byte, reference[byte], actual_first[byte], actual_second[byte]);
                break;
            }
        }
        return 1;
    }

    ds4_qwen38_ple_cuda_stats cuda_stats;
    ds4_ple_stats store_stats;
    ds4_qwen38_ple_cuda_get_stats(cuda_context, &cuda_stats);
    ds4_ple_store_get_stats(store, &store_stats);
    uint64_t acquire_histogram_samples = 0;
    for (uint32_t i = 0; i < DS4_PLE_LATENCY_BUCKETS; i++)
        acquire_histogram_samples +=
            cuda_stats.acquire_latency_histogram[i];
    if (acquire_histogram_samples != cuda_stats.gather_calls) {
        fprintf(stderr,
                "CUDA PLE acquire histogram mismatch: samples=%" PRIu64
                " gathers=%" PRIu64 "\n",
                acquire_histogram_samples, cuda_stats.gather_calls);
        return 1;
    }
    const ds4_ple_layout *layout = ds4_ple_store_layout(store);
    struct rusage usage;
    memset(&usage, 0, sizeof(usage));
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        perror("getrusage");
        return 1;
    }
    const double host_elapsed_ms =
        (double)(host_finished - host_started) / 1.0e6;
    const double acquire_mean_us = cuda_stats.gather_calls
        ? (double)cuda_stats.acquire_nanoseconds_total /
              (double)cuda_stats.gather_calls / 1.0e3
        : 0.0;

    printf("GPU: %s (sm_%d%d)\n", properties.name,
           properties.major, properties.minor);
    printf("cache: %zu bytes, direct files: %u/%u\n",
           layout->cache_bytes, layout->direct_io_file_count,
           layout->physical_file_count);
    printf("verified: %zu tokens, %zu rows, %zu bytes, %zu page-crossing rows, 2 streams\n",
           token_count, row_count, output_bytes, crossing_rows);
    printf("timing: paired host %.3f ms, first-stream event %.3f ms, mean acquire %.3f us, max acquire %.3f us\n",
           host_elapsed_ms, first_stream_ms, acquire_mean_us,
           (double)cuda_stats.acquire_nanoseconds_max / 1.0e3);
    printf("store: reads=%" PRIu64 " physical=%" PRIu64
           " hits=%" PRIu64 " inflight=%" PRIu64
           " misses=%" PRIu64 " evictions=%" PRIu64 "\n",
           store_stats.read_operations, store_stats.physical_bytes,
           store_stats.cache_hits, store_stats.cache_inflight_hits,
           store_stats.cache_misses, store_stats.cache_evictions);
    printf("process peak RSS: %ld KiB (sidecars: %.3f GiB)\n",
           usage.ru_maxrss,
           (double)layout->total_file_bytes /
               (1024.0 * 1024.0 * 1024.0));

    /* Smallest cache the store accepts: one 16-way set.  A tile that has
     * leased all sixteen ways must end and be enqueued rather than wait
     * for a slot its own leases hold (they are released only by the
     * stream callback behind the tile); the gather must still complete
     * and match, tile by tile, through the single set. */
    const size_t minimum_cache_pages = 16u;   /* one set of the store's 16 ways */
    const size_t minimum_cache_bytes = minimum_cache_pages * DS4_PLE_PAGE_BYTES;
    ds4_ple_store *small_store = ds4_ple_store_open(
        argv[1], manifest, minimum_cache_bytes, 2u, true,
        error, sizeof(error));
    if (!small_store) fail_ple("ds4_ple_store_open (minimum cache)", error);
    ds4_qwen38_ple_cuda *small_context =
        ds4_qwen38_ple_cuda_create(small_store, error, sizeof(error));
    if (!small_context)
        fail_ple("ds4_qwen38_ple_cuda_create (minimum cache)", error);
    check_cuda(cudaMemset(device_first, 0, output_bytes),
               "cudaMemset minimum-cache output");
    const uint64_t small_started = now_ns();
    if (!ds4_qwen38_ple_cuda_gather(
            small_context, row_ids, token_count, device_first,
            (void *)first_stream, error, sizeof(error)))
        fail_ple("minimum-cache ds4_qwen38_ple_cuda_gather", error);
    check_cuda(cudaStreamSynchronize(first_stream),
               "cudaStreamSynchronize minimum cache");
    const uint64_t small_finished = now_ns();
    check_cuda(cudaMemcpy(actual_first, device_first, output_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy minimum-cache output");
    if (memcmp(reference, actual_first, output_bytes) != 0) {
        fprintf(stderr, "CUDA PLE minimum-cache gather mismatch\n");
        return 1;
    }
    ds4_qwen38_ple_cuda_stats small_stats;
    ds4_qwen38_ple_cuda_get_stats(small_context, &small_stats);
    printf("minimum cache: %zu pages, %zu rows gathered and matched in %.3f ms\n",
           minimum_cache_bytes / DS4_PLE_PAGE_BYTES,
           (size_t)small_stats.gathered_rows,
           (double)(small_finished - small_started) / 1.0e6);
    ds4_qwen38_ple_cuda_destroy(small_context);
    ds4_ple_store_close(small_store);

    check_cuda(cudaEventDestroy(started), "cudaEventDestroy started");
    check_cuda(cudaEventDestroy(finished), "cudaEventDestroy finished");
    check_cuda(cudaStreamDestroy(first_stream), "cudaStreamDestroy first");
    check_cuda(cudaStreamDestroy(second_stream), "cudaStreamDestroy second");
    check_cuda(cudaFree(device_first), "cudaFree first output");
    check_cuda(cudaFree(device_second), "cudaFree second output");
    ds4_qwen38_ple_cuda_destroy(cuda_context);
    ds4_ple_store_close(store);
    free(tokens);
    free(row_ids);
    free(reference);
    free(actual_first);
    free(actual_second);
    return 0;
}
