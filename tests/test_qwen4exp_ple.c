#include "../ds4_ple.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ASSERT(expr)                                                     \
    do {                                                                      \
        if (!(expr)) {                                                        \
            fprintf(stderr, "%s:%d: assertion failed: %s\n",                 \
                    __FILE__, __LINE__, #expr);                               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

typedef struct {
    ds4_ple_store *store;
    const uint64_t *rows;
    const uint8_t *expected;
    size_t row_count;
    uint32_t worker_index;
    bool failed;
    char error[256];
} stress_context;

static void *stress_reader(void *opaque) {
    stress_context *context = opaque;
    uint8_t row[DS4_PLE_ROW_BYTES];
    for (uint32_t iteration = 0; iteration < 1024u; iteration++) {
        const size_t index =
            ((size_t)context->worker_index * 131u +
             (size_t)iteration * 197u) % context->row_count;
        if (!ds4_ple_store_read_row(
                context->store, context->rows[index],
                row, sizeof(row),
                context->error, sizeof(context->error)) ||
            memcmp(row,
                   context->expected + index * DS4_PLE_ROW_BYTES,
                   DS4_PLE_ROW_BYTES) != 0) {
            context->failed = true;
            return NULL;
        }
    }
    return NULL;
}

static void check_concurrent_reads(
        ds4_ple_store *direct,
        ds4_ple_store *reference,
        uint64_t usable_rows) {
    enum { N_ROWS = 512, N_WORKERS = 8 };
    uint64_t *rows = malloc(N_ROWS * sizeof(*rows));
    uint8_t *expected = malloc(N_ROWS * DS4_PLE_ROW_BYTES);
    TEST_ASSERT(rows && expected);
    char error[256] = {0};
    uint64_t value = UINT64_C(0x51454e38504c4501);
    for (uint32_t i = 0; i < N_ROWS; i++) {
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        rows[i] = (value * UINT64_C(0x2545f4914f6cdd1d)) %
                  usable_rows;
        TEST_ASSERT(ds4_ple_store_read_row(
            reference, rows[i],
            expected + (size_t)i * DS4_PLE_ROW_BYTES,
            DS4_PLE_ROW_BYTES, error, sizeof(error)));
    }
    TEST_ASSERT(ds4_ple_store_prefetch_rows(
        direct, rows, N_ROWS, error, sizeof(error)));

    pthread_t workers[N_WORKERS];
    stress_context contexts[N_WORKERS];
    memset(contexts, 0, sizeof(contexts));
    for (uint32_t i = 0; i < N_WORKERS; i++) {
        contexts[i].store = direct;
        contexts[i].rows = rows;
        contexts[i].expected = expected;
        contexts[i].row_count = N_ROWS;
        contexts[i].worker_index = i;
        TEST_ASSERT(pthread_create(
            &workers[i], NULL, stress_reader, &contexts[i]) == 0);
    }
    for (uint32_t i = 0; i < N_WORKERS; i++) {
        TEST_ASSERT(pthread_join(workers[i], NULL) == 0);
        if (contexts[i].failed)
            fprintf(stderr, "concurrent PLE reader %u failed: %s\n",
                    i, contexts[i].error);
        TEST_ASSERT(!contexts[i].failed);
    }
    free(expected);
    free(rows);
}

static void check_hash_chunking(const ds4_ple_hash_config *config) {
    const int64_t tokens[] = {
        101, 202, 248044, 303, 404, 405, 248044, 7, 8, 9,
    };
    const size_t n_tokens = sizeof(tokens) / sizeof(tokens[0]);
    uint64_t whole[n_tokens * DS4_PLE_N_HEADS];
    uint64_t chunked[n_tokens * DS4_PLE_N_HEADS];
    ds4_ple_hash_state whole_state;
    ds4_ple_hash_state chunked_state;
    char error[256] = {0};
    ds4_ple_hash_state_reset(&whole_state, config);
    ds4_ple_hash_state_reset(&chunked_state, config);
    TEST_ASSERT(ds4_ple_hash_rows(config, &whole_state, tokens, n_tokens,
                                  whole, error, sizeof(error)));
    TEST_ASSERT(ds4_ple_hash_rows(config, &chunked_state, tokens, 1,
                                  chunked, error, sizeof(error)));
    TEST_ASSERT(ds4_ple_hash_rows(config, &chunked_state, tokens + 1, 4,
                                  chunked + DS4_PLE_N_HEADS,
                                  error, sizeof(error)));
    TEST_ASSERT(ds4_ple_hash_rows(config, &chunked_state, tokens + 5,
                                  n_tokens - 5,
                                  chunked + 5 * DS4_PLE_N_HEADS,
                                  error, sizeof(error)));
    TEST_ASSERT(memcmp(whole, chunked, sizeof(whole)) == 0);
    TEST_ASSERT(memcmp(&whole_state, &chunked_state,
                       sizeof(whole_state)) == 0);

    ds4_ple_hash_state before = whole_state;
    const int64_t invalid = config->unigram_vocab_size;
    uint64_t ignored[DS4_PLE_N_HEADS];
    TEST_ASSERT(!ds4_ple_hash_rows(config, &whole_state, &invalid, 1,
                                   ignored, error, sizeof(error)));
    TEST_ASSERT(memcmp(&whole_state, &before, sizeof(before)) == 0);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <Qwen SSD-PLE artifact root>\n",
                argv[0]);
        return 2;
    }
    TEST_ASSERT(setenv("DS4_PLE_LATENCY_STATS", "1", 1) == 0);

    char error[512] = {0};
    ds4_ple_store *direct = ds4_ple_store_open(
        argv[1], "ple/ple-manifest.json", 1024 * 1024, 4, true,
        error, sizeof(error));
    if (!direct) {
        fprintf(stderr, "direct-preferred store open failed: %s\n", error);
        return 1;
    }
    ds4_ple_store *buffered = ds4_ple_store_open(
        argv[1], "ple/ple-manifest.json", 1024 * 1024, 2, false,
        error, sizeof(error));
    if (!buffered) {
        fprintf(stderr, "buffered store open failed: %s\n", error);
        ds4_ple_store_close(direct);
        return 1;
    }

    const ds4_ple_layout *layout = ds4_ple_store_layout(direct);
    TEST_ASSERT(layout);
    TEST_ASSERT(layout->format_version == 1);
    TEST_ASSERT(layout->logical_part_count == 128);
    TEST_ASSERT(layout->physical_file_count == 4);
    TEST_ASSERT(layout->usable_vocabulary_rows == UINT64_C(320001446));
    TEST_ASSERT(layout->padded_vocabulary_rows == UINT64_C(320001536));
    TEST_ASSERT(layout->total_payload_bytes == UINT64_C(102400491520));
    TEST_ASSERT(layout->total_file_bytes == UINT64_C(102400786432));
    TEST_ASSERT(layout->cache_bytes == 1024 * 1024);
    TEST_ASSERT(layout->cache_slots == 256);

    const ds4_ple_hash_config *config =
        ds4_ple_store_hash_config(direct);
    TEST_ASSERT(config);
    check_hash_chunking(config);

    const uint64_t rows[] = {
        0, 1, 12, 13, UINT64_C(2500011), UINT64_C(2500012),
        UINT64_C(80000106), UINT64_C(160000374),
        UINT64_C(317501524), UINT64_C(320001445),
    };
    const size_t n_rows = sizeof(rows) / sizeof(rows[0]);
    uint8_t direct_row[DS4_PLE_ROW_BYTES];
    uint8_t buffered_row[DS4_PLE_ROW_BYTES];

    TEST_ASSERT(ds4_ple_store_prefetch_rows(
        direct, rows, n_rows, error, sizeof(error)));
    for (size_t i = 0; i < n_rows; i++) {
        TEST_ASSERT(ds4_ple_store_read_row(
            direct, rows[i], direct_row, sizeof(direct_row),
            error, sizeof(error)));
        TEST_ASSERT(ds4_ple_store_read_row(
            buffered, rows[i], buffered_row, sizeof(buffered_row),
            error, sizeof(error)));
        TEST_ASSERT(memcmp(direct_row, buffered_row,
                           sizeof(direct_row)) == 0);
    }
    TEST_ASSERT(ds4_ple_store_read_row(
        direct, rows[2], direct_row, sizeof(direct_row),
        error, sizeof(error)));
    TEST_ASSERT(!ds4_ple_store_read_row(
        direct, layout->usable_vocabulary_rows,
        direct_row, sizeof(direct_row), error, sizeof(error)));

    ds4_ple_stats stats = {0};
    ds4_ple_store_get_stats(direct, &stats);
    TEST_ASSERT(stats.row_lookups == n_rows + 1);
    TEST_ASSERT(stats.logical_bytes ==
                (n_rows + 1) * DS4_PLE_ROW_BYTES);
    TEST_ASSERT(stats.read_operations > 0);
    TEST_ASSERT(stats.physical_bytes ==
                stats.read_operations * DS4_PLE_PAGE_BYTES);
    TEST_ASSERT(stats.read_errors == 0);
    TEST_ASSERT(stats.cache_hits + stats.cache_inflight_hits > 0);
    uint64_t read_histogram_samples = 0;
    for (uint32_t i = 0; i < DS4_PLE_LATENCY_BUCKETS; i++)
        read_histogram_samples += stats.read_latency_histogram[i];
    TEST_ASSERT(stats.read_latency_samples == stats.read_operations);
    TEST_ASSERT(read_histogram_samples == stats.read_latency_samples);
    TEST_ASSERT(stats.read_nanoseconds_total >= stats.read_nanoseconds_max);
    TEST_ASSERT(stats.read_nanoseconds_max > 0);

    check_concurrent_reads(
        direct, buffered, layout->usable_vocabulary_rows);
    ds4_ple_store_get_stats(direct, &stats);
    TEST_ASSERT(stats.row_lookups == n_rows + 1 + 8 * 1024);
    TEST_ASSERT(stats.read_errors == 0);

    printf("Qwen3.8-Flash-Next PLE store: valid "
           "(cache=%zu bytes, direct_files=%u/4, reads=%" PRIu64
           ", physical_bytes=%" PRIu64 ", hits=%" PRIu64
           ", inflight_hits=%" PRIu64 ", misses=%" PRIu64 ")\n",
           layout->cache_bytes, layout->direct_io_file_count,
           stats.read_operations, stats.physical_bytes,
           stats.cache_hits, stats.cache_inflight_hits,
           stats.cache_misses);

    ds4_ple_store_close(buffered);
    ds4_ple_store_close(direct);
    return 0;
}
