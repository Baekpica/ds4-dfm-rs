#ifndef DS4_PLE_H
#define DS4_PLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_PLE_NGRAM_SIZE 3u
#define DS4_PLE_HEADS_PER_NGRAM 8u
#define DS4_PLE_N_HEADS 16u
#define DS4_PLE_N_LOGICAL_PARTS 128u
#define DS4_PLE_N_PHYSICAL_FILES 4u
#define DS4_PLE_ROW_DIM 160u
#define DS4_PLE_ROW_BYTES 320u
#define DS4_PLE_PAGE_BYTES 4096u
#define DS4_PLE_LATENCY_BUCKETS 64u

/* Log2 buckets use an inclusive power-of-two upper bound in nanoseconds. */
static inline uint32_t ds4_ple_latency_bucket(uint64_t nanoseconds) {
    if (nanoseconds <= 1u) return 0u;
#if defined(__GNUC__) || defined(__clang__)
    const uint32_t bucket =
        64u - (uint32_t)__builtin_clzll(nanoseconds - 1u);
    return bucket < DS4_PLE_LATENCY_BUCKETS
        ? bucket : DS4_PLE_LATENCY_BUCKETS - 1u;
#else
    uint32_t bucket = 0;
    uint64_t upper = 1u;
    while (upper < nanoseconds && bucket < DS4_PLE_LATENCY_BUCKETS - 1u) {
        upper <<= 1u;
        bucket++;
    }
    return bucket;
#endif
}

static inline uint64_t ds4_ple_latency_bucket_upper_ns(uint32_t bucket) {
    return bucket >= DS4_PLE_LATENCY_BUCKETS - 1u
        ? UINT64_MAX : UINT64_C(1) << bucket;
}

typedef struct {
    uint32_t unigram_vocab_size;
    uint32_t eos_token_id;
    uint64_t layer_multipliers[DS4_PLE_NGRAM_SIZE];
    uint64_t head_vocab_sizes[DS4_PLE_N_HEADS];
    uint64_t head_offsets[DS4_PLE_N_HEADS];
} ds4_ple_hash_config;

/* Two raw preceding token ids are sufficient to reproduce the reference
 * cache state. EOS-aware masking is applied while deriving a row, not while
 * storing the history, matching Qwen4ExpTextNGramEmbedding. */
typedef struct {
    int64_t previous[DS4_PLE_NGRAM_SIZE - 1u];
} ds4_ple_hash_state;

typedef struct {
    uint32_t format_version;
    uint32_t alignment_bytes;
    uint32_t row_stride_bytes;
    uint32_t embedding_row_dimension;
    uint32_t logical_part_count;
    uint32_t physical_file_count;
    uint64_t usable_vocabulary_rows;
    uint64_t padded_vocabulary_rows;
    uint64_t total_payload_bytes;
    uint64_t total_file_bytes;
    size_t cache_bytes;
    uint32_t cache_slots;
    uint32_t worker_count;
    uint32_t direct_io_file_count;
} ds4_ple_layout;

typedef struct {
    uint64_t row_lookups;
    uint64_t logical_bytes;
    uint64_t page_requests;
    uint64_t cache_hits;
    uint64_t cache_inflight_hits;
    uint64_t cache_misses;
    uint64_t cache_evictions;
    uint64_t prefetch_dropped;
    uint64_t read_operations;
    uint64_t physical_bytes;
    uint64_t read_errors;
    uint64_t read_latency_samples;
    uint64_t read_nanoseconds_total;
    uint64_t read_nanoseconds_max;
    uint64_t read_latency_histogram[DS4_PLE_LATENCY_BUCKETS];
    uint64_t wait_samples;
    uint64_t wait_nanoseconds_total;
    uint64_t wait_nanoseconds_max;
    /* Row acquisitions that found their page still loading, and the time
     * they spent waiting for it. */
    uint64_t wait_blocked;
    uint64_t wait_blocked_nanoseconds;
    /* Completed page reads per second since the store opened (latency
     * statistics only; the last bucket absorbs the remainder). */
    uint64_t timeline_reads[256];
} ds4_ple_stats;

typedef struct ds4_ple_store ds4_ple_store;

typedef struct {
    const uint8_t *segments[2];
    uint32_t segment_bytes[2];
    uint32_t slots[2];
    uint32_t segment_count;
} ds4_ple_row_view;

bool ds4_ple_hash_config_validate(
    const ds4_ple_hash_config *config,
    char *error,
    size_t error_size);

void ds4_ple_hash_state_reset(
    ds4_ple_hash_state *state,
    const ds4_ple_hash_config *config);

/* Output is token-major and contains 16 global PLE row ids per token:
 * eight bigram heads followed by eight trigram heads. The state is updated
 * only after the complete input has been validated. */
bool ds4_ple_hash_rows(
    const ds4_ple_hash_config *config,
    ds4_ple_hash_state *state,
    const int64_t *input_ids,
    size_t token_count,
    uint64_t *row_ids,
    char *error,
    size_t error_size);

/* artifact_root is the directory containing the main GGUF shards. The
 * manifest path is relative to that root (normally ple/ple-manifest.json).
 * cache_bytes is rounded down to a sixteen-way set-associative number of 4 KiB
 * pages and is the only sidecar payload allocation made by this subsystem. */
ds4_ple_store *ds4_ple_store_open(
    const char *artifact_root,
    const char *manifest_relative_path,
    size_t cache_bytes,
    uint32_t worker_count,
    bool prefer_direct_io,
    char *error,
    size_t error_size);

void ds4_ple_store_close(ds4_ple_store *store);

const ds4_ple_layout *ds4_ple_store_layout(const ds4_ple_store *store);
const ds4_ple_hash_config *ds4_ple_store_hash_config(const ds4_ple_store *store);

/* Queue all pages touched by the supplied global row ids and return without
 * waiting for I/O. A busy cache set may drop a speculative prefetch; a later
 * blocking read will retry it. */
bool ds4_ple_store_prefetch_rows(
    ds4_ple_store *store,
    const uint64_t *row_ids,
    size_t row_count,
    char *error,
    size_t error_size);

/* Copy one 160-element BF16 row (320 bytes) from the bounded cache. */
bool ds4_ple_store_read_row(
    ds4_ple_store *store,
    uint64_t global_row,
    void *output,
    size_t output_size,
    char *error,
    size_t error_size);

/* Acquire one or two cache-page segments for a row without copying it.
 * Callers must release the view after their CPU operation or CUDA event is
 * complete; referenced slots cannot be evicted in the meantime. */
bool ds4_ple_store_acquire_row(
    ds4_ple_store *store,
    uint64_t global_row,
    ds4_ple_row_view *view,
    char *error,
    size_t error_size);

/* Non-blocking only with respect to cache-set capacity. Missing pages may
 * still wait for their already-submitted I/O. A true return with acquired
 * false means the caller must release earlier leases before retrying. */
bool ds4_ple_store_try_acquire_row(
    ds4_ple_store *store,
    uint64_t global_row,
    ds4_ple_row_view *view,
    bool *acquired,
    char *error,
    size_t error_size);

void ds4_ple_store_release_row(
    ds4_ple_store *store,
    ds4_ple_row_view *view);

/* The complete fixed-size cache allocation can be registered once with CUDA.
 * Row-view pointers are offsets into this span. */
bool ds4_ple_store_cache_span(
    ds4_ple_store *store,
    void **base,
    size_t *bytes);

void ds4_ple_store_get_stats(ds4_ple_store *store, ds4_ple_stats *stats);

#ifdef __cplusplus
}
#endif

#endif
