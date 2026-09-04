#include "ds4_ple.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

/* A prefill chunk's rows land on pages spread uniformly over the sets, and
 * every page a set cannot hold becomes a blocking read when the row is
 * gathered.  Four ways overflowed ~7 % of the sets for an 8,192-token chunk
 * in a 1 GiB cache (~6K stalls per chunk); sixteen ways overflow <1 %. */
#define DS4_PLE_CACHE_WAYS 16u
#define DS4_PLE_JSON_MAX_DEPTH 64u
#define DS4_PLE_PATH_CAP 256u

typedef struct {
    uint32_t logical_part;
    uint32_t physical_file_index;
    uint64_t global_row_start;
    uint64_t rows;
    uint64_t file_offset;
    uint64_t payload_bytes;
    char physical_file[DS4_PLE_PATH_CAP];
} ds4_ple_logical_part;

typedef struct {
    uint32_t index;
    uint64_t file_bytes;
    uint64_t payload_bytes;
    char path[DS4_PLE_PATH_CAP];
    int fd;
    bool direct_io;
} ds4_ple_physical_file;

typedef enum {
    DS4_PLE_PAGE_EMPTY = 0,
    DS4_PLE_PAGE_LOADING,
    DS4_PLE_PAGE_READY,
    DS4_PLE_PAGE_ERROR,
} ds4_ple_page_state;

typedef struct {
    uint32_t file_index;
    uint64_t page_offset;
    uint64_t generation;
    uint64_t last_access;
    uint32_t refcount;
    int error_number;
    ds4_ple_page_state state;
    /* A reader sleeps on state_cond until this page leaves LOADING. */
    bool awaited;
    uint8_t *data;
} ds4_ple_cache_slot;

typedef struct {
    uint32_t slot;
    uint64_t generation;
} ds4_ple_work_item;

struct ds4_ple_store {
    ds4_ple_layout layout;
    ds4_ple_hash_config hash_config;
    ds4_ple_logical_part logical[DS4_PLE_N_LOGICAL_PARTS];
    ds4_ple_physical_file physical[DS4_PLE_N_PHYSICAL_FILES];

    void *cache_memory;
    ds4_ple_cache_slot *slots;
    uint32_t set_count;
    uint64_t access_clock;

    ds4_ple_work_item *queue;
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_count;

    pthread_t *workers;
    uint32_t workers_started;
    /* Threads sleeping on state_cond for any slot of a set to free up. */
    uint32_t set_waiters;
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t state_cond;
    bool mutex_ready;
    bool work_cond_ready;
    bool state_cond_ready;
    bool stopping;
    bool latency_stats;
    uint64_t opened_ns;
    ds4_ple_stats stats;
};

typedef struct {
    const char *cursor;
    const char *end;
    char *error;
    size_t error_size;
} ds4_ple_json;

static bool ple_error(char *error, size_t error_size, const char *format, ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, error_size, format, args);
        va_end(args);
    }
    return false;
}

static uint64_t ple_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

bool ds4_ple_hash_config_validate(
        const ds4_ple_hash_config *config,
        char *error,
        size_t error_size) {
    if (!config) return ple_error(error, error_size, "PLE hash config is null");
    if (config->unigram_vocab_size == 0)
        return ple_error(error, error_size, "PLE unigram vocabulary is zero");
    if (config->eos_token_id >= config->unigram_vocab_size)
        return ple_error(error, error_size, "PLE EOS token is outside the vocabulary");

    const uint64_t max_token = (uint64_t)config->unigram_vocab_size - 1u;
    for (uint32_t i = 0; i < DS4_PLE_NGRAM_SIZE; i++) {
        const uint64_t multiplier = config->layer_multipliers[i];
        if (multiplier == 0 || (multiplier & 1u) == 0)
            return ple_error(error, error_size, "PLE multiplier %u is not positive and odd", i);
        if (max_token && multiplier > (uint64_t)INT64_MAX / max_token)
            return ple_error(error, error_size, "PLE multiplier %u can overflow signed int64", i);
    }

    uint64_t next_offset = 0;
    for (uint32_t i = 0; i < DS4_PLE_N_HEADS; i++) {
        if (config->head_vocab_sizes[i] == 0)
            return ple_error(error, error_size, "PLE head %u has an empty vocabulary", i);
        if (config->head_offsets[i] != next_offset)
            return ple_error(error, error_size, "PLE head %u has a non-contiguous offset", i);
        if (UINT64_MAX - next_offset < config->head_vocab_sizes[i])
            return ple_error(error, error_size, "PLE head vocabulary sum overflows");
        next_offset += config->head_vocab_sizes[i];
    }
    return true;
}

void ds4_ple_hash_state_reset(
        ds4_ple_hash_state *state,
        const ds4_ple_hash_config *config) {
    if (!state || !config) return;
    state->previous[0] = (int64_t)config->eos_token_id;
    state->previous[1] = (int64_t)config->eos_token_id;
}

bool ds4_ple_hash_rows(
        const ds4_ple_hash_config *config,
        ds4_ple_hash_state *state,
        const int64_t *input_ids,
        size_t token_count,
        uint64_t *row_ids,
        char *error,
        size_t error_size) {
    if (!state) return ple_error(error, error_size, "PLE hash state is null");
    if (token_count && (!input_ids || !row_ids))
        return ple_error(error, error_size, "PLE hash input or output is null");
    if (!ds4_ple_hash_config_validate(config, error, error_size)) return false;

    /* Validate before mutating state so callers can retry a rejected batch. */
    for (size_t i = 0; i < token_count; i++) {
        if (input_ids[i] < 0 || (uint64_t)input_ids[i] >= config->unigram_vocab_size)
            return ple_error(error, error_size,
                             "PLE token %zu (%" PRId64 ") is outside the vocabulary",
                             i, input_ids[i]);
    }

    int64_t older = state->previous[0];
    int64_t newer = state->previous[1];
    const int64_t eos = (int64_t)config->eos_token_id;
    for (size_t token = 0; token < token_count; token++) {
        const uint64_t current = (uint64_t)input_ids[token];
        const uint64_t previous_1 = (uint64_t)newer;
        /* A two-token shift cannot cross the immediately preceding EOS. The
         * older value itself may be EOS, which already gives the reference
         * padding value without another branch. */
        const uint64_t previous_2 = newer == eos ? (uint64_t)eos : (uint64_t)older;
        const uint64_t bigram =
            current * config->layer_multipliers[0] ^
            previous_1 * config->layer_multipliers[1];
        const uint64_t trigram =
            bigram ^ previous_2 * config->layer_multipliers[2];

        for (uint32_t head = 0; head < DS4_PLE_HEADS_PER_NGRAM; head++) {
            row_ids[token * DS4_PLE_N_HEADS + head] =
                bigram % config->head_vocab_sizes[head] + config->head_offsets[head];
        }
        for (uint32_t local = 0; local < DS4_PLE_HEADS_PER_NGRAM; local++) {
            const uint32_t head = DS4_PLE_HEADS_PER_NGRAM + local;
            row_ids[token * DS4_PLE_N_HEADS + head] =
                trigram % config->head_vocab_sizes[head] + config->head_offsets[head];
        }
        older = newer;
        newer = (int64_t)current;
    }
    state->previous[0] = older;
    state->previous[1] = newer;
    return true;
}

static void json_fail(ds4_ple_json *json, const char *format, ...) {
    if (!json->error || !json->error_size || json->error[0]) return;
    va_list args;
    va_start(args, format);
    vsnprintf(json->error, json->error_size, format, args);
    va_end(args);
}

static void json_ws(ds4_ple_json *json) {
    while (json->cursor < json->end &&
           (*json->cursor == ' ' || *json->cursor == '\n' ||
            *json->cursor == '\r' || *json->cursor == '\t'))
        json->cursor++;
}

static bool json_take(ds4_ple_json *json, char value) {
    json_ws(json);
    if (json->cursor >= json->end || *json->cursor != value) {
        json_fail(json, "PLE manifest JSON expected '%c'", value);
        return false;
    }
    json->cursor++;
    return true;
}

static bool json_string(
        ds4_ple_json *json,
        const char **start,
        size_t *length,
        bool allow_escapes) {
    json_ws(json);
    if (json->cursor >= json->end || *json->cursor != '"') {
        json_fail(json, "PLE manifest JSON expected a string");
        return false;
    }
    const char *p = ++json->cursor;
    const char *begin = p;
    bool escaped = false;
    while (p < json->end) {
        const unsigned char ch = (unsigned char)*p;
        if (ch < 0x20u) {
            json_fail(json, "PLE manifest JSON contains a control byte in a string");
            return false;
        }
        if (*p == '"') {
            if (escaped && !allow_escapes) {
                json_fail(json, "PLE manifest required strings may not contain escapes");
                return false;
            }
            *start = begin;
            *length = (size_t)(p - begin);
            json->cursor = p + 1;
            return true;
        }
        if (*p == '\\') {
            escaped = true;
            p++;
            if (p >= json->end) break;
            if (*p == 'u') {
                if ((size_t)(json->end - p) < 5u) break;
                p += 5;
                continue;
            }
        }
        p++;
    }
    json_fail(json, "PLE manifest JSON contains an unterminated string");
    return false;
}

static bool json_key_eq(const char *key, size_t length, const char *expected) {
    const size_t n = strlen(expected);
    return length == n && memcmp(key, expected, n) == 0;
}

static bool json_u64(ds4_ple_json *json, uint64_t *value) {
    json_ws(json);
    if (json->cursor >= json->end || *json->cursor < '0' || *json->cursor > '9') {
        json_fail(json, "PLE manifest JSON expected an unsigned integer");
        return false;
    }
    uint64_t result = 0;
    while (json->cursor < json->end &&
           *json->cursor >= '0' && *json->cursor <= '9') {
        const uint32_t digit = (uint32_t)(*json->cursor - '0');
        if (result > (UINT64_MAX - digit) / 10u) {
            json_fail(json, "PLE manifest integer overflows uint64");
            return false;
        }
        result = result * 10u + digit;
        json->cursor++;
    }
    *value = result;
    return true;
}

static bool json_skip_value(ds4_ple_json *json, uint32_t depth);

static bool json_skip_object(ds4_ple_json *json, uint32_t depth) {
    if (!json_take(json, '{')) return false;
    json_ws(json);
    if (json->cursor < json->end && *json->cursor == '}') {
        json->cursor++;
        return true;
    }
    for (;;) {
        const char *key = NULL;
        size_t length = 0;
        if (!json_string(json, &key, &length, true) || !json_take(json, ':') ||
            !json_skip_value(json, depth + 1u))
            return false;
        json_ws(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            return true;
        }
        if (!json_take(json, ',')) return false;
    }
}

static bool json_skip_array(ds4_ple_json *json, uint32_t depth) {
    if (!json_take(json, '[')) return false;
    json_ws(json);
    if (json->cursor < json->end && *json->cursor == ']') {
        json->cursor++;
        return true;
    }
    for (;;) {
        if (!json_skip_value(json, depth + 1u)) return false;
        json_ws(json);
        if (json->cursor < json->end && *json->cursor == ']') {
            json->cursor++;
            return true;
        }
        if (!json_take(json, ',')) return false;
    }
}

static bool json_skip_value(ds4_ple_json *json, uint32_t depth) {
    if (depth > DS4_PLE_JSON_MAX_DEPTH) {
        json_fail(json, "PLE manifest JSON nesting is too deep");
        return false;
    }
    json_ws(json);
    if (json->cursor >= json->end) {
        json_fail(json, "PLE manifest JSON ended inside a value");
        return false;
    }
    if (*json->cursor == '{') return json_skip_object(json, depth);
    if (*json->cursor == '[') return json_skip_array(json, depth);
    if (*json->cursor == '"') {
        const char *start = NULL;
        size_t length = 0;
        return json_string(json, &start, &length, true);
    }
    static const char *literals[] = {"true", "false", "null"};
    for (uint32_t i = 0; i < 3u; i++) {
        const size_t n = strlen(literals[i]);
        if ((size_t)(json->end - json->cursor) >= n &&
            memcmp(json->cursor, literals[i], n) == 0) {
            json->cursor += n;
            return true;
        }
    }
    const char *p = json->cursor;
    if (*p == '-') p++;
    bool digit = false;
    while (p < json->end && *p >= '0' && *p <= '9') {
        digit = true;
        p++;
    }
    if (p < json->end && *p == '.') {
        p++;
        while (p < json->end && *p >= '0' && *p <= '9') {
            digit = true;
            p++;
        }
    }
    if (p < json->end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < json->end && (*p == '+' || *p == '-')) p++;
        while (p < json->end && *p >= '0' && *p <= '9') {
            digit = true;
            p++;
        }
    }
    if (!digit) {
        json_fail(json, "PLE manifest JSON contains an invalid value");
        return false;
    }
    json->cursor = p;
    return true;
}

static bool json_copy_string(
        ds4_ple_json *json,
        char *output,
        size_t output_size) {
    const char *start = NULL;
    size_t length = 0;
    if (!json_string(json, &start, &length, false)) return false;
    if (length + 1u > output_size) {
        json_fail(json, "PLE manifest string exceeds its destination");
        return false;
    }
    memcpy(output, start, length);
    output[length] = 0;
    return true;
}

static bool json_u64_array(
        ds4_ple_json *json,
        uint64_t *values,
        uint32_t expected) {
    if (!json_take(json, '[')) return false;
    for (uint32_t i = 0; i < expected; i++) {
        if (i && !json_take(json, ',')) return false;
        if (!json_u64(json, &values[i])) return false;
    }
    return json_take(json, ']');
}

static bool manifest_parse_logical_part(
        ds4_ple_json *json,
        ds4_ple_logical_part *part) {
    uint32_t seen = 0;
    if (!json_take(json, '{')) return false;
    for (;;) {
        json_ws(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            break;
        }
        const char *key = NULL;
        size_t key_len = 0;
        if (!json_string(json, &key, &key_len, false) || !json_take(json, ':')) return false;
        uint64_t value = 0;
        uint32_t bit = 0;
        if (json_key_eq(key, key_len, "logical_part")) {
            bit = 1u << 0;
            if (!json_u64(json, &value) || value > UINT32_MAX) return false;
            part->logical_part = (uint32_t)value;
        } else if (json_key_eq(key, key_len, "physical_file_index")) {
            bit = 1u << 1;
            if (!json_u64(json, &value) || value > UINT32_MAX) return false;
            part->physical_file_index = (uint32_t)value;
        } else if (json_key_eq(key, key_len, "global_row_start")) {
            bit = 1u << 2;
            if (!json_u64(json, &part->global_row_start)) return false;
        } else if (json_key_eq(key, key_len, "rows")) {
            bit = 1u << 3;
            if (!json_u64(json, &part->rows)) return false;
        } else if (json_key_eq(key, key_len, "file_offset")) {
            bit = 1u << 4;
            if (!json_u64(json, &part->file_offset)) return false;
        } else if (json_key_eq(key, key_len, "payload_bytes")) {
            bit = 1u << 5;
            if (!json_u64(json, &part->payload_bytes)) return false;
        } else if (json_key_eq(key, key_len, "row_stride_bytes")) {
            bit = 1u << 6;
            if (!json_u64(json, &value) || value != DS4_PLE_ROW_BYTES) return false;
        } else if (json_key_eq(key, key_len, "embedding_row_dimension")) {
            bit = 1u << 7;
            if (!json_u64(json, &value) || value != DS4_PLE_ROW_DIM) return false;
        } else if (json_key_eq(key, key_len, "physical_file")) {
            bit = 1u << 8;
            if (!json_copy_string(json, part->physical_file, sizeof(part->physical_file))) return false;
        } else if (!json_skip_value(json, 0)) {
            return false;
        }
        if (bit) {
            if (seen & bit) {
                json_fail(json, "PLE manifest logical part contains a duplicate key");
                return false;
            }
            seen |= bit;
        }
        json_ws(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            break;
        }
        if (!json_take(json, ',')) return false;
    }
    if (seen != ((1u << 9) - 1u)) {
        json_fail(json, "PLE manifest logical part is missing a required field");
        return false;
    }
    return true;
}

static bool manifest_parse_logical_parts(ds4_ple_json *json, ds4_ple_store *store) {
    if (!json_take(json, '[')) return false;
    for (uint32_t i = 0; i < DS4_PLE_N_LOGICAL_PARTS; i++) {
        if (i && !json_take(json, ',')) return false;
        if (!manifest_parse_logical_part(json, &store->logical[i])) return false;
    }
    return json_take(json, ']');
}

static bool manifest_parse_physical_file(
        ds4_ple_json *json,
        ds4_ple_physical_file *file) {
    uint32_t seen = 0;
    if (!json_take(json, '{')) return false;
    for (;;) {
        json_ws(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            break;
        }
        const char *key = NULL;
        size_t key_len = 0;
        if (!json_string(json, &key, &key_len, false) || !json_take(json, ':')) return false;
        uint64_t value = 0;
        uint32_t bit = 0;
        if (json_key_eq(key, key_len, "index")) {
            bit = 1u << 0;
            if (!json_u64(json, &value) || value > UINT32_MAX) return false;
            file->index = (uint32_t)value;
        } else if (json_key_eq(key, key_len, "path")) {
            bit = 1u << 1;
            if (!json_copy_string(json, file->path, sizeof(file->path))) return false;
        } else if (json_key_eq(key, key_len, "file_bytes")) {
            bit = 1u << 2;
            if (!json_u64(json, &file->file_bytes)) return false;
        } else if (json_key_eq(key, key_len, "payload_bytes")) {
            bit = 1u << 3;
            if (!json_u64(json, &file->payload_bytes)) return false;
        } else if (!json_skip_value(json, 0)) {
            return false;
        }
        if (bit) {
            if (seen & bit) {
                json_fail(json, "PLE manifest physical file contains a duplicate key");
                return false;
            }
            seen |= bit;
        }
        json_ws(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            break;
        }
        if (!json_take(json, ',')) return false;
    }
    if (seen != ((1u << 4) - 1u)) {
        json_fail(json, "PLE manifest physical file is missing a required field");
        return false;
    }
    return true;
}

static bool manifest_parse_physical_files(ds4_ple_json *json, ds4_ple_store *store) {
    if (!json_take(json, '[')) return false;
    for (uint32_t i = 0; i < DS4_PLE_N_PHYSICAL_FILES; i++) {
        if (i && !json_take(json, ',')) return false;
        if (!manifest_parse_physical_file(json, &store->physical[i])) return false;
    }
    return json_take(json, ']');
}

static bool manifest_parse_hash_reference(ds4_ple_json *json) {
    static const char *expected =
        "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459";
    bool saw_sha = false;
    if (!json_take(json, '{')) return false;
    for (;;) {
        json_ws(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            break;
        }
        const char *key = NULL;
        size_t key_len = 0;
        if (!json_string(json, &key, &key_len, false) || !json_take(json, ':')) return false;
        if (json_key_eq(key, key_len, "implementation_sha256")) {
            char value[80];
            if (saw_sha || !json_copy_string(json, value, sizeof(value))) return false;
            if (strcmp(value, expected) != 0) {
                json_fail(json, "PLE manifest pins an unsupported hash implementation");
                return false;
            }
            saw_sha = true;
        } else if (!json_skip_value(json, 0)) {
            return false;
        }
        json_ws(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            break;
        }
        if (!json_take(json, ',')) return false;
    }
    if (!saw_sha) json_fail(json, "PLE manifest hash reference is missing its source SHA-256");
    return saw_sha;
}

enum {
    TOP_FORMAT = UINT64_C(1) << 0,
    TOP_VARIANT = UINT64_C(1) << 1,
    TOP_BYTE_ORDER = UINT64_C(1) << 2,
    TOP_STORAGE = UINT64_C(1) << 3,
    TOP_DTYPE = UINT64_C(1) << 4,
    TOP_ALIGNMENT = UINT64_C(1) << 5,
    TOP_ROW_STRIDE = UINT64_C(1) << 6,
    TOP_ROW_DIM = UINT64_C(1) << 7,
    TOP_NGRAM = UINT64_C(1) << 8,
    TOP_HEADS_PER = UINT64_C(1) << 9,
    TOP_N_HEADS = UINT64_C(1) << 10,
    TOP_N_LAYERS = UINT64_C(1) << 11,
    TOP_LOGICAL_COUNT = UINT64_C(1) << 12,
    TOP_PHYSICAL_COUNT = UINT64_C(1) << 13,
    TOP_USABLE_ROWS = UINT64_C(1) << 14,
    TOP_PADDED_ROWS = UINT64_C(1) << 15,
    TOP_TOTAL_PAYLOAD = UINT64_C(1) << 16,
    TOP_TOTAL_FILE = UINT64_C(1) << 17,
    TOP_MULTIPLIERS = UINT64_C(1) << 18,
    TOP_VOCABS = UINT64_C(1) << 19,
    TOP_OFFSETS = UINT64_C(1) << 20,
    TOP_LOGICAL = UINT64_C(1) << 21,
    TOP_PHYSICAL = UINT64_C(1) << 22,
    TOP_HASH_REFERENCE = UINT64_C(1) << 23,
};

static bool manifest_seen(ds4_ple_json *json, uint64_t *seen, uint64_t bit) {
    if (*seen & bit) {
        json_fail(json, "PLE manifest contains a duplicate required top-level key");
        return false;
    }
    *seen |= bit;
    return true;
}

static bool manifest_expect_string(ds4_ple_json *json, const char *expected) {
    char value[128];
    if (!json_copy_string(json, value, sizeof(value))) return false;
    if (strcmp(value, expected) != 0) {
        json_fail(json, "PLE manifest expected string '%s'", expected);
        return false;
    }
    return true;
}

static bool manifest_expect_qwen_variant(ds4_ple_json *json) {
    char value[128];
    if (!json_copy_string(json, value, sizeof(value))) return false;
    if (strcmp(value, "MQ-Q5-SSD-PLE-BF16") != 0 &&
        strcmp(value, "MQ-Q6-SSD-PLE-BF16") != 0) {
        json_fail(json, "PLE manifest has an unsupported artifact variant");
        return false;
    }
    return true;
}

static bool manifest_expect_u64(ds4_ple_json *json, uint64_t expected) {
    uint64_t value = 0;
    if (!json_u64(json, &value)) return false;
    if (value != expected) {
        json_fail(json, "PLE manifest expected integer %" PRIu64 ", got %" PRIu64,
                  expected, value);
        return false;
    }
    return true;
}

static bool manifest_parse(ds4_ple_store *store, const char *data, size_t size,
                           char *error, size_t error_size) {
    ds4_ple_json json = {
        .cursor = data,
        .end = data + size,
        .error = error,
        .error_size = error_size,
    };
    uint64_t seen = 0;
    if (error && error_size) error[0] = 0;
    if (!json_take(&json, '{')) return false;
    for (;;) {
        json_ws(&json);
        if (json.cursor < json.end && *json.cursor == '}') {
            json.cursor++;
            break;
        }
        const char *key = NULL;
        size_t key_len = 0;
        if (!json_string(&json, &key, &key_len, false) || !json_take(&json, ':')) return false;
        uint64_t bit = 0;
        bool ok = true;
        if (json_key_eq(key, key_len, "format_version")) {
            bit = TOP_FORMAT;
            ok = manifest_expect_u64(&json, 1u);
            store->layout.format_version = 1u;
        } else if (json_key_eq(key, key_len, "artifact_variant")) {
            bit = TOP_VARIANT;
            ok = manifest_expect_qwen_variant(&json);
        } else if (json_key_eq(key, key_len, "byte_order")) {
            bit = TOP_BYTE_ORDER;
            ok = manifest_expect_string(&json, "little");
        } else if (json_key_eq(key, key_len, "storage")) {
            bit = TOP_STORAGE;
            ok = manifest_expect_string(&json, "ssd_backed_bounded_page_cache");
        } else if (json_key_eq(key, key_len, "storage_dtype")) {
            bit = TOP_DTYPE;
            ok = manifest_expect_string(&json, "BF16");
        } else if (json_key_eq(key, key_len, "alignment_bytes")) {
            bit = TOP_ALIGNMENT;
            ok = manifest_expect_u64(&json, DS4_PLE_PAGE_BYTES);
            store->layout.alignment_bytes = DS4_PLE_PAGE_BYTES;
        } else if (json_key_eq(key, key_len, "row_stride_bytes")) {
            bit = TOP_ROW_STRIDE;
            ok = manifest_expect_u64(&json, DS4_PLE_ROW_BYTES);
            store->layout.row_stride_bytes = DS4_PLE_ROW_BYTES;
        } else if (json_key_eq(key, key_len, "embedding_row_dimension")) {
            bit = TOP_ROW_DIM;
            ok = manifest_expect_u64(&json, DS4_PLE_ROW_DIM);
            store->layout.embedding_row_dimension = DS4_PLE_ROW_DIM;
        } else if (json_key_eq(key, key_len, "ngram_size")) {
            bit = TOP_NGRAM;
            ok = manifest_expect_u64(&json, DS4_PLE_NGRAM_SIZE);
        } else if (json_key_eq(key, key_len, "heads_per_ngram")) {
            bit = TOP_HEADS_PER;
            ok = manifest_expect_u64(&json, DS4_PLE_HEADS_PER_NGRAM);
        } else if (json_key_eq(key, key_len, "number_of_ngram_heads")) {
            bit = TOP_N_HEADS;
            ok = manifest_expect_u64(&json, DS4_PLE_N_HEADS);
        } else if (json_key_eq(key, key_len, "number_of_ple_layers")) {
            bit = TOP_N_LAYERS;
            ok = manifest_expect_u64(&json, 1u);
        } else if (json_key_eq(key, key_len, "logical_shard_count")) {
            bit = TOP_LOGICAL_COUNT;
            ok = manifest_expect_u64(&json, DS4_PLE_N_LOGICAL_PARTS);
            store->layout.logical_part_count = DS4_PLE_N_LOGICAL_PARTS;
        } else if (json_key_eq(key, key_len, "physical_file_count")) {
            bit = TOP_PHYSICAL_COUNT;
            ok = manifest_expect_u64(&json, DS4_PLE_N_PHYSICAL_FILES);
            store->layout.physical_file_count = DS4_PLE_N_PHYSICAL_FILES;
        } else if (json_key_eq(key, key_len, "usable_vocabulary_rows")) {
            bit = TOP_USABLE_ROWS;
            ok = json_u64(&json, &store->layout.usable_vocabulary_rows);
        } else if (json_key_eq(key, key_len, "padded_vocabulary_rows")) {
            bit = TOP_PADDED_ROWS;
            ok = json_u64(&json, &store->layout.padded_vocabulary_rows);
        } else if (json_key_eq(key, key_len, "total_payload_bytes")) {
            bit = TOP_TOTAL_PAYLOAD;
            ok = json_u64(&json, &store->layout.total_payload_bytes);
        } else if (json_key_eq(key, key_len, "total_file_bytes_including_alignment")) {
            bit = TOP_TOTAL_FILE;
            ok = json_u64(&json, &store->layout.total_file_bytes);
        } else if (json_key_eq(key, key_len, "layer_multipliers")) {
            bit = TOP_MULTIPLIERS;
            ok = json_u64_array(&json, store->hash_config.layer_multipliers,
                                DS4_PLE_NGRAM_SIZE);
        } else if (json_key_eq(key, key_len, "per_head_vocabulary_sizes")) {
            bit = TOP_VOCABS;
            ok = json_u64_array(&json, store->hash_config.head_vocab_sizes,
                                DS4_PLE_N_HEADS);
        } else if (json_key_eq(key, key_len, "per_head_offsets")) {
            bit = TOP_OFFSETS;
            ok = json_u64_array(&json, store->hash_config.head_offsets,
                                DS4_PLE_N_HEADS);
        } else if (json_key_eq(key, key_len, "logical_parts")) {
            bit = TOP_LOGICAL;
            ok = manifest_parse_logical_parts(&json, store);
        } else if (json_key_eq(key, key_len, "physical_files")) {
            bit = TOP_PHYSICAL;
            ok = manifest_parse_physical_files(&json, store);
        } else if (json_key_eq(key, key_len, "hash_reference")) {
            bit = TOP_HASH_REFERENCE;
            ok = manifest_parse_hash_reference(&json);
        } else {
            ok = json_skip_value(&json, 0);
        }
        if (!ok || (bit && !manifest_seen(&json, &seen, bit))) return false;
        json_ws(&json);
        if (json.cursor < json.end && *json.cursor == '}') {
            json.cursor++;
            break;
        }
        if (!json_take(&json, ',')) return false;
    }
    json_ws(&json);
    if (json.cursor != json.end) return ple_error(error, error_size, "PLE manifest has trailing data");
    const uint64_t required = (UINT64_C(1) << 24) - 1u;
    if (seen != required) return ple_error(error, error_size, "PLE manifest is missing required top-level fields");

    store->hash_config.unigram_vocab_size = 248320u;
    store->hash_config.eos_token_id = 248044u;
    return ds4_ple_hash_config_validate(&store->hash_config, error, error_size);
}

static bool path_is_safe_relative(const char *path) {
    if (!path || !path[0] || path[0] == '/') return false;
    const char *segment = path;
    for (const char *p = path;; p++) {
        if (*p != '/' && *p != 0) continue;
        const size_t n = (size_t)(p - segment);
        if (n == 0 || (n == 1 && segment[0] == '.') ||
            (n == 2 && segment[0] == '.' && segment[1] == '.'))
            return false;
        if (*p == 0) break;
        segment = p + 1;
    }
    return true;
}

static char *path_join(const char *root, const char *relative) {
    const size_t root_len = strlen(root);
    const size_t relative_len = strlen(relative);
    const bool slash = root_len && root[root_len - 1u] != '/';
    if (root_len > SIZE_MAX - relative_len - (slash ? 2u : 1u)) return NULL;
    char *result = malloc(root_len + relative_len + (slash ? 2u : 1u));
    if (!result) return NULL;
    memcpy(result, root, root_len);
    size_t at = root_len;
    if (slash) result[at++] = '/';
    memcpy(result + at, relative, relative_len + 1u);
    return result;
}

static bool read_small_file(const char *path, char **data, size_t *size,
                            char *error, size_t error_size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return ple_error(error, error_size, "cannot open PLE manifest '%s': %s",
                         path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0 || st.st_size > 16 * 1024 * 1024) {
        const int saved = errno;
        close(fd);
        return ple_error(error, error_size, "PLE manifest has an invalid size: %s",
                         saved ? strerror(saved) : "outside 1..16 MiB");
    }
    char *buffer = malloc((size_t)st.st_size);
    if (!buffer) {
        close(fd);
        return ple_error(error, error_size, "cannot allocate PLE manifest buffer");
    }
    size_t done = 0;
    while (done < (size_t)st.st_size) {
        ssize_t got = read(fd, buffer + done, (size_t)st.st_size - done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            const int saved = errno;
            free(buffer);
            close(fd);
            return ple_error(error, error_size, "cannot read PLE manifest: %s",
                             got == 0 ? "unexpected EOF" : strerror(saved));
        }
        done += (size_t)got;
    }
    close(fd);
    *data = buffer;
    *size = done;
    return true;
}

static bool manifest_validate_layout(ds4_ple_store *store,
                                     char *error, size_t error_size) {
    if (store->layout.usable_vocabulary_rows != UINT64_C(320001446) ||
        store->layout.padded_vocabulary_rows != UINT64_C(320001536) ||
        store->layout.total_payload_bytes != UINT64_C(102400491520) ||
        store->layout.total_file_bytes != UINT64_C(102400786432))
        return ple_error(error, error_size,
                         "PLE manifest aggregate sizes do not match the BF16 reference");

    uint64_t row_cursor = 0;
    uint64_t payload_sum = 0;
    for (uint32_t i = 0; i < DS4_PLE_N_LOGICAL_PARTS; i++) {
        ds4_ple_logical_part *part = &store->logical[i];
        if (part->logical_part != i || part->global_row_start != row_cursor ||
            part->rows == 0 || part->physical_file_index >= DS4_PLE_N_PHYSICAL_FILES)
            return ple_error(error, error_size,
                             "PLE logical part %u has an invalid mapping", i);
        if ((part->file_offset % DS4_PLE_PAGE_BYTES) != 0)
            return ple_error(error, error_size,
                             "PLE logical part %u is not page aligned", i);
        if (part->rows > UINT64_MAX / DS4_PLE_ROW_BYTES ||
            part->payload_bytes != part->rows * DS4_PLE_ROW_BYTES)
            return ple_error(error, error_size,
                             "PLE logical part %u has an invalid payload", i);
        if (UINT64_MAX - row_cursor < part->rows ||
            UINT64_MAX - payload_sum < part->payload_bytes)
            return ple_error(error, error_size,
                             "PLE logical part aggregate overflows");
        row_cursor += part->rows;
        payload_sum += part->payload_bytes;
    }
    if (row_cursor != store->layout.padded_vocabulary_rows ||
        payload_sum != store->layout.total_payload_bytes)
        return ple_error(error, error_size,
                         "PLE logical parts do not match manifest aggregates");

    uint64_t file_sum = 0;
    uint64_t file_payload_sum = 0;
    for (uint32_t i = 0; i < DS4_PLE_N_PHYSICAL_FILES; i++) {
        ds4_ple_physical_file *file = &store->physical[i];
        file->fd = -1;
        if (file->index != i || !path_is_safe_relative(file->path) ||
            file->file_bytes == 0 ||
            (file->file_bytes % DS4_PLE_PAGE_BYTES) != 0)
            return ple_error(error, error_size,
                             "PLE physical file %u has an invalid descriptor", i);
        if (UINT64_MAX - file_sum < file->file_bytes ||
            UINT64_MAX - file_payload_sum < file->payload_bytes)
            return ple_error(error, error_size,
                             "PLE physical file aggregate overflows");
        file_sum += file->file_bytes;
        file_payload_sum += file->payload_bytes;
    }
    if (file_sum != store->layout.total_file_bytes ||
        file_payload_sum != store->layout.total_payload_bytes)
        return ple_error(error, error_size,
                         "PLE physical files do not match manifest aggregates");

    for (uint32_t i = 0; i < DS4_PLE_N_LOGICAL_PARTS; i++) {
        ds4_ple_logical_part *part = &store->logical[i];
        ds4_ple_physical_file *file =
            &store->physical[part->physical_file_index];
        if (strcmp(part->physical_file, file->path) != 0 ||
            part->file_offset > file->file_bytes ||
            part->payload_bytes > file->file_bytes - part->file_offset)
            return ple_error(error, error_size,
                             "PLE logical part %u exceeds its physical file", i);
    }
    const uint64_t last_head_end =
        store->hash_config.head_offsets[DS4_PLE_N_HEADS - 1u] +
        store->hash_config.head_vocab_sizes[DS4_PLE_N_HEADS - 1u];
    if (last_head_end != store->layout.usable_vocabulary_rows)
        return ple_error(error, error_size,
                         "PLE head vocabularies do not match usable rows");
    return true;
}

static bool store_open_files(ds4_ple_store *store, const char *root,
                             bool prefer_direct, char *error,
                             size_t error_size) {
    for (uint32_t i = 0; i < DS4_PLE_N_PHYSICAL_FILES; i++) {
        ds4_ple_physical_file *file = &store->physical[i];
        char *path = path_join(root, file->path);
        if (!path)
            return ple_error(error, error_size,
                             "cannot allocate PLE sidecar path");
        int fd = -1;
#ifdef O_DIRECT
        if (prefer_direct) {
            fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
            if (fd >= 0) file->direct_io = true;
        }
#else
        (void)prefer_direct;
#endif
        if (fd < 0) {
            fd = open(path, O_RDONLY | O_CLOEXEC);
            file->direct_io = false;
        }
        if (fd < 0) {
            const int saved = errno;
            free(path);
            return ple_error(error, error_size,
                             "cannot open PLE sidecar %u: %s",
                             i, strerror(saved));
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 0 ||
            (uint64_t)st.st_size != file->file_bytes) {
            const int saved = errno;
            close(fd);
            free(path);
            return ple_error(error, error_size,
                             "PLE sidecar %u has the wrong file size: %s",
                             i, saved ? strerror(saved) : "size mismatch");
        }
#ifdef POSIX_FADV_RANDOM
        if (!file->direct_io)
            (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
        file->fd = fd;
        if (file->direct_io) store->layout.direct_io_file_count++;
        free(path);
    }
    return true;
}

static uint64_t cache_hash(uint32_t file_index, uint64_t page_offset) {
    uint64_t value = page_offset / DS4_PLE_PAGE_BYTES;
    value ^= (uint64_t)file_index * UINT64_C(0x9E3779B97F4A7C15);
    value ^= value >> 30;
    value *= UINT64_C(0xBF58476D1CE4E5B9);
    value ^= value >> 27;
    value *= UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

static bool cache_queue_locked(ds4_ple_store *store, uint32_t slot) {
    if (store->queue_count >= store->layout.cache_slots) return false;
    store->queue[store->queue_tail].slot = slot;
    store->queue[store->queue_tail].generation =
        store->slots[slot].generation;
    store->queue_tail =
        (store->queue_tail + 1u) % store->layout.cache_slots;
    store->queue_count++;
    pthread_cond_signal(&store->work_cond);
    return true;
}

/* hold=true returns a referenced slot. wait_for_slot=false is used by
 * speculative prefetch and may drop a colliding request without blocking. */
static bool cache_request_page(
        ds4_ple_store *store,
        uint32_t file_index,
        uint64_t page_offset,
        bool hold,
        bool wait_for_slot,
        uint32_t *slot_out,
        char *error,
        size_t error_size) {
    if (file_index >= DS4_PLE_N_PHYSICAL_FILES ||
        (page_offset % DS4_PLE_PAGE_BYTES) != 0 ||
        page_offset >
            store->physical[file_index].file_bytes - DS4_PLE_PAGE_BYTES)
        return ple_error(error, error_size,
                         "PLE cache request is outside a sidecar");

    pthread_mutex_lock(&store->mutex);
    for (;;) {
        store->stats.page_requests++;
        const uint32_t set =
            (uint32_t)(cache_hash(file_index, page_offset) %
                       store->set_count);
        const uint32_t first = set * DS4_PLE_CACHE_WAYS;
        int32_t victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t way = 0; way < DS4_PLE_CACHE_WAYS; way++) {
            const uint32_t index = first + way;
            ds4_ple_cache_slot *slot = &store->slots[index];
            if (slot->state != DS4_PLE_PAGE_EMPTY &&
                slot->file_index == file_index &&
                slot->page_offset == page_offset) {
                /* An I/O error is not a permanent negative cache entry.
                 * Recycle it once the failed holder has observed the error so
                 * a transient device condition can be retried. */
                if (slot->state == DS4_PLE_PAGE_ERROR &&
                    slot->refcount == 0) {
                    victim = (int32_t)index;
                    oldest = 0;
                    continue;
                }
                slot->last_access = ++store->access_clock;
                if (slot->state == DS4_PLE_PAGE_READY)
                    store->stats.cache_hits++;
                else if (slot->state == DS4_PLE_PAGE_LOADING)
                    store->stats.cache_inflight_hits++;
                else
                    store->stats.cache_misses++;
                if (hold) slot->refcount++;
                *slot_out = index;
                pthread_mutex_unlock(&store->mutex);
                return true;
            }
            if (slot->state == DS4_PLE_PAGE_EMPTY) {
                victim = (int32_t)index;
                oldest = 0;
            } else if (slot->state != DS4_PLE_PAGE_LOADING &&
                       slot->refcount == 0 &&
                       slot->last_access < oldest) {
                /* Least recently touched among the evictable ways.  The
                 * earlier first-fit choice (stop at the first evictable
                 * way) evicted pages a prefetch had just brought in for
                 * the same set, so a chunk re-read up to a third of its
                 * pages at gather time. */
                victim = (int32_t)index;
                oldest = slot->last_access;
            }
        }
        if (victim >= 0) {
            ds4_ple_cache_slot *slot = &store->slots[victim];
            if (slot->state != DS4_PLE_PAGE_EMPTY)
                store->stats.cache_evictions++;
            slot->file_index = file_index;
            slot->page_offset = page_offset;
            slot->generation++;
            slot->last_access = ++store->access_clock;
            slot->refcount = hold ? 1u : 0u;
            slot->error_number = 0;
            slot->state = DS4_PLE_PAGE_LOADING;
            store->stats.cache_misses++;
            if (!cache_queue_locked(store, (uint32_t)victim)) {
                slot->state = DS4_PLE_PAGE_EMPTY;
                slot->refcount = 0;
                pthread_mutex_unlock(&store->mutex);
                return ple_error(error, error_size,
                                 "PLE page work queue is full");
            }
            *slot_out = (uint32_t)victim;
            pthread_mutex_unlock(&store->mutex);
            return true;
        }
        if (!wait_for_slot) {
            store->stats.prefetch_dropped++;
            pthread_mutex_unlock(&store->mutex);
            *slot_out = UINT32_MAX;
            return true;
        }
        store->set_waiters++;
        pthread_cond_wait(&store->state_cond, &store->mutex);
        store->set_waiters--;
        if (store->stopping) {
            pthread_mutex_unlock(&store->mutex);
            return ple_error(error, error_size, "PLE store is stopping");
        }
    }
}

static bool cache_wait_ready(ds4_ple_store *store, uint32_t slot_index,
                             const uint8_t **data, char *error,
                             size_t error_size) {
    pthread_mutex_lock(&store->mutex);
    ds4_ple_cache_slot *slot = &store->slots[slot_index];
    if (slot->state == DS4_PLE_PAGE_LOADING) {
        /* Workers wake this sleeper only for the awaited page.  Waking it
         * on every completed read (the previous behaviour) made the reader
         * and the sixteen workers convoy on the store mutex, cutting the
         * effective read rate to a small fraction of the device's. */
        const uint64_t blocked_at = ple_now_ns();
        store->stats.wait_blocked++;
        slot->awaited = true;
        while (slot->state == DS4_PLE_PAGE_LOADING && !store->stopping)
            pthread_cond_wait(&store->state_cond, &store->mutex);
        slot->awaited = false;
        store->stats.wait_blocked_nanoseconds += ple_now_ns() - blocked_at;
    }
    if (slot->state != DS4_PLE_PAGE_READY) {
        const int saved = slot->error_number;
        if (slot->refcount) slot->refcount--;
        if (store->set_waiters) pthread_cond_broadcast(&store->state_cond);
        pthread_mutex_unlock(&store->mutex);
        return ple_error(error, error_size,
                         "PLE sidecar page read failed: %s",
                         saved ? strerror(saved) : "store stopped");
    }
    *data = slot->data;
    pthread_mutex_unlock(&store->mutex);
    return true;
}

static void cache_release(ds4_ple_store *store, uint32_t slot_index) {
    pthread_mutex_lock(&store->mutex);
    ds4_ple_cache_slot *slot = &store->slots[slot_index];
    if (slot->refcount) slot->refcount--;
    if (store->set_waiters) pthread_cond_broadcast(&store->state_cond);
    pthread_mutex_unlock(&store->mutex);
}

static void *cache_worker(void *opaque) {
    ds4_ple_store *store = opaque;
    for (;;) {
        pthread_mutex_lock(&store->mutex);
        while (!store->stopping && store->queue_count == 0)
            pthread_cond_wait(&store->work_cond, &store->mutex);
        if (store->stopping) {
            pthread_mutex_unlock(&store->mutex);
            return NULL;
        }
        const ds4_ple_work_item item =
            store->queue[store->queue_head];
        store->queue_head =
            (store->queue_head + 1u) % store->layout.cache_slots;
        store->queue_count--;
        ds4_ple_cache_slot *slot = &store->slots[item.slot];
        const uint32_t file_index = slot->file_index;
        const uint64_t page_offset = slot->page_offset;
        uint8_t *data = slot->data;
        pthread_mutex_unlock(&store->mutex);

        const uint64_t read_started =
            store->latency_stats ? ple_now_ns() : 0;
        ssize_t got;
        do {
            got = pread(store->physical[file_index].fd, data,
                        DS4_PLE_PAGE_BYTES, (off_t)page_offset);
        } while (got < 0 && errno == EINTR);
        const uint64_t read_finished =
            store->latency_stats ? ple_now_ns() : 0;
        const uint64_t read_elapsed =
            read_finished >= read_started
                ? read_finished - read_started : 0;
        const int saved =
            got == (ssize_t)DS4_PLE_PAGE_BYTES
                ? 0
                : (got < 0 ? errno : EIO);
#ifdef POSIX_FADV_DONTNEED
        if (!store->physical[file_index].direct_io)
            (void)posix_fadvise(store->physical[file_index].fd,
                                (off_t)page_offset,
                                DS4_PLE_PAGE_BYTES,
                                POSIX_FADV_DONTNEED);
#endif

        pthread_mutex_lock(&store->mutex);
        slot = &store->slots[item.slot];
        if (slot->generation == item.generation &&
            slot->state == DS4_PLE_PAGE_LOADING) {
            store->stats.read_operations++;
            if (store->latency_stats) {
                uint64_t second =
                    (read_finished - store->opened_ns) / UINT64_C(1000000000);
                if (second > 255u) second = 255u;
                store->stats.timeline_reads[second]++;
                store->stats.read_latency_samples++;
                store->stats.read_nanoseconds_total += read_elapsed;
                if (read_elapsed > store->stats.read_nanoseconds_max)
                    store->stats.read_nanoseconds_max = read_elapsed;
                store->stats.read_latency_histogram[
                    ds4_ple_latency_bucket(read_elapsed)]++;
            }
            if (saved == 0) {
                slot->state = DS4_PLE_PAGE_READY;
                store->stats.physical_bytes += DS4_PLE_PAGE_BYTES;
            } else {
                slot->state = DS4_PLE_PAGE_ERROR;
                slot->error_number = saved;
                store->stats.read_errors++;
            }
        }
        if (slot->awaited || store->set_waiters)
            pthread_cond_broadcast(&store->state_cond);
        pthread_mutex_unlock(&store->mutex);
    }
}

static bool resolve_row(ds4_ple_store *store, uint64_t global_row,
                        uint32_t *file_index, uint64_t *file_offset,
                        char *error, size_t error_size) {
    if (global_row >= store->layout.usable_vocabulary_rows)
        return ple_error(error, error_size,
                         "PLE row %" PRIu64
                         " is outside the usable vocabulary",
                         global_row);
    uint32_t low = 0;
    uint32_t high = DS4_PLE_N_LOGICAL_PARTS;
    while (low + 1u < high) {
        const uint32_t mid = low + (high - low) / 2u;
        if (store->logical[mid].global_row_start <= global_row)
            low = mid;
        else
            high = mid;
    }
    ds4_ple_logical_part *part = &store->logical[low];
    const uint64_t local = global_row - part->global_row_start;
    if (local >= part->rows)
        return ple_error(error, error_size,
                         "PLE row mapping has a gap");
    *file_index = part->physical_file_index;
    *file_offset =
        part->file_offset + local * DS4_PLE_ROW_BYTES;
    return true;
}

static bool prefetch_one_row(ds4_ple_store *store, uint64_t row,
                             char *error, size_t error_size) {
    uint32_t file_index = 0;
    uint64_t offset = 0;
    if (!resolve_row(store, row, &file_index, &offset,
                     error, error_size))
        return false;
    const uint64_t page0 =
        offset & ~(uint64_t)(DS4_PLE_PAGE_BYTES - 1u);
    const uint32_t within = (uint32_t)(offset - page0);
    uint32_t ignored = 0;
    if (!cache_request_page(store, file_index, page0, false, false,
                            &ignored, error, error_size))
        return false;
    if (within + DS4_PLE_ROW_BYTES > DS4_PLE_PAGE_BYTES) {
        if (!cache_request_page(store, file_index,
                                page0 + DS4_PLE_PAGE_BYTES,
                                false, false, &ignored,
                                error, error_size))
            return false;
    }
    return true;
}

static void store_destroy(ds4_ple_store *store) {
    if (!store) return;
    if (store->mutex_ready) {
        pthread_mutex_lock(&store->mutex);
        store->stopping = true;
        if (store->work_cond_ready)
            pthread_cond_broadcast(&store->work_cond);
        if (store->state_cond_ready)
            pthread_cond_broadcast(&store->state_cond);
        pthread_mutex_unlock(&store->mutex);
    }
    for (uint32_t i = 0; i < store->workers_started; i++)
        pthread_join(store->workers[i], NULL);
    for (uint32_t i = 0; i < DS4_PLE_N_PHYSICAL_FILES; i++)
        if (store->physical[i].fd >= 0)
            close(store->physical[i].fd);
    if (store->state_cond_ready)
        pthread_cond_destroy(&store->state_cond);
    if (store->work_cond_ready)
        pthread_cond_destroy(&store->work_cond);
    if (store->mutex_ready)
        pthread_mutex_destroy(&store->mutex);
    free(store->workers);
    free(store->queue);
    free(store->slots);
    free(store->cache_memory);
    free(store);
}

ds4_ple_store *ds4_ple_store_open(
        const char *artifact_root,
        const char *manifest_relative_path,
        size_t cache_bytes,
        uint32_t worker_count,
        bool prefer_direct_io,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!artifact_root || !artifact_root[0] ||
        !path_is_safe_relative(manifest_relative_path)) {
        ple_error(error, error_size,
                  "PLE artifact root or manifest path is invalid");
        return NULL;
    }
    if (worker_count == 0 || worker_count > 64u) {
        ple_error(error, error_size,
                  "PLE worker count must be in 1..64");
        return NULL;
    }
    const size_t requested_slots =
        cache_bytes / DS4_PLE_PAGE_BYTES;
    const size_t slots =
        requested_slots -
        requested_slots % DS4_PLE_CACHE_WAYS;
    if (slots < DS4_PLE_CACHE_WAYS || slots > UINT32_MAX) {
        ple_error(error, error_size,
                  "PLE cache must contain 4..UINT32_MAX pages");
        return NULL;
    }

    ds4_ple_store *store = calloc(1, sizeof(*store));
    if (!store) {
        ple_error(error, error_size,
                  "cannot allocate PLE store");
        return NULL;
    }
    const char *latency_stats = getenv("DS4_PLE_LATENCY_STATS");
    store->latency_stats =
        latency_stats && strcmp(latency_stats, "0") != 0;
    store->opened_ns = ple_now_ns();
    for (uint32_t i = 0; i < DS4_PLE_N_PHYSICAL_FILES; i++)
        store->physical[i].fd = -1;

    char *manifest_path =
        path_join(artifact_root, manifest_relative_path);
    char *manifest_data = NULL;
    size_t manifest_size = 0;
    if (!manifest_path ||
        !read_small_file(manifest_path,
                         &manifest_data, &manifest_size,
                         error, error_size) ||
        !manifest_parse(store, manifest_data, manifest_size,
                        error, error_size) ||
        !manifest_validate_layout(store, error, error_size)) {
        free(manifest_path);
        free(manifest_data);
        store_destroy(store);
        return NULL;
    }
    free(manifest_path);
    free(manifest_data);

    if (!store_open_files(store, artifact_root,
                          prefer_direct_io,
                          error, error_size)) {
        store_destroy(store);
        return NULL;
    }

    store->layout.cache_slots = (uint32_t)slots;
    store->layout.cache_bytes =
        slots * DS4_PLE_PAGE_BYTES;
    store->layout.worker_count = worker_count;
    store->set_count =
        (uint32_t)slots / DS4_PLE_CACHE_WAYS;
    if (posix_memalign(&store->cache_memory,
                       DS4_PLE_PAGE_BYTES,
                       store->layout.cache_bytes) != 0 ||
        !(store->slots =
              calloc(slots, sizeof(*store->slots))) ||
        !(store->queue =
              calloc(slots, sizeof(*store->queue))) ||
        !(store->workers =
              calloc(worker_count, sizeof(*store->workers)))) {
        ple_error(error, error_size,
                  "cannot allocate bounded PLE cache");
        store_destroy(store);
        return NULL;
    }
    for (uint32_t i = 0; i < (uint32_t)slots; i++)
        store->slots[i].data =
            (uint8_t *)store->cache_memory +
            (size_t)i * DS4_PLE_PAGE_BYTES;

    if (pthread_mutex_init(&store->mutex, NULL) != 0) {
        ple_error(error, error_size,
                  "cannot initialize PLE cache mutex");
        store_destroy(store);
        return NULL;
    }
    store->mutex_ready = true;
    if (pthread_cond_init(&store->work_cond, NULL) != 0) {
        ple_error(error, error_size,
                  "cannot initialize PLE work condition");
        store_destroy(store);
        return NULL;
    }
    store->work_cond_ready = true;
    if (pthread_cond_init(&store->state_cond, NULL) != 0) {
        ple_error(error, error_size,
                  "cannot initialize PLE state condition");
        store_destroy(store);
        return NULL;
    }
    store->state_cond_ready = true;
    for (uint32_t i = 0; i < worker_count; i++) {
        const int rc =
            pthread_create(&store->workers[i], NULL,
                           cache_worker, store);
        if (rc != 0) {
            ple_error(error, error_size,
                      "cannot create PLE I/O worker: %s",
                      strerror(rc));
            store_destroy(store);
            return NULL;
        }
        store->workers_started++;
    }
    return store;
}

void ds4_ple_store_close(ds4_ple_store *store) {
    store_destroy(store);
}

const ds4_ple_layout *ds4_ple_store_layout(
        const ds4_ple_store *store) {
    return store ? &store->layout : NULL;
}

const ds4_ple_hash_config *ds4_ple_store_hash_config(
        const ds4_ple_store *store) {
    return store ? &store->hash_config : NULL;
}

bool ds4_ple_store_prefetch_rows(
        ds4_ple_store *store,
        const uint64_t *row_ids,
        size_t row_count,
        char *error,
        size_t error_size) {
    if (!store || (row_count && !row_ids))
        return ple_error(error, error_size,
                         "PLE prefetch input is null");
    for (size_t i = 0; i < row_count; i++)
        if (!prefetch_one_row(store, row_ids[i],
                              error, error_size))
            return false;
    return true;
}

static bool store_acquire_row(
        ds4_ple_store *store,
        uint64_t global_row,
        ds4_ple_row_view *view,
        bool wait_for_slot,
        bool *acquired,
        char *error,
        size_t error_size) {
    if (!store || !view || !acquired)
        return ple_error(error, error_size,
                         "PLE row view is null");
    memset(view, 0, sizeof(*view));
    *acquired = false;
    uint32_t file_index = 0;
    uint64_t offset = 0;
    if (!resolve_row(store, global_row,
                     &file_index, &offset,
                     error, error_size))
        return false;

    const uint64_t started = ple_now_ns();
    const uint64_t page0 =
        offset & ~(uint64_t)(DS4_PLE_PAGE_BYTES - 1u);
    const uint32_t within = (uint32_t)(offset - page0);
    const uint32_t first_bytes =
        within + DS4_PLE_ROW_BYTES <= DS4_PLE_PAGE_BYTES
            ? DS4_PLE_ROW_BYTES
            : DS4_PLE_PAGE_BYTES - within;
    uint32_t slot0 = 0;
    const uint8_t *data0 = NULL;
    if (!cache_request_page(store, file_index, page0,
                            true, wait_for_slot, &slot0,
                            error, error_size))
        return false;
    if (slot0 == UINT32_MAX) return true;
    if (!cache_wait_ready(store, slot0, &data0,
                          error, error_size))
        return false;
    view->segments[0] = data0 + within;
    view->segment_bytes[0] = first_bytes;
    view->slots[0] = slot0;
    view->segment_count = 1;

    if (first_bytes < DS4_PLE_ROW_BYTES) {
        uint32_t slot1 = 0;
        const uint8_t *data1 = NULL;
        if (!cache_request_page(
                store, file_index,
                page0 + DS4_PLE_PAGE_BYTES,
                true, wait_for_slot, &slot1,
                error, error_size)) {
            cache_release(store, slot0);
            memset(view, 0, sizeof(*view));
            return false;
        }
        if (slot1 == UINT32_MAX) {
            cache_release(store, slot0);
            memset(view, 0, sizeof(*view));
            return true;
        }
        if (!cache_wait_ready(store, slot1, &data1,
                              error, error_size)) {
            cache_release(store, slot0);
            memset(view, 0, sizeof(*view));
            return false;
        }
        view->segments[1] = data1;
        view->segment_bytes[1] =
            DS4_PLE_ROW_BYTES - first_bytes;
        view->slots[1] = slot1;
        view->segment_count = 2;
    }

    const uint64_t finished = ple_now_ns();
    const uint64_t elapsed =
        finished >= started ? finished - started : 0;
    pthread_mutex_lock(&store->mutex);
    store->stats.row_lookups++;
    store->stats.logical_bytes += DS4_PLE_ROW_BYTES;
    store->stats.wait_samples++;
    store->stats.wait_nanoseconds_total += elapsed;
    if (elapsed > store->stats.wait_nanoseconds_max)
        store->stats.wait_nanoseconds_max = elapsed;
    pthread_mutex_unlock(&store->mutex);
    *acquired = true;
    return true;
}

bool ds4_ple_store_acquire_row(
        ds4_ple_store *store,
        uint64_t global_row,
        ds4_ple_row_view *view,
        char *error,
        size_t error_size) {
    bool acquired = false;
    if (!store_acquire_row(store, global_row, view, true,
                           &acquired, error, error_size))
        return false;
    if (!acquired)
        return ple_error(error, error_size,
                         "PLE blocking row acquisition made no progress");
    return true;
}

bool ds4_ple_store_try_acquire_row(
        ds4_ple_store *store,
        uint64_t global_row,
        ds4_ple_row_view *view,
        bool *acquired,
        char *error,
        size_t error_size) {
    return store_acquire_row(store, global_row, view, false,
                             acquired, error, error_size);
}

void ds4_ple_store_release_row(
        ds4_ple_store *store,
        ds4_ple_row_view *view) {
    if (!store || !view) return;
    for (uint32_t i = 0;
         i < view->segment_count && i < 2u;
         i++)
        cache_release(store, view->slots[i]);
    memset(view, 0, sizeof(*view));
}

bool ds4_ple_store_read_row(
        ds4_ple_store *store,
        uint64_t global_row,
        void *output,
        size_t output_size,
        char *error,
        size_t error_size) {
    if (!store || !output ||
        output_size < DS4_PLE_ROW_BYTES)
        return ple_error(error, error_size,
                         "PLE row output buffer is too small");
    ds4_ple_row_view view;
    if (!ds4_ple_store_acquire_row(
            store, global_row, &view,
            error, error_size))
        return false;
    memcpy(output, view.segments[0],
           view.segment_bytes[0]);
    if (view.segment_count == 2)
        memcpy((uint8_t *)output + view.segment_bytes[0],
               view.segments[1], view.segment_bytes[1]);
    ds4_ple_store_release_row(store, &view);
    return true;
}

bool ds4_ple_store_cache_span(
        ds4_ple_store *store,
        void **base,
        size_t *bytes) {
    if (!store || !base || !bytes) return false;
    *base = store->cache_memory;
    *bytes = store->layout.cache_bytes;
    return store->cache_memory != NULL &&
           store->layout.cache_bytes != 0;
}

void ds4_ple_store_get_stats(
        ds4_ple_store *store,
        ds4_ple_stats *stats) {
    if (!store || !stats) return;
    pthread_mutex_lock(&store->mutex);
    *stats = store->stats;
    pthread_mutex_unlock(&store->mutex);
}
