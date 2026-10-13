#ifndef DS4_HOST_LOAD_H
#define DS4_HOST_LOAD_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Host tensor directory consumed by native load.  Layout is the bridge ABI
 * for Model::open.  When installed, ds4.c skips parse_tensors and applies
 * this table (owned name copies).  The C ds4 / ds4-server path never
 * installs a directory, so the GGUF cursor walk stays the production
 * default. */

#define DS4_HOST_MAX_DIMS 8
/* Same bound as ds4.c's DS4_MAX_LAYER: the host only ever names a catalogued
 * shape, and Bonsai (qwen35) is 64 blocks wide.  Unused today; kept honest so
 * the next reader does not reach for the old DeepSeek bound. */
#define DS4_HOST_MAX_LAYER 64

static inline void ds4_host_set_err(char *err, size_t errlen, const char *msg)
{
    if (!err || errlen == 0) return;
    snprintf(err, errlen, "%s", msg ? msg : "unknown error");
}

/* Host-selected shape.  When installed, ds4_engine_open applies the pinned
 * C literal + compress table and skips config_validate_model.  The C
 * CLI/server leave this NULL.  variant matches ds4_variant / Rust Variant.
 * The v41_* block is the DeepSeek V4.1 wiring (variant 16): borrowed arrays,
 * n_layer long except v41_engram_layers (v41_n_engram entries), valid for
 * the open call.  The Rust host fills them from V41Wire::load, which mirrors
 * the engine's v41_load_metadata (core_validate_v41.c:46-154); the native
 * stores them in g_ds4_v41 and never re-reads the metadata on this path.
 * Other variants leave every v41_* field NULL/0. */
typedef struct {
    uint32_t variant;
    uint32_t n_compress;
    const uint32_t *compress; /* borrowed; DeepSeek only */
    /* V4.1 wiring */
    const uint8_t *v41_kv_source;            /* per layer: compresses, holds compressed KV */
    const uint8_t *v41_index_source;         /* per layer: runs the indexer */
    const int16_t *v41_kv_source_of;         /* per layer: nearest source, -1 = none */
    const int16_t *v41_index_source_of;
    const int16_t *v41_engram_index_of;      /* per layer: engram index, -1 = none */
    uint32_t v41_n_engram;
    const int32_t *v41_engram_layers;        /* metadata order */
    uint32_t v41_engram_max_ngram;
    uint32_t v41_engram_heads;
    uint32_t v41_engram_head_dim;
    uint32_t v41_engram_pad;
    int32_t v41_candidate_source_layer;
    int32_t v41_candidate_topk_blocks;
    int32_t v41_candidate_block_size;
    uint32_t v41_mtp_towers;
    uint32_t v41_mtp_experts;
    /* P4-3: the engram table on disk.  Per engram layer (v41_n_engram
     * entries): the row count, the two plane offsets inside that layer's
     * shard, and the shard path (NUL-terminated, shorter than the native
     * 1024-byte buffer).  Borrowed for the open call; all four NULL when
     * n_engram is 0, and the native then leaves its engram arrays zeroed.
     * The Rust host fills these from V41Wire; the native copies without
     * re-checking readability (the host owns the --engram-dir rewrite). */
    const uint64_t *v41_engram_rows;
    const uint64_t *v41_engram_weight_off;
    const uint64_t *v41_engram_scale_off;
    const char *const *v41_engram_table_path;
    /* P5: the draft parameters (core_validate_v41.c:91-110; the C GGUF path
     * parses the same five keys in v41_load_wiring).  They arm the drafter:
     * without them the towers bind but speculation stays off.  block == 0
     * means "not armed" (no towers, or the GGUF predates the parameters) and
     * every other draft field is then ignored; the native prints the engine's
     * warning when towers > 0.  When block != 0, n_mtp_target >= 1 and
     * mtp_target (borrowed, DS4_MTP_MAX_TOWERS*2 entries max) is required. */
    uint32_t v41_mtp_block;
    uint32_t v41_mtp_used;
    uint32_t v41_mtp_noise_id;
    uint32_t v41_mtp_markov_rank;
    uint32_t v41_n_mtp_target;
    const int16_t *v41_mtp_target;
    /* P5.4: the context length and the engram vocab pair.  The C GGUF path
     * reads all three as required keys (v41_load_wiring; core_validate_v41.c
     * :51-53, :152-153); without ctx the native generate entry and the state
     * alloc refuse ("no deepseek4.context_length"), so the Rust host must
     * carry it.  The vocab pair has no native reader yet but the C path
     * fills it; the host shape must not leave it zero. */
    uint32_t v41_ctx;
    uint32_t v41_engram_vocab;
    uint32_t v41_engram_cvocab;
} ds4_host_shape;

void ds4_host_shape_install(const ds4_host_shape *s);
void ds4_host_shape_clear(void);

/* Host tokenizer tables.  When installed, ds4_engine_open applies these
 * strings/specials and skips C vocab_load.  Pointers stay borrowed for
 * the engine lifetime (Rust Model owns Vocab).  The C CLI/server leave
 * this NULL. */
typedef struct {
    const char *ptr; /* borrowed; empty token still has a non-NULL ptr */
    uint64_t len;
} ds4_host_str;

typedef struct {
    uint32_t n_vocab;
    const ds4_host_str *tokens;
    uint32_t n_merges;
    const ds4_host_str *merges;
    uint32_t n_user_defined;
    const int32_t *user_defined; /* token ids; borrowed */
    uint32_t user_defined_max_len;
    uint8_t user_defined_first[256];
    uint8_t motif3_added_first[256];
    int32_t bos_id;
    int32_t eos_id;
    int32_t system_id;
    int32_t eot_id;
    int32_t im_start_id;
    int32_t im_content_id;
    int32_t im_end_id;
    int32_t user_id;
    int32_t assistant_id;
    int32_t start_of_turn_id;
    int32_t end_of_turn_id;
    int32_t tool_id;
    int32_t reference_id;
    int32_t plan_start_id;
    int32_t plan_end_id;
    int32_t observation_id;
    int32_t sop_id;
    int32_t think_start_id;
    int32_t think_end_id;
    int32_t tool_call_start_id;
    int32_t tool_call_end_id;
    int32_t tool_response_start_id;
    int32_t tool_response_end_id;
    int32_t arg_key_start_id;
    int32_t arg_key_end_id;
    int32_t arg_value_start_id;
    int32_t latent_start_id;
    int32_t latent_pad_id;
    int32_t latent_end_id;
    int32_t tool_schema_start_id;
    int32_t tool_schema_end_id;
    int32_t dsml_id;
    int32_t dots3_endofsystem_id;
    int32_t dots3_endofuser_id;
    int32_t dots3_endoftext_id;
} ds4_host_vocab;

void ds4_host_vocab_install(const ds4_host_vocab *v);
void ds4_host_vocab_clear(void);

/* Identity check before native copies pointers into ds4_vocab. */
static inline int ds4_host_vocab_apply(const ds4_host_vocab *h,
                                       char *err, size_t errlen)
{
    uint32_t i;

    if (!h) {
        ds4_host_set_err(err, errlen, "vocab-null");
        return 1;
    }
    if (h->n_vocab > 0 && !h->tokens) {
        ds4_host_set_err(err, errlen, "tokens-null");
        return 1;
    }
    if (h->n_merges > 0 && !h->merges) {
        ds4_host_set_err(err, errlen, "merges-null");
        return 1;
    }
    if (h->n_user_defined > 0 && !h->user_defined) {
        ds4_host_set_err(err, errlen, "ud-null");
        return 1;
    }
    for (i = 0; i < h->n_vocab; i++) {
        if (!h->tokens[i].ptr) {
            ds4_host_set_err(err, errlen, "token-empty");
            return 1;
        }
    }
    for (i = 0; i < h->n_merges; i++) {
        if (!h->merges[i].ptr) {
            ds4_host_set_err(err, errlen, "merge-empty");
            return 1;
        }
    }
    for (i = 0; i < h->n_user_defined; i++) {
        int32_t id = h->user_defined[i];
        if (id < 0 || (uint32_t)id >= h->n_vocab) {
            ds4_host_set_err(err, errlen, "ud-range");
            return 1;
        }
        if (h->tokens[id].len == 0) {
            ds4_host_set_err(err, errlen, "ud-empty");
            return 1;
        }
    }
    return 0;
}

typedef struct {
    const char *name; /* borrowed for the open call */
    uint32_t ndim;
    uint64_t dim[DS4_HOST_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t bytes;
} ds4_host_tensor;

typedef struct {
    uint32_t n;
    const ds4_host_tensor *v;
    uint64_t data_pos;
    uint64_t alignment;
} ds4_host_tensor_dir;

/* Native parsed row (name points into the GGUF map).  Consume writes host
 * offsets/bytes back into these fields when identity matches. */
typedef struct {
    const char *name;
    uint32_t name_len;
    uint32_t ndim;
    uint64_t dim[DS4_HOST_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t bytes;
} ds4_host_native_tensor;

void ds4_host_tensor_dir_install(const ds4_host_tensor_dir *d);
void ds4_host_tensor_dir_clear(void);

/* Index identity + host-authoritative offsets.  Absence of a directory is
 * the C default path (return 0).  A present directory must cover every
 * native tensor in file order. */
static inline int ds4_host_tensor_dir_consume(
        ds4_host_native_tensor *native, uint64_t n_native,
        uint64_t native_data_pos, uint64_t native_alignment,
        const ds4_host_tensor_dir *dir,
        char *err, size_t errlen)
{
    uint64_t i, d;

    if (!dir) return 0; /* C default: no host directory, keep parsed table */
    if (dir->n > 0 && !dir->v) {
        ds4_host_set_err(err, errlen, "tensors-null");
        return 1;
    }
    if ((uint64_t)dir->n != n_native) {
        ds4_host_set_err(err, errlen, "count-mismatch");
        return 1;
    }
    for (i = 0; i < n_native; i++) {
        const ds4_host_tensor *h = &dir->v[i];
        ds4_host_native_tensor *n = &native[i];
        size_t zlen;

        if (!h->name || !h->name[0]) {
            ds4_host_set_err(err, errlen, "name-empty");
            return 1;
        }
        zlen = strlen(h->name);
        if (!n->name || n->name_len != (uint32_t)zlen ||
            memcmp(n->name, h->name, zlen) != 0) {
            ds4_host_set_err(err, errlen, "name-mismatch");
            return 1;
        }
        if (n->type != h->type) {
            ds4_host_set_err(err, errlen, "type-mismatch");
            return 1;
        }
        if (n->ndim != h->ndim || h->ndim == 0 ||
            h->ndim > DS4_HOST_MAX_DIMS) {
            ds4_host_set_err(err, errlen, "dim-mismatch");
            return 1;
        }
        for (d = 0; d < h->ndim; d++) {
            if (n->dim[d] != h->dim[d]) {
                ds4_host_set_err(err, errlen, "dim-mismatch");
                return 1;
            }
        }
        if (n->rel_offset != h->rel_offset || n->abs_offset != h->abs_offset) {
            ds4_host_set_err(err, errlen, "offset-mismatch");
            return 1;
        }
        if (n->bytes != h->bytes) {
            ds4_host_set_err(err, errlen, "bytes-mismatch");
            return 1;
        }
        n->rel_offset = h->rel_offset;
        n->abs_offset = h->abs_offset;
        n->bytes = h->bytes;
    }
    if (dir->data_pos != native_data_pos || dir->alignment != native_alignment) {
        ds4_host_set_err(err, errlen, "data-mismatch");
        return 1;
    }
    return 0;
}

/* Build a native view from the host directory (no prior parse).  Name
 * pointers stay borrowed; the engine must copy them before clear. */
static inline int ds4_host_tensor_dir_apply(
        ds4_host_native_tensor *out, uint32_t cap,
        const ds4_host_tensor_dir *dir,
        char *err, size_t errlen)
{
    uint32_t i, d;

    if (!dir) {
        ds4_host_set_err(err, errlen, "dir-null");
        return 1;
    }
    if (dir->n > 0 && !dir->v) {
        ds4_host_set_err(err, errlen, "tensors-null");
        return 1;
    }
    if (dir->n != cap) {
        ds4_host_set_err(err, errlen, "count-mismatch");
        return 1;
    }
    if (dir->n > 0 && !out) {
        ds4_host_set_err(err, errlen, "out-null");
        return 1;
    }
    for (i = 0; i < dir->n; i++) {
        const ds4_host_tensor *h = &dir->v[i];
        ds4_host_native_tensor *n = &out[i];
        size_t zlen;

        if (!h->name || !h->name[0]) {
            ds4_host_set_err(err, errlen, "name-empty");
            return 1;
        }
        if (h->ndim == 0 || h->ndim > DS4_HOST_MAX_DIMS) {
            ds4_host_set_err(err, errlen, "dim-mismatch");
            return 1;
        }
        zlen = strlen(h->name);
        memset(n, 0, sizeof(*n));
        n->name = h->name;
        n->name_len = (uint32_t)zlen;
        n->ndim = h->ndim;
        n->type = h->type;
        n->rel_offset = h->rel_offset;
        n->abs_offset = h->abs_offset;
        n->bytes = h->bytes;
        for (d = 0; d < DS4_HOST_MAX_DIMS; d++) n->dim[d] = h->dim[d];
    }
    return 0;
}

/* Host bind name → tensor-dir index.  When installed, model_find_tensor
 * resolves through this table instead of scanning m->tensors.  Names
 * not in the map (return 2 / "unknown") fall back to the C linear
 * scan so MTP/DSpark sibling binds stay valid.  The C CLI/server
 * leave this NULL. */

#define DS4_HOST_BIND_MISS UINT32_MAX

typedef struct {
    const char *name; /* borrowed; empty name is invalid */
    uint32_t required;
    uint32_t found;
    uint32_t index; /* tensor-dir index when found; else DS4_HOST_BIND_MISS */
} ds4_host_bind_look;

typedef struct {
    uint32_t n;
    const ds4_host_bind_look *v;
} ds4_host_bind_map;

void ds4_host_bind_map_install(const ds4_host_bind_map *m);
void ds4_host_bind_map_clear(void);

/* MTP/DSpark sibling bind maps.  Separate install slots (the base map is
 * cleared after base weights_bind); ds4_engine_open swaps each into the
 * active map only around its own sibling open+bind window and skips that
 * sibling's C layout check, mirroring the base skip.  The C CLI/server
 * leave these NULL. */
void ds4_host_mtp_bind_map_install(const ds4_host_bind_map *m);
void ds4_host_mtp_bind_map_clear(void);
void ds4_host_dspark_bind_map_install(const ds4_host_bind_map *m);
void ds4_host_dspark_bind_map_clear(void);

/* 0 = resolved (index_out is a valid index, or DS4_HOST_BIND_MISS for
 * an optional miss).  1 = hard error (token in err).  2 = name not in
 * the map (caller falls back to the C walk). */
static inline int ds4_host_bind_lookup(
        const ds4_host_bind_map *map,
        const char *name,
        uint32_t n_tensors,
        uint32_t *index_out,
        char *err, size_t errlen)
{
    uint32_t i;

    if (index_out) *index_out = DS4_HOST_BIND_MISS;
    if (!map) {
        ds4_host_set_err(err, errlen, "map-null");
        return 1;
    }
    if (!name || !name[0]) {
        ds4_host_set_err(err, errlen, "name-empty");
        return 1;
    }
    if (map->n > 0 && !map->v) {
        ds4_host_set_err(err, errlen, "looks-null");
        return 1;
    }
    for (i = 0; i < map->n; i++) {
        const ds4_host_bind_look *e = &map->v[i];

        if (!e->name || !e->name[0]) {
            ds4_host_set_err(err, errlen, "name-empty");
            return 1;
        }
        if (strcmp(e->name, name) != 0) continue;
        if (!e->found) {
            if (e->required) {
                if (err && errlen) snprintf(err, errlen, "missing %s", name);
                return 1;
            }
            return 0;
        }
        if (e->index == DS4_HOST_BIND_MISS || e->index >= n_tensors) {
            ds4_host_set_err(err, errlen, "index-range");
            return 1;
        }
        if (index_out) *index_out = e->index;
        return 0;
    }
    ds4_host_set_err(err, errlen, "unknown");
    return 2;
}

#endif
