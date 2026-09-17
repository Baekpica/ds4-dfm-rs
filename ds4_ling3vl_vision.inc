/* Ling-3.0-flash-VL image spans: M-RoPE coordinates and projected rows.
 *
 * Rust owns decoding, bounds and placeholder markup; this boundary
 * independently re-checks the span geometry before any device write, because
 * a wrong span silently corrupts both the embedding rows and every later
 * position. */

#include "ds4_ling3vl_rope.h"

enum { L3V_HIDDEN = 2560, L3V_HEADS = 32, L3V_KDA_HEAD = 128,
       L3V_KDA_DIM = L3V_HEADS * L3V_KDA_HEAD,
       L3V_CONV = 4, L3V_KV_LORA = 512,
       L3V_KEY_MLA = 192, L3V_NOPE = L3V_KEY_MLA - L3V_ROPE,
       L3V_VALUE_MLA = 128, L3V_KV_ROW = L3V_KV_LORA + L3V_ROPE,
       L3V_QDIM = L3V_HEADS * L3V_KEY_MLA,
       L3V_LATENT = L3V_HEADS * L3V_KV_LORA,
       L3V_VDIM = L3V_HEADS * L3V_VALUE_MLA,
       L3V_DENSE = 6144, L3V_EXPERT_FF = 768, L3V_EXPERTS = 512,
       L3V_USED = 8,
       /* BF16 K/V segments: 3 x cap x (6144 + 4096) x 2 B fits the
        * cap x 32 x 512 x 4 B q_absorbed buffer they alias (94%). */
       L3V_MLA_SEG_CHUNKS = 3,
       L3V_SECTION_T = 8, L3V_SECTION_H = 12 };

/* Contiguous [T, H, W] split of the 32 rotary pairs, adjacent-pair rotation.
 * Text rows set all three axes to the same index, so text is 1-D RoPE. */
enum { L3V_MROPE_PAIRS = L3V_ROPE / 2 };

/* The span the vision bridge fills is a fixed id triple in this vocabulary:
 * <|vision_start|> <|image_pad|> ... <|vision_end|>. */
enum { LING3VL_IMAGE_TOKEN = 157157u, LING3VL_VISION_START_TOKEN = 157158u,
       LING3VL_VISION_END_TOKEN = 157159u };
enum { LING3VL_MEDIA_INPUTS = 4u, LING3VL_MEDIA_ROWS = 16384u };

typedef struct {
    unsigned position;   /* first prompt row of the span */
    unsigned rows;       /* merged vision tokens in the span */
    unsigned feature;    /* first row inside the projected plane */
    unsigned grid_h;     /* pre-merge patch grid, always even */
    unsigned grid_w;
} ds4_ling3vl_image_span;

typedef struct {
    ds4_gpu_tensor *features;
    ds4_ling3vl_image_span spans[LING3VL_MEDIA_INPUTS];
    /* `capacity` is the session's feature plane in rows: a prompt whose image
     * spans exceed it has nowhere to put the projected rows. */
    unsigned rows, count, capacity;
} ds4_ling3vl_media;

/* Interleaved M-RoPE coordinates.
 *
 *   text          t = h = w = cursor, then cursor += 1
 *   image span    t = base, h = base + y, w = base + x over the merged
 *                 (grid_h/2) x (grid_w/2) grid, then cursor = base + max(H, W)
 *
 * Text therefore reduces to ordinary 1-D RoPE, and a span costs the larger of
 * its two axes rather than its token count. */
static bool ling3vl_media_positions(const ds4_ling3vl_media *media,
                                    const int *tokens, unsigned n,
                                    unsigned pos, int32_t *out) {
    if (!tokens || !n || !out) { return false; }
    if (!media || !media->count) {
        for (unsigned i = 0; i < n; i++) {
            const int32_t p = (int32_t)(pos + i);
            out[3u * i] = p;
            out[3u * i + 1u] = p;
            out[3u * i + 2u] = p;
        }
        return true;
    }
    /* Replay the plan from row zero so a chunked prefill keeps one cursor. */
    const unsigned end = pos + n;
    unsigned cursor = 0u, row = 0u, image = 0u;
    while (row < end) {
        if (image < media->count && row == media->spans[image].position) {
            const ds4_ling3vl_image_span *s = &media->spans[image];
            const unsigned gh = s->grid_h / 2u, gw = s->grid_w / 2u;
            if (!gh || !gw || gh * gw != s->rows) { return false; }
            for (unsigned y = 0; y < gh; y++) {
                for (unsigned x = 0; x < gw; x++) {
                    const unsigned r = row + y * gw + x;
                    if (r < pos || r >= end) { continue; }
                    int32_t *p = out + 3u * (r - pos);
                    p[0] = (int32_t)cursor;
                    p[1] = (int32_t)(cursor + y);
                    p[2] = (int32_t)(cursor + x);
                }
            }
            row += s->rows;
            cursor += gh > gw ? gh : gw;
            image++;
            continue;
        }
        if (row >= pos) {
            int32_t *p = out + 3u * (row - pos);
            p[0] = p[1] = p[2] = (int32_t)cursor;
        }
        row++;
        cursor++;
    }
    return cursor <= INT32_MAX;
}

/* Copy the projected image rows over the placeholder embeddings that the
 * token path just wrote, for the slice of the span this chunk covers. */
static bool ling3vl_media_apply(const ds4_ling3vl_media *media,
                                ds4_gpu_tensor *out, unsigned position,
                                unsigned rows) {
    if (!media || !media->count) { return true; }
    if (!out || !media->features || !rows || media->count > LING3VL_MEDIA_INPUTS ||
        !media->rows || media->rows > media->capacity ||
        media->capacity > LING3VL_MEDIA_ROWS ||
        position > L3V_CONTEXT || rows > L3V_CONTEXT - position) { return false; }
    unsigned previous_end = 0u;
    for (unsigned i = 0; i < media->count; i++) {
        const ds4_ling3vl_image_span *s = &media->spans[i];
        if (!s->rows || s->position < previous_end || s->position > L3V_CONTEXT ||
            s->rows > L3V_CONTEXT - s->position || s->feature > media->rows ||
            s->rows > media->rows - s->feature) { return false; }
        previous_end = s->position + s->rows;
    }
    const uint64_t row_bytes = (uint64_t)L3V_HIDDEN * sizeof(float);
    for (unsigned i = 0; i < media->count; i++) {
        const ds4_ling3vl_image_span *s = &media->spans[i];
        const unsigned start = position > s->position ? position : s->position;
        const unsigned stop = position + rows < s->position + s->rows
            ? position + rows : s->position + s->rows;
        if (start >= stop) { continue; }
        if (!ds4_gpu_tensor_copy(out, (uint64_t)(start - position) * row_bytes,
                                 media->features,
                                 (uint64_t)(s->feature + start - s->position) * row_bytes,
                                 (uint64_t)(stop - start) * row_bytes)) {
            return false;
        }
    }
    return true;
}

/* Qwen3-VL ViT with this checkpoint's own precision and projector.
 *
 * The encoder is structurally the Qwen4Exp tower already in this engine --
 * 27 blocks of 1152 with fused QKV, LayerNorm and a 2-D rotary over the patch
 * grid -- so it reuses those operators directly. Three things differ and are
 * why this has its own binder and forward:
 *
 *   - the weights are BF16/F32 here, not Q8_0;
 *   - the Conv3D patch embedding is split along the temporal axis into two
 *     tensors, and a still image feeds both halves the same pixels;
 *   - `disable_merger_proj` moves the projection out of the merger, so the
 *     merger is a norm plus the top-level linear_proj pair (mm.0, mm.2).
 */
enum { L3V_VIT_LAYERS = 27, L3V_VIT_HIDDEN = 1152, L3V_VIT_FF = 4304,
       L3V_VIT_HEADS = 16, L3V_VIT_HEAD = L3V_VIT_HIDDEN / L3V_VIT_HEADS,
       L3V_VIT_PATCH = 16, L3V_VIT_PATCH_DIM = 3 * L3V_VIT_PATCH * L3V_VIT_PATCH,
       L3V_VIT_MERGE = 2, L3V_VIT_MERGED = 4 * L3V_VIT_HIDDEN,
       L3V_VIT_POSITIONS = 2304, L3V_VIT_POS_SIDE = 48,
       L3V_VIT_MAX_PATCHES = 4 * LING3VL_MEDIA_ROWS / LING3VL_MEDIA_INPUTS };

#define L3V_VIT_LN_EPS 1e-6f
#define L3V_VIT_ROPE_BASE 10000.0f
/* OpenAI CLIP normalization, not the [-1, 1] scaling Qwen's processor uses. */
static const float L3V_IMAGE_MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
static const float L3V_IMAGE_STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};

typedef struct {
    uint64_t ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b;
    uint64_t ln2_w, ln2_b, up_w, up_b, down_w, down_b;
} ds4_ling3vl_vision_layer;

typedef struct {
    const ds4_tensor *patch0, *patch1;
    uint64_t patch_bias, position, post_ln_w, post_ln_b;
    const ds4_tensor *mm0, *mm2;
    uint64_t mm0_bias, mm2_bias;
    ds4_ling3vl_vision_layer layer[L3V_VIT_LAYERS];
    bool bound;
} ds4_ling3vl_vision_weights;

typedef struct {
    ds4_gpu_tensor *patches, *hidden, *norm, *qkv, *ff, *layer_out, *projected;
    ds4_gpu_tensor *pos_indices, *pos_weights, *height, *width;
    ds4_gpu_tensor *seg_start, *seg_end;
    uint32_t capacity;
    uint64_t bytes;
} ds4_ling3vl_vision;

static const ds4_tensor *ling3vl_vision_tensor(const ds4_model *m,
                                               const char *name, uint32_t type,
                                               uint32_t ndim,
                                               const uint64_t *dims) {
    const ds4_tensor *t = required_tensor(m, name);
    if (t->type != type || t->ndim != ndim) {
        fprintf(stderr, "ds4: vision tensor %s has type %s/rank %u\n", name,
                tensor_type_name(t->type), t->ndim);
        exit(1);
    }
    for (uint32_t d = 0; d < ndim; d++) {
        if (t->dim[d] == dims[d]) { continue; }
        fprintf(stderr, "ds4: vision tensor %s dim[%u]=%" PRIu64 ", expected %"
                PRIu64 "\n", name, d, t->dim[d], dims[d]);
        exit(1);
    }
    return t;
}

static uint64_t ling3vl_vision_offset(const ds4_model *m, const char *name,
                                      uint32_t type, uint32_t ndim,
                                      const uint64_t *dims) {
    return ling3vl_vision_tensor(m, name, type, ndim, dims)->abs_offset;
}

static void ling3vl_vision_bind(ds4_ling3vl_vision_weights *w,
                                const ds4_model *m) {
    ds4_str arch = {0};
    memset(w, 0, sizeof(*w));
    if (!model_get_string(m, "general.architecture", &arch) ||
        !ds4_streq(arch, "clip")) {
        ds4_die("--vision file is not a Ling-3.0-flash-VL mmproj GGUF");
    }
    if (m->n_tensors != 334u) {
        fprintf(stderr, "ds4: mmproj has %" PRIu64 " tensors, expected 334\n",
                m->n_tensors);
        exit(1);
    }
    const uint64_t patch[4] = {L3V_VIT_PATCH, L3V_VIT_PATCH, 3, L3V_VIT_HIDDEN};
    const uint64_t vec[1] = {L3V_VIT_HIDDEN};
    const uint64_t embd[1] = {L3V_HIDDEN};
    const uint64_t pos[2] = {L3V_VIT_HIDDEN, L3V_VIT_POSITIONS};
    const uint64_t mm0[2] = {L3V_VIT_MERGED, L3V_HIDDEN};
    const uint64_t mm2[2] = {L3V_HIDDEN, L3V_HIDDEN};
    w->patch0 = ling3vl_vision_tensor(m, "v.patch_embd.weight", DS4_TENSOR_F32, 4, patch);
    w->patch1 = ling3vl_vision_tensor(m, "v.patch_embd.weight.1", DS4_TENSOR_F32, 4, patch);
    w->patch_bias = ling3vl_vision_offset(m, "v.patch_embd.bias", DS4_TENSOR_F32, 1, vec);
    w->position = ling3vl_vision_offset(m, "v.position_embd.weight", DS4_TENSOR_F32, 2, pos);
    w->post_ln_w = ling3vl_vision_offset(m, "v.post_ln.weight", DS4_TENSOR_F32, 1, vec);
    w->post_ln_b = ling3vl_vision_offset(m, "v.post_ln.bias", DS4_TENSOR_F32, 1, vec);
    w->mm0 = ling3vl_vision_tensor(m, "mm.0.weight", DS4_TENSOR_BF16, 2, mm0);
    w->mm2 = ling3vl_vision_tensor(m, "mm.2.weight", DS4_TENSOR_BF16, 2, mm2);
    w->mm0_bias = ling3vl_vision_offset(m, "mm.0.bias", DS4_TENSOR_F32, 1, embd);
    w->mm2_bias = ling3vl_vision_offset(m, "mm.2.bias", DS4_TENSOR_F32, 1, embd);
    for (uint32_t il = 0; il < L3V_VIT_LAYERS; il++) {
        char name[64];
        ds4_ling3vl_vision_layer *l = &w->layer[il];
        const uint64_t qkv_w[2] = {L3V_VIT_HIDDEN, 3 * L3V_VIT_HIDDEN};
        const uint64_t qkv_b[1] = {3 * L3V_VIT_HIDDEN};
        const uint64_t square[2] = {L3V_VIT_HIDDEN, L3V_VIT_HIDDEN};
        const uint64_t up_w[2] = {L3V_VIT_HIDDEN, L3V_VIT_FF};
        const uint64_t up_b[1] = {L3V_VIT_FF};
        const uint64_t down_w[2] = {L3V_VIT_FF, L3V_VIT_HIDDEN};
#define L3V_VIT_BIND(field, suffix, type, ndim, dims)                          \
        do {                                                                   \
            snprintf(name, sizeof(name), "v.blk.%u." suffix, il);              \
            l->field = ling3vl_vision_offset(m, name, type, ndim, dims);       \
        } while (0)
        L3V_VIT_BIND(ln1_w, "ln1.weight", DS4_TENSOR_F32, 1, vec);
        L3V_VIT_BIND(ln1_b, "ln1.bias", DS4_TENSOR_F32, 1, vec);
        L3V_VIT_BIND(qkv_w, "attn_qkv.weight", DS4_TENSOR_BF16, 2, qkv_w);
        L3V_VIT_BIND(qkv_b, "attn_qkv.bias", DS4_TENSOR_F32, 1, qkv_b);
        L3V_VIT_BIND(out_w, "attn_out.weight", DS4_TENSOR_BF16, 2, square);
        L3V_VIT_BIND(out_b, "attn_out.bias", DS4_TENSOR_F32, 1, vec);
        L3V_VIT_BIND(ln2_w, "ln2.weight", DS4_TENSOR_F32, 1, vec);
        L3V_VIT_BIND(ln2_b, "ln2.bias", DS4_TENSOR_F32, 1, vec);
        L3V_VIT_BIND(up_w, "ffn_up.weight", DS4_TENSOR_BF16, 2, up_w);
        L3V_VIT_BIND(up_b, "ffn_up.bias", DS4_TENSOR_F32, 1, up_b);
        L3V_VIT_BIND(down_w, "ffn_down.weight", DS4_TENSOR_BF16, 2, down_w);
        L3V_VIT_BIND(down_b, "ffn_down.bias", DS4_TENSOR_F32, 1, vec);
#undef L3V_VIT_BIND
    }
    w->bound = true;
}

static void ling3vl_vision_free(ds4_ling3vl_vision *v) {
    if (!v) { return; }
    ds4_gpu_tensor *all[] = {v->patches, v->hidden, v->norm, v->qkv, v->ff,
                             v->layer_out, v->projected, v->pos_indices,
                             v->pos_weights, v->height, v->width,
                             v->seg_start, v->seg_end};
    for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        ds4_gpu_tensor_free(all[i]);
    }
    memset(v, 0, sizeof(*v));
}

static bool ling3vl_vision_alloc(ds4_ling3vl_vision *v, uint32_t patches) {
    memset(v, 0, sizeof(*v));
    if (!patches || patches % 4u || patches > L3V_VIT_MAX_PATCHES) { return false; }
    v->capacity = patches;
    const uint64_t p = patches;
#define L3V_VIT_ALLOC(field, elems) do {                                       \
    const uint64_t nbytes = (uint64_t)(elems) * sizeof(float);                 \
    v->field = ds4_gpu_tensor_alloc(nbytes);                                   \
    if (!v->field) { ling3vl_vision_free(v); return false; }                   \
    v->bytes += nbytes;                                                        \
} while (0)
    L3V_VIT_ALLOC(patches, p * L3V_VIT_PATCH_DIM);
    L3V_VIT_ALLOC(hidden, p * L3V_VIT_HIDDEN);
    L3V_VIT_ALLOC(norm, p * L3V_VIT_HIDDEN);
    L3V_VIT_ALLOC(qkv, p * 3ull * L3V_VIT_HIDDEN);
    L3V_VIT_ALLOC(ff, p * L3V_VIT_FF);
    L3V_VIT_ALLOC(layer_out, p * L3V_VIT_HIDDEN);
    L3V_VIT_ALLOC(projected, p / 4ull * L3V_HIDDEN);
    L3V_VIT_ALLOC(pos_indices, p * 4ull);
    L3V_VIT_ALLOC(pos_weights, p * 4ull);
    L3V_VIT_ALLOC(height, p);
    L3V_VIT_ALLOC(width, p);
    /* One image per encode, so every patch shares one attention segment. */
    L3V_VIT_ALLOC(seg_start, p);
    L3V_VIT_ALLOC(seg_end, p);
#undef L3V_VIT_ALLOC
    return true;
}

/* Pack one resized image into 2x2 block-major patches, in the (c, py, px)
 * order this checkpoint's patch tensors expect, and precompute the bilinear
 * taps into the 48x48 learned position grid. */
static bool ling3vl_vision_pack(const uint8_t *rgb, uint32_t width,
                                uint32_t grid_h, uint32_t grid_w, float *patches,
                                int32_t *indices, float *weights,
                                int32_t *height, int32_t *pos_width) {
    if (!rgb || !patches || !indices || !weights || !height || !pos_width ||
        !grid_h || !grid_w || (grid_h & 1u) || (grid_w & 1u)) { return false; }
    uint32_t row = 0u;
    for (uint32_t bh = 0; bh < grid_h / 2u; bh++) {
        for (uint32_t bw = 0; bw < grid_w / 2u; bw++) {
            for (uint32_t ih = 0; ih < 2u; ih++) {
                for (uint32_t iw = 0; iw < 2u; iw++) {
                    const uint32_t ph = 2u * bh + ih, pw = 2u * bw + iw;
                    float *dst = patches + (size_t)row * L3V_VIT_PATCH_DIM;
                    for (uint32_t c = 0; c < 3u; c++) {
                        for (uint32_t py = 0; py < L3V_VIT_PATCH; py++) {
                            const uint32_t y = ph * L3V_VIT_PATCH + py;
                            for (uint32_t px = 0; px < L3V_VIT_PATCH; px++) {
                                const uint32_t x = pw * L3V_VIT_PATCH + px;
                                const uint8_t v =
                                    rgb[((size_t)y * width + x) * 3u + c];
                                *dst++ = ((float)v / 255.0f - L3V_IMAGE_MEAN[c]) /
                                         L3V_IMAGE_STD[c];
                            }
                        }
                    }
                    const double span = L3V_VIT_POS_SIDE - 1;
                    const double sy = grid_h > 1u ? (double)ph * span / (grid_h - 1u) : 0.0;
                    const double sx = grid_w > 1u ? (double)pw * span / (grid_w - 1u) : 0.0;
                    const int y0 = (int)floor(sy), x0 = (int)floor(sx);
                    const int y1 = y0 < (int)span ? y0 + 1 : (int)span;
                    const int x1 = x0 < (int)span ? x0 + 1 : (int)span;
                    const float fy = (float)(sy - y0), fx = (float)(sx - x0);
                    indices[4u * row + 0u] = L3V_VIT_POS_SIDE * y0 + x0;
                    indices[4u * row + 1u] = L3V_VIT_POS_SIDE * y0 + x1;
                    indices[4u * row + 2u] = L3V_VIT_POS_SIDE * y1 + x0;
                    indices[4u * row + 3u] = L3V_VIT_POS_SIDE * y1 + x1;
                    weights[4u * row + 0u] = (1.0f - fy) * (1.0f - fx);
                    weights[4u * row + 1u] = (1.0f - fy) * fx;
                    weights[4u * row + 2u] = fy * (1.0f - fx);
                    weights[4u * row + 3u] = fy * fx;
                    height[row] = (int32_t)ph;
                    pos_width[row] = (int32_t)pw;
                    row++;
                }
            }
        }
    }
    return row == grid_h * grid_w;
}

static bool ling3vl_vision_forward(ds4_ling3vl_vision *v, const ds4_model *m,
                                   const ds4_ling3vl_vision_weights *w,
                                   uint32_t patches) {
    if (!v || !m || !w || !w->bound || !patches || patches > v->capacity ||
        patches % 4u) { return false; }
    const uint32_t p = patches, merged = patches / 4u;
    if (!plain_graph_matmul_tensor(v->hidden, m, w->patch0, L3V_VIT_PATCH_DIM,
                                   L3V_VIT_HIDDEN, v->patches, p) ||
        !plain_graph_matmul_tensor(v->norm, m, w->patch1, L3V_VIT_PATCH_DIM,
                                   L3V_VIT_HIDDEN, v->patches, p) ||
        /* A still image duplicates the frame across the temporal axis, so the
         * two Conv3D halves see the same pixels and their outputs add. */
        !ds4_gpu_exaone_add_tensor(v->hidden, v->norm,
                                   (uint64_t)p * L3V_VIT_HIDDEN) ||
        !ds4_gpu_ling3vl_vision_patch_position(
            v->hidden, m->map, m->size, w->patch_bias, w->position,
            v->pos_indices, v->pos_weights, p, L3V_VIT_HIDDEN,
            L3V_VIT_POSITIONS)) {
        return false;
    }
    for (uint32_t il = 0; il < L3V_VIT_LAYERS; il++) {
        const ds4_ling3vl_vision_layer *l = &w->layer[il];
        const bool ok =
            ds4_gpu_qwen4exp_vision_layernorm_tensor(
                v->norm, v->hidden, m->map, m->size, l->ln1_w, l->ln1_b, p,
                L3V_VIT_HIDDEN, L3V_VIT_LN_EPS) &&
            ds4_gpu_matmul_bf16_tensor(v->qkv, m->map, m->size, l->qkv_w,
                                       L3V_VIT_HIDDEN, 3u * L3V_VIT_HIDDEN,
                                       v->norm, p) &&
            ds4_gpu_qwen4exp_vision_qkv_rope_tensor(
                v->qkv, m->map, m->size, l->qkv_b, v->height, v->width, p,
                L3V_VIT_HEADS, L3V_VIT_HEAD, L3V_VIT_ROPE_BASE) &&
            ds4_gpu_qwen4exp_vision_attention_tensor(
                v->norm, v->qkv, v->seg_start, v->seg_end, p, L3V_VIT_HEADS,
                L3V_VIT_HEAD) &&
            ds4_gpu_matmul_bf16_tensor(v->layer_out, m->map, m->size, l->out_w,
                                       L3V_VIT_HIDDEN, L3V_VIT_HIDDEN, v->norm, p) &&
            ds4_gpu_qwen4exp_vision_bias_residual_tensor(
                v->hidden, v->layer_out, m->map, m->size, l->out_b, p,
                L3V_VIT_HIDDEN) &&
            ds4_gpu_qwen4exp_vision_layernorm_tensor(
                v->norm, v->hidden, m->map, m->size, l->ln2_w, l->ln2_b, p,
                L3V_VIT_HIDDEN, L3V_VIT_LN_EPS) &&
            ds4_gpu_matmul_bf16_tensor(v->ff, m->map, m->size, l->up_w,
                                       L3V_VIT_HIDDEN, L3V_VIT_FF, v->norm, p) &&
            ds4_gpu_qwen4exp_vision_bias_gelu_tensor(
                v->ff, m->map, m->size, l->up_b, p, L3V_VIT_FF, 1) &&
            ds4_gpu_matmul_bf16_tensor(v->layer_out, m->map, m->size, l->down_w,
                                       L3V_VIT_FF, L3V_VIT_HIDDEN, v->ff, p) &&
            ds4_gpu_qwen4exp_vision_bias_residual_tensor(
                v->hidden, v->layer_out, m->map, m->size, l->down_b, p,
                L3V_VIT_HIDDEN);
        if (!ok) {
            fprintf(stderr, "ds4: Ling vision failed at block %u\n", il);
            return false;
        }
    }
    /* The merger norm is per patch; the projection then reads each 2x2 block
     * as one 4608-wide row.  mm.0 already lands in the language width, so the
     * second projection is square. */
    return ds4_gpu_qwen4exp_vision_layernorm_tensor(
               v->norm, v->hidden, m->map, m->size, w->post_ln_w, w->post_ln_b,
               p, L3V_VIT_HIDDEN, L3V_VIT_LN_EPS) &&
        ds4_gpu_matmul_bf16_tensor(v->hidden, m->map, m->size,
                                   w->mm0->abs_offset, L3V_VIT_MERGED,
                                   L3V_HIDDEN, v->norm, merged) &&
        ds4_gpu_qwen4exp_vision_bias_gelu_tensor(
            v->hidden, m->map, m->size, w->mm0_bias, merged, L3V_HIDDEN, 0) &&
        ds4_gpu_matmul_bf16_tensor(v->projected, m->map, m->size,
                                   w->mm2->abs_offset, L3V_HIDDEN, L3V_HIDDEN,
                                   v->hidden, merged) &&
        ds4_gpu_qwen4exp_vision_bias_tensor(
            v->projected, m->map, m->size, w->mm2_bias, merged, L3V_HIDDEN);
}

/* Re-check the placeholder coverage the host claims, then plan the spans.
 * The prompt must hold exactly token_count image ids at each offset and no
 * image id anywhere else, or the projected rows would land on text. */
static bool ling3vl_media_check(const ds4_tokens *prompt,
                                const ds4_ling3vl_image_input *images,
                                uint32_t count,
                                const ds4_ling3vl_image_info *info,
                                uint32_t capacity,
                                ds4_ling3vl_media *out) {
    if (!prompt || !prompt->v || prompt->len <= 0 || prompt->len > (int)L3V_CONTEXT ||
        !images || !count || count > LING3VL_MEDIA_INPUTS || !info || !out ||
        !capacity || capacity > LING3VL_MEDIA_ROWS) {
        return false;
    }
    ds4_ling3vl_media media = {0};
    media.capacity = capacity;
    unsigned previous = 0u;
    for (unsigned i = 0; i < count; i++) {
        const unsigned start = images[i].token_offset;
        const unsigned n = info[i].token_count;
        if (!n || n != info[i].grid_h * info[i].grid_w / 4u || start < previous ||
            start >= (unsigned)prompt->len || n > (unsigned)prompt->len - start ||
            n > capacity - media.rows) { return false; }
        for (unsigned pos = previous; pos < start + n; pos++) {
            const int token = prompt->v[pos];
            if (token < 0 || (unsigned)token >= DS4_N_VOCAB ||
                (pos >= start ? (unsigned)token != LING3VL_IMAGE_TOKEN
                              : (unsigned)token == LING3VL_IMAGE_TOKEN)) {
                return false;
            }
        }
        media.spans[i] = (ds4_ling3vl_image_span){
            .position = start, .rows = n, .feature = media.rows,
            .grid_h = info[i].grid_h, .grid_w = info[i].grid_w};
        media.rows += n;
        previous = start + n;
    }
    for (unsigned pos = previous; pos < (unsigned)prompt->len; pos++) {
        const int token = prompt->v[pos];
        if (token < 0 || (unsigned)token >= DS4_N_VOCAB ||
            (unsigned)token == LING3VL_IMAGE_TOKEN) { return false; }
    }
    media.count = count;
    *out = media;
    return true;
}

/* Decode, resize and encode one image into the shared feature plane. */
static bool ling3vl_vision_encode_one(ds4_ling3vl_vision *v, const ds4_model *m,
                                      const ds4_ling3vl_vision_weights *w,
                                      const ds4_ling3vl_image_input *image,
                                      const ds4_ling3vl_image_info *info,
                                      ds4_gpu_tensor *features,
                                      unsigned feature_row) {
    const uint32_t patches = info->grid_h * info->grid_w;
    if (patches > v->capacity) { return false; }
    int width = 0, height = 0, channels = 0;
    uint8_t *decoded = stbi_load_from_memory(image->data, (int)image->data_len,
                                             &width, &height, &channels, 3);
    if (!decoded || width != (int)info->source_width ||
        height != (int)info->source_height) {
        stbi_image_free(decoded);
        return false;
    }
    uint8_t *resized = qwen4exp_resize_rgb_pillow(
        decoded, info->source_width, info->source_height, info->resized_width,
        info->resized_height);
    stbi_image_free(decoded);
    if (!resized) { return false; }

    float *host_patches = xmalloc((size_t)patches * L3V_VIT_PATCH_DIM * sizeof(float));
    int32_t *indices = xmalloc((size_t)patches * 4u * sizeof(int32_t));
    float *weights = xmalloc((size_t)patches * 4u * sizeof(float));
    int32_t *rows = xmalloc((size_t)patches * sizeof(int32_t));
    int32_t *cols = xmalloc((size_t)patches * sizeof(int32_t));
    int32_t *zero = xcalloc(patches, sizeof(int32_t));
    int32_t *end = xmalloc((size_t)patches * sizeof(int32_t));
    for (uint32_t i = 0; i < patches; i++) { end[i] = (int32_t)patches; }
    bool ok = ling3vl_vision_pack(resized, info->resized_width, info->grid_h,
                                  info->grid_w, host_patches, indices, weights,
                                  rows, cols);
    free(resized);
    ok = ok &&
        ds4_gpu_tensor_write(v->patches, 0, host_patches,
                             (uint64_t)patches * L3V_VIT_PATCH_DIM * sizeof(float)) &&
        ds4_gpu_tensor_write(v->pos_indices, 0, indices,
                             (uint64_t)patches * 4u * sizeof(int32_t)) &&
        ds4_gpu_tensor_write(v->pos_weights, 0, weights,
                             (uint64_t)patches * 4u * sizeof(float)) &&
        ds4_gpu_tensor_write(v->height, 0, rows, (uint64_t)patches * sizeof(int32_t)) &&
        ds4_gpu_tensor_write(v->width, 0, cols, (uint64_t)patches * sizeof(int32_t)) &&
        ds4_gpu_tensor_write(v->seg_start, 0, zero, (uint64_t)patches * sizeof(int32_t)) &&
        ds4_gpu_tensor_write(v->seg_end, 0, end, (uint64_t)patches * sizeof(int32_t)) &&
        ling3vl_vision_forward(v, m, w, patches) &&
        ds4_gpu_tensor_copy(features, (uint64_t)feature_row * L3V_HIDDEN * sizeof(float),
                            v->projected, 0,
                            (uint64_t)info->token_count * L3V_HIDDEN * sizeof(float));
    free(host_patches);
    free(indices);
    free(weights);
    free(rows);
    free(cols);
    free(zero);
    free(end);
    return ok;
}
