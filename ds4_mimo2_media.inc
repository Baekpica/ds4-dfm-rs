/* MiMo BF16 mmproj. Still images arrive with the temporal frame duplicated.
 * Video is one two-frame group per call. Sinks are an extra zero-value key.
 * Merger biases are absent and stay zero. Codec attention is causal. */
enum {
    M2V_H = 1280, M2V_HEADS = 32, M2V_KV = 8, M2V_HD = 64,
    M2V_Q = M2V_HEADS * M2V_HD, M2V_KVW = M2V_KV * M2V_HD,
    M2V_QKV = M2V_Q + 2 * M2V_KVW, M2V_FF = 4608, M2V_DEPTH = 28,
    M2V_OUT = 4096, M2V_MERGED = M2V_H * 4, M2V_MAX = 32768,
    M2A_H = 1024, M2A_HEADS = 16, M2A_HD = 64, M2A_FF = 4096,
    M2A_LAYERS = 24, M2A_LOCAL = 6, M2A_CH = 20, M2A_VOCAB = 1280,
    M2A_MELS = 128, M2A_GROUP = 4, M2A_MAX_FRAMES = 30000,
    M2B_IN, M2B_HID, M2B_NORM, M2B_QKV, M2B_ATTN, M2B_FF, M2B_FF2,
    M2B_PROJ, M2B_COS, M2B_SIN, M2B_IDX, M2B_N
};

typedef struct {
    const ds4_tensor *qkv, *out, *gate, *up, *down;
    uint64_t ln1, ln2, qkv_b, out_b, gate_b, up_b, down_b, sinks;
    int mode;
} m2v_layer;

typedef struct {
    const ds4_tensor *q, *k, *v, *out, *up, *down, *gate;
    uint64_t ln1, ln1_b, ln2, ln2_b, q_b, k_b, v_b, out_b, up_b, down_b;
    int have_k_b, have_out_b, have_ff_b, window;
} m2a_layer;

typedef struct {
    bool bound;
    const ds4_tensor *patch0, *patch1, *mm0, *mm2;
    uint64_t post_w;
    int vision_window;
    m2v_layer vision[M2V_DEPTH];
    const ds4_tensor *conv1, *conv2, *down_conv, *codebook, *codes, *proj1, *proj2;
    uint64_t conv1_b, conv2_b, post_a, post_ab, down_w, down_b, local_norm;
    int bins[M2A_CH], max_bins;
    m2a_layer codec[M2A_LAYERS];
    m2a_layer local[M2A_LOCAL];
    ds4_gpu_tensor *buf[M2B_N];
} ds4_mimo2_media;

static const ds4_tensor *m2_req(const ds4_model *m, const char *name,
                                uint32_t type, uint32_t ndim, const uint64_t *dims) {
    const ds4_tensor *t = required_tensor(m, name);
    if (t->type != type || t->ndim != ndim) { ds4_die(name); }
    for (uint32_t d = 0; d < ndim; d++) {
        if (t->dim[d] != dims[d]) { ds4_die(name); }
    }
    return t;
}

static bool m2_i32s(const ds4_model *m, const char *key, int *out, uint32_t n) {
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) || arr.type != GGUF_VALUE_INT32 || arr.len != n) {
        return false;
    }
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < n; i++) {
        int32_t value = 0;
        if (!cursor_read(&c, &value, sizeof(value))) { return false; }
        out[i] = value;
    }
    return true;
}

static ds4_gpu_tensor *m2_slot(ds4_mimo2_media *media, int slot, uint64_t bytes) {
    ds4_gpu_tensor **buf = &media->buf[slot];
    if (*buf && ds4_gpu_tensor_bytes(*buf) >= bytes) { return *buf; }
    ds4_gpu_tensor_free(*buf);
    *buf = ds4_gpu_tensor_alloc(bytes);
    return *buf;
}

static void mimo2_media_free(ds4_mimo2_media *media) {
    if (!media) { return; }
    for (int i = 0; i < M2B_N; i++) {
        ds4_gpu_tensor_free(media->buf[i]);
        media->buf[i] = NULL;
    }
}

static int m2_finite(const float *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(data[i])) { return 0; }
    }
    return 1;
}

static void m2_rope_1d(float *cos, float *sin, const uint32_t *pos,
                       uint32_t n, uint32_t hd, float theta) {
    float inv[32];
    const uint32_t half = hd / 2;
    for (uint32_t d = 0; d < half; d++) { inv[d] = powf(theta, -(2.f * d) / (float)hd); }
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t d = 0; d < half; d++) {
            const float phase = pos[i] * inv[d];
            cos[(size_t)i * hd + d] = cosf(phase);
            sin[(size_t)i * hd + d] = sinf(phase);
            cos[(size_t)i * hd + half + d] = cos[(size_t)i * hd + d];
            sin[(size_t)i * hd + half + d] = sin[(size_t)i * hd + d];
        }
    }
}

static int m2_upload_rope(ds4_mimo2_media *media, const float *cos, const float *sin, uint32_t n, uint32_t hd) {
    const uint64_t bytes = (uint64_t)n * hd * sizeof(float);
    ds4_gpu_tensor *c = m2_slot(media, M2B_COS, bytes);
    ds4_gpu_tensor *s = m2_slot(media, M2B_SIN, bytes);
    return c && s &&
        ds4_gpu_tensor_write(c, 0, cos, bytes) &&
        ds4_gpu_tensor_write(s, 0, sin, bytes);
}

static void mimo2_media_bind(ds4_mimo2_media *w, const ds4_model *m) {
    ds4_str arch = {0}, vision = {0}, audio = {0};
    int vision_mode[M2V_DEPTH], audio_mode[M2A_LAYERS];
    memset(w, 0, sizeof(*w));
    if (!model_get_string(m, "general.architecture", &arch) || !ds4_streq(arch, "clip") ||
        !model_get_string(m, "clip.vision.projector_type", &vision) || !ds4_streq(vision, "mimovl") ||
        !model_get_string(m, "clip.audio.projector_type", &audio) || !ds4_streq(audio, "mimo_audio") ||
        !m2_i32s(m, "clip.vision.wa_pattern_mode", vision_mode, M2V_DEPTH) ||
        !m2_i32s(m, "clip.audio.wa_pattern_mode", audio_mode, M2A_LAYERS) ||
        !m2_i32s(m, "clip.audio.rvq.codebook_size", w->bins, M2A_CH)) {
        ds4_die("MiMo mmproj metadata does not match the V2.6 projector");
    }
    w->vision_window = 64;
    model_get_u32(m, "clip.vision.window_size", (uint32_t *)&w->vision_window);
    const uint64_t patch[4] = {16, 16, 3, M2V_H};
    const uint64_t h1[1] = {M2V_H};
    const uint64_t qkv[2] = {M2V_H, M2V_QKV};
    const uint64_t qkv_b[1] = {M2V_QKV};
    const uint64_t outw[2] = {M2V_Q, M2V_H};
    const uint64_t ffw[2] = {M2V_H, M2V_FF};
    const uint64_t ffb[1] = {M2V_FF};
    const uint64_t down[2] = {M2V_FF, M2V_H};
    const uint64_t sinks[1] = {M2V_HEADS};
    const uint64_t mm0[2] = {M2V_MERGED, M2V_MERGED};
    const uint64_t mm2[2] = {M2V_MERGED, M2V_OUT};
    w->patch0 = m2_req(m, "v.patch_embd.weight", DS4_TENSOR_F32, 4, patch);
    w->patch1 = m2_req(m, "v.patch_embd.weight.1", DS4_TENSOR_F32, 4, patch);
    w->post_w = m2_req(m, "v.post_ln.weight", DS4_TENSOR_F32, 1, h1)->abs_offset;
    w->mm0 = m2_req(m, "mm.0.weight", DS4_TENSOR_BF16, 2, mm0);
    w->mm2 = m2_req(m, "mm.2.weight", DS4_TENSOR_BF16, 2, mm2);
    for (uint32_t il = 0; il < M2V_DEPTH; il++) {
        char name[64];
        m2v_layer *layer = &w->vision[il];
        layer->mode = vision_mode[il];
#define M2_BIND(field, suffix, type, ndim, dims)                               \
        do {                                                                   \
            snprintf(name, sizeof(name), "v.blk.%u." suffix, il);              \
            field = m2_req(m, name, type, ndim, dims);                         \
        } while (0)
        M2_BIND(layer->qkv, "attn_qkv.weight", DS4_TENSOR_BF16, 2, qkv);
        M2_BIND(layer->out, "attn_out.weight", DS4_TENSOR_BF16, 2, outw);
        M2_BIND(layer->gate, "ffn_gate.weight", DS4_TENSOR_BF16, 2, ffw);
        M2_BIND(layer->up, "ffn_up.weight", DS4_TENSOR_BF16, 2, ffw);
        M2_BIND(layer->down, "ffn_down.weight", DS4_TENSOR_BF16, 2, down);
#undef M2_BIND
        snprintf(name, sizeof(name), "v.blk.%u.ln1.weight", il);
        layer->ln1 = m2_req(m, name, DS4_TENSOR_F32, 1, h1)->abs_offset;
        snprintf(name, sizeof(name), "v.blk.%u.ln2.weight", il);
        layer->ln2 = m2_req(m, name, DS4_TENSOR_F32, 1, h1)->abs_offset;
        snprintf(name, sizeof(name), "v.blk.%u.attn_qkv.bias", il);
        layer->qkv_b = m2_req(m, name, DS4_TENSOR_F32, 1, qkv_b)->abs_offset;
        snprintf(name, sizeof(name), "v.blk.%u.attn_out.bias", il);
        layer->out_b = m2_req(m, name, DS4_TENSOR_F32, 1, h1)->abs_offset;
        snprintf(name, sizeof(name), "v.blk.%u.ffn_gate.bias", il);
        layer->gate_b = m2_req(m, name, DS4_TENSOR_F32, 1, ffb)->abs_offset;
        snprintf(name, sizeof(name), "v.blk.%u.ffn_up.bias", il);
        layer->up_b = m2_req(m, name, DS4_TENSOR_F32, 1, ffb)->abs_offset;
        snprintf(name, sizeof(name), "v.blk.%u.ffn_down.bias", il);
        layer->down_b = m2_req(m, name, DS4_TENSOR_F32, 1, h1)->abs_offset;
        if (layer->mode == -1) { continue; }
        snprintf(name, sizeof(name), "v.blk.%u.attn_sinks", il);
        layer->sinks = m2_req(m, name, DS4_TENSOR_F32, 1, sinks)->abs_offset;
    }
    const uint64_t c1[3] = {3, M2A_MELS, M2A_H};
    const uint64_t c2[3] = {3, M2A_H, M2A_H};
    const uint64_t bias1[2] = {1, M2A_H};
    const uint64_t cd[3] = {2, M2A_H, M2A_H};
    const uint64_t sq[2] = {M2A_H, M2A_H};
    const uint64_t up[2] = {M2A_H, M2A_FF};
    const uint64_t dn[2] = {M2A_FF, M2A_H};
    const uint64_t one[1] = {M2A_H};
    const uint64_t upb[1] = {M2A_FF};
    const uint64_t book[3] = {M2A_H, 1024, M2A_CH};
    const uint64_t emb[3] = {M2A_H, M2A_VOCAB, M2A_CH};
    const uint64_t p1[2] = {M2A_H * M2A_GROUP, M2A_H * M2A_GROUP * 4};
    const uint64_t p2[2] = {M2A_H * M2A_GROUP * 4, M2V_OUT};
    w->conv1 = m2_req(m, "a.conv1d.1.weight", DS4_TENSOR_F32, 3, c1);
    w->conv2 = m2_req(m, "a.conv1d.2.weight", DS4_TENSOR_F32, 3, c2);
    w->conv1_b = m2_req(m, "a.conv1d.1.bias", DS4_TENSOR_F32, 2, bias1)->abs_offset;
    w->conv2_b = m2_req(m, "a.conv1d.2.bias", DS4_TENSOR_F32, 2, bias1)->abs_offset;
    w->down_conv = m2_req(m, "a.downsample.conv.weight", DS4_TENSOR_F32, 3, cd);
    w->down_w = m2_req(m, "a.downsample.norm.weight", DS4_TENSOR_F32, 1, one)->abs_offset;
    w->down_b = m2_req(m, "a.downsample.norm.bias", DS4_TENSOR_F32, 1, one)->abs_offset;
    w->post_a = m2_req(m, "a.post_ln.weight", DS4_TENSOR_F32, 1, one)->abs_offset;
    w->post_ab = m2_req(m, "a.post_ln.bias", DS4_TENSOR_F32, 1, one)->abs_offset;
    w->codebook = m2_req(m, "a.rvq.codebook.weight", DS4_TENSOR_F32, 3, book);
    w->codes = m2_req(m, "mm.a.code_embd.weight", DS4_TENSOR_F32, 3, emb);
    w->proj1 = m2_req(m, "mm.a.mlp.1.weight", DS4_TENSOR_BF16, 2, p1);
    w->proj2 = m2_req(m, "mm.a.mlp.2.weight", DS4_TENSOR_BF16, 2, p2);
    w->local_norm = m2_req(m, "mm.a.local_norm.weight", DS4_TENSOR_F32, 1, one)->abs_offset;
    w->max_bins = 1024;
    for (int q = 0; q < M2A_CH; q++) {
        if (w->bins[q] <= 0 || w->bins[q] > w->max_bins) { ds4_die("MiMo RVQ bins"); }
    }
    for (uint32_t il = 0; il < M2A_LAYERS; il++) {
        char name[64];
        m2a_layer *layer = &w->codec[il];
        layer->window = audio_mode[il] == -1 ? -1 : 128;
        model_get_u32(m, "clip.audio.window_size", (uint32_t *)&layer->window);
        if (audio_mode[il] == -1) { layer->window = -1; }
#define M2A(field, suffix, type, ndim, dims)                                   \
        do {                                                                   \
            snprintf(name, sizeof(name), "a.blk.%u." suffix, il);              \
            field = m2_req(m, name, type, ndim, dims);                         \
        } while (0)
        M2A(layer->q, "attn_q.weight", DS4_TENSOR_BF16, 2, sq);
        M2A(layer->k, "attn_k.weight", DS4_TENSOR_BF16, 2, sq);
        M2A(layer->v, "attn_v.weight", DS4_TENSOR_BF16, 2, sq);
        M2A(layer->out, "attn_out.weight", DS4_TENSOR_BF16, 2, sq);
        M2A(layer->up, "ffn_up.weight", DS4_TENSOR_BF16, 2, up);
        M2A(layer->down, "ffn_down.weight", DS4_TENSOR_BF16, 2, dn);
#undef M2A
        snprintf(name, sizeof(name), "a.blk.%u.ln1.weight", il);
        layer->ln1 = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "a.blk.%u.ln1.bias", il);
        layer->ln1_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "a.blk.%u.ln2.weight", il);
        layer->ln2 = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "a.blk.%u.ln2.bias", il);
        layer->ln2_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "a.blk.%u.attn_q.bias", il);
        layer->q_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "a.blk.%u.attn_v.bias", il);
        layer->v_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "a.blk.%u.attn_out.bias", il);
        layer->out_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        layer->have_out_b = 1;
        snprintf(name, sizeof(name), "a.blk.%u.ffn_up.bias", il);
        layer->up_b = m2_req(m, name, DS4_TENSOR_F32, 1, upb)->abs_offset;
        snprintf(name, sizeof(name), "a.blk.%u.ffn_down.bias", il);
        layer->down_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        layer->have_ff_b = 1;
    }
    for (uint32_t il = 0; il < M2A_LOCAL; il++) {
        char name[80];
        m2a_layer *layer = &w->local[il];
        layer->window = -1;
#define M2L(field, suffix, type, ndim, dims)                                   \
        do {                                                                   \
            snprintf(name, sizeof(name), "mm.a.local_blk.%u." suffix, il);     \
            field = m2_req(m, name, type, ndim, dims);                         \
        } while (0)
        M2L(layer->q, "attn_q.weight", DS4_TENSOR_BF16, 2, sq);
        M2L(layer->k, "attn_k.weight", DS4_TENSOR_BF16, 2, sq);
        M2L(layer->v, "attn_v.weight", DS4_TENSOR_BF16, 2, sq);
        M2L(layer->out, "attn_out.weight", DS4_TENSOR_BF16, 2, sq);
        M2L(layer->gate, "ffn_gate.weight", DS4_TENSOR_BF16, 2, up);
        M2L(layer->up, "ffn_up.weight", DS4_TENSOR_BF16, 2, up);
        M2L(layer->down, "ffn_down.weight", DS4_TENSOR_BF16, 2, dn);
#undef M2L
        snprintf(name, sizeof(name), "mm.a.local_blk.%u.ln1.weight", il);
        layer->ln1 = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "mm.a.local_blk.%u.ln2.weight", il);
        layer->ln2 = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "mm.a.local_blk.%u.attn_q.bias", il);
        layer->q_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        snprintf(name, sizeof(name), "mm.a.local_blk.%u.attn_k.bias", il);
        layer->k_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
        layer->have_k_b = 1;
        snprintf(name, sizeof(name), "mm.a.local_blk.%u.attn_v.bias", il);
        layer->v_b = m2_req(m, name, DS4_TENSOR_F32, 1, one)->abs_offset;
    }
    w->bound = true;
}

static int m2_fail(const char *what) {
    fprintf(stderr, "ds4: MiMo media failed: %s\n", what);
    return 1;
}

static int mimo2_vision_rows(ds4_mimo2_media *media, const ds4_model *model,
                             const float *patches, uint32_t n, uint32_t gh, uint32_t gw,
                             float *out) {
    if (!media || !media->bound || !model || !patches || !out || !n || n != gh * gw ||
        n > M2V_MAX || (gh % 2) || (gw % 2) || (n % 4)) {
        return m2_fail("vision shape");
    }
    const uint32_t units = n / 4;
    const uint64_t hidden_bytes = (uint64_t)n * M2V_H * sizeof(float);
    ds4_gpu_tensor *in = m2_slot(media, M2B_IN, (uint64_t)n * 3 * 2 * 16 * 16 * sizeof(float));
    ds4_gpu_tensor *hid = m2_slot(media, M2B_HID, hidden_bytes);
    ds4_gpu_tensor *norm = m2_slot(media, M2B_NORM, hidden_bytes);
    ds4_gpu_tensor *qkv = m2_slot(media, M2B_QKV, (uint64_t)n * M2V_QKV * sizeof(float));
    ds4_gpu_tensor *attn = m2_slot(media, M2B_ATTN, (uint64_t)n * M2V_Q * sizeof(float));
    ds4_gpu_tensor *ff = m2_slot(media, M2B_FF, (uint64_t)n * M2V_FF * sizeof(float));
    ds4_gpu_tensor *ff2 = m2_slot(media, M2B_FF2, (uint64_t)n * M2V_FF * sizeof(float));
    ds4_gpu_tensor *proj = m2_slot(media, M2B_PROJ, (uint64_t)units * M2V_OUT * sizeof(float));
    if (!in || !hid || !norm || !qkv || !attn || !ff || !ff2 || !proj ||
        !ds4_gpu_tensor_write(in, 0, patches, (uint64_t)n * 1536 * sizeof(float)) ||
        !ds4_gpu_mimo2_patch(hid, in, model->map, model->size,
                             media->patch0->abs_offset, media->patch1->abs_offset,
                             n, M2V_H, 3, 2, 16)) {
        return m2_fail("patch");
    }
    if (getenv("DS4_MIMO2_VISION_TRACE")) {
        float probe[4] = {0};
        if (!ds4_gpu_tensor_read(hid, 0, probe, sizeof(probe))) { return m2_fail("patch read"); }
        fprintf(stderr, "mimo-patch in0=%.8g out=%.8g %.8g %.8g %.8g\n",
                patches[0], probe[0], probe[1], probe[2], probe[3]);
    }
    uint32_t *h = calloc(n, sizeof(uint32_t));
    uint32_t *wv = calloc(n, sizeof(uint32_t));
    int *idx = calloc(units, sizeof(int));
    int *inv = calloc(units, sizeof(int));
    float *cos = calloc((size_t)n * M2V_HD, sizeof(float));
    float *sin = calloc((size_t)n * M2V_HD, sizeof(float));
    float *cos_col = calloc((size_t)n * M2V_HD, sizeof(float));
    float *sin_col = calloc((size_t)n * M2V_HD, sizeof(float));
    if (!h || !wv || !idx || !inv || !cos || !sin || !cos_col || !sin_col) {
        free(h); free(wv); free(idx); free(inv); free(cos); free(sin); free(cos_col); free(sin_col);
        return m2_fail("vision scratch");
    }
    uint32_t cursor = 0;
    const uint32_t llm_h = gh / 2, llm_w = gw / 2;
    for (uint32_t ty = 0; ty < llm_h; ty++) {
        for (uint32_t tx = 0; tx < llm_w; tx++) {
            for (uint32_t dy = 0; dy < 2; dy++) {
                for (uint32_t dx = 0; dx < 2; dx++) {
                    h[cursor] = ty * 2 + dy;
                    wv[cursor] = tx * 2 + dx;
                    cursor++;
                }
            }
        }
    }
    for (uint32_t r = 0; r < llm_h; r++) {
        for (uint32_t c = 0; c < llm_w; c++) { idx[c * llm_h + r] = (int)(r * llm_w + c); }
    }
    for (uint32_t u = 0; u < units; u++) { inv[idx[u]] = (int)u; }
    float invf[16];
    for (int d = 0; d < 16; d++) { invf[d] = powf(10000.f, -(2.f * d) / 32.f); }
    for (uint32_t i = 0; i < n; i++) {
        float emb[64];
        for (int d = 0; d < 16; d++) {
            emb[d] = h[i] * invf[d];
            emb[16 + d] = wv[i] * invf[d];
        }
        for (int d = 0; d < 32; d++) { emb[32 + d] = emb[d]; }
        for (int d = 0; d < 64; d++) {
            cos[(size_t)i * 64 + d] = cosf(emb[d]);
            sin[(size_t)i * 64 + d] = sinf(emb[d]);
        }
    }
    for (uint32_t u = 0; u < units; u++) {
        const uint32_t src = (uint32_t)idx[u];
        memcpy(cos_col + ((size_t)u * 4) * 64, cos + ((size_t)src * 4) * 64, 4 * 64 * sizeof(float));
        memcpy(sin_col + ((size_t)u * 4) * 64, sin + ((size_t)src * 4) * 64, 4 * 64 * sizeof(float));
    }
    int prev = -1, rc = 0;
    for (uint32_t il = 0; il < M2V_DEPTH && !rc; il++) {
        const m2v_layer *layer = &media->vision[il];
        const int col = layer->mode == 1;
        if (col && prev != 1) {
            ds4_gpu_tensor *id = m2_slot(media, M2B_IDX, units * sizeof(int));
            rc = !id || !ds4_gpu_tensor_write(id, 0, idx, units * sizeof(int)) ||
                !ds4_gpu_mimo2_gather(norm, hid, id, units, M2V_H * 4) ||
                !ds4_gpu_tensor_copy(hid, 0, norm, 0, hidden_bytes);
        } else if (!col && prev == 1) {
            ds4_gpu_tensor *id = m2_slot(media, M2B_IDX, units * sizeof(int));
            rc = !id || !ds4_gpu_tensor_write(id, 0, inv, units * sizeof(int)) ||
                !ds4_gpu_mimo2_gather(norm, hid, id, units, M2V_H * 4) ||
                !ds4_gpu_tensor_copy(hid, 0, norm, 0, hidden_bytes);
        }
        if (rc) { break; }
        rc = !m2_upload_rope(media, col ? cos_col : cos, col ? sin_col : sin, n, M2V_HD) ||
            !ds4_gpu_rms_norm_weight_rows_tensor(norm, hid, model->map, model->size,
                                                 layer->ln1, M2V_H, n, 1e-6f) ||
            !plain_graph_matmul_tensor(qkv, model, layer->qkv, M2V_H, M2V_QKV, norm, n) ||
            !ds4_gpu_mimo2_bias(qkv, model->map, model->size, layer->qkv_b, n, M2V_QKV) ||
            !ds4_gpu_mimo2_rope(qkv, media->buf[M2B_COS], media->buf[M2B_SIN],
                                n, M2V_HEADS, M2V_HD, M2V_QKV, 0) ||
            !ds4_gpu_mimo2_rope(qkv, media->buf[M2B_COS], media->buf[M2B_SIN],
                                n, M2V_KV, M2V_HD, M2V_QKV, M2V_Q) ||
            !ds4_gpu_mimo2_attn(attn, qkv, qkv, qkv, model->map, model->size, layer->sinks,
                                layer->mode != -1, n, M2V_HEADS, M2V_KV, M2V_HD,
                                M2V_QKV, M2V_QKV, M2V_QKV, 0, M2V_Q, M2V_Q + M2V_KVW,
                                layer->mode == -1 ? -1 : media->vision_window, 0, 0) ||
            !plain_graph_matmul_tensor(norm, model, layer->out, M2V_Q, M2V_H, attn, n) ||
            !ds4_gpu_mimo2_bias(norm, model->map, model->size, layer->out_b, n, M2V_H) ||
            !ds4_gpu_exaone_add_tensor(hid, norm, (uint64_t)n * M2V_H) ||
            !ds4_gpu_rms_norm_weight_rows_tensor(norm, hid, model->map, model->size,
                                                 layer->ln2, M2V_H, n, 1e-6f) ||
            !plain_graph_matmul_tensor(ff, model, layer->gate, M2V_H, M2V_FF, norm, n) ||
            !plain_graph_matmul_tensor(ff2, model, layer->up, M2V_H, M2V_FF, norm, n) ||
            !ds4_gpu_mimo2_swiglu(ff, ff, ff2, model->map, model->size,
                                  layer->gate_b, layer->up_b, 1, n, M2V_FF) ||
            !plain_graph_matmul_tensor(norm, model, layer->down, M2V_FF, M2V_H, ff, n) ||
            !ds4_gpu_mimo2_bias(norm, model->map, model->size, layer->down_b, n, M2V_H) ||
            !ds4_gpu_exaone_add_tensor(hid, norm, (uint64_t)n * M2V_H);
        prev = layer->mode;
    }
    if (!rc && prev == 1) {
        ds4_gpu_tensor *id = m2_slot(media, M2B_IDX, units * sizeof(int));
        rc = !id || !ds4_gpu_tensor_write(id, 0, inv, units * sizeof(int)) ||
            !ds4_gpu_mimo2_gather(norm, hid, id, units, M2V_H * 4) ||
            !ds4_gpu_tensor_copy(hid, 0, norm, 0, hidden_bytes);
    }
    rc = rc ||
        !ds4_gpu_mimo2_ln(norm, hid, model->map, model->size, media->post_w, 0, 0,
                          n, M2V_H, 1e-6f) ||
        !plain_graph_matmul_tensor(hid, model, media->mm0, M2V_MERGED, M2V_MERGED, norm, units) ||
        !ds4_gpu_mimo2_gelu(hid, units * M2V_MERGED) ||
        !plain_graph_matmul_tensor(proj, model, media->mm2, M2V_MERGED, M2V_OUT, hid, units) ||
        !ds4_gpu_tensor_read(proj, 0, out, (uint64_t)units * M2V_OUT * sizeof(float)) ||
        !m2_finite(out, (size_t)units * M2V_OUT);
    if (!rc && getenv("DS4_MIMO2_VISION_TRACE")) {
        double sum = 0.0;
        const size_t count = (size_t)units * M2V_OUT;
        for (size_t i = 0; i < count; i++) { sum += out[i] < 0 ? -(double)out[i] : out[i]; }
        fprintf(stderr, "mimo-out o0=%.8g o1=%.8g mean_abs=%.8g\n",
                out[0], out[1], sum / (double)count);
    }
    free(h); free(wv); free(idx); free(inv); free(cos); free(sin); free(cos_col); free(sin_col);
    return rc ? m2_fail("vision") : 0;
}

static int m2_mha(ds4_mimo2_media *media, const ds4_model *model, ds4_gpu_tensor *hidden,
                  ds4_gpu_tensor *norm, uint32_t n, const m2a_layer *layer,
                  int causal, int group) {
    const uint32_t width = M2A_HEADS * M2A_HD;
    ds4_gpu_tensor *q = m2_slot(media, M2B_QKV, (uint64_t)n * width * sizeof(float));
    ds4_gpu_tensor *k = m2_slot(media, M2B_ATTN, (uint64_t)n * width * sizeof(float));
    ds4_gpu_tensor *v = m2_slot(media, M2B_FF, (uint64_t)n * width * sizeof(float));
    ds4_gpu_tensor *attn = m2_slot(media, M2B_FF2, (uint64_t)n * width * sizeof(float));
    ds4_gpu_tensor *proj = m2_slot(media, M2B_PROJ, (uint64_t)n * width * sizeof(float));
    if (!q || !k || !v || !attn || !proj ||
        !plain_graph_matmul_tensor(q, model, layer->q, M2A_H, width, norm, n) ||
        !plain_graph_matmul_tensor(k, model, layer->k, M2A_H, width, norm, n) ||
        !plain_graph_matmul_tensor(v, model, layer->v, M2A_H, width, norm, n) ||
        !ds4_gpu_mimo2_bias(q, model->map, model->size, layer->q_b, n, width) ||
        (layer->have_k_b && !ds4_gpu_mimo2_bias(k, model->map, model->size, layer->k_b, n, width)) ||
        !ds4_gpu_mimo2_bias(v, model->map, model->size, layer->v_b, n, width) ||
        !ds4_gpu_mimo2_rope(q, media->buf[M2B_COS], media->buf[M2B_SIN], n, M2A_HEADS, M2A_HD, width, 0) ||
        !ds4_gpu_mimo2_rope(k, media->buf[M2B_COS], media->buf[M2B_SIN], n, M2A_HEADS, M2A_HD, width, 0) ||
        !ds4_gpu_mimo2_attn(attn, q, k, v, NULL, 0, 0, 0, n, M2A_HEADS, M2A_HEADS, M2A_HD,
                            width, width, width, 0, 0, 0, layer->window, causal, group) ||
        !plain_graph_matmul_tensor(proj, model, layer->out, width, M2A_H, attn, n) ||
        (layer->have_out_b && !ds4_gpu_mimo2_bias(proj, model->map, model->size, layer->out_b, n, M2A_H)) ||
        !ds4_gpu_exaone_add_tensor(hidden, proj, (uint64_t)n * M2A_H)) {
        return 0;
    }
    return 1;
}

static int mimo2_audio_rows(ds4_mimo2_media *media, const ds4_model *model,
                            const float *mel, uint32_t frames, float *out,
                            uint32_t cap, uint32_t *rows_out) {
    if (!media || !media->bound || !model || !mel || !out || !rows_out ||
        !frames || frames > M2A_MAX_FRAMES) {
        return m2_fail("audio shape");
    }
    const uint32_t n2 = (frames - 1) / 2 + 1;
    const uint32_t n3 = n2 / 2 + (n2 % 2);
    const uint32_t padded = (n3 + M2A_GROUP - 1) / M2A_GROUP * M2A_GROUP;
    const uint32_t groups = padded / M2A_GROUP;
    if (groups == 0 || groups > cap) { return m2_fail("audio rows"); }
    float *packed = calloc((size_t)M2A_MELS * frames, sizeof(float));
    if (!packed) { return m2_fail("audio scratch"); }
    for (uint32_t frame = 0; frame < frames; frame++) {
        for (uint32_t bin = 0; bin < M2A_MELS; bin++) {
            packed[frame + (size_t)frames * bin] = mel[(size_t)frame * M2A_MELS + bin];
        }
    }
    ds4_gpu_tensor *in = m2_slot(media, M2B_IN, (uint64_t)M2A_MELS * frames * sizeof(float));
    ds4_gpu_tensor *hid = m2_slot(media, M2B_HID, (uint64_t)frames * M2A_H * sizeof(float));
    ds4_gpu_tensor *norm = m2_slot(media, M2B_NORM, (uint64_t)frames * M2A_H * sizeof(float));
    int rc = !in || !hid || !norm ||
        !ds4_gpu_tensor_write(in, 0, packed, (uint64_t)M2A_MELS * frames * sizeof(float)) ||
        !ds4_gpu_mimo2_conv1d(hid, in, model->map, model->size, media->conv1->abs_offset,
                              media->conv1_b, 1, frames, frames, M2A_MELS, M2A_H, 3, 1, 1) ||
        !ds4_gpu_mimo2_gelu(hid, frames * M2A_H) ||
        !ds4_gpu_mimo2_to_ctime(norm, hid, frames, M2A_H) ||
        !ds4_gpu_mimo2_conv1d(hid, norm, model->map, model->size, media->conv2->abs_offset,
                              media->conv2_b, 1, frames, n2, M2A_H, M2A_H, 3, 2, 1) ||
        !ds4_gpu_mimo2_gelu(hid, n2 * M2A_H);
    free(packed);
    uint32_t *pos = calloc(n2 > padded ? n2 : padded, sizeof(uint32_t));
    float *cos = calloc((size_t)(n2 > padded ? n2 : padded) * M2A_HD, sizeof(float));
    float *sin = calloc((size_t)(n2 > padded ? n2 : padded) * M2A_HD, sizeof(float));
    if (!pos || !cos || !sin) {
        free(pos); free(cos); free(sin);
        return m2_fail("audio rope");
    }
    for (uint32_t i = 0; i < n2; i++) { pos[i] = i; }
    m2_rope_1d(cos, sin, pos, n2, M2A_HD, 10000.f);
    ds4_gpu_tensor *skip = NULL;
    for (uint32_t il = 0; il < M2A_LAYERS && !rc; il++) {
        const m2a_layer *layer = &media->codec[il];
        rc = !m2_upload_rope(media, cos, sin, n2, M2A_HD) ||
            !ds4_gpu_mimo2_ln(norm, hid, model->map, model->size, layer->ln1, layer->ln1_b, 1,
                              n2, M2A_H, 1e-5f) ||
            !m2_mha(media, model, hid, norm, n2, layer, 1, 0) ||
            !ds4_gpu_mimo2_ln(norm, hid, model->map, model->size, layer->ln2, layer->ln2_b, 1,
                              n2, M2A_H, 1e-5f);
        ds4_gpu_tensor *up = m2_slot(media, M2B_FF, (uint64_t)n2 * M2A_FF * sizeof(float));
        ds4_gpu_tensor *proj = m2_slot(media, M2B_PROJ, (uint64_t)n2 * M2A_H * sizeof(float));
        rc = rc || !up || !proj ||
            !plain_graph_matmul_tensor(up, model, layer->up, M2A_H, M2A_FF, norm, n2) ||
            !ds4_gpu_mimo2_bias(up, model->map, model->size, layer->up_b, n2, M2A_FF) ||
            !ds4_gpu_mimo2_gelu(up, n2 * M2A_FF) ||
            !plain_graph_matmul_tensor(proj, model, layer->down, M2A_FF, M2A_H, up, n2) ||
            !ds4_gpu_mimo2_bias(proj, model->map, model->size, layer->down_b, n2, M2A_H) ||
            !ds4_gpu_exaone_add_tensor(hid, proj, (uint64_t)n2 * M2A_H);
        if (il == 2) {
            skip = m2_slot(media, M2B_IN, (uint64_t)n2 * M2A_H * sizeof(float));
            rc = rc || !skip || !ds4_gpu_tensor_copy(skip, 0, hid, 0, (uint64_t)n2 * M2A_H * sizeof(float));
        }
    }
    rc = rc || !skip || !ds4_gpu_exaone_add_tensor(hid, skip, (uint64_t)n2 * M2A_H) ||
        !ds4_gpu_mimo2_ln(norm, hid, model->map, model->size, media->post_a, media->post_ab, 1,
                          n2, M2A_H, 1e-5f) ||
        !ds4_gpu_tensor_copy(hid, 0, norm, 0, (uint64_t)n2 * M2A_H * sizeof(float));
    ds4_gpu_tensor *ctime = m2_slot(media, M2B_QKV, (uint64_t)n2 * M2A_H * sizeof(float));
    ds4_gpu_tensor *down = m2_slot(media, M2B_NORM, (uint64_t)n3 * M2A_H * sizeof(float));
    rc = rc || !ctime || !down ||
        !ds4_gpu_mimo2_to_ctime(ctime, hid, n2, M2A_H) ||
        !ds4_gpu_mimo2_conv1d(down, ctime, model->map, model->size, media->down_conv->abs_offset,
                              0, 0, n2, n3, M2A_H, M2A_H, 2, 2, 0) ||
        !ds4_gpu_mimo2_gelu(down, n3 * M2A_H) ||
        !ds4_gpu_mimo2_ln(hid, down, model->map, model->size, media->down_w, media->down_b, 1,
                          n3, M2A_H, 1e-5f);
    ds4_gpu_tensor *ids = m2_slot(media, M2B_IDX, (uint64_t)padded * M2A_CH * sizeof(int));
    rc = rc || !ids;
    for (int q = 0; q < M2A_CH && !rc; q++) {
        const uint64_t off = media->codebook->abs_offset +
            (uint64_t)q * (uint64_t)media->max_bins * M2A_H * sizeof(float);
        ds4_gpu_tensor *idq = m2_slot(media, M2B_PROJ, (uint64_t)n3 * sizeof(int));
        rc = !idq || !ds4_gpu_mimo2_rvq(idq, hid, model->map, model->size, off, n3, M2A_H,
                                        (uint32_t)media->bins[q]);
        if (rc) { break; }
        int *host = calloc(n3, sizeof(int));
        rc = !host || !ds4_gpu_tensor_read(idq, 0, host, (uint64_t)n3 * sizeof(int));
        if (!rc) {
            int *wide = calloc(padded, sizeof(int));
            if (!wide) { rc = 1; }
            else {
                for (uint32_t row = 0; row < padded; row++) {
                    wide[row] = host[row < n3 ? row : n3 - 1];
                }
                for (uint32_t row = 0; row < padded; row++) {
                    rc = rc || !ds4_gpu_tensor_write(
                        ids, ((uint64_t)row * M2A_CH + (uint32_t)q) * sizeof(int),
                        &wide[row], sizeof(int));
                }
            }
            free(wide);
        }
        free(host);
    }
    ds4_gpu_tensor *emb = m2_slot(media, M2B_HID, (uint64_t)padded * M2A_H * sizeof(float));
    rc = rc || !emb || !ds4_gpu_mimo2_code_sum(
        emb, ids, model->map, model->size, media->codes->abs_offset,
        padded, M2A_H, M2A_VOCAB, M2A_CH);
    for (uint32_t i = 0; i < padded; i++) { pos[i] = i % M2A_GROUP; }
    m2_rope_1d(cos, sin, pos, padded, M2A_HD, 640000.f);
    for (uint32_t il = 0; il < M2A_LOCAL && !rc; il++) {
        const m2a_layer *layer = &media->local[il];
        rc = !m2_upload_rope(media, cos, sin, padded, M2A_HD) ||
            !ds4_gpu_rms_norm_weight_rows_tensor(norm, emb, model->map, model->size,
                                                 layer->ln1, M2A_H, padded, 1e-6f) ||
            !m2_mha(media, model, emb, norm, padded, layer, 0, M2A_GROUP) ||
            !ds4_gpu_rms_norm_weight_rows_tensor(norm, emb, model->map, model->size,
                                                 layer->ln2, M2A_H, padded, 1e-6f);
        ds4_gpu_tensor *gate = m2_slot(media, M2B_FF, (uint64_t)padded * M2A_FF * sizeof(float));
        ds4_gpu_tensor *up = m2_slot(media, M2B_FF2, (uint64_t)padded * M2A_FF * sizeof(float));
        ds4_gpu_tensor *proj = m2_slot(media, M2B_PROJ, (uint64_t)padded * M2A_H * sizeof(float));
        rc = rc || !gate || !up || !proj ||
            !plain_graph_matmul_tensor(gate, model, layer->gate, M2A_H, M2A_FF, norm, padded) ||
            !plain_graph_matmul_tensor(up, model, layer->up, M2A_H, M2A_FF, norm, padded) ||
            !ds4_gpu_mimo2_swiglu(gate, gate, up, NULL, 0, 0, 0, 0, padded, M2A_FF) ||
            !plain_graph_matmul_tensor(proj, model, layer->down, M2A_FF, M2A_H, gate, padded) ||
            !ds4_gpu_exaone_add_tensor(emb, proj, (uint64_t)padded * M2A_H);
    }
    ds4_gpu_tensor *flat = m2_slot(media, M2B_FF, (uint64_t)groups * M2A_H * M2A_GROUP * 4 * sizeof(float));
    ds4_gpu_tensor *proj = m2_slot(media, M2B_PROJ, (uint64_t)groups * M2V_OUT * sizeof(float));
    norm = m2_slot(media, M2B_NORM, (uint64_t)padded * M2A_H * sizeof(float));
    rc = rc || !flat || !proj || !norm ||
        !ds4_gpu_rms_norm_weight_rows_tensor(norm, emb, model->map, model->size,
                                             media->local_norm, M2A_H, padded, 1e-6f) ||
        !plain_graph_matmul_tensor(flat, model, media->proj1, M2A_H * M2A_GROUP,
                                   M2A_H * M2A_GROUP * 4, norm, groups) ||
        !ds4_gpu_mimo2_gelu(flat, groups * M2A_H * M2A_GROUP * 4) ||
        !plain_graph_matmul_tensor(proj, model, media->proj2, M2A_H * M2A_GROUP * 4,
                                   M2V_OUT, flat, groups) ||
        !ds4_gpu_tensor_read(proj, 0, out, (uint64_t)groups * M2V_OUT * sizeof(float)) ||
        !m2_finite(out, (size_t)groups * M2V_OUT);
    free(pos); free(cos); free(sin);
    if (rc) { return m2_fail("audio"); }
    *rows_out = groups;
    return 0;
}
