/* One crop's native vision executor. Rust supplies validated F16 bindings
 * and normalized CHW pixels; outputs are row-major 4096-wide embeddings. */
enum { S37V_LAYERS = 47, S37V_DIM = 1536, S37V_FFN = 8960, S37V_PATCH = 14, S37V_F16 = 1 };
typedef struct {
    ds4_tensor *qkv, *qkv_bias, *out, *out_bias, *ln1, *ln1_bias, *ln2, *ln2_bias;
    ds4_tensor *ls1, *ls2, *up, *up_bias, *down, *down_bias;
} ds4_step37_vision_layer;
typedef struct {
    ds4_tensor *patch, *position, *pre_ln, *pre_bias, *down[2], *bias[2], *projection;
    ds4_step37_vision_layer layer[S37V_LAYERS];
} ds4_step37_vision_weights;
typedef struct {
    ds4_gpu_tensor *pixels, *columns, *hidden, *norm, *qkv, *attention, *temp, *mlp;
    ds4_gpu_tensor *down1, *down2, *output, *start, *end;
    unsigned edge, rows, output_rows, capacity;
    uint64_t bytes;
} ds4_step37_vision;

static void step37_vision_bind(ds4_step37_vision_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->patch = required_tensor(m, "v.patch_embd.weight");
    w->position = required_tensor(m, "v.position_embd.weight");
    w->pre_ln = required_tensor(m, "v.pre_ln.weight");
    w->pre_bias = required_tensor(m, "v.pre_ln.bias");
    w->projection = required_tensor(m, "mm.model.fc.weight");
    for (unsigned i = 0; i < 2; i++) {
        w->down[i] = required_tensorf(m, "mm.%u.weight", i);
        w->bias[i] = required_tensorf(m, "mm.%u.bias", i);
    }
    for (unsigned i = 0; i < S37V_LAYERS; i++) {
        ds4_step37_vision_layer *l = &w->layer[i];
#define S37V_BIND(field, name) l->field = required_tensorf(m, "v.blk.%u." name, i)
        S37V_BIND(qkv, "attn_qkv.weight"); S37V_BIND(qkv_bias, "attn_qkv.bias");
        S37V_BIND(out, "attn_out.weight"); S37V_BIND(out_bias, "attn_out.bias");
        S37V_BIND(ln1, "ln1.weight"); S37V_BIND(ln1_bias, "ln1.bias");
        S37V_BIND(ln2, "ln2.weight"); S37V_BIND(ln2_bias, "ln2.bias");
        S37V_BIND(ls1, "ls1.weight"); S37V_BIND(ls2, "ls2.weight");
        S37V_BIND(up, "ffn_up.weight"); S37V_BIND(up_bias, "ffn_up.bias");
        S37V_BIND(down, "ffn_down.weight"); S37V_BIND(down_bias, "ffn_down.bias");
#undef S37V_BIND
    }
}

static void step37_vision_free(ds4_step37_vision *g) {
    ds4_gpu_tensor *all[] = {g->pixels, g->columns, g->hidden, g->norm, g->qkv,
        g->attention, g->temp, g->mlp, g->down1, g->down2, g->output, g->start, g->end};
    for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) { ds4_gpu_tensor_free(all[i]); }
    memset(g, 0, sizeof(*g));
}

static uint64_t step37_vision_bytes(unsigned edge) {
    if (edge != 504 && edge != 728) { return 0; }
    const unsigned rows = (edge / S37V_PATCH) * (edge / S37V_PATCH);
    return (uint64_t)rows * (3 * S37V_PATCH * S37V_PATCH + 10 * S37V_DIM + S37V_FFN + 4096 / 16 + 2) * sizeof(float);
}

/* One maximum-crop workspace serves patches and base crops serially. Both
 * shape and attention limits must change; stale limits attend padding rows. */
static bool step37_vision_shape(ds4_step37_vision *g, unsigned edge) {
    if (!g || (edge != 504 && edge != 728)) { return false; }
    const unsigned rows = (edge / S37V_PATCH) * (edge / S37V_PATCH);
    if (rows > g->capacity) { return false; }
    if (g->edge == edge) { return true; }
    int segments[52 * 52];
    for (unsigned i = 0; i < rows; i++) { segments[i] = (int)rows; }
    if (!ds4_gpu_tensor_write(g->end, 0, segments, rows * sizeof(int))) { return false; }
    g->edge = edge; g->rows = rows; g->output_rows = rows / 16;
    return true;
}

/* Allocate a fresh or freed graph, retaining its maximum row capacity. */
static bool step37_vision_alloc(ds4_step37_vision *g, unsigned edge) {
    memset(g, 0, sizeof(*g));
    if (edge != 504 && edge != 728) { return false; }
    g->edge = edge;
    const unsigned grid = edge / S37V_PATCH, rows = grid * grid;
    g->rows = rows; g->output_rows = rows / 16; g->capacity = rows;
#define S37V_ALLOC(field, count) do { \
    uint64_t bytes = (uint64_t)(count) * sizeof(float); \
    g->field = ds4_gpu_tensor_alloc(bytes); \
    if (!g->field) { step37_vision_free(g); return false; } \
    g->bytes += bytes; \
} while (0)
    S37V_ALLOC(pixels, edge * edge * 3);
    // The first 3x3 convolution has the largest im2col footprint. Reuse it
    // for both downsamplers and the initial patch projection.
    S37V_ALLOC(columns, rows / 4 * 9 * S37V_DIM);
    S37V_ALLOC(hidden, rows * S37V_DIM); S37V_ALLOC(norm, rows * S37V_DIM);
    S37V_ALLOC(qkv, rows * 3 * S37V_DIM); S37V_ALLOC(attention, rows * S37V_DIM);
    S37V_ALLOC(temp, rows * S37V_DIM); S37V_ALLOC(mlp, rows * S37V_FFN);
    S37V_ALLOC(down1, rows / 4 * 2 * S37V_DIM); S37V_ALLOC(down2, rows / 16 * 4 * S37V_DIM);
    S37V_ALLOC(output, g->output_rows * 4096); S37V_ALLOC(start, rows); S37V_ALLOC(end, rows);
#undef S37V_ALLOC
    int *segments = xcalloc(rows, sizeof(int));
    bool ok = ds4_gpu_tensor_write(g->start, 0, segments, rows * sizeof(int));
    for (unsigned i = 0; i < rows; i++) { segments[i] = (int)rows; }
    ok = ok && ds4_gpu_tensor_write(g->end, 0, segments, rows * sizeof(int));
    free(segments);
    if (!ok) { step37_vision_free(g); }
    return ok;
}

static bool step37_vision_mm(ds4_gpu_tensor *out, const ds4_model *m,
        const ds4_tensor *weight, const ds4_gpu_tensor *in, unsigned width,
        unsigned output, unsigned rows) {
    return weight && weight->type == S37V_F16 && ds4_gpu_matmul_f16_tensor(
        out, m->map, m->size, weight->abs_offset, width, output, in, rows);
}

static bool step37_vision_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
        const ds4_model *m, const ds4_tensor *weight, const ds4_tensor *bias, unsigned rows) {
    return ds4_gpu_qwen4exp_vision_layernorm_tensor(out, in, m->map, m->size,
        weight->abs_offset, bias->abs_offset, rows, S37V_DIM, 1e-5f);
}

#ifndef STEP37_VISION_TRACE
#define STEP37_VISION_TRACE(name, tensor, width, rows) ((void)0)
#endif
static bool step37_vision_layer(ds4_step37_vision *g, const ds4_model *m,
        const ds4_step37_vision_layer *w, unsigned index) {
    const unsigned n = g->rows;
    if (!step37_vision_norm(g->norm, g->hidden, m, w->ln1, w->ln1_bias, n) ||
        !step37_vision_mm(g->qkv, m, w->qkv, g->norm, S37V_DIM, 3 * S37V_DIM, n) ||
        !ds4_gpu_step37_vqkv(g->qkv, m->map, m->size, w->qkv_bias->abs_offset, g->edge / S37V_PATCH) ||
        !ds4_gpu_qwen4exp_vision_attention_tensor(g->attention, g->qkv, g->start, g->end, n, 16, 96) ||
        !step37_vision_mm(g->temp, m, w->out, g->attention, S37V_DIM, S37V_DIM, n) ||
        !ds4_gpu_step37_vresidual(g->hidden, g->temp, m->map, m->size,
            w->out_bias->abs_offset, w->ls1->abs_offset, n)) { return false; }
    char name[32]; snprintf(name, sizeof(name), "attn-%u", index);
    STEP37_VISION_TRACE(name, g->hidden, S37V_DIM, n);
    if (!step37_vision_norm(g->norm, g->hidden, m, w->ln2, w->ln2_bias, n) ||
        !step37_vision_mm(g->mlp, m, w->up, g->norm, S37V_DIM, S37V_FFN, n) ||
        !ds4_gpu_step37_vgelu(g->mlp, m->map, m->size, w->up_bias->abs_offset, n) ||
        !step37_vision_mm(g->temp, m, w->down, g->mlp, S37V_FFN, S37V_DIM, n) ||
        !ds4_gpu_step37_vresidual(g->hidden, g->temp, m->map, m->size,
            w->down_bias->abs_offset, w->ls2->abs_offset, n)) { return false; }
    snprintf(name, sizeof(name), "layer-%u", index);
    STEP37_VISION_TRACE(name, g->hidden, S37V_DIM, n);
    return true;
}

static bool step37_vision_forward(ds4_step37_vision *g, const ds4_model *m,
        const ds4_step37_vision_weights *w, const float *pixels) {
    if (!g || !m || !w || !pixels || (g->edge != 504 && g->edge != 728)) { return false; }
    const unsigned grid = g->edge / S37V_PATCH, n = g->rows;
    if (!ds4_gpu_tensor_write(g->pixels, 0, pixels, (uint64_t)g->edge * g->edge * 3 * sizeof(float)) ||
        !ds4_gpu_step37_columns(g->columns, g->pixels, g->edge, 3) ||
        !step37_vision_mm(g->hidden, m, w->patch, g->columns, 3 * S37V_PATCH * S37V_PATCH, S37V_DIM, n) ||
        !ds4_gpu_step37_position(g->hidden, m->map, m->size, w->position->abs_offset, grid) ||
        !step37_vision_norm(g->hidden, g->hidden, m, w->pre_ln, w->pre_bias, n)) { return false; }
    STEP37_VISION_TRACE("pre", g->hidden, S37V_DIM, n);
    for (unsigned i = 0; i < S37V_LAYERS; i++) {
        if (!step37_vision_layer(g, m, &w->layer[i], i)) { return false; }
    }
    // The source has neither a post-LN nor an activation between convolutions.
    ds4_gpu_tensor *inputs[] = {g->hidden, g->down1}, *outputs[] = {g->down1, g->down2};
    for (unsigned i = 0; i < 2; i++) {
        unsigned channels = S37V_DIM << i, edge = grid >> i, rows = edge * edge / 4;
        if (!ds4_gpu_step37_columns(g->columns, inputs[i], edge, channels) ||
            !step37_vision_mm(outputs[i], m, w->down[i], g->columns, 9 * channels, 2 * channels, rows) ||
            !ds4_gpu_qwen4exp_vision_bias_tensor(outputs[i], m->map, m->size, w->bias[i]->abs_offset,
                rows, 2 * channels)) { return false; }
        STEP37_VISION_TRACE(i == 0 ? "down-0" : "down-1", outputs[i], 2 * channels, rows);
    }
    if (!step37_vision_mm(g->output, m, w->projection, g->down2, 4 * S37V_DIM, 4096, g->output_rows)) { return false; }
    STEP37_VISION_TRACE("projection", g->output, 4096, g->output_rows);
    return true;
}
