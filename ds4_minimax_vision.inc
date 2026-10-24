/* MiniMax-M3.1 vision tower over the release's vision part (vision-bf16.gguf:
 * the source tensors under their HF names, BF16 except the F32 patch
 * embedding). The handoff's minimax_vision_ref.py is the contract:
 *
 *   patches [n][1176] ─▶ patch embedding (F32) ─▶ pre_layrnorm
 *     ─▶ 32 x [ x + out_proj(attn(rope(q), rope(k), v)) over LN1(x);
 *               x + fc2(gelu(fc1(LN2(x)))) ]          (no post-LN)
 *     ─▶ projector per patch (1280 -> 6144, GELU, 6144 -> 6144)
 *     ─▶ 2 x 2 groups of patches (24,576) ─▶ merge MLP (GELU) ─▶ [n/4][6144]
 *
 * Attention is bidirectional within a segment: an image, or at most four
 * frame pairs of a video, each with its own temporal RoPE origin. The weights
 * are never resident: each layer's tensors stream off the mapping through
 * V4.1's pinned window into reused device slots when media arrives, and the
 * part's page cache is dropped afterwards (ds4_ds41_vision.inc). Linears,
 * GELU and attention are V4.1's primitives; LayerNorm, the 3-D RoPE and the
 * patch embedding are MiniMax's (ds4_minimax_gpu.cuh). Values sit on the
 * BF16 grid where the reference's BF16 modules round. */

#ifndef DS4_NO_GPU

#define MM_VIT_DIM DS4_MINIMAX_VIT_DIM
#define MM_VIT_HEADS DS4_MINIMAX_VIT_HEADS
#define MM_VIT_HEAD DS4_MINIMAX_VIT_HEAD
#define MM_VIT_PATCH DS4_MINIMAX_VIT_PATCH

/* BF16 vector tensors (norms, biases) widened to F32 slots at `at` floats. */
static bool mm_vis_f32(const ds4_model *vm, const ds4_tensor *t, ds4_gpu_tensor **slot, uint64_t at, uint64_t floats,
        float *host) {
    bool ok = true;
    if (!v41_vis_grow(slot, floats * sizeof(float), &ok) || !ok) { return false; }
    const uint16_t *src = tensor_data(vm, t);
    const uint64_t n = t->dim[0];
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(host + i, &bits, sizeof(float));
    }
    return ds4_gpu_tensor_write(*slot, at * sizeof(float), host, n * sizeof(float)) != 0;
}

/* BF16 matrices into one slot at byte offset `at` (q, k and v stack into
 * one [3840][1280] weight), through V4.1's pinned window. */
static bool mm_vis_bf16(const ds4_model *vm, const ds4_tensor *t, ds4_gpu_tensor **slot, uint64_t at, uint64_t bytes,
        void *stage) {
    bool ok = true;
    if (!v41_vis_grow(slot, bytes, &ok) || !ok) { return false; }
    const uint8_t *src = tensor_data(vm, t);
    for (uint64_t off = 0; off < t->bytes; off += V41_VIS_STAGE_BYTES) {
        const uint64_t n = t->bytes - off < V41_VIS_STAGE_BYTES ? t->bytes - off : V41_VIS_STAGE_BYTES;
        memcpy(stage, src + off, (size_t)n);
        if (!ds4_gpu_tensor_write(*slot, at + off, stage, n)) { return false; }
    }
    return true;
}

static bool mm_vis_linear(ds4_gpu_tensor *out, const ds4_gpu_tensor *w, const ds4_gpu_tensor *b, uint64_t in,
        uint64_t outs, const ds4_gpu_tensor *x, uint32_t rows) {
    return ds4_gpu_v41_matmul_bf16_dev_tensor(out, w, in, outs, x, rows) &&
        ds4_gpu_v41_vision_bias_add_tensor(out, b, rows, (uint32_t)outs);
}

static ds4_gpu_tensor *mm_vis_view(ds4_gpu_tensor *t, uint64_t row, uint64_t rows, uint64_t width) {
    return ds4_gpu_tensor_view(t, row * width * sizeof(float), rows * width * sizeof(float));
}

/* One pass over rows [0, n) of whole segments (`pairs` frame pairs of gh x gw
 * patches; segments of MM_VIS_SEG_PAIRS pairs restart t). out: [n/4][6144]. */
static bool mm_vis_pass(const ds4_model *vm, const ds4_minimax_vision *V, const float *patches, uint32_t pairs,
        uint32_t gh, uint32_t gw, float *out, void *stage, float *host) {
    const uint32_t per = gh * gw, n = pairs * per, D = MM_VIT_DIM;
    bool ok = true;
    ds4_gpu_tensor *pix = v41_alloc((uint64_t)n * MM_VIT_PATCH * 4, &ok), *x = v41_alloc((uint64_t)n * D * 4, &ok);
    ds4_gpu_tensor *xn = v41_alloc((uint64_t)n * D * 4, &ok), *qkv = v41_alloc((uint64_t)n * 3 * D * 4, &ok);
    ds4_gpu_tensor *at = v41_alloc((uint64_t)n * D * 4, &ok), *mid = v41_alloc((uint64_t)n * MM_VIS_FF * 4, &ok);
    ds4_gpu_tensor *w[4] = {0}, *b[4] = {0};   // weight / F32 vector slots, reused across layers
    ok = ok && ds4_gpu_tensor_write(pix, 0, patches, (uint64_t)n * MM_VIT_PATCH * 4);
    // Patch embedding (F32 weight) and pre_layrnorm.
    ok = ok && mm_vis_bf16(vm, V->patch, &w[0], 0, V->patch->bytes, stage) &&
        ds4_gpu_minimax_vit_patch(x, w[0], pix, n) &&
        mm_vis_f32(vm, V->pre_w, &b[0], 0, D, host) && mm_vis_f32(vm, V->pre_b, &b[1], 0, D, host) &&
        ds4_gpu_minimax_vit_ln(x, x, b[0], b[1], n, D, 1e-5f);
    const uint64_t square = (uint64_t)D * D * 2;
    for (uint32_t il = 0; ok && il < MM_VIS_LAYERS; il++) {
        const minimax_vision_layer *l = &V->layer[il];
        // q, k, v stack into one GEMM; their biases into one F32 row.
        ok = mm_vis_f32(vm, l->ln1_w, &b[0], 0, D, host) && mm_vis_f32(vm, l->ln1_b, &b[1], 0, D, host) &&
            mm_vis_bf16(vm, l->q_w, &w[0], 0, 3 * square, stage) &&
            mm_vis_bf16(vm, l->k_w, &w[0], square, 3 * square, stage) &&
            mm_vis_bf16(vm, l->v_w, &w[0], 2 * square, 3 * square, stage) &&
            mm_vis_f32(vm, l->q_b, &b[2], 0, 3 * D, host) && mm_vis_f32(vm, l->k_b, &b[2], D, 3 * D, host) &&
            mm_vis_f32(vm, l->v_b, &b[2], 2 * D, 3 * D, host) &&
            ds4_gpu_minimax_vit_ln(xn, x, b[0], b[1], n, D, 1e-5f) &&
            mm_vis_linear(qkv, w[0], b[2], D, 3 * D, xn, n) &&
            ds4_gpu_minimax_vit_rope(qkv, n, gh, gw, MM_VIS_SEG_PAIRS);
        for (uint32_t p0 = 0; ok && p0 < pairs; p0 += MM_VIS_SEG_PAIRS) {
            const uint32_t sp = pairs - p0 < MM_VIS_SEG_PAIRS ? pairs - p0 : MM_VIS_SEG_PAIRS;
            ds4_gpu_tensor *o = mm_vis_view(at, (uint64_t)p0 * per, (uint64_t)sp * per, D);
            ds4_gpu_tensor *q = mm_vis_view(qkv, (uint64_t)p0 * per, (uint64_t)sp * per, 3 * D);
            ok = o && q && ds4_gpu_v41_dense_attn_tensor(o, q, sp * per, MM_VIT_HEADS, MM_VIT_HEAD);
            ds4_gpu_tensor_free(o);
            ds4_gpu_tensor_free(q);
        }
        ok = ok && mm_vis_bf16(vm, l->o_w, &w[1], 0, square, stage) && mm_vis_f32(vm, l->o_b, &b[3], 0, D, host) &&
            mm_vis_linear(xn, w[1], b[3], D, D, at, n) && ds4_gpu_v41_add_tensor(x, xn, (uint64_t)n * D) &&
            ds4_gpu_v41_round_bf16_tensor(x, (uint64_t)n * D) &&
            mm_vis_f32(vm, l->ln2_w, &b[0], 0, D, host) && mm_vis_f32(vm, l->ln2_b, &b[1], 0, D, host) &&
            ds4_gpu_minimax_vit_ln(xn, x, b[0], b[1], n, D, 1e-5f) &&
            mm_vis_bf16(vm, l->fc1_w, &w[2], 0, l->fc1_w->bytes, stage) &&
            mm_vis_f32(vm, l->fc1_b, &b[2], 0, MM_VIS_FF, host) &&
            mm_vis_linear(mid, w[2], b[2], D, MM_VIS_FF, xn, n) && ds4_gpu_v41_gelu_tensor(mid, (uint64_t)n * MM_VIS_FF) &&
            mm_vis_bf16(vm, l->fc2_w, &w[3], 0, l->fc2_w->bytes, stage) && mm_vis_f32(vm, l->fc2_b, &b[3], 0, D, host) &&
            mm_vis_linear(xn, w[3], b[3], MM_VIS_FF, D, mid, n) && ds4_gpu_v41_add_tensor(x, xn, (uint64_t)n * D) &&
            ds4_gpu_v41_round_bf16_tensor(x, (uint64_t)n * D);
    }
    // Projector per patch, then the merge MLP over 2 x 2 groups (4 consecutive rows).
    ds4_gpu_tensor *p1 = v41_alloc((uint64_t)n * MM_VIS_OUT * 4, &ok), *p2 = v41_alloc((uint64_t)n * MM_VIS_OUT * 4, &ok);
    const uint32_t groups = n / 4;
    ok = ok && mm_vis_bf16(vm, V->p1_w, &w[0], 0, V->p1_w->bytes, stage) &&
        mm_vis_f32(vm, V->p1_b, &b[0], 0, MM_VIS_OUT, host) &&
        mm_vis_linear(p1, w[0], b[0], D, MM_VIS_OUT, x, n) && ds4_gpu_v41_gelu_tensor(p1, (uint64_t)n * MM_VIS_OUT) &&
        mm_vis_bf16(vm, V->p2_w, &w[1], 0, V->p2_w->bytes, stage) &&
        mm_vis_f32(vm, V->p2_b, &b[1], 0, MM_VIS_OUT, host) &&
        mm_vis_linear(p2, w[1], b[1], MM_VIS_OUT, MM_VIS_OUT, p1, n) &&
        mm_vis_bf16(vm, V->m1_w, &w[2], 0, V->m1_w->bytes, stage) &&
        mm_vis_f32(vm, V->m1_b, &b[2], 0, MM_VIS_OUT, host) &&
        mm_vis_linear(p1, w[2], b[2], 4 * MM_VIS_OUT, MM_VIS_OUT, p2, groups) &&
        ds4_gpu_v41_gelu_tensor(p1, (uint64_t)groups * MM_VIS_OUT) &&
        mm_vis_bf16(vm, V->m2_w, &w[3], 0, V->m2_w->bytes, stage) &&
        mm_vis_f32(vm, V->m2_b, &b[3], 0, MM_VIS_OUT, host) &&
        mm_vis_linear(p2, w[3], b[3], MM_VIS_OUT, MM_VIS_OUT, p1, groups) &&
        ds4_gpu_tensor_read(p2, 0, out, (uint64_t)groups * MM_VIS_OUT * 4);
    ds4_gpu_tensor *all[] = {pix, x, xn, qkv, at, mid, p1, p2, w[0], w[1], w[2], w[3], b[0], b[1], b[2], b[3]};
    v41_vis_free_list(all, sizeof(all) / sizeof(all[0]));
    return ok;
}

/* Language rows of one image or video: patches [gt * gh * gw][1176] in the
 * processor's order -> out [gt * gh * gw / 4][6144] F32 (host). */
static bool minimax_vision_rows(ds4_engine *e, const float *patches, uint32_t gt, uint32_t gh, uint32_t gw,
        float *out) {
    const ds4_minimax_vision *V = &e->minimax_vision;
    const uint32_t per = gh * gw;
    if (!V->bound || !patches || !out || !gt || !gh || !gw || gh % 2 || gw % 2 || per > MM_VIS_MAX_ROWS) {
        fprintf(stderr, "ds4: MiniMax vision refuses grid %ux%ux%u\n", gt, gh, gw);
        return false;
    }
    // Whole segments per pass, as many as MM_VIS_MAX_ROWS holds.
    // ponytail: a pass restages the tower (~1.7 GB); pass layer-major across groups if long videos matter.
    uint32_t pass_pairs = MM_VIS_MAX_ROWS / per / MM_VIS_SEG_PAIRS * MM_VIS_SEG_PAIRS;
    if (!pass_pairs) { pass_pairs = MM_VIS_SEG_PAIRS <= MM_VIS_MAX_ROWS / per ? MM_VIS_SEG_PAIRS : 1; }
    if (pass_pairs < MM_VIS_SEG_PAIRS && gt > pass_pairs) {
        fprintf(stderr, "ds4: MiniMax vision: a %ux%u video segment exceeds one pass\n", gh, gw);
        return false;
    }
    void *stage = ds4_gpu_host_alloc(V41_VIS_STAGE_BYTES);
    float *host = malloc((size_t)4 * MM_VIS_OUT * sizeof(float));
    const double t0 = now_sec();
    bool ok = stage && host;
    for (uint32_t p0 = 0; ok && p0 < gt; p0 += pass_pairs) {
        const uint32_t pairs = gt - p0 < pass_pairs ? gt - p0 : pass_pairs;
        ok = mm_vis_pass(&e->vision_model, V, patches + (uint64_t)p0 * per * MM_VIT_PATCH, pairs, gh, gw,
            out + (uint64_t)p0 * per / 4 * MM_VIS_OUT, stage, host);
    }
    ok = ok && ds4_gpu_synchronize();
    if (stage) { ds4_gpu_host_free(stage); }
    free(host);
    v41_vis_drop_pages(&e->vision_model);
    fprintf(stderr, "ds4: MiniMax vision %ux%ux%u -> %u rows in %.2f s%s\n", gt, gh, gw, gt * per / 4,
        now_sec() - t0, ok ? "" : " (failed)");
    return ok;
}

int ds4_session_minimax_encode_vision(ds4_session *s, const float *patches, unsigned grid_t, unsigned grid_h,
        unsigned grid_w, float *out, char *err, size_t errlen) {
    if (!ds4_session_is_minimax(s) || !s->engine->minimax_vision.bound) {
        payload_set_err(err, errlen, "MiniMax vision part is not open");
        return 1;
    }
    if (!minimax_vision_rows(s->engine, patches, grid_t, grid_h, grid_w, out)) {
        payload_set_err(err, errlen, "MiniMax vision tower failed");
        return 1;
    }
    return 0;
}

#else
int ds4_session_minimax_encode_vision(ds4_session *s, const float *patches, unsigned grid_t, unsigned grid_h,
        unsigned grid_w, float *out, char *err, size_t errlen) {
    (void)s; (void)patches; (void)grid_t; (void)grid_h; (void)grid_w; (void)out;
    payload_set_err(err, errlen, "this build has no CUDA vision support");
    return 1;
}
#endif
