/* Inkling-2 vision tower over the release's vision part (vision-bf16.gguf,
 * BF16 source tensors). transformers Inkling2VisionModel @109a7d19 is the
 * contract (tools/vq_calib/inkling2_vision.py runs it):
 *
 *   pixels [tokens][40 x 40 x 3] ─▶ 4 patches per token (2 x 2 of 20 x 20)
 *     ─▶ patch embedding (1200 -> 1024) + the 32 x 32 positions resized
 *        bilinearly to the patch grid; one CLS row in front
 *     ─▶ pre LayerNorm ─▶ 23 x [ x + ls1 * out(attn(rope(q), rope(k), v)) over LN1(x);
 *                                x + ls2 * fc2(gelu(fc1(LN2(x)))) ]
 *     ─▶ patches without the CLS, 4 per row (4096) ─▶ proj_in, GELU, proj_out
 *     ─▶ [tokens][4608]: the rows of the image's placeholders, unnormalized
 *
 * One image is one bidirectional attention segment. The weights are never
 * resident: each layer streams through V4.1's pinned window into reused
 * device slots and the part's page cache is dropped afterwards. Linears,
 * GELU and attention are V4.1's primitives, LayerNorm and the staging
 * MiniMax's; the patch layout, positions, 2-D RoPE and layer scale are
 * Inkling-2's (ds4_inkling2_gpu.cuh). */

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)

/* Row tile: only the residual (F32) and the BF16 [q | k | v] rows every query
 * reads span the image (10 KiB per row); every other activation is at most
 * one tile, so an image of 10,609 tokens peaks near 0.85 GB. An image of one
 * tile runs the same launches as an untiled pass. */
enum { INK2_VIS_TILE = 4096, INK2_VIS_GROUP = INK2_VIS_TILE / 4, INK2_VIS_HOST = 4096 };

/* A BF16 tensor widened to F32 into `slot` at float `at` (slot holds `floats`). */
static bool ink2_vis_f32(const ds4_model *vm, const ds4_tensor *t, ds4_gpu_tensor **slot, uint64_t at,
        uint64_t floats, float *host) {
    bool ok = true;
    if (!v41_vis_grow(slot, floats * sizeof(float), &ok) || !ok) { return false; }
    const uint16_t *src = tensor_data(vm, t);
    const uint64_t n = t->bytes / sizeof(uint16_t);
    for (uint64_t i0 = 0; i0 < n; i0 += INK2_VIS_HOST) {
        const uint64_t k = n - i0 < INK2_VIS_HOST ? n - i0 : INK2_VIS_HOST;
        for (uint64_t i = 0; i < k; i++) {
            const uint32_t bits = (uint32_t)src[i0 + i] << 16;
            memcpy(host + i, &bits, sizeof(float));
        }
        if (!ds4_gpu_tensor_write(*slot, (at + i0) * sizeof(float), host, k * sizeof(float))) { return false; }
    }
    return true;
}

/* rows x cols language tokens of one image -> out [tokens][4608] (host). */
static bool ink2_vis_pass(const ds4_model *vm, const ds4_inkling2_vision *V, const float *pixels, uint32_t rows,
        uint32_t cols, float *out, void *stage, float *host) {
    const uint32_t tokens = rows * cols, n = 4 * tokens + 1, D = DS4_INK2_VIT_DIM, F = INKLING2_VIT_FF;
    const uint32_t tile = INK2_VIS_TILE, group = INK2_VIS_GROUP;
    const uint64_t square = (uint64_t)D * D * 2, values = DS4_INK2_IMAGE_VALUES;
    bool ok = true;
    ds4_gpu_tensor *x = v41_alloc((uint64_t)n * D * 4, &ok), *qkv16 = v41_alloc((uint64_t)n * 3 * D * 2, &ok);
    ds4_gpu_tensor *xn = v41_alloc((uint64_t)tile * D * 4, &ok), *qkv = v41_alloc((uint64_t)tile * 3 * D * 4, &ok);
    ds4_gpu_tensor *at = v41_alloc((uint64_t)tile * D * 4, &ok), *mid = v41_alloc((uint64_t)tile * F * 4, &ok);
    ds4_gpu_tensor *pix = v41_alloc(group * values * 4, &ok), *patches = v41_alloc(group * values * 4, &ok);
    ds4_gpu_tensor *w[4] = {0}, *b[10] = {0};   // weight / F32 vector slots, reused across layers

    // Patches by token groups, their embedding (into xn), positions and the
    // CLS row into x, then pre LN.
    ok = ok && mm_vis_bf16(vm, V->patch, &w[0], 0, V->patch->bytes, stage) &&
        ink2_vis_f32(vm, V->table, &b[0], 0, (uint64_t)DS4_INK2_VIT_GRID * DS4_INK2_VIT_GRID * D, host) &&
        ink2_vis_f32(vm, V->cls, &b[1], 0, 2 * D, host) && ink2_vis_f32(vm, V->cls_pos, &b[1], D, 2 * D, host);
    for (uint32_t t0 = 0; ok && t0 < tokens; t0 += group) {
        const uint32_t k = tokens - t0 < group ? tokens - t0 : group;
        ok = ds4_gpu_tensor_write(pix, 0, pixels + (uint64_t)t0 * values, (uint64_t)k * values * 4) &&
            ds4_gpu_ink2_vit_patches(patches, pix, k) &&
            v41_vis_proj(xn, w[0], NULL, DS4_INK2_VIT_PATCH, D, patches, 4 * k) &&
            ds4_gpu_ink2_vit_embed(x, xn, b[0], b[1], rows, cols, 4 * t0, 4 * k);
    }
    ok = ok && ink2_vis_f32(vm, V->pre_w, &b[0], 0, D, host) && ink2_vis_f32(vm, V->pre_b, &b[1], 0, D, host) &&
        ds4_gpu_minimax_vit_ln(x, x, b[0], b[1], n, D, 1e-5f);
    // cudaMalloc-owned tensors free only after their queued readers complete.
    mm_vis_free(&pix);
    mm_vis_free(&patches);

    // Second-pass row spans: whole query tiles of the attention.
    const uint32_t qc = ds4_gpu_ink2_vit_attn_rows(n), span = n <= tile ? n : qc * (tile / qc);
    for (uint32_t il = 0; ok && il < INKLING2_VIT_LAYERS; il++) {
        const inkling2_vision_layer *l = &V->layer[il];
        // q, k, v stack into one GEMM; their biases into one F32 row.
        ok = ink2_vis_f32(vm, l->ln1_w, &b[0], 0, D, host) && ink2_vis_f32(vm, l->ln1_b, &b[1], 0, D, host) &&
            mm_vis_bf16(vm, l->q_w, &w[0], 0, 3 * square, stage) &&
            mm_vis_bf16(vm, l->k_w, &w[0], square, 3 * square, stage) &&
            mm_vis_bf16(vm, l->v_w, &w[0], 2 * square, 3 * square, stage) &&
            ink2_vis_f32(vm, l->q_b, &b[2], 0, 3 * D, host) && ink2_vis_f32(vm, l->k_b, &b[2], D, 3 * D, host) &&
            ink2_vis_f32(vm, l->v_b, &b[2], 2 * D, 3 * D, host) &&
            mm_vis_bf16(vm, l->o_w, &w[1], 0, square, stage) && ink2_vis_f32(vm, l->o_b, &b[3], 0, D, host) &&
            ink2_vis_f32(vm, l->ls1, &b[4], 0, D, host) &&
            ink2_vis_f32(vm, l->ln2_w, &b[5], 0, D, host) && ink2_vis_f32(vm, l->ln2_b, &b[6], 0, D, host) &&
            mm_vis_bf16(vm, l->fc1_w, &w[2], 0, l->fc1_w->bytes, stage) &&
            ink2_vis_f32(vm, l->fc1_b, &b[7], 0, F, host) &&
            mm_vis_bf16(vm, l->fc2_w, &w[3], 0, l->fc2_w->bytes, stage) &&
            ink2_vis_f32(vm, l->fc2_b, &b[8], 0, D, host) && ink2_vis_f32(vm, l->ls2, &b[9], 0, D, host);
        // LN1, q, k, v and RoPE by row tiles: the BF16 rows every query reads.
        for (uint32_t r0 = 0; ok && r0 < n; r0 += tile) {
            const uint32_t m = n - r0 < tile ? n - r0 : tile;
            ds4_gpu_tensor *xv = mm_vis_view(x, r0, m, D);
            ok = xv && ds4_gpu_minimax_vit_ln(xn, xv, b[0], b[1], m, D, 1e-5f) &&
                v41_vis_proj(qkv, w[0], b[2], D, 3 * D, xn, m) && ds4_gpu_ink2_vit_rope(qkv16, qkv, rows, cols, r0, m);
            ds4_gpu_tensor_free(xv);
        }
        // Then per span, attention, out projection and layer scale, and the
        // MLP of the same rows: rows are independent once every k and v is in.
        for (uint32_t r0 = 0; ok && r0 < n; r0 += span) {
            const uint32_t m = n - r0 < span ? n - r0 : span;
            ds4_gpu_tensor *xv = mm_vis_view(x, r0, m, D);
            ok = xv && ds4_gpu_ink2_vit_attn(at, qkv16, n, r0, m) && v41_vis_proj(xn, w[1], b[3], D, D, at, m) &&
                ds4_gpu_ink2_vit_scale_add(xv, xn, b[4], m, D) &&
                ds4_gpu_minimax_vit_ln(xn, xv, b[5], b[6], m, D, 1e-5f) &&
                v41_vis_proj(mid, w[2], b[7], D, F, xn, m) && ds4_gpu_v41_gelu_tensor(mid, (uint64_t)m * F) &&
                v41_vis_proj(at, w[3], b[8], F, D, mid, m) && ds4_gpu_ink2_vit_scale_add(xv, at, b[9], m, D);
            ds4_gpu_tensor_free(xv);
        }
    }
    // Only x crosses the tower boundary. Drop attention scratch before the
    // merger; its BF16 linear slot regrows as needed.
    const bool cleared = ds4_gpu_minimax_vit_clear();
    ok = ok && cleared;
    ds4_gpu_tensor *tower[] = {qkv16, xn, qkv, at, mid, w[2], w[3]};
    v41_vis_free_list(tower, sizeof(tower) / sizeof(tower[0]));
    v41_vis_free_list(b, sizeof(b) / sizeof(b[0]));
    w[2] = w[3] = NULL;

    // Merger over each token's four patches (consecutive rows after the CLS),
    // by token groups.
    ds4_gpu_tensor *p1 = ok ? v41_alloc((uint64_t)group * INKLING2_VIT_MERGE * 4, &ok) : NULL;
    ds4_gpu_tensor *p2 = ok ? v41_alloc((uint64_t)group * DS4_INK2_HIDDEN * 4, &ok) : NULL;
    ok = ok && mm_vis_bf16(vm, V->proj_in, &w[0], 0, V->proj_in->bytes, stage) &&
        mm_vis_bf16(vm, V->proj_out, &w[1], 0, V->proj_out->bytes, stage);
    for (uint32_t t0 = 0; ok && t0 < tokens; t0 += group) {
        const uint32_t k = tokens - t0 < group ? tokens - t0 : group;
        ds4_gpu_tensor *merged = mm_vis_view(x, 1 + 4 * (uint64_t)t0, 4 * k, D);
        ok = merged && v41_vis_proj(p1, w[0], NULL, 4 * D, INKLING2_VIT_MERGE, merged, k) &&
            ds4_gpu_v41_gelu_tensor(p1, (uint64_t)k * INKLING2_VIT_MERGE) &&
            v41_vis_proj(p2, w[1], NULL, INKLING2_VIT_MERGE, DS4_INK2_HIDDEN, p1, k) &&
            ds4_gpu_tensor_read(p2, 0, out + (uint64_t)t0 * DS4_INK2_HIDDEN, (uint64_t)k * DS4_INK2_HIDDEN * 4);
        ds4_gpu_tensor_free(merged);
    }
    ds4_gpu_tensor *all[] = {pix, patches, x, p1, p2, w[0], w[1]};
    v41_vis_free_list(all, sizeof(all) / sizeof(all[0]));
    return ok;
}

static bool inkling2_vision_rows(ds4_engine *e, const float *pixels, uint32_t rows, uint32_t cols, float *out) {
    const ds4_inkling2_vision *V = &e->inkling2_vision;
    if (!V->bound || !pixels || !out || !rows || !cols || (uint64_t)rows * cols > INKLING2_IMAGE_TOKENS) {
        fprintf(stderr, "ds4: Inkling-2 vision refuses grid %ux%u\n", rows, cols);
        return false;
    }
    for (uint64_t i = 0; i < (uint64_t)rows * cols * DS4_INK2_IMAGE_VALUES; i++) {
        if (!isfinite(pixels[i])) { return false; }
    }
    void *stage = ds4_gpu_host_alloc(V41_VIS_STAGE_BYTES);
    float *host = malloc(INK2_VIS_HOST * sizeof(float));
    const double t0 = now_sec();
    bool ok = stage && host && ink2_vis_pass(&e->vision_model, V, pixels, rows, cols, out, stage, host);
    // Run even after a failed pass: no request retains grow-only media scratch.
    const bool cleared = ds4_gpu_minimax_vit_clear();
    ok = ok && cleared;
    if (stage) { ds4_gpu_host_free(stage); }
    free(host);
    v41_vis_drop_pages(&e->vision_model);
    fprintf(stderr, "ds4: Inkling-2 vision %ux%u -> %u rows in %.2f s%s\n", rows, cols, rows * cols,
        now_sec() - t0, ok ? "" : " (failed)");
    return ok;
}

int ds4_session_inkling2_encode_vision(ds4_session *s, const float *pixels, unsigned rows, unsigned cols,
        float *out, char *err, size_t errlen) {
    if (!ds4_session_is_inkling2(s) || !s->engine->inkling2_vision.bound) {
        payload_set_err(err, errlen, "Inkling-2 vision part is not open");
        return 1;
    }
    if (!inkling2_vision_rows(s->engine, pixels, rows, cols, out)) {
        payload_set_err(err, errlen, "Inkling-2 vision tower failed");
        return 1;
    }
    return 0;
}

#else
int ds4_session_inkling2_encode_vision(ds4_session *s, const float *pixels, unsigned rows, unsigned cols,
        float *out, char *err, size_t errlen) {
    (void)s; (void)pixels; (void)rows; (void)cols; (void)out;
    payload_set_err(err, errlen, "this build has no CUDA vision support");
    return 1;
}
#endif
