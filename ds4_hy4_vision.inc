// The CLIP projector stays mapped on the host. Stage matrix weights one layer at a time.
#ifndef DS4_NO_GPU
#ifndef __APPLE__
static bool hy4_vis_vec(const ds4_model *m, const ds4_tensor *t, ds4_gpu_tensor **slot, uint32_t start,
                        uint32_t width) {
    bool ok = true;
    return t && t->type == DS4_TENSOR_F32 && t->bytes <= (uint64_t)(width - start) * 4 &&
           v41_vis_grow(slot, (uint64_t)width * 4, &ok) && ok &&
           ds4_gpu_tensor_write(*slot, (uint64_t)start * 4, tensor_data(m, t), t->bytes);
}

static bool hy4_vis_frame(const ds4_model *m, const float *patches, uint32_t gh, uint32_t gw, float *out,
                          void *stage) {
    const uint32_t n = gh * gw, d = 1152, groups = n / 4;
    bool ok = true;
    ds4_gpu_tensor *x = v41_alloc((uint64_t)n * d * 4, &ok), *xn = v41_alloc((uint64_t)n * d * 4, &ok);
    ds4_gpu_tensor *qkv = v41_alloc((uint64_t)n * 3 * d * 4, &ok),
                   *mid = v41_alloc((uint64_t)n * 4304 * 4, &ok);
    ds4_gpu_tensor *w = NULL, *b = NULL, *nb = NULL;
    const ds4_tensor *t = required_tensor(m, "v.patch_embd.weight");
    ok = ok && ds4_gpu_tensor_write(mid, 0, patches, (uint64_t)n * 768 * 4) &&
         mm_vis_bf16(m, t, &w, 0, t->bytes, stage) &&
         hy4_vis_vec(m, required_tensor(m, "v.patch_embd.bias"), &b, 0, d) &&
         mm_vis_linear(x, w, b, 768, d, mid, n);
    t = required_tensor(m, "v.position_embd.weight");
    ok = ok && mm_vis_bf16(m, t, &w, 0, t->bytes, stage) && ds4_gpu_hy4_vision_pos(x, w, gh, gw);
    for (uint32_t il = 0; ok && il < 27; il++) {
#define HY4_VT(name) required_tensorf(m, "v.blk.%u." name, il)
#define HY4_VN(name)                                                                                       \
    (hy4_vis_vec(m, HY4_VT(name ".weight"), &b, 0, d) &&                                                   \
     hy4_vis_vec(m, HY4_VT(name ".bias"), &nb, 0, d) && ds4_gpu_minimax_vit_ln(xn, x, b, nb, n, d, 1e-5f))
        ok = HY4_VN("ln1");
        const char *weights[] = {"v.blk.%u.attn_q.weight", "v.blk.%u.attn_k.weight",
                                 "v.blk.%u.attn_v.weight"};
        const char *biases[] = {"v.blk.%u.attn_q.bias", "v.blk.%u.attn_k.bias", "v.blk.%u.attn_v.bias"};
        for (uint32_t j = 0; ok && j < 3; j++) {
            t = required_tensorf(m, weights[j], il);
            ok = mm_vis_bf16(m, t, &w, j * t->bytes, 3 * t->bytes, stage) &&
                 hy4_vis_vec(m, required_tensorf(m, biases[j], il), &b, j * d, 3 * d);
        }
        ok = ok && mm_vis_linear(qkv, w, b, d, 3 * d, xn, n) &&
             ds4_gpu_v41_dense_attn_tensor(xn, qkv, n, 16, 72);
#define HY4_VLINEAR(name, from, to, dst, src)                                                              \
    (mm_vis_bf16(m, HY4_VT(name ".weight"), &w, 0, HY4_VT(name ".weight")->bytes, stage) &&                \
     hy4_vis_vec(m, HY4_VT(name ".bias"), &b, 0, to) && mm_vis_linear(dst, w, b, from, to, src, n))
        ok = ok && HY4_VLINEAR("attn_out", d, d, mid, xn) &&
             ds4_gpu_v41_add_tensor(x, mid, (uint64_t)n * d) &&
             ds4_gpu_v41_round_bf16_tensor(x, (uint64_t)n * d) && HY4_VN("ln2") &&
             HY4_VLINEAR("ffn_up", d, 4304, mid, xn) && ds4_gpu_v41_gelu_tensor(mid, (uint64_t)n * 4304) &&
             HY4_VLINEAR("ffn_down", 4304, d, xn, mid) && ds4_gpu_v41_add_tensor(x, xn, (uint64_t)n * d) &&
             ds4_gpu_v41_round_bf16_tensor(x, (uint64_t)n * d);
#undef HY4_VLINEAR
#undef HY4_VN
#undef HY4_VT
    }
    ok = ok && hy4_vis_vec(m, required_tensor(m, "mm.pre_norm.weight"), &b, 0, d) &&
         ds4_gpu_v41_rms_norm_dev_tensor(xn, x, b, d, n, 1e-5f) &&
         ds4_gpu_v41_vision_unfold_tensor(x, xn, gh, gw, d, 2);
    t = required_tensor(m, "mm.0.weight");
    ok = ok && mm_vis_bf16(m, t, &w, 0, t->bytes, stage) &&
         hy4_vis_vec(m, required_tensor(m, "mm.0.bias"), &b, 0, 4608) &&
         mm_vis_linear(xn, w, b, 4608, 4608, x, groups) &&
         ds4_gpu_v41_gelu_tensor(xn, (uint64_t)groups * 4608);
    t = required_tensor(m, "mm.2.weight");
    ok = ok && mm_vis_bf16(m, t, &w, 0, t->bytes, stage) &&
         hy4_vis_vec(m, required_tensor(m, "mm.2.bias"), &b, 0, 3072) &&
         mm_vis_linear(x, w, b, 4608, 3072, xn, groups);
    const uint16_t *newline = tensor_data(m, required_tensor(m, "v.image_newline"));
    for (uint32_t y = 0; ok && y < gh / 2; y++) {
        const uint32_t width = gw / 2;
        float *row = out + (uint64_t)y * (width + 1) * 3072;
        ok = ds4_gpu_tensor_read(x, (uint64_t)y * width * 3072 * 4, row, (uint64_t)width * 3072 * 4);
        for (uint32_t j = 0; j < 3072; j++) {
            uint32_t bits = (uint32_t)newline[j] << 16;
            memcpy(row + (uint64_t)width * 3072 + j, &bits, 4);
        }
    }
    ds4_gpu_tensor *all[] = {x, xn, qkv, mid, w, b, nb};
    v41_vis_free_list(all, sizeof(all) / sizeof(*all));
    return ok;
}
#endif
#endif

int ds4_session_hy4_encode_vision(ds4_session *s, const float *pixels, uint32_t gh, uint32_t gw,
                                  float *rows, char *err, size_t len) {
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (ds4_session_is_hy4(s) && s->engine->vision_model.map && pixels && rows && gh && gw &&
        !(gh % 2 || gw % 2) && (uint64_t)gh * gw <= 16384) {
        void *stage = ds4_gpu_host_alloc(V41_VIS_STAGE_BYTES);
        bool ok = stage && hy4_vis_frame(&s->engine->vision_model, pixels, gh, gw, rows, stage);
        if (stage) {
            ds4_gpu_host_free(stage);
        }
        ok = ds4_gpu_minimax_vit_clear() && ok;
        v41_vis_drop_pages(&s->engine->vision_model);
        if (ok) {
            return 0;
        }
    }
#else
    (void)s;
    (void)pixels;
    (void)gh;
    (void)gw;
    (void)rows;
#endif
    payload_set_err(err, len, "Hy4 visual encoder or frame geometry unavailable");
    return 1;
}
