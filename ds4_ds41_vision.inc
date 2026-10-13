/* ds4_ds41_vision.inc — DeepSeek V4.1 vision tower host forward (2026-10-13).
 * 1:1 port of the engine's src/core/core_v41_vision.c (the official
 * inference/vision.py ViT + Aligner, transcribed expression by expression).
 *
 * ★Weights are not resident★: the --vision-gguf sidecar is mmap'd only; when an
 * image arrives each block's weights are copied off the mapping into a
 * temporary device tensor (cold page cache = a real SSD read), and after the
 * request the sidecar's page cache is dropped again.  Text requests never
 * touch it.  Discipline: all GPU; activations f32 with values on the bf16 grid
 * at the official module boundaries (v41_vis_proj's bias add / round,
 * v41_vis_residual's round).  Primitives live in cuda/ds41_vision.cuh.
 * Gate: the engine's golden aligner rows compared per-row by cosine
 * (core_v41_vision_gate.c: thresholds VIS_GATE_MEAN_COS 0.999 / MIN 0.95). */

#ifndef DS4_NO_GPU

#define V41_VIS_RMS_EPS 1e-6f   /* official vision.py RMSNorm(dim, eps=1e-6); not in config.json, so pinned at the consumer */

/* Per-layer image-row router bias (blk.N.exp_probs_b_vl) -> device tensors, one
 * upload of 40 x 384 f32 (~61 KB) and then resident (core_v41_vision.c:132-146). */
static ds4_gpu_tensor *g_v41_bias_vl[DS4_MAX_LAYER];

/* One image's tower activations: residual stream x[n][D], aligner output al[n_llm][E]. */
typedef struct { ds4_gpu_tensor *x, *al; uint32_t n, nh, nw, n_llm; } v41_vact;
/* One image as the tower consumes it (preprocessing output). */
typedef struct { const float *patches; uint32_t n_vit_h, n_vit_w, n_llm_h, n_llm_w; } v41_vis_image;

/* The file that holds the tower weights: the --vision-gguf sidecar.  vis_layers
 * is set only by v41_load_vision_metadata on that sidecar, so >0 means it is
 * open (core_v41_vision.c:29-30). */
static const ds4_model *v41_vision_model(const ds4_engine *e) {
    return g_ds4_v41.vis_layers ? &e->vision_model : &e->model;
}

/* Copy a sidecar tensor's bytes into a temporary device tensor, reusing the
 * slot when it is already big enough (the 32 blocks share shapes)
 * (core_v41_vision.c:21-27). */
static bool v41_vis_stage(const ds4_model *vm, const ds4_tensor *t, ds4_gpu_tensor **slot) {
    if (!t) return false;
    if (*slot && ds4_gpu_tensor_bytes(*slot) < t->bytes) { ds4_gpu_tensor_free(*slot); *slot = NULL; }
    bool ok = true;
    if (!*slot) *slot = v41_alloc(t->bytes, &ok);
    return ok && *slot && ds4_gpu_tensor_write(*slot, 0, tensor_data(vm, t), t->bytes) != 0;
}
static ds4_gpu_tensor *v41_vis_grow(ds4_gpu_tensor **t, uint64_t bytes, bool *ok) {
    if (*t && ds4_gpu_tensor_bytes(*t) >= bytes) return *t;
    if (*t) ds4_gpu_tensor_free(*t);
    *t = v41_alloc(bytes, ok);
    return *t;
}
static void v41_vis_free_list(ds4_gpu_tensor **l, size_t n) {
    for (size_t i = 0; i < n; i++) { if (l[i]) { ds4_gpu_tensor_free(l[i]); l[i] = NULL; } }
}

/* Drop the sidecar's page cache after a request so the next image reads from
 * SSD.  ONLY the sidecar: the main mapping's expert streaming relies on its
 * cache (core_v41_vision.c:37-42). */
static void v41_vis_drop_pages(const ds4_model *vm) {
    if (!vm || !vm->map || !vm->size) return;
#ifdef __linux__
    if (vm->fd >= 0) (void)posix_fadvise(vm->fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    (void)madvise((void *)vm->map, (size_t)vm->size, MADV_DONTNEED);
}

/* The official Linear: bf16 GEMM, then bias add rounded to bf16 (or a plain
 * round when there is no bias) (core_v41_vision.c:45-49). */
static bool v41_vis_proj(ds4_gpu_tensor *out, const ds4_gpu_tensor *W, const ds4_gpu_tensor *b,
                         uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n) {
    if (!ds4_gpu_v41_matmul_bf16_dev_tensor(out, W, in_dim, out_dim, x, n)) return false;
    if (b) return ds4_gpu_v41_vision_bias_add_tensor(out, b, n, (uint32_t)out_dim) != 0;
    return ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)n * out_dim) != 0;
}
/* The official x + attn(...) of two bf16 tensors, out bf16 (core_v41_vision.c:50-52). */
static bool v41_vis_residual(ds4_gpu_tensor *x, const ds4_gpu_tensor *y, uint64_t n) {
    return ds4_gpu_v41_add_tensor(x, y, n) && ds4_gpu_v41_round_bf16_tensor(x, n);
}

/* Run the whole tower for a batch of images: block-major (weights read once)
 * with images inner (core_v41_vision.c:57-114).  A[i].al is the aligner output
 * [n_llm][E]; on failure every A[i].al is freed and NULL. */
static bool v41_vis_run_images(ds4_engine *e, const v41_vis_image *im, uint32_t n_img, v41_vact *A) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const ds4_v41_vision_weights *V = &e->weights.v41_vision;
    const ds4_model *vm = v41_vision_model(e);
    const uint32_t D = v->vis_dim, H = v->vis_heads, I = v->vis_inter, R = v->vis_ds, PP = 3u * v->vis_patch * v->vis_patch;
    const uint64_t E = DS4_N_EMBD;
    uint32_t nmax = 0, lmax = 0;
    for (uint32_t i = 0; i < n_img; i++) {
        A[i].n = im[i].n_vit_h * im[i].n_vit_w; A[i].nh = im[i].n_vit_h; A[i].nw = im[i].n_vit_w;
        A[i].n_llm = im[i].n_llm_h * im[i].n_llm_w;
        if (A[i].n > nmax) nmax = A[i].n;
        if (A[i].n_llm > lmax) lmax = A[i].n_llm;
    }
    bool ok = true;
    ds4_gpu_tensor *pat = NULL, *xn = NULL, *qkv = NULL, *o = NULL, *gu = NULL, *hm = NULL, *unf = NULL, *a1 = NULL;
    ds4_gpu_tensor *w[10] = {0};   /* scratch slots: 0 n1 1 wqkv 2 bqkv 3 wo 4 bo 5 n2 6 w1 7 w2 8 patch_w 9 patch_b */
    v41_vis_grow(&pat, (uint64_t)nmax * PP * 4, &ok); v41_vis_grow(&xn, (uint64_t)nmax * D * 4, &ok);
    v41_vis_grow(&qkv, (uint64_t)nmax * 3 * D * 4, &ok); v41_vis_grow(&o, (uint64_t)nmax * D * 4, &ok);
    v41_vis_grow(&gu, (uint64_t)nmax * 2 * I * 4, &ok); v41_vis_grow(&hm, (uint64_t)nmax * I * 4, &ok);
    v41_vis_grow(&unf, (uint64_t)lmax * D * R * R * 4, &ok); v41_vis_grow(&a1, (uint64_t)lmax * E * 4, &ok);
    for (uint32_t i = 0; ok && i < n_img; i++) {
        A[i].x = v41_alloc((uint64_t)A[i].n * D * 4, &ok);
        A[i].al = v41_alloc((uint64_t)A[i].n_llm * E * 4, &ok);
    }
    const double t0 = now_sec();
    /* patch_embed */
    if (ok) ok = v41_vis_stage(vm, V->patch_w, &w[8]) && v41_vis_stage(vm, V->patch_b, &w[9]);
    for (uint32_t i = 0; ok && i < n_img; i++)
        ok = ds4_gpu_tensor_write(pat, 0, im[i].patches, (uint64_t)A[i].n * PP * 4) != 0 &&
             v41_vis_proj(A[i].x, w[8], w[9], PP, D, pat, A[i].n);
    /* vis_layers blocks: per block the weights are read once, then each image */
    for (uint32_t il = 0; ok && il < v->vis_layers; il++) {
        const ds4_v41_vision_block_weights *b = &V->blk[il];
        ok = v41_vis_stage(vm, b->norm1, &w[0]) && v41_vis_stage(vm, b->qkv_w, &w[1]) && v41_vis_stage(vm, b->qkv_b, &w[2]) &&
             v41_vis_stage(vm, b->o_w, &w[3]) && v41_vis_stage(vm, b->o_b, &w[4]) && v41_vis_stage(vm, b->norm2, &w[5]) &&
             v41_vis_stage(vm, b->up_w, &w[6]) && v41_vis_stage(vm, b->down_w, &w[7]);
        for (uint32_t i = 0; ok && i < n_img; i++) {
            const uint32_t n = A[i].n; ds4_gpu_tensor *x = A[i].x;
            ok = ds4_gpu_v41_rms_norm_dev_tensor(xn, x, w[0], D, n, V41_VIS_RMS_EPS) &&
                 v41_vis_proj(qkv, w[1], w[2], D, 3u * D, xn, n) &&
                 ds4_gpu_v41_vision_rope_tensor(qkv, n, A[i].nw, H, D / H, v->vis_theta) &&
                 ds4_gpu_v41_dense_attn_tensor(o, qkv, n, H, D / H) &&
                 v41_vis_proj(xn, w[3], w[4], D, D, o, n) &&
                 v41_vis_residual(x, xn, (uint64_t)n * D) &&
                 ds4_gpu_v41_rms_norm_dev_tensor(xn, x, w[5], D, n, V41_VIS_RMS_EPS) &&
                 v41_vis_proj(gu, w[6], NULL, D, 2u * I, xn, n) &&
                 ds4_gpu_v41_swiglu_pair_tensor(hm, gu, n, I) &&
                 v41_vis_proj(xn, w[7], NULL, I, D, hm, n) &&
                 v41_vis_residual(x, xn, (uint64_t)n * D);
        }
    }
    /* norm -> aligner (r x r pixel re-arrange -> Linear -> GELU -> Linear) */
    if (ok) ok = v41_vis_stage(vm, V->out_norm, &w[0]) && v41_vis_stage(vm, V->mm0_w, &w[6]) && v41_vis_stage(vm, V->mm0_b, &w[2]) &&
                 v41_vis_stage(vm, V->mm1_w, &w[7]) && v41_vis_stage(vm, V->mm1_b, &w[4]);
    for (uint32_t i = 0; ok && i < n_img; i++)
        ok = ds4_gpu_v41_rms_norm_dev_tensor(xn, A[i].x, w[0], D, A[i].n, V41_VIS_RMS_EPS) &&
             ds4_gpu_v41_vision_unfold_tensor(unf, xn, A[i].nh, A[i].nw, D, R) &&
             v41_vis_proj(a1, w[6], w[2], (uint64_t)D * R * R, E, unf, A[i].n_llm) &&
             ds4_gpu_v41_gelu_tensor(a1, (uint64_t)A[i].n_llm * E) &&
             v41_vis_proj(A[i].al, w[7], w[4], E, E, a1, A[i].n_llm);
    if (ok && !ds4_gpu_synchronize()) ok = false;
    ds4_gpu_tensor *scr[] = { pat, xn, qkv, o, gu, hm, unf, a1 };
    v41_vis_free_list(scr, sizeof scr / sizeof scr[0]); v41_vis_free_list(w, sizeof w / sizeof w[0]);
    for (uint32_t i = 0; i < n_img; i++) { if (A[i].x) { ds4_gpu_tensor_free(A[i].x); } A[i].x = NULL; }
    v41_vis_drop_pages(vm);
    if (!ok) {
        fprintf(stderr, "ds4: [v41] vision tower forward failed (%u image(s))\n", n_img);
        for (uint32_t i = 0; i < n_img; i++) { if (A[i].al) { ds4_gpu_tensor_free(A[i].al); A[i].al = NULL; } }
    } else {
        fprintf(stderr, "[v41] vision tower: %u image(s) %.2fs (weights read per block from the sidecar, freed after)\n",
                n_img, now_sec() - t0);
    }
    return ok;
}

/* One image through the tower -> malloc'd aligner rows [n_llm][E], row-major
 * (core_v41_vision.c:116-131).  Entry for the golden gate and the caller that
 * lays out the span. */
float *v41_vision_run(ds4_engine *e, const float *patches, uint32_t n_vit_h, uint32_t n_vit_w,
                      uint32_t *n_llm_h, uint32_t *n_llm_w) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    if (!v->vis_layers) { fprintf(stderr, "ds4: this GGUF has no vision tower\n"); return NULL; }
    v41_vis_image im; memset(&im, 0, sizeof im);
    im.n_vit_h = n_vit_h; im.n_vit_w = n_vit_w;
    im.n_llm_h = (n_vit_h + v->vis_ds - 1) / v->vis_ds; im.n_llm_w = (n_vit_w + v->vis_ds - 1) / v->vis_ds;
    im.patches = patches;
    v41_vact A; memset(&A, 0, sizeof A);
    if (!v41_vis_run_images(e, &im, 1, &A)) return NULL;
    const uint64_t E = DS4_N_EMBD;
    float *out = xmalloc((size_t)A.n_llm * E * 4);
    if (!ds4_gpu_tensor_read(A.al, 0, out, (uint64_t)A.n_llm * E * 4)) { free(out); out = NULL; }
    ds4_gpu_tensor_free(A.al);
    if (out) { *n_llm_h = im.n_llm_h; *n_llm_w = im.n_llm_w; }
    return out;
}

/* Per-layer bias_vl -> device tensors, once (core_v41_vision.c:132-146).  A
 * failure encodes nothing (never substitutes the text bias). */
static bool v41_vis_bias_vl_upload(ds4_engine *e) {
    const ds4_model *m = v41_vision_model(e);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_tensor *t = e->weights.layer[il].ffn_exp_probs_b_vl;
        if (!t) { fprintf(stderr, "ds4: [v41] layer %u has no bias_vl tensor\n", il); return false; }
        if (g_v41_bias_vl[il]) continue;
        bool ok = true;
        ds4_gpu_tensor *d = v41_alloc(t->bytes, &ok);
        if (!ok || !ds4_gpu_tensor_write(d, 0, tensor_data(m, t), t->bytes)) { if (d) ds4_gpu_tensor_free(d); return false; }
        g_v41_bias_vl[il] = d;
    }
    return true;
}

#endif /* !DS4_NO_GPU */
