/* K-EXAONE CUDA kernel checks against the CPU reference.
 *
 *   ./tests/test_exaone_kernels [model.gguf]
 *
 * Two tiers. The first needs no model: attention, QK-norm + NeoX rotary, the
 * KV ring, the router, SwiGLU and the routed-slot combine all run on
 * synthetic data against a CPU implementation written here from the same
 * rules as the reference forward in ds4.c.
 *
 * The second tier needs the artifact, because the only honest test of the
 * quantized routed-expert matmul is real expert blocks: it compares the
 * mmq path against ds4's own CPU dequant-and-dot for one expert of each
 * quant type the recipe emits (IQ2_XXS, Q3_K, Q4_K, and Q2_K for the pilot).
 *
 * Tolerances are stated per check and are not "the sentences look similar".
 * The matmul tier carries the loosest one because mmq quantizes activations
 * to Q8_1 while the CPU reference keeps them f32 -- that difference is
 * expected, bounded, and the reason the number is 2e-2 relative and not 1e-5.
 */
#include "../ds4.c"

static int g_fail = 0;

static void test_context_memory_plan(void) {
    g_ds4_shape = DS4_SHAPE_KEXAONE_236B;
    const ds4_context_memory m =
        ds4_context_memory_estimate(DS4_BACKEND_CUDA, 262144);
    ds4_engine engine = {0};
    const int ok =
        ds4_engine_hidden_f32_values(&engine) == 6144u &&
        ds4_engine_n_hc(&engine) == 1 &&
        m.prefill_cap == 512u &&
        m.raw_cap == 262144u &&
        exaone_graph_layer_kv_cap(0u, 262144u, m.prefill_cap) == 640u &&
        exaone_graph_layer_kv_cap(3u, 262144u, m.prefill_cap) == 262144u &&
        m.raw_bytes == UINT64_C(12979273728) &&
        m.compressed_bytes == 0u &&
        m.scratch_bytes == UINT64_C(429564480) &&
        m.total_bytes == UINT64_C(13408838208);
    if (!ok) g_fail++;
    printf("%-38s KV=%.2f GiB scratch=%.2f GiB total=%.2f GiB  %s\n",
           "262144-context memory plan",
           (double)m.raw_bytes / 1073741824.0,
           (double)m.scratch_bytes / 1073741824.0,
           (double)m.total_bytes / 1073741824.0,
           ok ? "ok" : "FAIL");
}

static void test_k2_full_attention_memory_plan(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_K2_HORIZON_375B;
    const ds4_context_memory m =
        exaone_graph_context_memory_estimate(32768u, 512u);
    const int ok =
        m.total_bytes != 0u &&
        m.prefill_cap == 512u &&
        m.raw_cap == 32768u &&
        m.raw_bytes == UINT64_C(8187281408) &&
        !exaone_layer_is_sliding(0u) &&
        !exaone_layer_is_sliding(60u) &&
        exaone_graph_layer_kv_cap(0u, 32768u, m.prefill_cap) == 32768u &&
        exaone_graph_layer_kv_cap(60u, 32768u, m.prefill_cap) == 32768u;
    if (!ok) g_fail++;
    printf("%-38s KV=%.2f GiB scratch=%.2f GiB total=%.2f GiB  %s\n",
           "K2 full-attention memory plan",
           (double)m.raw_bytes / 1073741824.0,
           (double)m.scratch_bytes / 1073741824.0,
           (double)m.total_bytes / 1073741824.0,
           ok ? "ok" : "FAIL");
    g_ds4_shape = saved;
}

static void test_batch_memory_plan(void) {
    uint64_t shared = 0u, per_bank = 0u, total1 = 0u, total2 = 0u;
    const ds4_context_memory one =
        exaone_graph_context_memory_estimate(262144u, 512u);
    const bool estimated1 = exaone_graph_batch_memory_estimate(
        262144u, 512u, 1u, &shared, &per_bank, &total1);
    const bool estimated2 = exaone_graph_batch_memory_estimate(
        262144u, 512u, 2u, NULL, NULL, &total2);
    const uint64_t logits_row =
        (uint64_t)DS4_N_VOCAB * sizeof(float);
    const int ok = estimated1 && estimated2 && shared != 0u &&
                   per_bank != 0u && shared + per_bank == one.total_bytes &&
                   total1 == one.total_bytes + logits_row &&
                   total2 == total1 + per_bank + logits_row;
    if (!ok) g_fail++;
    printf("%-38s shared=%.2f GiB bank=%.2f GiB  %s\n",
           "persistent-bank memory plan",
           (double)shared / 1073741824.0,
           (double)per_bank / 1073741824.0,
           ok ? "ok" : "FAIL");
}

static void test_family_session_graph_memory_plan(void) {
    const ds4_shape saved = g_ds4_shape;
    const struct {
        ds4_shape shape;
        int ctx;
    } cases[] = {
        {DS4_SHAPE_SOLAR_OPEN2_250B, 196608},
        {DS4_SHAPE_KEXAONE_236B, 262144},
        {DS4_SHAPE_MOTIF3, 262144},
        {DS4_SHAPE_DOTS3_NOTE_PREV, 524288},
    };
    ds4_engine engine = {.backend = DS4_BACKEND_CUDA};

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        g_ds4_shape = cases[i].shape;
        const uint64_t bytes =
            ds4_engine_session_graph_bytes_estimate(&engine, cases[i].ctx);
        const int ok = bytes != 0u;
        if (!ok) g_fail++;
        printf("%-38s %-22s %.2f GiB  %s\n",
               "family session graph plan", DS4_MODEL_SHAPE_NAME,
               (double)bytes / 1073741824.0, ok ? "ok" : "FAIL");
    }
    g_ds4_shape = saved;
}

static void test_exaone_rewind_span(void) {
    ds4_engine engine = {0};
    ds4_session session = {0};
    session.engine = &engine;
    session.exaone_graph_ready = true;
    session.exaone_graph.ctx_size = 262144u;
    session.exaone_graph.layer_kv[0] =
        (ds4_gpu_tensor *)(uintptr_t)1u;
    session.exaone_graph.layer_kv_cap[0] = 640u;
    session.checkpoint_valid = true;

    session.checkpoint.len = 500;
    const int short_span = ds4_session_exaone_rewind_span(&session);
    session.checkpoint.len = 1000;
    const int wrapped_span = ds4_session_exaone_rewind_span(&session);
    const int ok = short_span == 500 && wrapped_span == 513;
    if (!ok) g_fail++;
    printf("%-38s short=%d wrapped=%d  %s\n",
           "diverged-prefix rewind span", short_span, wrapped_span,
           ok ? "ok" : "FAIL");
}

static void report(const char *what, double max_abs, double max_rel, double tol_abs,
                   double tol_rel) {
    const int ok = (max_abs <= tol_abs) || (max_rel <= tol_rel);
    if (!ok) g_fail++;
    printf("%-38s max_abs=%.3e max_rel=%.3e  (tol_abs=%.0e tol_rel=%.0e)  %s\n",
           what, max_abs, max_rel, tol_abs, tol_rel, ok ? "ok" : "FAIL");
}

static void diff_f32(const char *what, const float *a, const float *b, size_t n,
                     double tol_abs, double tol_rel) {
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        const double m = fmax(fabs((double)a[i]), fabs((double)b[i]));
        if (d > max_abs) max_abs = d;
        if (m > 1e-6 && d / m > max_rel) max_rel = d / m;
    }
    report(what, max_abs, max_rel, tol_abs, tol_rel);
}

/* Vector-level agreement for a quantized-activation GEMM. Per-element
 * max-relative error is dominated by outputs near zero and does not reliably
 * distinguish quantization noise from a transposed or misindexed expert. */
static void diff_quant_gemm(const char *what, const float *got, const float *ref,
                            size_t n, double tol_rel_rms) {
    double se = 0.0, sr = 0.0, sg = 0.0, dot = 0.0;
    double max_abs = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double g = got[i], r = ref[i], d = g - r;
        se += d * d;
        sr += r * r;
        sg += g * g;
        dot += g * r;
        if (fabs(d) > max_abs) max_abs = fabs(d);
        if (fabs(r) > max_ref) max_ref = fabs(r);
    }
    const double rel_rms = sr > 0.0 ? sqrt(se / sr)
                                    : (se > 0.0 ? INFINITY : 0.0);
    const double cosine = (sg > 0.0 && sr > 0.0)
        ? dot / sqrt(sg * sr) : 0.0;
    const int ok = rel_rms <= tol_rel_rms &&
                   (1.0 - cosine) <= tol_rel_rms;
    if (!ok) g_fail++;
    printf("%-38s rel_rms=%.3e 1-cos=%.3e max_abs=%.2e "
           "(|ref|max=%.2e)  %s\n",
           what, rel_rms, 1.0 - cosine, max_abs, max_ref,
           ok ? "ok" : "FAIL");
}

static float frand(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((int32_t)(*s >> 33) % 20000 - 10000) / 10000.0f;
}

/* ---- CPU mirrors of the kernels ---------------------------------------- */

static void test_k2_partial_rope_layout(void) {
    float v[128] = {0};
    v[0] = 1.0f;
    v[32] = 3.0f;
    v[64] = 2.0f;
    v[96] = 4.0f;
    exaone_rope_neox(v, 1u, 128u, 64u, 1u, 10000000.0f);
    const float c = cosf(1.0f), s = sinf(1.0f);
    const int ok = fabsf(v[0] - (c - 2.0f * s)) < 1e-6f &&
                   fabsf(v[64] - (s + 2.0f * c)) < 1e-6f &&
                   v[32] == 3.0f && v[96] == 4.0f;
    if (!ok) g_fail++;
    printf("%-38s pair=0/64 nope=32/96  %s\n",
           "K2 partial RoPE layout", ok ? "ok" : "FAIL");
}

static void cpu_qk_norm_rope(float *v, const float *w, uint32_t n_heads,
                             uint32_t head_dim, uint32_t n_rot, uint32_t pos,
                             float freq_base, int do_rope) {
    for (uint32_t h = 0; h < n_heads; h++) {
        float *p = v + (size_t)h * head_dim;
        rms_norm_weight(p, p, w, head_dim, DS4_DEFAULT_RMS_EPS);
        if (!do_rope) continue;
        const uint32_t half = n_rot / 2;
        const uint32_t pair_stride = head_dim / 2;
        for (uint32_t i = 0; i < half; i++) {
            /* Same double-precision frequency and reduction as the kernel and
             * the reference; see exaone_rope_neox in ds4.c for why. */
            const float freq = (float)pow((double)freq_base,
                                          -2.0 * (double)i / (double)n_rot);
            const float theta = (float)pos * freq;
            const double tr = fmod((double)theta, 2.0 * M_PI);
            const float c = (float)cos(tr), s = (float)sin(tr);
            const float a = p[i], b = p[i + pair_stride];
            p[i]        = a * c - b * s;
            p[i + pair_stride] = a * s + b * c;
        }
    }
}

/* Attention over an f16 KV ring, exactly as exaone_attn_one does it. */
static void cpu_attn(float *out, const float *q, const uint16_t *kv,
                     uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
                     uint32_t kv_cap, uint32_t first, uint32_t last) {
    const uint32_t kv_dim = n_head_kv * head_dim;
    const uint32_t group  = n_head / n_head_kv;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *score = xmalloc((size_t)(last - first + 1) * sizeof(float));
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (size_t)h * head_dim;
        const uint32_t kvh = h / group;
        float maxv = -INFINITY;
        for (uint32_t t = first; t <= last; t++) {
            const uint16_t *kh = kv + (size_t)(t % kv_cap) * kv_dim * 2u +
                                 (size_t)kvh * head_dim;
            float s = 0.0f;
            for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * f16_to_f32(kh[i]);
            s *= scale;
            score[t - first] = s;
            if (s > maxv) maxv = s;
        }
        float sum = 0.0f;
        for (uint32_t t = first; t <= last; t++) {
            score[t - first] = expf(score[t - first] - maxv);
            sum += score[t - first];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        float *oh = out + (size_t)h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));
        for (uint32_t t = first; t <= last; t++) {
            const float w = score[t - first] * inv;
            const uint16_t *vh = kv + (size_t)(t % kv_cap) * kv_dim * 2u +
                                 kv_dim + (size_t)kvh * head_dim;
            for (uint32_t i = 0; i < head_dim; i++) oh[i] += w * f16_to_f32(vh[i]);
        }
    }
    free(score);
}

static void cpu_router(uint32_t *sel, float *w, const float *logits,
                       const float *bias, uint32_t n_expert, uint32_t n_used,
                       float scale) {
    float *probs = xmalloc((size_t)n_expert * sizeof(float));
    for (uint32_t e = 0; e < n_expert; e++) probs[e] = sigmoid_stable(logits[e]);
    bool *taken = xcalloc(n_expert, sizeof(bool));
    for (uint32_t i = 0; i < n_used; i++) {
        int best = -1; float best_v = -INFINITY;
        for (uint32_t e = 0; e < n_expert; e++) {
            if (taken[e]) continue;
            const float v = bias ? probs[e] + bias[e] : probs[e];
            if (v > best_v) { best_v = v; best = (int)e; }
        }
        sel[i] = (uint32_t)best;
        taken[best] = true;
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_used; i++) { w[i] = probs[sel[i]]; sum += w[i]; }
    const float denom = sum > 6.103515625e-5f ? sum : 6.103515625e-5f;
    for (uint32_t i = 0; i < n_used; i++) w[i] = (w[i] / denom) * scale;
    free(taken); free(probs);
}

/* ---- tier 1: no model -------------------------------------------------- */

enum {
    T_HEAD     = 64,
    T_HEAD_KV  = 8,
    T_HEAD_DIM = 128,
    T_EXPERT   = 128,
    T_USED     = 8,
};

static void test_shared_prefill_workspace(void) {
    const uint32_t P = 2u;
    const uint64_t n_embd = 256u;
    const uint64_t q_dim = 512u;
    const uint64_t kv_dim = 128u;
    const uint64_t n_used = DS4_N_EXPERT_USED;
    const uint64_t n_ff_shared =
        DS4_N_FF_SHEXP ? DS4_N_FF_SHEXP : DS4_N_FF_EXP;
    const uint64_t elems_per_token =
        5u * n_embd + 2u * q_dim + 2u * kv_dim +
        3u * DS4_N_FF_DENSE + 3u * n_ff_shared +
        DS4_N_EXPERT + 2u * n_used +
        3u * n_used * DS4_N_FF_EXP + n_used * n_embd;
    const uint64_t expected_bytes =
        (uint64_t)P * elems_per_token * sizeof(float);

    ds4_exaone_batch_ws shared = {0};
    bool ok = exaone_batch_ws_alloc(&shared, P, n_embd, q_dim, kv_dim) &&
              shared.prefill_cap == P && shared.bytes == expected_bytes &&
              shared.b_cur && shared.b_routed_down;
    ds4_gpu_tensor *borrowed_cur = shared.b_cur;

    ds4_model model = {0};
    ds4_weights weights = {0};
    ds4_exaone_gpu_graph graph = {0};
    if (ok) {
        ok = exaone_graph_alloc_with_ws(&graph, &model, &weights,
                                         8u, P, &shared) &&
             graph.ws == &shared && !graph.ws_owned;
    }
    exaone_graph_free(&graph);
    ok = ok && shared.b_cur == borrowed_cur && shared.bytes == expected_bytes;
    exaone_batch_ws_free(&shared);
    ok = ok && !shared.b_cur && shared.bytes == 0u;

    if (!ok) g_fail++;
    printf("%-38s bytes=%.2f MiB  %s\n", "shared prefill workspace ownership",
           (double)expected_bytes / 1048576.0, ok ? "ok" : "FAIL");
}

static void test_persistent_bank_runtime_ownership(void) {
    ds4_engine engine = {0};
    ds4_exaone_batch_runtime *rt =
        exaone_batch_runtime_create(&engine, 8u, 3u, 2u);
    bool ok = rt && rt->max_seq == 3u &&
              rt->graph[0].ws == &rt->shared_ws &&
              rt->graph[1].ws == &rt->shared_ws &&
              rt->graph[2].ws == &rt->shared_ws &&
              !rt->graph[0].ws_owned && !rt->graph[1].ws_owned;
    if (ok) {
        float *src = rt->bank_logits;
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) src[i] = (float)i;
        rt->bank_logits_valid[0] = 1u;
        ok = exaone_batch_runtime_copy_bank(rt, 0u, 1u, 0u) &&
             rt->bank_logits_valid[1] &&
             rt->bank_logits[DS4_N_VOCAB + 17u] == 17.0f &&
             !exaone_batch_runtime_copy_bank(rt, 3u, 3u, 0u);
    }
    exaone_batch_runtime_free(rt);
    if (!ok) g_fail++;
    printf("%-38s banks=3/prefill=2  %s\n", "persistent bank shared ownership",
           ok ? "ok" : "FAIL");
}

static void test_qk_norm_rope(void *map, uint64_t map_size, uint64_t w_off,
                              uint32_t pos0) {
    const uint32_t n_tok = 3;
    const size_t n = (size_t)T_HEAD * T_HEAD_DIM * n_tok;
    float *host = xmalloc(n * sizeof(float));
    float *want = xmalloc(n * sizeof(float));
    uint64_t s = 12345;
    for (size_t i = 0; i < n; i++) host[i] = frand(&s);
    memcpy(want, host, n * sizeof(float));

    for (int rope = 0; rope <= 1; rope++) {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
        ds4_gpu_tensor_write(t, 0, host, n * sizeof(float));
        if (!ds4_gpu_exaone_qk_norm_rope_tensor(t, map, map_size, w_off, T_HEAD,
                                                T_HEAD_DIM, T_HEAD_DIM, pos0,
                                                n_tok, 1000000.0f,
                                                DS4_DEFAULT_RMS_EPS, rope)) {
            printf("qk_norm_rope launch failed\n"); g_fail++; return;
        }
        float *got = xmalloc(n * sizeof(float));
        ds4_gpu_tensor_read(t, 0, got, n * sizeof(float));
        memcpy(want, host, n * sizeof(float));
        for (uint32_t tk = 0; tk < n_tok; tk++)
            cpu_qk_norm_rope(want + (size_t)tk * T_HEAD * T_HEAD_DIM,
                             (const float *)((char *)map + w_off), T_HEAD,
                             T_HEAD_DIM, T_HEAD_DIM, pos0 + tk, 1000000.0f, rope);
        char label[64];
        snprintf(label, sizeof(label), rope ? "qk_norm + neox rope @pos %u"
                                            : "qk_norm only @pos %u", pos0);
        diff_f32(label, got, want, n, 3e-5, 3e-5);
        free(got);
        ds4_gpu_tensor_free(t);
    }
    free(want); free(host);
}

static void test_neox_rope_only(uint32_t pos0) {
    const uint32_t n_tok = 3u;
    const uint32_t n_rot = T_HEAD_DIM / 2u;
    const size_t n = (size_t)T_HEAD * T_HEAD_DIM * n_tok;
    float *host = xmalloc(n * sizeof(float));
    float *want = xmalloc(n * sizeof(float));
    uint64_t s = 98765u;
    for (size_t i = 0; i < n; i++) host[i] = frand(&s);
    memcpy(want, host, n * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++) {
        exaone_rope_neox(want + (size_t)t * T_HEAD * T_HEAD_DIM,
                         T_HEAD, T_HEAD_DIM, n_rot, pos0 + t, 10000000.0f);
    }

    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor_write(v, 0, host, n * sizeof(float));
    if (!ds4_gpu_exaone_rope_tensor(v, T_HEAD, T_HEAD_DIM, n_rot,
                                    pos0, n_tok, 10000000.0f)) {
        printf("neox-only rope launch failed\n");
        g_fail++;
    } else {
        float *got = xmalloc(n * sizeof(float));
        ds4_gpu_tensor_read(v, 0, got, n * sizeof(float));
        char label[64];
        snprintf(label, sizeof(label), "neox-only partial rope @pos %u", pos0);
        diff_f32(label, got, want, n, 3e-5, 3e-5);
        free(got);
    }
    ds4_gpu_tensor_free(v);
    free(want);
    free(host);
}

static void test_attention(int sliding) {
    const uint32_t window = sliding ? 128u : 0u;
    const uint32_t pos    = sliding ? 300u : 300u;
    const uint32_t kv_cap = sliding ? 128u : 512u;
    const uint32_t kv_dim = T_HEAD_KV * T_HEAD_DIM;
    const uint32_t first  = (window && pos + 1u > window) ? pos + 1u - window : 0u;

    uint64_t s = 999;
    float *q = xmalloc((size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
    for (size_t i = 0; i < (size_t)T_HEAD * T_HEAD_DIM; i++) q[i] = frand(&s);

    const size_t kv_elems = (size_t)kv_cap * kv_dim * 2u;
    uint16_t *kv = xmalloc(kv_elems * sizeof(uint16_t));
    for (size_t i = 0; i < kv_elems; i++) kv[i] = f32_to_f16(frand(&s) * 0.5f);

    ds4_gpu_tensor *gq  = ds4_gpu_tensor_alloc((size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *gkv = ds4_gpu_tensor_alloc(kv_elems * sizeof(uint16_t));
    ds4_gpu_tensor *go  = ds4_gpu_tensor_alloc((size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor_write(gq, 0, q, (size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor_write(gkv, 0, kv, kv_elems * sizeof(uint16_t));

    if (!ds4_gpu_exaone_attention_decode_tensor(go, gq, gkv, T_HEAD, T_HEAD_KV,
                                                T_HEAD_DIM, kv_cap, pos, window)) {
        printf("attention decode launch failed\n"); g_fail++; return;
    }
    float *got  = xmalloc((size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
    float *want = xmalloc((size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor_read(go, 0, got, (size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
    cpu_attn(want, q, kv, T_HEAD, T_HEAD_KV, T_HEAD_DIM, kv_cap, first, pos);
    diff_f32(sliding ? "attention decode (window 128)"
                     : "attention decode (full, NoPE)",
             got, want, (size_t)T_HEAD * T_HEAD_DIM, 2e-5, 2e-5);

    (void)setenv("DS4_EXAONE_ATTN_GQA", "0", 1);
    if (!ds4_gpu_exaone_attention_decode_tensor(go, gq, gkv, T_HEAD, T_HEAD_KV,
                                                T_HEAD_DIM, kv_cap, pos, window)) {
        printf("attention decode serial launch failed\n"); g_fail++;
    } else {
        ds4_gpu_tensor_read(go, 0, want, (size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
        (void)setenv("DS4_EXAONE_ATTN_GQA", "1", 1);
        if (!ds4_gpu_exaone_attention_decode_tensor(go, gq, gkv, T_HEAD, T_HEAD_KV,
                                                    T_HEAD_DIM, kv_cap, pos, window)) {
            printf("attention decode GQA launch failed\n"); g_fail++;
        } else {
            ds4_gpu_tensor_read(go, 0, got, (size_t)T_HEAD * T_HEAD_DIM * sizeof(float));
            diff_f32(sliding ? "attention GQA vs serial (window 128)"
                             : "attention GQA vs serial (full, NoPE)",
                     got, want, (size_t)T_HEAD * T_HEAD_DIM, 1e-6, 1e-6);
        }
    }
    (void)unsetenv("DS4_EXAONE_ATTN_GQA");

    free(want); free(got);
    ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gkv); ds4_gpu_tensor_free(gq);
    free(kv); free(q);
}

/* Decode 2: split-K decode attention vs the pair kernel on the K2 shape
 * (48 heads over 8 KV heads, head_dim 128). Same per-key math, different
 * fp32 merge order, so a tolerance; below the split floor, with a sliding
 * window and under the kill switch the pair kernel must run bit-identically. */
static void test_decode_split(void) {
    const uint32_t nh = 48u, nkv = 8u, hd = 128u, cap = 32768u;
    static const struct { uint32_t pos, window; const char *env; int exact; } cases[] = {
        {1023u, 0u, "1", 1},    /* below the split floor */
        {4095u, 0u, "1", 0},
        {8191u, 0u, "1", 0},
        {32767u, 0u, "1", 0},
        {8191u, 128u, "1", 1},  /* sliding window */
        {8191u, 0u, "0", 1},    /* kill switch */
    };
    const size_t n = (size_t)nh * hd, kvn = (size_t)cap * nkv * hd * 2u;
    float *q = xmalloc(n * sizeof(float));
    float *ref = xmalloc(n * sizeof(float));
    float *got = xmalloc(n * sizeof(float));
    float *want = xmalloc(n * sizeof(float));
    uint16_t *kv = xmalloc(kvn * sizeof(uint16_t));
    uint64_t seed = 1717u;
    for (size_t i = 0; i < n; i++) { q[i] = frand(&seed); }
    for (size_t i = 0; i < kvn; i++) { kv[i] = f32_to_f16(frand(&seed) * 0.5f); }
    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *go = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *gkv = ds4_gpu_tensor_alloc(kvn * sizeof(uint16_t));
    int ok = gq && go && gkv &&
        ds4_gpu_tensor_write(gq, 0, q, n * sizeof(float)) &&
        ds4_gpu_tensor_write(gkv, 0, kv, kvn * sizeof(uint16_t));
    for (size_t c = 0; ok && c < sizeof(cases) / sizeof(cases[0]); c++) {
        const uint32_t pos = cases[c].pos, window = cases[c].window;
        double ms[2] = {0.0, 0.0};
        for (int split = 0; ok && split < 2; split++) {
            (void)setenv("DS4_EXAONE_ATTN_SPLIT", split ? cases[c].env : "0", 1);
            ok = ds4_gpu_exaone_attention_decode_tensor(
                     go, gq, gkv, nh, nkv, hd, cap, pos, window) &&
                 ds4_gpu_synchronize();
            const double start = now_sec();
            for (int i = 0; ok && i < 20; i++) {
                ok = ds4_gpu_exaone_attention_decode_tensor(
                         go, gq, gkv, nh, nkv, hd, cap, pos, window);
            }
            ok = ok && ds4_gpu_synchronize();
            ms[split] = (now_sec() - start) * 50.0;
            ok = ok && ds4_gpu_tensor_read(go, 0, split ? got : ref, n * sizeof(float));
        }
        (void)unsetenv("DS4_EXAONE_ATTN_SPLIT");
        if (!ok) break;
        char label[80];
        snprintf(label, sizeof(label), "K2 split attention @%u window=%u env=%s",
                 pos, window, cases[c].env);
        diff_quant_gemm(label, got, ref, n, 1e-5);
        if (cases[c].exact && memcmp(got, ref, n * sizeof(float)) != 0) {
            printf("%s: pair-kernel fallback is not bit-identical\n", label); g_fail++;
        }
        printf("%s pair=%.3f ms split=%.3f ms\n", label, ms[0], ms[1]);
        if (pos == 8191u && window == 0u && cases[c].env[0] == '1') {
            cpu_attn(want, q, kv, nh, nkv, hd, cap, 0u, pos);
            diff_f32("K2 split attention @8191 vs CPU", got, want, n, 2e-5, 2e-5);
        }
    }
    if (!ok) { printf("K2 split attention launch failed\n"); g_fail++; }
    ds4_gpu_tensor_free(gkv); ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gq);
    free(kv); free(want); free(got); free(ref); free(q);
}

static void test_attention_prefill(int sliding) {
    const uint32_t n_tok  = 200;
    const uint32_t window = sliding ? 128u : 0u;
    const uint32_t kv_cap = sliding ? 128u : 256u;
    const uint32_t kv_dim = T_HEAD_KV * T_HEAD_DIM;

    uint64_t s = 4242;
    const size_t qn = (size_t)n_tok * T_HEAD * T_HEAD_DIM;
    float *q = xmalloc(qn * sizeof(float));
    for (size_t i = 0; i < qn; i++) q[i] = frand(&s);

    /* Fill the ring the way prefill does, through the store kernel, so the
     * ring index and the f16 rounding are under test too. */
    const size_t kvn = (size_t)n_tok * kv_dim;
    float *k = xmalloc(kvn * sizeof(float));
    float *v = xmalloc(kvn * sizeof(float));
    for (size_t i = 0; i < kvn; i++) { k[i] = frand(&s) * 0.5f; v[i] = frand(&s) * 0.5f; }

    ds4_gpu_tensor *gk  = ds4_gpu_tensor_alloc(kvn * sizeof(float));
    ds4_gpu_tensor *gv  = ds4_gpu_tensor_alloc(kvn * sizeof(float));
    ds4_gpu_tensor *gkv = ds4_gpu_tensor_alloc((size_t)kv_cap * kv_dim * 2u * sizeof(uint16_t));
    ds4_gpu_tensor_write(gk, 0, k, kvn * sizeof(float));
    ds4_gpu_tensor_write(gv, 0, v, kvn * sizeof(float));
    ds4_gpu_tensor_fill_f32(gkv, 0.0f, (size_t)kv_cap * kv_dim);
    if (!ds4_gpu_exaone_kv_store_tensor(gkv, gk, gv, kv_dim, n_tok, 0, kv_cap)) {
        printf("kv store launch failed\n"); g_fail++; return;
    }

    uint16_t *kv = xmalloc((size_t)kv_cap * kv_dim * 2u * sizeof(uint16_t));
    ds4_gpu_tensor_read(gkv, 0, kv, (size_t)kv_cap * kv_dim * 2u * sizeof(uint16_t));
    /* CPU mirror of the ring write, to catch a slot-index mistake directly
     * instead of only through attention. */
    {
        uint16_t *want_kv = xcalloc((size_t)kv_cap * kv_dim * 2u, sizeof(uint16_t));
        for (uint32_t t = 0; t < n_tok; t++) {
            uint16_t *row = want_kv + (size_t)(t % kv_cap) * kv_dim * 2u;
            for (uint32_t i = 0; i < kv_dim; i++) {
                row[i]          = f32_to_f16(k[(size_t)t * kv_dim + i]);
                row[kv_dim + i] = f32_to_f16(v[(size_t)t * kv_dim + i]);
            }
        }
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)kv_cap * kv_dim * 2u; i++)
            if (kv[i] != want_kv[i]) bad++;
        printf("%-38s mismatched halfs=%zu  %s\n",
               sliding ? "kv ring store (wrap)" : "kv ring store (linear)",
               bad, bad ? "FAIL" : "ok");
        if (bad) g_fail++;
        free(want_kv);
    }

    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(qn * sizeof(float));
    ds4_gpu_tensor *go = ds4_gpu_tensor_alloc(qn * sizeof(float));
    ds4_gpu_tensor_write(gq, 0, q, qn * sizeof(float));
    /* CPU oracle is the f32 warp kernel. HMMA is fp16 MMA; compare it to
     * that warp output separately, not to the CPU reference. */
    (void)setenv("DS4_EXAONE_PREFILL_HMMA", "0", 1);
    if (!ds4_gpu_exaone_attention_prefill_tensor(go, gq, gkv, n_tok, 0, T_HEAD,
                                                 T_HEAD_KV, T_HEAD_DIM,
                                                 kv_cap, window)) {
        printf("attention prefill launch failed\n"); g_fail++; return;
    }
    float *got  = xmalloc(qn * sizeof(float));
    float *want = xmalloc(qn * sizeof(float));
    ds4_gpu_tensor_read(go, 0, got, qn * sizeof(float));
    /* Only the last kv_cap positions survive in a wrapped ring, so compare
     * the rows whose whole window is still resident -- which is every row
     * for the full-attention case and the tail for the sliding one. */
    const uint32_t t0 = sliding ? (n_tok > kv_cap ? n_tok - kv_cap : 0u) : 0u;
    for (uint32_t t = t0; t < n_tok; t++) {
        const uint32_t first = (window && t + 1u > window) ? t + 1u - window : 0u;
        cpu_attn(want + (size_t)t * T_HEAD * T_HEAD_DIM,
                 q + (size_t)t * T_HEAD * T_HEAD_DIM,
                 kv, T_HEAD, T_HEAD_KV, T_HEAD_DIM, kv_cap, first, t);
    }
    diff_f32(sliding ? "attention prefill (window 128)"
                     : "attention prefill (full, NoPE)",
             got + (size_t)t0 * T_HEAD * T_HEAD_DIM,
             want + (size_t)t0 * T_HEAD * T_HEAD_DIM,
             (size_t)(n_tok - t0) * T_HEAD * T_HEAD_DIM, 2e-5, 2e-5);

    if (!sliding) {
        (void)unsetenv("DS4_EXAONE_PREFILL_HMMA");
        if (!ds4_gpu_exaone_attention_prefill_tensor(
                go, gq, gkv, n_tok, 0, T_HEAD, T_HEAD_KV, T_HEAD_DIM,
                kv_cap, window)) {
            printf("attention prefill HMMA launch failed\n");
            g_fail++;
        } else {
            float *hmma = xmalloc(qn * sizeof(float));
            ds4_gpu_tensor_read(go, 0, hmma, qn * sizeof(float));
            /* fp16 MMA vs f32 warp: bound is RMS, not per-element rel.
             * A layout/mask bug is rel_rms ~ 1, not 1e-3. */
            diff_quant_gemm("attention prefill HMMA vs warp",
                            hmma, got, qn, 1e-3);
            free(hmma);
        }
    } else {
        (void)unsetenv("DS4_EXAONE_PREFILL_HMMA");
    }

    free(want); free(got);
    ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gq);
    ds4_gpu_tensor_free(gkv); ds4_gpu_tensor_free(gv); ds4_gpu_tensor_free(gk);
    free(v); free(k); free(q); free(kv);
}

/* A chunk stores all of its K/V rows before attention consumes any row.  A
 * sliding ring must therefore hold the logical window plus the whole chunk.
 * Compare that path with the one-token-at-a-time oracle, which never loses a
 * still-visible window. */
static uint32_t prefill_vs_incremental_bad_rows(uint32_t kv_cap,
                                                uint32_t n_tok,
                                                uint32_t window) {
    const uint32_t kv_dim = T_HEAD_KV * T_HEAD_DIM;
    const size_t row = (size_t)T_HEAD * T_HEAD_DIM;
    const size_t qn = (size_t)n_tok * row;
    const size_t kvn = (size_t)n_tok * kv_dim;

    uint64_t s = 90210;
    float *q = xmalloc(qn * sizeof(float));
    float *k = xmalloc(kvn * sizeof(float));
    float *v = xmalloc(kvn * sizeof(float));
    for (size_t i = 0; i < qn; i++) q[i] = frand(&s);
    for (size_t i = 0; i < kvn; i++) {
        k[i] = frand(&s) * 0.5f;
        v[i] = frand(&s) * 0.5f;
    }

    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(qn * sizeof(float));
    ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(kvn * sizeof(float));
    ds4_gpu_tensor *gv = ds4_gpu_tensor_alloc(kvn * sizeof(float));
    ds4_gpu_tensor_write(gq, 0, q, qn * sizeof(float));
    ds4_gpu_tensor_write(gk, 0, k, kvn * sizeof(float));
    ds4_gpu_tensor_write(gv, 0, v, kvn * sizeof(float));

    const size_t ring_floats = (size_t)kv_cap * kv_dim;
    ds4_gpu_tensor *ring_chunk =
        ds4_gpu_tensor_alloc(ring_floats * 2u * sizeof(uint16_t));
    ds4_gpu_tensor *ring_incr =
        ds4_gpu_tensor_alloc(ring_floats * 2u * sizeof(uint16_t));
    ds4_gpu_tensor *out_chunk = ds4_gpu_tensor_alloc(qn * sizeof(float));
    ds4_gpu_tensor *out_incr = ds4_gpu_tensor_alloc(qn * sizeof(float));
    ds4_gpu_tensor_fill_f32(ring_chunk, 0.0f, ring_floats);
    ds4_gpu_tensor_fill_f32(ring_incr, 0.0f, ring_floats);

    uint32_t bad = UINT32_MAX;
    if (!ds4_gpu_exaone_kv_store_tensor(
            ring_chunk, gk, gv, kv_dim, n_tok, 0, kv_cap) ||
        !ds4_gpu_exaone_attention_prefill_tensor(
            out_chunk, gq, ring_chunk, n_tok, 0,
            T_HEAD, T_HEAD_KV, T_HEAD_DIM, kv_cap, window)) {
        printf("chunked prefill launch failed\n");
        goto done;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        ds4_gpu_tensor *kr = ds4_gpu_tensor_view(
            gk, (uint64_t)t * kv_dim * sizeof(float),
            kv_dim * sizeof(float));
        ds4_gpu_tensor *vr = ds4_gpu_tensor_view(
            gv, (uint64_t)t * kv_dim * sizeof(float),
            kv_dim * sizeof(float));
        ds4_gpu_tensor *qr = ds4_gpu_tensor_view(
            gq, (uint64_t)t * row * sizeof(float), row * sizeof(float));
        ds4_gpu_tensor *orow = ds4_gpu_tensor_view(
            out_incr, (uint64_t)t * row * sizeof(float),
            row * sizeof(float));
        const int ok = kr && vr && qr && orow &&
            ds4_gpu_exaone_kv_store_tensor(
                ring_incr, kr, vr, kv_dim, 1u, t, kv_cap) &&
            ds4_gpu_exaone_attention_decode_tensor(
                orow, qr, ring_incr, T_HEAD, T_HEAD_KV, T_HEAD_DIM,
                kv_cap, t, window);
        ds4_gpu_tensor_free(kr);
        ds4_gpu_tensor_free(vr);
        ds4_gpu_tensor_free(qr);
        ds4_gpu_tensor_free(orow);
        if (!ok) {
            printf("incremental prefill launch failed at %u\n", t);
            goto done;
        }
    }

    {
        float *a = xmalloc(qn * sizeof(float));
        float *b = xmalloc(qn * sizeof(float));
        ds4_gpu_tensor_read(out_chunk, 0, a, qn * sizeof(float));
        ds4_gpu_tensor_read(out_incr, 0, b, qn * sizeof(float));
        bad = 0u;
        for (uint32_t t = 0; t < n_tok; t++) {
            double worst = 0.0;
            for (size_t i = 0; i < row; i++) {
                const double d =
                    fabs((double)a[(size_t)t * row + i] -
                         (double)b[(size_t)t * row + i]);
                if (d > worst) worst = d;
            }
            if (worst > 2e-5) bad++;
        }
        free(a);
        free(b);
    }

done:
    ds4_gpu_tensor_free(out_incr);
    ds4_gpu_tensor_free(out_chunk);
    ds4_gpu_tensor_free(ring_incr);
    ds4_gpu_tensor_free(ring_chunk);
    ds4_gpu_tensor_free(gv);
    ds4_gpu_tensor_free(gk);
    ds4_gpu_tensor_free(gq);
    free(v);
    free(k);
    free(q);
    return bad;
}

static uint32_t expected_bad_rows(uint32_t cap, uint32_t n_tok,
                                  uint32_t window) {
    const uint32_t resident_from = n_tok > cap ? n_tok - cap : 0u;
    if (resident_from == 0u) return 0u;
    const uint32_t first_good = resident_from + window - 1u;
    return first_good < n_tok ? first_good : n_tok;
}

static void test_prefill_chunk_residency(void) {
    const uint32_t n_tok = 200u;
    const uint32_t window = 128u;
    const uint32_t tight =
        prefill_vs_incremental_bad_rows(window, n_tok, window);
    const uint32_t sized =
        prefill_vs_incremental_bad_rows(window + n_tok, n_tok, window);

    printf("%-38s bad_rows=%u expected=%u  %s\n",
           "sliding ring == window (undersized)", tight,
           expected_bad_rows(window, n_tok, window),
           tight == expected_bad_rows(window, n_tok, window) ? "ok" : "FAIL");
    if (tight != expected_bad_rows(window, n_tok, window)) g_fail++;

    printf("%-38s bad_rows=%u  %s\n",
           "sliding ring == window + chunk", sized,
           sized == 0u ? "ok" : "FAIL");
    if (sized != 0u) g_fail++;
}

static void test_router(void *map, uint64_t map_size, uint64_t bias_off) {
    const uint32_t n_tok = 4;
    uint64_t s = 7;
    float *logits = xmalloc((size_t)T_EXPERT * n_tok * sizeof(float));
    for (size_t i = 0; i < (size_t)T_EXPERT * n_tok; i++) logits[i] = frand(&s) * 4.0f;

    ds4_gpu_tensor *gl = ds4_gpu_tensor_alloc((size_t)T_EXPERT * n_tok * sizeof(float));
    ds4_gpu_tensor *gs = ds4_gpu_tensor_alloc((size_t)T_USED * n_tok * sizeof(int32_t));
    ds4_gpu_tensor *gw = ds4_gpu_tensor_alloc((size_t)T_USED * n_tok * sizeof(float));
    ds4_gpu_tensor_write(gl, 0, logits, (size_t)T_EXPERT * n_tok * sizeof(float));
    if (!ds4_gpu_exaone_router_tensor(gs, gw, gl, map, map_size, bias_off, 1,
                                      T_EXPERT, T_USED, n_tok, 2.5f)) {
        printf("router launch failed\n"); g_fail++; return;
    }
    int32_t *sel = xmalloc((size_t)T_USED * n_tok * sizeof(int32_t));
    float   *w   = xmalloc((size_t)T_USED * n_tok * sizeof(float));
    ds4_gpu_tensor_read(gs, 0, sel, (size_t)T_USED * n_tok * sizeof(int32_t));
    ds4_gpu_tensor_read(gw, 0, w,   (size_t)T_USED * n_tok * sizeof(float));

    uint32_t *wsel = xmalloc((size_t)T_USED * n_tok * sizeof(uint32_t));
    float    *ww   = xmalloc((size_t)T_USED * n_tok * sizeof(float));
    const float *bias = (const float *)((char *)map + bias_off);
    size_t bad_sel = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        cpu_router(wsel + t * T_USED, ww + t * T_USED, logits + (size_t)t * T_EXPERT,
                   bias, T_EXPERT, T_USED, 2.5f);
        for (uint32_t i = 0; i < T_USED; i++)
            if ((uint32_t)sel[t * T_USED + i] != wsel[t * T_USED + i]) bad_sel++;
    }
    printf("%-38s mismatched slots=%zu  %s\n", "router top-8 selection (biased)",
           bad_sel, bad_sel ? "FAIL" : "ok");
    if (bad_sel) g_fail++;
    diff_f32("router weights (unbiased, scaled)", w, ww,
             (size_t)T_USED * n_tok, 1e-6, 1e-6);

    free(ww); free(wsel); free(w); free(sel);
    ds4_gpu_tensor_free(gw); ds4_gpu_tensor_free(gs); ds4_gpu_tensor_free(gl);
    free(logits);
}

static void test_swiglu_and_combine(void) {
    const uint32_t n_tok = 3, n_ff = 2048, n_embd = 512;
    const size_t mid_n = (size_t)n_ff * T_USED * n_tok;
    uint64_t s = 31337;
    float *gate = xmalloc(mid_n * sizeof(float));
    float *up   = xmalloc(mid_n * sizeof(float));
    for (size_t i = 0; i < mid_n; i++) { gate[i] = frand(&s) * 3.0f; up[i] = frand(&s) * 3.0f; }

    ds4_gpu_tensor *gg = ds4_gpu_tensor_alloc(mid_n * sizeof(float));
    ds4_gpu_tensor *gu = ds4_gpu_tensor_alloc(mid_n * sizeof(float));
    ds4_gpu_tensor *gm = ds4_gpu_tensor_alloc(mid_n * sizeof(float));
    ds4_gpu_tensor_write(gg, 0, gate, mid_n * sizeof(float));
    ds4_gpu_tensor_write(gu, 0, up,   mid_n * sizeof(float));
    if (!ds4_gpu_exaone_swiglu_tensor(gm, gg, gu, mid_n)) {
        printf("swiglu launch failed\n"); g_fail++; return;
    }
    float *got = xmalloc(mid_n * sizeof(float));
    float *want = xmalloc(mid_n * sizeof(float));
    ds4_gpu_tensor_read(gm, 0, got, mid_n * sizeof(float));
    for (size_t i = 0; i < mid_n; i++) want[i] = silu(gate[i]) * up[i];
    diff_f32("swiglu", got, want, mid_n, 1e-5, 1e-5);
    free(want); free(got);
    ds4_gpu_tensor_free(gm); ds4_gpu_tensor_free(gu); ds4_gpu_tensor_free(gg);
    free(up); free(gate);

    /* combine: column-major down[col*n_embd + d], col = token*n_used + slot */
    const size_t down_n = (size_t)n_embd * T_USED * n_tok;
    float *down = xmalloc(down_n * sizeof(float));
    float *w    = xmalloc((size_t)T_USED * n_tok * sizeof(float));
    float *shr  = xmalloc((size_t)n_embd * n_tok * sizeof(float));
    for (size_t i = 0; i < down_n; i++) down[i] = frand(&s);
    for (size_t i = 0; i < (size_t)T_USED * n_tok; i++) w[i] = fabsf(frand(&s));
    for (size_t i = 0; i < (size_t)n_embd * n_tok; i++) shr[i] = frand(&s);

    ds4_gpu_tensor *gd = ds4_gpu_tensor_alloc(down_n * sizeof(float));
    ds4_gpu_tensor *gwt = ds4_gpu_tensor_alloc((size_t)T_USED * n_tok * sizeof(float));
    ds4_gpu_tensor *gsh = ds4_gpu_tensor_alloc((size_t)n_embd * n_tok * sizeof(float));
    ds4_gpu_tensor *go  = ds4_gpu_tensor_alloc((size_t)n_embd * n_tok * sizeof(float));
    ds4_gpu_tensor_write(gd, 0, down, down_n * sizeof(float));
    ds4_gpu_tensor_write(gwt, 0, w, (size_t)T_USED * n_tok * sizeof(float));
    ds4_gpu_tensor_write(gsh, 0, shr, (size_t)n_embd * n_tok * sizeof(float));
    if (!ds4_gpu_exaone_moe_combine_tensor(go, gd, gwt, gsh, n_embd, T_USED, n_tok)) {
        printf("moe combine launch failed\n"); g_fail++; return;
    }
    float *cgot  = xmalloc((size_t)n_embd * n_tok * sizeof(float));
    float *cwant = xmalloc((size_t)n_embd * n_tok * sizeof(float));
    ds4_gpu_tensor_read(go, 0, cgot, (size_t)n_embd * n_tok * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = 0.0f;
            for (uint32_t sl = 0; sl < T_USED; sl++) {
                const size_t col = (size_t)t * T_USED + sl;
                acc += w[col] * down[col * n_embd + d];
            }
            cwant[(size_t)t * n_embd + d] = acc + shr[(size_t)t * n_embd + d];
        }
    diff_f32("routed combine + shared expert", cgot, cwant,
             (size_t)n_embd * n_tok, 1e-4, 1e-5);
    free(cwant); free(cgot);
    ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gsh);
    ds4_gpu_tensor_free(gwt); ds4_gpu_tensor_free(gd);
    free(shr); free(w); free(down);
}

/* ---- tier 2: real expert blocks from the artifact ---------------------- */

static void test_moe_worklist(const ds4_model *m, const ds4_tensor *w,
                              uint32_t used) {
    const uint32_t k = (uint32_t)w->dim[0];
    const uint32_t widths[] = {17u, 257u, used == 1u ? 4096u : 512u, 257u};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        const uint32_t nt = widths[wi];
        /* Reinterpret the leading real blocks as a ragged 129-row stack
         * to exercise the checked tile; both schedules see the same bytes. */
        const uint32_t rows = wi == 3u ? 129u : (uint32_t)w->dim[1];
        const size_t nx = (size_t)nt * k, ny = (size_t)nt * used * rows;
        float *x = xmalloc(nx * sizeof(float));
        float *ref = xmalloc(ny * sizeof(float));
        float *got = xmalloc(ny * sizeof(float));
        int32_t *ids = xmalloc((size_t)nt * used * sizeof(int32_t));
        uint64_t seed = 1729u;
        for (size_t i = 0; i < nx; i++) { x[i] = frand(&seed) * 0.5f; }
        for (uint32_t t = 0; t < nt; t++) {
            for (uint32_t s = 0; s < used; s++) {
                ids[(size_t)t * used + s] =
                    ((t % 5u == 0u ? 7u : t * 13u) + s * 7u) % DS4_N_EXPERT;
            }
        }
        ids[used] = -1; /* Both schedules must preserve the invalid-slot zero. */
        ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc(nx * sizeof(float));
        ds4_gpu_tensor *gi = ds4_gpu_tensor_alloc((size_t)nt * used * sizeof(int32_t));
        ds4_gpu_tensor *go = ds4_gpu_tensor_alloc(ny * sizeof(float));
        int ok = gx && gi && go &&
            ds4_gpu_tensor_write(gx, 0, x, nx * sizeof(float)) &&
            ds4_gpu_tensor_write(gi, 0, ids, (size_t)nt * used * sizeof(int32_t));
        double ms[2] = {0.0, 0.0};
        for (int fast = 0; ok && fast < 2; fast++) {
            (void)setenv("DS4_MMQ_WORKLIST", fast ? "1" : "0", 1);
            ok = ds4_gpu_tensor_fill_f32(go, 0.0f, ny) &&
                ds4_gpu_routed_matmul_tensor(go, gx, gi, m->map, m->size,
                    w->abs_offset, w->bytes, w->type, k, rows, DS4_N_EXPERT, nt, used) &&
                ds4_gpu_synchronize();
            const double start = now_sec();
            for (int i = 0; ok && i < 4; i++) {
                ok = ds4_gpu_routed_matmul_tensor(go, gx, gi, m->map, m->size,
                    w->abs_offset, w->bytes, w->type, k, rows, DS4_N_EXPERT, nt, used);
            }
            ok = ok && ds4_gpu_synchronize();
            ms[fast] = (now_sec() - start) * 250.0;
            ok = ok && ds4_gpu_tensor_read(go, 0, fast ? got : ref, ny * sizeof(float));
        }
        (void)unsetenv("DS4_MMQ_WORKLIST");
        if (ok) {
            char label[80];
            snprintf(label, sizeof(label), "%s worklist nt=%u used=%u rows=%u",
                     tensor_type_name(w->type), nt, used, rows);
            diff_quant_gemm(label, got, ref, ny, 1e-5);
            printf("%s old=%.3f ms compact=%.3f ms\n", label, ms[0], ms[1]);
        } else {
            printf("worklist type=%u nt=%u launch failed\n", w->type, nt); g_fail++;
        }
        ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gi); ds4_gpu_tensor_free(gx);
        free(ids); free(got); free(ref); free(x);
    }
}

/* Assign-major IQ1_M prefill vs the per-token MMVQ kill switch. */
static int run_iq1m_prefill_pair(const void *map, uint64_t map_size,
                                 uint64_t w_off, uint64_t w_bytes,
                                 uint32_t k, uint32_t rows, uint32_t nexp,
                                 uint32_t nt, uint32_t used) {
    const size_t nx = (size_t)nt * k, ny = (size_t)nt * used * rows;
    float *x = xmalloc(nx * sizeof(float));
    float *ref = xmalloc(ny * sizeof(float));
    float *got = xmalloc(ny * sizeof(float));
    int32_t *ids = xmalloc((size_t)nt * used * sizeof(int32_t));
    uint64_t seed = 20260905ull + nt;
    for (size_t i = 0; i < nx; i++) { x[i] = frand(&seed) * 0.5f; }
    for (uint32_t t = 0; t < nt; t++) {
        for (uint32_t s = 0; s < used; s++) {
            ids[(size_t)t * used + s] =
                ((t % 5u == 0u ? 7u : t * 13u) + s * 7u) % nexp;
        }
    }
    ids[used] = -1;
    ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc(nx * sizeof(float));
    ds4_gpu_tensor *gi = ds4_gpu_tensor_alloc((size_t)nt * used * sizeof(int32_t));
    ds4_gpu_tensor *go = ds4_gpu_tensor_alloc(ny * sizeof(float));
    int ok = gx && gi && go &&
        ds4_gpu_tensor_write(gx, 0, x, nx * sizeof(float)) &&
        ds4_gpu_tensor_write(gi, 0, ids, (size_t)nt * used * sizeof(int32_t));
    double ms[2] = {0.0, 0.0};
    /* Pin the assign-major tier: from 256 routed rows the default would
     * take the MMQ worklist tile, which run_iq1m_tile_pair covers. */
    (void)setenv("DS4_MMQ_IQ1M_WORKLIST", "0", 1);
    for (int fast = 0; ok && fast < 2; fast++) {
        (void)setenv("DS4_MMQ_IQ1M_PREFILL", fast ? "1" : "0", 1);
        ok = ds4_gpu_tensor_fill_f32(go, 0.0f, ny) &&
            ds4_gpu_routed_matmul_tensor(go, gx, gi, map, map_size,
                w_off, w_bytes, DS4_TENSOR_IQ1_M, k, rows, nexp, nt, used) &&
            ds4_gpu_synchronize();
        const double start = now_sec();
        for (int i = 0; ok && i < 2; i++) {
            ok = ds4_gpu_routed_matmul_tensor(go, gx, gi, map, map_size,
                w_off, w_bytes, DS4_TENSOR_IQ1_M, k, rows, nexp, nt, used);
        }
        ok = ok && ds4_gpu_synchronize();
        ms[fast] = (now_sec() - start) * 500.0;
        ok = ok && ds4_gpu_tensor_read(go, 0, fast ? got : ref, ny * sizeof(float));
    }
    (void)unsetenv("DS4_MMQ_IQ1M_PREFILL");
    if (ok) {
        char label[80];
        snprintf(label, sizeof(label), "IQ1_M assign nt=%u used=%u rows=%u",
                 nt, used, rows);
        diff_quant_gemm(label, got, ref, ny, 1e-5);
        printf("%s token-loop=%.3f ms assign=%.3f ms\n", label, ms[0], ms[1]);
        (void)setenv("DS4_MMQ_IQ1M_PREFILL", "1", 1);
        double slot_ms[2] = {0.0, 0.0};
        for (int loop = 0; ok && loop < 2; loop++) {
            (void)setenv("DS4_MMQ_IQ1M_SLOT_LOOP", loop ? "1" : "0", 1);
            ok = ds4_gpu_tensor_fill_f32(go, 0.0f, ny) &&
                ds4_gpu_routed_matmul_tensor(go, gx, gi, map, map_size,
                    w_off, w_bytes, DS4_TENSOR_IQ1_M, k, rows, nexp, nt, used) &&
                ds4_gpu_synchronize();
            const double start = now_sec();
            for (int i = 0; ok && i < 2; i++) {
                ok = ds4_gpu_routed_matmul_tensor(go, gx, gi, map, map_size,
                    w_off, w_bytes, DS4_TENSOR_IQ1_M, k, rows, nexp, nt, used);
            }
            ok = ok && ds4_gpu_synchronize();
            slot_ms[loop] = (now_sec() - start) * 500.0;
            ok = ok && ds4_gpu_tensor_read(go, 0, loop ? got : ref, ny * sizeof(float));
        }
        (void)unsetenv("DS4_MMQ_IQ1M_SLOT_LOOP");
        (void)unsetenv("DS4_MMQ_IQ1M_PREFILL");
        (void)unsetenv("DS4_MMQ_IQ1M_WORKLIST");
        if (ok) {
            snprintf(label, sizeof(label), "IQ1_M slot-loop nt=%u used=%u rows=%u",
                     nt, used, rows);
            diff_quant_gemm(label, got, ref, ny, 1e-5);
            printf("%s grid3d=%.3f ms slot-loop=%.3f ms\n",
                   label, slot_ms[0], slot_ms[1]);
        } else {
            printf("IQ1_M slot-loop nt=%u launch failed\n", nt); g_fail++;
        }
    } else {
        (void)unsetenv("DS4_MMQ_IQ1M_WORKLIST");
        printf("IQ1_M assign nt=%u launch failed\n", nt); g_fail++;
    }
    ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gi); ds4_gpu_tensor_free(gx);
    free(ids); free(got); free(ref); free(x);
    return ok;
}

static void test_iq1m_prefill_synth(void) {
    const uint32_t k = 256u, rows = 32u, nexp = 16u, used = 8u;
    const uint64_t w_bytes = (uint64_t)nexp * rows * (k / 256u) * 56u;
    const uint64_t w_off = 256;
    const uint64_t map_size = w_off + w_bytes + 256;
    void *map = NULL;
    if (posix_memalign(&map, 4096, (size_t)map_size) != 0) {
        printf("IQ1_M synth map alloc failed\n"); g_fail++; return;
    }
    memset(map, 0, (size_t)map_size);
    uint64_t seed = 20260905ull;
    unsigned char *w = (unsigned char *)map + w_off;
    for (uint64_t i = 0; i < w_bytes; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        w[i] = (unsigned char)(seed >> 33);
    }
    if (!ds4_gpu_set_model_map(map, map_size)) {
        printf("IQ1_M synth map failed\n"); g_fail++; free(map); return;
    }
    /* 8192×8 is the production assignment count (65536 > 65535). */
    const uint32_t widths[] = {17u, 257u, 8192u};
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        (void)run_iq1m_prefill_pair(map, map_size, w_off, w_bytes,
                                    k, rows, nexp, widths[i], used);
    }
    /* The map is host-registered (pinned) with CUDA; freeing it would let
     * a later malloc straddle the pinned range and fail cudaMemcpy with
     * "invalid argument". Keep it until exit. */
}

/* IQ1_M compact-worklist tile vs the assign-major MMVQ kill switch. Both
 * quantize the activation to Q8_1 blocks of 32, but the MMQ producer rounds
 * x*(127/amax) where MMVQ rounds x/(amax/127), so random rows agree to a
 * tolerance, not bit identity. Rows whose 32-value blocks sit on exact
 * Q8_1 points (m*amax/127 with one |m| = 127 per block, amax/127 = 2^-8)
 * make both producers emit the same q and scale, leaving only fp32
 * accumulation order; that
 * mode separates producer rounding from a tile fault (a wrong grid, delta
 * or scale lands near 1e-1). */
enum iq1m_tile_x { IQ1M_TILE_X_RANDOM, IQ1M_TILE_X_Q8_EXACT };
static const double IQ1M_TILE_REL_TOL = 1e-3;
static const double IQ1M_TILE_EXACT_REL_TOL = 1e-5;

static int run_iq1m_tile_pair(const void *map, uint64_t map_size,
                              uint64_t w_off, uint64_t w_bytes,
                              uint32_t k, uint32_t rows, uint32_t nexp,
                              uint32_t nt, uint32_t used,
                              enum iq1m_tile_x xmode) {
    const size_t nx = (size_t)nt * k, ny = (size_t)nt * used * rows;
    float *x = xmalloc(nx * sizeof(float));
    float *ref = xmalloc(ny * sizeof(float));
    float *got = xmalloc(ny * sizeof(float));
    int32_t *ids = xmalloc((size_t)nt * used * sizeof(int32_t));
    uint64_t seed = 20260906ull + nt;
    /* Power-of-two step: the MMVQ tier keeps the Q8_1 scale as fp16 in
     * block_q8_1.ds, the MMQ tier as fp32, so only an exact scale makes
     * the two tiers' scaled activations identical. */
    const float q8_step = 1.0f / 256.0f;
    for (size_t i = 0; i < nx; i++) {
        if (xmode == IQ1M_TILE_X_RANDOM) { x[i] = frand(&seed) * 0.5f; continue; }
        const int m = i % 32u == 0u ? 127 : (int)(frand(&seed) * 127.0f);
        x[i] = (float)m * q8_step;
    }
    for (uint32_t t = 0; t < nt; t++) {
        for (uint32_t s = 0; s < used; s++) {
            ids[(size_t)t * used + s] =
                ((t % 5u == 0u ? 7u : t * 13u) + s * 7u) % nexp;
        }
    }
    ids[used] = -1; /* Both tiers must preserve the invalid-slot zero. */
    ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc(nx * sizeof(float));
    ds4_gpu_tensor *gi = ds4_gpu_tensor_alloc((size_t)nt * used * sizeof(int32_t));
    ds4_gpu_tensor *go = ds4_gpu_tensor_alloc(ny * sizeof(float));
    int ok = gx && gi && go &&
        ds4_gpu_tensor_write(gx, 0, x, nx * sizeof(float)) &&
        ds4_gpu_tensor_write(gi, 0, ids, (size_t)nt * used * sizeof(int32_t));
    double ms[2] = {0.0, 0.0};
    for (int tile = 0; ok && tile < 2; tile++) {
        (void)setenv("DS4_MMQ_IQ1M_WORKLIST", tile ? "1" : "0", 1);
        ok = ds4_gpu_tensor_fill_f32(go, 0.0f, ny) &&
            ds4_gpu_routed_matmul_tensor(go, gx, gi, map, map_size,
                w_off, w_bytes, DS4_TENSOR_IQ1_M, k, rows, nexp, nt, used) &&
            ds4_gpu_synchronize();
        const double start = now_sec();
        for (int i = 0; ok && i < 2; i++) {
            ok = ds4_gpu_routed_matmul_tensor(go, gx, gi, map, map_size,
                w_off, w_bytes, DS4_TENSOR_IQ1_M, k, rows, nexp, nt, used);
        }
        ok = ok && ds4_gpu_synchronize();
        ms[tile] = (now_sec() - start) * 500.0;
        ok = ok && ds4_gpu_tensor_read(go, 0, tile ? got : ref, ny * sizeof(float));
    }
    (void)unsetenv("DS4_MMQ_IQ1M_WORKLIST");
    if (ok) {
        char label[80];
        snprintf(label, sizeof(label), "IQ1_M tile%s nt=%u used=%u rows=%u k=%u",
                 xmode == IQ1M_TILE_X_Q8_EXACT ? " q8-exact" : "",
                 nt, used, rows, k);
        diff_quant_gemm(label, got, ref, ny,
                        xmode == IQ1M_TILE_X_Q8_EXACT ? IQ1M_TILE_EXACT_REL_TOL
                                                      : IQ1M_TILE_REL_TOL);
        printf("%s assign=%.3f ms tile=%.3f ms\n", label, ms[0], ms[1]);
    } else {
        printf("IQ1_M tile nt=%u launch failed\n", nt); g_fail++;
    }
    ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gi); ds4_gpu_tensor_free(gx);
    free(ids); free(got); free(ref); free(x);
    return ok;
}

/* Valid 56-byte IQ1_M blocks: random grids, deltas and 3-bit sub-scales,
 * and an fp16 block scale in [2^-7, 2^-4) spread over the scale-word top
 * nibbles. Random scale bytes would give NaN or huge d, which the two
 * tiers overflow and sanitize differently. */
static void synth_iq1m_blocks(unsigned char *w, uint64_t nblocks, uint64_t *seed) {
    for (uint64_t b = 0; b < nblocks; b++) {
        unsigned char *blk = w + b * 56u;
        for (int i = 0; i < 48; i++) {
            *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
            blk[i] = (unsigned char)(*seed >> 33);
        }
        *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint16_t d = (uint16_t)(0x2000u + ((*seed >> 33) & 0x0FFFu));
        for (int j = 0; j < 4; j++) {
            *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
            const uint16_t word = (uint16_t)((((d >> (4 * j)) & 0xFu) << 12) |
                                             ((*seed >> 33) & 0x0FFFu));
            blk[48 + 2 * j] = (unsigned char)(word & 0xFFu);
            blk[49 + 2 * j] = (unsigned char)(word >> 8);
        }
    }
}

static void test_iq1m_tile_synth(void) {
    /* 32 experts is the worklist floor; rows 128 = one unchecked MMQ tile,
     * rows 96 = the checked tile; k 512 = two K iterations. All three
     * tensors share one map: the CUDA backend host-registers a map once
     * and refuses a re-registration that overlaps the previous range. */
    static const struct { uint32_t k, rows, nt; } cases[] = {
        {256u, 128u, 32u}, {512u, 96u, 257u}, {512u, 128u, 512u},
    };
    const size_t ncase = sizeof(cases) / sizeof(cases[0]);
    const uint32_t nexp = 32u, used = 8u;
    uint64_t w_off[3], w_bytes[3], map_size = 256;
    for (size_t c = 0; c < ncase; c++) {
        w_off[c] = map_size;
        w_bytes[c] = (uint64_t)nexp * cases[c].rows * (cases[c].k / 256u) * 56u;
        map_size += (w_bytes[c] + 255u) & ~(uint64_t)255u;
    }
    map_size += 256;
    void *map = NULL;
    if (posix_memalign(&map, 4096, (size_t)map_size) != 0) {
        printf("IQ1_M tile synth map alloc failed\n"); g_fail++; return;
    }
    memset(map, 0, (size_t)map_size);
    uint64_t seed = 20260906ull;
    for (size_t c = 0; c < ncase; c++) {
        synth_iq1m_blocks((unsigned char *)map + w_off[c], w_bytes[c] / 56u, &seed);
    }
    if (!ds4_gpu_set_model_map(map, map_size)) {
        printf("IQ1_M tile synth map failed\n"); g_fail++; free(map); return;
    }
    for (size_t c = 0; c < ncase; c++) {
        (void)run_iq1m_tile_pair(map, map_size, w_off[c], w_bytes[c],
                                 cases[c].k, cases[c].rows, nexp,
                                 cases[c].nt, used, IQ1M_TILE_X_RANDOM);
        (void)run_iq1m_tile_pair(map, map_size, w_off[c], w_bytes[c],
                                 cases[c].k, cases[c].rows, nexp,
                                 cases[c].nt, used, IQ1M_TILE_X_Q8_EXACT);
    }
    /* Registered map: kept until exit (see test_iq1m_prefill_synth). */
}

static void test_iq1m_prefill(const ds4_model *m, const ds4_tensor *w,
                              uint32_t used) {
    const uint32_t k = (uint32_t)w->dim[0];
    const uint32_t widths[] = {17u, 257u, 512u};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        (void)run_iq1m_prefill_pair(m->map, m->size, w->abs_offset, w->bytes,
                                    k, (uint32_t)w->dim[1], DS4_N_EXPERT,
                                    widths[wi], used);
        (void)run_iq1m_tile_pair(m->map, m->size, w->abs_offset, w->bytes,
                                 k, (uint32_t)w->dim[1], DS4_N_EXPERT,
                                 widths[wi], used, IQ1M_TILE_X_RANDOM);
        (void)run_iq1m_tile_pair(m->map, m->size, w->abs_offset, w->bytes,
                                 k, (uint32_t)w->dim[1], DS4_N_EXPERT,
                                 widths[wi], used, IQ1M_TILE_X_Q8_EXACT);
    }
}

static void test_moe_matmul(const ds4_model *m, const ds4_weights *wts) {
    /* One layer of each quant type present. The recipe puts Q4_K on the edge
     * layers and IQ2_XXS/Q3_K (or Q2_K in the pilot) on the interior ones, so
     * scanning for the first layer of each type covers both artifacts without
     * hardcoding the layer numbers. */
    uint32_t seen[64] = {0};
    for (uint32_t il = 1; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        const ds4_layer_weights *l = &wts->layer[il];
        if (!l->ffn_gate_exps) continue;
        const uint32_t types[2] = { l->ffn_gate_exps->type, l->ffn_down_exps->type };
        for (int which = 0; which < 2; which++) {
            const ds4_tensor *w = which ? l->ffn_down_exps : l->ffn_gate_exps;
            const uint32_t ty = types[which];
            if (ty >= 64 || seen[ty]) continue;
            seen[ty] = 1;

            if (ty == DS4_TENSOR_IQ2_XXS && which) { test_moe_worklist(m, w, 1u); }
            if (ty == DS4_TENSOR_IQ2_XS && which) { test_moe_worklist(m, w, 1u); }
            if (ty == DS4_TENSOR_IQ1_S && !which) { test_moe_worklist(m, w, 8u); }
            if (ty == DS4_TENSOR_IQ1_M && !which) { test_iq1m_prefill(m, w, 8u); }

            printf("testing routed tensor %.*s type=%s(%u) layer=%u\n",
                   (int)w->name.len, w->name.ptr, tensor_type_name(ty), ty, il);

            const uint32_t in_dim  = (uint32_t)w->dim[0];
            const uint32_t out_dim = (uint32_t)w->dim[1];
            const uint32_t expert  = 7;   /* arbitrary, not slot 0 */
            uint64_t s = 555 + ty;
            float *x = xmalloc((size_t)in_dim * sizeof(float));
            for (uint32_t i = 0; i < in_dim; i++) x[i] = frand(&s) * 0.5f;

            ds4_gpu_tensor *gx  = ds4_gpu_tensor_alloc((size_t)in_dim * sizeof(float));
            ds4_gpu_tensor *gid = ds4_gpu_tensor_alloc(sizeof(int32_t));
            ds4_gpu_tensor *go  = ds4_gpu_tensor_alloc((size_t)out_dim * sizeof(float));
            const int32_t id = (int32_t)expert;
            ds4_gpu_tensor_write(gx, 0, x, (size_t)in_dim * sizeof(float));
            ds4_gpu_tensor_write(gid, 0, &id, sizeof(id));

            if (!ds4_gpu_exaone_moe_matmul_tensor(go, gx, gid, m->map, m->size,
                                                  w->abs_offset, w->bytes, ty,
                                                  in_dim, out_dim, DS4_N_EXPERT,
                                                  1, 1)) {
                printf("moe matmul %s launch failed\n", tensor_type_name(ty));
                g_fail++;
            } else {
                float *got  = xmalloc((size_t)out_dim * sizeof(float));
                ds4_gpu_tensor_read(go, 0, got, (size_t)out_dim * sizeof(float));
                char label[96];
                snprintf(label, sizeof(label), "moe matmul %s [%u x %u] L%u",
                         tensor_type_name(ty), in_dim, out_dim, il);
                if (ty == DS4_TENSOR_IQ2_XS || ty == DS4_TENSOR_IQ1_S ||
                    ty == DS4_TENSOR_IQ1_M) {
                    size_t finite = 0u;
                    double energy = 0.0;
                    for (uint32_t i = 0; i < out_dim; i++) {
                        if (isfinite(got[i])) finite++;
                        energy += (double)got[i] * got[i];
                    }
                    const int ok = finite == out_dim && energy > 0.0;
                    if (!ok) g_fail++;
                    printf("%-38s finite=%zu/%u energy=%.3e  %s\n",
                           label, finite, out_dim, energy,
                           ok ? "ok" : "FAIL");
                } else {
                    float *want = xmalloc((size_t)out_dim * sizeof(float));
                    exaone_linear_expert(want, m, w, expert, x);
                    /* MMQ quantizes the activation to Q8_1 while the CPU oracle
                     * keeps f32. 2e-2 bounds that vector-level noise; a wrong
                     * expert index or transposed layout lands near one. */
                    diff_quant_gemm(label, got, want, out_dim, 2e-2);
                    free(want);
                }
                free(got);
            }
            ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gid); ds4_gpu_tensor_free(gx);
            free(x);
        }
    }
}

int main(int argc, char **argv) {
    test_k2_partial_rope_layout();
    test_context_memory_plan();
    test_k2_full_attention_memory_plan();
    test_batch_memory_plan();
    test_family_session_graph_memory_plan();
    test_exaone_rewind_span();
    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }
    test_shared_prefill_workspace();
    test_persistent_bank_runtime_ownership();

    /* Synthetic "model map": a QK-norm weight vector followed by a router
     * bias, at 256-byte-aligned offsets, registered the same way a real
     * mapping is so the weight-pointer resolution is under test too. */
    const uint64_t w_off = 256, bias_off = 256 + 4096;
    const uint64_t map_size = bias_off + 4096;
    void *map = NULL;
    if (posix_memalign(&map, 4096, (size_t)map_size) != 0) return 1;
    memset(map, 0, (size_t)map_size);
    {
        uint64_t s = 2024;
        float *nw = (float *)((char *)map + w_off);
        for (int i = 0; i < T_HEAD_DIM; i++) nw[i] = 1.0f + 0.1f * frand(&s);
        float *b = (float *)((char *)map + bias_off);
        for (int i = 0; i < T_EXPERT; i++) b[i] = 0.2f * frand(&s);
    }
    if (!ds4_gpu_set_model_map(map, map_size)) {
        fprintf(stderr, "ds4_gpu_set_model_map failed\n"); return 1;
    }

    printf("== exaone CUDA kernels vs CPU reference ==\n");
    test_qk_norm_rope(map, map_size, w_off, 40);
    /* The far end of the 262144-token window: at i == 0 the rotation angle is
     * the position itself, which is where a fast-math trig intrinsic stops
     * meaning anything. */
    test_qk_norm_rope(map, map_size, w_off, 262143);
    test_neox_rope_only(40u);
    test_neox_rope_only(524287u);
    test_attention(1);
    test_attention(0);
    test_decode_split();
    test_attention_prefill(1);
    test_attention_prefill(0);
    test_prefill_chunk_residency();
    test_router(map, map_size, bias_off);
    test_swiglu_and_combine();
    printf("\n== IQ1_M assign-major vs per-token MMVQ ==\n");
    test_iq1m_prefill_synth();
    printf("\n== IQ1_M worklist tile vs assign-major MMVQ ==\n");
    test_iq1m_tile_synth();

    if (argc > 1) {
        ds4_model model;
        model_open(&model, argv[1], false, false);
        config_validate_model(&model);
        if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_EXAONE_MOE) {
            fprintf(stderr, "not an exaone-moe model: %s\n", DS4_MODEL_SHAPE_NAME);
            return 1;
        }
        ds4_weights weights;
        weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
        if (!ds4_gpu_set_model_map(model.map, model.size)) {
            fprintf(stderr, "model map registration failed\n"); return 1;
        }
        printf("\n== routed-expert matmul vs CPU dequant-and-dot ==\n");
        test_moe_matmul(&model, &weights);
        model_close(&model);
    } else {
        printf("\n(no model path given -- routed-expert matmul tier skipped)\n");
    }

    printf("\n%s\n", g_fail ? "FAILURES" : "all checks passed");
    return g_fail ? 1 : 0;
}
