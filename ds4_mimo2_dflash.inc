/* External five-layer DFlash draft. It is not the trunk's nextn blocks.
 * Q comes from the noise tokens. K and V are the target hidden (fc of the
 * five tapped layers) followed by the noise. Attention is not causal.
 * The block is verified by the trunk; commit keeps only the prefix. */

#ifndef MIMO2_DFLASH_TRIAL

static const ds4_tensor *m2df_tensor(const ds4_model *m, const char *name,
                                     unsigned ndim, unsigned d0, unsigned d1) {
    const ds4_tensor *t = model_find_tensor(m, name);
    if (!t || t->ndim != ndim || t->dim[0] != d0) { return NULL; }
    if (ndim == 2 && t->dim[1] != d1) { return NULL; }
    return t;
}

static bool mimo2_dflash_bind(ds4_mimo2_dflash *d, const ds4_model *m) {
    ds4_str arch = {0};
    uint32_t blocks = 0, block = 0;
    memset(d, 0, sizeof(*d));
    if (!m || !model_get_string(m, "general.architecture", &arch) ||
        !ds4_streq(arch, "dflash") ||
        !model_get_u32(m, "dflash.block_count", &blocks) || blocks != 5 ||
        !model_get_u32(m, "dflash.block_size", &block) || block != 8 ||
        model_find_tensor(m, "blk.48.nextn.eh_proj.weight")) { return false; }
    d->model = m;
    d->fc = m2df_tensor(m, "fc.weight", 2, 20480, 4096);
    d->hidden_norm = m2df_tensor(m, "enc.output_norm.weight", 1, 4096, 0);
    d->output_norm = m2df_tensor(m, "output_norm.weight", 1, 4096, 0);
    if (!d->fc || !d->hidden_norm || !d->output_norm) { return false; }
    for (unsigned il = 0; il < 5; il++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", il);
        d->attn_norm[il] = m2df_tensor(m, name, 1, 4096, 0);
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", il);
        d->q[il] = m2df_tensor(m, name, 2, 4096, 8192);
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", il);
        d->k[il] = m2df_tensor(m, name, 2, 4096, 1024);
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", il);
        d->v[il] = m2df_tensor(m, name, 2, 4096, 1024);
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", il);
        d->o[il] = m2df_tensor(m, name, 2, 8192, 4096);
        snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", il);
        d->q_norm[il] = m2df_tensor(m, name, 1, 128, 0);
        snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", il);
        d->k_norm[il] = m2df_tensor(m, name, 1, 128, 0);
        snprintf(name, sizeof(name), "blk.%u.attn_sinks.weight", il);
        d->sinks[il] = m2df_tensor(m, name, 1, 64, 0);
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", il);
        d->ffn_norm[il] = m2df_tensor(m, name, 1, 4096, 0);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", il);
        d->gate[il] = m2df_tensor(m, name, 2, 4096, 16384);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", il);
        d->up[il] = m2df_tensor(m, name, 2, 4096, 16384);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", il);
        d->down[il] = m2df_tensor(m, name, 2, 16384, 4096);
        if (!d->attn_norm[il] || !d->q[il] || !d->k[il] || !d->v[il] || !d->o[il] ||
            !d->q_norm[il] || !d->k_norm[il] || !d->sinks[il] || !d->ffn_norm[il] ||
            !d->gate[il] || !d->up[il] || !d->down[il]) { return false; }
    }
    return true;
}

#else

#define MIMO2_DFLASH_HOST_FNS
#include "cuda/mimo2_dflash_host.h"
enum { DF_FF = 16384, DF_BLOCK = 8, DF_MASK = 151675 };

static int df_argmax(const float *logits) {
    unsigned best = 0;
    for (unsigned i = 1; i < DS4_N_VOCAB; i++) {
        if (logits[i] > logits[best]) { best = i; }
    }
    if (!isfinite(logits[best])) { return -1; }
    return (int)best;
}

/* One ring slot is the five target hiddens for a single position. */
enum { DF_SLOT = 5 * DF_H };

/* Verify writes the whole trial into the ring. A rejected row lands on a
 * context position the next 1024-wide window still reads. Keep the bytes
 * from before that write and put the rejected slots back on commit. */
static bool mimo2_dflash_save_ring(ds4_mimo2_graph *g, unsigned base, unsigned n) {
    if (!g->dflash_on) { return true; }
    if (!g->dflash_tap || !g->dflash_saved || !g->dflash_pos || n > M2_TRIAL) { return false; }
    const uint64_t bytes = (uint64_t)DF_SLOT * sizeof(float);
    for (unsigned i = 0; i < n; i++) {
        const unsigned slot = (base + i) % 1024u;
        g->dflash_saved_pos[i] = g->dflash_pos[slot];
        if (!ds4_gpu_tensor_read(g->dflash_tap, (uint64_t)slot * bytes,
                g->dflash_saved + (size_t)i * DF_SLOT, bytes)) {
            return false;
        }
    }
    g->dflash_saved_n = n;
    return true;
}

static bool mimo2_dflash_restore(ds4_mimo2_graph *g, unsigned base, unsigned keep) {
    if (!g->dflash_on || !g->dflash_saved_n) { return true; }
    if (keep > g->dflash_saved_n) { return false; }
    const uint64_t bytes = (uint64_t)DF_SLOT * sizeof(float);
    for (unsigned i = keep; i < g->dflash_saved_n; i++) {
        const unsigned slot = (base + i) % 1024u;
        if (!ds4_gpu_tensor_write(g->dflash_tap, (uint64_t)slot * bytes,
                g->dflash_saved + (size_t)i * DF_SLOT, bytes)) {
            return false;
        }
        g->dflash_pos[slot] = g->dflash_saved_pos[i];
    }
    g->dflash_saved_n = 0;
    return true;
}

/* Disk restore keeps trunk KV, not this ring. A hole means the next
 * speculative step must not fail the request. */
static bool mimo2_dflash_window_ready(const ds4_mimo2_graph *g, unsigned start) {
    if (!g->dflash_on || !g->dflash_pos || !start) { return false; }
    const unsigned begin = start > DF_WIN ? start - DF_WIN : 0;
    for (unsigned p = begin; p < start; p++) {
        if (g->dflash_pos[p % DF_WIN] != p) { return false; }
    }
    return true;
}

static bool mimo2_dflash_fill(ds4_session *s, int first, unsigned n) {
    ds4_mimo2_graph *g = &s->mimo2_graph;
    const ds4_mimo2_dflash *d = &s->engine->dflash;
    const ds4_weights *base = &s->engine->weights;
    const unsigned start = g->position;
    if (!d->model || !g->dflash_tap || n < 2 || start == 0) { return false; }
    unsigned ctx_pos[DF_WIN];
    unsigned ctx = 0;
    const unsigned begin = start > DF_WIN ? start - DF_WIN : 0;
    for (unsigned p = begin; p < start; p++) {
        if (g->dflash_pos[p % DF_WIN] != p) { return false; }
        ctx_pos[ctx++] = p;
    }
    if (!ctx) { return false; }
    ds4_gpu_tensor *cat = ds4_gpu_tensor_alloc((uint64_t)ctx * 5 * DF_H * sizeof(float));
    ds4_gpu_tensor *proj = ds4_gpu_tensor_alloc((uint64_t)ctx * DF_H * sizeof(float));
    ds4_gpu_tensor *noise = ds4_gpu_tensor_alloc((uint64_t)n * DF_H * sizeof(float));
    ds4_gpu_tensor *normed = ds4_gpu_tensor_alloc((uint64_t)n * DF_H * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)n * DF_Q * sizeof(float));
    ds4_gpu_tensor *k_ctx = ds4_gpu_tensor_alloc((uint64_t)ctx * DF_KV * sizeof(float));
    ds4_gpu_tensor *v_ctx = ds4_gpu_tensor_alloc((uint64_t)ctx * DF_KV * sizeof(float));
    ds4_gpu_tensor *k_noise = ds4_gpu_tensor_alloc((uint64_t)n * DF_KV * sizeof(float));
    ds4_gpu_tensor *v_noise = ds4_gpu_tensor_alloc((uint64_t)n * DF_KV * sizeof(float));
    ds4_gpu_tensor *attn = ds4_gpu_tensor_alloc((uint64_t)n * DF_Q * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n * DF_FF * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)n * DF_FF * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)n * DF_FF * sizeof(float));
    ds4_gpu_tensor *ffn = ds4_gpu_tensor_alloc((uint64_t)n * DF_H * sizeof(float));
    float *host_cat = malloc((size_t)ctx * 5 * DF_H * sizeof(float));
    float *hq = malloc((size_t)n * DF_Q * sizeof(float));
    float *hk = malloc(((size_t)ctx + n) * DF_KV * sizeof(float));
    float *hv = malloc(((size_t)ctx + n) * DF_KV * sizeof(float));
    float *ho = malloc((size_t)n * DF_Q * sizeof(float));
    unsigned *k_pos = malloc(((size_t)ctx + n) * sizeof(unsigned));
    int ids[DF_BLOCK];
    const char *cpu_env = getenv("DS4_MIMO2_DFLASH_CPU");
    const int cpu_attn = cpu_env && cpu_env[0] == '1' && cpu_env[1] == '\0';
    bool ok = cat && proj && noise && normed && q && k_ctx && v_ctx && k_noise && v_noise &&
        attn && gate && up && mid && ffn && host_cat && hq && hk && hv && ho && k_pos;
    if (ok) {
        for (unsigned i = 0; i < ctx; i++) {
            const unsigned slot = ctx_pos[i] % DF_WIN;
            ok = ds4_gpu_tensor_read(g->dflash_tap,
                ((uint64_t)slot * 5) * DF_H * sizeof(float),
                host_cat + (size_t)i * 5 * DF_H,
                5ull * DF_H * sizeof(float));
            if (!ok) { break; }
        }
    }
    ok = ok && ds4_gpu_tensor_write(cat, 0, host_cat, (uint64_t)ctx * 5 * DF_H * sizeof(float)) &&
        plain_graph_matmul_tensor(proj, d->model, d->fc, 20480, DF_H, cat, ctx) &&
        ds4_gpu_exaone_rms_norm_tensor(proj, proj, d->model->map, d->model->size,
            d->hidden_norm->abs_offset, DF_H, ctx, 1e-6f);
    ids[0] = first;
    for (unsigned i = 1; i < n; i++) { ids[i] = DF_MASK; }
    ok = ok && ds4_gpu_tensor_write(g->tokens, 0, ids, n * sizeof(int)) &&
        ds4_gpu_embed_tokens_quant_tensor(noise, g->tokens, s->engine->model.map,
            s->engine->model.size, base->token_embd->abs_offset, base->token_embd->type,
            DS4_N_VOCAB, n, DF_H);
    for (unsigned il = 0; ok && il < 5; il++) {
        ok = ds4_gpu_exaone_rms_norm_tensor(normed, noise, d->model->map, d->model->size,
                d->attn_norm[il]->abs_offset, DF_H, n, 1e-6f) &&
            plain_graph_matmul_tensor(q, d->model, d->q[il], DF_H, DF_Q, normed, n) &&
            plain_graph_matmul_tensor(k_noise, d->model, d->k[il], DF_H, DF_KV, normed, n) &&
            plain_graph_matmul_tensor(v_noise, d->model, d->v[il], DF_H, DF_KV, normed, n) &&
            plain_graph_matmul_tensor(k_ctx, d->model, d->k[il], DF_H, DF_KV, proj, ctx) &&
            plain_graph_matmul_tensor(v_ctx, d->model, d->v[il], DF_H, DF_KV, proj, ctx);
        if (!ok) { break; }
        /* Host loops are the previous path. They stay in this C file so the
         * switch keeps the gcc build. The device path skips the copies. */
        if (cpu_attn) {
            ok = ds4_gpu_synchronize() &&
                ds4_gpu_tensor_read(q, 0, hq, (uint64_t)n * DF_Q * sizeof(float)) &&
                ds4_gpu_tensor_read(k_ctx, 0, hk, (uint64_t)ctx * DF_KV * sizeof(float)) &&
                ds4_gpu_tensor_read(k_noise, 0, hk + (size_t)ctx * DF_KV, (uint64_t)n * DF_KV * sizeof(float)) &&
                ds4_gpu_tensor_read(v_ctx, 0, hv, (uint64_t)ctx * DF_KV * sizeof(float)) &&
                ds4_gpu_tensor_read(v_noise, 0, hv + (size_t)ctx * DF_KV, (uint64_t)n * DF_KV * sizeof(float));
            if (!ok) { break; }
            float q_weight[DF_HD], k_weight[DF_HD], sinks[DF_QH];
            memcpy(q_weight, (const char *)d->model->map + d->q_norm[il]->abs_offset, sizeof(q_weight));
            memcpy(k_weight, (const char *)d->model->map + d->k_norm[il]->abs_offset, sizeof(k_weight));
            memcpy(sinks, (const char *)d->model->map + d->sinks[il]->abs_offset, sizeof(sinks));
            for (unsigned row = 0; row < n * DF_QH; row++) { df_rms(hq + row * DF_HD, q_weight, DF_HD); }
            for (unsigned row = 0; row < (ctx + n) * DF_KH; row++) { df_rms(hk + row * DF_HD, k_weight, DF_HD); }
            for (unsigned i = 0; i < n; i++) { df_rope(hq + (size_t)i * DF_Q, DF_QH, start + i); }
            for (unsigned i = 0; i < ctx; i++) { df_rope(hk + (size_t)i * DF_KV, DF_KH, ctx_pos[i]); }
            for (unsigned i = 0; i < n; i++) {
                df_rope(hk + ((size_t)ctx + i) * DF_KV, DF_KH, start + i);
            }
            for (unsigned i = 0; i < ctx; i++) { k_pos[i] = ctx_pos[i]; }
            for (unsigned i = 0; i < n; i++) { k_pos[ctx + i] = start + i; }
            df_attn(ho, hq, hk, hv, sinks, k_pos, start, n, ctx + n);
            ok = ds4_gpu_tensor_write(attn, 0, ho, (uint64_t)n * DF_Q * sizeof(float));
        } else {
            ok = ds4_gpu_mimo2_dflash_attn(attn, q, k_ctx, k_noise, v_ctx, v_noise,
                    (const float *)((const char *)d->model->map + d->q_norm[il]->abs_offset),
                    (const float *)((const char *)d->model->map + d->k_norm[il]->abs_offset),
                    (const float *)((const char *)d->model->map + d->sinks[il]->abs_offset),
                    start, n, ctx) == 2;
        }
        ok = ok &&
            plain_graph_matmul_tensor(ffn, d->model, d->o[il], DF_Q, DF_H, attn, n) &&
            ds4_gpu_dots3_scale_tensor(ffn, 1.0f, (uint64_t)n * DF_H) &&
            ds4_gpu_exaone_add_tensor(noise, ffn, (uint64_t)n * DF_H) &&
            ds4_gpu_exaone_rms_norm_tensor(normed, noise, d->model->map, d->model->size,
                d->ffn_norm[il]->abs_offset, DF_H, n, 1e-6f) &&
            plain_graph_matmul_tensor(gate, d->model, d->gate[il], DF_H, DF_FF, normed, n) &&
            plain_graph_matmul_tensor(up, d->model, d->up[il], DF_H, DF_FF, normed, n) &&
            ds4_gpu_step37_swiglu(mid, gate, up, NULL, DF_FF, n, 0) &&
            plain_graph_matmul_tensor(ffn, d->model, d->down[il], DF_FF, DF_H, mid, n) &&
            ds4_gpu_exaone_add_tensor(noise, ffn, (uint64_t)n * DF_H);
    }
    ok = ok && ds4_gpu_exaone_rms_norm_tensor(normed, noise, d->model->map, d->model->size,
            d->output_norm->abs_offset, DF_H, n, 1e-6f);
    for (unsigned row = 1; ok && row < n; row++) {
        ds4_gpu_tensor *one = ds4_gpu_tensor_view(normed, (uint64_t)row * DF_H * sizeof(float),
            DF_H * sizeof(float));
        ok = one && plain_graph_matmul_tensor(g->logits, &s->engine->model, base->output,
                DF_H, DS4_N_VOCAB, one, 1) &&
            ds4_gpu_tensor_read(g->logits, 0, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float));
        ds4_gpu_tensor_free(one);
        if (!ok) { break; }
        const int id = df_argmax(s->logits);
        if (id < 0) { ok = false; break; }
        g->trial[row] = id;
    }
    ds4_gpu_tensor_free(cat); ds4_gpu_tensor_free(proj); ds4_gpu_tensor_free(noise);
    ds4_gpu_tensor_free(normed); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k_ctx);
    ds4_gpu_tensor_free(v_ctx); ds4_gpu_tensor_free(k_noise); ds4_gpu_tensor_free(v_noise);
    ds4_gpu_tensor_free(attn); ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(ffn);
    free(host_cat); free(hq); free(hk); free(hv); free(ho); free(k_pos);
    return ok;
}

#endif
