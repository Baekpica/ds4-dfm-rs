/* dots3-note CUDA kernel parity, model-free.
 *
 * Latent attention: the tensor-core prefill kernel (cuda/mmq/ds4_fattn.cu)
 * against the scalar warp-per-(token, head) kernel on random data, for the
 * full geometry (latent 512, DSA selection and plain causal) and the SWA
 * geometry (latent 1024, 513-window ring).  The two differ by the BF16
 * rounding of Q and P and by fp32 reordering; the gate is a relative RMS
 * band, not bit identity.
 *
 * DS4_DOTS3_PROFILE_ATTN=1 times both kernels on the production 4,096-row
 * chunk shapes instead (NCU / nsys probe, no model needed).
 */
#include "ds4_gpu.h"
#include "cuda/mmq/ds4_mmq.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1u) {}
    uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (uint32_t)(s >> 11);
    }
    float uniform() { return (float)(next() & 0xffffffu) / 16777216.0f; }
    float normal() {
        const float u1 = uniform() + 1e-7f, u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
    }
};

void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

ds4_gpu_tensor *upload(const void *data, size_t bytes, const char *what) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (!t || !ds4_gpu_tensor_write(t, 0, data, bytes)) {
        fprintf(stderr, "upload failed: %s\n", what);
        std::exit(1);
    }
    return t;
}

uint16_t to_bf16(float x) {
    __nv_bfloat16 b = __float2bfloat16_rn(x);
    uint16_t u;
    memcpy(&u, &b, 2);
    return u;
}

struct attn_case {
    const char *name;
    uint32_t heads, latent, nope, rows, pos0, cache_cap, window, sel_stride;
    bool selection;
    bool fillers;          /* mix DSA filler ids into the selection */
    bool split;            /* decode split-K entry vs the serial kernel */
};

struct attn_result { double rel_rms, max_abs; bool finite; };

/* Build the inputs once, run the scalar kernel (DS4_DOTS3_ATTN_NO_HMMA=1)
 * and the default dispatch (tensor cores at prefill widths), compare. */
attn_result run_case(const attn_case &c, bool profile, int reps) {
    rng r(c.rows * 131u + c.heads + c.latent);
    const uint32_t key_dim = c.nope + 64u;
    const size_t q_n = (size_t)c.rows * c.heads * key_dim;
    const size_t qa_n = (size_t)c.rows * c.heads * c.latent;
    std::vector<float> q(q_n), qa(qa_n);
    for (auto &x : q) x = r.normal();
    for (auto &x : qa) x = r.normal() * 0.5f;
    std::vector<uint16_t> latent((size_t)c.cache_cap * c.latent);
    std::vector<uint16_t> kpe((size_t)c.cache_cap * 64u);
    for (auto &x : latent) x = to_bf16(r.normal());
    for (auto &x : kpe) x = to_bf16(r.normal());
    std::vector<int32_t> sel;
    if (c.selection) {
        sel.resize((size_t)c.rows * c.sel_stride);
        std::vector<int32_t> ids;
        for (uint32_t t = 0; t < c.rows; t++) {
            const uint32_t qpos = c.pos0 + t;
            ids.resize(qpos + 1u);
            for (uint32_t i = 0; i <= qpos; i++) ids[i] = (int32_t)i;
            for (uint32_t i = qpos; i > 0; i--) {
                const uint32_t j = r.next() % (i + 1u);
                std::swap(ids[i], ids[j]);
            }
            for (uint32_t j = 0; j < c.sel_stride; j++) {
                int32_t id = j < ids.size() ? ids[j] : -1;
                if (c.fillers && (r.next() % 20u) == 0u) {
                    id = (r.next() & 1u) ? -1 : (int32_t)(qpos + 1u + r.next() % 5u);
                }
                sel[(size_t)t * c.sel_stride + j] = id;
            }
        }
    }
    ds4_gpu_tensor *t_q = upload(q.data(), q_n * 4u, "q");
    ds4_gpu_tensor *t_qa = upload(qa.data(), qa_n * 4u, "q_absorbed");
    ds4_gpu_tensor *t_lat = upload(latent.data(), latent.size() * 2u, "latent");
    ds4_gpu_tensor *t_kpe = upload(kpe.data(), kpe.size() * 2u, "k_pe");
    ds4_gpu_tensor *t_sel = c.selection ? upload(sel.data(), sel.size() * 4u, "selected") : nullptr;
    ds4_gpu_tensor *t_ref = ds4_gpu_tensor_alloc(qa_n * 4u);
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(qa_n * 4u);
    if (!t_ref || !t_out) { fprintf(stderr, "output alloc failed\n"); std::exit(1); }
    const float scale = 1.0f / std::sqrt((float)key_dim);

    ds4_gpu_tensor *t_part = c.split
        ? ds4_gpu_tensor_alloc((uint64_t)c.rows * c.heads * 16u * (c.latent + 4u) * 4u)
        : nullptr;
    auto launch = [&](ds4_gpu_tensor *out, bool scalar) {
        if (scalar) setenv("DS4_DOTS3_ATTN_NO_HMMA", "1", 1);
        else unsetenv("DS4_DOTS3_ATTN_NO_HMMA");
        const int ok = c.split
            ? (scalar ? ds4_gpu_dots3_latent_attention_tensor(
                            out, t_q, t_qa, t_lat, t_kpe, t_sel,
                            c.selection ? c.sel_stride : 0u, c.rows, c.pos0,
                            c.cache_cap, c.window, c.heads, c.latent, c.nope,
                            64u, scale)
                      : ds4_gpu_dots3_latent_attention_split_tensor(
                            out, t_part, t_q, t_qa, t_lat, t_kpe, t_sel,
                            c.selection ? c.sel_stride : 0u, c.rows, c.pos0,
                            c.cache_cap, c.window, c.heads, c.latent, c.nope,
                            64u, scale))
            : ds4_gpu_dots3_latent_attention_tensor(
                  out, t_q, t_qa, t_lat, t_kpe, t_sel,
                  c.selection ? c.sel_stride : 0u, c.rows, c.pos0, c.cache_cap,
                  c.window, c.heads, c.latent, c.nope, 64u, scale);
        if (!ok) {
            fprintf(stderr, "%s: latent attention launch failed (%s)\n",
                    c.name, scalar ? "scalar" : "hmma/split");
            std::exit(1);
        }
    };
    launch(t_ref, true);
    launch(t_out, false);
    check(cudaDeviceSynchronize(), "attention sync");

    if (profile) {
        cudaEvent_t e0, e1;
        check(cudaEventCreate(&e0), "event0");
        check(cudaEventCreate(&e1), "event1");
        float ms[2] = {0.0f, 0.0f};
        for (int which = 0; which < 2; which++) {
            const bool scalar = which == 0;
            const int n = scalar && !c.split ? 2 : reps;
            check(cudaEventRecord(e0), "t0");
            for (int i = 0; i < n; i++) launch(scalar ? t_ref : t_out, scalar);
            check(cudaEventRecord(e1), "t1");
            check(cudaEventSynchronize(e1), "timed sync");
            check(cudaEventElapsedTime(&ms[which], e0, e1), "elapsed");
            ms[which] /= (float)n;
        }
        printf("dots3 attention profile %-14s rows=%u heads=%u latent=%u keys=%u: "
               "scalar %.3f ms, %s %.3f ms (%.2fx)\n",
               c.name, c.rows, c.heads, c.latent,
               c.selection ? c.sel_stride : c.window, ms[0],
               c.split ? "split" : "hmma", ms[1],
               ms[1] > 0.0f ? ms[0] / ms[1] : 0.0f);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }

    std::vector<float> ref(qa_n), out(qa_n);
    if (!ds4_gpu_tensor_read(t_ref, 0, ref.data(), qa_n * 4u) ||
        !ds4_gpu_tensor_read(t_out, 0, out.data(), qa_n * 4u)) {
        fprintf(stderr, "readback failed\n");
        std::exit(1);
    }
    double num = 0.0, den = 0.0, max_abs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < qa_n; i++) {
        if (!std::isfinite(out[i]) || !std::isfinite(ref[i])) finite = false;
        const double d = (double)out[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
    }
    ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_ref);
    if (t_part) ds4_gpu_tensor_free(t_part);
    if (t_sel) ds4_gpu_tensor_free(t_sel);
    ds4_gpu_tensor_free(t_kpe);
    ds4_gpu_tensor_free(t_lat);
    ds4_gpu_tensor_free(t_qa);
    ds4_gpu_tensor_free(t_q);
    return {den > 0.0 ? std::sqrt(num / den) : std::sqrt(num), max_abs, finite};
}

/* Latent value projection: synthetic Q8_0 W_UV rows laid out as the owner's
 * transposed artifact planes, HMMA against an FP32 host reference on the
 * dequantized weights. */
struct vp_case { const char *name; uint32_t heads, latent, rows; };

struct vp_result { double rel_rms, max_abs; bool finite; float ms; };

vp_result run_vp_case(const vp_case &c, bool profile, int reps) {
    rng r(c.rows * 7u + c.latent);
    const uint32_t k_blocks = c.latent / 32u;
    const size_t lat_n = (size_t)c.rows * c.heads * c.latent;
    const size_t out_n = (size_t)c.rows * c.heads * 128u;
    std::vector<float> latent(lat_n);
    for (auto &x : latent) x = r.normal();
    /* Planes: scale[(head*k_blocks+b)*128+v], code[((head*k_blocks+b)*32+k)*128+v]. */
    std::vector<uint16_t> scale((size_t)c.heads * k_blocks * 128u);
    std::vector<int8_t> code((size_t)c.heads * k_blocks * 32u * 128u);
    std::vector<float> scale_f(scale.size());
    for (size_t i = 0; i < scale.size(); i++) {
        const float s = 0.01f + 0.02f * r.uniform();
        const __half h = __float2half(s);
        memcpy(&scale[i], &h, 2);
        scale_f[i] = __half2float(h);
    }
    for (auto &x : code) x = (int8_t)((int)(r.next() % 255u) - 127);
    std::vector<float> ref(out_n);
    if (!profile) {
        for (uint32_t t = 0; t < c.rows; t++) {
            for (uint32_t h = 0; h < c.heads; h++) {
                const float *x = latent.data() + ((size_t)t * c.heads + h) * c.latent;
                for (uint32_t v = 0; v < 128u; v++) {
                    double acc = 0.0;
                    for (uint32_t b = 0; b < k_blocks; b++) {
                        const size_t hb = (size_t)h * k_blocks + b;
                        double dot = 0.0;
                        for (uint32_t k = 0; k < 32u; k++)
                            dot += (double)code[(hb * 32u + k) * 128u + v] * x[b * 32u + k];
                        acc += (double)scale_f[hb * 128u + v] * dot;
                    }
                    ref[((size_t)t * c.heads + h) * 128u + v] = (float)acc;
                }
            }
        }
    }
    ds4_gpu_tensor *t_lat = upload(latent.data(), lat_n * 4u, "latent");
    ds4_gpu_tensor *t_scale = upload(scale.data(), scale.size() * 2u, "scale");
    ds4_gpu_tensor *t_code = upload(code.data(), code.size(), "code");
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(out_n * 4u);
    if (!t_out) { fprintf(stderr, "vp out alloc failed\n"); std::exit(1); }
    auto launch = [&]() {
        if (!ds4_gpu_dots3_value_project_planes_tensor(
                t_out, t_lat, ds4_gpu_tensor_ptr(t_scale), ds4_gpu_tensor_ptr(t_code),
                nullptr, c.rows, c.heads, c.latent)) {
            fprintf(stderr, "%s: value projection launch failed\n", c.name);
            std::exit(1);
        }
    };
    launch();
    check(cudaDeviceSynchronize(), "vp sync");
    /* Decode widths: the grouped kernel against the one-group transposed
     * kernel it replaces (fp32 reorder of the block sum only). */
    double alt_rel = 0.0;
    if (c.rows < 16u && !profile) {
        std::vector<float> a(out_n), b(out_n);
        if (!ds4_gpu_tensor_read(t_out, 0, a.data(), out_n * 4u)) std::exit(1);
        setenv("DS4_DOTS3_VALUE_NO_DECODE", "1", 1);
        launch();
        check(cudaDeviceSynchronize(), "vp alt sync");
        unsetenv("DS4_DOTS3_VALUE_NO_DECODE");
        if (!ds4_gpu_tensor_read(t_out, 0, b.data(), out_n * 4u)) std::exit(1);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < out_n; i++) {
            const double d = (double)a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        alt_rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
        launch();
        check(cudaDeviceSynchronize(), "vp sync");
    }
    float ms = 0.0f;
    if (profile) {
        cudaEvent_t e0, e1;
        check(cudaEventCreate(&e0), "event0");
        check(cudaEventCreate(&e1), "event1");
        check(cudaEventRecord(e0), "t0");
        for (int i = 0; i < reps; i++) launch();
        check(cudaEventRecord(e1), "t1");
        check(cudaEventSynchronize(e1), "timed sync");
        check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        ms /= (float)reps;
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    std::vector<float> out(out_n);
    if (!ds4_gpu_tensor_read(t_out, 0, out.data(), out_n * 4u)) {
        fprintf(stderr, "vp readback failed\n");
        std::exit(1);
    }
    double num = 0.0, den = 0.0, max_abs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < out_n; i++) {
        if (!std::isfinite(out[i])) finite = false;
        if (profile) continue;
        const double d = (double)out[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
    }
    ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_code);
    ds4_gpu_tensor_free(t_scale);
    ds4_gpu_tensor_free(t_lat);
    if (alt_rel > 1.0e-5) finite = false;   /* grouped vs one-group walk */
    return {den > 0.0 ? std::sqrt(num / den) : 0.0, max_abs, finite, ms};
}

/* Q/K absorption: synthetic raw Q8_0 attn_kv_b rows (row h*(nope+128)+d,
 * latent_dim columns in 34-byte blocks), HMMA against an FP32 host
 * reference on the dequantized weights.  Decode widths keep the engine's
 * wide kernel (see ds4_cuda.cu) and are not exercised here. */
struct ab_case { const char *name; uint32_t heads, latent, nope, rows; };

vp_result run_ab_case(const ab_case &c, bool profile, int reps) {
    rng r(c.rows * 11u + c.latent + c.nope);
    const uint32_t key_dim = c.nope + 64u;
    const uint32_t value_dim = 128u;
    const uint32_t k_blocks = c.latent / 32u;
    const size_t row_bytes = (size_t)k_blocks * 34u;
    const size_t weight_rows = (size_t)c.heads * (c.nope + value_dim);
    const size_t q_n = (size_t)c.rows * c.heads * key_dim;
    const size_t out_n = (size_t)c.rows * c.heads * c.latent;
    std::vector<float> q(q_n);
    for (auto &x : q) x = r.normal();
    std::vector<uint8_t> raw(weight_rows * row_bytes);
    std::vector<float> w((size_t)c.heads * c.nope * c.latent);   /* dequantized [h][d][j] */
    for (size_t row = 0; row < weight_rows; row++) {
        const uint32_t h = (uint32_t)(row / (c.nope + value_dim));
        const uint32_t d = (uint32_t)(row % (c.nope + value_dim));
        for (uint32_t b = 0; b < k_blocks; b++) {
            uint8_t *blk = raw.data() + row * row_bytes + (size_t)b * 34u;
            const float s = 0.005f + 0.01f * r.uniform();
            const __half hs = __float2half(s);
            memcpy(blk, &hs, 2);
            for (uint32_t k = 0; k < 32u; k++) {
                const int8_t code = (int8_t)((int)(r.next() % 255u) - 127);
                blk[2u + k] = (uint8_t)code;
                if (d < c.nope) {
                    w[((size_t)h * c.nope + d) * c.latent + b * 32u + k] =
                        __half2float(hs) * (float)code;
                }
            }
        }
    }
    std::vector<float> ref(out_n);
    if (!profile) {
        for (uint32_t t = 0; t < c.rows; t++) {
            for (uint32_t h = 0; h < c.heads; h++) {
                const float *qh = q.data() + ((size_t)t * c.heads + h) * key_dim;
                for (uint32_t j = 0; j < c.latent; j++) {
                    double acc = 0.0;
                    for (uint32_t d = 0; d < c.nope; d++)
                        acc += (double)qh[d] * w[((size_t)h * c.nope + d) * c.latent + j];
                    ref[((size_t)t * c.heads + h) * c.latent + j] = (float)acc;
                }
            }
        }
    }
    ds4_gpu_tensor *t_q = upload(q.data(), q_n * 4u, "q");
    ds4_gpu_tensor *t_w = upload(raw.data(), raw.size(), "raw kv_b");
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(out_n * 4u);
    if (!t_out) { fprintf(stderr, "absorb out alloc failed\n"); std::exit(1); }
    auto launch = [&]() {
        if (ds4_mmq_dots3_absorb_hmma(
                (float *)ds4_gpu_tensor_ptr(t_out), (const float *)ds4_gpu_tensor_ptr(t_q),
                ds4_gpu_tensor_ptr(t_w), (int)c.rows, (int)c.heads, (int)c.latent,
                (int)c.nope, (int)key_dim, (int)value_dim, row_bytes, 0) != 0) {
            fprintf(stderr, "%s: absorb hmma launch failed\n", c.name);
            std::exit(1);
        }
    };
    launch();
    check(cudaDeviceSynchronize(), "absorb sync");
    float ms = 0.0f;
    if (profile) {
        cudaEvent_t e0, e1;
        check(cudaEventCreate(&e0), "event0");
        check(cudaEventCreate(&e1), "event1");
        check(cudaEventRecord(e0), "t0");
        for (int i = 0; i < reps; i++) launch();
        check(cudaEventRecord(e1), "t1");
        check(cudaEventSynchronize(e1), "timed sync");
        check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        ms /= (float)reps;
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    std::vector<float> out(out_n);
    if (!ds4_gpu_tensor_read(t_out, 0, out.data(), out_n * 4u)) {
        fprintf(stderr, "absorb readback failed\n");
        std::exit(1);
    }
    double num = 0.0, den = 0.0, max_abs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < out_n; i++) {
        if (!std::isfinite(out[i])) finite = false;
        if (profile) continue;
        const double d = (double)out[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
    }
    ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_w);
    ds4_gpu_tensor_free(t_q);
    return {den > 0.0 ? std::sqrt(num / den) : 0.0, max_abs, finite, ms};
}

/* D3 fusions: every fused launch must reproduce the separate kernels it
 * replaces byte for byte (same arithmetic, same order). */
struct fused_result { const char *name; bool same; };

std::vector<uint8_t> read_bytes(const ds4_gpu_tensor *t, size_t bytes) {
    std::vector<uint8_t> out(bytes);
    if (!ds4_gpu_tensor_read(t, 0, out.data(), bytes)) {
        fprintf(stderr, "readback failed\n");
        std::exit(1);
    }
    return out;
}

std::vector<fused_result> run_fused_cases() {
    std::vector<fused_result> results;
    rng r(4242u);
    const float eps = 1e-6f;
    std::vector<int32_t> positions;
    std::vector<float> inv_full(32), inv_swa(32);
    for (uint32_t i = 0; i < 32u; i++) {
        const float expo = (2.0f * (float)i) / 64.0f;
        inv_full[i] = 1.0f / powf(8.0e7f, expo);
        inv_swa[i] = 1.0f / powf(5.0e4f, expo);
    }
    ds4_gpu_tensor *t_inv_full = upload(inv_full.data(), 32u * 4u, "inv full");
    ds4_gpu_tensor *t_inv_swa = upload(inv_swa.data(), 32u * 4u, "inv swa");

    /* (b) kv finish, both geometries, ring and linear. */
    for (int geo = 0; geo < 2; geo++) {
        const uint32_t kv_lora = geo == 0 ? 512u : 1024u;
        const uint32_t rows = 37u, cache_cap = geo == 0 ? 4096u : 513u + 4096u;
        const uint32_t pos0 = geo == 0 ? 1000u : 4590u;
        const bool ring = geo == 1;
        const uint32_t raw_dim = kv_lora + 64u;
        std::vector<float> kv_raw((size_t)rows * raw_dim), w_l(kv_lora), w_r(64u);
        for (auto &x : kv_raw) x = r.normal() * 3.0f;
        for (auto &x : w_l) x = 0.5f + r.uniform();
        for (auto &x : w_r) x = 0.5f + r.uniform();
        positions.resize(rows);
        for (uint32_t i = 0; i < rows; i++) positions[i] = (int32_t)(pos0 + i);
        ds4_gpu_tensor *t_raw = upload(kv_raw.data(), kv_raw.size() * 4u, "kv raw");
        ds4_gpu_tensor *t_wl = upload(w_l.data(), w_l.size() * 4u, "w latent");
        ds4_gpu_tensor *t_wr = upload(w_r.data(), w_r.size() * 4u, "w rope");
        ds4_gpu_tensor *t_pos = upload(positions.data(), rows * 4u, "positions");
        const size_t lat_bytes = (size_t)cache_cap * kv_lora * 2u;
        const size_t kpe_bytes = (size_t)cache_cap * 64u * 2u;
        ds4_gpu_tensor *t_lat_a = ds4_gpu_tensor_alloc(lat_bytes);
        ds4_gpu_tensor *t_kpe_a = ds4_gpu_tensor_alloc(kpe_bytes);
        ds4_gpu_tensor *t_lat_b = ds4_gpu_tensor_alloc(lat_bytes);
        ds4_gpu_tensor *t_kpe_b = ds4_gpu_tensor_alloc(kpe_bytes);
        ds4_gpu_tensor *t_norm = ds4_gpu_tensor_alloc((size_t)rows * kv_lora * 4u);
        ds4_gpu_tensor *t_kpe = ds4_gpu_tensor_alloc((size_t)rows * 64u * 4u);
        const ds4_gpu_tensor *inv = geo == 0 ? t_inv_full : t_inv_swa;
        std::vector<uint8_t> zero_l(lat_bytes, 0), zero_k(kpe_bytes, 0);
        ds4_gpu_tensor_write(t_lat_a, 0, zero_l.data(), lat_bytes);
        ds4_gpu_tensor_write(t_lat_b, 0, zero_l.data(), lat_bytes);
        ds4_gpu_tensor_write(t_kpe_a, 0, zero_k.data(), kpe_bytes);
        ds4_gpu_tensor_write(t_kpe_b, 0, zero_k.data(), kpe_bytes);
        const bool ok_a =
            ds4_gpu_dots3_rms_norm_dev_tensor(t_norm, t_raw, t_wl, kv_lora, raw_dim, 0u, rows, eps) &&
            ds4_gpu_dots3_rms_norm_dev_tensor(t_kpe, t_raw, t_wr, 64u, raw_dim, kv_lora, rows, eps) &&
            ds4_gpu_dots3_rope_interleaved_tensor(t_kpe, t_pos, inv, rows, 1u, 64u, 64u, 0u) &&
            ds4_gpu_dots3_store_latent_kpe_tensor(t_lat_a, t_kpe_a, t_norm, t_kpe, t_pos, rows, cache_cap, kv_lora, 64u, ring);
        const bool ok_b = ds4_gpu_dots3_kv_finish_tensor(
            t_lat_b, t_kpe_b, t_raw, t_wl, t_wr, t_pos, inv, rows, kv_lora, 64u, cache_cap, ring, eps);
        check(cudaDeviceSynchronize(), "kv finish sync");
        bool same = ok_a && ok_b &&
            read_bytes(t_lat_a, lat_bytes) == read_bytes(t_lat_b, lat_bytes) &&
            read_bytes(t_kpe_a, kpe_bytes) == read_bytes(t_kpe_b, kpe_bytes);
        results.push_back({geo == 0 ? "kv-finish-full" : "kv-finish-swa", same});
        for (ds4_gpu_tensor *x : {t_raw, t_wl, t_wr, t_pos, t_lat_a, t_kpe_a, t_lat_b, t_kpe_b, t_norm, t_kpe})
            ds4_gpu_tensor_free(x);
    }

    /* (d) indexer key finish, query finish, and the weight scale fold. */
    {
        const uint32_t rows = 29u, cache_cap = 4096u, pos0 = 700u;
        std::vector<float> xk((size_t)rows * 128u), w(128u), b(128u);
        for (auto &x : xk) x = r.normal() * 2.0f;
        for (auto &x : w) x = 0.5f + r.uniform();
        for (auto &x : b) x = r.normal() * 0.1f;
        positions.resize(rows);
        for (uint32_t i = 0; i < rows; i++) positions[i] = (int32_t)(pos0 + i);
        ds4_gpu_tensor *t_pos = upload(positions.data(), rows * 4u, "positions");
        ds4_gpu_tensor *t_xa = upload(xk.data(), xk.size() * 4u, "idx k a");
        ds4_gpu_tensor *t_xb = upload(xk.data(), xk.size() * 4u, "idx k b");
        ds4_gpu_tensor *t_w = upload(w.data(), 128u * 4u, "ln w");
        ds4_gpu_tensor *t_b = upload(b.data(), 128u * 4u, "ln b");
        const size_t cache_bytes = (size_t)cache_cap * 128u * 4u;
        ds4_gpu_tensor *t_ca = ds4_gpu_tensor_alloc(cache_bytes);
        ds4_gpu_tensor *t_cb = ds4_gpu_tensor_alloc(cache_bytes);
        std::vector<uint8_t> zero(cache_bytes, 0);
        ds4_gpu_tensor_write(t_ca, 0, zero.data(), cache_bytes);
        ds4_gpu_tensor_write(t_cb, 0, zero.data(), cache_bytes);
        const bool ok_a =
            ds4_gpu_dots3_layernorm_dev_tensor(t_xa, t_w, t_b, 128u, rows, eps) &&
            ds4_gpu_dots3_rope_interleaved_tensor(t_xa, t_pos, t_inv_full, rows, 1u, 128u, 64u, 0u) &&
            ds4_gpu_motif3_round_bf16_tensor(t_xa, t_xa, (uint64_t)rows * 128u) &&
            ds4_gpu_dots3_fp8_roundtrip_tensor(t_xa, rows) &&
            ds4_gpu_dots3_idx_store_tensor(t_ca, t_xa, t_pos, rows, cache_cap);
        const bool ok_b = ds4_gpu_dots3_idx_k_finish_tensor(
            t_cb, t_xb, t_w, t_b, t_pos, t_inv_full, rows, cache_cap, eps);
        check(cudaDeviceSynchronize(), "idx k finish sync");
        results.push_back({"idx-k-finish", ok_a && ok_b &&
            read_bytes(t_ca, cache_bytes) == read_bytes(t_cb, cache_bytes)});

        const uint32_t heads = 64u;
        std::vector<float> xq((size_t)rows * heads * 128u);
        for (auto &x : xq) x = r.normal() * 2.0f;
        ds4_gpu_tensor *t_qa = upload(xq.data(), xq.size() * 4u, "idx q a");
        ds4_gpu_tensor *t_qb = upload(xq.data(), xq.size() * 4u, "idx q b");
        const bool ok_qa =
            ds4_gpu_dots3_rope_interleaved_tensor(t_qa, t_pos, t_inv_full, rows, heads, 128u, 64u, 0u) &&
            ds4_gpu_dots3_fp8_roundtrip_tensor(t_qa, rows * heads);
        const bool ok_qb = ds4_gpu_dots3_idx_q_finish_tensor(t_qb, t_pos, t_inv_full, rows, heads);
        check(cudaDeviceSynchronize(), "idx q finish sync");
        results.push_back({"idx-q-finish", ok_qa && ok_qb &&
            read_bytes(t_qa, xq.size() * 4u) == read_bytes(t_qb, xq.size() * 4u)});

        /* Score with the scale folded vs the separate scale pass. */
        const uint32_t n_keys = pos0 + rows;
        std::vector<float> wq((size_t)rows * heads);
        for (auto &x : wq) x = r.normal();
        ds4_gpu_tensor *t_wa = upload(wq.data(), wq.size() * 4u, "idx w a");
        ds4_gpu_tensor *t_wb = upload(wq.data(), wq.size() * 4u, "idx w b");
        ds4_gpu_tensor *t_sa = ds4_gpu_tensor_alloc((size_t)rows * n_keys * 4u);
        ds4_gpu_tensor *t_sb = ds4_gpu_tensor_alloc((size_t)rows * n_keys * 4u);
        const float w_scale = 1.0f / sqrtf(128.0f * 64.0f);
        const bool ok_sa =
            ds4_gpu_dots3_scale_tensor(t_wa, w_scale, (uint64_t)rows * heads) &&
            ds4_gpu_dots3_idx_score_tensor(t_sa, t_qb, t_wa, t_cb, t_pos, rows, n_keys, cache_cap, 1.0f);
        const bool ok_sb =
            ds4_gpu_dots3_idx_score_tensor(t_sb, t_qb, t_wb, t_cb, t_pos, rows, n_keys, cache_cap, w_scale);
        check(cudaDeviceSynchronize(), "idx score sync");
        results.push_back({"idx-score-scale", ok_sa && ok_sb &&
            read_bytes(t_sa, (size_t)rows * n_keys * 4u) == read_bytes(t_sb, (size_t)rows * n_keys * 4u)});
        for (ds4_gpu_tensor *x : {t_pos, t_xa, t_xb, t_w, t_b, t_ca, t_cb, t_qa, t_qb, t_wa, t_wb, t_sa, t_sb})
            ds4_gpu_tensor_free(x);
    }

    /* (c) gated value projection (planes entry) vs value + separate gate. */
    for (int width = 0; width < 2; width++) {
        const uint32_t heads = 8u, latent = 512u, rows = width == 0 ? 3u : 40u;
        const uint32_t k_blocks = latent / 32u;
        std::vector<float> lat((size_t)rows * heads * latent), gate((size_t)rows * heads);
        for (auto &x : lat) x = r.normal();
        for (auto &x : gate) x = r.normal() * 2.0f;
        std::vector<uint16_t> scale((size_t)heads * k_blocks * 128u);
        std::vector<int8_t> code((size_t)heads * k_blocks * 32u * 128u);
        for (auto &s : scale) { const __half h = __float2half(0.01f + 0.02f * r.uniform()); memcpy(&s, &h, 2); }
        for (auto &x : code) x = (int8_t)((int)(r.next() % 255u) - 127);
        ds4_gpu_tensor *t_lat = upload(lat.data(), lat.size() * 4u, "latent");
        ds4_gpu_tensor *t_gate = upload(gate.data(), gate.size() * 4u, "gate");
        ds4_gpu_tensor *t_scale = upload(scale.data(), scale.size() * 2u, "scale");
        ds4_gpu_tensor *t_code = upload(code.data(), code.size(), "code");
        const size_t out_bytes = (size_t)rows * heads * 128u * 4u;
        ds4_gpu_tensor *t_oa = ds4_gpu_tensor_alloc(out_bytes);
        ds4_gpu_tensor *t_ob = ds4_gpu_tensor_alloc(out_bytes);
        const bool ok_a =
            ds4_gpu_dots3_value_project_planes_tensor(t_oa, t_lat, ds4_gpu_tensor_ptr(t_scale), ds4_gpu_tensor_ptr(t_code), nullptr, rows, heads, latent) &&
            ds4_gpu_dots3_gate_mul_tensor(t_oa, t_gate, rows, heads, 128u);
        const bool ok_b =
            ds4_gpu_dots3_value_project_planes_tensor(t_ob, t_lat, ds4_gpu_tensor_ptr(t_scale), ds4_gpu_tensor_ptr(t_code), t_gate, rows, heads, latent);
        check(cudaDeviceSynchronize(), "gated value sync");
        results.push_back({width == 0 ? "value-gate-decode" : "value-gate-hmma",
            ok_a && ok_b && read_bytes(t_oa, out_bytes) == read_bytes(t_ob, out_bytes)});
        for (ds4_gpu_tensor *x : {t_lat, t_gate, t_scale, t_code, t_oa, t_ob}) ds4_gpu_tensor_free(x);
    }

    /* (e) FFN residual in one pass vs add + residual add. */
    {
        const uint64_t n = 7u * 5120u;
        std::vector<float> x(n), a(n), b(n);
        for (auto &v : x) v = r.normal();
        for (auto &v : a) v = r.normal();
        for (auto &v : b) v = r.normal();
        ds4_gpu_tensor *t_xa = upload(x.data(), n * 4u, "x a");
        ds4_gpu_tensor *t_xb = upload(x.data(), n * 4u, "x b");
        ds4_gpu_tensor *t_a = upload(a.data(), n * 4u, "routed");
        ds4_gpu_tensor *t_b = upload(b.data(), n * 4u, "shared");
        ds4_gpu_tensor *t_sum = ds4_gpu_tensor_alloc(n * 4u);
        const bool ok_a = ds4_gpu_add_tensor(t_sum, t_a, t_b, (uint32_t)n) &&
                          ds4_gpu_exaone_add_tensor(t_xa, t_sum, n);
        const bool ok_b = ds4_gpu_dots3_ffn_residual_tensor(t_xb, t_a, t_b, n);
        check(cudaDeviceSynchronize(), "ffn residual sync");
        results.push_back({"ffn-residual", ok_a && ok_b &&
            read_bytes(t_xa, n * 4u) == read_bytes(t_xb, n * 4u)});
        for (ds4_gpu_tensor *t2 : {t_xa, t_xb, t_a, t_b, t_sum}) ds4_gpu_tensor_free(t2);
    }
    ds4_gpu_tensor_free(t_inv_swa);
    ds4_gpu_tensor_free(t_inv_full);
    return results;
}

}  // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    if (!ds4_gpu_init()) { fprintf(stderr, "CUDA init failed\n"); return 1; }
    const bool profile = std::getenv("DS4_DOTS3_PROFILE_ATTN") != nullptr;
    std::vector<attn_case> cases;
    if (profile) {
        cases = {
            {"full-dsa-4k", 128u, 512u, 128u, 4096u, 4096u, 8257u, 0u, 2048u, true, false, false},
            {"swa-4k", 64u, 1024u, 192u, 4096u, 4096u, 513u + 4096u, 513u, 0u, false, false, false},
            {"dec-full-dsa", 128u, 512u, 128u, 1u, 8000u, 8257u, 0u, 2048u, true, false, true},
            {"dec-swa", 64u, 1024u, 192u, 1u, 8000u, 513u + 4096u, 513u, 0u, false, false, true},
        };
    } else {
        cases = {
            {"full-dsa", 128u, 512u, 128u, 64u, 3000u, 4096u, 0u, 2048u, true, true, false},
            {"full-dsa-short", 128u, 512u, 128u, 40u, 0u, 4096u, 0u, 2048u, true, true, false},
            {"full-causal", 128u, 512u, 128u, 48u, 1000u, 2048u, 0u, 0u, false, false, false},
            {"swa-ring", 64u, 1024u, 192u, 40u, 4600u, 513u + 4096u, 513u, 0u, false, false, false},
            {"swa-open", 64u, 1024u, 192u, 40u, 0u, 513u + 4096u, 513u, 0u, false, false, false},
            {"dec-full-dsa", 128u, 512u, 128u, 1u, 3000u, 4096u, 0u, 2048u, true, true, true},
            {"dec-full-short", 128u, 512u, 128u, 2u, 30u, 4096u, 0u, 2048u, true, true, true},
            {"dec-full-causal", 128u, 512u, 128u, 1u, 1500u, 2048u, 0u, 0u, false, false, true},
            {"dec-swa-ring", 64u, 1024u, 192u, 1u, 4620u, 513u + 4096u, 513u, 0u, false, false, true},
            {"dec-swa-open", 64u, 1024u, 192u, 2u, 100u, 513u + 4096u, 513u, 0u, false, false, true},
        };
    }
    int failures = 0;
    for (const auto &c : cases) {
        const attn_result r = run_case(c, profile, 8);
        const bool ok = r.finite && r.rel_rms <= (c.split ? 1.0e-4 : 1.0e-2);
        printf("dots3 attention %-14s rows=%u heads=%u latent=%u: rel_rms=%.3e max_abs=%.3e %s\n",
               c.name, c.rows, c.heads, c.latent, r.rel_rms, r.max_abs, ok ? "OK" : "FAIL");
        if (!ok) failures++;
    }
    std::vector<vp_case> vp_cases;
    if (profile) {
        vp_cases = {{"full-4k", 128u, 512u, 4096u}, {"swa-4k", 64u, 1024u, 4096u},
                    {"dec-full", 128u, 512u, 1u}, {"dec-swa", 64u, 1024u, 1u}};
    } else {
        vp_cases = {{"full", 4u, 512u, 100u}, {"swa", 4u, 1024u, 67u}, {"full-16", 2u, 512u, 16u},
                    {"dec-full", 8u, 512u, 1u}, {"dec-swa", 8u, 1024u, 3u}};
    }
    for (const auto &c : vp_cases) {
        const vp_result r = run_vp_case(c, profile, 8);
        const bool ok = r.finite && r.rel_rms <= 1.0e-2;
        if (profile) {
            printf("dots3 value profile %-10s rows=%u heads=%u latent=%u: %.3f ms %s\n",
                   c.name, c.rows, c.heads, c.latent, r.ms, r.finite ? "finite" : "NAN");
        } else {
            printf("dots3 value %-10s rows=%u heads=%u latent=%u: rel_rms=%.3e max_abs=%.3e %s\n",
                   c.name, c.rows, c.heads, c.latent, r.rel_rms, r.max_abs, ok ? "OK" : "FAIL");
        }
        if (!ok) failures++;
    }
    std::vector<ab_case> ab_cases;
    if (profile) {
        ab_cases = {{"full-4k", 128u, 512u, 128u, 4096u}, {"swa-4k", 64u, 1024u, 192u, 4096u}};
    } else {
        ab_cases = {{"full", 4u, 512u, 128u, 100u}, {"swa", 4u, 1024u, 192u, 67u}, {"full-16", 2u, 512u, 128u, 16u}};
    }
    for (const auto &c : ab_cases) {
        const vp_result r = run_ab_case(c, profile, 8);
        const bool ok = r.finite && r.rel_rms <= 1.0e-2;
        if (profile) {
            printf("dots3 absorb profile %-10s rows=%u heads=%u latent=%u nope=%u: %.3f ms %s\n",
                   c.name, c.rows, c.heads, c.latent, c.nope, r.ms, r.finite ? "finite" : "NAN");
        } else {
            printf("dots3 absorb %-10s rows=%u heads=%u latent=%u nope=%u: rel_rms=%.3e max_abs=%.3e %s\n",
                   c.name, c.rows, c.heads, c.latent, c.nope, r.rel_rms, r.max_abs, ok ? "OK" : "FAIL");
        }
        if (!ok) failures++;
    }
    if (!profile) {
        for (const fused_result &r : run_fused_cases()) {
            printf("dots3 fused %-18s %s\n", r.name, r.same ? "bit-identical" : "MISMATCH");
            if (!r.same) failures++;
        }
    }
    ds4_gpu_cleanup();
    return failures ? 1 : 0;
}
