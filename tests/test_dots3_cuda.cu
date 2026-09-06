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

    auto launch = [&](ds4_gpu_tensor *out, bool scalar) {
        if (scalar) setenv("DS4_DOTS3_ATTN_NO_HMMA", "1", 1);
        else unsetenv("DS4_DOTS3_ATTN_NO_HMMA");
        if (!ds4_gpu_dots3_latent_attention_tensor(
                out, t_q, t_qa, t_lat, t_kpe, t_sel,
                c.selection ? c.sel_stride : 0u, c.rows, c.pos0, c.cache_cap,
                c.window, c.heads, c.latent, c.nope, 64u, scale)) {
            fprintf(stderr, "%s: latent attention launch failed (%s)\n",
                    c.name, scalar ? "scalar" : "hmma");
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
            const int n = scalar ? 2 : reps;
            check(cudaEventRecord(e0), "t0");
            for (int i = 0; i < n; i++) launch(scalar ? t_ref : t_out, scalar);
            check(cudaEventRecord(e1), "t1");
            check(cudaEventSynchronize(e1), "timed sync");
            check(cudaEventElapsedTime(&ms[which], e0, e1), "elapsed");
            ms[which] /= (float)n;
        }
        printf("dots3 attention profile %-14s rows=%u heads=%u latent=%u keys=%u: "
               "scalar %.3f ms, hmma %.3f ms (%.2fx)\n",
               c.name, c.rows, c.heads, c.latent,
               c.selection ? c.sel_stride : c.window, ms[0], ms[1],
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
                c.rows, c.heads, c.latent)) {
            fprintf(stderr, "%s: value projection launch failed\n", c.name);
            std::exit(1);
        }
    };
    launch();
    check(cudaDeviceSynchronize(), "vp sync");
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
    return {den > 0.0 ? std::sqrt(num / den) : 0.0, max_abs, finite, ms};
}

}  // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    if (!ds4_gpu_init()) { fprintf(stderr, "CUDA init failed\n"); return 1; }
    const bool profile = std::getenv("DS4_DOTS3_PROFILE_ATTN") != nullptr;
    std::vector<attn_case> cases;
    if (profile) {
        cases = {
            {"full-dsa-4k", 128u, 512u, 128u, 4096u, 4096u, 8257u, 0u, 2048u, true, false},
            {"swa-4k", 64u, 1024u, 192u, 4096u, 4096u, 513u + 4096u, 513u, 0u, false, false},
        };
    } else {
        cases = {
            {"full-dsa", 128u, 512u, 128u, 64u, 3000u, 4096u, 0u, 2048u, true, true},
            {"full-dsa-short", 128u, 512u, 128u, 40u, 0u, 4096u, 0u, 2048u, true, true},
            {"full-causal", 128u, 512u, 128u, 48u, 1000u, 2048u, 0u, 0u, false, false},
            {"swa-ring", 64u, 1024u, 192u, 40u, 4600u, 513u + 4096u, 513u, 0u, false, false},
            {"swa-open", 64u, 1024u, 192u, 40u, 0u, 513u + 4096u, 513u, 0u, false, false},
        };
    }
    int failures = 0;
    for (const auto &c : cases) {
        const attn_result r = run_case(c, profile, 8);
        const bool ok = r.finite && r.rel_rms <= 1.0e-2;
        printf("dots3 attention %-14s rows=%u heads=%u latent=%u: rel_rms=%.3e max_abs=%.3e %s\n",
               c.name, c.rows, c.heads, c.latent, r.rel_rms, r.max_abs, ok ? "OK" : "FAIL");
        if (!ok) failures++;
    }
    std::vector<vp_case> vp_cases;
    if (profile) {
        vp_cases = {{"full-4k", 128u, 512u, 4096u}, {"swa-4k", 64u, 1024u, 4096u}};
    } else {
        vp_cases = {{"full", 4u, 512u, 100u}, {"swa", 4u, 1024u, 67u}, {"full-16", 2u, 512u, 16u}};
    }
    for (const auto &c : vp_cases) {
        const vp_result r = run_vp_case(c, profile, 8);
        const bool ok = r.finite && r.rel_rms <= 1.0e-2;
        if (profile) {
            printf("dots3 value profile %-10s rows=%u heads=%u latent=%u: hmma %.3f ms %s\n",
                   c.name, c.rows, c.heads, c.latent, r.ms, r.finite ? "finite" : "NAN");
        } else {
            printf("dots3 value %-10s rows=%u heads=%u latent=%u: rel_rms=%.3e max_abs=%.3e %s\n",
                   c.name, c.rows, c.heads, c.latent, r.rel_rms, r.max_abs, ok ? "OK" : "FAIL");
        }
        if (!ok) failures++;
    }
    ds4_gpu_cleanup();
    return failures ? 1 : 0;
}
