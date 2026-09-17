/* Ling expanded-MLA prefill (per-head K/V + range attention, merged across
 * three key segments), FP32 scratch on Motif's kernel and BF16 scratch on
 * the Ling kernel, against the absorbed path and a double reference.  The
 * range kernels round Q/K/V/P to BF16, so their bound is looser than the
 * FP16 HMMA; a layout or merge bug shows up as O(1), not 1e-2. */
#include "ds4_gpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

enum { HEADS = 32, LAT = 512, ROPE = 64, NOPE = 128, KEY = NOPE + ROPE,
       VALUE = 128, POS0 = 96, ROWS = 80, CTX = POS0 + ROWS, SEG = 64 };

static const double kExpandedRelRms = 2.0e-2;
static const double kAbsorbedRelRms = 2.0e-3;

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static float pseudo(size_t i, unsigned salt) {
    return ((int)((i * (17u + salt)) % 61) - 30) / 23.0f;
}

static double rel_rms(const std::vector<float> &a, const std::vector<double> &b) {
    CHECK(a.size() == b.size());
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        CHECK(std::isfinite(a[i]));
        const double d = (double)a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return sqrt(num) / (sqrt(den) + 1e-12);
}

/* Non-absorbed attention in double on the BF16-rounded inputs. */
static void reference(std::vector<double> &out, const std::vector<float> &q,
                      const std::vector<uint16_t> &lat, const std::vector<uint16_t> &pe,
                      const std::vector<uint16_t> &k_b, const std::vector<uint16_t> &v_b,
                      float scale) {
    std::vector<double> k((size_t)CTX * HEADS * KEY), v((size_t)CTX * HEADS * VALUE);
    for (size_t t = 0; t < CTX; t++) {
        for (size_t h = 0; h < HEADS; h++) {
            for (size_t d = 0; d < NOPE; d++) {
                double s = 0.0;
                for (size_t j = 0; j < LAT; j++) {
                    s += (double)bf16_to_f32(lat[t * LAT + j]) *
                         bf16_to_f32(k_b[(h * LAT + j) * NOPE + d]);
                }
                k[(t * HEADS + h) * KEY + d] = s;
            }
            for (size_t c = 0; c < ROPE; c++) {
                k[(t * HEADS + h) * KEY + NOPE + c] = bf16_to_f32(pe[t * ROPE + c]);
            }
            for (size_t d = 0; d < VALUE; d++) {
                double s = 0.0;
                for (size_t j = 0; j < LAT; j++) {
                    s += (double)bf16_to_f32(lat[t * LAT + j]) *
                         bf16_to_f32(v_b[(h * VALUE + d) * LAT + j]);
                }
                v[(t * HEADS + h) * VALUE + d] = s;
            }
        }
    }
    std::vector<double> score(CTX);
    for (size_t r = 0; r < ROWS; r++) {
        const size_t qpos = POS0 + r;
        for (size_t h = 0; h < HEADS; h++) {
            const float *qh = &q[(r * HEADS + h) * KEY];
            double m = -INFINITY;
            for (size_t t = 0; t <= qpos; t++) {
                double s = 0.0;
                for (size_t d = 0; d < KEY; d++) s += (double)qh[d] * k[(t * HEADS + h) * KEY + d];
                score[t] = s * scale;
                m = fmax(m, score[t]);
            }
            double denom = 0.0;
            for (size_t t = 0; t <= qpos; t++) { score[t] = exp(score[t] - m); denom += score[t]; }
            for (size_t d = 0; d < VALUE; d++) {
                double s = 0.0;
                for (size_t t = 0; t <= qpos; t++) s += score[t] * v[(t * HEADS + h) * VALUE + d];
                out[(r * HEADS + h) * VALUE + d] = s / denom;
            }
        }
    }
}

int main(void) {
    CHECK(ds4_gpu_init());
    const size_t k_b_bytes = (size_t)HEADS * LAT * NOPE * sizeof(uint16_t);
    const size_t v_b_bytes = (size_t)HEADS * VALUE * LAT * sizeof(uint16_t);
    const size_t map_bytes = k_b_bytes + v_b_bytes;
    void *map = NULL;
    CHECK(!posix_memalign(&map, 4096, map_bytes));
    std::vector<uint16_t> k_b(k_b_bytes / 2), v_b(v_b_bytes / 2);
    for (size_t i = 0; i < k_b.size(); i++) k_b[i] = f32_to_bf16(pseudo(i, 1) * 0.25f);
    for (size_t i = 0; i < v_b.size(); i++) v_b[i] = f32_to_bf16(pseudo(i, 2) * 0.25f);
    memcpy(map, k_b.data(), k_b_bytes);
    memcpy((char *)map + k_b_bytes, v_b.data(), v_b_bytes);
    /* Production weights are device-resident; cuBLAS's TMA kernels fault on
     * the raw host map, so copy this one to the device the same way. */
    CHECK(!setenv("DS4_CUDA_COPY_MODEL", "1", 1));
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    CHECK(ds4_gpu_ling3vl_expand_ready(map, map_bytes, 0u, k_b_bytes, HEADS, LAT,
                                       NOPE, VALUE) == 1);
    /* A plain host buffer is refused.  The resolver pins it as a side
     * effect, so it stays allocated until the backend is torn down. */
    void *host_only = NULL;
    CHECK(!posix_memalign(&host_only, 4096, map_bytes));
    CHECK(ds4_gpu_ling3vl_expand_ready(host_only, map_bytes, 0u, k_b_bytes, HEADS,
                                       LAT, NOPE, VALUE) == 0);

    std::vector<float> q((size_t)ROWS * HEADS * KEY);
    std::vector<uint16_t> lat((size_t)CTX * LAT), pe((size_t)CTX * ROPE);
    for (size_t i = 0; i < q.size(); i++) q[i] = pseudo(i, 3);
    for (size_t i = 0; i < lat.size(); i++) lat[i] = f32_to_bf16(pseudo(i, 4));
    for (size_t i = 0; i < pe.size(); i++) pe[i] = f32_to_bf16(pseudo(i, 5));
    const float scale = 1.0f / sqrtf((float)KEY);

    const size_t out_bytes = (size_t)ROWS * HEADS * VALUE * sizeof(float);
    const size_t lse_bytes = (size_t)ROWS * HEADS * sizeof(float);
    ds4_gpu_tensor *dq = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
    ds4_gpu_tensor *dlat = ds4_gpu_tensor_alloc(lat.size() * sizeof(uint16_t));
    ds4_gpu_tensor *dpe = ds4_gpu_tensor_alloc(pe.size() * sizeof(uint16_t));
    /* Segment scratch holds the widest segment: the current chunk (ROWS). */
    ds4_gpu_tensor *k_full = ds4_gpu_tensor_alloc((size_t)ROWS * HEADS * KEY * sizeof(float));
    ds4_gpu_tensor *value = ds4_gpu_tensor_alloc((size_t)ROWS * HEADS * VALUE * sizeof(float));
    ds4_gpu_tensor *out_exp = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *out_tmp = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *lse = ds4_gpu_tensor_alloc(lse_bytes);
    ds4_gpu_tensor *lse_tmp = ds4_gpu_tensor_alloc(lse_bytes);
    ds4_gpu_tensor *qa = ds4_gpu_tensor_alloc((size_t)ROWS * HEADS * LAT * sizeof(float));
    ds4_gpu_tensor *latent_out = ds4_gpu_tensor_alloc((size_t)ROWS * HEADS * LAT * sizeof(float));
    ds4_gpu_tensor *out_abs = ds4_gpu_tensor_alloc(out_bytes);
    CHECK(dq && dlat && dpe && k_full && value && out_exp && out_tmp && lse &&
          lse_tmp && qa && latent_out && out_abs);
    CHECK(ds4_gpu_tensor_write(dq, 0, q.data(), q.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(dlat, 0, lat.data(), lat.size() * sizeof(uint16_t)));
    CHECK(ds4_gpu_tensor_write(dpe, 0, pe.data(), pe.size() * sizeof(uint16_t)));

    /* Expanded, FP32 then BF16 scratch: prefix segments [0,64) [64,96),
     * then the chunk [96,176), merged through the LSE. */
    const uint32_t segs[3][2] = {{0u, SEG}, {SEG, POS0 - SEG}, {POS0, ROWS}};
    std::vector<float> exp_h[2] = {std::vector<float>(out_bytes / 4),
                                   std::vector<float>(out_bytes / 4)};
    for (int bf16 = 0; bf16 < 2; bf16++) {
        for (int s = 0; s < 3; s++) {
            ds4_gpu_tensor *out = s == 0 ? out_exp : out_tmp;
            ds4_gpu_tensor *l = s == 0 ? lse : lse_tmp;
            CHECK(ds4_gpu_ling3vl_expand_kv(k_full, value, dlat, dpe, map, map_bytes,
                                            0u, k_b_bytes, segs[s][0], segs[s][1],
                                            HEADS, LAT, NOPE, ROPE, VALUE, bf16));
            CHECK(ds4_gpu_ling3vl_expanded_attn(
                out, l, dq, k_full, value, ROWS, POS0, segs[s][1], segs[s][0],
                HEADS, KEY, VALUE, scale, bf16));
            if (s) {
                CHECK(ds4_gpu_motif3_merge_attention_states_tensor(
                    out_exp, lse, out_tmp, lse_tmp, ROWS, HEADS, VALUE));
            }
        }
        CHECK(ds4_gpu_tensor_read(out_exp, 0, exp_h[bf16].data(), out_bytes));
    }

    /* Absorbed, on the Motif FP32 walk (the prefill HMMA kill). */
    CHECK(!setenv("DS4_LING3VL_NO_MLA_HMMA", "1", 1));
    CHECK(ds4_gpu_ling3vl_qk_absorb(qa, dq, map, map_bytes, 0u, ROWS, HEADS,
                                    KEY, NOPE, LAT));
    CHECK(ds4_gpu_ling3vl_latent_attn(latent_out, dq, qa, dlat, dpe, ROWS, POS0,
                                      CTX, 0u, HEADS, LAT, NOPE, ROPE, scale));
    CHECK(ds4_gpu_ling3vl_value_project(out_abs, latent_out, map, map_bytes,
                                        k_b_bytes, ROWS, HEADS, LAT, VALUE));

    std::vector<float> abs_h(out_bytes / 4);
    CHECK(ds4_gpu_tensor_read(out_abs, 0, abs_h.data(), out_bytes));
    std::vector<double> ref(out_bytes / 4);
    reference(ref, q, lat, pe, k_b, v_b, scale);
    const double f32_err = rel_rms(exp_h[0], ref);
    const double bf16_err = rel_rms(exp_h[1], ref);
    const double abs_err = rel_rms(abs_h, ref);
    printf("Ling MLA expanded prefill: rel-rms %.3g (FP32 K/V, Motif kernel), "
           "%.3g (BF16 K/V, Ling kernel) vs reference; absorbed %.3g\n",
           f32_err, bf16_err, abs_err);
    CHECK(f32_err < kExpandedRelRms);
    CHECK(bf16_err < kExpandedRelRms);
    CHECK(abs_err < kAbsorbedRelRms);

    ds4_gpu_tensor *all[] = {dq, dlat, dpe, k_full, value, out_exp, out_tmp,
                             lse, lse_tmp, qa, latent_out, out_abs};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
    ds4_gpu_cleanup();
    free(host_only);
    free(map);
    return 0;
}
