/* Ling MLA prefill: default HMMA vs Motif kill vs rejected tile. */
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

enum { HEADS = 32, ROWS = 32, LAT = 512, ROPE = 64, NOPE = 128, KEY = NOPE + ROPE };
enum L3vMlaPath { L3V_MLA_MOTIF, L3V_MLA_DEFAULT, L3V_MLA_TILE };

static const double kSamePathRelRms = 1.0e-6;
static const double kHmmaRelRms = 1.0e-2;
static const double kTileRelRms = 2.0e-5;

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static void apply_path(L3vMlaPath path) {
    unsetenv("DS4_LING3VL_MLA_TILE");
    unsetenv("DS4_LING3VL_NO_MLA_HMMA");
    unsetenv("DS4_DOTS3_ATTN_NO_HMMA");
    switch (path) {
    case L3V_MLA_MOTIF:
        CHECK(!setenv("DS4_LING3VL_NO_MLA_HMMA", "1", 1));
        break;
    case L3V_MLA_TILE:
        CHECK(!setenv("DS4_LING3VL_MLA_TILE", "1", 1));
        break;
    case L3V_MLA_DEFAULT:
        break;
    }
}

static double rel_rms(const std::vector<float> &a, const std::vector<float> &b) {
    CHECK(a.size() == b.size());
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return sqrt(num) / (sqrt(den) + 1e-12);
}

static void fill_host(std::vector<float> &q, std::vector<float> &qa,
                      std::vector<uint16_t> &lat, std::vector<uint16_t> &pe) {
    for (size_t i = 0; i < q.size(); i++) {
        q[i] = ((int)(i * 17 % 63) - 31) / 19.0f;
    }
    for (size_t i = 0; i < qa.size(); i++) {
        qa[i] = ((int)(i * 11 % 41) - 20) / 13.0f;
    }
    for (size_t i = 0; i < lat.size(); i++) {
        lat[i] = f32_to_bf16(((int)(i * 7 % 29) - 14) / 9.0f);
    }
    for (size_t i = 0; i < pe.size(); i++) {
        pe[i] = f32_to_bf16(((int)(i * 5 % 17) - 8) / 11.0f);
    }
}

int main(void) {
    CHECK(ds4_gpu_init());
    const uint32_t cap = ROWS;
    const size_t q_bytes = (size_t)ROWS * HEADS * KEY * sizeof(float);
    const size_t abs_bytes = (size_t)ROWS * HEADS * LAT * sizeof(float);
    const size_t lat_bytes = (size_t)cap * LAT * sizeof(uint16_t);
    const size_t pe_bytes = (size_t)cap * ROPE * sizeof(uint16_t);
    std::vector<float> q(q_bytes / sizeof(float)), qa(abs_bytes / sizeof(float));
    std::vector<uint16_t> lat(lat_bytes / sizeof(uint16_t)), pe(pe_bytes / sizeof(uint16_t));
    fill_host(q, qa, lat, pe);

    ds4_gpu_tensor *dq = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *dqa = ds4_gpu_tensor_alloc(abs_bytes);
    ds4_gpu_tensor *dlat = ds4_gpu_tensor_alloc(lat_bytes);
    ds4_gpu_tensor *dpe = ds4_gpu_tensor_alloc(pe_bytes);
    ds4_gpu_tensor *out_motif = ds4_gpu_tensor_alloc(abs_bytes);
    ds4_gpu_tensor *out_default = ds4_gpu_tensor_alloc(abs_bytes);
    ds4_gpu_tensor *out_dots3 = ds4_gpu_tensor_alloc(abs_bytes);
    ds4_gpu_tensor *out_tile = ds4_gpu_tensor_alloc(abs_bytes);
    ds4_gpu_tensor *out_decode_a = ds4_gpu_tensor_alloc(HEADS * LAT * sizeof(float));
    ds4_gpu_tensor *out_decode_b = ds4_gpu_tensor_alloc(HEADS * LAT * sizeof(float));
    CHECK(dq && dqa && dlat && dpe && out_motif && out_default && out_dots3 &&
          out_tile && out_decode_a && out_decode_b);
    CHECK(ds4_gpu_tensor_write(dq, 0, q.data(), q_bytes));
    CHECK(ds4_gpu_tensor_write(dqa, 0, qa.data(), abs_bytes));
    CHECK(ds4_gpu_tensor_write(dlat, 0, lat.data(), lat_bytes));
    CHECK(ds4_gpu_tensor_write(dpe, 0, pe.data(), pe_bytes));

    const float scale = 1.0f / sqrtf((float)KEY);

    apply_path(L3V_MLA_MOTIF);
    CHECK(ds4_gpu_ling3vl_latent_attn(
        out_motif, dq, dqa, dlat, dpe, ROWS, 0u, cap, 0u, HEADS, LAT,
        NOPE, ROPE, scale));

    apply_path(L3V_MLA_DEFAULT);
    CHECK(ds4_gpu_ling3vl_latent_attn(
        out_default, dq, dqa, dlat, dpe, ROWS, 0u, cap, 0u, HEADS, LAT,
        NOPE, ROPE, scale));
    CHECK(ds4_gpu_dots3_latent_attention_tensor(
        out_dots3, dq, dqa, dlat, dpe, NULL, 0u, ROWS, 0u, cap, 0u, HEADS, LAT,
        NOPE, ROPE, scale));

    apply_path(L3V_MLA_TILE);
    CHECK(ds4_gpu_ling3vl_latent_attn(
        out_tile, dq, dqa, dlat, dpe, ROWS, 0u, cap, 0u, HEADS, LAT,
        NOPE, ROPE, scale));

    /* n=1 decode stays on Motif whether or not the prefill HMMA kill is set. */
    apply_path(L3V_MLA_DEFAULT);
    CHECK(ds4_gpu_ling3vl_latent_attn(
        out_decode_a, dq, dqa, dlat, dpe, 1u, 0u, cap, 0u, HEADS, LAT,
        NOPE, ROPE, scale));
    apply_path(L3V_MLA_MOTIF);
    CHECK(ds4_gpu_ling3vl_latent_attn(
        out_decode_b, dq, dqa, dlat, dpe, 1u, 0u, cap, 0u, HEADS, LAT,
        NOPE, ROPE, scale));
    apply_path(L3V_MLA_DEFAULT);

    std::vector<float> motif(qa.size()), def(qa.size()), dots3(qa.size()),
        tile(qa.size());
    std::vector<float> dec_a((size_t)HEADS * LAT), dec_b((size_t)HEADS * LAT);
    CHECK(ds4_gpu_tensor_read(out_motif, 0, motif.data(), abs_bytes));
    CHECK(ds4_gpu_tensor_read(out_default, 0, def.data(), abs_bytes));
    CHECK(ds4_gpu_tensor_read(out_dots3, 0, dots3.data(), abs_bytes));
    CHECK(ds4_gpu_tensor_read(out_tile, 0, tile.data(), abs_bytes));
    CHECK(ds4_gpu_tensor_read(out_decode_a, 0, dec_a.data(), dec_a.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(out_decode_b, 0, dec_b.data(), dec_b.size() * sizeof(float)));

    const double hmma_vs_dots3 = rel_rms(def, dots3);
    const double hmma_vs_motif = rel_rms(def, motif);
    const double tile_vs_motif = rel_rms(tile, motif);
    const double decode_vs_motif = rel_rms(dec_a, dec_b);
    CHECK(hmma_vs_dots3 < kSamePathRelRms);
    CHECK(hmma_vs_motif < kHmmaRelRms);
    CHECK(tile_vs_motif < kTileRelRms);
    CHECK(decode_vs_motif < kSamePathRelRms);

    ds4_gpu_tensor_free(dq);
    ds4_gpu_tensor_free(dqa);
    ds4_gpu_tensor_free(dlat);
    ds4_gpu_tensor_free(dpe);
    ds4_gpu_tensor_free(out_motif);
    ds4_gpu_tensor_free(out_default);
    ds4_gpu_tensor_free(out_dots3);
    ds4_gpu_tensor_free(out_tile);
    ds4_gpu_tensor_free(out_decode_a);
    ds4_gpu_tensor_free(out_decode_b);
    ds4_gpu_cleanup();
    printf("Ling MLA prefill HMMA: rel-rms %.3g vs dots3 FATTN, %.3g vs Motif\n",
           hmma_vs_dots3, hmma_vs_motif);
    printf("Ling MLA prefill tile: rel-rms %.3g vs Motif\n", tile_vs_motif);
    printf("Ling MLA decode n=1: rel-rms %.3g vs Motif kill\n", decode_vs_motif);
    return 0;
}
