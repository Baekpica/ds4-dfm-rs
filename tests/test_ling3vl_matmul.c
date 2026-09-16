/* Ling n=1 BF16 projections use the row-stable warp GEMV. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { OFFSET = 4096, K = 2560, M = 64,
       MAP_BYTES = OFFSET + (int)(M * K * sizeof(uint16_t)) };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static uint16_t bf16_from_f32(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

int main(void) {
    CHECK(ds4_gpu_init());
    void *map = NULL;
    CHECK(!posix_memalign(&map, OFFSET, MAP_BYTES));
    memset(map, 0, MAP_BYTES);
    uint16_t *w = (uint16_t *)((char *)map + OFFSET);
    for (unsigned i = 0; i < M * K; i++) {
        w[i] = bf16_from_f32(((int)(i % 17) - 8) / 11.0f);
    }
    CHECK(ds4_gpu_set_model_map(map, MAP_BYTES));

    float x[K], cublas[M], vec[M], restored[M];
    for (unsigned i = 0; i < K; i++) {
        x[i] = ((int)(i * 13 % 29) - 14) / 7.0f;
    }
    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(cublas));
    CHECK(dx && out);
    CHECK(ds4_gpu_tensor_write(dx, 0, x, sizeof(x)));

    CHECK(!setenv("DS4_LING3VL_NO_BF16_VEC", "1", 1));
    CHECK(ds4_gpu_ling3vl_matmul_bf16(
        out, map, MAP_BYTES, OFFSET, K, M, dx, 1u));
    CHECK(ds4_gpu_tensor_read(out, 0, cublas, sizeof(cublas)));
    CHECK(ds4_gpu_matmul_bf16_tensor(
        out, map, MAP_BYTES, OFFSET, K, M, dx, 1u));
    CHECK(ds4_gpu_tensor_read(out, 0, restored, sizeof(restored)));
    CHECK(!memcmp(cublas, restored, sizeof(cublas)));

    CHECK(!unsetenv("DS4_LING3VL_NO_BF16_VEC"));
    CHECK(ds4_gpu_ling3vl_matmul_bf16(
        out, map, MAP_BYTES, OFFSET, K, M, dx, 1u));
    CHECK(ds4_gpu_tensor_read(out, 0, vec, sizeof(vec)));
    CHECK(ds4_gpu_matmul_bf16_stable_rows_tensor(
        out, map, MAP_BYTES, OFFSET, K, M, dx, 1u));
    CHECK(ds4_gpu_tensor_read(out, 0, restored, sizeof(restored)));
    CHECK(!memcmp(vec, restored, sizeof(vec)));

    float nodual[M];
    CHECK(!setenv("DS4_LING3VL_NO_GEMV_XREG", "1", 1));
    CHECK(ds4_gpu_ling3vl_matmul_bf16(
        out, map, MAP_BYTES, OFFSET, K, M, dx, 1u));
    CHECK(ds4_gpu_tensor_read(out, 0, nodual, sizeof(nodual)));
    CHECK(!memcmp(vec, nodual, sizeof(vec)));
    CHECK(!unsetenv("DS4_LING3VL_NO_GEMV_XREG"));

    float odd[3], odd_ref[3];
    CHECK(ds4_gpu_ling3vl_matmul_bf16(
        out, map, MAP_BYTES, OFFSET, K, 3u, dx, 1u));
    CHECK(ds4_gpu_tensor_read(out, 0, odd, sizeof(odd)));
    CHECK(ds4_gpu_matmul_bf16_stable_rows_tensor(
        out, map, MAP_BYTES, OFFSET, K, 3u, dx, 1u));
    CHECK(ds4_gpu_tensor_read(out, 0, odd_ref, sizeof(odd_ref)));
    CHECK(!memcmp(odd, odd_ref, sizeof(odd)));

    double num = 0.0, den = 0.0;
    for (unsigned i = 0; i < M; i++) {
        const double d = (double)vec[i] - (double)cublas[i];
        num += d * d;
        den += (double)cublas[i] * (double)cublas[i];
        CHECK(isfinite(vec[i]) && isfinite(cublas[i]));
    }
    const double rel = sqrt(num) / (sqrt(den) + 1e-12);
    CHECK(rel < 1e-3);

    ds4_gpu_tensor_free(dx);
    ds4_gpu_tensor_free(out);

    /* Prefill: one convert + ds4_gpu_matmul_bf16_input_tensor matches the
     * per-call convert inside ds4_gpu_matmul_bf16_tensor. */
    enum { N = 8 };
    const size_t x_bytes = (size_t)N * K * sizeof(float);
    const size_t y_bytes = (size_t)N * M * sizeof(float);
    const size_t bf_bytes = (size_t)N * K * sizeof(uint16_t);
    float *x_n = (float *)malloc(x_bytes);
    float *y_once = (float *)malloc(y_bytes);
    float *y_each = (float *)malloc(y_bytes);
    CHECK(x_n && y_once && y_each);
    for (unsigned i = 0; i < N * K; i++) {
        x_n[i] = ((int)(i * 13 % 29) - 14) / 7.0f;
    }
    ds4_gpu_tensor *dxn = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *dbf = ds4_gpu_tensor_alloc(bf_bytes);
    ds4_gpu_tensor *outn = ds4_gpu_tensor_alloc(y_bytes);
    CHECK(dxn && dbf && outn);
    CHECK(ds4_gpu_tensor_write(dxn, 0, x_n, x_bytes));
    CHECK(ds4_gpu_f32_to_bf16(dbf, dxn, (uint64_t)N * K));
    CHECK(ds4_gpu_matmul_bf16_input_tensor(
        outn, map, MAP_BYTES, OFFSET, K, M, dbf, N));
    CHECK(ds4_gpu_tensor_read(outn, 0, y_once, y_bytes));
    CHECK(ds4_gpu_matmul_bf16_tensor(
        outn, map, MAP_BYTES, OFFSET, K, M, dxn, N));
    CHECK(ds4_gpu_tensor_read(outn, 0, y_each, y_bytes));
    CHECK(!memcmp(y_once, y_each, y_bytes));

    ds4_gpu_tensor_free(dxn);
    ds4_gpu_tensor_free(dbf);
    ds4_gpu_tensor_free(outn);
    free(x_n);
    free(y_once);
    free(y_each);
    ds4_gpu_cleanup();
    free(map);
    printf("Ling BF16 n=1 GEMV: kill-switch matches cuBLAS, vec matches stable rows, rel-rms %.3g\n",
           rel);
    printf("Ling BF16 convert-reuse: packed input matches per-call convert\n");

    /* Decode pair: one convert + fused warp GEMVs vs two singles. */
    enum { M2 = 32 };
    const uint64_t off1 = OFFSET;
    const uint64_t off2 = OFFSET + (uint64_t)M * K * sizeof(uint16_t);
    const size_t map2 = (size_t)off2 + M2 * K * sizeof(uint16_t);
    void *map_pair = NULL;
    CHECK(!posix_memalign(&map_pair, OFFSET, map2));
    memset(map_pair, 0, map2);
    uint16_t *w0 = (uint16_t *)((char *)map_pair + off1);
    uint16_t *w1 = (uint16_t *)((char *)map_pair + off2);
    for (unsigned i = 0; i < M * K; i++) {
        w0[i] = bf16_from_f32(((int)(i % 17) - 8) / 11.0f);
    }
    for (unsigned i = 0; i < M2 * K; i++) {
        w1[i] = bf16_from_f32(((int)(i % 13) - 6) / 9.0f);
    }
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(map_pair, map2));
    ds4_gpu_tensor *dx1 = ds4_gpu_tensor_alloc(K * sizeof(float));
    ds4_gpu_tensor *o0 = ds4_gpu_tensor_alloc(M * sizeof(float));
    ds4_gpu_tensor *o1 = ds4_gpu_tensor_alloc(M2 * sizeof(float));
    ds4_gpu_tensor *r0 = ds4_gpu_tensor_alloc(M * sizeof(float));
    ds4_gpu_tensor *r1 = ds4_gpu_tensor_alloc(M2 * sizeof(float));
    CHECK(dx1 && o0 && o1 && r0 && r1);
    CHECK(ds4_gpu_tensor_write(dx1, 0, x, K * sizeof(float)));
    CHECK(ds4_gpu_matmul_bf16_stable_rows_pair_tensor(
        o0, o1, map_pair, map2, off1, off2, K, M, M2, dx1, 1u));
    CHECK(ds4_gpu_matmul_bf16_stable_rows_tensor(
        r0, map_pair, map2, off1, K, M, dx1, 1u));
    CHECK(ds4_gpu_matmul_bf16_stable_rows_tensor(
        r1, map_pair, map2, off2, K, M2, dx1, 1u));
    float p0[M], p1[M2], s0[M], s1[M2];
    CHECK(ds4_gpu_tensor_read(o0, 0, p0, sizeof(p0)));
    CHECK(ds4_gpu_tensor_read(o1, 0, p1, sizeof(p1)));
    CHECK(ds4_gpu_tensor_read(r0, 0, s0, sizeof(s0)));
    CHECK(ds4_gpu_tensor_read(r1, 0, s1, sizeof(s1)));
    CHECK(!memcmp(p0, s0, sizeof(p0)));
    CHECK(!memcmp(p1, s1, sizeof(p1)));

    float sh0[M], sh1[M2];
    CHECK(ds4_gpu_ling3vl_gemv_pair(
        o0, o1, map_pair, map2, off1, off2, K, M, M2, dx1));
    CHECK(ds4_gpu_tensor_read(o0, 0, sh0, sizeof(sh0)));
    CHECK(ds4_gpu_tensor_read(o1, 0, sh1, sizeof(sh1)));
    CHECK(!memcmp(sh0, s0, sizeof(sh0)));
    CHECK(!memcmp(sh1, s1, sizeof(sh1)));
    CHECK(!setenv("DS4_LING3VL_NO_GEMV_XREG", "1", 1));
    CHECK(ds4_gpu_ling3vl_gemv_pair(
        o0, o1, map_pair, map2, off1, off2, K, M, M2, dx1));
    CHECK(ds4_gpu_tensor_read(o0, 0, p0, sizeof(p0)));
    CHECK(ds4_gpu_tensor_read(o1, 0, p1, sizeof(p1)));
    CHECK(!memcmp(p0, s0, sizeof(p0)));
    CHECK(!memcmp(p1, s1, sizeof(p1)));
    CHECK(!unsetenv("DS4_LING3VL_NO_GEMV_XREG"));

    ds4_gpu_tensor_free(dx1);
    ds4_gpu_tensor_free(o0);
    ds4_gpu_tensor_free(o1);
    ds4_gpu_tensor_free(r0);
    ds4_gpu_tensor_free(r1);
    ds4_gpu_cleanup();
    free(map_pair);
    printf("Ling BF16 pair GEMV: matches two singles\n");
    printf("Ling BF16 GEMV xreg: matches stable-rows (kill DS4_LING3VL_NO_GEMV_XREG)\n");

    enum { F32_OFF = 4096, F32_MAP = F32_OFF + (int)(M * K * sizeof(float)) };
    void *fmap = NULL;
    CHECK(!posix_memalign(&fmap, F32_OFF, F32_MAP));
    float *fw = (float *)((char *)fmap + F32_OFF);
    for (unsigned i = 0; i < M * K; i++) {
        fw[i] = ((int)(i % 17) - 8) / 11.0f;
    }
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(fmap, F32_MAP));
    ds4_gpu_tensor *fx = ds4_gpu_tensor_alloc(K * sizeof(float));
    ds4_gpu_tensor *fo = ds4_gpu_tensor_alloc(M * sizeof(float));
    CHECK(fx && fo);
    CHECK(ds4_gpu_tensor_write(fx, 0, x, K * sizeof(float)));
    CHECK(ds4_gpu_ling3vl_matmul_f32(
        fo, fmap, F32_MAP, F32_OFF, K, M, fx, 1u));
    float fwarp[M], fref[M];
    CHECK(ds4_gpu_tensor_read(fo, 0, fwarp, sizeof(fwarp)));
    CHECK(ds4_gpu_matmul_f32_tensor(
        fo, fmap, F32_MAP, F32_OFF, K, M, fx, 1u));
    CHECK(ds4_gpu_tensor_read(fo, 0, fref, sizeof(fref)));
    double fnum = 0.0, fden = 0.0;
    for (unsigned i = 0; i < M; i++) {
        const double d = (double)fwarp[i] - (double)fref[i];
        fnum += d * d;
        fden += (double)fref[i] * (double)fref[i];
        CHECK(isfinite(fwarp[i]) && isfinite(fref[i]));
    }
    CHECK(sqrt(fnum) / (sqrt(fden) + 1e-12) < 1e-5);
    CHECK(!setenv("DS4_LING3VL_NO_F32_VEC", "1", 1));
    CHECK(!ds4_gpu_ling3vl_matmul_f32(
        fo, fmap, F32_MAP, F32_OFF, K, M, fx, 1u));
    CHECK(!unsetenv("DS4_LING3VL_NO_F32_VEC"));
    ds4_gpu_tensor_free(fx);
    ds4_gpu_tensor_free(fo);
    ds4_gpu_cleanup();
    free(fmap);
    printf("Ling F32 n=1 warp GEMV: rel-rms vs block GEMV < 1e-5 (kill DS4_LING3VL_NO_F32_VEC)\n");
    return 0;
}
