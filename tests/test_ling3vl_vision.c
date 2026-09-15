/* Ling vision tower against the real BF16 mmproj; no language weights.
 * usage: test_ling3vl_vision MMPROJ_GGUF [GRID_H GRID_W] */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "Ling vision FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

int main(int argc, char **argv) {
    CHECK(argc == 2 || argc == 4);
    const uint32_t grid_h = argc == 4 ? (uint32_t)atoi(argv[2]) : 14u;
    const uint32_t grid_w = argc == 4 ? (uint32_t)atoi(argv[3]) : 14u;
    CHECK(grid_h && grid_w && !(grid_h & 1u) && !(grid_w & 1u));
    g_ds4_shape = DS4_SHAPE_LING30_FLASH_VL;

    ds4_model m;
    model_open(&m, argv[1], true, false);
    ds4_ling3vl_vision_weights weights;
    ling3vl_vision_bind(&weights, &m);
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_aux_model_map_range(m.map, m.size, m.tensor_data_pos,
                                          m.size - m.tensor_data_pos));

    const uint32_t patches = grid_h * grid_w;
    ds4_ling3vl_vision v;
    CHECK(ling3vl_vision_alloc(&v, patches));

    /* A deterministic RGB ramp; only finiteness and shape are asserted here. */
    const uint32_t width = grid_w * L3V_VIT_PATCH, height = grid_h * L3V_VIT_PATCH;
    uint8_t *rgb = xmalloc((size_t)width * height * 3u);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t *p = rgb + ((size_t)y * width + x) * 3u;
            p[0] = (uint8_t)(x * 255u / (width - 1u));
            p[1] = (uint8_t)(y * 255u / (height - 1u));
            p[2] = (uint8_t)((x + y) & 0xffu);
        }
    }
    float *host = xmalloc((size_t)patches * L3V_VIT_PATCH_DIM * sizeof(float));
    int32_t *indices = xmalloc((size_t)patches * 4u * sizeof(int32_t));
    float *weights_bilinear = xmalloc((size_t)patches * 4u * sizeof(float));
    int32_t *rows = xmalloc((size_t)patches * sizeof(int32_t));
    int32_t *cols = xmalloc((size_t)patches * sizeof(int32_t));
    int32_t *zero = xcalloc(patches, sizeof(int32_t));
    int32_t *end = xmalloc((size_t)patches * sizeof(int32_t));
    for (uint32_t i = 0; i < patches; i++) { end[i] = (int32_t)patches; }
    CHECK(ling3vl_vision_pack(rgb, width, grid_h, grid_w, host, indices,
                              weights_bilinear, rows, cols));

    CHECK(ds4_gpu_tensor_write(v.patches, 0, host,
                               (uint64_t)patches * L3V_VIT_PATCH_DIM * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(v.pos_indices, 0, indices,
                               (uint64_t)patches * 4u * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(v.pos_weights, 0, weights_bilinear,
                               (uint64_t)patches * 4u * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(v.height, 0, rows, (uint64_t)patches * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(v.width, 0, cols, (uint64_t)patches * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(v.seg_start, 0, zero, (uint64_t)patches * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(v.seg_end, 0, end, (uint64_t)patches * sizeof(int32_t)));
    CHECK(ling3vl_vision_forward(&v, &m, &weights, patches));
    CHECK(ds4_gpu_synchronize());

    const uint32_t merged = patches / 4u;
    float *out = xmalloc((size_t)merged * L3V_HIDDEN * sizeof(float));
    CHECK(ds4_gpu_tensor_read(v.projected, 0, out,
                              (uint64_t)merged * L3V_HIDDEN * sizeof(float)));
    double energy = 0.0;
    for (size_t i = 0; i < (size_t)merged * L3V_HIDDEN; i++) {
        CHECK(isfinite(out[i]));
        energy += (double)out[i] * out[i];
    }
    /* Distinct rows: a collapsed projector would emit one repeated vector. */
    double diff = 0.0;
    for (uint32_t d = 0; d < L3V_HIDDEN && merged > 1u; d++) {
        const double e = (double)out[d] - out[(size_t)(merged - 1u) * L3V_HIDDEN + d];
        diff += e * e;
    }
    CHECK(energy > 0.0 && (merged == 1u || diff > 0.0));
    printf("Ling vision grid=%ux%u patches=%u merged=%u rms=%.6g PASS\n",
           grid_h, grid_w, patches, merged,
           sqrt(energy / ((double)merged * L3V_HIDDEN)));
    free(out);
    free(end); free(zero); free(cols); free(rows);
    free(weights_bilinear); free(indices); free(host); free(rgb);
    ling3vl_vision_free(&v);
    ds4_gpu_cleanup();
    model_close(&m);
    return 0;
}
