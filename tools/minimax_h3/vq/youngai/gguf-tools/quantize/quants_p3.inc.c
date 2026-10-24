static void ds4q_write_iq2_xxs_block(const float *x, uint8_t *y, const float *quant_weights) {
    enum { d_off = 0, qs_off = 2, block_size = 32, k_max_q = 3 };
    assert(quant_weights);

    uint32_t q2[2 * (QK_K / block_size)];
    float scales[QK_K / block_size];
    float weight[block_size];
    float xval[block_size];
    uint8_t L[block_size];
    uint8_t Laux[block_size];
    float waux[block_size];
    uint8_t block_signs[4];

    uint16_t hd = ds4q_f32_to_f16(0.0f);
    memcpy(y + d_off, &hd, sizeof(hd));
    memset(q2, 0, sizeof(q2));

    const uint64_t *grid = ds4q_iq2_xxs_data.grid;
    const int *map = ds4q_iq2_xxs_data.map;
    const uint16_t *neighbours = ds4q_iq2_xxs_data.neighbours;
    assert(grid && map && neighbours);

    float sumx2 = 0;
    for (int i = 0; i < QK_K; i++) sumx2 += x[i] * x[i];
    float sigma2 = sumx2 / QK_K;
    float max_scale = 0;

    for (int ib = 0; ib < QK_K / block_size; ib++) {
        const float *xb = x + block_size * ib;
        const float *qw = quant_weights + block_size * ib;
        for (int i = 0; i < block_size; i++) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 4; k++) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; i++) {
                float v = xb[8 * k + i];
                if (v >= 0) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    nflip++;
                    s |= (uint8_t)(1u << i);
                }
            }
            if (nflip % 2) {
                int imin = 0;
                float min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                for (int i = 1; i < 8; i++) {
                    float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (ax < min) {
                        min = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (uint8_t)(1u << imin);
            }
            block_signs[k] = s & 127;
        }

        float max = xval[0];
        for (int i = 1; i < block_size; i++) max = DS4Q_MAX(max, xval[i]);
        if (max < DS4Q_GROUP_MAX_EPS) {
            scales[ib] = 0;
            memset(L, 0, sizeof(L));
            continue;
        }

        float scale = ds4q_make_qp_quants(block_size, k_max_q + 1, xval, L, weight);
        float eff_max = scale * k_max_q;
        if (eff_max <= 0) {
            scales[ib] = 0;
            memset(L, 0, sizeof(L));
            continue;
        }

        float best = 0;
        for (int is = -6; is <= 6; is++) {
            float id = (2 * k_max_q - 1 + is * 0.1f) / eff_max;
            float this_scale = 1 / id;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(k_max_q - 1, l));
                    Laux[8 * k + i] = (uint8_t)l;
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k, waux + 8 * k,
                                                 this_scale, Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; i++) {
                float w = weight[i];
                float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                memcpy(L, Laux, sizeof(L));
            }
        }

        if (scale > 0) {
            float id = 1 / scale;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(k_max_q - 1, l));
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    grid_index = ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k,
                                                              waux + 8 * k, scale, L + 8 * k);
                }
                const int8_t *pg = (const int8_t *)(grid + grid_index);
                for (int i = 0; i < 8; i++) L[8 * k + i] = (uint8_t)((pg[i] - 1) / 2);
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; i++) {
                float w = weight[i];
                float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; k++) block_signs[k] = (~block_signs[k]) & 127;
        }

        for (int k = 0; k < 4; k++) {
            uint16_t u = 0;
            for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
            int grid_index = map[u];
            assert(grid_index >= 0);
            q2[2 * ib + 0] |= (uint32_t)grid_index << (8 * k);
            q2[2 * ib + 1] |= (uint32_t)block_signs[k] << (7 * k);
        }
        assert(scale >= 0);
        scales[ib] = scale;
        max_scale = DS4Q_MAX(max_scale, scale);
    }

    if (!max_scale) {
        memset(y + qs_off, 0, QK_K / 4);
        return;
    }

    float d = max_scale / 31;
    hd = ds4q_f32_to_f16(d);
    memcpy(y + d_off, &hd, sizeof(hd));
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / block_size; ib++) {
        int l = ds4q_nearest_int(0.5f * (id * scales[ib] - 1));
        l = DS4Q_MAX(0, DS4Q_MIN(15, l));
        q2[2 * ib + 1] |= (uint32_t)l << 28;
    }
    memcpy(y + qs_off, q2, QK_K / 4);
}

/* 标量(CPU)编码: nrows 行 * ncols 列 -> out。GPU 路径的对拍参照也走这里。 */
static void ds4q_iq2_xxs_encode_cpu(const float *src, uint8_t *out, int64_t nrows,
                                    int64_t ncols, size_t row_size, const float *quant_weights) {
    const int64_t blocks_per_row = ncols / QK_K;
    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_IQ2_XXS].type_size;
            ds4q_write_iq2_xxs_block(xrow + (size_t)b * QK_K, block,
                                     quant_weights + (size_t)b * QK_K);
        }
    }
}

/* --gpu-verify 请求位(原 DS4Q_GPU_VERIFY env): setter 无条件存在(非 CUDA 构建
 * 置了也没有 GPU 结果可对拍, 与旧 env 在 CPU 构建下无效的行为一致)。 */
static int ds4q_gpu_verify_requested = 0;
void ds4q_gpu_verify_set(int on) { ds4q_gpu_verify_requested = on ? 1 : 0; }

#ifdef DS4Q_CUDA
/* quantize_gpu.cu: 返回 1 = GPU 已写出全部字节, 0 = 不可用/失败(回落 CPU)。 */
int ds4q_iq2xxs_encode_gpu(const float *src, void *dst, int64_t nrows, int64_t ncols,
                           const float *imatrix_row_weights);

static int ds4q_gpu_verify_on(void) { return ds4q_gpu_verify_requested; }
#endif

static size_t ds4q_quantize_iq2_xxs(const float *src, void *dst, int64_t start,
                                    int64_t nrows, int64_t ncols, const float *quant_weights) {
    assert(quant_weights);
    ds4q_iq2_xxs_init();
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_IQ2_XXS, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;

#ifdef DS4Q_CUDA
    if (ds4q_iq2xxs_encode_gpu(src + start, out, nrows, ncols, quant_weights)) {
        if (ds4q_gpu_verify_on()) {
            uint8_t *ref = malloc((size_t)nrows * row_size);
            if (ref) {
                ds4q_iq2_xxs_encode_cpu(src + start, ref, nrows, ncols, row_size, quant_weights);
                size_t bad = 0, first = 0;
                for (size_t i = 0; i < (size_t)nrows * row_size; i++) {
                    if (ref[i] != out[i]) {
                        if (!bad) first = i;
                        bad++;
                    }
                }
                fprintf(stderr, "iq2_xxs verify: bytes=%zu mismatch=%zu%s first=%zu\n",
                        (size_t)nrows * row_size, bad, bad ? " FAIL" : " OK", bad ? first : (size_t)0);
                free(ref);
            }
        }
        return (size_t)nrows * row_size;
    }
#endif

    ds4q_iq2_xxs_encode_cpu(src + start, out, nrows, ncols, row_size, quant_weights);
    return (size_t)nrows * row_size;
}

const char *ds4q_type_name(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return NULL;
    return ds4q_type_traits[type].name;
}

bool ds4q_can_quantize(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return false;
    return ds4q_type_traits[type].can_quantize;
}

int64_t ds4q_block_size(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return 0;
    return ds4q_type_traits[type].block_size;
}

size_t ds4q_row_size(ds4q_type type, int64_t ne) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return 0;
    /* GO1B stores one fp16 scale per row plus packed sign bits, which does not fit
     * the uniform type_size*(ne/block_size) block model; compute it directly. */
    if (type == DS4Q_TYPE_GO1B) return go1b_row_bytes(ne);
    const ds4q_traits *tr = &ds4q_type_traits[type];
    if (tr->block_size <= 0 || tr->type_size == 0 || ne % tr->block_size != 0) return 0;
    return tr->type_size * (size_t)(ne / tr->block_size);
}

bool ds4q_requires_imatrix(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return false;
    return ds4q_type_traits[type].requires_imatrix;
}

void ds4q_quantize_init(ds4q_type type) {
    if (type == DS4Q_TYPE_IQ2_XXS) {
        ds4q_iq2_xxs_init();
    }
}

size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix) {
    if (type == DS4Q_TYPE_Q8_0) {
        (void)imatrix;
        return ds4q_quantize_q8_0(src, dst, start, nrows, ncols);
    }
    if (type == DS4Q_TYPE_Q2_K) {
        return ds4q_quantize_q2_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_Q4_K) {
        return ds4q_quantize_q4_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_IQ2_XXS) {
        return ds4q_quantize_iq2_xxs(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_GO1B) {
        /* Per-row strict-binary encode. Mirror the chunk addressing used above:
         * `start` is an element offset (= start_row*ncols); rows are independent. */
        (void)imatrix;
        const size_t rb = go1b_row_bytes(ncols);
        const int64_t start_row = start / ncols;
        uint8_t *out = (uint8_t *)dst + (size_t)start_row * rb;
        return go1b_quantize(src + start, out, nrows, ncols);
    }
    if (type == DS4Q_TYPE_GO1B_BLK) {
        /* Strict-binary 256-element BLOCK encode (runtime/Metal on-disk format, ggml
         * type 40). Same chunk addressing as the other fixed-block encoders: `start`
         * is an element offset (= start_row*ncols), rows independent. Scale =
         * mean(|w|), or the Go-activation-weighted L_fix re-fit when an imatrix
         * (per-expert E[x²] slice, ncols floats) is supplied via --imatrix. */
        const size_t row_size = ds4q_row_size(DS4Q_TYPE_GO1B_BLK, ncols); /* 34*(ncols/256) */
        const int64_t start_row = start / ncols;
        uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
        return go1b_blk_quantize_imat(src + start, out, nrows, ncols, imatrix);
    }
    (void)src;
    (void)dst;
    (void)start;
    (void)nrows;
    (void)ncols;
    (void)imatrix;
    assert(!"unsupported DS4 quantization target");
    return 0;
}

float ds4q_f16_to_f32(uint16_t bits) {
    const uint32_t w = (uint32_t)bits << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = ds4q_f32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = ds4q_f32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? ds4q_f32_to_bits(denormalized_value) : ds4q_f32_to_bits(normalized_value));
    return ds4q_f32_from_bits(result);
}

float ds4q_bf16_to_f32(uint16_t bits) {
    return ds4q_f32_from_bits((uint32_t)bits << 16);
}

void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = ds4q_f32_to_f16(src[i]);
}

void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = ds4q_f32_to_bits(src[i]);
        if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
            dst[i] = (uint16_t)((bits >> 16) | 64);
        } else {
            dst[i] = (uint16_t)((bits + (UINT32_C(0x7fff) + ((bits >> 16) & 1))) >> 16);
        }
    }
}
