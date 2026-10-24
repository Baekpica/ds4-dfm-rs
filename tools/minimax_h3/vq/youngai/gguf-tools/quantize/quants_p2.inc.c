static void ds4q_write_q4_k_block_weighted(const float *x, uint8_t *y, const float *quant_weights) {
    enum { scales_off = 4, qs_off = 16 };
    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K / 32];
    uint8_t Lm[QK_K / 32];
    float weights[32];
    float sw[QK_K / 32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; l++) sum_x2 += x[l] * x[l];
    float sigma2 = 2 * sum_x2 / QK_K;
    float av_x = sqrtf(sigma2);

    for (int j = 0; j < QK_K / 32; j++) {
        if (quant_weights) {
            const float *qw = quant_weights + 32 * j;
            for (int l = 0; l < 32; l++) weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
        } else {
            for (int l = 0; l < 32; l++) weights[l] = av_x + fabsf(x[32 * j + l]);
        }
        float sumw = 0;
        for (int l = 0; l < 32; l++) sumw += weights[l];
        sw[j] = sumw;
        scales[j] = ds4q_make_qkx3_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                           &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float d_block = ds4q_make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
    float m_block = ds4q_make_qp_quants(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; j++) {
        uint8_t ls = Ls[j];
        uint8_t lm = Lm[j];
        if (j < 4) {
            scales_out[j] = ls;
            scales_out[j + 4] = lm;
        } else {
            scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales_out[j - 4] |= ((ls >> 4) << 6);
            scales_out[j - 0] |= ((lm >> 4) << 6);
        }
    }

    uint16_t d = ds4q_f32_to_f16(d_block);
    uint16_t dmin = ds4q_f32_to_f16(m_block);
    memcpy(y + 0, &d, sizeof(d));
    memcpy(y + 2, &dmin, sizeof(dmin));

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; j++) {
        ds4q_get_scale_min_k4(j, scales_out, &sc, &m);
        const float dd = ds4q_f16_to_f32(d) * sc;
        if (!dd) continue;
        const float dm = ds4q_f16_to_f32(dmin) * m;
        for (int ii = 0; ii < 32; ii++) {
            int l = ds4q_nearest_int((x[32 * j + ii] + dm) / dd);
            l = DS4Q_MAX(0, DS4Q_MIN(15, l));
            L[32 * j + ii] = l;
        }
    }

    uint8_t *q = qs_out;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; l++) q[l] = L[j + l] | (L[j + l + 32] << 4);
        q += 32;
    }
}

static size_t ds4q_quantize_q4_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q4_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q4_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q4_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q4_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

static void ds4q_write_q2_k_block_ref(const float *x, uint8_t *y) {
    enum { scales_off = 0, qs_off = 16, d_off = 80, dmin_off = 82 };
    const float q4scale = 15.0f;
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float weights[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; j++) {
        for (int l = 0; l < 16; l++) weights[l] = fabsf(x[16 * j + l]);
        scales[j] = ds4q_make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j,
                                           &mins[j], Laux, -0.5f, 0.1f, 15, true);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    uint16_t hd, hmin;
    if (max_scale > 0) {
        float iscale = q4scale / max_scale;
        for (int j = 0; j < QK_K / 16; j++) scales_out[j] = ds4q_nearest_int(iscale * scales[j]);
        hd = ds4q_f32_to_f16(max_scale / q4scale);
    } else {
        memset(scales_out, 0, QK_K / 16);
        hd = ds4q_f32_to_f16(0.0f);
    }
    if (max_min > 0) {
        float iscale = q4scale / max_min;
        for (int j = 0; j < QK_K / 16; j++) scales_out[j] |= ds4q_nearest_int(iscale * mins[j]) << 4;
        hmin = ds4q_f32_to_f16(max_min / q4scale);
    } else {
        hmin = ds4q_f32_to_f16(0.0f);
    }
    memcpy(y + d_off, &hd, sizeof(hd));
    memcpy(y + dmin_off, &hmin, sizeof(hmin));

    for (int j = 0; j < QK_K / 16; j++) {
        const float d = ds4q_f16_to_f32(hd) * (scales_out[j] & 0xF);
        if (!d) continue;
        const float dm = ds4q_f16_to_f32(hmin) * (scales_out[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int((x[16 * j + ii] + dm) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(3, l));
            L[16 * j + ii] = l;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            qs_out[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static void ds4q_write_q2_k_block_weighted(const float *x, uint8_t *y, const float *quant_weights) {
    enum { scales_off = 0, qs_off = 16, d_off = 80, dmin_off = 82 };
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    float sw[QK_K / 16];
    float weight[16];
    uint8_t Ls[QK_K / 16], Lm[QK_K / 16];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    memset(sw, 0, sizeof(sw));
    float sumx2 = 0;
    for (int j = 0; j < QK_K; j++) sumx2 += x[j] * x[j];
    float sigma2 = sumx2 / QK_K;
    for (int j = 0; j < QK_K / 16; j++) {
        const float *qw = quant_weights + 16 * j;
        for (int l = 0; l < 16; l++) weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
        for (int l = 0; l < QK_K / 16; l++) sw[j] += weight[l];
        scales[j] = ds4q_make_qkx3_quants(16, 3, x + 16 * j, weight, L + 16 * j,
                                           &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float dm = ds4q_make_qp_quants(QK_K / 16, 15, scales, Ls, sw);
    float mm = ds4q_make_qp_quants(QK_K / 16, 15, mins, Lm, sw);
    uint16_t hd = ds4q_f32_to_f16(dm);
    uint16_t hmin = ds4q_f32_to_f16(mm);
    memcpy(y + d_off, &hd, sizeof(hd));
    memcpy(y + dmin_off, &hmin, sizeof(hmin));
    dm = ds4q_f16_to_f32(hd);
    mm = ds4q_f16_to_f32(hmin);

    for (int j = 0; j < QK_K / 16; j++) scales_out[j] = Ls[j] | (Lm[j] << 4);

    for (int j = 0; j < QK_K / 16; j++) {
        const float d = dm * (scales_out[j] & 0xF);
        if (!d) continue;
        const float m = mm * (scales_out[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int((x[16 * j + ii] + m) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(3, l));
            L[16 * j + ii] = l;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            qs_out[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static size_t ds4q_quantize_q2_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q2_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q2_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q2_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q2_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

typedef struct {
    uint64_t *grid;
    int *map;
    uint16_t *neighbours;
    /* neighbours 是变长表(每个缺失码一个 count 头 + 若干邻居 id); 长度只有建表时
     * 才知道, 记下来供 GPU 编码器整表上传。 */
    int64_t neighbours_len;
} ds4q_iq2_data;

static ds4q_iq2_data ds4q_iq2_xxs_data;

static int ds4q_iq2_compare_func(const void *left, const void *right) {
    const int *l = (const int *)left;
    const int *r = (const int *)right;
    return l[0] < r[0] ? -1 :
           l[0] > r[0] ?  1 :
           l[1] < r[1] ? -1 :
           l[1] > r[1] ?  1 : 0;
}

/*
 * IQ2_XXS quantizes a 256-value row block as eight 32-value groups.  Each
 * group stores four 8-value grid indices plus four 7-bit sign masks; the
 * single f16 block scale is refined by 4-bit per-group scale nibbles.
 *
 * The grid is tiny, but not every possible 2-bit 8-tuple is allowed.  During
 * initialization we build the direct map for allowed tuples and a nearest-grid
 * list for the missing ones, matching the GGML search exactly.
 */
static void ds4q_iq2_xxs_init(void) {
    if (ds4q_iq2_xxs_data.grid) return;
    pthread_mutex_lock(&ds4q_init_mutex);
    if (ds4q_iq2_xxs_data.grid) {
        pthread_mutex_unlock(&ds4q_init_mutex);
        return;
    }

    enum { grid_size = 256, map_size = 43692, neighbour_shells = 2 };
    static const uint16_t kgrid[256] = {
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    };

    uint64_t *grid = malloc((size_t)grid_size * sizeof(grid[0]));
    int *map = malloc((size_t)map_size * sizeof(map[0]));
    int *dist2 = malloc((size_t)2 * grid_size * sizeof(dist2[0]));
    assert(grid && map && dist2);

    for (int k = 0; k < grid_size; k++) {
        int8_t *pos = (int8_t *)(grid + k);
        for (int i = 0; i < 8; i++) {
            int l = (kgrid[k] >> (2 * i)) & 3;
            pos[i] = 2 * l + 1;
        }
    }

    for (int i = 0; i < map_size; i++) map[i] = -1;
    for (int i = 0; i < grid_size; i++) map[kgrid[i]] = i;

    int8_t pos[8];
    int num_neighbors = 0;
    int num_not_in_map = 0;
    for (int i = 0; i < map_size; i++) {
        if (map[i] >= 0) continue;
        num_not_in_map++;
        for (int k = 0; k < 8; k++) pos[k] = 2 * ((i >> (2 * k)) & 3) + 1;
        for (int j = 0; j < grid_size; j++) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq2_compare_func);
        int d2 = dist2[0], have = 1;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                have++;
            }
            num_neighbors++;
        }
    }

    uint16_t *neighbours = malloc((size_t)(num_neighbors + num_not_in_map) * sizeof(neighbours[0]));
    assert(neighbours);
    int counter = 0;
    for (int i = 0; i < map_size; i++) {
        if (map[i] >= 0) continue;
        for (int k = 0; k < 8; k++) pos[k] = 2 * ((i >> (2 * k)) & 3) + 1;
        for (int j = 0; j < grid_size; j++) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq2_compare_func);
        map[i] = -(counter + 1);
        int d2 = dist2[0], have = 1;
        uint16_t *start = &neighbours[counter++];
        int n = 0;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                have++;
            }
            neighbours[counter++] = (uint16_t)dist2[2 * j + 1];
            n++;
        }
        *start = (uint16_t)n;
    }

    free(dist2);
    ds4q_iq2_xxs_data.map = map;
    ds4q_iq2_xxs_data.neighbours = neighbours;
    ds4q_iq2_xxs_data.neighbours_len = counter;
    ds4q_iq2_xxs_data.grid = grid; /* grid 最后赋值: 它是"已初始化"哨兵 */
    pthread_mutex_unlock(&ds4q_init_mutex);
}

/* IQ2_XXS 搜索表导出(GPU 编码器一次性上传常驻显存用)。表本身仍由上面的 host
 * 侧建表产生 —— 建表里的 qsort 不进 device。 */
void ds4q_iq2_xxs_tables(const uint64_t **grid, const int **map, const uint16_t **neighbours,
                         int *grid_size, int *map_size, int64_t *neighbours_len) {
    ds4q_iq2_xxs_init();
    if (grid) *grid = ds4q_iq2_xxs_data.grid;
    if (map) *map = ds4q_iq2_xxs_data.map;
    if (neighbours) *neighbours = ds4q_iq2_xxs_data.neighbours;
    if (grid_size) *grid_size = 256;
    if (map_size) *map_size = 43692;
    if (neighbours_len) *neighbours_len = ds4q_iq2_xxs_data.neighbours_len;
}

static int ds4q_iq2_find_best_neighbour(const uint16_t *neighbours, const uint64_t *grid,
                                        const float *xval, const float *weight,
                                        float scale, uint8_t *L) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; j++) {
        const int8_t *pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; i++) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t *pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; i++) L[i] = (uint8_t)((pg[i] - 1) / 2);
    return grid_index;
}

