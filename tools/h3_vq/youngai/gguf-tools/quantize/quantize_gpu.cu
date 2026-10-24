/*
 * quantize_gpu.cu — IQ2_XXS 超块编码器的 CUDA 移植。
 *
 * 为什么: 标量量化器的 IQ2_XXS 搜索是纯 CPU 的(20 线程约 1 分钟/层)。一个 256
 * 元素超块的编码与其它超块完全独立 —— 天然一线程一超块; 一张专家张量
 * (nrows*ncols/256 个超块, 典型 32768 个)一把提交。
 *
 * 数值契约(硬约束): 逐位复刻 quants.c 的 CPU 编码器。
 *   - CPU 侧用 gcc -std=c11(ISO 模式默认 -ffp-contract=off), 汇编里没有 fmadd,
 *     即每个乘/加/除/开方都是独立 IEEE 单精度运算;
 *   - 因此本文件必须用 nvcc -fmad=false(禁 FMA 融合) + 不开 --use_fast_math
 *     (保 prec-div/prec-sqrt), 见 scripts/build_dsq_gpu_spark.sh。
 *   任何"顺手优化"(重排累加、改用 rsqrtf、开 fast math)都会破坏逐字节一致, 禁止。
 *
 * 表(grid/map/neighbours)仍由 quants.c 的 host 侧 init 建好(那里的 qsort 只在建表
 * 时用, 不进 device), 通过 ds4q_iq2_xxs_tables() 取指针后一次性上传常驻显存。
 */

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256
#define IQ2_XXS_BLOCK_BYTES 66
#define DS4Q_GROUP_MAX_EPS 1e-15f
#define DMIN(a, b) ((a) < (b) ? (a) : (b))
#define DMAX(a, b) ((a) > (b) ? (a) : (b))

/* quants.c 的表导出(host 侧, 已完成建表) */
extern "C" void ds4q_iq2_xxs_tables(const uint64_t **grid, const int **map,
                                    const uint16_t **neighbours,
                                    int *grid_size, int *map_size, int64_t *neighbours_len);

/* ===== device 侧: quants.c 标量函数的逐式移植 ===== */

/* ds4q_f32_to_f16 的移植: 位操作 + 一次乘法链, 与 CPU 版同序同精度。 */
__device__ static inline uint16_t d_f32_to_f16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = __float_as_uint(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;

    base = __uint_as_float((bias >> 1) + 0x07800000u) + base;
    const uint32_t out = __float_as_uint(base);
    const uint32_t exp_bits = (out >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = out & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? (uint32_t)0x7E00 : nonsign));
}

/* ds4q_nearest_int: 加 magic number 后取尾数位。memcpy → __float_as_int(同为位重解释)。 */
__device__ static inline int d_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i = __float_as_int(val);
    return (i & 0x007fffff) - 0x00400000;
}

/* grid 条目是 8 个 int8(取值 1/3/5/7) 打进一个 u64; 小端下等价于按字节取。 */
__device__ static inline float d_grid_q(uint64_t g, int i) {
    return (float)(int8_t)((g >> (8 * i)) & 0xffu);
}

/* ds4q_iq2_find_best_neighbour 的移植。 */
__device__ static int d_find_best_neighbour(const uint16_t *__restrict__ neighbours,
                                            const uint64_t *__restrict__ grid,
                                            const float *xval, const float *weight,
                                            float scale, uint8_t *L) {
    int num_neighbors = neighbours[0];
    float best_d2 = 3.402823466e+38f; /* FLT_MAX */
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; j++) {
        const uint64_t g = grid[neighbours[j]];
        float d2 = 0;
        for (int i = 0; i < 8; i++) {
            float q = d_grid_q(g, i);
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    const uint64_t gb = grid[grid_index];
    for (int i = 0; i < 8; i++) L[i] = (uint8_t)(((int)d_grid_q(gb, i) - 1) / 2);
    return grid_index;
}

/* ds4q_make_qp_quants 的移植(n=32, nmax=4 固定调用面, 但保留通用形参)。 */
__device__ static float d_make_qp_quants(int n, int nmax, const float *x, uint8_t *L,
                                         const float *quant_weights) {
    float max = 0;
    for (int i = 0; i < n; i++) max = DMAX(max, x[i]);
    if (max < DS4Q_GROUP_MAX_EPS) {
        for (int i = 0; i < n; i++) L[i] = 0;
        return 0.0f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; i++) L[i] = (uint8_t)d_nearest_int(iscale * x[i]);
    float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - scale * L[i];
        best_mse += quant_weights[i] * diff * diff;
    }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        float iscale_is = (0.1f * is + nmax) / max;
        float scale_is = 1 / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; i++) {
            int l = d_nearest_int(iscale_is * x[i]);
            l = DMIN(nmax, l);
            float diff = x[i] - scale_is * l;
            mse += quant_weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; i++) {
        int l = d_nearest_int(iscale * x[i]);
        l = DMIN(nmax, l);
        L[i] = (uint8_t)l;
        float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int n_changed = 0;
        for (int i = 0; i < n; i++) {
            float w = quant_weights[i];
            float slx = sumlx - w * x[i] * L[i];
            float sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = d_nearest_int(x[i] * sl2 / slx);
                new_l = DMIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (uint8_t)new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed++;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

/* ds4q_write_iq2_xxs_block 的移植。y 指向 66 字节输出块(可能非对齐 → 按字节写)。 */
__device__ static void d_write_iq2_xxs_block(const float *__restrict__ x,
                                             uint8_t *__restrict__ y,
                                             const float *__restrict__ quant_weights,
                                             const uint64_t *__restrict__ grid,
                                             const int *__restrict__ map,
                                             const uint16_t *__restrict__ neighbours) {
    const int block_size = 32;
    const int k_max_q = 3;

    uint32_t q2[2 * (QK_K / 32)];
    float scales[QK_K / 32];
    float weight[32];
    float xval[32];
    uint8_t L[32];
    uint8_t Laux[32];
    float waux[32];
    uint8_t block_signs[4];

    uint16_t hd = d_f32_to_f16(0.0f);
    y[0] = (uint8_t)(hd & 0xff);
    y[1] = (uint8_t)(hd >> 8);
    for (int i = 0; i < 2 * (QK_K / 32); i++) q2[i] = 0;

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
        for (int i = 1; i < block_size; i++) max = DMAX(max, xval[i]);
        if (max < DS4Q_GROUP_MAX_EPS) {
            scales[ib] = 0;
            for (int i = 0; i < 32; i++) L[i] = 0;
            continue;
        }

        float scale = d_make_qp_quants(block_size, k_max_q + 1, xval, L, weight);
        float eff_max = scale * k_max_q;
        if (eff_max <= 0) {
            scales[ib] = 0;
            for (int i = 0; i < 32; i++) L[i] = 0;
            continue;
        }

        float best = 0;
        for (int is = -6; is <= 6; is++) {
            float id = (2 * k_max_q - 1 + is * 0.1f) / eff_max;
            float this_scale = 1 / id;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = d_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DMAX(0, DMIN(k_max_q - 1, l));
                    Laux[8 * k + i] = (uint8_t)l;
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    d_find_best_neighbour(nbs, grid, xval + 8 * k, waux + 8 * k,
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
                for (int i = 0; i < 32; i++) L[i] = Laux[i];
            }
        }

        if (scale > 0) {
            float id = 1 / scale;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = d_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DMAX(0, DMIN(k_max_q - 1, l));
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    grid_index = d_find_best_neighbour(nbs, grid, xval + 8 * k,
                                                       waux + 8 * k, scale, L + 8 * k);
                }
                const uint64_t g = grid[grid_index];
                for (int i = 0; i < 8; i++) L[8 * k + i] = (uint8_t)(((int)d_grid_q(g, i) - 1) / 2);
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
            for (int k = 0; k < 4; k++) block_signs[k] = (uint8_t)((~block_signs[k]) & 127);
        }

        for (int k = 0; k < 4; k++) {
            uint16_t u = 0;
            for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
            int grid_index = map[u];
            q2[2 * ib + 0] |= (uint32_t)grid_index << (8 * k);
            q2[2 * ib + 1] |= (uint32_t)block_signs[k] << (7 * k);
        }
        scales[ib] = scale;
        max_scale = DMAX(max_scale, scale);
    }

    if (!max_scale) {
        for (int i = 0; i < QK_K / 4; i++) y[2 + i] = 0;
        return;
    }

    float d = max_scale / 31;
    hd = d_f32_to_f16(d);
    y[0] = (uint8_t)(hd & 0xff);
    y[1] = (uint8_t)(hd >> 8);
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / block_size; ib++) {
        int l = d_nearest_int(0.5f * (id * scales[ib] - 1));
        l = DMAX(0, DMIN(15, l));
        q2[2 * ib + 1] |= (uint32_t)l << 28;
    }
    /* memcpy(y + 2, q2, 64): 输出块 66 字节步进 → 目标地址无 4 字节对齐保证, 按字节写。 */
    for (int i = 0; i < 2 * (QK_K / 32); i++) {
        uint32_t v = q2[i];
        y[2 + 4 * i + 0] = (uint8_t)(v & 0xff);
        y[2 + 4 * i + 1] = (uint8_t)((v >> 8) & 0xff);
        y[2 + 4 * i + 2] = (uint8_t)((v >> 16) & 0xff);
        y[2 + 4 * i + 3] = (uint8_t)((v >> 24) & 0xff);
    }
}

/* 一线程一超块。grid(256*8B=2KB) 拷进 shared: 邻居搜索是发散随机访问, 常量内存
 * 在 warp 内地址发散时会串行化, shared 不会。 */
__global__ __launch_bounds__(128) void iq2xxs_encode_kernel(
        const float *__restrict__ src, const float *__restrict__ qw,
        uint8_t *__restrict__ dst,
        const uint64_t *__restrict__ grid_g, const int *__restrict__ map,
        const uint16_t *__restrict__ neighbours,
        int64_t total_blocks, int64_t blocks_per_row, int64_t row_size) {
    __shared__ uint64_t s_grid[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_grid[i] = grid_g[i];
    __syncthreads();

    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total_blocks) return;
    const int64_t row = t / blocks_per_row;
    const int64_t b = t - row * blocks_per_row;

    const float *x = src + (row * blocks_per_row + b) * QK_K;
    uint8_t *y = dst + row * row_size + b * IQ2_XXS_BLOCK_BYTES;
    d_write_iq2_xxs_block(x, y, qw + b * QK_K, s_grid, map, neighbours);
}

/* ===== host 侧: 表常驻 + 多流缓冲池 ===== */

typedef struct {
    cudaStream_t stream;
    float *d_src;
    size_t src_cap;
    uint8_t *d_dst;
    size_t dst_cap;
    float *d_qw;
    size_t qw_cap;
    int busy;
} gpu_ctx;

#define DSQ_MAX_CTX 16

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;
static int g_state = 0; /* 0=未初始化 1=可用 -1=不可用 */
static int g_nctx = 0;
static int g_threads_per_block = 128;   /* __launch_bounds__(128) 的上限值; 原 DS4Q_GPU_BLOCK 从无活脚本改过, 写死 */
static gpu_ctx g_ctx[DSQ_MAX_CTX];
static uint64_t *g_grid = NULL;
static int *g_map = NULL;
static uint16_t *g_neigh = NULL;

/* 表上传 + 流/缓冲创建。持 g_lock 调用。
 * 编了 DS4Q_CUDA 就走 GPU(★只写 GPU 铁律 2026-08-28★); 原 DS4Q_GPU=0 强制 CPU
 * 的诊断分支已删, 仅设备真不可用时回落 CPU 编码器。 */
static int gpu_init_locked(void) {
    if (g_state) return g_state > 0;
    g_state = -1;

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        fprintf(stderr, "iq2_xxs: 无可用 CUDA 设备, 回落 CPU 编码器\n");
        return 0;
    }

    const uint64_t *grid = NULL;
    const int *map = NULL;
    const uint16_t *neigh = NULL;
    int grid_n = 0, map_n = 0;
    int64_t neigh_n = 0;
    ds4q_iq2_xxs_tables(&grid, &map, &neigh, &grid_n, &map_n, &neigh_n);
    if (!grid || !map || !neigh || neigh_n <= 0) {
        fprintf(stderr, "iq2_xxs: 表未就绪, 回落 CPU 编码器\n");
        return 0;
    }

    cudaError_t e = cudaSuccess;
    e = e ? e : cudaMalloc((void **)&g_grid, (size_t)grid_n * sizeof(uint64_t));
    e = e ? e : cudaMalloc((void **)&g_map, (size_t)map_n * sizeof(int));
    e = e ? e : cudaMalloc((void **)&g_neigh, (size_t)neigh_n * sizeof(uint16_t));
    e = e ? e : cudaMemcpy(g_grid, grid, (size_t)grid_n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    e = e ? e : cudaMemcpy(g_map, map, (size_t)map_n * sizeof(int), cudaMemcpyHostToDevice);
    e = e ? e : cudaMemcpy(g_neigh, neigh, (size_t)neigh_n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        fprintf(stderr, "iq2_xxs: 表上传失败(%s), 回落 CPU 编码器\n", cudaGetErrorString(e));
        return 0;
    }

    const int n = 4;   /* 并发流数: 原 DS4Q_GPU_STREAMS 缺省, 从无活脚本改过, 写死 */
    for (int i = 0; i < n; i++) {
        if (cudaStreamCreate(&g_ctx[i].stream) != cudaSuccess) break;
        g_nctx++;
    }
    if (g_nctx < 1) {
        fprintf(stderr, "iq2_xxs: 创建 stream 失败, 回落 CPU 编码器\n");
        return 0;
    }

    fprintf(stderr, "iq2_xxs: CUDA 编码器启用 (streams=%d, block=%d, neigh=%lld)\n",
            g_nctx, g_threads_per_block, (long long)neigh_n);
    g_state = 1;
    return 1;
}

/* 取一个空闲上下文(缓冲按需扩容, 常驻不释放)。返回索引, -1=不可用。 */
static int ctx_acquire(void) {
    pthread_mutex_lock(&g_lock);
    if (!gpu_init_locked()) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    for (;;) {
        for (int i = 0; i < g_nctx; i++) {
            if (!g_ctx[i].busy) {
                g_ctx[i].busy = 1;
                pthread_mutex_unlock(&g_lock);
                return i;
            }
        }
        pthread_cond_wait(&g_cond, &g_lock);
    }
}

static void ctx_release(int idx) {
    pthread_mutex_lock(&g_lock);
    g_ctx[idx].busy = 0;
    pthread_cond_signal(&g_cond);
    pthread_mutex_unlock(&g_lock);
}

static int ctx_reserve(void **buf, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*buf) cudaFree(*buf);
    *buf = NULL;
    *cap = 0;
    if (cudaMalloc(buf, need) != cudaSuccess) return 0;
    *cap = need;
    return 1;
}

/*
 * GPU IQ2_XXS 编码入口。
 *   src  = 张量起始(nrows*ncols 个 f32, 行主序)
 *   dst  = 输出起始(nrows*row_size 字节, row_size = 66*(ncols/256))
 *   imatrix_row_weights = 每列一个权重(长 ncols, 所有行共用), 必需
 * 返回 1 = GPU 已写出全部字节; 0 = 不可用/失败, 调用侧回落 CPU。
 */
extern "C" int ds4q_iq2xxs_encode_gpu(const float *src, void *dst, int64_t nrows, int64_t ncols,
                                      const float *imatrix_row_weights) {
    if (!src || !dst || !imatrix_row_weights) return 0;
    if (nrows <= 0 || ncols <= 0 || ncols % QK_K != 0) return 0;

    const int64_t blocks_per_row = ncols / QK_K;
    const int64_t total_blocks = nrows * blocks_per_row;
    const size_t row_size = (size_t)blocks_per_row * IQ2_XXS_BLOCK_BYTES;
    const size_t src_bytes = (size_t)nrows * (size_t)ncols * sizeof(float);
    const size_t dst_bytes = (size_t)nrows * row_size;
    const size_t qw_bytes = (size_t)ncols * sizeof(float);

    int idx = ctx_acquire();
    if (idx < 0) return 0;
    gpu_ctx *c = &g_ctx[idx];

    int ok = ctx_reserve((void **)&c->d_src, &c->src_cap, src_bytes) &&
             ctx_reserve((void **)&c->d_dst, &c->dst_cap, dst_bytes) &&
             ctx_reserve((void **)&c->d_qw, &c->qw_cap, qw_bytes);
    cudaError_t e = ok ? cudaSuccess : cudaErrorMemoryAllocation;

    if (e == cudaSuccess)
        e = cudaMemcpyAsync(c->d_src, src, src_bytes, cudaMemcpyHostToDevice, c->stream);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(c->d_qw, imatrix_row_weights, qw_bytes, cudaMemcpyHostToDevice, c->stream);
    if (e == cudaSuccess) {
        int tpb = g_threads_per_block;
        int64_t nblk = (total_blocks + tpb - 1) / tpb;
        iq2xxs_encode_kernel<<<(unsigned)nblk, tpb, 0, c->stream>>>(
                c->d_src, c->d_qw, c->d_dst, g_grid, g_map, g_neigh,
                total_blocks, blocks_per_row, (int64_t)row_size);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(dst, c->d_dst, dst_bytes, cudaMemcpyDeviceToHost, c->stream);
    if (e == cudaSuccess)
        e = cudaStreamSynchronize(c->stream);

    ctx_release(idx);

    if (e != cudaSuccess) {
        fprintf(stderr, "iq2_xxs: GPU 编码失败(%s), 本张量回落 CPU\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}
