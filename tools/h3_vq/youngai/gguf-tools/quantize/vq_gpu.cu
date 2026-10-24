/* vq_gpu.cu — 量化器 GPU 常驻核(2026-08-18 用户令"改成 gpu 常驻, 先工程再量化")。
 * 两个热点整段上 GPU(段计时归因: kmeans assign + GPTQ 列组反馈 = 层耗时 ~70%):
 *   vqg_assign      k-means 批 assign: 每线程一向量, 码本 shared, 判据与 CPU 同
 *                   (值=|c|²-2v·c, 严格 < 保首现; cblas 浮点序差异容忍, held 对拍验)
 *   vqg_gptq_group  一组 g=128 列的整个"逐 dim 段 assign+误差反馈"循环:
 *                   行间完全独立(反馈只改本行后续列) → 每线程一行, 行内串行 32 段,
 *                   数学与 CPU 行内序逐项一致(Hi double→f32 为唯一近似)。
 * GB10 统一内存: malloc 指针直传, 无显式搬运。每 CPU 线程 thread-local stream。 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define VQG_MAX_NC 512
#define VQG_MAX_DIM 8

static __thread cudaStream_t g_vqg_stream = NULL;
static int g_vqg_ok = -1;

extern "C" void vqg_last_err(const char *tag) {
    cudaError_t e = cudaGetLastError();   /* 读并清除粘连错误 */
    if (e != cudaSuccess) { static int w=0; if(w++<4) fprintf(stderr, "[vqg] ★%s: 粘连CUDA错误=%s★\n", tag, cudaGetErrorString(e)); }
}
extern "C" int vqg_ready(void) {
    if (g_vqg_ok < 0) g_vqg_ok = (cudaFree(0) == cudaSuccess) ? 1 : 0;
    return g_vqg_ok;
}
static cudaStream_t vqg_stream(void) {
    if (!g_vqg_stream) cudaStreamCreateWithFlags(&g_vqg_stream, cudaStreamNonBlocking);
    return g_vqg_stream;
}

/* ★别用魔法数卡 nc★(2026-08-27 实测事故): vq_qc.h 三处调用点原先写死 nc<=512, 于是
 * nc=1024 静默掉回 CPU 标量 —— 实测 kmeans 慢 53×/gptq 慢 36×(每单位 15.5 vs 0.072),
 * 一层从 2.5 分钟变 23 分钟, 直接撞"必须用 GPU"铁律。而 kernel 对 nc 本就是通用循环
 * (c 从 threadIdx.x 步进 blockDim.x, 无 512 假设), 真正的限制只有两条:
 *   ① dim ≤ VQG_MAX_DIM —— 栈数组 float v[VQG_MAX_DIM] 的硬约束, 越界即错值
 *   ② 动态共享内存 nc*(dim+1)*4 字节装得下 —— 装不下 launch 直接失败
 * 所以按真限制放行, 不再拍数字。(nc=1024: dim=4 需 20KB, dim=8 需 36KB, 均在 48KB 内。)
 * 改这里前先想: 放行范围变了 = 更多情形从 CPU 路切到 GPU 路, 两路必须同判
 * (assign 的 "严格 <" 保首现语义就是为此); 验证法见 dynladder 标定的 cos 对表。 */
extern "C" int vqg_shm_ok(int nc, int dim) {
    if (dim > VQG_MAX_DIM || nc < 1) return 0;
    static int lim = -1;
    if (lim < 0) { if (cudaDeviceGetAttribute(&lim, cudaDevAttrMaxSharedMemoryPerBlock, 0) != cudaSuccess) lim = 48 * 1024; }
    return (size_t)(nc * dim + nc) * sizeof(float) <= (size_t)lim;
}

/* ---- assign: 每线程一向量 ---- */
__global__ static void vqg_assign_kernel(
        const float *__restrict__ V, int nv,
        const float *__restrict__ C, int nc, int dim, int *__restrict__ idx) {
    extern __shared__ float sh[];              /* C[nc*dim] + c2[nc] */
    float *Cs = sh, *c2 = sh + nc * dim;
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) Cs[i] = C[i];
    __syncthreads();
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Cs[c * dim + d] * Cs[c * dim + d];
        c2[c] = s;
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv) return;
    float v[VQG_MAX_DIM];
    for (int d = 0; d < dim; d++) v[d] = V[(size_t)i * dim + d];
    float best = 3.4e38f; int bi = 0;
    for (int c = 0; c < nc; c++) {
        float g = 0.0f;
        for (int d = 0; d < dim; d++) g += v[d] * Cs[c * dim + d];
        float val = c2[c] - 2.0f * g;
        if (val < best) { best = val; bi = c; }   /* 严格 < 保首现, 与 CPU 同判 */
    }
    idx[i] = bi;
}

extern "C" void vqg_assign(const float *V, int nv, const float *C, int nc, int dim, int *idx) {
    cudaStream_t st = vqg_stream();
    size_t shm = (size_t)(nc * dim + nc) * sizeof(float);
    vqg_assign_kernel<<<(nv + 255) / 256, 256, shm, st>>>(V, nv, C, nc, dim, idx);
    { vqg_last_err("vqg_assign_kernel后"); }
    cudaStreamSynchronize(st);
}

/* ---- GPTQ 组 v2(08-18 warp 化): 每 warp 一行 —— 32 lane 分摊 512 质心的 assign
 * (16 质心/lane + shfl 归约 argmin, 首现语义=最小 (val,idx) 对), 误差反馈 128 列由
 * 32 lane 并行更新(列间独立)。行内段序与 CPU 一致; Hi 常驻 shared(64KB, GB10 动态上限内)。 */
__global__ static void vqg_gptq_rows_warp_kernel(
        float *__restrict__ Wk, const float *__restrict__ C,
        const float *__restrict__ Hi, int rows, int g, int dim, int nc, int fb,
        int *__restrict__ sidx,
        const float *__restrict__ colw) {   /* [g] 通道能量权; NULL=平权(旧径逐位同) */
    extern __shared__ float sh[];
    float *Cs = sh, *c2 = sh + nc * dim;          /* 码本 + |c|² */
    float *His = c2 + nc;                          /* [g,g] f32 (fb 时装载) */
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) Cs[i] = C[i];
    __syncthreads();
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Cs[c * dim + d] * Cs[c * dim + d];
        c2[c] = s;
    }
    if (fb) for (int i = threadIdx.x; i < g * g; i += blockDim.x) His[i] = Hi[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    const int nseg = g / dim;
    for (int r = blockIdx.x * nwarp + warp; r < rows; r += gridDim.x * nwarp) {
        float *wkr = Wk + (size_t)r * g;
        for (int s = 0; s < nseg; s++) {
            const int jj = s * dim;
            float seg[VQG_MAX_DIM];
            for (int d = 0; d < dim; d++) seg[d] = wkr[jj + d];
            /* ★通道能量加权选码字(2026-08-31 巨值通道矿)★ dist=Σ s_d(v−c)²:
             * s_d=E[x²] ⇒ 输出误差口径的最优码字, 平权 L2 抹零 massive 通道的病根。
             * 常数项 Σs·v² 不进 argmin; colw=NULL 走旧平权路逐位不变。 */
            float ws[VQG_MAX_DIM], wseg[VQG_MAX_DIM];
            if (colw) for (int d = 0; d < dim; d++) { ws[d] = colw[jj + d]; wseg[d] = ws[d] * seg[d]; }
            /* lane 分质心: 每 lane nc/32 个, 保序 argmin(先 val 后 idx = 全局首现) */
            float best = 3.4e38f; int bi = nc;
            for (int c = lane; c < nc; c += 32) {
                float val;
                if (colw) {
                    val = 0.0f;
                    for (int d = 0; d < dim; d++) { const float cd = Cs[c * dim + d]; val += cd * (ws[d] * cd - 2.0f * wseg[d]); }
                } else {
                    float gd = 0.0f;
                    for (int d = 0; d < dim; d++) gd += seg[d] * Cs[c * dim + d];
                    val = c2[c] - 2.0f * gd;
                }
                if (val < best || (val == best && c < bi)) { best = val; bi = c; }
            }
            for (int off = 16; off > 0; off >>= 1) {
                float ob = __shfl_down_sync(0xffffffffu, best, off);
                int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (ob < best || (ob == best && oi < bi)) { best = ob; bi = oi; }
            }
            bi = __shfl_sync(0xffffffffu, bi, 0);
            if (lane == 0) sidx[(size_t)r * nseg + s] = bi;
            if (fb && jj + dim < g) {
                const float *q = Cs + bi * dim;
                for (int d = 0; d < dim; d++) {
                    float hjj = His[(size_t)(jj + d) * g + jj + d];
                    if (fabsf(hjj) < 1e-30f) hjj = 1e-30f;
                    float er = (seg[d] - q[d]) / hjj;
                    if (er == 0.0f) continue;
                    const float *hrow = His + (size_t)(jj + d) * g;
                    for (int j2 = jj + dim + lane; j2 < g; j2 += 32)
                        wkr[j2] -= er * hrow[j2];   /* 列间独立, lane 并行 */
                }
                __syncwarp();
            }
        }
    }
}

/* ---- GPTQ 组: 每线程一行, 行内串行段循环(与 CPU 逐项同序) ---- */
__global__ static void vqg_gptq_rows_kernel(
        float *__restrict__ Wk,                /* [rows, g] in/out */
        const float *__restrict__ C,           /* [nc, dim] */
        const float *__restrict__ Hi,          /* [g, g] f32(double 转换) */
        int rows, int g, int dim, int nc, int fb,
        int *__restrict__ sidx,                /* [rows, g/dim] */
        const float *__restrict__ colw) {      /* [g] 通道能量权; NULL=平权 */
    extern __shared__ float sh[];
    float *Cs = sh, *c2 = sh + nc * dim;
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) Cs[i] = C[i];
    __syncthreads();
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Cs[c * dim + d] * Cs[c * dim + d];
        c2[c] = s;
    }
    __syncthreads();
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float *wkr = Wk + (size_t)r * g;
    const int nseg = g / dim;
    for (int s = 0; s < nseg; s++) {
        const int jj = s * dim;
        float seg[VQG_MAX_DIM];
        for (int d = 0; d < dim; d++) seg[d] = wkr[jj + d];
        float ws[VQG_MAX_DIM], wseg[VQG_MAX_DIM];   /* 加权选码字, 见 warp 核注释 */
        if (colw) for (int d = 0; d < dim; d++) { ws[d] = colw[jj + d]; wseg[d] = ws[d] * seg[d]; }
        float best = 3.4e38f; int bi = 0;
        for (int c = 0; c < nc; c++) {
            float val;
            if (colw) {
                val = 0.0f;
                for (int d = 0; d < dim; d++) { const float cd = Cs[c * dim + d]; val += cd * (ws[d] * cd - 2.0f * wseg[d]); }
            } else {
                float gd = 0.0f;
                for (int d = 0; d < dim; d++) gd += seg[d] * Cs[c * dim + d];
                val = c2[c] - 2.0f * gd;
            }
            if (val < best) { best = val; bi = c; }
        }
        sidx[(size_t)r * nseg + s] = bi;
        if (fb && jj + dim < g) {
            const float *q = Cs + bi * dim;
            for (int d = 0; d < dim; d++) {
                float hjj = Hi[(size_t)(jj + d) * g + jj + d];
                if (fabsf(hjj) < 1e-30f) hjj = 1e-30f;
                float er = (seg[d] - q[d]) / hjj;
                if (er == 0.0f) continue;
                const float *hrow = Hi + (size_t)(jj + d) * g;
                for (int j2 = jj + dim; j2 < g; j2++) wkr[j2] -= er * hrow[j2];
            }
        }
    }
}

/* ---- v3 全矩阵批量(08-18): 组间独立 → 32 组一次 launch(grid.y=组), 调用数
 * 24.6k→768/层, 消灭 per-组 sync(实测 7.8ms/调用 = 193s/层的本体)。
 * 每 warp 一行×一组: 行内段循环与 CPU 同序; Wk 列窗口驻全局 buffer。 */
__global__ static void vqg_gptq_full_kernel(
        float *__restrict__ Wk,               /* [rows, cols] 工作副本(in/out) */
        const float *__restrict__ C,
        const float *__restrict__ Hi_all,     /* [ngrp, g, g] f32; NULL=无反馈 */
        int rows, int cols, int grp, int dim, int nc,
        int *__restrict__ sidx,               /* [rows, cols/dim] */
        const float *__restrict__ colw) {     /* [cols] 通道能量权; NULL=平权 */
    extern __shared__ float sh[];
    float *Cs = sh, *c2 = sh + nc * dim;
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) Cs[i] = C[i];
    __syncthreads();
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Cs[c * dim + d] * Cs[c * dim + d];
        c2[c] = s;
    }
    __syncthreads();
    const int grpi = blockIdx.y;
    const int j0 = grpi * grp;
    const int g = (cols - j0 < grp) ? (cols - j0) : grp;
    const float *Hi = Hi_all ? Hi_all + (size_t)grpi * grp * grp : NULL;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    const int nseg_all = cols / dim;
    const int nseg = g / dim;
    for (int r = blockIdx.x * nwarp + warp; r < rows; r += gridDim.x * nwarp) {
        float *wkr = Wk + (size_t)r * cols + j0;
        for (int s = 0; s < nseg; s++) {
            const int jj = s * dim;
            float seg[VQG_MAX_DIM];
            for (int d = 0; d < dim; d++) seg[d] = wkr[jj + d];
            float ws[VQG_MAX_DIM], wseg[VQG_MAX_DIM];   /* 加权选码字, 见 warp 核注释 */
            if (colw) for (int d = 0; d < dim; d++) { ws[d] = colw[j0 + jj + d]; wseg[d] = ws[d] * seg[d]; }
            float best = 3.4e38f; int bi = nc;
            for (int c = lane; c < nc; c += 32) {
                float val;
                if (colw) {
                    val = 0.0f;
                    for (int d = 0; d < dim; d++) { const float cd = Cs[c * dim + d]; val += cd * (ws[d] * cd - 2.0f * wseg[d]); }
                } else {
                    float gd = 0.0f;
                    for (int d = 0; d < dim; d++) gd += seg[d] * Cs[c * dim + d];
                    val = c2[c] - 2.0f * gd;
                }
                if (val < best || (val == best && c < bi)) { best = val; bi = c; }
            }
            for (int off = 16; off > 0; off >>= 1) {
                float ob = __shfl_down_sync(0xffffffffu, best, off);
                int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (ob < best || (ob == best && oi < bi)) { best = ob; bi = oi; }
            }
            bi = __shfl_sync(0xffffffffu, bi, 0);
            if (lane == 0) sidx[(size_t)r * nseg_all + (j0 + jj) / dim] = bi;
            if (Hi && jj + dim < g) {
                const float *q = Cs + bi * dim;
                for (int d = 0; d < dim; d++) {
                    float hjj = Hi[(size_t)(jj + d) * g + jj + d];
                    if (fabsf(hjj) < 1e-30f) hjj = 1e-30f;
                    float er = (seg[d] - q[d]) / hjj;
                    if (er == 0.0f) continue;
                    const float *hrow = Hi + (size_t)(jj + d) * g;
                    for (int j2 = jj + dim + lane; j2 < g; j2 += 32)
                        wkr[j2] -= er * hrow[j2];
                }
                __syncwarp();
            }
        }
    }
}

/* colw 设备缓存: 每矩阵一次 [cols] 上传(≤16KB), 线程私有随用随长 */
static __thread float *g_colw_d = NULL;
static __thread int g_colw_cap = 0;
static const float *vqg_colw_up(const float *colw_h, int n) {
    if (!colw_h) return NULL;
    if (g_colw_cap < n) { if (g_colw_d) cudaFree(g_colw_d); if (cudaMalloc((void **)&g_colw_d, (size_t)n * sizeof(float)) != cudaSuccess) { g_colw_d = NULL; g_colw_cap = 0; return NULL; } g_colw_cap = n; }
    if (cudaMemcpy(g_colw_d, colw_h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    return g_colw_d;
}

extern "C" void vqg_gptq_full(float *Wk, const float *C, const double *Hi_all_d,
                              int rows, int cols, int grp, int dim, int nc,
                              int has_fb, int *sidx, const float *colw_h) {
    static __thread float *hi_f = NULL;
    static __thread int hi_cap = 0;
    const int ngrp = (cols + grp - 1) / grp;
    if (has_fb) {
        const int need = ngrp * grp * grp;
        if (hi_cap < need) { if (hi_f) cudaFreeHost(hi_f); cudaMallocHost((void **)&hi_f, (size_t)need * sizeof(float)); hi_cap = need; }
        for (int i = 0; i < need; i++) hi_f[i] = (float)Hi_all_d[i];
    }
    const float *cw_d = vqg_colw_up(colw_h, cols);
    cudaStream_t st = vqg_stream();
    size_t shm = (size_t)(nc * dim + nc) * sizeof(float);
    dim3 grid((rows + 3) / 4, ngrp);
    vqg_gptq_full_kernel<<<grid, 128, shm, st>>>(
        Wk, C, has_fb ? hi_f : NULL, rows, cols, grp, dim, nc, sidx, cw_d);
    cudaStreamSynchronize(st);
}

extern "C" void vqg_gptq_group(float *Wk, const float *C, const double *Hi_d,
                               int rows, int g, int dim, int nc, int has_fb, int *sidx,
                               const float *colw_h) {
    static __thread float *hi_f = NULL;
    static __thread int hi_cap = 0;
    if (has_fb) {
        if (hi_cap < g * g) { if (hi_f) cudaFreeHost(hi_f); cudaMallocHost((void **)&hi_f, (size_t)g * g * sizeof(float)); hi_cap = g * g; }
        for (int i = 0; i < g * g; i++) hi_f[i] = (float)Hi_d[i];
    }
    const float *cw_d = vqg_colw_up(colw_h, g);
    cudaStream_t st = vqg_stream();
    size_t shm2 = (size_t)(nc * dim + nc + (has_fb ? g * g : 0)) * sizeof(float);
    if (shm2 <= 96 * 1024) {   /* warp 版: Hi 进 shared(128²×4=64KB, GB10 上限内) */
        static int attr_set = 0;
        if (!attr_set) { cudaFuncSetAttribute(vqg_gptq_rows_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); attr_set = 1; }
        vqg_gptq_rows_warp_kernel<<<(rows + 3) / 4, 128, shm2, st>>>(
            Wk, C, has_fb ? hi_f : NULL, rows, g, dim, nc, has_fb, sidx, cw_d);
    } else {
        size_t shm = (size_t)(nc * dim + nc) * sizeof(float);
        vqg_gptq_rows_kernel<<<(rows + 127) / 128, 128, shm, st>>>(
            Wk, C, has_fb ? hi_f : NULL, rows, g, dim, nc, has_fb, sidx, cw_d);
    }
    cudaStreamSynchronize(st);
}

/* ---- B 回放 dequant(08-18 第二刀): 位流→fp32 权重整矩阵, 每线程一行 ----
 * 语义与 vq_qc.h vq_unpack_dequant 逐位同(LE 位流+fp16 码本/行gain);
 * B 回放评估 18s/层的主体=768 矩阵 CPU 标量位流解码。 */
__global__ static void vqg_dequant_kernel(
        const uint8_t *__restrict__ cb_h,     /* fp16 码本 [nc*dim] */
        const uint8_t *__restrict__ gr_h,     /* fp16 行 gain [rows] */
        const uint8_t *__restrict__ ix,       /* 位流 */
        float *__restrict__ W, int rows, int cols, int dim, int nc, int nbit) {
    extern __shared__ float sh[];
    float *Cs = sh;                            /* nc*dim 码本 fp32 */
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) {
        uint16_t h = (uint16_t)cb_h[i * 2] | ((uint16_t)cb_h[i * 2 + 1] << 8);
        Cs[i] = __half2float(*(const __half *)&h);
    }
    __syncthreads();
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    uint16_t gh = (uint16_t)gr_h[r * 2] | ((uint16_t)gr_h[r * 2 + 1] << 8);
    const float g = __half2float(*(const __half *)&gh);
    const int nidx_row = cols / dim;
    const size_t i0 = (size_t)r * nidx_row;
    float *wr = W + (size_t)r * cols;
    for (int c = 0; c < nidx_row; c++) {
        uint32_t v;
        if (nbit == 8) v = ix[i0 + c];
        else {
            const size_t bit = (i0 + c) * (size_t)nbit;
            uint32_t w0 = ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8) | ((uint32_t)ix[(bit >> 3) + 2] << 16);
            v = (w0 >> (bit & 7)) & ((1u << nbit) - 1u);
        }
        const float *ce = Cs + v * dim;
        for (int d = 0; d < dim; d++) wr[c * dim + d] = g * ce[d];
    }
}

extern "C" int vqg_unpack_dequant(const uint8_t *pay, float *W,
                                  int rows, int cols, int dim, int nc, int nbit) {
    if (nc > VQG_MAX_NC * 2 || dim > VQG_MAX_DIM) return 0;
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)nc * dim * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    cudaStream_t st = vqg_stream();
    size_t shm = (size_t)nc * dim * sizeof(float);
    vqg_dequant_kernel<<<(rows + 127) / 128, 128, shm, st>>>(cb, gr, ix, W, rows, cols, dim, nc, nbit);
    { vqg_last_err("vqg_dequant_kernel后"); }
    return cudaStreamSynchronize(st) == cudaSuccess;
}

/* ---- 批量 dequant(08-18 第三刀): 一层 768 矩阵一次 launch, 消 99k 次 sync ----
 * B 回放段计时实锤: dequant 752s 累计 vs 前向 83s, 7.6ms/调用全是 launch+sync 争用。 */
typedef struct { uint64_t pay_off; uint64_t dst_off; int rows, cols, nc, nbit; } vqg_deq_job;

__global__ static void vqg_dequant_batch_kernel(
        const uint8_t *__restrict__ base, float *__restrict__ dst_base,
        const vqg_deq_job *__restrict__ jobs, int njobs, int dim) {
    const int ji = blockIdx.y;
    if (ji >= njobs) return;
    const vqg_deq_job j = jobs[ji];
    const uint8_t *pay = base + j.pay_off;
    const uint8_t *cb_h = pay + 16;
    const uint8_t *gr_h = cb_h + (size_t)j.nc * dim * 2;
    const uint8_t *ix = gr_h + (size_t)j.rows * 2;
    float *W = dst_base + j.dst_off;
    extern __shared__ float sh[];
    float *Cs = sh;
    for (int i = threadIdx.x; i < j.nc * dim; i += blockDim.x) {
        uint16_t h = (uint16_t)cb_h[i * 2] | ((uint16_t)cb_h[i * 2 + 1] << 8);
        Cs[i] = __half2float(*(const __half *)&h);
    }
    __syncthreads();
    /* ★写地址必须合并★(2026-08-28 实测 bug): 原版是"一个线程负责一整行"——
     * for(r = blockIdx.x*blockDim.x + threadIdx.x; ...) 里层再扫完整行。于是同一 warp 的
     * 32 个线程写的是【相邻的行】, 而一行 = cols(4096) 个 float = 16KB ⇒ 相邻线程地址相隔
     * 16KB, 完全不合并, 每次写都是独立事务。实测整层 26GB 写用 6.69s = ★3.9 GB/s★,
     * 在 GB10 上低了一个数量级, 且它不随抽格行数缩小(行少了权重还是要全量 dequant),
     * 占 bytes_moe 8.3s 的 81%、占单层 11.3s 的 59%。
     * 改为"一个 block 负责一行(blockIdx.x 跨行), threadIdx.x 沿列走": 相邻线程写相邻的
     * dim(=4) 个 float, 16B 步长 ⇒ 每 8 线程凑满一个 128B 事务。
     * ★数值零影响★: 同样的值、同样的目标地址, 只是写入顺序变了; 无归约无竞争(每个
     * (r,c) 由唯一线程独占写)。 */
    const int nidx_row = j.cols / dim;
    for (int r = blockIdx.x; r < j.rows; r += gridDim.x) {
        uint16_t gh = (uint16_t)gr_h[r * 2] | ((uint16_t)gr_h[r * 2 + 1] << 8);
        const float g = __half2float(*(const __half *)&gh);
        const size_t i0 = (size_t)r * nidx_row;
        float *wr = W + (size_t)r * j.cols;
        for (int c = threadIdx.x; c < nidx_row; c += blockDim.x) {
            uint32_t v;
            if (j.nbit == 8) v = ix[i0 + c];
            else {
                const size_t bit = (i0 + c) * (size_t)j.nbit;
                uint32_t w0 = ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8) | ((uint32_t)ix[(bit >> 3) + 2] << 16);
                v = (w0 >> (bit & 7)) & ((1u << j.nbit) - 1u);
            }
            const float *ce = Cs + v * dim;
            for (int d = 0; d < dim; d++) wr[c * dim + d] = g * ce[d];
        }
    }
}

extern "C" int vqg_dequant_batch(const uint8_t *base, float *dst_base,
                                 const vqg_deq_job *jobs_h, int njobs, int dim, int nc_max) {
    static __thread vqg_deq_job *jobs_d = NULL;
    static __thread int jcap = 0;
    if (jcap < njobs) { if (jobs_d) cudaFree(jobs_d); cudaMalloc((void **)&jobs_d, (size_t)njobs * sizeof(vqg_deq_job)); jcap = njobs; }
    cudaStream_t st = vqg_stream();
    cudaMemcpyAsync(jobs_d, jobs_h, (size_t)njobs * sizeof(vqg_deq_job), cudaMemcpyHostToDevice, st);
    size_t shm = (size_t)nc_max * dim * sizeof(float);
    /* grid.x = 并发处理的行数(一 block 一行)。原值 16 是"线程跨行"布局下的分片数,
     * 新布局下它直接决定行并行度, 提到 256 让 SM 吃饱(rows=2048/4096, 循环步进覆盖余下)。 */
    dim3 grid(256, njobs);
    vqg_dequant_batch_kernel<<<grid, 128, shm, st>>>(base, dst_base, jobs_d, njobs, dim);
    { vqg_last_err("vqg_dequant_batch_kernel后"); }
    return cudaStreamSynchronize(st) == cudaSuccess;
}

extern "C" int vqg_host_alloc(void **p, size_t bytes) {   /* 钉页宿主内存: DMA 满带宽(pageable 走 staging 慢 3-5x) */
    return cudaMallocHost(p, bytes) == cudaSuccess;
}
extern "C" void vqg_host_free(void *p) { cudaFreeHost(p); }
extern "C" int vqg_alloc_managed(void **p, size_t bytes) {
    if (cudaMallocManaged(p, bytes) != cudaSuccess) return 0;
    /* ★把首选驻留地钉在 GPU★(2026-08-28): 这块 26GiB 是批 dequant 的目的地, 写方是 GPU
     * kernel、读方也是 GPU(dq_matmul 已改为托管指针直喂 cuBLAS, CPU 不再碰权重)。
     * 不给提示时托管页会按首触方在 CPU/GPU 间来回迁移, 26GiB 逐页搬运 = 实测整层 dequant
     * 5.62s(4.6 GB/s), 比 GB10 该有的带宽低一到两个数量级。
     * SetPreferredLocation + 预取到设备 ⇒ 页常驻 GPU, 免掉每层的迁移往返。
     * ★数值零影响★: 只是页驻留策略, 不改任何值。 */
    int dev = 0; cudaGetDevice(&dev);
    /* CUDA 13 起 cudaMemAdvise/Prefetch 收 cudaMemLocation 结构体(不再是 int device) */
    cudaMemLocation loc; loc.type = cudaMemLocationTypeDevice; loc.id = dev;
    cudaMemAdvise(*p, bytes, cudaMemAdviseSetPreferredLocation, loc);
    cudaMemPrefetchAsync(*p, bytes, loc, 0, (cudaStream_t)0);
    cudaDeviceSynchronize();
    cudaGetLastError();   /* 提示类 API 失败不致命, 清错继续(退化=原迁移行为) */
    return 1;
}

/* attention 整层驻留 GPU 的实现(2026-08-28 拆出): 单 TU 语义, 见 vq_gpu_attn.inc.cu */
#include "vq_gpu_attn.inc.cu"
#include "vq_gpu_moe.inc.cu"
#include "vq_gpu_moefused.inc.cu"   /* VQ-fused 批量专家(dequant 融进 GEMM), 依赖 moe 分片的池与流 */   /* 批量专家前向(sweep GPU 化) */
