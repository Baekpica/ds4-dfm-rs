/* vq_em.cu — VQ 码本 GPU-EM 精修(2026-08-23 用户批准"同体积编码效率, 利用设备特性")。
 *
 * 同体积技术路线: blob 格式/体积一个字节不变(码本 nc×dim f16 + g_r rows f16 + 9bit
 * 索引流原位重写), 引擎零改动 —— 纯把现有 2.25bpw 的比特用得更好:
 *   ① 现有码本/索引/g_r 是一次性拟合的 → EM 交替精修(warm start, 目标单调下降);
 *   ② 目标口径从"权重 MSE"升级为"激活加权 MSE"(列权 = 校准 x 的 E[x²],
 *      对齐输出误差 —— 用户设计"还原率优先"的量化域落法)。
 *
 *   E 步(GPU): 每 block(dim=4 列) 对 nc=512 码字算加权距离 argmin → 索引重指派
 *   M 步(GPU): g_r 闭式 = Σ imp·w·(C∘g?)… 交替:
 *       g_r = Σ_j imp_j·w_j·C[idx]_j / Σ_j imp_j·C[idx]_j²   (每行)
 *       C_cd = Σ_{assigned} imp·g_r·w / Σ imp·g_r²           (每码字每维)
 *
 * 输入: dql_vq_L%02d.bin(原位改写, 先备份) + HF 原权重(st_read/MXFP4) + 锚 fin(列权)
 * 用法: vq_em --hf DIR --layers a-b --dql DIR --anchor F [--iters N=8] [--ntok N=8192]
 * 判据: 每(专家,矩阵)报 加权relL2 前→后; 层级汇总。全层后由 wt2 Σmin 终判。
 *
 * 挖矿③(2026-08-23 用户令"把vq矿挖干净"): VQ-GPTQ 误差补偿(GPTVQ 思路)。
 *   U = chol(H⁻¹) 上三角(H = 锚激活 Gram, 阻尼 1%·mean(diag));
 *   每 block 联合决策 argmin_k ||(w−g·c_k)·inv(U_BB)||²(= 真增量输出损失),
 *   误差前向传播 W_rest -= (E·inv(U_BB))·U[B,rest]; act-order 按块重要性降序
 *   只改处理顺序, 存储序/格式/体积一个字节不变。w1/w3 用 H_x, w2 用 H_h(8 采样专家)。
 *
 * 挖矿④(2026-08-23 用户令"大码本也要"): 码本升格 ×2(LBG 分裂)。
 *   w1/w3 nc 512→1024(9→10bit), w2 256→512(8→9bit); 码本=[旧码字|0.9×旧码字],
 *   旧索引天然有效(指前半), EM 迭代拉开分裂对; blob 重建(体积 87→约95GB, 用户已批)。
 *   诊断依据: GPTQ 在 nc512 上 w1/w3 微输(×1.03)=码本容量覆盖不住传播漂移,
 *   升格后 ③+④ 联动(GPTQ gate 无损, 只在赢时写入)。引擎: 判决走批量 dequant 路
 *   (位宽通用✓); fused2 高速 decode 路 nc1024 特化=后续速度债。 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <pthread.h>
#include <malloc.h>

extern "C" {
#include "st_read_decl.h"
}
#include <ctime>

#define DM   4096
#define MOEI 2048
#define NEXP 256

static void ck(cudaError_t e, const char *m) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

/* fp16 帮助(host) */
static float h2f(uint16_t h) {
    __half x; memcpy(&x, &h, 2); return __half2float(x);
}
static uint16_t f2h(float f) {
    __half x = __float2half(f); uint16_t o; memcpy(&o, &x, 2); return o;
}

/* ---- blob 解析(vq_fmt.h 同构, 本文件自带避免 include 链) ---- */
static inline uint64_t blob_slot(const uint8_t *blob, int e, int which) {
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    return off;
}

/* ══ E 步: 每 block 对 nc 码字加权距离 argmin ══
 * grid: (nblocks/256, ), block 256 线程, 码本+列权进 SMEM。
 * W[rows][cols] f32(dequant HF 真值), g[rows], imp[cols](列权), 输出 idx[nblocks]。 */
__global__ void em_assign_kernel(
        const float *__restrict__ W, const float *__restrict__ g,
        const float *__restrict__ imp, const __half *__restrict__ cb,
        uint16_t *__restrict__ idx_out,
        uint32_t rows, uint32_t cols, uint32_t dim, uint32_t nc) {
    extern __shared__ float sm[];               /* 码本 f32 [nc*dim] */
    const uint32_t bpr = cols / dim;            /* blocks per row */
    const uint32_t nblk = rows * bpr;
    for (uint32_t i = threadIdx.x; i < nc * dim; i += blockDim.x)
        sm[i] = __half2float(cb[i]);
    __syncthreads();
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const uint32_t r = b / bpr, c0 = (b % bpr) * dim;
    const float gr = g[r];
    float w[8], ip[8];
    for (uint32_t d = 0; d < dim; d++) {
        w[d] = W[(size_t)r * cols + c0 + d];
        ip[d] = imp[c0 + d];
    }
    float best = 3.4e38f; uint32_t bi = 0;
    for (uint32_t cc = 0; cc < nc; cc++) {
        const float *cw = sm + (size_t)cc * dim;
        float s = 0;
        for (uint32_t d = 0; d < dim; d++) {
            float e2 = w[d] - gr * cw[d];
            s += ip[d] * e2 * e2;
        }
        if (s < best) { best = s; bi = cc; }
    }
    idx_out[b] = (uint16_t)bi;
}

/* ══ M 步 A: g_r 闭式(每行一 warp 归约) ══ */
__global__ void em_grow_kernel(
        const float *__restrict__ W, const uint16_t *__restrict__ idx,
        const float *__restrict__ imp, const __half *__restrict__ cb,
        float *__restrict__ g_out,
        uint32_t rows, uint32_t cols, uint32_t dim, uint32_t nc) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const uint32_t bpr = cols / dim;
    float num = 0, den = 0;
    for (uint32_t bb = threadIdx.x; bb < bpr; bb += blockDim.x) {
        const uint32_t ci = idx[(size_t)r * bpr + bb];
        for (uint32_t d = 0; d < dim; d++) {
            const float cv = __half2float(cb[(size_t)ci * dim + d]);
            const float wv = W[(size_t)r * cols + bb * dim + d];
            const float ip = imp[bb * dim + d];
            num += ip * wv * cv;
            den += ip * cv * cv;
        }
    }
    __shared__ float sn[256], sd_[256];
    sn[threadIdx.x] = num; sd_[threadIdx.x] = den;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sn[threadIdx.x] += sn[threadIdx.x + s]; sd_[threadIdx.x] += sd_[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) g_out[r] = sd_[0] > 1e-20f ? sn[0] / sd_[0] : g_out[r];
}

/* ══ M 步 B: 码字闭式(атomic 聚合 num/den per (code,dim)) ══ */
__global__ void em_cnum_kernel(
        const float *__restrict__ W, const uint16_t *__restrict__ idx,
        const float *__restrict__ g, const float *__restrict__ imp,
        float *__restrict__ cnum, float *__restrict__ cden,
        uint32_t rows, uint32_t cols, uint32_t dim, uint32_t nc) {
    /* SMEM 局部聚合 + grid-stride: global atomic 从 3400 万高冲突次降到 块数×nc·dim */
    extern __shared__ float smcd[];   /* [nc*dim*2] */
    const uint32_t nd = nc * dim;
    for (uint32_t i = threadIdx.x; i < nd * 2; i += blockDim.x) smcd[i] = 0;
    __syncthreads();
    const uint32_t bpr = cols / dim;
    const uint32_t nblk = rows * bpr;
    for (uint32_t b = blockIdx.x * blockDim.x + threadIdx.x; b < nblk; b += gridDim.x * blockDim.x) {
        const uint32_t r = b / bpr, c0 = (b % bpr) * dim;
        const float gr = g[r];
        const uint32_t ci = idx[b];
        for (uint32_t d = 0; d < dim; d++) {
            const float ip = imp[c0 + d];
            atomicAdd(&smcd[(size_t)ci * dim + d], ip * gr * W[(size_t)r * cols + c0 + d]);
            atomicAdd(&smcd[nd + (size_t)ci * dim + d], ip * gr * gr);
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < nd; i += blockDim.x) {
        if (smcd[i] != 0.0f) atomicAdd(&cnum[i], smcd[i]);
        if (smcd[nd + i] != 0.0f) atomicAdd(&cden[i], smcd[nd + i]);
    }
}

/* ══ M 步 C: 码字更新(device 内, 免 host 往返 —— 段账实锤 EM 96s 全是同步开销) ══ */
__global__ void em_cbup_kernel(__half *cb, const float *cn, const float *cd, uint32_t n) {
    uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n && cd[q] > 1e-12f) cb[q] = __float2half(cn[q] / cd[q]);
}

/* ══ h 行采集: h[t][j] = swiglu(x·w1ᵀ, x·w3ᵀ)(w2 列权 + H_h 共用) ══ */
__global__ void hrow_kernel(
        const float *__restrict__ x, const float *__restrict__ w1,
        const float *__restrict__ w3, float *__restrict__ hout,
        uint32_t ntok, uint32_t din, uint32_t mid) {
    const uint32_t t = blockIdx.y;
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntok || j >= mid) return;
    const float *xt = x + (size_t)t * din;
    const float *w1j = w1 + (size_t)j * din, *w3j = w3 + (size_t)j * din;
    float g = 0, u = 0;
    for (uint32_t d = 0; d < din; d++) { g += xt[d] * w1j[d]; u += xt[d] * w3j[d]; }
    if (g > 10.f) g = 10.f; if (g < -10.f) g = -10.f;   /* swiglu clip 产线口径 ±10 */
    if (u > 10.f) u = 10.f; if (u < -10.f) u = -10.f;
    hout[(size_t)t * mid + j] = (g / (1.f + expf(-g))) * u;
}

/* ══ 转置(tiled) ══ */
__global__ void tr_kernel(const float *__restrict__ A, float *__restrict__ At,
                          uint32_t r, uint32_t c) {
    __shared__ float tl[32][33];
    uint32_t i = blockIdx.y * 32 + threadIdx.y, j = blockIdx.x * 32 + threadIdx.x;
    if (i < r && j < c) tl[threadIdx.y][threadIdx.x] = A[(size_t)i * c + j];
    __syncthreads();
    uint32_t oi = blockIdx.x * 32 + threadIdx.y, oj = blockIdx.y * 32 + threadIdx.x;
    if (oi < c && oj < r) At[(size_t)oi * r + oj] = tl[threadIdx.x][threadIdx.y];
}

/* ══ Gram: H = Xt·Xtᵀ, Xt=[n][m](行=特征) ══ */
__global__ void gram_kernel(const float *__restrict__ Xt, float *__restrict__ H,
                            uint32_t n, uint32_t m) {
    __shared__ float ta[16][17], tb[16][17];
    const uint32_t i = blockIdx.y * 16 + threadIdx.y;
    const uint32_t j = blockIdx.x * 16 + threadIdx.x;
    float s = 0;
    for (uint32_t t0 = 0; t0 < m; t0 += 16) {
        const uint32_t ii = blockIdx.y * 16 + threadIdx.y;
        const uint32_t jj = blockIdx.x * 16 + threadIdx.y;
        ta[threadIdx.y][threadIdx.x] = (ii < n && t0 + threadIdx.x < m) ? Xt[(size_t)ii * m + t0 + threadIdx.x] : 0.f;
        tb[threadIdx.y][threadIdx.x] = (jj < n && t0 + threadIdx.x < m) ? Xt[(size_t)jj * m + t0 + threadIdx.x] : 0.f;
        __syncthreads();
        for (int t = 0; t < 16; t++) s += ta[threadIdx.y][t] * tb[threadIdx.x][t];
        __syncthreads();
    }
    if (i < n && j < n) H[(size_t)i * n + j] = s;
}

/* ══ GPTQ 前评估: 旧索引未补偿误差的 Hessian 口径损失 Σ‖E·V_p‖²(对表用) ══ */
__global__ void gptq_eval_kernel(
        const float *__restrict__ W, const uint16_t *__restrict__ idx,
        const float *__restrict__ g, const __half *__restrict__ cb,
        const float *__restrict__ V, const int *__restrict__ bperm,
        double *__restrict__ loss, uint32_t rows, uint32_t cols) {
    __shared__ double sl[256];
    const uint32_t nb = cols / 4, tot = rows * nb;
    const uint32_t bidx = blockIdx.x * blockDim.x + threadIdx.x;
    double s = 0;
    if (bidx < tot) {
        const uint32_t r = bidx / nb, p = bidx % nb, bs = bperm[p];
        const float *Vp = V + (size_t)p * 16;
        const float gr = g[r];
        const uint32_t k = idx[(size_t)r * nb + bs];
        float e0 = W[(size_t)r * cols + bs * 4 + 0] - gr * __half2float(cb[k * 4 + 0]);
        float e1 = W[(size_t)r * cols + bs * 4 + 1] - gr * __half2float(cb[k * 4 + 1]);
        float e2 = W[(size_t)r * cols + bs * 4 + 2] - gr * __half2float(cb[k * 4 + 2]);
        float e3 = W[(size_t)r * cols + bs * 4 + 3] - gr * __half2float(cb[k * 4 + 3]);
        float t0 = e0 * Vp[0];
        float t1 = e0 * Vp[1] + e1 * Vp[5];
        float t2 = e0 * Vp[2] + e1 * Vp[6] + e2 * Vp[10];
        float t3 = e0 * Vp[3] + e1 * Vp[7] + e2 * Vp[11] + e3 * Vp[15];
        s = (double)t0 * t0 + (double)t1 * t1 + (double)t2 * t2 + (double)t3 * t3;
    }
    sl[threadIdx.x] = s;
    __syncthreads();
    for (uint32_t st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sl[threadIdx.x] += sl[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss, sl[0]);
}

/* ══ GPTQ 主 kernel: 16 行/块, 置换序逐 block {argmin(V度量) → 误差传播} ══
 * 数学: err·U_BB = E 的解 err = E·inv(U_BB); W_rest -= err·U[B,rest];
 * 每 block 损失 = ‖err‖² = ‖E·V‖² 即 argmin 度量 —— 决策即最小化真增量损失。
 * ★带宽手术(2026-08-23 399s/层→目标<150s)★: 单行/块时 U 上三角(33MB)每行块各自
 * 从 DRAM 流一遍(~68GB/slot); 16 行共块后传播读 U[d][c] 一次服务 16 行, 流量÷16。
 * 布局: 256 线程 = argmin 期 16 线程/行 × 16 行; 传播期 256 线程跨列, 行在寄存器循环。 */
#define GQ_ROWS 16u
__global__ void gptq_kernel(
        float *__restrict__ W, const __half *__restrict__ cb,
        const float *__restrict__ g, const float *__restrict__ U,
        const float *__restrict__ V, const int *__restrict__ bperm,
        uint16_t *__restrict__ idx_out, double *__restrict__ loss_after,
        uint32_t rows, uint32_t cols, uint32_t nc) {
    const uint32_t r0 = blockIdx.x * GQ_ROWS;
    if (r0 >= rows) return;
    const uint32_t nr = (rows - r0 < GQ_ROWS) ? rows - r0 : GQ_ROWS;
    const uint32_t nb = cols / 4;
    const uint32_t lr = threadIdx.x / 16u;     /* argmin 期: 本线程管的行 */
    const uint32_t lt = threadIdx.x % 16u;     /* 行内 16 线程分摊 nc 码字 */
    __shared__ float sw[GQ_ROWS][4], seh[GQ_ROWS][4], sgr[GQ_ROWS];
    __shared__ float sbest[GQ_ROWS][16]; __shared__ uint32_t sbi[GQ_ROWS][16];
    __shared__ double srl;
    if (threadIdx.x == 0) srl = 0;
    if (threadIdx.x < nr) sgr[threadIdx.x] = g[r0 + threadIdx.x];
    __syncthreads();
    for (uint32_t p = 0; p < nb; p++) {
        const uint32_t bs = bperm[p];
        const float *Vp = V + (size_t)p * 16;
        if (threadIdx.x < nr * 4u)
            sw[threadIdx.x >> 2][threadIdx.x & 3] = W[(size_t)(r0 + (threadIdx.x >> 2)) * cols + bs * 4 + (threadIdx.x & 3)];
        __syncthreads();
        /* argmin: 行 lr, 16 线程跨码字 */
        if (lr < nr) {
            const float gr = sgr[lr];
            const float w0 = sw[lr][0], w1 = sw[lr][1], w2 = sw[lr][2], w3 = sw[lr][3];
            float best = 3.4e38f; uint32_t bi = 0;
            for (uint32_t k = lt; k < nc; k += 16u) {
                const float e0 = w0 - gr * __half2float(cb[k * 4 + 0]);
                const float e1 = w1 - gr * __half2float(cb[k * 4 + 1]);
                const float e2 = w2 - gr * __half2float(cb[k * 4 + 2]);
                const float e3 = w3 - gr * __half2float(cb[k * 4 + 3]);
                const float t0 = e0 * Vp[0];
                const float t1 = e0 * Vp[1] + e1 * Vp[5];
                const float t2 = e0 * Vp[2] + e1 * Vp[6] + e2 * Vp[10];
                const float t3 = e0 * Vp[3] + e1 * Vp[7] + e2 * Vp[11] + e3 * Vp[15];
                const float sd = t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
                if (sd < best) { best = sd; bi = k; }
            }
            sbest[lr][lt] = best; sbi[lr][lt] = bi;
        }
        __syncthreads();
        /* 行内 16 元 reduce + 终选(每行线程 0) */
        if (lt == 0 && lr < nr) {
            float best = sbest[lr][0]; uint32_t bi = sbi[lr][0];
            for (uint32_t t = 1; t < 16u; t++)
                if (sbest[lr][t] < best) { best = sbest[lr][t]; bi = sbi[lr][t]; }
            idx_out[(size_t)(r0 + lr) * nb + bs] = (uint16_t)bi;
            atomicAdd(&srl, (double)best);
            const float gr = sgr[lr];
            const float e0 = sw[lr][0] - gr * __half2float(cb[bi * 4 + 0]);
            const float e1 = sw[lr][1] - gr * __half2float(cb[bi * 4 + 1]);
            const float e2 = sw[lr][2] - gr * __half2float(cb[bi * 4 + 2]);
            const float e3 = sw[lr][3] - gr * __half2float(cb[bi * 4 + 3]);
            seh[lr][0] = e0 * Vp[0];
            seh[lr][1] = e0 * Vp[1] + e1 * Vp[5];
            seh[lr][2] = e0 * Vp[2] + e1 * Vp[6] + e2 * Vp[10];
            seh[lr][3] = e0 * Vp[3] + e1 * Vp[7] + e2 * Vp[11] + e3 * Vp[15];
        }
        __syncthreads();
        /* 传播: 256 线程跨列, U[0..3][c] 读一次 → 寄存器行循环 16 行共用 */
        const float *U0 = U + (size_t)(p * 4 + 0) * cols, *U1 = U + (size_t)(p * 4 + 1) * cols;
        const float *U2 = U + (size_t)(p * 4 + 2) * cols, *U3 = U + (size_t)(p * 4 + 3) * cols;
        for (uint32_t c = p * 4 + 4 + threadIdx.x; c < cols; c += blockDim.x) {
            const uint32_t sc = (uint32_t)bperm[c >> 2] * 4 + (c & 3);
            const float u0 = U0[c], u1 = U1[c], u2 = U2[c], u3 = U3[c];
            for (uint32_t r = 0; r < nr; r++)
                W[(size_t)(r0 + r) * cols + sc] -= seh[r][0] * u0 + seh[r][1] * u1 + seh[r][2] * u2 + seh[r][3] * u3;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_after, srl);
}

/* ════ host: GPTQ 因子链(分块 Cholesky, pthread) ════ */
typedef struct { void (*fn)(int, int, void *); void *ud; int t, nt; } pf2_arg;
static void *pf2_run(void *v) { pf2_arg *a = (pf2_arg *)v; a->fn(a->t, a->nt, a->ud); return NULL; }
static void pfor2(void (*fn)(int, int, void *), void *ud) {
    enum { NT2 = 20 };
    pthread_t th[NT2]; pf2_arg pa[NT2]; int made[NT2];
    for (int t = 0; t < NT2; t++) {
        pa[t] = (pf2_arg){fn, ud, t, NT2};
        made[t] = pthread_create(&th[t], NULL, pf2_run, &pa[t]) == 0;
        if (!made[t]) pf2_run(&pa[t]);
    }
    for (int t = 0; t < NT2; t++) if (made[t]) pthread_join(th[t], NULL);
}

static struct { float *A; int n, k, kb; } chs;
static void ch_solve_rows(int t, int nt, void *ud) {
    (void)ud; const int n = chs.n, k = chs.k, kb = chs.kb; float *A = chs.A;
    for (int i = k + kb + t; i < n; i += nt) {
        float *Ai = A + (size_t)i * n;
        for (int j = k; j < k + kb; j++) {
            const float *Aj = A + (size_t)j * n;
            float sv = Ai[j];
            for (int p = k; p < j; p++) sv -= Ai[p] * Aj[p];
            Ai[j] = sv / Aj[j];
        }
    }
}
static void ch_trail(int t, int nt, void *ud) {
    (void)ud; const int n = chs.n, k = chs.k, kb = chs.kb; float *A = chs.A;
    for (int i = k + kb + t; i < n; i += nt) {
        float *Ai = A + (size_t)i * n;
        for (int j = k + kb; j <= i; j++) {
            const float *Aj = A + (size_t)j * n;
            float sv = 0;
            for (int p = k; p < k + kb; p++) sv += Ai[p] * Aj[p];
            Ai[j] -= sv;
        }
    }
}
static int chol_lower(float *A, int n) {   /* 右视分块, 下三角 in-place; 0=成功 */
    const int NB = 128;
    chs.A = A; chs.n = n;
    for (int k = 0; k < n; k += NB) {
        const int kb = n - k < NB ? n - k : NB;
        for (int j = k; j < k + kb; j++) {
            float *Aj = A + (size_t)j * n;
            float sv = Aj[j];
            for (int p = k; p < j; p++) sv -= Aj[p] * Aj[p];
            if (sv <= 0) return -1;
            Aj[j] = sqrtf(sv);
            for (int i = j + 1; i < k + kb; i++) {
                float *Ai = A + (size_t)i * n;
                float s2 = Ai[j];
                for (int p = k; p < j; p++) s2 -= Ai[p] * Aj[p];
                Ai[j] = s2 / Aj[j];
            }
        }
        chs.k = k; chs.kb = kb;
        if (k + kb < n) { pfor2(ch_solve_rows, NULL); pfor2(ch_trail, NULL); }
    }
    return 0;
}

static struct { const float *L; float *MT; int n; } tis;
static void ti_cols(int t, int nt, void *ud) {   /* MT 行 j = L⁻¹ 列 j */
    (void)ud; const int n = tis.n; const float *L = tis.L;
    for (int j = t; j < n; j += nt) {
        float *m = tis.MT + (size_t)j * n;
        for (int k = 0; k < j; k++) m[k] = 0;
        m[j] = 1.0f / L[(size_t)j * n + j];
        for (int i = j + 1; i < n; i++) {
            const float *Li = L + (size_t)i * n;
            float sv = 0;
            for (int k = j; k < i; k++) sv += Li[k] * m[k];
            m[i] = -sv / Li[i];
        }
    }
}
static struct { const float *MT; float *A; int n; } mms;
static void mm_rows(int t, int nt, void *ud) {   /* A = MᵀM = H_p⁻¹ */
    (void)ud; const int n = mms.n;
    for (int i = t; i < n; i += nt) {
        const float *Mi = mms.MT + (size_t)i * n;
        for (int j = 0; j <= i; j++) {
            const float *Mj = mms.MT + (size_t)j * n;
            float sv = 0;
            for (int k = i; k < n; k++) sv += Mi[k] * Mj[k];
            mms.A[(size_t)i * n + j] = sv;
            mms.A[(size_t)j * n + i] = sv;
        }
    }
}
static struct { const float *H0; float *Hp; const int *cm; int n; float damp; } hps;
static void hp_rows(int t, int nt, void *ud) {
    (void)ud; const int n = hps.n;
    for (int a = t; a < n; a += nt) {
        const float *Hr = hps.H0 + (size_t)hps.cm[a] * n;
        float *Pr = hps.Hp + (size_t)a * n;
        for (int b = 0; b < n; b++) Pr[b] = Hr[hps.cm[b]];
        Pr[a] += hps.damp;
    }
}
static struct { const float *L2; float *U; int n; } uts;
static void ut_rows(int t, int nt, void *ud) {   /* U = L2ᵀ 上三角(行主) */
    (void)ud; const int n = uts.n;
    for (int i = t; i < n; i += nt) {
        float *Ui = uts.U + (size_t)i * n;
        for (int j = 0; j < i; j++) Ui[j] = 0;
        for (int j = i; j < n; j++) Ui[j] = uts.L2[(size_t)j * n + i];
    }
}

static int g_estep_gptvq = 1;   /* gate 口径分支: gptvq=V(输出空间), em=relL2 */
static const double *qs_bi;
static int qs_cmp(const void *a, const void *b) {
    const double x = qs_bi[*(const int *)a], y = qs_bi[*(const int *)b];
    return x < y ? 1 : (x > y ? -1 : 0);
}

typedef struct { float *U; float *V; int *bperm; int n, nb; } gctx;
/* H0(原空间 Gram) → act-order 置换 + 阻尼 → U=chol(H_p⁻¹)ᵀ + V_p=inv(U_BB) */
static int build_gptq(const float *H0, int n, gctx *G) {
    const int nb = n / 4;
    G->n = n; G->nb = nb;
    G->bperm = (int *)malloc(nb * 4);
    double *bi = (double *)malloc(nb * 8);
    for (int b = 0; b < nb; b++) {
        double sv = 0;
        for (int d = 0; d < 4; d++) sv += H0[(size_t)(b * 4 + d) * n + b * 4 + d];
        bi[b] = sv; G->bperm[b] = b;
    }
    qs_bi = bi;
    qsort(G->bperm, nb, 4, qs_cmp);   /* act-order: 块 diag(H) 降序 */
    int *cm = (int *)malloc(n * 4);
    for (int p = 0; p < nb; p++)
        for (int d = 0; d < 4; d++) cm[p * 4 + d] = G->bperm[p] * 4 + d;
    double md = 0;
    for (int i = 0; i < n; i++) md += H0[(size_t)i * n + i];
    md /= n;
    float *Hp = (float *)malloc((size_t)n * n * 4);
    int ok = -1; float damp = 0;
    for (int att = 0; att < 3 && ok != 0; att++) {   /* 阻尼 1%→10%→100% 直到正定 */
        damp = (float)(md * 0.01 * pow(10.0, att));
        hps.H0 = H0; hps.Hp = Hp; hps.cm = cm; hps.n = n; hps.damp = damp;
        pfor2(hp_rows, NULL);
        ok = chol_lower(Hp, n);
    }
    if (ok != 0) { fprintf(stderr, "GPTQ chol 不正定(n=%d)\n", n); return -1; }
    float *MT = (float *)malloc((size_t)n * n * 4);
    tis.L = Hp; tis.MT = MT; tis.n = n;
    pfor2(ti_cols, NULL);
    float *A = (float *)malloc((size_t)n * n * 4);
    mms.MT = MT; mms.A = A; mms.n = n;
    pfor2(mm_rows, NULL);
    if (chol_lower(A, n) != 0) { fprintf(stderr, "GPTQ chol2 失败(n=%d)\n", n); return -1; }
    G->U = (float *)malloc((size_t)n * n * 4);
    uts.L2 = A; uts.U = G->U; uts.n = n;
    pfor2(ut_rows, NULL);
    /* V_p = inv(U_BB) 上三角 4×4(行主 16 float, 下三角置 0) */
    G->V = (float *)calloc((size_t)nb * 16, 4);
    for (int p = 0; p < nb; p++) {
        float B[16], *Vp = G->V + (size_t)p * 16;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                B[i * 4 + j] = (j >= i) ? G->U[(size_t)(p * 4 + i) * n + p * 4 + j] : 0.f;
        for (int i = 3; i >= 0; i--) {
            Vp[i * 4 + i] = 1.0f / B[i * 4 + i];
            for (int j = i + 1; j < 4; j++) {
                float sv = 0;
                for (int k = i + 1; k <= j; k++) sv += B[i * 4 + k] * Vp[k * 4 + j];
                Vp[i * 4 + j] = -sv / B[i * 4 + i];
            }
        }
    }
    /* 自检: 随机 v, w = Hp_damped·(UᵀU·v) ≈ v(H_p 已被 chol 摧毁 → 重 gather) */
    {
        hps.H0 = H0; hps.Hp = Hp; hps.cm = cm; hps.n = n; hps.damp = damp;
        pfor2(hp_rows, NULL);
        float *v = (float *)malloc(n * 4), *y = (float *)calloc(n, 4), *w = (float *)calloc(n, 4);
        unsigned sd = 12345;
        for (int i = 0; i < n; i++) { sd = sd * 1103515245u + 12345u; v[i] = ((sd >> 16) & 1023) / 512.0f - 1.0f; }
        /* y = UᵀU v: t = U v(上三角), y = Uᵀ t */
        float *tv = (float *)calloc(n, 4);
        for (int i = 0; i < n; i++) { const float *Ui = G->U + (size_t)i * n; float sv = 0; for (int j = i; j < n; j++) sv += Ui[j] * v[j]; tv[i] = sv; }
        for (int j = 0; j < n; j++) { float sv = 0; for (int i = 0; i <= j; i++) sv += G->U[(size_t)i * n + j] * tv[i]; y[j] = sv; }
        for (int i = 0; i < n; i++) { const float *Hi = Hp + (size_t)i * n; float sv = 0; for (int j = 0; j < n; j++) sv += Hi[j] * y[j]; w[i] = sv; }
        double e2 = 0, a2 = 0;
        for (int i = 0; i < n; i++) { e2 += (double)(w[i] - v[i]) * (w[i] - v[i]); a2 += (double)v[i] * v[i]; }
        fprintf(stderr, "  GPTQ 因子自检 n=%d: ‖H·H⁻¹v−v‖/‖v‖=%.2e (阻尼=%.3g)\n", n, sqrt(e2 / a2), damp);
        free(v); free(y); free(w); free(tv);
    }
    free(bi); free(cm); free(Hp); free(MT); free(A);
    return 0;
}

/* ══ 评估: 加权 relL2 ══ */
__global__ void em_eval_kernel(
        const float *__restrict__ W, const uint16_t *__restrict__ idx,
        const float *__restrict__ g, const float *__restrict__ imp,
        const __half *__restrict__ cb,
        double *__restrict__ e2, double *__restrict__ a2,
        uint32_t rows, uint32_t cols, uint32_t dim) {
    /* ★atomic 冲突修复(段账实锤 EM 99s)★: 原版 210 万线程 atomicAdd 同 2 地址=全串行。
     * SMEM 归约后每 block 仅 1 次 global atomic。 */
    __shared__ double sse[256], ssa[256];
    const uint32_t bpr = cols / dim;
    const uint32_t nblk = rows * bpr;
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    double se = 0, sa = 0;
    if (b < nblk) {
        const uint32_t r = b / bpr, c0 = (b % bpr) * dim;
        const float gr = g[r];
        const uint32_t ci = idx[b];
        for (uint32_t d = 0; d < dim; d++) {
            const float wv = W[(size_t)r * cols + c0 + d];
            const float qv = gr * __half2float(cb[(size_t)ci * dim + d]);
            const float ip = imp[c0 + d];
            se += (double)ip * (wv - qv) * (wv - qv);
            sa += (double)ip * wv * wv;
        }
    }
    sse[threadIdx.x] = se; ssa[threadIdx.x] = sa;
    __syncthreads();
    for (uint32_t st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) { sse[threadIdx.x] += sse[threadIdx.x + st]; ssa[threadIdx.x] += ssa[threadIdx.x + st]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(e2, sse[0]); atomicAdd(a2, ssa[0]); }
}

/* ---- 9bit 位流 读/写(host, 分段并行: 段界按 8 索引=9 字节对齐, 写无竞争) ---- */
typedef struct { const uint8_t *ix; uint8_t *ixw; const uint16_t *in; uint16_t *out;
                 uint32_t i0, i1; int nbit; } bits_arg;
static void *bits_r_worker(void *va) {
    bits_arg *p = (bits_arg *)va;
    for (uint32_t i = p->i0; i < p->i1; i++) {
        size_t bit = (size_t)i * p->nbit;
        uint32_t w = (uint32_t)p->ix[bit >> 3] | ((uint32_t)p->ix[(bit >> 3) + 1] << 8) |
                     ((uint32_t)p->ix[(bit >> 3) + 2] << 16);
        p->out[i] = (uint16_t)((w >> (bit & 7)) & ((1u << p->nbit) - 1u));
    }
    return NULL;
}
static void *bits_w_worker(void *va) {
    bits_arg *p = (bits_arg *)va;
    for (uint32_t i = p->i0; i < p->i1; i++) {
        size_t bit = (size_t)i * p->nbit;
        uint32_t v = (uint32_t)p->in[i] & ((1u << p->nbit) - 1u);
        p->ixw[bit >> 3] |= (uint8_t)(v << (bit & 7));
        p->ixw[(bit >> 3) + 1] |= (uint8_t)(v >> (8 - (bit & 7)));
        if ((bit & 7) + p->nbit > 16)
            p->ixw[(bit >> 3) + 2] |= (uint8_t)(v >> (16 - (bit & 7)));
    }
    return NULL;
}
/* ---- 9bit 位流 读/写(host) ---- */
static void bits_read(const uint8_t *ix, uint32_t nidx, int nbit, uint16_t *out) {
    if (nbit == 8) { for (uint32_t i = 0; i < nidx; i++) out[i] = ix[i]; return; }
    const int nth = 16;
    pthread_t th[16]; bits_arg pa[16];
    uint32_t per = ((nidx / 8 + nth - 1) / nth) * 8;   /* 段界 8 索引对齐 */
    int cnt = 0;
    for (int t = 0; t < nth; t++) {
        uint32_t i0 = t * per, i1 = i0 + per > nidx ? nidx : i0 + per;
        if (i0 >= i1) break;
        pa[cnt] = (bits_arg){ix, NULL, NULL, out, i0, i1, nbit};
        if (pthread_create(&th[cnt], NULL, bits_r_worker, &pa[cnt])) { bits_r_worker(&pa[cnt]); continue; }
        cnt++;
    }
    for (int t = 0; t < cnt; t++) pthread_join(th[t], NULL);
}
static void bits_write(uint8_t *ix, uint32_t nidx, int nbit, const uint16_t *in) {
    if (nbit == 8) { for (uint32_t i = 0; i < nidx; i++) ix[i] = (uint8_t)in[i]; return; }
    size_t nbytes = ((size_t)nidx * nbit + 7) / 8 + 1;
    memset(ix, 0, nbytes);
    const int nth = 16;
    pthread_t th[16]; bits_arg pa[16];
    uint32_t per = ((nidx / 8 + nth - 1) / nth) * 8;   /* 8 索引=9 字节整段, 写无竞争 */
    int cnt = 0;
    for (int t = 0; t < nth; t++) {
        uint32_t i0 = t * per, i1 = i0 + per > nidx ? nidx : i0 + per;
        if (i0 >= i1) break;
        pa[cnt] = (bits_arg){NULL, ix, in, NULL, i0, i1, nbit};
        if (pthread_create(&th[cnt], NULL, bits_w_worker, &pa[cnt])) { bits_w_worker(&pa[cnt]); continue; }
        cnt++;
    }
    for (int t = 0; t < cnt; t++) pthread_join(th[t], NULL);
}

int main(int argc, char **argv) {
    const char *hf = NULL, *dql = NULL, *anchor = NULL, *layers = NULL;
    int iters = 8; long long ntok = 8192; int estep_gptvq = 1;
    #define ESTEP_SYNC() (g_estep_gptvq = estep_gptvq)   /* E步: 1=gptvq(补偿指派) 0=em(独立argmin) */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--hf") && i + 1 < argc) hf = argv[++i];
        else if (!strcmp(argv[i], "--dql") && i + 1 < argc) dql = argv[++i];
        else if (!strcmp(argv[i], "--anchor") && i + 1 < argc) anchor = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ntok") && i + 1 < argc) ntok = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--estep") && i + 1 < argc) estep_gptvq = !strcmp(argv[++i], "gptvq");
    
    }
    if (!hf || !dql || !anchor || !layers) {
        fprintf(stderr, "用法: vq_em --hf DIR --dql DIR --anchor F --layers a-b [--iters N]\n");
        return 2;
    }
    int l0, l1;
    if (sscanf(layers, "%d-%d", &l0, &l1) != 2) l0 = l1 = atoi(layers);
    g_estep_gptvq = estep_gptvq;

    /* mallopt(量化器同款实锤): st_read 每矩阵 malloc/free 33MB → 默认走 mmap =
     * 每次全新零页+逐 4KB 首触缺页(19GB/层), 单线程用户态吃掉大头。放开 top_pad
     * 后堆复用, 量化器实测 164s→65s。 */
    mallopt(M_TOP_PAD, 256 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 1024 * 1024 * 1024);
    mallopt(M_MMAP_THRESHOLD, 1024 * 1024 * 1024);
    void *SC = stb_open(hf);

    /* 锚头 + fin 段偏移 */
    FILE *af = fopen(anchor, "rb");
    if (!af) { fprintf(stderr, "锚打不开\n"); return 2; }
    uint32_t hd[8]; if (fread(hd, 4, 8, af) != 8 || hd[0] != 0x32415144u) { fprintf(stderr, "非 DQA2\n"); return 2; }
    const long long S = hd[1], NL = hd[4];
    if (ntok > S) ntok = S;

    /* ★8-stream 批量流水(2026-08-23 用户令"把 spark 性能用好")★:
     * 单矩阵 8k blocks 填不满 GB10 且矩阵间串行 —— 8 路并发把上传/EM/回传全异步重叠。 */
    #define NS 8
    cudaStream_t st_[NS];
    float *dW_[NS], *dG_[NS], *dImp_[NS], *dCn_[NS], *dCd_[NS]; uint16_t *dIdx_[NS];
    __half *dCb_[NS]; double *dE_[NS];
    for (int q = 0; q < NS; q++) {
        ck(cudaStreamCreate(&st_[q]), "st");
        ck(cudaMalloc(&dW_[q], (size_t)MOEI * DM * 4), "W");
        ck(cudaMalloc(&dG_[q], (size_t)DM * 4), "G");
        ck(cudaMalloc(&dImp_[q], (size_t)DM * 4), "imp");
        ck(cudaMalloc(&dIdx_[q], (size_t)MOEI * DM / 4 * 2), "idx");
        ck(cudaMalloc(&dCb_[q], (size_t)512 * 8 * 2), "cb");
        ck(cudaMalloc(&dCn_[q], (size_t)512 * 8 * 4), "cn");
        ck(cudaMalloc(&dCd_[q], (size_t)512 * 8 * 4), "cd");
        ck(cudaMalloc(&dE_[q], 8 * sizeof(double)), "e");
    }
    /* pinned host W 缓冲池: 免 per-矩阵上传同步(那个 sync 等的是本 slot 整条 EM 链,
     * 把 8-stream 流水打回串行 —— L20 对拍 108s 实锤) */
    float *hW_[NS];
    for (int q = 0; q < NS; q++) ck(cudaHostAlloc(&hW_[q], (size_t)MOEI * DM * 4, cudaHostAllocDefault), "hW");

    for (int L = l0; L <= l1; L++) {
        time_t t0 = time(NULL);
        /* blob 读入(整层) + 备份 */
        char bp[1024], bk[1024];
        snprintf(bp, sizeof bp, "%s/dql_vq_L%02d.bin", dql, L);
        snprintf(bk, sizeof bk, "%s/dql_vq_L%02d.bin.pre_em", dql, L);
        struct stat stb;
        if (stat(bk, &stb) != 0) { char cmd[3072]; snprintf(cmd, sizeof cmd, "cp %s %s", bp, bk); if (system(cmd)) { fprintf(stderr, "备份失败\n"); return 3; } }
        FILE *bf = fopen(bp, "rb");
        if (!bf) { fprintf(stderr, "%s 打不开\n", bp); return 3; }
        fseek(bf, 0, SEEK_END); long bsz = ftell(bf); fseek(bf, 0, SEEK_SET);
        uint8_t *blob = (uint8_t *)malloc(bsz);
        if (fread(blob, 1, bsz, bf) != (size_t)bsz) { fprintf(stderr, "blob 读断\n"); return 3; }
        fclose(bf);

        /* ══ 挖矿④: 码本升格 ×2(LBG 分裂) + 索引重编码 + blob 重建 ══ */
        {
            size_t nsz = 16 + (size_t)NEXP * 3 * 8;
            nsz = (nsz + 7) & ~(size_t)7;
            for (int e = 0; e < NEXP; e++) for (int wh = 0; wh < 3; wh++) {
                uint64_t off = blob_slot(blob, e, wh);
                if (!off) continue;
                uint8_t *pay = blob + off;
                uint16_t dim, nc; uint32_t rows, cols;
                memcpy(&dim, pay + 4, 2); memcpy(&nc, pay + 6, 2);
                memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
                uint32_t nc2 = (dim == 4 && nc < 1024) ? (uint32_t)nc * 2 : nc;
                int nb2 = 0; while ((1 << nb2) < (int)nc2) nb2++;
                const uint32_t nblk = rows * (cols / dim);
                nsz += 16 + (size_t)nc2 * dim * 2 + (size_t)rows * 2 + ((size_t)nblk * nb2 + 7) / 8 + 1;
                nsz = (nsz + 7) & ~(size_t)7;
            }
            uint8_t *nb_ = (uint8_t *)calloc(nsz, 1);
            memcpy(nb_, blob, 16);
            uint64_t *vt2 = (uint64_t *)(nb_ + 16);
            size_t cur = 16 + (size_t)NEXP * 3 * 8;
            cur = (cur + 7) & ~(size_t)7;
            for (int e = 0; e < NEXP; e++) for (int wh = 0; wh < 3; wh++) {
                uint64_t off = blob_slot(blob, e, wh);
                if (!off) { vt2[(size_t)e * 3 + wh] = 0; continue; }
                uint8_t *pay = blob + off;
                uint16_t dim, nc; uint32_t rows, cols;
                memcpy(&dim, pay + 4, 2); memcpy(&nc, pay + 6, 2);
                memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
                uint16_t nc2 = (dim == 4 && nc < 1024) ? nc * 2 : nc;
                int nbo = 0; while ((1 << nbo) < (int)nc) nbo++;
                int nb2 = 0; while ((1 << nb2) < (int)nc2) nb2++;
                const uint32_t nblk = rows * (cols / dim);
                const size_t paysz = 16 + (size_t)nc2 * dim * 2 + (size_t)rows * 2 + ((size_t)nblk * nb2 + 7) / 8 + 1;
                uint8_t *np = nb_ + cur;
                memcpy(np, pay, 4);            /* magic 'DQVQ' */
                memcpy(np + 4, &dim, 2); memcpy(np + 6, &nc2, 2);
                memcpy(np + 8, &rows, 4); memcpy(np + 12, &cols, 4);
                uint8_t *ncb = np + 16;
                memcpy(ncb, pay + 16, (size_t)nc * dim * 2);   /* 前半 = 旧码字(旧索引天然有效) */
                if (nc2 > nc)                                  /* 后半 = 0.9×旧(LBG 分裂对, EM 拉开) */
                    for (uint32_t i = 0; i < (uint32_t)nc * dim; i++) {
                        uint16_t h; memcpy(&h, pay + 16 + 2 * (size_t)i, 2);
                        uint16_t h2 = f2h(0.9f * h2f(h));
                        memcpy(ncb + (size_t)nc * dim * 2 + 2 * (size_t)i, &h2, 2);
                    }
                memcpy(ncb + (size_t)nc2 * dim * 2, pay + 16 + (size_t)nc * dim * 2, (size_t)rows * 2);   /* g 原样 */
                if (nb2 != nbo) {
                    uint16_t *tmp = (uint16_t *)malloc((size_t)nblk * 2);
                    bits_read(pay + 16 + (size_t)nc * dim * 2 + (size_t)rows * 2, nblk, nbo, tmp);
                    bits_write(ncb + (size_t)nc2 * dim * 2 + (size_t)rows * 2, nblk, nb2, tmp);
                    free(tmp);
                } else {
                    memcpy(ncb + (size_t)nc2 * dim * 2 + (size_t)rows * 2,
                           pay + 16 + (size_t)nc * dim * 2 + (size_t)rows * 2,
                           ((size_t)nblk * nbo + 7) / 8 + 1);
                }
                vt2[(size_t)e * 3 + wh] = cur;
                cur += paysz;
                cur = (cur + 7) & ~(size_t)7;
            }
            free(blob);
            blob = nb_;
            bsz = (long)cur;
            fprintf(stderr, "L%d 码本升格: blob %.2fGB(nc×2, 9→10bit/8→9bit)\n", L, bsz / 1e9);
        }

        /* 列权: 锚 fin 该层 E[x²](w1/w3 用); w2 用均匀(第一版) */
        float *impx = (float *)calloc(DM, sizeof(float));
        {
            long long fin_off = 40 + (long long)L * S * DM * 4;
            fseek(af, (long)fin_off, SEEK_SET);
            float *row = (float *)malloc(DM * 4);
            for (long long t = 0; t < ntok; t++) {
                if (fread(row, 4, DM, af) != DM) break;
                for (int j = 0; j < DM; j++) impx[j] += row[j] * row[j];
            }
            free(row);
            double mi = 0;
            for (int j = 0; j < DM; j++) mi += impx[j];
            mi /= DM;
            for (int j = 0; j < DM; j++) impx[j] = (float)(impx[j] / mi + 0.30);   /* 归一+floor
                * (0.05→0.30, 2026-08-23: 终判 KLD/p95 退化=罕见方向被降权过狠, 尾部保护) */
        }
        float *impu = (float *)malloc(DM * 4);
        for (int j = 0; j < DM; j++) impu[j] = 1.0f;
        /* ══ GPTQ 前奏: H_x(锚 4096 行 Gram) + H_h/imp_h(8 采样专家 h 行) ══ */
        float *imp_h = (float *)malloc(MOEI * 4);
        gctx GX, GH;
        {
            const int NTH = 4096, NTX = 512;
            float *hx = (float *)malloc((size_t)NTH * DM * 4);
            long long fin_off2 = 40 + (long long)L * S * DM * 4;
            fseek(af, (long)fin_off2, SEEK_SET);
            if (fread(hx, 4, (size_t)NTH * DM, af) != (size_t)NTH * DM) { fprintf(stderr, "锚x读断\n"); exit(3); }
            float *dXr, *dXT, *dHx, *dW1, *dW3, *dHr, *dHT, *dHh;
            ck(cudaMalloc(&dXr, (size_t)NTH * DM * 4), "xr");
            ck(cudaMalloc(&dXT, (size_t)DM * NTH * 4), "xt");
            ck(cudaMalloc(&dHx, (size_t)DM * DM * 4), "hx");
            ck(cudaMalloc(&dW1, (size_t)MOEI * DM * 4), "w1h");
            ck(cudaMalloc(&dW3, (size_t)MOEI * DM * 4), "w3h");
            ck(cudaMalloc(&dHr, (size_t)8 * NTX * MOEI * 4), "hr");
            ck(cudaMalloc(&dHT, (size_t)MOEI * 8 * NTX * 4), "ht");
            ck(cudaMalloc(&dHh, (size_t)MOEI * MOEI * 4), "hh");
            ck(cudaMemcpy(dXr, hx, (size_t)NTH * DM * 4, cudaMemcpyHostToDevice), "xr up");
            dim3 gt1((DM + 31) / 32, (NTH + 31) / 32), bt(32, 32);
            tr_kernel<<<gt1, bt>>>(dXr, dXT, NTH, DM);
            dim3 gg1((DM + 15) / 16, (DM + 15) / 16), bg(16, 16);
            gram_kernel<<<gg1, bg>>>(dXT, dHx, DM, NTH);
            /* 8 采样专家 h 行(w2 列权 + H_h 共用) */
            const char *wns[2] = {"w1", "w3"};
            int nsamp = 0;
            for (int es = 0; es < 256; es += 32) {
                float *wbuf[2] = {NULL, NULL};
                int okr = 1;
                for (int k2 = 0; k2 < 2; k2++) {
                    char nm2[256]; long R2, C2;
                    snprintf(nm2, sizeof nm2, "layers.%d.ffn.experts.%d.%s.weight", L, es, wns[k2]);
                    wbuf[k2] = stb_read(SC, nm2, &R2, &C2);
                    if (!wbuf[k2]) okr = 0;
                }
                if (okr) {
                    ck(cudaMemcpy(dW1, wbuf[0], (size_t)MOEI * DM * 4, cudaMemcpyHostToDevice), "w1 up");
                    ck(cudaMemcpy(dW3, wbuf[1], (size_t)MOEI * DM * 4, cudaMemcpyHostToDevice), "w3 up");
                    dim3 gh_((MOEI + 127) / 128, NTX);
                    hrow_kernel<<<gh_, 128>>>(dXr, dW1, dW3, dHr + (size_t)nsamp * NTX * MOEI, NTX, DM, MOEI);
                    nsamp++;
                }
                free(wbuf[0]); free(wbuf[1]);
            }
            const int nh = nsamp * NTX;
            dim3 gt2((MOEI + 31) / 32, (nh + 31) / 32);
            tr_kernel<<<gt2, bt>>>(dHr, dHT, nh, MOEI);
            dim3 gg2((MOEI + 15) / 16, (MOEI + 15) / 16);
            gram_kernel<<<gg2, bg>>>(dHT, dHh, MOEI, nh);
            ck(cudaDeviceSynchronize(), "gram sync");
            float *Hx = (float *)malloc((size_t)DM * DM * 4);
            float *Hh = (float *)malloc((size_t)MOEI * MOEI * 4);
            ck(cudaMemcpy(Hx, dHx, (size_t)DM * DM * 4, cudaMemcpyDeviceToHost), "hx dn");
            ck(cudaMemcpy(Hh, dHh, (size_t)MOEI * MOEI * 4, cudaMemcpyDeviceToHost), "hh dn");
            cudaFree(dXr); cudaFree(dXT); cudaFree(dHx); cudaFree(dW1); cudaFree(dW3);
            cudaFree(dHr); cudaFree(dHT); cudaFree(dHh);
            free(hx);
            /* imp_h = diag(H_h) 归一 + floor(挖矿①②) */
            double mh = 0;
            for (int j = 0; j < MOEI; j++) mh += Hh[(size_t)j * MOEI + j];
            mh /= MOEI;
            for (int j = 0; j < MOEI; j++) imp_h[j] = (float)(Hh[(size_t)j * MOEI + j] / (mh + 1e-30) + 0.30);
            fprintf(stderr, "L%d w2列权+Gram 就绪(采样专家=%d)\n", L, nsamp);
            /* CPU 因子链(w1/w3 用 H_x, w2 用 H_h); EM 模式不需要 */
            memset(&GX, 0, sizeof GX); memset(&GH, 0, sizeof GH);
            if (estep_gptvq && (build_gptq(Hx, DM, &GX) || build_gptq(Hh, MOEI, &GH))) { fprintf(stderr, "GPTQ 因子失败\n"); exit(6); }
            free(Hx); free(Hh);
        }
        /* 因子上载(整层驻留) */
        float *dUx = NULL, *dUh = NULL, *dVx = NULL, *dVh = NULL; int *dBx = NULL, *dBh = NULL;
        if (estep_gptvq) {
        ck(cudaMalloc(&dUx, (size_t)DM * DM * 4), "Ux");
        ck(cudaMalloc(&dUh, (size_t)MOEI * MOEI * 4), "Uh");
        ck(cudaMalloc(&dVx, (size_t)(DM / 4) * 16 * 4), "Vx");
        ck(cudaMalloc(&dVh, (size_t)(MOEI / 4) * 16 * 4), "Vh");
        ck(cudaMalloc(&dBx, (size_t)(DM / 4) * 4), "Bx");
        ck(cudaMalloc(&dBh, (size_t)(MOEI / 4) * 4), "Bh");
        ck(cudaMemcpy(dUx, GX.U, (size_t)DM * DM * 4, cudaMemcpyHostToDevice), "Ux up");
        ck(cudaMemcpy(dUh, GH.U, (size_t)MOEI * MOEI * 4, cudaMemcpyHostToDevice), "Uh up");
        ck(cudaMemcpy(dVx, GX.V, (size_t)(DM / 4) * 16 * 4, cudaMemcpyHostToDevice), "Vx up");
        ck(cudaMemcpy(dVh, GH.V, (size_t)(MOEI / 4) * 16 * 4, cudaMemcpyHostToDevice), "Vh up");
        ck(cudaMemcpy(dBx, GX.bperm, (size_t)(DM / 4) * 4, cudaMemcpyHostToDevice), "Bx up");
        ck(cudaMemcpy(dBh, GH.bperm, (size_t)(MOEI / 4) * 4, cudaMemcpyHostToDevice), "Bh up");
        }

        double tot_e0 = 0, tot_e1 = 0, tot_a = 0, gq_b = 0, gq_a = 0; int gq_skip = 0;
        /* 流水收口记录: 每 slot 一个在途矩阵 */
        struct pend_t { int live; uint8_t *grp, *cbp, *ixp; uint32_t rows, nblk; int nbit;
                        uint32_t ncd; float *gh; uint16_t *ih; uint16_t cbh[512*8]; double eh[8]; } pend[NS];
        memset(pend, 0, sizeof(pend));
        const char *wn[3] = {"w1", "w3", "w2"};
        for (int e = 0; e < NEXP; e++) {
            for (int wh = 0; wh < 3; wh++) {
                uint64_t off = blob_slot(blob, e, wh);
                if (!off) continue;
                uint8_t *pay = blob + off;
                uint16_t dim, nc; uint32_t rows, cols;
                memcpy(&dim, pay + 4, 2); memcpy(&nc, pay + 6, 2);
                memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
                if (dim != 4) continue;
                if ((uint32_t)nc * dim > 4096) { fprintf(stderr, "nc%u×dim%u 超缓冲\n", nc, dim); return 4; }
                uint8_t *cbp = pay + 16;
                uint8_t *grp = cbp + (size_t)nc * dim * 2;
                uint8_t *ixp = grp + (size_t)rows * 2;
                int nbit = 0; while ((1 << nbit) < nc) nbit++;
                const uint32_t bpr = cols / dim, nblk = rows * bpr;

                const int q = (e * 3 + wh) % NS;
                cudaStream_t sq = st_[q];
                /* 收口该 slot 的上一个在途矩阵 */
                if (pend[q].live) {
                    ck(cudaStreamSynchronize(sq), "sync");
                    struct pend_t *P = &pend[q];
                    const double pe0 = P->eh[0], pa0 = P->eh[1], pe1 = P->eh[4];
                    const int vwin = g_estep_gptvq ? (P->eh[3] < P->eh[2]) : (pe1 < pe0);   /* gate 口径随 E 步 */
                    if (vwin) {
                        for (uint32_t r = 0; r < P->rows; r++) ((uint16_t *)P->grp)[r] = f2h(P->gh[r]);
                        memcpy(P->cbp, P->cbh, (size_t)P->ncd * 2);
                        bits_write(P->ixp, P->nblk, P->nbit, P->ih);
                    }
                    tot_e0 += pe0; tot_e1 += (vwin ? pe1 : pe0); tot_a += pa0;
                    gq_b += P->eh[2]; gq_a += (vwin ? P->eh[3] : P->eh[2]); if (!vwin) gq_skip++;
                    free(P->gh); free(P->ih); P->live = 0;
                }
                /* HF 真值 */
                char nm[256];
                snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.%s.weight", L, e, wn[wh]);
                long R, C;
                float *Wf = stb_read(SC, nm, &R, &C);
                /* 拷入本 slot pinned 缓冲(收口已同步=可安全复用), Wf 立即释放, 上传全异步 */
                if (!Wf) { fprintf(stderr, "HF %s 缺\n", nm); return 4; }
                if ((uint32_t)R != rows) { /* 转置口径 */ long t2 = R; R = C; C = t2; }
                memcpy(hW_[q], Wf, (size_t)rows * cols * 4);
                free(Wf);
                ck(cudaMemcpyAsync(dW_[q], hW_[q], (size_t)rows * cols * 4, cudaMemcpyHostToDevice, sq), "W up");

                /* 现有 g_r/码本/索引 上传(异步) */
                float *gh = (float *)malloc(rows * 4);
                for (uint32_t r = 0; r < rows; r++) gh[r] = h2f(((uint16_t *)grp)[r]);
                ck(cudaMemcpyAsync(dG_[q], gh, rows * 4, cudaMemcpyHostToDevice, sq), "g up");
                ck(cudaMemcpyAsync(dCb_[q], cbp, (size_t)nc * dim * 2, cudaMemcpyHostToDevice, sq), "cb up");
                uint16_t *ih = (uint16_t *)malloc((size_t)nblk * 2);
                bits_read(ixp, nblk, nbit, ih);
                ck(cudaMemcpyAsync(dIdx_[q], ih, (size_t)nblk * 2, cudaMemcpyHostToDevice, sq), "idx up");
                const float *impsel = (wh == 2) ? imp_h : impx;   /* w2=E[h²] 列权(挖矿①) */
                ck(cudaMemcpyAsync(dImp_[q], impsel, cols * 4, cudaMemcpyHostToDevice, sq), "imp up");

                /* ══ 完整 GPTVQ(挖矿⑤终形态): E 步=补偿指派(gptq_kernel, 每轮原始 W 起步),
                 * M 步=补偿后 W 上闭式 g/码本 —— 补偿感知的自洽平衡。
                 * (串联"EM 完再 GPTQ"/轮间联动实测恒输 skip=766-768, 机制=破坏 EM 平衡)
                 * 口径: gate 用 Hessian(V 度量, =输出空间, 与 Σmin 同向) eh[2]旧 vs eh[3]新;
                 * relL2 双报表 eh[0][1]基线 eh[4][5]终值。 */
                const float *dUs = wh < 2 ? dUx : dUh;
                const float *dVs = wh < 2 ? dVx : dVh;
                const int *dBs = wh < 2 ? dBx : dBh;
                ck(cudaMemsetAsync(dE_[q], 0, 64, sq), "e0");
                em_eval_kernel<<<(nblk + 255) / 256, 256, 0, sq>>>(dW_[q], dIdx_[q], dG_[q], dImp_[q], dCb_[q], dE_[q], dE_[q] + 1, rows, cols, dim);
                if (estep_gptvq) {
                    gptq_eval_kernel<<<(nblk + 255) / 256, 256, 0, sq>>>(dW_[q], dIdx_[q], dG_[q], dCb_[q], dVs, dBs, dE_[q] + 2, rows, cols);
                    for (int it = 0; it < iters; it++) {
                        ck(cudaMemcpyAsync(dW_[q], hW_[q], (size_t)rows * cols * 4, cudaMemcpyHostToDevice, sq), "W re");
                        gptq_kernel<<<(rows + GQ_ROWS - 1) / GQ_ROWS, 256, 0, sq>>>(
                            dW_[q], dCb_[q], dG_[q], dUs, dVs, dBs, dIdx_[q], (double *)dCn_[q], rows, cols, nc);
                        em_grow_kernel<<<rows, 256, 0, sq>>>(dW_[q], dIdx_[q], dImp_[q], dCb_[q], dG_[q], rows, cols, dim, nc);
                        ck(cudaMemsetAsync(dCn_[q], 0, (size_t)nc * dim * 4, sq), "cn0");
                        ck(cudaMemsetAsync(dCd_[q], 0, (size_t)nc * dim * 4, sq), "cd0");
                        em_cnum_kernel<<<128, 256, (size_t)nc * dim * 2 * 4, sq>>>(dW_[q], dIdx_[q], dG_[q], dImp_[q], dCn_[q], dCd_[q], rows, cols, dim, nc);
                        em_cbup_kernel<<<((uint32_t)nc * dim + 255) / 256, 256, 0, sq>>>(dCb_[q], dCn_[q], dCd_[q], (uint32_t)nc * dim);
                    }
                    /* 终指派(终稿码本, 原始 W 起步) + 双口径终评 */
                    ck(cudaMemcpyAsync(dW_[q], hW_[q], (size_t)rows * cols * 4, cudaMemcpyHostToDevice, sq), "W re2");
                    gptq_kernel<<<(rows + GQ_ROWS - 1) / GQ_ROWS, 256, 0, sq>>>(
                        dW_[q], dCb_[q], dG_[q], dUs, dVs, dBs, dIdx_[q], dE_[q] + 3, rows, cols, nc);
                    ck(cudaMemcpyAsync(dW_[q], hW_[q], (size_t)rows * cols * 4, cudaMemcpyHostToDevice, sq), "W re3");
                } else {
                    for (int it = 0; it < iters; it++) {
                        em_assign_kernel<<<(nblk + 255) / 256, 256, (size_t)nc * dim * 4, sq>>>(
                            dW_[q], dG_[q], dImp_[q], dCb_[q], dIdx_[q], rows, cols, dim, nc);
                        em_grow_kernel<<<rows, 256, 0, sq>>>(dW_[q], dIdx_[q], dImp_[q], dCb_[q], dG_[q], rows, cols, dim, nc);
                        ck(cudaMemsetAsync(dCn_[q], 0, (size_t)nc * dim * 4, sq), "cn0");
                        ck(cudaMemsetAsync(dCd_[q], 0, (size_t)nc * dim * 4, sq), "cd0");
                        em_cnum_kernel<<<128, 256, (size_t)nc * dim * 2 * 4, sq>>>(dW_[q], dIdx_[q], dG_[q], dImp_[q], dCn_[q], dCd_[q], rows, cols, dim, nc);
                        em_cbup_kernel<<<((uint32_t)nc * dim + 255) / 256, 256, 0, sq>>>(dCb_[q], dCn_[q], dCd_[q], (uint32_t)nc * dim);
                    }
                }
                em_eval_kernel<<<(nblk + 255) / 256, 256, 0, sq>>>(dW_[q], dIdx_[q], dG_[q], dImp_[q], dCb_[q], dE_[q] + 4, dE_[q] + 5, rows, cols, dim);
                /* 异步回传(收口时消费) */
                ck(cudaMemcpyAsync(pend[q].eh, dE_[q], 64, cudaMemcpyDeviceToHost, sq), "e dn");
                ck(cudaMemcpyAsync(gh, dG_[q], rows * 4, cudaMemcpyDeviceToHost, sq), "g dn");
                ck(cudaMemcpyAsync(pend[q].cbh, dCb_[q], (size_t)nc * dim * 2, cudaMemcpyDeviceToHost, sq), "cb dn");
                ck(cudaMemcpyAsync(ih, dIdx_[q], (size_t)nblk * 2, cudaMemcpyDeviceToHost, sq), "idx dn");
                pend[q].live = 1; pend[q].grp = grp; pend[q].cbp = cbp; pend[q].ixp = ixp;
                pend[q].rows = rows; pend[q].nblk = nblk; pend[q].nbit = nbit;
                pend[q].ncd = (uint32_t)nc * dim;
                pend[q].gh = gh; pend[q].ih = ih;
            }
            if ((e & 31) == 31) { fprintf(stderr, "  L%d EM …%d/256 (relL2 %.5f→%.5f)\r", L, e + 1, sqrt(tot_e0 / (tot_a + 1e-30)), sqrt(tot_e1 / (tot_a + 1e-30))); fflush(stderr); }
        }
        /* 层尾: 收口全部在途 slot */
        for (int q = 0; q < NS; q++) {
            if (!pend[q].live) continue;
            ck(cudaStreamSynchronize(st_[q]), "final sync");
            struct pend_t *P = &pend[q];
            const double pe0 = P->eh[0], pa0 = P->eh[1], pe1 = P->eh[4];
            const int vwin = g_estep_gptvq ? (P->eh[3] < P->eh[2]) : (pe1 < pe0);
            if (vwin) {
                for (uint32_t r = 0; r < P->rows; r++) ((uint16_t *)P->grp)[r] = f2h(P->gh[r]);
                memcpy(P->cbp, P->cbh, (size_t)P->ncd * 2);
                bits_write(P->ixp, P->nblk, P->nbit, P->ih);
            }
            tot_e0 += pe0; tot_e1 += (vwin ? pe1 : pe0); tot_a += pa0;
            gq_b += P->eh[2]; gq_a += (vwin ? P->eh[3] : P->eh[2]); if (!vwin) gq_skip++;
            free(P->gh); free(P->ih); P->live = 0;
        }
        fprintf(stderr, "\n");
        if (dUx) { cudaFree(dUx); cudaFree(dUh); cudaFree(dVx); cudaFree(dVh); cudaFree(dBx); cudaFree(dBh); }
        free(GX.U); free(GX.V); free(GX.bperm); free(GH.U); free(GH.V); free(GH.bperm);
        /* 整层写回盘(升格后尺寸变 → 全新写) */
        {
            FILE *wf = fopen(bp, "wb");
            if (!wf || fwrite(blob, 1, bsz, wf) != (size_t)bsz) { fprintf(stderr, "写回失败\n"); return 5; }
            fclose(wf);
        }

        printf("★L%d %s: Hes损失(V口径) ×%.4f skip=%d | relL2 %.5f→%.5f | iters=%d %lds\n",
               L, g_estep_gptvq ? "GPTVQ" : "升格EM", sqrt(gq_a / (gq_b + 1e-30)), gq_skip,
               sqrt(tot_e0 / (tot_a + 1e-30)), sqrt(tot_e1 / (tot_a + 1e-30)), iters,
               (long)(time(NULL) - t0));
        fflush(stdout);
        free(blob); free(impx); free(impu); free(imp_h);
    }
    fclose(af);
    return 0;
}
