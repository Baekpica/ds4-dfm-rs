/* v41_vq.cu — V4.1 专家权重的向量量化(VQ), GPU 实现(2026-09-11; 09-12 加落盘编码/解码)。
 *
 * 【为什么走 VQ 而不是标量】
 *   ① 表达力: 多维联合编码能突破标量的率失真界(同 bpw 下)。
 *   ② ★省 scale★: 标量档必须付 8/32 = 0.25 bpw 养每 32 列一个的 ue8m0 scale;
 *      VQ 只付每行一个 f16 行增益 = 0.003 bpw。前提是行内动态范围小 —— 实测 V4.1 专家的
 *      行内 scale 跨度只有平均 2×、最坏 4×(L0/L20/L39 一致), 单行增益完全覆盖得住。
 *
 * 【配方】dim=8, nc=4096 ⇒ 12 bit 索引 / 8 元素 = 1.5 bpw + 行增益 0.003。
 * 码本粒度 = 每专家一份(w1/w3/w2 共享), 64 KB/份 × 15360 = 983 MB ⇒ +0.014 bpw。
 *
 * 【落盘布局(09-12)】每矩阵: 索引流 U8 [rows, ceil(nidx_row·bits/8)] —— 逐行字节对齐、
 * LSB 先; 行增益 F16 [rows]; 每专家码本 F16 [nc, dim]。解码 = cb[idx]·gain, f32 算。
 * ★量化器内部的重建统计走同一个 unpack_decode 核, 吃的是【打包后的字节】★ —— 报出来的
 * 残差就是部署解出来的残差, 不存在"内部好看、落地劣化"的口子。
 *
 * 【确定性】码本初值 = 数据等距采样(无随机源), 迭代轮数固定; M 步用 double 原子累加
 * (float 原子加的顺序差会让码本末位抖动, 两次跑同一专家出不同字节 —— 铁律"产物可复现")。
 * 码本/行增益先舍入到 f16 再做最终 assign, 所以文件里的值与 assign 时用的值逐位相同。
 *
 * 【★为什么不用仓里的 vqg_assign★】它把码本整份塞 shared memory, nc=4096/dim=8 要 144 KB,
 * GB10 opt-in 上限 101376 字节。这里码本分块进 shared, 每线程在寄存器维护 argmin。
 * 【DIM 必须是编译期常量】运行期 dim 的 float v[16] 带动态索引会落 local memory, 慢 17 倍。 */
#include <cuda_fp16.h>
#include <stdlib.h>
#include "v41_dq.cuh"
#include "v41_vq_rate.h"

#define CK V41_CK
/* 率侧探针(2026-09-21): 索引熵 / ECVQ 惩罚 / 分块定宽 / 码本 E4M3 舍入。只读索引或只动码本舍入, λ=0 时一个核不发。 */
#include "v41_vq_rate.inc.cu"

__global__ static void row_gain_kernel(const float *w, float *g, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float *p = w + (long long)r * cols;
    float s = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) s += p[i] * p[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float v = sqrtf(sh[0] / cols);
        g[r] = v > 0.f ? v : 1.f;     /* 全零行: 增益取 1, 后面码字自然落到零附近 */
    }
}

__global__ static void scale_rows_kernel(float *w, const float *g, int rows, int cols, int inv) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)rows * cols;
    if (i >= n) return;
    float gg = g[i / cols];
    w[i] = inv ? w[i] / gg : w[i] * gg;
}

/* ★融合 assign★: 距离矩阵不落地(首版 cublas 算 G=V·Cᵀ 再 argmin, 访存 144 GB/专家)。
 * CHUNK=512 ⇒ shared = 512*dim*4 + 512*4 = 18 KB(dim=8)。 */
#define VQ_CHUNK 512
#define VQ_MAXDIM 16
/* 列权(校准路, 2026-09-20 深夜; 09-11 只在最终 assign 用过, 现在训练也吃): 向量的【全局号】gi = i0 + i·stride, 第 d 分量对应
 * 权重矩阵的列号 → 列权。前 nv13 个向量是 w1/w3(输入 x, 周期 = D), 其后是 w2(输入 h, 周期 = MID); 指针 NULL ⇒ 该段平权。
 * 为什么按全局号算相位: 09-11 训练走平权的唯一理由是"stride 采样后列对应关系被打乱" —— 采样只是把全局号乘了 stride, 相位算得回来。 */
typedef struct { const float *x; int px; long long nv13; const float *h; int ph; } vq_cw;
template <int DIM>
__device__ __forceinline__ static void vq_load_w(const vq_cw cw, long long gi, float *w) {
    if (gi < cw.nv13) {
        const long long b = gi * DIM;
#pragma unroll
        for (int d = 0; d < DIM; d++) w[d] = cw.x ? cw.x[(int)((b + d) % cw.px)] : 1.f;
    } else {
        const long long b = (gi - cw.nv13) * DIM;
#pragma unroll
        for (int d = 0; d < DIM; d++) w[d] = cw.h ? cw.h[(int)((b + d) % cw.ph)] : 1.f;
    }
}
/* PEN=1: ECVQ, 距离加码长惩罚 pen[k](见 v41_vq_rate.inc.cu); PEN=0 与 09-12 起的核一字不差(nocal 产物逐字节不变)。 */
template <int DIM, int PEN>
__global__ static void assign_fused_kernel(const float *__restrict__ V, const float *__restrict__ C,
                                           const float *__restrict__ pen, long long nv, int nc,
                                           int *__restrict__ idx, const vq_cw cw, long long i0, int stride) {
    extern __shared__ float sh[];
    float *shC = sh, *shN = sh + VQ_CHUNK * DIM;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float v[DIM], w[DIM];
#pragma unroll
    for (int d = 0; d < DIM; d++) v[d] = (i < nv) ? V[i * DIM + d] : 0.f;
    vq_load_w<DIM>(cw, i0 + i * stride, w);
    float bd = 3.0e38f; int best = 0;
    for (int k0 = 0; k0 < nc; k0 += VQ_CHUNK) {
        int n = nc - k0 < VQ_CHUNK ? nc - k0 : VQ_CHUNK;
        for (int t = threadIdx.x; t < n * DIM; t += blockDim.x) shC[t] = C[(long long)k0 * DIM + t];
        if (PEN) for (int t = threadIdx.x; t < n; t += blockDim.x) shN[t] = pen[k0 + t];
        __syncthreads();
        if (i < nv) {
            for (int k = 0; k < n; k++) {
                float dist = PEN ? shN[k] : 0.f;
#pragma unroll
                for (int d = 0; d < DIM; d++) { float e = v[d] - shC[k * DIM + d]; dist += w[d] * e * e; }
                if (dist < bd) { bd = dist; best = k0 + k; }
            }
        }
        __syncthreads();
    }
    if (i < nv) idx[i] = best;
}

/* pen = NULL ⇒ 纯 Lloyd 核(PEN=0); 非 NULL ⇒ ECVQ 核 */
static int vq_assign(const float *V, const float *C, const float *pen, long long nv, int nc, int dim,
                     int *idx, size_t shb, const vq_cw cw, long long i0, int stride) {
    unsigned g = (unsigned)((nv + 255) / 256);
#define VQ_ASSIGN(D) (pen ? assign_fused_kernel<D, 1><<<g, 256, shb>>>(V, C, pen, nv, nc, idx, cw, i0, stride) \
                          : assign_fused_kernel<D, 0><<<g, 256, shb>>>(V, C, pen, nv, nc, idx, cw, i0, stride))
    switch (dim) {
        case 4:  VQ_ASSIGN(4); break;
        case 8:  VQ_ASSIGN(8); break;
        case 16: VQ_ASSIGN(16); break;
        default: fprintf(stderr, "★dim=%d 未特化(只支持 4/8/16)★\n", dim); return -1;
    }
#undef VQ_ASSIGN
    return 0;
}

/* M 步: double 原子累加(见文件头"确定性") */
__global__ static void accum_kernel(const float *V, const int *idx, int nv, int dim, double *sum, int *cnt) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv) return;
    int k = idx[i];
    atomicAdd(&cnt[k], 1);
    const float *v = V + i * (long long)dim;
    for (int d = 0; d < dim; d++) atomicAdd(&sum[k * dim + d], (double)v[d]);
}

/* 空簇: 保持原码字不动(不随机重播种) —— 确定性优先, 空簇只是浪费一个码字 */
__global__ static void update_kernel(float *C, const double *sum, const int *cnt, int nc, int dim) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nc) return;
    int n = cnt[k];
    if (n <= 0) return;
    for (int d = 0; d < dim; d++) C[k * dim + d] = (float)(sum[k * dim + d] / n);
}

/* 加权 M 步(校准路): 列权随向量相位变, 同一码字下不同向量的第 d 分量权重不同 ⇒ 质心按维算 Σw·v / Σw。
 * 平权时数学上退化成上面的 accum/update(Σw = 计数), 但为了 nocal 产物逐字节可复现, 平权仍走老核。 */
template <int DIM>
__global__ static void accum_w_kernel(const float *V, const int *idx, int nv, const vq_cw cw, int stride, double *sum, double *wsum) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv) return;
    int k = idx[i];
    float w[DIM]; vq_load_w<DIM>(cw, i * stride, w);
    const float *v = V + i * (long long)DIM;
#pragma unroll
    for (int d = 0; d < DIM; d++) { atomicAdd(&sum[k * DIM + d], (double)(w[d] * v[d])); atomicAdd(&wsum[k * DIM + d], (double)w[d]); }
}
static int vq_accum_w(const float *V, const int *idx, long long nv, int dim, const vq_cw cw, int stride, double *sum, double *wsum) {
    unsigned g = (unsigned)((nv + 255) / 256);
    switch (dim) {
        case 4:  accum_w_kernel<4><<<g, 256>>>(V, idx, (int)nv, cw, stride, sum, wsum); break;
        case 8:  accum_w_kernel<8><<<g, 256>>>(V, idx, (int)nv, cw, stride, sum, wsum); break;
        case 16: accum_w_kernel<16><<<g, 256>>>(V, idx, (int)nv, cw, stride, sum, wsum); break;
        default: return -1;
    }
    return 0;
}
__global__ static void update_w_kernel(float *C, const double *sum, const double *wsum, int nc, int dim) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nc) return;
    for (int d = 0; d < dim; d++) { const double ws = wsum[k * dim + d]; if (ws > 0) C[k * dim + d] = (float)(sum[k * dim + d] / ws); }
}

/* 加权残差账(校准路报数用): Σ w_col·(a−b)² 与 Σ w_col·a², 列权按列号取(NULL = 平权 = 与 v41_sse_kernel 同数) */
__global__ static void wsse_kernel(const float *a, const float *b, long long n, int cols, const float *colw, double *acc) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double se[256], sn[256];
    double d = 0, e = 0;
    if (i < n) { const float wc = colw ? colw[(int)(i % cols)] : 1.f; const float x = a[i], y = b[i]; d = (double)wc * (x - y) * (x - y); e = (double)wc * x * x; }
    se[threadIdx.x] = d; sn[threadIdx.x] = e;
    __syncthreads();
    for (int o = blockDim.x / 2; o; o >>= 1) {
        if (threadIdx.x < o) { se[threadIdx.x] += se[threadIdx.x + o]; sn[threadIdx.x] += sn[threadIdx.x + o]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(&acc[0], se[0]); atomicAdd(&acc[1], sn[0]); }
}

__global__ static void init_kernel(const float *V, float *C, int nv, int nc, int dim) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nc) return;
    long long src = (long long)k * nv / nc;
    for (int d = 0; d < dim; d++) C[k * dim + d] = V[src * dim + d];
}

__global__ static void gather_stride_kernel(const float *V, float *out, int ntr, int dim, int stride) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntr) return;
    for (int d = 0; d < dim; d++) out[i * dim + d] = V[i * (long long)stride * dim + d];
}

__global__ static void decode_kernel(float *V, const int *idx, const float *C, int nv, int dim) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nv * dim) return;
    V[i] = C[idx[i / dim] * dim + (int)(i % dim)];
}

/* f16 往返: x 变成 f16 可表示的值, bits 存位型 —— 文件里的值与后续计算用的值逐位相同 */
__global__ static void round_f16_kernel(float *x, uint16_t *bits, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half h = __float2half_rn(x[i]);
    x[i] = __half2float(h);
    bits[i] = __half_as_ushort(h);
}

/* 逐行字节对齐的位流打包: 一个线程产出一个字节(无写冲突 ⇒ 确定性) */
__global__ static void pack_kernel(const int *idx, uint8_t *out, int rows, int nidx_row, int bits, int bytes_row) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)rows * bytes_row) return;
    int r = (int)(t / bytes_row), b = (int)(t % bytes_row);
    const int *ri = idx + (long long)r * nidx_row;
    unsigned v = 0;
    for (int k = 0; k < 8; k++) {
        int bp = b * 8 + k, ii = bp / bits, bb = bp % bits;
        if (ii < nidx_row && ((ri[ii] >> bb) & 1)) v |= 1u << k;
    }
    out[t] = (uint8_t)v;
}

/* 解包 + 解码: out[r][k*dim+d] = f16(cb[idx][d]) × f16(gain[r]), f32 算。部署与量化器内部统计同用。 */
__global__ static void unpack_decode_kernel(const uint8_t *packed, const uint16_t *cb, const uint16_t *gain,
                                            int rows, int cols, int dim, int bits, int bytes_row, float *out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int nidx_row = cols / dim;
    if (i >= (long long)rows * nidx_row) return;
    int r = (int)(i / nidx_row), k = (int)(i % nidx_row);
    const uint8_t *rp = packed + (long long)r * bytes_row;
    unsigned v = 0;
    for (int b = 0; b < bits; b++) { int p = k * bits + b; if ((rp[p >> 3] >> (p & 7)) & 1) v |= 1u << b; }
    float g = __half2float(__ushort_as_half(gain[r]));
    float *o = out + (long long)r * cols + (long long)k * dim;
    for (int d = 0; d < dim; d++) o[d] = __half2float(__ushort_as_half(cb[v * dim + d])) * g;
}

static int vq_bits(int nc) { int b = 1; while ((1 << b) < nc) b++; return b; }

/* 训练码本(采样集上 Lloyd 交替)并做全量最终 assign。V 已归一化(行增益已除)。
 * cw = 两段列权(x 给前 nv13 个向量, h 给其后; 见 vq_load_w), 都 NULL ⇒ 全平权走老核(nocal 产物逐字节不变)。
 * 率侧探针(2026-09-21): lam > 0 = ECVQ(每轮按本轮划分重估码长, 下一轮与最终指派带惩罚); fixed = C 已是给定码本(共享码本),
 * 不初始化不更新, 只为算惩罚走一遍采样指派; cb_fp8 = 码本舍到 E4M3 格点; rate 非 NULL 回填最终指派的率统计。 */
static int vq_fit_assign(float *V, long long nv, int dim, int nc, int iters, int stride,
                         float *C, int *idx, const vq_cw cw, int round_cb, uint16_t *cbbits,
                         float lam, int fixed, int cb_fp8, v41_vq_rate *rate) {
    const int weighted = (cw.x != NULL) || (cw.h != NULL);
    float *pen = NULL; double *sum, *wsum = NULL; int *cnt;
    CK(cudaMalloc(&sum, sizeof(double) * nc * dim));
    CK(cudaMalloc(&cnt, sizeof(int) * nc));
    if (weighted) CK(cudaMalloc(&wsum, sizeof(double) * nc * dim));
    if (lam > 0.f) { CK(cudaMalloc(&pen, sizeof(float) * nc)); CK(cudaMemset(pen, 0, sizeof(float) * nc)); }
    size_t SHB = (size_t)VQ_CHUNK * dim * sizeof(float) + VQ_CHUNK * sizeof(float);
    if (stride < 1) stride = 1;
    long long ntr = nv / stride;
    float *Vtr = V;
    if (stride > 1) {
        CK(cudaMalloc(&Vtr, sizeof(float) * ntr * dim));
        gather_stride_kernel<<<(unsigned)((ntr + 255) / 256), 256>>>(V, Vtr, (int)ntr, dim, stride);
    }
    if (!fixed) init_kernel<<<(nc + 255) / 256, 256>>>(Vtr, C, (int)ntr, nc, dim);
    const vq_cw none = {NULL, 1, 0, NULL, 1};
    const int nit = fixed ? (pen ? 1 : 0) : iters;
    for (int it = 0; it < nit; it++) {
        if (weighted) {   /* 采样向量 i 的全局号 = i·stride ⇒ 列相位按全局号算, 训练与最终 assign 用同一套权 */
            if (vq_assign(Vtr, C, pen, ntr, nc, dim, idx, SHB, cw, 0, stride)) return -1;
            if (!fixed) {
                CK(cudaMemset(sum, 0, sizeof(double) * nc * dim));
                CK(cudaMemset(wsum, 0, sizeof(double) * nc * dim));
                if (vq_accum_w(Vtr, idx, ntr, dim, cw, stride, sum, wsum)) return -1;
                update_w_kernel<<<(nc + 255) / 256, 256>>>(C, sum, wsum, nc, dim);
            }
        } else {
            if (vq_assign(Vtr, C, pen, ntr, nc, dim, idx, SHB, none, 0, 1)) return -1;
            if (!fixed) {
                CK(cudaMemset(sum, 0, sizeof(double) * nc * dim));
                CK(cudaMemset(cnt, 0, sizeof(int) * nc));
                accum_kernel<<<(unsigned)((ntr + 255) / 256), 256>>>(Vtr, idx, (int)ntr, dim, sum, cnt);
                update_kernel<<<(nc + 255) / 256, 256>>>(C, sum, cnt, nc, dim);
            }
        }
        if (pen) {   /* ECVQ: 码长按本轮划分重估 */
            CK(cudaMemset(cnt, 0, sizeof(int) * nc));
            vq_hist_kernel<<<(unsigned)((ntr + 255) / 256), 256>>>(idx, ntr, cnt);
            vq_pen_kernel<<<(nc + 255) / 256, 256>>>(cnt, ntr, nc, lam, pen);
        }
    }
    if (round_cb && !fixed) {
        round_f16_kernel<<<(unsigned)(((long long)nc * dim + 255) / 256), 256>>>(C, cbbits, (long long)nc * dim);
        if (cb_fp8) vq_round_e4m3_kernel<<<(unsigned)(((long long)nc * dim + 255) / 256), 256>>>(C, cbbits, (long long)nc * dim);
    }
    if (vq_assign(V, C, pen, nv, nc, dim, idx, SHB, weighted ? cw : none, 0, 1)) return -1;
    CK(cudaDeviceSynchronize());
    if (rate && vq_rate_stats(idx, nv, nc, rate)) return -1;
    if (stride > 1) cudaFree(Vtr);
    cudaFree(sum); cudaFree(cnt);
    if (wsum) cudaFree(wsum);
    if (pen) cudaFree(pen);
    return 0;
}

/* 共同的前半段: 行增益(可选 f16 舍入) + 归一化 + 拼接成连续区 V。 */
static int vq_prepare(float **w, const int *rows, const int *cols, int nmat, int dim, int round_gain,
                      uint16_t **gainbits, float **g, float **V, long long *nv_out) {
    long long tot = 0;
    for (int m = 0; m < nmat; m++) {
        if (((long long)rows[m] * cols[m]) % dim) { fprintf(stderr, "★元素数不是 dim 的整数倍★\n"); return -1; }
        tot += (long long)rows[m] * cols[m];
    }
    for (int m = 0; m < nmat; m++) {
        CK(cudaMalloc(&g[m], sizeof(float) * rows[m]));
        row_gain_kernel<<<rows[m], 256>>>(w[m], g[m], rows[m], cols[m]);
        if (round_gain) round_f16_kernel<<<(unsigned)((rows[m] + 255) / 256), 256>>>(g[m], gainbits[m], rows[m]);
        long long n = (long long)rows[m] * cols[m];
        scale_rows_kernel<<<(unsigned)((n + 255) / 256), 256>>>(w[m], g[m], rows[m], cols[m], 1);
    }
    CK(cudaDeviceSynchronize());
    CK(cudaMalloc(V, sizeof(float) * tot));
    long long off = 0;
    for (int m = 0; m < nmat; m++) {
        long long n = (long long)rows[m] * cols[m];
        CK(cudaMemcpy(*V + off, w[m], sizeof(float) * n, cudaMemcpyDeviceToDevice));
        off += n;
    }
    *nv_out = tot / dim;
    return 0;
}

/* ---- judge 路(原地量化-反量化, 教师前向同条路跑五指标): 接口不变 ---- */
extern "C" int v41_vq_expert_gpu(float **w, const int *rows, const int *cols, int nmat,
                                 int dim, int nc, int iters, int stride,
                                 const float *colw_host, int colw_len) {
    if (dim <= 0 || dim > VQ_MAXDIM || nc < 2 || nmat <= 0) { fprintf(stderr, "★VQ 参数不合法★\n"); return -1; }
    float *g[8], *V; long long nv;
    if (vq_prepare(w, rows, cols, nmat, dim, 0, NULL, g, &V, &nv)) return -1;
    float *C; int *idx;
    CK(cudaMalloc(&C, sizeof(float) * nc * dim));
    CK(cudaMalloc(&idx, sizeof(int) * nv));
    float *dcolw = NULL; vq_cw cw = {NULL, 1, 0, NULL, 1};
    if (colw_host && colw_len > 0) {
        if (colw_len != cols[0]) { fprintf(stderr, "★列权长度 %d != w1 列数 %d★\n", colw_len, cols[0]); return -1; }
        CK(cudaMalloc(&dcolw, sizeof(float) * colw_len));
        CK(cudaMemcpy(dcolw, colw_host, sizeof(float) * colw_len, cudaMemcpyHostToDevice));
        /* 拼接顺序 w1,w3,w2: 前两个吃 x(带列权), w2 吃 h(这条 judge 路只有 x 列权, w2 平权) */
        cw.x = dcolw; cw.px = colw_len;
        cw.nv13 = nmat == 3 ? ((long long)rows[0] * cols[0] + (long long)rows[1] * cols[1]) / dim : nv;
    }
    if (vq_fit_assign(V, nv, dim, nc, iters, stride, C, idx, cw, 0, NULL, 0.f, 0, 0, NULL)) return -1;
    decode_kernel<<<(unsigned)((nv * dim + 255) / 256), 256>>>(V, idx, C, (int)nv, dim);
    long long off = 0;
    for (int m = 0; m < nmat; m++) {
        long long n = (long long)rows[m] * cols[m];
        CK(cudaMemcpy(w[m], V + off, sizeof(float) * n, cudaMemcpyDeviceToDevice));
        scale_rows_kernel<<<(unsigned)((n + 255) / 256), 256>>>(w[m], g[m], rows[m], cols[m], 0);
        off += n;
    }
    CK(cudaDeviceSynchronize());
    for (int m = 0; m < nmat; m++) cudaFree(g[m]);
    cudaFree(V); cudaFree(C); cudaFree(idx);
    if (dcolw) cudaFree(dcolw);
    return 0;
}

/* ---- 落盘编码: w[m] 是 device f32 矩阵(返回时被改成部署解码值); 产物落主机缓冲 ----
 * idx_out[m]: rows[m]×bytes_row 字节; cb_out: nc*dim 个 f16; gain_out[m]: rows[m] 个 f16;
 * stats[0]=Σ(w−ŵ)², stats[1]=Σw²(对打包后字节解码算的), stats[2..3] = 列权加权的同两项。
 * ef/ef_ud: 级 2 误差反馈钩子(v41_ef.h 的 v41_ef_callback), NULL = 不做。opt: 率侧探针选项(v41_vq_rate.h), NULL = 现役。 */
typedef int (*v41_vq_ef_fn)(void *ud, int m, float *V, int rows, int cols, const float *C, int nc, int dim, const float *g, int *idx);
extern "C" int v41_vq_expert_encode_gpu(float **w, const int *rows, const int *cols, int nmat,
                                        int dim, int nc, int iters, int stride, const float *cwx, const float *cwh, int cw_stats_only,
                                        v41_vq_ef_fn ef, void *ef_ud, const v41_vq_opt *opt,
                                        uint8_t **idx_out, uint16_t *cb_out, uint16_t **gain_out, double *stats) {
    if (dim <= 0 || dim > VQ_MAXDIM || nc < 2 || nc > 65536 || nmat <= 0 || nmat > 8) { fprintf(stderr, "★VQ 参数不合法★\n"); return -1; }
    if ((cwx || cwh) && nmat != 3) { fprintf(stderr, "★列权只支持 w1/w3/w2 三矩阵★\n"); return -1; }
    const float lam = opt ? opt->lam : 0.f; const int fixed = opt && opt->cb_fixed;
    int bits = vq_bits(nc);
    /* 列权上设备(主机来的, 每专家 D + MID 个 float, 拷贝可忽略); stats[2..3] = 加权残差账。
     * cw_stats_only = 1: 指派/训练仍平权, 列权只进残差账 —— --calib-ab 机制审计的对照组(同一专家、同一把加权尺) */
    vq_cw cw = {NULL, 1, 0, NULL, 1}; const vq_cw none = {NULL, 1, 0, NULL, 1}; float *dcwx = NULL, *dcwh = NULL;
    if (cwx) { CK(cudaMalloc(&dcwx, sizeof(float) * cols[0])); CK(cudaMemcpy(dcwx, cwx, sizeof(float) * cols[0], cudaMemcpyHostToDevice)); cw.x = dcwx; cw.px = cols[0]; }
    if (cwh) { CK(cudaMalloc(&dcwh, sizeof(float) * cols[2])); CK(cudaMemcpy(dcwh, cwh, sizeof(float) * cols[2], cudaMemcpyHostToDevice)); cw.h = dcwh; cw.ph = cols[2]; }
    if (cwx || cwh) cw.nv13 = ((long long)rows[0] * cols[0] + (long long)rows[1] * cols[1]) / dim;
    long long tot = 0;
    for (int m = 0; m < nmat; m++) tot += (long long)rows[m] * cols[m];
    float *Wo; CK(cudaMalloc(&Wo, sizeof(float) * tot));            /* 原值, 算残差用 */
    { long long off = 0; for (int m = 0; m < nmat; m++) { long long n = (long long)rows[m] * cols[m];
        CK(cudaMemcpy(Wo + off, w[m], sizeof(float) * n, cudaMemcpyDeviceToDevice)); off += n; } }
    float *g[8], *V; long long nv; uint16_t *gb[8];
    for (int m = 0; m < nmat; m++) CK(cudaMalloc(&gb[m], sizeof(uint16_t) * rows[m]));
    if (vq_prepare(w, rows, cols, nmat, dim, 1, gb, g, &V, &nv)) return -1;
    float *C; int *idx; uint16_t *cbb;
    CK(cudaMalloc(&C, sizeof(float) * nc * dim));
    CK(cudaMalloc(&cbb, sizeof(uint16_t) * nc * dim));
    CK(cudaMalloc(&idx, sizeof(int) * nv));
    if (fixed) {   /* 给定码本(共享码本探针): 位型直接上设备, f32 副本由位型解出 ⇒ 指派用的值与文件里的逐位同 */
        CK(cudaMemcpy(cbb, opt->cb_fixed, sizeof(uint16_t) * nc * dim, cudaMemcpyHostToDevice));
        vq_f16_to_f32_kernel<<<(unsigned)(((long long)nc * dim + 255) / 256), 256>>>(cbb, C, (long long)nc * dim);
    }
    if (vq_fit_assign(V, nv, dim, nc, iters, stride, C, idx, cw_stats_only ? none : cw, 1, cbb,
                      lam, fixed, opt ? opt->cb_fp8 : 0, opt ? opt->rate : NULL)) return -1;
    /* 级 2 钩子(2026-09-21): 码本与行增益已定死(C 已舍到 f16 格点, g 已舍), 钩子按矩阵重选索引(误差反馈), V 被它改写也无妨 —— 下面打包/解码只看索引 */
    if (ef) {
        long long o = 0;
        for (int m = 0; m < nmat; m++) {
            const long long n = (long long)rows[m] * cols[m];
            const int rc = ef(ef_ud, m, V + o, rows[m], cols[m], C, nc, dim, g[m], idx + o / dim);
            if (rc < 0) return -1;
            o += n;
        }
    }
    double *acc; CK(cudaMalloc(&acc, sizeof(double) * 4));
    CK(cudaMemset(acc, 0, sizeof(double) * 4));
    long long off = 0;
    for (int m = 0; m < nmat; m++) {
        int nidx_row = cols[m] / dim, bytes_row = (nidx_row * bits + 7) / 8;
        long long nb = (long long)rows[m] * bytes_row, n = (long long)rows[m] * cols[m];
        uint8_t *pk; CK(cudaMalloc(&pk, nb));
        pack_kernel<<<(unsigned)((nb + 255) / 256), 256>>>(idx + off / dim, pk, rows[m], nidx_row, bits, bytes_row);
        /* 部署口径解码回 w[m], 并对原值算残差 */
        unpack_decode_kernel<<<(unsigned)((n / dim + 255) / 256), 256>>>(pk, cbb, gb[m], rows[m], cols[m], dim, bits, bytes_row, w[m]);
        v41_sse_kernel<<<(unsigned)((n + 255) / 256), 256>>>(Wo + off, w[m], n, acc);
        wsse_kernel<<<(unsigned)((n + 255) / 256), 256>>>(Wo + off, w[m], n, cols[m], m < 2 ? cw.x : cw.h, acc + 2);
        CK(cudaGetLastError());
        CK(cudaMemcpy(idx_out[m], pk, nb, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gain_out[m], gb[m], sizeof(uint16_t) * rows[m], cudaMemcpyDeviceToHost));
        cudaFree(pk);
        off += n;
    }
    CK(cudaMemcpy(cb_out, cbb, sizeof(uint16_t) * nc * dim, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(stats, acc, sizeof(double) * 4, cudaMemcpyDeviceToHost));
    for (int m = 0; m < nmat; m++) { cudaFree(g[m]); cudaFree(gb[m]); }
    cudaFree(Wo); cudaFree(V); cudaFree(C); cudaFree(cbb); cudaFree(idx); cudaFree(acc);
    if (dcwx) cudaFree(dcwx);
    if (dcwh) cudaFree(dcwh);
    return 0;
}

/* 部署解码(教师端 ctypes 调用, 全 device 指针): 与量化器内部统计同一个核。 */
extern "C" int v41_vq_decode_gpu(const uint8_t *packed, const uint16_t *cb, const uint16_t *gain,
                                 int rows, int cols, int dim, int nc, float *out) {
    if (dim <= 0 || cols % dim || nc < 2) { fprintf(stderr, "★解码参数不合法★\n"); return -1; }
    int bits = vq_bits(nc), nidx_row = cols / dim, bytes_row = (nidx_row * bits + 7) / 8;
    long long n = (long long)rows * nidx_row;
    unpack_decode_kernel<<<(unsigned)((n + 255) / 256), 256>>>(packed, cb, gain, rows, cols, dim, bits, bytes_row, out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    return 0;
}

/* 共享码本探针的两个入口 + 出厂 FP4 上设备的公共小函数(2026-09-21) */
#include "v41_vq_pool.inc.cu"

/* 量化器入口(C 驱动调用, 全主机指针): HF 出厂 FP4 的三个矩阵 → VQ 编码产物。
 * cwx[cols0] / cwh[cols2] = 校准列权(w1/w3 的 x 侧 / w2 的 h 侧), NULL = 平权(2026-09-20 深夜加, 见 v41_calib.h)。 */
extern "C" int v41_vq_expert_from_fp4(const uint8_t *const *w_host, const uint8_t *const *s_host,
                                      const int *rows, const int *cols, int nmat, int dim, int nc, int iters, int stride,
                                      const float *cwx, const float *cwh, int cw_stats_only, v41_vq_ef_fn ef, void *ef_ud, const v41_vq_opt *opt,
                                      uint8_t **idx_out, uint16_t *cb_out, uint16_t **gain_out, double *stats) {
    float *w[8];
    if (vq_upload_fp4(w_host, s_host, rows, cols, nmat, w)) return -1;
    return v41_vq_expert_encode_gpu(w, rows, cols, nmat, dim, nc, iters, stride, cwx, cwh, cw_stats_only, ef, ef_ud, opt, idx_out, cb_out, gain_out, stats);
}
