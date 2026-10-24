/* v41_ef.cu — 级 2: 误差反馈重指派(GPTQ/GPTVQ 同式, 2026-09-21 凌晨, 用户令"按金融域量化")。
 * 码本与行增益按级 1 定死, 只重选索引: 8 列一组从左到右, 组内按 H⁻¹ 的 Cholesky 对角加权选码字
 * (列 c 的损失 = err_c²/U[c,c]²), 组的误差按 U 行传给右边还没量化的列(v[:, c'] −= Σ_c err_c/U[c,c]·U[c,c'])。
 * GPTQ 的懒惰分块: 128 列一块, 块内各组的传播先记在 Err 里、下一组开工时补上, 块尾一次 GEMM 推到块外。
 * H(x 侧, w1/w3 共用) = X_eᵀdiag(rw²)X_e/Σrw² + β·E_layer[x xᵀ] + δ·mean(diag)·I, β 按本专家 val 行的输出误差择优;
 * H(h 侧, w2) = hᵀdiag(rw²)h/Σrw² + δ·mean(diag)·I, h 在【已量化】的 Ŵ1/Ŵ3 上算 = 部署时 w2 真吃的输入。
 * 为什么锚层级 Gram: 每专家中位只有几十行(09-21 取料自检: L39 中位 5 行), X_e 张不开 5120 维, 纯本专家的 H 在其余方向全靠 δ 兜底
 * (09-20 码本重解"128 行只辨认幅度"的墙); 层级 8192 行把方向补上, β 定它的话语权。名下 <8 行的专家不做(保级 1 指派, wt2 不退步)。
 * 全 GPU: cuBLAS(Gram/GEMM) + cuSOLVER(potrf/potri), 无 CPU 参考路。 */
#include "v41_dq.cuh"
#include "v41_ef.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EF_BLK 128           /* 懒惰分块列数(= GPTQ 默认) */
#define EF_MAXB 4
#define EF_MINROWS 8
#define EF_DAMP 0.01f        /* δ = 1% × mean(diag), GPTQ percdamp 同值; potrf 失败 ×10 再试 */
#define EF_CHUNK 512

struct v41_ef {
    v41_calib *cal; int D, e, m_rows, nb; float betas[EF_MAXB]; float nval;
    const float *dx, *drw2, *dval;          /* 本专家名下行(v41_calib 持有) */
    const float *dHL;                       /* 层级 Gram [D][D] */
    v41_dbuf He, Ub[EF_MAXB], Vbak, Vtry, idxtry, Err, work, xs, dh, dg, du, outbuf, acc;
    const int *idx01[2]; const float *g01[2];   /* 本专家 w1/w3 的索引/增益(w2 的 h 要它们) */
    cublasHandle_t bl; cusolverDnHandle_t sv; int *dinfo;
    int done, skipped, bsel[EF_MAXB]; double ratio_sum; int ratio_n;   /* 层账 */
};

#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "★cuBLAS %s @%d: %d★\n", #x, __LINE__, (int)s_); return -1; } } while (0)
#define CS(x) do { cusolverStatus_t s_ = (x); if (s_ != CUSOLVER_STATUS_SUCCESS) { fprintf(stderr, "★cuSOLVER %s @%d: %d★\n", #x, __LINE__, (int)s_); return -1; } } while (0)

__global__ static void k_mix_h(float *H, const float *He, const float *HL, float b, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) H[i] = He[i] + (HL ? b * HL[i] : 0.f);
}
__global__ static void k_diag_sum(const float *H, int K, double *acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) atomicAdd(acc, (double)H[(size_t)i * K + i]);
}
__global__ static void k_add_diag(float *H, int K, float d) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < K) H[(size_t)i * K + i] += d; }
/* potri 只填下三角(列主序 LOWER): 上三角从下三角抄齐, 之后 potrf(UPPER) 才有东西读 */
__global__ static void k_sym_lower_to_upper(float *A, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= K || r >= K || c <= r) return;
    A[(size_t)c * K + r] = A[(size_t)r * K + c];
}
__global__ static void k_scale_rows(float *x, const float *w, float inv_sum, long long n, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sqrtf(w[i / K] * inv_sum);
}
/* Ŵ = g_r·C[idx](行主序 [R][K]) */
__global__ static void k_decode(float *W, const int *idx, const float *C, const float *g, long long n, int K, int dim) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int r = (int)(i / K), c = (int)(i % K);
    W[i] = g[r] * C[(size_t)idx[(size_t)r * (K / dim) + c / dim] * dim + c % dim];
}
/* E = g·Vbak − E(E 进来是 Ŵ, 出去是 W − Ŵ) */
__global__ static void k_err(float *E, const float *Vbak, const float *g, long long n, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) E[i] = g[i / K] * Vbak[i] - E[i];
}
__global__ static void k_swiglu(float *h, const float *gt, const float *up, long long n, float clamp) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = gt[i], uv = up[i];
    if (clamp > 0.f) { if (gv > clamp) gv = clamp; if (uv > clamp) uv = clamp; if (uv < -clamp) uv = -clamp; }
    h[i] = (gv / (1.f + expf(-gv))) * uv;
}
/* Σ_i val_i·rw²_i·Σ_r Out[i][r]²(Out 行主序 [m][R]) */
__global__ static void k_val_loss(const float *Out, const float *w, const float *val, int m, int R, double *acc) {
    const int i = blockIdx.x;
    if (i >= m || val[i] <= 0.f) return;
    __shared__ double sh[256];
    double s = 0; for (int r = threadIdx.x; r < R; r += blockDim.x) { const float v = Out[(size_t)i * R + r]; s += (double)v * v; }
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(acc, sh[0] * w[i]);
}

/* ★误差反馈一组(8 列)★: 一线程一行。①补上本块里前面各组欠的传播 ②按 1/U[c,c]² 加权选码字 ③记 err_c = (v_c − q_c)/U[c,c], 列写成 q。
 * U 列主序 K×K, 只读上三角 U[i][j](i≤j) = U[j*K+i]。shared: shU[np*DIM](待传播 U 片) + shC[512*DIM](码本块) */
template <int DIM>
__global__ static void k_ef_group(float *V, int R, int K, const float *__restrict__ U, int c0, int b0, float *Err,
                                  const float *__restrict__ C, int nc, int *idx) {
    extern __shared__ float sh[];
    float *shU = sh, *shC = sh + EF_BLK * DIM;
    const int np = c0 - b0;
    for (int t = threadIdx.x; t < np * DIM; t += blockDim.x) { const int p = t / DIM, d = t % DIM; shU[t] = U[(size_t)(c0 + d) * K + b0 + p]; }
    float diag[DIM];
#pragma unroll
    for (int d = 0; d < DIM; d++) diag[d] = U[(size_t)(c0 + d) * K + c0 + d];
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    float v[DIM], w[DIM];
#pragma unroll
    for (int d = 0; d < DIM; d++) v[d] = 0.f;
    if (r < R) {
        const float *er = Err + (size_t)r * EF_BLK;
#pragma unroll
        for (int d = 0; d < DIM; d++) v[d] = V[(size_t)r * K + c0 + d];
        for (int p = 0; p < np; p++) { const float e = er[p];
#pragma unroll
            for (int d = 0; d < DIM; d++) v[d] -= e * shU[p * DIM + d]; }
    }
#pragma unroll
    for (int d = 0; d < DIM; d++) w[d] = 1.f / (diag[d] * diag[d]);
    float bd = 3.0e38f; int best = 0;
    for (int k0 = 0; k0 < nc; k0 += EF_CHUNK) {
        const int n = nc - k0 < EF_CHUNK ? nc - k0 : EF_CHUNK;
        __syncthreads();
        for (int t = threadIdx.x; t < n * DIM; t += blockDim.x) shC[t] = C[(size_t)k0 * DIM + t];
        __syncthreads();
        if (r < R) for (int k = 0; k < n; k++) {
            float dist = 0.f;
#pragma unroll
            for (int d = 0; d < DIM; d++) { const float e = v[d] - shC[k * DIM + d]; dist += w[d] * e * e; }
            if (dist < bd) { bd = dist; best = k0 + k; }
        }
    }
    if (r < R) {
        idx[(size_t)r * (K / DIM) + c0 / DIM] = best;
#pragma unroll
        for (int d = 0; d < DIM; d++) { const float q = C[(size_t)best * DIM + d]; Err[(size_t)r * EF_BLK + np + d] = (v[d] - q) / diag[d]; V[(size_t)r * K + c0 + d] = q; }
    }
}

extern "C" v41_ef *v41_ef_open(v41_calib *cal, const float *betas, int nbeta) {
    if (nbeta < 1 || nbeta > EF_MAXB) { fprintf(stderr, "★β 候选数 %d 不合法(1..%d)★\n", nbeta, EF_MAXB); return NULL; }
    v41_ef *ef = (v41_ef *)calloc(1, sizeof *ef);
    ef->cal = cal; ef->D = v41_calib_D(cal); ef->nb = nbeta; ef->e = -1;
    for (int i = 0; i < nbeta; i++) ef->betas[i] = betas[i];
    if (cublasCreate(&ef->bl) != CUBLAS_STATUS_SUCCESS || cusolverDnCreate(&ef->sv) != CUSOLVER_STATUS_SUCCESS || cudaMalloc(&ef->dinfo, sizeof(int)) != cudaSuccess) { fprintf(stderr, "★级 2 句柄创建失败★\n"); free(ef); return NULL; }
    if (v41_calib_layer_gram(cal, &ef->dHL)) { free(ef); return NULL; }
    return ef;
}

extern "C" void v41_ef_set_expert(v41_ef *ef, int e) { ef->e = e; ef->m_rows = -1; }

/* H(K×K 列主序, 已含 δ) → U = chol(H⁻¹, upper), 原地。非正定返回 1 */
static int chol_inv_chol(v41_ef *ef, float *H, int K) {
    int lw1 = 0, lw2 = 0, info = 0;
    CS(cusolverDnSpotrf_bufferSize(ef->sv, CUBLAS_FILL_MODE_LOWER, K, H, K, &lw1));
    CS(cusolverDnSpotri_bufferSize(ef->sv, CUBLAS_FILL_MODE_LOWER, K, H, K, &lw2));
    const int lw = lw1 > lw2 ? lw1 : lw2;
    if (v41_dbuf_need(&ef->work, sizeof(float) * (size_t)lw + 256)) return -1;
    float *work = (float *)ef->work.p;
    CS(cusolverDnSpotrf(ef->sv, CUBLAS_FILL_MODE_LOWER, K, H, K, work, lw1, ef->dinfo));
    V41_CK(cudaMemcpy(&info, ef->dinfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) return 1;
    CS(cusolverDnSpotri(ef->sv, CUBLAS_FILL_MODE_LOWER, K, H, K, work, lw2, ef->dinfo));
    V41_CK(cudaMemcpy(&info, ef->dinfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) return 1;
    k_sym_lower_to_upper<<<dim3((unsigned)((K + 255) / 256), (unsigned)K), 256>>>(H, K);
    CS(cusolverDnSpotrf(ef->sv, CUBLAS_FILL_MODE_UPPER, K, H, K, work, lw1, ef->dinfo));
    V41_CK(cudaMemcpy(&info, ef->dinfo, sizeof(int), cudaMemcpyDeviceToHost));
    return info != 0 ? 1 : 0;
}

/* U ← chol((He + β·HL + δ·mean(diag)·I)⁻¹, upper) */
static int build_u(v41_ef *ef, float *U, const float *He, const float *HL, float beta, int K) {
    if (v41_dbuf_need(&ef->acc, sizeof(double))) return -1;
    double *acc = (double *)ef->acc.p; const long long n = (long long)K * K;
    for (int attempt = 0; attempt < 4; attempt++) {
        const float damp = EF_DAMP * powf(10.f, (float)attempt);
        k_mix_h<<<(unsigned)((n + 255) / 256), 256>>>(U, He, HL, beta, n);
        V41_CK(cudaMemset(acc, 0, sizeof(double)));
        k_diag_sum<<<(unsigned)((K + 255) / 256), 256>>>(U, K, acc);
        double ds = 0; V41_CK(cudaMemcpy(&ds, acc, sizeof(double), cudaMemcpyDeviceToHost));
        k_add_diag<<<(unsigned)((K + 255) / 256), 256>>>(U, K, (float)(damp * ds / K));
        const int rc = chol_inv_chol(ef, U, K);
        if (rc < 0) return -1;
        if (rc == 0) return 0;
        fprintf(stderr, "  [EF] 专家 %d K=%d δ=%g 非正定, 加大重试\n", ef->e, K, damp);
    }
    fprintf(stderr, "★EF 专家 %d K=%d: H 四次都非正定★\n", ef->e, K);
    return -1;
}

/* V(行主序 [R][K], 归一化)上按 U 做误差反馈重指派, 写 idx; V 被改写成量化值 */
static int ef_run(v41_ef *ef, const float *U, float *V, int R, int K, const float *C, int nc, int dim, int *idx) {
    if (dim != 8 || K % EF_BLK) { fprintf(stderr, "★EF 只实现了 dim=8 且列数是 %d 的倍数(收到 dim %d K %d)★\n", EF_BLK, dim, K); return -1; }
    if (v41_dbuf_need(&ef->Err, sizeof(float) * (size_t)R * EF_BLK)) return -1;
    float *Err = (float *)ef->Err.p;
    const size_t shb = (size_t)EF_BLK * 8 * sizeof(float) + (size_t)EF_CHUNK * 8 * sizeof(float);
    const float mone = -1.f, one = 1.f;
    for (int b0 = 0; b0 < K; b0 += EF_BLK) {
        const int b1 = b0 + EF_BLK;
        for (int c0 = b0; c0 < b1; c0 += 8)
            k_ef_group<8><<<(unsigned)((R + 255) / 256), 256, shb>>>(V, R, K, U, c0, b0, Err, C, nc, idx);
        V41_CK(cudaGetLastError());
        /* 块尾: V[:, b1:] −= Err(R×128)·U[b0:b1, b1:]。列主序: V'(K×R) 第 b1 行起 −= U_subᵀ((K−b1)×128)·Err'(128×R) */
        if (b1 < K) CB(cublasSgemm(ef->bl, CUBLAS_OP_T, CUBLAS_OP_N, K - b1, R, EF_BLK, &mone, U + (size_t)b1 * K + b0, K, Err, EF_BLK, &one, V + b1, K));
    }
    V41_CK(cudaDeviceSynchronize());
    return 0;
}

/* val 行输出误差 Σ val·rw²·‖(W − Ŵ)x‖², W = g·Vbak(级 1 归一权重原值), Ŵ = g·C[idx]; X 行主序 [m][K] */
static int val_loss(v41_ef *ef, const float *Vbak, const int *idx, const float *C, const float *g, int R, int K, int dim,
                    const float *X, int m, double *out) {
    const long long n = (long long)R * K;
    if (v41_dbuf_need(&ef->Vtry, sizeof(float) * (size_t)n) || v41_dbuf_need(&ef->outbuf, sizeof(float) * (size_t)m * R) || v41_dbuf_need(&ef->acc, sizeof(double))) return -1;
    float *E = (float *)ef->Vtry.p, *Out = (float *)ef->outbuf.p; double *acc = (double *)ef->acc.p;
    k_decode<<<(unsigned)((n + 255) / 256), 256>>>(E, idx, C, g, n, K, dim);
    k_err<<<(unsigned)((n + 255) / 256), 256>>>(E, Vbak, g, n, K);
    const float one = 1.f, zero = 0.f;   /* Out[m][R] = X·Eᵀ: 列主序 Out'(R×m) = E'(K×R)ᵀ·X'(K×m) */
    CB(cublasSgemm(ef->bl, CUBLAS_OP_T, CUBLAS_OP_N, R, m, K, &one, E, K, X, K, &zero, Out, R));
    V41_CK(cudaMemset(acc, 0, sizeof(double)));
    k_val_loss<<<(unsigned)m, 256>>>(Out, ef->drw2, ef->dval, m, R, acc);
    V41_CK(cudaMemcpy(out, acc, sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

/* He = Xsᵀ·Xs, Xs = X 行乘 sqrt(rw²/Σrw²)(X 行主序 [m][K]) */
static int gram(v41_ef *ef, const float *X, int m, int K, float *He) {
    float sum = 0.f; CB(cublasSasum(ef->bl, m, ef->drw2, 1, &sum));
    if (v41_dbuf_need(&ef->xs, sizeof(float) * (size_t)m * K)) return -1;
    float *xs = (float *)ef->xs.p;
    V41_CK(cudaMemcpy(xs, X, sizeof(float) * (size_t)m * K, cudaMemcpyDeviceToDevice));
    k_scale_rows<<<(unsigned)(((long long)m * K + 255) / 256), 256>>>(xs, ef->drw2, sum > 0.f ? 1.f / sum : 1.f, (long long)m * K, K);
    const float one = 1.f, zero = 0.f;   /* 行主序 Xs[m][K] = 列主序 Xs'(K×m) ⇒ He(K×K) = Xs'·Xs'ᵀ */
    CB(cublasSgemm(ef->bl, CUBLAS_OP_N, CUBLAS_OP_T, K, K, m, &one, xs, K, xs, K, &zero, He, K));
    return 0;
}

extern "C" int v41_ef_callback(void *ud, int m, float *V, int rows, int cols, const float *C, int nc, int dim, const float *g, int *idx) {
    v41_ef *ef = (v41_ef *)ud; const int R = rows, K = cols; const long long n = (long long)R * K;
    if (m == 0) {   /* 新专家: 取料 + x 侧 Gram + 各 β 的 U */
        ef->m_rows = v41_calib_expert_x(ef->cal, ef->e, &ef->dx, &ef->drw2, &ef->dval);
        if (ef->m_rows < 0) return -1;
        ef->idx01[0] = ef->idx01[1] = NULL;
        if (ef->m_rows < EF_MINROWS) { ef->skipped++; return 1; }
        CB(cublasSasum(ef->bl, ef->m_rows, ef->dval, 1, &ef->nval));
        if (v41_dbuf_need(&ef->He, sizeof(float) * (size_t)K * K)) return -1;
        if (gram(ef, ef->dx, ef->m_rows, K, (float *)ef->He.p)) return -1;
        const int nb = ef->nval >= 1.f ? ef->nb : 1;
        for (int b = 0; b < nb; b++) {
            if (v41_dbuf_need(&ef->Ub[b], sizeof(float) * (size_t)K * K)) return -1;
            if (build_u(ef, (float *)ef->Ub[b].p, (const float *)ef->He.p, ef->dHL, ef->betas[b], K)) return -1;
        }
        ef->done++;
    }
    if (ef->m_rows < EF_MINROWS) return 1;
    if (v41_dbuf_need(&ef->Vbak, sizeof(float) * (size_t)n) || v41_dbuf_need(&ef->idxtry, sizeof(int) * (size_t)n / dim)) return -1;
    float *Vbak = (float *)ef->Vbak.p; int *idxtry = (int *)ef->idxtry.p;
    V41_CK(cudaMemcpy(Vbak, V, sizeof(float) * (size_t)n, cudaMemcpyDeviceToDevice));
    if (m < 2) {   /* x 侧: 每个 β 在 Vtry 上试一遍, 按 val 输出误差择优, 最优索引拷进 idx(val 不足 1 行 ⇒ 只跑 betas[0]) */
        const int nb = ef->nval >= 1.f ? ef->nb : 1;
        double l1 = 0, lbest = 0; int bbest = -1;
        if (nb > 1 && val_loss(ef, Vbak, idx, C, g, R, K, dim, ef->dx, ef->m_rows, &l1)) return -1;
        if (v41_dbuf_need(&ef->Vtry, sizeof(float) * (size_t)n)) return -1;
        for (int b = 0; b < nb; b++) {
            float *Vt = (float *)ef->Vtry.p;
            V41_CK(cudaMemcpy(Vt, Vbak, sizeof(float) * (size_t)n, cudaMemcpyDeviceToDevice));
            if (ef_run(ef, (const float *)ef->Ub[b].p, Vt, R, K, C, nc, dim, idxtry)) return -1;
            double lb = 0;   /* val_loss 借 Vtry 当 E 缓冲: 候选的 V 不再需要, 只留索引(编码器按索引打包/解码, 不看 V) */
            if (nb > 1 && val_loss(ef, Vbak, idxtry, C, g, R, K, dim, ef->dx, ef->m_rows, &lb)) return -1;
            if (bbest < 0 || lb < lbest) { lbest = lb; bbest = b; V41_CK(cudaMemcpy(idx, idxtry, sizeof(int) * (size_t)(n / dim), cudaMemcpyDeviceToDevice)); }
        }
        ef->bsel[bbest]++;
        if (nb > 1 && l1 > 0) { ef->ratio_sum += lbest / l1; ef->ratio_n++; }
        ef->idx01[m] = idx; ef->g01[m] = g;
        return 0;
    }
    /* h 侧(w2): Ŵ1/Ŵ3 从已重指派的索引解码, h = swiglu(X·Ŵ1ᵀ, X·Ŵ3ᵀ), Gram 只用本专家 + δ */
    if (!ef->idx01[0] || !ef->idx01[1]) { fprintf(stderr, "★EF w2 前没有 w1/w3 的索引★\n"); return -1; }
    const int MID = K, mr = ef->m_rows, D = ef->D;
    if (v41_dbuf_need(&ef->Vtry, sizeof(float) * (size_t)MID * D) || v41_dbuf_need(&ef->xs, sizeof(float) * (size_t)MID * D) ||
        v41_dbuf_need(&ef->dg, sizeof(float) * (size_t)mr * MID) || v41_dbuf_need(&ef->du, sizeof(float) * (size_t)mr * MID) || v41_dbuf_need(&ef->dh, sizeof(float) * (size_t)mr * MID)) return -1;
    float *W1 = (float *)ef->Vtry.p, *W3 = (float *)ef->xs.p, *G = (float *)ef->dg.p, *Up = (float *)ef->du.p, *h = (float *)ef->dh.p;
    const long long n13 = (long long)MID * D;
    k_decode<<<(unsigned)((n13 + 255) / 256), 256>>>(W1, ef->idx01[0], C, ef->g01[0], n13, D, dim);
    k_decode<<<(unsigned)((n13 + 255) / 256), 256>>>(W3, ef->idx01[1], C, ef->g01[1], n13, D, dim);
    const float one = 1.f, zero = 0.f;   /* G[m][MID] = X·Ŵ1ᵀ: 列主序 G'(MID×m) = Ŵ1'(D×MID)ᵀ·X'(D×m) */
    CB(cublasSgemm(ef->bl, CUBLAS_OP_T, CUBLAS_OP_N, MID, mr, D, &one, W1, D, ef->dx, D, &zero, G, MID));
    CB(cublasSgemm(ef->bl, CUBLAS_OP_T, CUBLAS_OP_N, MID, mr, D, &one, W3, D, ef->dx, D, &zero, Up, MID));
    k_swiglu<<<(unsigned)(((long long)mr * MID + 255) / 256), 256>>>(h, G, Up, (long long)mr * MID, v41_calib_clamp(ef->cal));
    if (v41_dbuf_need(&ef->He, sizeof(float) * (size_t)MID * MID) || v41_dbuf_need(&ef->Ub[0], sizeof(float) * (size_t)MID * MID)) return -1;
    if (gram(ef, h, mr, MID, (float *)ef->He.p)) return -1;
    if (build_u(ef, (float *)ef->Ub[0].p, (const float *)ef->He.p, NULL, 0.f, MID)) return -1;
    double l1 = 0, l2 = 0;
    if (ef->nval >= 1.f && val_loss(ef, Vbak, idx, C, g, R, K, dim, h, mr, &l1)) return -1;
    if (ef_run(ef, (const float *)ef->Ub[0].p, V, R, K, C, nc, dim, idx)) return -1;
    if (ef->nval >= 1.f) { if (val_loss(ef, Vbak, idx, C, g, R, K, dim, h, mr, &l2)) return -1; if (l1 > 0) { ef->ratio_sum += l2 / l1; ef->ratio_n++; } }
    return 0;
}

extern "C" void v41_ef_report(const v41_ef *ef, char *buf, size_t n) {
    char bs[128] = ""; size_t o = 0;
    for (int b = 0; b < ef->nb; b++) o += (size_t)snprintf(bs + o, sizeof bs - o, "%sβ=%g×%d", b ? " " : "", ef->betas[b], ef->bsel[b]);
    snprintf(buf, n, "级 2 做 %d / 跳过(<%d 行) %d; %s; val 输出误差 级2÷级1 均值 %.3f(n=%d)", ef->done, EF_MINROWS, ef->skipped, bs,
             ef->ratio_n ? ef->ratio_sum / ef->ratio_n : 1.0, ef->ratio_n);
}

extern "C" void v41_ef_close(v41_ef *ef) {
    if (!ef) return;
    v41_dbuf *bs[13] = {&ef->He, &ef->Vbak, &ef->Vtry, &ef->idxtry, &ef->Err, &ef->work, &ef->xs, &ef->dh, &ef->dg, &ef->du, &ef->outbuf, &ef->acc, &ef->Ub[0]};
    for (int i = 0; i < 13; i++) if (bs[i]->p) cudaFree(bs[i]->p);
    for (int b = 1; b < EF_MAXB; b++) if (ef->Ub[b].p) cudaFree(ef->Ub[b].p);
    if (ef->dinfo) cudaFree(ef->dinfo);
    if (ef->bl) cublasDestroy(ef->bl);
    if (ef->sv) cusolverDnDestroy(ef->sv);
    free(ef);
}
