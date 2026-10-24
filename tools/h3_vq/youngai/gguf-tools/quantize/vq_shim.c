/* vq_shim.c — VQ 码本量化器 C 实现(2026-07-25 磨刀) + go2b/signref 导出, 供探针 ctypes 与后续量化器复用。
 * 编码链与 g1c_vq_sweep.vq_gptq 同口径: 分块 kmeans(确定性步长子采样) → 逐 dim 段最近邻
 * + GPTQ 列组误差反馈 → 行级激活乘子。构建: cc -O3 -shared -fPIC -framework Accelerate -o vq_shim.dylib vq_shim.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

void dq_matmul(const float *X, const float *W, float *out, int S, int K, int M) {
    /* out[S,M] = X[S,K] @ W[M,K]^T (与 ds4quant_fwd.c 同语义) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, M, K,
                1.0f, X, K, W, K, 0.0f, out, M);
}
#include "onebit_quant.c"
#include "../amp/ds4quant_qhelp.h"
#include "go2b_qc.h"

/* 最近邻(分块 sgemm): V[nv,dim] vs C[nc,dim] → idx[nv] */
static void vq_assign(const float *V, int nv, int dim, const float *C, int nc, int *idx) {
    const int BS = 8192;
    float *c2 = malloc((size_t)nc * 4), *G = malloc((size_t)BS * nc * 4);
    for (int c = 0; c < nc; c++) { double s = 0; for (int d = 0; d < dim; d++) s += (double)C[c*dim+d]*C[c*dim+d]; c2[c] = (float)s; }
    for (int i0 = 0; i0 < nv; i0 += BS) {
        int b = nv - i0 < BS ? nv - i0 : BS;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, b, nc, dim,
                    -2.0f, V + (size_t)i0*dim, dim, C, dim, 0.0f, G, nc);
        for (int i = 0; i < b; i++) {
            const float *g = G + (size_t)i*nc; int best = 0; float m = g[0] + c2[0];
            for (int c = 1; c < nc; c++) { float v = g[c] + c2[c]; if (v < m) { m = v; best = c; } }
            idx[i0+i] = best;
        }
    }
    free(c2); free(G);
}

/* 导出: VQ+GPTQ 编码, Wq_out[rows*cols] = dequant 浮点 */
void vq_encode_c(const float *W, int rows, int cols, int dim, int nc,
                 const float *X, int n, float *Wq_out) {
    size_t N = (size_t)rows * cols; int nv = (int)(N / dim);
    /* kmeans: 步长子采样 ≤200k 向量, 8 轮 Lloyd */
    int nsub = nv < 200000 ? nv : 200000, stride = nv / nsub;
    float *sub = malloc((size_t)nsub * dim * 4);
    for (int i = 0; i < nsub; i++) memcpy(sub + (size_t)i*dim, W + (size_t)i*stride*dim, (size_t)dim*4);
    float *C = malloc((size_t)nc * dim * 4);
    int cst = nsub / nc; if (cst < 1) cst = 1;
    for (int c = 0; c < nc; c++) memcpy(C + (size_t)c*dim, sub + (size_t)(c*cst)*dim, (size_t)dim*4);
    int *aidx = malloc((size_t)nsub * sizeof(int));
    double *acc = malloc((size_t)nc * dim * 8); long *cnt = malloc((size_t)nc * 8);
    for (int it = 0; it < 8; it++) {
        vq_assign(sub, nsub, dim, C, nc, aidx);
        memset(acc, 0, (size_t)nc*dim*8); memset(cnt, 0, (size_t)nc*8);
        for (int i = 0; i < nsub; i++) { int c = aidx[i]; cnt[c]++;
            for (int d = 0; d < dim; d++) acc[(size_t)c*dim+d] += sub[(size_t)i*dim+d]; }
        for (int c = 0; c < nc; c++) if (cnt[c] > 0)
            for (int d = 0; d < dim; d++) C[(size_t)c*dim+d] = (float)(acc[(size_t)c*dim+d]/cnt[c]);
    }
    free(sub); free(aidx); free(acc); free(cnt);
    /* GPTQ 列组误差反馈 + 段最近邻 */
    int grp = 128;
    float *Wk = malloc((size_t)rows * grp * 4), *seg = malloc((size_t)rows * dim * 4);
    int *sidx = malloc((size_t)rows * sizeof(int));
    double *H = malloc((size_t)grp*grp*8), *Hi = malloc((size_t)grp*grp*8);
    float *XbT = malloc((size_t)grp * n * 4);
    for (int j0 = 0; j0 < cols; j0 += grp) {
        int g = cols - j0 < grp ? cols - j0 : grp;
        for (int j = 0; j < g; j++) for (int t = 0; t < n; t++) XbT[(size_t)j*n+t] = X[(size_t)t*cols + j0 + j];
        for (int i = 0; i < g; i++) for (int j = i; j < g; j++) { double s = 0;
            const float *xi = XbT + (size_t)i*n, *xj = XbT + (size_t)j*n;
            for (int t = 0; t < n; t++) s += (double)xi[t]*xj[t];
            H[(size_t)i*g+j] = H[(size_t)j*g+i] = s; }
        double dm = 0; for (int i = 0; i < g; i++) dm += H[(size_t)i*g+i]; dm /= g;
        for (int i = 0; i < g; i++) H[(size_t)i*g+i] += 0.02*(dm + 1e-9);
        int ok = g2_inv(H, g, Hi) == 0;
        for (int r = 0; r < rows; r++) memcpy(Wk + (size_t)r*g, W + (size_t)r*cols + j0, (size_t)g*4);
        for (int jj = 0; jj + dim <= g; jj += dim) {
            for (int r = 0; r < rows; r++) memcpy(seg + (size_t)r*dim, Wk + (size_t)r*g + jj, (size_t)dim*4);
            vq_assign(seg, rows, dim, C, nc, sidx);
            for (int r = 0; r < rows; r++) {
                const float *q = C + (size_t)sidx[r]*dim;
                memcpy(Wq_out + (size_t)r*cols + j0 + jj, q, (size_t)dim*4);
                if (ok && jj + dim < g) {
                    for (int d = 0; d < dim; d++) {
                        double hjj = Hi[(size_t)(jj+d)*g + jj + d];
                        if (fabs(hjj) < 1e-30) hjj = 1e-30;
                        float er = (float)((seg[(size_t)r*dim+d] - q[d]) / hjj);
                        if (er == 0.0f) continue;
                        const double *hrow = Hi + (size_t)(jj+d)*g;
                        float *wkr = Wk + (size_t)r*g;
                        for (int j2 = jj + dim; j2 < g; j2++) wkr[j2] -= er * (float)hrow[j2];
                    }
                }
            }
        }
    }
    free(Wk); free(seg); free(sidx); free(H); free(Hi); free(XbT); free(C);
    /* 行级激活乘子 g_r(闭式 ridge, 与 python 同口径) */
    if (X && n >= 8) {
        float *P = malloc((size_t)n * rows * 4), *Y = malloc((size_t)n * rows * 4);
        dq_matmul(X, Wq_out, P, n, cols, rows);
        dq_matmul(X, W, Y, n, cols, rows);
        for (int r = 0; r < rows; r++) {
            double sp2 = 0, spy = 0;
            for (int t = 0; t < n; t++) { double p = P[(size_t)t*rows+r]; sp2 += p*p; spy += p*(double)Y[(size_t)t*rows+r]; }
            double lam = sp2 / (n + 1.0), l0 = 1e-3*sp2 + 1e-9; if (lam < l0) lam = l0;
            double gr = (spy + lam) / (sp2 + lam);
            if (gr < 0.25) gr = 0.25; else if (gr > 4.0) gr = 4.0;
            float *o = Wq_out + (size_t)r*cols;
            for (int j = 0; j < cols; j++) o[j] *= (float)gr;
        }
        free(P); free(Y);
    }
}

void go2b_encode_cext(const float *W, int rows, int cols, const float *X, int n, float *Wq_out) {
    float *q = dq_quant_expert_go2b(W, rows, cols, X, n);
    memcpy(Wq_out, q, (size_t)rows*cols*4); free(q);
}
void signref_encode_cext(const float *W, int rows, int cols, const float *X, int n, float *Wq_out) {
    float *q = dq_quant_expert_signref(W, rows, cols, X, n);
    memcpy(Wq_out, q, (size_t)rows*cols*4); free(q);
}
