/* v41_calib.cu — 量化校准料 → 列权(2026-09-20 深夜, 用户令"按金融域量化")。读 v41_amp_run --dump-calib 落的
 * calib_Lnn.bin(格式见 v41_calib.h), 给量化器两种列权:
 *   ① 层级 E[x²][D]: 全部 token 的 MoE 入口 x 二阶矩, 权重 Σ_k rw_k²(一个 token 的 x 进 n_used 个专家, 各带各的 rw)。
 *      w1/w3 的输入就是它 ⇒ 全层 384 个专家共用一份 —— 每专家只有 ~128 行, 逐专家估 5120 个数太薄, 层级 8192 行够。
 *   ② 逐专家 E[h_e²][MID]: 专家 e 名下的 token 在【出厂 FP4 权重】上算 h = silu(clamp(x·W1ᵀ))·clamp(x·W3ᵀ), rw² 加权。
 *      w2 的输入是专家自己的 h ⇒ 必须逐专家。名下 < 8 行不算, 调用方退回平权(= 今天的行为, 冷门专家不退步)。
 * 【为什么 rw²】y 的误差 = Σ_k rw_k·(W_k − Ŵ_k)x, 专家 e 那一项对 ‖误差‖² 的贡献带 rw_e²。
 * 【口径】列权只作相对权重(均值归一到 1, 下限 1e-4): 它进 VQ 的加权距离与加权质心, 绝对尺度无意义。
 * 全 GPU(cuBLAS sgemm + 归约核), 无 CPU 参考路; swiglu 与 v41_fp_moe.cu 的 k_swiglu 同式。
 * 出错会怎样: 文件头/形状对不上直接失败(err 写明), 不猜 —— 校准料错位不报错只出假数。 */
#include "v41_dq.cuh"
#include "v41_calib.h"
#include <cublas_v2.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct v41_calib {
    v41_calib_hdr h; int n_expert;
    uint16_t *xb; int *sel; float *rw;      /* 主机: 整层 */
    float *colw_x;                           /* 主机 [D] */
    int *eoff, *erow; float *erw;            /* 逐专家行表(CSR): token 行号 + 该 pick 的 rw */
    uint8_t *isval;                          /* 主机 [ntok] 1 = val 行(calib.layout 窗分层, 与 v41_amp_run 的 layout_split 同规则) */
    float *hx; size_t hx_rows;               /* 主机: 本专家的 x 行 f32 */
    v41_dbuf dx, dw1, dw3, dg, du, dacc, scr;   /* 设备工作区 */
    float *drw_e; size_t drw_cap; v41_dbuf dval;
    v41_dbuf dH, dxs;                        /* 层级 Gram(级 2)与它的暂存 */
    cublasHandle_t bl;
};

/* calib.layout(取料时从 <ids>.layout 抄来): 每域内窗号 k%4==3 的窗作 val —— 与 v41_amp_run layout_split 同一条规则,
 * 级 2 的 β 二选一只认 val 行的输出误差。没有这个文件 = 全部当拟合行(β 退回默认)。 */
static void load_layout(v41_calib *c, const char *dir) {
    char p[4300]; snprintf(p, sizeof p, "%s/calib.layout", dir);
    FILE *f = fopen(p, "r"); if (!f) return;
    int win = 128; char line[512];
    while (fgets(line, sizeof line, f)) {
        char dom[128]; int off, cnt;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "win %d", &win) == 1) continue;
        if (sscanf(line, "%127s %d %d", dom, &off, &cnt) != 3 || win <= 0) continue;
        for (int k = 3; k < cnt / win; k += 4) for (int r = off + k * win; r < off + (k + 1) * win; r++) if (r >= 0 && r < c->h.ntok) c->isval[r] = 1;
    }
    fclose(f);
}

#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "★cuBLAS %s @%d: %d★\n", #x, __LINE__, (int)s_); return -1; } } while (0)

/* h 列权归约: 一线程一列, 顺着 m 行累加(m ≤ 几百, 顺序固定 ⇒ 可复现) */
__global__ static void k_h_colw(const float *g, const float *u, const float *rw2, int m, int MID, float clamp, float *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= MID) return;
    float acc = 0.f;
    for (int i = 0; i < m; i++) {
        float gv = g[(size_t)i * MID + j], uv = u[(size_t)i * MID + j];
        if (clamp > 0.f) { if (gv > clamp) gv = clamp; if (uv > clamp) uv = clamp; if (uv < -clamp) uv = -clamp; }
        float hv = (gv / (1.f + expf(-gv))) * uv;
        acc += rw2[i] * hv * hv;
    }
    out[j] = acc;
}

static float g_alpha = 1.f;
extern "C" void v41_calib_set_alpha(float alpha) { g_alpha = alpha; }
static void normalize_mean1(float *w, int n) {
    if (g_alpha != 1.f) for (int i = 0; i < n; i++) w[i] = w[i] > 0.f ? powf(w[i], g_alpha) : 0.f;
    double s = 0; for (int i = 0; i < n; i++) s += w[i];
    const double mean = n ? s / n : 1.0;
    for (int i = 0; i < n; i++) { double v = mean > 0 ? w[i] / mean : 1.0; if (v < 1e-4) v = 1e-4; w[i] = (float)v; }
}

extern "C" v41_calib *v41_calib_open(const char *dir, int L, int n_expert, char *err, size_t errn) {
    char p[4300]; snprintf(p, sizeof p, "%s/calib_L%02d.bin", dir, L);
    FILE *f = fopen(p, "rb");
    if (!f) { snprintf(err, errn, "打不开 %s", p); return NULL; }
    v41_calib *c = (v41_calib *)calloc(1, sizeof *c);
    c->n_expert = n_expert;
    if (fread(&c->h, sizeof c->h, 1, f) != 1 || memcmp(c->h.magic, V41_CALIB_MAGIC, 4) || c->h.ntok <= 0 || c->h.D <= 0 || c->h.n_used <= 0 || c->h.n_used > 16) {
        snprintf(err, errn, "%s 头不合法", p); fclose(f); free(c); return NULL; }
    const size_t nD = (size_t)c->h.ntok * c->h.D, nU = (size_t)c->h.ntok * c->h.n_used;
    c->xb = (uint16_t *)malloc(nD * 2); c->sel = (int *)malloc(nU * 4); c->rw = (float *)malloc(nU * 4);
    c->colw_x = (float *)calloc((size_t)c->h.D, sizeof(float));
    c->eoff = (int *)calloc((size_t)n_expert + 1, sizeof(int)); c->erow = (int *)malloc(nU * 4); c->erw = (float *)malloc(nU * 4);
    c->isval = (uint8_t *)calloc((size_t)c->h.ntok, 1);
    if (!c->xb || !c->sel || !c->rw || !c->colw_x || !c->eoff || !c->erow || !c->erw || !c->isval) { snprintf(err, errn, "主机缓冲分配失败"); fclose(f); v41_calib_close(c); return NULL; }
    int ok = fread(c->xb, 2, nD, f) == nD && fread(c->sel, 4, nU, f) == nU && fread(c->rw, 4, nU, f) == nU;
    fclose(f);
    if (!ok) { snprintf(err, errn, "%s 读不全(期望 %zu 字节)", p, v41_calib_size(&c->h)); v41_calib_close(c); return NULL; }
    load_layout(c, dir);
    /* 层级 E[x²]: 权重 Σ_k rw_k² */
    double *e2 = (double *)calloc((size_t)c->h.D, sizeof(double)); double wsum = 0;
    int *cnt = (int *)calloc((size_t)n_expert, sizeof(int));
    for (int i = 0; i < c->h.ntok; i++) {
        double wi = 0;
        for (int k = 0; k < c->h.n_used; k++) {
            const int e = c->sel[(size_t)i * c->h.n_used + k]; const float r = c->rw[(size_t)i * c->h.n_used + k];
            if (e < 0 || e >= n_expert) { snprintf(err, errn, "行 %d pick %d 专家号 %d 越界", i, k, e); free(e2); free(cnt); v41_calib_close(c); return NULL; }
            wi += (double)r * r; cnt[e]++;
        }
        const uint16_t *xr = c->xb + (size_t)i * c->h.D;
        for (int d = 0; d < c->h.D; d++) { uint32_t u = (uint32_t)xr[d] << 16; float v; memcpy(&v, &u, 4); e2[d] += wi * (double)v * v; }
        wsum += wi;
    }
    for (int d = 0; d < c->h.D; d++) c->colw_x[d] = (float)(wsum > 0 ? e2[d] / wsum : 1.0);
    /* 自检数(归一前): 前 12 通道能量占比 */
    double tot = 0, top = 0;
    for (int d = 0; d < c->h.D; d++) tot += e2[d];
    for (int t = 0; t < 12; t++) { int bi = 0; for (int d = 1; d < c->h.D; d++) if (e2[d] > e2[bi]) bi = d; top += e2[bi]; e2[bi] = -1; }
    free(e2);
    normalize_mean1(c->colw_x, c->h.D);
    /* 逐专家 CSR */
    for (int e = 0; e < n_expert; e++) c->eoff[e + 1] = c->eoff[e] + cnt[e];
    int *cur = (int *)malloc(sizeof(int) * (size_t)n_expert); memcpy(cur, c->eoff, sizeof(int) * (size_t)n_expert);
    int lt8 = 0, mn = cnt[0], mx = cnt[0];
    for (int e = 0; e < n_expert; e++) { if (cnt[e] < 8) lt8++; if (cnt[e] < mn) mn = cnt[e]; if (cnt[e] > mx) mx = cnt[e]; }
    for (int i = 0; i < c->h.ntok; i++) for (int k = 0; k < c->h.n_used; k++) {
        const int e = c->sel[(size_t)i * c->h.n_used + k]; const int pos = cur[e]++;
        c->erow[pos] = i; c->erw[pos] = c->rw[(size_t)i * c->h.n_used + k];
    }
    free(cur); free(cnt);
    int nval = 0; for (int i = 0; i < c->h.ntok; i++) nval += c->isval[i];
    printf("  [校准] L%02d 行 %d(val %d)  E[x²] top12 通道占 %.1f%%  名下行数 最少/最多 %d/%d, <8 行(退回平权) %d 个专家, clamp %g\n",
           L, c->h.ntok, nval, 100.0 * top / (tot > 0 ? tot : 1), mn, mx, lt8, c->h.clamp);
    return c;
}

extern "C" const float *v41_calib_colw_x(const v41_calib *c) { return c->colw_x; }
extern "C" int v41_calib_rows(const v41_calib *c, int e) { return (e >= 0 && e < c->n_expert) ? c->eoff[e + 1] - c->eoff[e] : 0; }
extern "C" int v41_calib_D(const v41_calib *c) { return c->h.D; }
extern "C" float v41_calib_clamp(const v41_calib *c) { return c->h.clamp; }

/* 专家 e 名下行 → 设备(x f32 / rw² / val 标记); colw_h 与级 2 共用这一份上传 */
static int expert_rows_up(v41_calib *c, int e, int *m_out) {
    const int D = c->h.D, m = v41_calib_rows(c, e);
    *m_out = m;
    if (m <= 0) return 0;
    if (c->hx_rows < (size_t)m) { free(c->hx); c->hx = (float *)malloc((size_t)m * D * 4); c->hx_rows = (size_t)m; if (!c->hx) return -1; }
    float *rw2 = (float *)malloc(sizeof(float) * (size_t)m), *val = (float *)malloc(sizeof(float) * (size_t)m);
    if (!rw2 || !val) { free(rw2); free(val); return -1; }
    for (int i = 0; i < m; i++) {
        const int row = c->erow[c->eoff[e] + i]; const float r = c->erw[c->eoff[e] + i];
        rw2[i] = r * r; val[i] = c->isval[row] ? 1.f : 0.f;
        const uint16_t *xr = c->xb + (size_t)row * D; float *o = c->hx + (size_t)i * D;
        for (int d = 0; d < D; d++) { uint32_t u = (uint32_t)xr[d] << 16; memcpy(&o[d], &u, 4); }
    }
    int rc = -1;
    do {
        if (v41_dbuf_need(&c->dx, (size_t)m * D * 4) || v41_dbuf_need(&c->dval, sizeof(float) * (size_t)m)) break;
        if (c->drw_cap < (size_t)m) { if (c->drw_e) cudaFree(c->drw_e); c->drw_e = NULL; if (cudaMalloc(&c->drw_e, sizeof(float) * (size_t)m) != cudaSuccess) break; c->drw_cap = (size_t)m; }
        if (cudaMemcpy(c->dx.p, c->hx, (size_t)m * D * 4, cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMemcpy(c->drw_e, rw2, sizeof(float) * (size_t)m, cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMemcpy(c->dval.p, val, sizeof(float) * (size_t)m, cudaMemcpyHostToDevice) != cudaSuccess) break;
        rc = 0;
    } while (0);
    free(rw2); free(val);
    return rc;
}

extern "C" int v41_calib_expert_x(v41_calib *c, int e, const float **dx, const float **drw2, const float **dval) {
    int m = 0;
    if (expert_rows_up(c, e, &m)) return -1;
    *dx = (const float *)c->dx.p; *drw2 = c->drw_e; *dval = (const float *)c->dval.p;
    return m;
}

/* 行按 sqrt(权) 缩放, 之后 Gram = Xsᵀ·Xs */
__global__ static void k_scale_rows(float *x, const float *w, int n, int D) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * D) return;
    x[i] *= sqrtf(w[i / D]);
}

extern "C" int v41_calib_layer_gram(v41_calib *c, const float **dH) {
    const int D = c->h.D, n = c->h.ntok;
    if (c->dH.p) { *dH = (const float *)c->dH.p; return 0; }
    if (!c->bl) CB(cublasCreate(&c->bl));
    /* 每 token 一行, 权 = Σ_k rw²(它进 n_used 个专家, 各带各的 rw); 权和归一 ⇒ H = E[x xᵀ] */
    float *w = (float *)malloc(sizeof(float) * (size_t)n); double ws = 0;
    for (int i = 0; i < n; i++) { double s = 0; for (int k = 0; k < c->h.n_used; k++) { const float r = c->rw[(size_t)i * c->h.n_used + k]; s += (double)r * r; } w[i] = (float)s; ws += s; }
    for (int i = 0; i < n; i++) w[i] = (float)(ws > 0 ? w[i] / ws : 1.0 / n);
    float *hx = (float *)malloc((size_t)n * D * 4);
    if (!w || !hx) { free(w); free(hx); return -1; }
    for (size_t i = 0; i < (size_t)n * D; i++) { uint32_t u = (uint32_t)c->xb[i] << 16; memcpy(&hx[i], &u, 4); }
    int rc = -1; float *dw = NULL;
    do {
        if (v41_dbuf_need(&c->dxs, (size_t)n * D * 4) || v41_dbuf_need(&c->dH, (size_t)D * D * 4)) break;
        if (cudaMalloc(&dw, sizeof(float) * (size_t)n) != cudaSuccess) break;
        if (cudaMemcpy(c->dxs.p, hx, (size_t)n * D * 4, cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMemcpy(dw, w, sizeof(float) * (size_t)n, cudaMemcpyHostToDevice) != cudaSuccess) break;
        k_scale_rows<<<(unsigned)(((long long)n * D + 255) / 256), 256>>>((float *)c->dxs.p, dw, n, D);
        /* 行主序 Xs[n][D] = 列主序 Xs'(D×n) ⇒ H(D×D) = Xs' · Xs'ᵀ */
        const float one = 1.f, zero = 0.f;
        if (cublasSgemm(c->bl, CUBLAS_OP_N, CUBLAS_OP_T, D, D, n, &one, (const float *)c->dxs.p, D, (const float *)c->dxs.p, D, &zero, (float *)c->dH.p, D) != CUBLAS_STATUS_SUCCESS) break;
        if (cudaDeviceSynchronize() != cudaSuccess) break;
        rc = 0;
    } while (0);
    free(w); free(hx);
    if (dw) cudaFree(dw);
    if (c->dxs.p) { cudaFree(c->dxs.p); c->dxs.p = NULL; c->dxs.cap = 0; }   /* 168 MB 暂存用完就还 */
    if (rc) { fprintf(stderr, "★层级 Gram 失败: %s★\n", cudaGetErrorString(cudaGetLastError())); return -1; }
    *dH = (const float *)c->dH.p;
    return 0;
}

extern "C" int v41_calib_colw_h(v41_calib *c, int e, const uint8_t *w1, const uint8_t *s1, const uint8_t *w3, const uint8_t *s3,
                                int MID, int D, float *out) {
    if (D != c->h.D) { fprintf(stderr, "★校准料 D=%d 与专家 D=%d 不符★\n", c->h.D, D); return -1; }
    int m = 0;
    if (expert_rows_up(c, e, &m)) return -1;
    if (m < 8) return m;
    if (!c->bl) CB(cublasCreate(&c->bl));
    int rc = -1;
    do {
        if (v41_dbuf_need(&c->dw1, (size_t)MID * D * 4) || v41_dbuf_need(&c->dw3, (size_t)MID * D * 4) ||
            v41_dbuf_need(&c->dg, (size_t)m * MID * 4) || v41_dbuf_need(&c->du, (size_t)m * MID * 4) || v41_dbuf_need(&c->dacc, (size_t)MID * 4)) break;
        int nan1 = 0, nan3 = 0;
        if (v41_upload_dequant(w1, s1, "I8", MID, D, 1, 32, (float *)c->dw1.p, &c->scr, &nan1) || v41_upload_dequant(w3, s3, "I8", MID, D, 1, 32, (float *)c->dw3.p, &c->scr, &nan3)) break;
        if (nan1 || nan3) { fprintf(stderr, "★专家 %d 出厂权重有 NaN(%d/%d)★\n", e, nan1, nan3); break; }
        /* 行主序 X[m][D]·W1ᵀ = 列主序视角 G(MID×m) = op(W1')ᵀ(MID×D) · X'(D×m) */
        const float one = 1.f, zero = 0.f;
        if (cublasSgemm(c->bl, CUBLAS_OP_T, CUBLAS_OP_N, MID, m, D, &one, (const float *)c->dw1.p, D, (const float *)c->dx.p, D, &zero, (float *)c->dg.p, MID) != CUBLAS_STATUS_SUCCESS) break;
        if (cublasSgemm(c->bl, CUBLAS_OP_T, CUBLAS_OP_N, MID, m, D, &one, (const float *)c->dw3.p, D, (const float *)c->dx.p, D, &zero, (float *)c->du.p, MID) != CUBLAS_STATUS_SUCCESS) break;
        k_h_colw<<<(unsigned)((MID + 255) / 256), 256>>>((const float *)c->dg.p, (const float *)c->du.p, c->drw_e, m, MID, c->h.clamp, (float *)c->dacc.p);
        if (cudaGetLastError() != cudaSuccess) break;
        if (cudaMemcpy(out, c->dacc.p, (size_t)MID * 4, cudaMemcpyDeviceToHost) != cudaSuccess) break;
        rc = 0;
    } while (0);
    if (rc) { fprintf(stderr, "★专家 %d 的 h 列权 GPU 段失败: %s★\n", e, cudaGetErrorString(cudaGetLastError())); return -1; }
    normalize_mean1(out, MID);
    return m;
}

extern "C" void v41_calib_close(v41_calib *c) {
    if (!c) return;
    free(c->xb); free(c->sel); free(c->rw); free(c->colw_x); free(c->eoff); free(c->erow); free(c->erw); free(c->hx); free(c->isval);
    v41_dbuf *bs[10] = {&c->dx, &c->dw1, &c->dw3, &c->dg, &c->du, &c->dacc, &c->scr, &c->dval, &c->dH, &c->dxs};
    for (int i = 0; i < 10; i++) if (bs[i]->p) cudaFree(bs[i]->p);
    if (c->drw_e) cudaFree(c->drw_e);
    if (c->bl) cublasDestroy(c->bl);
    free(c);
}
