/* onebit_quant.c — GO1B strict-binary (±1) 1-bit row quantizer.
 * See onebit_quant.h for the authoritative byte layout and the per-row scale math. */

#include "onebit_quant.h"
#include <stdlib.h>

size_t go1b_row_bytes(int64_t ncols) {
    if (ncols <= 0) return sizeof(uint16_t); /* scale only; no sign words */
    int64_t nwords = (ncols + 31) / 32;      /* ceil(ncols/32) */
    return sizeof(uint16_t) + (size_t)nwords * sizeof(uint32_t);
}

size_t go1b_quantize(const float *src, void *dst, int64_t nrows, int64_t ncols) {
    const size_t rb = go1b_row_bytes(ncols);
    uint8_t *out = (uint8_t *)dst;
    const int64_t nwords = (ncols > 0) ? (ncols + 31) / 32 : 0;

    for (int64_t r = 0; r < nrows; r++) {
        const float *rs  = src + (size_t)r * (size_t)ncols;
        uint8_t     *rd  = out + (size_t)r * rb;

        /* Per-row L2-optimal scale a = mean(|w|). Accumulate in double so very wide
         * rows (e.g. 4096+ experts cols) don't lose precision. */
        double sum_abs = 0.0;
        for (int64_t j = 0; j < ncols; j++) {
            float v = rs[j];
            sum_abs += (v < 0.0f) ? -(double)v : (double)v; /* |v|, no math.h */
        }
        float scale = (ncols > 0) ? (float)(sum_abs / (double)ncols) : 0.0f;
        go1b_store_u16_le(rd, go1b_fp32_to_fp16(scale));

        /* Pack sign bits: bit set <=> w[j] >= 0 (so it reconstructs to +scale).
         * -0.0f compares >= 0 and maps to +scale, which is L2-equivalent to -scale. */
        uint8_t *wp = rd + sizeof(uint16_t);
        for (int64_t wi = 0; wi < nwords; wi++) {
            uint32_t word = 0;
            int64_t base = wi * 32;
            int64_t lim  = base + 32;
            if (lim > ncols) lim = ncols;            /* final partial word */
            for (int64_t j = base; j < lim; j++) {
                if (rs[j] >= 0.0f) word |= (1u << (uint32_t)(j - base));
            }
            /* bits >= ncols in the last word stay 0 -> zero-padded. */
            go1b_store_u32_le(wp, word);
            wp += sizeof(uint32_t);
        }
    }
    return (size_t)nrows * rb;
}

void go1b_dequantize_row(const void *row, float *dst, int64_t ncols) {
    const uint8_t *p = (const uint8_t *)row;
    float scale = go1b_fp16_to_fp32(go1b_load_u16_le(p));
    const uint8_t *wp = p + sizeof(uint16_t);
    const int64_t nwords = (ncols > 0) ? (ncols + 31) / 32 : 0;

    for (int64_t wi = 0; wi < nwords; wi++) {
        uint32_t word = go1b_load_u32_le(wp + (size_t)wi * sizeof(uint32_t));
        int64_t base = wi * 32;
        int64_t lim  = base + 32;
        if (lim > ncols) lim = ncols;
        for (int64_t j = base; j < lim; j++) {
            uint32_t bit = (word >> (uint32_t)(j - base)) & 1u;
            dst[j] = bit ? scale : -scale; /* exact ±scale */
        }
    }
}

/* ===========================================================================
 * BLOCK layout (block_go1b, 256-element fixed block) — see onebit_quant.h.
 * This is the format the ds4 runtime / Metal kernel actually consumes (ggml type
 * 40). The per-row scale is replicated into every block so blocks stream uniformly.
 * =========================================================================== */

size_t go1b_blk_row_bytes(int64_t ncols) {
    if (ncols <= 0) return 0;
    /* ncols is required to be a multiple of GO1B_BLK_QK; round up defensively so a
     * stray tail still occupies a whole block instead of being silently dropped. */
    int64_t nblocks = (ncols + GO1B_BLK_QK - 1) / GO1B_BLK_QK;
    return (size_t)nblocks * GO1B_BLK_BYTES;
}

size_t go1b_blk_quantize(const float *src, void *dst, int64_t nrows, int64_t ncols) {
    const int64_t nblocks  = (ncols > 0) ? (ncols + GO1B_BLK_QK - 1) / GO1B_BLK_QK : 0;
    const size_t  row_bytes = (size_t)nblocks * GO1B_BLK_BYTES;
    uint8_t *out = (uint8_t *)dst;

    for (int64_t r = 0; r < nrows; r++) {
        const float *rs = src + (size_t)r * (size_t)ncols;
        uint8_t     *rd = out + (size_t)r * row_bytes;

        /* Per-row L2-optimal scale a = mean(|w|) over the WHOLE row; accumulate in
         * double so wide expert rows (2048/4096 cols) don't lose precision. */
        double sum_abs = 0.0;
        for (int64_t j = 0; j < ncols; j++) {
            float v = rs[j];
            sum_abs += (v < 0.0f) ? -(double)v : (double)v; /* |v|, no math.h */
        }
        float scale = (ncols > 0) ? (float)(sum_abs / (double)ncols) : 0.0f;
        const uint16_t hd = go1b_fp32_to_fp16(scale);

        /* One 34-byte block per 256 columns. d = per-row scale, replicated into every
         * block (per-block local scale variant retired 2026-08-31 with the env purge);
         * signs packed bit j -> signs[j/8] bit (j%8). */
        for (int64_t b = 0; b < nblocks; b++) {
            uint8_t *bd = rd + (size_t)b * GO1B_BLK_BYTES;
            const int64_t base = b * GO1B_BLK_QK;
            go1b_store_u16_le(bd, hd);                 /* offset 0: fp16 scale */
            uint8_t *sg  = bd + sizeof(uint16_t);      /* offset 2: 32 sign bytes */
            for (int k = 0; k < GO1B_BLK_QK / 8; k++) { /* 32 bytes = 256 sign bits */
                uint8_t byte = 0;
                for (int bit = 0; bit < 8; bit++) {
                    int64_t j = base + (int64_t)k * 8 + bit; /* global element index */
                    /* bit set <=> w[j] >= 0 -> +scale (-0.0f maps +); past ncols stays 0. */
                    if (j < ncols && rs[j] >= 0.0f) byte |= (uint8_t)(1u << bit);
                }
                sg[k] = byte;
            }
        }
    }
    return (size_t)nrows * row_bytes;
}

/* nblk×nblk 对称正定稠密解 A·s=b (高斯消元+部分主元). n≤64. 奇异→false. */
static int go1b_solve_dense(double *A, const double *bb, double *x, int n) {
    double b[64];
    for (int i = 0; i < n; i++) b[i] = bb[i];
    for (int col = 0; col < n; col++) {
        int piv = col; double best = A[col*n+col] < 0 ? -A[col*n+col] : A[col*n+col];
        for (int r = col+1; r < n; r++) { double v = A[r*n+col]<0?-A[r*n+col]:A[r*n+col]; if (v>best){best=v;piv=r;} }
        if (best < 1e-18) return 0;
        if (piv != col) { for (int k=0;k<n;k++){double t=A[col*n+k];A[col*n+k]=A[piv*n+k];A[piv*n+k]=t;} double t=b[col];b[col]=b[piv];b[piv]=t; }
        double d = A[col*n+col];
        for (int r = col+1; r < n; r++) {
            double f = A[r*n+col]/d;
            if (f == 0.0) continue;
            for (int k=col;k<n;k++) A[r*n+k] -= f*A[col*n+k];
            b[r] -= f*b[col];
        }
    }
    for (int i = n-1; i >= 0; i--) {
        double s = b[i];
        for (int k=i+1;k<n;k++) s -= A[i*n+k]*x[k];
        x[i] = s / A[i*n+i];
    }
    return 1;
}

/* ★joint-LS 输出最优 per-block scale (numpy go1b_q 的 C 实现, 79% 质量关键)★
 * 固定 signs=sign(w), per-row 联合解 nblk 个 block scale s_b 使
 *   Σ_b s_b·(B_b·x) ≈ w·x  最小二乘, 用全激活 X[n_act×ncols] (行主序)。
 *   A[b][c]=Σ_t P_tb·P_tc, rhs[b]=Σ_t P_tb·Y_t; P_tb=Σ_{j∈块b}X_tj·sign(w_j), Y_t=Σ_j X_tj·w_j;
 *   ridge A_bb+=1e-3·tr/nblk。块大小=GO1B_BLK_QK(256,匹配格式,对2048列w2也对)。
 * X=NULL / n_act<4 / 解奇异 → 该行退回 mean|w| per-block (永不比 baseline 差)。
 *
 * ★2026-07-10 病态修复(差层可救的根因)★: n_act(≈6 校准行) < nblk(16 未知) 时 LS 严重欠定,
 * 旧 λ=1e-3·tr/nblk 压不住 → 欠定方向解由噪声定, 且负/爆 scale 直接写盘(负 scale=整块符号
 * 翻转) → 该行输出比零还差 = 差层局部 R² −100%~−300% 的机制。修复 = MAP(ridge-to-prior):
 * 收缩目标从 0 改为 s0[b]=块 mean|w|(零样本最优), λ 随欠定度自适应; 解出负→回 s0, 爆值→clip。
 * 计数器(线程安全)供逐层报告验证。 */
long go1b_joint_neg = 0, go1b_joint_clip = 0, go1b_joint_fail = 0, go1b_joint_rows = 0;

size_t go1b_blk_quantize_joint(const float *src, void *dst, int64_t nrows, int64_t ncols,
                               const float *X, int64_t n_act) {
    const int64_t nblk = (ncols > 0) ? (ncols + GO1B_BLK_QK - 1) / GO1B_BLK_QK : 0;
    const size_t  row_bytes = (size_t)nblk * GO1B_BLK_BYTES;
    if (!X || n_act < 4 || nblk < 1 || nblk > 64)
        return go1b_blk_quantize_imat(src, dst, nrows, ncols, NULL);
    uint8_t *out = (uint8_t *)dst;
    float  *P = (float *)malloc((size_t)n_act * (size_t)nblk * sizeof(float));
    double *Y = (double *)malloc((size_t)n_act * sizeof(double));
    if (!P || !Y) { free(P); free(Y); return go1b_blk_quantize_imat(src, dst, nrows, ncols, NULL); }
    for (int64_t r = 0; r < nrows; r++) {
        const float *w  = src + (size_t)r * (size_t)ncols;
        uint8_t     *rd = out + (size_t)r * row_bytes;
        for (int64_t t = 0; t < n_act; t++) {
            const float *x = X + (size_t)t * (size_t)ncols;
            double y = 0.0;
            for (int64_t b = 0; b < nblk; b++) {
                double p = 0.0; const int64_t j0 = b * GO1B_BLK_QK;
                for (int64_t k = 0; k < GO1B_BLK_QK; k++) {
                    int64_t j = j0 + k; if (j >= ncols) break;
                    p += (w[j] >= 0.0f) ? (double)x[j] : -(double)x[j];
                }
                P[t * nblk + b] = (float)p;
            }
            for (int64_t j = 0; j < ncols; j++) y += (double)x[j] * (double)w[j];
            Y[t] = y;
        }
        /* 先验 s0[b] = 块 mean|w| (零样本下的 L2 最优 scale = 收缩目标) */
        double s0[64];
        for (int64_t b = 0; b < nblk; b++) {
            double bsum = 0.0; int64_t bn = 0; const int64_t j0 = b * GO1B_BLK_QK;
            for (int k = 0; k < GO1B_BLK_QK; k++) { int64_t j = j0+k; if (j<ncols){ float v=w[j]; bsum += v<0?-(double)v:(double)v; bn++; } }
            s0[b] = (bn>0) ? bsum/(double)bn : 0.0;
        }
        double A[64 * 64], rhs[64], S[64];
        for (int64_t b = 0; b < nblk; b++) { rhs[b] = 0.0; for (int64_t c = 0; c < nblk; c++) A[b*nblk+c] = 0.0; }
        for (int64_t t = 0; t < n_act; t++) {
            const float *p = P + t * nblk;
            for (int64_t b = 0; b < nblk; b++) {
                rhs[b] += (double)p[b] * Y[t];
                for (int64_t c = 0; c < nblk; c++) A[b*nblk+c] += (double)p[b] * (double)p[c];
            }
        }
        double tr = 0.0; for (int64_t b = 0; b < nblk; b++) tr += A[b*nblk+b];
        /* ridge-to-prior: 解 (A+λI)s = rhs+λ·s0。λ=tr/nblk · nblk/(n_act+nblk):
         * 欠定(n_act≪nblk)→ λ≈tr/nblk 强收缩到 s0; 样本多 → λ→tr/n_act 数据主导。旧值为下限。 */
        double lam = (tr / (double)nblk) * ((double)nblk / ((double)n_act + (double)nblk));
        double lam0 = 1e-3 * (tr / (double)nblk) + 1e-9;
        if (lam < lam0) lam = lam0;
        for (int64_t b = 0; b < nblk; b++) { A[b*nblk+b] += lam; rhs[b] += lam * s0[b]; }
        int ok = go1b_solve_dense(A, rhs, S, (int)nblk);
        __sync_fetch_and_add(&go1b_joint_rows, (long)nblk);
        for (int64_t b = 0; b < nblk; b++) {
            uint8_t *bd = rd + (size_t)b * GO1B_BLK_BYTES;
            const int64_t base = b * GO1B_BLK_QK;
            float sc;
            if (ok && S[b] == S[b]) {
                double s = S[b];
                if (s < 0.0) { s = s0[b]; __sync_fetch_and_add(&go1b_joint_neg, 1); }               /* 负 scale 无意义(符号已在 bit) */
                else if (s > 4.0 * s0[b] + 1e-12) { s = 4.0 * s0[b]; __sync_fetch_and_add(&go1b_joint_clip, 1); } /* 爆值防护 */
                sc = (float)s;
            } else { sc = (float)s0[b]; __sync_fetch_and_add(&go1b_joint_fail, 1); }                /* 解失败 → 先验 */
            go1b_store_u16_le(bd, go1b_fp32_to_fp16(sc));
            uint8_t *sg = bd + sizeof(uint16_t);
            for (int k = 0; k < GO1B_BLK_QK/8; k++) {
                uint8_t byte = 0;
                for (int bit = 0; bit < 8; bit++) { int64_t j = base + (int64_t)k*8 + bit; if (j<ncols && w[j]>=0.0f) byte |= (uint8_t)(1u<<bit); }
                sg[k] = byte;
            }
        }
    }
    free(P); free(Y);
    return (size_t)nrows * row_bytes;
}

/* Input-aware variant (L_fix): ew[ncols] = Go-domain per-input-channel second
 * moments E[x_j²] (per-expert slice of the imatrix). Sign stays sign(w); only
 * the scale re-fits so the OUTPUT error (w−s·b)·x is minimized under the
 * diagonal activation model:  s* = Σ_j ew_j·|w_j| / Σ_j ew_j   (closed form).
 * ew NULL / non-positive mass → identical to go1b_blk_quantize (mean|w|). */
size_t go1b_blk_quantize_imat(const float *src, void *dst, int64_t nrows, int64_t ncols,
                              const float *ew) {
    if (!ew) return go1b_blk_quantize(src, dst, nrows, ncols);
    const int64_t nblocks  = (ncols > 0) ? (ncols + GO1B_BLK_QK - 1) / GO1B_BLK_QK : 0;
    const size_t  row_bytes = (size_t)nblocks * GO1B_BLK_BYTES;
    uint8_t *out = (uint8_t *)dst;

    for (int64_t r = 0; r < nrows; r++) {
        const float *rs = src + (size_t)r * (size_t)ncols;
        uint8_t     *rd = out + (size_t)r * row_bytes;

        double num = 0.0, den = 0.0, sum_abs = 0.0;
        for (int64_t j = 0; j < ncols; j++) {
            float v = rs[j];
            double av = (v < 0.0f) ? -(double)v : (double)v;
            double w = (double)ew[j];
            if (w > 0.0) { num += w * av; den += w; }
            sum_abs += av;
        }
        /* degenerate weights (expert never fired / all-zero channel stats) →
         * fall back to the unweighted mean so Θ_fix never regresses there */
        float scale = (den > 0.0) ? (float)(num / den)
                                  : ((ncols > 0) ? (float)(sum_abs / (double)ncols) : 0.0f);
        const uint16_t hd = go1b_fp32_to_fp16(scale);

        for (int64_t b = 0; b < nblocks; b++) {
            uint8_t *bd = rd + (size_t)b * GO1B_BLK_BYTES;
            const int64_t base = b * GO1B_BLK_QK;
            go1b_store_u16_le(bd, hd);
            uint8_t *sg  = bd + sizeof(uint16_t);
            for (int k = 0; k < GO1B_BLK_QK / 8; k++) {
                uint8_t byte = 0;
                for (int bit = 0; bit < 8; bit++) {
                    int64_t j = base + (int64_t)k * 8 + bit;
                    if (j < ncols && rs[j] >= 0.0f) byte |= (uint8_t)(1u << bit);
                }
                sg[k] = byte;
            }
        }
    }
    return (size_t)nrows * row_bytes;
}

void go1b_blk_dequantize_row(const void *row, float *dst, int64_t ncols) {
    const uint8_t *p = (const uint8_t *)row;
    const int64_t nblocks = (ncols > 0) ? (ncols + GO1B_BLK_QK - 1) / GO1B_BLK_QK : 0;
    for (int64_t b = 0; b < nblocks; b++) {
        const uint8_t *bd = p + (size_t)b * GO1B_BLK_BYTES;
        float scale = go1b_fp16_to_fp32(go1b_load_u16_le(bd)); /* this block's copy */
        const uint8_t *sg = bd + sizeof(uint16_t);
        const int64_t base = b * GO1B_BLK_QK;
        for (int j = 0; j < GO1B_BLK_QK; j++) {
            int64_t gj = base + j;
            if (gj >= ncols) break;
            uint8_t bit = (uint8_t)((sg[j >> 3] >> (j & 7)) & 1u);
            dst[gj] = bit ? scale : -scale; /* exact ±scale, matches Metal dequant */
        }
    }
}

/* ===========================================================================
 * Self-test:  cc -std=c99 -O2 -DGO1B_TEST onebit_quant.c -o a1test -lm
 * =========================================================================== */
#ifdef GO1B_TEST
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>   /* test-only: sqrt/log/cos for Gaussian sampling + L2 metric */
#ifndef M_PI
#define M_PI 3.14159265358979323846  /* Linux -std=c11 严格模式 math.h 不给(Darwin 给) */
#endif

#define GO1B_PI 3.14159265358979323846 /* M_PI is not standard C99 */

/* Box–Muller standard-normal sample. */
static float randn(void) {
    double u1 = (rand() + 1.0) / ((double)RAND_MAX + 2.0); /* in (0,1) */
    double u2 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * GO1B_PI * u2));
}

int main(void) {
    srand(1234);

    /* 7 x 130: 130 = 4*32 + 2, so ceil(130/32)=5 words and the LAST word is partial
     * (only bits 0,1 valid; bits 2..31 zero-padded) — exercises the partial-word path. */
    const int64_t nrows = 7, ncols = 130;
    const int64_t nwords = (ncols + 31) / 32;
    assert(nwords == 5);

    float *W = (float *)malloc(sizeof(float) * (size_t)(nrows * ncols));
    for (int64_t i = 0; i < nrows * ncols; i++) W[i] = randn();

    size_t rb = go1b_row_bytes(ncols);
    size_t rb_expect = sizeof(uint16_t) + (size_t)nwords * sizeof(uint32_t); /* 2 + 20 = 22 */
    printf("row_bytes = %zu (expected %zu)\n", rb, rb_expect);
    assert(rb == rb_expect);

    void *buf = malloc(rb * (size_t)nrows);
    size_t total = go1b_quantize(W, buf, nrows, ncols);
    printf("total_bytes = %zu (expected %zu)\n", total, rb * (size_t)nrows);
    assert(total == rb * (size_t)nrows);

    /* Verify the last word's pad bits are physically zero on disk. */
    for (int64_t r = 0; r < nrows; r++) {
        const uint8_t *row = (const uint8_t *)buf + (size_t)r * rb;
        uint32_t last = go1b_load_u32_le(row + sizeof(uint16_t) + (size_t)(nwords - 1) * 4);
        uint32_t valid_bits = (uint32_t)(ncols - (nwords - 1) * 32); /* = 2 */
        uint32_t pad_mask = ~((valid_bits >= 32) ? 0xffffffffu : ((1u << valid_bits) - 1u));
        assert((last & pad_mask) == 0u);
    }

    float *deq = (float *)malloc(sizeof(float) * (size_t)ncols);
    double err_acc = 0.0;
    for (int64_t r = 0; r < nrows; r++) {
        const uint8_t *row = (const uint8_t *)buf + (size_t)r * rb;
        float scale = go1b_fp16_to_fp32(go1b_load_u16_le(row));
        go1b_dequantize_row(row, deq, ncols);

        double num = 0.0, den = 0.0;
        for (int64_t j = 0; j < ncols; j++) {
            float w = W[r * ncols + j];
            /* every dequant value is EXACTLY +scale or -scale ... */
            assert(deq[j] == scale || deq[j] == -scale);
            /* ... and matches the encoder's sign convention (w>=0 -> +scale). */
            float expect = (w >= 0.0f) ? scale : -scale;
            assert(deq[j] == expect);
            double d = (double)w - (double)deq[j];
            num += d * d;
            den += (double)w * (double)w;
        }
        double rel = sqrt(num / den);
        err_acc += rel;
        printf("row %lld: scale=%.6f  relL2=%.4f\n", (long long)r, (double)scale, rel);
    }
    double mean_rel = err_acc / (double)nrows;
    printf("mean relative L2 error = %.4f  (expect ~%.4f = sqrt(1-2/pi))\n",
           mean_rel, sqrt(1.0 - 2.0 / GO1B_PI));

    /* ---- block_go1b (256-element fixed block) format test ---------------------
     * 4 x 512: 512 = 2*256 -> exactly 2 full 34-byte blocks per row (no partial). */
    {
        const int64_t bnrows = 4, bncols = 512;
        const int64_t bnblocks = bncols / GO1B_BLK_QK; /* 2 blocks per row */
        assert(GO1B_BLK_BYTES == 34 && GO1B_BLK_QK == 256);

        float *BW = (float *)malloc(sizeof(float) * (size_t)(bnrows * bncols));
        for (int64_t i = 0; i < bnrows * bncols; i++) BW[i] = randn();

        size_t brb = go1b_blk_row_bytes(bncols);
        printf("\n[block] row_bytes = %zu (expected %zu)\n", brb, (size_t)bnblocks * GO1B_BLK_BYTES);
        assert(brb == (size_t)bnblocks * GO1B_BLK_BYTES); /* 2*34 = 68 */

        void *bbuf = malloc(brb * (size_t)bnrows);
        size_t btotal = go1b_blk_quantize(BW, bbuf, bnrows, bncols);
        size_t btotal_expect = (size_t)bnrows * (size_t)(bncols / GO1B_BLK_QK) * GO1B_BLK_BYTES;
        printf("[block] total_bytes = %zu (expected nrows*(ncols/256)*34 = %zu)\n", btotal, btotal_expect);
        assert(btotal == btotal_expect);          /* 4*(512/256)*34 = 272 */
        assert(btotal == brb * (size_t)bnrows);

        /* The per-row scale must be byte-identical (replicated) in every block. */
        for (int64_t r = 0; r < bnrows; r++) {
            const uint8_t *rowp = (const uint8_t *)bbuf + (size_t)r * brb;
            uint16_t d0 = go1b_load_u16_le(rowp);
            for (int64_t b = 1; b < bnblocks; b++)
                assert(go1b_load_u16_le(rowp + (size_t)b * GO1B_BLK_BYTES) == d0);
        }

        float *bdeq = (float *)malloc(sizeof(float) * (size_t)bncols);
        for (int64_t r = 0; r < bnrows; r++) {
            const uint8_t *rowp = (const uint8_t *)bbuf + (size_t)r * brb;
            float scale = go1b_fp16_to_fp32(go1b_load_u16_le(rowp));
            go1b_blk_dequantize_row(rowp, bdeq, bncols);
            for (int64_t j = 0; j < bncols; j++) {
                float w = BW[r * bncols + j];
                assert(bdeq[j] == scale || bdeq[j] == -scale);      /* exactly ±scale */
                assert(bdeq[j] == ((w >= 0.0f) ? scale : -scale));  /* sign convention */
            }
            if (r < 2)
                printf("[block] row %lld: scale=%.6f  w[0..2]=% .3f % .3f % .3f  deq[0..2]=% .3f % .3f % .3f\n",
                       (long long)r, (double)scale,
                       (double)BW[r*bncols], (double)BW[r*bncols+1], (double)BW[r*bncols+2],
                       (double)bdeq[0], (double)bdeq[1], (double)bdeq[2]);
        }
        printf("[block] dequant exact-sign check PASSED (%lld rows x %lld cols, %lld blocks/row)\n",
               (long long)bnrows, (long long)bncols, (long long)bnblocks);
        free(BW); free(bbuf); free(bdeq);
    }

    /* ---- input-aware (L_fix) weighted-scale fit: s* = Σ ew|w| / Σ ew -------- */
    {
        const int64_t incols = 512, inrows = 2;
        float *IW = (float *)malloc((size_t)inrows * incols * sizeof(float));
        float *ew = (float *)malloc((size_t)incols * sizeof(float));
        uint8_t *ibuf = (uint8_t *)malloc((size_t)inrows * go1b_blk_row_bytes(incols));
        float *ideq = (float *)malloc((size_t)incols * sizeof(float));
        for (int64_t i = 0; i < inrows * incols; i++) IW[i] = randn();
        for (int64_t j = 0; j < incols; j++) ew[j] = (j < incols / 2) ? 4.0f : 0.25f;
        go1b_blk_quantize_imat(IW, ibuf, inrows, incols, ew);
        for (int64_t r = 0; r < inrows; r++) {
            double num = 0, den = 0;
            for (int64_t j = 0; j < incols; j++) {
                double av = fabs((double)IW[r * incols + j]);
                num += (double)ew[j] * av; den += (double)ew[j];
            }
            float want = go1b_fp16_to_fp32(go1b_fp32_to_fp16((float)(num / den)));
            go1b_blk_dequantize_row(ibuf + (size_t)r * go1b_blk_row_bytes(incols), ideq, incols);
            for (int64_t j = 0; j < incols; j++) {
                assert(fabsf(fabsf(ideq[j]) - want) < 1e-7f);                    /* weighted scale */
                assert(ideq[j] == ((IW[r * incols + j] >= 0.0f) ? want : -want)); /* sign unchanged */
            }
        }
        /* NULL ew must be bit-identical to the unweighted encoder */
        uint8_t *a = (uint8_t *)malloc((size_t)inrows * go1b_blk_row_bytes(incols));
        uint8_t *b = (uint8_t *)malloc((size_t)inrows * go1b_blk_row_bytes(incols));
        go1b_blk_quantize(IW, a, inrows, incols);
        go1b_blk_quantize_imat(IW, b, inrows, incols, NULL);
        assert(memcmp(a, b, (size_t)inrows * go1b_blk_row_bytes(incols)) == 0);
        printf("[imat] weighted-scale fit + NULL-fallback PASSED\n");
        free(IW); free(ew); free(ibuf); free(ideq); free(a); free(b);
    }

    free(W); free(buf); free(deq);
    printf("ALL ASSERTS PASSED\n");
    return 0;
}
#endif /* GO1B_TEST */
