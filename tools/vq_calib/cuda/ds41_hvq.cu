// hVQ primitives for the DeepSeek-V4.1 routed experts: sequential block
// assignment with error feedback through the input second moment H, and the
// least-squares row gain under H.
//
// H = V V^T with V upper triangular (the Cholesky factor of the reversed
// matrix, reversed back). The output error of a row is
//
//     (w - q) H (w - q)^T = || (w - q) V ||^2 = sum_B || e_B ||^2
//     e_B = (w_B - q_B) V_BB + sum_{A < B} (w_A - q_A) V_AB
//
// and e_B depends on blocks A <= B only, so choosing q_B to minimize ||e_B||
// in ascending order is GPTQ's update written with the factor of H instead of
// the factor of its inverse:
//
//     target  w~_B = w_B + r_B V_BB^-1,   r_B = sum_{A < B} (w_A - q_A) V_AB
//     metric  M_B  = V_BB V_BB^T          (8 x 8)
//     pick    c    = argmin (w~_B / g - c) M_B (w~_B / g - c)^T
//
// With a diagonal H this is axk2_vq_convert.cu's column-weighted assignment
// bit for bit: r stays zero, M_B is the column weights, and the distance adds
// its off-diagonal terms (exact zeros) after the same fused diagonal sum.
//
// The off-diagonal part of the distance is expanded, so a codeword costs 8
// products instead of 35:
//
//     2 sum_{a<b} M_ab (x_a - c_a)(x_b - c_b) = Q(x) - c . lin + Q(c)
//     lin_a = sum_{b != a} 2 M_ab x_b      (per row and block)
//     Q(c)  = 2 sum_{a<b} M_ab c_a c_b     (per codeword and block, shared by the rows)
//
// Q(x) is the same for every codeword and is dropped.
// Copyright (c) 2026 The ds4.c authors; MIT (see LICENSE-YoungAi).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

constexpr int DIM = 8;
constexpr int NOFF = DIM * (DIM - 1) / 2;   // 28 strictly upper metric entries
constexpr int NMET = DIM + NOFF;            // 36 per block: diagonal, then doubled upper
constexpr int TILE = 1024;                  // codewords per shared-memory tile (32 KiB)
constexpr int MAX_PANEL = 16;               // blocks per launch (128 columns)
constexpr int PANEL_COLS = MAX_PANEL * DIM;
constexpr int THREADS = 256;

// One block of threads = THREADS rows of one problem; grid (ceil(rows / THREADS), problems).
// Blocks j0..j1-1 of every row, in order. `rpan` holds r for the panel's columns as the
// host left it (feedback of all earlier panels); feedback inside the panel stays in the
// thread. Writes ids and the residual w - g c of the panel's columns.
__global__ void hvq_panel_kernel(const float* __restrict__ w, const float* __restrict__ rpan,
                                 float* __restrict__ res, const __half* __restrict__ gains,
                                 const float* __restrict__ v, const float* __restrict__ vinv,
                                 const float* __restrict__ met, const float* __restrict__ cb,
                                 int* __restrict__ ids, int rows, int cols, int k, int j0, int j1) {
    __shared__ float4 words4[TILE * 2];
    __shared__ float qtile[TILE];
    __shared__ float vrow[DIM][PANEL_COLS];
    __shared__ float coef[DIM * DIM + NMET];

    const int prob = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = row < rows;
    const int nblk = cols / DIM;
    const int pcols = (j1 - j0) * DIM;
    const int64_t wbase = (int64_t(prob) * rows + (live ? row : 0)) * cols;
    const float* vp = v + int64_t(prob) * cols * cols;
    const float g = live ? __half2float(gains[int64_t(prob) * rows + row]) : 0.f;

    float r[PANEL_COLS];
    for (int c = 0; c < pcols; ++c) {
        r[c] = live ? rpan[(int64_t(prob) * rows + row) * pcols + c] : 0.f;
    }

    for (int j = j0; j < j1; ++j) {
        const int lo = (j - j0) * DIM;          // this block inside the panel
        const int rest = pcols - lo - DIM;      // panel columns after it

        // Stage the block's 8 x 8 tables and its V rows over the rest of the panel.
        __syncthreads();
        const float* tab = vinv + (int64_t(prob) * nblk + j) * DIM * DIM;
        const float* mtab = met + (int64_t(prob) * nblk + j) * NMET;
        for (int t = threadIdx.x; t < DIM * DIM + NMET; t += blockDim.x) {
            coef[t] = t < DIM * DIM ? tab[t] : mtab[t - DIM * DIM];
        }
        for (int t = threadIdx.x; t < DIM * rest; t += blockDim.x) {
            const int a = t / rest, c = t % rest;
            vrow[a][c] = vp[int64_t(j * DIM + a) * cols + (j + 1) * DIM + c];
        }
        __syncthreads();

        // Corrected target: w + r V_BB^-1 (upper triangular), normalized by the row gain.
        float w8[DIM], x[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; ++d) {
            w8[d] = live ? w[wbase + j * DIM + d] : 0.f;
            float t = 0.f;
            #pragma unroll
            for (int a = 0; a <= d; ++a) { t = fmaf(r[lo + a], coef[a * DIM + d], t); }
            x[d] = g > 0.f ? (w8[d] + t) / g : 0.f;
        }
        float md[DIM], mo[NOFF];
        #pragma unroll
        for (int d = 0; d < DIM; ++d) { md[d] = coef[DIM * DIM + d]; }
        #pragma unroll
        for (int t = 0; t < NOFF; ++t) { mo[t] = coef[DIM * DIM + DIM + t]; }

        // Linear term of the off-diagonal part: lin_a = sum_{b != a} (2 M_ab) x_b.
        float lin[DIM];
        #pragma unroll
        for (int a = 0; a < DIM; ++a) { lin[a] = 0.f; }
        {
            int t = 0;
            #pragma unroll
            for (int a = 0; a < DIM - 1; ++a) {
                #pragma unroll
                for (int b = a + 1; b < DIM; ++b) {
                    lin[a] = fmaf(mo[t], x[b], lin[a]);
                    lin[b] = fmaf(mo[t], x[a], lin[b]);
                    ++t;
                }
            }
        }

        float best = INFINITY;
        int index = 0;
        const float4* cb4 = reinterpret_cast<const float4*>(cb);
        for (int base = 0; base < k; base += TILE) {
            const int tile = k - base < TILE ? k - base : TILE;
            __syncthreads();
            // Stage the tile and each codeword's own off-diagonal form Q(c).
            for (int i = threadIdx.x; i < tile; i += blockDim.x) {
                const float4 c0 = cb4[(base + i) * 2], c1 = cb4[(base + i) * 2 + 1];
                words4[2 * i] = c0;
                words4[2 * i + 1] = c1;
                const float c[DIM] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
                float q = 0.f;
                int t = 0;
                #pragma unroll
                for (int a = 0; a < DIM - 1; ++a) {
                    float p = 0.f;
                    #pragma unroll
                    for (int b = a + 1; b < DIM; ++b) { p = fmaf(mo[t++], c[b], p); }
                    q = fmaf(c[a], p, q);
                }
                qtile[i] = q;
            }
            __syncthreads();
            for (int i = 0; i < tile; ++i) {
                const float4 c0 = words4[2 * i], c1 = words4[2 * i + 1];
                const float c[DIM] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
                float distance = 0.f, dot = 0.f;
                // The diagonal first, as axk2_vq_convert.cu's weighted distance.
                #pragma unroll
                for (int d = 0; d < DIM; ++d) {
                    const float delta = x[d] - c[d];
                    distance = fmaf(md[d] * delta, delta, distance);
                }
                // Then Q(c) - c . lin; exact zeros when H is diagonal.
                #pragma unroll
                for (int d = 0; d < DIM; ++d) { dot = fmaf(c[d], lin[d], dot); }
                distance += qtile[i] - dot;
                // Ascending codeword order gives the lowest index on ties.
                if (distance < best) { best = distance; index = base + i; }
            }
        }

        // Residual against the original weights; its feedback into the rest of the panel.
        float d8[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; ++d) {
            d8[d] = __fsub_rn(w8[d], __fmul_rn(g, cb[index * DIM + d]));
        }
        if (live) {
            ids[(int64_t(prob) * rows + row) * nblk + j] = index;
            #pragma unroll
            for (int d = 0; d < DIM; ++d) { res[wbase + j * DIM + d] = d8[d]; }
        }
        for (int c = 0; c < rest; ++c) {
            float acc = r[lo + DIM + c];
            #pragma unroll
            for (int a = 0; a < DIM; ++a) { acc = fmaf(d8[a], vrow[a][c], acc); }
            r[lo + DIM + c] = acc;
        }
    }
}

// Least-squares row gain under H for fixed indices:
//   g = (sum_c h_c w_c c_c + noff) / (sum_c h_c c_c^2 + doff),
// h = diag(H); noff = w H_off c^T and doff = c H_off c^T come from the host. The
// diagonal sums are axk2_vq_convert.cu's refit_gain_kernel, so a diagonal H (zero
// noff, doff) gives its gains bit for bit. Rows whose refit is not a positive
// finite F16 keep their previous gain.
__global__ void hvq_refit_kernel(const float* __restrict__ w, const int* __restrict__ ids,
                                 const float* __restrict__ cb, const float* __restrict__ hdiag,
                                 const double* __restrict__ noff, const double* __restrict__ doff,
                                 __half* gains, int rows, int cols) {
    const int64_t row = int64_t(blockIdx.y) * rows + blockIdx.x;
    const float* colw = hdiag + int64_t(blockIdx.y) * cols;
    const int nvec = cols / DIM;
    __shared__ double nums[256], dens[256];
    double num = 0., den = 0.;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const int id = ids[row * nvec + c / DIM];
        const double cw = cb[id * DIM + (c % DIM)];
        const double h = double(colw[c]);
        num += h * double(w[row * cols + c]) * cw;
        den += h * cw * cw;
    }
    nums[threadIdx.x] = num;
    dens[threadIdx.x] = den;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (threadIdx.x < s) {
            nums[threadIdx.x] += nums[threadIdx.x + s];
            dens[threadIdx.x] += dens[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) { return; }
    const double n = nums[0] + noff[row], d = dens[0] + doff[row];
    if (!(d > 0.)) { return; }
    const double g = n / d;
    const __half h = __double2half(g);
    const float gf = __half2float(h);
    if (g > 0. && isfinite(gf) && gf > 0.f) { gains[row] = h; }
}

static bool valid_k(int k) { return k >= 8 && k <= 8192; }

// w, res [problems, rows, cols]; rpan [problems, rows, (j1 - j0) * 8]; gains [problems, rows];
// v [problems, cols, cols]; vinv [problems, cols / 8, 64]; met [problems, cols / 8, 36];
// cb [k, 8]; ids [problems, rows, cols / 8].
extern "C" int hvq_panel(const float* w, const float* rpan, float* res, const __half* gains,
                         const float* v, const float* vinv, const float* met, const float* cb,
                         int* ids, int problems, int rows, int cols, int k, int j0, int j1,
                         cudaStream_t stream) {
    if (problems <= 0 || rows <= 0 || cols <= 0 || cols % DIM || !valid_k(k)) { return int(cudaErrorInvalidValue); }
    if (j0 < 0 || j1 <= j0 || j1 > cols / DIM || j1 - j0 > MAX_PANEL) { return int(cudaErrorInvalidValue); }
    const dim3 grid(unsigned((rows + THREADS - 1) / THREADS), unsigned(problems));
    hvq_panel_kernel<<<grid, THREADS, 0, stream>>>(w, rpan, res, gains, v, vinv, met, cb, ids, rows, cols, k, j0, j1);
    return int(cudaGetLastError());
}

// w [problems, rows, cols]; ids [problems, rows, cols / 8]; hdiag [problems, cols];
// noff, doff, gains [problems, rows].
extern "C" int hvq_refit(const float* w, const int* ids, const float* cb, const float* hdiag,
                         const double* noff, const double* doff, __half* gains, int problems,
                         int rows, int cols, cudaStream_t stream) {
    if (problems <= 0 || rows <= 0 || cols <= 0 || cols % DIM) { return int(cudaErrorInvalidValue); }
    const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(problems));
    hvq_refit_kernel<<<grid, 256, 0, stream>>>(w, ids, cb, hdiag, noff, doff, gains, rows, cols);
    return int(cudaGetLastError());
}

extern "C" int hvq_max_panel() { return MAX_PANEL; }

