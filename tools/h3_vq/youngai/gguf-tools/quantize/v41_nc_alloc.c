/* v41_nc_alloc.c — 逐专家位宽分配表, ★纯权重零语料★(2026-09-21, 用户令"不要用语料跑量化, 直接根据权重跑动态量化")。
 *
 * 【干什么】扫 HF 原始权重, 算每个专家三个矩阵的能量 E_e = Σw²(FP4 nibble × ue8m0 scale, 精确逐元素),
 * 按率失真反注水给每个专家分码本大小 nc_e = 2^b_e, 输出 "L e nc" 表给 v41_quantize --vq-nc-table。
 * 层与层之间也一起分(--global): 层体积随本层权重能量走 ⇒ 每层体积 + 每专家体积都是权重定的。
 *
 * 【为什么是 Σw²】VQ 解码 ŵ = cb[idx]·gain, 相对失真(1−cos²)在全部专家上几乎一样(实测 cos 0.896~0.901),
 *   所以专家 e 的【绝对】重建误差 ≈ E_e·(1−cos²)。输出误差看的是绝对值不是相对值 ⇒ 能量大的专家该多给位。
 *   ★零语料★: 不看路由、不看激活、不看任何文本 —— 与"全域平权"原则一致(按语料分配 = 域偏置, 08-08 判过)。
 * 【率失真】8 维 VQ 每加 1 位索引, 相对失真 ×2^(−1/4)(高码率理论; 与 09-20 实测 12→11 位残差 17.1% → 20.3% 对上)。
 *   目标 min Σ E_e·2^(−b_e/4), 约束 Σ bytes(b_e) ≤ 预算。bytes(b) = 三矩阵索引流(逐行字节对齐, 与量化器
 *   plan_vq 同一条公式) + 码本 2^b×dim×2×份数。目标凸、预算线性 ⇒ 逐位贪心(每步选 ΔD/Δbytes 最大的)就是最优解。
 * 【预算口径】--cb-copies 3(默认): 码本按 GGUF 落地口径算三份(转换器每矩阵各带一份) —— 体积口径 = 落地文件全字节。
 * 【出错会怎样】缺张量/dtype 不对直接停; 表里 nc 不是 2 的幂量化器会拒收; 用量超预算是本程序的 bug, 自检停。
 * 【耗时】要把 272 GB 专家权重整读一遍(--threads 并行, --row-stride N 可按行抽样换速度), 收工打实测秒数。
 * 用法: v41_nc_alloc <hf-dir> <层a:b> <输出表> [--nexp 384] [--dim 8] [--nc-ref 4096] [--bmin 9] [--bmax 14]
 *                    [--cb-copies 3] [--global] [--threads 8] [--row-stride 1]
 * 编译: make -C gguf-tools v41_nc_alloc (纯 C, 无 CUDA) */
#include "../../src/common/ds4_st41.h"
#include "../../src/common/ds4_fp8.h"
#include <math.h>
#include <pthread.h>
#include <time.h>

static int bits_of(int nc) { int b = 1; while ((1 << b) < nc) b++; return b; }
static int g_dim = 8, g_copies = 3, g_stride = 1;
static double g_tab[256];        /* 一个字节(两个 nibble)的平方和 */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* 一个专家在 b 位下的字节数(行增益两边一样多, 不计): 三矩阵索引流 + 码本×份数 */
static long long expert_bytes(int b, int D, int mid) {
    long long r13 = (long long)mid * (((long long)(D / g_dim) * b + 7) / 8);
    long long r2 = (long long)D * (((long long)(mid / g_dim) * b + 7) / 8);
    return 2 * r13 + r2 + ((long long)1 << b) * g_dim * 2 * g_copies;
}
static double dist(int b) { return pow(2.0, -b / 4.0); }
/* 贪心逐位加: 全员从 bmin 起, 每步给 ΔD/Δbytes 最大的候选加一位, 加不进预算就找次优, 都加不进就停。返回用量 */
static long long greedy(const double *I, int *b, int n, int bmin, int bmax, long long budget, int D, int mid) {
    long long used = 0; for (int i = 0; i < n; i++) { b[i] = bmin; used += expert_bytes(bmin, D, mid); }
    for (;;) {
        int best = -1; double bestg = 0.0;
        for (int i = 0; i < n; i++) {
            if (b[i] >= bmax) continue;
            const long long db = expert_bytes(b[i] + 1, D, mid) - expert_bytes(b[i], D, mid);
            if (used + db > budget) continue;
            const double g = I[i] * (dist(b[i]) - dist(b[i] + 1)) / (double)db;
            if (g > bestg) { bestg = g; best = i; }
        }
        if (best < 0) return used;
        used += expert_bytes(b[best] + 1, D, mid) - expert_bytes(b[best], D, mid);
        b[best]++;
    }
}

/* 一层的取数计划(主线程解出指针, 工作线程只读内存 —— v41_st_data 会 mmap, 不是线程安全的) */
typedef struct { const uint8_t *w[3], *s[3]; int rows[3], cols[3]; } exp_plan;
typedef struct { const exp_plan *P; double *I, *row; int nexp, t, nt, rpe; double mat[3]; } job_arg;
static void *energy_worker(void *ud) {
    job_arg *A = (job_arg *)ud;
    for (int e = A->t; e < A->nexp; e += A->nt) {
        const exp_plan *p = &A->P[e];
        double tot = 0; int ri = 0;
        for (int m = 0; m < 3; m++) {
            const int rows = p->rows[m], cols = p->cols[m], nb = cols / 32, wr = cols / 2;
            for (int r = 0; r < rows; r += g_stride) {
                const uint8_t *W = p->w[m] + (size_t)r * wr, *S = p->s[m] + (size_t)r * nb;
                double re = 0;
                for (int j = 0; j < nb; j++) {
                    double acc = 0; const uint8_t *b16 = W + (size_t)j * 16;
                    for (int k = 0; k < 16; k++) acc += g_tab[b16[k]];
                    const double sc = ds4_e8m0_to_f32(S[j]);
                    re += sc * sc * acc;
                }
                tot += re; A->mat[m] += re;   /* 逐矩阵账: w1/w3/w2 之间有没有跨度(社区经验 down 该多给位) */
                /* 逐行能量(动态位宽的下一个粒度: 行。索引流本来就逐行字节对齐, 行级位宽不改 blob 布局) */
                if (A->row && ri < A->rpe) A->row[(size_t)e * A->rpe + ri++] = re;
            }
        }
        A->I[e] = tot * g_stride;   /* 抽样时按步长放大回全行口径(只作相对权重, 常数因子不影响分配) */
    }
    return NULL;
}
static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

/* ★行级动态位宽的头寸★(只算账不落产物, 2026-09-21): 索引流本来就逐行字节对齐 ⇒ 行级位宽不改 blob 布局,
 * 代价是码本 —— 一个专家一本码本要按最大位宽养(2^bmax), 低位行用它的前缀(嵌套码本)。
 * 拉格朗日水填: 给定 λ, 每行独立选最省的 b; 二分 λ 卡住预算。报净失真比(含码本那笔多花的钱)。 */
static long long row_bytes(int b, int cols) { return ((long long)(cols / g_dim) * b + 7) / 8; }
static double row_alloc_report(const double *row, const int *cols_of_row, int nrow, int bmin, int bmax, long long budget, double Duni) {
    double lo = 1e-12, hi = 1e12;
    for (int it = 0; it < 80; it++) {
        const double lam = sqrt(lo * hi);
        long long used = 0;
        for (int r = 0; r < nrow; r++) {
            double best = 1e300; int bb = bmin;
            for (int b = bmin; b <= bmax; b++) { const double c = row[r] * dist(b) + lam * (double)row_bytes(b, cols_of_row[r]); if (c < best) { best = c; bb = b; } }
            used += row_bytes(bb, cols_of_row[r]);
        }
        if (used > budget) lo = lam; else hi = lam;
    }
    const double lam = hi; double D = 0; long long used = 0; int hist[17] = {0};
    for (int r = 0; r < nrow; r++) {
        double best = 1e300; int bb = bmin;
        for (int b = bmin; b <= bmax; b++) { const double c = row[r] * dist(b) + lam * (double)row_bytes(b, cols_of_row[r]); if (c < best) { best = c; bb = b; } }
        D += row[r] * dist(bb); used += row_bytes(bb, cols_of_row[r]); hist[bb]++;
    }
    printf("  行级水填: 用 %lld B / 预算 %lld B; 位宽", used, budget);
    for (int k = bmin; k <= bmax; k++) if (hist[k]) printf(" %d:%d", k, hist[k]);
    printf("; ★失真 动态/均匀 = %.5f(%+.2f%%) = %+.3f 位等效★\n", D / Duni, 100.0 * (D / Duni - 1.0), log(D / Duni) / log(dist(1) / dist(0)));
    return D / Duni;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "用法: v41_nc_alloc <hf-dir> <层a:b> <输出表> [--nexp 384] [--dim 8] [--nc-ref 4096] [--bmin 9] [--bmax 14] [--cb-copies 3] [--global] [--threads 8] [--row-stride 1]\n"); return 2; }
    const char *hf = argv[1], *outp = argv[3];
    int l0, l1; if (sscanf(argv[2], "%d:%d", &l0, &l1) != 2 || l0 < 0 || l1 <= l0) { fprintf(stderr, "★层要 a:b★\n"); return 2; }
    int nexp = 384, ncref = 4096, bmin = 9, bmax = 14, glob = 0, nt = 8, rowstat = 0;
    const char *statp = NULL; double *rowbuf = NULL;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--nexp") && i + 1 < argc) nexp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim") && i + 1 < argc) g_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nc-ref") && i + 1 < argc) ncref = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bmin") && i + 1 < argc) bmin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bmax") && i + 1 < argc) bmax = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cb-copies") && i + 1 < argc) g_copies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--row-stride") && i + 1 < argc) g_stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--global")) glob = 1;
        else if (!strcmp(argv[i], "--row-stat")) rowstat = 1;
        else if (!strcmp(argv[i], "--stats") && i + 1 < argc) statp = argv[++i];   /* 量化器 --stats-out 的逐专家真实误差当重要性 */
        else { fprintf(stderr, "★不认识的参数 %s★\n", argv[i]); return 2; }
    }
    const int bref = bits_of(ncref), nl = l1 - l0;
    if (bmin < 1 || bmax > 16 || bmin > bref || bref > bmax) { fprintf(stderr, "★要 1 ≤ bmin ≤ log2(nc_ref) ≤ bmax ≤ 16★\n"); return 2; }
    if (nt < 1) nt = 1; if (g_stride < 1) g_stride = 1;
    for (int i = 0; i < 256; i++) {
        const float a = ds4_fp4_nibble_to_f32((uint8_t)(i & 15)), b = ds4_fp4_nibble_to_f32((uint8_t)(i >> 4));
        g_tab[i] = (double)a * a + (double)b * b;
    }
    v41_st S; double t0 = now_s();
    if (v41_st_open(&S, hf)) return 1;
    const size_t N = (size_t)nl * nexp;
    double *I = (double *)calloc(N, sizeof(double));
    int *b = (int *)calloc(N, sizeof(int)), *ord = (int *)calloc((size_t)nexp, sizeof(int));
    exp_plan *P = (exp_plan *)calloc((size_t)nexp, sizeof(exp_plan));
    pthread_t *th = (pthread_t *)calloc((size_t)nt, sizeof(pthread_t));
    job_arg *A = (job_arg *)calloc((size_t)nt, sizeof(job_arg));
    static const char *mat[3] = {"w1", "w3", "w2"};
    int D = 0, mid = 0;
    for (int L = l0; L < l1; L++) {
        const double tl = now_s();
        for (int e = 0; e < nexp; e++) {
            for (int m = 0; m < 3; m++) {
                char n[192]; snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.weight", L, e, mat[m]);
                const v41_st_ent *W = v41_st_find(&S, n);
                snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.scale", L, e, mat[m]);
                const v41_st_ent *Sc = v41_st_find(&S, n);
                if (!W || !Sc || strcmp(W->dtype, "I8")) { fprintf(stderr, "★L%02d e%d %s: 缺张量或非 FP4★\n", L, e, mat[m]); return 1; }
                P[e].rows[m] = (int)W->shape[0]; P[e].cols[m] = (int)W->shape[1] * 2;
                if (P[e].cols[m] % 32 || Sc->shape[1] != P[e].cols[m] / 32) { fprintf(stderr, "★L%02d e%d %s: scale 块形状不是 1×32★\n", L, e, mat[m]); return 1; }
                P[e].w[m] = v41_st_data(&S, W); P[e].s[m] = v41_st_data(&S, Sc);
                if (!P[e].w[m] || !P[e].s[m]) return 1;
            }
        }
        if (!D) { D = P[0].cols[0]; mid = P[0].rows[0]; printf("[alloc] 形状 D=%d mid=%d; 每专家 %.1f MB; 线程 %d 行步长 %d\n", D, mid, (2.0 * mid * (D / 2) + (double)D * (mid / 2)) / 1e6, nt, g_stride); }
        double *IL = I + (size_t)(L - l0) * nexp;
        const int rpe = (2 * mid + D) / g_stride;   /* 一个专家的行数(w1+w3+w2, 按步长) */
        if (rowstat && !rowbuf) rowbuf = (double *)malloc(sizeof(double) * (size_t)nexp * rpe);
        for (int t = 0; t < nt; t++) { A[t].P = P; A[t].I = IL; A[t].row = rowbuf; A[t].rpe = rpe; A[t].nexp = nexp; A[t].t = t; A[t].nt = nt; A[t].mat[0] = A[t].mat[1] = A[t].mat[2] = 0; pthread_create(&th[t], NULL, energy_worker, &A[t]); }
        for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
        { double mt[3] = {0, 0, 0}; for (int t = 0; t < nt; t++) for (int m = 0; m < 3; m++) mt[m] += A[t].mat[m];
          printf("[矩阵] L%02d 能量占比 w1 %.1f%% w3 %.1f%% w2 %.1f%% (元素数相同 ⇒ 占比差就是该不该分不同位宽)\n",
                 L, 100 * mt[0] / (mt[0] + mt[1] + mt[2]), 100 * mt[1] / (mt[0] + mt[1] + mt[2]), 100 * mt[2] / (mt[0] + mt[1] + mt[2])); }
        double s = 0, mx = 0, mn = 1e300; for (int e = 0; e < nexp; e++) { s += IL[e]; if (IL[e] > mx) mx = IL[e]; if (IL[e] < mn) mn = IL[e]; }
        printf("[能量] L%02d %5.0fs  Σw² 均值 %.4g  最大/最小 %.3f  (读 %.1f GB)\n", L, now_s() - tl, s / nexp, mn > 0 ? mx / mn : 0.0,
               (double)nexp * (2.0 * mid * (D / 2) + (double)D * (mid / 2)) / 1e9 / g_stride);
        if (rowbuf) {   /* 行级跨度: 动态位宽的下一个候选粒度值不值得, 看这一行 */
            const size_t nr = (size_t)nexp * rpe;
            double *c = (double *)malloc(sizeof(double) * nr); memcpy(c, rowbuf, sizeof(double) * nr);
            qsort(c, nr, sizeof(double), cmpd);
            printf("[行能量] L%02d %zu 行: p1 %.4g p10 %.4g 中位 %.4g p90 %.4g p99 %.4g | p90/p10 %.2f p99/p1 %.2f 最大/最小 %.1f\n",
                   L, nr, c[nr / 100], c[nr / 10], c[nr / 2], c[nr * 9 / 10], c[nr * 99 / 100],
                   c[nr / 10] > 0 ? c[nr * 9 / 10] / c[nr / 10] : 0.0, c[nr / 100] > 0 ? c[nr * 99 / 100] / c[nr / 100] : 0.0, c[0] > 0 ? c[nr - 1] / c[0] : 0.0);
            free(c);
            /* 行级头寸: 预算 = 均匀 bref 的索引流字节 − 码本按 bmax 养多出来的那笔(嵌套码本, 一个专家一本) */
            int *cr = (int *)malloc(sizeof(int) * nr); long long ib = 0;
            for (int e = 0; e < nexp; e++) for (int i = 0; i < rpe; i++) {
                const int m = i < mid / g_stride ? 0 : (i < 2 * (mid / g_stride) ? 1 : 2);
                cr[(size_t)e * rpe + i] = P[e].cols[m]; ib += row_bytes(bref, P[e].cols[m]);
            }
            const long long cbx = (long long)nexp * ((1LL << bmax) - (1LL << bref)) * g_dim * 2 * g_copies;
            double Duni = 0; for (size_t i = 0; i < nr; i++) Duni += rowbuf[i] * dist(bref);
            row_alloc_report(rowbuf, cr, (int)nr, bmin, bmax, ib - cbx, Duni);
            free(cr);
        }
        fflush(stdout);
        v41_st_release_idle(&S);
    }
    v41_st_close(&S);
    if (statp) {   /* 有实测逐专家误差就用它当重要性(能量 × 好不好编 都在里面), 没覆盖到的层保持 Σw² */
        FILE *f = fopen(statp, "r"); if (!f) { fprintf(stderr, "★打不开 %s★\n", statp); return 1; }
        char line[256]; int L, e, nc, n = 0; double sse, en;
        while (fgets(line, sizeof line, f))
            if (line[0] != '#' && sscanf(line, "%d %d %d %lf %lf", &L, &e, &nc, &sse, &en) == 5 && L >= l0 && L < l1 && e >= 0 && e < nexp)
                { I[(size_t)(L - l0) * nexp + e] = sse; n++; }
        fclose(f);
        printf("[alloc] 重要性换成实测误差: %s 覆盖 %d 个专家\n", statp, n);
    }
    const long long layer_budget = (long long)nexp * expert_bytes(bref, D, mid);
    long long tot_used = 0;
    if (glob) tot_used = greedy(I, b, (int)N, bmin, bmax, layer_budget * nl, D, mid);
    else for (int l = 0; l < nl; l++) tot_used += greedy(I + (size_t)l * nexp, b + (size_t)l * nexp, nexp, bmin, bmax, layer_budget, D, mid);
    if (tot_used > layer_budget * nl) { fprintf(stderr, "★用量 %lld > 预算 %lld, 分配器 bug★\n", tot_used, layer_budget * nl); return 1; }
    FILE *fo = fopen(outp, "w"); if (!fo) { fprintf(stderr, "★写不了 %s★\n", outp); return 1; }
    fprintf(fo, "# v41_nc_alloc(纯权重 Σw², 零语料): %s 层 %d:%d 等体积对 nc%d(码本×%d), 位宽 [%d,%d], %s预算\n", hf, l0, l1, ncref, g_copies, bmin, bmax, glob ? "全局" : "逐层");
    for (int l = 0; l < nl; l++) {   /* 账: 本层用量对均匀预算 / 位宽直方图 / 能量集中度(top 10% 占多少) */
        const double *IL = I + (size_t)l * nexp; const int *bL = b + (size_t)l * nexp;
        double Itot = 0, Ihi = 0; int hist[17] = {0}; long long used = 0;
        for (int e = 0; e < nexp; e++) { Itot += IL[e]; hist[bL[e]]++; if (bL[e] > bref) Ihi += IL[e]; ord[e] = e; used += expert_bytes(bL[e], D, mid); }
        for (int i = 1; i < nexp; i++) { int k = ord[i], j = i - 1; while (j >= 0 && IL[ord[j]] < IL[k]) { ord[j + 1] = ord[j]; j--; } ord[j + 1] = k; }
        double Itop = 0; const int ntop = nexp / 10 > 0 ? nexp / 10 : 1; for (int i = 0; i < ntop; i++) Itop += IL[ord[i]];
        printf("[alloc] L%02d 用 %.4f GB(均匀 %.4f, %+.2f%%); 位宽", l0 + l, used / 1e9, layer_budget / 1e9, 100.0 * (used - layer_budget) / (double)layer_budget);
        for (int k = bmin; k <= bmax; k++) if (hist[k]) printf(" %d:%d", k, hist[k]);
        printf("; 能量 top10%%(%d 个)占 %.1f%%, >%d 位的专家占 %.1f%%; 最肥 e%d %.1f%%\n",
               ntop, 100.0 * Itop / (Itot > 0 ? Itot : 1), bref, 100.0 * Ihi / (Itot > 0 ? Itot : 1), ord[0], 100.0 * IL[ord[0]] / (Itot > 0 ? Itot : 1));
        for (int e = 0; e < nexp; e++) fprintf(fo, "%d %d %d\n", l0 + l, e, 1 << bL[e]);
    }
    fclose(fo);
    /* ★动态划分到底值多少★: 同预算下, 分出来的位宽对全均匀 bref 的总权重失真比。1.000 = 动态等于均匀, 没有肉 */
    double Duni = 0, Ddyn = 0;
    for (size_t i = 0; i < N; i++) { Duni += I[i] * dist(bref); Ddyn += I[i] * dist(b[i]); }
    printf("[alloc] ★同体积下总权重失真 动态/均匀 = %.5f(%+.3f%%)★\n", Duni > 0 ? Ddyn / Duni : 1.0, Duni > 0 ? 100.0 * (Ddyn / Duni - 1.0) : 0.0);
    printf("[alloc] 合计 预算 %.6f GB 用 %.6f GB (差 %.3f%%), 全程 %.0fs → %s\n", layer_budget * nl / 1e9, tot_used / 1e9,
           100.0 * (tot_used - layer_budget * nl) / (double)(layer_budget * nl), now_s() - t0, outp);
    free(I); free(b); free(ord); free(P); free(th); free(A);
    return 0;
}
