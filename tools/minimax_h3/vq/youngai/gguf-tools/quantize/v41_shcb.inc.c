/* v41_shcb.inc.c — 共享码本(一组一本)的取样与训练驱动(2026-09-21, 113.md 方案 v3), v41_quantize.c include。
 *
 * 【为什么一组一本】09-21 实测: 384 个专家的相对量化误差一致到 1.9%(权重分布形状同构), 每层一本码本与每专家一本
 * 在 L20 上 0.17062 vs 0.17072(还低 0.06%), L39 0.17209 vs 0.17152(亏 0.33%) ⇒ 平均 0.15%, 而 GGUF 里省掉 3.09 GB
 * (每专家一本 × 三个矩阵各存一份 × 15,744 本), 这笔钱换 14 个浅层升 13 位。
 *
 * 【组怎么分】主干层 = 一层一组(384 个专家共用); common 里的 MTP 三塔 = 一塔一组(各 128 个专家) ——
 * 三塔的路由与主干不同, 混在一个池里没有依据, 而分塔的代价只是多两本码本(0.065 MB)。
 *
 * 【池多厚】按"每个码字 ~1000 个训练向量"定取样步长: 一层 384 专家 × 3.93M 向量 = 1.51e9 个,
 * nc4096 要 4.1M(1/369), nc8192 要 8.2M(1/184)。★池薄会亏★(L20 实测 13 位: 1/768 亏 0.96%, 1/384 亏 0.66%),
 * 12 位在 1/384 时反而比每专家一本好 0.26% —— 所以生产不用 --vq-stride 推出来的步长, 按 nc 自己算。
 * 上限 SHCB_POOL_MAX 个向量(主机内存 = ×8×4 B): nc16384 也只要 524 MB。
 *
 * 【确定性】等距取样(无随机源) + 池按专家号顺序拼接 + Lloyd 固定轮数 + double 原子累加 ⇒ 同输入同字节(铁律"产物可复现")。 */
#define SHCB_PER_CODE   1000            /* 每个码字的目标训练向量数 */
#define SHCB_POOL_MAX   16777216LL      /* 池上限(向量): 16 M × 32 B = 512 MB 主机内存 */
#define SHCB_MAX_GRP    8               /* 组数上限: 主干 1 组 / common 三塔 3 组 */

static uint16_t *g_shcb[SHCB_MAX_GRP];  /* 每组一本码本(f16 位型), NULL = 该组走每专家一本 */

static void shcb_free(void) {
    for (int i = 0; i < SHCB_MAX_GRP; i++) { free(g_shcb[i]); g_shcb[i] = NULL; }
}

/* 一组的池取样 + 训练。jobs 里 grp == g 且 kind == J_VQ 的专家参与。返回 0 成功。 */
static int shcb_build_group(v41_st *S, const cfg_t *C, const job_t *jobs, int nj, int g, const char *tag, double t0) {
    long long tot = 0;                  /* 本组全部向量数 */
    int nexp_g = 0;
    for (int j = 0; j < nj; j++) {
        if (jobs[j].kind != J_VQ || jobs[j].grp != g) continue;
        for (int m = 0; m < 3; m++) tot += (long long)jobs[j].ew[m]->shape[0] * jobs[j].ew[m]->shape[1] * 2 / C->dim;
        nexp_g++;
    }
    if (!nexp_g) return 0;
    long long want = (long long)C->nc * SHCB_PER_CODE;   /* 池目标向量数(按全局 nc; 逐层表下 13 位层各自更厚见下) */
    for (int j = 0; j < nj; j++) if (jobs[j].kind == J_VQ && jobs[j].grp == g && jobs[j].nc > C->nc)
        want = (long long)jobs[j].nc * SHCB_PER_CODE;    /* 组内有更大的码本 ⇒ 按最大的定池厚度(同一本码本要喂饱它) */
    if (want > SHCB_POOL_MAX) want = SHCB_POOL_MAX;
    int ps = (int)(tot / want); if (ps < 1) ps = 1;
    /* 组内的 nc 必须一致(同一本码本) —— 逐层表把某层内部拆成两种 nc 就是配方错, 这里硬停 */
    int nc_g = -1;
    for (int j = 0; j < nj; j++) if (jobs[j].kind == J_VQ && jobs[j].grp == g) {
        if (nc_g < 0) nc_g = jobs[j].nc;
        else if (nc_g != jobs[j].nc) { fprintf(stderr, "★共享码本组 %s 内 nc 不一致(%d vs %d): 位宽表必须整组同档★\n", tag, nc_g, jobs[j].nc); return -1; }
    }
    long long cap = 0;
    for (int j = 0; j < nj; j++) if (jobs[j].kind == J_VQ && jobs[j].grp == g) {
        long long nv = 0;
        for (int m = 0; m < 3; m++) nv += (long long)jobs[j].ew[m]->shape[0] * jobs[j].ew[m]->shape[1] * 2 / C->dim;
        cap += nv / ps + 1;
    }
    float *pool = (float *)malloc(sizeof(float) * (size_t)cap * C->dim);
    if (!pool) { fprintf(stderr, "★共享码本池 %.2f GB 要不到★\n", sizeof(float) * (double)cap * C->dim / 1e9); return -1; }
    long long n = 0;
    for (int j = 0; j < nj; j++) {
        if (jobs[j].kind != J_VQ || jobs[j].grp != g) continue;
        const job_t *J = &jobs[j];
        const uint8_t *w[3], *s[3]; int rows[3], cols[3];
        for (int m = 0; m < 3; m++) {
            w[m] = v41_st_data(S, J->ew[m]); s[m] = v41_st_data(S, J->es[m]);
            if (!w[m] || !s[m]) { free(pool); return -1; }
            rows[m] = (int)J->ew[m]->shape[0]; cols[m] = (int)J->ew[m]->shape[1] * 2;
        }
        if (v41_vq_pool_sample_from_fp4(w, s, rows, cols, 3, C->dim, ps, pool, cap, &n)) { free(pool); return -1; }
    }
    g_shcb[g] = (uint16_t *)malloc(sizeof(uint16_t) * nc_g * C->dim);
    if (!g_shcb[g] || v41_vq_train_codebook(pool, n, C->dim, nc_g, C->iters, C->lam, C->cb_fp8, g_shcb[g])) { free(pool); return -1; }
    free(pool);
    printf("  [共享码本] %s: %d 专家池 %lld 向量(1/%d 取样, %.0f/码字) → 一本 nc%d%s, %.0fs\n",
           tag, nexp_g, n, ps, (double)n / nc_g, nc_g, C->cb_fp8 ? " E4M3" : "", now_s() - t0);
    fflush(stdout);
    return 0;
}
