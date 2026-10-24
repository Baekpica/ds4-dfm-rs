/* v41_quantize_cli.inc.c — v41_quantize 的命令行解析与位宽表装填(2026-09-21 从 v41_quantize.c 拆出, 500 行守卫)。
 *
 * 只做两件事, 都不碰数值: ①把 argv 变成 cfg_t 并校验(不合法就硬停 —— 带着错参数跑一小时是最贵的 bug)
 * ②把"整层段 --vq-nc-layers"与"逐专家表 --vq-nc-table"合成一张 [nlayers][nexp] 的 nc 表(段先铺, 表后覆盖)。
 * 拆文件不改语义: 原来这两段就在 main 里, 一个字没动过判据。 */

static void qcli_usage(void) {
    fprintf(stderr, "用法: v41_quantize <hf-dir> <out-dir> [--vq-dim 8] [--vq-nc 4096] [--vq-iters 8] [--vq-stride 8] [--skel fp4|q4k] [--mtp-vq] [--layers a:b] [--no-common] [--force]\n"
                    "                    [--calib <取料目录>] [--calib-ab] [--calib-ef β1,β2]   (金融域校准: v41_amp_run --dump-calib 的产物; -ab = 平权对照审计, 冒烟用; -ef = 级 2 误差反馈, β 按 val 择优)\n"
                    "                    [--vq-nc-table FILE] [--stats-out FILE]   (逐专家 nc 表 / 逐专家误差落盘, 动态位宽)\n"
                    "                    [--vq-nc-layers a:b=NC]   整层一档位宽(可重复; 方案 v3 的 14 层 13 位 = --vq-nc-layers 0:14=8192)\n"
                    "                    [--vq-ecvq λ] [--vq-shared-cb] [--vq-cb-fp8]   (ECVQ 码长惩罚 / 一组一本共享码本 / 码本舍 E4M3, 见 v41_vq_rate.h)\n");
}

/* 返回 0 成功, 否则 main 的退出码 */
static int qcli_parse(int argc, char **argv, cfg_t *C) {
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--vq-dim") && i + 1 < argc) C->dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vq-nc") && i + 1 < argc) C->nc = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vq-iters") && i + 1 < argc) C->iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vq-stride") && i + 1 < argc) C->stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) { if (sscanf(argv[++i], "%d:%d", &C->l0, &C->l1) != 2) { fprintf(stderr, "★--layers 要 a:b★\n"); return 2; } }
        else if (!strcmp(argv[i], "--skel") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "q4k")) C->skel_q4k = 1;
            else if (!strcmp(v, "fp4")) C->skel_q4k = 0;
            else { fprintf(stderr, "★--skel 只认 fp4/q4k, 收到 %s★\n", v); return 2; }
        }
        else if (!strcmp(argv[i], "--mtp-vq")) C->mtp_vq = 1;
        else if (!strcmp(argv[i], "--no-common")) C->common = 0;
        else if (!strcmp(argv[i], "--force")) C->force = 1;
        else if (!strcmp(argv[i], "--calib") && i + 1 < argc) { char *p = argv[++i]; size_t n = strlen(p); while (n > 1 && p[n - 1] == '/') p[--n] = 0; C->calib = p; }
        else if (!strcmp(argv[i], "--calib-ab")) C->calib_ab = 1;
        else if (!strcmp(argv[i], "--calib-ef") && i + 1 < argc) {   /* 级 2: β 候选表, 如 0.1,1 */
            char *p = argv[++i]; C->ef_nb = 0;
            for (char *t = strtok(p, ","); t && C->ef_nb < 4; t = strtok(NULL, ",")) C->ef_betas[C->ef_nb++] = (float)atof(t);
            if (C->ef_nb < 1) { fprintf(stderr, "★--calib-ef 要 β 表(如 0.1,1)★\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--calib-alpha") && i + 1 < argc) C->alpha = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--vq-nc-table") && i + 1 < argc) g_nct_path = argv[++i];
        else if (!strcmp(argv[i], "--vq-nc-layers") && i + 1 < argc) {
            if (g_nncl >= NCL_MAX) { fprintf(stderr, "★--vq-nc-layers 最多 %d 段★\n", NCL_MAX); return 2; }
            int a, b, nc;
            if (sscanf(argv[++i], "%d:%d=%d", &a, &b, &nc) != 3 || a < 0 || b <= a || nc < 2 || nc > 65536 || (nc & (nc - 1))) {
                fprintf(stderr, "★--vq-nc-layers 要 a:b=NC 且 NC 是 2 的幂, 收到 %s★\n", argv[i]); return 2;
            }
            g_ncl[g_nncl].a = a; g_ncl[g_nncl].b = b; g_ncl[g_nncl].nc = nc; g_nncl++;
        }
        else if (!strcmp(argv[i], "--stats-out") && i + 1 < argc) {
            if (!(g_stats = fopen(argv[++i], "w"))) { fprintf(stderr, "★写不了 %s★\n", argv[i]); return 2; }
            fprintf(g_stats, "# L e nc sse energy(Σw²) H(bit/索引) used mass_half w16 w32\n");
        }
        else if (!strcmp(argv[i], "--vq-ecvq") && i + 1 < argc) C->lam = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--vq-shared-cb")) C->shared_cb = 1;
        else if (!strcmp(argv[i], "--vq-cb-fp8")) C->cb_fp8 = 1;
        else { fprintf(stderr, "★不认识的参数 %s★\n", argv[i]); return 2; }
    }
    if ((C->calib_ab || C->ef_nb || C->alpha != 1.f) && !C->calib) { fprintf(stderr, "★--calib-ab/--calib-ef/--calib-alpha 要配 --calib★\n"); return 2; }
    if (C->alpha < 0.f || C->alpha > 1.f) { fprintf(stderr, "★--calib-alpha %g 要在 [0,1]★\n", C->alpha); return 2; }
    if (C->calib) {   /* 校准料的来源账必须在(取料程序写的 calib.txt), 没有就是半份料 */
        char p[4300]; snprintf(p, sizeof p, "%s/calib.txt", C->calib); struct stat sb;
        if (stat(p, &sb)) { fprintf(stderr, "★校准目录 %s 没有 calib.txt(取料没收工?)★\n", C->calib); return 2; }
    }
    return 0;
}

/* 位宽表: 段先铺全层, 表再逐项覆盖; 越界/非 2 的幂一律拒。返回 0 成功。 */
static int qcli_nctab(const cfg_t *C, int nlayers, int nexp) {
    if (!g_nct_path && !g_nncl) return 0;
    g_nct = (int *)malloc(sizeof(int) * (size_t)nlayers * nexp);
    if (!g_nct) { fprintf(stderr, "★位宽表要不到内存★\n"); return 2; }
    for (size_t i = 0; i < (size_t)nlayers * nexp; i++) g_nct[i] = C->nc;
    for (int k = 0; k < g_nncl; k++) {
        if (g_ncl[k].b > nlayers) { fprintf(stderr, "★--vq-nc-layers %d:%d 超过层数 %d★\n", g_ncl[k].a, g_ncl[k].b, nlayers); return 2; }
        for (int L = g_ncl[k].a; L < g_ncl[k].b; L++) for (int e = 0; e < nexp; e++) g_nct[(size_t)L * nexp + e] = g_ncl[k].nc;
        printf("[位宽段] L[%d,%d) → nc%d(%d bit)\n", g_ncl[k].a, g_ncl[k].b, g_ncl[k].nc, bits_of(g_ncl[k].nc));
    }
    if (!g_nct_path) return 0;
    FILE *f = fopen(g_nct_path, "r"); if (!f) { fprintf(stderr, "★打不开位宽表 %s★\n", g_nct_path); return 2; }
    char line[256]; int n = 0, L, e, nc;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%d %d %d", &L, &e, &nc) != 3 || L < 0 || L >= nlayers || e < 0 || e >= nexp || nc < 2 || nc > 65536 || (nc & (nc - 1))) {
            fprintf(stderr, "★位宽表第 %d 项坏: %s★\n", n + 1, line); fclose(f); return 2;
        }
        g_nct[(size_t)L * nexp + e] = nc; n++;
    }
    fclose(f);
    printf("[位宽表] %s: %d 项(其余专家用全局 nc%d)\n", g_nct_path, n, C->nc);
    return 0;
}
