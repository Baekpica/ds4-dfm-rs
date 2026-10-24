/* rplan_solve.c — 体积预算 → 逐层档位配置(2026-08-27, 用户令"按 86G 生成动态配置 json")。
 *
 * 【这个程序解决什么】平权量化给 43 层每层同样的位宽。但实测每层的量化误差差 5 倍
 * (vq86h_noz/plan.txt: L0 relh=0.1055, L28 relh=0.5315)。把易层省下的字节挪给难层,
 * 总体积不变而总误差更小 —— 前提是"加位宽真能减误差", 这由档梯 cos 表(实测)决定。
 *
 * 【为什么重写 rplan_solve_v4.py】① 全仓零 Python 裁决; ② 原版档梯只有 vq4x512 往【下】
 * 的档(cos 0.740/0.646/0.573/0.470), 所以在 86G 预算下它的最优解必然是"每层都拉满到
 * vq4x512"= 平权 —— 分配器根本没有往上的档可选。这就是当年 86G 上平权赢的机制原因,
 * 不是平权比动态好, 是动态在那个档梯下无处可去。本版档梯双向延伸。
 *
 * 【与 v4 的另一处不同: 不做动态专家】v4 把每层分成 hot(固定 vq4x512)/cold(便宜档)两档。
 * 本版一律 hot=0, 全 256 专家在层内同档 —— 因为"丢专家/降级专家"已被实测两次否决
 * (纯剪枝 q2 输出崩成乱码; 冷热分档在案裁决为设计缺陷)。所以分配变量只剩【每层档位】,
 * 即动态层 + 层内平权。rplan 的 hot=0 语义正好是"全专家走 dim/nc 档", 量化器零改动。
 *
 * 【难度权重为什么取平方差】d[L] = relh²[L] − relh²[L−1]。relh 是【累积】相对误差,
 * 直接用它会让 L0 虚高(L0 没有上游, 它的 relh 是从零开始的全部误差, 别的层是增量)。
 * 平方 = 误差能量, 才是可加的同一把尺。v4 原版踩过这个坑并修正, 此处沿用其结论。
 *
 * 【贪心为什么正确】每步挑"每字节换回的加权精度增益"最大的一层升档, 直到预算用完。
 * 档梯单调(越贵越准)时这是背包问题的标准贪心近似; 档梯非单调会制造负收益台阶把所有层
 * 卡死在最稀档(v4 原版实测事故), 所以启动时强制校验单调性。
 *
 * 用法: rplan_solve --held <plan.txt> --ladder <ladder.txt> --budget-gib N
 *                   [--out-json f.json] [--out-rplan f.txt] [--floor-nc N]
 *   plan.txt   量化器产的逐层账: "L=<n> cand=... relh=<v> gate=..."(取 L 与 relh)
 *   ladder.txt 每行 "dim nc cos  # 出处", 按字节从小到大; cos = 该档实测重建余弦
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vq_qc_bytes.h"

#define MAXL   64
#define MAXT   16
#define NEXP   256
#define MOEI   2048
#define DMODEL 4096

typedef struct { int dim, nc; double cos_; size_t bytes; char note[128]; } tier_t;

static size_t expert_bytes(const tier_t *t) {
    return 2 * vq_payload_bytes(MOEI, DMODEL, t->dim, t->nc) + vq_payload_bytes(DMODEL, MOEI, t->dim, t->nc);
}
#define VQ_HDR (16 + (size_t)NEXP * 3 * 8)

static size_t layer_bytes(const tier_t *t) {
    /* 与量化器 vq_total_bytes(hot=0 分支)逐字对应: 表头 + 256 专家 ×(w1+w3+w2)。
     * w2 与 w1/w3 同档(rplan 的 w2dim/w2nc 我们填成一样) —— 口径差一点, 体积账就对不上, 跑完
     * 5 小时才发现超预算。 */
    size_t hdr = 16 + (size_t)NEXP * 3 * 8;
    size_t w13 = vq_payload_bytes(MOEI, DMODEL, t->dim, t->nc);
    size_t w2  = vq_payload_bytes(DMODEL, MOEI, t->dim, t->nc);
    return hdr + (size_t)NEXP * (2 * w13 + w2);
}

static int parse_kv(const char *ln, const char *key, double *out) {
    const char *p = strstr(ln, key);
    if (!p) return 0;
    return sscanf(p + strlen(key), "%lf", out) == 1;
}

/* hotcurve.json 的 cover_curve 读取。刻意不引通用 JSON 解析器 —— 只认这一种形状,
 * 顺序扫 "L": <n> 记住当前层, 遇 "cover_curve": [ 就读 NEXP 个浮点。
 * cover_curve[k-1] = 该层点火权重最高的前 k 个专家吃掉的权重比例(锚路由实测)。 */
static int load_cover(const char *path, double cov[MAXL][NEXP], int *nl_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "覆盖曲线打不开 %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    size_t rd = fread(buf, 1, (size_t)sz, f); buf[rd] = 0; fclose(f);
    int curL = -1, nl = 0, nfilled = 0;
    for (char *p = buf; *p; ) {
        if (!strncmp(p, "\"L\":", 4)) { curL = atoi(p + 4); p += 4; continue; }
        if (!strncmp(p, "\"cover_curve\":", 14)) {
            p = strchr(p, '['); if (!p) break; p++;
            if (curL < 0 || curL >= MAXL) { fprintf(stderr, "覆盖曲线层号越界 %d\n", curL); free(buf); return -1; }
            for (int k = 0; k < NEXP; k++) {
                cov[curL][k] = strtod(p, &p);
                while (*p == ' ' || *p == ',') p++;
            }
            /* 曲线必须单调不减且收于 1: 不满足说明文件不是这个语义, 硬拒 —— 静默用错
             * 曲线会让分配器在假信号上做最优化, 跑完三小时才发现。 */
            if (cov[curL][NEXP-1] < 0.999) {
                fprintf(stderr, "★L%d 覆盖曲线末值 %.4f != 1, 语义不符, 硬拒★\n", curL, cov[curL][NEXP-1]);
                free(buf); return -1;
            }
            nfilled++; if (curL + 1 > nl) nl = curL + 1;
            continue;
        }
        p++;
    }
    free(buf);
    if (nfilled < 2) { fprintf(stderr, "★覆盖曲线只读到 %d 层★\n", nfilled); return -1; }
    if (nfilled != nl) { fprintf(stderr, "★覆盖曲线层号不连续(%d 条/最大层号 %d)★\n", nfilled, nl); return -1; }
    *nl_out = nl;
    return 0;
}

int main(int argc, char **argv) {
    const char *held_p = NULL, *lad_p = NULL, *oj = NULL, *ot = NULL, *wts_p = NULL, *hc_p = NULL;
    double budget_gib = 0.0; int hot_nc = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--held") && i + 1 < argc) held_p = argv[++i];
        else if (!strcmp(argv[i], "--ladder") && i + 1 < argc) lad_p = argv[++i];
        else if (!strcmp(argv[i], "--budget-gib") && i + 1 < argc) budget_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--out-json") && i + 1 < argc) oj = argv[++i];
        else if (!strcmp(argv[i], "--out-rplan") && i + 1 < argc) ot = argv[++i];
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) wts_p = argv[++i];
        else if (!strcmp(argv[i], "--hotcurve") && i + 1 < argc) hc_p = argv[++i];
        else if (!strcmp(argv[i], "--hot-nc") && i + 1 < argc) hot_nc = atoi(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }
    if ((!held_p && !wts_p) || !lad_p || budget_gib <= 0) {
        fprintf(stderr, "用法: rplan_solve (--weights <w.txt> | --held <plan.txt>) --ladder <l.txt> --budget-gib N\n"
                        "                 [--out-json f.json] [--out-rplan f.txt]\n");
        return 2;
    }

    /* ---- 档梯 ---- */
    tier_t T[MAXT]; int NT = 0;
    FILE *f = fopen(lad_p, "r");
    if (!f) { fprintf(stderr, "档梯打不开 %s\n", lad_p); return 2; }
    char ln[512];
    while (fgets(ln, sizeof ln, f)) {
        if (ln[0] == '#' || ln[0] == '\n') continue;
        tier_t t; memset(&t, 0, sizeof t);
        const char *h = strchr(ln, '#');
        if (sscanf(ln, "%d %d %lf", &t.dim, &t.nc, &t.cos_) != 3) continue;
        if (h) { snprintf(t.note, sizeof t.note, "%s", h + 1); char *nl = strchr(t.note, '\n'); if (nl) *nl = 0; }
        t.bytes = layer_bytes(&t);
        if (NT >= MAXT) { fprintf(stderr, "档梯过长\n"); return 2; }
        T[NT++] = t;
    }
    fclose(f);
    if (NT < 2) { fprintf(stderr, "★档梯至少两档, 只读到 %d★\n", NT); return 2; }
    /* 单调性硬校验: 越贵必须越准。非单调会在贪心里造负收益台阶, 把所有层卡死在最稀档
     * (v4 原版实测事故: v24x256 cos 0.468 < v32 的 0.470 却更贵, 全线停在 v32 升不上去)。 */
    for (int i = 1; i < NT; i++)
        if (T[i].bytes <= T[i-1].bytes || T[i].cos_ <= T[i-1].cos_) {
            fprintf(stderr, "★档梯非单调: 第%d档 vq%dx%d(%zuB cos%.4f) vs 第%d档 vq%dx%d(%zuB cos%.4f)"
                    " — 越贵必须越准, 否则贪心必卡死★\n",
                    i-1, T[i-1].dim, T[i-1].nc, T[i-1].bytes, T[i-1].cos_,
                    i, T[i].dim, T[i].nc, T[i].bytes, T[i].cos_);
            return 3;
        }

    /* ---- 逐层难度 ---- */
    double relh[MAXL]; int NL = 0;
    memset(relh, 0, sizeof relh);
    if (held_p) {
    f = fopen(held_p, "r");
    if (!f) { fprintf(stderr, "held 表打不开 %s\n", held_p); return 2; }
    while (fgets(ln, sizeof ln, f)) {
        double L, v;
        if (!parse_kv(ln, "L=", &L) || !parse_kv(ln, "relh=", &v)) continue;
        int li = (int)L;
        if (li < 0 || li >= MAXL) continue;
        relh[li] = v; if (li + 1 > NL) NL = li + 1;
    }
    fclose(f);
    if (NL < 2) { fprintf(stderr, "★held 表只解析到 %d 层★\n", NL); return 2; }
    }

    /* ★难度信号的来源决定成败, 所以做成可注入的输入而不是写死的推导★
     *
     * --weights: 外部实测的逐层难度(首选)。每行 "L w  # 出处"。
     * --held:    从量化器 plan.txt 的累积 relh 推导 —— ★已知有缺陷, 只作兜底★。
     *   缺陷: relh 是【累积】相对误差, 深层饱和(vq86h 实测 L27 起恒在 0.52 附近), 于是
     *   能量增量 relh²[L]−relh²[L−1] 趋近 0, 全模型错得最狠的 L28 反被判成"最不需要 bit"。
     *   实跑 43 层有 17 层撞地板(2026-08-27 实测), 撞地板的层之间只能靠循环次序随机排位。
     *   v4 原版同样撞这堵墙(其注释: 累积曲线 L18 0.886 见顶 → L42 0.837), 它没修信号,
     *   是加"逐层不退步底线"绕过去的。本版选择修信号: 用端到端实测贡献注入。 */
    double w[MAXL];
    const double FLOOR = 0.004;   /* 自愈层(能量负增量)仍要参与竞价, 不能给 0 权重 */
    if (wts_p) {
        for (int L = 0; L < MAXL; L++) w[L] = -1.0;
        FILE *wf = fopen(wts_p, "r");
        if (!wf) { fprintf(stderr, "难度表打不开 %s\n", wts_p); return 2; }
        int nw = 0;
        while (fgets(ln, sizeof ln, wf)) {
            if (ln[0] == '#' || ln[0] == '\n') continue;
            int L; double v;
            if (sscanf(ln, "%d %lf", &L, &v) != 2) continue;
            if (L < 0 || L >= MAXL) continue;
            w[L] = v; nw++; if (L + 1 > NL) NL = L + 1;
        }
        fclose(wf);
        if (nw < 2) { fprintf(stderr, "★难度表只解析到 %d 行★\n", nw); return 2; }
        for (int L = 0; L < NL; L++)
            if (w[L] < 0) { fprintf(stderr, "★难度表缺 L%d — 缺层会被静默当 0 资源, 硬拒★\n", L); return 2; }
        printf("[难度] 外部实测注入 %s (%d 层)\n", wts_p, nw);
    } else {
        for (int L = 0; L < NL; L++) {
            double h2 = relh[L] * relh[L], p2 = L ? relh[L-1] * relh[L-1] : 0.0;
            w[L] = h2 - p2; if (w[L] < FLOOR) w[L] = FLOOR;
        }
        int nfloor = 0;
        for (int L = 0; L < NL; L++) if (w[L] <= FLOOR) nfloor++;
        printf("[难度] 由累积 relh 推导(兜底路) — %d/%d 层撞地板 %.3f\n", nfloor, NL, FLOOR);
        if (nfloor * 3 > NL)
            fprintf(stderr, "★警告: 过半以上层撞地板, 该信号已饱和失效, 分配基本是随机排位。"
                            "请改用 --weights 注入端到端实测难度★\n");
    }

    /* ---- 覆盖曲线(可选): 有则开双信号(每层热数 + 冷档), 无则只动每层档位 ---- */
    /* 【为什么要双信号】67.7G 超冠的实证配方就是"每层各自的 top-K 专家给密档, 其余给稀档"
     * (r64 战役: 逐层 top-108 @ vq4x512 + 其余 1bit, 混合 1.53bpw, Same-top 81.35%)。
     * 单看层不看专家, 等于把密档白送给那些几乎不点火的专家。覆盖曲线告诉我们每层的
     * 路由集中度: L28 前 108 个专家吃掉 94.5% 权重, 而 L0(哈希层)只吃 70.0% ——
     * 所以 K 必须逐层选, 冠军对全部层用同一个 108 反而是浪费。 */
    static double cov[MAXL][NEXP];
    int two_sig = 0, HR = NT - 1;
    if (hc_p) {
        int nl_c = 0;
        if (load_cover(hc_p, cov, &nl_c) != 0) return 2;
        if (nl_c != NL) { fprintf(stderr, "★覆盖曲线 %d 层 vs 难度 %d 层, 不匹配★\n", nl_c, NL); return 2; }
        if (hot_nc) { HR = -1; for (int t = 0; t < NT; t++) if (T[t].nc == hot_nc) HR = t;
                      if (HR < 0) { fprintf(stderr, "★档梯里没有 nc=%d★\n", hot_nc); return 2; } }
        if (HR < 1) { fprintf(stderr, "★热档必须严格贵于至少一个冷档★\n"); return 2; }
        two_sig = 1;
        printf("[双信号] 覆盖曲线 %s; 热档 vq%dx%d(cos %.4f), 冷档梯 %d 级\n",
               hc_p, T[HR].dim, T[HR].nc, T[HR].cos_, HR);
    }

    /* ---- 贪心分配 ---- */
    size_t B = (size_t)(budget_gib * (double)(1ull << 30));
    int tier[MAXL], hotk[MAXL];
    size_t used = 0;
    for (int L = 0; L < NL; L++) { tier[L] = 0; hotk[L] = 0;
        used += two_sig ? (VQ_HDR + (size_t)NEXP * expert_bytes(&T[0])) : T[0].bytes; }
    if (used > B) {
        fprintf(stderr, "★最稀档 %d 层已占 %.3f GiB > 预算 %.3f GiB★\n",
                NL, used / (double)(1ull << 30), budget_gib);
        return 3;
    }
    printf("[起点] %d 层全最稀档 vq%dx%d = %.3f GiB / 预算 %.3f GiB\n",
           NL, T[0].dim, T[0].nc, used / (double)(1ull << 30), budget_gib);

    long steps = 0;
    for (;;) {
        int bestL = -1, bestMove = 0; double bestR = 0.0; size_t bestD = 0;
        for (int L = 0; L < NL; L++) {
            /* 动作①: 冷档升一级 —— 收益作用在"没被热覆盖"的那部分权重上 */
            if (tier[L] + 1 < (two_sig ? HR : NT)) {
                size_t dB = (size_t)(NEXP - hotk[L]) * (expert_bytes(&T[tier[L]+1]) - expert_bytes(&T[tier[L]]));
                double share = (two_sig && hotk[L]) ? (1.0 - cov[L][hotk[L]-1]) : 1.0;   /* 冷档只管没被热覆盖的权重 */
                double r = w[L] * share * (T[tier[L]+1].cos_ - T[tier[L]].cos_) / (double)dB;
                if (dB && used + dB <= B && r > bestR) { bestR = r; bestL = L; bestMove = 1; bestD = dB; }
            }
            /* 动作②: 多一个热专家 —— 收益 = 这一个专家承担的路由权重 ×(热档 − 冷档)cos 差 */
            if (two_sig && hotk[L] < NEXP) {
                size_t dB = expert_bytes(&T[HR]) - expert_bytes(&T[tier[L]]);
                double dcov = cov[L][hotk[L]] - (hotk[L] ? cov[L][hotk[L]-1] : 0.0);
                double r = w[L] * dcov * (T[HR].cos_ - T[tier[L]].cos_) / (double)dB;
                if (dB && used + dB <= B && r > bestR) { bestR = r; bestL = L; bestMove = 2; bestD = dB; }
            }
        }
        if (bestL < 0) break;
        used += bestD;
        if (bestMove == 1) tier[bestL]++; else hotk[bestL]++;
        steps++;
    }
    double obj = 0, wsum = 0;
    for (int L = 0; L < NL; L++) {
        double c = two_sig && hotk[L]
                 ? cov[L][hotk[L]-1] * T[HR].cos_ + (1.0 - cov[L][hotk[L]-1]) * T[tier[L]].cos_
                 : T[tier[L]].cos_;
        obj += w[L] * c; wsum += w[L];
    }
    printf("[收敛] %ld 步, 落位 %.4f GiB / 预算 %.4f GiB (余 %.1f MiB), 加权精度 %.6f\n",
           steps, used / (double)(1ull << 30), budget_gib, (B - used) / 1048576.0, obj / wsum);

    /* ---- 逐层表 ---- */
    printf("\n%-4s %9s %8s %11s %6s %9s %9s\n", "层", "难度w", "热覆盖", "冷档", "热数", "层精度", "MiB");
    int hist[MAXT]; memset(hist, 0, sizeof hist);
    size_t bl_all = 0;
    for (int L = 0; L < NL; L++) {
        double hc = (two_sig && hotk[L]) ? cov[L][hotk[L]-1] : 0.0;
        double c  = two_sig && hotk[L] ? hc * T[HR].cos_ + (1.0 - hc) * T[tier[L]].cos_ : T[tier[L]].cos_;
        size_t bl = two_sig ? VQ_HDR + (size_t)hotk[L] * expert_bytes(&T[HR])
                                     + (size_t)(NEXP - hotk[L]) * expert_bytes(&T[tier[L]])
                            : T[tier[L]].bytes;
        bl_all += bl; hist[tier[L]]++;
        printf("L%-3d %9.5f %7.1f%% vq%dx%-6d %6d %9.4f %9.1f\n",
               L, w[L], hc * 100.0, T[tier[L]].dim, T[tier[L]].nc, hotk[L], c, bl / 1048576.0);
    }
    printf("\n冷档分布: ");
    for (int t = 0; t < NT; t++) if (hist[t]) printf("vq%dx%d×%d  ", T[t].dim, T[t].nc, hist[t]);
    if (two_sig) {
        int hs = 0, hmin = NEXP, hmax = 0;
        for (int L = 0; L < NL; L++) { hs += hotk[L]; if (hotk[L] < hmin) hmin = hotk[L]; if (hotk[L] > hmax) hmax = hotk[L]; }
        printf("\n热数: 均值 %.1f 范围 %d-%d (热档 vq%dx%d)", hs / (double)NL, hmin, hmax, T[HR].dim, T[HR].nc);
        printf("\n混合位宽: %.4f bpw", bl_all * 8.0 / ((double)NL * NEXP * 3 * MOEI * DMODEL));
    }
    printf("\n");

    /* ---- 产物 ---- */
    if (ot) {
        FILE *o = fopen(ot, "w");
        if (!o) { fprintf(stderr, "写不了 %s\n", ot); return 4; }
        fprintf(o, "# rplan(rplan_solve 生成) 预算 %.4f GiB, 落位 %.4f GiB\n", budget_gib, used / (double)(1ull << 30));
        fprintf(o, "# dim/nc=冷档(hot 之外的专家), hot=热专家个数, hotdim/hotnc=热档(缺省 4/512)\n");
        for (int L = 0; L < NL; L++)
            fprintf(o, "L=%d dim=%d nc=%d hot=%d w2dim=%d w2nc=%d hotdim=%d hotnc=%d\n",
                    L, T[tier[L]].dim, T[tier[L]].nc, hotk[L], T[tier[L]].dim, T[tier[L]].nc,
                    two_sig ? T[HR].dim : 4, two_sig ? T[HR].nc : 512);
        fclose(o);
        printf("计划表 → %s\n", ot);
    }
    if (oj) {
        FILE *o = fopen(oj, "w");
        if (!o) { fprintf(stderr, "写不了 %s\n", oj); return 4; }
        fprintf(o, "{\n  \"budget_GiB\": %.4f,\n  \"payload_GiB\": %.4f,\n  \"objective\": %.6f,\n"
                   "  \"n_layers\": %d,\n  \"two_signal\": %d,\n"
                   "  \"hot_tier\": {\"dim\": %d, \"nc\": %d, \"cos\": %.4f},\n  \"cold_ladder\": [\n",
                budget_gib, used / (double)(1ull << 30), obj / wsum, NL, two_sig,
                two_sig ? T[HR].dim : 4, two_sig ? T[HR].nc : 512, two_sig ? T[HR].cos_ : 0.0);
        for (int t = 0; t < (two_sig ? HR : NT); t++)
            fprintf(o, "    {\"dim\": %d, \"nc\": %d, \"cos\": %.4f, \"MiB\": %.1f}%s\n",
                    T[t].dim, T[t].nc, T[t].cos_, T[t].bytes / 1048576.0, t + 1 < (two_sig ? HR : NT) ? "," : "");
        fprintf(o, "  ],\n  \"layers\": [\n");
        for (int L = 0; L < NL; L++) {
            double hc = (two_sig && hotk[L]) ? cov[L][hotk[L]-1] : 0.0;
            size_t bl = two_sig ? VQ_HDR + (size_t)hotk[L] * expert_bytes(&T[HR])
                                         + (size_t)(NEXP - hotk[L]) * expert_bytes(&T[tier[L]])
                                : T[tier[L]].bytes;
            fprintf(o, "    {\"L\": %d, \"weight\": %.5f, \"cold_dim\": %d, \"cold_nc\": %d, \"hot\": %d,"
                       " \"hot_cover\": %.4f, \"MiB\": %.1f}%s\n",
                    L, w[L], T[tier[L]].dim, T[tier[L]].nc, hotk[L], hc, bl / 1048576.0, L + 1 < NL ? "," : "");
        }
        fprintf(o, "  ]\n}\n");
        fclose(o);
        printf("配置 JSON → %s\n", oj);
    }
    return 0;
}
