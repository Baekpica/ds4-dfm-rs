/* v41_to_gguf_vq3.inc.c — DQVL v3 专家 blob 的登记与字节生成(2026-09-21, 113.md 方案 v3)。
 * v41_to_gguf.c 单 TU include(拆出去只为守 500 行, 与 v41_to_gguf_mtp.inc.c 同例)。
 *
 * 【v3 相对 v2 改三件, 都只搬字节不改数值】
 *   ① 码本一层一本: v2 每个矩阵载荷各带一份拷贝(同一本抄 3 遍 × 15,744 本 = 3.09 GB)。v3 把它提到 blob 头后面,
 *      载荷用 cb_off 指过去 —— 省下的 3.09 GB 就是 14 个浅层升 13 位的钱。
 *      ★只有目录里 384 份码本逐字节相同才切 v3★(量化器 --vq-shared-cb 的产物); 否则原样走 v2, 老目录转出来的文件一个字节不变。
 *   ② 码本只存 E4M3(1 B/元素): 量化器 --vq-cb-fp8 已把码本值舍到 E4M3 格点 ⇒ f16 → E4M3 字节【无损】。
 *      为的是 8192 词码本 = 64 KB, 与今天 4096 词 f16 一样大, 塞得进同一块 shared(GB10 每 block 上限 99 KB)。
 *      每个值往返一趟核对, 有一个回不来就**硬停**(不退 f16): 那只会在"量化忘了传 --vq-cb-fp8"时发生, 是配方错,
 *      该停车重量化, 而不是替它生一种解码核没有实例的盘上格式。
 *   ③ 13 位索引拆成"12 位主流 + 1 位平面": 目录里是直排 13 位, 直接搬进 GGUF 会让解码核一块变成 3.25 条 128 B 线、
 *      行首只 16 B 对齐 —— 09-18 用 mem_ceiling 量死过"读不满整线掉到 142 GB/s"。拆开后主流几何与 12 位逐字节同,
 *      第 13 位单独一个每行 (nidx_row+7)/8 字节的平面, 解码核每块多一条 LDG.32 + 每轮一条 shfl。
 *
 * 【字节布局】见 113.md §3.4.1。槽表仍在 offset 16 ⇒ 老引擎读 v3 会在载荷 magic('DQV3' ≠ 'DQVQ')上认不出来,
 * 结果是"专家输出全零"而不是假数; 新引擎用 vq_fmt.h 的 ver 白名单在加载时硬停。 */

/* 位搬运与它的金标自测住 v41_vq3_bits.c(单独可编译: make -C gguf-tools tools-test) */
#include "v41_vq3_bits.c"

#define VQ3_BLOB_VER   3u
#define VQ3_MAT_MAGIC  DS4VQ_MAT3_MAGIC     /* 魔数住 vq_fmt.h(引擎也读它) */
#define VQ3_HDR        32u                  /* 载荷头字节 */
#define VQ3_FLAG_FP8   1u                   /* 码本 E4M3 */
#define VQ3_FLAG_EXT   2u                   /* 有第 13 位平面 */

/* 本层/本塔的码本: 是否可以一层一本(384 份逐字节同) + 能不能存 E4M3。类型与声明在 v41_to_gguf.c(recipe KV 也要问它)。 */
static vq3_cb vq3_probe(const char *pre, int nexp) {
    vq3_cb r; memset(&r, 0, sizeof r);
    char n[192];
    snprintf(n, sizeof n, "%s.ffn.experts.0.vq.cb", pre);
    const v41_st_ent *cb0 = need_in(n, 0); const uint8_t *p0 = v41_st_data(&S, cb0);
    if (!p0) exit(1);
    r.nc = (int)cb0->shape[0]; r.dim = (int)cb0->shape[1];
    const size_t nb = (size_t)r.nc * r.dim * 2;
    r.shared = 1;
    for (int e = 1; e < nexp && r.shared; e++) {
        snprintf(n, sizeof n, "%s.ffn.experts.%d.vq.cb", pre, e);
        const v41_st_ent *cb = need_in(n, 0);
        if ((int)cb->shape[0] != r.nc || (int)cb->shape[1] != r.dim) r.shared = 0;
        else { const uint8_t *p = v41_st_data(&S, cb); if (!p) exit(1); if (memcmp(p0, p, nb)) r.shared = 0; }
    }
    if (r.shared) {
        /* ★v3 的码本必须是 E4M3★: 每个值 E4M3 往返一趟, 一个回不来就硬停(不退 f16)。
         * 为什么不留 f16 变体: 留了就多一种盘上格式, 解码核要多一族实例(FP8 0/1 × EXT 0/1 = 4 档,
         * 编译时间与寄存器压力都翻倍), 而它只会在"量化时忘了传 --vq-cb-fp8"这一种情况下出现 ——
         * 那是配方错, 该在这里停车重量化, 不是替它生一种引擎没跑过的文件。 */
        for (int i = 0; i < r.nc * r.dim; i++) {
            uint16_t h; memcpy(&h, p0 + 2 * (size_t)i, 2);
            const float v = ds4_f16_to_f32(h);
            if (ds4_e4m3fn_to_f32(ds4_e4m3fn_f32_to_byte(v)) != v)
                die("共享码本的值不在 E4M3 格点上 —— 量化时要带 --vq-cb-fp8(v3 的码本只存 E4M3)");
        }
        r.fp8 = 1;
        r.cb_bytes = (uint64_t)r.nc * r.dim;
    }
    return r;
}

/* 一个矩阵的 v3 载荷字节数; 回填 rows/cols 与位宽相关的行字节 */
static uint64_t vq3_payload_bytes(const char *pre, int e, const char *m, const vq3_cb *cbi,
                                  uint64_t *rows, uint64_t *cols, uint64_t *main_row, uint64_t *ext_row, int *nbit) {
    char n[192];
    snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.idx", pre, e, m); const v41_st_ent *ix = need_in(n, 0);
    snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.gain", pre, e, m); const v41_st_ent *g = need_in(n, 0);
    int nb = 0; while ((1 << nb) < cbi->nc) nb++;
    if (nb < 1) nb = 1;
    *nbit = nb;
    *rows = (uint64_t)ix->shape[0];
    const uint64_t nidx_row = (uint64_t)ix->shape[1] * 8 / (uint64_t)nb;
    *cols = nidx_row * (uint64_t)cbi->dim;
    if ((uint64_t)g->shape[0] != *rows) die("vq.gain 行数与 idx 不符");
    if (nb < (int)VQ3_MAIN_NBIT || nb > (int)VQ3_MAIN_NBIT + 1) die("v3 只支持 12 或 13 位索引(更宽要加第二个位平面)");
    *main_row = (nidx_row * VQ3_MAIN_NBIT + 7) / 8;
    *ext_row = (nb > (int)VQ3_MAIN_NBIT) ? (nidx_row + 7) / 8 : 0;
    return VQ3_HDR + *rows * 2 + *rows * (*main_row) + *rows * (*ext_row) + 8;
}

/* 登记: 一层(或一塔)一个 blob 张量。v3 不成立(码本不同)就回退 v2 的登记器。 */
static void plan_vqblob_v3(const char *pre, int L, int nexp, const char *gname) {
    const vq3_cb cbi = vq3_probe(pre, nexp);
    if (!cbi.shared) { plan_vqblob(pre, L, nexp, gname); return; }   /* 老目录: 原样 v2 */
    static const char *mats[3] = {"w1", "w3", "w2"};
    uint64_t total = 16 + (uint64_t)nexp * 3 * 8;
    total = (total + 15) & ~15ull;
    total += cbi.cb_bytes;
    total = (total + 15) & ~15ull;
    for (int e = 0; e < nexp; e++) for (int w = 0; w < 3; w++) {
        uint64_t r, c, mr, er; int nb;
        total += vq3_payload_bytes(pre, e, mats[w], &cbi, &r, &c, &mr, &er, &nb);
    }
    uint64_t ne[1] = {total};
    add_job(J_VQBLOB3, gname, DS4_GGT_VQBLOB, 1, ne, total, pre, L, NULL);
    g_jobs[g_nj - 1].nexp = nexp;
}

static uint64_t gen_vqblob_v3(const job_t *j) {
    const int L = j->layer, nexp = j->nexp;
    const char *pre = j->src;
    static const char *mats[3] = {"w1", "w3", "w2"};
    const vq3_cb cbi = vq3_probe(pre, nexp);
    if (!cbi.shared) die("gen v3 却探到码本不共享(plan 与 gen 不一致)");
    uint64_t tab_end = 16 + (uint64_t)nexp * 3 * 8;
    const uint64_t cb_off = (tab_end + 15) & ~15ull;
    uint64_t off = (cb_off + cbi.cb_bytes + 15) & ~15ull;
    const uint64_t body = off;
    uint64_t total = off;
    for (int e = 0; e < nexp; e++) for (int w = 0; w < 3; w++) {
        uint64_t r, c, mr, er; int nb;
        total += vq3_payload_bytes(pre, e, mats[w], &cbi, &r, &c, &mr, &er, &nb);
    }
    uint8_t *o = buf(total);
    memset(o, 0, body);
    const uint32_t hdr[4] = {DS4VQ_BLOB_MAGIC, VQ3_BLOB_VER, (uint32_t)L, (uint32_t)nexp};
    memcpy(o, hdr, 16);
    uint64_t *tab = (uint64_t *)(o + 16);
    {   /* 层码本: 专家 0 的那一份(已验 384 份逐字节同) */
        char n[192]; snprintf(n, sizeof n, "%s.ffn.experts.0.vq.cb", pre);
        const uint8_t *p0 = v41_st_data(&S, need_in(n, 0)); if (!p0) exit(1);
        if (cbi.fp8) for (int i = 0; i < cbi.nc * cbi.dim; i++) {
            uint16_t h; memcpy(&h, p0 + 2 * (size_t)i, 2);
            o[cb_off + i] = ds4_e4m3fn_f32_to_byte(ds4_f16_to_f32(h));
        }
        else memcpy(o + cb_off, p0, (size_t)cbi.nc * cbi.dim * 2);
    }
    for (int e = 0; e < nexp; e++) {
        char n[192];
        for (int w = 0; w < 3; w++) {
            uint64_t rows, cols, mrow, erow; int nbit;
            vq3_payload_bytes(pre, e, mats[w], &cbi, &rows, &cols, &mrow, &erow, &nbit);
            snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.idx", pre, e, mats[w]);
            const v41_st_ent *ixe = need_in(n, 0); const uint8_t *ixp = v41_st_data(&S, ixe);
            snprintf(n, sizeof n, "%s.ffn.experts.%d.%s.vq.gain", pre, e, mats[w]);
            const uint8_t *gp = v41_st_data(&S, need_in(n, 0));
            if (!ixp || !gp) exit(1);
            uint8_t *p = o + off; tab[e * 3 + w] = off;
            const uint32_t mg = VQ3_MAT_MAGIC;
            const uint16_t d16 = (uint16_t)cbi.dim, n16 = (uint16_t)cbi.nc;
            const uint32_t r32 = (uint32_t)rows, c32 = (uint32_t)cols;
            const uint32_t flags = (cbi.fp8 ? VQ3_FLAG_FP8 : 0u) | (erow ? VQ3_FLAG_EXT : 0u), mnb = VQ3_MAIN_NBIT;
            memcpy(p, &mg, 4); memcpy(p + 4, &d16, 2); memcpy(p + 6, &n16, 2);
            memcpy(p + 8, &r32, 4); memcpy(p + 12, &c32, 4);
            memcpy(p + 16, &flags, 4); memcpy(p + 20, &mnb, 4); memcpy(p + 24, &cb_off, 8);
            p += VQ3_HDR;
            memcpy(p, gp, rows * 2); p += rows * 2;
            const uint64_t srow = (uint64_t)ixe->shape[1], nidx = cols / (uint64_t)cbi.dim;
            uint8_t *mp = p, *ep = p + rows * mrow;
            if (nbit == (int)VQ3_MAIN_NBIT) memcpy(mp, ixp, rows * mrow);   /* 12 位: 逐字节照搬, 与 v2 同 */
            else for (uint64_t r = 0; r < rows; r++)
                vq3_split_row(ixp + r * srow, srow, nidx, nbit, mp + r * mrow, ep + r * erow);
            p += rows * (mrow + erow);
            memset(p, 0, 8); p += 8;
            off = (uint64_t)(p - o);
        }
    }
    if (off != total) die("v3 blob 字节账不平");
    return total;
}
