/* dump_gguf_meta.c — 内存安全 GGUF 元数据 dump(只读头/KV/张量表, 永不碰张量数据)。
 * (C, 2026-08-25 Python→C 迁移; 取代 dump_gguf_meta.py, 输出对齐 python 打印形态:
 *  bool=True/False, 数组=python list 形态, 字符串截80+"...", f32 用最短往返打印 —
 *  奇异浮点 KV 的 repr 尾数可能与 python 差末位, 属打印面非数值面)
 * 用法: dump_gguf_meta FILE.gguf [--stats] [--vqhead] [--fp4scan]
 *
 * --fp4scan(2026-09-15, speed.md S1 前置): 扫全部 fp4x32(type 43)张量的 ue8m0 缩放指数直方图。
 * 【为什么要它】S0 判决 = 预填要走 NVFP4(E2M1 + 每 16 通道 E4M3 缩放), 而盘上是 MXFP4(ue8m0/32)。
 * 两者数值恒等的前提是那个 2 的幂能被 e4m3 精确表示, 即 2^(e−127) ∈ [2^−9, 2^8] ⇒ e ∈ [118, 135]。
 * 越界块若非全零, 转换就是有损的 —— 那必须在转换器里停车, 而不是静默饱和。本扫描先回答"有没有越界块"。*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

static const char *GT[13] = {"u8","i8","u16","i16","u32","i32","f32","bool","str","arr","u64","i64","f64"};
static const int GSZ[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};

static FILE *F;
static void rd(void *p, size_t n) { if (fread(p, 1, n, F) != n) { fprintf(stderr, "读不满\n"); exit(2); } }
static uint32_t r32(void) { uint32_t v; rd(&v, 4); return v; }
static uint64_t r64(void) { uint64_t v; rd(&v, 8); return v; }
static char *rstr(void) {
    uint64_t ln = r64();
    char *s = malloc(ln + 1); rd(s, ln); s[ln] = 0; return s;
}
/* python repr(float) 最短往返近似: 整数值浮点在 |v|<1e16 时 python 打定点(如 10000.0),
 * 其余走最短 %g。奇异非整值大数的 repr 尾形可能与 python 差(打印面, 非数值面)。 */
static void pfloat(char *buf, double v) {
    if (v == (double)(long long)v && v < 1e16 && v > -1e16) {
        snprintf(buf, 64, "%lld.0", (long long)v);
        if (strtod(buf, NULL) == v) return;
    }
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(buf, 64, "%.*g", prec, v);
        if (strtod(buf, NULL) == v) break;
    }
    if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'n') && !strchr(buf, 'i'))
        strcat(buf, ".0");
}
static void pscalar(char *buf, int t, const uint8_t *p) {
    switch (t) {
        case 0: snprintf(buf, 64, "%u", *(uint8_t *)p); break;
        case 1: snprintf(buf, 64, "%d", *(int8_t *)p); break;
        case 2: snprintf(buf, 64, "%u", *(uint16_t *)p); break;
        case 3: snprintf(buf, 64, "%d", *(int16_t *)p); break;
        case 4: snprintf(buf, 64, "%u", *(uint32_t *)p); break;
        case 5: snprintf(buf, 64, "%d", *(int32_t *)p); break;
        case 6: { float f; memcpy(&f, p, 4); pfloat(buf, (double)f); break; }
        case 7: snprintf(buf, 64, "%s", *(uint8_t *)p ? "True" : "False"); break;
        case 10: snprintf(buf, 64, "%llu", (unsigned long long)*(uint64_t *)p); break;
        case 11: snprintf(buf, 64, "%lld", (long long)*(int64_t *)p); break;
        case 12: { double d; memcpy(&d, p, 8); pfloat(buf, d); break; }
        default: snprintf(buf, 64, "?"); break;
    }
}

typedef struct { uint64_t off; char *name; int nd; uint64_t dims[8]; uint32_t tt; } tinfo_t;
static int cmp_tinfo(const void *a, const void *b) {
    const tinfo_t *x = a, *y = b;
    return x->off < y->off ? -1 : (x->off > y->off ? 1 : 0);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: dump_gguf_meta FILE.gguf [--stats] [--vqhead]\n"); return 1; }
    int stats = 0, vqhead = 0, fp4scan = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--stats")) stats = 1;
        if (!strcmp(argv[i], "--vqhead")) vqhead = 1;
        if (!strcmp(argv[i], "--fp4scan")) fp4scan = 1;
    }
    F = fopen(argv[1], "rb");
    if (!F) { fprintf(stderr, "%s 打不开\n", argv[1]); return 2; }
    char magic[4]; rd(magic, 4);
    if (memcmp(magic, "GGUF", 4)) { fprintf(stderr, "AssertionError: bad magic\n"); return 1; }
    uint32_t version = r32();
    uint64_t n_tensors = r64(), n_kv = r64();
    printf("== %s\n", argv[1]);
    printf("version=%u n_tensors=%llu n_kv=%llu\n", version, (unsigned long long)n_tensors, (unsigned long long)n_kv);
    printf("--- KV ---\n");
    for (uint64_t k = 0; k < n_kv; k++) {
        char *key = rstr();
        uint32_t t = r32();
        if (t == 9) {
            uint32_t et = r32();
            uint64_t cnt = r64();
            char head[2048] = "["; int hn = 0;
            if (et == 8) {
                for (uint64_t i = 0; i < cnt; i++) {
                    char *s = rstr();
                    if (i < 12) {   /* python repr 引号规则: 含 ' 且不含 " → 双引号包裹 */
                        char q = (strchr(s, '\'') && !strchr(s, '"')) ? '"' : '\'';
                        char e[256]; snprintf(e, sizeof e, "%s%c%s%c", i ? ", " : "", q, s, q);
                        if (strlen(head) + strlen(e) < 2000) strcat(head, e); hn++; }
                    free(s);
                }
            } else {
                int sz = GSZ[et];
                for (uint64_t i = 0; i < cnt; i++) {
                    uint8_t raw[8]; rd(raw, sz);
                    if (i < 12) { char v[64], e[80]; pscalar(v, (int)et, raw);
                        snprintf(e, sizeof e, "%s%s", i ? ", " : "", v);
                        if (strlen(head) + strlen(e) < 2000) strcat(head, e); hn++; }
                }
            }
            strcat(head, "]");
            printf("  %s : arr<%s>[%llu] %s%s\n", key, et < 13 ? GT[et] : "?", (unsigned long long)cnt,
                   head, cnt > 12 ? " ..." : "");
        } else if (t == 8) {
            char *v = rstr();
            char sv[100];
            if (strlen(v) > 80) { memcpy(sv, v, 80); strcpy(sv + 80, "..."); }
            else strcpy(sv, v);
            printf("  %s : str = %s\n", key, sv);
            free(v);
        } else {
            uint8_t raw[8]; rd(raw, GSZ[t]);
            char v[64]; pscalar(v, (int)t, raw);
            printf("  %s : %s = %s\n", key, t < 13 ? GT[t] : "?", v);
        }
        free(key);
    }
    printf(stats ? "--- TENSORS (type stats)\n" : "--- TENSORS (expert + sample) ---\n");
    tinfo_t *infos = malloc(n_tensors * sizeof(tinfo_t));
    int shown = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo_t *ti = &infos[i];
        ti->name = rstr();
        ti->nd = (int)r32();
        for (int d = 0; d < ti->nd; d++) ti->dims[d] = r64();
        ti->tt = r32(); ti->off = r64();
        int is_exp = strstr(ti->name, "_exps.") && (strstr(ti->name, "blk.0.") || strstr(ti->name, "blk.1."));
        if (!stats && (is_exp || shown < 6)) {
            printf("  %s  dims=[", ti->name);
            for (int d = 0; d < ti->nd; d++) printf("%s%llu", d ? ", " : "", (unsigned long long)ti->dims[d]);
            printf("] type=%u off=%llu\n", ti->tt, (unsigned long long)ti->off);
            shown++;
        }
    }
    if (vqhead) {
        long align = 32;
        long long base = ftell(F);
        if (base % align) base += align - (base % align);
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (!strstr(infos[i].name, "ffn_exps_vq.blob")) continue;
            fseek(F, (long)(base + infos[i].off), SEEK_SET);
            uint8_t hdr[16 + 256 * 3 * 8]; rd(hdr, sizeof hdr);
            uint32_t mg, ver2, L, nexp;
            memcpy(&mg, hdr, 4); memcpy(&ver2, hdr + 4, 4); memcpy(&L, hdr + 8, 4); memcpy(&nexp, hdr + 12, 4);
            uint64_t tab[256 * 3]; memcpy(tab, hdr + 16, sizeof tab);
            printf("  %s: magic=%08x ver=%u L=%u nexp=%u bytes=[", infos[i].name, mg, ver2, L, nexp);
            for (int d = 0; d < infos[i].nd; d++) printf("%s%llu", d ? ", " : "", (unsigned long long)infos[i].dims[d]);
            printf("]\n");
            const char *wn[3] = {"w1", "w3", "w2"};
            for (int w = 0; w < 3; w++) {
                uint64_t po = tab[w];
                if (!po) { printf("    %s: 槽缺席(off=0)\n", wn[w]); continue; }
                fseek(F, (long)(base + infos[i].off + po), SEEK_SET);
                uint8_t ph[16]; rd(ph, 16);
                uint32_t pmg, rows, cols; uint16_t dim, nc;
                memcpy(&pmg, ph, 4); memcpy(&dim, ph + 4, 2); memcpy(&nc, ph + 6, 2);
                memcpy(&rows, ph + 8, 4); memcpy(&cols, ph + 12, 4);
                printf("    %s: magic=%08x dim=%u nc=%u rows=%u cols=%u\n", wn[w], pmg, dim, nc, rows, cols);
            }
            int zeros[3] = {0, 0, 0};
            typedef struct { int w, d, n, c; } nck_t;
            nck_t ncs[64]; int nn = 0;
            for (int e = 0; e < 256; e++) for (int w = 0; w < 3; w++) {
                uint64_t o = tab[e * 3 + w];
                if (!o) { zeros[w]++; continue; }
                fseek(F, (long)(base + infos[i].off + o), SEEK_SET);
                uint8_t ph[8]; rd(ph, 8);
                uint16_t d16, n16; memcpy(&d16, ph + 4, 2); memcpy(&n16, ph + 6, 2);
                int found = 0;
                for (int j = 0; j < nn; j++) if (ncs[j].w == w && ncs[j].d == d16 && ncs[j].n == n16) { ncs[j].c++; found = 1; break; }
                if (!found && nn < 64) { ncs[nn++] = (nck_t){w, d16, n16, 1}; }
            }
            printf("    槽缺席: w1=%d w3=%d w2=%d (fused2 要求全 0)\n", zeros[0], zeros[1], zeros[2]);
            /* 排序 (w,d,n) 升序 = python sorted(ncs.items()) 同序 */
            for (int a = 1; a < nn; a++) { nck_t x = ncs[a]; int b = a - 1;
                while (b >= 0 && (ncs[b].w > x.w || (ncs[b].w == x.w && (ncs[b].d > x.d || (ncs[b].d == x.d && ncs[b].n > x.n))))) { ncs[b+1] = ncs[b]; b--; }
                ncs[b+1] = x; }
            for (int j = 0; j < nn; j++)
                printf("    形态 %s: dim=%d nc=%d × %d 个专家\n", wn[ncs[j].w], ncs[j].d, ncs[j].n, ncs[j].c);
            break;
        }
        fclose(F); return 0;
    }
    if (fp4scan) {
        /* e4m3fn 能精确表示的 2 的幂: 次正规 2^-9..2^-7 + 正规 2^-6..2^8 ⇒ ue8m0 字节 e ∈ [118, 135] */
        enum { E_LO = 118, E_HI = 135, CHUNK = 17 * 65536 };
        long align = 32;
        long long base = ftell(F);
        if (base % align) base += align - (base % align);
        unsigned char *buf = malloc(CHUNK);
        long long hist[256]; memset(hist, 0, sizeof hist);
        long long nblk_tot = 0, nbad = 0, nbad_nz = 0, ntensor = 0;
        char worst[128] = ""; int worst_e = -1;
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (infos[i].tt != 43) continue;   /* DS4_GGT_FP4X32 */
            uint64_t numel = 1;
            for (int d = 0; d < infos[i].nd; d++) numel *= infos[i].dims[d];
            const uint64_t nblk = numel / 32;
            ntensor++;
            if (fseek(F, (long)(base + infos[i].off), SEEK_SET)) { printf("  seek 失败 %s\n", infos[i].name); continue; }
            uint64_t left = nblk;
            while (left) {
                const uint64_t take = left > 65536 ? 65536 : left;
                if (fread(buf, 1, (size_t)take * 17, F) != (size_t)take * 17) { printf("  读不满 %s\n", infos[i].name); break; }
                for (uint64_t b = 0; b < take; b++) {
                    const unsigned char *blk = buf + b * 17;
                    const unsigned e = blk[16];
                    hist[e]++;
                    if (e < E_LO || e > E_HI) {
                        nbad++;
                        int nz = 0;
                        for (int j = 0; j < 16; j++) if (blk[j]) { nz = 1; break; }
                        if (nz) { nbad_nz++; if ((int)e > worst_e) { worst_e = (int)e; snprintf(worst, sizeof worst, "%s", infos[i].name); } }
                    }
                }
                left -= take; nblk_tot += take;
            }
        }
        free(buf);
        printf("--- FP4X32 缩放扫描(%lld 个张量, %lld 块)\n", ntensor, nblk_tot);
        int lo = 255, hi = 0;
        for (int e = 0; e < 256; e++) if (hist[e]) { if (e < lo) lo = e; if (e > hi) hi = e; }
        printf("  ue8m0 指数字节范围 [%d, %d] = 2^[%d, %d]   (e4m3 可表示区间 [%d, %d] = 2^[-9, 8])\n",
               lo, hi, lo - 127, hi - 127, E_LO, E_HI);
        for (int e = lo; e <= hi && e < 256; e++)
            if (hist[e]) printf("    2^%-4d %12lld 块 %6.2f%%%s\n", e - 127, hist[e],
                                100.0 * hist[e] / (nblk_tot ? nblk_tot : 1), (e < E_LO || e > E_HI) ? "  ★越界★" : "");
        printf("  越界块 %lld(其中非全零 %lld)%s\n", nbad, nbad_nz,
               nbad_nz ? "  ★MXFP4→NVFP4 非恒等, 转换器必须停车★" : "  ⇒ ★MXFP4→NVFP4 数值恒等★");
        if (nbad_nz) printf("  最大越界指数出现在 %s (2^%d)\n", worst, worst_e - 127);
    }
    if (stats) {
        struct stat st; stat(argv[1], &st);
        long long end = st.st_size;
        qsort(infos, n_tensors, sizeof(tinfo_t), cmp_tinfo);
        typedef struct { uint32_t tt; char grp[16]; long long c, nb; } agg_t;
        agg_t agg[128]; int na = 0;
        long long tot = 0;
        for (uint64_t i = 0; i < n_tensors; i++) {
            long long nb = (i + 1 < n_tensors) ? (long long)(infos[i+1].off - infos[i].off) : 0;
            const char *grp = strstr(infos[i].name, "_exps.") ? "routed_exps" :
                              (strstr(infos[i].name, ".attn") ? "attn" :
                               (strstr(infos[i].name, "ffn") ? "ffn_shared" : "other"));
            int found = 0;
            for (int j = 0; j < na; j++) if (agg[j].tt == infos[i].tt && !strcmp(agg[j].grp, grp)) { agg[j].c++; agg[j].nb += nb; found = 1; break; }
            if (!found && na < 128) { agg[na].tt = infos[i].tt; snprintf(agg[na].grp, 16, "%s", grp); agg[na].c = 1; agg[na].nb = nb; na++; }
            tot += nb;
        }
        /* 按 nb 降序 = python sorted(key=-nb) */
        for (int a = 1; a < na; a++) { agg_t x = agg[a]; int b = a - 1;
            while (b >= 0 && agg[b].nb < x.nb) { agg[b+1] = agg[b]; b--; } agg[b+1] = x; }
        printf("  %5s %-12s %7s %9s %7s\n", "type", "组", "张量数", "GiB", "占比");
        for (int j = 0; j < na; j++)
            printf("  %5u %-12s %7lld %9.2f %6.1f%%\n", agg[j].tt, agg[j].grp, agg[j].c,
                   agg[j].nb / 1073741824.0, 100.0 * agg[j].nb / (tot > 0 ? tot : 1));
        printf("  合计(按 offset 差) = %.2f GiB, 文件 %.2f GiB\n", tot / 1073741824.0, end / 1073741824.0);
    }
    fclose(F);
    return 0;
}
