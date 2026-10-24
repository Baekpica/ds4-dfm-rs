/* vq_merge_v4.c — v4 战役产物合并(2026-08-25 Python→C 迁移 Wave A): 换代式重打包+路由偏置烘焙。
 * 忠实转录 quant/vq_merge_v4.py(v4.1), 逐位同语义, 不改进不重构。
 *
 * 新一代合一文件 = 骨架(非专家张量, 取自现役 base, gate.bias 烘入 α·Δb 路由偏置)
 *               + 新 down(M1 dql_L*.bin 的 D 段, ssh 流式)
 *               + 新 blob(M1 dql_vq_L*.bin 真载荷修剪, ssh 流式, type42)。
 * blob 真尺寸由 scripts/vq_blob_truesize.py 预扫文件 --blob-sizes 提供(修剪预留空洞);
 * --route-bias/--route-alpha 把 RBIA 侧车(ds4quant_run DS4_ROUTE_BIAS_FIT 产物)按 mincnt
 * 门后 α 缩放烘进 blk.L.exp_probs_b.bias(F32 256, 只影响专家选择不影响混合权重)。
 * down 与 blob 必须同代(冷 w2 的 hc 顺序补偿是对新 q1/q3 算的)。
 *
 * 两阶段:
 *   --extract-skeleton: 从 base 抽非专家张量(~8.6G; 不含 down/blob/opt) → skeleton.gguf
 *   --merge: skeleton + ssh(M1 blobs 修剪流式) + ssh(M1 dql D 段) [+gate.bias 烘焙] → 输出
 * 用法:
 *   vq_merge_v4 --extract-skeleton --base ds4-vq22.gguf --out skeleton.gguf
 *   vq_merge_v4 --merge --skeleton skeleton.gguf --blob-sizes blob_sizes.txt \
 *       --route-bias route_bias_v4.bin --route-alpha 2.5 \
 *       --dql-host 192.168.1.2 --dql-dir /Users/fodelf/ds4-main/gguf/go-onebit/layers \
 *       --out ds4-vq4bf.gguf
 *
 * ★金标口径(硬闸)★: 同参数同输入下与 vq_merge_v4.py 的输出 GGUF 逐字节 md5 相同 ——
 *   ① --extract-skeleton;
 *   ② --merge 走本地 dql 目录模式(--dql-host local/""/-/127.0.0.1/localhost)。
 * 歧义点与处理决定见 migrate/vq_merge_v4_transcription_notes.md。
 *
 * 编译: gcc -O3 -o vq_merge_v4 vq_merge_v4.c -lm
 * 前提: 小端主机(GGUF/dql/RBIA 全部小端, 与 .py struct "<" 同; 大端机不支持)。
 */
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include "../../src/common/ds4_amp_fmt.h"   /* 116 记录头契约(单一定义源) */
#include <sys/stat.h>

#define ALIGN            32
#define N_LAYER          43
#define NEXP             256
#define SZ_G             (2048ULL * 16 * 34)      /* dql G/U 段每专家字节 = 1,114,112 */
#define D_MODEL          4096                     /* DeepSeek V4 Flash 隐维(z^L dout/V8 行宽同源) */
#define SZ_D             ((uint64_t)D_MODEL * 8 * 34)   /* dql D 段每专家字节 = 1,114,112 */
#define DQL_HDR          35104                    /* dql_L*.bin 头(实测 dql_L00); 只作文档常量,
                                                   * 真 D 偏移一律走 --down-offsets 逐层文件 */
#define DOWN_LAYER_BYTES ((uint64_t)NEXP * SZ_D)  /* 285,212,672 */
#define RB_MINCNT        8                        /* 与 ds4quant_run RB_MINCNT 同步 */
#define REC_HDR          DS4_AMP_REC_HDR   /* 116: 单一定义在 src/common/ds4_amp_fmt.h */

#define CHUNK_SKEL (1u << 26)   /* .py: src.read(min(1 << 26, left)) */
#define CHUNK_STRM (1u << 24)   /* .py: f.read(min(1 << 24, left)) / p.stdout.read(1 << 24) */

/* ---------------- 基础工具 ---------------- */

/* .py 里的 assert/异常一律终止进程; 这里统一 exit(1), 消息带 Python 异常名前缀便于 grep 对拍 */
static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr); exit(1);
}
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("MemoryError: malloc %zu", n); return p; }
static void *xrealloc(void *p, size_t n) { void *q = realloc(p, n ? n : 1); if (!q) die("MemoryError: realloc %zu", n); return q; }
static char *xstrdup(const char *s) { size_t n = strlen(s) + 1; char *d = xmalloc(n); memcpy(d, s, n); return d; }

/* 饱和加: .py 用大整数, 溢出不可能; C 侧饱和后一样必然触发上界 break, 判定等价 */
static uint64_t sat_add(uint64_t a, uint64_t b) { uint64_t s = a + b; return s < a ? UINT64_MAX : s; }
static uint64_t align_up(uint64_t v) { return (v + ALIGN - 1) / ALIGN * ALIGN; }

typedef struct { uint8_t *p; size_t n, cap; } buf_t;
static void buf_add(buf_t *b, const void *d, size_t n) {
    if (b->n + n > b->cap) { b->cap = (b->n + n) * 2 + 64; b->p = xrealloc(b->p, b->cap); }
    if (n) memcpy(b->p + b->n, d, n);
    b->n += n;
}
static void buf_u32(buf_t *b, uint32_t v) { buf_add(b, &v, 4); }
static void buf_u64(buf_t *b, uint64_t v) { buf_add(b, &v, 8); }

static uint64_t fsize(const char *p) {
    struct stat st;
    if (stat(p, &st)) die("FileNotFoundError: %s: %s", p, strerror(errno));
    return (uint64_t)st.st_size;
}
static int fexists(const char *p) { return access(p, F_OK) == 0; }

static uint8_t *read_all(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb");
    if (!f) die("FileNotFoundError: %s: %s", p, strerror(errno));
    fseeko(f, 0, SEEK_END); off_t sz = ftello(f); fseeko(f, 0, SEEK_SET);
    uint8_t *b = xmalloc((size_t)sz + 1);
    if (sz && fread(b, 1, (size_t)sz, f) != (size_t)sz) die("IOError: 短读 %s", p);
    fclose(f); *len = (size_t)sz; return b;
}

/* float16 → float32(位精确, 含次正规/inf/nan 尾数保位; 与 numpy .astype(f32) 同) */
static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, o;
    if (e == 31) o = s | 0x7F800000u | (m << 13);
    else if (e == 0) {
        if (!m) o = s;
        else { int sh = 0; while (!(m & 1024)) { m <<= 1; sh++; }
               m &= 1023; o = s | ((uint32_t)(113 - sh) << 23) | (m << 13); }
    } else o = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &o, 4); return f;
}

/* Python repr(float): 最短往返表示 + 整数值补 ".0"; 指数形式条件 decpt<=-4 || decpt>16。
 * 只用于 [route-bias] 那行 α 打印 —— 不进文件字节, 但对拍 stdout 时要一致。 */
static const char *py_float_repr(double v) {
    static char out[64], e[64];
    if (isnan(v)) { strcpy(out, "nan"); return out; }
    if (isinf(v)) { strcpy(out, v > 0 ? "inf" : "-inf"); return out; }
    int p;
    for (p = 0; p <= 17; p++) { snprintf(e, sizeof e, "%.*e", p, v); if (strtod(e, NULL) == v) break; }
    const char *s = e; int neg = 0;
    if (*s == '-') { neg = 1; s++; }
    char dig[40]; int nd = 0;
    dig[nd++] = *s++;
    if (*s == '.') { s++; while (*s != 'e' && *s != 'E') dig[nd++] = *s++; }
    while (*s != 'e' && *s != 'E') s++;
    int decpt = atoi(s + 1) + 1;
    while (nd > 1 && dig[nd - 1] == '0') nd--;
    dig[nd] = 0;
    char *o = out;
    if (neg) *o++ = '-';
    if (decpt <= -4 || decpt > 16) {
        *o++ = dig[0];
        if (nd > 1) { *o++ = '.'; memcpy(o, dig + 1, nd - 1); o += nd - 1; }
        o += sprintf(o, "e%+03d", decpt - 1);
    } else if (decpt <= 0) {
        *o++ = '0'; *o++ = '.';
        for (int i = 0; i < -decpt; i++) *o++ = '0';
        memcpy(o, dig, nd); o += nd;
    } else if (decpt >= nd) {
        memcpy(o, dig, nd); o += nd;
        for (int i = 0; i < decpt - nd; i++) *o++ = '0';
        *o++ = '.'; *o++ = '0';
    } else {
        memcpy(o, dig, decpt); o += decpt; *o++ = '.';
        memcpy(o, dig + decpt, nd - decpt); o += nd - decpt;
    }
    *o = 0; return out;
}

/* nm.split(".")[idx] */
static const char *name_field(const char *nm, int idx, char *out, size_t osz) {
    const char *s = nm;
    for (int k = 0; k < idx; k++) {
        const char *d = strchr(s, '.');
        if (!d) die("IndexError: list index out of range: %s [%d]", nm, idx);
        s = d + 1;
    }
    const char *e2 = strchr(s, '.');
    size_t n = e2 ? (size_t)(e2 - s) : strlen(s);
    if (n >= osz) n = osz - 1;
    memcpy(out, s, n); out[n] = 0; return out;
}
static int name_field_int(const char *nm, int idx) { char t[64]; name_field(nm, idx, t, sizeof t); return atoi(t); }

static int str_endswith(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return m <= n && !memcmp(s + n - m, suf, m);
}
static int str_startswith(const char *s, const char *pre) { return !strncmp(s, pre, strlen(pre)); }

/* dict[int]->int64: 保 .py 的 dict 语义(同键覆盖 / len() 计唯一键 / 缺键 KeyError) */
typedef struct { long long *k, *v; size_t n, cap; } imap_t;
static void imap_set(imap_t *m, long long k, long long v) {
    for (size_t i = 0; i < m->n; i++) if (m->k[i] == k) { m->v[i] = v; return; }
    if (m->n == m->cap) { m->cap = m->cap * 2 + 16; m->k = xrealloc(m->k, m->cap * sizeof *m->k); m->v = xrealloc(m->v, m->cap * sizeof *m->v); }
    m->k[m->n] = k; m->v[m->n] = v; m->n++;
}
static long long imap_get(const imap_t *m, long long k) {
    for (size_t i = 0; i < m->n; i++) if (m->k[i] == k) return m->v[i];
    die("KeyError: %lld", k); return 0;
}

/* ---------------- 张量/条目 ---------------- */

enum { K_SKEL = 0, K_OPT, K_GUD, K_STUB, K_DOWN, K_BLOB, K_SRC };

typedef struct {
    char    *name;
    uint64_t*ne;
    uint32_t nd;
    uint32_t type;
    uint64_t off;        /* 解析时=源数据段内偏移; write_gguf 内重算=目的偏移 */
    uint64_t bytes;
    int      kind;
    uint64_t src_off;
    uint64_t skip;
    int      L;
    int      last;
    uint8_t *data;       /* kind=K_OPT 的内嵌载荷 */
    uint32_t idx;        /* sorted() 稳定性: 同 off 时保持原序 */
} ent_t;

typedef struct { ent_t *v; size_t n, cap; } elist_t;
static ent_t *el_push(elist_t *l) {
    if (l->n == l->cap) { l->cap = l->cap * 2 + 64; l->v = xrealloc(l->v, l->cap * sizeof *l->v); }
    ent_t *e = &l->v[l->n]; memset(e, 0, sizeof *e); e->idx = (uint32_t)l->n; l->n++; return e;
}
static ent_t *el_push1(elist_t *l, const char *name, uint64_t ne0, uint32_t type, uint64_t bytes, int kind) {
    ent_t *e = el_push(l);
    e->name = xstrdup(name); e->nd = 1; e->ne = xmalloc(8); e->ne[0] = ne0;
    e->type = type; e->bytes = bytes; e->kind = kind; return e;
}

/* ---------------- GGUF 读 ---------------- */

static void rdn(FILE *f, void *dst, size_t n) { if (n && fread(dst, 1, n, f) != n) die("AssertionError: 短读 %zu B", n); }
static uint32_t rdu32(FILE *f) { uint32_t v; rdn(f, &v, 4); return v; }
static uint64_t rdu64(FILE *f) { uint64_t v; rdn(f, &v, 8); return v; }
static char *rdstr(FILE *f) {
    uint64_t n = rdu64(f);
    char *s = xmalloc((size_t)n + 1);
    rdn(f, s, (size_t)n); s[n] = 0; return s;
}
static void skipv(FILE *f, uint32_t t) {
    static const int sz[13] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
    if (t == 8) { free(rdstr(f)); }
    else if (t == 9) {
        uint32_t et = rdu32(f); uint64_t n = rdu64(f);
        for (uint64_t i = 0; i < n; i++) skipv(f, et);
    } else {
        uint8_t tmp[8];
        if (t > 12 || sz[t] == 0) die("KeyError: %u", t);
        rdn(f, tmp, (size_t)sz[t]);
    }
}

static void parse_header(FILE *f, uint64_t *n_kv_o, buf_t *kv_raw, elist_t *tens, uint64_t *data0_o) {
    uint32_t magic = rdu32(f), ver = rdu32(f);
    uint64_t n_t = rdu64(f), n_kv = rdu64(f);
    if (!(magic == 0x46554747u && ver == 3)) die("AssertionError: 非 GGUF v3(magic=%08x ver=%u)", magic, ver);
    off_t kv_start = ftello(f);
    for (uint64_t i = 0; i < n_kv; i++) { free(rdstr(f)); uint32_t t = rdu32(f); skipv(f, t); }
    off_t kv_end = ftello(f);
    fseeko(f, kv_start, SEEK_SET);
    kv_raw->n = 0;
    { size_t n = (size_t)(kv_end - kv_start); uint8_t *tmp = xmalloc(n); rdn(f, tmp, n); buf_add(kv_raw, tmp, n); free(tmp); }
    for (uint64_t i = 0; i < n_t; i++) {
        ent_t *e = el_push(tens);
        e->name = rdstr(f);
        e->nd = rdu32(f);
        e->ne = xmalloc((size_t)e->nd * 8);
        rdn(f, e->ne, (size_t)e->nd * 8);
        e->type = rdu32(f);
        e->off = rdu64(f);
    }
    *data0_o = align_up((uint64_t)ftello(f));
    *n_kv_o = n_kv;
}

static int cmp_off(const void *a, const void *b) {
    const ent_t *const *x = a, *const *y = b;
    if ((*x)->off < (*y)->off) return -1;
    if ((*x)->off > (*y)->off) return 1;
    return (*x)->idx < (*y)->idx ? -1 : (*x)->idx > (*y)->idx;   /* sorted() 稳定 */
}
static void sizes_by_offset(elist_t *tens, uint64_t fsz, uint64_t data0) {
    size_t n = tens->n;
    ent_t **ord = xmalloc(n * sizeof *ord);
    for (size_t i = 0; i < n; i++) ord[i] = &tens->v[i];
    qsort(ord, n, sizeof *ord, cmp_off);
    for (size_t i = 0; i < n; i++) {
        uint64_t end = i + 1 < n ? ord[i + 1]->off : fsz - data0;
        ord[i]->bytes = end - ord[i]->off;
    }
    free(ord);
}

static void ser_info(buf_t *b, const ent_t *t) {
    size_t nl = strlen(t->name);
    buf_u64(b, nl); buf_add(b, t->name, nl);
    buf_u32(b, t->nd); buf_add(b, t->ne, (size_t)t->nd * 8);
    buf_u32(b, t->type); buf_u64(b, t->off);
}

static int is_dropped(const char *nm, int skeleton_mode) {
    if (str_endswith(nm, "ffn_exps_vq.blob")) return 1;
    if (strstr(nm, ".opt_")) return 1;
    if (skeleton_mode && str_endswith(nm, "ffn_down_exps.weight")) return 1;
    /* ★gate/up 路由专家也必须丢(2026-08-02 修)★
     * 原来没这两条是因为骨架源一直是【冠军 VQ 合一 GGUF】—— 那种文件里 gate/up exps 早已
     * 不存在(专家字节全在 blk.L.ffn_exps_vq.blob 里)。而自产骨架的源是 deepseek4-quantize
     * --experts-hole 产出的留洞文件, 它的张量表沿用 published 模版, gate/up exps **在表里**
     * (只是数据是洞)。不丢的话 extract_skeleton 会把这些洞当骨架张量写出来:
     * 实测文件从 8.2 GiB 撑到 17.7+ GiB, 两次把磁盘写穿(OSError:28)。 */
    if (skeleton_mode && (str_endswith(nm, "ffn_gate_exps.weight") ||
                          str_endswith(nm, "ffn_up_exps.weight"))) return 1;
    return 0;
}

/* entries 顺序即数据序; writer(out, e) 负责写数据。 */
static uint64_t write_gguf(const char *out_p, const buf_t *kv_raw, uint64_t n_kv,
                           elist_t *ents, void (*writer)(FILE *, ent_t *)) {
    uint64_t off = 0;
    for (size_t i = 0; i < ents->n; i++) { ents->v[i].off = off; off += align_up(ents->v[i].bytes); }
    buf_t hdr = { 0 }, info = { 0 };
    buf_u32(&hdr, 0x46554747u); buf_u32(&hdr, 3);
    buf_u64(&hdr, (uint64_t)ents->n); buf_u64(&hdr, n_kv);
    buf_add(&hdr, kv_raw->p, kv_raw->n);
    for (size_t i = 0; i < ents->n; i++) ser_info(&info, &ents->v[i]);
    uint64_t data0 = align_up(hdr.n + info.n);
    FILE *out = fopen(out_p, "wb");
    if (!out) die("IOError: 打不开 %s: %s", out_p, strerror(errno));
    if (fwrite(hdr.p, 1, hdr.n, out) != hdr.n) die("IOError: 写头失败");
    if (fwrite(info.p, 1, info.n, out) != info.n) die("IOError: 写张量表失败");
    { size_t pad = (size_t)(data0 - hdr.n - info.n); uint8_t z[ALIGN] = { 0 };
      if (pad && fwrite(z, 1, pad, out) != pad) die("IOError: 写对齐失败"); }
    for (size_t i = 0; i < ents->n; i++) {
        if (fseeko(out, (off_t)(data0 + ents->v[i].off), SEEK_SET)) die("IOError: seek 失败");
        writer(out, &ents->v[i]);
    }
    if (fclose(out)) die("IOError: 关闭 %s 失败: %s", out_p, strerror(errno));
    free(hdr.p); free(info.p);
    return data0;
}

/* ---------------- 命令行参数(dest 名与 .py argparse 一一对应) ---------------- */

typedef struct {
    int extract_skeleton, extract_blobs, merge, no_down, consume, gud;
    const char *out_dir, *base, *skeleton, *blob_sizes, *down_offsets, *route_bias;
    double route_alpha;
    const char *blob_layers, *dql_host, *dql_dir, *out;
} args_t;
static args_t A;

/* 写盘上下文(.py 里是闭包捕获的 src/data0/rb/doff) */
static FILE    *g_src;
static uint64_t g_src_data0;
static double   g_rb[N_LAYER][NEXP];   /* f64: .py 的 row 是 f64, 只在烘焙时落一次 f32 */
static int      g_rb_has[N_LAYER];
static imap_t   g_doff;
static uint8_t *g_chunk_skel, *g_chunk_strm;

static uint8_t *chunk_skel(void) { if (!g_chunk_skel) g_chunk_skel = xmalloc(CHUNK_SKEL); return g_chunk_skel; }
static uint8_t *chunk_strm(void) { if (!g_chunk_strm) g_chunk_strm = xmalloc(CHUNK_STRM); return g_chunk_strm; }

/* ---------------- --extract-blobs ---------------- */

/* ★反修v4逆向(2026-08-10): 从合一 GGUF 反抽 blk.L.ffn_exps_vq.blob → dql_vq_LXX.bin
 * (--consume 吃掉层件后的链式反修回收路; 字节=blob 原样, vq_slot/zlayer 直读)。 */
static void extract_blobs(void) {
    FILE *f = fopen(A.base, "rb");
    if (!f) die("FileNotFoundError: %s: %s", A.base, strerror(errno));
    uint64_t n_kv, data0; buf_t kv_raw = { 0 }; elist_t tens = { 0 };
    parse_header(f, &n_kv, &kv_raw, &tens, &data0);
    uint64_t fsz = fsize(A.base);
    sizes_by_offset(&tens, fsz, data0);
    int n = 0;
    for (size_t i = 0; i < tens.n; i++) {
        ent_t *t = &tens.v[i];
        if (!str_endswith(t->name, "ffn_exps_vq.blob")) continue;
        int L = name_field_int(t->name, 1);
        char outp[2048]; snprintf(outp, sizeof outp, "%s/dql_vq_L%02d.bin", A.out_dir, L);
        fseeko(f, (off_t)(data0 + t->off), SEEK_SET);
        uint64_t remain = t->bytes;
        FILE *o = fopen(outp, "wb");
        if (!o) die("IOError: 打不开 %s: %s", outp, strerror(errno));
        uint8_t *buf = chunk_strm();
        while (remain > 0) {
            size_t want = remain < CHUNK_STRM ? (size_t)remain : CHUNK_STRM;
            size_t got = fread(buf, 1, want, f);
            if (!got) break;
            if (fwrite(buf, 1, got, o) != got) die("IOError: 写 %s 失败", outp);
            remain -= got;
        }
        fclose(o);
        n++;
        printf("L%02d blob %.2fGB -> %s\n", L, t->bytes / 1e9, outp); fflush(stdout);
    }
    printf("extract-blobs done: %d\n", n); fflush(stdout);
    fclose(f);
}

/* ---------------- --extract-skeleton ---------------- */

static void skel_writer(FILE *out, ent_t *e) {
    fseeko(g_src, (off_t)(g_src_data0 + e->src_off), SEEK_SET);
    uint64_t left = e->bytes;
    uint8_t *buf = chunk_skel();
    while (left) {
        size_t want = left < CHUNK_SKEL ? (size_t)left : CHUNK_SKEL;
        size_t got = fread(buf, 1, want, g_src);
        if (!got) die("IOError: 骨架源短读(剩 %llu B)", (unsigned long long)left);
        if (fwrite(buf, 1, got, out) != got) die("IOError: 写骨架失败");
        left -= got;
    }
}

static void extract_skeleton(void) {
    uint64_t fsz = fsize(A.base);
    FILE *f = fopen(A.base, "rb");
    if (!f) die("FileNotFoundError: %s: %s", A.base, strerror(errno));
    uint64_t n_kv, data0; buf_t kv_raw = { 0 }; elist_t tens = { 0 };
    parse_header(f, &n_kv, &kv_raw, &tens, &data0);
    fclose(f);
    sizes_by_offset(&tens, fsz, data0);
    elist_t ents = { 0 };
    for (size_t i = 0; i < tens.n; i++) {
        ent_t *t = &tens.v[i];
        if (is_dropped(t->name, 1)) continue;
        ent_t *e = el_push(&ents);
        e->name = t->name; e->nd = t->nd; e->ne = t->ne; e->type = t->type;
        e->bytes = t->bytes; e->src_off = t->off; e->kind = K_SRC;
    }
    g_src = fopen(A.base, "rb");
    if (!g_src) die("FileNotFoundError: %s: %s", A.base, strerror(errno));
    g_src_data0 = data0;
    write_gguf(A.out, &kv_raw, n_kv, &ents, skel_writer);
    printf("[skeleton] %s %.2f GiB (%zu 张量)\n", A.out, fsize(A.out) / (double)(1ULL << 30), ents.n);
    fflush(stdout);
}

/* ---------------- 路由偏置(RBIA → 每层 256×f32 Δb·α) ---------------- */

static void load_route_bias(const char *path, double alpha) {
    FILE *f = fopen(path, "rb");
    if (!f) die("FileNotFoundError: %s: %s", path, strerror(errno));
    uint32_t hd[4]; rdn(f, hd, 16);
    if (!(hd[0] == 0x41494252u && hd[1] == N_LAYER && hd[2] == NEXP)) die("AssertionError: RBIA 头不对");
    float *acc = xmalloc((size_t)N_LAYER * NEXP * 4);
    uint32_t *cnt = xmalloc((size_t)N_LAYER * NEXP * 4);
    rdn(f, acc, (size_t)N_LAYER * NEXP * 4);
    rdn(f, cnt, (size_t)N_LAYER * NEXP * 4);
    fclose(f);
    long long armed = 0; int nlayer = 0;
    for (int L = 0; L < N_LAYER; L++) {
        /* .py: row 是 f64(alpha 与 acc 都升 f64 再乘); 烘焙时才落回 f32 —— 舍入点照抄, 见
         * merge_writer 里 (double)v + (double)row 后一次 f32 舍入。 */
        int any = 0, na = 0;
        for (int e = 0; e < NEXP; e++) {
            g_rb[L][e] = cnt[L * NEXP + e] >= RB_MINCNT ? alpha * (double)acc[L * NEXP + e] : 0.0;
            if (g_rb[L][e] != 0.0) { any = 1; na++; }
        }
        if (any) { g_rb_has[L] = 1; nlayer++; armed += na; }
    }
    free(acc); free(cnt);
    printf("[route-bias] α=%s mincnt=%d 武装槽=%lld 层=%d\n", py_float_repr(alpha), RB_MINCNT, armed, nlayer);
    fflush(stdout);
}

