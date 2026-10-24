/* ---------------- --merge ---------------- */

static ent_t *el_push3(elist_t *l, const char *name, uint64_t a0, uint64_t a1, uint64_t a2,
                       uint32_t type, uint64_t bytes, int kind) {
    ent_t *e = el_push(l);
    e->name = xstrdup(name); e->nd = 3; e->ne = xmalloc(24);
    e->ne[0] = a0; e->ne[1] = a1; e->ne[2] = a2;
    e->type = type; e->bytes = bytes; e->kind = kind; return e;
}

/* "L 值" 两列文本 → dict(与 .py 的 line.split() + int() 同严格度) */
static void read_int_pairs(const char *path, imap_t *m) {
    FILE *f = fopen(path, "r");
    if (!f) die("FileNotFoundError: %s: %s", path, strerror(errno));
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        char a1[256], a2[256], extra[256];
        int nf = sscanf(line, "%255s %255s %255s", a1, a2, extra);
        if (nf > 2) die("ValueError: too many values to unpack (expected 2)");
        if (nf != 2) die("ValueError: not enough values to unpack (expected 2, got %d)", nf < 0 ? 0 : nf);
        char *end1, *end2;
        long long k = strtoll(a1, &end1, 10), v = strtoll(a2, &end2, 10);
        if (*end1 || *end2) die("ValueError: invalid literal for int(): '%s' / '%s'", a1, a2);
        imap_set(m, k, v);
    }
    fclose(f);
}

static void merge_writer(FILE *out, ent_t *e) {
    char pth[2048], fld[64];
    if (e->kind == K_OPT) {
        if (fwrite(e->data, 1, e->bytes, out) != e->bytes) die("IOError: 写 opt 失败");
        return;
    }
    if (e->kind == K_SKEL) {
        fseeko(g_src, (off_t)(g_src_data0 + e->src_off), SEEK_SET);
        const char *nm = e->name;
        if (str_startswith(nm, "blk.") && str_endswith(nm, ".exp_probs_b.bias")) {
            int L = name_field_int(nm, 1);
            uint8_t *raw = xmalloc((size_t)e->bytes);
            if (fread(raw, 1, (size_t)e->bytes, g_src) != e->bytes) die("IOError: 骨架短读 %s", nm);
            if (L >= 0 && L < N_LAYER && g_rb_has[L]) {
                if (e->bytes < (uint64_t)NEXP * 4) die("struct.error: unpack requires a buffer of 1024 bytes");
                for (int x = 0; x < NEXP; x++) {
                    float v; memcpy(&v, raw + 4 * x, 4);
                    v = (float)((double)v + g_rb[L][x]);   /* f64 加, 一次 f32 舍入(照抄 .py) */
                    memcpy(raw + 4 * x, &v, 4);
                }
                printf("[merge] gate.bias L%d 烘焙 ✓\n", L); fflush(stdout);
            }
            if (fwrite(raw, 1, (size_t)e->bytes, out) != e->bytes) die("IOError: 写 bias 失败");
            free(raw); return;
        }
        uint64_t left = e->bytes;
        uint8_t *buf = chunk_skel();
        while (left) {
            size_t want = left < CHUNK_SKEL ? (size_t)left : CHUNK_SKEL;
            size_t got = fread(buf, 1, want, g_src);
            if (!got) die("IOError: 骨架源短读(剩 %llu B)", (unsigned long long)left);
            if (fwrite(buf, 1, got, out) != got) die("IOError: 写骨架失败");
            left -= got;
        }
        return;
    }
    if (e->kind == K_GUD) {
        snprintf(pth, sizeof pth, "%s/dql_L%02d.bin", A.dql_dir, e->L);
        ssh_stream(A.dql_host, pth, e->bytes, out, e->skip);
        int consumed = A.consume && e->last;
        if (consumed) { fflush(out); if (remove(pth)) die("OSError: 删 %s 失败: %s", pth, strerror(errno)); }
        name_field(e->name, 2, fld, sizeof fld);
        printf("[merge] %s L%d ✓%s\n", fld, e->L, consumed ? " (dql已消费)" : ""); fflush(stdout);
        return;
    }
    if (e->kind == K_STUB) {
        uint32_t magic = 0x4C565144u;   /* DQVL 魔数 + 全零槽表 */
        if (fwrite(&magic, 1, 4, out) != 4) die("IOError: 写存根失败");
        uint64_t left = e->bytes - 4;
        uint8_t z[4096]; memset(z, 0, sizeof z);
        while (left) { size_t w = left < sizeof z ? (size_t)left : sizeof z;
                       if (fwrite(z, 1, w, out) != w) die("IOError: 写存根失败"); left -= w; }
        return;
    }
    if (e->kind == K_DOWN) {
        snprintf(pth, sizeof pth, "%s/dql_L%02d.bin", A.dql_dir, e->L);
        ssh_stream(A.dql_host, pth, DOWN_LAYER_BYTES, out, (uint64_t)imap_get(&g_doff, e->L));
        /* ★--consume 也吃 dql(2026-08-03 冠军复刻盘账)★: D 段是 dql 里唯一被合并读的段,
         * 读完该层 dql 即死重(反修/回放此时已完成)。54G 输出必须靠这 ~13G 边合并边释放。 */
        if (A.consume) { fflush(out); if (remove(pth)) die("OSError: 删 %s 失败: %s", pth, strerror(errno)); }
        printf("[merge] down L%d ✓%s\n", e->L, A.consume ? " (dql已消费)" : ""); fflush(stdout);
        return;
    }
    /* K_BLOB */
    snprintf(pth, sizeof pth, "%s/dql_vq_L%02d.bin", A.dql_dir, e->L);
    ssh_stream(A.dql_host, pth, e->bytes, out, 0);
    /* ★--consume(R30)★: 该层 blob 已完整写进输出 → 立删源文件。M1 free 50G 放不下
     * 层27.8+骨架8.2+输出36 三者同存; 消费式把峰值净增压到 ~9G(输出36−层27.8)。
     * 代价: 合并中途失败已删的层要重量化 —— 只在盘账过不去时用, 由调用方显式开。 */
    if (A.consume) {
        if (!host_is_local(A.dql_host)) { fprintf(stderr, "--consume 只允许本地 dql-host(不做远程删)\n"); exit(1); }
        fflush(out); if (remove(pth)) die("OSError: 删 %s 失败: %s", pth, strerror(errno));
    }
    printf("[merge] blob L%d %.0fMiB ✓%s\n", e->L, e->bytes / (double)(1ULL << 20),
           A.consume ? " (源已消费)" : ""); fflush(stdout);
}

static void merge(void) {
    uint64_t fsz = fsize(A.skeleton);
    FILE *f = fopen(A.skeleton, "rb");
    if (!f) die("FileNotFoundError: %s: %s", A.skeleton, strerror(errno));
    uint64_t n_kv, data0; buf_t kv_raw = { 0 }; elist_t tens = { 0 };
    parse_header(f, &n_kv, &kv_raw, &tens, &data0);
    fclose(f);
    sizes_by_offset(&tens, fsz, data0);
    if (A.route_bias) load_route_bias(A.route_bias, A.route_alpha);
    imap_t bsz = { 0 };
    read_int_pairs(A.blob_sizes, &bsz);
    if (bsz.n != N_LAYER) die("AssertionError: blob 尺寸表 %zu/43", bsz.n);
    /* D 段权威偏移(dql_down_offset.py 记录链解析): 固定 DQL_HDR=35104 是错误假设 —
     * 1bit 记录载荷@128, 实测全层 570,425,472; 仍按每层文件读取防未来漂移。 */
    if (!A.no_down) {
        read_int_pairs(A.down_offsets, &g_doff);
        if (g_doff.n != N_LAYER) die("AssertionError: down 偏移表 %zu/43", g_doff.n);
    }
    elist_t ents = { 0 };
    for (size_t i = 0; i < tens.n; i++) {
        ent_t *t = &tens.v[i], *e = el_push(&ents);
        e->name = t->name; e->nd = t->nd; e->ne = t->ne; e->type = t->type;
        e->bytes = t->bytes; e->kind = K_SKEL; e->src_off = t->off;
    }
    /* opt 内嵌(本地 dql_dir 读; --consume 前于 ents 构造期完成全部读取) */
    int n_opt = 0;
    if (host_is_local(A.dql_host))
        for (int L = 0; L < N_LAYER; L++) n_opt += build_opt_tensors(A.dql_dir, L, &ents);
    if (n_opt) {
        kv_append_bool(&kv_raw, &n_kv, "ds4.zchain.present", 1);
        printf("[merge] opt 内嵌 %d 张量 + ds4.zchain.present=true\n", n_opt);
    } else {
        printf("[merge] ★opt 张量零内嵌(侧车缺?)— 反修效果不在此模型, 需人工确认★\n");
    }
    fflush(stdout);
    /* ★--gud(2026-08-03 code2b 42G 复刻)★: G/U/D 三段 signref 全搬(1bit 记录载荷布局
     * [G 256×szG][U 256×szG][D 256×szD], dql_down_offset 的 D 偏移回推 G/U)。blob(热侧车)
     * 不进 GGUF — 运行时外挂, 42.468G 账目与 code2b 一字对齐。 */
    if (A.gud) {
        uint64_t sz_g = 2048ULL * (4096 / 256) * 34;   /* = 全局 SZ_G, .py 里同名局部量 */
        char nm[128];
        for (int L = 0; L < N_LAYER; L++) {
            uint64_t g_off = (uint64_t)(imap_get(&g_doff, L) - 2LL * NEXP * (long long)sz_g);
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", L);
            ent_t *e = el_push3(&ents, nm, 4096, 2048, 256, 40, NEXP * sz_g, K_GUD);
            e->L = L; e->skip = g_off;
            snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", L);
            e = el_push3(&ents, nm, 4096, 2048, 256, 40, NEXP * sz_g, K_GUD);
            e->L = L; e->skip = g_off + NEXP * sz_g;
            snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", L);
            e = el_push3(&ents, nm, 2048, 4096, 256, 40, DOWN_LAYER_BYTES, K_GUD);
            e->L = L; e->skip = (uint64_t)imap_get(&g_doff, L); e->last = 1;
        }
    }
    int has_blr = 0, blo = 0, bhi = 0;
    if (A.blob_layers[0]) {
        char lo[64], hi[64];
        const char *c = strchr(A.blob_layers, ':');
        if (!c || strchr(c + 1, ':')) die("ValueError: too many/few values to unpack: %s", A.blob_layers);
        snprintf(lo, sizeof lo, "%.*s", (int)(c - A.blob_layers), A.blob_layers);
        snprintf(hi, sizeof hi, "%s", c + 1);
        blo = atoi(lo); bhi = atoi(hi); has_blr = 1;
        printf("[merge] 切片模式: 只带 L%d..L%d 的 blob(%d 层)\n", blo, bhi, bhi - blo + 1);
        fflush(stdout);
    }
    char nm[128];
    for (int L = 0; L < (A.gud ? 0 : N_LAYER); L++) {
        /* --no-down(R28): 冷 w2 已编在 blob 的 which=2 槽里, base down 是纯死重
         * (43 × 0.2656 = 11.42 GiB)。省掉它才是 28 GiB 目标能成立的原因; 引擎侧
         * 由 routed_down_shadow() 合成影子张量顶上(ds4.c)。冠军 vq4bf 那种冷 w2 从
         * base go1b 读的配方不能带这个开关, 否则冷专家权重直接没了。 */
        if (has_blr && !(blo <= L && L <= bhi)) {
            /* 存根 blob(6160B 零头): 装载器认层过"required tensor"检查, 槽位全 0 →
             * 前向若误触即报"缺 vq 槽"硬错; 分布式非本机层永不前向 ⇒ 纯占位。 */
            snprintf(nm, sizeof nm, "blk.%d.ffn_exps_vq.blob", L);
            el_push1(&ents, nm, 6160, 42, 6160, K_STUB)->L = L;
            continue;
        }
        if (!A.no_down) {
            snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", L);
            el_push3(&ents, nm, 2048, 4096, 256, 40, DOWN_LAYER_BYTES, K_DOWN)->L = L;
        }
        snprintf(nm, sizeof nm, "blk.%d.ffn_exps_vq.blob", L);
        uint64_t b = (uint64_t)imap_get(&bsz, L);
        el_push1(&ents, nm, b, 42, b, K_BLOB)->L = L;
    }
    g_src = fopen(A.skeleton, "rb");
    if (!g_src) die("FileNotFoundError: %s: %s", A.skeleton, strerror(errno));
    g_src_data0 = data0;
    write_gguf(A.out, &kv_raw, n_kv, &ents, merge_writer);
    printf("[merge] 完成 %s %.2f GiB\n", A.out, fsize(A.out) / (double)(1ULL << 30));
    fflush(stdout);
}

/* ---------------- main ---------------- */

static const char *USAGE =
    "usage: vq_merge_v4 [-h] [--extract-skeleton] [--extract-blobs] [--out-dir OUT_DIR]\n"
    "                   [--merge] [--base BASE] [--skeleton SKELETON]\n"
    "                   [--blob-sizes BLOB_SIZES] [--down-offsets DOWN_OFFSETS] [--no-down]\n"
    "                   [--route-bias ROUTE_BIAS] [--route-alpha ROUTE_ALPHA] [--consume]\n"
    "                   [--gud] [--blob-layers BLOB_LAYERS] [--dql-host DQL_HOST]\n"
    "                   [--dql-dir DQL_DIR] --out OUT\n";

static void argerr(const char *msg) { fputs(USAGE, stderr); fprintf(stderr, "vq_merge_v4: error: %s\n", msg); exit(2); }

/* 前缀匹配 --name / --name=VALUE */
static int arg_is(const char *a, const char *name, const char **val) {
    size_t n = strlen(name);
    if (strncmp(a, name, n)) return 0;
    if (a[n] == 0) { *val = NULL; return 1; }
    if (a[n] == '=') { *val = a + n + 1; return 1; }
    return 0;
}

int main(int argc, char **argv) {
    A.out_dir = "."; A.route_alpha = 2.5; A.blob_layers = "";
    /* 层件主机/目录没有默认值(2026-10-07 前写死了一台 Mac 的局域网 IP 和家目录): --merge 必须显式给, 见下方检查 */
    A.dql_host = NULL; A.dql_dir = NULL;

    for (int i = 1; i < argc; i++) {
        const char *s = argv[i], *v = NULL;
        char e[256];
#define FLAG(nm, fld) if (arg_is(s, nm, &v)) { if (v) { snprintf(e, sizeof e, "ignored explicit argument '%s'", v); argerr(e); } A.fld = 1; continue; }
#define OPTV(nm, fld, conv) if (arg_is(s, nm, &v)) { if (!v) { if (i + 1 >= argc) { snprintf(e, sizeof e, "argument %s: expected one argument", nm); argerr(e); } v = argv[++i]; } conv; continue; }
        if (!strcmp(s, "-h") || !strcmp(s, "--help")) { fputs(USAGE, stdout); return 0; }
        FLAG("--extract-skeleton", extract_skeleton)
        FLAG("--extract-blobs", extract_blobs)
        FLAG("--merge", merge)
        FLAG("--no-down", no_down)
        FLAG("--consume", consume)
        FLAG("--gud", gud)
        OPTV("--out-dir", out_dir, A.out_dir = v)
        OPTV("--base", base, A.base = v)
        OPTV("--skeleton", skeleton, A.skeleton = v)
        OPTV("--blob-sizes", blob_sizes, A.blob_sizes = v)
        OPTV("--down-offsets", down_offsets, A.down_offsets = v)
        OPTV("--route-bias", route_bias, A.route_bias = v)
        OPTV("--blob-layers", blob_layers, A.blob_layers = v)
        OPTV("--dql-host", dql_host, A.dql_host = v)
        OPTV("--dql-dir", dql_dir, A.dql_dir = v)
        OPTV("--out", out, A.out = v)
        OPTV("--route-alpha", route_alpha, {
            char *end; A.route_alpha = strtod(v, &end);
            if (end == v || *end) { snprintf(e, sizeof e, "argument --route-alpha: invalid float value: '%s'", v); argerr(e); }
        })
#undef FLAG
#undef OPTV
        snprintf(e, sizeof e, "unrecognized arguments: %s", s); argerr(e);
    }
    if (!A.out) argerr("the following arguments are required: --out");

    if (A.extract_blobs) { extract_blobs(); return 0; }
    if (A.extract_skeleton) extract_skeleton();
    else if (A.merge) {
        if (!A.dql_host || !A.dql_dir) die("AssertionError: --merge 需 --dql-host 与 --dql-dir(本机用 --dql-host localhost)");
        if (!A.blob_sizes) die("AssertionError: --merge 需 --blob-sizes");
        if (!(A.down_offsets || A.no_down)) die("AssertionError: --merge 需 --down-offsets(或 --no-down)");
        merge();
    } else { fprintf(stderr, "需 --extract-skeleton 或 --merge\n"); return 1; }
    return 0;
}
