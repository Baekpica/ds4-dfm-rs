static gguf_file load_gguf_metadata(const char *path) {
    gguf_file g = {0};
    g.path = xstrdup(path);
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) || memcmp(magic, "GGUF", 4) != 0) {
        die("bad GGUF template");
    }
    g.version = read_u32_le_fp(fp, "GGUF version");
    g.n_tensors = read_u64_le_fp(fp, "GGUF tensor count");
    g.n_kv = read_u64_le_fp(fp, "GGUF KV count");
    g.alignment = DS4_GGUF_DEFAULT_ALIGNMENT;
    byte_span *kv_keep = xcalloc((size_t)g.n_kv, sizeof(kv_keep[0]));
    uint64_t n_kv_keep = 0;

    off_t kv_start = ftello(fp);
    if (kv_start < 0) die("GGUF ftell failed");
    for (uint64_t i = 0; i < g.n_kv; i++) {
        off_t rec_start = ftello(fp);
        if (rec_start < 0 || rec_start < kv_start) die("GGUF ftell failed");
        char *key = read_gguf_string_fp(fp);
        uint32_t type = read_u32_le_fp(fp, "GGUF KV type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = read_u32_le_fp(fp, "GGUF alignment");
            if (a) g.alignment = a;
        } else if (strcmp(key, "deepseek4.expert_count") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t n = read_u32_le_fp(fp, "GGUF expert count");
            if (n <= (uint32_t)INT_MAX) g.n_experts = (int)n;
        } else if (strcmp(key, "deepseek4.expert_count") == 0 && type == GGUF_TYPE_UINT64) {
            uint64_t n = read_u64_le_fp(fp, "GGUF expert count");
            if (n <= (uint64_t)INT_MAX) g.n_experts = (int)n;
        } else {
            skip_gguf_value_fp(fp, type);
        }
        off_t rec_end = ftello(fp);
        if (rec_end < 0 || rec_end < rec_start) die("GGUF ftell failed");

        /*
         * Template GGUFs may already carry imatrix provenance from a previous
         * quantization.  Drop those keys and write the current run's keys later,
         * otherwise the output can contain duplicate GGUF metadata with stale
         * and new values.
         */
        if (!is_imatrix_kv_key(key)) {
            kv_keep[n_kv_keep++] = (byte_span){
                .start = (size_t)(rec_start - kv_start),
                .end = (size_t)(rec_end - kv_start),
            };
        }
        free(key);
    }
    off_t tensor_start = ftello(fp);
    if (tensor_start < 0 || tensor_start < kv_start) die("GGUF ftell failed");
    size_t kv_full_len = (size_t)(tensor_start - kv_start);
    uint8_t *kv_full = xmalloc(kv_full_len);
    if (fseeko(fp, kv_start, SEEK_SET) != 0) die("GGUF seek failed");
    if (kv_full_len && fread(kv_full, 1, kv_full_len, fp) != kv_full_len) die("GGUF KV read failed");

    for (uint64_t i = 0; i < n_kv_keep; i++) g.kv_raw_len += kv_keep[i].end - kv_keep[i].start;
    g.kv_raw = xmalloc(g.kv_raw_len);
    size_t kv_pos = 0;
    for (uint64_t i = 0; i < n_kv_keep; i++) {
        size_t n = kv_keep[i].end - kv_keep[i].start;
        memcpy(g.kv_raw + kv_pos, kv_full + kv_keep[i].start, n);
        kv_pos += n;
    }
    g.n_kv = n_kv_keep;
    free(kv_full);
    free(kv_keep);
    if (fseeko(fp, tensor_start, SEEK_SET) != 0) die("GGUF seek failed");

    g.tensors = xcalloc((size_t)g.n_tensors, sizeof(g.tensors[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        t->name = read_gguf_string_fp(fp);
        t->n_dims = (int)read_u32_le_fp(fp, "GGUF tensor rank");
        if (t->n_dims < 1 || t->n_dims > DS4Q_MAX_DIMS) die("bad GGUF tensor rank");
        for (int j = 0; j < t->n_dims; j++) t->ne[j] = (int64_t)read_u64_le_fp(fp, "GGUF tensor dim");
        t->type = (ds4q_type)read_u32_le_fp(fp, "GGUF tensor type");
        t->old_offset = read_u64_le_fp(fp, "GGUF tensor offset");
        t->size = tensor_nbytes(t->type, t->ne, t->n_dims);
    }
    off_t meta_end = ftello(fp);
    if (meta_end < 0) die("GGUF ftell failed");
    g.data_offset = ds4q_pad((size_t)meta_end, g.alignment);
    char **keys = xmalloc((size_t)g.n_tensors * sizeof(keys[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) keys[i] = g.tensors[i].name;
    hmap_build(&g.tensor_map, keys, (int)g.n_tensors);
    free(keys);
    fclose(fp);
    return g;
}

/* --mtp-append(2026-08-18): 把 DSpark drafter(mtp.<N>.*) 张量条目注入输出目录。
 * 官方 template 头不含 mtp 条目(这就是历来量化产物没带 drafter 的根因); mtp 层与主层
 * 同构 ⇒ 层内张量 shape/type 兜底抄 blk.0 同名条目, 特殊头(main_proj/confidence/
 * markov/norm/hc_head)从 HF dims 现读(GGUF ne = HF shape 反序)。HF 里不存在的
 * 条目(indexer/compressor 等)自动跳过。档位仍走 policy_type(可用 --tensor-type
 * mtp. 前缀整体覆盖)。 */
static void append_mtp_tensors(gguf_file *g, const char *hf_dir, int n_mtp) {
    st_db db;
    db_open(&db, hf_dir);
    const uint64_t base_n = g->n_tensors;
    uint64_t cap = base_n + (uint64_t)n_mtp * 96u;
    g->tensors = realloc(g->tensors, (size_t)cap * sizeof(g->tensors[0]));
    if (!g->tensors) die("mtp-append: realloc failed");
    uint64_t added = 0;
    char gname[512], hfbuf[560];
    for (int N = 0; N < n_mtp; N++) {
        /* 层内条目: 以 blk.0.* 为形态模板 */
        for (uint64_t i = 0; i < base_n; i++) {
            const char *nm = g->tensors[i].name;
            if (strncmp(nm, "blk.0.", 6) != 0) continue;
            snprintf(gname, sizeof(gname), "mtp.%d.%s", N, nm + 6);
            bool is_exps = strstr(nm, "_exps.weight") != NULL;
            if (!is_exps) {
                /* 存在性检查: 经映射得 HF 名; 映射不到或 HF 没有则跳过 */
                const char *suffix = nm + 6;
                const char *hf_suffix = NULL;
                for (size_t mi = 0; mi < sizeof(mtp_map) / sizeof(mtp_map[0]); mi++)
                    if (strcmp(suffix, mtp_map[mi].gguf) == 0) { hf_suffix = mtp_map[mi].hf; break; }
                if (!hf_suffix) {
                    for (size_t mi = 0; mi < sizeof(layer_map) / sizeof(layer_map[0]); mi++)
                        if (strcmp(suffix, layer_map[mi].gguf) == 0) { hf_suffix = layer_map[mi].hf; break; }
                }
                if (!hf_suffix) continue;
                snprintf(hfbuf, sizeof(hfbuf), "mtp.%d.%s", N, hf_suffix);
                if (!db_has(&db, hfbuf)) continue;
            }
            tensor_meta t = g->tensors[i];
            t.name = xstrdup(gname);
            t.old_offset = 0;
            g->tensors[base_n + added] = t;
            added++;
            if (base_n + added >= cap) die("mtp-append: cap overflow");
        }
        /* 特殊头(blk 层没有的形态): shape 从 HF 读 */
        static const struct { const char *g; ds4q_type fallback; } specials[] = {
            { "exp_probs_b.bias",       DS4Q_TYPE_F32 },
            { "main_proj.weight",       DS4Q_TYPE_Q8_0 },
            { "main_norm.weight",       DS4Q_TYPE_F32 },
            { "confidence_proj.weight", DS4Q_TYPE_Q8_0 },
            { "markov_w1.weight",       DS4Q_TYPE_Q8_0 },
            { "markov_w2.weight",       DS4Q_TYPE_Q8_0 },
            { "norm.weight",            DS4Q_TYPE_F32 },
            { "hc_head_base.weight",    DS4Q_TYPE_F32 },
            { "hc_head_fn.weight",      DS4Q_TYPE_F32 },
            { "hc_head_scale.weight",   DS4Q_TYPE_F32 },
        };
        for (size_t si = 0; si < sizeof(specials) / sizeof(specials[0]); si++) {
            snprintf(gname, sizeof(gname), "mtp.%d.%s", N, specials[si].g);
            const char *hf_suffix = NULL;
            for (size_t mi = 0; mi < sizeof(mtp_map) / sizeof(mtp_map[0]); mi++)
                if (strcmp(specials[si].g, mtp_map[mi].gguf) == 0) { hf_suffix = mtp_map[mi].hf; break; }
            if (!hf_suffix) continue;
            snprintf(hfbuf, sizeof(hfbuf), "mtp.%d.%s", N, hf_suffix);
            if (!db_has(&db, hfbuf)) continue;
            tensor_entry *te = db_tensor(&db, hfbuf, NULL);
            tensor_meta t = {0};
            t.name = xstrdup(gname);
            t.n_dims = te->info.n_dims;
            for (int d = 0; d < t.n_dims; d++) t.ne[d] = te->info.shape[t.n_dims - 1 - d];
            while (t.n_dims > 1 && t.ne[t.n_dims - 1] == 1) t.n_dims--;   /* HF 前导 1 维 */
            /* 1D 或小张量保 F32; 矩阵默认 q8_0 兜底(policy/override 可重定) */
            t.type = (t.n_dims <= 1) ? DS4Q_TYPE_F32 : specials[si].fallback;
            t.old_offset = 0;
            t.size = tensor_nbytes(t.type, t.ne, t.n_dims);
            g->tensors[base_n + added] = t;
            added++;
            if (base_n + added >= cap) die("mtp-append: cap overflow");
        }
    }
    g->n_tensors = base_n + added;
    /* 重建名字索引(read_gguf_tensor_data 等按 map 查) */
    free(g->tensor_map.slots);
    char **keys = xmalloc((size_t)g->n_tensors * sizeof(keys[0]));
    for (uint64_t i = 0; i < g->n_tensors; i++) keys[i] = g->tensors[i].name;
    hmap_build(&g->tensor_map, keys, (int)g->n_tensors);
    free(keys);
    db_close(&db);
    fprintf(stderr, "mtp-append: injected %llu drafter tensors (%d modules)\n",
            (unsigned long long)added, n_mtp);
}


/* --mtp-only: append 之后把非 mtp.* 条目全部剔除, 输出成独立 drafter 文件
 * (官方开源 DSpark 量化版形态: 单 mtp 模块 ≈7GB)。KV 头照抄 template。 */
static void keep_mtp_only(gguf_file *g) {
    uint64_t kept = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strncmp(g->tensors[i].name, "mtp.", 4) == 0) g->tensors[kept++] = g->tensors[i];
    if (!kept) die("mtp-only: no mtp.* tensors (use with --mtp-append N)");
    g->n_tensors = kept;
    free(g->tensor_map.slots);
    char **keys = xmalloc((size_t)kept * sizeof(keys[0]));
    for (uint64_t i = 0; i < kept; i++) keys[i] = g->tensors[i].name;
    hmap_build(&g->tensor_map, keys, (int)kept);
    free(keys);
    fprintf(stderr, "mtp-only: kept %llu drafter tensors\n", (unsigned long long)kept);
}

static byte_buf read_gguf_tensor_data(const gguf_file *g, const char *path, const char *name) {
    int idx = hmap_get(&g->tensor_map, name);
    if (idx < 0) {
        fprintf(stderr, "error: tensor not found in GGUF: %s\n", name);
        exit(1);
    }
    const tensor_meta *t = &g->tensors[idx];
    byte_buf b = { .size = t->size, .data = xmalloc(t->size) };
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    if (fseeko(fp, (off_t)(g->data_offset + t->old_offset), SEEK_SET) != 0) die_errno("seek GGUF", path);
    if (b.size && fread(b.data, 1, b.size, fp) != b.size) die_errno("read GGUF tensor", path);
    fclose(fp);
    return b;
}

static uint64_t fnv1a64_bytes(const uint8_t *data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* ===== --zchain: go-onebit 优化链(DQZ2)并入输出 GGUF 为原生张量 =====
 * 用户产品形态: 量化(1bit 专家字节)与优化(每层最优 z/向后/GE; 四损失+感知已重解进
 * 系数)合并输出【一个】GGUF。每层张量:
 *   blk.L.opt_chain.weight  F32 [n_ops*16]   — 引擎打包 op 记录(16 float/op:
 *       [0]=type 1GL|2dyn2|3dyn8|4TREF, [1]=g/t, [2..5]=w2p, [6..14]=w8,
 *       [15]=层内 v8 块号(-1 无))
 *   blk.L.opt_ge.weight     F32 [n_expert]   — 有效 GE(最后一条 type5)
 *   blk.L.opt_v8.weight     F16 [nblk*8*d_model] — dyn8 投影(层内块序)
 * + KV ds4.zchain.present=true(引擎自动装载开关)。 */
typedef struct {
    int       n_layer, n_expert, d_model;
    float   **chain;  int *n_ops;      /* [L] 打包 op 记录 / op 数 */
    float   **ge;                      /* [L] F32 增益或 NULL */
    uint16_t **v8;    int *n_v8;       /* [L] fp16 dyn8 块 / 块数 */
    uint16_t **zlm;   uint32_t *zlm_ne;/* [L] 冻结 z^L fp16 载荷{z[k],U[d·k],V[d·k]} / 元素数(type6, 2026-07-14) */
    int       n_extra_tensors;
} zchain_in;
static zchain_in g_zc = {0};

static int zchain_in_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open zchain", path);
    uint32_t magic = read_u32_le_fp(fp, "zchain magic");
    uint32_t nl = read_u32_le_fp(fp, "zchain layers");
    if (magic != 0x325A5144u) die("bad zchain magic (want DQZ2)");
    if (nl == 0 || nl > 512) die("bad zchain layer count");
    g_zc.n_layer = (int)nl;
    g_zc.chain = xcalloc(nl, sizeof(g_zc.chain[0]));
    g_zc.n_ops = xcalloc(nl, sizeof(g_zc.n_ops[0]));
    g_zc.ge    = xcalloc(nl, sizeof(g_zc.ge[0]));
    g_zc.v8    = xcalloc(nl, sizeof(g_zc.v8[0]));
    g_zc.n_v8  = xcalloc(nl, sizeof(g_zc.n_v8[0]));
    g_zc.zlm   = xcalloc(nl, sizeof(g_zc.zlm[0]));
    g_zc.zlm_ne= xcalloc(nl, sizeof(g_zc.zlm_ne[0]));
    for (uint32_t li = 0; li < nl; li++) {
        uint32_t L = read_u32_le_fp(fp, "zchain L");
        uint32_t nops = read_u32_le_fp(fp, "zchain nops");
        if (L >= nl) die("zchain layer index out of range");
        float *ch = nops ? xcalloc((size_t)nops * 16, sizeof(float)) : NULL;
        int oi = 0;
        for (uint32_t k = 0; k < nops; k++) {
            uint32_t ty = read_u32_le_fp(fp, "zchain op type");
            uint32_t psz = read_u32_le_fp(fp, "zchain op paysz");
            uint8_t *pay = xmalloc(psz ? psz : 1);
            if (psz && fread(pay, 1, psz, fp) != psz) die("zchain payload read");
            if (ty == 6u && psz >= 16) {         /* 冻结 z^L(2026-07-14): 头{k,tr,din,dout}+fp16{z,U,V} */
                uint32_t zk, din, dout; float tr;
                memcpy(&zk, pay, 4); memcpy(&tr, pay + 4, 4);
                memcpy(&din, pay + 8, 4); memcpy(&dout, pay + 12, 4);
                size_t nh = DS4_AMP_ZL_ELEMS(zk, din, dout);
                /* 旧闸 zk<=64 && din==dout 会把冠军级秩/ftA(din=3D)记录静默丢掉:
                 * 合出的 GGUF 少放大器还不吭声。契约=zk≤1024, din∈{D,3D}, d_model 取 dout。 */
                if (!(zk > 0 && zk <= DS4_AMP_ZK_MAX && (din == dout || din == 3u * dout)
                      && psz >= DS4_AMP_OP_HDR + nh * 2)) {
                    fprintf(stderr, "zchain zl 记录非法: k=%u din=%u dout=%u psz=%u need=%zu\n",
                            zk, din, dout, psz, (size_t)DS4_AMP_OP_HDR + nh * 2);
                    die("zchain zl record invalid");
                }
                if (!g_zc.d_model) g_zc.d_model = (int)dout;
                if ((int)dout != g_zc.d_model) die("zchain zl d_model mismatch");
                float *f = ch + (size_t)oi * DS4_AMP_CHAIN_FLOATS;
                f[0] = 6.0f; f[1] = tr; f[2] = (float)zk; f[3] = (float)din; f[15] = 0.0f;
                free(g_zc.zlm[L]);
                g_zc.zlm[L] = xmalloc(nh * 2);
                memcpy(g_zc.zlm[L], pay + 16, nh * 2);
                g_zc.zlm_ne[L] = (uint32_t)nh;
                oi++;
            } else if (ty == 5u && psz >= 2) {   /* GE: 取最后一条(量化器回放口径) */
                int ne = (int)(psz / 2);
                if (!g_zc.n_expert) g_zc.n_expert = ne;
                if (ne != g_zc.n_expert) die("zchain GE width mismatch");
                if (!g_zc.ge[L]) g_zc.ge[L] = xmalloc((size_t)ne * sizeof(float));
                for (int e = 0; e < ne; e++)
                    g_zc.ge[L][e] = ds4q_f16_to_f32(load_u16_le(pay + (size_t)e * 2));
            } else if (ty >= 1u && ty <= 4u) {
                float *f = ch + (size_t)oi * DS4_AMP_CHAIN_FLOATS;
                f[0] = (float)ty; f[15] = -1.0f;
                if (ty == 1u && psz >= 4) memcpy(&f[1], pay, 4);
                else if (ty == 2u && psz >= 16) memcpy(&f[2], pay, 16);
                else if (ty == 3u && psz > 36) {     /* 带 V8 才收; 无 V8 dyn8(psz==36)= 回放 no-op,
                                                        落到末尾 else 丢弃。w8 必须在判收之后才写 —
                                                        早写会把 w8 残留进本槽, 下一 op 复用槽位时
                                                        f[6..14] 带脏字节(引擎按型不读但字节必须干净) */
                    memcpy(&f[6], pay, 36);
                    int dm = (int)((psz - 36) / (8 * 2));   /* 携带 V8: 层内块序 */
                    if (!g_zc.d_model) g_zc.d_model = dm;
                    if (dm != g_zc.d_model) die("zchain V8 d_model mismatch");
                    int blk = g_zc.n_v8[L]++;
                    g_zc.v8[L] = xrealloc(g_zc.v8[L],
                        (size_t)g_zc.n_v8[L] * 8 * dm * sizeof(uint16_t));
                    memcpy(g_zc.v8[L] + (size_t)blk * 8 * dm, pay + 36,
                           (size_t)8 * dm * sizeof(uint16_t));
                    f[15] = (float)blk;
                }
                else if (ty == 4u && psz >= 4) memcpy(&f[1], pay, 4);
                else { free(pay); continue; }
                oi++;
            }
            free(pay);
        }
        g_zc.n_ops[L] = oi;
        if (oi) g_zc.chain[L] = ch; else free(ch);
    }
    fclose(fp);
    int nt = 0, nops_tot = 0;
    for (int L = 0; L < g_zc.n_layer; L++) {
        if (g_zc.n_ops[L]) nt++;
        if (g_zc.ge[L]) nt++;
        if (g_zc.n_v8[L]) nt++;
        if (g_zc.zlm_ne[L]) nt++;
        nops_tot += g_zc.n_ops[L];
    }
    g_zc.n_extra_tensors = nt;
    fprintf(stderr, "zchain: %s -> %d extra tensors (%d chain ops, n_expert=%d, d_model=%d)\n",
            path, nt, nops_tot, g_zc.n_expert, g_zc.d_model);
    return nt;
}

/* extra 张量的数据源(与 build_output_context 追加顺序一一对应) */
typedef struct { const void *data; size_t size; } zc_payload;
static zc_payload *g_zc_pay = NULL;
static uint64_t g_zc_first_extra = 0;
static const char *ZC_PRESENT_KEY = "ds4.zchain.present";

static output_context build_output_context(const gguf_file *tmpl, const quant_policy *policy,
                                            const imatrix_store *im, const hot_mask *hm) {
    output_context out = {0};
    out.n_tensors = tmpl->n_tensors;
    out.n_kv_extra = extra_imatrix_kv_count(im) + extra_keepmap_kv_count(hm);
    out.alignment = tmpl->alignment;
    out.tensors = xcalloc((size_t)out.n_tensors, sizeof(out.tensors[0]));
    size_t tensor_info = 0;
    size_t off = 0;
    for (uint64_t i = 0; i < out.n_tensors; i++) {
        const tensor_meta *src = &tmpl->tensors[i];
        tensor_meta *dst = &out.tensors[i];
        *dst = *src;
        dst->name = src->name;
        ds4q_type type = policy_type(policy, src->name, src);
        if (type == DS4Q_TYPE_COUNT) type = src->type;
        /* 1D norm/bias/sinks 数学上非矩阵, 任何档位/override 下都保源精度 */
        if (src->n_dims <= 1 && ds4q_can_quantize(type) && type != src->type) {
            fprintf(stderr, "1d-guard: %s keeps source type\n", src->name);
            type = src->type;
        }
        if (type != DS4Q_TYPE_I32 && !is_quantizable_target(type)) die("unsupported planned tensor type");
        if (ds4q_can_quantize(type) && src->ne[0] % ds4q_block_size(type) != 0) {
            fprintf(stderr, "block-size fallback: %s ne0=%lld type=%d -> keep source type %d\n",
                    src->name, (long long)src->ne[0], (int)type, (int)src->type);
            type = src->type;
            if (ds4q_can_quantize(type) && src->ne[0] % ds4q_block_size(type) != 0)
                die("ne[0] not divisible by block size (even at source type)");
        }
        /* Hot mask shrinks the routed-expert dim (last) of exps tensors, incl.
         * the MTP draft block (mtp.0) when a matched MTP mask is supplied. */
        if (hm && hm->active) {
            expert_tensor e = parse_expert_tensor(src->name);
            if (e.is_expert) {
                if (e.layer < 0 || e.layer >= hm->n_layers) die("hot mask: expert layer out of range");
                int last = src->n_dims - 1;
                if (src->ne[last] != (int64_t)hm->n_expert)
                    die("hot mask: expert tensor last-dim != mask n_expert");
                dst->ne[last] = (int64_t)hm->kept_counts[e.layer];
            }
        }
        dst->type = type;
        dst->size = tensor_nbytes(type, dst->ne, dst->n_dims);
        dst->new_offset = off;
        off += ds4q_pad(dst->size, tmpl->alignment);
        tensor_info += gguf_string_size(dst->name) + 4 + (size_t)dst->n_dims * 8 + 4 + 8;
    }
    /* --zchain: 追加优化链张量(量化+优化合一 GGUF)。名字持有独立分配, 进程存续。 */
    if (g_zc.n_extra_tensors) {
        g_zc_first_extra = out.n_tensors;
        out.tensors = xrealloc(out.tensors,
            (size_t)(out.n_tensors + g_zc.n_extra_tensors) * sizeof(out.tensors[0]));
        g_zc_pay = xcalloc((size_t)g_zc.n_extra_tensors, sizeof(g_zc_pay[0]));
        int xi = 0;
        for (int L = 0; L < g_zc.n_layer; L++) {
            struct { const char *fmt; ds4q_type ty; int64_t ne0; const void *d; } add[4];
            int na = 0;
            if (g_zc.n_ops[L]) { add[na].fmt="blk.%d.opt_chain.weight"; add[na].ty=DS4Q_TYPE_F32;
                add[na].ne0=(int64_t)g_zc.n_ops[L]*16; add[na].d=g_zc.chain[L]; na++; }
            if (g_zc.ge[L])   { add[na].fmt="blk.%d.opt_ge.weight"; add[na].ty=DS4Q_TYPE_F32;
                add[na].ne0=(int64_t)g_zc.n_expert; add[na].d=g_zc.ge[L]; na++; }
            if (g_zc.n_v8[L]) { add[na].fmt="blk.%d.opt_v8.weight"; add[na].ty=DS4Q_TYPE_F16;
                add[na].ne0=(int64_t)g_zc.n_v8[L]*8*g_zc.d_model; add[na].d=g_zc.v8[L]; na++; }
            if (g_zc.zlm_ne[L]) { add[na].fmt="blk.%d.opt_zlm.weight"; add[na].ty=DS4Q_TYPE_F16;
                add[na].ne0=(int64_t)g_zc.zlm_ne[L]; add[na].d=g_zc.zlm[L]; na++; }
            for (int a = 0; a < na; a++) {
                tensor_meta *dst = &out.tensors[out.n_tensors];
                memset(dst, 0, sizeof(*dst));
                char *nm = xmalloc(48); snprintf(nm, 48, add[a].fmt, L);
                dst->name = nm; dst->n_dims = 1; dst->ne[0] = add[a].ne0;
                dst->type = add[a].ty;
                dst->size = tensor_nbytes(dst->type, dst->ne, dst->n_dims);
                dst->new_offset = off;
                off += ds4q_pad(dst->size, tmpl->alignment);
                tensor_info += gguf_string_size(dst->name) + 4 + (size_t)dst->n_dims * 8 + 4 + 8;
                g_zc_pay[xi++] = (zc_payload){ add[a].d, dst->size };
                out.n_tensors++;
            }
        }
        out.n_kv_extra += 1;   /* ds4.zchain.present */
    }
    out.tensor_bytes = off;
    out.meta_size = 4 + 4 + 8 + 8 + tmpl->kv_raw_len
                  + extra_imatrix_kv_size(im) + extra_keepmap_kv_size(hm) + tensor_info
                  + (g_zc.n_extra_tensors ? gguf_string_size(ZC_PRESENT_KEY) + 4 + 1 : 0);
    out.data_offset = ds4q_pad(out.meta_size, tmpl->alignment);
    return out;
}

static void write_padding(FILE *fp, size_t n) {
    static const uint8_t zeros[4096] = {0};
    while (n) {
        size_t chunk = n < sizeof(zeros) ? n : sizeof(zeros);
        if (fwrite(zeros, 1, chunk, fp) != chunk) die("write padding failed");
        n -= chunk;
    }
}

/* Cluster split-quantize (no shared FS): --layers lo-hi selects which block
 * layers THIS host generates; unselected tensors become fseek holes at their
 * exact planned offsets, so two hosts produce structurally identical files
 * whose written ranges are disjoint. --manifest records {name, abs_offset,
 * bytes} per written tensor; quant_assemble.py splices a peer's ranges into
 * the coordinator's file and the result is byte-identical to a single-host
 * run (generation is deterministic). Layer classing: token_embd/embed-side
 * globals ride with layer 0 (lo side); output.*/
/* /mtp.* ride with the last layer (hi side). */
static int tensor_layer_class(const char *name) {
    int layer;
    if (sscanf(name, "blk.%d.", &layer) == 1) return layer;
    if (strncmp(name, "mtp.", 4) == 0) return 1 << 20;      /* hi side */
    if (strncmp(name, "output", 6) == 0) return 1 << 20;    /* output.*, output_norm */
    return -1;                                              /* embed-side globals: lo side */
}

/* --experts-hole: routed 专家 tensor(blk.*.ffn_*_exps)只留稀疏洞不计算不写 —
 * 骨架实占≈骨干几GB(APFS 稀疏), 专家字节由 go-onebit 层文件 merge 时 pwrite 填入。
 * mtp.* 专家与 shexp 不匹配此模式(正常量化写入)。 */
static int g_experts_hole = 0;
static int is_routed_exps_name(const char *name) {
    int layer, rest = 0; char kind[16];
    return sscanf(name, "blk.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2 && name[rest] == 0;
}
static int layer_selected(int cls, int lo, int hi, int n_layers) {
    if (lo < 0) return 1;                    /* no --layers: everything */
    if (cls < 0) return lo == 0;             /* embed-globals with the lo end */
    if (cls >= (1 << 20)) return hi >= n_layers - 1;  /* output/mtp with the hi end */
    return cls >= lo && cls <= hi;
}

