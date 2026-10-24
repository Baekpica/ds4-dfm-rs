/* ---------------- ssh/本地 流式取字节 ---------------- */

static int host_is_local(const char *h) {
    return !strcmp(h, "") || !strcmp(h, "-") || !strcmp(h, "local") ||
           !strcmp(h, "127.0.0.1") || !strcmp(h, "localhost");
}

/* 把 s 包成单引号 shell 词(内嵌单引号转成 '\'') —— popen 走本地 /bin/sh, 必须保证
 * ssh 收到的 argv 与 .py 的 Popen(list) 完全一致(host 一个词, 远端命令串一个词)。 */
static void shq(const char *s, char *dst, size_t dsz) {
    size_t o = 0;
    if (o + 1 < dsz) dst[o++] = '\'';
    for (const char *p = s; *p; p++) {
        if (*p == '\'') { const char *r = "'\\''"; for (int i = 0; i < 4 && o + 1 < dsz; i++) dst[o++] = r[i]; }
        else if (o + 1 < dsz) dst[o++] = *p;
    }
    if (o + 1 < dsz) dst[o++] = '\'';
    if (o >= dsz) die("ValueError: shell 引用溢出");
    dst[o] = 0;
}

/* 流式拷贝 nbytes 到 out(逐块, 校验总长)。tail/head 而非 dd:
 * macOS BSD dd 不支持 GNU iflag=skip_bytes,count_bytes(实测 0B, 2026-07-28)。 */
static void ssh_stream(const char *host, const char *path, uint64_t nbytes, FILE *out, uint64_t skip) {
    uint8_t *buf = chunk_strm();
    if (host_is_local(host)) {
        /* 本地直读(2026-07-31 R28: 合并在产物同机跑, 免 ssh 往返/免自连密钥) */
        FILE *f = fopen(path, "rb");
        if (!f) die("FileNotFoundError: %s: %s", path, strerror(errno));
        fseeko(f, (off_t)skip, SEEK_SET);
        uint64_t left = nbytes;
        while (left) {
            size_t want = left < CHUNK_STRM ? (size_t)left : CHUNK_STRM;
            size_t got = fread(buf, 1, want, f);
            if (!got) break;
            if (fwrite(buf, 1, got, out) != got) die("IOError: 写输出失败");
            left -= got;
        }
        fclose(f);
        if (left != 0) die("AssertionError: %s: 本地读 %lluB ≠ %lluB", path,
                           (unsigned long long)(nbytes - left), (unsigned long long)nbytes);
        return;
    }
    char qh[1024], qc[4096], rem[3072], cmd[8192];
    snprintf(rem, sizeof rem, "tail -c +%llu %s | head -c %llu",
             (unsigned long long)(skip + 1), path, (unsigned long long)nbytes);
    shq(host, qh, sizeof qh); shq(rem, qc, sizeof qc);
    snprintf(cmd, sizeof cmd, "ssh %s %s", qh, qc);
    FILE *p = popen(cmd, "r");
    if (!p) die("OSError: popen 失败: %s", cmd);
    uint64_t got_tot = 0;
    for (;;) {
        size_t n = fread(buf, 1, CHUNK_STRM, p);
        if (!n) break;
        if (fwrite(buf, 1, n, out) != n) die("IOError: 写输出失败");
        got_tot += n;
    }
    pclose(p);
    if (got_tot != nbytes) die("AssertionError: %s: 流式 %lluB ≠ %lluB", path,
                                (unsigned long long)got_tot, (unsigned long long)nbytes);
}

/* ---------------- opt 内嵌(反修 op 编进合一 GGUF) ---------------- */

/* ★opt 内嵌(2026-08-06 用户令"修通op")★: 反修 op(dql_ops 侧车终值+dql 正位 zl.RRR)
 * 编成引擎 zchain_from_model 期望的四类张量内嵌进合并 GGUF(blk.L.opt_chain/ge/v8/zlm
 * + kv ds4.zchain.present=true)。模型自包含, 运行时零外挂零开关。 */

#define V8SZ (8u * D_MODEL * 2u)   /* 65536 */

typedef struct { float (*r)[16]; size_t n, cap; } chain_t;
static void chain_add(chain_t *c, const float *row) {
    if (c->n == c->cap) { c->cap = c->cap * 2 + 16; c->r = xrealloc(c->r, c->cap * sizeof *c->r); }
    memcpy(c->r[c->n++], row, 16 * sizeof(float));
}
typedef struct { uint8_t **b; size_t n, cap; } v8_t;
static void v8_add(v8_t *v, const uint8_t *blk) {
    if (v->n == v->cap) { v->cap = v->cap * 2 + 8; v->b = xrealloc(v->b, v->cap * sizeof *v->b); }
    v->b[v->n] = xmalloc(V8SZ); memcpy(v->b[v->n], blk, V8SZ); v->n++;
}

/* → 追加 [entries(kind=K_OPT, data=bytes)] 引擎 zchain_from_model 格式; 返回新增数量。 */
static int build_opt_tensors(const char *dql_dir, int L, elist_t *out) {
    chain_t chain = { 0 };
    v8_t v8 = { 0 };
    uint8_t *ge = NULL; size_t ge_len = 0;
    uint8_t *zlm = NULL; size_t zlm_len = 0;
    int has_glhc = 0; float gh = 0, gc = 0;   /* (gh,gc): 折进 per-expert ge 表(引擎 GE 通道零改动) */
    char p[2048];

    snprintf(p, sizeof p, "%s/dql_ops_L%02d.bin", dql_dir, L);
    if (fexists(p)) {
        size_t rlen = 0; uint8_t *raw = read_all(p, &rlen);
        if (rlen >= 12 && (!memcmp(raw, "DQL2", 4) || !memcmp(raw, "DQO2", 4))) {
            uint32_t nrec; memcpy(&nrec, raw + 8, 4);
            uint64_t off = 12;
            for (uint32_t k = 0; k < nrec; k++) {
                if (off > rlen || rlen - off < REC_HDR) break;
                char nm[17]; memcpy(nm, raw + (size_t)off, 16); nm[16] = 0;   /* split(b"\0")[0] */
                uint64_t psz; memcpy(&psz, raw + (size_t)off + DS4_AMP_REC_OFF_PSZ, 8);
                int32_t vd; memcpy(&vd, raw + (size_t)off + DS4_AMP_REC_OFF_VD, 4);
                uint64_t pstart = off + REC_HDR;
                uint64_t plen = rlen - pstart; if (plen > psz) plen = psz;   /* py 切片自动截断 */
                const uint8_t *pay = raw + (size_t)pstart;
                off = sat_add(sat_add(off, REC_HDR), psz);
                if (vd != 1) continue;
                float row[16]; memset(row, 0, sizeof row);
                if (strstr(nm, "GLdyn2") && plen >= 16) {
                    row[0] = 2.0f; memcpy(row + 2, pay, 16); chain_add(&chain, row);
                } else if (strstr(nm, "GLdyn8") && plen >= 36) {
                    row[0] = 3.0f; memcpy(row + 6, pay, 36);
                    if (plen >= 36 + (uint64_t)V8SZ) { row[15] = (float)v8.n; v8_add(&v8, pay + 36); }
                    else continue;                    /* V8-less dyn8 = 引擎侧 no-op, 不嵌 */
                    chain_add(&chain, row);
                } else if (strstr(nm, "bf.GE") && plen >= 512) {
                    float *g = xmalloc(NEXP * 4);
                    for (int e = 0; e < NEXP; e++) { uint16_t h; memcpy(&h, pay + 2 * e, 2); g[e] = f16_to_f32(h); }
                    free(ge); ge = (uint8_t *)g; ge_len = NEXP * 4;
                } else if (strstr(nm, "zl.RRR") && plen >= 16) {
                    /* 与下方 dql 正位读取点同契约(旧版此处同样 zk<=16+写死4096 静默丢冠军) */
                    uint32_t zk, din, dout; float tr;
                    memcpy(&zk, pay, 4); memcpy(&tr, pay + 4, 4);
                    memcpy(&din, pay + 8, 4); memcpy(&dout, pay + 12, 4);
                    uint64_t nh = DS4_AMP_ZL_ELEMS(zk, din, dout);
                    if (!(zk > 0 && zk <= DS4_AMP_ZK_MAX && dout == (uint32_t)D_MODEL
                          && (din == dout || din == 3u * dout) && plen >= DS4_AMP_OP_HDR + nh * 2))
                        die("zl.RRR L%d 混装记录非法: k=%u din=%u dout=%u plen=%llu need=%llu — 拒绝静默丢放大器",
                            L, zk, din, dout, (unsigned long long)plen,
                            (unsigned long long)(DS4_AMP_OP_HDR + nh * 2));
                    row[0] = 6.0f; row[1] = tr; row[2] = (float)zk; row[3] = (float)din;
                    chain_add(&chain, row);
                    free(zlm); zlm_len = (size_t)(nh * 2); zlm = xmalloc(zlm_len);
                    memcpy(zlm, pay + 16, zlm_len);
                } else if (strstr(nm, "GLhc") && plen >= 8) {
                    memcpy(&gh, pay, 4); memcpy(&gc, pay + 4, 4); has_glhc = 1;
                } else if (strstr(nm, ".GL") && plen >= 4) {
                    row[0] = 1.0f; memcpy(&row[1], pay, 4); chain_add(&chain, row);
                } else if ((strstr(nm, "TREF") || strstr(nm, "xlayer")) && plen >= 4) {
                    row[0] = 4.0f; memcpy(&row[1], pay, 4); chain_add(&chain, row);
                }
            }
        }
        free(raw);
    }

    if (!zlm) {   /* zl.RRR 正位在 dql 主文件(2026-08-04 正位直写) */
        snprintf(p, sizeof p, "%s/dql_L%02d.bin", dql_dir, L);
        if (fexists(p)) {
            FILE *f = fopen(p, "rb");
            if (!f) die("FileNotFoundError: %s: %s", p, strerror(errno));
            uint8_t head[12]; size_t hn = fread(head, 1, 12, f);
            if (hn >= 4 && !memcmp(head, "DQL2", 4)) {
                if (hn < 12) die("struct.error: unpack_from requires a buffer of at least 12 bytes");
                uint32_t nrec; memcpy(&nrec, head + 8, 4);
                for (uint32_t k = 0; k < nrec; k++) {
                    uint8_t hdr[REC_HDR];
                    if (fread(hdr, 1, REC_HDR, f) < REC_HDR) break;
                    char nm[17]; memcpy(nm, hdr, 16); nm[16] = 0;
                    uint64_t psz; memcpy(&psz, hdr + DS4_AMP_REC_OFF_PSZ, 8);
                    if (strstr(nm, "zl.RRR") && psz >= 16) {
                        uint8_t *pay = xmalloc((size_t)psz);
                        size_t pn = fread(pay, 1, (size_t)psz, f);
                        if (pn < 16) die("struct.error: zl.RRR 头截断(<16B)");
                        /* 尺寸按载荷自述 din/dout 算(旧版写死 din=dout=4096 且 zk<=16:
                         * ftA 载荷 V 被截 1/3, 冠军 k=64 直接静默丢 —— 判的模型≠部署的模型)。 */
                        uint32_t zk, din, dout; float tr;
                        memcpy(&zk, pay, 4); memcpy(&tr, pay + 4, 4);
                        memcpy(&din, pay + 8, 4); memcpy(&dout, pay + 12, 4);
                        uint64_t nh = DS4_AMP_ZL_ELEMS(zk, din, dout);
                        if (!(zk > 0 && zk <= DS4_AMP_ZK_MAX && dout == (uint32_t)D_MODEL
                              && (din == dout || din == 3u * dout) && pn >= DS4_AMP_OP_HDR + nh * 2))
                            die("zl.RRR L%d 记录非法: k=%u din=%u dout=%u pn=%zu need=%llu — 拒绝静默丢放大器",
                                L, zk, din, dout, pn, (unsigned long long)(DS4_AMP_OP_HDR + nh * 2));
                        float row[16]; memset(row, 0, sizeof row);
                        row[0] = 6.0f; row[1] = tr; row[2] = (float)zk; row[3] = (float)din;
                        chain_add(&chain, row);
                        zlm_len = (size_t)(nh * 2); zlm = xmalloc(zlm_len);
                        memcpy(zlm, pay + 16, zlm_len);
                        free(pay);
                        break;
                    }
                    fseeko(f, (off_t)psz, SEEK_CUR);
                }
            }
            fclose(f);
        }
    }

    if (has_glhc) {
        float *gev = xmalloc(NEXP * 4);
        if (ge_len) memcpy(gev, ge, NEXP * 4);
        else for (int e = 0; e < NEXP; e++) gev[e] = 1.0f;
        char hot[NEXP]; memset(hot, 0, sizeof hot);
        snprintf(p, sizeof p, "%s/dql_vq_L%02d.bin", dql_dir, L);
        if (fexists(p)) {
            FILE *vf = fopen(p, "rb");
            if (!vf) die("FileNotFoundError: %s: %s", p, strerror(errno));
            fseeko(vf, 16, SEEK_SET);
            uint64_t *vt = xmalloc(768 * 8);
            if (fread(vt, 1, 768 * 8, vf) != 768 * 8) die("struct.error: unpack requires a buffer of 6144 bytes");
            for (int e = 0; e < NEXP; e++) if (vt[e * 3 + 2] != 0) hot[e] = 1;
            free(vt); fclose(vf);
        }
        /* numpy: f32 数组 *= python float(弱标量) → 全程 f32 乘, 与这里一致 */
        for (int e = 0; e < NEXP; e++) gev[e] *= hot[e] ? gh : gc;
        free(ge); ge = (uint8_t *)gev; ge_len = NEXP * 4;
    }

    int n = 0;
    char nmbuf[128];
    if (chain.n) {
        size_t nb = chain.n * DS4_AMP_CHAIN_FLOATS * sizeof(float);
        uint8_t *data = xmalloc(nb);
        memcpy(data, chain.r, nb);
        snprintf(nmbuf, sizeof nmbuf, "blk.%d.opt_chain.weight", L);
        el_push1(out, nmbuf, (uint64_t)(chain.n * DS4_AMP_CHAIN_FLOATS), 0, nb, K_OPT)->data = data;
        n++;
    }
    if (ge_len) {
        snprintf(nmbuf, sizeof nmbuf, "blk.%d.opt_ge.weight", L);
        el_push1(out, nmbuf, NEXP, 0, ge_len, K_OPT)->data = ge;
        n++;
    }
    if (v8.n) {
        size_t nb = v8.n * V8SZ;
        uint8_t *data = xmalloc(nb);
        for (size_t i = 0; i < v8.n; i++) memcpy(data + i * V8SZ, v8.b[i], V8SZ);
        snprintf(nmbuf, sizeof nmbuf, "blk.%d.opt_v8.weight", L);
        el_push1(out, nmbuf, (uint64_t)(v8.n * 8 * D_MODEL), 1, nb, K_OPT)->data = data;
        n++;
    }
    if (zlm) {
        snprintf(nmbuf, sizeof nmbuf, "blk.%d.opt_zlm.weight", L);
        el_push1(out, nmbuf, (uint64_t)(zlm_len / 2), 1, zlm_len, K_OPT)->data = zlm;
        n++;
    }
    free(chain.r);
    for (size_t i = 0; i < v8.n; i++) free(v8.b[i]);
    free(v8.b);
    return n;
}

static void kv_append_bool(buf_t *kv_raw, uint64_t *n_kv, const char *key, int val) {
    size_t kl = strlen(key);
    buf_u64(kv_raw, kl); buf_add(kv_raw, key, kl);
    buf_u32(kv_raw, 7);
    uint8_t b = val ? 1 : 0; buf_add(kv_raw, &b, 1);
    (*n_kv)++;
}

