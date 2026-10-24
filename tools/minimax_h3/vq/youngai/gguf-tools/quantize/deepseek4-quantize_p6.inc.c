static void write_full_gguf(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                            const char *out_path, int n_experts, int n_threads,
                            const imatrix_store *imatrix, const hot_mask *hm,
                            int layers_lo, int layers_hi, const char *manifest_path) {
    FILE *fp = fopen(out_path, "wb");
    if (!fp) die_errno("open output", out_path);
    FILE *mf = NULL;
    if (manifest_path) {
        mf = fopen(manifest_path, "w");
        if (!mf) die_errno("open manifest", manifest_path);
    }
    int n_layers = 0;   /* highest blk index + 1, for hi-side global assignment */
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        int c = tensor_layer_class(out_ctx->tensors[i].name);
        if (c >= 0 && c < (1 << 20) && c + 1 > n_layers) n_layers = c + 1;
    }
    if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
    write_u32(fp, tmpl->version);
    write_u64(fp, out_ctx->n_tensors);   /* 含 --zchain 追加的 opt_* 张量 */
    write_u64(fp, tmpl->n_kv + out_ctx->n_kv_extra);
    if (fwrite(tmpl->kv_raw, 1, tmpl->kv_raw_len, fp) != tmpl->kv_raw_len) die("write GGUF KV failed");
    write_imatrix_kvs(fp, imatrix);
    write_keepmap_kvs(fp, hm);
    if (g_zc.n_extra_tensors) {          /* ds4.zchain.present=true: 引擎自动装载 opt_* */
        write_gguf_string(fp, ZC_PRESENT_KEY);
        write_u32(fp, GGUF_TYPE_BOOL);
        if (fputc(1, fp) == EOF) die("write zchain KV failed");
    }
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *t = &out_ctx->tensors[i];
        write_gguf_string(fp, t->name);
        write_u32(fp, (uint32_t)t->n_dims);
        for (int j = 0; j < t->n_dims; j++) write_u64(fp, (uint64_t)t->ne[j]);
        write_u32(fp, (uint32_t)t->type);
        write_u64(fp, t->new_offset);
    }
    long pos = ftell(fp);
    if (pos < 0) die("ftell failed");
    if ((size_t)pos > out_ctx->data_offset) die("GGUF metadata larger than planned");
    write_padding(fp, out_ctx->data_offset - (size_t)pos);

    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *dst = &out_ctx->tensors[i];
        const size_t padded = ds4q_pad(dst->size, out_ctx->alignment);
        if (!layer_selected(tensor_layer_class(dst->name), layers_lo, layers_hi, n_layers)) {
            /* peer's share: leave an exact-size hole (sparse seek, no write) */
            if (fseeko(fp, (off_t)padded, SEEK_CUR) != 0) die_errno("seek hole", out_path);
            continue;
        }
        if (g_zc.n_extra_tensors && i >= g_zc_first_extra) {
            /* --zchain 追加张量: 数据即内存里的优化链载荷, 直接写(无模板源) */
            const zc_payload *zp = &g_zc_pay[i - g_zc_first_extra];
            if (zp->size != dst->size) die("zchain payload size mismatch");
            if (mf) { fprintf(mf, "%s\t%llu\t%zu\n", dst->name,
                        (unsigned long long)(out_ctx->data_offset + dst->new_offset), zp->size); fflush(mf); }
            if (fwrite(zp->data, 1, zp->size, fp) != zp->size) die_errno("write zchain tensor", out_path);
            write_padding(fp, padded - zp->size);
            continue;
        }
        const tensor_meta *src = &tmpl->tensors[i];
        if (g_experts_hole && is_routed_exps_name(dst->name)) {
            /* 骨架模式: routed 专家留洞(不计算不写), merge 阶段由层文件填 */
            if (fseeko(fp, (off_t)padded, SEEK_CUR) != 0) die_errno("seek hole", out_path);
            continue;
        }
        fprintf(stderr, "[%4" PRIu64 "/%4" PRIu64 "] %s -> %s\n", i + 1, out_ctx->n_tensors, dst->name, ds4q_type_name(dst->type));
        byte_buf data = generate_tensor(db, dst->name, src, dst->type, n_experts, n_threads, imatrix, hm);
        size_t expected = dst->size;
        if (data.size != expected) {
            fprintf(stderr, "error: generated size mismatch for %s: got %zu expected %zu\n", dst->name, data.size, expected);
            exit(1);
        }
        if (mf) {
            fprintf(mf, "%s\t%llu\t%zu\n", dst->name,
                    (unsigned long long)(out_ctx->data_offset + dst->new_offset), data.size);
            fflush(mf);
        }
        if (fwrite(data.data, 1, data.size, fp) != data.size) die_errno("write tensor", out_path);
        write_padding(fp, padded - data.size);
        fprintf(stderr, "       generated %.2f MiB\n", (double)data.size / 1048576.0);
        free(data.data);
    }
    /* materialize the file's full length even when the tail was a hole */
    if (fseeko(fp, 0, SEEK_END) == 0) {
        off_t end = ftello(fp);
        off_t want = (off_t)(out_ctx->data_offset + out_ctx->tensor_bytes);
        if (end < want) {
            if (fseeko(fp, want - 1, SEEK_SET) != 0) die_errno("seek eof", out_path);
            fputc(0, fp);
        }
    }
    if (mf) fclose(mf);
    fclose(fp);
}

static void print_plan(const gguf_file *tmpl, const output_context *out_ctx) {
    size_t tensor_bytes = 0;
    size_t changed = 0;
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        tensor_bytes += out_ctx->tensors[i].size;
        if (i >= tmpl->n_tensors) continue;   /* --zchain 追加张量: 无模板源 */
        const tensor_meta *src = &tmpl->tensors[i];
        const tensor_meta *dst = &out_ctx->tensors[i];
        if (src->type != dst->type) {
            changed++;
            printf("type_change: %s %s -> %s\n", dst->name, ds4q_type_name(src->type), ds4q_type_name(dst->type));
        }
    }
    printf("n_tensors: %" PRIu64 "\n", out_ctx->n_tensors);
    printf("meta_bytes: %zu\n", out_ctx->data_offset);
    printf("tensor_bytes_unpadded: %zu\n", tensor_bytes);
    printf("approx_file_bytes: %zu\n", out_ctx->data_offset + out_ctx->tensor_bytes);
    printf("type_changes: %zu\n", changed);
}

/* =====
 * CLI
 */

typedef struct {
    char *hf_dir;
    char *template_gguf;
    char *out_gguf;
    char *compare_gguf;
    char *compare_tensor;
    char *imatrix_file;
    char *hot_mask_file;
    quant_policy policy;
    int n_experts;
    int n_threads;
    bool dry_run;
    bool overwrite;
    bool imatrix_strict;
    int layers_lo;        /* --layers lo-hi cluster split; -1 = all */
    int layers_hi;
    char *manifest_file;  /* --manifest: written-tensor {name,offset,bytes} list */
    int mtp_append;       /* --mtp-append: 注入 DSpark drafter(mtp.0..N-1) 张量 */
    int mtp_only;         /* --mtp-only: 输出仅保留 mtp.* 张量(独立 drafter 文件, 对齐官方开源形态) */
} params;

static void usage(const char *argv0) {
    printf("usage: %s --hf DIR --template MODEL.gguf --out OUT.gguf [options]\n", argv0);
    printf("\nDeepSeek V4 Flash/Pro safetensors -> GGUF quantizer in plain C.\n\n");
    printf("options:\n");
    printf("  --hf DIR               Hugging Face model directory with model.safetensors.index.json\n");
    printf("  --template FILE        existing DS4 GGUF used for metadata, tensor order, shapes\n");
    printf("  --mtp-append N         inject DSpark drafter tensors (mtp.0..N-1) absent from template\n");
    printf("  --mtp-only             emit ONLY mtp.* tensors (standalone drafter file)\n");
    printf("  --out FILE             output GGUF path\n");
    printf("  --compare-gguf FILE    reference GGUF for --compare-tensor, default template\n");
    printf("  --compare-tensor NAME  regenerate one tensor, byte-compare, and exit\n");
    printf("  --overwrite            replace --out if it already exists\n");
    printf("  --dry-run              print output plan without reading HF tensor data\n");
    printf("  --imatrix FILE         legacy .dat imatrix from ds4 --imatrix-out\n");
    printf("  --imatrix-strict       fail if a quantized tensor has no matching imatrix vector\n");
    printf("  --gpu-verify           re-run the CPU encoder after each GPU encode and byte-compare\n");
    printf("  --experts-hot-mask F   DSXM mask: emit ONLY kept (hot) experts per layer,\n");
    printf("                         compacted, + ds4.expert_keep_map.* metadata (cold\n");
    printf("                         experts dropped; streamed from HF in hybrid layout)\n");
    printf("  --experts TYPE         set routed w1/w2/w3 expert tensors to TYPE\n");
    printf("  --routed-w1 TYPE       routed gate expert tensor type\n");
    printf("  --routed-w2 TYPE       routed down expert tensor type\n");
    printf("  --routed-w3 TYPE       routed up expert tensor type\n");
    printf("  --hash-layers N        first N layers are hash-routed; apply --hash-w* to their experts\n");
    printf("  --hash-w1/w2/w3 TYPE   hash-layer expert type (non-specialty; default: same as routed)\n");
    printf("  --attention-proj TYPE  attn_q/kv/output projection type\n");
    printf("  --attention TYPE       other 2D attention/indexer/compressor type\n");
    printf("  --shared TYPE          shared expert tensor type\n");
    printf("  --embedding TYPE       token embedding type\n");
    printf("  --output TYPE          output.* tensor type\n");
    printf("  --dense TYPE           remaining 2D+ non-routed tensor type\n");
    printf("  --tensor-type PFX=TYPE exact tensor-name or prefix override; may repeat\n");
    printf("  --n-experts N          routed expert count, default template metadata\n");
    printf("  --threads N            expert worker count, default 8\n");
    printf("\nTYPE examples: f16, f32, bf16, q8_0, q4_k, q2_k, iq2_xxs, go1b (strict 1-bit routed experts)\n");
}

static char *need_value(int argc, char **argv, int *i, const char *arg) {
    if (++*i >= argc) {
        fprintf(stderr, "error: missing value for %s\n", arg);
        exit(1);
    }
    return argv[*i];
}

static bool file_exists(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    fclose(fp);
    return true;
}

static params parse_args(int argc, char **argv) {
    params p = {0};
    p.policy.routed_w1 = p.policy.routed_w2 = p.policy.routed_w3 = DS4Q_TYPE_COUNT;
    p.policy.hash_w1 = p.policy.hash_w2 = p.policy.hash_w3 = DS4Q_TYPE_COUNT;
    p.policy.n_hash_layers = 0;
    p.policy.go1b_lo = 0; p.policy.go1b_hi = -1;          /* 禁用 = 全按 routed_w* */
    p.policy.go1b_outside = DS4Q_TYPE_Q8_0;               /* 范围外高精 */
    p.policy.attention_proj = p.policy.attention = p.policy.shared = DS4Q_TYPE_COUNT;
    p.policy.embedding = p.policy.output = p.policy.dense = DS4Q_TYPE_COUNT;
    p.n_experts = 0;
    p.n_threads = 8;
    p.layers_lo = -1;
    p.layers_hi = -1;
    p.manifest_file = NULL;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--hf") == 0) {
            p.hf_dir = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--mtp-append") == 0) {
            p.mtp_append = atoi(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--mtp-only") == 0) {
            p.mtp_only = 1;
        } else if (strcmp(arg, "--template") == 0) {
            p.template_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--out") == 0) {
            p.out_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--compare-gguf") == 0) {
            p.compare_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--compare-tensor") == 0) {
            p.compare_tensor = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--overwrite") == 0) {
            p.overwrite = true;
        } else if (strcmp(arg, "--dry-run") == 0) {
            p.dry_run = true;
        } else if (strcmp(arg, "--imatrix") == 0) {
            p.imatrix_file = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--imatrix-strict") == 0) {
            p.imatrix_strict = true;
        } else if (strcmp(arg, "--gpu-verify") == 0) {
            ds4q_gpu_verify_set(1);   /* 读点在 quants.c(GPU 编码后 CPU 逐字节对拍闸) */
        } else if (strcmp(arg, "--experts-hot-mask") == 0) {
            p.hot_mask_file = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--experts") == 0 || strcmp(arg, "--routed") == 0) {
            ds4q_type t = parse_type(need_value(argc, argv, &i, arg));
            p.policy.routed_w1 = p.policy.routed_w2 = p.policy.routed_w3 = t;
        } else if (strcmp(arg, "--routed-w1") == 0 || strcmp(arg, "--routed-gate") == 0) {
            p.policy.routed_w1 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--routed-w2") == 0 || strcmp(arg, "--routed-down") == 0) {
            p.policy.routed_w2 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--routed-w3") == 0 || strcmp(arg, "--routed-up") == 0) {
            p.policy.routed_w3 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--go1b-layers") == 0) {
            const char *v = need_value(argc, argv, &i, arg);
            int a = 0, b = 0;
            if (sscanf(v, "%d:%d", &a, &b) != 2) die("--go1b-layers needs A:B");
            p.policy.go1b_lo = a; p.policy.go1b_hi = b;
        } else if (strcmp(arg, "--go1b-outside") == 0) {
            p.policy.go1b_outside = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--hash-layers") == 0) {
            p.policy.n_hash_layers = atoi(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--hash-w1") == 0) {
            p.policy.hash_w1 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--hash-w2") == 0) {
            p.policy.hash_w2 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--hash-w3") == 0) {
            p.policy.hash_w3 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--attention-proj") == 0 || strcmp(arg, "--attn-proj") == 0) {
            p.policy.attention_proj = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--attention") == 0) {
            p.policy.attention = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--shared") == 0) {
            p.policy.shared = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--embedding") == 0) {
            p.policy.embedding = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--output") == 0) {
            p.policy.output = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--dense") == 0) {
            p.policy.dense = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--tensor-type") == 0) {
            char *spec = need_value(argc, argv, &i, arg);
            char *eq = strchr(spec, '=');
            if (!eq || eq == spec || !eq[1]) die("bad --tensor-type, expected NAME=TYPE");
            *eq = '\0';
            p.policy.overrides = xrealloc(p.policy.overrides, (size_t)(p.policy.n_overrides + 1) * sizeof(p.policy.overrides[0]));
            p.policy.overrides[p.policy.n_overrides++] = (type_override){ xstrdup(spec), parse_type(eq + 1) };
        } else if (strcmp(arg, "--n-experts") == 0) {
            p.n_experts = atoi(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--threads") == 0) {
            p.n_threads = atoi(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--experts-hole") == 0) {
            g_experts_hole = 1;
        } else if (strcmp(arg, "--zchain") == 0) {
            /* go-onebit 优化链(DQZ2)并入输出 GGUF 为 blk.L.opt_* 张量(量化+优化单文件) */
            zchain_in_load(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--layers") == 0) {
            const char *v = need_value(argc, argv, &i, arg);
            if (sscanf(v, "%d-%d", &p.layers_lo, &p.layers_hi) != 2 ||
                p.layers_lo < 0 || p.layers_hi < p.layers_lo)
                die("--layers expects LO-HI (e.g. 0-11)");
        } else if (strcmp(arg, "--manifest") == 0) {
            p.manifest_file = need_value(argc, argv, &i, arg);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg);
            exit(1);
        }
    }
    if (!p.hf_dir) die("--hf is required");
    if (!p.template_gguf) die("--template is required");
    if (!p.dry_run && !p.compare_tensor && !p.out_gguf) die("--out is required unless --dry-run or --compare-tensor is used");
    if (p.compare_tensor && !p.compare_gguf) p.compare_gguf = p.template_gguf;
    if (p.out_gguf && file_exists(p.out_gguf) && !p.overwrite) die("output exists; use --overwrite");
    return p;
}

static void free_gguf_file(gguf_file *g) {
    free(g->path);
    free(g->kv_raw);
    for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
    free(g->tensors);
    hmap_free(&g->tensor_map);
    memset(g, 0, sizeof(*g));
}

static void compare_one_tensor(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                               const params *p, const imatrix_store *imatrix, const hot_mask *hm) {
    int idx = hmap_get(&tmpl->tensor_map, p->compare_tensor);
    if (idx < 0) {
        fprintf(stderr, "error: tensor not found in template: %s\n", p->compare_tensor);
        exit(1);
    }
    fprintf(stderr, "regenerating %s as %s\n",
            p->compare_tensor, ds4q_type_name(out_ctx->tensors[idx].type));
    byte_buf generated = generate_tensor(db, p->compare_tensor, &tmpl->tensors[idx],
                                         out_ctx->tensors[idx].type, p->n_experts, p->n_threads, imatrix, hm);
    gguf_file ref = load_gguf_metadata(p->compare_gguf);
    byte_buf reference = read_gguf_tensor_data(&ref, p->compare_gguf, p->compare_tensor);
    printf("tensor: %s\n", p->compare_tensor);
    printf("type: %s\n", ds4q_type_name(out_ctx->tensors[idx].type));
    printf("generated_bytes: %zu\n", generated.size);
    printf("reference_bytes: %zu\n", reference.size);
    printf("generated_fnv1a64: %016" PRIx64 "\n", fnv1a64_bytes(generated.data, generated.size));
    printf("reference_fnv1a64: %016" PRIx64 "\n", fnv1a64_bytes(reference.data, reference.size));
    size_t mismatches = 0;
    size_t first = SIZE_MAX;
    const size_t n = generated.size < reference.size ? generated.size : reference.size;
    for (size_t i = 0; i < n; i++) {
        if (generated.data[i] != reference.data[i]) {
            if (first == SIZE_MAX) first = i;
            mismatches++;
        }
    }
    if (generated.size != reference.size) {
        if (first == SIZE_MAX) first = n;
        mismatches += generated.size > reference.size ? generated.size - reference.size : reference.size - generated.size;
    }
    if (!mismatches) {
        printf("byte_compare: OK\n");
    } else {
        printf("byte_compare: FAIL mismatches=%zu first=%zu\n", mismatches, first);
    }
    free(generated.data);
    free(reference.data);
    free_gguf_file(&ref);
}

int main(int argc, char **argv) {
    params p = parse_args(argc, argv);
    imatrix_store imatrix = {0};
    if (p.imatrix_file) imatrix_load(&imatrix, p.imatrix_file, p.imatrix_strict);

    gguf_file tmpl = load_gguf_metadata(p.template_gguf);
    if (p.mtp_append > 0) append_mtp_tensors(&tmpl, p.hf_dir, p.mtp_append);
    if (p.mtp_only) keep_mtp_only(&tmpl);
    if (p.n_experts <= 0) {
        if (tmpl.n_experts > 0) {
            p.n_experts = tmpl.n_experts;
            fprintf(stderr, "using %d routed experts from template metadata\n", p.n_experts);
        } else {
            p.n_experts = 256;
            fprintf(stderr, "warning: template has no deepseek4.expert_count; using Flash default %d routed experts\n", p.n_experts);
        }
    } else {
        fprintf(stderr, "using %d routed experts from --n-experts\n", p.n_experts);
    }
    hot_mask hm = {0};
    if (p.hot_mask_file) {
        hot_mask_load(&hm, p.hot_mask_file);
        if (hm.n_expert != p.n_experts)
            die("--experts-hot-mask n_expert does not match routed expert count");
        fprintf(stderr, "hot-mask build: emitting %d hot experts total (%.1f%% of %d)\n",
                hm.total_kept, 100.0 * (double)hm.total_kept / ((double)hm.n_layers * p.n_experts),
                hm.n_layers * p.n_experts);
    }
    output_context out_ctx = build_output_context(&tmpl, &p.policy, &imatrix, &hm);
    print_plan(&tmpl, &out_ctx);
    if (p.dry_run) { hot_mask_free(&hm); return 0; }

    st_db db;
    db_open(&db, p.hf_dir);
    if (p.compare_tensor) {
        compare_one_tensor(&db, &tmpl, &out_ctx, &p, &imatrix, &hm);
        db_close(&db);
        hot_mask_free(&hm);
        imatrix_free(&imatrix);
        free_gguf_file(&tmpl);
        free(out_ctx.tensors);
        return 0;
    }
    write_full_gguf(&db, &tmpl, &out_ctx, p.out_gguf, p.n_experts, p.n_threads, &imatrix, &hm,
                    p.layers_lo, p.layers_hi, p.manifest_file);
    fprintf(stderr, "wrote %s\n", p.out_gguf);

    db_close(&db);
    hot_mask_free(&hm);
    imatrix_free(&imatrix);
    free_gguf_file(&tmpl);
    free(out_ctx.tensors);
    for (int i = 0; i < p.policy.n_overrides; i++) free(p.policy.overrides[i].prefix);
    free(p.policy.overrides);
    return 0;
}
