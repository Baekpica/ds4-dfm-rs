/* v41_cfg.h — 读 V4.1 官方 inference/config.json 的超参(2026-09-12, v41_to_gguf 用)。
 * 扁平 JSON, 只有标量/一维数组, 用 strstr 定位键再 strtod —— 不引 JSON 库。
 * 键名与官方 ModelArgs 字段一一对应(见 hf/inference/model.py), 缺键 = 硬停(不给默认值,
 * 默认值是"不报错只出错"的温床)。 */
#ifndef V41_CFG_H
#define V41_CFG_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define V41_MAX_LAYERS 64
#define V41_MAX_LIST   64

typedef struct {
    int vocab_size, dim, moe_inter_dim, n_layers, n_mtp_layers, n_heads;
    int n_routed_experts, n_shared_experts, n_activated_experts;
    float route_scale, swiglu_limit, norm_eps, hc_eps;
    int q_lora_rank, head_dim, rope_head_dim, o_groups, o_lora_rank, window_size;
    int compress_ratios[V41_MAX_LAYERS], n_compress_ratios;
    int kv_source_layers[V41_MAX_LIST], n_kv_source;
    int index_source_layers[V41_MAX_LIST], n_index_source;
    int original_seq_len; float rope_theta, rope_factor, compress_rope_theta; int beta_fast, beta_slow;
    int index_n_heads, index_head_dim, index_topk;
    int candidate_source_layer, candidate_topk_blocks, candidate_block_size;
    int hc_mult, hc_sinkhorn_iters;
    int engram_layer_ids[V41_MAX_LIST], n_engram;
    long long engram_num_embeddings[V41_MAX_LIST];
    int engram_vocab_size, engram_max_ngram_size, engram_pad_id, engram_compressed_vocab_size, engram_n_heads, engram_head_dim;
    int dspark_block_size, dspark_noise_token_id, dspark_markov_rank, dspark_n_routed_experts, dspark_n_activated_experts;
    int dspark_target_layer_ids[V41_MAX_LIST], n_dspark_target;
    char score_func[32];
    /* 视觉塔(2026-10-13): HF 顶层 config.json 的 vision_config 子对象 + image_token_id(官方 ModelArgs 的 vision_* 字段)。
     * vis_layers == 0 = 这份权重没有视觉塔(转换器不写 v.* 张量, 引擎拒图片)。 */
    int vis_layers, vis_dim, vis_heads, vis_inter, vis_patch, vis_ds, vis_max_tok, vis_min_px, image_token_id;
    float vis_rope_theta;
} v41_cfg;

static const char *v41_cfg_find(const char *js, const char *key) {
    char pat[128]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) { fprintf(stderr, "★config.json 缺键 %s★\n", key); exit(1); }
    p = strchr(p + strlen(pat), ':');
    if (!p) { fprintf(stderr, "★config.json 键 %s 无值★\n", key); exit(1); }
    p++; while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
    return p;
}
static double v41_cfg_num(const char *js, const char *key) {
    const char *p = v41_cfg_find(js, key);
    if (!strncmp(p, "null", 4)) return 0;
    char *q; double v = strtod(p, &q);
    if (q == p) { fprintf(stderr, "★config.json 键 %s 不是数★\n", key); exit(1); }
    return v;
}
static int v41_cfg_list(const char *js, const char *key, long long *out, int max) {
    const char *p = v41_cfg_find(js, key);
    if (*p != '[') { fprintf(stderr, "★config.json 键 %s 不是数组★\n", key); exit(1); }
    p++; int n = 0;
    for (;;) {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == ',') p++;
        if (*p == ']') break;
        char *q; long long v = strtoll(p, &q, 10);
        if (q == p) { fprintf(stderr, "★config.json 数组 %s 解析失败★\n", key); exit(1); }
        if (n < max) out[n] = v;
        n++; p = q;
    }
    if (n > max) { fprintf(stderr, "★config.json 数组 %s 太长(%d > %d)★\n", key, n, max); exit(1); }
    return n;
}
static void v41_cfg_ints(const char *js, const char *key, int *out, int *n, int max) {
    long long tmp[V41_MAX_LAYERS];
    *n = v41_cfg_list(js, key, tmp, max);
    for (int i = 0; i < *n; i++) out[i] = (int)tmp[i];
}
static void v41_cfg_str(const char *js, const char *key, char *out, size_t cap) {
    const char *p = v41_cfg_find(js, key);
    if (*p != '"') { fprintf(stderr, "★config.json 键 %s 不是字串★\n", key); exit(1); }
    p++; size_t i = 0;
    while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++;
    out[i] = 0;
}

static char *v41_cfg_read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "★打不开 %s★\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *js = (char *)malloc((size_t)L + 1);
    if (fread(js, 1, (size_t)L, f) != (size_t)L) { fprintf(stderr, "★读不满 %s★\n", path); exit(1); }
    js[L] = 0; fclose(f);
    return js;
}
/* 上下文长度 = 模型自己声明的 max_position_embeddings。它只在 HF 顶层 config.json 里(quant 目录里是软链;
 * inference/config.json 只有 YaRN 的 original_seq_len), 转换器把它写成 GGUF 的 deepseek4.context_length, 引擎只认这个键 ——
 * 用户令(2026-09-22)"不要任何写死的上下文", 所以转换器与引擎两边都没有一个写死的上下文数。 */
static uint64_t v41_cfg_context_length(const char *qdir) {
    char hp[4200]; snprintf(hp, sizeof hp, "%s/config.json", qdir);
    char *js = v41_cfg_read_file(hp);
    const double v = v41_cfg_num(js, "max_position_embeddings");
    free(js);
    if (!(v >= 1.0)) { fprintf(stderr, "★%s 的 max_position_embeddings 不合法: %g★\n", hp, v); exit(1); }
    fprintf(stderr, "[配置] 上下文 %.0f(max_position_embeddings)\n", v);
    return (uint64_t)v;
}
/* 视觉塔超参在 HF 顶层 config.json 的 "vision_config": {…} 里(inference/config.json 没有); 子对象的键名(num_hidden_layers/hidden_size…)
 * 与顶层不撞(顶层没有这几个键, 2026-10-13 核对过), 但保险起见只在子对象的花括号里找。没有 vision_config = 文本模型, vis_layers 留 0。 */
static void v41_cfg_vision_load(v41_cfg *c, const char *qdir) {
    char hp[4200]; snprintf(hp, sizeof hp, "%s/config.json", qdir);
    char *js = v41_cfg_read_file(hp);
    const char *v = strstr(js, "\"vision_config\"");
    if (!v) { free(js); fprintf(stderr, "[配置] config.json 没有 vision_config: 文本模型, 不写视觉塔\n"); return; }
    const char *o = strchr(v, '{'); if (!o) { fprintf(stderr, "★vision_config 不是对象★\n"); exit(1); }
    const char *q = strchr(o, '}'); if (!q) { fprintf(stderr, "★vision_config 没闭合★\n"); exit(1); }
    char *sub = (char *)malloc((size_t)(q - o) + 2); memcpy(sub, o, (size_t)(q - o) + 1); sub[q - o + 1] = 0;
    c->vis_layers = (int)v41_cfg_num(sub, "num_hidden_layers");
    c->vis_dim = (int)v41_cfg_num(sub, "hidden_size");
    c->vis_heads = (int)v41_cfg_num(sub, "num_attention_heads");
    c->vis_inter = (int)v41_cfg_num(sub, "intermediate_size");
    c->vis_patch = (int)v41_cfg_num(sub, "patch_size");
    c->vis_rope_theta = (float)v41_cfg_num(sub, "rope_theta");
    c->vis_ds = (int)v41_cfg_num(sub, "downsample_ratio");
    c->vis_max_tok = (int)v41_cfg_num(sub, "max_image_tokens");
    c->vis_min_px = (int)v41_cfg_num(sub, "min_pixels");
    free(sub);
    c->image_token_id = (int)v41_cfg_num(js, "image_token_id");
    free(js);
    if (c->vis_layers <= 0 || c->vis_dim <= 0 || c->vis_heads <= 0 || c->vis_dim % c->vis_heads || c->vis_inter <= 0 || c->vis_patch <= 0 || c->vis_ds <= 0 || c->vis_max_tok <= 0 || c->image_token_id <= 0) {
        fprintf(stderr, "★vision_config 不合法★\n"); exit(1);
    }
    fprintf(stderr, "[配置] 视觉塔 %d 块 dim %d 头 %d inter %d patch %d 下采样 %d 每图 ≤%d token min_px %d theta %g 占位 id %d\n",
            c->vis_layers, c->vis_dim, c->vis_heads, c->vis_inter, c->vis_patch, c->vis_ds, c->vis_max_tok, c->vis_min_px, (double)c->vis_rope_theta, c->image_token_id);
}
static void v41_cfg_load(v41_cfg *c, const char *path) {
    char *js = v41_cfg_read_file(path);
    memset(c, 0, sizeof *c);
#define I(k) c->k = (int)v41_cfg_num(js, #k)
#define F(k) c->k = (float)v41_cfg_num(js, #k)
    I(vocab_size); I(dim); I(moe_inter_dim); I(n_layers); I(n_mtp_layers); I(n_heads);
    I(n_routed_experts); I(n_shared_experts); I(n_activated_experts);
    F(route_scale); F(swiglu_limit); F(norm_eps); F(hc_eps);
    I(q_lora_rank); I(head_dim); I(rope_head_dim); I(o_groups); I(o_lora_rank); I(window_size);
    I(original_seq_len); F(rope_theta); F(rope_factor); F(compress_rope_theta); I(beta_fast); I(beta_slow);
    I(index_n_heads); I(index_head_dim); I(index_topk);
    I(candidate_source_layer); I(candidate_topk_blocks); I(candidate_block_size);
    I(hc_mult); I(hc_sinkhorn_iters);
    I(engram_vocab_size); I(engram_max_ngram_size); I(engram_pad_id); I(engram_compressed_vocab_size); I(engram_n_heads); I(engram_head_dim);
    I(dspark_block_size); I(dspark_noise_token_id); I(dspark_markov_rank); I(dspark_n_routed_experts); I(dspark_n_activated_experts);
#undef I
#undef F
    v41_cfg_ints(js, "compress_ratios", c->compress_ratios, &c->n_compress_ratios, V41_MAX_LAYERS);
    v41_cfg_ints(js, "kv_source_layers", c->kv_source_layers, &c->n_kv_source, V41_MAX_LIST);
    v41_cfg_ints(js, "index_source_layers", c->index_source_layers, &c->n_index_source, V41_MAX_LIST);
    v41_cfg_ints(js, "engram_layer_ids", c->engram_layer_ids, &c->n_engram, V41_MAX_LIST);
    v41_cfg_ints(js, "dspark_target_layer_ids", c->dspark_target_layer_ids, &c->n_dspark_target, V41_MAX_LIST);
    int ne = v41_cfg_list(js, "engram_num_embeddings", c->engram_num_embeddings, V41_MAX_LIST);
    if (ne != c->n_engram) { fprintf(stderr, "★engram_num_embeddings 数 %d != engram 层数 %d★\n", ne, c->n_engram); exit(1); }
    v41_cfg_str(js, "score_func", c->score_func, sizeof c->score_func);
    if (c->n_compress_ratios < c->n_layers + c->n_mtp_layers) { fprintf(stderr, "★compress_ratios 短于层数★\n"); exit(1); }
    free(js);
}
#endif
