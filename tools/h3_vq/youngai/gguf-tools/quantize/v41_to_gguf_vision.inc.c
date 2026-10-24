/* v41_to_gguf_vision.inc.c — 视觉塔(DeepSeek-ViT)+ aligner 的登记(2026-10-13; v41_to_gguf.c 的分片, 单 TU)。
 *
 * ★只从原始 HF 取(S2), 与三塔同一条规矩★: 量化目录里 vision.* 的大矩阵已被量化器压成 q4_K, 而视觉塔总共 0.3B 参数(BF16 0.6 GB),
 * 占文件 0.5%, 不值得为它冒"看图糊"的险 —— 原件精度一位不动。没给原始 HF 目录就不许带视觉塔(硬停, 不悄悄降级)。
 * 消费端 src/core/core_bind_vision.c 按这里的名字/形状/类型绑定 —— 改名两边同改。
 *   v.patch_embd.{weight,bias}           [3·p·p → D] BF16 / f32
 *   v.blk.N.attn_norm.weight             f32 [D]
 *   v.blk.N.attn_qkv.{weight,bias}       [D → 3D] BF16 / f32
 *   v.blk.N.attn_output.{weight,bias}    [D → D]  BF16 / f32
 *   v.blk.N.ffn_norm.weight              f32 [D]
 *   v.blk.N.ffn_gate_up.weight           [D → 2·inter] BF16(官方 w1: gate|up 并排)
 *   v.blk.N.ffn_down.weight              [inter → D] BF16(官方 w2)
 *   v.output_norm.weight                 f32 [D]
 *   mm.0.{weight,bias} / mm.1.{weight,bias}   aligner w1 [D·r² → E] / w2 [E → E]
 *   v.image_{start,end,newline}.weight   f32 [E]
 *   blk.N.exp_probs_b_vl.bias            f32 [n_expert](路由图片位偏置, 官方 gate.bias_vl; 主干每层一份) */
static void plan_vision(void) {
    if (!C.vis_layers) return;
    if (!g_has_s2) die("带视觉塔的模型必须给原始 HF 目录(第 5 个参数): 视觉塔取原件 BF16");
    g_plan_s2 = 1;
    plan_bf16("v.patch_embd.weight", "vision.patch_embed.proj.weight");
    plan_small("v.patch_embd.bias", "vision.patch_embed.proj.bias");
    char g[96], s[128];
    for (int L = 0; L < C.vis_layers; L++) {
#define VS(gf, sf) do { snprintf(g, sizeof g, gf, L); snprintf(s, sizeof s, sf, L); plan_small(g, s); } while (0)
#define VB(gf, sf) do { snprintf(g, sizeof g, gf, L); snprintf(s, sizeof s, sf, L); plan_bf16(g, s); } while (0)
        VS("v.blk.%d.attn_norm.weight", "vision.blocks.%d.norm1.weight");
        VB("v.blk.%d.attn_qkv.weight", "vision.blocks.%d.attn.wqkv.weight");
        VS("v.blk.%d.attn_qkv.bias", "vision.blocks.%d.attn.wqkv.bias");
        VB("v.blk.%d.attn_output.weight", "vision.blocks.%d.attn.wo.weight");
        VS("v.blk.%d.attn_output.bias", "vision.blocks.%d.attn.wo.bias");
        VS("v.blk.%d.ffn_norm.weight", "vision.blocks.%d.norm2.weight");
        VB("v.blk.%d.ffn_gate_up.weight", "vision.blocks.%d.mlp.w1.weight");
        VB("v.blk.%d.ffn_down.weight", "vision.blocks.%d.mlp.w2.weight");
#undef VS
#undef VB
    }
    plan_small("v.output_norm.weight", "vision.norm.weight");
    plan_bf16("mm.0.weight", "aligner.w1.weight"); plan_small("mm.0.bias", "aligner.w1.bias");
    plan_bf16("mm.1.weight", "aligner.w2.weight"); plan_small("mm.1.bias", "aligner.w2.bias");
    plan_small("v.image_start.weight", "image_start");
    plan_small("v.image_end.weight", "image_end");
    plan_small("v.image_newline.weight", "image_newline");
    for (int L = 0; L < C.n_layers; L++) { snprintf(g, sizeof g, "blk.%d.exp_probs_b_vl.bias", L); snprintf(s, sizeof s, "layers.%d.ffn.gate.bias_vl", L); plan_small(g, s); }
    g_plan_s2 = 0;
    fprintf(stderr, "[视觉] %d 块 + aligner + %d 层 bias_vl 已登记(原件 BF16)\n", C.vis_layers, C.n_layers);
}

/* 引擎 core_validate_v41.c 按这些键认视觉塔(全部必给; 缺一件引擎就只能猜) */
static void emit_vision_kv(void);
static int write_jobs(void);   /* 登记完的张量逐个生成+写出(定义在 v41_to_gguf.c 主流程末尾, 两种模式共用) */

/* ★侧车模式★ `v41_to_gguf vision <量化目录> <原始HF目录> <out.gguf>`(2026-10-13): 只出视觉塔 + aligner + 各层 bias_vl 的小 GGUF(≈1 GB),
 * 引擎 --vision-gguf 当第二个映射装。为什么不进主文件: GGUF 没有数据区起点字段, 头部一长整个数据区后移, HF 上 40 片每一片都变 =
 * 重传 115 GB; 侧车只多传这 1 GB。主 GGUF 的 recipe/tokenizer/engram 元数据一样不碰。 */
static int main_vision(const char *qdir, const char *hfdir, const char *out) {
    g_qdir = qdir; g_out = out;
    if (v41_st_open(&S, qdir)) return 1;
    if (v41_st_open(&S2, hfdir)) { fprintf(stderr, "★打不开原始 HF 目录 %s★\n", hfdir); return 1; }
    g_has_s2 = 1;
    char cfgp[4200]; snprintf(cfgp, sizeof cfgp, "%s/inference/config.json", qdir); v41_cfg_load(&C, cfgp);
    v41_cfg_vision_load(&C, qdir);
    if (!C.vis_layers) die("config.json 没有 vision_config: 这份权重没有视觉塔");
    if (ds4gw_begin(&W, out)) return 1;
    ds4gw_kv_str(&W, "general.architecture", "deepseek4");
    ds4gw_kv_str(&W, "general.name", "DeepSeek V4.1 Flash vision sidecar");
    ds4gw_kv_u32(&W, "general.alignment", 32);
    ds4gw_kv_str(&W, "deepseek4.variant", "v41_flash");
    ds4gw_kv_str(&W, "deepseek4.sidecar", "vision");
    ds4gw_kv_u32(&W, "deepseek4.block_count", (uint32_t)C.n_layers);        /* bias_vl 覆盖的主干层数 */
    ds4gw_kv_u32(&W, "deepseek4.embedding_length", (uint32_t)C.dim);
    emit_vision_kv();
    plan_vision();
    fprintf(stderr, "[登记] %d 张量, 数据区 %.3f GB\n", W.nt, (double)W.total / 1e9);
    if (ds4gw_header(&W)) return 1;
    const int rc = write_jobs();
    v41_st_close(&S);
    return rc;
}

static void emit_vision_kv(void) {
    if (!C.vis_layers) return;
    ds4gw_kv_u32(&W, "deepseek4.vision.block_count", (uint32_t)C.vis_layers);
    ds4gw_kv_u32(&W, "deepseek4.vision.embedding_length", (uint32_t)C.vis_dim);
    ds4gw_kv_u32(&W, "deepseek4.vision.attention.head_count", (uint32_t)C.vis_heads);
    ds4gw_kv_u32(&W, "deepseek4.vision.feed_forward_length", (uint32_t)C.vis_inter);
    ds4gw_kv_u32(&W, "deepseek4.vision.patch_size", (uint32_t)C.vis_patch);
    ds4gw_kv_f32(&W, "deepseek4.vision.rope.freq_base", C.vis_rope_theta);
    ds4gw_kv_u32(&W, "deepseek4.vision.downsample_ratio", (uint32_t)C.vis_ds);
    ds4gw_kv_u32(&W, "deepseek4.vision.max_image_tokens", (uint32_t)C.vis_max_tok);
    ds4gw_kv_u32(&W, "deepseek4.vision.min_pixels", (uint32_t)C.vis_min_px);
    ds4gw_kv_u32(&W, "deepseek4.image_token_id", (uint32_t)C.image_token_id);
}
