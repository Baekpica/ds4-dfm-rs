/* v41_to_gguf_mtp.inc.c — v41_to_gguf.c 的 DSpark 三塔登记段(2026-09-19 从主文件拆出, 守单文件 ≤500 行)。
 * 聚合根 v41_to_gguf.c 按序 #include 它, 单 TU 语义不变: 用的都是主文件里的 plan_* / C / g_plan_s2。
 * 拆点选在这里是因为三塔这一段自成一体 —— 它是唯一走【第二个源(原始 HF)】的部分, 与主干登记无耦合。 */
/* ---- DSpark 三塔(speed.md 段 6 S1, 2026-09-15) ----
 * 官方 model.py: `mtp.0/1/2` 是三个 SWA(128) 草稿块, 各 128 个专家 top-3。结构与普通层一模一样
 * (hc 六件 + attn 五投影 + ffn gate/专家/shared), 外加五个只此一份的头:
 *   mtp.0.main_proj / main_norm  —— 主模型 L37/38/39 的注意力输入(hc 四路均值)拼成 15360 维再投回 5120,
 *                                   得到 main_x, 它是块注意力的主 KV 来源。取错位置不报错, 只是接受率掉到 1 附近。
 *   mtp.2.markov_head.embed/head —— 秩 256 的 markov 头, 按块内前一个草稿 token 给 logit 加偏置。
 *   mtp.2.confidence_head.proj   —— 出每一位的条件接受概率, 调度器据此决定每轮验证几个。
 *   mtp.2.norm                   —— 三塔共用的出口 norm(官方 self.mtp[-1] 借主模型的 embed/head, GGUF 不再存一份)。
 * ★专家走 FP4 直透, 不做 VQ★(speed.md §2: "草稿质量 = 接受率 = 速度, 不省这里")。
 * ★不要的★: gate.bias_vl(视觉路), 与文件头第 16 行同一条规矩。 */
/* 塔数与每塔专家数一律从 config.json 读(C.n_mtp_layers / C.dspark_n_routed_experts) —— 写死数字
 * 就是把模型架构焊进工具, 换一版模型静默出错(铁律 2026-09-15: 禁魔数硬编码)。 */
/* ★三塔一律从**原始 HF** 取, 按原生精度存(2026-09-17)★ —— 见 S2 那段注释的所以然。
 * MFP8 = 原件的 F8_E4M3 + 32×32 缩放, 原样搬(attn 五投影 / shared 专家三件 / main_proj);
 * MBF16 = 原件的 BF16, 原样搬(路由 gate / markov 两件 / confidence) —— 原来 plan_small 把它展成 f32,
 *   字节翻倍而值没变, 草稿器每轮为此白读 130 MB。★norm 不在此列★: 引擎的 rms_norm 核把权重当 f32 读
 *   (cuda_v41_1.inc.cu), 存成 BF16 它照样按 f32 解释 —— 不报错, 直接出垃圾(实撞: 首位一致率 0.035)。
 *   norm 一律走 MSMALL(BF16→f32, 值无损), 只有过"按类型分发"那条路的矩阵才许存 BF16;
 * MFP4 = 原件出厂就是 FP4 的那一类(128 个路由专家), 照旧;
 * MSMALL = 原件就是 F32 的(sink / gate.bias / hc_*)。
 * ★哪个张量是哪一类不许猜★: plan_* 里都按源 dtype 校验, 对不上直接停车。 */
static void plan_mtp(int T) {
    char g[96], s[192];
    if (!g_has_s2) die("三塔要从原始 HF 取原生精度: 第 5 个参数给 HF 目录(见 --help)");
    g_plan_s2 = 1;
#define MSMALL(gf, sf) do { snprintf(g, sizeof g, gf, T); snprintf(s, sizeof s, sf, T); plan_small(g, s); } while (0)
#define MBF16(gf, sf)  do { snprintf(g, sizeof g, gf, T); snprintf(s, sizeof s, sf, T); plan_bf16(g, s); } while (0)
#define MFP8(gf, sf)   do { snprintf(g, sizeof g, gf, T); snprintf(s, sizeof s, sf, T); plan_fp8blk(g, s); } while (0)
#define MFP4(gf, sf)   do { snprintf(g, sizeof g, gf, T); snprintf(s, sizeof s, sf, T); plan_fp4(g, s); } while (0)
    MSMALL("mtp.%d.hc_attn_fn.weight", "mtp.%d.hc_attn_fn");
    MSMALL("mtp.%d.hc_attn_scale.weight", "mtp.%d.hc_attn_scale");
    MSMALL("mtp.%d.hc_attn_base.weight", "mtp.%d.hc_attn_base");
    MSMALL("mtp.%d.attn_norm.weight", "mtp.%d.attn_norm.weight");
    MFP8("mtp.%d.attn_q_a.weight", "mtp.%d.attn.wq_a.weight");
    MSMALL("mtp.%d.attn_q_a_norm.weight", "mtp.%d.attn.q_norm.weight");
    MFP8("mtp.%d.attn_q_b.weight", "mtp.%d.attn.wq_b.weight");
    MFP8("mtp.%d.attn_kv.weight", "mtp.%d.attn.wkv.weight");
    MSMALL("mtp.%d.attn_kv_a_norm.weight", "mtp.%d.attn.kv_norm.weight");
    MSMALL("mtp.%d.attn_sinks.weight", "mtp.%d.attn.attn_sink");
    MFP8("mtp.%d.attn_output_a.weight", "mtp.%d.attn.wo_a.weight");
    MFP8("mtp.%d.attn_output_b.weight", "mtp.%d.attn.wo_b.weight");
    MSMALL("mtp.%d.hc_ffn_fn.weight", "mtp.%d.hc_ffn_fn");
    MSMALL("mtp.%d.hc_ffn_scale.weight", "mtp.%d.hc_ffn_scale");
    MSMALL("mtp.%d.hc_ffn_base.weight", "mtp.%d.hc_ffn_base");
    MSMALL("mtp.%d.ffn_norm.weight", "mtp.%d.ffn_norm.weight");
    MBF16("mtp.%d.ffn_gate_inp.weight", "mtp.%d.ffn.gate.weight");
    MSMALL("mtp.%d.exp_probs_b.bias", "mtp.%d.ffn.gate.bias");
    MFP8("mtp.%d.ffn_gate_shexp.weight", "mtp.%d.ffn.shared_experts.w1.weight");
    MFP8("mtp.%d.ffn_up_shexp.weight", "mtp.%d.ffn.shared_experts.w3.weight");
    MFP8("mtp.%d.ffn_down_shexp.weight", "mtp.%d.ffn.shared_experts.w2.weight");
    if (mtp_has_vq(T)) {
        /* ★三塔专家走 VQ(2026-09-19 的 100 GB 配方)★: 7.22 → 2.36 GB。VQ 三件只住【量化目录】,
         * 所以这一段要把源切回 S —— 其余 mtp.* 仍从原件取原生精度(09-17 的裁决没变)。 */
        int save = g_plan_s2; g_plan_s2 = 0;
        char gn[96], pre[64];
        snprintf(gn, sizeof gn, "mtp.%d.ffn_exps_vq.blob", T);
        snprintf(pre, sizeof pre, "mtp.%d", T);
        plan_vqblob_v3(pre, T, C.dspark_n_routed_experts, gn);
        g_plan_s2 = save;
    } else {
        for (int e = 0; e < C.dspark_n_routed_experts; e++) {
            snprintf(g, sizeof g, "mtp.%d.ffn_exp.%d.gate.weight", T, e); snprintf(s, sizeof s, "mtp.%d.ffn.experts.%d.w1.weight", T, e); plan_fp4(g, s);
            snprintf(g, sizeof g, "mtp.%d.ffn_exp.%d.up.weight",   T, e); snprintf(s, sizeof s, "mtp.%d.ffn.experts.%d.w3.weight", T, e); plan_fp4(g, s);
            snprintf(g, sizeof g, "mtp.%d.ffn_exp.%d.down.weight", T, e); snprintf(s, sizeof s, "mtp.%d.ffn.experts.%d.w2.weight", T, e); plan_fp4(g, s);
        }
    }
    if (T == 0) {
        plan_fp8blk("mtp.main_proj.weight", "mtp.0.main_proj.weight");   /* 原件 F8_E4M3 32×32 */
        plan_small("mtp.main_norm.weight", "mtp.0.main_norm.weight");
    }
    if (T == 2) {
        plan_bf16("mtp.markov_embd.weight", "mtp.2.markov_head.embed.weight");
        plan_bf16("mtp.markov_head.weight", "mtp.2.markov_head.head.weight");
        plan_bf16("mtp.confidence.weight", "mtp.2.confidence_head.proj.weight");
        plan_small("mtp.out_norm.weight", "mtp.2.norm.weight");
    }
    g_plan_s2 = 0;
#undef MBF16
#undef MFP8
#undef MSMALL
#undef MFP4
}
