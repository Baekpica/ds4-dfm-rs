//! Frozen `g_ds4_shape` catalog copied from the v0.6.5-dfm Qwen golden.
//!
//! Unset C designated-init fields are 0 / false / 0.0. Do not "fill in"
//! Motif `rope_freq_base_swa` or EXAONE `use_rope` from the validator.

use std::fmt::Write as _;

pub const DEFAULT_RMS_EPS: f32 = 1.0e-6;
pub const DEFAULT_HC_EPS: f32 = 1.0e-6;
pub const DEFAULT_SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const DEFAULT_ROPE_FREQ_BASE: f32 = 10000.0;
pub const DEFAULT_ROPE_SCALE_FACTOR: f32 = 16.0;
pub const DEFAULT_ROPE_YARN_BETA_FAST: f32 = 32.0;
pub const DEFAULT_ROPE_YARN_BETA_SLOW: f32 = 1.0;
pub const DEFAULT_COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
pub const DEFAULT_ROPE_ORIG_CTX: u64 = 65536;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFamily {
    DeepSeek4 = 0,
    SolarOpen2 = 1,
    Motif3 = 2,
    ExaoneMoe = 3,
    Dots3Note = 4,
    Qwen4Exp = 5,
    Glm53 = 6,
}

impl ModelFamily {
    pub fn from_oracle_name(s: &str) -> Option<Self> {
        match s {
            "deepseek4" => Some(Self::DeepSeek4),
            "solar-open2" => Some(Self::SolarOpen2),
            "motif3" => Some(Self::Motif3),
            "exaone-moe" => Some(Self::ExaoneMoe),
            "dots3-note" => Some(Self::Dots3Note),
            "qwen4exp" => Some(Self::Qwen4Exp),
            "glm5-next" => Some(Self::Glm53),
            _ => None,
        }
    }

    pub fn oracle_name(self) -> &'static str {
        match self {
            Self::DeepSeek4 => "deepseek4",
            Self::SolarOpen2 => "solar-open2",
            Self::Motif3 => "motif3",
            Self::ExaoneMoe => "exaone-moe",
            Self::Dots3Note => "dots3-note",
            Self::Qwen4Exp => "qwen4exp",
            Self::Glm53 => "glm5-next",
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Flash = 0,
    Pro = 1,
    SolarOpen2_250B = 2,
    Motif3 = 3,
    Kexaone236B = 4,
    Dots3NotePrev = 5,
    Qwen38FlashNext = 6,
    Glm53Flash = 7,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub name: &'static str,
    pub family: ModelFamily,
    pub variant: Variant,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_vocab: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_noise_head: u32,
    pub n_head_dim: u32,
    pub n_value_dim: u32,
    pub n_rot: u32,
    pub n_out_group: u32,
    pub n_lora_q: u32,
    pub n_lora_o: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_expert_shared: u32,
    pub n_ff_exp: u32,
    pub n_ff_dense: u32,
    pub n_ff_shexp: u32,
    pub n_hash_layer: u32,
    pub n_swa: u32,
    pub n_swa_period: u32,
    pub n_indexer_head: u32,
    pub n_indexer_head_dim: u32,
    pub n_indexer_top_k: u32,
    pub n_hc: u32,
    pub n_hc_sinkhorn_iter: u32,
    pub n_nextn_predict: u32,
    pub n_leading_dense: u32,
    pub n_kv_lora: u32,
    pub n_key_mla: u32,
    pub n_value_mla: u32,
    pub n_swa_head: u32,
    pub n_swa_kv_lora: u32,
    pub n_swa_key_mla: u32,
    pub n_full_attn_count: u32,
    pub n_kda_head_dim: u32,
    pub n_ssm_conv: u32,
    pub use_rope: bool,
    pub use_qk_norm: bool,
    pub rms_eps: f32,
    pub kda_l2_eps: f32,
    pub kda_gate_clamp_min: f32,
    pub hc_eps: f32,
    pub expert_weight_scale: f32,
    pub swiglu_clamp_exp: f32,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub rope_scale_factor: f32,
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    pub compress_rope_freq_base: f32,
    pub rope_orig_ctx: u64,
}

impl Shape {
    pub fn model_id(self) -> i32 {
        self.variant as i32
    }

    pub fn dump_line(&self, tag: &str) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "SHAPE\t{tag}\t{name}\t{family}\t{variant}\t{n_layer}\t{n_embd}\t{n_vocab}\t{n_head}\t{n_head_kv}\t{n_noise_head}\t{n_head_dim}\t{n_value_dim}\t{n_rot}\t{n_out_group}\t{n_lora_q}\t{n_lora_o}\t{n_expert}\t{n_expert_used}\t{n_expert_shared}\t{n_ff_exp}\t{n_ff_dense}\t{n_ff_shexp}\t{n_hash_layer}\t{n_swa}\t{n_swa_period}\t{n_indexer_head}\t{n_indexer_head_dim}\t{n_indexer_top_k}\t{n_hc}\t{n_hc_sinkhorn_iter}\t{n_nextn_predict}\t{n_leading_dense}\t{n_kv_lora}\t{n_key_mla}\t{n_value_mla}\t{n_swa_head}\t{n_swa_kv_lora}\t{n_swa_key_mla}\t{n_full_attn_count}\t{n_kda_head_dim}\t{n_ssm_conv}\t{use_rope}\t{use_qk_norm}\t{rms_eps:08x}\t{kda_l2_eps:08x}\t{kda_gate_clamp_min:08x}\t{hc_eps:08x}\t{expert_weight_scale:08x}\t{swiglu_clamp_exp:08x}\t{rope_freq_base:08x}\t{rope_freq_base_swa:08x}\t{rope_scale_factor:08x}\t{rope_yarn_beta_fast:08x}\t{rope_yarn_beta_slow:08x}\t{compress_rope_freq_base:08x}\t{rope_orig_ctx}",
            name = self.name,
            family = self.family as u32,
            variant = self.variant as u32,
            n_layer = self.n_layer,
            n_embd = self.n_embd,
            n_vocab = self.n_vocab,
            n_head = self.n_head,
            n_head_kv = self.n_head_kv,
            n_noise_head = self.n_noise_head,
            n_head_dim = self.n_head_dim,
            n_value_dim = self.n_value_dim,
            n_rot = self.n_rot,
            n_out_group = self.n_out_group,
            n_lora_q = self.n_lora_q,
            n_lora_o = self.n_lora_o,
            n_expert = self.n_expert,
            n_expert_used = self.n_expert_used,
            n_expert_shared = self.n_expert_shared,
            n_ff_exp = self.n_ff_exp,
            n_ff_dense = self.n_ff_dense,
            n_ff_shexp = self.n_ff_shexp,
            n_hash_layer = self.n_hash_layer,
            n_swa = self.n_swa,
            n_swa_period = self.n_swa_period,
            n_indexer_head = self.n_indexer_head,
            n_indexer_head_dim = self.n_indexer_head_dim,
            n_indexer_top_k = self.n_indexer_top_k,
            n_hc = self.n_hc,
            n_hc_sinkhorn_iter = self.n_hc_sinkhorn_iter,
            n_nextn_predict = self.n_nextn_predict,
            n_leading_dense = self.n_leading_dense,
            n_kv_lora = self.n_kv_lora,
            n_key_mla = self.n_key_mla,
            n_value_mla = self.n_value_mla,
            n_swa_head = self.n_swa_head,
            n_swa_kv_lora = self.n_swa_kv_lora,
            n_swa_key_mla = self.n_swa_key_mla,
            n_full_attn_count = self.n_full_attn_count,
            n_kda_head_dim = self.n_kda_head_dim,
            n_ssm_conv = self.n_ssm_conv,
            use_rope = u32::from(self.use_rope),
            use_qk_norm = u32::from(self.use_qk_norm),
            rms_eps = self.rms_eps.to_bits(),
            kda_l2_eps = self.kda_l2_eps.to_bits(),
            kda_gate_clamp_min = self.kda_gate_clamp_min.to_bits(),
            hc_eps = self.hc_eps.to_bits(),
            expert_weight_scale = self.expert_weight_scale.to_bits(),
            swiglu_clamp_exp = self.swiglu_clamp_exp.to_bits(),
            rope_freq_base = self.rope_freq_base.to_bits(),
            rope_freq_base_swa = self.rope_freq_base_swa.to_bits(),
            rope_scale_factor = self.rope_scale_factor.to_bits(),
            rope_yarn_beta_fast = self.rope_yarn_beta_fast.to_bits(),
            rope_yarn_beta_slow = self.rope_yarn_beta_slow.to_bits(),
            compress_rope_freq_base = self.compress_rope_freq_base.to_bits(),
            rope_orig_ctx = self.rope_orig_ctx,
        );
        s
    }
}

/// Dimensions compared by C `ds4_shape_matches_metadata` (DeepSeek only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeepSeekDims {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_vocab: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_head_dim: u32,
    pub n_value_dim: u32,
    pub n_rot: u32,
    pub n_lora_q: u32,
    pub n_lora_o: u32,
    pub n_out_group: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_ff_exp: u32,
    pub n_expert_shared: u32,
    pub n_hash_layer: u32,
    pub n_swa: u32,
    pub n_indexer_head: u32,
    pub n_indexer_head_dim: u32,
    pub n_indexer_top_k: u32,
    pub n_hc: u32,
    pub n_hc_sinkhorn_iter: u32,
}

impl DeepSeekDims {
    pub fn from_shape(s: &Shape) -> Self {
        Self {
            n_layer: s.n_layer,
            n_embd: s.n_embd,
            n_vocab: s.n_vocab,
            n_head: s.n_head,
            n_head_kv: s.n_head_kv,
            n_head_dim: s.n_head_dim,
            n_value_dim: s.n_value_dim,
            n_rot: s.n_rot,
            n_lora_q: s.n_lora_q,
            n_lora_o: s.n_lora_o,
            n_out_group: s.n_out_group,
            n_expert: s.n_expert,
            n_expert_used: s.n_expert_used,
            n_ff_exp: s.n_ff_exp,
            n_expert_shared: s.n_expert_shared,
            n_hash_layer: s.n_hash_layer,
            n_swa: s.n_swa,
            n_indexer_head: s.n_indexer_head,
            n_indexer_head_dim: s.n_indexer_head_dim,
            n_indexer_top_k: s.n_indexer_top_k,
            n_hc: s.n_hc,
            n_hc_sinkhorn_iter: s.n_hc_sinkhorn_iter,
        }
    }
}

fn shape_matches_metadata(s: &Shape, d: &DeepSeekDims) -> bool {
    s.n_layer == d.n_layer
        && s.n_embd == d.n_embd
        && s.n_vocab == d.n_vocab
        && s.n_head == d.n_head
        && s.n_head_kv == d.n_head_kv
        && s.n_head_dim == d.n_head_dim
        && s.n_value_dim == d.n_value_dim
        && s.n_rot == d.n_rot
        && s.n_lora_q == d.n_lora_q
        && s.n_lora_o == d.n_lora_o
        && s.n_out_group == d.n_out_group
        && s.n_expert == d.n_expert
        && s.n_expert_used == d.n_expert_used
        && s.n_ff_exp == d.n_ff_exp
        && s.n_expert_shared == d.n_expert_shared
        && s.n_hash_layer == d.n_hash_layer
        && s.n_swa == d.n_swa
        && s.n_indexer_head == d.n_indexer_head
        && s.n_indexer_head_dim == d.n_indexer_head_dim
        && s.n_indexer_top_k == d.n_indexer_top_k
        && s.n_hc == d.n_hc
        && s.n_hc_sinkhorn_iter == d.n_hc_sinkhorn_iter
}

/// C `ds4_select_shape_from_metadata`. Flash first, then Pro.
pub fn select_shape_from_metadata(d: &DeepSeekDims) -> Option<Shape> {
    if shape_matches_metadata(&SHAPE_FLASH, d) {
        return Some(SHAPE_FLASH);
    }
    if shape_matches_metadata(&SHAPE_PRO, d) {
        return Some(SHAPE_PRO);
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchRoute {
    DeepSeekSelect,
    Fixed(Variant),
    Unsupported,
}

/// C `config_validate_model` architecture arm, without the validators.
/// Missing `general.architecture` is DeepSeek (`||` short-circuit).
pub fn route_architecture(arch: Option<&[u8]>) -> ArchRoute {
    match arch {
        None | Some(b"deepseek4") => ArchRoute::DeepSeekSelect,
        Some(b"exaone-moe") => ArchRoute::Fixed(Variant::Kexaone236B),
        Some(b"solar-open2") => ArchRoute::Fixed(Variant::SolarOpen2_250B),
        Some(b"motif3") => ArchRoute::Fixed(Variant::Motif3),
        Some(b"dots3-note") => ArchRoute::Fixed(Variant::Dots3NotePrev),
        Some(b"qwen4exp") => ArchRoute::Fixed(Variant::Qwen38FlashNext),
        Some(b"glm5-next") => ArchRoute::Fixed(Variant::Glm53Flash),
        Some(_) => ArchRoute::Unsupported,
    }
}

pub fn shape_for_variant(v: Variant) -> Shape {
    match v {
        Variant::Flash => SHAPE_FLASH,
        Variant::Pro => SHAPE_PRO,
        Variant::SolarOpen2_250B => SHAPE_SOLAR_OPEN2_250B,
        Variant::Motif3 => SHAPE_MOTIF3,
        Variant::Kexaone236B => SHAPE_KEXAONE_236B,
        Variant::Dots3NotePrev => SHAPE_DOTS3_NOTE_PREV,
        Variant::Qwen38FlashNext => SHAPE_QWEN38_FLASH_NEXT,
        Variant::Glm53Flash => SHAPE_GLM53_FLASH,
    }
}

fn arch_dump_name(route: ArchRoute) -> &'static str {
    match route {
        ArchRoute::DeepSeekSelect => "deepseek-select",
        ArchRoute::Fixed(v) => shape_for_variant(v).name,
        ArchRoute::Unsupported => "unsupported",
    }
}

pub fn dump_oracle() -> String {
    let mut out = String::new();
    let _ = writeln!(out, "FAMILY\t0\t1\t2\t3\t4\t5\t6");
    let _ = writeln!(out, "VARIANT\t0\t1\t2\t3\t4\t5\t6\t7");
    let _ = writeln!(out, "{}", SHAPE_FLASH.dump_line("DEFAULT"));
    let _ = writeln!(out, "{}", SHAPE_FLASH.dump_line("FLASH"));
    let _ = writeln!(out, "{}", SHAPE_PRO.dump_line("PRO"));
    let _ = writeln!(out, "{}", SHAPE_MOTIF3.dump_line("MOTIF3"));
    let _ = writeln!(out, "{}", SHAPE_SOLAR_OPEN2_250B.dump_line("SOLAR"));
    let _ = writeln!(out, "{}", SHAPE_KEXAONE_236B.dump_line("KEXAONE"));
    let _ = writeln!(out, "{}", SHAPE_DOTS3_NOTE_PREV.dump_line("DOTS3"));
    let _ = writeln!(out, "{}", SHAPE_QWEN38_FLASH_NEXT.dump_line("QWEN38"));
    let _ = writeln!(out, "{}", SHAPE_GLM53_FLASH.dump_line("GLM53"));

    let flash = DeepSeekDims::from_shape(&SHAPE_FLASH);
    let pro = DeepSeekDims::from_shape(&SHAPE_PRO);
    let mut miss = flash;
    miss.n_layer = 1;
    let _ = writeln!(
        out,
        "SELECT\tflash\t{}",
        select_shape_from_metadata(&flash)
            .map(|s| s.name)
            .unwrap_or("unsupported")
    );
    let _ = writeln!(
        out,
        "SELECT\tpro\t{}",
        select_shape_from_metadata(&pro)
            .map(|s| s.name)
            .unwrap_or("unsupported")
    );
    let _ = writeln!(
        out,
        "SELECT\tmiss\t{}",
        select_shape_from_metadata(&miss)
            .map(|s| s.name)
            .unwrap_or("unsupported")
    );

    for (label, arch) in [
        ("missing", None),
        ("deepseek4", Some(&b"deepseek4"[..])),
        ("exaone-moe", Some(&b"exaone-moe"[..])),
        ("solar-open2", Some(&b"solar-open2"[..])),
        ("motif3", Some(&b"motif3"[..])),
        ("dots3-note", Some(&b"dots3-note"[..])),
        ("qwen4exp", Some(&b"qwen4exp"[..])),
        ("glm5-next", Some(&b"glm5-next"[..])),
        ("glm-dsa", Some(&b"glm-dsa"[..])),
    ] {
        let _ = writeln!(
            out,
            "ARCH\t{label}\t{}",
            arch_dump_name(route_architecture(arch))
        );
    }
    out
}

pub const SHAPE_FLASH: Shape = Shape {
    name: "DeepSeek V4 Flash",
    family: ModelFamily::DeepSeek4,
    variant: Variant::Flash,
    n_layer: 43,
    n_embd: 4096,
    n_vocab: 129280,
    n_head: 64,
    n_head_kv: 1,
    n_noise_head: 0,
    n_head_dim: 512,
    n_value_dim: 512,
    n_rot: 64,
    n_out_group: 8,
    n_lora_q: 1024,
    n_lora_o: 1024,
    n_expert: 256,
    n_expert_used: 6,
    n_expert_shared: 1,
    n_ff_exp: 2048,
    n_ff_dense: 0,
    n_ff_shexp: 0,
    n_hash_layer: 3,
    n_swa: 128,
    n_swa_period: 0,
    n_indexer_head: 64,
    n_indexer_head_dim: 128,
    n_indexer_top_k: 512,
    n_hc: 4,
    n_hc_sinkhorn_iter: 20,
    n_nextn_predict: 0,
    n_leading_dense: 0,
    n_kv_lora: 0,
    n_key_mla: 0,
    n_value_mla: 0,
    n_swa_head: 0,
    n_swa_kv_lora: 0,
    n_swa_key_mla: 0,
    n_full_attn_count: 0,
    n_kda_head_dim: 0,
    n_ssm_conv: 0,
    use_rope: true,
    use_qk_norm: false,
    rms_eps: DEFAULT_RMS_EPS,
    kda_l2_eps: 0.0,
    kda_gate_clamp_min: 0.0,
    hc_eps: DEFAULT_HC_EPS,
    expert_weight_scale: 1.5,
    swiglu_clamp_exp: DEFAULT_SWIGLU_CLAMP_EXP,
    rope_freq_base: DEFAULT_ROPE_FREQ_BASE,
    rope_freq_base_swa: 0.0,
    rope_scale_factor: DEFAULT_ROPE_SCALE_FACTOR,
    rope_yarn_beta_fast: DEFAULT_ROPE_YARN_BETA_FAST,
    rope_yarn_beta_slow: DEFAULT_ROPE_YARN_BETA_SLOW,
    compress_rope_freq_base: DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    rope_orig_ctx: DEFAULT_ROPE_ORIG_CTX,
};

pub const SHAPE_PRO: Shape = Shape {
    name: "DeepSeek V4 Pro",
    family: ModelFamily::DeepSeek4,
    variant: Variant::Pro,
    n_layer: 61,
    n_embd: 7168,
    n_vocab: 129280,
    n_head: 128,
    n_head_kv: 1,
    n_noise_head: 0,
    n_head_dim: 512,
    n_value_dim: 512,
    n_rot: 64,
    n_out_group: 16,
    n_lora_q: 1536,
    n_lora_o: 1024,
    n_expert: 384,
    n_expert_used: 6,
    n_expert_shared: 1,
    n_ff_exp: 3072,
    n_ff_dense: 0,
    n_ff_shexp: 0,
    n_hash_layer: 3,
    n_swa: 128,
    n_swa_period: 0,
    n_indexer_head: 64,
    n_indexer_head_dim: 128,
    n_indexer_top_k: 1024,
    n_hc: 4,
    n_hc_sinkhorn_iter: 20,
    n_nextn_predict: 0,
    n_leading_dense: 0,
    n_kv_lora: 0,
    n_key_mla: 0,
    n_value_mla: 0,
    n_swa_head: 0,
    n_swa_kv_lora: 0,
    n_swa_key_mla: 0,
    n_full_attn_count: 0,
    n_kda_head_dim: 0,
    n_ssm_conv: 0,
    use_rope: true,
    use_qk_norm: false,
    rms_eps: DEFAULT_RMS_EPS,
    kda_l2_eps: 0.0,
    kda_gate_clamp_min: 0.0,
    hc_eps: DEFAULT_HC_EPS,
    expert_weight_scale: 2.5,
    swiglu_clamp_exp: DEFAULT_SWIGLU_CLAMP_EXP,
    rope_freq_base: DEFAULT_ROPE_FREQ_BASE,
    rope_freq_base_swa: 0.0,
    rope_scale_factor: DEFAULT_ROPE_SCALE_FACTOR,
    rope_yarn_beta_fast: DEFAULT_ROPE_YARN_BETA_FAST,
    rope_yarn_beta_slow: DEFAULT_ROPE_YARN_BETA_SLOW,
    compress_rope_freq_base: DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    rope_orig_ctx: DEFAULT_ROPE_ORIG_CTX,
};

pub const SHAPE_MOTIF3: Shape = Shape {
    name: "Motif-3",
    family: ModelFamily::Motif3,
    variant: Variant::Motif3,
    n_layer: 53,
    n_embd: 4096,
    n_vocab: 220160,
    n_head: 80,
    n_head_kv: 16,
    n_noise_head: 16,
    n_head_dim: 192,
    n_value_dim: 128,
    n_rot: 64,
    n_out_group: 0,
    n_lora_q: 1024,
    n_lora_o: 0,
    n_expert: 384,
    n_expert_used: 8,
    n_expert_shared: 1,
    n_ff_exp: 1280,
    n_ff_dense: 12288,
    n_ff_shexp: 0,
    n_hash_layer: 0,
    n_swa: 128,
    n_swa_period: 4,
    n_indexer_head: 0,
    n_indexer_head_dim: 0,
    n_indexer_top_k: 0,
    n_hc: 4,
    n_hc_sinkhorn_iter: 20,
    n_nextn_predict: 1,
    n_leading_dense: 2,
    n_kv_lora: 512,
    n_key_mla: 192,
    n_value_mla: 128,
    n_swa_head: 0,
    n_swa_kv_lora: 0,
    n_swa_key_mla: 0,
    n_full_attn_count: 0,
    n_kda_head_dim: 0,
    n_ssm_conv: 0,
    use_rope: false,
    use_qk_norm: false,
    rms_eps: 1.0e-5,
    kda_l2_eps: 0.0,
    kda_gate_clamp_min: 0.0,
    hc_eps: 1.0e-6,
    expert_weight_scale: 2.0,
    swiglu_clamp_exp: 0.0,
    rope_freq_base: 10000.0,
    rope_freq_base_swa: 0.0,
    rope_scale_factor: 64.0,
    rope_yarn_beta_fast: 32.0,
    rope_yarn_beta_slow: 1.0,
    compress_rope_freq_base: 0.0,
    rope_orig_ctx: 4096,
};

pub const SHAPE_SOLAR_OPEN2_250B: Shape = Shape {
    name: "Solar Open2 250B",
    family: ModelFamily::SolarOpen2,
    variant: Variant::SolarOpen2_250B,
    n_layer: 48,
    n_embd: 4096,
    n_vocab: 196608,
    n_head: 64,
    n_head_kv: 8,
    n_noise_head: 0,
    n_head_dim: 128,
    n_value_dim: 128,
    n_rot: 0,
    n_out_group: 0,
    n_lora_q: 0,
    n_lora_o: 0,
    n_expert: 320,
    n_expert_used: 8,
    n_expert_shared: 1,
    n_ff_exp: 1280,
    n_ff_dense: 10240,
    n_ff_shexp: 1280,
    n_hash_layer: 0,
    n_swa: 0,
    n_swa_period: 0,
    n_indexer_head: 0,
    n_indexer_head_dim: 0,
    n_indexer_top_k: 0,
    n_hc: 0,
    n_hc_sinkhorn_iter: 0,
    n_nextn_predict: 0,
    n_leading_dense: 0,
    n_kv_lora: 0,
    n_key_mla: 0,
    n_value_mla: 0,
    n_swa_head: 0,
    n_swa_kv_lora: 0,
    n_swa_key_mla: 0,
    n_full_attn_count: 0,
    n_kda_head_dim: 128,
    n_ssm_conv: 4,
    use_rope: false,
    use_qk_norm: false,
    rms_eps: 1.0e-5,
    kda_l2_eps: 1.0e-6,
    kda_gate_clamp_min: -5.0,
    hc_eps: 0.0,
    expert_weight_scale: 1.0,
    swiglu_clamp_exp: 0.0,
    rope_freq_base: DEFAULT_ROPE_FREQ_BASE,
    rope_freq_base_swa: 0.0,
    rope_scale_factor: 1.0,
    rope_yarn_beta_fast: 0.0,
    rope_yarn_beta_slow: 0.0,
    compress_rope_freq_base: 0.0,
    rope_orig_ctx: 1_048_576,
};

pub const SHAPE_KEXAONE_236B: Shape = Shape {
    name: "K-EXAONE 236B A23B",
    family: ModelFamily::ExaoneMoe,
    variant: Variant::Kexaone236B,
    n_layer: 49,
    n_embd: 6144,
    n_vocab: 153600,
    n_head: 64,
    n_head_kv: 8,
    n_noise_head: 0,
    n_head_dim: 128,
    n_value_dim: 128,
    n_rot: 128,
    n_out_group: 0,
    n_lora_q: 0,
    n_lora_o: 0,
    n_expert: 128,
    n_expert_used: 8,
    n_expert_shared: 1,
    n_ff_exp: 2048,
    n_ff_dense: 18432,
    n_ff_shexp: 2048,
    n_hash_layer: 0,
    n_swa: 128,
    n_swa_period: 4,
    n_indexer_head: 0,
    n_indexer_head_dim: 0,
    n_indexer_top_k: 0,
    n_hc: 0,
    n_hc_sinkhorn_iter: 0,
    n_nextn_predict: 1,
    n_leading_dense: 1,
    n_kv_lora: 0,
    n_key_mla: 0,
    n_value_mla: 0,
    n_swa_head: 0,
    n_swa_kv_lora: 0,
    n_swa_key_mla: 0,
    n_full_attn_count: 0,
    n_kda_head_dim: 0,
    n_ssm_conv: 0,
    use_rope: false,
    use_qk_norm: true,
    rms_eps: 1.0e-5,
    kda_l2_eps: 0.0,
    kda_gate_clamp_min: 0.0,
    hc_eps: 0.0,
    expert_weight_scale: 2.5,
    swiglu_clamp_exp: 0.0,
    rope_freq_base: 1_000_000.0,
    rope_freq_base_swa: 0.0,
    rope_scale_factor: 1.0,
    rope_yarn_beta_fast: 0.0,
    rope_yarn_beta_slow: 0.0,
    compress_rope_freq_base: 0.0,
    rope_orig_ctx: 262144,
};

pub const SHAPE_DOTS3_NOTE_PREV: Shape = Shape {
    name: "dots3-note-prev",
    family: ModelFamily::Dots3Note,
    variant: Variant::Dots3NotePrev,
    n_layer: 47,
    n_embd: 5120,
    n_vocab: 152064,
    n_head: 128,
    n_head_kv: 128,
    n_noise_head: 0,
    n_head_dim: 192,
    n_value_dim: 128,
    n_rot: 64,
    n_out_group: 0,
    n_lora_q: 1024,
    n_lora_o: 0,
    n_expert: 256,
    n_expert_used: 8,
    n_expert_shared: 1,
    n_ff_exp: 1536,
    n_ff_dense: 13824,
    n_ff_shexp: 0,
    n_hash_layer: 0,
    n_swa: 513,
    n_swa_period: 4,
    n_indexer_head: 64,
    n_indexer_head_dim: 128,
    n_indexer_top_k: 2048,
    n_hc: 0,
    n_hc_sinkhorn_iter: 0,
    n_nextn_predict: 1,
    n_leading_dense: 1,
    n_kv_lora: 512,
    n_key_mla: 192,
    n_value_mla: 128,
    n_swa_head: 64,
    n_swa_kv_lora: 1024,
    n_swa_key_mla: 256,
    n_full_attn_count: 13,
    n_kda_head_dim: 0,
    n_ssm_conv: 0,
    use_rope: true,
    use_qk_norm: false,
    rms_eps: 1.0e-5,
    kda_l2_eps: 0.0,
    kda_gate_clamp_min: 0.0,
    hc_eps: 0.0,
    expert_weight_scale: 1.0,
    swiglu_clamp_exp: 0.0,
    rope_freq_base: 80_000_000.0,
    rope_freq_base_swa: 50_000.0,
    rope_scale_factor: 1.0,
    rope_yarn_beta_fast: 0.0,
    rope_yarn_beta_slow: 0.0,
    compress_rope_freq_base: 0.0,
    rope_orig_ctx: 524288,
};

pub const SHAPE_QWEN38_FLASH_NEXT: Shape = Shape {
    name: "Qwen3.8-Flash-Next",
    family: ModelFamily::Qwen4Exp,
    variant: Variant::Qwen38FlashNext,
    n_layer: 48,
    n_embd: 2560,
    n_vocab: 248320,
    n_head: 24,
    n_head_kv: 2,
    n_noise_head: 0,
    n_head_dim: 256,
    n_value_dim: 256,
    n_rot: 64,
    n_out_group: 0,
    n_lora_q: 0,
    n_lora_o: 0,
    n_expert: 512,
    n_expert_used: 10,
    n_expert_shared: 1,
    n_ff_exp: 640,
    n_ff_dense: 0,
    n_ff_shexp: 640,
    n_hash_layer: 0,
    n_swa: 0,
    n_swa_period: 4,
    n_indexer_head: 4,
    n_indexer_head_dim: 128,
    n_indexer_top_k: 2048,
    n_hc: 4,
    n_hc_sinkhorn_iter: 0,
    n_nextn_predict: 1,
    n_leading_dense: 0,
    n_kv_lora: 0,
    n_key_mla: 0,
    n_value_mla: 0,
    n_swa_head: 0,
    n_swa_kv_lora: 0,
    n_swa_key_mla: 0,
    n_full_attn_count: 12,
    n_kda_head_dim: 128,
    n_ssm_conv: 4,
    use_rope: true,
    use_qk_norm: true,
    rms_eps: 1.0e-6,
    kda_l2_eps: 0.0,
    kda_gate_clamp_min: 0.0,
    hc_eps: 1.0e-6,
    expert_weight_scale: 1.0,
    swiglu_clamp_exp: 0.0,
    rope_freq_base: 10_000_000.0,
    rope_freq_base_swa: 0.0,
    rope_scale_factor: 1.0,
    rope_yarn_beta_fast: 0.0,
    rope_yarn_beta_slow: 0.0,
    compress_rope_freq_base: 0.0,
    rope_orig_ctx: 262144,
};

pub const SHAPE_GLM53_FLASH: Shape = Shape {
    name: "GLM 5.3 Flash",
    family: ModelFamily::Glm53,
    variant: Variant::Glm53Flash,
    n_layer: 46,
    n_embd: 4096,
    n_vocab: 154880,
    n_head: 64,
    n_head_kv: 1,
    n_noise_head: 0,
    n_head_dim: 512,
    n_value_dim: 256,
    n_rot: 0,
    n_out_group: 0,
    n_lora_q: 1536,
    n_lora_o: 0,
    n_expert: 288,
    n_expert_used: 8,
    n_expert_shared: 1,
    n_ff_exp: 2048,
    n_ff_dense: 12288,
    n_ff_shexp: 0,
    n_hash_layer: 0,
    n_swa: 0,
    n_swa_period: 0,
    n_indexer_head: 32,
    n_indexer_head_dim: 128,
    n_indexer_top_k: 2048,
    n_hc: 4,
    n_hc_sinkhorn_iter: 20,
    n_nextn_predict: 1,
    n_leading_dense: 3,
    n_kv_lora: 512,
    n_key_mla: 256,
    n_value_mla: 256,
    n_swa_head: 0,
    n_swa_kv_lora: 0,
    n_swa_key_mla: 0,
    n_full_attn_count: 0,
    n_kda_head_dim: 128,
    n_ssm_conv: 4,
    use_rope: false,
    use_qk_norm: false,
    rms_eps: 1.0e-5,
    kda_l2_eps: 1.0e-6,
    kda_gate_clamp_min: -5.0,
    hc_eps: 1.0e-6,
    expert_weight_scale: 2.5,
    swiglu_clamp_exp: 10.0,
    rope_freq_base: 0.0,
    rope_freq_base_swa: 0.0,
    rope_scale_factor: 1.0,
    rope_yarn_beta_fast: 0.0,
    rope_yarn_beta_slow: 0.0,
    compress_rope_freq_base: 0.0,
    rope_orig_ctx: 1_048_576,
};
