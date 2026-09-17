//! Ling-3.0-flash-VL MQ-Q5 artifact contract and per-layer native bind plan.
//!
//! The language tower is `bailingmoe3`: 42 blocks that alternate 35 KDA
//! linear-attention blocks with 7 MLA blocks, over a 512-way routed MoE with
//! one always-active shared expert. The vision tower ships as a separate
//! `clip` mmproj whose projector is the Qwen3-VL merger.

use std::collections::BTreeSet;
use std::path::Path;

use crate::gguf::{GgufError, GgufFile};
use crate::tensors::{TensorError, TensorInfo, TensorInventory};

pub(crate) const LAYERS: u32 = 42;
/// A block is MLA when it closes a group of six; the other 35 are KDA.
pub(crate) const LAYER_GROUP: u32 = 6;
pub(crate) const EMBED: u64 = 2560;
pub(crate) const VOCAB: u64 = 157_184;
pub(crate) const HEADS: u64 = 32;
pub(crate) const KDA_HEAD_DIM: u64 = 128;
pub(crate) const KDA_DIM: u64 = HEADS * KDA_HEAD_DIM;
pub(crate) const CONV_KERNEL: u64 = 4;
pub(crate) const KV_LORA: u64 = 512;
pub(crate) const ROPE_DIM: u64 = 64;
pub(crate) const KEY_MLA: u64 = 192;
pub(crate) const VALUE_MLA: u64 = 128;
/// `attention.key_length` is the stored latent row: kv_lora + rope.
pub(crate) const KEY_LENGTH: u64 = KV_LORA + ROPE_DIM;
pub(crate) const DENSE_LAYERS: u32 = 2;
pub(crate) const DENSE_FF: u64 = 6144;
pub(crate) const EXPERT_FF: u64 = 768;
pub(crate) const EXPERTS: u64 = 512;
pub(crate) const EXPERTS_USED: u64 = 8;
pub(crate) const EXPERT_GROUPS: u32 = 8;
pub(crate) const EXPERT_GROUPS_USED: u32 = 4;
pub(crate) const EXPERT_WEIGHT_SCALE: f32 = 2.5;
pub(crate) const CONTEXT: u32 = 131_072;
/// Static YaRN factor 2 from the official 256K serving recipe.
pub(crate) const YARN_CONTEXT: u32 = CONTEXT * 2;
pub(crate) const RMS_EPS: f32 = 1e-6;
pub(crate) const ROPE_FREQ_BASE: f32 = 6_000_000.0;
pub(crate) const KDA_GATE_LOWER_BOUND: f32 = -5.0;

/// Interleaved M-RoPE halves: 8 temporal + 12 height + 12 width = 32 pairs.
pub(crate) const MROPE_SECTIONS: [u32; 4] = [8, 12, 12, 0];
pub(crate) const IMAGE_TOKEN: u32 = 157_157;
pub(crate) const VIDEO_TOKEN: u32 = 156_909;
pub(crate) const VISION_START_TOKEN: u32 = 157_158;
pub(crate) const VISION_END_TOKEN: u32 = 157_159;

const TENSORS: usize = 917;
const SHARDS: u32 = 3;

pub(crate) const VISION_LAYERS: u32 = 27;
pub(crate) const VISION_EMBED: u64 = 1152;
pub(crate) const VISION_HEADS: u64 = 16;
pub(crate) const VISION_FF: u64 = 4304;
pub(crate) const VISION_PATCH: u32 = 16;
pub(crate) const VISION_MERGE: u32 = 2;
pub(crate) const VISION_POSITIONS: u64 = 2304;
pub(crate) const VISION_IMAGE_SIZE: u32 = 768;
pub(crate) const VISION_MERGER_IN: u64 = VISION_EMBED * (VISION_MERGE * VISION_MERGE) as u64;
const VISION_TENSORS: usize = 334;

const F32: u32 = 0;
const Q8_0: u32 = 8;
const Q4_K: u32 = 12;
const Q5_K: u32 = 13;
const BF16: u32 = 30;

#[derive(Debug)]
pub enum Ling3VlError {
    Metadata(GgufError),
    Inventory(TensorError),
    Contract(String),
}

impl std::fmt::Display for Ling3VlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metadata(e) => write!(f, "Ling-3.0-flash-VL metadata: {e}"),
            Self::Inventory(e) => write!(f, "Ling-3.0-flash-VL inventory: {e}"),
            Self::Contract(e) => write!(f, "Ling-3.0-flash-VL contract: {e}"),
        }
    }
}

impl std::error::Error for Ling3VlError {}

impl From<GgufError> for Ling3VlError {
    fn from(e: GgufError) -> Self {
        Self::Metadata(e)
    }
}

impl From<TensorError> for Ling3VlError {
    fn from(e: TensorError) -> Self {
        Self::Inventory(e)
    }
}

fn mismatch(key: &str) -> Ling3VlError {
    Ling3VlError::Contract(key.into())
}

/// C `ds4_ling3vl_layer_is_kda`: every block that does not close a group.
pub fn layer_is_kda(il: u32) -> bool {
    il < LAYERS && !(il + 1).is_multiple_of(LAYER_GROUP)
}

#[derive(Clone, Copy, Debug)]
pub struct Ling3VlLayer {
    index: u32,
}

impl Ling3VlLayer {
    pub fn is_kda(self) -> bool {
        layer_is_kda(self.index)
    }

    pub fn is_dense_ffn(self) -> bool {
        self.index < DENSE_LAYERS
    }

    /// Routed and shared SwiGLU clamps, respectively; zero disables clipping.
    /// The two schedules differ: the shared expert starts clamping one block
    /// earlier and widens again for the last two blocks.
    pub fn swiglu_clamps(self) -> (f32, f32) {
        let routed = if self.index >= 35 { 4.0 } else { 0.0 };
        let shared = match self.index {
            34..=39 => 5.0,
            40..=41 => 7.0,
            _ => 0.0,
        };
        (routed, shared)
    }

    /// Q5_K protects the edge MoE blocks; the interior gate/up pair is Q4_K.
    fn expert_gate_up_type(self) -> u32 {
        match self.index {
            2..=6 | 34..=41 => Q5_K,
            _ => Q4_K,
        }
    }
}

#[derive(Debug)]
pub struct Ling3VlPlan {
    inventory: TensorInventory,
    bindings: Vec<usize>,
    layers: Vec<Ling3VlLayer>,
}

struct Spec {
    name: String,
    typ: u32,
    dims: Vec<u64>,
}

fn layout_specs(specs: Vec<Spec>) -> Vec<crate::layout::LayoutSpec> {
    specs
        .into_iter()
        .map(|s| {
            let mut dim = [0; 8];
            dim[..s.dims.len()].copy_from_slice(&s.dims);
            crate::layout::LayoutSpec {
                name: s.name,
                class: crate::layout::TypeClass::Exact(s.typ),
                ndim: s.dims.len() as u32,
                dim,
            }
        })
        .collect()
}

impl Ling3VlPlan {
    pub(crate) fn validate_inventory(inv: &TensorInventory) -> Result<(), Ling3VlError> {
        check_shards(inv)?;
        check_specs(inv, specs()).map(|_| ())
    }

    /// Shape validation used by the host before native allocation.
    pub(crate) fn validate(g: &GgufFile) -> Result<(), crate::validate::ValidateError> {
        check_metadata(g)
            .map_err(|e| crate::validate::ValidateError::TokenKey("ling3vl", e.to_string()))
    }

    /// Publish the same semantic tensor contract to the host layout/bind seam.
    pub(crate) fn layouts() -> Vec<crate::layout::LayoutSpec> {
        layout_specs(specs())
    }

    /// Read metadata through mmap and resolve all shards without copying
    /// weights. This accepts the three-shard language artifact only, never
    /// the mmproj sidecar.
    pub fn inspect(path: &Path) -> Result<Self, Ling3VlError> {
        let first = GgufFile::open(path)?;
        check_metadata(&first)?;
        let inventory = TensorInventory::from_file(path, &first)?;
        check_shards(&inventory)?;
        let bindings = check_specs(&inventory, specs())?;
        Ok(Self {
            inventory,
            bindings,
            layers: (0..LAYERS).map(|index| Ling3VlLayer { index }).collect(),
        })
    }

    pub fn layers(&self) -> &[Ling3VlLayer] {
        &self.layers
    }

    /// Stable semantic order, independent of the source shard's tensor order.
    pub fn bindings(&self) -> impl Iterator<Item = &TensorInfo> {
        self.bindings.iter().map(|&i| &self.inventory.tensors[i])
    }

    pub fn payload_bytes(&self) -> u64 {
        self.bindings().map(|t| t.bytes).sum()
    }
}

fn check_shards(inv: &TensorInventory) -> Result<(), Ling3VlError> {
    if inv.shards.len() != SHARDS as usize {
        return Err(mismatch("expected three main shards"));
    }
    for (index, shard) in inv.shards.iter().enumerate() {
        let g = GgufFile::open(&shard.path)?;
        if g.get_u16("split.no") != Some(index as u16)
            || g.split_count() != SHARDS
            || g.get_token_id("split.tensors.count") != Some(TENSORS as i32)
        {
            return Err(mismatch(&format!(
                "{}: split identity",
                shard.path.display()
            )));
        }
    }
    Ok(())
}

fn check_metadata(g: &GgufFile) -> Result<(), Ling3VlError> {
    if g.get_string("general.architecture") != Some(b"bailingmoe3") {
        return Err(mismatch("general.architecture"));
    }
    for (key, expected) in [
        ("tokenizer.ggml.model", b"gpt2".as_slice()),
        ("tokenizer.ggml.pre", b"bailingmoe2".as_slice()),
    ] {
        if g.get_string(key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    if g.split_count() != SHARDS || g.get_u16("split.no") != Some(0) {
        return Err(mismatch("expected first of three main shards"));
    }
    for (suffix, expected) in [
        ("block_count", LAYERS),
        ("context_length", CONTEXT),
        ("embedding_length", EMBED as u32),
        ("feed_forward_length", DENSE_FF as u32),
        ("vocab_size", VOCAB as u32),
        ("attention.head_count", HEADS as u32),
        ("attention.key_length", KEY_LENGTH as u32),
        ("attention.value_length", VALUE_MLA as u32),
        ("attention.key_length_mla", KEY_MLA as u32),
        ("attention.value_length_mla", VALUE_MLA as u32),
        ("attention.kv_lora_rank", KV_LORA as u32),
        ("rope.dimension_count", ROPE_DIM as u32),
        ("ssm.conv_kernel", CONV_KERNEL as u32),
        ("kda.head_dim", KDA_HEAD_DIM as u32),
        ("expert_count", EXPERTS as u32),
        ("expert_used_count", EXPERTS_USED as u32),
        ("expert_group_count", EXPERT_GROUPS),
        ("expert_group_used_count", EXPERT_GROUPS_USED),
        ("expert_gating_func", 2), // GGUF sigmoid routing, not softmax.
        ("expert_feed_forward_length", EXPERT_FF as u32),
        ("expert_shared_feed_forward_length", EXPERT_FF as u32),
        ("expert_shared_count", 1),
        ("leading_dense_block_count", DENSE_LAYERS),
        ("vision.image_token_id", IMAGE_TOKEN),
        ("vision.video_token_id", VIDEO_TOKEN),
        ("vision.start_token_id", VISION_START_TOKEN),
        ("vision.end_token_id", VISION_END_TOKEN),
    ] {
        let key = format!("bailingmoe3.{suffix}");
        if g.get_u32(&key) != Some(expected) {
            return Err(mismatch(&key));
        }
    }
    for (suffix, expected) in [
        ("rope.freq_base", ROPE_FREQ_BASE),
        ("expert_weights_scale", EXPERT_WEIGHT_SCALE),
        ("attention.layer_norm_rms_epsilon", RMS_EPS),
        ("kda.gate_lower_bound", KDA_GATE_LOWER_BOUND),
    ] {
        let key = format!("bailingmoe3.{suffix}");
        if g.get_f32_compat(&key) != Some(expected) {
            return Err(mismatch(&key));
        }
    }
    for (suffix, expected) in [
        ("expert_weights_norm", true),
        // A non-safe gate would need an unbounded softplus decay instead.
        ("kda.safe_gate", true),
    ] {
        let key = format!("bailingmoe3.{suffix}");
        if g.get_bool(&key) != Some(expected) {
            return Err(mismatch(&key));
        }
    }
    // Per-block KV head counts carry the hybrid schedule: 1 selects MLA and
    // 0 selects the recurrent KDA block.
    let key = "bailingmoe3.attention.head_count_kv";
    let arr = g.get_array(key).ok_or_else(|| mismatch(key))?;
    let expected: Vec<_> = (0..LAYERS).map(|il| u32::from(!layer_is_kda(il))).collect();
    if g.array_le_u32s(&arr)? != expected {
        return Err(mismatch(key));
    }
    let key = "bailingmoe3.rope.dimension_sections";
    let arr = g.get_array(key).ok_or_else(|| mismatch(key))?;
    if g.array_le_u32s(&arr)? != MROPE_SECTIONS {
        return Err(mismatch(key));
    }
    for (suffix, pick) in [("swiglu_clamp_exp", 0usize), ("swiglu_clamp_shexp", 1usize)] {
        let key = format!("bailingmoe3.{suffix}");
        let arr = g.get_array(&key).ok_or_else(|| mismatch(&key))?;
        let expected: Vec<_> = (0..LAYERS)
            .map(|index| {
                let clamps = Ling3VlLayer { index }.swiglu_clamps();
                if pick == 0 {
                    clamps.0
                } else {
                    clamps.1
                }
            })
            .collect();
        if g.array_f32s(&arr)? != expected {
            return Err(mismatch(&key));
        }
    }
    let tokens = g
        .get_array("tokenizer.ggml.tokens")
        .ok_or_else(|| mismatch("vocabulary"))?;
    if tokens.typ != crate::gguf::GGUF_VALUE_STRING
        || tokens.len != VOCAB
        || g.get_token_id("tokenizer.ggml.bos_token_id") != Some(156_891)
        || g.get_token_id("tokenizer.ggml.eos_token_id") != Some(156_895)
    {
        return Err(mismatch("vocabulary/BOS/EOS"));
    }
    Ok(())
}

fn specs() -> Vec<Spec> {
    let mut out = Vec::with_capacity(TENSORS);
    let mut add = |name: String, typ, dims: &[u64]| {
        out.push(Spec {
            name,
            typ,
            dims: dims.to_vec(),
        });
    };
    add("token_embd.weight".into(), Q8_0, &[EMBED, VOCAB]);
    add("output_norm.weight".into(), F32, &[EMBED]);
    add("output.weight".into(), BF16, &[EMBED, VOCAB]);
    for index in 0..LAYERS {
        let layer = Ling3VlLayer { index };
        let prefix = format!("blk.{index}");
        for name in ["attn_norm.weight", "ffn_norm.weight"] {
            add(format!("{prefix}.{name}"), F32, &[EMBED]);
        }
        if layer.is_kda() {
            // q/k/v are one full-rank projection each; the conv1d, decay,
            // gate and beta control path stays outside the quantized region.
            for name in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
                add(format!("{prefix}.{name}"), BF16, &[EMBED, KDA_DIM]);
            }
            for name in ["ssm_f_a.weight", "ssm_g_a.weight"] {
                add(format!("{prefix}.{name}"), BF16, &[EMBED, KDA_DIM]);
            }
            add(format!("{prefix}.ssm_beta.weight"), BF16, &[EMBED, HEADS]);
            for name in [
                "ssm_conv1d_q.weight",
                "ssm_conv1d_k.weight",
                "ssm_conv1d_v.weight",
            ] {
                add(format!("{prefix}.{name}"), F32, &[CONV_KERNEL, 1, KDA_DIM]);
            }
            // ssm_a is already exp(A_log); the kernel must not exponentiate.
            add(format!("{prefix}.ssm_a"), F32, &[1, HEADS]);
            add(format!("{prefix}.ssm_dt.bias"), F32, &[KDA_DIM]);
            add(format!("{prefix}.ssm_norm.weight"), F32, &[KDA_HEAD_DIM]);
            add(
                format!("{prefix}.attn_output.weight"),
                BF16,
                &[KDA_DIM, EMBED],
            );
        } else {
            add(
                format!("{prefix}.attn_q.weight"),
                BF16,
                &[EMBED, HEADS * KEY_MLA],
            );
            add(
                format!("{prefix}.attn_kv_a_mqa.weight"),
                BF16,
                &[EMBED, KEY_LENGTH],
            );
            add(format!("{prefix}.attn_kv_a_norm.weight"), F32, &[KV_LORA]);
            add(
                format!("{prefix}.attn_k_b.weight"),
                BF16,
                &[KEY_MLA - ROPE_DIM, KV_LORA, HEADS],
            );
            add(
                format!("{prefix}.attn_v_b.weight"),
                BF16,
                &[KV_LORA, VALUE_MLA, HEADS],
            );
            add(format!("{prefix}.attn_gate.weight"), BF16, &[EMBED, HEADS]);
            add(
                format!("{prefix}.attn_output.weight"),
                BF16,
                &[HEADS * VALUE_MLA, EMBED],
            );
        }
        if layer.is_dense_ffn() {
            for name in ["ffn_gate.weight", "ffn_up.weight"] {
                add(format!("{prefix}.{name}"), BF16, &[EMBED, DENSE_FF]);
            }
            add(
                format!("{prefix}.ffn_down.weight"),
                BF16,
                &[DENSE_FF, EMBED],
            );
            continue;
        }
        add(
            format!("{prefix}.ffn_gate_inp.weight"),
            F32,
            &[EMBED, EXPERTS],
        );
        add(format!("{prefix}.exp_probs_b.bias"), F32, &[EXPERTS]);
        let gate_up = layer.expert_gate_up_type();
        for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
            add(
                format!("{prefix}.{name}"),
                gate_up,
                &[EMBED, EXPERT_FF, EXPERTS],
            );
        }
        add(
            format!("{prefix}.ffn_down_exps.weight"),
            Q5_K,
            &[EXPERT_FF, EMBED, EXPERTS],
        );
        // The shared expert is always active, so it stays unquantized.
        for name in ["ffn_gate_shexp.weight", "ffn_up_shexp.weight"] {
            add(format!("{prefix}.{name}"), BF16, &[EMBED, EXPERT_FF]);
        }
        add(
            format!("{prefix}.ffn_down_shexp.weight"),
            BF16,
            &[EXPERT_FF, EMBED],
        );
    }
    out
}

fn check_specs(inv: &TensorInventory, specs: Vec<Spec>) -> Result<Vec<usize>, Ling3VlError> {
    if inv.tensors.len() != specs.len() {
        return Err(mismatch("tensor count"));
    }
    let mut names = BTreeSet::new();
    for t in &inv.tensors {
        if !names.insert(&t.name) {
            return Err(mismatch(&format!("duplicate {}", t.name)));
        }
    }
    let mut bindings = Vec::with_capacity(specs.len());
    for spec in specs {
        let index = inv
            .find_index(&spec.name)
            .ok_or_else(|| mismatch(&spec.name))?;
        let t = &inv.tensors[index];
        if t.typ != spec.typ
            || t.ndim as usize != spec.dims.len()
            || t.dim[..spec.dims.len()] != spec.dims
        {
            return Err(mismatch(&spec.name));
        }
        bindings.push(index);
    }
    for shard in 0..inv.shards.len() {
        let mut spans: Vec<_> = inv
            .tensors
            .iter()
            .filter(|t| t.shard as usize == shard)
            .map(|t| (t.abs_offset, t.abs_offset.checked_add(t.bytes)))
            .collect();
        spans.sort_unstable();
        if spans.iter().any(|s| s.1.is_none()) || spans.windows(2).any(|s| s[0].1.unwrap() > s[1].0)
        {
            return Err(mismatch("overlapping tensor payloads"));
        }
    }
    Ok(bindings)
}

/// The BF16 vision tower and its Qwen3-VL merger, published separately so the
/// language artifact stays loadable without image support.
#[derive(Debug)]
pub struct Ling3VlVisionPlan {
    inventory: TensorInventory,
    bindings: Vec<usize>,
}

impl Ling3VlVisionPlan {
    pub fn inspect(path: &Path) -> Result<Self, Ling3VlError> {
        let first = GgufFile::open(path)?;
        check_vision_metadata(&first)?;
        let inventory = TensorInventory::from_file(path, &first)?;
        let bindings = check_specs(&inventory, vision_specs())?;
        Ok(Self {
            inventory,
            bindings,
        })
    }

    pub fn bindings(&self) -> impl Iterator<Item = &TensorInfo> {
        self.bindings.iter().map(|&i| &self.inventory.tensors[i])
    }

    pub fn payload_bytes(&self) -> u64 {
        self.bindings().map(|t| t.bytes).sum()
    }
}

fn check_vision_metadata(g: &GgufFile) -> Result<(), Ling3VlError> {
    if g.split_count() > 1 || g.get_u16("split.no").is_some_and(|v| v != 0) {
        return Err(mismatch("sidecar split identity"));
    }
    for (key, expected) in [
        ("general.architecture", b"clip".as_slice()),
        ("general.type", b"mmproj".as_slice()),
        ("clip.projector_type", b"qwen3vl_merger".as_slice()),
    ] {
        if g.get_string(key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    for (key, expected) in [
        ("clip.vision.block_count", VISION_LAYERS),
        ("clip.vision.embedding_length", VISION_EMBED as u32),
        ("clip.vision.feed_forward_length", VISION_FF as u32),
        ("clip.vision.attention.head_count", VISION_HEADS as u32),
        ("clip.vision.patch_size", VISION_PATCH),
        ("clip.vision.spatial_merge_size", VISION_MERGE),
        ("clip.vision.image_size", VISION_IMAGE_SIZE),
        // out_hidden_size is stale upstream; the emitted dim is the real one.
        ("clip.vision.projection_dim", EMBED as u32),
    ] {
        if g.get_u32(key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    if g.get_bool("clip.has_vision_encoder") != Some(true)
        || g.get_bool("clip.use_gelu") != Some(true)
    {
        return Err(mismatch("clip encoder flags"));
    }
    // This checkpoint has no deepstack taps: every layer feeds only the next.
    let key = "clip.vision.is_deepstack_layers";
    let arr = g.get_array(key).ok_or_else(|| mismatch(key))?;
    if g.array_bools(&arr)? != vec![false; VISION_LAYERS as usize] {
        return Err(mismatch(key));
    }
    Ok(())
}

fn vision_specs() -> Vec<Spec> {
    let mut out = Vec::with_capacity(VISION_TENSORS);
    let mut add = |name: String, typ, dims: &[u64]| {
        out.push(Spec {
            name,
            typ,
            dims: dims.to_vec(),
        });
    };
    // Conv3D patch embedding, split along the temporal axis by the converter.
    let patch = [
        u64::from(VISION_PATCH),
        u64::from(VISION_PATCH),
        3,
        VISION_EMBED,
    ];
    add("v.patch_embd.weight".into(), F32, &patch);
    add("v.patch_embd.weight.1".into(), F32, &patch);
    add("v.patch_embd.bias".into(), F32, &[VISION_EMBED]);
    add(
        "v.position_embd.weight".into(),
        F32,
        &[VISION_EMBED, VISION_POSITIONS],
    );
    for name in ["v.post_ln.weight", "v.post_ln.bias"] {
        add(name.into(), F32, &[VISION_EMBED]);
    }
    // disable_merger_proj: the merger keeps only its norm, and the real
    // projection is the top-level linear_proj pair folded in as mm.0/mm.2.
    add("mm.0.weight".into(), BF16, &[VISION_MERGER_IN, EMBED]);
    add("mm.0.bias".into(), F32, &[EMBED]);
    add("mm.2.weight".into(), BF16, &[EMBED, EMBED]);
    add("mm.2.bias".into(), F32, &[EMBED]);
    for index in 0..VISION_LAYERS {
        let prefix = format!("v.blk.{index}");
        for name in ["ln1", "ln2"] {
            for field in ["weight", "bias"] {
                add(format!("{prefix}.{name}.{field}"), F32, &[VISION_EMBED]);
            }
        }
        add(
            format!("{prefix}.attn_qkv.weight"),
            BF16,
            &[VISION_EMBED, 3 * VISION_EMBED],
        );
        add(format!("{prefix}.attn_qkv.bias"), F32, &[3 * VISION_EMBED]);
        add(
            format!("{prefix}.attn_out.weight"),
            BF16,
            &[VISION_EMBED, VISION_EMBED],
        );
        add(format!("{prefix}.attn_out.bias"), F32, &[VISION_EMBED]);
        add(
            format!("{prefix}.ffn_up.weight"),
            BF16,
            &[VISION_EMBED, VISION_FF],
        );
        add(format!("{prefix}.ffn_up.bias"), F32, &[VISION_FF]);
        add(
            format!("{prefix}.ffn_down.weight"),
            BF16,
            &[VISION_FF, VISION_EMBED],
        );
        add(format!("{prefix}.ffn_down.bias"), F32, &[VISION_EMBED]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_schedule_matches_the_artifact() {
        let mla: Vec<_> = (0..LAYERS).filter(|&il| !layer_is_kda(il)).collect();
        assert_eq!(mla, vec![5, 11, 17, 23, 29, 35, 41]);
        assert_eq!((0..LAYERS).filter(|&il| layer_is_kda(il)).count(), 35);
    }

    #[test]
    fn tensor_table_covers_the_published_inventory() {
        let specs = specs();
        assert_eq!(specs.len(), TENSORS);
        let counts = |typ| specs.iter().filter(|s| s.typ == typ).count();
        assert_eq!(counts(BF16), 414);
        assert_eq!(counts(F32), 382);
        assert_eq!(counts(Q4_K), 54);
        assert_eq!(counts(Q5_K), 66);
        assert_eq!(counts(Q8_0), 1);
    }

    #[test]
    fn vision_table_covers_the_published_mmproj() {
        assert_eq!(vision_specs().len(), VISION_TENSORS);
    }

    #[test]
    fn clamp_schedule_matches_the_config_lists() {
        let routed: Vec<_> = (0..LAYERS)
            .map(|index| Ling3VlLayer { index }.swiglu_clamps().0)
            .collect();
        let shared: Vec<_> = (0..LAYERS)
            .map(|index| Ling3VlLayer { index }.swiglu_clamps().1)
            .collect();
        assert_eq!(routed.iter().filter(|&&v| v == 4.0).count(), 7);
        assert_eq!(routed[34], 0.0);
        assert_eq!(shared[33], 0.0);
        assert_eq!(shared[34], 5.0);
        assert_eq!(shared[39], 5.0);
        assert_eq!(shared[40], 7.0);
        assert_eq!(shared[41], 7.0);
    }
}
