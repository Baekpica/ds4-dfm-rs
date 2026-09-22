//! MiMo-V2.6-Flash-RL mixed artifact and per-layer execution contract.
//! Metadata includes three dense prediction blocks after the 48-layer trunk.
use std::collections::BTreeSet;
use std::path::Path;

use crate::gguf::{GgufError, GgufFile};
use crate::layout::{LayoutSpec, TypeClass};
use crate::tensors::{TensorError, TensorInfo, TensorInventory};

const TRUNK: u32 = 48;
const BLOCKS: u32 = 51;
const TENSORS: usize = 508;
const SHARDS: u32 = 4;

pub const MIXED_SHARDS: u32 = SHARDS;
pub const LANGUAGE_TENSORS: usize = TENSORS;
pub const TRUNK_LAYERS: u32 = TRUNK;
pub const MTP_BLOCKS: u32 = BLOCKS - TRUNK;
/// P1 serving context. Indexing still reaches `INDEX_LIMIT`.
pub const QUALIFIED_CONTEXT: u32 = 262_144;
pub const INDEX_LIMIT: u32 = 1_048_576;
pub const SWA_WINDOW: u32 = 128;
pub const PREFILL_CAP: u32 = 512;
pub const PATCH: u32 = 16;
pub const MERGE: u32 = 2;
pub const TEMPORAL: u32 = 2;
pub const IMAGE_MIN_PIXELS: u32 = 8_192;
pub const IMAGE_MAX_PIXELS: u32 = 8_388_608;
pub const AUDIO_GROUP: u32 = 4;
pub const AUDIO_IDS_PER_SEC: f32 = 25.0;
/// Source codec groups four ids, so feature tokens run at 6.25 Hz.
pub const AUDIO_TOKENS_PER_SEC: f32 = AUDIO_IDS_PER_SEC / AUDIO_GROUP as f32;

pub const VISION_START: i32 = 151652;
pub const VISION_END: i32 = 151653;
pub const IMAGE_PAD: i32 = 151655;
pub const VIDEO_PAD: i32 = 151656;
pub const AUDIO_PAD: i32 = 151669;
pub const VIDEO_START: i32 = 151670;
pub const VIDEO_END: i32 = 151671;
pub const AUDIO_START: i32 = 151673;
pub const AUDIO_END: i32 = 151674;

pub const TRIAL_CAP: usize = 4;
const EMBED: u64 = 4096;
const VOCAB: u64 = 152576;
const HEADS: u64 = 64;
const KEY: u64 = 192;
const VALUE: u64 = 128;
const DENSE: u64 = 16384;
const EXPERT_FF: u64 = 2048;
const EXPERTS: u64 = 256;
const F32: u32 = 0;
const Q8: u32 = 8;
const IQ2_XXS: u32 = 16;
const IQ2_XS: u32 = 17;
const FULL: [u32; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];

#[derive(Debug)]
pub struct Mimo2Error(String);
impl std::fmt::Display for Mimo2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MiMo-V2.6 contract: {}", self.0)
    }
}
impl std::error::Error for Mimo2Error {}
impl From<GgufError> for Mimo2Error {
    fn from(e: GgufError) -> Self {
        Self(e.to_string())
    }
}
impl From<TensorError> for Mimo2Error {
    fn from(e: TensorError) -> Self {
        Self(e.to_string())
    }
}
fn mismatch(key: &str) -> Mimo2Error {
    Mimo2Error(key.into())
}

#[derive(Clone, Copy, Debug)]
pub struct Mimo2Layer {
    index: u32,
}
impl Mimo2Layer {
    pub fn new(index: u32) -> Option<Self> {
        (index < BLOCKS).then_some(Self { index })
    }
    pub fn is_prediction(self) -> bool {
        self.index >= TRUNK
    }
    pub fn is_routed(self) -> bool {
        (1..TRUNK).contains(&self.index)
    }
    pub fn sliding_window(self) -> Option<u32> {
        (!FULL.contains(&self.index)).then_some(128)
    }
    pub fn kv_heads(self) -> u32 {
        if self.sliding_window().is_some() {
            8
        } else {
            4
        }
    }
    pub fn rope_base(self) -> f32 {
        if self.sliding_window().is_some() {
            10_000.0
        } else {
            10_000_000.0
        }
    }
    pub fn rotary_dims(self) -> u32 {
        64
    }
    pub fn key_dims(self) -> u32 {
        KEY as u32
    }
    pub fn value_dims(self) -> u32 {
        VALUE as u32
    }
    pub fn value_scale(self) -> f32 {
        0.707
    }
    /// Converter already deinterleaves TP=4 source rows into Q, K, V blocks.
    pub fn qkv_rows(self) -> (u64, u64, u64) {
        (
            HEADS * KEY,
            u64::from(self.kv_heads()) * KEY,
            u64::from(self.kv_heads()) * VALUE,
        )
    }
}

#[derive(Debug)]
pub struct Mimo2Plan {
    inventory: TensorInventory,
    bindings: Vec<usize>,
}
impl Mimo2Plan {
    /// Directory-only mmap validation; no device allocation or inference.
    pub fn inspect(path: &Path) -> Result<Self, Mimo2Error> {
        let first = GgufFile::open(path)?;
        Self::check_metadata(&first)?;
        let inventory = TensorInventory::from_file(path, &first)?;
        let bindings = check_inventory(&inventory)?;
        Ok(Self {
            inventory,
            bindings,
        })
    }
    pub fn layers(&self) -> impl Iterator<Item = Mimo2Layer> {
        (0..TRUNK).map(|index| Mimo2Layer { index })
    }
    pub fn bindings(&self) -> impl Iterator<Item = &TensorInfo> {
        self.bindings.iter().map(|&i| &self.inventory.tensors[i])
    }
    pub fn payload_bytes(&self) -> u64 {
        self.bindings().map(|t| t.bytes).sum()
    }
    pub(crate) fn validate_inventory(inv: &TensorInventory) -> Result<(), Mimo2Error> {
        check_inventory(inv).map(|_| ())
    }
    pub(crate) fn validate(g: &GgufFile) -> Result<(), crate::validate::ValidateError> {
        Self::check_metadata(g)
            .map_err(|e| crate::validate::ValidateError::TokenKey("mimo2", e.to_string()))
    }
    pub fn check_metadata(g: &GgufFile) -> Result<(), Mimo2Error> {
        for (key, expected) in [
            ("general.architecture", b"mimo2".as_slice()),
            ("tokenizer.ggml.model", b"gpt2"),
            ("tokenizer.ggml.pre", b"qwen2"),
        ] {
            if g.get_string(key) != Some(expected) {
                return Err(mismatch(key));
            }
        }
        if g.split_count() != SHARDS || g.get_u16("split.no") != Some(0) {
            return Err(mismatch("expected first of four shards"));
        }
        for (suffix, expected) in [
            ("block_count", BLOCKS),
            ("context_length", 1048576),
            ("embedding_length", EMBED as u32),
            ("feed_forward_length", DENSE as u32),
            ("attention.head_count", HEADS as u32),
            ("attention.key_length", KEY as u32),
            ("attention.value_length", VALUE as u32),
            ("attention.sliding_window", 128),
            ("rope.dimension_count", 64),
            ("expert_count", EXPERTS as u32),
            ("expert_used_count", 8),
            ("expert_group_count", 1),
            ("expert_group_used_count", 1),
            ("expert_gating_func", 2),
            ("expert_feed_forward_length", EXPERT_FF as u32),
            ("nextn_predict_layers", 3),
        ] {
            let key = format!("mimo2.{suffix}");
            if g.get_u32(&key) != Some(expected) {
                return Err(mismatch(&key));
            }
        }
        for (suffix, expected) in [
            ("rope.freq_base", 10_000_000.0),
            ("rope.freq_base_swa", 10_000.0),
            ("attention.value_scale", 0.707),
            ("attention.layer_norm_rms_epsilon", 1e-6),
        ] {
            let key = format!("mimo2.{suffix}");
            if g.get_f32_compat(&key) != Some(expected) {
                return Err(mismatch(&key));
            }
        }
        for (key, expected) in [
            (
                "mimo2.attention.head_count_kv",
                (0..BLOCKS)
                    .map(|index| Mimo2Layer { index }.kv_heads())
                    .collect::<Vec<_>>(),
            ),
            (
                "mimo2.attention.sliding_window_pattern",
                (0..BLOCKS)
                    .map(|index| u32::from(Mimo2Layer { index }.sliding_window().is_some()))
                    .collect(),
            ),
        ] {
            let array = g.get_array(key).ok_or_else(|| mismatch(key))?;
            if g.array_le_u32s(&array)? != expected {
                return Err(mismatch(key));
            }
        }
        let tokens = g
            .get_array("tokenizer.ggml.tokens")
            .ok_or_else(|| mismatch("vocabulary"))?;
        if tokens.typ != crate::gguf::GGUF_VALUE_STRING
            || tokens.len != VOCAB
            || g.get_token_id("tokenizer.ggml.eos_token_id") != Some(151645)
            || g.get_token_id("tokenizer.ggml.padding_token_id") != Some(151643)
            || g.get_token_id("tokenizer.ggml.bos_token_id").is_some()
        {
            return Err(mismatch("vocabulary/EOS/PAD/no BOS"));
        }
        Ok(())
    }
    pub(crate) fn layouts() -> Vec<LayoutSpec> {
        let mut specs = Vec::with_capacity(TENSORS);
        let mut add = |name: String, typ, dims: &[u64]| {
            let mut dim = [0; 8];
            dim[..dims.len()].copy_from_slice(dims);
            specs.push(LayoutSpec {
                name,
                class: TypeClass::Exact(typ),
                ndim: dims.len() as u32,
                dim,
            });
        };
        for name in ["token_embd.weight", "output.weight"] {
            add(name.into(), Q8, &[EMBED, VOCAB]);
        }
        add("output_norm.weight".into(), F32, &[EMBED]);
        for index in 0..BLOCKS {
            let layer = Mimo2Layer { index };
            let prefix = format!("blk.{index}");
            for name in ["attn_norm.weight", "ffn_norm.weight"] {
                add(format!("{prefix}.{name}"), F32, &[EMBED]);
            }
            let (q, k, v) = layer.qkv_rows();
            add(format!("{prefix}.attn_qkv.weight"), Q8, &[EMBED, q + k + v]);
            add(
                format!("{prefix}.attn_output.weight"),
                Q8,
                &[HEADS * VALUE, EMBED],
            );
            if layer.sliding_window().is_some() {
                add(format!("{prefix}.attn_sinks.weight"), F32, &[HEADS]);
            }
            if layer.is_routed() {
                add(
                    format!("{prefix}.ffn_gate_inp.weight"),
                    F32,
                    &[EMBED, EXPERTS],
                );
                add(format!("{prefix}.exp_probs_b.bias"), F32, &[EXPERTS]);
                for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
                    add(
                        format!("{prefix}.{name}"),
                        IQ2_XXS,
                        &[EMBED, EXPERT_FF, EXPERTS],
                    );
                }
                add(
                    format!("{prefix}.ffn_down_exps.weight"),
                    IQ2_XS,
                    &[EXPERT_FF, EMBED, EXPERTS],
                );
            } else {
                for name in ["ffn_gate.weight", "ffn_up.weight"] {
                    add(format!("{prefix}.{name}"), Q8, &[EMBED, DENSE]);
                }
                add(format!("{prefix}.ffn_down.weight"), Q8, &[DENSE, EMBED]);
            }
            if layer.is_prediction() {
                add(
                    format!("{prefix}.nextn.eh_proj.weight"),
                    Q8,
                    &[2 * EMBED, EMBED],
                );
                for name in [
                    "nextn.enorm.weight",
                    "nextn.hnorm.weight",
                    "layer_output_norm.weight",
                ] {
                    add(format!("{prefix}.{name}"), F32, &[EMBED]);
                }
            }
        }
        specs
    }
}
fn check_inventory(inv: &TensorInventory) -> Result<Vec<usize>, Mimo2Error> {
    if inv.shards.len() != SHARDS as usize || inv.tensors.len() != TENSORS {
        return Err(mismatch("four shards / 508 tensors required"));
    }
    for (index, shard) in inv.shards.iter().enumerate() {
        let g = GgufFile::open(&shard.path)?;
        if g.get_u16("split.no") != Some(index as u16)
            || g.split_count() != SHARDS
            || g.get_token_id("split.tensors.count") != Some(TENSORS as i32)
        {
            return Err(mismatch("shard identity"));
        }
    }
    let mut names = BTreeSet::new();
    if inv.tensors.iter().any(|t| !names.insert(&t.name)) {
        return Err(mismatch("duplicate tensor"));
    }
    let mut bindings = Vec::with_capacity(TENSORS);
    for spec in Mimo2Plan::layouts() {
        let index = inv
            .find_index(&spec.name)
            .ok_or_else(|| mismatch(&spec.name))?;
        let t = &inv.tensors[index];
        if TypeClass::Exact(t.typ) != spec.class
            || t.ndim != spec.ndim
            || t.dim[..t.ndim as usize] != spec.dim[..spec.ndim as usize]
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

/// Source tokenizer NFC. `encode` calls this before Qwen2 BPE.
pub fn nfc(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfc().collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mimo2Admission {
    pub requested: u32,
    pub effective: u32,
    pub qualified: u32,
    pub index_limit: u32,
    pub swa_window: u32,
    pub dflash_qualified: bool,
    pub context_512k_qualified: bool,
    pub context_1m_qualified: bool,
}

/// Admit `requested` tokens. Values through the source limit stay intact.
/// 512k and 1M are not qualified contexts.
pub fn admit_context(requested: u32) -> Result<Mimo2Admission, Mimo2Error> {
    if requested == 0 || requested > INDEX_LIMIT {
        return Err(mismatch("context"));
    }

    Ok(Mimo2Admission {
        requested,
        effective: requested,
        qualified: QUALIFIED_CONTEXT,
        index_limit: INDEX_LIMIT,
        swa_window: SWA_WINDOW,
        dflash_qualified: false,
        context_512k_qualified: false,
        context_1m_qualified: false,
    })
}

/// Trunk KV rows for one layer. Full layers keep `ctx`; the rest keep
/// the SWA window plus one prefill batch, matching `mimo2_kv_capacity`.
pub fn kv_rows(layer: u32, ctx: u32, cap: u32) -> Option<u32> {
    let layer = Mimo2Layer::new(layer)?;
    if layer.is_prediction() || ctx == 0 || ctx > INDEX_LIMIT || cap == 0 || cap > ctx {
        return None;
    }
    if layer.sliding_window().is_none() {
        return Some(ctx);
    }
    Some((SWA_WINDOW + cap - 1).min(ctx))
}

/// Scratch plus trunk KV bytes from `ds4_mimo2_plan.h`. Draft KV is extra.
pub fn context_bytes(ctx: u32, cap: u32) -> Option<u64> {
    if ctx == 0 || ctx > INDEX_LIMIT || cap == 0 || cap > ctx || cap > 4096 {
        return None;
    }
    let mut raw = 0u64;
    for layer in 0..TRUNK {
        let rows = kv_rows(layer, ctx, cap)?;
        let heads = u64::from(Mimo2Layer::new(layer)?.kv_heads());
        raw = raw.saturating_add(u64::from(rows) * heads * (192 + 128) * 2);
    }
    // 4*embed + Q + K + V + heads + 3*dense + experts + 2*used
    // + 3*used*ff + used*embed + qkv + 2 scalars + full/swa rope pairs.
    const ROWS: u64 = 4 * 4096
        + 64 * 192
        + 8 * 192
        + 8 * 128
        + 64 * 128
        + 3 * 16384
        + 256
        + 2 * 8
        + 3 * 8 * 2048
        + 8 * 4096
        + (64 * 192 + 8 * 192 + 8 * 128)
        + 2
        + 2 * 64;
    let scratch = (u64::from(cap) * ROWS + 64 + VOCAB) * 4;
    Some(raw.saturating_add(scratch))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadKind {
    Image,
    Video,
    Audio,
}

impl PadKind {
    pub fn token(self) -> i32 {
        match self {
            Self::Image => IMAGE_PAD,
            Self::Video => VIDEO_PAD,
            Self::Audio => AUDIO_PAD,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaSpan {
    pub start: u32,
    pub count: u32,
    pub kind: PadKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaPlan {
    pub tokens: Vec<i32>,
    pub spans: Vec<MediaSpan>,
}

impl MediaPlan {
    /// Language positions stay one-dimensional. A span does not add axes.
    pub fn positions(&self) -> Vec<u32> {
        (0..self.tokens.len() as u32).collect()
    }
}

pub fn format_timestamp(seconds: f32) -> String {
    let whole = if seconds.is_finite() && seconds > 0.0 {
        seconds as u32
    } else {
        0
    };
    format!("{:02}:{:02}", whole / 60, whole % 60)
}

/// Still images duplicate one frame into the temporal patch, so `grid_t` is 1.
pub fn visual_tokens(height: u32, width: u32) -> Result<u32, Mimo2Error> {
    let factor = PATCH * MERGE;
    if height == 0 || width == 0 || height % factor != 0 || width % factor != 0 {
        return Err(mismatch("visual grid"));
    }
    let grid_h = height / PATCH;
    let grid_w = width / PATCH;
    let tokens = grid_h * grid_w / (MERGE * MERGE);
    if tokens == 0 {
        return Err(mismatch("visual grid"));
    }
    Ok(tokens)
}

pub fn temporal_groups(frames: u32) -> u32 {
    if frames == 0 {
        return 0;
    }
    frames.div_ceil(TEMPORAL)
}

pub fn smart_resize(height: u32, width: u32) -> Result<(u32, u32), Mimo2Error> {
    let factor = PATCH * MERGE;
    if height == 0 || width == 0 {
        return Err(mismatch("visual grid"));
    }
    let mut height = height;
    let mut width = width;
    let short = height.min(width);
    let long = height.max(width);
    if short < factor {
        let scale = f64::from(factor) / f64::from(short);
        height = (f64::from(height) * scale).round() as u32;
        width = (f64::from(width) * scale).round() as u32;
    } else if f64::from(long) / f64::from(short) > 200.0 {
        return Err(mismatch("aspect"));
    }
    // Python `round(x / factor) * factor` for the resized edges.
    let mut h_bar = ((f64::from(height) / f64::from(factor)).round() as u32) * factor;
    let mut w_bar = ((f64::from(width) / f64::from(factor)).round() as u32) * factor;
    let pixels = u64::from(h_bar) * u64::from(w_bar);
    if pixels > u64::from(IMAGE_MAX_PIXELS) {
        let beta =
            ((u64::from(height) * u64::from(width)) as f64 / f64::from(IMAGE_MAX_PIXELS)).sqrt();
        h_bar = (f64::from(height) / beta / f64::from(factor)).floor() as u32 * factor;
        w_bar = (f64::from(width) / beta / f64::from(factor)).floor() as u32 * factor;
    } else if pixels < u64::from(IMAGE_MIN_PIXELS) {
        let beta =
            (f64::from(IMAGE_MIN_PIXELS) / (u64::from(height) * u64::from(width)) as f64).sqrt();
        h_bar = (f64::from(height) * beta / f64::from(factor)).ceil() as u32 * factor;
        w_bar = (f64::from(width) * beta / f64::from(factor)).ceil() as u32 * factor;
    }
    if h_bar == 0 || w_bar == 0 {
        return Err(mismatch("visual grid"));
    }
    Ok((h_bar, w_bar))
}

fn push_pads(tokens: &mut Vec<i32>, spans: &mut Vec<MediaSpan>, kind: PadKind, count: u32) {
    let start = tokens.len() as u32;
    tokens.extend(std::iter::repeat_n(kind.token(), count as usize));
    spans.push(MediaSpan { start, count, kind });
}

pub fn image_pad_count(data: &[u8]) -> Result<u32, Mimo2Error> {
    let image = image::load_from_memory(data).map_err(|_| mismatch("image"))?;
    let (height, width) = smart_resize(image.height(), image.width())?;
    visual_tokens(height, width)
}

pub fn audio_pad_count(data: &[u8]) -> Result<u32, Mimo2Error> {
    let reader = hound::WavReader::new(std::io::Cursor::new(data)).map_err(|_| mismatch("wav"))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err(mismatch("wav"));
    }
    let samples = u64::from(reader.duration()) * 24_000 / u64::from(spec.sample_rate);
    if samples == 0 {
        return Err(mismatch("wav"));
    }
    let mel = (samples / 240) as u32 + 1;
    let count = audio_feat_len(mel);
    if count == 0 {
        return Err(mismatch("audio"));
    }
    Ok(count)
}

pub fn image_plan(height: u32, width: u32) -> Result<MediaPlan, Mimo2Error> {
    // A still image is duplicated into one temporal group.
    if temporal_groups(TEMPORAL) != 1 {
        return Err(mismatch("temporal"));
    }
    let count = visual_tokens(height, width)?;
    let mut tokens = vec![VISION_START];
    let mut spans = Vec::new();
    push_pads(&mut tokens, &mut spans, PadKind::Image, count);
    tokens.push(VISION_END);
    Ok(MediaPlan { tokens, spans })
}

pub struct VideoPair {
    pub timestamp_s: f32,
    pub timestamp_ids: Vec<i32>,
    pub height: u32,
    pub width: u32,
    pub audio_tokens: u32,
}

fn push_video_pair(plan: &mut MediaPlan, pair: &VideoPair) -> Result<(), Mimo2Error> {
    let count = visual_tokens(pair.height, pair.width)?;
    plan.tokens.extend_from_slice(&pair.timestamp_ids);
    plan.tokens.push(VISION_START);
    push_pads(&mut plan.tokens, &mut plan.spans, PadKind::Video, count);
    plan.tokens.push(VISION_END);
    Ok(())
}

pub fn video_plan(pairs: &[VideoPair]) -> Result<MediaPlan, Mimo2Error> {
    if pairs.is_empty() {
        return Err(mismatch("video"));
    }
    let mut plan = MediaPlan {
        tokens: vec![VIDEO_START],
        spans: Vec::new(),
    };
    for pair in pairs {
        if pair.audio_tokens != 0 {
            return Err(mismatch("video audio"));
        }
        push_video_pair(&mut plan, pair)?;
    }
    plan.tokens.push(VIDEO_END);
    Ok(plan)
}

pub fn audio_plan(count: u32) -> Result<MediaPlan, Mimo2Error> {
    if count == 0 {
        return Err(mismatch("audio"));
    }
    let mut tokens = vec![AUDIO_START];
    let mut spans = Vec::new();
    push_pads(&mut tokens, &mut spans, PadKind::Audio, count);
    tokens.push(AUDIO_END);
    Ok(MediaPlan { tokens, spans })
}

/// One group per frame pair. Each pair is followed by its audio interval once.
pub fn joint_plan(pairs: &[VideoPair]) -> Result<MediaPlan, Mimo2Error> {
    if pairs.is_empty() {
        return Err(mismatch("joint"));
    }
    let mut plan = MediaPlan {
        tokens: vec![VIDEO_START],
        spans: Vec::new(),
    };
    for pair in pairs {
        if pair.audio_tokens == 0 {
            return Err(mismatch("joint audio"));
        }
        push_video_pair(&mut plan, pair)?;
        plan.tokens.push(AUDIO_START);
        push_pads(
            &mut plan.tokens,
            &mut plan.spans,
            PadKind::Audio,
            pair.audio_tokens,
        );
        plan.tokens.push(AUDIO_END);
    }
    plan.tokens.push(VIDEO_END);
    Ok(plan)
}

/// Mel frames after the source conv/pool/group. Kernel 3, stride 2, pool 2.
pub fn audio_feat_len(mel_len: u32) -> u32 {
    let mut n = mel_len;
    n = (n + 2 - 3) / 2 + 1;
    n = n / 2 + u32::from(n % 2 != 0);
    n.div_ceil(AUDIO_GROUP)
}

pub fn audio_interval(start_s: f32, end_s: f32, audio_len: u32) -> Result<u32, Mimo2Error> {
    if !start_s.is_finite() || !end_s.is_finite() || start_s < 0.0 || end_s < start_s {
        return Err(mismatch("audio interval"));
    }
    let start = (start_s * AUDIO_TOKENS_PER_SEC) as u32;
    let end = (end_s * AUDIO_TOKENS_PER_SEC) as u32;
    let end = end.min(audio_len);
    if end <= start {
        return Err(mismatch("audio interval"));
    }
    Ok(end - start)
}

pub fn check_spans(tokens: &[i32], spans: &[MediaSpan]) -> Result<(), Mimo2Error> {
    let mut end = 0u32;
    for span in spans {
        if span.count == 0 || span.start < end {
            return Err(mismatch("overlap"));
        }
        let Some(stop) = span.start.checked_add(span.count) else {
            return Err(mismatch("span"));
        };
        if stop as usize > tokens.len() {
            return Err(mismatch("span"));
        }
        let pad = span.kind.token();
        if tokens[span.start as usize..stop as usize]
            .iter()
            .any(|token| *token != pad)
        {
            return Err(mismatch("pad"));
        }
        end = stop;
    }
    Ok(())
}

fn audio_runs(tokens: &[i32]) -> Vec<MediaSpan> {
    let mut runs = Vec::new();
    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index] != AUDIO_PAD {
            index += 1;
            continue;
        }
        let start = index;
        while index < tokens.len() && tokens[index] == AUDIO_PAD {
            index += 1;
        }
        runs.push(MediaSpan {
            start: start as u32,
            count: (index - start) as u32,
            kind: PadKind::Audio,
        });
    }
    runs
}

/// Audio pad runs must partition `audio_len` once. A second full copy fails.
pub fn check_joint(tokens: &[i32], audio_len: u32) -> Result<(), Mimo2Error> {
    if audio_len == 0 {
        return Err(mismatch("audio"));
    }
    let runs = audio_runs(tokens);
    if runs.is_empty() {
        return Err(mismatch("audio"));
    }
    let sum = runs
        .iter()
        .fold(0u32, |sum, run| sum.saturating_add(run.count));
    if runs.iter().any(|run| run.count == audio_len) && runs.len() != 1 {
        return Err(mismatch("second audio"));
    }
    if sum != audio_len {
        return Err(mismatch("audio coverage"));
    }
    Ok(())
}

/// FNV-1a over media bytes. Identical tokens with a different tag must miss.
pub fn media_tag(parts: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for part in parts {
        for byte in *part {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn reuse_media(previous: u64, next: u64) -> bool {
    previous == next
}

const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];
pub const VISION_WIDTH: usize = 4096;

#[derive(Clone, Debug)]
pub struct PackedVisual {
    pub patches: Vec<f32>,
    pub grid_h: u32,
    pub grid_w: u32,
    pub tokens: u32,
}

impl PackedVisual {
    pub fn height(&self) -> u32 {
        self.grid_h * PATCH
    }

    pub fn width(&self) -> u32 {
        self.grid_w * PATCH
    }
}

#[derive(Clone, Debug)]
pub struct Mel {
    pub frames: u32,
    pub bins: Vec<f32>,
}

pub enum MediaPiece {
    Image { count: u32 },
    Audio { count: u32 },
    Video { pairs: Vec<VideoPair> },
    Joint { pairs: Vec<VideoPair> },
}

fn sample(src: &[u8], width: u32, height: u32, x: i32, y: i32, channel: usize) -> f32 {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return 0.0;
    }
    src[((y as u32 * width + x as u32) as usize) * 3 + channel] as f32
}

fn resize_rgb(src: &[u8], width: u32, height: u32, dst_w: u32, dst_h: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((dst_w * dst_h * 3) as usize);
    let scale_x = width as f32 / dst_w as f32;
    let scale_y = height as f32 / dst_h as f32;
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx = (x as f32 + 0.5) * scale_x - 0.5;
            let sy = (y as f32 + 0.5) * scale_y - 0.5;
            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let wx = sx - x0 as f32;
            let wy = sy - y0 as f32;
            for channel in 0..3 {
                let v00 = sample(src, width, height, x0, y0, channel);
                let v10 = sample(src, width, height, x0 + 1, y0, channel);
                let v01 = sample(src, width, height, x0, y0 + 1, channel);
                let v11 = sample(src, width, height, x0 + 1, y0 + 1, channel);
                let value =
                    (1.0 - wy) * ((1.0 - wx) * v00 + wx * v10) + wy * ((1.0 - wx) * v01 + wx * v11);
                out.push((value - MEAN[channel]) / STD[channel]);
            }
        }
    }
    out
}

/// Two temporal frames, merge-tile order, channel then time then 16x16.
pub fn pack_frames(
    first: &[f32],
    second: &[f32],
    height: u32,
    width: u32,
) -> Result<PackedVisual, Mimo2Error> {
    let tokens = visual_tokens(height, width)?;
    let grid_h = height / PATCH;
    let grid_w = width / PATCH;
    let pixels = (height * width * 3) as usize;
    if first.len() != pixels || second.len() != pixels {
        return Err(mismatch("frame"));
    }
    let mut patches = Vec::with_capacity((grid_h * grid_w) as usize * 1536);
    let llm_h = grid_h / MERGE;
    let llm_w = grid_w / MERGE;
    for ty in 0..llm_h {
        for tx in 0..llm_w {
            for dy in 0..MERGE {
                for dx in 0..MERGE {
                    let y0 = (ty * MERGE + dy) * PATCH;
                    let x0 = (tx * MERGE + dx) * PATCH;
                    for channel in 0..3 {
                        for time in 0..TEMPORAL {
                            let frame = if time == 0 { first } else { second };
                            for ky in 0..PATCH {
                                for kx in 0..PATCH {
                                    let at = (((y0 + ky) * width + x0 + kx) * 3 + channel) as usize;
                                    patches.push(frame[at]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(PackedVisual {
        patches,
        grid_h,
        grid_w,
        tokens,
    })
}

fn normalize_rgb(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks(3)
        .flat_map(|pixel| {
            (0..3).map(|channel| (pixel[channel] as f32 - MEAN[channel]) / STD[channel])
        })
        .collect()
}

/// Raw RGB ceiling for one clip. A 640×480 hour at 2 fps is several
/// gigabytes; the pipe stops before that buffer exists.
pub const VIDEO_RAW_CAP: usize = 64 * 1024 * 1024;
/// Smallest `smart_resize` frame: 32×256 lands on the factor grid at
/// `IMAGE_MIN_PIXELS`. Fewer bytes per frame would mean more spans.
const MIN_FRAME_BYTES: usize = 32 * 256 * 3;
/// Serial requests allow four clips. A joint clip spends two spans per
/// pair (the frame pair and its audio interval).
pub const MEDIA_INPUTS: usize = 4;
pub const SPAN_MAX: usize = {
    let frames = VIDEO_RAW_CAP / MIN_FRAME_BYTES;
    let pairs = if frames == 0 { 0 } else { (frames + 1) / 2 };
    pairs * 2 * MEDIA_INPUTS
};

pub fn media_span_count(tokens: &[i32]) -> usize {
    let mut count = 0usize;
    let mut index = 0usize;
    while index < tokens.len() {
        let token = tokens[index];
        if token != IMAGE_PAD && token != VIDEO_PAD && token != AUDIO_PAD {
            index += 1;
            continue;
        }
        count += 1;
        while index < tokens.len() && tokens[index] == token {
            index += 1;
        }
    }
    count
}

pub fn check_span_budget(spans: usize) -> Result<(), Mimo2Error> {
    if spans > SPAN_MAX {
        return Err(mismatch("span budget"));
    }
    Ok(())
}

fn video_frame_cap(width: u32, height: u32) -> Result<usize, Mimo2Error> {
    let frame = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| mismatch("video"))?;
    if frame > VIDEO_RAW_CAP {
        return Err(mismatch("video budget"));
    }
    Ok(VIDEO_RAW_CAP / frame)
}

fn video_duration_fits(width: u32, height: u32, duration: f32) -> Result<(), Mimo2Error> {
    let cap = video_frame_cap(width, height)?;
    if duration.is_finite() && duration > 0.0 && (duration * 2.0).ceil() as usize > cap {
        return Err(mismatch("video budget"));
    }
    Ok(())
}

fn read_bounded(reader: &mut impl std::io::Read, cap: usize) -> Result<Vec<u8>, Mimo2Error> {
    let mut out = Vec::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).map_err(|_| mismatch("video"))?;
        if n == 0 {
            return Ok(out);
        }
        if out.len().saturating_add(n) > cap {
            return Err(mismatch("video budget"));
        }
        out.extend_from_slice(&buf[..n]);
    }
}

/// Sampled at 2 fps and resized with the source factor. Odd frame counts repeat the last frame.
pub fn load_video(data: &[u8]) -> Result<(f32, Vec<PackedVisual>), Mimo2Error> {
    let path = std::env::temp_dir().join(format!(
        "ds4-mimo-{}-{}-{}.mp4",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        data.len()
    ));
    std::fs::write(&path, data).map_err(|_| mismatch("video"))?;
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration",
            "-of",
            "json",
        ])
        .arg(&path)
        .output();
    let probe = match probe {
        Ok(output) if output.status.success() => output.stdout,
        _ => {
            let _ = std::fs::remove_file(&path);
            return Err(mismatch("video"));
        }
    };
    let meta: serde_json::Value = serde_json::from_slice(&probe).unwrap_or(serde_json::Value::Null);
    let stream = &meta["streams"][0];
    let width = stream["width"].as_u64().unwrap_or(0) as u32;
    let height = stream["height"].as_u64().unwrap_or(0) as u32;
    let duration = stream["duration"]
        .as_str()
        .and_then(|text| text.parse::<f32>().ok())
        .unwrap_or(0.0);
    let (height, width) = match smart_resize(height, width) {
        Ok(size) => size,
        Err(error) => {
            let _ = std::fs::remove_file(&path);
            return Err(error);
        }
    };
    if let Err(error) = video_duration_fits(width, height, duration) {
        let _ = std::fs::remove_file(&path);
        return Err(error);
    }
    let frame_bytes = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(3);
    let byte_cap = frame_bytes.saturating_mul(video_frame_cap(width, height)?);
    let mut child = match std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args([
            "-vf",
            &format!("fps=2,scale={width}:{height}:flags=bilinear"),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => {
            let _ = std::fs::remove_file(&path);
            return Err(mismatch("video"));
        }
    };
    let mut stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&path);
            return Err(mismatch("video"));
        }
    };
    let read = read_bounded(&mut stdout, byte_cap);
    drop(stdout);
    let raw = match read {
        Ok(raw) => {
            let status = child.wait();
            let _ = std::fs::remove_file(&path);
            if status.map(|code| code.success()).unwrap_or(false) {
                raw
            } else {
                return Err(mismatch("video"));
            }
        }
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&path);
            return Err(error);
        }
    };
    if frame_bytes == 0 || raw.is_empty() || raw.len() % frame_bytes != 0 {
        return Err(mismatch("video"));
    }
    let mut frames: Vec<Vec<f32>> = raw.chunks(frame_bytes).map(normalize_rgb).collect();
    if frames.len() % 2 == 1 {
        frames.push(frames.last().cloned().unwrap());
    }
    let mut packed = Vec::new();
    for pair in frames.chunks(2) {
        packed.push(pack_frames(&pair[0], &pair[1], height, width)?);
    }
    if packed.is_empty() {
        return Err(mismatch("video"));
    }
    Ok((duration, packed))
}

pub fn pack_still(data: &[u8]) -> Result<PackedVisual, Mimo2Error> {
    let image = image::load_from_memory(data).map_err(|_| mismatch("image"))?;
    let rgb = image.to_rgb8();
    let (height, width) = smart_resize(rgb.height(), rgb.width())?;
    let resized = resize_rgb(rgb.as_raw(), rgb.width(), rgb.height(), width, height);
    pack_frames(&resized, &resized, height, width)
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

pub fn wav_mel(data: &[u8]) -> Result<Mel, Mimo2Error> {
    let mut reader =
        hound::WavReader::new(std::io::Cursor::new(data)).map_err(|_| mismatch("wav"))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err(mismatch("wav"));
    }
    let channels = spec.channels as usize;
    let mut mono = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Int if spec.bits_per_sample == 16 => {
            let samples = reader
                .samples::<i16>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| mismatch("wav"))?;
            for frame in samples.chunks(channels) {
                let sum: f32 = frame.iter().map(|v| *v as f32 / 32768.0).sum();
                mono.push(sum / channels as f32);
            }
        }
        hound::SampleFormat::Float => {
            let samples = reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| mismatch("wav"))?;
            for frame in samples.chunks(channels) {
                let sum: f32 = frame.iter().sum();
                mono.push(sum / channels as f32);
            }
        }
        _ => return Err(mismatch("wav")),
    }
    if mono.is_empty() {
        return Err(mismatch("wav"));
    }
    let target = (mono.len() as u64 * 24_000 / u64::from(spec.sample_rate)) as usize;
    let wave = if spec.sample_rate == 24_000 {
        mono
    } else if target == 0 {
        return Err(mismatch("wav"));
    } else {
        (0..target)
            .map(|i| {
                let src = i as f64 * (mono.len() as f64) / target as f64;
                let i0 = src.floor() as usize;
                let i1 = (i0 + 1).min(mono.len() - 1);
                let w = (src - i0 as f64) as f32;
                mono[i0] * (1.0 - w) + mono[i1] * w
            })
            .collect()
    };
    const NFFT: usize = 960;
    const HOP: usize = 240;
    const MELS: usize = 128;
    let frames = wave.len() / HOP + 1;
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(NFFT);
    let mut scratch =
        vec![rustfft::num_complex::Complex32::default(); fft.get_inplace_scratch_len()];
    let mel_points: Vec<f64> = {
        let hi = hz_to_mel(12_000.0);
        (0..=MELS + 1)
            .map(|i| mel_to_hz(hi * i as f64 / (MELS + 1) as f64))
            .collect()
    };
    let mut filters = Vec::with_capacity(MELS);
    for bin in 0..MELS {
        let mut row = Vec::new();
        for k in 0..=NFFT / 2 {
            let hz = k as f64 * 24_000.0 / NFFT as f64;
            let lower = (hz - mel_points[bin]) / (mel_points[bin + 1] - mel_points[bin]);
            let upper = (mel_points[bin + 2] - hz) / (mel_points[bin + 2] - mel_points[bin + 1]);
            let weight = lower.min(upper).max(0.0) as f32;
            if weight > 0.0 {
                row.push((k, weight));
            }
        }
        filters.push(row);
    }
    let mut bins = Vec::with_capacity(frames * MELS);
    let mut spec = vec![rustfft::num_complex::Complex32::default(); NFFT];
    for frame in 0..frames {
        for i in 0..NFFT {
            let index = frame * HOP + i;
            let sample = index
                .checked_sub(NFFT / 2)
                .and_then(|at| wave.get(at))
                .copied()
                .unwrap_or(0.0);
            let window = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / NFFT as f32).cos();
            spec[i] = rustfft::num_complex::Complex32::new(sample * window, 0.0);
        }
        fft.process_with_scratch(&mut spec, &mut scratch);
        for filter in &filters {
            let mut energy = 0.0f32;
            for (k, weight) in filter {
                let bin = spec[*k];
                energy += (bin.re * bin.re + bin.im * bin.im).sqrt() * weight;
            }
            bins.push(energy.max(1e-7).ln());
        }
    }
    let frames = frames as u32;
    if audio_feat_len(frames) == 0 {
        return Err(mismatch("audio"));
    }
    Ok(Mel { frames, bins })
}

fn push_plan(out: &mut Vec<i32>, spans: &mut Vec<MediaSpan>, plan: MediaPlan) {
    let base = out.len() as u32;
    for span in plan.spans {
        spans.push(MediaSpan {
            start: base + span.start,
            count: span.count,
            kind: span.kind,
        });
    }
    out.extend(plan.tokens);
}

/// Replace one image, audio, or video placeholder with the source span layout.
pub fn expand_pieces(
    tokens: &[i32],
    pieces: &[MediaPiece],
) -> Result<(Vec<i32>, Vec<MediaSpan>), Mimo2Error> {
    let mut out = Vec::new();
    let mut spans = Vec::new();
    let mut piece = 0usize;
    let mut index = 0usize;
    let mut joint_audio = 0u32;
    while index < tokens.len() {
        if tokens[index] == VISION_START
            && index + 2 < tokens.len()
            && tokens[index + 1] == VIDEO_PAD
            && tokens[index + 2] == VISION_END
        {
            let item = pieces.get(piece).ok_or_else(|| mismatch("piece"))?;
            piece += 1;
            match item {
                MediaPiece::Video { pairs } => {
                    push_plan(&mut out, &mut spans, video_plan(pairs)?);
                    index += 3;
                }
                MediaPiece::Joint { pairs } => {
                    if tokens.get(index + 3..index + 6)
                        != Some(&[AUDIO_START, AUDIO_PAD, AUDIO_END])
                    {
                        return Err(mismatch("joint audio"));
                    }
                    joint_audio = joint_audio.saturating_add(
                        pairs
                            .iter()
                            .fold(0u32, |sum, pair| sum.saturating_add(pair.audio_tokens)),
                    );
                    push_plan(&mut out, &mut spans, joint_plan(pairs)?);
                    index += 6;
                }
                _ => return Err(mismatch("video")),
            }
            continue;
        }
        if tokens[index] == IMAGE_PAD {
            let Some(MediaPiece::Image { count }) = pieces.get(piece) else {
                return Err(mismatch("image"));
            };
            piece += 1;
            push_pads(&mut out, &mut spans, PadKind::Image, *count);
            index += 1;
            continue;
        }
        if tokens[index] == AUDIO_PAD {
            let Some(MediaPiece::Audio { count }) = pieces.get(piece) else {
                return Err(mismatch("audio"));
            };
            piece += 1;
            push_pads(&mut out, &mut spans, PadKind::Audio, *count);
            index += 1;
            continue;
        }
        out.push(tokens[index]);
        index += 1;
    }
    if piece != pieces.len() {
        return Err(mismatch("piece"));
    }
    check_spans(&out, &spans)?;
    if joint_audio != 0 {
        check_joint(&out, joint_audio)?;
    }
    Ok((out, spans))
}

/// Greedy prefix. `tokens[0]` is always committed. Later drafts stick only
/// while they equal the previous verify target and the prior token is not EOS.
pub fn accepted_prefix(tokens: &[i32], target: &[i32], eos: i32) -> Option<usize> {
    if tokens.is_empty() || tokens.len() > TRIAL_CAP || tokens.len() != target.len() {
        return None;
    }
    let mut keep = 1usize;
    while keep < tokens.len() && tokens[keep - 1] != eos && tokens[keep] == target[keep - 1] {
        keep += 1;
    }
    Some(keep)
}

pub fn committed_frontier(
    prompt_len: u32,
    tokens: &[i32],
    target: &[i32],
    eos: i32,
) -> Result<u32, Mimo2Error> {
    let keep = accepted_prefix(tokens, target, eos).ok_or_else(|| mismatch("trial"))?;
    prompt_len
        .checked_add(keep as u32)
        .ok_or_else(|| mismatch("frontier"))
}

impl crate::Session<'_> {
    pub(super) fn eval_mimo2_argmax(
        &mut self,
        first: i32,
        max_tokens: i32,
        eos: i32,
    ) -> crate::Result<Vec<i32>> {
        if max_tokens <= 0 {
            return Ok(Vec::new());
        }
        let mut tokens = [0; TRIAL_CAP];
        let mut target = [0; TRIAL_CAP];
        let mut err = [0u8; 512];
        let n = unsafe {
            ds4_sys::ds4_bridge_mimo2_trial(
                self.raw.as_ptr(),
                first,
                max_tokens,
                tokens.as_mut_ptr(),
                target.as_mut_ptr(),
                TRIAL_CAP as i32,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if n == 0 {
            self.eval(first)?;
            return Ok(vec![first]);
        }
        if n < 0 {
            self.step_failed();
            return Err(crate::fail(n, &err));
        }
        let n = n as usize;
        if n > TRIAL_CAP || tokens[0] != first {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid MiMo trial result".into(),
            });
        }
        let prompt = self.pos();
        if prompt < 0 {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid MiMo frontier".into(),
            });
        }
        let frontier = committed_frontier(prompt as u32, &tokens[..n], &target[..n], eos).map_err(
            |error| crate::Error {
                code: 1,
                message: error.to_string(),
            },
        )?;
        let keep = frontier as i32 - prompt;
        if std::env::var_os("DS4_MIMO2_MTP_TRACE").is_some() {
            eprintln!(
                "mimo-mtp keep={keep} n={n} tokens={:?} target={:?}",
                &tokens[..n],
                &target[..n]
            );
        }
        if keep <= 0 || keep as usize > n {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "MiMo reject passed the accepted prefix".into(),
            });
        }
        let rc = unsafe {
            ds4_sys::ds4_bridge_mimo2_commit(
                self.raw.as_ptr(),
                keep,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if rc != 0 {
            self.step_failed();
            return Err(crate::fail(rc, &err));
        }
        for &token in &tokens[..keep as usize] {
            self.host.commit_eval(token);
        }
        Ok(tokens[..keep as usize].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn asymmetric_attention_and_mtp() {
        let full = Mimo2Layer::new(47).unwrap();
        assert_eq!(full.qkv_rows(), (12288, 768, 512));
        assert_eq!(full.sliding_window(), None);
        let prediction = Mimo2Layer::new(48).unwrap();
        assert_eq!(prediction.qkv_rows(), (12288, 1536, 1024));
        assert_eq!(prediction.sliding_window(), Some(128));
        assert!(prediction.is_prediction());
        assert!(!prediction.is_routed());
        assert!(Mimo2Layer::new(51).is_none());
        assert_eq!(
            (0..TRUNK)
                .filter(|&i| Mimo2Layer { index: i }.sliding_window().is_none())
                .count(),
            9
        );
    }
    #[test]
    fn mixed_inventory_geometry() {
        let specs = Mimo2Plan::layouts();
        assert_eq!(specs.len(), TENSORS);
        let find = |name| specs.iter().find(|s| s.name == name).unwrap();
        assert_eq!(
            find("blk.1.ffn_down_exps.weight").class,
            TypeClass::Exact(IQ2_XS)
        );
        assert_eq!(find("blk.48.ffn_down.weight").dim[..2], [16384, 4096]);
        assert!(!specs.iter().any(|s| s.name == "blk.0.attn_sinks.weight"));
    }

    #[test]
    fn unknown_architecture_rejects() {
        use crate::shape::{route_architecture, ArchRoute};
        assert!(matches!(
            route_architecture(Some(b"mimo_v2")),
            ArchRoute::Unsupported
        ));
        assert!(matches!(
            route_architecture(Some(b"not-a-model")),
            ArchRoute::Unsupported
        ));
        assert!(matches!(
            route_architecture(Some(b"mimo2")),
            ArchRoute::Fixed(_)
        ));
    }

    #[test]
    fn published_inventory_counts() {
        assert_eq!(Mimo2Plan::layouts().len(), LANGUAGE_TENSORS);
        assert_eq!(MIXED_SHARDS, 4);
        assert_eq!(TRUNK_LAYERS, 48);
        assert_eq!(MTP_BLOCKS, 3);
        for index in TRUNK_LAYERS..TRUNK_LAYERS + MTP_BLOCKS {
            let layer = Mimo2Layer::new(index).unwrap();
            assert!(layer.is_prediction());
            assert!(!layer.is_routed());
        }
    }

    #[test]
    fn span_budget_matches_the_decode_cap() {
        let frames = VIDEO_RAW_CAP / (32 * 256 * 3);
        let pairs = frames.div_ceil(2);
        let per_clip = pairs * 2;
        assert_eq!(SPAN_MAX, per_clip * MEDIA_INPUTS);
        assert!(per_clip > 8);
        let units: Vec<_> = (0..pairs)
            .map(|index| VideoPair {
                timestamp_s: index as f32,
                timestamp_ids: vec![11],
                height: 32,
                width: 256,
                audio_tokens: 1,
            })
            .collect();
        let joint = joint_plan(&units).unwrap();
        assert_eq!(joint.spans.len(), per_clip);
        assert_eq!(media_span_count(&joint.tokens), per_clip);
        check_span_budget(per_clip).unwrap();
        check_span_budget(SPAN_MAX).unwrap();
        assert!(check_span_budget(SPAN_MAX + 1)
            .unwrap_err()
            .to_string()
            .contains("span budget"));
    }

    #[test]
    fn long_video_exceeds_the_decode_budget() {
        assert!(video_duration_fits(64, 64, 1.0).is_ok());
        let err = video_duration_fits(640, 480, 3600.0).unwrap_err();
        assert!(err.to_string().contains("video budget"), "{err}");
        assert!(video_frame_cap(5000, 5000).is_err());
        let mut big = std::io::Cursor::new(vec![1u8; 128]);
        assert!(read_bounded(&mut big, 64).is_err());
        let mut small = std::io::Cursor::new(vec![1u8; 32]);
        assert_eq!(read_bounded(&mut small, 64).unwrap().len(), 32);
    }

    #[test]
    fn admit_256k_without_clamping() {
        let admitted = admit_context(QUALIFIED_CONTEXT).unwrap();
        assert_eq!(admitted.effective, 262_144);
        assert!(admitted.effective >= admitted.requested);
        assert_eq!(admitted.qualified, 262_144);
        assert_eq!(admitted.index_limit, 1_048_576);
        assert_eq!(admitted.swa_window, 128);
        assert!(!admitted.dflash_qualified);
        assert!(!admitted.context_512k_qualified);
        assert!(!admitted.context_1m_qualified);
        assert_eq!(admit_context(INDEX_LIMIT).unwrap().effective, INDEX_LIMIT);
        assert!(admit_context(INDEX_LIMIT + 1).is_err());
        assert_eq!(kv_rows(0, 262_144, PREFILL_CAP), Some(262_144));
        assert_eq!(
            kv_rows(1, 262_144, PREFILL_CAP),
            Some(128 + PREFILL_CAP - 1)
        );
        assert!(context_bytes(262_144, PREFILL_CAP).is_some());
        assert!(context_bytes(INDEX_LIMIT + 1, 1).is_none());
        let full = (0..TRUNK_LAYERS)
            .filter(|layer| Mimo2Layer::new(*layer).unwrap().sliding_window().is_none())
            .count();
        assert_eq!(full, 9);

        let caps = crate::serving_caps(crate::ModelFamily::Mimo2, crate::Variant::Mimo26Flash);
        assert_eq!(caps.qualified_ctx, Some(QUALIFIED_CONTEXT));
        assert_eq!(caps.ctx_max, Some(INDEX_LIMIT));
        assert_eq!(caps.mtp, crate::serving::MtpKind::Embedded);
        let mut request = crate::serving::ServingRequest::default();
        request.ctx = QUALIFIED_CONTEXT as i32;
        let plan = crate::serving::resolve_plan(
            &request,
            Some(caps),
            &crate::serving::EngineFacts::default(),
        );
        assert_eq!(plan.effective.ctx, QUALIFIED_CONTEXT as i32);
        assert_ne!(plan.effective.ctx, 8192);
        assert_eq!(caps.mtp_support, crate::serving::Support::Qualified);
        assert_eq!(plan.qualified.mtp, crate::serving::Support::Qualified);
        assert_eq!(plan.effective.mtp_mode, crate::serving::MtpMode::Auto);
        assert!(plan.effective.mtp_weights);
        assert!(!plan
            .issues
            .iter()
            .any(|issue| issue.code == "mtp_unverified" || issue.code == "ctx_unavailable"));
        request.mtp_mode = crate::serving::MtpMode::Off;
        let off = crate::serving::resolve_plan(
            &request,
            Some(caps),
            &crate::serving::EngineFacts::default(),
        );
        assert_eq!(off.effective.ctx, QUALIFIED_CONTEXT as i32);
        assert_eq!(off.effective.mtp_mode, crate::serving::MtpMode::Off);
        assert!(!off.effective.mtp_weights);
        request.mtp_mode = crate::serving::MtpMode::On;
        request.mtp_path = Some("MiMo-V2.6-Flash-RL-DFlash-Q8_0.gguf".into());
        let sidecar = crate::serving::resolve_plan(
            &request,
            Some(caps),
            &crate::serving::EngineFacts::default(),
        );
        assert_eq!(sidecar.effective.mtp_mode, crate::serving::MtpMode::Off);
        assert!(sidecar
            .issues
            .iter()
            .any(|issue| issue.code == "mtp_contract"));
        request.mtp_mode = crate::serving::MtpMode::Auto;
        request.mtp_path = None;
        request.ctx = 524_288;
        let wide = crate::serving::resolve_plan(
            &request,
            Some(caps),
            &crate::serving::EngineFacts::default(),
        );
        assert_eq!(wide.effective.ctx, 524_288);
        assert!(wide
            .issues
            .iter()
            .any(|issue| issue.code == "ctx_unqualified"));
        assert!(!wide
            .issues
            .iter()
            .any(|issue| issue.level == crate::serving::IssueLevel::Error));
    }

    #[test]
    fn media_spans_follow_the_source_contract() {
        assert_eq!(nfc("e\u{0301}"), "\u{00e9}");
        assert_eq!(temporal_groups(1), 1);
        assert_eq!(temporal_groups(TEMPORAL), 1);
        assert_eq!(temporal_groups(3), 2);
        assert_eq!(format_timestamp(0.0), "00:00");
        assert_eq!(format_timestamp(61.9), "01:01");

        let image = image_plan(32, 32).unwrap();
        assert_eq!(image.tokens, vec![VISION_START, IMAGE_PAD, VISION_END]);
        assert_eq!(image.positions().len(), image.tokens.len());
        check_spans(&image.tokens, &image.spans).unwrap();

        let audio = audio_plan(4).unwrap();
        assert_eq!(audio.spans[0].kind, PadKind::Audio);
        check_spans(&audio.tokens, &audio.spans).unwrap();

        let pair = VideoPair {
            timestamp_s: 0.0,
            timestamp_ids: vec![11, 12],
            height: 32,
            width: 64,
            audio_tokens: 0,
        };
        let video = video_plan(&[pair]).unwrap();
        assert_eq!(video.tokens[0], VIDEO_START);
        assert_eq!(*video.tokens.last().unwrap(), VIDEO_END);
        assert!(video.tokens.contains(&VIDEO_PAD));
        assert_eq!(video.positions().len(), video.tokens.len());
        check_spans(&video.tokens, &video.spans).unwrap();

        let units = [
            VideoPair {
                timestamp_s: 0.0,
                timestamp_ids: vec![11],
                height: 32,
                width: 32,
                audio_tokens: audio_interval(0.0, 1.0, 10).unwrap(),
            },
            VideoPair {
                timestamp_s: 1.0,
                timestamp_ids: vec![12],
                height: 32,
                width: 32,
                audio_tokens: audio_interval(1.0, 2.0, 10).unwrap(),
            },
        ];
        assert_eq!(units[0].audio_tokens, 6);
        assert_eq!(units[1].audio_tokens, 4);
        let joint = joint_plan(&units).unwrap();
        let audio_len = units[0].audio_tokens + units[1].audio_tokens;
        check_joint(&joint.tokens, audio_len).unwrap();
        assert_eq!(joint.positions().len(), joint.tokens.len());
        check_spans(&joint.tokens, &joint.spans).unwrap();

        let mut overlapped = joint.spans.clone();
        overlapped.push(MediaSpan {
            start: overlapped[0].start + 1,
            count: 1,
            kind: PadKind::Image,
        });
        assert!(check_spans(&joint.tokens, &overlapped).is_err());

        let mut wrong = joint.spans.clone();
        wrong[0].kind = PadKind::Image;
        assert!(check_spans(&joint.tokens, &wrong).is_err());

        let mut doubled = joint.tokens.clone();
        doubled.extend(audio_plan(audio_len).unwrap().tokens);
        assert!(check_joint(&doubled, audio_len).is_err());

        let same = media_tag(&[b"frame", b"pcm"]);
        let other = media_tag(&[b"frame", b"PCM"]);
        assert_ne!(same, other);
        assert!(reuse_media(same, same));
        assert!(!reuse_media(same, other));
    }

    #[test]
    fn expand_replaces_placeholders_and_rejects_a_second_audio() {
        let image = [VISION_START, IMAGE_PAD, VISION_END];
        let (tokens, spans) = expand_pieces(&image, &[MediaPiece::Image { count: 3 }]).unwrap();
        assert_eq!(
            tokens,
            vec![VISION_START, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, VISION_END]
        );
        assert_eq!(
            spans,
            vec![MediaSpan {
                start: 1,
                count: 3,
                kind: PadKind::Image
            }]
        );

        let stub = [
            VISION_START,
            VIDEO_PAD,
            VISION_END,
            AUDIO_START,
            AUDIO_PAD,
            AUDIO_END,
        ];
        let pairs = vec![VideoPair {
            timestamp_s: 0.0,
            timestamp_ids: vec![7],
            height: 32,
            width: 32,
            audio_tokens: 4,
        }];
        let (joint, spans) = expand_pieces(&stub, &[MediaPiece::Joint { pairs }]).unwrap();
        assert_eq!(joint[0], VIDEO_START);
        assert!(joint.contains(&VIDEO_PAD));
        assert!(joint.contains(&AUDIO_PAD));
        assert_eq!(
            spans.iter().map(|span| span.count).sum::<u32>(),
            visual_tokens(32, 32).unwrap() + 4
        );
        let mut again = stub.to_vec();
        again.extend_from_slice(&[AUDIO_START, AUDIO_PAD, AUDIO_END]);
        assert!(expand_pieces(
            &again,
            &[
                MediaPiece::Joint {
                    pairs: vec![VideoPair {
                        timestamp_s: 0.0,
                        timestamp_ids: vec![7],
                        height: 32,
                        width: 32,
                        audio_tokens: 4,
                    }]
                },
                MediaPiece::Audio { count: 4 },
            ]
        )
        .is_err());
    }

    #[test]
    fn still_pack_duplicates_the_temporal_frame() {
        let mut png = Vec::new();
        use image::ImageEncoder;
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(
                &[10, 20, 30, 40, 50, 60],
                2,
                1,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        let packed = pack_still(&png).unwrap();
        assert_eq!(
            packed.tokens,
            visual_tokens(packed.grid_h * PATCH, packed.grid_w * PATCH).unwrap()
        );
        assert_eq!(
            packed.patches.len(),
            (packed.grid_h * packed.grid_w) as usize * 1536
        );
        let patch = &packed.patches[..1536];
        for channel in 0..3 {
            let time0 = &patch[channel * 2 * 256..channel * 2 * 256 + 256];
            let time1 = &patch[channel * 2 * 256 + 256..channel * 2 * 256 + 512];
            assert_eq!(time0, time1);
        }
    }

    #[test]
    fn mtp_reject_stops_at_the_accepted_prefix() {
        let prompt = 8u32;
        let frontier = committed_frontier(prompt, &[10, 11, 12], &[11, 99, 12], 99).unwrap();
        assert_eq!(frontier, prompt + 2);
        assert!(frontier < prompt + 3);
        assert_eq!(
            committed_frontier(4, &[1, 2, 3, 4], &[2, 3, 4, 5], 99).unwrap(),
            8
        );
        assert_eq!(
            committed_frontier(4, &[7, 8, 9], &[1, 2, 3], 99).unwrap(),
            5
        );
        assert!(committed_frontier(1, &[], &[], 1).is_err());
    }

    #[test]
    fn tokenizer_fixture_covers_nfc_and_media_controls() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/mimo2-tokenizer.json")).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        let blob: String = cases
            .iter()
            .map(|case| {
                case["context"]["messages"][0]["content"]
                    .as_str()
                    .unwrap_or("")
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(blob.contains('\u{0301}'));
        assert!(blob.contains("<|image_pad|>"));
        assert!(blob.contains("<|audio_pad|>"));
        assert!(blob.contains("<|vision_start|>"));
        assert_eq!(nfc("e\u{0301}"), "\u{00e9}");
    }
}
