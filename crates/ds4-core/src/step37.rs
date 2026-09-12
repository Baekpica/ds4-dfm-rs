//! Step 3.7 MQ83 preflight and bind plan. Native execution is not enabled.
//! Per-layer attention widths must survive the future native ABI adapter.
use std::collections::BTreeSet;
use std::path::Path;

use crate::gguf::{GgufError, GgufFile};
use crate::tensors::{TensorError, TensorInfo, TensorInventory};

const LAYERS: u32 = 45;
const FULL_PERIOD: u32 = 4;
const CLAMP_START: u32 = 43;
const EMBED: u64 = 4096;
const VOCAB: u64 = 128896;
const HEAD_DIM: u64 = 128;
const KV_HEADS: u64 = 8;
const DENSE_LAYERS: u32 = 3;
const DENSE_FF: u64 = 11264;
const EXPERT_FF: u64 = 1280;
const EXPERTS: u64 = 288;
const TENSORS: usize = 754;
const SHARDS: u32 = 9;
const F32: u32 = 0;
const Q8_0: u32 = 8;
const Q4_K: u32 = 12;
const IQ2_XXS: u32 = 16;
const SOURCE_REV: &[u8] = b"5f6244077ac62e04eec3f320501ff8c2b293373a";

#[derive(Debug)]
pub enum Step37Error {
    Metadata(GgufError),
    Inventory(TensorError),
    Contract(String),
}

impl std::fmt::Display for Step37Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metadata(e) => write!(f, "Step 3.7 metadata: {e}"),
            Self::Inventory(e) => write!(f, "Step 3.7 inventory: {e}"),
            Self::Contract(e) => write!(f, "Step 3.7 contract: {e}"),
        }
    }
}

impl std::error::Error for Step37Error {}

impl From<GgufError> for Step37Error {
    fn from(e: GgufError) -> Self {
        Self::Metadata(e)
    }
}

impl From<TensorError> for Step37Error {
    fn from(e: TensorError) -> Self {
        Self::Inventory(e)
    }
}

fn mismatch(key: &str) -> Step37Error {
    Step37Error::Contract(key.into())
}

#[derive(Clone, Copy, Debug)]
pub struct Step37Layer {
    index: u32,
}

impl Step37Layer {
    pub fn query_heads(self) -> u32 {
        if self.index.is_multiple_of(FULL_PERIOD) {
            64
        } else {
            96
        }
    }

    pub fn rotary_dims(self) -> u32 {
        if self.index.is_multiple_of(FULL_PERIOD) {
            64
        } else {
            128
        }
    }

    pub fn rope_base(self) -> f32 {
        if self.index.is_multiple_of(FULL_PERIOD) {
            5_000_000.0
        } else {
            10_000.0
        }
    }

    /// None means full causal attention, not a zero-length cache.
    pub fn sliding_window(self) -> Option<u32> {
        if self.index.is_multiple_of(FULL_PERIOD) {
            None
        } else {
            Some(512)
        }
    }

    /// Routed and shared SwiGLU clamps, respectively; zero disables clipping.
    pub fn swiglu_clamps(self) -> (f32, f32) {
        if self.index >= CLAMP_START {
            (7.0, 16.0)
        } else {
            (0.0, 0.0)
        }
    }
}

#[derive(Debug)]
pub struct Step37Plan {
    inventory: TensorInventory,
    bindings: Vec<usize>,
    layers: Vec<Step37Layer>,
}

impl Step37Plan {
    /// Read metadata through mmap and resolve all shards without copying weights.
    /// This accepts the MQ83 main artifact only, not MTP or vision sidecars.
    pub fn inspect(path: &Path) -> Result<Self, Step37Error> {
        let first = GgufFile::open(path)?;
        check_metadata(&first)?;
        let inventory = TensorInventory::from_file(path, &first)?;
        for (index, shard) in inventory.shards.iter().enumerate() {
            let g = GgufFile::open(&shard.path)?;
            if g.get_u16("split.no") != Some(index as u16)
                || g.split_count() != SHARDS
                || g.get_token_id("split.tensors.count") != Some(TENSORS as i32)
            {
                return Err(mismatch("split identity"));
            }
        }
        let bindings = check_tensors(&inventory)?;
        Ok(Self {
            inventory,
            bindings,
            layers: (0..LAYERS).map(|index| Step37Layer { index }).collect(),
        })
    }

    pub fn layers(&self) -> &[Step37Layer] {
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

fn check_metadata(g: &GgufFile) -> Result<(), Step37Error> {
    if g.get_string("general.architecture") != Some(b"step35") {
        return Err(mismatch("general.architecture"));
    }
    for (key, expected) in [
        ("step37.source_revision", SOURCE_REV),
        ("tokenizer.ggml.model", b"gpt2".as_slice()),
        ("tokenizer.ggml.pre", b"deepseek-v3".as_slice()),
    ] {
        if g.get_string(key) != Some(expected) {
            return Err(mismatch(key));
        }
    }
    if g.split_count() != SHARDS || g.get_u16("split.no") != Some(0) {
        return Err(mismatch("expected first of nine main shards"));
    }
    // Block count distinguishes the main backbone from the 48-block MTP metadata.
    for (suffix, expected) in [
        ("block_count", LAYERS),
        ("context_length", 262144),
        ("embedding_length", EMBED as u32),
        ("feed_forward_length", DENSE_FF as u32),
        ("attention.key_length", HEAD_DIM as u32),
        ("attention.value_length", HEAD_DIM as u32),
        ("attention.sliding_window", 512),
        ("expert_count", EXPERTS as u32),
        ("expert_used_count", 8),
        ("expert_feed_forward_length", EXPERT_FF as u32),
        ("expert_shared_feed_forward_length", EXPERT_FF as u32),
        ("expert_gating_func", 2), // GGUF sigmoid routing, not softmax.
        ("leading_dense_block_count", DENSE_LAYERS),
        ("moe_every_n_layers", 1),
    ] {
        let key = format!("step35.{suffix}");
        if g.get_u32(&key) != Some(expected) {
            return Err(mismatch(&key));
        }
    }
    for (suffix, expected) in [
        ("rope.freq_base", 5_000_000.0),
        ("rope.freq_base_swa", 10_000.0),
        ("expert_weights_scale", 3.0),
        ("attention.layer_norm_rms_epsilon", 1e-5),
    ] {
        let key = format!("step35.{suffix}");
        if g.get_f32_compat(&key) != Some(expected) {
            return Err(mismatch(&key));
        }
    }
    if g.get_bool("step35.expert_weights_norm") != Some(true) {
        return Err(mismatch("step35.expert_weights_norm"));
    }
    for (key, expected) in [
        (
            "step35.attention.head_count",
            (0..LAYERS)
                .map(|index| Step37Layer { index }.query_heads())
                .collect(),
        ),
        (
            "step35.attention.head_count_kv",
            vec![KV_HEADS as u32; LAYERS as usize],
        ),
    ] {
        let arr = g.get_array(key).ok_or_else(|| mismatch(key))?;
        // Official head schedules use INT32 arrays; negative values still fail
        // the exact positive schedule comparison after reading their bits.
        if g.array_le_u32s(&arr)? != expected {
            return Err(mismatch(key));
        }
    }
    let key = "step35.attention.sliding_window_pattern";
    let arr = g.get_array(key).ok_or_else(|| mismatch(key))?;
    let expected: Vec<_> = (0..LAYERS)
        .map(|i| !i.is_multiple_of(FULL_PERIOD))
        .collect();
    if g.array_bools(&arr)? != expected {
        return Err(mismatch(key));
    }
    for (key, clamp) in [
        ("step35.swiglu_clamp_exp", 7.0),
        ("step35.swiglu_clamp_shexp", 16.0),
    ] {
        let arr = g.get_array(key).ok_or_else(|| mismatch(key))?;
        let expected: Vec<_> = (0..LAYERS)
            .map(|i| if i >= CLAMP_START { clamp } else { 0.0 })
            .collect();
        if g.array_f32s(&arr)? != expected {
            return Err(mismatch(key));
        }
    }
    let tokens = g
        .get_array("tokenizer.ggml.tokens")
        .ok_or_else(|| mismatch("vocabulary"))?;
    if tokens.typ != crate::gguf::GGUF_VALUE_STRING
        || tokens.len != VOCAB
        || g.get_token_id("tokenizer.ggml.bos_token_id") != Some(0)
        || g.get_token_id("tokenizer.ggml.eos_token_id") != Some(128007)
    {
        return Err(mismatch("vocabulary/BOS/EOS"));
    }
    Ok(())
}

struct Spec {
    name: String,
    typ: u32,
    dims: Vec<u64>,
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
    add("rope_freqs.weight".into(), F32, &[64]);
    add("output_norm.weight".into(), F32, &[EMBED]);
    for name in ["token_embd.weight", "output.weight"] {
        add(name.into(), Q8_0, &[EMBED, VOCAB]);
    }
    for index in 0..LAYERS {
        let prefix = format!("blk.{index}");
        let heads = u64::from(Step37Layer { index }.query_heads());
        for name in ["attn_norm.weight", "ffn_norm.weight"] {
            add(format!("{prefix}.{name}"), F32, &[EMBED]);
        }
        for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            add(format!("{prefix}.{name}"), F32, &[HEAD_DIM]);
        }
        for (name, cols, rows) in [
            ("attn_q.weight", EMBED, heads * HEAD_DIM),
            ("attn_k.weight", EMBED, KV_HEADS * HEAD_DIM),
            ("attn_v.weight", EMBED, KV_HEADS * HEAD_DIM),
            ("attn_output.weight", heads * HEAD_DIM, EMBED),
            ("attn_gate.weight", EMBED, heads),
        ] {
            add(format!("{prefix}.{name}"), Q8_0, &[cols, rows]);
        }
        if index < DENSE_LAYERS {
            for name in ["ffn_gate.weight", "ffn_up.weight"] {
                add(format!("{prefix}.{name}"), Q8_0, &[EMBED, DENSE_FF]);
            }
            add(
                format!("{prefix}.ffn_down.weight"),
                Q8_0,
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
        let gate_type = if (7..=40).contains(&index) {
            IQ2_XXS
        } else {
            Q4_K
        };
        for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
            add(
                format!("{prefix}.{name}"),
                gate_type,
                &[EMBED, EXPERT_FF, EXPERTS],
            );
        }
        add(
            format!("{prefix}.ffn_down_exps.weight"),
            Q4_K,
            &[EXPERT_FF, EMBED, EXPERTS],
        );
        // Shared experts remain Q8, even where routed gate/up use IQ2_XXS.
        for name in ["ffn_gate_shexp.weight", "ffn_up_shexp.weight"] {
            add(format!("{prefix}.{name}"), Q8_0, &[EMBED, EXPERT_FF]);
        }
        add(
            format!("{prefix}.ffn_down_shexp.weight"),
            Q8_0,
            &[EXPERT_FF, EMBED],
        );
    }
    out
}

fn check_tensors(inv: &TensorInventory) -> Result<Vec<usize>, Step37Error> {
    if inv.tensors.len() != TENSORS {
        return Err(mismatch("tensor count"));
    }
    let mut names = BTreeSet::new();
    for t in &inv.tensors {
        if !names.insert(&t.name) {
            return Err(mismatch(&format!("duplicate {}", t.name)));
        }
    }
    let mut bindings = Vec::with_capacity(TENSORS);
    for spec in specs() {
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

#[cfg(test)]
mod tests;
