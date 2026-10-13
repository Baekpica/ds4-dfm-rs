//! DeepSeek V4.1 wiring: the `deepseek4.*` metadata keys the engine keeps in
//! `g_ds4_v41` and the conditionals the V4.1 bind and layout arms are driven
//! by.
//!
//! Source of truth is the C engine at `/data/YoungAi` (commit 3946dbc):
//!
//! - `core_validate_v41.c:46-154` `v41_load_metadata`: the keys, the array
//!   types, the source-layer wiring derivation and its hard stops. A missing
//!   key is a hard stop, never a default (`core_validate_v41.c:8`).
//! - `core_validate.c:5-46` `validate_compress_ratio_metadata`: the ratios are
//!   metadata truth for V4.1 (0/1/2, no formula), loaded once per model.
//! - `core_globals.c:75-95` `ds4_layer_compress_ratio` /
//!   `ds4_expected_layer_compress_ratio`: V4.1 returns the metadata value.
//! - `core_bind_v41.c:41-145`: what consumes the wiring at bind time.
//!
//! V4.1 differs from V4 Flash in wiring, not dimensions: only the kv source
//! layers compress and hold compressed KV (the rest read the nearest earlier
//! source layer), only index source layers run the indexer, engram lives on
//! two layers, and the three draft towers mirror a layer with either one VQ
//! blob or per-expert FP4 tensors.

use crate::gguf::{GgufFile, GGUF_VALUE_INT32, GGUF_VALUE_UINT32};
use crate::shape::{Shape, Variant};
use crate::tensors::TensorInventory;

/// `DS4_V41_MAX_ENGRAM` (`ds4_internal.h:53`).
pub const V41_MAX_ENGRAM: usize = 4;
/// `DS4_MTP_MAX_TOWERS` (`ds4_internal.h:54`); the real count is metadata.
pub const MTP_MAX_TOWERS: u32 = 4;
/// `DS4_MTP_MAX_EXPERTS` (`ds4_internal.h:55`); the real count is metadata.
pub const MTP_MAX_EXPERTS: u32 = 256;
/// V4.1 compression ratios are 0/1/2 (`core_validate.c:31-37`).
pub const V41_MAX_COMPRESS_RATIO: u32 = 2;

const KEY_COMPRESS_RATIOS: &str = "deepseek4.attention.compress_ratios";
const KEY_CTX: &str = "deepseek4.context_length";
const KEY_KV_SOURCE_LAYERS: &str = "deepseek4.attention.kv_source_layers";
const KEY_INDEX_SOURCE_LAYERS: &str = "deepseek4.attention.index_source_layers";
const KEY_ENGRAM_LAYER_IDS: &str = "deepseek4.engram.layer_ids";
const KEY_ENGRAM_ROWS: &str = "deepseek4.engram.num_embeddings";
const KEY_ENGRAM_MAX_NGRAM: &str = "deepseek4.engram.max_ngram_size";
const KEY_ENGRAM_HEADS: &str = "deepseek4.engram.head_count";
const KEY_ENGRAM_HEAD_DIM: &str = "deepseek4.engram.head_dim";
const KEY_ENGRAM_PAD: &str = "deepseek4.engram.pad_id_compressed";
const KEY_ENGRAM_VOCAB: &str = "deepseek4.engram.vocab_size";
const KEY_ENGRAM_CVOCAB: &str = "deepseek4.engram.compressed_vocab_size";
const KEY_CANDIDATE_SOURCE_LAYER: &str = "deepseek4.attention.candidate.source_layer";
const KEY_CANDIDATE_TOPK_BLOCKS: &str = "deepseek4.attention.candidate.topk_blocks";
const KEY_CANDIDATE_BLOCK_SIZE: &str = "deepseek4.attention.candidate.block_size";
const KEY_MTP_TOWER_COUNT: &str = "deepseek4.mtp.tower_count";
const KEY_MTP_EXPERT_COUNT: &str = "deepseek4.mtp.expert_count";
const KEY_MTP_TARGET_LAYERS: &str = "deepseek4.mtp.target_layers";
const KEY_MTP_MARKOV_RANK: &str = "deepseek4.mtp.markov_rank";
const KEY_MTP_BLOCK_SIZE: &str = "deepseek4.mtp.block_size";
const KEY_MTP_EXPERT_USED: &str = "deepseek4.mtp.expert_used_count";
const KEY_MTP_NOISE_TOKEN_ID: &str = "deepseek4.mtp.noise_token_id";

#[derive(Debug, PartialEq, Eq)]
pub enum V41WireError {
    NotV41,
    MissingKey(&'static str),
    /// A key whose name carries the engram index (`deepseek4.engram.<i>.*`).
    MissingKeyAt(String),
    ArrayType(&'static str),
    ArrayLen(&'static str),
    /// `deepseek4.context_length` present but 0 or over `u32::MAX`
    /// (`core_validate_v41.c:52-53`).
    CtxInvalid,
    LayerRange(&'static str, i64),
    NoSourceBefore(u32),
    KvSourceNotIndexSource(u32),
    EngramRowsMismatch,
    TowerLimit,
}

impl V41WireError {
    pub fn token(&self) -> String {
        match self {
            V41WireError::NotV41 => "v41-not-v41".into(),
            V41WireError::MissingKey(k) => format!("v41-missing-key {k}"),
            V41WireError::MissingKeyAt(k) => format!("v41-missing-key {k}"),
            V41WireError::ArrayType(k) => format!("v41-array-type {k}"),
            V41WireError::ArrayLen(k) => format!("v41-array-len {k}"),
            V41WireError::CtxInvalid => "v41-ctx-invalid".into(),
            V41WireError::LayerRange(k, id) => format!("v41-layer-range {k} {id}"),
            V41WireError::NoSourceBefore(il) => format!("v41-no-source-before {il}"),
            V41WireError::KvSourceNotIndexSource(il) => {
                format!("v41-kv-source-not-index-source {il}")
            }
            V41WireError::EngramRowsMismatch => "v41-engram-rows-mismatch".into(),
            V41WireError::TowerLimit => "v41-tower-limit".into(),
        }
    }
}

impl std::fmt::Display for V41WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for V41WireError {}

/// The V4.1 wiring table, per layer and per tower.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V41Wire {
    /// `deepseek4.context_length` (the HF `max_position_embeddings`; 1M in
    /// the artifact).  The engine's only context source -- no default, no
    /// cap constant, no `--ctx` (`core_validate_v41.c:51-53`); the native
    /// generate entry and state alloc refuse a zero.
    pub ctx: u32,
    /// `deepseek4.attention.compress_ratios`, first `n_layer` entries (V4.1:
    /// 0/1/2). `core_validate.c:14-46`.
    pub compress_ratios: Vec<u32>,
    /// `deepseek4.attention.kv_source_layers` as a per-layer flag
    /// (`core_validate_v41.c:59-63`).
    pub is_kv_source: Vec<bool>,
    /// Layer -> the kv source it reads; -1 when the layer does not compress
    /// (`core_validate_v41.c:69-83`).
    pub kv_source_of: Vec<i16>,
    /// `deepseek4.attention.index_source_layers` as a per-layer flag.
    pub is_index_source: Vec<bool>,
    /// Layer -> the index source it reuses; -1 when it does not compress.
    pub index_source_of: Vec<i16>,
    /// `deepseek4.engram.layer_ids`, in metadata order.
    pub engram_layers: Vec<u32>,
    /// Layer -> engram index, -1 = none (`core_validate_v41.c:118-125`).
    pub engram_index_of: Vec<i16>,
    /// `deepseek4.engram.max_ngram_size` (4 in the artifact).
    pub engram_max_ngram: u32,
    /// `deepseek4.engram.head_count` (8 in the artifact).
    pub engram_heads: u32,
    /// `deepseek4.engram.head_dim` (256 in the artifact).
    pub engram_head_dim: u32,
    /// `deepseek4.engram.num_embeddings` per layer; the table row count.
    pub engram_rows: Vec<u64>,
    /// `deepseek4.engram.<i>.weight_offset` / `.scale_offset`, byte offsets
    /// of the two row planes in that layer's shard
    /// (`core_validate_v41.c:147-148`).
    pub engram_weight_off: Vec<u64>,
    pub engram_scale_off: Vec<u64>,
    /// `deepseek4.engram.<i>.table_path`, absolute as the converter wrote it
    /// (`core_validate_v41.c:126-141`). The published GGUF must not be
    /// rewritten for a new machine, so `apply_engram_dir` swaps the directory
    /// and keeps the file name.
    pub engram_table_path: Vec<String>,
    /// `deepseek4.engram.pad_id_compressed`: the hash's filler for positions
    /// before the start of the sequence and for out-of-vocab tokens
    /// (`core_v41_engram.c:104-107`).
    pub engram_pad: u32,
    /// `deepseek4.engram.vocab_size` / `deepseek4.engram.compressed_vocab_size`
    /// (required keys, `core_validate_v41.c:152-153`; no native reader yet,
    /// but the C GGUF path fills them and the host shape must not diverge).
    pub engram_vocab: u32,
    pub engram_cvocab: u32,
    /// `deepseek4.attention.candidate.source_layer` / `.topk_blocks` /
    /// `.block_size`: the layer that screens candidate blocks and the two
    /// two-level-topk widths (`core_validate_v41.c:113-115`). Stored signed
    /// like the engine's `g_ds4_v41` fields.
    pub candidate_source_layer: i32,
    pub candidate_topk_blocks: i32,
    pub candidate_block_size: i32,
    /// `deepseek4.mtp.tower_count`; 0 = the GGUF carries no towers and the
    /// engine decodes one token at a time (`core_validate_v41.c:85-90`).
    pub mtp_towers: u32,
    /// `deepseek4.mtp.expert_count`; per-expert form only.
    pub mtp_experts: u32,
    /// `deepseek4.mtp.block_size` / `.expert_used_count` / `.noise_token_id`:
    /// the draft scalars (`core_validate_v41.c:97-99`).  None when absent
    /// (or when the GGUF carries no towers); the drafter arms only when all
    /// four scalars and the target list are present (`mtp_armed`).
    pub mtp_block: Option<u32>,
    pub mtp_used: Option<u32>,
    pub mtp_noise_id: Option<u32>,
    /// `deepseek4.mtp.target_layers` (the layers whose attention input feeds
    /// `main_proj`); empty when the key is absent. The engine reads the same
    /// array when the draft parameters are complete
    /// (`core_validate_v41.c:100-110`).
    pub mtp_targets: Vec<u32>,
    /// `deepseek4.mtp.markov_rank`; None when absent. The markov head and the
    /// confidence head are shaped by it.
    pub mtp_markov_rank: Option<u32>,
}

impl V41Wire {
    /// The engine's arming rule (`core_validate_v41.c:97-110`): the towers,
    /// all four draft scalars and a nonempty target list, or the drafter
    /// stays off and the engine warns instead of guessing a default.
    pub fn mtp_armed(&self) -> bool {
        self.mtp_towers > 0
            && self.mtp_block.is_some()
            && self.mtp_used.is_some()
            && self.mtp_noise_id.is_some()
            && self.mtp_markov_rank.is_some()
            && !self.mtp_targets.is_empty()
    }
}

/// C `v41_arr_i32`: an INT32 or UINT32 array, read signed. Returns the values.
fn i32_array(g: &GgufFile, key: &'static str) -> Result<Vec<i32>, V41WireError> {
    let arr = g.get_array(key).ok_or(V41WireError::MissingKey(key))?;
    if arr.typ != GGUF_VALUE_INT32 && arr.typ != GGUF_VALUE_UINT32 {
        return Err(V41WireError::ArrayType(key));
    }
    let data = g.as_bytes();
    let mut out = Vec::with_capacity(arr.len.min(1024) as usize);
    let mut pos = arr.data_pos;
    for _ in 0..arr.len {
        let b = data.get(pos..pos + 4).ok_or(V41WireError::ArrayLen(key))?;
        out.push(i32::from_le_bytes(b.try_into().unwrap()));
        pos += 4;
    }
    Ok(out)
}

fn layer_ids(g: &GgufFile, key: &'static str, n_layer: u32) -> Result<Vec<u32>, V41WireError> {
    let raw = i32_array(g, key)?;
    let mut out = Vec::with_capacity(raw.len());
    for id in raw {
        if id < 0 || id as u32 >= n_layer {
            return Err(V41WireError::LayerRange(key, i64::from(id)));
        }
        out.push(id as u32);
    }
    Ok(out)
}

impl V41Wire {
    /// Load the wiring from the artifact's metadata. Hard stops mirror
    /// `v41_load_metadata`; the compress ratios are read here because the
    /// bind and layout conditionals need them per layer, and the range rule
    /// (0/1/2) stays in `validate_compress` (`validate.rs:257`), which every
    /// load path runs first.
    pub fn load(g: &GgufFile, shape: &Shape) -> Result<Self, V41WireError> {
        if shape.variant != Variant::DeepSeek41Flash {
            return Err(V41WireError::NotV41);
        }
        let n = shape.n_layer as usize;

        // The context length is the engine's only context source
        // (core_validate_v41.c:51-53): a missing key is a hard stop and a
        // zero or over-u32::MAX value is invalid.
        let ctx = g
            .get_u64_compat(KEY_CTX)
            .ok_or(V41WireError::MissingKey(KEY_CTX))?;
        if ctx == 0 || ctx > u32::MAX as u64 {
            return Err(V41WireError::CtxInvalid);
        }
        let ctx = ctx as u32;

        let ratios = i32_array(g, KEY_COMPRESS_RATIOS)?;
        if ratios.len() < n {
            return Err(V41WireError::ArrayLen(KEY_COMPRESS_RATIOS));
        }
        let compress_ratios: Vec<u32> = ratios[..n].iter().map(|&v| v.max(0) as u32).collect();

        let mut is_kv_source = vec![false; n];
        for id in layer_ids(g, KEY_KV_SOURCE_LAYERS, shape.n_layer)? {
            is_kv_source[id as usize] = true;
        }
        let mut is_index_source = vec![false; n];
        for id in layer_ids(g, KEY_INDEX_SOURCE_LAYERS, shape.n_layer)? {
            is_index_source[id as usize] = true;
        }

        // A compressing layer reads the nearest source at or before it, and a
        // kv source must also be an index source because the indexer keys come
        // from its latent (core_validate_v41.c:69-83).
        let mut kv_source_of = vec![-1i16; n];
        let mut index_source_of = vec![-1i16; n];
        let mut last_kv: i16 = -1;
        let mut last_idx: i16 = -1;
        for il in 0..n {
            if is_kv_source[il] {
                last_kv = il as i16;
            }
            if is_index_source[il] {
                last_idx = il as i16;
            }
            if compress_ratios[il] == 0 {
                continue;
            }
            if last_kv < 0 || last_idx < 0 {
                return Err(V41WireError::NoSourceBefore(il as u32));
            }
            kv_source_of[il] = last_kv;
            index_source_of[il] = last_idx;
            if is_kv_source[il] && !is_index_source[il] {
                return Err(V41WireError::KvSourceNotIndexSource(il as u32));
            }
        }

        let engram_layers = layer_ids(g, KEY_ENGRAM_LAYER_IDS, shape.n_layer)?;
        if engram_layers.len() > V41_MAX_ENGRAM {
            return Err(V41WireError::ArrayLen(KEY_ENGRAM_LAYER_IDS));
        }
        let mut engram_index_of = vec![-1i16; n];
        for (i, &il) in engram_layers.iter().enumerate() {
            engram_index_of[il as usize] = i as i16;
        }
        // The engram table dimensions are required even with zero engram
        // layers, mirroring required_u32 (core_validate_v41.c:149-154).
        let engram_max_ngram = g
            .get_u32(KEY_ENGRAM_MAX_NGRAM)
            .ok_or(V41WireError::MissingKey(KEY_ENGRAM_MAX_NGRAM))?;
        let engram_heads = g
            .get_u32(KEY_ENGRAM_HEADS)
            .ok_or(V41WireError::MissingKey(KEY_ENGRAM_HEADS))?;
        let engram_head_dim = g
            .get_u32(KEY_ENGRAM_HEAD_DIM)
            .ok_or(V41WireError::MissingKey(KEY_ENGRAM_HEAD_DIM))?;
        let engram_pad = g
            .get_u32(KEY_ENGRAM_PAD)
            .ok_or(V41WireError::MissingKey(KEY_ENGRAM_PAD))?;
        let engram_vocab = g
            .get_u32(KEY_ENGRAM_VOCAB)
            .ok_or(V41WireError::MissingKey(KEY_ENGRAM_VOCAB))?;
        let engram_cvocab = g
            .get_u32(KEY_ENGRAM_CVOCAB)
            .ok_or(V41WireError::MissingKey(KEY_ENGRAM_CVOCAB))?;

        // The candidate two-level-topk triple is required (core_validate_v41.c:113-115).
        let candidate_source_layer = g
            .get_u32(KEY_CANDIDATE_SOURCE_LAYER)
            .ok_or(V41WireError::MissingKey(KEY_CANDIDATE_SOURCE_LAYER))?;
        let candidate_topk_blocks = g
            .get_u32(KEY_CANDIDATE_TOPK_BLOCKS)
            .ok_or(V41WireError::MissingKey(KEY_CANDIDATE_TOPK_BLOCKS))?;
        let candidate_block_size = g
            .get_u32(KEY_CANDIDATE_BLOCK_SIZE)
            .ok_or(V41WireError::MissingKey(KEY_CANDIDATE_BLOCK_SIZE))?;

        // One u64 row count per engram layer, and per-layer offsets into the
        // shard (core_validate_v41.c:119-148).
        let rows_arr = g
            .get_array(KEY_ENGRAM_ROWS)
            .ok_or(V41WireError::MissingKey(KEY_ENGRAM_ROWS))?;
        if rows_arr.typ != crate::gguf::GGUF_VALUE_UINT64 {
            return Err(V41WireError::ArrayType(KEY_ENGRAM_ROWS));
        }
        let mut engram_rows = Vec::with_capacity(rows_arr.len as usize);
        {
            let data = g.as_bytes();
            let mut pos = rows_arr.data_pos;
            for _ in 0..rows_arr.len {
                let b = data
                    .get(pos..pos + 8)
                    .ok_or(V41WireError::ArrayLen(KEY_ENGRAM_ROWS))?;
                engram_rows.push(u64::from_le_bytes(b.try_into().unwrap()));
                pos += 8;
            }
        }
        if engram_rows.len() != engram_layers.len() {
            return Err(V41WireError::EngramRowsMismatch);
        }
        let mut engram_weight_off = Vec::with_capacity(engram_layers.len());
        let mut engram_scale_off = Vec::with_capacity(engram_layers.len());
        let mut engram_table_path = Vec::with_capacity(engram_layers.len());
        for i in 0..engram_layers.len() {
            let key = format!("deepseek4.engram.{i}.table_path");
            let s = g
                .get_string(&key)
                .filter(|s| !s.is_empty())
                .ok_or(V41WireError::MissingKeyAt(key.clone()))?;
            engram_table_path.push(String::from_utf8_lossy(s).into_owned());
            let key = format!("deepseek4.engram.{i}.weight_offset");
            engram_weight_off.push(
                g.get_u64_compat(&key)
                    .ok_or(V41WireError::MissingKeyAt(key.clone()))?,
            );
            let key = format!("deepseek4.engram.{i}.scale_offset");
            engram_scale_off.push(
                g.get_u64_compat(&key)
                    .ok_or(V41WireError::MissingKeyAt(key.clone()))?,
            );
        }

        // Towers are optional: a GGUF converted before 2026-09-15 carries
        // none, and the engine then decodes one token at a time
        // (core_validate_v41.c:85-90).
        let mtp_towers = g.get_u32(KEY_MTP_TOWER_COUNT).unwrap_or(0);
        let mtp_experts = g.get_u32(KEY_MTP_EXPERT_COUNT).unwrap_or(0);
        if mtp_towers > MTP_MAX_TOWERS || mtp_experts > MTP_MAX_EXPERTS {
            return Err(V41WireError::TowerLimit);
        }
        // The four draft scalars, read only when the towers are there
        // (core_validate_v41.c:97-99).  A missing scalar leaves the drafter
        // unarmed -- never a guessed default, because a wrong guess is
        // silent: the draft is simply wrong and acceptance collapses to 0.
        let (mtp_block, mtp_used, mtp_noise_id, mtp_markov_rank) = if mtp_towers > 0 {
            (
                g.get_u32(KEY_MTP_BLOCK_SIZE),
                g.get_u32(KEY_MTP_EXPERT_USED),
                g.get_u32(KEY_MTP_NOISE_TOKEN_ID),
                g.get_u32(KEY_MTP_MARKOV_RANK),
            )
        } else {
            (None, None, None, None)
        };
        let mtp_have = mtp_block.is_some()
            && mtp_used.is_some()
            && mtp_noise_id.is_some()
            && mtp_markov_rank.is_some();
        let mtp_targets = if mtp_have || g.get_array(KEY_MTP_TARGET_LAYERS).is_some() {
            // With the scalars complete the engine requires this list and
            // exits when it is missing (v41_arr_i32), so a missing key here
            // is a hard stop, not an empty list.
            let targets = layer_ids(g, KEY_MTP_TARGET_LAYERS, shape.n_layer)?;
            if targets.len() > MTP_MAX_TOWERS as usize * 2 {
                // The engine dies on an over-cap array (v41_arr_i32).
                return Err(V41WireError::ArrayLen(KEY_MTP_TARGET_LAYERS));
            }
            targets
        } else {
            Vec::new()
        };

        Ok(Self {
            ctx,
            compress_ratios,
            is_kv_source,
            kv_source_of,
            is_index_source,
            index_source_of,
            engram_layers,
            engram_index_of,
            engram_max_ngram,
            engram_heads,
            engram_head_dim,
            engram_rows,
            engram_weight_off,
            engram_scale_off,
            engram_table_path,
            engram_pad,
            engram_vocab,
            engram_cvocab,
            candidate_source_layer: candidate_source_layer as i32,
            candidate_topk_blocks: candidate_topk_blocks as i32,
            candidate_block_size: candidate_block_size as i32,
            mtp_towers,
            mtp_experts,
            mtp_block,
            mtp_used,
            mtp_noise_id,
            mtp_targets,
            mtp_markov_rank,
        })
    }

    /// `--engram-dir`: the converter wrote the converting machine's absolute
    /// paths, so the loader swaps the directory and keeps the file name
    /// instead of rewriting the published GGUF (`core_validate_v41.c:11-13,
    /// 134-141`). The C caps the result at 1024 bytes; a Rust `String` has no
    /// fixed buffer to overflow.
    pub fn apply_engram_dir(&mut self, dir: &str) {
        for path in &mut self.engram_table_path {
            let name = path.rsplit('/').next().unwrap_or(path.as_str());
            *path = format!("{dir}/{name}");
        }
    }

    /// The tower expert form: one VQ blob per tower, or per-expert FP4
    /// tensors. The engine probes the blob and requires the per-expert set
    /// only when it is absent (core_bind_v41.c:126-134), so the choice is an
    /// inventory fact, not a metadata one.
    pub fn tower_uses_blob(&self, inventory: &TensorInventory, tower: u32) -> bool {
        inventory
            .find(&format!("mtp.{tower}.ffn_exps_vq.blob"))
            .is_some()
    }
}
