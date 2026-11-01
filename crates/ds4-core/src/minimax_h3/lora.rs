//! The FastH3 distilled adapter merged into a derived MiniMax-H3 checkpoint.
//!
//! The released `MiniMax-H3/FL2VA/transformer` runs the *base* model, which
//! needs twenty denoise evaluations. The reference's stage 1 runs four of them
//! because it merges `FastH3-VSA/vsa-datafree/adapter_model.safetensors` before
//! its FP8 conversion (`Sol-H3-Spark/configs/default.json:7` `dit:
//! W8A8_FP8_after_BF16_LoRA_merge`, `:6` `lora: FastH3_VSA_DataFree`,
//! `:9` `updates: 4`). This module performs that merge offline, into a derived
//! checkpoint; the released directory is only ever read.
//!
//! Source of truth for the adapter's semantics is FastVideo at
//! `3d8ac9d14bd697a89ede8f170cbfbca012a9edcc` (the reference's pin,
//! `Sol-H3-Spark/configs/dependencies.json` `sources[fastvideo_root]`), read
//! at its pristine bytes:
//!
//! - `fastvideo/models/loader/lora_patch.py:1-32` names the three payload
//!   classes and what each targets: `.diff` -> `.weight` and `.diff_b` ->
//!   `.bias` (additive), `.set_weight` -> `.weight` (a whole parameter the
//!   base checkpoint does not carry). Its example is this adapter's own case:
//!   "MiniMax H3's VSA `to_gate_compress` ... exists only under the
//!   sparse-attention backend, and `load_model_from_full_model_state_dict`
//!   otherwise zero-initializes it".
//! - `:53-56` `ADDITIVE_SUFFIXES` / `REPLACEMENT_SUFFIXES`, tested longest
//!   first so `.diff_b` never matches as `.diff`.
//! - `:170-186` `apply_to`: `tensor.to(float32) + delta.to(float32) *
//!   strength`, the sum taken in float32 whatever the operands are.
//! - `:212-227` `report_unapplied`: an adapter key the loader silently ignores
//!   is the failure this path exists to fix. This module makes it an error.
//! - `fastvideo/layers/lora/linear.py:183-195` the low-rank half:
//!   `lora_delta = B @ A` with both operands cast to the base weight's dtype,
//!   `scale = alpha/rank if alpha and rank else 1.0`, then `data += lora_delta`.
//!   This adapter states no `alpha`, so the scale is 1.0; the reference also
//!   sets `lora_strength` 1.0 (`Sol-H3-Spark/runtime/stage1.py:301`).
//!
//! The adapter addresses the checkpoint the way the *diffusers* packaging names
//! it (`transformer_blocks.N.attn.to_q`, `ff.net.0.proj`), which is not the
//! packaging this engine loads. The two packagings were compared byte for byte,
//! one row range at a time, over HTTP range requests against
//! `MiniMaxAI/MiniMax-H3/transformer` and `.../FL2VA/transformer`, at both
//! `9bfb6693` (the adapter's `base_revision`) and `42ed227e` (the revision on
//! disk; every shard's LFS oid, its size and the index are identical between
//! them, so the adapter's base is exactly what this engine serves):
//!
//! - The engine packaging has no `to_q`/`to_k`/`to_v`: it fuses them into
//!   `blocks.N.attn.qkv_proj.weight` at `3 * heads * head_dim` rows, ordered
//!   per head, `[q | k | v]`. Rows `384h .. 384h + 128` are `to_q` rows
//!   `128h .. 128h + 128`, `+128` is `to_k`, `+256` is `to_v` - the layout the
//!   engine's own kernel reads (`h3_gpu_cuda.cu:3093-3097`, the `grouped`
//!   branch, reached with `grouped = 1` for the DiT at `h3_dit.c:1912` through
//!   `h3_gpu_grouped_qkv_linear_rope_bf16`, `h3_gpu_cuda.cu:3241-3250`; the
//!   port mirrors it at `native/h3/h3_dit_gpu.cu:1004-1006`).
//! - `mlp.fc1.weight` and `ff.net.0.proj.weight` hold the same two FFN halves
//!   swapped: engine rows `0..14336` are the packaging's rows
//!   `14336..28672`. Both engines agree on which half is the gate - the
//!   reference kernel takes the first half (`h3_gpu_cuda.cu:2842-2851`) and
//!   FastVideo the second (`models/dits/minimax_h3.py:126-129`
//!   `hidden_states, gate = hidden_states.chunk(2, dim=-1)`), so the byte
//!   comparison and both implementations agree independently.
//! - Every other path is the same tensor value for value: `attn.to_out.0` ->
//!   `attn.out_proj`, `ff.net.2` -> `mlp.fc2`, `norm_q`/`norm_k` ->
//!   `attn.q_norm`/`attn.k_norm`, `context_embedder` -> `condition_proj`,
//!   `proj_in` -> `video_patch_proj`, `proj_out` -> `final_layer.video_out`,
//!   `audio_proj_in` -> `audio_patch_proj`, `audio_proj_out` ->
//!   `final_layer.audio_out`, `norm_out.linear` ->
//!   `final_layer.adaln_proj.linear`, `norm_out.norm` -> `final_layer.norm`,
//!   `time_embedder.linear_1/2` -> `time_embedder.proj_in/proj_out`, and the
//!   blocks' own norms and AdaLN.
//!
//! Deviations, both named:
//!
//! 1. The reference merges into a diffusers model and only then quantizes, so
//!    its merged weights are FP8. This port writes the delta into the engine
//!    packaging's own tensor and keeps that tensor's released dtype: an F32
//!    boundary projection stays F32, a BF16 tensor is rounded once at the end.
//! 2. The dense half is added in F32 and rounded to the destination dtype,
//!    which is `apply_to`'s contract; the low-rank half rounds the product to
//!    the destination dtype *before* adding, which is what a bf16
//!    `torch.matmul` followed by an in-place bf16 `+=` does.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::minimax_h3::layout::{Config, LayoutError};
use crate::minimax_h3::safetensors::{
    self, Dtype, Entry, Shard, ShardWriter, StError, Store, Tensor,
};
use crate::minimax_h3::vision::{bf16_to_f32, f32_to_bf16};

/// `__metadata__.format` of the adapter this module merges.
pub const ADAPTER_FORMAT: &str = "fastvideo-lora-v2";
/// `__metadata__.rank`, the width of every low-rank factor.
pub const ADAPTER_RANK: u64 = 64;
/// The whole-parameter payload's only module: a compression gate the released
/// checkpoint does not carry and this engine does not read.
const GATE_SUFFIX: &str = "attn.to_gate_compress";
/// Where the whole-parameter payload goes, outside the engine's component
/// directories.
const SET_WEIGHT_DIR: &str = "set_weight";

/// The released partition this module derives.
pub const PARTITION: &str = "FL2VA";
const TRANSFORMER: &str = "transformer";
const CONFIG_FILE: &str = "config.json";
const INDEX_FILE: &str = "model.safetensors.index.json";

/// Target elements per parallel task: about 16 MiB of F32 accumulator plus
/// its output. A wider tensor is tiled down to this many rows.
const TASK_ELEMENTS: u64 = 4 << 20;
/// Copy chunk for a tensor that carries no delta.
const COPY_CHUNK: usize = 16 << 20;

#[derive(Debug)]
pub enum MergeError {
    Layout(LayoutError),
    Reader(StError),
    Io {
        path: PathBuf,
        detail: String,
    },
    /// The adapter's own declaration does not match what it holds.
    Adapter {
        detail: String,
    },
    /// An adapter key no rule here places. Never skipped.
    Unresolved {
        key: String,
        detail: String,
    },
    Plan {
        tensor: String,
        detail: String,
    },
    /// The destination already exists: a derivation never overwrites.
    Exists(PathBuf),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Layout(error) => write!(f, "layout: {}", error.token()),
            Self::Reader(error) => write!(f, "safetensors: {error}"),
            Self::Io { path, detail } => write!(f, "{}: {detail}", path.display()),
            Self::Adapter { detail } => write!(f, "adapter: {detail}"),
            Self::Unresolved { key, detail } => write!(f, "unresolved key {key}: {detail}"),
            Self::Plan { tensor, detail } => write!(f, "{tensor}: {detail}"),
            Self::Exists(path) => write!(f, "{} already exists", path.display()),
        }
    }
}

impl std::error::Error for MergeError {}

impl From<StError> for MergeError {
    fn from(error: StError) -> Self {
        Self::Reader(error)
    }
}

impl From<LayoutError> for MergeError {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}

/// Which of the adapter's three payload classes a key belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Payload {
    /// `.lora_A.weight` / `.lora_B.weight`, a rank-64 product.
    LowRank,
    /// `.diff` / `.diff_b`, an additive delta.
    Additive,
    /// `.set_weight`, the whole parameter.
    Whole,
}

/// Where a delta's rows land in the served tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The module is the tensor.
    Identity,
    /// `attn.qkv_proj`: one interleaved slot per head, `part` 0/1/2 for q/k/v.
    Qkv { part: u64 },
    /// `mlp.fc1`: two halves, swapped.
    Swap,
}

/// What one adapter key states about the served tensor it lands on.
#[derive(Clone, Debug)]
pub enum Delta {
    /// `B @ A` onto the destination's rows in `Rows` order.
    LowRank { a: String, b: String, rows: Rows },
    /// `apply_to`'s F32 add.
    Add { diff: String },
}

impl Delta {
    fn low_rank(a: String, b: String, rows: Rows) -> Delta {
        Delta::LowRank { a, b, rows }
    }
}

/// A row range of one delta's destination and the source rows behind it. For
/// every rule here the source rows are contiguous, so a segment is one
/// `B @ A` block.
#[derive(Clone, Copy, Debug)]
struct Segment {
    delta: usize,
    dest: u64,
    src: u64,
    rows: u64,
}

/// The derivation, built and checked from the adapter before any byte moves.
pub struct Plan {
    touches: BTreeMap<String, Vec<Delta>>,
    segments: BTreeMap<String, Vec<Segment>>,
    /// Adapter modules whose parameter the released checkpoint lacks.
    whole: Vec<String>,
    low_rank_modules: u64,
    additive_keys: u64,
    /// The adapter's own tensor counts, by payload class.
    tensors: BTreeMap<Payload, u64>,
}

impl Plan {
    /// Resolve every adapter key. A key no rule places is an error.
    pub fn build(adapter: &Adapter, store: &Store, config: &Config) -> Result<Plan, MergeError> {
        let mut pairs: Vec<(String, Delta)> = Vec::new();
        let mut whole: Vec<String> = Vec::new();
        let mut counts: BTreeMap<Payload, u64> = BTreeMap::new();
        let mut additive_keys = 0u64;
        // A rank-64 delta is stated by two keys, so the pair is collected by
        // module and both factors are required before anything is planned.
        let mut low_rank: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();

        for tensor in adapter.tensors() {
            let key = &tensor.name;
            let payload = classify(key).ok_or_else(|| MergeError::Unresolved {
                key: key.clone(),
                detail: "no payload class".into(),
            })?;
            let module = module_of(key, payload)
                .ok_or_else(|| MergeError::Unresolved {
                    key: key.clone(),
                    detail: "no module path".into(),
                })?
                .to_string();

            *counts.entry(payload).or_insert(0) += 1;

            match payload {
                Payload::Whole => {
                    if !module.ends_with(GATE_SUFFIX) {
                        return Err(MergeError::Unresolved {
                            key: key.clone(),
                            detail: "a whole-parameter payload outside to_gate_compress".into(),
                        });
                    }
                    if whole.contains(&module) {
                        return Err(MergeError::Adapter {
                            detail: format!("{module} carries two whole-parameter payloads"),
                        });
                    }
                    whole.push(module);
                }
                Payload::Additive => {
                    let bias = key.ends_with(".diff_b");
                    let (name, rows) = destination(&module, bias)?;
                    if rows != Rows::Identity {
                        return Err(MergeError::Plan {
                            tensor: name,
                            detail: format!("{key} is additive onto a reordered tensor"),
                        });
                    }
                    pairs.push((name, Delta::Add { diff: key.clone() }));
                    additive_keys += 1;
                }
                Payload::LowRank => {
                    let slot = low_rank.entry(module).or_default();
                    let target = if key.ends_with(".lora_A.weight") {
                        &mut slot.0
                    } else {
                        &mut slot.1
                    };
                    if target.replace(key.clone()).is_some() {
                        return Err(MergeError::Adapter {
                            detail: format!("{key} is stated twice"),
                        });
                    }
                }
            }
        }

        let low_rank_modules = low_rank.len();
        for (module, (a, b)) in low_rank {
            let (a, b) = match (a, b) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    return Err(MergeError::Adapter {
                        detail: format!("{module} states only one of its two factors"),
                    })
                }
            };
            let (name, rows) = destination(&module, false)?;
            pairs.push((name, Delta::low_rank(a, b, rows)));
        }

        // Every touched tensor must exist.
        let mut touches: BTreeMap<String, Vec<Delta>> = BTreeMap::new();
        for (name, delta) in pairs {
            if store.find(&name).is_none() {
                return Err(MergeError::Plan {
                    tensor: name,
                    detail: "the served shard set does not hold it".into(),
                });
            }

            if let Delta::LowRank { a, b, .. } = &delta {
                if adapter.find(a).is_none() || adapter.find(b).is_none() {
                    return Err(MergeError::Adapter {
                        detail: format!("{name} names a factor the adapter does not hold"),
                    });
                }
            }

            touches.entry(name).or_default().push(delta);
        }

        let mut plan = Plan {
            touches,
            segments: BTreeMap::new(),
            whole,
            low_rank_modules: low_rank_modules as u64,
            additive_keys,
            tensors: counts,
        };
        plan.segment(store, config)?;

        Ok(plan)
    }

    /// Lay every delta's rows out and prove they cover each destination
    /// tensor exactly once: two deltas on one row would double a weight.
    fn segment(&mut self, store: &Store, config: &Config) -> Result<(), MergeError> {
        for (name, deltas) in &self.touches {
            let (_, tensor) = store.find(name).expect("checked in build");
            let (rows, _) = geometry(tensor)?;
            let mut segments = Vec::new();

            for (index, delta) in deltas.iter().enumerate() {
                match delta {
                    Delta::Add { .. } => segments.push(Segment {
                        delta: index,
                        dest: 0,
                        src: 0,
                        rows,
                    }),
                    Delta::LowRank { rows: order, .. } => match order {
                        Rows::Identity => segments.push(Segment {
                            delta: index,
                            dest: 0,
                            src: 0,
                            rows,
                        }),
                        Rows::Swap => {
                            if rows % 2 != 0 {
                                return Err(MergeError::Plan {
                                    tensor: name.clone(),
                                    detail: format!("{rows} rows hold no two swapped halves"),
                                });
                            }
                            let half = rows / 2;
                            segments.push(Segment {
                                delta: index,
                                dest: 0,
                                src: half,
                                rows: half,
                            });
                            segments.push(Segment {
                                delta: index,
                                dest: half,
                                src: 0,
                                rows: half,
                            });
                        }
                        Rows::Qkv { part } => {
                            let head = config.head_dim;
                            if head == 0 || config.heads * head * 3 != rows {
                                return Err(MergeError::Plan {
                                    tensor: name.clone(),
                                    detail: format!(
                                        "{rows} rows are not three {}x{head} projections",
                                        config.heads
                                    ),
                                });
                            }
                            for h in 0..config.heads {
                                segments.push(Segment {
                                    delta: index,
                                    dest: h * head * 3 + part * head,
                                    src: h * head,
                                    rows: head,
                                });
                            }
                        }
                    },
                }
            }

            segments.sort_by_key(|segment| segment.dest);
            let mut covered = 0u64;
            for segment in &segments {
                if segment.dest != covered {
                    return Err(MergeError::Plan {
                        tensor: name.clone(),
                        detail: format!("the deltas skip or overlap row {covered}"),
                    });
                }
                covered += segment.rows;
            }

            if covered != rows {
                return Err(MergeError::Plan {
                    tensor: name.clone(),
                    detail: format!("the deltas cover {covered} of {rows} rows"),
                });
            }

            self.segments.insert(name.clone(), segments);
        }

        Ok(())
    }

    pub fn touches(&self) -> &BTreeMap<String, Vec<Delta>> {
        &self.touches
    }

    /// Adapter modules whose parameter the released checkpoint does not carry.
    pub fn whole(&self) -> &[String] {
        &self.whole
    }

    pub fn low_rank_modules(&self) -> u64 {
        self.low_rank_modules
    }

    /// The adapter's tensor count for one payload class.
    pub fn tensors(&self, payload: Payload) -> u64 {
        self.tensors.get(&payload).copied().unwrap_or(0)
    }

    pub fn additive_keys(&self) -> u64 {
        self.additive_keys
    }
}

/// The released adapter, read as its own metadata describes it.
#[derive(Debug)]
pub struct Adapter {
    shard: Shard,
}

impl Adapter {
    /// Open the adapter and prove it is the artifact it claims: the format
    /// string and the rank are fixed, and the three payload counts the
    /// metadata declares are counted from the header instead of trusted.
    ///
    /// Two artifact classes are read. The FastH3 distilled adapter declares all
    /// of the above. A few-step adapter such as
    /// `larryvrh/MiniMax-H3-Turbo-Lora` declares none of it and only states
    /// `W_eff = W + lora_B @ lora_A`, the same scale-1.0 convention this merge
    /// applies, so it is read as bare. Bare loses only the count check, which
    /// was a second opinion: `Plan::build` still proves every adapter key
    /// resolves to a served tensor and every destination's rows are covered
    /// exactly once. A partially declared adapter is refused rather than read
    /// as bare, so a damaged artifact cannot pass as an undeclared one.
    pub fn open(path: &Path) -> Result<Adapter, MergeError> {
        let shard = safetensors::read_shard(path)?;

        if !shard.metadata.contains_key("format") {
            for key in ["rank", "low_rank_tensors", "diff_tensors", "set_weight_tensors"] {
                if shard.metadata.contains_key(key) {
                    return Err(MergeError::Adapter {
                        detail: format!("metadata {key} without a format"),
                    });
                }
            }
            return Ok(Adapter { shard });
        }

        for (key, expected) in [("format", ADAPTER_FORMAT), ("rank", "64")] {
            let got = shard.metadata.get(key).ok_or_else(|| MergeError::Adapter {
                detail: format!("no metadata key {key}"),
            })?;

            if got != expected {
                return Err(MergeError::Adapter {
                    detail: format!("metadata {key} is {got}, expected {expected}"),
                });
            }
        }

        let adapter = Adapter { shard };
        for (payload, key) in [
            (Payload::LowRank, "low_rank_tensors"),
            (Payload::Additive, "diff_tensors"),
            (Payload::Whole, "set_weight_tensors"),
        ] {
            let declared: u64 = adapter
                .shard
                .metadata
                .get(key)
                .ok_or_else(|| MergeError::Adapter {
                    detail: format!("no metadata key {key}"),
                })?
                .parse()
                .map_err(|_| MergeError::Adapter {
                    detail: format!("metadata {key} is not a number"),
                })?;

            let counted = adapter
                .tensors()
                .iter()
                .filter(|tensor| classify(&tensor.name) == Some(payload))
                .count() as u64;

            if counted != declared {
                return Err(MergeError::Adapter {
                    detail: format!(
                        "{payload:?} tensors: header holds {counted}, metadata says {declared}"
                    ),
                });
            }
        }

        Ok(adapter)
    }

    pub fn shard(&self) -> &Shard {
        &self.shard
    }

    pub fn tensors(&self) -> &[Tensor] {
        &self.shard.tensors
    }

    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.shard.find(name)
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>, MergeError> {
        let tensor = self.shard.find(name).ok_or_else(|| MergeError::Adapter {
            detail: format!("no tensor {name}"),
        })?;
        let mut buffer = vec![0u8; tensor.bytes() as usize];
        self.shard.read_data(tensor, &mut buffer)?;
        Ok(buffer)
    }
}

/// `lora_patch.py:53-56`: the suffix decides the payload, longest first.
fn classify(key: &str) -> Option<Payload> {
    if key.ends_with(".lora_A.weight") || key.ends_with(".lora_B.weight") {
        return Some(Payload::LowRank);
    }
    if key.ends_with(".diff_b") || key.ends_with(".diff") {
        return Some(Payload::Additive);
    }
    if key.ends_with(".set_weight") {
        return Some(Payload::Whole);
    }
    None
}

/// The module path a key belongs to.
fn module_of(key: &str, payload: Payload) -> Option<&str> {
    let suffix = match payload {
        Payload::LowRank if key.ends_with(".lora_A.weight") => ".lora_A.weight",
        Payload::LowRank if key.ends_with(".lora_B.weight") => ".lora_B.weight",
        Payload::Additive if key.ends_with(".diff_b") => ".diff_b",
        Payload::Additive if key.ends_with(".diff") => ".diff",
        Payload::Whole if key.ends_with(".set_weight") => ".set_weight",
        _ => return None,
    };
    Some(&key[..key.len() - suffix.len()])
}

/// The nine modules outside the two block families, named as FastVideo's own
/// mapping renames them (`configs/models/dits/minimax_h3.py:37-45`) and as the
/// byte comparison above confirmed in the served checkpoint.
const GLOBAL_MAP: &[(&str, &str)] = &[
    ("audio_proj_in", "audio_patch_proj"),
    ("audio_proj_out", "final_layer.audio_out"),
    ("context_embedder", "condition_proj"),
    ("norm_out.linear", "final_layer.adaln_proj.linear"),
    ("norm_out.norm", "final_layer.norm"),
    ("proj_in", "video_patch_proj"),
    ("proj_out", "final_layer.video_out"),
    ("time_embedder.linear_1", "time_embedder.proj_in"),
    ("time_embedder.linear_2", "time_embedder.proj_out"),
];

/// The served tensor a module's weight (or bias) lands on, and the row order
/// that carries it there.
fn destination(module: &str, bias: bool) -> Result<(String, Rows), MergeError> {
    // The nine global projections carry both a weight and a bias, and the
    // adapter states a `.diff_b` for every one of them.
    for (from, to) in GLOBAL_MAP {
        if module == *from {
            let suffix = if bias { "bias" } else { "weight" };
            return Ok((format!("{to}.{suffix}"), Rows::Identity));
        }
    }

    let (prefix, rest) = if let Some(rest) = module.strip_prefix("transformer_blocks.") {
        let (index, rest) = split_index(rest)?;
        (format!("blocks.{index}."), rest)
    } else if let Some(rest) = module.strip_prefix("token_refiner.refiner_blocks.") {
        let (index, rest) = split_index(rest)?;
        (format!("token_refiner.blocks.{index}."), rest)
    } else if let Some(rest) = module.strip_prefix("token_refiner.blocks.") {
        // Already in the served namespace: a few-step adapter such as
        // `larryvrh/MiniMax-H3-Turbo-Lora` states its deltas against the
        // tensors this engine loads, not against the diffusers packaging.
        let (index, rest) = split_index(rest)?;
        (format!("token_refiner.blocks.{index}."), rest)
    } else if let Some(rest) = module.strip_prefix("blocks.") {
        let (index, rest) = split_index(rest)?;
        (format!("blocks.{index}."), rest)
    } else if module == "final_layer.adaln_proj.linear" {
        // The one served global projection that carries a low-rank delta.
        return Ok((
            format!("{module}.{}", if bias { "bias" } else { "weight" }),
            Rows::Identity,
        ));
    } else {
        return Err(MergeError::Unresolved {
            key: module.to_string(),
            detail: "not in either block family and not a global projection".into(),
        });
    };

    let identity = |name: String| (name, Rows::Identity);

    Ok(match rest {
        "attn.to_q" if !bias => (
            format!("{prefix}attn.qkv_proj.weight"),
            Rows::Qkv { part: 0 },
        ),
        "attn.to_k" if !bias => (
            format!("{prefix}attn.qkv_proj.weight"),
            Rows::Qkv { part: 1 },
        ),
        "attn.to_v" if !bias => (
            format!("{prefix}attn.qkv_proj.weight"),
            Rows::Qkv { part: 2 },
        ),
        "attn.to_out.0" if !bias => identity(format!("{prefix}attn.out_proj.weight")),
        "ff.net.0.proj" if !bias => (format!("{prefix}mlp.fc1.weight"), Rows::Swap),
        "ff.net.2" if !bias => identity(format!("{prefix}mlp.fc2.weight")),
        "adaln_proj.linear" => identity(format!(
            "{prefix}adaln_proj.linear.{}",
            if bias { "bias" } else { "weight" }
        )),
        "norm1" if !bias => identity(format!("{prefix}norm1.weight")),
        "norm2" if !bias => identity(format!("{prefix}norm2.weight")),
        // The served namespace's own names. A fused `qkv_proj` already holds
        // `[q | k | v]` head-major and `fc1` already holds the fused gate/up
        // pair, so each delta adds row for row with nothing reordered.
        "attn.qkv_proj" if !bias => identity(format!("{prefix}attn.qkv_proj.weight")),
        "attn.out_proj" if !bias => identity(format!("{prefix}attn.out_proj.weight")),
        "mlp.fc1" if !bias => identity(format!("{prefix}mlp.fc1.weight")),
        "mlp.fc2" if !bias => identity(format!("{prefix}mlp.fc2.weight")),
        other => {
            return Err(MergeError::Unresolved {
                key: module.to_string(),
                detail: format!("no rule for {other}"),
            })
        }
    })
}

fn split_index(rest: &str) -> Result<(String, &str), MergeError> {
    let (index, rest) = rest.split_once('.').ok_or_else(|| MergeError::Unresolved {
        key: rest.to_string(),
        detail: "no block index".into(),
    })?;

    if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(MergeError::Unresolved {
            key: rest.to_string(),
            detail: format!("block index {index} is not a number"),
        });
    }

    Ok((index.to_string(), rest))
}

/// A tensor's row count and row width, with rank-1 read as one row.
fn geometry(tensor: &Tensor) -> Result<(u64, u64), MergeError> {
    if tensor.shape.is_empty() {
        return Err(MergeError::Plan {
            tensor: tensor.name.clone(),
            detail: "rank 0".into(),
        });
    }

    if tensor.shape.len() == 1 {
        return Ok((1, tensor.shape[0]));
    }

    Ok((tensor.shape[0], tensor.shape[1..].iter().product()))
}

/// What one derivation produced.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub shards: u64,
    pub tensors: u64,
    pub touched_tensors: u64,
    pub low_rank_tensors: u64,
    pub additive_tensors: u64,
    pub whole_tensors: u64,
    pub bytes: u64,
    pub seconds: f64,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shards={} tensors={} touched={} low_rank={} additive={} whole={} bytes={} seconds={:.1}",
            self.shards,
            self.tensors,
            self.touched_tensors,
            self.low_rank_tensors,
            self.additive_tensors,
            self.whole_tensors,
            self.bytes,
            self.seconds
        )
    }
}

/// Merge `adapter` into the released `model` and write a derived checkpoint at
/// `out`, which must not exist.
pub fn derive(
    model: &Path,
    adapter_path: &Path,
    out: &Path,
    threads: usize,
    dry_run: bool,
) -> Result<Report, MergeError> {
    let started = Instant::now();
    let source = model.join(PARTITION).join(TRANSFORMER);

    if !dry_run && out.exists() {
        return Err(MergeError::Exists(out.to_path_buf()));
    }

    let config = Config::read(&source)?;
    let adapter = Adapter::open(adapter_path)?;
    let store = safetensors::open_dir(&source)?;
    let plan = Plan::build(&adapter, &store, &config)?;

    let mut report = Report {
        shards: store.shards().len() as u64,
        tensors: store.tensor_count(),
        touched_tensors: plan.touches().len() as u64,
        low_rank_tensors: plan.tensors(Payload::LowRank),
        additive_tensors: plan.tensors(Payload::Additive),
        whole_tensors: plan.whole().len() as u64,
        ..Report::default()
    };

    if dry_run {
        report.seconds = started.elapsed().as_secs_f64();
        return Ok(report);
    }

    let target = out.join(PARTITION).join(TRANSFORMER);
    fs::create_dir_all(&target).map_err(|error| MergeError::Io {
        path: target.clone(),
        detail: error.to_string(),
    })?;

    for shard in store.shards() {
        report.bytes += rewrite_shard(shard, &plan, &adapter, &target, threads)?;
    }

    copy_file(&source.join(CONFIG_FILE), &target.join(CONFIG_FILE))?;
    copy_file(&source.join(INDEX_FILE), &target.join(INDEX_FILE))?;
    link_tree(model, out)?;

    if !plan.whole().is_empty() {
        write_whole(&plan, &adapter, &out.join(SET_WEIGHT_DIR))?;
    }

    report.seconds = started.elapsed().as_secs_f64();
    Ok(report)
}

/// Rewrite one released shard with every delta applied, in the file's own
/// tensor order, so a shard with no touched tensor comes out identical.
fn rewrite_shard(
    shard: &Shard,
    plan: &Plan,
    adapter: &Adapter,
    target: &Path,
    threads: usize,
) -> Result<u64, MergeError> {
    let order = shard.file_order();
    let entries: Vec<Entry> = order
        .iter()
        .map(|tensor| Entry {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
        })
        .collect();

    let name = shard
        .path
        .file_name()
        .map(PathBuf::from)
        .ok_or_else(|| MergeError::Io {
            path: shard.path.clone(),
            detail: "no file name".into(),
        })?;

    let writer = ShardWriter::create(&target.join(name), &entries, &shard.metadata)?;

    let mut bytes = 0u64;
    for (index, tensor) in order.iter().enumerate() {
        match plan.touches().get(&tensor.name) {
            Some(deltas) => {
                let segments = plan
                    .segments
                    .get(&tensor.name)
                    .expect("a touched tensor branches like a segmented one");
                apply(
                    shard, tensor, deltas, segments, adapter, &writer, index, threads,
                )?;
            }
            None => copy_tensor(shard, tensor, &writer, index)?,
        }
        bytes += tensor.bytes();
    }

    Ok(bytes)
}

/// Copy one tensor's bytes verbatim, in chunks.
fn copy_tensor(
    shard: &Shard,
    tensor: &Tensor,
    writer: &ShardWriter,
    index: usize,
) -> Result<(), MergeError> {
    let total = tensor.bytes() as usize;
    let mut buffer = vec![0u8; total.min(COPY_CHUNK)];
    let mut done = 0u64;

    while (done as usize) < total {
        let take = buffer.len().min(total - done as usize);
        let slice = &mut buffer[..take];
        shard.read_data(&row_span(tensor, done, take as u64, 1)?, slice)?;
        writer.write_at(index, done, slice)?;
        done += take as u64;
    }

    Ok(())
}

/// A byte span of a tensor, addressed by byte offset, for the copy path.
fn row_span(tensor: &Tensor, offset: u64, len: u64, _rows: u64) -> Result<Tensor, MergeError> {
    let mut span = tensor.clone();
    span.data_begin = tensor.data_begin + offset;
    span.data_end = span.data_begin + len;
    span.file_offset = tensor.file_offset + offset;
    Ok(span)
}

/// One tile's share of the work.
#[derive(Clone, Copy, Debug)]
struct Task {
    segment: usize,
    dest: u64,
    src: u64,
    rows: u64,
}

/// Apply every delta of one tensor, in parallel over its segments' tiles.
fn apply(
    shard: &Shard,
    tensor: &Tensor,
    deltas: &[Delta],
    segments: &[Segment],
    adapter: &Adapter,
    writer: &ShardWriter,
    index: usize,
    threads: usize,
) -> Result<(), MergeError> {
    if tensor.dtype != Dtype::Bf16 && tensor.dtype != Dtype::F32 {
        return Err(MergeError::Plan {
            tensor: tensor.name.clone(),
            detail: format!("a delta onto {:?}", tensor.dtype),
        });
    }

    let (_, cols) = geometry(tensor)?;
    let width = tensor.dtype.size() as usize;

    let mut tasks = Vec::new();
    for (order, segment) in segments.iter().enumerate() {
        let tile = tile_rows(cols).min(segment.rows);
        let mut offset = 0;
        while offset < segment.rows {
            let rows = tile.min(segment.rows - offset);
            tasks.push(Task {
                segment: order,
                dest: segment.dest + offset,
                src: segment.src + offset,
                rows,
            });
            offset += rows;
        }
    }

    if tasks.is_empty() {
        return Ok(());
    }

    // Read each delta's rank-64 factor A once (a few MiB) and its additive
    // payload once; B's rows are read per tile because they are indexed by
    // the tile's own source rows.
    let mut prepared = Vec::with_capacity(deltas.len());
    for delta in deltas {
        prepared.push(match delta {
            Delta::LowRank { a, b, .. } => {
                let factor = adapter.find(a).ok_or_else(|| MergeError::Adapter {
                    detail: format!("no tensor {a}"),
                })?;

                // The rank is the pair's own: A is `[rank, cols]` and B has to
                // agree, whatever the rank happens to be for this module.
                if factor.shape.len() != 2 {
                    return Err(MergeError::Adapter {
                        detail: format!("{a} is not [rank, cols]"),
                    });
                }
                let rank = factor.shape[0];

                let other = adapter.find(b).ok_or_else(|| MergeError::Adapter {
                    detail: format!("no tensor {b}"),
                })?;
                if other.shape.last() != Some(&rank) || other.shape.len() != 2 {
                    return Err(MergeError::Adapter {
                        detail: format!("{b} is not [rows, {rank}]"),
                    });
                }

                Prepared::LowRank {
                    a: decode(&adapter.read(a)?, factor.dtype)?,
                    b_rows: other.shape[0],
                    rank: rank as usize,
                }
            }
            Delta::Add { diff } => {
                let factor = adapter.find(diff).expect("checked when the plan was built");
                if factor.elements() != tensor.elements() {
                    return Err(MergeError::Adapter {
                        detail: format!(
                            "{diff} holds {} values, {} holds {}",
                            factor.elements(),
                            tensor.name,
                            tensor.elements()
                        ),
                    });
                }
                Prepared::Add {
                    diff: adapter.read(diff)?,
                    dtype: factor.dtype,
                }
            }
        });
    }

    let cursor = AtomicUsize::new(0);
    let failures: Mutex<Vec<MergeError>> = Mutex::new(Vec::new());
    let workers = threads.max(1).min(tasks.len());

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let next = cursor.fetch_add(1, Ordering::Relaxed);
                if next >= tasks.len() {
                    break;
                }

                let result = run_task(
                    shard,
                    tensor,
                    deltas,
                    tasks[next],
                    segments,
                    cols,
                    width,
                    &prepared,
                    adapter,
                    writer,
                    index,
                );

                if let Err(error) = result {
                    failures.lock().expect("lock").push(error);
                    break;
                }
            });
        }
    });

    let mut failures = failures.into_inner().expect("lock");
    if let Some(error) = failures.pop() {
        return Err(error);
    }

    Ok(())
}

/// A delta's payload, read once per tensor.
enum Prepared {
    /// Factor A in F32, the rows B states, and the pair's own rank. The rank
    /// is read per module rather than held to one constant: the FastH3 adapter
    /// is uniform at 64, while a few-step adapter such as the Turbo LoRA states
    /// rank 16 on the AdaLN projections and 64 elsewhere.
    LowRank {
        a: Vec<f32>,
        b_rows: u64,
        rank: usize,
    },
    Add {
        diff: Vec<u8>,
        dtype: Dtype,
    },
}

/// One tile: read the destination rows, add the delta, write them back.
#[allow(clippy::too_many_arguments)]
fn run_task(
    shard: &Shard,
    tensor: &Tensor,
    deltas: &[Delta],
    task: Task,
    segments: &[Segment],
    cols: u64,
    width: usize,
    prepared: &[Prepared],
    adapter: &Adapter,
    writer: &ShardWriter,
    index: usize,
) -> Result<(), MergeError> {
    let stride = cols * width as u64;
    let base = read_rows(shard, tensor, task.dest, task.rows)?;
    let mut values = decode(&base, tensor.dtype)?;
    let delta = &deltas[segments[task.segment].delta];

    match (delta, &prepared[segments[task.segment].delta]) {
        (Delta::LowRank { b, .. }, Prepared::LowRank { a, b_rows, rank }) => {
            let factor = adapter.find(b).ok_or_else(|| MergeError::Adapter {
                detail: format!("no tensor {b}"),
            })?;
            let (dest_rows, _) = geometry(tensor)?;
            if factor.shape.first() != Some(&dest_rows) {
                return Err(MergeError::Adapter {
                    detail: format!(
                        "{b} holds {b_rows} rows, {} holds {dest_rows}",
                        tensor.name
                    ),
                });
            }

            let bytes = read_rows(adapter.shard(), factor, task.src, task.rows)?;
            let b_values = decode(&bytes, factor.dtype)?;
            add_low_rank(
                &mut values,
                a,
                &b_values,
                *rank,
                task.rows,
                cols,
                tensor.dtype,
            );
        }
        (Delta::Add { .. }, Prepared::Add { diff, dtype }) => {
            // The delta's own dtype indexes it: an F32 boundary projection
            // takes a BF16 delta of the same element count.
            let diff_stride = cols * dtype.size();
            let start = (task.dest * diff_stride) as usize;
            let span = &diff[start..start + (task.rows * diff_stride) as usize];
            let adds = decode(span, *dtype)?;

            if adds.len() != values.len() {
                return Err(MergeError::Adapter {
                    detail: format!(
                        "{}: {} values against {} of delta",
                        tensor.name,
                        values.len(),
                        adds.len()
                    ),
                });
            }

            for (value, add) in values.iter_mut().zip(adds.iter()) {
                *value += *add;
            }
        }
        _ => {
            return Err(MergeError::Plan {
                tensor: tensor.name.clone(),
                detail: "a delta without its prepared payload".into(),
            })
        }
    }

    let bytes = encode(&values, tensor.dtype);
    writer.write_at(index, task.dest * stride, &bytes)?;
    Ok(())
}

/// `values += round(B @ A)`: the product is rounded to the destination's own
/// precision before the add, because the reference's `data += lora_delta` adds
/// a delta `torch.matmul` already rounded to the weight's dtype.
fn add_low_rank(
    values: &mut [f32],
    a: &[f32],
    b: &[f32],
    rank: usize,
    rows: u64,
    cols: u64,
    dtype: Dtype,
) {
    let cols = cols as usize;
    let mut delta = vec![0.0f32; rows as usize * cols];

    // A is [rank, cols] and B is [rows, rank], k ascending, so a row's sum is
    // the same whichever thread computes it.
    for row in 0..rows as usize {
        let out = &mut delta[row * cols..(row + 1) * cols];
        let b_row = &b[row * rank..(row + 1) * rank];

        for k in 0..rank {
            let weight = b_row[k];
            if weight == 0.0 {
                continue;
            }
            let a_row = &a[k * cols..(k + 1) * cols];
            for (value, factor) in out.iter_mut().zip(a_row.iter()) {
                *value += weight * factor;
            }
        }
    }

    for (value, add) in values.iter_mut().zip(delta.iter()) {
        *value += rounded(*add, dtype);
    }
}

/// A value as the destination's dtype holds it.
fn rounded(value: f32, dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F32 => value,
        _ => bf16_to_f32(f32_to_bf16(value)),
    }
}

/// Read `rows` rows of a tensor starting at row `first`.
fn read_rows(shard: &Shard, tensor: &Tensor, first: u64, rows: u64) -> Result<Vec<u8>, MergeError> {
    let (_, cols) = geometry(tensor)?;
    let stride = cols * tensor.dtype.size();

    let mut span = tensor.clone();
    span.data_begin = tensor.data_begin + first * stride;
    span.data_end = span.data_begin + rows * stride;
    span.file_offset = tensor.file_offset + first * stride;

    let mut buffer = vec![0u8; (rows * stride) as usize];
    shard.read_data(&span, &mut buffer)?;
    Ok(buffer)
}

/// F32 from a tensor's bytes.
fn decode(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>, MergeError> {
    match dtype {
        Dtype::Bf16 => Ok(bytes
            .chunks_exact(2)
            .map(|pair| bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect()),
        Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()),
        other => Err(MergeError::Adapter {
            detail: format!("cannot read {other:?} as a weight"),
        }),
    }
}

/// Bytes in a tensor's dtype, rounding the way `f32_to_bf16` states.
fn encode(values: &[f32], dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out
        }
        _ => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for value in values {
                out.extend_from_slice(&f32_to_bf16(*value).to_le_bytes());
            }
            out
        }
    }
}

fn tile_rows(cols: u64) -> u64 {
    (TASK_ELEMENTS / cols.max(1)).clamp(1, 4096)
}

/// The whole-parameter payloads, written beside the checkpoint: the released
/// shard set must stay exactly the schema the engine validates, and
/// `layout.rs:254-257` refuses an unlisted tensor.
fn write_whole(plan: &Plan, adapter: &Adapter, dir: &Path) -> Result<(), MergeError> {
    fs::create_dir_all(dir).map_err(|error| MergeError::Io {
        path: dir.to_path_buf(),
        detail: error.to_string(),
    })?;

    let mut entries = Vec::new();
    for module in plan.whole() {
        let key = format!("{module}.set_weight");
        let tensor = adapter.find(&key).ok_or_else(|| MergeError::Adapter {
            detail: format!("no tensor {key}"),
        })?;
        entries.push(Entry {
            name: format!("{module}.weight"),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
        });
    }

    let writer = ShardWriter::create(
        &dir.join("attn_gate_compress.safetensors"),
        &entries,
        &adapter.shard().metadata,
    )?;

    for (index, module) in plan.whole().iter().enumerate() {
        writer.write(index, &adapter.read(&format!("{module}.set_weight"))?)?;
    }

    Ok(())
}

fn copy_file(from: &Path, to: &Path) -> Result<(), MergeError> {
    fs::copy(from, to).map_err(|error| MergeError::Io {
        path: from.to_path_buf(),
        detail: error.to_string(),
    })?;
    Ok(())
}

/// Every file outside the derived partition, hard-linked so the derived tree
/// shares the released bytes instead of doubling them.
fn link_tree(model: &Path, out: &Path) -> Result<(), MergeError> {
    // The derived tree may be written inside the released one; walking into it
    // would link the derivation into itself, one level deeper every pass.
    let out = fs::canonicalize(out).unwrap_or_else(|_| out.to_path_buf());
    link_dir(model, model, &out, &Path::new(PARTITION).join(TRANSFORMER))
}

fn link_dir(root: &Path, dir: &Path, out: &Path, skip: &Path) -> Result<(), MergeError> {
    let entries = fs::read_dir(dir).map_err(|error| MergeError::Io {
        path: dir.to_path_buf(),
        detail: error.to_string(),
    })?;

    for entry in entries {
        let entry = entry.map_err(|error| MergeError::Io {
            path: dir.to_path_buf(),
            detail: error.to_string(),
        })?;
        let path = entry.path();
        let relative = path.strip_prefix(root).expect("inside the root");

        if relative == skip {
            continue;
        }

        if fs::canonicalize(&path)
            .map(|here| here == out)
            .unwrap_or(false)
        {
            continue;
        }

        let target = out.join(relative);
        let kind = entry.file_type().map_err(|error| MergeError::Io {
            path: path.clone(),
            detail: error.to_string(),
        })?;

        if kind.is_dir() {
            fs::create_dir_all(&target).map_err(|error| MergeError::Io {
                path: target.clone(),
                detail: error.to_string(),
            })?;
            link_dir(root, &path, out, skip)?;
            continue;
        }

        if kind.is_file() {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|error| MergeError::Io {
                    path: parent.to_path_buf(),
                    detail: error.to_string(),
                })?;
            }

            // A hard link where the filesystem allows one, a copy otherwise.
            if fs::hard_link(&path, &target).is_err() {
                fs::copy(&path, &target).map_err(|error| MergeError::Io {
                    path: target.clone(),
                    detail: error.to_string(),
                })?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MiniMax-H3/FL2VA/transformer/config.json`, verbatim.
    const CONFIG: &str = r#"{
  "_class_name": "MiniMaxH3DiTModel",
  "_diffusers_version": "0.32.2",
  "hidden_size": 5376,
  "num_layers": 50,
  "token_refiner_num_layers": 2,
  "num_attention_heads": 56,
  "attention_head_dim": 128,
  "ffn_hidden_size": 14336,
  "latents_dim": 24,
  "audio_latents_dim": 32,
  "patch_size": [1, 2, 2],
  "text_dim": 5120,
  "timestep_input_dim": 256,
  "time_embed_hidden_size": 5376,
  "time_embed_dim": 2688,
  "adaln_out_features": 96768,
  "final_adaln_out_features": 10752,
  "rope_inv_freq_len": 16,
  "norm_eps": 1e-05,
  "qk_norm_eps": 1e-05,
  "final_norm_eps": 1e-05
}"#;

    fn config() -> Config {
        Config::parse(CONFIG).expect("the released config")
    }

    #[test]
    fn the_payload_classes_are_tested_longest_first() {
        assert_eq!(classify("a.diff_b"), Some(Payload::Additive));
        assert_eq!(classify("a.diff"), Some(Payload::Additive));
        assert_eq!(classify("a.lora_A.weight"), Some(Payload::LowRank));
        assert_eq!(classify("a.set_weight"), Some(Payload::Whole));
        assert_eq!(classify("a.weight"), None);
    }

    #[test]
    fn a_module_resolves_to_its_served_tensor() {
        let cases = [
            (
                "transformer_blocks.7.attn.to_q",
                false,
                "blocks.7.attn.qkv_proj.weight",
                Rows::Qkv { part: 0 },
            ),
            (
                "transformer_blocks.7.attn.to_k",
                false,
                "blocks.7.attn.qkv_proj.weight",
                Rows::Qkv { part: 1 },
            ),
            (
                "transformer_blocks.7.attn.to_v",
                false,
                "blocks.7.attn.qkv_proj.weight",
                Rows::Qkv { part: 2 },
            ),
            (
                "transformer_blocks.7.attn.to_out.0",
                false,
                "blocks.7.attn.out_proj.weight",
                Rows::Identity,
            ),
            (
                "transformer_blocks.7.ff.net.0.proj",
                false,
                "blocks.7.mlp.fc1.weight",
                Rows::Swap,
            ),
            (
                "transformer_blocks.7.ff.net.2",
                false,
                "blocks.7.mlp.fc2.weight",
                Rows::Identity,
            ),
            (
                "transformer_blocks.7.adaln_proj.linear",
                false,
                "blocks.7.adaln_proj.linear.weight",
                Rows::Identity,
            ),
            (
                "transformer_blocks.7.adaln_proj.linear",
                true,
                "blocks.7.adaln_proj.linear.bias",
                Rows::Identity,
            ),
            (
                "transformer_blocks.7.norm2",
                false,
                "blocks.7.norm2.weight",
                Rows::Identity,
            ),
            (
                "token_refiner.refiner_blocks.1.attn.to_q",
                false,
                "token_refiner.blocks.1.attn.qkv_proj.weight",
                Rows::Qkv { part: 0 },
            ),
            ("proj_in", false, "video_patch_proj.weight", Rows::Identity),
            (
                "time_embedder.linear_2",
                false,
                "time_embedder.proj_out.weight",
                Rows::Identity,
            ),
            (
                "norm_out.norm",
                false,
                "final_layer.norm.weight",
                Rows::Identity,
            ),
            (
                "audio_proj_in",
                true,
                "audio_patch_proj.bias",
                Rows::Identity,
            ),
            (
                "audio_proj_out",
                true,
                "final_layer.audio_out.bias",
                Rows::Identity,
            ),
            (
                "proj_out",
                true,
                "final_layer.video_out.bias",
                Rows::Identity,
            ),
            (
                "norm_out.linear",
                true,
                "final_layer.adaln_proj.linear.bias",
                Rows::Identity,
            ),
            (
                "context_embedder",
                true,
                "condition_proj.bias",
                Rows::Identity,
            ),
            (
                "time_embedder.linear_1",
                true,
                "time_embedder.proj_in.bias",
                Rows::Identity,
            ),
            ("proj_in", true, "video_patch_proj.bias", Rows::Identity),
        ];

        for (module, bias, name, rows) in cases {
            let (got, order) = destination(module, bias).expect(module);
            assert_eq!((got.as_str(), order), (name, rows), "{module}");
        }
    }

    #[test]
    fn an_unknown_module_is_refused() {
        assert!(destination("blocks.0.attn.to_q", false).is_err());
        assert!(destination("transformer_blocks.0.attn.to_gate_compress", false).is_err());
        assert!(destination("transformer_blocks.x.attn.to_q", false).is_err());
        assert!(destination("transformer_blocks.0.attn.to_q", true).is_err());
    }

    #[test]
    fn the_released_config_is_the_geometry_the_qkv_interleave_needs() {
        let config = config();
        assert_eq!(config.heads, 56);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.heads * config.head_dim * 3, 21504);
        assert_eq!(config.ffn, 14336);
    }

    #[test]
    fn the_add_of_a_rank_product_is_the_reference_sum() {
        // B @ A with rank 2 over 2 rows and 3 columns, and the F32 add.
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut values = vec![10.0f32, 10.0, 10.0, 10.0, 10.0, 10.0];
        add_low_rank(&mut values, &a, &b, 2, 2, 3, Dtype::F32);
        assert_eq!(values, vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    fn the_encoding_rounds_the_way_the_engine_writes_bf16() {
        let bytes = encode(&[1.0f32, -2.0], Dtype::Bf16);
        assert_eq!(bytes, vec![0x80, 0x3f, 0x00, 0xc0]);
        assert_eq!(decode(&bytes, Dtype::Bf16).expect("bf16"), vec![1.0, -2.0]);

        let bytes = encode(&[0.5f32], Dtype::F32);
        assert_eq!(decode(&bytes, Dtype::F32).expect("f32"), vec![0.5]);
    }

    #[test]
    fn rank_one_reads_as_one_row() {
        let tensor = Tensor {
            name: "norm1.weight".into(),
            dtype: Dtype::Bf16,
            shape: vec![5376],
            data_begin: 0,
            data_end: 10752,
            file_offset: 0,
        };
        assert_eq!(geometry(&tensor).expect("geometry"), (1, 5376));
    }

    /// Mini geometry: two heads of four, so the qkv interleave is visible, and
    /// an FFN of four, so the swapped halves are too.
    const MINI_CONFIG: &str = r#"{
  "_class_name": "MiniMaxH3DiTModel",
  "hidden_size": 8,
  "num_layers": 1,
  "token_refiner_num_layers": 0,
  "num_attention_heads": 2,
  "attention_head_dim": 4,
  "ffn_hidden_size": 4,
  "latents_dim": 1,
  "audio_latents_dim": 1,
  "patch_size": [1, 2, 2],
  "text_dim": 4,
  "timestep_input_dim": 256,
  "time_embed_hidden_size": 8,
  "time_embed_dim": 4,
  "adaln_out_features": 12,
  "final_adaln_out_features": 8,
  "rope_inv_freq_len": 4,
  "norm_eps": 1e-05,
  "qk_norm_eps": 1e-05,
  "final_norm_eps": 1e-05
}"#;

    /// The served tensors the mini adapter touches, and their released values.
    fn mini_checkpoint(dir: &Path) -> Vec<(String, Dtype, Vec<u64>, Vec<f32>)> {
        let mut out: Vec<(String, Dtype, Vec<u64>, Vec<f32>)> = Vec::new();
        let mut push = |name: &str, dtype: Dtype, shape: Vec<u64>, seed: f32| {
            let elements: u64 = shape.iter().product();
            let values = (0..elements)
                .map(|index| seed + index as f32 * 0.5)
                .collect();
            out.push((name.to_string(), dtype, shape, values));
        };

        push(
            "blocks.0.attn.qkv_proj.weight",
            Dtype::Bf16,
            vec![24, 8],
            0.0,
        );
        push(
            "blocks.0.attn.out_proj.weight",
            Dtype::Bf16,
            vec![8, 8],
            1.0,
        );
        push("blocks.0.mlp.fc1.weight", Dtype::Bf16, vec![8, 8], 2.0);
        push("blocks.0.mlp.fc2.weight", Dtype::Bf16, vec![8, 4], 3.0);
        push("blocks.0.norm1.weight", Dtype::Bf16, vec![8], 4.0);
        push(
            "blocks.0.adaln_proj.linear.weight",
            Dtype::Bf16,
            vec![12, 4],
            5.0,
        );
        push(
            "blocks.0.adaln_proj.linear.bias",
            Dtype::Bf16,
            vec![12],
            6.0,
        );
        push("condition_proj.weight", Dtype::Bf16, vec![8, 4], 7.0);
        push("video_patch_proj.weight", Dtype::F32, vec![8, 4], 8.0);
        push("video_patch_proj.bias", Dtype::F32, vec![8], 0.0);
        push("final_layer.video_out.weight", Dtype::F32, vec![4, 8], 9.0);

        let store = dir.join(PARTITION).join(TRANSFORMER);
        fs::create_dir_all(&store).expect("store dir");
        fs::write(store.join(CONFIG_FILE), MINI_CONFIG).expect("config");

        let entries: Vec<Entry> = out
            .iter()
            .map(|(name, dtype, shape, _)| Entry {
                name: name.clone(),
                dtype: *dtype,
                shape: shape.clone(),
            })
            .collect();
        let writer = ShardWriter::create(
            &store.join("model-00001-of-00001.safetensors"),
            &entries,
            &BTreeMap::new(),
        )
        .expect("shard");

        let mut map = String::from("{\"metadata\":{\"total_size\":0},\"weight_map\":{");
        for (index, (name, dtype, shape, values)) in out.iter().enumerate() {
            writer
                .write(index, &encode(values, *dtype))
                .expect("tensor bytes");
            if index > 0 {
                map.push(',');
            }
            map.push_str(&format!(
                "{}:\"model-00001-of-00001.safetensors\"",
                serde_json::to_string(name).expect("name")
            ));
            let bytes: u64 = shape.iter().product::<u64>() * dtype.size();
            let _ = bytes;
        }
        map.push_str("}}");

        // The index's own byte total, as the released file states it.
        let total: u64 = out
            .iter()
            .map(|(_, dtype, shape, _)| shape.iter().product::<u64>() * dtype.size())
            .sum();
        let map = map.replace("\"total_size\":0", &format!("\"total_size\":{total}"));
        fs::write(store.join(INDEX_FILE), map).expect("index");

        out
    }

    /// The value a BF16 tensor holds for an F32 sum: the reference adds into a
    /// bf16 tensor, so the sum is rounded once, at the destination's precision.
    fn stored(value: f32) -> f32 {
        bf16_to_f32(f32_to_bf16(value))
    }

    #[test]
    fn a_mini_checkpoint_merges_the_three_payload_classes_in_place() {
        let dir = std::env::temp_dir().join(format!("h3-lora-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let base = mini_checkpoint(&dir);

        // The adapter: one low-rank pair per qkv projection, one for the fused
        // FFN input, a `.diff` on a norm, a `.diff_b` on an AdaLN bias, and one
        // whole parameter the checkpoint does not carry.
        let mut adapter: Vec<(String, Dtype, Vec<u64>, Vec<f32>)> = Vec::new();
        let pair = |adapter: &mut Vec<(String, Dtype, Vec<u64>, Vec<f32>)>,
                    module: &str,
                    rows: u64,
                    seed: f32| {
            adapter.push((
                format!("{module}.lora_A.weight"),
                Dtype::Bf16,
                vec![ADAPTER_RANK, 8],
                (0..ADAPTER_RANK * 8)
                    .map(|i| seed + i as f32 * 0.01)
                    .collect(),
            ));
            adapter.push((
                format!("{module}.lora_B.weight"),
                Dtype::Bf16,
                vec![rows, ADAPTER_RANK],
                (0..rows * ADAPTER_RANK)
                    .map(|i| seed + i as f32 * 0.02)
                    .collect(),
            ));
        };
        pair(&mut adapter, "transformer_blocks.0.attn.to_q", 8, 1.0);
        pair(&mut adapter, "transformer_blocks.0.attn.to_k", 8, 2.0);
        pair(&mut adapter, "transformer_blocks.0.attn.to_v", 8, 3.0);
        pair(&mut adapter, "transformer_blocks.0.ff.net.0.proj", 8, 4.0);
        adapter.push((
            "transformer_blocks.0.norm1.diff".into(),
            Dtype::Bf16,
            vec![8],
            vec![0.25; 8],
        ));
        // The boundary projections are F32 in the released checkpoint and the
        // adapter still states their deltas in BF16.
        adapter.push((
            "proj_in.diff".into(),
            Dtype::Bf16,
            vec![8, 4],
            vec![0.125; 32],
        ));
        adapter.push(("proj_in.diff_b".into(), Dtype::Bf16, vec![8], vec![0.75; 8]));
        adapter.push((
            "transformer_blocks.0.adaln_proj.linear.diff_b".into(),
            Dtype::Bf16,
            vec![12],
            vec![0.5; 12],
        ));
        adapter.push((
            "transformer_blocks.0.attn.to_gate_compress.set_weight".into(),
            Dtype::Bf16,
            vec![8, 8],
            (0..64).map(|i| i as f32).collect(),
        ));

        let path = dir.join("adapter_model.safetensors");
        let entries: Vec<Entry> = adapter
            .iter()
            .map(|(name, dtype, shape, _)| Entry {
                name: name.clone(),
                dtype: *dtype,
                shape: shape.clone(),
            })
            .collect();
        let metadata: BTreeMap<String, String> = [
            ("format", ADAPTER_FORMAT),
            ("rank", "64"),
            ("low_rank_tensors", "8"),
            ("diff_tensors", "4"),
            ("set_weight_tensors", "1"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
        let writer = ShardWriter::create(&path, &entries, &metadata).expect("adapter");
        for (index, (_, dtype, _, values)) in adapter.iter().enumerate() {
            writer.write(index, &encode(values, *dtype)).expect("bytes");
        }

        let out = dir.join("derived");
        let report = derive(&dir, &path, &out, 2, false).expect("merge");

        // One tensor for the qkv fusion, one for the fused FFN input, and one
        // each for the norm and the AdaLN bias.
        assert_eq!(report.touched_tensors, 6);
        assert_eq!(report.low_rank_tensors, 8);
        assert_eq!(report.additive_tensors, 4);
        assert_eq!(report.whole_tensors, 1);

        let merged = safetensors::open_dir(&out.join(PARTITION).join(TRANSFORMER)).expect("store");
        let read = |name: &str| -> Vec<f32> {
            let (shard, tensor) = merged.find(name).expect(name);
            let mut bytes = vec![0u8; tensor.bytes() as usize];
            shard.read_data(tensor, &mut bytes).expect("read");
            decode(&bytes, tensor.dtype).expect("decode")
        };
        let released: std::collections::HashMap<&str, &(String, Dtype, Vec<u64>, Vec<f32>)> =
            base.iter().map(|entry| (entry.0.as_str(), entry)).collect();

        // A tensor with no delta is the released tensor, value for value.
        for name in [
            "blocks.0.attn.out_proj.weight",
            "blocks.0.mlp.fc2.weight",
            "condition_proj.weight",
            "final_layer.video_out.weight",
        ] {
            let released = released[name];
            assert_eq!(read(name), released.3, "{name}");

            // and byte for byte
            let (shard, tensor) = merged.find(name).expect(name);
            let mut got = vec![0u8; tensor.bytes() as usize];
            shard.read_data(tensor, &mut got).expect("read");
            assert_eq!(got, encode(&released.3, released.1), "{name} bytes");
        }

        // The dense payloads, added in F32 and stored in the tensor's dtype.
        let norm = read("blocks.0.norm1.weight");
        for (index, value) in norm.iter().enumerate() {
            let expect = released["blocks.0.norm1.weight"].3[index] + 0.25;
            assert_eq!(value, &stored(expect), "norm1 {index}");
        }
        // An F32 tensor takes a BF16 delta: the add stays in F32 and the
        // destination keeps its own dtype.
        let projected = read("video_patch_proj.weight");
        for (index, value) in projected.iter().enumerate() {
            let expect = released["video_patch_proj.weight"].3[index] + 0.125;
            assert_eq!(value, &expect, "proj_in.weight {index}");
        }
        let projected = read("video_patch_proj.bias");
        for (index, value) in projected.iter().enumerate() {
            let expect = released["video_patch_proj.bias"].3[index] + 0.75;
            assert_eq!(value, &expect, "proj_in.bias {index}");
        }

        let bias = read("blocks.0.adaln_proj.linear.bias");
        for (index, value) in bias.iter().enumerate() {
            let expect = released["blocks.0.adaln_proj.linear.bias"].3[index] + 0.5;
            assert_eq!(value, &stored(expect), "adaln bias {index}");
        }

        // The qkv rows: each head's slot is its own projection, and the delta
        // of `to_q` lands in slot 0 of every head.
        let qkv = read("blocks.0.attn.qkv_proj.weight");
        let base_qkv = &released["blocks.0.attn.qkv_proj.weight"].3;
        let delta = |module: &str, rows: u64| -> Vec<f32> {
            let a: Vec<f32> = adapter
                .iter()
                .find(|entry| entry.0 == format!("{module}.lora_A.weight"))
                .map(|entry| entry.3.clone())
                .expect("A");
            let b: Vec<f32> = adapter
                .iter()
                .find(|entry| entry.0 == format!("{module}.lora_B.weight"))
                .map(|entry| entry.3.clone())
                .expect("B");
            let a: Vec<f32> = a
                .iter()
                .map(|value| bf16_to_f32(f32_to_bf16(*value)))
                .collect();
            let b: Vec<f32> = b
                .iter()
                .map(|value| bf16_to_f32(f32_to_bf16(*value)))
                .collect();

            let mut out = vec![0.0f32; (rows * 8) as usize];
            for row in 0..rows as usize {
                for k in 0..ADAPTER_RANK as usize {
                    for column in 0..8 {
                        out[row * 8 + column] += b[row * 64 + k] * a[k * 8 + column];
                    }
                }
            }
            out.iter()
                .map(|value| bf16_to_f32(f32_to_bf16(*value)))
                .collect()
        };

        let q = delta("transformer_blocks.0.attn.to_q", 8);
        let k = delta("transformer_blocks.0.attn.to_k", 8);
        let v = delta("transformer_blocks.0.attn.to_v", 8);
        for head in 0..2usize {
            for slot in 0..3usize {
                for row in 0..4usize {
                    for column in 0..8usize {
                        let dest = (head * 4 * 3 + slot * 4 + row) * 8 + column;
                        let src = (head * 4 + row) * 8 + column;
                        let add = match slot {
                            0 => q[src],
                            1 => k[src],
                            _ => v[src],
                        };
                        assert_eq!(qkv[dest], stored(base_qkv[dest] + add), "qkv row {dest}");
                    }
                }
            }
        }

        // The FFN input: the source row's delta lands in the other half.
        let fc1 = read("blocks.0.mlp.fc1.weight");
        let base_fc1 = &released["blocks.0.mlp.fc1.weight"].3;
        let ffn = delta("transformer_blocks.0.ff.net.0.proj", 8);
        for row in 0..8usize {
            for column in 0..8usize {
                let source = if row < 4 { row + 4 } else { row - 4 };
                let expect = base_fc1[row * 8 + column] + ffn[source * 8 + column];
                assert_eq!(fc1[row * 8 + column], stored(expect), "fc1 row {row}");
            }
        }

        // The whole parameter, written beside the checkpoint.
        let set = safetensors::read_shard(
            &out.join(SET_WEIGHT_DIR)
                .join("attn_gate_compress.safetensors"),
        )
        .expect("set_weight");
        assert_eq!(
            set.find("transformer_blocks.0.attn.to_gate_compress.weight")
                .is_some(),
            true
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_adapter_key_no_rule_places_stops_the_merge() {
        let dir = std::env::temp_dir().join(format!("h3-lora-bad-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        mini_checkpoint(&dir);

        let path = dir.join("adapter_model.safetensors");
        let entries = vec![
            Entry {
                name: "transformer_blocks.0.attn.to_q_instead.lora_A.weight".into(),
                dtype: Dtype::Bf16,
                shape: vec![ADAPTER_RANK, 8],
            },
            Entry {
                name: "transformer_blocks.0.attn.to_q_instead.lora_B.weight".into(),
                dtype: Dtype::Bf16,
                shape: vec![8, ADAPTER_RANK],
            },
        ];
        let metadata: BTreeMap<String, String> = [
            ("format", ADAPTER_FORMAT),
            ("rank", "64"),
            ("low_rank_tensors", "2"),
            ("diff_tensors", "0"),
            ("set_weight_tensors", "0"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
        let writer = ShardWriter::create(&path, &entries, &metadata).expect("adapter");
        writer
            .write(0, &encode(&[0.0; ADAPTER_RANK as usize * 8], Dtype::Bf16))
            .expect("bytes");
        writer
            .write(1, &encode(&[0.0; 8 * ADAPTER_RANK as usize], Dtype::Bf16))
            .expect("bytes");

        let error = derive(&dir, &path, &dir.join("derived"), 1, false).expect_err("refused");
        assert!(matches!(error, MergeError::Unresolved { .. }), "{error}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_adapter_whose_counts_do_not_match_is_refused() {
        let dir = std::env::temp_dir().join(format!("h3-lora-count-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("dir");

        let path = dir.join("adapter_model.safetensors");
        let entries = vec![Entry {
            name: "transformer_blocks.0.norm1.diff".into(),
            dtype: Dtype::Bf16,
            shape: vec![8],
        }];
        let metadata: BTreeMap<String, String> = [
            ("format", ADAPTER_FORMAT),
            ("rank", "64"),
            ("low_rank_tensors", "0"),
            ("diff_tensors", "2"),
            ("set_weight_tensors", "0"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
        let writer = ShardWriter::create(&path, &entries, &metadata).expect("adapter");
        writer
            .write(0, &encode(&[0.0; 8], Dtype::Bf16))
            .expect("bytes");

        let error = Adapter::open(&path).expect_err("refused");
        assert!(matches!(error, MergeError::Adapter { .. }), "{error}");

        let _ = fs::remove_dir_all(&dir);
    }
}
