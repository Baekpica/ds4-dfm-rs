//! MiniMax-H3 FL2VA DiT tensor layout.
//!
//! The reference engine names its weights by string and checks each one's
//! dtype and shape at load time. This module turns that implicit schema into
//! data: from `FL2VA/transformer/config.json` it derives every expected tensor
//! name with its role, dtype and shape, then checks a shard set against it.
//!
//! Source of truth is the reference's own loaders:
//!
//! - `h3_dit.c:501-525` `load_block`: `norm1`, `norm2`,
//!   `attn.qkv_proj.weight` at `INNER*3 x HIDDEN`, `attn.q_norm.weight` and
//!   `attn.k_norm.weight` at `HEAD_DIM`, `attn.out_proj.weight` at
//!   `HIDDEN x INNER`, `mlp.fc1.weight` at `FFN*2 x HIDDEN`, `mlp.fc2.weight`
//!   at `HIDDEN x FFN`.
//! - `h3_dit.c:819-823` the same eight tensors under `token_refiner.blocks.N.`
//!   with no AdaLN, then `token_refiner.final_norm.weight`.
//! - `h3_dit.c:808-811` `condition_proj`; `h3_dit.c:884` `rope.inv_freq`;
//!   `h3_dit.c:1295-1311` the patch projections and `final_layer`.
//! - `h3_dit_schedule.c:170-178` `time_embedder.proj_in` / `proj_out` in F32;
//!   `h3_dit_schedule.c:267-277` `blocks.N.adaln_proj.linear` at
//!   `BLOCK_OUTPUT x H3_DIT_TIME_DIM`; `h3_dit_schedule.c:304-307`
//!   `final_layer.adaln_proj.linear` at `FINAL_OUTPUT x H3_DIT_TIME_DIM`.
//! - Geometry: `h3_dit.c:16-29` and `h3_dit_schedule.c:11-16`.
//!
//! Deviation, named: the reference reads geometry from compile-time constants
//! and the checkpoint's `config.json` is never consulted. This module takes the
//! geometry from `config.json` and the checkpoint is checked against it, so a
//! config that disagrees with the reference is reported instead of ignored.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::minimax_h3::safetensors::{Dtype, Store};

/// `H3_DIT_BLOCKS` (`h3_dit_schedule.h:11`).
pub const REFERENCE_BLOCKS: u32 = 50;

/// `TIME_INPUT` (`h3_dit_schedule.c:12`), the sinusoid width.
pub const REFERENCE_TIME_INPUT: u64 = 256;

/// The `_class_name` the layout was derived from.
pub const DIT_CLASS: &str = "MiniMaxH3DiTModel";

const CONFIG_FILE: &str = "config.json";
const BLOCK_PREFIX: &str = "blocks.";
const REFINER_PREFIX: &str = "token_refiner.blocks.";
const REFINER_FINAL: &str = "token_refiner.final_norm.weight";
const PATCH_RANK: usize = 3;

/// The geometry the checkpoint's `FL2VA/transformer/config.json` declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub hidden: u64,
    pub layers: u32,
    pub refiner_layers: u32,
    pub heads: u64,
    pub head_dim: u64,
    pub ffn: u64,
    pub video_latents: u64,
    pub audio_latents: u64,
    pub patch: [u64; PATCH_RANK],
    pub text_dim: u64,
    pub time_input: u64,
    pub time_hidden: u64,
    pub time_dim: u64,
    pub adaln_out: u64,
    pub final_adaln_out: u64,
    pub rope_freqs: u64,
}

impl Config {
    /// Read `<dir>/config.json`.
    pub fn read(dir: &Path) -> Result<Config, LayoutError> {
        let path = dir.join(CONFIG_FILE);
        let text = std::fs::read_to_string(&path).map_err(|err| LayoutError::Io {
            path: path.clone(),
            detail: err.to_string(),
        })?;

        Config::parse(&text)
    }

    /// Parse the JSON text.
    pub fn parse(text: &str) -> Result<Config, LayoutError> {
        let root: Value = serde_json::from_str(text).map_err(|err| LayoutError::Config {
            detail: err.to_string(),
        })?;

        let patch_values = array(&root, "patch_size")?;
        if patch_values.len() != PATCH_RANK {
            return Err(LayoutError::Config {
                detail: format!("patch_size has {} values, expected 3", patch_values.len()),
            });
        }

        let class = string(&root, "_class_name")?;
        if class != DIT_CLASS {
            return Err(LayoutError::Config {
                detail: format!("_class_name is {class}, expected {DIT_CLASS}"),
            });
        }

        // The reference's time sinusoids are `TIME_INPUT` wide
        // (`h3_dit_schedule.c:150-155`); a different width would silently
        // reshape proj_in's columns.
        let time_input = uint(&root, "timestep_input_dim")?;
        if time_input != REFERENCE_TIME_INPUT {
            return Err(LayoutError::Config {
                detail: format!(
                    "timestep_input_dim is {time_input}, the reference expects {REFERENCE_TIME_INPUT}"
                ),
            });
        }

        Ok(Config {
            hidden: uint(&root, "hidden_size")?,
            layers: count(&root, "num_layers")?,
            refiner_layers: count(&root, "token_refiner_num_layers")?,
            heads: uint(&root, "num_attention_heads")?,
            head_dim: uint(&root, "attention_head_dim")?,
            ffn: uint(&root, "ffn_hidden_size")?,
            video_latents: uint(&root, "latents_dim")?,
            audio_latents: uint(&root, "audio_latents_dim")?,
            patch: [patch_values[0], patch_values[1], patch_values[2]],
            text_dim: uint(&root, "text_dim")?,
            time_input,
            time_hidden: uint(&root, "time_embed_hidden_size")?,
            time_dim: uint(&root, "time_embed_dim")?,
            adaln_out: uint(&root, "adaln_out_features")?,
            final_adaln_out: uint(&root, "final_adaln_out_features")?,
            rope_freqs: uint(&root, "rope_inv_freq_len")?,
        })
    }

    /// `INNER` (`h3_dit.c:20`).
    pub fn inner(&self) -> u64 {
        self.heads * self.head_dim
    }

    /// The patchified video row width (`VIDEO_PATCH`, `h3_dit.c:23`).
    pub fn video_patch(&self) -> u64 {
        self.video_latents * self.patch.iter().product::<u64>()
    }
}

/// What a tensor is for. One role per distinct reference load site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    ConditionWeight,
    ConditionBias,
    RopeFreq,
    TimeInWeight,
    TimeInBias,
    TimeOutWeight,
    TimeOutBias,
    VideoPatchWeight,
    VideoPatchBias,
    AudioPatchWeight,
    AudioPatchBias,
    FinalNorm,
    FinalAdalnWeight,
    FinalAdalnBias,
    FinalVideoWeight,
    FinalVideoBias,
    FinalAudioWeight,
    FinalAudioBias,
    BlockNorm1(u32),
    BlockNorm2(u32),
    BlockQkv(u32),
    BlockQNorm(u32),
    BlockKNorm(u32),
    BlockOut(u32),
    BlockFc1(u32),
    BlockFc2(u32),
    BlockAdalnWeight(u32),
    BlockAdalnBias(u32),
    RefinerNorm1(u32),
    RefinerNorm2(u32),
    RefinerQkv(u32),
    RefinerQNorm(u32),
    RefinerKNorm(u32),
    RefinerOut(u32),
    RefinerFc1(u32),
    RefinerFc2(u32),
    RefinerFinalNorm,
}

/// A tensor as the layout needs it: no bytes, no offsets.
pub struct TensorRef<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [u64],
}

/// A checked tensor.
#[derive(Debug)]
pub struct Entry {
    pub name: String,
    pub role: Role,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

/// The expected tensor at a name.
struct Spec {
    name: String,
    role: Role,
    dtype: Dtype,
    shape: Vec<u64>,
}

/// A checkpoint that matches the config, in name order.
#[derive(Debug)]
pub struct Layout {
    config: Config,
    entries: Vec<Entry>,
}

impl Layout {
    pub fn build(config: &Config, tensors: &[TensorRef<'_>]) -> Result<Layout, LayoutError> {
        let specs = specs(config);
        let mut actual: BTreeMap<&str, &TensorRef<'_>> = BTreeMap::new();

        for tensor in tensors {
            if actual.insert(tensor.name, tensor).is_some() {
                return Err(LayoutError::Duplicate(tensor.name.to_string()));
            }
        }

        let mut entries = Vec::with_capacity(specs.len());

        for spec in &specs {
            let found = actual
                .get(spec.name.as_str())
                .ok_or_else(|| LayoutError::Absent(spec.name.clone()))?;

            check(spec, found)?;
            entries.push(Entry {
                name: spec.name.clone(),
                role: spec.role,
                dtype: found.dtype,
                shape: found.shape.to_vec(),
            });
        }

        let expected: std::collections::BTreeSet<&str> =
            specs.iter().map(|spec| spec.name.as_str()).collect();

        if let Some(extra) = actual.keys().find(|name| !expected.contains(*name)) {
            return Err(LayoutError::Unlisted(extra.to_string()));
        }

        entries.sort_by(|left, right| left.name.cmp(&right.name));

        Ok(Layout {
            config: config.clone(),
            entries,
        })
    }

    /// Open a shard directory and check it.
    pub fn open(dir: &Path) -> Result<Layout, LayoutError> {
        let config = Config::read(dir)?;
        let store = crate::minimax_h3::safetensors::open_dir(dir)
            .map_err(|err| LayoutError::Reader(err.token()))?;

        Layout::from_store(&config, &store)
    }

    /// Check an already-open store.
    pub fn from_store(config: &Config, store: &Store) -> Result<Layout, LayoutError> {
        let mut tensors = Vec::new();

        for shard in store.shards() {
            for tensor in &shard.tensors {
                tensors.push(TensorRef {
                    name: &tensor.name,
                    dtype: tensor.dtype,
                    shape: &tensor.shape,
                });
            }
        }

        Layout::build(config, &tensors)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn find(&self, name: &str) -> Option<&Entry> {
        self.entries
            .binary_search_by(|entry| entry.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.entries[index])
    }

    /// Tensors per main block, `adaln_proj` included.
    pub fn block_stride(&self) -> usize {
        BLOCK_TENSORS
    }
}

/// Ten per main block: eight block tensors plus the AdaLN pair.
const BLOCK_TENSORS: usize = 10;

fn check(spec: &Spec, found: &TensorRef<'_>) -> Result<(), LayoutError> {
    if found.dtype != spec.dtype {
        return Err(LayoutError::Dtype {
            name: spec.name.clone(),
            expected: spec.dtype,
            found: found.dtype,
        });
    }

    if found.shape != spec.shape.as_slice() {
        return Err(LayoutError::Shape {
            name: spec.name.clone(),
            expected: spec.shape.clone(),
            found: found.shape.to_vec(),
        });
    }

    Ok(())
}

/// Every tensor the config implies, in generation order.
fn specs(config: &Config) -> Vec<Spec> {
    let hidden = config.hidden;
    let mut specs = Vec::new();

    // Text conditioning and position (`h3_dit.c:808-811`, `:884`).
    spec(
        &mut specs,
        "condition_proj.weight",
        Role::ConditionWeight,
        Dtype::Bf16,
        &[hidden, config.text_dim],
    );
    spec(
        &mut specs,
        "condition_proj.bias",
        Role::ConditionBias,
        Dtype::Bf16,
        &[hidden],
    );
    spec(
        &mut specs,
        "rope.inv_freq",
        Role::RopeFreq,
        Dtype::F32,
        &[config.rope_freqs],
    );

    // Timestep MLP, F32 end to end (`h3_dit_schedule.c:170-178`).
    let time_hidden = config.time_hidden;
    spec(
        &mut specs,
        "time_embedder.proj_in.weight",
        Role::TimeInWeight,
        Dtype::F32,
        &[time_hidden, config.time_input],
    );
    spec(
        &mut specs,
        "time_embedder.proj_in.bias",
        Role::TimeInBias,
        Dtype::F32,
        &[time_hidden],
    );
    spec(
        &mut specs,
        "time_embedder.proj_out.weight",
        Role::TimeOutWeight,
        Dtype::F32,
        &[config.time_dim, time_hidden],
    );
    spec(
        &mut specs,
        "time_embedder.proj_out.bias",
        Role::TimeOutBias,
        Dtype::F32,
        &[config.time_dim],
    );

    // Patch projections are F32 (`h3_dit.c:1295-1301`); their widths are the
    // patchified latent row sizes.
    let video_patch = config.video_patch();
    let audio_patch = config.audio_latents;
    spec(
        &mut specs,
        "video_patch_proj.weight",
        Role::VideoPatchWeight,
        Dtype::F32,
        &[hidden, video_patch],
    );
    spec(
        &mut specs,
        "video_patch_proj.bias",
        Role::VideoPatchBias,
        Dtype::F32,
        &[hidden],
    );
    spec(
        &mut specs,
        "audio_patch_proj.weight",
        Role::AudioPatchWeight,
        Dtype::F32,
        &[hidden, audio_patch],
    );
    spec(
        &mut specs,
        "audio_patch_proj.bias",
        Role::AudioPatchBias,
        Dtype::F32,
        &[hidden],
    );

    // Final layer (`h3_dit.c:1303-1311`, `h3_dit_schedule.c:304-307`).
    spec(
        &mut specs,
        "final_layer.norm.weight",
        Role::FinalNorm,
        Dtype::Bf16,
        &[hidden],
    );
    spec(
        &mut specs,
        "final_layer.video_out.weight",
        Role::FinalVideoWeight,
        Dtype::F32,
        &[video_patch, hidden],
    );
    spec(
        &mut specs,
        "final_layer.video_out.bias",
        Role::FinalVideoBias,
        Dtype::F32,
        &[video_patch],
    );
    spec(
        &mut specs,
        "final_layer.audio_out.weight",
        Role::FinalAudioWeight,
        Dtype::F32,
        &[audio_patch, hidden],
    );
    spec(
        &mut specs,
        "final_layer.audio_out.bias",
        Role::FinalAudioBias,
        Dtype::F32,
        &[audio_patch],
    );
    spec(
        &mut specs,
        "final_layer.adaln_proj.linear.weight",
        Role::FinalAdalnWeight,
        Dtype::Bf16,
        &[config.final_adaln_out, config.time_dim],
    );
    spec(
        &mut specs,
        "final_layer.adaln_proj.linear.bias",
        Role::FinalAdalnBias,
        Dtype::Bf16,
        &[config.final_adaln_out],
    );

    // The refiner runs the same block body on text rows (`h3_dit.c:819-823`).
    for block in 0..config.refiner_layers {
        let prefix = format!("{REFINER_PREFIX}{block}.");
        block_specs(&mut specs, &prefix, block, config, RoleSet::Refiner);
    }
    spec(
        &mut specs,
        REFINER_FINAL,
        Role::RefinerFinalNorm,
        Dtype::Bf16,
        &[hidden],
    );

    for block in 0..config.layers {
        let prefix = format!("{BLOCK_PREFIX}{block}.");
        block_specs(&mut specs, &prefix, block, config, RoleSet::Main);
    }

    specs
}

enum RoleSet {
    Main,
    Refiner,
}

fn block_specs(specs: &mut Vec<Spec>, prefix: &str, block: u32, config: &Config, set: RoleSet) {
    let hidden = config.hidden;
    let inner = config.inner();
    let head_dim = config.head_dim;
    let ffn = config.ffn;

    if let RoleSet::Refiner = set {
        refiner_specs(specs, prefix, block, hidden, inner, head_dim, ffn);
        return;
    }

    spec(
        specs,
        &format!("{prefix}norm1.weight"),
        Role::BlockNorm1(block),
        Dtype::Bf16,
        &[hidden],
    );
    spec(
        specs,
        &format!("{prefix}norm2.weight"),
        Role::BlockNorm2(block),
        Dtype::Bf16,
        &[hidden],
    );
    spec(
        specs,
        &format!("{prefix}attn.qkv_proj.weight"),
        Role::BlockQkv(block),
        Dtype::Bf16,
        &[inner * 3, hidden],
    );
    spec(
        specs,
        &format!("{prefix}attn.q_norm.weight"),
        Role::BlockQNorm(block),
        Dtype::Bf16,
        &[head_dim],
    );
    spec(
        specs,
        &format!("{prefix}attn.k_norm.weight"),
        Role::BlockKNorm(block),
        Dtype::Bf16,
        &[head_dim],
    );
    spec(
        specs,
        &format!("{prefix}attn.out_proj.weight"),
        Role::BlockOut(block),
        Dtype::Bf16,
        &[hidden, inner],
    );
    spec(
        specs,
        &format!("{prefix}mlp.fc1.weight"),
        Role::BlockFc1(block),
        Dtype::Bf16,
        &[ffn * 2, hidden],
    );
    spec(
        specs,
        &format!("{prefix}mlp.fc2.weight"),
        Role::BlockFc2(block),
        Dtype::Bf16,
        &[hidden, ffn],
    );

    // Six slots for three modalities (`h3_dit_schedule.c:12-14`).
    spec(
        specs,
        &format!("{prefix}adaln_proj.linear.weight"),
        Role::BlockAdalnWeight(block),
        Dtype::Bf16,
        &[config.adaln_out, config.time_dim],
    );
    spec(
        specs,
        &format!("{prefix}adaln_proj.linear.bias"),
        Role::BlockAdalnBias(block),
        Dtype::Bf16,
        &[config.adaln_out],
    );
}

fn refiner_specs(
    specs: &mut Vec<Spec>,
    prefix: &str,
    block: u32,
    hidden: u64,
    inner: u64,
    head_dim: u64,
    ffn: u64,
) {
    spec(
        specs,
        &format!("{prefix}norm1.weight"),
        Role::RefinerNorm1(block),
        Dtype::Bf16,
        &[hidden],
    );
    spec(
        specs,
        &format!("{prefix}norm2.weight"),
        Role::RefinerNorm2(block),
        Dtype::Bf16,
        &[hidden],
    );
    spec(
        specs,
        &format!("{prefix}attn.qkv_proj.weight"),
        Role::RefinerQkv(block),
        Dtype::Bf16,
        &[inner * 3, hidden],
    );
    spec(
        specs,
        &format!("{prefix}attn.q_norm.weight"),
        Role::RefinerQNorm(block),
        Dtype::Bf16,
        &[head_dim],
    );
    spec(
        specs,
        &format!("{prefix}attn.k_norm.weight"),
        Role::RefinerKNorm(block),
        Dtype::Bf16,
        &[head_dim],
    );
    spec(
        specs,
        &format!("{prefix}attn.out_proj.weight"),
        Role::RefinerOut(block),
        Dtype::Bf16,
        &[hidden, inner],
    );
    spec(
        specs,
        &format!("{prefix}mlp.fc1.weight"),
        Role::RefinerFc1(block),
        Dtype::Bf16,
        &[ffn * 2, hidden],
    );
    spec(
        specs,
        &format!("{prefix}mlp.fc2.weight"),
        Role::RefinerFc2(block),
        Dtype::Bf16,
        &[hidden, ffn],
    );
}

fn spec(specs: &mut Vec<Spec>, name: &str, role: Role, dtype: Dtype, shape: &[u64]) {
    specs.push(Spec {
        name: name.to_string(),
        role,
        dtype,
        shape: shape.to_vec(),
    });
}

fn string<'a>(root: &'a Value, key: &str) -> Result<&'a str, LayoutError> {
    root.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| LayoutError::Config {
            detail: format!("{key} is missing or not a string"),
        })
}

fn uint(root: &Value, key: &str) -> Result<u64, LayoutError> {
    root.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| LayoutError::Config {
            detail: format!("{key} is missing or not a non-negative integer"),
        })
}

fn count(root: &Value, key: &str) -> Result<u32, LayoutError> {
    let value = uint(root, key)?;

    u32::try_from(value).map_err(|_| LayoutError::Config {
        detail: format!("{key} {value} does not fit a u32"),
    })
}

fn array(root: &Value, key: &str) -> Result<Vec<u64>, LayoutError> {
    let values = root
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| LayoutError::Config {
            detail: format!("{key} is missing or not an array"),
        })?;

    values
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| LayoutError::Config {
                detail: format!("{key} holds a non-integer value"),
            })
        })
        .collect()
}

/// Why a shard set is not the checkpoint the config describes.
#[derive(Debug)]
pub enum LayoutError {
    Io {
        path: std::path::PathBuf,
        detail: String,
    },
    Config {
        detail: String,
    },
    Reader(String),
    /// The config expects this name and no shard has it.
    Absent(String),
    /// A shard has this name and the config does not expect it.
    Unlisted(String),
    /// Two shards carry the same name.
    Duplicate(String),
    Dtype {
        name: String,
        expected: Dtype,
        found: Dtype,
    },
    Shape {
        name: String,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
}

impl LayoutError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            LayoutError::Io { path, detail } => {
                format!("io {}: {detail}", path.display())
            }
            LayoutError::Config { detail } => format!("config: {detail}"),
            LayoutError::Reader(detail) => format!("reader: {detail}"),
            LayoutError::Absent(name) => format!("absent {name}"),
            LayoutError::Unlisted(name) => format!("unlisted {name}"),
            LayoutError::Duplicate(name) => format!("duplicate {name}"),
            LayoutError::Dtype {
                name,
                expected,
                found,
            } => format!(
                "dtype {name}: expected {}, found {}",
                expected.name(),
                found.name()
            ),
            LayoutError::Shape {
                name,
                expected,
                found,
            } => format!("shape {name}: expected {expected:?}, found {found:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint's own `FL2VA/transformer/config.json`.
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
        Config::parse(CONFIG).expect("the checkpoint config parses")
    }

    /// Every expected tensor, as a synthetic shard set.
    fn synthesized(config: &Config) -> Vec<(String, Dtype, Vec<u64>)> {
        specs(config)
            .into_iter()
            .map(|spec| (spec.name, spec.dtype, spec.shape))
            .collect()
    }

    fn refs(tensors: &[(String, Dtype, Vec<u64>)]) -> Vec<TensorRef<'_>> {
        tensors
            .iter()
            .map(|(name, dtype, shape)| TensorRef {
                name,
                dtype: *dtype,
                shape,
            })
            .collect()
    }

    #[test]
    fn config_parses_the_checkpoint_geometry() {
        let config = config();

        assert_eq!(config.hidden, 5376);
        assert_eq!(config.layers, 50);
        assert_eq!(config.refiner_layers, 2);
        assert_eq!(config.inner(), 7168);
        assert_eq!(config.video_patch(), 96);
        assert_eq!(config.time_dim, 2688);
        assert_eq!(config.adaln_out, 96768);
        assert_eq!(config.final_adaln_out, 10752);
    }

    #[test]
    fn expects_the_checkpoints_535_tensors() {
        let tensors = synthesized(&config());
        let refiner = tensors
            .iter()
            .filter(|(name, _, _)| name.starts_with("token_refiner.blocks.0."))
            .count();

        // 50 blocks x 10, 2 refiner blocks x 8, the refiner norm, and 18 globals.
        assert_eq!(tensors.len(), 500 + 16 + 1 + 18);
        assert_eq!(tensors.len(), 535);
        // `load_block` loads eight (`h3_dit.c:517-524`); the refiner has no AdaLN.
        assert_eq!(refiner, 8);
    }

    #[test]
    fn accepts_the_exact_set() {
        let config = config();
        let tensors = synthesized(&config);
        let layout = Layout::build(&config, &refs(&tensors)).expect("the exact set passes");

        assert_eq!(layout.entries().len(), 535);
        assert_eq!(layout.block_stride(), BLOCK_TENSORS);
        assert_eq!(
            layout.find("blocks.7.attn.qkv_proj.weight").map(|e| e.role),
            Some(Role::BlockQkv(7))
        );
        assert_eq!(
            layout
                .find("token_refiner.blocks.1.mlp.fc2.weight")
                .map(|e| e.role),
            Some(Role::RefinerFc2(1))
        );
        assert_eq!(
            layout
                .find("final_layer.adaln_proj.linear.weight")
                .map(|e| e.role),
            Some(Role::FinalAdalnWeight)
        );
    }

    #[test]
    fn rejects_a_missing_tensor() {
        let config = config();
        let mut tensors = synthesized(&config);
        tensors.retain(|(name, _, _)| name != "blocks.13.norm2.weight");

        let error = Layout::build(&config, &refs(&tensors)).expect_err("a hole fails");

        assert!(
            error.token().contains("absent blocks.13.norm2.weight"),
            "{}",
            error.token()
        );
    }

    #[test]
    fn rejects_an_extra_tensor() {
        let config = config();
        let mut tensors = synthesized(&config);
        tensors.push((
            "blocks.50.norm1.weight".to_string(),
            Dtype::Bf16,
            vec![5376],
        ));

        let error = Layout::build(&config, &refs(&tensors)).expect_err("an extra fails");

        assert!(
            error.token().contains("unlisted blocks.50.norm1.weight"),
            "{}",
            error.token()
        );
    }

    #[test]
    fn rejects_a_wrong_dtype() {
        let config = config();
        let mut tensors = synthesized(&config);
        let index = tensors
            .iter()
            .position(|(name, _, _)| name == "blocks.0.mlp.fc1.weight")
            .expect("the tensor exists");
        tensors[index].1 = Dtype::F32;

        let error = Layout::build(&config, &refs(&tensors)).expect_err("f32 fc1 fails");

        assert!(
            error.token().starts_with("dtype blocks.0.mlp.fc1.weight"),
            "{}",
            error.token()
        );
    }

    #[test]
    fn rejects_a_wrong_shape() {
        let config = config();
        let mut tensors = synthesized(&config);
        let index = tensors
            .iter()
            .position(|(name, _, _)| name == "video_patch_proj.weight")
            .expect("the tensor exists");
        tensors[index].2 = vec![5376, 72];

        let error = Layout::build(&config, &refs(&tensors)).expect_err("patch width 72 fails");

        assert!(
            error.token().contains("shape video_patch_proj.weight"),
            "{}",
            error.token()
        );
    }

    #[test]
    fn rejects_a_different_time_width() {
        let text = CONFIG.replace("\"timestep_input_dim\": 256", "\"timestep_input_dim\": 512");

        let error = Config::parse(&text).expect_err("512 sinusoids fail");

        assert!(
            error.token().contains("timestep_input_dim"),
            "{}",
            error.token()
        );
    }
}
