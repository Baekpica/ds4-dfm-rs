//! Qwen3-VL text-encoder tensor layout for the prompt path.
//!
//! The reference names its weights by string and checks each one as it loads
//! it. This module turns that implicit schema into data: the 551 tensors the
//! encoder path reads, with their role, dtype and shape, checked against a
//! shard set before the forward is written.
//!
//! Source of truth `/data/h3.c` at `eae5d54`, `h3_text_encoder.c`:
//!
//! - `:12-24` the geometry constants: 50 layers, 5120 hidden, 25600
//!   intermediate, 64 query heads, 8 KV heads, head dimension 128, vocabulary
//!   151936, rope theta 5e6, RMS epsilon 1e-6.
//! - `:147-180` `layer_weights_load`: the eleven names and shapes of one
//!   layer under `model.language_model.layers.N.`.
//! - `:622-625` the token embedding at `[VOCAB, HIDDEN]`.
//! - `:219-247` `read_weight_bf16`: dtype BF16, the rank, and every dimension
//!   checked before a byte is read.
//! - `h3_text_encoder.h:6` the encoder's output width, `H3_TEXT_HIDDEN_SIZE`.
//!
//! Deviation, named, as in `minimax_h3::layout`: the reference reads geometry
//! from compile-time constants and never consults `config.json`. This module
//! takes the geometry from the checkpoint's `text_config` and refuses a config
//! that disagrees with the reference's constants, so a mismatched artifact is
//! reported instead of silently loaded.
//!
//! The reference runs 50 of the 64 declared layers (`:14`, `:649`) and never
//! reads `model.language_model.norm.weight` or the vision tower on this path.
//! Those tensors are counted, not rejected: `Outside` says why each one is not
//! used.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::minimax_h3::safetensors::{Dtype, Store};

/// `TEXT_LAYERS` (`h3_text_encoder.c:14`).
pub const LAYERS: u32 = 50;

/// `TEXT_HIDDEN` (`h3_text_encoder.c:16`).
pub const HIDDEN: u64 = 5120;

/// `TEXT_INTERMEDIATE` (`h3_text_encoder.c:17`).
pub const INTERMEDIATE: u64 = 25600;

/// `TEXT_QUERY_HEADS` (`h3_text_encoder.c:18`).
pub const QUERY_HEADS: u64 = 64;

/// `TEXT_KV_HEADS` (`h3_text_encoder.c:19`).
pub const KV_HEADS: u64 = 8;

/// `TEXT_HEAD_DIM` (`h3_text_encoder.c:20`).
pub const HEAD_DIM: u64 = 128;

/// `TEXT_VOCAB` (`h3_text_encoder.c:15`).
pub const VOCAB: u64 = 151936;

/// `TEXT_ROPE_THETA` (`h3_text_encoder.c:27`).
pub const ROPE_THETA: f64 = 5_000_000.0;

/// `TEXT_RMS_EPSILON` (`h3_text_encoder.c:26`).
pub const RMS_EPSILON: f64 = 1e-6;

/// `LANGUAGE_MODEL` prefix of every tensor this path reads.
pub const LANGUAGE_PREFIX: &str = "model.language_model.";

/// The embedding (`h3_text_encoder.c:624`).
pub const EMBEDDING: &str = "model.language_model.embed_tokens.weight";

const CONFIG_FILE: &str = "config.json";
const LAYER_PREFIX: &str = "model.language_model.layers.";
const FINAL_NORM: &str = "model.language_model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";
const VISION_PREFIX: &str = "model.visual.";

/// Which tensor of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Part {
    InputNorm,
    Query,
    Key,
    Value,
    QueryNorm,
    KeyNorm,
    Output,
    PostNorm,
    Gate,
    Up,
    Down,
}

/// One place in the encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Embedding,
    Layer { layer: u32, part: Part },
}

/// One expected tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub role: Role,
    pub shape: Vec<u64>,
}

/// Why a tensor outside the encoder path is there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outside {
    /// Layers 50..63: declared by the config, never executed
    /// (`h3_text_encoder.c:14`).
    pub deferred_layers: usize,
    /// `model.language_model.norm.weight`: the encoder returns the layer-49
    /// hidden state, so the final norm is not applied.
    pub final_norm: usize,
    /// `lm_head.weight`: the prompt path needs hidden states, never logits.
    pub lm_head: usize,
    /// The vision tower, used by the multimodal path, not by prompt encoding.
    pub vision: usize,
    /// Anything else; a non-zero count is worth reporting.
    pub other: usize,
}

impl Outside {
    /// The total counted.
    pub fn total(&self) -> usize {
        self.deferred_layers + self.final_norm + self.lm_head + self.vision + self.other
    }
}

/// A borrowed tensor, so the layout can be checked against a store or a
/// synthetic set without copying payloads.
pub struct TensorRef<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [u64],
}

/// The geometry the checkpoint's `FL2VA/text_encoder/config.json` declares.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub declared_layers: u32,
    pub hidden: u64,
    pub intermediate: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub vocab: u64,
    pub rope_theta: f64,
    pub rms_epsilon: f64,
}

impl Config {
    /// Read `<dir>/config.json`.
    pub fn read(dir: &Path) -> Result<Config, TextError> {
        let path = dir.join(CONFIG_FILE);
        let text = std::fs::read_to_string(&path).map_err(|error| TextError::Io {
            path: path.clone(),
            detail: error.to_string(),
        })?;

        Config::parse(&text)
    }

    /// Parse the JSON text.
    pub fn parse(text: &str) -> Result<Config, TextError> {
        let root: Value = serde_json::from_str(text).map_err(|error| TextError::Config {
            detail: error.to_string(),
        })?;

        let model_type = string(&root, "model_type")?;

        if model_type != "qwen3_vl" {
            return Err(TextError::Config {
                detail: format!("model_type is {model_type}, expected qwen3_vl"),
            });
        }

        let text_config = root.get("text_config").ok_or(TextError::Config {
            detail: "text_config is absent".to_string(),
        })?;

        let config = Config {
            declared_layers: count(text_config, "num_hidden_layers")?,
            hidden: uint(text_config, "hidden_size")?,
            intermediate: uint(text_config, "intermediate_size")?,
            heads: uint(text_config, "num_attention_heads")?,
            kv_heads: uint(text_config, "num_key_value_heads")?,
            head_dim: uint(text_config, "head_dim")?,
            vocab: uint(text_config, "vocab_size")?,
            rope_theta: real(text_config, "rope_theta")?,
            rms_epsilon: real(text_config, "rms_norm_eps")?,
        };

        config.check()?;

        Ok(config)
    }

    /// The reference reads these as compile-time constants; a config that
    /// disagrees would shape every tensor differently.
    fn check(&self) -> Result<(), TextError> {
        let fields = [
            ("hidden_size", self.hidden as f64, HIDDEN as f64),
            (
                "intermediate_size",
                self.intermediate as f64,
                INTERMEDIATE as f64,
            ),
            ("num_attention_heads", self.heads as f64, QUERY_HEADS as f64),
            ("num_key_value_heads", self.kv_heads as f64, KV_HEADS as f64),
            ("head_dim", self.head_dim as f64, HEAD_DIM as f64),
            ("vocab_size", self.vocab as f64, VOCAB as f64),
            ("rope_theta", self.rope_theta, ROPE_THETA),
            ("rms_norm_eps", self.rms_epsilon, RMS_EPSILON),
        ];

        for (field, got, want) in fields {
            if got != want {
                return Err(TextError::Config {
                    detail: format!("text_config.{field} is {got}, the reference uses {want}"),
                });
            }
        }

        if self.declared_layers < LAYERS {
            return Err(TextError::Config {
                detail: format!(
                    "text_config.num_hidden_layers is {}, the encoder runs {LAYERS}",
                    self.declared_layers
                ),
            });
        }

        Ok(())
    }

    /// Query width, `heads * head_dim` (`h3_text_encoder.c:21`).
    pub fn query_dim(&self) -> u64 {
        self.heads * self.head_dim
    }

    /// Key and value width, `kv_heads * head_dim` (`h3_text_encoder.c:22`).
    pub fn kv_dim(&self) -> u64 {
        self.kv_heads * self.head_dim
    }
}

/// The checked encoder path over one shard set.
#[derive(Debug)]
pub struct Layout {
    config: Config,
    entries: Vec<Entry>,
    outside: Outside,
}

impl Layout {
    /// Check the expected names against an arbitrary tensor set.
    pub fn build(config: &Config, tensors: &[TensorRef<'_>]) -> Result<Layout, TextError> {
        let mut actual: BTreeMap<&str, &TensorRef<'_>> = BTreeMap::new();

        for tensor in tensors {
            if actual.insert(tensor.name, tensor).is_some() {
                return Err(TextError::Duplicate {
                    name: tensor.name.to_string(),
                });
            }
        }

        let specs = specs(config);
        let mut entries = Vec::with_capacity(specs.len());

        for spec in &specs {
            let found = actual.get(spec.name.as_str()).ok_or(TextError::Missing {
                name: spec.name.clone(),
            })?;

            if found.dtype != Dtype::Bf16 {
                return Err(TextError::Dtype {
                    name: spec.name.clone(),
                    got: found.dtype.name(),
                });
            }

            if found.shape != spec.shape.as_slice() {
                return Err(TextError::Shape {
                    name: spec.name.clone(),
                    want: spec.shape.clone(),
                    got: found.shape.to_vec(),
                });
            }

            entries.push(spec.clone());
        }

        let outside = classify(&specs, tensors);

        Ok(Layout {
            config: config.clone(),
            entries,
            outside,
        })
    }

    /// Read `config.json`, every shard header, and check the encoder path.
    pub fn open(dir: &Path) -> Result<Layout, TextError> {
        Layout::with_store(&Config::read(dir)?, &store(dir)?)
    }

    /// Check an already-opened store.
    pub fn with_store(config: &Config, store: &Store) -> Result<Layout, TextError> {
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

    /// The 551 encoder tensors, embedding first.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// What the shard set holds that the prompt path never reads.
    pub fn outside(&self) -> Outside {
        self.outside
    }
}

fn store(dir: &Path) -> Result<Store, TextError> {
    crate::minimax_h3::safetensors::open_dir(dir).map_err(|error| TextError::Reader(error.token()))
}

/// Sort every actual tensor into "expected" or one `Outside` reason.
fn classify(specs: &[Entry], tensors: &[TensorRef<'_>]) -> Outside {
    let expected: std::collections::BTreeSet<&str> =
        specs.iter().map(|spec| spec.name.as_str()).collect();
    let mut outside = Outside::default();

    for tensor in tensors {
        if expected.contains(tensor.name) {
            continue;
        }

        if tensor.name.starts_with(VISION_PREFIX) {
            outside.vision += 1;
        } else if tensor.name == FINAL_NORM {
            outside.final_norm += 1;
        } else if tensor.name == LM_HEAD {
            outside.lm_head += 1;
        } else if deferred_layer(tensor.name).is_some() {
            outside.deferred_layers += 1;
        } else {
            outside.other += 1;
        }
    }

    outside
}

/// The layer index of a tensor the encoder does not read.
fn deferred_layer(name: &str) -> Option<u32> {
    let rest = name.strip_prefix(LAYER_PREFIX)?;
    let (layer, _) = rest.split_once('.')?;
    let layer: u32 = layer.parse().ok()?;

    (layer >= LAYERS).then_some(layer)
}

/// Every encoder tensor, embedding first, then layer by layer.
fn specs(config: &Config) -> Vec<Entry> {
    let mut specs = Vec::with_capacity(1 + LAYERS as usize * PARTS.len());

    specs.push(Entry {
        name: EMBEDDING.to_string(),
        role: Role::Embedding,
        shape: vec![config.vocab, config.hidden],
    });

    for layer in 0..LAYERS {
        for (part, suffix, shape) in parts(config) {
            specs.push(Entry {
                name: format!("{LAYER_PREFIX}{layer}.{suffix}"),
                role: Role::Layer { layer, part },
                shape,
            });
        }
    }

    specs
}

/// The eleven tensors of one layer (`h3_text_encoder.c:166-177`).
const PARTS: [Part; 11] = [
    Part::InputNorm,
    Part::Query,
    Part::Key,
    Part::Value,
    Part::QueryNorm,
    Part::KeyNorm,
    Part::Output,
    Part::PostNorm,
    Part::Gate,
    Part::Up,
    Part::Down,
];

fn parts(config: &Config) -> Vec<(Part, &'static str, Vec<u64>)> {
    let hidden = config.hidden;
    let intermediate = config.intermediate;

    vec![
        (Part::InputNorm, "input_layernorm.weight", vec![hidden]),
        (
            Part::Query,
            "self_attn.q_proj.weight",
            vec![config.query_dim(), hidden],
        ),
        (
            Part::Key,
            "self_attn.k_proj.weight",
            vec![config.kv_dim(), hidden],
        ),
        (
            Part::Value,
            "self_attn.v_proj.weight",
            vec![config.kv_dim(), hidden],
        ),
        (
            Part::QueryNorm,
            "self_attn.q_norm.weight",
            vec![config.head_dim],
        ),
        (
            Part::KeyNorm,
            "self_attn.k_norm.weight",
            vec![config.head_dim],
        ),
        (
            Part::Output,
            "self_attn.o_proj.weight",
            vec![hidden, config.query_dim()],
        ),
        (
            Part::PostNorm,
            "post_attention_layernorm.weight",
            vec![hidden],
        ),
        (
            Part::Gate,
            "mlp.gate_proj.weight",
            vec![intermediate, hidden],
        ),
        (Part::Up, "mlp.up_proj.weight", vec![intermediate, hidden]),
        (
            Part::Down,
            "mlp.down_proj.weight",
            vec![hidden, intermediate],
        ),
    ]
}

/// Why a text-encoder layout could not be checked.
#[derive(Debug)]
pub enum TextError {
    Io {
        path: PathBuf,
        detail: String,
    },
    /// The config is not the shape the reference reads.
    Config {
        detail: String,
    },
    /// The shard set could not be read.
    Reader(String),
    /// One name appears in two shards.
    Duplicate {
        name: String,
    },
    /// A tensor the encoder reads is absent.
    Missing {
        name: String,
    },
    /// A tensor is not BF16.
    Dtype {
        name: String,
        got: &'static str,
    },
    /// A tensor's shape is not the one the reference loads.
    Shape {
        name: String,
        want: Vec<u64>,
        got: Vec<u64>,
    },
}

impl TextError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            TextError::Io { path, detail } => format!("io {}: {detail}", path.display()),
            TextError::Config { detail } => format!("config: {detail}"),
            TextError::Reader(detail) => format!("reader: {detail}"),
            TextError::Duplicate { name } => format!("duplicate tensor: {name}"),
            TextError::Missing { name } => format!("missing tensor: {name}"),
            TextError::Dtype { name, got } => format!("tensor {name} is {got}, expected BF16"),
            TextError::Shape { name, want, got } => {
                format!("tensor {name} is {got:?}, expected {want:?}")
            }
        }
    }
}

impl std::fmt::Display for TextError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for TextError {}

fn uint(root: &Value, key: &str) -> Result<u64, TextError> {
    root.get(key)
        .and_then(Value::as_u64)
        .ok_or(TextError::Config {
            detail: format!("{key} is absent or not an unsigned integer"),
        })
}

fn count(root: &Value, key: &str) -> Result<u32, TextError> {
    let value = uint(root, key)?;

    u32::try_from(value).map_err(|_| TextError::Config {
        detail: format!("{key} is {value}, which does not fit a layer count"),
    })
}

fn real(root: &Value, key: &str) -> Result<f64, TextError> {
    root.get(key)
        .and_then(Value::as_f64)
        .ok_or(TextError::Config {
            detail: format!("{key} is absent or not a number"),
        })
}

fn string<'a>(root: &'a Value, key: &str) -> Result<&'a str, TextError> {
    root.get(key)
        .and_then(Value::as_str)
        .ok_or(TextError::Config {
            detail: format!("{key} is absent or not a string"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint's own `FL2VA/text_encoder/config.json`.
    const CONFIG: &str = r#"{
      "architectures": ["Qwen3VLForConditionalGeneration"],
      "model_type": "qwen3_vl",
      "text_config": {
        "head_dim": 128,
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_attention_heads": 64,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000,
        "vocab_size": 151936
      }
    }"#;

    fn config() -> Config {
        Config::parse(CONFIG).expect("the checkpoint config parses")
    }

    fn synthesized(config: &Config) -> Vec<(String, Dtype, Vec<u64>)> {
        specs(config)
            .into_iter()
            .map(|spec| (spec.name, Dtype::Bf16, spec.shape))
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
    fn the_config_declares_the_geometry_the_reference_hard_codes() {
        let config = config();

        assert_eq!(config.declared_layers, 64);
        assert_eq!(config.query_dim(), 8192);
        assert_eq!(config.kv_dim(), 1024);
    }

    #[test]
    fn the_path_is_the_embedding_plus_fifty_layers_of_eleven() {
        let config = config();
        let layout = Layout::build(&config, &refs(&synthesized(&config))).expect("a layout");

        assert_eq!(layout.entries().len(), 1 + LAYERS as usize * 11);
        assert_eq!(layout.entries()[0].role, Role::Embedding);
        assert_eq!(
            layout.entries()[0].shape,
            vec![VOCAB, HIDDEN],
            "the embedding is [vocab, hidden]"
        );
        assert_eq!(
            layout.entries()[1].role,
            Role::Layer {
                layer: 0,
                part: Part::InputNorm
            }
        );
        assert_eq!(layout.outside(), Outside::default());
    }

    #[test]
    fn the_remaining_layers_the_norm_and_the_vision_tower_are_counted() {
        let config = config();
        let mut tensors = synthesized(&config);

        tensors.push((FINAL_NORM.to_string(), Dtype::Bf16, vec![HIDDEN]));
        tensors.push((LM_HEAD.to_string(), Dtype::Bf16, vec![VOCAB, HIDDEN]));
        for layer in LAYERS..config.declared_layers {
            tensors.push((
                format!("{LAYER_PREFIX}{layer}.input_layernorm.weight"),
                Dtype::Bf16,
                vec![HIDDEN],
            ));
        }
        for block in 0..27 {
            tensors.push((
                format!("model.visual.blocks.{block}.norm1.weight"),
                Dtype::Bf16,
                vec![1152],
            ));
        }

        let layout = Layout::build(&config, &refs(&tensors)).expect("a layout");
        let outside = layout.outside();

        assert_eq!(
            outside.deferred_layers,
            (config.declared_layers - LAYERS) as usize
        );
        assert_eq!(outside.final_norm, 1);
        assert_eq!(outside.lm_head, 1);
        assert_eq!(outside.vision, 27);
        assert_eq!(outside.other, 0);
    }

    #[test]
    fn a_wrong_shape_is_named_with_both_shapes() {
        let config = config();
        let mut tensors = synthesized(&config);
        // q_proj is [8192, 5120]; halve the rows.
        tensors[2].2[0] = 4096;

        let error = Layout::build(&config, &refs(&tensors)).expect_err("a refusal");

        assert!(error.token().contains("self_attn.q_proj.weight"));
        assert!(error.token().contains("4096"));
    }

    #[test]
    fn a_non_bf16_tensor_and_an_absent_tensor_are_refused() {
        let config = config();
        let mut tensors = synthesized(&config);
        tensors[0].1 = Dtype::F32;
        assert!(matches!(
            Layout::build(&config, &refs(&tensors)),
            Err(TextError::Dtype { .. })
        ));

        let mut tensors = synthesized(&config);
        tensors.remove(0);
        assert!(matches!(
            Layout::build(&config, &refs(&tensors)),
            Err(TextError::Missing { .. })
        ));
    }

    #[test]
    fn a_config_that_disagrees_with_the_reference_is_refused() {
        let wrong = CONFIG.replace("\"hidden_size\": 5120", "\"hidden_size\": 4096");
        assert!(matches!(
            Config::parse(&wrong),
            Err(TextError::Config { .. })
        ));

        let short = CONFIG.replace("\"num_hidden_layers\": 64", "\"num_hidden_layers\": 32");
        assert!(matches!(
            Config::parse(&short),
            Err(TextError::Config { .. })
        ));

        let other = CONFIG.replace("\"model_type\": \"qwen3_vl\"", "\"model_type\": \"qwen3\"");
        assert!(matches!(
            Config::parse(&other),
            Err(TextError::Config { .. })
        ));
    }
}
