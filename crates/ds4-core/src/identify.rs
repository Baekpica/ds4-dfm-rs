//! Host-owned family/shape selection from mmap'd GGUF metadata.
//!
//! Architecture dispatch matches C `config_validate_model`. DeepSeek
//! Flash/Pro uses `ds4_select_shape_from_metadata`. Other families take
//! the pinned catalog shape. Full required-key validators are host-owned
//! (`validate`). Tensor directory + split remap are host-owned (`tensors`).

use std::path::Path;

use crate::gguf::{GgufError, GgufFile};
use crate::shape::{
    route_architecture, select_shape_from_metadata, shape_for_variant, ArchRoute, DeepSeekDims,
    Shape,
};

#[derive(Debug)]
pub enum IdentifyError {
    Gguf(GgufError),
    MissingKey(&'static str),
    UnsupportedShape,
    UnsupportedArch(Vec<u8>),
    Artifact(String),
}

impl IdentifyError {
    pub fn token(&self) -> String {
        match self {
            IdentifyError::Gguf(e) => e.token(),
            IdentifyError::MissingKey(k) => format!("missing-key {k}"),
            IdentifyError::UnsupportedShape => "unsupported".into(),
            IdentifyError::Artifact(e) => format!("invalid-artifact {e}"),
            IdentifyError::UnsupportedArch(a) => {
                format!("unsupported-arch {}", String::from_utf8_lossy(a))
            }
        }
    }
}

impl std::fmt::Display for IdentifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for IdentifyError {}

impl From<GgufError> for IdentifyError {
    fn from(e: GgufError) -> Self {
        IdentifyError::Gguf(e)
    }
}

#[derive(Debug, Clone)]
pub struct Identified {
    pub shape: Shape,
    pub architecture: Option<Vec<u8>>,
    pub split_count: u32,
    pub n_kv: u64,
    pub n_tensors: u64,
    pub alignment: u64,
    pub version: u32,
}

impl Identified {
    pub fn report_line(&self, path: &str) -> String {
        let arch = self
            .architecture
            .as_deref()
            .map(|a| String::from_utf8_lossy(a).into_owned())
            .unwrap_or_else(|| "-".into());
        format!(
            "identify path={path} name={} family={} variant={} model_id={} arch={} n_layer={} n_embd={} n_vocab={} split_count={} n_kv={} n_tensors={} alignment={}",
            self.shape.name,
            self.shape.family as u32,
            self.shape.variant as u32,
            self.shape.model_id(),
            arch,
            self.shape.n_layer,
            self.shape.n_embd,
            self.shape.n_vocab,
            self.split_count,
            self.n_kv,
            self.n_tensors,
            self.alignment,
        )
    }

    pub fn identify_line(&self) -> String {
        format!(
            "IDENTIFY {} family={} variant={}",
            self.shape.name, self.shape.family as u32, self.shape.variant as u32
        )
    }
}

const DS_KEYS: [(&str, fn(&mut DeepSeekDims, u32)); 22] = [
    ("deepseek4.block_count", |d, v| d.n_layer = v),
    ("deepseek4.embedding_length", |d, v| d.n_embd = v),
    ("deepseek4.vocab_size", |d, v| d.n_vocab = v),
    ("deepseek4.attention.head_count", |d, v| d.n_head = v),
    ("deepseek4.attention.head_count_kv", |d, v| d.n_head_kv = v),
    ("deepseek4.attention.key_length", |d, v| d.n_head_dim = v),
    ("deepseek4.attention.value_length", |d, v| d.n_value_dim = v),
    ("deepseek4.rope.dimension_count", |d, v| d.n_rot = v),
    ("deepseek4.attention.q_lora_rank", |d, v| d.n_lora_q = v),
    ("deepseek4.attention.output_lora_rank", |d, v| {
        d.n_lora_o = v
    }),
    ("deepseek4.attention.output_group_count", |d, v| {
        d.n_out_group = v
    }),
    ("deepseek4.expert_count", |d, v| d.n_expert = v),
    ("deepseek4.expert_used_count", |d, v| d.n_expert_used = v),
    ("deepseek4.expert_feed_forward_length", |d, v| {
        d.n_ff_exp = v
    }),
    ("deepseek4.expert_shared_count", |d, v| {
        d.n_expert_shared = v
    }),
    ("deepseek4.hash_layer_count", |d, v| d.n_hash_layer = v),
    ("deepseek4.attention.sliding_window", |d, v| d.n_swa = v),
    ("deepseek4.attention.indexer.head_count", |d, v| {
        d.n_indexer_head = v
    }),
    ("deepseek4.attention.indexer.key_length", |d, v| {
        d.n_indexer_head_dim = v
    }),
    ("deepseek4.attention.indexer.top_k", |d, v| {
        d.n_indexer_top_k = v
    }),
    ("deepseek4.hyper_connection.count", |d, v| d.n_hc = v),
    ("deepseek4.hyper_connection.sinkhorn_iterations", |d, v| {
        d.n_hc_sinkhorn_iter = v
    }),
];

fn deepseek_dims(g: &GgufFile) -> Result<DeepSeekDims, IdentifyError> {
    let mut d = DeepSeekDims {
        n_layer: 0,
        n_embd: 0,
        n_vocab: 0,
        n_head: 0,
        n_head_kv: 0,
        n_head_dim: 0,
        n_value_dim: 0,
        n_rot: 0,
        n_lora_q: 0,
        n_lora_o: 0,
        n_out_group: 0,
        n_expert: 0,
        n_expert_used: 0,
        n_ff_exp: 0,
        n_expert_shared: 0,
        n_hash_layer: 0,
        n_swa: 0,
        n_indexer_head: 0,
        n_indexer_head_dim: 0,
        n_indexer_top_k: 0,
        n_hc: 0,
        n_hc_sinkhorn_iter: 0,
    };
    for (key, set) in DS_KEYS {
        let v = g.get_u32(key).ok_or(IdentifyError::MissingKey(key))?;
        set(&mut d, v);
    }
    Ok(d)
}

pub fn identify_file(g: &GgufFile) -> Result<Identified, IdentifyError> {
    let architecture = g.get_string("general.architecture").map(|s| s.to_vec());
    let shape = match route_architecture(architecture.as_deref()) {
        ArchRoute::DeepSeekSelect => {
            let dims = deepseek_dims(g)?;
            select_shape_from_metadata(&dims).ok_or(IdentifyError::UnsupportedShape)?
        }
        ArchRoute::Fixed(v) => shape_for_variant(v),
        ArchRoute::Unsupported => {
            return Err(IdentifyError::UnsupportedArch(
                architecture.unwrap_or_default(),
            ));
        }
    };
    if shape.family == crate::ModelFamily::Glm53Vq
        && !matches!(g.get_u32("glm_vq.format_version"), Some(1 | 2))
    {
        return Err(IdentifyError::MissingKey("glm_vq.format_version=1|2"));
    }
    Ok(Identified {
        shape,
        architecture,
        split_count: g.split_count(),
        n_kv: g.n_kv,
        n_tensors: g.n_tensors,
        alignment: g.alignment,
        version: g.version,
    })
}

pub fn identify_gguf(path: &Path) -> Result<Identified, IdentifyError> {
    if path.is_dir() {
        if crate::beam_vq::is_release(path) {
            return crate::beam_vq::identify(path);
        }
        if crate::minimax_vq::is_release(path) {
            return crate::minimax_vq::identify(path);
        }
        if crate::mimo2_vq::is_release(path) {
            return crate::mimo2_vq::identify(path);
        }
        if crate::axk2_vq::is_release(path) {
            return crate::axk2_vq::identify(path);
        }
        if crate::ling31_vq::is_release(path) {
            return crate::ling31_vq::identify(path);
        }
        return crate::glm_vq::identify(path);
    }
    let g = GgufFile::open(path)?;
    identify_file(&g)
}

/// C `catalog_c_oracle parse` stdout.
pub fn dump_parse(path: &Path) -> String {
    match GgufFile::open(path) {
        Ok(g) => {
            let mut out = g.dump_header_kv();
            match identify_file(&g) {
                Ok(id) => {
                    out.push_str(&id.identify_line());
                    out.push('\n');
                }
                Err(IdentifyError::UnsupportedShape) => out.push_str("IDENTIFY unsupported\n"),
                Err(e) => {
                    out.push_str("ERROR ");
                    out.push_str(&e.token());
                    out.push('\n');
                }
            }
            out
        }
        Err(e) => format!("ERROR {}\n", e.token()),
    }
}
