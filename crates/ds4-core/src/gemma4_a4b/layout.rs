//! The text tensor set this family binds, checked against the checkpoint's
//! own configuration.
//!
//! The names are the artifact's names, not a guess: the release of the
//! reference artifact holds 657 text entries, and every layer carries the same
//! 21 roles plus a `v_proj` only where `attention_k_eq_v` lets the full
//! attention derive the value from the key. The shape rules below are the
//! artifact's measured shapes expressed through the configuration:
//!
//! - `q_proj` is `heads * head_dim(kind)` by hidden; the sliding layers state
//!   4096 and the full layers 8192;
//! - `k_proj` is `kv_heads(kind) * head_dim(kind)` by hidden: 2048 sliding,
//!   1024 full;
//! - `experts.gate_up_proj` stacks 128 experts of gate and up at 704 rows
//!   each (180224 rows), and `experts.down_proj` stacks 128 experts of 2816
//!   rows by 704 columns (360448 rows);
//! - the norms, `router.scale`, `router.per_expert_scale` and `layer_scalar`
//!   are copied f32 sidecars, not containers.
//!
//! Vision tensors are not part of this artifact. A release that carries a
//! non-text part is refused here: the vision tower is a sidecar by the family
//! contract, and binding it as a text weight would be wrong.

use std::collections::BTreeMap;

use super::config::{Config, LayerType};
use super::release::Entry;
use super::Error;

/// One bindable role. The dense and routed paths both exist in every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Embed,
    FinalNorm,
    InputNorm,
    PostAttentionNorm,
    PreFfnNorm,
    PostFfnNorm,
    PreFfnNorm2,
    PostFfnNorm1,
    PostFfnNorm2,
    QProj,
    KProj,
    VProj,
    OProj,
    QNorm,
    KNorm,
    LayerScalar,
    DenseGate,
    DenseUp,
    DenseDown,
    RouterProj,
    RouterScale,
    RouterPerExpertScale,
    ExpertGateUp,
    ExpertDown,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Embed => "embed",
            Role::FinalNorm => "final_norm",
            Role::InputNorm => "input_norm",
            Role::PostAttentionNorm => "post_attention_norm",
            Role::PreFfnNorm => "pre_ffn_norm",
            Role::PostFfnNorm => "post_ffn_norm",
            Role::PreFfnNorm2 => "pre_ffn_norm_2",
            Role::PostFfnNorm1 => "post_ffn_norm_1",
            Role::PostFfnNorm2 => "post_ffn_norm_2",
            Role::QProj => "q_proj",
            Role::KProj => "k_proj",
            Role::VProj => "v_proj",
            Role::OProj => "o_proj",
            Role::QNorm => "q_norm",
            Role::KNorm => "k_norm",
            Role::LayerScalar => "layer_scalar",
            Role::DenseGate => "dense_gate",
            Role::DenseUp => "dense_up",
            Role::DenseDown => "dense_down",
            Role::RouterProj => "router_proj",
            Role::RouterScale => "router_scale",
            Role::RouterPerExpertScale => "router_per_expert_scale",
            Role::ExpertGateUp => "expert_gate_up",
            Role::ExpertDown => "expert_down",
        }
    }

    /// The role a tensor name states, or `None` when the name is not this
    /// family's.
    fn from_suffix(suffix: &str) -> Option<Role> {
        match suffix {
            "input_layernorm.weight" => Some(Role::InputNorm),
            "post_attention_layernorm.weight" => Some(Role::PostAttentionNorm),
            "pre_feedforward_layernorm.weight" => Some(Role::PreFfnNorm),
            "post_feedforward_layernorm.weight" => Some(Role::PostFfnNorm),
            "pre_feedforward_layernorm_2.weight" => Some(Role::PreFfnNorm2),
            "post_feedforward_layernorm_1.weight" => Some(Role::PostFfnNorm1),
            "post_feedforward_layernorm_2.weight" => Some(Role::PostFfnNorm2),
            "self_attn.q_proj.weight" => Some(Role::QProj),
            "self_attn.k_proj.weight" => Some(Role::KProj),
            "self_attn.v_proj.weight" => Some(Role::VProj),
            "self_attn.o_proj.weight" => Some(Role::OProj),
            "self_attn.q_norm.weight" => Some(Role::QNorm),
            "self_attn.k_norm.weight" => Some(Role::KNorm),
            "layer_scalar" => Some(Role::LayerScalar),
            "mlp.gate_proj.weight" => Some(Role::DenseGate),
            "mlp.up_proj.weight" => Some(Role::DenseUp),
            "mlp.down_proj.weight" => Some(Role::DenseDown),
            "router.proj.weight" => Some(Role::RouterProj),
            "router.scale" => Some(Role::RouterScale),
            "router.per_expert_scale" => Some(Role::RouterPerExpertScale),
            "experts.gate_up_proj" => Some(Role::ExpertGateUp),
            "experts.down_proj" => Some(Role::ExpertDown),
            _ => None,
        }
    }

    /// The roles every layer must state.
    const PER_LAYER: [Role; 21] = [
        Role::InputNorm,
        Role::PostAttentionNorm,
        Role::PreFfnNorm,
        Role::PostFfnNorm,
        Role::PreFfnNorm2,
        Role::PostFfnNorm1,
        Role::PostFfnNorm2,
        Role::QProj,
        Role::KProj,
        Role::OProj,
        Role::QNorm,
        Role::KNorm,
        Role::LayerScalar,
        Role::DenseGate,
        Role::DenseUp,
        Role::DenseDown,
        Role::RouterProj,
        Role::RouterScale,
        Role::RouterPerExpertScale,
        Role::ExpertGateUp,
        Role::ExpertDown,
    ];
}

/// One layer's bound tensors.
#[derive(Clone, Debug)]
pub struct Layer {
    pub index: usize,
    pub kind: LayerType,
    pub roles: BTreeMap<Role, Entry>,
}

impl Layer {
    pub fn get(&self, role: Role) -> Result<&Entry, Error> {
        self.roles.get(&role).ok_or_else(|| {
            Error::new(format!(
                "layer {}: no {} (this layer is {:?})",
                self.index,
                role.as_str(),
                self.kind
            ))
        })
    }
}

/// The whole text path: the embedding, the final norm and every layer.
#[derive(Clone, Debug)]
pub struct Layout {
    pub embed: Entry,
    pub final_norm: Entry,
    pub layers: Vec<Layer>,
    pub other_parts: Vec<String>,
}

const PREFIX: &str = "model.language_model.";

/// Classify and check every release entry against the configuration.
pub fn check(entries: &[Entry], config: &Config) -> Result<Layout, Error> {
    let mut embed = None;
    let mut final_norm = None;
    let mut layers: Vec<BTreeMap<Role, Entry>> =
        (0..config.layers).map(|_| BTreeMap::new()).collect();
    let mut other_parts = Vec::new();

    for entry in entries {
        if entry.part != "text" {
            /* A sidecar part is named, not silently ignored: the family's
             * contract puts vision outside the text artifact. */
            other_parts.push(format!("{}:{}", entry.part, entry.name));
            continue;
        }
        let rest = entry
            .name
            .strip_prefix(PREFIX)
            .ok_or_else(|| Error::new(format!("{}: not a text tensor name", entry.name)))?;
        if rest == "embed_tokens.weight" {
            slot(&mut embed, entry)?;
            check_shape(entry, Role::Embed, config.vocab, config.hidden)?;
            continue;
        }
        if rest == "norm.weight" {
            slot(&mut final_norm, entry)?;
            check_shape(entry, Role::FinalNorm, config.hidden, 1)?;
            continue;
        }
        let rest = rest
            .strip_prefix("layers.")
            .ok_or_else(|| Error::new(format!("{}: unknown text tensor", entry.name)))?;
        let (index, suffix) = rest
            .split_once('.')
            .ok_or_else(|| Error::new(format!("{}: malformed layer name", entry.name)))?;
        let index: usize = index
            .parse()
            .map_err(|_| Error::new(format!("{}: malformed layer index", entry.name)))?;
        if index >= config.layers {
            return Err(Error::new(format!(
                "{}: layer {index} is outside 0..{}",
                entry.name, config.layers
            )));
        }
        let role = Role::from_suffix(suffix)
            .ok_or_else(|| Error::new(format!("{}: unknown layer tensor", entry.name)))?;
        if layers[index].insert(role, entry.clone()).is_some() {
            return Err(Error::new(format!(
                "{}: layer {index} states {} twice",
                entry.name,
                role.as_str()
            )));
        }
        let kind = config.layer_type(index);
        check_role_shape(entry, role, kind, config)?;
    }

    let embed = embed.ok_or_else(|| Error::new("release: no embed_tokens.weight"))?;
    let final_norm = final_norm.ok_or_else(|| Error::new("release: no model.norm.weight"))?;
    let mut out = Vec::with_capacity(config.layers);
    for (index, roles) in layers.into_iter().enumerate() {
        let kind = config.layer_type(index);
        for role in Role::PER_LAYER {
            if !roles.contains_key(&role) {
                return Err(Error::new(format!(
                    "layer {index}: missing {}",
                    role.as_str()
                )));
            }
        }
        let wants_v = config.has_v_proj(kind);
        match (wants_v, roles.contains_key(&Role::VProj)) {
            (true, false) => {
                return Err(Error::new(format!(
                    "layer {index}: {kind:?} attention needs a v_proj"
                )))
            }
            (false, true) => {
                return Err(Error::new(format!(
                    "layer {index}: {kind:?} attention uses k as v and must not carry a v_proj"
                )))
            }
            _ => {}
        }
        out.push(Layer { index, kind, roles });
    }

    /* The text set is closed: embed + final norm + one v_proj per sliding
     * layer + 21 roles per layer. A stray tensor would otherwise be invisible
     * to a binder that looks names up by role. */
    let expected = 2
        + config.layers * Role::PER_LAYER.len()
        + config
            .layer_types
            .iter()
            .filter(|kind| config.has_v_proj(**kind))
            .count();
    let observed = 2 + out.iter().map(|l| l.roles.len()).sum::<usize>();
    if observed != entries.iter().filter(|e| e.part == "text").count() {
        return Err(Error::new(format!(
            "release: {observed} text roles bound from {} text entries",
            entries.iter().filter(|e| e.part == "text").count()
        )));
    }
    if observed != expected {
        return Err(Error::new(format!(
            "release: {observed} text roles bound, the configuration implies {expected}"
        )));
    }

    Ok(Layout {
        embed,
        final_norm,
        layers: out,
        other_parts,
    })
}

fn slot(slot: &mut Option<Entry>, entry: &Entry) -> Result<(), Error> {
    if slot.is_some() {
        return Err(Error::new(format!("{}: stated twice", entry.name)));
    }
    *slot = Some(entry.clone());
    Ok(())
}

/// The shape a role states, from the configuration.
fn role_shape(role: Role, kind: LayerType, config: &Config) -> (usize, usize) {
    let q = config.q_rows(kind);
    let kv = config.kv_rows(kind);
    let head = config.head_dim_of(kind);
    match role {
        Role::Embed => (config.vocab, config.hidden),
        Role::FinalNorm
        | Role::InputNorm
        | Role::PostAttentionNorm
        | Role::PreFfnNorm
        | Role::PostFfnNorm
        | Role::PreFfnNorm2
        | Role::PostFfnNorm1
        | Role::PostFfnNorm2
        | Role::RouterScale => (config.hidden, 1),
        Role::QProj => (q, config.hidden),
        Role::KProj | Role::VProj => (kv, config.hidden),
        Role::OProj => (config.hidden, q),
        Role::QNorm | Role::KNorm => (head, 1),
        Role::LayerScalar => (1, 1),
        Role::DenseGate | Role::DenseUp => (config.dense_ff, config.hidden),
        Role::DenseDown => (config.hidden, config.dense_ff),
        Role::RouterProj => (config.experts, config.hidden),
        Role::RouterPerExpertScale => (config.experts, 1),
        Role::ExpertGateUp => (config.experts * 2 * config.moe_ff, config.hidden),
        Role::ExpertDown => (config.experts * config.hidden, config.moe_ff),
    }
}

/// Which roles the builder stored as containers, and which as copied f32.
fn role_encoded(role: Role) -> bool {
    !matches!(
        role,
        Role::FinalNorm
            | Role::InputNorm
            | Role::PostAttentionNorm
            | Role::PreFfnNorm
            | Role::PostFfnNorm
            | Role::PreFfnNorm2
            | Role::PostFfnNorm1
            | Role::PostFfnNorm2
            | Role::QNorm
            | Role::KNorm
            | Role::LayerScalar
            | Role::RouterScale
            | Role::RouterPerExpertScale
    )
}

fn check_role_shape(
    entry: &Entry,
    role: Role,
    kind: LayerType,
    config: &Config,
) -> Result<(), Error> {
    let (rows, cols) = role_shape(role, kind, config);
    check_shape(entry, role, rows, cols)?;
    if entry.encoded != role_encoded(role) {
        return Err(Error::new(format!(
            "{}: {} is {}, but this role is {}",
            entry.name,
            role.as_str(),
            if entry.encoded { "encoded" } else { "copied" },
            if role_encoded(role) {
                "encoded"
            } else {
                "copied"
            }
        )));
    }
    Ok(())
}

fn check_shape(entry: &Entry, role: Role, rows: usize, cols: usize) -> Result<(), Error> {
    if entry.rows != rows || entry.cols != cols {
        return Err(Error::new(format!(
            "{}: {} is {}x{}, the configuration implies {rows}x{cols}",
            entry.name,
            role.as_str(),
            entry.rows,
            entry.cols
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, role: &str, rows: usize, cols: usize, encoded: bool) -> Entry {
        Entry {
            name: name.to_string(),
            part: "text".to_string(),
            role: role.to_string(),
            encoded,
            rows,
            cols,
            dim: encoded.then_some(8),
            nc: encoded.then_some(4096),
            stages: encoded.then_some(2),
            groups: encoded.then_some(1),
            codebook: encoded.then_some(super::super::release::Codebook::F16),
            payload: None,
            bytes: 0,
            sha256: "00".repeat(32),
        }
    }

    fn build(config: &Config) -> Vec<Entry> {
        let mut out = vec![
            entry(
                "model.language_model.embed_tokens.weight",
                "embedding",
                config.vocab,
                config.hidden,
                true,
            ),
            entry(
                "model.language_model.norm.weight",
                "final_norm",
                config.hidden,
                1,
                false,
            ),
        ];
        for (index, kind) in config.layer_types.iter().enumerate() {
            for role in Role::PER_LAYER {
                let (rows, cols) = role_shape(role, *kind, config);
                out.push(entry(
                    &format!("model.language_model.layers.{index}.{}", suffix(role)),
                    role.as_str(),
                    rows,
                    cols,
                    role_encoded(role),
                ));
            }
            if config.has_v_proj(*kind) {
                let (rows, cols) = role_shape(Role::VProj, *kind, config);
                out.push(entry(
                    &format!("model.language_model.layers.{index}.self_attn.v_proj.weight"),
                    "v_proj",
                    rows,
                    cols,
                    true,
                ));
            }
        }
        out
    }

    fn suffix(role: Role) -> &'static str {
        match role {
            Role::InputNorm => "input_layernorm.weight",
            Role::PostAttentionNorm => "post_attention_layernorm.weight",
            Role::PreFfnNorm => "pre_feedforward_layernorm.weight",
            Role::PostFfnNorm => "post_feedforward_layernorm.weight",
            Role::PreFfnNorm2 => "pre_feedforward_layernorm_2.weight",
            Role::PostFfnNorm1 => "post_feedforward_layernorm_1.weight",
            Role::PostFfnNorm2 => "post_feedforward_layernorm_2.weight",
            Role::QProj => "self_attn.q_proj.weight",
            Role::KProj => "self_attn.k_proj.weight",
            Role::OProj => "self_attn.o_proj.weight",
            Role::QNorm => "self_attn.q_norm.weight",
            Role::KNorm => "self_attn.k_norm.weight",
            Role::LayerScalar => "layer_scalar",
            Role::DenseGate => "mlp.gate_proj.weight",
            Role::DenseUp => "mlp.up_proj.weight",
            Role::DenseDown => "mlp.down_proj.weight",
            Role::RouterProj => "router.proj.weight",
            Role::RouterScale => "router.scale",
            Role::RouterPerExpertScale => "router.per_expert_scale",
            Role::ExpertGateUp => "experts.gate_up_proj",
            Role::ExpertDown => "experts.down_proj",
            Role::Embed | Role::FinalNorm | Role::VProj => unreachable!(),
        }
    }

    #[test]
    fn the_artifact_shape_binds_every_role() {
        let config = Config::embedded().unwrap();
        let entries = build(&config);
        assert_eq!(entries.len(), 657);
        let layout = check(&entries, &config).unwrap();
        assert_eq!(layout.layers.len(), 30);
        assert_eq!(layout.layers[0].kind, LayerType::Sliding);
        assert_eq!(layout.layers[5].kind, LayerType::Full);
        assert!(layout.layers[5].roles.get(&Role::VProj).is_none());
        assert_eq!(
            layout.layers[5].get(Role::QProj).unwrap().rows,
            8192,
            "full attention's query width"
        );
        assert_eq!(
            layout.layers[0].get(Role::ExpertGateUp).unwrap().rows,
            180224
        );
        assert_eq!(layout.layers[0].get(Role::ExpertDown).unwrap().rows, 360448);
        assert!(layout.other_parts.is_empty());
    }

    #[test]
    fn a_missing_value_projection_on_a_sliding_layer_is_refused() {
        let config = Config::embedded().unwrap();
        let mut entries = build(&config);
        entries.retain(|e| !e.name.contains("layers.0.self_attn.v_proj"));
        let error = check(&entries, &config).unwrap_err().to_string();
        assert!(error.contains("layer 0"), "{error}");
        assert!(error.contains("needs a v_proj"), "{error}");
    }

    #[test]
    fn a_value_projection_on_a_full_layer_is_refused() {
        let config = Config::embedded().unwrap();
        let mut entries = build(&config);
        let mut stray = entry(
            "model.language_model.layers.5.self_attn.v_proj.weight",
            "v_proj",
            1024,
            config.hidden,
            true,
        );
        stray.part = "text".into();
        entries.push(stray);
        let error = check(&entries, &config).unwrap_err().to_string();
        assert!(error.contains("uses k as v"), "{error}");
    }

    #[test]
    fn an_unknown_text_tensor_is_refused() {
        let config = Config::embedded().unwrap();
        let mut entries = build(&config);
        entries.push(entry(
            "model.language_model.layers.0.self_attn.surprise.weight",
            "unknown",
            8,
            8,
            true,
        ));
        let error = check(&entries, &config).unwrap_err().to_string();
        assert!(error.contains("unknown layer tensor"), "{error}");
    }

    #[test]
    fn a_vision_part_is_named_not_bound() {
        let config = Config::embedded().unwrap();
        let mut entries = build(&config);
        let mut vision = entry(
            "model.vision_tower.layers.0.attn.q_proj.weight",
            "q_proj",
            8,
            8,
            true,
        );
        vision.part = "vision".into();
        entries.push(vision);
        let layout = check(&entries, &config).unwrap();
        assert_eq!(layout.other_parts.len(), 1);
        assert!(layout.other_parts[0].starts_with("vision:"));
    }

    #[test]
    fn a_wrong_full_attention_query_width_is_refused() {
        let config = Config::embedded().unwrap();
        let mut entries = build(&config);
        for e in &mut entries {
            if e.name == "model.language_model.layers.5.self_attn.q_proj.weight" {
                e.rows = 4096;
            }
        }
        let error = check(&entries, &config).unwrap_err().to_string();
        assert!(error.contains("layers.5.self_attn.q_proj"), "{error}");
        assert!(error.contains("8192"), "{error}");
    }
}
