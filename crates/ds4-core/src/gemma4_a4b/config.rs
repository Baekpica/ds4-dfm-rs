//! The checkpoint's own configuration, as the family's shape contract.
//!
//! `config.json` beside this file is the reference checkpoint's configuration
//! copied verbatim (sha256
//! `ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3`), not a
//! digest of a summary. The fields below are the ones the artifact layout and
//! a forward need; every other key stays in the file and is not reinterpreted.
//!
//! Two attention geometries are the family's defining trait and are kept per
//! layer type rather than collapsed into one: 25 sliding layers at head width
//! 256 with 8 KV heads and theta 1e4, and 5 full layers at head width 512 with
//! 2 KV heads, theta 1e6 and a quarter-width rotary. `attention_k_eq_v` makes
//! the full layers carry no value projection.

use std::path::Path;

use serde_json::Value;

use super::Error;

/// Which attention a layer runs. The checkpoint names them in `layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

impl LayerType {
    fn from_name(name: &str) -> Option<LayerType> {
        match name {
            "sliding_attention" => Some(LayerType::Sliding),
            "full_attention" => Some(LayerType::Full),
            _ => None,
        }
    }
}

/// The configuration values the layout and the forward bind.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub max_context: usize,
    pub layer_types: Vec<LayerType>,
    pub heads: usize,
    pub kv_heads: usize,
    pub global_kv_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub sliding_window: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_ff: usize,
    pub dense_ff: usize,
    pub rms_eps: f32,
    pub logit_softcap: f32,
    pub k_eq_v: bool,
    pub tie_embeddings: bool,
    pub sliding_theta: f64,
    pub full_theta: f64,
    pub partial_rotary_factor: f64,
}

impl Config {
    /// The embedded copy of the checkpoint's configuration.
    pub fn embedded() -> Result<Config, Error> {
        let value: Value = serde_json::from_str(include_str!("config.json"))
            .map_err(|e| Error::new(format!("config.json: {e}")))?;
        Config::from_value(&value)
    }

    /// A configuration file in an artifact directory, for a future artifact
    /// that carries its own copy. It must state the same topology.
    pub fn open(path: &Path) -> Result<Config, Error> {
        let bytes =
            std::fs::read(path).map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        Config::from_value(&value)
    }

    pub fn from_value(value: &Value) -> Result<Config, Error> {
        let text = value
            .get("text_config")
            .ok_or_else(|| Error::new("config.json has no text_config"))?;
        let number = |key: &str| -> Result<u64, Error> {
            text.get(key).and_then(Value::as_u64).ok_or_else(|| {
                Error::new(format!("config.json text_config.{key} is not an integer"))
            })
        };
        let flag = |key: &str| -> Result<bool, Error> {
            text.get(key).and_then(Value::as_bool).ok_or_else(|| {
                Error::new(format!("config.json text_config.{key} is not a boolean"))
            })
        };
        let real = |table: &Value, key: &str| -> Result<f64, Error> {
            table.get(key).and_then(Value::as_f64).ok_or_else(|| {
                Error::new(format!("config.json rope_parameters.{key} is not a number"))
            })
        };

        let layers = number("num_hidden_layers")? as usize;
        let names = text
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("config.json text_config.layer_types is not an array"))?;
        if names.len() != layers {
            return Err(Error::new(format!(
                "config.json lists {} layer types for {layers} layers",
                names.len()
            )));
        }
        let mut layer_types = Vec::with_capacity(layers);
        for (i, name) in names.iter().enumerate() {
            let name = name
                .as_str()
                .ok_or_else(|| Error::new(format!("layer_types[{i}] is not a string")))?;
            layer_types.push(
                LayerType::from_name(name)
                    .ok_or_else(|| Error::new(format!("layer_types[{i}] is {name:?}")))?,
            );
        }

        let rope = text
            .get("rope_parameters")
            .ok_or_else(|| Error::new("config.json has no rope_parameters"))?;
        let sliding = rope
            .get("sliding_attention")
            .ok_or_else(|| Error::new("config.json has no sliding_attention rope"))?;
        let full = rope
            .get("full_attention")
            .ok_or_else(|| Error::new("config.json has no full_attention rope"))?;

        let config = Config {
            hidden: number("hidden_size")? as usize,
            layers,
            vocab: number("vocab_size")? as usize,
            max_context: number("max_position_embeddings")? as usize,
            layer_types,
            heads: number("num_attention_heads")? as usize,
            kv_heads: number("num_key_value_heads")? as usize,
            global_kv_heads: number("num_global_key_value_heads")? as usize,
            head_dim: number("head_dim")? as usize,
            global_head_dim: number("global_head_dim")? as usize,
            sliding_window: number("sliding_window")? as usize,
            experts: number("num_experts")? as usize,
            top_k: number("top_k_experts")? as usize,
            moe_ff: number("moe_intermediate_size")? as usize,
            dense_ff: number("intermediate_size")? as usize,
            rms_eps: text
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .ok_or_else(|| Error::new("config.json has no rms_norm_eps"))?
                as f32,
            logit_softcap: text
                .get("final_logit_softcapping")
                .and_then(Value::as_f64)
                .ok_or_else(|| Error::new("config.json has no final_logit_softcapping"))?
                as f32,
            k_eq_v: flag("attention_k_eq_v")?,
            tie_embeddings: flag("tie_word_embeddings")?,
            sliding_theta: real(sliding, "rope_theta")?,
            full_theta: real(full, "rope_theta")?,
            partial_rotary_factor: real(full, "partial_rotary_factor")?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Error> {
        if self.layers == 0 || self.hidden == 0 || self.vocab == 0 {
            return Err(Error::new("config.json states an empty model"));
        }
        if self.heads == 0 || self.kv_heads == 0 || self.global_kv_heads == 0 {
            return Err(Error::new("config.json states zero attention heads"));
        }
        if self.heads % self.kv_heads != 0 || self.heads % self.global_kv_heads != 0 {
            return Err(Error::new(
                "config.json's KV head count does not divide the query head count",
            ));
        }
        if self.top_k == 0 || self.top_k > self.experts {
            return Err(Error::new(format!(
                "config.json routes {} of {} experts",
                self.top_k, self.experts
            )));
        }
        if !self.tie_embeddings {
            return Err(Error::new(
                "this family's artifact has no lm_head; tie_word_embeddings must be true",
            ));
        }
        if !(0.0..=1.0).contains(&self.partial_rotary_factor) {
            return Err(Error::new("partial_rotary_factor is outside 0..=1"));
        }
        let full = self
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::Full)
            .count();
        if full == 0 {
            return Err(Error::new(
                "config.json has no full_attention layer, so the full geometry is unbound",
            ));
        }
        Ok(())
    }

    pub fn layer_type(&self, layer: usize) -> LayerType {
        self.layer_types[layer]
    }

    pub fn head_dim_of(&self, kind: LayerType) -> usize {
        match kind {
            LayerType::Sliding => self.head_dim,
            LayerType::Full => self.global_head_dim,
        }
    }

    pub fn kv_heads_of(&self, kind: LayerType) -> usize {
        match kind {
            LayerType::Sliding => self.kv_heads,
            LayerType::Full => self.global_kv_heads,
        }
    }

    /// The width the rotary actually rotates. Sliding layers rotate the whole
    /// head; full layers rotate `partial_rotary_factor` of it (rounded down to
    /// a whole number of pairs, as the reference parameter does).
    pub fn rotated_dims(&self, kind: LayerType) -> usize {
        let width = self.head_dim_of(kind);
        match kind {
            LayerType::Sliding => width,
            LayerType::Full => {
                let pairs = (width as f64 * self.partial_rotary_factor / 2.0) as usize;
                pairs * 2
            }
        }
    }

    pub fn rope_theta(&self, kind: LayerType) -> f64 {
        match kind {
            LayerType::Sliding => self.sliding_theta,
            LayerType::Full => self.full_theta,
        }
    }

    /// `attention_k_eq_v` derives the value from the key projection on full
    /// layers, so those layers carry no `v_proj`.
    pub fn has_v_proj(&self, kind: LayerType) -> bool {
        !(self.k_eq_v && kind == LayerType::Full)
    }

    /// The query width a layer's `q_proj` states.
    pub fn q_rows(&self, kind: LayerType) -> usize {
        self.heads * self.head_dim_of(kind)
    }

    /// The key (and value, where present) width a layer's projections state.
    pub fn kv_rows(&self, kind: LayerType) -> usize {
        self.kv_heads_of(kind) * self.head_dim_of(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value() -> Value {
        serde_json::from_str(include_str!("config.json")).unwrap()
    }

    #[test]
    fn the_embedded_config_states_the_two_geometries() {
        let config = Config::embedded().unwrap();
        assert_eq!(config.layers, 30);
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::Sliding)
                .count(),
            25
        );
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::Full)
                .count(),
            5
        );
        assert_eq!(config.q_rows(LayerType::Sliding), 4096);
        assert_eq!(config.q_rows(LayerType::Full), 8192);
        assert_eq!(config.kv_rows(LayerType::Sliding), 2048);
        assert_eq!(config.kv_rows(LayerType::Full), 1024);
        assert_eq!(config.rotated_dims(LayerType::Sliding), 256);
        assert_eq!(config.rotated_dims(LayerType::Full), 128);
        assert!(config.has_v_proj(LayerType::Sliding));
        assert!(!config.has_v_proj(LayerType::Full));
    }

    #[test]
    fn a_layer_type_list_of_the_wrong_length_is_refused() {
        let mut value = value();
        value["text_config"]["layer_types"]
            .as_array_mut()
            .unwrap()
            .pop();
        let error = Config::from_value(&value).unwrap_err().to_string();
        assert!(error.contains("29 layer types for 30 layers"), "{error}");
    }

    #[test]
    fn an_untied_artifact_is_refused() {
        let mut value = value();
        value["text_config"]["tie_word_embeddings"] = Value::Bool(false);
        let error = Config::from_value(&value).unwrap_err().to_string();
        assert!(error.contains("lm_head"), "{error}");
    }
}
