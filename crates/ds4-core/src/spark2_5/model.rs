//! The release keeps the checkpoint's JSON configuration beside its parts.
//! A matching architecture string alone is not this contract: the subset below
//! pins the topology the layout was transcribed from, so a checkpoint with a
//! different layer count, head split, rope schedule or vocabulary is refused
//! before any weight is bound.
use std::path::Path;

use serde_json::Value;

use super::release;
use crate::{identify_file, GgufFile, Identified, IdentifyError};

fn error(value: impl ToString) -> IdentifyError {
    IdentifyError::Artifact(value.to_string())
}

fn subset(actual: &Value, expected: &Value, path: &str) -> Result<(), IdentifyError> {
    if let Some(fields) = expected.as_object() {
        for (name, value) in fields {
            subset(&actual[name], value, &format!("{path}/{name}"))?;
        }
        return Ok(());
    }
    if actual != expected {
        return Err(error(format!("configuration mismatch at {path}")));
    }
    Ok(())
}

fn config(value: &Value) -> Result<(), IdentifyError> {
    // The trunk checkpoint's own config.json (the Spark-X2.6-4B warehouse
    // checkpoint the release was converted from).
    let contract: Value = serde_json::from_str(include_str!("config.json")).map_err(error)?;
    subset(value, &contract, "config.json")
}

pub(super) fn identify(root: &Path) -> Result<Identified, IdentifyError> {
    let bytes = release::asset(root, "config.json").map_err(error)?;
    let value: Value = serde_json::from_slice(&bytes).map_err(error)?;
    config(&value)?;
    let g = GgufFile::open(&release::first_part(root).map_err(error)?)?;
    let mut id = identify_file(&g)?;
    if id.shape.variant != crate::Variant::SparkX26_4bVq {
        return Err(error("configuration/part architecture mismatch"));
    }
    id.split_count = release::part_count(root).map_err(error)?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_contract_rejects_topology_changes() {
        let original: Value = serde_json::from_str(include_str!("config.json")).unwrap();
        config(&original).unwrap();
        // This checkpoint has no `text_config` level: every path is top level.
        for pointer in [
            "/hidden_size",
            "/num_hidden_layers",
            "/num_attention_heads",
            "/num_key_value_heads",
            "/head_dim",
            "/intermediate_size",
            "/sliding_window",
            "/vocab_size",
            "/max_position_embeddings",
            "/layer_types/3",
            "/rope_parameters/full_attention/partial_rotary_factor",
            "/rope_parameters/full_attention/rope_theta",
            "/rope_parameters/sliding_attention/rope_theta",
            "/attention_bias",
            "/headwise_attn_output_gate",
            "/tie_word_embeddings",
        ] {
            let mut bad = original.clone();
            let slot = bad
                .pointer_mut(pointer)
                .unwrap_or_else(|| panic!("{pointer}"));
            *slot = json!(-1);
            assert!(config(&bad).is_err(), "accepted {pointer}");
        }
        // The architecture string is part of the topology too: a checkpoint
        // for another model cannot pass by keeping the numeric fields.
        let mut renamed = original.clone();
        renamed["model_type"] = json!("spark2_6");
        assert!(config(&renamed).is_err());
    }

    /// The layer schedule is what makes the geometry dual: three sliding
    /// layers then one full, nine times. A checkpoint that keeps the counts
    /// but moves a full layer is still a different model.
    #[test]
    fn the_release_schedule_is_three_sliding_then_one_full() {
        let config: Value = serde_json::from_str(include_str!("config.json")).unwrap();
        let types = config["layer_types"].as_array().expect("layer_types");
        assert_eq!(types.len(), 36);
        let full: Vec<usize> = types
            .iter()
            .enumerate()
            .filter(|(_, t)| t.as_str() == Some("full_attention"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full, [3, 7, 11, 15, 19, 23, 27, 31, 35]);
        assert_eq!(
            types
                .iter()
                .filter(|t| t.as_str() == Some("sliding_attention"))
                .count(),
            27
        );
    }
}
