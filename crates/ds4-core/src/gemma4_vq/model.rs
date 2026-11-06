//! The release keeps the checkpoint's JSON configuration beside its parts.
//! A matching architecture string alone is not this contract: the subset below
//! pins the topology the layout was transcribed from, so a checkpoint with a
//! different layer count, head split, expert count or rope schedule is refused
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
    // The trunk checkpoint's own config.json (`google/gemma-4-26B-A4B-it`).
    let contract: Value = serde_json::from_str(include_str!("config.json")).map_err(error)?;
    subset(value, &contract, "config.json")
}

pub(super) fn identify(root: &Path) -> Result<Identified, IdentifyError> {
    let bytes = release::asset(root, "config.json").map_err(error)?;
    let value: Value = serde_json::from_slice(&bytes).map_err(error)?;
    config(&value)?;
    let g = GgufFile::open(&release::first_part(root).map_err(error)?)?;
    let mut id = identify_file(&g)?;
    if id.shape.variant != crate::Variant::Gemma4A4bVq {
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
        for pointer in [
            "/text_config/hidden_size",
            "/text_config/num_hidden_layers",
            "/text_config/num_key_value_heads",
            "/text_config/num_global_key_value_heads",
            "/text_config/head_dim",
            "/text_config/global_head_dim",
            "/text_config/num_experts",
            "/text_config/top_k_experts",
            "/text_config/moe_intermediate_size",
            "/text_config/intermediate_size",
            "/text_config/sliding_window",
            "/text_config/vocab_size",
            "/text_config/layer_types/5",
            "/text_config/rope_parameters/full_attention/partial_rotary_factor",
            "/text_config/attention_k_eq_v",
            "/text_config/tie_word_embeddings",
        ] {
            let mut bad = original.clone();
            let slot = bad.pointer_mut(pointer).unwrap();
            *slot = json!(-1);
            assert!(config(&bad).is_err(), "accepted {pointer}");
        }
    }
}
