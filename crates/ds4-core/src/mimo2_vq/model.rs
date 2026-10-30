//! The release keeps the original JSON configuration beside its named GGUF
//! parts. A matching architecture string alone is not this contract.
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
    // Source config of both releases (sha256 61bea4a0...). This admits one
    // topology; it does not transfer any other artifact's qualification.
    let contract: Value = serde_json::from_str(include_str!("config.json")).map_err(error)?;
    subset(value, &contract, "config.json")
}

pub(super) fn identify(root: &Path) -> Result<Identified, IdentifyError> {
    let bytes = release::asset(root, "config.json").map_err(error)?;
    let value: Value = serde_json::from_slice(&bytes).map_err(error)?;
    config(&value)?;
    let g = GgufFile::open(&release::first_part(root).map_err(error)?)?;
    let mut id = identify_file(&g)?;
    if id.shape.variant != crate::Variant::Mimo26FlashVq {
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
            "/hidden_size",
            "/num_hidden_layers",
            "/num_key_value_heads",
            "/swa_num_key_value_heads",
            "/n_routed_experts",
            "/num_experts_per_tok",
            "/sliding_window",
            "/attention_value_scale",
            "/rope_theta",
            "/max_position_embeddings",
            "/hybrid_layer_pattern/5",
            "/vision_config/depth",
            "/audio_config/input_local_layers",
        ] {
            let mut bad = original.clone();
            *bad.pointer_mut(pointer).unwrap() = json!(-1);
            assert!(config(&bad).is_err(), "accepted {pointer}");
        }
    }
}
