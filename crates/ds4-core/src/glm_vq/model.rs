//! GVQ1 retains the original JSON configuration beside independently named
//! GGUF parts. A matching architecture string alone is not this contract.
use std::path::Path;

use serde_json::{json, Value};

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
    // Source config at 91b69215804f2d2036a8867f6ce44848e278dcda. This
    // admits one topology; it does not transfer MXQ arithmetic qualification.
    let contract: Value = serde_json::from_str(include_str!("config.json")).map_err(error)?;
    subset(value, &contract, "config.json")?;
    let layers: Vec<_> = (0..45)
        .map(|i| {
            if i % 4 == 3 {
                "deepseek_sparse_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let kda: Vec<_> = (0..45).filter(|i| i % 4 != 3).collect();
    let dsa: Vec<_> = (0..45).filter(|i| i % 4 == 3).collect();
    subset(
        &value["text_config"],
        &json!({
            "layer_types":layers, "indexer_types":vec!["full";45],
            "linear_attn_config":{"kda_layers":kda,"full_attn_layers":dsa}
        }),
        "config.json/text_config",
    )
}

pub(super) fn identify(root: &Path) -> Result<Identified, IdentifyError> {
    let bytes = release::asset(root, "config.json").map_err(error)?;
    let value: Value = serde_json::from_slice(&bytes).map_err(error)?;
    config(&value)?;
    let (path, count) = release::first_part(root).map_err(error)?;
    let g = GgufFile::open(&path)?;
    let mut id = identify_file(&g)?;
    if id.shape.family != crate::ModelFamily::Glm53Vq {
        return Err(error("configuration/part architecture mismatch"));
    }
    id.split_count = count;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the pinned local JSON assets, no GPU or tensor hashing"]
    fn pinned_config_contract() {
        let root = std::env::var_os("DS4_GLM_VQ_MODEL").expect("DS4_GLM_VQ_MODEL");
        let root = Path::new(&root);
        let id = identify(root).unwrap();
        assert_eq!(id.shape.name, "GLM-5.3-Flash-VQ");
        assert_eq!(id.shape.family, crate::ModelFamily::Glm53Vq);
        assert_eq!(id.split_count, 56);
        let original: Value =
            serde_json::from_slice(&release::asset(root, "config.json").unwrap()).unwrap();
        for pointer in [
            "/text_config/hidden_size",
            "/text_config/swiglu_limit",
            "/text_config/hc_sinkhorn_iters",
            "/text_config/linear_attn_config/gate_lower_bound",
            "/text_config/layer_types/0",
            "/text_config/n_routed_experts",
            "/vision_config/temporal_patch_size",
            "/quantization_config/weight_block_size/0",
        ] {
            let mut bad = original.clone();
            *bad.pointer_mut(pointer).unwrap() = json!(-1);
            assert!(config(&bad).is_err(), "accepted {pointer}");
        }
    }

    #[test]
    #[ignore = "requires the pinned manifest only; no tensor payload scan"]
    fn pinned_layout_contract() {
        let root = std::env::var_os("DS4_GLM_VQ_MODEL").expect("DS4_GLM_VQ_MODEL");
        let bytes = release::asset(Path::new(&root), "tensor-manifest.json").unwrap();
        let manifest: Value = serde_json::from_slice(&bytes).unwrap();
        let rows = manifest["tensors"].as_array().unwrap();
        let tensors = rows
            .iter()
            .map(|row| {
                let shape = row["stored_shape"].as_array().unwrap();
                let mut dim = [0; 8];
                for (i, n) in shape.iter().rev().enumerate() {
                    dim[i] = n.as_u64().unwrap();
                }
                crate::TensorInfo {
                    name: row["name"].as_str().unwrap().into(),
                    ndim: shape.len() as u32,
                    dim,
                    typ: row["type_id"].as_u64().unwrap() as u32,
                    bytes: row["payload_bytes"].as_u64().unwrap(),
                    elements: dim[..shape.len()].iter().product(),
                    rel_offset: 0,
                    abs_offset: 0,
                    shard: 0,
                }
            })
            .collect();
        let mut inventory = crate::TensorInventory {
            tensors,
            shards: Vec::new(),
            data_pos: 0,
            alignment: 128,
            page: 4096,
        };
        super::super::validate_inventory(&inventory).unwrap();
        assert_eq!(inventory.tensors.len(), 39862);
        inventory.tensors[0].typ = 0;
        assert!(super::super::validate_inventory(&inventory).is_err());
        println!("GVQ1_LAYOUT_PASS tensors=39862");
    }
}
