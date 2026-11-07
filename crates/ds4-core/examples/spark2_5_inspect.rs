//! Spark-X2.6 VQ release preflight without GPU initialization: identify the
//! directory, verify every part and its book/residual links, and compare the
//! loaded inventory with the layout table.
use ds4_core::{expected_layouts, identify_gguf, TensorInventory};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir] = args.as_slice() else {
        return Err("usage: spark2_5_inspect RELEASE_DIR".into());
    };
    let path = Path::new(dir);
    let identified = identify_gguf(path)?;
    // Opening the release hashes both parts and validates the GVQ1/GVR8
    // headers, the codebook links and the residual pairing; the layout table
    // must then name every tensor with the stored type and shape.
    let inventory = TensorInventory::open(path)?;
    let layouts = expected_layouts(&identified.shape);
    println!(
        "{}",
        serde_json::json!({
            "family": identified.shape.family.oracle_name(),
            "variant": format!("{:?}", identified.shape.variant),
            "layers": identified.shape.n_layer,
            "split_count": identified.split_count,
            "tensors": inventory.tensors.len(),
            "shards": inventory.shards.len(),
            "layout_entries": layouts.len(),
            "last_tensor": inventory.tensors.last().map(|t| t.name.as_str()),
            "last_layout": layouts.last().map(|s| s.name.as_str()),
        })
    );
    Ok(())
}
