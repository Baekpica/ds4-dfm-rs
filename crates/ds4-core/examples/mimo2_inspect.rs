//! MiMo metadata/inventory preflight without GPU initialization.
use ds4_core::{expected_layouts, identify_gguf, GgufFile, Mimo2Plan};
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let path = Path::new(
        args.first()
            .ok_or("usage: mimo2_inspect FIRST_SHARD [--metadata-only]")?,
    );
    if args.len() > 2 || args.get(1).is_some_and(|x| x != "--metadata-only") {
        return Err("unexpected arguments".into());
    }
    let identified = identify_gguf(path)?;
    Mimo2Plan::check_metadata(&GgufFile::open(path)?)?;
    if args.len() == 2 {
        let layouts: Vec<_> = expected_layouts(&identified.shape).into_iter().map(|x| {
            serde_json::json!({"name": x.name, "type": x.class.token(), "shape": x.dim[..x.ndim as usize]})
        }).collect();
        println!(
            "{}",
            serde_json::json!({"metadata_valid": true, "family": identified.shape.family.oracle_name(), "expected_layouts": layouts})
        );
        return Ok(());
    }
    let plan = Mimo2Plan::inspect(path)?;
    println!(
        "{}",
        serde_json::json!({"trunk_layers": plan.layers().count(), "tensors": plan.bindings().count(), "payload_bytes": plan.payload_bytes()})
    );
    Ok(())
}
