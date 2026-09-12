//! Metadata-only preflight; never allocates GPU weights or starts inference.
use ds4_core::Step37Plan;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: step37_inspect FIRST_MQ83_SHARD")?;
    let plan = Step37Plan::inspect(Path::new(&path))?;
    println!(
        "Step 3.7 MQ83: {} layers, {} tensors, {} payload bytes; native execution pending",
        plan.layers().len(),
        plan.bindings().count(),
        plan.payload_bytes()
    );
    Ok(())
}
