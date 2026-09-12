//! Metadata-only preflight; never allocates GPU weights or starts inference.
use ds4_core::{Step37Plan, Step37Sidecar, Step37SidecarPlan};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: step37_inspect FIRST_MQ83_SHARD [MTP_Q8 VISION_F16]")?;
    let plan = Step37Plan::inspect(Path::new(&path))?;
    println!(
        "Step 3.7 MQ83: {} layers, {} tensors, {} payload bytes; native execution pending",
        plan.layers().len(),
        plan.bindings().count(),
        plan.payload_bytes()
    );
    for kind in [Step37Sidecar::Mtp, Step37Sidecar::Vision] {
        let Some(path) = args.next() else { break };
        let plan = Step37SidecarPlan::inspect(Path::new(&path), kind)?;
        println!(
            "{kind:?}: {} tensors, {} payload bytes",
            plan.bindings().count(),
            plan.payload_bytes()
        );
    }
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    Ok(())
}
