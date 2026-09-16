//! Metadata-only preflight; never allocates GPU weights or starts inference.
use ds4_core::{Ling3VlPlan, Ling3VlVisionPlan};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: ling3vl_inspect FIRST_MAIN_SHARD [MMPROJ_BF16]")?;
    let plan = Ling3VlPlan::inspect(Path::new(&path))?;
    let kda = plan.layers().iter().filter(|l| l.is_kda()).count();
    println!(
        "Ling-3.0-flash-VL: {} layers ({kda} KDA / {} MLA), {} tensors, {} payload bytes",
        plan.layers().len(),
        plan.layers().len() - kda,
        plan.bindings().count(),
        plan.payload_bytes()
    );
    if let Some(path) = args.next() {
        let vision = Ling3VlVisionPlan::inspect(Path::new(&path))?;
        println!(
            "mmproj: {} tensors, {} payload bytes",
            vision.bindings().count(),
            vision.payload_bytes()
        );
    }
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    Ok(())
}
