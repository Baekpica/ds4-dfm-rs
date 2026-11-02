//! Teacher-forced NLL after a long prefill: gates prefill and decode value
//! changes on many positions instead of one chaotic frontier.
//! Usage: glm53_nll MODEL TEXT_FILE OUT_JSON PREFIX_TOKENS TARGET_TOKENS [CONTEXT]
use ds4_core::{Backend, Model, TokenBuffer};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=6).contains(&args.len()) {
        return Err("usage: glm53_nll MODEL TEXT_FILE OUT_JSON PREFIX_TOKENS TARGET_TOKENS [CONTEXT]".into());
    }
    let prefix: usize = args[3].parse()?;
    let targets: usize = args[4].parse()?;
    let context: i32 = match args.get(5) {
        Some(value) => value.parse()?,
        None => (prefix + targets + 1) as i32,
    };
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    let text = std::fs::read(&args[1])?;
    let tokens = model.vocab().encode_bytes(&text);
    if prefix == 0 || tokens.len() < prefix + targets || prefix + targets >= context as usize {
        return Err("text or context too short for prefix + targets".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let mut session = model.session(context)?;
    session.sync(&TokenBuffer::from_tokens(tokens[..prefix].to_vec()))?;

    // Each target is scored from the state its true prefix produced.
    let (mut nll, mut top1) = (Vec::with_capacity(targets), Vec::with_capacity(targets));
    for step in 0..targets {
        let next = tokens[prefix + step];
        let logits = session.copy_logits(vocab)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(format!("nonfinite logits at target {step}").into());
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
        nll.push(max + sum.ln() - logits[next as usize] as f64);
        let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap_or(0);
        top1.push(argmax as i32 == next);
        session.eval(next)?;
    }
    let mean = nll.iter().sum::<f64>() / targets as f64;
    let hits = top1.iter().filter(|&&hit| hit).count();
    std::fs::write(
        &args[2],
        serde_json::to_vec(&json!({"prefix": prefix, "targets": targets, "context": context,
            "mean_nll": mean, "top1": hits, "nll": nll, "top1_hits": top1}))?,
    )?;
    println!("prefix {prefix} targets {targets}: mean NLL {mean:.5}, top-1 {hits}/{targets}");
    Ok(())
}
