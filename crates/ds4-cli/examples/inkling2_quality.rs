//! Teacher-force pinned assistant tokens and record NLL/top-1 for paired
//! arithmetic comparisons. Acceptance uses a measured configuration noise
//! reference on the same host; this recorder imposes no fixed tolerance.
use ds4_core::{Backend, Model, ModelFamily, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;

fn score(logits: &[f32], next: usize) -> Result<(usize, f64), &'static str> {
    if next >= logits.len() || logits.iter().any(|x| !x.is_finite()) {
        return Err("invalid target or non-finite logits");
    }
    let mut top = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[top] {
            top = i;
        }
    }
    let peak = f64::from(logits[top]);
    let z: f64 = logits.iter().map(|&v| (f64::from(v) - peak).exp()).sum();
    Ok((top, peak + z.ln() - f64::from(logits[next])))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--self-test"] {
        let (top, nll) = score(&[0.0, 0.0], 1)?;
        assert_eq!(top, 0);
        assert!((nll - 2.0_f64.ln()).abs() < 1e-12);
        assert_eq!(score(&[-1000.0, 0.0], 0)?, (1, 1000.0));
        assert!(score(&[], 0).is_err());
        assert!(score(&[f32::NAN], 0).is_err());
        assert!(score(&[0.0], 1).is_err());
        println!("Inkling-2 NLL/top-1 self-test passed");
        return Ok(());
    }
    if args.len() != 3 {
        return Err("usage: inkling2_quality MODEL FIXTURE_JSON OUTPUT_JSON | --self-test".into());
    }
    let fixture: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let cases = fixture["cases"].as_array().ok_or("fixture needs cases")?;
    if cases.is_empty() {
        return Err("empty fixture".into());
    }
    let model = Model::open(&args[0], Backend::Cuda, 0, true)?;
    if model.family() != ModelFamily::Inkling2Vq {
        return Err("this recorder requires Inkling-2 VQ".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let mut results = Vec::new();
    for case in cases {
        let id = case["id"].as_str().ok_or("case needs id")?;
        let tokens: Vec<i32> = serde_json::from_value(case["tokens"].clone())?;
        let prefix: usize = serde_json::from_value(case["prefix"].clone())?;
        if prefix == 0
            || prefix >= tokens.len()
            || tokens.iter().any(|&t| t < 0 || t as usize >= vocab)
        {
            return Err(format!("invalid tokens or prefix in {}", id).into());
        }
        let mut session = model.session(i32::try_from(tokens.len() + 1)?)?;
        let started = std::time::Instant::now();
        session.sync(&TokenBuffer::from_tokens(tokens[..prefix].to_vec()))?;
        let prefill_s = started.elapsed().as_secs_f64();
        let mut nll = Vec::new();
        let mut top1 = Vec::new();
        for at in prefix..tokens.len() {
            if session.pos() != at as i32 {
                return Err(format!("incorrect frontier in {}", id).into());
            }
            let logits = session.copy_logits(vocab)?;
            let (top, loss) = score(&logits, tokens[at] as usize)?;
            top1.push(top);
            nll.push(loss);
            if at + 1 < tokens.len() {
                session.eval(tokens[at])?;
            }
        }
        println!(
            "{}",
            json!({"case": id, "prefix": prefix,
            "positions": nll.len(), "prefill_s": prefill_s,
            "seconds": started.elapsed().as_secs_f64()})
        );
        std::io::stdout().flush()?;
        results.push(json!({"id": id, "prefix": prefix,
            "nll": nll, "top1": top1}));
    }
    let out = json!({"vocab": vocab, "cases": results});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&out)?)?;
    Ok(())
}
