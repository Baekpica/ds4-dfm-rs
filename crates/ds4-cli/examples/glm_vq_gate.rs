//! Bounded source-forward receipt. Fixed source token IDs avoid input drift.
use ds4_core::{Backend, Model, ModelFamily, TokenBuffer};
use serde_json::json;
use std::io::Write;
use std::path::Path;

const CONTEXT: i32 = 256;
const DECODE: usize = 2;

fn save_logits(path: &Path, logits: &[f32]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    for value in logits {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: glm_vq_gate MODEL TOKENS_JSON OUTPUT".into());
    }
    let ids: Vec<i32> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    if ids.is_empty() || ids.len() + DECODE >= CONTEXT as usize {
        return Err("prompt exceeds bounded gate context".into());
    }
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let opened = std::time::Instant::now();
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    assert_eq!(model.family(), ModelFamily::Glm53Vq);
    let open_s = opened.elapsed().as_secs_f64();
    let mut session = model.session(CONTEXT)?;
    let start = std::time::Instant::now();
    session.sync(&TokenBuffer::from_tokens(ids.clone()))?;
    let prefill_s = start.elapsed().as_secs_f64();
    let mut generated = Vec::new();
    let mut decode_s = Vec::new();
    for step in 0..=DECODE {
        let logits = session.copy_logits(model.vocab().n_vocab() as usize)?;
        assert!(logits.iter().all(|v| v.is_finite()), "nonfinite logits");
        save_logits(&out.join(format!("logits_{step}.f32")), &logits)?;
        if step == DECODE {
            break;
        }
        let token = session.argmax();
        assert!(token >= 0);
        generated.push(token);
        let start = std::time::Instant::now();
        session.eval(token)?;
        decode_s.push(start.elapsed().as_secs_f64());
    }
    assert_eq!(session.pos() as usize, ids.len() + DECODE);
    let report = json!({"model": args[0], "prompt_ids": ids, "generated": generated,
        "open_s": open_s, "prefill_s": prefill_s, "decode_s": decode_s,
        "context": CONTEXT, "mtp": false, "capture": false,
        "scope": "finite source-forward receipt; quality and performance not qualified"});
    std::fs::write(out.join("result.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
