//! Teacher-force pinned token fixtures through the production Rust session.
use ds4_core::{Backend, Model, ModelFamily, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: naive_gate MODEL CASES.json OUTPUT_DIR".into());
    }
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let out = std::path::Path::new(&args[2]);
    std::fs::create_dir(out)?;
    let model = Model::open(&args[0], Backend::Cuda, 0, true)?;
    if model.family() != ModelFamily::NaiveN05 {
        return Err("this gate requires the Naive target".into());
    }
    for case in cases {
        let id = case["id"].as_str().ok_or("case needs id")?;
        let tokens: Vec<i32> = serde_json::from_value(case["tokens"].clone())?;
        let positions: Vec<usize> = serde_json::from_value(case["positions"].clone())?;
        if tokens.is_empty()
            || positions.windows(2).any(|p| p[0] >= p[1])
            || positions.iter().any(|&p| p >= tokens.len())
        {
            return Err("invalid teacher-force positions".into());
        }
        let mut session = model.session(i32::try_from(tokens.len() + 1)?)?;
        let mut file = std::fs::File::create(out.join(format!("{id}.f32")))?;
        let begin = std::time::Instant::now();
        for &position in &positions {
            session.sync(&TokenBuffer::from_tokens(tokens[..=position].to_vec()))?;
            let logits = session.copy_logits(model.vocab().n_vocab() as usize)?;
            if logits.iter().any(|v| !v.is_finite()) || session.pos() != position as i32 + 1 {
                return Err("non-finite logits or incorrect frontier".into());
            }
            // Little-endian full vocabulary rows are compared outside the
            // host, so the reference decoder cannot influence inference.
            for value in logits {
                file.write_all(&value.to_le_bytes())?;
            }
        }
        file.flush()?;
        println!(
            "{}",
            json!({"id": id, "positions": positions,
            "vocab": model.vocab().n_vocab(), "seconds": begin.elapsed().as_secs_f64()})
        );
        std::io::stdout().flush()?;
    }
    Ok(())
}
