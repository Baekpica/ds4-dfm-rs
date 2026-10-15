//! Same-VQ arithmetic comparison: held-out likelihoods and generated answers.
use ds4_core::{Backend, Model, ModelFamily};
use serde_json::{json, Value};
use std::io::{BufWriter, Write};
use std::path::Path;

const CONTEXT: i32 = 2048;
const TARGET_LIMIT: usize = 24;
const OUTPUT_LIMIT: usize = 96;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: glm_vq_quality MODEL CASES_JSON OUTPUT_DIR".into());
    }
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    if model.family() != ModelFamily::Glm53Vq {
        return Err("this receipt requires the GLM VQ artifact".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    for (index, case) in cases.iter().enumerate() {
        let id = case["id"].as_str().ok_or("case needs id")?;
        let question = case["prompt"].as_str().ok_or("case needs prompt")?;
        let target = case["target"].as_str().ok_or("case needs target")?;
        let prompt = model.encode_chat_prompt_bytes(
            Some(b"Follow the request. Answer directly without analysis."),
            question.as_bytes(),
            0,
        )?;
        let mut targets = model.vocab().encode_bytes(target.as_bytes());
        targets.truncate(TARGET_LIMIT);
        if targets.is_empty() || prompt.len() + OUTPUT_LIMIT >= CONTEXT as usize {
            return Err("case exceeds bounded text quality scope".into());
        }
        let prefix = format!("case-{index}");
        let mut file = BufWriter::new(std::fs::File::create(out.join(format!("{prefix}.f32")))?);
        let mut nll = Vec::new();
        // Fixed target continuations keep the comparison on identical states
        // even when greedy samples differ near a logit tie.
        {
            let mut session = model.session(CONTEXT)?;
            session.sync(&prompt)?;
            for (step, &token) in targets.iter().enumerate() {
                let logits = session.copy_logits(vocab)?;
                if logits.iter().any(|v| !v.is_finite()) {
                    return Err(format!("{id} has nonfinite logits").into());
                }
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
                nll.push(max + sum.ln() - logits[token as usize] as f64);
                for value in logits {
                    file.write_all(&value.to_le_bytes())?;
                }
                session.eval(token)?;
                if session.pos() as usize != prompt.len() + step + 1 {
                    return Err("teacher-forced position transition failed".into());
                }
            }
        }
        file.flush()?;
        let mut generated = Vec::new();
        let mut answer = Vec::new();
        let mut stopped = false;
        {
            let mut session = model.session(CONTEXT)?;
            session.sync(&prompt)?;
            for step in 0..OUTPUT_LIMIT {
                let token = session.argmax();
                if model.token_is_stop(token) {
                    stopped = true;
                    break;
                }
                generated.push(token);
                answer.extend(model.token_text(token)?);
                session.eval(token)?;
                if session.pos() as usize != prompt.len() + step + 1 {
                    return Err("generated position transition failed".into());
                }
            }
        }
        let result = json!({"id": id, "prompt": question,
            "prompt_tokens": prompt.len(), "target_ids": targets, "target_nll": nll,
            "generated_ids": generated, "answer": String::from_utf8_lossy(&answer),
            "stopped": stopped, "vocab": vocab, "context": CONTEXT,
            "mtp": false, "capture": false,
            "scope": "bounded text arithmetic quality; not API/tool/media/state qualification"});
        std::fs::write(out.join(format!("{prefix}.json")), serde_json::to_vec_pretty(&result)?)?;
        println!("{}", json!({"case": id, "prompt_tokens": prompt.len(),
            "stopped": stopped, "answer": result["answer"]}));
        std::io::stdout().flush()?;
    }
    Ok(())
}
