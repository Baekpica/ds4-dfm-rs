//! Ling-3.1 VQ against reference A (the BF16 source model on the same token
//! ids): teacher-forced target NLL and top-1, then a greedy continuation.
//! Usage: ling31_quality MODEL FIXTURE_JSON OUTPUT_DIR [REFERENCE_DIR] [CONTEXT]
use ds4_core::{Backend, Model, ModelFamily, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;

const DEFAULT_CONTEXT: i32 = 4096;
const OUTPUT_LIMIT: usize = 256;
// Tokenizer ids; the backend masks the head's 31 padded rows past them.
const TOKENS: usize = 157_153;
const PAD_LOGIT_MAX: f32 = -1e29;

fn ids(value: &Value) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let list = value.as_array().ok_or("expected a token list")?;
    list.iter()
        .map(|v| {
            v.as_i64()
                .map(|t| t as i32)
                .ok_or_else(|| "bad token".into())
        })
        .collect()
}

fn reference(dir: Option<&String>, id: &str) -> Option<Value> {
    let path = Path::new(dir?).join(format!("{id}.result.json"));
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=5).contains(&args.len()) {
        return Err(
            "usage: ling31_quality MODEL FIXTURE_JSON OUTPUT_DIR [REFERENCE_DIR] [CONTEXT]".into(),
        );
    }
    let fixture: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let context = args
        .get(4)
        .map(|v| v.parse::<i32>())
        .transpose()?
        .unwrap_or(DEFAULT_CONTEXT);
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    if model.family() != ModelFamily::Ling31Vq {
        return Err("this receipt requires the Ling-3.1 VQ release".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let (mut total_nll, mut total_tokens, mut ref_nll, mut agree, mut compared) =
        (0.0, 0usize, 0.0, 0usize, 0usize);
    for case in fixture["cases"].as_array().ok_or("fixture needs cases")? {
        let id = case["id"].as_str().ok_or("case needs id")?;
        let prompt = ids(&case["prompt_ids"])?;
        let targets = ids(&case["target_ids"])?;
        if prompt.len() + targets.len().max(OUTPUT_LIMIT) >= context as usize {
            return Err(format!("{id} exceeds context {context}").into());
        }
        let reference = reference(args.get(3), id);
        let ref_top1 = reference
            .as_ref()
            .and_then(|r| ids(&r["teacher_forced"]["top1_ids"]).ok());

        // Teacher forcing keeps both models on identical states even where
        // greedy samples part near a logit tie.
        let mut nll = Vec::with_capacity(targets.len());
        let mut top1 = Vec::with_capacity(targets.len());
        let mut session = model.session(context)?;
        session.sync(&TokenBuffer::from_tokens(prompt.clone()))?;
        for &token in &targets {
            let logits = session.copy_logits(vocab)?;
            if logits[TOKENS..].iter().any(|&v| v > PAD_LOGIT_MAX) {
                return Err("padded head rows are not masked".into());
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
            nll.push(max + sum.ln() - logits[token as usize] as f64);
            top1.push(session.argmax());
            session.eval(token)?;
        }
        drop(session);

        // Greedy answer from the prompt alone.
        let mut generated = Vec::new();
        let mut session = model.session(context)?;
        session.sync(&TokenBuffer::from_tokens(prompt.clone()))?;
        for _ in 0..OUTPUT_LIMIT {
            let token = session.argmax();
            generated.push(token);
            if model.token_is_stop(token) {
                break;
            }
            session.eval(token)?;
        }
        drop(session);

        let mean = nll.iter().sum::<f64>() / nll.len() as f64;
        total_nll += nll.iter().sum::<f64>();
        total_tokens += nll.len();
        let ref_mean = reference
            .as_ref()
            .and_then(|r| r["teacher_forced"]["nll"].as_f64());
        let matched = ref_top1
            .as_ref()
            .map(|r| r.iter().zip(&top1).filter(|(a, b)| a == b).count());
        if let (Some(r), Some(m)) = (ref_mean, matched) {
            ref_nll += r * nll.len() as f64;
            agree += m;
            compared += nll.len();
        }
        let ref_gen = reference
            .as_ref()
            .and_then(|r| ids(&r["generation"]["ids"]).ok());
        let prefix = ref_gen
            .as_ref()
            .map(|r| r.iter().zip(&generated).take_while(|(a, b)| a == b).count());
        let mut answer = Vec::new();
        for &token in &generated {
            answer.extend(model.token_text(token)?);
        }
        let result = json!({"id": id, "prompt_tokens": prompt.len(), "target_tokens": targets.len(),
            "nll": mean, "reference_nll": ref_mean, "top1_agree": matched,
            "target_nll": nll, "top1_ids": top1, "generated_ids": generated,
            "reference_prefix_match": prefix, "answer": String::from_utf8_lossy(&answer)});
        std::fs::write(
            out.join(format!("{id}.json")),
            serde_json::to_vec_pretty(&result)?,
        )?;
        println!(
            "{}",
            json!({"id": id, "nll": mean, "reference_nll": ref_mean,
            "top1_agree": matched, "targets": targets.len(), "gen_prefix": prefix,
            "gen_tokens": generated.len()})
        );
        std::io::stdout().flush()?;
    }
    let summary = json!({"tokens": total_tokens, "nll": total_nll / total_tokens as f64,
        "compared_tokens": compared,
        "reference_nll": (compared > 0).then(|| ref_nll / compared as f64),
        "top1_agreement": (compared > 0).then(|| agree as f64 / compared as f64)});
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
