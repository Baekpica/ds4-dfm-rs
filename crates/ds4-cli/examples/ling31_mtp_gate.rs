//! Ling-3.1 MTP losslessness at the token level: greedy generation through
//! drafts and batched verify, then a plain one-row decode teacher-forced on
//! the same tokens. Every generated token must be that decode's argmax up to
//! a BF16-scale tie (verify and decode widths round differently); a broken
//! KDA or draft-row rollback shows up as a large gap.
//! Usage: ling31_mtp_gate MODEL FIXTURE_JSON DRAFTS [OUTPUT_TOKENS]
use ds4_core::{Backend, Model, ModelFamily, ModelOpenOption, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;

const CONTEXT: i32 = 4096;
const DEFAULT_OUTPUT: usize = 192;
// Logit gap still counted as a tie: a few BF16 steps at |logit| ~ 32.
const TIE: f32 = 0.5;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: ling31_mtp_gate MODEL FIXTURE_JSON DRAFTS [OUTPUT_TOKENS]".into());
    }
    let fixture: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let drafts: i32 = args[2].parse()?;
    let limit = args
        .get(3)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(DEFAULT_OUTPUT);
    let options = [ModelOpenOption::MtpDraftTokens(drafts)];
    let model =
        Model::open_with_support_options(&args[0], Backend::Cuda, 8, true, None, None, &options)?;
    if model.family() != ModelFamily::Ling31Vq {
        return Err("this gate requires the Ling-3.1 VQ release".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let eos = model.token_eos();
    let (mut total, mut calls, mut breaks) = (0usize, 0usize, 0usize);
    let mut worst = 0f32;
    for case in fixture["cases"].as_array().ok_or("fixture needs cases")? {
        let id = case["id"].as_str().ok_or("case needs id")?;
        let prompt = TokenBuffer::from_tokens(ids(&case["prompt_ids"])?);

        // Drafts + batched verify, as ds4/ds4-server generate greedily.
        let mut generated = Vec::new();
        let mut verifies = 0usize;
        let mut session = model.session(CONTEXT)?;
        session.sync(&prompt)?;
        while generated.len() < limit {
            let token = session.argmax();
            if model.token_is_stop(token) {
                break;
            }
            let accepted =
                session.eval_speculative_argmax(token, (limit - generated.len()) as i32, eos)?;
            verifies += 1;
            generated.extend(accepted);
        }
        drop(session);

        // One-row decode on the same tokens.
        let mut session = model.session(CONTEXT)?;
        session.sync(&prompt)?;
        let mut gap_max = 0f32;
        let mut over = 0usize;
        for &token in &generated {
            let logits = session.copy_logits(vocab)?;
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let gap = max - logits[token as usize];
            gap_max = gap_max.max(gap);
            if gap > TIE {
                over += 1;
            }
            session.eval(token)?;
        }
        drop(session);

        total += generated.len();
        calls += verifies;
        breaks += over;
        worst = worst.max(gap_max);
        println!(
            "{}",
            json!({"id": id, "tokens": generated.len(), "speculative_calls": verifies,
                "tokens_per_call": generated.len() as f64 / verifies.max(1) as f64,
                "max_gap": gap_max, "over_tie": over})
        );
        std::io::stdout().flush()?;
    }
    println!(
        "{}",
        json!({"drafts": drafts, "tokens": total, "speculative_calls": calls,
            "tokens_per_call": total as f64 / calls.max(1) as f64, "max_gap": worst,
            "over_tie": breaks, "tie": TIE})
    );
    if breaks > 0 {
        return Err(format!("{breaks} tokens beyond the {TIE} tie margin").into());
    }
    Ok(())
}
