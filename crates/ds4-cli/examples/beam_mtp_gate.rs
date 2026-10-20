//! Beam DFlash2 losslessness at the token level: greedy generation through
//! drafted blocks and one batched verify per block, then a plain one-row
//! decode teacher-forced on the same tokens. Each generated token's gap below
//! that decode's argmax is held to the model's own floor: the handoff's
//! reference (same student arithmetic) on an H100 and rerun on this GB10
//! picks a token the other run scores more than 0.5 below its argmax at
//! 1.73-1.97% of 5,491 positions, more than 4 at 0.09-0.22%, and up to 8.3
//! (15.5 past the other's top-8). Beam's router margins (~1e-5 relative)
//! turn any rounding into route flips; verify rows round differently from
//! one-row decode (Q8_1 projection activations, E4M3 expert activations
//! above four rows). A first a-priori cap of 4 logits on every token failed
//! at 5.6 before that floor was measured; the floor is the gate.
//! Counts past TIE (a few BF16 steps at the 30 cap) and past HARD may not
//! exceed the 95% Poisson quantile of the floor's rates at the run's token
//! count; no gap may pass MAX, beyond the floor's worst. A broken context
//! ring, KV position or verify row lands far past these, on many tokens.
//! Usage: beam_mtp_gate MODEL DFLASH2 FIXTURE_JSON DRAFTS [MARGIN] [PROMPT_TOKENS] [OUTPUT_TOKENS]
use ds4_core::{Backend, Model, ModelFamily, ModelOpenOption, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;

const CONTEXT: i32 = 4096;
const DEFAULT_PROMPT: usize = 128;
const DEFAULT_OUTPUT: usize = 192;
const TIE: f32 = 0.5;
const TIE_RATE: f64 = 0.0197;
const HARD: f32 = 4.0;
const HARD_RATE: f64 = 0.0022;
const MAX: f32 = 16.0;

// Smallest k with P(Poisson(rate * n) <= k) >= 0.95.
fn allowed(rate: f64, n: usize) -> usize {
    let lambda = rate * n as f64;
    let (mut k, mut term) = (0usize, (-lambda).exp());
    let mut sum = term;
    while sum < 0.95 {
        k += 1;
        term *= lambda / k as f64;
        sum += term;
    }
    k
}

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
    if !(4..=7).contains(&args.len()) {
        return Err("usage: beam_mtp_gate MODEL DFLASH2 FIXTURE_JSON DRAFTS [MARGIN] [PROMPT_TOKENS] [OUTPUT_TOKENS]".into());
    }
    let fixture: Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let drafts: i32 = args[3].parse()?;
    let margin: f32 = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(3.0);
    let prompt_len: usize = args
        .get(5)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(DEFAULT_PROMPT);
    let limit: usize = args
        .get(6)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(DEFAULT_OUTPUT);
    let options = [
        ModelOpenOption::MtpDraftTokens(drafts),
        ModelOpenOption::MtpMargin(margin),
    ];
    let model = Model::open_with_support_options(
        &args[0],
        Backend::Cuda,
        8,
        true,
        Some(&args[1]),
        None,
        &options,
    )?;
    if model.family() != ModelFamily::BeamVq {
        return Err("this gate requires the Beam-1.0 VQ release".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let eos = model.token_eos();
    let (mut total, mut calls, mut over, mut worst) = (0usize, 0usize, 0usize, 0f32);
    let mut hard = 0usize;
    for case in fixture["cases"].as_array().ok_or("fixture needs cases")? {
        let id = case["id"].as_str().ok_or("case needs id")?;
        let all = ids(&case["ids"])?;
        let prompt = TokenBuffer::from_tokens(all[..prompt_len.min(all.len())].to_vec());

        // Drafted blocks + batched verify, as ds4-server decodes greedily.
        let mut generated = Vec::new();
        let mut steps = 0usize;
        let mut session = model.session(CONTEXT)?;
        session.sync(&prompt)?;
        while generated.len() < limit {
            let token = session.argmax();
            if model.token_is_stop(token) {
                break;
            }
            let accepted =
                session.eval_speculative_argmax(token, (limit - generated.len()) as i32, eos)?;
            steps += 1;
            generated.extend(accepted);
        }
        drop(session);

        // One-row decode on the same tokens.
        let mut session = model.session(CONTEXT)?;
        session.sync(&prompt)?;
        let (mut gap_max, mut case_over, mut misses) = (0f32, 0usize, Vec::new());
        let mut case_hard = 0usize;
        for (i, &token) in generated.iter().enumerate() {
            let logits = session.copy_logits(vocab)?;
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let gap = max - logits[token as usize];
            gap_max = gap_max.max(gap);
            case_hard += usize::from(gap > HARD);
            if gap > TIE {
                case_over += 1;
                // Absolute position of the token, as DS4_MTP_SPEC_LOG prints them.
                misses.push(json!([prompt.len() + i, gap]));
            }
            session.eval(token)?;
        }
        drop(session);

        total += generated.len();
        calls += steps;
        over += case_over;
        hard += case_hard;
        worst = worst.max(gap_max);
        println!(
            "{}",
            json!({"id": id, "tokens": generated.len(), "steps": steps,
                "tokens_per_step": generated.len() as f64 / steps.max(1) as f64,
                "max_gap": gap_max, "over_tie": case_over, "misses": misses})
        );
        std::io::stdout().flush()?;
    }
    let (over_max, hard_max) = (allowed(TIE_RATE, total), allowed(HARD_RATE, total));
    println!(
        "{}",
        json!({"drafts": drafts, "margin": margin, "tokens": total, "steps": calls,
            "tokens_per_step": total as f64 / calls.max(1) as f64, "max_gap": worst,
            "over_tie": over, "over_tie_max": over_max, "over_hard": hard, "over_hard_max": hard_max,
            "tie": TIE, "hard": HARD, "max": MAX})
    );
    if worst > MAX || over > over_max || hard > hard_max {
        return Err(format!(
            "max gap {worst} (cap {MAX}); {over} tokens over {TIE} (max {over_max}), \
             {hard} over {HARD} (max {hard_max}) of {total}"
        )
        .into());
    }
    Ok(())
}
