//! MiniMax-M3.1 VQ against a rerun of the handoff's reference forward
//! (tools/minimax_ref.py) on this host over the same release bytes, with
//! NVFP4 K/V and index keys simulated as the engine stores them: per
//! position top-1 and top-8 overlap with the reference, the logit error on
//! its top-8 and the paired next-token NLL. A case prefills its first rows
//! (checked at the last row), then decodes the rest one by one; a case longer
//! than MSA's dense window (16 blocks of 128) prefills all but LONG_TAIL rows
//! so both paths run the sparse block selection.
//!
//! The source is chaotic on untrained positions (system prompts, tool
//! schemas, user turns): the reference against itself batched differently
//! has KL 0.565 over all positions and 0.003 on assistant turns (handoff
//! noise-floor.json). The gate therefore reads assistant positions:
//! paired NLL within NLL_BAND of the reference and top-1 at least TOP1_FLOOR,
//! bands fixed from the engine's known differences (Q8_1 / E4M3 prefill
//! activations, gain after the dot, accumulation order), not exact agreement.
//! Usage: minimax_quality [--record] MODEL REFERENCE_JSON OUTPUT_JSON [PREFILL] [CONTEXT]
//! --record writes measured positions in the reference schema for same-host comparisons.
//! Input cases need tokens only; unmeasured prefix positions remain null.
use ds4_core::{Backend, Model, ModelFamily, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;

const DEFAULT_PREFILL: usize = 64;
const DEFAULT_CONTEXT: i32 = 16384;
// Rows a long case decodes after its prefill.
const LONG_TAIL: usize = 256;
const DENSE_WINDOW: usize = 16 * 128;
// Tokenizer ids; the backend masks the head's 3 padded rows past them.
const TOKENS: usize = 200_061;
const PAD_LOGIT_MAX: f32 = -1e29;
// Chat markers of the handoff's assistant mask (measure_r8_mm31.py).
const ROLE: i32 = 200_019;
const EOS: i32 = 200_020;
const AI: i32 = 1361;
// Relative assistant-turn NLL band around the reference.
const NLL_BAND: f64 = 0.02;
// Assistant-turn top-1 agreement with the reference.
const TOP1_FLOOR: f64 = 0.95;

fn list<T>(value: &Value, f: impl Fn(&Value) -> Option<T>) -> Result<Vec<T>, String> {
    value
        .as_array()
        .ok_or("expected a list")?
        .iter()
        .map(|v| f(v).ok_or_else(|| "bad list entry".to_string()))
        .collect()
}

fn top8(logits: &[f32]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.select_nth_unstable_by(7, |&a, &b| logits[b].total_cmp(&logits[a]));
    ids.truncate(8);
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids
}

// Positions inside assistant turns: after `]~b]ai` up to and with `[e~[`.
fn assistant_mask(tokens: &[i32]) -> Vec<bool> {
    let mut mask = vec![false; tokens.len()];
    let mut inside = false;
    for i in 0..tokens.len() {
        if i > 0 && tokens[i - 1] == ROLE {
            inside = tokens[i] == AI;
        } else if inside {
            mask[i] = true;
        }
        if tokens[i] == EOS {
            inside = false;
        }
    }
    mask
}

#[derive(Default, Clone)]
struct Tally {
    positions: usize,
    top1: usize,
    overlap: usize,
    err_sum: f64,
    err_max: f64,
    nll_sum: f64,
    ref_nll_sum: f64,
    diff_sq: f64,
    paired: usize,
}

impl Tally {
    fn add(
        &mut self,
        ours: &[usize],
        theirs: &[usize],
        err: f64,
        err_max: f64,
        nll: Option<(f64, f64)>,
    ) {
        self.positions += 1;
        self.top1 += usize::from(ours[0] == theirs[0]);
        self.overlap += ours.iter().filter(|t| theirs.contains(t)).count();
        self.err_sum += err;
        self.err_max = self.err_max.max(err_max);
        if let Some((nll, reference)) = nll {
            self.nll_sum += nll;
            self.ref_nll_sum += reference;
            self.diff_sq += (nll - reference) * (nll - reference);
            self.paired += 1;
        }
    }

    fn merge(&mut self, o: &Tally) {
        self.positions += o.positions;
        self.top1 += o.top1;
        self.overlap += o.overlap;
        self.err_sum += o.err_sum;
        self.err_max = self.err_max.max(o.err_max);
        self.nll_sum += o.nll_sum;
        self.ref_nll_sum += o.ref_nll_sum;
        self.diff_sq += o.diff_sq;
        self.paired += o.paired;
    }

    fn report(&self) -> Value {
        let n = self.positions.max(1) as f64;
        let p = self.paired.max(1) as f64;
        json!({"positions": self.positions, "top1": self.top1 as f64 / n,
            "top8_overlap": self.overlap as f64 / (8.0 * n),
            "logit_err_mean": self.err_sum / (8.0 * n), "logit_err_max": self.err_max,
            "nll": self.nll_sum / p, "reference_nll": self.ref_nll_sum / p,
            "nll_rms_diff": (self.diff_sq / p).sqrt()})
    }
}

fn capture(logits: &[f32], next: Option<i32>) -> Result<Value, String> {
    if logits.len() <= TOKENS
        || logits.iter().any(|v| !v.is_finite())
        || logits[TOKENS..].iter().any(|&v| v > PAD_LOGIT_MAX)
        || next.is_some_and(|v| v < 0 || v as usize >= TOKENS)
    {
        return Err("invalid logits or next token for a reference position".into());
    }
    let ids = top8(logits);
    let values: Vec<f32> = ids.iter().map(|&id| logits[id]).collect();
    let nll = next.map(|next| {
        let max = logits[ids[0]] as f64;
        let sum: f64 = logits[..TOKENS]
            .iter()
            .map(|&v| (v as f64 - max).exp())
            .sum();
        max + sum.ln() - logits[next as usize] as f64
    });
    Ok(json!({"top8_ids": ids, "top8_logits": values, "nll": nll}))
}

// One position: ours against the reference's top-8 and next-token NLL.
fn check(
    logits: &[f32],
    reference: &Value,
    at: usize,
    next: Option<i32>,
) -> Result<(Vec<usize>, Vec<usize>, f64, f64, Option<(f64, f64)>), String> {
    if logits[TOKENS..].iter().any(|&v| v > PAD_LOGIT_MAX) {
        return Err("padded head rows are not masked".into());
    }
    let theirs = list(&reference["top8_ids"][at], |v| {
        v.as_u64().map(|t| t as usize)
    })?;
    let values = list(&reference["top8_logits"][at], Value::as_f64)?;
    let ours = top8(logits);
    let (mut err, mut worst) = (0.0f64, 0.0f64);
    for (&id, &value) in theirs.iter().zip(&values) {
        let e = (logits[id] as f64 - value).abs();
        err += e;
        worst = worst.max(e);
    }
    let nll = match (next, reference["nll"][at].as_f64()) {
        (Some(next), Some(theirs)) => {
            let max = logits[ours[0]] as f64;
            let sum: f64 = logits[..TOKENS]
                .iter()
                .map(|&v| (v as f64 - max).exp())
                .sum();
            Some((max + sum.ln() - logits[next as usize] as f64, theirs))
        }
        _ => None,
    };
    Ok((ours, theirs, err, worst, nll))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let recording = args.first().is_some_and(|s| s == "--record");
    if recording {
        args.remove(0);
    }
    if !(3..=5).contains(&args.len()) {
        return Err(
            "usage: minimax_quality [--record] MODEL REFERENCE_JSON OUTPUT_JSON [PREFILL] [CONTEXT]".into(),
        );
    }
    let reference: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let prefill = args
        .get(3)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(DEFAULT_PREFILL)
        .max(1);
    let context = args
        .get(4)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(DEFAULT_CONTEXT);
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    if model.family() != ModelFamily::MinimaxVq {
        return Err("this receipt requires the MiniMax-M3.1 VQ release".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let (mut all, mut assistant) = (Tally::default(), Tally::default());
    let mut cases = Vec::new();
    let mut recorded_cases = Vec::new();
    for (index, case) in reference["cases"]
        .as_array()
        .ok_or("reference needs cases")?
        .iter()
        .enumerate()
    {
        let ids = list(&case["tokens"], |v| v.as_i64().map(|t| t as i32))?;
        if ids.len() >= context as usize {
            return Err(format!("case {index} exceeds context {context}").into());
        }
        let first = if ids.len() > DENSE_WINDOW {
            ids.len() - LONG_TAIL
        } else {
            prefill.min(ids.len())
        };
        if ids.is_empty() {
            return Err(format!("case {index} is empty").into());
        }
        let mut captured = recording.then(|| {
            let mut value = case.clone();
            for key in ["top8_ids", "top8_logits", "nll"] {
                value[key] = json!(vec![Value::Null; ids.len()]);
            }
            value
        });
        let mask = assistant_mask(&ids);
        let (mut tally, mut ai) = (Tally::default(), Tally::default());
        let mut session = model.session(context)?;
        let started = std::time::Instant::now();
        session.sync(&TokenBuffer::from_tokens(ids[..first].to_vec()))?;
        let prefill_s = started.elapsed().as_secs_f64();
        let mut record = |logits: &[f32], at: usize| -> Result<(), String> {
            if let Some(ref mut value) = captured {
                let position = capture(logits, ids.get(at + 1).copied())?;
                for key in ["top8_ids", "top8_logits", "nll"] {
                    value[key][at] = position[key].clone();
                }
                return Ok(());
            }
            let (ours, theirs, err, worst, nll) =
                check(logits, case, at, ids.get(at + 1).copied())?;
            tally.add(&ours, &theirs, err, worst, nll);
            if mask[at] {
                ai.add(&ours, &theirs, err, worst, nll);
            }
            Ok(())
        };
        record(&session.copy_logits(vocab)?, first - 1)?;
        let started = std::time::Instant::now();
        let mut decode_forward_s = 0.0;
        for (at, &token) in ids.iter().enumerate().skip(first) {
            let forward = std::time::Instant::now();
            session.eval(token)?;
            let logits = session.copy_logits(vocab)?;
            // The copy completes GPU work; reference scoring stays outside this timer.
            decode_forward_s += forward.elapsed().as_secs_f64();
            record(&logits, at)?;
        }
        let decode_s = started.elapsed().as_secs_f64();
        drop(session);
        let row = json!({"case": index, "tokens": ids.len(), "prefill": first,
            "prefill_s": prefill_s, "decode_s": decode_s, "decode_forward_s": decode_forward_s,
            "all": tally.report(), "assistant": ai.report()});
        println!("{row}");
        std::io::stdout().flush()?;
        all.merge(&tally);
        assistant.merge(&ai);
        cases.push(row);
        if let Some(value) = captured {
            recorded_cases.push(value);
        }
    }
    if recording {
        let mut out = reference;
        out["source"] = json!("ds4 MiniMax same-host teacher-forced reference");
        out["cases"] = json!(recorded_cases);
        out["measurements"] = json!(cases);
        std::fs::write(&args[2], serde_json::to_vec_pretty(&out)?)?;
        return Ok(());
    }
    let pass = (assistant.paired > 0).then(|| {
        let n = assistant.positions as f64;
        let reference = assistant.ref_nll_sum / assistant.paired as f64;
        let ours = assistant.nll_sum / assistant.paired as f64;
        (ours - reference).abs() <= NLL_BAND * reference && assistant.top1 as f64 / n >= TOP1_FLOOR
    });
    let out = json!({"prefill": prefill, "context": context, "nll_band": NLL_BAND, "top1_floor": TOP1_FLOOR,
        "pass": pass, "all": all.report(), "assistant": assistant.report(), "cases": cases});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&out)?)?;
    println!(
        "{}",
        json!({"all": out["all"], "assistant": out["assistant"], "pass": pass})
    );
    if pass == Some(false) {
        return Err(
            "the engine is outside the reference's assistant-turn NLL or top-1 band".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_position_round_trips_and_rejects_invalid_values() {
        let mut logits = vec![0.0; TOKENS + 3];
        logits[TOKENS..].fill(-1e30);
        logits[7] = 1.0;
        let row = capture(&logits, Some(7)).unwrap();
        let expected = (1.0 + (TOKENS - 1) as f64 * (-1.0f64).exp()).ln();
        assert!((row["nll"].as_f64().unwrap() - expected).abs() < 1e-10);
        let reference = json!({"top8_ids": [row["top8_ids"]], "top8_logits": [row["top8_logits"]], "nll": [row["nll"]]});
        let (a, b, err, worst, nll) = check(&logits, &reference, 0, Some(7)).unwrap();
        assert_eq!(a, b);
        assert_eq!((err, worst), (0.0, 0.0));
        let (ours, theirs) = nll.unwrap();
        assert!((ours - theirs).abs() < 1e-12);
        assert!(capture(&logits, Some(-1)).is_err());
        logits[TOKENS] = 0.0;
        assert!(capture(&logits, None).is_err());
        logits[TOKENS] = -1e30;
        logits[0] = f32::NAN;
        assert!(capture(&logits, None).is_err());
    }
}
