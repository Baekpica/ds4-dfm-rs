//! A.X-K2 VQ against the handoff's teacher-forced goldens (eval_e2e.py
//! --goldens): per position top-1 against the student (this release in the
//! reference arithmetic) and the teacher (the FP8 source), top-8 overlap and
//! the logit error on the student's top-8 and the next-token NLL. A case
//! prefills its first PREFILL tokens (checked at the last row), then decodes
//! the rest one by one; a case longer than the indexer's 2,048 keys
//! prefills all but LONG_TAIL rows, so both paths run the sparse selection.
//!
//! The gate is the final output against a rerun of the reference on this
//! host (REFERENCE_JSON: top8_ids and nll per position), with widths fixed
//! from the known differences rather than exact agreement: the goldens'
//! own reference reproduces them at ~95% top-1 from the RTX PRO 6000 to the
//! GB10, because group and top-8 router margins are often tiny and
//! near-ties flip with any accumulation order. The engine adds FP8 latent
//! and index-key rows, gain-after-dot and Q8_1 / E4M3 prefill activations.
//! Pass: NLL within NLL_BAND of the reference on the same positions, and
//! top-1 against the student at most TOP1_BAND below the reference's own.
//! SPLIT `decode` decodes long cases from PREFILL too: the prefill-built and
//! decode-built caches then score the same tail.
//! Usage: axk2_quality MODEL GOLDENS_JSON OUTPUT_JSON [PREFILL] [CONTEXT] [REFERENCE_JSON] [tail|decode]
use ds4_core::{Backend, Model, ModelFamily, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;

const DEFAULT_PREFILL: usize = 64;
const DEFAULT_CONTEXT: i32 = 16384;
// Rows a long case decodes after its prefill; queries see every key up to
// the indexer's top-k.
const LONG_TAIL: usize = 256;
const DENSE_KEYS: usize = 2048;

// How a long case splits between prefill and one-row decodes.
#[derive(Clone, Copy, PartialEq)]
enum Split {
    Tail,
    Decode,
}
// Relative NLL band around the reference: 0.3% across GPUs plus the
// engine's own rounding (FP8 KV, quantized prefill activations).
const NLL_BAND: f64 = 0.01;
// Top-1 points the engine may trail the reference's agreement with the
// student: its extra rounding flips near-ties the reference keeps.
const TOP1_BAND: f64 = 0.02;
// A miss where the student's top two logits are this far apart is no tie.
const HARD_GAP: f64 = 0.25;

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

#[derive(Default)]
struct Tally {
    // Per position: index, our top-1, the student's, its top-2 gap, max
    // error, our next-token NLL and the reference's.
    rows: Vec<Value>,
    positions: usize,
    student: usize,
    teacher: usize,
    hard: usize,
    overlap: usize,
    err_sum: f64,
    err_max: f64,
    nll_sum: f64,
    nll_count: usize,
    reference: usize,
    // The reference rerun: its NLL, the paired difference and its top-1
    // against the student (the cross-GPU floor).
    ref_nll_sum: f64,
    diff_sum: f64,
    diff_sq: f64,
    paired: usize,
    ref_student: usize,
}

impl Tally {
    fn add(
        &mut self,
        logits: &[f32],
        golden: &Value,
        reference: Option<&Value>,
        at: usize,
        next: Option<i32>,
    ) -> Result<(), String> {
        let ids = |key: &str| list(&golden[key][at], |v| v.as_u64().map(|t| t as usize));
        let (student, teacher) = (ids("student_top8_ids")?, ids("teacher_top8_ids")?);
        let values = list(&golden["student_top8_logits"][at], Value::as_f64)?;
        let ours = top8(logits);
        self.positions += 1;
        self.student += usize::from(ours[0] == student[0]);
        self.teacher += usize::from(ours[0] == teacher[0]);
        self.hard += usize::from(ours[0] != student[0] && values[0] - values[1] >= HARD_GAP);
        self.overlap += ours.iter().filter(|t| student.contains(t)).count();
        let mut worst = 0.0f64;
        for (&id, &value) in student.iter().zip(&values) {
            let err = (logits[id] as f64 - value).abs();
            self.err_sum += err;
            worst = worst.max(err);
        }
        self.err_max = self.err_max.max(worst);
        let nll = next.map(|next| {
            let max = logits[ours[0]] as f64;
            let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
            max + sum.ln() - logits[next as usize] as f64
        });
        if let Some(nll) = nll {
            self.nll_sum += nll;
            self.nll_count += 1;
        }
        if let Some(reference) = reference {
            let top1 = reference["top8_ids"][at][0]
                .as_u64()
                .ok_or("reference needs top8_ids")?;
            self.reference += usize::from(ours[0] as u64 == top1);
            self.ref_student += usize::from(top1 == student[0] as u64);
            if let (Some(nll), Some(theirs)) = (nll, reference["nll"][at].as_f64()) {
                self.ref_nll_sum += theirs;
                self.diff_sum += nll - theirs;
                self.diff_sq += (nll - theirs) * (nll - theirs);
                self.paired += 1;
            }
        }
        let theirs = reference.and_then(|r| r["nll"][at].as_f64());
        self.rows.push(json!([
            at,
            ours[0],
            student[0],
            values[0] - values[1],
            worst,
            nll,
            theirs
        ]));
        Ok(())
    }

    fn merge(&mut self, o: &Tally) {
        self.positions += o.positions;
        self.student += o.student;
        self.teacher += o.teacher;
        self.hard += o.hard;
        self.overlap += o.overlap;
        self.err_sum += o.err_sum;
        self.err_max = self.err_max.max(o.err_max);
        self.nll_sum += o.nll_sum;
        self.nll_count += o.nll_count;
        self.reference += o.reference;
        self.ref_nll_sum += o.ref_nll_sum;
        self.diff_sum += o.diff_sum;
        self.diff_sq += o.diff_sq;
        self.paired += o.paired;
        self.ref_student += o.ref_student;
    }

    // Paired NLL difference (engine - reference) and its standard error.
    fn delta(&self) -> Option<(f64, f64)> {
        if self.paired < 2 {
            return None;
        }
        let n = self.paired as f64;
        let mean = self.diff_sum / n;
        let var = (self.diff_sq / n - mean * mean).max(0.0) * n / (n - 1.0);
        Some((mean, (var / n).sqrt()))
    }

    fn report(&self) -> Value {
        let n = self.positions.max(1) as f64;
        let (delta, se) = self
            .delta()
            .map_or((None, None), |(d, s)| (Some(d), Some(s)));
        json!({"positions": self.positions,
            "top1_student": self.student as f64 / n, "top1_teacher": self.teacher as f64 / n,
            "hard_misses": self.hard, "top8_overlap": self.overlap as f64 / (8.0 * n),
            "logit_err_mean": self.err_sum / (8.0 * n), "logit_err_max": self.err_max,
            "nll": self.nll_sum / self.nll_count.max(1) as f64,
            "reference_nll": (self.paired > 0).then(|| self.ref_nll_sum / self.paired as f64),
            "nll_delta": delta, "nll_delta_se": se,
            "top1_reference": self.reference as f64 / n,
            "reference_top1_student": self.ref_student as f64 / n})
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=7).contains(&args.len()) {
        return Err(
            "usage: axk2_quality MODEL GOLDENS_JSON OUTPUT_JSON [PREFILL] [CONTEXT] [REFERENCE_JSON] [tail|decode]".into(),
        );
    }
    let goldens: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let reference: Option<Value> = args
        .get(5)
        .map(|p| std::fs::read(p))
        .transpose()?
        .map(|b| serde_json::from_slice(&b))
        .transpose()?;
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
    let split = match args.get(6).map(String::as_str) {
        None | Some("tail") => Split::Tail,
        Some("decode") => Split::Decode,
        Some(_) => return Err("SPLIT must be tail or decode".into()),
    };
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    if model.family() != ModelFamily::Axk2Vq {
        return Err("this receipt requires the A.X-K2 VQ release".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let (mut all, mut cases) = (Tally::default(), serde_json::Map::new());
    for (name, golden) in goldens.as_object().ok_or("goldens must be an object")? {
        let ids = list(&golden["ids"], |v| v.as_i64().map(|t| t as i32))?;
        if ids.len() >= context as usize {
            return Err(format!("{name} exceeds context {context}").into());
        }
        let first = if split == Split::Tail && ids.len() > DENSE_KEYS {
            ids.len() - LONG_TAIL
        } else {
            prefill.min(ids.len())
        };
        let case_ref = reference.as_ref().map(|r| &r[name.as_str()]);
        let mut tally = Tally::default();
        let mut session = model.session(context)?;
        let started = std::time::Instant::now();
        session.sync(&TokenBuffer::from_tokens(ids[..first].to_vec()))?;
        let prefill_s = started.elapsed().as_secs_f64();
        tally.add(
            &session.copy_logits(vocab)?,
            golden,
            case_ref,
            first - 1,
            ids.get(first).copied(),
        )?;
        let started = std::time::Instant::now();
        for (at, &token) in ids.iter().enumerate().skip(first) {
            session.eval(token)?;
            tally.add(
                &session.copy_logits(vocab)?,
                golden,
                case_ref,
                at,
                ids.get(at + 1).copied(),
            )?;
        }
        let decode_s = started.elapsed().as_secs_f64();
        drop(session);
        let report = tally.report();
        println!(
            "{}",
            json!({"case": name, "tokens": ids.len(), "prefill": first, "prefill_s": prefill_s,
                "decode_s": decode_s, "result": report})
        );
        std::io::stdout().flush()?;
        all.merge(&tally);
        cases.insert(name.clone(), json!({"result": report, "rows": tally.rows}));
    }
    let summary = all.report();
    // Without a rerun of the reference there is no floor: report only.
    let pass = (all.paired > 0).then(|| {
        let reference = all.ref_nll_sum / all.paired as f64;
        let n = all.positions.max(1) as f64;
        (all.diff_sum / all.paired as f64).abs() <= NLL_BAND * reference
            && all.student as f64 / n >= all.ref_student as f64 / n - TOP1_BAND
    });
    let out = json!({"prefill": prefill, "context": context, "nll_band": NLL_BAND,
        "top1_band": TOP1_BAND, "pass": pass, "all": summary, "cases": cases});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&out)?)?;
    println!("{}", json!({"all": out["all"], "pass": pass}));
    if pass == Some(false) {
        return Err("the engine is outside the reference's NLL or top-1 band".into());
    }
    Ok(())
}
