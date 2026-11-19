//! Teacher-forced logits on selected held-out frontiers. Each position is zero-based,
//! including the token already consumed. Output keeps each measured reference pair.
use ds4_core::{Backend, Model, ModelFamily, ModelOpenOption, Sampler, TokenBuffer};
use serde_json::{json, Value};
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=5).contains(&args.len()) {
        return Err("usage: hy4_quality MODEL CASES.json OUTPUT.jsonl [DFLY2 [DRAFTS]]".into());
    }
    let (path, reference, output) = (&args[0], &args[1], &args[2]);
    let drafter = args.get(3).map(String::as_str);
    let drafts: usize = args.get(4).map_or(Ok(2), |d| d.parse())?;
    let cases: Value = serde_json::from_reader(std::fs::File::open(reference)?)?;
    let cases = cases.as_array().ok_or("expected an array of cases")?;
    let depth = [ModelOpenOption::MtpDraftTokens(i32::try_from(drafts)?)];
    let model = Model::open_with_support_options(
        path,
        Backend::Cuda,
        8,
        true,
        drafter,
        None,
        if drafter.is_some() { &depth } else { &[] },
    )?;
    if model.family() != ModelFamily::Hy4 {
        return Err("Hy4 release required".into());
    }
    let mut writer = BufWriter::new(std::fs::File::create(output)?);
    for case in cases {
        let ids: Vec<i32> = serde_json::from_value(case["ids"].clone())?;
        let positions = case["positions"]
            .as_array()
            .ok_or("missing teacher positions")?;
        let mut session = model.session(i32::try_from(ids.len())?)?;
        let mut consumed = 0;
        for reference in positions {
            let at = usize::try_from(reference["position"].as_u64().ok_or("missing position")?)?;
            if at + 1 >= ids.len() || at + 1 <= consumed {
                return Err("invalid teacher frontier".into());
            }
            let (mut verify_rows, mut accepted) = (0, 0);
            if at == consumed && consumed != 0 && drafter.is_some() {
                verify_rows = (ids.len() - at).min(drafts + 1);
                // The trial's own prefix (teacher token plus matching drafts)
                // measures draft acceptance on fixed text; only the teacher
                // token is committed.
                let kept = session.eval_speculative(
                    ids[at],
                    verify_rows as i32,
                    model.token_eos(),
                    Sampler::GREEDY,
                    &mut 0,
                    &mut |prefix| {
                        accepted = prefix.len();
                        1
                    },
                )?;
                if kept != [ids[at]] {
                    return Err("verify did not commit the teacher token".into());
                }
            } else if at == consumed && consumed != 0 {
                session.eval(ids[at])?;
            } else {
                session.sync(&TokenBuffer::from_tokens(ids[..=at].to_vec()))?;
            }
            consumed = at + 1;
            let logits = session.copy_logits(120832)?;
            if !logits.iter().all(|x| x.is_finite()) {
                return Err("nonfinite target logits".into());
            }
            let (top1, &peak) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .ok_or("empty logits")?;
            let lse = f64::from(peak)
                + logits
                    .iter()
                    .map(|&x| f64::from(x - peak).exp())
                    .sum::<f64>()
                    .ln();
            let next = usize::try_from(ids[at + 1])?;
            let nll = lse - f64::from(*logits.get(next).ok_or("next token out of range")?);
            let row = json!({"case":case["case"],"position":at,"token":ids[at+1],"nll":nll,"top1":top1,
                "verify_rows":verify_rows,"accepted":accepted,"teacher_nll":reference["nll"],"teacher_top1":reference["top1"]});
            serde_json::to_writer(&mut writer, &row)?;
            writeln!(writer)?;
        }
        writer.flush()?;
    }
    Ok(())
}
