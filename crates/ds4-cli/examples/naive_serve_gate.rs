//! Live target/draft generation and cache gates through the production host.
use ds4_core::{Backend, Model, ModelFamily, ModelOpenOption, SessionSnapshot, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;

type GateResult<T> = Result<T, Box<dyn std::error::Error>>;
enum DraftMode {
    Off,
    On,
}

fn prompt(model: &Model, case: &Value) -> GateResult<TokenBuffer> {
    if let Some(path) = case["tokens_file"].as_str() {
        let raw = std::fs::read(path)?;
        if raw.len() % 4 != 0 {
            return Err("unaligned token file".into());
        }
        return Ok(TokenBuffer::from_tokens(
            raw.chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ));
    }
    let text = case["prompt"]
        .as_str()
        .ok_or("case needs prompt or tokens_file")?;
    Ok(model.encode_chat_prompt(Some("You are a helpful assistant."), text, 0)?)
}

fn run_case(model: &Model, case: &Value, mode: DraftMode) -> GateResult<Value> {
    let input = prompt(model, case)?;
    if let Some(path) = case["prompt_ids_file"].as_str() {
        let mut file = std::fs::File::create(path)?;
        for token in input.as_slice() {
            file.write_all(&token.to_le_bytes())?;
        }
    }
    let ctx = case["ctx"].as_i64().unwrap_or(8192);
    let limit = case["generate"].as_u64().unwrap_or(128) as usize;
    let mut session = model.session(i32::try_from(ctx)?)?;
    let started = std::time::Instant::now();
    session.sync(&input)?;
    let prefill = started.elapsed().as_secs_f64();
    if session.pos() != input.len() as i32 {
        return Err("incorrect prefill frontier".into());
    }
    eprintln!(
        "prefill id={} tokens={} seconds={prefill:.3}",
        case["id"],
        input.len()
    );

    // Read back complete logits before and after both native cache APIs.
    let mut cache_checked = false;
    if let Some(path) = case["cache_path"].as_str() {
        let logits = session.copy_logits(model.vocab().n_vocab() as usize)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("non-finite prefill logits".into());
        }
        session.save_payload(path)?;
        let mut snapshot = SessionSnapshot::new()?;
        session.save_snapshot(&mut snapshot)?;
        session.eval(session.argmax())?;
        session.load_snapshot(&snapshot)?;
        if logits != session.copy_logits(logits.len())? {
            return Err("snapshot changed logits".into());
        }
        session.invalidate();
        session.load_payload(path)?;
        if session.pos() != input.len() as i32 || logits != session.copy_logits(logits.len())? {
            return Err("disk payload changed frontier or logits".into());
        }
        cache_checked = true;
    }

    let started = std::time::Instant::now();
    let mut tokens = Vec::new();
    let mut widths = Vec::new();
    let mut text = Vec::new();
    let mut stopped = false;
    while tokens.len() < limit {
        let first = session.argmax();
        if first < 0 {
            return Err("invalid generation logits".into());
        }
        if model.token_is_stop(first) {
            stopped = true;
            break;
        }
        let before = session.pos();
        let accepted = match mode {
            DraftMode::On => session.eval_speculative_argmax(
                first,
                i32::try_from(limit - tokens.len())?,
                model.token_eos(),
            )?,
            DraftMode::Off => {
                session.eval(first)?;
                vec![first]
            }
        };
        if accepted.is_empty()
            || accepted[0] != first
            || accepted.len() > limit - tokens.len()
            || session.pos() != before + accepted.len() as i32
        {
            return Err("invalid accepted frontier".into());
        }
        widths.push(accepted.len());
        for token in accepted {
            tokens.push(token);
            if model.token_is_stop(token) {
                stopped = true;
                break;
            }
            text.extend(model.token_text(token)?);
        }
        if stopped {
            break;
        }
    }
    let decode = started.elapsed().as_secs_f64();
    Ok(
        json!({"id": case["id"], "ctx": ctx, "prompt_tokens": input.len(),
        "mode": if matches!(mode, DraftMode::On) { "on" } else { "off" },
        "tokens": tokens, "text": String::from_utf8_lossy(&text), "accepted_widths": widths,
        "position": session.pos(), "stopped": stopped, "cache_checked": cache_checked,
        "prefill_seconds": prefill, "decode_seconds": decode}),
    )
}

fn main() -> GateResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: naive_serve_gate MODEL CASES.json REPORT.jsonl".into());
    }
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let model = Model::open_with_support_options(
        &args[0],
        Backend::Cuda,
        0,
        true,
        None,
        None,
        &[
            ModelOpenOption::MtpDraftTokens(6),
            ModelOpenOption::MtpMargin(0.0),
        ],
    )?;
    if model.family() != ModelFamily::NaiveN05 {
        return Err("gate needs the Naive target".into());
    }
    let mut report = std::fs::File::create(&args[2])?;
    for case in cases {
        let mode = match case["mode"].as_str() {
            Some("on") => DraftMode::On,
            Some("off") => DraftMode::Off,
            _ => return Err("case mode must be on or off".into()),
        };
        if matches!(mode, DraftMode::On) && model.dspark().is_none() {
            return Err("on gate needs DS4_DSPARK_MODEL".into());
        }
        serde_json::to_writer(&mut report, &run_case(&model, &case, mode)?)?;
        report.write_all(b"\n")?;
        report.flush()?;
    }
    Ok(())
}
