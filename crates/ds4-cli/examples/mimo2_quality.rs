//! Compare long-context answers through the production host and chat template.
//! Usage: mimo2_quality MODEL CASES.json OUTPUT_DIR [padding_tokens] [draft_tokens] [dflash]
use ds4_core::{Backend, Model, ModelFamily, ModelOpenOption};
use serde_json::{json, Value};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const NO_THINKING: i32 = 0;
    const MAX_OUTPUT: i32 = 128;
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=6).contains(&args.len()) {
        return Err("usage: mimo2_quality MODEL CASES.json OUTPUT_DIR [padding_tokens] [draft_tokens] [dflash]".into());
    }
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let out = std::path::Path::new(&args[2]);
    std::fs::create_dir(out)?;
    let draft: i32 = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(1);
    if !(1..=8).contains(&draft) {
        return Err("draft_tokens must be in 1..=8".into());
    }
    let dflash = args.get(5).map(String::as_str);
    let model = Model::open_with_support_options(
        &args[0],
        Backend::Cuda,
        0,
        false,
        dflash,
        None,
        &[ModelOpenOption::MtpDraftTokens(draft)],
    )?;
    if model.family() != ModelFamily::Mimo2 {
        return Err("this fixture is scoped to MiMo".into());
    }
    let padding: usize = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(0);
    // Raw corpus padding is identical across both arms; each case starts a
    // fresh session so a reused prompt cannot bypass the attention under test.
    let corpus = std::fs::read("speed-bench/promessi_sposi.txt")?;
    let ids = model.vocab().encode_bytes(&corpus);
    if padding > ids.len() {
        return Err("padding exceeds the corpus".into());
    }
    for (index, case) in cases.iter().enumerate() {
        let question = case["prompt"].as_str().ok_or("case needs prompt")?;
        let mut background = Vec::new();
        let mut start = 0;
        // Insert facts at declared corpus offsets, so long-context checks
        // exercise retrieval as well as following an instruction at the end.
        if let Some(records) = case["records"].as_array() {
            for record in records {
                let at =
                    usize::try_from(record["at_token"].as_u64().ok_or("record needs at_token")?)?;
                if at < start || at > padding {
                    return Err("record offsets must be ordered and within padding".into());
                }
                for &token in &ids[start..at] {
                    background.extend(model.token_text(token)?);
                }
                background.extend_from_slice(b"\n\n");
                background.extend_from_slice(
                    record["text"]
                        .as_str()
                        .ok_or("record needs text")?
                        .as_bytes(),
                );
                background.extend_from_slice(b"\n\n");
                start = at;
            }
        }
        for &token in &ids[start..padding] {
            background.extend(model.token_text(token)?);
        }
        let background = String::from_utf8_lossy(&background);
        let prompt = format!("Reference material:\n{background}\n\nTask:\n{question}");
        let tokens = model.encode_chat_prompt_bytes(
            Some(b"Follow the final task. Answer directly without analysis."),
            prompt.as_bytes(),
            NO_THINKING,
        )?;
        let context = i32::try_from(tokens.len())? + MAX_OUTPUT + 1;
        let mut session = model.session(context)?;
        session.sync(&tokens)?;
        let logits = session.copy_logits(model.vocab().n_vocab() as usize)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(format!("non-finite logits in case {index}").into());
        }
        let argmax = session.argmax();
        let mut generated = Vec::new();
        let mut answer = Vec::new();
        let mut stopped = false;
        let mut cycles = Vec::new();
        while generated.len() < MAX_OUTPUT as usize {
            let token = session.argmax();
            if model.token_is_stop(token) {
                stopped = true;
                break;
            }
            let before = session.pos();
            let remaining = MAX_OUTPUT - generated.len() as i32;
            let accepted = if draft > 1 {
                session.eval_speculative_argmax(token, remaining, model.token_eos())?
            } else {
                session.eval(token)?;
                vec![token]
            };
            if accepted.is_empty()
                || accepted.len() > remaining as usize
                || session.pos() != before + accepted.len() as i32
            {
                return Err("invalid accepted-prefix transition".into());
            }
            // A one-token result can be either rejection or a plain fallback;
            // retain the native trace to distinguish actual draft activation.
            cycles.push(json!({"before": before, "after": session.pos(), "tokens": accepted}));
            for token in accepted {
                if model.token_is_stop(token) {
                    stopped = true;
                    break;
                }
                generated.push(token);
                answer.extend(model.token_text(token)?);
            }
            if stopped {
                break;
            }
        }
        let result = json!({
            "case": case, "prompt_tokens": tokens.len(), "argmax_id": argmax,
            "stopped": stopped, "tokens": generated,
            "answer": String::from_utf8_lossy(&answer), "logits": logits,
            "draft_tokens_requested": draft, "dflash": dflash, "cycles": cycles,
        });
        std::fs::write(
            out.join(format!("case-{index}.json")),
            serde_json::to_vec(&result)?,
        )?;
        println!(
            "{}",
            json!({"case": index, "prompt_tokens": tokens.len(),
            "stopped": stopped, "answer": result["answer"]})
        );
        std::io::stdout().flush()?;
    }
    Ok(())
}
