//! Gemma-4 MTP losslessness: drafting changes acceptance and speed,
//! never the target's token stream. The trial lives in
//! `ds4_core::gemma4_mtp_gate` so `--serving-check` runs the same code:
//! greedy off/on, the state after the first multi-token block against a
//! one-row replay, prefix reuse, snapshot restore, a stop inside an accepted
//! block, and sample-and-match at temperature 0.8.
//!
//! The exact-prefill run (`DS4_GEMMA4_EXACT_PREFILL=1`) also requires the
//! block and stop payloads to be byte-identical; the default batched path
//! reports the comparison, because its verify projections take linear_many,
//! the known chunk trade. Every run requires the token streams and the
//! frontier argmax to agree.
//!
//! usage: gemma4_mtp_gate ARTIFACT MTP_GGUF PROMPT_TXT [OUTPUT_TOKENS] [DRAFTS]

use ds4_core::{gemma4_mtp_gate, Backend, Model, ModelFamily, ModelOpenOption};
use serde_json::json;

const OUTPUT: usize = 24;
const DRAFTS: i32 = 6;

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Fallible<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=5).contains(&args.len()) {
        return Err(
            "usage: gemma4_mtp_gate ARTIFACT MTP_GGUF PROMPT_TXT [OUTPUT_TOKENS] [DRAFTS]".into(),
        );
    }
    let output: usize = args
        .get(3)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(OUTPUT);
    let drafts: i32 = args
        .get(4)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(DRAFTS);
    if output < 8 {
        return Err("OUTPUT_TOKENS must be at least 8".into());
    }
    let text = std::fs::read_to_string(&args[2])?;
    let model = Model::open_with_support_options(
        &args[0],
        Backend::Cuda,
        8,
        true,
        Some(args[1].as_str()),
        None,
        &[ModelOpenOption::MtpDraftTokens(drafts)],
    )?;
    if model.family() != ModelFamily::Gemma4Vq {
        return Err("this gate requires the Gemma-4 VQ release".into());
    }
    if model.mtp().is_none() {
        return Err("the drafter sidecar did not bind; pass the mtp GGUF".into());
    }
    let prompt = model.tokenize_text(&text)?;
    let need = prompt.len() + output + 64;
    let ctx = i32::try_from((need + 255) / 256 * 256)?.max(1024);
    let exact = std::env::var_os("DS4_GEMMA4_EXACT_PREFILL").is_some();

    eprintln!(
        "gemma4 mtp gate: prompt {} tokens, ctx {ctx}, drafts {drafts}, exact {exact}",
        prompt.len()
    );
    let report = gemma4_mtp_gate(&model, prompt.as_slice(), ctx, output)?;
    let block = report
        .block
        .as_ref()
        .ok_or("no accepted block of three or more tokens; raise OUTPUT_TOKENS")?;
    if report.prefix_reuse.is_none() {
        eprintln!(
            "gemma4 mtp gate: prefix reuse skipped (the prompt is shorter than one prefill chunk; use a longer prompt or DS4_GEMMA4_EXACT_PREFILL=1)"
        );
    } else {
        eprintln!(
            "gemma4 mtp gate: prefix reuse equal, {} tokens",
            report.ids.len()
        );
    }
    eprintln!(
        "gemma4 mtp gate: ids equal, {} tokens, {:.2} tokens/call, accepted {:?}",
        report.ids.len(),
        report.tokens_per_call(),
        report.counts
    );
    eprintln!(
        "gemma4 mtp gate: block {:?} state max_abs {:.3e}, relL2 {:.3e}, payload_equal {}",
        block.tokens, block.max_abs, block.rel_l2, block.payload_equal
    );
    eprintln!(
        "gemma4 mtp gate: stop inside block accepted {:?}, max_abs {:.3e}, payload_equal {}",
        block.stop_accepted, block.stop_max_abs, block.stop_payload_equal
    );
    eprintln!(
        "gemma4 mtp gate: sampling equal at temp 0.8, {} tokens, {:.2} tokens/call",
        report.sampling_ids.len(),
        report.sampling_ids.len() as f64 / report.sampling_counts.len().max(1) as f64
    );

    println!(
        "{}",
        json!({
            "result": "PASS",
            "context": ctx,
            "prompt_tokens": prompt.len(),
            "drafts": drafts,
            "exact_prefill": exact,
            "greedy_ids": report.ids,
            "greedy_tokens_per_call": report.tokens_per_call(),
            "accepted_counts": report.counts,
            "state_frontier": block.frontier,
            "state_block": block.tokens,
            "state_max_abs": block.max_abs,
            "state_relL2": block.rel_l2,
            "state_payload_equal": block.payload_equal,
            "stop_accepted": block.stop_accepted,
            "stop_max_abs": block.stop_max_abs,
            "stop_relL2": block.stop_rel_l2,
            "stop_payload_equal": block.stop_payload_equal,
            "prefix_reuse": if report.prefix_reuse == Some(true) { "equal_ids" } else { "skipped" },
            "snapshot_restore": "equal_ids",
            "sampling_temperature": 0.8,
            "sampling_tokens": report.sampling_ids,
        })
    );
    Ok(())
}
