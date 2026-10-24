//! MiniMax DSpark restore gate, one model and sequential sessions.
//! Set DS4_MINIMAX_DSPARK_GATE=0. Run under the host memory guard:
//! minimax_state MODEL PROMPT_FILE OUTPUT_DIR MODE [PREFIX] [OUTPUT_TOKENS]
//! MODE is --legacy-expect-fallback (MMX1 has no draft context) or --preserve
//! (restores must reproduce the uninterrupted draft cycles and target logits).
use ds4_core::{
    snapshot_spec, Backend, Model, ModelFamily, ModelOpenOption, Session, SessionSnapshot,
    TokenBuffer,
};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const PREFIX: usize = 8192;
const OUTPUT_TOKENS: usize = 32;
const DRAFTS: i32 = 3;
const CONTEXT_MARGIN: usize = 512;
const HEADER_WORDS: usize = 13;
const LAYOUT_WORD: usize = 5;
const MMX1: u32 = u32::from_le_bytes(*b"MMX1");

#[derive(Clone, Copy)]
enum Mode {
    Legacy,
    Preserve,
}

struct Cycle {
    tokens: Vec<i32>,
    logits: Vec<u32>,
    drafts: u64,
    hits: u64,
}

fn logits(session: &Session<'_>, vocab: usize) -> Result<Vec<u32>> {
    let values = session.copy_logits(vocab)?;
    if values.len() != vocab || values.iter().any(|v| !v.is_finite()) {
        return Err("missing or nonfinite full-vocabulary logits".into());
    }
    Ok(values.into_iter().map(f32::to_bits).collect())
}

fn step(session: &mut Session<'_>, model: &Model, budget: usize) -> Result<Cycle> {
    let first = session.argmax();
    let before = snapshot_spec();
    let tokens = session.eval_speculative_argmax(first, budget as i32, model.token_eos())?;
    let after = snapshot_spec();
    if tokens.is_empty() || tokens[0] != first || tokens.len() > budget {
        return Err("invalid speculative committed prefix".into());
    }
    Ok(Cycle {
        tokens,
        logits: logits(session, model.vocab().n_vocab() as usize)?,
        drafts: after.drafts - before.drafts,
        hits: after.hits - before.hits,
    })
}

fn generate(session: &mut Session<'_>, model: &Model, count: usize) -> Result<Vec<Cycle>> {
    let mut cycles = Vec::new();
    let mut remaining = count;
    while remaining > 0 {
        let cycle = step(session, model, remaining)?;
        remaining -= cycle.tokens.len();
        let stopped = cycle.tokens.iter().any(|&t| model.token_is_stop(t));
        cycles.push(cycle);
        if stopped {
            break;
        }
    }
    Ok(cycles)
}

fn summary(cycles: &[Cycle]) -> Value {
    json!({
        "tokens": cycles.iter().flat_map(|c| &c.tokens).collect::<Vec<_>>(),
        "cycle_lengths": cycles.iter().map(|c| c.tokens.len()).collect::<Vec<_>>(),
        "drafts": cycles.iter().map(|c| c.drafts).collect::<Vec<_>>(),
        "hits": cycles.iter().map(|c| c.hits).collect::<Vec<_>>(),
    })
}

fn same_prefix(session: &Session<'_>, prompt: &[i32], frontier: &[u32]) -> Result<()> {
    if session.pos() as usize != prompt.len() || session.host().tokens() != prompt {
        return Err("restored token frontier differs".into());
    }
    if session.generation() != session.native_generation() {
        return Err("host/native generation differs".into());
    }
    if logits(session, frontier.len())? != frontier {
        return Err("restored frontier logits differ".into());
    }
    Ok(())
}

fn contaminate(session: &mut Session<'_>, prompt: &TokenBuffer) -> Result<()> {
    // A different full history overwrites the 4K draft rings before restoring A.
    session.invalidate();
    session.sync(prompt)?;
    Ok(())
}

fn legacy_probe(
    session: &mut Session<'_>,
    model: &Model,
    label: &str,
    expected: &Cycle,
) -> Result<Value> {
    let actual = step(session, model, (DRAFTS + 1) as usize)?;
    let pass = actual.drafts == 0
        && actual.hits == 0
        && actual.tokens == expected.tokens
        && actual.logits == expected.logits;
    let row = json!({"case": label, "pass": pass, "drafts": actual.drafts,
        "hits": actual.hits, "tokens": actual.tokens,
        "expected_tokens": expected.tokens,
        "logit_differences": actual.logits.iter().zip(&expected.logits).filter(|(a,b)| a != b).count()});
    println!("{row}");
    std::io::stdout().flush()?;
    Ok(row)
}

fn preserve_probe(
    session: &mut Session<'_>,
    model: &Model,
    label: &str,
    expected: &[Cycle],
    count: usize,
) -> Result<Value> {
    let actual = generate(session, model, count)?;
    let cycle_match = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.tokens == b.tokens && a.drafts == b.drafts && a.hits == b.hits);
    let logit_match = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.logits == b.logits);
    let row = json!({"case": label, "pass": cycle_match && logit_match,
        "cycle_match": cycle_match, "logit_match": logit_match, "actual": summary(&actual)});
    println!("{row}");
    std::io::stdout().flush()?;
    Ok(row)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
        return Err(
            "usage: minimax_state MODEL PROMPT_FILE OUTPUT_DIR MODE [PREFIX] [OUTPUT_TOKENS]"
                .into(),
        );
    }
    let mode = match args[3].as_str() {
        "--legacy-expect-fallback" => Mode::Legacy,
        "--preserve" => Mode::Preserve,
        _ => return Err("MODE must be --legacy-expect-fallback or --preserve".into()),
    };
    if std::env::var("DS4_MINIMAX_DSPARK_GATE").as_deref() != Ok("0") {
        return Err("set DS4_MINIMAX_DSPARK_GATE=0 for deterministic draft cycles".into());
    }
    // Native getenv treats even an empty value or "0" as disabled.
    if std::env::var_os("DS4_MTP_SPEC_DISABLE").is_some() {
        return Err("unset DS4_MTP_SPEC_DISABLE; disabled drafting makes this gate vacuous".into());
    }
    let prefix: usize = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(PREFIX);
    let count: usize = args
        .get(5)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(OUTPUT_TOKENS);
    if prefix < 2 || count < (DRAFTS + 1) as usize {
        return Err("prefix needs at least 2 tokens and output needs at least 4".into());
    }
    let context = i32::try_from(
        prefix
            .checked_add(count)
            .and_then(|v| v.checked_add(CONTEXT_MARGIN))
            .ok_or("context overflow")?,
    )?;
    let out = Path::new(&args[2]);
    std::fs::create_dir(out)?;
    let model = Model::open_configured(
        &args[0],
        Backend::Cuda,
        8,
        true,
        None,
        &[ModelOpenOption::MtpDraftTokens(DRAFTS)],
    )?;
    if model.family() != ModelFamily::MinimaxVq {
        return Err("this gate requires MiniMax-M3.1 VQ".into());
    }
    let input = model.tokenize_text(&std::fs::read_to_string(&args[1])?)?;
    if input.len() < prefix {
        return Err("prompt corpus is shorter than PREFIX".into());
    }
    let a = input.as_slice()[..prefix].to_vec();
    let mut b = a.clone();
    b.rotate_left(prefix / 2);
    if a == b {
        return Err("corpus does not provide distinct contamination history".into());
    }
    let b = TokenBuffer::from_tokens(b);
    let payload = out.join("a.payload.bin");
    let mut snapshot = SessionSnapshot::new()?;
    let mut session = model.session(context)?;
    session.sync(&TokenBuffer::from_tokens(a.clone()))?;
    let frontier = logits(&session, model.vocab().n_vocab() as usize)?;
    session.save_snapshot(&mut snapshot)?;
    session.save_payload(&payload)?;
    let mut header = [0u8; HEADER_WORDS * 4];
    std::fs::File::open(&payload)?.read_exact(&mut header)?;
    let layout = u32::from_le_bytes(header[LAYOUT_WORD * 4..LAYOUT_WORD * 4 + 4].try_into()?);
    if matches!(mode, Mode::Legacy) && layout != MMX1 {
        return Err(
            "legacy mode requires a writer producing MMX1; no guessed format conversion".into(),
        );
    }
    let baseline = match mode {
        Mode::Legacy => {
            let first = session.argmax();
            session.eval(first)?;
            vec![Cycle {
                tokens: vec![first],
                logits: logits(&session, frontier.len())?,
                drafts: 0,
                hits: 0,
            }]
        }
        Mode::Preserve => generate(&mut session, &model, count)?,
    };
    if matches!(mode, Mode::Preserve) && baseline.iter().all(|c| c.drafts == 0) {
        return Err("baseline never drafted; restore test would be vacuous".into());
    }
    println!(
        "{}",
        json!({"case":"baseline", "layout":format!("{layout:08x}"), "prefix":prefix,
        "context":context, "snapshot_bytes":snapshot.len(), "continuation":summary(&baseline)})
    );
    let probe = |session: &mut Session<'_>, label: &str| -> Result<Value> {
        same_prefix(session, &a, &frontier)?;
        match mode {
            Mode::Legacy => legacy_probe(session, &model, label, &baseline[0]),
            Mode::Preserve => preserve_probe(session, &model, label, &baseline, count),
        }
    };
    let mut rows = Vec::new();
    contaminate(&mut session, &b)?;
    session.load_snapshot(&snapshot)?;
    rows.push(probe(&mut session, "contaminated_snapshot")?);
    contaminate(&mut session, &b)?;
    session.load_payload(&payload)?;
    rows.push(probe(&mut session, "contaminated_payload")?);
    drop(session);
    {
        let mut fresh = model.session(context)?;
        fresh.load_payload(&payload)?;
        rows.push(probe(&mut fresh, "fresh_payload")?);
    }
    {
        let mut fresh = model.session(context)?;
        fresh.load_snapshot(&snapshot)?;
        rows.push(probe(&mut fresh, "fresh_snapshot")?);
    }
    let pass = rows.iter().all(|r| r["pass"] == true);
    let report = json!({"pass":pass, "mode":args[3], "prefix":prefix, "context":context,
        "baseline":summary(&baseline), "cases":rows});
    std::fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if !pass {
        return Err("MiniMax draft state restore gate failed; see report.json".into());
    }
    Ok(())
}
