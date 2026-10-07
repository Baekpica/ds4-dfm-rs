//! Compare real accepted prefixes with ordinary width-one target transitions.
use ds4_core::{Backend, Model, ModelFamily, ModelOpenOption, SessionSnapshot};
use serde_json::json;
use std::io::Read;
use std::path::Path;

const SHORT_CTX: i32 = 2048;
const LONG_CTX: i32 = 8192;
const OUTPUT_TOKENS: i32 = 64;
const TRIAL_ROWS: i32 = 4;
const MULTI_MIN: usize = 2;
const CACHE_BYTES: u64 = 24 << 30;
const IO_BYTES: usize = 65536;

enum Input {
    Prompt,
    Payload,
}

fn same_files(a: &Path, b: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut left = std::fs::File::open(a)?;
    let mut right = std::fs::File::open(b)?;
    assert_eq!(left.metadata()?.len(), right.metadata()?.len());
    let mut x = [0u8; IO_BYTES];
    let mut y = [0u8; IO_BYTES];
    let mut offset = 0;
    loop {
        let n = left.read(&mut x)?;
        if n == 0 {
            return Ok(());
        }
        right.read_exact(&mut y[..n])?;
        if x[..n] != y[..n] {
            let first = x[..n]
                .iter()
                .zip(&y[..n])
                .position(|(a, b)| a != b)
                .unwrap();
            return Err(format!("state differs at byte {}", offset + first).into());
        }
        offset += n;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3
        || args[3..]
            .iter()
            .any(|arg| !matches!(arg.as_str(), "--resident" | "--payload"))
    {
        return Err("usage: glm53_mtp_gate MODEL INPUT OUTPUT_DIR [--resident] [--payload]".into());
    }
    let input = if args[3..].iter().any(|arg| arg == "--payload") {
        Input::Payload
    } else {
        Input::Prompt
    };
    let ctx = match input {
        Input::Prompt => SHORT_CTX,
        Input::Payload => LONG_CTX,
    };
    let mut options = vec![
        ModelOpenOption::MtpDraftTokens(TRIAL_ROWS - 1),
        ModelOpenOption::MtpMargin(0.0),
    ];
    if args[3..].iter().any(|arg| arg == "--resident") {
        if std::env::var_os("DS4_CUDA_WEIGHT_IPC_MANIFEST").is_none_or(|v| v.is_empty()) {
            return Err("--resident requires a weight owner manifest".into());
        }
    } else {
        options.extend([
            ModelOpenOption::SsdStreaming,
            ModelOpenOption::SsdCacheBytes(CACHE_BYTES),
        ]);
    }
    let out = Path::new(&args[2]);
    std::fs::create_dir(out)?;
    let model = Model::open_configured(&args[0], Backend::Cuda, 8, true, None, &options)?;
    assert_eq!(model.family(), ModelFamily::Glm53);
    let mut session = model.session(ctx)?;
    match input {
        Input::Prompt => {
            session.sync(&model.tokenize_rendered_chat(&std::fs::read_to_string(&args[1])?)?)?
        }
        // The launcher pins the artifact and checkpoint provenance. This is
        // an explicit local import, not automatic cross-runtime compatibility.
        Input::Payload => session.load_payload(&args[1])?,
    }
    let prefix = session.pos();
    assert!(prefix > 0 && prefix + OUTPUT_TOKENS <= ctx);
    let mut history = session.host().tokens().to_vec();
    let baseline = out.join("baseline.payload.bin");
    let restored = out.join("restored.payload.bin");
    let committed = out.join("committed.payload.bin");
    let target = out.join("target.payload.bin");
    let mut snapshot = SessionSnapshot::new()?;
    let mut generated = Vec::new();
    let mut counts = Vec::new();
    while generated.len() < OUTPUT_TOKENS as usize {
        session.save_payload(&baseline)?;
        session.save_snapshot(&mut snapshot)?;
        let first = session.argmax();
        assert!(first >= 0);
        let budget = OUTPUT_TOKENS - generated.len() as i32;
        let accepted = session.eval_speculative_argmax(first, budget, model.token_eos())?;
        assert!(!accepted.is_empty() && accepted.len() <= TRIAL_ROWS as usize);
        let logits = session.copy_logits(model.vocab().n_vocab() as usize)?;
        assert!(logits.iter().all(|value| value.is_finite()));
        session.save_payload(&committed)?;

        // Restore plumbing is checked separately. Both transitions retain
        // MTP state, so the predictor cache is part of the byte comparison.
        session.load_snapshot(&snapshot)?;
        session.save_payload(&restored)?;
        same_files(&baseline, &restored)?;
        for &token in &accepted {
            assert_eq!(session.argmax(), token, "draft differs from target argmax");
            session.eval(token)?;
        }
        session.save_payload(&target)?;
        same_files(&committed, &target)?;
        let replay = session.copy_logits(logits.len())?;
        assert!(logits
            .iter()
            .zip(replay)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        history.extend_from_slice(&accepted);
        assert_eq!(session.host().tokens(), history);
        assert_eq!(session.pos() as usize, history.len());
        assert_eq!(session.generation(), session.native_generation());
        counts.push(accepted.len());
        generated.extend_from_slice(&accepted);
        if accepted.iter().any(|&token| model.token_is_stop(token)) {
            break;
        }
    }
    let max_accepted = counts.iter().copied().max().unwrap_or(0);
    let memory = ds4_core::snapshot_mem();
    assert!(memory.census.supported);
    assert_eq!(memory.census.faults, 0);
    let mut text = Vec::new();
    for &token in &generated {
        if !model.token_is_stop(token) {
            text.extend(model.token_text(token)?);
        }
    }
    println!(
        "{}",
        json!({"result": if max_accepted >= MULTI_MIN { "PASS" } else { "NO_MULTI_TOKEN_ACCEPTANCE" },
        "context":ctx, "prompt_tokens":prefix, "generated_ids":generated, "accepted_counts":counts,
        "generated_text":String::from_utf8(text)?,
        "max_accepted":max_accepted, "snapshot_restore":"byte_exact", "target_state":"byte_exact",
        "target_argmax":"all_ids_equal", "target_logits":"all_f32_bits_equal", "census_faults":memory.census.faults})
    );
    if max_accepted < MULTI_MIN {
        return Err("prompt did not exercise a multi-token accepted prefix".into());
    }
    Ok(())
}
