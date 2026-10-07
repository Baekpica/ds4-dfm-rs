//! Exercise Rust's serial payload and snapshot surfaces with one live model.
use ds4_core::{Backend, Model, ModelFamily, ModelOpenOption, Session, SessionSnapshot};
use serde_json::json;
use std::io::{Read, Write};
use std::path::Path;

const CONTEXT: i32 = 2048;
const CACHE_BYTES: u64 = 24 << 30;
const IO_BYTES: usize = 65536;
const RANGE_PREFIX: usize = 73;
const RANGE_SUFFIX: usize = 29;

enum Weights {
    Ssd,
    Resident,
}

fn same_files(a: &Path, b: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut left = std::fs::File::open(a)?;
    let mut right = std::fs::File::open(b)?;
    assert_eq!(left.metadata()?.len(), right.metadata()?.len());
    let mut x = [0u8; IO_BYTES];
    let mut y = [0u8; IO_BYTES];
    loop {
        let n = left.read(&mut x)?;
        if n == 0 {
            return Ok(());
        }
        right.read_exact(&mut y[..n])?;
        assert_eq!(&x[..n], &y[..n], "serialized state differs");
    }
}

fn check_frontier(session: &Session<'_>, tokens: &[i32], logits: &[f32]) {
    assert_eq!(session.pos() as usize, tokens.len());
    assert_eq!(session.host().tokens(), tokens);
    assert_eq!(session.generation(), session.native_generation());
    let restored = session.copy_logits(logits.len()).unwrap();
    assert!(restored.iter().all(|v| v.is_finite()));
    assert!(restored
        .iter()
        .zip(logits)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let weights = match args.as_slice() {
        [_, _, _] => Weights::Ssd,
        [_, _, _, mode] if mode == "--resident" => Weights::Resident,
        _ => {
            return Err(
                "usage: glm53_state_gate MODEL RENDERED_PROMPT OUTPUT_DIR [--resident]".into(),
            );
        }
    };
    let mut options = vec![
        ModelOpenOption::MtpDraftTokens(3),
        ModelOpenOption::MtpMargin(0.0),
    ];
    match weights {
        Weights::Ssd => options.extend([
            ModelOpenOption::SsdStreaming,
            ModelOpenOption::SsdCacheBytes(CACHE_BYTES),
        ]),
        Weights::Resident => {
            // Refuse an accidental independent weight copy in the resident gate.
            if std::env::var_os("DS4_CUDA_WEIGHT_IPC_MANIFEST").is_none_or(|v| v.is_empty()) {
                return Err("--resident requires a weight owner manifest".into());
            }
        }
    }
    let text = std::fs::read_to_string(&args[1])?;
    let out = Path::new(&args[2]);
    std::fs::create_dir(out)?;
    let model = Model::open_configured(&args[0], Backend::Cuda, 8, true, None, &options)?;
    assert_eq!(model.family(), ModelFamily::Glm53);
    let prompt = model.tokenize_rendered_chat(&text)?;
    assert!(!prompt.is_empty() && prompt.len() < CONTEXT as usize - 4);
    let mut session = model.session(CONTEXT)?;
    session.sync(&prompt)?;
    let logits = session.copy_logits(model.vocab().n_vocab() as usize)?;
    check_frontier(&session, prompt.as_slice(), &logits);
    let baseline = out.join("baseline.payload.bin");
    session.save_payload(&baseline)?;
    let mut snapshot = SessionSnapshot::new()?;
    session.save_snapshot(&mut snapshot)?;
    assert!(snapshot.len() > 0);

    // Capture a real greedy/MTP transition, then replay from each restore
    // surface. Byte-exact self restore isolates plumbing from cross-width FP.
    let first = session.argmax();
    assert!(first >= 0);
    let accepted = session.eval_speculative_argmax(first, 4, model.token_eos())?;
    assert!(!accepted.is_empty() && accepted.len() <= 4 && accepted[0] == first);
    let mut history = prompt.as_slice().to_vec();
    history.extend_from_slice(&accepted);
    let end_logits = session.copy_logits(logits.len())?;
    check_frontier(&session, &history, &end_logits);
    let expected = out.join("transition.payload.bin");
    session.save_payload(&expected)?;

    session.load_payload(&baseline)?;
    check_frontier(&session, prompt.as_slice(), &logits);
    let restored = out.join("file-restored.payload.bin");
    session.save_payload(&restored)?;
    same_files(&baseline, &restored)?;
    assert_eq!(
        session.eval_speculative_argmax(first, 4, model.token_eos())?,
        accepted
    );
    check_frontier(&session, &history, &end_logits);
    let replay = out.join("file-replay.payload.bin");
    session.save_payload(&replay)?;
    same_files(&expected, &replay)?;

    session.load_snapshot(&snapshot)?;
    check_frontier(&session, prompt.as_slice(), &logits);
    let restored = out.join("snapshot-restored.payload.bin");
    session.save_payload(&restored)?;
    same_files(&baseline, &restored)?;

    // Bounded-range restore is the serial disk-KV path. The surrounding
    // bytes must not be consumed as native payload or host token history.
    let range = out.join("embedded.payload.bin");
    let mut file = std::fs::File::create(&range)?;
    file.write_all(&[0xa5; RANGE_PREFIX])?;
    let bytes = std::io::copy(&mut std::fs::File::open(&baseline)?, &mut file)?;
    file.write_all(&[0x5a; RANGE_SUFFIX])?;
    file.flush()?;
    session.eval(first)?;
    session.load_payload_range(&range, RANGE_PREFIX as u64, bytes)?;
    check_frontier(&session, prompt.as_slice(), &logits);
    let restored = out.join("range-restored.payload.bin");
    session.save_payload(&restored)?;
    same_files(&baseline, &restored)?;
    assert_eq!(
        session.eval_speculative_argmax(first, 4, model.token_eos())?,
        accepted
    );
    check_frontier(&session, &history, &end_logits);
    let replay = out.join("range-replay.payload.bin");
    session.save_payload(&replay)?;
    same_files(&expected, &replay)?;
    let memory = ds4_core::snapshot_mem();
    assert!(memory.census.supported);
    assert_eq!(memory.census.faults, 0);
    println!(
        "{}",
        json!({"result": "PASS", "context": CONTEXT,
        "prompt_tokens": prompt.len(), "accepted": accepted,
        "file_snapshot_range": "byte_exact", "transition_replay": "byte_exact",
        "payload_bytes": bytes, "census_faults": memory.census.faults})
    );
    Ok(())
}
