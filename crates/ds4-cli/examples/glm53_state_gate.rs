//! Exercise Rust's serial payload and snapshot surfaces with one live model.
use ds4_core::{
    Backend, Model, ModelFamily, ModelOpenOption, Session, SessionSnapshot, TokenBuffer,
};
use serde_json::json;
use std::io::{Read, Write};
use std::path::Path;

const DEFAULT_CONTEXT: i32 = 2048;
const ANSWER_TOKENS: usize = 128;
const APPEND_ROWS: [usize; 4] = [1, 16, 32, 128];
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
    if args.len() < 3 {
        return Err("usage: glm53_state_gate MODEL PROMPT OUT [--resident] [--ctx N]".into());
    }
    let mut weights = Weights::Ssd;
    let mut context = DEFAULT_CONTEXT;
    let mut tail = args[3..].iter();
    while let Some(arg) = tail.next() {
        match arg.as_str() {
            "--resident" => weights = Weights::Resident,
            "--ctx" => context = tail.next().ok_or("--ctx needs N")?.parse()?,
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    let mut options = vec![
        ModelOpenOption::MtpDraftTokens(3),
        ModelOpenOption::MtpMargin(0.0),
        ModelOpenOption::ServingBudget(ds4_core::ServingRequest {
            ctx: context,
            max_seqs: ds4_core::MaxSeqs::Off,
            ..ds4_core::ServingRequest::default()
        }),
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
    assert!(context > 0 && !prompt.is_empty() && prompt.len() + ANSWER_TOKENS < context as usize);
    let mut session = model.session(context)?;
    let prefill = std::time::Instant::now();
    session.sync(&prompt)?;
    let prefill_ms = prefill.elapsed().as_secs_f64() * 1000.0;
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
    // Fix append lengths while replaying the same committed prefix. API tool
    // grammar is checked separately; this gate isolates state and supply.
    let tool_text = model.tokenize_rendered_chat("<|observation|>Search completed. The result contains the requested record. Continue using the recorded data.")?;
    let mut appends = Vec::new();
    for n in APPEND_ROWS {
        session.load_snapshot(&snapshot)?;
        let mut joined = prompt.as_slice().to_vec();
        joined.extend(tool_text.as_slice().iter().copied().cycle().take(n));
        let joined = TokenBuffer::from_tokens(joined);
        let start = std::time::Instant::now();
        session.sync(&joined)?;
        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
        let reference = session.copy_logits(logits.len())?;
        check_frontier(&session, joined.as_slice(), &reference);
        let first = session.argmax();
        let decode = std::time::Instant::now();
        session.eval(first)?;
        let decode_ms = decode.elapsed().as_secs_f64() * 1000.0;
        session.load_snapshot(&snapshot)?;
        session.sync(&joined)?;
        check_frontier(&session, joined.as_slice(), &reference);
        appends
            .push(json!({"rows":n,"ttft_ms":ttft_ms,"decode_ms":decode_ms,"replay":"byte_exact"}));
    }
    session.load_snapshot(&snapshot)?;
    let mut answer = Vec::new();
    let mut answer_ids = Vec::new();
    for _ in 0..ANSWER_TOKENS {
        let token = session.argmax();
        if model.token_is_stop(token) {
            break;
        }
        answer_ids.push(token);
        answer.extend(model.token_text(token)?);
        session.eval(token)?;
    }
    std::fs::write(out.join("answer.txt"), &answer)?;
    std::fs::write(
        out.join("answer.tokens.json"),
        serde_json::to_vec(&answer_ids)?,
    )?;

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
        json!({"result": "PASS", "context": context,
        "prompt_tokens": prompt.len(), "prefill_ms": prefill_ms, "accepted": accepted,
        "file_snapshot_range": "byte_exact", "transition_replay": "byte_exact",
        "payload_bytes": bytes, "appends":appends,"answer_tokens":answer_ids.len(),
        "census_faults": memory.census.faults})
    );
    Ok(())
}
