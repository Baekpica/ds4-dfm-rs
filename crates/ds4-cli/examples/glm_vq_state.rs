//! Bounded VQ snapshot/bank state gate, one model owner and forced tokens.
use ds4_core::{
    cont_sample_token, Backend, BatchCtx, ContAdmit, ContDone, ContDriver, Model, ModelFamily,
    SessionSnapshot, TokenBuffer,
};
use serde_json::json;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CONTEXT: i32 = 8192;
const PREFIX: usize = 6147;
const CUT: usize = 4097;
const GENERATED: i32 = 3;
const FORCED_TOKEN: i32 = 100;

fn same_file(a: &Path, b: &Path) -> Result<()> {
    let mut a = std::fs::File::open(a)?;
    let mut b = std::fs::File::open(b)?;
    if a.metadata()?.len() != b.metadata()?.len() {
        return Err("payload lengths differ".into());
    }
    let mut x = [0u8; 65536];
    let mut y = [0u8; 65536];
    loop {
        let n = a.read(&mut x)?;
        b.read_exact(&mut y[..n])?;
        if x[..n] != y[..n] {
            return Err("payload state/logits differ".into());
        }
        if n == 0 {
            return Ok(());
        }
    }
}

fn session_gate(model: &Model, prompt: &[i32], out: &Path) -> Result<()> {
    let mut session = model.session(CONTEXT)?;
    session.sync(&TokenBuffer::from_tokens(prompt[..CUT].to_vec()))?;
    let saved = out.join("session-saved.bin");
    let restored = out.join("session-restored.bin");
    session.save_payload(&saved)?;
    let mut snapshot = SessionSnapshot::new()?;
    session.save_snapshot(&mut snapshot)?;
    session.sync(&TokenBuffer::from_tokens(prompt.to_vec()))?;
    session.load_snapshot(&snapshot)?;
    assert_eq!(session.pos(), CUT as i32);
    session.save_payload(&restored)?;
    same_file(&saved, &restored)?;

    // Restore the same frontier after a different continuation. Byte equality
    // covers logits, KDA state, latent cache, pooled keys and partial tails.
    session.eval(FORCED_TOKEN)?;
    session.load_payload(&saved)?;
    session.save_payload(&restored)?;
    same_file(&saved, &restored)?;
    let before = session.copy_logits(model.vocab().n_vocab() as usize)?;
    {
        let mut dense = model.session(2048)?;
        dense.sync(&TokenBuffer::from_tokens(prompt[..1025].to_vec()))?;
        dense.save_payload(out.join("dense-only.bin"))?;
    }
    assert!(session.load_payload(out.join("dense-only.bin")).is_err());
    assert_eq!(session.pos(), CUT as i32);
    assert_eq!(before, session.copy_logits(before.len())?);
    println!(
        "{}",
        json!({"session_snapshot":"byte_exact", "disk_restore":"byte_exact",
        "dense_to_sparse":"rejected_without_mutation", "snapshot_bytes":snapshot.len()})
    );
    Ok(())
}

struct Driver<'a, 'm> {
    batch: &'a BatchCtx<'m>,
    request: Option<ContAdmit>,
    checkpoint: Option<PathBuf>,
    checkpoint_saved: bool,
    failure: Option<String>,
    admitted: Option<(i32, i32, i32)>,
    generated: Vec<i32>,
    done: bool,
}

impl ContDriver for Driver<'_, '_> {
    fn admit(&mut self) -> Option<ContAdmit> {
        self.request.take()
    }
    fn on_token(&mut self, _: usize, token: i32) -> bool {
        self.generated.push(token);
        true
    }
    fn on_done(&mut self, _: usize, tokens: &[i32], _: i32, stats: ContDone) {
        self.done = tokens == self.generated && stats.spec_drafts == 0;
    }
    fn sample_override(&mut self, _: usize) -> i32 {
        cont_sample_token(FORCED_TOKEN)
    }
    fn on_admitted(&mut self, _: usize, cached: i32, computed: i32, bank: i32) -> bool {
        self.admitted = Some((cached, computed, bank));
        true
    }
    fn on_checkpoint(&mut self, _: usize, bank: i32, tokens: &[i32]) {
        if tokens.len() != CUT {
            return;
        }
        let Some(path) = &self.checkpoint else {
            return;
        };
        match self.batch.save_bank_payload(bank, path) {
            Ok(()) => self.checkpoint_saved = true,
            Err(error) => self.failure = Some(error.to_string()),
        }
    }
}

fn run_bank(
    batch: &BatchCtx<'_>,
    request: ContAdmit,
    checkpoint: Option<PathBuf>,
) -> Result<(i32, i32, i32)> {
    let needs_checkpoint = checkpoint.is_some();
    let mut driver = Driver {
        batch,
        request: Some(request),
        checkpoint,
        checkpoint_saved: false,
        failure: None,
        admitted: None,
        generated: Vec::new(),
        done: false,
    };
    batch.continuous_generate(&mut driver)?;
    if let Some(error) = driver.failure {
        return Err(error.into());
    }
    assert!(driver.done && driver.generated == vec![FORCED_TOKEN; GENERATED as usize]);
    assert!(!needs_checkpoint || driver.checkpoint_saved);
    driver
        .admitted
        .ok_or_else(|| "request was not admitted".into())
}

fn bank_gate(model: &Model, prompt: &[i32], out: &Path) -> Result<()> {
    let batch = model.batch_ctx_fit(CONTEXT, 2, 2 * CONTEXT)?;
    assert_eq!(batch.max_seq(), 2);
    assert!(batch.supports_partial_reuse());
    let checkpoint = out.join("bank-checkpoint.bin");
    let mut request = ContAdmit::cold(1, prompt.to_vec(), GENERATED);
    request.place_bank = 1;
    request.checkpoint_at = CUT as i32;
    let cold = run_bank(&batch, request, Some(checkpoint.clone()))?;
    assert_eq!(cold, (0, PREFIX as i32, 0));
    let stable = out.join("bank0-saved.bin");
    let after = out.join("bank0-after.bin");
    batch.save_bank_payload(0, &stable)?;

    let mut request = ContAdmit::cold(2, prompt[..37].to_vec(), GENERATED);
    request.place_bank = 2;
    run_bank(&batch, request, None)?;
    batch.save_bank_payload(0, &after)?;
    same_file(&stable, &after)?;

    // Fork at a non-multiple-of-four checkpoint after the source advanced.
    // Restoring only the KV count would leave the wrong KDA/pooling frontier.
    let mut fork = prompt[..CUT].to_vec();
    fork.extend(
        model
            .vocab()
            .encode_bytes(b"\nContinue with the corrected implementation."),
    );
    let mut request = ContAdmit::cold(3, fork.clone(), GENERATED);
    request.place_bank = 2;
    request.fork_bank = 1;
    request.n_cached = CUT as i32;
    let partial = run_bank(&batch, request, None)?;
    assert_eq!(partial, (CUT as i32, (fork.len() - CUT) as i32, 1));
    let forked = batch.bank_snapshot(1)?;
    assert_eq!(&forked.tokens[..fork.len()], &fork);
    let bank_result = out.join("bank1-partial.bin");
    batch.save_bank_payload(1, &bank_result)?;
    {
        let mut reference = model.session(CONTEXT)?;
        reference.load_payload(&checkpoint)?;
        reference.sync(&TokenBuffer::from_tokens(fork.clone()))?;
        for &token in &forked.tokens[fork.len()..] {
            reference.eval(token)?;
        }
        let reference_path = out.join("partial-reference.bin");
        reference.save_payload(&reference_path)?;
        same_file(&bank_result, &reference_path)?;
    }

    // The public bank payload API must restore actual mutable state too.
    batch.load_bank_payload_range(1, &stable, 0, std::fs::metadata(&stable)?.len())?;
    let imported = out.join("bank1-imported.bin");
    batch.save_bank_payload(1, &imported)?;
    same_file(&stable, &imported)?;
    let saved = batch.bank_snapshot(1)?;
    let mut warm = saved.tokens.clone();
    warm.extend(model.vocab().encode_bytes(b"\nExplain the next step."));
    let mut request = ContAdmit::cold(4, warm.clone(), GENERATED);
    request.place_bank = 2;
    request.n_cached = saved.tokens.len() as i32;
    let reused = run_bank(&batch, request, None)?;
    assert_eq!(
        reused,
        (
            saved.tokens.len() as i32,
            (warm.len() - saved.tokens.len()) as i32,
            1
        )
    );
    let frontier = batch.bank_snapshot(1)?;
    let warm_path = out.join("bank1-warm.bin");
    batch.save_bank_payload(1, &warm_path)?;
    {
        let mut reference = model.session(CONTEXT)?;
        reference.load_payload(&stable)?;
        reference.sync(&TokenBuffer::from_tokens(warm.clone()))?;
        for &token in &frontier.tokens[warm.len()..] {
            reference.eval(token)?;
        }
        let path = out.join("warm-reference.bin");
        reference.save_payload(&path)?;
        same_file(&warm_path, &path)?;
    }
    batch.save_bank_payload(0, &after)?;
    same_file(&stable, &after)?;
    println!(
        "{}",
        json!({"banks":batch.max_seq(), "isolation":"byte_exact",
        "partial_prefix":partial.0, "disk_prefix":reused.0,
        "continuation":"byte_exact_vs_same_state_serial", "sampling":"forced token 100"})
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: glm_vq_state MODEL PROMPT_FILE OUTPUT_DIR".into());
    }
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    assert_eq!(model.family(), ModelFamily::Glm53Vq);
    let mut prompt = model.vocab().encode_bytes(&std::fs::read(&args[1])?);
    assert!(prompt.len() >= PREFIX);
    prompt.truncate(PREFIX);
    let start = Instant::now();
    session_gate(&model, &prompt, out)?;
    std::io::stdout().flush()?;
    bank_gate(&model, &prompt, out)?;
    println!(
        "{}",
        json!({"result":"GLMVQ_STATE_PASS", "seconds":start.elapsed().as_secs_f64(),
        "context":CONTEXT, "prompt_tokens":PREFIX, "checkpoint":CUT,
        "scope":"single-owner native state and Rust host; no HTTP/MTP/capture/vision qualification"})
    );
    Ok(())
}
