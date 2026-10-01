//! Actual-artifact speculative/serial parity. Diagnostic readback timings do
//! not establish source-model quality or serving throughput.
use ds4_core::{
    parse_prefix, snapshot_mem, snapshot_spec, Backend, IQuestCache, IQuestPlan, MemSnap, Model,
    ModelFamily, ModelOpenOption, Session, SessionSnapshot, HEADER_BYTES,
};
use ds4_server::metrics::{MEM_CLASS_NAMES, MEM_DOMAIN_NAMES};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type GateResult<T> = Result<T, Box<dyn std::error::Error>>;
const DEFAULT_CONTEXT: i32 = 512;
const DEFAULT_DRAFT: i32 = 3;
const DEFAULT_STEPS: usize = 4;
const MAX_CONTEXT: i32 = 524_288;
const MAIN_WINDOW: usize = 4096;
const DRAFT_WINDOW: usize = 512;
const DRAFT_RING: usize = DRAFT_WINDOW + IQuestPlan::MAX_MTP_DRAFT as usize;
const COMPARE_CHUNK: usize = 1024 * 1024;
const TRUNCATED_TAIL_BYTES: u64 = 64;
const PAYLOAD_CHUNK_FIELD: usize = 3;
const PAYLOAD_MTP_FIELD: usize = 8;
const DEFAULT_PROMPT: &str = "Write a short sentence about a mountain.";
const HELP: &str = "usage: iquest_verify MODEL.gguf OUTPUT_DIR [real] [OPTIONS]\n\
  --ctx N             Session capacity, 1..524288 (default 512)\n\
  --prompt-file PATH  UTF-8 user message rendered by the official chat template\n\
  --draft N           Recursive draft count, 2..7 (default 3; margin 0)\n\
  --steps N           Committed output-token budget, including EOS (default 4)\n\
  --help              Show help without loading a model\n\n\
Only the published real artifact is accepted. EOS can shorten the workload.\n\
Each speculative round is replayed one token at a time from a snapshot.\n\
Payload comparison uses bounded reads. The native snapshot retains one full\n\
committed payload in host memory. Reports distinguish logical windows from\n\
physical ring rollover and record actual prompt/output token counts.\n";

struct Options {
    model: String,
    out: PathBuf,
    ctx: i32,
    prompt_file: Option<PathBuf>,
    draft: i32,
    steps: usize,
}

enum Command {
    Help,
    Run(Options),
}

fn parse_args(args: &[String]) -> GateResult<Command> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        return Ok(Command::Help);
    }
    if args.len() < 2 {
        return Err(HELP.into());
    }
    let mut options = Options {
        model: args[0].clone(),
        out: PathBuf::from(&args[1]),
        ctx: DEFAULT_CONTEXT,
        prompt_file: None,
        draft: DEFAULT_DRAFT,
        steps: DEFAULT_STEPS,
    };
    let mut index = 2;
    if args.get(index).is_some_and(|arg| arg == "real") {
        index += 1;
    }
    while index < args.len() {
        let flag = &args[index];
        if !matches!(
            flag.as_str(),
            "--ctx" | "--prompt-file" | "--draft" | "--steps"
        ) {
            return Err(
                format!("unsupported argument {flag}; fixture mode is not supported").into(),
            );
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--ctx" => options.ctx = value.parse()?,
            "--prompt-file" => options.prompt_file = Some(PathBuf::from(value)),
            "--draft" => options.draft = value.parse()?,
            "--steps" => options.steps = value.parse()?,
            _ => unreachable!(),
        }
        index += 2;
    }
    if !(1..=MAX_CONTEXT).contains(&options.ctx) {
        return Err("--ctx must be 1..524288".into());
    }
    if !(2..=IQuestPlan::MAX_MTP_DRAFT as i32).contains(&options.draft) {
        return Err("--draft must be 2..7".into());
    }
    if options.steps == 0 || options.steps > options.ctx as usize {
        return Err("--steps must be positive and no larger than --ctx".into());
    }
    Ok(Command::Run(options))
}

fn finite_logits(session: &Session<'_>, vocab: usize) -> GateResult<Vec<f32>> {
    let values = session.copy_logits(vocab)?;
    if values.len() != vocab || values.iter().any(|value| !value.is_finite()) {
        return Err("incomplete or nonfinite full-vocabulary logits".into());
    }
    if values.iter().all(|value| *value == 0.0) {
        return Err("identically zero logits".into());
    }
    Ok(values)
}

fn equal_logits(left: &[f32], right: &[f32]) -> GateResult<()> {
    if left.len() != right.len() {
        return Err("logit vector length changed".into());
    }
    if let Some((index, (a, b))) = left
        .iter()
        .zip(right)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
    {
        return Err(format!("logit {index} differs: {a} != {b}").into());
    }
    Ok(())
}

fn valid_token(token: i32, vocab: usize) -> GateResult<()> {
    if token < 0 || token as usize >= vocab {
        return Err(format!("invalid token {token} for vocabulary {vocab}").into());
    }
    Ok(())
}

fn compare_files(left: &Path, right: &Path) -> GateResult<u64> {
    let mut a = File::open(left)?;
    let mut b = File::open(right)?;
    let len = a.metadata()?.len();
    if b.metadata()?.len() != len {
        return Err("target/MTP payload lengths differ".into());
    }
    let mut a_bytes = vec![0; COMPARE_CHUNK];
    let mut b_bytes = vec![0; COMPARE_CHUNK];
    let mut offset = 0;
    while offset < len {
        let count = (len - offset).min(COMPARE_CHUNK as u64) as usize;
        a.read_exact(&mut a_bytes[..count])?;
        b.read_exact(&mut b_bytes[..count])?;
        if let Some(index) = a_bytes[..count]
            .iter()
            .zip(&b_bytes[..count])
            .position(|(a, b)| a != b)
        {
            return Err(format!(
                "target/MTP payload differs at byte {}",
                offset + index as u64
            )
            .into());
        }
        offset += count as u64;
    }
    Ok(len)
}

fn payload_prefix(path: &Path, tokens: usize) -> GateResult<ds4_core::HostPrefix> {
    let mut file = File::open(path)?;
    let mut bytes = vec![0; HEADER_BYTES + tokens * std::mem::size_of::<u32>()];
    file.read_exact(&mut bytes)?;
    Ok(parse_prefix(&bytes)?)
}

fn copy_prefix(source: impl Read, target: &mut impl Write, bytes: u64) -> GateResult<()> {
    if std::io::copy(&mut source.take(bytes), target)? != bytes {
        return Err("source payload shorter than truncation fixture".into());
    }
    Ok(())
}

fn memory(snap: &MemSnap) -> Value {
    let mut cells = serde_json::Map::new();
    for (index, class) in snap.census.cells.iter().enumerate() {
        let mut domains = serde_json::Map::new();
        for (domain, cell) in class.iter().enumerate() {
            domains.insert(
                MEM_DOMAIN_NAMES[domain].into(),
                json!({
                    "requested":cell.requested,"committed":cell.committed,
                    "freed_requested":cell.freed_requested,"freed_committed":cell.freed_committed,
                    "alloc_calls":cell.alloc_calls,"free_calls":cell.free_calls,
                }),
            );
        }
        cells.insert(MEM_CLASS_NAMES[index].into(), domains.into());
    }
    json!({"supported":snap.census.supported,"faults":snap.census.faults,
        "substrate_outstanding":snap.substrate_outstanding,"cells":cells,
        "observation":{"status":snap.observe.status,"source":snap.observe.source,
        "free_bytes":snap.observe.free_bytes,"total_bytes":snap.observe.total_bytes,
        "cuda_free_bytes":snap.observe.cuda_free_bytes,"meminfo_available_bytes":snap.observe.meminfo_avail_bytes}})
}

fn ring_wraps(start: usize, end: usize, capacity: usize) -> usize {
    if end <= start || capacity == 0 {
        return 0;
    }
    (end - 1) / capacity - start.saturating_sub(1) / capacity
}

fn run(options: Options) -> GateResult<()> {
    let wall = Instant::now();
    let prompt = match &options.prompt_file {
        Some(path) => std::fs::read_to_string(path)?,
        None => DEFAULT_PROMPT.into(),
    };
    std::fs::create_dir_all(&options.out)?;
    let before_model = snapshot_mem();
    let started = Instant::now();
    let model = Model::open_with_support_options(
        &options.model,
        Backend::Cuda,
        0,
        true,
        None,
        None,
        &[
            ModelOpenOption::MtpDraftTokens(options.draft),
            ModelOpenOption::MtpMargin(0.0),
        ],
    )?;
    let model_seconds = started.elapsed().as_secs_f64();
    if model.family() != ModelFamily::IQuestQ1 || model.mtp_draft_tokens() != options.draft {
        return Err("the requested IQuest-Q1 recursive draft did not open".into());
    }
    let loaded_memory = snapshot_mem();
    let vocab = model.vocab().n_vocab() as usize;
    let eos = model.token_eos();
    valid_token(eos, vocab)?;
    let input = model.encode_chat_prompt(None, &prompt, 0)?;
    if input.is_empty() || input.len() + options.steps > options.ctx as usize {
        return Err(format!(
            "prompt {} + requested output {} exceeds context {}",
            input.len(),
            options.steps,
            options.ctx
        )
        .into());
    }
    let mut ids = BufWriter::new(File::create(options.out.join("prompt.tokens.i32"))?);
    for &token in input.as_slice() {
        valid_token(token, vocab)?;
        ids.write_all(&token.to_le_bytes())?;
    }
    ids.flush()?;

    let mut session = model.session(options.ctx)?;
    let started = Instant::now();
    session.sync(&input)?;
    let mut current = finite_logits(&session, vocab)?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    if session.pos() != input.len() as i32 {
        return Err("incorrect prefill frontier".into());
    }
    let prefilled_memory = snapshot_mem();
    eprintln!(
        "prefill tokens={} ctx={} seconds={prefill_seconds:.3}",
        input.len(),
        session.ctx()
    );
    let mut logits_file = BufWriter::new(File::create(options.out.join("prompt-logits.f32"))?);
    for value in &current {
        logits_file.write_all(&value.to_le_bytes())?;
    }
    logits_file.flush()?;

    let prompt_path = options.out.join("prompt.kv");
    let comparison_path = options.out.join("comparison.kv");
    let reference_path = options.out.join("reference.kv");
    session.save_payload(&prompt_path)?;
    let prefix = payload_prefix(&prompt_path, input.len())?;
    let native_chunk = prefix.fields[PAYLOAD_CHUNK_FIELD];
    if prefix.ctx() != options.ctx as u32
        || prefix.fields[PAYLOAD_MTP_FIELD] != 1
        || native_chunk == 0
    {
        return Err("payload lacks the requested context or active MTP".into());
    }
    let main_ring = (options.ctx as usize).min(MAIN_WINDOW + native_chunk as usize - 1);
    // Read back restored KV before using it as either side of the parity gate.
    session.invalidate();
    session.load_payload(&prompt_path)?;
    equal_logits(&current, &finite_logits(&session, vocab)?)?;
    session.save_payload(&comparison_path)?;
    let prompt_payload_bytes = compare_files(&prompt_path, &comparison_path)?;

    // Keep a valid host prefix so rejection reaches the native IQuest loader,
    // which must invalidate its graph before a valid restore can recover it.
    let truncated_path = options.out.join("truncated.kv");
    let truncated_bytes = prefix.prefix_len() as u64 + TRUNCATED_TAIL_BYTES;
    if truncated_bytes >= prompt_payload_bytes {
        return Err("prompt payload too short for the native truncation gate".into());
    }
    copy_prefix(
        File::open(&prompt_path)?,
        &mut File::create(&truncated_path)?,
        truncated_bytes,
    )?;
    if payload_prefix(&truncated_path, input.len())? != prefix {
        return Err("truncation changed the valid host prefix".into());
    }
    let started = Instant::now();
    let generation_before = session.native_generation();
    let rejection = match session.load_payload(&truncated_path) {
        Ok(()) => return Err("native loader accepted a truncated target/MTP payload".into()),
        Err(error) => error.to_string(),
    };
    let generation_rejected = session.native_generation();
    if generation_rejected <= generation_before || session.pos() != 0 {
        return Err("truncated native restore did not invalidate the checkpoint".into());
    }
    session.load_payload(&prompt_path)?;
    if session.pos() != input.len() as i32 {
        return Err("truncated-payload recovery lost the committed frontier".into());
    }
    equal_logits(&current, &finite_logits(&session, vocab)?)?;
    session.save_payload(&comparison_path)?;
    compare_files(&prompt_path, &comparison_path)?;
    let truncation_gate = json!({"fixture":"truncated.kv","fixture_bytes":truncated_bytes,
        "source_payload_bytes":prompt_payload_bytes,"host_prefix_unchanged":true,
        "native_error":rejection,"generation_before":generation_before,
        "generation_rejected":generation_rejected,"generation_recovered":session.native_generation(),
        "rejected_checkpoint_position":0,"recovered_position":session.pos(),
        "recovery_full_vocab_logits_bit_exact":true,"recovery_payload_byte_exact":true,
        "elapsed_seconds":started.elapsed().as_secs_f64()});
    std::fs::write(
        options.out.join("truncation.json"),
        serde_json::to_vec_pretty(&truncation_gate)?,
    )?;
    eprintln!("truncated payload rejected; committed target/MTP payload restored exactly");

    let mut snapshot = SessionSnapshot::new()?;
    let mut rounds_file = BufWriter::new(File::create(options.out.join("rounds.jsonl"))?);
    let mut tokens = Vec::new();
    let mut text = Vec::new();
    let mut rounds = Vec::new();
    let mut drafts = 0;
    let mut hits = 0;
    let mut trial_width = 0;
    let mut max_snapshot_bytes = 0;
    let mut payload_bytes = prompt_payload_bytes;
    let mut speculative_seconds = 0.0;
    let mut ordinary_seconds = 0.0;
    let mut eos_seen = false;
    while tokens.len() < options.steps && !eos_seen {
        let round = rounds.len();
        let before = session.pos() as usize;
        let limit = (options.steps - tokens.len()).min(options.draft as usize + 1);
        let first = session.argmax();
        valid_token(first, vocab)?;
        let started = Instant::now();
        session.save_snapshot(&mut snapshot)?;
        let snapshot_seconds = started.elapsed().as_secs_f64();
        max_snapshot_bytes = max_snapshot_bytes.max(payload_bytes + 1);
        let counters = snapshot_spec();
        let started = Instant::now();
        let accepted = session.eval_speculative_argmax(first, limit as i32, eos)?;
        let speculative = finite_logits(&session, vocab)?;
        let spec_seconds = started.elapsed().as_secs_f64();
        let after_counters = snapshot_spec();
        if accepted.is_empty()
            || accepted.len() > limit
            || accepted[0] != first
            || session.pos() != (before + accepted.len()) as i32
        {
            return Err(format!("round {round}: invalid speculative frontier").into());
        }
        for (index, &token) in accepted.iter().enumerate() {
            valid_token(token, vocab)?;
            if token == eos && index + 1 != accepted.len() {
                return Err(format!("round {round}: accepted tokens follow EOS").into());
            }
        }
        let started = Instant::now();
        session.save_payload(&comparison_path)?;
        session.load_snapshot(&snapshot)?;
        if session.pos() != before as i32 {
            return Err(format!("round {round}: snapshot frontier changed").into());
        }
        equal_logits(&current, &finite_logits(&session, vocab)?)?;
        let restore_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        for &token in &accepted {
            if token != session.argmax() {
                return Err(format!("round {round}: target token mismatch").into());
            }
            session.eval(token)?;
            current = finite_logits(&session, vocab)?;
        }
        let serial_seconds = started.elapsed().as_secs_f64();
        if session.pos() != (before + accepted.len()) as i32 {
            return Err(format!("round {round}: serial frontier differs").into());
        }
        equal_logits(&speculative, &current)?;
        let started = Instant::now();
        session.save_payload(&reference_path)?;
        payload_bytes = compare_files(&comparison_path, &reference_path)?;
        let payload_seconds = started.elapsed().as_secs_f64();
        let proposed = after_counters
            .drafts
            .checked_sub(counters.drafts)
            .ok_or("draft counter regressed")?;
        let accepted_drafts = after_counters
            .hits
            .checked_sub(counters.hits)
            .ok_or("accept counter regressed")?;
        if proposed != limit.saturating_sub(1) as u64
            || accepted_drafts != accepted.len().saturating_sub(1) as u64
        {
            return Err(format!(
                "round {round}: margin-zero trial counters disagree with committed tokens"
            )
            .into());
        }
        let trial_end = before + proposed as usize + 1;
        trial_width = trial_width.max(proposed + 1);
        drafts += proposed;
        hits += accepted_drafts;
        speculative_seconds += spec_seconds;
        ordinary_seconds += serial_seconds;
        let after = session.pos() as usize;
        let result = json!({"round":round,"start":before,"end":after,"limit":limit,
            "accepted_token_ids":accepted,"draft_proposals":proposed,"draft_hits":accepted_drafts,
            "quench_delta":after_counters.quench.saturating_sub(counters.quench),
            "full_vocab_logits_bit_exact":true,"committed_payload_byte_exact":true,
            "payload_bytes":payload_bytes,"snapshot_native_buffer_bytes":max_snapshot_bytes,
            "snapshot_save_seconds":snapshot_seconds,"speculative_readback_seconds":spec_seconds,
            "spec_payload_snapshot_restore_seconds":restore_seconds,"ordinary_readback_seconds":serial_seconds,
            "serial_payload_compare_seconds":payload_seconds,
            "trial_end":trial_end,
            "trial_swa_ring_wraps":ring_wraps(before,trial_end,main_ring),
            "trial_mtp_ring_wraps":ring_wraps(before.saturating_sub(1),trial_end.saturating_sub(1),DRAFT_RING),
            "committed_swa_ring_wraps":ring_wraps(before,after,main_ring),
            "committed_mtp_ring_wraps":ring_wraps(before.saturating_sub(1),after.saturating_sub(1),DRAFT_RING)});
        writeln!(rounds_file, "{result}")?;
        rounds_file.flush()?;
        eprintln!("round={round} position={after} committed={} proposed={proposed} hits={accepted_drafts}", accepted.len());
        rounds.push(result);
        for &token in &accepted {
            if token != eos {
                text.extend(model.token_text(token)?);
            }
        }
        eos_seen = accepted.last() == Some(&eos);
        tokens.extend(accepted);
    }
    if drafts == 0 {
        return Err("no recursive proposals executed; MTP gate is unproven".into());
    }
    // Reuse this path so only three full payload files remain on disk.
    session.invalidate();
    session.load_payload(&reference_path)?;
    equal_logits(&current, &finite_logits(&session, vocab)?)?;
    session.save_payload(&comparison_path)?;
    compare_files(&reference_path, &comparison_path)?;
    let final_position = session.pos() as usize;
    if final_position != input.len() + tokens.len() {
        return Err("final disk frontier changed".into());
    }
    let final_memory = snapshot_mem();
    if final_memory.census.faults != before_model.census.faults {
        return Err("memory census fault count increased".into());
    }
    let weights: u64 = model
        .inventory()
        .tensors
        .iter()
        .map(|tensor| tensor.bytes)
        .sum();
    let report = json!({"artifact":options.model,"scope":"actual checkpoint greedy serial/MTP parity",
        "architecture":"iquest_q1","requested":{"ctx":options.ctx,"draft":options.draft,"steps":options.steps,"margin":0.0},
        "effective":{"ctx":session.ctx(),"draft":model.mtp_draft_tokens(),"max_seqs":1,"native_chunk":native_chunk},
        "prompt_file":options.prompt_file,"prompt_utf8_bytes":prompt.len(),"prompt_tokens":input.len(),
        "prompt_token_file":"prompt.tokens.i32","generated_tokens":tokens.len(),"generated_token_ids":tokens,
        "generated_text":String::from_utf8_lossy(&text),"eos_seen":eos_seen,"final_position":final_position,
        "comparison_rounds":rounds.len(),"largest_trial_width":trial_width,
        "requested_draft_exercised":trial_width==options.draft as u64+1,
        "draft_proposals":drafts,"draft_hits":hits,"rounds":rounds,
        "truncated_payload_gate":truncation_gate,
        "artifacts":{"initial_payload":"prompt.kv","final_payload_pair":["reference.kv","comparison.kv"],
            "round_log":"rounds.jsonl","prefill_logits":"prompt-logits.f32","truncation":"truncation.json"},
        "timing":{"model_load_seconds":model_seconds,"prefill_readback_seconds":prefill_seconds,
            "speculative_readback_seconds":speculative_seconds,"ordinary_readback_seconds":ordinary_seconds,
            "total_wall_seconds":wall.elapsed().as_secs_f64(),"scope":"diagnostic wall times include full-logit readback; no throughput qualification"},
        "workload":{"main_window":MAIN_WINDOW,"main_ring_rows":main_ring,"mtp_window":DRAFT_WINDOW,"mtp_ring_rows":DRAFT_RING,
            "prefill_swa_ring_wraps":ring_wraps(0,input.len(),main_ring),
            "prefill_mtp_ring_wraps":ring_wraps(0,input.len().saturating_sub(1),DRAFT_RING),
            "decode_swa_ring_wraps":ring_wraps(input.len(),final_position,main_ring),
            "decode_mtp_ring_wraps":ring_wraps(input.len().saturating_sub(1),final_position.saturating_sub(1),DRAFT_RING),
            "swa_window_exceeded":final_position>MAIN_WINDOW,"mtp_window_exceeded":final_position.saturating_sub(1)>DRAFT_WINDOW},
        "memory":{"weight_payload_bytes":weights,
            "session_tensor_quote_bytes":IQuestPlan::session_bytes(options.ctx as u32,native_chunk),
            "active_kv_quote_bytes":IQuestPlan::kv_bytes(options.ctx as u32,native_chunk,IQuestCache::WithMtp),
            "snapshot_native_buffer_bytes":max_snapshot_bytes,
            "snapshot_scope":"native payload buffer only; Rust token ledger and allocator overhead are additional",
            "comparison_host_buffers_bytes":2*COMPARE_CHUNK,
            "retained_payload_file_bytes":prompt_payload_bytes+2*payload_bytes+truncated_bytes,
            "before_model":memory(&before_model),"loaded":memory(&loaded_memory),
            "prefilled":memory(&prefilled_memory),"final":memory(&final_memory)},
        "gates":{"finite_full_vocab_logits":true,"valid_token_ids":true,"eos_prefix":true,
            "snapshot_restore":true,"disk_restore_payload_exact":true,"target_argmax_match":true,
            "truncated_payload_native_rejection":true,"failed_graph_recovery_payload_exact":true,
            "mtp_logits_bit_exact":true,"mtp_committed_payload_exact":true,"memory_faults_unchanged":true},
        "quality_comparison_qualified":false,"serving_throughput_qualified":false});
    std::fs::write(
        options.out.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}

fn main() -> GateResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match parse_args(&args)? {
        Command::Help => {
            print!("{HELP}");
            Ok(())
        }
        Command::Run(options) => run(options),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).into()).collect()
    }

    #[test]
    fn parses_real_workloads() {
        let Command::Run(short) = parse_args(&args(&["model.gguf", "out", "real"])).unwrap() else {
            panic!("expected workload");
        };
        assert_eq!((short.ctx, short.draft, short.steps), (512, 3, 4));
        let Command::Run(long) = parse_args(&args(&[
            "model.gguf",
            "out",
            "--ctx",
            "8192",
            "--draft",
            "7",
            "--steps",
            "32",
            "--prompt-file",
            "prompt with spaces.txt",
        ]))
        .unwrap() else {
            panic!("expected workload");
        };
        assert_eq!((long.ctx, long.draft, long.steps), (8192, 7, 32));
        assert_eq!(
            long.prompt_file,
            Some(PathBuf::from("prompt with spaces.txt"))
        );
        assert!(matches!(
            parse_args(&args(&["--help"])).unwrap(),
            Command::Help
        ));
    }

    #[test]
    fn refuses_invalid_workloads() {
        for extra in [
            vec!["--ctx", "0"],
            vec!["--ctx", "524289"],
            vec!["--draft", "1"],
            vec!["--draft", "8"],
            vec!["--steps", "0"],
            vec!["--steps", "513"],
            vec!["--prompt-file"],
            vec!["nonzero-fixture"],
            vec!["--unknown", "2"],
        ] {
            let mut input = args(&["model.gguf", "out"]);
            input.extend(args(&extra));
            assert!(parse_args(&input).is_err(), "{input:?}");
        }
    }

    #[test]
    fn prefix_copy_is_bounded() {
        let source = [7u8; 256];
        let mut copied = Vec::new();
        copy_prefix(&source[..], &mut copied, TRUNCATED_TAIL_BYTES).unwrap();
        assert_eq!(copied, vec![7; TRUNCATED_TAIL_BYTES as usize]);
        assert!(copy_prefix(&source[..4], &mut Vec::new(), TRUNCATED_TAIL_BYTES).is_err());
    }
}
