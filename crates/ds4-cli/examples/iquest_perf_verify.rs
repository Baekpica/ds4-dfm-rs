//! Ordinary IQuest off/on state evidence. Readback and disk I/O are diagnostic,
//! not throughput measurements. Each arm opens one model in a fresh process.
use ds4_core::{
    parse_prefix, snapshot_mem, snapshot_spec, Backend, HostPrefix, IQuestCache, IQuestPlan,
    MemSnap, Model, ModelFamily, ModelOpenOption, Session, TokenBuffer, HEADER_BYTES,
};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type GateResult<T> = Result<T, Box<dyn std::error::Error>>;
const VOCAB: usize = 160_000;
const MAX_CONTEXT: usize = 524_288;
const MAIN_WINDOW: usize = 4096;
const IO_CHUNK: usize = 1024 * 1024;
const MAX_JSON_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PROMPT_BYTES: u64 = 64 * 1024 * 1024;
const THINK_NONE: i32 = 0;
const PAYLOAD_CHUNK: usize = 3;
const PAYLOAD_MTP: usize = 8;
const OPT_CONTROLS: [&str; 6] = [
    "DS4_IQUEST_ATTN_SHUFFLE",
    "DS4_IQUEST_ATTN_WARP",
    "DS4_IQUEST_ATTN_TILED",
    "DS4_IQUEST_ATTN_CACHED",
    "DS4_IQUEST_ROUTER_WARP",
    "DS4_IQUEST_ATTN_ASYNC",
];
const HELP: &str =
    "usage: iquest_perf_verify MODEL OUT --prompt-file PATH --control SWITCH [OPTIONS]\n\
  --control SWITCH   Targeted IQuest optimization switch; explicit env 0 or 1\n\
  --frontier N        Exact prefill token count (default 2048)\n\
  --capacity N        Session capacity (default 8192)\n\
  --steps N           Ordinary committed decode rows (default 32)\n\
  --chunk N           Expected DS4_IQUEST_PREFILL_CHUNK (default 128)\n\
  --prompt-mode MODE  raw (bench --prompt-file) or chat (official template)\n\
  --tokens-file PATH  Fixed JSON token IDs; otherwise generate non-EOS greedy IDs\n\
  --help              Help without opening the model\n\n\
CPU-only comparison: iquest_perf_verify --compare LEFT_DIR RIGHT_DIR\n\
Both arms must name the same switch, set it to opposite 0/1 values, and keep\n\
all other recorded environment values, workload and decoded IDs identical.\n\
Raw mode matches ds4-bench: tokenize then truncate; never replicate tokens.\n\
Chat mode uses the official template with reasoning disabled, then truncates.\n\
A short prompt is rejected. Set mapped/base IPC and chunk in the environment.\n\
MTP is disabled (draft=1). Run arms sequentially against one live owner.\n\
Prefill/final full-vocabulary logits and committed native payloads are saved.\n\
Restore checks run after all decode rows, using one temporary payload removed\n\
after comparison. Timings include diagnostics and do not measure throughput.\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptMode {
    Raw,
    Chat,
}

impl PromptMode {
    fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Chat => "chat",
        }
    }
}

struct Options {
    model: String,
    out: PathBuf,
    prompt: PathBuf,
    frontier: usize,
    capacity: usize,
    steps: usize,
    chunk: usize,
    mode: PromptMode,
    tokens: Option<PathBuf>,
    control: String,
}

enum Command {
    Help,
    Record(Options),
    Compare(PathBuf, PathBuf),
}

fn parse_args(args: &[String]) -> GateResult<Command> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        return Ok(Command::Help);
    }
    if args.first().is_some_and(|arg| arg == "--compare") {
        if args.len() != 3 {
            return Err("--compare needs exactly two recorded directories".into());
        }
        return Ok(Command::Compare(
            args[1].clone().into(),
            args[2].clone().into(),
        ));
    }
    if args.len() < 4 || (args.len() - 2) % 2 != 0 {
        return Err(HELP.into());
    }
    let mut options = Options {
        model: args[0].clone(),
        out: args[1].clone().into(),
        prompt: PathBuf::new(),
        frontier: 2048,
        capacity: 8192,
        steps: 32,
        chunk: 128,
        mode: PromptMode::Raw,
        tokens: None,
        control: String::new(),
    };
    for pair in args[2..].chunks_exact(2) {
        match pair[0].as_str() {
            "--prompt-file" => options.prompt = pair[1].clone().into(),
            "--frontier" => options.frontier = pair[1].parse()?,
            "--capacity" => options.capacity = pair[1].parse()?,
            "--steps" => options.steps = pair[1].parse()?,
            "--chunk" => options.chunk = pair[1].parse()?,
            "--tokens-file" => options.tokens = Some(pair[1].clone().into()),
            "--control" => options.control = pair[1].clone(),
            "--prompt-mode" => {
                options.mode = match pair[1].as_str() {
                    "raw" => PromptMode::Raw,
                    "chat" => PromptMode::Chat,
                    _ => return Err("--prompt-mode must be raw or chat".into()),
                }
            }
            flag => return Err(format!("unsupported argument {flag}").into()),
        }
    }
    if !OPT_CONTROLS.contains(&options.control.as_str()) {
        return Err("--control must name an IQuest optimization switch".into());
    }
    if options.prompt.as_os_str().is_empty()
        || !(2..=MAX_CONTEXT).contains(&options.capacity)
        || options.frontier == 0
        || options.steps == 0
        || options.frontier >= options.capacity
        || options.steps >= options.capacity - options.frontier
        || options.chunk == 0
        || options.chunk > options.capacity.min(8192)
    {
        return Err("invalid prompt, frontier, steps, capacity or chunk".into());
    }
    Ok(Command::Record(options))
}

fn read_bounded(path: &Path, limit: u64) -> GateResult<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(format!("{} exceeds {limit} bytes", path.display()).into());
    }
    Ok(bytes)
}

fn prompt_prefix(bytes: &[u8]) -> &[u8] {
    // Match bench::read_prompt, including the retained C-string input boundary.
    &bytes[..bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len())]
}

fn write_json(path: &Path, value: &Value) -> GateResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn valid_tokens(tokens: &[i32], expected: usize, eos: Option<i32>) -> GateResult<()> {
    if tokens.len() != expected
        || tokens
            .iter()
            .any(|&token| token < 0 || token as usize >= VOCAB || eos == Some(token))
    {
        return Err("wrong token count, invalid vocabulary ID or excluded EOS".into());
    }
    Ok(())
}

fn finite_logits(values: &[f32]) -> GateResult<()> {
    if values.len() != VOCAB
        || values.iter().any(|value| !value.is_finite())
        || values.iter().all(|value| *value == 0.0)
    {
        return Err("incomplete, nonfinite or all-zero full-vocabulary logits".into());
    }
    Ok(())
}

fn write_logits(path: &Path, values: &[f32]) -> GateResult<()> {
    finite_logits(values)?;
    let mut file = BufWriter::new(File::create(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn read_logits(path: &Path) -> GateResult<Vec<f32>> {
    let bytes = read_bounded(path, (VOCAB * 4) as u64)?;
    if bytes.len() != VOCAB * 4 {
        return Err("wrong raw full-vocabulary logit length".into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four-byte chunk")))
        .collect();
    finite_logits(&values)?;
    Ok(values)
}

fn equal_logits(left: &[f32], right: &[f32]) -> GateResult<()> {
    finite_logits(left)?;
    finite_logits(right)?;
    if let Some(index) = left
        .iter()
        .zip(right)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(format!("full-vocabulary logits differ at {index}").into());
    }
    Ok(())
}

fn compare_files(a: &Path, b: &Path) -> GateResult<u64> {
    let mut a = File::open(a)?;
    let mut b = File::open(b)?;
    let length = a.metadata()?.len();
    if b.metadata()?.len() != length || length == 0 {
        return Err("artifact lengths differ or are empty".into());
    }
    let mut left = vec![0; IO_CHUNK];
    let mut right = vec![0; IO_CHUNK];
    for offset in (0..length).step_by(IO_CHUNK) {
        let count = (length - offset).min(IO_CHUNK as u64) as usize;
        a.read_exact(&mut left[..count])?;
        b.read_exact(&mut right[..count])?;
        if let Some(index) = left[..count]
            .iter()
            .zip(&right[..count])
            .position(|(a, b)| a != b)
        {
            return Err(format!("artifacts differ at byte {}", offset + index as u64).into());
        }
    }
    Ok(length)
}

fn payload_prefix(path: &Path, tokens: usize) -> GateResult<HostPrefix> {
    let mut bytes = vec![0; HEADER_BYTES + tokens * 4];
    File::open(path)?.read_exact(&mut bytes)?;
    Ok(parse_prefix(&bytes)?)
}

fn check_prefix(prefix: &HostPrefix, tokens: &[i32], options: &Options) -> GateResult<()> {
    if prefix.ctx() != options.capacity as u32
        || prefix.fields[PAYLOAD_CHUNK] != options.chunk as u32
        || prefix.fields[PAYLOAD_MTP] != 0
        || prefix.tokens.len() != tokens.len()
        || !prefix
            .tokens
            .iter()
            .zip(tokens)
            .all(|(&a, &b)| a == b as u32)
    {
        return Err("native payload shape, MTP mode or token ledger differs".into());
    }
    Ok(())
}

fn memory(snap: &MemSnap) -> Value {
    let cells: Vec<Value> = snap
        .census
        .cells
        .iter()
        .enumerate()
        .map(|(class, domains)| {
            json!({"class": class, "domains": domains.iter().enumerate().map(|(domain, cell)| {
            json!({"domain":domain,"requested":cell.requested,"committed":cell.committed,
                "freed_requested":cell.freed_requested,"freed_committed":cell.freed_committed,
                "alloc_calls":cell.alloc_calls,"free_calls":cell.free_calls})
        }).collect::<Vec<_>>()})
        })
        .collect();
    json!({"supported":snap.census.supported,"faults":snap.census.faults,"cells":cells,
        "substrate_outstanding":snap.substrate_outstanding,
        "free_bytes":snap.observe.free_bytes,"cuda_free_bytes":snap.observe.cuda_free_bytes,
        "meminfo_available_bytes":snap.observe.meminfo_avail_bytes})
}

fn environment(options: &Options) -> GateResult<Value> {
    for (key, expected) in [
        ("DS4_WEIGHT_RESIDENCY", "mapped"),
        ("DS4_CUDA_WEIGHT_IPC_SCOPE", "base"),
    ] {
        if std::env::var(key).ok().as_deref() != Some(expected) {
            return Err(format!("{key} must be {expected}").into());
        }
    }
    let ipc = std::env::var("DS4_CUDA_WEIGHT_IPC_MANIFEST")?;
    if !Path::new(&ipc).is_file() {
        return Err("IPC manifest does not exist; caller must own a ready weight server".into());
    }
    if std::env::var("DS4_IQUEST_PREFILL_CHUNK")?.parse::<usize>()? != options.chunk {
        return Err("environment prefill chunk differs from --chunk".into());
    }
    Ok(std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("DS4_") || key.starts_with("CUDA_") || key == "LD_LIBRARY_PATH"
        })
        .map(|(key, value)| (key, Value::String(value)))
        .collect::<serde_json::Map<_, _>>()
        .into())
}

fn prompt_tokens(
    model: &Model,
    text: &[u8],
    options: &Options,
) -> GateResult<(TokenBuffer, usize)> {
    let all = match options.mode {
        PromptMode::Raw => TokenBuffer::from_tokens(model.vocab().encode_bytes(text)),
        PromptMode::Chat => model.encode_chat_prompt_bytes(None, text, THINK_NONE)?,
    };
    if all.len() < options.frontier {
        return Err(format!("prompt has {} tokens, need {}", all.len(), options.frontier).into());
    }
    let tokens = all.as_slice()[..options.frontier].to_vec();
    valid_tokens(&tokens, options.frontier, None)?;
    Ok((TokenBuffer::from_tokens(tokens), all.len()))
}

fn save_stage(session: &Session<'_>, options: &Options, name: &str) -> GateResult<Value> {
    let started = Instant::now();
    let logits = session.copy_logits(VOCAB)?;
    write_logits(&options.out.join(format!("{name}.f32")), &logits)?;
    let path = options.out.join(format!("{name}.kv"));
    session.save_payload(&path)?;
    let prefix = payload_prefix(&path, session.host().tokens().len())?;
    check_prefix(&prefix, session.host().tokens(), options)?;
    Ok(
        json!({"position":session.pos(),"logits_values":VOCAB,"logits_finite":true,
        "logits_file":format!("{name}.f32"),"payload_file":format!("{name}.kv"),
        "payload_bytes":fs::metadata(path)?.len(),"payload_header":prefix.fields,
        "diagnostic_save_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn restore_stage(session: &mut Session<'_>, options: &Options, name: &str) -> GateResult<Value> {
    let started = Instant::now();
    let path = options.out.join(format!("{name}.kv"));
    let temporary = options.out.join("restore.tmp.kv");
    let logits = read_logits(&options.out.join(format!("{name}.f32")))?;
    session.invalidate();
    session.load_payload(&path)?;
    equal_logits(&logits, &session.copy_logits(VOCAB)?)?;
    let prefix = payload_prefix(&path, session.host().tokens().len())?;
    check_prefix(&prefix, session.host().tokens(), options)?;
    if session.pos() as usize != prefix.tokens.len() {
        return Err("restore changed native position".into());
    }
    session.save_payload(&temporary)?;
    let bytes = compare_files(&path, &temporary)?;
    fs::remove_file(temporary)?;
    Ok(
        json!({"stage":name,"position":session.pos(),"payload_bytes":bytes,
        "full_vocab_logits_bit_exact":true,"payload_byte_exact":true,
        "diagnostic_restore_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn decode(
    session: &mut Session<'_>,
    options: &Options,
    fixed: Option<&[i32]>,
    eos: i32,
) -> GateResult<Value> {
    let started = Instant::now();
    let mut ids = Vec::with_capacity(options.steps);
    let mut mismatches = Vec::new();
    let mut rows = BufWriter::new(File::create(options.out.join("steps.jsonl"))?);
    for index in 0..options.steps {
        let selected = session.argmax_excluding(eos);
        valid_tokens(&[selected], 1, Some(eos))?;
        let token = fixed.map_or(selected, |tokens| tokens[index]);
        if selected != token {
            mismatches.push(json!({"step":index,"argmax":selected,"forced":token}));
        }
        session.eval(token)?;
        if session.pos() as usize != options.frontier + index + 1 {
            return Err("ordinary eval did not advance exactly one committed row".into());
        }
        // Full readback catches transient nonfinite rows that later recover.
        finite_logits(&session.copy_logits(VOCAB)?)?;
        ids.push(token);
        serde_json::to_writer(
            &mut rows,
            &json!({"step":index,"token":token,
            "argmax_excluding_eos":selected,"position":session.pos(),"logits_finite":true}),
        )?;
        rows.write_all(b"\n")?;
        rows.flush()?;
    }
    write_json(&options.out.join("tokens.json"), &json!(ids))?;
    Ok(
        json!({"token_source":if fixed.is_some() {"fixed_json"} else {"greedy_excluding_eos"},
        "committed_tokens":ids.len(),"all_steps_full_vocab_finite":true,
        "greedy_matches_forced":mismatches.is_empty(),"greedy_mismatches":mismatches,
        "diagnostic_decode_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn record(options: &Options) -> GateResult<Value> {
    let wall = Instant::now();
    let env = environment(options)?;
    let optimization = optimization(&env, &options.control)?;
    let text = read_bounded(&options.prompt, MAX_PROMPT_BYTES)?;
    let fixed: Option<Vec<i32>> = options
        .tokens
        .as_ref()
        .map(|path| {
            let bytes = read_bounded(path, MAX_JSON_BYTES)?;
            Ok::<_, Box<dyn std::error::Error>>(serde_json::from_slice(&bytes)?)
        })
        .transpose()?;
    if let Some(tokens) = &fixed {
        valid_tokens(tokens, options.steps, None)?;
    }
    let before = snapshot_mem();
    let spec_before = snapshot_spec();
    let started = Instant::now();
    let model = Model::open_configured(
        &options.model,
        Backend::Cuda,
        0,
        false,
        None,
        &[
            ModelOpenOption::MtpDraftTokens(1),
            ModelOpenOption::MtpMargin(0.0),
        ],
    )?;
    let model_seconds = started.elapsed().as_secs_f64();
    if model.family() != ModelFamily::IQuestQ1
        || model.mtp_draft_tokens() != 1
        || model.vocab().n_vocab() as usize != VOCAB
    {
        return Err("expected the canonical IQuest artifact with MTP disabled".into());
    }
    let eos = model.token_eos();
    valid_tokens(&[eos], 1, None)?;
    if let Some(tokens) = &fixed {
        valid_tokens(tokens, options.steps, Some(eos))?;
    }
    let (prompt, source_tokens) = prompt_tokens(&model, prompt_prefix(&text), options)?;
    write_json(
        &options.out.join("prompt.tokens.json"),
        &json!(prompt.as_slice()),
    )?;
    let mut session = model.session(options.capacity as i32)?;
    let started = Instant::now();
    session.sync(&prompt)?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    if session.pos() as usize != options.frontier {
        return Err("wrong prefill position".into());
    }
    let prefill = save_stage(&session, options, "prefill")?;
    let after_prefill = snapshot_mem();
    let decoded = decode(&mut session, options, fixed.as_deref(), eos)?;
    let final_state = save_stage(&session, options, "final")?;
    let after_decode = snapshot_mem();
    // Preserve the resident decode path: restore only after both stages exist.
    let restore_final = restore_stage(&mut session, options, "final")?;
    let restore_prefill = restore_stage(&mut session, options, "prefill")?;
    let after_restore = snapshot_mem();
    let spec_after = snapshot_spec();
    drop(session);
    drop(model);
    let after_drop = snapshot_mem();
    let fault_clean = [after_prefill, after_decode, after_restore, after_drop]
        .iter()
        .all(|snap| snap.census.faults == before.census.faults);
    let passed =
        fault_clean && spec_before == spec_after && decoded["greedy_matches_forced"] == true;
    let ring = options.capacity.min(MAIN_WINDOW + options.chunk - 1);
    Ok(
        json!({"schema":2,"passed":passed,"scope":"ordinary target-state diagnostic; not throughput or family-wide quality",
        "model":options.model,"prompt_file":options.prompt,"prompt_mode":options.mode.name(),
        "source_prompt_tokens":source_tokens,"frontier":options.frontier,"capacity":options.capacity,
        "steps":options.steps,"chunk":options.chunk,"vocab":VOCAB,"eos":eos,
        "mtp_enabled":false,"mtp_draft":1,"environment":env,"optimization":optimization,
        "decode_before_any_restore":true,"prefill":prefill,"final":final_state,"decode":decoded,
        "restores":[restore_final,restore_prefill],"faults_unchanged":fault_clean,
        "speculative_counters_unchanged":spec_before==spec_after,
        "speculative_before":{"drafts":spec_before.drafts,"hits":spec_before.hits,"quench":spec_before.quench},
        "speculative_after":{"drafts":spec_after.drafts,"hits":spec_after.hits,"quench":spec_after.quench},
        "main_swa_window":MAIN_WINDOW,"main_swa_capacity":ring,
        "decode_crosses_physical_ring":(options.frontier - 1)/ring != (options.frontier + options.steps - 1)/ring,
        "memory":{"before":memory(&before),"after_prefill":memory(&after_prefill),
            "after_decode":memory(&after_decode),"after_restore":memory(&after_restore),"after_drop":memory(&after_drop),
            "session_tensor_quote_bytes":IQuestPlan::session_bytes(options.capacity as u32,options.chunk as u32),
            "target_kv_quote_bytes":IQuestPlan::kv_bytes(options.capacity as u32,options.chunk as u32,IQuestCache::Target),
            "full_payload_host_snapshots":0,"comparison_buffers_bytes":2*IO_CHUNK,
            "peak_temporary_payload_files":1},
        "diagnostic_seconds":{"model":model_seconds,"prefill":prefill_seconds,"total":wall.elapsed().as_secs_f64()}}),
    )
}

fn optimization(env: &Value, control: &str) -> GateResult<Value> {
    if !OPT_CONTROLS.contains(&control) {
        return Err("unknown IQuest optimization switch".into());
    }
    let arm = match env[control].as_str() {
        Some("0") => "off",
        Some("1") => "on",
        _ => return Err(format!("{control} must be explicitly 0 or 1").into()),
    };
    Ok(json!({"control":control,"arm":arm}))
}

fn compare_arms(a: &Value, b: &Value) -> GateResult<Value> {
    let control = a["optimization"]["control"]
        .as_str()
        .ok_or("recording must name its optimization switch")?;
    if b["optimization"]["control"].as_str() != Some(control) {
        return Err("recorded optimization switches differ or are absent".into());
    }
    let left = optimization(&a["environment"], control)?;
    let right = optimization(&b["environment"], control)?;
    if left != a["optimization"] || right != b["optimization"] {
        return Err("recorded optimization arm differs from its environment".into());
    }
    if left["arm"] == right["arm"] {
        return Err("opposite optimization arms are required".into());
    }
    // Only the targeted switch may differ; duplicate controls cannot prove A/B.
    let mut left_env = a["environment"].as_object().unwrap().clone();
    let mut right_env = b["environment"].as_object().unwrap().clone();
    left_env.remove(control);
    right_env.remove(control);
    if left_env != right_env {
        return Err("non-target environment values differ".into());
    }
    Ok(json!({"control":control,"left_arm":left["arm"],"right_arm":right["arm"]}))
}

fn compare(left: &Path, right: &Path) -> GateResult<Value> {
    if fs::canonicalize(left)? == fs::canonicalize(right)? {
        return Err("independent run directories are required".into());
    }
    let a: Value =
        serde_json::from_slice(&read_bounded(&left.join("report.json"), MAX_JSON_BYTES)?)?;
    let b: Value =
        serde_json::from_slice(&read_bounded(&right.join("report.json"), MAX_JSON_BYTES)?)?;
    if a["passed"] != true
        || b["passed"] != true
        || a["mtp_enabled"] != false
        || b["mtp_enabled"] != false
    {
        return Err("both recordings must pass with MTP disabled".into());
    }
    for key in [
        "schema",
        "model",
        "prompt_mode",
        "frontier",
        "capacity",
        "steps",
        "chunk",
        "vocab",
        "eos",
    ] {
        if a[key].is_null() || a[key] != b[key] {
            return Err(format!("recording field {key} differs or is absent").into());
        }
    }
    let optimization = compare_arms(&a, &b)?;
    for name in ["prompt.tokens.json", "tokens.json"] {
        compare_files(&left.join(name), &right.join(name))?;
    }
    let mut stages = Vec::new();
    for name in ["prefill", "final"] {
        let logits = format!("{name}.f32");
        equal_logits(
            &read_logits(&left.join(&logits))?,
            &read_logits(&right.join(&logits))?,
        )?;
        let bytes = compare_files(
            &left.join(format!("{name}.kv")),
            &right.join(format!("{name}.kv")),
        )?;
        stages.push(
            json!({"stage":name,"logit_values":VOCAB,"logits_finite":true,
            "logits_bit_exact":true,"payload_bytes":bytes,"payload_byte_exact":true}),
        );
    }
    Ok(
        json!({"schema":2,"passed":true,"left":left,"right":right,"stages":stages,
        "optimization":optimization,
        "tokens_identical":true,"scope":"recorded ordinary target-state parity; no throughput claim"}),
    )
}

fn main() -> GateResult<()> {
    match parse_args(&std::env::args().skip(1).collect::<Vec<_>>())? {
        Command::Help => print!("{HELP}"),
        Command::Compare(left, right) => {
            let report = compare(&left, &right)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Record(options) => {
            // Refuse overwriting old evidence, including failed recordings.
            fs::create_dir(&options.out)?;
            match record(&options) {
                Ok(report) => {
                    write_json(&options.out.join("report.json"), &report)?;
                    println!(
                        "{}",
                        serde_json::to_string(
                            &json!({"passed":report["passed"],"out":options.out})
                        )?
                    );
                    if report["passed"] != true {
                        return Err("state recording failed; inspect report.json".into());
                    }
                }
                Err(error) => {
                    write_json(
                        &options.out.join("report.json"),
                        &json!({"passed":false,"error":error.to_string()}),
                    )?;
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| (*s).into()).collect()
    }

    #[test]
    fn parses_raw_and_chat() {
        let Command::Record(options) = parse_args(&args(&[
            "model",
            "out",
            "--prompt-file",
            "prompt",
            "--control",
            OPT_CONTROLS[0],
        ]))
        .unwrap() else {
            panic!("expected record");
        };
        assert_eq!(options.mode, PromptMode::Raw);
        assert_eq!(
            (
                options.frontier,
                options.capacity,
                options.steps,
                options.chunk
            ),
            (2048, 8192, 32, 128)
        );
        let Command::Record(options) = parse_args(&args(&[
            "model",
            "out",
            "--prompt-file",
            "prompt",
            "--prompt-mode",
            "chat",
            "--frontier",
            "4220",
            "--steps",
            "128",
            "--tokens-file",
            "ids.json",
            "--control",
            OPT_CONTROLS[0],
        ]))
        .unwrap() else {
            panic!("expected record");
        };
        assert_eq!(options.mode, PromptMode::Chat);
        assert_eq!(options.tokens.unwrap(), PathBuf::from("ids.json"));
    }

    #[test]
    fn rejects_invalid_workloads() {
        for extra in [
            ["--capacity", "2048"],
            ["--steps", "0"],
            ["--frontier", "0"],
            ["--chunk", "8193"],
            ["--prompt-mode", "other"],
            ["--steps", "18446744073709551615"],
        ] {
            let mut input = args(&[
                "model",
                "out",
                "--prompt-file",
                "prompt",
                "--control",
                OPT_CONTROLS[0],
            ]);
            input.extend(args(&extra));
            assert!(parse_args(&input).is_err(), "{extra:?}");
        }
        assert!(parse_args(&args(&["model", "out", "--steps", "32"])).is_err());
        assert!(parse_args(&args(&["model", "out", "--prompt-file", "prompt"])).is_err());
    }

    #[test]
    fn compare_help_need_no_model() {
        assert!(matches!(
            parse_args(&args(&["--help"])).unwrap(),
            Command::Help
        ));
        assert!(matches!(
            parse_args(&args(&["--compare", "left", "right"])).unwrap(),
            Command::Compare(..)
        ));
        assert!(parse_args(&args(&["--compare", "left"])).is_err());
    }

    #[test]
    fn prompt_matches_bench_nul() {
        assert_eq!(prompt_prefix(b"hello\0ignored"), b"hello");
        assert_eq!(prompt_prefix(b"\0ignored"), b"");
        assert_eq!(prompt_prefix(b"hello"), b"hello");
    }

    #[test]
    fn token_contract_checks_eos() {
        assert!(valid_tokens(&[1, 2], 2, Some(3)).is_ok());
        for tokens in [vec![-1, 2], vec![VOCAB as i32, 2], vec![1, 3], vec![1]] {
            assert!(valid_tokens(&tokens, 2, Some(3)).is_err());
        }
    }

    #[test]
    fn logits_use_bits_and_finite() {
        let mut a = vec![1.0; VOCAB];
        let mut b = a.clone();
        assert!(equal_logits(&a, &b).is_ok());
        a[1] = 0.0;
        b[1] = -0.0;
        assert!(equal_logits(&a, &b).is_err());
        a[1] = f32::NAN;
        b[1] = f32::NAN;
        assert!(equal_logits(&a, &b).is_err());
        assert!(finite_logits(&vec![0.0; VOCAB]).is_err());
        assert!(finite_logits(&[1.0]).is_err());
    }

    struct CompareFixture {
        root: PathBuf,
        left: PathBuf,
        right: PathBuf,
        report: Value,
    }

    impl CompareFixture {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir()
                .join(format!("ds4-iquest-compare-{}-{nonce}", std::process::id()));
            let left = root.join("left");
            let right = root.join("right");
            let report = json!({
                "schema":2,"passed":true,"mtp_enabled":false,"model":"fixture",
                "prompt_mode":"raw","frontier":128,"capacity":256,"steps":1,
                "chunk":128,"vocab":VOCAB,"eos":0,
                "environment":{"DS4_IQUEST_ATTN_SHUFFLE":"1",
                    "DS4_IQUEST_ATTN_ASYNC":"0"},
                "optimization":{"control":"DS4_IQUEST_ATTN_ASYNC","arm":"off"}
            });
            for dir in [&left, &right] {
                fs::create_dir_all(dir).unwrap();
                write_json(&dir.join("report.json"), &report).unwrap();
                write_json(&dir.join("prompt.tokens.json"), &json!(vec![1; 128])).unwrap();
                write_json(&dir.join("tokens.json"), &json!([2])).unwrap();
                for stage in ["prefill", "final"] {
                    write_logits(&dir.join(format!("{stage}.f32")), &vec![1.0; VOCAB]).unwrap();
                    fs::write(dir.join(format!("{stage}.kv")), b"native fixture").unwrap();
                }
            }
            Self {
                root,
                left,
                right,
                report,
            }
        }

        fn candidate(&self) -> Value {
            let mut report = self.report.clone();
            report["environment"]["DS4_IQUEST_ATTN_ASYNC"] = json!("1");
            report["optimization"]["arm"] = json!("on");
            report
        }

        fn write_right(&self, report: &Value) {
            write_json(&self.right.join("report.json"), report).unwrap();
        }
    }

    impl Drop for CompareFixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[test]
    fn compare_requires_opposite_arms() {
        let fixture = CompareFixture::new();
        let error = compare(&fixture.left, &fixture.right).unwrap_err();
        assert!(error.to_string().contains("opposite"), "{error}");

        fixture.write_right(&fixture.candidate());
        assert_eq!(
            compare(&fixture.left, &fixture.right).unwrap()["passed"],
            true
        );
        assert_eq!(
            compare(&fixture.right, &fixture.left).unwrap()["passed"],
            true
        );
    }

    #[test]
    fn optimization_needs_explicit_value() {
        for control in OPT_CONTROLS {
            for value in [json!("0"), json!("1")] {
                let env = json!({control:value});
                assert_eq!(optimization(&env, control).unwrap()["control"], control);
            }
            for value in [Value::Null, json!(0), json!(true), json!(""), json!("2")] {
                assert!(optimization(&json!({control:value}), control).is_err());
            }
            assert!(optimization(&json!({}), control).is_err());
        }
        assert!(optimization(
            &json!({"DS4_IQUEST_PREFILL_CHUNK":"1"}),
            "DS4_IQUEST_PREFILL_CHUNK"
        )
        .is_err());
    }

    #[test]
    fn compare_checks_switch_provenance() {
        let fixture = CompareFixture::new();
        for (field, value) in [
            ("optimization", Value::Null),
            ("optimization/control", json!("DS4_IQUEST_ATTN_WARP")),
            ("optimization/arm", json!("off")),
            ("environment/DS4_IQUEST_ATTN_ASYNC", Value::Null),
            ("environment/DS4_IQUEST_ATTN_SHUFFLE", json!("0")),
        ] {
            let mut report = fixture.candidate();
            *report.pointer_mut(&format!("/{field}")).unwrap() = value;
            fixture.write_right(&report);
            assert!(compare(&fixture.left, &fixture.right).is_err(), "{field}");
        }
        let mut control_on = fixture.candidate();
        write_json(&fixture.left.join("report.json"), &control_on).unwrap();
        fixture.write_right(&control_on);
        assert!(compare(&fixture.left, &fixture.right).is_err());
        control_on["optimization"] = json!({"control":"UNKNOWN","arm":"on"});
        write_json(&fixture.left.join("report.json"), &control_on).unwrap();
        fixture.write_right(&control_on);
        assert!(compare(&fixture.left, &fixture.right).is_err());
    }
}
