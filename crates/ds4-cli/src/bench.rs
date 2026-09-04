use ds4_core::{
    Backend, Model, ModelFamily, ModelOpenOption, Session, SessionSnapshot, TokenBuffer,
};
use std::io::{BufWriter, Write};
use std::time::Instant;

const CSV_HEADER: &str = "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_tps_ss,first_token_sec,kvcache_bytes";
const THINK_NONE: i32 = 0;
const PROMPT_SELECTION_ERR: &str = "specify exactly one of --prompt-file or --chat-prompt-file";

#[derive(Debug)]
pub struct BenchArgs {
    model: String,
    prompt_file: Option<String>,
    chat_prompt_file: Option<String>,
    system: Option<String>,
    backend: Backend,
    threads: i32,
    ctx_start: i32,
    ctx_max: i32,
    ctx_alloc: i32,
    step_incr: i32,
    gen_tokens: i32,
    step_mul: f64,
    csv: Option<String>,
    quality: bool,
    warm_weights: bool,
    power_percent: i32,
    mtp: Option<String>,
    mtp_draft: i32,
    mtp_margin: f32,
    output_head_bench_iters: i32,
    dump_frontier_logits_dir: Option<String>,
    dist: ds4_dist::Options,
    help: bool,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            model: "ds4flash.gguf".into(),
            prompt_file: None,
            chat_prompt_file: None,
            system: Some("You are a helpful assistant.".into()),
            backend: default_backend(),
            threads: 0,
            ctx_start: 2048,
            ctx_max: 32768,
            ctx_alloc: 0,
            step_incr: 2048,
            gen_tokens: 128,
            step_mul: 1.0,
            csv: None,
            quality: false,
            warm_weights: false,
            power_percent: 100,
            mtp: None,
            mtp_draft: 1,
            mtp_margin: 3.0,
            output_head_bench_iters: 0,
            dump_frontier_logits_dir: None,
            dist: ds4_dist::Options::default(),
            help: false,
        }
    }
}

struct BenchRow {
    ctx_tokens: i32,
    prefill_tokens: i32,
    prefill_tps: f64,
    gen_tokens: i32,
    gen_tps: f64,
    gen_tps_ss: f64,
    first_token_sec: f64,
    kvcache_bytes: u64,
}

impl BenchRow {
    fn csv_line(&self) -> String {
        format!(
            "{},{},{:.2},{},{:.2},{:.2},{:.4},{}",
            self.ctx_tokens,
            self.prefill_tokens,
            self.prefill_tps,
            self.gen_tokens,
            self.gen_tps,
            self.gen_tps_ss,
            self.first_token_sec,
            self.kvcache_bytes,
        )
    }
}

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<BenchArgs, String> {
    let mut parsed = BenchArgs::default();
    let mut iter = args.into_iter();
    let _argv0 = iter.next();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                parsed.help = true;
                return Ok(parsed);
            }
            "-m" | "--model" => parsed.model = require_value(&arg, iter.next())?,
            "--prompt-file" => parsed.prompt_file = Some(require_value(&arg, iter.next())?),
            "--chat-prompt-file" => {
                parsed.chat_prompt_file = Some(require_value(&arg, iter.next())?);
            }
            "-sys" | "--system" => parsed.system = Some(require_value(&arg, iter.next())?),
            "--backend" => parsed.backend = parse_backend(&require_value(&arg, iter.next())?)?,
            "--cuda" => parsed.backend = Backend::Cuda,
            "--metal" => parsed.backend = Backend::Metal,
            "--cpu" => parsed.backend = Backend::Cpu,
            "-t" | "--threads" => {
                parsed.threads = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--ctx-start" => {
                parsed.ctx_start = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--ctx-max" => {
                parsed.ctx_max = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--ctx-alloc" => {
                parsed.ctx_alloc = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--step-incr" => {
                parsed.step_incr = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--step-mul" => {
                parsed.step_mul = parse_f64(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--gen-tokens" | "--tokens" | "-n" => {
                parsed.gen_tokens =
                    parse_nonnegative_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--csv" => parsed.csv = Some(require_value(&arg, iter.next())?),
            "--quality" => parsed.quality = true,
            "--warm-weights" => parsed.warm_weights = true,
            "--power" => {
                parsed.power_percent = parse_power(&require_value(&arg, iter.next())?)?;
            }
            "--mtp" => parsed.mtp = Some(require_value(&arg, iter.next())?),
            "--mtp-draft" => {
                parsed.mtp_draft = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--mtp-margin" => {
                let value = require_value(&arg, iter.next())?;
                let parsed_margin = parse_f64(&arg, &value)?;
                if !(0.0..=1000.0).contains(&parsed_margin) {
                    return Err(format!("invalid value for {arg}: {parsed_margin}"));
                }
                parsed.mtp_margin = parsed_margin as f32;
            }
            "--output-head-bench" => {
                parsed.output_head_bench_iters =
                    parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--dump-frontier-logits-dir" => {
                parsed.dump_frontier_logits_dir = Some(require_value(&arg, iter.next())?);
            }
            _ => match ds4_dist::parse_cli_arg(&arg, &mut iter, &mut parsed.dist)? {
                ds4_dist::CliResult::Matched => {}
                ds4_dist::CliResult::NotMatched => {
                    return Err(format!("unsupported ds4-bench-rs option: {arg}"));
                }
                ds4_dist::CliResult::Error => unreachable!(),
            },
        }
    }

    if parsed.prompt_file.is_some() == parsed.chat_prompt_file.is_some() {
        return Err(PROMPT_SELECTION_ERR.into());
    }
    if parsed.ctx_start > parsed.ctx_max {
        return Err("--ctx-start must be <= --ctx-max".into());
    }
    if parsed.step_mul < 1.0 {
        return Err("--step-mul must be >= 1".into());
    }
    let measured_ctx = if parsed.output_head_bench_iters > 0 {
        parsed.ctx_start
    } else {
        parsed.ctx_max
    };
    let decode_spare = i32::from(parsed.gen_tokens > 0);
    let required_ctx = measured_ctx
        .checked_add(parsed.gen_tokens)
        .and_then(|ctx| ctx.checked_add(decode_spare))
        .ok_or_else(|| "requested context is too large".to_string())?;
    if parsed.ctx_alloc == 0 {
        parsed.ctx_alloc = required_ctx;
    }
    if parsed.ctx_alloc < required_ctx {
        return Err("--ctx-alloc is too small for the measured context and generation".into());
    }
    ds4_dist::prepare_engine_options(&parsed.dist)?;
    if parsed.dist.role == ds4_dist::Role::Worker {
        return Err("--role worker is a serving mode; start workers with ./ds4".into());
    }
    Ok(parsed)
}

fn uses_distributed_replay(args: &BenchArgs) -> bool {
    args.dist.role == ds4_dist::Role::Coordinator
}

fn use_mtp_spec(family: ModelFamily, mtp: Option<&str>, draft: i32) -> bool {
    draft > 1
        && (mtp.is_some() || family == ModelFamily::Qwen4Exp)
        && std::env::var_os("DS4_MTP_SPEC_DISABLE").is_none()
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda => "cuda",
        Backend::Metal => "metal",
        Backend::Cpu => "cpu",
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_f32(v: f32) -> String {
    if v.is_finite() {
        v.to_string()
    } else {
        "null".into()
    }
}

fn frontier_logits_path(dir: &str, frontier: i32) -> String {
    format!("{dir}/frontier_{frontier:06}.logits.json")
}

fn write_frontier_logits_json(
    args: &BenchArgs,
    model: &Model,
    session: &Session<'_>,
    frontier: i32,
    previous: i32,
) -> Result<(), String> {
    let Some(dir) = args.dump_frontier_logits_dir.as_deref() else {
        return Ok(());
    };
    let vocab = model.vocab().n_vocab() as usize;
    let logits = session
        .copy_logits(vocab)
        .map_err(|_| format!("failed to copy frontier logits at {frontier}"))?;
    let argmax = session.argmax();
    if !(0..logits.len() as i32).contains(&argmax) {
        return Err(format!("failed to copy frontier logits at {frontier}"));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("failed to create {dir}: {e}"))?;
    let path = frontier_logits_path(dir, frontier);
    let mut body = String::new();
    body.push_str("{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    body.push_str(&json_escape(&args.model));
    body.push_str(&format!(
        ",\n  \"backend\":\"{}\",\n  \"quality\":{},\n  \"quant_bits\":{},\n  \"prompt_tokens\":{},\n  \"frontier_tokens\":{},\n  \"prefill_tokens\":{},\n  \"ctx\":{},\n  \"vocab\":{},\n  \"argmax_id\":{},\n  \"argmax_logit\":{},\n  \"logits\":[",
        backend_name(args.backend),
        if args.quality { "true" } else { "false" },
        model.routed_quant_bits(),
        frontier,
        frontier,
        frontier - previous,
        args.ctx_alloc,
        vocab,
        argmax,
        json_f32(logits[argmax as usize]),
    ));
    for (i, logit) in logits.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        if i % 8 == 0 {
            body.push_str("\n    ");
        }
        body.push_str(&json_f32(*logit));
    }
    body.push_str("\n  ]\n}\n");
    std::fs::write(&path, body).map_err(|e| format!("failed to write {path}: {e}"))?;
    Ok(())
}

fn prompt_source(args: &BenchArgs) -> Result<(&str, bool), String> {
    match (
        args.prompt_file.as_deref(),
        args.chat_prompt_file.as_deref(),
    ) {
        (Some(path), None) => Ok((path, false)),
        (None, Some(path)) => Ok((path, true)),
        _ => Err(PROMPT_SELECTION_ERR.into()),
    }
}

fn read_prompt(path: &str) -> Result<Vec<u8>, String> {
    let mut bytes = std::fs::read(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    if let Some(nul) = bytes.iter().position(|byte| *byte == 0) {
        bytes.truncate(nul);
    }
    Ok(bytes)
}

fn wait_distributed_route(session: &Session<'_>) -> Result<(), String> {
    let mut ticks = 0u32;
    loop {
        if session
            .distributed_route_ready()
            .map_err(|e| format!("distributed route readiness failed: {e}"))?
        {
            if ticks != 0 {
                eprintln!("ds4-bench-rs: distributed route ready");
            }
            return Ok(());
        }
        if ticks % 20 == 0 {
            eprintln!("ds4-bench-rs: waiting for distributed route: route incomplete");
        }
        ticks = ticks.wrapping_add(1);
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

fn maybe_warn_distributed_step_shape(args: &BenchArgs, session: &Session<'_>) {
    let chunk = if args.dist.prefill_chunk != 0 {
        args.dist.prefill_chunk
    } else {
        session.host().prefill_cap
    };
    if chunk != 0
        && args.step_mul == 1.0
        && args.step_incr > 0
        && (args.step_incr as u32) < chunk
        && args.ctx_start < args.ctx_max
    {
        eprintln!(
            "ds4-bench-rs: note: --step-incr={} is smaller than distributed prefill chunk {}; suffix rows will not show multi-chunk pipeline overlap",
            args.step_incr, chunk
        );
    }
}

pub fn run(args: BenchArgs) -> Result<i32, String> {
    if args.help {
        print!("{}\nDistributed:\n{}", help_text(), ds4_dist::USAGE);
        return Ok(0);
    }

    let (prompt_path, chat_prompt) = prompt_source(&args)?;
    let text = read_prompt(prompt_path)?;
    let native_dist = crate::distributed_config(&args.dist);
    let mut open_options = Vec::with_capacity(5);
    if args.quality {
        open_options.push(ModelOpenOption::Quality);
    }
    if args.warm_weights {
        open_options.push(ModelOpenOption::WarmWeights);
    }
    open_options.push(ModelOpenOption::PowerPercent(args.power_percent as u8));
    open_options.push(ModelOpenOption::MtpDraftTokens(args.mtp_draft));
    open_options.push(ModelOpenOption::MtpMargin(args.mtp_margin));
    let model = if let Some(config) = native_dist.as_ref() {
        Model::open_distributed_options(
            &args.model,
            args.backend,
            args.threads,
            false,
            args.mtp.as_deref(),
            None,
            config,
            &open_options,
        )
    } else if args.mtp.is_some() {
        Model::open_with_support_options(
            &args.model,
            args.backend,
            args.threads,
            false,
            args.mtp.as_deref(),
            None,
            &open_options,
        )
    } else {
        Model::open_configured(
            &args.model,
            args.backend,
            args.threads,
            false,
            None,
            &open_options,
        )
    }
    .map_err(|e| e.to_string())?;
    let prompt = if chat_prompt {
        model
            .encode_chat_prompt_bytes(args.system.as_deref().map(str::as_bytes), &text, THINK_NONE)
            .map_err(|e| e.to_string())?
    } else {
        TokenBuffer::from_tokens(model.vocab().encode_bytes(&text))
    };
    let needed_prompt = if args.output_head_bench_iters > 0 {
        args.ctx_start
    } else {
        args.ctx_max
    };
    if prompt.len() < needed_prompt as usize {
        return Err(format!(
            "prompt has {} tokens, need at least {}",
            prompt.len(),
            needed_prompt
        ));
    }

    let mut session = model.session(args.ctx_alloc).map_err(|e| e.to_string())?;
    if uses_distributed_replay(&args) {
        wait_distributed_route(&session)?;
        maybe_warn_distributed_step_shape(&args, &session);
    }
    if args.output_head_bench_iters > 0 {
        let prefix =
            TokenBuffer::from_tokens(prompt.as_slice()[..args.ctx_start as usize].to_vec());
        session.sync(&prefix).map_err(|e| e.to_string())?;
        session
            .output_head_bench(args.output_head_bench_iters, args.csv.as_deref())
            .map_err(|e| format!("output-head bench failed: {e}"))?;
        return Ok(0);
    }
    let mut snapshot = if uses_distributed_replay(&args) {
        None
    } else {
        Some(SessionSnapshot::new().map_err(|e| e.to_string())?)
    };

    if let Some(path) = args.csv.as_deref() {
        let file =
            std::fs::File::create(path).map_err(|e| format!("failed to open {path}: {e}"))?;
        let mut out = BufWriter::new(file);
        run_sweep(
            &args,
            &model,
            &prompt,
            &mut session,
            &mut snapshot,
            &mut out,
        )?;
    } else {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        run_sweep(
            &args,
            &model,
            &prompt,
            &mut session,
            &mut snapshot,
            &mut out,
        )?;
    }
    Ok(0)
}

fn run_sweep<W: Write>(
    args: &BenchArgs,
    model: &Model,
    prompt: &TokenBuffer,
    session: &mut Session<'_>,
    snapshot: &mut Option<SessionSnapshot>,
    out: &mut W,
) -> Result<(), String> {
    writeln!(out, "{CSV_HEADER}").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;

    let eos = model.token_eos();
    let use_mtp = use_mtp_spec(model.family(), args.mtp.as_deref(), args.mtp_draft);
    let distributed = uses_distributed_replay(args);
    let mut previous = 0;
    let mut frontier = args.ctx_start;

    loop {
        let prefix = TokenBuffer::from_tokens(prompt.as_slice()[..frontier as usize].to_vec());
        let prefill_t0 = Instant::now();
        session.sync(&prefix).map_err(|e| e.to_string())?;
        let prefill_sec = prefill_t0.elapsed().as_secs_f64();
        let prefill_tokens = frontier - previous;
        write_frontier_logits_json(args, model, session, frontier, previous)?;

        if args.gen_tokens > 0 && !distributed {
            let snapshot = snapshot
                .as_mut()
                .ok_or_else(|| "local bench snapshot is missing".to_string())?;
            session
                .save_snapshot(snapshot)
                .map_err(|e| format!("snapshot at {frontier} failed: {e}"))?;
        }

        let gen_t0 = Instant::now();
        let mut generated = 0;
        let mut after_first = None;
        let mut first_call_tokens = 0;
        while generated < args.gen_tokens {
            if session.pos().saturating_add(1) >= session.ctx() {
                return Err(format!(
                    "generation would exceed allocated context at frontier {frontier}"
                ));
            }
            let token = session.argmax_excluding(eos);
            if token < 0 {
                return Err(format!(
                    "failed to choose non-EOS token at frontier {frontier}"
                ));
            }
            if use_mtp {
                let accepted = session
                    .eval_speculative_argmax(token, args.gen_tokens - generated, eos)
                    .map_err(|e| {
                        format!("speculative decode at frontier {frontier} failed: {e}")
                    })?;
                for t in accepted {
                    if t != eos && generated < args.gen_tokens {
                        generated += 1;
                    }
                }
            } else {
                session
                    .eval(token)
                    .map_err(|e| format!("decode at frontier {frontier} failed: {e}"))?;
                generated += 1;
            }
            if after_first.is_none() {
                after_first = Some(Instant::now());
                first_call_tokens = generated;
            }
        }
        let gen_t1 = Instant::now();

        if args.gen_tokens > 0 && frontier < args.ctx_max {
            if distributed {
                session
                    .sync(&prefix)
                    .map_err(|e| format!("distributed replay restore at {frontier} failed: {e}"))?;
            } else {
                let snapshot = snapshot
                    .as_ref()
                    .ok_or_else(|| "local bench snapshot is missing".to_string())?;
                session
                    .load_snapshot(snapshot)
                    .map_err(|e| format!("restore at {frontier} failed: {e}"))?;
            }
        }

        let gen_sec = gen_t1.duration_since(gen_t0).as_secs_f64();
        let first_token_sec = after_first
            .map(|t| t.duration_since(gen_t0).as_secs_f64())
            .unwrap_or(0.0);
        let ss_sec = after_first
            .map(|t| gen_t1.duration_since(t).as_secs_f64())
            .unwrap_or(0.0);
        let ss_tokens = args.gen_tokens - first_call_tokens;
        let row = BenchRow {
            ctx_tokens: frontier,
            prefill_tokens,
            prefill_tps: rate(prefill_tokens, prefill_sec),
            gen_tokens: args.gen_tokens,
            gen_tps: rate(args.gen_tokens, gen_sec),
            gen_tps_ss: rate(ss_tokens, ss_sec),
            first_token_sec,
            kvcache_bytes: if distributed {
                0
            } else {
                snapshot.as_ref().map_or(0, SessionSnapshot::len)
            },
        };
        writeln!(out, "{}", row.csv_line()).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;

        previous = frontier;
        if frontier >= args.ctx_max {
            break;
        }
        frontier = next_frontier(args, frontier);
    }
    Ok(())
}

fn rate(tokens: i32, seconds: f64) -> f64 {
    if tokens > 0 && seconds > 0.0 {
        f64::from(tokens) / seconds
    } else {
        0.0
    }
}

fn next_frontier(args: &BenchArgs, cur: i32) -> i32 {
    if cur >= args.ctx_max {
        return args.ctx_max;
    }
    let next = if args.step_mul == 1.0 {
        cur.checked_add(args.step_incr).unwrap_or(args.ctx_max)
    } else {
        let value = (f64::from(cur) * args.step_mul).ceil();
        if value > f64::from(i32::MAX) {
            args.ctx_max
        } else {
            let next = value as i32;
            if next <= cur {
                cur + 1
            } else {
                next
            }
        }
    };
    next.min(args.ctx_max)
}

fn require_value(flag: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_positive_i32(flag: &str, value: &str) -> Result<i32, String> {
    let parsed = value
        .parse::<i32>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))?;
    if parsed <= 0 {
        return Err(format!("invalid value for {flag}: {value}"));
    }
    Ok(parsed)
}

fn parse_nonnegative_i32(flag: &str, value: &str) -> Result<i32, String> {
    let parsed = value
        .parse::<i32>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))?;
    if parsed < 0 {
        return Err(format!("invalid value for {flag}: {value}"));
    }
    Ok(parsed)
}

fn parse_f64(flag: &str, value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))?;
    if !parsed.is_finite() {
        return Err(format!("invalid value for {flag}: {value}"));
    }
    Ok(parsed)
}

fn parse_power(value: &str) -> Result<i32, String> {
    let parsed = parse_positive_i32("--power", value)?;
    if parsed > 100 {
        return Err("--power must be between 1 and 100".into());
    }
    Ok(parsed)
}

fn parse_backend(value: &str) -> Result<Backend, String> {
    match value {
        "cuda" => Ok(Backend::Cuda),
        "metal" => Ok(Backend::Metal),
        "cpu" => Ok(Backend::Cpu),
        _ => Err(format!("invalid backend: {value}")),
    }
}

#[cfg(target_os = "macos")]
fn default_backend() -> Backend {
    Backend::Metal
}

#[cfg(not(target_os = "macos"))]
fn default_backend() -> Backend {
    Backend::Cuda
}

fn help_text() -> &'static str {
    "Usage: ds4-bench-rs (--prompt-file FILE | --chat-prompt-file FILE) [options]\n\
     \n\
     Throughput sweep over one fixed prompt.\n\
     \n\
     -m, --model FILE       GGUF model path (default: ds4flash.gguf)\n\
     --mtp FILE             Optional MTP draft GGUF for speculative decode\n\
     --mtp-draft N          Draft tokens per speculative cycle (default: 1)\n\
     --mtp-margin F         Non-exact MTP margin (default: 3.0)\n\
     --cuda|--metal|--cpu   Select backend\n\
     --backend NAME         metal, cuda, or cpu\n\
     -t, --threads N        CPU helper threads\n\
     --quality              Prefer exact kernels where applicable\n\
     --warm-weights         Touch mapped tensor pages before benchmarking\n\
     --power N              GPU duty cycle, 1..100 (default: 100)\n\
     --prompt-file FILE     Raw UTF-8 benchmark prompt\n\
     --chat-prompt-file FILE\n\
                             One no-thinking chat user message\n\
     -sys, --system TEXT    System prompt used only with --chat-prompt-file\n\
     --ctx-start N          First frontier (default: 2048)\n\
     --ctx-max N            Last frontier (default: 32768)\n\
     --ctx-alloc N          Allocated context\n\
     --step-mul F           Multiplicative step (default: 1)\n\
     --step-incr N          Linear step (default: 2048)\n\
     --gen-tokens N         Greedy decode tokens (default: 128)\n\
     --csv FILE             Write CSV instead of stdout\n\
     --output-head-bench N  CUDA output-head verifier at --ctx-start, then exit\n\
     --dump-frontier-logits-dir DIR\n\
                             Write one full-logit JSON file per measured frontier\n\
     -h, --help             Show this help\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once("ds4-bench-rs")
            .chain(args.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn parses_mmq_harness_arguments() {
        let args = parse_args(argv(&[
            "--cuda",
            "--model",
            "model.gguf",
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "1024",
            "--ctx-max",
            "16384",
            "--step-mul",
            "2",
            "--step-incr",
            "2048",
            "--gen-tokens",
            "128",
            "--csv",
            "out.csv",
        ]))
        .unwrap();

        assert_eq!(args.backend, ds4_core::Backend::Cuda);
        assert_eq!(args.model, "model.gguf");
        assert_eq!(args.prompt_file.as_deref(), Some("prompt.txt"));
        assert_eq!(args.ctx_alloc, 16513);
        assert_eq!(args.csv.as_deref(), Some("out.csv"));
        assert!(!args.quality);
        assert!(!args.warm_weights);
        assert_eq!(args.power_percent, 100);
    }

    #[test]
    fn parses_engine_benchmark_options() {
        let args = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--quality",
            "--warm-weights",
            "--power",
            "37",
        ]))
        .unwrap();

        assert!(args.quality);
        assert!(args.warm_weights);
        assert_eq!(args.power_percent, 37);
        assert!(help_text().contains("--quality"));
        assert!(help_text().contains("--warm-weights"));
        assert!(help_text().contains("--power N"));

        for power in ["0", "-1"] {
            assert_eq!(
                parse_args(argv(&["--prompt-file", "prompt.txt", "--power", power,])).unwrap_err(),
                format!("invalid value for --power: {power}")
            );
        }
        assert_eq!(
            parse_args(argv(&["--prompt-file", "prompt.txt", "--power", "101"])).unwrap_err(),
            "--power must be between 1 and 100"
        );
        for power in ["wat", "999999999999999999999999999999999999"] {
            assert_eq!(
                parse_args(argv(&["--prompt-file", "prompt.txt", "--power", power,])).unwrap_err(),
                format!("invalid value for --power: {power}")
            );
        }
        assert_eq!(
            parse_args(argv(&["--prompt-file", "prompt.txt", "--power"])).unwrap_err(),
            "--power requires a value"
        );
    }

    #[test]
    fn selects_chat_prompt_and_system_aliases() {
        let default = parse_args(argv(&["--chat-prompt-file", "chat.txt"])).unwrap();
        assert_eq!(
            default.system.as_deref(),
            Some("You are a helpful assistant.")
        );

        let args = parse_args(argv(&[
            "--chat-prompt-file",
            "chat.txt",
            "--system",
            "first",
            "-sys",
            "second",
        ]))
        .unwrap();

        assert_eq!(args.prompt_file, None);
        assert_eq!(args.chat_prompt_file.as_deref(), Some("chat.txt"));
        assert_eq!(args.system.as_deref(), Some("second"));
        assert_eq!(prompt_source(&args).unwrap(), ("chat.txt", true));
        assert!(help_text().contains("--chat-prompt-file FILE"));
        assert!(help_text().contains("-sys, --system TEXT"));
    }

    #[test]
    fn requires_exactly_one_prompt_file_and_ignores_raw_system() {
        assert_eq!(parse_args(argv(&[])).unwrap_err(), PROMPT_SELECTION_ERR);
        assert_eq!(
            parse_args(argv(&[
                "--prompt-file",
                "raw.txt",
                "--chat-prompt-file",
                "chat.txt",
            ]))
            .unwrap_err(),
            PROMPT_SELECTION_ERR
        );

        let raw = parse_args(argv(&["--prompt-file", "raw.txt", "--system", "ignored"])).unwrap();
        assert_eq!(raw.system.as_deref(), Some("ignored"));
        assert_eq!(prompt_source(&raw).unwrap(), ("raw.txt", false));
    }

    #[test]
    fn prompt_files_preserve_bytes_and_stop_at_c_nul() {
        let path =
            std::env::temp_dir().join(format!("ds4-bench-rs-prompt-bytes-{}", std::process::id()));
        std::fs::write(&path, [0xff, b'a', 0, b'b']).unwrap();
        assert_eq!(read_prompt(path.to_str().unwrap()).unwrap(), [0xff, b'a']);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn parses_distributed_coordinator_for_native_bench_runtime() {
        let args = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--role",
            "coordinator",
            "--layers",
            "0:20",
            "--listen",
            "127.0.0.1",
            "7000",
            "--dist-prefill-chunk",
            "4096",
            "--dist-prefill-window",
            "4",
            "--dist-activation-bits",
            "16",
        ]))
        .unwrap();

        assert_eq!(args.dist.role, ds4_dist::Role::Coordinator);
        assert_eq!(args.dist.layers.start, 0);
        assert_eq!(args.dist.layers.end, 20);
        assert_eq!(args.dist.listen_host.as_deref(), Some("127.0.0.1"));
        assert_eq!(args.dist.listen_port, 7000);
        assert_eq!(args.dist.prefill_chunk, 4096);
        assert_eq!(args.dist.prefill_window, 4);
        assert_eq!(args.dist.activation_bits, 16);
        assert!(uses_distributed_replay(&args));
    }

    #[test]
    fn rejects_distributed_worker_as_a_serving_mode() {
        let err = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--role",
            "worker",
            "--layers",
            "21:output",
            "--coordinator",
            "127.0.0.1",
            "7000",
        ]))
        .unwrap_err();

        assert_eq!(
            err,
            "--role worker is a serving mode; start workers with ./ds4"
        );
    }

    #[test]
    fn walks_linear_and_multiplicative_frontiers() {
        let args = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "1024",
            "--ctx-max",
            "16384",
            "--step-mul",
            "2",
        ]))
        .unwrap();
        let mut got = Vec::new();
        let mut cur = args.ctx_start;
        loop {
            got.push(cur);
            if cur >= args.ctx_max {
                break;
            }
            cur = next_frontier(&args, cur);
        }
        assert_eq!(got, [1024, 2048, 4096, 8192, 16384]);

        let linear = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "2048",
            "--ctx-max",
            "6144",
            "--step-incr",
            "2048",
        ]))
        .unwrap();
        assert_eq!(next_frontier(&linear, 2048), 4096);
        assert_eq!(next_frontier(&linear, 4096), 6144);
    }

    #[test]
    fn validates_context_and_formats_c_csv() {
        let err = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "2048",
            "--ctx-max",
            "2048",
            "--ctx-alloc",
            "2176",
            "--gen-tokens",
            "128",
        ]))
        .unwrap_err();
        assert!(err.contains("ctx-alloc"));

        let row = BenchRow {
            ctx_tokens: 2048,
            prefill_tokens: 2048,
            prefill_tps: 123.456,
            gen_tokens: 128,
            gen_tps: 10.004,
            gen_tps_ss: 9.5,
            first_token_sec: 0.12345,
            kvcache_bytes: 4096,
        };
        assert_eq!(
            CSV_HEADER,
            "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_tps_ss,first_token_sec,kvcache_bytes"
        );
        assert_eq!(
            row.csv_line(),
            "2048,2048,123.46,128,10.00,9.50,0.1235,4096"
        );
    }

    #[test]
    fn allows_exact_context_allocation_without_decode() {
        let args = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "262144",
            "--ctx-max",
            "262144",
            "--ctx-alloc",
            "262144",
            "--gen-tokens",
            "0",
        ]))
        .unwrap();

        assert_eq!(args.ctx_alloc, 262144);
    }

    #[test]
    fn parses_mtp_and_output_head_options() {
        let args = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--mtp",
            "draft.gguf",
            "--mtp-draft",
            "2",
            "--mtp-margin",
            "4.5",
            "--output-head-bench",
            "8",
            "--dump-frontier-logits-dir",
            "/tmp/logits",
        ]))
        .unwrap();
        assert_eq!(args.mtp.as_deref(), Some("draft.gguf"));
        assert_eq!(args.mtp_draft, 2);
        assert_eq!(args.mtp_margin, 4.5);
        assert_eq!(args.output_head_bench_iters, 8);
        assert_eq!(
            args.dump_frontier_logits_dir.as_deref(),
            Some("/tmp/logits")
        );
        assert_eq!(args.ctx_alloc, 2048 + 128 + 1);
    }

    #[test]
    fn rejects_mtp_margin_outside_c_range() {
        let err = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--mtp-margin",
            "1000.1",
        ]))
        .unwrap_err();
        assert!(err.contains("invalid value for --mtp-margin"));
    }

    #[test]
    fn mtp_spec_follows_c_gates() {
        assert!(!use_mtp_spec(ModelFamily::DeepSeek4, None, 2));
        assert!(use_mtp_spec(ModelFamily::Qwen4Exp, None, 2));
        assert!(!use_mtp_spec(ModelFamily::DeepSeek4, Some("draft.gguf"), 1));
        assert!(use_mtp_spec(ModelFamily::DeepSeek4, Some("draft.gguf"), 2));
    }

    #[test]
    fn frontier_logits_path_matches_c() {
        assert_eq!(
            frontier_logits_path("/tmp/logits", 2048),
            "/tmp/logits/frontier_002048.logits.json"
        );
        assert_eq!(json_escape(r#"a"b"#), r#""a\"b""#);
        assert_eq!(json_f32(f32::NAN), "null");
        assert!(json_f32(1.5).starts_with("1.5"));
        let small = f32::from_bits(3_152_357_376);
        assert_eq!(
            json_f32(small).parse::<f32>().unwrap().to_bits(),
            small.to_bits()
        );
    }

    /// C `ds4-bench --help` flags this shadow already parses (8.3 claimed modes).
    /// `--mtp ` keeps `--mtp-draft` from counting as `--mtp`.
    const CLAIMED_C_BENCH_FLAGS: &[&str] = &[
        "--prompt-file",
        "--chat-prompt-file",
        "-sys, --system",
        "-m, --model",
        "--mtp ",
        "--mtp-draft",
        "--mtp-margin",
        "--metal",
        "--cuda",
        "--cpu",
        "--backend",
        "-t, --threads",
        "--quality",
        "--warm-weights",
        "--power",
        "--role",
        "--layers",
        "--listen",
        "--coordinator",
        "--dist-prefill-chunk",
        "--dist-prefill-window",
        "--dist-activation-bits",
        "--dist-replay-check",
        "--debug",
        "--ctx-start",
        "--ctx-max",
        "--ctx-alloc",
        "--step-mul",
        "--step-incr",
        "--gen-tokens",
        "--csv",
        "--output-head-bench",
        "--dump-frontier-logits-dir",
        "-h, --help",
    ];

    /// C `ds4_bench.c` fprintf header (not a named C constant; byte-identical).
    const C_CSV_HEADER: &str = "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_tps_ss,first_token_sec,kvcache_bytes";

    #[test]
    fn help_contains_each_claimed_c_flag() {
        let help = format!("{}\nDistributed:\n{}", help_text(), ds4_dist::USAGE);
        for flag in CLAIMED_C_BENCH_FLAGS {
            assert!(
                help.contains(flag),
                "ds4-bench-rs help missing claimed C flag {flag}"
            );
        }
    }

    #[test]
    fn csv_header_equals_c_constant() {
        assert_eq!(CSV_HEADER, C_CSV_HEADER);
    }
}
