//! Shadow CLI host. Calls the same C inference core through `ds4-core`.

pub mod agent;
pub mod bench;
pub mod repl;
pub mod session_exec;
pub mod session_snapshot;
mod token_printer;
pub mod worker_hello;
pub mod worker_run;

use token_printer::TokenPrinter;

use ds4_core::{Backend, ModelOpenOption};

pub(crate) fn distributed_config(opt: &ds4_dist::Options) -> Option<ds4_core::DistributedConfig> {
    let role = match opt.role {
        ds4_dist::Role::None => return None,
        ds4_dist::Role::Coordinator => ds4_core::DistributedRole::Coordinator,
        ds4_dist::Role::Worker => ds4_core::DistributedRole::Worker,
    };
    Some(ds4_core::DistributedConfig {
        role,
        layer_start: opt.layers.start,
        layer_end: opt.layers.end,
        has_output: opt.layers.has_output,
        listen_host: opt.listen_host.clone(),
        listen_port: opt.listen_port,
        coordinator_host: opt.coordinator_host.clone(),
        coordinator_port: opt.coordinator_port,
        prefill_chunk: opt.prefill_chunk,
        prefill_window: opt.prefill_window,
        activation_bits: opt.activation_bits,
        replay_check: opt.replay_check,
        debug: opt.debug,
    })
}

#[derive(Debug)]
pub struct ShadowArgs {
    pub model: Option<String>,
    pub mtp: Option<String>,
    pub mtp_draft: i32,
    pub mtp_margin: f32,
    pub dspark: Option<String>,
    pub backend: Backend,
    pub ctx: i32,
    pub ctx_set: bool,
    pub threads: i32,
    pub tokens: Vec<i32>,
    pub predict: i32,
    pub n_predict: i32,
    pub system: String,
    pub nothink: bool,
    pub prompt: Option<String>,
    pub prompt_file: Option<String>,
    pub dump_logprobs: Option<String>,
    pub dump_logits: Option<String>,
    pub dump_tokens: bool,
    pub logprobs_top_k: i32,
    pub temp: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub lifecycle_only: bool,
    pub identify: bool,
    pub inventory: bool,
    pub tokenize: bool,
    pub tok_family: Option<String>,
    pub tok_cmd: Option<String>,
    pub tok_arg: Option<String>,
    pub session_plan: bool,
    pub session_cmd: Option<String>,
    pub session_args: Vec<String>,
    pub session_payload: bool,
    pub payload_cmd: Option<String>,
    pub payload_args: Vec<String>,
    pub bind_names: bool,
    pub bind_names_variant: Option<String>,
    pub bind_plan: bool,
    pub validate: bool,
    pub layout: bool,
    pub layout_variant: Option<String>,
    pub dist: ds4_dist::Options,
    pub help: bool,
}

impl Default for ShadowArgs {
    fn default() -> Self {
        Self {
            model: Some("ds4flash.gguf".into()),
            mtp: None,
            mtp_draft: 1,
            mtp_margin: 3.0,
            dspark: None,
            backend: default_backend(),
            ctx: 32768,
            ctx_set: false,
            threads: 0,
            tokens: Vec::new(),
            predict: 0,
            n_predict: 50000,
            system: "You are a helpful assistant".into(),
            nothink: false,
            prompt: None,
            prompt_file: None,
            dump_logprobs: None,
            dump_logits: None,
            dump_tokens: false,
            logprobs_top_k: 20,
            temp: 1.0,
            top_p: 1.0,
            min_p: 0.05,
            seed: 0,
            lifecycle_only: false,
            identify: false,
            inventory: false,
            tokenize: false,
            tok_family: None,
            tok_cmd: None,
            tok_arg: None,
            session_plan: false,
            session_cmd: None,
            session_args: Vec::new(),
            session_payload: false,
            payload_cmd: None,
            payload_args: Vec::new(),
            bind_names: false,
            bind_names_variant: None,
            bind_plan: false,
            validate: false,
            layout: false,
            layout_variant: None,
            dist: ds4_dist::Options::default(),
            help: false,
        }
    }
}

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<ShadowArgs, String> {
    let mut parsed = ShadowArgs::default();
    let mut iter = args.into_iter().peekable();
    let _argv0 = iter.next();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => parsed.help = true,
            "-m" | "--model" => {
                parsed.model = Some(require_value(&arg, iter.next())?);
            }
            "--mtp" => {
                parsed.mtp = Some(require_value(&arg, iter.next())?);
            }
            "--mtp-draft" => {
                parsed.mtp_draft = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--mtp-margin" => {
                parsed.mtp_margin =
                    parse_f32_range(&arg, &require_value(&arg, iter.next())?, 0.0, 1000.0)?;
            }
            "--dspark" => {
                parsed.dspark = Some(require_value(&arg, iter.next())?);
            }
            "--backend" => {
                parsed.backend = parse_backend(&require_value(&arg, iter.next())?)?;
            }
            /* C CLI backend spellings, so the proof harness command line
             * (`--cuda -m ...`) drives this shadow unchanged. */
            "--cuda" => parsed.backend = Backend::Cuda,
            "--cpu" => parsed.backend = Backend::Cpu,
            "--metal" => parsed.backend = Backend::Metal,
            "-c" | "--ctx" => {
                parsed.ctx = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
                parsed.ctx_set = true;
            }
            "-n" | "--tokens" | "--n-predict" => {
                let value = require_value(&arg, iter.next())?;
                if arg == "--tokens" && value.contains(',') {
                    return Err(
                        "--tokens is the output budget; use --token-ids for raw token IDs".into(),
                    );
                }
                parsed.n_predict = parse_positive_i32(&arg, &value)?;
            }
            "--temp" => {
                let v = require_value(&arg, iter.next())?;
                parsed.temp = parse_f32_range(&arg, &v, 0.0, 100.0)?;
            }
            "--top-p" => {
                let v = require_value(&arg, iter.next())?;
                parsed.top_p = parse_f32_range(&arg, &v, 0.0, 1.0)?;
            }
            "--min-p" => {
                let v = require_value(&arg, iter.next())?;
                parsed.min_p = parse_f32_range(&arg, &v, 0.0, 1.0)?;
            }
            "--seed" => {
                let v = require_value(&arg, iter.next())?;
                parsed.seed = parse_positive_u64(&arg, &v)?;
            }
            "--think" => parsed.nothink = false,
            "--nothink" => parsed.nothink = true,
            "-sys" | "--system" => {
                parsed.system = require_value(&arg, iter.next())?;
            }
            "-p" | "--prompt" => {
                if parsed.prompt.is_some() || parsed.prompt_file.is_some() {
                    return Err("specify only one prompt source".into());
                }
                parsed.prompt = Some(require_value(&arg, iter.next())?);
            }
            "--prompt-file" => {
                if parsed.prompt.is_some() || parsed.prompt_file.is_some() {
                    return Err("specify only one prompt source".into());
                }
                parsed.prompt_file = Some(require_value(&arg, iter.next())?);
            }
            "--dump-tokens" => parsed.dump_tokens = true,
            "--dump-logits" => {
                parsed.dump_logits = Some(
                    iter.next()
                        .ok_or_else(|| "ds4: missing value for --dump-logits".to_string())?,
                );
            }
            "--dump-logprobs" => {
                parsed.dump_logprobs = Some(require_value(&arg, iter.next())?);
            }
            "--logprobs-top-k" => {
                parsed.logprobs_top_k = parse_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "-t" | "--threads" => {
                parsed.threads = parse_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--token-ids" => {
                parsed.tokens = parse_tokens(&require_value(&arg, iter.next())?)?;
            }
            "--predict" => {
                parsed.predict = parse_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--lifecycle" => parsed.lifecycle_only = true,
            "--identify" => parsed.identify = true,
            "--inventory" => parsed.inventory = true,
            "--tokenize" => {
                parsed.tokenize = true;
                parsed.tok_family = Some(require_value(&arg, iter.next())?);
                parsed.tok_cmd = Some(require_value(&arg, iter.next())?);
                if let Some(peek) = iter.peek() {
                    if !peek.starts_with('-') {
                        parsed.tok_arg = iter.next();
                    }
                }
            }
            "--session-plan" => {
                parsed.session_plan = true;
                parsed.session_cmd = Some(require_value(&arg, iter.next())?);
                while let Some(peek) = iter.peek() {
                    if peek.starts_with('-') {
                        break;
                    }
                    parsed.session_args.push(iter.next().unwrap());
                }
            }
            "--session-payload" => {
                parsed.session_payload = true;
                parsed.payload_cmd = Some(require_value(&arg, iter.next())?);
                while let Some(peek) = iter.peek() {
                    if peek.starts_with('-') {
                        break;
                    }
                    parsed.payload_args.push(iter.next().unwrap());
                }
            }
            "--bind-names" => {
                parsed.bind_names = true;
                parsed.bind_names_variant = Some(require_value(&arg, iter.next())?);
            }
            "--bind-plan" => parsed.bind_plan = true,
            "--validate" => parsed.validate = true,
            "--layout" => {
                parsed.layout = true;
                parsed.layout_variant = Some(require_value(&arg, iter.next())?);
            }
            other => match ds4_dist::parse_cli_arg(other, &mut iter, &mut parsed.dist)? {
                ds4_dist::CliResult::Matched => {}
                ds4_dist::CliResult::NotMatched => {
                    return Err(format!("unknown argument: {other}"));
                }
                ds4_dist::CliResult::Error => unreachable!(),
            },
        }
    }
    ds4_dist::prepare_engine_options(&parsed.dist)?;
    Ok(parsed)
}

const DEEPSEEK_BOS: &str = "<｜begin▁of▁sentence｜>";

fn is_rendered_chat_prompt(prompt: &str) -> bool {
    prompt.starts_with(DEEPSEEK_BOS)
}

fn prompt_text(args: &ShadowArgs) -> Result<String, String> {
    match (&args.prompt, &args.prompt_file) {
        (Some(prompt), None) => Ok(prompt.clone()),
        (None, Some(path)) => {
            std::fs::read_to_string(path).map_err(|e| format!("prompt-file: {e}"))
        }
        (Some(_), Some(_)) => Err("specify only one prompt source".into()),
        (None, None) => Err("one-shot generation requires -p or --prompt-file".into()),
    }
}

fn validate_one_shot_args(args: &ShadowArgs) -> Result<(), String> {
    if args.prompt.is_none() && args.prompt_file.is_none() {
        return Err("one-shot generation requires -p or --prompt-file".into());
    }
    Ok(())
}

fn use_mtp_spec(temp: f32, mtp: Option<&str>, draft: i32) -> bool {
    temp <= 0.0 && mtp.is_some() && draft > 1 && std::env::var_os("DS4_MTP_SPEC_DISABLE").is_none()
}

fn mtp_open_options(args: &ShadowArgs) -> Vec<ModelOpenOption> {
    let mut options = Vec::new();
    if args.mtp.is_some() {
        options.push(ModelOpenOption::MtpDraftTokens(args.mtp_draft));
        options.push(ModelOpenOption::MtpMargin(args.mtp_margin));
    }
    options
}

fn generation_limit(ctx: i32, pos: i32, requested: i32) -> i32 {
    let room = ctx.saturating_sub(pos);
    requested.min(room).max(0)
}

fn sampled_generation_limit(ctx: i32, pos: i32, requested: i32) -> i32 {
    let room = ctx.saturating_sub(pos).saturating_sub(1);
    requested.min(room).max(0)
}

fn session_ready<'m>(
    model: &'m ds4_core::Model,
    args: &ShadowArgs,
    ctx: i32,
) -> Result<ds4_core::Session<'m>, String> {
    let session = model.session(ctx).map_err(|e| e.to_string())?;
    if args.dist.role != ds4_dist::Role::Coordinator {
        return Ok(session);
    }
    let mut ticks = 0u32;
    loop {
        if session
            .distributed_route_ready()
            .map_err(|e| e.to_string())?
        {
            if ticks != 0 {
                eprintln!("ds4-rs: distributed route ready");
            }
            return Ok(session);
        }
        if ticks % 4 == 0 {
            eprintln!("ds4-rs: waiting for distributed route");
        }
        ticks = ticks.wrapping_add(1);
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

fn apply_effort_prefix(
    vocab: &ds4_core::Vocab,
    chat: &mut repl::ReplChat,
    session: Option<&mut ds4_core::Session<'_>>,
) {
    let want = chat.wants_effort_prefix();
    if want && chat.prefix_tokens == 0 {
        let mut prefix = ds4_core::TokenBuffer::new();
        vocab.chat_append_effort_prefix(&mut prefix, chat.effective_think());
        chat.transcript.insert(1, prefix.as_slice());
        chat.prefix_tokens = prefix.len();
        if let Some(session) = session {
            session.invalidate();
        }
    } else if !want && chat.prefix_tokens > 0 {
        chat.transcript.remove(1, chat.prefix_tokens);
        chat.prefix_tokens = 0;
        if let Some(session) = session {
            session.invalidate();
        }
    }
}

fn run_chat_turn(
    model: &ds4_core::Model,
    args: &ShadowArgs,
    chat: &mut repl::ReplChat,
    session: &mut ds4_core::Session<'_>,
    user_text: &str,
) -> Result<(), String> {
    use std::io::Write;

    let vocab = model.vocab();
    apply_effort_prefix(vocab, chat, Some(session));
    let rollback = chat.transcript.len();
    vocab
        .chat_append_message(&mut chat.transcript, "user", user_text.as_bytes())
        .map_err(|e| e.to_string())?;
    let think = chat.effective_think();
    vocab
        .chat_append_assistant_prefix(&mut chat.transcript, think)
        .map_err(|e| e.to_string())?;

    if let Err(e) = session.sync(&chat.transcript) {
        chat.transcript.truncate(rollback);
        eprintln!("ds4: prompt processing failed: {e}");
        return Ok(());
    }

    let max_tokens = sampled_generation_limit(session.ctx(), session.pos(), args.n_predict);
    let mut rng = args.seed;
    if rng == 0 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        rng = (now ^ (u64::from(std::process::id()) << 32)) | 1;
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut printer = TokenPrinter::new(chat.thinking_enabled());
    let use_mtp = use_mtp_spec(args.temp, args.mtp.as_deref(), args.mtp_draft);
    let eos = model.token_eos();
    let mut generated = 0i32;
    repl::interrupt_clear();

    while generated < max_tokens && !repl::interrupt_requested() {
        let token = session.sample(args.temp, 0, args.top_p, args.min_p, &mut rng);
        if token < 0 {
            eprintln!("ds4: decode failed: failed to sample the next token");
            printer.finish(&mut out).map_err(|e| e.to_string())?;
            return Ok(());
        }
        if model.token_is_stop(token) {
            break;
        }
        let accepted = if use_mtp {
            match session.eval_speculative_argmax(token, max_tokens - generated, eos) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("ds4: decode failed: {e}");
                    printer.finish(&mut out).map_err(|e| e.to_string())?;
                    return Ok(());
                }
            }
        } else if let Err(e) = session.eval(token) {
            eprintln!("ds4: decode failed: {e}");
            printer.finish(&mut out).map_err(|e| e.to_string())?;
            return Ok(());
        } else {
            vec![token]
        };
        let mut stop = false;
        for t in accepted {
            if model.token_is_stop(t) {
                stop = true;
                break;
            }
            let piece = model.token_text(t).map_err(|e| e.to_string())?;
            printer
                .write_text(&mut out, &piece)
                .map_err(|e| e.to_string())?;
            out.flush().map_err(|e| e.to_string())?;
            chat.transcript.push(t);
            generated += 1;
            if generated >= max_tokens {
                break;
            }
        }
        if stop {
            break;
        }
    }
    printer.finish(&mut out).map_err(|e| e.to_string())?;
    let interrupted = repl::interrupt_requested();
    match repl::interrupt_end(interrupted, generated) {
        repl::InterruptEnd::Rollback => {
            chat.transcript.truncate(rollback);
            session.invalidate();
        }
        repl::InterruptEnd::KeepEos => chat.transcript.push(eos),
    }
    if interrupted {
        repl::interrupt_clear();
    }
    Ok(())
}

fn run_one_shot(model: &ds4_core::Model, args: &ShadowArgs, text: &str) -> Result<i32, String> {
    use std::io::Write;

    const THINK_NONE: i32 = 0;
    const THINK_LOW: i32 = 1;

    let prompt = if is_rendered_chat_prompt(&text) {
        model.tokenize_rendered_chat(&text)
    } else {
        model.encode_chat_prompt(
            Some(args.system.as_str()),
            &text,
            if args.nothink { THINK_NONE } else { THINK_LOW },
        )
    }
    .map_err(|e| e.to_string())?;

    let mut session = session_ready(model, args, args.ctx)?;
    session.sync(&prompt).map_err(|e| e.to_string())?;

    let sampled = args.temp > 0.0;
    let max_tokens = if sampled {
        sampled_generation_limit(session.ctx(), session.pos(), args.n_predict)
    } else {
        generation_limit(session.ctx(), session.pos(), args.n_predict)
    };
    let mut rng = args.seed;
    if sampled && rng == 0 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        rng = (now ^ (u64::from(std::process::id()) << 32)) | 1;
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut printer = TokenPrinter::new(!args.nothink);
    let mut decode_error = None;
    let use_mtp = use_mtp_spec(args.temp, args.mtp.as_deref(), args.mtp_draft);
    let eos = model.token_eos();
    let mut generated = 0;

    while generated < max_tokens {
        let token = if sampled {
            session.sample(args.temp, 0, args.top_p, args.min_p, &mut rng)
        } else {
            session.argmax()
        };
        if token < 0 {
            decode_error = Some(if sampled {
                "failed to sample the next token".into()
            } else {
                "failed to select the next token".into()
            });
            break;
        }
        if model.token_is_stop(token) {
            break;
        }
        if use_mtp {
            let accepted = match session.eval_speculative_argmax(token, max_tokens - generated, eos)
            {
                Ok(v) => v,
                Err(e) => {
                    decode_error = Some(e.to_string());
                    break;
                }
            };
            let mut stop = false;
            for t in accepted {
                if model.token_is_stop(t) {
                    stop = true;
                    break;
                }
                let piece = model.token_text(t).map_err(|e| e.to_string())?;
                printer
                    .write_text(&mut out, &piece)
                    .map_err(|e| e.to_string())?;
                out.flush().map_err(|e| e.to_string())?;
                generated += 1;
                if generated >= max_tokens {
                    break;
                }
            }
            if stop {
                break;
            }
            continue;
        }
        if sampled {
            if let Err(e) = session.eval(token) {
                decode_error = Some(e.to_string());
                break;
            }
        }

        let piece = model.token_text(token).map_err(|e| e.to_string())?;
        printer
            .write_text(&mut out, &piece)
            .map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
        generated += 1;

        if !sampled {
            if generated >= max_tokens {
                break;
            }
            if let Err(e) = session.eval(token) {
                decode_error = Some(e.to_string());
                break;
            }
        }
    }
    printer.finish(&mut out).map_err(|e| e.to_string())?;
    match decode_error {
        Some(error) => Err(error),
        None => Ok(0),
    }
}

/// Proof-harness dump: mirror of the C CLI `run_logprob_dump` loop
/// (top_logprobs -> argmax -> write step -> stop check -> eval).  Prompt
/// rendering follows the C CLI; token text and the stop set use the host vocab.
fn run_logprob_dump(
    model: &ds4_core::Model,
    args: &ShadowArgs,
    prompt_text: &str,
) -> Result<i32, String> {
    use std::io::Write;

    const THINK_NONE: i32 = 0;
    const THINK_LOW: i32 = 1;
    const TOP_K_CAP: usize = 128;

    let think = if args.nothink { THINK_NONE } else { THINK_LOW };
    let prompt = if is_rendered_chat_prompt(&prompt_text) {
        model.tokenize_rendered_chat(&prompt_text)
    } else {
        model.encode_chat_prompt(Some(args.system.as_str()), &prompt_text, think)
    }
    .map_err(|e| e.to_string())?;
    let n_prompt = prompt.len() as i32;

    /* C defaults ctx to 262144 on CUDA; the ids do not depend on ctx size,
     * so grow to fit unless -c was explicit. */
    let ctx = if args.ctx_set {
        args.ctx
    } else {
        args.ctx.max(n_prompt + args.n_predict + 8)
    };
    let mut session = session_ready(model, args, ctx)?;
    session.sync(&prompt).map_err(|e| e.to_string())?;

    let path = args.dump_logprobs.as_deref().unwrap();
    let mut fp = std::io::BufWriter::new(
        std::fs::File::create(path).map_err(|e| format!("dump-logprobs {path}: {e}"))?,
    );
    let k = (args.logprobs_top_k.max(1) as usize).min(TOP_K_CAP);
    let mut w = |s: String| fp.write_all(s.as_bytes()).map_err(|e| e.to_string());

    w(format!(
        "{{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":{n_prompt},\n  \"ctx\":{ctx},\n  \"top_k\":{k},\n  \"steps\":[\n"
    ))?;

    let mut max_tokens = args.n_predict;
    let room = session.ctx() - session.pos();
    if room <= 1 {
        max_tokens = 0;
    } else if max_tokens > room - 1 {
        max_tokens = room - 1;
    }

    for generated in 0..max_tokens {
        let scores = session.top_logprobs(k);
        let token = session.argmax();
        if generated > 0 {
            w(",\n".into())?;
        }
        w(format!("    {{\"step\":{generated},\"selected\":"))?;
        w(json_token(model, token))?;
        w(",\"top_logprobs\":[".into())?;
        for (i, s) in scores.iter().take_while(|s| s.id >= 0).enumerate() {
            if i > 0 {
                w(",".into())?;
            }
            /* Rust shortest-roundtrip floats, not C's %.9g: the md5
             * contract reads only the selected ids. */
            w(format!(
                "{{\"token\":{},\"logit\":{},\"logprob\":{}}}",
                json_token(model, s.id),
                s.logit,
                s.logprob
            ))?;
        }
        w("]}".into())?;

        if model.token_is_stop(token) {
            break;
        }
        session.eval(token).map_err(|e| e.to_string())?;
    }

    w("\n  ]\n}\n".into())?;
    fp.flush().map_err(|e| e.to_string())?;
    Ok(0)
}

fn json_token(model: &ds4_core::Model, token: i32) -> String {
    let bytes = model.token_text(token).unwrap_or_default();
    let mut s = format!("{{\"id\":{token},\"text\":\"");
    for &b in &bytes {
        match b {
            b'"' => s.push_str("\\\""),
            b'\\' => s.push_str("\\\\"),
            0x20..=0x7e => s.push(b as char),
            _ => s.push_str(&format!("\\u{:04x}", b)),
        }
    }
    s.push_str("\",\"bytes\":[");
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&b.to_string());
    }
    s.push_str("]}");
    s
}

struct LogitsDumpJson<'a> {
    model: &'a str,
    backend: &'a str,
    quant_bits: i32,
    prompt_tokens: i32,
    ctx: i32,
    argmax_token_json: &'a str,
    argmax_logit: f32,
    logits: &'a [f32],
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda => "cuda",
        Backend::Metal => "metal",
        Backend::Cpu => "cpu",
    }
}

fn json_escape_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_logit(v: f32) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".into()
    }
}

fn format_logits_json(dump: &LogitsDumpJson<'_>) -> String {
    let mut body = String::from("{\n  \"source\":\"ds4\",\n  \"model\":");
    body.push_str(&json_escape_str(dump.model));
    body.push_str(&format!(
        ",\n  \"backend\":\"{}\",\n  \"quant_bits\":{},\n  \"prompt_tokens\":{},\n  \"ctx\":{},\n  \"vocab\":{},\n  \"argmax_token\":{},\n  \"argmax_logit\":{},\n  \"logits\":[",
        dump.backend,
        dump.quant_bits,
        dump.prompt_tokens,
        dump.ctx,
        dump.logits.len(),
        dump.argmax_token_json,
        json_logit(dump.argmax_logit),
    ));
    for (i, logit) in dump.logits.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        if i % 8 == 0 {
            body.push_str("\n    ");
        }
        body.push_str(&json_logit(*logit));
    }
    body.push_str("\n  ]\n}\n");
    body
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvalPlan {
    Skip,
    Run,
}

fn eval_plan(args: &ShadowArgs) -> EvalPlan {
    if args.dump_tokens {
        EvalPlan::Skip
    } else {
        EvalPlan::Run
    }
}

fn format_dump_tokens(ids: &[i32], lines: &[(i32, &[u8])]) -> Vec<u8> {
    let mut out = Vec::from(b"[".as_slice());
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b", ");
        }
        out.extend_from_slice(id.to_string().as_bytes());
    }
    out.extend_from_slice(b"]\n");
    for (id, text) in lines {
        out.extend_from_slice(format!("{id:6}  ").as_bytes());
        out.extend_from_slice(text);
        out.push(b'\n');
    }
    out
}

fn write_dump_tokens(
    out: &mut impl std::io::Write,
    ids: &[i32],
    vocab: &ds4_core::Vocab,
) -> Result<(), String> {
    let table = vocab.tokens();
    let lines: Vec<(i32, &[u8])> = ids
        .iter()
        .copied()
        .filter_map(|id| {
            if id < 0 {
                return None;
            }
            table.get(id as usize).map(|text| (id, text.as_slice()))
        })
        .collect();
    out.write_all(&format_dump_tokens(ids, &lines))
        .map_err(|e| e.to_string())
}

fn run_dump_tokens(model_path: &str, text: &str) -> Result<i32, String> {
    let id =
        ds4_core::identify_gguf(std::path::Path::new(model_path)).map_err(|e| e.to_string())?;
    let vocab = ds4_core::Vocab::load_path(std::path::Path::new(model_path), id.shape.family)
        .map_err(|e| e.to_string())?;
    let ids = vocab.encode_rendered_chat(text);
    let mut stdout = std::io::stdout().lock();
    write_dump_tokens(&mut stdout, &ids, &vocab)?;
    Ok(0)
}

fn run_dump_tokens_cli(args: &ShadowArgs) -> Result<i32, String> {
    if args.prompt.is_none() && args.prompt_file.is_none() {
        return Err("ds4: --dump-tokens requires -p or --prompt-file".into());
    }
    let text = prompt_text(args)?;
    let model_path = args
        .model
        .as_deref()
        .ok_or_else(|| "missing -m/--model (or pass --help)".to_string())?;
    run_dump_tokens(model_path, &text)
}

fn write_logits_file(path: &str, body: &str) -> Result<(), String> {
    use std::io::Write;
    let mut fp = std::fs::File::create(path)
        .map_err(|_| format!("ds4: failed to open --dump-logits file: {path}"))?;
    fp.write_all(body.as_bytes()).map_err(|e| e.to_string())?;
    fp.flush()
        .map_err(|_| format!("ds4: failed to close --dump-logits file: {path}"))?;
    Ok(())
}

fn run_logits_dump(
    model: &ds4_core::Model,
    args: &ShadowArgs,
    prompt_text: &str,
) -> Result<i32, String> {
    const THINK_NONE: i32 = 0;
    const THINK_LOW: i32 = 1;

    let think = if args.nothink { THINK_NONE } else { THINK_LOW };
    let prompt = if is_rendered_chat_prompt(prompt_text) {
        model.tokenize_rendered_chat(prompt_text)
    } else {
        model.encode_chat_prompt(Some(args.system.as_str()), prompt_text, think)
    }
    .map_err(|e| e.to_string())?;
    let n_prompt = prompt.len() as i32;
    let ctx = args.ctx;
    let mut session = session_ready(model, args, ctx)?;
    session.sync(&prompt).map_err(|e| e.to_string())?;

    let vocab = model.vocab().n_vocab() as usize;
    let logits = session.copy_logits(vocab).map_err(|e| e.to_string())?;
    let argmax = session.argmax();
    let argmax_logit = if argmax >= 0 {
        logits.get(argmax as usize).copied().unwrap_or(0.0)
    } else {
        0.0
    };
    let argmax_token_json = json_token(model, argmax);
    let path = args
        .dump_logits
        .as_deref()
        .ok_or_else(|| "ds4: missing value for --dump-logits".to_string())?;
    let body = format_logits_json(&LogitsDumpJson {
        model: args.model.as_deref().unwrap_or("ds4flash.gguf"),
        backend: backend_name(args.backend),
        quant_bits: model.routed_quant_bits(),
        prompt_tokens: n_prompt,
        ctx,
        argmax_token_json: &argmax_token_json,
        argmax_logit,
        logits: &logits,
    });
    write_logits_file(path, &body)?;
    Ok(0)
}

pub fn help_text(name: &str) -> String {
    format!(
        "\
{name} — Rust shadow of the C host (same C inference core)

Usage:
  {name} --help
  {name} -m MODEL --identify
  {name} -m MODEL --inventory
  {name} --bind-names VARIANT
  {name} --layout VARIANT
  {name} -m MODEL --bind-plan
  {name} -m MODEL --validate
  {name} -m MODEL --tokenize FAMILY CMD [ARG]
  {name} --session-plan CMD [ARGS...]
  {name} --session-payload CMD [ARGS...]
  {name} -m MODEL [--backend cuda|cpu|metal] [-c CTX] [--lifecycle]
  {name} -m MODEL (-p PROMPT | --prompt-file FILE) [-n N]
      [--think|--nothink] [--temp F --top-p F --min-p F --seed N]
  {name} -m MODEL --token-ids 1,2,3 [--predict N]
  {name} -m MODEL [--mtp GGUF] [--dspark GGUF] ...
  {name} --cuda -m MODEL --temp 0 -n N [--nothink] --dump-logprobs F \\
      --logprobs-top-k K (-p PROMPT | --prompt-file FILE)
  {name} --cuda -m MODEL [--nothink] --dump-logits F \\
      (-p PROMPT | --prompt-file FILE)
  {name} -m MODEL --dump-tokens (-p PROMPT | --prompt-file FILE)

C-compatible flags (same names as `ds4 --help`):
  -m, --model FILE        GGUF model path. Default: ds4flash.gguf
  --mtp FILE              Optional MTP support GGUF
  --mtp-draft N           Maximum MTP draft tokens. Default: 1
  --mtp-margin F          MTP verifier margin. Default: 3
  -c, --ctx N             Context size. Default: 32768
  --metal, --cuda, --cpu  Select backend explicitly
  --backend NAME          metal, cuda, or cpu
  -t, --threads N         CPU helper threads
  -p, --prompt TEXT       Prompt to generate from
  --prompt-file FILE      Read the prompt text from FILE
  -sys, --system TEXT     System prompt
  -n, --tokens N          Maximum tokens to generate. Default: 50000
  --temp F --top-p F --min-p F --seed N
  --think --nothink
  --dump-tokens --dump-logits FILE --dump-logprobs FILE --logprobs-top-k N
  -h, --help              Show this help

--mtp/--dspark attach the DeepSeek-only sibling support models; the host
resolves each sibling bind catalog + expected layouts, native skips that
sibling's name walk and layout check.
--dump-logprobs mirrors the C CLI proof loop (chat-template encode via
the engine, argmax decode, host stop set); ctx grows to fit prompt+n
unless -c is explicit.
--dump-logits writes full next-token logits as JSON after prompt prefill.
--dump-tokens tokenizes -p/--prompt-file exactly as written, then exits
without inference.
One-shot sampling uses the native C sampler; an explicit --seed is
reproducible across the C and Rust hosts.
Thinking tags follow the C contract: TTY greys the body; pipes stay uncolored.
--tokens is the C-compatible output budget; --token-ids is the shadow-only
raw token-ID diagnostic.

Distributed worker/coordinator options use the frozen C DS4D runtime behind
the Rust-owned CLI/model lifecycle in this first production integration slice:
{dist_usage}

--identify mmaps GGUF metadata only (no CUDA, no ds4_bridge_model_open).
--inventory mmaps the tensor directory + split remap (no CUDA, no engine open).
--bind-names dumps the host weights_bind catalog (no GGUF, no engine open).
--layout dumps the host weights_validate_layout table (no GGUF, no engine open).
--bind-plan resolves a catalog against the host inventory (no CUDA).
With --bind-names VARIANT, --bind-plan uses that catalog (including
mtp-flash / dspark-pro) instead of identifying the GGUF family.
--validate runs host-owned config_validate (no CUDA, no engine open).
VARIANT is flash|pro|solar-open2|motif3|exaone-moe|dots3-note|qwen4exp|glm5-next
or DeepSeek sibling mtp-flash|mtp-pro|dspark-flash|dspark-pro.
--tokenize loads the host-owned GPT-2/BPE vocab (no engine open).
--session-plan dumps the host session ledger (no engine open).
--session-payload dumps the host DSV4 prefix codec (no engine open).
FAMILY is deepseek4|motif3|solar-open2|exaone-moe|dots3-note|qwen4exp|glm5-next.
CMD is specials | encode HEX | render HEX | decode ID | stop ID.
",
        dist_usage = ds4_dist::USAGE,
    )
}

pub fn run(name: &str, args: ShadowArgs) -> Result<i32, String> {
    if args.help {
        print!("{}", help_text(name));
        return Ok(0);
    }
    if eval_plan(&args) == EvalPlan::Skip {
        return run_dump_tokens_cli(&args);
    }
    if args.session_plan {
        let cmd = args
            .session_cmd
            .as_deref()
            .ok_or_else(|| "--session-plan requires CMD".to_string())?;
        let argv: Vec<&str> = args.session_args.iter().map(String::as_str).collect();
        print!("{}", ds4_core::session_dump_cmd(cmd, &argv));
        return Ok(0);
    }
    if args.session_payload {
        let cmd = args
            .payload_cmd
            .as_deref()
            .ok_or_else(|| "--session-payload requires CMD".to_string())?;
        let argv: Vec<&str> = args.payload_args.iter().map(String::as_str).collect();
        print!("{}", ds4_core::payload_dump_cmd(cmd, &argv));
        return Ok(0);
    }
    if args.layout {
        let name = args
            .layout_variant
            .as_deref()
            .ok_or_else(|| "--layout requires VARIANT".to_string())?;
        let dump = ds4_core::dump_expected_layouts_variant(name)
            .ok_or_else(|| format!("unknown layout variant: {name}"))?;
        print!("{dump}");
        return Ok(0);
    }
    if args.bind_names && !args.bind_plan {
        let name = args
            .bind_names_variant
            .as_deref()
            .ok_or_else(|| "--bind-names requires VARIANT".to_string())?;
        let dump = ds4_core::dump_bind_names_variant(name)
            .ok_or_else(|| format!("unknown bind-names variant: {name}"))?;
        print!("{dump}");
        return Ok(0);
    }

    let model_path = args
        .model
        .as_deref()
        .ok_or_else(|| "missing -m/--model (or pass --help)".to_string())?;

    if args.identify {
        let id =
            ds4_core::identify_gguf(std::path::Path::new(model_path)).map_err(|e| e.to_string())?;
        println!("{}", id.report_line(model_path));
        return Ok(0);
    }

    if args.inventory {
        let inv = ds4_core::TensorInventory::open(std::path::Path::new(model_path))
            .map_err(|e| e.to_string())?;
        print!("{}", inv.dump());
        return Ok(0);
    }
    if args.bind_plan {
        let inv = ds4_core::TensorInventory::open(std::path::Path::new(model_path))
            .map_err(|e| e.to_string())?;
        let (support, shape) = if let Some(name) = args.bind_names_variant.as_deref() {
            let (support, v) = ds4_core::catalog_from_bind_name(name)
                .ok_or_else(|| format!("unknown bind-names variant: {name}"))?;
            (support, ds4_core::shape_for_variant(v))
        } else {
            let id = ds4_core::identify_gguf(std::path::Path::new(model_path))
                .map_err(|e| e.to_string())?;
            (None, id.shape)
        };
        print!(
            "{}",
            ds4_core::BindPlan::resolve_catalog(support, shape, &inv).dump()
        );
        return Ok(0);
    }
    if args.validate {
        print!(
            "{}",
            ds4_core::dump_validate(std::path::Path::new(model_path))
        );
        return Ok(0);
    }

    if args.tokenize {
        let fam_name = args
            .tok_family
            .as_deref()
            .ok_or_else(|| " --tokenize requires FAMILY".to_string())?;
        let family = ds4_core::ModelFamily::from_oracle_name(fam_name)
            .ok_or_else(|| format!("unknown tokenizer family: {fam_name}"))?;
        let cmd = args
            .tok_cmd
            .as_deref()
            .ok_or_else(|| "--tokenize requires CMD".to_string())?;
        let vocab = ds4_core::Vocab::load_path(std::path::Path::new(model_path), family)
            .map_err(|e| e.to_string())?;
        print!(
            "{}",
            ds4_core::dump_cmd(&vocab, cmd, args.tok_arg.as_deref().unwrap_or(""))
        );
        return Ok(0);
    }

    let worker = args.dist.role == ds4_dist::Role::Worker;
    let raw_token_diagnostic = !args.tokens.is_empty() || args.predict > 0;
    let generation_text = if !worker && (args.prompt.is_some() || args.prompt_file.is_some()) {
        Some(prompt_text(&args)?)
    } else {
        None
    };
    let want_repl = !worker
        && args.prompt.is_none()
        && args.prompt_file.is_none()
        && args.dump_logprobs.is_none()
        && args.dump_logits.is_none()
        && !args.dump_tokens
        && !args.lifecycle_only
        && !raw_token_diagnostic;
    if !want_repl
        && !worker
        && args.dump_logprobs.is_none()
        && args.dump_logits.is_none()
        && !args.dump_tokens
        && !args.lifecycle_only
        && !raw_token_diagnostic
    {
        validate_one_shot_args(&args)?;
    }
    if !worker && args.dump_logits.is_some() && generation_text.is_none() {
        return Err("--dump-logits requires -p or --prompt-file".into());
    }
    if !worker && args.dump_logprobs.is_some() && generation_text.is_none() {
        return Err("--dump-logprobs requires -p or --prompt-file".into());
    }

    let native_dist = distributed_config(&args.dist);
    let mtp_opts = mtp_open_options(&args);
    let model = match native_dist.as_ref() {
        Some(config) => ds4_core::Model::open_distributed_options(
            model_path,
            args.backend,
            args.threads,
            true,
            args.mtp.as_deref(),
            args.dspark.as_deref(),
            config,
            &mtp_opts,
        ),
        None => ds4_core::Model::open_with_support_options(
            model_path,
            args.backend,
            args.threads,
            true,
            args.mtp.as_deref(),
            args.dspark.as_deref(),
            &mtp_opts,
        ),
    }
    .map_err(|e| e.to_string())?;

    if worker {
        model.boot_prewarm();
        return worker_run::run_assembled_worker(&model, args.ctx, &args.dist);
    }

    if args.dump_logits.is_some() {
        return run_logits_dump(
            &model,
            &args,
            generation_text
                .as_deref()
                .ok_or_else(|| "--dump-logits requires -p or --prompt-file".to_string())?,
        );
    }

    if args.dump_logprobs.is_some() {
        return run_logprob_dump(&model, &args, generation_text.as_deref().unwrap());
    }

    if args.lifecycle_only {
        let session = model.session(args.ctx).map_err(|e| e.to_string())?;
        println!(
            "lifecycle ok backend={:?} ctx={} pos={}",
            args.backend,
            args.ctx,
            session.pos()
        );
        return Ok(0);
    }

    if !args.tokens.is_empty() {
        let mut session = session_ready(&model, &args, args.ctx)?;
        let buf = ds4_core::TokenBuffer::from_tokens(args.tokens);
        session.sync(&buf).map_err(|e| e.to_string())?;
        print!("{}", session.argmax());
        for _ in 0..args.predict {
            let token = session.argmax();
            session.eval(token).map_err(|e| e.to_string())?;
            print!(" {}", session.argmax());
        }
        println!();
        return Ok(0);
    }
    if args.predict > 0 {
        /* Preserve the old diagnostic's no-op behavior when --predict was
         * supplied without raw token IDs. */
        let _session = model.session(args.ctx).map_err(|e| e.to_string())?;
        return Ok(0);
    }
    if want_repl {
        return run_repl(&model, args);
    }
    run_one_shot(&model, &args, generation_text.as_deref().unwrap())
}

fn run_repl(model: &ds4_core::Model, mut args: ShadowArgs) -> Result<i32, String> {
    let mut chat = repl::ReplChat::new(args.nothink, args.ctx);
    let vocab = model.vocab();
    vocab
        .chat_begin(&mut chat.transcript)
        .map_err(|e| e.to_string())?;
    apply_effort_prefix(vocab, &mut chat, None);
    if !args.system.is_empty() {
        vocab
            .chat_append_message(&mut chat.transcript, "system", args.system.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    let mut session = session_ready(model, &args, chat.ctx)?;
    let _sigint = repl::InterruptGuard::install();
    let history_path = repl::history_file_path();
    let tty = repl::stdin_is_tty();
    if tty {
        repl::init_linenoise_history(&history_path);
    }
    let mut history = repl::History::load(history_path.clone());

    print!("{}", repl::REPL_HELP);
    loop {
        let line = match repl::read_repl_line("ds4> ")? {
            repl::ReplRead::Eof => break,
            repl::ReplRead::Interrupted => {
                repl::interrupt_clear();
                continue;
            }
            repl::ReplRead::Line(line) => line,
        };
        let cmd = repl::trim_inplace(&line);
        if !cmd.is_empty() {
            if tty {
                repl::remember_linenoise_line(&history_path, cmd);
            } else if history.add(cmd.to_string()) {
                let _ = history.save();
            }
        }
        match repl::parse_repl_line(&line) {
            Ok(repl::ReplLine::Empty) => {}
            Ok(repl::ReplLine::Help) => print!("{}", repl::REPL_HELP),
            Ok(repl::ReplLine::Quit) => break,
            Ok(repl::ReplLine::Think) => {
                chat.think = repl::ReplThinkCmd::Low;
                args.nothink = false;
                apply_effort_prefix(vocab, &mut chat, Some(&mut session));
                println!("{}", chat.think_message());
            }
            Ok(repl::ReplLine::ThinkMax) => {
                chat.think = repl::ReplThinkCmd::Max;
                args.nothink = false;
                apply_effort_prefix(vocab, &mut chat, Some(&mut session));
                println!("{}", chat.think_message());
            }
            Ok(repl::ReplLine::NoThink) => {
                chat.think = repl::ReplThinkCmd::None;
                args.nothink = true;
                apply_effort_prefix(vocab, &mut chat, Some(&mut session));
                println!("{}", chat.think_message());
            }
            Ok(repl::ReplLine::Ctx(n)) => {
                args.ctx = n;
                args.ctx_set = true;
                chat.ctx = n;
                session = session_ready(model, &args, chat.ctx)?;
                apply_effort_prefix(vocab, &mut chat, Some(&mut session));
            }
            Ok(repl::ReplLine::Power(None)) => {
                println!("Power: {}%.", session.power());
            }
            Ok(repl::ReplLine::Power(Some(n))) => {
                if session.set_power(n).is_err() {
                    eprint!("ds4: failed to set /power\n");
                } else {
                    println!("Power: {n}%.");
                }
            }
            Ok(repl::ReplLine::Read(path)) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("ds4: failed to read {path}: {e}"))?;
                run_chat_turn(model, &args, &mut chat, &mut session, &text)?;
            }
            Ok(repl::ReplLine::Prompt(text)) => {
                run_chat_turn(model, &args, &mut chat, &mut session, &text)?;
            }
            Err(err) => eprint!("{}", err.message()),
        }
    }
    Ok(0)
}

fn require_value(flag: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_i32(flag: &str, value: &str) -> Result<i32, String> {
    value
        .parse()
        .map_err(|_| format!("{flag}: invalid integer {value}"))
}

fn parse_positive_i32(flag: &str, value: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("invalid value for {flag}: {value}"))
}

fn parse_f32_range(flag: &str, value: &str, min: f32, max: f32) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite() && *v >= min && *v <= max)
        .ok_or_else(|| format!("invalid value for {flag}: {value}"))?;
    Ok(parsed)
}

fn parse_positive_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("invalid value for {flag}: {value}"))
}

#[cfg(target_os = "macos")]
fn default_backend() -> Backend {
    Backend::Metal
}

#[cfg(not(target_os = "macos"))]
fn default_backend() -> Backend {
    Backend::Cuda
}

fn parse_backend(value: &str) -> Result<Backend, String> {
    match value {
        "cuda" => Ok(Backend::Cuda),
        "cpu" => Ok(Backend::Cpu),
        "metal" => Ok(Backend::Metal),
        other => Err(format!("unknown backend: {other}")),
    }
}

fn parse_tokens(value: &str) -> Result<Vec<i32>, String> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse()
                .map_err(|_| format!("invalid token id: {part}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format_fragments(format_thinking: bool, parts: &[&[u8]]) -> Vec<u8> {
        format_fragments_colored(format_thinking, false, parts)
    }

    fn format_fragments_colored(
        format_thinking: bool,
        use_color: bool,
        parts: &[&[u8]],
    ) -> Vec<u8> {
        let mut printer = TokenPrinter::with_color(format_thinking, use_color);
        let mut out = Vec::new();
        for part in parts {
            printer.write_text(&mut out, part).unwrap();
        }
        printer.finish(&mut out).unwrap();
        out
    }

    fn args(parts: &[&str]) -> Vec<String> {
        std::iter::once("ds4-rs".to_string())
            .chain(parts.iter().map(|s| (*s).to_string()))
            .collect()
    }

    #[test]
    fn parses_lifecycle() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--lifecycle", "-c", "4096"])).unwrap();
        assert_eq!(parsed.model.as_deref(), Some("m.gguf"));
        assert_eq!(parsed.ctx, 4096);
        assert!(parsed.lifecycle_only);
        assert_eq!(parsed.backend, default_backend());
    }

    #[test]
    fn parses_distributed_worker_options_for_native_oracle_runtime() {
        let parsed = parse_args(args(&[
            "--role",
            "worker",
            "--layers",
            "21:output",
            "--listen",
            "0.0.0.0",
            "7100",
            "--coordinator",
            "10.0.0.1",
            "7000",
        ]))
        .unwrap();

        assert_eq!(parsed.dist.role, ds4_dist::Role::Worker);
        assert_eq!(parsed.dist.layers.start, 21);
        assert!(parsed.dist.layers.has_output);
        assert_eq!(parsed.dist.listen_host.as_deref(), Some("0.0.0.0"));
        assert_eq!(parsed.dist.listen_port, 7100);
        assert_eq!(parsed.dist.coordinator_host.as_deref(), Some("10.0.0.1"));
        assert_eq!(parsed.dist.coordinator_port, 7000);

        let native = distributed_config(&parsed.dist).unwrap();
        assert_eq!(native.role, ds4_core::DistributedRole::Worker);
        assert_eq!(native.layer_start, 21);
        assert_eq!(native.layer_end, u32::MAX);
        assert!(native.has_output);
    }

    #[test]
    fn worker_role_and_layers_hello_uses_bound_listen_port() {
        // Given: worker role + layers, listen port unset
        let parsed = parse_args(args(&[
            "--role",
            "worker",
            "--layers",
            "20:output",
            "--coordinator",
            "127.0.0.1",
            "7000",
        ]))
        .unwrap();
        assert_eq!(parsed.dist.role, ds4_dist::Role::Worker);
        assert!(parsed.dist.layers.set);

        // When: bind from listen_host / listen_port (0 if unset) and plan HELLO
        let requested = worker_run::worker_listen_port(parsed.dist.listen_port);
        let (_listener, port) =
            ds4_dist::open_data_listener(parsed.dist.listen_host.as_deref(), requested).unwrap();
        let meta = session_exec::slice_meta(7, &ds4_core::SHAPE_FLASH, 4096, &parsed.dist.layers);
        let plan = worker_run::worker_plan(&meta, 2, u32::from(port), ds4_core::SHAPE_FLASH.name);

        // Then: HELLO carries the bound nonzero data port
        assert_eq!(requested, 0);
        assert_ne!(plan.hello.listen_port, 0);
        assert_eq!(plan.hello.listen_port, u32::from(port));
    }

    #[test]
    fn worker_missing_coordinator_host_errors_via_validate_options() {
        // Given: worker role + layers, no coordinator host
        let err = parse_args(args(&["--role", "worker", "--layers", "20:output"])).unwrap_err();

        // When/Then: validate_options (via parse_args) still requires coordinator
        assert_eq!(err, "--role worker requires --coordinator HOST PORT");
        let mut opt = ds4_dist::Options {
            role: ds4_dist::Role::Worker,
            ..ds4_dist::Options::default()
        };
        opt.layers.set = true;
        assert_eq!(
            ds4_dist::validate_options(&opt).unwrap_err(),
            "--role worker requires --coordinator HOST PORT"
        );
    }

    #[test]
    fn maps_distributed_coordinator_options_to_native_oracle_runtime() {
        let parsed = parse_args(args(&[
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

        let native = distributed_config(&parsed.dist).unwrap();
        assert_eq!(native.role, ds4_core::DistributedRole::Coordinator);
        assert_eq!(native.layer_start, 0);
        assert_eq!(native.layer_end, 20);
        assert_eq!(native.prefill_chunk, 4096);
        assert_eq!(native.prefill_window, 4);
        assert_eq!(native.activation_bits, 16);
    }

    #[test]
    fn uses_c_cli_core_defaults() {
        let parsed = parse_args(args(&[])).unwrap();

        assert_eq!(parsed.model.as_deref(), Some("ds4flash.gguf"));
        assert_eq!(parsed.backend, default_backend());
        assert_eq!(parsed.ctx, 32768);
        assert_eq!(parsed.n_predict, 50000);
        assert_eq!(parsed.system, "You are a helpful assistant");
        assert_eq!(parsed.temp, 1.0);
        assert_eq!(parsed.top_p, 1.0);
        assert_eq!(parsed.min_p, 0.05);
        assert_eq!(parsed.seed, 0);
        assert!(!parsed.nothink);
        assert_eq!(parsed.mtp_draft, 1);
        assert_eq!(parsed.mtp_margin, 3.0);
    }

    #[test]
    fn parses_mtp_draft_and_margin() {
        let parsed = parse_args(args(&[
            "--mtp",
            "draft.gguf",
            "--mtp-draft",
            "2",
            "--mtp-margin",
            "4.5",
        ]))
        .unwrap();
        assert_eq!(parsed.mtp.as_deref(), Some("draft.gguf"));
        assert_eq!(parsed.mtp_draft, 2);
        assert_eq!(parsed.mtp_margin, 4.5);
    }

    #[test]
    fn rejects_mtp_draft_and_margin_outside_c_ranges() {
        for bad in [
            ["--mtp-draft", "0"],
            ["--mtp-draft", "-1"],
            ["--mtp-draft", "x"],
            ["--mtp-margin", "-0.1"],
            ["--mtp-margin", "1000.1"],
            ["--mtp-margin", "NaN"],
        ] {
            assert!(parse_args(args(&bad)).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn parses_c_cli_sampling_options() {
        let parsed = parse_args(args(&[
            "--temp", "0.8", "--top-p", "0.9", "--min-p", "0.02", "--seed", "424242",
        ]))
        .unwrap();

        assert_eq!(parsed.temp, 0.8);
        assert_eq!(parsed.top_p, 0.9);
        assert_eq!(parsed.min_p, 0.02);
        assert_eq!(parsed.seed, 424242);
    }

    #[test]
    fn rejects_sampling_outside_c_cli_ranges() {
        for bad in [
            ["--temp", "NaN"],
            ["--temp", "101"],
            ["--top-p", "-0.1"],
            ["--top-p", "1.1"],
            ["--min-p", "-0.1"],
            ["--min-p", "1.1"],
            ["--seed", "0"],
        ] {
            assert!(parse_args(args(&bad)).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn parses_output_budget_and_explicit_raw_token_ids() {
        let parsed = parse_args(args(&[
            "--tokens",
            "17",
            "--token-ids",
            "1, 2,3",
            "--predict",
            "4",
        ]))
        .unwrap();

        assert_eq!(parsed.n_predict, 17);
        assert_eq!(parsed.tokens, vec![1, 2, 3]);
        assert_eq!(parsed.predict, 4);
    }

    #[test]
    fn parses_token_ids() {
        let parsed = parse_args(args(&["--token-ids", "1, 2,3", "--predict", "4"])).unwrap();
        assert_eq!(parsed.tokens, vec![1, 2, 3]);
        assert_eq!(parsed.predict, 4);
    }

    #[test]
    fn raw_token_ids_have_an_actionable_tokens_collision_error() {
        let err = parse_args(args(&["--tokens", "1,2,3"])).unwrap_err();

        assert!(err.contains("use --token-ids for raw token IDs"), "{err}");
    }

    #[test]
    fn rejects_non_positive_c_cli_integer_ranges() {
        assert_eq!(
            parse_args(args(&["--tokens", "0"])).unwrap_err(),
            "invalid value for --tokens: 0"
        );
        assert_eq!(
            parse_args(args(&["--ctx", "-1"])).unwrap_err(),
            "invalid value for --ctx: -1"
        );
    }

    #[test]
    fn rejects_more_than_one_prompt_source() {
        assert_eq!(
            parse_args(args(&["-p", "one", "--prompt-file", "two.txt"])).unwrap_err(),
            "specify only one prompt source"
        );
        assert_eq!(
            parse_args(args(&["--prompt-file", "one.txt", "-p", "two"])).unwrap_err(),
            "specify only one prompt source"
        );
    }

    #[test]
    fn parses_system_and_last_think_mode_wins() {
        let thinking = parse_args(args(&[
            "-sys",
            "system",
            "--nothink",
            "--think",
            "-p",
            "hello",
        ]))
        .unwrap();
        assert_eq!(thinking.system, "system");
        assert!(!thinking.nothink);

        let direct = parse_args(args(&["--think", "--nothink", "-p", "hello"])).unwrap();
        assert!(direct.nothink);
    }

    #[test]
    fn validates_only_supported_one_shot_routes() {
        let greedy = parse_args(args(&["-p", "hello", "--temp", "0", "--nothink"])).unwrap();
        assert!(validate_one_shot_args(&greedy).is_ok());

        let sampled = parse_args(args(&["-p", "hello", "--temp", "0.5", "--nothink"])).unwrap();
        assert!(validate_one_shot_args(&sampled).is_ok());

        let thinking = parse_args(args(&["-p", "hello", "--temp", "0", "--think"])).unwrap();
        assert!(validate_one_shot_args(&thinking).is_ok());

        let mtp = parse_args(args(&[
            "-p",
            "hello",
            "--temp",
            "0",
            "--nothink",
            "--mtp",
            "mtp.gguf",
            "--mtp-draft",
            "2",
        ]))
        .unwrap();
        assert!(validate_one_shot_args(&mtp).is_ok());
        assert!(use_mtp_spec(mtp.temp, mtp.mtp.as_deref(), mtp.mtp_draft));

        let dspark = parse_args(args(&[
            "-p",
            "hello",
            "--temp",
            "0",
            "--nothink",
            "--dspark",
            "draft.gguf",
        ]))
        .unwrap();
        assert!(validate_one_shot_args(&dspark).is_ok());
        assert_eq!(dspark.dspark.as_deref(), Some("draft.gguf"));
    }

    #[test]
    fn mtp_spec_follows_c_gates() {
        assert!(!use_mtp_spec(0.0, None, 2));
        assert!(!use_mtp_spec(0.0, Some("mtp.gguf"), 1));
        assert!(!use_mtp_spec(0.5, Some("mtp.gguf"), 2));
        assert!(use_mtp_spec(0.0, Some("mtp.gguf"), 2));
    }

    #[test]
    fn missing_prompt_is_repl_not_one_shot() {
        let parsed = parse_args(args(&[])).unwrap();
        assert!(parsed.prompt.is_none());
        assert!(parsed.prompt_file.is_none());
        assert!(validate_one_shot_args(&parsed)
            .unwrap_err()
            .contains("one-shot generation requires -p or --prompt-file"));
    }

    #[test]
    fn clamps_generation_to_the_remaining_context() {
        assert_eq!(generation_limit(32768, 100, 50000), 32668);
        assert_eq!(generation_limit(128, 127, 10), 1);
        assert_eq!(generation_limit(128, 128, 10), 0);
        assert_eq!(generation_limit(128, 120, 4), 4);

        assert_eq!(sampled_generation_limit(128, 127, 10), 0);
        assert_eq!(sampled_generation_limit(128, 126, 10), 1);
        assert_eq!(sampled_generation_limit(128, 120, 4), 4);
    }

    #[test]
    fn detects_rendered_chat_prompt() {
        assert!(is_rendered_chat_prompt(
            "<｜begin▁of▁sentence｜>already rendered"
        ));
        assert!(!is_rendered_chat_prompt("plain user prompt"));
    }

    #[test]
    fn formats_thinking_tags_split_across_pieces() {
        assert_eq!(
            format_fragments(true, &[b"<thi", b"nk>plan", b"</thi", b"nk>answer"],),
            b"plan\nanswer\n"
        );
        assert_eq!(format_fragments(true, &[b"<thi", b"x"]), b"<thix\n");
    }

    #[test]
    fn preserves_c_thinking_newline_and_finish_rules() {
        assert_eq!(
            format_fragments(true, &[b"</think>", b"answer\n"]),
            b"answer\n"
        );
        assert_eq!(
            format_fragments(true, &[b"plan\n</think>", b"\nanswer"],),
            b"plan\n\nanswer\n"
        );
        assert_eq!(format_fragments(true, &[b"plan</thi"]), b"plan</thi\n");
        assert_eq!(format_fragments(true, &[]), b"");
    }

    #[test]
    fn leaves_nothink_output_unformatted() {
        assert_eq!(
            format_fragments(false, &[b"<thi", b"nk>x</think>"]),
            b"<think>x</think>\n"
        );
    }

    #[test]
    fn tty_thinking_body_uses_c_grey_ansi() {
        assert_eq!(
            format_fragments_colored(true, true, &[b"<thi", b"nk>plan", b"</thi", b"nk>answer"],),
            b"\x1b[90mplan\x1b[0m\nanswer\n"
        );
        assert_eq!(
            format_fragments_colored(true, true, &[b"plan</think>answer"]),
            b"\x1b[90mplan\x1b[0m\nanswer\n"
        );
        assert_eq!(
            format_fragments_colored(true, true, &[b"plan"]),
            b"\x1b[90mplan\x1b[0m\n"
        );
    }

    #[test]
    fn pipe_thinking_output_has_no_esc() {
        let out = format_fragments_colored(true, false, &[b"<think>plan</think>answer"]);
        assert!(!out.contains(&0x1b), "pipe must not emit ESC, got {out:?}");
        assert_eq!(out, b"plan\nanswer\n");
        assert!(!format_fragments(true, &[b"<think>plan</think>answer"]).contains(&0x1b));
    }

    #[test]
    fn rejects_unknown() {
        assert!(parse_args(args(&["--nope"])).is_err());
    }

    #[test]
    fn parses_identify() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--identify"])).unwrap();
        assert!(parsed.identify);
        assert_eq!(parsed.model.as_deref(), Some("m.gguf"));
    }

    #[test]
    fn parses_inventory() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--inventory"])).unwrap();
        assert!(parsed.inventory);
    }

    #[test]
    fn parses_tokenize() {
        let parsed = parse_args(args(&[
            "-m",
            "m.gguf",
            "--tokenize",
            "motif3",
            "encode",
            "6869",
        ]))
        .unwrap();
        assert!(parsed.tokenize);
        assert_eq!(parsed.tok_family.as_deref(), Some("motif3"));
        assert_eq!(parsed.tok_cmd.as_deref(), Some("encode"));
        assert_eq!(parsed.tok_arg.as_deref(), Some("6869"));
    }

    #[test]
    fn parses_session_plan() {
        let parsed =
            parse_args(args(&["--session-plan", "rewrite", "1024", "1100", "1024"])).unwrap();
        assert!(parsed.session_plan);
        assert_eq!(parsed.session_cmd.as_deref(), Some("rewrite"));
        assert_eq!(parsed.session_args, vec!["1024", "1100", "1024"]);
    }

    #[test]
    fn parses_session_payload() {
        let parsed = parse_args(args(&["--session-payload", "encode-deepseek"])).unwrap();
        assert!(parsed.session_payload);
        assert_eq!(parsed.payload_cmd.as_deref(), Some("encode-deepseek"));
        assert!(parsed.payload_args.is_empty());
    }

    #[test]
    fn parses_bind_names() {
        let parsed = parse_args(args(&["--bind-names", "motif3"])).unwrap();
        assert!(parsed.bind_names);
        assert_eq!(parsed.bind_names_variant.as_deref(), Some("motif3"));
    }

    #[test]
    fn parses_bind_names_support() {
        let parsed = parse_args(args(&["--bind-names", "mtp-flash"])).unwrap();
        assert!(parsed.bind_names);
        assert_eq!(parsed.bind_names_variant.as_deref(), Some("mtp-flash"));
    }

    #[test]
    fn parses_layout_support() {
        let parsed = parse_args(args(&["--layout", "dspark-pro"])).unwrap();
        assert!(parsed.layout);
        assert_eq!(parsed.layout_variant.as_deref(), Some("dspark-pro"));
    }

    #[test]
    fn parses_bind_plan() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--bind-plan"])).unwrap();
        assert!(parsed.bind_plan);
        assert_eq!(parsed.model.as_deref(), Some("m.gguf"));
    }

    #[test]
    fn parses_bind_plan_with_support_catalog() {
        let parsed = parse_args(args(&[
            "-m",
            "mtp.gguf",
            "--bind-names",
            "mtp-flash",
            "--bind-plan",
        ]))
        .unwrap();
        assert!(parsed.bind_plan);
        assert!(parsed.bind_names);
        assert_eq!(parsed.bind_names_variant.as_deref(), Some("mtp-flash"));
        assert_eq!(parsed.model.as_deref(), Some("mtp.gguf"));
    }

    #[test]
    fn parses_validate() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--validate"])).unwrap();
        assert!(parsed.validate);
        assert_eq!(parsed.model.as_deref(), Some("m.gguf"));
    }

    #[test]
    fn parses_layout() {
        let parsed = parse_args(args(&["--layout", "flash"])).unwrap();
        assert!(parsed.layout);
        assert_eq!(parsed.layout_variant.as_deref(), Some("flash"));
    }

    #[test]
    fn parses_dump_logits() {
        let parsed = parse_args(args(&["--dump-logits", "/tmp/logits.json"])).unwrap();
        assert_eq!(parsed.dump_logits.as_deref(), Some("/tmp/logits.json"));
        assert!(parsed.dump_logprobs.is_none());
    }

    #[test]
    fn help_contains_dump_logits() {
        assert!(help_text("ds4-rs").contains("--dump-logits"));
    }

    #[test]
    fn dump_logits_missing_value_errors_like_c() {
        assert_eq!(
            parse_args(args(&["--dump-logits"])).unwrap_err(),
            "ds4: missing value for --dump-logits"
        );
    }

    #[test]
    fn dump_logits_missing_file_errors_like_c() {
        let dir = std::env::temp_dir().join("ds4-dump-logits-not-a-file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.to_str().unwrap();
        let err = write_logits_file(path, "{}").unwrap_err();
        assert_eq!(
            err,
            format!("ds4: failed to open --dump-logits file: {path}")
        );
    }

    #[test]
    fn dump_logits_json_matches_c_shape() {
        let body = format_logits_json(&LogitsDumpJson {
            model: "m.gguf",
            backend: "cuda",
            quant_bits: 2,
            prompt_tokens: 3,
            ctx: 128,
            argmax_token_json: "{\"id\":1,\"text\":\"a\",\"bytes\":[97]}",
            argmax_logit: 1.5,
            logits: &[1.0, 1.5, f32::INFINITY],
        });
        assert_eq!(
            body,
            "{\n  \"source\":\"ds4\",\n  \"model\":\"m.gguf\",\n  \"backend\":\"cuda\",\n  \"quant_bits\":2,\n  \"prompt_tokens\":3,\n  \"ctx\":128,\n  \"vocab\":3,\n  \"argmax_token\":{\"id\":1,\"text\":\"a\",\"bytes\":[97]},\n  \"argmax_logit\":1.5,\n  \"logits\":[\n    1,1.5,null\n  ]\n}\n"
        );
    }

    #[test]
    fn parses_dump_tokens() {
        let parsed = parse_args(args(&["--dump-tokens", "-p", "hi"])).unwrap();
        assert!(parsed.dump_tokens);
        assert_eq!(parsed.prompt.as_deref(), Some("hi"));
        assert!(parsed.dump_logits.is_none());
        assert!(parsed.dump_logprobs.is_none());
    }

    #[test]
    fn help_contains_dump_tokens() {
        assert!(help_text("ds4-rs").contains("--dump-tokens"));
    }

    /// C `ds4 --help` flags the shadow already parses (8.3 claimed modes).
    /// `--mtp ` keeps `--mtp-draft` from counting as `--mtp`.
    const CLAIMED_C_CLI_FLAGS: &[&str] = &[
        "-m, --model",
        "--mtp ",
        "--mtp-draft",
        "--mtp-margin",
        "-c, --ctx",
        "--metal",
        "--cuda",
        "--cpu",
        "--backend",
        "-t, --threads",
        "-p, --prompt",
        "--prompt-file",
        "-sys, --system",
        "-n, --tokens",
        "--temp",
        "--top-p",
        "--min-p",
        "--seed",
        "--think",
        "--nothink",
        "--dump-tokens",
        "--dump-logits",
        "--dump-logprobs",
        "--logprobs-top-k",
        "--role",
        "--layers",
        "--listen",
        "--coordinator",
        "--dist-prefill-chunk",
        "--dist-prefill-window",
        "--dist-activation-bits",
        "--dist-replay-check",
        "--debug",
        "-h, --help",
    ];

    #[test]
    fn help_contains_each_claimed_c_flag() {
        let help = help_text("ds4-rs");
        for flag in CLAIMED_C_CLI_FLAGS {
            assert!(
                help.contains(flag),
                "ds4-rs help missing claimed C flag {flag}"
            );
        }
    }

    #[test]
    fn dump_tokens_missing_prompt_errors_like_c() {
        let parsed = parse_args(args(&["--dump-tokens"])).unwrap();
        assert_eq!(
            run_dump_tokens_cli(&parsed).unwrap_err(),
            "ds4: --dump-tokens requires -p or --prompt-file"
        );
    }

    #[test]
    fn dump_tokens_path_does_not_call_eval() {
        let parsed = parse_args(args(&["--dump-tokens", "-p", "hi"])).unwrap();
        assert_eq!(eval_plan(&parsed), EvalPlan::Skip);
        assert_eq!(
            format_dump_tokens(&[7, 8], &[(7, b"ab".as_slice()), (8, b"cd".as_slice())]),
            b"[7, 8]\n     7  ab\n     8  cd\n"
        );
    }
}
