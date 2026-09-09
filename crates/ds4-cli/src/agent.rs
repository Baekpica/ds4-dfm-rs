const BUILT_IN_TOOLS_PROMPT: &str = include_str!("agent_tools_prompt.txt");
const DSML_OPEN: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const DSML_OPEN_MISSING_BAR: &[u8] = "<DSML｜tool_calls>".as_bytes();
const THINK_OPEN: &[u8] = b"<think>";
const THINK_CLOSE: &[u8] = b"</think>";
const TOOL_UNSUPPORTED_ERROR: &str =
    "tool execution is not implemented in ds4-agent-rs; use ./ds4-agent";

mod approval;
mod bash;
mod compact;
mod edit;
mod kv;
mod search;
mod trace;
mod tui;
mod web_tools;
mod write;

#[cfg(test)]
mod approval_round;
#[cfg(test)]
mod bash_parity;
#[cfg(test)]
mod compact_parity;
#[cfg(test)]
mod edit_parity;
#[cfg(test)]
mod search_parity;
#[cfg(test)]
mod write_parity;

#[derive(Debug)]
pub struct AgentArgs {
    model: String,
    backend: ds4_core::Backend,
    ctx: i32,
    threads: i32,
    tokens: i32,
    prompt: Option<String>,
    system: String,
    temp: f32,
    top_p: f32,
    min_p: f32,
    seed: u64,
    think: ds4_core::ChatThinkMode,
    chdir: Option<String>,
    mtp: Option<String>,
    mtp_draft: i32,
    mtp_margin: f32,
    quality: bool,
    warm_weights: bool,
    power_percent: i32,
    steering_file: Option<String>,
    steering_attn: f32,
    steering_ffn: f32,
    trace: Option<String>,
    dist: ds4_dist::Options,
    non_interactive: bool,
    help: bool,
}

impl Default for AgentArgs {
    fn default() -> Self {
        Self {
            model: "ds4flash.gguf".into(),
            backend: default_backend(),
            ctx: 100000,
            threads: 0,
            tokens: 50000,
            prompt: None,
            system: "You are a helpful coding assistant running inside ds4-agent.".into(),
            temp: 1.0,
            top_p: 1.0,
            min_p: 0.05,
            seed: 0,
            think: ds4_core::ChatThinkMode::Low,
            chdir: None,
            mtp: None,
            mtp_draft: 1,
            mtp_margin: 3.0,
            quality: false,
            warm_weights: false,
            power_percent: 100,
            steering_file: None,
            steering_attn: 0.0,
            steering_ffn: 0.0,
            trace: None,
            dist: ds4_dist::Options::default(),
            non_interactive: false,
            help: false,
        }
    }
}

fn default_backend() -> ds4_core::Backend {
    if cfg!(target_os = "macos") {
        ds4_core::Backend::Metal
    } else {
        ds4_core::Backend::Cuda
    }
}

fn need_value(option: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("missing value for {option}"))
}

fn parse_positive_i32(option: &str, value: String) -> Result<i32, String> {
    value
        .parse::<i32>()
        .ok()
        .filter(|parsed| *parsed > 0)
        .ok_or_else(|| format!("invalid value for {option}: {value}"))
}

fn parse_f32_range(option: &str, value: String, min: f32, max: f32) -> Result<f32, String> {
    value
        .parse::<f32>()
        .ok()
        .filter(|parsed| parsed.is_finite() && *parsed >= min && *parsed <= max)
        .ok_or_else(|| format!("invalid value for {option}: {value}"))
}

fn parse_seed(option: &str, value: String) -> Result<u64, String> {
    value
        .parse::<u64>()
        .ok()
        .filter(|parsed| *parsed != 0)
        .ok_or_else(|| format!("invalid value for {option}: {value}"))
}

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<AgentArgs, String> {
    let mut parsed = AgentArgs::default();
    let mut steering_scale_set = false;
    let mut args = args.into_iter();
    let _argv0 = args.next();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                parsed.help = true;
                return Ok(parsed);
            }
            "--non-interactive" => parsed.non_interactive = true,
            "-p" | "--prompt" => parsed.prompt = Some(need_value(&arg, args.next())?),
            "-m" | "--model" => parsed.model = need_value(&arg, args.next())?,
            "-c" | "--ctx" => {
                let value = need_value(&arg, args.next())?;
                parsed.ctx = parse_positive_i32(&arg, value)?;
            }
            "-n" | "--tokens" => {
                let value = need_value(&arg, args.next())?;
                parsed.tokens = parse_positive_i32(&arg, value)?;
            }
            "-sys" | "--system" => parsed.system = need_value(&arg, args.next())?,
            "--temp" => {
                let value = need_value(&arg, args.next())?;
                parsed.temp = parse_f32_range(&arg, value, 0.0, 100.0)?;
            }
            "--top-p" => {
                let value = need_value(&arg, args.next())?;
                parsed.top_p = parse_f32_range(&arg, value, 0.0, 1.0)?;
            }
            "--min-p" => {
                let value = need_value(&arg, args.next())?;
                parsed.min_p = parse_f32_range(&arg, value, 0.0, 1.0)?;
            }
            "--seed" => {
                let value = need_value(&arg, args.next())?;
                parsed.seed = parse_seed(&arg, value)?;
            }
            "--think" => parsed.think = ds4_core::ChatThinkMode::Low,
            "--think-max" => parsed.think = ds4_core::ChatThinkMode::High,
            "--nothink" => parsed.think = ds4_core::ChatThinkMode::None,
            "--backend" => {
                parsed.backend = match need_value(&arg, args.next())?.as_str() {
                    "cuda" => ds4_core::Backend::Cuda,
                    "metal" => ds4_core::Backend::Metal,
                    "cpu" => ds4_core::Backend::Cpu,
                    value => return Err(format!("invalid backend: {value}")),
                };
            }
            "--cuda" => parsed.backend = ds4_core::Backend::Cuda,
            "--metal" => parsed.backend = ds4_core::Backend::Metal,
            "--cpu" => parsed.backend = ds4_core::Backend::Cpu,
            "-t" | "--threads" => {
                let value = need_value(&arg, args.next())?;
                parsed.threads = parse_positive_i32(&arg, value)?;
            }
            "--chdir" => parsed.chdir = Some(need_value(&arg, args.next())?),
            "--mtp" => parsed.mtp = Some(need_value(&arg, args.next())?),
            "--mtp-draft" => {
                let value = need_value(&arg, args.next())?;
                parsed.mtp_draft = parse_positive_i32(&arg, value)?;
            }
            "--mtp-margin" => {
                let value = need_value(&arg, args.next())?;
                parsed.mtp_margin = parse_f32_range(&arg, value, 0.0, 1000.0)?;
            }
            "--quality" => parsed.quality = true,
            "--warm-weights" => parsed.warm_weights = true,
            "--power" => {
                let value = need_value(&arg, args.next())?;
                let parsed_power = parse_positive_i32(&arg, value)?;
                if !(1..=100).contains(&parsed_power) {
                    return Err("--power must be between 1 and 100".into());
                }
                parsed.power_percent = parsed_power;
            }
            "--trace" => parsed.trace = Some(need_value(&arg, args.next())?),
            "--dir-steering-file" => {
                parsed.steering_file = Some(need_value(&arg, args.next())?);
            }
            "--dir-steering-ffn" => {
                let value = need_value(&arg, args.next())?;
                parsed.steering_ffn = parse_f32_range(&arg, value, -100.0, 100.0)?;
                steering_scale_set = true;
            }
            "--dir-steering-attn" => {
                let value = need_value(&arg, args.next())?;
                parsed.steering_attn = parse_f32_range(&arg, value, -100.0, 100.0)?;
                steering_scale_set = true;
            }
            other => match ds4_dist::parse_cli_arg(other, &mut args, &mut parsed.dist)? {
                ds4_dist::CliResult::Matched => {}
                ds4_dist::CliResult::NotMatched => {
                    return Err(format!("unknown option: {other}"));
                }
                ds4_dist::CliResult::Error => unreachable!(),
            },
        }
    }
    if parsed.steering_file.is_some() && !steering_scale_set {
        parsed.steering_ffn = 1.0;
    }
    ds4_dist::prepare_engine_options(&parsed.dist)?;
    if parsed.dist.role == ds4_dist::Role::Worker {
        return Err("--role worker is a serving mode; start workers with ./ds4".into());
    }
    Ok(parsed)
}

fn session_ready<'m>(
    model: &'m ds4_core::Model,
    args: &AgentArgs,
) -> Result<ds4_core::Session<'m>, String> {
    let session = model.session(args.ctx).map_err(|error| error.to_string())?;
    if args.dist.role != ds4_dist::Role::Coordinator {
        return Ok(session);
    }
    let mut ticks = 0u32;
    let mut last = String::new();
    loop {
        match session.distributed_route_ready() {
            Ok(true) => {
                if ticks != 0 {
                    eprintln!("ds4-agent: distributed route ready");
                }
                return Ok(session);
            }
            Ok(false) => {
                let why = "route incomplete";
                if last != why || ticks % 20 == 0 {
                    eprintln!("ds4-agent: waiting for distributed route: {why}");
                    last = why.into();
                }
            }
            Err(error) => {
                let why = error.to_string();
                if last != why || ticks % 20 == 0 {
                    eprintln!("ds4-agent: waiting for distributed route: {why}");
                    last = why;
                }
            }
        }
        ticks = ticks.wrapping_add(1);
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

fn use_mtp_spec(temp: f32, mtp: Option<&str>, draft: i32) -> bool {
    temp <= 0.0 && mtp.is_some() && draft > 1 && std::env::var_os("DS4_MTP_SPEC_DISABLE").is_none()
}

fn mtp_open_options(args: &AgentArgs) -> Vec<ds4_core::ModelOpenOption> {
    let mut options = Vec::new();
    if args.mtp.is_some() {
        options.push(ds4_core::ModelOpenOption::MtpDraftTokens(args.mtp_draft));
        options.push(ds4_core::ModelOpenOption::MtpMargin(args.mtp_margin));
    }
    if args.quality {
        options.push(ds4_core::ModelOpenOption::Quality);
    }
    if args.warm_weights {
        options.push(ds4_core::ModelOpenOption::WarmWeights);
    }
    options.push(ds4_core::ModelOpenOption::PowerPercent(
        args.power_percent as u8,
    ));
    if let Some(file) = &args.steering_file {
        options.push(ds4_core::ModelOpenOption::SteeringFile(file.clone()));
    }
    if args.steering_attn != 0.0 {
        options.push(ds4_core::ModelOpenOption::SteeringAttn(args.steering_attn));
    }
    if args.steering_ffn != 0.0 {
        options.push(ds4_core::ModelOpenOption::SteeringFfn(args.steering_ffn));
    }
    options
}

const THINK_EFFORT_MIN_CONTEXT: i32 = 393216;

fn agent_family(family: ds4_core::ModelFamily) -> Result<(), String> {
    if family == ds4_core::ModelFamily::DeepSeek4 {
        return Ok(());
    }
    Err("ds4-agent's built-in executor requires DeepSeek DSML; use an HTTP tool client for this family".into())
}

fn effective_think(args: &AgentArgs) -> ds4_core::ChatThinkMode {
    if matches!(
        args.think,
        ds4_core::ChatThinkMode::High | ds4_core::ChatThinkMode::Max
    ) && args.ctx < THINK_EFFORT_MIN_CONTEXT
    {
        ds4_core::ChatThinkMode::Low
    } else {
        args.think
    }
}

fn epoch_seconds() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".into())
}

fn current_local_datetime() -> String {
    // The parity oracle supplies a fixed `when`; this production-only clock leaf
    // is deliberately outside the byte-for-byte parity claim.
    let fallback = epoch_seconds();
    match std::process::Command::new("/bin/date")
        .arg("+%Y-%m-%d %H:%M:%S %Z")
        .output()
    {
        Ok(output) => validate_clock_output(output.status.success(), &output.stdout, &fallback),
        Err(_) => fallback,
    }
}

fn build_system_transcript(
    vocab: &ds4_core::Vocab,
    args: &AgentArgs,
    think: ds4_core::ChatThinkMode,
) -> Result<ds4_core::TokenBuffer, String> {
    let mut transcript = ds4_core::TokenBuffer::new();
    vocab
        .chat_begin(&mut transcript)
        .map_err(|error| error.to_string())?;
    vocab.chat_append_effort_prefix(&mut transcript, think);
    for token in vocab.encode_rendered_chat(built_in_tools_prompt()) {
        transcript.push(token);
    }
    if !args.system.is_empty() {
        let extra = format!("\n\n{}", args.system);
        vocab
            .chat_append_message(&mut transcript, "system", extra.as_bytes())
            .map_err(|error| error.to_string())?;
    }
    Ok(transcript)
}

fn append_user_turn(
    vocab: &ds4_core::Vocab,
    transcript: &mut ds4_core::TokenBuffer,
    prompt: &str,
    datetime: Option<&str>,
    think: ds4_core::ChatThinkMode,
) -> Result<(), String> {
    if let Some(datetime) = datetime {
        vocab
            .chat_append_message(transcript, "system", datetime.as_bytes())
            .map_err(|error| error.to_string())?;
    }
    vocab
        .chat_append_message(transcript, "user", prompt.as_bytes())
        .map_err(|error| error.to_string())?;
    vocab
        .chat_append_assistant_prefix(transcript, think)
        .map_err(|error| error.to_string())
}

fn random_seed() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    now.as_secs() ^ (u64::from(std::process::id()) << 32) ^ u64::from(now.subsec_nanos())
}

#[cfg(test)]
fn tool_marker_in_new_suffix(raw: &[u8], appended: usize) -> bool {
    let max_marker = DSML_OPEN.len().max(DSML_OPEN_MISSING_BAR.len());
    let start = raw
        .len()
        .saturating_sub(appended.saturating_add(max_marker.saturating_sub(1)));
    contains_bytes(&raw[start..], DSML_OPEN) || contains_bytes(&raw[start..], DSML_OPEN_MISSING_BAR)
}

pub fn run(name: &str, args: AgentArgs) -> Result<i32, String> {
    use std::io::Write;

    if args.help {
        print!("{}", help_text(name));
        return Ok(0);
    }
    if let Some(path) = args.chdir.as_deref() {
        std::env::set_current_dir(path)
            .map_err(|error| format!("failed to chdir to {path}: {error}"))?;
    }

    let identified = ds4_core::identify_gguf(std::path::Path::new(&args.model))
        .map_err(|error| error.to_string())?;
    agent_family(identified.shape.family)?;

    let think = effective_think(&args);
    if args.think == ds4_core::ChatThinkMode::High && think != ds4_core::ChatThinkMode::High {
        eprintln!(
            "{name}: warning: --think-max needs --ctx >= {THINK_EFFORT_MIN_CONTEXT}; \
             ctx={} uses normal thinking instead",
            args.ctx
        );
    }

    let mut log = trace::Trace::open(args.trace.as_deref(), name)?;
    let model = match crate::distributed_config(&args.dist) {
        Some(config) => ds4_core::Model::open_distributed_options(
            &args.model,
            args.backend,
            args.threads,
            false,
            args.mtp.as_deref(),
            None,
            &config,
            &mtp_open_options(&args),
        ),
        None => ds4_core::Model::open_with_support_options(
            &args.model,
            args.backend,
            args.threads,
            false,
            args.mtp.as_deref(),
            None,
            &mtp_open_options(&args),
        ),
    }
    .map_err(|error| error.to_string())?;
    let mut session = session_ready(&model, &args)?;
    let cache_dir = kv::default_cache_dir();
    let _ = std::fs::create_dir_all(&cache_dir);
    let store = kv::SessionStore::open(cache_dir, u8::try_from(model.model_id()).unwrap_or(0));
    let mut identity = kv::SessionIdentity::default();
    let mut transcript = build_system_transcript(model.vocab(), &args, think)?;
    session
        .sync(&transcript)
        .map_err(|error| error.to_string())?;
    log.event(&format!(
        "agent worker start ctx={} backend={} model={} trace={}",
        args.ctx,
        trace::backend_name(args.backend),
        args.model,
        args.trace.as_deref().unwrap_or("")
    ))?;
    log.tokens("initial_system_prompt", &model, transcript.as_slice(), 0)?;

    let mut rng = if args.seed == 0 {
        random_seed()
    } else {
        args.seed
    };
    let mut web = web_tools::non_interactive_web();
    let mut read_cursor = web_tools::ReadCursor::default();
    let mut bash_jobs = bash::BashTable::default();
    let mut ask = approval::StdinAsk;
    let one_shot = args.prompt.clone();
    let mut first_turn = true;
    let surface = tui::Surface::from_stdout();
    loop {
        let prompt = if let Some(prompt) = one_shot.as_ref() {
            if !first_turn {
                break;
            }
            prompt.clone()
        } else {
            match read_stdin_prompt()? {
                Some(prompt) if !prompt.is_empty() => prompt,
                Some(_) => continue,
                None => break,
            }
        };
        if one_shot.is_none() {
            if let Some(message) = kv::handle_slash(
                prompt.trim(),
                kv::Live {
                    store: &store,
                    identity: &mut identity,
                    model: &model,
                    session: &mut session,
                    transcript: &mut transcript,
                },
            ) {
                print!("{message}");
                continue;
            }
        }
        identity.note_user_prompt(&prompt, kv::unix_now());
        let sys = build_system_transcript(model.vocab(), &args, think)?;
        compact::compact_if_needed(
            &model,
            &mut session,
            model.vocab(),
            args.ctx,
            &sys,
            &mut transcript,
            "soft limit before user turn",
        )?;
        let datetime = if first_turn {
            Some(format_datetime_context(&current_local_datetime()))
        } else {
            None
        };
        if let Some(datetime) = datetime.as_deref() {
            log.text("datetime-context", datetime.as_bytes())?;
        }
        append_user_turn(
            model.vocab(),
            &mut transcript,
            &prompt,
            datetime.as_deref(),
            think,
        )?;
        log.text("user", prompt.as_bytes())?;
        let cached = session.pos();
        let prompt_len = i32::try_from(transcript.len()).unwrap_or(i32::MAX);
        let suffix = prompt_len.saturating_sub(cached);
        log.event(&format!(
            "prefill tool_round=0 transcript={prompt_len} prompt={prompt_len} cached={cached} suffix={suffix} think={}",
            trace::think_mode_name(think)
        ))?;
        log.tokens(
            "prefill_suffix",
            &model,
            transcript.as_slice(),
            cached.max(0) as usize,
        )?;
        session
            .sync_progress(&transcript, |checkpoint| {
                let done = i32::try_from(checkpoint.current()).unwrap_or(i32::MAX);
                let total = i32::try_from(checkpoint.total()).unwrap_or(i32::MAX);
                tui::emit_prefill(
                    surface,
                    tui::PrefillView {
                        ctx_size: args.ctx,
                        done,
                        total,
                        label: 0,
                        power_percent: args.power_percent,
                    },
                );
            })
            .map_err(|error| error.to_string())?;
        let _ = tui::clear_progress_frame(&mut std::io::stderr(), surface);
        let mut output = Vec::new();
        loop {
            let room = session
                .ctx()
                .saturating_sub(session.pos())
                .saturating_sub(1);
            let max_tokens = args.tokens.min(room).max(0);
            let use_mtp = use_mtp_spec(args.temp, args.mtp.as_deref(), args.mtp_draft);
            let eos = model.token_eos();
            let mut raw = Vec::new();
            let mut generated = 0i32;
            let gen_started = std::time::Instant::now();
            while generated < max_tokens {
                let token = session.sample(args.temp, 0, args.top_p, args.min_p, &mut rng);
                if token < 0 {
                    return Err("failed to sample the next token".into());
                }
                if token == eos || model.token_is_stop(token) {
                    break;
                }
                let accepted = if use_mtp {
                    session
                        .eval_speculative_argmax(token, max_tokens - generated, eos)
                        .map_err(|error| error.to_string())?
                } else {
                    session.eval(token).map_err(|error| error.to_string())?;
                    vec![token]
                };
                let mut stop = false;
                for accepted in accepted {
                    if accepted == eos || model.token_is_stop(accepted) {
                        stop = true;
                        break;
                    }
                    transcript.push(accepted);
                    let piece = model
                        .token_text(accepted)
                        .map_err(|error| error.to_string())?;
                    log.token(accepted, &piece, generated + 1)?;
                    raw.extend_from_slice(&piece);
                    generated += 1;
                    let elapsed = gen_started.elapsed().as_secs_f64();
                    tui::emit_generation(
                        surface,
                        tui::GenerationView {
                            ctx_used: i32::try_from(transcript.len()).unwrap_or(i32::MAX),
                            ctx_size: args.ctx,
                            generated,
                            gen_tps: if elapsed > 0.0 {
                                f64::from(generated) / elapsed
                            } else {
                                0.0
                            },
                            power_percent: args.power_percent,
                        },
                    );
                    if web_tools::block_complete(&raw) || generated >= max_tokens {
                        stop = true;
                        break;
                    }
                }
                if stop {
                    break;
                }
            }
            transcript.push(model.token_eos());

            if web_tools::has_block(&raw) {
                let sys = build_system_transcript(model.vocab(), &args, think)?;
                compact::compact_if_needed(
                    &model,
                    &mut session,
                    model.vocab(),
                    args.ctx,
                    &sys,
                    &mut transcript,
                    "soft limit before tool continuation",
                )?;
                let mut gate = if args.non_interactive {
                    approval::Approval::NonInteractive
                } else {
                    approval::Approval::Interactive(&mut ask)
                };
                let round = web_tools::handle_round_with_tools(
                    &raw,
                    &mut web,
                    &mut read_cursor,
                    &mut bash_jobs,
                    &mut gate,
                )
                .map_err(|_| TOOL_UNSUPPORTED_ERROR.to_string())?;
                output.extend_from_slice(&project_output(&[&round.visible])?);
                let observation = fit_tool_observation(
                    &model,
                    &mut session,
                    model.vocab(),
                    &args,
                    think,
                    &mut transcript,
                    round.observation,
                )?;
                model
                    .vocab()
                    .chat_append_message(&mut transcript, "tool", &observation)
                    .map_err(|error| error.to_string())?;
                model
                    .vocab()
                    .chat_append_assistant_prefix(&mut transcript, think)
                    .map_err(|error| error.to_string())?;
                session
                    .sync_progress(&transcript, |checkpoint| {
                        let done = i32::try_from(checkpoint.current()).unwrap_or(i32::MAX);
                        let total = i32::try_from(checkpoint.total()).unwrap_or(i32::MAX);
                        tui::emit_prefill(
                            surface,
                            tui::PrefillView {
                                ctx_size: args.ctx,
                                done,
                                total,
                                label: 0,
                                power_percent: args.power_percent,
                            },
                        );
                    })
                    .map_err(|error| error.to_string())?;
                continue;
            }
            output.extend_from_slice(&project_output(&[&raw])?);
            break;
        }
        let _ = tui::clear_progress_frame(&mut std::io::stderr(), surface);
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        if surface.is_tui() {
            tui::GeneratedSink::new(&mut stdout, surface)
                .push(&output)
                .map_err(|error| error.to_string())?;
        } else {
            stdout
                .write_all(&output)
                .map_err(|error| error.to_string())?;
        }
        stdout.flush().map_err(|error| error.to_string())?;
        first_turn = false;
        if one_shot.is_some() {
            break;
        }
    }
    Ok(0)
}

fn read_stdin_prompt() -> Result<Option<String>, String> {
    use std::io::Read;
    use std::os::fd::AsRawFd;
    use std::time::{Duration, Instant};

    eprint!("+DWARFSTAR_WAITING\n");
    let stdin = std::io::stdin();
    let fd = stdin.as_raw_fd();
    // SAFETY: [Category 8 — FFI] `fd` is the live stdin descriptor from
    // `AsRawFd`; `fcntl(F_GETFL)` only reads flags and does not retain `fd`.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err("ds4-agent: nonblocking stdin".into());
    }
    // SAFETY: [Category 8 — FFI] same live stdin `fd`; `F_SETFL` applies the
    // captured flags plus `O_NONBLOCK`. The restore closure resets those flags.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } != 0 {
        return Err("ds4-agent: nonblocking stdin".into());
    }
    // SAFETY: [Category 8 — FFI] restores the flags captured before SETFL on
    // the same stdin `fd`; the descriptor outlives this closure.
    let restore = || unsafe {
        libc::fcntl(fd, libc::F_SETFL, flags);
    };

    let mut buf = Vec::new();
    let mut last_data = None;
    let mut eof = false;
    loop {
        let mut chunk = [0u8; 4096];
        match stdin.lock().read(&mut chunk) {
            Ok(0) => {
                eof = true;
            }
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                last_data = Some(Instant::now());
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => {
                restore();
                return Err(format!("ds4-agent: read stdin: {error}"));
            }
        }
        if eof {
            break;
        }
        let timeout_ms = match last_data {
            None => -1,
            Some(at) => {
                let elapsed = at.elapsed();
                let quiet = Duration::from_millis(200);
                if elapsed >= quiet {
                    break;
                }
                i32::try_from((quiet - elapsed).as_millis()).unwrap_or(1)
            }
        };
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: [Category 8 — FFI] `pfd` is a stack-owned `pollfd` of length 1;
        // `fd` is still the live stdin descriptor. libc does not retain the pointer.
        let prc = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if prc < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            restore();
            return Err(format!("ds4-agent: poll: {err}"));
        }
        if prc == 0 && !buf.is_empty() {
            break;
        }
    }
    restore();
    if buf.is_empty() {
        return Ok(None);
    }
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

fn help_text(name: &str) -> String {
    format!(
        "Usage: {name} --non-interactive [-p TEXT] [options]\n\
         \n\
         Non-interactive ds4-agent shadow. Supports google_search, visit_page, read, more, list, search, write, edit, bash, bash_status, and bash_stop. \
         Use ./ds4-agent for the TUI or interactive KV. Coordinator --role/--layers/--listen are accepted.\n\
         \n\
         Options:\n\
           -m, --model FILE        GGUF model path. Default: ds4flash.gguf\n\
           -c, --ctx N            Context size. Default: 100000\n\
           -n, --tokens N         Max generated tokens. Default: 50000\n\
           -p, --prompt TEXT      One-shot prompt; omit to read repeated stdin prompts.\n\
           --non-interactive      Required shadow mode. Without -p, read stdin prompts.\n\
           -sys, --system TEXT    Extra system prompt.\n\
           --temp F               Sampling temperature. Default: 1\n\
           --top-p F              Nucleus probability. Default: 1\n\
           --min-p F              Min-p threshold. Default: 0.05\n\
           --seed N               Nonzero sampling seed.\n\
           --think                Normal thinking mode. Default.\n\
           --think-max            High effort when ctx >= 393216.\n\
           --nothink              Disable thinking.\n\
           --mtp FILE             Optional MTP support GGUF.\n\
           --mtp-draft N          Maximum MTP draft tokens. Default: 1\n\
           --mtp-margin F         MTP verifier margin. Default: 3\n\
           --quality              Prefer exact kernels where available.\n\
           --warm-weights         Touch mapped tensor pages before generation.\n\
            --power N              Target GPU duty cycle percentage, 1..100. Default: 100\n\
            --dir-steering-file FILE  Directional steering vector file.\n\
            --dir-steering-ffn F   FFN steering scale, -100..100. Default: 1 with a file.\n\
            --dir-steering-attn F  Attention steering scale, -100..100.\n\
            --backend NAME         metal, cuda, or cpu.\n\
           --metal, --cuda, --cpu Select backend explicitly.\n\
           -t, --threads N        CPU helper threads.\n\
           --chdir DIR            Change directory before model load.\n\
           --trace FILE           Write prompt, token, and DSML debug trace.\n\
           --role coordinator     Distributed coordinator; workers stay on ./ds4.\n\
           --layers A:B           Inclusive coordinator layer slice.\n\
           --listen HOST PORT     Coordinator TCP listen address.\n\
           -h, --help             Show this help.\n"
    )
}

fn built_in_tools_prompt() -> &'static str {
    BUILT_IN_TOOLS_PROMPT
}

fn observation_fits(
    vocab: &ds4_core::Vocab,
    transcript: &ds4_core::TokenBuffer,
    ctx: i32,
    observation: &[u8],
) -> Result<(bool, i32), String> {
    let mut projected = ds4_core::TokenBuffer::from_tokens(transcript.as_slice().to_vec());
    vocab
        .chat_append_message(&mut projected, "tool", observation)
        .map_err(|error| error.to_string())?;
    let projected_len = i32::try_from(projected.len()).unwrap_or(i32::MAX);
    Ok((
        projected_len + compact::TOOL_RESULT_RESERVE_TOKENS < ctx,
        projected_len,
    ))
}

fn fit_tool_observation(
    model: &ds4_core::Model,
    session: &mut ds4_core::Session<'_>,
    vocab: &ds4_core::Vocab,
    args: &AgentArgs,
    think: ds4_core::ChatThinkMode,
    transcript: &mut ds4_core::TokenBuffer,
    observation: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let (fits, _) = observation_fits(vocab, transcript, args.ctx, &observation)?;
    if fits {
        return Ok(observation);
    }
    let sys = build_system_transcript(vocab, args, think)?;
    compact::compact(
        model,
        session,
        vocab,
        args.ctx,
        &sys,
        transcript,
        "tool result would exceed context",
    )?;
    let (fits, projected_len) = observation_fits(vocab, transcript, args.ctx, &observation)?;
    if fits {
        Ok(observation)
    } else {
        Ok(compact::overflow_error(projected_len, args.ctx))
    }
}

fn format_datetime_context(when: &str) -> String {
    format!(
        "Current local date and time at session start: {when}. \
         Use this only when date or time matters."
    )
}

fn validate_clock_output(success: bool, stdout: &[u8], fallback: &str) -> String {
    if success {
        if let Ok(text) = std::str::from_utf8(stdout) {
            let text = text.trim_end_matches(['\r', '\n']);
            if !text.is_empty() {
                return text.to_string();
            }
        }
    }
    fallback.to_string()
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn project_output(chunks: &[&[u8]]) -> Result<Vec<u8>, &'static str> {
    let raw: Vec<u8> = chunks
        .iter()
        .flat_map(|chunk| chunk.iter().copied())
        .collect();
    if contains_bytes(&raw, DSML_OPEN) || contains_bytes(&raw, DSML_OPEN_MISSING_BAR) {
        return Err(TOOL_UNSUPPORTED_ERROR);
    }

    let mut out = Vec::with_capacity(raw.len() + 2);
    let mut post_think_gap = false;
    let mut wrote_visible = false;
    let mut last_output_newline = true;
    let mut i = 0;
    while i < raw.len() {
        let rem = &raw[i..];
        if rem.starts_with(THINK_OPEN) {
            post_think_gap = false;
            i += THINK_OPEN.len();
            continue;
        }
        if rem.starts_with(THINK_CLOSE) {
            if !last_output_newline {
                out.push(b'\n');
            }
            out.push(b'\n');
            last_output_newline = true;
            post_think_gap = true;
            i += THINK_CLOSE.len();
            continue;
        }

        let byte = raw[i];
        if post_think_gap && matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
            i += 1;
            continue;
        }
        post_think_gap = false;
        out.push(byte);
        wrote_visible = true;
        last_output_newline = byte == b'\n';
        i += 1;
    }

    if wrote_visible {
        if !last_output_newline {
            out.push(b'\n');
        }
        out.push(b'\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_requires_dsml_family() {
        assert!(agent_family(ds4_core::ModelFamily::DeepSeek4).is_ok());
        assert!(agent_family(ds4_core::ModelFamily::Inkling).is_err());
    }
    use std::path::PathBuf;
    use std::process::Command;

    fn oracle() -> PathBuf {
        if let Ok(path) = std::env::var("DS4_AGENT_C_ORACLE") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/agent_c_oracle")
    }

    fn c_out(args: &[&str]) -> Vec<u8> {
        let path = oracle();
        assert!(
            path.exists(),
            "build the C oracle first: make tests/parity/agent_c_oracle (missing {})",
            path.display()
        );
        let out = Command::new(path)
            .args(args)
            .output()
            .expect("run agent_c_oracle");
        assert!(
            out.status.success(),
            "agent_c_oracle failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    }

    fn unhex_line(bytes: &[u8]) -> Vec<u8> {
        let text = std::str::from_utf8(bytes).expect("oracle hex utf8").trim();
        assert_eq!(text.len() % 2, 0, "even oracle hex length");
        text.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let pair = std::str::from_utf8(pair).expect("hex pair utf8");
                u8::from_str_radix(pair, 16).expect("hex pair")
            })
            .collect()
    }

    fn argv(items: &[&str]) -> Vec<String> {
        std::iter::once("ds4-agent-rs".to_string())
            .chain(items.iter().map(|item| (*item).to_string()))
            .collect()
    }

    #[test]
    fn built_in_prompt_matches_current_c_bytes() {
        assert_eq!(
            built_in_tools_prompt().as_bytes(),
            unhex_line(&c_out(&["prompt"]))
        );
    }

    #[test]
    fn fixed_datetime_message_matches_current_c_bytes() {
        let when = "2026-08-25 12:34:56 KST";
        assert_eq!(
            format_datetime_context(when).as_bytes(),
            unhex_line(&c_out(&["datetime", when]))
        );
    }

    #[test]
    fn non_tty_projection_matches_c_tapes() {
        let cases: &[(bool, &[&str])] = &[
            (false, &["hello"]),
            (true, &["hello"]),
            (true, &["reasoning", "</thi", "nk>", "  \nanswer"]),
            (false, &["<thi", "nk>trace</think>\n\nanswer"]),
            (true, &["line with newline\n", "</think>", "\t \nanswer\n"]),
        ];
        for (thinking, chunks) in cases {
            let mut args = vec!["project", if *thinking { "1" } else { "0" }];
            args.extend_from_slice(chunks);
            let rust = project_output(&chunks.iter().map(|s| s.as_bytes()).collect::<Vec<_>>())
                .expect("no tool tape");
            assert_eq!(rust, unhex_line(&c_out(&args)), "case {args:?}");
        }
    }

    #[test]
    fn both_c_dsml_openers_are_rejected() {
        for marker in ["<｜DSML｜tool_calls>", "<DSML｜tool_calls>"] {
            assert_eq!(c_out(&["dsml", marker]), b"match=1 complete=1\n");
            assert!(project_output(&[b"prefix ", marker.as_bytes()]).is_err());

            let split = marker.len() / 2;
            let mut raw = b"prefix ".to_vec();
            raw.extend_from_slice(&marker.as_bytes()[..split]);
            assert!(!tool_marker_in_new_suffix(&raw, split));
            raw.extend_from_slice(&marker.as_bytes()[split..]);
            assert!(tool_marker_in_new_suffix(&raw, marker.len() - split));
        }
    }

    #[test]
    fn clock_output_validation_falls_back_on_failure_empty_or_invalid_utf8() {
        let fallback = "1770000000";
        assert_eq!(
            validate_clock_output(true, b"2026-08-25 12:34:56 KST\n", fallback),
            "2026-08-25 12:34:56 KST"
        );
        assert_eq!(validate_clock_output(false, b"ignored", fallback), fallback);
        assert_eq!(validate_clock_output(true, b"\r\n", fallback), fallback);
        assert_eq!(validate_clock_output(true, b"\xff\n", fallback), fallback);
    }

    #[test]
    fn narrow_cli_defaults_match_c_agent() {
        let parsed =
            parse_args(argv(&["--non-interactive", "-p", "hello"])).expect("minimal invocation");
        assert_eq!(parsed.model, "ds4flash.gguf");
        assert_eq!(parsed.ctx, 100000);
        assert_eq!(parsed.tokens, 50000);
        assert_eq!(
            parsed.system,
            "You are a helpful coding assistant running inside ds4-agent."
        );
        assert_eq!(parsed.temp, 1.0);
        assert_eq!(parsed.top_p, 1.0);
        assert_eq!(parsed.min_p, 0.05);
        assert_eq!(parsed.think, ds4_core::ChatThinkMode::Low);
        assert_eq!(parsed.prompt.as_deref(), Some("hello"));
    }

    #[test]
    fn high_thinking_uses_the_c_context_boundary() {
        let mut args = AgentArgs::default();
        args.think = ds4_core::ChatThinkMode::High;
        args.ctx = THINK_EFFORT_MIN_CONTEXT - 1;
        assert_eq!(effective_think(&args), ds4_core::ChatThinkMode::Low);
        args.ctx = THINK_EFFORT_MIN_CONTEXT;
        assert_eq!(effective_think(&args), ds4_core::ChatThinkMode::High);
    }

    #[test]
    fn parse_args_allows_interactive_when_tui_path_exists() {
        // Given: TUI/KV/approval landed (todos 32-34)
        // When: parse without --non-interactive
        let parsed = parse_args(argv(&["-p", "hello"])).expect("interactive allowed");
        // Then: Ok, flag stays off
        assert!(!parsed.non_interactive);
        assert_eq!(parsed.prompt.as_deref(), Some("hello"));
    }

    #[test]
    fn parse_args_keeps_non_interactive_and_rejects_unknown_flags() {
        let stdin_repeat = parse_args(argv(&["--non-interactive"])).unwrap();
        assert!(stdin_repeat.non_interactive);
        assert!(stdin_repeat.prompt.is_none());
        assert!(parse_args(argv(&[
            "--non-interactive",
            "-p",
            "hello",
            "--no-such-flag"
        ]))
        .unwrap_err()
        .contains("unknown option"));
        assert!(parse_args(argv(&["-p", "hello", "--no-such-flag"]))
            .unwrap_err()
            .contains("unknown option"));
    }

    #[test]
    fn parses_coordinator_flags_and_rejects_worker() {
        let parsed = parse_args(argv(&[
            "--non-interactive",
            "-p",
            "hello",
            "--role",
            "coordinator",
            "--layers",
            "0:19",
            "--listen",
            "127.0.0.1",
            "1234",
        ]))
        .unwrap();
        assert_eq!(parsed.dist.role, ds4_dist::Role::Coordinator);
        assert_eq!(parsed.dist.layers.start, 0);
        assert_eq!(parsed.dist.layers.end, 19);
        assert_eq!(parsed.dist.listen_host.as_deref(), Some("127.0.0.1"));
        assert_eq!(parsed.dist.listen_port, 1234);
        assert!(crate::distributed_config(&parsed.dist).is_some());

        let worker = parse_args(argv(&[
            "--non-interactive",
            "-p",
            "hello",
            "--role",
            "worker",
            "--layers",
            "20:output",
            "--coordinator",
            "127.0.0.1",
            "1234",
        ]))
        .unwrap_err();
        assert!(worker.contains("--role worker is a serving mode"));
    }

    #[test]
    fn parses_mtp_flags_and_follows_c_gates() {
        let parsed = parse_args(argv(&[
            "--non-interactive",
            "-p",
            "hello",
            "--temp",
            "0",
            "--mtp",
            "mtp.gguf",
            "--mtp-draft",
            "2",
            "--mtp-margin",
            "4.5",
        ]))
        .unwrap();
        assert_eq!(parsed.mtp.as_deref(), Some("mtp.gguf"));
        assert_eq!(parsed.mtp_draft, 2);
        assert_eq!(parsed.mtp_margin, 4.5);
        assert!(use_mtp_spec(
            parsed.temp,
            parsed.mtp.as_deref(),
            parsed.mtp_draft
        ));
        assert!(!use_mtp_spec(0.0, None, 2));
        assert!(!use_mtp_spec(0.0, Some("mtp.gguf"), 1));
        assert!(!use_mtp_spec(0.5, Some("mtp.gguf"), 2));
    }

    #[test]
    fn parses_power_quality_and_warm_weight_open_options() {
        let parsed = parse_args(argv(&[
            "--non-interactive",
            "-p",
            "hello",
            "--quality",
            "--warm-weights",
            "--power",
            "40",
        ]))
        .unwrap();
        assert!(parsed.quality);
        assert!(parsed.warm_weights);
        assert_eq!(parsed.power_percent, 40);
        let options = mtp_open_options(&parsed);
        assert!(options.contains(&ds4_core::ModelOpenOption::Quality));
        assert!(options.contains(&ds4_core::ModelOpenOption::WarmWeights));
        assert!(options.contains(&ds4_core::ModelOpenOption::PowerPercent(40)));
        assert_eq!(
            parse_args(argv(&["--non-interactive", "-p", "hello", "--power", "0"])).unwrap_err(),
            "invalid value for --power: 0"
        );
        assert_eq!(
            parse_args(argv(&[
                "--non-interactive",
                "-p",
                "hello",
                "--power",
                "101"
            ]))
            .unwrap_err(),
            "--power must be between 1 and 100"
        );
    }

    #[test]
    fn parses_dir_steering_open_options() {
        let file_only = parse_args(argv(&[
            "--non-interactive",
            "-p",
            "hello",
            "--dir-steering-file",
            "dirs.bin",
        ]))
        .unwrap();
        assert_eq!(file_only.steering_file.as_deref(), Some("dirs.bin"));
        assert_eq!(file_only.steering_ffn, 1.0);
        assert_eq!(file_only.steering_attn, 0.0);
        let options = mtp_open_options(&file_only);
        assert!(options.contains(&ds4_core::ModelOpenOption::SteeringFile("dirs.bin".into())));
        assert!(options.contains(&ds4_core::ModelOpenOption::SteeringFfn(1.0)));
        assert!(!options
            .iter()
            .any(|option| matches!(option, ds4_core::ModelOpenOption::SteeringAttn(_))));

        let with_attn = parse_args(argv(&[
            "--non-interactive",
            "-p",
            "hello",
            "--dir-steering-file",
            "dirs.bin",
            "--dir-steering-attn",
            "0.5",
        ]))
        .unwrap();
        assert_eq!(with_attn.steering_ffn, 0.0);
        assert_eq!(with_attn.steering_attn, 0.5);
        let options = mtp_open_options(&with_attn);
        assert!(options.contains(&ds4_core::ModelOpenOption::SteeringAttn(0.5)));
        assert!(!options
            .iter()
            .any(|option| matches!(option, ds4_core::ModelOpenOption::SteeringFfn(_))));

        assert_eq!(
            parse_args(argv(&[
                "--non-interactive",
                "-p",
                "hello",
                "--dir-steering-ffn",
                "101"
            ]))
            .unwrap_err(),
            "invalid value for --dir-steering-ffn: 101"
        );
        assert!(help_text("ds4-agent-rs").contains("--dir-steering-file FILE"));
    }

    #[test]
    fn parses_trace_path() {
        let parsed = parse_args(argv(&[
            "--non-interactive",
            "-p",
            "hello",
            "--trace",
            "/tmp/ds4-agent.trace",
        ]))
        .unwrap();
        assert_eq!(parsed.trace.as_deref(), Some("/tmp/ds4-agent.trace"));
        assert!(help_text("ds4-agent-rs").contains("--trace FILE"));
    }

    #[test]
    fn help_names_the_supported_tool_subset() {
        let help = help_text("ds4-agent-rs");
        assert!(
            help.contains("google_search, visit_page, read, more, list, search, write, edit, bash, bash_status, and bash_stop")
        );
        assert!(!help.contains("Tool calls are rejected"));
    }
}
