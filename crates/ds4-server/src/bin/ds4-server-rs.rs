//! Shadow HTTP host. GET surfaces are live; family decode uses the
//! native FFI when `-m` opens a model. Continuation registry is host-owned.
//! Incremental live DSML tool projection is host-owned.

use ds4_core::{
    caps_from_ident, identify_gguf, resolve_plan, Backend, DistributedConfig, DistributedRole,
    EngineFacts, MaxSeqs, Model, ModelOpenOption, MtpMode, PrefixReuse, ServingRequest,
};
use ds4_server::kv_cli::DiskKvArgs;
use ds4_server::{
    accept_loop, accept_loop_with_engine, accept_loop_with_engine_cont, listen,
    model_id_from_gguf_path, run_assembled_worker, server_launch, ContLane, DistArgs, NativeDecode,
    ServerConfig, ServerLaunch, WORKER_REQUIRES_MODEL,
};

fn distributed_config(opt: &ds4_dist::Options) -> Option<DistributedConfig> {
    let role = match opt.role {
        ds4_dist::Role::None => return None,
        ds4_dist::Role::Coordinator => DistributedRole::Coordinator,
        ds4_dist::Role::Worker => DistributedRole::Worker,
    };
    Some(DistributedConfig {
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

fn main() {
    let mut cfg = ServerConfig::default();
    let mut model_path: Option<String> = None;
    let mut mtp_path: Option<String> = None;
    let mut backend = Backend::Cuda;
    let mut n_threads = 0i32;
    let mut serve_req = ServingRequest::from_env();
    let mut model_options = Vec::new();
    let mut kv = DiskKvArgs::default();
    let mut dist = DistArgs::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if dist
            .parse_arg(&arg, &mut args)
            .unwrap_or_else(|error| cli_error(&format!("ds4-server-rs: {error}")))
        {
            continue;
        }
        if kv
            .parse_arg(&arg, &mut args)
            .unwrap_or_else(|error| cli_error(&error))
        {
            continue;
        }
        match arg.as_str() {
            "--host" => cfg.listen_host = args.next().unwrap_or_else(|| usage()),
            "--port" => {
                cfg.listen_port = args
                    .next()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--model-id" => cfg.model_id = args.next().unwrap_or_else(|| usage()),
            "--model" | "-m" => {
                let path = args.next().unwrap_or_else(|| usage());
                if let Some(id) = model_id_from_gguf_path(&path) {
                    if cfg.model_id == "ds4" {
                        cfg.model_id = id;
                    }
                }
                model_path = Some(path);
            }
            "--vision" => model_options.push(ModelOpenOption::Vision(
                args.next().unwrap_or_else(|| usage()),
            )),
            "--mtp" => {
                let path = args.next().unwrap_or_else(|| usage());
                serve_req.mtp_path = Some(path.clone());
                mtp_path = Some(path);
            }
            "--mtp-mode" => {
                serve_req.mtp_mode = MtpMode::parse(&args.next().unwrap_or_else(|| usage()))
                    .unwrap_or_else(|e| {
                        cli_error(&e);
                    });
            }
            "--prefix-reuse" => {
                serve_req.prefix_reuse =
                    PrefixReuse::parse(&args.next().unwrap_or_else(|| usage()))
                        .unwrap_or_else(|e| cli_error(&e));
            }
            "--max-seqs" => {
                serve_req.max_seqs = MaxSeqs::parse(&args.next().unwrap_or_else(|| usage()))
                    .unwrap_or_else(|e| cli_error(&e));
            }
            "--prefill-chunk" => {
                serve_req.sched_chunk = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
            }
            "--prefill-chunk-live" => {
                serve_req.sched_chunk_live = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
            }
            "--print-plan" => serve_req.print_plan = true,
            "--check-config" => serve_req.check_config = true,
            "--backend" => {
                backend = match args.next().unwrap_or_else(|| usage()).as_str() {
                    "cuda" => Backend::Cuda,
                    "cpu" => Backend::Cpu,
                    "metal" => Backend::Metal,
                    other => {
                        eprintln!("ds4-server-rs: unknown backend {other}");
                        std::process::exit(2);
                    }
                };
            }
            "--cuda" => backend = Backend::Cuda,
            "--tokens" | "-n" => {
                cfg.default_tokens = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--mtp-draft" => {
                let n = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
                serve_req.mtp_draft = Some(n);
                model_options.push(ModelOpenOption::MtpDraftTokens(n));
            }
            "--mtp-margin" => model_options.push(ModelOpenOption::MtpMargin(
                args.next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage()),
            )),
            "--version" => {
                println!("ds4-server v{}", env!("CARGO_PKG_VERSION"));
                return;
            }
            "--no-update-check" => {}
            "-c" | "--ctx" => {
                cfg.ctx = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
                serve_req.ctx = cfg.ctx;
            }
            "-t" | "--threads" => {
                n_threads = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            // Hidden rust-shadow alias for DS4_SERVER_COALESCE_MAX.
            // Not a C flag; kept for rust-host-live scripts (e.g. --cont-width 1).
            "--cont-width" => {
                serve_req.max_seqs =
                    MaxSeqs::parse_coalesce(&args.next().unwrap_or_else(|| usage()))
                        .unwrap_or_else(|e| cli_error(&e));
            }
            "--cors" => cfg.cors = true,
            "--mem-floor-gb" => {
                let raw = args.next().unwrap_or_else(|| usage());
                cfg.apply_mem_floor_gb(&raw);
                serve_req.mem_floor_gb = cfg.mem_floor_gb;
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("ds4-server-rs: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    kv.validate().unwrap_or_else(|error| cli_error(&error));
    if mtp_path.is_some() && model_path.is_none() {
        cli_error("--mtp requires --model");
    }
    dist.finish(&mut cfg.listen_host, &mut cfg.listen_port)
        .unwrap_or_else(|error| cli_error(&format!("ds4-server-rs: {error}")));
    if cfg.model_name == "ds4" {
        cfg.model_name = cfg.model_id.clone();
    }
    serve_req.ctx = cfg.ctx;
    serve_req.mem_floor_gb = cfg.mem_floor_gb;
    serve_req.backend = backend;
    if let Some(dir) = kv.dir() {
        serve_req.kv_disk_dir = Some(dir.display().to_string());
    }
    if kv.space_mb() > 0 {
        serve_req.kv_disk_space_mb = Some(kv.space_mb());
    }
    serve_req.kv_min_tokens = Some(kv.min_tokens());

    let mut facts = EngineFacts::default();
    let mut kv_store = None;
    if kv.dir().is_some() {
        match kv.open() {
            Some(store) => {
                facts.disk_ready = Some(true);
                kv_store = Some(store);
            }
            None => facts.disk_ready = Some(false),
        }
    }
    if let Some(path) = mtp_path.as_deref() {
        facts.mtp_path_ok = Some(std::path::Path::new(path).is_file());
    }

    let caps = model_path
        .as_deref()
        .and_then(|path| identify_gguf(std::path::Path::new(path)).ok())
        .map(|id| caps_from_ident(&id));
    let plan = resolve_plan(&serve_req, caps, &facts);
    plan.apply_env();
    cfg.adopt_plan(&plan);
    eprint!("{}", plan.report());
    if serve_req.print_plan || serve_req.check_config {
        println!("{}", plan.to_json());
    }
    if serve_req.check_config {
        std::process::exit(if plan.has_errors() { 2 } else { 0 });
    }
    if plan.has_errors() {
        eprint!("{}", plan.report());
        cli_error("ds4-server-rs: serving plan rejected unsupported options");
    }
    let cont_width = if serve_req.max_seqs == MaxSeqs::Off {
        0
    } else {
        plan.effective.max_seqs as i32
    };

    let native_dist = distributed_config(&dist.opt);
    let launch = server_launch(dist.opt.role, model_path.is_some())
        .unwrap_or_else(|error| cli_error(&error));
    let model = match model_path.as_deref() {
        Some(path) => {
            let opened = match native_dist.as_ref() {
                Some(config) => Model::open_distributed_options(
                    path,
                    backend,
                    n_threads,
                    true,
                    mtp_path.as_deref(),
                    None,
                    config,
                    &model_options,
                ),
                None => Model::open_with_support_options(
                    path,
                    backend,
                    n_threads,
                    true,
                    mtp_path.as_deref(),
                    None,
                    &model_options,
                ),
            };
            match opened {
                Ok(m) => {
                    cfg.have_engine = true;
                    Some(m)
                }
                Err(e) => {
                    eprintln!("ds4-server-rs: open {path}: {e}");
                    std::process::exit(1);
                }
            }
        }
        None => None,
    };
    if launch == ServerLaunch::Worker {
        let Some(model) = model else {
            cli_error(WORKER_REQUIRES_MODEL);
        };
        model.boot_prewarm();
        match run_assembled_worker(&model, cfg.ctx, &dist.opt) {
            Ok(rc) => std::process::exit(rc),
            Err(e) => {
                eprintln!("ds4-server-rs: {e}");
                std::process::exit(1);
            }
        }
    }
    let kv_store = if model.is_some() { kv_store } else { None };

    let lane = if let Some(ref model) = model {
        // What only the open engine knows. The refit re-resolves so a
        // fitted-down width or a refused lane cannot stay silently claimed.
        let opened = EngineFacts {
            mtp_loaded: mtp_path.is_some() || model.mtp().is_some(),
            vision_loaded: model_options
                .iter()
                .any(|opt| matches!(opt, ModelOpenOption::Vision(_))),
            ..facts.clone()
        };
        if cont_width > 0 && backend == Backend::Cuda {
            match model.batch_ctx_fit(cfg.ctx, cont_width, cfg.ctx.saturating_mul(cont_width)) {
                Ok(batch) => {
                    eprintln!(
                        "ds4-server-rs: continuous lane ready (width={} seq_cap={})",
                        batch.max_seq(),
                        batch.seq_cap()
                    );
                    let facts = EngineFacts {
                        banks_fitted: Some(batch.max_seq() as u32),
                        seq_cap: Some(batch.seq_cap() as u32),
                        cont_lane: Some(true),
                        partial_reuse: Some(batch.supports_partial_reuse()),
                        ..opened
                    };
                    let fitted = resolve_plan(&serve_req, caps, &facts);
                    eprint!("{}", fitted.report());
                    if fitted.has_errors() {
                        cli_error("ds4-server-rs: fitted serving plan rejected");
                    }
                    fitted.apply_env();
                    cfg.adopt_plan(&fitted);
                    Some(
                        ContLane::new(
                            batch,
                            model.vocab(),
                            model.model_id(),
                            model.routed_quant_bits(),
                            cfg.ctx,
                            model.token_eos(),
                        )
                        .with_template(model.chat_template()),
                    )
                }
                Err(e) => {
                    eprintln!("ds4-server-rs: continuous lane unavailable ({e}); serial only");
                    let facts = EngineFacts {
                        banks_fitted: Some(1),
                        cont_lane: Some(false),
                        partial_reuse: Some(false),
                        ..opened
                    };
                    let serial = resolve_plan(&serve_req, caps, &facts);
                    eprint!("{}", serial.report());
                    if serial.has_errors() {
                        cli_error("ds4-server-rs: serial fallback plan rejected");
                    }
                    serial.apply_env();
                    cfg.adopt_plan(&serial);
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };
    if let Some(ref model) = model {
        model.boot_prewarm();
    }

    if !ds4_sys::install_stop_handlers() {
        eprintln!("ds4-server-rs: failed to install stop handlers");
        std::process::exit(1);
    }
    cfg.stop_requested = Some(ds4_sys::stop_requested);

    let listener = listen(&cfg).unwrap_or_else(|e| {
        eprintln!(
            "ds4-server-rs: listen {}:{}: {e}",
            cfg.listen_host, cfg.listen_port
        );
        std::process::exit(1);
    });
    eprintln!(
        "ds4-server-rs: listening on {}:{} model_id={} engine={} host_vocab={} (host continuation registry + incremental live DSML tool stream + corrective retry)",
        cfg.listen_host,
        cfg.listen_port,
        cfg.model_id,
        if cfg.have_engine { "open" } else { "none" },
        if model.is_some() { "yes" } else { "no" }
    );

    if let Some(ref model) = model {
        let mut engine = NativeDecode::new(model, cfg.ctx)
            .with_vocab(model.vocab())
            .with_prefix_reuse(plan.effective.prefix_reuse);
        if let Some(store) = kv_store {
            engine = engine.with_store(store);
        }
        match lane {
            Some(mut lane) => accept_loop_with_engine_cont(listener, cfg, &mut engine, &mut lane),
            None => accept_loop_with_engine(listener, cfg, &mut engine),
        }
    } else {
        accept_loop(listener, cfg);
    }
}

fn cli_error(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

fn usage() -> ! {
    eprintln!(
        "usage: ds4-server-rs [--version] [--host HOST] [--port PORT] [--listen HOST PORT] [--model-id ID] [-m GGUF] [--vision GGUF] [--mtp GGUF] [--mtp-mode off|auto|on] [--backend cuda|cpu|metal|--cuda] [--tokens N|-n N] [-c N] [--max-seqs N|auto] [--prefix-reuse off|exact|partial|auto] [--prefill-chunk N] [--prefill-chunk-live N] [--print-plan] [--check-config] [-t N] [--mtp-draft N] [--mtp-margin N] [--mem-floor-gb N] [--cors]\n\
Disk KV: [--kv-disk-dir DIR] [--kv-disk-space-mb N] [--kv-disk-space 32G] [--kv-cache-min-tokens N]\n\
         [--kv-cache-cold-max-tokens N] [--kv-cache-continued-interval-tokens N]\n\
         [--kv-cache-boundary-trim-tokens N]\n\
         [--kv-cache-boundary-align-tokens N]\n\
         [--kv-cache-reject-different-quant]\n\
         Distributed: [--role coordinator|worker] [--layers A:B] [--listen HOST PORT] [--coordinator HOST PORT]\n\
         [--dist-prefill-chunk N] [--dist-prefill-window N] [--dist-activation-bits N] [--dist-replay-check] [--debug]"
    );
    std::process::exit(2);
}
