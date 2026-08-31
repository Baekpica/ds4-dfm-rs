//! Shadow HTTP host. GET surfaces are live; family decode uses the
//! native FFI when `-m` opens a model. Continuation registry is host-owned.
//! Incremental live DSML tool projection is host-owned.

use ds4_core::{Backend, DistributedConfig, DistributedRole, Model, ModelOpenOption};
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
    let mut backend = Backend::Cuda;
    let mut n_threads = 0i32;
    let mut cont_width = std::env::var("DS4_SERVER_COALESCE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
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
            "--mtp-draft" => model_options.push(ModelOpenOption::MtpDraftTokens(
                args.next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage()),
            )),
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
                cont_width = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--cors" => cfg.cors = true,
            "--mem-floor-gb" => {
                let raw = args.next().unwrap_or_else(|| usage());
                cfg.apply_mem_floor_gb(&raw);
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("ds4-server-rs: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    kv.validate().unwrap_or_else(|error| cli_error(&error));
    dist.finish(&mut cfg.listen_host, &mut cfg.listen_port)
        .unwrap_or_else(|error| cli_error(&format!("ds4-server-rs: {error}")));
    if cfg.model_name == "ds4" {
        cfg.model_name = cfg.model_id.clone();
    }

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
                    None,
                    None,
                    config,
                    &model_options,
                ),
                None => {
                    Model::open_configured(path, backend, n_threads, true, None, &model_options)
                }
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
    let kv_store = if model.is_some() { kv.open() } else { None };

    let lane = if let Some(ref model) = model {
        if cont_width > 0 && backend == Backend::Cuda {
            match model.batch_ctx_fit(cfg.ctx, cont_width, cfg.ctx.saturating_mul(cont_width)) {
                Ok(batch) => {
                    eprintln!(
                        "ds4-server-rs: continuous lane ready (width={} seq_cap={})",
                        batch.max_seq(),
                        batch.seq_cap()
                    );
                    Some(ContLane::new(
                        batch,
                        model.vocab(),
                        model.model_id(),
                        model.routed_quant_bits(),
                        cfg.ctx,
                        model.token_eos(),
                    ))
                }
                Err(e) => {
                    eprintln!("ds4-server-rs: continuous lane unavailable ({e}); serial only");
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
        let mut engine = NativeDecode::new(model, cfg.ctx).with_vocab(model.vocab());
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
        "usage: ds4-server-rs [--version] [--host HOST] [--port PORT] [--listen HOST PORT] [--model-id ID] [-m GGUF] [--backend cuda|cpu|metal|--cuda] [--tokens N|-n N] [-c N] [-t N] [--mtp-draft N] [--mtp-margin N] [--mem-floor-gb N] [--cors]\n\
Disk KV: [--kv-disk-dir DIR] [--kv-disk-space-mb N] [--kv-cache-min-tokens N]\n\
         [--kv-cache-cold-max-tokens N] [--kv-cache-continued-interval-tokens N]\n\
         [--kv-cache-boundary-trim-tokens N]\n\
         [--kv-cache-boundary-align-tokens N]\n\
         [--kv-cache-reject-different-quant]\n\
         Distributed: [--role coordinator|worker] [--layers A:B] [--listen HOST PORT] [--coordinator HOST PORT]\n\
         [--dist-prefill-chunk N] [--dist-prefill-window N] [--dist-activation-bits N] [--dist-replay-check] [--debug]"
    );
    std::process::exit(2);
}
