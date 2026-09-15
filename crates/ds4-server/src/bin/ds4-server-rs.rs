//! Shadow HTTP host. GET surfaces are live; family decode uses the
//! native FFI when `-m` opens a model. Continuation registry is host-owned.
//! Incremental live DSML tool projection is host-owned.

use ds4_core::{
    attach_host_quote, caps_from_ident, identify_gguf, probe_dspark_sidecar, probe_model_artifact,
    probe_mtp_sidecar, probe_vision_sidecar, resolve_plan, Backend, DistributedConfig,
    DistributedRole, Distribution, EngineFacts, Identified, MaxSeqs, Model, ModelOpenOption,
    MtpMode, PrefixReuse, ServingCaps, ServingRequest,
};
use ds4_server::kv_cli::DiskKvArgs;
use ds4_server::{
    accept_loop, accept_loop_with_engine, accept_loop_with_engine_cont, listen_if_allowed,
    model_id_from_gguf_path, run_assembled_worker, server_launch, ContLane, DistArgs, NativeDecode,
    ServerConfig, ServerLaunch, WORKER_REQUIRES_MODEL,
};
use std::path::Path;

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
    let mut vision_path: Option<String> = None;
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
            "--vision" => {
                let path = args.next().unwrap_or_else(|| usage());
                vision_path = Some(path.clone());
                model_options.push(ModelOpenOption::Vision(path));
            }
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
                serve_req.sched_chunk = Some(positive_chunk(&arg, args.next()));
            }
            "--prefill-chunk-live" => {
                serve_req.sched_chunk_live = Some(positive_chunk(&arg, args.next()));
            }
            "--native-chunk" => {
                serve_req.native_chunk = Some(positive_chunk(&arg, args.next()));
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
                let n = positive_count(&arg, args.next());
                serve_req.mtp_draft = Some(n);
                model_options.push(ModelOpenOption::MtpDraftTokens(n));
            }
            "--mtp-margin" => {
                model_options.push(ModelOpenOption::MtpMargin(margin(&arg, args.next())))
            }
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
    serve_req.distribution = match distributed_config(&dist.opt) {
        Some(_) => Distribution::Sliced,
        None => Distribution::Single,
    };
    if let Some(dir) = kv.dir() {
        serve_req.kv_disk_dir = Some(dir.display().to_string());
    }
    if kv.space_mb() > 0 {
        serve_req.kv_disk_space_mb = Some(kv.space_mb());
    }
    serve_req.kv_min_tokens = Some(kv.min_tokens());

    let launch = server_launch(dist.opt.role, model_path.is_some())
        .unwrap_or_else(|error| cli_error(&error));
    launch.configure_serving(&mut serve_req);

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
    let ident = model_path
        .as_deref()
        .and_then(|path| identify_gguf(std::path::Path::new(path)).ok());
    let caps = ident.as_ref().map(caps_from_ident);
    let dist_probe = distributed_config(&dist.opt);
    if let Some(path) = vision_path.as_deref() {
        // The same rules the open applies: only a full GLM-5.3 or Step CUDA
        // model takes an encoder, and then the artifact itself is opened.
        if let Some(id) = ident.as_ref() {
            facts.vision_path_ok = Some(
                match probe_vision_sidecar(id.shape, backend, dist_probe.as_ref(), path) {
                    Ok(()) => true,
                    Err(error) => {
                        eprintln!("ds4-server-rs: --vision {path}: {error}");
                        false
                    }
                },
            );
        }
    }
    // The open still consumes this fallback, and only DeepSeek accepts a
    // drafter at all, so the check has to look at it.
    let dspark_path = std::env::var("DS4_DSPARK_MODEL")
        .ok()
        .filter(|path| !path.is_empty());
    if let (Some(id), Some(path)) = (ident.as_ref(), dspark_path.as_deref()) {
        facts.dspark_ok = Some(
            match probe_dspark_sidecar(id.shape, dist_probe.as_ref(), path) {
                Ok(()) => true,
                Err(error) => {
                    eprintln!("ds4-server-rs: DS4_DSPARK_MODEL {path}: {error}");
                    false
                }
            },
        );
    }
    if let Some(path) = mtp_path.as_deref() {
        // The same attach the open performs: family acceptance, sidecar
        // metadata, required tensors and layouts. A merely readable GGUF
        // would let `--check-config` exit 0 on an artifact that cannot load.
        if let Some(id) = ident.as_ref() {
            facts.mtp_path_ok = Some(
                match probe_mtp_sidecar(id.shape, dist_probe.as_ref(), path) {
                    Ok(()) => true,
                    Err(error) => {
                        eprintln!("ds4-server-rs: --mtp {path}: {error}");
                        false
                    }
                },
            );
        }
    }
    if serve_req.check_config {
        // Nothing else opens the model on this path, so the check has to do
        // the open's own pre-device validation itself.
        if let Some(path) = model_path.as_deref() {
            facts.artifact_ok = Some(match probe_model_artifact(path) {
                Ok(()) => true,
                Err(error) => {
                    eprintln!("ds4-server-rs: -m {path}: {error}");
                    false
                }
            });
        }
    }
    apply_host_quote(
        &mut facts,
        &serve_req,
        caps,
        ident.as_ref(),
        model_path.as_deref(),
        mtp_path.as_deref(),
        vision_path.as_deref(),
        dspark_path.as_deref(),
        vision_path.is_some(),
        false,
    );
    let plan = resolve_plan(&serve_req, caps, &facts);
    plan.apply_env();
    cfg.adopt_plan(&plan);
    eprint!("{}", plan.report());
    if serve_req.check_config {
        println!("{}", plan.to_json());
        std::process::exit(if plan.may_listen() { 0 } else { 2 });
    }
    if !plan.may_listen() {
        eprint!("{}", plan.report());
        cli_error("ds4-server-rs: serving plan rejected unsupported options");
    }
    // The engine allocates a speculative runtime only above the family's
    // draft minimum, so an unspecified draft takes the resolved one.
    if serve_req.mtp_draft.is_none() {
        if let Some(draft) = plan.effective.mtp_draft {
            model_options.push(ModelOpenOption::MtpDraftTokens(draft));
        }
    }
    let cont_width = if serve_req.max_seqs == MaxSeqs::Off {
        0
    } else {
        plan.effective.max_seqs as i32
    };

    let native_dist = distributed_config(&dist.opt);
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
        if serve_req.print_plan {
            // A worker never fits a lane, so this is the whole plan it has.
            print_plan(&cfg);
        }
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
            drafter_shared: Some(model.drafter_shared()),
            mtp_loaded: mtp_path.is_some() || model.mtp().is_some(),
            vision_loaded: model_options
                .iter()
                .any(|opt| matches!(opt, ModelOpenOption::Vision(_))),
            ..facts.clone()
        };
        if cont_width > 0 && backend == Backend::Cuda {
            match model.batch_ctx_fit(
                cfg.ctx,
                cont_width,
                plan.batch_max_total_tokens(cfg.ctx, cont_width),
            ) {
                Ok(batch) => {
                    eprintln!(
                        "ds4-server-rs: continuous lane ready (width={} seq_cap={})",
                        batch.max_seq(),
                        batch.seq_cap()
                    );
                    let mut facts = EngineFacts {
                        banks_fitted: Some(batch.max_seq() as u32),
                        seq_cap: Some(batch.seq_cap() as u32),
                        cont_lane: Some(true),
                        partial_reuse: Some(batch.supports_partial_reuse()),
                        ..opened
                    };
                    let vision = vision_path.is_some() || facts.vision_loaded;
                    apply_host_quote(
                        &mut facts,
                        &serve_req,
                        caps,
                        ident.as_ref(),
                        model_path.as_deref(),
                        mtp_path.as_deref(),
                        vision_path.as_deref(),
                        dspark_path.as_deref(),
                        vision,
                        true,
                    );
                    let fitted = resolve_plan(&serve_req, caps, &facts);
                    eprint!("{}", fitted.report());
                    if !fitted.may_listen() {
                        cli_error("ds4-server-rs: fitted serving plan rejected");
                    }
                    fitted.apply_env();
                    let fitted_reuse = fitted.effective.prefix_reuse;
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
                        .with_template(model.chat_template())
                        .with_prefix_reuse(fitted_reuse),
                    )
                }
                Err(e) => {
                    eprintln!("ds4-server-rs: continuous lane unavailable ({e}); serial only");
                    // cont_lane=false also limits resident credit to model
                    // mappings: the failed batch left no live runtime.
                    let mut facts = EngineFacts {
                        banks_fitted: Some(1),
                        cont_lane: Some(false),
                        partial_reuse: Some(false),
                        ..opened
                    };
                    let vision = vision_path.is_some() || facts.vision_loaded;
                    apply_host_quote(
                        &mut facts,
                        &serve_req,
                        caps,
                        ident.as_ref(),
                        model_path.as_deref(),
                        mtp_path.as_deref(),
                        vision_path.as_deref(),
                        dspark_path.as_deref(),
                        vision,
                        true,
                    );
                    let serial = resolve_plan(&serve_req, caps, &facts);
                    eprint!("{}", serial.report());
                    if !serial.may_listen() {
                        cli_error("ds4-server-rs: serial fallback plan rejected");
                    }
                    serial.apply_env();
                    cfg.adopt_plan(&serial);
                    None
                }
            }
        } else if dspark_path.is_some() {
            // A deferred drafter quote also needs its import result when
            // no batch fit runs; serial graph allocations are still lazy.
            let mut facts = EngineFacts {
                banks_fitted: Some(1),
                cont_lane: Some(false),
                partial_reuse: Some(false),
                ..opened
            };
            let vision = vision_path.is_some() || facts.vision_loaded;
            apply_host_quote(
                &mut facts,
                &serve_req,
                caps,
                ident.as_ref(),
                model_path.as_deref(),
                mtp_path.as_deref(),
                vision_path.as_deref(),
                dspark_path.as_deref(),
                vision,
                true,
            );
            let serial = resolve_plan(&serve_req, caps, &facts);
            eprint!("{}", serial.report());
            if !serial.may_listen() {
                cli_error("ds4-server-rs: opened serial plan rejected");
            }
            serial.apply_env();
            cfg.adopt_plan(&serial);
            None
        } else {
            None
        }
    } else {
        None
    };
    // Print what serves, not what was asked: the native fit can still take
    // banks, partial reuse and MTP away from the pre-open plan.
    if serve_req.print_plan {
        print_plan(&cfg);
    }
    if let Some(ref model) = model {
        model.boot_prewarm();
    }

    if !ds4_sys::install_stop_handlers() {
        eprintln!("ds4-server-rs: failed to install stop handlers");
        std::process::exit(1);
    }
    cfg.stop_requested = Some(ds4_sys::stop_requested);

    let serving = cfg.serving_plan.as_ref().unwrap_or(&plan);
    let listener = match listen_if_allowed(&cfg, serving) {
        Ok(Some(listener)) => listener,
        Ok(None) => cli_error("ds4-server-rs: serving plan rejected unsupported options"),
        Err(e) => {
            eprintln!(
                "ds4-server-rs: listen {}:{}: {e}",
                cfg.listen_host, cfg.listen_port
            );
            std::process::exit(1);
        }
    };
    eprintln!(
        "ds4-server-rs: listening on {}:{} model_id={} engine={} host_vocab={} (host continuation registry + incremental live DSML tool stream + corrective retry)",
        cfg.listen_host,
        cfg.listen_port,
        cfg.model_id,
        if cfg.have_engine { "open" } else { "none" },
        if model.is_some() { "yes" } else { "no" }
    );

    if let Some(ref model) = model {
        // The fitted plan, not the pre-open one: the refit can downgrade
        // reuse after the runtime says what it has.
        let reuse = cfg
            .serving_plan
            .as_ref()
            .map_or(plan.effective.prefix_reuse, |p| p.effective.prefix_reuse);
        let mut engine = NativeDecode::new(model, cfg.ctx)
            .with_vocab(model.vocab())
            .with_prefix_reuse(reuse);
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

/// A scheduler yield of zero is not a chunk, and the engine refuses a
/// nonpositive MTP draft: either would let `--check-config` approve a boot
/// failure.
fn positive_chunk(flag: &str, raw: Option<String>) -> u32 {
    u32::try_from(positive_count(flag, raw)).unwrap_or_else(|_| {
        cli_error(&format!(
            "ds4-server-rs: {flag} wants a positive token count"
        ))
    })
}

/// C `open_tuning` accepts 0 through 1000; anything else, NaN included,
/// aborts the open, so `--check-config` must not approve it.
fn margin(flag: &str, raw: Option<String>) -> f32 {
    match raw
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|m| (0.0..=1000.0).contains(m))
    {
        Some(m) => m,
        None => cli_error(&format!("ds4-server-rs: {flag} wants 0 to 1000")),
    }
}

fn positive_count(flag: &str, raw: Option<String>) -> i32 {
    match raw.and_then(|v| v.parse::<i32>().ok()).filter(|n| *n > 0) {
        Some(n) => n,
        None => cli_error(&format!("ds4-server-rs: {flag} wants a positive count")),
    }
}

fn print_plan(cfg: &ServerConfig) {
    if let Some(plan) = cfg.serving_plan.as_ref() {
        println!("{}", plan.to_json());
    }
}

fn apply_host_quote(
    facts: &mut EngineFacts,
    req: &ServingRequest,
    caps: Option<ServingCaps>,
    ident: Option<&Identified>,
    model_path: Option<&str>,
    mtp_path: Option<&str>,
    vision_path: Option<&str>,
    dspark_path: Option<&str>,
    vision: bool,
    resident: bool,
) {
    let Some(caps) = caps else {
        return;
    };
    attach_host_quote(
        facts,
        req,
        caps,
        ident.map(|id| id.shape),
        model_path.map(Path::new),
        mtp_path.map(Path::new),
        vision_path.map(Path::new),
        dspark_path.map(Path::new),
        ident.map(|id| id.split_count).unwrap_or(1),
        vision,
        resident,
    );
}

fn cli_error(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

fn usage() -> ! {
    eprintln!(
        "usage: ds4-server-rs [--version] [--host HOST] [--port PORT] [--listen HOST PORT] [--model-id ID] [-m GGUF] [--vision GGUF] [--mtp GGUF] [--mtp-mode off|auto|on] [--backend cuda|cpu|metal|--cuda] [--tokens N|-n N] [-c N] [--max-seqs N|auto] [--prefix-reuse off|exact|partial|auto] [--prefill-chunk N] [--prefill-chunk-live N] [--native-chunk N] [--print-plan] [--check-config] [-t N] [--mtp-draft N] [--mtp-margin N] [--mem-floor-gb N] [--cors]\n\
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
