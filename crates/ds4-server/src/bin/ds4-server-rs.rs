//! Shadow HTTP host. GET surfaces are live; family decode uses the
//! native FFI when `-m` opens a model. Continuation registry is host-owned.
//! Incremental live DSML tool projection is host-owned.

use ds4_core::{
    attach_host_quote, caps_from_ident, identify_gguf, probe_dspark_sidecar, probe_model_artifact,
    probe_mtp_sidecar, probe_vision_sidecar, resolve_plan, Backend, DistributedConfig,
    DistributedRole, Distribution, EngineFacts, GgufFile, Identified, MaxSeqs, Model,
    ModelOpenOption, MtpMode, PrefixReuse, ServingCaps, ServingRequest, Support, Vocab,
    WeightSlice,
};
use ds4_server::cache_identity::CacheIdentity;
use ds4_server::expected_plan::ExpectedPlan;
use ds4_server::kv_cli::DiskKvArgs;
use ds4_server::parse::EosPolicy;
use ds4_server::{
    accept_loop, accept_loop_with_engine, accept_loop_with_engine_cont, dist_weight_slice,
    listen_if_allowed, model_id_from_gguf_path, run_assembled_worker, server_launch, ContLane,
    DistArgs, NativeDecode, ServerConfig, ServerLaunch, WORKER_REQUIRES_MODEL,
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
    let mut expected_plan: Option<ExpectedPlan> = None;
    let mut dist = DistArgs::default();
    let mut ctx_set = false;
    let mut v41_flags = V41Flags::default();
    let mut mem_budget_mb: Option<i32> = None;
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
            "--vision" | "--vision-gguf" => {
                // "--vision" is the external encoder for GLM-5.3/Step/Ling/MiMo;
                // "--vision-gguf" is the reference install.sh name for the V4.1
                // model's own tower sidecar.  Both ride one native open slot;
                // the native open picks the family from the loaded variant.
                let path = args.next().unwrap_or_else(|| usage());
                vision_path = Some(path.clone());
                model_options.push(ModelOpenOption::Vision(path));
            }
            "--ssd-streaming" => {
                serve_req.ssd_streaming = true;
                model_options.push(ModelOpenOption::SsdStreaming);
            }
            "--ssd-streaming-cold" => {
                serve_req.ssd_streaming_cold = true;
                model_options.push(ModelOpenOption::SsdStreamingCold);
            }
            "--ssd-streaming-cache-experts" => {
                let value = args
                    .next()
                    .unwrap_or_else(|| cli_error("--ssd-streaming-cache-experts requires a value"));
                let option = ModelOpenOption::ssd_cache(&value)
                    .unwrap_or_else(|error| cli_error(&error.message));
                match option {
                    ModelOpenOption::SsdCacheAuto => {}
                    ModelOpenOption::SsdCacheExperts(count) => {
                        serve_req.ssd_streaming_cache_experts = Some(count)
                    }
                    ModelOpenOption::SsdCacheBytes(bytes) => {
                        serve_req.ssd_streaming_cache_bytes = Some(bytes)
                    }
                    _ => unreachable!(),
                }
                model_options.push(option);
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
            "--expect-plan" => {
                if expected_plan.is_some() {
                    cli_error("--expect-plan may be supplied only once");
                }
                let path = args.next().unwrap_or_else(|| usage());
                expected_plan = Some(
                    ExpectedPlan::load(Path::new(&path)).unwrap_or_else(|error| cli_error(&error)),
                );
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
            "--mtp-draft" => {
                serve_req.mtp_draft = Some(positive_count(&arg, args.next()));
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
                ctx_set = true;
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
            "--ignore-eos-in-reasoning" => {
                if cfg.eos_policy == EosPolicy::Default {
                    cfg.eos_policy = EosPolicy::Reasoning;
                }
            }
            "--ignore-eos" => cfg.eos_policy = EosPolicy::Global,
            "--mem-floor-gb" => {
                let raw = args.next().unwrap_or_else(|| usage());
                cfg.apply_mem_floor_gb(&raw);
                serve_req.mem_floor_gb = cfg.mem_floor_gb;
            }
            // Unit Z: the engine's process red line (core_util.c:198-200);
            // install.sh boots with --mem-budget-mb 110000.  Armed before
            // the model open so the L1 gate sees it.
            "--mem-budget-mb" => {
                mem_budget_mb = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
            }
            // DeepSeek V4.1 (ds41) run surface — the engine's own spellings
            // (cli_opts.c:258-395).  They bind only when the opened model is
            // V4.1; the boot below refuses --ctx for V4.1 (the context comes
            // only from the metadata) and names the flags when the model is
            // not V4.1.  --zchain is not ported on this route.
            "--engram-dir" => {
                v41_flags.engram_dir = Some(args.next().unwrap_or_else(|| usage()));
            }
            "--v41-no-engram" => v41_flags.no_engram = true,
            "--no-dspark" => v41_flags.no_dspark = true,
            "--dspark" => v41_flags.dspark = true,
            "--dspark-verify" => {
                v41_flags.verify_k = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--no-graph" => v41_flags.no_graph = true,
            "--no-kv-reuse" => v41_flags.no_kv_reuse = true,
            "--emit-trace" => v41_flags.emit_trace = true,
            "--v41-prof" => v41_flags.prof = true,
            // Unit Z: the sidecar mount the deployment boots with
            // (install.sh: --zchain <domain sidecar> [--posttrain <dir>]).
            "--zchain" => {
                v41_flags.zchain = Some(args.next().unwrap_or_else(|| usage()));
            }
            "--posttrain" => {
                v41_flags.posttrain = Some(args.next().unwrap_or_else(|| usage()));
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
    if cfg.eos_policy != EosPolicy::Default && model_path.is_none() {
        cli_error("EOS policy requires --model");
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
    // A sliced boot maps only its own layers, so the quote prices that
    // interval instead of the whole sharded artifact.
    let weight_slice = dist_weight_slice(dist.opt.role, &dist.opt.layers);

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
    // A family whose context contract is its native maximum follows it
    // unless --ctx was given (GLM VQ: 1,048,576 tokens).
    if let Some(ctx) = ident
        .as_ref()
        .filter(|_| !ctx_set)
        .and_then(|id| ds4_core::native_ctx_default(caps_from_ident(id)))
    {
        cfg.ctx = ctx as i32;
        serve_req.ctx = cfg.ctx;
    }
    // V4.1 (ds41): the context comes only from the metadata
    // (core_validate_v41.c:51-53) — refuse --ctx before the model load, not
    // after it (the engine refuses it at parse time, cli_opts.c:285-293).
    if ctx_set
        && ident
            .as_ref()
            .is_some_and(|id| id.shape.variant == ds4_core::Variant::DeepSeek41Flash)
    {
        cli_error("ds4-server-rs: V4.1 context comes from the model metadata (deepseek4.context_length); --ctx is refused");
    }
    let v41_wire = ident
        .as_ref()
        .filter(|id| id.shape.variant == ds4_core::Variant::DeepSeek41Flash)
        .map(|id| {
            let path = model_path.as_deref().unwrap();
            let gguf = GgufFile::open(Path::new(path))
                .unwrap_or_else(|error| cli_error(&format!("V4.1 GGUF: {error}")));
            let mut wire = ds4_core::V41Wire::load(&gguf, &id.shape)
                .unwrap_or_else(|error| cli_error(&format!("V4.1 metadata: {error}")));
            let draft_block = wire
                .mtp_armed()
                .then_some(wire.mtp_block)
                .flatten()
                .filter(|block| *block > 0);
            let native_default = if std::env::var_os("DS41_NO_DSPARK").is_some() {
                MtpMode::Off
            } else {
                MtpMode::Auto
            };
            v41_request(&v41_flags, &mut serve_req, draft_block, native_default)
                .unwrap_or_else(|error| cli_error(&format!("ds4-server-rs: {error}")));
            if let Some(dir) = v41_flags.engram_dir.as_deref() {
                wire.apply_engram_dir(dir);
            }
            cfg.ctx = wire.ctx as i32;
            serve_req.ctx = cfg.ctx;
            wire
        });
    // Auto admission needs the requested workload before any cache is sized.
    let mut preflight_options = model_options.clone();
    preflight_options.push(ModelOpenOption::ServingBudget(serve_req.clone()));
    ds4_core::check_ssd_options(
        &preflight_options,
        ident.as_ref().map(|model| model.shape.family),
        backend,
        distributed_config(&dist.opt).as_ref(),
    )
    .unwrap_or_else(|error| cli_error(&error.message));
    let caps = ident.as_ref().map(caps_from_ident);
    if ident
        .as_ref()
        .is_some_and(|id| id.shape.family == ds4_core::ModelFamily::Glm53)
    {
        // Tensor/template validation is not qualification of the loaded model.
        facts.artifact_qualified = Some(false);
    }
    if serve_req.ssd_streaming {
        let identified = ident
            .as_ref()
            .unwrap_or_else(|| cli_error("SSD streaming model is not identified"));
        let inventory = ds4_core::TensorInventory::open(Path::new(model_path.as_deref().unwrap()))
            .unwrap_or_else(|error| cli_error(&format!("SSD inventory: {error}")));
        ds4_core::probe_ssd_quote(&mut facts, &serve_req, identified.shape, &inventory)
            .unwrap_or_else(|error| cli_error(&error.message));
    }
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
            if cfg.eos_policy != EosPolicy::Default && facts.artifact_ok == Some(true) {
                let family = ident
                    .as_ref()
                    .unwrap_or_else(|| cli_error("EOS policy requires an identified model"))
                    .shape
                    .family;
                // A GVQ1 release directory keeps its tables beside the parts.
                let vocab = Vocab::load_path(Path::new(path), family)
                    .unwrap_or_else(|error| cli_error(&format!("EOS policy vocab: {error}")));
                if vocab.eos_id < 0 {
                    cli_error("EOS policy requires a model EOS token");
                }
            }
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
        weight_slice,
        vision_path.is_some(),
        false,
    );
    let plan = resolve_plan(&serve_req, caps, &facts);
    plan.apply_env();
    cfg.adopt_plan(&plan);
    eprint!("{}", plan.report());
    if serve_req.check_config {
        if let Some(expected) = &expected_plan {
            expected
                .check_preflight(&plan)
                .unwrap_or_else(|error| cli_error(&error));
        }
        println!("{}", plan.to_json());
        std::process::exit(if plan.may_listen() { 0 } else { 2 });
    }
    if !plan.may_listen() {
        eprint!("{}", plan.report());
        cli_error("ds4-server-rs: serving plan rejected unsupported options");
    }
    // The engine allocates a speculative runtime only above the family's
    // draft minimum, so an unspecified draft takes the resolved one.
    if let Some(draft) = ds4_core::open_draft_tokens(
        plan.effective.mtp_mode,
        serve_req.mtp_draft,
        plan.effective.mtp_draft,
    ) {
        model_options.push(ModelOpenOption::MtpDraftTokens(draft));
    }
    model_options.push(ModelOpenOption::ServingBudget(serve_req.clone()));
    // A family whose caps refuse banks has no multi-sequence graph to open;
    // asking native for one only reports the refusal. Bonsai (qwen35) is the
    // resident case: its session is the trunk state itself.
    let banks_available = plan
        .caps
        .is_some_and(|caps| caps.bank_support != Support::None);
    let cont_width =
        if serve_req.max_seqs == MaxSeqs::Off || plan.uses_serial_mtp() || !banks_available {
            0
        } else {
            plan.effective.max_seqs as i32
        };

    let native_dist = distributed_config(&dist.opt);
    // Snapshot every artifact before native open, then recheck before enabling
    // disk reads. Equal prompt tokens alone do not establish equal model state.
    let cache_identity = model_path.as_deref().filter(|_| kv_store.is_some()).map(|path| {
        let sidecars: Vec<_> = [mtp_path.as_deref(), vision_path.as_deref(), dspark_path.as_deref()]
            .into_iter().flatten().map(Path::new).collect();
        let sources = engram_sources(
            v41_wire.as_ref().map_or(&[], |wire| wire.engram_table_path.as_slice()),
            &v41_flags,
        );
        let settings = format!("backend={backend:?};threads={n_threads};options={model_options:?};dist={native_dist:?};v41={v41_flags:?}");
        CacheIdentity::capture_with_sources(Path::new(path), &sidecars, &sources, &settings)
            .unwrap_or_else(|error| cli_error(&format!("disk KV identity: {error}")))
    });
    // --mem-budget-mb (unit Z) arms the engine's L1 load gate; it must be
    // set before the model open reads it (core_util.c:238-256).
    if let Some(mb) = mem_budget_mb {
        ds4_core::set_mem_budget_mb(mb);
    }
    let mut v41_route: Option<ds4_server::V41ServeRoute> = None;
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
                    if cfg.eos_policy != EosPolicy::Default && m.vocab().eos_id < 0 {
                        cli_error("EOS policy requires a model EOS token");
                    }
                    // V4.1 (ds41): no ds4_session exists for the family, so
                    // the serving path is the engine's push route
                    // (server_generate_v41.c) and the context comes only
                    // from the metadata (core_validate_v41.c:51-53) —
                    // --ctx is refused, never silently ignored.
                    if let Some(ctx) = m.v41_ctx() {
                        if ctx_set {
                            cli_error("ds4-server-rs: V4.1 context comes from the model metadata (deepseek4.context_length); --ctx is refused");
                        }
                        cfg.ctx = ctx as i32;
                        serve_req.ctx = cfg.ctx;
                        // Unit H: the engine's request-omitted parse defaults
                        // (thinking off unless asked, effort HIGH, min_p 0.0)
                        // apply to the engine's own model set -- exactly this
                        // branch's model.
                        cfg.engine_defaults = true;
                        v41_route = Some(resolve_v41_route(
                            path,
                            &v41_flags,
                            &plan,
                            vision_path.as_deref(),
                        ));
                        eprintln!(
                            "ds4-server-rs: V4.1 route: ctx={} MTP={} verify={} prefix_reuse={}",
                            cfg.ctx,
                            plan.effective.mtp_mode.as_str(),
                            v41_flags.verify_k,
                            plan.effective.prefix_reuse.as_str(),
                        );
                    } else if v41_flags.given() {
                        eprintln!(
                            "ds4-server-rs: warning: V4.1 flags given but the model is not V4.1; they are ignored"
                        );
                    }
                    // The GLM-5.3 scope: the artifact's own request-omitted
                    // values and level (chat_template.jinja:2-3,
                    // generation_config.json), each settable per deployment.
                    if ds4_server::syntax_for_model_id(m.model_id())
                        == ds4_server::ModelSyntax::Glm53
                    {
                        cfg.glm_values = true;
                        eprintln!("ds4-server-rs: {}", ds4_server::glm_values_report());
                    }
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
    let mut kv_store = if model.is_some() { kv_store } else { None };

    let lane = if let Some(ref model) = model {
        // What only the open engine knows. The refit re-resolves so a
        // fitted-down width or a refused lane cannot stay silently claimed.
        let mut opened = EngineFacts {
            drafter_shared: Some(model.drafter_shared()),
            mtp_loaded: mtp_path.is_some() || model.mtp().is_some(),
            vision_loaded: model_options
                .iter()
                .any(|opt| matches!(opt, ModelOpenOption::Vision(_))),
            ..facts.clone()
        };
        model
            .ssd_quote(&mut opened)
            .unwrap_or_else(|error| cli_error(&error.message));
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
                        weight_slice,
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
                        weight_slice,
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
        } else {
            // Serial HTTP and distributed workers also need confirmed IPC
            // ownership after open; their session graphs are still lazy.
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
                weight_slice,
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
        }
    } else {
        None
    };
    if let Some(expected) = &expected_plan {
        expected
            .check(cfg.serving_plan.as_ref().unwrap_or(&plan))
            .unwrap_or_else(|error| cli_error(&error));
    }
    if launch == ServerLaunch::Worker {
        let Some(ref model) = model else {
            cli_error(WORKER_REQUIRES_MODEL);
        };
        if serve_req.print_plan {
            // A worker never fits a lane, so this is the whole plan it has.
            print_plan(&cfg);
        }
        model.boot_prewarm();
        match run_assembled_worker(model, cfg.ctx, &dist.opt) {
            Ok(rc) => std::process::exit(rc),
            Err(e) => {
                eprintln!("ds4-server-rs: {e}");
                std::process::exit(1);
            }
        }
    }
    // Print what serves, not what was asked: the native fit can still take
    // banks, partial reuse and MTP away from the pre-open plan.
    if serve_req.print_plan {
        print_plan(&cfg);
    }
    if let Some(ref model) = model {
        model.boot_prewarm();
        // KU-2: the V4.1 route pays its asset opens (sidecar mount, engram
        // feed) inside request one unless they are paid here.  A failure must
        // not stop serving: the request path still opens them itself.
        if let Some(route) = &v41_route {
            if let Err(error) = model.v41_prewarm(
                &route.model_path,
                route.zchain.as_deref(),
                route.posttrain.as_deref(),
                route.engram_dir.as_deref(),
                route.no_engram,
            ) {
                eprintln!("ds4-server-rs: V4.1 prewarm skipped: {error}");
            }
        }
    }
    if let (Some(identity), Some(store)) = (cache_identity, kv_store.as_mut()) {
        let effective = &cfg.serving_plan.as_ref().unwrap_or(&plan).effective;
        let settings = format!(
            "ctx={};banks={};native={:?};schedule={}/{};mtp={:?}/{}/{:?}",
            effective.ctx,
            effective.max_seqs,
            effective.native_chunk,
            effective.sched_chunk,
            effective.sched_chunk_live,
            effective.mtp_mode,
            effective.mtp_weights,
            effective.mtp_draft
        );
        let digest = identity
            .finish(&settings)
            .unwrap_or_else(|error| cli_error(&format!("disk KV identity: {error}")));
        store.bind_identity(digest);
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        eprintln!("disk KV identity: local-file-stat-v1 {hex} (file metadata, not full-content attestation)");
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
        if let Some(route) = v41_route.take() {
            engine = engine.with_v41_route(route);
        }
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
    slice: Option<WeightSlice>,
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
        slice,
        vision,
        resident,
    );
}

fn cli_error(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

/// The DeepSeek V4.1 run switches the server accepts (the engine's
/// cli_opts.c:258-395 spellings).  They bind only when the opened model is
/// V4.1; otherwise the boot warns and ignores them.
#[derive(Default, Debug)]
struct V41Flags {
    engram_dir: Option<String>,
    no_engram: bool,
    no_dspark: bool,
    dspark: bool,
    verify_k: i32,
    no_graph: bool,
    emit_trace: bool,
    prof: bool,
    no_kv_reuse: bool,
    zchain: Option<String>,
    posttrain: Option<String>,
}

impl V41Flags {
    fn given(&self) -> bool {
        self.engram_dir.is_some()
            || self.no_engram
            || self.no_dspark
            || self.dspark
            || self.verify_k != 0
            || self.no_graph
            || self.emit_trace
            || self.prof
            || self.no_kv_reuse
            || self.zchain.is_some()
            || self.posttrain.is_some()
    }
}

fn resolve_v41_route(
    path: &str,
    flags: &V41Flags,
    plan: &ds4_core::ResolvedPlan,
    vision_path: Option<&str>,
) -> ds4_server::V41ServeRoute {
    ds4_server::V41ServeRoute {
        model_path: std::path::PathBuf::from(path),
        engram_dir: flags.engram_dir.clone(),
        no_engram: flags.no_engram,
        dspark: Some(match plan.effective.mtp_mode {
            MtpMode::Off => 0,
            MtpMode::Auto => 1,
            MtpMode::On => 2,
        }),
        graph: flags.no_graph.then_some(false),
        verify_k: flags.verify_k,
        emit_trace: flags.emit_trace,
        prof: flags.prof,
        no_kv_reuse: plan.effective.prefix_reuse == ds4_core::ReuseKind::None,
        zchain: flags.zchain.clone(),
        posttrain: flags.posttrain.clone(),
        // The V4.1 tower geometry + placeholder id, read from the sidecar the
        // probe already validated; the request expands the placeholder with it.
        vision: vision_path.map(|p| {
            ds4_core::v41_media::V41VisionMeta::read(p)
                .unwrap_or_else(|error| cli_error(&format!("--vision-gguf: {error}")))
        }),
    }
}

fn v41_request(
    flags: &V41Flags,
    req: &mut ServingRequest,
    draft_block: Option<u32>,
    native_default: MtpMode,
) -> Result<(), String> {
    if flags.no_dspark && flags.dspark {
        return Err("--no-dspark and --dspark are mutually exclusive".into());
    }
    if req.mtp_draft.is_some() {
        // The artifact fixes the draft block; verify_k controls verification,
        // not allocation or the number of tokens the towers draft.
        return Err("V4.1 --mtp-draft cannot change the artifact's draft block; use --dspark-verify to pin verification".into());
    }
    if req.mtp_path.is_some() {
        return Err("V4.1 uses embedded MTP towers; --mtp sidecars are unsupported".into());
    }
    if (flags.no_dspark && req.mtp_mode == MtpMode::On)
        || (flags.dspark && req.mtp_mode == MtpMode::Off)
    {
        return Err("V4.1 DSpark alias conflicts with --mtp-mode".into());
    }
    if flags.no_kv_reuse && !matches!(req.prefix_reuse, PrefixReuse::Auto | PrefixReuse::Off) {
        return Err("--no-kv-reuse conflicts with --prefix-reuse".into());
    }
    if flags.verify_k < 0 {
        return Err("--dspark-verify wants zero (automatic) or a positive count".into());
    }
    if flags.no_dspark {
        req.mtp_mode = MtpMode::Off;
    } else if flags.dspark {
        req.mtp_mode = MtpMode::On;
    }
    if req.mtp_mode == MtpMode::Auto && native_default == MtpMode::Off {
        req.mtp_mode = MtpMode::Off;
    }
    if flags.no_kv_reuse {
        req.prefix_reuse = PrefixReuse::Off;
    }
    if draft_block.is_none() {
        if req.mtp_mode == MtpMode::On {
            return Err("V4.1 artifact has no armed embedded MTP towers".into());
        }
        req.mtp_mode = MtpMode::Off;
    }
    if flags.verify_k == 0 {
        return Ok(());
    }
    if req.mtp_mode == MtpMode::Off {
        return Err("--dspark-verify requires enabled MTP".into());
    }
    if flags.verify_k as u32 > draft_block.unwrap_or(0) {
        return Err("--dspark-verify exceeds the artifact's draft block".into());
    }
    Ok(())
}

fn engram_sources<'a>(paths: &'a [String], flags: &V41Flags) -> Vec<&'a Path> {
    if flags.no_engram {
        return Vec::new();
    }
    paths.iter().map(Path::new).collect()
}

#[cfg(test)]
mod v41_controls {
    use super::*;
    use ds4_core::{ModelFamily, ReuseKind, Variant};

    fn plan(req: &ServingRequest) -> ds4_core::ResolvedPlan {
        let caps = ds4_core::serving_caps(ModelFamily::DeepSeek4, Variant::DeepSeek41Flash);
        resolve_plan(req, Some(caps), &EngineFacts::default())
    }

    #[test]
    fn engram_selection_identity() {
        let paths = vec![
            "/metadata/shard-a.safetensors".into(),
            "/override/shard-b.safetensors".into(),
        ];
        assert_eq!(
            engram_sources(&paths, &V41Flags::default()),
            vec![Path::new(&paths[0]), Path::new(&paths[1])]
        );
        let flags = V41Flags {
            no_engram: true,
            ..V41Flags::default()
        };
        assert!(engram_sources(&paths, &flags).is_empty());
        assert!(format!("{flags:?}").contains("no_engram: true"));
    }

    #[test]
    fn common_modes_reach_route() {
        for (mode, native) in [(MtpMode::Off, 0), (MtpMode::On, 2)] {
            let req = ServingRequest {
                mtp_mode: mode,
                ..ServingRequest::default()
            };
            let plan = plan(&req);
            assert!(!plan.has_errors());
            assert_eq!(
                resolve_v41_route("model.gguf", &V41Flags::default(), &plan, None).dspark,
                Some(native)
            );
        }
    }

    #[test]
    fn reuse_off_reaches_route() {
        let req = ServingRequest {
            prefix_reuse: PrefixReuse::Off,
            ..ServingRequest::default()
        };
        let plan = plan(&req);
        assert_eq!(plan.effective.prefix_reuse, ReuseKind::None);
        assert!(resolve_v41_route("model.gguf", &V41Flags::default(), &plan, None).no_kv_reuse);
    }

    #[test]
    fn aliases_match_report() {
        let mut req = ServingRequest::default();
        let flags = V41Flags {
            no_dspark: true,
            no_kv_reuse: true,
            ..V41Flags::default()
        };
        v41_request(&flags, &mut req, Some(3), MtpMode::Auto).unwrap();
        assert_eq!(req.mtp_mode, MtpMode::Off);
        assert_eq!(req.prefix_reuse, PrefixReuse::Off);
        let resolved = plan(&req);
        let route = resolve_v41_route("model.gguf", &flags, &resolved, None);
        assert_eq!(route.dspark, Some(0));
        assert!(route.no_kv_reuse);
        assert_eq!(resolved.effective.mtp_mode, MtpMode::Off);
    }

    #[test]
    fn defaults_keep_native_reuse() {
        let req = ServingRequest::default();
        let plan = plan(&req);
        assert_eq!(plan.effective.prefix_reuse, ReuseKind::Exact);
        assert_eq!(plan.effective.mtp_mode, MtpMode::Auto);
        assert_eq!(plan.effective.mtp_draft, None);
        assert_eq!(
            resolve_v41_route("model.gguf", &V41Flags::default(), &plan, None).dspark,
            Some(1)
        );
        assert!(!resolve_v41_route("model.gguf", &V41Flags::default(), &plan, None).no_kv_reuse);
    }

    #[test]
    fn reject_discarded_draft() {
        let mut req = ServingRequest {
            mtp_draft: Some(2),
            ..ServingRequest::default()
        };
        assert!(v41_request(&V41Flags::default(), &mut req, Some(3), MtpMode::Auto).is_err());
    }

    #[test]
    fn reject_conflicting_controls() {
        let flags = V41Flags {
            dspark: true,
            ..V41Flags::default()
        };
        let mut req = ServingRequest {
            mtp_mode: MtpMode::Off,
            ..ServingRequest::default()
        };
        assert!(v41_request(&flags, &mut req, Some(3), MtpMode::Auto).is_err());
        let flags = V41Flags {
            no_dspark: true,
            ..V41Flags::default()
        };
        req.mtp_mode = MtpMode::On;
        assert!(v41_request(&flags, &mut req, Some(3), MtpMode::Auto).is_err());
        let flags = V41Flags {
            no_kv_reuse: true,
            ..V41Flags::default()
        };
        req = ServingRequest {
            prefix_reuse: PrefixReuse::Exact,
            ..ServingRequest::default()
        };
        assert!(v41_request(&flags, &mut req, Some(3), MtpMode::Auto).is_err());
    }

    #[test]
    fn verify_is_bounded_and_active() {
        for verify_k in [-1, 4] {
            let flags = V41Flags {
                verify_k,
                ..V41Flags::default()
            };
            assert!(v41_request(
                &flags,
                &mut ServingRequest::default(),
                Some(3),
                MtpMode::Auto
            )
            .is_err());
        }
        let flags = V41Flags {
            no_dspark: true,
            verify_k: 1,
            ..V41Flags::default()
        };
        assert!(v41_request(
            &flags,
            &mut ServingRequest::default(),
            Some(3),
            MtpMode::Auto
        )
        .is_err());
    }

    #[test]
    fn exact_reuse_only() {
        let mut req = ServingRequest {
            prefix_reuse: PrefixReuse::Exact,
            ..ServingRequest::default()
        };
        assert!(!plan(&req).has_errors());
        req.prefix_reuse = PrefixReuse::Partial;
        assert!(plan(&req)
            .issues
            .iter()
            .any(|issue| issue.code == "partial_unsupported"));
    }

    #[test]
    fn verify_alias_preserves_pin() {
        let flags = V41Flags {
            dspark: true,
            verify_k: 3,
            ..V41Flags::default()
        };
        let mut req = ServingRequest::default();
        v41_request(&flags, &mut req, Some(3), MtpMode::Off).unwrap();
        let resolved = plan(&req);
        let route = resolve_v41_route("model.gguf", &flags, &resolved, None);
        assert_eq!(resolved.effective.mtp_mode, MtpMode::On);
        assert_eq!(route.dspark, Some(2));
        assert_eq!(route.verify_k, 3);
    }

    #[test]
    fn reuse_off_disables_disk() {
        let req = ServingRequest {
            prefix_reuse: PrefixReuse::Off,
            kv_disk_dir: Some("/tmp/v41-kv".into()),
            ..ServingRequest::default()
        };
        assert!(!plan(&req).effective.disk);
    }

    #[test]
    fn auto_honors_native_disable() {
        let mut req = ServingRequest::default();
        v41_request(&V41Flags::default(), &mut req, Some(3), MtpMode::Off).unwrap();
        assert_eq!(req.mtp_mode, MtpMode::Off);
        req.mtp_mode = MtpMode::On;
        v41_request(&V41Flags::default(), &mut req, Some(3), MtpMode::Off).unwrap();
        assert_eq!(req.mtp_mode, MtpMode::On);
    }

    #[test]
    fn force_needs_towers() {
        let mut req = ServingRequest {
            mtp_mode: MtpMode::On,
            ..ServingRequest::default()
        };
        assert!(v41_request(&V41Flags::default(), &mut req, None, MtpMode::Auto).is_err());
        req.mtp_mode = MtpMode::Auto;
        v41_request(&V41Flags::default(), &mut req, None, MtpMode::Auto).unwrap();
        assert_eq!(req.mtp_mode, MtpMode::Off);
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: ds4-server-rs [--version] [--host HOST] [--port PORT] [--listen HOST PORT] [--model-id ID] [-m GGUF] [--vision GGUF] [--vision-gguf GGUF] [--mtp GGUF] [--mtp-mode off|auto|on] [--backend cuda|cpu|metal|--cuda] [--tokens N|-n N] [-c N] [--max-seqs N|auto] [--prefix-reuse off|exact|partial|auto] [--prefill-chunk N] [--prefill-chunk-live N] [--native-chunk N] [--print-plan] [--check-config] [-t N] [--mtp-draft N] [--mtp-margin N] [--mem-floor-gb N] [--cors]\n\
         [--ssd-streaming] [--ssd-streaming-cache-experts auto|N|GB] [--ssd-streaming-cold]\n\
         [--ignore-eos-in-reasoning] [--ignore-eos]\n\
Disk KV: [--kv-disk-dir DIR] [--kv-disk-space-mb N] [--kv-disk-space 32G] [--kv-cache-min-tokens N]\n\
         [--kv-cache-cold-max-tokens N] [--kv-cache-continued-interval-tokens N]\n\
         [--kv-cache-boundary-trim-tokens N]\n\
         [--kv-cache-boundary-align-tokens N]\n\
         [--kv-cache-reject-different-quant]\n\
         V4.1 (ds41): --zchain DIR [--posttrain DIR] [--engram-dir DIR] [--v41-no-engram]\n\
         [--dspark] [--no-dspark] [--dspark-verify K] [--no-graph] [--no-kv-reuse] [--v41-prof]\n\
         [--emit-trace] [--mem-budget-mb N]\n\
         Distributed: [--role coordinator|worker] [--layers A:B] [--listen HOST PORT] [--coordinator HOST PORT]\n\
         [--dist-prefill-chunk N] [--dist-prefill-window N] [--dist-activation-bits N] [--dist-replay-check] [--debug]"
    );
    std::process::exit(2);
}
