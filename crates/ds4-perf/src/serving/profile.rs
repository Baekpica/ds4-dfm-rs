use super::*;
use crate::experiment::InputFile;
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf, process::Command};

const MIN_REPEATS: u32 = 3;
const SWITCHES: &[&str] = &[
    "--cuda",
    "--cors",
    "--no-update-check",
    "--print-plan",
    "--kv-cache-reject-different-quant",
];

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    name: String,
    runs: Vec<PathBuf>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    workload: PathBuf,
    candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Launch {
    executable: PathBuf,
    executable_sha256: String,
    arguments: Vec<String>,
    cwd: PathBuf,
    environment: BTreeMap<String, String>,
    source: Value,
    device: Value,
    expected_clock_range_mhz: Value,
    serving_plan: Value,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Qualified {
    name: String,
    worst_latency_fraction: f64,
    samples_per_workload: u32,
    launch: Launch,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    selection: Selection,
    qualified: bool,
    scope: String,
    selected: Option<Qualified>,
    rejected: BTreeMap<String, String>,
}
impl Payload for Profile {
    const KIND: &'static str = "serving-profile";
    fn validate(&self) -> Result<(), String> {
        if self.qualified != self.selected.is_some()
            || self.scope != "declared-serving-workload-only"
            || self.selected.as_ref().is_some_and(|s| {
                s.samples_per_workload < MIN_REPEATS
                    || !s.worst_latency_fraction.is_finite()
                    || !(0.0..=1.0).contains(&s.worst_latency_fraction)
            })
        {
            return Err("invalid serving profile qualification".into());
        }
        Ok(())
    }
}

fn json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > MAX_MANIFEST_BYTES {
        return Err("profile input exceeds 16 MiB".into());
    }
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn json_verified<T: serde::de::DeserializeOwned>(
    path: &Path,
    checked: &mut artifact::Verification,
) -> Result<T, String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > MAX_MANIFEST_BYTES {
        return Err("profile input exceeds 16 MiB".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err("profile input exceeds 16 MiB".into());
    }
    checked.verify(path, &artifact::hash_bytes(&bytes))?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn selection(path: &Path, checked: &mut artifact::Verification) -> Result<Selection, String> {
    let mut selection: Selection = json_verified(path, checked)?;
    let mut names = BTreeSet::new();
    if selection.candidates.is_empty() || selection.candidates.len() > 64 {
        return Err("selection requires 1..64 named candidates".into());
    }
    for candidate in &selection.candidates {
        if candidate.name.is_empty()
            || !names.insert(&candidate.name)
            || candidate.runs.is_empty()
            || candidate.runs.len() > 100
        {
            return Err("each unique candidate needs at least one run (maximum 100)".into());
        }
    }
    let base = path.parent().unwrap_or(Path::new("."));
    selection.workload = base
        .join(&selection.workload)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    for candidate in &mut selection.candidates {
        for run in &mut candidate.runs {
            *run = base.join(&*run).canonicalize().map_err(|e| e.to_string())?;
        }
        if candidate.runs.iter().collect::<BTreeSet<_>>().len() != candidate.runs.len() {
            return Err("duplicate run cannot supply additional repeats".into());
        }
    }
    Ok(selection)
}

fn workload(path: &Path, checked: &mut artifact::Verification) -> Result<Workload, String> {
    let mut workload: Workload = json_verified(path, checked)?;
    workload.validate()?;
    if let Some(seed) = &mut workload.restart_from {
        *seed = path
            .parent()
            .unwrap_or(Path::new("."))
            .join(&*seed)
            .canonicalize()
            .map_err(|e| e.to_string())?;
    }
    if workload.expected_clock_range_mhz.is_none() {
        return Err("profile requires an expected clock range".into());
    }
    for input in workload.inputs.values_mut() {
        let consumed = path.parent().unwrap_or(Path::new(".")).join(&input.path);
        checked
            .verify(&consumed, &input.sha256)
            .map_err(|error| format!("profile workload input changed: {error}"))?;
        input.path = consumed.canonicalize().map_err(|e| e.to_string())?;
    }
    Ok(workload)
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| format!("profile lacks {key}"))
}

// Only direct server launches are replayed. Unknown or duplicate options cannot
// introduce an untracked config file, wrapper, or later option override.
fn options(args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut result = BTreeMap::new();
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let key = match flag.as_str() {
            "-m" => "--model",
            "-c" => "--ctx",
            "-n" => "--tokens",
            "-t" => "--threads",
            "--cont-width" => "--max-seqs",
            key => key,
        };
        let value = match key {
            key if SWITCHES.contains(&key) => "true".into(),
            "--host"
            | "--port"
            | "--model-id"
            | "--model"
            | "--vision"
            | "--mtp"
            | "--mtp-mode"
            | "--prefix-reuse"
            | "--max-seqs"
            | "--prefill-chunk"
            | "--prefill-chunk-live"
            | "--native-chunk"
            | "--backend"
            | "--tokens"
            | "--mtp-draft"
            | "--mtp-margin"
            | "--ctx"
            | "--threads"
            | "--mem-floor-gb"
            | "--kv-disk-dir"
            | "--kv-disk-space-mb"
            | "--kv-disk-space"
            | "--kv-cache-min-tokens"
            | "--kv-cache-cold-max-tokens"
            | "--kv-cache-continued-interval-tokens"
            | "--kv-cache-boundary-trim-tokens"
            | "--kv-cache-boundary-align-tokens"
            | "--expect-plan" => args.next().ok_or("missing server option value")?.clone(),
            _ => return Err(format!("profile does not cover server option {flag}")),
        };
        if result.insert(key.into(), value).is_some() {
            return Err(format!("duplicate server option {key}"));
        }
    }
    Ok(result)
}

fn input_scope(
    launch: &Launch,
    workload: &Workload,
    manifest: &Path,
    checked: &mut artifact::Verification,
) -> Result<(), String> {
    let options = options(&launch.arguments)?;
    let count = options
        .get("--max-seqs")
        .and_then(|v| v.parse::<u64>().ok());
    if count.is_none() || count != launch.serving_plan["effective"]["max_seqs"].as_u64() {
        return Err(
            "profile needs explicit --max-seqs matching effective banks; auto cannot pin a launch"
                .into(),
        );
    }
    let model = options
        .get("--model")
        .ok_or("profile needs a direct server --model argument")?;
    let mut command: Vec<OsString> = vec![
        "ds4-bench".into(),
        "-m".into(),
        launch.cwd.join(model).into(),
        "--prompt-file".into(),
        manifest.into(),
    ];
    if let Some(mtp) = options.get("--mtp") {
        command.extend(["--mtp".into(), launch.cwd.join(mtp).into()]);
    }
    let mut files = workload.inputs.clone();
    files.insert(
        "prompt".into(),
        InputFile {
            path: manifest.into(),
            sha256: artifact::hash(manifest)?,
        },
    );
    let benchmark = crate::experiment::Workload {
        protocol: "ds4-bench-v1".into(),
        name: workload.name.clone(),
        family: workload.family.clone(),
        files,
        shape: BTreeMap::new(),
        cache_state: "serving".into(),
    };
    let mut environment: BTreeMap<OsString, OsString> = launch
        .environment
        .iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
    for key in [
        "DS4_QWEN_PLE_DIR",
        "DS4_CUDA_WEIGHT_IPC_MANIFEST",
        "DS4_WEIGHT_SERVER",
    ] {
        if let Some(value) = environment.get_mut(std::ffi::OsStr::new(key)) {
            *value = launch.cwd.join(&*value).into();
        }
    }
    for input in benchmark.files.values() {
        checked.verify(&input.path, &input.sha256)?;
    }
    benchmark.verify_scope_checked(&command, &environment, checked)?;
    let model = launch.cwd.join(model);
    let parent = model.parent().ok_or("model has no parent directory")?;
    // Either adjacent file changes the template loader's behavior, including
    // when it first appears after the original capture.
    for name in ["tokenizer_config.json", "chat_template.jinja"] {
        let file = parent.join(name);
        if file.exists() {
            let canonical = file.canonicalize().map_err(|e| e.to_string())?;
            let input = workload
                .inputs
                .values()
                .find(|input| input.path == canonical)
                .ok_or_else(|| format!("workload lacks template input: {name}"))?;
            checked.verify(&file, &input.sha256)?;
        }
    }
    if let Some(vision) = options.get("--vision") {
        let consumed = launch.cwd.join(vision);
        let path = consumed.canonicalize().map_err(|e| e.to_string())?;
        let input = workload
            .inputs
            .values()
            .find(|input| input.path == path)
            .ok_or("workload lacks vision input")?;
        checked.verify(&consumed, &input.sha256)?;
        let mut vision_scope = benchmark.clone();
        vision_scope.files.insert(
            "model".into(),
            InputFile {
                path: path.clone(),
                sha256: input.sha256.clone(),
            },
        );
        vision_scope.verify_scope_checked(
            &[
                "ds4-bench".into(),
                "-m".into(),
                consumed.into(),
                "--prompt-file".into(),
                manifest.into(),
            ],
            &BTreeMap::new(),
            checked,
        )?;
    }
    if launch
        .environment
        .get("LD_PRELOAD")
        .is_some_and(|v| !v.is_empty())
    {
        return Err("preloaded libraries require a separate profile contract".into());
    }
    // External owners can outlive or replace their manifest. Their process and
    // device allocation identity are outside this direct-server profile.
    for key in ["DS4_WEIGHT_SERVER", "DS4_CUDA_WEIGHT_IPC_MANIFEST"] {
        if launch.environment.get(key).is_some_and(|v| !v.is_empty()) {
            return Err("external owner identity is not covered by serving profiles".into());
        }
    }
    Ok(())
}

fn device(sample: &Value) -> Result<Value, String> {
    for key in ["uuid", "name", "driver"] {
        text(sample, key)?;
    }
    if sample["temperature_c"].as_u64().is_none()
        || sample["device_ordinal"].as_u64().is_none()
        || !sample["unavailable_reason"].is_null()
    {
        return Err("profile lacks complete clock/temperature/device evidence".into());
    }
    Ok(
        serde_json::json!({"uuid":sample["uuid"], "name":sample["name"], "driver":sample["driver"], "device_ordinal":sample["device_ordinal"]}),
    )
}

fn launch(
    run: &review::Run,
    workload: &Workload,
    manifest: &Path,
    checked: &mut artifact::Verification,
) -> Result<Launch, String> {
    let data = run.data()?;
    let identity: Value = run.json(Path::new("identity.json"))?;
    if identity != serde_json::to_value(&data.identity).map_err(|e| e.to_string())?
        || identity["inputs"]
            != serde_json::to_value(&workload.inputs).map_err(|e| e.to_string())?
        || identity["expected_clock_range_mhz"]
            != serde_json::to_value(&workload.expected_clock_range_mhz)
                .map_err(|e| e.to_string())?
    {
        return Err("raw identity differs from workload or summary".into());
    }
    let server = &identity["server"];
    if server["unreviewed_environment"]
        .as_array()
        .is_none_or(|v| !v.is_empty())
    {
        return Err("unreviewed server environment cannot qualify".into());
    }
    let argv = run.bytes(Path::new("server.argv"))?;
    if artifact::hash_bytes(&argv) != text(server, "argv_sha256")? {
        return Err("server argv hash differs".into());
    }
    let mut args = argv
        .split(|v| *v == 0)
        .filter(|v| !v.is_empty())
        .map(|v| String::from_utf8(v.into()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if args.is_empty() {
        return Err("empty server argv".into());
    }
    args.remove(0);
    let mut launch = Launch {
        executable: text(server, "executable")?.into(),
        executable_sha256: text(server, "executable_sha256")?.into(),
        arguments: args,
        cwd: text(server, "cwd")?.into(),
        environment: serde_json::from_value(server["environment"].clone())
            .map_err(|e| e.to_string())?,
        source: server["source"].clone(),
        device: device(
            &serde_json::to_value(data.gpu_samples.first().ok_or("missing device evidence")?)
                .map_err(|e| e.to_string())?,
        )?,
        // Admission quotes include transient available memory. Only resolved
        // options/capabilities define the launch; each run still checks its
        // complete raw plan at every request boundary.
        serving_plan: ["family", "requested", "effective", "qualified", "controls"]
            .into_iter()
            .map(|key| (key.to_owned(), data.serving_plan[key].clone()))
            .collect::<serde_json::Map<_, _>>()
            .into(),
        expected_clock_range_mhz: identity["expected_clock_range_mhz"].clone(),
    };
    launch.serving_plan["quote"] = post_open_quote(&data.serving_plan);
    normalize_guard(&mut launch, workload)?;
    verify_static(&launch)?;
    input_scope(&launch, workload, manifest, checked)?;
    Ok(launch)
}

fn normalize_guard(launch: &mut Launch, workload: &Workload) -> Result<(), String> {
    let options = options(&launch.arguments)?;
    let Some(path) = options.get("--expect-plan") else {
        return Ok(());
    };
    let path = launch
        .cwd
        .join(path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let digest = artifact::hash(&path)?;
    if !workload
        .inputs
        .values()
        .any(|input| input.path == path && input.sha256 == digest)
        || json::<Value>(&path)? != expected_plan(launch)
    {
        return Err("recorded --expect-plan is not pinned to the measured serving plan".into());
    }
    // A prior profile's guard is evidence, not a reusable output path. The new
    // application injects its own guard after validating the launch.
    let mut index = 0;
    while index < launch.arguments.len() {
        if launch.arguments[index] == "--expect-plan" {
            launch.arguments.drain(index..index + 2);
            break;
        }
        index += if SWITCHES.contains(&launch.arguments[index].as_str()) {
            1
        } else {
            2
        };
    }
    Ok(())
}

fn verify_static(launch: &Launch) -> Result<(), String> {
    if artifact::hash(&launch.executable)? != launch.executable_sha256 {
        return Err("profile server executable changed".into());
    }
    let source_root = text(&launch.source, "root")?;
    if context::source_snapshot(Path::new(source_root))? != launch.source {
        return Err("profile runtime source changed".into());
    }
    if !launch.cwd.is_dir() {
        return Err("profile working directory disappeared".into());
    }
    Ok(())
}

fn repeat_observations(
    run: &review::Run,
    root: &Path,
    workload: &Workload,
) -> Result<Vec<(String, String)>, String> {
    let mut paths: Vec<_> = (0..workload.cases.len())
        .map(|index| format!("case-{index:03}"))
        .collect();
    for index in 0..workload.overlaps.len() {
        for stream in ["decode", "prefill"] {
            paths.push(format!("overlap-{index:03}/{stream}"));
        }
    }
    paths
        .into_iter()
        .map(|path| {
            let events: Vec<Event> = run.json(&root.join(&path).join("events.json"))?;
            let bytes = serde_json::to_vec(&events).map_err(|e| e.to_string())?;
            Ok((path, artifact::hash_bytes(&bytes)))
        })
        .collect()
}

fn qualify(
    candidate: &Candidate,
    selection: &Selection,
    workload: &Workload,
    checked: &mut artifact::Verification,
) -> Result<Qualified, String> {
    let mut selected = None;
    let mut count = 0u32;
    let mut worst = 0f64;
    let mut hashes = BTreeSet::new();
    let mut observations = BTreeSet::new();
    for file in &candidate.runs {
        if !hashes.insert(artifact::hash(file)?) {
            return Err("copied evidence cannot supply additional repeats".into());
        }
        let run = review::Run::load_verified(file, checked)?;
        let data = run.data()?;
        restart::review(&run, workload, checked)?;
        if !data.passed
            || !data.context_failures.is_empty()
            || data.requested_repeats == 0
            || data.name != workload.name
            || data.family != workload.family
            || run.bytes(Path::new("workload.json"))?
                != fs::read(&selection.workload).map_err(|e| e.to_string())?
            || data.cases.len() != workload.cases.len() * data.requested_repeats as usize
            || data.overlaps.len() != workload.overlaps.len() * data.requested_repeats as usize
            || data.gpu_samples.len() != 2 * data.requested_repeats as usize
            || data.gpu_windows.len() != data.requested_repeats as usize
        {
            return Err("incomplete workload, repeat, clock or correctness evidence".into());
        }
        let launch = launch(&run, workload, &selection.workload, checked)?;
        if selected
            .as_ref()
            .is_some_and(|previous| previous != &launch)
        {
            return Err("candidate runtime identity or effective plan differs across runs".into());
        }
        for repeat in 0..data.requested_repeats {
            let root = if data.requested_repeats == 1 {
                PathBuf::new()
            } else {
                PathBuf::from(format!("repeat-{repeat:03}"))
            };
            for observation in repeat_observations(&run, &root, workload)? {
                if !observations.insert(observation) {
                    return Err(
                        "repeated runtime observations cannot supply independent samples".into(),
                    );
                }
            }
            for (offset, label) in ["gpu-before", "gpu-after"].iter().enumerate() {
                let sample = &data.gpu_samples[repeat as usize * 2 + offset];
                let raw: Value = run.json(&root.join(format!("{label}.json")))?;
                sample.verify_raw(&run.bytes(&root.join(format!("{label}.stdout")))?)?;
                if raw != serde_json::to_value(sample).map_err(|e| e.to_string())?
                    || device(&raw)? != launch.device
                    || !sample
                        .failures(workload.expected_clock_range_mhz.as_ref())
                        .is_empty()
                {
                    return Err("raw clock/device evidence fails contract".into());
                }
            }
            let cases_start = repeat as usize * workload.cases.len();
            let overlaps_start = repeat as usize * workload.overlaps.len();
            let span_ms = data.cases[cases_start..cases_start + workload.cases.len()]
                .iter()
                .map(|case| case.total_ms)
                .sum::<f64>()
                + data.overlaps[overlaps_start..overlaps_start + workload.overlaps.len()]
                    .iter()
                    .map(overlap::Observation::span_ms)
                    .sum::<f64>();
            data.gpu_windows[repeat as usize].review(
                &run,
                &root,
                workload.expected_clock_range_mhz.as_ref(),
                &launch.device,
                span_ms,
            )?;
            for (index, case) in workload.cases.iter().enumerate() {
                let observed = &data.cases[repeat as usize * workload.cases.len() + index];
                if observed.name != case.name
                    || observed.repeat != repeat
                    || !observed.failures.is_empty()
                    || serde_json::to_value(&observed.scenario).map_err(|e| e.to_string())?
                        != serde_json::to_value(&case.scenario).map_err(|e| e.to_string())?
                {
                    return Err("case identity or repeat mismatch".into());
                }
                worst = worst.max(run.case(
                    &root.join(format!("case-{index:03}")),
                    case,
                    observed,
                    &data.serving_plan,
                    &workload.family,
                )?);
            }
            for (index, overlap) in workload.overlaps.iter().enumerate() {
                worst = worst.max(overlap::review(
                    &run,
                    &root.join(format!("overlap-{index:03}")),
                    overlap,
                    &data.overlaps[repeat as usize * workload.overlaps.len() + index],
                    &data.serving_plan,
                    &workload.family,
                    repeat,
                )?);
            }
        }
        count = count
            .checked_add(data.requested_repeats)
            .ok_or("repeat count overflow")?;
        selected = Some(launch);
    }
    if count < MIN_REPEATS {
        return Err(format!(
            "profile needs at least {MIN_REPEATS} samples of every declared workload"
        ));
    }
    Ok(Qualified {
        name: candidate.name.clone(),
        worst_latency_fraction: worst,
        samples_per_workload: count,
        launch: selected.ok_or("missing candidate runs")?,
    })
}

#[cfg(test)]
fn choose(selection: Selection) -> Result<Profile, String> {
    let mut checked = artifact::Verification::default();
    let profile = choose_verified(selection, &mut checked)?;
    checked.finish()?;
    Ok(profile)
}

fn choose_verified(
    selection: Selection,
    checked: &mut artifact::Verification,
) -> Result<Profile, String> {
    let workload = workload(&selection.workload, checked)?;
    let mut selected: Option<Qualified> = None;
    let mut rejected = BTreeMap::new();
    for candidate in &selection.candidates {
        match qualify(candidate, &selection, &workload, checked) {
            Ok(value) => {
                if selected
                    .as_ref()
                    .is_some_and(|s| s.launch.device != value.launch.device)
                {
                    rejected.insert(
                        candidate.name.clone(),
                        "candidate device differs from matched workload campaign".into(),
                    );
                    continue;
                }
                if selected
                    .as_ref()
                    .is_none_or(|s| value.worst_latency_fraction < s.worst_latency_fraction)
                {
                    selected = Some(value);
                }
            }
            Err(error) => {
                rejected.insert(candidate.name.clone(), error);
            }
        }
    }
    Ok(Profile {
        selection,
        qualified: selected.is_some(),
        scope: "declared-serving-workload-only".into(),
        selected,
        rejected,
    })
}

pub(super) fn run(args: &cli::ServingProfile) -> Result<(), String> {
    let mut checked = artifact::Verification::default();
    let selection = selection(&args.plan, &mut checked)?;
    let profile = choose_verified(selection, &mut checked)?;
    let qualified = profile.qualified;
    fs::create_dir(&args.out).map_err(|e| e.to_string())?;
    let mut artifact = Artifact::new(profile);
    artifact.warnings.push("Preflight quote comparison excluded: pre-open and post-open resident credits differ. Final startup and captured candidates compare post-open quotes with only available excluded.".into());
    let data = artifact.require()?;
    let mut paths = vec![args.plan.clone(), data.selection.workload.clone()];
    paths.extend(
        data.selection
            .candidates
            .iter()
            .flat_map(|c| c.runs.clone()),
    );
    for path in paths {
        artifact.inputs.push(artifact::reference(&path, &args.out)?);
    }
    checked.finish()?;
    artifact::save(&args.out.join("profile.json"), &artifact)?;
    if !qualified {
        return Err(
            "no candidate meets every serving profile contract; see profile.json rejected reasons"
                .into(),
        );
    }
    Ok(())
}

pub(super) fn apply(args: &cli::ApplyProfile) -> Result<(), String> {
    let mut checked = artifact::Verification::default();
    let artifact: Artifact<Profile> = artifact::load_verified(&args.profile, &mut checked)?;
    let profile = artifact.require()?;
    let selected = profile
        .selected
        .as_ref()
        .ok_or("unqualified serving profile")?;
    let replay = choose_verified(profile.selection.clone(), &mut checked)?;
    if serde_json::to_value(&replay).map_err(|e| e.to_string())?
        != serde_json::to_value(profile).map_err(|e| e.to_string())?
    {
        return Err("profile differs from revalidated evidence".into());
    }
    fs::create_dir(&args.out).map_err(|e| e.to_string())?;
    fs::write(
        args.out.join("launch.json"),
        serde_json::to_vec_pretty(&selected.launch).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let expected = args.out.join("expected-plan.json");
    fs::write(
        &expected,
        serde_json::to_vec_pretty(&expected_plan(&selected.launch)).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let expected = expected.canonicalize().map_err(|e| e.to_string())?;
    checked.finish()?;
    if !args.execute {
        return Ok(());
    }
    execute(&selected.launch, &args.out, &expected, &checked)
}

fn command(launch: &Launch) -> Command {
    let mut command = Command::new(&launch.executable);
    command.args(&launch.arguments).current_dir(&launch.cwd);
    for (key, _) in std::env::vars_os() {
        if [
            "DS4_",
            "CUDA_",
            "NVIDIA_",
            "LD_",
            "CUBLAS_",
            "OMP_",
            "OPENBLAS_",
        ]
        .iter()
        .any(|prefix| key.to_string_lossy().starts_with(prefix))
        {
            command.env_remove(key);
        }
    }
    command.envs(&launch.environment);
    command
}

fn expected_plan(launch: &Launch) -> Value {
    serde_json::json!({
        "family":launch.serving_plan["family"],
        "backend":launch.serving_plan["requested"]["backend"],
        "effective":launch.serving_plan["effective"],
        "qualified":launch.serving_plan["qualified"],
        "controls":launch.serving_plan["controls"],
        "post_open_quote":launch.serving_plan["quote"]
    })
}

fn guarded(launch: &Launch, expected: &Path) -> Command {
    let mut command = command(launch);
    command.arg("--expect-plan").arg(expected);
    command
}

fn execute(
    launch: &Launch,
    out: &Path,
    expected: &Path,
    checked: &artifact::Verification,
) -> Result<(), String> {
    // Re-run the server's model-free resolver before replacing this process.
    let mut check = guarded(launch, expected);
    check
        .arg("--check-config")
        .stdout(fs::File::create(out.join("check.json")).map_err(|e| e.to_string())?)
        .stderr(fs::File::create(out.join("check.stderr")).map_err(|e| e.to_string())?);
    let budget = cli::Budget {
        timeout_seconds: 120,
        max_output_mib: 64,
    };
    if !process::run(&mut check, out, budget.limits())?.success() {
        return Err("profile server configuration check failed".into());
    }
    let plan: Value = json(&out.join("check.json"))?;
    for key in ["family", "effective", "qualified", "controls"] {
        if plan[key] != launch.serving_plan[key] {
            return Err(format!("profile configuration changed: {key}"));
        }
    }
    if plan["requested"]["backend"] != launch.serving_plan["requested"]["backend"] {
        return Err("profile configuration changed: backend".into());
    }
    let serving = cli::Serving {
        budget,
        url: "http://127.0.0.1:1".into(),
        workload: PathBuf::new(),
        out: out.into(),
        repeats: 1,
        server_pid: None,
        device: launch.device["device_ordinal"]
            .as_u64()
            .ok_or("profile missing device ordinal")? as usize,
    };
    let sample = context::gpu(&serving, out, "gpu-apply")?;
    let range: context::ClockRange =
        serde_json::from_value(launch.expected_clock_range_mhz.clone())
            .map_err(|e| e.to_string())?;
    if !sample.failures(Some(&range)).is_empty() {
        return Err("profile current GPU clock fails workload contract".into());
    }
    if device(&serde_json::to_value(&sample).map_err(|e| e.to_string())?)? != launch.device {
        return Err("profile device/driver changed".into());
    }
    verify_static(launch)?;
    checked.finish()?;
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(guarded(launch, expected).exec().to_string())
    }
    #[cfg(not(unix))]
    {
        Err("profile execution requires Unix".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("ds4-qualified-{}-{nonce}", std::process::id()));
            fs::create_dir(&path).unwrap();
            assert!(Command::new("git")
                .arg("init")
                .arg("--quiet")
                .arg(&path)
                .status()
                .unwrap()
                .success());
            fs::write(path.join("runtime.rs"), "fixture source\n").unwrap();
            fs::write(path.join("model.gguf"), b"fixture model; no inference").unwrap();
            fs::copy("/bin/true", path.join("server")).unwrap();
            Self(path)
        }
        fn write(&self, path: &Path, value: &Value) {
            fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
        }
        fn workload(&self) -> PathBuf {
            let file = self.0.join("workload.json");
            self.write(&file, &json!({"protocol":PROTOCOL,"name":"synthetic","family":"fixture",
                "inputs":{"model":{"path":self.0.join("model.gguf"),"sha256":artifact::hash(&self.0.join("model.gguf")).unwrap()}},
                "expected_clock_range_mhz":{"min":300,"max":2200},
                "cases":[{"name":"cold","scenario":"kv_cold_prefill","request":{"model":"fixture","stream":true,"messages":[{"role":"user","content":"hello"}]},
                    "expect":{"content":"hello","finish_reason":"stop","reuse_kind":"cold","effective_lane":"serial","speculation_active":false,"fallback_reason":null},
                    "limits":{"ttft_ms":100.0,"total_ms":200.0,"min_host_available_bytes":900}}]}));
            file
        }
        fn run(
            &self,
            manifest: &Path,
            name: &str,
            latency: f64,
            available: u64,
            repeats: u32,
        ) -> PathBuf {
            let root = self.0.join(name);
            fs::create_dir(&root).unwrap();
            fs::copy(manifest, root.join("workload.json")).unwrap();
            let workload: Value = json_read(manifest);
            let plan = json!({"family":"fixture","requested":{},"effective":{"max_seqs":1},"qualified":{},"issues":[],"controls":{}});
            let argv = format!(
                "{}\0-m\0{}\0--max-seqs\01\0",
                self.0.join("server").display(),
                self.0.join("model.gguf").display()
            );
            fs::write(root.join("server.argv"), &argv).unwrap();
            let identity = json!({"server":{"pid":1,"boot_id":"fixture","start_ticks":1,
                "executable":self.0.join("server"),"executable_sha256":artifact::hash(&self.0.join("server")).unwrap(),"argv_sha256":artifact::hash_bytes(argv.as_bytes()),
                "cwd":self.0,"environment":{},"unreviewed_environment":[],"source":context::source_snapshot(&self.0).unwrap()},
                "inputs":workload["inputs"],"expected_clock_range_mhz":workload["expected_clock_range_mhz"],"limitations":[]});
            self.write(&root.join("identity.json"), &identity);
            let mut cases = Vec::new();
            let mut samples = Vec::new();
            let mut windows = Vec::new();
            for repeat in 0..repeats {
                let parent = if repeats == 1 {
                    root.clone()
                } else {
                    let p = root.join(format!("repeat-{repeat:03}"));
                    fs::create_dir(&p).unwrap();
                    p
                };
                let started = 1000 + repeat as u64 * 1000;
                let finished = started + latency.ceil() as u64 + 3;
                for (label, time) in [("gpu-before", started), ("gpu-after", finished + 1)] {
                    let sample = json!({"label":label,"unix_ms":time,"device_ordinal":0,"uuid":"fixture-uuid","name":"fixture GPU","driver":"fixture-driver","sm_clock_mhz":1000,"temperature_c":45,"unavailable_reason":null});
                    self.write(&parent.join(format!("{label}.json")), &sample);
                    fs::write(
                        parent.join(format!("{label}.stdout")),
                        "fixture-uuid, fixture GPU, fixture-driver, 1000, 45\n",
                    )
                    .unwrap();
                    samples.push(sample);
                }
                let mut sample = samples.last().unwrap().clone();
                sample["label"] = json!("gpu-poll-00000");
                sample["unix_ms"] = json!(started + 1);
                self.write(&parent.join("gpu-poll-00000.json"), &sample);
                fs::write(
                    parent.join("gpu-poll-00000.stdout"),
                    "fixture-uuid, fixture GPU, fixture-driver, 1000, 45\n",
                )
                .unwrap();
                let window = json!({"poll_interval_ms":500,"started_unix_ms":started,"finished_unix_ms":finished,"samples":[sample]});
                self.write(&parent.join("gpu-window.json"), &window);
                windows.push(window);
                let case = parent.join("case-000");
                fs::create_dir(&case).unwrap();
                self.write(&case.join("request.json"), &workload["cases"][0]["request"]);
                let first = json!({"id":format!("{name}-{repeat}"),"choices":[{"delta":{"content":"hello"}}]}).to_string();
                let finish = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
                self.write(&case.join("events.json"), &json!([{"elapsed_ms":latency,"data":first},{"elapsed_ms":latency+1.0,"data":finish},{"elapsed_ms":latency+2.0,"data":"[DONE]"}]));
                fs::write(
                    case.join("response.sse"),
                    format!("data: {first}\n\ndata: {finish}\n\ndata: [DONE]\n\n"),
                )
                .unwrap();
                fs::write(case.join("response.headers"), "HTTP/1.1 200 OK\r\n\r\n").unwrap();
                self.write(&case.join("memory.json"), &json!([{"elapsed_ms":0.0,"available_bytes":available},{"elapsed_ms":latency+3.0,"available_bytes":available}]));
                let trace = json!({"reuse_kind":"cold","effective_lane":"serial","speculation_active":false,"fallback_reason":null});
                for (name, count) in [("before", 0), ("after", 1)] {
                    self.write(&case.join(format!("{name}.json")), &json!({"serving":plan,"queue_depth":0,"clients":0,"routes":{"serial":count},"last_request":trace}));
                }
                let observed = json!({"name":"cold","repeat":repeat,"scenario":"kv_cold_prefill","ttft_ms":latency,"first_content_ms":latency,"total_ms":latency+2.0,"content":"hello","finish_reason":"stop","host_available_before_bytes":available,"host_available_after_bytes":available,"host_min_available_bytes":available,"trace":trace,"failures":[]});
                self.write(&case.join("result.json"), &observed);
                cases.push(observed);
            }
            let data: Evidence = serde_json::from_value(json!({"protocol":PROTOCOL,"name":"synthetic","family":"fixture","serving_plan":plan,"cases":cases,"overlaps":[],"requested_repeats":repeats,"identity":identity,"gpu_samples":samples,"gpu_windows":windows,"context_failures":[],"passed":true,"timing_method":"synthetic model-free test"})).unwrap();
            let mut evidence = Artifact::new(data);
            references(&root, &root, &mut evidence.inputs).unwrap();
            evidence
                .data
                .as_ref()
                .unwrap()
                .identity
                .add_refs(&mut evidence.inputs);
            let path = root.join("serving.json");
            artifact::save(&path, &evidence).unwrap();
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn json_read(path: &Path) -> Value {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn rewrapped_capture_is_not_repeat() {
        let f = Fixture::new();
        let manifest = f.workload();
        let capture = f.run(&manifest, "single", 20.0, 1000, 1);
        let mut runs = vec![capture.clone()];
        for index in 1..3 {
            let mut copied = json_read(&capture);
            copied["created_unix"] = json!(index);
            copied["warnings"] = json!([format!("metadata copy {index}")]);
            let path = capture.with_file_name(format!("copy-{index}.json"));
            f.write(&path, &copied);
            runs.push(path);
        }
        let profile = choose(Selection {
            workload: manifest,
            candidates: vec![Candidate {
                name: "copied".into(),
                runs,
            }],
        })
        .unwrap();
        assert!(
            !profile.qualified,
            "one runtime capture was counted three times"
        );
        assert!(profile.rejected["copied"].contains("runtime observations"));
    }

    #[test]
    fn gpu_window_spans_actual_requests() {
        let f = Fixture::new();
        let manifest = f.workload();
        let capture = f.run(&manifest, "short-window", 20.0, 1000, 3);
        let mut evidence = json_read(&capture);
        for (index, window) in evidence["data"]["gpu_windows"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            window["finished_unix_ms"] = window["started_unix_ms"].clone();
            f.write(
                &capture
                    .parent()
                    .unwrap()
                    .join(format!("repeat-{index:03}/gpu-window.json")),
                window,
            );
        }
        for input in evidence["inputs"].as_array_mut().unwrap() {
            if input["path"].as_str().unwrap().ends_with("gpu-window.json") {
                let path = capture
                    .parent()
                    .unwrap()
                    .join(input["path"].as_str().unwrap());
                input["sha256"] = json!(artifact::hash(&path).unwrap());
            }
        }
        f.write(&capture, &evidence);
        let profile = choose(Selection {
            workload: manifest,
            candidates: vec![Candidate {
                name: "short-window".into(),
                runs: vec![capture],
            }],
        })
        .unwrap();
        assert!(!profile.qualified);
        assert!(profile.rejected["short-window"].contains("telemetry"));
    }

    #[test]
    fn selects_complete_contract_before_speed() {
        let f = Fixture::new();
        let manifest = f.workload();
        let good = f.run(&manifest, "good", 20.0, 1000, 3);
        let unsafe_fast = f.run(&manifest, "unsafe", 1.0, 100, 3);
        let short = f.run(&manifest, "short", 2.0, 1000, 1);
        let profile = choose(Selection {
            workload: manifest,
            candidates: vec![
                Candidate {
                    name: "unsafe".into(),
                    runs: vec![unsafe_fast],
                },
                Candidate {
                    name: "short".into(),
                    runs: vec![short],
                },
                Candidate {
                    name: "good".into(),
                    runs: vec![good],
                },
            ],
        })
        .unwrap();
        assert!(profile.qualified);
        assert_eq!(profile.selected.unwrap().name, "good");
        assert!(profile.rejected["unsafe"].contains("memory"));
        assert!(profile.rejected["short"].contains("at least 3"));
    }

    #[test]
    fn replay_rechecks_source_binary_and_inputs() {
        let f = Fixture::new();
        let manifest = f.workload();
        let run = f.run(&manifest, "good", 20.0, 1000, 3);
        let selection = Selection {
            workload: manifest,
            candidates: vec![Candidate {
                name: "good".into(),
                runs: vec![run],
            }],
        };
        let profile = choose(selection.clone()).unwrap();
        assert!(profile.qualified);
        let launch = profile.selected.unwrap().launch;
        fs::write(f.0.join("runtime.rs"), "changed source").unwrap();
        assert!(verify_static(&launch)
            .unwrap_err()
            .contains("source changed"));
        fs::write(f.0.join("runtime.rs"), "fixture source\n").unwrap();
        fs::write(f.0.join("server"), b"changed binary").unwrap();
        assert!(verify_static(&launch)
            .unwrap_err()
            .contains("executable changed"));
        fs::write(f.0.join("model.gguf"), b"changed model").unwrap();
        assert!(choose(selection).is_err_and(|e| e.contains("input changed")));
    }

    #[test]
    fn missing_shard_and_late_override_are_rejected() {
        let f = Fixture::new();
        let manifest = f.workload();
        let run = f.run(&manifest, "good", 20.0, 1000, 3);
        let workload = workload(&manifest, &mut artifact::Verification::default()).unwrap();
        let mut launch = launch(
            &review::Run::load(&run).unwrap(),
            &workload,
            &manifest,
            &mut artifact::Verification::default(),
        )
        .unwrap();
        launch.arguments.extend(["--max-seqs".into(), "64".into()]);
        assert!(input_scope(
            &launch,
            &workload,
            &manifest,
            &mut artifact::Verification::default()
        )
        .unwrap_err()
        .contains("duplicate"));
        launch.arguments.truncate(4);
        launch.arguments[3] = "auto".into();
        assert!(input_scope(
            &launch,
            &workload,
            &manifest,
            &mut artifact::Verification::default()
        )
        .unwrap_err()
        .contains("explicit --max-seqs"));
        launch.arguments[3] = "1".into();
        let first = f.0.join("model-00001-of-00002.gguf");
        let second = f.0.join("model-00002-of-00002.gguf");
        fs::write(&first, b"part1").unwrap();
        fs::write(&second, b"part2").unwrap();
        launch.arguments[1] = first.to_string_lossy().into();
        let mut workload = workload;
        workload.inputs.insert(
            "model".into(),
            InputFile {
                path: first.clone(),
                sha256: artifact::hash(&first).unwrap(),
            },
        );
        assert!(input_scope(
            &launch,
            &workload,
            &manifest,
            &mut artifact::Verification::default()
        )
        .unwrap_err()
        .contains("lacks consumed input"));
    }

    #[test]
    fn unpinned_template_sidecar_is_rejected() {
        let f = Fixture::new();
        let manifest = f.workload();
        let run = f.run(&manifest, "good", 20.0, 1000, 3);
        let workload = workload(&manifest, &mut artifact::Verification::default()).unwrap();
        let launch = launch(
            &review::Run::load(&run).unwrap(),
            &workload,
            &manifest,
            &mut artifact::Verification::default(),
        )
        .unwrap();
        fs::write(f.0.join("chat_template.jinja"), "changed template").unwrap();
        assert!(input_scope(
            &launch,
            &workload,
            &manifest,
            &mut artifact::Verification::default()
        )
        .is_err_and(|e| e.contains("template")));
    }

    #[cfg(unix)]
    #[test]
    fn launch_alias_is_rechecked_before_use() {
        let fixture = Fixture::new();
        let manifest = fixture.workload();
        let run = fixture.run(&manifest, "good", 20.0, 1000, 3);
        let mut checked = artifact::Verification::default();
        let workload = workload(&manifest, &mut checked).unwrap();
        let mut launch = launch(
            &review::Run::load(&run).unwrap(),
            &workload,
            &manifest,
            &mut checked,
        )
        .unwrap();
        let alias = fixture.0.join("model-alias");
        std::os::unix::fs::symlink(fixture.0.join("model.gguf"), &alias).unwrap();
        launch.arguments[1] = alias.to_string_lossy().into();
        input_scope(&launch, &workload, &manifest, &mut checked).unwrap();
        let other = fixture.0.join("other-model");
        fs::copy(fixture.0.join("model.gguf"), &other).unwrap();
        fs::remove_file(&alias).unwrap();
        std::os::unix::fs::symlink(other, &alias).unwrap();
        assert!(checked
            .finish()
            .is_err_and(|error| error.contains("path changed")));
    }

    #[test]
    fn checked_application_rederives_the_launch() {
        let f = Fixture::new();
        let manifest = f.workload();
        let capture = f.run(&manifest, "good", 20.0, 1000, 3);
        let selection = f.0.join("selection.json");
        f.write(
            &selection,
            &json!({"workload":manifest,"candidates":[{"name":"good","runs":[capture]}]}),
        );
        let output = f.0.join("profile");
        run(&cli::ServingProfile {
            plan: selection,
            out: output.clone(),
        })
        .unwrap();
        let profile = output.join("profile.json");
        let args = cli::ApplyProfile {
            profile: profile.clone(),
            out: f.0.join("checked"),
            execute: false,
        };
        apply(&args).unwrap();
        assert_eq!(
            json_read(&args.out.join("launch.json"))["arguments"][3],
            "1"
        );
        let mut forged = json_read(&profile);
        forged["data"]["selected"]["launch"]["arguments"][3] = json!("64");
        f.write(&profile, &forged);
        assert!(apply(&cli::ApplyProfile {
            profile,
            out: f.0.join("forged"),
            execute: false
        })
        .is_err_and(|e| e.contains("differs from revalidated")));
    }

    #[test]
    fn recorded_guard_must_be_pinned_and_replaced() {
        let f = Fixture::new();
        let manifest = f.workload();
        let capture = f.run(&manifest, "good", 20.0, 1000, 3);
        let mut workload = workload(&manifest, &mut artifact::Verification::default()).unwrap();
        let mut launch = launch(
            &review::Run::load(&capture).unwrap(),
            &workload,
            &manifest,
            &mut artifact::Verification::default(),
        )
        .unwrap();
        let guard = f.0.join("expected.json");
        f.write(&guard, &expected_plan(&launch));
        launch
            .arguments
            .extend(["--expect-plan".into(), guard.to_string_lossy().into()]);
        assert!(normalize_guard(&mut launch, &workload).is_err());
        workload.inputs.insert(
            "guard".into(),
            InputFile {
                path: guard.clone(),
                sha256: artifact::hash(&guard).unwrap(),
            },
        );
        normalize_guard(&mut launch, &workload).unwrap();
        assert!(!launch.arguments.iter().any(|a| a == "--expect-plan"));
        let command = guarded(&launch, &guard);
        let arguments = command.get_args().collect::<Vec<_>>();
        assert_eq!(arguments[arguments.len() - 2], "--expect-plan");
        assert_eq!(arguments[arguments.len() - 1], guard);
    }
}
