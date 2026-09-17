use crate::{
    cli, process,
    serving_http::{Event, Http},
};
use ds4_perf::artifact::{self, Artifact, Payload};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};

mod context;
mod controls;
mod overlap;
mod profile;
mod restart;
mod review;
mod telemetry;
mod tools;

pub fn controls(args: &cli::ServingControls) -> Result<(), String> {
    controls::run(args)
}

pub fn profile(args: &cli::ServingProfile) -> Result<(), String> {
    profile::run(args)
}

pub fn apply(args: &cli::ApplyProfile) -> Result<(), String> {
    profile::apply(args)
}

const PROTOCOL: &str = "ds4-serving-v1";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Workload {
    protocol: String,
    name: String,
    family: String,
    cases: Vec<Case>,
    #[serde(default)]
    overlaps: Vec<overlap::Workload>,
    #[serde(default)]
    inputs: std::collections::BTreeMap<String, crate::experiment::InputFile>,
    #[serde(default)]
    expected_clock_range_mhz: Option<context::ClockRange>,
    #[serde(default)]
    restart_from: Option<std::path::PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    scenario: Scenario,
    request: Value,
    expect: Expect,
    limits: Contract,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Scenario {
    KvColdPrefill,
    WarmAppend,
    PartialBranch,
    Media,
    Mtp,
    Tool,
    RestartRestore,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expect {
    content: String,
    finish_reason: String,
    reuse_kind: String,
    effective_lane: String,
    speculation_active: bool,
    fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<tools::Call>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_cached_tokens: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Contract {
    ttft_ms: f64,
    total_ms: f64,
    min_host_available_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    name: String,
    repeat: u32,
    scenario: Scenario,
    ttft_ms: f64,
    first_content_ms: Option<f64>,
    total_ms: f64,
    content: String,
    finish_reason: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<tools::Call>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cached_tokens: Option<u64>,
    host_available_before_bytes: u64,
    host_available_after_bytes: u64,
    #[serde(default)]
    host_min_available_bytes: Option<u64>,
    trace: Value,
    failures: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    protocol: String,
    name: String,
    family: String,
    serving_plan: Value,
    cases: Vec<Observation>,
    overlaps: Vec<overlap::Observation>,
    requested_repeats: u32,
    identity: context::Identity,
    gpu_samples: Vec<context::GpuSample>,
    #[serde(default)]
    gpu_windows: Vec<telemetry::Window>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    restart_from: Option<artifact::Reference>,
    context_failures: Vec<String>,
    passed: bool,
    timing_method: String,
}

impl Payload for Evidence {
    const KIND: &'static str = "serving";
    fn validate(&self) -> Result<(), String> {
        if self.protocol != PROTOCOL
            || (self.cases.is_empty() && self.overlaps.is_empty())
            || self.passed
                != (self.cases.iter().all(|case| case.failures.is_empty())
                    && self.overlaps.iter().all(overlap::Observation::passed)
                    && self.context_failures.is_empty())
            || self.cases.iter().any(|case| {
                !case.ttft_ms.is_finite()
                    || case.ttft_ms <= 0.0
                    || !case.total_ms.is_finite()
                    || case.total_ms < case.ttft_ms
            })
        {
            return Err("invalid serving evidence".into());
        }
        Ok(())
    }
}

impl Workload {
    fn validate(&self) -> Result<(), String> {
        if let Some(range) = &self.expected_clock_range_mhz {
            range.validate()?;
        }
        if self.protocol != PROTOCOL
            || self.name.is_empty()
            || self.family.is_empty()
            || (self.cases.is_empty() && self.overlaps.is_empty())
            || self.cases.len() > 128
            || self.overlaps.len() > 32
        {
            return Err(
                "serving workload needs protocol ds4-serving-v1, name, family and 1..128 cases"
                    .into(),
            );
        }
        let mut names = BTreeSet::new();
        if self.restart_from.is_some()
            != self
                .cases
                .first()
                .is_some_and(|case| matches!(case.scenario, Scenario::RestartRestore))
        {
            return Err("restart_from requires restart_restore as the first case".into());
        }
        for case in &self.cases {
            if case.name.is_empty()
                || !names.insert(&case.name)
                || case.request.get("stream") != Some(&Value::Bool(true))
                || case
                    .request
                    .get("messages")
                    .and_then(Value::as_array)
                    .is_none_or(Vec::is_empty)
                || case.request.get("n").is_some_and(|n| n.as_u64() != Some(1))
                || !tools::valid_finish(
                    &case.expect.content,
                    &case.expect.finish_reason,
                    &case.expect.tool_calls,
                )
                || case
                    .expect
                    .tool_calls
                    .iter()
                    .any(|call| !call.declared(&case.request))
                || case.expect.min_cached_tokens.is_some_and(|n| {
                    n == 0 || case.request["stream_options"]["include_usage"] != true
                })
                || !["cold", "exact", "partial", "fork"].contains(&case.expect.reuse_kind.as_str())
                || !["serial", "continuous", "static"]
                    .contains(&case.expect.effective_lane.as_str())
                || [case.limits.ttft_ms, case.limits.total_ms]
                    .iter()
                    .any(|v| !v.is_finite() || *v <= 0.0)
                || case.limits.min_host_available_bytes == 0
            {
                return Err(format!(
                    "{}: invalid request, expectation or limits",
                    case.name
                ));
            }
            let valid = match case.scenario {
                Scenario::KvColdPrefill => case.expect.reuse_kind == "cold",
                Scenario::WarmAppend => {
                    matches!(
                        case.expect.reuse_kind.as_str(),
                        "exact" | "partial" | "fork"
                    )
                }
                Scenario::PartialBranch => {
                    matches!(case.expect.reuse_kind.as_str(), "partial" | "fork")
                }
                Scenario::Mtp => case.expect.speculation_active,
                Scenario::Tool => !case.expect.tool_calls.is_empty(),
                Scenario::RestartRestore => {
                    self.restart_from.is_some()
                        && case.expect.min_cached_tokens.is_some()
                        && matches!(
                            case.expect.reuse_kind.as_str(),
                            "exact" | "partial" | "fork"
                        )
                }
                Scenario::Media => case.request["messages"].as_array().is_some_and(|messages| {
                    messages.iter().any(|message| {
                        message["content"].as_array().is_some_and(|parts| {
                            parts.iter().any(|part| {
                                matches!(part["type"].as_str(), Some("image_url" | "input_audio"))
                            })
                        })
                    })
                }),
            };
            if !valid {
                return Err(format!(
                    "{}: expectation contradicts the serving scenario",
                    case.name
                ));
            }
        }
        for overlap in &self.overlaps {
            overlap.validate()?;
            if !names.insert(overlap.name()) {
                return Err("serving case names must be unique".into());
            }
        }
        Ok(())
    }
}

fn host_available() -> Result<u64, String> {
    let text = fs::read_to_string("/proc/meminfo")
        .map_err(|e| format!("local host memory unavailable: {e}"))?;
    text.lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse::<u64>().ok())
        })
        .and_then(|kb| kb.checked_mul(1024))
        .ok_or("MemAvailable missing".into())
}

fn plan(stats: &Value, family: &str) -> Result<Value, String> {
    let plan = &stats["serving"];
    if plan["family"] != family
        || !plan["requested"].is_object()
        || !plan["effective"].is_object()
        || !plan["qualified"].is_object()
        || plan["issues"]
            .as_array()
            .is_none_or(|issues| issues.iter().any(|i| i["level"] == "error"))
    {
        return Err("server stats lack an error-free resolved plan for the workload family".into());
    }
    if stats["queue_depth"].as_u64() != Some(0)
        || stats["clients"].as_u64().is_none_or(|clients| clients > 1)
    {
        return Err("serving collection requires an idle server with no competing clients".into());
    }
    Ok(plan.clone())
}

fn route_count(stats: &Value) -> Result<u64, String> {
    stats["routes"]
        .as_object()
        .ok_or("server stats lack route counters")?
        .values()
        .try_fold(0u64, |sum, n| {
            sum.checked_add(n.as_u64().ok_or("invalid route counter")?)
                .ok_or("route counter overflow")
        })
        .map_err(Into::into)
}

fn post_open_quote(plan: &Value) -> Value {
    let mut quote = plan["quote"].clone();
    if let Some(fields) = quote.as_object_mut() {
        fields.remove("available");
    }
    quote
}

struct Completion {
    ttft_ms: f64,
    first_content_ms: Option<f64>,
    total_ms: f64,
    content: String,
    finish_reason: String,
    tool_calls: Vec<tools::Call>,
    cached_tokens: Option<u64>,
}

fn summarize(events: &[Event]) -> Result<Completion, String> {
    let mut first = None;
    let mut visible = None;
    let mut done = None;
    let mut content = String::new();
    let mut finished = None;
    let mut tool_calls = tools::Calls::default();
    let mut cached_tokens = None;
    let mut usage_seen = false;
    for event in events {
        if done.is_some() {
            return Err("SSE data after DONE".into());
        }
        if event.data == "[DONE]" {
            done = Some(event.elapsed_ms);
            continue;
        }
        let value: Value =
            serde_json::from_str(&event.data).map_err(|e| format!("invalid SSE JSON: {e}"))?;
        if value.get("error").is_some() {
            return Err(format!("server streaming error: {}", value["error"]));
        }
        if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
            if usage_seen {
                return Err("duplicate streaming usage".into());
            }
            usage_seen = true;
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"].as_u64();
            if cached_tokens.is_some_and(|cached| {
                usage["prompt_tokens"]
                    .as_u64()
                    .is_none_or(|prompt| cached > prompt)
            }) {
                return Err("cached token usage exceeds prompt".into());
            }
        }
        let choices = value["choices"]
            .as_array()
            .ok_or("SSE event lacks choices")?;
        if choices.len() > 1 {
            return Err("serving collection supports one completion per request".into());
        }
        let Some(choice) = choices.first() else {
            continue;
        };
        if finished.is_some() {
            return Err("SSE choice after finish reason".into());
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            if !["stop", "length", "tool_calls"].contains(&reason) || finished.is_some() {
                return Err("unsupported or duplicate SSE finish reason".into());
            }
            finished = Some(reason.to_owned());
        }
        let delta = &choice["delta"];
        if tool_calls.add(delta)? {
            first.get_or_insert(event.elapsed_ms);
        }
        if let Some(text) = delta["content"].as_str().filter(|text| !text.is_empty()) {
            visible.get_or_insert(event.elapsed_ms);
            content.push_str(text);
        }
        if delta["content"]
            .as_str()
            .is_some_and(|text| !text.is_empty())
            || delta["reasoning_content"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        {
            first.get_or_insert(event.elapsed_ms);
        }
    }
    let finish_reason = finished.ok_or("SSE stream lacks a finish reason")?;
    let tool_calls = tool_calls.finish()?;
    let generated_calls = !tool_calls.is_empty();
    if generated_calls != (finish_reason == "tool_calls") {
        return Err("tool call finish reason disagrees with generated calls".into());
    }
    Ok(Completion {
        ttft_ms: first.ok_or("SSE stream has no generated output")?,
        first_content_ms: visible,
        total_ms: done.ok_or("SSE stream lacks DONE")?,
        content,
        finish_reason,
        tool_calls,
        cached_tokens,
    })
}

fn generated(event: &Event) -> bool {
    serde_json::from_str::<Value>(&event.data).is_ok_and(|value| {
        let delta = &value["choices"][0]["delta"];
        ["content", "reasoning_content"]
            .into_iter()
            .any(|key| delta[key].as_str().is_some_and(|text| !text.is_empty()))
            || delta["tool_calls"].as_array().is_some_and(|calls| {
                calls.iter().any(|call| {
                    ["name", "arguments"].iter().any(|key| {
                        call["function"][key]
                            .as_str()
                            .is_some_and(|text| !text.is_empty())
                    })
                })
            })
    })
}

fn references(dir: &Path, root: &Path, out: &mut Vec<artifact::Reference>) -> Result<(), String> {
    let mut paths = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .map(|entry| entry.map(|e| e.path()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            references(&path, root, out)?;
        } else {
            out.push(artifact::reference(&path, root)?);
        }
    }
    Ok(())
}

fn collect(args: &cli::Serving, workload: Workload) -> Result<Evidence, String> {
    let http = Http::new(&args.url, &args.out, args.budget.limits())?;
    let mut checked = artifact::Verification::default();
    let identity = context::Identity::capture(args, &workload, &mut checked)?;
    let restart_from =
        restart::prepare(&workload, &identity, &args.out, args.repeats, &mut checked)?;
    checked.finish()?;
    let mut gpu_samples = Vec::new();
    let mut gpu_windows = Vec::new();
    let mut context_failures = Vec::new();
    let mut observations = Vec::new();
    let mut overlaps = Vec::new();
    let mut resolved = None;
    for repeat in 0..args.repeats {
        let root = if args.repeats == 1 {
            args.out.clone()
        } else {
            let root = args.out.join(format!("repeat-{repeat:03}"));
            fs::create_dir(&root).map_err(|e| e.to_string())?;
            root
        };
        let gpu = context::gpu(args, &root, "gpu-before")?;
        context_failures.extend(gpu.failures(workload.expected_clock_range_mhz.as_ref()));
        gpu_samples.push(gpu);
        let (_, window) = telemetry::observe(args, &root, || {
            for (index, case) in workload.cases.iter().cloned().enumerate() {
                let dir = root.join(format!("case-{index:03}"));
                fs::create_dir(&dir).map_err(|e| e.to_string())?;
                fs::write(
                    dir.join("request.json"),
                    serde_json::to_vec_pretty(&case.request).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                let before = http.stats(&dir, "before")?;
                let current = plan(&before, &workload.family)?;
                if matches!(case.scenario, Scenario::RestartRestore) {
                    restart::first_request(
                        &workload,
                        &current,
                        &before,
                        index,
                        repeat,
                        &mut checked,
                    )?;
                }
                if resolved
                    .as_ref()
                    .is_some_and(|previous| previous != &current)
                {
                    return Err("resolved serving plan changed during collection".into());
                }
                resolved = Some(current.clone());
                let available_before = host_available()?;
                let (events, available_min) = context::measure(&dir, || http.stream(&dir))?;
                let available_after = host_available()?;
                let after = http.stats(&dir, "after")?;
                if current != plan(&after, &workload.family)? {
                    return Err("resolved serving plan changed during request".into());
                }
                if route_count(&after)?.checked_sub(route_count(&before)?) != Some(1) {
                    return Err(
                        "request trace cannot be attributed: route count did not advance by one"
                            .into(),
                    );
                }
                let output = summarize(&events)?;
                let trace = &after["last_request"];
                let mut failures = Vec::new();
                for (key, expected) in [
                    ("reuse_kind", Value::String(case.expect.reuse_kind)),
                    ("effective_lane", Value::String(case.expect.effective_lane)),
                    (
                        "speculation_active",
                        Value::Bool(case.expect.speculation_active),
                    ),
                    (
                        "fallback_reason",
                        case.expect
                            .fallback_reason
                            .map(Value::String)
                            .unwrap_or(Value::Null),
                    ),
                ] {
                    if trace.get(key) != Some(&expected) {
                        failures.push(format!(
                            "{key}: expected {expected}, observed {}",
                            trace[key]
                        ));
                    }
                }
                if output.content != case.expect.content {
                    failures.push("content differs from expected output".into());
                }
                if output.finish_reason != case.expect.finish_reason {
                    failures.push("finish_reason differs from expected output".into());
                }
                if output.tool_calls != case.expect.tool_calls {
                    failures.push("tool calls differ from expected names/JSON arguments".into());
                }
                if case.expect.min_cached_tokens.is_some_and(|minimum| {
                    output.cached_tokens.is_none_or(|cached| cached < minimum)
                }) {
                    failures.push("cached token usage below contract".into());
                }
                if output.ttft_ms > case.limits.ttft_ms {
                    failures.push(format!(
                        "TTFT {:.3} ms exceeds {} ms",
                        output.ttft_ms, case.limits.ttft_ms
                    ));
                }
                if output.total_ms > case.limits.total_ms {
                    failures.push(format!(
                        "total {:.3} ms exceeds {} ms",
                        output.total_ms, case.limits.total_ms
                    ));
                }
                if available_min.min(available_before).min(available_after)
                    < case.limits.min_host_available_bytes
                {
                    failures.push("sampled host available memory below contract".into());
                }
                let observation = Observation {
                    name: case.name,
                    repeat,
                    scenario: case.scenario,
                    ttft_ms: output.ttft_ms,
                    first_content_ms: output.first_content_ms,
                    total_ms: output.total_ms,
                    content: output.content,
                    finish_reason: output.finish_reason,
                    tool_calls: output.tool_calls,
                    cached_tokens: output.cached_tokens,
                    host_available_before_bytes: available_before,
                    host_available_after_bytes: available_after,
                    host_min_available_bytes: Some(available_min),
                    trace: trace.clone(),
                    failures,
                };
                fs::write(
                    dir.join("result.json"),
                    serde_json::to_vec_pretty(&observation).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                observations.push(observation);
                process::check_bytes(&args.out, args.budget.limits())?;
            }
            for (index, overlap) in workload.overlaps.iter().enumerate() {
                let dir = root.join(format!("overlap-{index:03}"));
                let (observation, current) = overlap::collect(
                    &http,
                    overlap,
                    &dir,
                    &workload.family,
                    resolved.as_ref(),
                    repeat,
                )?;
                resolved = Some(current);
                overlaps.push(observation);
                process::check_bytes(&args.out, args.budget.limits())?;
            }
            Ok(())
        })?;
        context_failures.extend(window.failures(workload.expected_clock_range_mhz.as_ref()));
        gpu_windows.push(window);
        let gpu = context::gpu(args, &root, "gpu-after")?;
        context_failures.extend(gpu.failures(workload.expected_clock_range_mhz.as_ref()));
        gpu_samples.push(gpu);
    }
    identity.verify(args)?;
    let passed = observations.iter().all(|case| case.failures.is_empty())
        && overlaps.iter().all(overlap::Observation::passed)
        && context_failures.is_empty();
    Ok(Evidence { protocol: PROTOCOL.into(), name: workload.name, family: workload.family,
        serving_plan: resolved.ok_or("no serving plan collected")?, cases: observations, overlaps, requested_repeats: args.repeats, passed,
        identity, gpu_samples, gpu_windows, restart_from, context_failures,
        timing_method: "client dispatch through complete SSE event; curl --no-buffer; 1 ms file observer; includes curl startup; no profiler".into() })
}

pub fn run(args: &cli::Serving) -> Result<(), String> {
    if fs::metadata(&args.workload)
        .map_err(|e| e.to_string())?
        .len()
        > MAX_MANIFEST_BYTES
    {
        return Err("serving workload exceeds 16 MiB".into());
    }
    let bytes = fs::read(&args.workload).map_err(|e| e.to_string())?;
    let mut workload: Workload = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    workload.validate()?;
    if let Some(path) = &mut workload.restart_from {
        *path = args
            .workload
            .parent()
            .unwrap_or(Path::new("."))
            .join(&*path)
            .canonicalize()
            .map_err(|e| e.to_string())?;
    }
    for input in workload.inputs.values_mut() {
        input.path = args
            .workload
            .parent()
            .unwrap_or(Path::new("."))
            .join(&input.path)
            .canonicalize()
            .map_err(|e| e.to_string())?;
    }
    fs::create_dir(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    fs::write(args.out.join("workload.json"), bytes).map_err(|e| e.to_string())?;
    let result = collect(args, workload);
    let (mut evidence, failure) = match result {
        Ok(data) => {
            let failure = (!data.passed).then(|| "serving workload contract failed".to_owned());
            (Artifact::new(data), failure)
        }
        Err(error) => (Artifact::<Evidence>::failed(error.clone()), Some(error)),
    };
    references(&args.out, &args.out, &mut evidence.inputs)?;
    if let Some(data) = &evidence.data {
        data.identity.add_refs(&mut evidence.inputs);
        if let Some(seed) = &data.restart_from {
            evidence.inputs.push(seed.clone());
        }
    }
    artifact::save(&args.out.join("serving.json"), &evidence)?;
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_append_uses_observed_reuse() {
        for reuse in ["exact", "partial", "fork", "cold"] {
            let workload: Workload = serde_json::from_value(serde_json::json!({
                "protocol":PROTOCOL,"name":"append","family":"qwen4exp",
                "cases":[{
                    "name":"append","scenario":"warm_append",
                    "request":{"stream":true,"stream_options":{"include_usage":true},
                        "messages":[{"role":"user","content":"Continue"}]},
                    "expect":{"content":"READY","finish_reason":"stop","reuse_kind":reuse,
                        "effective_lane":"continuous","speculation_active":true,
                        "fallback_reason":null,"min_cached_tokens":32},
                    "limits":{"ttft_ms":1000,"total_ms":2000,"min_host_available_bytes":1}
                }]
            }))
            .unwrap();
            assert_eq!(workload.validate().is_ok(), reuse != "cold", "{reuse}");
        }
    }

    #[test]
    fn disk_restore_can_place_a_fork() {
        let workload: Workload = serde_json::from_value(serde_json::json!({
            "protocol":PROTOCOL,"name":"restore-fork","family":"qwen4exp",
            "restart_from":"seed/serving.json","cases":[{
                "name":"restored","scenario":"restart_restore",
                "request":{"stream":true,"stream_options":{"include_usage":true},
                    "messages":[{"role":"user","content":"Continue"}]},
                "expect":{"content":"READY","finish_reason":"stop","reuse_kind":"fork",
                    "effective_lane":"continuous","speculation_active":true,
                    "fallback_reason":null,"min_cached_tokens":32},
                "limits":{"ttft_ms":1000,"total_ms":2000,"min_host_available_bytes":1}
            }]
        }))
        .unwrap();
        assert!(
            workload.validate().is_ok(),
            "a restored full prefix may fork into a free bank"
        );
    }

    fn event(elapsed_ms: f64, data: &str) -> Event {
        Event {
            elapsed_ms,
            data: data.into(),
        }
    }

    #[test]
    fn reasoning_and_visible_ttft() {
        let events = [
            event(1.0, r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
            event(
                12.0,
                r#"{"choices":[{"delta":{"reasoning_content":"think"}}]}"#,
            ),
            event(23.0, r#"{"choices":[{"delta":{"content":"answer"}}]}"#),
            event(25.0, r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
            event(30.0, "[DONE]"),
        ];
        let output = summarize(&events).unwrap();
        assert_eq!(output.ttft_ms, 12.0);
        assert_eq!(output.first_content_ms, Some(23.0));
        assert_eq!(output.total_ms, 30.0);
        assert_eq!(output.content, "answer");
        assert_eq!(output.finish_reason, "stop");
        assert!(summarize(&events[..4]).is_err_and(|e| e.contains("DONE")));
        assert!(summarize(&events[..3]).is_err_and(|e| e.contains("finish")));
    }

    #[test]
    fn incomplete_or_error_stream() {
        for data in [
            r#"{"error":{"message":"failed"}}"#,
            "not json",
            r#"{"choices":[{"delta":{"tool_calls":[{}]}}]}"#,
        ] {
            assert!(summarize(&[event(1.0, data), event(2.0, "[DONE]")]).is_err());
        }
        assert!(summarize(&[event(1.0, "[DONE]"), event(2.0, "[DONE]")]).is_err());
    }

    #[test]
    fn finish_is_terminal_for_choice_deltas() {
        for delta in [
            r#"{"content":"late"}"#,
            r#"{"reasoning_content":"late"}"#,
            r#"{"tool_calls":[{"index":0,"id":"late","function":{"name":"f","arguments":"{}"}}]}"#,
        ] {
            let stream = [
                event(1.0, r#"{"choices":[{"delta":{"content":"ok"}}]}"#),
                event(2.0, r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                event(3.0, &format!(r#"{{"choices":[{{"delta":{delta}}}]}}"#)),
                event(4.0, "[DONE]"),
            ];
            assert!(summarize(&stream).is_err_and(|e| e.contains("after finish")));
        }
        let stream = [
            event(
                1.0,
                r#"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
            ),
            event(
                2.0,
                r#"{"choices":[],"usage":{"prompt_tokens":1,"prompt_tokens_details":{"cached_tokens":0}}}"#,
            ),
            event(3.0, "[DONE]"),
        ];
        assert!(
            summarize(&stream).is_ok(),
            "usage after final delta is valid"
        );
    }

    #[test]
    fn fragmented_tool_arguments_are_checked() {
        let stream = [
            event(1.0, r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
            event(
                2.0,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-a","type":"function","function":{"name":"weather","arguments":""}}]}}]}"#,
            ),
            event(
                3.0,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"#,
            ),
            event(
                4.0,
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Seoul\"}"}}]}}]}"#,
            ),
            event(
                5.0,
                r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            ),
            event(6.0, "[DONE]"),
        ];
        let output = summarize(&stream).unwrap();
        assert_eq!(output.ttft_ms, 2.0);
        assert_eq!(output.first_content_ms, None);
        assert_eq!(
            serde_json::to_value(&output.tool_calls).unwrap(),
            serde_json::json!([{"name":"weather","arguments":{"city":"Seoul"}}])
        );
        let mut broken = stream.to_vec();
        broken.remove(3);
        assert!(summarize(&broken).is_err_and(|e| e.contains("arguments")));
        let mut wrong_finish = stream.to_vec();
        wrong_finish[4] = event(5.0, r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#);
        assert!(summarize(&wrong_finish).is_err_and(|e| e.contains("finish reason")));
    }
}
