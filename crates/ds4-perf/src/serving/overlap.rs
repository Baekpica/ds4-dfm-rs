use super::*;
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Workload {
    name: String,
    decode: Request,
    prefill: Request,
    min_decode_events_during_prefill: u32,
    min_prefill_tokens: u64,
    max_decode_gap_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    request: Value,
    expect: Expected,
    limits: Contract,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    content: String,
    finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Observation {
    name: String,
    repeat: u32,
    decode_ttft_ms: f64,
    decode_total_ms: f64,
    prefill_dispatch_ms: f64,
    prefill_ttft_ms: f64,
    prefill_total_ms: f64,
    decode_events_during_prefill: u32,
    prefill_computed_tokens: u64,
    decode_committed_tokens: u64,
    max_decode_gap_ms: f64,
    host_available_before_bytes: u64,
    host_available_after_bytes: u64,
    #[serde(default)]
    host_min_available_bytes: Option<u64>,
    failures: Vec<String>,
}

impl Observation {
    pub(super) fn span_ms(&self) -> f64 {
        self.decode_total_ms
            .max(self.prefill_dispatch_ms + self.prefill_total_ms)
    }

    pub(super) fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

impl Workload {
    pub(super) fn name(&self) -> &String {
        &self.name
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        if self.name.is_empty()
            || self.min_decode_events_during_prefill == 0
            || self.min_prefill_tokens == 0
            || !self.max_decode_gap_ms.is_finite()
            || self.max_decode_gap_ms <= 0.0
        {
            return Err("overlap workload requires name and positive progress/gap limits".into());
        }
        for request in [&self.decode, &self.prefill] {
            if request.request["stream"] != true
                || request.request["stream_options"]["include_usage"] != true
                || request.request["messages"]
                    .as_array()
                    .is_none_or(Vec::is_empty)
                || request
                    .request
                    .get("n")
                    .is_some_and(|n| n.as_u64() != Some(1))
                || request.expect.content.is_empty()
                || !["stop", "length"].contains(&request.expect.finish_reason.as_str())
                || [request.limits.ttft_ms, request.limits.total_ms]
                    .iter()
                    .any(|v| !v.is_finite() || *v <= 0.0)
                || request.limits.min_host_available_bytes == 0
            {
                return Err("invalid overlap request/output/latency contract".into());
            }
        }
        Ok(())
    }
}

fn usage(events: &[Event]) -> Result<(u64, u64), String> {
    let usages = events
        .iter()
        .filter_map(|event| serde_json::from_str::<Value>(&event.data).ok())
        .filter_map(|value| value.get("usage").filter(|v| v.is_object()).cloned())
        .collect::<Vec<_>>();
    if usages.len() != 1 {
        return Err("overlap requires exactly one streaming usage record per request".into());
    }
    let usage = &usages[0];
    let prompt = usage["prompt_tokens"]
        .as_u64()
        .ok_or("missing prompt token usage")?;
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .ok_or("missing cached token usage")?;
    let completion = usage["completion_tokens"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("missing completion token usage")?;
    Ok((
        prompt
            .checked_sub(cached)
            .ok_or("cached tokens exceed prompt tokens")?,
        completion,
    ))
}

fn prepare(dir: &Path, request: &Request) -> Result<(), String> {
    fs::create_dir(dir).map_err(|e| e.to_string())?;
    fs::write(
        dir.join("request.json"),
        serde_json::to_vec_pretty(&request.request).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn check(
    output: &Completion,
    request: &Request,
    dispatch_ms: f64,
    name: &str,
    failures: &mut Vec<String>,
) {
    if output.content != request.expect.content
        || output.finish_reason != request.expect.finish_reason
    {
        failures.push(format!(
            "{name}: content/finish_reason differs from expected output"
        ));
    }
    if output.ttft_ms - dispatch_ms > request.limits.ttft_ms {
        failures.push(format!("{name}: TTFT exceeds contract"));
    }
    if output.total_ms - dispatch_ms > request.limits.total_ms {
        failures.push(format!("{name}: total time exceeds contract"));
    }
}

pub(super) fn collect(
    http: &Http,
    workload: &Workload,
    dir: &Path,
    family: &str,
    previous: Option<&Value>,
    repeat: u32,
) -> Result<(Observation, Value), String> {
    fs::create_dir(dir).map_err(|e| e.to_string())?;
    let decode_dir = dir.join("decode");
    let prefill_dir = dir.join("prefill");
    prepare(&decode_dir, &workload.decode)?;
    prepare(&prefill_dir, &workload.prefill)?;
    let before = http.stats(dir, "before")?;
    let current = plan(&before, family)?;
    if previous.is_some_and(|previous| previous != &current) {
        return Err("resolved plan changed before overlap workload".into());
    }
    if current["effective"]["max_seqs"]
        .as_u64()
        .is_none_or(|n| n < 2)
    {
        return Err("overlap workload requires at least two effective banks".into());
    }
    let available_before = host_available()?;
    let start = Instant::now();
    let (sender, receiver) = mpsc::channel();
    let ((decode, prefill, dispatch_ms), available_min) = context::measure(dir, || {
        Ok(std::thread::scope(|scope| {
            let decoder = scope.spawn(|| http.stream_events(&decode_dir, start, Some(sender)));
            // Start the long request only after decode has demonstrably begun.
            let ready = loop {
                match receiver.recv_timeout(Duration::from_millis(25)) {
                    Ok(event) if generated(&event) => break Ok(()),
                    Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        break Err("decode ended before generating output".to_owned())
                    }
                }
            };
            let dispatch_ms = start.elapsed().as_secs_f64() * 1000.0;
            let prefill = ready.and_then(|()| http.stream_events(&prefill_dir, start, None));
            let decode = decoder
                .join()
                .map_err(|_| "decode collector panicked".to_owned())
                .and_then(|v| v);
            (decode, prefill, dispatch_ms)
        }))
    })?;
    let decode_events = decode?;
    let prefill_events = prefill?;
    let decode = summarize(&decode_events)?;
    let prefill = summarize(&prefill_events)?;
    let (prefill_computed_tokens, _) = usage(&prefill_events)?;
    let (_, decode_committed_tokens) = usage(&decode_events)?;
    let available_after = host_available()?;
    let after = http.stats(dir, "after")?;
    if current != plan(&after, family)?
        || route_count(&after)?.checked_sub(route_count(&before)?) != Some(2)
    {
        return Err("overlap plan/route counters changed unexpectedly".into());
    }
    let mut failures = Vec::new();
    check(&decode, &workload.decode, 0.0, "decode", &mut failures);
    check(
        &prefill,
        &workload.prefill,
        dispatch_ms,
        "prefill",
        &mut failures,
    );
    if prefill_computed_tokens < workload.min_prefill_tokens {
        failures.push(format!(
            "peer computed {prefill_computed_tokens} prefill tokens; need {}",
            workload.min_prefill_tokens
        ));
    }
    let mut last = dispatch_ms;
    let mut gap = 0f64;
    let mut progress = 0;
    for event in &decode_events {
        if !generated(event)
            || event.elapsed_ms <= dispatch_ms
            || event.elapsed_ms >= prefill.ttft_ms
        {
            continue;
        }
        gap = gap.max(event.elapsed_ms - last);
        last = event.elapsed_ms;
        progress += 1;
    }
    gap = gap.max(prefill.ttft_ms - last);
    if progress < workload.min_decode_events_during_prefill {
        failures.push(format!(
            "decode made {progress} SSE output events during peer prefill; need {}",
            workload.min_decode_events_during_prefill
        ));
    }
    if gap > workload.max_decode_gap_ms {
        failures.push(format!(
            "decode gap {gap:.3} ms exceeds {} ms during peer prefill",
            workload.max_decode_gap_ms
        ));
    }
    if decode.total_ms <= prefill.ttft_ms {
        failures
            .push("decode ended before peer prefill finished; use a longer decode request".into());
    }
    if available_min.min(available_before).min(available_after)
        < workload
            .decode
            .limits
            .min_host_available_bytes
            .max(workload.prefill.limits.min_host_available_bytes)
    {
        failures.push("host memory below contract at overlap boundary".into());
    }
    let observation = Observation {
        name: workload.name.clone(),
        repeat,
        decode_ttft_ms: decode.ttft_ms,
        decode_total_ms: decode.total_ms,
        prefill_dispatch_ms: dispatch_ms,
        prefill_ttft_ms: prefill.ttft_ms - dispatch_ms,
        prefill_total_ms: prefill.total_ms - dispatch_ms,
        decode_events_during_prefill: progress,
        prefill_computed_tokens,
        decode_committed_tokens,
        max_decode_gap_ms: gap,
        host_available_before_bytes: available_before,
        host_available_after_bytes: available_after,
        host_min_available_bytes: Some(available_min),
        failures,
    };
    fs::write(
        dir.join("result.json"),
        serde_json::to_vec_pretty(&observation).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok((observation, current))
}

pub(super) fn review(
    run: &review::Run,
    dir: &Path,
    workload: &Workload,
    observation: &Observation,
    expected: &Value,
    family: &str,
    repeat: u32,
) -> Result<f64, String> {
    run.stats(dir, expected, 2, family)?;
    let raw: Value = run.json(&dir.join("result.json"))?;
    let (decode_events, decode) = run.stream(&dir.join("decode"), &workload.decode.request)?;
    let (prefill_events, prefill) = run.stream(&dir.join("prefill"), &workload.prefill.request)?;
    let (computed, _) = usage(&prefill_events)?;
    let (_, committed) = usage(&decode_events)?;
    let memory = run.memory(dir, decode.total_ms.max(prefill.total_ms))?;
    let dispatch = observation.prefill_dispatch_ms;
    let mut failures = Vec::new();
    check(&decode, &workload.decode, 0.0, "decode", &mut failures);
    check(
        &prefill,
        &workload.prefill,
        dispatch,
        "prefill",
        &mut failures,
    );
    let mut last = dispatch;
    let mut gap = 0f64;
    let mut progress = 0;
    for event in &decode_events {
        if generated(event) && event.elapsed_ms > dispatch && event.elapsed_ms < prefill.ttft_ms {
            gap = gap.max(event.elapsed_ms - last);
            last = event.elapsed_ms;
            progress += 1;
        }
    }
    gap = gap.max(prefill.ttft_ms - last);
    if raw != serde_json::to_value(observation).map_err(|e| e.to_string())?
        || observation.name != workload.name
        || observation.repeat != repeat
        || !observation.failures.is_empty()
        || !failures.is_empty()
        || !dispatch.is_finite()
        || dispatch <= decode.ttft_ms
        || dispatch >= prefill.ttft_ms
        || decode.total_ms <= prefill.ttft_ms
        || observation.decode_ttft_ms != decode.ttft_ms
        || observation.decode_total_ms != decode.total_ms
        || observation.prefill_ttft_ms != prefill.ttft_ms - dispatch
        || observation.prefill_total_ms != prefill.total_ms - dispatch
        || observation.decode_events_during_prefill != progress
        || observation.max_decode_gap_ms != gap
        || observation.prefill_computed_tokens != computed
        || observation.decode_committed_tokens != committed
        || progress < workload.min_decode_events_during_prefill
        || computed < workload.min_prefill_tokens
        || gap > workload.max_decode_gap_ms
        || observation.host_min_available_bytes != Some(memory)
        || memory
            .min(observation.host_available_before_bytes)
            .min(observation.host_available_after_bytes)
            < workload
                .decode
                .limits
                .min_host_available_bytes
                .max(workload.prefill.limits.min_host_available_bytes)
    {
        return Err(format!("{}: raw overlap contract failed", workload.name));
    }
    Ok([
        decode.ttft_ms / workload.decode.limits.ttft_ms,
        decode.total_ms / workload.decode.limits.total_ms,
        (prefill.ttft_ms - dispatch) / workload.prefill.limits.ttft_ms,
        (prefill.total_ms - dispatch) / workload.prefill.limits.total_ms,
        gap / workload.max_decode_gap_ms,
    ]
    .into_iter()
    .fold(0.0, f64::max))
}
