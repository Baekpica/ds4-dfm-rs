//! Recompute contracts from pinned raw records; summaries alone never qualify.
use super::*;
use serde::de::DeserializeOwned;
use std::path::PathBuf;

pub(super) struct Run {
    root: PathBuf,
    evidence: Artifact<Evidence>,
}

impl Run {
    #[cfg(test)]
    pub(super) fn load(path: &Path) -> Result<Self, String> {
        Ok(Self {
            root: path.parent().ok_or("run has no directory")?.into(),
            evidence: artifact::load(path)?,
        })
    }

    pub(super) fn load_verified(
        path: &Path,
        checked: &mut artifact::Verification,
    ) -> Result<Self, String> {
        Ok(Self {
            root: path.parent().ok_or("run has no directory")?.into(),
            evidence: artifact::load_verified(path, checked)?,
        })
    }

    pub(super) fn data(&self) -> Result<&Evidence, String> {
        self.evidence.require()
    }

    pub(super) fn bytes(&self, path: &Path) -> Result<Vec<u8>, String> {
        let file = self
            .root
            .join(path)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        if !self
            .evidence
            .inputs
            .iter()
            .any(|input| self.root.join(&input.path).canonicalize().ok().as_ref() == Some(&file))
        {
            return Err(format!("raw evidence is not pinned: {}", path.display()));
        }
        fs::read(file).map_err(|e| e.to_string())
    }

    pub(super) fn json<T: DeserializeOwned>(&self, path: &Path) -> Result<T, String> {
        serde_json::from_slice(&self.bytes(path)?).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub(super) fn stream(
        &self,
        dir: &Path,
        request: &Value,
    ) -> Result<(Vec<Event>, Completion), String> {
        // Remote media can change while the request JSON stays identical.
        for message in request["messages"].as_array().into_iter().flatten() {
            for part in message["content"].as_array().into_iter().flatten() {
                let pinned = match part["type"].as_str() {
                    Some("image_url") => part["image_url"]["url"]
                        .as_str()
                        .is_some_and(|url| url.starts_with("data:image/")),
                    Some("input_audio") => part["input_audio"]["data"]
                        .as_str()
                        .is_some_and(|data| !data.is_empty()),
                    _ => true,
                };
                if !pinned {
                    return Err("profile media must be embedded in the hashed request".into());
                }
            }
        }
        let actual: Value = self.json(&dir.join("request.json"))?;
        if &actual != request {
            return Err("captured request differs from workload".into());
        }
        let events: Vec<Event> = self.json(&dir.join("events.json"))?;
        if events
            .iter()
            .any(|e| !e.elapsed_ms.is_finite() || e.elapsed_ms <= 0.0)
            || events.windows(2).any(|e| e[1].elapsed_ms < e[0].elapsed_ms)
        {
            return Err("invalid SSE event timing".into());
        }
        let raw =
            String::from_utf8(self.bytes(&dir.join("response.sse"))?).map_err(|e| e.to_string())?;
        let mut payloads = Vec::new();
        let mut lines = Vec::new();
        for line in raw.lines() {
            if line.is_empty() {
                if !lines.is_empty() {
                    payloads.push(lines.join("\n"));
                    lines.clear();
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                lines.push(data.strip_prefix(' ').unwrap_or(data));
            }
        }
        if !lines.is_empty()
            || payloads != events.iter().map(|e| e.data.clone()).collect::<Vec<_>>()
        {
            return Err("SSE events differ from raw response".into());
        }
        let headers = String::from_utf8(self.bytes(&dir.join("response.headers"))?)
            .map_err(|e| e.to_string())?;
        if headers
            .lines()
            .rfind(|l| l.starts_with("HTTP/"))
            .and_then(|l| l.split_whitespace().nth(1))
            != Some("200")
        {
            return Err("captured response was not HTTP 200".into());
        }
        let completion = summarize(&events)?;
        Ok((events, completion))
    }

    pub(super) fn memory(&self, dir: &Path, duration_ms: f64) -> Result<u64, String> {
        self.bytes(&dir.join("memory.json"))?;
        context::memory_min(&self.root.join(dir), duration_ms)
    }

    pub(super) fn stats(
        &self,
        dir: &Path,
        expected: &Value,
        routes: u64,
        family: &str,
    ) -> Result<Value, String> {
        let before: Value = self.json(&dir.join("before.json"))?;
        let after: Value = self.json(&dir.join("after.json"))?;
        if &plan(&before, family)? != expected
            || &plan(&after, family)? != expected
            || route_count(&after)?.checked_sub(route_count(&before)?) != Some(routes)
        {
            return Err("raw serving plan or request attribution changed".into());
        }
        Ok(after["last_request"].clone())
    }

    pub(super) fn case(
        &self,
        dir: &Path,
        case: &Case,
        observation: &Observation,
        expected: &Value,
        family: &str,
    ) -> Result<f64, String> {
        let (_, output) = self.stream(dir, &case.request)?;
        let trace = self.stats(dir, expected, 1, family)?;
        let memory = self.memory(dir, output.total_ms)?;
        let raw: Value = self.json(&dir.join("result.json"))?;
        if raw != serde_json::to_value(observation).map_err(|e| e.to_string())?
            || output.ttft_ms != observation.ttft_ms
            || output.total_ms != observation.total_ms
            || output.first_content_ms != observation.first_content_ms
            || output.content != observation.content
            || output.finish_reason != observation.finish_reason
            || output.content != case.expect.content
            || output.finish_reason != case.expect.finish_reason
            || output.tool_calls != observation.tool_calls
            || output.tool_calls != case.expect.tool_calls
            || output.cached_tokens != observation.cached_tokens
            || case
                .expect
                .min_cached_tokens
                .is_some_and(|minimum| output.cached_tokens.is_none_or(|cached| cached < minimum))
            || observation.host_min_available_bytes != Some(memory)
            || memory
                .min(observation.host_available_before_bytes)
                .min(observation.host_available_after_bytes)
                < case.limits.min_host_available_bytes
            || output.ttft_ms > case.limits.ttft_ms
            || output.total_ms > case.limits.total_ms
            || observation.trace != trace
            || trace["reuse_kind"] != case.expect.reuse_kind
            || trace["effective_lane"] != case.expect.effective_lane
            || trace["speculation_active"] != case.expect.speculation_active
            || trace["fallback_reason"]
                != serde_json::to_value(&case.expect.fallback_reason).map_err(|e| e.to_string())?
        {
            return Err(format!(
                "{}: raw accuracy, latency, memory or trace contract failed",
                case.name
            ));
        }
        Ok((output.ttft_ms / case.limits.ttft_ms).max(output.total_ms / case.limits.total_ms))
    }
}
