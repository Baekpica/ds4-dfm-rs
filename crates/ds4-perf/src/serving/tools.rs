//! Streamed function calls need an argument contract; text equality is insufficient.
use super::*;
use std::collections::BTreeMap;

const MAX_CALLS: u64 = 32;
const MAX_ARGUMENT_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Call {
    name: String,
    arguments: Value,
}
impl Call {
    pub(super) fn valid(&self) -> bool {
        !self.name.is_empty() && self.arguments.is_object()
    }
    pub(super) fn declared(&self, request: &Value) -> bool {
        request["tools"].as_array().is_some_and(|tools| {
            tools
                .iter()
                .any(|tool| tool["type"] == "function" && tool["function"]["name"] == self.name)
        })
    }
}

#[derive(Default)]
struct Partial {
    id: Option<String>,
    name: String,
    arguments: String,
}

#[derive(Default)]
pub(super) struct Calls(BTreeMap<u64, Partial>);
impl Calls {
    pub(super) fn add(&mut self, delta: &Value) -> Result<bool, String> {
        let Some(value) = delta.get("tool_calls") else {
            return Ok(false);
        };
        let calls = value.as_array().ok_or("tool_calls delta is not an array")?;
        let mut generated = false;
        for call in calls {
            let index = call["index"]
                .as_u64()
                .filter(|i| *i < MAX_CALLS)
                .ok_or("invalid tool call index")?;
            if call.get("type").is_some_and(|kind| kind != "function") {
                return Err("unsupported tool call type".into());
            }
            let partial = self.0.entry(index).or_default();
            if let Some(id) = call.get("id") {
                let id = id
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .ok_or("invalid tool call id")?;
                if partial.id.as_ref().is_some_and(|previous| previous != id) {
                    return Err("tool call id changed during stream".into());
                }
                partial.id = Some(id.into());
            }
            let function = call
                .get("function")
                .filter(|v| v.is_object())
                .ok_or("tool call lacks function delta")?;
            for (key, output) in [
                ("name", &mut partial.name),
                ("arguments", &mut partial.arguments),
            ] {
                if let Some(value) = function.get(key) {
                    let text = value
                        .as_str()
                        .ok_or("tool name/arguments delta must be text")?;
                    generated |= !text.is_empty();
                    output.push_str(text);
                    if output.len() > MAX_ARGUMENT_BYTES {
                        return Err("tool call exceeds argument budget".into());
                    }
                }
            }
        }
        Ok(generated)
    }

    pub(super) fn finish(self) -> Result<Vec<Call>, String> {
        let mut ids = BTreeSet::new();
        self.0
            .into_iter()
            .enumerate()
            .map(|(position, (index, partial))| {
                if index != position as u64
                    || !ids.insert(partial.id.ok_or("tool call id missing")?)
                {
                    return Err("tool call indexes or ids are not unique/contiguous".into());
                }
                let call = Call {
                    name: partial.name,
                    arguments: serde_json::from_str(&partial.arguments)
                        .map_err(|e| format!("incomplete tool arguments: {e}"))?,
                };
                if !call.valid() {
                    return Err("tool call needs a name and JSON object arguments".into());
                }
                Ok(call)
            })
            .collect()
    }
}

pub(super) fn valid_finish(content: &str, reason: &str, calls: &[Call]) -> bool {
    if calls.is_empty() {
        !content.is_empty() && matches!(reason, "stop" | "length")
    } else {
        reason == "tool_calls" && calls.iter().all(Call::valid)
    }
}
