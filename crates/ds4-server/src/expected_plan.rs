//! Profiles bind the resolved runtime, including facts learned after native open.
use ds4_core::ResolvedPlan;
use serde_json::Value;
use std::path::Path;

const MAX_PLAN_BYTES: u64 = 64 * 1024;
const KEYS: [&str; 5] = ["family", "backend", "effective", "qualified", "controls"];

pub struct ExpectedPlan {
    value: Value,
}

impl ExpectedPlan {
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("expected serving plan: {e}"))?;
        if file.metadata().map_err(|e| e.to_string())?.len() > MAX_PLAN_BYTES {
            return Err("expected serving plan exceeds 64 KiB".into());
        }
        let value: Value =
            serde_json::from_reader(file).map_err(|e| format!("expected serving plan: {e}"))?;
        Self::parse(value)
    }

    fn parse(value: Value) -> Result<Self, String> {
        if value.as_object().is_none_or(|v| v.len() != KEYS.len() + 1)
            || value
                .get("post_open_quote")
                .is_none_or(|v| !v.is_null() && !v.is_object())
            || ["family", "backend"]
                .iter()
                .any(|key| value[key].as_str().is_none_or(str::is_empty))
            || ["effective", "qualified", "controls"]
                .iter()
                .any(|key| !value[key].is_object())
        {
            return Err(
                "expected serving plan requires family/backend/effective/qualified/controls/post_open_quote".into(),
            );
        }
        Ok(Self { value })
    }

    /// Preflight has not earned resident-memory credits; check policy here.
    pub fn check_preflight(&self, actual: &ResolvedPlan) -> Result<(), String> {
        if !actual.may_listen() {
            return Err("expected serving plan rejected: runtime has configuration errors".into());
        }
        let actual = actual.to_json();
        for key in KEYS {
            let value = if key == "backend" {
                &actual["requested"]["backend"]
            } else {
                &actual[key]
            };
            if self.value[key] != *value {
                return Err(format!(
                    "expected serving plan mismatch: {key}; refusing to listen"
                ));
            }
        }
        Ok(())
    }

    pub fn check(&self, actual: &ResolvedPlan) -> Result<(), String> {
        self.check_preflight(actual)?;
        if self.value["post_open_quote"] != quote(&actual.to_json()) {
            return Err(
                "expected serving plan mismatch: post-open memory quote; refusing to listen".into(),
            );
        }
        Ok(())
    }
}

fn quote(plan: &Value) -> Value {
    let mut quote = plan["quote"].clone();
    if let Some(fields) = quote.as_object_mut() {
        fields.remove("available");
    }
    quote
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_core::{
        resolve_plan, serving_caps, EngineFacts, MaxSeqs, ModelFamily, ServingRequest, Variant,
    };
    use serde_json::json;

    fn actual() -> ResolvedPlan {
        let request = ServingRequest {
            max_seqs: MaxSeqs::Fixed(1),
            ..ServingRequest::default()
        };
        resolve_plan(
            &request,
            Some(serving_caps(
                ModelFamily::Qwen4Exp,
                Variant::Qwen38FlashNext,
            )),
            &EngineFacts::default(),
        )
    }
    fn expected(actual: &ResolvedPlan) -> Value {
        let value = actual.to_json();
        json!({"family":value["family"],"backend":value["requested"]["backend"],"effective":value["effective"],"qualified":value["qualified"],"controls":value["controls"],"post_open_quote":quote(&value)})
    }
    #[test]
    fn rejects_post_open_drift() {
        let actual = actual();
        for (section, key, changed) in [
            ("effective", "max_seqs", json!(2)),
            (
                "effective",
                "mtp_weights",
                json!(!actual.effective.mtp_weights),
            ),
            ("qualified", "banks_n", json!(999)),
        ] {
            let mut value = expected(&actual);
            value[section][key] = changed;
            assert!(
                ExpectedPlan::parse(value).unwrap().check(&actual).is_err(),
                "{section}.{key}"
            );
        }
        let mut value = expected(&actual);
        value["backend"] = json!("cpu");
        assert!(ExpectedPlan::parse(value).unwrap().check(&actual).is_err());
    }
    #[test]
    fn accepts_exact_contract_and_rejects_partial_schema() {
        let actual = actual();
        assert!(ExpectedPlan::parse(expected(&actual))
            .unwrap()
            .check(&actual)
            .is_ok());
        assert!(ExpectedPlan::parse(json!({"effective":{}})).is_err());
    }

    #[test]
    fn quote_comparison_respects_accounting_phase() {
        let mut actual = actual();
        actual.quote = Some(ds4_core::ServingQuote {
            shared_weights: 1,
            per_bank: 2,
            mtp_state: 3,
            scratch: 4,
            checkpoint_pool: 5,
            ple: 6,
            media_reserve: 7,
            floor: 8,
            available: 100,
            banks: 1,
            total: 36,
        });
        let guard = ExpectedPlan::parse(expected(&actual)).unwrap();
        actual.quote.as_mut().unwrap().available = 200;
        assert!(guard.check(&actual).is_ok());
        actual.quote.as_mut().unwrap().shared_weights = 99;
        assert!(guard.check_preflight(&actual).is_ok());
        assert!(guard
            .check(&actual)
            .is_err_and(|e| e.contains("post-open memory quote")));
    }
}
