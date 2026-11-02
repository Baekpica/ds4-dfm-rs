//! Memory policy values. The host decides them and native applies them at
//! its live allocation checks (`ds4_host_mem_policy`). Native used to read
//! each from the environment at the check, and the server published its
//! floor through `set_var`. Defaults and precedence are the C hosts' reads.

use ds4_sys::ds4_host_mem_policy;

use crate::mem_gov::{gov_mode_name, gov_mode_parse, GovMode};
use crate::ModelFamily;

const FLOOR_ENV: &str = "DS4_MEM_FLOOR_GB";
const FIT_ENV: &str = "DS4_SESSION_GRAPH_FIT";
const SESSION_HEADROOM_ENV: &str = "DS4_SESSION_GRAPH_HEADROOM_MB";
const BATCH_HEADROOM_ENV: &str = "DS4_BATCH_FIT_HEADROOM_MB";
const BATCH_DERIVED_ENV: &str = "DS4_BATCH_FIT_HEADROOM_DERIVED";
const BATCH_BURST_ENV: &str = "DS4_BATCH_FIT_BURST_MB";
const GOV_ENV: &str = "DS4_MEMGOV";
/// `ds4_gov_consumer` order.
const GOV_KNOBS: [&str; 5] = [
    "DS4_MEMGOV_BOOT",
    "DS4_MEMGOV_PREWARM",
    "DS4_MEMGOV_BANK",
    "DS4_MEMGOV_SERIAL",
    "DS4_MEMGOV_STATIC",
];

const GIB_SHIFT: u32 = 30;
const MIB_SHIFT: u32 = 20;
const DEFAULT_FLOOR_GB: u64 = crate::serving::DEFAULT_MEM_FLOOR_GB;
const SESSION_HEADROOM_MB: u64 = 1024;
/// `DS4_BATCH_FIT_HEADROOM_DERIVED=0`: the pre-derivation constant.
const STATIC_BATCH_HEADROOM_MB: u64 = 6144;
const BURST_MB: u64 = 2048;
/// Ling-3.1 preallocates its graphs; first requests grew 0.41-0.48 GiB.
const LING31_BURST_MB: u64 = 640;

/// `floor_gb` and `budget_mb` are the host's flags when it has them.
pub(crate) fn resolve(
    family: ModelFamily,
    floor_gb: Option<u64>,
    budget_mb: Option<u64>,
) -> ds4_host_mem_policy {
    let floor_gb = floor_gb
        .or_else(|| env_atol(FLOOR_ENV))
        .unwrap_or(DEFAULT_FLOOR_GB);
    let fit_off = std::env::var_os(FIT_ENV).is_some_and(|v| v == "0");
    let mut policy = ds4_host_mem_policy {
        floor_bytes: floor_gb << GIB_SHIFT,
        budget_bytes: budget_mb.unwrap_or(0) << MIB_SHIFT,
        session_graph_headroom: env_atol(SESSION_HEADROOM_ENV).unwrap_or(SESSION_HEADROOM_MB)
            << MIB_SHIFT,
        batch_fit_headroom: 0,
        batch_fit_burst: 0,
        batch_fit_derived: 0,
        session_graph_fit: u32::from(!fit_off),
        gov_mode: gov_modes(),
        reserved: [0; 3],
    };

    // A pin rules; the static kill switch next; else floor + burst.
    if let Some(mb) = env_atol(BATCH_HEADROOM_ENV) {
        policy.batch_fit_headroom = mb << MIB_SHIFT;
    } else if std::env::var_os(BATCH_DERIVED_ENV).is_some_and(|v| v == "0") {
        policy.batch_fit_headroom = STATIC_BATCH_HEADROOM_MB << MIB_SHIFT;
    } else {
        let burst = match family {
            ModelFamily::Ling31Vq => LING31_BURST_MB,
            _ => BURST_MB,
        };
        policy.batch_fit_derived = 1;
        policy.batch_fit_burst = env_atol(BATCH_BURST_ENV).unwrap_or(burst) << MIB_SHIFT;
    }
    policy
}

/// C `atol` on a set, non-empty variable; a negative value keeps the default.
fn env_atol(name: &str) -> Option<u64> {
    let raw = std::env::var_os(name).filter(|v| !v.is_empty())?;
    u64::try_from(ds4_sys::libc_atoi(raw.as_encoded_bytes())).ok()
}

/// `DS4_MEMGOV` sets every consumer and a consumer's own knob wins; the
/// default is enforce. Unrecognized values warn as ds4_gov_modes_init did.
fn gov_modes() -> [u8; GOV_KNOBS.len()] {
    let all = std::env::var(GOV_ENV)
        .ok()
        .filter(|v| !v.is_empty())
        .and_then(|v| {
            let mode = gov_mode_parse(&v);
            if mode.is_none() {
                eprintln!(
                    "ds4: memgov: {GOV_ENV}={v} unrecognized (off|observe|enforce); \
                     keeping per-family defaults"
                );
            }
            mode
        });
    GOV_KNOBS.map(|knob| {
        let mut mode = all.unwrap_or(GovMode::Enforce);
        let Some(value) = std::env::var(knob).ok().filter(|v| !v.is_empty()) else {
            return mode as u8;
        };
        match gov_mode_parse(&value) {
            Some(parsed) => mode = parsed,
            None => eprintln!(
                "ds4: memgov: {knob}={value} unrecognized (off|observe|enforce); keeping {}",
                gov_mode_name(mode)
            ),
        }
        mode as u8
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no knobs set the values equal the native constants.
    #[test]
    fn defaults_match_native() {
        let names = [
            FLOOR_ENV,
            FIT_ENV,
            SESSION_HEADROOM_ENV,
            BATCH_HEADROOM_ENV,
            BATCH_DERIVED_ENV,
            BATCH_BURST_ENV,
            GOV_ENV,
        ];
        if names
            .iter()
            .chain(&GOV_KNOBS)
            .any(|n| std::env::var_os(n).is_some())
        {
            return; // An operator environment; the defaults are not observable.
        }
        let policy = resolve(ModelFamily::Qwen4Exp, None, None);
        assert_eq!(policy.floor_bytes, 4 << 30);
        assert_eq!(policy.budget_bytes, 0);
        assert_eq!(policy.session_graph_headroom, 1024 << 20);
        assert_eq!(policy.session_graph_fit, 1);
        assert_eq!(
            (policy.batch_fit_derived, policy.batch_fit_burst),
            (1, 2048 << 20)
        );
        assert_eq!(policy.gov_mode, [GovMode::Enforce as u8; 5]);
        let ling = resolve(ModelFamily::Ling31Vq, Some(2), Some(110_000));
        assert_eq!(
            (ling.floor_bytes, ling.batch_fit_burst),
            (2 << 30, 640 << 20)
        );
        assert_eq!(ling.budget_bytes, 110_000 << 20);
    }
}
