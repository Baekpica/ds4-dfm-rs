//! C continuous prefill-chunk interleave (R4 overlap-lite) on the rolling
//! scheduler. Env names match C and must not change.

use std::cell::Cell;
use std::ffi::OsStr;

/// C `DS4_CONT_PREFILL_CHUNK`. Do not rename.
pub(crate) const ENV_PREFILL_CHUNK: &str = "DS4_CONT_PREFILL_CHUNK";
/// C `DS4_CONT_PREFILL_CHUNK_LIVE`. Do not rename.
pub(crate) const ENV_PREFILL_CHUNK_LIVE: &str = "DS4_CONT_PREFILL_CHUNK_LIVE";

pub(crate) const DEFAULT_PREFILL_CHUNK: u32 = 4096;
pub(crate) const DEFAULT_PREFILL_CHUNK_LIVE: u32 = 512;
/// One definition with the serving plan, so `--print-plan` cannot advertise
/// a yield this policy then caps.
const PREFILL_FENCE: u32 = ds4_core::PREFILL_CHUNK_FENCE;

/// C `bg_prefill_chunk_tokens` / `bg_prefill_chunk_live_tokens`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefillChunkPolicy {
    pub boot: u32,
    pub live: u32,
}

impl PrefillChunkPolicy {
    pub(crate) fn from_env() -> Self {
        let nofence =
            std::env::var_os("DS4_CONT_PREFILL_NOFENCE").as_deref() == Some(OsStr::new("1"));
        Self::from_os(
            std::env::var_os(ENV_PREFILL_CHUNK).as_deref(),
            std::env::var_os(ENV_PREFILL_CHUNK_LIVE).as_deref(),
            nofence,
        )
    }

    pub(crate) fn from_os(boot: Option<&OsStr>, live: Option<&OsStr>, nofence: bool) -> Self {
        Self::from_raw(
            parse_chunk(boot, DEFAULT_PREFILL_CHUNK),
            parse_chunk(live, DEFAULT_PREFILL_CHUNK_LIVE),
            nofence,
        )
    }

    pub(crate) fn from_raw(boot: i32, live: i32, nofence: bool) -> Self {
        let mut boot = boot.max(0) as u32;
        let mut live = live.max(0) as u32;
        if boot > PREFILL_FENCE && !nofence {
            boot = PREFILL_FENCE;
        }
        if live > boot {
            live = boot;
        }
        Self { boot, live }
    }

    /// C: `W_boot != 0 && W_live != 0`.
    pub(crate) const fn interleave(self) -> bool {
        self.boot != 0 && self.live != 0
    }

    /// C `W_eff`: LIVE width only while someone is decoding; else boot.
    /// `0` means one-shot (whole remaining suffix).
    pub(crate) const fn chunk_width(self, live_decode: bool) -> u32 {
        if self.boot == 0 {
            0
        } else if self.interleave() && live_decode {
            self.live
        } else {
            self.boot
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RollPhase {
    Prefill { remaining: u32 },
    Decode { remaining: u32 },
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TickOp {
    Prefill { user: usize, tokens: u32 },
    Decode { user: usize },
}

/// One C outer-loop pass: advance pending prefills, then step live decode.
/// Interleave runs one chunk then falls through; LIVE=0 drains at boot width.
pub(crate) fn tick_roll_prefill(
    policy: PrefillChunkPolicy,
    jobs: &mut [(usize, RollPhase)],
) -> Vec<TickOp> {
    let mut ops = Vec::new();
    let live_decode = jobs
        .iter()
        .any(|(_, phase)| matches!(phase, RollPhase::Decode { remaining } if *remaining > 0));
    let mut idx = 0;
    while idx < jobs.len() {
        let RollPhase::Prefill { remaining } = jobs[idx].1 else {
            idx += 1;
            continue;
        };
        if remaining == 0 {
            jobs[idx].1 = RollPhase::Done;
            idx += 1;
            continue;
        }
        let width = policy.chunk_width(live_decode);
        let n = if width == 0 {
            remaining
        } else {
            remaining.min(width)
        };
        let user = jobs[idx].0;
        let left = remaining - n;
        jobs[idx].1 = if left == 0 {
            RollPhase::Done
        } else {
            RollPhase::Prefill { remaining: left }
        };
        ops.push(TickOp::Prefill { user, tokens: n });
        if policy.interleave() {
            break;
        }
        if left > 0 {
            continue;
        }
        idx += 1;
    }
    for (user, phase) in jobs.iter_mut() {
        let RollPhase::Decode { remaining } = *phase else {
            continue;
        };
        if remaining == 0 {
            *phase = RollPhase::Done;
            continue;
        }
        let left = remaining - 1;
        *phase = if left == 0 {
            RollPhase::Done
        } else {
            RollPhase::Decode { remaining: left }
        };
        ops.push(TickOp::Decode { user: *user });
    }
    ops
}

thread_local! {
    static OWNER_TICK_CALLS: Cell<u32> = const { Cell::new(0) };
}

pub(crate) fn owner_tick_pair(
    policy: PrefillChunkPolicy,
    live_decode_remaining: u32,
    prefill_remaining: u32,
) -> Vec<TickOp> {
    OWNER_TICK_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    let mut jobs = [
        (
            1,
            RollPhase::Decode {
                remaining: live_decode_remaining,
            },
        ),
        (
            2,
            RollPhase::Prefill {
                remaining: prefill_remaining,
            },
        ),
    ];
    tick_roll_prefill(policy, &mut jobs)
}

#[cfg(test)]
pub(crate) fn owner_tick_call_count() -> u32 {
    OWNER_TICK_CALLS.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_owner_tick_call_count() {
    OWNER_TICK_CALLS.with(|calls| calls.set(0));
}

fn parse_chunk(value: Option<&OsStr>, default: u32) -> i32 {
    let Some(value) = value else {
        return default as i32;
    };
    let bytes = os_bytes(value);
    if bytes.is_empty() {
        return default as i32;
    }
    c_atoi(bytes)
}

fn os_bytes(value: &OsStr) -> &[u8] {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        value.as_bytes()
    }
    #[cfg(not(unix))]
    {
        value.as_encoded_bytes()
    }
}

fn c_atoi(bytes: &[u8]) -> i32 {
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let mut neg = false;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        neg = bytes[i] == b'-';
        i += 1;
    }
    let mut n: i32 = 0;
    let mut saw_digit = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        saw_digit = true;
        n = n
            .saturating_mul(10)
            .saturating_add(i32::from(bytes[i] - b'0'));
        i += 1;
    }
    if !saw_digit {
        return 0;
    }
    if neg {
        -n
    } else {
        n
    }
}
