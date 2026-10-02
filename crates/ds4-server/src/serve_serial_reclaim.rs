//! C serial emergency reclaim (`serial_session_ensure_fit` /
//! `serial_reclaim_rank` / `serial_reclaim_gate`).
//!
//! Before a typed serial refuse, collect idle banks. Never admit a spend
//! that would leave the box under `--mem-floor-gb` (`DS4_MEM_FLOOR_GB`).

use crate::serve_cont_roll::RejectReason;

/// C `ds4_mem_floor_bytes` default: `4ull << 30`.
pub(crate) const DEFAULT_MEM_FLOOR_GB: u64 = 4;

/// C `ds4_mem_floor_bytes` / `--mem-floor-gb` live line, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MemFloor(u64);

/// Observed free bytes at the fit quote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AvailBytes(u64);

/// Serial graph bytes the request still needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NeedBytes(u64);

/// Idle-bank pages the ranker may return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReclaimableBytes(u64);

/// C `fq.headroom_bytes` slack added to the reclaim want.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadroomBytes(u64);

impl MemFloor {
    pub(crate) const fn from_gb(gb: u64) -> Self {
        Self(gb << 30)
    }

    pub(crate) const fn bytes(self) -> u64 {
        self.0
    }

    pub(crate) const fn gb(self) -> u64 {
        self.0 >> 30
    }

    /// C `ds4_mem_floor_bytes`: default 4 GiB; `DS4_MEM_FLOOR_GB` via atol.
    pub(crate) fn from_env_gb(raw: Option<&[u8]>) -> Self {
        let Some(raw) = raw else {
            return Self::from_gb(DEFAULT_MEM_FLOOR_GB);
        };
        if raw.is_empty() {
            return Self::from_gb(DEFAULT_MEM_FLOOR_GB);
        }
        let fv = ds4_sys::libc_atoi(raw);
        if fv < 0 {
            Self::from_gb(DEFAULT_MEM_FLOOR_GB)
        } else {
            Self::from_gb(fv as u64)
        }
    }

    /// CLI `--mem-floor-gb` wins over `DS4_MEM_FLOOR_GB`; both use C atol.
    pub(crate) fn from_cli_or_env(cli: Option<&[u8]>, env: Option<&[u8]>) -> Self {
        match cli {
            Some(cli) => Self::from_env_gb(Some(cli)),
            None => Self::from_env_gb(env),
        }
    }
}

impl AvailBytes {
    pub(crate) const fn from_raw(bytes: u64) -> Self {
        Self(bytes)
    }

    pub(crate) const fn raw(self) -> u64 {
        self.0
    }
}

impl NeedBytes {
    pub(crate) const fn from_raw(bytes: u64) -> Self {
        Self(bytes)
    }

    pub(crate) const fn raw(self) -> u64 {
        self.0
    }
}

impl ReclaimableBytes {
    pub(crate) const fn from_raw(bytes: u64) -> Self {
        Self(bytes)
    }

    pub(crate) const fn raw(self) -> u64 {
        self.0
    }
}

impl HeadroomBytes {
    pub(crate) const fn from_raw(bytes: u64) -> Self {
        Self(bytes)
    }

    pub(crate) const fn raw(self) -> u64 {
        self.0
    }
}

/// Quoted avail/need for the live serial path.
/// C `ds4_session_graph_fit_quote` fields we can host without native trim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SerialFitQuote {
    pub avail: AvailBytes,
    pub need: NeedBytes,
    pub reclaimable: ReclaimableBytes,
    pub headroom: HeadroomBytes,
}

impl SerialFitQuote {
    pub(crate) const fn ask(self, floor: MemFloor) -> SerialReclaimAsk {
        SerialReclaimAsk {
            avail: self.avail,
            need: self.need,
            floor,
            reclaimable: self.reclaimable,
            headroom: self.headroom,
        }
    }
}

/// No native graph-fit quote yet. need=0 so the gate still runs and
/// cannot invent a harder refuse than C (margin, not floor).
pub(crate) const fn unquoted_serial_fit() -> SerialFitQuote {
    SerialFitQuote {
        avail: AvailBytes::from_raw(u64::MAX),
        need: NeedBytes::from_raw(0),
        reclaimable: ReclaimableBytes::from_raw(0),
        headroom: HeadroomBytes::from_raw(0),
    }
}

/// Map C `ds4_session_graph_fit_quote` bytes. `fail_open` means the
/// probe had no budget numbers; treat that as quote failure so the
/// caller keeps the unquoted margin instead of inventing a refuse.
pub(crate) fn serial_fit_from_native(
    need_bytes: u64,
    avail_bytes: u64,
    headroom_bytes: u64,
    reclaimable_bytes: u64,
    fail_open: bool,
) -> Option<SerialFitQuote> {
    if fail_open {
        return None;
    }
    Some(SerialFitQuote {
        avail: AvailBytes::from_raw(avail_bytes),
        need: NeedBytes::from_raw(need_bytes),
        reclaimable: ReclaimableBytes::from_raw(reclaimable_bytes),
        headroom: HeadroomBytes::from_raw(headroom_bytes),
    })
}

/// Test override, then a live quote, else the unquoted margin.
pub(crate) fn resolve_serial_fit(
    configured: Option<SerialFitQuote>,
    live: Option<SerialFitQuote>,
) -> SerialFitQuote {
    configured.or(live).unwrap_or_else(unquoted_serial_fit)
}

/// One serial fit-quote plus what idle banks can still return.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SerialReclaimAsk {
    pub avail: AvailBytes,
    pub need: NeedBytes,
    pub floor: MemFloor,
    pub reclaimable: ReclaimableBytes,
    pub headroom: HeadroomBytes,
}

/// C `serial_reclaim_gate` outcome. Reclaim is attempted before Refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SerialReclaimOutcome {
    Admit {
        reclaimed: u64,
    },
    Refuse {
        reclaimed: u64,
        reason: RejectReason,
    },
}

impl SerialReclaimOutcome {
    pub(crate) const fn reclaimed(self) -> u64 {
        match self {
            Self::Admit { reclaimed } | Self::Refuse { reclaimed, .. } => reclaimed,
        }
    }

    pub(crate) const fn admitted(self) -> bool {
        matches!(self, Self::Admit { .. })
    }
}

/// C `serial_session_ensure_fit` 503 after reclaim still cannot fund the graph.
pub(crate) fn serial_capacity_refuse_msg(prompt_n: i32) -> String {
    format!(
        "Server is temporarily at capacity for a {prompt_n}-token serial request \
         (no session graph fits beside the batch banks); retry shortly"
    )
}

/// One idle bank for C `serial_reclaim_rank`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReclaimBank {
    pub id: u32,
    pub valid: bool,
    pub last_use: u64,
    pub committed: i32,
    pub protected: bool,
    pub superseded: bool,
}

const fn fits_above_floor(avail: AvailBytes, need: NeedBytes, floor: MemFloor) -> bool {
    avail.raw() >= floor.bytes() && avail.raw() - floor.bytes() >= need.raw()
}

const fn serial_deficit(ask: SerialReclaimAsk) -> u64 {
    ask.need
        .raw()
        .saturating_add(ask.floor.bytes())
        .saturating_sub(ask.avail.raw())
}

/// C `serial_session_ensure_fit` collect want: deficit + headroom.
pub(crate) fn serial_reclaim_want(ask: SerialReclaimAsk) -> u64 {
    if fits_above_floor(ask.avail, ask.need, ask.floor) {
        return 0;
    }
    serial_deficit(ask).saturating_add(ask.headroom.raw())
}

/// C `serial_reclaim_gate`: reclaim idle pages before a typed refuse.
/// Usable = avail − floor; the floor is never spent.
pub(crate) fn serial_reclaim_gate(ask: SerialReclaimAsk) -> SerialReclaimOutcome {
    if fits_above_floor(ask.avail, ask.need, ask.floor) {
        return SerialReclaimOutcome::Admit { reclaimed: 0 };
    }
    let want = serial_reclaim_want(ask);
    let reclaimed = want.min(ask.reclaimable.raw());
    let after = AvailBytes::from_raw(ask.avail.raw().saturating_add(reclaimed));
    if fits_above_floor(after, ask.need, ask.floor) {
        SerialReclaimOutcome::Admit { reclaimed }
    } else {
        SerialReclaimOutcome::Refuse {
            reclaimed,
            reason: RejectReason::LiveHeadroom,
        }
    }
}

/// C `serial_reclaim_rank`: cheapest warm value first. Hard-excludes
/// grace/pin-protected banks and deep records (`committed >= warm_pin_min`).
pub(crate) fn serial_reclaim_rank(banks: &[ReclaimBank], warm_pin_min: i32) -> Vec<u32> {
    let mut no_value = Vec::new();
    let mut superseded = Vec::new();
    let mut plain = Vec::new();
    for bank in banks {
        if bank.protected {
            continue;
        }
        if !bank.valid {
            no_value.push(bank.id);
            continue;
        }
        if warm_pin_min > 0 && bank.committed >= warm_pin_min {
            continue;
        }
        if bank.superseded {
            superseded.push(*bank);
        } else {
            plain.push(*bank);
        }
    }
    superseded.sort_by_key(|bank| bank.last_use);
    plain.sort_by_key(|bank| bank.last_use);
    no_value.extend(superseded.iter().map(|bank| bank.id));
    no_value.extend(plain.iter().map(|bank| bank.id));
    no_value
}
