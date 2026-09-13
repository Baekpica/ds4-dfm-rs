//! Process-global speculative inference counters via the native bridge.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecMetrics {
    pub drafts: u64,
    pub hits: u64,
    pub quench: u64,
}

/// Read cumulative native counters, including activity in live requests.
/// Each counter is atomic; the three fields are not a coherent transaction.
pub fn snapshot_spec() -> SpecMetrics {
    let mut raw = ds4_sys::ds4_bridge_spec_metrics {
        drafts: 0,
        hits: 0,
        quench: 0,
    };
    // SAFETY: the bridge fills this call-scoped POD and retains no pointers.
    unsafe { ds4_sys::ds4_bridge_spec_snapshot(&mut raw) };
    SpecMetrics {
        drafts: raw.drafts,
        hits: raw.hits,
        quench: raw.quench,
    }
}
