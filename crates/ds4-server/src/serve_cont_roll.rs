//! Rolling continuous admit table. The engine's `ContDriver::admit` pulls
//! the next pending job while others are still live; one-at-a-time is not
//! the only path.

use std::collections::{HashSet, VecDeque};

use crate::metrics::REJECT_REASON_NAMES;

/// Existing serial-cont 503 body for an engine/governor admit refuse.
/// Rolling must reuse this string; do not invent a new HTTP body.
pub(crate) const CONT_ADMIT_REFUSED: &str = "continuous admission rejected; serial fallback";

#[cfg(test)]
use ds4_core::GovStatus;

/// C `DS4_REJECT_*` / `/metrics` `ds4_requests_rejected_total{reason=}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum RejectReason {
    ClassBudget = 0,
    LiveHeadroom = 1,
    ObsRetry = 2,
    Unsupported = 3,
    Fault = 4,
    DeepPolicy = 5,
    LaneDisabled = 6,
}

impl RejectReason {
    pub(crate) const fn name(self) -> &'static str {
        REJECT_REASON_NAMES[self as usize]
    }
}

/// C `ds4_reject_reason_from_gov`: ADMIT is not a refusal; every other
/// governed status maps 1:1 onto the typed reject family.
#[cfg(test)]
const fn reject_reason_from_gov(status: GovStatus) -> Option<RejectReason> {
    match status {
        GovStatus::Admit => None,
        GovStatus::RefuseClass => Some(RejectReason::ClassBudget),
        GovStatus::RefuseLive => Some(RejectReason::LiveHeadroom),
        GovStatus::RetryObs => Some(RejectReason::ObsRetry),
        GovStatus::Unsupported => Some(RejectReason::Unsupported),
        GovStatus::Fault => Some(RejectReason::Fault),
    }
}

/// Banks already claimed by a prior rolling prepare. The next `prepare_slot`
/// ORs these onto the continuation-hold mask so fork/pin/evict cannot spend
/// a live target or drop protected saturation.
#[derive(Debug, Default)]
pub(crate) struct RollReserve {
    banks: Vec<usize>,
}

impl RollReserve {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record the 1-based `ContAdmit::place_bank` (0 = unset).
    pub(crate) fn note_place(&mut self, place_bank: i32) {
        let Some(bank) = (place_bank > 0)
            .then(|| usize::try_from(place_bank - 1).ok())
            .flatten()
        else {
            return;
        };
        if !self.banks.contains(&bank) {
            self.banks.push(bank);
        }
    }

    pub(crate) fn contains(&self, bank: i32) -> bool {
        usize::try_from(bank)
            .ok()
            .is_some_and(|bank| self.banks.contains(&bank))
    }

    pub(crate) fn protect(&self, hold: &[bool]) -> Vec<bool> {
        let mut protected = hold.to_vec();
        for &bank in &self.banks {
            if bank >= protected.len() {
                protected.resize(bank + 1, false);
            }
            protected[bank] = true;
        }
        protected
    }
}

/// Host-side rolling job ids. `admit` may return another job while one is
/// already generating; `complete` retires a live job.
#[derive(Debug, Default)]
pub(crate) struct ContRoll {
    pending: VecDeque<usize>,
    live: HashSet<usize>,
    done: Vec<usize>,
}

impl ContRoll {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn enqueue(&mut self, user: usize) {
        self.pending.push_back(user);
    }

    /// Pull the next pending job into the live set. Returns `Some` even when
    /// another job is already generating.
    pub(crate) fn admit(&mut self) -> Option<usize> {
        let user = self.pending.pop_front()?;
        self.live.insert(user);
        Some(user)
    }

    pub(crate) fn live_count(&self) -> usize {
        self.live.len()
    }

    pub(crate) fn complete(&mut self, user: usize) {
        if self.live.remove(&user) {
            self.done.push(user);
        }
    }

    pub(crate) fn completed(&self) -> &[usize] {
        &self.done
    }
}

#[cfg(test)]
mod tests {
    use super::{reject_reason_from_gov, ContRoll, GovStatus, RejectReason, CONT_ADMIT_REFUSED};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct FakeJob {
        id: usize,
    }

    #[test]
    fn admits_two_fake_jobs_and_both_complete() {
        let mut roll = ContRoll::new();
        let first_job = FakeJob { id: 1 };
        let second_job = FakeJob { id: 2 };
        roll.enqueue(first_job.id);
        roll.enqueue(second_job.id);

        let first = roll.admit().expect("first job admits");
        assert_eq!(first, first_job.id);
        let second = roll
            .admit()
            .expect("second job admits while first is generating");
        assert_eq!(second, second_job.id);
        assert_eq!(roll.live_count(), 2, "both jobs generate together");
        assert!(roll.admit().is_none());

        roll.complete(first);
        roll.complete(second);
        assert_eq!(roll.completed(), &[first_job.id, second_job.id]);
    }

    #[test]
    fn rolling_reserve_protects_the_live_place_bank() {
        // Given: first rolling admit placed on bank 2 (1-based place_bank = 3)
        let mut reserve = super::RollReserve::new();
        reserve.note_place(3);

        // When: the second admit merges that reserve onto an empty hold mask
        let protected = reserve.protect(&[false, false, false]);

        // Then: only the live target is protected
        assert_eq!(protected, vec![false, false, true]);
    }

    #[test]
    fn rolling_reserve_keeps_hold_saturation() {
        // Given: continuation hold already pins bank 0
        let mut reserve = super::RollReserve::new();
        reserve.note_place(2);

        // When: the second admit ORs reserved banks onto that hold
        let protected = reserve.protect(&[true, false, false]);

        // Then: hold is kept and the live target is also protected
        assert_eq!(protected, vec![true, true, false]);
    }

    #[test]
    fn rolling_reserve_ignores_unset_place_bank() {
        let mut reserve = super::RollReserve::new();
        reserve.note_place(0);
        assert_eq!(reserve.protect(&[false, false]), vec![false, false]);
    }

    #[test]
    fn reject_reason_from_gov_matches_c_serial_cont() {
        // Given: C ds4_reject_reason_from_gov
        // When: each governed status is mapped
        // Then: ADMIT has no tick; refusals use the frozen /metrics names
        assert_eq!(reject_reason_from_gov(GovStatus::Admit), None);
        assert_eq!(
            reject_reason_from_gov(GovStatus::RefuseClass).map(RejectReason::name),
            Some("class_budget")
        );
        assert_eq!(
            reject_reason_from_gov(GovStatus::RefuseLive).map(RejectReason::name),
            Some("live_headroom")
        );
        assert_eq!(
            reject_reason_from_gov(GovStatus::RetryObs).map(RejectReason::name),
            Some("obs_retry")
        );
        assert_eq!(
            reject_reason_from_gov(GovStatus::Unsupported).map(RejectReason::name),
            Some("unsupported")
        );
        assert_eq!(
            reject_reason_from_gov(GovStatus::Fault).map(RejectReason::name),
            Some("fault")
        );
        assert_eq!(RejectReason::DeepPolicy.name(), "deep_policy");
        assert_eq!(RejectReason::LaneDisabled.name(), "lane_disabled");
        assert_eq!(
            CONT_ADMIT_REFUSED,
            "continuous admission rejected; serial fallback"
        );
    }

    fn c_d0b_ledger() -> ds4_core::GovLedger {
        let mut lg = ds4_core::GovLedger::default();
        let mut faults = 0;
        assert!(ds4_core::gov_lease_publish(
            &mut lg,
            ds4_core::GovConsumer::EngineBoot as i32,
            10000,
            10000,
            0,
            &mut faults
        ));
        assert!(ds4_core::gov_lease_publish(
            &mut lg,
            ds4_core::GovConsumer::Prewarm as i32,
            200,
            200,
            0,
            &mut faults
        ));
        assert!(ds4_core::gov_lease_publish(
            &mut lg,
            ds4_core::GovConsumer::BatchBankPlan as i32,
            1200,
            1000,
            0,
            &mut faults
        ));
        assert!(ds4_core::gov_lease_publish(
            &mut lg,
            ds4_core::GovConsumer::SerialSession as i32,
            0,
            0,
            200,
            &mut faults
        ));
        lg.floor_bytes = 600;
        lg.substrate_outstanding = 50;
        lg
    }

    fn bank_claim(proposed: u64, class_limit: u64) -> ds4_core::GovClaim {
        ds4_core::GovClaim {
            requester: ds4_core::GovConsumer::BatchBankPlan as i32,
            memc: 12,
            domain: 0,
            proposed_outstanding: proposed,
            operation_transient: 0,
            class_limit,
        }
    }

    fn host_status(free_bytes: u64, proposed: u64, class_limit: u64) -> GovStatus {
        ds4_core::gov_evaluate(
            &c_d0b_ledger(),
            &ds4_core::MemObservation {
                status: ds4_core::MemObsStatus::Ok,
                source: ds4_core::MemObsSource::MeminfoAvailable,
                free_bytes,
                total_bytes: 20000,
            },
            &bank_claim(proposed, class_limit),
        )
        .status()
    }

    #[test]
    fn host_mem_gov_admits_c_d0b_exact_headroom() {
        let status = host_status(1350, 1500, 2000);
        assert_eq!(reject_reason_from_gov(status), None);
    }

    #[test]
    fn host_mem_gov_refuses_live_one_byte_short() {
        let status = host_status(1349, 1500, 2000);
        assert_eq!(
            reject_reason_from_gov(status),
            Some(RejectReason::LiveHeadroom)
        );
    }

    #[test]
    fn host_mem_gov_refuses_class_before_live() {
        let status = host_status(u64::MAX / 2, 2500, 2000);
        assert_eq!(
            reject_reason_from_gov(status),
            Some(RejectReason::ClassBudget)
        );
    }
}
