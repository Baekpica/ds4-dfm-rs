//! Inc 5a/5b/5c continuation registry from `ds4_server.c` at v0.6.3-dfm.
//! Host-owned: publish / resolve / hold / pin / TTL / bank claim.
//! Native session still executes prefill; this decides reuse vs 409/503.

use crate::route::Api;
use std::sync::OnceLock;
use std::time::Instant;

pub const CONT_REGISTRY_MAX_DEFAULT: i32 = 64;
pub const CONT_GRACE_S: f64 = 60.0;
pub const CONT_TTL_S: f64 = 300.0;
pub const CONT_PIN_DEADLINE_S: f64 = 60.0;
pub const CONT_HOLD_SHED_S: f64 = 5.0;

pub(crate) fn monotonic_now() -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContState {
    LiveFrontier = 0,
    ReplayOnly = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContOwner {
    SerialSession = 0,
    BatchBank = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContPin(u64);

#[derive(Clone, Debug)]
pub struct ContRecord {
    record_id: u64,
    pub state: ContState,
    pub owner: ContOwner,
    pub protocol: u8,
    pub owner_id: i32,
    pub owner_gen: u64,
    pub frontier: i32,
    pub call_ids: Vec<String>,
    pub publish_time: f64,
    pub hard_refs: i32,
    pub pin_expiry: f64,
}

#[derive(Clone, Debug)]
pub struct ContRegistry {
    records: Vec<ContRecord>,
    next_record_id: u64,
    pub max_records: i32,
    pub grace_s: f64,
    pub ttl_s: f64,
    pub pin_deadline_s: f64,
    pub hold_shed_s: f64,
    pub serial_live: Option<usize>,
}

impl Default for ContRegistry {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            next_record_id: 1,
            max_records: CONT_REGISTRY_MAX_DEFAULT,
            grace_s: CONT_GRACE_S,
            ttl_s: CONT_TTL_S,
            pin_deadline_s: CONT_PIN_DEADLINE_S,
            hold_shed_s: CONT_HOLD_SHED_S,
            serial_live: None,
        }
    }
}

impl ContRegistry {
    pub fn n_records(&self) -> i32 {
        self.records.len() as i32
    }

    pub fn n_live(&self) -> i32 {
        self.records
            .iter()
            .filter(|r| r.state == ContState::LiveFrontier)
            .count() as i32
    }

    pub fn live_ids(&self, proto: Api) -> Vec<String> {
        let mut out = Vec::new();
        for rec in &self.records {
            if rec.state != ContState::LiveFrontier || rec.protocol != proto as u8 {
                continue;
            }
            for id in &rec.call_ids {
                if !id.is_empty() && !out.iter().any(|x| x == id) {
                    out.push(id.clone());
                }
            }
        }
        out
    }

    fn key(proto: u8, id: &str) -> String {
        let mut id = id;
        if id.len() > 94 {
            let mut end = 94;
            while !id.is_char_boundary(end) {
                end -= 1;
            }
            id = &id[..end];
        }
        let mut s = String::with_capacity(2 + id.len());
        s.push(char::from(b'0' + proto));
        s.push('\u{001f}');
        s.push_str(id);
        s
    }

    fn find_idx(&self, proto: u8, id: &str) -> Option<usize> {
        if id.is_empty() {
            return None;
        }
        let k = Self::key(proto, id);
        self.records
            .iter()
            .position(|r| r.call_ids.iter().any(|cid| Self::key(r.protocol, cid) == k))
    }

    fn set_eq(a: &[String], b: &[String]) -> bool {
        a.len() == b.len() && a.iter().all(|id| b.iter().any(|x| x == id))
    }

    fn demote_idx(&mut self, idx: usize) {
        if self.records[idx].state != ContState::LiveFrontier {
            return;
        }
        self.records[idx].state = ContState::ReplayOnly;
        if self.serial_live == Some(idx) {
            self.serial_live = None;
        }
    }

    fn prune(&mut self) {
        while self.n_records() > self.max_records {
            let oldest = self
                .records
                .iter()
                .enumerate()
                .rev()
                .find(|(_, r)| r.state == ContState::ReplayOnly && r.hard_refs <= 0);
            match oldest {
                Some((i, _)) => {
                    self.remove_idx(i);
                }
                None => break,
            }
        }
    }

    fn remove_idx(&mut self, idx: usize) {
        self.demote_idx(idx);
        self.records.remove(idx);
        if let Some(s) = self.serial_live {
            if s == idx {
                self.serial_live = None;
            } else if s > idx {
                self.serial_live = Some(s - 1);
            }
        }
    }

    pub fn expire(&mut self, now: f64) {
        if self.ttl_s <= 0.0 || self.n_live() == 0 {
            return;
        }
        let mut i = 0;
        while i < self.records.len() {
            if self.records[i].state == ContState::LiveFrontier
                && now - self.records[i].publish_time > self.ttl_s
            {
                self.demote_idx(i);
            }
            i += 1;
        }
    }

    pub fn publish(
        &mut self,
        proto: Api,
        ids: &[String],
        owner: ContOwner,
        owner_id: i32,
        gen: u64,
        frontier: i32,
        now: f64,
    ) {
        if ids.is_empty() || gen == 0 || frontier <= 0 {
            return;
        }
        let ids: Vec<String> = {
            let mut out = Vec::new();
            for id in ids {
                if !id.is_empty() && !out.iter().any(|x| x == id) {
                    out.push(id.clone());
                }
            }
            out
        };
        if ids.is_empty() {
            return;
        }
        if self.max_records <= 0 {
            self.max_records = CONT_REGISTRY_MAX_DEFAULT;
        }
        match owner {
            ContOwner::SerialSession => {
                if let Some(i) = self.serial_live {
                    self.demote_idx(i);
                }
            }
            ContOwner::BatchBank => {
                if let Some(i) = self.records.iter().position(|r| {
                    r.state == ContState::LiveFrontier
                        && r.owner == ContOwner::BatchBank
                        && r.owner_id == owner_id
                }) {
                    self.demote_idx(i);
                }
            }
        }
        let record_id = self.next_record_id;
        self.next_record_id = self.next_record_id.wrapping_add(1).max(1);
        let rec = ContRecord {
            record_id,
            state: ContState::LiveFrontier,
            owner,
            protocol: proto as u8,
            owner_id: if owner == ContOwner::BatchBank {
                owner_id
            } else {
                0
            },
            owner_gen: gen,
            frontier,
            call_ids: ids,
            publish_time: now,
            hard_refs: 0,
            pin_expiry: 0.0,
        };
        self.records.insert(0, rec);
        if owner == ContOwner::SerialSession {
            self.serial_live = Some(0);
        } else if let Some(s) = self.serial_live {
            self.serial_live = Some(s + 1);
        }
        self.prune();
    }

    pub fn publish_serial(
        &mut self,
        proto: Api,
        ids: &[String],
        gen: u64,
        frontier: i32,
        now: f64,
    ) {
        self.publish(proto, ids, ContOwner::SerialSession, 0, gen, frontier, now);
    }

    pub fn publish_bank(
        &mut self,
        proto: Api,
        ids: &[String],
        bank: i32,
        gen: u64,
        frontier: i32,
        now: f64,
    ) {
        if bank < 0 {
            return;
        }
        self.publish(proto, ids, ContOwner::BatchBank, bank, gen, frontier, now);
    }

    pub fn demote_serial(&mut self) {
        if let Some(i) = self.serial_live {
            self.demote_idx(i);
        }
    }

    pub fn live_has_id(&mut self, proto: Api, id: &str, now: f64) -> bool {
        self.expire(now);
        self.find_idx(proto as u8, id)
            .map(|i| self.records[i].state == ContState::LiveFrontier)
            .unwrap_or(false)
    }

    pub fn id_known(&self, id: &str) -> bool {
        for proto in [
            Api::Openai as u8,
            Api::Anthropic as u8,
            Api::Responses as u8,
        ] {
            if self.find_idx(proto, id).is_some() {
                return true;
            }
        }
        false
    }

    pub fn resolve_serial(
        &mut self,
        proto: Api,
        ids: &[String],
        session_gen: u64,
        live_pos: i32,
        now: f64,
    ) -> bool {
        if ids.is_empty() || session_gen == 0 {
            return false;
        }
        self.expire(now);
        let Some(i) = self.find_idx(proto as u8, &ids[0]) else {
            return false;
        };
        let rec = &self.records[i];
        rec.state == ContState::LiveFrontier
            && rec.owner == ContOwner::SerialSession
            && rec.protocol == proto as u8
            && Self::set_eq(&rec.call_ids, ids)
            && rec.owner_gen == session_gen
            && rec.frontier == live_pos
    }

    pub fn bank_claim(&mut self, proto: Api, ids: &[String], now: f64) -> Option<(i32, u64, i32)> {
        if ids.is_empty() {
            return None;
        }
        self.expire(now);
        let i = self.find_idx(proto as u8, &ids[0])?;
        let rec = &self.records[i];
        if rec.state == ContState::LiveFrontier
            && rec.owner == ContOwner::BatchBank
            && rec.protocol == proto as u8
            && Self::set_eq(&rec.call_ids, ids)
        {
            Some((rec.owner_id, rec.owner_gen, rec.frontier))
        } else {
            None
        }
    }

    /// C `cont_record_bank_protects`: an unverified native reference fails
    /// closed for eviction, while a proven generation/frontier mismatch does not.
    pub fn bank_protected(&self, bank: i32, live: Option<(u64, i32)>, now: f64) -> bool {
        if bank < 0 {
            return false;
        }
        self.records.iter().any(|record| {
            if record.state != ContState::LiveFrontier
                || record.owner != ContOwner::BatchBank
                || record.owner_id != bank
            {
                return false;
            }
            if let Some((generation, frontier)) = live {
                if generation == 0
                    || frontier <= 0
                    || record.owner_gen != generation
                    || record.frontier != frontier
                {
                    return false;
                }
            }
            let in_grace = self.grace_s > 0.0 && now - record.publish_time < self.grace_s;
            let pinned =
                record.hard_refs > 0 && self.pin_deadline_s > 0.0 && now < record.pin_expiry;
            in_grace || pinned
        })
    }

    pub fn bank_hold_retry(&self, bank: i32, live: Option<(u64, i32)>, now: f64) -> Option<i32> {
        if !self.bank_protected(bank, live, now) {
            return None;
        }
        let record = self.records.iter().find(|record| {
            record.state == ContState::LiveFrontier
                && record.owner == ContOwner::BatchBank
                && record.owner_id == bank
                && live.is_none_or(|(generation, frontier)| {
                    generation != 0
                        && frontier > 0
                        && record.owner_gen == generation
                        && record.frontier == frontier
                })
        })?;
        let grace_left = if self.grace_s > 0.0 {
            self.grace_s - (now - record.publish_time)
        } else {
            0.0
        };
        let pin_left =
            if record.hard_refs > 0 && self.pin_deadline_s > 0.0 && now < record.pin_expiry {
                record.pin_expiry - now
            } else {
                0.0
            };
        let left = grace_left.max(pin_left);
        let retry = (left + 0.999) as i32;
        Some(retry.max(1))
    }

    pub fn pin_live(&mut self, proto: Api, id: &str, now: f64) -> Option<ContPin> {
        self.expire(now);
        let i = self.find_idx(proto as u8, id)?;
        if self.records[i].state != ContState::LiveFrontier {
            return None;
        }
        self.records[i].hard_refs += 1;
        if self.pin_deadline_s > 0.0 {
            let expiry = now + self.pin_deadline_s;
            if expiry > self.records[i].pin_expiry {
                self.records[i].pin_expiry = expiry;
            }
        }
        Some(ContPin(self.records[i].record_id))
    }

    pub fn pin_owner(&self, pin: ContPin) -> Option<ContOwner> {
        self.records
            .iter()
            .find(|record| record.record_id == pin.0)
            .map(|record| record.owner)
    }

    pub fn unpin(&mut self, pin: ContPin) {
        if let Some(rec) = self
            .records
            .iter_mut()
            .find(|record| record.record_id == pin.0)
        {
            if rec.hard_refs > 0 {
                rec.hard_refs -= 1;
            }
        }
    }

    pub fn serial_hold(&mut self, proto: Api, req_ids: &[String], now: f64) -> Option<i32> {
        self.expire(now);
        let Some(i) = self.serial_live else {
            return None;
        };
        let rec = &self.records[i];
        if rec.protocol == proto as u8
            && Self::set_eq(&rec.call_ids, req_ids)
            && !req_ids.is_empty()
        {
            return None;
        }
        let shed_w = if self.hold_shed_s < self.grace_s {
            self.hold_shed_s
        } else {
            self.grace_s
        };
        let shed_left = if shed_w > 0.0 {
            shed_w - (now - rec.publish_time)
        } else {
            0.0
        };
        let pinned = rec.hard_refs > 0 && self.pin_deadline_s > 0.0 && now < rec.pin_expiry;
        if shed_left <= 0.0 && !pinned {
            return None;
        }
        let mut left = shed_left;
        if pinned && rec.pin_expiry - now > left {
            left = rec.pin_expiry - now;
        }
        let mut retry = (left + 0.999) as i32;
        if retry < 1 {
            retry = 1;
        }
        Some(retry)
    }

    pub fn serial_live_hard_refs(&self) -> i32 {
        self.serial_live
            .and_then(|i| self.records.get(i))
            .map(|r| r.hard_refs)
            .unwrap_or(0)
    }

    pub fn serial_live_state(&self) -> Option<ContState> {
        self.serial_live
            .and_then(|i| self.records.get(i))
            .map(|r| r.state)
    }

    pub fn set_serial_publish_time(&mut self, t: f64) {
        if let Some(i) = self.serial_live {
            self.records[i].publish_time = t;
        }
    }

    pub fn set_serial_pin_expiry(&mut self, t: f64) {
        if let Some(i) = self.serial_live {
            self.records[i].pin_expiry = t;
        }
    }

    pub fn rewind_live_publish(&mut self, delta: f64) {
        for rec in &mut self.records {
            if rec.state == ContState::LiveFrontier {
                rec.publish_time -= delta;
            }
        }
    }
}

fn push_msg_tool_ids(ids: &mut Vec<String>, m: &crate::parse::ChatMsg) {
    if !m.tool_call_id.is_empty() && !ids.iter().any(|x| x == &m.tool_call_id) {
        ids.push(m.tool_call_id.clone());
    }
    for id in &m.tool_call_ids {
        if !id.is_empty() && !ids.iter().any(|x| x == id) {
            ids.push(id.clone());
        }
    }
}

fn anthropic_tool_result_tail(m: &crate::parse::ChatMsg) -> bool {
    m.role == "user" && (!m.tool_call_id.is_empty() || !m.tool_call_ids.is_empty())
}

/// Call ids C `*_prepare_live_continuation` would bind (not the full history).
pub fn live_tool_result_ids(api: Api, messages: &[crate::parse::ChatMsg]) -> Vec<String> {
    match api {
        Api::Anthropic => {
            let mut tail_end = messages.len();
            while tail_end > 0 && crate::render::role_is_system(&messages[tail_end - 1].role) {
                tail_end -= 1;
            }
            let mut tail_start = tail_end;
            while tail_start > 0 && anthropic_tool_result_tail(&messages[tail_start - 1]) {
                tail_start -= 1;
            }
            if tail_start == tail_end {
                return Vec::new();
            }
            let mut ids = Vec::new();
            for m in &messages[tail_start..tail_end] {
                push_msg_tool_ids(&mut ids, m);
            }
            ids
        }
        Api::Responses => {
            let mut tail_start = messages.len();
            while tail_start > 0 {
                let role = messages[tail_start - 1].role.as_str();
                if role != "tool" && role != "function" {
                    break;
                }
                tail_start -= 1;
            }
            if tail_start == messages.len() {
                return Vec::new();
            }
            let mut ids = Vec::new();
            if tail_start > 0 {
                let assistant = &messages[tail_start - 1];
                if assistant.role != "assistant" || assistant.calls.is_empty() {
                    return Vec::new();
                }
                for c in &assistant.calls {
                    if !c.id.is_empty() && !ids.iter().any(|x| x == &c.id) {
                        ids.push(c.id.clone());
                    }
                }
                return ids;
            }
            for m in &messages[tail_start..] {
                push_msg_tool_ids(&mut ids, m);
            }
            ids
        }
        Api::Openai => Vec::new(),
    }
}

/// C `cont_bank_continuation_admit` equality: a live claim stays on its
/// bank only when generation and frontier still match. Any miss is the
/// protocol-native 409 full-replay surface.
pub fn place_bank_continuation(
    claim: Option<(i32, u64, i32)>,
    live: Option<(u64, i32)>,
) -> Result<i32, BankContConflict> {
    let Some((bank, generation, frontier)) = claim else {
        return Err(BankContConflict);
    };
    if bank < 0 || frontier <= 0 {
        return Err(BankContConflict);
    }
    let Some((live_generation, live_frontier)) = live else {
        return Err(BankContConflict);
    };
    if live_generation != generation || live_frontier != frontier {
        return Err(BankContConflict);
    }
    Ok(bank)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankContConflict;

fn csv(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|s| (*s).to_string()).collect()
}

fn hold_line(r: Option<i32>) -> String {
    match r {
        Some(n) => format!("HOLD 1 retry={n}"),
        None => "HOLD 0".into(),
    }
}

/// Tape matching `tests/parity/cont_c_oracle` scripts.
pub fn dump_script(name: &str) -> String {
    let mut out = String::new();
    match name {
        "publish-resolve-demote" => {
            let mut r = ContRegistry::default();
            let now = 1000.0;
            r.publish_serial(
                Api::Anthropic,
                &csv(&["toolu_regA", "toolu_regB"]),
                7,
                100,
                now,
            );
            out.push_str(&format!(
                "live_anth_a={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_regA", now))
            ));
            out.push_str(&format!(
                "live_anth_b={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_regB", now))
            ));
            out.push_str(&format!(
                "live_resp_a={}\n",
                u32::from(r.live_has_id(Api::Responses, "toolu_regA", now))
            ));
            let ids = csv(&["toolu_regA", "toolu_regB"]);
            out.push_str(&format!(
                "resolve_ok={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &ids, 7, 100, now))
            ));
            out.push_str(&format!(
                "resolve_gen={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &ids, 8, 100, now))
            ));
            out.push_str(&format!(
                "resolve_pos={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &ids, 7, 101, now))
            ));
            out.push_str(&format!(
                "resolve_proto={}\n",
                u32::from(r.resolve_serial(Api::Responses, &ids, 7, 100, now))
            ));
            out.push_str(&format!(
                "resolve_sub={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &csv(&["toolu_regA"]), 7, 100, now))
            ));
            out.push_str(&format!(
                "resolve_sup={}\n",
                u32::from(r.resolve_serial(
                    Api::Anthropic,
                    &csv(&["toolu_regA", "toolu_regB", "toolu_regC"]),
                    7,
                    100,
                    now
                ))
            ));
            r.demote_serial();
            out.push_str(&format!(
                "live_after_demote={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_regA", now))
            ));
            out.push_str(&format!(
                "resolve_after_demote={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &ids, 7, 100, now))
            ));
            out.push_str(&format!(
                "known_after_demote={}\n",
                u32::from(r.id_known("toolu_regA"))
            ));
        }
        "supersede-cap" => {
            let mut r = ContRegistry {
                max_records: 4,
                ..ContRegistry::default()
            };
            let now = 1000.0;
            for t in 1..=2 {
                r.publish_serial(
                    Api::Anthropic,
                    &csv(&[&format!("toolu_turn{t}")]),
                    3,
                    50 * t,
                    now,
                );
            }
            out.push_str(&format!(
                "live1={} live2={} n_live={} n_rec={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_turn1", now)),
                u32::from(r.live_has_id(Api::Anthropic, "toolu_turn2", now)),
                r.n_live(),
                r.n_records()
            ));
            for t in 3..=8 {
                r.publish_serial(
                    Api::Anthropic,
                    &csv(&[&format!("toolu_turn{t}")]),
                    3,
                    50 * t,
                    now,
                );
            }
            out.push_str(&format!(
                "n_rec={} known1={} known2={} live8={} n_live={}\n",
                r.n_records(),
                u32::from(r.id_known("toolu_turn1")),
                u32::from(r.id_known("toolu_turn2")),
                u32::from(r.live_has_id(Api::Anthropic, "toolu_turn8", now)),
                r.n_live()
            ));
        }
        "grace-hold" => {
            let mut r = ContRegistry::default();
            r.publish_serial(Api::Anthropic, &csv(&["toolu_hold"]), 4, 70, 1000.0);
            out.push_str(&format!(
                "{}\n",
                hold_line(r.serial_hold(Api::Openai, &[], 1001.0))
            ));
            out.push_str(&format!(
                "{}\n",
                hold_line(r.serial_hold(Api::Anthropic, &csv(&["toolu_hold"]), 1001.0))
            ));
            out.push_str(&format!(
                "{}\n",
                hold_line(r.serial_hold(Api::Openai, &[], 1011.0))
            ));
            out.push_str(&format!(
                "still_live={}\n",
                match r.serial_live_state() {
                    Some(ContState::LiveFrontier) => 1,
                    _ => 0,
                }
            ));
            out.push_str(&format!(
                "{}\n",
                hold_line(r.serial_hold(Api::Openai, &[], 1131.0))
            ));
            let pin = r.pin_live(Api::Anthropic, "toolu_hold", 1131.0);
            out.push_str(&format!(
                "{}\n",
                hold_line(r.serial_hold(Api::Openai, &[], 1131.0))
            ));
            r.set_serial_pin_expiry(1130.0);
            out.push_str(&format!(
                "{}\n",
                hold_line(r.serial_hold(Api::Openai, &[], 1131.0))
            ));
            if let Some(p) = pin {
                r.unpin(p);
            }
            out.push_str(&format!("hard_refs={}\n", r.serial_live_hard_refs()));
        }
        "ttl" => {
            let mut r = ContRegistry::default();
            r.publish_serial(Api::Anthropic, &csv(&["toolu_ttl"]), 4, 70, 1000.0);
            out.push_str(&format!(
                "live_before={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_ttl", 1000.0))
            ));
            out.push_str(&format!(
                "live_after={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_ttl", 1301.0))
            ));
            out.push_str(&format!("n_live={}\n", r.n_live()));
            out.push_str(&format!(
                "resolve={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &csv(&["toolu_ttl"]), 4, 70, 1301.0))
            ));
            out.push_str(&format!("known={}\n", u32::from(r.id_known("toolu_ttl"))));
        }
        "bank-claim" => {
            let mut r = ContRegistry::default();
            let now = 1000.0;
            r.publish_bank(Api::Anthropic, &csv(&["toolu_bk1"]), 2, 7, 100, now);
            out.push_str(&format!(
                "live={} resp={} serial_live={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk1", now)),
                u32::from(r.live_has_id(Api::Responses, "toolu_bk1", now)),
                u32::from(r.serial_live.is_some())
            ));
            let claim = r.bank_claim(Api::Anthropic, &csv(&["toolu_bk1"]), now);
            out.push_str(&format!(
                "claim={}\n",
                match claim {
                    Some((b, g, f)) => format!("{b},{g},{f}"),
                    None => "-".into(),
                }
            ));
            out.push_str(&format!(
                "claim_resp={}\n",
                u32::from(
                    r.bank_claim(Api::Responses, &csv(&["toolu_bk1"]), now)
                        .is_some()
                )
            ));
            out.push_str(&format!(
                "resolve_serial={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &csv(&["toolu_bk1"]), 7, 100, now))
            ));
            r.publish_serial(Api::Anthropic, &csv(&["toolu_ser1"]), 9, 40, now);
            out.push_str(&format!("n_live={}\n", r.n_live()));
            r.demote_serial();
            out.push_str(&format!(
                "n_live_after={} live_bk1={}\n",
                r.n_live(),
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk1", now))
            ));
            r.publish_bank(Api::Anthropic, &csv(&["toolu_bk2"]), 2, 8, 120, now);
            out.push_str(&format!(
                "live_bk1={} live_bk2={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk1", now)),
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk2", now))
            ));
            r.publish_bank(Api::Responses, &csv(&["toolu_bk3"]), 3, 2, 80, now);
            out.push_str(&format!("n_live={}\n", r.n_live()));
            r.publish_bank(Api::Anthropic, &csv(&["toolu_bk_dead"]), 4, 0, 100, now);
            r.publish_bank(Api::Anthropic, &csv(&["toolu_bk_dead"]), 4, 5, 0, now);
            out.push_str(&format!(
                "known_dead={}\n",
                u32::from(r.id_known("toolu_bk_dead"))
            ));
            r.rewind_live_publish(301.0);
            out.push_str(&format!(
                "ttl_live={} n_live={} known_bk2={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk2", now)),
                r.n_live(),
                u32::from(r.id_known("toolu_bk2"))
            ));
        }
        "bank-protection" => {
            let mut r = ContRegistry::default();
            let now = 1001.0;
            r.grace_s = 60.0;
            r.ttl_s = 300.0;
            r.pin_deadline_s = 20.0;
            r.publish_bank(
                Api::Anthropic,
                &csv(&["toolu_protected"]),
                5,
                3,
                200,
                1000.0,
            );
            out.push_str(&format!(
                "current={}\n",
                hold_line(r.bank_hold_retry(5, Some((3, 200)), now))
            ));
            out.push_str(&format!(
                "stale={}\n",
                hold_line(r.bank_hold_retry(5, Some((4, 200)), now))
            ));
            out.push_str(&format!(
                "unknown={}\n",
                hold_line(r.bank_hold_retry(5, None, now))
            ));
            r.rewind_live_publish(100.0);
            out.push_str(&format!(
                "lapsed={}\n",
                hold_line(r.bank_hold_retry(5, Some((3, 200)), now))
            ));
            let pin = r.pin_live(Api::Anthropic, "toolu_protected", now);
            out.push_str(&format!(
                "pinned={}\n",
                hold_line(r.bank_hold_retry(5, Some((3, 200)), now))
            ));
            if let Some(pin) = pin {
                r.unpin(pin);
            }
            out.push_str(&format!(
                "unpinned={}\n",
                hold_line(r.bank_hold_retry(5, Some((3, 200)), now))
            ));
        }
        _ => out.push_str("ERROR unknown-script\n"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::http_response_bytes;
    use crate::generate::{GenerateError, GenerateOutcome, ScriptedDecode};
    use crate::serve::{handle_client_inner, ServerConfig, ServerInner};
    use crate::serve_cont::ContExec;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Instant;

    #[test]
    fn multibyte_call_id_is_bounded_on_a_utf8_boundary() {
        let id = format!("{}한-tail", "a".repeat(93));
        let key = ContRegistry::key(Api::Responses as u8, &id);
        assert_eq!(&key[2..], "a".repeat(93));

        let mut registry = ContRegistry::default();
        registry.publish_bank(Api::Responses, &[id.clone()], 0, 1, 8, 1.0);
        assert!(registry.live_has_id(Api::Responses, &id, 1.0));
    }

    #[test]
    fn bank_protection_requires_a_current_reference_or_a_live_pin() {
        let mut registry = ContRegistry::default();
        let ids = csv(&["toolu_bank"]);
        registry.grace_s = 10.0;
        registry.ttl_s = 100.0;
        registry.pin_deadline_s = 20.0;
        registry.publish_bank(Api::Anthropic, &ids, 2, 7, 100, 100.0);

        assert!(registry.bank_protected(2, Some((7, 100)), 105.0));
        assert!(!registry.bank_protected(2, Some((8, 100)), 105.0));
        assert!(!registry.bank_protected(2, Some((7, 101)), 105.0));
        assert!(registry.bank_protected(2, None, 105.0));

        registry.rewind_live_publish(11.0);
        assert!(!registry.bank_protected(2, Some((7, 100)), 105.0));
        let pin = registry
            .pin_live(Api::Anthropic, "toolu_bank", 105.0)
            .expect("the live bank record pins");
        assert!(registry.bank_protected(2, Some((7, 100)), 105.0));
        registry.unpin(pin);
        assert!(!registry.bank_protected(2, Some((7, 100)), 105.0));
    }

    #[test]
    fn pin_survives_duplicate_id_republish() {
        let mut registry = ContRegistry::default();
        let ids = csv(&["toolu_same"]);
        registry.publish_serial(Api::Anthropic, &ids, 1, 10, 1.0);
        let pin = registry
            .pin_live(Api::Anthropic, "toolu_same", 1.0)
            .expect("live record should pin");

        registry.publish_serial(Api::Anthropic, &ids, 2, 20, 2.0);
        registry.unpin(pin);

        let old = registry
            .records
            .iter()
            .find(|record| record.owner_gen == 1)
            .expect("old record remains for replay");
        let new = registry
            .records
            .iter()
            .find(|record| record.owner_gen == 2)
            .expect("new record is live");
        assert_eq!(old.hard_refs, 0, "the originally pinned record is released");
        assert_eq!(new.hard_refs, 0, "republish is not accidentally unpinned");
    }

    #[test]
    fn continuation_clock_is_monotonic() {
        let first = monotonic_now();
        let second = monotonic_now();
        assert!(second >= first);
    }

    #[test]
    fn place_bank_continuation_stays_on_claimed_bank_when_live_matches() {
        let mut registry = ContRegistry::default();
        let ids = csv(&["toolu_follow"]);
        registry.publish_bank(Api::Anthropic, &ids, 2, 7, 100, 1000.0);

        let claim = registry.bank_claim(Api::Anthropic, &ids, 1000.0);
        let placed = place_bank_continuation(claim, Some((7, 100)));

        assert_eq!(placed, Ok(2));
    }

    #[test]
    fn place_bank_continuation_conflicts_when_generation_or_frontier_moved() {
        let mut registry = ContRegistry::default();
        let ids = csv(&["toolu_follow"]);
        registry.publish_bank(Api::Anthropic, &ids, 2, 7, 100, 1000.0);
        let claim = registry.bank_claim(Api::Anthropic, &ids, 1000.0);

        assert_eq!(
            place_bank_continuation(claim, Some((8, 100))),
            Err(BankContConflict)
        );
        assert_eq!(
            place_bank_continuation(claim, Some((7, 101))),
            Err(BankContConflict)
        );
    }

    #[test]
    fn pin_owner_reports_bank_owned_live_record() {
        let mut registry = ContRegistry::default();
        registry.publish_bank(Api::Anthropic, &csv(&["toolu_pin"]), 3, 4, 50, 1.0);
        let pin = registry
            .pin_live(Api::Anthropic, "toolu_pin", 1.0)
            .expect("live bank record pins");
        assert_eq!(registry.pin_owner(pin), Some(ContOwner::BatchBank));
    }

    struct BankLane {
        live: Option<(u64, i32)>,
        bank: i32,
        generated: Arc<Mutex<bool>>,
        tool_ids: Vec<String>,
        generation: u64,
        frontier: i32,
    }

    impl ContExec for BankLane {
        fn model_id(&self) -> i32 {
            0
        }
        fn seq_cap(&self) -> i32 {
            8192
        }
        fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
            vec![1]
        }
        fn encode_text(&self, _text: &str) -> Vec<i32> {
            vec![1]
        }
        fn generate(
            &mut self,
            parsed: &crate::parse::ParsedRequest,
            _job_id: &str,
            _created: i64,
            cors: bool,
            _default_tokens: i32,
            _t_arrive: Instant,
            _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
            _store: Option<&mut ds4_kv::Store>,
            out: &mut dyn Write,
        ) -> Result<GenerateOutcome, GenerateError> {
            *self.generated.lock().expect("generated lock") = true;
            if let Some(bank) = parsed.directed_bank {
                assert_eq!(bank, self.bank);
            }
            out.write_all(&http_response_bytes(
                200,
                Some("application/json"),
                None,
                cors,
                "{}",
            ))
            .map_err(|_| GenerateError::Io)?;
            Ok(GenerateOutcome {
                tool_ids: self.tool_ids.clone(),
                bank: Some(self.bank),
                generation: self.generation,
                frontier: self.frontier,
                finish: if self.tool_ids.is_empty() {
                    "stop".into()
                } else {
                    "tool_calls".into()
                },
                ..GenerateOutcome::default()
            })
        }
        fn bank_live(&self, bank: i32) -> Option<(u64, i32)> {
            if bank == self.bank {
                self.live
            } else {
                None
            }
        }
    }

    fn http_post(addr: std::net::SocketAddr, path: &str, body: &str) -> String {
        let mut client = TcpStream::connect(addr).expect("connect");
        let req = format!(
            "POST {path} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        client.write_all(req.as_bytes()).expect("write");
        let mut out = Vec::new();
        client.read_to_end(&mut out).expect("read");
        String::from_utf8_lossy(&out).into_owned()
    }

    #[test]
    fn streaming_anthropic_tool_turn_stays_on_its_bank_for_output_only_follow_up() {
        let cfg = ServerConfig {
            have_engine: true,
            default_tokens: 16,
            ..ServerConfig::default()
        };
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let generated = Arc::new(Mutex::new(false));
        let seen = Arc::clone(&generated);
        let h = thread::spawn(move || {
            let inner = Mutex::new(ServerInner::from_cfg(&cfg));
            let mut engine = ScriptedDecode::from_pieces(&[b"ok"]);
            let mut cont = BankLane {
                live: None,
                bank: 2,
                generated: Arc::clone(&seen),
                tool_ids: vec!["toolu_bankturn".into()],
                generation: 7,
                frontier: 100,
            };
            {
                let (mut first, _) = listener.accept().expect("accept first");
                handle_client_inner(&cfg, &inner, &mut first, Some(&mut engine), Some(&mut cont));
            }
            {
                let mut g = inner.lock().expect("inner");
                assert_eq!(
                    g.creg
                        .bank_claim(Api::Anthropic, &["toolu_bankturn".into()], monotonic_now()),
                    Some((2, 7, 100))
                );
            }
            cont.live = Some((7, 100));
            cont.tool_ids.clear();
            *seen.lock().expect("generated lock") = false;
            {
                let (mut second, _) = listener.accept().expect("accept follow-up");
                handle_client_inner(
                    &cfg,
                    &inner,
                    &mut second,
                    Some(&mut engine),
                    Some(&mut cont),
                );
            }
            assert!(*seen.lock().expect("generated lock"));
            let mut g = inner.lock().expect("inner");
            assert_eq!(
                g.creg
                    .bank_claim(Api::Anthropic, &["toolu_bankturn".into()], monotonic_now()),
                Some((2, 7, 100))
            );
        });
        let tools = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"thinking":{"type":"disabled"},"tools":[{"name":"bash","input_schema":{"type":"object"}}]}"#;
        let first = http_post(addr, "/v1/messages", tools);
        assert!(first.starts_with("HTTP/1.1 200 "), "{first}");
        let follow = http_post(
            addr,
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_bankturn","content":"ok"}]}],"max_tokens":8}"#,
        );
        h.join().expect("server thread");
        assert!(follow.starts_with("HTTP/1.1 200 "), "{follow}");
    }

    #[test]
    fn streaming_anthropic_bank_follow_up_conflicts_when_frontier_moved() {
        let cfg = ServerConfig {
            have_engine: true,
            default_tokens: 16,
            ..ServerConfig::default()
        };
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let generated = Arc::new(Mutex::new(false));
        let seen = Arc::clone(&generated);
        let h = thread::spawn(move || {
            let inner = Mutex::new(ServerInner::from_cfg(&cfg));
            {
                let mut g = inner.lock().expect("inner");
                g.creg.publish_bank(
                    Api::Anthropic,
                    &["toolu_moved".into()],
                    2,
                    7,
                    100,
                    monotonic_now(),
                );
            }
            let mut engine = ScriptedDecode::from_pieces(&[b"ok"]);
            let mut cont = BankLane {
                live: Some((8, 100)),
                bank: 2,
                generated: seen,
                tool_ids: Vec::new(),
                generation: 8,
                frontier: 100,
            };
            let (mut s, _) = listener.accept().expect("accept");
            handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine), Some(&mut cont));
            assert!(!*generated.lock().expect("generated lock"));
        });
        let follow = http_post(
            addr,
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_moved","content":"ok"}]}],"max_tokens":8}"#,
        );
        h.join().expect("server thread");
        assert!(follow.starts_with("HTTP/1.1 409 "), "{follow}");
        assert!(
            follow.contains("Anthropic continuation state is not available"),
            "{follow}"
        );
        assert!(
            follow.contains("replaying the full messages history"),
            "{follow}"
        );
    }
}
