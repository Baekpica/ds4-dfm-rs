//! Store-length, continued pacing, budget, and eviction scoring.
//! Port of the corresponding helpers in `ds4_kvstore.c`.

use crate::format::{text_sha_hex, Reason};

pub const DEFAULT_MB: u64 = 4096;
pub const HIT_HALF_LIFE_SECONDS: u64 = 6 * 60 * 60;
pub const DEFAULT_MIN_TOKENS: i32 = 512;
pub const DEFAULT_COLD_MAX_TOKENS: i32 = 30000;
pub const DEFAULT_BOUNDARY_TRIM_TOKENS: i32 = 32;
pub const DEFAULT_BOUNDARY_ALIGN_TOKENS: i32 = 2048;
pub const DEFAULT_CONTINUED_INTERVAL_TOKENS: i32 = 10000;
const MIN_EFFECTIVE_HITS: f64 = 0.01;
const CONTINUED_PREFIX_MIN_FACTOR: f64 = 0.05;
const CONTINUED_PREFIX_HIT_FACTOR: f64 = 0.45;
const ANCHOR_REASON_SCORE_FACTOR: f64 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub min_tokens: i32,
    pub cold_max_tokens: i32,
    pub continued_interval_tokens: i32,
    pub boundary_trim_tokens: i32,
    pub boundary_align_tokens: i32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            min_tokens: DEFAULT_MIN_TOKENS,
            cold_max_tokens: DEFAULT_COLD_MAX_TOKENS,
            continued_interval_tokens: DEFAULT_CONTINUED_INTERVAL_TOKENS,
            boundary_trim_tokens: DEFAULT_BOUNDARY_TRIM_TOKENS,
            boundary_align_tokens: DEFAULT_BOUNDARY_ALIGN_TOKENS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvictionContext<'a> {
    pub text: &'a [u8],
    pub model_id: u8,
    pub quant_bits: u8,
    pub ctx_size: u32,
    pub reject_different_quant: bool,
}

#[derive(Clone, Debug)]
pub struct ScoreEntry<'a> {
    pub sha: &'a str,
    pub quant_bits: u8,
    pub model_id: u8,
    pub reason: Reason,
    pub tokens: u32,
    pub hits: u32,
    pub ctx_size: u32,
    pub created_at: u64,
    pub last_used: u64,
    pub text_bytes: u64,
    pub file_size: u64,
}

pub fn store_len(opt: &Options, tokens: i32) -> i32 {
    let trim = opt.boundary_trim_tokens;
    let align = opt.boundary_align_tokens;
    if tokens > opt.min_tokens + trim {
        let mut stable = tokens - trim;
        if align > 0 {
            stable -= stable % align;
        }
        if stable >= opt.min_tokens {
            return stable;
        }
    }
    tokens
}

pub fn continued_step(opt: &Options) -> i32 {
    if opt.continued_interval_tokens <= 0 {
        return 0;
    }
    let mut step = opt.continued_interval_tokens;
    let align = opt.boundary_align_tokens;
    if align > 0 {
        let rounded =
            ((i64::from(step) + i64::from(align) - 1) / i64::from(align)) * i64::from(align);
        step = i32::try_from(rounded).unwrap_or(0);
    }
    step
}

pub fn continued_store_target(opt: &Options, last_store_tokens: i32, live_tokens: i32) -> i32 {
    let step = continued_step(opt);
    if step <= 0 {
        return 0;
    }
    if live_tokens < opt.min_tokens {
        return 0;
    }
    if live_tokens % step != 0 {
        return 0;
    }
    if live_tokens <= last_store_tokens {
        return 0;
    }
    live_tokens
}

pub fn bank_checkpoint_due(opt: &Options, committed: i32, stored_tokens: i32) -> bool {
    let step = continued_step(opt);
    step > 0 && committed >= opt.min_tokens && committed >= stored_tokens + step
}

pub fn file_size_bytes(text_bytes: u64, payload_bytes: u64, trailer_bytes: u64) -> Option<u64> {
    const FIXED: u64 = 48 + 4;
    FIXED
        .checked_add(text_bytes)?
        .checked_add(payload_bytes)?
        .checked_add(trailer_bytes)
}

pub fn budget_required(file_bytes: u64) -> Option<u64> {
    let mut slack = file_bytes / 100;
    if file_bytes % 100 != 0 {
        slack += 1;
    }
    file_bytes.checked_add(slack)
}

pub fn file_size_fits(
    budget_bytes: u64,
    text_bytes: u64,
    payload_bytes: u64,
    trailer_bytes: u64,
) -> bool {
    let Some(file_bytes) = file_size_bytes(text_bytes, payload_bytes, trailer_bytes) else {
        return false;
    };
    if budget_bytes == 0 {
        return true;
    }
    match budget_required(file_bytes) {
        Some(required) => required <= budget_bytes,
        None => false,
    }
}

fn reason_is_anchor(reason: Reason) -> bool {
    matches!(
        reason,
        Reason::Cold
            | Reason::Evict
            | Reason::Shutdown
            | Reason::BankEvict
            | Reason::BankShutdown
            | Reason::BankCheckpoint
    )
}

fn incoming_supersedes_continued(e: &ScoreEntry<'_>, incoming: &EvictionContext<'_>) -> bool {
    if e.reason != Reason::Continued && e.reason != Reason::BankCheckpoint {
        return false;
    }
    if e.text_bytes == 0 || e.text_bytes >= incoming.text.len() as u64 {
        return false;
    }
    if e.model_id != incoming.model_id {
        return false;
    }
    if incoming.reject_different_quant && e.quant_bits != incoming.quant_bits {
        return false;
    }
    if incoming.ctx_size > e.ctx_size {
        return false;
    }
    let prefix = &incoming.text[..e.text_bytes as usize];
    text_sha_hex(prefix) == e.sha
}

pub fn eviction_score(e: &ScoreEntry<'_>, now: u64, incoming: Option<&EvictionContext<'_>>) -> f64 {
    if e.file_size == 0 {
        return 0.0;
    }
    let mut effective_hits = e.hits as f64;
    let used_at = if e.last_used != 0 {
        e.last_used
    } else {
        e.created_at
    };
    if used_at == 0 {
        effective_hits = 0.0;
    } else if now > used_at {
        let elapsed = (now - used_at) as f64;
        effective_hits *= (-elapsed / HIT_HALF_LIFE_SECONDS as f64).exp2();
        if effective_hits < MIN_EFFECTIVE_HITS {
            effective_hits = 0.0;
        }
    }
    let mut score = (effective_hits + 1.0) * (e.tokens as f64) / (e.file_size as f64);
    if reason_is_anchor(e.reason) {
        score *= ANCHOR_REASON_SCORE_FACTOR;
    }
    if let Some(incoming) = incoming {
        if incoming_supersedes_continued(e, incoming) {
            let h = if effective_hits > 0.0 {
                effective_hits / (effective_hits + 1.0)
            } else {
                0.0
            };
            score *= CONTINUED_PREFIX_MIN_FACTOR + CONTINUED_PREFIX_HIT_FACTOR * h;
        }
    }
    score
}

pub fn chat_anchor_pos(
    opt: &Options,
    prompt: &[i32],
    user_token_id: i32,
    assistant_token_id: i32,
) -> i32 {
    if user_token_id < 0 || assistant_token_id < 0 {
        return -1;
    }
    let mut last_user = -1;
    for (i, &token) in prompt.iter().enumerate() {
        if token == assistant_token_id {
            break;
        }
        if token == user_token_id {
            last_user = i as i32;
        }
    }
    if last_user >= opt.min_tokens {
        last_user
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_len_trims_to_align() {
        let opt = Options::default();
        assert_eq!(store_len(&opt, 4096 + 40), 4096);
        assert_eq!(store_len(&opt, 100), 100);
    }

    #[test]
    fn continued_target_needs_aligned_interval() {
        let opt = Options::default();
        assert_eq!(continued_store_target(&opt, 0, 8192), 0);
        assert_eq!(continued_store_target(&opt, 0, 10240), 10240);
        assert_eq!(continued_store_target(&opt, 10240, 10240), 0);
    }

    #[test]
    fn continued_step_maps_default_10000_interval_to_10240() {
        let opt = Options::default();
        assert_eq!(DEFAULT_CONTINUED_INTERVAL_TOKENS, 10000);
        assert_eq!(
            opt.continued_interval_tokens,
            DEFAULT_CONTINUED_INTERVAL_TOKENS
        );
        assert_eq!(continued_step(&opt), 10240);
    }

    #[test]
    fn bank_checkpoint_due_when_committed_at_10240_aligned_frontier() {
        let opt = Options::default();
        assert!(bank_checkpoint_due(&opt, 10240, 0));
    }

    #[test]
    fn bank_checkpoint_not_due_when_committed_is_raw_10000_interval() {
        let opt = Options::default();
        assert!(!bank_checkpoint_due(&opt, 10000, 0));
    }

    #[test]
    fn bank_checkpoint_not_due_when_committed_is_9999() {
        let opt = Options::default();
        assert!(!bank_checkpoint_due(&opt, 9999, 0));
    }

    #[test]
    fn continued_step_disables_unrepresentable_aligned_interval() {
        let opt = Options {
            min_tokens: 1,
            continued_interval_tokens: i32::MAX,
            boundary_align_tokens: 2048,
            ..Options::default()
        };
        assert_eq!(continued_step(&opt), 0);
        assert_eq!(continued_store_target(&opt, 0, 2048), 0);
    }

    #[test]
    fn budget_adds_one_percent() {
        assert_eq!(budget_required(100), Some(101));
        assert_eq!(budget_required(101), Some(103));
    }

    #[test]
    fn equal_length_incoming_does_not_discount_continued() {
        let text = b"same prefix";
        let sha = text_sha_hex(text);
        let entry = ScoreEntry {
            sha: &sha,
            quant_bits: 2,
            model_id: 0,
            reason: Reason::Continued,
            tokens: 512,
            hits: 0,
            ctx_size: 2048,
            created_at: 1000,
            last_used: 1000,
            text_bytes: text.len() as u64,
            file_size: 4096,
        };
        let incoming = EvictionContext {
            text,
            model_id: 0,
            quant_bits: 2,
            ctx_size: 2048,
            reject_different_quant: true,
        };

        assert_eq!(
            eviction_score(&entry, 1000, Some(&incoming)),
            eviction_score(&entry, 1000, None)
        );
    }
}
