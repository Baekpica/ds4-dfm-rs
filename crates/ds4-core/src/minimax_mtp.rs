//! MiniMax DSpark speculative decode: the native trial drafts one block and
//! verifies it in a single target forward; the host keeps the accepted prefix.

use std::time::Instant;

/// Eight verify rows: the anchor and up to seven proposals (one block).
const TRIAL_CAP: usize = 8;

/// DSpark pays only while a trial's kept tokens outrun the plain decodes
/// they replace: on raw prose the drafter often misses (10.7-26.5 tok/s per
/// frontier against ~19 plain in the 2K-64K sweep). The gate keeps measured
/// rates of trials and of plain decodes and, while drafting runs slower,
/// decodes plainly for a cooldown before trying again. Greedy output is the
/// same either way. `DS4_MINIMAX_DSPARK_GATE=0` always drafts.
#[derive(Clone, Debug, Default)]
pub struct DsparkGate {
    draft_rate: f64,
    plain_rate: f64,
    trials: u32,
    plains: u32,
    cooldown: u32,
}

impl DsparkGate {
    /// Trials measured before the gate judges.
    const WARMUP: u32 = 2;
    /// Plain decodes per pause.
    const COOLDOWN: u32 = 16;
    /// Plain decodes that measure the plain rate when none is known yet.
    const PROBE: u32 = 4;

    fn ewma(rate: &mut f64, n: &mut u32, sample: f64) {
        *rate = if *n == 0 {
            sample
        } else {
            0.7 * *rate + 0.3 * sample
        };
        *n += 1;
    }

    /// A trial kept tokens at `rate` tokens/s.
    fn after_trial(&mut self, rate: f64) {
        Self::ewma(&mut self.draft_rate, &mut self.trials, rate);
        if self.trials >= Self::WARMUP {
            if self.plains == 0 {
                self.cooldown = Self::PROBE;
            } else if self.draft_rate < self.plain_rate {
                self.cooldown = Self::COOLDOWN;
            }
        }
    }

    /// A plain decode ran at `rate` tokens/s; drafting is measured afresh
    /// once the pause ends.
    fn after_plain(&mut self, rate: f64) {
        Self::ewma(&mut self.plain_rate, &mut self.plains, rate);
        self.cooldown = self.cooldown.saturating_sub(1);
        if self.cooldown == 0 {
            self.trials = 0;
        }
    }
}

fn gate_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS4_MINIMAX_DSPARK_GATE").ok().as_deref() != Some("0"))
}
/// Tokenizer ids; the head's padded rows are masked natively.
const TOKENS: i32 = 200_061;

impl crate::Session<'_> {
    pub(super) fn eval_minimax_argmax(
        &mut self,
        first: i32,
        max_tokens: i32,
        eos: i32,
        limit: &mut dyn FnMut(&[i32]) -> usize,
    ) -> crate::Result<Vec<i32>> {
        if max_tokens <= 0 {
            return Ok(Vec::new());
        }
        let gate = gate_enabled();
        if gate && self.host.dspark_gate.cooldown > 0 {
            let t0 = Instant::now();
            self.eval(first)?;
            let rate = 1.0 / t0.elapsed().as_secs_f64().max(1e-9);
            self.host.dspark_gate.after_plain(rate);
            return Ok(vec![first]);
        }
        let t0 = Instant::now();
        let mut tokens = [0; TRIAL_CAP];
        let mut target = [0; TRIAL_CAP];
        let mut err = [0u8; 512];
        // SAFETY: Exclusive session and bounded borrowed outputs; native keeps
        // only its device rows until commit, never these pointers.
        let n = unsafe {
            ds4_sys::ds4_bridge_minimax_trial(
                self.raw.as_ptr(),
                first,
                max_tokens,
                tokens.as_mut_ptr(),
                target.as_mut_ptr(),
                TRIAL_CAP as i32,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if n == 0 {
            self.eval(first)?;
            return Ok(vec![first]);
        }
        if n < 0 {
            self.step_failed();
            return Err(crate::fail(n, &err));
        }
        let keep = (n as usize <= TRIAL_CAP && n <= max_tokens && tokens[0] == first)
            .then(|| accepted_prefix(&tokens[..n as usize], &target[..n as usize], eos))
            .flatten();
        let Some(keep) = keep else {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid MiniMax trial result".into(),
            });
        };
        let keep = limit(&tokens[..keep]).clamp(1, keep);
        // SAFETY: keep is a prefix of the matching pending native trial.
        let rc = unsafe {
            ds4_sys::ds4_bridge_minimax_commit(
                self.raw.as_ptr(),
                keep as i32,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if rc != 0 {
            self.step_failed();
            return Err(crate::fail(rc, &err));
        }
        for &token in &tokens[..keep] {
            self.host.commit_eval(token);
        }
        if gate {
            let rate = keep as f64 / t0.elapsed().as_secs_f64().max(1e-9);
            self.host.dspark_gate.after_trial(rate);
        }
        Ok(tokens[..keep].to_vec())
    }
}

/// Rows kept: the anchor, then each proposal the target's argmax of the row
/// before agrees with; EOS ends the prefix.
fn accepted_prefix(tokens: &[i32], target: &[i32], eos: i32) -> Option<usize> {
    if tokens.is_empty()
        || tokens.len() > TRIAL_CAP
        || tokens.len() != target.len()
        || tokens
            .iter()
            .chain(target)
            .any(|&t| !(0..TOKENS).contains(&t))
    {
        return None;
    }
    let mut keep = 1;
    while keep < tokens.len() && tokens[keep - 1] != eos && tokens[keep] == target[keep - 1] {
        keep += 1;
    }
    Some(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: i32 = 200_020;

    #[test]
    fn first_miss_crops_the_block() {
        let tokens: Vec<i32> = (10..17).collect();
        let target: Vec<i32> = (11..18).collect();
        assert_eq!(accepted_prefix(&tokens, &target, EOS), Some(7));
        for miss in 0..6 {
            let mut target = target.clone();
            target[miss] = 99;
            assert_eq!(accepted_prefix(&tokens, &target, EOS), Some(miss + 1));
        }
    }

    #[test]
    fn gate_pauses_slow_drafting_and_retries() {
        let mut g = DsparkGate::default();
        // Two trials, no plain rate yet: probe plain decoding.
        g.after_trial(30.0);
        g.after_trial(30.0);
        assert_eq!(g.cooldown, DsparkGate::PROBE);
        for _ in 0..DsparkGate::PROBE {
            g.after_plain(20.0);
        }
        assert_eq!((g.cooldown, g.trials), (0, 0));
        // Faster drafting keeps drafting.
        g.after_trial(30.0);
        g.after_trial(30.0);
        assert_eq!(g.cooldown, 0);
        // Slower drafting pauses for a cooldown, then measures again.
        g.after_trial(5.0);
        g.after_trial(5.0);
        assert_eq!(g.cooldown, DsparkGate::COOLDOWN);
        for _ in 0..DsparkGate::COOLDOWN {
            g.after_plain(20.0);
        }
        assert_eq!((g.cooldown, g.trials), (0, 0));
    }

    #[test]
    fn eos_ends_the_prefix_and_bad_rows_refuse() {
        assert_eq!(
            accepted_prefix(&[10, EOS, 12], &[EOS, 12, 13], EOS),
            Some(2)
        );
        assert_eq!(accepted_prefix(&[10, 11], &[11], EOS), None);
        assert_eq!(accepted_prefix(&[10, TOKENS], &[TOKENS, 1], EOS), None);
    }
}
