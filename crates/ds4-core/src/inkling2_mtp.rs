//! Inkling-2 DSpark speculative decode: the native trial drafts one block
//! (the anchor and up to seven proposals) and verifies it in a single target
//! forward; the host keeps the accepted prefix and native rolls the
//! convolution histories back to it. MiniMax's rate gate decides when
//! drafting pays: while trials keep tokens slower than plain decodes, it
//! decodes plainly for a cooldown. `DS4_INKLING2_DSPARK_GATE=0` always drafts.

use std::time::Instant;

/// Eight verify rows: the anchor and up to seven proposals (one block).
const TRIAL_CAP: usize = 8;
/// Tokenizer ids; the head's padded rows are masked natively.
const TOKENS: i32 = 200_058;

fn gate_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS4_INKLING2_DSPARK_GATE").ok().as_deref() != Some("0"))
}

impl crate::Session<'_> {
    pub(super) fn eval_inkling2_argmax(
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
        if gate && self.host.dspark_gate.cooling() {
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
            ds4_sys::ds4_bridge_inkling2_trial(
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
                message: "invalid Inkling-2 trial result".into(),
            });
        };
        let keep = limit(&tokens[..keep]).clamp(1, keep);
        // SAFETY: keep is a prefix of the matching pending native trial.
        let rc = unsafe {
            ds4_sys::ds4_bridge_inkling2_commit(
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

/// Rows kept: the anchor, then each proposal the target's draw after the row
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

    const EOS: i32 = 200_006;

    #[test]
    fn first_miss_crops_the_block() {
        let tokens: Vec<i32> = (10..18).collect();
        let target: Vec<i32> = (11..19).collect();
        assert_eq!(accepted_prefix(&tokens, &target, EOS), Some(8));
        for miss in 0..7 {
            let mut target = target.clone();
            target[miss] = 99;
            assert_eq!(accepted_prefix(&tokens, &target, EOS), Some(miss + 1));
        }
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
