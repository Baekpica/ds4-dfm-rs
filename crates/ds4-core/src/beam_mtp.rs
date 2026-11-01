//! Beam DFlash2 speculative decode: the native trial drafts one block and
//! verifies it in a single target forward; the host keeps the accepted prefix.

const TRIAL_CAP: usize = 16;
/// Tokenizer ids; the head's padded rows are capped out natively.
const TOKENS: i32 = 151681;
/// Harmony's control tokens (`<|start|>`, `<|channel|>`, `<|end|>`, the stops)
/// and the other added tokens start here. A prefix ends at one, so the server
/// forces the next header tokens one at a time instead of committing the
/// target's argmax past it.
const CONTROL: i32 = 151643;

impl crate::Session<'_> {
    pub(super) fn eval_beam_argmax(
        &mut self,
        first: i32,
        max_tokens: i32,
        eos: i32,
        limit: &mut dyn FnMut(&[i32]) -> usize,
    ) -> crate::Result<Vec<i32>> {
        if max_tokens <= 0 {
            return Ok(Vec::new());
        }
        let mut tokens = [0; TRIAL_CAP];
        let mut target = [0; TRIAL_CAP];
        let mut err = [0u8; 512];
        // SAFETY: Exclusive session and bounded borrowed outputs; native keeps
        // only its device rows until commit, never these pointers.
        let n = unsafe {
            ds4_sys::ds4_bridge_beam_trial(
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
                message: "invalid Beam trial result".into(),
            });
        };
        let keep = limit(&tokens[..keep]).clamp(1, keep);
        // SAFETY: keep is a prefix of the matching pending native trial.
        let rc = unsafe {
            ds4_sys::ds4_bridge_beam_commit(
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
        Ok(tokens[..keep].to_vec())
    }
}

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
    let stop = |t| t == eos || t >= CONTROL;
    let mut keep = 1;
    while keep < tokens.len() && !stop(tokens[keep - 1]) && tokens[keep] == target[keep - 1] {
        keep += 1;
    }
    Some(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RETURN: i32 = 151674;

    #[test]
    fn first_miss_crops_the_block() {
        let tokens: Vec<i32> = (10..26).collect();
        let target: Vec<i32> = (11..27).collect();
        assert_eq!(accepted_prefix(&tokens, &target, RETURN), Some(16));
        for miss in 0..15 {
            let mut target = target.clone();
            target[miss] = 99;
            assert_eq!(accepted_prefix(&tokens, &target, RETURN), Some(miss + 1));
        }
    }

    #[test]
    fn control_tokens_end_the_prefix() {
        // <|end|>, <|start|>, <|channel|> and the stops are all committed last.
        for control in [151670, 151669, 151672, RETURN, 151675, CONTROL] {
            assert_eq!(
                accepted_prefix(&[10, control, 12], &[control, 12, 13], RETURN),
                Some(2)
            );
            assert_eq!(accepted_prefix(&[control, 11], &[11, 12], RETURN), Some(1));
        }
        assert_eq!(accepted_prefix(&[10, 11, 12], &[11, 12, 13], 11), Some(2));
    }

    #[test]
    fn malformed_trials_fail_closed() {
        assert_eq!(accepted_prefix(&[], &[], RETURN), None);
        assert_eq!(accepted_prefix(&[10; 17], &[10; 17], RETURN), None);
        assert_eq!(accepted_prefix(&[10], &[10, 11], RETURN), None);
        assert_eq!(accepted_prefix(&[-1], &[11], RETURN), None);
        assert_eq!(accepted_prefix(&[10], &[TOKENS], RETURN), None);
    }
}
