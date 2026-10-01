const TRIAL_CAP: usize = 7;
const VOCAB: i32 = 152576;
const END_TEXT: i32 = 151643;
const END_TURN: i32 = 151645;

pub(super) unsafe extern "C" fn accept_banked(
    tokens: *const i32,
    target: *const i32,
    n: i32,
    eos: i32,
) -> i32 {
    if tokens.is_null() || target.is_null() || n <= 0 || n as usize > TRIAL_CAP {
        return 0;
    }
    // SAFETY: The synchronous native trial lends n entries for this call.
    let tokens = unsafe { std::slice::from_raw_parts(tokens, n as usize) };
    let target = unsafe { std::slice::from_raw_parts(target, n as usize) };
    accepted_prefix(tokens, target, eos).unwrap_or(0) as i32
}

impl crate::Session<'_> {
    pub(super) fn eval_naive_argmax(
        &mut self,
        first: i32,
        max_tokens: i32,
        eos: i32,
    ) -> crate::Result<Vec<i32>> {
        if max_tokens <= 0 {
            return Ok(Vec::new());
        }
        let mut tokens = [0; TRIAL_CAP];
        let mut target = [0; TRIAL_CAP];
        let mut err = [0u8; 512];
        // SAFETY: Exclusive session and bounded borrowed outputs; native
        // retains only its device frontier until commit, never host pointers.
        let n = unsafe {
            ds4_sys::ds4_bridge_naive_trial(
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
                message: "invalid Naive trial result".into(),
            });
        };
        // SAFETY: keep is a prefix of the matching pending native trial.
        let rc = unsafe {
            ds4_sys::ds4_bridge_naive_commit(
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
            .any(|&t| !(0..VOCAB).contains(&t))
    {
        return None;
    }
    let stop = |t| t == eos || t == END_TEXT || t == END_TURN;
    let mut keep = 1;
    while keep < tokens.len() && !stop(tokens[keep - 1]) && tokens[keep] == target[keep - 1] {
        keep += 1;
    }
    Some(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_miss_crops_seven_rows() {
        let tokens = [10, 11, 12, 13, 14, 15, 16];
        assert_eq!(
            accepted_prefix(&tokens, &[11, 12, 13, 14, 15, 16, 17], END_TURN),
            Some(7)
        );
        for miss in 0..6 {
            let mut target = [11, 12, 13, 14, 15, 16, 17];
            target[miss] = 99;
            assert_eq!(accepted_prefix(&tokens, &target, END_TURN), Some(miss + 1));
        }
    }

    #[test]
    fn both_stops_end_prefix() {
        for stop in [END_TEXT, END_TURN, 99] {
            assert_eq!(
                accepted_prefix(&[10, stop, 12], &[stop, 12, 13], 99),
                Some(2)
            );
            assert_eq!(accepted_prefix(&[stop, 11], &[11, 12], 99), Some(1));
        }
    }

    #[test]
    fn malformed_trials_fail_closed() {
        assert_eq!(accepted_prefix(&[], &[], END_TURN), None);
        assert_eq!(accepted_prefix(&[10; 8], &[10; 8], END_TURN), None);
        assert_eq!(accepted_prefix(&[10], &[10, 11], END_TURN), None);
        assert_eq!(accepted_prefix(&[-1], &[11], END_TURN), None);
        assert_eq!(accepted_prefix(&[10], &[VOCAB], END_TURN), None);
        unsafe {
            assert_eq!(
                accept_banked(std::ptr::null(), [1].as_ptr(), 1, END_TURN),
                0
            );
            assert_eq!(accept_banked([1].as_ptr(), [1].as_ptr(), 8, END_TURN), 0);
        }
    }
}
