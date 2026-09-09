const TRIAL_CAP: usize = 9;

impl crate::Session<'_> {
    pub(super) fn eval_inkling_argmax(
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
        // SAFETY: Category 8 FFI. The exclusive session and bounded output
        // buffers remain live for the call; native retains no host pointers.
        let n = unsafe {
            ds4_sys::ds4_bridge_inkling_trial(
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
            self.invalidate();
            return Err(crate::fail(n, &err));
        }
        if n as usize > TRIAL_CAP || n > max_tokens || tokens[0] != first {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid Inkling trial result".into(),
            });
        }
        let n = n as usize;
        let Some(keep) = accepted_prefix(&tokens[..n], &target[..n], eos) else {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid Inkling trial shape".into(),
            });
        };
        // SAFETY: Category 8 FFI. keep is within the pending trial; native
        // commits device state and returns before Rust updates its timeline.
        let rc = unsafe {
            ds4_sys::ds4_bridge_inkling_commit(
                self.raw.as_ptr(),
                keep as i32,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if rc != 0 {
            self.invalidate();
            return Err(crate::fail(rc, &err));
        }
        for &token in &tokens[..keep] {
            self.host.commit_eval(token);
        }
        Ok(tokens[..keep].to_vec())
    }
}

fn accepted_prefix(tokens: &[i32], target: &[i32], eos: i32) -> Option<usize> {
    if tokens.is_empty() || tokens.len() > TRIAL_CAP || tokens.len() != target.len() {
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
    use super::accepted_prefix;

    #[test]
    fn acceptance_stops_at_first_miss() {
        assert_eq!(accepted_prefix(&[10, 11, 12], &[11, 12, 13], 99), Some(3));
        assert_eq!(accepted_prefix(&[10, 11, 12], &[13, 12, 13], 99), Some(1));
        assert_eq!(
            accepted_prefix(&[10, 11, 12, 13], &[11, 99, 13, 14], 99),
            Some(2)
        );
    }

    #[test]
    fn acceptance_includes_first_stop() {
        assert_eq!(accepted_prefix(&[99, 11, 12], &[11, 12, 13], 99), Some(1));
        assert_eq!(accepted_prefix(&[10, 99, 12], &[99, 12, 13], 99), Some(2));
        assert_eq!(accepted_prefix(&[10, 11, 99], &[11, 99, 13], 99), Some(3));
        assert_eq!(accepted_prefix(&[10, 11], &[99, 12], 99), Some(1));
    }

    #[test]
    fn acceptance_checks_trial_shape() {
        assert_eq!(accepted_prefix(&[10], &[11], 99), Some(1));
        assert_eq!(accepted_prefix(&[10; 9], &[10; 9], 99), Some(9));
        assert_eq!(accepted_prefix(&[], &[], 99), None);
        assert_eq!(accepted_prefix(&[10; 10], &[10; 10], 99), None);
        assert_eq!(accepted_prefix(&[10], &[], 99), None);
        assert_eq!(accepted_prefix(&[10], &[11, 12], 99), None);
    }
}
