type Trial = unsafe extern "C" fn(
    *mut ds4_sys::ds4_bridge_session,
    i32,
    i32,
    *mut i32,
    *mut i32,
    i32,
    *mut std::ffi::c_char,
    usize,
) -> i32;
type Commit = unsafe extern "C" fn(
    *mut ds4_sys::ds4_bridge_session,
    i32,
    *mut std::ffi::c_char,
    usize,
) -> i32;

pub(super) unsafe extern "C" fn accept_banked<const CAP: usize>(
    tokens: *const i32,
    target: *const i32,
    n: i32,
    eos: i32,
) -> i32 {
    if tokens.is_null() || target.is_null() || n <= 0 || n as usize > CAP {
        return 0;
    }
    // SAFETY: The synchronous native trial lends n entries for this call.
    let tokens = unsafe { std::slice::from_raw_parts(tokens, n as usize) };
    let target = unsafe { std::slice::from_raw_parts(target, n as usize) };
    accepted_prefix::<CAP>(tokens, target, eos).unwrap_or(0) as i32
}

impl crate::Session<'_> {
    pub(super) fn eval_block_argmax<const CAP: usize>(
        &mut self,
        first: i32,
        max_tokens: i32,
        eos: i32,
        limit: &mut dyn FnMut(&[i32]) -> usize,
    ) -> crate::Result<Vec<i32>> {
        if max_tokens <= 0 {
            return Ok(Vec::new());
        }
        let mut tokens = [0; CAP];
        let mut target = [0; CAP];
        let mut err = [0u8; 512];
        // SAFETY: Category 8 FFI. The exclusive session and bounded output
        // buffers remain live for the call; native retains no host pointers.
        let (trial_fn, commit_fn) = if self.host.family == crate::ModelFamily::Hy4 {
            (
                ds4_sys::ds4_bridge_hy4_trial as Trial,
                ds4_sys::ds4_bridge_hy4_commit as Commit,
            )
        } else {
            (
                ds4_sys::ds4_bridge_step37_trial as Trial,
                ds4_sys::ds4_bridge_step37_commit as Commit,
            )
        };
        let n = unsafe {
            trial_fn(
                self.raw.as_ptr(),
                first,
                max_tokens,
                tokens.as_mut_ptr(),
                target.as_mut_ptr(),
                CAP as i32,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if n == 0 {
            self.eval(first)?;
            return Ok(vec![first]);
        }
        if n < 0 {
            // Native advances generation only after device state is touched.
            self.step_failed();
            return Err(crate::fail(n, &err));
        }
        if n as usize > CAP || n > max_tokens || tokens[0] != first {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid block-drafter trial result".into(),
            });
        }
        let n = n as usize;
        let Some(keep) = accepted_prefix::<CAP>(&tokens[..n], &target[..n], eos) else {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid block-drafter trial shape".into(),
            });
        };
        let keep = limit(&tokens[..keep]).clamp(1, keep);
        // SAFETY: Category 8 FFI. keep is within the pending trial; native
        // commits device state and returns before Rust updates its timeline.
        let rc = unsafe {
            commit_fn(
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

fn accepted_prefix<const CAP: usize>(tokens: &[i32], target: &[i32], eos: i32) -> Option<usize> {
    if tokens.is_empty() || tokens.len() > CAP || tokens.len() != target.len() {
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
    fn bank_callback_matches_serial_policy() {
        let tokens = [10, 11, 12, 13];
        let target = [11, 12, 99, 14];
        // SAFETY: Both arrays have the advertised four live entries.
        unsafe {
            assert_eq!(
                super::accept_banked::<4>(tokens.as_ptr(), target.as_ptr(), 4, 99),
                3
            );
            assert_eq!(
                super::accept_banked::<4>(tokens.as_ptr(), target.as_ptr(), 4, 11),
                2
            );
            assert_eq!(
                super::accept_banked::<4>(std::ptr::null(), target.as_ptr(), 4, 99),
                0
            );
            assert_eq!(
                super::accept_banked::<4>(tokens.as_ptr(), target.as_ptr(), 5, 99),
                0
            );
        }
    }

    #[test]
    fn acceptance_stops_at_first_miss() {
        assert_eq!(
            accepted_prefix::<4>(&[10, 11, 12], &[11, 12, 13], 99),
            Some(3)
        );
        assert_eq!(
            accepted_prefix::<4>(&[10, 11, 12], &[13, 12, 13], 99),
            Some(1)
        );
        assert_eq!(
            accepted_prefix::<4>(&[10, 11, 12, 13], &[11, 99, 13, 14], 99),
            Some(2)
        );
    }

    #[test]
    fn acceptance_includes_first_stop() {
        assert_eq!(
            accepted_prefix::<4>(&[99, 11, 12], &[11, 12, 13], 99),
            Some(1)
        );
        assert_eq!(
            accepted_prefix::<4>(&[10, 99, 12], &[99, 12, 13], 99),
            Some(2)
        );
        assert_eq!(
            accepted_prefix::<4>(&[10, 11, 99], &[11, 99, 13], 99),
            Some(3)
        );
        assert_eq!(accepted_prefix::<4>(&[10, 11], &[99, 12], 99), Some(1));
    }

    #[test]
    fn acceptance_checks_trial_shape() {
        assert_eq!(accepted_prefix::<4>(&[10], &[11], 99), Some(1));
        assert_eq!(accepted_prefix::<4>(&[10; 4], &[10; 4], 99), Some(4));
        assert_eq!(accepted_prefix::<4>(&[], &[], 99), None);
        assert_eq!(accepted_prefix::<4>(&[10; 5], &[10; 5], 99), None);
        assert_eq!(accepted_prefix::<4>(&[10], &[], 99), None);
        assert_eq!(accepted_prefix::<4>(&[10], &[11, 12], 99), None);
    }
}
