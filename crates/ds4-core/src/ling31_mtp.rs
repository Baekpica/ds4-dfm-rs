// Ling-3.1 VQ's embedded MTP on the shared trial/commit contract: native
// drafts and verifies, Rust keeps the accepted prefix.
const TRIAL_CAP: usize = 4;

impl crate::Session<'_> {
    pub(super) fn eval_ling31_argmax(
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
        // SAFETY: Category 8 FFI. The exclusive session and bounded output
        // buffers remain live for the call; native retains no host pointers.
        let n = unsafe {
            ds4_sys::ds4_bridge_ling31_trial(
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
        if n as usize > TRIAL_CAP || n > max_tokens || tokens[0] != first {
            self.invalidate();
            return Err(crate::Error {
                code: 1,
                message: "invalid Ling-3.1 trial result".into(),
            });
        }
        let n = n as usize;
        let mut keep = 1;
        while keep < n && tokens[keep - 1] != eos && tokens[keep] == target[keep - 1] {
            keep += 1;
        }
        let keep = limit(&tokens[..keep]).clamp(1, keep);
        // SAFETY: Category 8 FFI. keep is within the pending trial; native
        // commits device state and returns before Rust updates its timeline.
        let rc = unsafe {
            ds4_sys::ds4_bridge_ling31_commit(
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
