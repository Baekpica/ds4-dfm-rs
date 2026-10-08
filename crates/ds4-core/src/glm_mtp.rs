const TRIAL_CAP: usize = 4;
const VOCAB: i32 = crate::SHAPE_GLM53_FLASH.n_vocab as i32;
// Official GLM generation stops include role transitions as well as EOS.
const USER: i32 = 154827;
const OBSERVATION: i32 = 154829;
pub(super) const DRAFT_MAX: i32 = TRIAL_CAP as i32 - 1;

pub(super) fn mode(draft: i32, enable: Option<&str>, disable: Option<&str>) -> crate::MtpMode {
    // Native enables embedded MTP for a multi-token draft unless disabled.
    if (draft > 1 || enable == Some("1")) && disable != Some("1") {
        crate::MtpMode::On
    } else {
        crate::MtpMode::Off
    }
}

#[cfg(test)]
#[test]
fn mode_matches_native() {
    use crate::MtpMode::{Off, On};
    for (draft, enable, disable, expected) in [
        (1, None, None, Off),
        (1, Some("1"), None, On),
        (3, Some("0"), None, On),
        (3, Some("1"), Some("1"), Off),
        (3, None, Some("1"), Off),
    ] {
        assert_eq!(mode(draft, enable, disable), expected);
    }
}

impl crate::Session<'_> {
    pub(super) fn eval_glm_argmax(
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
        // SAFETY: Exclusive session and four borrowed output rows. Native
        // keeps only its device journal until commit, never host pointers.
        let n = unsafe {
            ds4_sys::ds4_bridge_glm53_trial(
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
                message: "invalid GLM trial result".into(),
            });
        };
        // Restore the recorded accepted prefix; a second forward would
        // change KDA recurrence and pooled-indexer tail transitions.
        let rc = unsafe {
            ds4_sys::ds4_bridge_glm53_commit(
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
            .any(|&token| !(0..VOCAB).contains(&token))
    {
        return None;
    }
    let stop = |token| token == eos || token == USER || token == OBSERVATION;
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
    fn first_miss_crops_four_rows() {
        let tokens = [10, 11, 12, 13];
        assert_eq!(accepted_prefix(&tokens, &[11, 12, 13, 14], 99), Some(4));
        for miss in 0..3 {
            let mut target = [11, 12, 13, 14];
            target[miss] = 99;
            assert_eq!(accepted_prefix(&tokens, &target, 99), Some(miss + 1));
        }
    }

    #[test]
    fn official_stops_end_prefix() {
        for stop in [USER, OBSERVATION, 99] {
            assert_eq!(
                accepted_prefix(&[10, stop, 12], &[stop, 12, 13], 99),
                Some(2)
            );
            assert_eq!(accepted_prefix(&[stop, 11], &[11, 12], 99), Some(1));
        }
    }

    #[test]
    fn malformed_trials_fail_closed() {
        assert_eq!(accepted_prefix(&[], &[], 99), None);
        assert_eq!(accepted_prefix(&[10; 5], &[10; 5], 99), None);
        assert_eq!(accepted_prefix(&[10], &[10, 11], 99), None);
        assert_eq!(accepted_prefix(&[-1], &[11], 99), None);
        assert_eq!(accepted_prefix(&[10], &[VOCAB], 99), None);
    }
}
