//! The Gemma-4 MTP gate: drafting may change acceptance and speed, never
//! the target's token stream. `serving_check` runs it when the drafter is
//! bound; `examples/gemma4_mtp_gate.rs` runs it on its own.

use super::*;

/// One multi-token accepted block and the state checks around it.
#[derive(Clone, Debug)]
pub struct MtpBlock {
    pub frontier: i32,
    pub tokens: Vec<i32>,
    pub max_abs: f32,
    pub rel_l2: f32,
    pub payload_equal: bool,
    pub stop_accepted: Vec<i32>,
    pub stop_max_abs: f32,
    pub stop_rel_l2: f32,
    pub stop_payload_equal: bool,
}

/// The MTP losslessness evidence: the drafting stream, the accepted counts
/// and the state checks. `block` is None when no verify accepted three or
/// more tokens inside the budget.
#[derive(Clone, Debug)]
pub struct MtpReport {
    pub ids: Vec<i32>,
    pub counts: Vec<usize>,
    pub block: Option<MtpBlock>,
    pub prefix_reuse: Option<bool>,
    pub snapshot_restore: bool,
    pub sampling_ids: Vec<i32>,
    pub sampling_counts: Vec<usize>,
}

impl MtpReport {
    pub fn tokens_per_call(&self) -> f64 {
        self.ids.len() as f64 / self.counts.len().max(1) as f64
    }

    pub fn max_accepted(&self) -> usize {
        self.counts.iter().copied().max().unwrap_or(0)
    }
}

/// The sampling temperature the trial compares, and its seed.
pub(super) const MTP_TEMPERATURE: f32 = 0.8;
const MTP_TOP_P: f32 = 0.95;
const MTP_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// A verify block worth the stop check: the stop has at least one draft
/// behind it in the accepted prefix.
pub(super) const MTP_BLOCK_MIN: usize = 3;
const MTP_IO_BYTES: usize = 65536;

fn mtp_scratch(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ds4-check-{}-mtp-{tag}.payload",
        std::process::id()
    ))
}

/// The first byte two files disagree on, or None when they are equal.
fn mtp_first_diff(a: &Path, b: &Path) -> Result<Option<usize>> {
    let io = |e: std::io::Error| fail(format!("MTP payload compare: {e}"));
    let mut left = std::fs::File::open(a).map_err(io)?;
    let mut right = std::fs::File::open(b).map_err(io)?;
    let la = left.metadata().map_err(io)?.len();
    let lb = right.metadata().map_err(io)?.len();
    let mut x = [0u8; MTP_IO_BYTES];
    let mut y = [0u8; MTP_IO_BYTES];
    let mut offset = 0usize;
    loop {
        let n = left.read(&mut x).map_err(io)?;
        if n == 0 {
            return Ok((la != lb).then_some(offset));
        }
        let m = right.read(&mut y[..n]).map_err(io)?;
        if m != n || x[..n] != y[..n] {
            let at = x[..n.min(m)]
                .iter()
                .zip(&y[..n.min(m)])
                .position(|(p, q)| p != q);
            return Ok(Some(offset + at.unwrap_or(0)));
        }
        offset += n;
    }
}

fn mtp_max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn mtp_rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        num += d * d;
        den += f64::from(x) * f64::from(x);
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt() as f32
    }
}

fn mtp_plain_greedy(model: &Model, prompt: &TokenBuffer, ctx: i32, n: usize) -> Result<Vec<i32>> {
    let mut s = model.session(ctx)?;
    s.sync(prompt)?;
    let mut ids = Vec::with_capacity(n);
    while ids.len() < n {
        let token = s.argmax();
        if model.token_is_stop(token) {
            break;
        }
        s.eval(token)?;
        ids.push(token);
    }
    Ok(ids)
}

fn mtp_continue(
    s: &mut Session<'_>,
    model: &Model,
    n: usize,
    eos: i32,
    ids: &mut Vec<i32>,
    counts: &mut Vec<usize>,
) -> Result<()> {
    while ids.len() < n {
        let token = s.argmax();
        if model.token_is_stop(token) {
            break;
        }
        let accepted = s.eval_speculative_argmax(token, (n - ids.len()) as i32, eos)?;
        if accepted.is_empty() {
            return Err(fail("the speculative step accepted no token"));
        }
        counts.push(accepted.len());
        ids.extend_from_slice(&accepted);
    }
    Ok(())
}

/// The drafting stream plus the state captured before its first multi-token
/// accepted block.
struct MtpCapture {
    ids: Vec<i32>,
    counts: Vec<usize>,
    block: Option<(i32, Vec<i32>)>,
    snap: SessionSnapshot,
}

fn mtp_capture(
    model: &Model,
    prompt: &TokenBuffer,
    ctx: i32,
    n: usize,
    eos: i32,
) -> Result<MtpCapture> {
    let mut s = model.session(ctx)?;
    s.sync(prompt)?;
    let mut snap = SessionSnapshot::new()?;
    let mut taken = false;
    let mut block = None;
    let mut ids = Vec::with_capacity(n);
    let mut counts = Vec::new();
    while ids.len() < n {
        let token = s.argmax();
        if model.token_is_stop(token) {
            break;
        }
        if !taken {
            s.save_snapshot(&mut snap)?;
        }
        let frontier = s.pos();
        let accepted = s.eval_speculative_argmax(token, (n - ids.len()) as i32, eos)?;
        if accepted.is_empty() {
            return Err(fail("the speculative step accepted no token"));
        }
        if !taken && accepted.len() >= MTP_BLOCK_MIN {
            taken = true;
            block = Some((frontier, accepted.clone()));
        }
        counts.push(accepted.len());
        ids.extend_from_slice(&accepted);
    }
    Ok(MtpCapture {
        ids,
        counts,
        block,
        snap,
    })
}

/// A session whose frontier is `frontier`, reached by plain one-row decode so
/// the drafter's published hidden row is live (a snapshot restore drops it).
fn mtp_replay<'a>(
    model: &'a Model,
    prompt: &TokenBuffer,
    ctx: i32,
    frontier: i32,
) -> Result<Session<'a>> {
    let mut s = model.session(ctx)?;
    s.sync(prompt)?;
    while s.pos() < frontier {
        let token = s.argmax();
        if model.token_is_stop(token) {
            return Err(fail(format!(
                "the MTP replay hit a stop at pos {}",
                s.pos()
            )));
        }
        s.eval(token)?;
    }
    if s.pos() != frontier {
        return Err(fail(format!(
            "the MTP replay reached {} not {}",
            s.pos(),
            frontier
        )));
    }
    Ok(s)
}

/// Compare the state after a speculative block with a one-row replay of the
/// same tokens: the frontier logits, the argmax, and the saved payloads.
fn mtp_compare(
    s: &mut Session<'_>,
    snap: &SessionSnapshot,
    tokens: &[i32],
    spec_logits: &[f32],
    vocab: usize,
    tag: &str,
) -> Result<(f32, f32, bool, bool)> {
    let spec_path = mtp_scratch(&format!("{tag}-spec"));
    let plain_path = mtp_scratch(&format!("{tag}-plain"));
    s.save_payload(&spec_path)?;
    s.load_snapshot(snap)?;
    for &token in tokens {
        s.eval(token)?;
    }
    let plain_logits = s.copy_logits(vocab)?;
    s.save_payload(&plain_path)?;
    let same_pick = argmax(spec_logits) == argmax(&plain_logits);
    let abs = mtp_max_abs(spec_logits, &plain_logits);
    let rel = mtp_rel_l2(spec_logits, &plain_logits);
    let bytes = mtp_first_diff(&spec_path, &plain_path)?.is_none();
    let _ = std::fs::remove_file(&spec_path);
    let _ = std::fs::remove_file(&plain_path);
    Ok((abs, rel, same_pick, bytes))
}

fn mtp_plain_sampled(
    model: &Model,
    prompt: &TokenBuffer,
    ctx: i32,
    n: usize,
    sampler: crate::Sampler,
) -> Result<Vec<i32>> {
    let mut s = model.session(ctx)?;
    s.sync(prompt)?;
    let mut rng = MTP_SEED;
    let mut ids = Vec::with_capacity(n);
    while ids.len() < n {
        let token = s.sample(
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
            &mut rng,
        );
        if model.token_is_stop(token) {
            break;
        }
        s.eval(token)?;
        ids.push(token);
    }
    Ok(ids)
}

fn mtp_spec_sampled(
    model: &Model,
    prompt: &TokenBuffer,
    ctx: i32,
    n: usize,
    eos: i32,
    sampler: crate::Sampler,
) -> Result<(Vec<i32>, Vec<usize>)> {
    let mut s = model.session(ctx)?;
    s.sync(prompt)?;
    let mut rng = MTP_SEED;
    let mut ids = Vec::with_capacity(n);
    let mut counts = Vec::new();
    while ids.len() < n {
        let token = s.sample(
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
            &mut rng,
        );
        if model.token_is_stop(token) {
            break;
        }
        let accepted = s.eval_speculative(
            token,
            (n - ids.len()) as i32,
            eos,
            sampler,
            &mut rng,
            &mut |prefix: &[i32]| prefix.len(),
        )?;
        if accepted.is_empty() {
            return Err(fail("the sampled speculative step accepted no token"));
        }
        counts.push(accepted.len());
        ids.extend_from_slice(&accepted);
    }
    Ok((ids, counts))
}

/// The Gemma-4 MTP contract on a bound drafter: drafting may change
/// acceptance and speed, never the target's stream. Greedy off/on, the state
/// after the first multi-token block against a one-row replay, prefix reuse,
/// snapshot restore, a stop inside an accepted block, and sample-and-match at
/// temperature 0.8. Every token comparison is exact; a payload that differs
/// is reported, and the exact-prefill run requires it to match.
pub fn gemma4_mtp_gate(model: &Model, prompt: &[i32], ctx: i32, n: usize) -> Result<MtpReport> {
    if model.family() != ModelFamily::Gemma4Vq {
        return Err(fail("the MTP gate is Gemma-4 only"));
    }
    if model.mtp().is_none() {
        return Err(fail("no Gemma-4 drafter is bound"));
    }
    if prompt.len() < 2 {
        return Err(fail("the MTP gate needs a prompt of at least two tokens"));
    }
    if n < MTP_BLOCK_MIN {
        return Err(fail(format!(
            "the MTP gate needs at least {MTP_BLOCK_MIN} tokens"
        )));
    }
    let buf = TokenBuffer::from_tokens(prompt.to_vec());
    let vocab = usize::try_from(model.vocab().n_vocab()).unwrap_or(0);
    let eos = model.token_eos();
    // The engine's switch reading (ds4_gemma4_knob_bool): 1, true or on.
    let exact = matches!(
        std::env::var("DS4_GEMMA4_EXACT_PREFILL").as_deref(),
        Ok("1" | "true" | "on")
    );

    let plain = mtp_plain_greedy(model, &buf, ctx, n)?;
    let cap = mtp_capture(model, &buf, ctx, n, eos)?;
    if plain != cap.ids {
        return Err(fail(format!(
            "drafting moved the greedy stream: plain {:?}, drafting {:?}",
            plain, cap.ids
        )));
    }

    let block = match cap.block {
        Some((frontier, tokens)) => {
            let mut s = mtp_replay(model, &buf, ctx, frontier)?;
            let first = s.argmax();
            let accepted = s.eval_speculative_argmax(first, tokens.len() as i32, eos)?;
            if accepted != tokens {
                return Err(fail(format!(
                    "the replayed block differs: {:?} against {:?}",
                    accepted, tokens
                )));
            }
            let logits = s.copy_logits(vocab)?;
            let (max_abs, rel_l2, pick, payload_equal) =
                mtp_compare(&mut s, &cap.snap, &tokens, &logits, vocab, "block")?;
            if !pick {
                return Err(fail("the block frontier moved its argmax"));
            }
            if exact && !payload_equal {
                return Err(fail("the exact path's block payloads differ"));
            }
            let mut stop = mtp_replay(model, &buf, ctx, frontier)?;
            let first = stop.argmax();
            let stop_accepted =
                stop.eval_speculative_argmax(first, tokens.len() as i32, tokens[1])?;
            if stop_accepted != tokens[..2] {
                return Err(fail(format!(
                    "the stop round accepted {:?}, want {:?}",
                    stop_accepted,
                    &tokens[..2]
                )));
            }
            let stop_logits = stop.copy_logits(vocab)?;
            let (stop_max_abs, stop_rel_l2, stop_pick, stop_payload_equal) = mtp_compare(
                &mut stop,
                &cap.snap,
                &tokens[..2],
                &stop_logits,
                vocab,
                "stop",
            )?;
            if !stop_pick {
                return Err(fail("the stop frontier moved its argmax"));
            }
            if exact && !stop_payload_equal {
                return Err(fail("the exact path's stop payloads differ"));
            }
            Some(MtpBlock {
                frontier,
                tokens,
                max_abs,
                rel_l2,
                payload_equal,
                stop_accepted,
                stop_max_abs,
                stop_rel_l2,
                stop_payload_equal,
            })
        }
        None => None,
    };

    let prefix_reuse = {
        // The kept prefix must land on the prefill grid, or the fresh and the
        // reused syncs run different chunk widths and the comparison measures
        // the chunk trade, not reuse (session_items' own rule). The exact
        // path is one row at a time, so any split is exact there.
        let probe = model.session(ctx)?;
        let grid = probe.host().prefill_cap as usize;
        drop(probe);
        let split = if exact {
            prompt.len() / 2
        } else if grid > 0 && prompt.len() > grid {
            (prompt.len() - 1) / grid * grid
        } else {
            0
        };
        if split >= 2 && split < prompt.len() {
            let mut s = model.session(ctx)?;
            s.sync(&TokenBuffer::from_tokens(prompt[..split].to_vec()))?;
            s.sync(&buf)?;
            let mut ids = Vec::with_capacity(n);
            let mut _counts = Vec::new();
            mtp_continue(&mut s, model, n, eos, &mut ids, &mut _counts)?;
            if ids != cap.ids {
                return Err(fail(
                    "the reused-prefix stream differs from the fresh stream",
                ));
            }
            Some(true)
        } else {
            None
        }
    };

    let mut s = model.session(ctx)?;
    s.sync(&buf)?;
    let half = n / 2;
    let mut head = Vec::new();
    let mut head_counts = Vec::new();
    mtp_continue(&mut s, model, half, eos, &mut head, &mut head_counts)?;
    let mut snap = SessionSnapshot::new()?;
    s.save_snapshot(&mut snap)?;
    let mut first = head.clone();
    let mut first_counts = head_counts.clone();
    mtp_continue(&mut s, model, n, eos, &mut first, &mut first_counts)?;
    let mut second = head.clone();
    let mut second_counts = head_counts.clone();
    s.load_snapshot(&snap)?;
    mtp_continue(&mut s, model, n, eos, &mut second, &mut second_counts)?;
    if first != second {
        return Err(fail("the restored snapshot stream differs"));
    }
    if first != cap.ids {
        return Err(fail("the snapshot stream differs from the fresh stream"));
    }

    let sampler = crate::Sampler {
        temperature: MTP_TEMPERATURE,
        top_k: 0,
        top_p: MTP_TOP_P,
        min_p: 0.0,
    };
    let sampled_n = (n / 2).max(4);
    let sampled_plain = mtp_plain_sampled(model, &buf, ctx, sampled_n, sampler)?;
    let (sampling_ids, sampling_counts) =
        mtp_spec_sampled(model, &buf, ctx, sampled_n, eos, sampler)?;
    if sampled_plain != sampling_ids {
        return Err(fail(format!(
            "sampled ids differ: plain {:?}, drafting {:?}",
            sampled_plain, sampling_ids
        )));
    }

    Ok(MtpReport {
        ids: cap.ids,
        counts: cap.counts,
        block,
        prefix_reuse,
        snapshot_restore: true,
        sampling_ids,
        sampling_counts,
    })
}
