//! The request sampler over one logits row: temperature, top-k, top-p and
//! min-p with the native engine's xorshift stream (`ds4.c`
//! `sample_top_p_min_p` is its oracle). Plain decode and draft verification
//! draw through this one function, so the draw a speculative step hands on
//! is a plain draw of the same row.

use std::cell::RefCell;

/// Relative probabilities bounding growing heads of the distribution; 0 is
/// every token. A head token outranks every token below its floor.
const HEAD_FLOORS: [f32; 3] = [1e-3, 1e-6, 0.0];
/// The native engine's top-k bound.
const TOP_K_MAX: usize = 1024;

/// Sampling parameters of one request; temperature 0 is argmax.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
}

impl Sampler {
    /// Argmax.
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
    };
}

#[derive(Clone, Copy)]
struct Cand {
    id: i32,
    logit: f32,
    prob: f32,
}

/// The verify draw a speculative step handed to the next position (session
/// `pos`, native `generation`, the stream state `rng` its draws left): the
/// next plain sample there with the same sampler and stream returns it;
/// anything else, such as a new request reusing the frontier, drops it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PendingDraw {
    pub pos: i32,
    pub generation: u64,
    pub token: i32,
    pub sampler: Sampler,
    pub rng: u64,
}

impl PendingDraw {
    /// Whether the plain sample at `pos` with `sampler` and stream `rng` is
    /// the one this draw was handed to.
    pub(crate) fn here(&self, pos: i32, generation: u64, sampler: &Sampler, rng: u64) -> bool {
        self.pos == pos
            && self.generation == generation
            && self.sampler == *sampler
            && self.rng == rng
    }
}

thread_local! {
    static CANDS: RefCell<Vec<Cand>> = const { RefCell::new(Vec::new()) };
    static ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// `f` over this thread's reusable row of `len` floats.
pub(crate) fn with_row<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    ROW.with(|row| {
        let mut row = row.borrow_mut();
        row.resize(len, 0.0);
        f(&mut row)
    })
}

fn rng_next(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9e37_79b9_7f4a_7c15;
    }
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn rng_f32(state: &mut u64) -> f32 {
    ((rng_next(state) >> 40) & 0xff_ffff) as f32 / 16_777_216.0
}

/// The native engine's sampler (`ds4_set_host_sampler`): every native
/// draw, bank lanes and verify rows included, runs [`draw`].
///
/// # Safety
/// `logits` holds `n` floats and `rng` is a live stream, as native passes.
pub(crate) unsafe extern "C" fn host_draw(
    logits: *const f32,
    n: u32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    rng: *mut u64,
    excluded0: i32,
    excluded1: i32,
) -> i32 {
    if logits.is_null() || rng.is_null() || n == 0 {
        return -1;
    }
    let s = Sampler {
        temperature,
        top_k,
        top_p,
        min_p,
    };
    // SAFETY: the caller's contract above.
    let (row, rng) = unsafe { (std::slice::from_raw_parts(logits, n as usize), &mut *rng) };
    draw(row, &s, rng, [excluded0, excluded1])
}

/// One draw from `logits` with `s` and the stream `rng`, never returning an
/// `excluded` id (-1: none). Returns -1 when every token is excluded.
pub fn draw(logits: &[f32], s: &Sampler, rng: &mut u64, excluded: [i32; 2]) -> i32 {
    let skip = |i: usize| excluded.contains(&(i as i32));
    if s.temperature <= 0.0 {
        return argmax(logits, &skip, excluded);
    }
    let top_p = if s.top_p <= 0.0 || s.top_p > 1.0 {
        1.0
    } else {
        s.top_p
    };
    let min_p = s.min_p.max(0.0);
    if s.top_k > 0 {
        return draw_top_k(logits, s, top_p, min_p, rng, &skip, excluded);
    }

    // The top token sets the scale: p = exp((v - max) / t) is 1 for it.
    let mut max = f32::NEG_INFINITY;
    let mut best = if excluded[0] < 0 { 0 } else { -1 };
    let mut finite = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if skip(i) || !v.is_finite() {
            continue;
        }
        finite += 1;
        if best < 0 || v > max {
            max = v;
            best = i as i32;
        }
    }
    if finite == 0 {
        return argmax(logits, &skip, excluded);
    }
    let prob = |v: f32| ((v - max) / s.temperature).exp();
    if top_p >= 1.0 {
        return draw_min_p(logits, prob, min_p, best, rng, &skip);
    }

    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if !skip(i) && v.is_finite() {
            sum += prob(v);
        }
    }
    if sum <= 0.0 || !sum.is_finite() {
        return best;
    }
    CANDS.with(|cands| {
        let mut cands = cands.borrow_mut();
        for floor in HEAD_FLOORS {
            cands.clear();
            for (i, &v) in logits.iter().enumerate() {
                if skip(i) || !v.is_finite() {
                    continue;
                }
                let p = prob(v);
                if p >= floor {
                    cands.push(Cand {
                        id: i as i32,
                        logit: v,
                        prob: p,
                    });
                }
            }
            // Stable: equal logits keep token order, as the full sort does.
            cands.sort_by(|a, b| b.logit.total_cmp(&a.logit));
            let whole = cands.len() == finite;
            if let Some(id) = walk_top_p(&cands, sum, top_p, min_p, whole, rng) {
                return if id < 0 { best } else { id };
            }
        }
        best
    })
}

/// The nucleus walk over `cands` sorted by logit. None when the nucleus
/// may reach past this head (`whole` false), so a larger head must decide.
fn walk_top_p(
    cands: &[Cand],
    sum: f32,
    top_p: f32,
    min_p: f32,
    whole: bool,
    rng: &mut u64,
) -> Option<i32> {
    let min_prob = (cands[0].prob / sum) * min_p;
    let mut filtered_sum = 0.0f32;
    let mut filtered = 0usize;
    let mut closed = false;
    for (i, c) in cands.iter().enumerate() {
        if i > 0 && c.prob / sum < min_prob {
            closed = true;
            break;
        }
        filtered_sum += c.prob;
        filtered += 1;
        if filtered_sum / sum >= top_p {
            closed = true;
            break;
        }
    }
    if !closed && !whole {
        return None;
    }
    if filtered == 0 {
        return Some(-1);
    }
    let mut r = rng_f32(rng) * filtered_sum;
    for c in &cands[..filtered] {
        r -= c.prob;
        if r <= 0.0 {
            return Some(c.id);
        }
    }
    Some(cands[filtered - 1].id)
}

/// top-p 1: one pass over the tokens min-p keeps, in token order.
fn draw_min_p(
    logits: &[f32],
    prob: impl Fn(f32) -> f32,
    min_p: f32,
    best: i32,
    rng: &mut u64,
    skip: &impl Fn(usize) -> bool,
) -> i32 {
    let kept = |i: usize, v: f32| !skip(i) && v.is_finite() && prob(v) >= min_p;
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if kept(i, v) {
            sum += prob(v);
        }
    }
    if sum <= 0.0 || !sum.is_finite() {
        return best;
    }
    let mut r = rng_f32(rng) * sum;
    for (i, &v) in logits.iter().enumerate() {
        if !kept(i, v) {
            continue;
        }
        r -= prob(v);
        if r <= 0.0 {
            return i as i32;
        }
    }
    best
}

fn draw_top_k(
    logits: &[f32],
    s: &Sampler,
    top_p: f32,
    min_p: f32,
    rng: &mut u64,
    skip: &impl Fn(usize) -> bool,
    excluded: [i32; 2],
) -> i32 {
    let k = (s.top_k as usize).min(TOP_K_MAX).min(logits.len());
    // Descending insertion: a later equal logit lands after earlier ones.
    let mut top: Vec<(i32, f32)> = Vec::with_capacity(k);
    for (i, &v) in logits.iter().enumerate() {
        if skip(i) || !v.is_finite() || (top.len() == k && v <= top[k - 1].1) {
            continue;
        }
        if top.len() == k {
            top.pop();
        }
        let at = top.partition_point(|&(_, t)| t >= v);
        top.insert(at, (i as i32, v));
    }
    if top.is_empty() {
        return argmax(logits, skip, excluded);
    }
    let max = top[0].1;
    let probs: Vec<f32> = top
        .iter()
        .map(|&(_, v)| ((v - max) / s.temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return top[0].0;
    }
    let min_prob = (probs[0] / sum) * min_p;
    let mut filtered_sum = 0.0f32;
    let mut filtered = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        if i > 0 && p / sum < min_prob {
            break;
        }
        filtered_sum += p;
        filtered += 1;
        if filtered_sum / sum >= top_p {
            break;
        }
    }
    let mut r = rng_f32(rng) * filtered_sum;
    for (&(id, _), &p) in top.iter().zip(&probs).take(filtered) {
        r -= p;
        if r <= 0.0 {
            return id;
        }
    }
    top[filtered - 1].0
}

/// Sample-and-match over a trial's verify rows: row `i` (the target after
/// `tokens[..=i]`) is drawn in order with `s`, and the first draw that
/// differs from the next draft ends the trial. `target[i]` is row `i`'s draw;
/// rows past the end repeat the last draw, which acceptance never reads.
/// Every draw is spent: `rng` ends past all of them, so a draw whose position
/// the host decides itself never conditions a later one ([`pending_token`]
/// hands on the last). Returns how many rows were drawn.
pub fn match_rows<'r>(
    row: impl Fn(usize) -> &'r [f32],
    tokens: &[i32],
    s: &Sampler,
    rng: &mut u64,
    target: &mut [i32],
) -> usize {
    let n = tokens.len().min(target.len());
    let mut drawn = 0;
    while drawn < n {
        target[drawn] = draw(row(drawn), s, rng, [-1, -1]);
        drawn += 1;
        if drawn == n || target[drawn - 1] != tokens[drawn] {
            break;
        }
    }
    for i in drawn..n {
        target[i] = target[drawn - 1];
    }
    drawn
}

/// The token a speculative step leaves for the next position: the draw of
/// the last kept row, once `kept` tokens are committed. It stands only when
/// the host kept every accepted draft (no `cut`: a cut row's draw matched,
/// so it is conditioned) and samples that position with the verify sampler;
/// otherwise the position takes a fresh draw past all of the trial's.
pub fn pending_token(
    target: &[i32],
    drawn: usize,
    kept: usize,
    cut: bool,
    same_sampler: bool,
) -> Option<i32> {
    if cut || !same_sampler || kept == 0 || kept > drawn {
        return None;
    }
    target.get(kept - 1).copied()
}

/// The first largest finite logit outside `excluded`; without exclusions
/// token 0 when none is finite, as the native argmax.
fn argmax(logits: &[f32], skip: &impl Fn(usize) -> bool, excluded: [i32; 2]) -> i32 {
    let mut best = -1i32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if skip(i) || !v.is_finite() {
            continue;
        }
        if best < 0 || v > best_v {
            best = i as i32;
            best_v = v;
        }
    }
    if best < 0 && excluded == [-1, -1] {
        0
    } else {
        best
    }
}

#[cfg(test)]
mod tests {
    use super::{draw, Sampler};

    fn sampler(temperature: f32, top_k: i32, top_p: f32, min_p: f32) -> Sampler {
        Sampler {
            temperature,
            top_k,
            top_p,
            min_p,
        }
    }

    /// A peaked row over a large vocabulary: a few strong tokens, a long
    /// tail in [-spread, spread].
    fn row(n: usize, spread: f32) -> Vec<f32> {
        let mut x = 0x1234_5678_9abc_def0u64;
        let mut v: Vec<f32> = (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0) * spread
            })
            .collect();
        for k in 0..6 {
            v[(k * 7919) % n] += 3.0 * spread - k as f32;
        }
        v
    }

    #[test]
    fn greedy_takes_the_first_largest_allowed_token() {
        let v = [0.5, 2.0, 2.0, -1.0];
        assert_eq!(draw(&v, &Sampler::GREEDY, &mut 1, [-1, -1]), 1);
        assert_eq!(draw(&v, &Sampler::GREEDY, &mut 1, [1, -1]), 2);
        assert_eq!(draw(&[f32::NAN; 3], &Sampler::GREEDY, &mut 1, [-1, -1]), 0);
        assert_eq!(draw(&[f32::NAN; 3], &Sampler::GREEDY, &mut 1, [0, -1]), -1);
    }

    #[test]
    fn draws_stay_inside_top_k_top_p_and_min_p() {
        // Probabilities 0.5, 0.25, 0.125, 0.0625 ... (logit step ln 2).
        let v: Vec<f32> = (0..12)
            .map(|i| -(i as f32) * std::f32::consts::LN_2)
            .collect();
        let mut rng = 7;
        for _ in 0..4000 {
            assert!(draw(&v, &sampler(1.0, 0, 0.7, 0.0), &mut rng, [-1, -1]) <= 1);
            assert!(draw(&v, &sampler(1.0, 3, 1.0, 0.0), &mut rng, [-1, -1]) <= 2);
            assert!(draw(&v, &sampler(1.0, 0, 1.0, 0.2), &mut rng, [-1, -1]) <= 2);
            assert_ne!(draw(&v, &sampler(1.0, 0, 0.9, 0.0), &mut rng, [0, -1]), 0);
        }
    }

    #[test]
    fn frequencies_follow_the_distribution() {
        let v = [1.0f32.ln(), 2.0f32.ln(), 3.0f32.ln(), 4.0f32.ln()];
        let mut counts = [0u32; 4];
        let mut rng = 99;
        let n = 40_000;
        for _ in 0..n {
            counts[draw(&v, &sampler(1.0, 0, 0.999, 0.0), &mut rng, [-1, -1]) as usize] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            let p = (i + 1) as f64 / 10.0;
            let sigma = (n as f64 * p * (1.0 - p)).sqrt();
            assert!((c as f64 - n as f64 * p).abs() < 4.0 * sigma, "{counts:?}");
        }
    }

    /// Codex's case on PR #129: two uniform rows, draft 0 for position 1,
    /// whose policy differs from the verify sampler (here an EOS exclusion
    /// that keeps both tokens), so the host cuts a matched draft there. The
    /// token the plain loop then samples at position 1 must stay uniform.
    #[test]
    fn a_cut_position_draws_unconditioned() {
        let uniform = [0.0f32, 0.0];
        let s = sampler(1.0, 0, 1.0, 0.0);
        let (mut zeros, runs) = (0, 20_000);
        let mut rng = 0x5eed;
        for _ in 0..runs {
            let tokens = [7, 0];
            let mut target = [0; 2];
            let drawn = super::match_rows(|_| &uniform[..], &tokens, &s, &mut rng, &mut target);
            let keep = if target[0] == tokens[1] { 2 } else { 1 };
            // The host cuts to `first`: position 1 is its own decision.
            let cut = keep > 1;
            let kept = 1;
            let next = match super::pending_token(&target, drawn, kept, cut, false) {
                Some(token) => token,
                None => draw(&uniform, &s, &mut rng, [5, -1]),
            };
            zeros += (next == 0) as u32;
        }
        let share = zeros as f64 / runs as f64;
        assert!((share - 0.5).abs() < 0.02, "token 0 share {share}");
    }

    /// A free position keeps the target's distribution: the accepted draft
    /// or, after a mismatch, the handed-on draw.
    #[test]
    fn a_free_position_keeps_the_distribution() {
        let uniform = [0.0f32, 0.0];
        let s = sampler(1.0, 0, 1.0, 0.0);
        let (mut zeros, runs) = (0, 20_000);
        let mut rng = 0xfeed;
        for _ in 0..runs {
            let tokens = [7, 0];
            let mut target = [0; 2];
            let drawn = super::match_rows(|_| &uniform[..], &tokens, &s, &mut rng, &mut target);
            let kept = if target[0] == tokens[1] { 2 } else { 1 };
            let token = if kept == 2 {
                tokens[1]
            } else {
                super::pending_token(&target, drawn, kept, false, true).unwrap()
            };
            zeros += (token == 0) as u32;
        }
        let share = zeros as f64 / runs as f64;
        assert!((share - 0.5).abs() < 0.02, "token 0 share {share}");
    }

    /// A request that reuses another's frontier with the same sampler draws
    /// from its own stream, not the other request's handed-on draw.
    #[test]
    fn a_pending_draw_belongs_to_its_stream() {
        let s = sampler(1.0, 0, 0.95, 0.0);
        let pending = super::PendingDraw {
            pos: 40,
            generation: 3,
            token: 9,
            sampler: s,
            rng: 0xabc,
        };
        assert!(pending.here(40, 3, &s, 0xabc));
        assert!(!pending.here(40, 3, &s, 0x5eed));
        assert!(!pending.here(41, 3, &s, 0xabc));
        assert!(!pending.here(40, 3, &sampler(0.7, 0, 0.95, 0.0), 0xabc));
    }

    /// Uncut, the last kept row's draw is the next position's token.
    #[test]
    fn an_uncut_step_hands_on_its_last_draw() {
        assert_eq!(super::pending_token(&[3, 9, 9], 2, 2, false, true), Some(9));
        assert_eq!(super::pending_token(&[3, 9, 9], 2, 2, true, true), None);
        assert_eq!(super::pending_token(&[3, 9, 9], 2, 2, false, false), None);
    }

    #[test]
    fn head_first_order_matches_the_full_sort() {
        // Reference: the native walk over every token sorted by logit.
        fn full_sort(v: &[f32], t: f32, top_p: f32, rng: &mut u64) -> i32 {
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut c: Vec<(i32, f32, f32)> = v
                .iter()
                .enumerate()
                .map(|(i, &x)| (i as i32, x, ((x - max) / t).exp()))
                .collect();
            let sum: f32 = c.iter().map(|x| x.2).sum();
            c.sort_by(|a, b| b.1.total_cmp(&a.1));
            let (mut fs, mut n) = (0.0f32, 0);
            for x in &c {
                fs += x.2;
                n += 1;
                if fs / sum >= top_p {
                    break;
                }
            }
            let mut r = super::rng_f32(rng) * fs;
            for x in &c[..n] {
                r -= x.2;
                if r <= 0.0 {
                    return x.0;
                }
            }
            c[n - 1].0
        }
        for spread in [1.0, 3.0, 6.0] {
            let v = row(32_768, spread);
            for seed in 1..30u64 {
                let (mut a, mut b) = (seed, seed);
                let fast = draw(&v, &sampler(1.0, 0, 0.95, 0.0), &mut a, [-1, -1]);
                assert_eq!(fast, full_sort(&v, 1.0, 0.95, &mut b), "spread {spread}");
                assert_eq!(a, b);
            }
        }
    }
}
