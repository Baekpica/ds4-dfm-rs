//! The serving checklist: every ServingCaps category as one list, each item
//! run on the loaded model and printed beside its declaration, so every
//! family goes through the same procedure and the output shows what works
//! and what does not.
//!
//! Items run whether or not the caps declare them. A declared item that does
//! not work fails; a working undeclared item is a declaration gap; an item
//! that reused state but moved the output is wrong either way. Categories
//! without a check yet stay on the list as such.
//!
//! ```text
//! T prompt, m its split on the prefill grid, G fresh greedy tokens,
//! X a divergent filler, U = T + G + X
//! reuse/prefix      sync(T[..m]) sync(T)                    vs fresh T
//! reuse/rewind      sync(T[..m] + X) sync(T)                vs fresh T
//! reuse/continue    sync(T) eval(G) sync(U)                 vs fresh U,
//!                                                           T + eval(G + X)
//! disk/session      continue's state saved, dropped, loaded vs continue
//! snapshot/session  continue's state saved, X synced, loaded vs continue
//! banks/*           bank 0 seeded with T[..m] (partial: T[..m] + X);
//!                   warm, fork, partial and restored admits of T vs cold
//! banks/width       T and T[..m] + X admitted together      vs each alone
//! ```
//!
//! An item works when the engine kept state (the sync's kept prefix, the
//! admit's cached rows) and the continuation matches the reference. With m
//! on the prefill grid a kept prefix repeats the fresh arithmetic, so a
//! difference points at reuse, not at chunking. Where reuse changes the
//! arithmetic, a second reference without reuse carries the same change:
//! continue keeps rows decode wrote (U fed by decode), and a rewind kept off
//! the grid re-prefills from there (T synced in two parts split at the kept
//! row). A row agrees when its top-1 is a reference's. Two references set a
//! noise floor, the most they move a row's top-1 lead over its runner-up;
//! a pick a reference ranks within it is a near tie, any other flip reuse.
//! Session items compare the top-1 at the frontier and after each forced
//! token; banks expose only their greedy streams. Exactness is reported,
//! not required.

use std::collections::VecDeque;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::serving::{bank_opt_in_env, caps_from_shape, ReuseKind, ServingCaps, Support};
use crate::{
    v41_kvreuse_last_reused, BatchCtx, ContAdmit, ContDone, ContDriver, Error, Model, Result,
    Session, SessionSnapshot, TokenBuffer, V41KvClass, V41KvDisk, V41KvHit, V41RunOptions,
    V41Sampling,
};

/// Tokens decoded past each frontier.
const DECODE_TOKENS: usize = 16;
/// Length of the divergent filler X.
const FILLER_TOKENS: usize = 16;
/// A seeded source bank and a target.
const BANKS: i32 = 2;
/// The checklist's context is its need rounded up to this.
const CTX_ALIGN: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Item {
    Prefix,
    Rewind,
    Continue,
    SessionDisk,
    Snapshot,
    BankWarm,
    BankFork,
    BankPartial,
    BankDisk,
    BankWidth,
}

impl Item {
    const SESSION: [Item; 5] = [
        Self::Prefix,
        Self::Rewind,
        Self::Continue,
        Self::SessionDisk,
        Self::Snapshot,
    ];
    const BANK: [Item; 5] = [
        Self::BankWarm,
        Self::BankFork,
        Self::BankPartial,
        Self::BankDisk,
        Self::BankWidth,
    ];

    fn label(self) -> (&'static str, &'static str) {
        match self {
            Self::Prefix => ("reuse", "prefix"),
            Self::Rewind => ("reuse", "rewind"),
            Self::Continue => ("reuse", "continue"),
            Self::SessionDisk => ("disk", "session"),
            Self::Snapshot => ("snapshot", "session"),
            Self::BankWarm => ("banks", "warm"),
            Self::BankFork => ("banks", "fork"),
            Self::BankPartial => ("banks", "partial"),
            Self::BankDisk => ("banks", "disk"),
            Self::BankWidth => ("banks", "width"),
        }
    }

    /// The support the caps state for this item; `None` when they do not.
    fn declared(self, caps: &ServingCaps) -> Support {
        let reuse = if caps.reuse == ReuseKind::None {
            Support::None
        } else {
            caps.reuse_support
        };
        let partial = if caps.reuse == ReuseKind::Partial {
            reuse
        } else {
            Support::None
        };
        let bank = |s: Support| {
            if s == Support::None {
                s
            } else {
                caps.bank_support
            }
        };
        match self {
            Self::Prefix | Self::Continue => reuse,
            Self::Rewind => partial,
            Self::SessionDisk => caps.disk,
            Self::Snapshot => caps.snapshot,
            Self::BankWarm | Self::BankFork => bank(reuse),
            Self::BankPartial => bank(partial),
            Self::BankDisk => bank(caps.disk),
            Self::BankWidth => caps.bank_support,
        }
    }

    fn rewinds(self) -> bool {
        matches!(self, Self::Rewind | Self::BankPartial)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Measured {
    /// Kept state and matched the reference.
    Works(String),
    /// Prefilled from zero instead: nothing was kept.
    Cold,
    /// The engine or the setup refused the item.
    Refused(String),
    /// Kept state, but the continuation moved.
    Wrong(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Works,
    Undeclared,
    No,
    Fail,
    Wrong,
}

impl Verdict {
    fn of(declared: Support, measured: &Measured) -> Self {
        let declared = declared != Support::None;
        match measured {
            Measured::Wrong(_) => Self::Wrong,
            Measured::Works(_) if declared => Self::Works,
            Measured::Works(_) => Self::Undeclared,
            _ if declared => Self::Fail,
            _ => Self::No,
        }
    }

    fn failed(self) -> bool {
        matches!(self, Self::Fail | Self::Wrong)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Works => "works",
            Self::Undeclared => "works, undeclared",
            Self::No => "no",
            Self::Fail => "FAIL",
            Self::Wrong => "WRONG",
        }
    }
}

pub struct Checklist {
    caps: ServingCaps,
    ctx: i32,
    prompt: usize,
    split: usize,
    /// The session's prefill cap; 0 when no session opened.
    grid: u32,
    rows: Vec<(Item, Measured)>,
}

impl Checklist {
    /// No declared item fails and no item moves the output.
    pub fn passed(&self) -> bool {
        !self.verdicts().any(Verdict::failed)
    }

    fn verdicts(&self) -> impl Iterator<Item = Verdict> + '_ {
        self.rows
            .iter()
            .map(|(item, m)| Verdict::of(item.declared(&self.caps), m))
    }
}

fn support(s: Support) -> &'static str {
    match s {
        Support::None => "-",
        Support::Present => "present",
        Support::Qualified => "qualified",
    }
}

fn limit(v: Option<u32>) -> String {
    v.map_or_else(|| "-".into(), |v| v.to_string())
}

/// Categories the checklist lists but cannot run yet, with their declaration.
fn unchecked(c: &ServingCaps) -> [(&'static str, &'static str, String); 4] {
    [
        (
            "mtp",
            "lossless",
            format!("{:?} {} {:?}", c.mtp, support(c.mtp_support), c.spec_lane),
        ),
        (
            "context",
            "limit",
            format!(
                "max {} qualified {}",
                limit(c.ctx_max),
                limit(c.qualified_ctx)
            ),
        ),
        (
            "prompt",
            "limit",
            format!("qualified {}", limit(c.qualified_prompt)),
        ),
        (
            "media",
            "lane",
            (if c.media_serial { "serial" } else { "banks" }).into(),
        ),
    ]
}

impl fmt::Display for Checklist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.caps;
        // V4.1 runs without a session context or prefill grid.
        write!(f, "serving checklist: {:?} ({:?})", c.variant, c.family)?;
        if self.ctx > 0 {
            write!(f, " ctx {}", self.ctx)?;
        }
        write!(f, " prompt {} split {}", self.prompt, self.split)?;
        if self.grid > 0 {
            write!(f, " prefill grid {}", self.grid)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "  {:<9} {:<9} {:<10} {:<18} detail",
            "category", "item", "declared", "result"
        )?;
        for (item, measured) in &self.rows {
            let (category, name) = item.label();
            let declared = item.declared(c);
            let detail = match measured {
                Measured::Works(d) | Measured::Wrong(d) | Measured::Refused(d) => d.as_str(),
                Measured::Cold => "prefilled from zero: nothing kept",
            };
            writeln!(
                f,
                "  {category:<9} {name:<9} {:<10} {:<18} {detail}",
                support(declared),
                Verdict::of(declared, measured).label()
            )?;
        }
        for (category, name, declared) in unchecked(c) {
            writeln!(
                f,
                "  {category:<9} {name:<9} {:<10} {:<18} declared {declared}",
                "", "no check yet"
            )?;
        }
        let count = |v: Verdict| self.verdicts().filter(|x| *x == v).count();
        writeln!(
            f,
            "result: {} (works {}, FAIL {}, WRONG {}, undeclared {})",
            if self.passed() { "pass" } else { "FAIL" },
            count(Verdict::Works),
            count(Verdict::Fail),
            count(Verdict::Wrong),
            count(Verdict::Undeclared)
        )
    }
}

/// The context the checklist needs for a prompt of `n` tokens.
pub fn checklist_ctx(n: usize) -> i32 {
    let need = n + 2 * DECODE_TOKENS + FILLER_TOKENS + 1;
    i32::try_from(need.div_ceil(CTX_ALIGN) * CTX_ALIGN).unwrap_or(i32::MAX)
}

/// The last prefill-grid boundary inside the prompt, or its middle when the
/// prompt fits one chunk.
fn split_point(prompt: usize, grid: u32) -> usize {
    let grid = grid as usize;
    if grid > 0 && prompt > grid {
        return (prompt - 1) / grid * grid;
    }
    prompt / 2
}

fn fail(message: impl Into<String>) -> Error {
    Error {
        code: 1,
        message: message.into(),
    }
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ds4-check-{}-{name}.kv", std::process::id()))
}

struct Prompts {
    t: Vec<i32>,
    m: usize,
    /// T[..m] + X: leaves T at m.
    diverged: Vec<i32>,
    u: Vec<i32>,
}

impl Prompts {
    fn new(t: &[i32], m: usize, g: &[i32]) -> Self {
        let x: Vec<i32> = t.iter().take(FILLER_TOKENS).rev().copied().collect();
        Self {
            t: t.to_vec(),
            m,
            diverged: [&t[..m], &x].concat(),
            u: [t, g, &x].concat(),
        }
    }

    /// A filler repeating T would turn the rewind items into extensions.
    fn diverges(&self) -> bool {
        !self.t.starts_with(&self.diverged)
    }
}

/// Logits at a frontier and after each next token, with the tokens fed.
struct Rows {
    logits: Vec<Vec<f32>>,
    tokens: Vec<i32>,
}

/// First index of the largest logit.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Rows from the live frontier, feeding `forced`, or the greedy choice when
/// it is empty.
fn decode(s: &mut Session<'_>, vocab: usize, forced: &[i32]) -> Result<Rows> {
    let mut rows = Rows {
        logits: Vec::with_capacity(DECODE_TOKENS + 1),
        tokens: Vec::with_capacity(DECODE_TOKENS),
    };
    for step in 0..=DECODE_TOKENS {
        let logits = s.copy_logits(vocab)?;
        if step < DECODE_TOKENS {
            let token = forced
                .get(step)
                .copied()
                .unwrap_or_else(|| argmax(&logits) as i32);
            s.eval(token)?;
            rows.tokens.push(token);
        }
        rows.logits.push(logits);
    }
    Ok(rows)
}

/// A reference continuation and the arithmetic it stands for.
type Reference<'a> = (&'a str, &'a Rows);

/// Teacher-forced top-1 of `got` against references without reuse: a row
/// agrees when its top-1 is one of theirs. Each reference is an accepted
/// arithmetic, so a near-tie flip between them is noise, not reuse: a pick
/// a reference ranks within the noise floor of its own top-1 is a near tie.
/// Exactness and distance are against the first.
fn compare(refs: &[Reference<'_>], got: &Rows, kept: i32) -> Measured {
    let want = refs[0].1;
    let of = want.logits.len();
    let floor = noise_floor(refs);
    let mut agree = 0;
    let mut ties = 0;
    let mut each = vec![0; refs.len()];
    let mut exact = got.logits.len() == of;
    let mut max_abs = 0f32;
    let mut flips = Vec::new();
    for (row, (w, g)) in want.logits.iter().zip(&got.logits).enumerate() {
        let top = argmax(g);
        let hits: Vec<bool> = refs
            .iter()
            .map(|(_, r)| r.logits.get(row).is_some_and(|l| argmax(l) == top))
            .collect();
        agree += usize::from(hits.contains(&true));
        if !hits.contains(&true) {
            flips.push(flip(row, refs, g, top));
            let near = |l: &Vec<f32>| l[argmax(l)] - l[top] <= floor;
            ties += usize::from(
                refs.iter()
                    .any(|(_, r)| r.logits.get(row).is_some_and(near)),
            );
        }
        for (n, hit) in each.iter_mut().zip(hits) {
            *n += usize::from(hit);
        }

        exact &= w.len() == g.len();
        for (&a, &b) in w.iter().zip(g) {
            exact &= a.to_bits() == b.to_bits();
            max_abs = max_abs.max((b - a).abs());
        }
    }

    // "top-1 17/17 (prefill 16, decode-fed 17)" when there are several.
    let mut detail = format!("kept {kept}, top-1 {agree}/{of}");
    if refs.len() > 1 {
        let parts: Vec<String> = refs
            .iter()
            .zip(&each)
            .map(|((name, _), n)| format!("{name} {n}"))
            .collect();
        detail += &format!(" ({})", parts.join(", "));
    }
    if ties > 0 {
        detail += &format!(", near-tie {ties} <= {floor:.3}");
    }
    detail += &if exact {
        ", exact".to_string()
    } else {
        format!(", max_abs {max_abs:.3e}")
    };
    let detail = [detail]
        .into_iter()
        .chain(flips)
        .collect::<Vec<_>>()
        .join("; ");
    if agree + ties == of && got.logits.len() == of {
        Measured::Works(detail)
    } else {
        Measured::Wrong(detail)
    }
}

/// How far two accepted arithmetics move a decision: the largest change,
/// between the first two references, of a row's lead of its top-1 over its
/// runner-up. 0 with one reference, whose arithmetic is the item's own.
fn noise_floor(refs: &[Reference<'_>]) -> f32 {
    let [(_, a), (_, b), ..] = refs else {
        return 0.0;
    };
    let mut floor = 0f32;
    for (x, y) in a.logits.iter().zip(&b.logits) {
        let t = argmax(x);
        let Some(u) = (0..x.len())
            .filter(|&i| i != t)
            .max_by(|&i, &j| x[i].total_cmp(&x[j]))
        else {
            continue;
        };
        floor = floor.max(((x[t] - x[u]) - (y[t] - y[u])).abs());
    }
    floor
}

/// "row 9: 1234 +0.040 over prefill 5678 +0.120": how far `got` put its
/// pick over each reference's top-1, and how far that reference put its own.
fn flip(row: usize, refs: &[Reference<'_>], got: &[f32], top: usize) -> String {
    let leads: Vec<String> = refs
        .iter()
        .filter_map(|(name, r)| {
            let l = r.logits.get(row)?;
            let t = argmax(l);
            Some(format!("{name} {t} +{:.3}", l[t] - l[top]))
        })
        .collect();
    let t = argmax(&refs[0].1.logits[row]);
    format!(
        "row {row}: {top} +{:.3} over {}",
        got[top] - got[t],
        leads.join(", ")
    )
}

fn sync(s: &mut Session<'_>, tokens: &[i32]) -> Result<()> {
    s.sync(&TokenBuffer::from_tokens(tokens.to_vec()))
}

fn fresh(s: &mut Session<'_>, tokens: &[i32], vocab: usize) -> Result<Rows> {
    s.invalidate();
    sync(s, tokens)?;
    decode(s, vocab, &[])
}

/// T prefilled, then G decoded one token at a time, as a turn ends.
fn decoded(s: &mut Session<'_>, t: &[i32], g: &[i32]) -> Result<()> {
    s.invalidate();
    sync(s, t)?;
    g.iter().try_for_each(|&token| s.eval(token).map(|_| ()))
}

/// U with everything after T fed by decode: continue's history in decode
/// arithmetic, with no sync to keep anything.
fn fed(s: &mut Session<'_>, p: &Prompts, forced: &[i32], vocab: usize) -> Result<Rows> {
    decoded(s, &p.t, &p.u[p.t.len()..])?;
    decode(s, vocab, forced)
}

/// T prefilled in two syncs split at `k`: a rewind's chunking when it kept
/// `k` off the grid, from an extension the prefix item checks.
fn extended(
    s: &mut Session<'_>,
    t: &[i32],
    k: usize,
    forced: &[i32],
    vocab: usize,
) -> Result<Rows> {
    s.invalidate();
    sync(s, &t[..k])?;
    sync(s, t)?;
    decode(s, vocab, forced)
}

/// Builds `item`'s kept state and continues it; `None` when the sync would
/// prefill from zero.
fn serial(
    item: Item,
    s: &mut Session<'_>,
    p: &Prompts,
    g: &[i32],
    forced: &[i32],
    vocab: usize,
) -> Result<Option<(i32, Rows)>> {
    s.invalidate();
    let target: &[i32] = match item {
        Item::Prefix => {
            sync(s, &p.t[..p.m])?;
            &p.t
        }
        Item::Rewind => {
            sync(s, &p.diverged)?;
            &p.t
        }
        Item::Continue => {
            decoded(s, &p.t, g)?;
            &p.u
        }
        Item::SessionDisk => {
            decoded(s, &p.t, g)?;
            let path = scratch("session");
            let saved = s.save_payload(&path);
            s.invalidate();
            let loaded = saved.and_then(|()| s.load_payload(&path));
            let _ = std::fs::remove_file(&path);
            loaded?;
            &p.u
        }
        Item::Snapshot => {
            decoded(s, &p.t, g)?;
            let mut snapshot = SessionSnapshot::new()?;
            s.save_snapshot(&mut snapshot)?;
            sync(s, &p.diverged)?;
            s.load_snapshot(&snapshot)?;
            &p.u
        }
        _ => unreachable!("bank item {item:?} on a session"),
    };
    let kept = s.last_plan(target).start;
    if kept <= 0 {
        return Ok(None);
    }
    sync(s, target)?;
    Ok(Some((kept, decode(s, vocab, forced)?)))
}

/// The session items; returns the prompts the bank items reuse.
fn session_items(
    s: &mut Session<'_>,
    prompt: &[i32],
    vocab: usize,
    list: &mut Checklist,
) -> Result<Prompts> {
    list.grid = s.host().prefill_cap;
    list.split = split_point(prompt.len(), list.grid);
    let fresh_t = fresh(s, prompt, vocab)?;
    let p = Prompts::new(prompt, list.split, &fresh_t.tokens);
    let fresh_u = fresh(s, &p.u, vocab)?;
    // Continue keeps rows decode wrote; a fresh prefill writes them with
    // prefill arithmetic, so a near-tie may flip between the two.
    let fed_u = fed(s, &p, &fresh_u.tokens, vocab)?;

    // Restores compare with the live state they saved: same arithmetic.
    let mut live: Option<Rows> = None;
    for item in Item::SESSION {
        if item.rewinds() && !p.diverges() {
            list.rows.push((
                item,
                Measured::Refused("the filler repeats the prompt".into()),
            ));
            continue;
        }
        let forced = match item {
            Item::Prefix | Item::Rewind => &fresh_t.tokens,
            _ => &fresh_u.tokens,
        };
        let (kept, got) = match serial(item, s, &p, &fresh_t.tokens, forced, vocab) {
            Err(e) => {
                list.rows.push((item, Measured::Refused(e.message)));
                continue;
            }
            Ok(None) => {
                list.rows.push((item, Measured::Cold));
                continue;
            }
            Ok(Some(kept_rows)) => kept_rows,
        };

        // A rewind kept off the grid re-prefills from there: its chunking.
        let off_grid = match item {
            Item::Rewind if kept as usize != p.m => {
                Some(extended(s, &p.t, kept as usize, forced, vocab)?)
            }
            _ => None,
        };
        let mut refs: Vec<Reference<'_>> = match (item, &live) {
            (Item::Prefix | Item::Rewind, _) => vec![("prefill", &fresh_t)],
            (Item::SessionDisk | Item::Snapshot, Some(rows)) => vec![("live", rows)],
            _ => vec![("prefill", &fresh_u), ("decode-fed", &fed_u)],
        };
        if let Some(rows) = &off_grid {
            refs.push(("split", rows));
        }
        list.rows.push((item, compare(&refs, &got, kept)));

        if item == Item::Continue {
            live = Some(got);
        }
    }
    Ok(p)
}

/// Admits run together to completion: the rows the engine kept for each
/// and its tokens.
struct Admits {
    queue: VecDeque<ContAdmit>,
    kept: Vec<Option<i32>>,
    tokens: Vec<Vec<i32>>,
}

impl ContDriver for Admits {
    fn admit(&mut self) -> Option<ContAdmit> {
        self.queue.pop_front()
    }

    fn on_token(&mut self, user: usize, token: i32) -> bool {
        if let Some(tokens) = self.tokens.get_mut(user) {
            tokens.push(token);
        }
        true
    }

    fn on_done(&mut self, _user: usize, _tokens: &[i32], _finish: i32, _stats: ContDone) {}

    // The engine's verdict: a claim it cannot honour runs cold, silently.
    fn on_admitted(&mut self, user: usize, n_cached: i32, _n_computed: i32, _bank: i32) -> bool {
        if let Some(kept) = self.kept.get_mut(user) {
            *kept = Some(n_cached);
        }
        true
    }
}

/// Runs `admits` together; each admit's user is its index.
fn run_all(batch: &BatchCtx<'_>, admits: Vec<ContAdmit>) -> Result<Vec<(i32, Vec<i32>)>> {
    let n = admits.len();
    let mut driver = Admits {
        queue: admits
            .into_iter()
            .enumerate()
            .map(|(user, mut a)| {
                a.user = user;
                a
            })
            .collect(),
        kept: vec![None; n],
        tokens: vec![Vec::new(); n],
    };
    batch.continuous_generate(&mut driver)?;
    driver
        .kept
        .into_iter()
        .zip(driver.tokens)
        .map(|(kept, tokens)| {
            let kept = kept.ok_or_else(|| fail("the engine rejected the admit"))?;
            Ok((kept, tokens))
        })
        .collect()
}

fn run(batch: &BatchCtx<'_>, admit: ContAdmit) -> Result<(i32, Vec<i32>)> {
    Ok(run_all(batch, vec![admit])?.remove(0))
}

/// `tokens` on bank `place`. With `from`, it claims the prefix that bank
/// shares with them, as the server's warm plans do: a warm admit on the
/// same bank, a fork from another.
fn admit(
    batch: &BatchCtx<'_>,
    tokens: &[i32],
    max_new: usize,
    place: i32,
    from: Option<i32>,
) -> Result<ContAdmit> {
    let mut a = ContAdmit::cold(0, tokens.to_vec(), max_new as i32);
    a.place_bank = place + 1;
    let Some(source) = from else {
        return Ok(a);
    };
    let held = batch.bank_snapshot(source)?.tokens;
    let common = held.iter().zip(tokens).take_while(|(x, y)| x == y).count();
    a.n_cached = i32::try_from(common).unwrap_or(0);
    if source != place {
        a.fork_bank = source + 1;
    }
    Ok(a)
}

fn streams(want: &[i32], got: Result<(i32, Vec<i32>)>) -> Measured {
    let (kept, tokens) = match got {
        Ok(got) => got,
        Err(e) => return Measured::Refused(e.message),
    };
    if kept <= 0 {
        return Measured::Cold;
    }
    let matched = want.iter().zip(&tokens).take_while(|(a, b)| a == b).count();
    let of = want.len().max(tokens.len());
    let detail = format!("kept {kept}, tokens {matched}/{of}");
    if matched == of {
        Measured::Works(detail)
    } else {
        Measured::Wrong(detail)
    }
}

fn bank_items(model: &Model, p: &Prompts, ctx: i32) -> Vec<(Item, Measured)> {
    let refused = |e: Error| -> Vec<(Item, Measured)> {
        Item::BANK
            .iter()
            .map(|&item| (item, Measured::Refused(e.message.clone())))
            .collect()
    };
    // Opt-in families build banks only behind the switch the server sets.
    if let Some(key) = bank_opt_in_env(model.family()) {
        std::env::set_var(key, "1");
    }
    let need = p.u.len() + DECODE_TOKENS + 1;
    let total = i32::try_from(need * BANKS as usize).unwrap_or(i32::MAX);
    let batch = match model.batch_ctx_fit(ctx, BANKS, total) {
        Ok(batch) => batch,
        Err(e) => return refused(e),
    };
    if batch.max_seq() < BANKS {
        return refused(fail(format!(
            "{} bank fits; the items need {BANKS}",
            batch.max_seq()
        )));
    }

    // Bank 0 holds T[..m]; its payload feeds banks/disk.
    let k = DECODE_TOKENS;
    let seeded = admit(&batch, &p.t[..p.m], 1, 0, None).and_then(|a| run(&batch, a));
    if let Err(e) = seeded {
        return refused(e);
    }
    let file = scratch("bank");
    let saved = batch.save_bank_payload(0, &file);
    let fork = admit(&batch, &p.t, k, 1, Some(0)).and_then(|a| run(&batch, a));
    let warm = admit(&batch, &p.t, k, 0, Some(0)).and_then(|a| run(&batch, a));
    let want = match admit(&batch, &p.t, k, 1, None).and_then(|a| run(&batch, a)) {
        Ok((_, tokens)) => tokens,
        Err(e) => return refused(e),
    };
    let disk = saved.and_then(|()| {
        let len = std::fs::metadata(&file).map_or(0, |m| m.len());
        batch.load_bank_payload_range(1, &file, 0, len)?;
        run(&batch, admit(&batch, &p.t, k, 1, Some(1))?)
    });
    let _ = std::fs::remove_file(&file);

    // T[..m] + X decoded alone on bank 0: the partial fork's source and the
    // second width stream's reference.
    let alone = if p.diverges() {
        admit(&batch, &p.diverged, k, 0, None).and_then(|a| run(&batch, a))
    } else {
        Err(fail("the filler repeats the prompt"))
    };
    let (partial, width) = match &alone {
        Ok((_, alone)) => (
            admit(&batch, &p.t, k, 1, Some(0)).and_then(|a| run(&batch, a)),
            together(&batch, p, [&want, alone]),
        ),
        Err(e) => (
            Err(fail(e.message.clone())),
            Measured::Refused(e.message.clone()),
        ),
    };
    vec![
        (Item::BankWarm, streams(&want, warm)),
        (Item::BankFork, streams(&want, fork)),
        (Item::BankPartial, streams(&want, partial)),
        (Item::BankDisk, streams(&want, disk)),
        (Item::BankWidth, width),
    ]
}

/// T and T[..m] + X decoded together on two banks, each against its run
/// alone.
fn together(batch: &BatchCtx<'_>, p: &Prompts, alone: [&[i32]; 2]) -> Measured {
    let k = DECODE_TOKENS;
    let runs = [
        admit(batch, &p.t, k, 0, None),
        admit(batch, &p.diverged, k, 1, None),
    ]
    .into_iter()
    .collect::<Result<Vec<_>>>()
    .and_then(|admits| run_all(batch, admits));
    let runs = match runs {
        Ok(runs) => runs,
        Err(e) => return Measured::Refused(e.message),
    };
    let mut same = true;
    let mut counts = Vec::new();
    for (want, (_, got)) in alone.iter().zip(&runs) {
        let matched = want.iter().zip(got).take_while(|(a, b)| a == b).count();
        let of = want.len().max(got.len());
        same &= matched == of;
        counts.push(format!("{matched}/{of}"));
    }
    let detail = format!("tokens {}", counts.join(" + "));
    if same {
        Measured::Works(detail)
    } else {
        Measured::Wrong(detail)
    }
}

/// Runs every checklist item on `model` with `prompt` in `ctx` positions.
pub fn serving_check(model: &Model, prompt: &[i32], ctx: i32) -> Result<Checklist> {
    if prompt.len() < 2 {
        return Err(fail("the checklist needs a prompt of at least two tokens"));
    }
    let vocab = usize::try_from(model.vocab().n_vocab()).unwrap_or(0);
    let mut list = Checklist {
        caps: caps_from_shape(model.shape()),
        ctx,
        prompt: prompt.len(),
        split: 0,
        grid: 0,
        rows: Vec::new(),
    };
    let p = match model.session(ctx) {
        Ok(mut s) => session_items(&mut s, prompt, vocab, &mut list)?,
        Err(e) => {
            for item in Item::SESSION {
                list.rows.push((item, Measured::Refused(e.message.clone())));
            }
            list.split = split_point(prompt.len(), 0);
            Prompts::new(prompt, list.split, &[])
        }
    };
    list.rows.extend(bank_items(model, &p, ctx));
    Ok(list)
}

/// A store for the V4.1 disk item: each checkpoint in its own scratch file,
/// matched by token prefix under the run's identity.
#[derive(Default)]
struct ScratchKv {
    entries: Vec<(Vec<i32>, Vec<u64>, u64, PathBuf)>,
}

impl V41KvDisk for ScratchKv {
    fn find(&mut self, tokens: &[i32], sig: &[u64], digest: u64, have: u32) -> Option<V41KvHit> {
        let (held, _, _, path) = self
            .entries
            .iter()
            .filter(|(t, s, d, _)| {
                *d == digest
                    && t.len() > have as usize
                    && tokens.starts_with(t)
                    && sig.starts_with(s)
            })
            .max_by_key(|(t, ..)| t.len())?;
        Some(V41KvHit {
            path: path.clone(),
            offset: 0,
            bytes: std::fs::metadata(path).ok()?.len(),
            tokens: u32::try_from(held.len()).ok()?,
        })
    }

    fn store(
        &mut self,
        tokens: &[i32],
        sig: &[u64],
        digest: u64,
        _class: V41KvClass,
        save: &mut dyn FnMut(&Path) -> bool,
    ) {
        let path = scratch(&format!("v41-{}", self.entries.len()));
        if save(&path) {
            self.entries
                .push((tokens.to_vec(), sig.to_vec(), digest, path));
        }
    }
}

impl Drop for ScratchKv {
    fn drop(&mut self) {
        for (.., path) in &self.entries {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Checkpoint {
    Off,
    On,
}

/// One greedy V4.1 run without drafts: the rows it restored and its tokens.
fn v41_run(
    model: &Model,
    path: &Path,
    base: &V41RunOptions<'_>,
    tokens: &[i32],
    n: usize,
    checkpoint: Checkpoint,
    disk: Option<&mut ScratchKv>,
) -> Result<(i32, Vec<i32>)> {
    let mut opts = V41RunOptions {
        engram_dir: base.engram_dir,
        no_engram: base.no_engram,
        graph: base.graph,
        zchain: base.zchain,
        posttrain: base.posttrain,
        dspark: Some(0),
        kv_reuse: Some(checkpoint == Checkpoint::On),
        kv_disk: disk.map(|d| d as &mut dyn V41KvDisk),
        sampling: Some(V41Sampling::default()),
        ..Default::default()
    };
    let mut out = Vec::with_capacity(n);
    let mut emit = |token: i32| {
        out.push(token);
        true
    };
    let mut progress = |_: &str, _: i32, _: i32| true;
    let n = i32::try_from(n).unwrap_or(i32::MAX);
    model.v41_generate(path, tokens, n, &mut opts, &mut emit, &mut progress)?;
    Ok((v41_kvreuse_last_reused(), out))
}

/// V4.1 serves without a session or banks: its reuse is `v41_generate`'s
/// prompt-end checkpoint and the disk store under it, compared by greedy
/// stream against a run with the checkpoint off.
pub fn serving_check_v41(
    model: &Model,
    path: &Path,
    base: &V41RunOptions<'_>,
    prompt: &[i32],
) -> Result<Checklist> {
    if prompt.len() < 2 {
        return Err(fail("the checklist needs a prompt of at least two tokens"));
    }
    let mut list = Checklist {
        caps: caps_from_shape(model.shape()),
        ctx: 0,
        prompt: prompt.len(),
        split: split_point(prompt.len(), 0),
        grid: 0,
        rows: Vec::new(),
    };
    let k = DECODE_TOKENS;
    let run = |tokens: &[i32], n: usize, checkpoint: Checkpoint, disk: Option<&mut ScratchKv>| {
        v41_run(model, path, base, tokens, n, checkpoint, disk)
    };
    let (_, want_t) = run(prompt, k, Checkpoint::Off, None)?;
    let p = Prompts::new(prompt, list.split, &want_t);
    let (_, want_u) = run(&p.u, k, Checkpoint::Off, None)?;

    let on = Checkpoint::On;
    let mut store = ScratchKv::default();
    for item in Item::SESSION {
        if item.rewinds() && !p.diverges() {
            list.rows.push((
                item,
                Measured::Refused("the filler repeats the prompt".into()),
            ));
            continue;
        }
        let got = match item {
            Item::Prefix => run(&p.t[..p.m], 1, on, None).and_then(|_| run(&p.t, k, on, None)),
            Item::Rewind => run(&p.diverged, 1, on, None).and_then(|_| run(&p.t, k, on, None)),
            Item::Continue => run(&p.t, k, on, None).and_then(|_| run(&p.u, k, on, None)),
            // The middle run moves the in-memory checkpoint off T, so only
            // the store can give U its prefix.
            Item::SessionDisk => run(&p.t, 1, on, Some(&mut store))
                .and_then(|_| run(&p.diverged, 1, on, None))
                .and_then(|_| run(&p.u, k, on, Some(&mut store))),
            _ => Err(fail("V4.1 has no session snapshot")),
        };
        let want = match item {
            Item::Prefix | Item::Rewind => &want_t,
            _ => &want_u,
        };
        list.rows.push((item, streams(want, got)));
    }
    for item in Item::BANK {
        list.rows.push((
            item,
            Measured::Refused("V4.1 serves one sequence: no banks".into()),
        ));
    }
    Ok(list)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::shape_for_variant;
    use crate::Variant;

    /// Every variant; the match stops compiling when one is added.
    fn variants() -> Vec<Variant> {
        let all = vec![
            Variant::Flash,
            Variant::Pro,
            Variant::SolarOpen2_250B,
            Variant::Motif3,
            Variant::Kexaone236B,
            Variant::Dots3NotePrev,
            Variant::Qwen38FlashNext,
            Variant::Glm53Flash,
            Variant::K2Horizon375B,
            Variant::InklingSmall,
            Variant::Step37Flash,
            Variant::Ling30FlashVl,
            Variant::Mimo26Flash,
            Variant::Qwen35_27B,
            Variant::NaiveN05Flash,
            Variant::IQuestQ1,
            Variant::DeepSeek41Flash,
            Variant::Glm53FlashVq,
            Variant::Ling31FlashVq,
            Variant::BeamVq,
            Variant::MinimaxVq,
            Variant::Axk2Vq,
            Variant::Mimo26FlashVq,
        ];
        for v in &all {
            match v {
                Variant::Flash
                | Variant::Pro
                | Variant::SolarOpen2_250B
                | Variant::Motif3
                | Variant::Kexaone236B
                | Variant::Dots3NotePrev
                | Variant::Qwen38FlashNext
                | Variant::Glm53Flash
                | Variant::K2Horizon375B
                | Variant::InklingSmall
                | Variant::Step37Flash
                | Variant::Ling30FlashVl
                | Variant::Mimo26Flash
                | Variant::Qwen35_27B
                | Variant::NaiveN05Flash
                | Variant::IQuestQ1
                | Variant::DeepSeek41Flash
                | Variant::Glm53FlashVq
                | Variant::Ling31FlashVq
                | Variant::BeamVq
                | Variant::MinimaxVq
                | Variant::Axk2Vq
                | Variant::Mimo26FlashVq => {}
            }
        }
        all
    }

    fn declared(variant: Variant) -> Vec<&'static str> {
        let caps = caps_from_shape(shape_for_variant(variant));
        Item::SESSION
            .iter()
            .chain(&Item::BANK)
            .filter(|item| item.declared(&caps) != Support::None)
            .map(|item| item.label().1)
            .collect()
    }

    #[test]
    fn declarations_follow_the_caps() {
        // Every family states some reuse, so every family has items to run.
        for v in variants() {
            assert!(declared(v).contains(&"prefix"), "{v:?}");
        }
        // Bonsai keeps no banks, disk KV or snapshots; Inkling reuses exactly
        // (no rewind) without banks.
        assert_eq!(
            declared(Variant::Qwen35_27B),
            ["prefix", "rewind", "continue"]
        );
        assert_eq!(
            declared(Variant::InklingSmall),
            ["prefix", "continue", "session", "session"]
        );
    }

    #[test]
    fn verdict_beside_declaration() {
        let works = Measured::Works(String::new());
        assert_eq!(Verdict::of(Support::Present, &works), Verdict::Works);
        assert_eq!(Verdict::of(Support::None, &works), Verdict::Undeclared);
        assert_eq!(
            Verdict::of(Support::Qualified, &Measured::Cold),
            Verdict::Fail
        );
        assert_eq!(Verdict::of(Support::None, &Measured::Cold), Verdict::No);
        let wrong = Measured::Wrong(String::new());
        assert_eq!(Verdict::of(Support::None, &wrong), Verdict::Wrong);
        assert!(Verdict::Wrong.failed() && Verdict::Fail.failed());
        assert!(!Verdict::No.failed() && !Verdict::Undeclared.failed());
    }

    #[test]
    fn split_lands_on_the_grid() {
        assert_eq!(split_point(5000, 2048), 4096);
        assert_eq!(split_point(4097, 4096), 4096);
        assert_eq!(split_point(4096, 2048), 2048);
        assert_eq!(split_point(1000, 2048), 500);
        assert_eq!(checklist_ctx(7000), 8192);
    }

    #[test]
    fn rewind_filler_leaves_prompt() {
        let t: Vec<i32> = (0..40).collect();
        assert!(Prompts::new(&t, 20, &[]).diverges());
        // A prompt of one repeated token: T[..m] + X is a prefix of T.
        let flat = vec![7; 40];
        assert!(!Prompts::new(&flat, 20, &[]).diverges());
    }

    #[test]
    fn top1_decides_verdict() {
        let rows = |logits: Vec<Vec<f32>>| Rows {
            logits,
            tokens: Vec::new(),
        };
        let want = rows(vec![vec![0.0, 2.0, 1.0], vec![3.0, 0.0, 1.0]]);
        let only = [("prefill", &want)];
        let same = compare(&only, &rows(want.logits.clone()), 5);
        assert!(matches!(&same, Measured::Works(d) if d.ends_with("exact")));
        let noisy = compare(
            &only,
            &rows(vec![vec![0.0, 2.0, 1.001], vec![3.0, 0.0, 1.0]]),
            5,
        );
        assert!(matches!(&noisy, Measured::Works(d) if d.contains("max_abs")));
        let flip = rows(vec![vec![0.0, 1.0, 2.0], vec![3.0, 0.0, 1.0]]);
        let flipped = compare(&only, &flip, 5);
        assert!(matches!(&flipped, Measured::Wrong(d)
            if d.contains("top-1 1/2") && d.ends_with("row 0: 2 +1.000 over prefill 1 +1.000")));
        let short = compare(&only, &rows(vec![want.logits[0].clone()]), 5);
        assert!(matches!(short, Measured::Wrong(_)));

        // A second accepted arithmetic that picks the flipped top-1 explains
        // it; a top-1 no reference picks is a near tie within their own
        // spread (row 0's lead moved 1 -> -0.5: floor 1.5), a flip outside.
        let other = rows(vec![vec![0.0, 1.0, 1.5], vec![3.0, 0.0, 1.0]]);
        let both = [("prefill", &want), ("decode-fed", &other)];
        assert!(matches!(
            compare(&both, &flip, 5),
            Measured::Works(d) if d.contains("top-1 2/2 (prefill 1, decode-fed 2)")
        ));
        let stray = rows(vec![vec![2.0, 1.0, 0.0], vec![3.0, 0.0, 1.0]]);
        assert!(matches!(
            compare(&both, &stray, 5),
            Measured::Works(d) if d.contains("near-tie 1 <= 1.500")
        ));
        let wrong = rows(vec![vec![0.0, 2.0, 1.0], vec![0.0, 0.0, 9.0]]);
        assert!(matches!(compare(&both, &wrong, 5), Measured::Wrong(_)));
    }

    #[test]
    fn bank_streams_need_kept_rows() {
        let want = [1, 2, 3];
        assert_eq!(streams(&want, Ok((0, vec![1, 2, 3]))), Measured::Cold);
        assert!(matches!(
            streams(&want, Ok((4, vec![1, 2, 3]))),
            Measured::Works(_)
        ));
        assert!(matches!(
            streams(&want, Ok((4, vec![1, 9, 3]))),
            Measured::Wrong(_)
        ));
        assert!(matches!(
            streams(&want, Err(fail("no banks"))),
            Measured::Refused(_)
        ));
    }
}
