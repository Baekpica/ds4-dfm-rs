//! MiMo-V2.6-Flash VQ against the handoff's teacher: the original
//! checkpoint's held-out final states (tools/collect_mimo.py) over the 27
//! held-out text rows. Per assistant target (inside '<|im_start|>assistant\n'
//! .. '<|im_end|>', the handoff's measure_r8_mimo.py mask): KL(teacher ||
//! ours), top-1 agreement with the teacher and both next-token NLLs. The
//! 50 media rows need the September embeddings and are not scored here
//! (they hold 100 of the reference's 24,885 assistant targets).
//!
//! Teacher logits use the release's own head (Q8_0 lm_head over
//! rms(final_h) with the BF16 model.norm, both rounded to BF16 as the
//! reference does); the handoff used the BF16 source head (Q8_0 relative
//! MSE 3e-5). A chat row prefills each non-assistant span (scored at its
//! last row) and decodes assistant spans one token at a time.
//!
//! The reference is the handoff's evaluation of the same release bytes in
//! its own arithmetic (evidence/eval/release/{rl,unc}-rank-0-release.json:
//! assistant KL 0.088 / 0.089, top-1 92.2 / 91.9%). The original against
//! itself batched differently gives KL 0.020 and top-1 97.8%, so the engine's
//! own rounding (F16 or FP8 K/V, E4M3 prefill activations, gain after the
//! dot, F32 residuals) is allowed one such floor: KL at most KL_BAND above
//! the reference and top-1 at most TOP1_BAND below it.
//! Usage: mimo2_vq_quality MODEL HELDOUT TEACHER OUTPUT_JSON [CONTEXT] [ROWS]
//! (ROWS: comma-separated text row indices, default all).
//! mimo2_vq_quality --long MODEL TEXT PREFILL TAIL OUTPUT_JSON records the
//! teacher-forced next-token NLL and argmax of TAIL positions after a
//! PREFILL-token prompt of TEXT: long frontiers, which these rows do not
//! reach, compare as paired runs of two configurations.
use ds4_core::{Backend, GgufFile, Model, ModelFamily, TensorInventory, TokenBuffer};
use serde_json::{json, Value};
use std::io::Write;

const DEFAULT_CONTEXT: i32 = 8192;
const EMBED: usize = 4096;
const TOKENS: usize = 152_576;
const Q8_BLOCK: usize = 32;
const Q8_BYTES: usize = 34;
const EPS: f32 = 1e-6;
// Positions per batched teacher head (CPU).
const BATCH: usize = 64;
// Chat markers of the reference mask (measure_r8_mimo.py).
const IM_START: i32 = 151_644;
const IM_END: i32 = 151_645;
const ASSISTANT: i32 = 77_091;
const NL: i32 = 198;
// The release evaluation (unc-rank-0-release.json; rl: 0.0880 / 0.9221).
const REFERENCE_KL: f64 = 0.0893;
const REFERENCE_TOP1: f64 = 0.9187;
// One batching noise floor of the original (KL 0.020, top-1 97.8%).
const KL_BAND: f64 = 0.020;
const TOP1_BAND: f64 = 0.022;

type Tensors = std::collections::HashMap<String, (String, Vec<u64>, std::ops::Range<usize>)>;

/// safetensors header: name -> (dtype, shape, byte range).
fn safetensors(bytes: &[u8]) -> Result<Tensors, String> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| e.to_string())?;
    let mut out = Tensors::new();
    for (name, v) in header.as_object().ok_or("safetensors header")? {
        if name == "__metadata__" {
            continue;
        }
        let range = v["data_offsets"].as_array().ok_or("data_offsets")?;
        let (s, e) = (
            range[0].as_u64().unwrap() as usize,
            range[1].as_u64().unwrap() as usize,
        );
        let shape = v["shape"]
            .as_array()
            .ok_or("shape")?
            .iter()
            .map(|d| d.as_u64().unwrap())
            .collect();
        let dtype = v["dtype"].as_str().ok_or("dtype")?.to_string();
        out.insert(name.clone(), (dtype, shape, 8 + n + s..8 + n + e));
    }
    Ok(out)
}

fn ints(bytes: &[u8], t: &Tensors, name: &str) -> Result<Vec<i64>, String> {
    let (dtype, _, range) = t.get(name).ok_or(format!("missing {name}"))?;
    let raw = &bytes[range.clone()];
    Ok(match dtype.as_str() {
        "I32" => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
            .collect(),
        "I64" => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        _ => return Err(format!("{name}: integer tensor expected")),
    })
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn round_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e = 113u32;
            let mut m = m;
            while m & 1024 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 1023) << 13)
        }
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// The release's head: the body part's Q8_0 lm_head widened to F32 rows
/// and the BF16 final norm.
fn head(model_dir: &str) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(model_dir).join("body-nonexpert.gguf");
    let g = GgufFile::open(&path)?;
    let inv = TensorInventory::from_file(&path, &g)?;
    let bytes = g.as_bytes();
    let find = |name: &str| {
        inv.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or(format!("missing {name}"))
    };
    let lm = find("lm_head.weight")?;
    let norm = find("model.norm.weight")?;
    let raw = &bytes[lm.abs_offset as usize..(lm.abs_offset + lm.bytes) as usize];
    let mut out = vec![0f32; TOKENS * EMBED];
    std::thread::scope(|s| {
        for (rows, dst) in raw
            .chunks(EMBED / Q8_BLOCK * Q8_BYTES * 4096)
            .zip(out.chunks_mut(EMBED * 4096))
        {
            s.spawn(move || {
                for (block, w) in rows
                    .chunks_exact(Q8_BYTES)
                    .zip(dst.chunks_exact_mut(Q8_BLOCK))
                {
                    let d = f16(u16::from_le_bytes([block[0], block[1]]));
                    for (q, v) in block[2..].iter().zip(w) {
                        *v = d * f32::from(*q as i8);
                    }
                }
            });
        }
    });
    let n = &bytes[norm.abs_offset as usize..(norm.abs_offset + norm.bytes) as usize];
    let w = n
        .chunks_exact(2)
        .map(|c| bf16(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok((out, w))
}

/// rms(h) as the reference's BF16 RMSNorm: normalize in F32, round, scale by
/// the BF16 weight and round again.
fn normed(h: &[u16], w: &[f32]) -> Vec<f32> {
    let x: Vec<f32> = h.iter().map(|&b| bf16(b)).collect();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / EMBED as f32;
    let scale = 1.0 / (ms + EPS).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, w)| round_bf16(w * round_bf16(v * scale)))
        .collect()
}

/// F32 dot product with independent lanes the compiler vectorizes.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    for (x, y) in a.chunks_exact(16).zip(b.chunks_exact(16)) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum()
}

/// Teacher logits of `rows` normalized states (F32 dots over the head).
fn teacher(head: &[f32], rows: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0f32; TOKENS]; rows.len()];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let per = TOKENS.div_ceil(threads);
    let mut parts: Vec<Vec<Vec<f32>>> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let (lo, hi) = (t * per, ((t + 1) * per).min(TOKENS));
                    let mut part = vec![vec![0f32; hi.saturating_sub(lo)]; rows.len()];
                    for v in lo..hi {
                        let w = &head[v * EMBED..(v + 1) * EMBED];
                        for (r, x) in rows.iter().enumerate() {
                            part[r][v - lo] = dot(w, x);
                        }
                    }
                    part
                })
            })
            .collect();
        parts = handles.into_iter().map(|h| h.join().unwrap()).collect();
    });
    for (t, part) in parts.into_iter().enumerate() {
        for (r, p) in part.into_iter().enumerate() {
            out[r][t * per..t * per + p.len()].copy_from_slice(&p);
        }
    }
    out
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits[..TOKENS]
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let sum: f64 = logits[..TOKENS]
        .iter()
        .map(|&v| (v as f64 - max).exp())
        .sum();
    let lse = max + sum.ln();
    logits[..TOKENS].iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

#[derive(Default, Clone)]
struct Tally {
    positions: usize,
    kl: f64,
    top1: usize,
    nll: f64,
    teacher_nll: f64,
}

impl Tally {
    fn add(&mut self, t: &[f64], s: &[f64], target: usize) {
        self.positions += 1;
        self.kl += t.iter().zip(s).map(|(a, b)| a.exp() * (a - b)).sum::<f64>();
        self.top1 += usize::from(argmax(t) == argmax(s));
        self.nll -= s[target];
        self.teacher_nll -= t[target];
    }

    fn merge(&mut self, o: &Tally) {
        self.positions += o.positions;
        self.kl += o.kl;
        self.top1 += o.top1;
        self.nll += o.nll;
        self.teacher_nll += o.teacher_nll;
    }

    fn report(&self) -> Value {
        let n = self.positions.max(1) as f64;
        json!({"positions": self.positions, "kl": self.kl / n, "top1": self.top1 as f64 / n,
            "nll": self.nll / n, "teacher_nll": self.teacher_nll / n})
    }
}

/// Targets inside an assistant turn's content, closing <|im_end|> included.
fn assistant_mask(tokens: &[i32]) -> Vec<bool> {
    let mut mask = vec![false; tokens.len()];
    let mut inside = false;
    for i in 0..tokens.len() {
        if inside {
            mask[i] = true;
            if tokens[i] == IM_END {
                inside = false;
            }
        } else if i >= 2
            && tokens[i - 2] == IM_START
            && tokens[i - 1] == ASSISTANT
            && tokens[i] == NL
        {
            inside = true;
        }
    }
    mask
}

fn long(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let [model_dir, text, prefill, tail, output] = args else {
        return Err("usage: mimo2_vq_quality --long MODEL TEXT PREFILL TAIL OUTPUT_JSON".into());
    };
    let (prefill, tail): (usize, usize) = (prefill.parse()?, tail.parse()?);
    let model = Model::open(model_dir, Backend::Cuda, 8, true)?;
    let toks = model
        .tokenize_text(&std::fs::read_to_string(text)?)?
        .as_slice()
        .to_vec();
    if prefill < 1 || toks.len() <= prefill + tail {
        return Err(format!("{text} has {} tokens", toks.len()).into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let mut session = model.session((prefill + tail + 1) as i32)?;
    let started = std::time::Instant::now();
    session.sync(&TokenBuffer::from_tokens(toks[..prefill].to_vec()))?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    let (mut nll, mut top) = (Vec::with_capacity(tail), Vec::with_capacity(tail));
    for p in prefill - 1..prefill - 1 + tail {
        if p >= prefill {
            session.eval(toks[p])?;
        }
        let logits = session.copy_logits(vocab)?;
        if logits.len() < TOKENS || logits[..TOKENS].iter().any(|v| !v.is_finite()) {
            return Err(format!("position {p}: non-finite logits").into());
        }
        let s = log_softmax(&logits[..TOKENS]);
        nll.push(-s[toks[p + 1] as usize]);
        top.push(argmax(&s));
    }
    let mean = nll.iter().sum::<f64>() / tail as f64;
    let out = json!({"prefill": prefill, "tail": tail, "prefill_seconds": prefill_seconds,
        "mean_nll": mean, "nll": nll, "argmax": top});
    std::fs::write(output, serde_json::to_vec(&out)?)?;
    println!(
        "{}",
        json!({"prefill": prefill, "tail": tail, "prefill_seconds": prefill_seconds, "mean_nll": mean})
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--long") {
        return long(&args[1..]);
    }
    if !(4..=6).contains(&args.len()) {
        return Err(
            "usage: mimo2_vq_quality MODEL HELDOUT TEACHER OUTPUT_JSON [CONTEXT] [ROWS]".into(),
        );
    }
    let context = args
        .get(4)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(DEFAULT_CONTEXT);
    let held = std::fs::read(&args[1])?;
    let held_t = safetensors(&held)?;
    let teach = std::fs::read(&args[2])?;
    let teach_t = safetensors(&teach)?;
    let ids = ints(&held, &held_t, "input_ids")?;
    let offsets = ints(&held, &held_t, "offsets")?;
    let t_ids = ints(&teach, &teach_t, "input_ids")?;
    let t_offsets = ints(&teach, &teach_t, "offsets")?;
    if t_ids.len() < ids.len()
        || t_ids[..ids.len()] != ids[..]
        || t_offsets[..offsets.len()] != offsets[..]
    {
        return Err("teacher rows do not start with the held-out text rows".into());
    }
    let (dtype, shape, range) = teach_t.get("final_h").ok_or("missing final_h")?;
    if dtype != "BF16" || shape[1] as usize != EMBED {
        return Err("final_h must be BF16 [positions, 4096]".into());
    }
    let final_h: Vec<u16> = teach[range.clone()]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let rows: Vec<usize> = match args.get(5) {
        Some(list) => list
            .split(',')
            .map(|v| v.parse())
            .collect::<Result<_, _>>()?,
        None => (0..offsets.len() - 1).collect(),
    };
    let (head, norm_w) = head(&args[0])?;
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    if model.family() != ModelFamily::Mimo2 {
        return Err("this receipt requires the MiMo-V2.6-Flash VQ release".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    let mut all = Tally::default();
    let mut prefill = Tally::default();
    let mut rows_out = Vec::new();
    for &row in &rows {
        let (st, en) = (offsets[row] as usize, offsets[row + 1] as usize);
        let toks: Vec<i32> = ids[st..en].iter().map(|&t| t as i32).collect();
        let mask = assistant_mask(&toks);
        // Positions p whose target p + 1 is an assistant token.
        let wanted: Vec<usize> = (0..toks.len() - 1).filter(|&p| mask[p + 1]).collect();
        if wanted.is_empty() {
            continue;
        }
        if toks.len() >= context as usize {
            return Err(format!("row {row} exceeds context {context}").into());
        }
        let mut session = model.session(context)?;
        let (mut tally, mut first) = (Tally::default(), Tally::default());
        let started = std::time::Instant::now();
        let mut pending: Vec<(usize, Vec<f64>, bool)> = Vec::new();
        let mut done = 0usize;
        let score = |pending: &mut Vec<(usize, Vec<f64>, bool)>,
                     tally: &mut Tally,
                     first: &mut Tally| {
            let states: Vec<Vec<f32>> = pending
                .iter()
                .map(|(p, _, _)| normed(&final_h[(st + p) * EMBED..(st + p + 1) * EMBED], &norm_w))
                .collect();
            for ((p, s, prefilled), logits) in pending.drain(..).zip(teacher(&head, &states)) {
                let t = log_softmax(&logits);
                tally.add(&t, &s, toks[p + 1] as usize);
                if prefilled {
                    first.add(&t, &s, toks[p + 1] as usize);
                }
            }
        };
        for &p in &wanted {
            // Prefill a span (its last row scored), or decode one token.
            let prefilled = p + 1 - done > 1;
            if prefilled {
                session.sync(&TokenBuffer::from_tokens(toks[..=p].to_vec()))?;
            } else {
                session.eval(toks[p])?;
            }
            done = p + 1;
            let logits = session.copy_logits(vocab)?;
            if logits.len() < TOKENS || logits[..TOKENS].iter().any(|v| !v.is_finite()) {
                return Err(format!("row {row} position {p}: non-finite logits").into());
            }
            pending.push((p, log_softmax(&logits), prefilled));
            if pending.len() == BATCH {
                score(&mut pending, &mut tally, &mut first);
            }
        }
        score(&mut pending, &mut tally, &mut first);
        drop(session);
        let report = json!({"row": row, "tokens": toks.len(), "seconds": started.elapsed().as_secs_f64(),
            "assistant": tally.report(), "prefill_rows": first.report()});
        println!("{report}");
        std::io::stdout().flush()?;
        all.merge(&tally);
        prefill.merge(&first);
        rows_out.push(report);
    }
    let kl = all.kl / all.positions.max(1) as f64;
    let top1 = all.top1 as f64 / all.positions.max(1) as f64;
    let pass = kl <= REFERENCE_KL + KL_BAND && top1 >= REFERENCE_TOP1 - TOP1_BAND;
    let out = json!({"context": context, "reference": {"kl": REFERENCE_KL, "top1": REFERENCE_TOP1},
        "kl_band": KL_BAND, "top1_band": TOP1_BAND, "pass": pass, "assistant": all.report(),
        "prefill_rows": prefill.report(), "rows": rows_out});
    std::fs::write(&args[3], serde_json::to_vec_pretty(&out)?)?;
    println!(
        "{}",
        json!({"assistant": out["assistant"], "prefill_rows": out["prefill_rows"], "pass": pass})
    );
    if !pass {
        return Err(
            "the engine is outside the release evaluation's assistant KL or top-1 band".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_mask_covers_content_and_close() {
        let toks = [IM_START, ASSISTANT, NL, 5, 6, IM_END, 7, IM_START, 8, NL, 9];
        let mask = assistant_mask(&toks);
        assert_eq!(
            mask,
            [false, false, false, true, true, true, false, false, false, false, false]
        );
    }

    #[test]
    fn bf16_rounding_is_nearest_even() {
        assert_eq!(round_bf16(1.0), 1.0);
        assert_eq!(round_bf16(f32::from_bits(0x3f80_8000)), 1.0);
        assert_eq!(
            round_bf16(f32::from_bits(0x3f81_8000)),
            f32::from_bits(0x3f82_0000)
        );
        assert_eq!(f16(0x3c00), 1.0);
        assert_eq!(f16(0x0001), 2f32.powi(-24));
    }
}
