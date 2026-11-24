//! K2-Horizon hVQ teacher-forced logits for the conversion's held-out scorer
//! (tools/vq_calib/k2h_heldout.py score): each sequence of IDS_JSON (512-token
//! chunks without BOS) prefills its first token, then decodes the rest one by
//! one; row p of OUT/seq-NNNN.npy ([512, 250624] float16) holds the logits
//! after tokens 0..p (`score --logits OUT`). SEQS limits the sequences (`0-7`,
//! default all).
//!
//! With `--prefill STEP` each sequence instead grows one session STEP tokens
//! at a time (the prefill kernels from five rows up), and
//! OUT/seq-NNNN/frontier_FFFFFF.logits.json holds the logits after each
//! F = STEP, 2 STEP, ... tokens (`score --frontiers OUT`).
//! Usage: k2h_quality MODEL IDS_JSON OUT_DIR [SEQS] [--prefill STEP]
use ds4_core::{Backend, Model, ModelFamily, Session, TokenBuffer};
use std::io::Write;
use std::path::Path;

const CONTEXT: i32 = 4096;

// IEEE half of an F32 (round to nearest even), for numpy's '<f2'.
fn half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half_m = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rem > mid || (rem == mid && half_m & 1 == 1);
        return sign | (half_m + u32::from(up)) as u16;
    }
    let half_m = man >> 13;
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && half_m & 1 == 1);
    let v = ((e as u32) << 10 | half_m) + u32::from(up);
    sign | v as u16
}

// One .npy file: '<f2' rows x cols, C order.
fn write_npy(path: &Path, rows: &[Vec<f32>], cols: usize) -> std::io::Result<()> {
    let mut header = format!(
        "{{'descr': '<f2', 'fortran_order': False, 'shape': ({}, {}), }}",
        rows.len(),
        cols
    );
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(b"\x93NUMPY\x01\x00")?;
    out.write_all(&(header.len() as u16).to_le_bytes())?;
    out.write_all(header.as_bytes())?;
    for row in rows {
        let bytes: Vec<u8> = row.iter().flat_map(|&v| half(v).to_le_bytes()).collect();
        out.write_all(&bytes)?;
    }
    out.flush()
}

// The logits after each `step`-token prefill chunk of one session, in
// ds4-bench's frontier JSON.
fn prefill_frontiers(
    session: &mut Session<'_>,
    seq: &[i32],
    step: usize,
    vocab: usize,
    dir: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let mut frontiers = 0;
    for end in (step..=seq.len()).step_by(step) {
        session.sync(&TokenBuffer::from_tokens(seq[..end].to_vec()))?;
        let logits = session.copy_logits(vocab)?;
        let body = serde_json::json!({"vocab": vocab, "frontier_tokens": end, "logits": logits});
        std::fs::write(
            dir.join(format!("frontier_{end:06}.logits.json")),
            serde_json::to_vec(&body)?,
        )?;
        frontiers += 1;
    }
    Ok(frontiers)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let step = match args.iter().position(|a| a == "--prefill") {
        Some(at) if at + 1 < args.len() => {
            let step = args.remove(at + 1).parse::<usize>()?;
            args.remove(at);
            Some(step.max(1))
        }
        Some(_) => return Err("--prefill needs STEP".into()),
        None => None,
    };
    if !(3..=4).contains(&args.len()) {
        return Err("usage: k2h_quality MODEL IDS_JSON OUT_DIR [SEQS] [--prefill STEP]".into());
    }
    let ids: Vec<Vec<i32>> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let out = std::path::PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out)?;
    let (lo, hi) = match args.get(3) {
        Some(range) => {
            let (a, b) = range.split_once('-').ok_or("SEQS is FIRST-LAST")?;
            (a.parse::<usize>()?, b.parse::<usize>()? + 1)
        }
        None => (0, ids.len()),
    };
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    if model.family() != ModelFamily::K2hVq {
        return Err("this receipt requires the K2-Horizon hVQ release".into());
    }
    let vocab = model.vocab().n_vocab() as usize;
    for (i, seq) in ids.iter().enumerate().take(hi.min(ids.len())).skip(lo) {
        let mut session = model.session(CONTEXT)?;
        if let Some(step) = step {
            let started = std::time::Instant::now();
            let dir = out.join(format!("seq-{i:04}"));
            let frontiers = prefill_frontiers(&mut session, seq, step, vocab, &dir)?;
            let prefill_s = started.elapsed().as_secs_f64();
            println!(
                "{{\"sequence\": {i}, \"frontiers\": {frontiers}, \"prefill_s\": {prefill_s:.3}}}"
            );
            std::io::stdout().flush()?;
            continue;
        }
        let first = 1.min(seq.len());
        let started = std::time::Instant::now();
        session.sync(&TokenBuffer::from_tokens(seq[..first].to_vec()))?;
        let prefill_s = started.elapsed().as_secs_f64();
        let mut rows = vec![session.copy_logits(vocab)?];
        let started = std::time::Instant::now();
        for &token in &seq[first..] {
            session.eval(token)?;
            rows.push(session.copy_logits(vocab)?);
        }
        let decode_s = started.elapsed().as_secs_f64();
        drop(session);
        write_npy(&out.join(format!("seq-{i:04}.npy")), &rows, vocab)?;
        println!("{{\"sequence\": {i}, \"rows\": {}, \"prefill_s\": {prefill_s:.3}, \"decode_s\": {decode_s:.3}}}", rows.len());
        std::io::stdout().flush()?;
    }
    Ok(())
}
