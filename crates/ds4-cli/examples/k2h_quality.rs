//! K2-Horizon hVQ teacher-forced decode logits for the handoff's held-out
//! scorer (tools/vq_calib/k2h_heldout.py score --logits DIR): each sequence
//! of IDS_JSON (512-token chunks without BOS) prefills its first token, then
//! decodes the rest one by one; row p of OUT/seq-NNNN.npy ([512, 250624]
//! float16) holds the logits after tokens 0..p. SEQS limits the sequences
//! (`0-7`, default all).
//!
//! With DS4_K2H_TRACE=DIR set, each sequence instead prefills all its tokens
//! in one forward and OUT/final-hidden.safetensors collects the traced final
//! residual stream (layer 60's `hidden`, BF16 values) of every sequence with
//! the input ids, for `k2h_heldout.py score --hidden` (prefill arithmetic).
//! Usage: k2h_quality MODEL IDS_JSON OUT_DIR [SEQS]
use ds4_core::{Backend, Model, ModelFamily, TokenBuffer};
use std::io::Write;

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
fn write_npy(path: &std::path::Path, rows: &[Vec<f32>], cols: usize) -> std::io::Result<()> {
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

// safetensors of the final residual streams (BF16) and the input ids (I32).
fn write_hidden(
    path: &std::path::Path,
    hidden: &[u16],
    ids: &[Vec<i32>],
    tokens: usize,
) -> std::io::Result<()> {
    let (n, h) = (ids.len(), hidden.len() * 2);
    let mut header = format!(
        "{{\"final_hidden\":{{\"dtype\":\"BF16\",\"shape\":[{n},{tokens},6144],\"data_offsets\":[0,{h}]}},\
         \"input_ids\":{{\"dtype\":\"I32\",\"shape\":[{n},{tokens}],\"data_offsets\":[{h},{}]}}}}",
        h + n * tokens * 4
    );
    header.push_str(&" ".repeat((8 - header.len() % 8) % 8));
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(header.as_bytes())?;
    out.write_all(
        &hidden
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    out.write_all(
        &ids.iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    out.flush()
}

// Prefill each sequence in one forward; collect layer 60's traced rows.
fn prefill_hidden(
    model: &Model,
    ids: &[Vec<i32>],
    out: &std::path::Path,
    trace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = ids[0].len();
    let mut hidden: Vec<u16> = Vec::with_capacity(ids.len() * tokens * 6144);
    for (i, seq) in ids.iter().enumerate() {
        let mut session = model.session(CONTEXT)?;
        let started = std::time::Instant::now();
        session.sync(&TokenBuffer::from_tokens(seq.clone()))?;
        let prefill_s = started.elapsed().as_secs_f64();
        let rows = std::fs::read(std::path::Path::new(trace).join("layer_60_hidden.bin"))?;
        if rows.len() != tokens * 6144 * 4 {
            return Err(format!("sequence {i}: traced rows are {} bytes", rows.len()).into());
        }
        // Rows hold BF16-rounded F32 values: the high half is the BF16.
        hidden.extend(
            rows.as_chunks::<4>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[2], b[3]])),
        );
        println!("{{\"sequence\": {i}, \"prefill_s\": {prefill_s:.3}}}");
        std::io::stdout().flush()?;
    }
    Ok(write_hidden(
        &out.join("final-hidden.safetensors"),
        &hidden,
        ids,
        tokens,
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: k2h_quality MODEL IDS_JSON OUT_DIR [SEQS]".into());
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
    if let Ok(trace) = std::env::var("DS4_K2H_TRACE") {
        return prefill_hidden(&model, &ids, &out, &trace);
    }
    let vocab = model.vocab().n_vocab() as usize;
    for (i, seq) in ids.iter().enumerate().take(hi.min(ids.len())).skip(lo) {
        let first = 1.min(seq.len());
        let mut session = model.session(CONTEXT)?;
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
