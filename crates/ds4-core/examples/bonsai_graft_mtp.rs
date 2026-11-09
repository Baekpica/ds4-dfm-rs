//! bonsai_graft_mtp.rs — graft the Qwen3.8-27B MTP block into a Bonsai GGUF.
//!
//! Reproduces `decent-jawfish/bonsai-2-27b-mtp` byte for byte.  The published
//! `graft_mtp.py` in that repository is the reference; this tool follows it
//! step for step:
//!
//!   1. the 15 `blk.64.*` tensors are copied verbatim and unrotated from
//!      `unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-UD-Q2_K_XL.gguf` (they are
//!      stock Qwen weights and are deliberately absent from
//!      `prism.hadamard.weight_names`);
//!   2. `qwen35.block_count` becomes 65 and `qwen35.nextn_predict_layers` is
//!      inserted as 1 right after it;
//!   3. every base tensor keeps its bytes *and* its file order, then the MTP
//!      tensors follow in source-offset order;
//!   4. each tensor's byte count is the gap to the next offset in its own
//!      file, so the source's inter-tensor padding is carried over too.
//!
//! Usage:
//!   cargo run -p ds4-core --release --example bonsai_graft_mtp -- \
//!       plan <src.gguf> [--script URL]
//!   cargo run -p ds4-core --release --example bonsai_graft_mtp -- \
//!       graft <base.gguf> <src.gguf> <out.gguf>
//!
//! `plan` prints the header geometry and the byte ranges the graft needs, and
//! with `--script URL` prints a shell script that range-fetches them into a
//! sparse copy of the source.  `graft` writes the merged file; the recipe in
//! docs/BONSAI.md then compares its sha256 with the published digest.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747;
const MTP_PREFIX: &str = "blk.64.";
const MTP_EXPECTED: usize = 15;
const NEXTN_KEY: &str = "qwen35.nextn_predict_layers";
const BLOCK_KEY: &str = "qwen35.block_count";
const GGUF_U32: u32 = 4;
const COPY_CHUNK: usize = 8 << 20;

#[derive(Clone)]
struct Kv {
    key: String,
    typ: u32,
    raw: Vec<u8>,
}

#[derive(Clone)]
struct Tensor {
    name: String,
    dims: Vec<u64>,
    typ: u32,
    offset: u64,
}

struct Gguf {
    version: u32,
    kv: Vec<Kv>,
    tensors: Vec<Tensor>,
    data_start: u64,
    file_len: u64,
}

fn align_up(v: u64, a: u64) -> u64 {
    if a == 0 {
        return v;
    }
    v.div_ceil(a) * a
}

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(f: &mut File) -> io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(f: &mut File) -> io::Result<String> {
    let n = read_u64(f)?;
    let mut b = vec![0u8; usize::try_from(n).map_err(|_| io::Error::other("string too long"))?];
    f.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|_| io::Error::other("string is not utf-8"))
}

/// Fixed width of one scalar GGUF value type, or None for the variable ones.
fn scalar_width(t: u32) -> Option<usize> {
    match t {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

/// Consumes one value and returns its raw bytes, so a re-emit is byte-exact.
fn read_value(f: &mut File, t: u32) -> io::Result<Vec<u8>> {
    if let Some(w) = scalar_width(t) {
        let mut b = vec![0u8; w];
        f.read_exact(&mut b)?;
        return Ok(b);
    }
    match t {
        8 => {
            let n = read_u64(f)?;
            let mut b = vec![0u8; 8];
            b.copy_from_slice(&n.to_le_bytes());
            let mut s = vec![0u8; usize::try_from(n).map_err(|_| io::Error::other("string too long"))?];
            f.read_exact(&mut s)?;
            b.extend_from_slice(&s);
            Ok(b)
        }
        9 => {
            let et = read_u32(f)?;
            let n = read_u64(f)?;
            let mut out = Vec::new();
            out.extend_from_slice(&et.to_le_bytes());
            out.extend_from_slice(&n.to_le_bytes());
            for _ in 0..n {
                out.extend_from_slice(&read_value(f, et)?);
            }
            Ok(out)
        }
        _ => Err(io::Error::other(format!("unknown gguf value type {t}"))),
    }
}

fn parse(path: &Path) -> io::Result<Gguf> {
    let file_len = std::fs::metadata(path)?.len();
    let mut f = File::open(path)?;
    if read_u32(&mut f)? != GGUF_MAGIC {
        return Err(io::Error::other("not a GGUF"));
    }
    let version = read_u32(&mut f)?;
    let n_tensors = read_u64(&mut f)?;
    let n_kv = read_u64(&mut f)?;

    let mut kv = Vec::with_capacity(usize::try_from(n_kv).unwrap_or(0));
    for _ in 0..n_kv {
        let key = read_string(&mut f)?;
        let typ = read_u32(&mut f)?;
        let raw = read_value(&mut f, typ)?;
        kv.push(Kv { key, typ, raw });
    }

    let mut tensors = Vec::with_capacity(usize::try_from(n_tensors).unwrap_or(0));
    for _ in 0..n_tensors {
        let name = read_string(&mut f)?;
        let nd = read_u32(&mut f)?;
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(read_u64(&mut f)?);
        }
        let typ = read_u32(&mut f)?;
        let offset = read_u64(&mut f)?;
        tensors.push(Tensor { name, dims, typ, offset });
    }

    let header_end = f.stream_position()?;
    let alignment = kv
        .iter()
        .find(|e| e.key == "general.alignment")
        .and_then(|e| {
            let b = e.raw.get(..4)?;
            Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64)
        })
        .unwrap_or(32);
    Ok(Gguf {
        version,
        kv,
        tensors,
        data_start: align_up(header_end, alignment),
        file_len,
    })
}

/// Tensor byte count as the gap to the next offset: this is how the reference
/// script carries each file's inter-tensor padding into the merged file.
fn sizes_by_gap(g: &Gguf) -> Vec<(usize, u64)> {
    let mut order: Vec<usize> = (0..g.tensors.len()).collect();
    order.sort_by_key(|&i| g.tensors[i].offset);
    let mut out = Vec::with_capacity(order.len());
    for (j, &i) in order.iter().enumerate() {
        let start = g.tensors[i].offset;
        let end = if j + 1 < order.len() {
            g.tensors[order[j + 1]].offset
        } else {
            g.file_len - g.data_start
        };
        out.push((i, end - start));
    }
    out
}

fn kv_u32(kv: &[Kv], key: &str) -> Option<u32> {
    let e = kv.iter().find(|e| e.key == key)?;
    let b = e.raw.get(..4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn mtp_tensors(g: &Gguf) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..g.tensors.len())
        .filter(|&i| g.tensors[i].name.starts_with(MTP_PREFIX))
        .collect();
    idx.sort_by_key(|&i| g.tensors[i].offset);
    idx
}

fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn write_kv<W: Write>(w: &mut W, e: &Kv) -> io::Result<()> {
    write_string(w, &e.key)?;
    w.write_all(&e.typ.to_le_bytes())?;
    w.write_all(&e.raw)
}

fn write_tensor<W: Write>(w: &mut W, t: &Tensor) -> io::Result<()> {
    write_string(w, &t.name)?;
    w.write_all(&(t.dims.len() as u32).to_le_bytes())?;
    for d in &t.dims {
        w.write_all(&d.to_le_bytes())?;
    }
    w.write_all(&t.typ.to_le_bytes())?;
    w.write_all(&t.offset.to_le_bytes())
}

fn copy_range<W: Write>(out: &mut W, src: &Path, from: u64, bytes: u64) -> io::Result<()> {
    let mut f = File::open(src)?;
    f.seek(SeekFrom::Start(from))?;
    let mut left = bytes;
    let mut buf = vec![0u8; COPY_CHUNK];
    while left > 0 {
        let want = usize::try_from(left.min(COPY_CHUNK as u64)).unwrap();
        f.read_exact(&mut buf[..want])?;
        out.write_all(&buf[..want])?;
        left -= want as u64;
    }
    Ok(())
}

fn plan(src: &Path, script_url: Option<&str>, size: Option<u64>) -> io::Result<()> {
    let mut s = parse(src)?;
    /* A header-only fetch is shorter than the artifact: take the real size so
     * the last tensor's gap is computed against it, exactly as the reference
     * does with a HEAD request. */
    if let Some(n) = size {
        s.file_len = n;
    }
    let mtp = mtp_tensors(&s);
    if mtp.len() != MTP_EXPECTED {
        return Err(io::Error::other(format!(
            "expected {MTP_EXPECTED} {MTP_PREFIX}* tensors, found {}",
            mtp.len()
        )));
    }
    let sizes = sizes_by_gap(&s);
    let mut total = 0u64;
    let geometry = format!("source: data_start={} file_len={} tensors={}",
                           s.data_start, s.file_len, s.tensors.len());
    if script_url.is_some() {
        /* The script goes to stdout alone, so `plan --script > fetch.sh` is
         * runnable; the geometry belongs on stderr. */
        eprintln!("{geometry}");
    } else {
        println!("{geometry}");
    }
    if let Some(url) = script_url {
        println!("set -e");
        println!("# header (metadata + tensor table)");
        println!("curl -fsSL -r 0-{} '{url}' | dd of=\"$SRC\" bs=4M oflag=seek_bytes seek=0 conv=notrunc status=none", s.data_start - 1);
        for &i in &mtp {
            let t = &s.tensors[i];
            let n = sizes.iter().find(|e| e.0 == i).map(|e| e.1).unwrap_or(0);
            total += n;
            let lo = s.data_start + t.offset;
            println!("# {} {} B", t.name, n);
            println!("curl -fsSL -r {}-{} '{url}' | dd of=\"$SRC\" bs=4M oflag=seek_bytes seek={} conv=notrunc status=none", lo, lo + n - 1, lo);
        }
    } else {
        for &i in &mtp {
            let t = &s.tensors[i];
            let n = sizes.iter().find(|e| e.0 == i).map(|e| e.1).unwrap_or(0);
            total += n;
            println!("{:<40} data_off={:<12} bytes={:<12} range={}-{}", t.name, t.offset, n,
                     s.data_start + t.offset, s.data_start + t.offset + n - 1);
        }
    }
    println!("# mtp bytes total: {total}");
    Ok(())
}

fn graft(base: &Path, src: &Path, out: &Path) -> io::Result<()> {
    let b = parse(base)?;
    let s = parse(src)?;
    let mtp = mtp_tensors(&s);
    if mtp.len() != MTP_EXPECTED {
        return Err(io::Error::other(format!(
            "expected {MTP_EXPECTED} {MTP_PREFIX}* tensors, found {}",
            mtp.len()
        )));
    }
    if b.tensors.iter().any(|t| t.name.starts_with(MTP_PREFIX)) {
        return Err(io::Error::other("base already carries blk.64.* tensors"));
    }

    /* ---- metadata: block_count + 1, nextn_predict_layers = 1 ---- */
    let old_blocks = kv_u32(&b.kv, BLOCK_KEY)
        .ok_or_else(|| io::Error::other(format!("{BLOCK_KEY} missing or not u32")))?;
    let mut kv: Vec<Kv> = Vec::with_capacity(b.kv.len() + 1);
    for e in &b.kv {
        if e.key == BLOCK_KEY {
            let mut e = e.clone();
            e.raw[..4].copy_from_slice(&(old_blocks + 1).to_le_bytes());
            kv.push(e);
            if !b.kv.iter().any(|x| x.key == NEXTN_KEY) {
                kv.push(Kv { key: NEXTN_KEY.to_string(), typ: GGUF_U32, raw: 1u32.to_le_bytes().to_vec() });
            }
            continue;
        }
        if e.key == NEXTN_KEY {
            let mut e = e.clone();
            e.raw[..4].copy_from_slice(&1u32.to_le_bytes());
            kv.push(e);
            continue;
        }
        kv.push(e.clone());
    }

    /* ---- tensor table: base order, then MTP in source-offset order ---- */
    let bsz = sizes_by_gap(&b);
    let ssz = sizes_by_gap(&s);
    let mut tensors: Vec<Tensor> = b.tensors.clone();
    for &i in &mtp {
        tensors.push(s.tensors[i].clone());
    }

    let mut run = 0u64;
    let mut layout: Vec<(u64, u64)> = Vec::with_capacity(tensors.len()); /* (offset, bytes) */
    for (i, t) in tensors.iter_mut().enumerate() {
        let n = if i < b.tensors.len() {
            bsz.iter().find(|e| e.0 == i).map(|e| e.1).unwrap_or(0)
        } else {
            let si = mtp[i - b.tensors.len()];
            ssz.iter().find(|e| e.0 == si).map(|e| e.1).unwrap_or(0)
        };
        run = align_up(run, 32);
        t.offset = run;
        layout.push((run, n));
        run += n;
    }

    /* ---- write ---- */
    let mut w = BufWriter::new(OpenOptions::new().create(true).write(true).truncate(true).open(out)?);
    w.write_all(&GGUF_MAGIC.to_le_bytes())?;
    w.write_all(&b.version.to_le_bytes())?;
    w.write_all(&(tensors.len() as u64).to_le_bytes())?;
    w.write_all(&(kv.len() as u64).to_le_bytes())?;
    for e in &kv {
        write_kv(&mut w, e)?;
    }
    for t in &tensors {
        write_tensor(&mut w, t)?;
    }
    let header_end = w.stream_position()?;
    let data_start = align_up(header_end, 32);
    w.write_all(&vec![0u8; usize::try_from(data_start - header_end).unwrap()])?;

    for (i, &(off, n)) in layout.iter().enumerate() {
        let want = data_start + off;
        let cur = w.stream_position()?;
        if cur < want {
            w.write_all(&vec![0u8; usize::try_from(want - cur).unwrap()])?;
        }
        if i < b.tensors.len() {
            copy_range(&mut w, base, b.data_start + b.tensors[i].offset, n)?;
        } else {
            let si = mtp[i - b.tensors.len()];
            copy_range(&mut w, src, s.data_start + s.tensors[si].offset, n)?;
        }
        if i % 200 == 0 {
            eprintln!("  .. {i}/{}", layout.len());
        }
    }
    w.flush()?;
    drop(w);

    let size = std::fs::metadata(out)?.len();
    println!("wrote {out:?}: tensors={} kv={} data_start={data_start} size={size}",
             tensors.len(), kv.len());
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("plan") if args.len() >= 2 => {
            let url = args.iter().position(|a| a == "--script").map(|i| args[i + 1].as_str());
            let size = args.iter().position(|a| a == "--size")
                .and_then(|i| args[i + 1].parse::<u64>().ok());
            plan(Path::new(&args[1]), url, size)
        }
        Some("graft") if args.len() == 4 => {
            graft(Path::new(&args[1]), Path::new(&args[2]), Path::new(&args[3]))
        }
        _ => {
            eprintln!("usage: bonsai_graft_mtp plan <src.gguf> [--size N] [--script URL]");
            eprintln!("       bonsai_graft_mtp graft <base.gguf> <src.gguf> <out.gguf>");
            std::process::exit(2);
        }
    }
}
