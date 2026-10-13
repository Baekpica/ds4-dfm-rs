//! ds41_fp8_ref.rs — the P4-3 fp8_32x32 emulation oracle for tests/test_ds41_fp8.cu.
//!
//!   cargo run -p ds4-core --release --example ds41_fp8_ref -- <fp8.img> <fp8.cases.txt> <fp8.ref.f32>
//!
//! Reads the fixture image (tests/fixtures/ds41/fp8/gen_fp8.py) and the case
//! list, decodes the e4m3 plane with the tile scales, and writes one f32 row
//! of output values per token per case.
//!
//! The arithmetic mirrors cuda/ds41_fp8blk.cuh, not a plausible reading of it:
//! each weight element decodes as (e4m3 * scale) in f32 (one exact product —
//! the scale is a power of two), activations pass through the same RNE
//! f32->bf16->f32 (v41_bf16r) when n > 1 (the XB=1 arm) or through
//! __float2bfloat16 when the >8 arm expands the weight, and the dot
//! accumulates in f64 — an order-free sum, so the compare measures the
//! device's f32 accumulation (and the bf16-w expansion) alone.
//!
//! The onehot cases are built so every dot is a single-term sum: the entry
//! must be bit-exact. Dense cases compare within the recorded tolerance; the
//! round_out cases additionally allow one bf16 quantum, because rounding the
//! output can flip a value that sits on a bf16 boundary.

use std::io::Write;

/// RNE round f32 -> bf16 -> f32, the device's v41_bf16r bit for bit
/// (ds41_primitives.cuh:53-57; NaN/Inf pass through).
fn bf16r(x: f32) -> f32 {
    let mut u = x.to_bits();
    if (u & 0x7F80_0000) == 0x7F80_0000 {
        return x;
    }
    u = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    u &= 0xFFFF_0000;
    f32::from_bits(u)
}

/// __float2bfloat16 (RNE) for finite values — the device's conversion used by
/// v41_fp8blk_to_bf16_kernel and v41_x_to_bf16_kernel. bf16r returns the same
/// value for finite inputs, so one function serves both.
fn bf16_dev(x: f32) -> f32 {
    bf16r(x)
}

/// E4M3FN bit decode, mirroring ds4_e4m3fn_to_f32 (cuda/ds41_fp8blk.cuh; the
/// engine's ds4_fp8.h:33-42): abs==0x7f is NaN, exp==0 is man*2^-9.
fn e4m3_to_f32(x: u8) -> f32 {
    let abs = x & 0x7f;
    let sign = x & 0x80 != 0;
    if abs == 0 {
        return if sign { -0.0 } else { 0.0 };
    }
    if abs == 0x7f {
        return f32::NAN;
    }
    let exp = (x >> 3) & 0x0f;
    let man = x & 0x07;
    let v = if exp == 0 {
        (man as f32) * 2f32.powi(-9)
    } else {
        (1.0 + man as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign {
        -v
    } else {
        v
    }
}

/// E8M0 scale byte -> f32, mirroring ds4_e8m0_to_f32 (ds4_fp8.h:46-50).
fn e8m0_to_f32(e: u8) -> f32 {
    let bits: u32 = if e == 0 {
        0x0040_0000
    } else {
        (e as u32) << 23
    };
    f32::from_bits(bits)
}

struct Tensor {
    off: usize,
    rows: usize,
    cols: usize,
    groups: usize,
    gdim: usize,
    rank: usize,
}

impl Tensor {
    /// Decoded weight at (row, col), f32, exact.
    fn w(&self, img: &[u8], r: usize, c: usize) -> f32 {
        let sbc = self.cols / 32;
        let b = img[self.off + r * self.cols + c];
        let s = img[self.off + self.rows * self.cols + (r / 32) * sbc + c / 32];
        e4m3_to_f32(b) * e8m0_to_f32(s)
    }
}

struct Case {
    kind: String,
    entry: String,
    n: usize,
    round_out: bool,
    x: Vec<f32>,
}

fn parse(path: &str) -> (Vec<(String, Tensor)>, Vec<Case>) {
    let text = std::fs::read_to_string(path).expect("cases file");
    let mut tensors: Vec<(String, Tensor)> = Vec::new();
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        match it.next().unwrap() {
            "tensor" => {
                let name = it.next().unwrap().to_string();
                let mut t = Tensor {
                    off: 0,
                    rows: 0,
                    cols: 0,
                    groups: 1,
                    gdim: 0,
                    rank: 0,
                };
                for kv in it {
                    let (k, v) = kv.split_once('=').unwrap();
                    let v: usize = v.parse().unwrap();
                    match k {
                        "off" => t.off = v,
                        "in" => t.cols = v,
                        "out" => t.rows = v,
                        "groups" => t.groups = v,
                        "gdim" => t.gdim = v,
                        "rank" => t.rank = v,
                        other => panic!("unknown tensor key {other}"),
                    }
                }
                if t.groups > 1 {
                    t.rows = t.groups * t.rank;
                    t.cols = t.gdim;
                }
                tensors.push((name, t));
            }
            "case" => {
                it.next(); // index
                let kind = it.next().unwrap().to_string();
                let mut c = Case {
                    kind,
                    entry: String::new(),
                    n: 0,
                    round_out: false,
                    x: Vec::new(),
                };
                for kv in it {
                    let (k, v) = kv.split_once('=').unwrap();
                    match k {
                        "entry" => c.entry = v.to_string(),
                        "n" => c.n = v.parse().unwrap(),
                        "round" => c.round_out = v != "0",
                        "col" => {}
                        other => panic!("unknown case key {other}"),
                    }
                }
                cases.push(c);
            }
            "x" => {
                let c = cases.last_mut().unwrap();
                c.x = it
                    .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
                    .collect();
            }
            other => panic!("unknown line {other}"),
        }
    }
    (tensors, cases)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: ds41_fp8_ref <fp8.img> <fp8.cases.txt> <fp8.ref.f32>");
        std::process::exit(2);
    }
    let img = std::fs::read(&args[0]).expect("image");
    let (tensors, cases) = parse(&args[1]);
    let get = |name: &str| -> &Tensor { &tensors.iter().find(|(n, _)| n == name).unwrap().1 };

    let mut out_bytes: Vec<u8> = Vec::new();
    for (n, c) in cases.iter().enumerate() {
        // The round entry runs on the plain tensor's geometry.
        let t = get(if c.entry == "grouped" {
            "grouped"
        } else {
            "plain"
        });
        let (in_dim, out_dim) = if c.entry == "grouped" {
            (t.groups * t.gdim, t.groups * t.rank)
        } else {
            (t.cols, t.rows)
        };
        assert_eq!(c.x.len(), c.n * in_dim, "case {n} x width");
        // XB=1: n > 1 reads activations already rounded to bf16. The >8 arm
        // rounds both sides (v41_fp8blk_to_bf16_kernel + v41_x_to_bf16_kernel).
        let bf16_act = c.n > 1;
        let bf16_w = c.entry == "plain" && c.n > 8;
        for tok in 0..c.n {
            let xt: Vec<f32> = c.x[tok * in_dim..(tok + 1) * in_dim]
                .iter()
                .map(|&v| if bf16_act { bf16_dev(v) } else { v })
                .collect();
            let mut row = vec![0f64; out_dim];
            if c.entry == "grouped" {
                for g in 0..t.groups {
                    for r_loc in 0..t.rank {
                        let wr = g * t.rank + r_loc;
                        let mut acc = 0f64;
                        for col in 0..t.gdim {
                            let mut w = t.w(&img, wr, col);
                            if bf16_w {
                                w = bf16_dev(w);
                            }
                            acc += w as f64 * xt[g * t.gdim + col] as f64;
                        }
                        row[g * t.rank + r_loc] = acc;
                    }
                }
            } else {
                for r in 0..out_dim {
                    let mut acc = 0f64;
                    for col in 0..in_dim {
                        let mut w = t.w(&img, r, col);
                        if bf16_w {
                            w = bf16_dev(w);
                        }
                        acc += w as f64 * xt[col] as f64;
                    }
                    row[r] = acc;
                }
            }
            for v in row {
                let mut f = v as f32;
                if c.round_out {
                    f = bf16r(f);
                }
                out_bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        println!(
            "case {n} {} entry={} n={} round={} -> {}x{}",
            c.kind, c.entry, c.n, c.round_out as u8, c.n, out_dim
        );
    }
    let mut f = std::fs::File::create(&args[2]).expect("ref out");
    f.write_all(&out_bytes).unwrap();
    println!(
        "ds41_fp8_ref: {} cases, {} values",
        cases.len(),
        out_bytes.len() / 4
    );
}
