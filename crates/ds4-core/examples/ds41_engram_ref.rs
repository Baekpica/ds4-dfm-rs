//! ds41_engram_ref.rs — the P4-3 engram emulation oracle for tests/test_ds41_engram.cu.
//!
//!   cargo run -p ds4-core --release --example ds41_engram_ref -- \
//!     <engram.img> <engram.cases.txt> <engram.ref.f32> <rows.bin> <rows.ref.f32>
//!
//! The gate arithmetic mirrors cuda/ds41_engram.cuh (the engine's
//! cuda_v41_3.inc.cu:109-134) in f64: the three sums, rstd, dot, mag, z,
//! sigmoid, and the bf16r-quantized update h[d] = bf16r(h[d] + gate*val[d]).
//! The device's f32 reduction order and fast-math expf differ; the output is
//! bf16-quantized, so the compare is at bf16-ulp distance (the 'absorb' case
//! is engineered under half an ulp and must be bit-exact).
//!
//! The rows half decodes the on-disk row form (head_dim e4m3 bytes + the
//! ue8m0 tail of head_dim/32 scale bytes, ds4_fp8.h semantics) and must be
//! bit-exact against the device: every product is exact and the bf16r
//! quantization is the same function on both sides.

use std::io::Write;

/// RNE round f32 -> bf16 -> f32, the device's v41_bf16r bit for bit.
fn bf16r(x: f32) -> f32 {
    let mut u = x.to_bits();
    if (u & 0x7F80_0000) == 0x7F80_0000 {
        return x;
    }
    u = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    u &= 0xFFFF_0000;
    f32::from_bits(u)
}

/// E4M3FN bit decode, mirroring ds4_e4m3fn_to_f32 (ds41_fp8blk.cuh).
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

/// E8M0 scale byte -> f32, mirroring ds4_e8m0_to_f32.
fn e8m0_to_f32(e: u8) -> f32 {
    let bits: u32 = if e == 0 {
        0x0040_0000
    } else {
        (e as u32) << 23
    };
    f32::from_bits(bits)
}

struct GateCase {
    kind: String,
    n: usize,
    h: Vec<f32>,
    kv: Vec<f32>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        eprintln!("usage: ds41_engram_ref <engram.img> <engram.cases.txt> <engram.ref.f32> <rows.bin> <rows.ref.f32>");
        std::process::exit(2);
    }
    let img = std::fs::read(&args[0]).expect("image");
    let (e, hc, q_off, k_off, eps) = parse_head(&args[1]);
    let cases = parse_cases(&args[1], hc * e);
    let rows = std::fs::read(&args[3]).expect("rows");

    let rd = |off: usize| -> f32 { f32::from_le_bytes(img[off..off + 4].try_into().unwrap()) };

    let mut out: Vec<u8> = Vec::new();
    for (n, c) in cases.iter().enumerate() {
        let n_tok = c.n;
        assert_eq!(c.h.len(), n_tok * hc * e);
        assert_eq!(c.kv.len(), n_tok * (hc + 1) * e);
        for t in 0..n_tok {
            for ci in 0..hc {
                let hrow = &c.h[(t * hc + ci) * e..(t * hc + ci + 1) * e];
                let krow = &c.kv[(t * (hc + 1) + ci) * e..(t * (hc + 1) + ci + 1) * e];
                let vrow = &c.kv[(t * (hc + 1) + hc) * e..(t * (hc + 1) + hc + 1) * e];
                let mut sh = 0f64;
                let mut sk = 0f64;
                let mut sd = 0f64;
                for d in 0..e {
                    let w =
                        rd(q_off + (ci * e + d) * 4) as f64 * rd(k_off + (ci * e + d) * 4) as f64;
                    let hv = hrow[d] as f64;
                    let kv = krow[d] as f64;
                    sh += hv * hv;
                    sk += kv * kv;
                    sd += hv * w * kv;
                }
                let rstd =
                    (1.0 / (sh / e as f64 + eps).sqrt()) * (1.0 / (sk / e as f64 + eps).sqrt());
                let dot = sd * rstd / (e as f64).sqrt();
                let mag = dot.abs().max(1e-6).sqrt();
                let z = mag.copysign(dot);
                let gate = 1.0 / (1.0 + (-z).exp());
                for d in 0..e {
                    let v = (hrow[d] as f64 + gate * vrow[d] as f64) as f32;
                    out.extend_from_slice(&bf16r(v).to_le_bytes());
                }
            }
        }
        println!(
            "gate case {n} {} n={} -> {}x{}",
            c.kind,
            n_tok,
            n_tok * hc,
            e
        );
    }
    std::fs::File::create(&args[2])
        .expect("ref out")
        .write_all(&out)
        .unwrap();

    // Rows: head_dim e4m3 bytes + head_dim/32 ue8m0 bytes per row.
    let hd = 256usize;
    let stride = hd + hd / 32;
    assert_eq!(rows.len() % stride, 0, "rows.bin size");
    let n_rows = rows.len() / stride;
    let mut rout: Vec<u8> = Vec::new();
    for r in 0..n_rows {
        for d in 0..hd {
            let b = rows[r * stride + d];
            let s = rows[r * stride + hd + d / 32];
            let v = bf16r(e4m3_to_f32(b) * e8m0_to_f32(s));
            rout.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(&args[4])
        .expect("rows ref")
        .write_all(&rout)
        .unwrap();
    println!(
        "ds41_engram_ref: {} gate cases ({} values), {} rows ({} values)",
        cases.len(),
        out.len() / 4,
        n_rows,
        rout.len() / 4
    );
}

fn parse_head(path: &str) -> (usize, usize, usize, usize, f64) {
    let text = std::fs::read_to_string(path).expect("cases file");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("gate ") {
            let mut e = 0usize;
            let mut hc = 0usize;
            let mut q_off = 0usize;
            let mut k_off = 0usize;
            let mut eps = 0f64;
            for kv in rest.split_whitespace() {
                let (k, v) = kv.split_once('=').unwrap();
                match k {
                    "e" => e = v.parse().unwrap(),
                    "hc" => hc = v.parse().unwrap(),
                    "q_off" => q_off = v.parse().unwrap(),
                    "k_off" => k_off = v.parse().unwrap(),
                    "eps" => eps = v.parse().unwrap(),
                    other => panic!("unknown gate key {other}"),
                }
            }
            return (e, hc, q_off, k_off, eps);
        }
    }
    panic!("no gate descriptor line");
}

fn parse_cases(path: &str, hc_e: usize) -> Vec<GateCase> {
    let text = std::fs::read_to_string(path).expect("cases file");
    let mut cases: Vec<GateCase> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        match it.next().unwrap() {
            "gate" => {}
            "case" => {
                it.next();
                let kind = it.next().unwrap().to_string();
                let mut n = 0usize;
                for kv in it {
                    let (k, v) = kv.split_once('=').unwrap();
                    if k == "n" {
                        n = v.parse().unwrap();
                    }
                }
                assert!(n > 0);
                cases.push(GateCase {
                    kind,
                    n,
                    h: Vec::with_capacity(n * hc_e),
                    kv: Vec::new(),
                });
            }
            "h" => {
                let c = cases.last_mut().unwrap();
                c.h = it
                    .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
                    .collect();
            }
            "kv" => {
                let c = cases.last_mut().unwrap();
                c.kv = it
                    .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
                    .collect();
            }
            other => panic!("unknown line {other}"),
        }
    }
    cases
}
