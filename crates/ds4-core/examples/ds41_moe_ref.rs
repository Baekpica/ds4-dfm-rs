//! ds41_moe_ref.rs — the P4-2 MoE emulation oracle for tests/test_ds41_moe.cu.
//!
//!   cargo run -p ds4-core --release --example ds41_moe_ref -- <moe.blob> <moe.cases.txt> <moe.ref.f32>
//!
//! Reads the fixture blob (tests/fixtures/ds41/vq/gen_moe.py) and the case
//! list, decodes through `VqMatrix` (vq.rs) and writes one f32 row of OUT
//! values per case.
//!
//! The arithmetic mirrors cuda/ds41_vq_decode.cuh, not a plausible reading of
//! it: the dot accumulates sum(codebook * x) in f32 and applies the row gain
//! AFTER (row_dot: acc * gain), the gate/up results round through the same
//! RNE f32->bf16->f32 (v41_bf16r, ds41_primitives.cuh:53), swiglu clamps then
//! rounds silu(g)*u to bf16, the down result rounds through bf16r, and the
//! reduce accumulates w[k]*partial in k order.
//!
//! It does NOT replicate the kernel's accumulation ORDER (warp shuffles, FMA
//! contraction) nor CUDA's expf: the onehot cases are built so neither
//! matters (every dot is a single-term sum, and the sole expf output is
//! rounded to bf16 immediately), while the random cases compare within a
//! recorded tolerance.

use ds4_core::VqMatrix;
use std::io::Write;

const CLAMP: f32 = 10.0;

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

/// cuda/ds41_vq_decode.cuh:52-57: clamp -> silu(g)*u -> bf16.
fn swiglu(gi: f32, ui: f32) -> f32 {
    let (mut g, mut u) = (gi, ui);
    if CLAMP > 0.0 {
        if g > CLAMP {
            g = CLAMP;
        }
        if u > CLAMP {
            u = CLAMP;
        }
        if u < -CLAMP {
            u = -CLAMP;
        }
    }
    let sg = g / (1.0 + (-g).exp());
    bf16r(sg * u)
}

/// The kernel's row dot: f32 accumulation of codebook*x, gain applied after
/// (cuda/ds41_vq_row.inc.cu:230-252).
fn row_dot(m: &VqMatrix, r: usize, x: &[f32]) -> f32 {
    let nidx = (m.cols / 8) as usize;
    let mut acc = 0.0f32;
    for k in 0..nidx {
        let w = m.codebook_word(m.index(r, k) as usize);
        for j in 0..8 {
            acc += w[j] * x[k * 8 + j];
        }
    }
    acc * m.gain(r)
}

struct Case {
    kind: String,
    sel: Vec<usize>,
    w: Vec<f32>,
    x: Vec<f32>,
}

fn parse_cases(path: &str, in_dim: usize) -> Vec<Case> {
    let text = std::fs::read_to_string(path).expect("cases file");
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        match it.next().unwrap() {
            "case" => {
                it.next(); // index
                let kind = it.next().unwrap().to_string();
                let k = it
                    .next()
                    .unwrap()
                    .strip_prefix("K=")
                    .unwrap()
                    .parse()
                    .unwrap();
                cases.push(Case {
                    kind,
                    sel: Vec::with_capacity(k),
                    w: Vec::with_capacity(k),
                    x: Vec::new(),
                });
            }
            "sel" => {
                let c = cases.last_mut().unwrap();
                c.sel = it.map(|v| v.parse().unwrap()).collect();
            }
            "w" => {
                let c = cases.last_mut().unwrap();
                c.w = it
                    .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
                    .collect();
            }
            "x" => {
                let c = cases.last_mut().unwrap();
                c.x = it
                    .map(|v| f32::from_bits(u32::from_str_radix(v, 16).unwrap()))
                    .collect();
                assert_eq!(c.x.len(), in_dim, "x width");
            }
            other => panic!("unknown case line {other}"),
        }
    }
    cases
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 && args.len() != 6 {
        eprintln!("usage: ds41_moe_ref <moe.blob> <moe.cases.txt> <moe.ref.f32> [IN MID OUT]");
        std::process::exit(2);
    }
    let (in_dim, mid_dim, out_dim) = if args.len() == 6 {
        (
            args[3].parse::<u32>().expect("IN"),
            args[4].parse::<u32>().expect("MID"),
            args[5].parse::<u32>().expect("OUT"),
        )
    } else {
        (2048u32, 2048u32, 512u32)
    };
    let blob = std::fs::read(&args[0]).expect("blob");
    let cases = parse_cases(&args[1], in_dim as usize);
    let mut out_bytes: Vec<u8> = Vec::new();
    // DS41_MOE_REF_DUMP=1 writes the per-pair mid (bf16) and down partials
    // (f32) to /tmp/ref_{mid,part}.bin, the emulation side of the P4-2
    // acceptance comparison against the device's DS41_MOE_DUMP_MID dumps.
    let dump = std::env::var("DS41_MOE_REF_DUMP").is_ok();
    let mut mid_dump: Vec<u8> = Vec::new();
    let mut part_dump: Vec<u8> = Vec::new();
    for (n, c) in cases.iter().enumerate() {
        let k = c.sel.len();
        assert_eq!(c.w.len(), k);
        // The worker packs the activation to bf16 once (v41_vq_xpack_kernel);
        // the dots read the packed values, so the emulation must too.
        let xb: Vec<f32> = c.x.iter().map(|&v| bf16r(v)).collect();
        let mut acc = vec![0.0f32; out_dim as usize];
        for kk in 0..k {
            let e = c.sel[kk];
            let gate = VqMatrix::open(&blob, e, 0, mid_dim, in_dim).expect("gate");
            let up = VqMatrix::open(&blob, e, 1, mid_dim, in_dim).expect("up");
            let down = VqMatrix::open(&blob, e, 2, out_dim, mid_dim).expect("down");
            let mut mid = vec![0.0f32; mid_dim as usize];
            for r in 0..mid_dim as usize {
                let gv = bf16r(row_dot(&gate, r, &xb));
                let ui = bf16r(row_dot(&up, r, &xb));
                mid[r] = swiglu(gv, ui);
            }
            if dump {
                for v in &mid {
                    // the device stores the mid as bf16; dump in that form
                    let u = bf16r(*v).to_bits() >> 16;
                    mid_dump.extend_from_slice(&(u as u16).to_le_bytes());
                }
            }
            for o in 0..out_dim as usize {
                let partial = bf16r(row_dot(&down, o, &mid));
                acc[o] += c.w[kk] * partial;
                if dump {
                    part_dump.extend_from_slice(&partial.to_le_bytes());
                }
            }
        }
        for v in &acc {
            out_bytes.extend_from_slice(&v.to_le_bytes());
        }
        println!("case {n} {} K={k} -> {out_dim} values", c.kind);
    }
    let mut f = std::fs::File::create(&args[2]).expect("ref out");
    f.write_all(&out_bytes).unwrap();
    if dump {
        std::fs::write("/tmp/ref_mid.bin", &mid_dump).unwrap();
        std::fs::write("/tmp/ref_part.bin", &part_dump).unwrap();
        println!(
            "ds41_moe_ref: dumped mid {} B, part {} B",
            mid_dump.len(),
            part_dump.len()
        );
    }
    println!(
        "ds41_moe_ref: {} cases, {} values",
        cases.len(),
        out_bytes.len() / 4
    );
}
