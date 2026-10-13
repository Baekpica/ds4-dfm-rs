//! ds41_engram_real.rs — builds the P4-3 real-data gate inputs from the
//! artifact and writes the emulation references.
//!
//!   cargo run -p ds4-core --release --example ds41_engram_real -- \
//!     <model.gguf> <engram_dir> <erows_L01.txt> <layer> <out_prefix>
//!
//! Inputs: the golden erows file (the engine's own row ids per position) and
//! the layer's engram_wkv tensor in the GGUF.  Outputs, all under
//! <out_prefix>: .rows.bin (the raw rows: head_dim e4m3 bytes + head_dim/32
//! scale bytes each, the engine's on-disk row form), .erows.ref.f32 (the
//! decoded bf16 grid), .wkv.img (the raw fp8 tensor: e4m3 plane + scale
//! plane) and .wkv.ref.f32 (the wkv matmul over the decoded rows, f64
//! accumulation — the order-free side of the device's f32 GEMV).
//!
//! The row pread goes through EngramShard (O_DIRECT where available), the
//! same reader the P2 gate verified against these very golden files.

use ds4_core::{EngramShard, GgufFile, TensorInventory, V41Wire};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

fn bf16r(x: f32) -> f32 {
    let mut u = x.to_bits();
    if (u & 0x7F80_0000) == 0x7F80_0000 {
        return x;
    }
    u = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    u &= 0xFFFF_0000;
    f32::from_bits(u)
}

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

fn e8m0_to_f32(e: u8) -> f32 {
    let bits: u32 = if e == 0 {
        0x0040_0000
    } else {
        (e as u32) << 23
    };
    f32::from_bits(bits)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        eprintln!("usage: ds41_engram_real <model.gguf> <engram_dir> <erows_L01.txt> <layer> <out_prefix>");
        std::process::exit(2);
    }
    let (gguf, dir, erows_path, layer, prefix) = (
        &args[0],
        &args[1],
        &args[2],
        args[3].parse::<u32>()?,
        &args[4],
    );

    let g = GgufFile::open(Path::new(gguf))?;
    let inv = TensorInventory::open(Path::new(gguf))?;
    let id = ds4_core::identify_file(&g)?;
    let mut w = V41Wire::load(&g, &id.shape)?;
    w.apply_engram_dir(dir);
    let ei = w
        .engram_layers
        .iter()
        .position(|&l| l == layer)
        .ok_or("layer is not an engram layer")?;

    let cols = ((w.engram_max_ngram - 1) * w.engram_heads) as usize; // (G-1)*H = 24
    let hd = w.engram_head_dim as usize;
    let stride = hd + hd / 32;

    // The golden erows: one line per position, `cols` row ids each.
    let text = std::fs::read_to_string(erows_path)?;
    let ids: Vec<i64> = text
        .split_whitespace()
        .map(|v| v.parse::<i64>().unwrap())
        .collect();
    assert_eq!(ids.len() % cols, 0, "erows width");
    let n_pos = ids.len() / cols;

    // Read the rows through the shard (the P2-gated reader).
    let shard = EngramShard::open(Path::new(&w.engram_table_path[ei]))?;
    shard.check_span(
        w.engram_weight_off[ei],
        w.engram_scale_off[ei],
        w.engram_rows[ei],
        u64::from(w.engram_head_dim),
    )?;
    let mut raw: Vec<u8> = Vec::with_capacity(n_pos * cols * stride);
    let mut erows_ref: Vec<u8> = Vec::with_capacity(n_pos * cols * hd * 4);
    for p in 0..n_pos {
        for c in 0..cols {
            let r = ids[p * cols + c] as u64;
            let row = shard.row(
                w.engram_weight_off[ei],
                w.engram_scale_off[ei],
                r,
                w.engram_head_dim,
            )?;
            raw.extend_from_slice(&row.weights);
            raw.extend_from_slice(&row.scale);
            for d in 0..hd {
                let v = bf16r(e4m3_to_f32(row.weights[d]) * e8m0_to_f32(row.scale[d / 32]));
                erows_ref.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    // The layer's wkv tensor, straight from the GGUF.
    let name = format!("blk.{layer}.engram_wkv.weight");
    let t = inv
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("tensor {name} not in the inventory"))?;
    let in_dim = cols * hd;
    let out_dim = (id.shape.n_hc as usize + 1) * id.shape.n_embd as usize;
    let sbr = (out_dim + 31) / 32;
    let sbc = (in_dim + 31) / 32;
    assert_eq!(
        t.bytes as usize,
        in_dim * out_dim + sbr * sbc,
        "wkv tensor size"
    );
    let mut wkv = vec![0u8; t.bytes as usize];
    {
        let mut f = std::fs::File::open(gguf)?;
        f.seek(SeekFrom::Start(t.abs_offset))?;
        f.read_exact(&mut wkv)?;
    }

    // The wkv matmul: n=8 takes the GEMV arm, activations already on the bf16
    // grid (bf16r is the identity there), weights exact, f64 accumulation.
    let mut wkv_ref: Vec<u8> = Vec::with_capacity(n_pos * out_dim * 4);
    for p in 0..n_pos {
        let x = &erows_ref[p * in_dim * 4..(p + 1) * in_dim * 4];
        for r in 0..out_dim {
            let mut acc = 0f64;
            for c in 0..in_dim {
                let b = wkv[r * in_dim + c];
                let s = wkv[in_dim * out_dim + (r / 32) * sbc + c / 32];
                let xv = f32::from_le_bytes(x[c * 4..c * 4 + 4].try_into().unwrap());
                acc += (e4m3_to_f32(b) * e8m0_to_f32(s)) as f64 * xv as f64;
            }
            wkv_ref.extend_from_slice(&(acc as f32).to_le_bytes());
        }
    }

    std::fs::write(format!("{prefix}.rows.bin"), &raw)?;
    std::fs::write(format!("{prefix}.erows.ref.f32"), &erows_ref)?;
    std::fs::write(format!("{prefix}.wkv.img"), &wkv)?;
    std::fs::write(format!("{prefix}.wkv.ref.f32"), &wkv_ref)?;
    println!(
        "ds41_engram_real: L{layer} ei={ei} n_pos={n_pos} cols={cols} hd={hd} rows.bin={} erows_ref={} wkv={} (in {in_dim} out {out_dim}) wkv_ref={}",
        raw.len(),
        erows_ref.len(),
        wkv.len(),
        wkv_ref.len()
    );
    Ok(())
}
