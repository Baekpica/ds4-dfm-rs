//! Gate: the device VQ row dot against the CPU forward, on the real artifact.
//!
//! ```
//! make tests/gemma4_gpu_gate
//! ./tests/gemma4_gpu_gate --dir /data/vq-artifacts/gemma-4-26b-a4b-8x4096
//! ```
//!
//! Step 1 of the GPU unit is residency: every encoded container is mapped,
//! checked and uploaded once, and the VRAM account is printed before the first
//! launch. Step 2 is the DQV3 row kernel: the same input vector is dotted with
//! two tensors on both sides, the CPU fused dot
//! ([`Matrix::matvec`], the forward's own path) and the device kernel, and the
//! pair is compared as relL2 and maxabs.
//!
//! The two tensors are chosen because they are the artifact's two geometries:
//! `q_proj` is dim 8, nc 4096, so its 12-bit stream runs the block kernel;
//! `router.proj` is dim 4, nc 256, so its 8-bit stream runs the dim-4 fast
//! path. Both are two stages summed with the gain applied after each stage's
//! dot.
//!
//! The gate is not a tolerance dressed up as a proof: the CPU oracle runs on
//! the same input in the same process, and the printed relL2 is the whole
//! claim. A run that cannot upload the artifact fails loudly; nothing falls
//! back to host streaming.
//!
//! Step 3 is the routed gather: a mixture-of-experts layer dots eight chosen
//! experts, not 128, so the device side must read the selected rows only.
//! `GATHER` runs the eight-expert shape against the CPU's `dot_rows`, then
//! against a f64 reference built from the per-row decode (a different CPU
//! path), then `MIX` proves the list is honored as an order, not a range:
//! two experts interleaved and one row repeated.

use std::ffi::{c_char, c_void, CString};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::f16_to_f32;
use ds4_core::gemma4_a4b::{bf16_round, decode, gelu, rms_norm, Entry, Family, Matrix, Tensor};

/// Mirrors `gm4_spec` in native/gemma4/gemma4_gpu.h.
#[repr(C)]
struct Spec {
    path: *const c_char,
    rows: u32,
    cols: u32,
}

/// Mirrors `gm4_account`.
#[repr(C)]
#[derive(Default)]
struct Account {
    files_bytes: u64,
    payload_bytes: u64,
    device_bytes: u64,
    device_free_before: u64,
    device_free_after: u64,
    tensors: i32,
    payloads: i32,
}

extern "C" {
    fn gm4_gpu_open(specs: *const Spec, n: i32, err: *mut c_char, errlen: usize) -> *mut c_void;
    fn gm4_gpu_close(g: *mut c_void);
    fn gm4_gpu_account_of(g: *const c_void, out: *mut Account);
    fn gm4_gpu_device_name(g: *const c_void) -> *const c_char;
    fn gm4_gpu_variant(g: *const c_void, tensor: i32) -> i32;
    fn gm4_gpu_matvec(
        g: *mut c_void,
        tensor: i32,
        x: *const f32,
        y: *mut f32,
        threads: i32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_rows(
        g: *mut c_void,
        tensor: i32,
        rows: *const u32,
        nrows: i32,
        x: *const f32,
        y: *mut f32,
        threads: i32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_prefill(
        g: *mut c_void,
        tensor: i32,
        xt: *const f32,
        ntok: i32,
        y: *mut f32,
        bn: i32,
        cb: i32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn gm4_gpu_prefill_rows(
        g: *mut c_void,
        tensor: i32,
        row0: i32,
        nrows: i32,
        xt: *const f32,
        ntok: i32,
        y: *mut f32,
        bn: i32,
        cb: i32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_prefill_ready(
        g: *const c_void,
        tensor: i32,
        bn: i32,
        cb: i32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn gm4_gpu_rmsnorm(
        g: *mut c_void,
        x: *const f32,
        rows: i32,
        width: i32,
        weight: *const f32,
        y: *mut f32,
        eps: f32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_gelu_mul(
        g: *mut c_void,
        gate: *const f32,
        up: *const f32,
        act: *mut f32,
        n: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_residual_scale(
        g: *mut c_void,
        x: *const f32,
        add: *const f32,
        out: *mut f32,
        n: usize,
        scale: f32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_softcap(
        g: *mut c_void,
        y: *mut f32,
        n: usize,
        cap: f32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
}

struct Residency(*mut c_void);

impl Drop for Residency {
    fn drop(&mut self) {
        unsafe { gm4_gpu_close(self.0) }
    }
}

fn usage() -> &'static str {
    "gemma4_a4b_gpu_gate - the device VQ row dot against the CPU forward

  --dir DIR          artifact directory (vq-release.json)
  --threads N        launch width for the warp-per-row kernels (default 256)
  --tensor NAME      tensor to gate; repeatable (default: q_proj and router)
  --timing N         best of N launches per shape, plus the effective GB/s
  --skip-gather      skip the routed-expert gather (step 3)
  --skip-prefill     skip the tensor-core prefill (step 4)
  --skip-elementwise skip the norms and elementwise kernels (step 5)
  --help             this text"
}

const DEFAULT_Q: &str = "model.language_model.layers.0.self_attn.q_proj.weight";
const DEFAULT_ROUTER: &str = "model.language_model.layers.0.router.proj.weight";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(detail) => {
            eprintln!("FAIL {detail}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::type_complexity)]
fn run() -> Result<(), String> {
    let mut dir: Option<String> = None;
    let mut threads = 256i32;
    let mut timing = 0usize;
    let mut wanted: Vec<String> = Vec::new();
    let mut onehot: Vec<usize> = Vec::new();
    let mut gather = true;
    let mut prefill = true;
    let mut elementwise = true;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dir" => dir = args.next(),
            "--threads" => {
                threads = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--threads needs a number")?
            }
            "--tensor" => wanted.push(args.next().ok_or("--tensor needs a name")?),
            "--onehot" => {
                let list = args.next().ok_or("--onehot needs indices")?;
                for part in list.split(',') {
                    onehot.push(part.parse().map_err(|_| "bad --onehot index")?);
                }
            }
            "--timing" => {
                timing = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--timing needs a number")?
            }
            "--skip-gather" => gather = false,
            "--skip-prefill" => prefill = false,
            "--skip-elementwise" => elementwise = false,
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let dir: PathBuf = dir
        .ok_or_else(|| format!("--dir is required\n\n{}", usage()))?
        .into();
    let family = Family::open(&dir).map_err(|e| e.to_string())?;
    let root = family.dir.clone();
    if wanted.is_empty() {
        wanted.push(DEFAULT_Q.to_string());
        wanted.push(DEFAULT_ROUTER.to_string());
    }

    // The release's entry list is the residency's own inventory: only encoded
    // containers are uploaded, and each is named with the shape the release
    // records, which the device header check repeats.
    let encoded: Vec<&Entry> = family
        .release
        .entries()
        .iter()
        .filter(|e| e.encoded)
        .collect();
    if encoded.is_empty() {
        return Err("the release holds no encoded entries".into());
    }
    let mut names: Vec<CString> = Vec::with_capacity(encoded.len());
    let mut specs: Vec<Spec> = Vec::with_capacity(encoded.len());
    for entry in &encoded {
        let path = root.join(format!("{}.bin", entry.name));
        let name = CString::new(path.to_string_lossy().as_ref())
            .map_err(|_| format!("{}: path holds a NUL", path.display()))?;
        specs.push(Spec {
            path: name.as_ptr(),
            rows: entry.rows as u32,
            cols: entry.cols as u32,
        });
        names.push(name);
    }

    println!(
        "UPLOAD {} encoded containers from {}",
        specs.len(),
        root.display()
    );
    let started = Instant::now();
    let mut err = [0i8; 512];
    let handle = unsafe {
        gm4_gpu_open(
            specs.as_ptr(),
            specs.len() as i32,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if handle.is_null() {
        return Err(format!("open: {}", message(&err)));
    }
    let device = Residency(handle);
    let mut acct = Account::default();
    unsafe { gm4_gpu_account_of(device.0, &mut acct) };
    let name = unsafe { std::ffi::CStr::from_ptr(gm4_gpu_device_name(device.0)) }
        .to_string_lossy()
        .into_owned();
    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    println!(
        "DEVICE \"{name}\" tensors {} payloads {} files {:.1} MiB payload {:.1} MiB device {:.1} MiB",
        acct.tensors,
        acct.payloads,
        mib(acct.files_bytes),
        mib(acct.payload_bytes),
        mib(acct.device_bytes)
    );
    println!(
        "VRAM free before {:.1} MiB, after {:.1} MiB, residency took {:.2} s",
        mib(acct.device_free_before),
        mib(acct.device_free_after),
        started.elapsed().as_secs_f64()
    );

    let mut failed = false;
    for target in &wanted {
        let index = encoded
            .iter()
            .position(|e| &e.name == target)
            .ok_or_else(|| format!("{target}: not an encoded entry of this release"))?;
        let entry = encoded[index];
        let matrix = Matrix::open(&root, entry).map_err(|e| e.to_string())?;
        if matrix.rows() != entry.rows || matrix.cols() != entry.cols {
            return Err(format!(
                "{target}: CPU matrix shape disagrees with the release"
            ));
        }
        let x = probe_input(matrix.cols());

        let cpu_start = Instant::now();
        let cpu = matrix.matvec(&x).map_err(|e| e.to_string())?;
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3;

        let mut gpu = vec![0f32; entry.rows];
        let t0 = Instant::now();
        call_matvec(&device, index, &x, &mut gpu, threads)?;
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;

        let variant = unsafe { gm4_gpu_variant(device.0, index as i32) };
        let (rel_l2, max_abs, scale) = compare(&cpu, &gpu);
        println!(
            "TENSOR {target} {}x{} variant {variant} cpu {:.2} ms gpu {:.2} ms relL2 {:.3e} maxabs {:.3e} scale {:.3e}",
            entry.rows, entry.cols, cpu_ms, gpu_ms, rel_l2, max_abs, scale
        );
        if rel_l2 > 1e-4 {
            println!("  FAIL relL2 {rel_l2:.3e} is past 1e-4");
            failed = true;
        }

        // The independent reference: the whole tensor decoded to f32, then a
        // f64 dot with the same input. It says which side the 1e-4 gate is
        // actually measuring, instead of letting the two f32 paths agree with
        // each other and call it proof.
        let decoded = decode(&root, entry).map_err(|e| e.to_string())?;
        let f64_ref = f64_dots(&decoded, &x);
        let (cpu_vs_f64, _, _) = compare_f64(&cpu, &f64_ref);
        let (gpu_vs_f64, _, _) = compare_f64(&gpu, &f64_ref);
        println!(
            "  F64 cpu {cpu_vs_f64:.3e} gpu {gpu_vs_f64:.3e} over {} rows",
            entry.rows
        );
        if gpu_vs_f64 > 1e-5 || cpu_vs_f64 > 1e-5 {
            println!("  FAIL a path is past 1e-5 against the f64 reference");
            failed = true;
        }

        // One-hot probes isolate the gather from the accumulation: with x on a
        // single column, a row's dot is exactly one stored value, so any
        // difference is an index or gain error, not rounding.
        for &i in &onehot {
            if i >= entry.cols {
                return Err(format!("--onehot {i} is past {} columns", entry.cols));
            }
            let mut xh = vec![0f32; entry.cols];
            xh[i] = 1.0;
            let mut yh = vec![0f32; entry.rows];
            call_matvec(&device, index, &xh, &mut yh, threads)?;
            let mut max = 0f64;
            let mut worst = 0usize;
            for (r, &v) in yh.iter().enumerate() {
                let want = f64::from(decoded.data[r * entry.cols + i]);
                let d = (f64::from(v) - want).abs();
                if d > max {
                    max = d;
                    worst = r;
                }
            }
            println!(
                "  ONEHOT i={i} maxabs {max:.3e} at row {worst} (gpu {:.9}, decoded {:.9})",
                yh[worst],
                decoded.data[worst * entry.cols + i]
            );
            if max > 1e-6 {
                failed = true;
            }
        }

        if timing > 0 {
            let mut best = f64::INFINITY;
            for _ in 0..timing {
                let t = Instant::now();
                call_matvec(&device, index, &x, &mut gpu, threads)?;
                best = best.min(t.elapsed().as_secs_f64());
            }
            let gbs = entry.bytes as f64 / best / 1e9;
            println!(
                "  TIMING best {:.3} ms of {} launches, {:.1} GB/s over the container bytes ({} B)",
                best * 1e3,
                timing,
                gbs,
                entry.bytes
            );
        }
    }
    if gather {
        gather_gate(
            &device,
            &family,
            &root,
            &encoded,
            threads,
            timing,
            &mut failed,
        )?;
    }
    if prefill {
        prefill_gate(&device, &root, &encoded, timing, &mut failed)?;
    }
    if elementwise {
        elementwise_gate(&device, &mut failed)?;
    }
    if failed {
        return Err("a kernel is past the gate".into());
    }
    Ok(())
}

/// The eight experts the gather is measured on. The ids are arbitrary but
/// fixed, and they span the stack, so an expert base that is off by a slice
/// shows up as a large relL2 instead of a plausible number.
const GATHER_EXPERTS: [usize; 8] = [0, 3, 17, 42, 64, 91, 120, 127];

/// Step 3: the routed gather on both expert stacks of layer 0.
///
/// `experts.gate_up_proj` is 180224 x 2816 and `experts.down_proj` is
/// 360448 x 704, both dim 8 / nc 4096, so both run the block kernel. The
/// routed shape is eight whole expert slices, expert-major; a mixture layer
/// never reads the other 120.
fn gather_gate(
    device: &Residency,
    family: &Family,
    root: &Path,
    encoded: &[&Entry],
    threads: i32,
    timing: usize,
    failed: &mut bool,
) -> Result<(), String> {
    let config = &family.config;
    let layer = "model.language_model.layers.0";
    let stack = [
        (
            "expert_gate_up",
            format!("{layer}.experts.gate_up_proj"),
            2 * config.moe_ff,
        ),
        (
            "expert_down",
            format!("{layer}.experts.down_proj"),
            config.hidden,
        ),
    ];
    for (tag, name, per_expert) in stack {
        let index = encoded
            .iter()
            .position(|e| e.name == name)
            .ok_or_else(|| format!("{name}: not an encoded entry of this release"))?;
        let entry = encoded[index];
        let expected = config.experts * per_expert;
        if entry.rows != expected {
            return Err(format!(
                "{name}: {} rows, the config says {} experts x {per_expert} = {expected}",
                entry.rows, config.experts
            ));
        }
        if config.experts < 2 {
            return Err(format!(
                "{tag}: the config routes from {} experts",
                config.experts
            ));
        }
        let matrix = Matrix::open(root, entry).map_err(|e| e.to_string())?;
        let x = probe_input(entry.cols);

        let mut rows: Vec<u32> = Vec::with_capacity(GATHER_EXPERTS.len() * per_expert);
        for &expert in &GATHER_EXPERTS {
            let base = expert * per_expert;
            rows.extend((base..base + per_expert).map(|r| r as u32));
        }
        let named: Vec<usize> = rows.iter().map(|&r| r as usize).collect();

        let cpu_start = Instant::now();
        let cpu = matrix.dot_rows(&named, &x).map_err(|e| e.to_string())?;
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3;

        let mut gpu = vec![0f32; rows.len()];
        let started = Instant::now();
        call_rows(device, index, &rows, &x, &mut gpu, threads)?;
        let gpu_ms = started.elapsed().as_secs_f64() * 1e3;

        let (rel_l2, max_abs, scale) = compare(&cpu, &gpu);
        println!(
            "GATHER {name} {}x{} rows {} of {} experts {:?} cpu {:.1} ms gpu {:.3} ms relL2 {:.3e} maxabs {:.3e} scale {:.3e}",
            entry.rows,
            entry.cols,
            rows.len(),
            entry.rows,
            GATHER_EXPERTS,
            cpu_ms,
            gpu_ms,
            rel_l2,
            max_abs,
            scale
        );
        if rel_l2 > 1e-4 {
            println!("  FAIL {tag} gather relL2 {rel_l2:.3e} is past 1e-4");
            *failed = true;
        }

        // The independent reference, from the per-row decode path rather than
        // the batched one the CPU oracle used, accumulated in f64.
        let mut reference = vec![0f64; named.len()];
        for (slot, &row) in named.iter().enumerate() {
            let values = matrix.row(row).map_err(|e| e.to_string())?;
            let mut sum = 0f64;
            for (&w, &v) in values.iter().zip(&x) {
                sum += f64::from(w) * f64::from(v);
            }
            reference[slot] = sum;
        }
        let (cpu_vs_f64, _, _) = compare_f64(&cpu, &reference);
        let (gpu_vs_f64, _, _) = compare_f64(&gpu, &reference);
        println!(
            "  F64 cpu {cpu_vs_f64:.3e} gpu {gpu_vs_f64:.3e} over {} gathered rows",
            rows.len()
        );
        if gpu_vs_f64 > 1e-5 || cpu_vs_f64 > 1e-5 {
            println!("  FAIL a gather path is past 1e-5 against the f64 reference");
            *failed = true;
        }

        // The list is an order, not a range: interleave the first and last
        // expert in opposite directions, then repeat a row. A gather that
        // read contiguously from row 0 would still pass the shape above, and
        // would fail here.
        let mut mix: Vec<u32> = Vec::with_capacity(129);
        let low = 0usize;
        let high = (config.experts - 1) * per_expert;
        for k in 0..64 {
            mix.push((high + per_expert - 1 - k) as u32);
            mix.push((low + k) as u32);
        }
        mix.push(mix[0]);
        let mix_named: Vec<usize> = mix.iter().map(|&r| r as usize).collect();
        let mix_cpu = matrix.dot_rows(&mix_named, &x).map_err(|e| e.to_string())?;
        let mut mix_gpu = vec![0f32; mix.len()];
        call_rows(device, index, &mix, &x, &mut mix_gpu, threads)?;
        let (mix_rel, mix_abs, _) = compare(&mix_cpu, &mix_gpu);
        let repeat_exact = mix_gpu[0].to_bits() == mix_gpu[mix.len() - 1].to_bits();
        println!(
            "  MIX interleaved 128 rows + repeat: relL2 {mix_rel:.3e} maxabs {mix_abs:.3e} repeat bit-identical {repeat_exact}"
        );
        if mix_rel > 1e-4 || !repeat_exact {
            println!("  FAIL the gather does not follow the row list");
            *failed = true;
        }

        // A row past the tensor must be refused, not read as the next payload.
        let bad = [entry.rows as u32];
        let mut sink = [0f32; 1];
        let mut err = [0i8; 512];
        let rc = unsafe {
            gm4_gpu_rows(
                device.0,
                index as i32,
                bad.as_ptr(),
                1,
                x.as_ptr(),
                sink.as_mut_ptr(),
                threads,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc == 0 {
            println!(
                "  FAIL row {} was accepted, it is past {} rows",
                bad[0], entry.rows
            );
            *failed = true;
        } else {
            println!("  REFUSE row {}: {}", bad[0], message(&err));
        }

        if timing > 0 {
            let mut best = f64::INFINITY;
            for _ in 0..timing {
                let t = Instant::now();
                call_rows(device, index, &rows, &x, &mut gpu, threads)?;
                best = best.min(t.elapsed().as_secs_f64());
            }
            let per_row = entry.bytes as f64 / entry.rows as f64;
            let mib = (rows.len() as f64 * per_row) / (1024.0 * 1024.0);
            println!(
                "  TIMING best {:.3} ms of {} launches, {:.1} MiB of the stack read at {:.1} GB/s",
                best * 1e3,
                timing,
                mib,
                rows.len() as f64 * per_row / best / 1e9
            );
        }
    }
    Ok(())
}

/// The tile shapes the tools' build offers, as (token tile, codebook
/// placement). The column round is 64 in each: BK 128 would need a 96-bit
/// index window the A-tile fill does not have (`decode-gpu.cu:543-563`).
const PREFILL_TILES: [(i32, i32); 6] = [(64, 0), (128, 0), (256, 0), (64, 1), (128, 1), (256, 1)];

/// Step 4: the tiled f16 prefill on the real containers.
///
/// The tiled kernel rounds the activations to f16 (the codebook is already
/// f16, and an f16 x f16 product is exact in f32). The reference therefore
/// dots the same rows against activations rounded the same way, in f64:
/// comparing against an unrounded f32 reference would measure the f16 input
/// rounding instead of the kernel. That rounding is reported separately, as
/// the cost of the tensor-core path.
fn prefill_gate(
    device: &Residency,
    root: &Path,
    encoded: &[&Entry],
    timing: usize,
    failed: &mut bool,
) -> Result<(), String> {
    // The rounding helper is the reference for every gate below; if it were
    // wrong, the gates would be wrong with it. Round-trip known f16 patterns
    // first, including the subnormal the artifact actually carries.
    for (value, bits) in [
        (0.0f32, 0x0000u16),
        (-2.0, 0xC000),
        (1.0, 0x3C00),
        (2f32.powi(-24), 0x0001),
        (5.155_801_8e-5, 0x0361),
        (65504.0, 0x7BFF),
        (65520.0, 0x7C00),
    ] {
        assert_eq!(to_f16_bits(value), bits, "f16 rounding of {value}");
    }
    let layer = "model.language_model.layers.0";
    let q_name = format!("{layer}.self_attn.q_proj.weight");
    let index = encoded
        .iter()
        .position(|e| e.name == q_name)
        .ok_or_else(|| format!("{q_name}: not an encoded entry of this release"))?;
    let entry = encoded[index];
    let matrix = Matrix::open(root, entry).map_err(|e| e.to_string())?;

    // Two prompt sizes: a short one where the f64 reference is cheap, and one
    // long enough to fill the tile's 128 tokens with a partial second block.
    for &ntok in &[8usize, 64] {
        let xt = transposed_probe(entry.cols, ntok);
        let xt16: Vec<f32> = xt.iter().map(|&v| round_f16(v)).collect();
        let reference = f64_prefill(&matrix, &xt16, ntok, 0, entry.rows);
        println!(
            "PREFILL {q_name} {}x{} ntok {ntok} reference f64 over f16-round activations",
            entry.rows, entry.cols
        );
        let mut best_tile: Option<(f64, f64, i32, i32)> = None;
        for (bn, cb) in PREFILL_TILES {
            let mut why = [0i8; 512];
            let ready = unsafe {
                gm4_gpu_prefill_ready(device.0, index as i32, bn, cb, why.as_mut_ptr(), why.len())
            };
            if ready == 0 {
                println!("  TILE bn {bn} cb {cb} REFUSED: {}", message(&why));
                continue;
            }
            let mut y = vec![0f32; ntok * entry.rows];
            let started = Instant::now();
            call_prefill(device, index, &xt, ntok, &mut y, bn, cb)?;
            let ms = started.elapsed().as_secs_f64() * 1e3;
            let (rel, max) = compare_f32_f64(&y, &reference);
            println!("  TILE bn {bn} cb {cb} relL2 {rel:.3e} maxabs {max:.3e} {ms:.3} ms");
            if rel > 1e-5 {
                println!("  FAIL tile {bn}/{cb} is past 1e-5 against the f64 reference");
                *failed = true;
            }
            if best_tile.map_or(true, |(b, _, _, _)| ms < b) {
                best_tile = Some((ms, rel, bn, cb));
            }

            // The remaining checks run on one tile, so they do not repeat per
            // configuration.
            if (bn, cb) != (128, 0) {
                continue;
            }

            // What the f16 input rounding costs: the same tensor against the
            // unrounded CPU path, one matvec per token.
            let mut f32_ref = vec![0f32; ntok * entry.rows];
            for t in 0..ntok {
                let x: Vec<f32> = (0..entry.cols).map(|c| xt[c * ntok + t]).collect();
                let one = matrix.matvec(&x).map_err(|e| e.to_string())?;
                f32_ref[t * entry.rows..(t + 1) * entry.rows].copy_from_slice(&one);
            }
            let (vs32, max32, _) = compare(&f32_ref, &y);
            println!(
                "    VSF32 relL2 {vs32:.3e} maxabs {max32:.3e} (the f16 input rounding; gate 2e-3)"
            );
            if vs32 > 2e-3 {
                println!("  FAIL f16 rounding costs more than the 2e-3 band");
                *failed = true;
            }

            // A one-hot column is exact in f16, so every row's output must be
            // the decoded value at that column, for every token.
            let pick = entry.cols / 3;
            let mut xt_one = vec![0f32; entry.cols * ntok];
            for t in 0..ntok {
                xt_one[pick * ntok + t] = 1.0;
            }
            let mut y_one = vec![0f32; ntok * entry.rows];
            call_prefill(device, index, &xt_one, ntok, &mut y_one, bn, cb)?;
            let decoded: Vec<f32> = (0..entry.rows)
                .map(|r| {
                    // The decoded value is f16 codebook times f16 gain in f32,
                    // and the kernel applies the gain after the mma: nothing
                    // rounds the product, so the reference must not either.
                    let row = matrix.row(r).expect("row is inside the matrix");
                    row[pick]
                })
                .collect();
            let mut max = 0f64;
            let mut worst = (0usize, 0usize);
            for t in 0..ntok {
                for r in 0..entry.rows {
                    let got = f64::from(y_one[t * entry.rows + r]);
                    let d = (got - f64::from(decoded[r])).abs();
                    if d > max {
                        max = d;
                        worst = (t, r);
                    }
                }
            }
            println!(
                "    ONEHOT column {pick} maxabs {max:.3e} at token {} row {} (device {:.9})",
                worst.0,
                worst.1,
                y_one[worst.0 * entry.rows + worst.1]
            );
            if max > 1e-6 {
                println!("  FAIL a one-hot column is not reproduced exactly");
                *failed = true;
            }

            if timing > 0 {
                let mut best = f64::INFINITY;
                for _ in 0..timing {
                    let t = Instant::now();
                    call_prefill(device, index, &xt, ntok, &mut y, bn, cb)?;
                    best = best.min(t.elapsed().as_secs_f64());
                }
                let flops = 2.0 * ntok as f64 * entry.rows as f64 * entry.cols as f64;
                println!(
                    "    TIMING best {:.3} ms of {} launches, {:.1} GFLOP/s over the ideal count",
                    best * 1e3,
                    timing,
                    flops / best / 1e9
                );
            }
        }
        if let Some((ms, rel, bn, cb)) = best_tile {
            println!("  BEST tile {bn}/{cb} at {ms:.3} ms, relL2 {rel:.3e}");
        }
    }

    // The fallback: the router's dim-4, 8-bit codebook cannot run the tiled
    // kernel, so bn 0 takes the legacy one-thread-per-(row, token) kernel. It
    // keeps f32 activations, so its reference is unrounded.
    let router_name = format!("{layer}.router.proj.weight");
    let router_index = encoded
        .iter()
        .position(|e| e.name == router_name)
        .ok_or_else(|| format!("{router_name}: not an encoded entry of this release"))?;
    let router = encoded[router_index];
    let matrix = Matrix::open(root, router).map_err(|e| e.to_string())?;
    let ntok = 8usize;
    let xt = transposed_probe(router.cols, ntok);
    let reference = f64_prefill(&matrix, &xt, ntok, 0, router.rows);
    let mut y = vec![0f32; ntok * router.rows];
    call_prefill(device, router_index, &xt, ntok, &mut y, 0, 0)?;
    let (rel, max) = compare_f32_f64(&y, &reference);
    println!(
        "PREFILL {router_name} {}x{} ntok {ntok} bn 0 (legacy: dim 4 nc 256 does not fit the tiled kernel) relL2 {rel:.3e} maxabs {max:.3e}",
        router.rows, router.cols
    );
    if rel > 1e-5 {
        println!("  FAIL the legacy path is past 1e-5 against the f64 reference");
        *failed = true;
    }

    // A routed expert's prefill is a row slice of the stack. The slice must
    // write its own rows and nothing else, in the full-row layout.
    let gu_name = format!("{layer}.experts.gate_up_proj");
    let gu_index = encoded
        .iter()
        .position(|e| e.name == gu_name)
        .ok_or_else(|| format!("{gu_name}: not an encoded entry of this release"))?;
    let gu = encoded[gu_index];
    let matrix = Matrix::open(root, gu).map_err(|e| e.to_string())?;
    let ntok = 8usize;
    let slice_rows = 128usize; // one PRE_BM block
    let xt = transposed_probe(gu.cols, ntok);
    let xt16: Vec<f32> = xt.iter().map(|&v| round_f16(v)).collect();
    let reference = f64_prefill(&matrix, &xt16, ntok, 0, slice_rows);
    let mut y = vec![0f32; ntok * gu.rows];
    call_prefill_rows(device, gu_index, 0, slice_rows, &xt, ntok, &mut y, 128, 0)?;
    let mut got = Vec::with_capacity(ntok * slice_rows);
    for t in 0..ntok {
        got.extend_from_slice(&y[t * gu.rows..t * gu.rows + slice_rows]);
    }
    // The reference keeps the container's full-row layout; compact it the way
    // `got` was compacted before comparing.
    let mut want = Vec::with_capacity(ntok * slice_rows);
    for t in 0..ntok {
        want.extend_from_slice(&reference[t * gu.rows..t * gu.rows + slice_rows]);
    }
    let (rel, max) = compare_f32_f64(&got, &want);
    let untouched = y[slice_rows] == 0.0 && y[ntok * gu.rows - 1] == 0.0;
    println!(
        "PREFILL_SLICE {gu_name} rows 0..{slice_rows} of {} ntok {ntok} relL2 {rel:.3e} maxabs {max:.3e} rows outside the slice untouched {untouched}",
        gu.rows
    );
    if rel > 1e-5 || !untouched {
        println!("  FAIL the row-slice prefill is wrong");
        *failed = true;
    }
    Ok(())
}

/// Step 5: the layer's elementwise arithmetic, gated against the CPU
/// forward's own functions rather than against a second copy of the formula.
///
/// RMSNorm is the one that matters: it runs on every layer boundary and on
/// each attention head, with `eps` 1e-6, with and without a weight. The
/// reference is an f64 evaluation of the same definition, so the printed relL2
/// is the whole claim; the CPU function's own value is printed beside it.
fn elementwise_gate(device: &Residency, failed: &mut bool) -> Result<(), String> {
    let eps = 1e-6f32;
    // `Gemma4TextScaledWordEmbedding` multiplies by the scale downcast to the
    // checkpoint's dtype, so the embedding scale is bf16(sqrt(2816)).
    let embed_scale = bf16_round((2816f32).sqrt());
    println!(
        "ELEMENTWISE embed scale {embed_scale} (bf16 of sqrt(2816) = {})",
        (2816f32).sqrt()
    );
    if embed_scale != 53.0 {
        println!("  FAIL the embedding scale is not the bf16-rounded sqrt(hidden)");
        *failed = true;
    }

    for &(rows, width) in &[(4usize, 2816usize), (2, 256)] {
        let x = elementwise_probe(rows * width, 1.0);
        let weight = elementwise_probe(width, 1.0);
        for (tag, w) in [("weighted", Some(weight.as_slice())), ("weightless", None)] {
            let mut y = vec![0f32; rows * width];
            call_rmsnorm(device, &x, rows, width, w, &mut y, eps)?;
            let cpu: Vec<f32> = (0..rows)
                .flat_map(|r| rms_norm(&x[r * width..(r + 1) * width], w, eps))
                .collect();
            let want = f64_rmsnorm(&x, w, rows, width, eps);
            let (vs_cpu, max_cpu, _) = compare(&cpu, &y);
            let (vs_f64, max_f64) = compare_f32_f64(&y, &want);
            println!(
                "RMSNORM {tag} {rows}x{width} vs cpu {vs_cpu:.3e} (maxabs {max_cpu:.3e}) vs f64 {vs_f64:.3e} (maxabs {max_f64:.3e})"
            );
            if vs_cpu > 1e-6 || vs_f64 > 1e-6 {
                println!("  FAIL the device norm is past 1e-6");
                *failed = true;
            }
        }
    }

    // gelu(gate) * up: the dense and the expert feed-forward both write it.
    let n = 2816usize;
    let gate = elementwise_probe(n, 6.0);
    let up = elementwise_probe(n, 1.0);
    let mut act = vec![0f32; n];
    call_gelu_mul(device, &gate, &up, &mut act)?;
    let cpu: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| gelu(g) * u).collect();
    let want: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| f64_gelu(f64::from(g)) * f64::from(u))
        .collect();
    let (vs_cpu, max_cpu, _) = compare(&cpu, &act);
    let (vs_f64, max_f64) = compare_f32_f64(&act, &want);
    println!("GELU_MUL {n} vs cpu {vs_cpu:.3e} (maxabs {max_cpu:.3e}) vs f64 {vs_f64:.3e} (maxabs {max_f64:.3e})");
    if vs_cpu > 1e-6 || vs_f64 > 1e-6 {
        println!("  FAIL the activation is past 1e-6");
        *failed = true;
    }

    // The residual add and the layer scalar are f32 arithmetic on both sides,
    // so they must agree bit for bit; a difference would be a different
    // rounding, not a tolerance.
    let x = elementwise_probe(n, 1.0);
    let add = elementwise_probe(n, 1.0);
    for &scale in &[1.0f32, 1.7, 53.0] {
        let mut out = vec![0f32; n];
        call_residual_scale(device, &x, &add, &mut out, scale)?;
        let want: Vec<f32> = x.iter().zip(&add).map(|(&a, &b)| (a + b) * scale).collect();
        let differ = out
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("RESIDUAL_SCALE scale {scale} bit-identical {differ} differences of {n}");
        if differ != 0 {
            println!("  FAIL the residual add or the scale rounds differently");
            *failed = true;
        }
    }

    // The logit softcap the forward applies last.
    let logits = elementwise_probe(n, 50.0);
    let mut capped = logits.clone();
    call_softcap(device, &mut capped, 30.0)?;
    let want: Vec<f64> = logits
        .iter()
        .map(|&v| 30.0 * (f64::from(v) / 30.0).tanh())
        .collect();
    let (rel, max) = compare_f32_f64(&capped, &want);
    println!("SOFTCAP {n} cap 30 vs f64 {rel:.3e} maxabs {max:.3e}");
    if rel > 1e-6 {
        println!("  FAIL the softcap is past 1e-6");
        *failed = true;
    }
    Ok(())
}

/// A deterministic vector in [-scale, scale], distinct from the other probes.
fn elementwise_probe(n: usize, scale: f32) -> Vec<f32> {
    let mut state = 0x1f12_3456u32;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn f64_rmsnorm(x: &[f32], weight: Option<&[f32]>, rows: usize, width: usize, eps: f32) -> Vec<f64> {
    let mut out = vec![0f64; rows * width];
    for r in 0..rows {
        let row = &x[r * width..(r + 1) * width];
        let mean = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / width as f64
            + f64::from(eps);
        let inv = 1.0 / mean.sqrt();
        for i in 0..width {
            let w = weight.map_or(1.0, |w| f64::from(w[i]));
            out[r * width + i] = f64::from(row[i]) * inv * w;
        }
    }
    out
}

/// `nn.functional.gelu(x, approximate="tanh")` in f64.
fn f64_gelu(x: f64) -> f64 {
    const K: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    0.5 * x * (1.0 + (K * (x + 0.044_715 * x * x * x)).tanh())
}

#[allow(clippy::too_many_arguments)]
fn call_rmsnorm(
    device: &Residency,
    x: &[f32],
    rows: usize,
    width: usize,
    weight: Option<&[f32]>,
    y: &mut [f32],
    eps: f32,
) -> Result<(), String> {
    let mut err = [0i8; 512];
    let rc = unsafe {
        gm4_gpu_rmsnorm(
            device.0,
            x.as_ptr(),
            rows as i32,
            width as i32,
            weight.map_or(std::ptr::null(), |w| w.as_ptr()),
            y.as_mut_ptr(),
            eps,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        return Err(format!("rmsnorm: {}", message(&err)));
    }
    Ok(())
}

fn call_gelu_mul(
    device: &Residency,
    gate: &[f32],
    up: &[f32],
    act: &mut [f32],
) -> Result<(), String> {
    let mut err = [0i8; 512];
    let rc = unsafe {
        gm4_gpu_gelu_mul(
            device.0,
            gate.as_ptr(),
            up.as_ptr(),
            act.as_mut_ptr(),
            gate.len(),
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        return Err(format!("gelu_mul: {}", message(&err)));
    }
    Ok(())
}

fn call_residual_scale(
    device: &Residency,
    x: &[f32],
    add: &[f32],
    out: &mut [f32],
    scale: f32,
) -> Result<(), String> {
    let mut err = [0i8; 512];
    let rc = unsafe {
        gm4_gpu_residual_scale(
            device.0,
            x.as_ptr(),
            add.as_ptr(),
            out.as_mut_ptr(),
            x.len(),
            scale,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        return Err(format!("residual_scale: {}", message(&err)));
    }
    Ok(())
}

fn call_softcap(device: &Residency, y: &mut [f32], cap: f32) -> Result<(), String> {
    let mut err = [0i8; 512];
    let rc = unsafe {
        gm4_gpu_softcap(
            device.0,
            y.as_mut_ptr(),
            y.len(),
            cap,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        return Err(format!("softcap: {}", message(&err)));
    }
    Ok(())
}

/// The transposed activation matrix the prefill takes: [col][token], with a
/// distinct value per (col, token) so a transposition cannot pass unnoticed.
fn transposed_probe(cols: usize, ntok: usize) -> Vec<f32> {
    let mut state = 0x2545_f491u32;
    let mut xt = vec![0f32; cols * ntok];
    for slot in xt.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *slot = ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0;
    }
    xt
}

/// f32 -> f16 -> f32 with round-to-nearest-even, the same rounding
/// `__floats2half2_rn` applies in the kernel.
fn round_f16(v: f32) -> f32 {
    f16_to_f32(to_f16_bits(v))
}

/// The f16 bit pattern of a f32 value, round to nearest even. The crate's
/// `f16_to_f32` is the tested decoder on the other side of the round trip.
fn to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        let nan = if man == 0 { 0 } else { 0x0200 };
        return sign | 0x7C00 | nan;
    }
    if exp == 0 {
        return sign; // f32 subnormals are below half of f16's smallest subnormal
    }
    let e = exp - 127;
    if e > 15 {
        return sign | 0x7C00;
    }
    if e >= -14 {
        let mut m = man >> 13;
        let lost = man & 0x1FFF;
        if lost > 0x1000 || (lost == 0x1000 && (m & 1) == 1) {
            m += 1;
        }
        let mut ee = e + 15;
        if m == 0x400 {
            m = 0;
            ee += 1;
        }
        if ee >= 0x1F {
            return sign | 0x7C00;
        }
        return sign | ((ee as u16) << 10) | (m as u16);
    }
    // f16 subnormal: the quantum is 2^-24, so the value is m * 2^(e - 23)
    // and m * 2^(e + 1) counts quanta.
    let m = 0x80_0000u64 | u64::from(man);
    let shift = (-(e + 1)) as u32;
    if shift > 24 {
        return sign;
    }
    let q = m >> shift;
    let rem = m & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let mut q = q;
    if rem > half || (rem == half && (q & 1) == 1) {
        q += 1;
    }
    if q >= 0x400 {
        return sign | 0x0400;
    }
    sign | q as u16
}

/// The f64 dot of the named rows against the transposed activations, over an
/// f64 accumulator: the reference the tensor-core path approximates.
fn f64_prefill(matrix: &Matrix, xt: &[f32], ntok: usize, row0: usize, nrows: usize) -> Vec<f64> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut out = vec![0f64; ntok * rows];
    for r in 0..nrows {
        let values = matrix.row(row0 + r).expect("row is inside the matrix");
        for t in 0..ntok {
            let mut sum = 0f64;
            for c in 0..cols {
                sum += f64::from(values[c]) * f64::from(xt[c * ntok + t]);
            }
            out[t * rows + row0 + r] = sum;
        }
    }
    out
}

fn compare_f32_f64(values: &[f32], reference: &[f64]) -> (f64, f64) {
    let mut diff = 0f64;
    let mut norm = 0f64;
    let mut max_abs = 0f64;
    for (&v, &r) in values.iter().zip(reference) {
        let d = f64::from(v) - r;
        diff += d * d;
        norm += r * r;
        max_abs = max_abs.max(d.abs());
    }
    let rel = if norm > 0.0 {
        (diff / norm).sqrt()
    } else {
        0.0
    };
    (rel, max_abs)
}

fn call_prefill(
    device: &Residency,
    tensor: usize,
    xt: &[f32],
    ntok: usize,
    y: &mut [f32],
    bn: i32,
    cb: i32,
) -> Result<(), String> {
    let mut err = [0i8; 512];
    let rc = unsafe {
        gm4_gpu_prefill(
            device.0,
            tensor as i32,
            xt.as_ptr(),
            ntok as i32,
            y.as_mut_ptr(),
            bn,
            cb,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        return Err(format!("prefill: {}", message(&err)));
    }
    Ok(())
}

fn call_prefill_rows(
    device: &Residency,
    tensor: usize,
    row0: usize,
    nrows: usize,
    xt: &[f32],
    ntok: usize,
    y: &mut [f32],
    bn: i32,
    cb: i32,
) -> Result<(), String> {
    let mut err = [0i8; 512];
    let rc = unsafe {
        gm4_gpu_prefill_rows(
            device.0,
            tensor as i32,
            row0 as i32,
            nrows as i32,
            xt.as_ptr(),
            ntok as i32,
            y.as_mut_ptr(),
            bn,
            cb,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        return Err(format!("prefill_rows: {}", message(&err)));
    }
    Ok(())
}

fn call_matvec(
    device: &Residency,
    tensor: usize,
    x: &[f32],
    y: &mut [f32],
    threads: i32,
) -> Result<(), String> {
    let mut err = [0i8; 512];
    let rc = unsafe {
        gm4_gpu_matvec(
            device.0,
            tensor as i32,
            x.as_ptr(),
            y.as_mut_ptr(),
            threads,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        return Err(format!("matvec: {}", message(&err)));
    }
    Ok(())
}

fn call_rows(
    device: &Residency,
    tensor: usize,
    rows: &[u32],
    x: &[f32],
    y: &mut [f32],
    threads: i32,
) -> Result<(), String> {
    let mut err = [0i8; 512];
    let rc = unsafe {
        gm4_gpu_rows(
            device.0,
            tensor as i32,
            rows.as_ptr(),
            rows.len() as i32,
            x.as_ptr(),
            y.as_mut_ptr(),
            threads,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        return Err(format!("rows: {}", message(&err)));
    }
    Ok(())
}

fn message(buf: &[i8]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// A deterministic activation vector, so two runs and the two sides compare
/// the same input. `sum x^2` is printed nowhere on purpose: the probe is
/// stated here and needs no checksum to reproduce.
fn probe_input(cols: usize) -> Vec<f32> {
    let mut x = Vec::with_capacity(cols);
    let mut state = 0x9e37_79b9u32;
    for _ in 0..cols {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        x.push(((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0);
    }
    x
}

/// relL2 against the CPU vector, the largest absolute difference, and the
/// CPU vector's own scale for context.
fn compare(cpu: &[f32], gpu: &[f32]) -> (f64, f64, f64) {
    let mut diff = 0f64;
    let mut norm = 0f64;
    let mut max_abs = 0f64;
    let mut scale = 0f64;
    for (&c, &g) in cpu.iter().zip(gpu) {
        let d = f64::from(g) - f64::from(c);
        diff += d * d;
        norm += f64::from(c) * f64::from(c);
        max_abs = max_abs.max(d.abs());
        scale = scale.max(f64::from(c).abs());
    }
    let rel = if norm > 0.0 {
        (diff / norm).sqrt()
    } else {
        0.0
    };
    (rel, max_abs, scale)
}

/// The decoded tensor dotted with `x` in f64: the reference both f32 paths
/// approximate, computed from the same weights the artifact stores.
fn f64_dots(tensor: &Tensor, x: &[f32]) -> Vec<f64> {
    let mut out = vec![0f64; tensor.rows];
    for (row, slot) in out.iter_mut().enumerate() {
        let values = &tensor.data[row * tensor.cols..(row + 1) * tensor.cols];
        let mut sum = 0f64;
        for (&w, &v) in values.iter().zip(x) {
            sum += f64::from(w) * f64::from(v);
        }
        *slot = sum;
    }
    out
}

/// relL2 and maxabs of a f32 vector against a f64 reference.
fn compare_f64(values: &[f32], reference: &[f64]) -> (f64, f64, f64) {
    let mut diff = 0f64;
    let mut norm = 0f64;
    let mut max_abs = 0f64;
    for (&v, &r) in values.iter().zip(reference) {
        let d = f64::from(v) - r;
        diff += d * d;
        norm += r * r;
        max_abs = max_abs.max(d.abs());
    }
    let rel = if norm > 0.0 {
        (diff / norm).sqrt()
    } else {
        0.0
    };
    (rel, max_abs, 0.0)
}
