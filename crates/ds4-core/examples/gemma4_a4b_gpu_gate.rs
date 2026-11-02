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

use std::ffi::{c_char, c_void, CString};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::gemma4_a4b::{decode, Entry, Family, Matrix, Tensor};

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
  --timing N         best of N matvecs per tensor, plus the effective GB/s
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
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let dir: PathBuf = dir.ok_or_else(|| format!("--dir is required\n\n{}", usage()))?.into();
    let family = Family::open(&dir).map_err(|e| e.to_string())?;
    let root = family.dir.clone();
    if wanted.is_empty() {
        wanted.push(DEFAULT_Q.to_string());
        wanted.push(DEFAULT_ROUTER.to_string());
    }

    // The release's entry list is the residency's own inventory: only encoded
    // containers are uploaded, and each is named with the shape the release
    // records, which the device header check repeats.
    let encoded: Vec<&Entry> = family.release.entries().iter().filter(|e| e.encoded).collect();
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
            return Err(format!("{target}: CPU matrix shape disagrees with the release"));
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
    if failed {
        return Err("a kernel is past the gate".into());
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

fn message(buf: &[i8]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
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
    let rel = if norm > 0.0 { (diff / norm).sqrt() } else { 0.0 };
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
    let rel = if norm > 0.0 { (diff / norm).sqrt() } else { 0.0 };
    (rel, max_abs, 0.0)
}
