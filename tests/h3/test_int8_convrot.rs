//! The release's own INT8 convrot weights, reproduced from the BF16 sibling.
//!
//! The reference is `comfy-kitchen` 0.2.26 (`/data/refs/comfy-kitchen-v0.2.26`),
//! the quantizer the release's recipe runs offline; the two files it produced
//! are the artifact under test:
//!
//! - `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, the source, 686 tensors;
//! - `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`, the same
//!   model with 328 linears carried as I8 `[out, in]` + F32 `[out, 1]` rows.
//!
//! This is the port's external check: it reproduces the release's own bytes
//! from the release's own source weights, so a misread Hadamard, a wrong
//! rotation direction or a wrong rounding rule shows up as a mismatch against
//! the artifact rather than as agreement with our own transcription. Two
//! checks per tensor:
//!
//! 1. reproduce - rotate the BF16 weight in groups of 256 by the regular
//!    Hadamard and quantize it row-wise (`_rotate_weight`,
//!    `tensor/int8_utils.py:42-52`; `quantize_int8_rowwise`,
//!    `backends/eager/quantization.py:857-871`). The F32 row scale and the I8
//!    bytes are compared with the checkpoint's.
//! 2. round trip - dequantize the checkpoint's INT8 to F32, un-rotate by the
//!    same matrix (`dequantize_int8_convrot_weight`, `:909-911`; the matrix is
//!    symmetric, so `H H = I`), and compare with the BF16 source. The residual
//!    must sit at the INT8 quantum; a wrong matrix leaves the two uncorrelated,
//!    which is a residual near 1.0 rather than near 0.01.
//!
//! Model-free and host-only. The files are 15.4 GB and 26.3 GB, so they are
//! read over the mount one tensor at a time; only the selected tensors' bytes
//! move. `H3_GEMMA_DIR` names the directory and the first argument overrides
//! it. Exit 3 means the artifact is not reachable, which `tests/h3-suite`
//! treats as a skip.
//!
//! Measured against the release's own pair: the convention that reproduces the
//! checkpoint is the F32 one - the rotation's F32 accumulation is quantized
//! without a BF16 rounding, and the divisor is the F32 scale - under which the
//! checkpoint's INT8 equals this gate's on every element but a handful (a
//! rotation-order flip at a `.5` boundary: 0-6 per tensor in the millions) and
//! its F32 scales to within an F32 ulp. The BF16-operand convention, where
//! `_int8_scale_for_math`'s dtype cast makes the divisor and the quotient BF16,
//! is what a *runtime* activation path uses because a served activation is
//! BF16; the offline weight conversion ran with an F32 operand, which is the
//! only way the two `v_proj` rows below can match to the last bit.

use std::path::PathBuf;
use std::process::ExitCode;
use std::thread;

use ds4_core::minimax_h3::safetensors::{read_shard, Dtype, Shard};

/// The ConvRot group size every `*.comfy_quant` blob in the checkpoint states
/// (`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`).
const GROUP: usize = 256;
/// The int8 quantization maximum, `backends/eager/quantization.py:871`.
const INT8_MAX: f32 = 127.0;
/// `clamp(min=1e-30)` on the row scale, and the dtype floor below it.
const MIN_SCALE: f32 = 1e-30;
/// `torch.finfo(torch.bfloat16).tiny`, `_int8_scale_for_math` (`:815-817`).
const TINY_BF16: f32 = 1.175_494_4e-38;

const SOURCE_NAME: &str = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors";
const INT8_NAME: &str = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors";
const DEFAULT_DIR: &str =
    "/run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/mnt/h3-artifacts/Gemma4";

/// The 48 layers carry seven quantized linears each, except that eight layers
/// have no `v_proj` (328 = 48 * 7 - 8). The gate runs two layers: the first and
/// the last, so the tensor set spans every shape and both ends of the tower.
const LAYERS: [usize; 2] = [0, 47];
const SUFFIXES: [&str; 7] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// One candidate convention for the release's own quantizer. The reference
/// source pins the arithmetic up to two placeholders the artifact has to
/// decide: whether the rotated value is rounded to the operand's dtype before
/// the quantization (`_rotate_weight` runs a BF16 matmul, but a converter that
/// rotates an F32 model keeps the F32 result), and whether the divisor and the
/// stored scale are the same number. Each variant is measured against the
/// checkpoint rather than assumed.
struct Variant {
    name: &'static str,
    /// Round the rotation's F32 accumulation to BF16 (the operand's dtype).
    bf16_rotation: bool,
    /// Divide by the BF16 rounding of the F32 scale, as
    /// `_int8_scale_for_math` (`:815-817`) does before the division.
    bf16_divisor: bool,
    /// Store the BF16 rounding of the F32 scale, which is the value the
    /// division actually used.
    bf16_scale: bool,
}

const VARIANTS: [Variant; 4] = [
    Variant {
        name: "rot bf16, div bf16, scale f32",
        bf16_rotation: true,
        bf16_divisor: true,
        bf16_scale: false,
    },
    Variant {
        name: "rot bf16, div bf16, scale bf16",
        bf16_rotation: true,
        bf16_divisor: true,
        bf16_scale: true,
    },
    Variant {
        name: "rot bf16, div f32,  scale f32",
        bf16_rotation: true,
        bf16_divisor: false,
        bf16_scale: false,
    },
    Variant {
        name: "rot f32,  div f32,  scale f32",
        bf16_rotation: false,
        bf16_divisor: false,
        bf16_scale: false,
    },
];

/// One variant's verdict on one tensor.
struct Verdict {
    values_off: usize,
    worst: i32,
    scale_off: usize,
    scale_worst: f64,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(Failure::Missing(detail)) => {
            println!("skip: {detail}");
            ExitCode::from(3)
        }
        Err(Failure::Broken(detail)) => {
            eprintln!("FAIL tests/h3/test_int8_convrot.rs: {detail}");
            ExitCode::from(1)
        }
    }
}

enum Failure {
    Missing(String),
    Broken(String),
}

fn run() -> Result<bool, Failure> {
    let directory = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("H3_GEMMA_DIR").ok())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR));
    let source_path = directory.join(SOURCE_NAME);
    let int8_path = directory.join(INT8_NAME);
    for path in [&source_path, &int8_path] {
        if !path.exists() {
            return Err(Failure::Missing(format!("{} is not there", path.display())));
        }
    }

    let source = read_shard(&source_path)
        .map_err(|error| Failure::Broken(format!("{}: {error}", source_path.display())))?;
    let int8 = read_shard(&int8_path)
        .map_err(|error| Failure::Broken(format!("{}: {error}", int8_path.display())))?;
    println!(
        "source {} tensors {}, int8 {} tensors {}",
        source.tensors.len(),
        source_path.display(),
        int8.tensors.len(),
        int8_path.display()
    );

    let signs = hadamard(GROUP);
    let mut passed = true;
    for layer in LAYERS {
        for suffix in SUFFIXES {
            let name = format!("model.layers.{layer}.{suffix}");
            match one_tensor(&source, &int8, &name, &signs)? {
                Some(ok) => passed &= ok,
                None => println!("{name}: no INT8 counterpart, skipped"),
            }
        }
    }
    println!(
        "{}",
        if passed {
            "INT8-CONVROT-ARTIFACT-OK"
        } else {
            "INT8-CONVROT-ARTIFACT-OFF"
        }
    );
    Ok(passed)
}

/// One linear: the three comparisons the gate is made of.
fn one_tensor(
    source: &Shard,
    int8: &Shard,
    name: &str,
    signs: &[i8],
) -> Result<Option<bool>, Failure> {
    let Some(source_tensor) = source.find(name) else {
        return Ok(None);
    };
    let Some(quantized) = int8.find(name) else {
        return Ok(None);
    };
    /* The checkpoint's own naming: `...weight` carries `...weight_scale`. */
    let scale_name = format!("{name}_scale");
    let Some(scale_tensor) = int8.find(&scale_name) else {
        return Err(Failure::Broken(format!("{name}: no {scale_name}")));
    };
    let shape = source_tensor.shape.clone();
    if shape.len() != 2 || quantized.shape != shape || scale_tensor.shape != vec![shape[0], 1] {
        return Err(Failure::Broken(format!(
            "{name}: shapes {} / {} / {} do not agree",
            shape.len(),
            quantized.shape.len(),
            scale_tensor.shape.len()
        )));
    }
    let (rows, columns) = (shape[0] as usize, shape[1] as usize);
    if columns % GROUP != 0 {
        return Err(Failure::Broken(format!(
            "{name}: {columns} columns is not a multiple of {GROUP}"
        )));
    }

    let source_f32 = read_bf16_as_f32(source, name)?;
    let checkpoint_i8 = read_i8(int8, name)?;
    let checkpoint_scale = read_f32(int8, &scale_name)?;

    /* 1. reproduce: the release's own two steps from its own source weight,
     * under each candidate convention. */
    let round_rotation = rotate(&source_f32, columns, signs, GROUP, true);
    let exact_rotation = rotate(&source_f32, columns, signs, GROUP, false);
    let mut best: Option<(&Variant, Verdict, Vec<f32>, Vec<i8>)> = None;
    let mut lines = Vec::new();
    for variant in &VARIANTS {
        let rotated = if variant.bf16_rotation {
            &round_rotation
        } else {
            &exact_rotation
        };
        let (want_scale, want_i8) =
            quantize_rows(rotated, columns, variant.bf16_divisor, variant.bf16_scale);
        let verdict = compare(&checkpoint_i8, &checkpoint_scale, &want_i8, &want_scale);
        lines.push(format!(
            "    {}: int8 exact {:.5} ({} off, worst |d| {}); scale bits off {}/{} \
             (worst rel {:.2e})",
            variant.name,
            1.0 - verdict.values_off as f64 / checkpoint_i8.len() as f64,
            verdict.values_off,
            verdict.worst,
            verdict.scale_off,
            rows,
            verdict.scale_worst
        ));
        let better = match &best {
            None => true,
            Some((_, previous, _, _)) => {
                (verdict.values_off, verdict.scale_off) < (previous.values_off, previous.scale_off)
            }
        };
        if better {
            best = Some((variant, verdict, want_scale, want_i8));
        }
    }
    let (winner, verdict, want_scale, want_i8) = best.expect("a variant ran");
    for line in &lines {
        println!("{line}");
    }

    /* 2. round trip: dequantize the checkpoint and un-rotate it. */
    let dequantized: Vec<f32> = checkpoint_i8
        .iter()
        .enumerate()
        .map(|(at, value)| f32::from(*value) * checkpoint_scale[at / columns])
        .collect();
    let unrotated = rotate(&dequantized, columns, signs, GROUP, true);
    let checkpoint_residual = relative_rms(&unrotated, &source_f32);

    /* The same round trip on the values this gate produced, so a residual near
     * the quantum cannot be the checkpoint's alone. */
    let ours: Vec<f32> = want_i8
        .iter()
        .enumerate()
        .map(|(at, value)| f32::from(*value) * want_scale[at / columns])
        .collect();
    let ours_residual = relative_rms(&rotate(&ours, columns, signs, GROUP, true), &source_f32);

    let total = rows * columns;
    let value_fraction = 1.0 - verdict.values_off as f64 / total as f64;
    println!(
        "{name} [{rows}, {columns}]: best is `{}`; int8 exact {value_fraction:.6} \
         ({} off, worst |d| {}), scale bits off {}/{}; round trip rel RMS checkpoint \
         {checkpoint_residual:.4e}, ours {ours_residual:.4e}",
        winner.name, verdict.values_off, verdict.worst, verdict.scale_off, rows,
    );

    /* The gate's criterion, set from what the artifact can actually pin: the
     * rotation's summation order is the reference's cuBLAS order, which the
     * artifact does not record, so the last F32 bit of a rotated value - and
     * therefore the odd int8 rounding at a .5 boundary and the odd scale bit -
     * is not reproducible. What IS reproducible is everything else: a wrong
     * Hadamard, a wrong dtype rule or a wrong rounding rule misses by orders of
     * magnitude, not by a few elements. */
    let ok = value_fraction >= 0.99999
        && verdict.worst <= 1
        && verdict.scale_worst < 1e-6
        && checkpoint_residual < 0.02;
    Ok(Some(ok))
}

/// `_build_hadamard` (`tensor/int8_utils.py:10-38`): the regular Hadamard built
/// from `h4` by repeated Kronecker product, returned as its signs (the port
/// keeps the `1/sqrt(size)` factor as one power-of-two scale instead).
fn hadamard(size: usize) -> Vec<i8> {
    const H4: [[i8; 4]; 4] = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]];
    assert!(
        size >= 4 && (size as f64).log(4.0).fract() == 0.0,
        "the regular Hadamard needs a power of 4"
    );

    let mut matrix = vec![0i8; 16];
    for (row, entry) in matrix.iter_mut().enumerate() {
        *entry = H4[row / 4][row % 4];
    }
    let mut current = 4;
    while current < size {
        let side = current * 4;
        let mut next = vec![0i8; side * side];
        for row in 0..side {
            for column in 0..side {
                next[row * side + column] =
                    matrix[(row / 4) * current + (column / 4)] * H4[row % 4][column % 4];
            }
        }
        matrix = next;
        current = side;
    }
    matrix
}

/// `_rotate_weight`/`_rotate_activation` (`:42-70`) in one pass: the last
/// dimension is reshaped into groups of `size` and multiplied by the Hadamard
/// in the operand's own dtype, so the F32 accumulation is rounded to BF16 once
/// per output when the operand is BF16 (`round_bf16`). The sum runs in index
/// order; the reference's cuBLAS tile order is not pinned, which the report's
/// exact fraction measures.
fn rotate(input: &[f32], columns: usize, signs: &[i8], group: usize, round_bf16: bool) -> Vec<f32> {
    assert_eq!(signs.len(), group * group, "the sign matrix is square");
    assert!(columns % group == 0);
    let rows = input.len() / columns;
    let groups = columns / group;
    let recip = 1.0f32 / (group as f32).sqrt();
    let mut out = vec![0.0f32; input.len()];

    let workers = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
        .min(16)
        .min(rows.max(1));
    let per_worker = rows.div_ceil(workers);
    thread::scope(|scope| {
        for (worker, out_rows) in out.chunks_mut(per_worker * columns).enumerate() {
            let first_row = worker * per_worker;
            scope.spawn(move || {
                for (offset, out_line) in out_rows.chunks_mut(columns).enumerate() {
                    let base = (first_row + offset) * columns;
                    for index in 0..groups {
                        let in_line = &input[base + index * group..base + (index + 1) * group];
                        let out_group = &mut out_line[index * group..(index + 1) * group];
                        for column in 0..group {
                            let mut sum = 0.0f32;
                            for i in 0..group {
                                let value = in_line[i];
                                sum += if signs[i * group + column] > 0 {
                                    value
                                } else {
                                    -value
                                };
                            }
                            out_group[column] = if round_bf16 {
                                bf16_round(sum * recip)
                            } else {
                                sum * recip
                            };
                        }
                    }
                }
            });
        }
    });
    out
}

/// `quantize_int8_rowwise` (`:857-871`) with `_int8_scale_for_math` (`:815-817`)
/// and `_round_int8` (`:820-826`): the row's BF16 absmax, the F32 scale
/// `clamp(absmax / 127, min=1e-30)`, the divisor, the quotient in the
/// operand's dtype, `round_` (half to even) and the clamp.
fn quantize_rows(
    values: &[f32],
    columns: usize,
    bf16_divisor: bool,
    bf16_scale: bool,
) -> (Vec<f32>, Vec<i8>) {
    let rows = values.len() / columns;
    let mut scales = Vec::with_capacity(rows);
    let mut quantized = vec![0i8; values.len()];

    for row in 0..rows {
        let line = &values[row * columns..(row + 1) * columns];
        let mut amax = 0.0f32;
        for value in line {
            amax = amax.max(value.abs());
        }
        let scale = (amax / INT8_MAX).max(MIN_SCALE);
        scales.push(if bf16_scale { bf16_round(scale) } else { scale });

        let mut divisor = if bf16_divisor {
            bf16_round(scale)
        } else {
            scale
        };
        if divisor == 0.0 {
            divisor = if bf16_divisor {
                bf16_round(TINY_BF16)
            } else {
                TINY_BF16
            };
        }
        for (at, value) in line.iter().enumerate() {
            /* `x / scale.to(x.dtype)`, then the same BF16 rounding torch's
             * elementwise divide makes of an F32 quotient. */
            let quotient = if bf16_divisor {
                bf16_round(value / divisor)
            } else {
                value / divisor
            };
            let rounded = round_half_to_even(quotient);
            quantized[row * columns + at] = rounded.clamp(-128.0, INT8_MAX) as i8;
        }
    }
    (scales, quantized)
}

/// One variant's INT8 bytes and F32 scales against the checkpoint's.
fn compare(
    checkpoint_i8: &[i8],
    checkpoint_scale: &[f32],
    want_i8: &[i8],
    want_scale: &[f32],
) -> Verdict {
    let mut verdict = Verdict {
        values_off: 0,
        worst: 0,
        scale_off: 0,
        scale_worst: 0.0,
    };
    for (got, want) in checkpoint_i8.iter().zip(want_i8.iter()) {
        let delta = i32::from(*got) - i32::from(*want);
        if delta != 0 {
            verdict.values_off += 1;
            verdict.worst = verdict.worst.max(delta.abs());
        }
    }
    for (got, want) in checkpoint_scale.iter().zip(want_scale.iter()) {
        if got.to_bits() != want.to_bits() {
            verdict.scale_off += 1;
            verdict.scale_worst = verdict
                .scale_worst
                .max(f64::from((got - want).abs()) / f64::from(want.abs().max(f32::MIN_POSITIVE)));
        }
    }
    verdict
}

/// `torch.round`, half to even (`:824`).
fn round_half_to_even(value: f32) -> f32 {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    rounded
}

/// Round an F32 to the nearest BF16 value, ties to even, as `__float2bfloat16`
/// and torch's `.to(torch.bfloat16)` do.
fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    if value.is_nan() {
        return value;
    }
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

fn relative_rms(got: &[f32], want: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    let mut diff = 0.0f64;
    for (left, right) in got.iter().zip(want.iter()) {
        let (left, right) = (f64::from(*left), f64::from(*right));
        sum += right * right;
        diff += (left - right) * (left - right);
    }
    (diff / sum).sqrt()
}

fn read_bf16_as_f32(shard: &Shard, name: &str) -> Result<Vec<f32>, Failure> {
    let tensor = shard.find(name).expect("the caller found it");
    if tensor.dtype != Dtype::Bf16 {
        return Err(Failure::Broken(format!("{name}: not BF16")));
    }
    let mut bytes = vec![0u8; tensor.bytes() as usize];
    shard
        .read_data(tensor, &mut bytes)
        .map_err(|error| Failure::Broken(format!("{name}: {error}")))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect())
}

fn read_i8(shard: &Shard, name: &str) -> Result<Vec<i8>, Failure> {
    let tensor = shard.find(name).expect("the caller found it");
    let mut bytes = vec![0u8; tensor.bytes() as usize];
    shard
        .read_data(tensor, &mut bytes)
        .map_err(|error| Failure::Broken(format!("{name}: {error}")))?;
    Ok(bytes.into_iter().map(|byte| byte as i8).collect())
}

fn read_f32(shard: &Shard, name: &str) -> Result<Vec<f32>, Failure> {
    let tensor = shard.find(name).expect("the caller found it");
    let mut bytes = vec![0u8; tensor.bytes() as usize];
    shard
        .read_data(tensor, &mut bytes)
        .map_err(|error| Failure::Broken(format!("{name}: {error}")))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}
