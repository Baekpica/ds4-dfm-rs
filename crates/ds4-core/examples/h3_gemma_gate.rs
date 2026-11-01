//! The unit-2b gate: the Gemma4 text tower against a host transcription of
//! the same forward.
//!
//! Usage: h3_gemma_gate <checkpoint> [--layers N] [--rows R] [--padding P]
//!                       [--seed S] [--oracle] [--expect HEX]
//!
//! The tower is `minimax_h3::gemma`: the INT8 checkpoint's `model.*` tensors,
//! the 48-layer graph and the 49 hidden states the extractor stacks. There is
//! no Python in this campaign and the pinned encoder cannot run on either
//! host, so this gate's oracle is a second implementation of the reference's
//! forward, written in Rust beside the port and cited to
//! `modeling_gemma4.py` line by line. It shares nothing with the port but the
//! checkpoint: the kernels run on the device, the oracle runs here, and every
//! op is compared value by value.
//!
//! What the oracle does **not** reproduce, and what therefore shows up as
//! last-bit noise rather than as a mismatch: the kernels' reduction trees
//! (the oracle sums a row in index order, the kernels use a warp tree), and
//! the libm (`rsqrtf`, `tanhf`, `expf`, `powf`, `cosf`, `sinf`). The
//! primitives' own record (`gemma-tower-primitives-2026-10-29.md`) measures
//! those fractions; a wiring fault is four orders of magnitude larger.
//!
//! Token ids are synthetic when `--ids` is absent: the release's tokenizer is
//! unit 3 and no golden prompt exists yet, so the gate draws them from a
//! seeded LCG over the vocabulary. The left padding is the release's own
//! (`gemma_assets.py:162-176`): `--padding` tokens, the default `rows / 8`.
//!
//! Printed per state: index, FNV-1a over its BF16 payload, and the first and
//! last eight values, so a golden from the reference can be compared without
//! re-deriving anything. With `--oracle` each state also gets its comparison
//! against the transcription, and the gate fails on rel RMS above 1e-3.

use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::gemma::{self, Tower, GROUP, HIDDEN, INTERMEDIATE, STATES};
use ds4_core::minimax_h3::gemma_token::Tokenizer as GemmaTokenizer;
use ds4_core::minimax_h3::safetensors::{self, Dtype, Shard};
use ds4_core::minimax_h3::text_forward::fnv1a;
use ds4_sys::h3_dit::{Dtype as Device, Gpu, Tensor};

/// The row-local difference's ceiling: an order above the amplification a
/// single `rsqrtf` residue produces and an order below a wiring fault.
const TOLERANCE: f64 = 5e-2;
/// The largest run `--oracle` will compare: past this the INT8 route's row
/// scale makes one `rsqrtf` residue non-local and the comparison stops being a
/// statement about the wiring.
const ORACLE_ROWS_MAX: usize = 16;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match run(&args) {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let (positional, options) = split(args)?;

    if positional.len() != 1 {
        return Err(usage(
            "expected the checkpoint as the only positional argument",
        ));
    }

    let path = &positional[0];
    let layers = options.layers.unwrap_or(1);

    // `--prompt` runs the checkpoint's own tokenizer, which is unit 3's port,
    // and gives the tower the ids the reference's wrapper would produce:
    // 1024 rows, left-padded, BOS first.
    let (ids, mask) = match &options.prompt {
        Some(prompt) => {
            let tokenizer =
                GemmaTokenizer::open(std::path::Path::new(path)).map_err(|error| error.token())?;
            let (ids, mask) = tokenizer.tokenize(prompt);
            let padding = mask.iter().filter(|value| **value == 0).count();

            println!(
                "TOKENS width={} vocabulary={} bos={} padding={} first={:?} last={:?}",
                ids.len(),
                tokenizer.width(),
                tokenizer.bos(),
                padding,
                &ids[..8.min(ids.len())],
                &ids[ids.len().saturating_sub(8)..]
            );
            println!(
                "IDS {}",
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            );

            if options.tokens_only {
                return Ok(ExitCode::SUCCESS);
            }

            (ids, mask)
        }
        None => {
            let rows = options.rows.unwrap_or(8);
            // `--padding` is the release's left padding in front of the ids, so
            // the run's row count is `--rows` either way.
            let padding = options.padding.unwrap_or(rows / 8);

            if padding >= rows {
                return Err(usage("--padding must leave at least one token"));
            }

            let ids = if let Some(list) = &options.ids {
                list.clone()
            } else {
                synth_ids(rows - padding, options.seed.unwrap_or(1))
            };
            let ids = pad(&ids, padding);

            (ids.clone(), attention_mask(ids.len(), padding))
        }
    };

    let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;

    if options.probe {
        probe(&gpu, std::path::Path::new(path), options.rows.unwrap_or(8))?;

        return Ok(ExitCode::SUCCESS);
    }

    let start = Instant::now();
    let tower =
        Tower::open(&gpu, std::path::Path::new(path), layers).map_err(|error| error.token())?;
    let load = start.elapsed().as_secs_f64();
    let (tensors, bytes) = tower.loaded();

    let start = Instant::now();
    let states = tower
        .encode(&gpu, &ids, &mask, layers)
        .map_err(|error| error.token())?;
    let seconds = start.elapsed().as_secs_f64();

    println!(
        "TOWER layers={} rows={} padding={} tensors={} bytes={} load={:.3} run={:.3}",
        tower.layers(),
        ids.len(),
        mask.iter().filter(|value| **value == 0).count(),
        tensors,
        bytes,
        load,
        seconds
    );

    let mut payloads = Vec::with_capacity(states.len());
    let mut expect_failed = false;

    for (index, state) in states.iter().enumerate() {
        let mut bytes = vec![0u8; state.bytes()];
        gpu.read(state, &mut bytes)
            .map_err(|detail| format!("read state {index}: {detail}"))?;
        let hash = fnv1a(&bytes);

        println!(
            "STATE {index} values={} fnv={hash:016x} first={} last={}",
            bytes.len() / 2,
            words(&bytes[..16.min(bytes.len())]),
            words(&bytes[bytes.len().saturating_sub(16)..])
        );

        if let Some(wanted) = options.expect.get(index) {
            if *wanted != hash {
                println!("STATE {index} expected {wanted:016x}");
                expect_failed = true;
            }
        }

        payloads.push(to_f32(&bytes));
    }

    if options.oracle {
        // The oracle is a wiring proof and only that. Its one irreducible
        // difference from the device is the RMS norm's inverse - `rsqrtf` on
        // the device, the host's reciprocal square root here - and the INT8
        // route rounds each activation row's scale to BF16, so one such
        // residue steps a whole row's quantisation and the next layer consumes
        // that as its input. Measured: five of six layers bit-for-bit at 8
        // rows, and at 64 rows 2.9e-2 with every row adrift. Past a small run
        // the number stops being about wiring, so the gate refuses to print it.
        if ids.len() > ORACLE_ROWS_MAX {
            return Err(usage(
                "--oracle is meaningful only at a small row count; run it at \
                 --rows 8 or below, and gate a deep run on its state hashes",
            ));
        }

        let host = HostTower::open(std::path::Path::new(path), layers)?;
        let tables = DeviceTables::build(&gpu, ids.len(), &mask)?;
        let start = Instant::now();
        let wanted = host.encode(&ids, layers, &tables);
        let oracle_seconds = start.elapsed().as_secs_f64();
        let mut failed = false;

        // A state passes either bit-for-bit, or confined to whole rows.
        //
        // The second form is not a concession to sloppiness, it is the shape of
        // the one difference this pair of implementations can have. The INT8
        // route derives each activation row's scale from that row's absmax and
        // rounds the scale to BF16, so a last-bit move of one element - here
        // `rsqrtf` on the device against the host's reciprocal square root, the
        // one primitive with no host counterpart - can step a whole row's
        // quantisation and read out across that row's outputs.
        //
        // Measured both ways on this host: with the oracle's inverse written
        // `1.0/sqrt()` five of the six layers are bit-for-bit and the full
        // layer differs in exactly one row of 3,840; written `powf(-0.5)` every
        // row of every state differs, at 2.3e-2. A wiring fault moves every row
        // and by two orders more (0.35 in `gemma_tower_bm`), so the condition
        // below separates the two cases on both axes.
        for (index, (got, want)) in payloads.iter().zip(wanted.iter()).enumerate() {
            let (differ, rel_rms, worst) = compare(got, want);
            let rows = differing_rows(got, want, HIDDEN);
            println!(
                "ORACLE state {index} differ={differ} rel_rms={rel_rms:.3e} worst={worst:.3e} rows={rows:?}"
            );
            let exact = differ == 0;
            let row_local = rows.len() * 4 <= ids.len() && rel_rms <= TOLERANCE;
            failed |= !exact && !row_local;
        }

        println!("ORACLE seconds={oracle_seconds:.3} layers={layers}");

        if failed {
            return Ok(ExitCode::from(4));
        }
    }

    if expect_failed {
        return Ok(ExitCode::from(3));
    }

    println!("GEMMA-TOWER-OK");
    Ok(ExitCode::SUCCESS)
}

/// The release's left padding: `pad` leading pad tokens masked out of the
/// keys, the rest attended.
fn pad(ids: &[u32], padding: usize) -> Vec<u32> {
    let mut padded = vec![0u32; padding];
    padded.extend_from_slice(ids);

    padded
}

fn attention_mask(rows: usize, padding: usize) -> Vec<u32> {
    let mut mask = vec![0u32; padding];
    mask.resize(rows, 1);

    mask
}

/// A seeded LCG over the vocabulary, so a run is reproducible without the
/// tokenizer. The release's own ids arrive with the golden.
fn synth_ids(rows: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;
    let mut ids = Vec::with_capacity(rows);

    for _ in 0..rows {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ids.push(state % gemma::VOCABULARY as u32);
    }

    ids
}

fn words(bytes: &[u8]) -> String {
    bytes
        .chunks(2)
        .map(|pair| format!("{:02x}{:02x}", pair[1], pair[0]))
        .collect::<Vec<_>>()
        .join(".")
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

/// Count the values that differ, and the relative RMS against the reference's
/// own magnitude - the same pair every gate in this family reports.
fn compare(got: &[f32], want: &[f32]) -> (usize, f64, f64) {
    let mut differ = 0usize;
    let mut sum = 0.0f64;
    let mut sum_diff = 0.0f64;
    let mut worst = 0.0f64;

    for (left, right) in got.iter().zip(want.iter()) {
        let delta = f64::from(*left) - f64::from(*right);
        if *left != *right {
            differ += 1;
        }
        sum += f64::from(*right) * f64::from(*right);
        sum_diff += delta * delta;
        worst = worst.max(delta.abs());
    }

    let rel = if sum > 0.0 {
        (sum_diff / sum).sqrt()
    } else {
        0.0
    };

    (differ, rel, worst)
}

/// Which rows of a `[rows, width]` state differ, at most eight indices, so a
/// difference that is one activation row and its downstream reads differently
/// from one that is spread across the run.
fn differing_rows(got: &[f32], want: &[f32], width: usize) -> Vec<usize> {
    let mut rows = Vec::new();

    for (at, (left, right)) in got.iter().zip(want.iter()).enumerate() {
        if *left != *right && rows.len() < 8 && !rows.contains(&(at / width)) {
            rows.push(at / width);
        }
    }

    rows
}

fn bf16(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

// ---- the host transcription ----

/// The checkpoint read again, as host vectors.
struct HostTower {
    embed_tokens: Vec<f32>,
    norm: Vec<f32>,
    layers: Vec<HostLayer>,
    signs: Vec<i8>,
}

struct HostLayer {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    pre_feedforward_norm: Vec<f32>,
    post_feedforward_norm: Vec<f32>,
    layer_scalar: f32,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    query: HostLinear,
    key: HostLinear,
    value: Option<HostLinear>,
    output: HostLinear,
    gate: HostLinear,
    up: HostLinear,
    down: HostLinear,
}

struct HostLinear {
    weight: Vec<i8>,
    scale: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
}

impl HostTower {
    fn open(path: &std::path::Path, layers: usize) -> Result<HostTower, String> {
        let shard =
            safetensors::read_shard(path).map_err(|error| format!("reader: {}", error.token()))?;

        let embed_tokens = host_bf16(
            &shard,
            "model.embed_tokens.weight",
            &[gemma::VOCABULARY, HIDDEN],
        )?;
        let norm = host_bf16(&shard, "model.norm.weight", &[HIDDEN])?;
        let mut loaded = Vec::with_capacity(layers);

        for layer in 0..layers {
            loaded.push(HostLayer::open(&shard, layer)?);
        }

        Ok(HostTower {
            embed_tokens,
            norm,
            layers: loaded,
            signs: hadamard(GROUP),
        })
    }

    /// `Gemma4TextModel.forward` (`modeling_gemma4.py:1653-1742`), step for
    /// step, plus the hook's substitution of `last_hidden_state` into the last
    /// collected state (`utils/output_capturing.py:273-278`).
    fn encode(&self, ids: &[u32], layers: usize, tables: &DeviceTables) -> Vec<Vec<f32>> {
        let rows = ids.len();
        let mut states = Vec::with_capacity(STATES);
        let scale = gemma::embed_scale();
        let mut embedding = vec![0.0f32; rows * HIDDEN];

        for row in 0..rows {
            let source = ids[row] as usize * HIDDEN;
            for column in 0..HIDDEN {
                embedding[row * HIDDEN + column] =
                    gemma::bf16_round(self.embed_tokens[source + column] * scale);
            }
        }
        states.push(embedding);

        for layer in 0..layers {
            let weights = &self.layers[layer];
            let (flavour, mask) = if gemma::is_full(layer) {
                (&tables.full, &tables.full_mask)
            } else {
                (&tables.sliding, &tables.sliding_mask)
            };

            states.push(layer_forward(
                &states[layer],
                weights,
                &self.signs,
                flavour,
                mask,
                rows,
                layer,
            ));
        }

        // The same substitution the port makes, and under the same condition:
        // only a run that covers all 48 layers ends with `model.norm`.
        if layers == gemma::LAYERS {
            let closing = rms_norm(&states[STATES - 1], rows, HIDDEN, Some(&self.norm));
            states[STATES - 1] = closing;
        }

        states
    }
}

impl HostLayer {
    fn open(shard: &Shard, layer: usize) -> Result<HostLayer, String> {
        let full = gemma::is_full(layer);
        let width = gemma::head_dim(layer);
        let kv_width = gemma::kv_heads(layer) * width;
        let name = |suffix: &str| format!("model.layers.{layer}.{suffix}");

        let linear =
            |suffix: &str, input_dim: usize, output_dim: usize| -> Result<HostLinear, String> {
                Ok(HostLinear {
                    weight: host_i8(
                        shard,
                        &name(&format!("{suffix}.weight")),
                        &[output_dim, input_dim],
                    )?,
                    scale: host_f32(
                        shard,
                        &name(&format!("{suffix}.weight_scale")),
                        &[output_dim, 1],
                    )?,
                    input_dim,
                    output_dim,
                })
            };

        Ok(HostLayer {
            input_norm: host_bf16(shard, &name("input_layernorm.weight"), &[HIDDEN])?,
            post_attention_norm: host_bf16(
                shard,
                &name("post_attention_layernorm.weight"),
                &[HIDDEN],
            )?,
            pre_feedforward_norm: host_bf16(
                shard,
                &name("pre_feedforward_layernorm.weight"),
                &[HIDDEN],
            )?,
            post_feedforward_norm: host_bf16(
                shard,
                &name("post_feedforward_layernorm.weight"),
                &[HIDDEN],
            )?,
            layer_scalar: host_bf16(shard, &name("layer_scalar"), &[1])?[0],
            q_norm: host_bf16(shard, &name("self_attn.q_norm.weight"), &[width])?,
            k_norm: host_bf16(shard, &name("self_attn.k_norm.weight"), &[width])?,
            query: linear("self_attn.q_proj", HIDDEN, gemma::HEADS * width)?,
            key: linear("self_attn.k_proj", HIDDEN, kv_width)?,
            value: if full {
                None
            } else {
                Some(linear("self_attn.v_proj", HIDDEN, kv_width)?)
            },
            output: linear("self_attn.o_proj", gemma::HEADS * width, HIDDEN)?,
            gate: linear("mlp.gate_proj", HIDDEN, INTERMEDIATE)?,
            up: linear("mlp.up_proj", HIDDEN, INTERMEDIATE)?,
            down: linear("mlp.down_proj", INTERMEDIATE, HIDDEN)?,
        })
    }
}

/// One layer of `Gemma4TextDecoderLayer.forward` (`:1405-1463`).
#[allow(clippy::too_many_arguments)]
fn layer_forward(
    input: &[f32],
    weights: &HostLayer,
    signs: &[i8],
    tables: &Tables,
    mask: &[f32],
    rows: usize,
    layer: usize,
) -> Vec<f32> {
    let width = gemma::head_dim(layer);
    let heads = gemma::HEADS;
    let kv_heads = gemma::kv_heads(layer);

    let normed = rms_norm(input, rows, HIDDEN, Some(&weights.input_norm));

    // `q = rope(q_norm(q_proj(x)))`, `k = rope(k_norm(k_proj(x)))`, and V the
    // raw projection through the weightless norm (`:1249-1271`). `q_norm` and
    // `k_norm` are RMS norms over the head, so their rows are tokens x heads.
    let query_raw = host_linear(&normed, rows, &weights.query, signs);
    let query = rms_norm(&query_raw, rows * heads, width, Some(&weights.q_norm));

    let key_raw = host_linear(&normed, rows, &weights.key, signs);
    let key = rms_norm(&key_raw, rows * kv_heads, width, Some(&weights.k_norm));

    let value_raw = match &weights.value {
        Some(value) => host_linear(&normed, rows, value, signs),
        None => key_raw.clone(),
    };
    let value = rms_norm(&value_raw, rows * kv_heads, width, None);

    let query = rotate(&query, tables, rows, heads, width);
    let key = rotate(&key, tables, rows, kv_heads, width);
    let attention = host_attention(&query, &key, &value, mask, rows, heads, kv_heads, width);

    let projected = host_linear(&attention, rows, &weights.output, signs);
    let attention_normed = rms_norm(&projected, rows, HIDDEN, Some(&weights.post_attention_norm));
    let residual: Vec<f32> = input
        .iter()
        .zip(attention_normed.iter())
        .map(|(left, right)| gemma::bf16_round(left + right))
        .collect();

    let normed = rms_norm(&residual, rows, HIDDEN, Some(&weights.pre_feedforward_norm));
    let mut gate = host_linear(&normed, rows, &weights.gate, signs);

    for value in gate.iter_mut() {
        *value = host_gelu(*value);
    }

    let up = host_linear(&normed, rows, &weights.up, signs);
    let gated: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(left, right)| gemma::bf16_round(left * right))
        .collect();
    let feedforward = host_linear(&gated, rows, &weights.down, signs);
    let feedforward_normed = rms_norm(
        &feedforward,
        rows,
        HIDDEN,
        Some(&weights.post_feedforward_norm),
    );

    residual
        .iter()
        .zip(feedforward_normed.iter())
        .map(|(left, right)| {
            gemma::bf16_round(gemma::bf16_round(left + right) * weights.layer_scalar)
        })
        .collect()
}

/// `Gemma4RMSNorm.forward` (`:210-214`): F32 `x * pow(mean(x^2) + eps, -0.5)`,
/// the weight in F32, one cast back. `weight` absent is Gemma's `v_norm`
/// (`with_scale=False`, `:1220`).
///
/// The sum reproduces `h3d_rms_norm_kernel`'s own tree - 256 partials, each
/// striding by 256 with `fmaf`, then `sums[t] += sums[t + stride]` halving the
/// stride - rather than summing in index order. That matters more than it
/// looks: the tower's INT8 route derives each activation row's scale from that
/// row's absmax, and the scale is rounded to BF16, so a last-bit difference in
/// a row's sum can step the whole row's quantisation. The oracle and the
/// kernels agree on the tree; the only residue is `rsqrtf`, which has no host
/// counterpart.
fn rms_norm(input: &[f32], rows: usize, width: usize, weight: Option<&[f32]>) -> Vec<f32> {
    const THREADS: usize = 256;
    let mut output = vec![0.0f32; rows * width];

    for row in 0..rows {
        let source = &input[row * width..(row + 1) * width];
        let mut partial = [0.0f32; THREADS];

        for thread in 0..THREADS {
            let mut sum = 0.0f32;
            let mut column = thread;

            while column < width {
                sum = source[column].mul_add(source[column], sum);
                column += THREADS;
            }

            partial[thread] = sum;
        }

        let mut stride = THREADS / 2;

        while stride > 0 {
            for thread in 0..stride {
                partial[thread] += partial[thread + stride];
            }

            stride /= 2;
        }

        let inverse = 1.0f32 / (partial[0] / width as f32 + gemma::RMS_EPSILON).sqrt();
        for column in 0..width {
            let mut value = source[column] * inverse;
            if let Some(weight) = weight {
                value *= weight[column];
            }
            output[row * width + column] = gemma::bf16_round(value);
        }
    }

    output
}

fn host_gelu(x: f32) -> f32 {
    const BETA: f32 = 1.414_213_562_373_095_048_80 * 1.128_379_167_095_512_573_90 * 0.5;
    const KAPPA: f32 = 0.044_715;
    let cube = x * x * x;

    gemma::bf16_round(0.5 * x * (1.0 + (BETA * (x + KAPPA * cube)).tanh()))
}

fn host_linear(input: &[f32], rows: usize, linear: &HostLinear, signs: &[i8]) -> Vec<f32> {
    let group = GROUP;
    let mut rotated = vec![0.0f32; rows * linear.input_dim];
    rotate_groups(input, rows, linear.input_dim, group, signs, &mut rotated);

    let mut scales = vec![0.0f32; rows];
    let mut quantized = vec![0i8; rows * linear.input_dim];

    for row in 0..rows {
        quantize_row(
            &rotated[row * linear.input_dim..(row + 1) * linear.input_dim],
            &mut scales[row],
            &mut quantized[row * linear.input_dim..(row + 1) * linear.input_dim],
        );
    }

    let mut output = vec![0.0f32; rows * linear.output_dim];

    for row in 0..rows {
        let activation = &quantized[row * linear.input_dim..(row + 1) * linear.input_dim];

        for column in 0..linear.output_dim {
            let weight = &linear.weight[column * linear.input_dim..(column + 1) * linear.input_dim];
            let mut accumulator = 0i32;

            for (a, w) in activation.iter().zip(weight.iter()) {
                accumulator += i32::from(*a) * i32::from(*w);
            }

            output[row * linear.output_dim + column] =
                gemma::bf16_round(accumulator as f32 * (scales[row] * linear.scale[column]));
        }
    }

    output
}

/// `_rotate_activation` (`tensor/int8_utils.py:55-70`): the grouped Hadamard,
/// accumulated in F32 in index order and rounded to BF16 once.
fn rotate_groups(
    input: &[f32],
    rows: usize,
    columns: usize,
    group: usize,
    signs: &[i8],
    output: &mut [f32],
) {
    let reciprocal = 1.0f32 / (group as f32).sqrt();

    for row in 0..rows {
        let base = row * columns;

        for index in 0..columns / group {
            let at = base + index * group;

            for column in 0..group {
                let mut sum = 0.0f32;

                for source in 0..group {
                    let value = input[at + source];
                    sum += if signs[source * group + column] > 0 {
                        value
                    } else {
                        -value
                    };
                }

                output[at + column] = gemma::bf16_round(sum * reciprocal);
            }
        }
    }
}

/// `quantize_int8_rowwise` (`backends/eager/quantization.py:857-871`) with
/// `_int8_scale_for_math` (`:815-817`): the F32 scale is `amax / 127`, the
/// divisor and the quotient are BF16 because a served activation is, and the
/// rounding is half-to-even.
const INT8_MAX: f32 = 127.0;
const MIN_SCALE: f32 = 1e-30;
const TINY_BF16: f32 = 1.175_494_4e-38;

fn quantize_row(row: &[f32], scale: &mut f32, quantized: &mut [i8]) {
    let mut amax = 0.0f32;

    for value in row {
        amax = amax.max(value.abs());
    }

    *scale = (amax / INT8_MAX).max(MIN_SCALE);
    let mut divisor = gemma::bf16_round(*scale);

    if divisor == 0.0 {
        divisor = gemma::bf16_round(TINY_BF16);
    }

    for (at, value) in row.iter().enumerate() {
        let scaled = gemma::bf16_round(value / divisor);
        quantized[at] = scaled.round_ties_even().clamp(-128.0, 127.0) as i8;
    }
}

/// The RoPE pair tables (`:1164-1181`) and the two additive masks, built by
/// the **device** and read back, then handed to the oracle.
///
/// Both are already gated on their own (`tests/h3/gemma_tower_bm`: the tables
/// against a host transcription of the reference formula, the masks to the
/// value) and a host `cosf`/`sinf` does not land on the device's `cosf`/`sinf`
/// for every input, so building them twice would put a known last-bit
/// difference into the middle of a comparison that is about the layer graph.
/// Injecting them leaves the oracle comparing exactly the wiring and the
/// arithmetic it is for.
struct DeviceTables {
    sliding: Tables,
    full: Tables,
    sliding_mask: Vec<f32>,
    full_mask: Vec<f32>,
}

impl DeviceTables {
    fn build(gpu: &Gpu, rows: usize, padding: &[u32]) -> Result<DeviceTables, String> {
        let rows = rows as u32;
        let padding_dev = gpu
            .alloc(Device::U32, padding.len())
            .map_err(|detail| format!("mask padding: {detail}"))?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                padding.as_ptr().cast::<u8>(),
                std::mem::size_of_val(padding),
            )
        };
        gpu.upload(&padding_dev, bytes)
            .map_err(|detail| format!("mask padding upload: {detail}"))?;

        Ok(DeviceTables {
            sliding: Tables::from_device(
                gpu,
                rows,
                gemma::HEAD_DIM as u32,
                gemma::SLIDING_ANGLES as u32,
                gemma::SLIDING_BASE,
            )?,
            full: Tables::from_device(
                gpu,
                rows,
                gemma::GLOBAL_HEAD_DIM as u32,
                gemma::FULL_ANGLES as u32,
                gemma::FULL_BASE,
            )?,
            sliding_mask: device_mask(gpu, rows, gemma::SLIDING_WINDOW as u32, &padding_dev)?,
            full_mask: device_mask(gpu, rows, 0, &padding_dev)?,
        })
    }
}

fn device_mask(gpu: &Gpu, rows: u32, window: u32, padding: &Tensor) -> Result<Vec<f32>, String> {
    let output = gpu
        .alloc(Device::Bf16, rows as usize * rows as usize)
        .map_err(|detail| format!("mask: {detail}"))?;
    gpu.gemma_mask(&output, rows, window, Some(padding))
        .map_err(|detail| format!("mask: {detail}"))?;
    let mut bytes = vec![0u8; output.bytes()];
    gpu.read(&output, &mut bytes)
        .map_err(|detail| format!("mask read: {detail}"))?;

    Ok(to_f32(&bytes))
}

struct Tables {
    cos: Vec<f32>,
    sin: Vec<f32>,
    dim: usize,
}

impl Tables {
    /// `h3d_gemma_rope_table` at the layer flavour's own parameters, read back.
    fn from_device(
        gpu: &Gpu,
        rows: u32,
        dim: u32,
        angles: u32,
        base: f32,
    ) -> Result<Tables, String> {
        let elements = rows as usize * dim as usize;
        let cos = gpu
            .alloc(Device::Bf16, elements)
            .map_err(|detail| format!("rope table: {detail}"))?;
        let sin = gpu
            .alloc(Device::Bf16, elements)
            .map_err(|detail| format!("rope table: {detail}"))?;
        gpu.gemma_rope_table(&cos, &sin, rows, dim, angles, base)
            .map_err(|detail| format!("rope table: {detail}"))?;
        let mut cos_bytes = vec![0u8; cos.bytes()];
        let mut sin_bytes = vec![0u8; sin.bytes()];
        gpu.read(&cos, &mut cos_bytes)
            .map_err(|detail| format!("rope table read: {detail}"))?;
        gpu.read(&sin, &mut sin_bytes)
            .map_err(|detail| format!("rope table read: {detail}"))?;

        Ok(Tables {
            cos: to_f32(&cos_bytes),
            sin: to_f32(&sin_bytes),
            dim: dim as usize,
        })
    }

    fn at(&self, row: usize, dimension: usize) -> (f32, f32) {
        (
            self.cos[row * self.dim + dimension],
            self.sin[row * self.dim + dimension],
        )
    }
}

/// `apply_rotary_pos_emb` (`:793-811`) with `rotate_half` (`:786-791`), on a
/// snapshot of the row because `rotate_half` reads the two halves against each
/// other.
fn rotate(input: &[f32], tables: &Tables, rows: usize, heads: usize, width: usize) -> Vec<f32> {
    let mut output = input.to_vec();
    let half = width / 2;

    for row in 0..rows {
        for head in 0..heads {
            let base = (row * heads + head) * width;
            let snapshot = input[base..base + width].to_vec();

            for dimension in 0..width {
                let rotated = if dimension < half {
                    -snapshot[dimension + half]
                } else {
                    snapshot[dimension - half]
                };
                let (cos, sin) = tables.at(row, dimension);
                let turned = gemma::bf16_round(snapshot[dimension] * cos);
                let spin = gemma::bf16_round(rotated * sin);
                output[base + dimension] = gemma::bf16_round(turned + spin);
            }
        }
    }

    output
}

/// `eager_attention_forward` (`:824-858`) at `scaling = 1.0`, mirrored from
/// `h3d_gemma_attention_kernel` step for step rather than from the reference
/// alone, because the comparison has to hold at the last bit:
///
/// - the dot is 32 lanes striding by 32 with `fmaf`, then the warp's own
///   shuffle-down tree (`offset = 16, 8, 4, 2, 1`), which is a full sum of the
///   lanes;
/// - the dot rounds to BF16 before the mask add, and the masked sum rounds
///   again;
/// - the maximum and the sum are the block's 256-way tree;
/// - the normalised weight is `score * (1 / total)`, rounded to BF16 before
///   each `fmaf` with V, and the accumulation rounds once at the end.
///
/// The kernel's `__shfl_down_sync` semantics for an out-of-range source lane is
/// the lane's own value, which is what makes the five-step tree a plain sum;
/// the model below reproduces that.
#[allow(clippy::too_many_arguments)]
fn host_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: &[f32],
    rows: usize,
    heads: usize,
    kv_heads: usize,
    width: usize,
) -> Vec<f32> {
    const THREADS: usize = 256;
    let groups = heads / kv_heads;
    let mut output = vec![0.0f32; rows * heads * width];

    for row in 0..rows {
        for head in 0..heads {
            let source = head / groups;
            let q_row = &query[(row * heads + head) * width..(row * heads + head + 1) * width];
            let mut score = vec![0.0f32; rows];

            for k in 0..rows {
                let k_row =
                    &key[(k * kv_heads + source) * width..(k * kv_heads + source + 1) * width];
                let mut lanes = [0.0f32; 32];

                for lane in 0..32 {
                    let mut dot = 0.0f32;
                    let mut i = lane;

                    while i < width {
                        dot = q_row[i].mul_add(k_row[i], dot);
                        i += 32;
                    }

                    lanes[lane] = dot;
                }

                for offset in [16usize, 8, 4, 2, 1] {
                    let old = lanes;

                    for lane in 0..32 {
                        let partner = if lane + offset < 32 {
                            old[lane + offset]
                        } else {
                            old[lane]
                        };
                        lanes[lane] = old[lane] + partner;
                    }
                }

                let weighted = gemma::bf16_round(lanes[0] * gemma::ATTENTION_SCALE);
                score[k] = gemma::bf16_round(weighted + mask[row * rows + k]);
            }

            let mut partial = [0.0f32; THREADS];

            for thread in 0..THREADS {
                let mut maximum = f32::NEG_INFINITY;
                let mut k = thread;

                while k < rows {
                    maximum = maximum.max(score[k]);
                    k += THREADS;
                }

                partial[thread] = maximum;
            }

            tree(&mut partial, f32::max);
            let maximum = partial[0];

            for thread in 0..THREADS {
                let mut total = 0.0f32;
                let mut k = thread;

                while k < rows {
                    let weight = (score[k] - maximum).exp();
                    score[k] = weight;
                    total += weight;
                    k += THREADS;
                }

                partial[thread] = total;
            }

            tree(&mut partial, |left, right| left + right);
            let inverse = 1.0f32 / partial[0];

            for dimension in 0..width {
                let mut accumulator = 0.0f32;

                for k in 0..rows {
                    let weight = gemma::bf16_round(score[k] * inverse);
                    let v = value[(k * kv_heads + source) * width + dimension];
                    accumulator = weight.mul_add(v, accumulator);
                }

                output[(row * heads + head) * width + dimension] = gemma::bf16_round(accumulator);
            }
        }
    }

    output
}

/// `sums[t] += sums[t + stride]` with the stride halving from `len / 2`: the
/// block reduction `h3d_rms_norm_kernel` and `h3d_gemma_attention_kernel` both
/// run, with `max` or `+` as the operation.
fn tree(partial: &mut [f32], combine: impl Fn(f32, f32) -> f32) {
    let mut stride = partial.len() / 2;

    while stride > 0 {
        for thread in 0..stride {
            partial[thread] = combine(partial[thread], partial[thread + stride]);
        }

        stride /= 2;
    }
}

/// `_build_hadamard` (`tensor/int8_utils.py:10-38`): `h4`, Kronecker-multiplied
/// until the size, each step appending the new factor on the right.
fn hadamard(size: usize) -> Vec<i8> {
    const H4: [i8; 16] = [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1];
    let mut matrix: Vec<i8> = H4.to_vec();
    let mut current = 4;

    while current < size {
        let next_size = current * 4;
        let mut next = vec![0i8; next_size * next_size];

        for row in 0..next_size {
            for column in 0..next_size {
                next[row * next_size + column] =
                    matrix[(row / 4) * current + (column / 4)] * H4[(row % 4) * 4 + (column % 4)];
            }
        }

        matrix = next;
        current = next_size;
    }

    matrix
}

// ---- host tensor reads ----

fn host_read(shard: &Shard, name: &str, dtype: Dtype, shape: &[usize]) -> Result<Vec<u8>, String> {
    let tensor = shard.find(name).ok_or_else(|| format!("missing {name}"))?;
    let want: Vec<u64> = shape.iter().map(|dim| *dim as u64).collect();

    if tensor.dtype != dtype || tensor.shape != want {
        return Err(format!(
            "shape {name}: the checkpoint has {} {:?}, the oracle needs {} {:?}",
            tensor.dtype.name(),
            tensor.shape,
            dtype.name(),
            want
        ));
    }

    let mut bytes = vec![0u8; tensor.bytes() as usize];
    shard
        .read_data(tensor, &mut bytes)
        .map_err(|error| format!("reader: {}", error.token()))?;

    Ok(bytes)
}

fn host_bf16(shard: &Shard, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    Ok(host_read(shard, name, Dtype::Bf16, shape)?
        .chunks_exact(2)
        .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]])))
        .collect())
}

fn host_f32(shard: &Shard, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    Ok(host_read(shard, name, Dtype::F32, shape)?
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

fn host_i8(shard: &Shard, name: &str, shape: &[usize]) -> Result<Vec<i8>, String> {
    Ok(host_read(shard, name, Dtype::I8, shape)?
        .into_iter()
        .map(|byte| byte as i8)
        .collect())
}

// ---- the op probe ----

/// When the tower's comparison disagrees, this localizes the op: one synthetic
/// BF16 activation through one device op and the oracle's transcription of it,
/// with the real checkpoint's weights. It exists because a layer's output
/// difference says nothing about *which* step produced it.
fn probe(gpu: &Gpu, path: &std::path::Path, rows: usize) -> Result<(), String> {
    let shard =
        safetensors::read_shard(path).map_err(|error| format!("reader: {}", error.token()))?;
    let signs = hadamard(GROUP);

    // No padding here: the probe tests the ops, not the mask's padding arm.
    let padding: Vec<u32> = vec![1; rows];
    let padding_dev = gpu
        .alloc(Device::U32, rows)
        .map_err(|detail| format!("probe padding: {detail}"))?;
    let padding_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            padding.as_ptr().cast::<u8>(),
            std::mem::size_of_val(&padding[..]),
        )
    };
    gpu.upload(&padding_dev, padding_bytes)
        .map_err(|detail| format!("probe padding upload: {detail}"))?;

    // The RMS norm probe's own activation, BF16 with occasional outliers as a
    // real row has: the row's one absmax decides the whole row's INT8 scale.
    let mut state = 12345u32;
    let (activation, activation_bytes) = synthetic(rows * HIDDEN, &mut state);
    let input = upload_f32_bf16(gpu, &activation_bytes)?;

    // 1. the INT8 convrot linear, at every shape the two layer flavours use:
    // the sliding layer's three square-ish projections and the full layer's
    // three, whose output widths are 512 and 8192.
    for (name, tensor, input_dim, output_dim) in [
        (
            "layer0 q_proj",
            "model.layers.0.self_attn.q_proj",
            HIDDEN,
            4096,
        ),
        (
            "layer5 q_proj",
            "model.layers.5.self_attn.q_proj",
            HIDDEN,
            8192,
        ),
        (
            "layer5 k_proj",
            "model.layers.5.self_attn.k_proj",
            HIDDEN,
            512,
        ),
        (
            "layer5 o_proj",
            "model.layers.5.self_attn.o_proj",
            8192,
            HIDDEN,
        ),
    ] {
        // Each shape gets its own activation: `o_proj` reads 8,192 wide.
        let (activation, activation_bytes) = synthetic(rows * input_dim, &mut state);
        let input = upload_f32_bf16(gpu, &activation_bytes)?;
        let weight = host_i8(
            &shard,
            &format!("{tensor}.weight"),
            &[output_dim, input_dim],
        )?;
        let scale = host_f32(&shard, &format!("{tensor}.weight_scale"), &[output_dim, 1])?;
        let weight_bytes: Vec<u8> = weight.iter().map(|value| *value as u8).collect();
        let scale_bytes: Vec<u8> = scale.iter().flat_map(|value| value.to_le_bytes()).collect();
        let weight_dev = gpu
            .alloc(Device::I8, weight.len())
            .map_err(|detail| format!("probe weight: {detail}"))?;
        let scale_dev = gpu
            .alloc(Device::F32, scale.len())
            .map_err(|detail| format!("probe scale: {detail}"))?;
        gpu.upload(&weight_dev, &weight_bytes)
            .map_err(|detail| format!("probe weight upload: {detail}"))?;
        gpu.upload(&scale_dev, &scale_bytes)
            .map_err(|detail| format!("probe scale upload: {detail}"))?;
        let output = gpu
            .alloc(Device::Bf16, rows * output_dim)
            .map_err(|detail| format!("probe output: {detail}"))?;
        let plan = gpu
            .int8_plan(
                rows as u32,
                input_dim as u32,
                output_dim as u32,
                GROUP as u32,
            )
            .map_err(|detail| format!("probe plan: {detail}"))?;
        gpu.linear_int8_convrot(
            &output,
            &input,
            &weight_dev,
            &scale_dev,
            None,
            rows as u32,
            input_dim as u32,
            output_dim as u32,
            GROUP as u32,
            &plan,
        )
        .map_err(|detail| format!("probe linear: {detail}"))?;
        let mut got_bytes = vec![0u8; output.bytes()];
        gpu.read(&output, &mut got_bytes)
            .map_err(|detail| format!("probe read: {detail}"))?;
        let want = host_linear(
            &activation,
            rows,
            &HostLinear {
                weight,
                scale,
                input_dim,
                output_dim,
            },
            &signs,
        );
        let (differ, rel, worst) = compare(&to_f32(&got_bytes), &want);
        println!(
            "PROBE convrot {name} rows={rows} differ={differ} rel_rms={rel:.3e} worst={worst:.3e}"
        );
    }

    // 2. the RMS norm with layer 0's `input_layernorm` weight.
    let norm = host_bf16(&shard, "model.layers.0.input_layernorm.weight", &[HIDDEN])?;
    let norm_bytes: Vec<u8> = norm
        .iter()
        .flat_map(|value| bf16_bits(*value).to_le_bytes())
        .collect();
    let norm_dev = gpu
        .alloc(Device::Bf16, norm.len())
        .map_err(|detail| format!("probe norm: {detail}"))?;
    gpu.upload(&norm_dev, &norm_bytes)
        .map_err(|detail| format!("probe norm upload: {detail}"))?;
    let norm_out = gpu
        .alloc(Device::Bf16, rows * HIDDEN)
        .map_err(|detail| format!("probe norm output: {detail}"))?;
    gpu.rms_norm(
        &norm_out,
        &input,
        Some(&norm_dev),
        rows as u32,
        HIDDEN as u32,
        gemma::RMS_EPSILON,
    )
    .map_err(|detail| format!("probe rms: {detail}"))?;
    let mut norm_bytes_out = vec![0u8; norm_out.bytes()];
    gpu.read(&norm_out, &mut norm_bytes_out)
        .map_err(|detail| format!("probe norm read: {detail}"))?;
    let want = rms_norm(&activation, rows, HIDDEN, Some(&norm));
    let (differ, rel, worst) = compare(&to_f32(&norm_bytes_out), &want);
    println!("PROBE rms_norm rows={rows} differ={differ} rel_rms={rel:.3e} worst={worst:.3e}");

    // 3. the attention, both layer flavours, on synthetic heads. This is the
    // only op whose shape and head count change with the layer type, and the
    // mask is the device's own.
    for (name, heads, kv_heads, width, window) in [
        (
            "sliding",
            gemma::HEADS,
            gemma::KV_HEADS,
            gemma::HEAD_DIM,
            gemma::SLIDING_WINDOW as u32,
        ),
        (
            "full",
            gemma::HEADS,
            gemma::GLOBAL_KV_HEADS,
            gemma::GLOBAL_HEAD_DIM,
            0,
        ),
    ] {
        let (query, query_bytes) = synthetic(rows * heads * width, &mut state);
        let (key, key_bytes) = synthetic(rows * kv_heads * width, &mut state);
        let (value, value_bytes) = synthetic(rows * kv_heads * width, &mut state);
        let query_dev = upload_f32_bf16(gpu, &query_bytes)?;
        let key_dev = upload_f32_bf16(gpu, &key_bytes)?;
        let value_dev = upload_f32_bf16(gpu, &value_bytes)?;
        let mask_dev = gpu
            .alloc(Device::Bf16, rows * rows)
            .map_err(|detail| format!("probe mask: {detail}"))?;
        gpu.gemma_mask(&mask_dev, rows as u32, window, Some(&padding_dev))
            .map_err(|detail| format!("probe mask: {detail}"))?;
        let mut mask_bytes = vec![0u8; mask_dev.bytes()];
        gpu.read(&mask_dev, &mut mask_bytes)
            .map_err(|detail| format!("probe mask read: {detail}"))?;
        let out = gpu
            .alloc(Device::Bf16, rows * heads * width)
            .map_err(|detail| format!("probe attention: {detail}"))?;
        gpu.gemma_attention(
            &out,
            &query_dev,
            &key_dev,
            &value_dev,
            Some(&mask_dev),
            rows as u32,
            heads as u32,
            kv_heads as u32,
            width as u32,
            gemma::ATTENTION_SCALE,
        )
        .map_err(|detail| format!("probe attention: {detail}"))?;
        let mut got_bytes = vec![0u8; out.bytes()];
        gpu.read(&out, &mut got_bytes)
            .map_err(|detail| format!("probe attention read: {detail}"))?;
        let want = host_attention(
            &query,
            &key,
            &value,
            &to_f32(&mask_bytes),
            rows,
            heads,
            kv_heads,
            width,
        );
        let (differ, rel, worst) = compare(&to_f32(&got_bytes), &want);
        println!(
            "PROBE attention {name} rows={rows} differ={differ} rel_rms={rel:.3e} worst={worst:.3e}"
        );
    }

    // 4. the RoPE rotation, both flavours, with the device's own pair tables.
    for (name, heads, kv_heads, width, angles, base) in [
        (
            "sliding",
            gemma::HEADS,
            gemma::KV_HEADS,
            gemma::HEAD_DIM,
            gemma::SLIDING_ANGLES as u32,
            gemma::SLIDING_BASE,
        ),
        (
            "full",
            gemma::HEADS,
            gemma::GLOBAL_KV_HEADS,
            gemma::GLOBAL_HEAD_DIM,
            gemma::FULL_ANGLES as u32,
            gemma::FULL_BASE,
        ),
    ] {
        let tables = Tables::from_device(gpu, rows as u32, width as u32, angles, base)?;
        let (query, query_bytes) = synthetic(rows * heads * width, &mut state);
        let (key, key_bytes) = synthetic(rows * kv_heads * width, &mut state);
        let query_dev = upload_f32_bf16(gpu, &query_bytes)?;
        let key_dev = upload_f32_bf16(gpu, &key_bytes)?;
        let cos_dev = upload_f32_bf16(gpu, &bf16_bytes(&tables.cos))?;
        let sin_dev = upload_f32_bf16(gpu, &bf16_bytes(&tables.sin))?;
        gpu.gemma_rope(
            &query_dev,
            &key_dev,
            &cos_dev,
            &sin_dev,
            rows as u32,
            heads as u32,
            kv_heads as u32,
            width as u32,
        )
        .map_err(|detail| format!("probe rope: {detail}"))?;
        let mut query_out = vec![0u8; query_dev.bytes()];
        let mut key_out = vec![0u8; key_dev.bytes()];
        gpu.read(&query_dev, &mut query_out)
            .map_err(|detail| format!("probe rope read: {detail}"))?;
        gpu.read(&key_dev, &mut key_out)
            .map_err(|detail| format!("probe rope read: {detail}"))?;
        let want_query = rotate(&query, &tables, rows, heads, width);
        let want_key = rotate(&key, &tables, rows, kv_heads, width);
        let (differ, rel, worst) = compare(&to_f32(&query_out), &want_query);
        println!(
            "PROBE rope {name} query rows={rows} differ={differ} rel_rms={rel:.3e} worst={worst:.3e}"
        );
        let (differ, rel, worst) = compare(&to_f32(&key_out), &want_key);
        println!(
            "PROBE rope {name} key rows={rows} differ={differ} rel_rms={rel:.3e} worst={worst:.3e}"
        );
    }

    Ok(())
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16_bits(*value).to_le_bytes())
        .collect()
}

/// A BF16 value with occasional outliers, and its bytes.
fn synthetic(count: usize, state: &mut u32) -> (Vec<f32>, Vec<u8>) {
    let mut values = vec![0.0f32; count];
    let mut bytes = Vec::with_capacity(count * 2);

    for (at, value) in values.iter_mut().enumerate() {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (*state >> 8) as f32 / 8_388_608.0 - 1.0;
        let magnitude = if at % 4096 == 0 { 6.0 } else { 1.0 };
        *value = gemma::bf16_round(unit * magnitude);
        bytes.extend_from_slice(&bf16_bits(*value).to_le_bytes());
    }

    (values, bytes)
}

fn upload_f32_bf16(gpu: &Gpu, bytes: &[u8]) -> Result<Tensor, String> {
    let tensor = gpu
        .alloc(Device::Bf16, bytes.len() / 2)
        .map_err(|detail| format!("probe tensor: {detail}"))?;
    gpu.upload(&tensor, bytes)
        .map_err(|detail| format!("probe tensor upload: {detail}"))?;

    Ok(tensor)
}

fn bf16_bits(value: f32) -> u16 {
    (gemma::bf16_round(value).to_bits() >> 16) as u16
}

// ---- arguments ----

struct Options {
    layers: Option<usize>,
    prompt: Option<String>,
    tokens_only: bool,
    rows: Option<usize>,
    padding: Option<usize>,
    seed: Option<u32>,
    ids: Option<Vec<u32>>,
    oracle: bool,
    probe: bool,
    expect: Vec<u64>,
}

fn split(args: &[String]) -> Result<(Vec<String>, Options), String> {
    let mut positional = Vec::new();
    let mut options = Options {
        layers: None,
        prompt: None,
        tokens_only: false,
        rows: None,
        padding: None,
        seed: None,
        ids: None,
        oracle: false,
        probe: false,
        expect: Vec::new(),
    };
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];

        match argument.as_str() {
            "--oracle" => {
                options.oracle = true;
                index += 1;
            }
            "--probe" => {
                options.probe = true;
                index += 1;
            }
            "--tokens-only" => {
                options.tokens_only = true;
                index += 1;
            }
            "--prompt" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage("--prompt needs a value"))?;
                options.prompt = Some(value.clone());
                index += 2;
            }
            "--layers" | "--rows" | "--padding" | "--seed" | "--ids" | "--expect" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage(&format!("{argument} needs a value")))?;

                match argument.as_str() {
                    "--layers" => {
                        options.layers = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| usage("--layers needs a number"))?,
                        )
                    }
                    "--rows" => {
                        options.rows = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| usage("--rows needs a number"))?,
                        )
                    }
                    "--padding" => {
                        options.padding = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| usage("--padding needs a number"))?,
                        )
                    }
                    "--seed" => {
                        options.seed = Some(
                            value
                                .parse::<u32>()
                                .map_err(|_| usage("--seed needs a number"))?,
                        )
                    }
                    "--ids" => {
                        let mut ids = Vec::new();

                        for word in value.split(',') {
                            ids.push(
                                word.parse::<u32>()
                                    .map_err(|_| usage("--ids takes comma-separated numbers"))?,
                            );
                        }

                        options.ids = Some(ids);
                    }
                    _ => {
                        if value.len() != 16 {
                            return Err(usage("--expect needs a 16-digit hexadecimal hash"));
                        }

                        options.expect.push(
                            u64::from_str_radix(value, 16)
                                .map_err(|_| usage("--expect needs a hexadecimal hash"))?,
                        );
                    }
                }

                index += 2;
            }
            _ if argument.starts_with("--") => {
                return Err(usage(&format!("unknown option {argument}")));
            }
            _ => {
                positional.push(argument.clone());
                index += 1;
            }
        }
    }

    Ok((positional, options))
}

fn usage(detail: &str) -> String {
    format!(
        "{detail}\nusage: h3_gemma_gate <checkpoint> [--prompt TEXT] [--tokens-only] \
         [--layers N] [--rows R] [--padding P] [--seed S] [--ids a,b,c] \
         [--oracle] [--probe] [--expect HEX]"
    )
}
