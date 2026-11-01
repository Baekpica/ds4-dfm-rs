//! The unit-4 gate: the feature extractor and the AV connector.
//!
//! Usage: h3_gemma_connector_gate --gemma PATH --connector PATH
//!          [--rows R] [--padding P] [--seed S] [--probe] [--prompt TEXT]
//!          [--expect HEX ...]
//!
//! Three modes:
//!
//! - default: 49 synthetic hidden states through the whole
//!   `minimax_h3::gemma_connector` graph, one FNV-1a line per stage
//!   (`scales`, `normed`, both feature tensors, both contexts), so a golden
//!   from the reference is a straight comparison.
//! - `--probe`: each new device kernel against a host transcription of the
//!   same expression, value by value. The transcriptions reproduce the port's
//!   reduction order, so a kernel whose reference has no libm in it must come
//!   back **bit-identical**; the two that end in `rsqrtf` and `expf` are
//!   reported as the residues they are, with the number.
//! - `--prompt`: the full path - the checkpoint's own tokenizer, the Gemma4
//!   tower's 49 states, then the extractor and both connectors. This is the
//!   one the Spark runs; the tower alone needs more memory than the
//!   workstation's card has.
//!
//! There is no Python in this campaign and the pinned reference cannot run on
//! either host, so this gate is the same shape as the tower's: a second
//! implementation of the reference's own expressions, cited to
//! `feature_extractor.py` / `embeddings_connector.py` / `embeddings_processor
//! .py` / `rope.py` / `ops.py` line by line.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::gemma::{Tower, HIDDEN, LAYERS as TOWER_LAYERS, STATES};
use ds4_core::minimax_h3::gemma_connector::{
    self as connector, Processor, AUDIO_INNER, HEADS, REGISTERS, VIDEO_HEAD_DIM, VIDEO_INNER,
};
use ds4_core::minimax_h3::gemma_token::Tokenizer as GemmaTokenizer;
use ds4_core::minimax_h3::prompt_cache::{self, Cache, PromptStats};
use ds4_core::minimax_h3::text_forward::fnv1a;
use ds4_core::minimax_h3::torch_file::{Tensor as HostTensor, TorchDtype};
use ds4_sys::h3_dit::{Dtype as Device, Gpu, Tensor};

/// The ceiling a probe whose reference ends in a libm call must stay under.
/// BF16 holds eight mantissa bits, so one ulp of `rsqrtf` or `expf` moves a
/// rounded value by a few tenths of a percent; a wiring fault moves every
/// element, by two orders more.
const RESIDUAL_CEILING: f64 = 2e-2;

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
    let options = split(args)?;
    let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;

    if options.probe {
        probe(&gpu, options.rows.unwrap_or(8), options.seed.unwrap_or(1))?;

        return Ok(ExitCode::SUCCESS);
    }

    let gemma = options
        .gemma
        .clone()
        .ok_or_else(|| usage("--gemma is required"))?;
    let transformer = options
        .connector
        .clone()
        .ok_or_else(|| usage("--connector is required"))?;

    // `--prompt` runs the tower first, so the connector's input is the
    // reference's own: the tokenizer's 1024 ids, 49 states, the attention
    // mask. Without it the gate draws states from a seeded LCG and the mask
    // from `--padding`, exactly as the tower's gate does.
    let rows = match &options.prompt {
        Some(_) => connector::ROWS,
        None => options.rows.unwrap_or(REGISTERS),
    };
    if rows % REGISTERS != 0 {
        return Err(usage("--rows must be a multiple of 128"));
    }

    let started = Instant::now();
    let processor = Processor::open(&gpu, &gemma, &transformer).map_err(|error| error.token())?;
    let load = started.elapsed().as_secs_f64();
    let (tensors, bytes) = processor.loaded();

    let (states, padding) = match &options.prompt {
        Some(prompt) => {
            let tokenizer = GemmaTokenizer::open(&gemma).map_err(|error| error.token())?;
            let (ids, mask) = tokenizer.tokenize(prompt);

            println!(
                "TOKENS width={} vocabulary={} bos={} padding={} first={:?}",
                ids.len(),
                tokenizer.width(),
                tokenizer.bos(),
                mask.iter().filter(|value| **value == 0).count(),
                &ids[..8.min(ids.len())]
            );

            let tower = Tower::open(&gpu, &gemma, TOWER_LAYERS).map_err(|error| error.token())?;
            let states = tower
                .encode(&gpu, &ids, &mask, TOWER_LAYERS)
                .map_err(|error| error.token())?;

            (states, mask)
        }
        None => {
            let padding = options.padding.unwrap_or(rows / 8);
            if padding >= rows {
                return Err(usage("--padding must leave at least one token"));
            }

            (
                synthetic_states(&gpu, rows, options.seed.unwrap_or(1))?,
                mask(rows, padding),
            )
        }
    };

    let started = Instant::now();
    let stages = processor
        .process(&gpu, &states, &padding)
        .map_err(|error| error.token())?;
    let seconds = started.elapsed().as_secs_f64();

    println!(
        "CONNECTOR rows={} padding={} tensors={} bytes={} load={:.3} run={:.3}",
        stages.rows,
        padding.iter().filter(|value| **value == 0).count(),
        tensors,
        bytes,
        load,
        seconds
    );

    let mut expect_failed = false;

    if let Some(directory) = &options.dump {
        std::fs::create_dir_all(directory)
            .map_err(|error| format!("--dump {}: {error}", directory.display()))?;
    }

    for (name, tensor) in [
        ("scales", &stages.scales),
        ("normed", &stages.normed),
        ("video_features", &stages.video_features),
        ("audio_features", &stages.audio_features),
        ("video", &stages.video),
        ("audio", &stages.audio),
    ] {
        let mut payload = vec![0u8; tensor.bytes()];
        gpu.read(tensor, &mut payload)
            .map_err(|detail| format!("read {name}: {detail}"))?;
        let hash = fnv1a(&payload);

        println!(
            "STAGE {name} values={} fnv={hash:016x} first={} last={}",
            payload.len() / 2,
            words(&payload[..16.min(payload.len())]),
            words(&payload[payload.len().saturating_sub(16)..])
        );

        if let Some(directory) = &options.dump {
            std::fs::write(directory.join(format!("{name}.bf16")), &payload)
                .map_err(|error| format!("--dump {name}: {error}"))?;
        }

        if let Some(wanted) = options.expect.get(name) {
            if *wanted != hash {
                println!("STAGE {name} expected {wanted:016x}");
                expect_failed = true;
            }
        }
    }

    if expect_failed {
        return Ok(ExitCode::from(3));
    }

    if let Some(path) = &options.cache {
        write_cache(&gpu, &stages, options.prompt.as_deref(), path)?;
    }

    println!("GEMMA-CONNECTOR-OK");
    Ok(ExitCode::SUCCESS)
}

/// Write the port's own two contexts as a stage-2 cache the refiner loads. The
/// cache contract is `runtime/prompt_cache.py`; the payload is
/// `runtime/cache_builder.py:129-150`.
fn write_cache(
    gpu: &Gpu,
    stages: &connector::Stages,
    prompt: Option<&str>,
    path: &Path,
) -> Result<(), String> {
    let prompt = prompt.ok_or_else(|| usage("--cache needs --prompt"))?;
    if stages.rows as usize != connector::ROWS {
        return Err(usage("--cache needs the recipe's 1024 rows"));
    }
    let read = |tensor: &Tensor| -> Result<Vec<u8>, String> {
        let mut payload = vec![0u8; tensor.bytes()];
        gpu.read(tensor, &mut payload)
            .map_err(|detail| format!("read context: {detail}"))?;
        Ok(payload)
    };
    let shape = |inner: usize| vec![1, stages.rows as u64, inner as u64];
    let cache = Cache {
        prompt: prompt.to_string(),
        fingerprint: prompt_cache::fingerprint(prompt),
        video: HostTensor::contiguous(
            TorchDtype::BFloat16,
            shape(VIDEO_INNER),
            read(&stages.video)?,
        ),
        audio: HostTensor::contiguous(
            TorchDtype::BFloat16,
            shape(AUDIO_INNER),
            read(&stages.audio)?,
        ),
        stats: PromptStats {
            characters: prompt.chars().count() as u64,
            whitespace_words: Some(prompt.split_whitespace().count() as u64),
            raw_tokens_including_bos: None,
            kept_tokens: None,
            max_tokens: None,
            truncated: false,
        },
        generation: ds4_core::minimax_h3::torch_file::map(vec![
            (
                "builder",
                ds4_core::minimax_h3::torch_file::Value::Str(
                    "ds4-dfm-rs h3_gemma_connector_gate".to_string(),
                ),
            ),
            (
                "source",
                ds4_core::minimax_h3::torch_file::Value::Str(
                    "port tokenizer, tower and connector".to_string(),
                ),
            ),
        ]),
    };
    prompt_cache::save(path, &cache).map_err(|error| error.token())?;
    println!(
        "CACHE path={} rows={} fingerprint={}",
        path.display(),
        stages.rows,
        cache.fingerprint
    );
    Ok(())
}

/// Each new device kernel against a host transcription of the same
/// expression. The reductions are reproduced exactly (256 partials and the
/// same halving tree), so only the two probes that end in a libm call -
/// `rsqrtf` in the extractor's scale, `expf` in the gate - can differ at all,
/// and those are reported with the residue they produce.
fn probe(gpu: &Gpu, rows: usize, seed: u32) -> Result<(), String> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut failed = false;

    // The extractor's stack is `[layers, rows, width]`; a real `width` is
    // 3840 and the reduction is width-independent, so the probe uses 512 to
    // keep the transcription cheap. 49 layers is the stack's own depth.
    let width = 512usize;
    let layers = STATES;
    let (stacked, stacked_bytes) = seq(rows * layers * width, &mut state);
    let stacked_dev = upload_bf16(gpu, &stacked_bytes)?;
    let scales_dev = alloc(gpu, Device::Bf16, rows * layers)?;
    gpu.connector_stack_scale(
        &scales_dev,
        &stacked_dev,
        rows as u32,
        width as u32,
        layers as u32,
    )
    .map_err(|detail| format!("stack scale: {detail}"))?;
    let got = read_bf16(gpu, &scales_dev)?;
    let scales = host_stack_scale(&stacked, rows, width, layers);
    failed |= report("stack_scale", &got, &scales, Verdict::Residual);

    let output_dev = alloc(gpu, Device::Bf16, rows * width * layers)?;
    gpu.connector_interleave(
        &output_dev,
        &stacked_dev,
        &scales_dev,
        rows as u32,
        width as u32,
        layers as u32,
    )
    .map_err(|detail| format!("interleave: {detail}"))?;
    let got = read_bf16(gpu, &output_dev)?;
    let want = host_interleave(&stacked, &scales, rows, width, layers);
    failed |= report("interleave", &got, &want, Verdict::Exact);

    // The right-pad gather and the register fill share the `[rows, width]`
    // shape. The permutation and the binary mask are the ones a left-padded
    // prompt produces; the reverse permutation here makes the reorder visible
    // element by element.
    let (features, features_bytes) = seq(rows * width, &mut state);
    let features_dev = upload_bf16(gpu, &features_bytes)?;
    let order: Vec<u32> = (0..rows as u32).rev().collect();
    let order_dev = upload_u32(gpu, &order)?;
    let gathered_dev = alloc(gpu, Device::Bf16, rows * width)?;
    gpu.gather_rows(
        &gathered_dev,
        &features_dev,
        &order_dev,
        rows as u32,
        width as u32,
    )
    .map_err(|detail| format!("gather: {detail}"))?;
    let got = read_bf16(gpu, &gathered_dev)?;
    let gathered: Vec<f32> = (0..rows)
        .flat_map(|row| features[order[row] as usize * width..][..width].to_vec())
        .collect();
    failed |= report("gather_rows", &got, &gathered, Verdict::Exact);

    let (registers, registers_bytes) = seq(REGISTERS * width, &mut state);
    let registers_dev = upload_bf16(gpu, &registers_bytes)?;
    let binary: Vec<u32> = (0..rows as u32)
        .map(|row| u32::from(row < rows as u32 / 2))
        .collect();
    let binary_dev = upload_u32(gpu, &binary)?;
    let filled_dev = alloc(gpu, Device::Bf16, rows * width)?;
    gpu.register_fill(
        &filled_dev,
        &gathered_dev,
        &binary_dev,
        &registers_dev,
        rows as u32,
        width as u32,
        REGISTERS as u32,
    )
    .map_err(|detail| format!("register fill: {detail}"))?;
    let got = read_bf16(gpu, &filled_dev)?;
    let want = host_register_fill(&gathered, &registers, &binary, rows, width);
    failed |= report("register_fill", &got, &want, Verdict::Exact);

    failed |= probe_scale(gpu, rows, &mut state)?;

    // The split rotation at both connector geometries.
    for (head_dim, label) in [
        (VIDEO_HEAD_DIM, "video"),
        (connector::AUDIO_HEAD_DIM, "audio"),
    ] {
        let inner = HEADS * head_dim;
        let (values, values_bytes) = seq(rows * inner, &mut state);
        let query_dev = upload_bf16(gpu, &values_bytes)?;
        let key_dev = upload_bf16(gpu, &values_bytes)?;
        let (cos_dev, sin_dev) = connector::rope_tables(gpu, rows as u32, HEADS, head_dim)
            .map_err(|error| error.token())?;
        let cos = read_bf16(gpu, &cos_dev)?;
        let sin = read_bf16(gpu, &sin_dev)?;
        gpu.text_rope(
            &query_dev,
            &key_dev,
            &cos_dev,
            &sin_dev,
            rows as u32,
            HEADS as u32,
            head_dim as u32,
        )
        .map_err(|detail| format!("rope: {detail}"))?;
        let got = read_bf16(gpu, &query_dev)?;
        let want = host_rope(&values, &cos, &sin, rows, HEADS, head_dim);
        failed |= report(&format!("rope_{label}"), &got, &want, Verdict::Exact);
    }

    // The per-head gate.
    let inner = VIDEO_INNER;
    let (values, values_bytes) = seq(rows * inner, &mut state);
    let (logits, logits_bytes) = seq(rows * HEADS, &mut state);
    let values_dev = upload_bf16(gpu, &values_bytes)?;
    let logits_dev = upload_bf16(gpu, &logits_bytes)?;
    let gated_dev = alloc(gpu, Device::Bf16, rows * inner)?;
    gpu.per_head_gate(
        &gated_dev,
        &values_dev,
        &logits_dev,
        rows as u32,
        HEADS as u32,
        VIDEO_HEAD_DIM as u32,
    )
    .map_err(|detail| format!("per-head gate: {detail}"))?;
    let got = read_bf16(gpu, &gated_dev)?;
    let want = host_per_head_gate(&values, &logits, rows, HEADS, VIDEO_HEAD_DIM);
    failed |= report("per_head_gate", &got, &want, Verdict::Residual);

    // Dense attention at both connector geometries, transcribed the way the
    // reference's `PytorchAttention` reads its operands (`attention.py:82-102`):
    // `q.view(b, -1, heads, dim_head).transpose(1, 2)`, SDPA at
    // `1 / sqrt(dim_head)`, then `.transpose(1, 2).reshape(b, -1, heads * dim)`.
    // The operands are token-major - the layout every caller produces,
    // since `to_q`/`to_k`/`to_v` are plain `torch.nn.Linear`s.
    for (head_dim, label) in [
        (VIDEO_HEAD_DIM, "video"),
        (connector::AUDIO_HEAD_DIM, "audio"),
    ] {
        let inner = HEADS * head_dim;
        let (query, query_bytes) = seq(rows * inner, &mut state);
        let (key, key_bytes) = seq(rows * inner, &mut state);
        let (value, value_bytes) = seq(rows * inner, &mut state);
        let query_dev = upload_bf16(gpu, &query_bytes)?;
        let key_dev = upload_bf16(gpu, &key_bytes)?;
        let value_dev = upload_bf16(gpu, &value_bytes)?;
        let attention_dev = alloc(gpu, Device::Bf16, rows * inner)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        gpu.text_attention(
            &attention_dev,
            &query_dev,
            &key_dev,
            &value_dev,
            rows as u32,
            rows as u32,
            HEADS as u32,
            head_dim as u32,
            scale,
        )
        .map_err(|detail| format!("attention: {detail}"))?;
        let got = read_bf16(gpu, &attention_dev)?;
        let want = host_attention(&query, &key, &value, rows, rows, HEADS, head_dim, scale);
        failed |= report(
            &format!("attention_{label}"),
            &got,
            &want,
            Verdict::Residual,
        );
    }

    if failed {
        return Err("a probe failed its verdict".to_string());
    }

    println!("GEMMA-CONNECTOR-PROBE-OK");
    Ok(())
}

fn probe_scale(gpu: &Gpu, rows: usize, state: &mut u32) -> Result<bool, String> {
    let width = 512usize;
    let (values, values_bytes) = seq(rows * width, state);
    let input = upload_bf16(gpu, &values_bytes)?;
    let output = alloc(gpu, Device::Bf16, rows * width)?;
    let factor = (VIDEO_INNER as f32 / HIDDEN as f32).sqrt();
    gpu.scale_constant(&output, &input, factor)
        .map_err(|detail| format!("scale constant: {detail}"))?;
    let got = read_bf16(gpu, &output)?;
    let want: Vec<f32> = values.iter().map(|value| bf16(value * factor)).collect();

    Ok(report("scale_constant", &got, &want, Verdict::Exact))
}

/// Whether a probe's reference expression contains a libm call.
#[derive(Clone, Copy, PartialEq)]
enum Verdict {
    Exact,
    Residual,
}

fn report(name: &str, got: &[f32], want: &[f32], verdict: Verdict) -> bool {
    let (differ, rel_rms) = compare(got, want);
    let pass = match verdict {
        Verdict::Exact => differ == 0,
        Verdict::Residual => rel_rms <= RESIDUAL_CEILING,
    };

    println!(
        "PROBE {name} values={} differ={differ} rel_rms={rel_rms:.3e} verdict={} {}",
        got.len(),
        if verdict == Verdict::Exact {
            "exact"
        } else {
            "residual"
        },
        if pass { "pass" } else { "FAIL" }
    );

    !pass
}

/// The host transcription of `h3d_connector_stack_scale`: the same 256
/// partials (thread `t` sums `t, t + 256, ...`), the same halving tree, and
/// the same four BF16 roundings. The one difference the device has is
/// `rsqrtf` where this writes `1.0 / sqrt`, which is why the probe is a
/// residual.
fn host_stack_scale(stacked: &[f32], rows: usize, width: usize, layers: usize) -> Vec<f32> {
    let mut scales = vec![0.0f32; rows * layers];

    for index in 0..layers {
        for row in 0..rows {
            let source = &stacked[(index * rows + row) * width..][..width];
            let mut partial = [0.0f32; 256];

            for (thread, slot) in partial.iter_mut().enumerate() {
                let mut total = 0.0f32;
                let mut column = thread;
                while column < width {
                    let value = source[column];
                    total += bf16(value * value);
                    column += 256;
                }
                *slot = total;
            }

            let mut stride = 128;
            while stride > 0 {
                for thread in 0..stride {
                    partial[thread] += partial[thread + stride];
                }
                stride >>= 1;
            }

            let mean = bf16(partial[0] / width as f32);
            let shifted = bf16(mean + 1e-6);
            scales[row * layers + index] = bf16(1.0f32 / shifted.sqrt());
        }
    }

    scales
}

/// `normed.reshape(B, T, D * L)` with the layer's scale folded in.
fn host_interleave(
    stacked: &[f32],
    scales: &[f32],
    rows: usize,
    width: usize,
    layers: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * width * layers];

    for row in 0..rows {
        for layer in 0..layers {
            let scale = scales[row * layers + layer];
            for dimension in 0..width {
                output[row * width * layers + dimension * layers + layer] =
                    bf16(stacked[(layer * rows + row) * width + dimension] * scale);
            }
        }
    }

    output
}

/// `_replace_padded_with_learnable_registers`: keep the gathered row where the
/// binary mask is set, else the tiled register.
fn host_register_fill(
    gathered: &[f32],
    registers: &[f32],
    binary: &[u32],
    rows: usize,
    width: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * width];

    for row in 0..rows {
        let source = if binary[row] == 1 {
            &gathered[row * width..][..width]
        } else {
            &registers[(row % REGISTERS) * width..][..width]
        };
        output[row * width..][..width].copy_from_slice(source);
    }

    output
}

/// `apply_split_rotary_emb` (`rope.py:66-77`): `first * cos - second * sin`
/// and `second * cos + first * sin`, the product rounded to BF16 and the add
/// the fused multiply-add the reference's `addcmul_` compiles to.
fn host_rope(
    values: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut output = values.to_vec();

    for row in 0..rows {
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            let table = (row * heads + head) * half;

            for dimension in 0..half {
                let c = cos[table + dimension];
                let s = sin[table + dimension];
                let first = values[base + dimension];
                let second = values[base + half + dimension];
                let turned_first = bf16(first * c);
                let turned_second = bf16(second * c);
                output[base + dimension] = bf16((-s).mul_add(second, turned_first));
                output[base + half + dimension] = bf16(s.mul_add(first, turned_second));
            }
        }
    }

    output
}

/// `PytorchGatedAttention`: `2 * sigmoid(logit)` rounded, then the product.
/// The reference's `expf` has no host opposite, so this is a residual probe.
fn host_per_head_gate(
    values: &[f32],
    logits: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = values.to_vec();

    for row in 0..rows {
        for head in 0..heads {
            let logit = logits[row * heads + head];
            let gate = bf16(2.0f32 * (1.0f32 / (1.0f32 + (-logit).exp())));
            let base = (row * heads + head) * head_dim;

            for dimension in 0..head_dim {
                output[base + dimension] = bf16(values[base + dimension] * gate);
            }
        }
    }

    output
}

/// The host transcription of `PytorchAttention.__call__` (`attention.py:82-102`):
/// the operands are token-major `(rows, heads, dim_head)`, the softmax runs
/// over each `(head, row)`'s own keys at `1 / sqrt(dim_head)`, and the output
/// is written back token-major. `expf` has no host opposite, so this is a
/// residual probe.
fn host_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    q_rows: usize,
    kv_rows: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; q_rows * heads * head_dim];

    for row in 0..q_rows {
        for head in 0..heads {
            let query_base = (row * heads + head) * head_dim;
            let mut scores = vec![0.0f32; kv_rows];
            let mut maximum = f32::NEG_INFINITY;

            for source in 0..kv_rows {
                let key_base = (source * heads + head) * head_dim;
                let mut product = 0.0f32;

                for dimension in 0..head_dim {
                    product += query[query_base + dimension] * key[key_base + dimension];
                }

                let score = product * scale;
                scores[source] = score;
                maximum = maximum.max(score);
            }

            let mut denominator = 0.0f32;

            for score in scores.iter_mut() {
                *score = (*score - maximum).exp();
                denominator += *score;
            }

            for dimension in 0..head_dim {
                let mut total = 0.0f32;

                for source in 0..kv_rows {
                    let value_base = (source * heads + head) * head_dim;
                    total += scores[source] * value[value_base + dimension];
                }

                output[query_base + dimension] = total / denominator;
            }
        }
    }

    output
}

/// A bf16 value widened to f32, so both sides of a comparison see the same
/// input.
fn bf16(value: f32) -> f32 {
    f32::from_bits(
        value
            .to_bits()
            .wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1))
            & 0xffff_0000,
    )
}

fn compare(got: &[f32], want: &[f32]) -> (usize, f64) {
    let mut differ = 0usize;
    let mut square = 0.0f64;
    let mut reference = 0.0f64;

    for (left, right) in got.iter().zip(want.iter()) {
        if left.to_bits() != right.to_bits() {
            differ += 1;
        }
        square += ((*left as f64) - (*right as f64)).powi(2);
        reference += (*right as f64).powi(2);
    }

    (differ, (square / reference.max(1e-30)).sqrt())
}

/// 49 synthetic states: a seeded LCG over BF16, one `[rows, 3840]` each. The
/// release's tokenizer and tower produce the real ones; this is what the
/// workstation can afford, and it exercises the whole graph.
fn synthetic_states(gpu: &Gpu, rows: usize, seed: u32) -> Result<Vec<Tensor>, String> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut states = Vec::with_capacity(STATES);

    for _ in 0..STATES {
        let (_, bytes) = seq(rows * HIDDEN, &mut state);
        states.push(upload_bf16(gpu, &bytes)?);
    }

    Ok(states)
}

fn mask(rows: usize, padding: usize) -> Vec<u32> {
    let mut mask = vec![0u32; padding];
    mask.resize(rows, 1);

    mask
}

/// `n` BF16 values from a seeded LCG with occasional outliers, so an absmax
/// is not uniform.
fn seq(count: usize, state: &mut u32) -> (Vec<f32>, Vec<u8>) {
    let mut values = Vec::with_capacity(count);
    let mut bytes = Vec::with_capacity(count * 2);

    for _ in 0..count {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (*state >> 8) as f32 / (1u32 << 24) as f32;
        let value = bf16((unit * 2.0 - 1.0) * 2.0);
        values.push(value);
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }

    (values, bytes)
}

fn upload_bf16(gpu: &Gpu, bytes: &[u8]) -> Result<Tensor, String> {
    let tensor = gpu
        .alloc(Device::Bf16, bytes.len() / 2)
        .map_err(|detail| format!("alloc bf16: {detail}"))?;
    gpu.upload(&tensor, bytes)
        .map_err(|detail| format!("upload bf16: {detail}"))?;

    Ok(tensor)
}

fn upload_u32(gpu: &Gpu, values: &[u32]) -> Result<Tensor, String> {
    let tensor = gpu
        .alloc(Device::U32, values.len())
        .map_err(|detail| format!("alloc u32: {detail}"))?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload(&tensor, bytes)
        .map_err(|detail| format!("upload u32: {detail}"))?;

    Ok(tensor)
}

fn read_bf16(gpu: &Gpu, tensor: &Tensor) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; tensor.bytes()];
    gpu.read(tensor, &mut bytes)
        .map_err(|detail| format!("read bf16: {detail}"))?;

    Ok(bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
        .collect())
}

fn words(bytes: &[u8]) -> String {
    bytes
        .chunks_exact(2)
        .map(|pair| format!("{:04x}", u16::from_le_bytes([pair[0], pair[1]])))
        .collect::<Vec<_>>()
        .join(",")
}

fn alloc(gpu: &Gpu, dtype: Device, elements: usize) -> Result<Tensor, String> {
    gpu.alloc(dtype, elements)
        .map_err(|detail| format!("alloc: {detail}"))
}

struct Options {
    gemma: Option<PathBuf>,
    connector: Option<PathBuf>,
    rows: Option<usize>,
    padding: Option<usize>,
    seed: Option<u32>,
    probe: bool,
    prompt: Option<String>,
    dump: Option<PathBuf>,
    cache: Option<PathBuf>,
    expect: HashMap<&'static str, u64>,
}

fn split(args: &[String]) -> Result<Options, String> {
    let mut gemma = None;
    let mut connector = None;
    let mut rows = None;
    let mut padding = None;
    let mut seed = None;
    let mut probe = false;
    let mut prompt = None;
    let mut dump = None;
    let mut cache = None;
    let mut expect = HashMap::new();
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];

        match argument.as_str() {
            "--probe" => {
                probe = true;
                index += 1;
            }
            "--gemma" | "--connector" | "--rows" | "--padding" | "--seed" | "--prompt"
            | "--dump" | "--cache" | "--expect" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage(&format!("{argument} needs a value")))?;

                match argument.as_str() {
                    "--gemma" => gemma = Some(PathBuf::from(value)),
                    "--connector" => connector = Some(PathBuf::from(value)),
                    "--dump" => dump = Some(PathBuf::from(value)),
                    "--cache" => cache = Some(PathBuf::from(value)),
                    "--rows" => {
                        rows = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| usage("--rows needs a number"))?,
                        )
                    }
                    "--padding" => {
                        padding = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| usage("--padding needs a number"))?,
                        )
                    }
                    "--seed" => {
                        seed = Some(
                            value
                                .parse::<u32>()
                                .map_err(|_| usage("--seed needs a number"))?,
                        )
                    }
                    "--prompt" => prompt = Some(value.clone()),
                    _ => {
                        let (name, hash) = value
                            .split_once('=')
                            .ok_or_else(|| usage("--expect takes NAME=HEX"))?;
                        let name = name.to_string();

                        if hash.len() != 16 {
                            return Err(usage("--expect needs a 16-digit hexadecimal hash"));
                        }

                        let hash = u64::from_str_radix(hash, 16)
                            .map_err(|_| usage("--expect needs a hexadecimal hash"))?;
                        let known: &'static str = match name.as_str() {
                            "scales" => "scales",
                            "normed" => "normed",
                            "video_features" => "video_features",
                            "audio_features" => "audio_features",
                            "video" => "video",
                            "audio" => "audio",
                            _ => return Err(usage(&format!("unknown stage {name}"))),
                        };
                        expect.insert(known, hash);
                    }
                }

                index += 2;
            }
            _ if argument.starts_with("--") => {
                return Err(usage(&format!("unknown option {argument}")));
            }
            _ => return Err(usage(&format!("unexpected argument {argument}"))),
        }
    }

    Ok(Options {
        gemma,
        connector,
        rows,
        padding,
        seed,
        probe,
        prompt,
        dump,
        cache,
        expect,
    })
}

fn usage(detail: &str) -> String {
    format!(
        "{detail}\nusage: h3_gemma_connector_gate --gemma PATH --connector PATH \
         [--rows R] [--padding P] [--seed S] [--probe] [--prompt TEXT] \
         [--dump DIR] [--cache FILE] [--expect NAME=HEX ...]"
    )
}
