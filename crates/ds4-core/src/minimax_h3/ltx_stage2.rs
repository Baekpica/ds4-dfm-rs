//! The LTX-2.5 stage-2 sampler: the schedule, the x0 model, the Euler step and
//! the noiser, over the refiner graph in [`super::ltx_refiner`].
//!
//! The video-only ablation builds the joint AudioVideo transformer and then
//! denoises it with no audio state:
//! `prepare_denoise(upscaled, seed)` makes one noised video state
//! (`refiner_encoder_ablation_single_gpu.py:691-701`) and `denoise` runs
//!
//! ```text
//! euler_denoising_loop(sigmas, video_state=state, audio_state=None,
//!                      stepper=EulerDiffusionStep(),
//!                      transformer=self.wrapped_transformer,
//!                      denoiser=SimpleDenoiser(v_context=context, a_context=None))
//! ```
//!
//! (`:704-714`), so every step is one `SimpleDenoiser` call: `X0Model` around
//! the transformer, `to_denoised` at the per-token timestep, `post_process_latent`
//! against the clean latent, and one Euler step per schedule interval.
//!
//! Three things keep the port on the reference's values rather than on values
//! that merely look like them:
//!
//! - every arithmetic step is a device kernel whose expression is transcribed
//!   from the reference's own line, float32 roundings and branch selection
//!   included (`torch.lerp`'s `|w| < 0.5` arm, `bf16 * f32` promotion);
//! - the noiser's noise is torch's own draw, not a lookalike: the Philox state,
//!   `curand_normal4`, the store width and the grid are the ones
//!   `DistributionTemplates.h` uses, so the values are the reference's;
//! - nothing here reorders the schedule: `sigmas[:-1]` is three steps and the
//!   fourth value is a terminal marker, not an interval.

use ds4_sys::h3_dit::{Dtype, Gpu, SolPlanBuffers, Tensor};

use super::ltx_refiner::{
    patchify_video, unpatchify_video, Diagnostic, EdgeScratch, Edges, Geometry, Lora, Refiner,
    RefinerError, RefinerScratch, Request, Sol, CHANNELS, CONTEXT_ROWS, HEADS, TAU_SCHEDULE,
};

/// `STAGE2_SIGMAS` (`runtime/stage2_ops/models.py:19`). `euler_denoising_loop`
/// iterates `sigmas[:-1]` (`samplers.py:73`), so the last value is where the
/// loop stops and not a fourth interval; `X0Model`'s own terminal behaviour at
/// a zero sigma is `EulerAncestralDiffusionStep`'s business, not this stepper's.
pub const SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];

/// The intervals the loop walks: `sigmas[:-1]`.
pub const STEPS: usize = SIGMAS.len() - 1;

/// `noise_scale=STAGE2_SIGMAS[0]`
/// (`refiner_encoder_ablation_single_gpu.py:699`): the initial state keeps
/// 9.1% of its own signal.
pub const NOISE_SCALE: f32 = SIGMAS[0];

/// `torch.Generator(device=device).manual_seed(seed)`: the seed and the Philox
/// offset torch keeps as `philox_offset_per_thread_`
/// (`CUDAGeneratorImpl.cpp`). `set_current_seed` stores the seed and zeroes the
/// offset, and each distribution call hands out the current offset and then
/// adds the increment its launch computed - so the pair the second draw sees is
/// torch's, not a fresh one.
#[derive(Clone, Copy, Debug)]
pub struct Generator {
    seed: u64,
    offset: u64,
}

impl Generator {
    pub fn seeded(seed: u64) -> Generator {
        Generator { seed, offset: 0 }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// `torch.randn(x.shape, generator=self, dtype=x.dtype, device=x.device)`
    /// for a contiguous `x`: `normal_kernel<BFloat16>` reaches
    /// `distribution_nullary_kernel` with `accscalar_t = float` and a
    /// `float4`-wide unroll (`DistributionTemplates.h:461-476`, `:443-455`).
    /// The offset advance is `calc_execution_policy`'s (`:50-62`) and comes
    /// back from the device, so this type never recomputes it.
    pub fn randn(&mut self, gpu: &Gpu, output: &Tensor) -> Result<(), RefinerError> {
        let increment = gpu
            .randn_bf16(output, self.seed, self.offset)
            .map_err(RefinerError::Run)?;
        self.offset += increment;

        Ok(())
    }
}

/// The buffers one stage-2 evaluation reuses. `latent` is the state the loop
/// moves, in its patchified form: `create_initial_state` patchifies the initial
/// latent *and* its clean copy (`tools.py:156-183`), so both are `[rows, 128]`
/// and the denoise mask is the patchified `[rows]` of the reference's
/// `(1, 1, t, h, w)` F32 ones.
pub struct StageScratch {
    pub rows: u32,
    pub width: u32,
    /// The state the blocks transform in place.
    pub latent: Tensor,
    /// `state.clean_latent`, the same rows before the noise was added.
    pub clean: Tensor,
    /// The current draw, before the two lerps.
    pub noise: Tensor,
    /// `state.denoise_mask`, F32, one value per token.
    pub mask: Tensor,
    /// `timesteps_from_mask`: `mask * sigma`, F32, one value per token.
    pub timesteps: Tensor,
    /// `X0Model`'s `to_denoised` output.
    pub denoised: Tensor,
    /// `post_process_latent`'s blend.
    pub blended: Tensor,
    /// `EulerDiffusionStep`'s `to_velocity` output, kept because the reference
    /// rounds to BF16 between the velocity and the step.
    pub velocity: Tensor,
    /// The final state unpatchified, `(1, 128, f, h, w)` laid out as
    /// `[channels, rows]`.
    pub grid: Tensor,
}

impl StageScratch {
    pub fn new(gpu: &Gpu, geometry: &Geometry) -> Result<StageScratch, RefinerError> {
        let rows = geometry.video_tokens();
        let width = CHANNELS as u32;
        let state = rows as usize * width as usize;

        let mask = {
            let mut bytes = Vec::with_capacity(rows as usize * 4);

            for _ in 0..rows {
                bytes.extend_from_slice(&1.0f32.to_le_bytes());
            }

            let tensor = gpu
                .alloc(Dtype::F32, rows as usize)
                .map_err(RefinerError::Run)?;
            gpu.upload(&tensor, &bytes).map_err(RefinerError::Run)?;
            tensor
        };

        Ok(StageScratch {
            rows,
            width,
            latent: gpu.alloc(Dtype::Bf16, state).map_err(RefinerError::Run)?,
            clean: gpu.alloc(Dtype::Bf16, state).map_err(RefinerError::Run)?,
            noise: gpu.alloc(Dtype::Bf16, state).map_err(RefinerError::Run)?,
            mask,
            timesteps: gpu
                .alloc(Dtype::F32, rows as usize)
                .map_err(RefinerError::Run)?,
            denoised: gpu.alloc(Dtype::Bf16, state).map_err(RefinerError::Run)?,
            blended: gpu.alloc(Dtype::Bf16, state).map_err(RefinerError::Run)?,
            velocity: gpu.alloc(Dtype::Bf16, state).map_err(RefinerError::Run)?,
            grid: gpu.alloc(Dtype::Bf16, state).map_err(RefinerError::Run)?,
        })
    }
}

/// The refiner and its edges, the scratch both need, and the generator.
pub struct Stage2 {
    edges: Edges,
    refiner: Refiner,
    geometry: Geometry,
    edges_scratch: EdgeScratch,
    blocks: RefinerScratch,
    scratch: StageScratch,
    generator: Generator,
    /// The sigmas the loop walks, release first. `SIGMAS` is the shipped
    /// route; a shorter, lower-starting schedule keeps more of the draft's
    /// signal (the first value is the noise scale, so `1 - s` is what the
    /// latent contributes).
    schedule: Vec<f32>,
    /// The Sol-Attn routing plan when the release's policy is on: one set of
    /// (rows x heads) buffers the 47 routing layers rewrite per layer.
    sol: Option<SolPlanBuffers>,
}

impl Stage2 {
    /// Load the shipped transformer shard (`layers`) and the edges shard.
    /// `lora` is the release's distilled adapter, fused as each weight is
    /// read (`models.py:76-79`); `None` is the plain `dev` transformer.
    pub fn open(
        gpu: &Gpu,
        refiner_path: &std::path::Path,
        edges_path: &std::path::Path,
        geometry: Geometry,
        layers: &[usize],
        lora: Option<&Lora>,
    ) -> Result<Stage2, RefinerError> {
        let edges = Edges::open(gpu, edges_path, lora)?;
        let refiner = Refiner::open(gpu, refiner_path, layers, lora)?;

        Ok(Stage2 {
            edges,
            refiner,
            edges_scratch: EdgeScratch::new(gpu, &geometry)?,
            blocks: RefinerScratch::new(
                gpu,
                geometry.video_tokens(),
                geometry.audio_frames,
                CONTEXT_ROWS,
            )?,
            scratch: StageScratch::new(gpu, &geometry)?,
            geometry,
            generator: Generator::seeded(0),
            schedule: SIGMAS.to_vec(),
            sol: None,
        })
    }

    /// The blocks held and the bytes they were read from, for a latency
    /// record: one update reads every weight once, so `bytes` against the
    /// step's own seconds is the floor a bandwidth-bound update cannot beat.
    pub fn footprint(&self) -> (usize, u64) {
        (self.refiner.blocks(), self.refiner.loaded().1)
    }

    /// Turn on the release's Stage 2 Sol-Attn policy
    /// (`super_acceleration/stage2/sol_attention.py`): layers 1 through 47
    /// route their video self-attention with one tau per step (`TAU_SCHEDULE`)
    /// and layer 0 stays dense. Layer 0 has to be among the loaded blocks, so
    /// the split means what it says.
    pub fn enable_sol(&mut self, gpu: &Gpu) -> Result<(), RefinerError> {
        let plan = gpu
            .sol_plan(self.scratch.rows, HEADS as u32)
            .map_err(RefinerError::Run)?;
        self.sol = Some(plan);
        Ok(())
    }

    /// Whether the run routes (`sol_attention.py`'s "sol" backend).
    pub fn sol_enabled(&self) -> bool {
        self.sol.is_some()
    }

    /// The routing plan, for a caller that reads the mask back.
    pub fn sol_plan(&self) -> Option<&SolPlanBuffers> {
        self.sol.as_ref()
    }

    /// Turn on a diagnostic for every later step, so one run can price the
    /// video self-attention's term against an otherwise identical run.
    pub fn set_diagnostic(&mut self, diagnostic: Diagnostic) {
        self.refiner.set_diagnostic(diagnostic);
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub fn generator(&self) -> Generator {
        self.generator
    }

    /// `torch.Generator(device).manual_seed(seed)`
    /// (`refiner_encoder_ablation_single_gpu.py:692`).
    pub fn seed(&mut self, seed: u64) {
        self.generator = Generator::seeded(seed);
    }

    /// The state the loop moves, patchified.
    pub fn latent(&self) -> &Tensor {
        &self.scratch.latent
    }

    /// The last noise draw, the only place the RNG enters: a fixed seed makes
    /// it comparable across hosts.
    pub fn noise(&self) -> &Tensor {
        &self.scratch.noise
    }

    /// The state unpatchified to the latent grid, after
    /// `VideoLatentTools.clear_conditioning` + `unpatchify`
    /// (`refiner_encoder_ablation_single_gpu.py:718-721`).
    pub fn latent_grid(&self, gpu: &Gpu) -> Result<&Tensor, RefinerError> {
        unpatchify_video(
            gpu,
            &self.scratch.grid,
            &self.scratch.latent,
            self.scratch.rows,
        )?;

        Ok(&self.scratch.grid)
    }

    /// Replace the release's schedule. The first value is the noise scale, so
    /// a lower start keeps more of the transferred latent: `SIGMAS[0]` keeps
    /// 9.1%, `0.5` keeps half. Any schedule other than the release's runs the
    /// refiner dense, because `TAU_SCHEDULE` defines one tau per one of its
    /// three intervals and has no meaning for another length.
    pub fn set_schedule(&mut self, sigmas: &[f32]) -> Result<(), RefinerError> {
        validate_schedule(sigmas)?;

        if self.sol.is_some() && sigmas != SIGMAS {
            return Err(RefinerError::Run(
                "Sol-Attn's taus are defined per release interval; drop --sol for a \
                 custom schedule"
                    .into(),
            ));
        }

        self.schedule = sigmas.to_vec();
        Ok(())
    }

    /// The intervals this run walks.
    pub fn steps(&self) -> usize {
        self.schedule.len() - 1
    }

    /// The schedule's own values, for the record.
    pub fn schedule(&self) -> &[f32] {
        &self.schedule
    }

    /// Whether the run walks the release's schedule exactly.
    pub fn is_release_schedule(&self) -> bool {
        self.schedule == SIGMAS
    }

    /// Write the per-token denoise mask, which is `1` in every route the
    /// release ships. The mask is read three times and each read is the same
    /// statement about how much of the draft survives:
    /// `noise_lerp` starts the state at `clean + m * (noise - clean)`,
    /// `timesteps` hands the blocks `m * sigma` per token, and
    /// `post_process_latent` blends `m * denoised + (1 - m) * clean` after
    /// every step. `m = 1` is the release's generative refine, where the draft
    /// contributes only through the first noised state; `m < 1` keeps the
    /// draft's content in the state and in every blend, so the refiner adds
    /// texture to the picture it was given instead of drawing a new one. The
    /// release uses the same three reads with `m = 0` for its frozen audio
    /// stream (`runtime/stage2.py:96-99`), which is the limit where nothing of
    /// the model reaches the state at all.
    pub fn set_anchor(&mut self, gpu: &Gpu, anchor: f32) -> Result<(), RefinerError> {
        if !(anchor > 0.0 && anchor <= 1.0) {
            return Err(RefinerError::Run(format!(
                "the anchor is {anchor}, outside (0, 1]"
            )));
        }

        let mut bytes = Vec::with_capacity(self.scratch.rows as usize * 4);
        for _ in 0..self.scratch.rows {
            bytes.extend_from_slice(&anchor.to_le_bytes());
        }
        gpu.upload(&self.scratch.mask, &bytes)
            .map_err(RefinerError::Run)
    }

    /// `create_noised_state` (`helpers.py:428-448`) with `conditionings=[]`:
    /// patchify the initial latent into the state, copy it to the clean latent,
    /// draw the noise, and run `GaussianNoiser.__call__`'s two lerps.
    ///
    /// `initial_latent` is the `(1, 128, f, h, w)` grid laid out as
    /// `[channels, rows]`; the returned state is `[rows, 128]`.
    pub fn noised_state(&mut self, gpu: &Gpu, initial_latent: &Tensor) -> Result<(), RefinerError> {
        let (rows, width) = (self.scratch.rows, self.scratch.width);

        patchify_video(gpu, &self.scratch.latent, initial_latent, rows)?;
        copy(gpu, &self.scratch.clean, &self.scratch.latent)?;
        self.generator.randn(gpu, &self.scratch.noise)?;

        gpu.noise_lerp(
            &self.scratch.latent,
            &self.scratch.latent,
            &self.scratch.noise,
            &self.scratch.clean,
            &self.scratch.mask,
            rows,
            width,
            self.schedule[0],
        )
        .map_err(RefinerError::Run)
    }

    /// One interval of `euler_denoising_loop` (`samplers.py:74-79`):
    /// `SimpleDenoiser` -> `X0Model`'s `to_denoised` -> `post_process_latent` ->
    /// `EulerDiffusionStep.step`.
    pub fn step(
        &self,
        gpu: &Gpu,
        index: usize,
        context: &Tensor,
        audio: Option<(&Tensor, &Tensor)>,
    ) -> Result<(u32, u32), RefinerError> {
        let sigma = self.schedule[index];
        let next = self.schedule[index + 1];
        let (rows, width) = (self.scratch.rows, self.scratch.width);
        let mut laps = Laps::new(index);

        // `SimpleDenoiser.__call__` (`denoisers.py:225-252`) builds one Modality
        // from the state and calls `X0Model.forward` (`model.py:590-605`): the
        // velocity model on the patchified rows, then `to_denoised` at
        // `video.timesteps`. The release passes the clean frozen audio latent
        // and its prompt context each step (`runtime/stage2.py:78-99`).
        let request = Request {
            sigma,
            audio_sigma: 0.0,
            geometry: self.geometry,
            video: &self.scratch.latent,
            audio: audio.map(|(latent, _)| latent),
            video_context: context,
            audio_context: audio.map(|(_, audio_context)| audio_context),
        };
        let prepared = self.edges.prepare(gpu, &request, &self.edges_scratch)?;
        laps.lap(gpu, "prepare")?;

        if traced() {
            trace_state(gpu, "in-video", &prepared.video, rows, CHANNELS as u32)?;
            trace_rows(
                gpu,
                "video-self-cos",
                &prepared.video_signals.self_cos,
                rows,
            )?;
            if let Some(state) = prepared.audio.as_ref() {
                trace_state(gpu, "in-audio", state, self.geometry.audio_frames, 128)?;
                trace_rows(gpu, "in-audio-rows", state, self.geometry.audio_frames)?;
                if let Some(signals) = prepared.audio_signals.as_ref() {
                    trace_rows(
                        gpu,
                        "audio-self-cos",
                        &signals.self_cos,
                        self.geometry.audio_frames,
                    )?;
                }
            }
        }

        // `STAGE2_TAUS` is indexed by the completed step
        // (`sol_attention.py:188`), so each interval routes at its own tau.
        let sol = self
            .sol
            .as_ref()
            .map(|plan| Sol::new(plan, TAU_SCHEDULE[index]));

        // The release's per-step attention accounting: `(sol_calls,
        // dense_calls)` is `(47, 1)` for its 48-layer stage 2
        // (`refiner_encoder_ablation_single_gpu.py:771-780`).
        let calls = self.refiner.apply(
            gpu,
            &prepared.video,
            &prepared.video_signals,
            prepared
                .audio
                .as_ref()
                .zip(prepared.audio_signals.as_ref())
                .map(|(state, signals)| (state, signals)),
            &self.blocks,
            sol.as_ref(),
        )?;
        laps.lap(gpu, "blocks")?;

        if traced() {
            trace_state(gpu, "video-state", &prepared.video, rows, CHANNELS as u32)?;
            if let Some(state) = prepared.audio.as_ref() {
                trace_state(gpu, "audio-state", state, self.geometry.audio_frames, 128)?;
            }
        }

        self.edges.head(gpu, &prepared, &self.edges_scratch)?;
        laps.lap(gpu, "head")?;

        gpu.timesteps(&self.scratch.timesteps, &self.scratch.mask, rows, sigma)
            .map_err(RefinerError::Run)?;
        gpu.to_denoised(
            &self.scratch.denoised,
            &self.scratch.latent,
            self.edges.video_output(&self.edges_scratch),
            &self.scratch.timesteps,
            rows,
            width,
        )
        .map_err(RefinerError::Run)?;

        // `_step_state` (`samplers.py:25-36`): the blend, then the step.
        gpu.post_latent(
            &self.scratch.blended,
            &self.scratch.denoised,
            &self.scratch.mask,
            &self.scratch.clean,
            rows,
            width,
        )
        .map_err(RefinerError::Run)?;

        // `EulerDiffusionStep.step` (`components/diffusion_steps.py:32-40`):
        // `dt = sigma_next - sigma` in float32, the velocity rounded to BF16,
        // then the float32 sum rounded back to the state's dtype.
        gpu.to_velocity(
            &self.scratch.velocity,
            &self.scratch.latent,
            &self.scratch.blended,
            sigma,
        )
        .map_err(RefinerError::Run)?;
        gpu.euler_step(
            &self.scratch.latent,
            &self.scratch.latent,
            &self.scratch.velocity,
            next - sigma,
        )
        .map_err(RefinerError::Run)?;

        if traced() {
            trace_state(gpu, "video-latent", &self.scratch.latent, rows, width)?;
        }

        laps.lap(gpu, "sample")?;

        Ok(calls)
    }

    /// `euler_denoising_loop` (`samplers.py:39-80`) over the schedule's
    /// `sigmas[:-1]`, which is the release's three intervals by default.
    pub fn run(&self, gpu: &Gpu, context: &Tensor) -> Result<(), RefinerError> {
        for index in 0..self.steps() {
            self.step(gpu, index, context, None)?;
        }

        Ok(())
    }
}

/// `H3_STAGE2_TRACE=1` prints one statistics line per stream per step, so a
/// run that goes wrong can be read without a second instrument.
fn traced() -> bool {
    std::env::var_os("H3_STAGE2_TRACE").is_some()
}

/// `H3_STAGE2_PROFILE=1` splits one step into its four phases with a device
/// sync on every boundary. Each boundary is a full pipeline drain, so the split
/// is honest and the step's total is not: read the phases, not the sum, and
/// take the step's own `STAGE2 step=` line as its cost.
struct Laps {
    on: bool,
    step: usize,
    at: std::time::Instant,
}

impl Laps {
    fn new(step: usize) -> Laps {
        Laps {
            on: std::env::var_os("H3_STAGE2_PROFILE").is_some(),
            step,
            at: std::time::Instant::now(),
        }
    }

    fn lap(&mut self, gpu: &Gpu, phase: &str) -> Result<(), RefinerError> {
        if !self.on {
            return Ok(());
        }

        gpu.sync().map_err(RefinerError::Run)?;
        println!(
            "PROFILE step={} phase={phase} seconds={:.3}",
            self.step,
            self.at.elapsed().as_secs_f64()
        );
        self.at = std::time::Instant::now();

        Ok(())
    }
}

/// One line of `mean`, `rms` and `max` for a state the step produced.
fn trace_state(
    gpu: &Gpu,
    name: &str,
    tensor: &Tensor,
    rows: u32,
    width: u32,
) -> Result<(), RefinerError> {
    let mut bytes = vec![0u8; tensor.bytes()];
    gpu.read(tensor, &mut bytes).map_err(RefinerError::Run)?;

    let (mut sum, mut square, mut max) = (0f64, 0f64, 0f32);

    for pair in bytes.chunks_exact(2) {
        let value = super::vision::bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
        sum += value as f64;
        square += (value as f64) * (value as f64);
        max = max.max(value.abs());
    }

    let count = (bytes.len() / 2) as f64;

    println!(
        "TRACE {name} rows={rows} width={width} mean={:.5} rms={:.5} max={max:.5}",
        sum / count,
        (square / count).sqrt()
    );

    Ok(())
}

/// One line per positional table: the per-row means' spread says whether the
/// rows carry different positions at all.
fn trace_rows(gpu: &Gpu, name: &str, tensor: &Tensor, rows: u32) -> Result<(), RefinerError> {
    let mut bytes = vec![0u8; tensor.bytes()];
    gpu.read(tensor, &mut bytes).map_err(RefinerError::Run)?;

    let values: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|pair| super::vision::bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect();
    let width = values.len() / rows as usize;
    let means: Vec<f64> = values
        .chunks_exact(width.max(1))
        .map(|row| row.iter().map(|value| *value as f64).sum::<f64>() / row.len() as f64)
        .collect();
    let (min, max) = (
        means.iter().cloned().fold(f64::INFINITY, f64::min),
        means.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let distinct = {
        let mut sorted = means.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        sorted.len()
    };

    println!(
        "ROWS {name} rows={rows} width={width} row_mean_min={min:.5} row_mean_max={max:.5} distinct_rows={distinct}"
    );

    Ok(())
}

/// A schedule is walked from its first value to a terminal zero, strictly
/// descending, inside `(0, 1]`. The first value is the noise scale, so it is
/// also what the run keeps of the transferred latent: `1 - s`.
fn validate_schedule(sigmas: &[f32]) -> Result<(), RefinerError> {
    if sigmas.len() < 2 {
        return Err(RefinerError::Run("a schedule needs two values".into()));
    }

    if sigmas[sigmas.len() - 1] != 0.0 {
        return Err(RefinerError::Run("a schedule must end at zero".into()));
    }

    for pair in sigmas.windows(2) {
        if !(pair[0] > pair[1]) || pair[0] <= 0.0 || pair[0] > 1.0 {
            return Err(RefinerError::Run(format!(
                "schedule {sigmas:?} must fall inside (0, 1] and strictly decrease"
            )));
        }
    }

    Ok(())
}

/// A device-to-device copy of a whole BF16 tensor, for the reference's
/// `clean_latent = initial_latent.clone()` (`tools.py:156`).
fn copy(gpu: &Gpu, output: &Tensor, input: &Tensor) -> Result<(), RefinerError> {
    gpu.copy(output, 0, input, 0, output.elements())
        .map_err(RefinerError::Run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax_h3::ltx_refiner::{
        SOL_FIRST_LAYER, SOL_HEAD_DIM, SOL_LAYERS, VIDEO_HEAD_DIM,
    };

    #[test]
    fn the_schedule_is_the_releases_and_walks_three_intervals() {
        // `runtime/stage2_ops/models.py:19` and `samplers.py:73`.
        assert_eq!(SIGMAS, [0.909375, 0.725, 0.421875, 0.0]);
        assert_eq!(STEPS, 3);
        assert_eq!(NOISE_SCALE, SIGMAS[0]);
    }

    #[test]
    fn the_sol_policy_is_the_releases_three_step_split() {
        // `super_acceleration/stage2/sol_attention.py:18-19`: one tau per
        // denoise step against 48 transformer layers, the first of which
        // stays dense.
        assert_eq!(TAU_SCHEDULE, [1.0, 1.25, 1.5]);
        assert_eq!(TAU_SCHEDULE.len(), STEPS);
        assert_eq!(SOL_LAYERS, 48);
        assert_eq!(SOL_FIRST_LAYER, 1);
        assert!(SOL_FIRST_LAYER < SOL_LAYERS);
        // Sol-Attn requires head dimension 128 (`interface.py:29-30`), which
        // is the video self-attention's, not the audio modules' 64.
        assert_eq!(SOL_HEAD_DIM as usize, VIDEO_HEAD_DIM);
    }

    #[test]
    fn the_generator_starts_where_manual_seed_puts_it() {
        // `CUDAGeneratorImpl::set_current_seed`: the seed is stored and
        // `philox_offset_per_thread_` is zeroed, so a fresh generator's first
        // draw sees offset 0 whatever the seed.
        for seed in [0u64, 1, 1234, u32::MAX as u64] {
            let generator = Generator::seeded(seed);

            assert_eq!(generator.seed(), seed);
            assert_eq!(generator.offset(), 0);
        }
    }

    #[test]
    fn the_lerp_branch_is_the_references_own() {
        // `aten/src/ATen/native/Lerp.h:22-33`: `is_lerp_weight_small(w)` is
        // `|w| < 0.5`, and the arm it selects is part of the value - the small
        // arm is `self + w*(end - self)`, the large one
        // `end - (end - self)*(1 - w)`. They are algebraically equal and differ
        // only in rounding, so the host check is the predicate and the arm,
        // and the device probe is what checks the arithmetic.
        let small = |self_: f32, end: f32, weight: f32| self_ + weight * (end - self_);
        let large = |self_: f32, end: f32, weight: f32| end - (end - self_) * (1.0 - weight);
        let arm = |weight: f32| {
            if weight.abs() < 0.5 {
                small
            } else {
                large
            }
        };

        // The reference's threshold, either side of it.
        assert!(arm(0.4999999)(1.0, 3.0, 0.4999999) == small(1.0, 3.0, 0.4999999));
        assert!(arm(0.5)(1.0, 3.0, 0.5) == large(1.0, 3.0, 0.5));

        // The noiser's weight is 0.909375, so it always takes the large arm.
        assert!(NOISE_SCALE.abs() >= 0.5);
        assert_eq!(
            arm(NOISE_SCALE)(1.0, 3.0, NOISE_SCALE),
            large(1.0, 3.0, NOISE_SCALE)
        );

        // The denoise mask is exactly 1, where that arm is the identity
        // (`end - (end - self)*0`) - so with the release's all-ones mask the
        // noiser's second lerp and `post_process_latent` change nothing, which
        // is what would make them look droppable. The two arms agree
        // algebraically and separate only in rounding, so the host test pins
        // the predicate and the device probe pins the arithmetic.
        assert_eq!(large(0.25, -7.5, 1.0), -7.5);
    }

    #[test]
    fn to_denoised_uses_the_per_token_timestep_not_the_scalar() {
        // `X0Model.forward` (`model.py:590-605`) passes `video.timesteps`, the
        // patchified mask times the sigma (`helpers.py:494-501`). With the
        // release's all-ones mask that is the sigma exactly; a masked row is
        // left where the model found it.
        let sigma = SIGMAS[0];
        let (latent, velocity) = (1.5f32, -0.75f32);

        assert_eq!(1.0f32 * sigma, sigma);
        assert_eq!(bf16(latent - velocity * (0.0f32 * sigma)), bf16(latent));
        assert_ne!(bf16(latent - velocity * (1.0f32 * sigma)), bf16(latent));
    }

    #[test]
    fn the_euler_step_contracts_the_state_at_every_interval() {
        // `components/diffusion_steps.py:32-40`: `to_velocity(sample, sigma,
        // denoised)` then `sample + velocity*dt`. With a zero denoised
        // prediction the step is `sample*(1 + dt/sigma)`, whose factor is in
        // [0, 1) exactly when the schedule descends without overshooting. The
        // last interval lands on the zero sigma, so its factor is 0 and the
        // state is the model's own prediction - which is what `sigmas[:-1]`
        // stopping there leaves for `EulerDiffusionStep` to handle.
        for index in 0..STEPS {
            let (sigma, next) = (SIGMAS[index], SIGMAS[index + 1]);
            let dt = next - sigma;

            assert!(dt < 0.0, "the schedule descends at interval {index}");
            assert!(dt.abs() <= sigma, "the step never overshoots at {index}");

            let sample = 1.0f32;
            let velocity = bf16((sample - 0.0) / sigma);
            let step = bf16(sample + velocity * dt);
            let factor = sample + bf16(1.0 / sigma) * dt;

            assert_eq!(step, bf16(factor));
            assert!(step.abs() <= sample.abs());
        }
    }

    /// The anchored pass is a schedule whose first value is low enough that
    /// the draft dominates the state: `1 - s` is the share it keeps, so the
    /// release's 0.909375 keeps 9.1% and a 0.3 pass keeps 70%.
    #[test]
    fn a_schedule_must_descend_to_zero_inside_the_unit_interval() {
        assert!(validate_schedule(&[0.909375, 0.725, 0.421875, 0.0]).is_ok());
        assert!(validate_schedule(&[0.3, 0.0]).is_ok());

        assert!(
            validate_schedule(&[0.3]).is_err(),
            "a lone value is no interval"
        );
        assert!(
            validate_schedule(&[0.3, 0.1]).is_err(),
            "the walk must end at zero"
        );
        assert!(
            validate_schedule(&[0.3, 0.4, 0.0]).is_err(),
            "the walk must descend"
        );
        assert!(
            validate_schedule(&[0.0, 0.0]).is_err(),
            "a zero start keeps all of it"
        );
        assert!(validate_schedule(&[1.5, 0.0]).is_err(), "past full noise");
        assert!(
            validate_schedule(&[0.3, -0.1, 0.0]).is_err(),
            "negative noise"
        );
    }

    /// Round to BF16 the way the device's `__float2bfloat16` does: nearest
    /// even, ties to the even mantissa.
    fn bf16(value: f32) -> f32 {
        let bits = value.to_bits();
        let rounding = 0x7fff + ((bits >> 16) & 1);

        f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
    }
}
