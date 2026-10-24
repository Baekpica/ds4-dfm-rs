//! Host-side arithmetic the DiT path shares: temporal alignment, the latent
//! canvas, the seeded noise generator and the serving sigma grid.
//!
//! Source of truth is `/data/h3.c` at `eae5d54`:
//!
//! - `h3_host.c:17-41` `h3_align_frame_count`, `h3_video_latent_t`,
//!   `h3_temporal` (24 fps canvas frames, 40 fps audio latents).
//! - `h3_host.c:43-46` `h3_latent_canvas` (VAE spatial ratio 16).
//! - `h3_host.c:495-530` `h3_rng_*`, PCG32 plus Box-Muller on doubles. The
//!   released server seeds each modality from its own generator with the same
//!   requested seed (`h3.c:1574-1581`), which is why `fill_normal` is called
//!   twice on two generators rather than once on a shared one.
//! - `h3_host.c:152-167` `h3_serving_schedule_build`, the sampler grid the
//!   serving path uses: `steps` evaluations plus a terminal zero.
//!
//! The reference's older `h3_schedule_build` (`:137-150`) belongs to the RES
//! multistep sampler, which the serving path does not call; it is not ported
//! here. Deviation, named: that is a deliberate omission, not an equivalent.

/// `H3_FPS` (`h3_host.h:9`).
pub const FPS: i32 = 24;
/// `H3_AUDIO_LATENT_FPS` (`h3_host.h:10`).
pub const AUDIO_LATENT_FPS: i32 = 40;
/// `H3_CANVAS_MULTIPLE` (`h3_host.h:7`).
pub const CANVAS_MULTIPLE: i32 = 32;
/// `H3_MAX_PIXELS` (`h3_host.h:8`).
pub const MAX_PIXELS: i32 = 768 * 1344;
/// `H3_VAE_SPATIAL_RATIO` (`h3_host.h:11`).
pub const VAE_SPATIAL_RATIO: i32 = 16;
/// `H3_VIDEO_SIGMA_SHIFT` (`h3_host.h:12`).
pub const VIDEO_SIGMA_SHIFT: f32 = 12.0;
/// `H3_AUDIO_SIGMA_SHIFT` (`h3_host.h:13`).
pub const AUDIO_SIGMA_SHIFT: f32 = 3.0;
/// `H3_MAX_STEPS` (`h3_host.h:14`).
pub const MAX_STEPS: usize = 1000;

/// `h3_frame_per_token` (`h3_host.c:15`), the temporal RoPE stride cycle.
pub const FRAME_PER_TOKEN: [i32; 5] = [1, 4, 4, 4, 4];
/// `h3_frame_rescale` (`h3_host.c:16`).
pub const FRAME_RESCALE: f64 = 5.0 / 3.0;

/// `h3_align_frame_count` (`h3_host.c:17-23`): the canvas frame count is
/// `5 + 17n`, because the VAE encodes one anchor frame per 17 latent steps.
pub fn align_frame_count(requested: i32) -> i32 {
    let mut value = if requested < 5 { 5 } else { requested };
    let mut remainder = (value - 5) % 17;
    if remainder < 0 {
        remainder += 17;
    }
    if remainder != 0 {
        value += 17 - remainder;
    }
    value
}

/// `h3_video_latent_t` (`h3_host.c:25-28`).
pub fn video_latent_t(frame_count: i32) -> i32 {
    if frame_count <= 5 {
        return 2;
    }
    ((frame_count - 5) / 17) * 5 + 2
}

/// `h3_temporal` (`h3_host.c:34-41`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Temporal {
    pub frame_count: i32,
    pub video_t: i32,
    pub audio_t: i32,
}

pub fn temporal(requested_frames: i32) -> Temporal {
    let frame_count = align_frame_count(requested_frames);
    Temporal {
        frame_count,
        video_t: video_latent_t(frame_count),
        audio_t: (((frame_count as f64) * (AUDIO_LATENT_FPS as f64) / (FPS as f64)).round()) as i32,
    }
}

/// `h3_latent_canvas` (`h3_host.c:43-46`), returning `(latent_w, latent_h)` in
/// the order the reference writes them.
pub fn latent_canvas(width: i32, height: i32) -> (i32, i32) {
    (width / VAE_SPATIAL_RATIO, height / VAE_SPATIAL_RATIO)
}

/// `h3_rng` (`h3_host.h:88-93`), PCG32 with the Box-Muller spare.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    increment: u64,
    spare: f32,
    has_spare: bool,
}

impl Rng {
    /// `h3_rng_seed` (`h3_host.c:504-511`): four draws prime the state, exactly
    /// as the reference does, so the sequence matches from the first value.
    pub fn seed(seed: u64) -> Rng {
        let mut rng = Rng {
            state: 0,
            increment: (seed << 1) | 1,
            spare: 0.0,
            has_spare: false,
        };
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(seed ^ 0x9e37_79b9_7f4a_7c15);
        let _ = rng.next_u32();
        rng
    }

    /// `h3_rng_u32` (`h3_host.c:495-502`): an LCG step, then an xorshifted
    /// output rotated by the top five bits of the previous state.
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.increment);
        let shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rotation = (old_state >> 59) as u32;
        shifted.rotate_right(rotation)
    }

    /// `h3_rng_normal` (`h3_host.c:513-525`): Box-Muller in F64, one spare.
    pub fn normal(&mut self) -> f32 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        let u1 = ((self.next_u32() as f64) + 1.0) / 4294967297.0;
        let u2 = ((self.next_u32() as f64) + 0.5) / 4294967296.0;
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        self.spare = (radius * angle.sin()) as f32;
        self.has_spare = true;
        (radius * angle.cos()) as f32
    }

    /// `h3_rng_fill_normal` (`h3_host.c:526-530`).
    pub fn fill_normal(&mut self, values: &mut [f32]) {
        for value in values.iter_mut() {
            *value = self.normal();
        }
    }
}

/// `h3_sigma_schedule` (`h3_host.h:82-86`): `steps` evaluations and one
/// terminal zero, per modality.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    pub steps: usize,
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

impl Schedule {
    /// `h3_serving_schedule_build` (`h3_host.c:152-167`).
    ///
    /// The reference computes the shifted sigma in F32 from a linear base
    /// grid, and the expression order is kept: `s * base / (1 + (s - 1) *
    /// base)`. `steps == evaluations` and both arrays end in a terminal zero.
    pub fn serving(evaluations: usize) -> Option<Schedule> {
        if !(2..=MAX_STEPS).contains(&evaluations) {
            return None;
        }
        let denominator = evaluations as f32;
        let mut video = vec![0.0f32; evaluations + 1];
        let mut audio = vec![0.0f32; evaluations + 1];

        for index in 0..=evaluations {
            let base = 1.0f32 - (index as f32) / denominator;
            video[index] =
                VIDEO_SIGMA_SHIFT * base / (1.0f32 + (VIDEO_SIGMA_SHIFT - 1.0f32) * base);
            audio[index] =
                AUDIO_SIGMA_SHIFT * base / (1.0f32 + (AUDIO_SIGMA_SHIFT - 1.0f32) * base);
        }

        Some(Schedule {
            steps: evaluations,
            video,
            audio,
        })
    }
}

/// `h3_dit.c:2537-2554` `h3_dit_reuse_schedule`: with interval 1 every step
/// evaluates; otherwise step 0, the last step and every multiple evaluate.
///
/// A step that does not evaluate reuses the last velocity, extrapolated from
/// the two most recent evaluations.
pub fn reuse_schedule(steps: usize, interval: usize) -> Option<Vec<bool>> {
    if steps < 1 || !(1..=32).contains(&interval) {
        return None;
    }

    let mut selected = vec![false; steps];
    for (step, flag) in selected.iter_mut().enumerate() {
        *flag = interval == 1 || step == 0 || step == steps - 1 || step.is_multiple_of(interval);
    }
    Some(selected)
}

/// `h3_dit.c:2512-2523` `extrapolation_ratio`, including the +-2 clamp that
/// keeps a malformed reuse window from turning one evaluation into a blow-up.
pub fn extrapolation_ratio(
    current_sigma: f32,
    last_sigma: f32,
    previous_sigma: f32,
    have_previous: bool,
) -> f32 {
    if !have_previous {
        return 0.0;
    }

    let denominator = last_sigma - previous_sigma;
    let ratio = if denominator != 0.0 {
        (current_sigma - last_sigma) / denominator
    } else {
        0.0
    };

    // The reference writes this as two comparisons; `clamp` is the same
    // function on finite values and passes NaN through as they do.
    ratio.clamp(-2.0, 2.0)
}

/// `h3_dit.c:2525-2535` `extrapolate_velocity`.
///
/// The reference's host build is `-std=c11` (`/data/h3.c` Makefile:4), so GCC
/// contracts no floating-point expression; `last + ratio * delta` is written
/// as two roundings here as well, which `-ffp-contract=off` guarantees there.
pub fn extrapolate_velocity(
    output: &mut [f32],
    last: &[f32],
    previous: &[f32],
    current_sigma: f32,
    last_sigma: f32,
    previous_sigma: f32,
    have_previous: bool,
) {
    if !have_previous {
        output.copy_from_slice(last);
        return;
    }

    let ratio = extrapolation_ratio(current_sigma, last_sigma, previous_sigma, true);
    for (index, value) in output.iter_mut().enumerate() {
        let last_value = last[index];
        *value = last_value + ratio * (last_value - previous[index]);
    }
}

/// `h3_host.c:675-683` `h3_euler_velocity_step`: `sample += delta * velocity`
/// with `delta = sigma - sigma_next`. Rejects a non-decreasing or negative
/// pair exactly where the reference does.
pub fn euler_step(sample: &mut [f32], velocity: &[f32], sigma: f32, sigma_next: f32) -> bool {
    if sample.len() != velocity.len()
        || !sigma.is_finite()
        || !sigma_next.is_finite()
        || sigma <= sigma_next
        || sigma_next < 0.0
    {
        return false;
    }

    let delta = sigma - sigma_next;
    for (value, velocity) in sample.iter_mut().zip(velocity) {
        *value += delta * velocity;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `h3_host.c:17-23`: the canvas floor is 5 frames and the stride is 17.
    #[test]
    fn frames_align_to_five_plus_seventeen() {
        assert_eq!(align_frame_count(1), 5);
        assert_eq!(align_frame_count(5), 5);
        assert_eq!(align_frame_count(6), 22);
        assert_eq!(align_frame_count(22), 22);
        assert_eq!(align_frame_count(23), 39);
        assert_eq!(align_frame_count(362), 362);
    }

    /// `h3_host.c:25-28`: two latents for one latent block, then five per 17.
    #[test]
    fn latent_time_follows_the_anchor_cycle() {
        assert_eq!(video_latent_t(5), 2);
        assert_eq!(video_latent_t(22), 7);
        assert_eq!(video_latent_t(362), 107);
    }

    /// `h3_host.c:34-41` on the smallest legal clip: 5 frames, 2 video
    /// latents, 8 audio latents (40/24 fps rounded).
    #[test]
    fn temporal_of_the_smallest_clip() {
        assert_eq!(
            temporal(5),
            Temporal {
                frame_count: 5,
                video_t: 2,
                audio_t: 8
            }
        );
    }

    /// `h3_host.c:43-46`.
    #[test]
    fn canvas_divides_by_sixteen() {
        assert_eq!(latent_canvas(256, 256), (16, 16));
        assert_eq!(latent_canvas(1365, 768), (85, 48));
    }

    /// Golden from `tools/minimax_h3/dit_probe.c`: seed 1, the smallest clip's
    /// video latent (24 x 2 x 16 x 16) and audio latent (32 x 2 x 8). Two
    /// generators, same seed, as `h3.c:1574-1581` does.
    #[test]
    fn seeded_noise_matches_the_reference() {
        let mut video_rng = Rng::seed(1);
        let mut video = vec![0.0f32; 24 * 2 * 16 * 16];
        video_rng.fill_normal(&mut video);

        let mut audio_rng = Rng::seed(1);
        let mut audio = vec![0.0f32; 32 * 2 * 8];
        audio_rng.fill_normal(&mut audio);

        assert_eq!(fnv(&video), 0x5691_802b_dd5b_0074);
        assert_eq!(fnv(&audio), 0x094d_7198_5b0c_77d7);
        assert_eq!(video[0].to_bits(), 0xbf7b_c660);
        assert_eq!(video[video.len() - 1].to_bits(), 0x3e75_1030);
    }

    /// `h3_host.c:152-167` at two evaluations, as bit patterns. Step 0 is
    /// sigma 1 in both modalities; the probe's `SIGMAS` line is the golden.
    #[test]
    fn serving_grid_bits_at_two_steps() {
        let schedule = Schedule::serving(2).expect("two evaluations are legal");
        let bits: Vec<u32> = schedule.video.iter().map(|value| value.to_bits()).collect();
        let audio: Vec<u32> = schedule.audio.iter().map(|value| value.to_bits()).collect();

        assert_eq!(bits, vec![0x3f80_0000, 0x3f6c_4ec5, 0x0000_0000]);
        assert_eq!(audio, vec![0x3f80_0000, 0x3f40_0000, 0x0000_0000]);
    }

    /// `h3_host.c:152-167`: the grid is only defined from two evaluations.
    #[test]
    fn serving_grid_rejects_too_few_evaluations() {
        assert!(Schedule::serving(0).is_none());
        assert!(Schedule::serving(1).is_none());
        assert!(Schedule::serving(MAX_STEPS + 1).is_none());
    }

    /// `h3_dit.c:2537-2554`: 20 steps at interval 3 evaluate 0, 3, 6, 9, 12,
    /// 15, 18 and the last step 19.
    #[test]
    fn reuse_keeps_first_last_and_multiples() {
        let selected = reuse_schedule(20, 3).expect("interval 3 is legal");
        let evaluated: Vec<usize> = selected
            .iter()
            .enumerate()
            .filter(|(_, flag)| **flag)
            .map(|(step, _)| step)
            .collect();
        assert_eq!(evaluated, vec![0, 3, 6, 9, 12, 15, 18, 19]);

        assert!(reuse_schedule(4, 1)
            .expect("1 evaluates")
            .iter()
            .all(|f| *f));
        let coarse = reuse_schedule(5, 32).expect("32 is legal");
        assert_eq!(coarse, vec![true, false, false, false, true]);
    }

    /// `h3_dit.c:2537-2539`: out-of-contract arguments are refused.
    #[test]
    fn reuse_rejects_out_of_contract_arguments() {
        assert!(reuse_schedule(0, 1).is_none());
        assert!(reuse_schedule(2, 0).is_none());
        assert!(reuse_schedule(2, 33).is_none());
    }

    /// `h3_dit.c:2512-2523`: no second evaluation, a flat window and the clamp.
    #[test]
    fn extrapolation_ratio_covers_its_guards() {
        assert_eq!(extrapolation_ratio(0.5, 0.4, 0.2, false), 0.0);
        assert_eq!(extrapolation_ratio(0.5, 0.4, 0.4, true), 0.0);
        assert_eq!(extrapolation_ratio(0.375, 0.25, 0.0, true), 0.5);
        assert_eq!(extrapolation_ratio(-1.0, 0.4, 0.2, true), -2.0);
        assert_eq!(extrapolation_ratio(1.0, 0.4, 0.2, true), 2.0);
    }

    /// `h3_dit.c:2525-2535`: the first reuse copies; later ones extrapolate.
    #[test]
    fn extrapolate_velocity_copies_then_extends() {
        let mut output = [0.0f32; 2];
        extrapolate_velocity(
            &mut output,
            &[2.0, 4.0],
            &[1.0, 1.0],
            0.375,
            0.25,
            0.0,
            false,
        );
        assert_eq!(output, [2.0, 4.0]);

        extrapolate_velocity(
            &mut output,
            &[2.0, 4.0],
            &[1.0, 1.0],
            0.375,
            0.25,
            0.0,
            true,
        );
        assert_eq!(output, [2.5, 5.5]);
    }

    /// `h3_host.c:675-683`: multiply, then add, on an exactly representable
    /// step; and the sigma guards that reject a non-decreasing pair.
    #[test]
    fn euler_step_multiplies_then_adds() {
        let mut sample = [1.0f32, -2.0];
        assert!(euler_step(&mut sample, &[0.25, 3.0], 1.0, 0.75));
        assert_eq!(sample, [1.0625, -1.25]);

        assert!(!euler_step(&mut sample, &[0.25, 3.0], 0.5, 0.5));
        assert!(!euler_step(&mut sample, &[0.25, 3.0], 0.5, -0.25));
        assert!(!euler_step(&mut sample, &[0.25], 1.0, 0.5));
    }

    fn fnv(values: &[f32]) -> u64 {
        let mut hash = 1_469_598_103_934_665_603u64;
        for value in values {
            for byte in value.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(1_099_511_628_211);
            }
        }
        hash
    }
}
