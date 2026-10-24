//! Keyframe conditioning: the anchor rows that occupy the leading spans of the
//! DiT's video and audio inputs.
//!
//! Reference: `h3.c:315-322` `h3_augment_span` and `h3.c:323-368`
//! `h3_augment_conditions`. An anchor image is encoded by the video VAE
//! (`vae_encoder.rs`, `[3,1,H,W]` to `[24,1,h,w]`), patchified to 96-wide rows
//! (`h3_dit.c:3077-3097`), then jittered with `0.999 * value + 0.001 * normal`
//! from a generator seeded by the request seed. The reference rebuilds that
//! generator for each span, so two anchors of the same request see the same
//! jitter sequence; that is reproduced here rather than shared.
//!
//! The row counts are fixed by the packed layout, not by the caller:
//! `img_cond_rows * 96` and `audio_cond_rows * 32` (`h3_dit.c:1617-1624`), and
//! the DiT validates them again before it writes (`:2420-2428`).

use super::dit::{patchify_video, DitError};
use super::host::Rng;
use super::vae_encoder::{Latent, LATENT_CHANNELS};

/// `VIDEO_CHANNELS` (`h3_dit.c:22`): the latent channels, four per row patch.
const VIDEO_PATCH: usize = 4;

/// The row width one patch occupies: channels times the 2x2 patch
/// (`h3_dit.c:3077-3097`).
pub const VIDEO_ROW_WIDTH: usize = LATENT_CHANNELS * VIDEO_PATCH;

/// `h3.c:319-321`: the released jitter. The generator is created here, so each
/// call starts its sequence over, exactly as `h3_augment_span` does.
pub fn augment_span(values: &mut [f32], seed: u64) {
    let mut rng = Rng::seed(seed);

    for value in values.iter_mut() {
        *value = 0.999 * *value + 0.001 * rng.normal();
    }
}

/// The condition rows one anchor latent contributes: patchified, then jittered
/// in place (`h3.c:1384-1406`, `:319-321`).
pub fn anchor_rows(latent: &Latent, seed: u64) -> Result<Vec<f32>, DitError> {
    let rows = latent.time * latent.height * latent.width / VIDEO_PATCH;
    let mut values = vec![0.0f32; rows * VIDEO_ROW_WIDTH];

    patchify_video(
        &latent.values,
        LATENT_CHANNELS,
        latent.time,
        latent.height,
        latent.width,
        &mut values,
    )?;
    augment_span(&mut values, seed);

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent(height: usize, width: usize) -> Latent {
        let values = (0..LATENT_CHANNELS * height * width)
            .map(|index| index as f32 * 0.25 - 3.0)
            .collect();

        Latent {
            time: 1,
            height,
            width,
            values,
        }
    }

    /// `h3.c:319-321`: a zero span keeps only the jittered term, and that term
    /// is the release's own generator draw, not a rescaled one.
    #[test]
    fn a_zero_anchor_is_the_jitter_alone() {
        let mut values = vec![0.0f32; 16];
        augment_span(&mut values, 7);

        let mut rng = Rng::seed(7);
        for value in values {
            assert_eq!(value, 0.001 * rng.normal());
        }
    }

    /// The reference seeds a fresh generator per span (`h3.c:340`, `:352`), so
    /// two anchors of one request see the same sequence.
    #[test]
    fn each_span_is_seeded_freshly() {
        let mut first = vec![0.5f32; 8];
        let mut second = vec![0.5f32; 8];

        augment_span(&mut first, 11);
        augment_span(&mut second, 11);

        assert_eq!(first, second);
    }

    /// The rows are the patchified latent, jittered: the composition is the
    /// patchifier the forward already uses (`h3_dit.c:3077-3097`).
    #[test]
    fn the_rows_are_the_patchified_anchor_jittered() {
        let latent = latent(4, 6);
        let rows = anchor_rows(&latent, 3).expect("anchor rows");
        assert_eq!(rows.len(), 6 * VIDEO_ROW_WIDTH);

        let mut expected = vec![0.0f32; 6 * VIDEO_ROW_WIDTH];
        patchify_video(
            &latent.values,
            LATENT_CHANNELS,
            latent.time,
            latent.height,
            latent.width,
            &mut expected,
        )
        .expect("patchify");
        augment_span(&mut expected, 3);

        assert_eq!(rows, expected);
    }

    /// An odd latent axis cannot be patched, and the failure names the shape
    /// instead of returning short rows.
    #[test]
    fn an_unpatchable_anchor_is_refused() {
        assert!(anchor_rows(&latent(3, 4), 1).is_err());
    }
}
