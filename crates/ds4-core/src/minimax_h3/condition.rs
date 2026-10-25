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

use std::path::Path;

use super::dit::{patchify_video, DitError};
use super::host::Rng;
use super::multimodal;
use super::text_forward::{self, Embedding};
use super::tokenizer::Tokenizer;
use super::vae_encoder::{Latent, LATENT_CHANNELS};
use super::vision::Grid;
use super::vision_forward;

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

/// One anchored image at the render size, as the image reader produced it.
/// The Qwen vision tower and the video VAE encoder both read these pixels
/// (`h3.c:1231-1252`, `:1345-1397`), so one read feeds both.
pub struct AnchorPixels {
    pub values: Vec<f32>,
    pub height: u32,
    pub width: u32,
}

/// The FL2VA prompt embedding of a request that carries anchors.
///
/// The reference runs the Qwen vision tower over every anchor and splices the
/// features into the prompt as `<Picture n>:` spans before the text encoder
/// (`h3.c:1345-1414`). With no anchors it takes the plain text path
/// (`h3.c:1424-1436`), which is why callers use this only when one exists.
pub fn anchor_embedding(
    text_encoder: &Path,
    tokenizer: &Tokenizer,
    prompt: &str,
    anchors: &[AnchorPixels],
) -> Result<Embedding, String> {
    let mut outputs = Vec::with_capacity(anchors.len());

    for anchor in anchors {
        let grid = Grid::new(1, anchor.height as usize, anchor.width as usize)
            .map_err(|error| format!("vision grid: {}", error.token()))?;
        outputs.push(
            vision_forward::encode_dir(text_encoder, &anchor.values, 1, grid)
                .map_err(|error| format!("vision: {}", error.token()))?,
        );
    }

    let images: Vec<&vision_forward::VisionOutput> = outputs.iter().collect();
    let presentation = multimodal::fl2va(tokenizer, prompt, &images)
        .map_err(|error| format!("presentation: {}", error.token()))?;

    text_forward::encode_multimodal(
        text_encoder,
        presentation.ids(),
        presentation.spans(),
        presentation.positions(),
        presentation.tags(),
    )
    .map_err(|error| format!("text encoder: {}", error.token()))
}

/// One prepared Ref2VA reference as the prompt embedding needs it.
///
/// The visual half is the decoded pixels the tower reads: one image, or every
/// decoded frame of a video (`h3.c:1345-1414`). Audio references carry no
/// pixels; their soundtrack reaches the DiT as audio rows instead.
#[derive(Debug, Clone)]
pub struct ReferencePixels {
    pub presentation: multimodal::ReferenceKind,
    pub has_audio: bool,
    pub pixels: Vec<f32>,
    pub height: u32,
    pub width: u32,
    pub frames: usize,
    pub timestamps: Vec<f64>,
}

/// The Ref2VA prompt embedding: every reference's Qwen vision output in
/// request order, spliced as `<Picture n>: ` / `<Video n>: <t seconds>` spans
/// before the prompt (`h3_multimodal.c:243-360`).
pub fn reference_embedding(
    text_encoder: &Path,
    tokenizer: &Tokenizer,
    prompt: &str,
    references: &[ReferencePixels],
) -> Result<Embedding, String> {
    // `h3.c:1345-1414`: an image is one tower call; a video runs one call per
    // two-frame pair extracted from its channel-major pixels.
    let mut outputs: Vec<Vec<vision_forward::VisionOutput>> = Vec::with_capacity(references.len());
    for reference in references {
        match reference.presentation {
            multimodal::ReferenceKind::Audio => outputs.push(Vec::new()),
            multimodal::ReferenceKind::Image => {
                let grid = Grid::new(1, reference.height as usize, reference.width as usize)
                    .map_err(|error| format!("vision grid: {}", error.token()))?;
                outputs.push(vec![vision_forward::encode_dir(
                    text_encoder,
                    &reference.pixels,
                    1,
                    grid,
                )
                .map_err(|error| format!("vision: {}", error.token()))?]);
            }
            multimodal::ReferenceKind::Video => {
                let grid = Grid::new(2, reference.height as usize, reference.width as usize)
                    .map_err(|error| format!("vision grid: {}", error.token()))?;
                let area = reference.height as usize * reference.width as usize;
                let samples = reference.frames.div_ceil(12);
                let mut blocks = Vec::with_capacity(reference.timestamps.len());
                for block in 0..reference.timestamps.len() {
                    // `h3.c:1367-1372`: sample 2b and 2b+1, the last one
                    // repeating when it runs past the sample count.
                    let first_sample = 2 * block;
                    let second_sample = if first_sample + 1 < samples {
                        first_sample + 1
                    } else {
                        first_sample
                    };
                    let pair = super::reference::vision_pair(
                        &reference.pixels,
                        reference.frames,
                        reference.height as usize,
                        reference.width as usize,
                        first_sample * 12,
                        second_sample * 12,
                    )
                    .map_err(|error| error.to_string())?;
                    debug_assert_eq!(pair.len(), 2 * 3 * area);
                    blocks.push(
                        vision_forward::encode_dir(text_encoder, &pair, 2, grid)
                            .map_err(|error| format!("vision: {}", error.token()))?,
                    );
                }
                outputs.push(blocks);
            }
        }
    }

    let items: Vec<multimodal::Reference<'_>> = references
        .iter()
        .zip(outputs.iter())
        .map(|(reference, vision)| multimodal::Reference {
            kind: reference.presentation,
            has_audio: reference.has_audio,
            vision,
            timestamps: &reference.timestamps,
        })
        .collect();

    let presentation = multimodal::ref2va(tokenizer, prompt, &items)
        .map_err(|error| format!("presentation: {}", error.token()))?;

    text_forward::encode_multimodal(
        text_encoder,
        presentation.ids(),
        presentation.spans(),
        presentation.positions(),
        presentation.tags(),
    )
    .map_err(|error| format!("text encoder: {}", error.token()))
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
