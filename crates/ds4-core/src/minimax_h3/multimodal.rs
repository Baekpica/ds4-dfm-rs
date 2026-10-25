//! The released `<Picture n>:` presentation around encoded vision outputs.
//!
//! Source of truth `/data/h3.c` at `eae5d54`:
//!
//! - `h3_multimodal.c:8-11` the four special ids: `<|vision_start|>` 151652,
//!   `<|vision_end|>` 151653, `<|image_pad|>` 151655, `<|video_pad|>` 151656.
//! - `h3_multimodal.c:147-166` `append_vision`: the start id, the span (whose
//!   first row is the merged output), one pad id per vision token, the end id.
//! - `h3_multimodal.c:200-232` the FL2VA order: for each image, tokenize
//!   `"<Picture %zu>: "` (one-based), append the span, and tokenize the prompt
//!   after the last one.
//! - `h3_multimodal.c:90-128` `build_positions`, the axis-major `[3, tokens]`
//!   mRoPE coordinates: text before the first span keeps its own index; a span
//!   gets `base` on every axis, then `base + row` on axis 1 and `base + column`
//!   on axis 2 over the merged grid; text after it restarts at
//!   `start + max(merged_h, merged_w) + offset`, and each span advances the
//!   offset by `max(merged_h, merged_w) - tokens`.
//! - `h3_multimodal.c:236-252` `tags`: one for every row, zeroed over
//!   `[start - 1, start + tokens + 1)`, which covers the boundary ids.
//!
//! Positions and tags are integer arithmetic and are tested here without a
//! device. The `<Picture n>: ` prefix needs the real BPE, so its ids are pinned
//! by the gate against the reference, not hardcoded here
//! (`h3_multimodal.c:221-225`).
//!
//! Ref2VA (`h3_multimodal.c:243-360`) is the ordered presentation: the
//! references are labelled in request order and the prompt follows them.

use std::fmt;

use crate::minimax_h3::tokenizer::{EmptyPrompt, Tokenizer, TokenizerError};
use crate::minimax_h3::vision_forward::VisionOutput;

/// `<|vision_start|>` (`h3_multimodal.c:10`).
pub const VISION_START: u32 = 151_652;

/// `<|vision_end|>` (`h3_multimodal.c:11`).
pub const VISION_END: u32 = 151_653;

/// `<|image_pad|>` (`h3_multimodal.c:12`).
pub const IMAGE_PAD: u32 = 151_655;

/// `<|video_pad|>` (`h3_multimodal.c:13`).
pub const VIDEO_PAD: u32 = 151_656;

/// `h3_multimodal.c:160`: the reference refuses a presentation whose id count
/// does not fit an `int`.
const MAX_TOKENS: usize = i32::MAX as usize;

/// One Qwen vision span inside the presentation (`h3_text_vision_span`,
/// `h3_text_encoder.h:24-29`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// Index of the first merged row; the id before it is `VISION_START`.
    pub start: usize,
    /// Merged rows, one pad id each after `start`.
    pub tokens: usize,
}

impl Span {
    /// One past the last merged row, where `VISION_END` sits.
    pub fn end(&self) -> usize {
        self.start + self.tokens
    }
}

/// One encoded image as the presentation and the decoder see it. The payloads
/// are the tower's own buffers, borrowed for the life of the presentation.
#[derive(Clone, Copy)]
pub struct Image<'a> {
    /// Patch-grid edges, both even and at least two
    /// (`h3_multimodal.c:84-86`).
    pub grid: (usize, usize),
    /// Merged rows the tower emitted; the presentation refuses a grid that
    /// disagrees with it (`h3_multimodal.c:116-119`).
    pub tokens: usize,
    /// Final-merger rows, `[tokens, 5120]` BF16.
    pub merged: &'a [u8],
    /// The three deepstack rows, same shape.
    pub deepstack: [&'a [u8]; 3],
}

impl<'a> Image<'a> {
    /// Borrow one tower output.
    pub fn of(output: &'a VisionOutput) -> Image<'a> {
        Image {
            grid: output.grid().rows(),
            tokens: output.tokens(),
            merged: output.merged(),
            deepstack: [
                output.deepstack(0),
                output.deepstack(1),
                output.deepstack(2),
            ],
        }
    }
}

/// The decoder's input: the ids with their vision spans, the axis-major mRoPE
/// coordinates, and the DiT modality tag of every row.
#[derive(Debug)]
pub struct Presentation<'a> {
    ids: Vec<u32>,
    spans: Vec<VisionSpan<'a>>,
    positions: Vec<u32>,
    tags: Vec<u8>,
}

impl<'a> Presentation<'a> {
    /// The token sequence the decoder runs.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The vision spans, in order, with their payloads.
    pub fn spans(&self) -> &[VisionSpan<'a>] {
        &self.spans
    }

    /// Axis-major `[3, tokens]`: axis 0, then 1, then 2
    /// (`h3_text_encoder.h:40-43`).
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// One tag per row: 0 for a vision span and its boundary ids, 1 for text.
    pub fn tags(&self) -> &[u8] {
        &self.tags
    }

    /// Rows in the presentation.
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }
}

/// One span plus the payloads the decoder splices and adds.
#[derive(Clone, Copy, Debug)]
pub struct VisionSpan<'a> {
    pub start: usize,
    pub tokens: usize,
    pub merged: &'a [u8],
    pub deepstack: [&'a [u8]; 3],
}

impl VisionSpan<'_> {
    /// One past the last merged row.
    pub fn end(&self) -> usize {
        self.start + self.tokens
    }
}

/// Why a presentation could not be built.
#[derive(Debug)]
pub enum PresentationError {
    /// `invalid FL2VA multimodal presentation arguments` (`:210`).
    Arguments,
    /// An empty prompt, or one with no image.
    Empty,
    /// The tokenizer refused the prefix or the prompt.
    Tokenizer(TokenizerError),
    /// `invalid Qwen vision span geometry` (`:87`).
    Geometry(&'static str),
    /// `Qwen merged grid does not match token span` (`:117`).
    GridMismatch,
    /// `multimodal mRoPE position overflow` (`:102`).
    PositionOverflow,
    /// `multimodal tag span overflow` (`:179`).
    TagOverflow,
    /// `multimodal presentation is too large` (`:162`).
    TooLarge,
    /// `invalid Ref2VA presentation arguments` (`h3_multimodal.c:258`).
    ReferenceArguments,
    /// `invalid Ref2VA reference presentation` (`:340`).
    Reference,
    /// `Ref2VA audio requires an image or video reference` (`:300-302`).
    ReferenceAudioOnly,
    /// `invalid Ref2VA video timestamp` (`:342`).
    Timestamp,
}

impl PresentationError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            PresentationError::Arguments => {
                "invalid FL2VA multimodal presentation arguments".to_string()
            }
            PresentationError::Empty => {
                "the presentation needs a prompt and at least one image".to_string()
            }
            PresentationError::Tokenizer(error) => format!("tokenizer: {}", error.token()),
            PresentationError::Geometry(detail) => {
                format!("invalid Qwen vision span geometry: {detail}")
            }
            PresentationError::GridMismatch => {
                "Qwen merged grid does not match token span".to_string()
            }
            PresentationError::PositionOverflow => "multimodal mRoPE position overflow".to_string(),
            PresentationError::TagOverflow => "multimodal tag span overflow".to_string(),
            PresentationError::TooLarge => "multimodal presentation is too large".to_string(),
            PresentationError::ReferenceArguments => {
                "invalid Ref2VA presentation arguments".to_string()
            }
            PresentationError::Reference => {
                "invalid Ref2VA reference presentation".to_string()
            }
            PresentationError::ReferenceAudioOnly => {
                "Ref2VA audio requires an image or video reference".to_string()
            }
            PresentationError::Timestamp => {
                "invalid Ref2VA video timestamp".to_string()
            }
        }
    }
}

impl fmt::Display for PresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for PresentationError {}

impl From<TokenizerError> for PresentationError {
    fn from(error: TokenizerError) -> PresentationError {
        PresentationError::Tokenizer(error)
    }
}

/// Build the FL2VA presentation for `prompt` and `images`
/// (`h3_multimodal.c:197-241`).
pub fn fl2va<'a>(
    tokenizer: &Tokenizer,
    prompt: &str,
    images: &[&'a VisionOutput],
) -> Result<Presentation<'a>, PresentationError> {
    if prompt.is_empty() || images.is_empty() {
        return Err(PresentationError::Empty);
    }

    let borrowed: Vec<Image<'a>> = images.iter().map(|output| Image::of(output)).collect();

    assemble(
        |text| {
            tokenizer
                .encode(text, EmptyPrompt::Nothing)
                .map_err(PresentationError::Tokenizer)
        },
        prompt,
        &borrowed,
    )
}

/// The presentation walk, with the tokenizer as a function so the id layout can
/// be tested without the BPE.
fn assemble<'a, F>(
    mut token: F,
    prompt: &str,
    images: &[Image<'a>],
) -> Result<Presentation<'a>, PresentationError>
where
    F: FnMut(&str) -> Result<Vec<u32>, PresentationError>,
{
    if prompt.is_empty() || images.is_empty() {
        return Err(PresentationError::Empty);
    }

    let mut ids = Vec::new();
    let mut spans = Vec::with_capacity(images.len());
    let mut grids = Vec::with_capacity(images.len());

    for (index, image) in images.iter().enumerate() {
        check_geometry(image)?;

        // `h3_multimodal.c:221`: `"<Picture %zu>: "`, one-based.
        ids.extend(token(&format!("<Picture {}>: ", index + 1))?);
        let span = append_vision(&mut ids, IMAGE_PAD, image.tokens);
        spans.push(VisionSpan {
            start: span.start,
            tokens: span.tokens,
            merged: image.merged,
            deepstack: image.deepstack,
        });
        grids.push(image.grid);
    }

    // `h3_multimodal.c:228`: the prompt follows the last span.
    ids.extend(token(prompt)?);

    if ids.is_empty() || ids.len() > MAX_TOKENS {
        return Err(PresentationError::TooLarge);
    }

    let positions = build_positions(&spans, &grids, ids.len())?;
    let tags = build_tags(&spans, ids.len())?;

    Ok(Presentation {
        ids,
        spans,
        positions,
        tags,
    })
}

/// `h3_presentation_kind` (`h3_multimodal.h:25-29`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    Image,
    Video,
    Audio,
}

/// One ordered reference as the presentation sees it
/// (`h3_reference_presentation`, `h3_multimodal.h:32-39`). A video owns one
/// vision output per sampled two-frame block and one timestamp per block; an
/// image owns exactly one output; audio owns none.
pub struct Reference<'a> {
    pub kind: ReferenceKind,
    pub has_audio: bool,
    pub vision: &'a [VisionOutput],
    pub timestamps: &'a [f64],
}

/// Build the Ref2VA presentation for `prompt` and ordered `references`
/// (`h3_multimodal.c:243-360`).
///
/// Labels come in request order: `<Audio n>: ` for every reference that
/// carries a soundtrack, `<Picture n>: ` for an image, `<Video n>: ` once for
/// a video followed by one `<t.t seconds>` per block, then the prompt.
pub fn ref2va<'a>(
    tokenizer: &Tokenizer,
    prompt: &str,
    references: &'a [Reference<'a>],
) -> Result<Presentation<'a>, PresentationError> {
    assemble_references(
        |text| {
            tokenizer
                .encode(text, EmptyPrompt::Nothing)
                .map_err(PresentationError::Tokenizer)
        },
        prompt,
        references,
    )
}

/// The Ref2VA walk, with the tokenizer as a function so the id layout can be
/// tested without the BPE.
fn assemble_references<'a, F>(
    mut token: F,
    prompt: &str,
    references: &'a [Reference<'a>],
) -> Result<Presentation<'a>, PresentationError>
where
    F: FnMut(&str) -> Result<Vec<u32>, PresentationError>,
{
    if prompt.is_empty() || references.is_empty() {
        return Err(PresentationError::ReferenceArguments);
    }

    // `h3_multimodal.c:264-284`: each kind's structural requirement.
    let mut span_count = 0usize;
    let mut have_visual = false;
    for reference in references {
        match reference.kind {
            ReferenceKind::Image => {
                if reference.has_audio || reference.vision.len() != 1 {
                    return Err(PresentationError::Reference);
                }
                span_count += 1;
                have_visual = true;
            }
            ReferenceKind::Video => {
                if reference.vision.is_empty()
                    || reference.timestamps.len() != reference.vision.len()
                {
                    return Err(PresentationError::Reference);
                }
                span_count += reference.vision.len();
                have_visual = true;
            }
            ReferenceKind::Audio => {
                if !reference.has_audio || !reference.vision.is_empty() {
                    return Err(PresentationError::Reference);
                }
            }
        }
    }
    if !have_visual || span_count == 0 {
        return Err(PresentationError::ReferenceAudioOnly);
    }

    let mut ids = Vec::new();
    let mut spans = Vec::with_capacity(span_count);
    let mut grids = Vec::with_capacity(span_count);
    let (mut image_ordinal, mut video_ordinal, mut audio_ordinal) = (0usize, 0usize, 0usize);

    for reference in references {
        // `h3_multimodal.c:320-325`: an audio label is emitted for the video's
        // own soundtrack too, before the visual labels.
        if reference.has_audio {
            audio_ordinal += 1;
            ids.extend(token(&format!("<Audio {audio_ordinal}>: "))?);
        }

        match reference.kind {
            ReferenceKind::Audio => continue,
            ReferenceKind::Image => {
                image_ordinal += 1;
                ids.extend(token(&format!("<Picture {image_ordinal}>: "))?);
                span(&mut ids, &mut spans, &mut grids, &reference.vision[0], IMAGE_PAD)?;
            }
            ReferenceKind::Video => {
                video_ordinal += 1;
                for (block, output) in reference.vision.iter().enumerate() {
                    let timestamp = reference.timestamps[block];
                    if !timestamp.is_finite() || timestamp < 0.0 {
                        return Err(PresentationError::Timestamp);
                    }
                    if block == 0 {
                        ids.extend(token(&format!("<Video {video_ordinal}>: "))?);
                    }
                    ids.extend(token(&format!("<{timestamp:.1} seconds>"))?);
                    span(&mut ids, &mut spans, &mut grids, output, VIDEO_PAD)?;
                }
            }
        }
    }

    // `h3_multimodal.c:331-332`: the prompt follows the last reference.
    ids.extend(token(prompt)?);

    if ids.is_empty() || ids.len() > MAX_TOKENS {
        return Err(PresentationError::TooLarge);
    }

    let positions = build_positions(&spans, &grids, ids.len())?;
    let tags = build_tags(&spans, ids.len())?;

    Ok(Presentation {
        ids,
        spans,
        positions,
        tags,
    })
}

/// Append one vision span and keep the grid its positions are built from.
fn span<'a>(
    ids: &mut Vec<u32>,
    spans: &mut Vec<VisionSpan<'a>>,
    grids: &mut Vec<(usize, usize)>,
    output: &'a VisionOutput,
    pad: u32,
) -> Result<(), PresentationError> {
    let image = Image::of(output);
    check_geometry(&image)?;
    let span = append_vision(ids, pad, image.tokens);
    spans.push(VisionSpan {
        start: span.start,
        tokens: span.tokens,
        merged: image.merged,
        deepstack: image.deepstack,
    });
    grids.push(image.grid);

    Ok(())
}

/// Refuse a grid the reference refuses (`h3_multimodal.c:84-86`).
fn check_geometry(image: &Image<'_>) -> Result<(), PresentationError> {
    let (height, width) = image.grid;

    if height < 2 {
        return Err(PresentationError::Geometry("the grid height is below 2"));
    }
    if width < 2 {
        return Err(PresentationError::Geometry("the grid width is below 2"));
    }
    if height % 2 != 0 || width % 2 != 0 {
        return Err(PresentationError::Geometry("a grid edge is odd"));
    }

    Ok(())
}

/// `h3_multimodal.c:129-166`: the start id, the span, one pad id per row, the
/// end id. The span's first row is the first merged row, so `start` is the id
/// after the start marker.
pub fn append_vision(ids: &mut Vec<u32>, pad: u32, tokens: usize) -> Span {
    ids.push(VISION_START);
    let span = Span {
        start: ids.len(),
        tokens,
    };
    ids.extend(std::iter::repeat_n(pad, tokens));
    ids.push(VISION_END);

    span
}

/// `h3_multimodal.c:90-128`, axis-major `[3, sequence]`.
fn build_positions(
    spans: &[VisionSpan<'_>],
    grids: &[(usize, usize)],
    sequence: usize,
) -> Result<Vec<u32>, PresentationError> {
    let mut positions = vec![0u32; 3 * sequence];
    let mut offset: i64 = 0;

    for (index, span) in spans.iter().enumerate() {
        let (height, width) = grids[index];
        let start = span.start;
        let end = span.end();

        if start == 0 || start > sequence || end > sequence {
            return Err(PresentationError::Geometry("the span is outside the ids"));
        }
        let merged = (height / 2) * (width / 2);
        if merged != span.tokens {
            return Err(PresentationError::GridMismatch);
        }

        // Text before the first span keeps its own index on every axis.
        if index == 0 {
            for axis in 0..3 {
                for cell in 0..start {
                    positions[axis * sequence + cell] = cell as u32;
                }
            }
        }

        let longest = (height / 2).max(width / 2);
        let base = start as i64 + offset;
        let next = start as i64 + longest as i64 + offset;
        let rest = (sequence - end) as i64;
        if base < 0 || next < 0 || next + rest > u32::MAX as i64 {
            return Err(PresentationError::PositionOverflow);
        }

        // Axis 0 is the flat presentation coordinate for the whole span.
        positions[start..end].fill(base as u32);

        // Axes 1 and 2 walk the merged grid, row-major.
        let mut cursor = start;
        for row in 0..height / 2 {
            for column in 0..width / 2 {
                positions[sequence + cursor] = (base + row as i64) as u32;
                positions[2 * sequence + cursor] = (base + column as i64) as u32;
                cursor += 1;
            }
        }
        if cursor != end {
            return Err(PresentationError::GridMismatch);
        }

        for axis in 0..3 {
            for cell in end..sequence {
                positions[axis * sequence + cell] = (next + (cell - end) as i64) as u32;
            }
        }

        offset += longest as i64 - span.tokens as i64;
    }

    Ok(positions)
}

/// `h3_multimodal.c:236-252`: every row is text, then each span's own rows and
/// its two boundary ids are cleared.
fn build_tags(spans: &[VisionSpan<'_>], sequence: usize) -> Result<Vec<u8>, PresentationError> {
    let mut tags = vec![1u8; sequence];

    for span in spans {
        if span.start == 0 {
            return Err(PresentationError::TagOverflow);
        }
        let first = span.start - 1;
        let count = span.tokens + 2;
        if first > sequence || count > sequence - first {
            return Err(PresentationError::TagOverflow);
        }
        tags[first..first + count].fill(0);
    }

    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax_h3::vision::Grid;

    /// A stand-in tokenizer: one id per character, offset so the fake ids can
    /// never collide with a special one.
    fn characters(text: &str) -> Result<Vec<u32>, PresentationError> {
        Ok(text.bytes().map(|byte| u32::from(byte) + 1000).collect())
    }

    /// The payloads are only sliced by the decoder, which these tests do not
    /// run, so an empty slice is enough.
    fn image(grid: (usize, usize)) -> Image<'static> {
        Image {
            grid,
            tokens: (grid.0 / 2) * (grid.1 / 2),
            merged: &[],
            deepstack: [&[], &[], &[]],
        }
    }

    /// Render an id sequence as text, one `<vision>` per span, so the label
    /// order can be compared with `h3_multimodal.c:320-332`.
    fn render(ids: &[u32]) -> String {
        let mut text = String::new();
        for id in ids {
            match *id {
                VISION_START => text.push_str("<vision>"),
                VISION_END | IMAGE_PAD | VIDEO_PAD => {}
                value => text.push(char::from((value - 1000) as u8)),
            }
        }
        text
    }

    fn output(rows: usize) -> VisionOutput {
        let edge = rows * 16;
        VisionOutput::fixture(Grid::new(1, edge, edge).expect("grid"), rows * rows / 4)
    }

    fn ordered_references() -> (Vec<VisionOutput>, [f64; 2]) {
        let outputs = vec![output(4), output(4), output(4)];
        let timestamps = [0.0, 0.5];
        (outputs, timestamps)
    }

    /// `h3_multimodal.c:320-332`: labels follow request order, an audio label
    /// precedes the visual labels of the same reference, and the prompt is
    /// last.
    #[test]
    fn ref2va_labels_follow_request_order() {
        let (outputs, timestamps) = ordered_references();
        let references = [
            Reference {
                kind: ReferenceKind::Image,
                has_audio: false,
                vision: &outputs[0..1],
                timestamps: &[],
            },
            Reference {
                kind: ReferenceKind::Audio,
                has_audio: true,
                vision: &[],
                timestamps: &[],
            },
            Reference {
                kind: ReferenceKind::Video,
                has_audio: true,
                vision: &outputs[1..3],
                timestamps: &timestamps,
            },
        ];

        let presentation =
            assemble_references(characters, "p", &references).expect("the references are legal");

        assert_eq!(
            render(presentation.ids()),
            "<Picture 1>: <vision><Audio 1>: <Audio 2>: <Video 1>: <0.0 seconds><vision>\
             <0.5 seconds><vision>p"
        );
        assert_eq!(presentation.spans().len(), 3);
        assert_eq!(presentation.positions().len(), 3 * presentation.tokens());
        assert_eq!(presentation.tags().len(), presentation.tokens());
    }

    /// An image may not own a soundtrack and owns exactly one output.
    #[test]
    fn an_image_reference_is_one_silent_output() {
        let outputs = vec![output(4)];
        let with_audio = [Reference {
            kind: ReferenceKind::Image,
            has_audio: true,
            vision: &outputs,
            timestamps: &[],
        }];
        assert_eq!(
            assemble_references(characters, "p", &with_audio)
                .unwrap_err()
                .token(),
            "invalid Ref2VA reference presentation"
        );

        let doubled = vec![output(4), output(4)];
        let two_outputs = [Reference {
            kind: ReferenceKind::Image,
            has_audio: false,
            vision: &doubled,
            timestamps: &[],
        }];
        assert_eq!(
            assemble_references(characters, "p", &two_outputs)
                .unwrap_err()
                .token(),
            "invalid Ref2VA reference presentation"
        );
    }

    /// Audio alone cannot condition a clip (`h3_multimodal.c:283-285`).
    #[test]
    fn audio_needs_a_visual_reference() {
        let references = [Reference {
            kind: ReferenceKind::Audio,
            has_audio: true,
            vision: &[],
            timestamps: &[],
        }];
        assert_eq!(
            assemble_references(characters, "p", &references).unwrap_err().token(),
            "Ref2VA audio requires an image or video reference"
        );
    }

    /// A video needs one timestamp per block, finite and non-negative.
    #[test]
    fn video_blocks_need_matching_timestamps() {
        let outputs = vec![output(4), output(4)];
        let short = [Reference {
            kind: ReferenceKind::Video,
            has_audio: false,
            vision: &outputs,
            timestamps: &[0.0],
        }];
        assert_eq!(
            assemble_references(characters, "p", &short).unwrap_err().token(),
            "invalid Ref2VA reference presentation"
        );

        let negative = [Reference {
            kind: ReferenceKind::Video,
            has_audio: false,
            vision: &outputs,
            timestamps: &[0.0, -1.0],
        }];
        assert_eq!(
            assemble_references(characters, "p", &negative).unwrap_err().token(),
            "invalid Ref2VA video timestamp"
        );
    }

    /// Empty arguments are refused before anything is tokenized.
    #[test]
    fn ref2va_needs_a_prompt_and_references() {
        let (outputs, _) = ordered_references();
        let references = [Reference {
            kind: ReferenceKind::Image,
            has_audio: false,
            vision: &outputs[0..1],
            timestamps: &[],
        }];
        for (prompt, list) in [("", &references[..]), ("p", &[])] {
            assert_eq!(
                assemble_references(characters, prompt, list).unwrap_err().token(),
                "invalid Ref2VA presentation arguments"
            );
        }
    }

    /// A grid that disagrees with its merged rows is refused by the position
    /// builder (`h3_multimodal.c:116-119`).
    #[test]
    fn ref2va_refuses_a_grid_token_mismatch() {
        let outputs = vec![VisionOutput::fixture(Grid::new(1, 64, 64).expect("grid"), 3)];
        let references = [Reference {
            kind: ReferenceKind::Image,
            has_audio: false,
            vision: &outputs,
            timestamps: &[],
        }];
        assert_eq!(
            assemble_references(characters, "p", &references).unwrap_err().token(),
            "Qwen merged grid does not match token span"
        );
    }

    #[test]
    fn the_special_ids_are_the_released_ones() {
        assert_eq!(VISION_START, 151_652);
        assert_eq!(VISION_END, 151_653);
        assert_eq!(IMAGE_PAD, 151_655);
        assert_eq!(VIDEO_PAD, 151_656);
    }

    #[test]
    fn append_vision_wraps_the_pads_in_the_two_markers() {
        let mut ids = vec![7];
        let span = append_vision(&mut ids, IMAGE_PAD, 3);

        assert_eq!(span.start, 2);
        assert_eq!(span.tokens, 3);
        assert_eq!(span.end(), 5);
        assert_eq!(
            ids,
            vec![7, VISION_START, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, VISION_END]
        );
    }

    #[test]
    fn two_images_are_presented_in_order_with_the_prompt_last() {
        let images = [image((4, 4)), image((4, 8))];
        let presentation =
            assemble(characters, "go", &images).expect("the presentation is well formed");

        // `<Picture 1>: ` is 13 characters, `<Picture 2>: ` also. The fake
        // tokenizer turns each into one id per character.
        let first_start = 13 + 1;
        let second_start = first_start + 4 + 1 + 13 + 1;

        assert_eq!(presentation.ids()[0], u32::from(b'<') + 1000);
        assert_eq!(presentation.ids()[13], VISION_START);

        let spans = presentation.spans();
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].start, spans[0].tokens), (first_start, 4));
        assert_eq!((spans[1].start, spans[1].tokens), (second_start, 8));

        let tail = &presentation.ids()[second_start + 8..];
        assert_eq!(
            tail,
            &[VISION_END, u32::from(b'g') + 1000, u32::from(b'o') + 1000]
        );
        assert_eq!(presentation.tokens(), second_start + 8 + 3);
    }

    #[test]
    fn the_first_span_starts_at_one_and_tags_cover_the_markers() {
        let images = [image((4, 4))];
        let presentation = assemble(characters, "x", &images).expect("well formed");
        let span = presentation.spans()[0];

        assert_eq!(presentation.tags().len(), presentation.tokens());
        assert!(presentation.tags()[..span.start - 1]
            .iter()
            .all(|tag| *tag == 1));
        assert!(
            presentation.tags()[span.start - 1..span.start + span.tokens + 1]
                .iter()
                .all(|tag| *tag == 0)
        );
        assert!(presentation.tags()[span.start + span.tokens + 1..]
            .iter()
            .all(|tag| *tag == 1));
    }

    #[test]
    fn a_four_by_four_merged_grid_walks_rows_then_columns() {
        // Grid 8x8 merges to 4x4; the presentation is `<Picture 1>: ` plus the
        // span, so the span starts after the 13 prefix ids.
        let images = [image((8, 8))];
        let presentation = assemble(characters, "x", &images).expect("well formed");
        let sequence = presentation.tokens();
        let span = presentation.spans()[0];
        let start = span.start as u32;
        let positions = presentation.positions();

        assert_eq!(positions.len(), 3 * sequence);

        // Text before the span keeps its index on every axis.
        for axis in 0..3 {
            for cell in 0..span.start {
                assert_eq!(positions[axis * sequence + cell], cell as u32);
            }
        }

        // Axis 0 is flat, axes 1 and 2 are the row and the column.
        for cell in 0..16 {
            let (row, column) = (cell / 4, cell % 4);
            assert_eq!(positions[span.start + cell], start);
            assert_eq!(positions[sequence + span.start + cell], start + row as u32);
            assert_eq!(
                positions[2 * sequence + span.start + cell],
                start + column as u32
            );
        }

        // Text after the span restarts at start + max(4, 4).
        let next = start + 4;
        for axis in 0..3 {
            for cell in span.end()..sequence {
                assert_eq!(
                    positions[axis * sequence + cell],
                    next + (cell - span.end()) as u32
                );
            }
        }
    }

    #[test]
    fn a_second_span_is_shifted_by_the_first_spans_unused_extent() {
        // The first grid merges to 2x4: eight rows over a longest edge of four,
        // so the running offset becomes 4 - 8 = -4 (`h3_multimodal.c:124`).
        let images = [image((4, 8)), image((4, 4))];
        let presentation = assemble(characters, "x", &images).expect("well formed");
        let spans = presentation.spans();
        let positions = presentation.positions();

        assert_eq!(spans[0].tokens, 8);
        assert_eq!(spans[1].tokens, 4);

        let base = spans[1].start as i64 - 4;
        for cell in spans[1].start..spans[1].end() {
            assert_eq!(positions[cell] as i64, base);
        }

        // The text between the spans restarts at the first span's
        // `start + max(2, 4)`, which the offset has not moved yet.
        let between = spans[0].end();
        assert_eq!(positions[between], (spans[0].start + 4) as u32);
    }

    #[test]
    fn an_odd_grid_edge_is_refused() {
        let images = [image((8, 7))];

        assert!(matches!(
            assemble(characters, "x", &images),
            Err(PresentationError::Geometry(_))
        ));
    }

    #[test]
    fn a_grid_below_two_is_refused() {
        let images = [image((0, 8))];

        assert!(matches!(
            assemble(characters, "x", &images),
            Err(PresentationError::Geometry(_))
        ));
    }

    #[test]
    fn an_empty_prompt_or_no_image_is_refused() {
        let images = [image((4, 4))];

        assert!(matches!(
            assemble(characters, "", &images),
            Err(PresentationError::Empty)
        ));
        assert!(matches!(
            assemble(characters, "x", &[]),
            Err(PresentationError::Empty)
        ));
    }

    #[test]
    fn a_grid_that_does_not_match_its_rows_is_refused() {
        let mut broken = image((4, 4));
        broken.grid = (8, 8);

        assert!(matches!(
            assemble(characters, "x", &[broken]),
            Err(PresentationError::GridMismatch)
        ));
    }
}
