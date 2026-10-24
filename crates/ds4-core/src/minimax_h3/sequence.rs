//! The packed FL2VA sequence the DiT consumes.
//!
//! The reference builds one contiguous row layout per request: a text segment,
//! optional keyframe/reference segments, then the audio and video targets, each
//! row carrying a 3-axis RoPE position. That layout is what the AdaLN row map,
//! the RoPE tables and the final output slices are all indexed by, so it has to
//! be built identically before any block runs.
//!
//! Source of truth is `/data/h3.c` at `eae5d54`, `h3_host.c:315-475`
//! (`h3_layout_build`) with the helpers at `:232-313` (`h3_frame_grid`,
//! `h3_video_grid`, `h3_audio_grid`, `h3_video_span_sum`) and the position
//! strides at `:15-16`.
//!
//! ```text
//! row 0                text_len                       +audio_rows        +video_rows
//! |----- text -----|--- cond/ref segments (0 here) ---|--- audio ---|----- video -----|
//! t = token index      t = keyframe time              t = cursor+i       t = cursor, +5/3*f[i%5]
//! h = w = 0            h,w = 32-scaled grid           h = 0, w = edge     h,w = 32-scaled grid
//! ```
//!
//! Deviation, named: `H3_LAYOUT_REF_IMAGE` / `REF_AUDIO` / `REF_VIDEO`
//! (Ref2VA references, `h3_host.c:367-433`) are not ported; `build` returns
//! `Unsupported::References` for a non-empty reference list instead of
//! approximating them. Keyframes are ported because the same condition path
//! feeds them (`:351-366`).

/// `h3_segment_kind` (`h3_host.h:27-36`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Text,
    Cond,
    RefImage,
    RefAudio,
    Audio,
    Video,
}

/// `h3_segment` (`h3_host.h:37-41`): half-open row range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub start: usize,
    pub stop: usize,
    pub kind: SegmentKind,
}

/// `h3_position` (`h3_host.h:20-24`): the three RoPE axes, in doubles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub t: f64,
    pub h: f64,
    pub w: f64,
}

/// `h3_layout` (`h3_host.h:70-81`) without the owned-segment indirection.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub positions: Vec<Position>,
    pub segments: Vec<Segment>,
    pub img_cond_rows: usize,
    pub img_target_rows: usize,
    pub audio_cond_rows: usize,
    pub audio_target_rows: usize,
    /// `{text_len, latent_t, latent_h, latent_w, audio_t}`.
    pub signature: [i32; 5],
}

impl Sequence {
    pub fn rows(&self) -> usize {
        self.positions.len()
    }

    /// Row counts the DiT validates against the weight layout
    /// (`h3_dit.c:292-377`).
    pub fn video_total_rows(&self) -> usize {
        self.img_cond_rows + self.img_target_rows
    }

    pub fn audio_total_rows(&self) -> usize {
        self.audio_cond_rows + self.audio_target_rows
    }
}

/// `h3_layout_spec` (`h3_host.h:57-68`) restricted to what the port builds.
#[derive(Debug, Clone, Copy)]
pub struct Spec<'a> {
    pub text_len: usize,
    pub latent_t: i32,
    pub latent_h: i32,
    pub latent_w: i32,
    pub audio_t: i32,
    pub frame_count: i32,
    pub keyframes: &'a [i32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// `h3_host.c:322-325`: text, latent time, audio time or frame count.
    Dimensions,
    /// `h3_host.c:326-328`: keyframes and references together.
    KeyframesAndReferences,
    /// `h3_host.c:232-234`: the latent canvas must be even and at least 2x2.
    Canvas { latent_h: i32, latent_w: i32 },
    /// `h3_host.c:356-359`: only the first and last frame may be keyframes.
    Keyframe { index: i32, frame_count: i32 },
    /// The named deviation above.
    References,
}

/// `h3_layout_build` (`h3_host.c:315-475`) for the text/keyframe path.
pub fn build(spec: &Spec<'_>) -> Result<Sequence, LayoutError> {
    if spec.text_len < 1 || spec.latent_t < 1 || spec.audio_t < 0 || spec.frame_count < 5 {
        return Err(LayoutError::Dimensions);
    }

    let frame = frame_grid(spec.latent_h, spec.latent_w)?;
    let mut positions = Vec::with_capacity(spec.text_len + frame.len());
    let mut segments = Vec::new();

    // `h3_host.c:339-348`: text rows sit on the token axis, t = index.
    let text: Vec<Position> = (0..spec.text_len)
        .map(|index| Position {
            t: index as f64,
            h: 0.0,
            w: 0.0,
        })
        .collect();
    emit(&mut positions, &mut segments, SegmentKind::Text, &text);

    // `h3_host.c:349-366`: the cursor starts after the text and a keyframe
    // reuses either the first or the last video time.
    let cursor = spec.text_len as f64;
    let mut img_cond_rows = 0usize;
    let mut frame_rows = frame.clone();

    for &keyframe in spec.keyframes {
        let condition_time = if keyframe == 0 {
            spec.text_len as f64
        } else if keyframe == spec.frame_count - 1 {
            (spec.text_len as f64) + video_span_sum(spec.latent_t) - super::host::FRAME_RESCALE
        } else {
            return Err(LayoutError::Keyframe {
                index: keyframe,
                frame_count: spec.frame_count,
            });
        };

        for position in frame_rows.iter_mut() {
            position.t = condition_time;
        }
        emit(
            &mut positions,
            &mut segments,
            SegmentKind::Cond,
            &frame_rows,
        );
        img_cond_rows += frame_rows.len();
    }

    // `h3_host.c:435-457`: the audio target spans the canvas width edges, then
    // the video target walks the latent frames.
    let widths = width_axis(spec.latent_h, spec.latent_w)?;
    let audio = audio_grid(cursor, spec.audio_t, widths[0], widths[widths.len() - 1]);
    emit(&mut positions, &mut segments, SegmentKind::Audio, &audio);
    let audio_target_rows = audio.len();

    let video = video_grid(cursor, spec.latent_t, &frame);
    emit(&mut positions, &mut segments, SegmentKind::Video, &video);
    let img_target_rows = video.len();

    Ok(Sequence {
        positions,
        segments,
        img_cond_rows,
        img_target_rows,
        audio_cond_rows: 0,
        audio_target_rows,
        signature: [
            spec.text_len as i32,
            spec.latent_t,
            spec.latent_h,
            spec.latent_w,
            spec.audio_t,
        ],
    })
}

/// `h3_emit` (`h3_host.c:219-231`): append rows and one half-open segment.
fn emit(
    positions: &mut Vec<Position>,
    segments: &mut Vec<Segment>,
    kind: SegmentKind,
    rows: &[Position],
) {
    let start = positions.len();
    positions.extend_from_slice(rows);
    segments.push(Segment {
        start,
        stop: positions.len(),
        kind,
    });
}

/// `h3_frame_grid` (`h3_host.c:232-268`): an `(h/2) x (w/2)` grid whose
/// coordinates are normalized to the canvas area and scaled by 32.
///
/// The normalization is what makes the same canvas size share RoPE positions
/// across aspect ratios: a 16x16 and a 8x32 latent both span 32 units.
fn frame_grid(latent_h: i32, latent_w: i32) -> Result<Vec<Position>, LayoutError> {
    if latent_h < 2 || latent_w < 2 || latent_h % 2 != 0 || latent_w % 2 != 0 {
        return Err(LayoutError::Canvas { latent_h, latent_w });
    }
    let nh = (latent_h / 2) as usize;
    let nw = (latent_w / 2) as usize;

    let area = ((latent_h as f64) * (latent_w as f64)).sqrt();
    let ratio_h = latent_h as f64 / area;
    let ratio_w = latent_w as f64 / area;
    let step_h = ratio_h / nh as f64;
    let step_w = ratio_w / nw as f64;
    let base_h = (1.0 - ratio_h) / 2.0;
    let base_w = (1.0 - ratio_w) / 2.0;

    let mut grid = Vec::with_capacity(nh * nw);
    for row in 0..nh {
        let hh = ((row as f64) * step_h + base_h) * 32.0;
        for column in 0..nw {
            let ww = ((column as f64) * step_w + base_w) * 32.0;
            grid.push(Position {
                t: 0.0,
                h: hh,
                w: ww,
            });
        }
    }
    Ok(grid)
}

/// The `w` axis `h3_frame_grid` also returns (`widths`, `h3_host.c:254-256`);
/// the audio grid uses its two ends.
fn width_axis(latent_h: i32, latent_w: i32) -> Result<Vec<f64>, LayoutError> {
    let frame = frame_grid(latent_h, latent_w)?;
    let count = frame.len();
    Ok((0..count).map(|index| frame[index].w).collect())
}

/// `h3_audio_grid` (`h3_host.c:273-293`): two streams over the same time axis,
/// one per canvas width edge.
fn audio_grid(cursor: f64, audio_t: i32, low: f64, high: f64) -> Vec<Position> {
    if audio_t <= 0 {
        return Vec::new();
    }
    let rows = (audio_t as usize) * 2;
    let mut grid = Vec::with_capacity(rows);
    for index in 0..audio_t {
        grid.push(Position {
            t: cursor + (index as f64),
            h: 0.0,
            w: low,
        });
    }
    for index in 0..audio_t {
        grid.push(Position {
            t: cursor + (index as f64),
            h: 0.0,
            w: high,
        });
    }
    grid
}

/// `h3_video_grid` (`h3_host.c:295-313`): the frame grid repeated over latent
/// time, advancing by `5/3 * FRAME_PER_TOKEN[i % 5]` per frame.
fn video_grid(cursor: f64, latent_t: i32, frame: &[Position]) -> Vec<Position> {
    if latent_t < 0 {
        return Vec::new();
    }
    let mut grid = Vec::with_capacity((latent_t as usize) * frame.len());
    let mut time = cursor;
    for index in 0..latent_t {
        for position in frame {
            grid.push(Position {
                t: time,
                h: position.h,
                w: position.w,
            });
        }
        time += super::host::FRAME_RESCALE
            * (super::host::FRAME_PER_TOKEN[(index % 5) as usize] as f64);
    }
    grid
}

/// `h3_video_span_sum` (`h3_host.c:270-276`).
fn video_span_sum(latent_t: i32) -> f64 {
    (0..latent_t)
        .map(|index| {
            super::host::FRAME_RESCALE * (super::host::FRAME_PER_TOKEN[(index % 5) as usize] as f64)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The E5 probe's smallest clip: 6 text rows, 256x256 (16x16 latent),
    /// 5 frames (2 latent frames, 8 audio latents). `rows=150` and the FNV over
    /// the position bytes are the probe's `LAYOUT` line.
    #[test]
    fn smallest_clip_matches_the_reference_positions() {
        let spec = Spec {
            text_len: 6,
            latent_t: 2,
            latent_h: 16,
            latent_w: 16,
            audio_t: 8,
            frame_count: 5,
            keyframes: &[],
        };
        let sequence = build(&spec).expect("the smallest clip is legal");

        assert_eq!(sequence.rows(), 150);
        assert_eq!(sequence.segments.len(), 3);
        assert_eq!(sequence.img_cond_rows, 0);
        assert_eq!(sequence.audio_cond_rows, 0);
        assert_eq!(sequence.img_target_rows, 128);
        assert_eq!(sequence.audio_target_rows, 16);
        assert_eq!(sequence.signature, [6, 2, 16, 16, 8]);
        assert_eq!(fnv_positions(&sequence), 0x728d_afc2_ad79_9386);
    }

    /// `h3_host.c:232-268`: positions are area-normalized, so a square canvas
    /// spans 0..32 across `h/2` rows and `w/2` columns.
    #[test]
    fn frame_grid_is_area_normalized() {
        let grid = frame_grid(16, 16).expect("16x16 is legal");
        assert_eq!(grid.len(), 64);
        // 16x16: area 16, ratio 1, step 1/8, base 0, so h = row * 4.
        assert_eq!(grid[0].h, 0.0);
        assert_eq!(grid[0].w, 0.0);
        assert_eq!(grid[8].h, 4.0);
        assert_eq!(grid[63].h, 28.0);
        assert_eq!(grid[63].w, 28.0);
    }

    /// `h3_host.c:295-313`: the time axis advances 5/3 for the first latent
    /// frame and then 20/3 for the four-frame cycle, repeated over the frame
    /// grid's rows.
    #[test]
    fn video_time_advances_on_the_five_frame_cycle() {
        let frame = frame_grid(4, 4).expect("4x4 is legal");
        let rows = frame.len();
        assert_eq!(rows, 4);
        let video = video_grid(10.0, 6, &frame);
        assert_eq!(video[0].t, 10.0);
        assert_eq!(video[rows - 1].t, 10.0);
        assert_eq!(video[rows].t, 10.0 + 5.0 / 3.0);
        assert_eq!(video[2 * rows].t, 10.0 + 5.0 / 3.0 + 20.0 / 3.0);
    }

    /// `h3_host.c:273-293`: both audio streams share the time axis and sit on
    /// the first and last canvas column.
    #[test]
    fn audio_rows_use_the_canvas_edges() {
        let audio = audio_grid(6.0, 3, 2.5, 29.5);
        assert_eq!(audio.len(), 6);
        assert_eq!(audio[0].w, 2.5);
        assert_eq!(audio[3].w, 29.5);
        assert_eq!(audio[2].t, 8.0);
        assert_eq!(audio[5].t, 8.0);
    }

    /// `h3_host.c:356-359`.
    #[test]
    fn only_edge_keyframes_are_legal() {
        let spec = Spec {
            text_len: 4,
            latent_t: 2,
            latent_h: 8,
            latent_w: 8,
            audio_t: 8,
            frame_count: 22,
            keyframes: &[7],
        };
        assert_eq!(
            build(&spec),
            Err(LayoutError::Keyframe {
                index: 7,
                frame_count: 22
            })
        );
    }

    /// `h3_host.c:322-325` and `:232-234`.
    #[test]
    fn dimensions_and_canvas_are_checked() {
        let base = Spec {
            text_len: 4,
            latent_t: 2,
            latent_h: 8,
            latent_w: 8,
            audio_t: 8,
            frame_count: 5,
            keyframes: &[],
        };
        assert_eq!(
            build(&Spec {
                text_len: 0,
                ..base
            }),
            Err(LayoutError::Dimensions)
        );
        assert_eq!(
            build(&Spec {
                latent_h: 7,
                ..base
            }),
            Err(LayoutError::Canvas {
                latent_h: 7,
                latent_w: 8
            })
        );
    }

    /// The reference's own position hash: FNV over `seq_len` x 24 bytes of
    /// `{t, h, w}` in that order.
    fn fnv_positions(sequence: &Sequence) -> u64 {
        let mut hash = 1_469_598_103_934_665_603u64;
        for position in &sequence.positions {
            for value in [position.t, position.h, position.w] {
                for byte in value.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(1_099_511_628_211);
                }
            }
        }
        hash
    }
}
