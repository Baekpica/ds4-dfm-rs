//! Canvas sizes for the released checkpoint.
//!
//! The short edge is the caller's, the long edge follows the aspect ratio on
//! the released 32-pixel grid, and the released area clamps it (`h3.c:490-500`).
//! At the 768P tier this reproduces the platform's own 1344x768 16:9 artifact:
//! round(768 * 16/9) snaps to 1376, which exceeds the released `768*1344` area.
//! The local API delivers that canvas directly.
//!
//! 768 is the tier the model card documents, not the only short edge served.
//! The reference's own scale points are 256, 320 and 384, and its CLI refuses
//! only a canvas off the 32-pixel grid or over the released area (`h3.c`
//! README, "Pick resolution and duration"). A short edge below the tier is an
//! execution preview, not a second released tier. 480P and 2K belong to other
//! variants and are not served by this checkpoint.

use super::render::{CANVAS_MULTIPLE, MAX_PIXELS};

/// The 768P tier's short edge, the tier the model card documents.
pub const TIER_SHORT_EDGE: i32 = 768;

/// The smallest short edge the local API serves: one canvas unit.
pub const MIN_SHORT_EDGE: i32 = CANVAS_MULTIPLE;

/// The largest short edge the released area admits.
///
/// The requested short edge is the canvas's short side, so the smallest canvas
/// a request can produce is its square, and `MAX_PIXELS` = 768 * 1344 puts the
/// ceiling on the released grid at 992: 1024 * 1024 exceeds the area, and 1024's
/// own long edge would clamp to 992 and invert the canvas. At the ceiling and
/// below, the cap on the long edge can no longer fall under the short edge
/// (`h3_valid_params`, `h3.c:496`).
pub const MAX_SHORT_EDGE: i32 = {
    let unit = CANVAS_MULTIPLE as i64;
    let mut edge = MAX_PIXELS / unit * unit;
    while edge * edge > MAX_PIXELS {
        edge -= unit;
    }
    edge as i32
};

/// The largest ratio distortion the local API accepts.
pub const MAX_DISTORTION: f64 = 1.25;

/// The official aspect ratios (`video-generation-v2-create`, `ratio`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ratio {
    Wide21x9,
    Wide16x9,
    FourThree,
    Square,
    ThreeFour,
    Nine16,
}

impl Ratio {
    /// Numerator and denominator of width over height.
    pub fn parts(self) -> (i32, i32) {
        match self {
            Ratio::Wide21x9 => (21, 9),
            Ratio::Wide16x9 => (16, 9),
            Ratio::FourThree => (4, 3),
            Ratio::Square => (1, 1),
            Ratio::ThreeFour => (3, 4),
            Ratio::Nine16 => (9, 16),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Ratio::Wide21x9 => "21:9",
            Ratio::Wide16x9 => "16:9",
            Ratio::FourThree => "4:3",
            Ratio::Square => "1:1",
            Ratio::ThreeFour => "3:4",
            Ratio::Nine16 => "9:16",
        }
    }

    pub fn parse(text: &str) -> Option<Ratio> {
        match text {
            "21:9" => Some(Ratio::Wide21x9),
            "16:9" => Some(Ratio::Wide16x9),
            "4:3" => Some(Ratio::FourThree),
            "1:1" => Some(Ratio::Square),
            "3:4" => Some(Ratio::ThreeFour),
            "9:16" => Some(Ratio::Nine16),
            _ => None,
        }
    }
}

/// The 768P canvas the platform renders for one ratio: the short edge is the
/// tier's, the long edge is snapped to the released grid and clamped by the
/// released area. Measured against `assets/h3_direct_768p.mp4` (1344x768).
pub fn tier_canvas(ratio: Ratio) -> (i32, i32) {
    ratio_canvas(TIER_SHORT_EDGE, ratio)
}

/// The same rule at a caller-chosen short edge. The aspect a ratio names is
/// stated as width over height, so 9:16 asks for a tall canvas.
pub fn ratio_canvas(short_edge: i32, ratio: Ratio) -> (i32, i32) {
    let (width, height) = ratio.parts();
    canvas_for(short_edge, width as u32, height as u32)
}

/// The same rule for an aspect the caller states as pixels: an anchored image
/// whose ratio has no name (`aspect_ratio: auto`). The short edge is the
/// caller's, the long edge is snapped to the released grid and clamped by the
/// released area, exactly as [`ratio_canvas`] does for a name.
///
/// The caller keeps `short_edge` within [`MIN_SHORT_EDGE`] and
/// [`MAX_SHORT_EDGE`]; above the ceiling the clamp drops the long edge under
/// the short edge, so the canvas would come back inverted.
pub fn canvas_for(short_edge: i32, width: u32, height: u32) -> (i32, i32) {
    let (long, short) = if width >= height {
        (width, height)
    } else {
        (height, width)
    };
    if long == 0 || short == 0 || short_edge < MIN_SHORT_EDGE {
        return (0, 0);
    }

    let edge = f64::from(short_edge) * f64::from(long) / f64::from(short);
    let cap = MAX_PIXELS / i64::from(short_edge);
    let units = (edge / f64::from(CANVAS_MULTIPLE))
        .round()
        .min((cap / i64::from(CANVAS_MULTIPLE)) as f64);
    let clamped = units as i32 * CANVAS_MULTIPLE;

    if width >= height {
        (clamped, short_edge)
    } else {
        (short_edge, clamped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tier_rule_reproduces_the_platform_artifacts() {
        // The platform's own 768P 16:9 file is 1344x768: 1365.33 snaps to
        // 1376, and the released area clamps it to 1344.
        assert_eq!(tier_canvas(Ratio::Wide16x9), (1344, 768));
        assert_eq!(tier_canvas(Ratio::Nine16), (768, 1344));
        assert_eq!(tier_canvas(Ratio::FourThree), (1024, 768));
        assert_eq!(tier_canvas(Ratio::ThreeFour), (768, 1024));
        assert_eq!(tier_canvas(Ratio::Square), (768, 768));
        // 21:9 wants 1792 px of width, which the cap refuses.
        assert_eq!(tier_canvas(Ratio::Wide21x9), (1344, 768));

        // An anchored image states its aspect as pixels; the same rule
        // applies, including the square and the cap.
        assert_eq!(canvas_for(TIER_SHORT_EDGE, 1920, 1080), (1344, 768));
        assert_eq!(canvas_for(TIER_SHORT_EDGE, 1080, 1920), (768, 1344));
        assert_eq!(canvas_for(TIER_SHORT_EDGE, 1024, 1024), (768, 768));
        assert_eq!(canvas_for(TIER_SHORT_EDGE, 0, 0), (0, 0));
    }

    #[test]
    fn a_chosen_short_edge_scales_the_same_rule() {
        // 576 is on the released grid and under the area, so 16:9 is 1024x576
        // where the tier's 16:9 is 1344x768.
        assert_eq!(ratio_canvas(576, Ratio::Wide16x9), (1024, 576));
        assert_eq!(ratio_canvas(576, Ratio::Nine16), (576, 1024));
        assert_eq!(ratio_canvas(384, Ratio::Square), (384, 384));
        // 4:3 at a 256 short edge wants 341 px of width, which snaps up to
        // the nearest unit, 352.
        assert_eq!(ratio_canvas(256, Ratio::FourThree), (352, 256));

        // The released area still bounds the long edge: 21:9 fits at 576 and
        // does not at the tier.
        assert_eq!(ratio_canvas(576, Ratio::Wide21x9), (1344, 576));
        assert_eq!(tier_canvas(Ratio::Wide21x9), (1344, 768));

        // Below one canvas unit there is no canvas to serve.
        assert_eq!(canvas_for(MIN_SHORT_EDGE - 1, 16, 9), (0, 0));
    }
}
