//! Canvas sizes for the released 768P tier.
//!
//! The short edge is 768, the long edge follows the aspect ratio, and the
//! released area clamps it (`h3.c:490-500`). This reproduces the platform's
//! own 1344x768 16:9 artifact: round(768 * 16/9) snaps to 1376, which exceeds
//! the released `768*1344` area. The local API delivers that canvas directly.
//!
//! 480P and 2K belong to other variants and are not served by this checkpoint.

use super::render::{CANVAS_MULTIPLE, MAX_PIXELS};

/// The 768P tier's short edge, the default the model card documents.
pub const TIER_SHORT_EDGE: i32 = 768;

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
    let (width, height) = ratio.parts();
    canvas_for(width as u32, height as u32)
}

/// The same rule for an aspect the caller states as pixels: an anchored image
/// whose ratio has no name (`aspect_ratio: auto`). The short edge is the
/// tier's, the long edge is snapped to the released grid and clamped by the
/// released area, exactly as [`tier_canvas`] does for a name.
pub fn canvas_for(width: u32, height: u32) -> (i32, i32) {
    let (long, short) = if width >= height {
        (width, height)
    } else {
        (height, width)
    };
    if long == 0 || short == 0 {
        return (0, 0);
    }

    let edge = f64::from(TIER_SHORT_EDGE) * f64::from(long) / f64::from(short);
    let cap = MAX_PIXELS / i64::from(TIER_SHORT_EDGE);
    let units = (edge / f64::from(CANVAS_MULTIPLE))
        .round()
        .min((cap / i64::from(CANVAS_MULTIPLE)) as f64);
    let clamped = units as i32 * CANVAS_MULTIPLE;

    if width >= height {
        (clamped, TIER_SHORT_EDGE)
    } else {
        (TIER_SHORT_EDGE, clamped)
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
        assert_eq!(canvas_for(1920, 1080), (1344, 768));
        assert_eq!(canvas_for(1080, 1920), (768, 1344));
        assert_eq!(canvas_for(1024, 1024), (768, 768));
        assert_eq!(canvas_for(0, 0), (0, 0));
    }
}
