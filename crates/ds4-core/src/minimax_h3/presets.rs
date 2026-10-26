//! Delivery presets: the requested geometry, the canvas the model renders,
//! and the frame the file carries.
//!
//! The platform names geometry as a resolution tier and an aspect ratio, not
//! as pixels (`video-generation-v2-create`: `resolution` and `ratio`). The
//! released cap then fixes the pixels: the short edge is the tier's (768 for
//! this checkpoint), the long edge is `round(short * ratio)` snapped to a
//! multiple of 32, and the released area clamps it (`h3.c:490-500`). That rule
//! reproduces the platform's own 768P artifacts: the official 16:9 output is
//! 1344x768, square pixels, 24 fps (`assets/h3_direct_768p.mp4`, measured),
//! because round(768 * 16/9) is 1376 and 1376x768 exceeds `768*1344`. The same
//! clamp makes 9:16 768x1344, so both official wide and tall frames are 1.75:1
//! at this tier.
//!
//! Two kinds of preset follow from that:
//!
//! - a ratio preset delivers the exact ratio. Where the canvas is that ratio
//!   (4:3, 3:4, 1:1, 21:9) the scale is uniform; where the released cap forces
//!   a different canvas (16:9, 9:16) the delivery is an anamorphic scale of
//!   1.6 %, named here and logged by the engine (`render::OutputRule`);
//! - a parity preset delivers the canvas itself, which is what the platform
//!   ships, for comparison and reproduction.
//!
//! The reference refuses any frame that is not the canvas aspect
//! (`h3.c:505-520`), so an anamorphic preset is a named deviation. It is
//! measured, not assumed: the scale is per axis and the engine reports both.
//!
//! 480P and 2K are not served: they belong to the H3-Lite/Max variants and to
//! H3-Regenerate-2K, which are not this checkpoint.

use super::render::{CANVAS_MULTIPLE, MAX_PIXELS};

/// The 768P tier's short edge, the default the model card documents.
pub const TIER_SHORT_EDGE: i32 = 768;

/// The largest stretch a preset may ask for. Beyond a quarter the squeeze
/// stops reading as a rounding correction and starts reading as distortion.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Preset {
    pub name: &'static str,
    /// The model canvas: what is denoised and decoded.
    pub render: (i32, i32),
    /// The delivered frame: the MP4 geometry.
    pub output: (i32, i32),
}

impl Preset {
    /// The delivery scale per axis.
    pub fn scale(&self) -> (f64, f64) {
        (
            f64::from(self.output.0) / f64::from(self.render.0),
            f64::from(self.output.1) / f64::from(self.render.1),
        )
    }

    /// Horizontal scale over vertical: 1.0 is square pixels.
    pub fn distortion(&self) -> f64 {
        let (x, y) = self.scale();
        x / y
    }

    pub fn is_anamorphic(&self) -> bool {
        i64::from(self.render.0) * i64::from(self.output.1)
            != i64::from(self.render.1) * i64::from(self.output.0)
    }

    pub fn is_scaled(&self) -> bool {
        self.output != self.render
    }

    /// The rules the reference enforces on a canvas, checked at construction
    /// so an impossible preset is refused by name.
    fn check(&self) -> Result<(), String> {
        let (rw, rh) = self.render;
        let (ow, oh) = self.output;

        if rw < CANVAS_MULTIPLE
            || rh < CANVAS_MULTIPLE
            || rw % CANVAS_MULTIPLE != 0
            || rh % CANVAS_MULTIPLE != 0
        {
            return Err(format!(
                "{}: render canvas is not multiples of 32",
                self.name
            ));
        }

        if i64::from(rw) * i64::from(rh) > MAX_PIXELS {
            return Err(format!("{}: render canvas exceeds 768*1344", self.name));
        }

        // The delivery frame is not a canvas: 1080x1920 is not multiples of 32
        // and the model never renders it. It must still be even, which the mux
        // requires, and never smaller than the canvas.
        if ow < CANVAS_MULTIPLE || oh < CANVAS_MULTIPLE || ow % 2 != 0 || oh % 2 != 0 {
            return Err(format!(
                "{}: output frame must be even and at least 32",
                self.name
            ));
        }

        if rw > ow || rh > oh {
            return Err(format!(
                "{}: render canvas is larger than the output",
                self.name
            ));
        }

        let distortion = self.distortion().max(1.0 / self.distortion());
        if distortion > MAX_DISTORTION {
            return Err(format!(
                "{}: anamorphic scale {:.3} exceeds {MAX_DISTORTION}",
                self.name, distortion
            ));
        }

        Ok(())
    }
}

/// The presets. Ratio names deliver their exact ratio; `native_*` deliver the
/// model canvas the platform ships.
pub static PRESETS: [Preset; 9] = [
    // The released cap forces a 1.75:1 canvas at this tier, so the exact 16:9
    // frame is a 1.6 % stretch. The canvas is the platform's own 1344x768.
    Preset {
        name: "16:9",
        render: (1344, 768),
        output: (1920, 1080),
    },
    Preset {
        name: "9:16",
        render: (768, 1344),
        output: (1080, 1920),
    },
    // 21:9 is 7:3, whose largest canvas under the cap is 1344x576 (the next
    // 32-step, 1568x672, is 1,053,696 px and exceeds it). 2520x1080 is the
    // same ratio, so this delivery is uniform at 1.875.
    Preset {
        name: "21:9",
        render: (1344, 576),
        output: (2520, 1080),
    },
    // Exact ratios whose canvas is the tier canvas: no resize at all.
    Preset {
        name: "4:3",
        render: (1024, 768),
        output: (1024, 768),
    },
    Preset {
        name: "3:4",
        render: (768, 1024),
        output: (768, 1024),
    },
    Preset {
        name: "1:1",
        render: (768, 768),
        output: (768, 768),
    },
    // Platform parity: the file is the model canvas, 1.75:1 as shipped.
    Preset {
        name: "native_16x9",
        render: (1344, 768),
        output: (1344, 768),
    },
    Preset {
        name: "native_9x16",
        render: (768, 1344),
        output: (768, 1344),
    },
    // A feed format outside the official set: 4:5 at an exact canvas.
    Preset {
        name: "instagram_4x5",
        render: (768, 960),
        output: (1080, 1350),
    },
];

/// The operator's earlier vocabulary, kept as names for the same presets.
pub static ALIASES: [(&str, &str); 4] = [
    ("reel_phone", "9:16"),
    ("youtube_1080p", "16:9"),
    ("instagram_1x1", "1:1"),
    ("cinema_21x9", "21:9"),
];

pub fn resolve(name: &str) -> Option<&'static Preset> {
    if let Some(preset) = PRESETS.iter().find(|preset| preset.name == name) {
        // The table is static, so its rules are proved where they are
        // written; a debug build rechecks them on use.
        debug_assert!(preset.check().is_ok(), "{}", preset.name);

        return Some(preset);
    }

    let target = ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map(|(_, target)| *target)?;

    PRESETS.iter().find(|preset| preset.name == target)
}

pub fn names() -> impl Iterator<Item = &'static str> {
    PRESETS
        .iter()
        .map(|preset| preset.name)
        .chain(ALIASES.iter().map(|(alias, _)| *alias))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset(name: &str) -> &'static Preset {
        resolve(name).unwrap_or_else(|| panic!("{name} is a preset"))
    }

    #[test]
    fn every_preset_obeys_the_canvas_rules() {
        for entry in &PRESETS {
            assert_eq!(entry.check(), Ok(()), "{}", entry.name);
        }
    }

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

        for entry in &PRESETS {
            let (width, height) = entry.render;
            assert_eq!(width % CANVAS_MULTIPLE, 0, "{}", entry.name);
            assert_eq!(height % CANVAS_MULTIPLE, 0, "{}", entry.name);
        }
    }

    #[test]
    fn the_exact_ratio_presets_deliver_their_ratio() {
        for name in ["4:3", "3:4", "1:1", "21:9"] {
            let entry = preset(name);
            assert!(!entry.is_anamorphic(), "{name} is not anamorphic");
        }

        assert_eq!(preset("4:3").render, (1024, 768));
        assert_eq!(preset("3:4").render, (768, 1024));
        assert_eq!(preset("1:1").render, (768, 768));
        // 2520x1080 is 7:3 at exactly the canvas ratio.
        let cinema = preset("21:9");
        assert_eq!(cinema.render, (1344, 576));
        assert_eq!(cinema.output, (2520, 1080));
        assert!((cinema.scale().0 - cinema.scale().1).abs() < 1e-12);
    }

    #[test]
    fn the_wide_and_tall_presets_name_their_anamorphic_scale() {
        // The platform canvas is 1.75:1 and the delivered frames are 16:9 and
        // 9:16, so both scales stretch one axis by 1.6 % more than the other.
        let wide = preset("16:9");
        assert!(wide.is_anamorphic());
        assert!((wide.distortion() - 1.015_873_015_873_015_9).abs() < 1e-12);

        let tall = preset("9:16");
        assert!(tall.is_anamorphic());
        assert!((tall.distortion() - 0.984_375).abs() < 1e-12);

        // Above 1080 lines the delivery is larger than the canvas, which the
        // reference refuses; parity presets stay at the canvas.
        assert!(i64::from(wide.output.0) * i64::from(wide.output.1) > MAX_PIXELS);
        assert!(!preset("native_16x9").is_scaled());
        assert_eq!(preset("native_16x9").output, (1344, 768));
        assert_eq!(preset("native_9x16").output, (768, 1344));
    }

    #[test]
    fn the_aliases_keep_the_operator_vocabulary() {
        assert_eq!(preset("reel_phone").name, "9:16");
        assert_eq!(preset("youtube_1080p").name, "16:9");
        assert_eq!(preset("instagram_1x1").name, "1:1");
        assert_eq!(preset("cinema_21x9").name, "21:9");
    }

    #[test]
    fn a_preset_outside_the_named_deviations_is_refused() {
        // 35 % squeeze: the clamped 21:9 canvas delivered as true 21:9 would
        // be distortion, not a rounding correction.
        let squeezed = Preset {
            name: "probe",
            render: (1344, 768),
            output: (2560, 1080),
        };
        assert!(squeezed.check().is_err());

        // An odd delivery edge cannot be muxed.
        assert!(Preset {
            name: "probe",
            render: (768, 768),
            output: (1080, 1081),
        }
        .check()
        .is_err());

        // 1280x720 is 16:9 but 720 is not a multiple of 32.
        assert!(Preset {
            name: "probe",
            render: (1280, 720),
            output: (1280, 720),
        }
        .check()
        .is_err());
    }

    #[test]
    fn names_are_unique_and_resolve() {
        let mut seen = std::collections::BTreeSet::new();

        for name in names() {
            assert!(seen.insert(name), "duplicate {name}");
            assert!(resolve(name).is_some(), "{name} resolves");
        }

        assert_eq!(resolve("optical_disc"), None);
    }
}
