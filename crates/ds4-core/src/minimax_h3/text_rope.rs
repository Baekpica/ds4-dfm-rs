//! Host-side RoPE tables for the Qwen3-VL prompt encoder.
//!
//! The reference builds two `[tokens, 64]` float32 tables before the layer
//! loop and uploads them unchanged; the fused device kernel then reads them
//! per head (`h3_text_encoder.c:543-567`). This module is that construction
//! alone, so the tables can be gated before any layer runs.
//!
//! Source of truth `/data/h3.c` at `eae5d54`, `h3_text_encoder.c`:
//!
//! - `:543-548` `inverse_frequency[i] = 1 / powf(theta, 2i / head_dim)`,
//!   theta `5000000.0f` (`:27`), every term computed in float32.
//! - `:549-566` `angle = coordinate * inverse_frequency[i]`,
//!   `cosine = cosf(angle)`, `sine = sinf(angle)`.
//!
//! The text path passes `position_ids = NULL` (`:622-625` calls
//! `h3_text_encode_bf16` with no positions), so two branches of the reference
//! loop do not run and are deliberately absent here:
//!
//! - `:551-553` the mRoPE axis interleave (index % 3 picks the position axis)
//!   is guarded by `position_ids`.
//! - `:561-566` the BF16 round of both tables is likewise guarded by
//!   `position_ids`; without them the tables stay float32.
//!
//! Deviation, named: none. The arithmetic order is the reference's, including
//! `1.0f / powf(...)` rather than a negative exponent, because the two differ
//! in the last bit.
//!
//! The float32 `powf`, `sinf` and `cosf` calls resolve to the platform libm on
//! both sides (the reference is compiled by gcc, this module by rustc), so the
//! bit match is a property of the same C library, measured by the E4 Phase A
//! gate rather than assumed.

use crate::minimax_h3::text::{HEAD_DIM, ROPE_THETA};

/// `TEXT_ROPE_HALF` (`h3_text_encoder.c:22`): one table row covers half a
/// head, since a rotating pair shares one angle.
pub const ROPE_HALF: usize = HEAD_DIM as usize / 2;

/// `TEXT_ROPE_THETA` (`h3_text_encoder.c:27`) at its declared width.
pub const THETA: f32 = ROPE_THETA as f32;

/// The two `[tokens, ROPE_HALF]` tables of one prompt.
#[derive(Debug)]
pub struct Rope {
    tokens: usize,
    cosines: Vec<f32>,
    sines: Vec<f32>,
}

impl Rope {
    /// Build the tables for `tokens` positions starting at zero.
    ///
    /// `h3_text_encoder.c:549-566`: the text-only path has no `position_ids`,
    /// so the coordinate of every row is its index and no value is rounded to
    /// BF16.
    pub fn build(tokens: usize) -> Rope {
        let inverse = inverse_frequency();
        let mut cosines = vec![0.0f32; tokens * ROPE_HALF];
        let mut sines = vec![0.0f32; tokens * ROPE_HALF];

        for position in 0..tokens {
            for (index, frequency) in inverse.iter().enumerate() {
                let coordinate = position as f32;
                let angle = coordinate * frequency;

                cosines[position * ROPE_HALF + index] = angle.cos();
                sines[position * ROPE_HALF + index] = angle.sin();
            }
        }

        Rope {
            tokens,
            cosines,
            sines,
        }
    }

    /// Rows in each table.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Values per row, `ROPE_HALF`.
    pub fn half(&self) -> usize {
        ROPE_HALF
    }

    /// The cosine table, row-major `[tokens, ROPE_HALF]`.
    pub fn cosines(&self) -> &[f32] {
        &self.cosines
    }

    /// The sine table, row-major `[tokens, ROPE_HALF]`.
    pub fn sines(&self) -> &[f32] {
        &self.sines
    }
}

/// `h3_text_encoder.c:543-548`: `1 / theta^(2i/head_dim)` in float32.
pub fn inverse_frequency() -> [f32; ROPE_HALF] {
    let mut inverse = [0.0f32; ROPE_HALF];

    for (index, value) in inverse.iter_mut().enumerate() {
        let exponent = (index * 2) as f32 / HEAD_DIM as f32;

        *value = 1.0 / THETA.powf(exponent);
    }

    inverse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_frequency_is_exactly_one() {
        assert_eq!(inverse_frequency()[0], 1.0);
    }

    #[test]
    fn the_last_frequency_is_the_reference_exponent() {
        let exponent = 126.0f32 / 128.0f32;
        assert_eq!(
            inverse_frequency()[ROPE_HALF - 1],
            1.0 / THETA.powf(exponent)
        );
    }

    #[test]
    fn frequencies_decrease_with_the_index() {
        let inverse = inverse_frequency();
        assert!(inverse.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn the_zero_row_is_exactly_one_and_zero() {
        let rope = Rope::build(3);

        assert!(rope.cosines()[..ROPE_HALF]
            .iter()
            .all(|v| v.to_bits() == 1.0f32.to_bits()));
        assert!(rope.sines()[..ROPE_HALF].iter().all(|v| v.to_bits() == 0));
    }

    #[test]
    fn the_unit_row_follows_the_angle_table() {
        let rope = Rope::build(2);
        let (cosine, sine) = (1.0f32.cos(), 1.0f32.sin());

        assert_eq!(rope.cosines()[ROPE_HALF], cosine);
        assert_eq!(rope.sines()[ROPE_HALF], sine);
    }

    #[test]
    fn every_cell_keeps_unit_norm() {
        let rope = Rope::build(8);

        for (cosine, sine) in rope.cosines().iter().zip(rope.sines()) {
            let norm = cosine * cosine + sine * sine;
            assert!((norm - 1.0).abs() < 1e-6, "norm {norm} at cell");
        }
    }

    #[test]
    fn the_tables_are_row_major_and_the_requested_size() {
        let rope = Rope::build(6);

        assert_eq!(rope.tokens(), 6);
        assert_eq!(rope.half(), ROPE_HALF);
        assert_eq!(rope.cosines().len(), 6 * ROPE_HALF);
        assert_eq!(rope.sines().len(), 6 * ROPE_HALF);
    }
}
