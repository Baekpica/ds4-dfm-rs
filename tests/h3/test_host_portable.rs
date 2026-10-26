//! `h3_resize_rgb24_high_quality` on a constant frame.
//!
//! Reference: `/data/h3.c` at `eae5d54`, `tests/test_host_portable.c:13-23`.
//! One 2x2 constant frame (`h3_resize_rgb24_high_quality(constant, 1, 2, 2, 8,
//! 8, &output)`: one frame, 2x2 in, 8x8 out) upscaled to 8x8: all 64 output
//! pixels must equal the source triple, byte for byte. A bilinear kernel with
//! half-pixel centers and edge clamping returns the constant unchanged, so
//! this is the reference's check that the port's resize neither shifts nor
//! blends a flat field.
//!
//! Model-free and host-only: run with `make tests/h3-test_host_portable`.

use std::process::ExitCode;

use ds4_core::minimax_h3::render::resize_rgb24;

// Four pixels, all the same triple: the reference's 2x2 source.
const SOURCE: [u8; 12] = [
    17, 33, 201, 17, 33, 201, //
    17, 33, 201, 17, 33, 201,
];
const SOURCE_EDGE: i32 = 2;
const OUT_EDGE: i32 = 8;

fn main() -> ExitCode {
    match run() {
        Ok(()) => {
            println!("ok: portable RGB resize");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("FAIL tests/h3/test_host_portable.rs: {message}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let output = resize_rgb24(&SOURCE, 1, SOURCE_EDGE, SOURCE_EDGE, OUT_EDGE, OUT_EDGE)?;
    let expected = (OUT_EDGE * OUT_EDGE * 3) as usize;

    if output.len() != expected {
        return Err(format!(
            "resize returned {} bytes, expected {expected}",
            output.len()
        ));
    }

    for (pixel, triple) in output.chunks_exact(3).enumerate() {
        if triple != &SOURCE[..3] {
            return Err(format!(
                "pixel {pixel} is {triple:?}, expected {:?}",
                &SOURCE[..3]
            ));
        }
    }

    Ok(())
}
