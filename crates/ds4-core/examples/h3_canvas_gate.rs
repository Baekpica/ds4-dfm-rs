//! Replay the reference's reference-media canvas policy over a fixed sweep.
//!
//! Usage: h3_canvas_gate
//!
//! Output is the same three TSV row kinds the reference probe prints
//! (`tools/minimax_h3/reference_canvas_probe.c`, compiled against
//! `/data/h3.c` at `eae5d54`), so the gate is a line-for-line diff:
//!
//!   ADAPT <src_w> <src_h> <out_w> <out_h>
//!   IMAGE <src_w> <src_h> <tgt_w> <tgt_h> <cap> <out_w> <out_h>
//!   VIDEO <src_w> <src_h> <out_w> <out_h>
//!
//! A refusal prints `-` in place of the output pair. The sources, targets and
//! caps are the probe's own lists (`h3_host.c:49-124` is the policy).

use std::process::ExitCode;

use ds4_core::minimax_h3::host::{adapt_canvas, reference_image_canvas, reference_video_canvas};

/// The probe's sources: platform shapes, portrait mirrors, tiny and extreme
/// aspects, and one odd pair for the rounding.
const SOURCES: [(i32, i32); 17] = [
    (1344, 768),
    (768, 1344),
    (1920, 1080),
    (1080, 1920),
    (1280, 720),
    (720, 1280),
    (640, 480),
    (480, 640),
    (512, 512),
    (256, 256),
    (3000, 1000),
    (1000, 3000),
    (3840, 2160),
    (2160, 3840),
    (100, 100),
    (17, 9993),
    (999, 101),
];

/// The probe's targets: the two 768P canvases, the tier canvas and a small one.
const TARGETS: [(i32, i32); 4] = [(1344, 768), (768, 1344), (512, 512), (256, 256)];

/// `match` is cap 0, `max` is the reference's 2048 short-edge cap.
const CAPS: [i32; 2] = [0, 2048];

fn main() -> ExitCode {
    println!("# reference canvas sweep");
    let mut rows = 0;

    for (width, height) in SOURCES {
        match adapt_canvas(width, height) {
            Some((out_w, out_h)) => println!("ADAPT\t{width}\t{height}\t{out_w}\t{out_h}"),
            None => println!("ADAPT\t{width}\t{height}\t-\t-"),
        }
        match reference_video_canvas(width, height) {
            Some((out_w, out_h)) => println!("VIDEO\t{width}\t{height}\t{out_w}\t{out_h}"),
            None => println!("VIDEO\t{width}\t{height}\t-\t-"),
        }
        rows += 2;
    }

    for (width, height) in SOURCES {
        for (target_w, target_h) in TARGETS {
            for cap in CAPS {
                match reference_image_canvas(width, height, target_w, target_h, cap) {
                    Some((out_w, out_h)) => println!(
                        "IMAGE\t{width}\t{height}\t{target_w}\t{target_h}\t{cap}\t{out_w}\t{out_h}"
                    ),
                    None => {
                        println!("IMAGE\t{width}\t{height}\t{target_w}\t{target_h}\t{cap}\t-\t-")
                    }
                }
                rows += 1;
            }
        }
    }

    println!("TOTAL\trows={rows}");
    ExitCode::SUCCESS
}
