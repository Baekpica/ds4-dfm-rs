//! The CUDA device record `--info` reports, in h3 terms.
//!
//! The reference probes the device while opening a context (`h3.c:456-463`)
//! and prints the record before the checkpoint inventory (`main.c:127-140`,
//! the five fields the CUDA probe fills). The FFI and the field mapping live in
//! `ds4_sys::h3_device`; this module is the h3 host's access point so the CLI
//! stays on the ds4-core boundary (`AGENTS.md`, inference boundary).

pub use ds4_sys::h3_device::{probe, DeviceInfo};
