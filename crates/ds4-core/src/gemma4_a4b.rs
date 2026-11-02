//! Gemma-4-26B-A4B as a dense-VQ text family: the artifact contract.
//!
//! This module is the family's own folder, not an extension of the MiniMax-H3
//! tower (`minimax_h3/gemma.rs` serves that video pipeline and is not this
//! family's forward). What it binds:
//!
//! - `config.json`, the reference checkpoint's configuration copied verbatim,
//!   states the two attention geometries, the 128-expert block, the tied
//!   embedding and the softcapped logits;
//! - `vq-release.json` (schema `vq-release/1`, written by `vq-family release`
//!   in the VQ tools repository) names every tensor file with its role, shape,
//!   container geometry and digest;
//! - [`layout`] classifies those names into one role table per layer and
//!   checks each shape against the configuration;
//! - [`release`] checks every file's size and container header, and can
//!   re-read every digest on a gate path;
//! - [`forward`] is the CPU reference executor: one token at a time, f32,
//!   with the weights streamed through [`weights`] rather than decoded;
//! - [`tokenizer`] is the checkpoint's own BPE plus the Gemma-4 chat template
//!   and an incremental decoder.
//!
//! What this module does not do yet, named so it is not claimed: a batched
//! prefill, a device path (the native serving lanes still refuse this family),
//! the vision sidecar, tool rendering and a logits-level reference gate. The
//! OpenAI-compatible host is the separate `gemma4-server-rs` binary; it is not
//! the native engine.

mod config;
mod decode;
mod forward;
mod layout;
mod model;
mod release;
mod tokenizer;
mod weights;

use std::fmt;

/// A named failure of the family contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    pub fn new(message: impl Into<String>) -> Error {
        Error(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub use config::{Config, LayerType};
pub use decode::{decode, matvec, Tensor};
pub use forward::{Model, Session};
pub use layout::{Layer, Layout, Role};
pub use model::{identify, is_release, Family, NAME_PREFIX};
pub use release::{Codebook, Entry, Payload, Release, Totals};
pub use tokenizer::{Decoder, Tokenizer};
pub use weights::Matrix;
