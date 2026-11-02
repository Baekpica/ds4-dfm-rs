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
//!   re-read every digest on a gate path.
//!
//! What this module does not do yet, named so it is not claimed: the forward
//! pass (sliding and full attention, the RoPE split, the routed experts), the
//! logits, the tokenizer and the token-grammar chat protocol, the vision
//! sidecar and the OpenAI serving surface. Those are later units; this one
//! ends when the folder opens the reference artifact and refuses every
//! disagreement it can see without compute.

mod config;
mod layout;
mod model;
mod release;

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
pub use layout::{Layer, Layout, Role};
pub use model::{identify, is_release, Family, NAME_PREFIX};
pub use release::{Codebook, Entry, Payload, Release, Totals};
