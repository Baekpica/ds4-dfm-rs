//! MiniMax-H3 (h3) checkpoint access.
//!
//! Additive family: nothing in this module is called by an existing model
//! family, and nothing in an existing family's path is edited to serve it
//! (`AGENTS.md`, "MiniMax-H3 (h3) is additive").
//!
//! Source of truth is the reference engine `/data/h3.c` at `eae5d54`:
//!
//! - `h3_safetensors.c:367-460` `h3_st_read_header`: an 8-byte little-endian
//!   JSON length, a 256 MiB cap, then one entry per tensor with `dtype`,
//!   `shape` and `data_offsets`. Every entry is validated against the file
//!   size and against its own element count before it is kept.
//! - `h3_safetensors.c:240-320` the shape and descriptor rules: rank <= 8,
//!   unsigned integers only, `data_offsets` exactly two values.
//! - `h3_safetensors.c:322-345` `h3_validate_tensor`: offsets inside the
//!   data area, no element-count or byte-count overflow, and
//!   `elements * dtype_size == data_end - data_begin`.
//! - `h3_safetensors.c:540-585` `h3_st_inventory_dir`: `.safetensors` files
//!   only, no dot-files, totals of files, file bytes, tensors and tensor
//!   bytes.
//! - `h3_weights.c:44-114` `h3_weight_store_open`: a directory scan, sorted
//!   by path, every shard header parsed up front; `h3_weight_find` searches
//!   shards in that order.
//!
//! The reference never reads the HF index file. The port does, because P3's
//! gate is a two-sided manifest check (index against shard headers), so
//! `reconcile` compares the two and reports any disagreement.

pub mod condition;
pub mod device;
pub mod dit;
pub mod engine;
pub mod host;
pub mod layout;
pub mod media;
pub mod multimodal;
pub mod partition;
pub mod presets;
pub mod reference;
pub mod render;
pub mod safetensors;
pub mod sequence;
pub mod terminal;
pub mod text;
pub mod text_forward;
pub mod text_rope;
pub mod text_weights;
pub mod tokenizer;
pub mod vae_audio;
pub mod vae_encoder;
pub mod vae_store;
pub mod vae_video;
pub mod video_api;
pub mod vision;
pub mod vision_forward;
pub mod vision_weights;
pub mod weights;
