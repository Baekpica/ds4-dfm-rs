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

pub mod layout;
pub mod safetensors;
pub mod weights;
