//! Role-keyed access to the checked BF16 checkpoint.
//!
//! `minimax_h3::layout` proves the whole shard set before this module touches a
//! payload, so a role lookup cannot come back with the wrong dtype or shape.
//! What remains is the reference's resolution order and its mapping policy:
//!
//! - `h3_weights.c:129-142` `h3_weight_find`: shards in sorted path order, the
//!   first shard holding the name wins, and the shard travels with the tensor.
//! - `h3_weights.c:144-197` `load_tensor`: exact dtype and rank, shape checked
//!   per dimension, element count overflow-checked, then a load by dtype.
//! - `h3_gpu.m:608-660` maps a transformer payload instead of copying it (the
//!   `H3_ZERO_COPY_WEIGHTS` and M5 paths); `h3_gpu_cuda.cu:375-424` reads the
//!   same payload with `pread` into staging for the device upload.
//!
//! One mapping per shard, taken on first use, and borrowed spans handed out:
//! the host never holds a second copy of the 61.7 GiB DiT. `span` is what the
//! device route needs (path, absolute offset, bytes); `bytes` is the mapped
//! view for host code.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::mapped::MappedFile;
use crate::minimax_h3::layout::{Config, Layout, Role};
use crate::minimax_h3::safetensors::{self, Dtype, Shard, Store, Tensor};

/// One tensor's place in the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span<'a> {
    pub path: &'a Path,
    /// `8 + header_size + data_offsets[0]`, the absolute read offset.
    pub file_offset: u64,
    /// The `data_offsets` span; `read_shard` proved it equals
    /// `elements * dtype.size()`.
    pub bytes: u64,
    pub dtype: Dtype,
}

impl Span<'_> {
    /// `h3_st_tensor_elements` (`h3_safetensors.c:480-489`).
    pub fn elements(&self) -> u64 {
        self.bytes / self.dtype.size()
    }
}

/// A checked shard set with every `Role` resolved to one span.
pub struct Weights {
    layout: Layout,
    store: Store,
    sites: BTreeMap<Role, Site>,
    maps: Vec<OnceLock<MappedFile>>,
}

/// Where a role lives: shard index and tensor index, both from `open`.
#[derive(Clone, Copy, Debug)]
struct Site {
    shard: usize,
    tensor: usize,
}

impl Weights {
    /// Read `config.json`, every shard header, and check the layout.
    pub fn open(dir: &Path) -> Result<Weights, LoadError> {
        let config = Config::read(dir).map_err(|error| LoadError::Layout(error.token()))?;
        let store = safetensors::open_dir(dir).map_err(|error| LoadError::Reader(error.token()))?;
        Weights::from_store(&config, store)
    }

    /// Check an already-opened store. Takes ownership because the mapping
    /// table and the spans are keyed to its shard indices.
    pub fn from_store(config: &Config, store: Store) -> Result<Weights, LoadError> {
        let layout =
            Layout::from_store(config, &store).map_err(|error| LoadError::Layout(error.token()))?;
        let sites = resolve(&layout, &store)?;
        let maps = (0..store.shards().len()).map(|_| OnceLock::new()).collect();

        Ok(Weights {
            layout,
            store,
            sites,
            maps,
        })
    }

    pub fn config(&self) -> &Config {
        self.layout.config()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The descriptor a device upload reads from (`h3_gpu_cuda.cu:426-437`).
    pub fn span(&self, role: Role) -> Result<Span<'_>, LoadError> {
        let (shard, tensor) = self.locate(role)?;

        Ok(Span {
            path: &shard.path,
            file_offset: tensor.file_offset,
            bytes: tensor.bytes(),
            dtype: tensor.dtype,
        })
    }

    /// The mapped payload. The shard is mapped on first use.
    pub fn bytes(&self, role: Role) -> Result<&[u8], LoadError> {
        let site = self.sites.get(&role).ok_or(LoadError::Unmapped(role))?;
        let shard = &self.store.shards()[site.shard];
        let tensor = &shard.tensors[site.tensor];
        let map = self.map(site.shard)?;

        let begin = usize::try_from(tensor.file_offset).map_err(|_| LoadError::Range {
            role,
            offset: tensor.file_offset,
        })?;
        let width = usize::try_from(tensor.bytes()).map_err(|_| LoadError::Range {
            role,
            offset: tensor.file_offset,
        })?;
        let end = begin.checked_add(width).ok_or(LoadError::Range {
            role,
            offset: tensor.file_offset,
        })?;

        map.as_slice().get(begin..end).ok_or(LoadError::Bounds {
            role,
            end: end as u64,
            mapped: map.as_slice().len() as u64,
        })
    }

    /// How many shards are mapped right now.
    pub fn mapped(&self) -> usize {
        self.maps.iter().filter(|cell| cell.get().is_some()).count()
    }

    fn locate(&self, role: Role) -> Result<(&Shard, &Tensor), LoadError> {
        let site = self.sites.get(&role).ok_or(LoadError::Unmapped(role))?;
        let shard = &self.store.shards()[site.shard];

        Ok((shard, &shard.tensors[site.tensor]))
    }

    fn map(&self, index: usize) -> Result<&MappedFile, LoadError> {
        let cell = &self.maps[index];

        if let Some(map) = cell.get() {
            return Ok(map);
        }

        let path = &self.store.shards()[index].path;
        let opened = MappedFile::open_ro(path).map_err(|error| LoadError::Map {
            path: path.clone(),
            detail: error.to_string(),
        })?;

        // A concurrent caller that won the race leaves its mapping in place;
        // `opened` drops here and only one mapping survives.
        let _ = cell.set(opened);

        Ok(cell.get().expect("a mapping was just installed"))
    }
}

impl fmt::Debug for Weights {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Weights")
            .field("layout", &self.layout)
            .field("shards", &self.store.shards().len())
            .field("roles", &self.sites.len())
            .field("mapped", &self.mapped())
            .finish()
    }
}

/// `h3_weight_find` order (`h3_weights.c:129-142`): first shard in sorted path
/// order wins. The layout already proved every role has a name in the store,
/// so an absent name here means the layout and the store disagree.
fn resolve(layout: &Layout, store: &Store) -> Result<BTreeMap<Role, Site>, LoadError> {
    let mut first: HashMap<&str, Site> = HashMap::new();

    for (shard, unit) in store.shards().iter().enumerate() {
        for (tensor, entry) in unit.tensors.iter().enumerate() {
            first
                .entry(entry.name.as_str())
                .or_insert(Site { shard, tensor });
        }
    }

    let mut sites = BTreeMap::new();

    for entry in layout.entries() {
        let site = *first
            .get(entry.name.as_str())
            .ok_or(LoadError::Unmapped(entry.role))?;

        sites.insert(entry.role, site);
    }

    Ok(sites)
}

/// Why a checked role could not be read.
#[derive(Debug)]
pub enum LoadError {
    /// The config and the shards do not form the expected checkpoint.
    Layout(String),
    /// The directory could not be read as a shard set.
    Reader(String),
    /// A role the layout did not produce.
    Unmapped(Role),
    Map {
        path: PathBuf,
        detail: String,
    },
    /// An offset that does not fit this process's address space.
    Range {
        role: Role,
        offset: u64,
    },
    /// The mapping is shorter than the tensor's span.
    Bounds {
        role: Role,
        end: u64,
        mapped: u64,
    },
}

impl LoadError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            LoadError::Layout(detail) => format!("layout: {detail}"),
            LoadError::Reader(detail) => format!("reader: {detail}"),
            LoadError::Unmapped(role) => format!("unmapped {role:?}"),
            LoadError::Map { path, detail } => {
                format!("map {}: {detail}", path.display())
            }
            LoadError::Range { role, offset } => {
                format!("range {role:?}: offset {offset} does not fit")
            }
            LoadError::Bounds { role, end, mapped } => {
                format!("bounds {role:?}: span ends at {end}, mapping is {mapped}")
            }
        }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for LoadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ds4-h3-weights-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch directory");

        dir
    }

    #[test]
    fn open_requires_a_config() {
        let dir = scratch("no-config");
        let error = Weights::open(&dir).expect_err("an empty directory fails");

        assert!(error.token().starts_with("layout:"), "{}", error.token());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_requires_shards() {
        let dir = scratch("no-shards");
        let config = br#"{
            "_class_name": "MiniMaxH3DiTModel",
            "hidden_size": 8,
            "num_layers": 1,
            "token_refiner_num_layers": 1,
            "num_attention_heads": 1,
            "attention_head_dim": 2,
            "ffn_hidden_size": 12,
            "latents_dim": 4,
            "audio_latents_dim": 4,
            "patch_size": [1, 2, 2],
            "text_dim": 8,
            "timestep_input_dim": 256,
            "time_embed_hidden_size": 8,
            "time_embed_dim": 8,
            "adaln_out_features": 8,
            "final_adaln_out_features": 8,
            "rope_inv_freq_len": 2
        }"#;
        fs::write(dir.join("config.json"), config).expect("config");

        let error = Weights::open(&dir).expect_err("config without shards fails");

        assert!(error.token().starts_with("reader:"), "{}", error.token());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn span_counts_elements_by_dtype() {
        let span = Span {
            path: Path::new("shard.safetensors"),
            file_offset: 0,
            bytes: 3 * Dtype::Bf16.size(),
            dtype: Dtype::Bf16,
        };

        assert_eq!(span.elements(), 3);
    }
}
