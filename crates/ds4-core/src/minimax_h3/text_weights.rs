//! Role-keyed access to the prompt encoder's tensors.
//!
//! The DiT's [`crate::minimax_h3::weights::Weights`] is keyed by the DiT role
//! enum; the encoder has its own 551-tensor path (`minimax_h3::text`), so it
//! needs its own resolution table over the same shard set. The policy is the
//! reference's, unchanged:
//!
//! - `h3_weights.c:129-142` `h3_weight_find`: shards in sorted path order, the
//!   first shard holding the name wins.
//! - `h3_weights.c:144-197` `load_tensor`: the dtype and every dimension are
//!   checked before a byte is read. `minimax_h3::text` already proved all 551
//!   descriptors, so here a lookup only fails if the store changed underneath
//!   the layout.
//! - `h3_gpu_cuda.cu:375-424`: the device route reads the tensor's own byte
//!   span. This module hands out the mapped view; the forward uploads it.
//!
//! One mapping per shard, taken on first use: the encoder reads 48.8 GiB of
//! weights and the host must not hold a second copy.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::mapped::MappedFile;
use crate::minimax_h3::safetensors::{self, Store};
use crate::minimax_h3::text::{Config, Layout, Role};

/// A checked encoder shard set with every role resolved to one tensor.
pub struct TextWeights {
    layout: Layout,
    store: Store,
    sites: BTreeMap<Role, Site>,
    maps: Vec<OnceLock<MappedFile>>,
}

/// Where a role lives: shard index and tensor index.
#[derive(Clone, Copy, Debug)]
struct Site {
    shard: usize,
    tensor: usize,
}

impl TextWeights {
    /// Read `config.json`, every shard header, and resolve the encoder path.
    pub fn open(dir: &Path) -> Result<TextWeights, TextLoadError> {
        let config = Config::read(dir).map_err(|error| TextLoadError::Layout(error.token()))?;
        let store =
            safetensors::open_dir(dir).map_err(|error| TextLoadError::Reader(error.token()))?;

        TextWeights::from_store(&config, store)
    }

    /// Check an already-opened store.
    pub fn from_store(config: &Config, store: Store) -> Result<TextWeights, TextLoadError> {
        let layout = Layout::with_store(config, &store)
            .map_err(|error| TextLoadError::Layout(error.token()))?;
        let sites = resolve(&layout, &store)?;
        let maps = (0..store.shards().len()).map(|_| OnceLock::new()).collect();

        Ok(TextWeights {
            layout,
            store,
            sites,
            maps,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The mapped payload. The shard is mapped on first use.
    pub fn bytes(&self, role: Role) -> Result<&[u8], TextLoadError> {
        let site = *self.sites.get(&role).ok_or(TextLoadError::Unmapped(role))?;
        let shard = &self.store.shards()[site.shard];
        let tensor = &shard.tensors[site.tensor];
        let map = self.map(site.shard)?;

        let begin = usize::try_from(tensor.file_offset).map_err(|_| TextLoadError::Range {
            role,
            offset: tensor.file_offset,
        })?;
        let width = usize::try_from(tensor.bytes()).map_err(|_| TextLoadError::Range {
            role,
            offset: tensor.file_offset,
        })?;
        let end = begin.checked_add(width).ok_or(TextLoadError::Range {
            role,
            offset: tensor.file_offset,
        })?;

        map.as_slice().get(begin..end).ok_or(TextLoadError::Bounds {
            role,
            end: end as u64,
            mapped: map.as_slice().len() as u64,
        })
    }

    /// `h3_st_read_data` (`h3_safetensors.c:492-534`): read the tensor's own
    /// span into `buffer`, sized to the tensor.
    ///
    /// The reference's loader reads with `pread` into a host buffer
    /// (`h3_gpu_cuda.cu:455-478`), and the prefetch lanes need to own their
    /// bytes, so this is the read the forward uses. [`Self::bytes`] stays for
    /// callers that only want the mapped view.
    pub fn read_into(&self, role: Role, buffer: &mut Vec<u8>) -> Result<(), TextLoadError> {
        let site = *self.sites.get(&role).ok_or(TextLoadError::Unmapped(role))?;
        let shard = &self.store.shards()[site.shard];
        let tensor = &shard.tensors[site.tensor];
        let width = usize::try_from(tensor.bytes()).map_err(|_| TextLoadError::Range {
            role,
            offset: tensor.file_offset,
        })?;

        buffer.clear();
        buffer.resize(width, 0);
        shard
            .read_data(tensor, buffer)
            .map_err(|error| TextLoadError::Reader(error.token()))
    }

    /// How many shards are mapped right now.
    pub fn mapped(&self) -> usize {
        self.maps.iter().filter(|cell| cell.get().is_some()).count()
    }

    fn map(&self, index: usize) -> Result<&MappedFile, TextLoadError> {
        let cell = &self.maps[index];

        if let Some(map) = cell.get() {
            return Ok(map);
        }

        let path = &self.store.shards()[index].path;
        let opened = MappedFile::open_ro(path).map_err(|error| TextLoadError::Map {
            path: path.clone(),
            detail: error.to_string(),
        })?;

        // A concurrent caller that won the race leaves its mapping in place;
        // `opened` drops here and one mapping survives.
        let _ = cell.set(opened);

        Ok(cell.get().expect("a mapping was just installed"))
    }
}

impl fmt::Debug for TextWeights {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TextWeights")
            .field("shards", &self.store.shards().len())
            .field("roles", &self.sites.len())
            .field("mapped", &self.mapped())
            .finish()
    }
}

/// `h3_weight_find` order (`h3_weights.c:129-142`): first shard in sorted path
/// order wins. The layout already proved every role has a name in the store.
fn resolve(layout: &Layout, store: &Store) -> Result<BTreeMap<Role, Site>, TextLoadError> {
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
            .ok_or(TextLoadError::Unmapped(entry.role))?;

        sites.insert(entry.role, site);
    }

    Ok(sites)
}

/// Why a checked encoder role could not be read.
#[derive(Debug)]
pub enum TextLoadError {
    /// The config and the shards do not form the encoder path.
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

impl TextLoadError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            TextLoadError::Layout(detail) => format!("layout: {detail}"),
            TextLoadError::Reader(detail) => format!("reader: {detail}"),
            TextLoadError::Unmapped(role) => format!("unmapped {role:?}"),
            TextLoadError::Map { path, detail } => format!("map {}: {detail}", path.display()),
            TextLoadError::Range { role, offset } => {
                format!("range {role:?}: offset {offset} does not fit")
            }
            TextLoadError::Bounds { role, end, mapped } => {
                format!("bounds {role:?}: span ends at {end}, mapping is {mapped}")
            }
        }
    }
}

impl fmt::Display for TextLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for TextLoadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ds4-h3-text-weights-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch directory");

        dir
    }

    #[test]
    fn open_requires_a_config() {
        let dir = scratch("no-config");
        let error = TextWeights::open(&dir).expect_err("an empty directory fails");

        assert!(error.token().starts_with("layout:"), "{}", error.token());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_requires_shards() {
        let dir = scratch("no-shards");
        let config = br#"{
            "model_type": "qwen3_vl",
            "text_config": {
                "head_dim": 128,
                "hidden_size": 5120,
                "intermediate_size": 25600,
                "num_attention_heads": 64,
                "num_hidden_layers": 64,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_theta": 5000000,
                "vocab_size": 151936
            }
        }"#;
        fs::write(dir.join("config.json"), config).expect("config");

        let error = TextWeights::open(&dir).expect_err("config without shards fails");

        assert!(error.token().starts_with("reader:"), "{}", error.token());
        let _ = fs::remove_dir_all(&dir);
    }
}
