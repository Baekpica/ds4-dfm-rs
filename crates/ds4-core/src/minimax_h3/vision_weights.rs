//! Role-keyed access to the vision tower's 351 tensors.
//!
//! The tower's weights live inside the FL2VA text partition: the reference
//! hands `h3_vision_encode_bf16` the same directory it hands the prompt
//! encoder (`h3.c:1347`), and resolves `model.visual.*` names there. This
//! module resolves the same shard order and checks shapes, keyed by the vision
//! role table.
//!
//! Policy, unchanged from the reference:
//!
//! - `h3_weight_find` (`h3_weights.c:129-142`): shards in sorted path order,
//!   the first shard holding a name wins.
//! - `h3_weight_load_bf16` (`h3_weights.c:144-197`): dtype and every
//!   dimension checked before a byte is read; [`crate::minimax_h3::vision`]
//!   does that at open time.
//! - `h3_st_read_data` (`h3_safetensors.c:492-534`): the read is the tensor's
//!   own byte span, `pread` into a host buffer.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;

use crate::minimax_h3::safetensors::{self, Store};
use crate::minimax_h3::vision::{Role, VisionError, VisionLayout};

/// A checked vision path over one shard set.
pub struct VisionWeights {
    layout: VisionLayout,
    store: Store,
    sites: BTreeMap<Role, Site>,
}

/// Where a role lives: shard index and tensor index.
#[derive(Clone, Copy, Debug)]
struct Site {
    shard: usize,
    tensor: usize,
}

impl VisionWeights {
    /// Open `dir`, read every shard header, and resolve the vision path.
    pub fn open(dir: &Path) -> Result<VisionWeights, VisionLoadError> {
        let store =
            safetensors::open_dir(dir).map_err(|error| VisionLoadError::Reader(error.token()))?;
        VisionWeights::from_store(store)
    }

    /// Check an already-opened store.
    pub fn from_store(store: Store) -> Result<VisionWeights, VisionLoadError> {
        let layout = check(&store)?;
        let sites = resolve(&layout, &store)?;

        Ok(VisionWeights {
            layout,
            store,
            sites,
        })
    }

    pub fn layout(&self) -> &VisionLayout {
        &self.layout
    }

    /// `h3_st_read_data`: read the tensor's own span into `buffer`.
    pub fn read_into(&self, role: Role, buffer: &mut Vec<u8>) -> Result<(), VisionLoadError> {
        let site = *self
            .sites
            .get(&role)
            .ok_or(VisionLoadError::Unmapped(role))?;
        let shard = &self.store.shards()[site.shard];
        let tensor = &shard.tensors[site.tensor];
        let width = usize::try_from(tensor.bytes()).map_err(|_| VisionLoadError::Range {
            role,
            offset: tensor.file_offset,
        })?;

        buffer.clear();
        buffer.resize(width, 0);
        shard
            .read_data(tensor, buffer)
            .map_err(|error| VisionLoadError::Reader(error.token()))
    }
}

impl fmt::Debug for VisionWeights {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VisionWeights")
            .field("shards", &self.store.shards().len())
            .field("roles", &self.sites.len())
            .finish()
    }
}

/// Check the vision table against every tensor the store holds.
fn check(store: &Store) -> Result<VisionLayout, VisionLoadError> {
    let mut tensors = Vec::new();

    for shard in store.shards() {
        for tensor in &shard.tensors {
            tensors.push(crate::minimax_h3::vision::TensorRef {
                name: &tensor.name,
                dtype: tensor.dtype,
                shape: &tensor.shape,
            });
        }
    }

    VisionLayout::build(&tensors).map_err(VisionLoadError::Layout)
}

/// `h3_weight_find` order (`h3_weights.c:129-142`): first shard in sorted path
/// order wins.
fn resolve(layout: &VisionLayout, store: &Store) -> Result<BTreeMap<Role, Site>, VisionLoadError> {
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
            .ok_or(VisionLoadError::Unmapped(entry.role))?;

        sites.insert(entry.role, site);
    }

    Ok(sites)
}

/// Why the vision path could not be opened.
#[derive(Debug)]
pub enum VisionLoadError {
    /// The store does not carry the vision tower's tensors.
    Layout(VisionError),
    /// The directory could not be read as a shard set.
    Reader(String),
    /// A role the layout did not produce.
    Unmapped(Role),
    /// An offset that does not fit this process's address space.
    Range { role: Role, offset: u64 },
}

impl VisionLoadError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            VisionLoadError::Layout(error) => error.token(),
            VisionLoadError::Reader(detail) => format!("reader: {detail}"),
            VisionLoadError::Unmapped(role) => format!("unmapped {role:?}"),
            VisionLoadError::Range { role, offset } => {
                format!("range {role:?}: offset {offset} does not fit")
            }
        }
    }
}

impl fmt::Display for VisionLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for VisionLoadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ds4-h3-vision-weights-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch directory");

        dir
    }

    #[test]
    fn open_requires_shards() {
        let dir = scratch("no-shards");
        let error = VisionWeights::open(&dir).expect_err("an empty directory fails");

        assert!(error.token().starts_with("reader:"), "{}", error.token());
        let _ = fs::remove_dir_all(&dir);
    }
}
