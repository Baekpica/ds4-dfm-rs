//! The family's open path: release -> configuration -> layout.
//!
//! Opening binds names to roles and checks every file's header and size. It
//! does not hash the artifact (that is [`Release::verify`]) and it does not
//! touch a device: the forward, the attention kernels and the serving surface
//! are later units.

use std::path::{Path, PathBuf};

use super::config::Config;
use super::layout::{self, Layer, Layout, Role};
use super::release::{self, Entry, Release};
use super::Error;

/// An opened artifact: the configuration, the release and the bound layout.
#[derive(Clone, Debug)]
pub struct Family {
    pub dir: PathBuf,
    pub config: Config,
    pub release: Release,
    pub layout: Layout,
}

impl Family {
    /// Open an artifact directory. The release is read first, so a directory
    /// without one fails as "not this family" before any other file is read.
    pub fn open(dir: &Path) -> Result<Family, Error> {
        if !dir.is_dir() {
            return Err(Error::new(format!("{}: not a directory", dir.display())));
        }
        let release = release::open(dir)?;
        let config = match dir.join("config.json") {
            path if path.exists() => {
                let stated = Config::open(&path)?;
                if stated != Config::embedded()? {
                    return Err(Error::new(format!(
                        "{}: configuration differs from the family contract",
                        path.display()
                    )));
                }
                stated
            }
            _ => Config::embedded()?,
        };
        let layout = layout::check(release.entries(), &config)?;
        Ok(Family {
            dir: dir.to_path_buf(),
            config,
            release,
            layout,
        })
    }

    /// A layer's tensor for one role.
    pub fn role(&self, layer: usize, role: Role) -> Result<&Entry, Error> {
        let layer = self.layout.layers.get(layer).ok_or_else(|| {
            Error::new(format!(
                "layer {layer} is outside 0..{}",
                self.layout.layers.len()
            ))
        })?;
        layer.get(role)
    }

    pub fn layer(&self, layer: usize) -> Result<&Layer, Error> {
        self.layout.layers.get(layer).ok_or_else(|| {
            Error::new(format!(
                "layer {layer} is outside 0..{}",
                self.layout.layers.len()
            ))
        })
    }

    /// Re-read every file's sha256 against the release. A gate path, not the
    /// serving path.
    pub fn verify(&self) -> Result<(), Error> {
        self.release.verify()
    }
}

/// The prefix every entry of this artifact carries: `layout::check` insists
/// on it, and it is what distinguishes this release from the other VQ
/// families'. Theirs index GGUF part files (`axk2_vq/release.rs:145-231`, and
/// the same shape in `beam_vq`, `minimax_vq`, `ling31_vq`, `glm_vq`), while
/// this one lists HF-named tensors from the original checkpoint.
pub const NAME_PREFIX: &str = "model.language_model.";

/// Is `dir` a directory whose release names this schema and lists this
/// family's tensors? One small JSON read, used before the full open in an
/// identification chain. The schema string alone is shared by every
/// `vq-release/1` artifact, so the entry naming is what makes the claim.
pub fn is_release(dir: &Path) -> bool {
    let path = dir.join(release::FILE);
    let Ok(bytes) = std::fs::read(&path) else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    let schema = value.get("schema").and_then(serde_json::Value::as_str);
    if schema != Some(release::SCHEMA) {
        return false;
    }
    value
        .get("entries")
        .and_then(serde_json::Value::as_array)
        .and_then(|entries| entries.first())
        .and_then(|entry| entry.get("name"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|name| name.starts_with(NAME_PREFIX))
}

/// Identify a directory that already claimed this family through
/// [`is_release`]. The release has no GGUF header: `split_count` is one
/// artifact rather than a shard set, `n_tensors` counts the release's named
/// entries, and the GGUF-only fields are zero instead of borrowed from another
/// file.
pub fn identify(dir: &Path) -> Result<crate::Identified, crate::IdentifyError> {
    let family = Family::open(dir).map_err(|e| crate::IdentifyError::Artifact(e.to_string()))?;
    Ok(crate::Identified {
        shape: crate::shape_for_variant(crate::Variant::Gemma4A4bVq),
        architecture: Some(NAME_PREFIX.trim_end_matches('.').as_bytes().to_vec()),
        split_count: 1,
        n_kv: 0,
        n_tensors: family.release.entries().len() as u64,
        alignment: 0,
        version: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_without_a_release_is_not_this_family() {
        let dir = std::env::temp_dir().join(format!("ds4-gemma4-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!is_release(&dir));
        let error = Family::open(&dir).unwrap_err().to_string();
        assert!(error.contains("vq-release.json"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
