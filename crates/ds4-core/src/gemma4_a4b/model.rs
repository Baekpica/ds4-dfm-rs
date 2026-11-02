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

/// Is `dir` a directory whose release names this schema? Cheap: one small
/// JSON read, used before the full open in an identification chain.
pub fn is_release(dir: &Path) -> bool {
    let path = dir.join(release::FILE);
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| {
                v.get("schema")
                    .and_then(serde_json::Value::as_str)
                    .map(|s| s == release::SCHEMA)
            })
            .unwrap_or(false),
        Err(_) => false,
    }
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
