//! The F32 weight store both VAE decoders read.
//!
//! This is `h3_weight_store_open` plus `h3_weight_load_f32` (`h3_weights.c:44-114`,
//! `h3_audio_vae.c:128-133`): shards sorted by path, the first shard holding a
//! name wins, each shard mapped on first use and never copied. Both decoders
//! are F32 end to end, so a tensor of any other dtype is refused by name
//! rather than converted.
//!
//! `label` only prefixes the error text ("audio VAE", "video VAE"), so a
//! failure names the decoder that hit it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use crate::mapped::MappedFile;
use crate::minimax_h3::safetensors::{self, Dtype as StDtype};

/// One weight, resolved to its shard and byte range.
struct Site {
    shard: usize,
    begin: usize,
    bytes: usize,
    shape: Vec<u64>,
}

pub struct Store {
    shards: Vec<PathBuf>,
    sites: HashMap<String, Site>,
    maps: Vec<OnceLock<MappedFile>>,
}

impl Store {
    pub fn open(dir: &Path, label: &str) -> Result<Store, String> {
        let store =
            safetensors::open_dir(dir).map_err(|error| format!("{label} reader: {error:?}"))?;
        let mut shards = Vec::new();
        let mut sites = HashMap::new();

        for (index, shard) in store.shards().iter().enumerate() {
            shards.push(shard.path.clone());
            for tensor in &shard.tensors {
                if tensor.dtype != StDtype::F32 {
                    return Err(format!("{label} tensor {} is not F32", tensor.name));
                }
                let begin = usize::try_from(tensor.file_offset)
                    .map_err(|_| format!("{label} offset too wide"))?;
                let bytes = usize::try_from(tensor.bytes())
                    .map_err(|_| format!("{label} range too wide"))?;
                sites.entry(tensor.name.clone()).or_insert_with(|| Site {
                    shard: index,
                    begin,
                    bytes,
                    shape: tensor.shape.clone(),
                });
            }
        }
        let maps = (0..shards.len()).map(|_| OnceLock::new()).collect();

        Ok(Store {
            shards,
            sites,
            maps,
        })
    }

    fn map(&self, shard: usize) -> Result<&MappedFile, String> {
        let cell = &self.maps[shard];
        if let Some(map) = cell.get() {
            return Ok(map);
        }
        let map = MappedFile::open_ro(&self.shards[shard])
            .map_err(|error| format!("map VAE shard: {error}"))?;
        let _ = cell.set(map);
        cell.get().ok_or_else(|| "VAE shard map lost".to_string())
    }

    /// Upload `name` as F32 after checking its exact shape, the reference's
    /// `h3_weight_load_f32(..., ndim, shape, ...)` (`h3_audio_vae.c:128-133`).
    pub fn load(
        &self,
        gpu: &Gpu,
        name: &str,
        shape: &[u64],
        error: &str,
    ) -> Result<Tensor, String> {
        let bytes = self.bytes(name, shape)?;
        let tensor = gpu.alloc(Dtype::F32, bytes.len() / 4)?;
        gpu.upload(&tensor, bytes)
            .map_err(|detail| format!("{error}: {detail}"))?;
        Ok(tensor)
    }

    fn bytes(&self, name: &str, shape: &[u64]) -> Result<&[u8], String> {
        let site = self
            .sites
            .get(name)
            .ok_or_else(|| format!("VAE is missing {name}"))?;
        if shape != site.shape {
            return Err(format!(
                "{name}: plan shape {shape:?}, checkpoint has {:?}",
                site.shape
            ));
        }
        let map = self.map(site.shard)?;
        map.as_slice()
            .get(site.begin..site.begin + site.bytes)
            .ok_or_else(|| format!("{name}: range outside its shard"))
    }
}

pub fn f1(store: &Store, gpu: &Gpu, name: &str, width: u64) -> Result<Tensor, String> {
    store.load(gpu, name, &[width], name)
}

pub fn f2(store: &Store, gpu: &Gpu, name: &str, rows: u64, columns: u64) -> Result<Tensor, String> {
    store.load(gpu, name, &[rows, columns], name)
}

pub fn f3(
    store: &Store,
    gpu: &Gpu,
    name: &str,
    first: u64,
    second: u64,
    third: u64,
) -> Result<Tensor, String> {
    store.load(gpu, name, &[first, second, third], name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn fixture(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("h3-vae-store-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn shard(path: &Path, value: u8) {
        let header = r#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[value; 24]);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn upload_requires_exact_shape() {
        let dir = fixture("shape");
        shard(&dir.join("a.safetensors"), 1);
        let store = Store::open(&dir, "test VAE").unwrap();
        assert_eq!(store.bytes("w", &[2, 3]).unwrap(), &[1; 24]);
        assert!(store.bytes("w", &[3, 2]).is_err());
        assert!(store.bytes("w", &[6]).is_err());
        assert!(store.bytes("w", &[u64::MAX, u64::MAX]).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn first_shard_wins() {
        let dir = fixture("duplicate");
        shard(&dir.join("a.safetensors"), 1);
        shard(&dir.join("b.safetensors"), 2);
        let store = Store::open(&dir, "test VAE").unwrap();
        assert_eq!(store.bytes("w", &[2, 3]).unwrap(), &[1; 24]);
        fs::remove_dir_all(dir).unwrap();
    }
}
