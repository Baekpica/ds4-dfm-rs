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
#[derive(Clone, Copy)]
struct Site {
    shard: usize,
    begin: usize,
    bytes: usize,
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
                sites.insert(
                    tensor.name.clone(),
                    Site {
                        shard: index,
                        begin,
                        bytes,
                    },
                );
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
        let site = self
            .sites
            .get(name)
            .ok_or_else(|| format!("VAE is missing {name}"))?;
        let elements = shape.iter().product::<u64>();
        if elements * 4 != site.bytes as u64 {
            return Err(format!(
                "{name}: plan {shape:?} needs {} bytes, checkpoint has {}",
                elements * 4,
                site.bytes
            ));
        }
        let map = self.map(site.shard)?;
        let bytes = map
            .as_slice()
            .get(site.begin..site.begin + site.bytes)
            .ok_or_else(|| format!("{name}: range outside its shard"))?;

        let tensor = gpu.alloc(Dtype::F32, elements as usize)?;
        gpu.upload(&tensor, bytes)
            .map_err(|detail| format!("{error}: {detail}"))?;
        Ok(tensor)
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
