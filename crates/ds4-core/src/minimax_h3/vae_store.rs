//! The F32 weight store both VAE decoders read.
//!
//! This is `h3_weight_store_open` plus `h3_weight_load_f32` (`h3_weights.c:44-114`,
//! `h3_audio_vae.c:128-133`): shards sorted by path, the first shard holding a
//! name wins, each shard mapped on first use and never copied. Both decoders
//! are F32 end to end, so a tensor of any other dtype is refused by name
//! rather than converted. `load_bf16` is the one rounding that exists, for the
//! video decoder's BF16 arm, and it reads the same F32 bytes.
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
    /// F32 is the decoders' own layout; F16 is the latent upscaler's, which
    /// widens the halves itself (`latent_upscale::half_to_f32`). Anything else
    /// stays refused by name.
    f32: bool,
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
                let f32 = match tensor.dtype {
                    StDtype::F32 => true,
                    StDtype::F16 => false,
                    other => {
                        return Err(format!(
                            "{label} tensor {} is {other:?}, not F32 or F16",
                            tensor.name
                        ))
                    }
                };
                let begin = usize::try_from(tensor.file_offset)
                    .map_err(|_| format!("{label} offset too wide"))?;
                let bytes = usize::try_from(tensor.bytes())
                    .map_err(|_| format!("{label} range too wide"))?;
                sites.entry(tensor.name.clone()).or_insert_with(|| Site {
                    shard: index,
                    begin,
                    bytes,
                    shape: tensor.shape.clone(),
                    f32,
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
        self.require_f32(name, error)?;
        let bytes = self.bytes(name, shape)?;
        let tensor = gpu.alloc(Dtype::F32, bytes.len() / 4)?;
        gpu.upload(&tensor, bytes)
            .map_err(|detail| format!("{error}: {detail}"))?;
        Ok(tensor)
    }

    /// Upload `name` as BF16 after the same exact-shape check, rounding every
    /// value to nearest even at the BF16 boundary (`nvfp4::push_bf16`'s
    /// arithmetic, `h3_text_encoder.c:29-36`'s rule). The decoder's BF16 arm is
    /// the only caller: rounding once at load is where the arm wants it, and it
    /// halves the resident bytes of the four weights that arm narrows.
    pub fn load_bf16(
        &self,
        gpu: &Gpu,
        name: &str,
        shape: &[u64],
        error: &str,
    ) -> Result<Tensor, String> {
        self.require_f32(name, error)?;
        let bytes = self.bytes(name, shape)?;
        let narrowed = narrow_bytes(bytes);
        let tensor = gpu.alloc(Dtype::Bf16, narrowed.len() / 2)?;
        gpu.upload(&tensor, &narrowed)
            .map_err(|detail| format!("{error}: {detail}"))?;
        Ok(tensor)
    }

    /// The tensor's raw bytes after the same exact-shape check, for a caller
    /// whose checkpoint is not F32: the latent upscaler's is F16 and widens the
    /// halves itself. Every other caller wants [`Store::load`].
    pub fn raw(&self, name: &str, shape: &[u64]) -> Result<&[u8], String> {
        self.bytes(name, shape)
    }

    /// The F32 readers' guard: a caller that wants raw bytes instead
    /// ([`Store::raw`]) widens or rounds them itself and asks for no promise.
    fn require_f32(&self, name: &str, error: &str) -> Result<(), String> {
        match self.sites.get(name) {
            Some(site) if site.f32 => Ok(()),
            Some(_) => Err(format!("{error}: {name} is not F32")),
            None => Err(format!("{error}: checkpoint is missing {name}")),
        }
    }

    /// The shape the checkpoint records for `name`, so a reference
    /// implementation can read a tensor without restating its dimensions.
    pub fn shape(&self, name: &str) -> Option<Vec<u64>> {
        self.sites.get(name).map(|site| site.shape.clone())
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

/// Host threads a weight is narrowed with, at most. The decoder's four block
/// weights are 9.65 GB of F32 at job A's tile: single-threaded that is 2.1 s on
/// the workstation (0.517 s a 2.42 GB piece), against the 1.70 s the arm's
/// FRAMES wall grew past its own GPU window on the Spark, and 0.5-0.7 s split
/// over the machine's cores.
const NARROW_THREADS: usize = 16;

/// F32 bytes in, BF16 bytes out, nearest even, little endian both ways.
///
/// A BF16 is the F32's top 16 bits, so the round is the F32 word plus half a
/// BF16 ulp (`0x7FFF` in the dropped 16) plus the dropped word's own low bit,
/// which is what makes an exact tie land on an even mantissa. `bits` is taken
/// as unsigned, so a sign bit and a carry into the exponent behave as they do
/// in `text_rope::round_bf16`; NaN and infinity are preserved.
///
/// The ranges are independent per value, so splitting the bytes across threads
/// changes no bit of the result; the chunk sizes follow
/// `minimax_vq::release`'s worker count, without the atomic queue a pure
/// elementwise map does not need.
fn narrow_bytes(bytes: &[u8]) -> Vec<u8> {
    let workers = std::thread::available_parallelism()
        .map_or(1, |cores| cores.get())
        .min(NARROW_THREADS);

    narrow_bytes_with(bytes, workers)
}

/// The same map over `workers` contiguous ranges, one range a serial pass. Only
/// the thread count is a parameter, so a test can hold it fixed and show that
/// the split cannot move a bit.
fn narrow_bytes_with(bytes: &[u8], workers: usize) -> Vec<u8> {
    let values = bytes.as_chunks::<4>().0;
    let mut narrowed = vec![0u8; values.len() * 2];
    let words = narrowed.as_chunks_mut::<2>().0;
    let workers = workers.min(values.len());
    if workers < 2 {
        narrow_into(values, words);
        return narrowed;
    }

    let chunk = values.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let mut rest_values = values;
        let mut rest_words = words;
        while !rest_values.is_empty() {
            let take = chunk.min(rest_values.len());
            let (head_values, tail_values) = rest_values.split_at(take);
            let (head_words, tail_words) = rest_words.split_at_mut(take);
            rest_values = tail_values;
            rest_words = tail_words;
            scope.spawn(move || narrow_into(head_values, head_words));
        }
    });
    narrowed
}

/// One range of `narrow_bytes`: `width` F32 words in, one BF16 each out.
fn narrow_into(values: &[[u8; 4]], words: &mut [[u8; 2]]) {
    for (value, word) in values.iter().zip(words.iter_mut()) {
        let bits = u32::from_le_bytes(*value);
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        *word = ((rounded >> 16) as u16).to_le_bytes();
    }
}

pub fn f1(store: &Store, gpu: &Gpu, name: &str, width: u64) -> Result<Tensor, String> {
    store.load(gpu, name, &[width], name)
}

pub fn f2(store: &Store, gpu: &Gpu, name: &str, rows: u64, columns: u64) -> Result<Tensor, String> {
    store.load(gpu, name, &[rows, columns], name)
}

/// The BF16 arm's two weight widths (`load_bf16`).
pub fn f1_bf16(store: &Store, gpu: &Gpu, name: &str, width: u64) -> Result<Tensor, String> {
    store.load_bf16(gpu, name, &[width], name)
}

pub fn f2_bf16(
    store: &Store,
    gpu: &Gpu,
    name: &str,
    rows: u64,
    columns: u64,
) -> Result<Tensor, String> {
    store.load_bf16(gpu, name, &[rows, columns], name)
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

    /// The BF16 boundary at 1.0: a BF16 ulp is `2^-7`, so an F32 word's dropped
    /// 16 bits tie at `0x8000` and the round goes to whichever neighbour has an
    /// even mantissa, which is `1.0` when the kept word is even and the next
    /// value up when it is odd.
    #[test]
    fn narrows_f32_bytes_to_bf16_by_nearest_even() {
        let words: Vec<f32> = [
            0x3F80_0000, // 1.0, exact
            0x3F80_7FFF, // just below the tie
            0x3F80_8000, // the tie, kept word even: down
            0x3F80_8001, // just above the tie: up
            0x3F81_8000, // the tie, kept word odd: up
            0x3F81_0000, // exactly one BF16 ulp above 1.0
            0x4000_0000, // 2.0
            0xBF80_0000, // -1.0
            0x0000_0000,
            0x7F80_0000, // infinity survives
        ]
        .into_iter()
        .map(f32::from_bits)
        .collect();
        let mut bytes = Vec::new();
        for word in &words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }

        let narrowed = narrow_bytes(&bytes);
        let kept: Vec<u16> = narrowed
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| u16::from_le_bytes(*word))
            .collect();
        assert_eq!(
            kept,
            [0x3F80, 0x3F80, 0x3F80, 0x3F81, 0x3F82, 0x3F81, 0x4000, 0xBF80, 0x0000, 0x7F80]
        );
    }

    /// The load's rounding is split over the machine's cores, so the property
    /// that has to hold is that the split moves nothing: an odd chunk that does
    /// not divide the values, a range per value, and one range, all equal.
    #[test]
    fn narrowing_is_bit_identical_across_ranges() {
        let bytes: Vec<u8> = (0..(1usize << 18))
            .map(|index| (index as u32 ^ (index as u32).rotate_left(7)) as u8)
            .collect();
        let serial = narrow_bytes_with(&bytes, 1);
        assert_eq!(serial.len(), bytes.len() / 2);

        for workers in [2, 3, 8, 64, 1 << 16] {
            assert_eq!(
                narrow_bytes_with(&bytes, workers),
                serial,
                "{workers} ranges"
            );
        }
        assert!(narrow_bytes_with(&[], 8).is_empty());
    }
}
