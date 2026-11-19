//! HYV4FlashEngramLayer (Transformers c2ec8505). The release packs 41
//! independent 98-byte rows into each 4 KiB page of a layer-local file.
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::gguf::{GgufFile, GGUF_VALUE_INT32, GGUF_VALUE_UINT64};
use crate::uring::{Req, Ring, ERING_CAP};
use crate::TensorError;

const PAGE: usize = 4096;
const PER_PAGE: u64 = 41;
const RECORD: usize = 98;
const HEADS: usize = 16;
const LAYERS: usize = 4;
const PAD: i32 = 120002;
const IMAGE: i32 = 120120;
const VIDEO: i32 = 120124;
const LAYER_BYTES: usize = HEADS * RECORD;

#[derive(Deserialize)]
struct Manifest {
    tables: [Table; LAYERS],
}

#[derive(Deserialize)]
struct Table {
    file: String,
    layer: u32,
    rows: u64,
    row_base: u64,
    file_bytes: u64,
    token_map_entries: usize,
    token_map_sha256: String,
    compressed_vocab_size: u32,
    head_vocab_sizes: [u64; HEADS],
    head_offsets: [u64; HEADS],
    hash_multipliers: [u64; 3],
}

struct Layer {
    file: File,
    table: Table,
}

#[repr(align(4096))]
#[derive(Clone)]
struct Page([u8; PAGE]);

struct Read {
    layer: usize,
    page: u64,
    within: usize,
    dest: usize,
}

pub(crate) struct Engram {
    map: Vec<u32>,
    layers: Vec<Layer>,
    ring: Option<Ring>,
    pages: Vec<Page>,
    reads: Vec<Read>,
}

fn error(msg: impl Into<String>) -> TensorError {
    TensorError::Hy4(msg.into())
}

fn gguf_u64s(g: &GgufFile, key: &str, expected: &[u64]) -> bool {
    let Some(a) = g.get_array(key) else {
        return false;
    };
    if a.typ != GGUF_VALUE_UINT64 || a.len != expected.len() as u64 {
        return false;
    }
    let bytes = &g.as_bytes()[a.data_pos..a.data_pos + expected.len() * 8];
    bytes
        .chunks_exact(8)
        .zip(expected)
        .all(|(b, v)| b == v.to_le_bytes())
}

fn open_table(path: &Path, table: Table) -> Result<Layer, TensorError> {
    let mut options = File::options();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_DIRECT);
    }
    let path = path.join(&table.file);
    let file = options
        .open(&path)
        .map_err(|e| error(format!("{}: {e}", path.display())))?;
    let size = file.metadata().map_err(|e| error(e.to_string()))?.len();
    if table.rows.div_ceil(PER_PAGE) * PAGE as u64 != size || size != table.file_bytes {
        return Err(error(format!("Engram file length {}", path.display())));
    }
    Ok(Layer { file, table })
}

impl Engram {
    pub(crate) fn open(path: &Path, g: &GgufFile) -> Result<Self, TensorError> {
        let path = super::release::root(path)?;
        let manifest: Manifest = serde_json::from_slice(&super::release::table_manifest(path)?)
            .map_err(|e| error(e.to_string()))?;
        let a = g
            .get_array("hy_v4_flash.engram.token_map")
            .filter(|a| a.typ == GGUF_VALUE_INT32 && a.len == 120832)
            .ok_or_else(|| error("Engram token map"))?;
        let first = &manifest.tables[0];
        // The converter padded the LUT to the embedding width. hash_ngrams
        // maps those extra token IDs to the compressed pad ID.
        let bytes = &g.as_bytes()[a.data_pos..a.data_pos + first.token_map_entries * 4];
        let sha: String = Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if sha != first.token_map_sha256 {
            return Err(error("Engram token map digest"));
        }
        let map: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if map.iter().any(|&v| v >= first.compressed_vocab_size) || map.len() <= PAD as usize {
            return Err(error("Engram token map range"));
        }
        let offsets: Vec<_> = manifest
            .tables
            .iter()
            .flat_map(|t| t.head_offsets.map(|v| v + t.row_base))
            .collect();
        let sizes: Vec<_> = manifest
            .tables
            .iter()
            .flat_map(|t| t.head_vocab_sizes)
            .collect();
        let factors: Vec<_> = manifest
            .tables
            .iter()
            .flat_map(|t| t.hash_multipliers)
            .collect();
        if !gguf_u64s(g, "hy_v4_flash.ple.head_offsets", &offsets)
            || !gguf_u64s(g, "hy_v4_flash.ple.head_vocab_sizes", &sizes)
            || !gguf_u64s(g, "hy_v4_flash.ple.layer_multipliers", &factors)
        {
            return Err(error("Engram trunk/table hash geometry"));
        }
        let layers = manifest
            .tables
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                if t.layer != 1 + 10 * i as u32 {
                    return Err(error("Engram layer order"));
                }
                open_table(&path.join("engram"), t)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            map,
            layers,
            ring: Ring::open(ERING_CAP),
            pages: Vec::new(),
            reads: Vec::new(),
        })
    }

    // Positions belong to one sequence. The caller supplies its last two
    // accepted token IDs, even after a bank fork or a speculative rejection.
    pub(crate) fn gather(
        &mut self,
        history: &[i32],
        tokens: &[i32],
        layers: std::ops::Range<usize>,
        dst: &mut [u8],
    ) -> Result<(), TensorError> {
        if history.len() > 2
            || layers.is_empty()
            || layers.end > self.layers.len()
            || tokens.len().checked_mul(layers.len() * LAYER_BYTES) != Some(dst.len())
            || tokens.len() > super::PREFILL_MAX as usize
            || history
                .iter()
                .chain(tokens)
                .any(|t| !(0..120832).contains(t))
        {
            return Err(error("Engram gather bounds"));
        }
        let compress = |t: i32| {
            self.map
                .get(t as usize)
                .copied()
                .unwrap_or(self.map[PAD as usize]) as u64
        };
        self.reads.clear();
        let mut previous = [compress(PAD); 2];
        for &t in history {
            previous = [compress(t), previous[0]];
        }
        for (i, &token) in tokens.iter().enumerate() {
            let ids = [compress(token), previous[0], previous[1]];
            previous = [ids[0], ids[1]];
            for layer in layers.clone() {
                let l = &self.layers[layer];
                let dest = ((layer - layers.start) * tokens.len() + i) * LAYER_BYTES;
                if matches!(token, PAD | IMAGE | VIDEO) {
                    dst[dest..dest + HEADS * RECORD].fill(0);
                    continue;
                }
                for (head, row) in hash_rows(&l.table, ids).into_iter().enumerate() {
                    self.reads.push(Read {
                        layer,
                        page: row / PER_PAGE,
                        within: (row % PER_PAGE) as usize * RECORD,
                        dest: dest + head * RECORD,
                    });
                }
            }
        }
        self.reads
            .sort_unstable_by_key(|r| (r.layer, r.page, r.within));
        if let Some(ring) = &mut self.ring {
            // One request per distinct record, all in one round: the ring
            // keeps ERING_CAP reads in flight through bounce slots. Draining
            // each 512-read batch took ~112 ms per 2,048-token table, one
            // round ~100 ms. The rest is the DRAM-less SSD's mapping misses:
            // the same table read again right away takes ~45 ms.
            let records = self
                .reads
                .chunk_by(|a, b| (a.layer, a.page, a.within) == (b.layer, b.page, b.within));
            let requests = ring.reqs_mut();
            requests.clear();
            for group in records.clone() {
                let r = &group[0];
                requests.push(Req {
                    fd: self.layers[r.layer].file.as_raw_fd(),
                    direct: true,
                    len: RECORD as u32,
                    off: r.page * PAGE as u64 + r.within as u64,
                    dst: dst[r.dest..r.dest + RECORD].as_mut_ptr(),
                });
            }
            if !ring.submit() || !ring.wait() {
                return Err(error("Engram page read failed"));
            }
            // Repeated n-grams read their record once.
            for group in records {
                let (first, rest) = group.split_first().unwrap();
                for r in rest {
                    dst.copy_within(first.dest..first.dest + RECORD, r.dest);
                }
            }
            return Ok(());
        }
        let mut groups = self
            .reads
            .chunk_by(|a, b| a.layer == b.layer && a.page == b.page);
        let mut batch = Vec::with_capacity(ERING_CAP as usize);
        loop {
            batch.clear();
            batch.extend(groups.by_ref().take(ERING_CAP as usize));
            if batch.is_empty() {
                return Ok(());
            }
            self.pages
                .resize_with(self.pages.len().max(batch.len()), || Page([0; PAGE]));
            for (group, page) in batch.iter().zip(&mut self.pages) {
                self.layers[group[0].layer]
                    .file
                    .read_exact_at(&mut page.0, group[0].page * PAGE as u64)
                    .map_err(|e| error(format!("Engram page read: {e}")))?;
            }
            for (group, page) in batch.iter().zip(&self.pages) {
                for r in *group {
                    dst[r.dest..r.dest + RECORD]
                        .copy_from_slice(&page.0[r.within..r.within + RECORD]);
                }
            }
        }
    }
}

fn hash_rows(t: &Table, ids: [u64; 3]) -> [u64; HEADS] {
    let mixed = std::array::from_fn::<_, 3, _>(|i| ids[i] * t.hash_multipliers[i]);
    let two = mixed[0] ^ mixed[1];
    std::array::from_fn(|h| {
        let hash = if h < 8 { two } else { two ^ mixed[2] };
        t.head_offsets[h] + hash % t.head_vocab_sizes[h]
    })
}

// The numerical graph supplies the preceding two accepted token IDs. There
// is no mutable token history in the SSD reader for another bank to inherit.
pub(crate) unsafe extern "C" fn read_rows(
    user: *mut std::ffi::c_void,
    previous: *const i32,
    n_previous: u32,
    tokens: *const i32,
    count: u32,
    first_layer: u32,
    n_layers: u32,
    dst: *mut u8,
    bytes: usize,
) -> i32 {
    if user.is_null()
        || tokens.is_null()
        || dst.is_null()
        || count == 0
        || count > super::PREFILL_MAX
        || n_previous > 2
        || (n_previous != 0 && previous.is_null())
        || n_layers == 0
        || n_layers > LAYERS as u32
        || first_layer > LAYERS as u32 - n_layers
        || bytes != count as usize * n_layers as usize * LAYER_BYTES
    {
        return 0;
    }
    // The Model owns the boxed reader until native closes. Its single
    // execution thread lends disjoint input/output slices for this call.
    let reader = unsafe { &mut *user.cast::<Engram>() };
    let prev = if n_previous == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(previous, n_previous as usize) }
    };
    let ids = unsafe { std::slice::from_raw_parts(tokens, count as usize) };
    let rows = unsafe { std::slice::from_raw_parts_mut(dst, bytes) };
    match reader.gather(
        prev,
        ids,
        first_layer as usize..(first_layer + n_layers) as usize,
        rows,
    ) {
        Ok(()) => 1,
        Err(e) => {
            eprintln!("ds4: {e}");
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Deserialize)]
    struct Vectors {
        tokens: Vec<i32>,
        rows: Vec<Option<[[u64; HEADS]; LAYERS]>>,
    }
    #[test]
    #[ignore = "set HY4_MODEL_PATH to the complete release"]
    fn ssd_rows_across_chunks_and_media() {
        let root =
            std::path::PathBuf::from(std::env::var_os("HY4_MODEL_PATH").expect("HY4_MODEL_PATH"));
        let g = super::super::metadata(&root).unwrap();
        let mut reader = Engram::open(&root, &g).unwrap();
        let vectors: Vectors = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/hy4/engram-rows.json"
        ))
        .unwrap();
        let files: Vec<_> = reader
            .layers
            .iter()
            .map(|l| File::open(root.join("engram").join(&l.table.file)).unwrap())
            .collect();
        for width in [1, 2, 5, vectors.tokens.len()] {
            for (chunk, ids) in vectors.tokens.chunks(width).enumerate() {
                let start = chunk * width;
                let mut result = vec![0xff; ids.len() * LAYER_BYTES * LAYERS];
                reader
                    .gather(
                        &vectors.tokens[start.saturating_sub(2)..start],
                        ids,
                        0..LAYERS,
                        &mut result,
                    )
                    .unwrap();
                let mut staged = vec![0; ids.len() * LAYER_BYTES];
                for layer in 0..LAYERS {
                    reader
                        .gather(
                            &vectors.tokens[start.saturating_sub(2)..start],
                            ids,
                            layer..layer + 1,
                            &mut staged,
                        )
                        .unwrap();
                    assert_eq!(
                        staged,
                        result[layer * staged.len()..(layer + 1) * staged.len()]
                    );
                }
                for (r, rows) in vectors.rows[start..start + ids.len()].iter().enumerate() {
                    for (il, file) in files.iter().enumerate() {
                        for h in 0..HEADS {
                            let mut expected = [0u8; RECORD];
                            if let Some(table) = rows {
                                let id = table[il][h];
                                file.read_exact_at(
                                    &mut expected,
                                    id / PER_PAGE * PAGE as u64 + id % PER_PAGE * RECORD as u64,
                                )
                                .unwrap();
                            }
                            let off = ((il * ids.len() + r) * HEADS + h) * RECORD;
                            assert_eq!(
                                result[off..off + RECORD],
                                expected,
                                "row {r} head {h} layer {il}"
                            );
                        }
                    }
                }
            }
        }
    }
}
