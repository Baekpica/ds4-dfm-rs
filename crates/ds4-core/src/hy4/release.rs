use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::tensors::align_up;
use crate::{GgufError, GgufFile, Identified, IdentifyError, TensorError, TensorInventory};

const MANIFEST: &str = "release.json";
const FIRST: &str = "Hy4-flash-MQ-Q5-SSD-ENGRAM-FP8-00001-of-00004.gguf";
const EMBEDDING: &str = "embed-tokens-bf16.gguf";
// Public release 94cdace56aaa0afc463311e31a2bbc3e613dcce2. Pins the part
// filenames, their digests, placement and the original model revision.
const RELEASE_SHA: &str = "25058aa6c69896898c0f157d255df9663ad849e6e936970c767e5c4140cd2e4e";
const ENGRAM_SHA: &str = "fa3c6e9917a66e6ebb71cf03200605790551e130d23b524e7d83612f2ab50f6e";

type Result<T> = std::result::Result<T, TensorError>;

#[derive(Deserialize)]
struct Part {
    files: Vec<PartFile>,
}

#[derive(Deserialize)]
struct PartFile {
    path: String,
    bytes: u64,
}

#[derive(Deserialize)]
struct Parts {
    trunk: Part,
    embedding: Part,
    engram: Part,
}

#[derive(Deserialize)]
struct Release {
    format: String,
    parts: Parts,
}

fn fail(message: impl Into<String>) -> TensorError {
    TensorError::Hy4(message.into())
}

fn small_file(path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    let mut bytes = Vec::new();
    file.take(65_537)
        .read_to_end(&mut bytes)
        .map_err(GgufError::Io)?;
    if bytes.len() > 65_536 {
        return Err(fail(format!("oversized manifest {}", path.display())));
    }
    Ok(bytes)
}

pub(super) fn pinned(path: &Path, sha: &str) -> Result<Vec<u8>> {
    let bytes = small_file(path)?;
    if Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        != sha
    {
        return Err(fail(format!("manifest digest {}", path.display())));
    }
    Ok(bytes)
}

pub(crate) fn is_release(root: &Path) -> bool {
    small_file(&root.join(MANIFEST))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["format"] == "hy4-mxq-release")
}

pub(crate) fn root(path: &Path) -> Result<&Path> {
    if path.is_dir() {
        Ok(path)
    } else if path.file_name().is_some_and(|n| n == FIRST) {
        Ok(path.parent().unwrap_or(Path::new(".")))
    } else {
        Err(fail(
            "expected Hy4 release directory or its first trunk shard",
        ))
    }
}

fn manifest(root: &Path) -> Result<Release> {
    let release: Release = serde_json::from_slice(&pinned(&root.join(MANIFEST), RELEASE_SHA)?)
        .map_err(|e| fail(e.to_string()))?;
    if release.format != "hy4-mxq-release" {
        return Err(fail("release format"));
    }
    Ok(release)
}

pub(crate) fn trunk_bytes(path: &Path) -> Result<u64> {
    let release = manifest(root(path)?)?;
    Ok(release.parts.trunk.files.iter().map(|f| f.bytes).sum())
}

/// External inputs that affect a Hy4 text snapshot, beyond its GGUF shards.
pub fn snapshot_inputs(path: &Path) -> Result<Vec<PathBuf>> {
    let root = root(path)?;
    let mut files: Vec<_> = manifest(root)?
        .parts
        .engram
        .files
        .into_iter()
        .map(|f| root.join(f.path))
        .collect();
    files.extend(
        [
            MANIFEST,
            "engram/engram-manifest.json",
            "chat_template.jinja",
            "tokenizer_config.json",
        ]
        .map(|name| root.join(name)),
    );
    Ok(files)
}

pub(crate) fn metadata(path: &Path) -> Result<GgufFile> {
    let root = root(path)?;
    manifest(root)?;
    let g = GgufFile::open(&root.join(FIRST))?;
    super::validate(&g).map_err(|e| fail(e.token()))?;
    Ok(g)
}

pub(crate) fn identify(path: &Path) -> std::result::Result<Identified, IdentifyError> {
    let g = metadata(path).map_err(|e| IdentifyError::Artifact(e.token()))?;
    crate::identify_file(&g)
}

pub(super) fn table_manifest(root: &Path) -> Result<Vec<u8>> {
    pinned(&root.join("engram/engram-manifest.json"), ENGRAM_SHA)
}

fn check_size(root: &Path, file: &PartFile) -> Result<PathBuf> {
    let path = root.join(&file.path);
    let m = std::fs::metadata(&path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    if !m.is_file() || m.len() != file.bytes {
        return Err(fail(format!("incomplete {}", file.path)));
    }
    Ok(path)
}

pub(crate) fn inventory(path: &Path) -> Result<TensorInventory> {
    let root = root(path)?;
    let release = manifest(root)?;
    table_manifest(root)?;
    for f in release
        .parts
        .trunk
        .files
        .iter()
        .chain(&release.parts.embedding.files)
        .chain(&release.parts.engram.files)
    {
        check_size(root, f)?;
    }
    let first = root.join(FIRST);
    let g = GgufFile::open(&first)?;
    super::validate(&g).map_err(|e| fail(e.token()))?;
    let mut inv = TensorInventory::from_file(&first, &g)?;
    if inv.tensors.len() != 1386 || inv.shards.len() != 4 {
        return Err(fail("trunk tensor or shard count"));
    }
    let path = root.join(EMBEDDING);
    let g = GgufFile::open(&path)?;
    if g.get_string("hy4_mxq.part") != Some(b"embedding")
        || g.get_string("hy4_mxq.residency") != Some(b"ssd-rows")
        || g.get_string("general.source.revision") != Some(super::SOURCE)
        || g.split_count() > 1
        || g.n_tensors != 1
    {
        return Err(fail("token embedding GGUF"));
    }
    let mut embed = TensorInventory::from_file(&path, &g)?;
    let tensor = &mut embed.tensors[0];
    if tensor.name != "token_embd.weight"
        || tensor.ndim != 2
        || tensor.dim[..2] != [3072, 120832]
        || tensor.typ != 30
    {
        return Err(fail("BF16 token embedding layout"));
    }
    let last = inv.shards.last().ok_or_else(|| fail("missing trunk"))?;
    let offset = last
        .base
        .checked_add(align_up(last.size, inv.page))
        .ok_or(TensorError::Overflow)?;
    tensor.shard = inv.shards.len() as u32;
    tensor.abs_offset = offset
        .checked_add(tensor.abs_offset)
        .ok_or(TensorError::Overflow)?;
    embed.shards[0].base = offset;
    inv.shards.extend(embed.shards);
    inv.tensors.extend(embed.tensors);
    Ok(inv)
}

pub(crate) fn chat_assets(root: &Path) -> Result<()> {
    pinned(
        &root.join("chat_template.jinja"),
        "63ddd5de294df36f10c8f2c8e5f7b2a9dc2011c8481335422f04554256aee4ad",
    )?;
    pinned(
        &root.join("tokenizer_config.json"),
        "99454b593e1f893db010ab3d8968d32ff886e3711051e3e52243f0f6fc3b92dc",
    )?;
    Ok(())
}
