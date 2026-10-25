//! Checkpoint partitions: FL2VA for prompt-only requests, Ref2VA for ordered
//! references.
//!
//! Source of truth is `/data/h3.c` at `eae5d54`:
//!
//! - `h3.c:425-436` `h3_create`: `FL2VA/transformer/config.json` and
//!   `FL2VA/tokenizer/tokenizer.json` must be regular files, then
//!   `text_encoder`, `transformer`, `video_vae/source` and `audio_vae` are
//!   inventoried in that order (`h3_st_inventory_dir`,
//!   `h3_safetensors.c:540-585`).
//! - `h3.c:438-452`: Ref2VA is optional. Its presence is the regular file
//!   `Ref2VA/transformer/model.safetensors.index.json`; when it exists the
//!   transformer directory is inventoried too, so a broken optional component
//!   fails at open rather than at the first reference request.
//! - `h3.c:872-875`: any ordered reference requires that component, with the
//!   message this module reproduces verbatim.
//! - `h3.c:927-936`: a reference request reads *every* component from the
//!   Ref2VA partition (tokenizer, text encoder, transformer, video VAE, audio
//!   VAE), not only the transformer.
//! - `h3.c:373-396`: `h3_is_file`, `h3_require_file` and the missing-file
//!   message.
//!
//! What the module does not do: it does not open device state and it does not
//! inventory the four non-transformer Ref2VA components. The reference
//! resolves those paths at generation time (`h3.c:927-936`) and only the
//! transformer is checked at open, so a catalog reports the same surface.

use std::fmt;
use std::path::{Path, PathBuf};

use super::safetensors::{self, Inventory};

/// The prompt-only partition (`h3.c:430-433`).
pub const FL2VA: &str = "FL2VA";
/// The ordered-reference partition (`h3.c:444-452`).
pub const REF2VA: &str = "Ref2VA";

/// The four inventoried components, in `h3.c:430-433`'s order.
pub const COMPONENTS: [&str; 4] = [
    "text_encoder",
    "transformer",
    "video_vae/source",
    "audio_vae",
];

/// The files `h3.c:428-429` requires before any inventory.
const REQUIRED: [&str; 2] = ["transformer/config.json", "tokenizer/tokenizer.json"];

/// The index file whose presence selects the optional partition
/// (`h3.c:444-448`).
const REF2VA_INDEX: &str = "transformer/model.safetensors.index.json";

/// The prompt tokenizer (`h3.c:428-429`, `:927-928`).
pub const TOKENIZER: &str = "tokenizer/tokenizer.json";
/// The Qwen3-VL text and vision encoder (`h3.c:927-930`).
pub const TEXT_ENCODER: &str = "text_encoder";
/// The DiT (`h3.c:931-932`).
pub const TRANSFORMER: &str = "transformer";
/// The video VAE weights (`h3.c:933-934`).
pub const VIDEO_VAE: &str = "video_vae/source";
/// The AudioVAE weights (`h3.c:935-936`).
pub const AUDIO_VAE: &str = "audio_vae";

/// `h3.c:873`: the refusal when a reference request meets a model directory
/// without the optional component.
pub const REFERENCE_REFUSAL: &str = "ordered references require the Ref2VA checkpoint";

/// The partition a request reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partition {
    /// `FL2VA`, prompt-only and FL2VA anchors.
    Fl2va,
    /// `Ref2VA`, ordered image/video/audio references.
    Ref2va,
}

impl Partition {
    /// The directory name (`h3.c:928-935`).
    pub fn directory(self) -> &'static str {
        match self {
            Partition::Fl2va => FL2VA,
            Partition::Ref2va => REF2VA,
        }
    }

    /// One component's path under `root` (`h3.c:927-936`).
    pub fn component(self, root: &Path, relative: &str) -> PathBuf {
        root.join(self.directory()).join(relative)
    }
}

/// Why a model directory cannot serve.
#[derive(Debug)]
pub enum CatalogError {
    /// `h3_require_file` (`h3.c:385-395`): the path is not a regular file.
    Missing(PathBuf),
    /// A shard directory that failed to inventory (`h3_safetensors.c:540-585`).
    /// The detail carries the directory, as `h3_st_inventory_dir` prints it.
    Inventory(String),
    /// `h3.c:873-874`: a reference request without the Ref2VA component.
    References,
}

impl CatalogError {
    pub fn message(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatalogError::Missing(path) => {
                write!(formatter, "missing required model file: {}", path.display())
            }
            CatalogError::Inventory(detail) => formatter.write_str(detail),
            CatalogError::References => formatter.write_str(REFERENCE_REFUSAL),
        }
    }
}

/// What `h3_create` learns about the checkpoint before any device work.
#[derive(Debug)]
pub struct Catalog {
    root: PathBuf,
    fl2va: [Inventory; 4],
    ref2va_transformer: Option<Inventory>,
}

impl Catalog {
    /// `h3_create` (`h3.c:425-452`), minus the device probe.
    ///
    /// Reading four inventories costs one shard-header read per file: the
    /// tensor data is never touched, so a 143 GiB partition opens in
    /// milliseconds.
    pub fn open(root: &Path) -> Result<Catalog, CatalogError> {
        for relative in REQUIRED {
            let path = Partition::Fl2va.component(root, relative);
            if !path.is_file() {
                return Err(CatalogError::Missing(path));
            }
        }

        let mut fl2va = [Inventory::default(); 4];
        for (slot, relative) in fl2va.iter_mut().zip(COMPONENTS) {
            let path = Partition::Fl2va.component(root, relative);
            *slot = inventory(&path)?;
        }

        // The index is the whole presence test: a partially copied component
        // still counts as installed here and fails at its inventory.
        let index = Partition::Ref2va.component(root, REF2VA_INDEX);
        let ref2va_transformer = match index.is_file() {
            true => Some(inventory(&Partition::Ref2va.component(root, TRANSFORMER))?),
            false => None,
        };

        Ok(Catalog {
            root: root.to_path_buf(),
            fl2va,
            ref2va_transformer,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The FL2VA totals, in [`COMPONENTS`] order.
    pub fn fl2va(&self) -> &[Inventory; 4] {
        &self.fl2va
    }

    /// The optional partition's transformer totals (`h3.c:449-452`).
    pub fn ref2va_transformer(&self) -> Option<Inventory> {
        self.ref2va_transformer
    }

    /// `h3.c:438-452`: the optional component is installed.
    pub fn ref2va_present(&self) -> bool {
        self.ref2va_transformer.is_some()
    }

    /// `h3.c:872-875`: which partition a request with `references` ordered
    /// references reads.
    pub fn partition(&self, references: usize) -> Result<Partition, CatalogError> {
        match references {
            0 => Ok(Partition::Fl2va),
            _ if self.ref2va_present() => Ok(Partition::Ref2va),
            _ => Err(CatalogError::References),
        }
    }

    /// One component's path inside a partition (`h3.c:927-936`).
    pub fn component(&self, partition: Partition, relative: &str) -> PathBuf {
        partition.component(&self.root, relative)
    }
}

fn inventory(path: &Path) -> Result<Inventory, CatalogError> {
    safetensors::inventory(path).map_err(|error| CatalogError::Inventory(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A scratch directory per test, named so parallel tests never collide.
    fn scratch(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("ds4-h3-partition-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create");
        path
    }

    fn touch(root: &Path, relative: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("a parent")).expect("create");
        fs::write(&path, b"{}").expect("write");
    }

    fn shard(root: &Path, relative: &str) {
        // One 4-byte F32 tensor named `t`: 8-byte length, header, payload.
        let header = r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("a parent")).expect("create");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        fs::write(&path, bytes).expect("write");
    }

    fn minimal(root: &Path) {
        touch(root, "FL2VA/transformer/config.json");
        touch(root, "FL2VA/tokenizer/tokenizer.json");
        for component in COMPONENTS {
            shard(root, &format!("FL2VA/{component}/model.safetensors"));
        }
    }

    #[test]
    fn open_reports_each_fl2va_component() {
        let root = scratch("components");
        minimal(&root);

        let catalog = Catalog::open(&root).expect("open");
        assert!(!catalog.ref2va_present());
        assert_eq!(catalog.fl2va().len(), COMPONENTS.len());
        for totals in catalog.fl2va() {
            assert_eq!(totals.files, 1);
            assert_eq!(totals.tensors, 1);
            assert_eq!(totals.tensor_bytes, 4);
        }
    }

    #[test]
    fn open_names_the_missing_file() {
        let root = scratch("missing");
        minimal(&root);
        fs::remove_file(root.join("FL2VA/tokenizer/tokenizer.json")).expect("remove");

        let error = Catalog::open(&root).expect_err("open must fail");
        assert_eq!(
            error.message(),
            format!(
                "missing required model file: {}",
                root.join("FL2VA/tokenizer/tokenizer.json").display()
            )
        );
    }

    #[test]
    fn references_without_the_component_are_refused() {
        let root = scratch("refused");
        minimal(&root);
        let catalog = Catalog::open(&root).expect("open");

        assert_eq!(catalog.partition(0).expect("fl2va"), Partition::Fl2va);
        assert_eq!(
            catalog.partition(3).expect_err("refused").message(),
            REFERENCE_REFUSAL
        );
    }

    #[test]
    fn references_select_the_component_when_installed() {
        let root = scratch("installed");
        minimal(&root);
        touch(&root, "Ref2VA/transformer/model.safetensors.index.json");
        shard(&root, "Ref2VA/transformer/model-00001-of-00001.safetensors");

        let catalog = Catalog::open(&root).expect("open");
        assert_eq!(catalog.partition(1).expect("ref2va"), Partition::Ref2va);
        assert_eq!(catalog.ref2va_transformer().expect("installed").files, 1);
        assert_eq!(
            catalog.component(Partition::Ref2va, TRANSFORMER),
            root.join("Ref2VA").join("transformer")
        );
    }
}
