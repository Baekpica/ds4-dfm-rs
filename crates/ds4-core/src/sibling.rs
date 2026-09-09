//! Host-owned MTP / DSpark sibling attach.
//!
//! Bind maps (`BindPlan::resolve_mtp` / `resolve_dspark`) are the source of
//! truth. Native mmap of the sibling GGUF stays in C `model_open` (todo 45
//! KEEP). This module does not store raw sibling model pointers.

use crate::bind::{BindPlan, SupportCatalog};
use crate::layout::{validate_dspark_layouts, validate_mtp_layouts};
use crate::shape::{ModelFamily, Shape};
use crate::tensors::TensorInventory;
use crate::{Error, Result, DSPARK_MARKOV_RANK};

const DEEPSEEK_ONLY: &str = "MTP and DSpark support models are DeepSeek-only";

/// Host-owned sibling attach: path + resolved bind map. No raw C model pointer.
#[derive(Debug)]
pub struct SiblingAttach {
    kind: SupportCatalog,
    path: String,
    bind_plan: BindPlan,
}

impl SiblingAttach {
    pub fn kind(&self) -> SupportCatalog {
        self.kind
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn bind_plan(&self) -> &BindPlan {
        &self.bind_plan
    }
}

pub(crate) struct SiblingPaths<'a> {
    pub mtp: Option<&'a str>,
    pub dspark: Option<&'a str>,
}

/// Resolve MTP / DSpark attach from existing bind maps. Empty paths are absent
/// like C (`mtp_path && mtp_path[0]`). Missing files error; layouts unchanged.
pub(crate) fn attach_siblings(
    family: ModelFamily,
    shape: Shape,
    paths: SiblingPaths<'_>,
) -> Result<(Option<SiblingAttach>, Option<SiblingAttach>)> {
    let mtp_path = nonempty(paths.mtp);
    let dspark_path = nonempty(paths.dspark);
    let mtp_supported = matches!(family, ModelFamily::DeepSeek4 | ModelFamily::Inkling);
    if (mtp_path.is_some() && !mtp_supported)
        || (dspark_path.is_some() && family != ModelFamily::DeepSeek4)
    {
        return Err(Error {
            code: 1,
            message: DEEPSEEK_ONLY.into(),
        });
    }
    let mtp = match mtp_path {
        None => None,
        Some(path) => Some(open_one(SupportCatalog::Mtp, path, shape)?),
    };
    let dspark = match dspark_path {
        None => None,
        Some(path) => Some(open_one(SupportCatalog::Dspark, path, shape)?),
    };
    Ok((mtp, dspark))
}

fn nonempty(path: Option<&str>) -> Option<&str> {
    path.filter(|p| !p.is_empty())
}

fn kind_token(kind: SupportCatalog) -> &'static str {
    match kind {
        SupportCatalog::Mtp => "mtp",
        SupportCatalog::Dspark => "dspark",
    }
}

fn open_one(kind: SupportCatalog, path: &str, shape: Shape) -> Result<SiblingAttach> {
    let token = kind_token(kind);
    if shape.family == ModelFamily::Inkling {
        let g = crate::GgufFile::open(std::path::Path::new(path)).map_err(|e| Error {
            code: 1,
            message: format!("{token} metadata failed: {e}"),
        })?;
        crate::inkling::validate_mtp(&g).map_err(|e| Error {
            code: 1,
            message: format!("{token} metadata failed: {e}"),
        })?;
    }
    let inv = TensorInventory::open(std::path::Path::new(path)).map_err(|e| Error {
        code: 1,
        message: format!("{token} tensor inventory failed: {}", e.token()),
    })?;
    let plan = match kind {
        SupportCatalog::Mtp => BindPlan::resolve_mtp(shape, &inv),
        SupportCatalog::Dspark => BindPlan::resolve_dspark(shape, &inv),
    };
    if let Some(name) = plan.missing_required().first() {
        return Err(Error {
            code: 1,
            message: format!("{token} required tensor is missing: {name}"),
        });
    }
    match kind {
        SupportCatalog::Mtp => validate_mtp_layouts(&plan),
        SupportCatalog::Dspark => validate_dspark_layouts(&plan, DSPARK_MARKOV_RANK),
    }
    .map_err(|e| Error {
        code: 1,
        message: format!("{token} layout failed: {}", e.token()),
    })?;
    Ok(SiblingAttach {
        kind,
        path: path.to_string(),
        bind_plan: plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bind_mtp_names, SHAPE_FLASH, SHAPE_MOTIF3};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn flash(paths: SiblingPaths<'_>) -> Result<(Option<SiblingAttach>, Option<SiblingAttach>)> {
        attach_siblings(ModelFamily::DeepSeek4, SHAPE_FLASH, paths)
    }

    fn temp_gguf(tag: &str, names: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ds4-sibling-{tag}-{}-{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sib.gguf");
        write_gguf(&path, names);
        path
    }

    fn write_gguf(path: &Path, names: &[&str]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(names.len() as u64).to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&(b"general.alignment".len() as u64).to_le_bytes());
        buf.extend_from_slice(b"general.alignment");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());
        for (i, name) in names.iter().enumerate() {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&8u64.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&((i as u64) * 32).to_le_bytes());
        }
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.resize(buf.len() + pad + names.len() * 32, 0);
        fs::write(path, buf).unwrap();
    }

    #[test]
    fn attach_rejects_non_deepseek_like_c() {
        let err = attach_siblings(
            ModelFamily::Motif3,
            SHAPE_MOTIF3,
            SiblingPaths {
                mtp: Some("/tmp/mtp.gguf"),
                dspark: None,
            },
        )
        .unwrap_err();
        assert_eq!(err.message, DEEPSEEK_ONLY);
    }

    #[test]
    fn attach_empty_path_is_absent_like_c() {
        let (mtp, dspark) = flash(SiblingPaths {
            mtp: Some(""),
            dspark: Some(""),
        })
        .unwrap();
        assert!(mtp.is_none() && dspark.is_none());
    }

    #[test]
    fn inkling_mtp_checks_metadata() {
        let path = temp_gguf("inkling-mtp", &["model.mtp.layers.0.embed_norm.weight"]);
        let err = attach_siblings(
            ModelFamily::Inkling,
            crate::shape::SHAPE_INKLING_SMALL,
            SiblingPaths {
                mtp: path.to_str(),
                dspark: None,
            },
        )
        .unwrap_err();
        assert!(
            err.message.starts_with("mtp metadata failed:"),
            "{}",
            err.message
        );
    }

    #[test]
    fn inkling_rejects_dspark() {
        let err = attach_siblings(
            ModelFamily::Inkling,
            crate::shape::SHAPE_INKLING_SMALL,
            SiblingPaths {
                mtp: None,
                dspark: Some("/unused"),
            },
        )
        .unwrap_err();
        assert_eq!(err.message, DEEPSEEK_ONLY);
    }

    #[test]
    #[ignore = "requires the real Inkling MTP-BF16 sidecar"]
    fn attach_inkling_mtp_artifact() {
        let root = std::env::var("INKLING_ARTIFACT_DIR").expect("set INKLING_ARTIFACT_DIR");
        let path = format!("{root}/MTP-BF16/Inkling-Small-MTP-BF16.gguf");
        let (mtp, dspark) = attach_siblings(
            ModelFamily::Inkling,
            crate::shape::SHAPE_INKLING_SMALL,
            SiblingPaths {
                mtp: Some(&path),
                dspark: None,
            },
        )
        .unwrap();
        assert!(dspark.is_none());
        let plan = mtp.unwrap();
        assert_eq!(plan.bind_plan().slots.len(), 160);
        assert!(plan.bind_plan().missing_required().is_empty());
    }

    #[test]
    fn attach_missing_path_errors_like_c() {
        let missing = format!(
            "/tmp/ds4-sibling-missing-{}-{}/nope.gguf",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        );
        let err = flash(SiblingPaths {
            mtp: Some(&missing),
            dspark: None,
        })
        .unwrap_err();
        assert!(
            err.message.starts_with("mtp tensor inventory failed:"),
            "{}",
            err.message
        );
    }

    #[test]
    fn attach_missing_required_uses_bind_map() {
        let mtp = temp_gguf("mtp-miss", &["mtp.0.hc_head_base.weight"]);
        let err = flash(SiblingPaths {
            mtp: Some(mtp.to_str().unwrap()),
            dspark: None,
        })
        .unwrap_err();
        assert_eq!(
            err.message,
            "mtp required tensor is missing: mtp.0.hc_head_fn.weight"
        );
        let ds = temp_gguf("dspark-miss", &["dspark.main_proj.weight"]);
        let err = flash(SiblingPaths {
            mtp: None,
            dspark: Some(ds.to_str().unwrap()),
        })
        .unwrap_err();
        assert_eq!(
            err.message,
            "dspark required tensor is missing: dspark.main_norm.weight"
        );
        let _ = fs::remove_dir_all(mtp.parent().unwrap());
        let _ = fs::remove_dir_all(ds.parent().unwrap());
    }

    #[test]
    fn attach_mtp_complete_names_uses_bind_map_then_layout() {
        let owned = bind_mtp_names();
        let names: Vec<&str> = owned.iter().map(|n| n.name.as_str()).collect();
        let path = temp_gguf("mtp-names", &names);
        let err = flash(SiblingPaths {
            mtp: Some(path.to_str().unwrap()),
            dspark: None,
        })
        .unwrap_err();
        assert!(
            err.message.starts_with("mtp layout failed:"),
            "{}",
            err.message
        );
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }
}
