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

/// Validate an MTP sidecar the way `Model::open` will: family acceptance,
/// sidecar metadata, required tensors, and layouts. `--check-config` uses
/// this so a syntactically valid but incompatible GGUF fails before listen.
/// Validate a DSpark drafter the way the open will. Only DeepSeek accepts
/// one, only on an undistributed launch — the open reads the
/// `DS4_DSPARK_MODEL` fallback just there, so a distributed run would serve
/// without the drafter it was told to use.
pub fn probe_dspark_sidecar(
    shape: Shape,
    distributed: Option<&crate::DistributedConfig>,
    path: &str,
) -> Result<()> {
    if path.is_empty() {
        return Err(Error {
            code: 1,
            message: "dspark path must not be empty".into(),
        });
    }
    if distributed.is_some() {
        return Err(Error {
            code: 1,
            message: "a distributed launch does not attach a DSpark drafter".into(),
        });
    }
    attach_siblings(
        shape.family,
        shape,
        SiblingPaths {
            mtp: None,
            dspark: Some(path),
        },
    )
    .map(|_| ())
}

pub fn probe_mtp_sidecar(
    shape: Shape,
    distributed: Option<&crate::DistributedConfig>,
    path: &str,
) -> Result<()> {
    // `attach_siblings` reads an empty path as "no sidecar", but the plan
    // reads `Some("")` as loaded weights, so answer it here.
    if path.is_empty() {
        return Err(Error {
            code: 1,
            message: "mtp path must not be empty".into(),
        });
    }
    // The open loads `mtp_path` only when the role is none, so a distributed
    // process would run without the drafter it was told to use.
    if distributed.is_some() {
        return Err(Error {
            code: 1,
            message: "a distributed launch does not attach an MTP sidecar".into(),
        });
    }
    attach_siblings(
        shape.family,
        shape,
        SiblingPaths {
            mtp: Some(path),
            dspark: None,
        },
    )
    .map(|_| ())
}

/// What `glm53_vision_weights_bind` requires of the encoder before it binds
/// a tensor: architecture, tensor count, every config value, and the three
/// token ids. Tensor shapes stay with the native binder.
const GLM_VISION_ARCH: &[u8] = b"glm5-next-vision";
const GLM_VISION_TENSORS: u64 = 347;
/// Every required tensor, BF16 with these exact ranks and dims — the table
/// `glm53_vision_weights_bind` asserts before it binds an offset.
const GLM_VISION_LAYERS: u32 = 24;
const GLM_VISION_TENSORS_SPEC: [(&str, &[u64]); 11] = [
    (
        "model.visual.patch_embed.proj.weight",
        &[14, 14, 2, 3, 1024],
    ),
    ("model.visual.patch_embed.proj.bias", &[1024]),
    ("model.visual.post_layernorm.weight", &[1024]),
    ("model.visual.downsample.weight", &[2, 2, 1024, 4096]),
    ("model.visual.downsample.bias", &[4096]),
    ("model.visual.merger.proj.weight", &[4096, 4096]),
    ("model.visual.merger.post_projection_norm.weight", &[4096]),
    ("model.visual.merger.post_projection_norm.bias", &[4096]),
    ("model.visual.merger.gate_proj.weight", &[4096, 10240]),
    ("model.visual.merger.up_proj.weight", &[4096, 10240]),
    ("model.visual.merger.down_proj.weight", &[10240, 4096]),
];
const GLM_VISION_LAYER_SPEC: [(&str, &[u64]); 14] = [
    ("norm1.weight", &[1024]),
    ("attn.qkv.weight", &[1024, 3072]),
    ("attn.qkv.bias", &[3072]),
    ("attn.q_norm.weight", &[64]),
    ("attn.k_norm.weight", &[64]),
    ("attn.proj.weight", &[1024, 1024]),
    ("attn.proj.bias", &[1024]),
    ("norm2.weight", &[1024]),
    ("mlp.gate_proj.weight", &[1024, 4096]),
    ("mlp.gate_proj.bias", &[4096]),
    ("mlp.up_proj.weight", &[1024, 4096]),
    ("mlp.up_proj.bias", &[4096]),
    ("mlp.down_proj.weight", &[4096, 1024]),
    ("mlp.down_proj.bias", &[1024]),
];

const GLM_VISION_CONFIG: [(&str, u32); 12] = [
    ("glm5-next-vision.block_count", 24),
    ("glm5-next-vision.embedding_length", 1024),
    ("glm5-next-vision.feed_forward_length", 4096),
    ("glm5-next-vision.attention.head_count", 16),
    ("glm5-next-vision.projection_length", 4096),
    ("glm5-next-vision.projection.feed_forward_length", 10240),
    ("glm5-next-vision.patch_size", 14),
    ("glm5-next-vision.temporal_patch_size", 2),
    ("glm5-next-vision.spatial_merge_size", 2),
    ("glm5-next-vision.image_token_id", 154854),
    ("glm5-next-vision.image_start_token_id", 154830),
    ("glm5-next-vision.image_end_token_id", 154831),
];

/// Validate an external vision encoder the way `model_open` will: only a
/// full GLM-5.3 or Step CUDA model takes one, and then the artifact itself
/// is opened. `--check-config` uses this so a process that cannot boot is
/// not approved.
fn glm_vision_tensor(inventory: &crate::TensorInventory, name: &str, dims: &[u64]) -> Result<()> {
    let t = inventory.find(name).ok_or_else(|| Error {
        code: 1,
        message: format!("vision tensor {name} is missing"),
    })?;
    if crate::tensor_type_name(t.typ) != "bf16" || t.ndim as usize != dims.len() {
        return Err(Error {
            code: 1,
            message: format!(
                "vision tensor {name} has type {}/rank {}, expected BF16/rank {}",
                crate::tensor_type_name(t.typ),
                t.ndim,
                dims.len()
            ),
        });
    }
    for (d, want) in dims.iter().enumerate() {
        if t.dim[d] != *want {
            return Err(Error {
                code: 1,
                message: format!(
                    "vision tensor {name} has dim[{d}]={}, expected {want}",
                    t.dim[d]
                ),
            });
        }
    }
    Ok(())
}

pub fn probe_vision_sidecar(
    shape: Shape,
    backend: crate::Backend,
    distributed: Option<&crate::DistributedConfig>,
    path: &str,
) -> Result<()> {
    if path.is_empty() {
        return Err(Error {
            code: 1,
            message: "vision path must not be empty".into(),
        });
    }
    if shape.family == ModelFamily::Inkling {
        return Err(Error {
            code: 1,
            message: "Inkling uses embedded image/audio weights".into(),
        });
    }
    let takes_encoder = matches!(shape.family, ModelFamily::Glm53 | ModelFamily::Step37);
    if !takes_encoder || backend != crate::Backend::Cuda || distributed.is_some() {
        return Err(Error {
            code: 1,
            message: "--vision requires one full GLM-5.3 or Step CUDA model".into(),
        });
    }
    if shape.family == ModelFamily::Step37 {
        return crate::Step37SidecarPlan::inspect(
            std::path::Path::new(path),
            crate::Step37Sidecar::Vision,
        )
        .map(|_| ())
        .map_err(|e| Error {
            code: 1,
            message: format!("vision metadata failed: {e}"),
        });
    }
    let g = crate::GgufFile::open(std::path::Path::new(path)).map_err(|e| Error {
        code: 1,
        message: format!("vision open failed: {e}"),
    })?;
    if g.get_string("general.architecture") != Some(GLM_VISION_ARCH) {
        return Err(Error {
            code: 1,
            message: "--vision file is not a GLM-5.3 vision encoder GGUF".into(),
        });
    }
    if g.n_tensors != GLM_VISION_TENSORS {
        return Err(Error {
            code: 1,
            message: format!(
                "vision GGUF has {} tensors, expected {GLM_VISION_TENSORS}",
                g.n_tensors
            ),
        });
    }
    for (key, want) in GLM_VISION_CONFIG {
        match g.get_u32(key) {
            Some(id) if id == want => {}
            Some(id) => {
                return Err(Error {
                    code: 1,
                    message: format!("vision {key} is {id}, expected {want}"),
                })
            }
            None => {
                return Err(Error {
                    code: 1,
                    message: format!("vision metadata {key} is missing"),
                })
            }
        }
    }
    let inventory =
        crate::TensorInventory::open(std::path::Path::new(path)).map_err(|e| Error {
            code: 1,
            message: format!("vision tensor inventory failed: {}", e.token()),
        })?;
    for (name, dims) in GLM_VISION_TENSORS_SPEC {
        glm_vision_tensor(&inventory, name, dims)?;
    }
    for il in 0..GLM_VISION_LAYERS {
        for (suffix, dims) in GLM_VISION_LAYER_SPEC {
            glm_vision_tensor(
                &inventory,
                &format!("model.visual.blocks.{il}.{suffix}"),
                dims,
            )?;
        }
    }
    Ok(())
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
    let mtp_supported = matches!(
        family,
        ModelFamily::DeepSeek4 | ModelFamily::Inkling | ModelFamily::Step37
    );
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
    if shape.family == ModelFamily::Step37 {
        crate::Step37SidecarPlan::inspect(std::path::Path::new(path), crate::Step37Sidecar::Mtp)
            .map_err(|e| Error {
                code: 1,
                message: format!("{token} metadata failed: {e}"),
            })?;
    }
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
    fn step_mtp_checks_metadata() {
        let path = temp_gguf("step-mtp", &["blk.45.nextn.enorm.weight"]);
        let err = attach_siblings(
            ModelFamily::Step37,
            crate::shape::SHAPE_STEP37_FLASH,
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
    #[ignore = "requires the real Step MTP-Q8 sidecar"]
    fn attach_step_mtp_artifact() {
        let path = std::env::var("STEP37_MTP").expect("set STEP37_MTP");
        let (mtp, dspark) = attach_siblings(
            ModelFamily::Step37,
            crate::shape::SHAPE_STEP37_FLASH,
            SiblingPaths {
                mtp: Some(&path),
                dspark: None,
            },
        )
        .unwrap();
        assert!(dspark.is_none());
        assert_eq!(mtp.unwrap().bind_plan().slots.len(), 55);
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
