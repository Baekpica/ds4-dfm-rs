//! The release keeps the source checkpoint's configuration beside its named
//! GGUF parts, pinned by SHA-256 (release.rs ASSETS): one topology, whatever
//! the part metadata claims.
use std::path::Path;

use super::release;
use crate::{identify_file, GgufFile, Identified, IdentifyError};

fn error(value: impl ToString) -> IdentifyError {
    IdentifyError::Artifact(value.to_string())
}

pub(super) fn identify(root: &Path) -> Result<Identified, IdentifyError> {
    release::asset(root, "config.json").map_err(error)?;
    let g = GgufFile::open(&release::first_part(root).map_err(error)?)?;
    let mut id = identify_file(&g)?;
    if id.shape.family != crate::ModelFamily::Inkling2Vq {
        return Err(error("configuration/part architecture mismatch"));
    }
    id.split_count = release::part_count(root).map_err(error)?;
    Ok(id)
}
