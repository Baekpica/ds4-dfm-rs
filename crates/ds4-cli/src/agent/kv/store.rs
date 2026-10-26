use super::catalog::KvError;
use super::identity::{
    decode_title_trailer, encode_title_trailer, file_identity_sha, identity_sha, title_from_text,
};
use ds4_kv::{
    path_for_sha, read_envelope, read_identity, read_trailer, write_checkpoint, Header,
    PayloadGuard, Reason, EXT_SESSION_TITLE,
};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct SwitchPlan {
    pub(crate) sha: String,
    pub(crate) path: PathBuf,
    _held: PayloadGuard,
    pub(crate) needs_prefill: bool,
    pub(crate) title: String,
    pub(crate) created_at: u64,
    pub(crate) text: Vec<u8>,
    pub(crate) payload_offset: u64,
    pub(crate) payload_bytes: u64,
    pub(crate) tokens: u32,
}

pub(crate) struct SaveSpec<'a> {
    pub(crate) title: &'a str,
    pub(crate) created_at: u64,
    pub(crate) last_used: u64,
    pub(crate) text: &'a [u8],
    pub(crate) payload: &'a Path,
    pub(crate) tokens: u32,
    pub(crate) model_id: u8,
    pub(crate) quant_bits: u8,
    pub(crate) ctx_size: u32,
}

pub(crate) struct SessionStore {
    dir: PathBuf,
    model_id: u8,
    artifact: [u8; 32],
}

impl SessionStore {
    pub(crate) fn open(dir: impl Into<PathBuf>, model_id: u8, artifact: [u8; 32]) -> Self {
        Self {
            dir: dir.into(),
            model_id,
            artifact,
        }
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn model_id(&self) -> u8 {
        self.model_id
    }

    pub(crate) fn save(&self, spec: SaveSpec<'_>) -> Result<String, KvError> {
        if spec.quant_bits != 2 && spec.quant_bits != 4 {
            return Err(KvError::Io(
                "unsupported routed quantization for KV save".into(),
            ));
        }
        let sha = identity_sha(spec.title, spec.created_at);
        fs::create_dir_all(&self.dir).map_err(|error| KvError::Io(error.to_string()))?;
        let header = Header {
            quant_bits: spec.quant_bits,
            reason: Reason::AgentSession,
            ext_flags: EXT_SESSION_TITLE,
            model_id: spec.model_id,
            tokens: spec.tokens,
            hits: 0,
            ctx_size: spec.ctx_size,
            created_at: spec.created_at,
            last_used: spec.last_used,
            payload_bytes: 0,
            text_bytes: 0,
        };
        let path = path_for_sha(&self.dir, &sha);
        write_checkpoint(
            &path,
            &header,
            spec.text,
            Some(spec.payload),
            &encode_title_trailer(spec.title),
            Some(self.artifact),
        )
        .map_err(|error| KvError::Io(error.to_string()))?;
        Ok(sha)
    }

    pub(crate) fn delete(&self, prefix: &str) -> Result<String, KvError> {
        let (sha, path) = self.find(prefix)?;
        fs::remove_file(path).map_err(|error| KvError::Io(error.to_string()))?;
        Ok(sha)
    }

    pub(crate) fn strip(
        &self,
        prefix: &str,
        token_count: u32,
        now: u64,
    ) -> Result<String, KvError> {
        let (sha, path) = self.find(prefix)?;
        let held = PayloadGuard::open(&path).map_err(|e| KvError::Io(e.to_string()))?;
        let mut envelope = read_envelope(held.path()).map_err(|e| KvError::Io(e.to_string()))?;
        let (_, trailer) =
            read_trailer(held.path(), u64::MAX).map_err(|e| KvError::Io(e.to_string()))?;
        let title = decode_title_trailer(&trailer);
        if file_identity_sha(
            envelope.header.ext_flags,
            title.as_deref(),
            envelope.header.created_at,
            &envelope.text,
        ) != sha
        {
            return Err(KvError::IdentityMismatch);
        }
        envelope.header.tokens = token_count;
        envelope.header.last_used = now;
        write_checkpoint(
            &path,
            &envelope.header,
            &envelope.text,
            None,
            &trailer,
            None,
        )
        .map_err(|error| KvError::Io(error.to_string()))?;
        Ok(sha)
    }

    pub(crate) fn switch_plan(&self, prefix: &str) -> Result<SwitchPlan, KvError> {
        let (sha, path) = self.find(prefix)?;
        let held = PayloadGuard::open(&path).map_err(|e| KvError::Io(e.to_string()))?;
        let envelope = read_envelope(held.path()).map_err(|e| KvError::Io(e.to_string()))?;
        let (_, trailer) =
            read_trailer(held.path(), u64::MAX).map_err(|e| KvError::Io(e.to_string()))?;
        let trailer_title = decode_title_trailer(&trailer);
        let actual = file_identity_sha(
            envelope.header.ext_flags,
            trailer_title.as_deref(),
            envelope.header.created_at,
            &envelope.text,
        );
        if actual != sha {
            return Err(KvError::IdentityMismatch);
        }
        let title = trailer_title
            .filter(|_| envelope.header.ext_flags & EXT_SESSION_TITLE != 0)
            .unwrap_or_else(|| title_from_text(&envelope.text, 0));
        let compatible = read_identity(held.path()).map_err(|e| KvError::Io(e.to_string()))?
            == Some(self.artifact);

        Ok(SwitchPlan {
            sha,
            path: held.path().to_path_buf(),
            _held: held,
            needs_prefill: envelope.header.payload_bytes == 0 || !compatible,
            title,
            created_at: envelope.header.created_at,
            text: envelope.text,
            payload_offset: envelope.payload_offset,
            payload_bytes: envelope.header.payload_bytes,
            tokens: envelope.header.tokens,
        })
    }
}
