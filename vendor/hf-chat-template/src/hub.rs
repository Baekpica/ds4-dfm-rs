//! Optional Hugging Face Hub integration (`hub` feature).
//!
//! Fetches a model's template material straight from the Hub via the synchronous (`ureq`)
//! client, so callers can go from a repo id to a compiled template in one call.
//! Authentication follows `hf-hub`'s own discovery: the `HF_TOKEN` env var or the cached
//! token written by `huggingface-cli login`, which is what gated repos need.

use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};

use crate::config::TokenizerConfig;
use crate::error::Error;

/// Filename `transformers` uses for a standalone chat template (`utils.CHAT_TEMPLATE_FILE`).
/// When present it takes precedence over any inline `chat_template` in `tokenizer_config.json`.
const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";

/// Fetch the template material for `repo_id` at `revision` (`None` = default branch): the parsed
/// `tokenizer_config.json` (always, for special tokens) plus the standalone `chat_template.jinja`
/// if the repo ships one.
///
/// Presence of the standalone file is decided from the repo's file list (`info()`), so a genuine
/// fetch failure is never silently mistaken for "the repo has no standalone template". Network,
/// auth, and missing-file failures surface as [`Error::Hub`]; a malformed config as
/// [`Error::Config`].
pub(crate) fn fetch_template_material(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<(TokenizerConfig, Option<String>), Error> {
    let api = Api::new().map_err(|e| Error::Hub(e.to_string()))?;
    let repo = repo_handle(&api, repo_id, revision);

    let info = repo.info().map_err(|e| Error::Hub(e.to_string()))?;
    let has_standalone = info
        .siblings
        .iter()
        .any(|s| s.rfilename == CHAT_TEMPLATE_FILE);

    let config = {
        let path = repo
            .get("tokenizer_config.json")
            .map_err(|e| Error::Hub(e.to_string()))?;
        let bytes = std::fs::read(&path).map_err(|e| Error::Hub(e.to_string()))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(e.to_string()))?
    };

    let standalone = if has_standalone {
        let path = repo
            .get(CHAT_TEMPLATE_FILE)
            .map_err(|e| Error::Hub(e.to_string()))?;
        Some(std::fs::read_to_string(&path).map_err(|e| Error::Hub(e.to_string()))?)
    } else {
        None
    };

    Ok((config, standalone))
}

fn repo_handle(api: &Api, repo_id: &str, revision: Option<&str>) -> ApiRepo {
    match revision {
        Some(rev) => api.repo(Repo::with_revision(
            repo_id.to_owned(),
            RepoType::Model,
            rev.to_owned(),
        )),
        None => api.model(repo_id.to_owned()),
    }
}
