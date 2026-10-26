use super::catalog::KvError;
use super::identity::{title_from_prompt, title_from_text, SessionIdentity};
use super::store::{SaveSpec, SessionStore};
use super::{parse_slash, SlashCmd};
use ds4_core::{Model, Session, TokenBuffer};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) struct Live<'a, 'm> {
    pub(crate) store: &'a SessionStore,
    pub(crate) identity: &'a mut SessionIdentity,
    pub(crate) model: &'a Model,
    pub(crate) session: &'a mut Session<'m>,
    pub(crate) transcript: &'a mut TokenBuffer,
}

impl SessionIdentity {
    pub(crate) fn note_user_prompt(&mut self, prompt: &str, now: u64) {
        if self.has_user_turn {
            return;
        }
        self.title = Some(title_from_prompt(prompt));
        self.created_at = now;
        self.has_user_turn = true;
    }
}

pub(crate) fn handle_slash(line: &str, live: Live<'_, '_>) -> Option<String> {
    let cmd = match parse_slash(line)? {
        Ok(cmd) => cmd,
        Err(usage) => return Some(format!("{usage}\n")),
    };
    Some(match cmd {
        SlashCmd::Save => save(live),
        SlashCmd::List => list(live.store),
        SlashCmd::Switch(prefix) => switch(prefix, live),
        SlashCmd::Del(prefix) => match live.store.delete(prefix) {
            Ok(sha) => format!("deleted session {:.8}\n", sha_short(&sha)),
            Err(error) => format!("delete failed: {error}\n"),
        },
        SlashCmd::Strip(prefix) => strip(prefix, live),
    })
}

fn save(live: Live<'_, '_>) -> String {
    if !live.identity.has_user_turn {
        return format!("save failed: {}\n", KvError::NothingToSave);
    }
    let text = match render_tokens(live.model, live.transcript.as_slice()) {
        Ok(text) => text,
        Err(error) => return format!("save failed: {error}\n"),
    };
    let title = live
        .identity
        .title
        .clone()
        .unwrap_or_else(|| title_from_text(&text, 0));
    let created_at = if live.identity.created_at == 0 {
        unix_now()
    } else {
        live.identity.created_at
    };
    let payload = match stage_payload(live.store, live.session) {
        Ok(payload) => payload,
        Err(error) => return format!("save failed: {error}\n"),
    };
    let tokens = u32::try_from(live.transcript.len()).unwrap_or(u32::MAX);
    let model_id = u8::try_from(live.model.model_id()).unwrap_or(0);
    let quant_bits = u8::try_from(live.model.routed_quant_bits()).unwrap_or(0);
    let ctx_size = u32::try_from(live.session.ctx()).unwrap_or(0);
    match live.store.save(SaveSpec {
        title: &title,
        created_at,
        last_used: unix_now(),
        text: &text,
        payload: payload.path(),
        tokens,
        model_id,
        quant_bits,
        ctx_size,
    }) {
        Ok(sha) => {
            live.identity.title = Some(title);
            live.identity.created_at = created_at;
            live.identity.sha = Some(sha.clone());
            format!("saved session {:.8} ({tokens} tokens)\n", sha_short(&sha))
        }
        Err(error) => format!("save failed: {error}\n"),
    }
}

fn list(store: &SessionStore) -> String {
    match store.list() {
        Ok(sessions) if sessions.is_empty() => "no saved sessions\n".into(),
        Ok(sessions) => {
            let now = unix_now();
            let mut out = String::new();
            for session in &sessions {
                let when = if session.last_used != 0 {
                    session.last_used
                } else {
                    session.created_at
                };
                let stripped = if session.stripped { ", stripped" } else { "" };
                out.push_str(&format!(
                    "{:.8} > {}\n         > {}, {} tokens, {:.2} MB{stripped}\n\n",
                    sha_short(&session.sha),
                    session.title,
                    format_age(now, when),
                    session.tokens,
                    session.file_size as f64 / (1024.0 * 1024.0),
                ));
            }
            out.push_str(
                "Use /switch <id> to select a session, /del <id> to remove, /strip <id> to strip KV cache.\n",
            );
            out
        }
        Err(error) => format!("no sessions: {error}\n"),
    }
}

fn switch(prefix: &str, live: Live<'_, '_>) -> String {
    let plan = match live.store.switch_plan(prefix) {
        Ok(plan) => plan,
        Err(error) => return format!("switch failed: {error}\n"),
    };
    let mut out = String::new();
    if plan.needs_prefill {
        out.push_str(&format!(
            "rebuilding session {:.8} from rendered text...\n",
            sha_short(&plan.sha)
        ));
        let ids = live.model.vocab().encode_rendered_bytes(&plan.text);
        *live.transcript = TokenBuffer::from_tokens(ids);
        if let Err(error) = live.session.sync(live.transcript) {
            return format!("{out}switch failed: {error}\n");
        }
    } else if let Err(error) =
        live.session
            .load_payload_range(&plan.path, plan.payload_offset, plan.payload_bytes)
    {
        return format!("switch failed: {error}\n");
    } else {
        *live.transcript = TokenBuffer::from_tokens(live.session.host().tokens().to_vec());
    }
    live.identity.title = Some(plan.title);
    live.identity.created_at = if plan.created_at == 0 {
        unix_now()
    } else {
        plan.created_at
    };
    live.identity.sha = Some(plan.sha.clone());
    live.identity.has_user_turn = true;
    let rebuilt = if plan.needs_prefill {
        ", rebuilt from text"
    } else {
        ""
    };
    out.push_str(&format!(
        "switched to session {:.8} ({} tokens{rebuilt})\n",
        sha_short(&plan.sha),
        live.transcript.len()
    ));
    out
}

fn strip(prefix: &str, live: Live<'_, '_>) -> String {
    let plan = match live.store.switch_plan(prefix) {
        Ok(plan) => plan,
        Err(error) => return format!("strip failed: {error}\n"),
    };
    let tokens = u32::try_from(live.model.vocab().encode_rendered_bytes(&plan.text).len())
        .unwrap_or(plan.tokens);
    match live.store.strip(prefix, tokens, unix_now()) {
        Ok(sha) => format!(
            "stripped session {:.8} ({tokens} tokens)\n",
            sha_short(&sha)
        ),
        Err(error) => format!("strip failed: {error}\n"),
    }
}

fn render_tokens(model: &Model, tokens: &[i32]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for &token in tokens {
        out.extend(model.token_text(token).map_err(|error| error.to_string())?);
    }
    Ok(out)
}

fn stage_payload(
    store: &SessionStore,
    session: &Session<'_>,
) -> Result<ds4_kv::PayloadTemp, String> {
    std::fs::create_dir_all(store.dir()).map_err(|error| error.to_string())?;
    let tmp = ds4_kv::PayloadTemp::create(store.dir()).map_err(|error| error.to_string())?;
    session
        .save_payload(tmp.path())
        .map_err(|error| error.to_string())?;
    Ok(tmp)
}

fn format_age(now: u64, when: u64) -> String {
    let age = if when != 0 && now > when {
        now - when
    } else {
        0
    };
    if age < 60 {
        format!("{age}s ago")
    } else if age < 3600 {
        format!("{}m ago", age / 60)
    } else if age < 86400 {
        format!("{}h ago", age / 3600)
    } else {
        format!("{}d ago", age / 86400)
    }
}

fn sha_short(sha: &str) -> &str {
    sha.get(..8).unwrap_or(sha)
}
