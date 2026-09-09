//! The public [`ChatTemplate`] type and its builder.

use std::sync::Arc;

use minijinja::{Environment, UndefinedBehavior, Value};
use serde::Serialize;
use serde_json::{Map, Value as Json};

use crate::clock::{Clock, SystemClock};
use crate::config::{self, ChatTemplateField, TokenizerConfig};
use crate::engine::{self, EngineConfig};
use crate::error::Error;
use crate::model::{Message, RenderInput};

/// A compiled Hugging Face chat template, ready to render.
///
/// Construction compiles the Jinja source and installs the `transformers`-compatible
/// globals/filters. [`render`](ChatTemplate::render) borrows `&self`, so a single instance is
/// cheap to reuse and shareable across threads.
pub struct ChatTemplate {
    env: Environment<'static>,
    /// Special tokens captured from a [`TokenizerConfig`] (empty for raw-string construction),
    /// injected into the render context as `{{ bos_token }}` etc. unless the input overrides them.
    special_tokens: Vec<(String, String)>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The compiled environment is large and not user-meaningful; keep this opaque.
        f.debug_struct("ChatTemplate")
            .field("special_tokens", &self.special_tokens)
            .finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Compile from a raw Jinja chat-template string, using the default
    /// `transformers`-compatible settings (see [`ChatTemplateBuilder`] to customize).
    ///
    /// An inherent `from_str` (not just the [`FromStr`](std::str::FromStr) impl) so callers
    /// can write `ChatTemplate::from_str(s)` without importing the trait.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(source: &str) -> Result<Self, Error> {
        ChatTemplate::builder(source).build()
    }

    /// Compile from a parsed `tokenizer_config.json`, resolving the default template and
    /// injecting the config's special tokens (`bos_token`, …) into the render context.
    ///
    /// For named-template selection (`tool_use`, `rag`) or custom options, use
    /// [`builder_from_config`](ChatTemplate::builder_from_config).
    pub fn from_tokenizer_config(config: &TokenizerConfig) -> Result<Self, Error> {
        ChatTemplate::builder_from_config(config)?.build()
    }

    /// Compile from a standalone template string (the contents of a `chat_template.jinja` file),
    /// injecting special tokens from a separately-loaded [`TokenizerConfig`].
    ///
    /// Newer `transformers` ship the chat template as its own `chat_template.jinja` file rather
    /// than inside `tokenizer_config.json` (Gemma 3+, SmolLM3, …). The template body comes from
    /// the file; the special tokens (`bos_token`, …) still come from the config. Matching the
    /// `transformers` precedence, a standalone file overrides any inline `chat_template` field, so
    /// this ignores `config.chat_template` entirely.
    pub fn from_template_and_config(
        template: &str,
        config: &TokenizerConfig,
    ) -> Result<Self, Error> {
        ChatTemplate::builder(template)
            .special_tokens_from(config)
            .build()
    }

    /// Start a builder to compile `source` with non-default options (clock, undefined policy…).
    pub fn builder(source: &str) -> ChatTemplateBuilder {
        ChatTemplateBuilder::new(BuilderSource::Raw(source.to_owned()), Vec::new())
    }

    /// Start a builder from a `tokenizer_config.json`, carrying its special tokens. Use
    /// [`template_name`](ChatTemplateBuilder::template_name) to pick a named sub-template.
    ///
    /// Fails with [`Error::Config`] if the config has no `chat_template` field at all.
    pub fn builder_from_config(config: &TokenizerConfig) -> Result<ChatTemplateBuilder, Error> {
        let field = config
            .chat_template
            .clone()
            .ok_or_else(|| Error::Config("tokenizer config has no chat_template".into()))?;
        Ok(ChatTemplateBuilder::new(
            BuilderSource::Field(field),
            config.special_tokens(),
        ))
    }

    /// Fetch a Hub repo (default branch) and compile its default chat template, injecting the
    /// config's special tokens. Requires the `hub` feature.
    ///
    /// Loads `tokenizer_config.json`, plus a standalone `chat_template.jinja` if the repo ships
    /// one (Gemma 3+, SmolLM3, …); the standalone file takes precedence over an inline
    /// `chat_template` field, matching `transformers`.
    ///
    /// Authentication uses `hf-hub`'s discovery (the `HF_TOKEN` env var or the cached
    /// `huggingface-cli login` token); gated repos need a token with access. For a pinned
    /// commit or branch, use [`from_hub_revision`](ChatTemplate::from_hub_revision).
    ///
    /// ```no_run
    /// use hf_chat_template::{ChatTemplate, Message};
    /// let tmpl = ChatTemplate::from_hub("Qwen/Qwen2.5-0.5B-Instruct")?;
    /// let prompt = tmpl.render_messages(&[Message::user("Hi")], true)?;
    /// # Ok::<(), hf_chat_template::Error>(())
    /// ```
    #[cfg(feature = "hub")]
    pub fn from_hub(repo_id: &str) -> Result<Self, Error> {
        Self::from_hub_inner(repo_id, None)
    }

    /// Like [`from_hub`](ChatTemplate::from_hub), but pins a specific `revision` (a branch,
    /// tag, or commit SHA). Requires the `hub` feature.
    #[cfg(feature = "hub")]
    pub fn from_hub_revision(repo_id: &str, revision: &str) -> Result<Self, Error> {
        Self::from_hub_inner(repo_id, Some(revision))
    }

    /// Shared `from_hub` body: fetch `tokenizer_config.json` plus a standalone
    /// `chat_template.jinja` if the repo ships one, then compile with the `transformers`
    /// precedence — a standalone file overrides any inline `chat_template` field.
    #[cfg(feature = "hub")]
    fn from_hub_inner(repo_id: &str, revision: Option<&str>) -> Result<Self, Error> {
        let (config, standalone) = crate::hub::fetch_template_material(repo_id, revision)?;
        match standalone {
            Some(jinja) => ChatTemplate::from_template_and_config(&jinja, &config),
            None => ChatTemplate::from_tokenizer_config(&config),
        }
    }

    /// Render the typed input model to the final prompt string. Special tokens from the
    /// source config are injected first; any matching key in [`RenderInput::extra`] wins.
    pub fn render(&self, input: &RenderInput) -> Result<String, Error> {
        let ctx = self.build_context(input)?;
        self.render_prepared(ctx)
    }

    /// Convenience: render with only `messages` and the generation-prompt flag.
    pub fn render_messages(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, Error> {
        let input = RenderInput {
            messages: messages.to_vec(),
            add_generation_prompt,
            ..Default::default()
        };
        self.render(&input)
    }

    /// Render with an arbitrary context — the escape hatch for callers who need to pass template
    /// variables the typed model doesn't cover.
    ///
    /// Takes anything [`Serialize`]: your own struct, a `serde_json::Value`, or a map. Key order is
    /// preserved as given, which matters when the template pipes a value through `tojson`.
    ///
    /// Unlike [`render`](ChatTemplate::render), this does **not** inject the config's special
    /// tokens; the caller's context is used as-is.
    ///
    /// ```
    /// use hf_chat_template::ChatTemplate;
    /// use serde_json::json;
    ///
    /// let tmpl = ChatTemplate::from_str("Hi {{ name }}")?;
    /// assert_eq!(tmpl.render_context(&json!({ "name": "Ada" }))?, "Hi Ada");
    /// # Ok::<(), hf_chat_template::Error>(())
    /// ```
    pub fn render_context<T: Serialize + ?Sized>(&self, ctx: &T) -> Result<String, Error> {
        self.render_prepared(Value::from_serialize(ctx))
    }

    /// Render a context already converted to the engine's value type. Private: the engine type
    /// stays out of the public API so its major version can never force ours.
    fn render_prepared(&self, ctx: Value) -> Result<String, Error> {
        let tmpl = self
            .env
            .get_template("chat")
            .expect("the 'chat' template is always present after construction");
        tmpl.render(ctx).map_err(Error::from_render)
    }

    /// Build the Jinja context: special tokens, then the serialized input on top (input wins).
    /// Routed through `serde_json` to a single `minijinja::Value`; `preserve_order` on both
    /// crates keeps map key order intact end-to-end (load-bearing for `| tojson`).
    fn build_context(&self, input: &RenderInput) -> Result<Value, Error> {
        let mut ctx: Map<String, Json> = Map::new();
        for (k, v) in &self.special_tokens {
            ctx.insert(k.clone(), Json::String(v.clone()));
        }
        let serialized = serde_json::to_value(input)
            .map_err(|e| Error::Config(format!("failed to serialize render input: {e}")))?;
        if let Json::Object(map) = serialized {
            for (k, v) in map {
                ctx.insert(k, v);
            }
        }
        Ok(Value::from_serialize(Json::Object(ctx)))
    }
}

/// Where a builder draws its template source from.
enum BuilderSource {
    /// A raw Jinja string.
    Raw(String),
    /// A `chat_template` field whose concrete source is resolved at `build()`.
    Field(ChatTemplateField),
}

/// Builder for [`ChatTemplate`], exposing the knobs that affect rendering semantics.
///
/// Defaults mirror the `transformers` reference environment:
/// `trim_blocks = true`, `lstrip_blocks = true`, `keep_trailing_newline = false`,
/// lenient undefined behavior, pycompat enabled, [`SystemClock`].
pub struct ChatTemplateBuilder {
    source: BuilderSource,
    special_tokens: Vec<(String, String)>,
    template_name: Option<String>,
    cfg: EngineConfig,
}

impl ChatTemplateBuilder {
    fn new(source: BuilderSource, special_tokens: Vec<(String, String)>) -> Self {
        ChatTemplateBuilder {
            source,
            special_tokens,
            template_name: None,
            cfg: EngineConfig {
                // VERIFY against transformers source; pinned here as the documented baseline.
                trim_blocks: true,
                lstrip_blocks: true,
                // Jinja2 defaults this to false and `transformers` never overrides it, so the
                // reference strips exactly one trailing newline from the template source. See
                // `tests/trailing_newline.rs` for the shapes this covers.
                keep_trailing_newline: false,
                undefined: UndefinedBehavior::Lenient,
                pycompat: true,
                clock: Arc::new(SystemClock),
            },
        }
    }

    /// Inject special tokens (`bos_token`, …) from a parsed `tokenizer_config.json` into the
    /// render context, replacing any already set. Use with [`ChatTemplate::builder`] when the
    /// template source is a standalone `chat_template.jinja` but the tokens still live in the
    /// config (see [`ChatTemplate::from_template_and_config`]).
    pub fn special_tokens_from(mut self, config: &TokenizerConfig) -> Self {
        self.special_tokens = config.special_tokens();
        self
    }

    /// Select a named sub-template (e.g. `"tool_use"`, `"rag"`) when the config carries a
    /// list of them. Ignored for a single-string source.
    pub fn template_name(mut self, name: impl Into<String>) -> Self {
        self.template_name = Some(name.into());
        self
    }

    /// Inject a deterministic [`Clock`] for `strftime_now` (essential for golden tests).
    pub fn clock(mut self, clock: impl Clock + 'static) -> Self {
        self.cfg.clock = Arc::new(clock);
        self
    }

    /// Enable/disable the pycompat Python-method shim. Default: enabled.
    pub fn pycompat(mut self, enabled: bool) -> Self {
        self.cfg.pycompat = enabled;
        self
    }

    /// Override how undefined variables are treated. Default: [`UndefinedBehavior::Lenient`].
    pub fn undefined_behavior(mut self, ub: UndefinedBehavior) -> Self {
        self.cfg.undefined = ub;
        self
    }

    /// Compile the template. Returns [`Error::Compile`] on a Jinja syntax error, or
    /// [`Error::Config`] if a requested/needed named template can't be resolved.
    pub fn build(self) -> Result<ChatTemplate, Error> {
        let source: String = match &self.source {
            BuilderSource::Raw(s) => s.clone(),
            BuilderSource::Field(field) => {
                config::resolve_template(field, self.template_name.as_deref())?.to_owned()
            }
        };
        let env = engine::build(source, &self.cfg).map_err(Error::from_compile)?;
        Ok(ChatTemplate {
            env,
            special_tokens: self.special_tokens,
        })
    }
}

impl std::str::FromStr for ChatTemplate {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        ChatTemplate::from_str(s)
    }
}
