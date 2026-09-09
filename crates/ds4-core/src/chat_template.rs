//! Local adapter for model-provided input grammar. Tokenization, media
//! processing and generated-output parsing remain separate host contracts.

use serde_json::Value;

use crate::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub enum RenderClock {
    System,
    Fixed(i64),
}

#[derive(Debug)]
pub struct Template {
    renderer: hf_chat_template::ChatTemplate,
}

impl Template {
    pub fn compile(source: &str, clock: RenderClock) -> Result<Self> {
        let mut builder = hf_chat_template::ChatTemplate::builder(source);
        if let RenderClock::Fixed(seconds) = clock {
            builder = builder.clock(hf_chat_template::FixedClock::from_unix_secs(seconds));
        }
        Ok(Self {
            renderer: builder.build().map_err(template_error)?,
        })
    }

    /// Render the complete supplied context without adding or removing
    /// special tokens. In particular, tokenization must not add another BOS.
    pub fn render(&self, context: &Value) -> Result<String> {
        self.renderer
            .render_context(context)
            .map_err(template_error)
    }
}

fn template_error(error: hf_chat_template::Error) -> Error {
    Error {
        code: 1,
        message: format!("chat template: {error}"),
    }
}
