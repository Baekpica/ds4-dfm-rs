//! REPL history stays structured so the model template can rewrite old turns.

use ds4_core::ChatThinkMode;
use serde_json::{json, Value};

pub(crate) struct Conversation {
    messages: Vec<Value>,
}

impl Conversation {
    pub(crate) fn new(system: &str) -> Self {
        Self {
            messages: if system.is_empty() {
                Vec::new()
            } else {
                vec![json!({"role":"system", "content":system})]
            },
        }
    }

    pub(crate) fn prompt(&self, user: &str) -> Vec<Value> {
        let mut messages = self.messages.clone();
        messages.push(json!({"role":"user", "content":user}));
        messages
    }

    pub(crate) fn accept(
        &mut self,
        mut messages: Vec<Value>,
        model_id: i32,
        raw: &[u8],
        think: ChatThinkMode,
    ) -> Result<(), String> {
        // Reuse the HTTP host's output protocol; input Jinja cannot parse output.
        let syntax = ds4_server::syntax_for_model_id(model_id);
        let (generated, _) = ds4_server::parse_generated_for_response(
            syntax,
            raw,
            false,
            false,
            think != ChatThinkMode::None,
            ds4_server::generate::chat_format_for_syntax(syntax),
            &[],
            "stop",
        );
        if !generated.ok {
            return Err("malformed assistant output; previous conversation retained".into());
        }
        let mut assistant =
            json!({"role":"assistant", "content":String::from_utf8_lossy(&generated.content)});
        if !generated.reasoning.is_empty() {
            assistant["reasoning_content"] = String::from_utf8_lossy(&generated.reasoning)
                .into_owned()
                .into();
        }
        messages.push(assistant);
        self.messages = messages;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_core::chat_template::{ChatOptions, RenderClock, Template};

    #[test]
    fn qwen_followup_uses_history() {
        let mut chat = Conversation::new("Be concise.");
        chat.accept(
            chat.prompt("2+2?"),
            6,
            b"Check.\n</think>\n\n4",
            ChatThinkMode::Low,
        )
        .unwrap();
        let followup = chat.prompt("Add one.");
        assert_eq!(
            followup[2],
            json!({"role":"assistant", "content":"\n\n4", "reasoning_content":"Check.\n"})
        );
        let template = Template::compile(
            include_str!("../../../tests/fixtures/chat-template/models/qwen/chat_template.jinja"),
            RenderClock::Fixed(0),
        )
        .unwrap();
        let rendered = template
            .render_chat(&followup, &[], ChatOptions::new(6, ChatThinkMode::Low))
            .unwrap();
        assert!(rendered.contains("assistant\n<think>\nCheck.\n</think>\n\n4<|im_end|>"));
    }

    #[test]
    fn inkling_output_roundtrip() {
        let mut chat = Conversation::new("");
        chat.accept(chat.prompt("2+2?"), 9, b"<|content_thinking|>Check.<|end_message|><|message_model|><|content_text|>4<|end_message|>", ChatThinkMode::Low).unwrap();
        assert_eq!(
            chat.prompt("Add one.")[1],
            json!({"role":"assistant", "content":"4", "reasoning_content":"Check."})
        );
    }

    #[test]
    fn abandoned_prompt_is_atomic() {
        let chat = Conversation::new("Be concise.");
        let _failed = chat.prompt("Failed request.");
        assert_eq!(
            chat.prompt("Retry."),
            json!([
                {"role":"system", "content":"Be concise."},
                {"role":"user", "content":"Retry."}
            ])
            .as_array()
            .unwrap()
            .clone()
        );
    }
}
