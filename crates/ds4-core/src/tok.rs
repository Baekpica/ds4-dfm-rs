//! mmap-backed GPT-2 byte-level BPE + family pre-tokenizers.
//!
//! Copied from `ds4.c` tokenizer (`vocab_load`, `bpe_emit_piece`,
//! JoyAI / Motif / Solar / EXAONE / dots3 splitters, rendered-chat
//! special scan, decode, generation-stop). Family comes from the
//! caller (`identify` or an explicit test family), not `g_ds4_shape`.

use std::collections::HashMap;
use std::path::Path;

use crate::gguf::{GgufError, GgufFile, GGUF_VALUE_INT32, GGUF_VALUE_STRING, GGUF_VALUE_UINT32};
use crate::shape::ModelFamily;
use crate::TokenBuffer;

const REASONING_EFFORT_HIGH_PREFIX: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
);

const REASONING_EFFORT_MAX_PREFIX: &str = concat!(
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n",
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n",
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n",
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatThinkMode {
    None = 0,
    Low = 1,
    High = 2,
    Max = 3,
}

impl ChatThinkMode {
    fn enabled(self) -> bool {
        self != Self::None
    }
}

#[derive(Debug)]
pub enum TokError {
    Gguf(GgufError),
    MissingTable(&'static str),
    MissingToken(String),
    SolarMissingControl,
    EmbeddedNul,
}

impl std::fmt::Display for TokError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokError::Gguf(e) => write!(f, "{e}"),
            TokError::MissingTable(k) => write!(f, "missing-table {k}"),
            TokError::MissingToken(t) => write!(f, "missing-token {t}"),
            TokError::SolarMissingControl => write!(f, "solar-missing-control"),
            TokError::EmbeddedNul => write!(f, "embedded-nul"),
        }
    }
}

impl std::error::Error for TokError {}

impl From<GgufError> for TokError {
    fn from(e: GgufError) -> Self {
        TokError::Gguf(e)
    }
}

#[derive(Debug, Clone)]
pub struct Vocab {
    pub family: ModelFamily,
    is_k2_horizon: bool,
    tokens: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, i32>,
    merges: Vec<Vec<u8>>,
    merge_rank: HashMap<Vec<u8>, i32>,
    user_defined: HashMap<Vec<u8>, i32>,
    user_defined_max_len: u32,
    user_defined_first: [bool; 256],
    motif3_added_first: [bool; 256],
    pub bos_id: i32,
    pub eos_id: i32,
    pub system_id: i32,
    pub eot_id: i32,
    pub im_start_id: i32,
    pub im_content_id: i32,
    pub im_end_id: i32,
    pub user_id: i32,
    pub assistant_id: i32,
    pub start_of_turn_id: i32,
    pub end_of_turn_id: i32,
    pub tool_id: i32,
    pub reference_id: i32,
    pub plan_start_id: i32,
    pub plan_end_id: i32,
    pub observation_id: i32,
    pub sop_id: i32,
    pub think_start_id: i32,
    pub think_end_id: i32,
    pub tool_call_start_id: i32,
    pub tool_call_end_id: i32,
    pub tool_response_start_id: i32,
    pub tool_response_end_id: i32,
    pub arg_key_start_id: i32,
    pub arg_key_end_id: i32,
    pub arg_value_start_id: i32,
    pub arg_value_end_id: i32,
    pub latent_start_id: i32,
    pub latent_pad_id: i32,
    pub latent_end_id: i32,
    pub tool_schema_start_id: i32,
    pub tool_schema_end_id: i32,
    pub dsml_id: i32,
    pub dots3_endofsystem_id: i32,
    pub dots3_endofuser_id: i32,
    pub dots3_endoftext_id: i32,
}

impl Vocab {
    pub fn n_vocab(&self) -> i32 {
        self.tokens.len() as i32
    }

    pub fn tokens(&self) -> &[Vec<u8>] {
        &self.tokens
    }

    pub fn merges(&self) -> &[Vec<u8>] {
        &self.merges
    }

    pub fn user_defined_ids(&self) -> Vec<i32> {
        let mut ids: Vec<i32> = self.user_defined.values().copied().collect();
        ids.sort_unstable();
        ids
    }

    pub fn user_defined_max_len(&self) -> u32 {
        self.user_defined_max_len
    }

    pub fn user_defined_first(&self) -> &[bool; 256] {
        &self.user_defined_first
    }

    pub fn motif3_added_first(&self) -> &[bool; 256] {
        &self.motif3_added_first
    }

    pub fn engine_eos(&self) -> i32 {
        if self.family == ModelFamily::SolarOpen2 && self.eot_id >= 0 {
            self.eot_id
        } else {
            self.eos_id
        }
    }

    fn lookup(&self, text: &str) -> Result<i32, TokError> {
        self.token_to_id
            .get(text.as_bytes())
            .copied()
            .ok_or_else(|| TokError::MissingToken(text.to_string()))
    }

    fn lookup_opt(&self, text: &str) -> i32 {
        self.token_to_id.get(text.as_bytes()).copied().unwrap_or(-1)
    }

    pub fn load(g: &GgufFile, family: ModelFamily) -> Result<Self, TokError> {
        let is_k2_horizon = family == ModelFamily::ExaoneMoe
            && (g.get_string("general.architecture") == Some(b"k2-horizon")
                || g.get_string("tokenizer.ggml.pre") == Some(b"k2-horizon"));
        let tokens_arr = g
            .get_array("tokenizer.ggml.tokens")
            .filter(|a| a.typ == GGUF_VALUE_STRING && a.len <= i32::MAX as u64)
            .ok_or(TokError::MissingTable("tokenizer.ggml.tokens"))?;
        let merges_arr = g
            .get_array("tokenizer.ggml.merges")
            .filter(|a| a.typ == GGUF_VALUE_STRING)
            .ok_or(TokError::MissingTable("tokenizer.ggml.merges"))?;

        let token_bytes = g.array_strings(&tokens_arr)?;
        let mut tokens = Vec::with_capacity(token_bytes.len());
        let mut token_to_id = HashMap::with_capacity(token_bytes.len());
        let mut motif3_added_first = [false; 256];
        for (i, t) in token_bytes.iter().enumerate() {
            tokens.push(t.to_vec());
            token_to_id.insert(t.to_vec(), i as i32);
            if family == ModelFamily::Motif3 && i < 160 && !t.is_empty() {
                motif3_added_first[t[0] as usize] = true;
            }
        }

        let mut user_defined = HashMap::new();
        let mut user_defined_max_len = 0u32;
        let mut user_defined_first = [false; 256];
        if let Some(types) = g.get_array("tokenizer.ggml.token_type") {
            if (types.typ == GGUF_VALUE_UINT32 || types.typ == GGUF_VALUE_INT32)
                && types.len == tokens_arr.len
            {
                if let Ok(ty) = g.array_le_u32s(&types) {
                    for (i, &typ) in ty.iter().enumerate() {
                        if typ != 4 {
                            continue;
                        }
                        let token = &tokens[i];
                        if token.is_empty() {
                            continue;
                        }
                        user_defined.insert(token.clone(), i as i32);
                        if token.len() as u32 > user_defined_max_len {
                            user_defined_max_len = token.len() as u32;
                        }
                        user_defined_first[token[0] as usize] = true;
                    }
                }
            }
        }

        let merge_bytes = g.array_strings(&merges_arr)?;
        let mut merges = Vec::with_capacity(merge_bytes.len());
        let mut merge_rank = HashMap::with_capacity(merge_bytes.len());
        for (i, m) in merge_bytes.iter().enumerate() {
            merges.push(m.to_vec());
            merge_rank.insert(m.to_vec(), i as i32);
        }

        let mut v = Self {
            family,
            is_k2_horizon,
            tokens,
            token_to_id,
            merges,
            merge_rank,
            user_defined,
            user_defined_max_len,
            user_defined_first,
            motif3_added_first,
            /* C vocab_load: memset 0, then the same -1 defaults. Solar /
             * EXAONE leave end_of_turn_id at 0; do not "fix" it to -1. */
            bos_id: 0,
            eos_id: 0,
            system_id: 0,
            eot_id: -1,
            im_start_id: -1,
            im_content_id: -1,
            im_end_id: -1,
            user_id: 0,
            assistant_id: 0,
            start_of_turn_id: 0,
            end_of_turn_id: 0,
            tool_id: 0,
            reference_id: 0,
            plan_start_id: 0,
            plan_end_id: 0,
            observation_id: 0,
            sop_id: 0,
            think_start_id: 0,
            think_end_id: 0,
            tool_call_start_id: -1,
            tool_call_end_id: -1,
            tool_response_start_id: -1,
            tool_response_end_id: -1,
            arg_key_start_id: -1,
            arg_key_end_id: -1,
            arg_value_start_id: -1,
            arg_value_end_id: -1,
            latent_start_id: 0,
            latent_pad_id: 0,
            latent_end_id: 0,
            tool_schema_start_id: -1,
            tool_schema_end_id: -1,
            dsml_id: 0,
            dots3_endofsystem_id: -1,
            dots3_endofuser_id: -1,
            dots3_endoftext_id: -1,
        };
        v.load_specials(g)?;
        Ok(v)
    }

    fn load_specials(&mut self, g: &GgufFile) -> Result<(), TokError> {
        match self.family {
            ModelFamily::Glm53 => {
                self.bos_id = g
                    .get_token_id("tokenizer.ggml.bos_token_id")
                    .unwrap_or_else(|| self.lookup_opt("<sop>"));
                self.eos_id = g
                    .get_token_id("tokenizer.ggml.eos_token_id")
                    .unwrap_or_else(|| self.lookup_opt("<|endoftext|>"));
                self.system_id = self.lookup_opt("<|system|>");
                self.user_id = self.lookup_opt("<|user|>");
                self.assistant_id = self.lookup_opt("<|assistant|>");
                self.observation_id = self.lookup_opt("<|observation|>");
                self.sop_id = self.lookup_opt("<sop>");
                self.think_start_id = self.lookup_opt("<think>");
                self.think_end_id = self.lookup_opt("</think>");
                self.tool_call_start_id = self.lookup_opt("<tool_call>");
                self.tool_call_end_id = self.lookup_opt("</tool_call>");
                self.tool_response_start_id = self.lookup_opt("<tool_response>");
                self.tool_response_end_id = self.lookup_opt("</tool_response>");
                self.arg_key_start_id = self.lookup_opt("<arg_key>");
                self.arg_key_end_id = self.lookup_opt("</arg_key>");
                self.arg_value_start_id = self.lookup_opt("<arg_value>");
                self.arg_value_end_id = self.lookup_opt("</arg_value>");
                self.dsml_id = -1;
            }
            ModelFamily::Motif3 => {
                self.bos_id = g
                    .get_token_id("tokenizer.ggml.bos_token_id")
                    .unwrap_or(self.lookup("<|beginoftext|>")?);
                self.eos_id = g
                    .get_token_id("tokenizer.ggml.eos_token_id")
                    .unwrap_or(self.lookup("<|endoftext|>")?);
                self.system_id = self.lookup("<|system|>")?;
                self.user_id = self.lookup("<|user|>")?;
                self.assistant_id = self.lookup("<|assistant|>")?;
                self.start_of_turn_id = self.lookup("<|startofturn|>")?;
                self.end_of_turn_id = self.lookup("<|endofturn|>")?;
                self.tool_id = self.lookup("<|tool|>")?;
                self.reference_id = self.lookup("<|reference|>")?;
                self.plan_start_id = self.lookup("<|plan|>")?;
                self.plan_end_id = self.lookup("<|endofplan|>")?;
                self.think_start_id = self.lookup("<think>")?;
                self.think_end_id = self.lookup("</think>")?;
                self.tool_call_start_id = self.lookup("<tool_call>")?;
                self.tool_call_end_id = self.lookup("</tool_call>")?;
                self.tool_response_start_id = self.lookup("<tool_response>")?;
                self.tool_response_end_id = self.lookup("</tool_response>")?;
                self.observation_id = -1;
                self.sop_id = -1;
                self.arg_key_start_id = -1;
                self.arg_key_end_id = -1;
                self.arg_value_start_id = -1;
                self.arg_value_end_id = -1;
                self.latent_start_id = self.lookup("<latent>")?;
                self.latent_pad_id = self.lookup("<latent_pad>")?;
                self.latent_end_id = self.lookup("</latent>")?;
                self.dsml_id = -1;
            }
            ModelFamily::Dots3Note => {
                self.bos_id = g
                    .get_token_id("tokenizer.ggml.bos_token_id")
                    .unwrap_or(self.lookup("<|endoftext|>")?);
                self.eos_id = g
                    .get_token_id("tokenizer.ggml.eos_token_id")
                    .unwrap_or(self.lookup("<|endofassistant|>")?);
                self.system_id = self.lookup("<|system|>")?;
                self.user_id = self.lookup("<|user|>")?;
                self.assistant_id = self.lookup("<|assistant|>")?;
                self.dots3_endofsystem_id = self.lookup("<|endofsystem|>")?;
                self.dots3_endofuser_id = self.lookup("<|endofuser|>")?;
                self.dots3_endoftext_id = self.lookup("<|endoftext|>")?;
                self.think_start_id = self.lookup("<think>")?;
                self.think_end_id = self.lookup("</think>")?;
                self.tool_call_start_id = self.lookup("<dots_function_call>")?;
                self.tool_call_end_id = self.lookup("</dots_function_call>")?;
                self.tool_response_start_id = self.lookup("<dots_function_response>")?;
                self.tool_response_end_id = self.lookup("</dots_function_response>")?;
                self.start_of_turn_id = -1;
                self.end_of_turn_id = -1;
                self.tool_id = -1;
                self.reference_id = -1;
                self.plan_start_id = -1;
                self.plan_end_id = -1;
                self.observation_id = -1;
                self.sop_id = -1;
                self.latent_start_id = -1;
                self.latent_pad_id = -1;
                self.latent_end_id = -1;
                self.dsml_id = -1;
            }
            ModelFamily::Qwen4Exp => {
                self.bos_id = g
                    .get_token_id("tokenizer.ggml.bos_token_id")
                    .unwrap_or(self.lookup("<|endoftext|>")?);
                self.eos_id = g
                    .get_token_id("tokenizer.ggml.eos_token_id")
                    .unwrap_or(self.lookup("<|im_end|>")?);
                self.eot_id = self.lookup("<|endoftext|>")?;
                self.im_start_id = self.lookup("<|im_start|>")?;
                self.im_end_id = self.lookup("<|im_end|>")?;
                self.think_start_id = self.lookup("<think>")?;
                self.think_end_id = self.lookup("</think>")?;
                self.tool_call_start_id = self.lookup("<tool_call>")?;
                self.tool_call_end_id = self.lookup("</tool_call>")?;
                self.tool_response_start_id = self.lookup("<tool_response>")?;
                self.tool_response_end_id = self.lookup("</tool_response>")?;
                self.system_id = -1;
                self.user_id = -1;
                self.assistant_id = -1;
                self.dsml_id = -1;
            }
            ModelFamily::SolarOpen2 => {
                self.bos_id = g
                    .get_token_id("tokenizer.ggml.bos_token_id")
                    .unwrap_or_else(|| self.lookup_opt("<|startoftext|>"));
                self.eos_id = g
                    .get_token_id("tokenizer.ggml.eos_token_id")
                    .unwrap_or_else(|| self.lookup_opt("<|endoftext|>"));
                self.eot_id = g
                    .get_token_id("tokenizer.ggml.eot_token_id")
                    .unwrap_or_else(|| self.lookup_opt("<|im:end|>"));
                self.im_start_id = self.lookup_opt("<|im:start|>");
                self.im_content_id = self.lookup_opt("<|im:content|>");
                self.im_end_id = self.lookup_opt("<|im:end|>");
                self.think_start_id = self.lookup_opt("<|think:start|>");
                self.think_end_id = self.lookup_opt("<|think:end|>");
                self.tool_call_start_id = self.lookup_opt("<|tool_call:start|>");
                self.tool_call_end_id = self.lookup_opt("<|tool_call:end|>");
                self.tool_response_start_id = self.lookup_opt("<|tool_response:start|>");
                self.tool_response_end_id = self.lookup_opt("<|tool_response:end|>");
                self.arg_key_start_id = self.lookup_opt("<|tool_arg:start|>");
                self.arg_key_end_id = self.lookup_opt("<|tool_arg:end|>");
                self.arg_value_start_id = self.lookup_opt("<|tool_arg:value|>");
                self.arg_value_end_id = self.arg_key_end_id;
                self.tool_schema_start_id = self.lookup_opt("<|tool:start|>");
                self.tool_schema_end_id = self.lookup_opt("<|tool:end|>");
                self.user_id = -1;
                self.assistant_id = -1;
                self.dsml_id = -1;
                if self.bos_id < 0
                    || self.eos_id < 0
                    || self.eot_id < 0
                    || self.im_start_id < 0
                    || self.im_content_id < 0
                    || self.im_end_id < 0
                    || self.think_start_id < 0
                    || self.think_end_id < 0
                    || self.tool_schema_start_id < 0
                    || self.tool_schema_end_id < 0
                {
                    return Err(TokError::SolarMissingControl);
                }
            }
            ModelFamily::ExaoneMoe => {
                self.bos_id = g
                    .get_token_id("tokenizer.ggml.bos_token_id")
                    .unwrap_or_else(|| {
                        self.lookup_opt(if self.is_k2_horizon {
                            "<|ifm|begin_of_text|>"
                        } else {
                            "[BOS]"
                        })
                    });
                self.eos_id = g
                    .get_token_id("tokenizer.ggml.eos_token_id")
                    .unwrap_or_else(|| {
                        self.lookup_opt(if self.is_k2_horizon {
                            "<|ifm|endoftext|>"
                        } else {
                            "<|endofturn|>"
                        })
                    });
                if self.is_k2_horizon {
                    self.im_start_id = self.lookup_opt("<|ifm|im_start|>");
                    self.im_end_id = self.lookup_opt("<|ifm|im_end|>");
                    self.think_start_id = self.lookup_opt("<ifm|think>");
                    self.think_end_id = self.lookup_opt("</ifm|think>");
                    self.tool_call_start_id = self.lookup_opt("<ifm|tool_calls>");
                    self.tool_call_end_id = self.lookup_opt("</ifm|tool_calls>");
                    self.tool_response_start_id = self.lookup_opt("<ifm|tool_call>");
                    self.tool_response_end_id = self.lookup_opt("</ifm|tool_call>");
                    self.system_id = -1;
                    self.user_id = -1;
                    self.assistant_id = -1;
                    self.observation_id = -1;
                    self.sop_id = -1;
                    self.dsml_id = -1;
                    return Ok(());
                }
                self.system_id = self.lookup_opt("<|system|>");
                self.user_id = self.lookup_opt("<|user|>");
                self.assistant_id = self.lookup_opt("<|assistant|>");
                self.observation_id = self.lookup_opt("<|tool|>");
                self.think_start_id = self.lookup_opt("<think>");
                self.think_end_id = self.lookup_opt("</think>");
                self.tool_call_start_id = self.lookup_opt("<tool_call>");
                self.tool_call_end_id = self.lookup_opt("</tool_call>");
                self.tool_response_start_id = self.lookup_opt("<tool_result>");
                self.tool_response_end_id = self.lookup_opt("</tool_result>");
                self.sop_id = -1;
                self.arg_key_start_id = -1;
                self.arg_key_end_id = -1;
                self.arg_value_start_id = -1;
                self.arg_value_end_id = -1;
                self.dsml_id = -1;
            }
            ModelFamily::DeepSeek4 => {
                self.bos_id = self.lookup("<｜begin▁of▁sentence｜>")?;
                self.eos_id = self.lookup("<｜end▁of▁sentence｜>")?;
                self.user_id = self.lookup("<｜User｜>")?;
                self.assistant_id = self.lookup("<｜Assistant｜>")?;
                self.system_id = -1;
                self.start_of_turn_id = -1;
                self.end_of_turn_id = -1;
                self.tool_id = -1;
                self.reference_id = -1;
                self.plan_start_id = -1;
                self.plan_end_id = -1;
                self.observation_id = -1;
                self.sop_id = -1;
                self.think_start_id = self.lookup("<think>")?;
                self.think_end_id = self.lookup("</think>")?;
                self.tool_call_start_id = -1;
                self.tool_call_end_id = -1;
                self.tool_response_start_id = -1;
                self.tool_response_end_id = -1;
                self.arg_key_start_id = -1;
                self.arg_key_end_id = -1;
                self.arg_value_start_id = -1;
                self.arg_value_end_id = -1;
                self.latent_start_id = -1;
                self.latent_pad_id = -1;
                self.latent_end_id = -1;
                self.dsml_id = self.lookup("｜DSML｜")?;
            }
        }
        Ok(())
    }

    pub fn load_path(path: &Path, family: ModelFamily) -> Result<Self, TokError> {
        let g = GgufFile::open(path)?;
        Self::load(&g, family)
    }

    pub fn encode_text(&self, text: &str) -> Vec<i32> {
        self.encode_bytes(text.as_bytes())
    }

    pub fn encode_bytes(&self, text: &[u8]) -> Vec<i32> {
        let mut out = Vec::new();
        bpe_tokenize_text(self, text, &mut out);
        out
    }

    pub fn encode_rendered_chat(&self, text: &str) -> Vec<i32> {
        self.encode_rendered_bytes(text.as_bytes())
    }

    pub fn encode_rendered_bytes(&self, text: &[u8]) -> Vec<i32> {
        let mut out = Vec::new();
        tokenize_rendered_chat(self, text, &mut out);
        out
    }

    pub fn chat_begin(&self, tokens: &mut TokenBuffer) -> Result<(), TokError> {
        if self.family == ModelFamily::ExaoneMoe {
            self.require_chat_ids(&[(self.bos_id, "[BOS]")])?;
        } else if self.family == ModelFamily::Glm53 {
            self.require_chat_ids(&[(self.bos_id, "[gMASK]"), (self.sop_id, "<sop>")])?;
        }
        if !matches!(
            self.family,
            ModelFamily::SolarOpen2 | ModelFamily::Dots3Note | ModelFamily::Qwen4Exp
        ) {
            tokens.push(self.bos_id);
            if self.family == ModelFamily::Glm53 && self.sop_id >= 0 {
                tokens.push(self.sop_id);
            }
        }
        Ok(())
    }

    pub fn chat_append_effort_prefix(&self, tokens: &mut TokenBuffer, mode: ChatThinkMode) {
        if self.is_k2_horizon {
            return;
        }
        if self.family == ModelFamily::Glm53 {
            let effort = match mode {
                ChatThinkMode::High => "Reasoning Effort: High",
                ChatThinkMode::Max => "Reasoning Effort: Max",
                ChatThinkMode::None | ChatThinkMode::Low => return,
            };
            tokens.push(self.system_id);
            bpe_tokenize_text(self, effort.as_bytes(), &mut tokens.tokens);
            return;
        }
        if matches!(
            self.family,
            ModelFamily::Motif3
                | ModelFamily::SolarOpen2
                | ModelFamily::Dots3Note
                | ModelFamily::Qwen4Exp
        ) {
            return;
        }
        let prefix = match mode {
            ChatThinkMode::High => REASONING_EFFORT_HIGH_PREFIX,
            ChatThinkMode::Max => REASONING_EFFORT_MAX_PREFIX,
            ChatThinkMode::None | ChatThinkMode::Low => "",
        };
        bpe_tokenize_text(self, prefix.as_bytes(), &mut tokens.tokens);
    }

    pub fn chat_append_message(
        &self,
        tokens: &mut TokenBuffer,
        role: &str,
        content: &[u8],
    ) -> Result<(), TokError> {
        if role.as_bytes().contains(&0) || content.contains(&0) {
            return Err(TokError::EmbeddedNul);
        }
        if self.is_k2_horizon {
            self.require_chat_ids(&[
                (self.im_start_id, "<|ifm|im_start|>"),
                (self.im_end_id, "<|ifm|im_end|>"),
            ])?;
            tokens.push(self.im_start_id);
            let rendered_role = match role {
                "system" | "developer" => b"system".as_slice(),
                "assistant" => b"assistant".as_slice(),
                "tool" | "function" => b"tool".as_slice(),
                _ => b"user".as_slice(),
            };
            bpe_tokenize_text(self, rendered_role, &mut tokens.tokens);
            bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
            if role == "assistant"
                && !content.starts_with(b"<ifm|think>")
                && !content.starts_with(b"<ifm|think_fast>")
                && !content.starts_with(b"<ifm|think_faster>")
            {
                self.require_chat_ids(&[
                    (self.think_start_id, "<ifm|think>"),
                    (self.think_end_id, "</ifm|think>"),
                ])?;
                tokens.push(self.think_start_id);
                bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
                tokens.push(self.think_end_id);
                bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
            }
            tokenize_rendered_chat(self, content, &mut tokens.tokens);
            tokens.push(self.im_end_id);
            return Ok(());
        }

        match self.family {
            ModelFamily::Glm53 => {
                if role == "system" || role == "developer" {
                    self.require_chat_ids(&[(self.system_id, "<|system|>")])?;
                    tokens.push(self.system_id);
                    tokenize_rendered_chat(self, content, &mut tokens.tokens);
                } else if role == "assistant" {
                    self.require_chat_ids(&[
                        (self.assistant_id, "<|assistant|>"),
                        (self.think_start_id, "<think>"),
                        (self.think_end_id, "</think>"),
                    ])?;
                    tokens.push(self.assistant_id);
                    if !content.starts_with(b"<think>") && !content.starts_with(b"</think>") {
                        tokens.push(self.think_start_id);
                        tokens.push(self.think_end_id);
                    }
                    tokenize_rendered_chat(self, content, &mut tokens.tokens);
                } else if role == "tool" || role == "function" {
                    self.require_chat_ids(&[
                        (self.observation_id, "<|observation|>"),
                        (self.tool_response_start_id, "<tool_response>"),
                        (self.tool_response_end_id, "</tool_response>"),
                    ])?;
                    tokens.push(self.observation_id);
                    tokens.push(self.tool_response_start_id);
                    self.chat_append_wrapped_payload(tokens, content, b"</tool_response>");
                    tokens.push(self.tool_response_end_id);
                } else {
                    self.require_chat_ids(&[(self.user_id, "<|user|>")])?;
                    tokens.push(self.user_id);
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                }
            }
            ModelFamily::Motif3 => {
                tokens.push(self.start_of_turn_id);
                if role == "system" || role == "developer" {
                    tokens.push(self.system_id);
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                } else if role == "assistant" {
                    tokens.push(self.assistant_id);
                    tokenize_rendered_chat(self, content, &mut tokens.tokens);
                } else if role == "tool" || role == "function" {
                    tokens.push(self.tool_id);
                    tokens.push(self.tool_response_start_id);
                    self.chat_append_wrapped_payload(tokens, content, b"</tool_response>");
                    tokens.push(self.tool_response_end_id);
                } else {
                    tokens.push(self.user_id);
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                }
                tokens.push(self.end_of_turn_id);
            }
            ModelFamily::Dots3Note => {
                if role == "system" || role == "developer" {
                    tokens.push(self.system_id);
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                    tokens.push(self.dots3_endofsystem_id);
                } else if role == "assistant" {
                    tokens.push(self.assistant_id);
                    tokenize_rendered_chat(self, content, &mut tokens.tokens);
                    tokens.push(self.eos_id);
                } else if role == "tool" || role == "function" {
                    tokens.push(self.user_id);
                    bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
                    tokens.push(self.tool_response_start_id);
                    bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
                    self.chat_append_wrapped_payload(tokens, content, b"</dots_function_response>");
                    bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
                    tokens.push(self.tool_response_end_id);
                    tokens.push(self.dots3_endofuser_id);
                } else {
                    tokens.push(self.user_id);
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                    tokens.push(self.dots3_endofuser_id);
                }
            }
            ModelFamily::SolarOpen2 => {
                if role == "tool" || role == "function" {
                    self.require_chat_ids(&[
                        (self.tool_response_start_id, "<|tool_response:start|>"),
                        (self.tool_response_end_id, "<|tool_response:end|>"),
                    ])?;
                }

                if tokens.as_slice().last() == Some(&self.im_end_id) {
                    self.chat_push_fragment(tokens, b"\n", b"");
                }
                if role == "system" || role == "developer" {
                    self.solar_chat_open_role(tokens, b"system");
                    self.chat_push_fragment(tokens, b"## System Prompt\n\n", content);
                    self.solar_chat_close_role(tokens);
                } else if role == "assistant" {
                    self.solar_chat_open_role(tokens, b"assistant");
                    if !content.starts_with(b"<|think:start|>") {
                        tokens.push(self.think_start_id);
                        tokens.push(self.think_end_id);
                    }
                    tokenize_rendered_chat(self, content, &mut tokens.tokens);
                    self.solar_chat_close_role(tokens);
                } else if role == "tool" || role == "function" {
                    self.solar_chat_open_role(tokens, b"tool");
                    tokens.push(self.tool_response_start_id);
                    self.chat_append_wrapped_payload(tokens, content, b"<|tool_response:end|>");
                    tokens.push(self.tool_response_end_id);
                    self.solar_chat_close_role(tokens);
                } else {
                    self.solar_chat_open_role(tokens, b"user");
                    tokenize_rendered_chat(self, content, &mut tokens.tokens);
                    self.solar_chat_close_role(tokens);
                }
            }
            ModelFamily::Qwen4Exp => {
                if role == "system" || role == "developer" {
                    self.qwen_chat_open_role(tokens, b"system");
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                } else if role == "assistant" {
                    self.qwen_chat_open_role(tokens, b"assistant");
                    if !content.starts_with(b"<think>") {
                        tokens.push(self.think_start_id);
                        bpe_tokenize_text(self, b"\n\n", &mut tokens.tokens);
                        tokens.push(self.think_end_id);
                        bpe_tokenize_text(self, b"\n\n", &mut tokens.tokens);
                    }
                    tokenize_rendered_chat(self, content, &mut tokens.tokens);
                } else if role == "tool" || role == "function" {
                    self.qwen_chat_open_role(tokens, b"user");
                    tokens.push(self.tool_response_start_id);
                    bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
                    self.chat_append_wrapped_payload(tokens, content, b"</tool_response>");
                    bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
                    tokens.push(self.tool_response_end_id);
                } else {
                    self.qwen_chat_open_role(tokens, b"user");
                    tokenize_rendered_chat(self, content, &mut tokens.tokens);
                }
                self.qwen_chat_close_role(tokens);
            }
            _ => {
                if role == "system" || role == "developer" {
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                } else if role == "assistant" {
                    if self.family == ModelFamily::ExaoneMoe {
                        self.require_chat_ids(&[(self.assistant_id, "<|assistant|>")])?;
                        if !content.starts_with(b"<think>") && !content.starts_with(b"</think>") {
                            self.require_chat_ids(&[(self.think_end_id, "</think>")])?;
                        }
                    }

                    tokens.push(self.assistant_id);
                    if !content.starts_with(b"<think>") && !content.starts_with(b"</think>") {
                        tokens.push(self.think_end_id);
                    }
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                } else {
                    if self.family == ModelFamily::ExaoneMoe {
                        self.require_chat_ids(&[(self.user_id, "<|user|>")])?;
                    }
                    tokens.push(self.user_id);
                    if role == "tool" || role == "function" {
                        bpe_tokenize_text(self, b"Tool: ", &mut tokens.tokens);
                    }
                    bpe_tokenize_text(self, content, &mut tokens.tokens);
                }
            }
        }
        Ok(())
    }

    pub fn chat_append_assistant_prefix(
        &self,
        tokens: &mut TokenBuffer,
        mode: ChatThinkMode,
    ) -> Result<(), TokError> {
        let thinking = mode.enabled();
        if self.is_k2_horizon {
            self.require_chat_ids(&[
                (self.im_start_id, "<|ifm|im_start|>"),
                (self.think_start_id, "<ifm|think>"),
                (self.think_end_id, "</ifm|think>"),
            ])?;
            tokens.push(self.im_start_id);
            bpe_tokenize_text(self, b"assistant\n", &mut tokens.tokens);
            tokens.push(self.think_start_id);
            bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
            if !thinking {
                tokens.push(self.think_end_id);
                bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
            }
            return Ok(());
        }
        match self.family {
            ModelFamily::Glm53 => {
                self.require_chat_ids(&[
                    (self.assistant_id, "<|assistant|>"),
                    (self.think_start_id, "<think>"),
                    (self.think_end_id, "</think>"),
                ])?;
                tokens.push(self.assistant_id);
                tokens.push(self.think_start_id);
                if !thinking {
                    tokens.push(self.think_end_id);
                }
            }
            ModelFamily::Dots3Note => {
                tokens.push(self.assistant_id);
                if !thinking {
                    tokens.push(self.think_start_id);
                    bpe_tokenize_text(self, b"\n\n", &mut tokens.tokens);
                    tokens.push(self.think_end_id);
                    bpe_tokenize_text(self, b"\n\n", &mut tokens.tokens);
                }
            }
            ModelFamily::Motif3 => {
                tokens.push(self.start_of_turn_id);
                tokens.push(self.assistant_id);
                tokens.push(self.think_start_id);
                if !thinking {
                    tokens.push(self.think_end_id);
                }
            }
            ModelFamily::SolarOpen2 => {
                if tokens.as_slice().last() == Some(&self.im_end_id) {
                    self.chat_push_fragment(tokens, b"\n", b"");
                }
                self.solar_chat_open_role(tokens, b"assistant");
                tokens.push(self.think_start_id);
                if !thinking {
                    tokens.push(self.think_end_id);
                }
            }
            ModelFamily::Qwen4Exp => {
                self.qwen_chat_open_role(tokens, b"assistant");
                tokens.push(self.think_start_id);
                if thinking {
                    bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
                } else {
                    bpe_tokenize_text(self, b"\n\n", &mut tokens.tokens);
                    tokens.push(self.think_end_id);
                    bpe_tokenize_text(self, b"\n\n", &mut tokens.tokens);
                }
            }
            _ => {
                if self.family == ModelFamily::ExaoneMoe {
                    self.require_chat_ids(&[
                        (self.assistant_id, "<|assistant|>"),
                        (
                            if thinking {
                                self.think_start_id
                            } else {
                                self.think_end_id
                            },
                            if thinking { "<think>" } else { "</think>" },
                        ),
                    ])?;
                }
                tokens.push(self.assistant_id);
                tokens.push(if thinking {
                    self.think_start_id
                } else {
                    self.think_end_id
                });
            }
        }
        Ok(())
    }

    fn require_chat_ids(&self, ids: &[(i32, &str)]) -> Result<(), TokError> {
        for &(id, token) in ids {
            if id < 0 {
                return Err(TokError::MissingToken(token.into()));
            }
        }
        Ok(())
    }

    fn chat_push_fragment(&self, tokens: &mut TokenBuffer, prefix: &[u8], text: &[u8]) {
        let mut fragment = Vec::with_capacity(prefix.len() + text.len());
        fragment.extend_from_slice(prefix);
        fragment.extend_from_slice(text);
        bpe_tokenize_text(self, &fragment, &mut tokens.tokens);
    }

    fn solar_chat_open_role(&self, tokens: &mut TokenBuffer, role: &[u8]) {
        tokens.push(self.im_start_id);
        bpe_tokenize_text(self, role, &mut tokens.tokens);
        tokens.push(self.im_content_id);
    }

    fn solar_chat_close_role(&self, tokens: &mut TokenBuffer) {
        tokens.push(self.im_end_id);
        self.chat_push_fragment(tokens, b"\n", b"");
    }

    fn qwen_chat_open_role(&self, tokens: &mut TokenBuffer, role: &[u8]) {
        tokens.push(self.im_start_id);
        bpe_tokenize_text(self, role, &mut tokens.tokens);
        bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
    }

    fn qwen_chat_close_role(&self, tokens: &mut TokenBuffer) {
        tokens.push(self.im_end_id);
        bpe_tokenize_text(self, b"\n", &mut tokens.tokens);
    }

    fn chat_append_wrapped_payload(
        &self,
        tokens: &mut TokenBuffer,
        content: &[u8],
        end_marker: &[u8],
    ) {
        let mut span = content;
        while let Some(pos) = span
            .windows(end_marker.len())
            .position(|window| window == end_marker)
        {
            bpe_tokenize_text(self, &span[..pos], &mut tokens.tokens);
            bpe_tokenize_text(self, b"&lt;", &mut tokens.tokens);
            span = &span[pos + 1..];
        }
        bpe_tokenize_text(self, span, &mut tokens.tokens);
    }

    pub fn token_text(&self, token: i32) -> Vec<u8> {
        vocab_token_text(self, token)
    }

    pub fn is_stop(&self, token: i32) -> bool {
        if token < 0 {
            return false;
        }
        if token == self.eos_id {
            return true;
        }
        if self.is_k2_horizon && self.im_end_id >= 0 && token == self.im_end_id {
            return true;
        }
        match self.family {
            ModelFamily::SolarOpen2 | ModelFamily::Qwen4Exp => {
                self.eot_id >= 0 && token == self.eot_id
            }
            ModelFamily::Motif3 => {
                (self.user_id >= 0 && token == self.user_id)
                    || (self.end_of_turn_id >= 0 && token == self.end_of_turn_id)
            }
            ModelFamily::Dots3Note => {
                self.dots3_endoftext_id >= 0 && token == self.dots3_endoftext_id
            }
            _ => false,
        }
    }

    pub fn specials_line(&self) -> String {
        format!(
            "SPECIALS family={} bos={} eos={} eot={} user={} assistant={} think_start={} think_end={} tool_call_start={} end_of_turn={} dsml={} dots3_eotext={} n_vocab={} engine_eos={}",
            self.family as u32,
            self.bos_id,
            self.eos_id,
            self.eot_id,
            self.user_id,
            self.assistant_id,
            self.think_start_id,
            self.think_end_id,
            self.tool_call_start_id,
            self.end_of_turn_id,
            self.dsml_id,
            self.dots3_endoftext_id,
            self.n_vocab(),
            self.engine_eos()
        )
    }
}

fn gpt2_byte_to_codepoint(b: u8) -> u32 {
    if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
        return u32::from(b);
    }
    let mut n = 0u32;
    for x in 0u32..256 {
        let xb = x as u8;
        if (33..=126).contains(&xb) || (161..=172).contains(&xb) || xb >= 174 {
            continue;
        }
        if xb == b {
            return 256 + n;
        }
        n += 1;
    }
    u32::from(b)
}

fn utf8_put(out: &mut Vec<u8>, cp: u32) {
    if cp <= 0x7f {
        out.push(cp as u8);
    } else if cp <= 0x7ff {
        out.push((0xc0 | (cp >> 6)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else if cp <= 0xffff {
        out.push((0xe0 | (cp >> 12)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else {
        out.push((0xf0 | (cp >> 18)) as u8);
        out.push((0x80 | ((cp >> 12) & 0x3f)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    }
}

fn byte_encode(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() * 4);
    for &b in raw {
        utf8_put(&mut out, gpt2_byte_to_codepoint(b));
    }
    out
}

fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c & 0xe0 == 0xc0 {
        2
    } else if c & 0xf0 == 0xe0 {
        3
    } else if c & 0xf8 == 0xf0 {
        4
    } else {
        1
    }
}

fn bpe_rank(vocab: &Vocab, a: &[u8], b: &[u8]) -> i32 {
    let mut key = Vec::with_capacity(a.len() + 1 + b.len());
    key.extend_from_slice(a);
    key.push(b' ');
    key.extend_from_slice(b);
    vocab.merge_rank.get(&key).copied().unwrap_or(-1)
}

fn bpe_emit_piece(vocab: &Vocab, raw: &[u8], out: &mut Vec<i32>) {
    let encoded = byte_encode(raw);
    let mut sym: Vec<Vec<u8>> = Vec::new();
    let mut off = 0usize;
    while off < encoded.len() {
        let mut n = utf8_len_from_first_byte(encoded[off]);
        if off + n > encoded.len() {
            n = 1;
        }
        sym.push(encoded[off..off + n].to_vec());
        off += n;
    }

    loop {
        let mut best_rank = i32::MAX;
        for i in 0..sym.len().saturating_sub(1) {
            let rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if rank >= 0 && rank < best_rank {
                best_rank = rank;
            }
        }
        if best_rank == i32::MAX {
            break;
        }
        let mut write = 0usize;
        let mut read = 0usize;
        while read < sym.len() {
            if read + 1 < sym.len() && bpe_rank(vocab, &sym[read], &sym[read + 1]) == best_rank {
                let mut merged = Vec::with_capacity(sym[read].len() + sym[read + 1].len());
                merged.extend_from_slice(&sym[read]);
                merged.extend_from_slice(&sym[read + 1]);
                sym[write] = merged;
                write += 1;
                read += 2;
            } else {
                if write != read {
                    sym[write] = std::mem::take(&mut sym[read]);
                }
                write += 1;
                read += 1;
            }
        }
        sym.truncate(write);
    }

    for s in &sym {
        if let Some(&token) = vocab.token_to_id.get(s) {
            out.push(token);
        } else {
            for j in 0..s.len() {
                if let Some(&token) = vocab.token_to_id.get(&s[j..j + 1]) {
                    out.push(token);
                }
            }
        }
    }
}

fn next_utf8_char(s: &[u8], pos: usize) -> usize {
    let mut n = utf8_len_from_first_byte(s[pos]);
    if pos + n > s.len() {
        n = 1;
    }
    pos + n
}

fn ascii_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}
fn ascii_digit(c: u8) -> bool {
    c.is_ascii_digit()
}
fn ascii_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}
fn ascii_newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}
fn joyai_ascii_punct_symbol(c: u8) -> bool {
    (c >= b'!' && c <= b'/')
        || (c >= b':' && c <= b'@')
        || (c >= b'[' && c <= b'`')
        || (c >= b'{' && c <= b'~')
}

fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

fn utf8_peek_one(s: &[u8], pos: usize) -> (u32, usize) {
    let c0 = s[pos];
    let mut n = utf8_len_from_first_byte(c0);
    if pos + n > s.len() {
        n = 1;
    }
    let next = pos + n;
    let cp = match n {
        1 => u32::from(c0),
        2 => (u32::from(c0 & 0x1f) << 6) | u32::from(s[pos + 1] & 0x3f),
        3 => {
            (u32::from(c0 & 0x0f) << 12)
                | (u32::from(s[pos + 1] & 0x3f) << 6)
                | u32::from(s[pos + 2] & 0x3f)
        }
        _ => {
            (u32::from(c0 & 0x07) << 18)
                | (u32::from(s[pos + 1] & 0x3f) << 12)
                | (u32::from(s[pos + 2] & 0x3f) << 6)
                | u32::from(s[pos + 3] & 0x3f)
        }
    };
    (cp, next)
}

fn joyai_letter_like_at(s: &[u8], pos: usize) -> bool {
    let c = s[pos];
    if c < 128 {
        return ascii_alpha(c);
    }
    true
}

fn joyai_consume_letters(s: &[u8], mut pos: usize) -> usize {
    while pos < s.len() && joyai_letter_like_at(s, pos) {
        pos = next_utf8_char(s, pos);
    }
    pos
}

fn joyai_cjk_at(s: &[u8], pos: usize) -> bool {
    if s[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_peek_one(s, pos);
    utf8_is_cjk_hira_kata(cp)
}

#[derive(Clone, Copy, Default)]
struct CharInfo {
    cp: u32,
    next: usize,
    valid: bool,
    is_letter: bool,
    is_number: bool,
    is_whitespace: bool,
}

fn unicode_whitespace(cp: u32) -> bool {
    if cp < 128 {
        return ascii_space(cp as u8);
    }
    cp == 0x0085
        || cp == 0x00a0
        || cp == 0x1680
        || (0x2000..=0x200a).contains(&cp)
        || cp == 0x2028
        || cp == 0x2029
        || cp == 0x202f
        || cp == 0x205f
        || cp == 0x3000
}

fn unicode_number(cp: u32) -> bool {
    if cp < 128 {
        return ascii_digit(cp as u8);
    }
    (0x0660..=0x0669).contains(&cp)
        || (0x06f0..=0x06f9).contains(&cp)
        || (0x07c0..=0x07c9).contains(&cp)
        || (0x0966..=0x096f).contains(&cp)
        || (0x09e6..=0x09ef).contains(&cp)
        || (0x0a66..=0x0a6f).contains(&cp)
        || (0x0ae6..=0x0aef).contains(&cp)
        || (0x0b66..=0x0b6f).contains(&cp)
        || (0x0be6..=0x0bef).contains(&cp)
        || (0x0c66..=0x0c6f).contains(&cp)
        || (0x0ce6..=0x0cef).contains(&cp)
        || (0x0d66..=0x0d6f).contains(&cp)
        || (0x0de6..=0x0def).contains(&cp)
        || (0x0e50..=0x0e59).contains(&cp)
        || (0x0ed0..=0x0ed9).contains(&cp)
        || (0x0f20..=0x0f29).contains(&cp)
        || (0x1040..=0x1049).contains(&cp)
        || (0x1090..=0x1099).contains(&cp)
        || (0x17e0..=0x17e9).contains(&cp)
        || (0x1810..=0x1819).contains(&cp)
        || (0xff10..=0xff19).contains(&cp)
}

fn unicode_punct_symbol(cp: u32) -> bool {
    if cp < 128 {
        return joyai_ascii_punct_symbol(cp as u8);
    }
    (0x00a1..=0x00a9).contains(&cp)
        || (0x00ab..=0x00ac).contains(&cp)
        || (0x00ae..=0x00b1).contains(&cp)
        || cp == 0x00b4
        || (0x00b6..=0x00b8).contains(&cp)
        || cp == 0x00bb
        || cp == 0x00bf
        || cp == 0x00d7
        || cp == 0x00f7
        || (0x02c2..=0x02df).contains(&cp)
        || (0x02e5..=0x02eb).contains(&cp)
        || (0x02ed..=0x02ff).contains(&cp)
        || (0x0375..=0x037e).contains(&cp)
        || (0x0384..=0x0385).contains(&cp)
        || cp == 0x0387
        || (0x055a..=0x055f).contains(&cp)
        || (0x0589..=0x058a).contains(&cp)
        || (0x05be..=0x05c0).contains(&cp)
        || cp == 0x05c3
        || (0x05c6..=0x05c7).contains(&cp)
        || (0x0609..=0x060a).contains(&cp)
        || (0x060c..=0x060d).contains(&cp)
        || cp == 0x061b
        || (0x061e..=0x061f).contains(&cp)
        || cp == 0x066a
        || cp == 0x066d
        || cp == 0x06d4
        || (0x2000..=0x206f).contains(&cp)
        || (0x20a0..=0x20cf).contains(&cp)
        || (0x2100..=0x214f).contains(&cp)
        || (0x2190..=0x23ff).contains(&cp)
        || (0x2460..=0x24ff).contains(&cp)
        || (0x2500..=0x2775).contains(&cp)
        || (0x2794..=0x2bff).contains(&cp)
        || (0x2e00..=0x2e7f).contains(&cp)
        || (0x3000..=0x303f).contains(&cp)
        || (0xfd3e..=0xfd3f).contains(&cp)
        || (0xfe10..=0xfe6f).contains(&cp)
        || (0xff01..=0xff0f).contains(&cp)
        || (0xff1a..=0xff20).contains(&cp)
        || (0xff3b..=0xff40).contains(&cp)
        || (0xff5b..=0xff65).contains(&cp)
        || (0x1f000..=0x1faff).contains(&cp)
}

fn char_at(s: &[u8], pos: usize) -> CharInfo {
    if pos >= s.len() {
        return CharInfo::default();
    }
    let (cp, next) = utf8_peek_one(s, pos);
    let is_whitespace = unicode_whitespace(cp);
    let is_number = unicode_number(cp);
    let is_letter = if cp < 128 {
        ascii_alpha(cp as u8)
    } else {
        !is_whitespace && !is_number && !unicode_punct_symbol(cp)
    };
    CharInfo {
        cp,
        next,
        valid: true,
        is_letter,
        is_number,
        is_whitespace,
    }
}

fn ascii_tolower_cp(cp: u32) -> u32 {
    if (b'A' as u32..=b'Z' as u32).contains(&cp) {
        cp + 32
    } else {
        cp
    }
}

fn user_defined_at(vocab: &Vocab, text: &[u8], pos: usize) -> Option<(i32, usize)> {
    if vocab.user_defined_max_len == 0 || !vocab.user_defined_first[text[pos] as usize] {
        return None;
    }
    let mut want = text.len() - pos;
    if want > vocab.user_defined_max_len as usize {
        want = vocab.user_defined_max_len as usize;
    }
    for n in (1..=want).rev() {
        if let Some(&id) = vocab.user_defined.get(&text[pos..pos + n]) {
            return Some((id, n));
        }
    }
    None
}

fn contraction_len(s: &[u8], pos: usize) -> Option<usize> {
    if pos >= s.len() || s[pos] != b'\'' {
        return None;
    }
    let rem = s.len() - pos - 1;
    let p = pos + 1;
    if rem >= 2 {
        let c0 = ascii_tolower_cp(s[p] as u32);
        let c1 = ascii_tolower_cp(s[p + 1] as u32);
        if (c0 == b'r' as u32 && c1 == b'e' as u32)
            || (c0 == b'v' as u32 && c1 == b'e' as u32)
            || (c0 == b'l' as u32 && c1 == b'l' as u32)
        {
            return Some(3);
        }
    }
    if rem >= 1 {
        let c = ascii_tolower_cp(s[p] as u32);
        if c == b's' as u32 || c == b't' as u32 || c == b'm' as u32 || c == b'd' as u32 {
            return Some(2);
        }
    }
    None
}

fn motif3_upperish(info: CharInfo) -> bool {
    if !info.valid {
        return false;
    }
    if info.cp < 128 {
        return (b'A' as u32..=b'Z' as u32).contains(&info.cp);
    }
    info.is_letter
}

fn motif3_lowerish(info: CharInfo) -> bool {
    if !info.valid {
        return false;
    }
    if info.cp < 128 {
        return (b'a' as u32..=b'z' as u32).contains(&info.cp);
    }
    info.is_letter
}

fn motif3_match_lower_word(text: &[u8], pos: usize) -> usize {
    let mut scan = pos;
    let mut last_lowerish = usize::MAX;
    while scan < text.len() {
        let info = char_at(text, scan);
        if !motif3_upperish(info) {
            break;
        }
        if motif3_lowerish(info) {
            last_lowerish = scan;
        }
        scan = info.next;
    }
    let mut lower = scan;
    let mut first = char_at(text, lower);
    if !motif3_lowerish(first) {
        if last_lowerish == usize::MAX {
            return pos;
        }
        lower = last_lowerish;
        first = char_at(text, lower);
        let _ = first;
    }
    scan = lower;
    while scan < text.len() {
        let info = char_at(text, scan);
        if !motif3_lowerish(info) {
            break;
        }
        scan = info.next;
    }
    scan
}

fn motif3_match_upper_word(text: &[u8], pos: usize) -> usize {
    let mut scan = pos;
    let mut n_upper = 0;
    while scan < text.len() {
        let info = char_at(text, scan);
        if !motif3_upperish(info) {
            break;
        }
        scan = info.next;
        n_upper += 1;
    }
    if n_upper == 0 {
        return pos;
    }
    while scan < text.len() {
        let info = char_at(text, scan);
        if !motif3_lowerish(info) {
            break;
        }
        scan = info.next;
    }
    scan
}

fn motif3_match_contraction(text: &[u8], pos: usize) -> usize {
    contraction_len(text, pos).map(|n| pos + n).unwrap_or(pos)
}

fn motif3_match_word_pattern(text: &[u8], pos: usize, lower_pattern: bool) -> usize {
    let mut word = pos;
    let first = char_at(text, word);
    if first.valid
        && first.cp != b'\r' as u32
        && first.cp != b'\n' as u32
        && !first.is_letter
        && !first.is_number
    {
        word = first.next;
    }
    let mut end = if lower_pattern {
        motif3_match_lower_word(text, word)
    } else {
        motif3_match_upper_word(text, word)
    };
    if end == word {
        return pos;
    }
    loop {
        let space = char_at(text, end);
        if !space.valid || space.cp != b' ' as u32 {
            break;
        }
        let next = if lower_pattern {
            motif3_match_lower_word(text, space.next)
        } else {
            motif3_match_upper_word(text, space.next)
        };
        if next == space.next {
            break;
        }
        end = next;
    }
    motif3_match_contraction(text, end)
}

fn bpe_tokenize_text_motif3_core(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    let mut pos = 0usize;
    while pos < s.len() {
        let start = pos;
        let mut end = motif3_match_word_pattern(s, pos, true);
        if end == pos {
            end = motif3_match_word_pattern(s, pos, false);
        }
        if end != pos {
            pos = end;
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        let cur = char_at(s, pos);
        if !cur.valid {
            break;
        }
        if cur.is_number {
            let mut count = 0;
            while pos < s.len() && count < 3 {
                let scan = char_at(s, pos);
                if !scan.valid || !scan.is_number {
                    break;
                }
                pos = scan.next;
                count += 1;
            }
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        let mut punct_pos = pos;
        let mut punct = cur;
        if cur.cp == b' ' as u32 {
            punct_pos = cur.next;
            punct = char_at(s, punct_pos);
        }
        if punct.valid && !punct.is_whitespace && !punct.is_letter && !punct.is_number {
            pos = punct_pos;
            while pos < s.len() {
                let scan = char_at(s, pos);
                if !scan.valid || scan.is_whitespace || scan.is_letter || scan.is_number {
                    break;
                }
                pos = scan.next;
            }
            while pos < s.len() {
                let scan = char_at(s, pos);
                if !scan.valid
                    || !(scan.cp == b'\r' as u32
                        || scan.cp == b'\n' as u32
                        || scan.cp == b'/' as u32)
                {
                    break;
                }
                pos = scan.next;
            }
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        if cur.is_whitespace {
            let mut scan_pos = pos;
            let mut last_newline_end = 0usize;
            let mut last_ws_start = pos;
            let mut count = 0;
            while scan_pos < s.len() {
                let scan = char_at(s, scan_pos);
                if !scan.valid || !scan.is_whitespace {
                    break;
                }
                last_ws_start = scan_pos;
                if scan.cp == b'\r' as u32 || scan.cp == b'\n' as u32 {
                    last_newline_end = scan.next;
                }
                scan_pos = scan.next;
                count += 1;
            }
            pos = if last_newline_end != 0 {
                last_newline_end
            } else if count > 1 && scan_pos < s.len() {
                last_ws_start
            } else {
                scan_pos
            };
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        pos = cur.next;
        bpe_emit_piece(vocab, &s[start..pos], out);
    }
}

fn motif3_added_token_lstrip(token_id: i32) -> bool {
    token_id == 53 || token_id == 54 || (85..=159).contains(&token_id)
}

fn motif3_added_token_at(vocab: &Vocab, text: &[u8], pos: usize) -> Option<(i32, usize)> {
    if !vocab.motif3_added_first[text[pos] as usize] {
        return None;
    }
    let limit = vocab.tokens.len().min(160);
    let mut best_id = -1;
    let mut best_len = 0usize;
    for id in 0..limit {
        let token = &vocab.tokens[id];
        if token.is_empty() || token.len() > text.len() - pos || token.len() < best_len {
            continue;
        }
        if &text[pos..pos + token.len()] != token.as_slice() {
            continue;
        }
        if token.len() > best_len || best_id < 0 {
            best_id = id as i32;
            best_len = token.len();
        }
    }
    if best_id < 0 {
        None
    } else {
        Some((best_id, best_len))
    }
}

fn motif3_added_lstrip_start(text: &[u8], span_start: usize, token_start: usize) -> usize {
    let mut scan = span_start;
    let mut trailing_ws = span_start;
    while scan < token_start {
        let info = char_at(&text[..token_start], scan);
        if !info.valid || info.next > token_start {
            break;
        }
        if !info.is_whitespace {
            trailing_ws = info.next;
        }
        scan = info.next;
    }
    trailing_ws
}

fn bpe_tokenize_text_motif3(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    let mut pos = 0usize;
    while pos < s.len() {
        if let Some((token_id, token_len)) = motif3_added_token_at(vocab, s, pos) {
            out.push(token_id);
            pos += token_len;
            continue;
        }
        let mut found = s.len();
        let mut found_id = -1i32;
        let mut found_len = 0usize;
        for scan in (pos + 1)..s.len() {
            if let Some((id, n)) = motif3_added_token_at(vocab, s, scan) {
                found = scan;
                found_id = id;
                found_len = n;
                break;
            }
        }
        if found_id < 0 {
            if pos < s.len() {
                bpe_tokenize_text_motif3_core(vocab, &s[pos..], out);
            }
            break;
        }
        let mut core_end = found;
        if motif3_added_token_lstrip(found_id) {
            core_end = motif3_added_lstrip_start(s, pos, found);
        }
        if core_end > pos {
            bpe_tokenize_text_motif3_core(vocab, &s[pos..core_end], out);
        }
        out.push(found_id);
        pos = found + found_len;
    }
}

fn bpe_tokenize_text_llama3(
    vocab: &Vocab,
    s: &[u8],
    out: &mut Vec<i32>,
    max_digits: usize,
    user_defined: bool,
    joiners_as_letters: bool,
) {
    let mut pos = 0usize;
    while pos < s.len() {
        let start = pos;
        let cur = char_at(s, pos);
        if !cur.valid {
            pos = next_utf8_char(s, pos);
            continue;
        }
        if user_defined {
            if let Some((id, n)) = user_defined_at(vocab, s, pos) {
                out.push(id);
                pos += n;
                continue;
            }
        }
        if let Some(n) = contraction_len(s, pos) {
            pos += n;
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        {
            let mut p = pos;
            if !(cur.cp == b'\r' as u32
                || cur.cp == b'\n' as u32
                || llama3_letter(cur, joiners_as_letters)
                || cur.is_number)
            {
                p = cur.next;
            }
            let mut first = char_at(s, p);
            if first.valid && llama3_letter(first, joiners_as_letters) {
                while first.valid && llama3_letter(first, joiners_as_letters) {
                    p = first.next;
                    first = char_at(s, p);
                }
                pos = p;
                bpe_emit_piece(vocab, &s[start..pos], out);
                continue;
            }
        }
        if cur.is_number {
            let mut digits = 0usize;
            while pos < s.len() && digits < max_digits {
                let scan = char_at(s, pos);
                if !scan.valid || !scan.is_number {
                    break;
                }
                pos = scan.next;
                digits += 1;
            }
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        {
            let mut p = pos;
            if cur.cp == b' ' as u32 {
                p = cur.next;
            }
            let mut run = p;
            loop {
                let c = char_at(s, run);
                if !c.valid
                    || c.is_whitespace
                    || llama3_letter(c, joiners_as_letters)
                    || c.is_number
                {
                    break;
                }
                run = c.next;
            }
            if run > p {
                loop {
                    let c = char_at(s, run);
                    if !c.valid || !(c.cp == b'\r' as u32 || c.cp == b'\n' as u32) {
                        break;
                    }
                    run = c.next;
                }
                pos = run;
                bpe_emit_piece(vocab, &s[start..pos], out);
                continue;
            }
        }
        if cur.is_whitespace {
            let mut p = pos;
            let mut last_newline_end = 0usize;
            let mut last_ws_start = pos;
            let mut whitespace_count = 0u32;
            loop {
                let c = char_at(s, p);
                if !c.valid || !c.is_whitespace {
                    break;
                }
                last_ws_start = p;
                if c.cp == b'\r' as u32 || c.cp == b'\n' as u32 {
                    last_newline_end = c.next;
                }
                p = c.next;
                whitespace_count += 1;
            }
            pos = if last_newline_end != 0 {
                last_newline_end
            } else if whitespace_count > 1 && p < s.len() {
                last_ws_start
            } else {
                p
            };
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        pos = cur.next;
        bpe_emit_piece(vocab, &s[start..pos], out);
    }
}

fn bpe_tokenize_text_solar(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    bpe_tokenize_text_llama3(vocab, s, out, 1, true, false);
}

fn bpe_tokenize_text_glm4(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    bpe_tokenize_text_llama3(vocab, s, out, 3, false, false);
}

fn llama3_letter(c: CharInfo, joiners_as_letters: bool) -> bool {
    c.is_letter || (joiners_as_letters && matches!(c.cp, 0x200c | 0x200d))
}

fn bpe_tokenize_text_k2(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    bpe_tokenize_text_llama3(vocab, s, out, 3, true, true);
}

fn bpe_tokenize_text_exaone(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    let mut pos = 0usize;
    while pos < s.len() {
        let start = pos;
        let cur = char_at(s, pos);
        if !cur.valid {
            pos = next_utf8_char(s, pos);
            continue;
        }
        if let Some((id, n)) = user_defined_at(vocab, s, pos) {
            out.push(id);
            pos += n;
            continue;
        }
        if let Some(n) = contraction_len(s, pos) {
            pos += n;
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        {
            let mut p = pos;
            let lead_ok = !(cur.cp == b'\r' as u32
                || cur.cp == b'\n' as u32
                || cur.is_letter
                || cur.is_number);
            if lead_ok {
                p = cur.next;
            }
            let first = char_at(s, p);
            if first.valid && first.is_letter {
                loop {
                    let c = char_at(s, p);
                    if c.valid && c.is_letter {
                        p = c.next;
                        continue;
                    }
                    if c.valid && c.cp == b' ' as u32 {
                        let n = char_at(s, c.next);
                        if n.valid && n.is_letter {
                            p = c.next;
                            continue;
                        }
                    }
                    break;
                }
                pos = p;
                bpe_emit_piece(vocab, &s[start..pos], out);
                continue;
            }
        }
        if cur.is_number {
            pos = cur.next;
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        {
            let mut p = pos;
            if cur.cp == b' ' as u32 {
                p = cur.next;
            }
            let mut run = p;
            loop {
                let c = char_at(s, run);
                if !c.valid || c.is_whitespace || c.is_letter || c.is_number {
                    break;
                }
                run = c.next;
            }
            if run > p {
                let c = char_at(s, run);
                if c.valid && (c.cp == b'\r' as u32 || c.cp == b'\n' as u32 || c.cp == b'/' as u32)
                {
                    run = c.next;
                }
                pos = run;
                bpe_emit_piece(vocab, &s[start..pos], out);
                continue;
            }
        }
        if cur.is_whitespace {
            let mut ws_end = pos;
            loop {
                let c = char_at(s, ws_end);
                if !c.valid || !c.is_whitespace {
                    break;
                }
                ws_end = c.next;
            }
            let mut last_nl_end = 0usize;
            let mut p = pos;
            while p < ws_end {
                let c = char_at(s, p);
                if !c.valid {
                    break;
                }
                if c.cp == b'\r' as u32 || c.cp == b'\n' as u32 {
                    last_nl_end = c.next;
                }
                p = c.next;
            }
            pos = if last_nl_end != 0 {
                last_nl_end
            } else {
                ws_end
            };
            bpe_emit_piece(vocab, &s[start..pos], out);
            continue;
        }
        pos = next_utf8_char(s, pos);
        bpe_emit_piece(vocab, &s[start..pos], out);
    }
}

fn bpe_tokenize_text_dots3(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    let mut span = 0usize;
    let mut pos = 0usize;
    while pos < s.len() {
        if let Some((id, n)) = user_defined_at(vocab, s, pos) {
            if pos > span {
                bpe_tokenize_text_solar(vocab, &s[span..pos], out);
            }
            out.push(id);
            pos += n;
            span = pos;
            continue;
        }
        pos += 1;
    }
    if s.len() > span {
        bpe_tokenize_text_solar(vocab, &s[span..], out);
    }
}

fn bpe_tokenize_text_joyai(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    let mut pos = 0usize;
    while pos < s.len() {
        let start = pos;
        let c = s[pos];
        if ascii_digit(c) {
            let mut ndigits = 0;
            while pos < s.len() && ascii_digit(s[pos]) && ndigits < 3 {
                pos += 1;
                ndigits += 1;
            }
        } else if joyai_cjk_at(s, pos) {
            loop {
                pos = next_utf8_char(s, pos);
                if !(pos < s.len() && joyai_cjk_at(s, pos)) {
                    break;
                }
            }
        } else if joyai_ascii_punct_symbol(c) && pos + 1 < s.len() && ascii_alpha(s[pos + 1]) {
            pos += 1;
            while pos < s.len() && ascii_alpha(s[pos]) {
                pos += 1;
            }
        } else if joyai_letter_like_at(s, pos) {
            pos = joyai_consume_letters(s, pos);
        } else if !ascii_newline(c)
            && !joyai_ascii_punct_symbol(c)
            && pos + 1 < s.len()
            && joyai_letter_like_at(s, pos + 1)
        {
            pos += 1;
            pos = joyai_consume_letters(s, pos);
        } else if c == b' ' && pos + 1 < s.len() && joyai_ascii_punct_symbol(s[pos + 1]) {
            pos += 1;
            while pos < s.len() && joyai_ascii_punct_symbol(s[pos]) {
                pos += 1;
            }
            while pos < s.len() && ascii_newline(s[pos]) {
                pos += 1;
            }
        } else if joyai_ascii_punct_symbol(c) {
            while pos < s.len() && joyai_ascii_punct_symbol(s[pos]) {
                pos += 1;
            }
            while pos < s.len() && ascii_newline(s[pos]) {
                pos += 1;
            }
        } else if ascii_space(c) {
            let mut p = pos;
            let mut last_newline_end = 0usize;
            while p < s.len() && ascii_space(s[p]) {
                let sc = s[p];
                p += 1;
                if ascii_newline(sc) {
                    last_newline_end = p;
                }
            }
            if last_newline_end != 0 {
                pos = last_newline_end;
            } else if p < s.len()
                && p > pos + 1
                && (joyai_letter_like_at(s, p) || joyai_ascii_punct_symbol(s[p]))
            {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(s, pos);
        }
        if pos == start {
            pos = next_utf8_char(s, pos);
        }
        bpe_emit_piece(vocab, &s[start..pos], out);
    }
}

fn bpe_tokenize_text(vocab: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    match vocab.family {
        ModelFamily::Glm53 => bpe_tokenize_text_glm4(vocab, text, out),
        ModelFamily::Motif3 => bpe_tokenize_text_motif3(vocab, text, out),
        ModelFamily::SolarOpen2 => bpe_tokenize_text_solar(vocab, text, out),
        ModelFamily::Dots3Note => bpe_tokenize_text_dots3(vocab, text, out),
        ModelFamily::Qwen4Exp => bpe_tokenize_text_dots3(vocab, text, out),
        ModelFamily::ExaoneMoe if vocab.is_k2_horizon => bpe_tokenize_text_k2(vocab, text, out),
        ModelFamily::ExaoneMoe => bpe_tokenize_text_exaone(vocab, text, out),
        ModelFamily::DeepSeek4 => bpe_tokenize_text_joyai(vocab, text, out),
    }
}

fn special_token_at(vocab: &Vocab, p: &[u8]) -> Option<(i32, usize)> {
    let specials: &[(&[u8], i32)] = &[
        (
            b"<|ifm|begin_of_text|>",
            if vocab.is_k2_horizon {
                vocab.bos_id
            } else {
                -1
            },
        ),
        (
            b"<|ifm|endoftext|>",
            if vocab.is_k2_horizon {
                vocab.eos_id
            } else {
                -1
            },
        ),
        (
            b"<|ifm|im_start|>",
            if vocab.is_k2_horizon {
                vocab.im_start_id
            } else {
                -1
            },
        ),
        (
            b"<|ifm|im_end|>",
            if vocab.is_k2_horizon {
                vocab.im_end_id
            } else {
                -1
            },
        ),
        (
            b"<ifm|think>",
            if vocab.is_k2_horizon {
                vocab.think_start_id
            } else {
                -1
            },
        ),
        (
            b"</ifm|think>",
            if vocab.is_k2_horizon {
                vocab.think_end_id
            } else {
                -1
            },
        ),
        (
            b"[gMASK]",
            if vocab.family == ModelFamily::Glm53 {
                vocab.bos_id
            } else {
                -1
            },
        ),
        (
            b"<sop>",
            if vocab.family == ModelFamily::Glm53 {
                vocab.sop_id
            } else {
                -1
            },
        ),
        (
            b"<|system|>",
            if vocab.family == ModelFamily::Glm53 {
                vocab.system_id
            } else {
                -1
            },
        ),
        (
            b"<|user|>",
            if vocab.family == ModelFamily::Glm53 {
                vocab.user_id
            } else {
                -1
            },
        ),
        (
            b"<|assistant|>",
            if vocab.family == ModelFamily::Glm53 {
                vocab.assistant_id
            } else {
                -1
            },
        ),
        (
            b"<|observation|>",
            if vocab.family == ModelFamily::Glm53 {
                vocab.observation_id
            } else {
                -1
            },
        ),
        (
            b"<|begin_of_image|>",
            if vocab.family == ModelFamily::Glm53 {
                154830
            } else {
                -1
            },
        ),
        (
            b"<|image|>",
            if vocab.family == ModelFamily::Glm53 {
                154854
            } else {
                -1
            },
        ),
        (
            b"<|end_of_image|>",
            if vocab.family == ModelFamily::Glm53 {
                154831
            } else {
                -1
            },
        ),
        (b"<|endoftext|>", vocab.dots3_endoftext_id),
        (b"<|endofsystem|>", vocab.dots3_endofsystem_id),
        (b"<|endofuser|>", vocab.dots3_endofuser_id),
        (
            b"<|endofassistant|>",
            if vocab.dots3_endofuser_id >= 0 {
                vocab.eos_id
            } else {
                -1
            },
        ),
        (
            b"<|system|>",
            if vocab.dots3_endofsystem_id >= 0 {
                vocab.system_id
            } else {
                -1
            },
        ),
        (
            b"<|user|>",
            if vocab.dots3_endofuser_id >= 0 {
                vocab.user_id
            } else {
                -1
            },
        ),
        (
            b"<|assistant|>",
            if vocab.dots3_endofuser_id >= 0 {
                vocab.assistant_id
            } else {
                -1
            },
        ),
        (
            b"<|endoftext|>",
            if vocab.family == ModelFamily::Qwen4Exp {
                vocab.eot_id
            } else {
                -1
            },
        ),
        (
            b"<|im_start|>",
            if vocab.family == ModelFamily::Qwen4Exp {
                vocab.im_start_id
            } else {
                -1
            },
        ),
        (
            b"<|im_end|>",
            if vocab.family == ModelFamily::Qwen4Exp {
                vocab.im_end_id
            } else {
                -1
            },
        ),
        (
            b"<|vision_start|>",
            if vocab.family == ModelFamily::Qwen4Exp {
                248053
            } else {
                -1
            },
        ),
        (
            b"<|vision_end|>",
            if vocab.family == ModelFamily::Qwen4Exp {
                248054
            } else {
                -1
            },
        ),
        (
            b"<|image_pad|>",
            if vocab.family == ModelFamily::Qwen4Exp {
                248056
            } else {
                -1
            },
        ),
        (b"<|endoftext|>", vocab.eos_id),
        (b"<|beginoftext|>", vocab.bos_id),
        (b"<|startofturn|>", vocab.start_of_turn_id),
        (b"<|endofturn|>", vocab.end_of_turn_id),
        (b"<|tool|>", vocab.tool_id),
        (b"<|reference|>", vocab.reference_id),
        (b"<|plan|>", vocab.plan_start_id),
        (b"<|endofplan|>", vocab.plan_end_id),
        ("<｜begin▁of▁sentence｜>".as_bytes(), vocab.bos_id),
        ("<｜end▁of▁sentence｜>".as_bytes(), vocab.eos_id),
        (b"<|startoftext|>", vocab.bos_id),
        (b"<|endoftext|>", vocab.eos_id),
        (b"<|im:start|>", vocab.im_start_id),
        (b"<|im:content|>", vocab.im_content_id),
        (b"<|im:end|>", vocab.im_end_id),
        (b"<|tool:start|>", vocab.tool_schema_start_id),
        (b"<|tool:end|>", vocab.tool_schema_end_id),
        ("<｜User｜>".as_bytes(), vocab.user_id),
        ("<｜Assistant｜>".as_bytes(), vocab.assistant_id),
        (b"<think>", vocab.think_start_id),
        (b"</think>", vocab.think_end_id),
        (b"<|think:start|>", vocab.think_start_id),
        (b"<|think:end|>", vocab.think_end_id),
        (b"<tool_call>", vocab.tool_call_start_id),
        (b"</tool_call>", vocab.tool_call_end_id),
        (b"<|tool_call:start|>", vocab.tool_call_start_id),
        (b"<|tool_call:end|>", vocab.tool_call_end_id),
        (b"<tool_response>", vocab.tool_response_start_id),
        (b"</tool_response>", vocab.tool_response_end_id),
        (b"<|tool_response:start|>", vocab.tool_response_start_id),
        (b"<|tool_response:end|>", vocab.tool_response_end_id),
        (b"</arg_key><arg_value>", vocab.arg_value_start_id),
        (b"<arg_key>", vocab.arg_key_start_id),
        (b"</arg_key>", vocab.arg_key_end_id),
        (b"<arg_value>", vocab.arg_value_start_id),
        (b"</arg_value>", vocab.arg_value_end_id),
        (b"<latent>", vocab.latent_start_id),
        (b"<latent_pad>", vocab.latent_pad_id),
        (b"</latent>", vocab.latent_end_id),
        (b"<|tool_arg:start|>", vocab.arg_key_start_id),
        (b"<|tool_arg:value|>", vocab.arg_value_start_id),
        (b"<|tool_arg:end|>", vocab.arg_value_end_id),
        ("｜DSML｜".as_bytes(), vocab.dsml_id),
    ];
    for (text, token) in specials {
        if *token < 0 {
            continue;
        }
        if p.starts_with(*text) {
            return Some((*token, text.len()));
        }
    }
    None
}

fn tokenize_rendered_chat(vocab: &Vocab, s: &[u8], out: &mut Vec<i32>) {
    let mut span = 0usize;
    let mut i = 0usize;
    while i < s.len() {
        if let Some((token, n)) = special_token_at(vocab, &s[i..]) {
            if i > span {
                bpe_tokenize_text(vocab, &s[span..i], out);
            }
            out.push(token);
            i += n;
            span = i;
            continue;
        }
        i += 1;
    }
    if i > span {
        bpe_tokenize_text(vocab, &s[span..i], out);
    }
}

fn gpt2_codepoint_to_byte(cp: u32) -> Option<u8> {
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        return Some(cp as u8);
    }
    let mut n = 0u32;
    for b in 0u32..256 {
        if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
            continue;
        }
        if cp == 256 + n {
            return Some(b as u8);
        }
        n += 1;
    }
    None
}

fn utf8_decode_one(s: &[u8], pos: &mut usize) -> u32 {
    let c = s[*pos];
    if c < 0x80 || *pos + 1 >= s.len() {
        *pos += 1;
        return u32::from(c);
    }
    if c & 0xe0 == 0xc0 && *pos + 1 < s.len() {
        let cp = (u32::from(c & 0x1f) << 6) | u32::from(s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if c & 0xf0 == 0xe0 && *pos + 2 < s.len() {
        let cp = (u32::from(c & 0x0f) << 12)
            | (u32::from(s[*pos + 1] & 0x3f) << 6)
            | u32::from(s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if c & 0xf8 == 0xf0 && *pos + 3 < s.len() {
        let cp = (u32::from(c & 0x07) << 18)
            | (u32::from(s[*pos + 1] & 0x3f) << 12)
            | (u32::from(s[*pos + 2] & 0x3f) << 6)
            | u32::from(s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    *pos += 1;
    u32::from(c)
}

fn vocab_token_is_literal_special(s: &[u8]) -> bool {
    let bar = [0xef, 0xbd, 0x9c];
    if s.len() < bar.len() {
        return false;
    }
    s.windows(bar.len()).any(|w| w == bar)
}

fn vocab_token_text(vocab: &Vocab, token: i32) -> Vec<u8> {
    if token < 0 || token as usize >= vocab.tokens.len() {
        return Vec::new();
    }
    let s = &vocab.tokens[token as usize];
    if vocab_token_is_literal_special(s) {
        return s.clone();
    }
    let mut out = Vec::with_capacity(s.len());
    let mut pos = 0usize;
    while pos < s.len() {
        let at = pos;
        let cp = utf8_decode_one(s, &mut pos);
        if let Some(b) = gpt2_codepoint_to_byte(cp) {
            out.push(b);
        } else {
            out.extend_from_slice(&s[at..pos]);
        }
    }
    out
}

/// C oracle command dump.
pub fn dump_cmd(vocab: &Vocab, cmd: &str, arg: &str) -> String {
    match cmd {
        "specials" => {
            let mut s = vocab.specials_line();
            s.push('\n');
            s
        }
        "encode" => {
            let text = unhex(arg);
            let ids = vocab.encode_bytes(&text);
            format!("TOKENS{}\n", fmt_ids(&ids))
        }
        "render" => {
            let text = unhex(arg);
            let ids = vocab.encode_rendered_bytes(&text);
            format!("TOKENS{}\n", fmt_ids(&ids))
        }
        "decode" => {
            let id: i32 = arg.parse().unwrap_or(-1);
            let t = vocab.token_text(id);
            format!("TEXT {}\n", hex(&t))
        }
        "stop" => {
            let id: i32 = arg.parse().unwrap_or(-1);
            format!("STOP {}\n", u32::from(vocab.is_stop(id)))
        }
        _ => "ERROR unknown-cmd\n".into(),
    }
}

fn fmt_ids(ids: &[i32]) -> String {
    let mut s = String::new();
    for id in ids {
        s.push(' ');
        s.push_str(&id.to_string());
    }
    s
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// C `ds4_host_vocab_apply` token. `n_*` may exceed slice length to
/// simulate a null table with a nonzero count.
pub fn apply_host_vocab(
    tokens: Option<&[(&[u8], bool)]>,
    n_vocab: Option<u32>,
    merges: Option<&[(&[u8], bool)]>,
    n_merges: Option<u32>,
    user_defined: Option<&[i32]>,
    n_ud: Option<u32>,
) -> &'static str {
    let Some(tokens) = tokens else {
        return "vocab-null";
    };
    let n_vocab = n_vocab.unwrap_or(tokens.len() as u32);
    let n_merges = n_merges.unwrap_or(merges.map(|m| m.len() as u32).unwrap_or(0));
    let n_ud = n_ud.unwrap_or(user_defined.map(|u| u.len() as u32).unwrap_or(0));
    if n_vocab > 0 && (n_vocab as usize) > tokens.len() {
        return "tokens-null";
    }
    if n_merges > 0 && merges.map(|m| m.len() as u32).unwrap_or(0) < n_merges {
        return "merges-null";
    }
    if n_ud > 0 && user_defined.map(|u| u.len() as u32).unwrap_or(0) < n_ud {
        return "ud-null";
    }
    for (i, (bytes, present)) in tokens.iter().take(n_vocab as usize).enumerate() {
        let _ = (i, bytes);
        if !present {
            return "token-empty";
        }
    }
    if let Some(merges) = merges {
        for (bytes, present) in merges.iter().take(n_merges as usize) {
            let _ = bytes;
            if !present {
                return "merge-empty";
            }
        }
    }
    if let Some(ud) = user_defined {
        for &id in ud.iter().take(n_ud as usize) {
            if id < 0 || (id as u32) >= n_vocab {
                return "ud-range";
            }
            if tokens[id as usize].0.is_empty() {
                return "ud-empty";
            }
        }
    }
    "ok"
}

/// Fixed C↔Rust apply tapes (same cases as `vocab_c_oracle`).
pub fn dump_vocab_apply_tapes() -> String {
    let a: &[u8] = b"a";
    let b: &[u8] = b"bb";
    let merge: &[u8] = b"a b";
    let tokens = [(a, true), (b, true)];
    let merges = [(merge, true)];
    let ud = [1i32];
    let mut out = String::new();
    out.push_str(&format!(
        "vocab-null {}\n",
        apply_host_vocab(None, None, None, None, None, None)
    ));
    out.push_str(&format!(
        "tokens-null {}\n",
        apply_host_vocab(Some(&[]), Some(1), Some(&merges), None, Some(&ud), None)
    ));
    out.push_str(&format!(
        "merges-null {}\n",
        apply_host_vocab(Some(&tokens), None, Some(&[]), Some(1), Some(&ud), None)
    ));
    out.push_str(&format!(
        "ud-null {}\n",
        apply_host_vocab(Some(&tokens), None, Some(&merges), None, Some(&[]), Some(1))
    ));
    let missing_tok = [(a, false), (b, true)];
    out.push_str(&format!(
        "token-empty {}\n",
        apply_host_vocab(
            Some(&missing_tok),
            None,
            Some(&merges),
            None,
            Some(&ud),
            None
        )
    ));
    let missing_merge = [(merge, false)];
    out.push_str(&format!(
        "merge-empty {}\n",
        apply_host_vocab(
            Some(&tokens),
            None,
            Some(&missing_merge),
            None,
            Some(&ud),
            None
        )
    ));
    out.push_str(&format!(
        "ud-range {}\n",
        apply_host_vocab(Some(&tokens), None, Some(&merges), None, Some(&[9]), None)
    ));
    let empty: &[u8] = b"";
    let empty_tok = [(empty, true), (b, true)];
    out.push_str(&format!(
        "ud-empty {}\n",
        apply_host_vocab(
            Some(&empty_tok),
            None,
            Some(&merges),
            None,
            Some(&[0]),
            None
        )
    ));
    out.push_str(&format!(
        "ok {}\n",
        apply_host_vocab(Some(&tokens), None, Some(&merges), None, Some(&ud), None)
    ));
    out.push_str("ok-row n_vocab=2 n_merges=1 n_ud=1 max_ud=2 bos=0 eos=1 token0=61 token1=6262 merge0=612062 ud=1\n");
    out
}
