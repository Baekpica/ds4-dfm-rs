//! Scripted decode + HTTP generate path. No GGUF.

use ds4_server::parse::{parse_request, ChatMsg, ChatPart, ImageMime, RequestImage, ToolCall};
use ds4_server::route::WireSurface;
use ds4_server::tools::{
    DSML_INVOKE_END, DSML_INVOKE_END_SPACED, DSML_INVOKE_START, DSML_INVOKE_START_SPACED,
    DSML_PARAM_END, DSML_PARAM_END_SPACED, DSML_PARAM_START, DSML_PARAM_START_SPACED,
    DSML_TOOL_CALLS_END, DSML_TOOL_CALLS_END_SPACED, DSML_TOOL_CALLS_START,
    DSML_TOOL_CALLS_START_SPACED,
};
use ds4_server::{
    generate_and_write, generation_blocked, handle_client_inner, render_prompt,
    stop_list_find_from, think_start, ChatFormat, ContStepper, DecodeIo, GenerateError, ParseEnv,
    ParsedRequest, ReqTimings, ScriptedDecode, ScriptedStep, ServerConfig, ServerInner, ThinkMode,
    CREATED_TEST, DSML_ASSISTANT, DSML_BOS, DSML_SYSTEM_TOKEN, DSML_THINK_END, DSML_USER,
    TAPE_PLAIN,
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn env() -> ParseEnv {
    ParseEnv {
        default_model: "ds4".into(),
        default_tokens: 16,
        default_effort: ThinkMode::None,
        default_temp: 0.0,
        live_ids: Vec::new(),
        engine_defaults: false,
    }
}

fn user_req() -> ParsedRequest {
    let mut r = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"thinking":{"type":"disabled"}}"#,
    )
    .unwrap();
    r.think_mode = ThinkMode::None;
    r.temperature = 0.0;
    r
}

fn audit_completion_stream(lane: &str) {
    let cases: &[(&[&[u8]], &str, &str)] = &[
        (&[b"hello", b" world"], "ZZZ", "hello world"),
        (&[b"A", b"B", b"C"], "ZZ", "ABC"),
        (&[b"helloZ", b"Z", b"Ztail"], "ZZZ", "hello"),
        (&[b"\xed", b"\x95\x9c\xea", b"\xb8\x80!"], "ZZZ", "한글!"),
        (&[b"\xed", b"A"], "ZZZ", "�A"),
        (&[b"end\xed"], "ZZZ", "end�"),
    ];
    for &(pieces, stop, expected) in cases {
        for cap in [pieces.len(), pieces.len() + 1] {
            for stream in [false, true] {
                let mut parsed = user_req();
                parsed.kind = ds4_server::route::ReqKind::Completion;
                parsed.prompt_text = Some("prompt".into());
                parsed.stream = stream;
                parsed.stops = vec![stop.into()];
                parsed.max_tokens = cap as i32;
                let mut out = Vec::new();
                match lane {
                    "serial" => {
                        let mut engine = ScriptedDecode::from_pieces(pieces);
                        generate_and_write(
                            &mut engine,
                            &parsed,
                            "audit",
                            CREATED_TEST,
                            false,
                            16,
                            &mut out,
                        )
                        .unwrap();
                    }
                    "push" => {
                        let mut engine = ScriptedV41::new(&[], 8192);
                        engine.script = pieces
                            .iter()
                            .enumerate()
                            .map(|(i, p)| (i as i32 + 1, p.to_vec()))
                            .collect();
                        engine.script.push((engine.stop_id, Vec::new()));
                        generate_and_write(
                            &mut engine,
                            &parsed,
                            "audit",
                            CREATED_TEST,
                            false,
                            16,
                            &mut out,
                        )
                        .unwrap();
                    }
                    "continuous" => {
                        let (mut stepper, head) = ContStepper::new(
                            &parsed,
                            0,
                            "audit",
                            CREATED_TEST,
                            false,
                            16,
                            b"prompt".to_vec(),
                            1,
                            8192,
                        );
                        out.extend(head);
                        for piece in pieces {
                            let step = stepper.feed(piece);
                            out.extend(step.bytes);
                            if step.done {
                                break;
                            }
                        }
                        out.extend(
                            stepper
                                .finalize(cap > pieces.len(), 0, 1, ReqTimings::default(), false)
                                .0,
                        );
                    }
                    _ => unreachable!(),
                }
                let wire = String::from_utf8(out).unwrap();
                let text = if stream {
                    wire.lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter(|line| *line != "[DONE]")
                        .map(|line| {
                            let chunk: serde_json::Value = serde_json::from_str(line).unwrap();
                            chunk["choices"][0]["text"]
                                .as_str()
                                .unwrap_or("")
                                .to_owned()
                        })
                        .collect::<String>()
                } else {
                    let body = wire
                        .split_once("\r\n\r\n")
                        .map_or(wire.as_str(), |(_, b)| b);
                    let body: serde_json::Value = serde_json::from_str(body).unwrap();
                    body["choices"][0]["text"].as_str().unwrap().to_owned()
                };
                assert_eq!(text, expected, "{lane} stream={stream} cap={cap}: {wire}");
            }
        }
    }
}

#[test]
fn audit_completion_serial() {
    audit_completion_stream("serial");
}

#[test]
fn audit_completion_continuous() {
    audit_completion_stream("continuous");
}

#[test]
fn audit_completion_push() {
    audit_completion_stream("push");
}

#[test]
fn ds41_chat_takes_the_c_renderer_with_the_v41_head() {
    // The C-rendered DSML families ship no chat template: model 16 is admitted
    // beside model 0 (chat_input.rs) and renders through render_prompt with the
    // engine's official encoding.py head (server_dsml_render.c:375-383), the
    // scripted model standing in for a V4.1 tokenizer that owns the token.
    let parsed = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}]}"#,
    )
    .unwrap();
    let mut engine = ScriptedDecode::from_pieces(&[b"x"]);
    engine.model_id = ds4_core::Variant::DeepSeek41Flash as i32;
    let rendered = engine.render_request(&parsed).unwrap();
    assert_eq!(
        rendered,
        format!("{DSML_BOS}{DSML_SYSTEM_TOKEN}S{DSML_USER}U{DSML_ASSISTANT}{DSML_THINK_END}")
            .as_bytes()
    );
    // The base V4 family keeps its own head: no system token.
    engine.model_id = ds4_core::Variant::Flash as i32;
    let rendered = engine.render_request(&parsed).unwrap();
    assert_eq!(
        rendered,
        format!("{DSML_BOS}S{DSML_USER}U{DSML_ASSISTANT}{DSML_THINK_END}").as_bytes()
    );
}

struct PromptSyncDecode {
    template: Option<ds4_core::chat_template::Template>,
    inner: ScriptedDecode,
    /// What the sync decided, as the real engine would report it.
    reuse: ds4_core::ReuseTaken,
    miss: ds4_core::ReuseMiss,
    cached_tokens: i32,
    effective_prompt_pos: i32,
    prompt_sync_calls: usize,
    sync_calls: usize,
    disk_eligible: Vec<bool>,
    thinking_visible_eligible: Vec<bool>,
    prompt_sync_elapsed: Option<Duration>,
    remembered: Vec<(Vec<u8>, i32)>,
    invalidations: usize,
    continued_positions: Vec<i32>,
    fail_continued: bool,
    fail_prompt_sync: bool,
    events: Vec<&'static str>,
    replay_raw: Option<String>,
    replay_prompts: Vec<Vec<u8>>,
    rendered: std::cell::RefCell<Vec<Vec<u8>>>,
    remembered_tools: Vec<(Vec<String>, String)>,
    prefix_width: usize,
    prefix_budgets: Vec<i32>,
    trimmed_positions: Vec<i32>,
    sample_params: Vec<(f32, i32, f32, f32)>,
    spec_samplers: Vec<ds4_core::Sampler>,
    prefix_lengths: Vec<usize>,
}

impl PromptSyncDecode {
    fn new(inner: ScriptedDecode, cached_tokens: i32, effective_prompt_pos: i32) -> Self {
        Self {
            template: None,
            inner,
            reuse: ds4_core::ReuseTaken::Cold,
            miss: ds4_core::ReuseMiss::None,
            cached_tokens,
            effective_prompt_pos,
            prompt_sync_calls: 0,
            sync_calls: 0,
            disk_eligible: Vec::new(),
            thinking_visible_eligible: Vec::new(),
            prompt_sync_elapsed: None,
            remembered: Vec::new(),
            invalidations: 0,
            continued_positions: Vec::new(),
            fail_continued: false,
            fail_prompt_sync: false,
            events: Vec::new(),
            replay_raw: None,
            replay_prompts: Vec::new(),
            rendered: std::cell::RefCell::new(Vec::new()),
            remembered_tools: Vec::new(),
            prefix_width: 1,
            prefix_budgets: Vec::new(),
            trimmed_positions: Vec::new(),
            sample_params: Vec::new(),
            spec_samplers: Vec::new(),
            prefix_lengths: Vec::new(),
        }
    }
}

impl DecodeIo for PromptSyncDecode {
    fn template(&self) -> Option<&ds4_core::chat_template::Template> {
        self.template.as_ref()
    }
    fn model_id(&self) -> i32 {
        self.inner.model_id()
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        self.inner.tokenize_text(text)
    }

    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        self.rendered.borrow_mut().push(text.to_vec());
        self.inner.tokenize_rendered_chat(text)
    }

    fn tokenizes_control_literals(&self) -> bool {
        self.inner.tokenizes_control_literals()
    }

    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError> {
        self.inner.token_text(token)
    }

    fn token_is_stop(&self, token: i32) -> bool {
        self.inner.token_is_stop(token)
    }

    fn vision_tokens(&self, data: &[u8]) -> Result<Vec<i32>, GenerateError> {
        self.inner.vision_tokens(data)
    }
    fn sync_vision_prompt(
        &mut self,
        tokens: &[i32],
        images: &[ds4_server::generate::VisionPromptInput],
    ) -> Result<(), GenerateError> {
        self.inner.sync_vision_prompt(tokens, images)
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        self.sync_calls += 1;
        self.inner.sync(tokens)
    }

    fn sync_prompt(
        &mut self,
        _prompt: &[u8],
        tokens: &[i32],
        disk_eligible: bool,
        thinking_visible_eligible: bool,
    ) -> Result<i32, GenerateError> {
        self.events.push("sync");
        self.prompt_sync_calls += 1;
        // The real engine reports what its sync decided; the tape carries
        // whatever the situation under test set.

        self.disk_eligible.push(disk_eligible);
        self.thinking_visible_eligible
            .push(thinking_visible_eligible);
        if self.fail_prompt_sync {
            return Err(GenerateError::Engine("injected prompt sync failure".into()));
        }
        self.inner.live = tokens.to_vec();
        self.inner.pos = self.effective_prompt_pos;
        Ok(self.cached_tokens)
    }

    fn last_reuse(&self) -> ds4_core::ReuseTaken {
        self.reuse
    }

    fn last_miss(&self) -> ds4_core::ReuseMiss {
        self.miss
    }

    fn prompt_sync_elapsed(&self) -> Option<Duration> {
        self.prompt_sync_elapsed
    }

    fn restore_tool_replay(&mut self, messages: &mut [ChatMsg]) {
        self.events.push("restore");
        let Some(raw) = &self.replay_raw else {
            return;
        };
        for message in messages {
            if !message.calls.is_empty() {
                message.raw_dsml = raw.clone();
            }
        }
    }

    fn sync_tool_replay_prompt(
        &mut self,
        prompt: &[u8],
        tokens: &[i32],
    ) -> Result<i32, GenerateError> {
        self.events.push("tool-sync");
        self.replay_prompts.push(prompt.to_vec());
        if self.fail_prompt_sync {
            return Err(GenerateError::Engine("injected prompt sync failure".into()));
        }
        self.inner.live = tokens.to_vec();
        self.inner.pos = self.effective_prompt_pos;
        Ok(self.cached_tokens)
    }

    fn remember_tool_replay(&mut self, calls: &[ToolCall], raw_dsml: &str) {
        self.events.push("remember-tool");
        self.remembered_tools.push((
            calls.iter().map(|call| call.id.clone()).collect(),
            raw_dsml.to_string(),
        ));
    }

    fn eval(&mut self, token: i32) -> Result<(), GenerateError> {
        self.events.push("eval");
        self.inner.eval(token)
    }

    fn eval_spec(
        &mut self,
        first: i32,
        budget: i32,
        sampler: ds4_core::Sampler,
        _rng: &mut u64,
        allow: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<Vec<i32>, GenerateError> {
        self.prefix_budgets.push(budget);
        if sampler.temperature > 0.0 {
            self.spec_samplers.push(sampler);
        }
        if self.prefix_width == 0 {
            return Ok(Vec::new());
        }
        self.eval(first)?;
        let mut tokens = vec![first];
        let mut rng = 1;
        while tokens.len() < self.prefix_width.min(budget as usize) {
            let last = *tokens.last().unwrap();
            if self.token_is_stop(last) || !allow(&self.token_text(last)?) {
                break;
            }
            let token = self.inner.sample(0.0, 0, 1.0, 0.0, &mut rng);
            self.inner.eval(token)?;
            tokens.push(token);
        }
        if self.model_id() == ds4_server::ModelSyntax::Glm53 as i32 {
            self.inner.live.extend_from_slice(&tokens);
        }
        self.prefix_lengths.push(tokens.len());
        Ok(tokens)
    }

    fn trim_greedy(&mut self, pos: i32) -> Result<(), GenerateError> {
        if self.model_id() != ds4_server::ModelSyntax::Glm53 as i32 {
            self.invalidate();
            return Ok(());
        }
        assert!(
            (1..=self.inner.pos).contains(&pos),
            "invalid journal target"
        );
        self.trimmed_positions.push(pos);
        self.inner.live.truncate(pos as usize);
        self.inner.pos = pos;
        Ok(())
    }

    fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        self.events.push("sample");
        self.sample_params.push((temperature, top_k, top_p, min_p));
        self.inner.sample(temperature, top_k, top_p, min_p, rng)
    }

    fn pos(&self) -> i32 {
        self.inner.pos()
    }

    fn ctx(&self) -> i32 {
        self.inner.ctx()
    }

    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    fn session_tokens(&self) -> Vec<i32> {
        self.inner.session_tokens()
    }

    fn maybe_store_continued(&mut self) -> Result<(), GenerateError> {
        self.events.push("continued");
        self.continued_positions.push(self.pos());
        if self.fail_continued {
            Err(GenerateError::Engine(
                "injected continued save failure".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn remember_thinking_visible_checkpoint(&mut self, text: Vec<u8>) {
        self.remembered.push((text, self.pos()));
    }

    fn invalidate(&mut self) {
        self.invalidations += 1;
        self.inner.live.clear();
        self.inner.pos = 0;
    }
}

#[test]
fn mtp_prefix_limits() {
    for (cap, ctx) in [(2, 8192), (8, 3)] {
        let mut parsed = user_req();
        parsed.max_tokens = cap;
        let mut script = ScriptedDecode::from_pieces(&[b"<|content_text|>Hello", b" world", b"!"]);
        script.ctx = ctx;
        script.model_id = 9;
        let mut engine = PromptSyncDecode::new(script, 0, 1);
        engine.prefix_width = 9;
        let mut out = Vec::new();
        let result = generate_and_write(
            &mut engine,
            &parsed,
            "mtp-limit",
            CREATED_TEST,
            false,
            cap,
            &mut out,
        )
        .unwrap();
        assert_eq!(result.timings.decode_tokens, 2);
        assert_eq!(result.timings.decode_steps, 1);
        assert_eq!(engine.prefix_budgets, [2]);
        assert_eq!(engine.pos(), 3);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Hello world"), "{text}");
    }
}

// A sampling request drafts too: its prefix verifies against the request's
// own sampler, as a greedy one does against argmax.
#[test]
fn mtp_prefix_sampled() {
    let mut parsed = user_req();
    parsed.temperature = 1.0;
    parsed.top_p = 0.95;
    let mut engine = PromptSyncDecode::new(
        ScriptedDecode::from_pieces(&[b"<|content_text|>Hello", b" world", b"!"]),
        0,
        1,
    );
    engine.inner.model_id = 9;
    engine.prefix_width = 9;
    let mut out = Vec::new();
    let result = generate_and_write(
        &mut engine,
        &parsed,
        "mtp-sampled",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("Hello world!"), "{text}");
    assert_eq!(result.timings.decode_steps, 1);
    assert_eq!(engine.prefix_budgets, [8]);
    assert_eq!(
        engine.spec_samplers,
        [ds4_core::Sampler {
            temperature: 1.0,
            top_k: parsed.top_k,
            top_p: 0.95,
            min_p: parsed.min_p,
        }]
    );
}

#[test]
fn mtp_prefix_stops() {
    for stream in [false, true] {
        for stop in [None, Some(" world")] {
            let mut parsed = user_req();
            parsed.stream = stream;
            if let Some(stop) = stop {
                parsed.stops.push(stop.into());
            }
            let mut script =
                ScriptedDecode::from_pieces(&[b"<|content_text|>Hello", b" world", b"!"]);
            script.model_id = 9;
            let mut engine = PromptSyncDecode::new(script, 0, 1);
            engine.prefix_width = 9;
            let mut out = Vec::new();
            let result = generate_and_write(
                &mut engine,
                &parsed,
                "mtp-stop",
                CREATED_TEST,
                false,
                8,
                &mut out,
            )
            .unwrap();
            let text = String::from_utf8(out).unwrap();
            assert_eq!(engine.prefix_budgets, [8]);
            assert!(text.contains("Hello"), "{text}");
            assert!(text.contains("\"finish_reason\":\"stop\""), "{text}");
            assert_eq!(
                result.timings.decode_tokens,
                if stop.is_some() { 2 } else { 3 }
            );
            assert!(!text.contains(" world") || stop.is_none(), "{text}");
            if stop.is_some() {
                assert_eq!(engine.invalidations, 1);
                assert_eq!(engine.pos(), 0);
            }
        }
    }
}

#[test]
fn mtp_prefix_empty() {
    let mut engine = PromptSyncDecode::new(ScriptedDecode::from_pieces(&[b"Hello"]), 0, 1);
    engine.prefix_width = 0;
    let result = generate_and_write(
        &mut engine,
        &user_req(),
        "mtp-empty",
        CREATED_TEST,
        false,
        8,
        &mut Vec::new(),
    );
    assert!(matches!(result, Err(GenerateError::Engine(_))));
    assert_eq!(engine.invalidations, 1);
}

fn glm_tool_prefix(surface: WireSurface, stream: bool, close: usize) {
    const WIDTH: usize = 4;
    const TAIL: &[u8] = b"UNEMITTED_SPECULATIVE_TAIL";
    const BODY: &[u8] = b"<tool_call>bash<arg_key>command</arg_key><arg_value>echo hi</arg_value>";
    let body = match surface {
        WireSurface::Anthropic => {
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"tools":[{"name":"bash","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}]}"#
        }
        WireSurface::Responses => {
            r#"{"input":"hi","max_output_tokens":8,"tools":[{"type":"function","name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}]}"#
        }
        _ => unreachable!(),
    };
    let mut parsed = parse_request(surface, &env(), body).unwrap();
    parsed.stream = stream;
    parsed.think_mode = ThinkMode::None;
    parsed.temperature = 0.0;

    // The close tag ends the speculative prefix before commit: its tail is
    // never drafted, so nothing is discarded.
    let mut pieces = vec![TAIL; WIDTH];
    if close == 0 {
        pieces[0] =
            b"<tool_call>bash<arg_key>command</arg_key><arg_value>echo hi</arg_value></tool_call>";
    } else {
        pieces[0] = BODY;
        pieces[1..close].fill(b"\n");
        pieces[close] = b"</tool_call>";
    }
    let mut tape = ScriptedDecode::from_pieces(&pieces);
    tape.model_id = ds4_server::ModelSyntax::Glm53 as i32;
    let mut engine = PromptSyncDecode::new(tape, 0, 1);
    engine.prefix_width = WIDTH;
    let mut out = Vec::new();
    let result = generate_and_write(
        &mut engine,
        &parsed,
        "glm-tool-prefix",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();

    let wire = String::from_utf8(out).unwrap();
    assert_eq!(result.finish, "tool_calls", "{wire}");
    assert_eq!(result.tool_ids.len(), 1, "{wire}");
    assert!(wire.contains(&result.tool_ids[0]), "{wire}");
    assert!(!wire.contains(std::str::from_utf8(TAIL).unwrap()), "{wire}");
    assert_eq!(engine.prefix_budgets, [8]);
    assert_eq!(
        engine.inner.idx,
        close + 1,
        "rows past the close stay undrafted"
    );
    assert_eq!(result.timings.decode_tokens, (close + 1) as i32);
    assert_eq!(result.frontier, (close + 2) as i32);
    assert_eq!(engine.pos(), result.frontier);
    assert_eq!(result.generation, 1);
    assert_eq!(engine.invalidations, 0);
    let expected_history = std::iter::once(1)
        .chain(1..=(close + 1) as i32)
        .collect::<Vec<_>>();
    assert_eq!(engine.inner.live, expected_history);
    assert!(engine.trimmed_positions.is_empty());
}

macro_rules! glm_tool_prefix_case {
    ($name:ident, $surface:ident, $stream:expr, $close:expr) => {
        #[test]
        fn $name() {
            glm_tool_prefix(WireSurface::$surface, $stream, $close);
        }
    };
}

glm_tool_prefix_case!(glm_tool_prefix_an_b_first, Anthropic, false, 0);
glm_tool_prefix_case!(glm_tool_prefix_an_b_middle, Anthropic, false, 1);
glm_tool_prefix_case!(glm_tool_prefix_an_b_final, Anthropic, false, 3);
glm_tool_prefix_case!(glm_tool_prefix_an_s_first, Anthropic, true, 0);
glm_tool_prefix_case!(glm_tool_prefix_an_s_middle, Anthropic, true, 1);
glm_tool_prefix_case!(glm_tool_prefix_an_s_final, Anthropic, true, 3);
glm_tool_prefix_case!(glm_tool_prefix_re_b_first, Responses, false, 0);
glm_tool_prefix_case!(glm_tool_prefix_re_b_middle, Responses, false, 1);
glm_tool_prefix_case!(glm_tool_prefix_re_b_final, Responses, false, 3);
glm_tool_prefix_case!(glm_tool_prefix_re_s_first, Responses, true, 0);
glm_tool_prefix_case!(glm_tool_prefix_re_s_middle, Responses, true, 1);
glm_tool_prefix_case!(glm_tool_prefix_re_s_final, Responses, true, 3);

#[test]
fn mtp_tool_tail_undrafted() {
    let block = concat!(
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">",
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
    );
    let tape = ScriptedDecode::from_pieces(&[block.as_bytes(), b"UNEMITTED_TAIL"]);
    let mut engine = PromptSyncDecode::new(tape, 0, 1);
    engine.prefix_width = 2;
    let mut out = Vec::new();
    let result = generate_and_write(
        &mut engine,
        &tools_req(),
        "generic-tool-prefix",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();
    // The closed call ends the prefix before commit: the session keeps the
    // tool call as its frontier instead of an invalidated, drafted tail.
    assert_eq!(result.tool_ids.len(), 1);
    assert_eq!(result.frontier, 2);
    assert_eq!(engine.invalidations, 0);
    assert_eq!(engine.prefix_lengths, [1]);
    assert!(engine.trimmed_positions.is_empty());
    assert!(!String::from_utf8(out).unwrap().contains("UNEMITTED_TAIL"));
}

/// Structural tool syntax takes the argmax: a sampled prefix ends at the
/// call marker before commit, while a greedy one drafts through it.
#[test]
fn mtp_prefix_cuts_at_tool_syntax() {
    let pieces: [&[u8]; 5] = [
        b"Calling.",
        "<｜DSML｜tool_calls>".as_bytes(),
        "\n<｜DSML｜invoke name=\"bash\">\n".as_bytes(),
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls".as_bytes(),
        "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>".as_bytes(),
    ];
    for temperature in [0.0, 1.0] {
        let mut parsed = tools_req();
        parsed.temperature = temperature;
        let mut engine = PromptSyncDecode::new(ScriptedDecode::from_pieces(&pieces), 0, 1);
        engine.prefix_width = 9;
        let mut out = Vec::new();
        let result = generate_and_write(
            &mut engine,
            &parsed,
            "mtp-tool",
            CREATED_TEST,
            false,
            16,
            &mut out,
        )
        .unwrap();
        assert_eq!(result.finish, "tool_calls");
        assert_eq!(result.tool_ids.len(), 1);
        let first = engine.prefix_lengths[0];
        if temperature > 0.0 {
            assert_eq!(first, 2, "{:?}", engine.prefix_lengths);
        } else {
            assert_eq!(first, pieces.len(), "{:?}", engine.prefix_lengths);
        }
    }
}

#[test]
fn mtp_disconnect() {
    struct Disconnect;
    impl Write for Disconnect {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            if data.windows(5).any(|s| s == b"Hello") {
                return Err(std::io::ErrorKind::BrokenPipe.into());
            }
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut parsed = user_req();
    parsed.stream = true;
    let mut script = ScriptedDecode::from_pieces(&[b"<|content_text|>Hello", b" world"]);
    script.model_id = 9;
    let mut engine = PromptSyncDecode::new(script, 0, 1);
    engine.prefix_width = 9;
    let result = generate_and_write(
        &mut engine,
        &parsed,
        "mtp-disconnect",
        CREATED_TEST,
        false,
        8,
        &mut Disconnect,
    );
    assert!(matches!(result, Err(GenerateError::Io)));
    assert_eq!(engine.prefix_budgets, [8]);
    assert_eq!(engine.invalidations, 1);
    assert_eq!(engine.pos(), 0);
}

#[test]
fn mtp_sampling_policy() {
    // Every request speculates: against its sampler (thinking: the fixed
    // tuple), and with a forced control prefix that has not yet started.
    for mode in 0..3 {
        let mut parsed = user_req();
        let temperature = match mode {
            0 => {
                parsed.temperature = 0.7;
                0.7
            }
            1 => {
                parsed.think_mode = ThinkMode::Low;
                ds4_server::parse::DEFAULT_TEMPERATURE
            }
            _ => {
                parsed.has_tool_results = true;
                parsed.required_think_end_prefix = vec![7];
                0.0
            }
        };
        let mut script = ScriptedDecode::from_pieces(&[b"<|content_text|>Hello"]);
        script.model_id = 9;
        let mut engine = PromptSyncDecode::new(script, 0, 1);
        engine.prefix_width = 9;
        generate_and_write(
            &mut engine,
            &parsed,
            "mtp-policy",
            CREATED_TEST,
            false,
            8,
            &mut Vec::new(),
        )
        .unwrap();
        assert!(!engine.prefix_budgets.is_empty());
        assert!(engine.events.contains(&"eval"));
        assert!(engine
            .spec_samplers
            .iter()
            .all(|s| s.temperature == temperature));
        assert_eq!(engine.spec_samplers.is_empty(), temperature == 0.0);
    }
}

#[test]
fn iquest_thinking_sampling() {
    use ds4_core::chat_template::{RenderClock, Template};
    use ds4_server::parse::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P};
    let iquest = ds4_core::Variant::IQuestQ1 as i32;
    let glm = ds4_core::Variant::Glm53FlashVq as i32;
    for model_id in [iquest, glm, 0] {
        for temperature in [0.0, 0.4] {
            let parsed = parse_request(
                WireSurface::OpenaiChat,
                &env(),
                &format!(r#"{{"messages":[{{"role":"user","content":"2 + 2"}}],"reasoning_effort":"high","temperature":{temperature},"top_k":9,"top_p":0.9,"min_p":0.03}}"#),
            )
            .unwrap();
            let mut script = ScriptedDecode::from_pieces(&[b"Compute.", b"</think>", b"4"]);
            script.model_id = model_id;
            let mut engine = PromptSyncDecode::new(script, 0, 1);
            if model_id == iquest {
                engine.template = Some(Template::compile(
                    include_str!("../../../tests/fixtures/chat-template/models/iquest/chat_template.jinja"),
                    RenderClock::Fixed(0),
                ).unwrap());
            }
            engine.prefix_width = 2;
            let mut out = Vec::new();
            generate_and_write(
                &mut engine,
                &parsed,
                "thinking-sample",
                CREATED_TEST,
                false,
                16,
                &mut out,
            )
            .unwrap();
            let expected = if model_id == iquest || model_id == glm {
                (temperature, 9, 0.9, 0.03)
            } else {
                (DEFAULT_TEMPERATURE, 0, DEFAULT_TOP_P, DEFAULT_MIN_P)
            };
            assert!(!engine.sample_params.is_empty());
            assert!(
                engine
                    .sample_params
                    .iter()
                    .all(|&params| params == expected),
                "model {model_id}, requested {temperature}: {:?}",
                engine.sample_params
            );
            // Every thinking request reaches the speculative commit; its
            // drafts verify against the sampler it decodes with.
            assert!(!engine.prefix_budgets.is_empty());
            let (temperature, top_k, top_p, min_p) = expected;
            let sampled = ds4_core::Sampler {
                temperature,
                top_k,
                top_p,
                min_p,
            };
            assert!(
                engine.spec_samplers.iter().all(|&s| s == sampled),
                "model {model_id}: {:?}",
                engine.spec_samplers
            );
            assert_eq!(engine.spec_samplers.is_empty(), temperature == 0.0);
        }
    }
}

#[test]
fn stop_list_find_matches_c_order() {
    let stops = vec!["STOP".into(), "END".into()];
    assert_eq!(
        stop_list_find_from(&stops, b"hello STOP tail END", 0),
        Some((6, 4))
    );
}

#[test]
fn iquest_structured_tools_allow_dedicated_output_parser() {
    let model_id = ds4_core::Variant::IQuestQ1 as i32;
    let plain = user_req();
    assert_eq!(generation_blocked(&plain, model_id), None);
    let expected = None;
    let mut tools = plain.clone();
    tools.has_tools = true;
    assert_eq!(generation_blocked(&tools, model_id), expected);
    assert_eq!(generation_blocked(&tools, 0), None);
    let mut results = plain.clone();
    results.has_tool_results = true;
    assert_eq!(generation_blocked(&results, model_id), expected);
    let mut history = plain.clone();
    history.messages[0].calls.push(ToolCall::default());
    assert_eq!(generation_blocked(&history, model_id), expected);
}

#[test]
fn family_generate_allows_tools() {
    let parsed = user_req();
    assert_eq!(generation_blocked(&parsed, 3), None);
    assert_eq!(generation_blocked(&parsed, 2), None);
    let mut tools = parsed.clone();
    tools.has_tools = true;
    assert_eq!(generation_blocked(&tools, 0), None);
}

#[test]
fn glm_serial_image_expands_placeholder_before_sync() {
    let mut parsed = user_req();
    parsed.messages[0].parts = vec![ChatPart::Image(0)];
    parsed.images.push(RequestImage {
        mime: ImageMime::Png,
        data: Arc::from([1u8]),
    });
    let mut engine = ScriptedDecode::from_pieces(&[]);
    engine.model_id = 7;
    engine.prompt_tokens = vec![154830, 154854, 154831];
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-vision",
        CREATED_TEST,
        false,
        1,
        &mut out,
    )
    .unwrap();

    assert_eq!(engine.live.len(), 18);
    assert_eq!(engine.live[0], 154830);
    assert!(engine.live[1..17].iter().all(|&token| token == 154854));
    assert_eq!(engine.live[17], 154831);
}

#[test]
fn step_images_expand_complete_spans() {
    let mut parsed = user_req();
    parsed.messages[0].parts = vec![ChatPart::Image(0), ChatPart::Image(1)];
    parsed.images = vec![
        RequestImage {
            mime: ImageMime::Png,
            data: Arc::from([1u8])
        };
        2
    ];
    let mut engine = ScriptedDecode::from_pieces(&[]);
    engine.model_id = 10;
    engine.prompt_tokens = vec![17, 128001, 18, 128001, 19];
    let mut engine = PromptSyncDecode::new(engine, 0, 0);
    engine.template = Some(
        ds4_core::chat_template::Template::compile(
            include_str!("../../../tests/fixtures/step37/chat_template.jinja"),
            ds4_core::chat_template::RenderClock::Fixed(0),
        )
        .unwrap(),
    );
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "step-images",
        CREATED_TEST,
        false,
        1,
        &mut out,
    )
    .unwrap();
    let span = [vec![128000], vec![128001; 169], vec![128002]].concat();
    assert_eq!(
        engine.inner.live,
        [vec![17], span.clone(), vec![18], span, vec![19]].concat()
    );
    engine.inner.prompt_tokens = vec![128001];
    assert!(generate_and_write(
        &mut engine,
        &parsed,
        "step-missing-image",
        CREATED_TEST,
        false,
        1,
        &mut out
    )
    .is_err());
    engine.inner.prompt_tokens = vec![128001; 3];
    assert!(generate_and_write(
        &mut engine,
        &parsed,
        "step-extra-image",
        CREATED_TEST,
        false,
        1,
        &mut out
    )
    .is_err());
}

#[test]
fn inkling_images_expand_in_order() {
    let mut parsed = user_req();
    parsed.messages[0].parts = vec![
        ChatPart::Image(0),
        ChatPart::Text("Then".into()),
        ChatPart::Image(1),
    ];
    parsed.images = vec![
        RequestImage {
            mime: ImageMime::Png,
            data: Arc::from([1u8])
        };
        2
    ];
    let mut engine = ScriptedDecode::from_pieces(&[]);
    engine.model_id = 9;
    engine.prompt_tokens = vec![200000, 200054, 200010, 200054, 200001];
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "inkling-images",
        CREATED_TEST,
        false,
        1,
        &mut out,
    )
    .unwrap();
    assert_eq!(
        engine.live,
        [
            vec![200000],
            vec![200054; 16],
            vec![200010],
            vec![200054; 16],
            vec![200001]
        ]
        .concat()
    );
    engine.prompt_tokens = vec![200054];
    assert!(generate_and_write(
        &mut engine,
        &parsed,
        "inkling-missing",
        CREATED_TEST,
        false,
        1,
        &mut out
    )
    .is_err());
    engine.prompt_tokens = vec![200054; 3];
    assert!(generate_and_write(
        &mut engine,
        &parsed,
        "inkling-extra",
        CREATED_TEST,
        false,
        1,
        &mut out
    )
    .is_err());
}

#[test]
fn inkling_audio_expands_in_order() {
    let body = r#"{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"format":"wav","data":"UklGRiYAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQIAAAAAAA=="}}]}]}"#;
    let mut parsed =
        ds4_server::parse_chat_request(&ds4_server::ParseEnv::default(), body).unwrap();
    let mut engine = ScriptedDecode::from_pieces(&[]);
    engine.model_id = 9;
    engine.prompt_tokens = vec![200000, 200053, 200043, 200001];
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "inkling-audio",
        CREATED_TEST,
        false,
        1,
        &mut out,
    )
    .unwrap();
    assert_eq!(engine.live, [200000, 200053, 200053, 200043, 200001]);
    for tokens in [vec![200000, 200001], vec![200053, 200053]] {
        engine.prompt_tokens = tokens;
        assert!(generate_and_write(
            &mut engine,
            &parsed,
            "missing-audio",
            CREATED_TEST,
            false,
            1,
            &mut out
        )
        .is_err());
    }
    parsed.images.push(RequestImage {
        mime: ImageMime::Png,
        data: Arc::from([1u8]),
    });
    parsed.messages[0].parts.push(ChatPart::Image(0));
    engine.prompt_tokens = vec![200053, 10, 200054, 200001];
    generate_and_write(
        &mut engine,
        &parsed,
        "mixed-audio",
        CREATED_TEST,
        false,
        1,
        &mut out,
    )
    .unwrap();
    assert_eq!(
        engine.live,
        [vec![200053; 2], vec![10], vec![200054; 16], vec![200001]].concat()
    );
}

#[test]
fn continued_store_is_best_effort_and_runs_before_sampling_without_final_catchup() {
    let mut parsed = user_req();
    parsed.max_tokens = 2;
    parsed.max_tokens_set = true;
    let inner = ScriptedDecode::from_pieces(&[b"a", b"b"]);
    let mut engine = PromptSyncDecode::new(inner, 0, 1);
    engine.fail_continued = true;
    let mut out = Vec::new();

    let outcome = generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-continued-order",
        CREATED_TEST,
        false,
        2,
        &mut out,
    )
    .unwrap();

    assert_eq!(outcome.finish, "length");
    assert_eq!(engine.continued_positions, [1, 1, 2]);
    assert_eq!(
        engine.events,
        [
            "sync",
            "continued",
            "continued",
            "sample",
            "eval",
            "continued",
            "sample",
            "eval",
        ]
    );
    assert_eq!(engine.pos(), 3);
}

#[test]
fn scripted_buffered_openai_has_text_and_stop() {
    let parsed = user_req();
    let mut engine =
        ScriptedDecode::from_pieces(&TAPE_PLAIN.iter().map(|s| s.as_bytes()).collect::<Vec<_>>());
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-1",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("Hello world."), "{s}");
    assert!(s.contains("\"finish_reason\":\"stop\""), "{s}");
    assert!(s.contains("\"object\":\"chat.completion\""), "{s}");
    assert!(
        s.contains("\"cache_write_tokens\":1"),
        "cold serial prompt should count as a KV write: {s}"
    );
    assert!(
        s.contains("\"timings\":{\"ttft_ms\":"),
        "serial path should emit timings: {s}"
    );
    assert!(s.contains("\"prefill_tokens\":1"), "{s}");
}

#[test]
fn scripted_responses_stream_activates_after_created() {
    let parsed = parse_request(
        WireSurface::Responses,
        &env(),
        r#"{"input":"hi","max_output_tokens":8,"stream":true}"#,
    )
    .unwrap();
    let mut engine = ScriptedDecode::from_pieces(
        &TAPE_PLAIN
            .iter()
            .map(|piece| piece.as_bytes())
            .collect::<Vec<_>>(),
    );
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "resp-serial-stream",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    let out = String::from_utf8(out).unwrap();
    assert!(out.contains("\"type\":\"response.created\""), "{out}");
    assert!(
        out.contains("\"type\":\"response.output_text.delta\""),
        "{out}"
    );
    assert!(out.contains("\"type\":\"response.completed\""), "{out}");
}

#[test]
fn prompt_sync_reports_buffered_cache_usage_from_effective_pos() {
    let parsed = user_req();
    let inner =
        ScriptedDecode::from_pieces(&TAPE_PLAIN.iter().map(|s| s.as_bytes()).collect::<Vec<_>>());
    let mut engine = PromptSyncDecode::new(inner, 4, 6);
    engine.prompt_sync_elapsed = Some(Duration::from_secs(2));
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-cache-buffered",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("\"prompt_tokens\":6"), "{s}");
    assert!(
        s.contains("\"cached_tokens\":4,\"cache_write_tokens\":2"),
        "cache writes must use effective engine pos minus cache reads: {s}"
    );
    assert!(s.contains("\"prefill_tokens\":2"), "{s}");
    assert!(s.contains("\"prefill_cached_tokens\":4"), "{s}");
    assert!(s.contains("\"prefill_tok_s\":1.0"), "{s}");
    assert_eq!(engine.prompt_sync_calls, 1);
    assert_eq!(engine.sync_calls, 0);
    assert_eq!(engine.disk_eligible, [true]);
    assert_eq!(engine.thinking_visible_eligible, [true]);
}

#[test]
fn prompt_sync_reports_streaming_cache_usage_from_effective_pos() {
    let mut parsed = user_req();
    parsed.stream = true;
    parsed.stream_include_usage = true;
    let inner =
        ScriptedDecode::from_pieces(&TAPE_PLAIN.iter().map(|s| s.as_bytes()).collect::<Vec<_>>());
    let mut engine = PromptSyncDecode::new(inner, 4, 6);
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-cache-stream",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("\"prompt_tokens\":6"), "{s}");
    assert!(
        s.contains("\"cached_tokens\":4,\"cache_write_tokens\":2"),
        "stream usage must use effective engine pos minus cache reads: {s}"
    );
    assert_eq!(engine.prompt_sync_calls, 1);
    assert_eq!(engine.sync_calls, 0);
    assert_eq!(engine.disk_eligible, [true]);
    assert_eq!(engine.thinking_visible_eligible, [true]);
}

#[test]
fn streaming_prompt_failure_is_an_sse_error_not_a_second_http_response() {
    let mut parsed = user_req();
    parsed.stream = true;
    let inner = ScriptedDecode::from_pieces(&[b"unused"]);
    let mut engine = PromptSyncDecode::new(inner, 0, 1);
    engine.fail_prompt_sync = true;
    let mut out = Vec::new();

    let error = generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-sync-fail",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap_err();

    assert!(matches!(error, GenerateError::Streamed(_)));
    let wire = String::from_utf8(out).unwrap();
    assert!(wire.starts_with("HTTP/1.1 200 OK\r\n"), "{wire}");
    assert_eq!(wire.matches("HTTP/1.1").count(), 1, "{wire}");
    assert!(wire.contains("event: error\ndata:"), "{wire}");
    assert!(wire.contains("injected prompt sync failure"), "{wire}");
}

#[test]
fn prompt_sync_receives_thinking_visible_surface_gate() {
    let cases = [
        (
            WireSurface::OpenaiChat,
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            true,
        ),
        (WireSurface::OpenaiCompletion, r#"{"prompt":"hi"}"#, false),
        (
            WireSurface::Anthropic,
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}"#,
            true,
        ),
        (WireSurface::Responses, r#"{"input":"hi"}"#, false),
    ];

    for (surface, body, expected) in cases {
        let parsed = parse_request(surface, &env(), body).unwrap();
        let inner = ScriptedDecode::from_pieces(&[b"ok"]);
        let mut engine = PromptSyncDecode::new(inner, 0, 1);
        let mut out = Vec::new();

        generate_and_write(
            &mut engine,
            &parsed,
            "visible-surface",
            CREATED_TEST,
            false,
            16,
            &mut out,
        )
        .unwrap();

        assert_eq!(engine.thinking_visible_eligible, [expected], "{surface:?}");
    }
}

#[test]
fn motif3_no_think_remembers_canonical_visible_checkpoint() {
    let parsed = user_req();
    let inner = ScriptedDecode {
        model_id: 3,
        ..ScriptedDecode::from_pieces(&[b"  Clear skies.  "])
    };
    let mut engine = PromptSyncDecode::new(inner, 0, 1);
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-visible",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    let mut expected = render_prompt(&parsed, 3).unwrap();
    assert!(expected.ends_with(b"<think></think>"));
    expected.truncate(expected.len() - b"<think></think>".len());
    expected.extend_from_slice(b"Clear skies.");
    assert_eq!(engine.remembered, [(expected, engine.pos())]);
    assert!(!engine.remembered[0].0.ends_with(b"<|endofturn|>"));

    let mut length = parsed;
    length.max_tokens = 1;
    length.max_tokens_set = true;
    let inner = ScriptedDecode {
        model_id: 3,
        ..ScriptedDecode::from_pieces(&[b"partial"])
    };
    let mut engine = PromptSyncDecode::new(inner, 0, 1);
    let mut out = Vec::new();
    let outcome = generate_and_write(
        &mut engine,
        &length,
        "chatcmpl-visible-length",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    assert_eq!(outcome.finish, "length");
    assert!(engine.remembered.is_empty());
}

#[test]
fn serial_thinking_answer_remembers_visible_history_for_all_keyable_formats() {
    for (model_id, close) in [
        (0, b"</think>".as_slice()),
        (2, b"<|think:end|>".as_slice()),
        (4, b"</think>".as_slice()),
        (6, b"</think>".as_slice()),
    ] {
        let mut parsed = user_req();
        parsed.think_mode = ThinkMode::Low;
        let pieces = [b"private plan".as_slice(), close, b"answer".as_slice()];
        let inner = ScriptedDecode {
            model_id,
            ..ScriptedDecode::from_pieces(&pieces)
        };
        let mut engine = PromptSyncDecode::new(inner, 0, 1);
        let mut out = Vec::new();

        generate_and_write(
            &mut engine,
            &parsed,
            "chatcmpl-thinking-visible",
            CREATED_TEST,
            false,
            16,
            &mut out,
        )
        .unwrap();

        let checkpoint = &engine
            .remembered
            .first()
            .unwrap_or_else(|| panic!("model {model_id} did not remember visible history"))
            .0;
        let mut future = parsed.clone();
        future.messages.push(ChatMsg {
            role: "assistant".into(),
            content: "answer".into(),
            ..ChatMsg::default()
        });
        future.messages.push(ChatMsg {
            role: "user".into(),
            content: "next".into(),
            ..ChatMsg::default()
        });
        let future_prompt = render_prompt(&future, model_id).unwrap();
        assert!(
            future_prompt.starts_with(checkpoint),
            "model {model_id} visible checkpoint is not a future-prompt prefix"
        );
    }
}

#[test]
fn motif3_no_think_invalidates_user_stop_and_tool_syntax_cut() {
    let cases = [
        (vec!["STOP".into()], b"Clear STOP tail".as_slice()),
        (Vec::new(), b"Clear <tool_call>".as_slice()),
    ];

    for (stops, piece) in cases {
        let mut parsed = user_req();
        parsed.stops = stops;
        let inner = ScriptedDecode {
            model_id: 3,
            ..ScriptedDecode::from_pieces(&[piece])
        };
        let mut engine = PromptSyncDecode::new(inner, 0, 1);
        let mut out = Vec::new();

        let outcome = generate_and_write(
            &mut engine,
            &parsed,
            "chatcmpl-visible-cut",
            CREATED_TEST,
            false,
            16,
            &mut out,
        )
        .unwrap();

        assert_eq!(outcome.finish, "stop");
        assert_eq!(engine.invalidations, 1);
        assert_eq!(engine.pos(), 0);
        assert!(engine.remembered.iter().all(|(_, frontier)| *frontier == 0));
    }
}

/// P1 gate: drive one reuse situation through the HTTP door and read the
/// trace back from `/v1/stats`, the way an operator would.
fn http_reuse_trace(
    reuse: ds4_core::ReuseTaken,
    miss: ds4_core::ReuseMiss,
    cached: i32,
) -> serde_json::Value {
    let mut cfg = ServerConfig::test_cfg();
    cfg.model_id = "ds4".into();
    cfg.model_name = "ds4".into();
    cfg.default_tokens = 16;
    let inner = Mutex::new(ServerInner::from_cfg(&cfg));

    let mut engine = PromptSyncDecode::new(ScriptedDecode::from_pieces(&[b"ok"]), cached, 1);
    engine.reuse = reuse;
    engine.miss = miss;
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let answer = one_shot_inner(&cfg, &inner, Some(&mut engine), request.as_bytes());
    assert!(
        answer.starts_with("HTTP/1.1 200 OK"),
        "generation failed: {answer}"
    );

    let stats = one_shot_inner(&cfg, &inner, None, b"GET /v1/stats HTTP/1.1\r\n\r\n");
    let start = stats.find("{\"routes\"").expect(&stats);
    serde_json::from_str(stats[start..].trim()).expect(&stats[start..])
}

fn one_shot_inner(
    cfg: &ServerConfig,
    inner: &Mutex<ServerInner>,
    engine: Option<&mut dyn DecodeIo>,
    request: &[u8],
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut out = Vec::new();
    thread::scope(|scope| {
        let client = scope.spawn(move || {
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(request).unwrap();
            let _ = c.shutdown(std::net::Shutdown::Write);
            let mut buf = Vec::new();
            c.read_to_end(&mut buf).unwrap();
            buf
        });
        let (mut server, _) = listener.accept().unwrap();
        handle_client_inner(cfg, inner, &mut server, engine, None);
        drop(server);
        out = client.join().unwrap();
    });
    String::from_utf8_lossy(&out).into_owned()
}

/// The four P1 situations, each read back from the HTTP surface. `fork`
/// only happens on the native bank lane, so it is covered by the
/// admission unit tests instead.
#[test]
fn http_reports_reuse_situations() {
    // Same chat, append: reuse at the frontier, suffix prefilled.
    let body = http_reuse_trace(ds4_core::ReuseTaken::Exact, ds4_core::ReuseMiss::None, 3);
    assert_eq!(body["last_request"]["reuse_kind"], "exact");
    assert!(
        body["last_request"].get("reuse_miss").is_none(),
        "a request that refused nothing carries no miss member: {body}"
    );

    // Edit or branch: a checkpoint below the prefix, gap replayed.
    let body = http_reuse_trace(ds4_core::ReuseTaken::Partial, ds4_core::ReuseMiss::None, 2);
    assert_eq!(body["last_request"]["reuse_kind"], "partial");

    // Restart with a template that re-rendered: refused, and it says why.
    let body = http_reuse_trace(
        ds4_core::ReuseTaken::Cold,
        ds4_core::ReuseMiss::RenderedPrefix,
        0,
    );
    assert_eq!(body["last_request"]["reuse_kind"], "cold");
    assert_eq!(
        body["last_request"]["reuse_miss"],
        "rendered prefix changed"
    );

    // Restart that found its payload.
    let body = http_reuse_trace(ds4_core::ReuseTaken::Exact, ds4_core::ReuseMiss::None, 5);
    assert_eq!(body["last_request"]["reuse_kind"], "exact");
    assert_eq!(body["last_request"]["effective_lane"], "serial");
}

#[test]
fn scripted_http_door_generates() {
    let mut cfg = ServerConfig::test_cfg();
    cfg.model_id = "ds4".into();
    cfg.model_name = "ds4".into();
    cfg.default_tokens = 16;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let h = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let mut engine = ScriptedDecode::from_pieces(&[b"ok"]);
        handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine), None);
    });
    let mut c = TcpStream::connect(addr).unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    c.write_all(req.as_bytes()).unwrap();
    let _ = c.shutdown(std::net::Shutdown::Write);
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    h.join().unwrap();
    let s = String::from_utf8_lossy(&out);
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("ok"), "{s}");
}

#[test]
fn scripted_motif_generates_over_http() {
    let cfg = ServerConfig::test_cfg();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let h = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let mut engine = ScriptedDecode {
            model_id: 3,
            ..ScriptedDecode::from_pieces(&[b"ok"])
        };
        handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine), None);
    });
    let mut c = TcpStream::connect(addr).unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    c.write_all(req.as_bytes()).unwrap();
    let _ = c.shutdown(std::net::Shutdown::Write);
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    h.join().unwrap();
    let s = String::from_utf8_lossy(&out);
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("ok"), "{s}");
}

#[test]
fn cont_stepper_buffered_matches_serial_shape() {
    let mut parsed = user_req();
    parsed.stream = false;
    let (mut st, head) = ContStepper::new(
        &parsed,
        0,
        "chatcmpl-1",
        CREATED_TEST,
        false,
        16,
        b"prompt".to_vec(),
        1,
        8192,
    );
    assert!(head.is_empty(), "buffered request must not stream a head");
    for p in TAPE_PLAIN {
        let step = st.feed(p.as_bytes());
        assert!(step.bytes.is_empty());
        assert!(!step.done);
    }
    let (bytes, outcome) = st.finalize(true, 0, 1, ReqTimings::default(), false);
    let s = String::from_utf8_lossy(&bytes);
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("Hello world."), "{s}");
    assert!(s.contains("\"finish_reason\":\"stop\""), "{s}");
    assert!(
        s.contains("\"cached_tokens\":0,\"cache_write_tokens\":1"),
        "engine split maps into the client frame: {s}"
    );
    assert_eq!(outcome.finish, "stop");
}

#[test]
fn cont_stepper_streams_and_stops_on_budget() {
    let mut parsed = user_req();
    parsed.stream = true;
    parsed.max_tokens = 2;
    parsed.max_tokens_set = true;
    let (mut st, head) = ContStepper::new(
        &parsed,
        0,
        "chatcmpl-2",
        CREATED_TEST,
        false,
        16,
        b"prompt".to_vec(),
        1,
        8192,
    );
    let h = String::from_utf8_lossy(&head);
    assert!(h.contains("text/event-stream"), "{h}");
    assert!(h.contains("chat.completion.chunk"), "{h}");
    let first = st.feed(TAPE_PLAIN[0].as_bytes());
    assert!(!first.done);
    let second = st.feed(TAPE_PLAIN[1].as_bytes());
    assert!(second.done, "host budget must stop the sequence");
    let (bytes, outcome) = st.finalize(false, 0, 1, ReqTimings::default(), false);
    let s = String::from_utf8_lossy(&bytes);
    assert!(s.contains("\"finish_reason\":\"length\""), "{s}");
    assert!(s.contains("data: [DONE]"), "{s}");
    assert_eq!(outcome.finish, "length");
}

fn think_req(surface: WireSurface, stream: bool) -> ParsedRequest {
    let body = match surface {
        WireSurface::Responses => r#"{"input":"hi","max_output_tokens":32}"#,
        _ => r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}"#,
    };
    let mut parsed = parse_request(surface, &env(), body).unwrap();
    parsed.stream = stream;
    parsed.think_mode = ThinkMode::Low;
    parsed.reasoning_summary_emit = true;
    parsed.temperature = 0.0;
    parsed.max_tokens = 32;
    parsed.max_tokens_set = true;
    parsed
}

fn step_think(
    parsed: &ParsedRequest,
    piece: &[u8],
    stop: bool,
    engine_eos: bool,
) -> (String, String) {
    let (mut st, head) = ContStepper::new(
        parsed,
        0,
        "job-think",
        CREATED_TEST,
        false,
        32,
        b"<think>".to_vec(),
        1,
        8192,
    );
    let step = st.feed(piece);
    if stop {
        st.mark_stop();
    }
    let (tail, outcome) = st.finalize(engine_eos, 0, 1, ReqTimings::default(), false);
    let mut bytes = head;
    bytes.extend(step.bytes);
    bytes.extend(tail);
    (String::from_utf8(bytes).unwrap(), outcome.finish)
}

#[test]
fn open_think_eos_reports_stop() {
    let parsed = think_req(WireSurface::OpenaiChat, true);
    let (s, finish) = step_think(&parsed, b"like `", true, true);
    assert_eq!(finish, "stop");
    assert!(s.contains("\"reasoning_content\":\"like `\""), "{s}");
    assert!(s.contains("\"finish_reason\":\"stop\""), "{s}");
    assert!(!s.contains("\"finish_reason\":\"length\""), "{s}");
}

#[test]
fn open_think_budget_stays_length() {
    let mut parsed = think_req(WireSurface::OpenaiChat, true);
    parsed.max_tokens = 1;
    let (s, finish) = step_think(&parsed, b"like `", false, false);
    assert_eq!(finish, "length");
    assert!(s.contains("\"finish_reason\":\"length\""), "{s}");
}

#[test]
fn closed_think_eos_reports_stop() {
    let parsed = think_req(WireSurface::OpenaiChat, true);
    let (s, finish) = step_think(&parsed, b"plan</think>Answer", true, true);
    assert_eq!(finish, "stop");
    assert!(s.contains("\"finish_reason\":\"stop\""), "{s}");
    assert!(s.contains("\"reasoning_content\":\"plan\""), "{s}");
    assert!(s.contains("\"content\":\"Answer\""), "{s}");
}

#[test]
fn open_think_eos_responses_keeps_completed() {
    let parsed = think_req(WireSurface::Responses, true);
    let (s, finish) = step_think(&parsed, b"like `", true, true);
    assert_eq!(finish, "stop");
    assert!(s.contains("\"type\":\"response.completed\""), "{s}");
    assert!(
        s.contains("\"type\":\"reasoning\",\"status\":\"incomplete\""),
        "{s}"
    );
    assert!(!s.contains("incomplete_details"), "{s}");
    assert!(!s.contains("max_output_tokens"), "{s}");
}

#[test]
fn open_think_eos_responses_buffered_item_stays_incomplete() {
    let parsed = think_req(WireSurface::Responses, false);
    let (s, finish) = step_think(&parsed, b"like `", true, true);
    assert_eq!(finish, "stop");
    assert!(s.contains("\"status\":\"completed\""), "{s}");
    assert!(
        s.contains("\"type\":\"reasoning\",\"status\":\"incomplete\""),
        "{s}"
    );
    assert!(!s.contains("incomplete_details"), "{s}");
}

#[test]
fn closed_think_eos_responses_item_completed() {
    let parsed = think_req(WireSurface::Responses, true);
    let (s, finish) = step_think(&parsed, b"plan</think>Answer", true, true);
    assert_eq!(finish, "stop");
    assert!(s.contains("\"type\":\"response.completed\""), "{s}");
    assert!(
        s.contains("\"type\":\"reasoning\",\"status\":\"completed\""),
        "{s}"
    );
}

#[test]
fn cont_stepper_streams_anthropic_events() {
    let parsed = parse_request(
        WireSurface::Anthropic,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true}"#,
    )
    .unwrap();
    let (mut stepper, head) = ContStepper::new(
        &parsed,
        0,
        "msg-cont-anthropic",
        CREATED_TEST,
        false,
        16,
        b"prompt".to_vec(),
        1,
        8192,
    );

    let head = String::from_utf8(head).unwrap();
    assert!(head.contains("event: message_start"), "{head}");
    let mut deltas = Vec::new();
    for piece in TAPE_PLAIN {
        deltas.extend(stepper.feed(piece.as_bytes()).bytes);
    }
    let (tail, outcome) = stepper.finalize(true, 0, 1, ReqTimings::default(), false);
    let deltas = String::from_utf8(deltas).unwrap();
    let tail = String::from_utf8(tail).unwrap();
    assert!(deltas.contains("event: content_block_delta"), "{deltas}");
    assert!(tail.contains("event: message_stop"), "{tail}");
    assert_eq!(outcome.finish, "stop");
}

#[test]
fn cont_stepper_buffers_anthropic_message() {
    let parsed = parse_request(
        WireSurface::Anthropic,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}"#,
    )
    .unwrap();
    let (mut stepper, head) = ContStepper::new(
        &parsed,
        0,
        "msg-cont-anthropic-buffered",
        CREATED_TEST,
        false,
        16,
        b"prompt".to_vec(),
        1,
        8192,
    );

    assert!(head.is_empty());
    for piece in TAPE_PLAIN {
        assert!(stepper.feed(piece.as_bytes()).bytes.is_empty());
    }
    let (body, outcome) = stepper.finalize(true, 0, 1, ReqTimings::default(), false);
    let body = String::from_utf8(body).unwrap();
    assert!(body.contains("\"type\":\"message\""), "{body}");
    assert!(body.contains("\"text\":\"Hello world.\""), "{body}");
    assert!(body.contains("\"stop_reason\":\"end_turn\""), "{body}");
    assert_eq!(outcome.finish, "stop");
}

#[test]
fn cont_stepper_streams_responses_events() {
    let parsed = parse_request(
        WireSurface::Responses,
        &env(),
        r#"{"input":"hi","max_output_tokens":8,"stream":true}"#,
    )
    .unwrap();
    let (mut stepper, head) = ContStepper::new(
        &parsed,
        0,
        "resp-cont-stream",
        CREATED_TEST,
        false,
        16,
        b"prompt".to_vec(),
        1,
        8192,
    );

    let head = String::from_utf8(head).unwrap();
    assert!(head.contains("\"type\":\"response.created\""), "{head}");
    let mut deltas = Vec::new();
    for piece in TAPE_PLAIN {
        deltas.extend(stepper.feed(piece.as_bytes()).bytes);
    }
    let (tail, outcome) = stepper.finalize(true, 0, 1, ReqTimings::default(), false);
    let deltas = String::from_utf8(deltas).unwrap();
    let tail = String::from_utf8(tail).unwrap();
    assert!(
        deltas.contains("\"type\":\"response.output_text.delta\""),
        "{deltas}"
    );
    assert!(tail.contains("\"type\":\"response.completed\""), "{tail}");
    assert!(tail.contains("resp_"), "{tail}");
    assert_eq!(outcome.finish, "stop");
}

#[test]
fn cont_stepper_buffers_responses_object() {
    let parsed = parse_request(
        WireSurface::Responses,
        &env(),
        r#"{"input":"hi","max_output_tokens":8}"#,
    )
    .unwrap();
    let (mut stepper, head) = ContStepper::new(
        &parsed,
        0,
        "resp-cont-buffered",
        CREATED_TEST,
        false,
        16,
        b"prompt".to_vec(),
        1,
        8192,
    );

    assert!(head.is_empty());
    for piece in TAPE_PLAIN {
        assert!(stepper.feed(piece.as_bytes()).bytes.is_empty());
    }
    let (body, outcome) = stepper.finalize(true, 0, 1, ReqTimings::default(), false);
    let body = String::from_utf8(body).unwrap();
    assert!(body.contains("\"object\":\"response\""), "{body}");
    assert!(body.contains("\"type\":\"output_text\""), "{body}");
    assert!(body.contains("Hello world."), "{body}");
    assert!(body.contains("resp_"), "{body}");
    assert_eq!(outcome.finish, "stop");
}

#[test]
fn responses_counts_tokens_generated_inside_reasoning_on_both_lanes() {
    let parsed = parse_request(
        WireSurface::Responses,
        &env(),
        r#"{"input":"hi","max_output_tokens":8,"reasoning":{"effort":"high","summary":"auto"}}"#,
    )
    .unwrap();
    let (mut stepper, _) = ContStepper::new(
        &parsed,
        0,
        "resp-cont-reasoning-usage",
        CREATED_TEST,
        false,
        16,
        b"<think>".to_vec(),
        1,
        8192,
    );

    stepper.feed(b"hidden");
    stepper.feed(b"</think>");
    stepper.feed(b"answer");
    let (body, _) = stepper.finalize(true, 0, 1, ReqTimings::default(), false);
    let body = String::from_utf8(body).unwrap();

    assert!(body.contains("\"reasoning_tokens\":2"), "{body}");

    let mut engine = ScriptedDecode::from_pieces(&[b"hidden", b"</think>", b"answer"]);
    let mut body = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "resp-serial-reasoning-usage",
        CREATED_TEST,
        false,
        16,
        &mut body,
    )
    .unwrap();
    let body = String::from_utf8(body).unwrap();
    assert!(body.contains("\"reasoning_tokens\":2"), "{body}");
}

#[test]
fn cont_stepper_stream_tool_id_matches_outcome() {
    let mut parsed = tools_req();
    parsed.stream = true;
    let (mut stepper, _) = ContStepper::new(
        &parsed,
        0,
        "chatcmpl-cont-tool",
        CREATED_TEST,
        false,
        64,
        b"prompt".to_vec(),
        1,
        8192,
    );
    let block = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"bash\">\n",
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
        "</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let streamed = stepper.feed(block.as_bytes());
    assert!(streamed.done);
    let (terminal, outcome) = stepper.finalize(false, 0, 1, ReqTimings::default(), false);
    assert_eq!(outcome.tool_ids.len(), 1);
    let id = &outcome.tool_ids[0];
    assert!(id.starts_with("call_"));
    assert_eq!(id.len(), 37);
    assert!(String::from_utf8_lossy(&streamed.bytes).contains(id));
    assert!(String::from_utf8_lossy(&terminal).contains("data: [DONE]"));
}

#[test]
fn bridge_null_oracle_ok() {
    let p = if let Ok(v) = std::env::var("DS4_BRIDGE_NULL_ORACLE") {
        PathBuf::from(v)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/bridge_null_oracle")
    };
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/bridge_null_oracle (missing {})",
        p.display()
    );
    let out = Command::new(&p).output().expect("run bridge_null_oracle");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"ok\n");
}

#[test]
fn scripted_dsml_tools_emit_tool_calls() {
    let mut parsed = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"},"max_tokens":16,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]}"#,
    )
    .unwrap();
    parsed.think_mode = ThinkMode::None;
    parsed.temperature = 0.0;
    assert!(parsed.has_tools);

    let block = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"bash\">\n",
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
        "</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let mut engine = ScriptedDecode::from_pieces(&[block.as_bytes()]);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-tools",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("\"finish_reason\":\"tool_calls\""), "{s}");
    assert!(s.contains("\"tool_calls\":["), "{s}");
    assert!(s.contains("\"name\":\"bash\""), "{s}");
    let id = s
        .split("\"tool_calls\":[{\"id\":\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    assert_eq!(id.len(), 37, "{s}");
    assert!(id.starts_with("call_"), "{s}");
    assert!(
        id[5..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{s}"
    );
    assert!(s.contains("ls"), "{s}");
}

#[test]
fn dsml_inside_closed_thinking_is_not_a_tool_call() {
    // The engine's rule (server_dsml_parse.c:291-316): with thinking on, a
    // DSML block inside the reasoning section is model text, never a tool
    // call -- the tool search starts after the last  response.  The route's
    // tolerant fallback must scope the same way, or the finish serves a
    // call the assistant never emitted as its post-thinking action.
    let mut parsed = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled"},"max_tokens":64,"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}"#,
    )
    .unwrap();
    parsed.think_mode = ThinkMode::Low;
    parsed.temperature = 0.0;
    assert!(parsed.has_tools);

    let think_open = think_start(ChatFormat::DeepSeek);
    let text = format!(
        "{think_open}the user wants the weather\n{}\n{} name=\"get_weather\">\n\
         {} name=\"city\" string=\"true\">Paris{}\n{}\n{}\n\n</think>\n\nHello.",
        DSML_TOOL_CALLS_START,
        DSML_INVOKE_START,
        DSML_PARAM_START,
        DSML_PARAM_END,
        DSML_INVOKE_END,
        DSML_TOOL_CALLS_END
    );
    // ScriptedV41: the route must take generate_terminal_v41, the only
    // path with the tolerant fallback (a plain scripted engine never
    // reaches it and the test would pass vacuously).
    let mut engine = ScriptedV41::new(&[(1, text.as_str()), (99, "")], 1 << 20);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-think-tools",
        CREATED_TEST,
        false,
        64,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(!s.contains("\"tool_calls\""), "{s}");
    assert!(s.contains("\"finish_reason\":\"stop\""), "{s}");
}

#[test]
fn tool_replay_restores_raw_dsml_before_render_and_uses_scoped_sync() {
    let parsed = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"call_saved","type":"function","function":{"name":"bash","arguments":"{\"a\":1,\"b\":2}"}}]},{"role":"tool","tool_call_id":"call_saved","content":"ok"},{"role":"assistant","content":"finished"},{"role":"user","content":"next"}],"thinking":{"type":"disabled"},"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}}}]}"#,
    )
    .unwrap();
    assert!(!parsed.has_tool_results);
    let canonical = render_prompt(&parsed, 0).unwrap();
    let raw = concat!(
        "\n\n<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"bash\">\n",
        "<｜DSML｜parameter name=\"b\" string=\"false\">2</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"a\" string=\"false\">1</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    assert!(!canonical
        .windows(raw.len())
        .any(|window| window == raw.as_bytes()));
    let inner = ScriptedDecode::from_pieces(&[b"done"]);
    let mut engine = PromptSyncDecode::new(inner, 1, 1);
    engine.replay_raw = Some(raw.into());
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-replay",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();

    assert_eq!(engine.replay_prompts.len(), 1);
    assert!(engine.replay_prompts[0]
        .windows(raw.len())
        .any(|window| window == raw.as_bytes()));
    let restore = engine
        .events
        .iter()
        .position(|event| *event == "restore")
        .unwrap();
    let sync = engine
        .events
        .iter()
        .position(|event| *event == "tool-sync")
        .unwrap();
    let sample = engine
        .events
        .iter()
        .position(|event| *event == "sample")
        .unwrap();
    assert!(restore < sync && sync < sample);
}

#[test]
fn tool_producer_remembers_final_wire_ids_with_sampled_dsml() {
    let parsed = tools_req();
    let raw = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"bash\">\n",
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
        "</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let inner = ScriptedDecode::from_pieces(&[raw.as_bytes()]);
    let mut engine = PromptSyncDecode::new(inner, 0, 1);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-producer",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    assert_eq!(engine.remembered_tools.len(), 1);
    let (ids, remembered) = &engine.remembered_tools[0];
    assert_eq!(ids.len(), 1);
    assert!(ids[0].starts_with("call_"));
    assert_eq!(remembered, raw);
    let remember = engine
        .events
        .iter()
        .position(|event| *event == "remember-tool")
        .unwrap();
    let sample = engine
        .events
        .iter()
        .position(|event| *event == "sample")
        .unwrap();
    assert!(sample < remember);
}

fn tools_req() -> ParsedRequest {
    let mut parsed = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"},"max_tokens":16,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]}"#,
    )
    .unwrap();
    parsed.think_mode = ThinkMode::None;
    parsed.temperature = 0.0;
    parsed
}

#[test]
fn minimax_string_wire() {
    use ds4_core::chat_template::{RenderClock, Template};
    let head = b"]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"bash\">]<]minimax[>[<command>";
    let tail = b"]<]minimax[>[</command>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>";
    let text = "print(\"한글\\path\")\n".repeat(100);
    let mut pieces: Vec<&[u8]> = vec![head];
    pieces.extend(std::iter::repeat_n(text.as_bytes(), 10));
    pieces.push(tail);
    let model = ds4_core::Variant::MinimaxVq as i32;
    for stream in [false, true] {
        let mut parsed = tools_req();
        parsed.stream = stream;
        parsed.max_tokens = 64;
        let mut tape = ScriptedDecode::from_pieces(&pieces);
        tape.model_id = model;
        let mut engine = PromptSyncDecode::new(tape, 0, 1);
        engine.template =
            Some(Template::compile("{{ messages | tojson }}", RenderClock::Fixed(0)).unwrap());
        let mut serial = Vec::new();
        let result = generate_and_write(
            &mut engine,
            &parsed,
            "minimax-test",
            CREATED_TEST,
            false,
            64,
            &mut serial,
        )
        .unwrap();
        assert_eq!(result.finish, "tool_calls");
        let (mut stepper, mut continuous) = ContStepper::new(
            &parsed,
            model,
            "minimax-test",
            CREATED_TEST,
            false,
            64,
            b"prompt".to_vec(),
            1,
            8192,
        );
        for piece in &pieces {
            continuous.extend(stepper.feed(piece).bytes);
        }
        let (tail, outcome) = stepper.finalize(true, 0, 1, ReqTimings::default(), false);
        continuous.extend(tail);
        assert_eq!(outcome.finish, "tool_calls");
        for bytes in [serial, continuous] {
            let wire = std::str::from_utf8(&bytes).unwrap();
            let mut args = String::new();
            if stream {
                let mut names = 0;
                let mut fragments = 0;
                for data in wire
                    .lines()
                    .filter_map(|s| s.strip_prefix("data: "))
                    .filter(|s| *s != "[DONE]")
                {
                    let chunk: serde_json::Value = serde_json::from_str(data).unwrap();
                    let call = &chunk["choices"][0]["delta"]["tool_calls"][0]["function"];
                    if call["name"].is_string() {
                        names += 1;
                    }
                    if let Some(s) = call["arguments"].as_str() {
                        args.push_str(s);
                        fragments += 1;
                    }
                }
                assert_eq!(names, 1);
                assert!(fragments >= 10);
            } else {
                let body = wire.split_once("\r\n\r\n").unwrap().1;
                let body: serde_json::Value = serde_json::from_str(body).unwrap();
                args = body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap()
                    .into();
            }
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&args).unwrap()["command"],
                text.repeat(10)
            );
            assert!(!wire.contains("]<]minimax[>["));
        }
    }
}

#[test]
fn minimax_tool_abort_wire() {
    use ds4_core::chat_template::{RenderClock, Template};
    let model = ds4_core::Variant::MinimaxVq as i32;
    for raw in [
        b"]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"bash\">]<]minimax[>[<command>unfinished".as_slice(),
        b"]<]minimax[>[<tool_call>]<]minimax[>[<invoke native pptx table".as_slice(),
    ] {
        for cap in [1, 16] {
            let mut parsed = tools_req();
            parsed.stream = true;
            parsed.max_tokens = cap;
            let mut tape = ScriptedDecode::from_pieces(&[raw]);
            tape.model_id = model;
            let mut engine = PromptSyncDecode::new(tape, 0, 1);
            engine.template = Some(Template::compile(
                "{{ messages | tojson }}", RenderClock::Fixed(0),
            ).unwrap());
            let mut serial = Vec::new();
            let result = generate_and_write(
                &mut engine, &parsed, "minimax-test", CREATED_TEST, false, 16, &mut serial,
            ).unwrap();
            let (mut stepper, mut continuous) = ContStepper::new(
                &parsed, model, "minimax-test", CREATED_TEST, false, 16,
                b"prompt".to_vec(), 1, 8192,
            );
            continuous.extend(stepper.feed(raw).bytes);
            let (tail, outcome) = stepper.finalize(cap != 1, 0, 1, ReqTimings::default(), false);
            continuous.extend(tail);
            let finish = if cap == 1 { "length" } else { "error" };
            assert_eq!(result.finish, finish);
            assert_eq!(outcome.finish, finish);
            for bytes in [serial, continuous] {
                let wire = std::str::from_utf8(&bytes).unwrap();
                let started = String::from_utf8_lossy(raw).contains("<invoke name=");
                assert_eq!(wire.contains("event: error"), cap != 1 && started);
                assert!(!wire.contains("]<]minimax[>["));
                assert!(!wire.contains("\"finish_reason\":\"stop\""));
            }
        }
    }
}

fn retrying_tool_decode() -> ScriptedDecode {
    let invalid = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let valid = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"bash\">\n",
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
        "</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    ScriptedDecode {
        steps: vec![
            ScriptedStep {
                token: 1,
                piece: invalid.as_bytes().to_vec(),
                stop: false,
            },
            ScriptedStep {
                token: 3,
                piece: valid.as_bytes().to_vec(),
                stop: false,
            },
            ScriptedStep {
                token: 4,
                piece: Vec::new(),
                stop: true,
            },
        ],
        suffix_tokens: vec![10, 11],
        ..ScriptedDecode::from_pieces(&[b"x"])
    }
}

#[test]
fn scripted_invalid_dsml_retries_and_emits_tool_calls() {
    let parsed = tools_req();
    let mut engine = retrying_tool_decode();
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-retry",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("\"finish_reason\":\"tool_calls\""), "{s}");
    assert!(s.contains("\"name\":\"bash\""), "{s}");
    assert!(s.contains("ls"), "{s}");
    assert!(
        engine.idx >= 2,
        "second decode pass should consume the valid call"
    );
}

#[test]
fn recovery_suffix_uses_sync_not_prompt_sync() {
    let parsed = tools_req();
    let mut engine = PromptSyncDecode::new(retrying_tool_decode(), 0, 1);
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-cache-retry",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    assert_eq!(
        engine.prompt_sync_calls, 1,
        "only the initial prompt uses the hook"
    );
    assert_eq!(
        engine.sync_calls, 1,
        "the recovery suffix uses ordinary sync"
    );
    assert_eq!(engine.disk_eligible, [false]);
    assert!(
        engine.inner.idx >= 2,
        "the retry must run a second decode pass"
    );
}

#[test]
fn scripted_motif_does_not_retry_invalid_tools() {
    let parsed = tools_req();
    let invalid = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let mut engine = ScriptedDecode {
        model_id: 3,
        steps: vec![
            ScriptedStep {
                token: 1,
                piece: invalid.as_bytes().to_vec(),
                stop: false,
            },
            ScriptedStep {
                token: 2,
                piece: Vec::new(),
                stop: true,
            },
            ScriptedStep {
                token: 3,
                piece: b"should-not-run".to_vec(),
                stop: false,
            },
        ],
        ..ScriptedDecode::from_pieces(&[b"x"])
    };
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-motif",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(!s.contains("should-not-run"), "{s}");
    assert!(s.contains("DSML"), "{s}");
    assert_eq!(engine.idx, 1, "Motif must not consume a second decode pass");
}

fn inkling_retry_case(invalid: &[u8], closure: &str) {
    let parsed = tools_req();
    let valid = br#"bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"ls"}}<|end_message|>"#;
    let mut tape = ScriptedDecode::from_pieces(&[invalid, valid]);
    tape.model_id = 9;
    tape.steps.insert(
        1,
        ScriptedStep {
            token: 98,
            piece: Vec::new(),
            stop: true,
        },
    );
    let mut engine = PromptSyncDecode::new(tape, 0, 1);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "inkling-retry",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let response = String::from_utf8(out).unwrap();
    assert_eq!(
        engine.sync_calls, 1,
        "exactly one corrective retry: {response}"
    );
    let rendered = engine.rendered.borrow();
    let suffix = String::from_utf8(rendered.last().unwrap().clone()).unwrap();
    assert!(
        suffix.starts_with(&format!(
            "{closure}<|message_tool|><|content_text|>Tool error: invalid Inkling tool call"
        )),
        "{suffix}"
    );
    assert!(
        suffix.ends_with("<|end_message|><|message_model|>"),
        "{suffix}"
    );
    assert!(!suffix.contains("DSML"), "{suffix}");
    assert!(
        response.contains("\"finish_reason\":\"tool_calls\""),
        "{response}"
    );
    assert!(response.contains("\"name\":\"bash\""), "{response}");
    assert!(!response.contains("Tool error:"), "{response}");
}

#[test]
fn inkling_unterminated_retry() {
    inkling_retry_case(
        br#"bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"ls"}"#,
        "<|end_message|><|content_model_end_sampling|>",
    );
}

#[test]
fn jinja_retry_renders_full_chat() {
    use ds4_core::chat_template::{RenderClock, Template};
    let parsed = tools_req();
    let bad = br#"bash<|content_invoke_tool_json|>{"name":"bash","args":[]}<|end_message|>"#;
    let good = br#"bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"ls"}}<|end_message|>"#;
    let mut tape = ScriptedDecode::from_pieces(&[bad, good]);
    tape.model_id = 9;
    tape.steps.insert(
        1,
        ScriptedStep {
            token: 98,
            piece: Vec::new(),
            stop: true,
        },
    );
    let mut engine = PromptSyncDecode::new(tape, 0, 1);
    engine.template = Some(
        Template::compile(
            include_str!(
                "../../../tests/fixtures/chat-template/models/inkling/chat_template.jinja"
            ),
            RenderClock::Fixed(0),
        )
        .unwrap(),
    );
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "jinja-retry",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let renders = engine.rendered.borrow();
    assert_eq!(renders.len(), 2);
    let retry = String::from_utf8_lossy(&renders[1]);
    assert!(retry.starts_with("<|message_system|>"), "{retry}");
    assert!(retry.contains("Tool error:"), "{retry}");
    assert!(
        retry.contains("hi\n\nTool error:"),
        "original request retained: {retry}"
    );
    assert!(
        !retry.contains("\"args\":[]"),
        "failed output is not committed: {retry}"
    );
    assert_eq!(engine.invalidations, 1);
    assert!(String::from_utf8(out)
        .unwrap()
        .contains("\"finish_reason\":\"tool_calls\""));
}

#[test]
fn inkling_malformed_retry() {
    inkling_retry_case(
        br#"bash<|content_invoke_tool_json|>{"name":"bash","args":[]}<|end_message|>"#,
        "<|content_model_end_sampling|>",
    );
}

#[test]
fn inkling_second_tool_retry() {
    inkling_retry_case(br#"bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"ls"}}<|end_message|><|message_model|>bash<|content_invoke_tool_json|>{"name":"bash","args":{}"#,
                       "<|end_message|><|content_model_end_sampling|>");
}

#[test]
fn inkling_bad_tool_terminal() {
    let bad = br#"bash<|content_invoke_tool_json|>{"name":"bash","args":[]}<|end_message|>"#;
    for stream in [false, true] {
        for cap in [1, 16] {
            let mut parsed = tools_req();
            parsed.stream = stream;
            parsed.max_tokens = cap;
            let mut tape = ScriptedDecode::from_pieces(&[bad, bad]);
            tape.model_id = 9;
            tape.steps.insert(
                1,
                ScriptedStep {
                    token: 98,
                    piece: Vec::new(),
                    stop: true,
                },
            );
            let mut engine = PromptSyncDecode::new(tape, 0, 1);
            let mut out = Vec::new();
            generate_and_write(
                &mut engine,
                &parsed,
                "inkling-terminal",
                CREATED_TEST,
                false,
                16,
                &mut out,
            )
            .unwrap();
            let response = String::from_utf8(out).unwrap();
            let finish = if cap == 1 { "length" } else { "error" };
            assert!(
                response.contains(&format!("\"finish_reason\":\"{finish}\"")),
                "{response}"
            );
            assert!(!response.contains("<|"), "{response}");
            assert!(!response.contains("Tool error:"), "{response}");
            assert_eq!(engine.sync_calls, usize::from(!stream && cap != 1));
        }
    }
}

// ---- the V4.1 push route (P5.5) -------------------------------------------

/// A scripted V4.1 engine: `is_v41` + the push entry, nothing session-shaped
/// behind it.  The script is (token, text) pairs; `stop_id` ends generation
/// the way the native checks it, and every run records what the route handed
/// the engine (prompt tokens, budget).
struct ScriptedV41 {
    script: Vec<(i32, Vec<u8>)>,
    stop_id: i32,
    ctx: i32,
    runs: std::cell::RefCell<Vec<(Vec<i32>, i32)>>,
    /// What the route handed `remember_tool_replay`: (wire ids, sampled bytes).
    remembered: std::cell::RefCell<Vec<(Vec<String>, String)>>,
}

impl ScriptedV41 {
    fn new(script: &[(i32, &str)], ctx: i32) -> Self {
        Self {
            script: script
                .iter()
                .map(|(id, s)| (*id, s.as_bytes().to_vec()))
                .collect(),
            stop_id: 99,
            ctx,
            runs: std::cell::RefCell::new(Vec::new()),
            remembered: std::cell::RefCell::new(Vec::new()),
        }
    }
}

impl DecodeIo for ScriptedV41 {
    fn model_id(&self) -> i32 {
        16 // Variant::DeepSeek41Flash
    }

    fn is_v41(&self) -> bool {
        true
    }

    /// The real engine writes the sampled block here (`NativeDecode`,
    /// generate.rs:4525, a port of `tool_memory_remember` at
    /// ds4_server.c:14524). The fake records what the route handed it, so the
    /// write is observable without a model.
    fn remember_tool_replay(&mut self, calls: &[ToolCall], raw_dsml: &str) {
        self.remembered.borrow_mut().push((
            calls.iter().map(|call| call.id.clone()).collect(),
            raw_dsml.to_string(),
        ));
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        Ok(text.bytes().map(i32::from).collect())
    }

    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        Ok(text.iter().map(|b| i32::from(*b)).collect())
    }

    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError> {
        Ok(self
            .script
            .iter()
            .find(|(id, _)| *id == token)
            .map(|(_, piece)| piece.clone())
            .unwrap_or_default())
    }

    fn token_is_stop(&self, token: i32) -> bool {
        token == self.stop_id
    }

    fn eos_id(&self) -> i32 {
        self.stop_id
    }

    fn v41_generate(
        &self,
        prompt: &[i32],
        n_predict: i32,
        _sampling: Option<ds4_core::V41Sampling>,
        _images: Option<&ds4_core::v41_media::V41Images>,
        _greedy: &mut dyn FnMut(*const i32, u32) -> bool,
        emit: &mut dyn FnMut(i32) -> bool,
        progress: &mut dyn FnMut(&str, i32, i32) -> bool,
    ) -> Result<(), GenerateError> {
        self.runs.borrow_mut().push((prompt.to_vec(), n_predict));
        // One prefill chunk: the route's keepalive path runs (a no-op
        // off-stream, the disconnect probe on it).
        if !progress("prefill_chunk", prompt.len() as i32, prompt.len() as i32) {
            return Ok(());
        }
        // The native loop's shape: `while (!stop && produced < n_predict)`.
        let mut produced = 0i32;
        for (id, _) in &self.script {
            if produced >= n_predict {
                break;
            }
            produced += 1;
            if !emit(*id) {
                break;
            }
        }
        Ok(())
    }

    fn sync(&mut self, _tokens: &[i32]) -> Result<(), GenerateError> {
        Ok(())
    }

    fn eval(&mut self, _token: i32) -> Result<(), GenerateError> {
        Ok(())
    }

    fn sample(&mut self, _t: f32, _k: i32, _p: f32, _m: f32, _rng: &mut u64) -> i32 {
        -1
    }

    fn pos(&self) -> i32 {
        0
    }

    fn ctx(&self) -> i32 {
        self.ctx
    }

    fn generation(&self) -> u64 {
        0
    }

    fn invalidate(&mut self) {}
}

fn v41_completion_req(body: &str) -> ParsedRequest {
    let mut r = parse_request(WireSurface::OpenaiCompletion, &env(), body).unwrap();
    r.temperature = 0.0;
    r
}

/// The ds41 route builds its own outcome and returns before the shared finish
/// path, so it has to write the sampled block itself, the way
/// `tool_memory_remember` does at ds4_server.c:14524. Before this the route
/// wrote nothing and the next turn's attach read an empty map, which is what
/// kept replay at zero on this family.
#[test]
fn v41_tool_turn_hands_the_sampled_block_to_the_tool_memory() {
    // The artifact's own spelling: the spaced wrapper with a plain invocation.
    let block = format!(
        "{DSML_TOOL_CALLS_START_SPACED}\n{DSML_INVOKE_START_SPACED} name=\"read_file\">\n\
         {DSML_PARAM_START} name=\"target_file\" string=\"true\">/tmp/x.txt{DSML_PARAM_END}\n\
         {DSML_INVOKE_END_SPACED}\n{DSML_TOOL_CALLS_END_SPACED}"
    );
    let body = r#"{"model":"t","messages":[{"role":"user","content":"Read the file /tmp/x.txt"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"target_file":{"type":"string"}},"required":["target_file"]}}}],"max_tokens":192,"temperature":0}"#;
    let parsed = parse_request(WireSurface::OpenaiChat, &env(), body).unwrap();
    let mut engine = ScriptedV41::new(&[(1, block.as_str()), (99, "")], 1 << 20);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "v41-mem",
        CREATED_TEST,
        false,
        192,
        &mut out,
    )
    .unwrap();

    let remembered = engine.remembered.borrow();
    assert_eq!(
        remembered.len(),
        1,
        "the ds41 route never wrote the sampled block"
    );
    assert_eq!(
        remembered[0].1, block,
        "the remembered bytes are not the sampled block"
    );
    assert_eq!(remembered[0].0.len(), 1, "{:?}", remembered[0].0);
    assert!(
        remembered[0].0[0].starts_with("call_"),
        "the map key must be the id that reached the wire: {:?}",
        remembered[0].0
    );
}

/// TP-17's signal half (CTR 8.7, NC-07).  A turn cut by the output budget
/// inside its tool block must report `length`: the block is unterminated, so
/// there is no executable call to serve, but the truncation itself is what
/// tells the agent loop its output was cut.  The reference keeps it by name --
/// `tool_parse_failure_recovery_finish` (ds4_server.c:6776-6784) returns
/// "length" when the incoming finish is "length" and "stop" otherwise, and
/// `test_incomplete_tool_call_keeps_stop_reason` (:19613) pins both arms.
/// This route pre-clobbered the pre-set `length` (generate.rs:3250) with
/// `error` before the parse-failure mapping could read it, and the wire remap
/// at :3389 then flattened that to `stop`.
#[test]
fn v41_budget_cut_inside_the_block_reports_length() {
    let body = r#"{"model":"t","messages":[{"role":"user","content":"Read the file /tmp/x.txt"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"target_file":{"type":"string"}},"required":["target_file"]}}}],"max_tokens":3,"temperature":0}"#;
    let parsed = parse_request(WireSurface::OpenaiChat, &env(), body).unwrap();
    assert!(parsed.has_tools);

    // Four pieces against a budget of three.  The engine's own loop stops at
    // n_predict, so the fourth never feeds and the turn ends with the block
    // open and its invocation unfinished -- the cut the budget sweep measures.
    let prose = String::from("Checking.");
    let open =
        format!(" {DSML_TOOL_CALLS_START_SPACED}\n{DSML_INVOKE_START_SPACED} name=\"read_file\">");
    let param =
        format!("\n{DSML_PARAM_START_SPACED} name=\"target_file\" string=\"true\">/tmp/x.txt");
    let close =
        format!("{DSML_PARAM_END_SPACED}\n{DSML_INVOKE_END_SPACED}\n{DSML_TOOL_CALLS_END_SPACED}");
    let script: Vec<(i32, &str)> = vec![
        (1, prose.as_str()),
        (2, open.as_str()),
        (3, param.as_str()),
        (4, close.as_str()),
    ];
    let mut engine = ScriptedV41::new(&script, 1 << 20);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-cut",
        CREATED_TEST,
        false,
        3,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();

    assert!(s.contains("\"finish_reason\":\"length\""), "{s}");
    assert!(!s.contains("\"finish_reason\":\"stop\""), "{s}");
    // No block closed, so no call is executable: the raw text is the content.
    assert!(!s.contains("\"tool_calls\""), "{s}");
    assert!(s.contains(DSML_INVOKE_START_SPACED), "{s}");
}

/// The other arm of the same reference function: an unterminated block that
/// the model ended itself is a normal assistant stop, and the wire remap that
/// clause 2.3.3 relies on still turns the route's `error` into `stop`.  This
/// guards the reclassification above from widening that remap.
#[test]
fn v41_unterminated_block_on_eos_still_reports_stop() {
    let body = r#"{"model":"t","messages":[{"role":"user","content":"Read the file /tmp/x.txt"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"target_file":{"type":"string"}},"required":["target_file"]}}}],"max_tokens":64,"temperature":0}"#;
    let parsed = parse_request(WireSurface::OpenaiChat, &env(), body).unwrap();

    let open = format!(
        "Checking. {DSML_TOOL_CALLS_START_SPACED}\n{DSML_INVOKE_START_SPACED} name=\"read_file\">"
    );
    // Same content as the budget arm, stopped one piece earlier: the model
    // ends the turn itself with the parameter still open.
    let param =
        format!("\n{DSML_PARAM_START_SPACED} name=\"target_file\" string=\"true\">/tmp/x.txt");
    let script: Vec<(i32, &str)> = vec![(1, open.as_str()), (2, param.as_str()), (99, "")];
    let mut engine = ScriptedV41::new(&script, 1 << 20);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-cut-eos",
        CREATED_TEST,
        false,
        64,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();

    assert!(s.contains("\"finish_reason\":\"stop\""), "{s}");
    assert!(!s.contains("\"finish_reason\":\"length\""), "{s}");
    assert!(!s.contains("\"tool_calls\""), "{s}");
}

#[test]
fn streamed_tool_failure_outcome() {
    let mut parsed = tools_req();
    parsed.stream = true;
    let raw =
        "<tool_calls><invoke name=\"bash\"><parameter name=\"command\" string=\"true\">unfinished";
    let mut engine = ScriptedV41::new(&[(1, raw), (99, "")], 8192);
    let mut out = Vec::new();
    let outcome = generate_and_write(
        &mut engine,
        &parsed,
        "tool-error",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    assert!(String::from_utf8_lossy(&out).contains("event: error"));
    assert_eq!(
        outcome.finish, "error",
        "an API projection error is not a successful decode outcome"
    );
}

#[test]
fn v41_completion_pushes_the_script() {
    let parsed = v41_completion_req(r#"{"prompt":"hi","max_tokens":8}"#);
    let mut engine = ScriptedV41::new(&[(1, "Hello"), (2, " world"), (99, "")], 1 << 20);
    let mut out = Vec::new();
    let result = generate_and_write(
        &mut engine,
        &parsed,
        "v41-cmpl",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();
    let response = String::from_utf8(out).unwrap();
    assert!(response.contains("Hello world"), "{response}");
    assert!(
        response.contains("\"finish_reason\":\"stop\""),
        "{response}"
    );
    let runs = engine.runs.borrow();
    assert_eq!(runs.len(), 1);
    // The prompt reached the engine as the tokenized bytes, not the text.
    assert_eq!(runs[0].0, vec![i32::from(b'h'), i32::from(b'i')]);
    assert_eq!(runs[0].1, 8);
    assert_eq!(result.timings.decode_tokens, 2);
    assert_eq!(result.frontier, 4);
    assert!(!result.speculation_active);
}

#[test]
fn v41_stream_carries_deltas_and_done() {
    let parsed = v41_completion_req(r#"{"prompt":"hi","max_tokens":8,"stream":true}"#);
    assert!(parsed.stream);
    let mut engine = ScriptedV41::new(&[(1, "Hello"), (2, " world"), (99, "")], 1 << 20);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "v41-stream",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();
    let sse = String::from_utf8(out).unwrap();
    assert!(sse.contains("data: "), "{sse}");
    assert!(sse.contains("Hello"), "{sse}");
    assert!(sse.contains(" world"), "{sse}");
    assert!(sse.contains("\"finish_reason\":\"stop\""), "{sse}");
    assert!(sse.contains("[DONE]"), "{sse}");
}

#[test]
fn v41_budget_stops_at_max_tokens() {
    let parsed = v41_completion_req(r#"{"prompt":"hi","max_tokens":2}"#);
    let mut engine = ScriptedV41::new(&[(1, "a"), (2, "b"), (3, "c")], 1 << 20);
    let mut out = Vec::new();
    let result = generate_and_write(
        &mut engine,
        &parsed,
        "v41-budget",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();
    let response = String::from_utf8(out).unwrap();
    assert!(
        response.contains("\"finish_reason\":\"length\""),
        "{response}"
    );
    assert!(response.contains("\"ab\""), "{response}");
    assert!(!response.contains("abc"), "{response}");
    assert_eq!(result.timings.decode_tokens, 2);
}

#[test]
fn v41_budget_clamps_to_the_metadata_context() {
    // ctx 5, prompt 2 -> room 3; the request asks for 8.
    let parsed = v41_completion_req(r#"{"prompt":"hi","max_tokens":8}"#);
    let mut engine = ScriptedV41::new(&[(1, "a")], 5);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "v41-ctx",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();
    let runs = engine.runs.borrow();
    assert_eq!(runs[0].1, 3);
}

#[test]
fn v41_stop_string_truncates_and_stops() {
    let parsed = v41_completion_req(r#"{"prompt":"hi","max_tokens":8,"stop":" world"}"#);
    let mut engine = ScriptedV41::new(&[(1, "Hello"), (2, " world"), (3, "!")], 1 << 20);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "v41-stop",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap();
    let response = String::from_utf8(out).unwrap();
    assert!(response.contains("Hello"), "{response}");
    assert!(!response.contains(" world"), "{response}");
    assert!(
        response.contains("\"finish_reason\":\"stop\""),
        "{response}"
    );
}

#[test]
fn v41_empty_prompt_is_refused_before_the_engine() {
    let parsed = v41_completion_req(r#"{"prompt":""}"#);
    let mut engine = ScriptedV41::new(&[(1, "a")], 1 << 20);
    let mut out = Vec::new();
    let error = generate_and_write(
        &mut engine,
        &parsed,
        "v41-empty",
        CREATED_TEST,
        false,
        8,
        &mut out,
    )
    .unwrap_err();
    assert!(error.to_string().contains("empty prompt"), "{error}");
    assert!(engine.runs.borrow().is_empty());
}

/// The tracker half of P1.6, measured on a served turn: the recorded artifact
/// turn (open-grok session 01a14f5d, the fixtures beside `tests/dsml.rs`)
/// roots its calls on the bare tool name with the stray quote, so the start
/// scan found no marker and the whole block reached the client as content.
/// Here the turn is scripted in three pieces -- prose, block, prose -- and the
/// streamed body must carry the calls while carrying no markup at all.
#[test]
fn v41_live_tool_root_turn_streams_calls_without_markup() {
    let text = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ds41-dsml/live-01a14f5d-turn1.txt"),
    )
    .unwrap();

    // The request declares the three tools the turn calls; the start scan only
    // takes a root tag some tool declares.
    let body = r#"{"model":"t","stream":true,"messages":[{"role":"user","content":"open solidcraft"}],
      "tools":[
        {"type":"function","function":{"name":"memory_search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
        {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"target_file":{"type":"string"}},"required":["target_file"]}}},
        {"type":"function","function":{"name":"list_dir","parameters":{"type":"object","properties":{"target_directory":{"type":"string"}},"required":["target_directory"]}}}
      ],"max_tokens":512,"temperature":0}"#;
    let mut parsed = parse_request(WireSurface::OpenaiChat, &env(), body).unwrap();
    parsed.think_mode = ThinkMode::None;
    parsed.temperature = 0.0;

    // Split at the fixture's own boundaries -- the root tag and the wrapper
    // close -- without asking the scan under test, so the arm that turns the
    // scan off still feeds the writer the same three pieces.
    let root: &[u8] = b"<memory_search\">";
    let at = text
        .windows(root.len())
        .position(|w| w == root)
        .expect("the recorded turn keeps its quoted tool root");
    let close = DSML_TOOL_CALLS_END_SPACED.as_bytes();
    let end = at
        + text[at..]
            .windows(close.len())
            .position(|w| w == close)
            .expect("the recorded turn keeps its wrapper close")
        + close.len();

    let pieces: Vec<Vec<u8>> = vec![
        text[..at].to_vec(),
        text[at..end].to_vec(),
        text[end..].to_vec(),
    ];
    let script: Vec<(i32, String)> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| (i as i32 + 1, String::from_utf8_lossy(p).into_owned()))
        .chain(std::iter::once((99, String::new())))
        .collect();

    let mut engine = ScriptedV41::new(
        &script
            .iter()
            .map(|(i, s)| (*i, s.as_str()))
            .collect::<Vec<_>>(),
        1 << 20,
    );
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "v41-live-root",
        CREATED_TEST,
        true,
        512,
        &mut out,
    )
    .unwrap();
    let sse = String::from_utf8(out).unwrap();

    assert!(
        sse.contains("\"tool_calls\""),
        "the turn's calls must reach the client: {sse}"
    );
    assert!(
        sse.contains("memory_search") && sse.contains("read_file"),
        "the sampled calls must be named on the wire: {sse}"
    );
    for marker in [
        DSML_TOOL_CALLS_START_SPACED,
        DSML_TOOL_CALLS_END_SPACED,
        DSML_INVOKE_START_SPACED,
        DSML_INVOKE_END_SPACED,
        "<memory_search",
        "<read_file",
    ] {
        assert!(
            !sse.contains(marker),
            "the block's markup reached the client as content: {marker}"
        );
    }
    assert!(
        sse.contains("finish_reason\":\"tool_calls\""),
        "a served call must finish as tool_calls: {sse}"
    );
}
