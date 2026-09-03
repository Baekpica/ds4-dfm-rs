//! Scripted decode + HTTP generate path. No GGUF.

use ds4_server::parse::{parse_request, ChatMsg, ChatPart, ImageMime, RequestImage, ToolCall};
use ds4_server::route::WireSurface;
use ds4_server::{
    generate_and_write, generation_blocked, handle_client_inner, render_prompt,
    stop_list_find_from, ContStepper, DecodeIo, GenerateError, ParseEnv, ParsedRequest, ReqTimings,
    ScriptedDecode, ScriptedStep, ServerConfig, ServerInner, ThinkMode, CREATED_TEST, TAPE_PLAIN,
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

struct PromptSyncDecode {
    inner: ScriptedDecode,
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
    remembered_tools: Vec<(Vec<String>, String)>,
}

impl PromptSyncDecode {
    fn new(inner: ScriptedDecode, cached_tokens: i32, effective_prompt_pos: i32) -> Self {
        Self {
            inner,
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
            remembered_tools: Vec::new(),
        }
    }
}

impl DecodeIo for PromptSyncDecode {
    fn model_id(&self) -> i32 {
        self.inner.model_id()
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        self.inner.tokenize_text(text)
    }

    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
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

    fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        self.events.push("sample");
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
fn stop_list_find_matches_c_order() {
    let stops = vec!["STOP".into(), "END".into()];
    assert_eq!(
        stop_list_find_from(&stops, b"hello STOP tail END", 0),
        Some((6, 4))
    );
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
