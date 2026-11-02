//! A standalone OpenAI-compatible host for the Gemma-4-26B-A4B VQ artifact.
//!
//! The CPU reference forward in `ds4_core::gemma4_a4b` evaluates the release
//! one token at a time; this binary puts the three OpenAI wire surfaces a
//! client needs in front of it: `POST /v1/chat/completions` (buffered and
//! streamed), `POST /v1/completions`, and `GET /v1/models`.
//!
//! It is deliberately separate from `ds4-server-rs`: that host drives the
//! native engine and this family has no device kernel yet. Nothing here
//! claims throughput; one request runs at a time (the forward already uses
//! every core for its matrix walks) and the process reports the wall time of
//! each step so the cost is visible.
//!
//! The reply parser follows the template that produced the prompt
//! (`/data/llama.cpp/models/templates/google-gemma-4-31B-it.jinja`): a
//! `<|channel>thought` .. `<channel|>` span is reasoning, the text after it is
//! the answer, and `<turn|>`/`<eos>` end the turn.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ds4_core::gemma4_a4b::{Decoder, Model, Tokenizer};
use serde_json::{json, Value};

const DEFAULT_BIND: &str = "127.0.0.1:8080";
const DEFAULT_NAME: &str = "gemma-4-26b-a4b-vq";
const MAX_BODY: usize = 8 * 1024 * 1024;

struct Args {
    bind: String,
    dir: String,
    tokenizer: String,
    name: String,
    ctx: usize,
    max_tokens: usize,
}

fn usage() -> &'static str {
    "gemma4-server-rs - an OpenAI-compatible host for a Gemma-4 VQ artifact

Usage: gemma4-server-rs --dir DIR --tokenizer FILE [options]

  --bind ADDR:PORT    listen address (default 127.0.0.1:8080)
  --dir DIR           artifact directory holding vq-release.json
  --tokenizer FILE    the checkpoint's tokenizer.json
  --model-dir DIR     checkpoint root (tokenizer.json is taken from it)
  --name NAME         model id reported by the API (default gemma-4-26b-a4b-vq)
  --ctx N             context limit for a request (default 8192)
  --max-tokens N      default completion length (default 128)
  --help              this text"
}

fn parse_args() -> Args {
    let mut bind = DEFAULT_BIND.to_string();
    let mut dir = None;
    let mut tokenizer = None;
    let mut model_dir = None;
    let mut name = DEFAULT_NAME.to_string();
    let mut ctx = 8192;
    let mut max_tokens = 128;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bind" => bind = args.next().unwrap_or(bind),
            "--dir" => dir = args.next(),
            "--tokenizer" => tokenizer = args.next(),
            "--model-dir" => model_dir = args.next(),
            "--name" => name = args.next().unwrap_or(name),
            "--ctx" => ctx = args.next().and_then(|v| v.parse().ok()).unwrap_or(ctx),
            "--max-tokens" => {
                max_tokens = args.next().and_then(|v| v.parse().ok()).unwrap_or(max_tokens)
            }
            "--help" | "-h" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => {
                eprintln!("unknown argument {other}\n\n{}", usage());
                process::exit(2);
            }
        }
    }
    let Some(dir) = dir else {
        eprintln!("--dir is required\n\n{}", usage());
        process::exit(2);
    };
    let tokenizer = tokenizer.or_else(|| model_dir.map(|root| format!("{root}/tokenizer.json")));
    let Some(tokenizer) = tokenizer else {
        eprintln!("--tokenizer or --model-dir is required\n\n{}", usage());
        process::exit(2);
    };
    Args {
        bind,
        dir,
        tokenizer,
        name,
        ctx,
        max_tokens,
    }
}

struct Shared {
    model: Model,
    tok: Tokenizer,
    name: String,
    ctx: usize,
    max_tokens: usize,
    channel_start: u32,
    channel_end: u32,
    /// One generation at a time: the forward already spends every core.
    lock: Mutex<()>,
}

struct Sampling {
    temperature: f32,
    top_p: f32,
    max_tokens: usize,
    stops: Vec<String>,
}

struct Reply {
    content: String,
    reasoning: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &'static str,
}

fn main() {
    let args = parse_args();
    eprintln!("opening {}", args.dir);
    let started = Instant::now();
    let model = match Model::open(std::path::Path::new(&args.dir)) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("open: {e}");
            process::exit(1);
        }
    };
    let tok = match Tokenizer::open(std::path::Path::new(&args.tokenizer)) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            process::exit(1);
        }
    };
    let channel_start = tok.special("<|channel>").unwrap_or(100);
    let channel_end = tok.special("<channel|>").unwrap_or(101);
    eprintln!(
        "opened in {:.1} s: {} layers, hidden {}, {} experts top {}, context {}",
        started.elapsed().as_secs_f64(),
        model.config.layers,
        model.config.hidden,
        model.config.experts,
        model.config.top_k,
        args.ctx
    );
    let shared = Arc::new(Shared {
        model,
        tok,
        name: args.name,
        ctx: args.ctx,
        max_tokens: args.max_tokens,
        channel_start,
        channel_end,
        lock: Mutex::new(()),
    });
    let listener = match TcpListener::bind(&args.bind) {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("bind {}: {e}", args.bind);
            process::exit(1);
        }
    };
    eprintln!("listening on http://{}", args.bind);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    if let Err(e) = serve_connection(shared, stream) {
                        eprintln!("connection: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
}

fn serve_connection(shared: Arc<Shared>, mut stream: TcpStream) -> std::io::Result<()> {
    let Some(request) = read_request(&mut stream)? else {
        return Ok(());
    };
    let path = request.1.split('?').next().unwrap_or("").to_string();
    match (request.0.as_str(), path.as_str()) {
        ("GET", "/health") => {
            let body = json!({"status": "ok", "model": shared.name}).to_string();
            write_response(&mut stream, 200, "application/json", body.as_bytes())
        }
        ("GET", "/v1/models") => {
            let body = json!({
                "object": "list",
                "data": [{
                    "id": shared.name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }],
            })
            .to_string();
            write_response(&mut stream, 200, "application/json", body.as_bytes())
        }
        ("POST", "/v1/chat/completions") => chat(&shared, &mut stream, &request.2),
        ("POST", "/v1/completions") => completions(&shared, &mut stream, &request.2),
        _ => write_error(&mut stream, 404, "unknown route"),
    }
}

type Request = (String, String, Vec<u8>);

fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(at) = find_head_end(&buf) {
            break at;
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > MAX_BODY {
            return Ok(None);
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.lines();
    let Some(first) = lines.next() else {
        return Ok(None);
    };
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut length = 0usize;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = buf[head_end..].to_vec();
    while body.len() < length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(length);
    Ok(Some((method, path, body)))
}

fn find_head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|at| at + 4)
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn write_error(stream: &mut TcpStream, status: u16, message: &str) -> std::io::Result<()> {
    let body = json!({"error": {"message": message, "type": "invalid_request_error"}}).to_string();
    write_response(stream, status, "application/json", body.as_bytes())
}

fn write_chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    stream.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
    stream.write_all(data)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn write_sse_start(stream: &mut TcpStream) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

fn write_sse_end(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

fn created() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn id_of(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}-{nanos:x}")
}

fn request_value(body: &[u8]) -> Result<Value, String> {
    serde_json::from_slice(body).map_err(|e| format!("invalid JSON body: {e}"))
}

fn messages_of(value: &Value) -> Result<Vec<(String, String)>, String> {
    let list = value
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "messages is required".to_string())?;
    let mut out = Vec::with_capacity(list.len());
    for (i, message) in list.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{i}].role is required"))?;
        let content = match message.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    if let Some(piece) = part.get("text").and_then(Value::as_str) {
                        text.push_str(piece);
                    }
                }
                text
            }
            Some(Value::Null) | None => String::new(),
            Some(_) => return Err(format!("messages[{i}].content is not text")),
        };
        out.push((role.to_string(), content));
    }
    Ok(out)
}

fn sampling_of(value: &Value, default_max: usize, ctx: usize, prompt: usize) -> Sampling {
    let max_tokens = value
        .get("max_completion_tokens")
        .or_else(|| value.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(default_max)
        .min(ctx.saturating_sub(prompt).max(1));
    let temperature = value
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    let top_p = value.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let stops = match value.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(list)) => list
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    Sampling {
        temperature,
        top_p,
        max_tokens,
        stops,
    }
}

fn chat(shared: &Shared, stream: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    let value = match request_value(body) {
        Ok(value) => value,
        Err(e) => return write_error(stream, 400, &e),
    };
    let messages = match messages_of(&value) {
        Ok(messages) => messages,
        Err(e) => return write_error(stream, 400, &e),
    };
    let prompt = match shared.tok.chat_prompt(&messages, false) {
        Ok(ids) => ids,
        Err(e) => return write_error(stream, 500, &e.to_string()),
    };
    let streamed = value.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let sampling = sampling_of(&value, shared.max_tokens, shared.ctx, prompt.len());
    if prompt.len() >= shared.ctx {
        return write_error(stream, 400, "prompt exceeds the context limit");
    }
    let id = id_of("chatcmpl");
    let created = created();
    let name = shared.name.clone();

    if streamed {
        write_sse_start(stream)?;
        let mut send = |delta: &str, reasoning: bool, first: bool| -> std::io::Result<()> {
            let mut piece = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": null,
                }],
            });
            if first {
                piece["choices"][0]["delta"] = json!({"role": "assistant", "content": ""});
            }
            if !delta.is_empty() {
                if reasoning {
                    piece["choices"][0]["delta"] = json!({"content": delta});
                    piece["choices"][0]["reasoning_content"] = json!(delta);
                } else {
                    piece["choices"][0]["delta"] = json!({"content": delta});
                }
            }
            write_chunk(stream, format!("data: {}\n\n", piece).as_bytes())
        };
        send("", false, true)?;
        let reply = match generate(shared, &prompt, &sampling, Some(&mut send)) {
            Ok(reply) => reply,
            Err(e) => {
                let _ = write_chunk(
                    stream,
                    format!(
                        "data: {}\n\n",
                        json!({"error": {"message": e, "type": "server_error"}})
                    )
                    .as_bytes(),
                );
                return write_sse_end(stream);
            }
        };
        let final_piece = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": reply.finish_reason,
            }],
            "usage": {
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            },
        });
        write_chunk(stream, format!("data: {}\n\n", final_piece).as_bytes())?;
        write_chunk(stream, b"data: [DONE]\n\n")?;
        return write_sse_end(stream);
    }

    let reply = match generate(shared, &prompt, &sampling, None) {
        Ok(reply) => reply,
        Err(e) => return write_error(stream, 500, &e),
    };
    let mut message = json!({"role": "assistant", "content": reply.content});
    if !reply.reasoning.trim().is_empty() {
        message["reasoning_content"] = json!(reply.reasoning);
    }
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": reply.finish_reason,
        }],
        "usage": {
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
        },
    })
    .to_string();
    write_response(stream, 200, "application/json", body.as_bytes())
}

fn completions(shared: &Shared, stream: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    let value = match request_value(body) {
        Ok(value) => value,
        Err(e) => return write_error(stream, 400, &e),
    };
    let Some(prompt_text) = value.get("prompt").and_then(Value::as_str) else {
        return write_error(stream, 400, "prompt is required");
    };
    let prompt = shared.tok.encode(prompt_text);
    let streamed = value.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let sampling = sampling_of(&value, shared.max_tokens, shared.ctx, prompt.len());
    if prompt.len() >= shared.ctx {
        return write_error(stream, 400, "prompt exceeds the context limit");
    }
    let id = id_of("cmpl");
    let created = created();
    let name = shared.name.clone();

    if streamed {
        write_sse_start(stream)?;
        let mut send = |delta: &str, _reasoning: bool, _first: bool| -> std::io::Result<()> {
            let piece = json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": name,
                "choices": [{
                    "index": 0,
                    "text": delta,
                    "logprobs": null,
                    "finish_reason": null,
                }],
            });
            write_chunk(stream, format!("data: {}\n\n", piece).as_bytes())
        };
        let reply = match generate(shared, &prompt, &sampling, Some(&mut send)) {
            Ok(reply) => reply,
            Err(e) => {
                let _ = write_chunk(
                    stream,
                    format!(
                        "data: {}\n\n",
                        json!({"error": {"message": e, "type": "server_error"}})
                    )
                    .as_bytes(),
                );
                return write_sse_end(stream);
            }
        };
        let piece = json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": name,
            "choices": [{
                "index": 0,
                "text": "",
                "logprobs": null,
                "finish_reason": reply.finish_reason,
            }],
        });
        write_chunk(stream, format!("data: {}\n\n", piece).as_bytes())?;
        write_chunk(stream, b"data: [DONE]\n\n")?;
        return write_sse_end(stream);
    }

    let reply = match generate(shared, &prompt, &sampling, None) {
        Ok(reply) => reply,
        Err(e) => return write_error(stream, 500, &e),
    };
    let body = json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": name,
        "choices": [{
            "index": 0,
            "text": reply.content,
            "logprobs": null,
            "finish_reason": reply.finish_reason,
        }],
        "usage": {
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
        },
    })
    .to_string();
    write_response(stream, 200, "application/json", body.as_bytes())
}

fn generate(
    shared: &Shared,
    prompt: &[u32],
    sampling: &Sampling,
    mut emit: Option<&mut dyn FnMut(&str, bool, bool) -> std::io::Result<()>>,
) -> Result<Reply, String> {
    let _guard = shared
        .lock
        .lock()
        .map_err(|_| "generation lock poisoned".to_string())?;
    let started = Instant::now();
    let mut session = shared.model.session();
    let mut logits = session
        .eval(prompt)
        .map_err(|e| format!("prompt: {e}"))?;
    let mut rng = seed();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut in_reasoning = false;
    let mut decoders = (Decoder::default(), Decoder::default());
    let mut finish = "length";
    let mut produced = 0usize;
    let mut first = true;
    for step in 0..sampling.max_tokens {
        let token = pick(&logits, sampling, &mut rng);
        if shared.tok.is_eos(token) {
            finish = "stop";
            break;
        }
        produced += 1;
        if token == shared.channel_start {
            in_reasoning = true;
            continue;
        }
        if token == shared.channel_end {
            in_reasoning = false;
            continue;
        }
        let target = if in_reasoning {
            &mut reasoning
        } else {
            &mut content
        };
        let before = target.len();
        let decoder = if in_reasoning {
            &mut decoders.1
        } else {
            &mut decoders.0
        };
        decoder.push(&shared.tok, token, target);
        let delta = target[before..].to_string();
        if let Some(emit) = emit.as_mut() {
            if !delta.is_empty() {
                emit(&delta, in_reasoning, first).map_err(|e| e.to_string())?;
                first = false;
            }
        }
        if !sampling.stops.is_empty() {
            if let Some(end) = sampling.stops.iter().filter_map(|s| target.find(s)).min() {
                target.truncate(end);
                finish = "stop";
                break;
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "token {step:>3} {:.2} s  {:.2} tok/s  {}",
            elapsed / produced as f64,
            produced as f64 / elapsed.max(1e-9),
            if in_reasoning { "thought" } else { "content" }
        );
        logits = session
            .step(token)
            .map_err(|e| format!("step {step}: {e}"))?;
    }
    decoders.0.finish(&mut content);
    decoders.1.finish(&mut reasoning);
    let reasoning = reasoning
        .strip_prefix("thought\n")
        .unwrap_or(&reasoning)
        .to_string();
    Ok(Reply {
        content,
        reasoning,
        prompt_tokens: prompt.len(),
        completion_tokens: produced,
        finish_reason: finish,
    })
}

fn seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15)
}

fn next_random(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// Greedy at temperature 0, otherwise temperature then nucleus over the
/// sorted distribution. This is sampling, not a claim about the model's
/// preferred settings.
fn pick(logits: &[f32], sampling: &Sampling, rng: &mut u64) -> u32 {
    if sampling.temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id as u32)
            .unwrap_or(0);
    }
    let temperature = sampling.temperature.max(1e-3);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(id, v)| (id as u32, ((v - max) / temperature).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    for (_, p) in &mut probs {
        *p /= sum;
    }
    probs.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut cumulative = 0.0f32;
    let mut cut = probs.len();
    for (i, (_, p)) in probs.iter().enumerate() {
        cumulative += *p;
        if cumulative >= sampling.top_p {
            cut = i + 1;
            break;
        }
    }
    let mass: f32 = probs[..cut].iter().map(|(_, p)| *p).sum();
    let mut draw = next_random(rng) * mass;
    for (id, p) in &probs[..cut] {
        draw -= *p;
        if draw <= 0.0 {
            return *id;
        }
    }
    probs[0].0
}
