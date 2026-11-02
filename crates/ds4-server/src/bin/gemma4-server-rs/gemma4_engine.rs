//! The Gemma-4-26B-A4B VQ engine for the ds4 server's OpenAI host.
//!
//! The wire surfaces, the sampler face and the request loop are the ds4
//! server's own (`ds4_server::accept_loop_with_engine` over `DecodeIo`), the
//! same seam every other family plugs into. This module only answers the two
//! questions that seam asks a family: how one token is evaluated, and how a
//! request becomes a prompt. Both come from this repository's own pieces:
//!
//! - the forward is the device chain in `native/gemma4/gemma4_gpu.cu`, the
//!   same step the gate checks against the CPU forward (logits relL2 3.1e-5,
//!   "Stockholm" on both sides); it is built here exactly as
//!   `examples/gemma4_a4b_gpu_gate.rs` builds it, chain by chain from the
//!   release's own layout;
//! - the prompt is `gemma4_a4b::Tokenizer::chat_prompt`, the Gemma-4 template
//!   the artifact ships (`docs/gemma4-a4b-*.md`), reached through the rendered
//!   bytes the server passes back to `tokenize_rendered_chat`.
//!
//! Nothing here invents a sampler, a stop rule or a detokenizer: `sample`
//! calls `ds4_core::sample::draw`, `token_text`/`token_is_stop` read the
//! checkpoint's tokenizer, and the reply is the server's `generate` path.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;

use ds4_core::gemma4_a4b::{
    bf16_round, decode, Entry, Family, LayerType, Role, Tokenizer,
};
use ds4_core::sample::draw;
use ds4_core::Sampler;
use ds4_server::generate::{DecodeIo, GenerateError};
use ds4_server::parse::ParsedRequest;

/// Mirrors `gm4_spec` in native/gemma4/gemma4_gpu.h.
#[repr(C)]
struct Spec {
    path: *const c_char,
    rows: u32,
    cols: u32,
}

/// Mirrors `gm4_account`.
#[repr(C)]
#[derive(Default)]
struct Account {
    files_bytes: u64,
    payload_bytes: u64,
    device_bytes: u64,
    device_free_before: u64,
    device_free_after: u64,
    tensors: i32,
    payloads: i32,
}

/// Mirrors `gm4_layer_spec`, field for field.
#[repr(C)]
struct LayerSpec {
    q: i32,
    k: i32,
    v: i32,
    o: i32,
    dense_gate: i32,
    dense_up: i32,
    dense_down: i32,
    router: i32,
    expert_gate_up: i32,
    expert_down: i32,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    hidden: i32,
    dense_ff: i32,
    moe_ff: i32,
    experts: i32,
    top_k: i32,
    threads: i32,
    eps: f32,
    layer_scalar: f32,
    input_norm: *const f32,
    post_attn_norm: *const f32,
    pre_ffn_norm: *const f32,
    post_ffn_norm: *const f32,
    pre_ffn_norm2: *const f32,
    post_ffn_norm1: *const f32,
    post_ffn_norm2: *const f32,
    q_norm: *const f32,
    k_norm: *const f32,
    router_scale: *const f32,
    per_expert_scale: *const f32,
    rope_inv: *const f32,
}

/// Mirrors `gm4_model_spec`, field for field.
#[repr(C)]
struct ModelSpec {
    layers: *const *mut c_void,
    kv: *const *mut c_void,
    n: i32,
    embed: i32,
    head: i32,
    final_norm: *const f32,
    hidden: i32,
    vocab: i32,
    threads: i32,
    eps: f32,
    embed_scale: f32,
    softcap: f32,
}

#[allow(dead_code)]
extern "C" {
    fn gm4_gpu_open(specs: *const Spec, n: i32, err: *mut c_char, errlen: usize) -> *mut c_void;
    fn gm4_gpu_close(g: *mut c_void);
    fn gm4_gpu_account_of(g: *const c_void, out: *mut Account);
    fn gm4_gpu_device_name(g: *const c_void) -> *const c_char;
    fn gm4_gpu_layer_new(
        g: *mut c_void,
        spec: *const LayerSpec,
        out: *mut *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_layer_free(l: *mut c_void);
    fn gm4_gpu_layer_bytes(l: *const c_void) -> usize;
    fn gm4_gpu_kv_new(
        g: *mut c_void,
        kv_heads: i32,
        head_dim: i32,
        window: i32,
        max_pos: i32,
        out: *mut *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_kv_free(kv: *mut c_void);
    fn gm4_gpu_kv_bytes(kv: *const c_void) -> usize;
    fn gm4_gpu_kv_reset(g: *mut c_void, kv: *mut c_void, err: *mut c_char, errlen: usize) -> i32;
    fn gm4_gpu_model_new(
        g: *mut c_void,
        spec: *const ModelSpec,
        out: *mut *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn gm4_gpu_model_free(m: *mut c_void);
    fn gm4_gpu_model_bytes(m: *const c_void) -> usize;
    fn gm4_gpu_model_step(
        g: *mut c_void,
        m: *mut c_void,
        token: i32,
        pos: i32,
        logits: *mut f32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
}

fn message(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

struct Chain(*mut c_void);
impl Drop for Chain {
    fn drop(&mut self) {
        unsafe { gm4_gpu_layer_free(self.0) }
    }
}

struct Cache(*mut c_void);
impl Drop for Cache {
    fn drop(&mut self) {
        unsafe { gm4_gpu_kv_free(self.0) }
    }
}

/// The artifact on the device: every encoded container resident, one chain and
/// one cache per layer, and the two ends (the tied table and the final norm).
/// Built exactly as the gate builds it, so the serving path runs the step the
/// gate has already checked against the CPU forward.
pub struct GpuModel {
    gpu: *mut c_void,
    model: *mut c_void,
    /// Declared before `gpu` is dropped (declaration order is drop order for
    /// the last field first; the explicit order is enforced by `Drop` below).
    chains: Vec<Chain>,
    caches: Vec<Cache>,
    logits: Vec<f32>,
    pos: usize,
    pub vocab: usize,
    pub layers: usize,
    pub hidden: usize,
}

impl Drop for GpuModel {
    fn drop(&mut self) {
        unsafe { gm4_gpu_model_free(self.model) };
        self.chains.clear();
        self.caches.clear();
        unsafe { gm4_gpu_close(self.gpu) };
    }
}

fn tensor_index(encoded: &[&Entry], name: &str) -> Result<i32, String> {
    encoded
        .iter()
        .position(|e| e.name == name)
        .map(|i| i as i32)
        .ok_or_else(|| format!("{name}: not an encoded entry"))
}

impl GpuModel {
    /// Open the release directory and put the whole artifact on the device.
    /// The cache bound is the CPU session's: the window for a sliding layer,
    /// and `cap` positions for a full one (`forward.rs:194-225`).
    pub fn open(dir: &Path, cap_full: usize) -> Result<Self, String> {
        let family = Family::open(dir).map_err(|e| e.to_string())?;
        let root = family.dir.clone();
        let encoded: Vec<&Entry> = family
            .release
            .entries()
            .iter()
            .filter(|e| e.encoded)
            .collect();
        if encoded.is_empty() {
            return Err("the release holds no encoded entries".into());
        }
        let mut names: Vec<CString> = Vec::with_capacity(encoded.len());
        let mut specs: Vec<Spec> = Vec::with_capacity(encoded.len());
        for entry in &encoded {
            let path = root.join(format!("{}.bin", entry.name));
            let name = CString::new(path.to_string_lossy().as_ref())
                .map_err(|_| format!("{}: path holds a NUL", path.display()))?;
            specs.push(Spec {
                path: name.as_ptr(),
                rows: entry.rows as u32,
                cols: entry.cols as u32,
            });
            names.push(name);
        }
        let mut err = [0i8; 512];
        let gpu = unsafe { gm4_gpu_open(specs.as_ptr(), specs.len() as i32, err.as_mut_ptr(), err.len()) };
        if gpu.is_null() {
            return Err(format!("open: {}", message(&err)));
        }
        let mut account = Account::default();
        unsafe { gm4_gpu_account_of(gpu, &mut account) };
        let device = unsafe { CStr::from_ptr(gm4_gpu_device_name(gpu)) }
            .to_string_lossy()
            .into_owned();
        eprintln!(
            "device {device:?}: {} tensors, {} payloads, {:.1} MiB resident",
            account.tensors,
            account.payloads,
            account.device_bytes as f64 / (1024.0 * 1024.0)
        );

        let config = &family.config;
        let embed = tensor_index(&encoded, &family.layout.embed.name)?;
        let final_norm = decode(&root, &family.layout.final_norm)
            .map_err(|e| e.to_string())?
            .data;

        let mut chains = Vec::with_capacity(config.layers);
        let mut caches = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let (chain, cache) = build_chain(gpu, &family, &root, &encoded, layer, cap_full)?;
            chains.push(chain);
            caches.push(cache);
        }
        let layer_ptrs: Vec<*mut c_void> = chains.iter().map(|c| c.0).collect();
        let kv_ptrs: Vec<*mut c_void> = caches.iter().map(|c| c.0).collect();
        let spec = ModelSpec {
            layers: layer_ptrs.as_ptr(),
            kv: kv_ptrs.as_ptr(),
            n: config.layers as i32,
            embed,
            head: embed,
            final_norm: final_norm.as_ptr(),
            hidden: config.hidden as i32,
            vocab: config.vocab as i32,
            threads: 256,
            eps: config.rms_eps,
            embed_scale: bf16_round((config.hidden as f32).sqrt()),
            softcap: config.logit_softcap,
        };
        let mut model: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { gm4_gpu_model_new(gpu, &spec, &mut model, err.as_mut_ptr(), err.len()) };
        if rc != 0 {
            unsafe { gm4_gpu_close(gpu) };
            return Err(format!("model_new: {}", message(&err)));
        }
        eprintln!(
            "model: {} layers, hidden {}, vocab {}, chain+cache {:.1} MiB",
            config.layers,
            config.hidden,
            config.vocab,
            (chains.iter().map(|c| unsafe { gm4_gpu_layer_bytes(c.0) }).sum::<usize>()
                + caches.iter().map(|c| unsafe { gm4_gpu_kv_bytes(c.0) }).sum::<usize>())
                as f64
                / (1024.0 * 1024.0)
        );
        Ok(Self {
            gpu,
            model,
            chains,
            caches,
            logits: vec![0f32; config.vocab],
            pos: 0,
            vocab: config.vocab,
            layers: config.layers,
            hidden: config.hidden,
        })
    }

    /// Back to position zero with empty caches: the start of a request.
    pub fn reset(&mut self) -> Result<(), String> {
        let mut err = [0i8; 512];
        for cache in &self.caches {
            if unsafe { gm4_gpu_kv_reset(self.gpu, cache.0, err.as_mut_ptr(), err.len()) } != 0 {
                return Err(format!("kv_reset: {}", message(&err)));
            }
        }
        self.pos = 0;
        Ok(())
    }

    /// One token at the current position; leaves the logits of the position
    /// after it, exactly as the CPU session's `step` does.
    pub fn step(&mut self, token: u32) -> Result<(), String> {
        let mut err = [0i8; 512];
        let rc = unsafe {
            gm4_gpu_model_step(
                self.gpu,
                self.model,
                token as i32,
                self.pos as i32,
                self.logits.as_mut_ptr(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(format!("model_step at {}: {}", self.pos, message(&err)));
        }
        self.pos += 1;
        Ok(())
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn pos(&self) -> usize {
        self.pos
    }
}

fn build_chain(
    gpu: *mut c_void,
    family: &Family,
    root: &Path,
    encoded: &[&Entry],
    layer: usize,
    cap_full: usize,
) -> Result<(Chain, Cache), String> {
    let config = &family.config;
    let spec = &family.layout.layers[layer];
    let kind = spec.kind;
    let head_dim = config.head_dim_of(kind);
    let kv_heads = config.kv_heads_of(kind);
    let window = match kind {
        LayerType::Sliding => config.sliding_window as i32,
        LayerType::Full => 0,
    };
    let cap = match kind {
        LayerType::Sliding => config.sliding_window,
        LayerType::Full => cap_full,
    };
    let tensor = |role: Role| -> Result<i32, String> {
        let entry = spec.get(role).map_err(|e| e.to_string())?;
        tensor_index(encoded, &entry.name)
    };
    let vector = |role: Role| -> Result<Vec<f32>, String> {
        Ok(decode(root, spec.get(role).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
            .data)
    };
    let v = match config.has_v_proj(kind) {
        true => tensor(Role::VProj)?,
        false => -1,
    };
    let scalar = decode(root, spec.get(Role::LayerScalar).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        .data[0];
    let rope_inv = config.rope_inv(kind);
    let input_norm = vector(Role::InputNorm)?;
    let post_attn_norm = vector(Role::PostAttentionNorm)?;
    let pre_ffn_norm = vector(Role::PreFfnNorm)?;
    let post_ffn_norm = vector(Role::PostFfnNorm)?;
    let pre_ffn_norm2 = vector(Role::PreFfnNorm2)?;
    let post_ffn_norm1 = vector(Role::PostFfnNorm1)?;
    let post_ffn_norm2 = vector(Role::PostFfnNorm2)?;
    let q_norm = vector(Role::QNorm)?;
    let k_norm = vector(Role::KNorm)?;
    let router_scale = vector(Role::RouterScale)?;
    let per_expert_scale = vector(Role::RouterPerExpertScale)?;
    let lspec = LayerSpec {
        q: tensor(Role::QProj)?,
        k: tensor(Role::KProj)?,
        v,
        o: tensor(Role::OProj)?,
        dense_gate: tensor(Role::DenseGate)?,
        dense_up: tensor(Role::DenseUp)?,
        dense_down: tensor(Role::DenseDown)?,
        router: tensor(Role::RouterProj)?,
        expert_gate_up: tensor(Role::ExpertGateUp)?,
        expert_down: tensor(Role::ExpertDown)?,
        heads: config.heads as i32,
        kv_heads: kv_heads as i32,
        head_dim: head_dim as i32,
        hidden: config.hidden as i32,
        dense_ff: config.dense_ff as i32,
        moe_ff: config.moe_ff as i32,
        experts: config.experts as i32,
        top_k: config.top_k as i32,
        threads: 256,
        eps: config.rms_eps,
        layer_scalar: scalar,
        input_norm: input_norm.as_ptr(),
        post_attn_norm: post_attn_norm.as_ptr(),
        pre_ffn_norm: pre_ffn_norm.as_ptr(),
        post_ffn_norm: post_ffn_norm.as_ptr(),
        pre_ffn_norm2: pre_ffn_norm2.as_ptr(),
        post_ffn_norm1: post_ffn_norm1.as_ptr(),
        post_ffn_norm2: post_ffn_norm2.as_ptr(),
        q_norm: q_norm.as_ptr(),
        k_norm: k_norm.as_ptr(),
        router_scale: router_scale.as_ptr(),
        per_expert_scale: per_expert_scale.as_ptr(),
        rope_inv: rope_inv.as_ptr(),
    };
    let mut err = [0i8; 512];
    let mut chain: *mut c_void = std::ptr::null_mut();
    if unsafe { gm4_gpu_layer_new(gpu, &lspec, &mut chain, err.as_mut_ptr(), err.len()) } != 0 {
        return Err(format!("layer_new {layer}: {}", message(&err)));
    }
    let mut cache: *mut c_void = std::ptr::null_mut();
    let rc = unsafe {
        gm4_gpu_kv_new(
            gpu,
            kv_heads as i32,
            head_dim as i32,
            window,
            cap as i32,
            &mut cache,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        unsafe { gm4_gpu_layer_free(chain) };
        return Err(format!("kv_new {layer}: {}", message(&err)));
    }
    Ok((Chain(chain), Cache(cache)))
}

/// The family engine the ds4 server drives. All wire and reply handling stays
/// on the server side; this is only the family's half of `DecodeIo`.
pub struct Gemma4Engine {
    pub model: GpuModel,
    tok: Tokenizer,
    ctx: i32,
    generation: u64,
    /// id -> special-token text, for `token_text` on the control tokens.
    specials: Vec<(i32, String)>,
    eos: i32,
    end_turn: i32,
}

impl Gemma4Engine {
    pub fn new(model: GpuModel, tok: Tokenizer, ctx: i32) -> Self {
        let mut specials = Vec::new();
        for name in ["<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>", "<eos>"] {
            if let Ok(id) = tok.special(name) {
                specials.push((id as i32, name.to_string()));
            }
        }
        let eos = tok.special("<eos>").unwrap_or(1) as i32;
        let end_turn = tok.special("<turn|>").map(|id| id as i32).unwrap_or(-1);
        Self {
            model,
            tok,
            ctx,
            generation: 0,
            specials,
            eos,
            end_turn,
        }
    }
}

fn generate_err(e: impl std::fmt::Display) -> GenerateError {
    GenerateError::Engine(format!("gemma4: {e}"))
}

impl DecodeIo for Gemma4Engine {
    fn model_id(&self) -> i32 {
        0
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        Ok(self.tok.encode(text).into_iter().map(|t| t as i32).collect())
    }

    /// The server rendered this request through `render_request` below, which
    /// emits the message list the template needs; the prompt ids are the
    /// checkpoint template's own, identical to `chat_prompt`'s.
    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        let value: serde_json::Value =
            serde_json::from_slice(text).map_err(|e| generate_err(format!("render: {e}")))?;
        let thinking = value
            .get("thinking")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let mut messages = Vec::new();
        for item in value
            .get("messages")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = item.get("content").and_then(|v| v.as_str()).unwrap_or("");
            messages.push((role.to_string(), content.to_string()));
        }
        self.tok
            .chat_prompt(&messages, thinking)
            .map(|ids| ids.into_iter().map(|t| t as i32).collect())
            .map_err(|e| generate_err(e))
    }

    fn render_request(&self, parsed: &ParsedRequest) -> Result<Vec<u8>, GenerateError> {
        // The Gemma-4 template is rendered with the thought channel closed:
        // the checkpoint answers in the content channel, which is what the
        // gate's "Stockholm" greedy run exercises. The reply splitter is the
        // ds4 server's syntax table, and it has no Gemma-4 channel entry yet,
        // so leaving thinking on would put the whole answer in
        // reasoning_content and mix the markers into the text. `think_mode`
        // stays unread until that syntax entry lands.
        let _ = parsed.think_mode;
        let messages: Vec<serde_json::Value> = parsed
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();
        let body = serde_json::json!({
            "thinking": false,
            "messages": messages,
        });
        Ok(body.to_string().into_bytes())
    }

    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError> {
        if let Some(text) = self.tok.token(token as u32) {
            // The tokenizer's own piece rules (`Tokenizer::decode`,
            // gemma4_a4b/tokenizer.rs:296-320): the byte-fallback form
            // `<0xXX>` is the raw byte, and `▁` is a space.
            if let Some(hex) = text.strip_prefix("<0x").and_then(|t| t.strip_suffix('>')) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    return Ok(vec![byte]);
                }
            }
            let raw = text.as_bytes();
            let mut bytes = Vec::with_capacity(raw.len());
            let mut i = 0;
            while i < raw.len() {
                if raw[i..].starts_with("▁".as_bytes()) {
                    bytes.push(b' ');
                    i += "▁".len();
                } else {
                    bytes.push(raw[i]);
                    i += 1;
                }
            }
            return Ok(bytes);
        }
        if let Some((_, text)) = self.specials.iter().find(|(id, _)| *id == token) {
            return Ok(text.as_bytes().to_vec());
        }
        Ok(Vec::new())
    }

    fn token_is_stop(&self, token: i32) -> bool {
        token == self.eos || (self.end_turn >= 0 && token == self.end_turn)
    }

    fn eos_id(&self) -> i32 {
        self.eos
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        if std::env::var_os("GM4_PROMPT_DEBUG").is_some() {
            let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            eprintln!(
                "prompt {} tokens: {:?}",
                tokens.len(),
                self.tok.decode(&ids)
            );
        }
        self.model.reset().map_err(generate_err)?;
        for &token in tokens {
            self.model.step(token as u32).map_err(generate_err)?;
        }
        self.generation += 1;
        Ok(())
    }

    fn eval(&mut self, token: i32) -> Result<(), GenerateError> {
        self.model.step(token as u32).map_err(generate_err)
    }

    fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        let sampler = Sampler {
            temperature,
            top_k,
            top_p,
            min_p,
        };
        draw(self.model.logits(), &sampler, rng, [-1, -1])
    }

    fn pos(&self) -> i32 {
        self.model.pos() as i32
    }

    fn ctx(&self) -> i32 {
        self.ctx
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn invalidate(&mut self) {
        let _ = self.model.reset();
    }
}
