//! The CPU reference forward over the release.
//!
//! One token at a time, f32 throughout, with the release's weights read
//! through [`super::weights::Matrix`] (mapped, row-gathered) instead of being
//! materialised. It exists so the family has an executor that a serving host
//! can call before a device kernel exists; it claims correctness against the
//! reference architecture, not speed.
//!
//! Source of truth, read first-hand:
//!
//! - `/data/refs/transformers-5.14.1/transformers/models/gemma4/modeling_gemma4.py`
//!   (`Gemma4TextDecoderLayer.forward` :1405-1463, `Gemma4TextAttention.forward`
//!   :1235-1297, `Gemma4TextMLP.forward` :1088-1090, `Gemma4TextRouter.forward`
//!   :1339-1374, `Gemma4TextExperts.forward` :1300-1337,
//!   `Gemma4TextModel.forward` :1653-1742,
//!   `Gemma4TextScaledWordEmbedding.forward` :1474-1477, `Gemma4RMSNorm` :197-215).
//! - `/data/llama.cpp/src/models/gemma4.cpp`:155-441, the same graph as the
//!   GGML port reads it, including `f_attention_scale = 1.0` (:6) and the
//!   router's `rms_norm(attn_out) * 1/sqrt(n_embd) * ffn_gate_inp_s` (:308-313).
//!
//! Three facts the graph depends on and the shape constants do not express:
//!
//! - a full-attention layer has no `v_proj`: its value is the **raw** key
//!   projection (`value_states = key_states` before either norm, :1258-1263),
//!   so `v = v_norm(k_proj(h))` while `k = rope(k_norm(k_proj(h)))`;
//! - the routed block reads the post-attention residual twice: the router
//!   normalises that residual (`:1426-1428`), the experts normalise it with
//!   `pre_feedforward_layernorm_2`;
//! - attention has no `1/sqrt(head_dim)` scaling; the query/key norms carry
//!   it (`self.scaling = 1.0`, :1207).

use std::path::Path;

use super::config::{Config, LayerType};
use super::decode;
use super::layout::Role;
use super::release::Entry;
use super::weights::Matrix;
use super::{Error, Family};

/// `ACT2FN["gelu_pytorch_tanh"]`: `nn.functional.gelu(x, approximate="tanh")`.
pub fn gelu(x: f32) -> f32 {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (K * (x + 0.044_715 * x * x * x)).tanh())
}

/// Round to the nearest bfloat16 and back, which is what a bf16 checkpoint's
/// `embed_scale.to(embed_tokens.weight.dtype)` does before the multiply
/// (`modeling_gemma4.py:1477`; the note at :1605-1606 spells out the
/// sqrt(3072) rounding for the 31B sibling).
pub fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// `Gemma4RMSNorm.forward` (:210-214): `x * pow(mean(x^2) + eps, -0.5)`,
/// computed in f32, scaled by the weight when the norm has one.
pub fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps;
    let inv = mean.powf(-0.5);
    match weight {
        Some(w) => x.iter().zip(w).map(|(v, w)| v * inv * w).collect(),
        None => x.iter().map(|v| v * inv).collect(),
    }
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for v in &mut out {
        *v /= sum;
    }
    out
}

/// The top-k probabilities, highest first. Ties keep the lower index, which
/// is the order `torch.topk` returns for an exact tie.
fn top_k(probs: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    order.truncate(k);
    let weights = order.iter().map(|&i| probs[i]).collect();
    (order, weights)
}

/// One layer's weights.
struct LayerWeights {
    kind: LayerType,
    input_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    pre_ffn_norm: Vec<f32>,
    post_ffn_norm: Vec<f32>,
    pre_ffn_norm2: Vec<f32>,
    post_ffn_norm1: Vec<f32>,
    post_ffn_norm2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    layer_scalar: f32,
    q: Matrix,
    k: Matrix,
    v: Option<Matrix>,
    o: Matrix,
    dense_gate: Matrix,
    dense_up: Matrix,
    dense_down: Matrix,
    router: Matrix,
    router_scale: Vec<f32>,
    per_expert_scale: Vec<f32>,
    expert_gate_up: Matrix,
    expert_down: Matrix,
    /// `1 / theta^(2i/head_dim)` for the first half of the head, zero where
    /// proportional RoPE leaves the pair unrotated
    /// (`_compute_proportional_rope_parameters` :227-252).
    rope_inv: Vec<f32>,
}

/// The family's weights, opened for the forward.
pub struct Model {
    pub family: Family,
    pub config: Config,
    embed: Matrix,
    final_norm: Vec<f32>,
    layers: Vec<LayerWeights>,
}

fn scalar(dir: &Path, entry: &Entry) -> Result<f32, Error> {
    let t = decode::decode(dir, entry)?;
    t.data
        .first()
        .copied()
        .ok_or_else(|| Error::new(format!("{}: empty sidecar", entry.name)))
}

fn vector(dir: &Path, entry: &Entry) -> Result<Vec<f32>, Error> {
    Ok(decode::decode(dir, entry)?.data)
}

impl Model {
    /// Open the artifact and build every matrix handle. The payloads are
    /// checked once here (`Matrix::open`), so a step cannot fail on geometry.
    pub fn open(dir: &Path) -> Result<Model, Error> {
        let family = Family::open(dir)?;
        let config = family.config.clone();
        let embed = Matrix::open(dir, &family.layout.embed)?;
        let final_norm = vector(dir, &family.layout.final_norm)?;
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let spec = &family.layout.layers[index];
            let kind = spec.kind;
            let role = |role: Role| -> Result<&Entry, Error> { family.role(index, role) };
            let head_dim = config.head_dim_of(kind);
            let half = head_dim / 2;
            let theta = config.rope_theta(kind);
            let rotated = config.rotated_dims(kind);
            let mut rope_inv = vec![0f32; half];
            for (i, slot) in rope_inv.iter_mut().enumerate() {
                if i < rotated / 2 {
                    *slot = 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32);
                }
            }
            layers.push(LayerWeights {
                kind,
                input_norm: vector(dir, role(Role::InputNorm)?)?,
                post_attn_norm: vector(dir, role(Role::PostAttentionNorm)?)?,
                pre_ffn_norm: vector(dir, role(Role::PreFfnNorm)?)?,
                post_ffn_norm: vector(dir, role(Role::PostFfnNorm)?)?,
                pre_ffn_norm2: vector(dir, role(Role::PreFfnNorm2)?)?,
                post_ffn_norm1: vector(dir, role(Role::PostFfnNorm1)?)?,
                post_ffn_norm2: vector(dir, role(Role::PostFfnNorm2)?)?,
                q_norm: vector(dir, role(Role::QNorm)?)?,
                k_norm: vector(dir, role(Role::KNorm)?)?,
                layer_scalar: scalar(dir, role(Role::LayerScalar)?)?,
                q: Matrix::open(dir, role(Role::QProj)?)?,
                k: Matrix::open(dir, role(Role::KProj)?)?,
                v: match config.has_v_proj(kind) {
                    true => Some(Matrix::open(dir, role(Role::VProj)?)?),
                    false => None,
                },
                o: Matrix::open(dir, role(Role::OProj)?)?,
                dense_gate: Matrix::open(dir, role(Role::DenseGate)?)?,
                dense_up: Matrix::open(dir, role(Role::DenseUp)?)?,
                dense_down: Matrix::open(dir, role(Role::DenseDown)?)?,
                router: Matrix::open(dir, role(Role::RouterProj)?)?,
                router_scale: vector(dir, role(Role::RouterScale)?)?,
                per_expert_scale: vector(dir, role(Role::RouterPerExpertScale)?)?,
                expert_gate_up: Matrix::open(dir, role(Role::ExpertGateUp)?)?,
                expert_down: Matrix::open(dir, role(Role::ExpertDown)?)?,
                rope_inv,
            });
        }
        Ok(Model {
            family,
            config,
            embed,
            final_norm,
            layers,
        })
    }

    /// A fresh session (empty KV cache).
    pub fn session(&self) -> Session<'_> {
        let kv = self
            .layers
            .iter()
            .map(|layer| {
                let per = self.config.kv_heads_of(layer.kind) * self.config.head_dim_of(layer.kind);
                match layer.kind {
                    LayerType::Sliding => {
                        let cap = self.config.sliding_window;
                        Kv {
                            keys: vec![0f32; cap * per],
                            values: vec![0f32; cap * per],
                            cap,
                            per,
                            ring: true,
                        }
                    }
                    // The full layers grow with the context; allocating the
                    // checkpoint's 262,144 positions up front would be 5 GB.
                    LayerType::Full => Kv {
                        keys: Vec::new(),
                        values: Vec::new(),
                        cap: self.config.max_context,
                        per,
                        ring: false,
                    },
                }
            })
            .collect();
        Session {
            model: self,
            kv,
            len: 0,
        }
    }

    /// The tied output head: `lm_head.weight` is `embed_tokens.weight`
    /// (`Gemma4ForCausalLM._tied_weights_keys`), and it is the raw table.
    fn head(&self, x: &[f32]) -> Result<Vec<f32>, Error> {
        self.embed.matvec(x)
    }
}

/// One layer's cache. A sliding layer keeps `sliding_window` positions and
/// overwrites its oldest slot; a full layer keeps the whole context.
struct Kv {
    keys: Vec<f32>,
    values: Vec<f32>,
    cap: usize,
    per: usize,
    ring: bool,
}

impl Kv {
    fn slot(&self, pos: usize) -> usize {
        if self.ring {
            pos % self.cap
        } else {
            pos
        }
    }

    fn append(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        let slot = self.slot(pos);
        let need = (slot + 1) * self.per;
        if need > self.keys.len() {
            let grow = need
                .max(self.keys.len() + 256 * self.per)
                .min(self.cap * self.per);
            self.keys.resize(grow, 0f32);
            self.values.resize(grow, 0f32);
        }
        self.keys[slot * self.per..(slot + 1) * self.per].copy_from_slice(k);
        self.values[slot * self.per..(slot + 1) * self.per].copy_from_slice(v);
    }
}

/// A generation session: the KV cache and the position.
pub struct Session<'m> {
    model: &'m Model,
    kv: Vec<Kv>,
    len: usize,
}

impl<'m> Session<'m> {
    pub fn position(&self) -> usize {
        self.len
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Evaluate `tokens` in order and return the logits of the last one.
    /// A prompt is evaluated token by token: the fused gather makes one step
    /// cheap enough to prove the graph, and batching is a later unit.
    pub fn eval(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let mut logits = Vec::new();
        for &token in tokens {
            logits = self.step(token)?;
        }
        if tokens.is_empty() {
            return Err(Error::new("eval: no tokens"));
        }
        Ok(logits)
    }

    /// One forward step: append `token`, return the next-token logits.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        let config = &self.model.config;
        let pos = self.len;
        if pos >= config.max_context {
            return Err(Error::new(format!(
                "context is full at {pos} of {}",
                config.max_context
            )));
        }
        if token as usize >= config.vocab {
            return Err(Error::new(format!(
                "token {token} is outside the vocabulary"
            )));
        }
        let mut x = self.model.embed.row(token as usize)?;
        // `Gemma4TextScaledWordEmbedding.forward` multiplies by the scale
        // downcast to the weight dtype; the checkpoint is bf16, so the scale
        // is `bf16(sqrt(2816))` = 53, not 53.0659.
        let scale = bf16_round((config.hidden as f32).sqrt());
        for v in &mut x {
            *v *= scale;
        }
        for index in 0..self.model.layers.len() {
            x = self.layer(index, x, pos)?;
        }
        let x = rms_norm(&x, Some(&self.model.final_norm), config.rms_eps);
        let mut logits = self.model.head(&x)?;
        for v in &mut logits {
            *v = config.logit_softcap * (*v / config.logit_softcap).tanh();
        }
        self.len += 1;
        Ok(logits)
    }

    fn layer(&mut self, index: usize, x: Vec<f32>, pos: usize) -> Result<Vec<f32>, Error> {
        let model = self.model;
        let config = &model.config;
        let eps = config.rms_eps;
        let layer = &model.layers[index];
        let hidden = config.hidden;

        // residual = hidden_states; hidden_states = input_layernorm(...)
        let residual = x;
        let h = rms_norm(&residual, Some(&layer.input_norm), eps);
        let attn = self.attention(index, &h, pos)?;
        let attn = rms_norm(&attn, Some(&layer.post_attn_norm), eps);
        let x: Vec<f32> = residual.iter().zip(&attn).map(|(a, b)| a + b).collect();

        // The feed-forward block reads the post-attention residual twice.
        let h1 = rms_norm(&x, Some(&layer.pre_ffn_norm), eps);
        let gate = layer.dense_gate.matvec(&h1)?;
        let up = layer.dense_up.matvec(&h1)?;
        let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| gelu(*g) * u).collect();
        let down = layer.dense_down.matvec(&act)?;
        let dense = rms_norm(&down, Some(&layer.post_ffn_norm1), eps);

        // Router: rms_norm(attn_out) * router.scale / sqrt(hidden), then the
        // projection, an f32 softmax over every expert, top-k, renormalise,
        // and per-expert scale (`modeling_gemma4.py:1357-1372`).
        let mut r = rms_norm(&x, None, eps);
        let root = 1.0 / (hidden as f32).sqrt();
        for (slot, scale) in r.iter_mut().zip(&layer.router_scale) {
            *slot *= scale * root;
        }
        let scores = layer.router.matvec(&r)?;
        let probs = softmax(&scores);
        let (picked, mut weights) = top_k(&probs, config.top_k);
        let sum: f32 = weights.iter().sum();
        for (slot, &expert) in weights.iter_mut().zip(&picked) {
            *slot = *slot / sum * layer.per_expert_scale[expert];
        }

        let h2 = rms_norm(&x, Some(&layer.pre_ffn_norm2), eps);
        let mut moe = vec![0f32; hidden];
        for (&expert, &weight) in picked.iter().zip(&weights) {
            let base = expert * 2 * config.moe_ff;
            let rows: Vec<usize> = (base..base + 2 * config.moe_ff).collect();
            let gu = layer.expert_gate_up.dot_rows(&rows, &h2)?;
            let act: Vec<f32> = (0..config.moe_ff)
                .map(|i| gelu(gu[i]) * gu[config.moe_ff + i])
                .collect();
            let rows: Vec<usize> = (expert * hidden..(expert + 1) * hidden).collect();
            let d = layer.expert_down.dot_rows(&rows, &act)?;
            for (slot, value) in moe.iter_mut().zip(&d) {
                *slot += weight * value;
            }
        }
        let moe = rms_norm(&moe, Some(&layer.post_ffn_norm2), eps);

        let combined: Vec<f32> = dense.iter().zip(&moe).map(|(a, b)| a + b).collect();
        let combined = rms_norm(&combined, Some(&layer.post_ffn_norm), eps);
        let mut out: Vec<f32> = x.iter().zip(&combined).map(|(a, b)| a + b).collect();
        for v in &mut out {
            *v *= layer.layer_scalar;
        }
        Ok(out)
    }

    fn attention(&mut self, index: usize, h: &[f32], pos: usize) -> Result<Vec<f32>, Error> {
        let model = self.model;
        let config = &model.config;
        let layer = &model.layers[index];
        let kind = layer.kind;
        let heads = config.heads;
        let kv_heads = config.kv_heads_of(kind);
        let head_dim = config.head_dim_of(kind);
        let group = heads / kv_heads;

        let q = layer.q.matvec(h)?;
        let k_raw = layer.k.matvec(h)?;
        let v_raw = match &layer.v {
            Some(v) => v.matvec(h)?,
            // `value_states = key_states` before either norm, so the value is
            // the raw key projection (`:1258-1263`).
            None => k_raw.clone(),
        };

        let q = normalize_heads(&q, &layer.q_norm, heads, head_dim, config.rms_eps);
        let mut q = q;
        rope(&mut q, heads, head_dim, pos, &layer.rope_inv);
        let k = normalize_heads(&k_raw, &layer.k_norm, kv_heads, head_dim, config.rms_eps);
        let mut k = k;
        rope(&mut k, kv_heads, head_dim, pos, &layer.rope_inv);
        // `v_norm` has no weight (`with_scale=False`, :1240).
        let v = normalize_heads(&v_raw, &[], kv_heads, head_dim, config.rms_eps);

        let kv = &mut self.kv[index];
        kv.append(pos, &k, &v);
        let first = match kind {
            LayerType::Sliding => (pos + 1).saturating_sub(config.sliding_window),
            LayerType::Full => 0,
        };
        let mut out = vec![0f32; heads * head_dim];
        let mut scores = vec![0f32; pos + 1];
        for head in 0..heads {
            let kv_head = head / group;
            let q_head = &q[head * head_dim..(head + 1) * head_dim];
            let lo = first;
            for j in lo..=pos {
                let slot = kv.slot(j);
                let base = slot * kv.per + kv_head * head_dim;
                let k_head = &kv.keys[base..base + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[d] * k_head[d];
                }
                scores[j - lo] = dot;
            }
            let window = &scores[..pos + 1 - lo];
            let probs = softmax(window);
            let out_head = &mut out[head * head_dim..(head + 1) * head_dim];
            for (offset, &p) in probs.iter().enumerate() {
                let slot = kv.slot(lo + offset);
                let base = slot * kv.per + kv_head * head_dim;
                let v_head = &kv.values[base..base + head_dim];
                for d in 0..head_dim {
                    out_head[d] += p * v_head[d];
                }
            }
        }
        layer.o.matvec(&out)
    }
}

fn normalize_heads(x: &[f32], weight: &[f32], heads: usize, head_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for head in 0..heads {
        let start = head * head_dim;
        let slice = &x[start..start + head_dim];
        let mean = slice.iter().map(|v| v * v).sum::<f32>() / head_dim as f32 + eps;
        let inv = mean.powf(-0.5);
        let dst = &mut out[start..start + head_dim];
        for d in 0..head_dim {
            dst[d] = match weight.is_empty() {
                true => slice[d] * inv,
                false => slice[d] * inv * weight[d],
            };
        }
    }
    out
}

fn rope(x: &mut [f32], heads: usize, head_dim: usize, pos: usize, inv: &[f32]) {
    let half = head_dim / 2;
    for head in 0..heads {
        let base = head * head_dim;
        for d in 0..half {
            let angle = pos as f32 * inv[d];
            let (sin, cos) = angle.sin_cos();
            let a = x[base + d];
            let b = x[base + half + d];
            x[base + d] = a * cos - b * sin;
            x[base + half + d] = b * cos + a * sin;
        }
    }
}
