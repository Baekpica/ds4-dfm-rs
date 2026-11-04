//! Typed Gemma-4 tuning surface.
//!
//! Every knob here used to be a hand-parsed `DS4_GEMMA4_*` getenv string read
//! at the point of use in the C/CUDA headers. Nothing validated it, nothing
//! listed it in `--check-config`, and a typo took the zero branch: a span of
//! `12x8` was read as 12 and clamped to 16, a miss policy of `cpux` meant
//! evict. This module is the single source of truth: the environment and the
//! command line are parsed here, every value is range-checked with an error
//! that names the knob, the resolved surface is echoed in the plan JSON, and
//! the same struct is carried into `ds4_engine_open` so the engine never
//! re-parses the string.
//!
//! Diagnostics (`DS4_GEMMA4_DEBUG`, `_DUMP`, `_VERIFY`, `_HASH_*`, `_ATTN_CMP`)
//! deliberately stay environment variables: they switch printing, not
//! arithmetic.

use serde_json::{json, Value};

pub const SPLIT_SPAN_MIN: u32 = 16;
pub const SPLIT_SPAN_MAX: u32 = 4096;
pub const SPLIT_SPAN_DEFAULT: u32 = 128;
pub const CPU_THREADS_MIN: u32 = 1;
pub const CPU_THREADS_MAX: u32 = 64;
pub const CPU_THREADS_DEFAULT: u32 = 20;
/// `G4_PREFILL_MAX` in ds4_gemma4_plan.h.
pub const PREFILL_CHUNK_MAX: u32 = 8192;
pub const VQ_GEMV_THREADS_MIN: u32 = 32;
pub const VQ_GEMV_THREADS_MAX: u32 = 512;
pub const VQ_GEMV_THREADS_DEFAULT: u32 = 256;

/// A knob the surface reads from `DS4_GEMMA4_<name>` and from
/// `--gemma4-<name>`, and echoes as `<name>` in the plan JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnobInfo {
    pub env: &'static str,
    pub flag: &'static str,
    pub json: &'static str,
    pub values: &'static str,
}

macro_rules! knob {
    ($env:literal, $flag:literal, $json:literal, $values:literal) => {
        KnobInfo {
            env: $env,
            flag: $flag,
            json: $json,
            values: $values,
        }
    };
}

/// Every arithmetic knob, in the order the plan prints them.
pub const KNOBS: &[KnobInfo] = &[
    knob!(
        "DS4_GEMMA4_ATTN_SPLIT_SPAN",
        "--gemma4-attn-split-span",
        "attn_split_span",
        "16..4096"
    ),
    knob!(
        "DS4_GEMMA4_ATTN_SPLIT_MMA",
        "--gemma4-attn-split-mma",
        "attn_split_mma",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_ATTN_SPLIT",
        "--gemma4-attn-split",
        "attn_split",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_ATTN_FAST",
        "--gemma4-attn-fast",
        "attn_fast",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_ATTN_PREFILL",
        "--gemma4-attn-prefill",
        "attn_prefill",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_ATTN_MMA",
        "--gemma4-attn-mma",
        "attn_mma",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_FUSE_ROPE",
        "--gemma4-fuse-rope",
        "fuse_rope",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_HEAD_GEMV",
        "--gemma4-head-gemv",
        "head_gemv",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_ROUTER_DOT",
        "--gemma4-router-dot",
        "router_dot",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_ROUTER_PAR",
        "--gemma4-router-par",
        "router_par",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_Q8_BATCH",
        "--gemma4-q8-batch",
        "q8_batch",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_MOE_PERSIST",
        "--gemma4-moe-persist",
        "moe_persist",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_MOE_FUSED",
        "--gemma4-moe-fused",
        "moe_fused",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_VQ_PREFILL",
        "--gemma4-vq-prefill",
        "vq_prefill",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_EXACT_PREFILL",
        "--gemma4-exact-prefill",
        "exact_prefill",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_Q4_DECODE",
        "--gemma4-q4-decode",
        "q4_decode",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_CPU_MOE",
        "--gemma4-cpu-moe",
        "cpu_moe",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_CPU_MOE_THREADS",
        "--gemma4-cpu-moe-threads",
        "cpu_moe_threads",
        "1..64"
    ),
    knob!(
        "DS4_GEMMA4_MOE_HOST",
        "--gemma4-moe-host",
        "moe_host",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_MOE_MISS",
        "--gemma4-moe-miss",
        "moe_miss",
        "evict|cpu"
    ),
    knob!(
        "DS4_GEMMA4_PREFILL_CHUNK",
        "--gemma4-prefill-chunk",
        "prefill_chunk",
        "1..8192, 0 = default"
    ),
    knob!(
        "DS4_GEMMA4_FULL_KV",
        "--gemma4-full-kv",
        "full_kv",
        "bf16|fp8"
    ),
    knob!(
        "DS4_GEMMA4_VQ_GEMV",
        "--gemma4-vq-gemv",
        "vq_gemv",
        "on|off"
    ),
    knob!(
        "DS4_GEMMA4_VQ_GEMV_VARIANT",
        "--gemma4-vq-gemv-variant",
        "vq_gemv_variant",
        "auto|warp|block"
    ),
    knob!(
        "DS4_GEMMA4_VQ_GEMV_THREADS",
        "--gemma4-vq-gemv-threads",
        "vq_gemv_threads",
        "32..512 step 32"
    ),
];

/// Names under the Gemma prefix that are diagnostics, launcher-level settings
/// or already-typed host flags. A name outside this union is reported as
/// ignored rather than silently doing nothing.
pub const KNOWN_ENV: &[&str] = &[
    // Diagnostics: they print, they never change arithmetic.
    "DS4_GEMMA4_ATTN_CMP",
    "DS4_GEMMA4_CPU_MOE_DEBUG",
    "DS4_GEMMA4_CPU_MOE_DUMP",
    "DS4_GEMMA4_DEBUG",
    "DS4_GEMMA4_GPU_MOE_DUMP",
    "DS4_GEMMA4_HASH_LAYER",
    "DS4_GEMMA4_HASH_LAYERS",
    "DS4_GEMMA4_MOE_PERSIST_VERIFY",
    "DS4_GEMMA4_Q4_DECODE_VERIFY",
    "DS4_GEMMA4_VQ_VERIFY",
    // Launcher-level (run-gemma4.sh); the server never reads these.
    "DS4_GEMMA4_CTX",
    "DS4_GEMMA4_FAST_ATTN",
    "DS4_GEMMA4_GROK_NAME",
    "DS4_GEMMA4_HOST",
    "DS4_GEMMA4_KV",
    "DS4_GEMMA4_LOCK",
    "DS4_GEMMA4_LOG",
    "DS4_GEMMA4_MEM_FLOOR",
    "DS4_GEMMA4_MMA",
    "DS4_GEMMA4_MODEL",
    "DS4_GEMMA4_MTP",
    "DS4_GEMMA4_MTP_DRAFT",
    "DS4_GEMMA4_MTP_MODE",
    "DS4_GEMMA4_PERSIST",
    "DS4_GEMMA4_PORT",
    "DS4_GEMMA4_PREFILL",
    "DS4_GEMMA4_RESIDENT",
    "DS4_GEMMA4_SERVER_BIN",
    "DS4_GEMMA4_SPLIT",
    "DS4_GEMMA4_SSD",
    "DS4_GEMMA4_SSD_CACHE",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Toggle {
    Off,
    On,
}

impl Toggle {
    pub fn parse(label: &str, raw: &str) -> Result<Self, String> {
        match raw {
            "1" | "true" | "on" => Ok(Self::On),
            "0" | "false" | "off" => Ok(Self::Off),
            _ => Err(format!("{label} wants on|off (got '{raw}')")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }

    pub fn is_on(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissPolicy {
    /// Evict a resident expert slot for the missing route (the default).
    Evict,
    /// Compute the missing route on the host and keep the resident slot.
    Cpu,
}

impl MissPolicy {
    pub fn parse(label: &str, raw: &str) -> Result<Self, String> {
        match raw {
            "evict" => Ok(Self::Evict),
            "cpu" => Ok(Self::Cpu),
            _ => Err(format!("{label} wants evict|cpu (got '{raw}')")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Evict => "evict",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullKv {
    Bf16,
    Fp8,
}

impl FullKv {
    pub fn parse(label: &str, raw: &str) -> Result<Self, String> {
        match raw {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => Err(format!("{label} wants bf16|fp8 (got '{raw}')")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }
}

/// `g4_vq_gemv` launch shape: the runtime heuristic, one warp per row, or one
/// block per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VqVariant {
    Auto,
    Warp,
    Block,
}

impl VqVariant {
    pub fn parse(label: &str, raw: &str) -> Result<Self, String> {
        match raw {
            "0" | "auto" => Ok(Self::Auto),
            "1" | "warp" => Ok(Self::Warp),
            "2" | "block" => Ok(Self::Block),
            _ => Err(format!("{label} wants auto|warp|block (got '{raw}')")),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::Auto => 0,
            Self::Warp => 1,
            Self::Block => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Warp => "warp",
            Self::Block => "block",
        }
    }
}

/// The resolved surface. Every field carries its effective value, so the plan
/// JSON, the engine and the gate all read the same numbers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma4Options {
    pub attn_split_span: u32,
    pub attn_split_mma: Toggle,
    pub attn_split: Toggle,
    pub attn_fast: Toggle,
    pub attn_prefill: Toggle,
    pub attn_mma: Toggle,
    pub fuse_rope: Toggle,
    pub head_gemv: Toggle,
    pub router_dot: Toggle,
    pub router_par: Toggle,
    pub q8_batch: Toggle,
    pub moe_persist: Toggle,
    pub moe_fused: Toggle,
    pub vq_prefill: Toggle,
    pub exact_prefill: Toggle,
    pub q4_decode: Toggle,
    pub cpu_moe: Toggle,
    pub cpu_moe_threads: u32,
    pub moe_host: Toggle,
    pub moe_miss: MissPolicy,
    pub prefill_chunk: u32,
    pub full_kv: FullKv,
    pub vq_gemv: Toggle,
    pub vq_gemv_variant: VqVariant,
    pub vq_gemv_threads: u32,
    /// At least one knob came from the environment or a flag. Used to warn
    /// when the knobs are set for a non-Gemma model.
    pub explicit: bool,
}

impl Default for Gemma4Options {
    fn default() -> Self {
        Self {
            attn_split_span: SPLIT_SPAN_DEFAULT,
            attn_split_mma: Toggle::Off,
            attn_split: Toggle::Off,
            attn_fast: Toggle::Off,
            attn_prefill: Toggle::Off,
            attn_mma: Toggle::Off,
            fuse_rope: Toggle::Off,
            head_gemv: Toggle::Off,
            router_dot: Toggle::Off,
            router_par: Toggle::Off,
            q8_batch: Toggle::Off,
            moe_persist: Toggle::Off,
            moe_fused: Toggle::Off,
            vq_prefill: Toggle::Off,
            exact_prefill: Toggle::Off,
            q4_decode: Toggle::Off,
            cpu_moe: Toggle::Off,
            cpu_moe_threads: CPU_THREADS_DEFAULT,
            moe_host: Toggle::Off,
            moe_miss: MissPolicy::Evict,
            prefill_chunk: 0,
            full_kv: FullKv::Bf16,
            vq_gemv: Toggle::On,
            vq_gemv_variant: VqVariant::Auto,
            vq_gemv_threads: VQ_GEMV_THREADS_DEFAULT,
            explicit: false,
        }
    }
}

fn bounded(label: &str, raw: &str, min: u32, max: u32) -> Result<u32, String> {
    let n: u32 = raw
        .parse()
        .map_err(|_| format!("{label} wants an integer in {min}..{max} (got '{raw}')"))?;
    if !(min..=max).contains(&n) {
        return Err(format!("{label} wants {min}..{max} (got '{raw}')"));
    }
    Ok(n)
}

impl Gemma4Options {
    /// Strict environment read: any value the knob cannot take is an error
    /// that names the variable, never a silent zero.
    pub fn from_env() -> Result<Self, String> {
        let mut out = Self::default();
        for info in KNOBS {
            if let Ok(raw) = std::env::var(info.env) {
                out.set_env(info.env, &raw)?;
                out.explicit = true;
            }
        }
        Ok(out)
    }

    /// Applies one `DS4_GEMMA4_*` value. `false` means the name is not one of
    /// this surface's knobs.
    pub fn set_env(&mut self, env: &str, raw: &str) -> Result<bool, String> {
        self.set_key(env, env, raw)
    }

    /// `label` is what the operator typed (a variable name or a flag); `key`
    /// selects the field. Every error names the label, never a translated
    /// spelling.
    fn set_key(&mut self, label: &str, key: &str, raw: &str) -> Result<bool, String> {
        match key {
            "DS4_GEMMA4_ATTN_SPLIT_SPAN" => {
                self.attn_split_span = bounded(label, raw, SPLIT_SPAN_MIN, SPLIT_SPAN_MAX)?;
            }
            "DS4_GEMMA4_ATTN_SPLIT_MMA" => self.attn_split_mma = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_ATTN_SPLIT" => self.attn_split = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_ATTN_FAST" => self.attn_fast = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_ATTN_PREFILL" => self.attn_prefill = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_ATTN_MMA" => self.attn_mma = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_FUSE_ROPE" => self.fuse_rope = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_HEAD_GEMV" => self.head_gemv = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_ROUTER_DOT" => self.router_dot = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_ROUTER_PAR" => self.router_par = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_Q8_BATCH" => self.q8_batch = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_MOE_PERSIST" => self.moe_persist = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_MOE_FUSED" => self.moe_fused = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_VQ_PREFILL" => self.vq_prefill = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_EXACT_PREFILL" => self.exact_prefill = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_Q4_DECODE" => self.q4_decode = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_CPU_MOE" => self.cpu_moe = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_CPU_MOE_THREADS" => {
                self.cpu_moe_threads = bounded(label, raw, CPU_THREADS_MIN, CPU_THREADS_MAX)?;
            }
            "DS4_GEMMA4_MOE_HOST" => self.moe_host = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_MOE_MISS" => self.moe_miss = MissPolicy::parse(label, raw)?,
            "DS4_GEMMA4_PREFILL_CHUNK" => {
                self.prefill_chunk = bounded(label, raw, 1, PREFILL_CHUNK_MAX)?;
            }
            "DS4_GEMMA4_FULL_KV" => self.full_kv = FullKv::parse(label, raw)?,
            "DS4_GEMMA4_VQ_GEMV" => self.vq_gemv = Toggle::parse(label, raw)?,
            "DS4_GEMMA4_VQ_GEMV_VARIANT" => {
                self.vq_gemv_variant = VqVariant::parse(label, raw)?;
            }
            "DS4_GEMMA4_VQ_GEMV_THREADS" => {
                let n = bounded(label, raw, VQ_GEMV_THREADS_MIN, VQ_GEMV_THREADS_MAX)?;
                if n % 32 != 0 {
                    return Err(format!(
                        "{label} wants a multiple of 32 in {VQ_GEMV_THREADS_MIN}..{VQ_GEMV_THREADS_MAX} (got '{raw}')"
                    ));
                }
                self.vq_gemv_threads = n;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Applies one `--gemma4-*` flag. `false` means the flag is not one of
    /// this surface's knobs (the caller then reports an unknown option).
    pub fn set_flag(&mut self, flag: &str, raw: &str) -> Result<bool, String> {
        let Some(info) = KNOBS.iter().find(|info| info.flag == flag) else {
            return Ok(false);
        };
        self.set_key(flag, info.env, raw)?;
        self.explicit = true;
        Ok(true)
    }

    /// Environment names under the Gemma prefix the engine does not read.
    /// These are reported, never fatal: the launcher owns some of them and a
    /// stale export should not stop a server.
    pub fn unknown_env() -> Vec<String> {
        let mut unknown: Vec<String> = std::env::vars()
            .map(|(key, _)| key)
            .filter(|key| key.starts_with("DS4_GEMMA4_"))
            .filter(|key| !KNOBS.iter().any(|info| info.env == key))
            .filter(|key| !KNOWN_ENV.contains(&key.as_str()))
            .collect();
        unknown.sort();
        unknown
    }

    /// Replaces the generic prefill chunk with the serving plan's resolved
    /// `--native-chunk` when the operator named one; the Gemma bind reads the
    /// same number from this surface.
    pub fn with_native_chunk(mut self, native: Option<u32>) -> Self {
        if let Some(n) = native {
            self.prefill_chunk = n;
            self.explicit = true;
        }
        self
    }

    pub fn to_json(&self) -> Value {
        json!({
            "attn_split_span": self.attn_split_span,
            "attn_split_mma": self.attn_split_mma.as_str(),
            "attn_split": self.attn_split.as_str(),
            "attn_fast": self.attn_fast.as_str(),
            "attn_prefill": self.attn_prefill.as_str(),
            "attn_mma": self.attn_mma.as_str(),
            "fuse_rope": self.fuse_rope.as_str(),
            "head_gemv": self.head_gemv.as_str(),
            "router_dot": self.router_dot.as_str(),
            "router_par": self.router_par.as_str(),
            "q8_batch": self.q8_batch.as_str(),
            "moe_persist": self.moe_persist.as_str(),
            "moe_fused": self.moe_fused.as_str(),
            "vq_prefill": self.vq_prefill.as_str(),
            "exact_prefill": self.exact_prefill.as_str(),
            "q4_decode": self.q4_decode.as_str(),
            "cpu_moe": self.cpu_moe.as_str(),
            "cpu_moe_threads": self.cpu_moe_threads,
            "moe_host": self.moe_host.as_str(),
            "moe_miss": self.moe_miss.as_str(),
            "prefill_chunk": self.prefill_chunk,
            "full_kv": self.full_kv.as_str(),
            "vq_gemv": self.vq_gemv.as_str(),
            "vq_gemv_variant": self.vq_gemv_variant.as_str(),
            "vq_gemv_threads": self.vq_gemv_threads,
            "explicit": self.explicit,
        })
    }

    /// One line per knob for `--check-config`'s human report.
    pub fn rows(&self) -> Vec<String> {
        let json = self.to_json();
        KNOBS
            .iter()
            .map(|info| {
                let value = json
                    .get(info.json)
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                format!("{}={}", info.env, value)
            })
            .collect()
    }

    /// The C engine's typed struct. `present` is always set here: a Rust host
    /// that carries this surface is authoritative.
    pub fn to_ffi(&self) -> ds4_sys::ds4_gemma4_options {
        let on = |t: Toggle| i32::from(t.is_on());
        ds4_sys::ds4_gemma4_options {
            present: 1,
            attn_split_mma: on(self.attn_split_mma),
            attn_split: on(self.attn_split),
            attn_fast: on(self.attn_fast),
            attn_prefill: on(self.attn_prefill),
            attn_mma: on(self.attn_mma),
            fuse_rope: on(self.fuse_rope),
            head_gemv: on(self.head_gemv),
            router_dot: on(self.router_dot),
            router_par: on(self.router_par),
            q8_batch: on(self.q8_batch),
            moe_persist: on(self.moe_persist),
            moe_fused: on(self.moe_fused),
            vq_prefill: on(self.vq_prefill),
            exact_prefill: on(self.exact_prefill),
            q4_decode: on(self.q4_decode),
            cpu_moe: on(self.cpu_moe),
            moe_host: on(self.moe_host),
            moe_miss_cpu: i32::from(self.moe_miss == MissPolicy::Cpu),
            full_kv_fp8: i32::from(self.full_kv == FullKv::Fp8),
            vq_gemv: on(self.vq_gemv),
            attn_split_span: self.attn_split_span,
            cpu_moe_threads: self.cpu_moe_threads,
            prefill_chunk: self.prefill_chunk,
            vq_gemv_variant: self.vq_gemv_variant.as_u32(),
            vq_gemv_threads: self.vq_gemv_threads,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_engine() {
        let d = Gemma4Options::default();
        assert_eq!(d.attn_split_span, 128);
        assert_eq!(d.cpu_moe_threads, 20);
        assert_eq!(d.prefill_chunk, 0);
        assert_eq!(d.vq_gemv, Toggle::On);
        assert_eq!(d.vq_gemv_threads, 256);
        assert_eq!(d.moe_miss, MissPolicy::Evict);
        assert_eq!(d.full_kv, FullKv::Bf16);
        assert!(!d.explicit);
    }

    #[test]
    fn a_span_typo_fails_by_name() {
        let mut o = Gemma4Options::default();
        let err = o.set_env("DS4_GEMMA4_ATTN_SPLIT_SPAN", "12x8").unwrap_err();
        assert!(err.contains("DS4_GEMMA4_ATTN_SPLIT_SPAN"), "{err}");
        let err = o.set_env("DS4_GEMMA4_ATTN_SPLIT_SPAN", "8").unwrap_err();
        assert!(err.contains("16..4096"), "{err}");
        let err = o.set_env("DS4_GEMMA4_ATTN_SPLIT_SPAN", "4097").unwrap_err();
        assert!(err.contains("16..4096"), "{err}");
        assert_eq!(o.attn_split_span, 128, "a rejected value must not stick");
    }

    #[test]
    fn span_bounds_are_inclusive() {
        let mut o = Gemma4Options::default();
        o.set_env("DS4_GEMMA4_ATTN_SPLIT_SPAN", "16").unwrap();
        assert_eq!(o.attn_split_span, 16);
        o.set_env("DS4_GEMMA4_ATTN_SPLIT_SPAN", "4096").unwrap();
        assert_eq!(o.attn_split_span, 4096);
    }

    #[test]
    fn threads_are_bounded_and_rejected_not_clamped() {
        let mut o = Gemma4Options::default();
        for bad in ["0", "65", "x", ""] {
            assert!(
                o.set_env("DS4_GEMMA4_CPU_MOE_THREADS", bad).is_err(),
                "{bad}"
            );
        }
        assert_eq!(o.cpu_moe_threads, 20);
        o.set_env("DS4_GEMMA4_CPU_MOE_THREADS", "64").unwrap();
        assert_eq!(o.cpu_moe_threads, 64);
    }

    #[test]
    fn booleans_are_strict() {
        let mut o = Gemma4Options::default();
        for good in ["1", "0", "true", "false", "on", "off"] {
            o.set_env("DS4_GEMMA4_ATTN_SPLIT_MMA", good)
                .unwrap_or_else(|e| panic!("{good}: {e}"));
        }
        for bad in ["2", "01", "yes", "", "TRUE"] {
            let err = o.set_env("DS4_GEMMA4_ATTN_SPLIT_MMA", bad).unwrap_err();
            assert!(err.contains("DS4_GEMMA4_ATTN_SPLIT_MMA"), "{err}");
        }
    }

    #[test]
    fn modes_are_enums() {
        let mut o = Gemma4Options::default();
        o.set_env("DS4_GEMMA4_MOE_MISS", "cpu").unwrap();
        assert_eq!(o.moe_miss, MissPolicy::Cpu);
        o.set_env("DS4_GEMMA4_MOE_MISS", "evict").unwrap();
        assert_eq!(o.moe_miss, MissPolicy::Evict);
        assert!(o.set_env("DS4_GEMMA4_MOE_MISS", "cpux").is_err());
        o.set_env("DS4_GEMMA4_FULL_KV", "fp8").unwrap();
        assert_eq!(o.full_kv, FullKv::Fp8);
        assert!(o.set_env("DS4_GEMMA4_FULL_KV", "fp8x").is_err());
        o.set_env("DS4_GEMMA4_VQ_GEMV_VARIANT", "block").unwrap();
        assert_eq!(o.vq_gemv_variant, VqVariant::Block);
        assert!(o.set_env("DS4_GEMMA4_VQ_GEMV_VARIANT", "3").is_err());
    }

    #[test]
    fn vq_threads_must_be_a_multiple_of_32() {
        let mut o = Gemma4Options::default();
        o.set_env("DS4_GEMMA4_VQ_GEMV_THREADS", "384").unwrap();
        assert_eq!(o.vq_gemv_threads, 384);
        for bad in ["31", "33", "16", "544"] {
            assert!(
                o.set_env("DS4_GEMMA4_VQ_GEMV_THREADS", bad).is_err(),
                "{bad}"
            );
        }
        assert_eq!(o.vq_gemv_threads, 384);
    }

    #[test]
    fn every_knob_has_a_flag_and_a_json_key() {
        let json = Gemma4Options::default().to_json();
        for info in KNOBS {
            assert!(info.flag.starts_with("--gemma4-"), "{}", info.flag);
            assert!(
                json.get(info.json).is_some(),
                "{} is missing from the plan JSON",
                info.json
            );
            assert!(
                info.env.starts_with("DS4_GEMMA4_"),
                "{} is not under the Gemma prefix",
                info.env
            );
        }
        assert_eq!(KNOBS.len(), 25, "a new knob needs a table row");
    }

    #[test]
    fn flags_route_through_the_same_validation() {
        let mut o = Gemma4Options::default();
        assert!(o.set_flag("--gemma4-attn-split-span", "64").unwrap());
        assert_eq!(o.attn_split_span, 64);
        assert!(o.explicit);
        let err = o.set_flag("--gemma4-attn-split-span", "8").unwrap_err();
        assert!(err.contains("--gemma4-attn-split-span"), "{err}");
        assert!(!o.set_flag("--not-a-gemma-knob", "1").unwrap());
    }

    #[test]
    fn ffi_mirrors_the_resolved_surface() {
        let mut o = Gemma4Options::default();
        o.set_env("DS4_GEMMA4_CPU_MOE", "1").unwrap();
        o.set_env("DS4_GEMMA4_CPU_MOE_THREADS", "12").unwrap();
        o.set_env("DS4_GEMMA4_MOE_HOST", "1").unwrap();
        o.set_env("DS4_GEMMA4_MOE_MISS", "cpu").unwrap();
        o.set_env("DS4_GEMMA4_FULL_KV", "fp8").unwrap();
        o.set_env("DS4_GEMMA4_ATTN_SPLIT_SPAN", "32").unwrap();
        let ffi = o.to_ffi();
        assert_eq!(ffi.present, 1);
        assert_eq!(ffi.cpu_moe, 1);
        assert_eq!(ffi.cpu_moe_threads, 12);
        assert_eq!(ffi.moe_host, 1);
        assert_eq!(ffi.moe_miss_cpu, 1);
        assert_eq!(ffi.full_kv_fp8, 1);
        assert_eq!(ffi.attn_split_span, 32);
        assert_eq!(ffi.vq_gemv, 1);
        assert_eq!(ffi.vq_gemv_threads, 256);
    }

    #[test]
    fn native_chunk_overrides_the_prefill_knob() {
        let o = Gemma4Options::default().with_native_chunk(Some(2048));
        assert_eq!(o.prefill_chunk, 2048);
        assert!(o.explicit);
        let o = Gemma4Options::default().with_native_chunk(None);
        assert_eq!(o.prefill_chunk, 0);
    }

    #[test]
    fn the_plan_json_carries_every_value() {
        let o = Gemma4Options {
            attn_split_span: 64,
            moe_miss: MissPolicy::Cpu,
            cpu_moe_threads: 8,
            ..Gemma4Options::default()
        };
        let json = o.to_json();
        assert_eq!(json["attn_split_span"], 64);
        assert_eq!(json["moe_miss"], "cpu");
        assert_eq!(json["cpu_moe_threads"], 8);
        assert_eq!(json["vq_gemv"], "on");
    }
}
