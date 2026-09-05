//! Safe Model / Session wrappers around `ds4-sys`, plus the host-owned
//! GGUF shape catalog, tensor inventory, bind plan, validate, vocab,
//! bind lookup, layout, tokenizer, session ledger, DSV4 payload prefix,
//! live memgov census snapshot, and the host-owned memgov D0b evaluator
//! (Phase 8).
//!
//! `unsafe` is confined to the FFI calls and the mmap adapter in this
//! crate. Application crates (`ds4-cli`, `ds4-server`, …) must not call
//! `ds4-sys` directly.

mod batch;
mod bind;
mod gguf;
mod identify;
mod layout;
mod mapped;
mod mem;
mod mem_gov;
mod payload;
mod progress;
mod session;
mod shape;
mod sibling;
mod tensors;
mod tok;
mod validate;

pub use batch::{
    cont_sample_token, qwen_image_pixel_hash, qwen_image_probe, BankSnapshot, BatchCtx, ContAdmit,
    ContDriver, QwenImageInfo, QwenImageInput, StaticBatchFinish, StaticBatchRequest,
    StaticBatchResult, CONT_SAMPLE_GREEDY, CONT_SAMPLE_NONE,
};
pub use bind::{
    bind_dspark_names, bind_mtp_names, bind_names, catalog_from_bind_name,
    dots3_layer_is_full_attention, dump_bind_check_oracle, dump_bind_dspark_shape,
    dump_bind_lookup_tapes, dump_bind_match_oracle, dump_bind_mtp_shape, dump_bind_names,
    dump_bind_names_shape, dump_bind_names_variant, dump_bind_support, expected_compress_ratio,
    glm53_layer_is_kda, host_bind_lookup, match_plans, solar_layer_is_gqa, variant_from_bind_name,
    BindError, BindName, BindNeed, BindPlan, BindSlot, HostBindLook, SupportCatalog,
    DSPARK_MARKOV_RANK, DSPARK_N_LAYER, HOST_BIND_MISS,
};
pub use gguf::{GgufError, GgufFile};
pub use identify::{dump_parse, identify_file, identify_gguf, Identified, IdentifyError};
pub use layout::{
    dump_expected_dspark_shape, dump_expected_layouts, dump_expected_layouts_shape,
    dump_expected_layouts_variant, dump_expected_mtp_shape, dump_expected_support,
    dump_layout_check_tapes, expected_dspark_layouts, expected_layouts, expected_mtp_layouts,
    validate_dspark_layouts, validate_layouts, validate_mtp_layouts, validate_support_layouts,
    LayoutError, LayoutSpec, TypeClass,
};
pub use mem::{
    snapshot_mem, MemCell as HostMemCell, MemCensus, MemObserve, MemSnap, MEMC_COUNT, MEMD_COUNT,
};
pub use mem_gov::{
    gov_compare, gov_epoch_read_begin, gov_epoch_read_verify, gov_epoch_write_begin,
    gov_epoch_write_end, gov_evaluate, gov_lease_publish, gov_mode_name, gov_mode_parse, GovClaim,
    GovCmp, GovConsumer, GovLease, GovLedger, GovMode, GovQuote, GovStatus, MemObsSource,
    MemObsStatus, MemObservation, GOVC_COUNT,
};
pub use payload::{
    dump_cmd as payload_dump_cmd, dump_script as payload_dump_script, encode_fields, parse_prefix,
    put_u32, tail as payload_tail, HostPrefix, PayloadError, PayloadLayout, HEADER_BYTES,
    LAYOUT_DOTS3, LAYOUT_EXAONE, LAYOUT_MOTIF3, LAYOUT_SOLAR, MAGIC as PAYLOAD_MAGIC,
    U32_FIELDS as PAYLOAD_U32_FIELDS, VERSION as PAYLOAD_VERSION,
};
pub use progress::PrefillCheckpoint;
pub use session::{
    dump_cmd as session_dump_cmd, RewriteKind, SessionBackend, SessionLedger, SyncPlan,
};
pub use shape::{
    dump_oracle, route_architecture, select_shape_from_metadata, shape_for_variant, ArchRoute,
    DeepSeekDims, ModelFamily, Shape, Variant, SHAPE_DOTS3_NOTE_PREV, SHAPE_FLASH,
    SHAPE_GLM53_FLASH, SHAPE_K2_HORIZON_375B, SHAPE_KEXAONE_236B, SHAPE_MOTIF3, SHAPE_PRO,
    SHAPE_QWEN38_FLASH_NEXT, SHAPE_SOLAR_OPEN2_250B,
};
pub use sibling::SiblingAttach;
pub use tensors::{
    apply_host_dir, consume_host_dir, dump_apply_tapes, dump_consume_tapes, dump_nbytes_table,
    dump_sibling_script, model_split_sibling_path, tensor_nbytes, tensor_type_name, TensorError,
    TensorInfo, TensorInventory,
};
pub use tok::{dump_cmd, dump_vocab_apply_tapes, ChatThinkMode, TokError, Vocab};
pub use validate::{
    dump_validate, host_compress_ratios, validate_file, validate_gguf, validate_qwen_inventory,
    ValidateError,
};

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_char;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt as _;
use std::path::Path;
use std::ptr::{self, NonNull};

use ds4_sys::{
    ds4_bridge_bind_plan, ds4_bridge_bind_plan_check, ds4_bridge_bind_slot,
    ds4_bridge_distributed_options, ds4_bridge_encode_chat_prompt, ds4_bridge_eval,
    ds4_bridge_eval_speculative_argmax, ds4_bridge_graph_fit_quote, ds4_bridge_model,
    ds4_bridge_model_boot_prewarm, ds4_bridge_model_free, ds4_bridge_model_open,
    ds4_bridge_model_open_distributed, ds4_bridge_model_open_options,
    ds4_bridge_model_run_distributed_worker, ds4_bridge_model_vision_probe, ds4_bridge_session,
    ds4_bridge_session_argmax, ds4_bridge_session_argmax_excluding, ds4_bridge_session_copy_logits,
    ds4_bridge_session_create, ds4_bridge_session_ctx, ds4_bridge_session_distributed_route_ready,
    ds4_bridge_session_eval_layer_slice, ds4_bridge_session_free, ds4_bridge_session_generation,
    ds4_bridge_session_graph_fit_quote, ds4_bridge_session_graph_pending,
    ds4_bridge_session_invalidate, ds4_bridge_session_layer_slice_reset,
    ds4_bridge_session_load_layer_payload, ds4_bridge_session_load_payload,
    ds4_bridge_session_load_payload_range, ds4_bridge_session_load_snapshot,
    ds4_bridge_session_output_head_bench, ds4_bridge_session_power, ds4_bridge_session_prefill_cap,
    ds4_bridge_session_rewind, ds4_bridge_session_sample, ds4_bridge_session_save_layer_payload,
    ds4_bridge_session_save_payload, ds4_bridge_session_save_snapshot,
    ds4_bridge_session_set_power, ds4_bridge_session_sync, ds4_bridge_session_sync_vision,
    ds4_bridge_session_top_logprobs, ds4_bridge_shard, ds4_bridge_snapshot,
    ds4_bridge_snapshot_create, ds4_bridge_snapshot_free, ds4_bridge_snapshot_len,
    ds4_bridge_token_score, ds4_bridge_vision_info, ds4_bridge_vision_input, ds4_host_bind_look,
    ds4_host_bind_map, ds4_host_shape, ds4_host_str, ds4_host_tensor, ds4_host_tensor_dir,
    ds4_host_vocab, DS4_BRIDGE_BACKEND_CPU, DS4_BRIDGE_BACKEND_CUDA, DS4_BRIDGE_BACKEND_METAL,
    DS4_BRIDGE_DISTRIBUTED_COORDINATOR, DS4_BRIDGE_DISTRIBUTED_NONE, DS4_BRIDGE_DISTRIBUTED_WORKER,
    DS4_BRIDGE_MAX_DIMS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
    Cpu,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelOpenOption {
    Quality,
    WarmWeights,
    PowerPercent(u8),
    MtpDraftTokens(i32),
    MtpMargin(f32),
    SteeringFile(String),
    SteeringAttn(f32),
    SteeringFfn(f32),
    Vision(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisionImageInfo {
    pub source_width: u32,
    pub source_height: u32,
    pub content_width: u32,
    pub content_height: u32,
    pub padded_width: u32,
    pub padded_height: u32,
    pub grid_height: u32,
    pub grid_width: u32,
    pub token_count: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionInput<'a> {
    pub data: &'a [u8],
    pub token_offset: u32,
}

#[derive(Clone, Debug)]
struct OpenTuning {
    quality: bool,
    warm_weights: bool,
    power_percent: i32,
    mtp_draft_tokens: i32,
    mtp_margin: f32,
    steering_file: Option<String>,
    steering_attn: f32,
    steering_ffn: f32,
    vision_path: Option<String>,
}

impl Default for OpenTuning {
    fn default() -> Self {
        Self {
            quality: false,
            warm_weights: false,
            power_percent: 100,
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
            steering_file: None,
            steering_attn: 0.0,
            steering_ffn: 0.0,
            vision_path: None,
        }
    }
}

fn open_tuning(options: &[ModelOpenOption]) -> Result<OpenTuning> {
    let mut tuning = OpenTuning::default();

    for option in options {
        match option {
            ModelOpenOption::Quality => tuning.quality = true,
            ModelOpenOption::WarmWeights => tuning.warm_weights = true,
            ModelOpenOption::PowerPercent(percent) if (1..=100).contains(percent) => {
                tuning.power_percent = i32::from(*percent);
            }
            ModelOpenOption::PowerPercent(_) => {
                return Err(Error {
                    code: 1,
                    message: "power percent must be between 1 and 100".into(),
                });
            }
            ModelOpenOption::MtpDraftTokens(n) if *n > 0 => tuning.mtp_draft_tokens = *n,
            ModelOpenOption::MtpDraftTokens(_) => {
                return Err(Error {
                    code: 1,
                    message: "mtp draft tokens must be positive".into(),
                });
            }
            ModelOpenOption::MtpMargin(margin) if (0.0..=1000.0).contains(margin) => {
                tuning.mtp_margin = *margin;
            }
            ModelOpenOption::MtpMargin(_) => {
                return Err(Error {
                    code: 1,
                    message: "mtp margin must be between 0 and 1000".into(),
                });
            }
            ModelOpenOption::SteeringFile(path) if !path.is_empty() => {
                tuning.steering_file = Some(path.clone());
            }
            ModelOpenOption::SteeringFile(_) => {
                return Err(Error {
                    code: 1,
                    message: "directional steering needs --dir-steering-file".into(),
                });
            }
            ModelOpenOption::SteeringAttn(scale) if (-100.0..=100.0).contains(scale) => {
                tuning.steering_attn = *scale;
            }
            ModelOpenOption::SteeringAttn(_) => {
                return Err(Error {
                    code: 1,
                    message: "dir-steering-attn must be between -100 and 100".into(),
                });
            }
            ModelOpenOption::SteeringFfn(scale) if (-100.0..=100.0).contains(scale) => {
                tuning.steering_ffn = *scale;
            }
            ModelOpenOption::SteeringFfn(_) => {
                return Err(Error {
                    code: 1,
                    message: "dir-steering-ffn must be between -100 and 100".into(),
                });
            }
            ModelOpenOption::Vision(path) if !path.is_empty() => {
                tuning.vision_path = Some(path.clone());
            }
            ModelOpenOption::Vision(_) => {
                return Err(Error {
                    code: 1,
                    message: "vision path must not be empty".into(),
                });
            }
        }
    }

    if (tuning.steering_attn != 0.0 || tuning.steering_ffn != 0.0) && tuning.steering_file.is_none()
    {
        return Err(Error {
            code: 1,
            message: "directional steering needs --dir-steering-file".into(),
        });
    }

    Ok(tuning)
}

impl Backend {
    fn to_c(self) -> i32 {
        match self {
            Backend::Cuda => DS4_BRIDGE_BACKEND_CUDA,
            Backend::Metal => DS4_BRIDGE_BACKEND_METAL,
            Backend::Cpu => DS4_BRIDGE_BACKEND_CPU,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributedRole {
    None,
    Coordinator,
    Worker,
}

impl DistributedRole {
    fn to_c(self) -> i32 {
        match self {
            Self::None => DS4_BRIDGE_DISTRIBUTED_NONE,
            Self::Coordinator => DS4_BRIDGE_DISTRIBUTED_COORDINATOR,
            Self::Worker => DS4_BRIDGE_DISTRIBUTED_WORKER,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedConfig {
    pub role: DistributedRole,
    pub layer_start: u32,
    pub layer_end: u32,
    pub has_output: bool,
    pub listen_host: Option<String>,
    pub listen_port: i32,
    pub coordinator_host: Option<String>,
    pub coordinator_port: i32,
    pub prefill_chunk: u32,
    pub prefill_window: u32,
    pub activation_bits: u32,
    pub replay_check: bool,
    pub debug: bool,
}

#[derive(Debug)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.message.is_empty() {
            write!(f, "ds4 bridge error {}", self.code)
        } else {
            write!(f, "ds4 bridge error {}: {}", self.code, self.message)
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub struct TokenBuffer {
    tokens: Vec<i32>,
}

impl TokenBuffer {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    pub fn from_tokens(tokens: Vec<i32>) -> Self {
        Self { tokens }
    }

    pub fn as_slice(&self) -> &[i32] {
        &self.tokens
    }

    pub fn push(&mut self, token: i32) {
        self.tokens.push(token);
    }

    pub fn truncate(&mut self, len: usize) {
        self.tokens.truncate(len);
    }

    pub fn insert(&mut self, mut index: usize, tokens: &[i32]) {
        if tokens.is_empty() {
            return;
        }
        if index > self.tokens.len() {
            index = self.tokens.len();
        }
        self.tokens.splice(index..index, tokens.iter().copied());
    }

    pub fn remove(&mut self, index: usize, n: usize) {
        if n == 0 || index >= self.tokens.len() {
            return;
        }
        let end = (index + n).min(self.tokens.len());
        self.tokens.drain(index..end);
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl Default for TokenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalResult {
    pub pos: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenScore {
    pub id: i32,
    pub logit: f32,
    pub logprob: f32,
}

/// Host copy of `ds4_session_graph_fit_quote`. Margin, not a refuse floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphFitQuote {
    pub fits: bool,
    pub fail_open: bool,
    pub need_bytes: u64,
    pub headroom_bytes: u64,
    pub avail_bytes: u64,
    pub deficit_bytes: u64,
}

impl GraphFitQuote {
    pub fn from_bridge(raw: ds4_bridge_graph_fit_quote) -> Self {
        Self {
            fits: raw.fits != 0,
            fail_open: raw.fail_open != 0,
            need_bytes: raw.need_bytes,
            headroom_bytes: raw.headroom_bytes,
            avail_bytes: raw.avail_bytes,
            deficit_bytes: raw.deficit_bytes,
        }
    }
}

pub struct Model {
    raw: NonNull<ds4_bridge_model>,
    family: ModelFamily,
    backend: Backend,
    inventory: TensorInventory,
    bind_plan: BindPlan,
    mtp: Option<SiblingAttach>,
    dspark: Option<SiblingAttach>,
    vocab: Vocab,
    _distributed: Option<FfiDistributed>,
    _not_send: PhantomData<*const ()>,
}

struct FfiDistributed {
    _listen_host: Option<CString>,
    _coordinator_host: Option<CString>,
    raw: ds4_bridge_distributed_options,
}

fn pack_distributed(config: &DistributedConfig) -> Result<FfiDistributed> {
    let listen_host = config
        .listen_host
        .as_deref()
        .map(CString::new)
        .transpose()
        .map_err(|_| Error {
            code: 1,
            message: "distributed listen host contains NUL".into(),
        })?;
    let coordinator_host = config
        .coordinator_host
        .as_deref()
        .map(CString::new)
        .transpose()
        .map_err(|_| Error {
            code: 1,
            message: "distributed coordinator host contains NUL".into(),
        })?;
    let raw = ds4_bridge_distributed_options {
        role: config.role.to_c(),
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        has_output: i32::from(config.has_output),
        listen_host: listen_host.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        listen_port: config.listen_port,
        coordinator_host: coordinator_host
            .as_ref()
            .map_or(ptr::null(), |s| s.as_ptr()),
        coordinator_port: config.coordinator_port,
        prefill_chunk: config.prefill_chunk,
        prefill_window: config.prefill_window,
        activation_bits: config.activation_bits,
        replay_check: i32::from(config.replay_check),
        debug: i32::from(config.debug),
    };
    Ok(FfiDistributed {
        _listen_host: listen_host,
        _coordinator_host: coordinator_host,
        raw,
    })
}

struct FfiBindMap {
    _names: Vec<CString>,
    looks: Vec<ds4_host_bind_look>,
    raw: ds4_host_bind_map,
}

fn pack_host_bind_map(plan: &BindPlan) -> Result<FfiBindMap> {
    let mut names = Vec::with_capacity(plan.slots.len());
    for s in &plan.slots {
        names.push(CString::new(s.name.as_str()).map_err(|_| Error {
            code: 1,
            message: "bind slot name contains NUL".into(),
        })?);
    }
    let looks: Vec<ds4_host_bind_look> = plan
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| ds4_host_bind_look {
            name: names[i].as_ptr(),
            required: u32::from(s.need.required()),
            found: u32::from(s.tensor.is_some()),
            index: s.index.unwrap_or(u32::MAX),
        })
        .collect();
    let raw = ds4_host_bind_map {
        n: looks.len() as u32,
        v: looks.as_ptr(),
    };
    Ok(FfiBindMap {
        _names: names,
        looks,
        raw,
    })
}

impl FfiBindMap {
    fn as_c(&mut self) -> *const ds4_host_bind_map {
        self.raw.v = self.looks.as_ptr();
        &self.raw
    }
}

struct FfiBindPlan {
    _names: Vec<CString>,
    _paths: Vec<CString>,
    slots: Vec<ds4_bridge_bind_slot>,
    shards: Vec<ds4_bridge_shard>,
    plan: ds4_bridge_bind_plan,
}

fn pack_bind_plan(plan: &BindPlan, inventory: &TensorInventory) -> Result<FfiBindPlan> {
    let mut names = Vec::with_capacity(plan.slots.len());
    for s in &plan.slots {
        names.push(CString::new(s.name.as_str()).map_err(|_| Error {
            code: 1,
            message: "bind slot name contains NUL".into(),
        })?);
    }
    let mut paths = Vec::with_capacity(inventory.shards.len());
    for sh in &inventory.shards {
        paths.push(
            CString::new(sh.path.to_string_lossy().into_owned()).map_err(|_| Error {
                code: 1,
                message: "shard path contains NUL".into(),
            })?,
        );
    }
    let slots: Vec<ds4_bridge_bind_slot> = plan
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let t = s.tensor.as_ref();
            let mut dim = [0u64; DS4_BRIDGE_MAX_DIMS];
            if let Some(t) = t {
                dim.copy_from_slice(&t.dim);
            }
            ds4_bridge_bind_slot {
                name: names[i].as_ptr(),
                required: u32::from(s.need.required()),
                ndim: t.map(|t| t.ndim).unwrap_or(0),
                dim,
                r#type: t.map(|t| t.typ).unwrap_or(0),
                rel_offset: t.map(|t| t.rel_offset).unwrap_or(0),
                abs_offset: t.map(|t| t.abs_offset).unwrap_or(0),
                bytes: t.map(|t| t.bytes).unwrap_or(0),
                shard: t.map(|t| t.shard).unwrap_or(0),
                found: u32::from(t.is_some()),
            }
        })
        .collect();
    let shards: Vec<ds4_bridge_shard> = inventory
        .shards
        .iter()
        .enumerate()
        .map(|(i, sh)| ds4_bridge_shard {
            path: paths[i].as_ptr(),
            size: sh.size,
            base: sh.base,
        })
        .collect();
    let c_plan = ds4_bridge_bind_plan {
        n_slots: slots.len() as u32,
        slots: slots.as_ptr(),
        n_shards: shards.len() as u32,
        shards: shards.as_ptr(),
        data_pos: inventory.data_pos,
        alignment: inventory.alignment,
        page: inventory.page,
    };
    Ok(FfiBindPlan {
        _names: names,
        _paths: paths,
        slots,
        shards,
        plan: c_plan,
    })
}

struct FfiTensorDir {
    _names: Vec<CString>,
    rows: Vec<ds4_host_tensor>,
    dir: ds4_host_tensor_dir,
}

fn pack_tensor_dir(inventory: &TensorInventory) -> Result<FfiTensorDir> {
    let mut names = Vec::with_capacity(inventory.tensors.len());
    for t in &inventory.tensors {
        names.push(CString::new(t.name.as_str()).map_err(|_| Error {
            code: 1,
            message: "tensor name contains NUL".into(),
        })?);
    }
    let rows: Vec<ds4_host_tensor> = inventory
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut dim = [0u64; DS4_BRIDGE_MAX_DIMS];
            dim.copy_from_slice(&t.dim);
            ds4_host_tensor {
                name: names[i].as_ptr(),
                ndim: t.ndim,
                dim,
                r#type: t.typ,
                rel_offset: t.rel_offset,
                abs_offset: t.abs_offset,
                bytes: t.bytes,
            }
        })
        .collect();
    let dir = ds4_host_tensor_dir {
        n: rows.len() as u32,
        v: rows.as_ptr(),
        data_pos: inventory.data_pos,
        alignment: inventory.alignment,
    };
    Ok(FfiTensorDir {
        _names: names,
        rows,
        dir,
    })
}

impl FfiTensorDir {
    fn as_c(&mut self) -> *const ds4_host_tensor_dir {
        self.dir.v = self.rows.as_ptr();
        &self.dir
    }
}

struct FfiVocab {
    tokens: Vec<ds4_host_str>,
    merges: Vec<ds4_host_str>,
    user_defined: Vec<i32>,
    raw: ds4_host_vocab,
}

fn bools_to_u8(src: &[bool; 256]) -> [u8; 256] {
    let mut out = [0u8; 256];
    for (i, b) in src.iter().enumerate() {
        out[i] = u8::from(*b);
    }
    out
}

fn pack_host_vocab(v: &Vocab) -> FfiVocab {
    let tokens: Vec<ds4_host_str> = v
        .tokens()
        .iter()
        .map(|t| ds4_host_str {
            ptr: t.as_ptr() as *const c_char,
            len: t.len() as u64,
        })
        .collect();
    let merges: Vec<ds4_host_str> = v
        .merges()
        .iter()
        .map(|t| ds4_host_str {
            ptr: t.as_ptr() as *const c_char,
            len: t.len() as u64,
        })
        .collect();
    let user_defined = v.user_defined_ids();
    let raw = ds4_host_vocab {
        n_vocab: tokens.len() as u32,
        tokens: tokens.as_ptr(),
        n_merges: merges.len() as u32,
        merges: merges.as_ptr(),
        n_user_defined: user_defined.len() as u32,
        user_defined: user_defined.as_ptr(),
        user_defined_max_len: v.user_defined_max_len(),
        user_defined_first: bools_to_u8(v.user_defined_first()),
        motif3_added_first: bools_to_u8(v.motif3_added_first()),
        bos_id: v.bos_id,
        eos_id: v.eos_id,
        system_id: v.system_id,
        eot_id: v.eot_id,
        im_start_id: v.im_start_id,
        im_content_id: v.im_content_id,
        im_end_id: v.im_end_id,
        user_id: v.user_id,
        assistant_id: v.assistant_id,
        start_of_turn_id: v.start_of_turn_id,
        end_of_turn_id: v.end_of_turn_id,
        tool_id: v.tool_id,
        reference_id: v.reference_id,
        plan_start_id: v.plan_start_id,
        plan_end_id: v.plan_end_id,
        observation_id: v.observation_id,
        sop_id: v.sop_id,
        think_start_id: v.think_start_id,
        think_end_id: v.think_end_id,
        tool_call_start_id: v.tool_call_start_id,
        tool_call_end_id: v.tool_call_end_id,
        tool_response_start_id: v.tool_response_start_id,
        tool_response_end_id: v.tool_response_end_id,
        arg_key_start_id: v.arg_key_start_id,
        arg_key_end_id: v.arg_key_end_id,
        arg_value_start_id: v.arg_value_start_id,
        latent_start_id: v.latent_start_id,
        latent_pad_id: v.latent_pad_id,
        latent_end_id: v.latent_end_id,
        tool_schema_start_id: v.tool_schema_start_id,
        tool_schema_end_id: v.tool_schema_end_id,
        dsml_id: v.dsml_id,
        dots3_endofsystem_id: v.dots3_endofsystem_id,
        dots3_endofuser_id: v.dots3_endofuser_id,
        dots3_endoftext_id: v.dots3_endoftext_id,
    };
    FfiVocab {
        tokens,
        merges,
        user_defined,
        raw,
    }
}

impl FfiVocab {
    fn as_c(&mut self) -> *const ds4_host_vocab {
        self.raw.tokens = self.tokens.as_ptr();
        self.raw.merges = self.merges.as_ptr();
        self.raw.user_defined = self.user_defined.as_ptr();
        &self.raw
    }
}

impl FfiBindPlan {
    fn as_c(&mut self) -> *const ds4_bridge_bind_plan {
        self.plan.slots = self.slots.as_ptr();
        self.plan.shards = self.shards.as_ptr();
        &self.plan
    }
}

pub struct Session<'m> {
    raw: NonNull<ds4_bridge_session>,
    host: SessionLedger,
    _model: PhantomData<&'m Model>,
    _not_send: PhantomData<*const ()>,
}

pub struct SessionSnapshot {
    raw: NonNull<ds4_bridge_snapshot>,
    host: Option<SessionLedger>,
    _not_send: PhantomData<*const ()>,
}

impl SessionSnapshot {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        let mut err = [0u8; 256];
        let rc = unsafe {
            ds4_bridge_snapshot_create(&mut raw, err.as_mut_ptr() as *mut c_char, err.len())
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            code: 1,
            message: "ds4_bridge_snapshot_create returned NULL".into(),
        })?;
        Ok(Self {
            raw,
            host: None,
            _not_send: PhantomData,
        })
    }

    pub fn len(&self) -> u64 {
        if self.host.is_none() {
            0
        } else {
            unsafe { ds4_bridge_snapshot_len(self.raw.as_ptr()) }
        }
    }
}

impl Drop for SessionSnapshot {
    fn drop(&mut self) {
        unsafe { ds4_bridge_snapshot_free(self.raw.as_ptr()) }
    }
}

fn c_err(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

fn fail(code: i32, buf: &[u8]) -> Error {
    Error {
        code,
        message: c_err(buf),
    }
}

fn cstring_path(path: &str) -> Result<CString> {
    CString::new(path).map_err(|_| Error {
        code: 1,
        message: "model path contains NUL".into(),
    })
}

fn cstring_payload_path(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    let bytes = path.as_os_str().as_bytes();
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| Error {
            code: 1,
            message: "payload path is not UTF-8".into(),
        })?
        .as_bytes();
    CString::new(bytes).map_err(|_| Error {
        code: 1,
        message: "payload path contains NUL".into(),
    })
}

fn save_payload_checked(
    raw: NonNull<ds4_bridge_session>,
    path: &Path,
    expected: &[i32],
    family: ModelFamily,
    ctx: i32,
) -> Result<()> {
    let c_path = cstring_payload_path(path)?;
    let mut err = [0u8; 512];
    let rc = unsafe {
        ds4_bridge_session_save_payload(
            raw.as_ptr(),
            c_path.as_ptr(),
            err.as_mut_ptr() as *mut c_char,
            err.len(),
        )
    };
    if rc != 0 {
        return Err(fail(rc, &err));
    }
    let mut file = std::fs::File::open(path).map_err(|e| Error {
        code: 1,
        message: format!("failed to reopen session payload: {e}"),
    })?;
    let payload_bytes = file
        .metadata()
        .map_err(|e| Error {
            code: 1,
            message: format!("failed to measure session payload: {e}"),
        })?
        .len();
    let prefix = crate::payload::read_prefix_range(&mut file, 0, payload_bytes, family, ctx)
        .map_err(|e| Error {
            code: 1,
            message: e.to_string(),
        })?;
    let expected: Vec<u32> = expected.iter().map(|&token| token as u32).collect();
    if prefix.tokens != expected {
        return Err(Error {
            code: 1,
            message: "host/native token mismatch".into(),
        });
    }
    Ok(())
}

/// Packed sibling bind map for the bridge open call only. Host lifecycle
/// lives in [`SiblingAttach`]; this must outlive the FFI open.
struct FfiSupport {
    path: CString,
    map: FfiBindMap,
}

fn pack_sibling_ffi(attach: &SiblingAttach) -> Result<FfiSupport> {
    Ok(FfiSupport {
        path: cstring_path(attach.path())?,
        map: pack_host_bind_map(attach.bind_plan())?,
    })
}

// Phase 8.6 model-bridge inventory (todo 45). No CUDA/mmap move this slice.
// KEEP native (mmap GGUF + CUDA/VMM alloc + engine teardown):
//   ds4_bridge_model_open / open_distributed (open_impl)
//   ds4_bridge_model_free (Drop: session before model, siblings before base)
//   ds4_bridge_model_boot_prewarm (device graph/weight warm)
// MOVED (host catalog, no FFI):
//   model_id / routed_quant_bits
// MOVE later (production already left):
//   ds4_bridge_model_run_distributed_worker -> assemble_worker (oracle FFI)
impl Model {
    pub fn open(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
    ) -> Result<Self> {
        Self::open_configured(path, backend, n_threads, defer_boot_prewarm, None, &[])
    }

    pub fn open_configured(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
        distributed: Option<&DistributedConfig>,
        options: &[ModelOpenOption],
    ) -> Result<Self> {
        Self::open_impl(
            path,
            backend,
            n_threads,
            defer_boot_prewarm,
            None,
            None,
            distributed,
            options,
        )
    }

    /// `mtp_path` / `dspark_path` attach the DeepSeek-only sibling support
    /// models. The host resolves each sibling's bind catalog and expected
    /// layouts, then native skips that sibling's name walk and layout check.
    pub fn open_with_support(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
        mtp_path: Option<&str>,
        dspark_path: Option<&str>,
    ) -> Result<Self> {
        Self::open_with_support_options(
            path,
            backend,
            n_threads,
            defer_boot_prewarm,
            mtp_path,
            dspark_path,
            &[],
        )
    }

    pub fn open_with_support_options(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
        mtp_path: Option<&str>,
        dspark_path: Option<&str>,
        options: &[ModelOpenOption],
    ) -> Result<Self> {
        Self::open_impl(
            path,
            backend,
            n_threads,
            defer_boot_prewarm,
            mtp_path,
            dspark_path,
            None,
            options,
        )
    }

    pub fn open_distributed(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
        mtp_path: Option<&str>,
        dspark_path: Option<&str>,
        distributed: &DistributedConfig,
    ) -> Result<Self> {
        Self::open_distributed_options(
            path,
            backend,
            n_threads,
            defer_boot_prewarm,
            mtp_path,
            dspark_path,
            distributed,
            &[],
        )
    }

    pub fn open_distributed_options(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
        mtp_path: Option<&str>,
        dspark_path: Option<&str>,
        distributed: &DistributedConfig,
        options: &[ModelOpenOption],
    ) -> Result<Self> {
        Self::open_impl(
            path,
            backend,
            n_threads,
            defer_boot_prewarm,
            mtp_path,
            dspark_path,
            Some(distributed),
            options,
        )
    }

    fn open_impl(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
        mtp_path: Option<&str>,
        dspark_path: Option<&str>,
        distributed: Option<&DistributedConfig>,
        options: &[ModelOpenOption],
    ) -> Result<Self> {
        let tuning = open_tuning(options)?;
        let identified = identify_gguf(std::path::Path::new(path)).map_err(|e| Error {
            code: 1,
            message: format!("identify failed: {}", e.token()),
        })?;
        let g = GgufFile::open(std::path::Path::new(path)).map_err(|e| Error {
            code: 1,
            message: format!("validate failed: {}", e.token()),
        })?;
        validate_file(&g, &identified.shape).map_err(|e| Error {
            code: 1,
            message: format!("validate failed: {}", e.token()),
        })?;
        let vocab = Vocab::load(&g, identified.shape.family).map_err(|e| Error {
            code: 1,
            message: format!("vocab failed: {e}"),
        })?;
        let mut ffi_vocab = pack_host_vocab(&vocab);
        let compress = host_compress_ratios(&identified.shape);
        let ffi_shape = ds4_host_shape {
            variant: identified.shape.variant as u32,
            n_compress: compress.len() as u32,
            compress: if compress.is_empty() {
                ptr::null()
            } else {
                compress.as_ptr()
            },
        };
        let inventory = TensorInventory::open(std::path::Path::new(path)).map_err(|e| Error {
            code: 1,
            message: format!("tensor inventory failed: {}", e.token()),
        })?;
        validate_qwen_inventory(&g, &inventory).map_err(|e| Error {
            code: 1,
            message: format!("validate failed: {}", e.token()),
        })?;
        let bind_plan = BindPlan::resolve(identified.shape, &inventory);
        if let Some(name) = bind_plan.missing_required().first() {
            return Err(Error {
                code: 1,
                message: format!("required tensor is missing: {name}"),
            });
        }
        validate_layouts(&bind_plan).map_err(|e| Error {
            code: 1,
            message: format!("layout failed: {}", e.token()),
        })?;
        let mut ffi_plan = pack_bind_plan(&bind_plan, &inventory)?;
        let mut ffi_bind = pack_host_bind_map(&bind_plan)?;
        let mut ffi_dir = pack_tensor_dir(&inventory)?;
        let (mtp, dspark) = sibling::attach_siblings(
            identified.shape.family,
            identified.shape,
            sibling::SiblingPaths {
                mtp: mtp_path,
                dspark: dspark_path,
            },
        )?;
        let mut mtp_support = mtp.as_ref().map(pack_sibling_ffi).transpose()?;
        let mut dspark_support = dspark.as_ref().map(pack_sibling_ffi).transpose()?;
        let mut err = [0u8; 512];
        let check = unsafe {
            ds4_bridge_bind_plan_check(ffi_plan.as_c(), err.as_mut_ptr() as *mut c_char, err.len())
        };
        if check != 0 {
            return Err(fail(check, &err));
        }
        let c_path = cstring_path(path)?;
        let vision_path = tuning
            .vision_path
            .as_deref()
            .map(cstring_path)
            .transpose()?;
        let steering_file = tuning
            .steering_file
            .as_deref()
            .map(cstring_path)
            .transpose()?;
        let (mtp_path_ptr, mtp_bind_ptr) = match mtp_support.as_mut() {
            Some(s) => (s.path.as_ptr(), s.map.as_c()),
            None => (ptr::null(), ptr::null()),
        };
        let (dspark_path_ptr, dspark_bind_ptr) = match dspark_support.as_mut() {
            Some(s) => (s.path.as_ptr(), s.map.as_c()),
            None => (ptr::null(), ptr::null()),
        };
        let opt = ds4_bridge_model_open_options {
            model_path: c_path.as_ptr(),
            vision_path: vision_path
                .as_ref()
                .map(|path| path.as_ptr())
                .unwrap_or(ptr::null()),
            backend: backend.to_c(),
            n_threads,
            defer_boot_prewarm: i32::from(defer_boot_prewarm),
            power_percent: tuning.power_percent,
            warm_weights: i32::from(tuning.warm_weights),
            quality: i32::from(tuning.quality),
            plan: ffi_plan.as_c(),
            tensors: ffi_dir.as_c(),
            shape: &ffi_shape,
            vocab: ffi_vocab.as_c(),
            bind: ffi_bind.as_c(),
            mtp_path: mtp_path_ptr,
            dspark_path: dspark_path_ptr,
            mtp_bind: mtp_bind_ptr,
            dspark_bind: dspark_bind_ptr,
            mtp_draft_tokens: tuning.mtp_draft_tokens,
            mtp_margin: tuning.mtp_margin,
            directional_steering_file: steering_file
                .as_ref()
                .map(|path| path.as_ptr())
                .unwrap_or(ptr::null()),
            directional_steering_attn: tuning.steering_attn,
            directional_steering_ffn: tuning.steering_ffn,
        };
        let mut ffi_distributed = distributed.map(pack_distributed).transpose()?;
        let mut raw = ptr::null_mut();
        let rc = unsafe {
            match ffi_distributed.as_mut() {
                Some(distributed) => ds4_bridge_model_open_distributed(
                    &mut raw,
                    &opt,
                    &distributed.raw,
                    err.as_mut_ptr() as *mut c_char,
                    err.len(),
                ),
                None => ds4_bridge_model_open(
                    &mut raw,
                    &opt,
                    err.as_mut_ptr() as *mut c_char,
                    err.len(),
                ),
            }
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            code: 1,
            message: "ds4_bridge_model_open returned NULL".into(),
        })?;
        Ok(Self {
            raw,
            family: identified.shape.family,
            backend,
            inventory,
            bind_plan,
            mtp,
            dspark,
            vocab,
            _distributed: ffi_distributed,
            _not_send: PhantomData,
        })
    }

    pub fn family(&self) -> ModelFamily {
        self.family
    }

    pub fn inventory(&self) -> &TensorInventory {
        &self.inventory
    }

    pub fn bind_plan(&self) -> &BindPlan {
        &self.bind_plan
    }

    pub fn mtp(&self) -> Option<&SiblingAttach> {
        self.mtp.as_ref()
    }

    pub fn dspark(&self) -> Option<&SiblingAttach> {
        self.dspark.as_ref()
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    pub(crate) fn raw_ptr(&self) -> *mut ds4_bridge_model {
        self.raw.as_ptr()
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn vision_probe(&self, data: &[u8]) -> Result<VisionImageInfo> {
        let mut info = ds4_bridge_vision_info::default();
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_model_vision_probe(
                self.raw.as_ptr(),
                data.as_ptr(),
                data.len(),
                &mut info,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        Ok(VisionImageInfo {
            source_width: info.source_width,
            source_height: info.source_height,
            content_width: info.content_width,
            content_height: info.content_height,
            padded_width: info.padded_width,
            padded_height: info.padded_height,
            grid_height: info.grid_height,
            grid_width: info.grid_width,
            token_count: info.token_count,
        })
    }

    pub fn session(&self, ctx_size: i32) -> Result<Session<'_>> {
        let mut raw = ptr::null_mut();
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_create(
                &mut raw,
                self.raw.as_ptr(),
                ctx_size,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            code: 1,
            message: "ds4_bridge_session_create returned NULL".into(),
        })?;
        let prefill = unsafe { ds4_bridge_session_prefill_cap(raw.as_ptr()) };
        let native_ctx = unsafe { ds4_bridge_session_ctx(raw.as_ptr()) };
        let host_backend = match self.backend {
            Backend::Cpu => SessionBackend::Cpu,
            Backend::Cuda | Backend::Metal => SessionBackend::Cuda,
        };
        let mut host = SessionLedger::new(
            self.family,
            host_backend,
            ledger_ctx(ctx_size, native_ctx),
            prefill.max(0) as u32,
        );
        host.apply_shape(&self.bind_plan.shape);
        Ok(Session {
            raw,
            host,
            _model: PhantomData,
            _not_send: PhantomData,
        })
    }

    pub fn boot_prewarm(&self) {
        unsafe { ds4_bridge_model_boot_prewarm(self.raw.as_ptr()) }
    }

    pub fn shape(&self) -> Shape {
        self.bind_plan.shape
    }

    /// C `ds4_engine_model_id` is `DS4_MODEL_VARIANT`. Host shape is the truth.
    pub fn model_id(&self) -> i32 {
        self.bind_plan.shape.model_id()
    }

    /// C `ds4_engine_routed_quant_bits` walks base `ffn_gate_exps`.
    pub fn routed_quant_bits(&self) -> i32 {
        self.bind_plan.routed_quant_bits()
    }

    pub fn session_graph_fit_quote(&self, ctx_size: i32) -> Option<GraphFitQuote> {
        if ctx_size <= 0 {
            return None;
        }
        let mut raw = ds4_bridge_graph_fit_quote::default();
        let _fits =
            unsafe { ds4_bridge_session_graph_fit_quote(self.raw.as_ptr(), ctx_size, &mut raw) };
        Some(GraphFitQuote::from_bridge(raw))
    }

    pub fn run_distributed_worker(&self, ctx_size: i32) -> Result<i32> {
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_model_run_distributed_worker(
                self.raw.as_ptr(),
                ctx_size,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        Ok(rc)
    }

    pub fn token_eos(&self) -> i32 {
        self.vocab.engine_eos()
    }

    pub fn token_is_stop(&self, token: i32) -> bool {
        self.vocab.is_stop(token)
    }

    /// CLI chat-template encode through the engine (`ds4_encode_chat_prompt`):
    /// exact C `-p` prompt-token parity for the proof harness.
    pub fn encode_chat_prompt(
        &self,
        system: Option<&str>,
        prompt: &str,
        think_mode: i32,
    ) -> Result<TokenBuffer> {
        self.encode_chat_prompt_bytes(system.map(str::as_bytes), prompt.as_bytes(), think_mode)
    }

    /// Byte-input form for C callers that read prompt files before the first
    /// NUL and do not require UTF-8.
    pub fn encode_chat_prompt_bytes(
        &self,
        system: Option<&[u8]>,
        prompt: &[u8],
        think_mode: i32,
    ) -> Result<TokenBuffer> {
        let c_system = match system {
            Some(s) => Some(CString::new(s).map_err(|_| Error {
                code: 1,
                message: "system contains NUL".into(),
            })?),
            None => None,
        };
        let c_prompt = CString::new(prompt).map_err(|_| Error {
            code: 1,
            message: "prompt contains NUL".into(),
        })?;
        // BPE merges only shrink and specials add a bounded prefix.
        let cap = prompt.len() + system.map_or(0, <[u8]>::len) + 256;
        let mut out = vec![0i32; cap];
        let mut n_out = 0i32;
        let mut err = [0u8; 256];
        let rc = unsafe {
            ds4_bridge_encode_chat_prompt(
                self.raw.as_ptr(),
                c_system.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                c_prompt.as_ptr(),
                think_mode,
                out.as_mut_ptr(),
                cap as i32,
                &mut n_out,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        out.truncate(n_out.max(0) as usize);
        Ok(TokenBuffer::from_tokens(out))
    }

    pub fn tokenize_text(&self, text: &str) -> Result<TokenBuffer> {
        Ok(TokenBuffer::from_tokens(self.vocab.encode_text(text)))
    }

    pub fn tokenize_rendered_chat(&self, text: &str) -> Result<TokenBuffer> {
        Ok(TokenBuffer::from_tokens(
            self.vocab.encode_rendered_chat(text),
        ))
    }

    pub fn token_text(&self, token: i32) -> Result<Vec<u8>> {
        Ok(self.vocab.token_text(token))
    }
}

fn drop_model_native(raw: NonNull<ds4_bridge_model>) {
    // SAFETY: [Category 8 — FFI boundary]
    // `raw` is the unique owner from open, or a test dangling pointer
    // whose free is stubbed. Drop runs this once. C `ds4_engine_close`
    // closes MTP then DSpark siblings before the base GGUF; mmap and
    // CUDA teardown stay in that C path (todo 45 KEEP).
    unsafe { ds4_bridge_model_free(raw.as_ptr()) }
}

impl Drop for Model {
    fn drop(&mut self) {
        drop_model_native(self.raw);
    }
}

const fn ledger_ctx(configured: i32, native_effective: i32) -> i32 {
    if native_effective > 0 {
        native_effective
    } else {
        configured
    }
}

impl Session<'_> {
    pub fn host(&self) -> &SessionLedger {
        &self.host
    }

    pub fn generation(&self) -> u64 {
        self.host.generation
    }

    pub fn last_plan(&self, tokens: &[i32]) -> SyncPlan {
        self.host
            .plan_sync(tokens, self.host.planned_exaone_rewind_span())
    }

    fn check_sync(&self, tokens: &TokenBuffer) -> Result<SyncPlan> {
        if tokens.len() > i32::MAX as usize {
            return Err(Error {
                code: 1,
                message: "token buffer exceeds i32 length".into(),
            });
        }
        let plan = self.last_plan(tokens.as_slice());
        if plan.bounds {
            return Err(Error {
                code: 1,
                message: "prompt exceeds context".into(),
            });
        }
        if plan.fence {
            return Err(Error {
                code: 1,
                message: "whole-prompt prefill fenced".into(),
            });
        }
        Ok(plan)
    }

    pub fn sync(&mut self, tokens: &TokenBuffer) -> Result<()> {
        let plan = self.check_sync(tokens)?;
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_sync(
                self.raw.as_ptr(),
                tokens.as_slice().as_ptr(),
                tokens.len() as i32,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        self.host.commit_sync(tokens.as_slice(), &plan);
        Ok(())
    }

    pub fn sync_vision(&mut self, tokens: &TokenBuffer, images: &[VisionInput<'_>]) -> Result<()> {
        self.check_sync(tokens)?;
        if images.is_empty() || images.len() > 4 {
            return Err(Error {
                code: 1,
                message: "vision sync requires 1 to 4 images".into(),
            });
        }
        let images = images
            .iter()
            .map(|image| ds4_bridge_vision_input {
                data: image.data.as_ptr(),
                data_len: image.data.len(),
                token_offset: image.token_offset,
            })
            .collect::<Vec<_>>();
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_sync_vision(
                self.raw.as_ptr(),
                tokens.as_slice().as_ptr(),
                tokens.len() as i32,
                images.as_ptr(),
                images.len() as u32,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        self.host.invalidate();
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        self.host.replace_checkpoint(tokens.as_slice());
        Ok(())
    }

    pub fn eval(&mut self, token: i32) -> Result<EvalResult> {
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_eval(
                self.raw.as_ptr(),
                token,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        self.host.commit_eval(token);
        Ok(EvalResult {
            pos: self.host.pos(),
        })
    }

    /// Worker slice eval: native `ds4_session_eval_layer_slice`.
    pub fn eval_layer_slice(&mut self, req: LayerSliceEval<'_>) -> Result<()> {
        let n_tokens = u32::try_from(req.tokens.len()).map_err(|_| Error {
            code: 1,
            message: "layer slice token count exceeds u32".into(),
        })?;
        let mut err = [0u8; 512];
        // SAFETY: [Category 8 — FFI] `raw` is a live session from create. Token
        // and activation pointers are borrowed for the call only; empty slices
        // still yield a non-null aligned pointer. Optional buffers pass NULL.
        // The bridge does not retain the pointers.
        let rc = unsafe {
            ds4_bridge_session_eval_layer_slice(
                self.raw.as_ptr(),
                req.tokens.as_ptr(),
                n_tokens,
                req.pos0,
                req.layer_start,
                req.layer_end,
                req.input_hc.map_or(ptr::null(), |values| values.as_ptr()),
                req.output_hc
                    .map_or(ptr::null_mut(), |values| values.as_mut_ptr()),
                i32::from(req.logits.is_some()),
                req.logits
                    .map_or(ptr::null_mut(), |values| values.as_mut_ptr()),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        let prefix_len = (req.pos0 as usize).min(self.host.tokens().len());
        let mut timeline = self.host.tokens()[..prefix_len].to_vec();
        timeline.extend_from_slice(req.tokens);
        self.host.replace_checkpoint(&timeline);
        Ok(())
    }

    pub fn layer_slice_reset(&mut self) -> Result<()> {
        let mut err = [0u8; 512];
        // SAFETY: [Category 8 — FFI] `raw` is a live session from create. err is
        // a stack buffer whose length is passed to native. The bridge does not
        // retain the pointers.
        let rc = unsafe {
            ds4_bridge_session_layer_slice_reset(
                self.raw.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        self.host.invalidate();
        Ok(())
    }

    pub fn eval_speculative_argmax(
        &mut self,
        first: i32,
        max_tokens: i32,
        eos: i32,
    ) -> Result<Vec<i32>> {
        let mut accepted = vec![0i32; 17];
        let mut err = [0u8; 512];
        let n = unsafe {
            ds4_bridge_eval_speculative_argmax(
                self.raw.as_ptr(),
                first,
                max_tokens,
                eos,
                accepted.as_mut_ptr(),
                accepted.len() as i32,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if n < 0 {
            return Err(fail(n, &err));
        }
        accepted.truncate(n as usize);
        for &tok in &accepted {
            self.host.commit_eval(tok);
        }
        Ok(accepted)
    }

    pub fn rewind(&mut self, pos: i32) {
        unsafe { ds4_bridge_session_rewind(self.raw.as_ptr(), pos) };
        self.host.rewind(pos);
    }

    pub fn invalidate(&mut self) {
        unsafe { ds4_bridge_session_invalidate(self.raw.as_ptr()) };
        self.host.invalidate();
    }

    pub fn rewrite_from_common(&self, prompt: &[i32], common: i32) -> RewriteKind {
        self.host.rewrite_from_common(prompt, common)
    }

    pub fn native_generation(&self) -> u64 {
        unsafe { ds4_bridge_session_generation(self.raw.as_ptr()) }
    }

    pub fn distributed_route_ready(&self) -> Result<bool> {
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_distributed_route_ready(
                self.raw.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        match rc {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(fail(rc, &err)),
        }
    }

    pub fn argmax(&self) -> i32 {
        unsafe { ds4_bridge_session_argmax(self.raw.as_ptr()) }
    }

    pub fn argmax_excluding(&self, excluded_id: i32) -> i32 {
        unsafe { ds4_bridge_session_argmax_excluding(self.raw.as_ptr(), excluded_id) }
    }

    /// Post-prefill distribution head, up to `k` entries (`k` clamps to the
    /// C CLI's 128). Empty when the backend keeps no logits.
    pub fn copy_logits(&self, vocab: usize) -> Result<Vec<f32>> {
        if vocab == 0 {
            return Err(Error {
                code: 1,
                message: "vocab is empty".into(),
            });
        }
        let mut out = vec![0.0f32; vocab];
        let n = unsafe {
            ds4_bridge_session_copy_logits(self.raw.as_ptr(), out.as_mut_ptr(), vocab as i32)
        };
        if n != vocab as i32 {
            return Err(Error {
                code: 1,
                message: "failed to copy session logits".into(),
            });
        }
        Ok(out)
    }

    pub fn output_head_bench(&self, iters: i32, path: Option<&str>) -> Result<()> {
        let c_path = match path {
            Some(p) => Some(cstring_path(p)?),
            None => None,
        };
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_output_head_bench(
                self.raw.as_ptr(),
                iters,
                c_path.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        Ok(())
    }

    pub fn top_logprobs(&self, k: usize) -> Vec<TokenScore> {
        const SCORE_CAP: usize = 128;
        let k = k.clamp(1, SCORE_CAP);
        let mut raw = vec![
            ds4_bridge_token_score {
                id: -1,
                logit: 0.0,
                logprob: 0.0,
            };
            k
        ];
        let n = unsafe {
            ds4_bridge_session_top_logprobs(self.raw.as_ptr(), raw.as_mut_ptr(), k as i32)
        };
        if n <= 0 {
            return Vec::new();
        }
        raw.truncate(n as usize);
        raw.into_iter()
            .map(|s| TokenScore {
                id: s.id,
                logit: s.logit,
                logprob: s.logprob,
            })
            .collect()
    }

    pub fn pos(&self) -> i32 {
        self.host.pos()
    }

    pub fn ctx(&self) -> i32 {
        self.host.ctx
    }

    /// C `ds4_session_graph_pending`: true while the S6 lazy graph alloc is
    /// still deferred. A pending session can be re-created at another ctx for
    /// free; a committed graph's capacity is spent.
    pub fn graph_pending(&self) -> bool {
        unsafe { ds4_bridge_session_graph_pending(self.raw.as_ptr()) != 0 }
    }

    pub fn power(&self) -> i32 {
        unsafe { ds4_bridge_session_power(self.raw.as_ptr()) }
    }

    pub fn set_power(&mut self, power_percent: i32) -> Result<()> {
        let rc = unsafe { ds4_bridge_session_set_power(self.raw.as_ptr(), power_percent) };
        if rc != 0 {
            return Err(Error {
                code: rc,
                message: "failed to set /power".into(),
            });
        }
        Ok(())
    }

    pub fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        unsafe {
            ds4_bridge_session_sample(self.raw.as_ptr(), temperature, top_k, top_p, min_p, rng)
        }
    }

    pub fn save_snapshot(&self, snapshot: &mut SessionSnapshot) -> Result<()> {
        snapshot.host = None;
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_save_snapshot(
                self.raw.as_ptr(),
                snapshot.raw.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        snapshot.host = Some(self.host.clone());
        Ok(())
    }

    pub fn load_snapshot(&mut self, snapshot: &SessionSnapshot) -> Result<()> {
        let saved = snapshot.host.as_ref().ok_or_else(|| Error {
            code: 1,
            message: "session snapshot is empty".into(),
        })?;
        if saved.family != self.host.family
            || saved.backend != self.host.backend
            || saved.ctx != self.host.ctx
            || saved.prefill_cap != self.host.prefill_cap
        {
            return Err(Error {
                code: 1,
                message: "session snapshot belongs to an incompatible session".into(),
            });
        }

        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_load_snapshot(
                self.raw.as_ptr(),
                snapshot.raw.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            self.host.clear_checkpoint_keep_generation();
            self.host.generation = self.native_generation();
            return Err(fail(rc, &err));
        }
        self.host = saved.clone();
        self.host.generation = self.native_generation();
        Ok(())
    }

    /// Native writes the full DSV4 file (header + tokens + GPU tail).
    /// Host re-reads the prefix and requires token identity to match the ledger.
    pub fn save_payload(&self, path: impl AsRef<Path>) -> Result<()> {
        if !self.host.valid {
            return Err(Error {
                code: 1,
                message: "session has no valid checkpoint to save".into(),
            });
        }
        save_payload_checked(
            self.raw,
            path.as_ref(),
            self.host.tokens(),
            self.host.family,
            self.host.ctx,
        )
    }

    /// Host validates the DSV4 prefix independently, then native restores the
    /// GPU/logits tail. Generation follows native (`ds4_session_load_payload`
    /// bumps it). Tokens come from the host-parsed prefix.
    pub fn load_payload(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|e| Error {
            code: 1,
            message: format!("failed to open session payload: {e}"),
        })?;
        let payload_bytes = file
            .metadata()
            .map_err(|e| Error {
                code: 1,
                message: format!("failed to measure session payload: {e}"),
            })?
            .len();
        let prefix = crate::payload::read_prefix_range(
            &mut file,
            0,
            payload_bytes,
            self.host.family,
            self.host.ctx,
        )
        .map_err(|e| Error {
            code: 1,
            message: e.to_string(),
        })?;
        let c_path = cstring_payload_path(path)?;
        let mut err = [0u8; 512];
        let generation_before = self.native_generation();
        let rc = unsafe {
            ds4_bridge_session_load_payload(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            let generation_after = self.native_generation();
            if generation_after != generation_before {
                self.host.clear_checkpoint_keep_generation();
            }
            self.host.generation = generation_after;
            return Err(fail(rc, &err));
        }
        self.host.apply_payload(&prefix).map_err(|e| Error {
            code: 1,
            message: e.message.to_string(),
        })?;
        self.host.generation = self.native_generation();
        Ok(())
    }

    /// Restore one DSV4 payload embedded in a larger file. Only the host
    /// prefix is read in Rust; the native loader consumes the bounded range.
    pub fn load_payload_range(
        &mut self,
        path: impl AsRef<Path>,
        offset: u64,
        length: u64,
    ) -> Result<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|e| Error {
            code: 1,
            message: format!("failed to open session payload range: {e}"),
        })?;
        let prefix = crate::payload::read_prefix_range(
            &mut file,
            offset,
            length,
            self.host.family,
            self.host.ctx,
        )
        .map_err(|e| Error {
            code: 1,
            message: e.to_string(),
        })?;
        let c_path = cstring_payload_path(path)?;
        let mut err = [0u8; 512];
        let generation_before = self.native_generation();
        let rc = unsafe {
            ds4_bridge_session_load_payload_range(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                offset,
                length,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            let generation_after = self.native_generation();
            if generation_after != generation_before {
                self.host.clear_checkpoint_keep_generation();
            }
            self.host.generation = generation_after;
            return Err(fail(rc, &err));
        }
        self.host.apply_payload(&prefix).map_err(|e| Error {
            code: 1,
            message: e.message.to_string(),
        })?;
        self.host.generation = self.native_generation();
        Ok(())
    }

    /// Worker KV shard: native writes only the requested layer range.
    pub fn save_layer_payload(
        &self,
        path: impl AsRef<Path>,
        layer_start: u32,
        layer_end: u32,
    ) -> Result<()> {
        let c_path = cstring_payload_path(path.as_ref())?;
        let mut err = [0u8; 512];
        // SAFETY: [Category 8 — FFI] `raw` is a live session from create; path
        // is a NUL-terminated CString; err is a stack buffer whose length is
        // passed to native. The bridge does not retain the pointers.
        let rc = unsafe {
            ds4_bridge_session_save_layer_payload(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                layer_start,
                layer_end,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        Ok(())
    }

    /// Restore one worker KV shard and replace the host token timeline.
    pub fn load_layer_payload(&mut self, req: LayerPayloadLoad<'_>) -> Result<()> {
        let n_tokens = u32::try_from(req.tokens.len()).map_err(|_| Error {
            code: 1,
            message: "layer payload token count exceeds u32".into(),
        })?;
        let c_path = cstring_payload_path(req.path)?;
        let mut err = [0u8; 512];
        // SAFETY: [Category 8 — FFI] session/path/err as in save_layer_payload.
        // `tokens` is borrowed for the call only; empty slices still yield a
        // non-null aligned pointer, matching the C `tokens != NULL` contract
        // when n_tokens == 0.
        let rc = unsafe {
            ds4_bridge_session_load_layer_payload(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                req.payload_bytes,
                req.tokens.as_ptr(),
                n_tokens,
                req.layer_start,
                req.layer_end,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        self.host.replace_checkpoint(req.tokens);
        Ok(())
    }
}

/// Inputs for [`Session::eval_layer_slice`].
pub struct LayerSliceEval<'a> {
    pub tokens: &'a [i32],
    pub pos0: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub input_hc: Option<&'a [f32]>,
    pub output_hc: Option<&'a mut [f32]>,
    pub logits: Option<&'a mut [f32]>,
}

/// Inputs for [`Session::load_layer_payload`].
pub struct LayerPayloadLoad<'a> {
    pub path: &'a Path,
    pub payload_bytes: u64,
    pub tokens: &'a [i32],
    pub layer_start: u32,
    pub layer_end: u32,
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: [Category 8 — FFI boundary]
        // `self.raw` is the unique owner from `ds4_bridge_session_create`,
        // or a test dangling pointer whose free is stubbed. Drop runs
        // once. C `ds4_session_free` is NULL-safe. `'m` forbids Model
        // Drop while this Session is live (session before model).
        unsafe { ds4_bridge_session_free(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[test]
    fn graph_fit_quote_copies_c_fields() {
        let raw = ds4_bridge_graph_fit_quote {
            fits: 1,
            fail_open: 0,
            need_bytes: 7_090_000_000,
            headroom_bytes: 256,
            avail_bytes: 5_460_000_000,
            deficit_bytes: 1_630_000_256,
        };
        let quote = GraphFitQuote::from_bridge(raw);
        assert!(quote.fits);
        assert!(!quote.fail_open);
        assert_eq!(quote.need_bytes, 7_090_000_000);
        assert_eq!(quote.headroom_bytes, 256);
        assert_eq!(quote.avail_bytes, 5_460_000_000);
        assert_eq!(quote.deficit_bytes, 1_630_000_256);
    }

    #[test]
    fn backend_codes_match_bridge_header() {
        assert_eq!(Backend::Cuda.to_c(), 0);
        assert_eq!(Backend::Metal.to_c(), 1);
        assert_eq!(Backend::Cpu.to_c(), 2);
    }

    #[test]
    fn token_buffer_round_trip() {
        let mut buf = TokenBuffer::new();
        buf.push(1);
        buf.push(2);
        assert_eq!(buf.as_slice(), &[1, 2]);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn token_buffer_insert_remove_truncate_match_c() {
        let mut buf = TokenBuffer::from_tokens(vec![1, 2, 3]);
        buf.insert(1, &[8, 9]);
        assert_eq!(buf.as_slice(), &[1, 8, 9, 2, 3]);
        buf.remove(1, 2);
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
        buf.truncate(1);
        assert_eq!(buf.as_slice(), &[1]);
        buf.insert(8, &[4]);
        assert_eq!(buf.as_slice(), &[1, 4]);
        buf.remove(4, 2);
        assert_eq!(buf.as_slice(), &[1, 4]);
        buf.insert(0, &[]);
        assert_eq!(buf.as_slice(), &[1, 4]);
    }

    #[test]
    fn path_rejects_embedded_nul() {
        let err = cstring_path("a\0b").unwrap_err();
        assert_eq!(err.code, 1);
        assert!(err.message.contains("NUL"));
    }

    #[test]
    fn model_open_tuning_matches_c_defaults_and_bounds() {
        let defaults = open_tuning(&[]).unwrap();
        assert!(!defaults.quality);
        assert!(!defaults.warm_weights);
        assert_eq!(defaults.power_percent, 100);

        let configured = open_tuning(&[
            ModelOpenOption::Quality,
            ModelOpenOption::WarmWeights,
            ModelOpenOption::PowerPercent(37),
            ModelOpenOption::Vision("vision.gguf".into()),
        ])
        .unwrap();
        assert!(configured.quality);
        assert!(configured.warm_weights);
        assert_eq!(configured.power_percent, 37);
        assert_eq!(configured.vision_path.as_deref(), Some("vision.gguf"));
        assert!(open_tuning(&[ModelOpenOption::PowerPercent(0)]).is_err());
        assert!(open_tuning(&[ModelOpenOption::PowerPercent(101)]).is_err());
        assert!(open_tuning(&[ModelOpenOption::Vision(String::new())]).is_err());
    }

    #[test]
    fn model_open_tuning_applies_directional_steering() {
        let defaults = open_tuning(&[]).unwrap();
        assert!(defaults.steering_file.is_none());
        assert_eq!(defaults.steering_attn, 0.0);
        assert_eq!(defaults.steering_ffn, 0.0);

        let configured = open_tuning(&[
            ModelOpenOption::SteeringFile("dirs.bin".into()),
            ModelOpenOption::SteeringAttn(0.5),
            ModelOpenOption::SteeringFfn(-1.25),
        ])
        .unwrap();
        assert_eq!(configured.steering_file.as_deref(), Some("dirs.bin"));
        assert_eq!(configured.steering_attn, 0.5);
        assert_eq!(configured.steering_ffn, -1.25);
        assert_eq!(
            open_tuning(&[ModelOpenOption::SteeringFfn(1.0)])
                .unwrap_err()
                .message,
            "directional steering needs --dir-steering-file"
        );
        assert!(open_tuning(&[ModelOpenOption::SteeringAttn(101.0)]).is_err());
    }

    #[no_mangle]
    extern "C" fn ds4_bridge_session_ctx(s: *mut ds4_bridge_session) -> i32 {
        assert!(!s.is_null());
        4096
    }

    #[test]
    fn session_ctx_uses_host_ledger_not_native_scratch() {
        let session = std::mem::ManuallyDrop::new(Session {
            raw: NonNull::<ds4_bridge_session>::dangling(),
            host: SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, 8192, 1),
            _model: PhantomData,
            _not_send: PhantomData,
        });

        assert_eq!(session.ctx(), 8192);
        assert_eq!(session.ctx(), session.host().ctx);
    }

    #[test]
    fn ledger_ctx_adopts_native_effective_only_at_create_boundary() {
        assert_eq!(ledger_ctx(8192, 4096), 4096);
        assert_eq!(ledger_ctx(8192, 0), 8192);
        assert_eq!(ledger_ctx(8192, -1), 8192);
    }

    #[no_mangle]
    extern "C" fn ds4_bridge_session_eval_layer_slice(
        s: *mut ds4_bridge_session,
        _tokens: *const i32,
        _n_tokens: u32,
        _pos0: u32,
        _layer_start: u32,
        _layer_end: u32,
        _input_hc: *const f32,
        _output_hc: *mut f32,
        _output_logits: i32,
        _logits: *mut f32,
        _err: *mut c_char,
        _errlen: usize,
    ) -> i32 {
        i32::from(s.is_null())
    }

    #[test]
    fn eval_layer_slice_extends_host_timeline_from_pos0() {
        let mut session = std::mem::ManuallyDrop::new(Session {
            raw: NonNull::<ds4_bridge_session>::dangling(),
            host: SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, 4096, 1),
            _model: PhantomData,
            _not_send: PhantomData,
        });
        session.host.replace_checkpoint(&[1, 2, 3]);
        session
            .eval_layer_slice(LayerSliceEval {
                tokens: &[4, 5],
                pos0: 2,
                layer_start: 20,
                layer_end: 42,
                input_hc: None,
                output_hc: None,
                logits: None,
            })
            .unwrap();
        assert_eq!(session.host().tokens(), &[1, 2, 4, 5]);
    }

    #[no_mangle]
    extern "C" fn ds4_bridge_session_layer_slice_reset(
        s: *mut ds4_bridge_session,
        _err: *mut c_char,
        _errlen: usize,
    ) -> i32 {
        i32::from(s.is_null())
    }

    #[test]
    fn layer_slice_reset_clears_host_timeline() {
        let mut session = std::mem::ManuallyDrop::new(Session {
            raw: NonNull::<ds4_bridge_session>::dangling(),
            host: SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, 4096, 1),
            _model: PhantomData,
            _not_send: PhantomData,
        });
        session.host.replace_checkpoint(&[1, 2, 3]);
        session.layer_slice_reset().unwrap();
        assert!(session.host().tokens().is_empty());
        assert_eq!(session.host().live_len(), 0);
    }

    #[test]
    fn load_layer_payload_replaces_host_checkpoint() {
        let mut session = std::mem::ManuallyDrop::new(Session {
            raw: NonNull::<ds4_bridge_session>::dangling(),
            host: SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, 4096, 1),
            _model: PhantomData,
            _not_send: PhantomData,
        });
        let path = std::env::temp_dir().join(format!(
            "ds4-layer-payload-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, b"x").unwrap();
        session
            .load_layer_payload(LayerPayloadLoad {
                path: &path,
                payload_bytes: 1,
                tokens: &[9, 8],
                layer_start: 0,
                layer_end: 1,
            })
            .unwrap();
        assert_eq!(session.host().tokens(), &[9, 8]);
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn payload_path_preserves_non_utf8_bytes() {
        use std::os::unix::ffi::OsStringExt as _;

        let raw = b"/tmp/ds4-payload-\xff".to_vec();
        let path = std::path::PathBuf::from(std::ffi::OsString::from_vec(raw.clone()));

        let c_path = cstring_payload_path(&path).unwrap();
        assert_eq!(c_path.as_bytes(), raw);
    }

    #[cfg(unix)]
    #[test]
    fn payload_path_rejects_embedded_nul_with_specific_error() {
        use std::os::unix::ffi::OsStringExt as _;

        let path = std::path::PathBuf::from(std::ffi::OsString::from_vec(
            b"/tmp/ds4-payload-\0suffix".to_vec(),
        ));

        let err = cstring_payload_path(&path).unwrap_err();
        assert_eq!(err.code, 1);
        assert_eq!(err.message, "payload path contains NUL");
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum DestroyKind {
        Session,
        SiblingMtp,
        SiblingDspark,
        Base,
    }

    fn c_destroy_order(
        has_session: bool,
        mtp_ready: bool,
        dspark_ready: bool,
    ) -> [Option<DestroyKind>; 4] {
        let mut steps = [None; 4];
        let mut i = 0usize;
        if has_session {
            steps[i] = Some(DestroyKind::Session);
            i += 1;
        }
        if mtp_ready {
            steps[i] = Some(DestroyKind::SiblingMtp);
            i += 1;
        }
        if dspark_ready {
            steps[i] = Some(DestroyKind::SiblingDspark);
            i += 1;
        }
        steps[i] = Some(DestroyKind::Base);
        steps
    }

    thread_local! {
        static SESSION_FREE: Cell<u32> = const { Cell::new(0) };
        static MODEL_FREE: Cell<u32> = const { Cell::new(0) };
        static DESTROY_LOG: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    }

    #[no_mangle]
    extern "C" fn ds4_bridge_session_free(_s: *mut ds4_bridge_session) {
        SESSION_FREE.with(|c| c.set(c.get() + 1));
        DESTROY_LOG.with(|log| log.borrow_mut().push("session"));
    }

    #[no_mangle]
    extern "C" fn ds4_bridge_model_free(_m: *mut ds4_bridge_model) {
        MODEL_FREE.with(|c| c.set(c.get() + 1));
        DESTROY_LOG.with(|log| log.borrow_mut().push("model"));
    }

    fn reset_destroy_stubs() {
        SESSION_FREE.with(|c| c.set(0));
        MODEL_FREE.with(|c| c.set(0));
        DESTROY_LOG.with(|log| log.borrow_mut().clear());
    }

    #[test]
    fn c_destroy_order_session_before_model_siblings_before_base() {
        assert_eq!(
            c_destroy_order(true, true, true),
            [
                Some(DestroyKind::Session),
                Some(DestroyKind::SiblingMtp),
                Some(DestroyKind::SiblingDspark),
                Some(DestroyKind::Base),
            ]
        );
        assert_eq!(
            c_destroy_order(true, false, false),
            [
                Some(DestroyKind::Session),
                Some(DestroyKind::Base),
                None,
                None,
            ]
        );
        assert_eq!(
            c_destroy_order(false, true, false),
            [
                Some(DestroyKind::SiblingMtp),
                Some(DestroyKind::Base),
                None,
                None,
            ]
        );
    }

    #[test]
    fn drop_order_session_before_model_via_stub_counters() {
        reset_destroy_stubs();
        let session = Session {
            raw: NonNull::<ds4_bridge_session>::dangling(),
            host: SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, 4096, 1),
            _model: PhantomData,
            _not_send: PhantomData,
        };
        drop(session);
        drop_model_native(NonNull::<ds4_bridge_model>::dangling());
        assert_eq!(SESSION_FREE.with(Cell::get), 1);
        assert_eq!(MODEL_FREE.with(Cell::get), 1);
        DESTROY_LOG.with(|log| {
            assert_eq!(*log.borrow(), ["session", "model"]);
        });
    }
}
