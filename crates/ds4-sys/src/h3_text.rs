//! The MiniMax-H3 prompt-encoder device ABI (`native/h3/h3_text_gpu.h`).
//!
//! Additive family bindings: a separate context, a separate tensor type and
//! `h3t_` names, so no existing family's native entry point is shared or
//! changed. The safe wrappers take and return bytes only; the host loop that
//! orders the ops is `ds4_core::minimax_h3::text_forward`.
//!
//! Do not add a wrapper here that changes an argument's width or order: the
//! native side validates shapes itself and rejects a mismatch, and the
//! reference's arithmetic depends on the exact values passed.

use std::ffi::c_char;
use std::os::raw::c_int;
use std::ptr::NonNull;

/// `error[512]` in every reference entry point.
const ERROR_SIZE: usize = 512;

#[repr(C)]
struct Context {
    _private: [u8; 0],
}

#[repr(C)]
struct CTensor {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn h3t_create(error: *mut c_char, error_size: usize) -> *mut Context;
    fn h3t_destroy(context: *mut Context);

    fn h3t_alloc(
        context: *mut Context,
        dtype: c_int,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> *mut CTensor;
    fn h3t_free(tensor: *mut CTensor);

    fn h3t_upload(
        context: *mut Context,
        tensor: *mut CTensor,
        host: *const std::ffi::c_void,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    fn h3t_read(
        context: *mut Context,
        host: *mut std::ffi::c_void,
        tensor: *const CTensor,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_embed(
        context: *mut Context,
        output: *mut CTensor,
        weight: *const CTensor,
        ids: *const u32,
        tokens: u32,
        vocab: u32,
        width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_rms_norm(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        rows: u32,
        width: u32,
        epsilon: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_linear(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_head_rms_norm(
        context: *mut Context,
        tensor: *mut CTensor,
        weight: *const CTensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        epsilon: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_rope_text(
        context: *mut Context,
        query: *mut CTensor,
        key: *mut CTensor,
        cosines: *const CTensor,
        sines: *const CTensor,
        sequence: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_gqa_causal(
        context: *mut Context,
        output: *mut CTensor,
        query: *const CTensor,
        key: *const CTensor,
        value: *const CTensor,
        sequence: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        scale: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_add(
        context: *mut Context,
        output: *mut CTensor,
        left: *const CTensor,
        right: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_silu_mul(
        context: *mut Context,
        output: *mut CTensor,
        gate: *const CTensor,
        up: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3t_sync(context: *mut Context, error: *mut c_char, error_size: usize) -> c_int;
}

/// `enum h3t_dtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F32,
}

impl Dtype {
    fn code(self) -> c_int {
        match self {
            Dtype::Bf16 => 0,
            Dtype::F32 => 1,
        }
    }

    /// Bytes per element, so a host slice can be checked before a copy.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bf16 => 2,
            Dtype::F32 => 4,
        }
    }
}

/// A device buffer owned by one [`Gpu`].
#[derive(Debug)]
pub struct Tensor {
    raw: NonNull<CTensor>,
    dtype: Dtype,
    elements: usize,
}

impl Tensor {
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Bytes this buffer holds.
    pub fn bytes(&self) -> usize {
        self.elements * self.dtype.size()
    }

    fn raw(&self) -> *mut CTensor {
        self.raw.as_ptr()
    }
}

/// One CUDA stream and one cuBLASLt handle: the reference's `h3_gpu`.
#[derive(Debug)]
pub struct Gpu {
    raw: NonNull<Context>,
}

impl Gpu {
    pub fn create() -> Result<Gpu, String> {
        let mut error = [0 as c_char; ERROR_SIZE];
        let raw = unsafe { h3t_create(error.as_mut_ptr(), ERROR_SIZE) };

        NonNull::new(raw)
            .map(|raw| Gpu { raw })
            .ok_or_else(|| message(&error))
    }

    pub fn alloc(&self, dtype: Dtype, elements: usize) -> Result<Tensor, String> {
        let mut error = [0 as c_char; ERROR_SIZE];
        let raw = unsafe {
            h3t_alloc(
                self.raw.as_ptr(),
                dtype.code(),
                elements,
                error.as_mut_ptr(),
                ERROR_SIZE,
            )
        };
        let raw = NonNull::new(raw).ok_or_else(|| message(&error))?;

        Ok(Tensor {
            raw,
            dtype,
            elements,
        })
    }

    /// Copy `bytes` into a BF16 or F32 tensor from the host.
    pub fn upload(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() != tensor.bytes() {
            return Err(format!(
                "upload of {} bytes into a {} byte tensor",
                bytes.len(),
                tensor.bytes()
            ));
        }
        self.call(
            |error, size| unsafe {
                h3t_upload(
                    self.raw.as_ptr(),
                    tensor.raw(),
                    bytes.as_ptr().cast(),
                    error,
                    size,
                )
            },
            "upload",
        )
    }

    /// Copy a tensor back to the host, waiting for the stream.
    pub fn read(&self, tensor: &Tensor, out: &mut [u8]) -> Result<(), String> {
        if out.len() != tensor.bytes() {
            return Err(format!(
                "read of {} bytes from a {} byte tensor",
                out.len(),
                tensor.bytes()
            ));
        }
        self.call(
            |error, size| unsafe {
                h3t_read(
                    self.raw.as_ptr(),
                    out.as_mut_ptr().cast(),
                    tensor.raw(),
                    error,
                    size,
                )
            },
            "read",
        )
    }

    /// `h3_gpu_embedding_bf16`: one gathered row per token id.
    pub fn embed(
        &self,
        output: &Tensor,
        weight: &Tensor,
        ids: &[u32],
        vocab: u32,
        width: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3t_embed(
                    self.raw.as_ptr(),
                    output.raw(),
                    weight.raw(),
                    ids.as_ptr(),
                    ids.len() as u32,
                    vocab,
                    width,
                    error,
                    size,
                )
            },
            "embedding",
        )
    }

    /// `h3_gpu_rms_norm_bf16`.
    pub fn rms_norm(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3t_rms_norm(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    rows,
                    width,
                    epsilon,
                    error,
                    size,
                )
            },
            "RMS norm",
        )
    }

    /// `h3_gpu_linear_bf16`.
    pub fn linear(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3t_linear(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    rows,
                    input_dim,
                    output_dim,
                    error,
                    size,
                )
            },
            "linear",
        )
    }

    /// `h3_gpu_head_rms_norm_bf16`, in place.
    pub fn head_rms_norm(
        &self,
        tensor: &Tensor,
        weight: &Tensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3t_head_rms_norm(
                    self.raw.as_ptr(),
                    tensor.raw(),
                    weight.raw(),
                    sequence,
                    heads,
                    head_dim,
                    epsilon,
                    error,
                    size,
                )
            },
            "head RMS norm",
        )
    }

    /// `h3_gpu_rope_text_bf16`, in place on query and key.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_text(
        &self,
        query: &Tensor,
        key: &Tensor,
        cosines: &Tensor,
        sines: &Tensor,
        sequence: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3t_rope_text(
                    self.raw.as_ptr(),
                    query.raw(),
                    key.raw(),
                    cosines.raw(),
                    sines.raw(),
                    sequence,
                    query_heads,
                    kv_heads,
                    head_dim,
                    error,
                    size,
                )
            },
            "text RoPE",
        )
    }

    /// `h3_gpu_gqa_causal_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_causal(
        &self,
        output: &Tensor,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        sequence: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3t_gqa_causal(
                    self.raw.as_ptr(),
                    output.raw(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    sequence,
                    query_heads,
                    kv_heads,
                    head_dim,
                    scale,
                    error,
                    size,
                )
            },
            "causal GQA",
        )
    }

    /// `h3_gpu_add_bf16`.
    pub fn add(
        &self,
        output: &Tensor,
        left: &Tensor,
        right: &Tensor,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3t_add(
                    self.raw.as_ptr(),
                    output.raw(),
                    left.raw(),
                    right.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "add",
        )
    }

    /// `h3_gpu_silu_mul_bf16`.
    pub fn silu_mul(
        &self,
        output: &Tensor,
        gate: &Tensor,
        up: &Tensor,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3t_silu_mul(
                    self.raw.as_ptr(),
                    output.raw(),
                    gate.raw(),
                    up.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "SiLU multiply",
        )
    }

    /// Wait for every queued op.
    pub fn sync(&self) -> Result<(), String> {
        self.call(
            |error, size| unsafe { h3t_sync(self.raw.as_ptr(), error, size) },
            "sync",
        )
    }

    fn call<F>(&self, mut operation: F, label: &str) -> Result<(), String>
    where
        F: FnMut(*mut c_char, usize) -> c_int,
    {
        let mut error = [0 as c_char; ERROR_SIZE];

        if operation(error.as_mut_ptr(), ERROR_SIZE) == 1 {
            return Ok(());
        }

        Err(format!("{label}: {}", message(&error)))
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe { h3t_destroy(self.raw.as_ptr()) }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { h3t_free(self.raw.as_ptr()) }
    }
}

/// The C string in `error`, or a placeholder if the native side wrote none.
fn message(error: &[c_char; ERROR_SIZE]) -> String {
    let end = error
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(ERROR_SIZE);
    let bytes: Vec<u8> = error[..end].iter().map(|byte| *byte as u8).collect();
    let text = String::from_utf8_lossy(&bytes).into_owned();

    if text.is_empty() {
        "native h3 call failed without a message".to_string()
    } else {
        text
    }
}
