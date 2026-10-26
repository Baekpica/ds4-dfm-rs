//! The MiniMax-H3 vision-tower device ABI (`native/h3/h3_vision_gpu.h`).
//!
//! Additive family bindings: a separate context, a separate tensor type and
//! `h3v_` names, so no existing family's native entry point is shared or
//! changed. The safe wrappers take and return bytes only; the host loop that
//! orders the ops is `ds4_core::minimax_h3::vision_forward`.
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
    fn h3v_create(error: *mut c_char, error_size: usize) -> *mut Context;
    fn h3v_destroy(context: *mut Context);

    fn h3v_alloc(
        context: *mut Context,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> *mut CTensor;
    fn h3v_free(tensor: *mut CTensor);

    fn h3v_upload(
        context: *mut Context,
        tensor: *mut CTensor,
        host: *const std::ffi::c_void,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    fn h3v_read(
        context: *mut Context,
        host: *mut std::ffi::c_void,
        tensor: *const CTensor,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3v_linear(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3v_layer_norm(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        rows: u32,
        width: u32,
        epsilon: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3v_gelu(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        elements: u32,
        approximate: c_int,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3v_add(
        context: *mut Context,
        output: *mut CTensor,
        left: *const CTensor,
        right: *const CTensor,
        elements: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3v_qkv_rope(
        context: *mut Context,
        query: *mut CTensor,
        key: *mut CTensor,
        value: *mut CTensor,
        qkv: *const CTensor,
        cosines: *const CTensor,
        sines: *const CTensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        rope_half: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3v_sdpa(
        context: *mut Context,
        output: *mut CTensor,
        query: *const CTensor,
        key: *const CTensor,
        value: *const CTensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3v_sync(context: *mut Context, error: *mut c_char, error_size: usize) -> c_int;
}

/// A BF16 device buffer owned by one [`VisionGpu`].
#[derive(Debug)]
pub struct VisionTensor {
    raw: NonNull<CTensor>,
    elements: usize,
}

impl VisionTensor {
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Bytes this buffer holds.
    pub fn bytes(&self) -> usize {
        self.elements * 2
    }

    fn raw(&self) -> *mut CTensor {
        self.raw.as_ptr()
    }
}

/// One CUDA stream and one cuBLASLt handle: the reference's `h3_gpu`.
#[derive(Debug)]
pub struct VisionGpu {
    raw: NonNull<Context>,
}

impl VisionGpu {
    pub fn create() -> Result<VisionGpu, String> {
        let mut error = [0 as c_char; ERROR_SIZE];
        let raw = unsafe { h3v_create(error.as_mut_ptr(), ERROR_SIZE) };

        NonNull::new(raw)
            .map(|raw| VisionGpu { raw })
            .ok_or_else(|| message(&error))
    }

    pub fn alloc(&self, elements: usize) -> Result<VisionTensor, String> {
        let mut error = [0 as c_char; ERROR_SIZE];
        let raw = unsafe { h3v_alloc(self.raw.as_ptr(), elements, error.as_mut_ptr(), ERROR_SIZE) };
        let raw = NonNull::new(raw).ok_or_else(|| message(&error))?;

        Ok(VisionTensor { raw, elements })
    }

    /// Copy BF16 bytes into a tensor from the host.
    pub fn upload(&self, tensor: &VisionTensor, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() != tensor.bytes() {
            return Err(format!(
                "upload of {} bytes into a {} byte tensor",
                bytes.len(),
                tensor.bytes()
            ));
        }
        self.call(
            |error, size| unsafe {
                h3v_upload(
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
    pub fn read(&self, tensor: &VisionTensor, out: &mut [u8]) -> Result<(), String> {
        if out.len() != tensor.bytes() {
            return Err(format!(
                "read of {} bytes from a {} byte tensor",
                out.len(),
                tensor.bytes()
            ));
        }
        self.call(
            |error, size| unsafe {
                h3v_read(
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

    /// `h3_gpu_linear_bf16` with the bias kernel after it.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        output: &VisionTensor,
        input: &VisionTensor,
        weight: &VisionTensor,
        bias: Option<&VisionTensor>,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<(), String> {
        let bias = bias.map_or(std::ptr::null_mut(), VisionTensor::raw);
        self.call(
            |error, size| unsafe {
                h3v_linear(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias,
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

    /// `h3_gpu_layer_norm_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        output: &VisionTensor,
        input: &VisionTensor,
        weight: &VisionTensor,
        bias: &VisionTensor,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3v_layer_norm(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias.raw(),
                    rows,
                    width,
                    epsilon,
                    error,
                    size,
                )
            },
            "layer norm",
        )
    }

    /// `h3_gpu_gelu_bf16`; `approximate` selects the block branch.
    pub fn gelu(
        &self,
        output: &VisionTensor,
        input: &VisionTensor,
        elements: u32,
        approximate: bool,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3v_gelu(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    elements,
                    c_int::from(approximate),
                    error,
                    size,
                )
            },
            "GELU",
        )
    }

    /// `h3_gpu_add_bf16`.
    pub fn add(
        &self,
        output: &VisionTensor,
        left: &VisionTensor,
        right: &VisionTensor,
        elements: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3v_add(
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

    /// `h3_gpu_vision_qkv_rope_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_rope(
        &self,
        query: &VisionTensor,
        key: &VisionTensor,
        value: &VisionTensor,
        qkv: &VisionTensor,
        cosines: &VisionTensor,
        sines: &VisionTensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        rope_half: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3v_qkv_rope(
                    self.raw.as_ptr(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    qkv.raw(),
                    cosines.raw(),
                    sines.raw(),
                    sequence,
                    heads,
                    head_dim,
                    rope_half,
                    error,
                    size,
                )
            },
            "vision QKV RoPE",
        )
    }

    /// `h3_gpu_sdpa_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        &self,
        output: &VisionTensor,
        query: &VisionTensor,
        key: &VisionTensor,
        value: &VisionTensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3v_sdpa(
                    self.raw.as_ptr(),
                    output.raw(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    sequence,
                    heads,
                    head_dim,
                    scale,
                    error,
                    size,
                )
            },
            "attention",
        )
    }

    /// Wait for every queued op.
    pub fn sync(&self) -> Result<(), String> {
        self.call(
            |error, size| unsafe { h3v_sync(self.raw.as_ptr(), error, size) },
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

impl Drop for VisionGpu {
    fn drop(&mut self) {
        unsafe { h3v_destroy(self.raw.as_ptr()) }
    }
}

impl Drop for VisionTensor {
    fn drop(&mut self) {
        unsafe { h3v_free(self.raw.as_ptr()) }
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
        "no error text".to_string()
    } else {
        text
    }
}
