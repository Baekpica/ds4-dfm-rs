//! The MiniMax-H3 DiT device ABI (`native/h3/h3_dit_gpu.h`).
//!
//! Additive family bindings: a separate context, a separate tensor type and
//! `h3d_` names, so no existing family's native entry point is shared or
//! changed. The safe wrappers take and return bytes only; the host loop that
//! orders the ops is `ds4_core::minimax_h3::dit`.
//!
//! The wrappers mirror the reference's own argument order (`h3_gpu.h`) rather
//! than inventing a friendlier one: a swapped shape argument would still be
//! accepted by cuBLASLt and would silently change the graph.

use std::ffi::c_char;
use std::os::raw::c_int;
use std::ptr::NonNull;

/// `error[1024]` in the probe; the native side writes at most this much.
const ERROR_SIZE: usize = 1024;

#[repr(C)]
struct Context {
    _private: [u8; 0],
}

#[repr(C)]
struct CTensor {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn h3d_create(error: *mut c_char, error_size: usize) -> *mut Context;
    fn h3d_destroy(context: *mut Context);

    fn h3d_alloc(
        context: *mut Context,
        dtype: c_int,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> *mut CTensor;
    fn h3d_free(tensor: *mut CTensor);

    fn h3d_upload(
        context: *mut Context,
        tensor: *mut CTensor,
        host: *const std::ffi::c_void,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    fn h3d_upload_at(
        context: *mut Context,
        tensor: *mut CTensor,
        element: usize,
        host: *const std::ffi::c_void,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
    fn h3d_read(
        context: *mut Context,
        host: *mut std::ffi::c_void,
        tensor: *const CTensor,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_linear(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        dtype: c_int,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_patch_linear(
        context: *mut Context,
        output: *mut CTensor,
        output_element: usize,
        input: *const CTensor,
        input_element: usize,
        weight: *const CTensor,
        bias: *const CTensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_patch_linear_map(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        row_map: *const CTensor,
        output_rows: u32,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_adaln(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        input_element: usize,
        norm_weight: *const CTensor,
        modulation: *const CTensor,
        row_map: *const CTensor,
        rows: u32,
        width: u32,
        slots: u32,
        shift_slot: u32,
        scale_slot: u32,
        epsilon: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_gate(
        context: *mut Context,
        output: *mut CTensor,
        residual: *const CTensor,
        branch: *const CTensor,
        modulation: *const CTensor,
        row_map: *const CTensor,
        rows: u32,
        width: u32,
        slots: u32,
        gate_slot: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_rms_norm(
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

    fn h3d_swiglu(
        context: *mut Context,
        output: *mut CTensor,
        fused: *const CTensor,
        rows: u32,
        width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_qkv_rope(
        context: *mut Context,
        query: *mut CTensor,
        key: *mut CTensor,
        value: *mut CTensor,
        qkv: *const CTensor,
        q_norm: *const CTensor,
        k_norm: *const CTensor,
        rope_cos: *const CTensor,
        rope_sin: *const CTensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        rope_half: u32,
        epsilon: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_sdpa(
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

    fn h3d_add(
        context: *mut Context,
        output: *mut CTensor,
        left: *const CTensor,
        right: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_cast_f32_to_bf16(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_silu_f32(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_silu_bf16(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_copy(
        context: *mut Context,
        destination: *mut CTensor,
        destination_element: usize,
        source: *const CTensor,
        source_element: usize,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_sync(context: *mut Context, error: *mut c_char, error_size: usize) -> c_int;

    fn h3d_copy_f32(
        context: *mut Context,
        destination: *mut CTensor,
        destination_element: usize,
        source: *const CTensor,
        source_element: usize,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_conv1d(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        batch: u32,
        length: u32,
        input_channels: u32,
        output_channels: u32,
        kernel: u32,
        stride: u32,
        padding: u32,
        dilation: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_conv_transpose1d(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        batch: u32,
        length: u32,
        input_channels: u32,
        output_channels: u32,
        kernel: u32,
        stride: u32,
        padding: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_weight_norm(
        context: *mut Context,
        output: *mut CTensor,
        vector: *const CTensor,
        magnitude: *const CTensor,
        outer: u32,
        inner: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_alias_free_snake(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        alpha_log: *const CTensor,
        beta_log: *const CTensor,
        upsample_filter: *const CTensor,
        downsample_filter: *const CTensor,
        batch: u32,
        length: u32,
        channels: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_add_scaled(
        context: *mut Context,
        output: *mut CTensor,
        left: *const CTensor,
        right: *const CTensor,
        left_scale: f32,
        right_scale: f32,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_clip(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        elements: usize,
        minimum: f32,
        maximum: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_sdpa_f32(
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

    fn h3d_rms_norm_f32(
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

    fn h3d_layer_norm_f32(
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

    fn h3d_swiglu_f32(
        context: *mut Context,
        output: *mut CTensor,
        fused: *const CTensor,
        rows: u32,
        width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_scale_add_f32(
        context: *mut Context,
        output: *mut CTensor,
        residual: *const CTensor,
        branch: *const CTensor,
        scale: *const CTensor,
        rows: u32,
        width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_video_qkv_rope_f32(
        context: *mut Context,
        query: *mut CTensor,
        key: *mut CTensor,
        value: *mut CTensor,
        qkv: *const CTensor,
        rope_cos: *const CTensor,
        rope_sin: *const CTensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        rope_half: u32,
        epsilon: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn h3d_vae_encoder_pad(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        batch: u32,
        depth: u32,
        height: u32,
        width: u32,
        channels: u32,
        depth_front: u32,
        height_before: u32,
        height_after: u32,
        width_before: u32,
        width_after: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn h3d_conv3d(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        batch: u32,
        depth: u32,
        height: u32,
        width: u32,
        input_channels: u32,
        output_channels: u32,
        kernel_depth: u32,
        kernel_height: u32,
        kernel_width: u32,
        stride_depth: u32,
        stride_height: u32,
        stride_width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn h3d_group_norm_silu(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        batch: u32,
        depth: u32,
        height: u32,
        width: u32,
        channels: u32,
        groups: u32,
        epsilon: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
}

/// `h3d_dtype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F32,
    U32,
}

impl Dtype {
    fn code(self) -> c_int {
        match self {
            Dtype::Bf16 => 0,
            Dtype::F32 => 1,
            Dtype::U32 => 2,
        }
    }

    /// Bytes per element, so a host slice can be checked before a copy.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bf16 => 2,
            Dtype::F32 | Dtype::U32 => 4,
        }
    }
}

/// One device buffer. The element count is fixed at allocation, as in the
/// reference; ranges are checked by the native side on every call.
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
        let raw = unsafe { h3d_create(error.as_mut_ptr(), ERROR_SIZE) };

        NonNull::new(raw)
            .map(|raw| Gpu { raw })
            .ok_or_else(|| message(&error))
    }

    pub fn alloc(&self, dtype: Dtype, elements: usize) -> Result<Tensor, String> {
        let mut error = [0 as c_char; ERROR_SIZE];
        let raw = unsafe {
            h3d_alloc(
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

    /// Copy a whole tensor from the host.
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
                h3d_upload(
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

    /// Copy `bytes` into the tensor starting at element `element`.
    pub fn upload_at(&self, tensor: &Tensor, element: usize, bytes: &[u8]) -> Result<(), String> {
        let elements = bytes.len() / tensor.dtype().size();
        if !bytes.len().is_multiple_of(tensor.dtype().size())
            || element > tensor.elements()
            || elements > tensor.elements() - element
        {
            return Err(format!(
                "{elements} elements at {element} do not fit a {} element tensor",
                tensor.elements()
            ));
        }
        self.call(
            |error, size| unsafe {
                h3d_upload_at(
                    self.raw.as_ptr(),
                    tensor.raw(),
                    element,
                    bytes.as_ptr().cast(),
                    elements,
                    error,
                    size,
                )
            },
            "upload range",
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
                h3d_read(
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

    /// `h3_gpu_linear_bf16` (or `h3_gpu_linear_f32`): `rows x input_dim` by
    /// `output_dim x input_dim`, optional bias.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        dtype: Dtype,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_linear(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias.map_or(std::ptr::null(), |bias| bias.raw()),
                    rows,
                    input_dim,
                    output_dim,
                    dtype.code(),
                    error,
                    size,
                )
            },
            "linear",
        )
    }

    /// `h3_gpu_patch_linear_bf16_offset`: F32 matmul, bias and one BF16 cast.
    #[allow(clippy::too_many_arguments)]
    pub fn patch_linear(
        &self,
        output: &Tensor,
        output_element: usize,
        input: &Tensor,
        input_element: usize,
        weight: &Tensor,
        bias: &Tensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_patch_linear(
                    self.raw.as_ptr(),
                    output.raw(),
                    output_element,
                    input.raw(),
                    input_element,
                    weight.raw(),
                    bias.raw(),
                    rows,
                    input_dim,
                    output_dim,
                    error,
                    size,
                )
            },
            "patch linear",
        )
    }

    /// `h3_gpu_patch_linear_bf16_map`: the projection scattered by `row_map`.
    #[allow(clippy::too_many_arguments)]
    pub fn patch_linear_map(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        row_map: &Tensor,
        output_rows: u32,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_patch_linear_map(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias.raw(),
                    row_map.raw(),
                    output_rows,
                    rows,
                    input_dim,
                    output_dim,
                    error,
                    size,
                )
            },
            "mapped patch linear",
        )
    }

    /// `h3_gpu_adaln_bf16_offset`.
    #[allow(clippy::too_many_arguments)]
    pub fn adaln(
        &self,
        output: &Tensor,
        input: &Tensor,
        input_element: usize,
        norm_weight: &Tensor,
        modulation: &Tensor,
        row_map: &Tensor,
        rows: u32,
        width: u32,
        slots: u32,
        shift_slot: u32,
        scale_slot: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_adaln(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    input_element,
                    norm_weight.raw(),
                    modulation.raw(),
                    row_map.raw(),
                    rows,
                    width,
                    slots,
                    shift_slot,
                    scale_slot,
                    epsilon,
                    error,
                    size,
                )
            },
            "adaln",
        )
    }

    /// `h3_gpu_gate_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub fn gate(
        &self,
        output: &Tensor,
        residual: &Tensor,
        branch: &Tensor,
        modulation: &Tensor,
        row_map: &Tensor,
        rows: u32,
        width: u32,
        slots: u32,
        gate_slot: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_gate(
                    self.raw.as_ptr(),
                    output.raw(),
                    residual.raw(),
                    branch.raw(),
                    modulation.raw(),
                    row_map.raw(),
                    rows,
                    width,
                    slots,
                    gate_slot,
                    error,
                    size,
                )
            },
            "gate",
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
                h3d_rms_norm(
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
            "rms norm",
        )
    }

    /// `h3_gpu_swiglu_bf16`.
    pub fn swiglu(
        &self,
        output: &Tensor,
        fused: &Tensor,
        rows: u32,
        width: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_swiglu(
                    self.raw.as_ptr(),
                    output.raw(),
                    fused.raw(),
                    rows,
                    width,
                    error,
                    size,
                )
            },
            "swiglu",
        )
    }

    /// `h3_gpu_grouped_qkv_rope_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_rope(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        qkv: &Tensor,
        q_norm: &Tensor,
        k_norm: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        rope_half: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_qkv_rope(
                    self.raw.as_ptr(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    qkv.raw(),
                    q_norm.raw(),
                    k_norm.raw(),
                    rope_cos.raw(),
                    rope_sin.raw(),
                    sequence,
                    heads,
                    head_dim,
                    rope_half,
                    epsilon,
                    error,
                    size,
                )
            },
            "qkv rope",
        )
    }

    /// `h3_gpu_sdpa_bf16`, the tiled head-major-input dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        &self,
        output: &Tensor,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_sdpa(
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

    /// `h3_gpu_add_bf16`.
    pub fn add(&self, output: &Tensor, left: &Tensor, right: &Tensor) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_add(
                    self.raw.as_ptr(),
                    output.raw(),
                    left.raw(),
                    right.raw(),
                    output.elements(),
                    error,
                    size,
                )
            },
            "add",
        )
    }

    /// `h3_gpu_silu_f32`.
    pub fn silu_f32(&self, output: &Tensor, input: &Tensor) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_silu_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    output.elements(),
                    error,
                    size,
                )
            },
            "silu",
        )
    }

    /// `h3_gpu_silu_bf16`.
    pub fn silu_bf16(&self, output: &Tensor, input: &Tensor) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_silu_bf16(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    output.elements(),
                    error,
                    size,
                )
            },
            "silu",
        )
    }

    /// `h3_gpu_cast_f32_to_bf16`.
    pub fn cast_f32_to_bf16(&self, output: &Tensor, input: &Tensor) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_cast_f32_to_bf16(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    output.elements(),
                    error,
                    size,
                )
            },
            "cast",
        )
    }

    /// `h3_gpu_copy_bf16`.
    pub fn copy(
        &self,
        destination: &Tensor,
        destination_element: usize,
        source: &Tensor,
        source_element: usize,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_copy(
                    self.raw.as_ptr(),
                    destination.raw(),
                    destination_element,
                    source.raw(),
                    source_element,
                    elements,
                    error,
                    size,
                )
            },
            "copy",
        )
    }

    /// Wait for every queued op.
    pub fn sync(&self) -> Result<(), String> {
        self.call(
            |error, size| unsafe { h3d_sync(self.raw.as_ptr(), error, size) },
            "sync",
        )
    }

    pub fn copy_f32(
        &self,
        destination: &Tensor,
        destination_element: usize,
        source: &Tensor,
        source_element: usize,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_copy_f32(
                    self.raw.as_ptr(),
                    destination.raw(),
                    destination_element,
                    source.raw(),
                    source_element,
                    elements,
                    error,
                    size,
                )
            },
            "copy_f32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        batch: u32,
        length: u32,
        input_channels: u32,
        output_channels: u32,
        kernel: u32,
        stride: u32,
        padding: u32,
        dilation: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_conv1d(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias.map_or(std::ptr::null(), |bias| bias.raw().cast_const()),
                    batch,
                    length,
                    input_channels,
                    output_channels,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    error,
                    size,
                )
            },
            "conv1d",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        batch: u32,
        length: u32,
        input_channels: u32,
        output_channels: u32,
        kernel: u32,
        stride: u32,
        padding: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_conv_transpose1d(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias.map_or(std::ptr::null(), |bias| bias.raw().cast_const()),
                    batch,
                    length,
                    input_channels,
                    output_channels,
                    kernel,
                    stride,
                    padding,
                    error,
                    size,
                )
            },
            "conv_transpose1d",
        )
    }

    pub fn weight_norm(
        &self,
        output: &Tensor,
        vector: &Tensor,
        magnitude: &Tensor,
        outer: u32,
        inner: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_weight_norm(
                    self.raw.as_ptr(),
                    output.raw(),
                    vector.raw(),
                    magnitude.raw(),
                    outer,
                    inner,
                    error,
                    size,
                )
            },
            "weight_norm",
        )
    }

    pub fn alias_free_snake(
        &self,
        output: &Tensor,
        input: &Tensor,
        alpha_log: &Tensor,
        beta_log: &Tensor,
        upsample_filter: &Tensor,
        downsample_filter: &Tensor,
        batch: u32,
        length: u32,
        channels: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_alias_free_snake(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    alpha_log.raw(),
                    beta_log.raw(),
                    upsample_filter.raw(),
                    downsample_filter.raw(),
                    batch,
                    length,
                    channels,
                    error,
                    size,
                )
            },
            "alias_free_snake",
        )
    }

    pub fn add_scaled(
        &self,
        output: &Tensor,
        left: &Tensor,
        right: &Tensor,
        left_scale: f32,
        right_scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_add_scaled(
                    self.raw.as_ptr(),
                    output.raw(),
                    left.raw(),
                    right.raw(),
                    left_scale,
                    right_scale,
                    output.elements(),
                    error,
                    size,
                )
            },
            "add_scaled",
        )
    }

    pub fn clip(
        &self,
        output: &Tensor,
        input: &Tensor,
        minimum: f32,
        maximum: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_clip(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    output.elements(),
                    minimum,
                    maximum,
                    error,
                    size,
                )
            },
            "clip",
        )
    }

    /// `h3_gpu_sdpa_f32`, non-causal and row-major out; `head_dim == 64`
    /// selects the reference's tiled kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_f32(
        &self,
        output: &Tensor,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_sdpa_f32(
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
            "sdpa_f32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_f32(
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
                h3d_rms_norm_f32(
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
            "rms_norm_f32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_f32(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_layer_norm_f32(
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
            "layer_norm_f32",
        )
    }

    pub fn swiglu_f32(
        &self,
        output: &Tensor,
        fused: &Tensor,
        rows: u32,
        width: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_swiglu_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    fused.raw(),
                    rows,
                    width,
                    error,
                    size,
                )
            },
            "swiglu_f32",
        )
    }

    pub fn scale_add_f32(
        &self,
        output: &Tensor,
        residual: &Tensor,
        branch: &Tensor,
        scale: &Tensor,
        rows: u32,
        width: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_scale_add_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    residual.raw(),
                    branch.raw(),
                    scale.raw(),
                    rows,
                    width,
                    error,
                    size,
                )
            },
            "scale_add_f32",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn video_qkv_rope_f32(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        qkv: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        rope_half: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_video_qkv_rope_f32(
                    self.raw.as_ptr(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    qkv.raw(),
                    rope_cos.raw(),
                    rope_sin.raw(),
                    sequence,
                    heads,
                    head_dim,
                    rope_half,
                    epsilon,
                    error,
                    size,
                )
            },
            "video_qkv_rope_f32",
        )
    }

    /// `h3_gpu_vae_encoder_pad_f32` (`h3_gpu_cuda.cu:2002-2043`): zeros in
    /// front of the temporal axis, single-reflection spatial padding.
    #[allow(clippy::too_many_arguments)]
    pub fn vae_encoder_pad(
        &self,
        output: &Tensor,
        input: &Tensor,
        batch: u32,
        depth: u32,
        height: u32,
        width: u32,
        channels: u32,
        depth_front: u32,
        height_before: u32,
        height_after: u32,
        width_before: u32,
        width_after: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_vae_encoder_pad(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    batch,
                    depth,
                    height,
                    width,
                    channels,
                    depth_front,
                    height_before,
                    height_after,
                    width_before,
                    width_after,
                    error,
                    size,
                )
            },
            "vae_encoder_pad",
        )
    }

    /// `h3_gpu_conv3d_f32` (`h3_gpu_cuda.cu:2075-2127`): valid convolution
    /// over an already padded tensor, the reference's fmaf order.
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        batch: u32,
        depth: u32,
        height: u32,
        width: u32,
        input_channels: u32,
        output_channels: u32,
        kernel: [u32; 3],
        stride: [u32; 3],
    ) -> Result<(), String> {
        let bias_raw: *const CTensor = match bias {
            Some(tensor) => tensor.raw(),
            None => std::ptr::null(),
        };
        self.call(
            |error, size| unsafe {
                h3d_conv3d(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias_raw,
                    batch,
                    depth,
                    height,
                    width,
                    input_channels,
                    output_channels,
                    kernel[0],
                    kernel[1],
                    kernel[2],
                    stride[0],
                    stride[1],
                    stride[2],
                    error,
                    size,
                )
            },
            "conv3d",
        )
    }

    /// `h3_gpu_vae_encoder_group_norm_silu_f32` (`:2153-2178`): one thread per
    /// (temporal, group) row, then the SiLU epilogue.
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm_silu(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        batch: u32,
        depth: u32,
        height: u32,
        width: u32,
        channels: u32,
        groups: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_group_norm_silu(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias.raw(),
                    batch,
                    depth,
                    height,
                    width,
                    channels,
                    groups,
                    epsilon,
                    error,
                    size,
                )
            },
            "group_norm_silu",
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
        unsafe { h3d_destroy(self.raw.as_ptr()) }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { h3d_free(self.raw.as_ptr()) }
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
        "native h3 DiT call failed without a message".to_string()
    } else {
        text
    }
}
