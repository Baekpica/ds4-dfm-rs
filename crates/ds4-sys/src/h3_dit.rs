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

use std::ffi::{c_char, CString};
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
    fn h3d_set_attention(context: *mut Context, scalar: c_int);

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

    fn h3d_vae_linear(
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

    fn h3d_vae_linear_bf16(
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

    fn h3d_vae_gemm_mode(context: *const Context) -> c_int;

    fn h3d_vae_attention_mma(context: *const Context) -> c_int;

    fn h3d_vae_sdpa_rows(context: *const Context) -> c_int;

    fn h3d_fp8_scale(
        context: *mut Context,
        scale: *mut CTensor,
        input: *const CTensor,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_fp8_quantize(
        context: *mut Context,
        output: *mut CTensor,
        scale: *const CTensor,
        input: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_linear_e4m3(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        input_scale: *const CTensor,
        weight: *const CTensor,
        weight_scale: *const CTensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_fp8_scale_rows(
        context: *mut Context,
        scales: *mut CTensor,
        input: *const CTensor,
        rows: u32,
        columns: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_fp8_quantize_rows(
        context: *mut Context,
        output: *mut CTensor,
        scales: *const CTensor,
        input: *const CTensor,
        rows: u32,
        columns: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_linear_e4m3_raw(
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

    fn h3d_fp8_scale_epilogue(
        context: *mut Context,
        output: *mut CTensor,
        accumulator: *const CTensor,
        row_scale: *const CTensor,
        column_scale: *const CTensor,
        rows: u32,
        columns: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_set_vae_option(
        context: *mut Context,
        name: *const c_char,
        value: *const c_char,
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

    fn h3d_token_pool(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        input_element: usize,
        original: *mut CTensor,
        original_element: usize,
        baseline: *mut CTensor,
        baseline_element: usize,
        baseline_indices: *const CTensor,
        pairs: *const CTensor,
        input_rows: u32,
        rows: u32,
        baseline_rows: u32,
        width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_token_expand(
        context: *mut Context,
        output: *mut CTensor,
        original: *const CTensor,
        original_element: usize,
        reduced: *const CTensor,
        baseline: *const CTensor,
        baseline_element: usize,
        baseline_indices: *const CTensor,
        parents: *const CTensor,
        rows: u32,
        reduced_rows: u32,
        baseline_rows: u32,
        width: u32,
        prefix_rows: u32,
        scale: f32,
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

    fn h3d_text_attention(
        context: *mut Context,
        output: *mut CTensor,
        query: *const CTensor,
        key: *const CTensor,
        value: *const CTensor,
        q_rows: u32,
        kv_rows: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// `h3d_layer_norm`: the refiner's `elementwise_affine=False` output norm.
    fn h3d_layer_norm(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        rows: u32,
        width: u32,
        epsilon: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// `h3d_randn_bf16`: torch's CUDA normal draw at the generator's own
    /// `(seed, offset)`, reporting the offset advance the caller applies.
    fn h3d_randn_bf16(
        context: *mut Context,
        output: *mut CTensor,
        elements: usize,
        seed: u64,
        offset: u64,
        increment: *mut u64,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// `h3d_noise_lerp`: `GaussianNoiser.__call__`'s two float32 lerps.
    fn h3d_noise_lerp(
        context: *mut Context,
        output: *mut CTensor,
        latent: *const CTensor,
        noise: *const CTensor,
        clean: *const CTensor,
        mask: *const CTensor,
        rows: u32,
        width: u32,
        weight: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// `h3d_timesteps`: the F32 `[rows]` `denoise_mask * sigma`.
    fn h3d_timesteps(
        context: *mut Context,
        output: *mut CTensor,
        mask: *const CTensor,
        rows: u32,
        sigma: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// `h3d_to_denoised`: `X0Model`'s `to_denoised` at the per-token timestep.
    fn h3d_to_denoised(
        context: *mut Context,
        output: *mut CTensor,
        latent: *const CTensor,
        velocity: *const CTensor,
        timestep: *const CTensor,
        rows: u32,
        width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// `h3d_to_velocity`: `to_velocity`, whose BF16 round the step reads back.
    fn h3d_to_velocity(
        context: *mut Context,
        output: *mut CTensor,
        sample: *const CTensor,
        denoised: *const CTensor,
        sigma: f32,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// `h3d_euler_step`: `EulerDiffusionStep.step`'s `sample + velocity*dt`.
    fn h3d_euler_step(
        context: *mut Context,
        output: *mut CTensor,
        sample: *const CTensor,
        velocity: *const CTensor,
        delta: f32,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// `h3d_sol_words`: the mask words one query block carries at this length.
    fn h3d_sol_words(sequence: u32) -> u32;
    /// VSA's selector: the reference's own sparsity rule for the H3 stage.
    fn h3d_sol_vsa(
        context: *mut Context,
        plan: *const SolPlan,
        query: *const CTensor,
        key: *const CTensor,
        value: *const CTensor,
        sequence: u32,
        heads: u32,
        scale: f32,
        sparsity: f64,
        prefix_blocks: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    /// The sparse forward the selector's mask was built for.
    fn h3d_sol_forward(
        context: *mut Context,
        output: *mut CTensor,
        plan: *const SolPlan,
        query: *const CTensor,
        key: *const CTensor,
        value: *const CTensor,
        sequence: u32,
        heads: u32,
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

    fn h3d_add_f32(
        context: *mut Context,
        output: *mut CTensor,
        left: *const CTensor,
        right: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn h3d_conv3d_gemm(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        bias: *const CTensor,
        depth: u32,
        height: u32,
        width: u32,
        input_channels: u32,
        output_channels: u32,
        kernel: u32,
        tile: u32,
        scratch: *const CTensor,
        tf32: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn h3d_zero_pad3d(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        batch: u32,
        depth: u32,
        height: u32,
        width: u32,
        channels: u32,
        pad_depth: u32,
        pad_height: u32,
        pad_width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn h3d_group_norm(
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

    fn h3d_channel_affine(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        scale: *const CTensor,
        shift: *const CTensor,
        channels: u32,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn h3d_temporal_dwconv(
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
        kernel: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn h3d_resize_hw(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        batch: u32,
        depth: u32,
        in_height: u32,
        in_width: u32,
        out_height: u32,
        out_width: u32,
        channels: u32,
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

    fn h3d_cast_bf16_to_f32(
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

    fn h3d_sub(
        context: *mut Context,
        output: *mut CTensor,
        left: *const CTensor,
        right: *const CTensor,
        elements: usize,
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
        bias: *const CTensor,
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
        bias: *const CTensor,
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
        bias: *const CTensor,
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

    fn h3d_geglu_f32(
        context: *mut Context,
        output: *mut CTensor,
        gate: *const CTensor,
        linear: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_snake1d_f32(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        alpha: *const CTensor,
        batch: u32,
        length: u32,
        channels: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_audio_qkv_split_f32(
        context: *mut Context,
        query: *mut CTensor,
        key: *mut CTensor,
        value: *mut CTensor,
        qkv: *const CTensor,
        q_bias: *const CTensor,
        k_bias: *const CTensor,
        v_bias: *const CTensor,
        batch: u32,
        length: u32,
        heads: u32,
        head_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_audio_attention_pool_f32(
        context: *mut Context,
        output: *mut CTensor,
        attended: *const CTensor,
        batch: u32,
        length: u32,
        heads: u32,
        head_dim: u32,
        output_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_sdpa_causal_f32(
        context: *mut Context,
        output: *mut CTensor,
        query: *const CTensor,
        key: *const CTensor,
        value: *const CTensor,
        batch: u32,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    // ---- the Gemma4 encoder's INT8 convrot route (`native/h3/h3_dit_gpu.h`).
    // The reference is comfy-kitchen 0.2.26, called by the release from
    // `runtime/cache_ops/int8_policy.py`; `h3d_int8_plan` is the chain's
    // workspace and must outlive the call.

    fn h3d_int8_hadamard(
        context: *mut Context,
        words: *mut CTensor,
        group_size: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_int8_rotate(
        context: *mut Context,
        output: *mut CTensor,
        words: *const CTensor,
        input: *const CTensor,
        rows: u32,
        columns: u32,
        group_size: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_int8_scale_rows(
        context: *mut Context,
        scales: *mut CTensor,
        input: *const CTensor,
        rows: u32,
        columns: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_int8_quantize_rows(
        context: *mut Context,
        output: *mut CTensor,
        scales: *const CTensor,
        input: *const CTensor,
        rows: u32,
        columns: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_linear_int8_raw(
        context: *mut Context,
        accumulator: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_int8_scale_epilogue(
        context: *mut Context,
        output: *mut CTensor,
        accumulator: *const CTensor,
        row_scale: *const CTensor,
        column_scale: *const CTensor,
        rows: u32,
        columns: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_linear_int8_convrot(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        weight: *const CTensor,
        weight_scale: *const CTensor,
        bias: *const CTensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        group_size: u32,
        plan: *const Int8Plan,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_int8_pad_rows(rows: u32) -> u32;

    // The Gemma4 text tower's own primitives (`native/h3/h3_dit_gpu.h`), the
    // forward of HuggingFace's `gemma4` at the release's pin. BF16 in and out
    // with an F32 intermediate, one reference expression each.
    fn h3d_embedding(
        context: *mut Context,
        output: *mut CTensor,
        table: *const CTensor,
        ids: *const CTensor,
        rows: u32,
        width: u32,
        scale: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_gelu_tanh(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_mul(
        context: *mut Context,
        output: *mut CTensor,
        left: *const CTensor,
        right: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_scale_bf16(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        scalar: *const CTensor,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_gemma_rope(
        context: *mut Context,
        query: *mut CTensor,
        key: *mut CTensor,
        cos: *const CTensor,
        sin: *const CTensor,
        rows: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_gemma_rope_table(
        context: *mut Context,
        cos: *mut CTensor,
        sin: *mut CTensor,
        rows: u32,
        dim: u32,
        angles: u32,
        base: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_gemma_mask(
        context: *mut Context,
        output: *mut CTensor,
        rows: u32,
        window: u32,
        padding: *const CTensor,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_gemma_attention(
        context: *mut Context,
        output: *mut CTensor,
        query: *const CTensor,
        key: *const CTensor,
        value: *const CTensor,
        mask: *const CTensor,
        rows: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        scale: f32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_connector_stack_scale(
        context: *mut Context,
        scales: *mut CTensor,
        stacked: *const CTensor,
        rows: u32,
        width: u32,
        layers: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_scale_constant(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        factor: f32,
        elements: usize,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_connector_interleave(
        context: *mut Context,
        output: *mut CTensor,
        stacked: *const CTensor,
        scales: *const CTensor,
        rows: u32,
        width: u32,
        layers: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_gather_rows(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        indices: *const CTensor,
        rows: u32,
        width: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_register_fill(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        binary: *const CTensor,
        registers: *const CTensor,
        rows: u32,
        width: u32,
        count: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_text_rope(
        context: *mut Context,
        query: *mut CTensor,
        key: *mut CTensor,
        cos: *const CTensor,
        sin: *const CTensor,
        rows: u32,
        heads: u32,
        head_dim: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;

    fn h3d_per_head_gate(
        context: *mut Context,
        output: *mut CTensor,
        input: *const CTensor,
        logits: *const CTensor,
        rows: u32,
        heads: u32,
        head_dim: u32,
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
    /// E4M3, one byte per element: the FP8 projection route's operand type.
    Fp8,
    /// The INT8 convrot route's quantized weights and activations.
    I8,
    /// Its int32 accumulator.
    I32,
}

impl Dtype {
    fn code(self) -> c_int {
        match self {
            Dtype::Bf16 => 0,
            Dtype::F32 => 1,
            Dtype::U32 => 2,
            Dtype::Fp8 => 3,
            Dtype::I8 => 4,
            Dtype::I32 => 5,
        }
    }

    /// Bytes per element, so a host slice can be checked before a copy.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bf16 => 2,
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::Fp8 | Dtype::I8 => 1,
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

/// `H3D_SOL_BLOCK`, `H3D_SOL_HEAD_DIM` (`native/h3/h3_dit_gpu.h:45`).
const SOL_BLOCK: u32 = 64;
const SOL_HEAD_DIM: usize = 128;

/// `struct h3d_sol_plan` (`native/h3/h3_dit_gpu.h:65-74`), field for field:
/// the block summaries the search step pools, the routing mask, its count,
/// the threshold the tau policy writes, the query centroids and the kc
/// statistics. Field order is the C struct's.
#[repr(C)]
pub struct SolPlan {
    pub kc: *mut CTensor,
    pub vc: *mut CTensor,
    pub mask: *mut CTensor,
    pub count: *mut CTensor,
    pub threshold: *mut CTensor,
    pub query_bar: *mut CTensor,
    pub kc_mean: *mut CTensor,
    pub kc_var: *mut CTensor,
}

/// The plan and the buffers it points at: the plan holds raw pointers, so
/// the tensors must outlive it and cannot be moved out. `Send` is not
/// claimed, as the tensors are device memory owned by one stream.
pub struct SolPlanBuffers {
    plan: SolPlan,
    _kc: Tensor,
    _vc: Tensor,
    _mask: Tensor,
    _count: Tensor,
    _threshold: Tensor,
    _query_bar: Tensor,
    _kc_mean: Tensor,
    _kc_var: Tensor,
}

/// `struct h3d_int8_plan` (`native/h3/h3_dit_gpu.h`), field for field: the
/// INT8 convrot chain's workspace. `quantized` and `accumulator` are sized to
/// `h3d_int8_pad_rows(rows)`, the reference's own M padding; the two padded
/// rows past the real ones are written and never read.
#[repr(C)]
struct Int8Plan {
    words: *mut CTensor,
    rotated: *mut CTensor,
    quantized: *mut CTensor,
    row_scales: *mut CTensor,
    accumulator: *mut CTensor,
}

/// The plan and the tensors it points at, for the same reason
/// [`SolPlanBuffers`] exists: the C struct holds raw pointers, so the tensors
/// must outlive it and cannot move.
pub struct Int8PlanBuffers {
    plan: Int8Plan,
    _words: Tensor,
    _rotated: Tensor,
    _quantized: Tensor,
    _row_scales: Tensor,
    _accumulator: Tensor,
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

    /// Select the attention path for this context. `scalar` true asks for the
    /// byte-exact scalar kernel, the only path that reproduces the reference;
    /// false selects the tensor-core default. Overrides the environment.
    pub fn set_attention(&self, scalar: bool) {
        unsafe { h3d_set_attention(self.raw.as_ptr(), scalar as c_int) };
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

    /// Copy an F32 tensor back to the host as `f32`, waiting for the stream.
    ///
    /// The decoder reads 31 MB back per tile and 54 tiles per render, and the
    /// byte form of that readback cost it a fresh buffer plus a
    /// `f32::from_le_bytes` pass over 7.8 million words every tile. On a
    /// little-endian host the device's bytes are already the host's `f32`
    /// values and every bit pattern is a valid `f32`, so this is the same copy
    /// `read` makes with no conversion at all.
    pub fn read_f32(&self, tensor: &Tensor, out: &mut [f32]) -> Result<(), String> {
        if out.len() * 4 != tensor.bytes() {
            return Err(format!(
                "read_f32 of {} floats from a {} byte tensor",
                out.len(),
                tensor.bytes()
            ));
        }
        if !cfg!(target_endian = "little") {
            return Err("read_f32 needs a little-endian host; use read".into());
        }
        // SAFETY: four bytes per element with no padding, `out.len() * 4` bytes
        // long, and `f32` has no invalid bit patterns, so the view is in bounds
        // for its whole length and the copy writes only `f32` values.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        self.read(tensor, bytes)
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

    /// The release's FP8 route (`h3d_fp8_scale`): one element of `scale`
    /// receives the tensorwise scale of `input`, `max(amax / 448, 1/(448*512))`
    /// in F32, which is the number `torch._scaled_mm` receives as its scale.
    pub fn fp8_scale(&self, scale: &Tensor, input: &Tensor) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_fp8_scale(self.raw.as_ptr(), scale.raw(), input.raw(), error, size)
            },
            "fp8 scale",
        )
    }

    /// `h3d_fp8_quantize`: the first `elements` of `input` rounded to E4M3
    /// through `scale`, the reference's
    /// `(x / x_scale).clamp(-448, 448).to(float8_e4m3fn)`. `output` may be a
    /// scratch larger than the activation, as the four projections' shared one
    /// is.
    pub fn fp8_quantize(
        &self,
        output: &Tensor,
        scale: &Tensor,
        input: &Tensor,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_fp8_quantize(
                    self.raw.as_ptr(),
                    output.raw(),
                    scale.raw(),
                    input.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "fp8 quantize",
        )
    }

    /// `torch._scaled_mm`'s matmul (`h3d_linear_e4m3`): E4M3 operands at the
    /// same row-major TRANSB shapes as `linear`, a BF16 output, and the two
    /// tensorwise scales applied by the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_e4m3(
        &self,
        output: &Tensor,
        input: &Tensor,
        input_scale: &Tensor,
        weight: &Tensor,
        weight_scale: &Tensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_linear_e4m3(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    input_scale.raw(),
                    weight.raw(),
                    weight_scale.raw(),
                    rows,
                    input_dim,
                    output_dim,
                    error,
                    size,
                )
            },
            "fp8 linear",
        )
    }

    /// `h3d_fp8_scale_rows`: one element of `scale` per row of a row-major
    /// `[rows, columns]` BF16 tensor, the reference's `_quantize_rowwise`
    /// (`fp8_config.py:63-68`) and, for a `[n, k]` weight, its `abs().amax(dim=1)`
    /// (`:226`) whose rows are output channels.
    pub fn fp8_scale_rows(
        &self,
        scales: &Tensor,
        input: &Tensor,
        rows: u32,
        columns: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_fp8_scale_rows(
                    self.raw.as_ptr(),
                    scales.raw(),
                    input.raw(),
                    rows,
                    columns,
                    error,
                    size,
                )
            },
            "fp8 row scale",
        )
    }

    /// `h3d_fp8_quantize_rows`: `input` rounded to E4M3 through the scale of
    /// its own row.
    pub fn fp8_quantize_rows(
        &self,
        output: &Tensor,
        scales: &Tensor,
        input: &Tensor,
        rows: u32,
        columns: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_fp8_quantize_rows(
                    self.raw.as_ptr(),
                    output.raw(),
                    scales.raw(),
                    input.raw(),
                    rows,
                    columns,
                    error,
                    size,
                )
            },
            "fp8 row quantize",
        )
    }

    /// `h3d_linear_e4m3_raw`: the same E4M3 matmul with no scales at all and
    /// an F32 output, the accumulator `fp8_scale_epilogue` scales.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_e4m3_raw(
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
                h3d_linear_e4m3_raw(
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
            "raw fp8 linear",
        )
    }

    /// `h3d_fp8_scale_epilogue`: `out[i][j] = bf16(acc[i][j] * row[i] *
    /// col[j])`, the reference's `(A @ W^T) * scale_a * scale_b` applied to
    /// the accumulator `_scaled_mm` would have scaled itself.
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_scale_epilogue(
        &self,
        output: &Tensor,
        accumulator: &Tensor,
        row_scale: &Tensor,
        column_scale: &Tensor,
        rows: u32,
        columns: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_fp8_scale_epilogue(
                    self.raw.as_ptr(),
                    output.raw(),
                    accumulator.raw(),
                    row_scale.raw(),
                    column_scale.raw(),
                    rows,
                    columns,
                    error,
                    size,
                )
            },
            "fp8 scale epilogue",
        )
    }

    // ---- the Gemma4 encoder's INT8 convrot route. The reference is
    // comfy-kitchen 0.2.26's eager backend, cited in the native file; the
    // release drives it from `runtime/cache_ops/int8_policy.py`.

    /// How many rows the int8 chain pads to: `max(rows, 32)` rounded up to 32,
    /// the reference's own M padding. A plan built for `rows` must size
    /// `quantized`, `row_scales` and `accumulator` for the padded count.
    pub fn int8_pad_rows(rows: u32) -> u32 {
        unsafe { h3d_int8_pad_rows(rows) }
    }

    /// Build the grouped-Hadamard sign matrix for `group_size` (a power of 4,
    /// at most 1024) into `words`, a `Dtype::U32` tensor of
    /// `group_size * ceil(group_size / 32)` elements.
    pub fn int8_hadamard(&self, words: &Tensor, group_size: u32) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_int8_hadamard(self.raw.as_ptr(), words.raw(), group_size, error, size)
            },
            "int8 hadamard",
        )
    }

    /// The grouped rotation on its own (`h3d_int8_rotate`), for a bisect.
    pub fn int8_rotate(
        &self,
        output: &Tensor,
        words: &Tensor,
        input: &Tensor,
        rows: u32,
        columns: u32,
        group_size: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_int8_rotate(
                    self.raw.as_ptr(),
                    output.raw(),
                    words.raw(),
                    input.raw(),
                    rows,
                    columns,
                    group_size,
                    error,
                    size,
                )
            },
            "int8 rotate",
        )
    }

    /// The row absmax and scale on their own (`h3d_int8_scale_rows`).
    pub fn int8_scale_rows(
        &self,
        scales: &Tensor,
        input: &Tensor,
        rows: u32,
        columns: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_int8_scale_rows(
                    self.raw.as_ptr(),
                    scales.raw(),
                    input.raw(),
                    rows,
                    columns,
                    error,
                    size,
                )
            },
            "int8 row scale",
        )
    }

    /// The row-wise quantization on its own (`h3d_int8_quantize_rows`).
    pub fn int8_quantize_rows(
        &self,
        output: &Tensor,
        scales: &Tensor,
        input: &Tensor,
        rows: u32,
        columns: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_int8_quantize_rows(
                    self.raw.as_ptr(),
                    output.raw(),
                    scales.raw(),
                    input.raw(),
                    rows,
                    columns,
                    error,
                    size,
                )
            },
            "int8 row quantize",
        )
    }

    /// The int8 x int8 -> int32 matmul (`h3d_linear_int8_raw`), row-major with
    /// the weight transposed, as the BF16 and FP8 routes are laid out.
    pub fn linear_int8_raw(
        &self,
        accumulator: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_linear_int8_raw(
                    self.raw.as_ptr(),
                    accumulator.raw(),
                    input.raw(),
                    weight.raw(),
                    rows,
                    input_dim,
                    output_dim,
                    error,
                    size,
                )
            },
            "int8 linear raw",
        )
    }

    /// The F32 scale epilogue (`h3d_int8_scale_epilogue`):
    /// `bf16(acc * (row_scale * column_scale))`, the reference's own
    /// association. The bias, when there is one, is `linear`'s separate add.
    pub fn int8_scale_epilogue(
        &self,
        output: &Tensor,
        accumulator: &Tensor,
        row_scale: &Tensor,
        column_scale: &Tensor,
        rows: u32,
        columns: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_int8_scale_epilogue(
                    self.raw.as_ptr(),
                    output.raw(),
                    accumulator.raw(),
                    row_scale.raw(),
                    column_scale.raw(),
                    rows,
                    columns,
                    error,
                    size,
                )
            },
            "int8 scale epilogue",
        )
    }

    /// The whole INT8 convrot linear (`h3d_linear_int8_convrot`):
    /// `DevInt8ConvRotLinear.forward` end to end. `weight` is `Dtype::I8` at
    /// `[output_dim, input_dim]` and `weight_scale` `Dtype::F32` at
    /// `[output_dim]`, both as the checkpoint carries them - the weight is
    /// already rotated, so nothing here rotates a weight.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_int8_convrot(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        weight_scale: &Tensor,
        bias: Option<&Tensor>,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        group_size: u32,
        plan: &Int8PlanBuffers,
    ) -> Result<(), String> {
        let bias_raw = bias.map_or(std::ptr::null_mut(), |tensor| tensor.raw());
        self.call(
            |error, size| unsafe {
                h3d_linear_int8_convrot(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    weight_scale.raw(),
                    bias_raw,
                    rows,
                    input_dim,
                    output_dim,
                    group_size,
                    &plan.plan,
                    error,
                    size,
                )
            },
            "int8 linear convrot",
        )
    }

    /// Allocate the chain's workspace for one `[rows, input_dim]` activation
    /// and one `[output_dim, input_dim]` weight, the way
    /// `h3d_int8_plan`'s comments describe it. One plan serves every layer of
    /// one shape.
    pub fn int8_plan(
        &self,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
        group_size: u32,
    ) -> Result<Int8PlanBuffers, String> {
        let padded = Self::int8_pad_rows(rows) as usize;
        let input_dim = input_dim as usize;
        let output_dim = output_dim as usize;
        let group_size = group_size as usize;
        let per_row = group_size.div_ceil(32);

        let words = self.alloc(Dtype::U32, group_size * per_row)?;
        let rotated = self.alloc(Dtype::Bf16, rows as usize * input_dim)?;
        let quantized = self.alloc(Dtype::I8, padded * input_dim)?;
        let row_scales = self.alloc(Dtype::F32, padded)?;
        let accumulator = self.alloc(Dtype::I32, padded * output_dim)?;
        self.int8_hadamard(&words, group_size as u32)?;

        let plan = Int8Plan {
            words: words.raw(),
            rotated: rotated.raw(),
            quantized: quantized.raw(),
            row_scales: row_scales.raw(),
            accumulator: accumulator.raw(),
        };

        Ok(Int8PlanBuffers {
            plan,
            _words: words,
            _rotated: rotated,
            _quantized: quantized,
            _row_scales: row_scales,
            _accumulator: accumulator,
        })
    }

    /// The Gemma4 text tower's own primitives, the forward of HuggingFace's
    /// `gemma4` at the release's pin. Each takes BF16 and returns BF16 with an
    /// F32 intermediate, and each is one reference expression.

    /// `embed_tokens(ids) * embed_scale` (`modeling_gemma4.py:1476-1477`), the
    /// scale already carrying the reference's own BF16 cast. `ids` is
    /// `Dtype::U32` `[rows]`, `table` BF16 `[vocabulary, width]`.
    pub fn embedding(
        &self,
        output: &Tensor,
        table: &Tensor,
        ids: &Tensor,
        rows: u32,
        width: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_embedding(
                    self.raw.as_ptr(),
                    output.raw(),
                    table.raw(),
                    ids.raw(),
                    rows,
                    width,
                    scale,
                    error,
                    size,
                )
            },
            "gemma embedding",
        )
    }

    /// `gelu_pytorch_tanh`, ATen's tanh gelu
    /// (`ActivationGeluKernel.cu:21-31`).
    pub fn gelu_tanh(&self, output: &Tensor, input: &Tensor) -> Result<(), String> {
        let elements = output.elements();

        self.call(
            |error, size| unsafe {
                h3d_gelu_tanh(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "gemma tanh gelu",
        )
    }

    /// The MLP's `act_fn(gate_proj(x)) * up_proj(x)` (`:1089`).
    pub fn mul(&self, output: &Tensor, left: &Tensor, right: &Tensor) -> Result<(), String> {
        let elements = output.elements();

        self.call(
            |error, size| unsafe {
                h3d_mul(
                    self.raw.as_ptr(),
                    output.raw(),
                    left.raw(),
                    right.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "bf16 multiply",
        )
    }

    /// `hidden_states *= self.layer_scalar` (`:1462`): `scalar` is a BF16
    /// `[1]`.
    pub fn scale_bf16(
        &self,
        output: &Tensor,
        input: &Tensor,
        scalar: &Tensor,
    ) -> Result<(), String> {
        let elements = output.elements();

        self.call(
            |error, size| unsafe {
                h3d_scale_bf16(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    scalar.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "bf16 scale",
        )
    }

    /// `apply_rotary_pos_emb` (`:793-811`) with `rotate_half` (`:786-791`),
    /// in place on head-major query and key. `cos` and `sin` are
    /// [`Self::gemma_rope_table`]'s pair tables.
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_rope(
        &self,
        query: &Tensor,
        key: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rows: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_gemma_rope(
                    self.raw.as_ptr(),
                    query.raw(),
                    key.raw(),
                    cos.raw(),
                    sin.raw(),
                    rows,
                    heads,
                    kv_heads,
                    head_dim,
                    error,
                    size,
                )
            },
            "gemma rope",
        )
    }

    /// The RoPE pair tables (`Gemma4TextRotaryEmbedding.forward`,
    /// `:1164-1181`): `Dtype::Bf16` `[rows, dim]` each. `angles` counts the
    /// live frequencies; the rest are the identity pair (`dim / 2` for the
    /// sliding layers' default table, `dim / 4` for the full ones'
    /// proportional table).
    pub fn gemma_rope_table(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        rows: u32,
        dim: u32,
        angles: u32,
        base: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_gemma_rope_table(
                    self.raw.as_ptr(),
                    cos.raw(),
                    sin.raw(),
                    rows,
                    dim,
                    angles,
                    base,
                    error,
                    size,
                )
            },
            "gemma rope table",
        )
    }

    /// The additive attention mask (`masking_utils.py:608-610`): BF16
    /// `[rows, rows]`, 0 where a key is attended and `finfo(bfloat16).min`
    /// where it is not. `window` of 0 is the full layer type; `padding` is
    /// `Dtype::U32` `[rows]` over the keys, or `None`.
    pub fn gemma_mask(
        &self,
        output: &Tensor,
        rows: u32,
        window: u32,
        padding: Option<&Tensor>,
    ) -> Result<(), String> {
        let padding = padding.map_or(std::ptr::null(), |padding| padding.raw());

        self.call(
            |error, size| unsafe {
                h3d_gemma_mask(
                    self.raw.as_ptr(),
                    output.raw(),
                    rows,
                    window,
                    padding,
                    error,
                    size,
                )
            },
            "gemma mask",
        )
    }

    /// `eager_attention_forward` (`:824-858`) at `scaling = 1.0`, head-major
    /// Q/K/V and output, with `mask` the [`Self::gemma_mask`] tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_attention(
        &self,
        output: &Tensor,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        rows: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), String> {
        let mask = mask.map_or(std::ptr::null(), |mask| mask.raw());

        self.call(
            |error, size| unsafe {
                h3d_gemma_attention(
                    self.raw.as_ptr(),
                    output.raw(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    mask,
                    rows,
                    heads,
                    kv_heads,
                    head_dim,
                    scale,
                    error,
                    size,
                )
            },
            "gemma attention",
        )
    }

    /// `_rescale_norm` (`feature_extractor.py:68-69`): scale by an F32
    /// constant, the product rounding to BF16 once.
    pub fn scale_constant(
        &self,
        output: &Tensor,
        input: &Tensor,
        factor: f32,
    ) -> Result<(), String> {
        let elements = output.elements();

        self.call(
            |error, size| unsafe {
                h3d_scale_constant(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    factor,
                    elements,
                    error,
                    size,
                )
            },
            "bf16 constant scale",
        )
    }

    /// `norm_and_concat_per_token_rms`'s rsqrt (`feature_extractor.py:48-64`):
    /// the BF16 `[rows, layers]` scale of each `(row, layer)` of the
    /// `[layers, rows, width]` stack.
    pub fn connector_stack_scale(
        &self,
        scales: &Tensor,
        stacked: &Tensor,
        rows: u32,
        width: u32,
        layers: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_connector_stack_scale(
                    self.raw.as_ptr(),
                    scales.raw(),
                    stacked.raw(),
                    rows,
                    width,
                    layers,
                    error,
                    size,
                )
            },
            "connector stack scale",
        )
    }

    /// The extractor's `reshape(B, T, D * L)` (`:62`): the layered stack
    /// written interleaved, each value scaled by its layer's scale.
    pub fn connector_interleave(
        &self,
        output: &Tensor,
        stacked: &Tensor,
        scales: &Tensor,
        rows: u32,
        width: u32,
        layers: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_connector_interleave(
                    self.raw.as_ptr(),
                    output.raw(),
                    stacked.raw(),
                    scales.raw(),
                    rows,
                    width,
                    layers,
                    error,
                    size,
                )
            },
            "connector interleave",
        )
    }

    /// `_apply_right_pad_order` (`embeddings_processor.py:40-42`): gather the
    /// rows by a host-computed permutation.
    pub fn gather_rows(
        &self,
        output: &Tensor,
        input: &Tensor,
        indices: &Tensor,
        rows: u32,
        width: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_gather_rows(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    indices.raw(),
                    rows,
                    width,
                    error,
                    size,
                )
            },
            "row gather",
        )
    }

    /// `_replace_padded_with_learnable_registers`
    /// (`embeddings_connector.py:139-151`): keep the row where `binary` is set,
    /// else take the tiled register row.
    pub fn register_fill(
        &self,
        output: &Tensor,
        input: &Tensor,
        binary: &Tensor,
        registers: &Tensor,
        rows: u32,
        width: u32,
        count: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_register_fill(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    binary.raw(),
                    registers.raw(),
                    rows,
                    width,
                    count,
                    error,
                    size,
                )
            },
            "register fill",
        )
    }

    /// `apply_split_rotary_emb` (`rope.py:42-85`) in place, `cos`/`sin` the
    /// BF16 `[rows, heads, head_dim / 2]` tables.
    pub fn text_rope(
        &self,
        query: &Tensor,
        key: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rows: u32,
        heads: u32,
        head_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_text_rope(
                    self.raw.as_ptr(),
                    query.raw(),
                    key.raw(),
                    cos.raw(),
                    sin.raw(),
                    rows,
                    heads,
                    head_dim,
                    error,
                    size,
                )
            },
            "RoPE",
        )
    }

    /// `PytorchGatedAttention` (`ops.py:94-106`): `input * 2 *
    /// sigmoid(logits)`, one gate per head.
    pub fn per_head_gate(
        &self,
        output: &Tensor,
        input: &Tensor,
        logits: &Tensor,
        rows: u32,
        heads: u32,
        head_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_per_head_gate(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    logits.raw(),
                    rows,
                    heads,
                    head_dim,
                    error,
                    size,
                )
            },
            "per-head gate",
        )
    }

    /// The video VAE decoder's F32 linear (`h3d_vae_linear`): `linear` at
    /// `Dtype::F32` with the decoder's own bench options applied. Only the
    /// decoder uses it, so those options cannot reach the encoder or the audio
    /// VAE, whose gates are bit-exact against the reference.
    pub fn vae_linear(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        rows: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_vae_linear(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias.map_or(std::ptr::null(), |bias| bias.raw()),
                    rows,
                    input_dim,
                    output_dim,
                    error,
                    size,
                )
            },
            "video VAE linear",
        )
    }

    /// The decoder's block GEMM on BF16 operands and a BF16 output
    /// (`h3d_vae_linear_bf16`), the `bf16i` arm. It takes no bias: the arm's
    /// consumers add the bias in F32, as they do since `c7bb59e1`.
    pub fn vae_linear_bf16(
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
                h3d_vae_linear_bf16(
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
            "video VAE BF16 linear",
        )
    }

    /// Whether this context's video VAE decoder takes BF16 GEMM operands, which
    /// is `H3_VAE_GEMM=bf16i`. The context parsed the value when it was created,
    /// so the host asks it rather than reading the environment a second time.
    pub fn vae_bf16_operands(&self) -> bool {
        const BF16_INPUTS: c_int = 3; // H3D_VAE_GEMM_BF16_INPUTS

        unsafe { h3d_vae_gemm_mode(self.raw.as_ptr()) == BF16_INPUTS }
    }

    /// This context's video VAE GEMM mode by name, as the context parsed it:
    /// "fp32", "tf32", "bf16" or "bf16i" (`H3D_VAE_GEMM_*`).
    pub fn vae_gemm_name(&self) -> &'static str {
        match unsafe { h3d_vae_gemm_mode(self.raw.as_ptr()) } {
            1 => "tf32",
            2 => "bf16",
            3 => "bf16i",
            _ => "fp32",
        }
    }

    /// Whether this context's video VAE attention is the tensor-core F32 kernel
    /// (`H3_VAE_ATTENTION=tensor`) rather than the reference's scalar one.
    pub fn vae_attention_mma(&self) -> bool {
        unsafe { h3d_vae_attention_mma(self.raw.as_ptr()) != 0 }
    }

    /// Whether that attention walks its CTA order row-major
    /// (`H3_VAE_SDPA_ORDER=rows`) rather than head-major.
    pub fn vae_sdpa_rows(&self) -> bool {
        unsafe { h3d_vae_sdpa_rows(self.raw.as_ptr()) != 0 }
    }

    /// Set one video VAE decoder bench option on this context
    /// (`h3d_set_vae_option`). The native side owns the names and the accepted
    /// values — "attention" (scalar, tensor), "sdpa_order" (heads, rows),
    /// "gemm" (fp32, tf32, bf16, bf16i) — and refuses an unknown pair, so a
    /// typo is an error rather than a silent default.
    pub fn set_vae_option(&self, name: &str, value: &str) -> Result<(), String> {
        let name = CString::new(name).map_err(|_| "option name has a NUL".to_string())?;
        let value = CString::new(value).map_err(|_| "option value has a NUL".to_string())?;
        let accepted =
            unsafe { h3d_set_vae_option(self.raw.as_ptr(), name.as_ptr(), value.as_ptr()) };
        if accepted == 0 {
            return Err(format!(
                "unknown video VAE option {:?} = {:?}",
                name.to_string_lossy(),
                value.to_string_lossy()
            ));
        }
        Ok(())
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

    /// `h3_gpu_token_pool_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub fn token_pool(
        &self,
        output: &Tensor,
        input: &Tensor,
        input_element: usize,
        original: &Tensor,
        original_element: usize,
        baseline: &Tensor,
        baseline_element: usize,
        baseline_indices: &Tensor,
        pairs: &Tensor,
        input_rows: u32,
        rows: u32,
        baseline_rows: u32,
        width: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_token_pool(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    input_element,
                    original.raw(),
                    original_element,
                    baseline.raw(),
                    baseline_element,
                    baseline_indices.raw(),
                    pairs.raw(),
                    input_rows,
                    rows,
                    baseline_rows,
                    width,
                    error,
                    size,
                )
            },
            "token pool",
        )
    }

    /// `h3_gpu_token_expand_delta_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub fn token_expand(
        &self,
        output: &Tensor,
        original: &Tensor,
        original_element: usize,
        reduced: &Tensor,
        baseline: &Tensor,
        baseline_element: usize,
        baseline_indices: &Tensor,
        parents: &Tensor,
        rows: u32,
        reduced_rows: u32,
        baseline_rows: u32,
        width: u32,
        prefix_rows: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_token_expand(
                    self.raw.as_ptr(),
                    output.raw(),
                    original.raw(),
                    original_element,
                    reduced.raw(),
                    baseline.raw(),
                    baseline_element,
                    baseline_indices.raw(),
                    parents.raw(),
                    rows,
                    reduced_rows,
                    baseline_rows,
                    width,
                    prefix_rows,
                    scale,
                    error,
                    size,
                )
            },
            "token expand",
        )
    }

    /// `h3_gpu_sub_bf16`.
    pub fn sub(&self, output: &Tensor, left: &Tensor, right: &Tensor) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_sub(
                    self.raw.as_ptr(),
                    output.raw(),
                    left.raw(),
                    right.raw(),
                    output.elements(),
                    error,
                    size,
                )
            },
            "subtract",
        )
    }

    /// `h3_gpu_rms_norm_bf16`.
    /// The BF16 RMS norm (`h3d_rms_norm`) with the reference's reduction tree.
    /// `weight` is optional because Gemma's `v_norm` is built with
    /// `with_scale=False` and has none.
    pub fn rms_norm(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: Option<&Tensor>,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        let weight = match weight {
            Some(weight) => weight.raw(),
            None => std::ptr::null(),
        };

        self.call(
            |error, size| unsafe {
                h3d_rms_norm(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight,
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

    /// `h3d_text_attention`: the token-major dense attention
    /// `PytorchAttention.__call__` runs (`attention.py:82-102`), with the query
    /// and key row counts independent - the AV cross-attention reads 16,128
    /// video queries against 126 audio keys.
    #[allow(clippy::too_many_arguments)]
    pub fn text_attention(
        &self,
        output: &Tensor,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        q_rows: u32,
        kv_rows: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_text_attention(
                    self.raw.as_ptr(),
                    output.raw(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    q_rows,
                    kv_rows,
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

    /// The refiner's `elementwise_affine=False` output LayerNorm.
    pub fn layer_norm(
        &self,
        output: &Tensor,
        input: &Tensor,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_layer_norm(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
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

    // ---- the stage-2 sampler (`the stage-2 sampler module`). One wrapper per
    // reference expression; the Rust side keeps the schedule and the
    // generator's offset, the device keeps every float32 rounding.

    /// torch's own CUDA normal draw at `dtype=bfloat16`: the grid-stride
    /// kernel of `ATen/native/cuda/DistributionTemplates.h:66-89`, one Philox
    /// subsequence per thread, `curand_normal4` and a `__float2bfloat16`
    /// store. The returned value is the offset advance
    /// `calc_execution_policy` (`:50-62`) gives the caller, so the caller's
    /// generator keeps torch's `philox_offset_per_thread_` itself.
    pub fn randn_bf16(&self, output: &Tensor, seed: u64, offset: u64) -> Result<u64, String> {
        let elements = output.elements();
        let mut increment = 0u64;

        self.call(
            |error, size| unsafe {
                h3d_randn_bf16(
                    self.raw.as_ptr(),
                    output.raw(),
                    elements,
                    seed,
                    offset,
                    &mut increment,
                    error,
                    size,
                )
            },
            "bf16 randn",
        )?;

        Ok(increment)
    }

    /// `GaussianNoiser.__call__` (`components/noisers.py:30-37`): two float32
    /// `torch.lerp`s and one cast back to the latent's dtype.
    pub fn noise_lerp(
        &self,
        output: &Tensor,
        latent: &Tensor,
        noise: &Tensor,
        clean: &Tensor,
        mask: &Tensor,
        rows: u32,
        width: u32,
        weight: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_noise_lerp(
                    self.raw.as_ptr(),
                    output.raw(),
                    latent.raw(),
                    noise.raw(),
                    clean.raw(),
                    mask.raw(),
                    rows,
                    width,
                    weight,
                    error,
                    size,
                )
            },
            "bf16 noise lerp",
        )
    }

    /// `timesteps_from_mask` (`helpers.py:494-501`): the F32 `[rows]`
    /// `denoise_mask * sigma` the block AdaLN and `to_denoised` read.
    pub fn timesteps(
        &self,
        output: &Tensor,
        mask: &Tensor,
        rows: u32,
        sigma: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_timesteps(
                    self.raw.as_ptr(),
                    output.raw(),
                    mask.raw(),
                    rows,
                    sigma,
                    error,
                    size,
                )
            },
            "f32 timesteps",
        )
    }

    /// `to_denoised` (`utils.py:39-52`) at the per-token timestep that
    /// `X0Model.forward` (`model.py:590-605`) passes.
    pub fn to_denoised(
        &self,
        output: &Tensor,
        latent: &Tensor,
        velocity: &Tensor,
        timestep: &Tensor,
        rows: u32,
        width: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_to_denoised(
                    self.raw.as_ptr(),
                    output.raw(),
                    latent.raw(),
                    velocity.raw(),
                    timestep.raw(),
                    rows,
                    width,
                    error,
                    size,
                )
            },
            "bf16 to-denoised",
        )
    }

    /// `to_velocity` (`utils.py:21-35`), kept separate from the step because
    /// the reference rounds to BF16 between them.
    pub fn to_velocity(
        &self,
        output: &Tensor,
        sample: &Tensor,
        denoised: &Tensor,
        sigma: f32,
    ) -> Result<(), String> {
        let elements = output.elements();

        self.call(
            |error, size| unsafe {
                h3d_to_velocity(
                    self.raw.as_ptr(),
                    output.raw(),
                    sample.raw(),
                    denoised.raw(),
                    sigma,
                    elements,
                    error,
                    size,
                )
            },
            "bf16 to-velocity",
        )
    }

    /// `EulerDiffusionStep.step` (`diffusion_steps.py:32-40`) with
    /// `delta = sigma_next - sigma`.
    pub fn euler_step(
        &self,
        output: &Tensor,
        sample: &Tensor,
        velocity: &Tensor,
        delta: f32,
    ) -> Result<(), String> {
        let elements = output.elements();

        self.call(
            |error, size| unsafe {
                h3d_euler_step(
                    self.raw.as_ptr(),
                    output.raw(),
                    sample.raw(),
                    velocity.raw(),
                    delta,
                    elements,
                    error,
                    size,
                )
            },
            "bf16 Euler step",
        )
    }

    /// The mask words per query block at this sequence length
    /// (`h3d_sol_words`), for callers sizing anything against the plan.
    pub fn sol_words(&self, sequence: u32) -> u32 {
        unsafe { h3d_sol_words(sequence) }
    }

    /// Allocate the sparse attention's workspace for `sequence x heads`. Every
    /// shape is the one `h3d_sol_check` validates before a kernel writes: the
    /// summaries are `[heads][blocks][128]`, the mask is `[blocks][heads]` of
    /// `words` words, and the rest is per (block, head).
    pub fn sol_plan(&self, sequence: u32, heads: u32) -> Result<SolPlanBuffers, String> {
        if sequence == 0 || heads == 0 {
            return Err("Sol-Attn plan needs a sequence and a head count".into());
        }
        let blocks = sequence.div_ceil(SOL_BLOCK);
        let words = self.sol_words(sequence);
        let summary = heads as usize * blocks as usize * SOL_HEAD_DIM;
        let pairs = heads as usize * blocks as usize;

        let kc = self.alloc(Dtype::Bf16, summary)?;
        let vc = self.alloc(Dtype::Bf16, summary)?;
        let query_bar = self.alloc(Dtype::F32, summary)?;
        let mask = self.alloc(Dtype::U32, pairs * words as usize)?;
        let count = self.alloc(Dtype::U32, pairs)?;
        let threshold = self.alloc(Dtype::F32, pairs)?;
        let kc_mean = self.alloc(Dtype::F32, heads as usize * SOL_HEAD_DIM)?;
        let kc_var = self.alloc(Dtype::F32, heads as usize * SOL_HEAD_DIM)?;

        let plan = SolPlan {
            kc: kc.raw(),
            vc: vc.raw(),
            mask: mask.raw(),
            count: count.raw(),
            threshold: threshold.raw(),
            query_bar: query_bar.raw(),
            kc_mean: kc_mean.raw(),
            kc_var: kc_var.raw(),
        };
        Ok(SolPlanBuffers {
            plan,
            _kc: kc,
            _vc: vc,
            _mask: mask,
            _count: count,
            _threshold: threshold,
            _query_bar: query_bar,
            _kc_mean: kc_mean,
            _kc_var: kc_var,
        })
    }

    /// The exact-block share of the mask the last `sol_vsa` wrote, read back
    /// from the mask itself. This is the model's own realized sparsity, which
    /// no fixture can supply, and it costs one synchronised readback, so a
    /// caller asks for it once rather than per layer.
    pub fn sol_exact_share(
        &self,
        plan: &SolPlanBuffers,
        sequence: u32,
        heads: u32,
    ) -> Result<f64, String> {
        let blocks = sequence.div_ceil(SOL_BLOCK) as u64;
        let words = self.sol_words(sequence) as u64;
        let mut bytes = vec![0u8; (blocks * heads as u64 * words * 4) as usize];
        self.read(&plan._mask, &mut bytes)?;

        let exact: u64 = bytes
            .chunks_exact(4)
            .map(|word| {
                u32::from_le_bytes([word[0], word[1], word[2], word[3]]).count_ones() as u64
            })
            .sum();
        Ok(exact as f64 / (blocks * blocks * heads as u64) as f64)
    }

    /// The routing mask's words for `(sequence, heads)`, one readback. Word
    /// order is `[q_block][head][word]` (`native/h3/h3_dit_gpu.h:65-74`), so a
    /// caller can check the diagonal band and the sink itself instead of
    /// trusting a share.
    pub fn sol_mask(
        &self,
        plan: &SolPlanBuffers,
        sequence: u32,
        heads: u32,
    ) -> Result<Vec<u32>, String> {
        let blocks = sequence.div_ceil(SOL_BLOCK) as usize;
        let words = self.sol_words(sequence) as usize;
        let mut bytes = vec![0u8; blocks * heads as usize * words * 4];
        self.read(&plan._mask, &mut bytes)?;

        Ok(bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }

    /// VSA's selector at `sparsity`, exempting the first `prefix_blocks` blocks.    /// `sparsity` is a double because the reference computes its keep count from
    /// a Python float: 0.9 rounded to F32 asks for a different count.
    #[allow(clippy::too_many_arguments)]
    pub fn sol_vsa(
        &self,
        plan: &SolPlanBuffers,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        sequence: u32,
        heads: u32,
        scale: f32,
        sparsity: f64,
        prefix_blocks: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_sol_vsa(
                    self.raw.as_ptr(),
                    &plan.plan,
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    sequence,
                    heads,
                    scale,
                    sparsity,
                    prefix_blocks,
                    error,
                    size,
                )
            },
            "VSA selector",
        )
    }

    /// The sparse forward over the mask the selector just wrote.
    #[allow(clippy::too_many_arguments)]
    pub fn sol_forward(
        &self,
        output: &Tensor,
        plan: &SolPlanBuffers,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        sequence: u32,
        heads: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_sol_forward(
                    self.raw.as_ptr(),
                    output.raw(),
                    &plan.plan,
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    sequence,
                    heads,
                    scale,
                    error,
                    size,
                )
            },
            "sparse attention",
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

    /// The same cast over `elements` values, which may be fewer than the tensors
    /// hold: the video VAE decoder's BF16 arm keeps one scratch pair for every
    /// projection, sized to the tile's largest of each.
    pub fn cast_narrow(
        &self,
        output: &Tensor,
        input: &Tensor,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_cast_f32_to_bf16(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "narrow cast",
        )
    }

    /// `h3_gpu_cast_bf16_to_f32`, its reverse: exact, not a rounding.
    pub fn cast_widen(
        &self,
        output: &Tensor,
        input: &Tensor,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_cast_bf16_to_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "wide cast",
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

    /// `bias`, when given, is the fused linear's own: the kernel adds it on
    /// load, so its producer is called with `H3D`'s bias left off.
    pub fn swiglu_f32(
        &self,
        output: &Tensor,
        fused: &Tensor,
        bias: Option<&Tensor>,
        rows: u32,
        width: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_swiglu_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    fused.raw(),
                    bias.map_or(std::ptr::null(), |bias| bias.raw()),
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
        bias: Option<&Tensor>,
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
                    bias.map_or(std::ptr::null(), |bias| bias.raw()),
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
        bias: Option<&Tensor>,
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
                    bias.map_or(std::ptr::null(), |bias| bias.raw()),
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

    /// `h3d_add_f32`: elementwise F32 add, the resizer's identity skips.
    pub fn add_f32(&self, output: &Tensor, left: &Tensor, right: &Tensor) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_add_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    left.raw(),
                    right.raw(),
                    output.elements(),
                    error,
                    size,
                )
            },
            "F32 add",
        )
    }

    /// `h3d_conv3d_gemm`: the resizer's 3x3x3 convolution through im2col and
    /// the F32 GEMM. `input` is the padded volume, `scratch` the expanded
    /// matrix's home (`tile * in * k^3` floats).
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_gemm(
        &self,
        output: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        depth: u32,
        height: u32,
        width: u32,
        input_channels: u32,
        output_channels: u32,
        kernel: u32,
        tile: u32,
        scratch: &Tensor,
        tf32: bool,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_conv3d_gemm(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    weight.raw(),
                    bias.raw(),
                    depth,
                    height,
                    width,
                    input_channels,
                    output_channels,
                    kernel,
                    tile,
                    scratch.raw(),
                    tf32 as u32,
                    error,
                    size,
                )
            },
            "conv3d gemm",
        )
    }

    /// `h3d_zero_pad3d`: zero padding for the resizer's 3x3x3 convolutions,
    /// channels-last, `[batch, depth, height, width, channels]`.
    #[allow(clippy::too_many_arguments)]
    pub fn zero_pad3d(
        &self,
        output: &Tensor,
        input: &Tensor,
        batch: u32,
        depth: u32,
        height: u32,
        width: u32,
        channels: u32,
        pad: [u32; 3],
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_zero_pad3d(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    batch,
                    depth,
                    height,
                    width,
                    channels,
                    pad[0],
                    pad[1],
                    pad[2],
                    error,
                    size,
                )
            },
            "zero pad",
        )
    }

    /// `h3d_group_norm`: GroupNorm without the SiLU epilogue, statistics over
    /// the whole volume per (batch, group).
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm(
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
                h3d_group_norm(
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
            "GroupNorm",
        )
    }

    /// `h3d_channel_affine`: `input * scale[c] + shift[c]`, the resizer's
    /// embedding modulation.
    pub fn channel_affine(
        &self,
        output: &Tensor,
        input: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
        channels: u32,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_channel_affine(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    scale.raw(),
                    shift.raw(),
                    channels,
                    elements,
                    error,
                    size,
                )
            },
            "channel affine",
        )
    }

    /// `h3d_temporal_dwconv`: the resizer's depthwise convolution over time.
    #[allow(clippy::too_many_arguments)]
    pub fn temporal_dwconv(
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
        kernel: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_temporal_dwconv(
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
                    kernel,
                    error,
                    size,
                )
            },
            "temporal depthwise conv",
        )
    }

    /// `h3d_resize_hw`: trilinear resize with the temporal axis held fixed,
    /// the upscale inside the resizer.
    #[allow(clippy::too_many_arguments)]
    pub fn resize_hw(
        &self,
        output: &Tensor,
        input: &Tensor,
        batch: u32,
        depth: u32,
        in_height: u32,
        in_width: u32,
        out_height: u32,
        out_width: u32,
        channels: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_resize_hw(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    batch,
                    depth,
                    in_height,
                    in_width,
                    out_height,
                    out_width,
                    channels,
                    error,
                    size,
                )
            },
            "trilinear resize",
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

    /// `h3_gpu_geglu_f32` (`:2922-2933`): tanh GELU of the gate times the
    /// linear half.
    pub fn geglu_f32(
        &self,
        output: &Tensor,
        gate: &Tensor,
        linear: &Tensor,
        elements: usize,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_geglu_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    gate.raw(),
                    linear.raw(),
                    elements,
                    error,
                    size,
                )
            },
            "geglu_f32",
        )
    }

    /// `h3_gpu_snake1d_f32` (`:2945-2960`): the plain Snake1d, not the
    /// alias-free variant the decoder uses.
    pub fn snake1d_f32(
        &self,
        output: &Tensor,
        input: &Tensor,
        alpha: &Tensor,
        batch: u32,
        length: u32,
        channels: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_snake1d_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    input.raw(),
                    alpha.raw(),
                    batch,
                    length,
                    channels,
                    error,
                    size,
                )
            },
            "snake1d_f32",
        )
    }

    /// `h3_gpu_audio_qkv_split_f32` (`:1850-1888`): one interleaved row to
    /// three head-major tensors, each with its own bias.
    pub fn audio_qkv_split_f32(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        qkv: &Tensor,
        q_bias: &Tensor,
        k_bias: &Tensor,
        v_bias: &Tensor,
        batch: u32,
        length: u32,
        heads: u32,
        head_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_audio_qkv_split_f32(
                    self.raw.as_ptr(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    qkv.raw(),
                    q_bias.raw(),
                    k_bias.raw(),
                    v_bias.raw(),
                    batch,
                    length,
                    heads,
                    head_dim,
                    error,
                    size,
                )
            },
            "audio_qkv_split_f32",
        )
    }

    /// `h3_gpu_audio_attention_pool_f32` (`:1890-1926`): per-head slices to
    /// one `output_dim` row.
    pub fn audio_attention_pool_f32(
        &self,
        output: &Tensor,
        attended: &Tensor,
        batch: u32,
        length: u32,
        heads: u32,
        head_dim: u32,
        output_dim: u32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_audio_attention_pool_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    attended.raw(),
                    batch,
                    length,
                    heads,
                    head_dim,
                    output_dim,
                    error,
                    size,
                )
            },
            "audio_attention_pool_f32",
        )
    }

    /// `h3_gpu_sdpa_causal_f32` (`:1659-1667`): head-major input, causal
    /// mask, row-major output.
    pub fn sdpa_causal_f32(
        &self,
        output: &Tensor,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        batch: u32,
        sequence: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), String> {
        self.call(
            |error, size| unsafe {
                h3d_sdpa_causal_f32(
                    self.raw.as_ptr(),
                    output.raw(),
                    query.raw(),
                    key.raw(),
                    value.raw(),
                    batch,
                    sequence,
                    heads,
                    head_dim,
                    scale,
                    error,
                    size,
                )
            },
            "sdpa_causal_f32",
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
