/*
 * h3d_* — device primitives for the MiniMax-H3 FL2VA DiT.
 *
 * Source of truth `/data/h3.c` at `eae5d54`, `h3_gpu_cuda.cu`. Each kernel is
 * the reference kernel's arithmetic with the branch set the CUDA BF16 path
 * actually takes; the comments name the reference lines. Expressions are kept
 * verbatim so the compiler makes the same contraction decisions: the E5 gate
 * hashes the BF16 output, and a split versus fused multiply-add changes the
 * last bit.
 *
 * Build flags matter as much as the code: the reference Makefile (`:26`) does
 * not pass `--use_fast_math`, so `expf` in SwiGLU and attention stays the
 * accurate libm entry point.
 */

#include "h3_dit_gpu.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct h3d_tensor {
    void *data;
    size_t elements;
    int dtype;
};

/* The video VAE decoder's bench options. Every field defaults to the path the
 * H3 campaign measured before, so a job that sets nothing is unchanged, and
 * each option exists so the arms can be timed on one binary. The decoder's
 * arithmetic has no reference analogue for these: `/data/h3.c` runs the F32
 * scalar attention, the reference's CTA order, F32 SIMT matmuls and a separate
 * bias add, and that is exactly the default here. */
enum {
    H3D_VAE_GEMM_FP32 = 0,   /* CUBLAS_COMPUTE_32F, the reference's SIMT F32 */
    H3D_VAE_GEMM_TF32 = 1,   /* CUBLAS_COMPUTE_32F_FAST_TF32 */
    H3D_VAE_GEMM_BF16 = 2,   /* CUBLAS_COMPUTE_32F_FAST_16BF, which cuBLASLt
                              * serves with a TF32-class kernel */
    H3D_VAE_GEMM_BF16_INPUTS = 3, /* CUDA_R_16BF operands and output ("bf16i") */
};

struct h3d_vae_options {
    int attention_mma;   /* 1 = tensor-core F32 attention at head_dim 64 */
    int query_major;     /* 1 = grid(query blocks, heads): the reuse order */
    int gemm;            /* H3D_VAE_GEMM_* */
};

struct h3d_context {
    cudaStream_t stream;
    cublasLtHandle_t blas;
    void *stage;          /* pinned upload staging, allocated on first use */
    cudaEvent_t stage_done;
    int stage_pending;    /* a DMA out of `stage` has not been waited for yet */
    int attention_mma;    /* 1 = tensor-core attention (default), 0 = byte-exact scalar */
    struct h3d_vae_options vae;
};

/* Whether the byte-exact scalar attention was asked for, from the environment.
 * Unset means the tensor-core kernel, which is the default. `H3_DIT_ATTENTION`
 * takes `scalar` or `tensor`; the older `H3_DIT_ATTENTION_MMA` is still read so
 * the A/B scripts written before the inversion keep their meaning, 1 for the
 * tensor kernel and 0 or empty for the scalar one. */
static int h3d_attention_scalar_from_env(void) {
    const char *legacy = getenv("H3_DIT_ATTENTION_MMA");
    if (legacy) {
        return !(legacy[0] != '\0' && legacy[0] != '0');
    }
    const char *mode = getenv("H3_DIT_ATTENTION");
    if (mode) {
        return mode[0] == 's' || mode[0] == 'S';
    }
    return 0;
}

/* Select the attention per run. `scalar` nonzero asks for the byte-exact
 * kernel, which is the only path that carries the reference guarantee; zero is
 * the tensor-core default. Overrides the environment. */
void h3d_set_attention(h3d_context *context, int scalar) {
    if (context) {
        context->attention_mma = !scalar;
    }
}

/* The video VAE decoder's options, from the environment, read once per
 * context. Unset means the reference path, so an arm is always a comparison
 * against the same binary's own default. Each value is matched on its first
 * letter, which is unambiguous within a name: scalar/tensor, heads/rows,
 * fp32/tf32/bf16. `bf16i` is the one value whose first letter is not its key,
 * so it is matched whole. */
static struct h3d_vae_options h3d_vae_options_from_env(void) {
    struct h3d_vae_options options = {0, 0, H3D_VAE_GEMM_FP32};
    const char *attention = getenv("H3_VAE_ATTENTION");
    const char *order = getenv("H3_VAE_SDPA_ORDER");
    const char *gemm = getenv("H3_VAE_GEMM");
    if (attention) options.attention_mma = attention[0] == 't' || attention[0] == 'T';
    if (order) options.query_major = order[0] == 'r' || order[0] == 'R';
    if (gemm && (gemm[0] == 't' || gemm[0] == 'T')) options.gemm = H3D_VAE_GEMM_TF32;
    if (gemm && (gemm[0] == 'b' || gemm[0] == 'B')) {
        options.gemm = strcmp(gemm, "bf16i") == 0 ? H3D_VAE_GEMM_BF16_INPUTS
                                                  : H3D_VAE_GEMM_BF16;
    }
    return options;
}

/* The decoder's GEMM mode in effect on this context. A host that has to choose
 * the weight dtype and the kernel path asks for it here rather than parsing
 * `H3_VAE_GEMM` a second time. Returns an `H3D_VAE_GEMM_*` value. */
int h3d_vae_gemm_mode(const h3d_context *context) {
    return context ? context->vae.gemm : H3D_VAE_GEMM_FP32;
}

/* The other two decoder options, read back for a log line. */
int h3d_vae_attention_mma(const h3d_context *context) {
    return context ? context->vae.attention_mma : 0;
}

int h3d_vae_sdpa_rows(const h3d_context *context) {
    return context ? context->vae.query_major : 0;
}

/* Select one video VAE decoder option per run. Names and values: "attention"
 * (scalar, tensor), "sdpa_order" (heads, rows), "gemm" (fp32, tf32, bf16,
 * bf16i). Exact words here, unlike the environment's first-letter rule,
 * because a host that sets an option wants a typo refused rather than silently
 * read as a default. Overrides H3_VAE_* for this context. Returns 0 when the
 * name or the value is not one of those. */
int h3d_set_vae_option(h3d_context *context, const char *name,
                       const char *value) {
    if (!context || !name || !value) return 0;
    if (strcmp(name, "attention") == 0) {
        if (strcmp(value, "scalar") == 0) { context->vae.attention_mma = 0; return 1; }
        if (strcmp(value, "tensor") == 0) { context->vae.attention_mma = 1; return 1; }
        return 0;
    }
    if (strcmp(name, "sdpa_order") == 0) {
        if (strcmp(value, "heads") == 0) { context->vae.query_major = 0; return 1; }
        if (strcmp(value, "rows") == 0) { context->vae.query_major = 1; return 1; }
        return 0;
    }
    if (strcmp(name, "gemm") == 0) {
        if (strcmp(value, "fp32") == 0) { context->vae.gemm = H3D_VAE_GEMM_FP32; return 1; }
        if (strcmp(value, "tf32") == 0) { context->vae.gemm = H3D_VAE_GEMM_TF32; return 1; }
        if (strcmp(value, "bf16") == 0) { context->vae.gemm = H3D_VAE_GEMM_BF16; return 1; }
        if (strcmp(value, "bf16i") == 0) { context->vae.gemm = H3D_VAE_GEMM_BF16_INPUTS; return 1; }
        return 0;
    }
    return 0;
}

static int h3d_set_error(char *error, size_t error_size, const char *format,
                         ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

/* `h3_cuda_ok` (`h3_gpu_cuda.cu:220-235`). */
static int h3d_cuda_ok(cudaError_t status, const char *operation, char *error,
                       size_t error_size) {
    if (status == cudaSuccess) return 1;
    return h3d_set_error(error, error_size, "%s: %s", operation,
                         cudaGetErrorString(status));
}

/* `h3_launch_ok` (`:705-709`): a launch error must not read as success. */
static int h3d_launch_ok(const char *operation, char *error,
                         size_t error_size) {
    return h3d_cuda_ok(cudaGetLastError(), operation, error, error_size);
}

static int h3d_blas_ok(cublasStatus_t status, const char *operation,
                       char *error, size_t error_size) {
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    return h3d_set_error(error, error_size, "%s: cuBLASLt status %d",
                         operation, (int)status);
}

static size_t h3d_dtype_size(int dtype) {
    switch (dtype) {
        case H3D_F32:
        case H3D_U32:
        case H3D_I32: return sizeof(uint32_t);
        case H3D_FP8:
        case H3D_I8: return 1;
        default: return sizeof(__nv_bfloat16);
    }
}

static int h3d_is(const h3d_tensor *tensor, int dtype, size_t elements) {
    return tensor && tensor->dtype == dtype && tensor->elements >= elements;
}

/* `h3_mul_size` (`:716-720`). */
static int h3d_mul_size(size_t left, size_t right, size_t *result) {
    if (left && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

/* `h3_value`/`h3_store` (`:2639-2653`) for the only type this path uses: the
 * reference keeps them templated because the Metal side also runs F32. */
__device__ static float h3d_value(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ static __nv_bfloat16 h3d_store(float value) {
    return __float2bfloat16(value);
}

h3d_context *h3d_create(char *error, size_t error_size) {
    h3d_context *context = (h3d_context *)calloc(1, sizeof(*context));
    if (!context) {
        h3d_set_error(error, error_size, "out of memory creating DiT context");
        return NULL;
    }
    if (!h3d_cuda_ok(cudaStreamCreateWithFlags(&context->stream,
                                               cudaStreamNonBlocking),
                     "create stream", error, error_size)) {
        h3d_destroy(context);
        return NULL;
    }
    cublasStatus_t blas = cublasLtCreate(&context->blas);
    if (blas != CUBLAS_STATUS_SUCCESS) {
        h3d_set_error(error, error_size, "create cuBLASLt handle: status %d",
                      (int)blas);
        h3d_destroy(context);
        return NULL;
    }
    /* The tensor-core attention is the default and the byte-exact scalar kernel
     * is the option, as of the operator's decision on 2026-10-27. The hosts
     * choose with `h3d_set_attention`; the environment is read once here as the
     * default for callers that never call it, such as the reproducer. */
    context->attention_mma = !h3d_attention_scalar_from_env();
    context->vae = h3d_vae_options_from_env();
    return context;
}

void h3d_destroy(h3d_context *context) {
    if (!context) return;
    if (context->stage_done) cudaEventDestroy(context->stage_done);
    if (context->stage) cudaFreeHost(context->stage);
    if (context->stream) cudaStreamDestroy(context->stream);
    if (context->blas) cublasLtDestroy(context->blas);
    free(context);
}

h3d_tensor *h3d_alloc(h3d_context *context, int dtype, size_t elements,
                      char *error, size_t error_size) {
    if (!context || (dtype != H3D_BF16 && dtype != H3D_F32 &&
                     dtype != H3D_U32 && dtype != H3D_FP8 &&
                     dtype != H3D_I8 && dtype != H3D_I32)) {
        h3d_set_error(error, error_size, "unknown DiT dtype %d", dtype);
        return NULL;
    }
    if (elements > SIZE_MAX / h3d_dtype_size(dtype)) {
        h3d_set_error(error, error_size, "tensor byte size overflow");
        return NULL;
    }
    h3d_tensor *tensor = (h3d_tensor *)calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3d_set_error(error, error_size, "out of memory allocating tensor");
        return NULL;
    }
    tensor->elements = elements;
    tensor->dtype = dtype;
    if (elements) {
        size_t bytes = elements * h3d_dtype_size(dtype);
        if (!h3d_cuda_ok(cudaMalloc(&tensor->data, bytes), "allocate tensor",
                         error, error_size)) {
            free(tensor);
            return NULL;
        }
    }
    return tensor;
}

void h3d_free(h3d_tensor *tensor) {
    if (!tensor) return;
    if (tensor->data) cudaFree(tensor->data);
    free(tensor);
}

/* cudaMemcpyAsync from pageable memory is not a DMA: the driver stages the
 * source into pinned memory on the calling thread and may synchronize with the
 * stream (CUDA Runtime API, "API Synchronization behavior", asynchronous rule
 * 3). Every upload here reads out of a read-only checkpoint mapping, so each
 * one paid that staging: 126,551 MB in 2,210 copies took 105.4 s at 1.20 GB/s,
 * against 4.44 s for the same bytes out of pinned memory. Staging in bounded
 * pieces and DMAs from the pinned buffer keeps the copy on the caller's stream,
 * so stream order and the numeric result are unchanged. */
static const size_t H3D_STAGE_BYTES = (size_t)8 << 20;

/* The pinned staging buffer and its completion event, allocated on first use.
 * Both directions use it: uploads refill it only after the DMA that read it
 * has landed, reads only read out of it after the DMA that wrote it has. */
static int h3d_stage_ready(h3d_context *context, char *error,
                           size_t error_size) {
    if (context->stage) return 1;
    if (!h3d_cuda_ok(cudaHostAlloc(&context->stage, H3D_STAGE_BYTES,
                                   cudaHostAllocDefault),
                     "allocate staging", error, error_size)) {
        return 0;
    }
    return h3d_cuda_ok(cudaEventCreateWithFlags(&context->stage_done,
                                                cudaEventDisableTiming),
                       "create staging event", error, error_size);
}

static int h3d_upload_range(h3d_context *context, void *destination,
                            const void *source, size_t bytes, char *error,
                            size_t error_size) {
    if (!h3d_stage_ready(context, error, error_size)) {
        return 0;
    }

    const char *from = (const char *)source;
    char *to = (char *)destination;

    for (size_t done = 0; done < bytes;) {
        size_t piece = bytes - done < H3D_STAGE_BYTES ? bytes - done
                                                     : H3D_STAGE_BYTES;
        /* The pinned buffer may only be refilled once the DMA that read it has
         * landed. That includes the tail of the PREVIOUS upload, not just the
         * previous piece of this one: skipping that wait corrupted the first
         * 8 MiB of every tensor after the first, which shows up as a wrong
         * render rather than as an error. */
        if (context->stage_pending &&
            !h3d_cuda_ok(cudaEventSynchronize(context->stage_done),
                         "wait for staged upload", error, error_size)) {
            return 0;
        }
        memcpy(context->stage, from + done, piece);
        if (!h3d_cuda_ok(cudaMemcpyAsync(to + done, context->stage, piece,
                                         cudaMemcpyHostToDevice,
                                         context->stream),
                         "upload tensor", error, error_size)) {
            return 0;
        }
        if (!h3d_cuda_ok(cudaEventRecord(context->stage_done, context->stream),
                         "record staged upload", error, error_size)) {
            return 0;
        }
        context->stage_pending = 1;
        done += piece;
    }
    return 1;
}

int h3d_upload(h3d_context *context, h3d_tensor *tensor, const void *host,
               char *error, size_t error_size) {
    if (!context || !tensor || !host) {
        return h3d_set_error(error, error_size, "invalid upload");
    }
    if (!tensor->elements) return 1;
    return h3d_upload_range(context, tensor->data, host,
                            tensor->elements * h3d_dtype_size(tensor->dtype),
                            error, error_size);
}

int h3d_upload_at(h3d_context *context, h3d_tensor *tensor, size_t element,
                  const void *host, size_t elements, char *error,
                  size_t error_size) {
    if (!context || !tensor || !host || element > tensor->elements ||
        elements > tensor->elements - element) {
        return h3d_set_error(error, error_size, "invalid tensor upload range");
    }
    if (!elements) return 1;
    size_t item = h3d_dtype_size(tensor->dtype);
    return h3d_upload_range(context, (char *)tensor->data + element * item,
                            host, elements * item, error, error_size);
}

/* The readback's mirror of `h3d_upload_range`: `cudaMemcpyAsync` into pageable
 * memory is not a DMA either. The driver stages the destination through its own
 * pinned buffer, and because the requested destination is not pinned the call
 * does not return until the copy has landed - so the decoder's per-tile
 * readback paid the same penalty the uploads did: 1.65 GB in 53 copies took
 * 1.87 s, 0.88 GB/s, against 28.5 GB/s measured for the same bytes out of
 * pinned memory above. Staging in bounded pieces in the reverse direction
 * keeps the copy on the caller's stream, in stream order with the kernels that
 * produced the tensor, and leaves the bytes unchanged. */
static int h3d_read_staged(h3d_context *context, void *host,
                           const void *device, size_t bytes, char *error,
                           size_t error_size) {
    if (!h3d_stage_ready(context, error, error_size)) {
        return 0;
    }

    char *to = (char *)host;
    for (size_t done = 0; done < bytes;) {
        size_t piece = bytes - done < H3D_STAGE_BYTES ? bytes - done
                                                     : H3D_STAGE_BYTES;
        /* The staging buffer may only be written once whatever last used it
         * has landed: the previous upload's DMA out of it, or this read's
         * previous piece. */
        if (context->stage_pending &&
            !h3d_cuda_ok(cudaEventSynchronize(context->stage_done),
                         "wait for staged read", error, error_size)) {
            return 0;
        }
        if (!h3d_cuda_ok(cudaMemcpyAsync(context->stage,
                                         (const char *)device + done, piece,
                                         cudaMemcpyDeviceToHost,
                                         context->stream),
                         "read tensor", error, error_size)) {
            return 0;
        }
        if (!h3d_cuda_ok(cudaEventRecord(context->stage_done, context->stream),
                         "record staged read", error, error_size)) {
            return 0;
        }
        context->stage_pending = 1;
        /* And the host may only read it once that DMA has landed. */
        if (!h3d_cuda_ok(cudaEventSynchronize(context->stage_done),
                         "land staged read", error, error_size)) {
            return 0;
        }
        memcpy(to + done, context->stage, piece);
        done += piece;
    }
    return 1;
}

int h3d_read(h3d_context *context, void *host, const h3d_tensor *tensor,
             char *error, size_t error_size) {
    if (!context || !tensor || !host) {
        return h3d_set_error(error, error_size, "invalid read");
    }
    if (!tensor->elements) return 1;
    if (!h3d_read_staged(context, host, tensor->data,
                         tensor->elements * h3d_dtype_size(tensor->dtype),
                         error, error_size)) {
        return 0;
    }
    return h3d_cuda_ok(cudaStreamSynchronize(context->stream),
                       "read tensor sync", error, error_size);
}

/* `h3_linear_lt` (`h3_gpu_cuda.cu:733-779`), byte for byte: row-major
 * layouts, TRANSB, F32 compute, alpha 1 and beta 0. `compute` carries this
 * port's video VAE decoder compute-mode option; the plain `h3d_linear_lt`
 * below passes the reference's value, so every other caller is unchanged. */
static int h3d_linear_lt_compute(h3d_context *context, void *output,
                                 const void *input, const void *weight,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t output_dim, cudaDataType_t type,
                                 cublasComputeType_t compute, char *error,
                                 size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t input_layout = NULL;
    cublasLtMatrixLayout_t weight_layout = NULL;
    cublasLtMatrixLayout_t output_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasLtMatmulDescCreate(&operation, compute,
                                                     CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
            sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&input_layout, type, rows,
                                            input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&weight_layout, type, output_dim,
                                            input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&output_layout, type, rows,
                                            output_dim, output_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(context->blas, operation, &alpha, input,
                                input_layout, weight, weight_layout, &beta,
                                output, output_layout, output, output_layout,
                                NULL, NULL, 0, context->stream);
    }
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    return h3d_blas_ok(status, "cuBLASLt linear", error, error_size);
}

/* The reference's matmul, unchanged: F32 compute. */
static int h3d_linear_lt(h3d_context *context, void *output, const void *input,
                         const void *weight, uint32_t rows, uint32_t input_dim,
                         uint32_t output_dim, cudaDataType_t type,
                         char *error, size_t error_size) {
    return h3d_linear_lt_compute(context, output, input, weight, rows,
                                 input_dim, output_dim, type, CUBLAS_COMPUTE_32F,
                                 error, error_size);
}

/* `h3_linear_bias_kernel` (`:783-791`): the bias is added to the already
 * rounded output, then rounded again. Two roundings, as in the reference. */
template <typename T>
__global__ static void h3d_linear_bias_kernel(T *output, const T *bias,
                                              size_t elements,
                                              uint32_t columns) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = (float)output[index] + (float)bias[index % columns];
    output[index] = (T)value;
}

/* `h3_patch_convert_kernel` (`:808-819`): one rounding, bias added in F32. */
__global__ static void h3d_patch_convert_kernel(__nv_bfloat16 *output,
                                                const float *input,
                                                const float *bias,
                                                size_t elements,
                                                uint32_t columns,
                                                int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = input[index];
    if (has_bias) value += bias[index % columns];
    output[index] = __float2bfloat16(value);
}

int h3d_linear(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t rows, uint32_t input_dim,
               uint32_t output_dim, int dtype, char *error,
               size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        (dtype != H3D_BF16 && dtype != H3D_F32) ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        !h3d_is(input, dtype, inputs) || !h3d_is(weight, dtype, weights) ||
        !h3d_is(output, dtype, outputs) ||
        (bias && !h3d_is(bias, dtype, output_dim))) {
        return h3d_set_error(error, error_size,
                             "invalid linear tensors or shape");
    }

    cudaDataType_t type = dtype == H3D_F32 ? CUDA_R_32F : CUDA_R_16BF;
    if (!h3d_linear_lt(context, output->data, input->data, weight->data, rows,
                       input_dim, output_dim, type, error, error_size)) {
        return 0;
    }
    if (!bias) return 1;

    unsigned blocks = (unsigned)((outputs + 255) / 256);
    if (dtype == H3D_F32) {
        h3d_linear_bias_kernel<<<blocks, 256, 0, context->stream>>>(
            (float *)output->data, (const float *)bias->data, outputs,
            output_dim);
    } else {
        h3d_linear_bias_kernel<<<blocks, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)bias->data, outputs, output_dim);
    }
    return h3d_launch_ok("linear bias", error, error_size);
}

/* ---- the DiT's FP8 projections: the release's own W8A8 route ----
 *
 * `runtime/stage1.py:126-134` gives the projection linears
 * `runtime/stage1_ops/lookup.py:15-16` names (`.to_q`, `.to_k`, `.to_v`,
 * `.to_out`, `.ff.fc_in`, `.ff.fc_out`) an `FP8Config()` with
 * `FP8QuantizeMethod("tensor")` and converts the model in place, so the arm
 * below ports the release's route instead of inventing a scheme. Its
 * arithmetic is FastVideo's `fastvideo/layers/quantization/fp8_config.py` at
 * the revision `configs/dependencies.json` pins (`3d8ac9d1`), cited step by
 * step. The reference drops the BF16 weight it converted
 * (`mod._parameters.pop("weight")`), so an FP8 model is smaller than its BF16
 * source, and `torch._scaled_mm` receives the two per-tensor scales as F32
 * device tensors. */
#define H3D_FP8_MAX 448.0f
#define H3D_FP8_MIN_SCALE (1.0f / (H3D_FP8_MAX * 512.0f))
#define H3D_FP8_THREADS 256
#define H3D_FP8_MAX_BLOCKS 1024

static unsigned h3d_fp8_blocks(size_t elements) {
    size_t blocks = (elements + H3D_FP8_THREADS - 1) / H3D_FP8_THREADS;
    if (!blocks) return 1;
    if (blocks > H3D_FP8_MAX_BLOCKS) return H3D_FP8_MAX_BLOCKS;
    return (unsigned)blocks;
}

/* `x_2d.abs().amax()` (`weight_local.detach().abs().amax().nan_to_num()` for a
 * weight): one maximum over the whole tensor, into the one-element F32
 * accumulator the caller has zeroed. `fmaxf` drops a NaN operand, which is
 * what `nan_to_num` does to an all-NaN amax; no real weight or activation
 * reaches that. */
__global__ static void h3d_fp8_amax_kernel(const __nv_bfloat16 *input,
                                           float *result, size_t elements) {
    float local = 0.0f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += stride) {
        local = fmaxf(local, fabsf(__bfloat162float(input[index])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    __shared__ float partial[H3D_FP8_THREADS / 32];
    if ((threadIdx.x & 31u) == 0) {
        partial[threadIdx.x >> 5] = local;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    float block = 0.0f;
    for (unsigned lane = 0; lane < H3D_FP8_THREADS / 32; ++lane) {
        block = fmaxf(block, partial[lane]);
    }
    /* A maximum of absolute values is a maximum of non-negative floats, and
     * those order as their bit patterns do: an integer maximum is the same
     * answer, and a zero block cannot raise it. */
    if (block > 0.0f) {
        atomicMax((int *)result, __float_as_int(block));
    }
}

/* `(x_absmax / FP8_MAX).clamp(min=1 / (448 * 512))`, in place on the
 * accumulator: the reference computes this in float and hands the float to
 * `_scaled_mm`. */
__global__ static void h3d_fp8_scale_kernel(float *scale) {
    scale[0] = fmaxf(scale[0] / H3D_FP8_MAX, H3D_FP8_MIN_SCALE);
}

/* `(x_2d / x_scale.to(x_2d.dtype)).clamp(-448, 448).to(torch.float8_e4m3fn)`.
 * The divisor is the scale rounded to the operand's own dtype, because the
 * reference divides a BF16 tensor by a BF16 scale, and the quotient is rounded
 * to BF16 for the same reason, before the clamp and the cast. The cast rounds
 * to nearest even, as torch's `.to(float8_e4m3fn)` does. */
__global__ static void h3d_fp8_quantize_kernel(__nv_fp8_storage_t *output,
                                               const __nv_bfloat16 *input,
                                               const float *scale,
                                               size_t elements) {
    const float divisor = __bfloat162float(__float2bfloat16(scale[0]));
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += stride) {
        float value = __bfloat162float(input[index]) / divisor;
        value = __bfloat162float(__float2bfloat16(value));
        value = fminf(fmaxf(value, -H3D_FP8_MAX), H3D_FP8_MAX);
        output[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    }
}

/* Two launches, as the reference's own two steps: the amax reduction, then the
 * scale. The zeroing is part of the reduction's contract rather than a third
 * step of its own. */
int h3d_fp8_scale(h3d_context *context, h3d_tensor *scale,
                  const h3d_tensor *input, char *error, size_t error_size) {
    if (!context || !h3d_is(scale, H3D_F32, 1) ||
        !h3d_is(input, H3D_BF16, 1)) {
        return h3d_set_error(error, error_size, "invalid FP8 scale tensors");
    }
    if (!h3d_cuda_ok(cudaMemsetAsync(scale->data, 0, sizeof(float),
                                     context->stream),
                     "FP8 amax zero", error, error_size)) {
        return 0;
    }
    h3d_fp8_amax_kernel<<<h3d_fp8_blocks(input->elements), H3D_FP8_THREADS, 0,
                          context->stream>>>((const __nv_bfloat16 *)input->data,
                                             (float *)scale->data,
                                             input->elements);
    if (!h3d_launch_ok("FP8 amax", error, error_size)) return 0;
    h3d_fp8_scale_kernel<<<1, 1, 0, context->stream>>>((float *)scale->data);
    return h3d_launch_ok("FP8 scale", error, error_size);
}

/* `elements` is the length of the activation being quantized, and `output` may
 * be a scratch larger than it: the FP8 route reuses one buffer for all four
 * projections, so the caller names the range the way `h3d_copy` does. */
int h3d_fp8_quantize(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *scale, const h3d_tensor *input,
                     size_t elements, char *error, size_t error_size) {
    if (!context || !h3d_is(scale, H3D_F32, 1) ||
        !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(output, H3D_FP8, elements)) {
        return h3d_set_error(error, error_size, "invalid FP8 quantize tensors");
    }
    if (!elements) return 1;
    h3d_fp8_quantize_kernel<<<h3d_fp8_blocks(elements), H3D_FP8_THREADS, 0,
                              context->stream>>>(
        (__nv_fp8_storage_t *)output->data,
        (const __nv_bfloat16 *)input->data, (const float *)scale->data,
        elements);
    return h3d_launch_ok("FP8 quantize", error, error_size);
}

/* ---- the same route at the reference's per-row granularity ----
 *
 * `fp8_config.py`'s `granularity="channel"` branch takes the activation's amax
 * per row of `[rows, k]` (`:63-68`, `abs().amax(dim=-1, keepdim=True)`) and the
 * weight's the same way (`convert_model_to_fp8`, `:226`, `abs().amax(dim=1)`
 * over an `[n, k]` weight, so a row of a weight is one output channel); its own
 * summary is "higher accuracy but slower `_scaled_mm`". One block per row: the
 * row is the reduction's unit, so no atomics and no cross-block fold, and no
 * zeroing pass either - every row writes its own element. */
__global__ static void h3d_fp8_amax_rows_kernel(const __nv_bfloat16 *input,
                                                float *rows, size_t columns) {
    const __nv_bfloat16 *row = input + (size_t)blockIdx.x * columns;
    float local = 0.0f;
    for (size_t index = threadIdx.x; index < columns; index += blockDim.x) {
        local = fmaxf(local, fabsf(__bfloat162float(row[index])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    __shared__ float partial[H3D_FP8_THREADS / 32];
    if ((threadIdx.x & 31u) == 0) {
        partial[threadIdx.x >> 5] = local;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    float block = 0.0f;
    for (unsigned lane = 0; lane < H3D_FP8_THREADS / 32; ++lane) {
        block = fmaxf(block, partial[lane]);
    }
    rows[blockIdx.x] = block;
}

/* `(x_absmax / FP8_MAX).clamp(min=1 / (448 * 512))`, one element per row. */
__global__ static void h3d_fp8_scale_rows_kernel(float *scales, size_t rows) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows) return;
    scales[index] = fmaxf(scales[index] / H3D_FP8_MAX, H3D_FP8_MIN_SCALE);
}

/* The tensorwise kernel's own rounding, with the divisor taken from the
 * element's row: `(x_2d / x_scale.to(x.dtype)).clamp(-448, 448).to(fp8)`. */
__global__ static void h3d_fp8_quantize_rows_kernel(
    __nv_fp8_storage_t *output, const __nv_bfloat16 *input, const float *scales,
    size_t columns) {
    const float divisor = __bfloat162float(__float2bfloat16(scales[blockIdx.x]));
    const __nv_bfloat16 *row = input + (size_t)blockIdx.x * columns;
    __nv_fp8_storage_t *out = output + (size_t)blockIdx.x * columns;
    for (size_t index = threadIdx.x; index < columns; index += blockDim.x) {
        float value = __bfloat162float(row[index]) / divisor;
        value = __bfloat162float(__float2bfloat16(value));
        value = fminf(fmaxf(value, -H3D_FP8_MAX), H3D_FP8_MAX);
        out[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    }
}

/* The two vectors `torch._scaled_mm` would apply, applied here instead:
 * `out[i][j] = bf16(acc[i][j] * row[i] * col[j])`, the reference's own
 * `(A @ W^T) * scale_a * scale_b` order, both multiplies in F32 because its
 * matmul scales the F32 accumulator before the BF16 cast. */
__global__ static void h3d_fp8_scale_epilogue_kernel(
    __nv_bfloat16 *output, const float *accumulator, const float *row_scale,
    const float *column_scale, size_t columns) {
    const size_t row = blockIdx.x;
    const float row_factor = row_scale[row];
    const float *in = accumulator + row * columns;
    __nv_bfloat16 *out = output + row * columns;
    for (size_t index = threadIdx.x; index < columns; index += blockDim.x) {
        out[index] =
            __float2bfloat16(in[index] * row_factor * column_scale[index]);
    }
}

/* The per-row pair of launches, as `h3d_fp8_scale` runs its tensorwise pair.
 * `rows` and `columns` are the row-major shape, so the caller names the same
 * numbers the matmul takes. */
int h3d_fp8_scale_rows(h3d_context *context, h3d_tensor *scales,
                       const h3d_tensor *input, uint32_t rows,
                       uint32_t columns, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !columns ||
        !h3d_mul_size(rows, columns, &elements) ||
        !h3d_is(scales, H3D_F32, rows) || !h3d_is(input, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid FP8 row scale tensors");
    }
    h3d_fp8_amax_rows_kernel<<<rows, H3D_FP8_THREADS, 0, context->stream>>>(
        (const __nv_bfloat16 *)input->data, (float *)scales->data, columns);
    if (!h3d_launch_ok("FP8 row amax", error, error_size)) return 0;
    h3d_fp8_scale_rows_kernel<<<h3d_fp8_blocks(rows), H3D_FP8_THREADS, 0,
                                context->stream>>>((float *)scales->data,
                                                   rows);
    return h3d_launch_ok("FP8 row scale", error, error_size);
}

int h3d_fp8_quantize_rows(h3d_context *context, h3d_tensor *output,
                          const h3d_tensor *scales, const h3d_tensor *input,
                          uint32_t rows, uint32_t columns, char *error,
                          size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !columns ||
        !h3d_mul_size(rows, columns, &elements) ||
        !h3d_is(scales, H3D_F32, rows) || !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(output, H3D_FP8, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid FP8 row quantize tensors");
    }
    h3d_fp8_quantize_rows_kernel<<<rows, H3D_FP8_THREADS, 0,
                                   context->stream>>>(
        (__nv_fp8_storage_t *)output->data,
        (const __nv_bfloat16 *)input->data, (const float *)scales->data,
        columns);
    return h3d_launch_ok("FP8 row quantize", error, error_size);
}

int h3d_fp8_scale_epilogue(h3d_context *context, h3d_tensor *output,
                           const h3d_tensor *accumulator,
                           const h3d_tensor *row_scale,
                           const h3d_tensor *column_scale, uint32_t rows,
                           uint32_t columns, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !columns ||
        !h3d_mul_size(rows, columns, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(accumulator, H3D_F32, elements) ||
        !h3d_is(row_scale, H3D_F32, rows) ||
        !h3d_is(column_scale, H3D_F32, columns)) {
        return h3d_set_error(error, error_size,
                             "invalid FP8 scale epilogue tensors");
    }
    h3d_fp8_scale_epilogue_kernel<<<rows, H3D_FP8_THREADS, 0,
                                    context->stream>>>(
        (__nv_bfloat16 *)output->data, (const float *)accumulator->data,
        (const float *)row_scale->data, (const float *)column_scale->data,
        columns);
    return h3d_launch_ok("FP8 scale epilogue", error, error_size);
}

/* `torch._scaled_mm`: the row-major TRANSB configuration `h3d_linear_lt`
 * builds, with E4M3 operands. The output type is the caller's - BF16 for the
 * scaled matmul, F32 for the one whose scales an epilogue applies after it -
 * and `a_scale` / `b_scale` are the release's two tensorwise scales as
 * `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` / `_B_SCALE_POINTER` (the attributes'
 * documented type is the compute type, F32 here), or NULL when the caller
 * applies them itself. alpha stays 1 and beta 0: the scales are the kernel's,
 * not the caller's, so no projection needs a host round trip. The attributes
 * are not merely a hint - setting them selects a different kernel, and the
 * unscaled call is the faster of the two on this architecture. */
static int h3d_linear_e4m3_compute(h3d_context *context, void *output,
                                   cudaDataType_t output_type,
                                   const void *input, const void *weight,
                                   const void *a_scale, const void *b_scale,
                                   uint32_t rows, uint32_t input_dim,
                                   uint32_t output_dim, char *error,
                                   size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t input_layout = NULL;
    cublasLtMatrixLayout_t weight_layout = NULL;
    cublasLtMatrixLayout_t output_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasLtMatmulDescCreate(&operation,
                                                     CUBLAS_COMPUTE_32F,
                                                     CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
            sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS && a_scale) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale,
            sizeof(a_scale));
    }
    if (status == CUBLAS_STATUS_SUCCESS && b_scale) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale,
            sizeof(b_scale));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&input_layout, CUDA_R_8F_E4M3,
                                            rows, input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_8F_E4M3,
                                            output_dim, input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&output_layout, output_type, rows,
                                            output_dim, output_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(
            context->blas, operation, &alpha, input, input_layout, weight,
            weight_layout, &beta, output, output_layout, output, output_layout,
            NULL, NULL, 0, context->stream);
    }
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    return h3d_blas_ok(status, "cuBLASLt FP8 linear", error, error_size);
}

/* `torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_scale, scale_b=w_scale,
 * out_dtype=torch.bfloat16)`: the two tensorwise scales are the kernel's. */
int h3d_linear_e4m3(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, const h3d_tensor *input_scale,
                    const h3d_tensor *weight, const h3d_tensor *weight_scale,
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                    char *error, size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        !h3d_is(input, H3D_FP8, inputs) ||
        !h3d_is(weight, H3D_FP8, weights) ||
        !h3d_is(output, H3D_BF16, outputs) ||
        !h3d_is(input_scale, H3D_F32, 1) ||
        !h3d_is(weight_scale, H3D_F32, 1)) {
        return h3d_set_error(error, error_size,
                             "invalid FP8 linear tensors or shape");
    }
    return h3d_linear_e4m3_compute(context, output->data, CUDA_R_16BF,
                                   input->data, weight->data, input_scale->data,
                                   weight_scale->data, rows, input_dim,
                                   output_dim, error, error_size);
}

/* The same matmul with no scales at all and an F32 output, for the per-row arm:
 * `h3d_fp8_scale_epilogue` applies the two vectors afterwards, which needs the
 * accumulator the reference's own scaling reads. */
int h3d_linear_e4m3_raw(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                        char *error, size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        !h3d_is(input, H3D_FP8, inputs) ||
        !h3d_is(weight, H3D_FP8, weights) ||
        !h3d_is(output, H3D_F32, outputs)) {
        return h3d_set_error(error, error_size,
                             "invalid raw FP8 linear tensors or shape");
    }
    return h3d_linear_e4m3_compute(context, output->data, CUDA_R_32F,
                                   input->data, weight->data, NULL, NULL, rows,
                                   input_dim, output_dim, error, error_size);
}

/* ---- the Gemma4 encoder's INT8 convrot route ----
 *
 * The reference is `comfy-kitchen` 0.2.26, unpacked read-only at
 * `/data/refs/comfy-kitchen-v0.2.26`; the release calls it through
 * `runtime/cache_ops/int8_policy.py:58-107` (`DevInt8ConvRotLinear`), which
 * passes `convrot=True`, the layer's `convrot_groupsize` and a BF16 output
 * dtype code. Five steps, each cited where it is implemented:
 *
 *   1. grouped Hadamard rotation of the activation   `tensor/int8_utils.py:55-70`
 *   2. row-wise absmax and scale                     `backends/eager/quantization.py:857-871`
 *   3. row-wise INT8 quantization in the operand's dtype   (same lines)
 *   4. int8 x int8 -> int32 accumulation             `:749-781`
 *   5. F32 scale epilogue, BF16 cast, bias last      `:1006-1051`
 *
 * The Hadamard matrix is stored as sign bits rather than as BF16 values. That
 * is exact and not an approximation: `_build_hadamard` divides the Kronecker
 * power of `h4` by `sqrt(size)` (`tensor/int8_utils.py:10-38`) and `size` is a
 * power of 4, so every entry is exactly +/- 2^-k; adding `+x` or `-x` and
 * applying one power-of-two scale at the end is the same arithmetic the
 * reference's dense matmul does, term for term and rounding for rounding. The
 * gate checks the sign bits against a host Kronecker build. */
#define H3D_INT8_MAX 127.0f
#define H3D_INT8_MIN_SCALE 1e-30f
/* `torch.finfo(torch.bfloat16).tiny`, the `_int8_scale_for_math` floor
 * (`:815-817`). It cannot fire here - the scale is clamped at `1e-30` first -
 * but the reference's guard is kept rather than argued away. */
#define H3D_INT8_TINY_BF16 1.1754943508222875e-38f
#define H3D_INT8_THREADS 256

/* `kron(h4, h4, ...)`: for `size = 4^k` every entry is the product of `h4`'s
 * own entries over the base-4 digits of its two indices, and `h4[i][j]` is -1
 * exactly when `i + j == 3` (`tensor/int8_utils.py:24-27`). One bit per entry,
 * set for +1. */
__global__ static void h3d_int8_hadamard_kernel(uint32_t *words,
                                                uint32_t size,
                                                uint32_t words_per_row) {
    size_t total = (size_t)size * words_per_row;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / words_per_row);
        uint32_t first = (uint32_t)(index % words_per_row) * 32u;
        uint32_t packed = 0;
        for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t column = first + lane;
            if (column >= size) break;
            uint32_t negative = 0;
            for (uint32_t step = size / 4;; step /= 4) {
                if ((row / step) % 4 + (column / step) % 4 == 3) {
                    negative ^= 1u;
                }
                if (step <= 1) break;
            }
            if (!negative) packed |= 1u << lane;
        }
        words[index] = packed;
    }
}

/* `_rotate_activation` (`tensor/int8_utils.py:55-70`): the last dimension is
 * reshaped into groups of `group_size` and each group is multiplied by the
 * matrix, `out[j] = sum_i in[i] * h[i][j]`. One block per (row, group); every
 * thread owns one output column and walks the inputs in index order, so the
 * accumulation is a single F32 register and the two-or-three-rounding drift a
 * cuBLAS tile order would add is absent. The reference accumulates the same
 * sum in F32; its own tile order is not pinned, which the gate measures. */
__global__ static void h3d_int8_rotate_kernel(__nv_bfloat16 *output,
                                              const __nv_bfloat16 *input,
                                              const uint32_t *words,
                                              uint32_t group,
                                              uint32_t words_per_row,
                                              float recip) {
    __shared__ __nv_bfloat16 column[H3D_INT8_GROUP_MAX];
    const uint32_t row = blockIdx.x, index = blockIdx.y;
    const uint32_t groups = gridDim.y;
    const uint32_t columns = groups * group;
    const __nv_bfloat16 *in = input + (size_t)row * columns + (size_t)index * group;
    __nv_bfloat16 *out = output + (size_t)row * columns + (size_t)index * group;
    for (uint32_t i = threadIdx.x; i < group; i += blockDim.x) column[i] = in[i];
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t i = 0; i < group; ++i) {
        /* The sign bit is a broadcast read: every thread of a warp wants the
         * same word, since the word index moves once per 32 columns. */
        uint32_t word = __ldg(&words[i * words_per_row + (threadIdx.x >> 5)]);
        sum += ((word >> (threadIdx.x & 31u)) & 1u) ? __bfloat162float(column[i])
                                                    : -__bfloat162float(column[i]);
    }
    out[threadIdx.x] = __float2bfloat16(sum * recip);
}

/* `abs_max = x.abs().amax(dim=-1, keepdim=True)` (`:870`) and
 * `scale = (abs_max.float() / 127.0).clamp(min=1e-30)` (`:871`). The
 * reduction is the FP8 route's own per-row amax, which is the same maximum of
 * the same absolute values; only the divisor below differs. */
__global__ static void h3d_int8_scale_rows_kernel(float *scales, size_t rows) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows) return;
    scales[index] = fmaxf(scales[index] / H3D_INT8_MAX, H3D_INT8_MIN_SCALE);
}

/* `_round_int8(x / _int8_scale_for_math(scale, x))` (`:820-826`, `:857-871`).
 * The divisor is the scale rounded to the operand's dtype, because the
 * reference divides a BF16 tensor by a BF16 scale; the quotient is rounded to
 * BF16 for the same reason before `round_` (half to even) and the clamp. The
 * value is integral after the round, so the cast to int8 only reinterprets. */
__global__ static void h3d_int8_quantize_rows_kernel(int8_t *output,
                                                     const __nv_bfloat16 *input,
                                                     const float *scales,
                                                     size_t columns) {
    __nv_bfloat16 rounded = __float2bfloat16(scales[blockIdx.x]);
    if (__bfloat162float(rounded) == 0.0f) {
        rounded = __float2bfloat16(H3D_INT8_TINY_BF16);
    }
    const float divisor = __bfloat162float(rounded);
    for (size_t index = threadIdx.x; index < columns; index += blockDim.x) {
        size_t at = (size_t)blockIdx.x * columns + index;
        float value = __bfloat162float(input[at]) / divisor;
        value = __bfloat162float(__float2bfloat16(value));
        value = rintf(value);
        value = fminf(fmaxf(value, -128.0f), H3D_INT8_MAX);
        output[at] = (int8_t)value;
    }
}

/* `torch._int_mm(x_8, weight.T.contiguous())` (`:749-781`, `:1021`): int8 x
 * int8 into an int32 accumulator, row-major and TRANSB as the BF16 and FP8
 * routes are laid out. The reference zero-pads M, K and N to cuBLASLt's
 * alignments before the call; padding a dimension with zeros cannot change the
 * accumulator, so the port instead requires the alignments the artifacts
 * already satisfy (K and N multiples of 8) and pads only M, which costs the
 * caller nothing but a wider scratch. */
static int h3d_linear_int8_compute(h3d_context *context, void *output,
                                   const void *input, const void *weight,
                                   uint32_t rows, uint32_t input_dim,
                                   uint32_t output_dim, char *error,
                                   size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t input_layout = NULL;
    cublasLtMatrixLayout_t weight_layout = NULL;
    cublasLtMatrixLayout_t output_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    int32_t alpha = 1, beta = 0;
    cublasStatus_t status = cublasLtMatmulDescCreate(&operation,
                                                     CUBLAS_COMPUTE_32I,
                                                     CUDA_R_32I);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
            sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&input_layout, CUDA_R_8I, rows,
                                            input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_8I,
                                            output_dim, input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&output_layout, CUDA_R_32I, rows,
                                            output_dim, output_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(context->blas, operation, &alpha, input,
                                input_layout, weight, weight_layout, &beta,
                                output, output_layout, output, output_layout,
                                NULL, NULL, 0, context->stream);
    }
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    return h3d_blas_ok(status, "cuBLASLt INT8 linear", error, error_size);
}

/* `chunk.float() * (x_scale * weight_scale)` then `.to(bf16)` (`:1035-1040`):
 * the two F32 multiplies in the reference's own association, the row scale and
 * the column scale multiplied first and the F32 accumulator scaled by that
 * product, with one BF16 rounding. `h3d_linear_bias_kernel` adds the bias
 * afterwards in BF16, which is the reference's own `result + bias` (`:1047`). */
__global__ static void h3d_int8_scale_epilogue_kernel(
    __nv_bfloat16 *output, const int32_t *accumulator, const float *row_scale,
    const float *column_scale, size_t columns) {
    const size_t row = blockIdx.x;
    const float row_factor = row_scale[row];
    const int32_t *in = accumulator + row * columns;
    __nv_bfloat16 *out = output + row * columns;
    for (size_t index = threadIdx.x; index < columns; index += blockDim.x) {
        out[index] = __float2bfloat16((float)in[index] *
                                      (row_factor * column_scale[index]));
    }
}

/* `size` is a power of 4, so `sqrt(size)` is a power of two and the reciprocal
 * is exact in F32; no libm call is needed on either side. */
static float h3d_int8_reciprocal(uint32_t size) {
    float root = 1.0f;
    while (root * root < (float)size) root *= 2.0f;
    return 1.0f / root;
}

static int h3d_int8_check_group(uint32_t group_size, char *error,
                                size_t error_size) {
    if (group_size < 4 || group_size > H3D_INT8_GROUP_MAX ||
        (group_size & (group_size - 1)) != 0) {
        return h3d_set_error(error, error_size,
                             "INT8 convrot group %u is not a power of two in "
                             "[4, %d]",
                             group_size, (int)H3D_INT8_GROUP_MAX);
    }
    float root = 1.0f;
    while (root * root < (float)group_size) root *= 2.0f;
    if (root * root != (float)group_size) {
        return h3d_set_error(error, error_size,
                             "INT8 convrot group %u is not a power of 4",
                             group_size);
    }
    return 1;
}

int h3d_int8_hadamard(h3d_context *context, h3d_tensor *words,
                      uint32_t group_size, char *error, size_t error_size) {
    size_t per_row = (group_size + 31) / 32, total = 0;
    if (!context || !h3d_int8_check_group(group_size, error, error_size) ||
        !h3d_mul_size(group_size, per_row, &total) ||
        !h3d_is(words, H3D_U32, total)) {
        return h3d_set_error(error, error_size,
                             "invalid INT8 Hadamard tensors or group");
    }
    unsigned blocks = (unsigned)((total + H3D_INT8_THREADS - 1) / H3D_INT8_THREADS);
    if (blocks > H3D_FP8_MAX_BLOCKS) blocks = H3D_FP8_MAX_BLOCKS;
    h3d_int8_hadamard_kernel<<<blocks, H3D_INT8_THREADS, 0, context->stream>>>(
        (uint32_t *)words->data, group_size, (uint32_t)per_row);
    return h3d_launch_ok("INT8 Hadamard", error, error_size);
}

int h3d_int8_rotate(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *words, const h3d_tensor *input,
                    uint32_t rows, uint32_t columns, uint32_t group_size,
                    char *error, size_t error_size) {
    size_t elements = 0, total = 0;
    if (!context || !rows || !columns || !h3d_mul_size(rows, columns, &elements) ||
        !h3d_int8_check_group(group_size, error, error_size) ||
        columns % group_size != 0 ||
        !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid INT8 rotation tensors or shape");
    }
    size_t per_row = (group_size + 31) / 32;
    if (!h3d_mul_size(group_size, per_row, &total) ||
        !h3d_is(words, H3D_U32, total)) {
        return h3d_set_error(error, error_size, "invalid INT8 Hadamard tensor");
    }
    uint32_t groups = columns / group_size;
    if (groups > 65535) {
        return h3d_set_error(error, error_size,
                             "INT8 rotation needs %u groups, at most 65535",
                             groups);
    }
    dim3 grid(rows, groups);
    h3d_int8_rotate_kernel<<<grid, group_size, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const uint32_t *)words->data, group_size, (uint32_t)per_row,
        h3d_int8_reciprocal(group_size));
    return h3d_launch_ok("INT8 rotate", error, error_size);
}

int h3d_int8_scale_rows(h3d_context *context, h3d_tensor *scales,
                        const h3d_tensor *input, uint32_t rows,
                        uint32_t columns, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !columns ||
        !h3d_mul_size(rows, columns, &elements) ||
        !h3d_is(scales, H3D_F32, rows) || !h3d_is(input, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid INT8 row scale tensors");
    }
    h3d_fp8_amax_rows_kernel<<<rows, H3D_FP8_THREADS, 0, context->stream>>>(
        (const __nv_bfloat16 *)input->data, (float *)scales->data, columns);
    if (!h3d_launch_ok("INT8 row amax", error, error_size)) return 0;
    unsigned blocks = (unsigned)((rows + H3D_INT8_THREADS - 1) / H3D_INT8_THREADS);
    h3d_int8_scale_rows_kernel<<<blocks, H3D_INT8_THREADS, 0,
                                 context->stream>>>((float *)scales->data, rows);
    return h3d_launch_ok("INT8 row scale", error, error_size);
}

int h3d_int8_quantize_rows(h3d_context *context, h3d_tensor *output,
                           const h3d_tensor *scales, const h3d_tensor *input,
                           uint32_t rows, uint32_t columns, char *error,
                           size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !columns ||
        !h3d_mul_size(rows, columns, &elements) ||
        !h3d_is(scales, H3D_F32, rows) || !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(output, H3D_I8, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid INT8 row quantize tensors");
    }
    h3d_int8_quantize_rows_kernel<<<rows, H3D_INT8_THREADS, 0,
                                    context->stream>>>(
        (int8_t *)output->data, (const __nv_bfloat16 *)input->data,
        (const float *)scales->data, columns);
    return h3d_launch_ok("INT8 row quantize", error, error_size);
}

int h3d_linear_int8_raw(h3d_context *context, h3d_tensor *accumulator,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                        char *error, size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        !h3d_is(input, H3D_I8, inputs) || !h3d_is(weight, H3D_I8, weights) ||
        !h3d_is(accumulator, H3D_I32, outputs) ||
        (input_dim % 8) || (output_dim % 8)) {
        return h3d_set_error(error, error_size,
                             "invalid raw INT8 linear tensors or shape");
    }
    return h3d_linear_int8_compute(context, accumulator->data, input->data,
                                   weight->data, rows, input_dim, output_dim,
                                   error, error_size);
}

int h3d_int8_scale_epilogue(h3d_context *context, h3d_tensor *output,
                            const h3d_tensor *accumulator,
                            const h3d_tensor *row_scale,
                            const h3d_tensor *column_scale, uint32_t rows,
                            uint32_t columns, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !columns ||
        !h3d_mul_size(rows, columns, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(accumulator, H3D_I32, elements) ||
        !h3d_is(row_scale, H3D_F32, rows) ||
        !h3d_is(column_scale, H3D_F32, columns)) {
        return h3d_set_error(error, error_size,
                             "invalid INT8 scale epilogue tensors");
    }
    h3d_int8_scale_epilogue_kernel<<<rows, H3D_INT8_THREADS, 0,
                                     context->stream>>>(
        (__nv_bfloat16 *)output->data, (const int32_t *)accumulator->data,
        (const float *)row_scale->data, (const float *)column_scale->data,
        columns);
    return h3d_launch_ok("INT8 scale epilogue", error, error_size);
}

/* `max(rows, 32)` rounded up to 32: the reference's own M padding for CUDA
 * (`_int8_matmul_accumulate`, `:762-765`), so a short prompt's row count is
 * legal. Padding a row with zeros cannot change an int32 accumulator, and the
 * epilogue reads only the real rows, so the extra rows do not need zeroing. */
uint32_t h3d_int8_pad_rows(uint32_t rows) {
    uint32_t padded = rows < 32 ? 32 : rows;
    return (padded + 31u) & ~31u;
}

int h3d_linear_int8_convrot(h3d_context *context, h3d_tensor *output,
                            const h3d_tensor *input, const h3d_tensor *weight,
                            const h3d_tensor *weight_scale,
                            const h3d_tensor *bias, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim,
                            uint32_t group_size,
                            const struct h3d_int8_plan *plan, char *error,
                            size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0;
    uint32_t padded = h3d_int8_pad_rows(rows);
    if (!context || !rows || !input_dim || !output_dim || !plan ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        !h3d_is(input, H3D_BF16, inputs) || !h3d_is(weight, H3D_I8, weights) ||
        !h3d_is(weight_scale, H3D_F32, output_dim) ||
        !h3d_is(output, H3D_BF16, outputs) ||
        (bias && !h3d_is(bias, H3D_BF16, output_dim)) ||
        !h3d_int8_check_group(group_size, error, error_size)) {
        return h3d_set_error(error, error_size,
                             "invalid INT8 convrot linear tensors or shape");
    }
    if (!plan->rotated || !plan->quantized || !plan->row_scales ||
        !plan->accumulator || !plan->words ||
        !h3d_is(plan->rotated, H3D_BF16, inputs) ||
        !h3d_is(plan->quantized, H3D_I8, (size_t)padded * input_dim) ||
        !h3d_is(plan->row_scales, H3D_F32, padded) ||
        !h3d_is(plan->accumulator, H3D_I32, (size_t)padded * output_dim)) {
        return h3d_set_error(error, error_size,
                             "invalid INT8 convrot plan tensors");
    }
    if (!h3d_int8_rotate(context, plan->rotated, plan->words, input, rows,
                         input_dim, group_size, error, error_size) ||
        !h3d_int8_scale_rows(context, plan->row_scales, plan->rotated, rows,
                             input_dim, error, error_size) ||
        !h3d_int8_quantize_rows(context, plan->quantized, plan->row_scales,
                                plan->rotated, rows, input_dim, error,
                                error_size) ||
        !h3d_linear_int8_raw(context, plan->accumulator, plan->quantized,
                             weight, padded, input_dim, output_dim, error,
                             error_size) ||
        !h3d_int8_scale_epilogue(context, output, plan->accumulator,
                                 plan->row_scales, weight_scale, rows,
                                 output_dim, error, error_size)) {
        return 0;
    }
    if (!bias) return 1;
    unsigned blocks = (unsigned)((outputs + 255) / 256);
    h3d_linear_bias_kernel<<<blocks, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)bias->data,
        outputs, output_dim);
    return h3d_launch_ok("INT8 convrot bias", error, error_size);
}

/* ---- the Gemma4 text tower's own primitives ----
 *
 * Source: `/data/refs/transformers-5.14.1/transformers/models/gemma4/
 * modeling_gemma4.py` (`Gemma4TextModel.forward` `:1650-1740` carries the
 * graph) plus the two PyTorch expressions the tower's numerics rest on, read at
 * `/data/refs/torch-2.13.0-aten`. Every kernel below is BF16 in and out with an
 * F32 intermediate, one reference expression each, with the reference's own
 * rounding structure: torch's elementwise ops on a half type compute in float
 * and round once, so a chain of two multiplies and an add rounds three times
 * and the port reproduces that rather than keeping one wide accumulator.
 *
 * The reduction orders are this port's, and they are the deviation the tower's
 * gate measures - no reference tree is reproducible here. */

/* `finfo(bfloat16).min`, the fill `eager_mask` writes for a masked position
 * (`masking_utils.py:608-610` with `dtype = inputs_embeds.dtype`, BF16). Finite
 * by construction, which is what makes a fully masked row softmax to uniform
 * rather than to NaN. */
#define H3D_GEMMA_MASK_FILL (-3.3895313892515355e+38f)
enum { H3D_GEMMA_WARPS = 8, H3D_GEMMA_THREADS = H3D_GEMMA_WARPS * 32 };

/* `embed_tokens(input_ids) * embed_scale.to(weight.dtype)`
 * (`modeling_gemma4.py:1476-1477`): one row gather, one BF16 product in float,
 * one rounding. `scale` is the caller's already-rounded BF16 value widened. */
__global__ static void h3d_embedding_kernel(__nv_bfloat16 *output,
                                            const __nv_bfloat16 *table,
                                            const uint32_t *ids, uint32_t width,
                                            float scale) {
    const uint32_t row = blockIdx.x;
    const __nv_bfloat16 *source = table + (size_t)ids[row] * width;
    __nv_bfloat16 *destination = output + (size_t)row * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        destination[column] = __float2bfloat16(
            __bfloat162float(source[column]) * scale);
    }
}

/* ATen's tanh gelu (`ActivationGeluKernel.cu:21-31`), the constants as its own
 * `constexpr` product so the compiler folds the same value, and `tanhf` for
 * `c10::cuda::compat::tanh` (`CUDAMathCompat.h:136-137`). `gpu_kernel`'s
 * functor returns a `scalar_t`, so the expression rounds once, at the end. */
__global__ static void h3d_gelu_tanh_kernel(__nv_bfloat16 *output,
                                            const __nv_bfloat16 *input,
                                            size_t elements) {
    constexpr float beta = 1.41421356237309504880f * 1.12837916709551257390f *
                           0.5f;
    constexpr float kappa = 0.044715f;
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float x = __bfloat162float(input[index]);
    const float cube = x * x * x;
    const float inner = beta * (x + kappa * cube);
    output[index] = __float2bfloat16(0.5f * x * (1.0f + tanhf(inner)));
}

/* `act_fn(gate_proj(x)) * up_proj(x)` (`modeling_gemma4.py:1089`). */
__global__ static void h3d_mul_kernel(__nv_bfloat16 *output,
                                      const __nv_bfloat16 *left,
                                      const __nv_bfloat16 *right,
                                      size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = __float2bfloat16(__bfloat162float(left[index]) *
                                     __bfloat162float(right[index]));
}

/* `hidden_states *= self.layer_scalar` (`:1462`), the scalar a BF16 `[1]`. */
__global__ static void h3d_scale_bf16_kernel(__nv_bfloat16 *output,
                                             const __nv_bfloat16 *input,
                                             const __nv_bfloat16 *scalar,
                                             size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = __float2bfloat16(__bfloat162float(input[index]) *
                                     __bfloat162float(*scalar));
}

/* `x * cos + rotate_half(x) * sin` (`:793-811`): `rotate_half` (`:786-791`)
 * pairs an index with its half-shifted partner and negates the first half's
 * partner, and the two products and the sum each round to BF16 as torch's
 * elementwise ops do. The row is snapshotted in shared memory first because
 * the partner of dimension d is written by the same block: without the
 * snapshot, thread d's write races thread d +/- half's read, and the rotation
 * reads a half-updated row. One thread per output dimension; the launch is
 * exactly `head_dim` threads, so no thread is inactive across the barrier. */
__global__ static void h3d_gemma_rope_kernel(__nv_bfloat16 *x,
                                             const __nv_bfloat16 *cos,
                                             const __nv_bfloat16 *sin,
                                             uint32_t rows, uint32_t heads,
                                             uint32_t head_dim) {
    __shared__ float staged[H3D_GEMMA_HEAD_DIM_MAX];
    const uint32_t row = blockIdx.x, head = blockIdx.y;
    const uint32_t dimension = threadIdx.x;
    const size_t base = ((size_t)row * heads + head) * head_dim;
    staged[dimension] = __bfloat162float(x[base + dimension]);
    __syncthreads();
    const uint32_t half = head_dim / 2;
    const uint32_t partner =
        dimension < half ? dimension + half : dimension - half;
    const float rotated =
        dimension < half ? -staged[partner] : staged[partner];
    const size_t table = (size_t)row * head_dim + dimension;
    const float turned = __bfloat162float(
        __float2bfloat16(staged[dimension] * __bfloat162float(cos[table])));
    const float spin = __bfloat162float(
        __float2bfloat16(rotated * __bfloat162float(sin[table])));
    x[base + dimension] = __float2bfloat16(turned + spin);
}

/* `eager_mask`'s keep rule (`masking_utils.py:608-610`), the sliding overlay
 * included (`:92-100`), and the padding mask (`:168-179`) which intersects on
 * the KEY index only. */
__global__ static void h3d_gemma_mask_kernel(__nv_bfloat16 *output,
                                             const uint32_t *padding,
                                             uint32_t rows, uint32_t window) {
    const uint32_t query_index = blockIdx.x;
    for (uint32_t key_index = threadIdx.x; key_index < rows;
         key_index += blockDim.x) {
        int keep = key_index <= query_index;
        if (keep && window) {
            keep = (int64_t)key_index > (int64_t)query_index - (int64_t)window;
        }
        if (keep && padding) keep = padding[key_index] != 0;
        output[(size_t)query_index * rows + key_index] =
            keep ? __float2bfloat16(0.0f)
                 : __float2bfloat16(H3D_GEMMA_MASK_FILL);
    }
}

/* `eager_attention_forward` (`modeling_gemma4.py:824-858`) at `scaling = 1.0`.
 * One block per (query row, head); eight warps.
 *
 *   scores: one warp per key stride, the 32 lanes covering the head and one
 *           shuffle tree per key; the dot rounds to BF16 before the mask is
 *           added, and the sum rounds again - torch's elementwise add.
 *   softmax: F32 over the BF16-rounded scores, the max and the sum each a
 *           block reduction with this port's tree.
 *   output: all eight warps walk the keys, one 1/8 of the head per warp, the
 *           normalised weight rounded to BF16 before each product, and one
 *           rounding at the end.
 *
 * The layouts are the port's: Q and the output `[rows][heads][head_dim]`, K and
 * V `[rows][kv_heads][head_dim]`. */
__global__ static void h3d_gemma_attention_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        const __nv_bfloat16 *mask, uint32_t rows, uint32_t heads,
        uint32_t kv_heads, uint32_t head_dim, float scale) {
    __shared__ float qv[H3D_GEMMA_HEAD_DIM_MAX];
    __shared__ float score[H3D_GEMMA_ROWS_MAX];
    __shared__ float partial[H3D_GEMMA_THREADS];

    const uint32_t row = blockIdx.x, head = blockIdx.y;
    const uint32_t source = head / (heads / kv_heads);
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const __nv_bfloat16 *q_row =
        query + ((size_t)row * heads + head) * head_dim;

    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        qv[i] = __bfloat162float(q_row[i]);
    }
    __syncthreads();

    for (uint32_t k = warp; k < rows; k += H3D_GEMMA_WARPS) {
        const __nv_bfloat16 *k_row =
            key + ((size_t)k * kv_heads + source) * head_dim;
        float dot = 0.0f;
        for (uint32_t i = lane; i < head_dim; i += 32) {
            dot = fmaf(qv[i], __bfloat162float(k_row[i]), dot);
        }
        for (uint32_t offset = 16; offset; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            float weighted = __bfloat162float(__float2bfloat16(dot * scale));
            if (mask) {
                weighted = __bfloat162float(__float2bfloat16(
                    weighted + __bfloat162float(mask[(size_t)row * rows + k])));
            }
            score[k] = weighted;
        }
    }
    __syncthreads();

    float maximum = -INFINITY;
    for (uint32_t k = threadIdx.x; k < rows; k += blockDim.x) {
        maximum = fmaxf(maximum, score[k]);
    }
    partial[threadIdx.x] = maximum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    maximum = partial[0];
    /* `partial` is reused for the sum below, so every thread must have read the
     * maximum before any thread overwrites its own slot: without this barrier a
     * warp that reaches its (empty) sum loop ahead of another warp's read lets
     * that read pick up the sum instead of the maximum. It is timing-dependent,
     * which is why the gate's repeated runs - and sm_121 - caught it. */
    __syncthreads();

    float total = 0.0f;
    for (uint32_t k = threadIdx.x; k < rows; k += blockDim.x) {
        const float weight = expf(score[k] - maximum);
        score[k] = weight;
        total += weight;
    }
    partial[threadIdx.x] = total;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse = 1.0f / partial[0];

    const uint32_t per_lane = head_dim / H3D_GEMMA_THREADS;
    const uint32_t begin =
        warp * (head_dim / H3D_GEMMA_WARPS) + lane * per_lane;
    float accumulator[2] = {0.0f, 0.0f};
    for (uint32_t k = 0; k < rows; ++k) {
        const float weight =
            __bfloat162float(__float2bfloat16(score[k] * inverse));
        const __nv_bfloat16 *v_row =
            value + ((size_t)k * kv_heads + source) * head_dim;
        for (uint32_t e = 0; e < per_lane; ++e) {
            accumulator[e] =
                fmaf(weight, __bfloat162float(v_row[begin + e]),
                     accumulator[e]);
        }
    }

    __nv_bfloat16 *out_row = output + ((size_t)row * heads + head) * head_dim;
    for (uint32_t e = 0; e < per_lane; ++e) {
        out_row[begin + e] = __float2bfloat16(accumulator[e]);
    }
}

int h3d_embedding(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *table, const h3d_tensor *ids,
                  uint32_t rows, uint32_t width, float scale, char *error,
                  size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width || !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(ids, H3D_U32, rows) || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(table, H3D_BF16, table->elements) ||
        table->elements % width != 0) {
        return h3d_set_error(error, error_size,
                             "invalid Gemma embedding tensors or shape");
    }
    h3d_embedding_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)table->data,
        (const uint32_t *)ids->data, width, scale);
    return h3d_launch_ok("Gemma embedding", error, error_size);
}

int h3d_gelu_tanh(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *input, size_t elements, char *error,
                  size_t error_size) {
    if (!context || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size, "invalid Gemma GELU tensors");
    }
    if (!elements) return 1;
    h3d_gelu_tanh_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                           context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        elements);
    return h3d_launch_ok("Gemma tanh GELU", error, error_size);
}

int h3d_mul(h3d_context *context, h3d_tensor *output, const h3d_tensor *left,
            const h3d_tensor *right, size_t elements, char *error,
            size_t error_size) {
    if (!context || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(left, H3D_BF16, elements) ||
        !h3d_is(right, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 multiply tensors");
    }
    if (!elements) return 1;
    h3d_mul_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                     context->stream>>>((__nv_bfloat16 *)output->data,
                                        (const __nv_bfloat16 *)left->data,
                                        (const __nv_bfloat16 *)right->data,
                                        elements);
    return h3d_launch_ok("BF16 multiply", error, error_size);
}

int h3d_scale_bf16(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, const h3d_tensor *scalar,
                   size_t elements, char *error, size_t error_size) {
    if (!context || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements) || !h3d_is(scalar, H3D_BF16, 1)) {
        return h3d_set_error(error, error_size, "invalid BF16 scale tensors");
    }
    if (!elements) return 1;
    h3d_scale_bf16_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                            context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)scalar->data, elements);
    return h3d_launch_ok("BF16 scale", error, error_size);
}

int h3d_gemma_rope(h3d_context *context, h3d_tensor *query, h3d_tensor *key,
                   const h3d_tensor *cos, const h3d_tensor *sin, uint32_t rows,
                   uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
                   char *error, size_t error_size) {
    size_t query_elements = 0, key_elements = 0, table_elements = 0;
    if (!context || !rows || !heads || !kv_heads || !head_dim ||
        head_dim > H3D_GEMMA_HEAD_DIM_MAX || (head_dim % 2) ||
        !h3d_mul_size((size_t)rows * heads, head_dim, &query_elements) ||
        !h3d_mul_size((size_t)rows * kv_heads, head_dim, &key_elements) ||
        !h3d_mul_size(rows, head_dim, &table_elements) ||
        !h3d_is(query, H3D_BF16, query_elements) ||
        !h3d_is(key, H3D_BF16, key_elements) ||
        !h3d_is(cos, H3D_BF16, table_elements) ||
        !h3d_is(sin, H3D_BF16, table_elements)) {
        return h3d_set_error(error, error_size,
                             "invalid Gemma RoPE tensors or shape");
    }
    dim3 query_grid(rows, heads), key_grid(rows, kv_heads);
    h3d_gemma_rope_kernel<<<query_grid, head_dim, 0, context->stream>>>(
        (__nv_bfloat16 *)query->data, (const __nv_bfloat16 *)cos->data,
        (const __nv_bfloat16 *)sin->data, rows, heads, head_dim);
    if (!h3d_launch_ok("Gemma RoPE query", error, error_size)) return 0;
    h3d_gemma_rope_kernel<<<key_grid, head_dim, 0, context->stream>>>(
        (__nv_bfloat16 *)key->data, (const __nv_bfloat16 *)cos->data,
        (const __nv_bfloat16 *)sin->data, rows, kv_heads, head_dim);
    return h3d_launch_ok("Gemma RoPE key", error, error_size);
}

/* `Gemma4TextRotaryEmbedding.forward` (`modeling_gemma4.py:1164-1181`): one
 * thread per (position, dimension) of the `[rows, dim]` pair tables. The
 * exponent is `2 * angle / dim` (`:1159`, `arange(0, dim, 2) / dim`), the
 * frequency `1 / base ** exponent`, the angle its product with the position,
 * and `cat(freqs, freqs)` is the `dimension % (dim / 2)` fold. Frequencies past
 * `angles` are zero, which is the identity pair. */
__global__ static void h3d_gemma_rope_table_kernel(__nv_bfloat16 *cos,
                                                   __nv_bfloat16 *sin,
                                                   uint32_t rows, uint32_t dim,
                                                   uint32_t angles,
                                                   float base) {
    const uint32_t position = blockIdx.y;
    const uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= rows || dimension >= dim) return;
    const uint32_t angle =
        dimension < dim / 2 ? dimension : dimension - dim / 2;
    float frequency = 0.0f;
    if (angle < angles) {
        frequency = 1.0f / powf(base, 2.0f * (float)angle / (float)dim);
    }
    const float value = frequency * (float)position;
    const size_t at = (size_t)position * dim + dimension;
    cos[at] = __float2bfloat16(cosf(value));
    sin[at] = __float2bfloat16(sinf(value));
}

int h3d_gemma_rope_table(h3d_context *context, h3d_tensor *cos,
                         h3d_tensor *sin, uint32_t rows, uint32_t dim,
                         uint32_t angles, float base, char *error,
                         size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !dim || !h3d_mul_size(rows, dim, &elements) ||
        angles > dim / 2 || !(base > 0.0f) ||
        !h3d_is(cos, H3D_BF16, elements) ||
        !h3d_is(sin, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid Gemma RoPE table arguments");
    }
    dim3 grid((dim + 255) / 256, rows);
    h3d_gemma_rope_table_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)cos->data, (__nv_bfloat16 *)sin->data, rows, dim,
        angles, base);
    return h3d_launch_ok("Gemma RoPE table", error, error_size);
}

int h3d_gemma_mask(h3d_context *context, h3d_tensor *output, uint32_t rows,
                   uint32_t window, const h3d_tensor *padding, char *error,
                   size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !h3d_mul_size(rows, rows, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        (padding && !h3d_is(padding, H3D_U32, rows))) {
        return h3d_set_error(error, error_size, "invalid Gemma mask tensors");
    }
    h3d_gemma_mask_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data,
        padding ? (const uint32_t *)padding->data : NULL, rows, window);
    return h3d_launch_ok("Gemma mask", error, error_size);
}

int h3d_gemma_attention(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *query, const h3d_tensor *key,
                        const h3d_tensor *value, const h3d_tensor *mask,
                        uint32_t rows, uint32_t heads, uint32_t kv_heads,
                        uint32_t head_dim, float scale, char *error,
                        size_t error_size) {
    size_t query_elements = 0, key_elements = 0, mask_elements = 0;
    if (!context || !rows || !heads || !kv_heads || !head_dim ||
        rows > H3D_GEMMA_ROWS_MAX || head_dim > H3D_GEMMA_HEAD_DIM_MAX ||
        heads % kv_heads || head_dim % H3D_GEMMA_THREADS ||
        !h3d_mul_size((size_t)rows * heads, head_dim, &query_elements) ||
        !h3d_mul_size((size_t)rows * kv_heads, head_dim, &key_elements) ||
        !h3d_mul_size(rows, rows, &mask_elements) ||
        !h3d_is(query, H3D_BF16, query_elements) ||
        !h3d_is(key, H3D_BF16, key_elements) ||
        !h3d_is(value, H3D_BF16, key_elements) ||
        !h3d_is(output, H3D_BF16, query_elements) ||
        (mask && !h3d_is(mask, H3D_BF16, mask_elements))) {
        return h3d_set_error(error, error_size,
                             "invalid Gemma attention tensors or shape");
    }
    dim3 grid(rows, heads);
    h3d_gemma_attention_kernel<<<grid, H3D_GEMMA_THREADS, 0,
                                 context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
        (const __nv_bfloat16 *)key->data, (const __nv_bfloat16 *)value->data,
        mask ? (const __nv_bfloat16 *)mask->data : NULL, rows, heads, kv_heads,
        head_dim, scale);
    return h3d_launch_ok("Gemma attention", error, error_size);
}

/* ---- the LTX AV connector and its feature extractor ----
 *
 * The conditioning the refiner consumes is Gemma-4-12B hidden states through
 * LTX's own 1D connector (`ltx-core/src/ltx_core/text_encoders/gemma/
 * embeddings_connector.py`, `feature_extractor.py`, `embeddings_processor.py`,
 * read at `/data/ltx2-pin/packages`). Those three files build their blocks
 * from LTX's transformer primitives - `rms_norm`, `Attention`, `FeedForward`,
 * the split RoPE and the per-head gate - which is the same machinery the
 * refiner itself needs, so the kernels below are the LTX side's first entry
 * points in this file and the refiner will reuse them.
 *
 * Six kernels. The connector's own norms, linears, SDPA, GELU and residual
 * adds are the generic `h3d_rms_norm`, `h3d_linear`, `h3d_sdpa`,
 * `h3d_gelu_tanh` and `h3d_add` above; what is left is the feature extractor's
 * per-(token, layer) RMS, the interleaved scatter that RMS's output needs, the
 * right-pad gather, the learnable-register fill, the split RoPE and the gate.
 * The RoPE tables are built on the host (`gemma_connector.rs`): the reference
 * builds its frequency grid in float64 numpy and casts (`rope.py:113-131`),
 * which has no device counterpart worth writing. */

/* `norm_and_concat_per_token_rms` (`feature_extractor.py:48-64`) up to the
 * rsqrt. One block per (row, layer) of the `[layers, rows, width]` stack: the
 * reference squares in BF16 *before* it accumulates (`:60`), sums in float,
 * divides by `width`, adds `1e-6` and rsqrts in torch's half-precision path,
 * so each of those four steps rounds to BF16. 256 partials and a halving tree
 * is this port's reduction order - the one deviation, and it is the order, not
 * the arithmetic. */
__global__ static void h3d_connector_scale_kernel(__nv_bfloat16 *scales,
                                                  const __nv_bfloat16 *stacked,
                                                  uint32_t rows,
                                                  uint32_t width) {
    __shared__ float partial[256];
    const uint32_t row = blockIdx.x, index = blockIdx.y;
    const __nv_bfloat16 *source =
        stacked + ((size_t)index * rows + row) * width;
    float total = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += 256) {
        const float value = __bfloat162float(source[column]);
        total += __bfloat162float(__float2bfloat16(value * value));
    }
    partial[threadIdx.x] = total;
    __syncthreads();
    for (uint32_t stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float mean = __bfloat162float(
            __float2bfloat16(partial[0] / (float)width));
        const float shifted =
            __bfloat162float(__float2bfloat16(mean + 1e-6f));
        scales[(size_t)row * gridDim.y + index] =
            __float2bfloat16(rsqrtf(shifted));
    }
}

/* The extractor's `normed.reshape(B, T, D * L)` (`:62`): the `[layers, rows,
 * width]` stack is written `[rows, width, layers]`, so the flattened column a
 * row's aggregate embed reads is `d * layers + l`, not `l * width + d`. A
 * block per (row, width tile) stages the tile in shared memory layer-major
 * (coalesced loads) and writes it output-major (coalesced stores); the layer's
 * scale multiplies on the way in. The tile's stride is padded to 66 so the
 * output-major walk does not put all 49 layers on one bank. */
__global__ static void h3d_connector_interleave_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *stacked,
        const __nv_bfloat16 *scales, uint32_t rows, uint32_t width,
        uint32_t layers) {
    extern __shared__ __align__(16) __nv_bfloat16 connector_tile[];
    const uint32_t row = blockIdx.y, begin = blockIdx.x * 64;
    const uint32_t count = begin + 64 <= width ? 64 : width - begin;
    const uint32_t total = count * layers;

    for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
        const uint32_t l = i / count, d = i - l * count;
        const __nv_bfloat16 *source =
            stacked + ((size_t)l * rows + row) * width + begin;
        const float scale = __bfloat162float(scales[(size_t)row * layers + l]);
        connector_tile[l * 66 + d] =
            __float2bfloat16(__bfloat162float(source[d]) * scale);
    }
    __syncthreads();

    __nv_bfloat16 *destination =
        output + (size_t)row * width * layers + (size_t)begin * layers;
    for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
        const uint32_t d = i / layers, l = i - d * layers;
        destination[i] = connector_tile[l * 66 + d];
    }
}

/* `_apply_right_pad_order` (`embeddings_processor.py:40-42`): `torch.gather`
 * along the sequence, the permutation `_compute_right_pad_order` (`:23-38`)
 * derives from the mask. The permutation itself is a stable argsort of 1024
 * keys, so it is computed on the host and uploaded. */
__global__ static void h3d_gather_rows_kernel(__nv_bfloat16 *output,
                                              const __nv_bfloat16 *input,
                                              const uint32_t *indices,
                                              uint32_t width) {
    const uint32_t row = blockIdx.x;
    const __nv_bfloat16 *source = input + (size_t)indices[row] * width;
    __nv_bfloat16 *destination = output + (size_t)row * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        destination[column] = source[column];
    }
}

/* `_replace_padded_with_learnable_registers`
 * (`embeddings_connector.py:139-151`): `binary * hidden + (1 - binary) *
 * registers`, the register tiled `seq // count` times. The additive mask the
 * reference hands the blocks afterwards is `zeros_like` (`:151`), i.e. every
 * key valid, so the connector's attention needs no mask at all. */
__global__ static void h3d_register_fill_kernel(__nv_bfloat16 *output,
                                                const __nv_bfloat16 *input,
                                                const uint32_t *binary,
                                                const __nv_bfloat16 *registers,
                                                uint32_t width,
                                                uint32_t count) {
    const uint32_t row = blockIdx.x;
    const __nv_bfloat16 *source =
        binary[row] ? input + (size_t)row * width
                    : registers + (size_t)(row % count) * width;
    __nv_bfloat16 *destination = output + (size_t)row * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        destination[column] = source[column];
    }
}

/* `apply_split_rotary_emb` (`rope.py:42-85`) with `cos`/`sin` `[rows, heads,
 * half]` (`split_freqs_cis` `:163-184`). The two halves of each head rotate
 * against each other: `first * cos - second * sin` and `second * cos +
 * first * sin`. The reference materialises `split * cos` as its own BF16
 * tensor (`:70`) and then `addcmul_`s in float, so each half rounds twice;
 * this keeps that order and takes the float add as the fused multiply-add it
 * compiles to. */
__device__ static void h3d_ltx_rotate(__nv_bfloat16 *x, size_t base,
                                      uint32_t half, uint32_t d, float c,
                                      float s) {
    const float first = __bfloat162float(x[base + d]);
    const float second = __bfloat162float(x[base + half + d]);
    const float turned_first = __bfloat162float(__float2bfloat16(first * c));
    const float turned_second = __bfloat162float(__float2bfloat16(second * c));
    x[base + d] = __float2bfloat16(fmaf(-s, second, turned_first));
    x[base + half + d] = __float2bfloat16(fmaf(s, first, turned_second));
}

__global__ static void h3d_ltx_rope_kernel(__nv_bfloat16 *query,
                                           __nv_bfloat16 *key,
                                           const __nv_bfloat16 *cos,
                                           const __nv_bfloat16 *sin,
                                           uint32_t rows, uint32_t heads,
                                           uint32_t head_dim) {
    const uint32_t row = blockIdx.x, head = blockIdx.y;
    const uint32_t half = head_dim / 2;
    const size_t base = ((size_t)row * heads + head) * head_dim;
    const size_t table = ((size_t)row * heads + head) * half;
    for (uint32_t d = threadIdx.x; d < half; d += blockDim.x) {
        const float c = __bfloat162float(cos[table + d]);
        const float s = __bfloat162float(sin[table + d]);
        h3d_ltx_rotate(query, base, half, d, c, s);
        h3d_ltx_rotate(key, base, half, d, c, s);
    }
}

/* `PytorchGatedAttention` (`ops.py:94-106`): `gates = 2 * sigmoid(logits)`
 * taken from the block's own input, broadcast over each head's dimensions and
 * applied to the attention output before `to_out`. The reference rounds
 * `2 * sigmoid` to BF16 as its own tensor (`:104`) and then each product. */
__global__ static void h3d_per_head_gate_kernel(__nv_bfloat16 *output,
                                                const __nv_bfloat16 *input,
                                                const __nv_bfloat16 *logits,
                                                uint32_t heads,
                                                uint32_t head_dim) {
    const uint32_t row = blockIdx.x, head = blockIdx.y;
    const float logit = __bfloat162float(logits[(size_t)row * heads + head]);
    const float gate = __bfloat162float(
        __float2bfloat16(2.0f * (1.0f / (1.0f + expf(-logit)))));
    const size_t base = ((size_t)row * heads + head) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        output[base + d] =
            __float2bfloat16(__bfloat162float(input[base + d]) * gate);
    }
}

/* `_rescale_norm` (`feature_extractor.py:68-69`): `x * sqrt(target / source)`
 * with the root a Python float, so the multiplier is the F32 value and the
 * product rounds once to BF16 - not a BF16-rounded scalar. */
__global__ static void h3d_scale_constant_kernel(__nv_bfloat16 *output,
                                                 const __nv_bfloat16 *input,
                                                 float factor, size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] =
        __float2bfloat16(__bfloat162float(input[index]) * factor);
}

int h3d_scale_constant(h3d_context *context, h3d_tensor *output,
                       const h3d_tensor *input, float factor,
                       size_t elements, char *error, size_t error_size) {
    if (!context || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 constant-scale tensors");
    }
    if (!elements) return 1;
    h3d_scale_constant_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                               context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        factor, elements);
    return h3d_launch_ok("BF16 constant scale", error, error_size);
}

int h3d_connector_stack_scale(h3d_context *context, h3d_tensor *scales,
                              const h3d_tensor *stacked, uint32_t rows,
                              uint32_t width, uint32_t layers, char *error,
                              size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width || !layers ||
        !h3d_mul_size((size_t)rows * layers, width, &elements) ||
        !h3d_is(scales, H3D_BF16, (size_t)rows * layers) ||
        !h3d_is(stacked, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid connector stack-scale tensors");
    }
    dim3 grid(rows, layers);
    h3d_connector_scale_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)scales->data, (const __nv_bfloat16 *)stacked->data,
        rows, width);
    return h3d_launch_ok("connector stack scale", error, error_size);
}

int h3d_connector_interleave(h3d_context *context, h3d_tensor *output,
                             const h3d_tensor *stacked,
                             const h3d_tensor *scales, uint32_t rows,
                             uint32_t width, uint32_t layers, char *error,
                             size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width || !layers ||
        !h3d_mul_size((size_t)rows * width, layers, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(stacked, H3D_BF16, elements) ||
        !h3d_is(scales, H3D_BF16, (size_t)rows * layers)) {
        return h3d_set_error(error, error_size,
                             "invalid connector interleave tensors");
    }
    dim3 grid((width + 63) / 64, rows);
    const size_t shared = (size_t)layers * 66 * sizeof(__nv_bfloat16);
    h3d_connector_interleave_kernel<<<grid, 256, shared, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)stacked->data,
        (const __nv_bfloat16 *)scales->data, rows, width, layers);
    return h3d_launch_ok("connector interleave", error, error_size);
}

int h3d_gather_rows(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, const h3d_tensor *indices,
                    uint32_t rows, uint32_t width, char *error,
                    size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(indices, H3D_U32, rows)) {
        return h3d_set_error(error, error_size, "invalid row gather tensors");
    }
    h3d_gather_rows_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const uint32_t *)indices->data, width);
    return h3d_launch_ok("row gather", error, error_size);
}

int h3d_register_fill(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *input, const h3d_tensor *binary,
                      const h3d_tensor *registers, uint32_t rows,
                      uint32_t width, uint32_t count, char *error,
                      size_t error_size) {
    size_t elements = 0, register_elements = 0;
    if (!context || !rows || !width || !count ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_mul_size(count, width, &register_elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(binary, H3D_U32, rows) ||
        !h3d_is(registers, H3D_BF16, register_elements)) {
        return h3d_set_error(error, error_size,
                             "invalid register-fill tensors");
    }
    h3d_register_fill_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const uint32_t *)binary->data,
        (const __nv_bfloat16 *)registers->data, width, count);
    return h3d_launch_ok("register fill", error, error_size);
}

int h3d_ltx_rope(h3d_context *context, h3d_tensor *query, h3d_tensor *key,
                 const h3d_tensor *cos, const h3d_tensor *sin, uint32_t rows,
                 uint32_t heads, uint32_t head_dim, char *error,
                 size_t error_size) {
    size_t elements = 0, table_elements = 0;
    if (!context || !rows || !heads || !head_dim || (head_dim % 2) ||
        !h3d_mul_size((size_t)rows * heads, head_dim, &elements) ||
        !h3d_mul_size((size_t)rows * heads, head_dim / 2, &table_elements) ||
        !h3d_is(query, H3D_BF16, elements) ||
        !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(cos, H3D_BF16, table_elements) ||
        !h3d_is(sin, H3D_BF16, table_elements)) {
        return h3d_set_error(error, error_size, "invalid LTX RoPE tensors");
    }
    dim3 grid(rows, heads);
    h3d_ltx_rope_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (const __nv_bfloat16 *)cos->data, (const __nv_bfloat16 *)sin->data,
        rows, heads, head_dim);
    return h3d_launch_ok("LTX RoPE", error, error_size);
}

int h3d_per_head_gate(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *input, const h3d_tensor *logits,
                      uint32_t rows, uint32_t heads, uint32_t head_dim,
                      char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !heads || !head_dim ||
        !h3d_mul_size((size_t)rows * heads, head_dim, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(logits, H3D_BF16, (size_t)rows * heads)) {
        return h3d_set_error(error, error_size,
                             "invalid per-head gate tensors");
    }
    dim3 grid(rows, heads);
    h3d_per_head_gate_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)logits->data, heads, head_dim);
    return h3d_launch_ok("per-head gate", error, error_size);
}

/* The video VAE decoder's F32 linear: the reference's `h3_gpu_linear_f32`
 * (`:832-870`) with this port's compute-mode option applied. Only the decoder
 * calls it (`minimax_h3::vae_video`), so `H3_VAE_GEMM` cannot reach the video
 * VAE encoder or the audio VAE, whose gates compare against the reference bit
 * for bit. The bias add stays the reference's separate `h3_linear_bias_kernel`
 * in every mode: cuBLASLt refuses `CUBLASLT_EPILOGUE_BIAS` for this row-major
 * TRANSB configuration with CUBLAS_STATUS_NOT_SUPPORTED (measured on GB10
 * sm_121 CUDA 13.0 and on sm_89 CUDA 13.3), so folding the bias into the
 * epilogue is not available here and the arm was dropped rather than faked. */
int h3d_vae_linear(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, const h3d_tensor *weight,
                   const h3d_tensor *bias, uint32_t rows, uint32_t input_dim,
                   uint32_t output_dim, char *error, size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        !h3d_is(input, H3D_F32, inputs) || !h3d_is(weight, H3D_F32, weights) ||
        !h3d_is(output, H3D_F32, outputs) ||
        (bias && !h3d_is(bias, H3D_F32, output_dim))) {
        return h3d_set_error(error, error_size,
                             "invalid video VAE linear tensors or shape");
    }

    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    if (context->vae.gemm == H3D_VAE_GEMM_TF32) {
        compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    } else if (context->vae.gemm == H3D_VAE_GEMM_BF16 ||
               /* The `bf16i` arm serves only the four block GEMMs
                * (`h3d_vae_linear_bf16`); the projections around them are F32
                * operands and keep the FAST class the F32 route runs, which is
                * what this entry point gives them. */
               context->vae.gemm == H3D_VAE_GEMM_BF16_INPUTS) {
        compute = CUBLAS_COMPUTE_32F_FAST_16BF;
    }
    if (!h3d_linear_lt_compute(context, output->data, input->data,
                               weight->data, rows, input_dim, output_dim,
                               CUDA_R_32F, compute, error, error_size)) {
        return 0;
    }
    if (!bias) return 1;

    unsigned blocks = (unsigned)((outputs + 255) / 256);
    h3d_linear_bias_kernel<<<blocks, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)bias->data, outputs, output_dim);
    return h3d_launch_ok("video VAE linear bias", error, error_size);
}

/* The `bf16i` arm's block GEMM: the same cuBLASLt call as `h3d_vae_linear`'s,
 * with BF16 operands and a BF16 output instead of F32. `tests/h3/vae_gemm_bm`
 * priced this class at 3.634 ms a block against the 11.283 the FAST modes
 * reach, so it is the one lever section 8.3 of the 2026-10-29 attribution
 * leaves in the decoder.
 *
 * The output has to be BF16 because cuBLASLt refuses BF16 operands with an F32
 * output on both hosts (sm_89 `CUBLAS_STATUS_INTERNAL_ERROR`, sm_121 a segfault
 * inside the call), which is why the arm narrows its activation, computes, and
 * widens the result back for the F32 consumers. No bias argument: every block
 * linear's bias is added by its consumer since `c7bb59e1`, and this arm has no
 * F32 output for `h3d_linear_bias_kernel` to round. */
int h3d_vae_linear_bf16(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                        char *error, size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        !h3d_is(input, H3D_BF16, inputs) || !h3d_is(weight, H3D_BF16, weights) ||
        !h3d_is(output, H3D_BF16, outputs)) {
        return h3d_set_error(error, error_size,
                             "invalid video VAE BF16 linear tensors or shape");
    }
    return h3d_linear_lt_compute(context, output->data, input->data,
                                 weight->data, rows, input_dim, output_dim,
                                 CUDA_R_16BF, CUBLAS_COMPUTE_32F, error,
                                 error_size);
}

int h3d_patch_linear(h3d_context *context, h3d_tensor *output,
                     size_t output_element, const h3d_tensor *input,
                     size_t input_element, const h3d_tensor *weight,
                     const h3d_tensor *bias, uint32_t rows, uint32_t input_dim,
                     uint32_t output_dim, char *error, size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        input_element > SIZE_MAX - inputs ||
        output_element > SIZE_MAX - outputs ||
        !h3d_is(input, H3D_F32, input_element + inputs) ||
        !h3d_is(weight, H3D_F32, weights) ||
        !h3d_is(output, H3D_BF16, output_element + outputs) ||
        (bias && !h3d_is(bias, H3D_F32, output_dim))) {
        return h3d_set_error(error, error_size,
                             "invalid patch linear tensors or shape");
    }

    const float *input_data = (const float *)input->data + input_element;
    __nv_bfloat16 *output_data =
        (__nv_bfloat16 *)output->data + output_element;
    float *temporary = NULL;
    cudaError_t status = cudaMallocAsync((void **)&temporary,
                                         outputs * sizeof(*temporary),
                                         context->stream);
    if (!h3d_cuda_ok(status, "patch linear temporary allocation", error,
                     error_size)) {
        return 0;
    }

    int ok = h3d_linear_lt(context, temporary, input_data, weight->data, rows,
                           input_dim, output_dim, CUDA_R_32F, error,
                           error_size);
    if (ok) {
        unsigned blocks = (unsigned)((outputs + 255) / 256);
        h3d_patch_convert_kernel<<<blocks, 256, 0, context->stream>>>(
            output_data, temporary, bias ? (const float *)bias->data : temporary,
            outputs, output_dim, bias != NULL);
        ok = h3d_launch_ok("patch linear conversion", error, error_size);
    }

    status = cudaFreeAsync(temporary, context->stream);
    if (ok) {
        ok = h3d_cuda_ok(status, "patch linear temporary free", error,
                         error_size);
    }
    return ok;
}

/* `h3_patch_scatter_kernel` (`:917-930`): project `rows` source rows, add the
 * bias in F32, round once, and write each to `row_map[row]`. Rows whose map
 * entry is past the output are dropped. */
__global__ static void h3d_patch_scatter_kernel(
        __nv_bfloat16 *output, const float *input, const float *bias,
        const uint32_t *row_map, uint32_t output_rows, uint32_t rows,
        uint32_t columns, int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t elements = (size_t)rows * columns;
    if (index >= elements) return;
    uint32_t source_row = (uint32_t)(index / columns);
    uint32_t destination_row = row_map[source_row];
    if (destination_row < output_rows) {
        output[(size_t)destination_row * columns + index % columns] =
            __float2bfloat16(input[index] +
                             (has_bias ? bias[index % columns] : 0.0f));
    }
}

int h3d_patch_linear_map(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, const h3d_tensor *weight,
                         const h3d_tensor *bias, const h3d_tensor *row_map,
                         uint32_t output_rows, uint32_t rows,
                         uint32_t input_dim, uint32_t output_dim, char *error,
                         size_t error_size) {
    size_t inputs = 0, weights = 0, outputs = 0, mapped = 0;
    if (!context || !rows || !output_rows || !output_dim ||
        !h3d_mul_size(rows, input_dim, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(output_rows, output_dim, &outputs) ||
        !h3d_mul_size(rows, output_dim, &mapped) ||
        !h3d_is(input, H3D_F32, inputs) ||
        !h3d_is(weight, H3D_F32, weights) ||
        !h3d_is(output, H3D_BF16, outputs) ||
        !h3d_is(row_map, H3D_U32, rows) ||
        (bias && !h3d_is(bias, H3D_F32, output_dim))) {
        return h3d_set_error(error, error_size,
                             "invalid mapped patch linear tensors or shape");
    }

    float *temporary = NULL;
    cudaError_t status = cudaMallocAsync((void **)&temporary,
                                         mapped * sizeof(*temporary),
                                         context->stream);
    if (!h3d_cuda_ok(status, "mapped patch temporary allocation", error,
                     error_size)) {
        return 0;
    }

    int ok = h3d_linear_lt(context, temporary, input->data, weight->data, rows,
                           input_dim, output_dim, CUDA_R_32F, error,
                           error_size);
    if (ok) {
        unsigned blocks = (unsigned)((mapped + 255) / 256);
        h3d_patch_scatter_kernel<<<blocks, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data, temporary,
            bias ? (const float *)bias->data : temporary,
            (const uint32_t *)row_map->data, output_rows, rows, output_dim,
            bias != NULL);
        ok = h3d_launch_ok("mapped patch scatter", error, error_size);
    }

    status = cudaFreeAsync(temporary, context->stream);
    if (ok) {
        ok = h3d_cuda_ok(status, "mapped patch temporary free", error,
                         error_size);
    }
    return ok;
}

/* `H3_DIT_ELEMENTWISE` selects the spelling of the two elementwise kernels.
 * Unset means the staged pair below; `reference` restores the port's first
 * spelling, which is the reference's own (`h3_gpu_cuda.cu:2656`, `:3079`), for
 * an A/B in one binary. Read once per process. */
static int h3d_elementwise_reference(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("H3_DIT_ELEMENTWISE");
        cached = value && strcmp(value, "reference") == 0;
    }
    return cached;
}

/* Shared-memory row the staged AdaLN kernel walks, the elements one vector
 * read covers, and the head dimension the staged QKV/RoPE kernel stages. 8,192
 * bf16 is 16 KiB, which still admits six blocks per SM beside the
 * 1,536-thread limit. */
#define H3D_ADALN_STAGED_WIDTH 8192
#define H3D_QKV_STAGED_DIM 256
#define H3D_ELEMENTWISE_VECTOR 8

/* `h3_adaln_kernel` (`:2656-2687`): thread 0 owns the RMS sum, one block per
 * row, and the modulation row is selected per row through `row_map`. The chain
 * is one lane's serial accumulation, so it is kept as the reference spelling
 * and not as the default: a lane's 2-byte loads leave the other 255 threads
 * parked and expose the whole row's latency. */
__global__ static void h3d_adaln_reference_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *input,
        const __nv_bfloat16 *weight, const __nv_bfloat16 *modulation,
        const uint32_t *row_map, uint32_t rows, uint32_t width, uint32_t slots,
        uint32_t shift_slot, uint32_t scale_slot, float epsilon,
        size_t input_offset) {
    __shared__ float inverse;
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *source = input + input_offset + (size_t)row * width;
    if (threadIdx.x == 0) {
        float square_sum = 0.0f;
        for (uint32_t k = 0; k < width; ++k) {
            float value = h3d_value(source[k]);
            square_sum = fmaf(value, value, square_sum);
        }
        inverse = rsqrtf(square_sum / (float)width + epsilon);
    }
    __syncthreads();
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float normalized = h3d_value(source[column]) * inverse *
                           h3d_value(weight[column]);
        float shift =
            h3d_value(modulation[base + (size_t)shift_slot * width + column]);
        float scale =
            h3d_value(modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3d_store(normalized * (1.0f + scale) + shift);
    }
}

/* The same arithmetic with the row staged in shared memory once: the block
 * copies it with its 256 threads (coalesced, so the reads are in flight
 * together), one lane walks the copy in the same index order, and the
 * elementwise pass reads it there instead of a second global pass. The
 * accumulation order and every rounding is the reference's, so this kernel is
 * bit-identical to `h3d_adaln_reference_kernel`; only the traffic changes. */
__global__ static void h3d_adaln_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *input,
        const __nv_bfloat16 *weight, const __nv_bfloat16 *modulation,
        const uint32_t *row_map, uint32_t rows, uint32_t width, uint32_t slots,
        uint32_t shift_slot, uint32_t scale_slot, float epsilon,
        size_t input_offset) {
    __shared__ __align__(16) __nv_bfloat16 staged[H3D_ADALN_STAGED_WIDTH];
    __shared__ float inverse;
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *source = input + input_offset + (size_t)row * width;

    /* The block copies the row in 16-byte units when the row's start and the
     * width allow it, which is the model's case; the scalar copy is the
     * fallback, and the only difference is the instruction count. */
    const int vector = (((uintptr_t)source & 15u) == 0) &&
                       (width % H3D_ELEMENTWISE_VECTOR == 0);
    if (vector) {
        for (uint32_t k = threadIdx.x * H3D_ELEMENTWISE_VECTOR; k < width;
             k += blockDim.x * H3D_ELEMENTWISE_VECTOR) {
            *(uint4 *)(staged + k) = *(const uint4 *)(source + k);
        }
    } else {
        for (uint32_t k = threadIdx.x; k < width; k += blockDim.x) {
            staged[k] = source[k];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float square_sum = 0.0f;
        for (uint32_t k = 0; k < width; ++k) {
            float value = h3d_value(staged[k]);
            square_sum = fmaf(value, value, square_sum);
        }
        inverse = rsqrtf(square_sum / (float)width + epsilon);
    }
    __syncthreads();
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float normalized = h3d_value(staged[column]) * inverse *
                           h3d_value(weight[column]);
        float shift =
            h3d_value(modulation[base + (size_t)shift_slot * width + column]);
        float scale =
            h3d_value(modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3d_store(normalized * (1.0f + scale) + shift);
    }
}

int h3d_adaln(h3d_context *context, h3d_tensor *output,
              const h3d_tensor *input, size_t input_element,
              const h3d_tensor *norm_weight, const h3d_tensor *modulation,
              const h3d_tensor *row_map, uint32_t rows, uint32_t width,
              uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
              float epsilon, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !h3d_mul_size(rows, width, &elements) || !slots ||
        shift_slot >= slots || scale_slot >= slots || epsilon < 0.0f ||
        input_element > SIZE_MAX - elements ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, input_element + elements) ||
        !h3d_is(norm_weight, H3D_BF16, width) ||
        !h3d_is(modulation, H3D_BF16, (size_t)slots * width) ||
        !h3d_is(row_map, H3D_U32, rows)) {
        return h3d_set_error(error, error_size, "invalid AdaLN arguments");
    }
    if (h3d_elementwise_reference() || width > H3D_ADALN_STAGED_WIDTH) {
        h3d_adaln_reference_kernel<<<rows, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)norm_weight->data,
            (const __nv_bfloat16 *)modulation->data,
            (const uint32_t *)row_map->data, rows, width, slots, shift_slot,
            scale_slot, epsilon, input_element);
    } else {
        h3d_adaln_kernel<<<rows, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)norm_weight->data,
            (const __nv_bfloat16 *)modulation->data,
            (const uint32_t *)row_map->data, rows, width, slots, shift_slot,
            scale_slot, epsilon, input_element);
    }
    return h3d_launch_ok("BF16 AdaLN", error, error_size);
}

/* Long video/reference rows use grid.x; CUDA grid.y/z stop at 65,535. */
/* `h3_gate_kernel` (`:2689-2702`). */
__global__ static void h3d_gate_kernel(__nv_bfloat16 *output,
                                       const __nv_bfloat16 *residual,
                                       const __nv_bfloat16 *branch,
                                       const __nv_bfloat16 *modulation,
                                       const uint32_t *row_map, uint32_t rows,
                                       uint32_t width, uint32_t slots,
                                       uint32_t gate_slot) {
    uint32_t column = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.x;
    if (row >= rows || column >= width) return;
    size_t index = (size_t)row * width + column;
    size_t base = (size_t)row_map[row] * slots * width;
    float gate = h3d_value(modulation[base + (size_t)gate_slot * width + column]);
    output[index] =
        h3d_store(h3d_value(residual[index]) + h3d_value(branch[index]) * gate);
}

int h3d_gate(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *residual, const h3d_tensor *branch,
             const h3d_tensor *modulation, const h3d_tensor *row_map,
             uint32_t rows, uint32_t width, uint32_t slots,
             uint32_t gate_slot, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !h3d_mul_size(rows, width, &elements) || !slots ||
        gate_slot >= slots || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(residual, H3D_BF16, elements) ||
        !h3d_is(branch, H3D_BF16, elements) ||
        !h3d_is(modulation, H3D_BF16, (size_t)slots * width) ||
        !h3d_is(row_map, H3D_U32, rows)) {
        return h3d_set_error(error, error_size, "invalid gate arguments");
    }
    dim3 grid(rows, (width + 255) / 256);
    h3d_gate_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)residual->data,
        (const __nv_bfloat16 *)branch->data,
        (const __nv_bfloat16 *)modulation->data,
        (const uint32_t *)row_map->data, rows, width, slots, gate_slot);
    return h3d_launch_ok("BF16 gate", error, error_size);
}

/* ---------------------------------------------------------------------------
 * Token reduction: pool paired video columns before a block and expand the
 * reduced stack's delta back after it (`h3_dit.c:378-470` configures the
 * window; these two kernels are the arithmetic).
 * ------------------------------------------------------------------------- */

/* `h3_token_pool_kernel` (`:3271-3295`). */
__global__ static void h3d_token_pool_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *input,
        size_t input_offset, __nv_bfloat16 *original, size_t original_offset,
        __nv_bfloat16 *baseline, size_t baseline_offset,
        const uint32_t *baseline_indices, const uint32_t *pairs,
        uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.x;
    if (row >= rows || column >= width) return;
    uint32_t first_row = pairs[(size_t)row * 2];
    uint32_t second_row = pairs[(size_t)row * 2 + 1];
    __nv_bfloat16 first =
        input[input_offset + (size_t)first_row * width + column];
    original[original_offset + (size_t)first_row * width + column] = first;
    __nv_bfloat16 pooled = first;
    if (first_row != second_row) {
        __nv_bfloat16 second =
            input[input_offset + (size_t)second_row * width + column];
        original[original_offset + (size_t)second_row * width + column] =
            second;
        pooled = __float2bfloat16(
            (__bfloat162float(first) + __bfloat162float(second)) * 0.5f);
    }
    output[(size_t)row * width + column] = pooled;
    uint32_t baseline_row = baseline_indices[row];
    if (baseline_row != UINT32_MAX) {
        baseline[baseline_offset + (size_t)baseline_row * width + column] =
            pooled;
    }
}

int h3d_token_pool(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, size_t input_element,
                   h3d_tensor *original, size_t original_element,
                   h3d_tensor *baseline, size_t baseline_element,
                   const h3d_tensor *baseline_indices,
                   const h3d_tensor *pairs, uint32_t input_rows,
                   uint32_t rows, uint32_t baseline_rows, uint32_t width,
                   char *error, size_t error_size) {
    size_t input_elements = 0, output_elements = 0, baseline_elements = 0;
    if (!context ||
        !h3d_mul_size(input_rows, width, &input_elements) ||
        !h3d_mul_size(rows, width, &output_elements) ||
        !h3d_mul_size(baseline_rows, width, &baseline_elements) ||
        input_element > SIZE_MAX - input_elements ||
        original_element > SIZE_MAX - input_elements ||
        baseline_element > SIZE_MAX - baseline_elements ||
        !h3d_is(output, H3D_BF16, output_elements) ||
        !h3d_is(input, H3D_BF16, input_element + input_elements) ||
        !h3d_is(original, H3D_BF16, original_element + input_elements) ||
        !h3d_is(baseline, H3D_BF16, baseline_element + baseline_elements) ||
        !h3d_is(baseline_indices, H3D_U32, rows) ||
        rows > SIZE_MAX / 2 ||
        !h3d_is(pairs, H3D_U32, (size_t)rows * 2)) {
        return h3d_set_error(error, error_size, "invalid token pool arguments");
    }
    dim3 grid(rows, (width + 255) / 256);
    h3d_token_pool_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        input_element, (__nv_bfloat16 *)original->data, original_element,
        (__nv_bfloat16 *)baseline->data, baseline_element,
        (const uint32_t *)baseline_indices->data,
        (const uint32_t *)pairs->data, rows, width);
    return h3d_launch_ok("BF16 token pool", error, error_size);
}

/* `h3_token_expand_kernel` (`:3328-3351`). */
__global__ static void h3d_token_expand_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *original,
        size_t original_offset, const __nv_bfloat16 *reduced,
        const __nv_bfloat16 *baseline, size_t baseline_offset,
        const uint32_t *baseline_indices, const uint32_t *parents,
        uint32_t rows, uint32_t width, uint32_t exact_prefix_rows,
        float update_scale) {
    uint32_t column = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.x;
    if (row >= rows || column >= width) return;
    uint32_t parent = parents[row];
    size_t destination = (size_t)row * width + column;
    size_t reduced_index = (size_t)parent * width + column;
    uint32_t baseline_row = baseline_indices[parent];
    if (row < exact_prefix_rows || baseline_row == UINT32_MAX) {
        output[destination] = reduced[reduced_index];
        return;
    }
    float update = __bfloat162float(reduced[reduced_index]) -
                   __bfloat162float(baseline[baseline_offset +
                                             (size_t)baseline_row * width +
                                             column]);
    output[destination] = __float2bfloat16(
        __bfloat162float(original[original_offset + destination]) +
        update_scale * update);
}

int h3d_token_expand(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *original, size_t original_element,
                     const h3d_tensor *reduced, const h3d_tensor *baseline,
                     size_t baseline_element,
                     const h3d_tensor *baseline_indices,
                     const h3d_tensor *parents, uint32_t rows,
                     uint32_t reduced_rows, uint32_t baseline_rows,
                     uint32_t width, uint32_t prefix_rows, float scale,
                     char *error, size_t error_size) {
    size_t output_elements = 0, reduced_elements = 0, baseline_elements = 0;
    if (!context ||
        !h3d_mul_size(rows, width, &output_elements) ||
        !h3d_mul_size(reduced_rows, width, &reduced_elements) ||
        !h3d_mul_size(baseline_rows, width, &baseline_elements) ||
        prefix_rows > rows || original_element > SIZE_MAX - output_elements ||
        baseline_element > SIZE_MAX - baseline_elements ||
        !h3d_is(output, H3D_BF16, output_elements) ||
        !h3d_is(original, H3D_BF16, original_element + output_elements) ||
        !h3d_is(reduced, H3D_BF16, reduced_elements) ||
        !h3d_is(baseline, H3D_BF16, baseline_element + baseline_elements) ||
        !h3d_is(baseline_indices, H3D_U32, reduced_rows) ||
        !h3d_is(parents, H3D_U32, rows)) {
        return h3d_set_error(error, error_size,
                             "invalid token expand arguments");
    }
    dim3 grid(rows, (width + 255) / 256);
    h3d_token_expand_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)original->data, original_element,
        (const __nv_bfloat16 *)reduced->data,
        (const __nv_bfloat16 *)baseline->data, baseline_element,
        (const uint32_t *)baseline_indices->data,
        (const uint32_t *)parents->data, rows, width, prefix_rows, scale);
    return h3d_launch_ok("BF16 token expand", error, error_size);
}

/* `h3_rms_norm_bf16_kernel` (`:2462-2489`): 256 threads, a strided fmaf sum,
 * a shared-memory tree, then `value * inverse * weight`. The DiT always passes
 * a weight; Gemma's `v_norm` is built `with_scale=False` and passes NULL, which
 * is the reference's own missing `* weight` branch. */
__global__ static void h3d_rms_norm_kernel(__nv_bfloat16 *output,
                                           const __nv_bfloat16 *input,
                                           const __nv_bfloat16 *weight,
                                           uint32_t rows, uint32_t width,
                                           float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = __bfloat162float(input[(size_t)row * width + column]);
        sum = fmaf(value, value, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        float value = __bfloat162float(input[index]) * inverse;
        if (weight) value *= __bfloat162float(weight[column]);
        output[index] = __float2bfloat16(value);
    }
}

int h3d_rms_norm(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, const h3d_tensor *weight,
                 uint32_t rows, uint32_t width, float epsilon, char *error,
                 size_t error_size) {
    size_t elements = 0;
    if (!context || !h3d_mul_size(rows, width, &elements) || epsilon < 0.0f ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        (weight && !h3d_is(weight, H3D_BF16, width))) {
        return h3d_set_error(error, error_size, "invalid RMS norm arguments");
    }
    h3d_rms_norm_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        weight ? (const __nv_bfloat16 *)weight->data : NULL, rows, width,
        epsilon);
    return h3d_launch_ok("BF16 RMS norm", error, error_size);
}

/* `h3_swiglu_kernel` (`:2841-2855`). */
__global__ static void h3d_swiglu_kernel(__nv_bfloat16 *output,
                                         const __nv_bfloat16 *fused,
                                         uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.x;
    if (row >= rows || column >= width) return;
    size_t base = (size_t)row * width * 2;
    float gate = __bfloat162float(fused[base + column]);
    float up = __bfloat162float(fused[base + width + column]);
    output[(size_t)row * width + column] =
        __float2bfloat16(gate / (1.0f + expf(-gate)) * up);
}

int h3d_swiglu(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *fused, uint32_t rows, uint32_t width,
               char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(fused, H3D_BF16, elements * 2)) {
        return h3d_set_error(error, error_size, "invalid SwiGLU arguments");
    }
    dim3 grid(rows, (width + 255) / 256);
    h3d_swiglu_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)fused->data,
        rows, width);
    return h3d_launch_ok("BF16 SwiGLU", error, error_size);
}

/* `h3_qkv_rope_kernel` (`:3079-3136`) with `grouped = 1`, `normalize = 1`,
 * `weighted = 1`: the checkpoint's [head][q,k,v][dim] rows, Q/K RMS over the
 * head, then the pair rotation on the first `2 * rope_half` dimensions. Kept
 * as the reference spelling: every one of the block's 128 threads walks the
 * whole 128-element Q and K vectors for the two sums, so the block issues 512
 * broadcasts of the same 512 bytes. */
__global__ static void h3d_qkv_rope_reference_kernel(
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
        const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t dimension = blockIdx.z * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.x;
    if (row >= sequence || head >= heads || dimension >= head_dim) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim * 3;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float q = __bfloat162float(qkv[q_base + d]);
        float k = __bfloat162float(qkv[k_base + d]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = rsqrtf(q_sum / (float)head_dim + epsilon);
    float k_inverse = rsqrtf(k_sum / (float)head_dim + epsilon);

    float qw = __bfloat162float(q_weight[dimension]);
    float kw = __bfloat162float(k_weight[dimension]);
    float q0 = __bfloat162float(qkv[q_base + dimension]) * q_inverse * qw;
    float k0 = __bfloat162float(qkv[k_base + dimension]) * k_inverse * kw;
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = __bfloat162float(qkv[q_base + pair]) * q_inverse *
                   __bfloat162float(q_weight[pair]);
        float k1 = __bfloat162float(qkv[k_base + pair]) * k_inverse *
                   __bfloat162float(k_weight[pair]);
        float c = __bfloat162float(rope_cos[(size_t)row * rope_half + rope_index]);
        float s = __bfloat162float(rope_sin[(size_t)row * rope_half + rope_index]);
        if (dimension < rope_half) {
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else {
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
    }
    size_t output_index = ((size_t)head * sequence + row) * head_dim + dimension;
    query[output_index] = __float2bfloat16(q0);
    key[output_index] = __float2bfloat16(k0);
    value[output_index] = qkv[v_base + dimension];
}

/* The same arithmetic with the head staged in shared memory once: the block
 * copies its Q and K with 128 coalesced reads, writes V on the way (it is a
 * copy of `qkv`'s third block, untouched by the rotation), one lane walks the
 * copy in the same index order for both sums, and the epilogue reads it there
 * - the second read of Q and K was already in the block's L1. Bit-identical to
 * `h3d_qkv_rope_reference_kernel`; only the traffic changes. */
__global__ static void h3d_qkv_rope_kernel(
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
        const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon) {
    __shared__ __align__(16) __nv_bfloat16 staged[2 * H3D_QKV_STAGED_DIM];
    __shared__ float inverse[2];
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.x;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim * 3;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        staged[d] = qkv[q_base + d];
        staged[head_dim + d] = qkv[k_base + d];
        value[((size_t)head * sequence + row) * head_dim + d] = qkv[v_base + d];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float q_sum = 0.0f, k_sum = 0.0f;
        const uint32_t bulk = head_dim - (head_dim % H3D_ELEMENTWISE_VECTOR);
        for (uint32_t d = 0; d < bulk; d += H3D_ELEMENTWISE_VECTOR) {
            uint4 q_word = *(const uint4 *)(staged + d);
            uint4 k_word = *(const uint4 *)(staged + head_dim + d);
            uint32_t q_parts[4] = {q_word.x, q_word.y, q_word.z, q_word.w};
            uint32_t k_parts[4] = {k_word.x, k_word.y, k_word.z, k_word.w};
            #pragma unroll
            for (uint32_t part = 0; part < 4; ++part) {
                float q_low = h3d_value(
                    __ushort_as_bfloat16((unsigned short)q_parts[part]));
                float k_low = h3d_value(
                    __ushort_as_bfloat16((unsigned short)k_parts[part]));
                float q_high = h3d_value(
                    __ushort_as_bfloat16((unsigned short)(q_parts[part] >> 16)));
                float k_high = h3d_value(
                    __ushort_as_bfloat16((unsigned short)(k_parts[part] >> 16)));
                q_sum = fmaf(q_low, q_low, q_sum);
                k_sum = fmaf(k_low, k_low, k_sum);
                q_sum = fmaf(q_high, q_high, q_sum);
                k_sum = fmaf(k_high, k_high, k_sum);
            }
        }
        for (uint32_t d = bulk; d < head_dim; d++) {
            float q = h3d_value(staged[d]);
            float k = h3d_value(staged[head_dim + d]);
            q_sum = fmaf(q, q, q_sum);
            k_sum = fmaf(k, k, k_sum);
        }
        inverse[0] = rsqrtf(q_sum / (float)head_dim + epsilon);
        inverse[1] = rsqrtf(k_sum / (float)head_dim + epsilon);
    }
    __syncthreads();
    float q_inverse = inverse[0];
    float k_inverse = inverse[1];

    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float qw = h3d_value(q_weight[dimension]);
        float kw = h3d_value(k_weight[dimension]);
        float q0 = h3d_value(staged[dimension]) * q_inverse * qw;
        float k0 = h3d_value(staged[head_dim + dimension]) * k_inverse * kw;
        if (dimension < rope_half * 2) {
            uint32_t rope_index = dimension % rope_half;
            uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                     dimension - rope_half;
            float q1 = h3d_value(staged[pair]) * q_inverse *
                       h3d_value(q_weight[pair]);
            float k1 = h3d_value(staged[head_dim + pair]) * k_inverse *
                       h3d_value(k_weight[pair]);
            float c =
                h3d_value(rope_cos[(size_t)row * rope_half + rope_index]);
            float s =
                h3d_value(rope_sin[(size_t)row * rope_half + rope_index]);
            if (dimension < rope_half) {
                q0 = q0 * c - q1 * s;
                k0 = k0 * c - k1 * s;
            } else {
                q0 = q0 * c + q1 * s;
                k0 = k0 * c + k1 * s;
            }
        }
        size_t output_index =
            ((size_t)head * sequence + row) * head_dim + dimension;
        query[output_index] = h3d_store(q0);
        key[output_index] = h3d_store(k0);
    }
}

int h3d_qkv_rope(h3d_context *context, h3d_tensor *query, h3d_tensor *key,
                 h3d_tensor *value, const h3d_tensor *qkv,
                 const h3d_tensor *q_norm, const h3d_tensor *k_norm,
                 const h3d_tensor *rope_cos, const h3d_tensor *rope_sin,
                 uint32_t sequence, uint32_t heads, uint32_t head_dim,
                 uint32_t rope_half, float epsilon, char *error,
                 size_t error_size) {
    size_t elements = 0;
    if (!context || !sequence || !heads || !head_dim ||
        !h3d_mul_size((size_t)sequence * heads, head_dim, &elements) ||
        rope_half > head_dim / 2 || epsilon < 0.0f ||
        !h3d_is(query, H3D_BF16, elements) ||
        !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(value, H3D_BF16, elements) ||
        !h3d_is(qkv, H3D_BF16, elements * 3) ||
        !h3d_is(q_norm, H3D_BF16, head_dim) ||
        !h3d_is(k_norm, H3D_BF16, head_dim) ||
        !h3d_is(rope_cos, H3D_BF16, (size_t)sequence * rope_half) ||
        !h3d_is(rope_sin, H3D_BF16, (size_t)sequence * rope_half)) {
        return h3d_set_error(error, error_size, "invalid QKV/RoPE tensors");
    }
    if (h3d_elementwise_reference() || head_dim > H3D_QKV_STAGED_DIM) {
        dim3 grid(sequence, heads, (head_dim + 127) / 128);
        h3d_qkv_rope_reference_kernel<<<grid, 128, 0, context->stream>>>(
            (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
            (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
            (const __nv_bfloat16 *)q_norm->data,
            (const __nv_bfloat16 *)k_norm->data,
            (const __nv_bfloat16 *)rope_cos->data,
            (const __nv_bfloat16 *)rope_sin->data, sequence, heads, head_dim,
            rope_half, epsilon);
    } else {
        /* One block owns a whole head: it stages every dimension, so the
         * dimension axis is not a grid axis here. */
        dim3 grid(sequence, heads);
        h3d_qkv_rope_kernel<<<grid, 128, 0, context->stream>>>(
            (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
            (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
            (const __nv_bfloat16 *)q_norm->data,
            (const __nv_bfloat16 *)k_norm->data,
            (const __nv_bfloat16 *)rope_cos->data,
            (const __nv_bfloat16 *)rope_sin->data, sequence, heads, head_dim,
            rope_half, epsilon);
    }
    return h3d_launch_ok("BF16 grouped QKV RoPE", error, error_size);
}

/* Rows one warp owns. 1 is the reference's mapping (one warp per row); more
 * rows amortize the staged key rows and the barrier pair over them. The value
 * is per translation unit so a sweep can build the reproducer with
 * -DH3D_ATTENTION_ROWS=n and compare output hashes. */
#ifndef H3D_ATTENTION_ROWS
#define H3D_ATTENTION_ROWS 2
#endif

/* `h3_attention_tiled_bf16_kernel` (`:1418-1487`): one block per head, four
 * dimensions per lane, online softmax with a warp shuffle reduction, row-major
 * output (`head_major_output = 0`).
 *
 * Deviations from the reference, named and measured: the reference stages one
 * key row per barrier pair, so each row's load latency is exposed once per row
 * (measured on this host at ~1.3 us per row, 90% of a denoise step); this
 * stages KEYS rows per barrier pair, so one exposed latency covers KEYS rows.
 * The reference gives each warp one query row; this gives it H3D_ATTENTION_ROWS
 * rows, so the staged rows and the barrier pair serve that many rows. Both are
 * measured. Every row still runs the reference's arithmetic in the reference's
 * order, so the output stays bit-identical.
 *
 * `HEAD_DIM` is a template parameter because the AV connector's attention is
 * 64 wide (its audio half) where the video VAE's is 128; the body is symbolic
 * in it and the `128` instantiation is the code that was here before. */
template <uint32_t HEAD_DIM>
__global__ static void h3d_attention_tiled_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t sequence, uint32_t heads, float scale) {
    enum { PARTS = HEAD_DIM / 32, KEYS = 8, ROWS = H3D_ATTENTION_ROWS,
           QUERIES = 8 * ROWS };
    __shared__ float shared_key[KEYS][HEAD_DIM];
    __shared__ float shared_value[KEYS][HEAD_DIM];
    uint32_t warp = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    uint32_t head = blockIdx.y;
    uint32_t first_row = blockIdx.x * QUERIES + warp * ROWS;
    int active[ROWS];
    float result[ROWS][PARTS];
    float maximum[ROWS];
    float denominator[ROWS];
    /* Each row's four dimensions ride in registers: the reference re-reads them
     * from global for every key row, which exposes that load once per row. */
    float query_item[ROWS][PARTS];
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        uint32_t row = first_row + r;
        active[r] = row < sequence;
        maximum[r] = -INFINITY;
        denominator[r] = 0.0f;
#pragma unroll
        for (uint32_t part = 0; part < PARTS; part++) {
            result[r][part] = 0.0f;
            query_item[r][part] = 0.0f;
        }
        if (!active[r]) {
            continue;
        }
        size_t query_base = ((size_t)head * sequence + row) * HEAD_DIM;
#pragma unroll
        for (uint32_t part = 0; part < PARTS; part++) {
            query_item[r][part] =
                    __bfloat162float(query[query_base + lane + part * 32]);
        }
    }
    for (uint32_t key_block = 0; key_block < sequence; key_block += KEYS) {
        /* KEYS independent loads per thread: the staged rows' latencies
         * overlap instead of one row's latency stalling every iteration. */
        if (threadIdx.x < HEAD_DIM) {
#pragma unroll
            for (uint32_t item = 0; item < KEYS; item++) {
                uint32_t source = key_block + item;
                if (source < sequence) {
                    size_t index =
                            ((size_t)head * sequence + source) * HEAD_DIM +
                            threadIdx.x;
                    shared_key[item][threadIdx.x] =
                            __bfloat162float(key[index]);
                    shared_value[item][threadIdx.x] =
                            __bfloat162float(value[index]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t item = 0; item < KEYS; item++) {
            if (key_block + item >= sequence) {
                break;
            }
#pragma unroll
            for (uint32_t r = 0; r < ROWS; r++) {
                float product = 0.0f;
                if (active[r]) {
#pragma unroll
                    for (uint32_t part = 0; part < PARTS; part++) {
                        uint32_t dimension = lane + part * 32;
                        product += query_item[r][part] *
                                   shared_key[item][dimension];
                    }
#pragma unroll
                    for (uint32_t offset = 16; offset; offset >>= 1) {
                        product += __shfl_down_sync(0xffffffffu, product, offset);
                    }
                }
                float rescale = 1.0f;
                float weight = 0.0f;
                if (active[r] && lane == 0) {
                    float score = product * scale;
                    float next_maximum = fmaxf(maximum[r], score);
                    rescale = expf(maximum[r] - next_maximum);
                    weight = expf(score - next_maximum);
                    denominator[r] = denominator[r] * rescale + weight;
                    maximum[r] = next_maximum;
                }
                rescale = __shfl_sync(0xffffffffu, rescale, 0);
                weight = __shfl_sync(0xffffffffu, weight, 0);
                if (active[r]) {
#pragma unroll
                    for (uint32_t part = 0; part < PARTS; part++) {
                        uint32_t dimension = lane + part * 32;
                        result[r][part] = result[r][part] * rescale +
                                          weight *
                                          shared_value[item][dimension];
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        denominator[r] = __shfl_sync(0xffffffffu, denominator[r], 0);
        if (!active[r]) {
            continue;
        }
        uint32_t row = first_row + r;
#pragma unroll
        for (uint32_t item = 0; item < PARTS; item++) {
            uint32_t dimension = lane + item * 32;
            size_t index = ((size_t)row * heads + head) * HEAD_DIM + dimension;
            output[index] = __float2bfloat16(result[r][item] / denominator[r]);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Tensor-core attention — a benchmark arm, OFF BY DEFAULT.
 *
 * `H3_DIT_ATTENTION_MMA` selects this twin of `h3d_attention_tiled_kernel`. It
 * exists to measure what the scalar path cannot reach: the operator asked for
 * it as a switch that can be turned on and off (2026-10-27), so it is a
 * measurement arm and never the path a gate reads.
 *
 * It is NOT bit-identical to the reference, by construction, and the three
 * deviations are named rather than hidden:
 *   - the scores accumulate through `mma` dot products, so the reduction tree
 *     replaces the reference's four-element lane partial and five-step shuffle;
 *   - the online softmax takes one maximum per 64-key tile instead of rescaling
 *     on every key row;
 *   - P rounds to BF16 for the second MMA; the reference keeps it in FP32. This
 *     is the same named deviation the ds41 kernel carries
 *     (`cuda/ds41_attn_mma.cuh:135-139`).
 * Scores, the running maximum/sum and the output accumulator stay FP32, so the
 * arithmetic is the reference's in kind, not in order.
 *
 * Shape: one CTA covers H3D_MMA_ROWS query rows of one head and a warp owns
 * H3D_MMA_M of them with all H3D_MMA_HD key dimensions, so a row's softmax
 * statistics never cross a warp and no partial-score exchange is needed (the
 * `spart` stage ds41 needs for its K-split does not apply here).
 * ------------------------------------------------------------------------- */

/* `mma.sync.m16n8k16` BF16 in, FP32 accumulate; mirrors `cuda/ds41_attn_mma.cuh:151-155`. */
__device__ __forceinline__ static void h3d_mma_step(float *c, const uint32_t *a,
                                                    uint32_t b0, uint32_t b1) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* 16x16 BF16 tile from shared memory, row-major, and its transpose; mirrors
 * `cuda/ds41_attn_mma.cuh:144-150` (`ds4_fa_ldsm4`/`ds4_fa_ldsm4t`). */
__device__ __forceinline__ static void h3d_mma_load(uint32_t *r, const void *p) {
    const uint32_t at = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at));
}

__device__ __forceinline__ static void h3d_mma_load_trans(uint32_t *r,
                                                          const void *p) {
    const uint32_t at = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at));
}

/* Two values as one BF16x2 register: the low half is column c and the high half
 * c+1, which is the fragment element order. Mirrors
 * `cuda/ds41_attn_mma.cuh:155-157` (`ds4_fa_bf16x2`). A value already BF16
 * survives the round trip exactly, so this serves the Q fragments too. */
__device__ __forceinline__ static uint32_t h3d_mma_pair(float low, float high) {
    const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16(low));
    const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16(high));
    return lo | (hi << 16);
}

/* `PytorchAttention.__call__` (`attention.py:82-102`):
 * `q.view(b, -1, heads, dim_head).transpose(1, 2)`, SDPA at
 * `1 / sqrt(dim_head)`, then `.transpose(1, 2).reshape(b, -1, heads * dim_head)`.
 * The operands are therefore token-major `(rows, heads, head_dim)`, which is
 * what every LTX caller hands in (`to_q`/`to_k`/`to_v` are plain
 * `torch.nn.Linear`s) and the one thing `h3d_sdpa` does not accept: that
 * kernel is the H3 engine's, whose `h3d_qkv_rope` writes query/key/value
 * head-major `[heads][sequence][head_dim]`. The head count is shared; only the
 * two row counts differ, and the AV cross-attention needs them apart (16,128
 * video queries against 126 audio keys). The arithmetic is
 * `h3d_attention_tiled_kernel`'s, online softmax included; only the two
 * operand addresses and the key-loop bound change.
 *
 * `HEAD_DIM` is a template parameter for the same reason it is there: 128 for
 * the video half, 64 for the audio half and their cross-attentions. */
template <uint32_t HEAD_DIM>
__global__ static void h3d_ltx_attention_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t q_rows, uint32_t kv_rows, uint32_t heads, float scale) {
    enum { PARTS = HEAD_DIM / 32, KEYS = 8, ROWS = H3D_ATTENTION_ROWS,
           QUERIES = 8 * ROWS };
    __shared__ float shared_key[KEYS][HEAD_DIM];
    __shared__ float shared_value[KEYS][HEAD_DIM];
    uint32_t warp = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    uint32_t head = blockIdx.y;
    uint32_t first_row = blockIdx.x * QUERIES + warp * ROWS;
    int active[ROWS];
    float result[ROWS][PARTS];
    float maximum[ROWS];
    float denominator[ROWS];
    float query_item[ROWS][PARTS];
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        uint32_t row = first_row + r;
        active[r] = row < q_rows;
        maximum[r] = -INFINITY;
        denominator[r] = 0.0f;
#pragma unroll
        for (uint32_t part = 0; part < PARTS; part++) {
            result[r][part] = 0.0f;
            query_item[r][part] = 0.0f;
        }
        if (!active[r]) {
            continue;
        }
        size_t query_base = ((size_t)row * heads + head) * HEAD_DIM;
#pragma unroll
        for (uint32_t part = 0; part < PARTS; part++) {
            query_item[r][part] =
                    __bfloat162float(query[query_base + lane + part * 32]);
        }
    }
    for (uint32_t key_block = 0; key_block < kv_rows; key_block += KEYS) {
        if (threadIdx.x < HEAD_DIM) {
#pragma unroll
            for (uint32_t item = 0; item < KEYS; item++) {
                uint32_t source = key_block + item;
                if (source < kv_rows) {
                    size_t index =
                            ((size_t)source * heads + head) * HEAD_DIM +
                            threadIdx.x;
                    shared_key[item][threadIdx.x] =
                            __bfloat162float(key[index]);
                    shared_value[item][threadIdx.x] =
                            __bfloat162float(value[index]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t item = 0; item < KEYS; item++) {
            if (key_block + item >= kv_rows) {
                break;
            }
#pragma unroll
            for (uint32_t r = 0; r < ROWS; r++) {
                float product = 0.0f;
                if (active[r]) {
#pragma unroll
                    for (uint32_t part = 0; part < PARTS; part++) {
                        uint32_t dimension = lane + part * 32;
                        product += query_item[r][part] *
                                   shared_key[item][dimension];
                    }
#pragma unroll
                    for (uint32_t offset = 16; offset; offset >>= 1) {
                        product += __shfl_down_sync(0xffffffffu, product, offset);
                    }
                }
                float rescale = 1.0f;
                float weight = 0.0f;
                if (active[r] && lane == 0) {
                    float score = product * scale;
                    float next_maximum = fmaxf(maximum[r], score);
                    rescale = expf(maximum[r] - next_maximum);
                    weight = expf(score - next_maximum);
                    denominator[r] = denominator[r] * rescale + weight;
                    maximum[r] = next_maximum;
                }
                rescale = __shfl_sync(0xffffffffu, rescale, 0);
                weight = __shfl_sync(0xffffffffu, weight, 0);
                if (active[r]) {
#pragma unroll
                    for (uint32_t part = 0; part < PARTS; part++) {
                        uint32_t dimension = lane + part * 32;
                        result[r][part] = result[r][part] * rescale +
                                          weight *
                                          shared_value[item][dimension];
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        denominator[r] = __shfl_sync(0xffffffffu, denominator[r], 0);
        if (!active[r]) {
            continue;
        }
        uint32_t row = first_row + r;
#pragma unroll
        for (uint32_t item = 0; item < PARTS; item++) {
            uint32_t dimension = lane + item * 32;
            size_t index = ((size_t)row * heads + head) * HEAD_DIM + dimension;
            output[index] = __float2bfloat16(result[r][item] / denominator[r]);
        }
    }
}

int h3d_ltx_attention(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *query, const h3d_tensor *key,
                      const h3d_tensor *value, uint32_t q_rows,
                      uint32_t kv_rows, uint32_t heads, uint32_t head_dim,
                      float scale, char *error, size_t error_size) {
    size_t query_elements = 0, key_elements = 0;
    if (!context || !q_rows || !kv_rows || !heads ||
        (head_dim != 128 && head_dim != 64) ||
        !h3d_mul_size((size_t)q_rows * heads, head_dim, &query_elements) ||
        !h3d_mul_size((size_t)kv_rows * heads, head_dim, &key_elements) ||
        !h3d_is(output, H3D_BF16, query_elements) ||
        !h3d_is(query, H3D_BF16, query_elements) ||
        !h3d_is(key, H3D_BF16, key_elements) ||
        !h3d_is(value, H3D_BF16, key_elements)) {
        return h3d_set_error(error, error_size,
                             "invalid LTX attention tensors or shape");
    }
    dim3 grid((q_rows + 8 * H3D_ATTENTION_ROWS - 1) /
                      (8 * H3D_ATTENTION_ROWS),
              heads);
    if (head_dim == 64) {
        h3d_ltx_attention_kernel<64><<<grid, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
            (const __nv_bfloat16 *)key->data,
            (const __nv_bfloat16 *)value->data, q_rows, kv_rows, heads, scale);
        return h3d_launch_ok("LTX attention 64", error, error_size);
    }
    h3d_ltx_attention_kernel<128><<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
        (const __nv_bfloat16 *)key->data, (const __nv_bfloat16 *)value->data,
        q_rows, kv_rows, heads, scale);
    return h3d_launch_ok("LTX attention", error, error_size);
}

/* ---------------------------------------------------------------------------
 * The LTX-2.5 refiner block's elementwise set (`model/transformer/`, the pin
 * at `/data/ltx2-pin/packages`).
 *
 * Torch computes every BF16 elementwise op in FP32 and rounds once to BF16
 * (`opmath_type<BFloat16>` is `float`), and each PyTorch expression below is a
 * chain of separate ops, so the kernels here round where the expression does.
 * `rms_norm(x) * (1 + scale) + shift` is four ops and four roundings: the
 * normalization's own, the `1 + scale` tensor, the product, and the sum. A
 * fused kernel that rounds once is NOT the same function.
 * ------------------------------------------------------------------------- */

/* `PytorchPreAttention.__call__` (`ops.py:22-36`) applies `apply_rotary_emb`
 * to the query and to the key with *different* tables when `k_pe` is given
 * (`transformer.py:359-361`, the A2V and V2A cross-attention). The two sides
 * also have different row counts, so this is `h3d_ltx_rope` with the tables
 * and the row counts split. */
__global__ static void h3d_ltx_rope_pair_kernel(
        __nv_bfloat16 *query, __nv_bfloat16 *key, const __nv_bfloat16 *q_cos,
        const __nv_bfloat16 *q_sin, const __nv_bfloat16 *k_cos,
        const __nv_bfloat16 *k_sin, uint32_t q_rows, uint32_t k_rows,
        uint32_t heads, uint32_t head_dim) {
    const uint32_t row = blockIdx.x, head = blockIdx.y;
    const uint32_t half = head_dim / 2;

    if (row < q_rows) {
        const size_t base = ((size_t)row * heads + head) * head_dim;
        const size_t table = ((size_t)row * heads + head) * half;
        for (uint32_t d = threadIdx.x; d < half; d += blockDim.x) {
            h3d_ltx_rotate(query, base, half, d,
                           __bfloat162float(q_cos[table + d]),
                           __bfloat162float(q_sin[table + d]));
        }
    }

    if (row < k_rows) {
        const size_t base = ((size_t)row * heads + head) * head_dim;
        const size_t table = ((size_t)row * heads + head) * half;
        for (uint32_t d = threadIdx.x; d < half; d += blockDim.x) {
            h3d_ltx_rotate(key, base, half, d,
                           __bfloat162float(k_cos[table + d]),
                           __bfloat162float(k_sin[table + d]));
        }
    }
}

int h3d_ltx_rope_pair(h3d_context *context, h3d_tensor *query,
                      h3d_tensor *key, const h3d_tensor *q_cos,
                      const h3d_tensor *q_sin, const h3d_tensor *k_cos,
                      const h3d_tensor *k_sin, uint32_t q_rows, uint32_t k_rows,
                      uint32_t heads, uint32_t head_dim, char *error,
                      size_t error_size) {
    size_t query_elements = 0, key_elements = 0, q_table = 0, k_table = 0;
    if (!context || !q_rows || !k_rows || !heads || !head_dim ||
        (head_dim % 2) ||
        !h3d_mul_size((size_t)q_rows * heads, head_dim, &query_elements) ||
        !h3d_mul_size((size_t)k_rows * heads, head_dim, &key_elements) ||
        !h3d_mul_size((size_t)q_rows * heads, head_dim / 2, &q_table) ||
        !h3d_mul_size((size_t)k_rows * heads, head_dim / 2, &k_table) ||
        !h3d_is(query, H3D_BF16, query_elements) ||
        !h3d_is(key, H3D_BF16, key_elements) ||
        !h3d_is(q_cos, H3D_BF16, q_table) || !h3d_is(q_sin, H3D_BF16, q_table) ||
        !h3d_is(k_cos, H3D_BF16, k_table) || !h3d_is(k_sin, H3D_BF16, k_table)) {
        return h3d_set_error(error, error_size, "invalid LTX RoPE pair tensors");
    }
    dim3 grid(q_rows > k_rows ? q_rows : k_rows, heads);
    h3d_ltx_rope_pair_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (const __nv_bfloat16 *)q_cos->data, (const __nv_bfloat16 *)q_sin->data,
        (const __nv_bfloat16 *)k_cos->data, (const __nv_bfloat16 *)k_sin->data,
        q_rows, k_rows, heads, head_dim);
    return h3d_launch_ok("LTX RoPE pair", error, error_size);
}

/* `BasicAVTransformerBlock.get_ada_values` (`transformer.py:190-200`): the
 * block table's row plus the AdaLN MLP's matching coefficient row, one BF16
 * addition. `modulation` is `[rows, coefficient, width]` - the shape
 * `timestep.reshape(batch, T, num_ada_params, -1)` produces - so one row of
 * it is strided and cannot be handed to `h3d_add`. */
__global__ static void h3d_ada_add_kernel(__nv_bfloat16 *output,
                                          const __nv_bfloat16 *table,
                                          const __nv_bfloat16 *modulation,
                                          uint32_t rows, uint32_t width,
                                          uint32_t coefficient,
                                          uint32_t table_row,
                                          uint32_t value_row) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) return;
    const uint32_t row = (uint32_t)(index / width);
    const uint32_t column = (uint32_t)(index % width);
    const size_t parameter = ((size_t)row * coefficient + value_row) * width +
                             column;
    output[index] = __float2bfloat16(
        __bfloat162float(table[(size_t)table_row * width + column]) +
        __bfloat162float(modulation[parameter]));
}

int h3d_ada_add(h3d_context *context, h3d_tensor *output,
                const h3d_tensor *table, const h3d_tensor *modulation,
                uint32_t rows, uint32_t width, uint32_t coefficient,
                uint32_t table_row, uint32_t value_row, char *error,
                size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width || !coefficient ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(modulation, H3D_BF16, (size_t)rows * coefficient * width) ||
        value_row >= coefficient) {
        return h3d_set_error(error, error_size, "invalid AdaLN add arguments");
    }
    h3d_ada_add_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                         context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)table->data,
        (const __nv_bfloat16 *)modulation->data, rows, width, coefficient,
        table_row, value_row);
    return h3d_launch_ok("LTX AdaLN add", error, error_size);
}

/* `apply_cross_attention_adaln` (`transformer.py:446`) and
 * `_process_output` (`model.py:487-488`): `x * (1 + scale) + shift`, with the
 * affine row broadcast when `scale_rows` is 1 (the prompt table and the model
 * level output table are per request) and per token otherwise. */
__global__ static void h3d_modulate_kernel(__nv_bfloat16 *output,
                                           const __nv_bfloat16 *input,
                                           const __nv_bfloat16 *scale,
                                           const __nv_bfloat16 *shift,
                                           uint32_t rows, uint32_t width,
                                           uint32_t scale_rows) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) return;
    const uint32_t row = (uint32_t)(index / width);
    const uint32_t column = (uint32_t)(index % width);
    const size_t parameter = (size_t)(scale_rows == 1 ? 0 : row) * width +
                             column;
    const float factor = __bfloat162float(
        __float2bfloat16(1.0f + __bfloat162float(scale[parameter])));
    const float product = __bfloat162float(
        __float2bfloat16(__bfloat162float(input[index]) * factor));
    output[index] =
        __float2bfloat16(product + __bfloat162float(shift[parameter]));
}

int h3d_modulate(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, const h3d_tensor *scale,
                 const h3d_tensor *shift, uint32_t rows, uint32_t width,
                 uint32_t scale_rows, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        (scale_rows != 1 && scale_rows != rows) ||
        !h3d_is(scale, H3D_BF16, (size_t)scale_rows * width) ||
        !h3d_is(shift, H3D_BF16, (size_t)scale_rows * width)) {
        return h3d_set_error(error, error_size, "invalid BF16 modulate arguments");
    }
    h3d_modulate_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                          context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)scale->data, (const __nv_bfloat16 *)shift->data,
        rows, width, scale_rows);
    return h3d_launch_ok("BF16 modulate", error, error_size);
}

/* `transformer.py:362`: the A2V/V2A output is multiplied by a per-request
 * gate `(B, 1, D)` - one row broadcast over every token - where `h3d_mul`
 * needs equal element counts. */
__global__ static void h3d_mul_rows_kernel(__nv_bfloat16 *output,
                                           const __nv_bfloat16 *left,
                                           const __nv_bfloat16 *right,
                                           uint32_t rows, uint32_t width,
                                           uint32_t right_rows) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) return;
    const uint32_t row = (uint32_t)(index / width);
    const size_t factor =
        (size_t)(right_rows == 1 ? 0 : row) * width + (index % width);
    output[index] = __float2bfloat16(__bfloat162float(left[index]) *
                                     __bfloat162float(right[factor]));
}

int h3d_mul_rows(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *left, const h3d_tensor *right, uint32_t rows,
                 uint32_t width, uint32_t right_rows, char *error,
                 size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(left, H3D_BF16, elements) ||
        (right_rows != 1 && right_rows != rows) ||
        !h3d_is(right, H3D_BF16, (size_t)right_rows * width)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 row-broadcast multiply arguments");
    }
    h3d_mul_rows_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                          context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)left->data,
        (const __nv_bfloat16 *)right->data, rows, width, right_rows);
    return h3d_launch_ok("BF16 row broadcast multiply", error, error_size);
}

/* `PytorchAdaZeroFunction` (`ops.py:50-58`): `rms_norm(x) * (1 + scale) +
 * shift`. The reduction and `rsqrtf(sum / width + eps)` are
 * `h3d_rms_norm_kernel`'s own (`:2707`); the epilogue is
 * `h3d_modulate_kernel`'s, applied to the normalized value. */
__global__ static void h3d_ada_zero_kernel(__nv_bfloat16 *output,
                                           const __nv_bfloat16 *input,
                                           const __nv_bfloat16 *scale,
                                           const __nv_bfloat16 *shift,
                                           uint32_t rows, uint32_t width,
                                           uint32_t scale_rows,
                                           float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = __bfloat162float(input[(size_t)row * width + column]);
        sum = fmaf(value, value, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    const uint32_t source = scale_rows == 1 ? 0 : row;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        size_t parameter = (size_t)source * width + column;
        float normalized = __bfloat162float(__float2bfloat16(
            __bfloat162float(input[index]) * inverse));
        float factor = __bfloat162float(
            __float2bfloat16(1.0f + __bfloat162float(scale[parameter])));
        output[index] = __float2bfloat16(
            __bfloat162float(__float2bfloat16(normalized * factor)) +
            __bfloat162float(shift[parameter]));
    }
}

int h3d_ada_zero(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, const h3d_tensor *scale,
                 const h3d_tensor *shift, uint32_t rows, uint32_t width,
                 uint32_t scale_rows, float epsilon, char *error,
                 size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width || epsilon < 0.0f ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        (scale_rows != 1 && scale_rows != rows) ||
        !h3d_is(scale, H3D_BF16, (size_t)scale_rows * width) ||
        !h3d_is(shift, H3D_BF16, (size_t)scale_rows * width)) {
        return h3d_set_error(error, error_size, "invalid AdaLN zero arguments");
    }
    h3d_ada_zero_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)scale->data, (const __nv_bfloat16 *)shift->data,
        rows, width, scale_rows, epsilon);
    return h3d_launch_ok("LTX AdaLN zero", error, error_size);
}

/* `torch.nn.LayerNorm(..., elementwise_affine=False, eps=1e-6)`
 * (`model.py:232`, `:471-489`, `:520-536`): the mean with plain adds, the
 * biased variance with `fmaf` on the centered values, `rsqrtf(var + eps)`,
 * then one rounding - `h3d_layer_norm_f32_kernel`'s arithmetic (`:4697`) at
 * BF16 without the affine pair. */
__global__ static void h3d_layer_norm_kernel(__nv_bfloat16 *output,
                                             const __nv_bfloat16 *input,
                                             uint32_t rows, uint32_t width,
                                             float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        sum += __bfloat162float(input[(size_t)row * width + column]);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = sums[0] / (float)width;
    sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float centered =
            __bfloat162float(input[(size_t)row * width + column]) - mean;
        sum = fmaf(centered, centered, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        output[index] = __float2bfloat16(
            (__bfloat162float(input[index]) - mean) * inverse);
    }
}

int h3d_layer_norm(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, uint32_t rows, uint32_t width,
                   float epsilon, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width || epsilon < 0.0f ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size, "invalid BF16 layer norm arguments");
    }
    h3d_layer_norm_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        rows, width, epsilon);
    return h3d_launch_ok("BF16 layer norm", error, error_size);
}

/* `VideoLatentPatchifier.patchify`/`unpatchify` at patch size 1
 * (`components/patchifiers.py:27-64`): `b c (f p1) (h p2) (w p3) -> b (f h w)
 * (c p1 p2 p3)` is, at `p = 1`, a plain `[channels, tokens]` transpose, and
 * the inverse is the same transpose. `rows` is the token count and `width`
 * the channel count; the output is `[width, rows]`. */
#define H3D_TRANSPOSE_TILE 32
__global__ static void h3d_channel_transpose_kernel(__nv_bfloat16 *output,
                                                    const __nv_bfloat16 *input,
                                                    uint32_t rows,
                                                    uint32_t width) {
    __shared__ __nv_bfloat16 tile[H3D_TRANSPOSE_TILE][H3D_TRANSPOSE_TILE + 1];
    const uint32_t column = blockIdx.x * H3D_TRANSPOSE_TILE + threadIdx.x;
    const uint32_t row = blockIdx.y * H3D_TRANSPOSE_TILE + threadIdx.y;

    if (column < width && row < rows) {
        tile[threadIdx.y][threadIdx.x] = input[(size_t)row * width + column];
    }
    __syncthreads();

    // The tile is read transposed, so threadIdx.x now walks the input's token
    // axis and threadIdx.y its channel axis.
    const uint32_t out_row = blockIdx.x * H3D_TRANSPOSE_TILE + threadIdx.y;
    const uint32_t out_column = blockIdx.y * H3D_TRANSPOSE_TILE + threadIdx.x;

    if (out_column < rows && out_row < width) {
        output[(size_t)out_row * rows + out_column] =
            tile[threadIdx.x][threadIdx.y];
    }
}

int h3d_channel_transpose(h3d_context *context, h3d_tensor *output,
                          const h3d_tensor *input, uint32_t rows,
                          uint32_t width, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 channel transpose arguments");
    }
    dim3 grid((width + H3D_TRANSPOSE_TILE - 1) / H3D_TRANSPOSE_TILE,
              (rows + H3D_TRANSPOSE_TILE - 1) / H3D_TRANSPOSE_TILE);
    dim3 block(H3D_TRANSPOSE_TILE, H3D_TRANSPOSE_TILE);
    h3d_channel_transpose_kernel<<<grid, block, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        rows, width);
    return h3d_launch_ok("BF16 channel transpose", error, error_size);
}

/* `AudioPatchifier.patchify` (`components/patchifiers.py:288-304`): one audio
 * latent `[batch, channels, frames, bins]` to `[batch, frames, channels*bins]`.
 * The audio grid keeps `bins` as an axis, so this is not the channel
 * transpose above: output `(t, c, f)` reads input `(c, t, f)`, the `bins` axis
 * innermost on both sides. */
__global__ static void h3d_audio_patchify_kernel(__nv_bfloat16 *output,
                                                 const __nv_bfloat16 *input,
                                                 uint32_t batch,
                                                 uint32_t channels,
                                                 uint32_t frames,
                                                 uint32_t bins) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t rows = (size_t)batch * frames;
    const size_t width = (size_t)channels * bins;
    if (index >= rows * width) return;
    const size_t row = index / width;
    const size_t column = index % width;
    const size_t frame = row % frames;
    const size_t item = row / frames;
    const size_t channel = column / bins;
    const size_t bin = column % bins;
    output[index] = input[((item * channels + channel) * frames + frame) * bins +
                          bin];
}

int h3d_audio_patchify(h3d_context *context, h3d_tensor *output,
                       const h3d_tensor *input, uint32_t batch,
                       uint32_t channels, uint32_t frames, uint32_t bins,
                       char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !batch || !channels || !frames || !bins ||
        !h3d_mul_size((size_t)batch * frames, (size_t)channels * bins,
                      &elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 audio patchify arguments");
    }
    h3d_audio_patchify_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                                context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        batch, channels, frames, bins);
    return h3d_launch_ok("BF16 audio patchify", error, error_size);
}

/* `apply_keyframes_absolute_embedding` (`transformer_args.py:22-43`):
 * `hidden_states + mask * embedding`, one `[1, width]` row broadcast over the
 * rows the `[rows, 1]` mask marks. The mask is `(keyframes_mask > 0).to(dtype)`,
 * exactly 0 or 1, so `mask * embedding` is exact and the whole expression is
 * one BF16 add: the kernel adds the row where the mask is set and 0 where it
 * is not. */
__global__ static void h3d_keyframe_add_kernel(__nv_bfloat16 *output,
                                               const __nv_bfloat16 *input,
                                               const __nv_bfloat16 *mask,
                                               const __nv_bfloat16 *embedding,
                                               uint32_t rows, uint32_t width) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) return;
    const uint32_t row = (uint32_t)(index / width);
    const uint32_t column = (uint32_t)(index % width);
    const float marked = __bfloat162float(mask[row]) != 0.0f
                             ? __bfloat162float(embedding[column])
                             : 0.0f;
    output[index] = __float2bfloat16(__bfloat162float(input[index]) + marked);
}

int h3d_keyframe_add(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *input, const h3d_tensor *mask,
                     const h3d_tensor *embedding, uint32_t rows, uint32_t width,
                     char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width ||
        !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(input, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(mask, H3D_BF16, rows) || !h3d_is(embedding, H3D_BF16, width)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 keyframe add arguments");
    }
    h3d_keyframe_add_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                              context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)mask->data,
        (const __nv_bfloat16 *)embedding->data, rows, width);
    return h3d_launch_ok("BF16 keyframe add", error, error_size);
}

/* torch's own CUDA normal draw, transcribed so the port gets the reference's
 * values rather than a lookalike sequence.
 *
 * `torch.randn(x.shape, generator=g, dtype=x.dtype)` reaches
 * `normal_kernel<BFloat16>` (`ATen/native/cuda/DistributionTemplates.h:461-476`)
 * with `accscalar_t = float`, which calls `normal_and_transform<BFloat16,
 * float>` (`:443-455`) - `curand_normal4` and `unroll_factor =
 * sizeof(float4)/sizeof(float) = 4` (`:116`). The launch is
 * `distribution_elementwise_grid_stride_kernel<accscalar_t, 4>` (`:66-89`): one
 * Philox state per thread seeded from the generator's `(seed, offset)` pair at
 * subsequence `idx`, a `float4` per iteration, component `ii` stored to element
 * `linear_index + blockDim.x * gridDim.x * ii`. `transformation::normal<float>(v,
 * 0, 1)` is `v * 1 + 0` (`ATen/core/TransformationHelper.h:96-99`) and the
 * store is `static_cast<BFloat16>(float)` = `__float2bfloat16`
 * (`c10/util/BFloat16-inl.h` -> `torch/headeronly/util/BFloat16.h:125-130`).
 *
 * The grid is `calc_execution_policy`'s (`:50-62`): 256-thread blocks capped at
 * `multiProcessorCount * (maxThreadsPerMultiProcessor / 256)`, and the host then
 * advances the generator by `((elements-1)/(256*grid*4)+1)*4`. That increment
 * is what `h3d_randn_bf16` hands back, so the caller keeps torch's
 * `philox_offset_per_thread_` bookkeeping (`CUDAGeneratorImpl.cpp`,
 * `philox_cuda_state`).
 *
 * `curand_normal4` is `curand_box_muller4` (`curand_normal.h:165-179`) over
 * `curand4`: `sqrtf`, `logf` and `__sincosf` run on the device's own units.
 * That is why these values are reproducible only on the architecture the
 * reference ran on - a host reimplementation of Box-Muller with libm would
 * differ in the last bits of every value. */
__global__ static void __launch_bounds__(256, 4)
    h3d_randn_bf16_kernel(__nv_bfloat16 *output, uint64_t elements,
                          unsigned long long seed, unsigned long long offset) {
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    const uint64_t rounded = ((elements - 1) / (stride * 4) + 1) * stride * 4;
    const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;

    curand_init(seed, (unsigned long long)index, offset, &state);
    for (uint64_t linear = (uint64_t)index; linear < rounded;
         linear += stride * 4) {
        const float4 rand = curand_normal4(&state);

#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const uint64_t at = linear + stride * (uint64_t)ii;
            if (at < elements) {
                output[at] = __float2bfloat16((&rand.x)[ii]);
            }
        }
        __syncthreads();
    }
}

int h3d_randn_bf16(h3d_context *context, h3d_tensor *output, size_t elements,
                   uint64_t seed, uint64_t offset, uint64_t *increment,
                   char *error, size_t error_size) {
    int device = 0;
    int threads_per_sm = 0;
    int multiprocessors = 0;
    uint64_t grid = 0;
    uint64_t whole = 0;

    if (!context || !elements || !increment ||
        !h3d_is(output, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size, "invalid BF16 randn arguments");
    }
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&threads_per_sm,
                               cudaDevAttrMaxThreadsPerMultiProcessor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess) {
        return h3d_set_error(error, error_size,
                             "randn: device attributes unavailable");
    }

    /* `calc_execution_policy` (`DistributionTemplates.h:50-62`). */
    whole = (elements + 255) / 256;
    grid = (uint64_t)multiprocessors * ((uint32_t)threads_per_sm / 256);
    if (grid > whole) grid = whole;
    if (!grid) grid = 1;
    *increment = ((elements - 1) / (256 * grid * 4) + 1) * 4;

    h3d_randn_bf16_kernel<<<(unsigned)grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (uint64_t)elements, seed, offset);
    return h3d_launch_ok("BF16 randn", error, error_size);
}

/* `torch.lerp` (`aten/src/ATen/native/Lerp.h:22-33`) is not `a + w*(b-a)`
 * unconditionally: `|w| < 0.5` takes `self + w*(end - self)` and any other
 * weight takes `end - (end - self)*(1 - w)`. Both branches are in float32 with
 * an `opmath_t` weight, and the branch is part of the value. */
__device__ static float h3d_lerp(float self, float end, float weight) {
    return fabsf(weight) < 0.5f ? self + weight * (end - self)
                                : end - (end - self) * (1.0f - weight);
}

/* `GaussianNoiser.__call__` (`components/noisers.py:30-37`): two float32
 * `torch.lerp`s - the latent toward the noise at `noise_scale`, then the clean
 * latent toward that at the patched denoise mask - and one cast back to the
 * latent's dtype. `noise_scale` is `STAGE2_SIGMAS[0]` = 0.909375
 * (`refiner_encoder_ablation_single_gpu.py:699`), and the mask is one value per
 * token, so `mask` is indexed by row. */
__global__ static void h3d_noise_lerp_kernel(__nv_bfloat16 *output,
                                             const __nv_bfloat16 *latent,
                                             const __nv_bfloat16 *noise,
                                             const __nv_bfloat16 *clean,
                                             const float *mask, uint32_t rows,
                                             uint32_t width, float weight) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) return;
    const float noisy =
        h3d_lerp(__bfloat162float(latent[index]), __bfloat162float(noise[index]),
                 weight);
    output[index] = __float2bfloat16(
        h3d_lerp(__bfloat162float(clean[index]), noisy, mask[index / width]));
}

int h3d_noise_lerp(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *latent, const h3d_tensor *noise,
                   const h3d_tensor *clean, const h3d_tensor *mask,
                   uint32_t rows, uint32_t width, float weight, char *error,
                   size_t error_size) {
    size_t elements = 0;

    if (!context || !rows || !width || !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(latent, H3D_BF16, elements) ||
        !h3d_is(noise, H3D_BF16, elements) ||
        !h3d_is(clean, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(mask, H3D_F32, rows)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 noise lerp arguments");
    }
    h3d_noise_lerp_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                            context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)latent->data,
        (const __nv_bfloat16 *)noise->data, (const __nv_bfloat16 *)clean->data,
        (const float *)mask->data, rows, width, weight);
    return h3d_launch_ok("BF16 noise lerp", error, error_size);
}

/* `timesteps_from_mask` (`ltx_pipelines/utils/helpers.py:494-501`):
 * `denoise_mask * sigma`, the per-token timestep the block AdaLN
 * (`transformer_args.py:176-185`) and `X0Model`'s `to_denoised` read. */
__global__ static void h3d_timesteps_kernel(float *output, const float *mask,
                                            uint32_t rows, float sigma) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    output[row] = mask[row] * sigma;
}

int h3d_timesteps(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *mask, uint32_t rows, float sigma,
                  char *error, size_t error_size) {
    if (!context || !rows || !h3d_is(output, H3D_F32, rows) ||
        !h3d_is(mask, H3D_F32, rows)) {
        return h3d_set_error(error, error_size, "invalid F32 timesteps arguments");
    }
    h3d_timesteps_kernel<<<(rows + 255) / 256, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)mask->data, rows, sigma);
    return h3d_launch_ok("F32 timesteps", error, error_size);
}

/* `to_denoised` (`ltx-core/src/ltx_core/utils.py:39-52`):
 * `(sample - velocity * sigma).to(sample.dtype)` in float32. `sigma` is the
 * per-token timestep `X0Model.forward` passes (`model.py:590-605`), not the
 * scalar the block AdaLN takes. */
__global__ static void h3d_to_denoised_kernel(__nv_bfloat16 *output,
                                              const __nv_bfloat16 *latent,
                                              const __nv_bfloat16 *velocity,
                                              const float *timestep,
                                              uint32_t rows, uint32_t width) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) return;
    output[index] = __float2bfloat16(
        __bfloat162float(latent[index]) -
        __bfloat162float(velocity[index]) * timestep[index / width]);
}

int h3d_to_denoised(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *latent, const h3d_tensor *velocity,
                    const h3d_tensor *timestep, uint32_t rows, uint32_t width,
                    char *error, size_t error_size) {
    size_t elements = 0;

    if (!context || !rows || !width || !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(latent, H3D_BF16, elements) ||
        !h3d_is(velocity, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(timestep, H3D_F32, rows)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 to-denoised arguments");
    }
    h3d_to_denoised_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                             context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)latent->data,
        (const __nv_bfloat16 *)velocity->data, (const float *)timestep->data,
        rows, width);
    return h3d_launch_ok("BF16 to-denoised", error, error_size);
}

/* `to_velocity` (`utils.py:21-35`): `((sample - denoised) / sigma).to(sample.dtype)`
 * in float32. `EulerDiffusionStep.step` calls it once per step and then rounds
 * again, so the two roundings are separate: this kernel writes the bf16
 * velocity, `h3d_euler_step` reads it back as float32
 * (`components/diffusion_steps.py:32-40`). */
__global__ static void h3d_to_velocity_kernel(__nv_bfloat16 *output,
                                              const __nv_bfloat16 *sample,
                                              const __nv_bfloat16 *denoised,
                                              float sigma, size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] =
        __float2bfloat16((__bfloat162float(sample[index]) -
                          __bfloat162float(denoised[index])) /
                         sigma);
}

int h3d_to_velocity(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *sample, const h3d_tensor *denoised,
                    float sigma, size_t elements, char *error,
                    size_t error_size) {
    if (!context || !elements || sigma == 0.0f ||
        !h3d_is(sample, H3D_BF16, elements) ||
        !h3d_is(denoised, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 to-velocity arguments");
    }
    h3d_to_velocity_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                             context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)sample->data,
        (const __nv_bfloat16 *)denoised->data, sigma, elements);
    return h3d_launch_ok("BF16 to-velocity", error, error_size);
}

/* `EulerDiffusionStep.step` (`diffusion_steps.py:32-40`):
 * `sample + velocity * (sigma_next - sigma)` in float32, then one cast back to
 * the sample's dtype. `delta` is `sigma_next - sigma` computed in float32 by
 * the caller, which is how the reference forms `dt`. */
__global__ static void h3d_euler_step_kernel(__nv_bfloat16 *output,
                                             const __nv_bfloat16 *sample,
                                             const __nv_bfloat16 *velocity,
                                             float delta, size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = __float2bfloat16(__bfloat162float(sample[index]) +
                                     __bfloat162float(velocity[index]) * delta);
}

int h3d_euler_step(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *sample, const h3d_tensor *velocity,
                   float delta, size_t elements, char *error,
                   size_t error_size) {
    if (!context || !elements || !h3d_is(sample, H3D_BF16, elements) ||
        !h3d_is(velocity, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 Euler step arguments");
    }
    h3d_euler_step_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                            context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)sample->data,
        (const __nv_bfloat16 *)velocity->data, delta, elements);
    return h3d_launch_ok("BF16 Euler step", error, error_size);
}

/* `post_process_latent` (`helpers.py:461-464`):
 * `(denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)`.
 * `bf16 * f32` promotes to f32, so both products are float32 and the mask is
 * read once per row. With the release's all-ones mask this is the identity,
 * which is why it must be transcribed rather than dropped. */
__global__ static void h3d_post_latent_kernel(__nv_bfloat16 *output,
                                              const __nv_bfloat16 *denoised,
                                              const float *mask,
                                              const __nv_bfloat16 *clean,
                                              uint32_t rows, uint32_t width) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) return;
    const float row = mask[index / width];
    output[index] = __float2bfloat16(__bfloat162float(denoised[index]) * row +
                                     __bfloat162float(clean[index]) * (1.0f - row));
}

int h3d_post_latent(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *denoised, const h3d_tensor *mask,
                    const h3d_tensor *clean, uint32_t rows, uint32_t width,
                    char *error, size_t error_size) {
    size_t elements = 0;

    if (!context || !rows || !width || !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(denoised, H3D_BF16, elements) ||
        !h3d_is(clean, H3D_BF16, elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(mask, H3D_F32, rows)) {
        return h3d_set_error(error, error_size,
                             "invalid BF16 post-process arguments");
    }
    h3d_post_latent_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                             context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)denoised->data,
        (const float *)mask->data, (const __nv_bfloat16 *)clean->data, rows,
        width);
    return h3d_launch_ok("BF16 post-process", error, error_size);
}

enum {
    H3D_MMA_HD = 128,                 /* head dimension, the one shape served */
    H3D_MMA_KEYS = 64,                /* key rows staged per tile */
    H3D_MMA_M = 16,                   /* query rows a warp owns */
    H3D_MMA_WARPS = 4,
    H3D_MMA_ROWS = H3D_MMA_M * H3D_MMA_WARPS,
    H3D_MMA_LD = H3D_MMA_HD + 8,      /* +8 bf16 spreads ldmatrix rows across banks */
    H3D_MMA_DIMS = H3D_MMA_HD / 8,    /* n8 slices of the output */
    H3D_MMA_KS = H3D_MMA_HD / 16,     /* k16 steps over the head dimension */
    H3D_MMA_NS = H3D_MMA_KEYS / 8,    /* n8 slices of the scores */
    H3D_MMA_CHUNKS = H3D_MMA_HD / 8   /* 16-byte copies in one staged row */
};

__global__ static void h3d_attention_mma_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t sequence, uint32_t heads, float scale) {
    __shared__ __align__(16) __nv_bfloat16 shared_key[H3D_MMA_KEYS][H3D_MMA_LD];
    __shared__ __align__(16) __nv_bfloat16 shared_value[H3D_MMA_KEYS][H3D_MMA_LD];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t quad = lane >> 2u;             /* the accumulator row a lane serves */
    const uint32_t column = (lane & 3u) * 2u;     /* first of its two columns */
    const uint32_t matrix = lane >> 3u;           /* ldmatrix: which 8x8 quadrant */
    const uint32_t head = blockIdx.y;
    const uint32_t first_row = blockIdx.x * H3D_MMA_ROWS + warp * H3D_MMA_M;
    const uint32_t row0 = first_row + quad, row1 = row0 + 8u;
    const int live0 = row0 < sequence, live1 = row1 < sequence;
    /* A row past the end is read from the last real row and never written; that
     * keeps the CTA's reads in bounds without branching the MMAs. */
    const uint32_t read0 = live0 ? row0 : sequence - 1u;
    const uint32_t read1 = live1 ? row1 : sequence - 1u;
    const __nv_bfloat16 *query0 =
            query + ((size_t)head * sequence + read0) * H3D_MMA_HD;
    const __nv_bfloat16 *query1 =
            query + ((size_t)head * sequence + read1) * H3D_MMA_HD;

    /* Q lives in registers as the eight k16 A fragments, built once per CTA. */
    uint32_t a[H3D_MMA_KS][4];
#pragma unroll
    for (uint32_t step = 0; step < H3D_MMA_KS; step++) {
        const uint32_t dim = step * 16u + column;
        a[step][0] = h3d_mma_pair(__bfloat162float(query0[dim]),
                                  __bfloat162float(query0[dim + 1u]));
        a[step][1] = h3d_mma_pair(__bfloat162float(query1[dim]),
                                  __bfloat162float(query1[dim + 1u]));
        a[step][2] = h3d_mma_pair(__bfloat162float(query0[dim + 8u]),
                                  __bfloat162float(query0[dim + 9u]));
        a[step][3] = h3d_mma_pair(__bfloat162float(query1[dim + 8u]),
                                  __bfloat162float(query1[dim + 9u]));
    }

    float maximum0 = -INFINITY, maximum1 = -INFINITY;
    float denominator0 = 0.0f, denominator1 = 0.0f;
    float result[H3D_MMA_DIMS][4];
#pragma unroll
    for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
        result[dim][0] = result[dim][1] = result[dim][2] = result[dim][3] = 0.0f;
    }

    for (uint32_t base = 0; base < sequence; base += H3D_MMA_KEYS) {
        __syncthreads();   /* the previous tile's staged rows are fully consumed */
        /* Stage K and V: 16-byte copies, one row per sixteen lanes. A row past
         * the end is zeroed; its scores are masked below, so it contributes
         * nothing and the staging stays branch-free. */
        for (uint32_t index = threadIdx.x;
             index < H3D_MMA_KEYS * H3D_MMA_CHUNKS;
             index += H3D_MMA_WARPS * 32u) {
            const uint32_t row = index / H3D_MMA_CHUNKS;
            const uint32_t chunk = index % H3D_MMA_CHUNKS;
            const uint32_t source = base + row;
            if (source < sequence) {
                const size_t at =
                        ((size_t)head * sequence + source) * H3D_MMA_HD +
                        chunk * 8u;
                *(uint4 *)&shared_key[row][chunk * 8u] =
                        *(const uint4 *)(key + at);
                *(uint4 *)&shared_value[row][chunk * 8u] =
                        *(const uint4 *)(value + at);
            } else {
                const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
                *(uint4 *)&shared_key[row][chunk * 8u] = zero;
                *(uint4 *)&shared_value[row][chunk * 8u] = zero;
            }
        }
        __syncthreads();

        /* Scores: S = Q * K^T over all H3D_MMA_KEYS staged keys. */
        float score[H3D_MMA_NS][4];
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_MMA_NS; slice++) {
            score[slice][0] = score[slice][1] = 0.0f;
            score[slice][2] = score[slice][3] = 0.0f;
        }
#pragma unroll
        for (uint32_t step = 0; step < H3D_MMA_KS; step++) {
#pragma unroll
            for (uint32_t part = 0; part < H3D_MMA_KEYS / 16; part++) {
                const uint32_t row = part * 16u + (matrix >> 1u) * 8u + (lane & 7u);
                const uint32_t col = step * 16u + (matrix & 1u) * 8u;
                uint32_t b[4];
                h3d_mma_load(b, &shared_key[row][col]);
                h3d_mma_step(score[2u * part], a[step], b[0], b[1]);
                h3d_mma_step(score[2u * part + 1u], a[step], b[2], b[3]);
            }
        }

        /* Scale, then mask the tail keys out of the softmax. */
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_MMA_NS; slice++) {
            const uint32_t first = slice * 8u + column;
            score[slice][0] *= scale;
            score[slice][1] *= scale;
            score[slice][2] *= scale;
            score[slice][3] *= scale;
            if (base + first >= sequence) {
                score[slice][0] = -INFINITY;
                score[slice][2] = -INFINITY;
            }
            if (base + first + 1u >= sequence) {
                score[slice][1] = -INFINITY;
                score[slice][3] = -INFINITY;
            }
        }

        /* One maximum per tile, reduced across the four lanes of a row. */
        float tile0 = -INFINITY, tile1 = -INFINITY;
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_MMA_NS; slice++) {
            tile0 = fmaxf(tile0, fmaxf(score[slice][0], score[slice][1]));
            tile1 = fmaxf(tile1, fmaxf(score[slice][2], score[slice][3]));
        }
        tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 1));
        tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 2));
        tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 1));
        tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 2));
        const float next0 = fmaxf(maximum0, tile0);
        const float next1 = fmaxf(maximum1, tile1);
        const float rescale0 = expf(maximum0 - next0);
        const float rescale1 = expf(maximum1 - next1);

        float weight[H3D_MMA_NS][4];
        float part0 = 0.0f, part1 = 0.0f;
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_MMA_NS; slice++) {
            weight[slice][0] = expf(score[slice][0] - next0);
            weight[slice][1] = expf(score[slice][1] - next0);
            weight[slice][2] = expf(score[slice][2] - next1);
            weight[slice][3] = expf(score[slice][3] - next1);
            part0 += weight[slice][0] + weight[slice][1];
            part1 += weight[slice][2] + weight[slice][3];
        }
        part0 += __shfl_xor_sync(0xffffffffu, part0, 1);
        part0 += __shfl_xor_sync(0xffffffffu, part0, 2);
        part1 += __shfl_xor_sync(0xffffffffu, part1, 1);
        part1 += __shfl_xor_sync(0xffffffffu, part1, 2);
        denominator0 = denominator0 * rescale0 + part0;
        denominator1 = denominator1 * rescale1 + part1;
        maximum0 = next0;
        maximum1 = next1;
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
            result[dim][0] *= rescale0;
            result[dim][1] *= rescale0;
            result[dim][2] *= rescale1;
            result[dim][3] *= rescale1;
        }

        /* O += P * V, P packed from the weights as the A fragments. */
#pragma unroll
        for (uint32_t step = 0; step < H3D_MMA_KEYS / 16; step++) {
            uint32_t p[4];
            p[0] = h3d_mma_pair(weight[2u * step][0], weight[2u * step][1]);
            p[1] = h3d_mma_pair(weight[2u * step][2], weight[2u * step][3]);
            p[2] = h3d_mma_pair(weight[2u * step + 1u][0],
                                weight[2u * step + 1u][1]);
            p[3] = h3d_mma_pair(weight[2u * step + 1u][2],
                                weight[2u * step + 1u][3]);
#pragma unroll
            for (uint32_t part = 0; part < H3D_MMA_DIMS / 2; part++) {
                const uint32_t row =
                        step * 16u + (matrix & 1u) * 8u + (lane & 7u);
                const uint32_t col = part * 16u + (matrix >> 1u) * 8u;
                uint32_t b[4];
                h3d_mma_load_trans(b, &shared_value[row][col]);
                h3d_mma_step(result[2u * part], p, b[0], b[1]);
                h3d_mma_step(result[2u * part + 1u], p, b[2], b[3]);
            }
        }
    }

    __syncthreads();
    if (live0) {
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
            const size_t at = ((size_t)row0 * heads + head) * H3D_MMA_HD +
                              dim * 8u + column;
            output[at] = __float2bfloat16(result[dim][0] / denominator0);
            output[at + 1u] = __float2bfloat16(result[dim][1] / denominator0);
        }
    }
    if (live1) {
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
            const size_t at = ((size_t)row1 * heads + head) * H3D_MMA_HD +
                              dim * 8u + column;
            output[at] = __float2bfloat16(result[dim][2] / denominator1);
            output[at + 1u] = __float2bfloat16(result[dim][3] / denominator1);
        }
    }
}

/* `h3d_attention_mma_kernel` in the refiner's token-major layout
 * (`[tokens, heads, dim]`), and it is the same kernel otherwise: the same
 * 16-row query tile per warp, the same 64-key staging, the same online softmax
 * and the same `mma` fragments. Only the three input indices and the two
 * extents change, because the LTX refiner's operands are strided by
 * `heads * H3D_MMA_HD` between rows while the H3 tower's are contiguous within
 * a head. The **output was already token-major** in the head-major kernel
 * (`output[((row * heads) + head) * dim]`), which is why this is a port of the
 * reads and not of the whole walk.
 *
 * Separate `q_rows` and `kv_rows` because two of the refiner's three dense
 * attentions are not square: the text cross-attention is 16,128 queries against
 * 1,024 keys, and A2V/V2A are 16,128 by 126 - though head dimension 64 keeps
 * those two on the scalar kernel, which this one does not serve.
 *
 * Arithmetic: BF16 tensor-core products with F32 accumulation and a natural-log
 * online softmax, so it is **not** bit-identical to `h3d_ltx_attention_kernel`'s
 * row-wise FMA walk. It is a separate arm for that reason (`H3_LTX_ATTN_MMA`),
 * and the numbers it moves are the ones `h3_ltx_refiner_gate --compare` reports
 * as `rel_rms`, not the bit gate. */
__global__ static void h3d_ltx_attention_mma_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t q_rows, uint32_t kv_rows, uint32_t heads, float scale) {
    __shared__ __align__(16) __nv_bfloat16 shared_key[H3D_MMA_KEYS][H3D_MMA_LD];
    __shared__ __align__(16) __nv_bfloat16 shared_value[H3D_MMA_KEYS][H3D_MMA_LD];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t quad = lane >> 2u;             /* the accumulator row a lane serves */
    const uint32_t column = (lane & 3u) * 2u;     /* first of its two columns */
    const uint32_t matrix = lane >> 3u;           /* ldmatrix: which 8x8 quadrant */
    const uint32_t head = blockIdx.y;
    const uint32_t first_row = blockIdx.x * H3D_MMA_ROWS + warp * H3D_MMA_M;
    const uint32_t row0 = first_row + quad, row1 = row0 + 8u;
    const int live0 = row0 < q_rows, live1 = row1 < q_rows;
    /* A row past the end is read from the last real row and never written; that
     * keeps the CTA's reads in bounds without branching the MMAs. */
    const uint32_t read0 = live0 ? row0 : q_rows - 1u;
    const uint32_t read1 = live1 ? row1 : q_rows - 1u;
    const __nv_bfloat16 *query0 =
            query + ((size_t)read0 * heads + head) * H3D_MMA_HD;
    const __nv_bfloat16 *query1 =
            query + ((size_t)read1 * heads + head) * H3D_MMA_HD;

    /* Q lives in registers as the eight k16 A fragments, built once per CTA. */
    uint32_t a[H3D_MMA_KS][4];
#pragma unroll
    for (uint32_t step = 0; step < H3D_MMA_KS; step++) {
        const uint32_t dim = step * 16u + column;
        a[step][0] = h3d_mma_pair(__bfloat162float(query0[dim]),
                                  __bfloat162float(query0[dim + 1u]));
        a[step][1] = h3d_mma_pair(__bfloat162float(query1[dim]),
                                  __bfloat162float(query1[dim + 1u]));
        a[step][2] = h3d_mma_pair(__bfloat162float(query0[dim + 8u]),
                                  __bfloat162float(query0[dim + 9u]));
        a[step][3] = h3d_mma_pair(__bfloat162float(query1[dim + 8u]),
                                  __bfloat162float(query1[dim + 9u]));
    }

    float maximum0 = -INFINITY, maximum1 = -INFINITY;
    float denominator0 = 0.0f, denominator1 = 0.0f;
    float result[H3D_MMA_DIMS][4];
#pragma unroll
    for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
        result[dim][0] = result[dim][1] = result[dim][2] = result[dim][3] = 0.0f;
    }

    for (uint32_t base = 0; base < kv_rows; base += H3D_MMA_KEYS) {
        __syncthreads();   /* the previous tile's staged rows are fully consumed */
        /* Stage K and V: 16-byte copies, one row per sixteen lanes. A row past
         * the end is zeroed; its scores are masked below, so it contributes
         * nothing and the staging stays branch-free. */
        for (uint32_t index = threadIdx.x;
             index < H3D_MMA_KEYS * H3D_MMA_CHUNKS;
             index += H3D_MMA_WARPS * 32u) {
            const uint32_t row = index / H3D_MMA_CHUNKS;
            const uint32_t chunk = index % H3D_MMA_CHUNKS;
            const uint32_t source = base + row;
            if (source < kv_rows) {
                const size_t at =
                        ((size_t)source * heads + head) * H3D_MMA_HD +
                        chunk * 8u;
                *(uint4 *)&shared_key[row][chunk * 8u] =
                        *(const uint4 *)(key + at);
                *(uint4 *)&shared_value[row][chunk * 8u] =
                        *(const uint4 *)(value + at);
            } else {
                const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
                *(uint4 *)&shared_key[row][chunk * 8u] = zero;
                *(uint4 *)&shared_value[row][chunk * 8u] = zero;
            }
        }
        __syncthreads();

        /* Scores: S = Q * K^T over all H3D_MMA_KEYS staged keys. */
        float score[H3D_MMA_NS][4];
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_MMA_NS; slice++) {
            score[slice][0] = score[slice][1] = 0.0f;
            score[slice][2] = score[slice][3] = 0.0f;
        }
#pragma unroll
        for (uint32_t step = 0; step < H3D_MMA_KS; step++) {
#pragma unroll
            for (uint32_t part = 0; part < H3D_MMA_KEYS / 16; part++) {
                const uint32_t row = part * 16u + (matrix >> 1u) * 8u + (lane & 7u);
                const uint32_t col = step * 16u + (matrix & 1u) * 8u;
                uint32_t b[4];
                h3d_mma_load(b, &shared_key[row][col]);
                h3d_mma_step(score[2u * part], a[step], b[0], b[1]);
                h3d_mma_step(score[2u * part + 1u], a[step], b[2], b[3]);
            }
        }

        /* Scale, then mask the tail keys out of the softmax. */
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_MMA_NS; slice++) {
            const uint32_t first = slice * 8u + column;
            score[slice][0] *= scale;
            score[slice][1] *= scale;
            score[slice][2] *= scale;
            score[slice][3] *= scale;
            if (base + first >= kv_rows) {
                score[slice][0] = -INFINITY;
                score[slice][2] = -INFINITY;
            }
            if (base + first + 1u >= kv_rows) {
                score[slice][1] = -INFINITY;
                score[slice][3] = -INFINITY;
            }
        }

        /* One maximum per tile, reduced across the four lanes of a row. */
        float tile0 = -INFINITY, tile1 = -INFINITY;
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_MMA_NS; slice++) {
            tile0 = fmaxf(tile0, fmaxf(score[slice][0], score[slice][1]));
            tile1 = fmaxf(tile1, fmaxf(score[slice][2], score[slice][3]));
        }
        tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 1));
        tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 2));
        tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 1));
        tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 2));
        const float next0 = fmaxf(maximum0, tile0);
        const float next1 = fmaxf(maximum1, tile1);
        const float rescale0 = expf(maximum0 - next0);
        const float rescale1 = expf(maximum1 - next1);

        float weight[H3D_MMA_NS][4];
        float part0 = 0.0f, part1 = 0.0f;
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_MMA_NS; slice++) {
            weight[slice][0] = expf(score[slice][0] - next0);
            weight[slice][1] = expf(score[slice][1] - next0);
            weight[slice][2] = expf(score[slice][2] - next1);
            weight[slice][3] = expf(score[slice][3] - next1);
            part0 += weight[slice][0] + weight[slice][1];
            part1 += weight[slice][2] + weight[slice][3];
        }
        part0 += __shfl_xor_sync(0xffffffffu, part0, 1);
        part0 += __shfl_xor_sync(0xffffffffu, part0, 2);
        part1 += __shfl_xor_sync(0xffffffffu, part1, 1);
        part1 += __shfl_xor_sync(0xffffffffu, part1, 2);
        denominator0 = denominator0 * rescale0 + part0;
        denominator1 = denominator1 * rescale1 + part1;
        maximum0 = next0;
        maximum1 = next1;
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
            result[dim][0] *= rescale0;
            result[dim][1] *= rescale0;
            result[dim][2] *= rescale1;
            result[dim][3] *= rescale1;
        }

        /* O += P * V, P packed from the weights as the A fragments. */
#pragma unroll
        for (uint32_t step = 0; step < H3D_MMA_KEYS / 16; step++) {
            uint32_t p[4];
            p[0] = h3d_mma_pair(weight[2u * step][0], weight[2u * step][1]);
            p[1] = h3d_mma_pair(weight[2u * step][2], weight[2u * step][3]);
            p[2] = h3d_mma_pair(weight[2u * step + 1u][0],
                                weight[2u * step + 1u][1]);
            p[3] = h3d_mma_pair(weight[2u * step + 1u][2],
                                weight[2u * step + 1u][3]);
#pragma unroll
            for (uint32_t part = 0; part < H3D_MMA_DIMS / 2; part++) {
                const uint32_t row =
                        step * 16u + (matrix & 1u) * 8u + (lane & 7u);
                const uint32_t col = part * 16u + (matrix >> 1u) * 8u;
                uint32_t b[4];
                h3d_mma_load_trans(b, &shared_value[row][col]);
                h3d_mma_step(result[2u * part], p, b[0], b[1]);
                h3d_mma_step(result[2u * part + 1u], p, b[2], b[3]);
            }
        }
    }

    __syncthreads();
    if (live0) {
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
            const size_t at = ((size_t)row0 * heads + head) * H3D_MMA_HD +
                              dim * 8u + column;
            output[at] = __float2bfloat16(result[dim][0] / denominator0);
            output[at + 1u] = __float2bfloat16(result[dim][1] / denominator0);
        }
    }
    if (live1) {
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
            const size_t at = ((size_t)row1 * heads + head) * H3D_MMA_HD +
                              dim * 8u + column;
            output[at] = __float2bfloat16(result[dim][2] / denominator1);
            output[at + 1u] = __float2bfloat16(result[dim][3] / denominator1);
        }
    }
}

/* The refiner's dense attention, tensor-core arm. Same validation as
 * `h3d_ltx_attention` (`:3338`) so an arm only ever changes the arithmetic, plus
 * the one restriction the kernel carries: head dimension 128, because the
 * fragments are built for `H3D_MMA_HD` and A2V/V2A are 64. Those two stay on
 * the scalar kernel, which is 4.1 % of a block. */
int h3d_ltx_attention_mma(h3d_context *context, h3d_tensor *output,
                          const h3d_tensor *query, const h3d_tensor *key,
                          const h3d_tensor *value, uint32_t q_rows,
                          uint32_t kv_rows, uint32_t heads, uint32_t head_dim,
                          float scale, char *error, size_t error_size) {
    size_t query_elements = 0, key_elements = 0;
    if (!context || !q_rows || !kv_rows || !heads || head_dim != H3D_MMA_HD ||
        !h3d_mul_size((size_t)q_rows * heads, head_dim, &query_elements) ||
        !h3d_mul_size((size_t)kv_rows * heads, head_dim, &key_elements) ||
        !h3d_is(output, H3D_BF16, query_elements) ||
        !h3d_is(query, H3D_BF16, query_elements) ||
        !h3d_is(key, H3D_BF16, key_elements) ||
        !h3d_is(value, H3D_BF16, key_elements)) {
        return h3d_set_error(error, error_size,
                             "invalid LTX attention (mma) tensors or shape");
    }
    dim3 grid((q_rows + H3D_MMA_ROWS - 1) / H3D_MMA_ROWS, heads);
    h3d_ltx_attention_mma_kernel<<<grid, H3D_MMA_WARPS * 32u, 0,
                                   context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
        (const __nv_bfloat16 *)key->data, (const __nv_bfloat16 *)value->data,
        q_rows, kv_rows, heads, scale);
    return h3d_launch_ok("LTX attention (mma)", error, error_size);
}

int h3d_sdpa(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *query, const h3d_tensor *key,
             const h3d_tensor *value, uint32_t sequence, uint32_t heads,
             uint32_t head_dim, float scale, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !sequence || !heads ||
        (head_dim != 128 && head_dim != 64) ||
        !h3d_mul_size((size_t)sequence * heads, head_dim, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(query, H3D_BF16, elements) ||
        !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(value, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid attention tensors or shape");
    }
    /* The benchmark arm: same tensors and scale, tensor-core arithmetic. Its
     * kernel is the 128-wide one, so the 64-wide connector attention stays on
     * the tiled path. */
    if (context->attention_mma && head_dim == 128) {
        dim3 mma_grid((sequence + H3D_MMA_ROWS - 1) / H3D_MMA_ROWS, heads);
        h3d_attention_mma_kernel<<<mma_grid, H3D_MMA_WARPS * 32u, 0,
                                   context->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
            (const __nv_bfloat16 *)key->data,
            (const __nv_bfloat16 *)value->data, sequence, heads, scale);
        return h3d_launch_ok("BF16 attention (mma)", error, error_size);
    }
    dim3 grid((sequence + 8 * H3D_ATTENTION_ROWS - 1) / (8 * H3D_ATTENTION_ROWS),
              heads);
    if (head_dim == 64) {
        h3d_attention_tiled_kernel<64><<<grid, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
            (const __nv_bfloat16 *)key->data,
            (const __nv_bfloat16 *)value->data, sequence, heads, scale);
        return h3d_launch_ok("BF16 attention 64", error, error_size);
    }
    h3d_attention_tiled_kernel<128><<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
        (const __nv_bfloat16 *)key->data, (const __nv_bfloat16 *)value->data,
        sequence, heads, scale);
    return h3d_launch_ok("BF16 attention", error, error_size);
}

/* `h3_unary_f32_kernel` and `h3_unary_bf16_kernel` (`:2246-2274`), SiLU
 * branch only; both set `value /= 1 + expf(-value)` in F32. */
__global__ static void h3d_silu_f32_kernel(float *output, const float *input,
                                           size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = input[index];
    value /= 1.0f + expf(-value);
    output[index] = value;
}

__global__ static void h3d_silu_bf16_kernel(__nv_bfloat16 *output,
                                            const __nv_bfloat16 *input,
                                            size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = __bfloat162float(input[index]);
    value /= 1.0f + expf(-value);
    output[index] = __float2bfloat16(value);
}

int h3d_silu_f32(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, size_t elements, char *error,
                 size_t error_size) {
    if (!context || !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(input, H3D_F32, elements)) {
        return h3d_set_error(error, error_size, "invalid F32 SiLU tensors");
    }
    if (!elements) return 1;
    h3d_silu_f32_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                          context->stream>>>((float *)output->data,
                                             (const float *)input->data,
                                             elements);
    return h3d_launch_ok("F32 SiLU", error, error_size);
}

int h3d_silu_bf16(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *input, size_t elements, char *error,
                  size_t error_size) {
    if (!context || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size, "invalid BF16 SiLU tensors");
    }
    if (!elements) return 1;
    h3d_silu_bf16_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                           context->stream>>>((__nv_bfloat16 *)output->data,
                                              (const __nv_bfloat16 *)input->data,
                                              elements);
    return h3d_launch_ok("BF16 SiLU", error, error_size);
}

/* `h3_binary_bf16_kernel` (`:2347-2357`), add branch only. */
__global__ static void h3d_add_kernel(__nv_bfloat16 *output,
                                      const __nv_bfloat16 *left,
                                      const __nv_bfloat16 *right,
                                      size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float a = __bfloat162float(left[index]);
    float b = __bfloat162float(right[index]);
    output[index] = __float2bfloat16(a + b);
}

int h3d_add(h3d_context *context, h3d_tensor *output,
            const h3d_tensor *left, const h3d_tensor *right, size_t elements,
            char *error, size_t error_size) {
    if (!context || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(left, H3D_BF16, elements) ||
        !h3d_is(right, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size, "invalid BF16 add tensors");
    }
    if (!elements) return 1;
    h3d_add_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                     context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)left->data,
        (const __nv_bfloat16 *)right->data, elements);
    return h3d_launch_ok("BF16 add", error, error_size);
}

/* `h3_binary_bf16_kernel` (`:2347-2357`), subtract branch only. */
__global__ static void h3d_sub_kernel(__nv_bfloat16 *output,
                                      const __nv_bfloat16 *left,
                                      const __nv_bfloat16 *right,
                                      size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float a = __bfloat162float(left[index]);
    float b = __bfloat162float(right[index]);
    output[index] = __float2bfloat16(a - b);
}

int h3d_sub(h3d_context *context, h3d_tensor *output,
            const h3d_tensor *left, const h3d_tensor *right, size_t elements,
            char *error, size_t error_size) {
    if (!context || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(left, H3D_BF16, elements) ||
        !h3d_is(right, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size, "invalid BF16 subtract tensors");
    }
    if (!elements) return 1;
    h3d_sub_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                     context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)left->data,
        (const __nv_bfloat16 *)right->data, elements);
    return h3d_launch_ok("BF16 subtract", error, error_size);
}

/* `h3_f32_to_bf16_kernel` (`h3_gpu_cuda.cu:519-525`). */
__global__ static void h3d_cast_kernel(__nv_bfloat16 *output,
                                       const float *input, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = __float2bfloat16(input[index]);
}

int h3d_cast_f32_to_bf16(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, size_t elements, char *error,
                         size_t error_size) {
    if (!context || !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(input, H3D_F32, elements)) {
        return h3d_set_error(error, error_size, "invalid BF16 cast tensors");
    }
    if (!elements) return 1;
    h3d_cast_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                      context->stream>>>((__nv_bfloat16 *)output->data,
                                         (const float *)input->data, elements);
    return h3d_launch_ok("BF16 cast", error, error_size);
}

/* `h3d_cast_kernel`'s reverse: the video VAE decoder's BF16 arm runs its GEMMs
 * narrowed and its consumers in F32, so every result is widened once. The value
 * is exact - a BF16 is an F32 with the low 16 bits zero. */
__global__ static void h3d_uncast_kernel(float *output,
                                         const __nv_bfloat16 *input,
                                         size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = __bfloat162float(input[index]);
}

int h3d_cast_bf16_to_f32(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, size_t elements, char *error,
                         size_t error_size) {
    if (!context || !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(input, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size, "invalid F32 cast tensors");
    }
    if (!elements) return 1;
    h3d_uncast_kernel<<<(unsigned)((elements + 255) / 256), 256, 0,
                        context->stream>>>((float *)output->data,
                                           (const __nv_bfloat16 *)input->data,
                                           elements);
    return h3d_launch_ok("F32 cast", error, error_size);
}

/* `h3_copy` (`:672-688`). */
int h3d_copy(h3d_context *context, h3d_tensor *destination,
             size_t destination_element, const h3d_tensor *source,
             size_t source_element, size_t elements, char *error,
             size_t error_size) {
    if (!context || !destination || !source ||
        destination->dtype != source->dtype ||
        destination_element > destination->elements ||
        source_element > source->elements ||
        elements > destination->elements - destination_element ||
        elements > source->elements - source_element) {
        return h3d_set_error(error, error_size, "invalid tensor copy");
    }
    if (!elements) return 1;
    size_t item = h3d_dtype_size(destination->dtype);
    return h3d_cuda_ok(
        cudaMemcpyAsync((char *)destination->data + destination_element * item,
                        (const char *)source->data + source_element * item,
                        elements * item, cudaMemcpyDeviceToDevice,
                        context->stream),
        "tensor copy", error, error_size);
}

int h3d_sync(h3d_context *context, char *error, size_t error_size) {
    if (!context) return h3d_set_error(error, error_size, "no DiT context");
    return h3d_cuda_ok(cudaStreamSynchronize(context->stream),
                       "synchronize DiT stream", error, error_size);
}

/* ---------------------------------------------------------------------------
 * F32 primitives for the AudioVAE decoder. Same source of truth, same
 * expression order; the reference's decode path calls only these
 * (`h3_audio_vae.c:1-715`).
 * ------------------------------------------------------------------------- */

int h3d_copy_f32(h3d_context *context, h3d_tensor *destination,
                 size_t destination_element, const h3d_tensor *source,
                 size_t source_element, size_t elements, char *error,
                 size_t error_size) {
    if (!destination || !source || destination->dtype != H3D_F32 ||
        source->dtype != H3D_F32) {
        return h3d_set_error(error, error_size, "invalid F32 tensor copy");
    }
    return h3d_copy(context, destination, destination_element, source,
                    source_element, elements, error, error_size);
}

/* `h3_conv1d_kernel` (`:1681-1706`). */
__global__ static void h3d_conv1d_kernel(
    float *output, const float *input, const float *weight, const float *bias,
    uint32_t batch, uint32_t length, uint32_t input_channels,
    uint32_t output_channels, uint32_t kernel, uint32_t stride,
    uint32_t padding, uint32_t dilation, uint32_t output_length,
    size_t output_elements, int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    uint32_t output_channel = (uint32_t)(index % output_channels);
    size_t row = index / output_channels;
    uint32_t output_time = (uint32_t)(row % output_length);
    uint32_t batch_index = (uint32_t)(row / output_length);
    float sum = has_bias ? bias[output_channel] : 0.0f;
    for (uint32_t input_channel = 0; input_channel < input_channels;
         input_channel++) {
        for (uint32_t tap = 0; tap < kernel; tap++) {
            int64_t input_time = (int64_t)output_time * stride - padding +
                                 (int64_t)tap * dilation;
            if (input_time < 0 || input_time >= length) continue;
            size_t input_index = ((size_t)batch_index * length +
                                  (size_t)input_time) * input_channels +
                                 input_channel;
            size_t weight_index = ((size_t)output_channel * input_channels +
                                   input_channel) * kernel + tap;
            sum = fmaf(input[input_index], weight[weight_index], sum);
        }
    }
    output[((size_t)batch_index * output_length + output_time) *
           output_channels + output_channel] = sum;
}

int h3d_conv1d(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t batch, uint32_t length,
               uint32_t input_channels, uint32_t output_channels,
               uint32_t kernel, uint32_t stride, uint32_t padding,
               uint32_t dilation, char *error, size_t error_size) {
    uint64_t effective = (uint64_t)dilation * (kernel ? kernel - 1 : 0) + 1;
    if (!context || !batch || !length || !input_channels || !output_channels ||
        !kernel || !stride || !dilation ||
        (uint64_t)length + 2ull * padding < effective) {
        return h3d_set_error(error, error_size, "invalid Conv1d shape");
    }
    uint64_t output_length64 =
        ((uint64_t)length + 2ull * padding - effective) / stride + 1;
    if (output_length64 > UINT32_MAX) {
        return h3d_set_error(error, error_size, "Conv1d output overflows");
    }
    size_t input_elements, weight_elements, output_elements, bias_elements;
    if (!h3d_mul_size((size_t)batch * length, input_channels,
                      &input_elements) ||
        !h3d_mul_size((size_t)output_channels * input_channels, kernel,
                      &weight_elements) ||
        !h3d_mul_size((size_t)batch * (size_t)output_length64,
                      output_channels, &output_elements) ||
        !h3d_mul_size(output_channels, 1, &bias_elements) ||
        !h3d_is(input, H3D_F32, input_elements) ||
        !h3d_is(weight, H3D_F32, weight_elements) ||
        !h3d_is(output, H3D_F32, output_elements) ||
        (bias && !h3d_is(bias, H3D_F32, bias_elements))) {
        return h3d_set_error(error, error_size, "invalid Conv1d tensors");
    }
    if (!output_elements) return 1;
    h3d_conv1d_kernel<<<(output_elements + 255) / 256, 256, 0,
                        context->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data,
        bias ? (const float *)bias->data : (const float *)input->data, batch,
        length, input_channels, output_channels, kernel, stride, padding,
        dilation, (uint32_t)output_length64, output_elements, bias != NULL);
    return h3d_launch_ok("Conv1d", error, error_size);
}

/* `h3_conv_transpose1d_kernel` (`:1714-1743`). */
__global__ static void h3d_conv_transpose1d_kernel(
    float *output, const float *input, const float *weight, const float *bias,
    uint32_t batch, uint32_t length, uint32_t input_channels,
    uint32_t output_channels, uint32_t kernel, uint32_t stride,
    uint32_t padding, uint32_t output_length, size_t output_elements,
    int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    uint32_t output_channel = (uint32_t)(index % output_channels);
    size_t row = index / output_channels;
    uint32_t output_time = (uint32_t)(row % output_length);
    uint32_t batch_index = (uint32_t)(row / output_length);
    float sum = has_bias ? bias[output_channel] : 0.0f;
    for (uint32_t input_channel = 0; input_channel < input_channels;
         input_channel++) {
        for (uint32_t tap = 0; tap < kernel; tap++) {
            int64_t numerator = (int64_t)output_time + padding - tap;
            if (numerator < 0 || numerator % stride) continue;
            uint64_t input_time = (uint64_t)numerator / stride;
            if (input_time >= length) continue;
            size_t input_index = ((size_t)batch_index * length + input_time) *
                                 input_channels + input_channel;
            size_t weight_index = ((size_t)input_channel * output_channels +
                                   output_channel) * kernel + tap;
            sum = fmaf(input[input_index], weight[weight_index], sum);
        }
    }
    output[((size_t)batch_index * output_length + output_time) *
           output_channels + output_channel] = sum;
}

int h3d_conv_transpose1d(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, const h3d_tensor *weight,
                         const h3d_tensor *bias, uint32_t batch,
                         uint32_t length, uint32_t input_channels,
                         uint32_t output_channels, uint32_t kernel,
                         uint32_t stride, uint32_t padding, char *error,
                         size_t error_size) {
    uint64_t full = length ? (uint64_t)(length - 1) * stride + kernel : 0;
    if (!context || !batch || !length || !input_channels || !output_channels ||
        !kernel || !stride || full < 2ull * padding ||
        full - 2ull * padding > UINT32_MAX) {
        return h3d_set_error(error, error_size,
                             "invalid ConvTranspose1d shape");
    }
    uint32_t output_length = (uint32_t)(full - 2ull * padding);
    size_t input_elements, weight_elements, output_elements;
    if (!h3d_mul_size((size_t)batch * length, input_channels,
                      &input_elements) ||
        !h3d_mul_size((size_t)input_channels * output_channels, kernel,
                      &weight_elements) ||
        !h3d_mul_size((size_t)batch * output_length, output_channels,
                      &output_elements) ||
        !h3d_is(input, H3D_F32, input_elements) ||
        !h3d_is(weight, H3D_F32, weight_elements) ||
        !h3d_is(output, H3D_F32, output_elements) ||
        (bias && !h3d_is(bias, H3D_F32, output_channels))) {
        return h3d_set_error(error, error_size,
                             "invalid ConvTranspose1d tensors");
    }
    if (!output_elements) return 1;
    h3d_conv_transpose1d_kernel<<<(output_elements + 255) / 256, 256, 0,
                                  context->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data,
        bias ? (const float *)bias->data : (const float *)input->data, batch,
        length, input_channels, output_channels, kernel, stride, padding,
        output_length, output_elements, bias != NULL);
    return h3d_launch_ok("ConvTranspose1d", error, error_size);
}

/* `h3_weight_norm_kernel` (`:2958-2974`). */
__global__ static void h3d_weight_norm_kernel(float *output,
                                              const float *vector,
                                              const float *magnitude,
                                              uint32_t outer,
                                              uint32_t inner) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    size_t base = (size_t)row * inner;
    float square_sum = 0.0f;
    for (uint32_t column = 0; column < inner; column++) {
        square_sum =
            fmaf(vector[base + column], vector[base + column], square_sum);
    }
    float scale = magnitude[row] * rsqrtf(square_sum);
    for (uint32_t column = 0; column < inner; column++) {
        output[base + column] = vector[base + column] * scale;
    }
}

int h3d_weight_norm(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *vector, const h3d_tensor *magnitude,
                    uint32_t outer, uint32_t inner, char *error,
                    size_t error_size) {
    size_t elements;
    if (!context || !h3d_mul_size(outer, inner, &elements) ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(vector, H3D_F32, elements) ||
        !h3d_is(magnitude, H3D_F32, outer)) {
        return h3d_set_error(error, error_size,
                             "invalid weight norm arguments");
    }
    if (!outer) return 1;
    h3d_weight_norm_kernel<<<(outer + 255) / 256, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)vector->data,
        (const float *)magnitude->data, outer, inner);
    return h3d_launch_ok("weight norm", error, error_size);
}

/* `h3_alias_free_snake_kernel` (`:1907-1933`). */
__global__ static void h3d_alias_free_snake_kernel(
    float *output, const float *input, const float *alpha_log,
    const float *beta_log, const float *upsample_filter,
    const float *downsample_filter, uint32_t batch, uint32_t length,
    uint32_t channels, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = (uint32_t)(index % channels);
    size_t row = index / channels;
    uint32_t time = (uint32_t)(row % length);
    uint32_t batch_index = (uint32_t)(row / length);
    float alpha = expf(alpha_log[channel]);
    float beta = expf(beta_log[channel]);
    float result = 0.0f;
    for (int down_tap = 0; down_tap < 12; down_tap++) {
        int up_time = max(0, min((int)length * 2 - 1,
                                 (int)time * 2 + down_tap - 5));
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_tap = 0; up_tap < 12; up_tap++) {
            int numerator = raw_time - up_tap;
            if (numerator < 0 || (numerator & 1)) continue;
            int source_time = max(0, min((int)length - 1,
                                         numerator / 2 - 5));
            size_t source = ((size_t)batch_index * length + source_time) *
                            channels + channel;
            upsampled = fmaf(input[source], 2.0f * upsample_filter[up_tap],
                              upsampled);
        }
        float sine = sinf(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fmaf(activated, downsample_filter[down_tap], result);
    }
    output[((size_t)batch_index * length + time) * channels + channel] = result;
}

int h3d_alias_free_snake(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input,
                         const h3d_tensor *alpha_log,
                         const h3d_tensor *beta_log,
                         const h3d_tensor *upsample_filter,
                         const h3d_tensor *downsample_filter, uint32_t batch,
                         uint32_t length, uint32_t channels, char *error,
                         size_t error_size) {
    size_t elements;
    if (!context || !batch || !length || !channels ||
        !h3d_mul_size((size_t)batch * length, channels, &elements) ||
        !h3d_is(input, H3D_F32, elements) ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(alpha_log, H3D_F32, channels) ||
        !h3d_is(beta_log, H3D_F32, channels) ||
        !h3d_is(upsample_filter, H3D_F32, 12) ||
        !h3d_is(downsample_filter, H3D_F32, 12)) {
        return h3d_set_error(error, error_size,
                             "invalid alias-free Snake tensors");
    }
    if (!elements) return 1;
    h3d_alias_free_snake_kernel<<<(elements + 255) / 256, 256, 0,
                                  context->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)alpha_log->data, (const float *)beta_log->data,
        (const float *)upsample_filter->data,
        (const float *)downsample_filter->data, batch, length, channels,
        elements);
    return h3d_launch_ok("alias-free Snake", error, error_size);
}

/* `h3_add_scaled_kernel` (`:2385-2391`). */
__global__ static void h3d_add_scaled_kernel(float *output,
                                             const float *left,
                                             const float *right, size_t count,
                                             float left_scale,
                                             float right_scale) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left[index] * left_scale + right[index] * right_scale;
    }
}

int h3d_add_scaled(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *left, const h3d_tensor *right,
                   float left_scale, float right_scale, size_t elements,
                   char *error, size_t error_size) {
    if (!context || !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(left, H3D_F32, elements) || !h3d_is(right, H3D_F32, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid F32 scaled-add tensors");
    }
    if (!elements) return 1;
    h3d_add_scaled_kernel<<<(elements + 255) / 256, 256, 0,
                            context->stream>>>(
        (float *)output->data, (const float *)left->data,
        (const float *)right->data, elements, left_scale, right_scale);
    return h3d_launch_ok("F32 scaled add", error, error_size);
}

/* `H3_CLIP` in `h3_unary_f32_kernel` (`:2253-2261`). */
__global__ static void h3d_clip_kernel(float *output, const float *input,
                                       size_t count, float minimum,
                                       float maximum) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = input[index];
    output[index] = fminf(maximum, fmaxf(minimum, value));
}

int h3d_clip(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *input, size_t elements, float minimum,
             float maximum, char *error, size_t error_size) {
    if (!context || !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(input, H3D_F32, elements) || minimum > maximum) {
        return h3d_set_error(error, error_size, "invalid F32 clip arguments");
    }
    if (!elements) return 1;
    h3d_clip_kernel<<<(elements + 255) / 256, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)input->data, elements, minimum,
        maximum);
    return h3d_launch_ok("F32 clip", error, error_size);
}

/* ---- F32 primitives the VideoVAE decoder needs (`h3_video_vae.c`) ---- */

/* `h3_attention_kernel<float>` (`:1337-1416`): one block per (head, row,
 * batch), one dimension per thread, the softmax reduced over shared memory.
 * The reference keeps the shared scalars in `__shared__` so every thread sees
 * thread 0's rescale and weight. Used for any head width other than 64. */
__global__ static void h3d_attention_f32_kernel(
        float *output, const float *query, const float *key,
        const float *value, uint32_t batch, uint32_t sequence,
        uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
        float scale, int causal, int head_major_output,
        int head_major_input) {
    __shared__ float partial[1024];
    __shared__ float rescale;
    __shared__ float weight;
    __shared__ float denominator;
    uint32_t query_head = blockIdx.x;
    uint32_t row = blockIdx.y;
    uint32_t batch_index = blockIdx.z;
    uint32_t kv_head = query_head / (query_heads / kv_heads);
    uint32_t keys = causal ? row + 1 : sequence;
    size_t query_base = head_major_input ?
        (((size_t)batch_index * query_heads + query_head) * sequence + row) *
            head_dim :
        (((size_t)batch_index * sequence + row) * query_heads + query_head) *
            head_dim;
    uint32_t dimension = threadIdx.x;
    float result = 0.0f;
    float maximum = -INFINITY;
    if (threadIdx.x == 0) denominator = 0.0f;
    __syncthreads();
    for (uint32_t key_row = 0; key_row < keys; key_row++) {
        size_t key_base = head_major_input ?
            (((size_t)batch_index * kv_heads + kv_head) * sequence + key_row) *
                head_dim :
            (((size_t)batch_index * sequence + key_row) * kv_heads + kv_head) *
                head_dim;
        float product = 0.0f;
        if (dimension < head_dim) {
            product = query[query_base + dimension] * key[key_base + dimension];
        }
        partial[threadIdx.x] = product;
        __syncthreads();
        for (uint32_t offset = blockDim.x / 2; offset; offset >>= 1) {
            if (threadIdx.x < offset) {
                partial[threadIdx.x] += partial[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = partial[0] * scale;
            float next_maximum = fmaxf(maximum, score);
            rescale = expf(maximum - next_maximum);
            weight = expf(score - next_maximum);
            denominator = denominator * rescale + weight;
            maximum = next_maximum;
        }
        __syncthreads();
        if (dimension < head_dim) {
            size_t value_index = key_base + dimension;
            result = result * rescale + weight * value[value_index];
        }
        __syncthreads();
    }
    if (dimension < head_dim) {
        result /= denominator;
        size_t output_index = head_major_output ?
            (((size_t)batch_index * query_heads + query_head) * sequence + row) *
                head_dim + dimension :
            (((size_t)batch_index * sequence + row) * query_heads + query_head) *
                head_dim + dimension;
        output[output_index] = result;
    }
}

/* Key rows staged per barrier pair and rows one warp owns in the F32 kernel.
 * 1 and 1 are the reference's mapping; more of either amortizes the staged rows
 * and the barrier pair. Both are per translation unit so a sweep can build the
 * reproducer with -DH3D_ATTENTION_F32_KEYS=n and -DH3D_ATTENTION_F32_ROWS=n and
 * compare output hashes. */
#ifndef H3D_ATTENTION_F32_KEYS
#define H3D_ATTENTION_F32_KEYS 8
#endif
#ifndef H3D_ATTENTION_F32_ROWS
#define H3D_ATTENTION_F32_ROWS 2
#endif

/* `h3_attention_tiled_f32_kernel` (`:1489-1547`): one block per head, `HEAD_DIM
 * / 32` dimensions per lane, and a block row of QUERIES query rows. With a
 * 64-wide head two items per lane carry data and only the first 64 threads fill
 * a staged key/value row, exactly as the reference does.
 *
 * Deviations from the reference, named and measured: the reference stages one
 * key row per barrier pair and re-reads the query row from global for every key
 * row; this stages H3D_ATTENTION_F32_KEYS rows and holds the row's items in
 * registers. The reference also gives each warp one query row; this gives it
 * H3D_ATTENTION_F32_ROWS rows, served from the same staged rows. Every row runs
 * the reference's arithmetic in the reference's order, so the output stays
 * bit-identical. */
template <int HEAD_DIM, int QUERY_MAJOR>
__global__ static void h3d_attention_tiled_f32_kernel(
        float *output, const float *query, const float *key,
        const float *value, uint32_t sequence, uint32_t heads, float scale,
        int head_major_output) {
    enum { ITEMS = HEAD_DIM / 32, KEYS = H3D_ATTENTION_F32_KEYS,
           ROWS = H3D_ATTENTION_F32_ROWS, QUERIES = 8 * ROWS };
    __shared__ float shared_key[KEYS][HEAD_DIM];
    __shared__ float shared_value[KEYS][HEAD_DIM];
    uint32_t warp = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    /* `QUERY_MAJOR` is the CTA order only: `grid(query blocks, heads)` keeps
     * the CTAs resident together on one head, whose staged keys and values are
     * then read from L2 by the next query block; the reference's
     * `grid(heads, query blocks)` spreads the resident CTAs over every head.
     * Neither order changes a single arithmetic operation, so the two are
     * bit-identical and the option is a cache experiment (`H3_VAE_SDPA_ORDER`). */
    uint32_t head = QUERY_MAJOR ? blockIdx.y : blockIdx.x;
    uint32_t first_row =
            (QUERY_MAJOR ? blockIdx.x : blockIdx.y) * QUERIES + warp * ROWS;
    int active[ROWS];
    float result[ROWS][ITEMS];
    float maximum[ROWS];
    float denominator[ROWS];
    float query_item[ROWS][ITEMS];
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        uint32_t row = first_row + r;
        active[r] = row < sequence;
        maximum[r] = -INFINITY;
        denominator[r] = 0.0f;
#pragma unroll
        for (uint32_t item = 0; item < ITEMS; item++) {
            result[r][item] = 0.0f;
            query_item[r][item] = 0.0f;
        }
        if (!active[r]) {
            continue;
        }
        size_t query_base = ((size_t)head * sequence + row) * HEAD_DIM;
#pragma unroll
        for (uint32_t item = 0; item < ITEMS; item++) {
            query_item[r][item] = query[query_base + lane + item * 32];
        }
    }
    for (uint32_t key_block = 0; key_block < sequence; key_block += KEYS) {
        /* KEYS independent loads per thread: the staged rows' latencies
         * overlap instead of one row's latency stalling every iteration. */
        if (threadIdx.x < HEAD_DIM) {
#pragma unroll
            for (uint32_t item = 0; item < KEYS; item++) {
                uint32_t source = key_block + item;
                if (source < sequence) {
                    size_t index =
                            ((size_t)head * sequence + source) * HEAD_DIM +
                            threadIdx.x;
                    shared_key[item][threadIdx.x] = key[index];
                    shared_value[item][threadIdx.x] = value[index];
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t item = 0; item < KEYS; item++) {
            if (key_block + item >= sequence) {
                break;
            }
#pragma unroll
            for (uint32_t r = 0; r < ROWS; r++) {
                float product = 0.0f;
                if (active[r]) {
#pragma unroll
                    for (uint32_t part = 0; part < ITEMS; part++) {
                        uint32_t dimension = lane + part * 32;
                        product += query_item[r][part] *
                                   shared_key[item][dimension];
                    }
#pragma unroll
                    for (uint32_t offset = 16; offset; offset >>= 1) {
                        product += __shfl_down_sync(0xffffffffu, product, offset);
                    }
                }
                float rescale = 1.0f;
                float weight = 0.0f;
                if (active[r] && lane == 0) {
                    float score = product * scale;
                    float next_maximum = fmaxf(maximum[r], score);
                    rescale = expf(maximum[r] - next_maximum);
                    weight = expf(score - next_maximum);
                    denominator[r] = denominator[r] * rescale + weight;
                    maximum[r] = next_maximum;
                }
                rescale = __shfl_sync(0xffffffffu, rescale, 0);
                weight = __shfl_sync(0xffffffffu, weight, 0);
                if (active[r]) {
#pragma unroll
                    for (uint32_t part = 0; part < ITEMS; part++) {
                        uint32_t dimension = lane + part * 32;
                        result[r][part] = result[r][part] * rescale +
                                          weight *
                                          shared_value[item][dimension];
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        denominator[r] = __shfl_sync(0xffffffffu, denominator[r], 0);
        if (!active[r]) {
            continue;
        }
        uint32_t row = first_row + r;
#pragma unroll
        for (uint32_t item = 0; item < ITEMS; item++) {
            uint32_t dimension = lane + item * 32;
            size_t index = head_major_output ?
                ((size_t)head * sequence + row) * HEAD_DIM + dimension :
                ((size_t)row * heads + head) * HEAD_DIM + dimension;
            output[index] = result[r][item] / denominator[r];
        }
    }
}

/* ---------------------------------------------------------------------------
 * Tensor-core attention for the video VAE decoder, the arm behind
 * `H3_VAE_ATTENTION=tensor`.
 *
 * Shape: one CTA covers H3D_VFA_ROWS query rows of one head, a warp owns
 * H3D_VFA_M of them and all H3D_VFA_HD key dimensions, so a row's softmax
 * statistics never cross a warp and no partial-score exchange is needed: the
 * same choice this file's BF16 `h3d_attention_mma_kernel` makes, and the
 * opposite of `cuda/ds41_attn_mma.cuh`, whose 512-dimension head does split K.
 * The step, the ldmatrix pair and the BF16 pair are mirrored from
 * `cuda/ds41_attn_mma.cuh:144-157` and the Q fragments, accumulator layout and
 * P-to-A packing from `:206-277`. The decoder carries F32 tensors, so Q, K and
 * V round to BF16 as they are staged, which the BF16 kernel gets for free.
 *
 * Deviations from the reference, named rather than hidden: Q, K and V round to
 * BF16 at staging; the scores reduce through `mma` instead of the reference's
 * four-element lane partial and five-step shuffle; the online softmax takes one
 * maximum per 64-key tile instead of rescaling on every key row; and P rounds to
 * BF16 for the second `mma`. Scores, the running maximum and sum and the output
 * accumulator stay FP32, so the arithmetic is the reference's in kind and
 * differs only in order.
 * ------------------------------------------------------------------------- */
enum {
    H3D_VFA_HD = 64,                /* the decoder's head_dim, the one shape served */
    H3D_VFA_KEYS = 64,              /* key rows staged per tile */
    H3D_VFA_M = 16,                 /* query rows a warp owns */
    H3D_VFA_WARPS = 4,
    H3D_VFA_ROWS = H3D_VFA_M * H3D_VFA_WARPS,
    H3D_VFA_LD = H3D_VFA_HD + 8,    /* +8 bf16 spreads ldmatrix rows across banks */
    H3D_VFA_DIMS = H3D_VFA_HD / 8,  /* n8 slices of the output */
    H3D_VFA_KS = H3D_VFA_HD / 16,   /* k16 steps over the head dimension */
    H3D_VFA_NS = H3D_VFA_KEYS / 8,  /* n8 slices of the scores */
    H3D_VFA_CHUNKS = H3D_VFA_HD / 8 /* 16-byte copies in one staged row */
};

/* Eight consecutive F32 values as eight consecutive BF16 halves: one 16-byte
 * chunk of a staged row. */
__device__ __forceinline__ static uint4 h3d_mma_bf16_chunk(const float *source) {
    uint4 packed;
    packed.x = h3d_mma_pair(source[0], source[1]);
    packed.y = h3d_mma_pair(source[2], source[3]);
    packed.z = h3d_mma_pair(source[4], source[5]);
    packed.w = h3d_mma_pair(source[6], source[7]);
    return packed;
}

__global__ static void h3d_attention_mma_f32_kernel(
        float *output, const float *query, const float *key,
        const float *value, uint32_t sequence, uint32_t heads, float scale) {
    __shared__ __align__(16) __nv_bfloat16 shared_key[H3D_VFA_KEYS][H3D_VFA_LD];
    __shared__ __align__(16) __nv_bfloat16 shared_value[H3D_VFA_KEYS][H3D_VFA_LD];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t quad = lane >> 2u;         /* the accumulator row a lane serves */
    const uint32_t column = (lane & 3u) * 2u; /* first of its two columns */
    const uint32_t matrix = lane >> 3u;       /* ldmatrix: which 8x8 quadrant */
    const uint32_t head = blockIdx.y;
    const uint32_t first_row = blockIdx.x * H3D_VFA_ROWS + warp * H3D_VFA_M;
    const uint32_t row0 = first_row + quad, row1 = row0 + 8u;
    const int live0 = row0 < sequence, live1 = row1 < sequence;
    /* A row past the end reads the last real row and is never written, which
     * keeps the CTA's reads in bounds without branching the MMAs. */
    const uint32_t read0 = live0 ? row0 : sequence - 1u;
    const uint32_t read1 = live1 ? row1 : sequence - 1u;
    const float *query0 =
            query + ((size_t)head * sequence + read0) * H3D_VFA_HD;
    const float *query1 =
            query + ((size_t)head * sequence + read1) * H3D_VFA_HD;

    /* Q lives in registers as the four k16 A fragments, built once per CTA. */
    uint32_t a[H3D_VFA_KS][4];
#pragma unroll
    for (uint32_t step = 0; step < H3D_VFA_KS; step++) {
        const uint32_t dim = step * 16u + column;
        a[step][0] = h3d_mma_pair(query0[dim], query0[dim + 1u]);
        a[step][1] = h3d_mma_pair(query1[dim], query1[dim + 1u]);
        a[step][2] = h3d_mma_pair(query0[dim + 8u], query0[dim + 9u]);
        a[step][3] = h3d_mma_pair(query1[dim + 8u], query1[dim + 9u]);
    }

    float maximum0 = -INFINITY, maximum1 = -INFINITY;
    float denominator0 = 0.0f, denominator1 = 0.0f;
    float result[H3D_VFA_DIMS][4];
#pragma unroll
    for (uint32_t dim = 0; dim < H3D_VFA_DIMS; dim++) {
        result[dim][0] = result[dim][1] = result[dim][2] = result[dim][3] = 0.0f;
    }

    for (uint32_t base = 0; base < sequence; base += H3D_VFA_KEYS) {
        __syncthreads();  /* the previous tile's staged rows are fully consumed */
        /* Stage K and V: 16-byte copies, one row per eight lanes. A row past
         * the end is zeroed; its scores are masked below, so it contributes
         * nothing and the staging stays branch-free. */
        for (uint32_t index = threadIdx.x;
             index < H3D_VFA_KEYS * H3D_VFA_CHUNKS;
             index += H3D_VFA_WARPS * 32u) {
            const uint32_t row = index / H3D_VFA_CHUNKS;
            const uint32_t chunk = index % H3D_VFA_CHUNKS;
            const uint32_t source = base + row;
            if (source < sequence) {
                const size_t at =
                        ((size_t)head * sequence + source) * H3D_VFA_HD +
                        chunk * 8u;
                *(uint4 *)&shared_key[row][chunk * 8u] =
                        h3d_mma_bf16_chunk(key + at);
                *(uint4 *)&shared_value[row][chunk * 8u] =
                        h3d_mma_bf16_chunk(value + at);
            } else {
                const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
                *(uint4 *)&shared_key[row][chunk * 8u] = zero;
                *(uint4 *)&shared_value[row][chunk * 8u] = zero;
            }
        }
        __syncthreads();

        /* Scores: S = Q * K^T over all H3D_VFA_KEYS staged keys. */
        float score[H3D_VFA_NS][4];
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_VFA_NS; slice++) {
            score[slice][0] = score[slice][1] = 0.0f;
            score[slice][2] = score[slice][3] = 0.0f;
        }
#pragma unroll
        for (uint32_t step = 0; step < H3D_VFA_KS; step++) {
#pragma unroll
            for (uint32_t part = 0; part < H3D_VFA_KEYS / 16; part++) {
                const uint32_t row = part * 16u + (matrix >> 1u) * 8u + (lane & 7u);
                const uint32_t col = step * 16u + (matrix & 1u) * 8u;
                uint32_t b[4];
                h3d_mma_load(b, &shared_key[row][col]);
                h3d_mma_step(score[2u * part], a[step], b[0], b[1]);
                h3d_mma_step(score[2u * part + 1u], a[step], b[2], b[3]);
            }
        }

        /* Scale, then mask the tail keys out of the softmax. */
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_VFA_NS; slice++) {
            const uint32_t first = slice * 8u + column;
            score[slice][0] *= scale;
            score[slice][1] *= scale;
            score[slice][2] *= scale;
            score[slice][3] *= scale;
            if (base + first >= sequence) {
                score[slice][0] = -INFINITY;
                score[slice][2] = -INFINITY;
            }
            if (base + first + 1u >= sequence) {
                score[slice][1] = -INFINITY;
                score[slice][3] = -INFINITY;
            }
        }

        /* One maximum per tile, reduced across the four lanes of a row. */
        float tile0 = -INFINITY, tile1 = -INFINITY;
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_VFA_NS; slice++) {
            tile0 = fmaxf(tile0, fmaxf(score[slice][0], score[slice][1]));
            tile1 = fmaxf(tile1, fmaxf(score[slice][2], score[slice][3]));
        }
        tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 1));
        tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 2));
        tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 1));
        tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 2));
        const float next0 = fmaxf(maximum0, tile0);
        const float next1 = fmaxf(maximum1, tile1);
        const float rescale0 = expf(maximum0 - next0);
        const float rescale1 = expf(maximum1 - next1);

        float weight[H3D_VFA_NS][4];
        float part0 = 0.0f, part1 = 0.0f;
#pragma unroll
        for (uint32_t slice = 0; slice < H3D_VFA_NS; slice++) {
            weight[slice][0] = expf(score[slice][0] - next0);
            weight[slice][1] = expf(score[slice][1] - next0);
            weight[slice][2] = expf(score[slice][2] - next1);
            weight[slice][3] = expf(score[slice][3] - next1);
            part0 += weight[slice][0] + weight[slice][1];
            part1 += weight[slice][2] + weight[slice][3];
        }
        part0 += __shfl_xor_sync(0xffffffffu, part0, 1);
        part0 += __shfl_xor_sync(0xffffffffu, part0, 2);
        part1 += __shfl_xor_sync(0xffffffffu, part1, 1);
        part1 += __shfl_xor_sync(0xffffffffu, part1, 2);
        denominator0 = denominator0 * rescale0 + part0;
        denominator1 = denominator1 * rescale1 + part1;
        maximum0 = next0;
        maximum1 = next1;
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_VFA_DIMS; dim++) {
            result[dim][0] *= rescale0;
            result[dim][1] *= rescale0;
            result[dim][2] *= rescale1;
            result[dim][3] *= rescale1;
        }

        /* O += P * V, P packed from the weights as the A fragments. */
#pragma unroll
        for (uint32_t step = 0; step < H3D_VFA_KEYS / 16; step++) {
            uint32_t p[4];
            p[0] = h3d_mma_pair(weight[2u * step][0], weight[2u * step][1]);
            p[1] = h3d_mma_pair(weight[2u * step][2], weight[2u * step][3]);
            p[2] = h3d_mma_pair(weight[2u * step + 1u][0],
                                weight[2u * step + 1u][1]);
            p[3] = h3d_mma_pair(weight[2u * step + 1u][2],
                                weight[2u * step + 1u][3]);
#pragma unroll
            for (uint32_t part = 0; part < H3D_VFA_DIMS / 2; part++) {
                const uint32_t row =
                        step * 16u + (matrix & 1u) * 8u + (lane & 7u);
                const uint32_t col = part * 16u + (matrix >> 1u) * 8u;
                uint32_t b[4];
                h3d_mma_load_trans(b, &shared_value[row][col]);
                h3d_mma_step(result[2u * part], p, b[0], b[1]);
                h3d_mma_step(result[2u * part + 1u], p, b[2], b[3]);
            }
        }
    }

    __syncthreads();
    /* The output is row-major in F32, the layout the scalar dispatch writes. */
    if (live0) {
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_VFA_DIMS; dim++) {
            const size_t at = ((size_t)row0 * heads + head) * H3D_VFA_HD +
                              dim * 8u + column;
            output[at] = result[dim][0] / denominator0;
            output[at + 1u] = result[dim][1] / denominator0;
        }
    }
    if (live1) {
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_VFA_DIMS; dim++) {
            const size_t at = ((size_t)row1 * heads + head) * H3D_VFA_HD +
                              dim * 8u + column;
            output[at] = result[dim][2] / denominator1;
            output[at + 1u] = result[dim][3] / denominator1;
        }
    }
}

int h3d_sdpa_f32(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *query, const h3d_tensor *key,
                 const h3d_tensor *value, uint32_t sequence, uint32_t heads,
                 uint32_t head_dim, float scale, char *error,
                 size_t error_size) {
    size_t elements = 0;
    if (!context || !sequence || !heads || !head_dim || head_dim > 1024 ||
        !h3d_mul_size((size_t)sequence * heads, head_dim, &elements) ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(query, H3D_F32, elements) ||
        !h3d_is(key, H3D_F32, elements) ||
        !h3d_is(value, H3D_F32, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid attention tensors or shape");
    }
    if (head_dim == 64) {
        /* The tensor-core arm, off unless the decoder's options ask for it. */
        if (context->vae.attention_mma) {
            dim3 mma_grid((sequence + H3D_VFA_ROWS - 1) / H3D_VFA_ROWS, heads);
            h3d_attention_mma_f32_kernel<<<mma_grid, H3D_VFA_WARPS * 32u, 0,
                                           context->stream>>>(
                (float *)output->data, (const float *)query->data,
                (const float *)key->data, (const float *)value->data, sequence,
                heads, scale);
            return h3d_launch_ok("F32 attention (mma)", error, error_size);
        }
        const uint32_t row_blocks = (sequence + 8 * H3D_ATTENTION_F32_ROWS - 1) /
                                    (8 * H3D_ATTENTION_F32_ROWS);
        if (context->vae.query_major) {
            dim3 grid(row_blocks, heads);
            h3d_attention_tiled_f32_kernel<64, 1><<<grid, 256, 0,
                                                    context->stream>>>(
                (float *)output->data, (const float *)query->data,
                (const float *)key->data, (const float *)value->data, sequence,
                heads, scale, 0);
            return h3d_launch_ok("F32 attention", error, error_size);
        }
        dim3 grid(heads, row_blocks);
        h3d_attention_tiled_f32_kernel<64, 0><<<grid, 256, 0,
                                                context->stream>>>(
            (float *)output->data, (const float *)query->data,
            (const float *)key->data, (const float *)value->data, sequence,
            heads, scale, 0);
        return h3d_launch_ok("F32 attention", error, error_size);
    }
    uint32_t threads = 128;
    while (threads < head_dim) threads <<= 1;
    dim3 grid(heads, sequence);
    h3d_attention_f32_kernel<<<grid, threads, 0, context->stream>>>(
        (float *)output->data, (const float *)query->data,
        (const float *)key->data, (const float *)value->data, 1, sequence,
        heads, heads, head_dim, scale, 0, 0, 1);
    return h3d_launch_ok("F32 attention", error, error_size);
}

/* `h3_rms_norm_f32_kernel` (`:2438-2460`): one block per row, 256 threads,
 * fmaf square sum, tree reduction over shared memory, rsqrtf. */
__global__ static void h3d_rms_norm_f32_kernel(float *output,
                                               const float *input,
                                               const float *weight,
                                               uint32_t rows, uint32_t width,
                                               float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = input[(size_t)row * width + column];
        sum = fmaf(value, value, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        output[index] = input[index] * inverse * weight[column];
    }
}

int h3d_rms_norm_f32(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *input, const h3d_tensor *weight,
                     uint32_t rows, uint32_t width, float epsilon, char *error,
                     size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width ||
        !h3d_mul_size(rows, width, &elements) || epsilon < 0.0f ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(input, H3D_F32, elements) ||
        !h3d_is(weight, H3D_F32, width)) {
        return h3d_set_error(error, error_size,
                             "invalid F32 RMS norm arguments");
    }
    h3d_rms_norm_f32_kernel<<<rows, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data, rows, width, epsilon);
    return h3d_launch_ok("F32 RMS norm", error, error_size);
}

/* `h3_layer_norm_f32_kernel` (`:2529-2561`): the mean accumulates with plain
 * adds, the variance with fmaf; the epilogue is
 * `(input - mean) * inverse * weight + bias` in that order. */
__global__ static void h3d_layer_norm_f32_kernel(
        float *output, const float *input, const float *weight,
        const float *bias, uint32_t rows, uint32_t width, float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        sum += input[(size_t)row * width + column];
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = sums[0] / (float)width;
    sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float centered = input[(size_t)row * width + column] - mean;
        sum = fmaf(centered, centered, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        output[index] = (input[index] - mean) * inverse * weight[column] +
                        bias[column];
    }
}

int h3d_layer_norm_f32(h3d_context *context, h3d_tensor *output,
                       const h3d_tensor *input, const h3d_tensor *weight,
                       const h3d_tensor *bias, uint32_t rows, uint32_t width,
                       float epsilon, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !rows || !width ||
        !h3d_mul_size(rows, width, &elements) || epsilon < 0.0f ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(input, H3D_F32, elements) ||
        !h3d_is(weight, H3D_F32, width) ||
        !h3d_is(bias, H3D_F32, width)) {
        return h3d_set_error(error, error_size,
                             "invalid F32 layer norm arguments");
    }
    h3d_layer_norm_f32_kernel<<<rows, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data, (const float *)bias->data, rows, width,
        epsilon);
    return h3d_launch_ok("F32 layer norm", error, error_size);
}

/* `h3_swiglu_kernel<float>` (`:2841-2852`): the fused row holds gate then up,
 * and the gate passes through `gate / (1 + expf(-gate))`, not `1 / (1 + ...)`. */
__global__ static void h3d_swiglu_f32_kernel(float *output,
                                             const float *fused,
                                             const float *bias,
                                             uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    size_t base = (size_t)row * width * 2;
    float gate = fused[base + column];
    float up = fused[base + width + column];
    if (bias) {
        gate += bias[column];
        up += bias[width + column];
    }
    output[(size_t)row * width + column] =
        gate / (1.0f + expf(-gate)) * up;
}

int h3d_swiglu_f32(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *fused, const h3d_tensor *bias,
                   uint32_t rows, uint32_t width,
                   char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !h3d_mul_size(rows, width, &elements) ||
        elements > SIZE_MAX / 2 ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(fused, H3D_F32, elements * 2) ||
        (bias && !h3d_is(bias, H3D_F32, (size_t)width * 2))) {
        return h3d_set_error(error, error_size, "invalid SwiGLU arguments");
    }
    dim3 grid((width + 255) / 256, rows);
    h3d_swiglu_f32_kernel<<<grid, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)fused->data,
        bias ? (const float *)bias->data : NULL, rows, width);
    return h3d_launch_ok("F32 SwiGLU", error, error_size);
}

/* `h3_scale_add_kernel` (`:2885-2893`): `residual + branch * scale`, one
 * multiply, and no fma. `bias` is added to the branch on load, which is the
 * same F32 add the bias kernel would have made in place. */
__global__ static void h3d_scale_add_f32_kernel(float *output,
                                                const float *residual,
                                                const float *branch,
                                                const float *bias,
                                                const float *scale,
                                                uint32_t rows,
                                                uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    size_t index = (size_t)row * width + column;
    float value = branch[index];
    if (bias) value += bias[column];
    output[index] = residual[index] + value * scale[column];
}

int h3d_scale_add_f32(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *residual, const h3d_tensor *branch,
                      const h3d_tensor *bias, const h3d_tensor *scale,
                      uint32_t rows, uint32_t width,
                      char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(residual, H3D_F32, elements) ||
        !h3d_is(branch, H3D_F32, elements) ||
        !h3d_is(scale, H3D_F32, width) ||
        (bias && !h3d_is(bias, H3D_F32, width))) {
        return h3d_set_error(error, error_size,
                             "invalid F32 scale-add arguments");
    }
    dim3 grid((width + 255) / 256, rows);
    h3d_scale_add_f32_kernel<<<grid, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)residual->data,
        (const float *)branch->data, bias ? (const float *)bias->data : NULL,
        (const float *)scale->data, rows, width);
    return h3d_launch_ok("F32 scale add", error, error_size);
}

/* `h3_qkv_rope_kernel<float>` (`:3078-3136`) at `grouped = 1`,
 * `normalize = 1`, `weighted = 0` (`:3199-3203`): the checkpoint's
 * [head][q,k,v][dim] rows, the Q/K RMS over the head with no learned scale,
 * then the pair rotation over the first `2 * rope_half` dimensions. */
/* The QKV linear's bias, added on load: `h3d_linear_bias_kernel<float>`'s own
 * F32 add, one column per qkv element. `bias` is NULL when the caller already
 * added it. */
static __device__ inline float h3d_qkv_bias(const float *qkv, const float *bias,
                                            size_t index, size_t column) {
    return bias ? qkv[index] + bias[column] : qkv[index];
}

__global__ static void h3d_video_qkv_rope_f32_kernel(
        float *query, float *key, float *value, const float *qkv,
        const float *bias, const float *rope_cos, const float *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (row >= sequence || head >= heads || dimension >= head_dim) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t column_base = (size_t)head * head_dim * 3;
    size_t q_base = row_base + column_base;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float q = h3d_qkv_bias(qkv, bias, q_base + d, column_base + d);
        float k = h3d_qkv_bias(qkv, bias, k_base + d, column_base + head_dim + d);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = rsqrtf(q_sum / (float)head_dim + epsilon);
    float k_inverse = rsqrtf(k_sum / (float)head_dim + epsilon);

    float q0 = h3d_qkv_bias(qkv, bias, q_base + dimension,
                            column_base + dimension) * q_inverse;
    float k0 = h3d_qkv_bias(qkv, bias, k_base + dimension,
                            column_base + head_dim + dimension) * k_inverse;
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = h3d_qkv_bias(qkv, bias, q_base + pair,
                                column_base + pair) * q_inverse;
        float k1 = h3d_qkv_bias(qkv, bias, k_base + pair,
                                column_base + head_dim + pair) * k_inverse;
        float c = rope_cos[(size_t)row * rope_half + rope_index];
        float s = rope_sin[(size_t)row * rope_half + rope_index];
        if (dimension < rope_half) {
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else {
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
    }
    size_t output_index = ((size_t)head * sequence + row) * head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = h3d_qkv_bias(qkv, bias, v_base + dimension,
                                       column_base + 2 * head_dim + dimension);
}

int h3d_video_qkv_rope_f32(h3d_context *context, h3d_tensor *query,
                           h3d_tensor *key, h3d_tensor *value,
                           const h3d_tensor *qkv, const h3d_tensor *bias,
                           const h3d_tensor *rope_cos,
                           const h3d_tensor *rope_sin, uint32_t sequence,
                           uint32_t heads, uint32_t head_dim,
                           uint32_t rope_half, float epsilon, char *error,
                           size_t error_size) {
    size_t elements = 0;
    if (!context || !sequence || !heads || !head_dim ||
        !h3d_mul_size((size_t)sequence * heads, head_dim, &elements) ||
        rope_half > head_dim / 2 || epsilon < 0.0f ||
        !h3d_is(query, H3D_F32, elements) ||
        !h3d_is(key, H3D_F32, elements) ||
        !h3d_is(value, H3D_F32, elements) ||
        !h3d_is(qkv, H3D_F32, elements * 3) ||
        (bias && !h3d_is(bias, H3D_F32, (size_t)heads * head_dim * 3)) ||
        !h3d_is(rope_cos, H3D_F32, (size_t)sequence * rope_half) ||
        !h3d_is(rope_sin, H3D_F32, (size_t)sequence * rope_half)) {
        return h3d_set_error(error, error_size, "invalid QKV/RoPE tensors");
    }
    dim3 grid((head_dim + 127) / 128, heads, sequence);
    h3d_video_qkv_rope_f32_kernel<<<grid, 128, 0, context->stream>>>(
        (float *)query->data, (float *)key->data, (float *)value->data,
        (const float *)qkv->data, bias ? (const float *)bias->data : NULL,
        (const float *)rope_cos->data, (const float *)rope_sin->data, sequence,
        heads, head_dim, rope_half, epsilon);
    return h3d_launch_ok("F32 video QKV RoPE", error, error_size);
}

/* ---------------------------------------------------------------------------
 * F32 primitives for the VideoVAE encoder (`h3_video_encoder.c`). Same source
 * of truth, same expression order: the encoder's gate compares the latent hash
 * bit for bit against the reference, so the reduction tree and the padding
 * rule are not free choices.
 * ------------------------------------------------------------------------- */

static __device__ int h3d_reflect(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

/* `h3_vae_pad_kernel` (`h3_gpu_cuda.cu:1974-2000`). */
__global__ static void h3d_vae_pad_kernel(
    float *output, const float *input, uint32_t batch, uint32_t depth,
    uint32_t height, uint32_t width, uint32_t channels, uint32_t depth_front,
    uint32_t height_before, uint32_t height_after, uint32_t width_before,
    uint32_t width_after) {
    uint32_t channel = blockIdx.z * blockDim.x + threadIdx.x;
    uint32_t output_x = blockIdx.y;
    uint32_t plane = blockIdx.x;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    if (channel >= channels || output_x >= output_width ||
        plane >= batch * output_depth * output_height) return;
    uint32_t output_y = plane % output_height;
    uint32_t temporal = plane / output_height;
    uint32_t output_t = temporal % output_depth;
    uint32_t batch_index = temporal / output_depth;
    size_t destination = ((((size_t)batch_index * output_depth + output_t) *
        output_height + output_y) * output_width + output_x) * channels +
        channel;
    if (output_t < depth_front) { output[destination] = 0.0f; return; }
    int source_y = h3d_reflect((int)output_y - (int)height_before, (int)height);
    int source_x = h3d_reflect((int)output_x - (int)width_before, (int)width);
    uint32_t source_t = output_t - depth_front;
    size_t source = ((((size_t)batch_index * depth + source_t) * height +
        (uint32_t)source_y) * width + (uint32_t)source_x) * channels + channel;
    output[destination] = input[source];
}

int h3d_vae_encoder_pad(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, uint32_t batch,
                        uint32_t depth, uint32_t height, uint32_t width,
                        uint32_t channels, uint32_t depth_front,
                        uint32_t height_before, uint32_t height_after,
                        uint32_t width_before, uint32_t width_after,
                        char *error, size_t error_size) {
    if (!context || !batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) {
        return h3d_set_error(error, error_size, "invalid VAE padding shape");
    }
    uint64_t output_depth = (uint64_t)depth + depth_front;
    uint64_t output_height = (uint64_t)height + height_before + height_after;
    uint64_t output_width = (uint64_t)width + width_before + width_after;
    size_t input_elements, output_elements;
    if (output_depth > UINT32_MAX || output_height > UINT32_MAX ||
        output_width > UINT32_MAX ||
        !h3d_mul_size((size_t)batch * depth * height * width, channels,
                      &input_elements) ||
        !h3d_mul_size((size_t)batch * (size_t)output_depth *
                      (size_t)output_height * (size_t)output_width, channels,
                      &output_elements) ||
        !h3d_is(input, H3D_F32, input_elements) ||
        !h3d_is(output, H3D_F32, output_elements)) {
        return h3d_set_error(error, error_size, "invalid VAE padding tensors");
    }
    dim3 grid((uint32_t)((uint64_t)batch * output_depth * output_height),
              (uint32_t)output_width, (channels + 127) / 128);
    h3d_vae_pad_kernel<<<grid, 128, 0, context->stream>>>(
        (float *)output->data, (const float *)input->data, batch, depth,
        height, width, channels, depth_front, height_before, height_after,
        width_before, width_after);
    return h3d_launch_ok("VAE encoder padding", error, error_size);
}

/* `h3_conv3d_kernel` (`:2034-2073`). */
__global__ static void h3d_conv3d_kernel(
    float *output, const float *input, const float *weight, const float *bias,
    uint32_t depth, uint32_t height, uint32_t width, uint32_t input_channels,
    uint32_t output_channels, uint32_t kernel_depth, uint32_t kernel_height,
    uint32_t kernel_width, uint32_t stride_depth, uint32_t stride_height,
    uint32_t stride_width, uint32_t output_depth, uint32_t output_height,
    uint32_t output_width, size_t output_elements, int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    uint32_t output_channel = (uint32_t)(index % output_channels);
    size_t row = index / output_channels;
    uint32_t spatial_count = output_depth * output_height * output_width;
    uint32_t spatial = (uint32_t)(row % spatial_count);
    uint32_t batch_index = (uint32_t)(row / spatial_count);
    uint32_t output_x = spatial % output_width;
    uint32_t output_y = (spatial / output_width) % output_height;
    uint32_t output_t = spatial / (output_width * output_height);
    if (output_t >= output_depth) return;
    float sum = has_bias ? bias[output_channel] : 0.0f;
    for (uint32_t input_channel = 0; input_channel < input_channels;
         input_channel++)
        for (uint32_t kt = 0; kt < kernel_depth; kt++)
            for (uint32_t ky = 0; ky < kernel_height; ky++)
                for (uint32_t kx = 0; kx < kernel_width; kx++) {
                    uint32_t it = output_t * stride_depth + kt;
                    uint32_t iy = output_y * stride_height + ky;
                    uint32_t ix = output_x * stride_width + kx;
                    size_t input_index = ((((size_t)batch_index * depth + it) *
                        height + iy) * width + ix) * input_channels +
                        input_channel;
                    size_t weight_index = (((((size_t)output_channel *
                        input_channels + input_channel) * kernel_depth + kt) *
                        kernel_height + ky) * kernel_width + kx);
                    sum = fmaf(input[input_index], weight[weight_index], sum);
                }
    size_t destination = ((((size_t)batch_index * output_depth + output_t) *
        output_height + output_y) * output_width + output_x) * output_channels +
        output_channel;
    output[destination] = sum;
}

int h3d_conv3d(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t batch, uint32_t depth,
               uint32_t height, uint32_t width, uint32_t input_channels,
               uint32_t output_channels, uint32_t kernel_depth,
               uint32_t kernel_height, uint32_t kernel_width,
               uint32_t stride_depth, uint32_t stride_height,
               uint32_t stride_width, char *error, size_t error_size) {
    if (!context || !batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width ||
        depth < kernel_depth || height < kernel_height ||
        width < kernel_width) {
        return h3d_set_error(error, error_size, "invalid Conv3d shape");
    }
    uint32_t od = (depth - kernel_depth) / stride_depth + 1;
    uint32_t oh = (height - kernel_height) / stride_height + 1;
    uint32_t ow = (width - kernel_width) / stride_width + 1;
    size_t input_elements, weight_elements, output_elements;
    if (!h3d_mul_size((size_t)batch * depth * height * width, input_channels,
                      &input_elements) ||
        !h3d_mul_size((size_t)output_channels * input_channels *
                      kernel_depth * kernel_height, kernel_width,
                      &weight_elements) ||
        !h3d_mul_size((size_t)batch * od * oh * ow, output_channels,
                      &output_elements) ||
        !h3d_is(input, H3D_F32, input_elements) ||
        !h3d_is(weight, H3D_F32, weight_elements) ||
        !h3d_is(output, H3D_F32, output_elements) ||
        (bias && !h3d_is(bias, H3D_F32, output_channels))) {
        return h3d_set_error(error, error_size, "invalid Conv3d tensors");
    }
    h3d_conv3d_kernel<<<(output_elements + 255) / 256, 256, 0,
                        context->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data,
        bias ? (const float *)bias->data : (const float *)input->data,
        depth, height, width, input_channels, output_channels, kernel_depth,
        kernel_height, kernel_width, stride_depth, stride_height, stride_width,
        od, oh, ow, output_elements, bias != NULL);
    return h3d_launch_ok("Conv3d", error, error_size);
}

/* `h3_group_norm_silu_kernel` (`:2114-2151`). */
__global__ static void h3d_group_norm_silu_kernel(
    float *output, const float *input, const float *weight, const float *bias,
    uint32_t height, uint32_t width, uint32_t channels, uint32_t groups,
    uint32_t rows, float epsilon) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    uint32_t channels_per_group = channels / groups;
    uint32_t group = row % groups;
    uint32_t temporal = row / groups;
    uint32_t elements = height * width * channels_per_group;
    float mean = 0.0f;
    for (uint32_t index = 0; index < elements; index++) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group * channels_per_group + index % channels_per_group;
        mean += input[((size_t)temporal * height * width + spatial) * channels +
                      channel];
    }
    mean /= (float)elements;
    float variance = 0.0f;
    for (uint32_t index = 0; index < elements; index++) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group * channels_per_group + index % channels_per_group;
        float centered = input[((size_t)temporal * height * width + spatial) *
                               channels + channel] - mean;
        variance = fmaf(centered, centered, variance);
    }
    float inverse = rsqrtf(variance / (float)elements + epsilon);
    for (uint32_t index = 0; index < elements; index++) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group * channels_per_group + index % channels_per_group;
        size_t destination = ((size_t)temporal * height * width + spatial) *
                             channels + channel;
        float value = (input[destination] - mean) * inverse * weight[channel] +
                      bias[channel];
        output[destination] = value / (1.0f + expf(-value));
    }
}

int h3d_group_norm_silu(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        const h3d_tensor *bias, uint32_t batch,
                        uint32_t depth, uint32_t height, uint32_t width,
                        uint32_t channels, uint32_t groups, float epsilon,
                        char *error, size_t error_size) {
    size_t elements;
    uint64_t rows64 = (uint64_t)batch * depth * groups;
    if (!context || !batch || !depth || !height || !width || !channels ||
        !groups || channels % groups || !(epsilon > 0.0f) ||
        rows64 > UINT32_MAX ||
        !h3d_mul_size((size_t)batch * depth * height * width, channels,
                      &elements) ||
        !h3d_is(input, H3D_F32, elements) ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(weight, H3D_F32, channels) ||
        !h3d_is(bias, H3D_F32, channels)) {
        return h3d_set_error(error, error_size,
                             "invalid VAE group norm tensors or shape");
    }
    uint32_t rows = (uint32_t)rows64;
    h3d_group_norm_silu_kernel<<<(rows + 127) / 128, 128, 0,
                                 context->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data, (const float *)bias->data, height, width,
        channels, groups, rows, epsilon);
    return h3d_launch_ok("VAE group norm SiLU", error, error_size);
}

/* `h3_geglu_kernel` (`h3_gpu_cuda.cu:2912-2920`): tanh-approximated GELU of
 * the gate times the linear half, the same constant 0.7978845608028654 and
 * 0.044715 the reference uses. */
__global__ static void h3d_geglu_kernel(float *output, const float *gate,
                                        const float *linear, uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = gate[index];
    float gelu = 0.5f * value * (1.0f + tanhf(0.7978845608028654f *
        (value + 0.044715f * value * value * value)));
    output[index] = gelu * linear[index];
}

int h3d_geglu_f32(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *gate, const h3d_tensor *linear,
                  size_t elements, char *error, size_t error_size) {
    if (!context || !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(gate, H3D_F32, elements) ||
        !h3d_is(linear, H3D_F32, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid F32 GEGLU tensors");
    }
    if (!elements) {
        return 1;
    }
    h3d_geglu_kernel<<<(elements + 255) / 256, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)gate->data,
        (const float *)linear->data, (uint32_t)elements);
    return h3d_launch_ok("F32 GEGLU", error, error_size);
}

/* `h3_snake_kernel` (`h3_gpu_cuda.cu:2935-2943`): the plain Snake1d the
 * AudioVAE encoder uses, `x + sin(a*x)^2 / (a + 1e-9)`, with `a = alpha`
 * already exponentiated by the caller (`h3_audio_vae.c:839`, `:917`, `:965`).
 * Not the alias-free variant above; they are different ops. */
__global__ static void h3d_snake1d_kernel(float *output, const float *input,
                                          const float *alpha, size_t count,
                                          uint32_t channels) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float a = alpha[index % channels];
    float value = input[index];
    float wave = sinf(a * value);
    output[index] = value + wave * wave / (a + 1e-9f);
}

int h3d_snake1d_f32(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, const h3d_tensor *alpha,
                    uint32_t batch, uint32_t length, uint32_t channels,
                    char *error, size_t error_size) {
    size_t count;
    if (!context || !batch || !length || !channels ||
        !h3d_mul_size((size_t)batch * length, channels, &count) ||
        !h3d_is(output, H3D_F32, count) ||
        !h3d_is(input, H3D_F32, count) ||
        !h3d_is(alpha, H3D_F32, channels)) {
        return h3d_set_error(error, error_size,
                             "invalid F32 Snake1d tensors or shape");
    }
    if (!count) {
        return 1;
    }
    h3d_snake1d_kernel<<<(count + 255) / 256, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)alpha->data, count, channels);
    return h3d_launch_ok("F32 Snake1d", error, error_size);
}

/* `h3_audio_qkv_split_kernel` (`h3_gpu_cuda.cu:1829-1848`): q, k and v come
 * out of one interleaved `[batch, length, 3*width]` row, each with its own
 * bias, and land head-major `[batch, heads, length, head_dim]`. */
__global__ static void h3d_audio_qkv_split_kernel(
    float *query, float *key, float *value, const float *qkv,
    const float *q_bias, const float *k_bias, const float *v_bias,
    size_t count, uint32_t length, uint32_t heads, uint32_t head_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    uint32_t width = heads * head_dim;
    uint32_t column = (uint32_t)(index % width);
    size_t row = index / width;
    uint32_t time = (uint32_t)(row % length);
    uint32_t batch_index = (uint32_t)(row / length);
    uint32_t head = column / head_dim;
    uint32_t dimension = column % head_dim;
    size_t output = (((size_t)batch_index * heads + head) * length + time) *
                    head_dim + dimension;
    size_t base = row * width * 3;
    query[output] = qkv[base + column] + q_bias[column];
    key[output] = qkv[base + width + column] + k_bias[column];
    value[output] = qkv[base + width * 2 + column] + v_bias[column];
}

int h3d_audio_qkv_split_f32(h3d_context *context, h3d_tensor *query,
                            h3d_tensor *key, h3d_tensor *value,
                            const h3d_tensor *qkv, const h3d_tensor *q_bias,
                            const h3d_tensor *k_bias,
                            const h3d_tensor *v_bias, uint32_t batch,
                            uint32_t length, uint32_t heads,
                            uint32_t head_dim, char *error,
                            size_t error_size) {
    size_t width, count;
    if (!context || !batch || !length ||
        !h3d_mul_size(heads, head_dim, &width) || !width ||
        !h3d_mul_size((size_t)batch * length, width, &count) ||
        count > SIZE_MAX / 3 ||
        !h3d_is(qkv, H3D_F32, count * 3) ||
        !h3d_is(q_bias, H3D_F32, width) ||
        !h3d_is(k_bias, H3D_F32, width) ||
        !h3d_is(v_bias, H3D_F32, width) ||
        !h3d_is(query, H3D_F32, count) || !h3d_is(key, H3D_F32, count) ||
        !h3d_is(value, H3D_F32, count)) {
        return h3d_set_error(error, error_size,
                             "invalid audio QKV tensors or shape");
    }
    h3d_audio_qkv_split_kernel<<<(count + 255) / 256, 256, 0,
                                 context->stream>>>(
        (float *)query->data, (float *)key->data, (float *)value->data,
        (const float *)qkv->data, (const float *)q_bias->data,
        (const float *)k_bias->data, (const float *)v_bias->data, count,
        length, heads, head_dim);
    return h3d_launch_ok("audio QKV split", error, error_size);
}

/* `h3_audio_pool_kernel` (`h3_gpu_cuda.cu:1874-1893`): one output value per
 * (row, output channel) sums the matching slice of every head and divides by
 * `heads * pool`; the reference's accumulation order is head-major. */
__global__ static void h3d_audio_pool_kernel(float *output,
                                             const float *attended,
                                             size_t count, uint32_t heads,
                                             uint32_t head_dim,
                                             uint32_t output_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    uint32_t column = (uint32_t)(index % output_dim);
    size_t row = index / output_dim;
    uint32_t pool = head_dim / output_dim;
    float sum = 0.0f;
    for (uint32_t head = 0; head < heads; head++) {
        size_t base = (row * heads + head) * head_dim + column * pool;
        for (uint32_t item = 0; item < pool; item++) {
            sum += attended[base + item];
        }
    }
    output[index] = sum / (float)(heads * pool);
}

int h3d_audio_attention_pool_f32(h3d_context *context, h3d_tensor *output,
                                 const h3d_tensor *attended, uint32_t batch,
                                 uint32_t length, uint32_t heads,
                                 uint32_t head_dim, uint32_t output_dim,
                                 char *error, size_t error_size) {
    size_t input_elements, output_elements;
    if (!context || !batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim ||
        !h3d_mul_size((size_t)batch * length * heads, head_dim,
                      &input_elements) ||
        !h3d_mul_size((size_t)batch * length, output_dim,
                      &output_elements) ||
        !h3d_is(attended, H3D_F32, input_elements) ||
        !h3d_is(output, H3D_F32, output_elements)) {
        return h3d_set_error(error, error_size,
                             "invalid audio attention pool tensors");
    }
    h3d_audio_pool_kernel<<<(output_elements + 255) / 256, 256, 0,
                            context->stream>>>(
        (float *)output->data, (const float *)attended->data,
        output_elements, heads, head_dim, output_dim);
    return h3d_launch_ok("audio attention pool", error, error_size);
}

/* `h3_attention_kernel<float>` (`h3_gpu_cuda.cu:1337-1419`) with
 * `causal = 1`, `head_major_input = 1`, `head_major_output = 0`, as
 * `h3_gpu_sdpa_causal_f32` (`:1669-1676`) dispatches it: one block per
 * (head, row, batch), 128 threads, a tree reduction for the dot product, the
 * reference's running-maximum softmax, and the output written
 * `[batch, length, heads, head_dim]`. */
__global__ static void h3d_sdpa_causal_kernel(
    float *output, const float *query, const float *key, const float *value,
    uint32_t batch, uint32_t sequence, uint32_t heads, uint32_t head_dim,
    float scale) {
    __shared__ float partial[1024];
    __shared__ float rescale;
    __shared__ float weight;
    __shared__ float denominator;
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y;
    uint32_t batch_index = blockIdx.z;
    uint32_t keys = row + 1;
    size_t query_base =
        (((size_t)batch_index * heads + head) * sequence + row) * head_dim;
    uint32_t dimension = threadIdx.x;
    float result = 0.0f;
    float maximum = -INFINITY;
    if (threadIdx.x == 0) {
        denominator = 0.0f;
    }
    __syncthreads();
    for (uint32_t key_row = 0; key_row < keys; key_row++) {
        size_t key_base =
            (((size_t)batch_index * heads + head) * sequence + key_row) *
            head_dim;
        float product = 0.0f;
        if (dimension < head_dim) {
            product = query[query_base + dimension] * key[key_base + dimension];
        }
        partial[threadIdx.x] = product;
        __syncthreads();
        for (uint32_t offset = blockDim.x / 2; offset; offset >>= 1) {
            if (threadIdx.x < offset) {
                partial[threadIdx.x] += partial[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = partial[0] * scale;
            float next_maximum = fmaxf(maximum, score);
            rescale = expf(maximum - next_maximum);
            weight = expf(score - next_maximum);
            denominator = denominator * rescale + weight;
            maximum = next_maximum;
        }
        __syncthreads();
        if (dimension < head_dim) {
            result = result * rescale + weight * value[key_base + dimension];
        }
        __syncthreads();
    }
    if (dimension < head_dim) {
        result /= denominator;
        size_t output_index =
            (((size_t)batch_index * sequence + row) * heads + head) * head_dim +
            dimension;
        output[output_index] = result;
    }
}

int h3d_sdpa_causal_f32(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *query, const h3d_tensor *key,
                        const h3d_tensor *value, uint32_t batch,
                        uint32_t sequence, uint32_t heads, uint32_t head_dim,
                        float scale, char *error, size_t error_size) {
    size_t elements;
    if (!context || !batch || !sequence || !heads || !head_dim ||
        head_dim > 1024 ||
        !h3d_mul_size((size_t)batch * sequence * heads, head_dim,
                      &elements) ||
        !h3d_is(query, H3D_F32, elements) || !h3d_is(key, H3D_F32, elements) ||
        !h3d_is(value, H3D_F32, elements) ||
        !h3d_is(output, H3D_F32, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid F32 causal attention tensors or shape");
    }
    uint32_t threads = 128;
    while (threads < head_dim) {
        threads <<= 1;
    }
    dim3 grid(heads, sequence, batch);
    h3d_sdpa_causal_kernel<<<grid, threads, 0, context->stream>>>(
        (float *)output->data, (const float *)query->data,
        (const float *)key->data, (const float *)value->data, batch, sequence,
        heads, head_dim, scale);
    return h3d_launch_ok("F32 causal attention", error, error_size);
}

/* ---------------------------------------------------------------------------
 * Sol-Attn's search step: the routing decision that picks which KV blocks
 * attend exactly and which are approximated from their summaries.
 *
 * Port of NVIDIA Sana `sol-engine` @ `6e4a440`,
 * `techniques/sparse_backends/sol_attn/`. The reference's rule, in order:
 *
 *   kc[h,n,:] = sum over the block's 64 rows of k / block_len        (BF16)
 *   vc[h,n,:] = sum over the block's rows of v                       (BF16)
 *     `preprocess.py:280-322`, `triton_ref/preprocess.py:34-68`
 *   kc_mean[h,:] = mean over blocks of kc; kc_var[h,:] = its population
 *   variance; both read the BF16 kc                                  (`:96-135`)
 *   threshold[q,h] = q_bar . kc_mean * log2_scale
 *                  + tau * sqrt(max(q_bar^2 . kc_var * log2_scale^2, 0) + 1e-6)
 *   with q_bar the query block's mean row and log2_scale = scale*log2(e)
 *                                                                    (`:143-186`)
 *   col_mean[n] = column mean over the query block of (q @ kc[n]^T) * log2_scale
 *                                                                    (`fwd.py:250-266`)
 *   exact = valid and (col_mean[n] > threshold[q] or |q - n| <= 1), the sink
 *   block range forced exact                              (`selector.py:132-142`,
 *                                                     `interface.py:134-142`)
 *   count[q] = that query block's exact blocks, sinks included
 *
 * Against the reference's column mean, which sums 64 row scores:
 *   sum_q (q.kc) / len = (sum_q q / len).kc = q_bar . kc
 * so this port routes from `q_bar` directly: the same number with 64x fewer
 * multiplies. It also keeps the comparison's two sides on one centroid, which
 * the reference's raw-row mean does not: `threshold`'s mean term is
 * `q_bar . kc_mean` and the route's column mean is `q_bar . kc[n]`, both in
 * F32 from the same `q_bar`. The reference's own order stays as the oracle in
 * `tests/h3/test_native.cu`, and `tests/h3/sol_route_bm` reports what the
 * substitution moves.
 *
 * Layout deviation, named: head-major `[heads][sequence][128]` with batch 1,
 * the port's layout for every H3 attention primitive, against the reference's
 * `[batch, tokens, heads, 128]`.
 * ------------------------------------------------------------------------- */

enum {
    H3D_SOL_THREADS = 128,       /* pooling and statistics kernel width */
    H3D_SOL_WARP = 32,           /* route kernel: one lane per mask bit */
    H3D_SOL_QM = 64,             /* query blocks one route CTA covers */
    H3D_SOL_KLD = H3D_SOL_HEAD_DIM + 8,  /* +8 bf16 keeps kc rows off a bank */
    H3D_SOL_QLD = H3D_SOL_HEAD_DIM + 4,  /* +4 f32 ditto for the centroids */
};

uint32_t h3d_sol_words(uint32_t sequence) {
    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    return (blocks + H3D_SOL_GROUP - 1) / H3D_SOL_GROUP;
}

/* The block summaries: K's block mean and V's block sum. One CTA per (block,
 * head), one thread per dimension, sequential over the block's rows so the sum
 * order is the reference's row order. */
__global__ static void h3d_sol_pool_kernel(
        __nv_bfloat16 *kc, __nv_bfloat16 *vc, const __nv_bfloat16 *key,
        const __nv_bfloat16 *value, uint32_t sequence, uint32_t blocks) {
    const uint32_t head = blockIdx.y;
    const uint32_t block = blockIdx.x;
    const uint32_t column = threadIdx.x;
    if (column >= H3D_SOL_HEAD_DIM) return;
    const uint32_t first = block * H3D_SOL_BLOCK;
    uint32_t rows = sequence - first;
    if (rows > H3D_SOL_BLOCK) rows = H3D_SOL_BLOCK;
    const size_t base = ((size_t)head * sequence + first) * H3D_SOL_HEAD_DIM;
    float key_sum = 0.0f;
    float value_sum = 0.0f;
    for (uint32_t row = 0; row < rows; ++row) {
        const size_t at = base + (size_t)row * H3D_SOL_HEAD_DIM + column;
        key_sum += h3d_value(key[at]);
        value_sum += h3d_value(value[at]);
    }
    const size_t out =
        ((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM + column;
    kc[out] = h3d_store(key_sum / (float)rows);  /* `preprocess.py:64` */
    vc[out] = h3d_store(value_sum);              /* `:66`: a sum, no divide */
}

/* The per-head statistics of kc: its mean over blocks and the diagonal of its
 * variance, read from the BF16 summaries (`preprocess.py:96-135`). */
__global__ static void h3d_sol_stats_kernel(float *kc_mean, float *kc_var,
                                            const __nv_bfloat16 *kc,
                                            uint32_t blocks) {
    const uint32_t head = blockIdx.x;
    const uint32_t dim = threadIdx.x;
    if (dim >= H3D_SOL_HEAD_DIM) return;
    float total = 0.0f;
    float total_sq = 0.0f;
    for (uint32_t block = 0; block < blocks; ++block) {
        const float value =
            h3d_value(kc[((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM +
                         dim]);
        total += value;
        total_sq += value * value;
    }
    const float count = (float)blocks;
    const float mean = total / count;
    float variance = total_sq / count - mean * mean;
    if (variance < 0.0f) variance = 0.0f;   /* `:124` tl.maximum(..., 0.0) */
    kc_mean[(size_t)head * H3D_SOL_HEAD_DIM + dim] = mean;
    kc_var[(size_t)head * H3D_SOL_HEAD_DIM + dim] = variance;
}

/* The routing threshold and the query centroid it is built from: one warp per
 * (query block, head). The centroid is the reference's `_pool_query_kernel`
 * (`preprocess.py:189-213`) kept F32. */
__global__ static void h3d_sol_threshold_kernel(
        float *threshold, float *query_bar, const float *kc_mean,
        const float *kc_var, const __nv_bfloat16 *query, uint32_t sequence,
        uint32_t heads, float scale, float tau) {
    enum { CHUNKS = H3D_SOL_HEAD_DIM / H3D_SOL_WARP };
    const uint32_t head = blockIdx.y;
    const uint32_t q_block = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t first = q_block * H3D_SOL_BLOCK;
    uint32_t rows = sequence - first;
    if (rows > H3D_SOL_BLOCK) rows = H3D_SOL_BLOCK;
    const size_t base = ((size_t)head * sequence + first) * H3D_SOL_HEAD_DIM;
    float centroid[CHUNKS];
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        const uint32_t dim = chunk * H3D_SOL_WARP + lane;
        float total = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            total += h3d_value(query[base + (size_t)row * H3D_SOL_HEAD_DIM + dim]);
        }
        centroid[chunk] = total / (float)rows;
        query_bar[((size_t)q_block * heads + head) * H3D_SOL_HEAD_DIM + dim] =
            centroid[chunk];
    }
    /* `:177-186`: the diagonal model's mean and variance of the route logits,
     * then mean + tau * std in log2 space. */
    const float log2_scale = scale * 1.4426950408889634f;
    float mean = 0.0f;
    float variance = 0.0f;
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        const uint32_t dim = chunk * H3D_SOL_WARP + lane;
        const float m = kc_mean[(size_t)head * H3D_SOL_HEAD_DIM + dim];
        const float v = kc_var[(size_t)head * H3D_SOL_HEAD_DIM + dim];
        mean += centroid[chunk] * m;
        variance += centroid[chunk] * centroid[chunk] * v;
    }
    for (int offset = H3D_SOL_WARP / 2; offset > 0; offset >>= 1) {
        mean += __shfl_xor_sync(0xffffffffu, mean, offset);
        variance += __shfl_xor_sync(0xffffffffu, variance, offset);
    }
    if (lane != 0) return;
    /* `:179-182`: both moments are scaled before the standard deviation, so
     * the 1e-6 floor is added in scaled units. */
    variance *= log2_scale * log2_scale;
    const float clamped = variance > 0.0f ? variance : 0.0f;
    const float std = sqrtf(clamped + 1.0e-6f);
    threshold[(size_t)q_block * heads + head] =
        mean * log2_scale + tau * std;
}

/* The route. One CTA per (tile of H3D_SOL_QM query blocks, head), one lane per
 * KV block of the current mask word, the word's kc rows staged in shared so a
 * CTA reads each summary once however many query blocks it covers. */
__global__ static void h3d_sol_route_kernel(
        uint32_t *mask, uint32_t *count, const __nv_bfloat16 *kc,
        const float *query_bar, const float *threshold, uint32_t heads,
        uint32_t blocks, uint32_t words, float scale, uint32_t sink_first,
        uint32_t sink_last) {
    const uint32_t q_first = blockIdx.x * H3D_SOL_QM;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    uint32_t rows = blocks - q_first;        /* query blocks in this tile */
    if (rows > H3D_SOL_QM) rows = H3D_SOL_QM;
    __shared__ __align__(16) float s_query[H3D_SOL_QM][H3D_SOL_QLD];
    __shared__ __align__(16) __nv_bfloat16 s_kc[H3D_SOL_GROUP][H3D_SOL_KLD];
    __shared__ uint32_t s_count[H3D_SOL_QM];
    const float log2_scale = scale * 1.4426950408889634f;

    /* Stage the query centroids one row at a time, all lanes on the same row
     * and lane `lane` taking every 32nd dimension: any other mapping leaves
     * the positions of a row that the staging lane does not own uninitialized. */
    for (uint32_t row = 0; row < rows; ++row) {
        const float *centroid = query_bar +
            ((size_t)(q_first + row) * heads + head) * H3D_SOL_HEAD_DIM;
        for (uint32_t dim = lane; dim < H3D_SOL_HEAD_DIM; dim += H3D_SOL_WARP) {
            s_query[row][dim] = centroid[dim];
        }
        if (lane == 0) s_count[row] = 0;
    }
    __syncwarp();

    for (uint32_t word = 0; word < words; ++word) {
        /* Stage one word of kc: 32 blocks of 128 dimensions, zero-filled past
         * the last block so the lane's dot is defined. */
        enum { CHUNKS = H3D_SOL_GROUP * H3D_SOL_HEAD_DIM / 8 };  /* uint4 */
        for (uint32_t index = lane; index < CHUNKS; index += H3D_SOL_WARP) {
            const uint32_t block =
                word * H3D_SOL_GROUP + index / (H3D_SOL_HEAD_DIM / 8);
            const uint32_t chunk = index % (H3D_SOL_HEAD_DIM / 8);
            uint4 packed = make_uint4(0u, 0u, 0u, 0u);
            if (block < blocks) {
                packed = *(const uint4 *)((const char *)kc +
                    ((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM * 2 +
                    chunk * 16);
            }
            *(uint4 *)((char *)s_kc[index / (H3D_SOL_HEAD_DIM / 8)] +
                       chunk * 16) = packed;
        }
        __syncwarp();
        const uint32_t block = word * H3D_SOL_GROUP + lane;
        const bool sink = sink_last > sink_first && block >= sink_first &&
                          block < sink_last;
        for (uint32_t row = 0; row < rows; ++row) {
            /* Four accumulators: the two halves of the dimension range do not
             * wait on one another's FMA latency. */
            float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t chunk = 0; chunk < 4; ++chunk) {
                for (uint32_t step = 0; step < H3D_SOL_WARP; ++step) {
                    const uint32_t dim = chunk * H3D_SOL_WARP + step;
                    part[chunk] +=
                        s_query[row][dim] * h3d_value(s_kc[lane][dim]);
                }
            }
            const float column_mean =
                ((part[0] + part[1]) + (part[2] + part[3])) * log2_scale;
            const uint32_t q_block = q_first + row;
            const uint32_t distance =
                q_block > block ? q_block - block : block - q_block;
            bool exact = false;
            if (block < blocks) {
                exact = distance <= 1 || sink ||
                        column_mean >
                            threshold[(size_t)q_block * heads + head];
            }
            const uint32_t bits = __ballot_sync(0xffffffffu, exact);
            if (lane == 0) {
                mask[((size_t)q_block * heads + head) * words + word] = bits;
                s_count[row] += __popc(bits);
            }
        }
        __syncwarp();
    }
    for (uint32_t row = lane; row < rows; row += H3D_SOL_WARP) {
        count[(size_t)(q_first + row) * heads + head] = s_count[row];
    }
}

/* Every plan buffer's shape, checked before a kernel writes anything. */
static int h3d_sol_check(const struct h3d_sol_plan *plan, uint32_t sequence,
                         uint32_t heads, char *error, size_t error_size) {
    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    const uint32_t words = h3d_sol_words(sequence);
    /* Reachable only after `h3d_sol_search` has bounded `sequence x heads`, but
     * the products are checked here too so the function stands on its own. */
    size_t summary = 0, threshold = 0, words_ = 0;
    if (!h3d_mul_size(blocks, heads, &summary) ||
        !h3d_mul_size(summary, H3D_SOL_HEAD_DIM, &summary) ||
        !h3d_mul_size(blocks, heads, &threshold) ||
        !h3d_mul_size(threshold, words, &words_)) {
        return h3d_set_error(error, error_size,
                             "invalid Sol-Attn search tensors or shape");
    }
    if (!plan || !h3d_is(plan->kc, H3D_BF16, summary) ||
        !h3d_is(plan->vc, H3D_BF16, summary) ||
        !h3d_is(plan->mask, H3D_U32, words_) ||
        !h3d_is(plan->count, H3D_U32, threshold) ||
        !h3d_is(plan->threshold, H3D_F32, threshold) ||
        !h3d_is(plan->query_bar, H3D_F32, summary) ||
        !h3d_is(plan->kc_mean, H3D_F32,
                (size_t)heads * H3D_SOL_HEAD_DIM) ||
        !h3d_is(plan->kc_var, H3D_F32, (size_t)heads * H3D_SOL_HEAD_DIM)) {
        return h3d_set_error(error, error_size,
                             "invalid Sol-Attn search tensors or shape");
    }
    return 1;
}

int h3d_sol_search(h3d_context *context, const struct h3d_sol_plan *plan,
                   const h3d_tensor *query, const h3d_tensor *key,
                   const h3d_tensor *value, uint32_t sequence, uint32_t heads,
                   float scale, float tau, uint32_t sink_start,
                   uint32_t sink_tokens, char *error, size_t error_size) {
    /* The reference builds `[sink_start, sink_start + sink_tokens)` inside the
     * sequence and rejects nothing (`interface.py:134-142`), so an out-of-range
     * pair is refused here. The bound is written as a difference: the sum wraps
     * for a near-UINT32_MAX start (the idiom at `:556-560`). */
    if (!context || !sequence || !heads || sink_tokens > sequence ||
        sink_start > sequence - sink_tokens) {
        return h3d_set_error(error, error_size,
                             "invalid Sol-Attn search arguments");
    }
    /* `sequence + BLOCK - 1` wraps above UINT32_MAX - 63, and the element count
     * overflows size_t for large arguments; both are refused before a launch
     * (`h3d_mul_size`, `:716-720`). */
    size_t elements = 0;
    if (sequence > UINT32_MAX - (H3D_SOL_BLOCK - 1) ||
        !h3d_mul_size(sequence, heads, &elements) ||
        !h3d_mul_size(elements, H3D_SOL_HEAD_DIM, &elements)) {
        return h3d_set_error(error, error_size, "invalid Sol-Attn search shape");
    }
    if (!h3d_is(query, H3D_BF16, elements) || !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(value, H3D_BF16, elements) ||
        !h3d_sol_check(plan, sequence, heads, error, error_size)) {
        return 0;
    }
    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    const uint32_t words = h3d_sol_words(sequence);
    dim3 pool_grid(blocks, heads);
    h3d_sol_pool_kernel<<<pool_grid, H3D_SOL_THREADS, 0, context->stream>>>(
        (__nv_bfloat16 *)plan->kc->data, (__nv_bfloat16 *)plan->vc->data,
        (const __nv_bfloat16 *)key->data,
        (const __nv_bfloat16 *)value->data, sequence, blocks);
    if (!h3d_launch_ok("Sol-Attn pool", error, error_size)) return 0;
    h3d_sol_stats_kernel<<<heads, H3D_SOL_THREADS, 0, context->stream>>>(
        (float *)plan->kc_mean->data, (float *)plan->kc_var->data,
        (const __nv_bfloat16 *)plan->kc->data, blocks);
    if (!h3d_launch_ok("Sol-Attn statistics", error, error_size)) return 0;
    dim3 threshold_grid(blocks, heads);
    h3d_sol_threshold_kernel<<<threshold_grid, H3D_SOL_WARP, 0,
                               context->stream>>>(
        (float *)plan->threshold->data, (float *)plan->query_bar->data,
        (const float *)plan->kc_mean->data,
        (const float *)plan->kc_var->data,
        (const __nv_bfloat16 *)query->data, sequence, heads, scale, tau);
    if (!h3d_launch_ok("Sol-Attn threshold", error, error_size)) return 0;
    /* `_sink_block_range` (`interface.py:134-142`) returns an empty range when
     * `sink_tokens` is zero: `[blocks, blocks)`, so no block is forced exact.
     * Rounding the token range up without that case gives `[0, 1)` for any
     * `sink_start` below one block, which forces block 0 for every query
     * block - a decision the reference never makes. */
    const uint32_t sink_first = sink_tokens ? sink_start / H3D_SOL_BLOCK : blocks;
    const uint32_t sink_last = sink_tokens
        ? (sink_start + sink_tokens + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK
        : blocks;
    dim3 route_grid((blocks + H3D_SOL_QM - 1) / H3D_SOL_QM, heads);
    h3d_sol_route_kernel<<<route_grid, H3D_SOL_WARP, 0, context->stream>>>(
        (uint32_t *)plan->mask->data, (uint32_t *)plan->count->data,
        (const __nv_bfloat16 *)plan->kc->data,
        (const float *)plan->query_bar->data,
        (const float *)plan->threshold->data, heads, blocks, words,
        scale, sink_first, sink_last);
    return h3d_launch_ok("Sol-Attn route", error, error_size);
}

/* VSA's selection: the rule the reference's H3 stage runs, against Sol-Attn's
 * tau threshold above, which is its LTX refiner's. Read from the FastVideo
 * module the reference hashes and refuses to run without
 * (`fastvideo/attention/backends/video_sparse_attn_h3.py`, sha256
 * `b7c6d4e4...dc0fab9` = the pinned `VSA_SOURCE_SHA`):
 *
 *   scores = q_pool @ k_pool^T / sqrt(head_dim)                     (`:629`)
 *   k_vid  = max(1, min(ceil((1 - sparsity) * video_tiles), video_tiles))
 *                                            (`video_sparse_attn.py:161-163`)
 *   in `exempt` mode the video columns compete for the k_vid largest scores,
 *   every prefix column is selected, and every prefix query row is dense; a
 *   sparsity of 0 selects everything                          (`:387-410`)
 *
 * The `[heads][blocks][blocks]` score matrix the reference materialises never
 * exists here: one CTA per (query block, head) computes each video column's
 * score from `query_bar` and the `kc` row and keeps it in shared, then ranks
 * every column against the kept set. Deviations, named: `torch.topk`'s tie
 * order is undocumented, so ties go to the lower column index; `num_video_tiles`
 * is this port's count of video blocks, so a short trailing block counts as one
 * tile where the reference pads; and the rank pass compares all four moments of
 * the block count where `topk` selects, at the cost measured by
 * `tests/h3/sol_vsa_bm`. */

enum {
    /* The score and centroid buffers are dynamic shared, so the sequence bound
     * is the 48 KiB static limit: (128 + blocks) floats. 12,160 blocks is
     * 778,240 tokens, well past the largest H3 canvas (32,256 tokens). */
    H3D_SOL_VSA_MAX_BLOCKS = 12288 - H3D_SOL_HEAD_DIM,
};

__global__ static void h3d_sol_vsa_kernel(
        uint32_t *mask, uint32_t *count, const float *query_bar,
        const __nv_bfloat16 *kc, uint32_t heads, uint32_t blocks,
        uint32_t words, uint32_t prefix, uint32_t keep, float scale) {
    extern __shared__ __align__(16) float shared[];
    float *s_query = shared;                        /* [H3D_SOL_HEAD_DIM] */
    float *s_score = shared + H3D_SOL_HEAD_DIM;     /* [blocks] */
    const uint32_t q_block = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;

    for (uint32_t dim = lane; dim < H3D_SOL_HEAD_DIM; dim += H3D_SOL_WARP) {
        s_query[dim] =
            query_bar[((size_t)q_block * heads + head) * H3D_SOL_HEAD_DIM + dim];
    }
    __syncwarp();

    /* One `kc` row at a time, the whole warp on it: 128 dimensions over 32
     * lanes is four values each, and the row is a single 256-byte read. Two
     * columns are in flight, so the reduction of one overlaps the load of the
     * next; each column's own order is unchanged. */
    const __nv_bfloat16 *rows = kc + (size_t)head * blocks * H3D_SOL_HEAD_DIM;
    for (uint32_t block = 0; block + 1 < blocks; block += 2) {
        float part[2] = {0.0f, 0.0f};
        for (uint32_t dim = lane; dim < H3D_SOL_HEAD_DIM; dim += H3D_SOL_WARP) {
            part[0] += s_query[dim] *
                       h3d_value(rows[(size_t)block * H3D_SOL_HEAD_DIM + dim]);
            part[1] += s_query[dim] *
                       h3d_value(rows[(size_t)(block + 1) * H3D_SOL_HEAD_DIM + dim]);
        }
        for (int offset = H3D_SOL_WARP / 2; offset > 0; offset >>= 1) {
            part[0] += __shfl_xor_sync(0xffffffffu, part[0], offset);
            part[1] += __shfl_xor_sync(0xffffffffu, part[1], offset);
        }
        if (lane == 0) {
            s_score[block] = part[0] * scale;
            s_score[block + 1] = part[1] * scale;
        }
    }
    if (blocks & 1u) {
        const uint32_t block = blocks - 1;
        float part = 0.0f;
        for (uint32_t dim = lane; dim < H3D_SOL_HEAD_DIM; dim += H3D_SOL_WARP) {
            part += s_query[dim] *
                    h3d_value(rows[(size_t)block * H3D_SOL_HEAD_DIM + dim]);
        }
        for (int offset = H3D_SOL_WARP / 2; offset > 0; offset >>= 1) {
            part += __shfl_xor_sync(0xffffffffu, part, offset);
        }
        if (lane == 0) s_score[block] = part * scale;
    }
    __syncwarp();

    /* A word is 32 consecutive blocks, so a lane owns one column of it and the
     * ballot is the word: bit set means that block attends exactly, and the
     * bits past the last block stay clear. The rank is a sum of four
     * independent counts, so the chain of adds is not what the lane waits on. */
    const bool prefix_row = q_block < prefix;
    uint32_t total = 0;
    for (uint32_t word = 0; word < words; ++word) {
        const uint32_t block = word * H3D_SOL_GROUP + lane;
        bool exact = block < blocks;
        if (exact && !prefix_row && block >= prefix) {
            const float mine = s_score[block];
            uint32_t rank[4] = {0u, 0u, 0u, 0u};
            uint32_t other = prefix;
            for (; other + 4 <= blocks; other += 4) {
                const float s0 = s_score[other];
                const float s1 = s_score[other + 1];
                const float s2 = s_score[other + 2];
                const float s3 = s_score[other + 3];
                rank[0] += s0 > mine || (s0 == mine && other < block);
                rank[1] += s1 > mine || (s1 == mine && other + 1 < block);
                rank[2] += s2 > mine || (s2 == mine && other + 2 < block);
                rank[3] += s3 > mine || (s3 == mine && other + 3 < block);
            }
            for (; other < blocks; ++other) {
                const float theirs = s_score[other];
                rank[0] += theirs > mine || (theirs == mine && other < block);
            }
            const uint32_t summed = (rank[0] + rank[1]) + (rank[2] + rank[3]);
            exact = summed < keep;
        }
        const uint32_t bits = __ballot_sync(0xffffffffu, exact);
        if (lane == 0) {
            mask[((size_t)q_block * heads + head) * words + word] = bits;
            total += (uint32_t)__popc(bits);
        }
    }
    if (lane == 0) count[(size_t)q_block * heads + head] = total;
}

/* `compute_topk` (`video_sparse_attn.py:161-163`) in the reference's own double
 * precision, so a sparsity that lands on a boundary keeps its count. The clamp
 * is the reference's own: a negative sparsity keeps every tile, one above the
 * range keeps a single one. */
static uint32_t h3d_sol_keep(double sparsity, uint32_t tiles) {
    const double keep = ceil((1.0 - sparsity) * (double)tiles);
    if (keep < 1.0) return 1u;
    if (keep > (double)tiles) return tiles;
    return (uint32_t)keep;
}

int h3d_sol_vsa(h3d_context *context, const struct h3d_sol_plan *plan,
                const h3d_tensor *query, const h3d_tensor *key,
                const h3d_tensor *value, uint32_t sequence, uint32_t heads,
                float scale, double sparsity, uint32_t prefix_blocks,
                char *error, size_t error_size) {
    /* A NaN or infinite sparsity has no keep count in the reference either:
     * `ceil` propagates it into a `topk` that cannot be asked for. */
    if (!context || !sequence || !heads || !isfinite(sparsity)) {
        return h3d_set_error(error, error_size,
                             "invalid VSA selector arguments");
    }
    size_t elements = 0;
    if (sequence > UINT32_MAX - (H3D_SOL_BLOCK - 1) ||
        !h3d_mul_size(sequence, heads, &elements) ||
        !h3d_mul_size(elements, H3D_SOL_HEAD_DIM, &elements)) {
        return h3d_set_error(error, error_size, "invalid VSA selector shape");
    }
    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    if (blocks > H3D_SOL_VSA_MAX_BLOCKS) {
        return h3d_set_error(error, error_size,
                             "VSA selector sequence exceeds its shared budget");
    }
    /* `_build_block_mask` (`:387-410`) requires at least one video column: a
     * sequence that is all prefix asks its `topk` for more columns than exist,
     * so it is refused rather than answered with a rule the reference lacks. */
    if (prefix_blocks >= blocks) {
        return h3d_set_error(error, error_size, "invalid VSA prefix span");
    }
    if (!h3d_is(query, H3D_BF16, elements) || !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(value, H3D_BF16, elements) ||
        !h3d_sol_check(plan, sequence, heads, error, error_size)) {
        return 0;
    }
    const uint32_t words = h3d_sol_words(sequence);
    /* The tau path's kernels, unchanged: the pooling both policies need, the
     * statistics, and the threshold kernel that also writes the query
     * centroids this rule scores from. Its `threshold` output is not read
     * here. */
    dim3 pool_grid(blocks, heads);
    h3d_sol_pool_kernel<<<pool_grid, H3D_SOL_THREADS, 0, context->stream>>>(
        (__nv_bfloat16 *)plan->kc->data, (__nv_bfloat16 *)plan->vc->data,
        (const __nv_bfloat16 *)key->data,
        (const __nv_bfloat16 *)value->data, sequence, blocks);
    if (!h3d_launch_ok("VSA pool", error, error_size)) return 0;
    h3d_sol_stats_kernel<<<heads, H3D_SOL_THREADS, 0, context->stream>>>(
        (float *)plan->kc_mean->data, (float *)plan->kc_var->data,
        (const __nv_bfloat16 *)plan->kc->data, blocks);
    if (!h3d_launch_ok("VSA statistics", error, error_size)) return 0;
    h3d_sol_threshold_kernel<<<pool_grid, H3D_SOL_WARP, 0, context->stream>>>(
        (float *)plan->threshold->data, (float *)plan->query_bar->data,
        (const float *)plan->kc_mean->data,
        (const float *)plan->kc_var->data,
        (const __nv_bfloat16 *)query->data, sequence, heads, scale, 0.0f);
    if (!h3d_launch_ok("VSA centroids", error, error_size)) return 0;
    const uint32_t keep = h3d_sol_keep(sparsity, blocks - prefix_blocks);
    const uint32_t shared = (H3D_SOL_HEAD_DIM + blocks) * sizeof(float);
    h3d_sol_vsa_kernel<<<pool_grid, H3D_SOL_WARP, shared, context->stream>>>(
        (uint32_t *)plan->mask->data, (uint32_t *)plan->count->data,
        (const float *)plan->query_bar->data,
        (const __nv_bfloat16 *)plan->kc->data, heads, blocks, words,
        prefix_blocks, keep, scale);
    return h3d_launch_ok("VSA selection", error, error_size);
}

/* ---------------------------------------------------------------------------
 * The sparse forward the mask was built for. Structure and order are the
 * reference's (`triton_ref/fwd.py:88-173`):
 *
 *   - one CTA per (query block, head) of BLOCK = 64 query rows, four warps of
 *     16 rows, so a block's score and softmax statistics stay inside a warp -
 *     the shape `h3d_attention_mma_kernel` already uses;
 *   - groups of GROUP = 32 KV blocks, which is exactly one mask word, so a
 *     group's exact set is that word's bits and its approximated set is the
 *     complement (`fwd.py:92-108`);
 *   - per group, the whole approximated set is folded in one pass from the
 *     pooled summaries, then the group's exact blocks are walked in increasing
 *     block order, one 64-key tile at a time (`fwd.py:109-170`). The order is
 *     part of the FP result: each step rescales the accumulator by the exp2 of
 *     the maximum it moved.
 *
 * Nothing is skipped: `vc` is the block SUM of V (`preprocess.py:82`), so a
 * folded block's output arrives as `P @ vc` and its share of the denominator
 * as `P * block_length` (`fwd.py:126-131`).
 *
 * Q is read once into the MMA A fragments. kc, vc, key and value share two
 * padded tiles: the summaries are dead before a group's first exact block is
 * staged, which keeps the CTA inside the 48 KiB static limit. Every block's
 * staged tile is 16-byte rows in a 272-byte stride, as the ldmatrix loads need
 * (the +8 elements of `H3D_MMA_LD`).
 * ------------------------------------------------------------------------- */

enum {
    H3D_SOLF_WARPS = 4,       /* one warp per 16 query rows: 64 rows per CTA */
    H3D_SOLF_NS = H3D_SOL_BLOCK / 8,      /* n8 slices over 64 keys */
    H3D_SOLF_POOL = H3D_SOL_GROUP / 8,    /* n8 slices over 32 pooled blocks */
};

/* The block's live row count: the tail block is short (`fwd.py:126-128`), and
 * a block past the sequence has none. */
__device__ __forceinline__ static float h3d_sol_length(uint32_t block,
                                                       uint32_t sequence) {
    const uint32_t first = block * H3D_SOL_BLOCK;
    if (first >= sequence) return 0.0f;
    const uint32_t rest = sequence - first;
    return (float)(rest < H3D_SOL_BLOCK ? rest : H3D_SOL_BLOCK);
}

/* Stages `count` rows of a head-major `[heads][rows][128]` tensor from row
 * `first` into a padded shared tile. Rows past `rows` are zeroed so the
 * staging stays branch-free; the score pass masks them out
 * (`h3d_attention_mma_kernel`'s own idiom). */
__device__ __forceinline__ static void h3d_sol_stage(
        __nv_bfloat16 (*tile)[H3D_MMA_LD], const __nv_bfloat16 *source,
        uint32_t head, uint32_t rows, uint32_t first, uint32_t count,
        uint32_t stride, uint32_t thread) {
    for (uint32_t index = thread; index < count * H3D_MMA_CHUNKS;
         index += stride) {
        const uint32_t row = index / H3D_MMA_CHUNKS;
        const uint32_t chunk = index % H3D_MMA_CHUNKS;
        const uint32_t at = first + row;
        uint4 packed = make_uint4(0u, 0u, 0u, 0u);
        if (at < rows) {
            packed = *(const uint4 *)(source +
                ((size_t)head * rows + at) * H3D_SOL_HEAD_DIM + chunk * 8u);
        }
        *(uint4 *)&tile[row][chunk * 8u] = packed;
    }
}

/* `minBlocksPerMultiprocessor = 3` is load-bearing, not a hint: ptxas then
 * caps the kernel at 168 registers, which is what fits three CTAs of 128
 * threads in the 64K register file. At 176 registers it fits two, and on the
 * Spark's 18,432-token shape the same walk over the same tiles then costs
 * 1.83x the dense kernel's: the walk is latency-bound, so the third CTA's
 * eight extra warps per SM are most of the difference. */
__global__ static void __launch_bounds__(H3D_SOLF_WARPS * 32u, 3)
h3d_sol_forward_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        const __nv_bfloat16 *kc, const __nv_bfloat16 *vc,
        const uint32_t *mask, uint32_t sequence, uint32_t heads,
        uint32_t blocks, uint32_t words, float scale) {
    /* The scores' operand tile, aliased per stage: kc while a group's folded
     * set is summed, then that group's exact key block. */
    __shared__ __align__(16) __nv_bfloat16 shared_left[H3D_SOL_BLOCK][H3D_MMA_LD];
    /* The weights' operand tile, aliased the same way: vc, then value. */
    __shared__ __align__(16) __nv_bfloat16 shared_right[H3D_SOL_BLOCK][H3D_MMA_LD];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t quad = lane >> 2u;
    const uint32_t column = (lane & 3u) * 2u;
    const uint32_t matrix = lane >> 3u;
    const uint32_t head = blockIdx.y;
    const uint32_t q_block = blockIdx.x;
    const float log2_scale = scale * 1.4426950408889634f;
    const uint32_t first_row = q_block * H3D_SOL_BLOCK + warp * H3D_MMA_M;
    const uint32_t row0 = first_row + quad, row1 = row0 + 8u;
    const int live0 = row0 < sequence, live1 = row1 < sequence;
    /* A row past the end is read from the last real row and never written,
     * which keeps the CTA's reads in bounds without branching the MMAs. */
    const uint32_t read0 = live0 ? row0 : sequence - 1u;
    const uint32_t read1 = live1 ? row1 : sequence - 1u;
    const __nv_bfloat16 *query0 =
            query + ((size_t)head * sequence + read0) * H3D_SOL_HEAD_DIM;
    const __nv_bfloat16 *query1 =
            query + ((size_t)head * sequence + read1) * H3D_SOL_HEAD_DIM;
    const uint32_t *plan_word = mask + ((size_t)q_block * heads + head) * words;

    uint32_t a[H3D_MMA_KS][4];
#pragma unroll
    for (uint32_t step = 0; step < H3D_MMA_KS; step++) {
        const uint32_t dim = step * 16u + column;
        a[step][0] = h3d_mma_pair(__bfloat162float(query0[dim]),
                                  __bfloat162float(query0[dim + 1u]));
        a[step][1] = h3d_mma_pair(__bfloat162float(query1[dim]),
                                  __bfloat162float(query1[dim + 1u]));
        a[step][2] = h3d_mma_pair(__bfloat162float(query0[dim + 8u]),
                                  __bfloat162float(query0[dim + 9u]));
        a[step][3] = h3d_mma_pair(__bfloat162float(query1[dim + 8u]),
                                  __bfloat162float(query1[dim + 9u]));
    }

    float maximum0 = -INFINITY, maximum1 = -INFINITY;
    float denominator0 = 0.0f, denominator1 = 0.0f;
    float result[H3D_MMA_DIMS][4];
#pragma unroll
    for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
        result[dim][0] = result[dim][1] = result[dim][2] = result[dim][3] = 0.0f;
    }

    for (uint32_t word = 0; word < words; ++word) {
        const uint32_t group_start = word * H3D_SOL_GROUP;
        const uint32_t rest = blocks - group_start;
        const uint32_t valid =
                rest >= H3D_SOL_GROUP ? ~0u : (1u << rest) - 1u;
        const uint32_t bits = plan_word[word] & valid;
        const uint32_t folded = valid & ~bits;

        __syncthreads();  /* the previous block's reads of both tiles are done */
        h3d_sol_stage(shared_left, kc, head, blocks, group_start, H3D_SOL_GROUP,
                      H3D_SOLF_WARPS * 32u, threadIdx.x);
        h3d_sol_stage(shared_right, vc, head, blocks, group_start, H3D_SOL_GROUP,
                      H3D_SOLF_WARPS * 32u, threadIdx.x);
        __syncthreads();

        /* The folded pass, guarded as `has_approximate` (`fwd.py:256-262`): a
         * group with nothing folded takes its maximum over columns that are
         * all masked, so the whole stage is skipped and the running maximum
         * stands. */
        if (folded) {
            float score[H3D_SOLF_POOL][4];
#pragma unroll
            for (uint32_t slice = 0; slice < H3D_SOLF_POOL; slice++) {
                score[slice][0] = score[slice][1] = 0.0f;
                score[slice][2] = score[slice][3] = 0.0f;
            }
#pragma unroll
            for (uint32_t step = 0; step < H3D_MMA_KS; step++) {
#pragma unroll
                for (uint32_t part = 0; part < H3D_SOLF_POOL / 2; part++) {
                    const uint32_t row =
                            part * 16u + (matrix >> 1u) * 8u + (lane & 7u);
                    const uint32_t col = step * 16u + (matrix & 1u) * 8u;
                    uint32_t b[4];
                    h3d_mma_load(b, &shared_left[row][col]);
                    h3d_mma_step(score[2u * part], a[step], b[0], b[1]);
                    h3d_mma_step(score[2u * part + 1u], a[step], b[2], b[3]);
                }
            }
            /* The scale, then the folded set as the only live columns: -inf is
             * what the reference's `tl.where(approximate, scores, -inf)`
             * writes, and it is what zeroes the weights below. */
#pragma unroll
            for (uint32_t slice = 0; slice < H3D_SOLF_POOL; slice++) {
                const uint32_t within = slice * 8u + column;
                score[slice][0] *= log2_scale;
                score[slice][1] *= log2_scale;
                score[slice][2] *= log2_scale;
                score[slice][3] *= log2_scale;
                if (!(folded & (1u << within))) {
                    score[slice][0] = -INFINITY;
                    score[slice][2] = -INFINITY;
                }
                if (!(folded & (1u << (within + 1u)))) {
                    score[slice][1] = -INFINITY;
                    score[slice][3] = -INFINITY;
                }
            }
            float tile0 = -INFINITY, tile1 = -INFINITY;
#pragma unroll
            for (uint32_t slice = 0; slice < H3D_SOLF_POOL; slice++) {
                tile0 = fmaxf(tile0, fmaxf(score[slice][0], score[slice][1]));
                tile1 = fmaxf(tile1, fmaxf(score[slice][2], score[slice][3]));
            }
            tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 1));
            tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 2));
            tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 1));
            tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 2));
            const float next0 = fmaxf(maximum0, tile0);
            const float next1 = fmaxf(maximum1, tile1);
            const float rescale0 = exp2f(maximum0 - next0);
            const float rescale1 = exp2f(maximum1 - next1);

            float weight[H3D_SOLF_POOL][4];
            float part0 = 0.0f, part1 = 0.0f;
#pragma unroll
            for (uint32_t slice = 0; slice < H3D_SOLF_POOL; slice++) {
                weight[slice][0] = exp2f(score[slice][0] - next0);
                weight[slice][1] = exp2f(score[slice][1] - next0);
                weight[slice][2] = exp2f(score[slice][2] - next1);
                weight[slice][3] = exp2f(score[slice][3] - next1);
                /* The denominator takes the probability times the block's own
                 * length, not the probability alone (`fwd.py:126-131`). */
                const uint32_t within = slice * 8u + column;
                const float first = h3d_sol_length(group_start + within, sequence);
                const float second =
                        h3d_sol_length(group_start + within + 1u, sequence);
                part0 += weight[slice][0] * first + weight[slice][1] * second;
                part1 += weight[slice][2] * first + weight[slice][3] * second;
            }
            part0 += __shfl_xor_sync(0xffffffffu, part0, 1);
            part0 += __shfl_xor_sync(0xffffffffu, part0, 2);
            part1 += __shfl_xor_sync(0xffffffffu, part1, 1);
            part1 += __shfl_xor_sync(0xffffffffu, part1, 2);
            denominator0 = denominator0 * rescale0 + part0;
            denominator1 = denominator1 * rescale1 + part1;
            maximum0 = next0;
            maximum1 = next1;
#pragma unroll
            for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
                result[dim][0] *= rescale0;
                result[dim][1] *= rescale0;
                result[dim][2] *= rescale1;
                result[dim][3] *= rescale1;
            }
            /* The fold: P @ vc, P packed from the weights as the A fragments.
             * A folded block's weight is already zero everywhere the mask does
             * not set, so this needs no further gating. */
#pragma unroll
            for (uint32_t step = 0; step < H3D_SOLF_POOL / 2; step++) {
                uint32_t p[4];
                p[0] = h3d_mma_pair(weight[2u * step][0], weight[2u * step][1]);
                p[1] = h3d_mma_pair(weight[2u * step][2], weight[2u * step][3]);
                p[2] = h3d_mma_pair(weight[2u * step + 1u][0],
                                    weight[2u * step + 1u][1]);
                p[3] = h3d_mma_pair(weight[2u * step + 1u][2],
                                    weight[2u * step + 1u][3]);
#pragma unroll
                for (uint32_t part = 0; part < H3D_MMA_DIMS / 2; part++) {
                    const uint32_t row =
                            step * 16u + (matrix & 1u) * 8u + (lane & 7u);
                    const uint32_t col = part * 16u + (matrix >> 1u) * 8u;
                    uint32_t b[4];
                    h3d_mma_load_trans(b, &shared_right[row][col]);
                    h3d_mma_step(result[2u * part], p, b[0], b[1]);
                    h3d_mma_step(result[2u * part + 1u], p, b[2], b[3]);
                }
            }
        }

        /* The group's exact blocks, in increasing block order, one 64-key tile
         * each (`fwd.py:132-170`). `bits` is the whole CTA's mask word, so the
         * branch is uniform and the barriers inside stay legal. */
#pragma unroll 1
        for (uint32_t offset = 0; offset < H3D_SOL_GROUP; ++offset) {
            if (!(bits & (1u << offset))) {
                continue;
            }
            const uint32_t kv_first = (group_start + offset) * H3D_SOL_BLOCK;
            __syncthreads();
            h3d_sol_stage(shared_left, key, head, sequence, kv_first,
                          H3D_SOL_BLOCK, H3D_SOLF_WARPS * 32u, threadIdx.x);
            h3d_sol_stage(shared_right, value, head, sequence, kv_first,
                          H3D_SOL_BLOCK, H3D_SOLF_WARPS * 32u, threadIdx.x);
            __syncthreads();

            float score[H3D_SOLF_NS][4];
#pragma unroll
            for (uint32_t slice = 0; slice < H3D_SOLF_NS; slice++) {
                score[slice][0] = score[slice][1] = 0.0f;
                score[slice][2] = score[slice][3] = 0.0f;
            }
#pragma unroll
            for (uint32_t step = 0; step < H3D_MMA_KS; step++) {
#pragma unroll
                for (uint32_t part = 0; part < H3D_SOLF_NS / 2; part++) {
                    const uint32_t row =
                            part * 16u + (matrix >> 1u) * 8u + (lane & 7u);
                    const uint32_t col = step * 16u + (matrix & 1u) * 8u;
                    uint32_t b[4];
                    h3d_mma_load(b, &shared_left[row][col]);
                    h3d_mma_step(score[2u * part], a[step], b[0], b[1]);
                    h3d_mma_step(score[2u * part + 1u], a[step], b[2], b[3]);
                }
            }
#pragma unroll
            for (uint32_t slice = 0; slice < H3D_SOLF_NS; slice++) {
                const uint32_t key_index = kv_first + slice * 8u + column;
                score[slice][0] *= log2_scale;
                score[slice][1] *= log2_scale;
                score[slice][2] *= log2_scale;
                score[slice][3] *= log2_scale;
                if (key_index >= sequence) {   /* `fwd.py:150-153` */
                    score[slice][0] = -INFINITY;
                    score[slice][2] = -INFINITY;
                }
                if (key_index + 1u >= sequence) {
                    score[slice][1] = -INFINITY;
                    score[slice][3] = -INFINITY;
                }
            }
            float tile0 = -INFINITY, tile1 = -INFINITY;
#pragma unroll
            for (uint32_t slice = 0; slice < H3D_SOLF_NS; slice++) {
                tile0 = fmaxf(tile0, fmaxf(score[slice][0], score[slice][1]));
                tile1 = fmaxf(tile1, fmaxf(score[slice][2], score[slice][3]));
            }
            tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 1));
            tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, 2));
            tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 1));
            tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, 2));
            const float next0 = fmaxf(maximum0, tile0);
            const float next1 = fmaxf(maximum1, tile1);
            const float rescale0 = exp2f(maximum0 - next0);
            const float rescale1 = exp2f(maximum1 - next1);

            float weight[H3D_SOLF_NS][4];
            float part0 = 0.0f, part1 = 0.0f;
#pragma unroll
            for (uint32_t slice = 0; slice < H3D_SOLF_NS; slice++) {
                weight[slice][0] = exp2f(score[slice][0] - next0);
                weight[slice][1] = exp2f(score[slice][1] - next0);
                weight[slice][2] = exp2f(score[slice][2] - next1);
                weight[slice][3] = exp2f(score[slice][3] - next1);
                part0 += weight[slice][0] + weight[slice][1];
                part1 += weight[slice][2] + weight[slice][3];
            }
            part0 += __shfl_xor_sync(0xffffffffu, part0, 1);
            part0 += __shfl_xor_sync(0xffffffffu, part0, 2);
            part1 += __shfl_xor_sync(0xffffffffu, part1, 1);
            part1 += __shfl_xor_sync(0xffffffffu, part1, 2);
            denominator0 = denominator0 * rescale0 + part0;
            denominator1 = denominator1 * rescale1 + part1;
            maximum0 = next0;
            maximum1 = next1;
#pragma unroll
            for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
                result[dim][0] *= rescale0;
                result[dim][1] *= rescale0;
                result[dim][2] *= rescale1;
                result[dim][3] *= rescale1;
            }
#pragma unroll
            for (uint32_t step = 0; step < H3D_SOLF_NS / 2; step++) {
                uint32_t p[4];
                p[0] = h3d_mma_pair(weight[2u * step][0], weight[2u * step][1]);
                p[1] = h3d_mma_pair(weight[2u * step][2], weight[2u * step][3]);
                p[2] = h3d_mma_pair(weight[2u * step + 1u][0],
                                    weight[2u * step + 1u][1]);
                p[3] = h3d_mma_pair(weight[2u * step + 1u][2],
                                    weight[2u * step + 1u][3]);
#pragma unroll
                for (uint32_t part = 0; part < H3D_MMA_DIMS / 2; part++) {
                    const uint32_t row =
                            step * 16u + (matrix & 1u) * 8u + (lane & 7u);
                    const uint32_t col = part * 16u + (matrix >> 1u) * 8u;
                    uint32_t b[4];
                    h3d_mma_load_trans(b, &shared_right[row][col]);
                    h3d_mma_step(result[2u * part], p, b[0], b[1]);
                    h3d_mma_step(result[2u * part + 1u], p, b[2], b[3]);
                }
            }
        }
    }

    __syncthreads();
    if (live0) {
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
            const size_t at = ((size_t)row0 * heads + head) * H3D_SOL_HEAD_DIM +
                              dim * 8u + column;
            output[at] = __float2bfloat16(result[dim][0] / denominator0);
            output[at + 1u] = __float2bfloat16(result[dim][1] / denominator0);
        }
    }
    if (live1) {
#pragma unroll
        for (uint32_t dim = 0; dim < H3D_MMA_DIMS; dim++) {
            const size_t at = ((size_t)row1 * heads + head) * H3D_SOL_HEAD_DIM +
                              dim * 8u + column;
            output[at] = __float2bfloat16(result[dim][2] / denominator1);
            output[at + 1u] = __float2bfloat16(result[dim][3] / denominator1);
        }
    }
}

int h3d_sol_forward(h3d_context *context, h3d_tensor *output,
                    const struct h3d_sol_plan *plan, const h3d_tensor *query,
                    const h3d_tensor *key, const h3d_tensor *value,
                    uint32_t sequence, uint32_t heads, float scale, char *error,
                    size_t error_size) {
    if (!context || !sequence || !heads) {
        return h3d_set_error(error, error_size,
                             "invalid Sol-Attn forward arguments");
    }
    /* `sequence + BLOCK - 1` wraps above UINT32_MAX - 63 and the element count
     * overflows size_t for large arguments; both are refused before a launch
     * (`h3d_mul_size`, `:716-720`). */
    size_t elements = 0;
    if (sequence > UINT32_MAX - (H3D_SOL_BLOCK - 1) ||
        !h3d_mul_size(sequence, heads, &elements) ||
        !h3d_mul_size(elements, H3D_SOL_HEAD_DIM, &elements)) {
        return h3d_set_error(error, error_size, "invalid Sol-Attn forward shape");
    }
    if (!h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(query, H3D_BF16, elements) ||
        !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(value, H3D_BF16, elements) ||
        !h3d_sol_check(plan, sequence, heads, error, error_size)) {
        return 0;
    }
    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    dim3 grid(blocks, heads);
    h3d_sol_forward_kernel<<<grid, H3D_SOLF_WARPS * 32u, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
        (const __nv_bfloat16 *)key->data, (const __nv_bfloat16 *)value->data,
        (const __nv_bfloat16 *)plan->kc->data,
        (const __nv_bfloat16 *)plan->vc->data,
        (const uint32_t *)plan->mask->data, sequence, heads, blocks,
        h3d_sol_words(sequence), scale);
    return h3d_launch_ok("Sol-Attn forward", error, error_size);
}

/* -------------------------------------------------------------------------
 * The LTX-2.5 refiner's Sol-Attn, token-major.
 *
 * The rule is the one above, unchanged: the released kernel's `diag` tau
 * policy (`sol_attn/preprocess.py:280-322`, `sol_attn/triton_ref/fwd.py:40-173`),
 * which the Spark release enforces as its only backend
 * (`Sol-H3-Spark/runtime/stage2_ops/models.py:100-103`,
 * `_actual_sol_backend() != "triton"` raised).
 * What changes is the layout: the refiner's attention is `[rows][heads][128]`
 * (`ltx_refiner.rs`'s `Attention::run`), and a head-major kernel would need a
 * transpose of every operand per layer - 47 layers of 16,128x32x128 BF16, more
 * traffic than the attention it feeds. So the pooling, the query centroid and
 * the forward are re-expressed row-major, and the two layout-agnostic steps
 * (`h3d_sol_stats_kernel`, `h3d_sol_route_kernel`, which read only `kc` and
 * `query_bar`) are reused verbatim.
 *
 * The forward is the same walk as `h3d_sol_forward_kernel`, in the shape of
 * `h3d_ltx_attention_kernel`: one CTA per (16-row query tile, head), one warp
 * per H3D_ATTENTION_ROWS rows, each row's 128-wide dot split over the warp's
 * lanes. Per 64-token KV block: an exact block walks its keys, an approximated
 * one is folded from `kc`/`vc` in a single step - `denominator += w * length`
 * because `vc` is a SUM (`sol_attn/preprocess.py:82`,
 * `sol_attn/triton_ref/preprocess.py:53`), and the fold is
 * `output += w @ vc` with `row_sum += w * length`
 * (`sol_attn/triton_ref/fwd.py:121-130`). The block decision
 * is the plan's mask: the diagonal band and the sink were written into it by
 * `h3d_sol_ltx_search`, so this kernel reads `kc`, `vc` and `mask` and adds no
 * policy of its own.
 *
 * Deviations, named: row-major against the reference's `[B,T,H,D]` (the same
 * numbers, indexed differently); natural-log softmax where `fwd.py` tracks the
 * maximum in log2 units, the port's dense kernel's own units, so a tau that
 * forces every block exact reproduces `h3d_ltx_attention`'s arithmetic; the
 * accumulation orders the port fixes where Triton's are not reproducible here.
 * Sol-Attn requires head dimension 128 (`interface.py:29-30`), which the
 * refiner's video self-attention has (32 heads x 128).
 * ------------------------------------------------------------------------- */

__global__ static void h3d_sol_ltx_pool_kernel(
        __nv_bfloat16 *kc, __nv_bfloat16 *vc, const __nv_bfloat16 *key,
        const __nv_bfloat16 *value, uint32_t sequence, uint32_t heads,
        uint32_t blocks) {
    const uint32_t head = blockIdx.y;
    const uint32_t block = blockIdx.x;
    const uint32_t column = threadIdx.x;
    if (column >= H3D_SOL_HEAD_DIM) return;
    const uint32_t first = block * H3D_SOL_BLOCK;
    uint32_t rows = sequence - first;
    if (rows > H3D_SOL_BLOCK) rows = H3D_SOL_BLOCK;
    const size_t base = ((size_t)first * heads + head) * H3D_SOL_HEAD_DIM;
    const size_t stride = (size_t)heads * H3D_SOL_HEAD_DIM;
    float key_sum = 0.0f;
    float value_sum = 0.0f;
    /* Sequential over the block's rows, the reference's row order
     * (`preprocess.py:64`), so the sum is the same number as the head-major
     * pool's. */
    for (uint32_t row = 0; row < rows; ++row) {
        const size_t at = base + (size_t)row * stride + column;
        key_sum += h3d_value(key[at]);
        value_sum += h3d_value(value[at]);
    }
    const size_t out =
        ((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM + column;
    kc[out] = h3d_store(key_sum / (float)rows);
    vc[out] = h3d_store(value_sum);              /* `:66`: a sum, no divide */
}

/* The query centroid and the routing threshold, token-major. Identical to
 * `h3d_sol_threshold_kernel` but for the read index. */
__global__ static void h3d_sol_ltx_threshold_kernel(
        float *threshold, float *query_bar, const float *kc_mean,
        const float *kc_var, const __nv_bfloat16 *query, uint32_t sequence,
        uint32_t heads, float scale, float tau) {
    enum { CHUNKS = H3D_SOL_HEAD_DIM / H3D_SOL_WARP };
    const uint32_t head = blockIdx.y;
    const uint32_t q_block = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t first = q_block * H3D_SOL_BLOCK;
    uint32_t rows = sequence - first;
    if (rows > H3D_SOL_BLOCK) rows = H3D_SOL_BLOCK;
    const size_t base = ((size_t)first * heads + head) * H3D_SOL_HEAD_DIM;
    const size_t stride = (size_t)heads * H3D_SOL_HEAD_DIM;
    float centroid[CHUNKS];
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        const uint32_t dim = chunk * H3D_SOL_WARP + lane;
        float total = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            total +=
                h3d_value(query[base + (size_t)row * stride + dim]);
        }
        centroid[chunk] = total / (float)rows;
        query_bar[((size_t)q_block * heads + head) * H3D_SOL_HEAD_DIM + dim] =
            centroid[chunk];
    }
    const float log2_scale = scale * 1.4426950408889634f;
    float mean = 0.0f;
    float variance = 0.0f;
    for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
        const uint32_t dim = chunk * H3D_SOL_WARP + lane;
        const float m = kc_mean[(size_t)head * H3D_SOL_HEAD_DIM + dim];
        const float v = kc_var[(size_t)head * H3D_SOL_HEAD_DIM + dim];
        mean += centroid[chunk] * m;
        variance += centroid[chunk] * centroid[chunk] * v;
    }
    for (int offset = H3D_SOL_WARP / 2; offset > 0; offset >>= 1) {
        mean += __shfl_xor_sync(0xffffffffu, mean, offset);
        variance += __shfl_xor_sync(0xffffffffu, variance, offset);
    }
    if (lane != 0) return;
    variance *= log2_scale * log2_scale;
    const float clamped = variance > 0.0f ? variance : 0.0f;
    const float std = sqrtf(clamped + 1.0e-6f);
    threshold[(size_t)q_block * heads + head] = mean * log2_scale + tau * std;
}

__global__ static void h3d_sol_ltx_forward_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        const __nv_bfloat16 *kc, const __nv_bfloat16 *vc,
        const uint32_t *mask, uint32_t q_rows, uint32_t kv_rows,
        uint32_t heads, uint32_t blocks, uint32_t words, float scale) {
    enum { PARTS = H3D_SOL_HEAD_DIM / 32, KEYS = 8, ROWS = H3D_ATTENTION_ROWS,
           QUERIES = 8 * ROWS };
    __shared__ float shared_key[KEYS][H3D_SOL_HEAD_DIM];
    __shared__ float shared_value[KEYS][H3D_SOL_HEAD_DIM];
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t head = blockIdx.y;
    const uint32_t first_row = blockIdx.x * QUERIES + warp * ROWS;
    /* Every row of this CTA's 16-row tile lies in one 64-token query block
     * (16 divides 64), so the mask word is the tile's, not the warp's. */
    const uint32_t q_block = (blockIdx.x * QUERIES) / H3D_SOL_BLOCK;
    int active[ROWS];
    float result[ROWS][PARTS];
    float maximum[ROWS];
    float denominator[ROWS];
    float query_item[ROWS][PARTS];
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        uint32_t row = first_row + r;
        active[r] = row < q_rows;
        maximum[r] = -INFINITY;
        denominator[r] = 0.0f;
#pragma unroll
        for (uint32_t part = 0; part < PARTS; part++) {
            result[r][part] = 0.0f;
            query_item[r][part] = 0.0f;
        }
        if (!active[r]) {
            continue;
        }
        const size_t query_base = ((size_t)row * heads + head) * H3D_SOL_HEAD_DIM;
#pragma unroll
        for (uint32_t part = 0; part < PARTS; part++) {
            query_item[r][part] =
                __bfloat162float(query[query_base + lane + part * 32]);
        }
    }

    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t word = block / H3D_SOL_GROUP;
        const uint32_t bit = block % H3D_SOL_GROUP;
        const uint32_t bits =
            mask[((size_t)q_block * heads + head) * words + word];
        const bool exact = (bits >> bit) & 1u;
        const uint32_t first_key = block * H3D_SOL_BLOCK;
        uint32_t key_rows = kv_rows - first_key;
        if (key_rows > H3D_SOL_BLOCK) key_rows = H3D_SOL_BLOCK;
        if (!key_rows) {
            continue;
        }
        if (!exact) {
            /* One fold for the whole block: the score is the row's dot with
             * the block mean of K, the value is the block SUM of V, and the
             * denominator counts the block's length (`fwd.py:108-131`). */
            if (threadIdx.x < H3D_SOL_HEAD_DIM) {
                const size_t summary =
                    ((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM +
                    threadIdx.x;
                shared_key[0][threadIdx.x] = h3d_value(kc[summary]);
                shared_value[0][threadIdx.x] = h3d_value(vc[summary]);
            }
            __syncthreads();
#pragma unroll
            for (uint32_t r = 0; r < ROWS; r++) {
                float product = 0.0f;
                if (active[r]) {
#pragma unroll
                    for (uint32_t part = 0; part < PARTS; part++) {
                        product += query_item[r][part] *
                                   shared_key[0][lane + part * 32];
                    }
#pragma unroll
                    for (uint32_t offset = 16; offset; offset >>= 1) {
                        product += __shfl_down_sync(0xffffffffu, product, offset);
                    }
                }
                float rescale = 1.0f;
                float weight = 0.0f;
                if (active[r] && lane == 0) {
                    float score = product * scale;
                    float next_maximum = fmaxf(maximum[r], score);
                    rescale = expf(maximum[r] - next_maximum);
                    weight = expf(score - next_maximum);
                    denominator[r] = denominator[r] * rescale +
                                     weight * (float)key_rows;
                    maximum[r] = next_maximum;
                }
                rescale = __shfl_sync(0xffffffffu, rescale, 0);
                weight = __shfl_sync(0xffffffffu, weight, 0);
                if (active[r]) {
#pragma unroll
                    for (uint32_t part = 0; part < PARTS; part++) {
                        result[r][part] = result[r][part] * rescale +
                                          weight *
                                          shared_value[0][lane + part * 32];
                    }
                }
            }
            __syncthreads();
            continue;
        }
        for (uint32_t key_block = 0; key_block < key_rows; key_block += KEYS) {
            if (threadIdx.x < H3D_SOL_HEAD_DIM) {
#pragma unroll
                for (uint32_t item = 0; item < KEYS; item++) {
                    uint32_t source = first_key + key_block + item;
                    if (key_block + item < key_rows) {
                        size_t index = ((size_t)source * heads + head) *
                                       H3D_SOL_HEAD_DIM + threadIdx.x;
                        shared_key[item][threadIdx.x] =
                            __bfloat162float(key[index]);
                        shared_value[item][threadIdx.x] =
                            __bfloat162float(value[index]);
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (uint32_t item = 0; item < KEYS; item++) {
                if (key_block + item >= key_rows) {
                    break;
                }
#pragma unroll
                for (uint32_t r = 0; r < ROWS; r++) {
                    float product = 0.0f;
                    if (active[r]) {
#pragma unroll
                        for (uint32_t part = 0; part < PARTS; part++) {
                            uint32_t dimension = lane + part * 32;
                            product += query_item[r][part] *
                                       shared_key[item][dimension];
                        }
#pragma unroll
                        for (uint32_t offset = 16; offset; offset >>= 1) {
                            product +=
                                __shfl_down_sync(0xffffffffu, product, offset);
                        }
                    }
                    float rescale = 1.0f;
                    float weight = 0.0f;
                    if (active[r] && lane == 0) {
                        float score = product * scale;
                        float next_maximum = fmaxf(maximum[r], score);
                        rescale = expf(maximum[r] - next_maximum);
                        weight = expf(score - next_maximum);
                        denominator[r] = denominator[r] * rescale + weight;
                        maximum[r] = next_maximum;
                    }
                    rescale = __shfl_sync(0xffffffffu, rescale, 0);
                    weight = __shfl_sync(0xffffffffu, weight, 0);
                    if (active[r]) {
#pragma unroll
                        for (uint32_t part = 0; part < PARTS; part++) {
                            uint32_t dimension = lane + part * 32;
                            result[r][part] = result[r][part] * rescale +
                                              weight * shared_value[item][dimension];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        denominator[r] = __shfl_sync(0xffffffffu, denominator[r], 0);
        if (!active[r]) {
            continue;
        }
        uint32_t row = first_row + r;
#pragma unroll
        for (uint32_t part = 0; part < PARTS; part++) {
            uint32_t dimension = lane + part * 32;
            size_t index = ((size_t)row * heads + head) * H3D_SOL_HEAD_DIM +
                           dimension;
            output[index] = __float2bfloat16(result[r][part] / denominator[r]);
        }
    }
}

/* The refiner's search: pool, statistics, threshold, route. No sink: the
 * release's stage-2 video self-attention has no prefix in its sequence
 * (`sol_attention.py:3-4`, `attn1`), so `sink_start`/`sink_tokens` are the
 * empty range `[blocks, blocks)` and no block is forced exact beyond the
 * diagonal band the route kernel adds itself. */
int h3d_sol_ltx_search(h3d_context *context, const struct h3d_sol_plan *plan,
                       const h3d_tensor *query, const h3d_tensor *key,
                       const h3d_tensor *value, uint32_t sequence,
                       uint32_t heads, float scale, float tau, char *error,
                       size_t error_size) {
    size_t elements = 0;
    if (!context || !sequence || !heads ||
        sequence > UINT32_MAX - (H3D_SOL_BLOCK - 1) ||
        !h3d_mul_size(sequence, heads, &elements) ||
        !h3d_mul_size(elements, H3D_SOL_HEAD_DIM, &elements)) {
        return h3d_set_error(error, error_size,
                             "invalid Sol-Attn search shape");
    }
    if (!h3d_is(query, H3D_BF16, elements) || !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(value, H3D_BF16, elements) ||
        !h3d_sol_check(plan, sequence, heads, error, error_size)) {
        return 0;
    }
    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    const uint32_t words = h3d_sol_words(sequence);
    dim3 pool_grid(blocks, heads);
    h3d_sol_ltx_pool_kernel<<<pool_grid, H3D_SOL_THREADS, 0, context->stream>>>(
        (__nv_bfloat16 *)plan->kc->data, (__nv_bfloat16 *)plan->vc->data,
        (const __nv_bfloat16 *)key->data,
        (const __nv_bfloat16 *)value->data, sequence, heads, blocks);
    if (!h3d_launch_ok("Sol-Attn LTX pool", error, error_size)) return 0;
    h3d_sol_stats_kernel<<<heads, H3D_SOL_THREADS, 0, context->stream>>>(
        (float *)plan->kc_mean->data, (float *)plan->kc_var->data,
        (const __nv_bfloat16 *)plan->kc->data, blocks);
    if (!h3d_launch_ok("Sol-Attn LTX statistics", error, error_size)) return 0;
    dim3 threshold_grid(blocks, heads);
    h3d_sol_ltx_threshold_kernel<<<threshold_grid, H3D_SOL_WARP, 0,
                                   context->stream>>>(
        (float *)plan->threshold->data, (float *)plan->query_bar->data,
        (const float *)plan->kc_mean->data,
        (const float *)plan->kc_var->data,
        (const __nv_bfloat16 *)query->data, sequence, heads, scale, tau);
    if (!h3d_launch_ok("Sol-Attn LTX threshold", error, error_size)) return 0;
    dim3 route_grid((blocks + H3D_SOL_QM - 1) / H3D_SOL_QM, heads);
    h3d_sol_route_kernel<<<route_grid, H3D_SOL_WARP, 0, context->stream>>>(
        (uint32_t *)plan->mask->data, (uint32_t *)plan->count->data,
        (const __nv_bfloat16 *)plan->kc->data,
        (const float *)plan->query_bar->data,
        (const float *)plan->threshold->data, heads, blocks, words,
        scale, blocks, blocks);
    return h3d_launch_ok("Sol-Attn LTX route", error, error_size);
}

int h3d_sol_ltx_forward(h3d_context *context, h3d_tensor *output,
                        const struct h3d_sol_plan *plan, const h3d_tensor *query,
                        const h3d_tensor *key, const h3d_tensor *value,
                        uint32_t sequence, uint32_t heads, float scale,
                        char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !sequence || !heads ||
        sequence > UINT32_MAX - (H3D_SOL_BLOCK - 1) ||
        !h3d_mul_size(sequence, heads, &elements) ||
        !h3d_mul_size(elements, H3D_SOL_HEAD_DIM, &elements)) {
        return h3d_set_error(error, error_size,
                             "invalid Sol-Attn forward shape");
    }
    if (!h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(query, H3D_BF16, elements) ||
        !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(value, H3D_BF16, elements) ||
        !h3d_sol_check(plan, sequence, heads, error, error_size)) {
        return 0;
    }
    enum { QUERIES = 8 * H3D_ATTENTION_ROWS };
    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    dim3 grid((sequence + QUERIES - 1) / QUERIES, heads);
    h3d_sol_ltx_forward_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
        (const __nv_bfloat16 *)key->data,
        (const __nv_bfloat16 *)value->data,
        (const __nv_bfloat16 *)plan->kc->data,
        (const __nv_bfloat16 *)plan->vc->data,
        (const uint32_t *)plan->mask->data, sequence, sequence, heads, blocks,
        h3d_sol_words(sequence), scale);
    return h3d_launch_ok("Sol-Attn LTX forward", error, error_size);
}

/* ==========================================================================
 * h3b_* — the H3-to-LTX latent path's device ops.
 *
 * Source of truth: `runtime/stage2_ops/h3_upscale.py` (the pinned author node
 * wrapped) and `runtime/stage2_ops/h3_ltx_adapter/{constants,geometry,model}.
 * py`, against the architecture the node itself declares
 * (`nodes/minimax_h3_latent_upscaler_3d.py` at `d7c01b90`). Both models run
 * here; nothing in this section evaluates Python or calls out of process.
 *
 * Every kernel is templated on the element type, so one binary carries the
 * bf16 path the reference runs (`h3_upscale.py:82`, the adapter's own
 * `dtype=torch.bfloat16` default) and an f32 path that exists to be gated
 * tightly against an f32 reference vector. Both accumulate in f32 and round
 * once, which is what torch's own bf16 kernels do.
 *
 * Layouts are the H3 family's: channel-major `[channel][time][height][width]`,
 * weights as PyTorch stores them (`[out][in][kd][kh][kw]`).
 * ========================================================================== */

/* Element-type traits: `to` widens for the arithmetic, `from` rounds back. */
template <typename T> struct h3b_element;

template <> struct h3b_element<float> {
    static __device__ __forceinline__ float to(float value) { return value; }
    static __device__ __forceinline__ float from(float value) { return value; }
};

template <> struct h3b_element<__nv_bfloat16> {
    static __device__ __forceinline__ float to(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    static __device__ __forceinline__ __nv_bfloat16 from(float value) {
        return __float2bfloat16(value);
    }
};

/* `h3b_time_pack`'s "write zero" entry, `geometry.py:72-78`'s `occupied`. */
#define H3B_UNOCCUPIED 0xffffffffu

#define H3B_BLOCK 256u

/* A block-wide (sum, sum of squares) reduction through shared memory. */
__device__ static void h3b_block_sum2(float *sum, float *squares) {
    __shared__ float partial_sum[32];
    __shared__ float partial_squares[32];
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        *sum += __shfl_down_sync(mask, *sum, offset);
        *squares += __shfl_down_sync(mask, *squares, offset);
    }
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5u;
    if (lane == 0u) {
        partial_sum[warp] = *sum;
        partial_squares[warp] = *squares;
    }
    __syncthreads();
    unsigned warps = (blockDim.x + 31u) >> 5u;
    if (warp == 0u) {
        *sum = lane < warps ? partial_sum[lane] : 0.0f;
        *squares = lane < warps ? partial_squares[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            *sum += __shfl_down_sync(mask, *sum, offset);
            *squares += __shfl_down_sync(mask, *squares, offset);
        }
    } else {
        *sum = 0.0f;
        *squares = 0.0f;
    }
}

/* `F.pad`'s zero halo (`minimax_h3_latent_upscaler_3d.py:221-223`'s Conv3d
 * padding, applied once instead of per convolution). */
template <typename T>
__global__ static void h3b_pad3d_kernel(T *out, const T *in, uint32_t channels,
                                        uint32_t depth, uint32_t height,
                                        uint32_t width, uint32_t pad_t,
                                        uint32_t pad_h, uint32_t pad_w,
                                        size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t stride_w = width + 2 * pad_w;
    uint32_t stride_h = height + 2 * pad_h;
    uint32_t stride_t = depth + 2 * pad_t;
    uint32_t x = (uint32_t)(index % stride_w);
    size_t rest = index / stride_w;
    uint32_t y = (uint32_t)(rest % stride_h);
    rest /= stride_h;
    uint32_t t = (uint32_t)(rest % stride_t);
    uint32_t channel = (uint32_t)(rest / stride_t);
    if (t >= pad_t && y >= pad_h && x >= pad_w && t - pad_t < depth &&
        y - pad_h < height && x - pad_w < width) {
        out[index] = in[(((size_t)channel * depth + (t - pad_t)) * height +
                         (y - pad_h)) * width + (x - pad_w)];
        return;
    }
    out[index] = h3b_element<T>::from(0.0f);
}

/* The convolution's im2col: row `(t, y, x)` of the interior grid, column
 * `((channel * kd + kt) * kh + ky) * kw + kx`. The row-major weight is then
 * PyTorch's own `[out][in][kd][kh][kw]` flattened, so no weight transform
 * exists anywhere in this path. */
template <typename T>
__global__ static void h3b_im2col_kernel(T *out, const T *padded,
                                         uint32_t channels, uint32_t pad_depth,
                                         uint32_t pad_height, uint32_t pad_width,
                                         uint32_t kd, uint32_t kh, uint32_t kw,
                                         uint32_t depth, uint32_t height,
                                         uint32_t width, uint32_t t_begin,
                                         size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    size_t columns = (size_t)channels * kd * kh * kw;
    size_t row = index / columns;
    uint32_t column = (uint32_t)(index % columns);
    uint32_t kx = column % kw;
    uint32_t ky = (column / kw) % kh;
    uint32_t kt = (column / (kw * kh)) % kd;
    uint32_t channel = column / (kw * kh * kd);
    size_t plane = (size_t)height * width;
    uint32_t t = t_begin + (uint32_t)(row / plane);
    uint32_t spatial = (uint32_t)(row % plane);
    uint32_t x = spatial % width;
    uint32_t y = spatial / width;
    out[index] =
        padded[(((size_t)channel * pad_depth + (t + kt)) * pad_height +
                (y + ky)) * pad_width + (x + kx)];
}

/* The planar buffer, which is what replaces the im2col matrix: the padded
 * activation in channel-last form, once per kernel column tap.
 *
 * `planar[t'][y'][x][s][c] = padded[c][t'][y'][x + s]` with `s` the column tap,
 * so `planar[t'][y'][x][s][c]` sits at `((t'*pad_height + y')*out_width + x) *
 * kw*channels + s*channels + c`. One row of the buffer is a full image row of
 * `kw*channels` values, and the element (row `m = y*out_width + x`, column `k =
 * s*channels + c`) of the convolution's A operand for the tap `(kt, ky)` is at
 * `((t + kt)*pad_height + y + ky)*out_width*K + m*K + k` with `K = kw*channels`
 * — a plain row-major matrix with leading dimension `K`, so every tap is one
 * strided-batched `cublasLt` call over a dense operand and no im2col matrix is
 * ever materialized (`h3b_matmul_planar`).
 *
 * The source is channel-major, so the read is coalesced along the image's own
 * `x` and the write along the channel: the tile is transposed in shared memory
 * the way `h3b_epilogue_kernel` does it, and each source element is written
 * `kw` times, once per shift. */
#define H3B_PLANAR_CHANNELS 32u

template <typename T>
__global__ static void h3b_planar_kernel(T *out, const T *in, uint32_t channels,
                                         uint32_t depth, uint32_t height,
                                         uint32_t width, uint32_t kw,
                                         uint32_t pad_t, uint32_t pad_h,
                                         uint32_t pad_w, uint32_t pad_depth,
                                         uint32_t pad_height, uint32_t out_width,
                                         int replicate_t) {
    uint32_t padded_width = width + 2 * pad_w;
    uint32_t column = blockIdx.y * H3B_PLANAR_CHANNELS;
    uint32_t block_channels = channels - column < H3B_PLANAR_CHANNELS
                                  ? channels - column
                                  : H3B_PLANAR_CHANNELS;
    uint32_t t = blockIdx.x / pad_height;
    uint32_t y = blockIdx.x % pad_height;
    int source_t = (int)t - (int)pad_t;
    int source_y = (int)y - (int)pad_h;
    // `CausalConv3d`'s non-causal temporal halo is the edge frame repeated
    // (`convolution.py:307-311`, `(kernel_size - 1) // 2` frames each side),
    // not zeros: the clamp is what makes the decoder's first and last frames
    // read the right values.
    if (replicate_t && source_t < 0) source_t = 0;
    if (replicate_t && source_t >= (int)depth) source_t = (int)depth - 1;

    // Row `i` of the tile is `in[column + i]`'s image row, zero where the
    // padding reaches: the entry is read once per padded coordinate and the
    // shifts below read it again, so the halo costs nothing extra.
    extern __shared__ __align__(16) unsigned char raw[];
    float *tile = (float *)raw;
    size_t stride = padded_width + 1u;

    for (size_t index = threadIdx.x; index < (size_t)block_channels * padded_width;
         index += blockDim.x) {
        uint32_t i = (uint32_t)(index / padded_width);
        uint32_t px = (uint32_t)(index % padded_width);
        int source_x = (int)px - (int)pad_w;
        float value = 0.0f;
        if (source_t >= 0 && source_t < (int)depth && source_y >= 0 &&
            source_y < (int)height && source_x >= 0 && source_x < (int)width) {
            size_t at = (((size_t)(column + i) * depth + source_t) * height +
                         source_y) *
                            width +
                        source_x;
            value = h3b_element<T>::to(in[at]);
        }
        tile[i * stride + px] = value;
    }
    __syncthreads();

    uint32_t kernel_columns = kw * channels;
    size_t base = ((size_t)t * pad_height + y) * out_width * kernel_columns;
    size_t writes = (size_t)kw * out_width * block_channels;
    for (size_t index = threadIdx.x; index < writes; index += blockDim.x) {
        // Fastest first: the channel within the block, so a warp's 32 stores
        // are 32 consecutive `T` and every shift is its own coalesced pass.
        uint32_t i = (uint32_t)(index % block_channels);
        size_t rest = index / block_channels;
        uint32_t x = (uint32_t)(rest % out_width);
        uint32_t s = (uint32_t)(rest / out_width);
        out[base + (size_t)x * kernel_columns + (size_t)s * channels + column + i] =
            h3b_element<T>::from(tile[i * stride + x + s]);
    }
}

/* The pass decomposition's B operand: `packed[tap][o][s*channels + c] =
 * weight[o][c][kt][ky][s]` with `tap = kt*kh + ky`.
 *
 * The checkpoint's `[out][in][kd][kh][kw]` order cannot express it — there the
 * `kw` triple is contiguous and the channel stride is `kd*kh*kw` — so the
 * passes need their own copy. It is a layout copy at load and changes no
 * value; the gate checks it against the same permutation on the host. */
template <typename T>
__global__ static void h3b_pack_kernel(T *out, const T *weight,
                                       uint32_t out_channels,
                                       uint32_t in_channels, uint32_t taps,
                                       uint32_t kw, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    // Fastest first: the input channel, so the write is coalesced and the read
    // (stride `taps`) is the one that strides.
    uint32_t c = (uint32_t)(index % in_channels);
    size_t rest = index / in_channels;
    uint32_t s = (uint32_t)(rest % kw);
    size_t pass_and_out = rest / kw;
    uint32_t o = (uint32_t)(pass_and_out % out_channels);
    uint32_t pass = (uint32_t)(pass_and_out / out_channels);
    out[index] = weight[((size_t)o * in_channels + c) * taps +
                        (size_t)pass * kw + s];
}

/* `[channel][spatial]` -> `[spatial][channel]` with the bias added in f32 and
 * one rounding: the convolution's epilogue, a tiled transpose (both the f32
 * read and the `T` write stay coalesced, the tile is transposed in shared
 * memory with a one-element pad so neither pass conflicts on a bank).
 *
 * `row_stride` and `row_offset` place the tile inside a larger channel-major
 * tensor, which is how the convolution runs in temporal slabs without ever
 * materializing its whole im2col. */
#define H3B_TILE 32u

template <typename T>
__global__ static void h3b_epilogue_kernel(T *out, const float *temp,
                                           const T *bias, uint32_t spatial,
                                           uint32_t channels, size_t row_stride,
                                           size_t row_offset, int has_bias) {
    __shared__ float tile[H3B_TILE][H3B_TILE + 1];
    // Load: threadIdx.x walks the matrix's fast axis, which is the channel.
    uint32_t in_channel = blockIdx.x * H3B_TILE + threadIdx.x;
    uint32_t in_point = blockIdx.y * H3B_TILE + threadIdx.y;
    if (in_channel < channels && in_point < spatial) {
        tile[threadIdx.y][threadIdx.x] = temp[(size_t)in_point * channels + in_channel];
    }
    __syncthreads();
    // Store: the tile is read transposed, so threadIdx.x now walks the
    // destination's fast axis — the spatial index — and stays coalesced.
    uint32_t out_point = blockIdx.y * H3B_TILE + threadIdx.x;
    uint32_t out_channel = blockIdx.x * H3B_TILE + threadIdx.y;
    if (out_point < spatial && out_channel < channels) {
        float value = tile[threadIdx.x][threadIdx.y];
        if (has_bias) value += h3b_element<T>::to(bias[out_channel]);
        out[(size_t)out_channel * row_stride + row_offset + out_point] =
            h3b_element<T>::from(value);
    }
}

/* GroupNorm's first pass: one block per (group, group-local channel, chunk of
 * that channel's plane), writing an f32 (sum, sum of squares) pair. The
 * reduction is over the whole `(channels/groups, time, height, width)` volume,
 * which is what `nn.GroupNorm` normalizes over for a 5-D input. */
template <typename T>
__global__ static void h3b_norm_partial_kernel(float *partial, const T *in,
                                               uint32_t group_channels,
                                               size_t plane, uint32_t chunk,
                                               uint32_t chunks) {
    uint32_t group = blockIdx.z;
    uint32_t local = blockIdx.y;
    uint32_t index = blockIdx.x;
    size_t base = ((size_t)group * group_channels + local) * plane;
    size_t begin = (size_t)index * chunk;
    size_t count = plane - begin < chunk ? plane - begin : chunk;
    float sum = 0.0f;
    float squares = 0.0f;
    for (size_t offset = threadIdx.x; offset < count; offset += blockDim.x) {
        float value = h3b_element<T>::to(in[base + begin + offset]);
        sum += value;
        squares += value * value;
    }
    h3b_block_sum2(&sum, &squares);
    if (threadIdx.x == 0u) {
        size_t slot = ((size_t)group * group_channels + local) * chunks + index;
        partial[slot * 2] = sum;
        partial[slot * 2 + 1] = squares;
    }
}

/* GroupNorm's second pass: one block per group, mean and rstd in f32. */
__global__ static void h3b_norm_stats_kernel(float *mean, float *rstd,
                                             const float *partial,
                                             uint32_t group_channels,
                                             uint32_t chunks, float count,
                                             float epsilon) {
    size_t entries = (size_t)group_channels * chunks;
    float sum = 0.0f;
    float squares = 0.0f;
    for (size_t index = threadIdx.x; index < entries; index += blockDim.x) {
        size_t slot = (size_t)blockIdx.x * entries + index;
        sum += partial[slot * 2];
        squares += partial[slot * 2 + 1];
    }
    h3b_block_sum2(&sum, &squares);
    if (threadIdx.x == 0u) {
        float average = sum / count;
        float variance = squares / count - average * average;
        if (variance < 0.0f) variance = 0.0f;
        mean[blockIdx.x] = average;
        rstd[blockIdx.x] = rsqrtf(variance + epsilon);
    }
}

/* GroupNorm's third pass, with the ResBlock's two epilogues folded in:
 * `h = out_norm(h) * (1 + scale) + shift` (`:185`, scale and shift are the
 * halves of the embedding projection) and, where the graph has it, the SiLU
 * inside `in_layers`/`out_layers` (`:165`, `:171`). */
template <typename T>
__global__ static void h3b_norm_apply_kernel(
    T *out, const T *in, const T *weight, const T *bias, const T *emb,
    const float *mean, const float *rstd, uint32_t channels,
    uint32_t group_channels, size_t plane, int silu, int has_emb,
    size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = (uint32_t)(index / plane);
    uint32_t group = channel / group_channels;
    float value = (h3b_element<T>::to(in[index]) - mean[group]) *
                      rstd[group] * h3b_element<T>::to(weight[channel]) +
                  h3b_element<T>::to(bias[channel]);
    if (has_emb) {
        value = value * (1.0f + h3b_element<T>::to(emb[channel])) +
                h3b_element<T>::to(emb[channels + channel]);
    }
    if (silu) value = value / (1.0f + expf(-value));
    out[index] = h3b_element<T>::from(value);
}

template <typename T>
__global__ static void h3b_silu_kernel(T *out, const T *in, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = h3b_element<T>::to(in[index]);
    out[index] = h3b_element<T>::from(value / (1.0f + expf(-value)));
}

/* `inputs + out_proj(...)` (`model.py:22`) and `self.skip(x) + h`
 * (`:189`): one rounding, as torch's own bf16 add. */
template <typename T>
__global__ static void h3b_add_kernel(T *out, const T *left, const T *right,
                                      size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    out[index] = h3b_element<T>::from(h3b_element<T>::to(left[index]) +
                                      h3b_element<T>::to(right[index]));
}

/* The depthwise temporal convolution both models carry: kernel
 * `(kernel, 1, 1)`, `groups = channels`, symmetric padding, out-of-range
 * frames contributing nothing (`TemporalConv`, `:191-213`; the adapter's
 * `temporal`, `model.py:15`). */
template <typename T>
__global__ static void h3b_depthwise_temporal_kernel(
    T *out, const T *in, const T *weight, const T *bias, uint32_t rows,
    uint32_t spatial, uint32_t kernel, uint32_t pad, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t point = (uint32_t)(index % spatial);
    size_t rest = index / spatial;
    uint32_t row = (uint32_t)(rest % rows);
    uint32_t channel = (uint32_t)(rest / rows);
    float sum = h3b_element<T>::to(bias[channel]);
    for (uint32_t tap = 0; tap < kernel; tap++) {
        int source = (int)row + (int)tap - (int)pad;
        if (source < 0 || source >= (int)rows) continue;
        sum += h3b_element<T>::to(
                   in[((size_t)channel * rows + source) * spatial + point]) *
               h3b_element<T>::to(weight[(size_t)channel * kernel + tap]);
    }
    out[index] = h3b_element<T>::from(sum);
}

/* `F.interpolate(..., mode="trilinear", align_corners=False)` (`:327`):
 * `src = scale * (dst + 0.5) - 0.5` clamped at zero, the two taps per axis,
 * f32 weights and one rounding at the end. */
template <typename T>
__global__ static void h3b_resize3d_kernel(
    T *out, const T *in, uint32_t channels, uint32_t rows, uint32_t height,
    uint32_t width, uint32_t out_rows, uint32_t out_height, uint32_t out_width,
    size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t x = (uint32_t)(index % out_width);
    size_t rest = index / out_width;
    uint32_t y = (uint32_t)(rest % out_height);
    rest /= out_height;
    uint32_t row = (uint32_t)(rest % out_rows);
    uint32_t channel = (uint32_t)(rest / out_rows);

    float row_scale = (float)rows / (float)out_rows;
    float height_scale = (float)height / (float)out_height;
    float width_scale = (float)width / (float)out_width;

    float source_t = row_scale * ((float)row + 0.5f) - 0.5f;
    float source_y = height_scale * ((float)y + 0.5f) - 0.5f;
    float source_x = width_scale * ((float)x + 0.5f) - 0.5f;
    if (source_t < 0.0f) source_t = 0.0f;
    if (source_y < 0.0f) source_y = 0.0f;
    if (source_x < 0.0f) source_x = 0.0f;

    uint32_t t0 = (uint32_t)source_t;
    uint32_t y0 = (uint32_t)source_y;
    uint32_t x0 = (uint32_t)source_x;
    if (t0 > rows - 1) t0 = rows - 1;
    if (y0 > height - 1) y0 = height - 1;
    if (x0 > width - 1) x0 = width - 1;
    uint32_t t1 = t0 + 1 < rows ? t0 + 1 : t0;
    uint32_t y1 = y0 + 1 < height ? y0 + 1 : y0;
    uint32_t x1 = x0 + 1 < width ? x0 + 1 : x0;

    float frac_t = source_t - (float)t0;
    float frac_y = source_y - (float)y0;
    float frac_x = source_x - (float)x0;

    const T *plane = in + (size_t)channel * rows * height * width;
    float value = 0.0f;
    for (uint32_t ti = 0; ti < 2; ti++) {
        uint32_t t = ti ? t1 : t0;
        float weight_t = ti ? frac_t : 1.0f - frac_t;
        for (uint32_t yi = 0; yi < 2; yi++) {
            uint32_t row_index = yi ? y1 : y0;
            float weight_y = yi ? frac_y : 1.0f - frac_y;
            const T *line = plane + ((size_t)t * height + row_index) * width;
            for (uint32_t xi = 0; xi < 2; xi++) {
                float weight_x = xi ? frac_x : 1.0f - frac_x;
                value += weight_t * weight_y * weight_x *
                         h3b_element<T>::to(line[xi ? x1 : x0]);
            }
        }
    }
    out[index] = h3b_element<T>::from(value);
}

/* One frame per target from a source frame, with `H3B_UNOCCUPIED` writing
 * zero: `geometry.py:60-79`'s nearest pack (`slots` source frames per target)
 * and the replicate padding the chunked upscaler reads (`:271`). */
template <typename T>
__global__ static void h3b_time_gather_kernel(
    T *out, const T *in, const uint32_t *index, uint32_t in_channels,
    uint32_t source_frames, uint32_t target_frames, uint32_t spatial,
    uint32_t slots, size_t elements) {
    size_t element = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= elements) return;
    uint32_t point = (uint32_t)(element % spatial);
    size_t rest = element / spatial;
    uint32_t frame = (uint32_t)(rest % target_frames);
    uint32_t out_channel = (uint32_t)(rest / target_frames);
    uint32_t slot = out_channel / in_channels;
    uint32_t channel = out_channel % in_channels;
    uint32_t source = index[(size_t)frame * slots + slot];
    if (source == H3B_UNOCCUPIED) {
        out[element] = h3b_element<T>::from(0.0f);
        return;
    }
    out[element] =
        in[((size_t)channel * source_frames + source) * spatial + point];
}

/* `geometry.py:44-51`'s `temporal_resample`: `a + w * (b - a)` in f32. */
template <typename T>
__global__ static void h3b_time_lerp_kernel(
    T *out, const T *in, const uint32_t *left, const uint32_t *right,
    const float *weight, uint32_t channels, uint32_t source_frames,
    uint32_t target_frames, uint32_t spatial, size_t elements) {
    size_t element = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= elements) return;
    uint32_t point = (uint32_t)(element % spatial);
    size_t rest = element / spatial;
    uint32_t frame = (uint32_t)(rest % target_frames);
    uint32_t channel = (uint32_t)(rest / target_frames);
    float near = h3b_element<T>::to(
        in[((size_t)channel * source_frames + left[frame]) * spatial + point]);
    float far = h3b_element<T>::to(
        in[((size_t)channel * source_frames + right[frame]) * spatial + point]);
    out[element] = h3b_element<T>::from(near + weight[frame] * (far - near));
}

/* `geometry.py:105-112`: `F.pixel_unshuffle(..., 2)` over each latent frame's
 * `(height, width)`, giving the adapter its 384 channels. */
template <typename T>
__global__ static void h3b_unshuffle2_kernel(T *out, const T *in,
                                             uint32_t rows, uint32_t height,
                                             uint32_t width, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t out_width = width / 2;
    uint32_t out_height = height / 2;
    uint32_t x = (uint32_t)(index % out_width);
    size_t rest = index / out_width;
    uint32_t y = (uint32_t)(rest % out_height);
    rest /= out_height;
    uint32_t row = (uint32_t)(rest % rows);
    uint32_t out_channel = (uint32_t)(rest / rows);
    uint32_t channel = out_channel / 4;
    uint32_t sub = out_channel % 4;
    out[index] = in[(((size_t)channel * rows + row) * height +
                     (2u * y + sub / 2u)) * width + (2u * x + sub % 2u)];
}

/* `forward` (`:302-310`)'s weighted accumulation and its weighted average. */
template <typename T>
__global__ static void h3b_accumulate_kernel(
    float *accumulator, const T *segment, const float *weight, uint32_t frames,
    uint32_t segment_frames, uint32_t spatial, uint32_t in_frames,
    uint32_t in_begin, uint32_t out_begin, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t point = (uint32_t)(index % spatial);
    size_t rest = index / spatial;
    uint32_t frame = (uint32_t)(rest % segment_frames);
    uint32_t channel = (uint32_t)(rest / segment_frames);
    size_t source =
        ((size_t)channel * in_frames + (in_begin + frame)) * spatial + point;
    accumulator[((size_t)channel * frames + (out_begin + frame)) * spatial +
                point] += h3b_element<T>::to(segment[source]) * weight[frame];
}

template <typename T>
__global__ static void h3b_blend_finish_kernel(T *out,
                                               const float *accumulator,
                                               const float *weight_total,
                                               uint32_t frames, uint32_t spatial,
                                               size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t frame = (uint32_t)((index / spatial) % frames);
    float total = weight_total[frame];
    if (total < 1e-8f) total = 1e-8f;
    out[index] = h3b_element<T>::from(accumulator[index] / total);
}

/* `model.py:21`'s `value * self.activation(gate)` over the `in_proj` halves,
 * which `chunk(2, dim=1)` splits at `gate_offset` elements. */
template <typename T>
__global__ static void h3b_gate_mul_kernel(T *out, const T *in,
                                           size_t gate_offset,
                                           size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float gate = h3b_element<T>::to(in[gate_offset + index]);
    gate = gate / (1.0f + expf(-gate));
    out[index] = h3b_element<T>::from(h3b_element<T>::to(in[index]) * gate);
}

/* The node's `N` and `D` (`:586-592`): `(x - mean) / std` and `x * std + mean`,
 * with the statistics in the same dtype the reference keeps them in. */
template <typename T>
__global__ static void h3b_scale_kernel(T *out, const T *in, const T *mean,
                                        const T *scale, uint32_t channels,
                                        size_t plane, int denormalize,
                                        size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = (uint32_t)(index / plane);
    float value = h3b_element<T>::to(in[index]);
    float center = h3b_element<T>::to(mean[channel]);
    float factor = h3b_element<T>::to(scale[channel]);
    // `(x - mean) / std` and `x * std + mean`, the node's own two expressions
    // (`minimax_h3_latent_upscaler_3d.py:586,592`): a division one way, a
    // multiplication the other, each rounding once.
    out[index] = h3b_element<T>::from(
        denormalize ? value * factor + center : (value - center) / factor);
}

/* --- the host entry points, one per op -------------------------------- */

static int h3b_type(cudaDataType_t *type, int dtype) {
    if (dtype == H3D_BF16) {
        *type = CUDA_R_16BF;
        return 1;
    }
    if (dtype == H3D_F32) {
        *type = CUDA_R_32F;
        return 1;
    }
    return 0;
}

static unsigned h3b_blocks(size_t elements) {
    return (unsigned)((elements + H3B_BLOCK - 1) / H3B_BLOCK);
}

/* The convolution's two halves, as cuBLASLt calls them. `h3b_gemm_rows` is the
 * im2col form: A `(rows, k)` and B `(out, k)` both row-major — B is PyTorch's
 * weight untouched — into an f32 C, one accumulation, no intermediate
 * rounding. `h3b_gemm_channels` is the 1x1 form: A is the channel-major
 * activation `(k, spatial)` read column-major and B the weight `(out, k)` read
 * column-major, so a 1x1 convolution copies nothing at all. */
static int h3b_gemm_rows(h3d_context *context, float *out, const void *a,
                         const void *b, uint32_t rows, uint32_t input_dim,
                         uint32_t output_dim, cudaDataType_t type, char *error,
                         size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t a_layout = NULL;
    cublasLtMatrixLayout_t b_layout = NULL;
    cublasLtMatrixLayout_t c_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status =
        cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
            sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&a_layout, type, rows, input_dim,
                                            input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&b_layout, type, output_dim,
                                            input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, rows,
                                            output_dim, output_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(context->blas, operation, &alpha, a, a_layout,
                                b, b_layout, &beta, out, c_layout, out,
                                c_layout, NULL, NULL, 0, context->stream);
    }
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    return h3d_blas_ok(status, "cuBLASLt latent matmul", error, error_size);
}

/* The convolution's A operand as a dense matrix, which is what the planar
 * buffer makes possible: `a` is already offset to the tap's row of the padded
 * image, `rows` is one frame's output plane, `input_dim` is `kw*channels` (the
 * column taps of one channel block), and the leading dimension is `input_dim`
 * because consecutive image positions are exactly one matrix row apart.
 *
 * `accumulate` sets beta, so the taps of one convolution sum in f32 without a
 * second pass over the result: the same arithmetic the materialized im2col
 * performed in one K = channels*kd*kh*kw reduction, in `kd*kh` of them.
 *
 * The algorithm comes from the heuristic where one with no workspace exists —
 * the default `NULL` asks the library the same question, so the difference is
 * only which of the two answers is used — and stays `NULL` otherwise. */
/* Whether the planar matmul asks the heuristic for an algorithm. Measured on
 * the released shapes: the heuristic returns the same kernel the library picks
 * by itself (`H3_LTX_GEMM_ALGO=heuristic` reproduces the mix exactly), so the
 * explicit per-call query is off unless asked for. */
static int h3b_gemm_query(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *setting = getenv("H3_LTX_GEMM_ALGO");
        enabled = setting && strcmp(setting, "heuristic") == 0;
    }
    return enabled;
}

static int h3b_gemm_planar(h3d_context *context, float *out, const void *a,
                           const void *b, uint32_t rows, uint32_t input_dim,
                           uint32_t output_dim, int accumulate,
                           cudaDataType_t type, char *error,
                           size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t a_layout = NULL;
    cublasLtMatrixLayout_t b_layout = NULL;
    cublasLtMatrixLayout_t c_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    cublasLtMatmulAlgo_t algorithm;
    const cublasLtMatmulAlgo_t *chosen = NULL;
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    cublasStatus_t status =
        cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
            sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&a_layout, type, rows, input_dim,
                                            input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&b_layout, type, output_dim,
                                            input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, rows,
                                            output_dim, output_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS && h3b_gemm_query()) {
        cublasLtMatmulPreference_t preference = NULL;
        size_t workspace = 0;
        if (cublasLtMatmulPreferenceCreate(&preference) ==
            CUBLAS_STATUS_SUCCESS) {
            cublasLtMatmulHeuristicResult_t heuristic;
            int returned = 0;
            cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace, sizeof(workspace));
            if (cublasLtMatmulAlgoGetHeuristic(
                    context->blas, operation, a_layout, b_layout, c_layout,
                    c_layout, preference, 1, &heuristic, &returned) ==
                    CUBLAS_STATUS_SUCCESS &&
                returned == 1 && heuristic.workspaceSize == 0) {
                algorithm = heuristic.algo;
                chosen = &algorithm;
            }
            cublasLtMatmulPreferenceDestroy(preference);
        }
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(context->blas, operation, &alpha, a, a_layout,
                                b, b_layout, &beta, out, c_layout, out,
                                c_layout, chosen, NULL, 0, context->stream);
    }
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    return h3d_blas_ok(status, "cuBLASLt latent planar matmul", error,
                       error_size);
}

static int h3b_gemm_channels(h3d_context *context, float *out,
                             const void *input, const void *weight,
                             uint32_t spatial, uint32_t input_dim,
                             uint32_t output_dim, cudaDataType_t type,
                             char *error, size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t a_layout = NULL;
    cublasLtMatrixLayout_t b_layout = NULL;
    cublasLtMatrixLayout_t c_layout = NULL;
    cublasLtOrder_t column_major = CUBLASLT_ORDER_COL;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status =
        cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&a_layout, type, spatial, input_dim,
                                            spatial);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&b_layout, type, input_dim,
                                            output_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, spatial,
                                            output_dim, output_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &column_major,
            sizeof(column_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &column_major,
            sizeof(column_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(context->blas, operation, &alpha, input,
                                a_layout, weight, b_layout, &beta, out,
                                c_layout, out, c_layout, NULL, NULL, 0,
                                context->stream);
    }
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    return h3d_blas_ok(status, "cuBLASLt latent channel matmul", error,
                       error_size);
}

int h3b_pad3d(h3d_context *context, h3d_tensor *output,
              const h3d_tensor *input, int dtype, uint32_t channels,
              uint32_t depth, uint32_t height, uint32_t width, uint32_t pad_t,
              uint32_t pad_h, uint32_t pad_w, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t inputs = 0, outputs = 0;
    if (!context || !channels || !depth || !height || !width ||
        !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * depth * height, width, &inputs) ||
        !h3d_mul_size((size_t)channels * (depth + 2 * pad_t) *
                          (height + 2 * pad_h),
                      width + 2 * pad_w, &outputs) ||
        !h3d_is(input, dtype, inputs) || !h3d_is(output, dtype, outputs)) {
        return h3d_set_error(error, error_size, "invalid latent pad tensors");
    }
    if (outputs == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_pad3d_kernel<float><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                  context->stream>>>(
            (float *)output->data, (const float *)input->data, channels, depth,
            height, width, pad_t, pad_h, pad_w, outputs);
    } else {
        h3b_pad3d_kernel<__nv_bfloat16><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                          context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, channels, depth, height,
            width, pad_t, pad_h, pad_w, outputs);
    }
    return h3d_launch_ok("latent pad", error, error_size);
}

int h3b_im2col(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *padded, int dtype, uint32_t channels,
               uint32_t pad_depth, uint32_t pad_height, uint32_t pad_width,
               uint32_t kd, uint32_t kh, uint32_t kw, uint32_t depth,
               uint32_t height, uint32_t width, uint32_t t_begin, char *error,
               size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t sources = 0, outputs = 0;
    if (!context || !channels || !kd || !kh || !kw || !depth || !height ||
        !width || t_begin + depth > pad_depth || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * pad_depth * pad_height, pad_width,
                      &sources) ||
        !h3d_mul_size((size_t)depth * height * width,
                      (size_t)channels * kd * kh * kw, &outputs) ||
        !h3d_is(padded, dtype, sources) || !h3d_is(output, dtype, outputs)) {
        return h3d_set_error(error, error_size, "invalid latent im2col shape");
    }
    if (outputs == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_im2col_kernel<float><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                   context->stream>>>(
            (float *)output->data, (const float *)padded->data, channels,
            pad_depth, pad_height, pad_width, kd, kh, kw, depth, height, width,
            t_begin, outputs);
    } else {
        h3b_im2col_kernel<__nv_bfloat16><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                           context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)padded->data, channels, pad_depth,
            pad_height, pad_width, kd, kh, kw, depth, height, width, t_begin,
            outputs);
    }
    return h3d_launch_ok("latent im2col", error, error_size);
}

int h3b_planar(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, int dtype, uint32_t channels,
               uint32_t depth, uint32_t height, uint32_t width, uint32_t kw,
               uint32_t pad_t, uint32_t pad_h, uint32_t pad_w,
               uint32_t out_width, int replicate_t, char *error,
               size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t sources = 0, outputs = 0;
    uint32_t pad_depth = depth + 2 * pad_t;
    uint32_t pad_height = height + 2 * pad_h;
    // The A view `h3b_matmul_planar` takes is only valid for a same-shape
    // kernel, where every output row of the padded image is a row of the
    // buffer; anything else would need the halo folded back in.
    if (out_width + kw - 1 != width + 2 * pad_w) {
        return h3d_set_error(error, error_size, "latent planar shape is not a same-shape convolution");
    }
    if (!context || !channels || !depth || !height || !width || !kw ||
        !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * depth * height, width, &sources) ||
        !h3d_mul_size((size_t)channels * pad_depth * pad_height,
                      (size_t)out_width * kw, &outputs) ||
        !h3d_is(input, dtype, sources) || !h3d_is(output, dtype, outputs)) {
        return h3d_set_error(error, error_size, "invalid latent planar shape");
    }
    if (outputs == 0) return 1;
    uint32_t columns = (channels + H3B_PLANAR_CHANNELS - 1) / H3B_PLANAR_CHANNELS;
    dim3 grid(pad_depth * pad_height, columns);
    size_t shared = (size_t)H3B_PLANAR_CHANNELS * (width + 2 * pad_w + 1) *
                    sizeof(float);
    if (dtype == H3D_F32) {
        h3b_planar_kernel<float><<<grid, H3B_BLOCK, shared, context->stream>>>(
            (float *)output->data, (const float *)input->data, channels, depth,
            height, width, kw, pad_t, pad_h, pad_w, pad_depth, pad_height,
            out_width, replicate_t);
    } else {
        h3b_planar_kernel<__nv_bfloat16>
            <<<grid, H3B_BLOCK, shared, context->stream>>>(
                (__nv_bfloat16 *)output->data,
                (const __nv_bfloat16 *)input->data, channels, depth, height,
                width, kw, pad_t, pad_h, pad_w, pad_depth, pad_height,
                out_width, replicate_t);
    }
    return h3d_launch_ok("latent planar", error, error_size);
}

int h3b_weight_pack(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *weight, int dtype, uint32_t out_channels,
                    uint32_t in_channels, uint32_t taps, uint32_t kw,
                    char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t elements = 0;
    if (!context || !out_channels || !in_channels || !taps || !kw ||
        taps % kw || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)out_channels * in_channels, taps, &elements) ||
        !h3d_is(weight, dtype, elements) ||
        !h3d_is(output, dtype, elements)) {
        return h3d_set_error(error, error_size, "invalid latent pack shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_pack_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                 context->stream>>>(
            (float *)output->data, (const float *)weight->data, out_channels,
            in_channels, taps, kw, elements);
    } else {
        h3b_pack_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                         context->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)weight->data,
            out_channels, in_channels, taps, kw, elements);
    }
    return h3d_launch_ok("latent weight pack", error, error_size);
}

int h3b_matmul_planar(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *a, size_t a_offset,
                      const h3d_tensor *b, size_t b_offset, size_t c_offset,
                      uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                      int accumulate, int dtype, char *error,
                      size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t ae = 0, be = 0, ce = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        !h3b_type(&type, dtype) ||
        !h3d_mul_size(rows, input_dim, &ae) ||
        !h3d_mul_size(output_dim, input_dim, &be) ||
        !h3d_mul_size(rows, output_dim, &ce) ||
        !h3d_is(a, dtype, a_offset + ae) || !h3d_is(b, dtype, b_offset + be) ||
        !h3d_is(output, H3D_F32, c_offset + ce)) {
        return h3d_set_error(error, error_size, "invalid latent planar matmul shape");
    }
    return h3b_gemm_planar(context, (float *)output->data + c_offset,
                           (const char *)a->data +
                               a_offset * h3d_dtype_size(dtype),
                           (const char *)b->data +
                               b_offset * h3d_dtype_size(dtype),
                           rows, input_dim, output_dim, accumulate, type, error,
                           error_size);
}

int h3b_matmul(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *left, const h3d_tensor *right, uint32_t rows,
               uint32_t input_dim, uint32_t output_dim, int dtype, char *error,
               size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t lefts = 0, rights = 0, outputs = 0;
    if (!context || !rows || !input_dim || !output_dim ||
        !h3b_type(&type, dtype) ||
        !h3d_mul_size(rows, input_dim, &lefts) ||
        !h3d_mul_size(output_dim, input_dim, &rights) ||
        !h3d_mul_size(rows, output_dim, &outputs) ||
        !h3d_is(left, dtype, lefts) || !h3d_is(right, dtype, rights) ||
        !h3d_is(output, H3D_F32, outputs)) {
        return h3d_set_error(error, error_size, "invalid latent matmul shape");
    }
    return h3b_gemm_rows(context, (float *)output->data, left->data,
                         right->data, rows, input_dim, output_dim, type, error,
                         error_size);
}

int h3b_matmul_channels(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        uint32_t spatial, uint32_t input_dim,
                        uint32_t output_dim, int dtype, char *error,
                        size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t inputs = 0, weights = 0, outputs = 0;
    if (!context || !spatial || !input_dim || !output_dim ||
        !h3b_type(&type, dtype) ||
        !h3d_mul_size(input_dim, spatial, &inputs) ||
        !h3d_mul_size(output_dim, input_dim, &weights) ||
        !h3d_mul_size(spatial, output_dim, &outputs) ||
        !h3d_is(input, dtype, inputs) || !h3d_is(weight, dtype, weights) ||
        !h3d_is(output, H3D_F32, outputs)) {
        return h3d_set_error(error, error_size, "invalid latent 1x1 shape");
    }
    return h3b_gemm_channels(context, (float *)output->data, input->data,
                             weight->data, spatial, input_dim, output_dim, type,
                             error, error_size);
}

int h3b_epilogue(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *matrix, const h3d_tensor *bias, int dtype,
                 uint32_t channels, uint32_t spatial, size_t row_stride,
                 size_t row_offset, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t matrices = 0, needed = 0;
    if (!context || !channels || !spatial || !h3b_type(&type, dtype) ||
        !h3d_mul_size(spatial, channels, &matrices) ||
        !h3d_mul_size((size_t)channels - 1, row_stride, &needed) ||
        needed > SIZE_MAX - row_offset - spatial ||
        !h3d_is(matrix, H3D_F32, matrices) ||
        !h3d_is(output, dtype, needed + row_offset + spatial) ||
        (bias && !h3d_is(bias, dtype, channels))) {
        return h3d_set_error(error, error_size, "invalid latent epilogue shape");
    }
    dim3 grid((channels + H3B_TILE - 1) / H3B_TILE,
              (spatial + H3B_TILE - 1) / H3B_TILE);
    dim3 block(H3B_TILE, H3B_TILE);
    if (dtype == H3D_F32) {
        h3b_epilogue_kernel<float><<<grid, block, 0, context->stream>>>(
            (float *)output->data, (const float *)matrix->data,
            bias ? (const float *)bias->data : (const float *)matrix->data,
            spatial, channels, row_stride, row_offset, bias != NULL);
    } else {
        h3b_epilogue_kernel<__nv_bfloat16><<<grid, block, 0,
                                             context->stream>>>(
            (__nv_bfloat16 *)output->data, (const float *)matrix->data,
            bias ? (const __nv_bfloat16 *)bias->data
                 : (const __nv_bfloat16 *)matrix->data,
            spatial, channels, row_stride, row_offset, bias != NULL);
    }
    return h3d_launch_ok("latent epilogue", error, error_size);
}

/* Elements per partial-sum block: 8,192 gives the released shapes 19 blocks per
 * channel on the upscaler and 3 on the adapter, which is enough to fill the
 * device without making the second pass long. */
#define H3B_NORM_CHUNK 8192u

int h3b_norm(h3d_context *context, h3d_tensor *output, const h3d_tensor *input,
             const h3d_tensor *weight, const h3d_tensor *bias,
             const h3d_tensor *embedding, h3d_tensor *partial,
             h3d_tensor *mean, h3d_tensor *rstd, int dtype, uint32_t channels,
             uint32_t groups, uint32_t rows, uint32_t spatial, float epsilon,
             int silu, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t elements = 0, plane = 0, partials = 0;
    uint32_t group_channels = channels / groups;
    uint32_t chunks = 0;
    if (!context || !channels || !groups || !rows || !spatial ||
        channels % groups || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * rows, spatial, &elements) ||
        !h3d_mul_size(rows, spatial, &plane) ||
        !h3d_is(input, dtype, elements) || !h3d_is(output, dtype, elements) ||
        !h3d_is(weight, dtype, channels) || !h3d_is(bias, dtype, channels) ||
        (embedding && !h3d_is(embedding, dtype, 2 * (size_t)channels)) ||
        !h3d_is(mean, H3D_F32, groups) || !h3d_is(rstd, H3D_F32, groups)) {
        return h3d_set_error(error, error_size, "invalid latent norm shape");
    }
    chunks = (uint32_t)((plane + H3B_NORM_CHUNK - 1) / H3B_NORM_CHUNK);
    if (!h3d_mul_size((size_t)groups * group_channels * chunks, 2, &partials) ||
        !h3d_is(partial, H3D_F32, partials)) {
        return h3d_set_error(error, error_size,
                             "latent norm scratch is undersized");
    }
    if (group_channels > 65535 || groups > 65535) {
        return h3d_set_error(error, error_size, "latent norm grid too wide");
    }

    dim3 grid(chunks, group_channels, groups);
    if (dtype == H3D_F32) {
        h3b_norm_partial_kernel<float><<<grid, H3B_BLOCK, 0,
                                         context->stream>>>(
            (float *)partial->data, (const float *)input->data,
            group_channels, plane, H3B_NORM_CHUNK, chunks);
    } else {
        h3b_norm_partial_kernel<__nv_bfloat16><<<grid, H3B_BLOCK, 0,
                                                 context->stream>>>(
            (float *)partial->data, (const __nv_bfloat16 *)input->data,
            group_channels, plane, H3B_NORM_CHUNK, chunks);
    }
    if (!h3d_launch_ok("latent norm partial", error, error_size)) return 0;

    h3b_norm_stats_kernel<<<groups, H3B_BLOCK, 0, context->stream>>>(
        (float *)mean->data, (float *)rstd->data, (const float *)partial->data,
        group_channels, chunks, (float)(plane * group_channels), epsilon);
    if (!h3d_launch_ok("latent norm statistics", error, error_size)) return 0;

    if (dtype == H3D_F32) {
        h3b_norm_apply_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                       context->stream>>>(
            (float *)output->data, (const float *)input->data,
            (const float *)weight->data, (const float *)bias->data,
            embedding ? (const float *)embedding->data
                      : (const float *)weight->data,
            (const float *)mean->data, (const float *)rstd->data, channels,
            group_channels, plane, silu, embedding != NULL, elements);
    } else {
        h3b_norm_apply_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK,
                                               0, context->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)weight->data,
            (const __nv_bfloat16 *)bias->data,
            embedding ? (const __nv_bfloat16 *)embedding->data
                      : (const __nv_bfloat16 *)weight->data,
            (const float *)mean->data, (const float *)rstd->data, channels,
            group_channels, plane, silu, embedding != NULL, elements);
    }
    return h3d_launch_ok("latent norm apply", error, error_size);
}

int h3b_silu(h3d_context *context, h3d_tensor *output, const h3d_tensor *input,
             int dtype, size_t elements, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    if (!context || !h3b_type(&type, dtype) || !h3d_is(input, dtype, elements) ||
        !h3d_is(output, dtype, elements)) {
        return h3d_set_error(error, error_size, "invalid latent SiLU shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_silu_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                 context->stream>>>(
            (float *)output->data, (const float *)input->data, elements);
    } else {
        h3b_silu_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                         context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, elements);
    }
    return h3d_launch_ok("latent SiLU", error, error_size);
}

int h3b_add(h3d_context *context, h3d_tensor *output, const h3d_tensor *left,
            const h3d_tensor *right, int dtype, size_t elements, char *error,
            size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    if (!context || !h3b_type(&type, dtype) || !h3d_is(left, dtype, elements) ||
        !h3d_is(right, dtype, elements) || !h3d_is(output, dtype, elements)) {
        return h3d_set_error(error, error_size, "invalid latent add shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_add_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                context->stream>>>(
            (float *)output->data, (const float *)left->data,
            (const float *)right->data, elements);
    } else {
        h3b_add_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                        context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)left->data,
            (const __nv_bfloat16 *)right->data, elements);
    }
    return h3d_launch_ok("latent add", error, error_size);
}

int h3b_depthwise_temporal(h3d_context *context, h3d_tensor *output,
                           const h3d_tensor *input, const h3d_tensor *weight,
                           const h3d_tensor *bias, int dtype, uint32_t channels,
                           uint32_t rows, uint32_t spatial, uint32_t kernel,
                           uint32_t pad, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t elements = 0;
    if (!context || !channels || !rows || !spatial || !kernel ||
        kernel % 2 == 0 || pad != kernel / 2 || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * rows, spatial, &elements) ||
        !h3d_is(input, dtype, elements) || !h3d_is(output, dtype, elements) ||
        !h3d_is(weight, dtype, (size_t)channels * kernel) ||
        !h3d_is(bias, dtype, channels)) {
        return h3d_set_error(error, error_size,
                             "invalid latent depthwise shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_depthwise_temporal_kernel<float><<<h3b_blocks(elements), H3B_BLOCK,
                                               0, context->stream>>>(
            (float *)output->data, (const float *)input->data,
            (const float *)weight->data, (const float *)bias->data, rows,
            spatial, kernel, pad, elements);
    } else {
        h3b_depthwise_temporal_kernel<__nv_bfloat16>
            <<<h3b_blocks(elements), H3B_BLOCK, 0, context->stream>>>(
                (__nv_bfloat16 *)output->data,
                (const __nv_bfloat16 *)input->data,
                (const __nv_bfloat16 *)weight->data,
                (const __nv_bfloat16 *)bias->data, rows, spatial, kernel, pad,
                elements);
    }
    return h3d_launch_ok("latent depthwise temporal", error, error_size);
}

int h3b_resize3d(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, int dtype, uint32_t channels,
                 uint32_t rows, uint32_t height, uint32_t width,
                 uint32_t out_rows, uint32_t out_height, uint32_t out_width,
                 char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t inputs = 0, outputs = 0;
    if (!context || !channels || !rows || !height || !width || !out_rows ||
        !out_height || !out_width || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * rows * height, width, &inputs) ||
        !h3d_mul_size((size_t)channels * out_rows * out_height, out_width,
                      &outputs) ||
        !h3d_is(input, dtype, inputs) || !h3d_is(output, dtype, outputs)) {
        return h3d_set_error(error, error_size, "invalid latent resize shape");
    }
    if (outputs == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_resize3d_kernel<float><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                     context->stream>>>(
            (float *)output->data, (const float *)input->data, channels, rows,
            height, width, out_rows, out_height, out_width, outputs);
    } else {
        h3b_resize3d_kernel<__nv_bfloat16><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                             context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, channels, rows, height, width,
            out_rows, out_height, out_width, outputs);
    }
    return h3d_launch_ok("latent trilinear resize", error, error_size);
}

int h3b_time_gather(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, const h3d_tensor *index, int dtype,
                    uint32_t in_channels, uint32_t source_frames,
                    uint32_t target_frames, uint32_t spatial, uint32_t slots,
                    char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t inputs = 0, outputs = 0;
    if (!context || !in_channels || !source_frames || !target_frames ||
        !spatial || !slots || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)in_channels * source_frames, spatial, &inputs) ||
        !h3d_mul_size((size_t)slots * in_channels * target_frames, spatial,
                      &outputs) ||
        !h3d_is(input, dtype, inputs) || !h3d_is(output, dtype, outputs) ||
        !h3d_is(index, H3D_U32, (size_t)target_frames * slots)) {
        return h3d_set_error(error, error_size, "invalid latent gather shape");
    }
    if (outputs == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_time_gather_kernel<float><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                        context->stream>>>(
            (float *)output->data, (const float *)input->data,
            (const uint32_t *)index->data, in_channels, source_frames,
            target_frames, spatial, slots, outputs);
    } else {
        h3b_time_gather_kernel<__nv_bfloat16><<<h3b_blocks(outputs), H3B_BLOCK,
                                                0, context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data,
            (const uint32_t *)index->data, in_channels, source_frames,
            target_frames, spatial, slots, outputs);
    }
    return h3d_launch_ok("latent time gather", error, error_size);
}

int h3b_time_lerp(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *input, const h3d_tensor *left,
                  const h3d_tensor *right, const h3d_tensor *weight, int dtype,
                  uint32_t channels, uint32_t source_frames,
                  uint32_t target_frames, uint32_t spatial, char *error,
                  size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t inputs = 0, outputs = 0;
    if (!context || !channels || !source_frames || !target_frames || !spatial ||
        !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * source_frames, spatial, &inputs) ||
        !h3d_mul_size((size_t)channels * target_frames, spatial, &outputs) ||
        !h3d_is(input, dtype, inputs) || !h3d_is(output, dtype, outputs) ||
        !h3d_is(left, H3D_U32, target_frames) ||
        !h3d_is(right, H3D_U32, target_frames) ||
        !h3d_is(weight, H3D_F32, target_frames)) {
        return h3d_set_error(error, error_size, "invalid latent lerp shape");
    }
    if (outputs == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_time_lerp_kernel<float><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                      context->stream>>>(
            (float *)output->data, (const float *)input->data,
            (const uint32_t *)left->data, (const uint32_t *)right->data,
            (const float *)weight->data, channels, source_frames,
            target_frames, spatial, outputs);
    } else {
        h3b_time_lerp_kernel<__nv_bfloat16><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                              context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data,
            (const uint32_t *)left->data, (const uint32_t *)right->data,
            (const float *)weight->data, channels, source_frames,
            target_frames, spatial, outputs);
    }
    return h3d_launch_ok("latent time lerp", error, error_size);
}

int h3b_unshuffle2(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, int dtype, uint32_t in_channels,
                   uint32_t rows, uint32_t height, uint32_t width, char *error,
                   size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t inputs = 0, outputs = 0;
    if (!context || !in_channels || !rows || !height || !width ||
        height % 2 || width % 2 || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)in_channels * rows * height, width, &inputs) ||
        !h3d_mul_size((size_t)in_channels * 4 * rows * (height / 2), width / 2,
                      &outputs) ||
        !h3d_is(input, dtype, inputs) || !h3d_is(output, dtype, outputs)) {
        return h3d_set_error(error, error_size,
                             "invalid latent unshuffle shape");
    }
    if (outputs == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_unshuffle2_kernel<float><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                       context->stream>>>(
            (float *)output->data, (const float *)input->data, rows, height,
            width, outputs);
    } else {
        h3b_unshuffle2_kernel<__nv_bfloat16><<<h3b_blocks(outputs), H3B_BLOCK,
                                               0, context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, rows, height, width, outputs);
    }
    return h3d_launch_ok("latent unshuffle", error, error_size);
}

int h3b_accumulate(h3d_context *context, h3d_tensor *accumulator,
                   const h3d_tensor *segment, const h3d_tensor *weight,
                   int dtype, uint32_t channels, uint32_t frames,
                   uint32_t segment_frames, uint32_t spatial, uint32_t in_frames,
                   uint32_t in_begin, uint32_t out_begin, char *error,
                   size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t segments = 0, totals = 0;
    if (!context || !channels || !frames || !segment_frames || !spatial ||
        !in_frames || out_begin + segment_frames > frames ||
        in_begin + segment_frames > in_frames || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * in_frames, spatial, &segments) ||
        !h3d_mul_size((size_t)channels * frames, spatial, &totals) ||
        !h3d_is(segment, dtype, segments) ||
        !h3d_is(accumulator, H3D_F32, totals) ||
        !h3d_is(weight, H3D_F32, segment_frames)) {
        return h3d_set_error(error, error_size,
                             "invalid latent accumulate shape");
    }
    if (segment_frames == 0) return 1;
    size_t written = (size_t)channels * segment_frames * spatial;
    if (dtype == H3D_F32) {
        h3b_accumulate_kernel<float><<<h3b_blocks(written), H3B_BLOCK, 0,
                                       context->stream>>>(
            (float *)accumulator->data, (const float *)segment->data,
            (const float *)weight->data, frames, segment_frames, spatial,
            in_frames, in_begin, out_begin, written);
    } else {
        h3b_accumulate_kernel<__nv_bfloat16><<<h3b_blocks(written), H3B_BLOCK,
                                               0, context->stream>>>(
            (float *)accumulator->data,
            (const __nv_bfloat16 *)segment->data,
            (const float *)weight->data, frames, segment_frames, spatial,
            in_frames, in_begin, out_begin, written);
    }
    return h3d_launch_ok("latent accumulate", error, error_size);
}

int h3b_zero(h3d_context *context, h3d_tensor *tensor, char *error,
             size_t error_size) {
    if (!context || !tensor) {
        return h3d_set_error(error, error_size, "invalid zero tensor");
    }
    size_t bytes = tensor->elements * h3d_dtype_size(tensor->dtype);
    if (bytes == 0) return 1;
    if (!h3d_cuda_ok(cudaMemsetAsync(tensor->data, 0, bytes, context->stream),
                     "zero tensor", error, error_size)) {
        return 0;
    }
    return 1;
}

int h3b_blend_finish(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *accumulator, const h3d_tensor *weight,
                     int dtype, uint32_t channels, uint32_t frames,
                     uint32_t spatial, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t totals = 0;
    if (!context || !channels || !frames || !spatial ||
        !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * frames, spatial, &totals) ||
        !h3d_is(accumulator, H3D_F32, totals) ||
        !h3d_is(output, dtype, totals) || !h3d_is(weight, H3D_F32, frames)) {
        return h3d_set_error(error, error_size, "invalid latent blend shape");
    }
    if (totals == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_blend_finish_kernel<float><<<h3b_blocks(totals), H3B_BLOCK, 0,
                                         context->stream>>>(
            (float *)output->data, (const float *)accumulator->data,
            (const float *)weight->data, frames, spatial, totals);
    } else {
        h3b_blend_finish_kernel<__nv_bfloat16><<<h3b_blocks(totals), H3B_BLOCK,
                                                 0, context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const float *)accumulator->data, (const float *)weight->data,
            frames, spatial, totals);
    }
    return h3d_launch_ok("latent blend", error, error_size);
}

int h3b_gate_mul(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, int dtype, uint32_t half,
                 size_t plane, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t elements = 0, expected = 0;
    if (!context || !half || !plane || !h3b_type(&type, dtype) ||
        !h3d_mul_size(half, plane, &elements) ||
        !h3d_mul_size(2 * half, plane, &expected) ||
        !h3d_is(input, dtype, expected) || !h3d_is(output, dtype, elements)) {
        return h3d_set_error(error, error_size, "invalid latent gate shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_gate_mul_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                     context->stream>>>(
            (float *)output->data, (const float *)input->data, elements,
            elements);
    } else {
        h3b_gate_mul_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                             context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, elements, elements);
    }
    return h3d_launch_ok("latent gate multiply", error, error_size);
}

int h3b_scale(h3d_context *context, h3d_tensor *output, const h3d_tensor *input,
              const h3d_tensor *center, const h3d_tensor *scale, int dtype,
              uint32_t channels, size_t plane, int denormalize, char *error,
              size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t elements = 0;
    if (!context || !channels || !plane || !h3b_type(&type, dtype) ||
        !h3d_mul_size(channels, plane, &elements) ||
        !h3d_is(input, dtype, elements) || !h3d_is(output, dtype, elements) ||
        !h3d_is(center, dtype, channels) || !h3d_is(scale, dtype, channels)) {
        return h3d_set_error(error, error_size, "invalid latent scale shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_scale_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                  context->stream>>>(
            (float *)output->data, (const float *)input->data,
            (const float *)center->data, (const float *)scale->data, channels,
            plane, denormalize, elements);
    } else {
        h3b_scale_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                          context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)center->data,
            (const __nv_bfloat16 *)scale->data, channels, plane, denormalize,
            elements);
    }
    return h3d_launch_ok("latent scale", error, error_size);
}

/* `geometry.py:101`'s `torch.cat((linear, packed), dim=1)`: a plain copy into
 * an offset, the only piece of the alignment that needs one. */
template <typename T>
__global__ static void h3b_copy_offset_kernel(T *out, const T *in,
                                              size_t in_offset,
                                              size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    out[index] = in[in_offset + index];
}

int h3b_copy_offset(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, int dtype, size_t out_offset,
                    size_t in_offset, size_t elements, char *error,
                    size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    if (!context || !h3b_type(&type, dtype) ||
        !h3d_is(input, dtype, in_offset + elements) ||
        !h3d_is(output, dtype, out_offset + elements)) {
        return h3d_set_error(error, error_size, "invalid latent copy shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3b_copy_offset_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                        context->stream>>>(
            (float *)output->data + out_offset, (const float *)input->data,
            in_offset, elements);
    } else {
        h3b_copy_offset_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK,
                                                0, context->stream>>>(
            (__nv_bfloat16 *)output->data + out_offset,
            (const __nv_bfloat16 *)input->data, in_offset, elements);
    }
    return h3d_launch_ok("latent copy", error, error_size);
}

/* ------------------------------------------------------------------------
 * h3v_* — the LTX-2.5 video VAE decoder (`minimax_h3::ltx_vae`).
 *
 * The release decodes the refiner's latent with LTX's own ConvVideoDecoder
 * (`ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py`, read at
 * `/data/ltx2-pin/packages`, pin d1511477 = tag v1.2.0). Its convolutions are
 * the h3b planar path above with the artifact's own padding: 3x3x3, stride 1,
 * one zero frame of spatial halo, and one *replicated* frame each side in time
 * (`convolution.py:302-312`, `CausalConv3d` with `causal=False`). The four ops
 * below are the rest of the graph.
 * ------------------------------------------------------------------------ */

/* `PixelNorm` (`normalization.py:31-38`): `x / sqrt(mean(x^2, dim=1) + 1e-8)`
 * at every position. Torch's expression is five ops and five bf16 roundings -
 * the square, the mean, the epsilon add, the square root and the divide - and
 * the kernel rounds where the expression does, in that order.
 *
 * One thread per position, walking the channels: at every channel the block's
 * 256 threads read 256 consecutive positions, so every pass is coalesced. */
template <typename T>
__global__ static void h3v_pixel_norm_kernel(T *out, const T *in,
                                             uint32_t channels, size_t plane,
                                             float epsilon, size_t positions) {
    size_t position = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= positions) return;
    float total = 0.0f;
    for (uint32_t channel = 0; channel < channels; ++channel) {
        float value = h3b_element<T>::to(in[(size_t)channel * plane + position]);
        total += h3b_element<T>::to(h3b_element<T>::from(value * value));
    }
    float mean = h3b_element<T>::to(
        h3b_element<T>::from(total / (float)channels));
    float variance = h3b_element<T>::to(
        h3b_element<T>::from(mean + epsilon));
    float rms = h3b_element<T>::to(h3b_element<T>::from(sqrtf(variance)));
    for (uint32_t channel = 0; channel < channels; ++channel) {
        size_t at = (size_t)channel * plane + position;
        out[at] = h3b_element<T>::from(h3b_element<T>::to(in[at]) / rms);
    }
}

/* `PerChannelStatistics.un_normalize` (`ops.py:74-78`): `(x * std) + mean` per
 * channel, in bf16 - two ops and two roundings, which is why it is not
 * `h3b_scale`'s single-rounded form. The buffers arrive `.to(x)`
 * (`ops.py:75`), and the artifact stores them bf16 already. */
template <typename T>
__global__ static void h3v_denormalize_kernel(T *out, const T *in,
                                              const T *mean, const T *scale,
                                              uint32_t channels, size_t plane,
                                              size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = (uint32_t)(index / plane);
    float scaled = h3b_element<T>::to(
        h3b_element<T>::from(h3b_element<T>::to(in[index]) *
                             h3b_element<T>::to(scale[channel])));
    out[index] = h3b_element<T>::from(scaled +
                                      h3b_element<T>::to(mean[channel]));
}

/* `DepthToSpaceUpsample` (`sampling.py:66-124`): the convolution's output
 * pixel-shuffled by the block's stride, one frame dropped when the temporal
 * axis doubles (`:119-120`). The rearrange is
 * `(c p1 p2 p3) -> c (d p1) (h p2) (w p3)` (`:111-117`), so the source
 * channel is `((c p1 + a) p2 + b) p3 + g` for the temporal, height and width
 * sub-positions, and the output carries `p1*p2*p3` times fewer channels. */
template <typename T>
__global__ static void h3v_depth_to_space_kernel(
    T *out, const T *in, uint32_t in_channels, uint32_t depth, uint32_t height,
    uint32_t width, uint32_t p1, uint32_t p2, uint32_t p3, uint32_t drop,
    size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t out_depth = depth * p1 - drop;
    uint32_t out_height = height * p2;
    uint32_t out_width = width * p3;
    uint32_t x = (uint32_t)(index % out_width);
    size_t rest = index / out_width;
    uint32_t y = (uint32_t)(rest % out_height);
    rest /= out_height;
    uint32_t t = (uint32_t)(rest % out_depth);
    uint32_t channel = (uint32_t)(rest / out_depth);
    uint32_t g = x % p3, source_x = x / p3;
    uint32_t b = y % p2, source_y = y / p2;
    uint32_t a = (t + drop) % p1, source_t = (t + drop) / p1;
    uint32_t source = ((channel * p1 + a) * p2 + b) * p3 + g;
    out[index] = in[(((size_t)source * depth + source_t) * height + source_y) *
                        width +
                    source_x];
}

/* `unpatchify` at the artifact's patch size 4 and one temporal patch
 * (`ops.py:34-60`): the decoder's 48 output channels are `3 x 4 x 4` and the
 * rearrange `(c p r q) f h w -> c (f p) (h q) (w r)` reads source channel
 * `c p^2 + r p + q` for output row `h p + q` and column `w p + r`
 * (`:56-58`), with `p = 1`. */
template <typename T>
__global__ static void h3v_unpatchify_kernel(T *out, const T *in,
                                             uint32_t channels, uint32_t patch,
                                             uint32_t depth, uint32_t height,
                                             uint32_t width, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t out_height = height * patch;
    uint32_t out_width = width * patch;
    uint32_t x = (uint32_t)(index % out_width);
    size_t rest = index / out_width;
    uint32_t y = (uint32_t)(rest % out_height);
    rest /= out_height;
    uint32_t t = (uint32_t)(rest % depth);
    uint32_t channel = (uint32_t)(rest / depth);
    uint32_t q = y % patch, source_y = y / patch;
    uint32_t r = x % patch, source_x = x / patch;
    uint32_t source = (channel * patch + r) * patch + q;
    out[index] =
        in[(((size_t)source * depth + t) * height + source_y) * width +
           source_x];
}

/* `decode_video`'s `to_rgb` (`conv_video_decoder.py:497-499`):
 * `rearrange(frames[0], "c f h w -> f h w c")` then `add_(1).mul_(0.5)
 * .clamp_(0, 1)` - in bf16, so the add and the multiply each round once and
 * the clamp is exact. The output is the release's own chunk layout, RGB
 * interleaved fastest.
 *
 * The clamp is written as two comparisons rather than `fminf(fmaxf(...))` on
 * purpose: both of those return the *numeric* operand when the other is a NaN,
 * which would turn a NaN activation into 0 or 1 and hide it from every caller's
 * range check. A comparison with a NaN is false, so a NaN stays a NaN and the
 * caller sees it - which is also what `torch.clamp` does. */
template <typename T>
__global__ static void h3v_to_rgb_kernel(T *out, const T *in, uint32_t depth,
                                         uint32_t height, uint32_t width,
                                         size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    size_t plane = (size_t)depth * height * width;
    uint32_t channel = (uint32_t)(index % 3);
    size_t point = index / 3;
    float value = h3b_element<T>::to(in[(size_t)channel * plane + point]);
    float shifted = h3b_element<T>::to(h3b_element<T>::from(value + 1.0f));
    float scaled = h3b_element<T>::from(0.5f * shifted);
    if (scaled < 0.0f) scaled = 0.0f;
    if (scaled > 1.0f) scaled = 1.0f;
    out[index] = h3b_element<T>::from(scaled);
}

int h3v_pixel_norm(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, int dtype, uint32_t channels,
                   size_t plane, float epsilon, char *error,
                   size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t elements = 0;
    if (!context || !channels || !plane || !h3b_type(&type, dtype) ||
        !h3d_mul_size(channels, plane, &elements) ||
        !h3d_is(input, dtype, elements) || !h3d_is(output, dtype, elements)) {
        return h3d_set_error(error, error_size, "invalid pixel norm shape");
    }
    if (elements == 0) return 1;
    // The `plane` of a position: the op reduces over channels, so the walk is
    // strided and the thread count is the position count.
    size_t positions = plane;
    if (dtype == H3D_F32) {
        h3v_pixel_norm_kernel<float><<<h3b_blocks(positions), H3B_BLOCK, 0,
                                       context->stream>>>(
            (float *)output->data, (const float *)input->data, channels, plane,
            epsilon, positions);
    } else {
        h3v_pixel_norm_kernel<__nv_bfloat16><<<h3b_blocks(positions), H3B_BLOCK,
                                               0, context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, channels, plane, epsilon,
            positions);
    }
    return h3d_launch_ok("pixel norm", error, error_size);
}

int h3v_denormalize(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, const h3d_tensor *mean,
                    const h3d_tensor *scale, int dtype, uint32_t channels,
                    size_t plane, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t elements = 0;
    if (!context || !channels || !plane || !h3b_type(&type, dtype) ||
        !h3d_mul_size(channels, plane, &elements) ||
        !h3d_is(input, dtype, elements) || !h3d_is(output, dtype, elements) ||
        !h3d_is(mean, dtype, channels) || !h3d_is(scale, dtype, channels)) {
        return h3d_set_error(error, error_size, "invalid denormalize shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3v_denormalize_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                        context->stream>>>(
            (float *)output->data, (const float *)input->data,
            (const float *)mean->data, (const float *)scale->data, channels,
            plane, elements);
    } else {
        h3v_denormalize_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK,
                                                0, context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)mean->data,
            (const __nv_bfloat16 *)scale->data, channels, plane, elements);
    }
    return h3d_launch_ok("denormalize", error, error_size);
}

int h3v_depth_to_space(h3d_context *context, h3d_tensor *output,
                       const h3d_tensor *input, int dtype, uint32_t in_channels,
                       uint32_t depth, uint32_t height, uint32_t width,
                       uint32_t p1, uint32_t p2, uint32_t p3, uint32_t drop,
                       char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t sources = 0, outputs = 0;
    if (!context || !in_channels || !depth || !height || !width || !p1 || !p2 ||
        !p3 || drop > 1 || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)in_channels * depth * height, width, &sources) ||
        !h3d_mul_size((size_t)(in_channels / (p1 * p2 * p3)) * (depth * p1 - drop),
                      (size_t)(height * p2) * (width * p3), &outputs) ||
        !h3d_is(input, dtype, sources) || !h3d_is(output, dtype, outputs)) {
        return h3d_set_error(error, error_size, "invalid depth to space shape");
    }
    if (outputs == 0) return 1;
    if (dtype == H3D_F32) {
        h3v_depth_to_space_kernel<float><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                           context->stream>>>(
            (float *)output->data, (const float *)input->data, in_channels,
            depth, height, width, p1, p2, p3, drop, outputs);
    } else {
        h3v_depth_to_space_kernel<__nv_bfloat16>
            <<<h3b_blocks(outputs), H3B_BLOCK, 0, context->stream>>>(
                (__nv_bfloat16 *)output->data,
                (const __nv_bfloat16 *)input->data, in_channels, depth, height,
                width, p1, p2, p3, drop, outputs);
    }
    return h3d_launch_ok("depth to space", error, error_size);
}

int h3v_unpatchify(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, int dtype, uint32_t channels,
                   uint32_t patch, uint32_t depth, uint32_t height,
                   uint32_t width, char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t sources = 0, outputs = 0;
    if (!context || !channels || !patch || !depth || !height || !width ||
        !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)channels * patch * patch * depth,
                      (size_t)height * width, &sources) ||
        !h3d_mul_size((size_t)channels * depth, (size_t)height * patch, &outputs) ||
        !h3d_mul_size(outputs, (size_t)width * patch, &outputs) ||
        !h3d_is(input, dtype, sources) || !h3d_is(output, dtype, outputs)) {
        return h3d_set_error(error, error_size, "invalid unpatchify shape");
    }
    if (outputs == 0) return 1;
    if (dtype == H3D_F32) {
        h3v_unpatchify_kernel<float><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                       context->stream>>>(
            (float *)output->data, (const float *)input->data, channels, patch,
            depth, height, width, outputs);
    } else {
        h3v_unpatchify_kernel<__nv_bfloat16><<<h3b_blocks(outputs), H3B_BLOCK, 0,
                                               context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, channels, patch, depth, height,
            width, outputs);
    }
    return h3d_launch_ok("unpatchify", error, error_size);
}

int h3v_to_rgb(h3d_context *context, h3d_tensor *output, const h3d_tensor *input,
               int dtype, uint32_t depth, uint32_t height, uint32_t width,
               char *error, size_t error_size) {
    cudaDataType_t type = CUDA_R_32F;
    size_t plane = 0, elements = 0;
    if (!context || !depth || !height || !width || !h3b_type(&type, dtype) ||
        !h3d_mul_size((size_t)depth * height, width, &plane) ||
        !h3d_mul_size(plane, 3, &elements) ||
        !h3d_is(input, dtype, elements) || !h3d_is(output, dtype, elements)) {
        return h3d_set_error(error, error_size, "invalid to-rgb shape");
    }
    if (elements == 0) return 1;
    if (dtype == H3D_F32) {
        h3v_to_rgb_kernel<float><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                   context->stream>>>(
            (float *)output->data, (const float *)input->data, depth, height,
            width, elements);
    } else {
        h3v_to_rgb_kernel<__nv_bfloat16><<<h3b_blocks(elements), H3B_BLOCK, 0,
                                           context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, depth, height, width, elements);
    }
    return h3d_launch_ok("to rgb", error, error_size);
}

/* ------------------------------------------------------------------------
 * h3a_* — the LTX-2.5 audio VAE encoder (`minimax_h3::ltx_audio_vae`).
 *
 * Source of truth: `/data/ltx2-pin/packages/ltx-core`, pin d1511477 = tag
 * v1.2.0, `model/audio_vae/`. The graph is `audio_vae.py:172-247` and its two
 * layer types:
 *
 *   CausalConv2d, causality_axis=HEIGHT (`causal_conv_2d.py:47-57`):
 *     F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)) then a valid conv.
 *     Kernels are 3 (or 1), so the pad is (1, 1, 2, 0) on the mel-time plane.
 *   Downsample (`downsample.py:36-57`), same axis: F.pad(x, (0, 1, 2, 0)) with
 *     `mode="constant", value=0`, then a stride-2 valid conv.
 *
 * Every pad in this encoder is zeros - unlike the video VAE's replicated
 * temporal halo - so the two kernels below are what the F32 valid convolution
 * (`h3d_conv3d` above, channels-last `[b][d][h][w][c]`) needs around it.
 * ------------------------------------------------------------------------ */

/* Zero padding with a per-side extent on each axis of a channels-last plane:
 * the pad `F.pad(x, (left, right, top, bottom))` writes before the convolution
 * runs. One thread per output element; the interior copies, the rest is zero. */
__global__ static void h3a_pad_zero_kernel(
    float *out, const float *in, uint32_t height, uint32_t width,
    uint32_t channels, uint32_t before_h, uint32_t after_h, uint32_t before_w,
    uint32_t after_w, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = (uint32_t)(index % channels);
    size_t position = index / channels;
    uint32_t out_width = width + before_w + after_w;
    uint32_t out_height = height + before_h + after_h;
    uint32_t x = (uint32_t)(position % out_width);
    uint32_t y = (uint32_t)((position / out_width) % out_height);
    if (x < before_w || x >= before_w + width || y < before_h ||
        y >= before_h + height) {
        out[index] = 0.0f;
        return;
    }
    size_t source =
        ((size_t)(y - before_h) * width + (x - before_w)) * channels + channel;
    out[index] = in[source];
}

int h3a_pad_zero(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, uint32_t height, uint32_t width,
                 uint32_t channels, uint32_t before_h, uint32_t after_h,
                 uint32_t before_w, uint32_t after_w, char *error,
                 size_t error_size) {
    size_t sources = 0, outputs = 0;
    if (!context || !channels || !height || !width ||
        !h3d_mul_size((size_t)height * width, channels, &sources) ||
        !h3d_mul_size((size_t)(height + before_h + after_h) *
                          (width + before_w + after_w),
                      channels, &outputs) ||
        !h3d_is(input, H3D_F32, sources) ||
        !h3d_is(output, H3D_F32, outputs)) {
        return h3d_set_error(error, error_size, "invalid audio pad shape");
    }
    if (outputs == 0) return 1;
    h3a_pad_zero_kernel<<<h3b_blocks(outputs), H3B_BLOCK, 0,
                          context->stream>>>(
        (float *)output->data, (const float *)input->data, height, width,
        channels, before_h, after_h, before_w, after_w, outputs);
    return h3d_launch_ok("audio pad", error, error_size);
}

/* `PixelNorm(dim=1, eps=1e-6)` (`model/common/normalization.py:15-38`) on a
 * channels-last tensor: `x / sqrt(mean(x**2, channels) + eps)`. Torch's
 * expression is the square, the mean, the epsilon add, the square root and the
 * divide, each one rounding; the kernel keeps that order and one thread per
 * position owns its whole channel row, so an in-place call is safe. */
__global__ static void h3a_pixel_norm_kernel(float *out, const float *in,
                                             uint32_t channels, float epsilon,
                                             size_t positions) {
    size_t position = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= positions) return;
    const float *source = in + position * channels;
    float total = 0.0f;
    for (uint32_t channel = 0; channel < channels; ++channel) {
        total += source[channel] * source[channel];
    }
    float rms = sqrtf(total / (float)channels + epsilon);
    float *destination = out + position * channels;
    for (uint32_t channel = 0; channel < channels; ++channel) {
        destination[channel] = source[channel] / rms;
    }
}

int h3a_pixel_norm(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, uint32_t channels,
                   size_t positions, float epsilon, char *error,
                   size_t error_size) {
    size_t elements = 0;
    if (!context || !channels || epsilon < 0.0f ||
        !h3d_mul_size(positions, channels, &elements) ||
        !h3d_is(input, H3D_F32, elements) ||
        !h3d_is(output, H3D_F32, elements)) {
        return h3d_set_error(error, error_size, "invalid audio pixel norm shape");
    }
    if (elements == 0) return 1;
    h3a_pixel_norm_kernel<<<h3b_blocks(positions), H3B_BLOCK, 0,
                            context->stream>>>(
        (float *)output->data, (const float *)input->data, channels, epsilon,
        positions);
    return h3d_launch_ok("audio pixel norm", error, error_size);
}

/* `left + right` on F32, the encoder's residual add (`resnet.py:158-170`).
 * The shortcut and the block output are both F32 and torch's add rounds once,
 * so one thread per element with a single add is exact. `h3d_add` is the
 * BF16 pair kernel and cannot take these planes. */
__global__ static void h3a_add_kernel(float *output, const float *left,
                                      const float *right, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = left[index] + right[index];
}

int h3a_add(h3d_context *context, h3d_tensor *output,
            const h3d_tensor *left, const h3d_tensor *right, size_t elements,
            char *error, size_t error_size) {
    if (!context || !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(left, H3D_F32, elements) ||
        !h3d_is(right, H3D_F32, elements)) {
        return h3d_set_error(error, error_size, "invalid F32 audio add tensors");
    }
    if (elements == 0) return 1;
    h3a_add_kernel<<<h3b_blocks(elements), H3B_BLOCK, 0, context->stream>>>(
        (float *)output->data, (const float *)left->data,
        (const float *)right->data, elements);
    return h3d_launch_ok("audio add", error, error_size);
}
