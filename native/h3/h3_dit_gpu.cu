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
#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct h3d_tensor {
    void *data;
    size_t elements;
    int dtype;
};

struct h3d_context {
    cudaStream_t stream;
    cublasLtHandle_t blas;
    void *stage;          /* pinned upload staging, allocated on first use */
    cudaEvent_t stage_done;
    int stage_pending;    /* a DMA out of `stage` has not been waited for yet */
    int attention_mma;    /* 1 = tensor-core attention (default), 0 = byte-exact scalar */
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
        case H3D_U32: return sizeof(uint32_t);
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
                     dtype != H3D_U32)) {
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

static int h3d_upload_range(h3d_context *context, void *destination,
                            const void *source, size_t bytes, char *error,
                            size_t error_size) {
    if (!context->stage) {
        if (!h3d_cuda_ok(cudaHostAlloc(&context->stage, H3D_STAGE_BYTES,
                                       cudaHostAllocDefault),
                         "allocate upload staging", error, error_size)) {
            return 0;
        }
        if (!h3d_cuda_ok(cudaEventCreateWithFlags(&context->stage_done,
                                                  cudaEventDisableTiming),
                         "create upload event", error, error_size)) {
            return 0;
        }
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

int h3d_read(h3d_context *context, void *host, const h3d_tensor *tensor,
             char *error, size_t error_size) {
    if (!context || !tensor || !host) {
        return h3d_set_error(error, error_size, "invalid read");
    }
    if (!tensor->elements) return 1;
    if (!h3d_cuda_ok(
        cudaMemcpyAsync(host, tensor->data,
                        tensor->elements * h3d_dtype_size(tensor->dtype),
                        cudaMemcpyDeviceToHost, context->stream),
        "read tensor", error, error_size)) {
        return 0;
    }
    return h3d_cuda_ok(cudaStreamSynchronize(context->stream),
                       "read tensor sync", error, error_size);
}

/* `h3_linear_lt` (`h3_gpu_cuda.cu:733-779`), byte for byte: row-major
 * layouts, TRANSB, F32 compute, alpha 1 and beta 0. */
static int h3d_linear_lt(h3d_context *context, void *output, const void *input,
                         const void *weight, uint32_t rows, uint32_t input_dim,
                         uint32_t output_dim, cudaDataType_t type,
                         char *error, size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t input_layout = NULL;
    cublasLtMatrixLayout_t weight_layout = NULL;
    cublasLtMatrixLayout_t output_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
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

/* `h3_adaln_kernel` (`:2656-2687`): thread 0 owns the RMS sum, one block per
 * row, and the modulation row is selected per row through `row_map`. */
__global__ static void h3d_adaln_kernel(__nv_bfloat16 *output,
                                        const __nv_bfloat16 *input,
                                        const __nv_bfloat16 *weight,
                                        const __nv_bfloat16 *modulation,
                                        const uint32_t *row_map, uint32_t rows,
                                        uint32_t width, uint32_t slots,
                                        uint32_t shift_slot,
                                        uint32_t scale_slot, float epsilon,
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
    h3d_adaln_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)norm_weight->data,
        (const __nv_bfloat16 *)modulation->data,
        (const uint32_t *)row_map->data, rows, width, slots, shift_slot,
        scale_slot, epsilon, input_element);
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
 * a shared-memory tree, then `value * inverse * weight`. */
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
        float value = __bfloat162float(input[index]) * inverse *
                      __bfloat162float(weight[column]);
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
        !h3d_is(weight, H3D_BF16, width)) {
        return h3d_set_error(error, error_size, "invalid RMS norm arguments");
    }
    h3d_rms_norm_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)weight->data, rows, width, epsilon);
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
 * head, then the pair rotation on the first `2 * rope_half` dimensions. */
__global__ static void h3d_qkv_rope_kernel(
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
    dim3 grid(sequence, heads, (head_dim + 127) / 128);
    h3d_qkv_rope_kernel<<<grid, 128, 0, context->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
        (const __nv_bfloat16 *)q_norm->data,
        (const __nv_bfloat16 *)k_norm->data,
        (const __nv_bfloat16 *)rope_cos->data,
        (const __nv_bfloat16 *)rope_sin->data, sequence, heads, head_dim,
        rope_half, epsilon);
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
 * order, so the output stays bit-identical. */
__global__ static void h3d_attention_tiled_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t sequence, uint32_t heads, float scale) {
    enum { HEAD_DIM = 128, KEYS = 8, ROWS = H3D_ATTENTION_ROWS,
           QUERIES = 8 * ROWS };
    __shared__ float shared_key[KEYS][HEAD_DIM];
    __shared__ float shared_value[KEYS][HEAD_DIM];
    uint32_t warp = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    uint32_t head = blockIdx.y;
    uint32_t first_row = blockIdx.x * QUERIES + warp * ROWS;
    int active[ROWS];
    float result[ROWS][4];
    float maximum[ROWS];
    float denominator[ROWS];
    /* Each row's four dimensions ride in registers: the reference re-reads them
     * from global for every key row, which exposes that load once per row. */
    float query_item[ROWS][4];
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
        uint32_t row = first_row + r;
        active[r] = row < sequence;
        maximum[r] = -INFINITY;
        denominator[r] = 0.0f;
#pragma unroll
        for (uint32_t part = 0; part < 4; part++) {
            result[r][part] = 0.0f;
            query_item[r][part] = 0.0f;
        }
        if (!active[r]) {
            continue;
        }
        size_t query_base = ((size_t)head * sequence + row) * HEAD_DIM;
#pragma unroll
        for (uint32_t part = 0; part < 4; part++) {
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
                    for (uint32_t part = 0; part < 4; part++) {
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
                    for (uint32_t part = 0; part < 4; part++) {
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
        for (uint32_t item = 0; item < 4; item++) {
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

int h3d_sdpa(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *query, const h3d_tensor *key,
             const h3d_tensor *value, uint32_t sequence, uint32_t heads,
             uint32_t head_dim, float scale, char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !sequence || !heads || head_dim != 128 ||
        !h3d_mul_size((size_t)sequence * heads, head_dim, &elements) ||
        !h3d_is(output, H3D_BF16, elements) ||
        !h3d_is(query, H3D_BF16, elements) ||
        !h3d_is(key, H3D_BF16, elements) ||
        !h3d_is(value, H3D_BF16, elements)) {
        return h3d_set_error(error, error_size,
                             "invalid attention tensors or shape");
    }
    /* The benchmark arm: same tensors and scale, tensor-core arithmetic. */
    if (context->attention_mma) {
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
    h3d_attention_tiled_kernel<<<grid, 256, 0, context->stream>>>(
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
template <int HEAD_DIM>
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
    uint32_t head = blockIdx.x;
    uint32_t first_row = blockIdx.y * QUERIES + warp * ROWS;
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
        dim3 grid(heads, (sequence + 8 * H3D_ATTENTION_F32_ROWS - 1) /
                                 (8 * H3D_ATTENTION_F32_ROWS));
        h3d_attention_tiled_f32_kernel<64><<<grid, 256, 0,
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
                                             uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    size_t base = (size_t)row * width * 2;
    float gate = fused[base + column];
    float up = fused[base + width + column];
    output[(size_t)row * width + column] =
        gate / (1.0f + expf(-gate)) * up;
}

int h3d_swiglu_f32(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *fused, uint32_t rows, uint32_t width,
                   char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !h3d_mul_size(rows, width, &elements) ||
        elements > SIZE_MAX / 2 ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(fused, H3D_F32, elements * 2)) {
        return h3d_set_error(error, error_size, "invalid SwiGLU arguments");
    }
    dim3 grid((width + 255) / 256, rows);
    h3d_swiglu_f32_kernel<<<grid, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)fused->data, rows, width);
    return h3d_launch_ok("F32 SwiGLU", error, error_size);
}

/* `h3_scale_add_kernel` (`:2885-2893`): `residual + branch * scale`, one
 * multiply, and no fma. */
__global__ static void h3d_scale_add_f32_kernel(float *output,
                                                const float *residual,
                                                const float *branch,
                                                const float *scale,
                                                uint32_t rows,
                                                uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    size_t index = (size_t)row * width + column;
    output[index] = residual[index] + branch[index] * scale[column];
}

int h3d_scale_add_f32(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *residual, const h3d_tensor *branch,
                      const h3d_tensor *scale, uint32_t rows, uint32_t width,
                      char *error, size_t error_size) {
    size_t elements = 0;
    if (!context || !h3d_mul_size(rows, width, &elements) ||
        !h3d_is(output, H3D_F32, elements) ||
        !h3d_is(residual, H3D_F32, elements) ||
        !h3d_is(branch, H3D_F32, elements) ||
        !h3d_is(scale, H3D_F32, width)) {
        return h3d_set_error(error, error_size,
                             "invalid F32 scale-add arguments");
    }
    dim3 grid((width + 255) / 256, rows);
    h3d_scale_add_f32_kernel<<<grid, 256, 0, context->stream>>>(
        (float *)output->data, (const float *)residual->data,
        (const float *)branch->data, (const float *)scale->data, rows, width);
    return h3d_launch_ok("F32 scale add", error, error_size);
}

/* `h3_qkv_rope_kernel<float>` (`:3078-3136`) at `grouped = 1`,
 * `normalize = 1`, `weighted = 0` (`:3199-3203`): the checkpoint's
 * [head][q,k,v][dim] rows, the Q/K RMS over the head with no learned scale,
 * then the pair rotation over the first `2 * rope_half` dimensions. */
__global__ static void h3d_video_qkv_rope_f32_kernel(
        float *query, float *key, float *value, const float *qkv,
        const float *rope_cos, const float *rope_sin, uint32_t sequence,
        uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (row >= sequence || head >= heads || dimension >= head_dim) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim * 3;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float q = qkv[q_base + d];
        float k = qkv[k_base + d];
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = rsqrtf(q_sum / (float)head_dim + epsilon);
    float k_inverse = rsqrtf(k_sum / (float)head_dim + epsilon);

    float q0 = qkv[q_base + dimension] * q_inverse;
    float k0 = qkv[k_base + dimension] * k_inverse;
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = qkv[q_base + pair] * q_inverse;
        float k1 = qkv[k_base + pair] * k_inverse;
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
    value[output_index] = qkv[v_base + dimension];
}

int h3d_video_qkv_rope_f32(h3d_context *context, h3d_tensor *query,
                           h3d_tensor *key, h3d_tensor *value,
                           const h3d_tensor *qkv,
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
        !h3d_is(rope_cos, H3D_F32, (size_t)sequence * rope_half) ||
        !h3d_is(rope_sin, H3D_F32, (size_t)sequence * rope_half)) {
        return h3d_set_error(error, error_size, "invalid QKV/RoPE tensors");
    }
    dim3 grid((head_dim + 127) / 128, heads, sequence);
    h3d_video_qkv_rope_f32_kernel<<<grid, 128, 0, context->stream>>>(
        (float *)query->data, (float *)key->data, (float *)value->data,
        (const float *)qkv->data, (const float *)rope_cos->data,
        (const float *)rope_sin->data, sequence, heads, head_dim, rope_half,
        epsilon);
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
