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
};

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
    return context;
}

void h3d_destroy(h3d_context *context) {
    if (!context) return;
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

int h3d_upload(h3d_context *context, h3d_tensor *tensor, const void *host,
               char *error, size_t error_size) {
    if (!context || !tensor || !host) {
        return h3d_set_error(error, error_size, "invalid upload");
    }
    if (!tensor->elements) return 1;
    return h3d_cuda_ok(
        cudaMemcpyAsync(tensor->data, host,
                        tensor->elements * h3d_dtype_size(tensor->dtype),
                        cudaMemcpyHostToDevice, context->stream),
        "upload tensor", error, error_size);
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
    return h3d_cuda_ok(
        cudaMemcpyAsync((char *)tensor->data + element * item, host,
                        elements * item, cudaMemcpyHostToDevice,
                        context->stream),
        "upload tensor range", error, error_size);
}

int h3d_read(h3d_context *context, void *host, const h3d_tensor *tensor,
             char *error, size_t error_size) {
    if (!context || !tensor || !host) {
        return h3d_set_error(error, error_size, "invalid read");
    }
    if (!tensor->elements) return 1;
    return h3d_cuda_ok(
        cudaMemcpyAsync(host, tensor->data,
                        tensor->elements * h3d_dtype_size(tensor->dtype),
                        cudaMemcpyDeviceToHost, context->stream),
        "read tensor", error, error_size);
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

/* `h3_gate_kernel` (`:2689-2702`). */
__global__ static void h3d_gate_kernel(__nv_bfloat16 *output,
                                       const __nv_bfloat16 *residual,
                                       const __nv_bfloat16 *branch,
                                       const __nv_bfloat16 *modulation,
                                       const uint32_t *row_map, uint32_t rows,
                                       uint32_t width, uint32_t slots,
                                       uint32_t gate_slot) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
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
        !row_map || row_map->elements < rows) {
        return h3d_set_error(error, error_size, "invalid gate arguments");
    }
    dim3 grid((width + 255) / 256, rows);
    h3d_gate_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)residual->data,
        (const __nv_bfloat16 *)branch->data,
        (const __nv_bfloat16 *)modulation->data,
        (const uint32_t *)row_map->data, rows, width, slots, gate_slot);
    return h3d_launch_ok("BF16 gate", error, error_size);
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
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
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
    dim3 grid((width + 255) / 256, rows);
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
    dim3 grid((head_dim + 127) / 128, heads, sequence);
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

/* `h3_attention_tiled_bf16_kernel` (`:1418-1487`): one block per head and
 * eight rows, one warp per row, four dimensions per lane, online softmax with
 * a warp shuffle reduction, row-major output (`head_major_output = 0`). */
__global__ static void h3d_attention_tiled_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t sequence, uint32_t heads, float scale) {
    enum { HEAD_DIM = 128, QUERIES = 8 };
    __shared__ float shared_key[HEAD_DIM];
    __shared__ float shared_value[HEAD_DIM];
    uint32_t warp = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y * QUERIES + warp;
    int active = row < sequence;
    size_t query_base = ((size_t)head * sequence + row) * HEAD_DIM;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        if (threadIdx.x < HEAD_DIM) {
            size_t index = ((size_t)head * sequence + key_row) * HEAD_DIM +
                           threadIdx.x;
            shared_key[threadIdx.x] = __bfloat162float(key[index]);
            shared_value[threadIdx.x] = __bfloat162float(value[index]);
        }
        __syncthreads();
        float product = 0.0f;
        if (active) {
#pragma unroll
            for (uint32_t item = 0; item < 4; item++) {
                uint32_t dimension = lane + item * 32;
                product += __bfloat162float(query[query_base + dimension]) *
                           shared_key[dimension];
            }
#pragma unroll
            for (uint32_t offset = 16; offset; offset >>= 1) {
                product += __shfl_down_sync(0xffffffffu, product, offset);
            }
        }
        float rescale = 1.0f;
        float weight = 0.0f;
        if (active && lane == 0) {
            float score = product * scale;
            float next_maximum = fmaxf(maximum, score);
            rescale = expf(maximum - next_maximum);
            weight = expf(score - next_maximum);
            denominator = denominator * rescale + weight;
            maximum = next_maximum;
        }
        rescale = __shfl_sync(0xffffffffu, rescale, 0);
        weight = __shfl_sync(0xffffffffu, weight, 0);
        if (active) {
#pragma unroll
            for (uint32_t item = 0; item < 4; item++) {
                uint32_t dimension = lane + item * 32;
                result[item] = result[item] * rescale +
                               weight * shared_value[dimension];
            }
        }
        __syncthreads();
    }
    denominator = __shfl_sync(0xffffffffu, denominator, 0);
    if (!active) return;
#pragma unroll
    for (uint32_t item = 0; item < 4; item++) {
        uint32_t dimension = lane + item * 32;
        size_t index = ((size_t)row * heads + head) * HEAD_DIM + dimension;
        output[index] = __float2bfloat16(result[item] / denominator);
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
    dim3 grid(heads, (sequence + 7) / 8);
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
