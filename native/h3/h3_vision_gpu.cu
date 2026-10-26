/*
 * h3v_* — device primitives for the MiniMax-H3 vision tower.
 *
 * Source of truth `/data/h3.c` at `eae5d54`, `h3_gpu_cuda.cu`. Every kernel
 * below is the reference kernel's arithmetic with the branch set the vision
 * path uses; the comments name the reference lines.
 *
 * The numerical contract is the same as the text path's: BF16 weights, BF16
 * activations, F32 accumulators, the reference's reduction order. The vision
 * gate compares FNV-1a hashes of BF16 output, so a different reduction tree,
 * a fused bias, or a different cuBLASLt descriptor set fails it.
 */

#include "h3_vision_gpu.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct h3v_tensor {
    void *data;
    size_t elements;
};

struct h3v_context {
    cudaStream_t stream;
    cublasLtHandle_t blas;
};

static int h3v_set_error(char *error, size_t error_size, const char *format,
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
static int h3v_cuda_ok(cudaError_t status, const char *operation, char *error,
                       size_t error_size) {
    if (status == cudaSuccess) return 1;
    return h3v_set_error(error, error_size, "%s: %s", operation,
                         cudaGetErrorString(status));
}

/* `h3_tensor_is` (`h3_gpu_cuda.cu:713-717`): present and at least this big. */
static int h3v_is(const h3v_tensor *tensor, size_t elements) {
    return tensor && tensor->elements >= elements;
}

h3v_context *h3v_create(char *error, size_t error_size) {
    h3v_context *context = (h3v_context *)calloc(1, sizeof(*context));
    if (!context) {
        h3v_set_error(error, error_size, "out of memory creating h3v context");
        return NULL;
    }
    if (!h3v_cuda_ok(cudaStreamCreateWithFlags(&context->stream,
                                               cudaStreamNonBlocking),
                     "create stream", error, error_size)) {
        h3v_destroy(context);
        return NULL;
    }
    cublasStatus_t blas = cublasLtCreate(&context->blas);
    if (blas != CUBLAS_STATUS_SUCCESS) {
        h3v_set_error(error, error_size,
                      "create cuBLASLt handle: status %d", (int)blas);
        h3v_destroy(context);
        return NULL;
    }
    return context;
}

void h3v_destroy(h3v_context *context) {
    if (!context) return;
    if (context->stream) cudaStreamDestroy(context->stream);
    if (context->blas) cublasLtDestroy(context->blas);
    free(context);
}

h3v_tensor *h3v_alloc(h3v_context *context, size_t elements, char *error,
                      size_t error_size) {
    if (elements > SIZE_MAX / sizeof(__nv_bfloat16)) {
        h3v_set_error(error, error_size, "tensor byte size overflow");
        return NULL;
    }
    h3v_tensor *tensor = (h3v_tensor *)calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3v_set_error(error, error_size, "out of memory allocating tensor");
        return NULL;
    }
    tensor->elements = elements;
    if (elements &&
        !h3v_cuda_ok(cudaMalloc(&tensor->data,
                                elements * sizeof(__nv_bfloat16)),
                     "allocate tensor", error, error_size)) {
        free(tensor);
        return NULL;
    }
    (void)context;
    return tensor;
}

void h3v_free(h3v_tensor *tensor) {
    if (!tensor) return;
    if (tensor->data) cudaFree(tensor->data);
    free(tensor);
}

int h3v_upload(h3v_context *context, h3v_tensor *tensor, const void *host,
               char *error, size_t error_size) {
    if (!tensor || !host) {
        return h3v_set_error(error, error_size, "upload: null argument");
    }
    return h3v_cuda_ok(cudaMemcpyAsync(tensor->data, host,
                                       tensor->elements *
                                           sizeof(__nv_bfloat16),
                                       cudaMemcpyHostToDevice, context->stream),
                       "tensor upload", error, error_size);
}

int h3v_read(h3v_context *context, void *host, const h3v_tensor *tensor,
             char *error, size_t error_size) {
    if (!tensor || !host) {
        return h3v_set_error(error, error_size, "read: null argument");
    }
    if (!h3v_cuda_ok(cudaMemcpyAsync(host, tensor->data,
                                     tensor->elements * sizeof(__nv_bfloat16),
                                     cudaMemcpyDeviceToHost, context->stream),
                     "tensor read", error, error_size)) {
        return 0;
    }
    return h3v_cuda_ok(cudaStreamSynchronize(context->stream),
                       "tensor read sync", error, error_size);
}

int h3v_sync(h3v_context *context, char *error, size_t error_size) {
    return h3v_cuda_ok(cudaStreamSynchronize(context->stream), "stream sync",
                       error, error_size);
}

/*
 * `h3_linear_lt` (`:733-779`) with the BF16 types `h3_gpu_linear_bf16`
 * (`:852-870`) passes. Row-major A `[rows, input_dim]`, row-major B
 * `[output_dim, input_dim]` with TRANSB, row-major C `[rows, output_dim]`,
 * compute type F32, alpha 1, beta 0, no workspace, our stream.
 *
 * Do not change the descriptors: cuBLASLt picks its kernel from them, and a
 * different selection changes the accumulation order and the BF16 result.
 */
static int h3v_linear_lt(h3v_context *context, void *output, const void *input,
                         const void *weight, uint32_t rows,
                         uint32_t input_dim, uint32_t output_dim, char *error,
                         size_t error_size) {
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
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
            sizeof(transpose));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &input_layout, CUDA_R_16BF, rows, input_dim, input_dim);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &weight_layout, CUDA_R_16BF, output_dim, input_dim, input_dim);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &output_layout, CUDA_R_16BF, rows, output_dim, output_dim);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmul(context->blas, operation, &alpha, input,
                                input_layout, weight, weight_layout, &beta,
                                output, output_layout, output, output_layout,
                                NULL, NULL, 0, context->stream);
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return h3v_set_error(error, error_size, "cuBLASLt linear: status %d",
                             (int)status);
    }
    return 1;
}

/* `h3_linear_bias_kernel` (`:783-790`): one launch after the GEMM, so the
 * output is rounded to BF16 by the GEMM and again after the add. */
__global__ static void h3v_linear_bias_kernel(__nv_bfloat16 *output,
                                              const __nv_bfloat16 *bias,
                                              size_t elements,
                                              uint32_t columns) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = (float)output[index] + (float)bias[index % columns];
    output[index] = (__nv_bfloat16)value;
}

static int h3v_linear_bias(h3v_context *context, h3v_tensor *output,
                           const h3v_tensor *bias, size_t elements,
                           uint32_t columns, char *error, size_t error_size) {
    if (!bias) return 1;
    unsigned blocks = (unsigned)((elements + 255) / 256);
    h3v_linear_bias_kernel<<<blocks, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)bias->data,
        elements, columns);
    return h3v_cuda_ok(cudaGetLastError(), "linear bias", error, error_size);
}

int h3v_linear(h3v_context *context, h3v_tensor *output,
               const h3v_tensor *input, const h3v_tensor *weight,
               const h3v_tensor *bias, uint32_t rows, uint32_t input_dim,
               uint32_t output_dim, char *error, size_t error_size) {
    if (!rows || !input_dim || !output_dim ||
        !h3v_is(input, (size_t)rows * input_dim) ||
        !h3v_is(weight, (size_t)output_dim * input_dim) ||
        !h3v_is(output, (size_t)rows * output_dim) ||
        (bias && !h3v_is(bias, output_dim))) {
        return h3v_set_error(error, error_size,
                             "invalid BF16 linear tensors or shape");
    }
    if (!h3v_linear_lt(context, output->data, input->data, weight->data, rows,
                       input_dim, output_dim, error, error_size)) {
        return 0;
    }
    return h3v_linear_bias(context, output, bias,
                           (size_t)rows * output_dim, output_dim, error,
                           error_size);
}

/* `h3_layer_norm_bf16_kernel` (`:2563-2594`): 256 threads, one block per row,
 * a shared tree sum for the mean and again for the variance, then
 * `(x - mean) * rsqrt(var + eps) * weight + bias` rounded to BF16. */
__global__ static void h3v_layer_norm_kernel(__nv_bfloat16 *output,
                                             const __nv_bfloat16 *input,
                                             const __nv_bfloat16 *weight,
                                             const __nv_bfloat16 *bias,
                                             uint32_t rows, uint32_t width,
                                             float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x)
        sum += __bfloat162float(input[(size_t)row * width + column]);
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = sums[0] / (float)width;
    sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float centered = __bfloat162float(input[(size_t)row * width + column]) - mean;
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
        float value = (__bfloat162float(input[index]) - mean) * inverse *
                          __bfloat162float(weight[column]) +
                      __bfloat162float(bias[column]);
        output[index] = __float2bfloat16(value);
    }
}

int h3v_layer_norm(h3v_context *context, h3v_tensor *output,
                   const h3v_tensor *input, const h3v_tensor *weight,
                   const h3v_tensor *bias, uint32_t rows, uint32_t width,
                   float epsilon, char *error, size_t error_size) {
    size_t elements = (size_t)rows * width;
    if (!rows || !width || epsilon < 0.0f || !h3v_is(output, elements) ||
        !h3v_is(input, elements) || !h3v_is(weight, width) ||
        !h3v_is(bias, width)) {
        return h3v_set_error(error, error_size,
                             "invalid BF16 layer norm arguments");
    }
    h3v_layer_norm_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)weight->data,
        (const __nv_bfloat16 *)bias->data, rows, width, epsilon);
    return h3v_cuda_ok(cudaGetLastError(), "BF16 layer norm", error,
                       error_size);
}

/* `h3_unary_bf16_kernel` (`:2257-2278`), GELU branches. The +-10 clamps are
 * the reference's: outside them tanhf/erff would still be finite, but the
 * reference short-circuits, so the port must too. */
__global__ static void h3v_gelu_kernel(__nv_bfloat16 *output,
                                       const __nv_bfloat16 *input,
                                       size_t count, int approximate) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = __bfloat162float(input[index]);
    if (approximate) {
        float inner = 0.7978845608028654f *
                      (value + 0.044715f * value * value * value);
        value = inner <= -10.0f ? 0.0f : inner >= 10.0f ? value :
                0.5f * value * (1.0f + tanhf(inner));
    } else {
        value = value <= -10.0f ? 0.0f : value >= 10.0f ? value :
                0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
    }
    output[index] = __float2bfloat16(value);
}

int h3v_gelu(h3v_context *context, h3v_tensor *output, const h3v_tensor *input,
             uint32_t elements, int approximate, char *error,
             size_t error_size) {
    if (!h3v_is(output, elements) || !h3v_is(input, elements)) {
        return h3v_set_error(error, error_size, "invalid BF16 GELU tensors");
    }
    if (elements) {
        h3v_gelu_kernel<<<(elements + 255) / 256, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)input->data, elements, approximate);
    }
    return h3v_cuda_ok(cudaGetLastError(), "BF16 GELU", error, error_size);
}

/* `h3_binary_bf16_kernel` (`:2347-2357`), add branch: both operands read as
 * F32, added in F32, one rounding on store. */
__global__ static void h3v_add_kernel(__nv_bfloat16 *output,
                                      const __nv_bfloat16 *left,
                                      const __nv_bfloat16 *right,
                                      size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float a = __bfloat162float(left[index]);
    float b = __bfloat162float(right[index]);
    output[index] = __float2bfloat16(a + b);
}

int h3v_add(h3v_context *context, h3v_tensor *output, const h3v_tensor *left,
            const h3v_tensor *right, uint32_t elements, char *error,
            size_t error_size) {
    if (!h3v_is(output, elements) || !h3v_is(left, elements) ||
        !h3v_is(right, elements)) {
        return h3v_set_error(error, error_size, "invalid BF16 add tensors");
    }
    if (elements) {
        h3v_add_kernel<<<(elements + 255) / 256, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)left->data,
            (const __nv_bfloat16 *)right->data, elements);
    }
    return h3v_cuda_ok(cudaGetLastError(), "BF16 add", error, error_size);
}

/* `h3_qkv_rope_kernel` (`:3078-3137`) with the vision branch: `grouped`,
 * `normalize` and `weighted` are all zero (`h3_gpu_vision_qkv_rope_bf16`,
 * `:3252-3267`), so each of q and k is scaled by 1.0 and rotated in place of
 * qkv, and v is copied unchanged. The half-split pairing is the reference's:
 * a dimension below `rope_half` pairs with the one `rope_half` above it. */
__global__ static void h3v_qkv_rope_kernel(__nv_bfloat16 *query,
                                           __nv_bfloat16 *key,
                                           __nv_bfloat16 *value,
                                           const __nv_bfloat16 *qkv,
                                           const __nv_bfloat16 *rope_cos,
                                           const __nv_bfloat16 *rope_sin,
                                           uint32_t sequence, uint32_t heads,
                                           uint32_t head_dim,
                                           uint32_t rope_half) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (row >= sequence || head >= heads || dimension >= head_dim) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim;
    size_t k_base = q_base + inner;
    size_t v_base = q_base + inner * 2;
    float q0 = __bfloat162float(qkv[q_base + dimension]);
    float k0 = __bfloat162float(qkv[k_base + dimension]);
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = __bfloat162float(qkv[q_base + pair]);
        float k1 = __bfloat162float(qkv[k_base + pair]);
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

int h3v_qkv_rope(h3v_context *context, h3v_tensor *query, h3v_tensor *key,
                 h3v_tensor *value, const h3v_tensor *qkv,
                 const h3v_tensor *cosines, const h3v_tensor *sines,
                 uint32_t sequence, uint32_t heads, uint32_t head_dim,
                 uint32_t rope_half, char *error, size_t error_size) {
    if (!sequence || !heads || !head_dim || rope_half > head_dim / 2) {
        return h3v_set_error(error, error_size, "invalid QKV/RoPE shape");
    }
    size_t elements = (size_t)sequence * heads * head_dim;
    size_t ropes = (size_t)sequence * rope_half;
    if (!h3v_is(query, elements) || !h3v_is(key, elements) ||
        !h3v_is(value, elements) || !h3v_is(qkv, elements * 3) ||
        !h3v_is(cosines, ropes) || !h3v_is(sines, ropes)) {
        return h3v_set_error(error, error_size, "invalid QKV/RoPE tensors");
    }
    dim3 grid((head_dim + 127) / 128, heads, sequence);
    h3v_qkv_rope_kernel<<<grid, 128, 0, context->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
        (const __nv_bfloat16 *)cosines->data,
        (const __nv_bfloat16 *)sines->data, sequence, heads, head_dim,
        rope_half);
    return h3v_cuda_ok(cudaGetLastError(), "BF16 vision QKV RoPE", error,
                       error_size);
}

/* `h3_attention_kernel` (`:1338-1440`) with batch 1, one KV head per query
 * head, `causal` 0 and head-major input. Each block owns one (head, row): a
 * 128-thread tree reduction produces the score, thread 0 runs the online
 * softmax, and every thread carries its own output dimension. */
__global__ static void h3v_attention_kernel(__nv_bfloat16 *output,
                                            const __nv_bfloat16 *query,
                                            const __nv_bfloat16 *key,
                                            const __nv_bfloat16 *value,
                                            uint32_t sequence, uint32_t heads,
                                            uint32_t head_dim, float scale) {
    __shared__ float partial[1024];
    __shared__ float rescale;
    __shared__ float weight;
    __shared__ float denominator;
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y;
    size_t query_base = ((size_t)head * sequence + row) * head_dim;
    uint32_t dimension = threadIdx.x;
    float result = 0.0f;
    float maximum = -INFINITY;
    if (threadIdx.x == 0) denominator = 0.0f;
    __syncthreads();
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        size_t key_base = ((size_t)head * sequence + key_row) * head_dim;
        float product = 0.0f;
        if (dimension < head_dim) {
            product = __bfloat162float(query[query_base + dimension]) *
                      __bfloat162float(key[key_base + dimension]);
        }
        partial[threadIdx.x] = product;
        __syncthreads();
        for (uint32_t offset = blockDim.x / 2; offset; offset >>= 1) {
            if (threadIdx.x < offset)
                partial[threadIdx.x] += partial[threadIdx.x + offset];
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
            float element = __bfloat162float(value[key_base + dimension]);
            result = result * rescale + weight * element;
        }
        __syncthreads();
    }
    if (dimension < head_dim) {
        result /= denominator;
        size_t output_index =
            (((size_t)row * heads + head) * head_dim) + dimension;
        output[output_index] = __float2bfloat16(result);
    }
}

int h3v_sdpa(h3v_context *context, h3v_tensor *output, const h3v_tensor *query,
             const h3v_tensor *key, const h3v_tensor *value, uint32_t sequence,
             uint32_t heads, uint32_t head_dim, float scale, char *error,
             size_t error_size) {
    size_t elements = (size_t)sequence * heads * head_dim;
    if (!sequence || !heads || !head_dim || head_dim > 1024 ||
        !h3v_is(output, elements) || !h3v_is(query, elements) ||
        !h3v_is(key, elements) || !h3v_is(value, elements)) {
        return h3v_set_error(error, error_size,
                             "invalid attention tensors or shape");
    }
    /* `h3_attention_dispatch` (`:1583-1584`): 128 threads, doubled until the
     * head dimension fits. The vision tower's 72 lands on 128. */
    uint32_t threads = 128;
    while (threads < head_dim) threads <<= 1;
    dim3 grid(heads, sequence, 1);
    h3v_attention_kernel<<<grid, threads, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
        (const __nv_bfloat16 *)key->data,
        (const __nv_bfloat16 *)value->data, sequence, heads, head_dim, scale);
    return h3v_cuda_ok(cudaGetLastError(),
                       "scaled dot-product attention", error, error_size);
}
