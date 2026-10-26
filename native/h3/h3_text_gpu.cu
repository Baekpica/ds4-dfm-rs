/*
 * h3t_* — device primitives for the MiniMax-H3 prompt encoder.
 *
 * Source of truth `/data/h3.c` at `eae5d54`, `h3_gpu_cuda.cu`. Every kernel
 * below is the reference kernel's arithmetic with the branch set the text
 * path actually uses; the comments name the reference lines.
 *
 * Why the arithmetic order is copied exactly: the E4 gate compares an FNV-1a
 * hash of the BF16 output, so a different reduction tree or a fused vs split
 * multiply-add changes the last bit of a BF16 value and fails the gate. Where
 * the reference relies on library behavior (cuBLASLt's heuristic choice for
 * the projections), the same call is made with the same descriptors
 * (`h3_linear_lt`, `:733-779`).
 */

#include "h3_text_gpu.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct h3t_tensor {
    void *data;
    size_t elements;
    int dtype;
};

struct h3t_context {
    cudaStream_t stream;
    cublasLtHandle_t blas;
    /* Token ids for the embedding gather; grown on demand. */
    uint32_t *ids;
    size_t ids_capacity;
};

static int h3t_set_error(char *error, size_t error_size, const char *format,
                         ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

/* `h3_cuda_ok` (`h3_gpu_cuda.cu:220-235`): one place for launch and runtime
 * failures, so a silent launch error cannot pass as success. */
static int h3t_cuda_ok(cudaError_t status, const char *operation, char *error,
                       size_t error_size) {
    if (status == cudaSuccess) return 1;
    return h3t_set_error(error, error_size, "%s: %s", operation,
                         cudaGetErrorString(status));
}

static size_t h3t_dtype_size(int dtype) {
    return dtype == H3T_F32 ? sizeof(float) : sizeof(__nv_bfloat16);
}

static int h3t_is(const h3t_tensor *tensor, int dtype, size_t elements) {
    return tensor && tensor->dtype == dtype && tensor->elements >= elements;
}

h3t_context *h3t_create(char *error, size_t error_size) {
    h3t_context *context = (h3t_context *)calloc(1, sizeof(*context));
    if (!context) {
        h3t_set_error(error, error_size, "out of memory creating h3 context");
        return NULL;
    }
    if (!h3t_cuda_ok(cudaStreamCreateWithFlags(&context->stream,
                                               cudaStreamNonBlocking),
                     "create stream", error, error_size)) {
        h3t_destroy(context);
        return NULL;
    }
    cublasStatus_t blas = cublasLtCreate(&context->blas);
    if (blas != CUBLAS_STATUS_SUCCESS) {
        h3t_set_error(error, error_size,
                      "create cuBLASLt handle: status %d", (int)blas);
        h3t_destroy(context);
        return NULL;
    }
    return context;
}

void h3t_destroy(h3t_context *context) {
    if (!context) return;
    if (context->stream) cudaStreamDestroy(context->stream);
    if (context->blas) cublasLtDestroy(context->blas);
    /* `ids` is device memory: `free` on it corrupts the host heap. */
    if (context->ids) cudaFree(context->ids);
    free(context);
}

h3t_tensor *h3t_alloc(h3t_context *context, int dtype, size_t elements,
                      char *error, size_t error_size) {
    if (dtype != H3T_BF16 && dtype != H3T_F32) {
        h3t_set_error(error, error_size, "unknown dtype %d", dtype);
        return NULL;
    }
    if (elements > SIZE_MAX / h3t_dtype_size(dtype)) {
        h3t_set_error(error, error_size, "tensor byte size overflow");
        return NULL;
    }
    h3t_tensor *tensor = (h3t_tensor *)calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3t_set_error(error, error_size, "out of memory allocating tensor");
        return NULL;
    }
    tensor->elements = elements;
    tensor->dtype = dtype;
    if (elements) {
        size_t bytes = elements * h3t_dtype_size(dtype);
        if (!h3t_cuda_ok(cudaMalloc(&tensor->data, bytes), "allocate tensor",
                         error, error_size)) {
            free(tensor);
            return NULL;
        }
    }
    (void)context;
    return tensor;
}

void h3t_free(h3t_tensor *tensor) {
    if (!tensor) return;
    if (tensor->data) cudaFree(tensor->data);
    free(tensor);
}

int h3t_upload(h3t_context *context, h3t_tensor *tensor, const void *host,
               char *error, size_t error_size) {
    if (!tensor || !host) {
        return h3t_set_error(error, error_size, "upload: null argument");
    }
    size_t bytes = tensor->elements * h3t_dtype_size(tensor->dtype);
    return h3t_cuda_ok(cudaMemcpyAsync(tensor->data, host, bytes,
                                       cudaMemcpyHostToDevice, context->stream),
                       "tensor upload", error, error_size);
}

int h3t_write(h3t_context *context, h3t_tensor *tensor, size_t offset,
              const void *host, size_t elements, char *error,
              size_t error_size) {
    /* The reference's own bounds (`h3_gpu_cuda.cu:597-600`). */
    if (!tensor || !host || offset > tensor->elements ||
        elements > tensor->elements - offset) {
        return h3t_set_error(error, error_size,
                             "invalid tensor write range");
    }
    size_t size = h3t_dtype_size(tensor->dtype);
    return h3t_cuda_ok(
        cudaMemcpyAsync((char *)tensor->data + offset * size, host,
                        elements * size, cudaMemcpyHostToDevice,
                        context->stream),
        "tensor range write", error, error_size);
}

int h3t_read(h3t_context *context, void *host, const h3t_tensor *tensor,
             char *error, size_t error_size) {
    if (!tensor || !host) {
        return h3t_set_error(error, error_size, "read: null argument");
    }
    size_t bytes = tensor->elements * h3t_dtype_size(tensor->dtype);
    if (!h3t_cuda_ok(cudaMemcpyAsync(host, tensor->data, bytes,
                                     cudaMemcpyDeviceToHost, context->stream),
                     "tensor read", error, error_size)) {
        return 0;
    }
    return h3t_cuda_ok(cudaStreamSynchronize(context->stream), "tensor read sync",
                       error, error_size);
}

int h3t_sync(h3t_context *context, char *error, size_t error_size) {
    return h3t_cuda_ok(cudaStreamSynchronize(context->stream), "stream sync",
                       error, error_size);
}

/* `h3_embedding_kernel` (`:2811-2820`). */
__global__ static void h3t_embed_kernel(__nv_bfloat16 *output,
                                        const __nv_bfloat16 *weight,
                                        const uint32_t *token_ids,
                                        uint32_t tokens, uint32_t vocab_size,
                                        uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t token = blockIdx.y;
    if (token >= tokens || column >= width) return;
    uint32_t id = token_ids[token];
    output[(size_t)token * width + column] = id < vocab_size ?
        weight[(size_t)id * width + column] : __float2bfloat16(0.0f);
}

int h3t_embed(h3t_context *context, h3t_tensor *output,
              const h3t_tensor *weight, const uint32_t *ids, uint32_t tokens,
              uint32_t vocab, uint32_t width, char *error, size_t error_size) {
    if (!h3t_is(output, H3T_BF16, (size_t)tokens * width) ||
        !h3t_is(weight, H3T_BF16, (size_t)vocab * width) || !ids || !tokens ||
        !width) {
        return h3t_set_error(error, error_size, "invalid embedding arguments");
    }
    if (context->ids_capacity < tokens) {
        /* `ids` is device memory: `free` on it corrupts the host heap. */
        if (context->ids) cudaFree(context->ids);
        context->ids = NULL;
        context->ids_capacity = 0;
        if (!h3t_cuda_ok(cudaMalloc((void **)&context->ids,
                                    tokens * sizeof(uint32_t)),
                         "allocate token ids", error, error_size)) {
            return 0;
        }
        context->ids_capacity = tokens;
    }
    if (!h3t_cuda_ok(cudaMemcpyAsync(context->ids, ids,
                                     tokens * sizeof(uint32_t),
                                     cudaMemcpyHostToDevice, context->stream),
                     "token id upload", error, error_size)) {
        return 0;
    }
    dim3 grid((width + 255) / 256, tokens);
    h3t_embed_kernel<<<grid, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)weight->data,
        context->ids, tokens, vocab, width);
    return h3t_cuda_ok(cudaGetLastError(), "BF16 embedding", error, error_size);
}

/* `h3_rms_norm_bf16_kernel` (`:2462-2486`): 256 threads, one block per row,
 * an `fmaf` accumulation striding by `blockDim.x`, then a shared tree sum. */
__global__ static void h3t_rms_kernel(__nv_bfloat16 *output,
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

int h3t_rms_norm(h3t_context *context, h3t_tensor *output,
                 const h3t_tensor *input, const h3t_tensor *weight,
                 uint32_t rows, uint32_t width, float epsilon, char *error,
                 size_t error_size) {
    size_t elements = (size_t)rows * width;
    if (!rows || !width || !h3t_is(output, H3T_BF16, elements) ||
        !h3t_is(input, H3T_BF16, elements) ||
        !h3t_is(weight, H3T_BF16, width)) {
        return h3t_set_error(error, error_size, "invalid RMS norm arguments");
    }
    h3t_rms_kernel<<<rows, 256, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)weight->data, rows, width, epsilon);
    return h3t_cuda_ok(cudaGetLastError(), "BF16 RMS norm", error, error_size);
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
static int h3t_linear_lt(h3t_context *context, void *output, const void *input,
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
        return h3t_set_error(error, error_size, "cuBLASLt linear: status %d",
                             (int)status);
    }
    return 1;
}

int h3t_linear(h3t_context *context, h3t_tensor *output,
               const h3t_tensor *input, const h3t_tensor *weight,
               uint32_t rows, uint32_t input_dim, uint32_t output_dim,
               char *error, size_t error_size) {
    if (!rows || !input_dim || !output_dim ||
        !h3t_is(input, H3T_BF16, (size_t)rows * input_dim) ||
        !h3t_is(weight, H3T_BF16, (size_t)output_dim * input_dim) ||
        !h3t_is(output, H3T_BF16, (size_t)rows * output_dim)) {
        return h3t_set_error(error, error_size,
                             "invalid BF16 linear tensors or shape");
    }
    return h3t_linear_lt(context, output->data, input->data, weight->data, rows,
                         input_dim, output_dim, error, error_size);
}

/* `h3_head_rms_kernel` (`:2991-3008`): one thread per (row, head), the
 * per-head sum accumulated serially over `head_dim`, in place. */
__global__ static void h3t_head_rms_kernel(__nv_bfloat16 *tensor,
                                           const __nv_bfloat16 *weight,
                                           uint32_t sequence, uint32_t heads,
                                           uint32_t head_dim, float epsilon) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    if (row >= sequence || head >= heads) return;
    size_t base = ((size_t)row * heads + head) * head_dim;
    float sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float value = __bfloat162float(tensor[base + d]);
        sum = fmaf(value, value, sum);
    }
    float inverse = rsqrtf(sum / (float)head_dim + epsilon);
    for (uint32_t d = 0; d < head_dim; d++)
        tensor[base + d] = __float2bfloat16(
            __bfloat162float(tensor[base + d]) * inverse *
            __bfloat162float(weight[d]));
}

int h3t_head_rms_norm(h3t_context *context, h3t_tensor *tensor,
                      const h3t_tensor *weight, uint32_t sequence,
                      uint32_t heads, uint32_t head_dim, float epsilon,
                      char *error, size_t error_size) {
    size_t elements = (size_t)sequence * heads * head_dim;
    if (!sequence || !heads || !head_dim ||
        !h3t_is(tensor, H3T_BF16, elements) ||
        !h3t_is(weight, H3T_BF16, head_dim)) {
        return h3t_set_error(error, error_size, "invalid head RMS arguments");
    }
    dim3 grid((sequence + 127) / 128, heads);
    h3t_head_rms_kernel<<<grid, 128, 0, context->stream>>>(
        (__nv_bfloat16 *)tensor->data, (const __nv_bfloat16 *)weight->data,
        sequence, heads, head_dim, epsilon);
    return h3t_cuda_ok(cudaGetLastError(), "BF16 head RMS norm", error,
                       error_size);
}

/* `h3_rope_text_kernel` (`:3055-3090`): the pair (d, d + half) rotates by the
 * F32 table entry for (row, d). */
__global__ static void h3t_rope_kernel(__nv_bfloat16 *query,
                                       __nv_bfloat16 *key,
                                       const float *rope_cos,
                                       const float *rope_sin,
                                       uint32_t sequence,
                                       uint32_t query_heads,
                                       uint32_t kv_heads, uint32_t head_dim) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    if (row >= sequence) return;
    uint32_t half = head_dim / 2;
    for (uint32_t d = 0; d < half; d++) {
        float c = rope_cos[(size_t)row * half + d];
        float s = rope_sin[(size_t)row * half + d];
        if (head < query_heads) {
            size_t base = ((size_t)row * query_heads + head) * head_dim;
            float first = __bfloat162float(query[base + d]);
            float second = __bfloat162float(query[base + half + d]);
            query[base + d] = __float2bfloat16(first * c - second * s);
            query[base + half + d] = __float2bfloat16(second * c + first * s);
        }
        if (head < kv_heads) {
            size_t base = ((size_t)row * kv_heads + head) * head_dim;
            float first = __bfloat162float(key[base + d]);
            float second = __bfloat162float(key[base + half + d]);
            key[base + d] = __float2bfloat16(first * c - second * s);
            key[base + half + d] = __float2bfloat16(second * c + first * s);
        }
    }
}

int h3t_rope_text(h3t_context *context, h3t_tensor *query, h3t_tensor *key,
                  const h3t_tensor *cosines, const h3t_tensor *sines,
                  uint32_t sequence, uint32_t query_heads, uint32_t kv_heads,
                  uint32_t head_dim, char *error, size_t error_size) {
    if (!sequence || !query_heads || !kv_heads || !head_dim || head_dim % 2 ||
        !h3t_is(query, H3T_BF16, (size_t)sequence * query_heads * head_dim) ||
        !h3t_is(key, H3T_BF16, (size_t)sequence * kv_heads * head_dim) ||
        !h3t_is(cosines, H3T_F32, (size_t)sequence * (head_dim / 2)) ||
        !h3t_is(sines, H3T_F32, (size_t)sequence * (head_dim / 2))) {
        return h3t_set_error(error, error_size, "invalid text RoPE tensors");
    }
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    dim3 grid((sequence + 127) / 128, maximum_heads);
    h3t_rope_kernel<<<grid, 128, 0, context->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (const float *)cosines->data, (const float *)sines->data, sequence,
        query_heads, kv_heads, head_dim);
    return h3t_cuda_ok(cudaGetLastError(), "BF16 text RoPE", error,
                       error_size);
}

/*
 * `h3_attention_kernel<__nv_bfloat16>` (`:1338-1414`) at the arguments
 * `h3_gpu_gqa_causal_bf16` (`:1683-1689`) selects: batch 1, causal, both input
 * and output row-major `[sequence, heads, head_dim]` (`causal == 2` in
 * `h3_attention_dispatch`, `:1562-1564`). One block per (head, row), 128
 * threads, the query/key dot reduced through shared memory, and the online
 * softmax exactly as written: `expf`, rescale, `result * rescale + weight * v`.
 */
__global__ static void h3t_attention_kernel(__nv_bfloat16 *output,
                                            const __nv_bfloat16 *query,
                                            const __nv_bfloat16 *key,
                                            const __nv_bfloat16 *value,
                                            uint32_t sequence,
                                            uint32_t query_heads,
                                            uint32_t kv_heads,
                                            uint32_t head_dim, float scale) {
    __shared__ float partial[1024];
    __shared__ float rescale;
    __shared__ float weight;
    __shared__ float denominator;
    uint32_t query_head = blockIdx.x;
    uint32_t row = blockIdx.y;
    uint32_t kv_head = query_head / (query_heads / kv_heads);
    uint32_t keys = row + 1;
    size_t query_base =
        ((size_t)row * query_heads + query_head) * head_dim;
    uint32_t dimension = threadIdx.x;
    float result = 0.0f;
    float maximum = -INFINITY;
    if (threadIdx.x == 0) denominator = 0.0f;
    __syncthreads();
    for (uint32_t key_row = 0; key_row < keys; key_row++) {
        size_t key_base = ((size_t)key_row * kv_heads + kv_head) * head_dim;
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
            size_t value_index = key_base + dimension;
            float value_element = __bfloat162float(value[value_index]);
            result = result * rescale + weight * value_element;
        }
        __syncthreads();
    }
    if (dimension < head_dim) {
        result /= denominator;
        size_t output_index =
            ((size_t)row * query_heads + query_head) * head_dim + dimension;
        output[output_index] = __float2bfloat16(result);
    }
}

int h3t_gqa_causal(h3t_context *context, h3t_tensor *output,
                   const h3t_tensor *query, const h3t_tensor *key,
                   const h3t_tensor *value, uint32_t sequence,
                   uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
                   float scale, char *error, size_t error_size) {
    size_t query_elements = (size_t)sequence * query_heads * head_dim;
    size_t kv_elements = (size_t)sequence * kv_heads * head_dim;
    if (!sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads ||
        !h3t_is(output, H3T_BF16, query_elements) ||
        !h3t_is(query, H3T_BF16, query_elements) ||
        !h3t_is(key, H3T_BF16, kv_elements) ||
        !h3t_is(value, H3T_BF16, kv_elements)) {
        return h3t_set_error(error, error_size,
                             "invalid attention tensors or shape");
    }
    uint32_t threads = 128;
    while (threads < head_dim) threads <<= 1;
    dim3 grid(query_heads, sequence, 1);
    h3t_attention_kernel<<<grid, threads, 0, context->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)query->data,
        (const __nv_bfloat16 *)key->data, (const __nv_bfloat16 *)value->data,
        sequence, query_heads, kv_heads, head_dim, scale);
    return h3t_cuda_ok(cudaGetLastError(), "causal GQA", error, error_size);
}

enum h3t_binary_kind { H3T_ADD, H3T_SILU_MUL };

/* `h3_binary_bf16_kernel` (`:2347-2357`). */
__global__ static void h3t_binary_kernel(__nv_bfloat16 *output,
                                         const __nv_bfloat16 *left,
                                         const __nv_bfloat16 *right,
                                         size_t count, int kind) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float a = __bfloat162float(left[index]);
    float b = __bfloat162float(right[index]);
    float value = kind == H3T_ADD ? a + b : a / (1.0f + expf(-a)) * b;
    output[index] = __float2bfloat16(value);
}

static int h3t_binary(h3t_context *context, h3t_tensor *output,
                      const h3t_tensor *left, const h3t_tensor *right,
                      size_t elements, int kind, const char *label,
                      char *error, size_t error_size) {
    if (!h3t_is(output, H3T_BF16, elements) ||
        !h3t_is(left, H3T_BF16, elements) ||
        !h3t_is(right, H3T_BF16, elements)) {
        return h3t_set_error(error, error_size, "invalid %s tensors", label);
    }
    if (elements) {
        h3t_binary_kernel<<<(elements + 255) / 256, 256, 0, context->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)left->data,
            (const __nv_bfloat16 *)right->data, elements, kind);
    }
    return h3t_cuda_ok(cudaGetLastError(), label, error, error_size);
}

int h3t_add(h3t_context *context, h3t_tensor *output,
            const h3t_tensor *left, const h3t_tensor *right, size_t elements,
            char *error, size_t error_size) {
    return h3t_binary(context, output, left, right, elements, H3T_ADD,
                      "BF16 add", error, error_size);
}

int h3t_silu_mul(h3t_context *context, h3t_tensor *output,
                 const h3t_tensor *gate, const h3t_tensor *up,
                 size_t elements, char *error, size_t error_size) {
    return h3t_binary(context, output, gate, up, elements, H3T_SILU_MUL,
                      "BF16 SiLU multiply", error, error_size);
}
