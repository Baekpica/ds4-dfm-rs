/* Included by the CUDA backend.  Prism Bonsai (qwen35) entry points: the
 * folded-weight activation transform the family's matmuls need.  The kernels
 * live in cuda/qwen35_primitives.cuh; the gdn-output and attention-prep
 * wrappers land with the forward graph. */

#include "cuda/qwen35_primitives.cuh"

namespace qwen35_gpu {

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static int launched(const char *what) {
    return cuda_ok(cudaGetLastError(), what);
}

/* Validate the fold's shape contract: one or more rows of n values, n a whole
 * number of block_size blocks, block_size a power of two that fits a CUDA
 * block, and (for the gdn case) a head geometry that tiles the row exactly. */
static int fold_shape(uint32_t n, uint32_t n_tok, uint32_t bs,
                      uint32_t gdn, uint32_t hd, uint32_t nk, uint32_t rep) {
    if (n == 0 || n_tok == 0) return 0;
    if (bs == 0 || bs > 1024 || (bs & (bs - 1)) != 0) return 0;
    if (n % bs != 0) return 0;
    if (gdn) {
        if (hd == 0 || nk == 0 || rep == 0) return 0;
        if ((uint64_t) hd * nk * rep != n) return 0;
    }
    return 1;
}

static int fold_launch(
        ds4_gpu_tensor *x, uint32_t n, uint32_t n_tok, uint32_t bs,
        const ds4_gpu_tensor *signs, uint32_t gdn, uint32_t hd, uint32_t nk,
        uint32_t rep, int inverse) {
    if (!fold_shape(n, n_tok, bs, gdn, hd, nk, rep)) return 0;
    if (!tensor(x, (uint64_t) n * n_tok * sizeof(float))) return 0;
    if (signs && !tensor(signs, (uint64_t) n * sizeof(float))) return 0;

    const float *src = (const float *) x->ptr;
    float *dst = (float *) x->ptr;
    if (gdn) {
        /* The reorder mixes values across the whole row, so it cannot ride
         * along inside the block-local butterfly: permute into the shared
         * CUDA temporary first, then rotate from there into the tensor. */
        float *tmp = (float *) cuda_tmp_alloc((uint64_t) n * n_tok * sizeof(float),
                                              "Bonsai gdn fold permute");
        if (!tmp) return 0;
        qwen35_cuda::fold_gdn_permute<<<dim3((n + 255u) / 256u, n_tok), 256, 0, cuda_decode_stream()>>>(
            tmp, (const float *) x->ptr, n, hd, nk, rep);
        src = tmp;
    }

    const float *sg = signs ? (const float *) signs->ptr : NULL;
    qwen35_cuda::fold_rotate<<<dim3(n / bs, n_tok), bs, bs * sizeof(float),
                               cuda_decode_stream()>>>(src, dst, sg, n, bs, inverse);
    return launched("Bonsai fold");
}

} // namespace qwen35_gpu

/* Forward fold for every folded matmul input: a' = H_bs(s * a).  When gdn is
 * set the tiled-to-grouped reorder is applied first, as the ssm_out
 * projection requires. */
extern "C" int ds4_gpu_qwen35_fold_forward_tensor(
        ds4_gpu_tensor *x, uint32_t n, uint32_t n_tok, uint32_t block_size,
        const ds4_gpu_tensor *signs, uint32_t gdn, uint32_t hd, uint32_t nk,
        uint32_t rep) {
    return qwen35_gpu::fold_launch(x, n, n_tok, block_size, signs, gdn, hd, nk,
                                   rep, /*inverse=*/0);
}

/* Inverse fold for token-embedding lookups: x = s * H_bs(z). */
extern "C" int ds4_gpu_qwen35_fold_inverse_tensor(
        ds4_gpu_tensor *x, uint32_t n, uint32_t n_tok, uint32_t block_size,
        const ds4_gpu_tensor *signs) {
    return qwen35_gpu::fold_launch(x, n, n_tok, block_size, signs, /*gdn=*/0,
                                   0, 0, 0, /*inverse=*/1);
}
