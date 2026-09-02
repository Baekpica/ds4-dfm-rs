#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    // Post-Volta independent thread scheduling: the store[] writes above are done by
    // some warp lanes and read below by other lanes. Without an explicit warp barrier
    // those shared-memory writes are not guaranteed visible to the cross-lane reads,
    // so a lane could read a stale/uninitialized store[] slot -> wrong compacted
    // expert ids -> nondeterministic MoE routing. On GB10 (sm_121) this realized as
    // the cont-multiseq non-determinism + BOS-spam (compute-sanitizer racecheck:
    // "RAW hazard at __shared__" between the store writes and these reads).
    __syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
        ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

// ds4 local (P5): large-n variant of mm_ids_helper with NO shared-memory
// staging.  The smem kernel above stages the compact (it, iex_used) list in
// n_tokens*4 B of dynamic shared memory, capping n_tokens at smpbo/4 (~25k on
// GB10); the routed-MoE down matmul passes n_tokens = assignment rows
// (6x the forward width), so 8192-row prefill chunks hit 48384 "tokens" and
// the whole MoE block used to fall back to the pre-mmq expert-tile kernels.
// This variant runs the SAME per-expert scan twice: pass 0 only counts
// (nex_prev, bucket size), pass 1 re-scans and writes ids_src1/ids_dst
// directly at their final offsets.  Bit-identical to the smem kernel by
// construction: identical (it, iex_used) tuples in identical token order and
// identical output expressions — the (lossless) 22/10-bit store round-trip is
// simply removed.  The rescan is cheap: the ids array at the target shape is
// ~190 KB and L2-resident across all expert blocks.  Like the smem kernel,
// behavior is undefined if one token lists the same expert in multiple slots
// (the router's top-k is without replacement, so this cannot occur).
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper_global(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    int nex_prev         = 0; // Number of columns for experts with a lower index.
    int it_compact_count = 0; // Bucket size for this expert (pass-0 result).

#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        int it_compact = 0; // Running index for the compact slice of this expert.

        if constexpr (n_expert_used_template == 0) {
            // Generic implementation:
            for (int it = 0; it < n_tokens; ++it) {
                int iex_used = -1; // The index at which the expert is used, if any.
                for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                    const int expert_used = ids[it*si1 + iex];
                    if (pass == 0) {
                        nex_prev += expert_used < expert;
                    }
                    if (expert_used == expert) {
                        iex_used = iex;
                    }
                }

                if (pass == 1 && iex_used != -1) {
                    ids_src1[nex_prev + it_compact] = it*sis1          + iex_used % nchannels_y;
                    ids_dst [nex_prev + it_compact] = it*n_expert_used + iex_used;
                }

                if (warp_reduce_any<warp_size>(iex_used != -1)) {
                    it_compact++;
                }
            }
        } else {
            // Implementation optimized for specific numbers of experts used:
            static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
            const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
            for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
                const int it = it0 + threadIdx.x / neu_padded;

                const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
                const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;
                const int iex_used = expert_used == expert ? iex : -1;
                if (pass == 0) {
                    nex_prev += expert_used < expert;
                }

                // Whether the threads at this token position have used the expert:
                const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

                // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
                int it_compact_add_lower = 0;
#pragma unroll
                for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                    const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                    if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                        it_compact_add_lower += tmp;
                    }
                }

                if (pass == 1 && iex_used != -1) {
                    const int itc = it_compact + it_compact_add_lower;
                    ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
                    ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
                }

                // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
                it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
            }
        }

        if (pass == 0) {
            it_compact_count = it_compact;
            nex_prev = warp_reduce_sum<warp_size>(nex_prev);
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact_count;
}

// ds4 local: kill switch for the large-n global path (DS4_MMID_LARGE=0).
// With the switch off the ds4_mmq.cu callers refuse past-cap shapes exactly
// as before (whole-MoE fallback to the expert-tile kernels).
bool ds4_mmid_large_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("DS4_MMID_LARGE");
        cached = !(env && env[0] == '0');
    }
    return cached != 0;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);

    // ds4 local (P5): past the smem cap, take the two-pass global variant
    // (bit-identical outputs, see mm_ids_helper_global).  One-shot stderr
    // line = path proof for the gate harnesses.
    if (nbytes_shared > smpbo) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "ds4: mm_ids_helper large-n global path engaged (P5, n_tokens=%d > cap %zu)\n",
                    n_tokens, smpbo / sizeof(mm_ids_helper_store));
        }
        mm_ids_helper_global<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
            (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
        return;
    }

    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
}

// ds4 local: kill switch for the case-1 fast path below (DS4_MMID_CASE1=0).
static bool ds4_mmid_case1_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("DS4_MMID_CASE1");
        cached = !(env && env[0] == '0');
    }
    return cached != 0;
}

// ----------------------------------------------------------------------------
// ds4 local: chunked counting-sort id map for large n.
//
// The per-expert warp scans above cost O(n_experts * n_tokens) warp
// iterations: ~7.7 ms for Qwen's 8,025-token x top-10 gate/up map and
// ~2.3 ms for each 80,250-row down map, three launches per layer at an 8K
// prefill.  This builds the same map as a chunked stable counting sort:
//
//   count   : per (token chunk, expert) match counts plus the chunk's
//             negative-id count                          [grid = chunks]
//   scan    : column scan over chunks, exclusive scan over experts,
//             writes expert_bounds                       [grid = 1]
//   scatter : each block re-reads its chunk; thread e walks the chunk in
//             (token, slot) order and writes expert e's matches at
//             offset[chunk][e] + rank                    [grid = chunks]
//
// Bit-identical to mm_ids_helper by construction: expert e's bucket starts
// at #{ids < e} (negative ids sit below expert 0 and stay unwritten, ids
// >= n_experts are dropped), entries inside a bucket follow token then
// slot order, and the output expressions are the same.  Undefined for a
// token that lists one expert twice, like the scans above.
static constexpr int DS4_MMID_CHUNK_TOKENS = 128;
static constexpr int DS4_MMID_MAX_EXPERTS = 4096;
// Below this the per-expert scatter walk has too few threads to pay off
// (every routed family here has 256 or more experts).
static constexpr int DS4_MMID_MIN_EXPERTS = 128;
static constexpr int DS4_MMID_MAX_USED = 32;
static constexpr int DS4_MMID_THREADS = 256;
static constexpr int DS4_MMID_SCAN_THREADS = 1024;
// Below this many (token, slot) rows the warp scans are already cheap and
// decode-width captured graphs must not see the scratch allocation.
static constexpr int DS4_MMID_FAST_MIN_ROWS = 4096;

static __global__ void ds4_mmid_count_kernel(
        const int32_t * __restrict__ ids, int32_t * __restrict__ counts,
        int32_t * __restrict__ negatives,
        const int n_tokens, const int n_expert_used, const int n_experts, const int si1) {
    extern __shared__ int32_t mmid_hist[];   // n_experts + 1 (negatives last)
    const int chunk = blockIdx.x;
    const int t0 = chunk * DS4_MMID_CHUNK_TOKENS;
    const int t1 = min(t0 + DS4_MMID_CHUNK_TOKENS, n_tokens);
    for (int e = threadIdx.x; e <= n_experts; e += blockDim.x) {
        mmid_hist[e] = 0;
    }
    __syncthreads();
    const int n_ids = (t1 - t0) * n_expert_used;
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int it = t0 + i / n_expert_used;
        const int iex = i - (it - t0) * n_expert_used;
        const int e = ids[it*si1 + iex];
        if (e < 0) {
            atomicAdd(&mmid_hist[n_experts], 1);
        } else if (e < n_experts) {
            atomicAdd(&mmid_hist[e], 1);
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        counts[(size_t)chunk*n_experts + e] = mmid_hist[e];
    }
    if (threadIdx.x == 0) {
        negatives[chunk] = mmid_hist[n_experts];
    }
}

// counts[chunk][e] in, offsets[chunk][e] out (start of that chunk's slice of
// expert e's bucket); expert_bounds gets the bucket starts and the total.
static __global__ void ds4_mmid_scan_kernel(
        int32_t * __restrict__ counts, const int32_t * __restrict__ negatives,
        int32_t * __restrict__ expert_bounds,
        const int n_chunks, const int n_experts) {
    extern __shared__ int32_t mmid_sizes[];   // n_experts
    __shared__ int32_t mmid_negative_total;
    if (threadIdx.x == 0) {
        int total = 0;
        for (int c = 0; c < n_chunks; ++c) {
            total += negatives[c];
        }
        mmid_negative_total = total;
    }
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        int run = 0;
        for (int c = 0; c < n_chunks; ++c) {
            const size_t k = (size_t)c*n_experts + e;
            const int v = counts[k];
            counts[k] = run;
            run += v;
        }
        mmid_sizes[e] = run;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = mmid_negative_total;
        for (int e = 0; e < n_experts; ++e) {
            const int v = mmid_sizes[e];
            mmid_sizes[e] = run;
            expert_bounds[e] = run;
            run += v;
        }
        expert_bounds[n_experts] = run;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        const int base = mmid_sizes[e];
        for (int c = 0; c < n_chunks; ++c) {
            counts[(size_t)c*n_experts + e] += base;
        }
    }
}

static __global__ void ds4_mmid_scatter_kernel(
        const int32_t * __restrict__ ids, const int32_t * __restrict__ offsets,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_tokens, const int n_expert_used, const int n_experts,
        const int nchannels_y, const int si1, const int sis1) {
    extern __shared__ int32_t mmid_chunk[];   // DS4_MMID_CHUNK_TOKENS * n_expert_used
    const int chunk = blockIdx.x;
    const int t0 = chunk * DS4_MMID_CHUNK_TOKENS;
    const int t1 = min(t0 + DS4_MMID_CHUNK_TOKENS, n_tokens);
    const int n_ids = (t1 - t0) * n_expert_used;
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int it = t0 + i / n_expert_used;
        const int iex = i - (it - t0) * n_expert_used;
        mmid_chunk[i] = ids[it*si1 + iex];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        int pos = offsets[(size_t)chunk*n_experts + e];
        for (int i = 0; i < n_ids; ++i) {
            if (mmid_chunk[i] != e) {
                continue;
            }
            const int it = t0 + i / n_expert_used;
            const int iex = i - (it - t0) * n_expert_used;
            ids_src1[pos] = it*sis1          + iex % nchannels_y;
            ids_dst [pos] = it*n_expert_used + iex;
            pos++;
        }
    }
}

// ds4 local: kill switch for the counting-sort map (DS4_MMID_FAST=0) and a
// programmatic override for the parity harness.
static int g_ds4_mmid_fast_override = -1;

void ds4_mmid_fast_set_enabled(int enabled) {
    g_ds4_mmid_fast_override = enabled ? 1 : 0;
}

static bool ds4_mmid_fast_enabled(void) {
    if (g_ds4_mmid_fast_override >= 0) {
        return g_ds4_mmid_fast_override != 0;
    }
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("DS4_MMID_FAST");
        cached = !(env && env[0] == '0');
    }
    return cached != 0;
}

static bool ds4_mmid_fast_shape_ok(const int n_experts, const int n_tokens, const int n_expert_used) {
    return n_experts >= DS4_MMID_MIN_EXPERTS && n_experts <= DS4_MMID_MAX_EXPERTS &&
           n_expert_used > 0 && n_expert_used <= DS4_MMID_MAX_USED &&
           n_tokens > 0 &&
           (int64_t)n_tokens * n_expert_used >= DS4_MMID_FAST_MIN_ROWS &&
           (int64_t)n_tokens * n_expert_used <= INT_MAX;
}

// Scratch the counting sort needs for a shape, 0 when the shape stays on the
// warp scans (or the sort is disabled).  Callers take it from their stream-
// ordered pool: a per-call cudaMallocAsync/cudaFreeAsync pair cost ~1.6 ms
// here because the default mempool hands the pages back every time.
size_t ds4_mmid_fast_scratch_bytes(const int n_experts, const int n_tokens, const int n_expert_used) {
    if (!ds4_mmid_fast_enabled() || !ds4_mmid_fast_shape_ok(n_experts, n_tokens, n_expert_used)) {
        return 0;
    }
    const int n_chunks = (n_tokens + DS4_MMID_CHUNK_TOKENS - 1) / DS4_MMID_CHUNK_TOKENS;
    return (size_t)n_chunks * (size_t)n_experts * sizeof(int32_t) + (size_t)n_chunks * sizeof(int32_t);
}

static bool ds4_mmid_fast_launch(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        void * scratch_mem, const size_t scratch_bytes, cudaStream_t stream) {
    if (!scratch_mem || nchannels_y <= 0 ||
        !ds4_mmid_fast_shape_ok(n_experts, n_tokens, n_expert_used) ||
        scratch_bytes < ds4_mmid_fast_scratch_bytes(n_experts, n_tokens, n_expert_used)) {
        return false;
    }
    const int n_chunks = (n_tokens + DS4_MMID_CHUNK_TOKENS - 1) / DS4_MMID_CHUNK_TOKENS;
    int32_t * scratch = (int32_t *)scratch_mem;
    int32_t * counts = scratch;
    int32_t * negatives = scratch + (size_t)n_chunks * n_experts;
    const size_t hist_bytes = ((size_t)n_experts + 1u) * sizeof(int32_t);
    const size_t chunk_bytes = (size_t)DS4_MMID_CHUNK_TOKENS * n_expert_used * sizeof(int32_t);
    ds4_mmid_count_kernel<<<n_chunks, DS4_MMID_THREADS, hist_bytes, stream>>>(
        ids, counts, negatives, n_tokens, n_expert_used, n_experts, si1);
    ds4_mmid_scan_kernel<<<1, DS4_MMID_SCAN_THREADS, (size_t)n_experts * sizeof(int32_t), stream>>>(
        counts, negatives, expert_bounds, n_chunks, n_experts);
    ds4_mmid_scatter_kernel<<<n_chunks, DS4_MMID_THREADS, chunk_bytes, stream>>>(
        ids, counts, ids_src1, ids_dst, n_tokens, n_expert_used, n_experts, nchannels_y, si1, sis1);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        fprintf(stderr, "ds4: counting-sort mm_ids map failed: %s\n", cudaGetErrorString(launch));
        return false;
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "ds4: counting-sort mm_ids map engaged (n_tokens=%d used=%d experts=%d)\n",
                n_tokens, n_expert_used, n_experts);
    }
    return true;
}

void ggml_cuda_launch_mm_ids_helper_scratch(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        void * scratch, const size_t scratch_bytes, cudaStream_t stream) {
    if (ds4_mmid_fast_enabled() &&
        ds4_mmid_fast_launch(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1,
                             scratch, scratch_bytes, stream)) {
        return;
    }
    ggml_cuda_launch_mm_ids_helper(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    switch (n_expert_used) {
        case  1:
            // ds4 local: the routed-MoE down matmul reinterprets (token, slot)
            // assignment rows as single-expert "tokens" (ds4_mmq.cu
            // ds4_mmq_moe_impl, n_expert_used=1).  Without this case it fell to
            // the generic <0> template: one warp per expert scanning all
            // assignment rows with a single active lane -> 22.5 ms/launch at
            // W4096 prefill (2.90 s of a 12k admission).  The optimized template
            // at neu_padded=1 covers 32 rows/iteration and emits bit-identical
            // id maps (proto_mm_ids.cu: parity on uniform/skewed/2%-invalid/
            // decode shapes, 20.4x at the W4096 shape).  DS4_MMID_CASE1=0
            // reverts to the generic path.
            if (ds4_mmid_case1_enabled()) {
                launch_mm_ids_helper< 1>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            } else {
                launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            }
            break;
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
    }
}
