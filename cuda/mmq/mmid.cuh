#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream);

// ds4 local (P5): whether the large-n global-memory mm_ids path is enabled
// (default on; DS4_MMID_LARGE=0 reverts callers to the past-cap refusal).
bool ds4_mmid_large_enabled(void);

// ds4 local: chunked counting-sort id map for large routed shapes (see
// mmid.cu).  Callers size a stream-ordered scratch with
// ds4_mmid_fast_scratch_bytes (0 = shape stays on the warp scans) and pass
// it to the _scratch launcher, which falls back to the scans on any refusal.
size_t ds4_mmid_fast_scratch_bytes(int n_experts, int n_tokens, int n_expert_used);

void ggml_cuda_launch_mm_ids_helper_scratch(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1,
        void * scratch, size_t scratch_bytes, cudaStream_t stream);

// Override for the parity harness (tests/test_mmid_fast.cu); production reads
// DS4_MMID_FAST (default on).
void ds4_mmid_fast_set_enabled(int enabled);
