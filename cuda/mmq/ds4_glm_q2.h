/* Private GLM down admission; no inference ABI or generic-Q2 promotion. */
#pragma once
#include <climits>
#include <cstdint>

enum class MoePolicy { Generic, GlmQ2Down };
enum class MoeLayout { Raw, SoA };

enum { GLM_Q2_SPARK_CC = 1210, GLM_Q2_TYPE = 10, GLM_Q2_M = 4096,
       GLM_Q2_K = 2048, GLM_Q2_MIN_ROWS = 256, GLM_Q2_MIN_EXPERTS = 32,
       GLM_Q2_BLOCK_BYTES = 84, GLM_Q2_BLOCK_K = 256 };

static constexpr MoePolicy glm_q2_policy(int cc, uint32_t type, int m, int k,
        int rows, int experts, int used, uint64_t stride, MoeLayout layout) {
    if (cc != GLM_Q2_SPARK_CC || type != GLM_Q2_TYPE || layout != MoeLayout::Raw ||
        m != GLM_Q2_M || k != GLM_Q2_K || used != 1 ||
        rows < GLM_Q2_MIN_ROWS || experts < GLM_Q2_MIN_EXPERTS || experts == INT_MAX) {
        return MoePolicy::Generic;
    }
    constexpr uint64_t row_blocks = GLM_Q2_K / GLM_Q2_BLOCK_K;
    constexpr uint64_t expert_blocks = row_blocks * GLM_Q2_M;
    if (stride % GLM_Q2_BLOCK_BYTES || stride / GLM_Q2_BLOCK_BYTES < expert_blocks ||
        stride / GLM_Q2_BLOCK_BYTES > INT_MAX || uint64_t(rows) * GLM_Q2_M > INT_MAX) {
        return MoePolicy::Generic;
    }
    /* Existing MMQ tile offsets are signed ints. Prove the complete packed
     * table, including the last row, before admitting the private schedule. */
    const uint64_t count = uint64_t(experts - 1) * (stride / GLM_Q2_BLOCK_BYTES) + expert_blocks;
    return count <= INT_MAX ? MoePolicy::GlmQ2Down : MoePolicy::Generic;
}
