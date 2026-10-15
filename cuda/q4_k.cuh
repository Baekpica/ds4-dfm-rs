#ifndef DS4_Q4_K_CUH
#define DS4_Q4_K_CUH

enum { DS4_Q4_K_BLOCK = 256, DS4_Q4_K_BYTES = 144 };

// GGML Q4_K: two half scales, eight packed six-bit scale/min pairs,
// then paired low/high-nibble groups. Shared by row gathers and direct dots.
__device__ __forceinline__ float ds4_q4_k_value(const uint8_t *block, uint32_t j) {
    const uint32_t group = j / 32, offset = j % 32;
    const uint8_t *s = block + 4;
    const uint32_t scale = group < 4 ? s[group] & 63u :
        (s[group + 4] & 15u) | ((s[group - 4] >> 6) << 4);
    const uint32_t min = group < 4 ? s[group + 4] & 63u :
        (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
    const uint8_t byte = block[16 + group / 2 * 32 + offset];
    const uint32_t q = group & 1 ? byte >> 4 : byte & 15u;
    const __half *h = (const __half *)block;
    return __half2float(h[0]) * scale * q - __half2float(h[1]) * min;
}
#endif
