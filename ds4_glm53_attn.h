#ifndef DS4_GLM53_ATTN_H
#define DS4_GLM53_ATTN_H

#include "ds4_glm53_compact.h"

typedef enum { GLM53_ATTN_ALL, GLM53_ATTN_SELECTED } glm53_attn_mode;

enum { GLM53_DENSE_GROUP = 8, GLM53_DENSE_ROWS = 2048,
       GLM53_DENSE_LATENT = 512, GLM53_DENSE_HEADS = 64,
       GLM53_DENSE_ALIGN = 256, GLM53_DENSE_MIN_ROWS = 128 };

static inline int glm53_dense_shape(uint32_t rows, uint32_t keys,
        uint32_t heads, uint32_t latent, glm53_attn_mode mode) {
    return mode == GLM53_ATTN_ALL && rows >= GLM53_DENSE_MIN_ROWS &&
        rows <= GLM53_DENSE_ROWS && keys >= rows &&
        keys <= DS4_GLM53_MAX_SELECTED && heads == GLM53_DENSE_HEADS &&
        latent == GLM53_DENSE_LATENT;
}

static inline uint64_t glm53_dense_align(uint64_t n) {
    return (n + GLM53_DENSE_ALIGN - 1u) & ~(uint64_t)(GLM53_DENSE_ALIGN - 1u);
}

static inline uint64_t glm53_dense_bytes(uint32_t rows, uint32_t keys) {
    if (keys > DS4_GLM53_MAX_SELECTED) { keys = DS4_GLM53_MAX_SELECTED; }
    const uint64_t query = (uint64_t)GLM53_DENSE_GROUP * rows * GLM53_DENSE_LATENT;
    const uint64_t scores = (uint64_t)GLM53_DENSE_GROUP * rows * keys;
    return glm53_dense_align(glm53_dense_align(glm53_dense_align(query * 2u)
        + scores * 4u) + scores * 2u) + query * 4u;
}

static inline int glm53_attn_frontier(uint32_t rows, uint32_t pos0,
        uint32_t cap, uint32_t stride, glm53_attn_mode mode) {
    if (!rows || pos0 > cap || rows > cap - pos0) { return 0; }
    if (mode == GLM53_ATTN_SELECTED) {
        return stride && stride <= DS4_GLM53_MAX_SELECTED;
    }
    return !stride;
}

#endif
