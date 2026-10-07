#ifndef DS4_GLM53_ATTN_H
#define DS4_GLM53_ATTN_H

#include "ds4_glm53_compact.h"

typedef enum { GLM53_ATTN_ALL, GLM53_ATTN_SELECTED } glm53_attn_mode;

static inline int glm53_attn_frontier(uint32_t rows, uint32_t pos0,
        uint32_t cap, uint32_t stride, glm53_attn_mode mode) {
    if (!rows || pos0 > cap || rows > cap - pos0) { return 0; }
    if (mode == GLM53_ATTN_SELECTED) {
        return stride && stride <= DS4_GLM53_MAX_SELECTED;
    }
    return !stride;
}

#endif
