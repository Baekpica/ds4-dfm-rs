#ifndef DS4_QUANTS_H
#define DS4_QUANTS_H

/*
 * Narrow quantization API used by the DS4 GGUF writer.
 *
 * The enum values intentionally match GGUF/GGML type IDs so template metadata
 * can be copied without translation.  Only the formats used by the DS4 Flash
 * quantization recipes are implemented as output targets.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4Q_MAX_DIMS 4

typedef enum {
    DS4Q_TYPE_F32     = 0,
    DS4Q_TYPE_F16     = 1,
    DS4Q_TYPE_Q4_0    = 2,
    DS4Q_TYPE_Q4_1    = 3,
    DS4Q_TYPE_Q5_0    = 6,
    DS4Q_TYPE_Q5_1    = 7,
    DS4Q_TYPE_Q8_0    = 8,
    DS4Q_TYPE_Q8_1    = 9,
    DS4Q_TYPE_Q2_K    = 10,
    DS4Q_TYPE_Q3_K    = 11,
    DS4Q_TYPE_Q4_K    = 12,
    DS4Q_TYPE_Q5_K    = 13,
    DS4Q_TYPE_Q6_K    = 14,
    DS4Q_TYPE_Q8_K    = 15,
    DS4Q_TYPE_IQ2_XXS = 16,
    DS4Q_TYPE_IQ2_XS  = 17,
    DS4Q_TYPE_IQ3_XXS = 18,
    DS4Q_TYPE_IQ1_S   = 19,
    DS4Q_TYPE_IQ4_NL  = 20,
    DS4Q_TYPE_IQ3_S   = 21,
    DS4Q_TYPE_IQ2_S   = 22,
    DS4Q_TYPE_IQ4_XS  = 23,
    DS4Q_TYPE_I8      = 24,
    DS4Q_TYPE_I16     = 25,
    DS4Q_TYPE_I32     = 26,
    DS4Q_TYPE_I64     = 27,
    DS4Q_TYPE_F64     = 28,
    DS4Q_TYPE_IQ1_M   = 29,
    DS4Q_TYPE_BF16    = 30,
    /* ds4-private: Go-domain strict-binary (+/-1) 1-bit PER-ROW quantizer (go-onebit).
     * Variable-length row record (one fp16 scale + packed sign words); research/
     * calibration encoder only. CLI name "go1b_row". Fills an unused in-range slot;
     * not a standard GGUF/GGML id, only ds4 tooling emits/consumes it. The runtime
     * (Metal kernel) reads the BLOCK variant below, NOT this per-row record. */
    DS4Q_TYPE_GO1B    = 31,
    DS4Q_TYPE_TQ1_0   = 34,
    DS4Q_TYPE_TQ2_0   = 35,
    DS4Q_TYPE_MXFP4   = 39,
    /* ds4-private: Go-domain strict-binary (+/-1) 1-bit BLOCK quantizer — the format
     * the ds4 runtime / Metal `block_go1b` kernel actually consumes. Fixed 256-element
     * block = fp16 row scale (replicated per block) + 32 sign bytes = 34 bytes. CLI
     * name "go1b"; emitted for routed experts. PINNED on-disk ggml type number = 40
     * (repurposes the prior never-emitted NVFP4 placeholder that sat in this slot;
     * NVFP4 had can_quantize=false and was referenced nowhere else). */
    DS4Q_TYPE_GO1B_BLK = 40,
    DS4Q_TYPE_Q1_0    = 41,
    DS4Q_TYPE_COUNT   = 42,
} ds4q_type;

static inline size_t ds4q_pad(size_t x, size_t n) {
    return ((x + n - 1) / n) * n;
}

const char *ds4q_type_name(ds4q_type type);
bool ds4q_can_quantize(ds4q_type type);
int64_t ds4q_block_size(ds4q_type type);
size_t ds4q_row_size(ds4q_type type, int64_t ne);
bool ds4q_requires_imatrix(ds4q_type type);
void ds4q_quantize_init(ds4q_type type);
size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix);

/* IQ2_XXS 搜索表导出(GPU 编码器上传用; 调用即触发建表)。 */
void ds4q_iq2_xxs_tables(const uint64_t **grid, const int **map, const uint16_t **neighbours,
                         int *grid_size, int *map_size, int64_t *neighbours_len);

/* --gpu-verify: GPU 编码后再跑一遍 CPU 编码器逐字节对拍(慢, 只用于闸门)。
 * 解析器(deepseek4-quantize)与读点(quants.c)是不同 TU, 走这个外部 setter。 */
void ds4q_gpu_verify_set(int on);

float ds4q_f16_to_f32(uint16_t bits);
float ds4q_bf16_to_f32(uint16_t bits);
void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n);
void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n);

#endif
