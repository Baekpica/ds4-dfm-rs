/* gemma4_gpu.h - device residency and the fused VQ row dot for the
 * Gemma-4-26B-A4B release, behind a plain C ABI.
 *
 * The caller (crates/ds4-core, example gemma4_a4b_gpu_gate) opens a
 * vq-release/1 artifact, lists every encoded container with the shape the
 * release records, and this module maps, checks and uploads each one once.
 * A matvec then runs wholly on the device: one fused gather-and-dot per
 * payload, stages summed in the payload's own gain, nothing decoded into a
 * weight matrix. A row-list gather (gm4_gpu_rows) reads only the rows a
 * mixture layer routed to, from the same resident containers.
 *
 * The payload is the TOOLS' DQV3 (24-byte header, inline f16 codebook,
 * min(nbit, 12)-bit row stream, no plane), which is what this artifact
 * carries. It is not the engine's DQV3 in /data/YoungAi (32-byte header,
 * layer-shared codebook at cb_off, fixed 12 bits); the discriminator is the
 * word at payload offset 20, zero here and the engine's mnb there.
 *
 * Failures are returned as -1 with a message in the caller's buffer; nothing
 * falls back to host streaming, because a silent fallback would hide exactly
 * the residency failure this module exists to report.
 */
#ifndef GEMMA4_GPU_H
#define GEMMA4_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *path;   /* an RQVQ container file */
    uint32_t rows;      /* the release's shape, checked against the header */
    uint32_t cols;
} gm4_spec;

typedef struct {
    uint64_t files_bytes;       /* mmapped file size, all containers */
    uint64_t payload_bytes;     /* sum of the container payload region */
    uint64_t device_bytes;      /* cudaMalloc'd, including slack and padding */
    uint64_t device_free_before;
    uint64_t device_free_after;
    int tensors;
    int payloads;
} gm4_account;

typedef struct gm4_gpu gm4_gpu;

/* Upload every container in `specs`. Returns NULL on failure, with the reason
 * in `err`. The module is a singleton: cudaSetDevice(0) is fixed at open. */
gm4_gpu *gm4_gpu_open(const gm4_spec *specs, int n, char *err, size_t errlen);
void gm4_gpu_close(gm4_gpu *g);

/* What was allocated, and the device's free memory before and after. */
void gm4_gpu_account_of(const gm4_gpu *g, gm4_account *out);

/* The device name the residency was measured on ("NVIDIA GeForce RTX 4070
 * SUPER" here), owned by the handle. */
const char *gm4_gpu_device_name(const gm4_gpu *g);

/* y[rows] = sum over the container's payloads of payload_dot(row) * gain,
 * with x[cols] read from the host. `threads` is the launch width of the
 * warp-per-row kernels (a multiple of 32 up to 1024); it changes how many
 * warps share a block, never the arithmetic. Returns 0, or -1 with `err`. */
int gm4_gpu_matvec(gm4_gpu *g, int tensor, const float *x, float *y, int threads,
                   char *err, size_t errlen);

/* y[i] = the dot of x with row rows[i], in the order given, summed over the
 * container's payloads with each stage's own gain. The list is host row
 * numbers of the whole tensor; it is uploaded once per call and every entry
 * must be below the tensor's row count. This is the routed-expert gather:
 * eight chosen experts touch 11264 of the 180224 rows of
 * `experts.gate_up_proj` and 22528 of the 360448 rows of
 * `experts.down_proj`, so the read is the selected rows and their gains, not
 * the stack. A repeated row produces a repeated output. */
int gm4_gpu_rows(gm4_gpu *g, int tensor, const uint32_t *rows, int nrows,
                 const float *x, float *y, int threads, char *err, size_t errlen);

/* Which kernel a tensor's payloads run: 3 = the DQV3 block kernel, 2 = the
 * dim-4 fast path, 1 = the generic warp kernel. Per payload; they agree for
 * a real container because the geometry is constant across its payloads. */
int gm4_gpu_variant(const gm4_gpu *g, int tensor);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_GPU_H */
