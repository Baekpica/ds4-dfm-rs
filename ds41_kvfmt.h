/* ds41_kvfmt.h — the V4.1 packed-KV group geometry, one copy for both sides.
 *
 * The CUDA pack/unpack kernels (cuda/ds41_attn.cuh) and the host allocator
 * (ds4_ds41_forward.inc) both reference these; the engine keeps the same
 * constants once (ds4_gpu_v41.h:229-234) for exactly this reason: allocate
 * for one layout and pack for another and the write overruns silently into
 * the next layer's cache, with no error.
 *
 *   main KV : 512 dims = 256 B nibbles + 32 B E4M3 scales = 288 B/group
 *             (the official 890 B/token account: 512 FP4 + 32 scales)
 *   index K : 128 dims =  64 B nibbles +  4 B E8M0 scales = 68 B padded to 72
 *             (the 4-byte pad keeps a warp's 64 B read on one 32 B sector)
 */
#ifndef DS41_KVFMT_H
#define DS41_KVFMT_H

#define DS4_V41_CKV_BLK    16u
#define DS4_V41_CKV_NIB   256u
#define DS4_V41_CKV_BYTES 288u
#define DS4_V41_IDXK_BLK   32u
#define DS4_V41_IDXK_NIB   64u
#define DS4_V41_IDXK_BYTES 72u

#endif /* DS41_KVFMT_H */
