/*
 * Bounded reproducer for `h3d_sdpa_f32`, the video VAE's F32 attention kernel:
 * it times the kernel at a real decode shape and hashes the output, so a variant
 * can be measured and checked bit-equal without decoding a clip. Build it with
 * `make tests/h3/attn_f32_bm` (the same H3_NVCCFLAGS and sibling objects as
 * `tests/h3/attn_bm`), then run:
 *
 *   tests/h3/attn_f32_bm 16133 32 2   the 768x768 tile: 7 latent steps of
 *                                     48x48 plus the five suffix rows
 *   tests/h3/attn_f32_bm 7173 32 2    the 512x512 tile: 7 steps of 32x32 + 5
 *
 * The shapes are `CHUNK_LATENT_TIME * latent_h * latent_w + SUFFIX` rows at 32
 * heads of 64 dimensions, which is what `crates/ds4-core/src/minimax_h3/
 * vae_video.rs` passes to `h3d_sdpa_f32` once per layer per tile (36 layers,
 * 12 tiles, so 432 calls per decode). A change to the kernel must leave the
 * printed `out_fnv` unchanged; only then are its timings comparable.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

static uint64_t fnv1a(const void *data, size_t bytes) {
    constexpr uint64_t offset = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset;
    auto *bytes_data = static_cast<const unsigned char *>(data);
    for (size_t index = 0; index < bytes; index++) {
        hash = (hash ^ bytes_data[index]) * prime;
    }
    return hash;
}

/* Deterministic values in [-1, 1): the scaled scores stay finite, so the
 * softmax the kernel runs is the one the model runs. */
static void fill(std::vector<float> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = (float)(state >> 8) / 8388608.0f - 1.0f;
    }
}

int main(int argc, char **argv) {
    const uint32_t sequence = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 16133;
    const uint32_t heads = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 32;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 2;
    constexpr uint32_t head_dim = 64;
    char error[512] = {};

    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) {
        fail("create", error);
    }

    const size_t elements = (size_t)sequence * heads * head_dim;
    h3d_tensor *query = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    h3d_tensor *key = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    h3d_tensor *value = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    h3d_tensor *output = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    if (!query || !key || !value || !output) {
        fail("alloc", error);
    }

    std::vector<float> host(elements);
    uint32_t seed = 1;
    for (auto *tensor : {query, key, value}) {
        fill(host, seed++);
        if (!h3d_upload(gpu, tensor, host.data(), error, sizeof(error))) {
            fail("upload", error);
        }
    }

    const float scale = 1.0f / std::sqrt((float)head_dim);
    if (!h3d_sdpa_f32(gpu, output, query, key, value, sequence, heads, head_dim,
                      scale, error, sizeof(error))) {
        fail("sdpa", error);
    }
    if (!h3d_sync(gpu, error, sizeof(error))) {
        fail("sync", error);
    }

    auto start = std::chrono::steady_clock::now();
    for (int call = 0; call < calls; call++) {
        if (!h3d_sdpa_f32(gpu, output, query, key, value, sequence, heads,
                          head_dim, scale, error, sizeof(error))) {
            fail("sdpa", error);
        }
    }
    if (!h3d_sync(gpu, error, sizeof(error))) {
        fail("sync", error);
    }
    auto finish = std::chrono::steady_clock::now();
    const double total_ms =
            std::chrono::duration<double, std::milli>(finish - start).count();

    std::vector<float> result(elements);
    if (!h3d_read(gpu, result.data(), output, error, sizeof(error))) {
        fail("read", error);
    }

    std::printf("sequence=%u heads=%u calls=%d ms_per_call=%.3f out_fnv=%016llx\n",
                sequence, heads, calls, total_ms / calls,
                (unsigned long long)fnv1a(result.data(), result.size() * 4));
    std::printf("first=%.6f last=%.6f\n", result.front(), result.back());

    h3d_free(output);
    h3d_free(value);
    h3d_free(key);
    h3d_free(query);
    h3d_destroy(gpu);
    return 0;
}
