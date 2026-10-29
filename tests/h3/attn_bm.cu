/*
 * Bounded reproducer for `h3d_sdpa`, the BF16 attention kernel: it times the
 * kernel at a denoise shape and hashes the output, so a change can be measured
 * and checked bit-equal without loading the model. Build it with
 * `make tests/h3/attn_bm` (the same H3_NVCCFLAGS and sibling objects as
 * `tests/h3/native_test`), then run:
 *
 *   tests/h3/attn_bm 4352 56 5     the 512x512, 56-frame denoise shape
 *   tests/h3/attn_bm 18432 56 2    the 768x768, 107-frame denoise shape
 *
 * The shapes are `video_latent_t(frames) * (latent/2)^2` tokens at 56 heads of
 * 128 dimensions, the two the speed campaign measures (see
 * docs/minimax-h3/speed-2026-10-27.md). A change to the kernel must leave the
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
static void fill(std::vector<__nv_bfloat16> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = __float2bfloat16((float)(state >> 8) / 8388608.0f - 1.0f);
    }
}

int main(int argc, char **argv) {
    const uint32_t sequence = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 4352;
    const uint32_t heads = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 56;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 3;
    constexpr uint32_t head_dim = 128;
    char error[512] = {};

    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) {
        fail("create", error);
    }

    const size_t elements = (size_t)sequence * heads * head_dim;
    h3d_tensor *query = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    h3d_tensor *key = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    h3d_tensor *value = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    h3d_tensor *output = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    if (!query || !key || !value || !output) {
        fail("alloc", error);
    }

    std::vector<__nv_bfloat16> host(elements);
    uint32_t seed = 1;
    for (auto *tensor : {query, key, value}) {
        fill(host, seed++);
        if (!h3d_upload(gpu, tensor, host.data(), error, sizeof(error))) {
            fail("upload", error);
        }
    }

    const float scale = 1.0f / std::sqrt((float)head_dim);
    if (!h3d_sdpa(gpu, output, query, key, value, sequence, heads, head_dim,
                  scale, error, sizeof(error))) {
        fail("sdpa", error);
    }
    if (!h3d_sync(gpu, error, sizeof(error))) {
        fail("sync", error);
    }

    auto start = std::chrono::steady_clock::now();
    for (int call = 0; call < calls; call++) {
        if (!h3d_sdpa(gpu, output, query, key, value, sequence, heads, head_dim,
                      scale, error, sizeof(error))) {
            fail("sdpa", error);
        }
    }
    if (!h3d_sync(gpu, error, sizeof(error))) {
        fail("sync", error);
    }
    auto finish = std::chrono::steady_clock::now();
    const double total_ms =
            std::chrono::duration<double, std::milli>(finish - start).count();

    std::vector<__nv_bfloat16> result(elements);
    if (!h3d_read(gpu, result.data(), output, error, sizeof(error))) {
        fail("read", error);
    }

    std::printf("sequence=%u heads=%u calls=%d ms_per_call=%.3f out_fnv=%016llx\n",
                sequence, heads, calls, total_ms / calls,
                (unsigned long long)fnv1a(result.data(), result.size() * 2));
    std::printf("first=%.6f last=%.6f\n", __bfloat162float(result.front()),
                __bfloat162float(result.back()));

    /* The tensor-core arm is a measurement path, not a gated one, so its
     * deviation from the byte-exact kernel has to be quantified rather than
     * hashed. `H3_ATTN_DUMP=<path>` writes this arm's raw BF16 output, and a
     * fifth argument reads a dump of the other arm and prints the difference.
     * Both arms round to BF16 at the same place, so this compares what the
     * model would actually read. */
    const char *dump = std::getenv("H3_ATTN_DUMP");
    if (dump && *dump) {
        std::FILE *file = std::fopen(dump, "wb");
        if (!file) {
            fail("dump", dump);
        }
        std::fwrite(result.data(), 2, result.size(), file);
        std::fclose(file);
    }
    if (argc > 4) {
        std::FILE *file = std::fopen(argv[4], "rb");
        if (!file) {
            fail("reference", argv[4]);
        }
        std::vector<__nv_bfloat16> reference(elements);
        if (std::fread(reference.data(), 2, reference.size(), file) !=
            reference.size()) {
            fail("reference", "short read");
        }
        std::fclose(file);
        double worst = 0.0, square = 0.0;
        size_t differing = 0;
        for (size_t index = 0; index < reference.size(); index++) {
            const double left = __bfloat162float(result[index]);
            const double right = __bfloat162float(reference[index]);
            const double delta = left > right ? left - right : right - left;
            if (delta > worst) {
                worst = delta;
            }
            square += delta * delta;
            if (delta != 0.0) {
                differing++;
            }
        }
        std::printf("against=%s max_abs=%.8g rms=%.8g differing=%zu of %zu "
                    "(%.4f%%)\n",
                    argv[4], worst,
                    std::sqrt(square / (double)reference.size()), differing,
                    reference.size(),
                    100.0 * (double)differing / (double)reference.size());
    }

    h3d_free(output);
    h3d_free(value);
    h3d_free(key);
    h3d_free(query);
    h3d_destroy(gpu);
    return 0;
}
