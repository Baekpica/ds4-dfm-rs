/* H200 CUDA parity for Motif-3-only control and expanded-attention kernels. */
#include "ds4_gpu.h"
#include "cuda/mmq/ds4_mmq.h"
#include "cuda/mmq/ds4_mmq_d2r.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct array_value {
    uint32_t dtype;
    std::vector<uint64_t> dim;
    std::vector<uint8_t> bytes;
};

using fixture = std::unordered_map<std::string, array_value>;

static void read_exact(FILE *fp, void *dst, size_t n, const char *what) {
    if (fread(dst, 1, n, fp) != n) {
        fprintf(stderr, "short read: %s\n", what);
        std::exit(1);
    }
}

static fixture load_fixture(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); std::exit(1); }
    char magic[8];
    uint32_t version, count;
    read_exact(fp, magic, 8, "magic");
    read_exact(fp, &version, 4, "version");
    read_exact(fp, &count, 4, "count");
    if (memcmp(magic, "DS4FX1\0\0", 8) || version != 1 || count > 64) {
        fprintf(stderr, "bad fixture: %s\n", path);
        std::exit(1);
    }
    fixture result;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t name_len, dtype, ndim, reserved;
        uint64_t dim[4], nbytes;
        read_exact(fp, &name_len, 4, "name length");
        read_exact(fp, &dtype, 4, "dtype");
        read_exact(fp, &ndim, 4, "ndim");
        read_exact(fp, &reserved, 4, "reserved");
        read_exact(fp, dim, sizeof(dim), "dimensions");
        read_exact(fp, &nbytes, 8, "nbytes");
        if (!name_len || name_len > 255 || ndim > 4 || (dtype != 1 && dtype != 2)) {
            fprintf(stderr, "bad array descriptor\n"); std::exit(1);
        }
        std::string name(name_len, '\0');
        read_exact(fp, name.data(), name_len, "name");
        array_value value;
        value.dtype = dtype;
        value.dim.assign(dim, dim + ndim);
        value.bytes.resize((size_t)nbytes);
        read_exact(fp, value.bytes.data(), (size_t)nbytes, "data");
        result.emplace(std::move(name), std::move(value));
    }
    fclose(fp);
    return result;
}

static array_value &get(fixture &f, const char *name) {
    auto it = f.find(name);
    if (it == f.end()) { fprintf(stderr, "missing fixture array: %s\n", name); std::exit(1); }
    return it->second;
}

static const float *f32(fixture &f, const char *name) {
    auto &a = get(f, name);
    if (a.dtype != 1 || a.bytes.size() % sizeof(float)) std::exit(1);
    return reinterpret_cast<const float *>(a.bytes.data());
}

static const int32_t *i32(fixture &f, const char *name) {
    auto &a = get(f, name);
    if (a.dtype != 2 || a.bytes.size() % sizeof(int32_t)) std::exit(1);
    return reinterpret_cast<const int32_t *>(a.bytes.data());
}

struct gpu_tensor {
    ds4_gpu_tensor *p;
    explicit gpu_tensor(uint64_t bytes) : p(ds4_gpu_tensor_alloc(bytes)) {
        if (!p) { fprintf(stderr, "GPU allocation failed\n"); std::exit(1); }
    }
    ~gpu_tensor() { ds4_gpu_tensor_free(p); }
    gpu_tensor(const gpu_tensor &) = delete;
    gpu_tensor &operator=(const gpu_tensor &) = delete;
};

static void upload(gpu_tensor &dst, const array_value &src) {
    if (!ds4_gpu_tensor_write(dst.p, 0, src.bytes.data(), src.bytes.size())) {
        fprintf(stderr, "GPU write failed\n"); std::exit(1);
    }
}

static std::vector<float> download_f32(gpu_tensor &src, uint64_t count) {
    std::vector<float> out((size_t)count);
    if (!ds4_gpu_tensor_read(src.p, 0, out.data(), count * sizeof(float))) {
        fprintf(stderr, "GPU read failed\n"); std::exit(1);
    }
    return out;
}

static std::vector<int32_t> download_i32(gpu_tensor &src, uint64_t count) {
    std::vector<int32_t> out((size_t)count);
    if (!ds4_gpu_tensor_read(src.p, 0, out.data(), count * sizeof(int32_t))) {
        fprintf(stderr, "GPU read failed\n"); std::exit(1);
    }
    return out;
}

static std::vector<uint16_t> download_u16(gpu_tensor &src, uint64_t count) {
    std::vector<uint16_t> out((size_t)count);
    if (!ds4_gpu_tensor_read(src.p, 0, out.data(), count * sizeof(uint16_t))) {
        fprintf(stderr, "GPU read failed\n"); std::exit(1);
    }
    return out;
}

static void assert_close(const char *name, const float *got, const float *want,
                         uint64_t n, float atol, float rtol) {
    float worst = 0.0f;
    uint64_t wi = 0;
    for (uint64_t i = 0; i < n; i++) {
        const float err = std::fabs(got[i] - want[i]);
        const float limit = atol + rtol * std::fabs(want[i]);
        const float ratio = limit > 0 ? err / limit : err;
        if (ratio > worst) { worst = ratio; wi = i; }
    }
    if (worst > 1.0f) {
        fprintf(stderr, "%s mismatch at %llu: got %.9g want %.9g (%.2fx)\n",
                name, (unsigned long long)wi, got[wi], want[wi], worst);
        std::exit(1);
    }
}

static std::string path(const char *dir, const char *name) {
    return std::string(dir) + "/" + name;
}

static uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    return (uint16_t)(rounded >> 16u);
}

static float bf16_bits_to_f32(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16u;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static void assert_bits_equal(const char *name, const float *got,
                              const float *want, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        uint32_t gb, wb;
        std::memcpy(&gb, &got[i], sizeof(gb));
        std::memcpy(&wb, &want[i], sizeof(wb));
        if (gb != wb) {
            fprintf(stderr, "%s mismatch at %llu: got=0x%08x want=0x%08x\n",
                    name, (unsigned long long)i, gb, wb);
            std::exit(1);
        }
    }
}

static void test_round_bf16(void) {
    const uint64_t sizes[] = {1, 3, 4, 7, 128, 4096, 4097};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        const uint64_t n = sizes[s];
        std::vector<float> in((size_t)n);
        std::vector<float> want((size_t)n);
        for (uint64_t i = 0; i < n; i++) {
            const float x = (float)((int64_t)i - 2048) * 0.017578125f +
                            ((i & 1ull) ? 1.23456789e-4f : -9.87654321e-5f);
            in[i] = x;
            want[i] = bf16_bits_to_f32(f32_to_bf16_bits(x));
        }
        gpu_tensor src(n * sizeof(float));
        gpu_tensor dst(n * sizeof(float));
        if (!ds4_gpu_tensor_write(src.p, 0, in.data(), n * sizeof(float))) {
            fprintf(stderr, "round_bf16 upload failed\n");
            std::exit(1);
        }
        if (!ds4_gpu_motif3_round_bf16_tensor(dst.p, src.p, n)) {
            fprintf(stderr, "round_bf16 launch failed n=%llu\n",
                    (unsigned long long)n);
            std::exit(1);
        }
        const std::vector<float> got = download_f32(dst, n);
        for (uint64_t i = 0; i < n; i++) {
            uint32_t gb, wb;
            std::memcpy(&gb, &got[i], sizeof(gb));
            std::memcpy(&wb, &want[i], sizeof(wb));
            if (gb != wb) {
                fprintf(stderr,
                        "round_bf16 mismatch n=%llu i=%llu got=0x%08x want=0x%08x\n",
                        (unsigned long long)n, (unsigned long long)i, gb, wb);
                std::exit(1);
            }
        }
        if (!ds4_gpu_motif3_round_bf16_tensor(src.p, src.p, n)) {
            fprintf(stderr, "round_bf16 inplace launch failed n=%llu\n",
                    (unsigned long long)n);
            std::exit(1);
        }
        const std::vector<float> inplace = download_f32(src, n);
        for (uint64_t i = 0; i < n; i++) {
            uint32_t gb, wb;
            std::memcpy(&gb, &inplace[i], sizeof(gb));
            std::memcpy(&wb, &want[i], sizeof(wb));
            if (gb != wb) {
                fprintf(stderr,
                        "round_bf16 inplace mismatch n=%llu i=%llu\n",
                        (unsigned long long)n, (unsigned long long)i);
                std::exit(1);
            }
        }
    }
}

static void test_prepare_q_bf16_fuse(void) {
    constexpr uint32_t rows = 3u;
    constexpr uint32_t heads = 4u;
    constexpr uint32_t key_dim = 192u;
    constexpr uint32_t rope_dim = 64u;
    const uint64_t n = (uint64_t)rows * heads * key_dim;
    std::vector<float> q_raw((size_t)n);
    std::vector<int32_t> positions(rows);
    std::vector<float> inv(rope_dim / 2u);
    for (uint64_t i = 0; i < n; i++)
        q_raw[i] = (float)((int32_t)((i * 29u + 11u) % 257u) - 128) / 256.0f;
    for (uint32_t r = 0; r < rows; r++) positions[r] = (int32_t)(r * 17u + 3u);
    for (uint32_t i = 0; i < rope_dim / 2u; i++)
        inv[i] = 1.0f / std::pow(10000.0f, (2.0f * (float)i) / (float)rope_dim);
    gpu_tensor q_in(n * sizeof(float));
    gpu_tensor q_plain(n * sizeof(float));
    gpu_tensor q_round(n * sizeof(float));
    gpu_tensor q_fused(n * sizeof(float));
    gpu_tensor pos(rows * sizeof(int32_t));
    gpu_tensor freq((rope_dim / 2u) * sizeof(float));
    if (!ds4_gpu_tensor_write(q_in.p, 0, q_raw.data(), n * sizeof(float)) ||
        !ds4_gpu_tensor_write(pos.p, 0, positions.data(),
                              rows * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(freq.p, 0, inv.data(),
                              (rope_dim / 2u) * sizeof(float)) ||
        !ds4_gpu_motif3_prepare_q_tensor(q_plain.p, q_in.p, pos.p, freq.p,
                                         rows, heads, key_dim, rope_dim, 0) ||
        !ds4_gpu_motif3_round_bf16_tensor(q_round.p, q_plain.p, n) ||
        !ds4_gpu_motif3_prepare_q_tensor(q_fused.p, q_in.p, pos.p, freq.p,
                                         rows, heads, key_dim, rope_dim, 1)) {
        fprintf(stderr, "prepare_q BF16 fuse launch failed\n");
        std::exit(1);
    }
    const auto want = download_f32(q_round, n);
    const auto got = download_f32(q_fused, n);
    assert_bits_equal("prepare_q BF16 fuse", got.data(), want.data(), n);
}

static void test_differential_bf16_fuse(void) {
    constexpr uint32_t rows = 2u;
    constexpr uint32_t kv_heads = 16u;
    constexpr uint32_t group = 5u;
    constexpr uint32_t value_dim = 128u;
    const uint64_t attn_n = (uint64_t)rows * kv_heads * group * value_dim;
    const uint64_t out_n = (uint64_t)rows * kv_heads * (group - 1u) * value_dim;
    const uint64_t lambda_n = (uint64_t)rows * kv_heads * (group - 1u);
    std::vector<float> attention((size_t)attn_n);
    std::vector<float> lambda((size_t)lambda_n);
    std::vector<float> gate((size_t)out_n);
    for (uint64_t i = 0; i < attn_n; i++)
        attention[i] = (float)((int32_t)((i * 17u + 5u) % 251u) - 125) / 128.0f;
    for (uint64_t i = 0; i < lambda_n; i++)
        lambda[i] = (float)((int32_t)((i * 13u + 3u) % 97u) - 48) / 16.0f;
    for (uint64_t i = 0; i < out_n; i++)
        gate[i] = (float)((int32_t)((i * 19u + 7u) % 113u) - 56) / 16.0f;
    gpu_tensor attn(attn_n * sizeof(float));
    gpu_tensor lam(lambda_n * sizeof(float));
    gpu_tensor g(out_n * sizeof(float));
    gpu_tensor plain(out_n * sizeof(float));
    gpu_tensor rounded(out_n * sizeof(float));
    gpu_tensor fused(out_n * sizeof(float));
    if (!ds4_gpu_tensor_write(attn.p, 0, attention.data(),
                              attn_n * sizeof(float)) ||
        !ds4_gpu_tensor_write(lam.p, 0, lambda.data(),
                              lambda_n * sizeof(float)) ||
        !ds4_gpu_tensor_write(g.p, 0, gate.data(), out_n * sizeof(float)) ||
        !ds4_gpu_motif3_differential_tensor(plain.p, attn.p, lam.p, g.p,
                                            rows, kv_heads, group,
                                            value_dim, 0) ||
        !ds4_gpu_motif3_round_bf16_tensor(rounded.p, plain.p, out_n) ||
        !ds4_gpu_motif3_differential_tensor(fused.p, attn.p, lam.p, g.p,
                                            rows, kv_heads, group,
                                            value_dim, 1)) {
        fprintf(stderr, "differential BF16 fuse launch failed\n");
        std::exit(1);
    }
    const auto want = download_f32(rounded, out_n);
    const auto got = download_f32(fused, out_n);
    assert_bits_equal("differential BF16 fuse", got.data(), want.data(), out_n);
}

static void test_bf16_projection() {
    /* Authentic Motif mHC proj_res shape: [16, 4 * hidden_size]. */
    constexpr uint64_t in_dim = 4u * 4096u;
    constexpr uint64_t out_dim = 16u;
    constexpr uint64_t rows = 3u;
    std::vector<uint16_t> weight((size_t)(in_dim * out_dim));
    std::vector<uint16_t> model_storage(weight.size() + 1u);
    std::vector<float> input((size_t)(rows * in_dim));
    std::vector<float> input_bf16(input.size());
    std::vector<float> expected((size_t)(rows * out_dim), 0.0f);

    for (uint64_t o = 0; o < out_dim; o++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            const int32_t raw = (int32_t)((i * 17u + o * 29u) % 257u) - 128;
            weight[(size_t)(o * in_dim + i)] =
                f32_to_bf16_bits((float)raw / 2048.0f);
        }
    }
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            const int32_t raw = (int32_t)((i * 11u + r * 37u) % 251u) - 125;
            const float value = (float)raw / 1024.0f;
            input[(size_t)(r * in_dim + i)] = value;
            input_bf16[(size_t)(r * in_dim + i)] =
                bf16_bits_to_f32(f32_to_bf16_bits(value));
        }
    }
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (uint64_t i = 0; i < in_dim; i++) {
                sum += bf16_bits_to_f32(weight[(size_t)(o * in_dim + i)]) *
                       input_bf16[(size_t)(r * in_dim + i)];
            }
            expected[(size_t)(r * out_dim + o)] = sum;
        }
    }

    /* Reinstall a different-sized model at the same host address.  Short-lived
     * auxiliary mappings can reuse a freed allocator address; a stale range
     * keyed only by that address must not shadow the current full device copy. */
    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    std::fill(model_storage.begin(), model_storage.end(),
              f32_to_bf16_bits(0.5f));
    if (!ds4_gpu_set_model_map(model_storage.data(),
                               model_storage.size() * sizeof(uint16_t))) {
        fprintf(stderr, "could not install stale BF16 projection test map\n");
        std::exit(1);
    }
    std::copy(weight.begin(), weight.end(), model_storage.begin());
    if (!ds4_gpu_set_model_map(model_storage.data(),
                               weight.size() * sizeof(uint16_t))) {
        fprintf(stderr, "could not install BF16 projection test map\n");
        std::exit(1);
    }
    unsetenv("DS4_CUDA_COPY_MODEL");
    gpu_tensor x(input.size() * sizeof(float));
    gpu_tensor out(expected.size() * sizeof(float));
    if (!ds4_gpu_tensor_write(x.p, 0, input.data(), input.size() * sizeof(float)) ||
        !ds4_gpu_matmul_bf16_tensor(out.p, model_storage.data(),
                                    weight.size() * sizeof(uint16_t), 0,
                                    in_dim, out_dim, x.p, rows)) {
        fprintf(stderr, "BF16 projection dispatch failed\n");
        std::exit(1);
    }
    auto got = download_f32(out, expected.size());
    assert_close("CUDA BF16 mHC projection", got.data(), expected.data(),
                 expected.size(), 2e-3f, 8e-4f);
}

static void test_router(const char *dir) {
    fixture f = load_fixture(path(dir, "router-layer2.ds4fx").c_str());
    gpu_tensor logits(get(f, "logits").bytes.size()); upload(logits, get(f, "logits"));
    gpu_tensor bias(get(f, "expert_bias").bytes.size()); upload(bias, get(f, "expert_bias"));
    gpu_tensor selected(8u * 8u * sizeof(int32_t));
    gpu_tensor weights(8u * 8u * sizeof(float));
    gpu_tensor probs(8u * 384u * sizeof(float));
    if (!ds4_gpu_motif3_router_select_batch_tensor(selected.p, weights.p, probs.p,
                                                    logits.p, bias.p, 8, 384, 8, 2.0f)) std::exit(1);
    auto ids = download_i32(selected, 64);
    auto route = download_f32(weights, 64);
    const int32_t *want_ids = i32(f, "selected_experts");
    for (uint32_t i = 0; i < 64; i++) if (ids[i] != want_ids[i]) {
        fprintf(stderr, "router id mismatch at %u: %d != %d\n", i, ids[i], want_ids[i]);
        std::exit(1);
    }
    assert_close("CUDA router", route.data(), f32(f, "route_weights"), 64, 3e-7f, 3e-6f);
}

static void test_polynorm(const char *dir) {
    fixture f = load_fixture(path(dir, "polynorm-layer2-expert173.ds4fx").c_str());
    gpu_tensor gate(get(f, "gate").bytes.size()); upload(gate, get(f, "gate"));
    gpu_tensor up(get(f, "up").bytes.size()); upload(up, get(f, "up"));
    gpu_tensor coeff(get(f, "raw_coeff").bytes.size()); upload(coeff, get(f, "raw_coeff"));
    gpu_tensor bias(get(f, "raw_bias").bytes.size()); upload(bias, get(f, "raw_bias"));
    gpu_tensor out(4u * 1280u * sizeof(float));
    if (!ds4_gpu_motif3_polynorm_mul_tensor(out.p, gate.p, up.p, coeff.p, bias.p,
                                            4, 1280, 1e6f, .5f, .5f, 1e-6f)) std::exit(1);
    auto got = download_f32(out, 4u * 1280u);
    assert_close("CUDA PolyNorm", got.data(), f32(f, "activated_fp32"),
                 4u * 1280u, 4e-5f, 4e-5f);
}

static void test_mhc(const char *dir) {
    fixture f = load_fixture(path(dir, "mhc-layer0-attn.ds4fx").c_str());
    const char *names[] = {"projected_pre", "projected_post", "projected_res",
                           "alpha_pre", "alpha_post", "alpha_res",
                           "bias_pre", "bias_post", "bias_res", "hidden"};
    std::vector<gpu_tensor *> tensors;
    for (const char *name : names) {
        auto *t = new gpu_tensor(get(f, name).bytes.size());
        upload(*t, get(f, name)); tensors.push_back(t);
    }
    gpu_tensor h_pre(16u * sizeof(float)), h_post(16u * sizeof(float)), h_res(64u * sizeof(float));
    if (!ds4_gpu_motif3_mhc_controls_tensor(
            h_pre.p, h_post.p, h_res.p,
            tensors[0]->p, tensors[1]->p, tensors[2]->p,
            tensors[3]->p, tensors[4]->p, tensors[5]->p,
            tensors[6]->p, tensors[7]->p, tensors[8]->p,
            4, 4, 20, 1.0f)) std::exit(1);
    auto pre = download_f32(h_pre, 16), post = download_f32(h_post, 16), res = download_f32(h_res, 64);
    assert_close("CUDA mHC pre", pre.data(), f32(f, "h_pre"), 16, 3e-7f, 3e-6f);
    assert_close("CUDA mHC post", post.data(), f32(f, "h_post"), 16, 3e-7f, 3e-6f);
    assert_close("CUDA mHC Sinkhorn", res.data(), f32(f, "h_res"), 64, 4e-6f, 4e-5f);
    gpu_tensor reduced(4u * 4096u * sizeof(float));
    gpu_tensor mixed(4u * 4u * 4096u * sizeof(float));
    if (!ds4_gpu_motif3_mhc_apply_pre_tensor(reduced.p, tensors[9]->p, h_pre.p, 4, 4, 4096, 0) ||
        !ds4_gpu_motif3_mhc_apply_res_tensor(mixed.p, tensors[9]->p, h_res.p, 4, 4, 4096)) std::exit(1);
    auto reduced_h = download_f32(reduced, 4u * 4096u);
    auto mixed_h = download_f32(mixed, 4u * 4u * 4096u);
    assert_close("CUDA mHC pre apply", reduced_h.data(), f32(f, "reduced_input"), 4u * 4096u, 3e-6f, 3e-5f);
    assert_close("CUDA mHC residual apply", mixed_h.data(), f32(f, "residual_mixed"), 4u * 4u * 4096u, 3e-6f, 3e-5f);
    gpu_tensor reduced_round(4u * 4096u * sizeof(float));
    gpu_tensor reduced_fused(4u * 4096u * sizeof(float));
    if (!ds4_gpu_motif3_round_bf16_tensor(reduced_round.p, reduced.p, 4u * 4096u) ||
        !ds4_gpu_motif3_mhc_apply_pre_tensor(reduced_fused.p, tensors[9]->p, h_pre.p,
                                             4, 4, 4096, 1)) std::exit(1);
    const auto want_round = download_f32(reduced_round, 4u * 4096u);
    const auto got_fused = download_f32(reduced_fused, 4u * 4096u);
    for (uint32_t i = 0; i < 4u * 4096u; i++) {
        uint32_t gb, wb;
        std::memcpy(&gb, &got_fused[i], sizeof(gb));
        std::memcpy(&wb, &want_round[i], sizeof(wb));
        if (gb != wb) {
            fprintf(stderr, "mHC pre BF16 fuse mismatch at %u\n", i);
            std::exit(1);
        }
    }
    for (gpu_tensor *t : tensors) delete t;
}

static void test_gdla(const char *dir) {
    fixture f = load_fixture(path(dir, "gdla-expanded-layer0.ds4fx").c_str());
    gpu_tensor positions(get(f, "positions").bytes.size()); upload(positions, get(f, "positions"));
    gpu_tensor probes(get(f, "probe_positions").bytes.size()); upload(probes, get(f, "probe_positions"));
    gpu_tensor inv(get(f, "yarn_inv_freq").bytes.size()); upload(inv, get(f, "yarn_inv_freq"));
    gpu_tensor qpe(get(f, "q_pe_before").bytes.size()); upload(qpe, get(f, "q_pe_before"));
    gpu_tensor kpe(get(f, "k_pe_before").bytes.size()); upload(kpe, get(f, "k_pe_before"));
    gpu_tensor qrot(get(f, "q_pe_before").bytes.size());
    gpu_tensor krot(get(f, "k_pe_before").bytes.size());
    if (!ds4_gpu_motif3_rope_tensor(qrot.p, qpe.p, positions.p, inv.p, 8, 80, 64) ||
        !ds4_gpu_motif3_rope_tensor(krot.p, kpe.p, positions.p, inv.p, 8, 1, 64)) std::exit(1);
    auto qr = download_f32(qrot, 8u * 80u * 64u), kr = download_f32(krot, 8u * 64u);
    assert_close("CUDA GDLA q RoPE", qr.data(), f32(f, "q_pe_after_fp32"), qr.size(), 3e-5f, 3e-5f);
    assert_close("CUDA GDLA k RoPE", kr.data(), f32(f, "k_pe_after_fp32"), kr.size(), 3e-5f, 3e-5f);
    if (!ds4_gpu_motif3_rope_tensor(qrot.p, qpe.p, probes.p, inv.p, 8, 80, 64) ||
        !ds4_gpu_motif3_rope_tensor(krot.p, kpe.p, probes.p, inv.p, 8, 1, 64)) std::exit(1);
    qr = download_f32(qrot, 8u * 80u * 64u); kr = download_f32(krot, 8u * 64u);
    assert_close("CUDA GDLA 256K q RoPE", qr.data(), f32(f, "q_pe_probe_fp32"), qr.size(), 8e-4f, 6e-5f);
    assert_close("CUDA GDLA 256K k RoPE", kr.data(), f32(f, "k_pe_probe_fp32"), kr.size(), 8e-4f, 6e-5f);

    gpu_tensor q(get(f, "q_full").bytes.size()); upload(q, get(f, "q_full"));
    gpu_tensor k(get(f, "k_full").bytes.size()); upload(k, get(f, "k_full"));
    gpu_tensor v(get(f, "value").bytes.size()); upload(v, get(f, "value"));
    gpu_tensor attention(8u * 80u * 128u * sizeof(float));
    if (!ds4_gpu_motif3_expanded_attention_tensor(attention.p, q.p, k.p, v.p,
                                                  8, 80, 16, 192, 128,
                                                  f32(f, "attention_scale")[0], true)) std::exit(1);
    auto attn = download_f32(attention, 8u * 80u * 128u);
    assert_close("CUDA expanded GDLA", attn.data(), f32(f, "attention_fp32"),
                 attn.size(), 5e-4f, 5e-5f);
    /* The production prefill path computes the current causal block and each
     * cached prefix block separately, then merges the normalized outputs from
     * their log-sum-exp states.  Verify that split form against one HMMA call
     * over the same visible keys. */
    constexpr uint32_t tail_rows = 4u;
    constexpr uint32_t split = 4u;
    const uint64_t q_row = 80u * 192u;
    const uint64_t k_row = 16u * 192u;
    const uint64_t v_row = 16u * 128u;
    gpu_tensor q_tail(tail_rows * q_row * sizeof(float));
    gpu_tensor k_prefix(split * k_row * sizeof(float));
    gpu_tensor v_prefix(split * v_row * sizeof(float));
    gpu_tensor k_suffix(tail_rows * k_row * sizeof(float));
    gpu_tensor v_suffix(tail_rows * v_row * sizeof(float));
    gpu_tensor full_out(tail_rows * 80u * 128u * sizeof(float));
    gpu_tensor full_lse(tail_rows * 80u * sizeof(float));
    gpu_tensor merged_out(tail_rows * 80u * 128u * sizeof(float));
    gpu_tensor merged_lse(tail_rows * 80u * sizeof(float));
    gpu_tensor prefix_out(tail_rows * 80u * 128u * sizeof(float));
    gpu_tensor prefix_lse(tail_rows * 80u * sizeof(float));
    const float *q_host = f32(f, "q_full");
    const float *k_host = f32(f, "k_full");
    const float *v_host = f32(f, "value");
    if (!ds4_gpu_tensor_write(q_tail.p, 0, q_host + split * q_row,
                              tail_rows * q_row * sizeof(float)) ||
        !ds4_gpu_tensor_write(k_prefix.p, 0, k_host,
                              split * k_row * sizeof(float)) ||
        !ds4_gpu_tensor_write(v_prefix.p, 0, v_host,
                              split * v_row * sizeof(float)) ||
        !ds4_gpu_tensor_write(k_suffix.p, 0, k_host + split * k_row,
                              tail_rows * k_row * sizeof(float)) ||
        !ds4_gpu_tensor_write(v_suffix.p, 0, v_host + split * v_row,
                              tail_rows * v_row * sizeof(float)) ||
        !ds4_gpu_motif3_expanded_attention_range_tensor(
                full_out.p, full_lse.p, q_tail.p, k.p, v.p,
                tail_rows, split, 8u, 0u, 80u, 16u, 192u, 128u,
                f32(f, "attention_scale")[0], 0u) ||
        !ds4_gpu_motif3_expanded_attention_range_tensor(
                merged_out.p, merged_lse.p, q_tail.p,
                k_suffix.p, v_suffix.p,
                tail_rows, split, tail_rows, split,
                80u, 16u, 192u, 128u, f32(f, "attention_scale")[0], 0u) ||
        !ds4_gpu_motif3_expanded_attention_range_tensor(
                prefix_out.p, prefix_lse.p, q_tail.p,
                k_prefix.p, v_prefix.p,
                tail_rows, split, split, 0u,
                80u, 16u, 192u, 128u, f32(f, "attention_scale")[0], 0u)) {
        fprintf(stderr, "chunked expanded GDLA failed\n");
        std::exit(1);
    }
    const uint64_t merge_states = tail_rows * 80u;
    const uint64_t merge_values = merge_states * 128u;
    auto suffix_before = download_f32(merged_out, merge_values);
    auto suffix_lse_before = download_f32(merged_lse, merge_states);
    auto prefix_before = download_f32(prefix_out, merge_values);
    auto prefix_lse_before = download_f32(prefix_lse, merge_states);
    std::vector<float> merge_want(merge_values);
    std::vector<float> merge_lse_want(merge_states);
    for (uint64_t s = 0; s < merge_states; s++) {
        const float m = std::max(suffix_lse_before[s], prefix_lse_before[s]);
        const float sw = std::exp(suffix_lse_before[s] - m);
        const float pw = std::exp(prefix_lse_before[s] - m);
        const float inv = 1.0f / (sw + pw);
        merge_lse_want[s] = m + std::log(sw + pw);
        for (uint32_t d = 0; d < 128u; d++) {
            const uint64_t i = s * 128u + d;
            merge_want[i] =
                (suffix_before[i] * sw + prefix_before[i] * pw) * inv;
        }
    }
    if (!ds4_gpu_motif3_merge_attention_states_tensor(
            merged_out.p, merged_lse.p, prefix_out.p, prefix_lse.p,
            tail_rows, 80u, 128u)) {
        fprintf(stderr, "chunked expanded GDLA merge failed\n");
        std::exit(1);
    }
    auto full_chunked = download_f32(full_out, tail_rows * 80u * 128u);
    auto merged_chunked = download_f32(merged_out, tail_rows * 80u * 128u);
    auto merged_lse_host = download_f32(merged_lse, merge_states);
    assert_close("CUDA chunked GDLA merge kernel", merged_chunked.data(),
                 merge_want.data(), merge_values, 2e-6f, 2e-6f);
    assert_close("CUDA chunked GDLA merge LSE", merged_lse_host.data(),
                 merge_lse_want.data(), merge_states, 2e-6f, 2e-6f);
    assert_close("CUDA chunked GDLA versus single pass", merged_chunked.data(),
                 full_chunked.data(), full_chunked.size(), 5e-4f, 5e-5f);

    gpu_tensor lambda(get(f, "lambda").bytes.size()); upload(lambda, get(f, "lambda"));
    gpu_tensor gate(get(f, "gate_score").bytes.size()); upload(gate, get(f, "gate_score"));
    gpu_tensor diff(8u * 64u * 128u * sizeof(float));
    if (!ds4_gpu_motif3_differential_tensor(diff.p, attention.p, lambda.p, gate.p,
                                             8, 16, 5, 128, 0)) std::exit(1);
    auto d = download_f32(diff, 8u * 64u * 128u);
    assert_close("CUDA differential GDLA", d.data(), f32(f, "diff_attention_fp32"), d.size(), 4e-4f, 4e-5f);
}

static void test_latent_gdla() {
    const char *profile_rows_env = std::getenv("DS4_MOTIF3_PROFILE_ROWS");
    const char *dots_profile = std::getenv("DS4_DOTS3_PROFILE_ATTN");
    const bool profile_transposed =
        std::getenv("DS4_DOTS3_PROFILE_TRANSPOSED") != nullptr;
    const bool dots_full = dots_profile && !std::strcmp(dots_profile, "full");
    const bool dots_swa = dots_profile && !std::strcmp(dots_profile, "swa");
    const bool dots_dsa = dots_profile && !std::strcmp(dots_profile, "dsa");
    if (dots_profile && !dots_full && !dots_swa && !dots_dsa) {
        fprintf(stderr, "invalid DS4_DOTS3_PROFILE_ATTN=%s\n", dots_profile);
        std::exit(2);
    }
    const bool profile_only = (profile_rows_env && profile_rows_env[0]) ||
        dots_full || dots_swa || dots_dsa;
    uint32_t rows = dots_dsa ? 16u : dots_profile ? 1600u : 6u;
    if (profile_rows_env && profile_rows_env[0]) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(profile_rows_env, &end, 10);
        if (!end || end[0] || parsed == 0ul || parsed > 4096ul) {
            fprintf(stderr, "invalid DS4_MOTIF3_PROFILE_ROWS=%s\n", profile_rows_env);
            std::exit(2);
        }
        rows = (uint32_t)parsed;
    }
    const uint32_t q_heads = (dots_full || dots_dsa) ? 128u :
        dots_swa ? 64u : 80u;
    const uint32_t kv_heads = dots_profile ? q_heads : 16u;
    const uint32_t group = dots_profile ? 1u : 5u;
    const uint32_t latent_dim = dots_swa ? 1024u : 512u;
    const uint32_t qk_nope = dots_swa ? 192u : 128u;
    constexpr uint32_t rope_dim = 64u;
    const uint32_t key_dim = qk_nope + rope_dim;
    constexpr uint32_t value_dim = 128u;
    const uint32_t kv_raw_dim = latent_dim + rope_dim;
    const uint64_t row_bytes = (latent_dim / 32u) * 34u;
    const uint32_t weight_rows = kv_heads * (qk_nope + value_dim);
    constexpr float weight_scale = 1.0f / 64.0f;

    /* A deterministic authentic-shape Q8_0 kv_b matrix lets this test prove
     * the MLA identity used by production: q W_k C followed by attention and
     * C W_v is equivalent to (W_k^T q) C, latent accumulation, then W_v. */
    std::vector<uint8_t> model((size_t)weight_rows * row_bytes, 0u);
    for (uint32_t r = 0; r < weight_rows; r++) {
        uint8_t *row = model.data() + (uint64_t)r * row_bytes;
        for (uint32_t b = 0; b < latent_dim / 32u; b++) {
            uint8_t *block = row + (uint64_t)b * 34u;
            const uint16_t half_scale = 0x2400u; /* IEEE F16 2^-6. */
            std::memcpy(block, &half_scale, sizeof(half_scale));
            for (uint32_t j = 0; j < 32u; j++) {
                const uint32_t col = b * 32u + j;
                const int8_t q = (int8_t)((r * 17u + col * 13u + 3u) % 9u) - 4;
                std::memcpy(block + 2u + j, &q, sizeof(q));
            }
        }
    }
    auto weight = [&](uint32_t r, uint32_t c) -> float {
        const uint8_t *block = model.data() + (uint64_t)r * row_bytes +
                               (uint64_t)(c / 32u) * 34u;
        int8_t q;
        std::memcpy(&q, block + 2u + (c % 32u), sizeof(q));
        return weight_scale * (float)q;
    };

    std::vector<float> q_raw((size_t)rows * q_heads * key_dim);
    std::vector<float> kv_norm((size_t)rows * latent_dim);
    std::vector<float> kv_raw((size_t)rows * kv_raw_dim);
    std::vector<float> inv(rope_dim / 2u);
    std::vector<int32_t> positions(rows);
    std::vector<int32_t> selected;
    if (dots_dsa) {
        selected.assign((size_t)rows * rows, (int32_t)rows);
        for (uint32_t t = 0; t < rows; t++)
            for (uint32_t j = 0; j <= t; j++)
                selected[(size_t)t * rows + j] = (int32_t)j;
    }
    for (size_t i = 0; i < q_raw.size(); i++)
        q_raw[i] = (float)((int32_t)((i * 29u + 11u) % 257u) - 128) / 256.0f;
    for (size_t i = 0; i < kv_norm.size(); i++)
        kv_norm[i] = (float)((int32_t)((i * 31u + 7u) % 251u) - 125) / 512.0f;
    for (uint32_t r = 0; r < rows; r++) {
        positions[r] = (int32_t)r;
        std::memcpy(kv_raw.data() + (uint64_t)r * kv_raw_dim,
                    kv_norm.data() + (uint64_t)r * latent_dim,
                    latent_dim * sizeof(float));
        for (uint32_t d = 0; d < rope_dim; d++) {
            const uint64_t i = (uint64_t)r * rope_dim + d;
            kv_raw[(uint64_t)r * kv_raw_dim + latent_dim + d] =
                (float)((int32_t)((i * 19u + 5u) % 193u) - 96) / 256.0f;
        }
    }
    for (uint32_t i = 0; i < rope_dim / 2u; i++)
        inv[i] = 1.0f / std::pow(10000.0f, (2.0f * (float)i) / (float)rope_dim);

    static const char kv_b_name[] = "blk.0.attn_kv_b.weight";
    ds4_gpu_tensor_record kv_b_record = {};
    kv_b_record.name = kv_b_name;
    kv_b_record.name_len = sizeof(kv_b_name) - 1u;
    kv_b_record.type = 8u; /* GGML_TYPE_Q8_0 */
    kv_b_record.ndim = 2u;
    kv_b_record.dims[0] = latent_dim;
    kv_b_record.dims[1] = weight_rows;
    kv_b_record.offset = 0u;
    kv_b_record.bytes = model.size();
    if ((!dots_profile || profile_transposed) &&
        ds4_gpu_build_derived_artifacts_from_records(
            model.data(), model.size(), &kv_b_record, 1u) < 1) {
        fprintf(stderr, "could not build Motif kv_b value artifact\n");
        std::exit(1);
    }
    /* Build first, then hide the registry from the consumer so CI can keep
     * exercising the unchanged raw-weight fallback independently. */
    if (std::getenv("DS4_MOTIF3_TEST_RAW_VALUE"))
        setenv("DS4_CUDA_NO_DERIVED_WEIGHTS", "1", 1);

    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    if (!ds4_gpu_set_model_map(model.data(), model.size())) {
        fprintf(stderr, "could not install latent GDLA test map\n");
        std::exit(1);
    }
    unsetenv("DS4_CUDA_COPY_MODEL");

    gpu_tensor q_raw_gpu(q_raw.size() * sizeof(float));
    gpu_tensor q_full_gpu(q_raw.size() * sizeof(float));
    gpu_tensor kv_norm_gpu(kv_norm.size() * sizeof(float));
    gpu_tensor kv_raw_gpu(kv_raw.size() * sizeof(float));
    gpu_tensor positions_gpu(positions.size() * sizeof(int32_t));
    gpu_tensor inv_gpu(inv.size() * sizeof(float));
    gpu_tensor latent_cache((uint64_t)rows * latent_dim * sizeof(uint16_t));
    gpu_tensor k_pe_cache((uint64_t)rows * rope_dim * sizeof(uint16_t));
    gpu_tensor q_absorbed((uint64_t)rows * q_heads * latent_dim * sizeof(float));
    gpu_tensor latent_out((uint64_t)rows * q_heads * latent_dim * sizeof(float));
    gpu_tensor latent_ref(dots_dsa
        ? (uint64_t)rows * q_heads * latent_dim * sizeof(float)
        : sizeof(float));
    gpu_tensor selected_gpu(dots_dsa
        ? (uint64_t)rows * rows * sizeof(int32_t) : sizeof(int32_t));
    gpu_tensor heads((uint64_t)rows * q_heads * value_dim * sizeof(float));
    if (!ds4_gpu_tensor_write(q_raw_gpu.p, 0, q_raw.data(), q_raw.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(kv_norm_gpu.p, 0, kv_norm.data(), kv_norm.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(kv_raw_gpu.p, 0, kv_raw.data(), kv_raw.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(positions_gpu.p, 0, positions.data(), positions.size() * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(inv_gpu.p, 0, inv.data(), inv.size() * sizeof(float)) ||
        (dots_dsa && !ds4_gpu_tensor_write(
            selected_gpu.p, 0, selected.data(), selected.size() * sizeof(int32_t)))) {
        fprintf(stderr, "latent GDLA upload failed\n"); std::exit(1);
    }
    if (!ds4_gpu_motif3_prepare_q_tensor(q_full_gpu.p, q_raw_gpu.p,
                                          positions_gpu.p, inv_gpu.p,
                                          rows, q_heads, key_dim, rope_dim, 0) ||
        !ds4_gpu_motif3_round_bf16_tensor(q_full_gpu.p, q_full_gpu.p,
                                           (uint64_t)rows * q_heads * key_dim) ||
        !ds4_gpu_motif3_store_latent_kv_bf16_tensor(
                latent_cache.p, k_pe_cache.p, kv_norm_gpu.p, kv_raw_gpu.p,
                positions_gpu.p, inv_gpu.p, rows, rows, kv_raw_dim,
                latent_dim, rope_dim, false) ||
        !ds4_gpu_motif3_qk_absorb_q8_0_tensor(
                q_absorbed.p, q_full_gpu.p, model.data(), model.size(), 0,
                rows, q_heads, kv_heads, group, latent_dim, qk_nope,
                key_dim, value_dim)) {
        fprintf(stderr, "latent GDLA preparation failed\n"); std::exit(1);
    }

    if (dots_dsa) {
        const auto q_full = download_f32(
            q_full_gpu, (uint64_t)rows * q_heads * key_dim);
        const auto q_abs = download_f32(
            q_absorbed, (uint64_t)rows * q_heads * latent_dim);
        constexpr uint32_t sample[][3] = {
            {0u, 0u, 0u}, {7u, 63u, 257u}, {15u, 127u, 511u},
        };
        float got[3], want[3] = {};
        for (uint32_t i = 0; i < 3u; i++) {
            const uint32_t t = sample[i][0];
            const uint32_t h = sample[i][1];
            const uint32_t j = sample[i][2];
            const float *qh = q_full.data() +
                ((uint64_t)t * q_heads + h) * key_dim;
            for (uint32_t d = 0; d < qk_nope; d++)
                want[i] += qh[d] * weight(
                    h * (qk_nope + value_dim) + d, j);
            got[i] = q_abs[((uint64_t)t * q_heads + h) * latent_dim + j];
        }
        assert_close("CUDA Dots absorbed Q/K samples", got, want, 3u,
                     2e-5f, 2e-5f);
    }

    /* Nsight Compute kernel replay cannot coexist with the full 86 GiB VMM
     * owner on GB10: replay checkpoints exhaust unified memory.  This mode
     * keeps the production kernels and exact model-family dimensions while
     * expanding only the token axis used by a real prefill chunk. */
    if (profile_only) {
        const float scale = 1.0f / std::sqrt((float)key_dim);
        bool attention_ok = true;
        if (dots_dsa) {
            attention_ok = ds4_gpu_dots3_latent_attention_tensor(
                latent_ref.p, q_full_gpu.p, q_absorbed.p,
                latent_cache.p, k_pe_cache.p, nullptr, 0u,
                rows, 0u, rows, 0u, q_heads, latent_dim,
                qk_nope, rope_dim, scale);
        }
        attention_ok = attention_ok && (dots_profile
            ? ds4_gpu_dots3_latent_attention_tensor(
                  latent_out.p, q_full_gpu.p, q_absorbed.p,
                  latent_cache.p, k_pe_cache.p,
                  dots_dsa ? selected_gpu.p : nullptr,
                  dots_dsa ? rows : 0u, rows, 0u, rows,
                  dots_swa ? 513u : 0u, q_heads, latent_dim,
                  qk_nope, rope_dim, scale)
            : ds4_gpu_motif3_latent_attention_bf16_tensor(
                  latent_out.p, q_full_gpu.p, q_absorbed.p,
                  latent_cache.p, k_pe_cache.p, rows, 0, rows, 0,
                  q_heads, latent_dim, qk_nope, rope_dim, scale));
        if (!attention_ok ||
            !ds4_gpu_motif3_value_project_q8_0_tensor(
                    heads.p, latent_out.p, model.data(), model.size(), 0,
                    rows, q_heads, kv_heads, group, latent_dim,
                    qk_nope, value_dim, 0)) {
            fprintf(stderr, "latent GDLA profile launch failed\n");
            std::exit(1);
        }
        float sync = 0.0f;
        if (!ds4_gpu_tensor_read(heads.p, 0, &sync, sizeof(sync)) || !std::isfinite(sync)) {
            fprintf(stderr, "latent GDLA profile synchronization failed\n");
            std::exit(1);
        }
        if (profile_transposed && rows <= 16u) {
            const auto latent_host = download_f32(
                latent_out, (uint64_t)rows * q_heads * latent_dim);
            const auto heads_host = download_f32(
                heads, (uint64_t)rows * q_heads * value_dim);
            const uint32_t sample_t[] = {0u, rows - 1u, rows - 1u};
            const uint32_t sample_h[] = {0u, q_heads / 2u, q_heads - 1u};
            const uint32_t sample_d[] = {0u, 63u, 127u};
            float got[3] = {};
            float want[3] = {};
            for (uint32_t i = 0; i < 3u; i++) {
                const uint32_t t = sample_t[i];
                const uint32_t h = sample_h[i];
                const uint32_t d = sample_d[i];
                const uint32_t kh = h / group;
                const float *src = latent_host.data() +
                    ((uint64_t)t * q_heads + h) * latent_dim;
                for (uint32_t j = 0; j < latent_dim; j++) {
                    want[i] += src[j] * weight(
                        kh * (qk_nope + value_dim) + qk_nope + d, j);
                }
                got[i] = heads_host[
                    ((uint64_t)t * q_heads + h) * value_dim + d];
            }
            assert_close("CUDA Dots transposed value samples",
                         got, want, 3u, 2e-4f, 2e-4f);
        }
        if (dots_dsa) {
            auto ref = download_f32(
                latent_ref, (uint64_t)rows * q_heads * latent_dim);
            auto got = download_f32(
                latent_out, (uint64_t)rows * q_heads * latent_dim);
            if (std::memcmp(ref.data(), got.data(),
                            ref.size() * sizeof(float))) {
                fprintf(stderr, "dots3 DSA/full latent parity failed\n");
                std::exit(1);
            }
        }
        printf("%s NCU attention profile: rows=%u, q_heads=%u, "
               "latent_dim=%u, finite output\n",
               dots_profile ? "dots3-note" : "Motif-3",
               rows, q_heads, latent_dim);
        return;
    }

    auto q_full = download_f32(q_full_gpu, (uint64_t)rows * q_heads * key_dim);
    auto latent_bits = download_u16(latent_cache, (uint64_t)rows * latent_dim);
    auto k_pe_bits = download_u16(k_pe_cache, (uint64_t)rows * rope_dim);
    auto q_abs = download_f32(q_absorbed, (uint64_t)rows * q_heads * latent_dim);
    std::vector<float> latent((size_t)rows * latent_dim);
    std::vector<float> k_pe((size_t)rows * rope_dim);
    for (size_t i = 0; i < latent.size(); i++) {
        latent[i] = bf16_bits_to_f32(latent_bits[i]);
        const float want = bf16_bits_to_f32(f32_to_bf16_bits(kv_norm[i]));
        if (latent[i] != want) {
            fprintf(stderr, "latent BF16 cache mismatch at %zu\n", i);
            std::exit(1);
        }
    }
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t d = 0; d < rope_dim; d++) {
            const uint32_t half = rope_dim / 2u;
            const uint32_t freq = d % half;
            const float angle = (float)positions[r] * inv[freq];
            const float first = kv_raw[(uint64_t)r * kv_raw_dim + latent_dim + freq];
            const float second = kv_raw[(uint64_t)r * kv_raw_dim + latent_dim + half + freq];
            const float rotated = d < half
                ? first * (float)std::cos((double)angle) - second * (float)std::sin((double)angle)
                : second * (float)std::cos((double)angle) + first * (float)std::sin((double)angle);
            const size_t i = (size_t)r * rope_dim + d;
            k_pe[i] = bf16_bits_to_f32(k_pe_bits[i]);
            if (k_pe_bits[i] != f32_to_bf16_bits(rotated)) {
                fprintf(stderr, "k_pe BF16 cache mismatch at %zu\n", i);
                std::exit(1);
            }
        }
    }

    std::vector<float> q_abs_want(q_abs.size(), 0.0f);
    for (uint32_t t = 0; t < rows; t++) for (uint32_t h = 0; h < q_heads; h++) {
        const uint32_t kh = h / group;
        const float *q = q_full.data() + ((uint64_t)t * q_heads + h) * key_dim;
        float *dst = q_abs_want.data() + ((uint64_t)t * q_heads + h) * latent_dim;
        for (uint32_t j = 0; j < latent_dim; j++)
            for (uint32_t d = 0; d < qk_nope; d++)
                dst[j] += q[d] * weight(kh * (qk_nope + value_dim) + d, j);
    }
    assert_close("CUDA absorbed Motif Q/K", q_abs.data(), q_abs_want.data(),
                 q_abs.size(), 2e-5f, 2e-5f);

    const float scale = 1.0f / std::sqrt((float)key_dim);
    if (!ds4_gpu_motif3_latent_attention_bf16_tensor(
                latent_out.p, q_full_gpu.p, q_absorbed.p,
                latent_cache.p, k_pe_cache.p, rows, 0, rows, 0,
                q_heads, latent_dim, qk_nope, rope_dim, scale) ||
        !ds4_gpu_motif3_value_project_q8_0_tensor(
                heads.p, latent_out.p, model.data(), model.size(), 0,
                rows, q_heads, kv_heads, group, latent_dim,
                qk_nope, value_dim, 0)) {
        fprintf(stderr, "latent GDLA attention failed\n"); std::exit(1);
    }

    /* Independent expanded K/V reference, using the exact dequantized Q8_0
     * weights and BF16 persistent state consumed by the latent kernels. */
    std::vector<float> expanded_k((size_t)rows * kv_heads * qk_nope, 0.0f);
    std::vector<float> expanded_v((size_t)rows * kv_heads * value_dim, 0.0f);
    for (uint32_t t = 0; t < rows; t++) for (uint32_t kh = 0; kh < kv_heads; kh++) {
        const float *c = latent.data() + (uint64_t)t * latent_dim;
        for (uint32_t d = 0; d < qk_nope; d++)
            for (uint32_t j = 0; j < latent_dim; j++)
                expanded_k[((uint64_t)t * kv_heads + kh) * qk_nope + d] +=
                    weight(kh * (qk_nope + value_dim) + d, j) * c[j];
        for (uint32_t d = 0; d < value_dim; d++)
            for (uint32_t j = 0; j < latent_dim; j++)
                expanded_v[((uint64_t)t * kv_heads + kh) * value_dim + d] +=
                    weight(kh * (qk_nope + value_dim) + qk_nope + d, j) * c[j];
    }
    std::vector<float> heads_want((size_t)rows * q_heads * value_dim, 0.0f);
    std::vector<float> scores(rows), probs(rows);
    for (uint32_t t = 0; t < rows; t++) for (uint32_t h = 0; h < q_heads; h++) {
        const uint32_t kh = h / group;
        const float *q = q_full.data() + ((uint64_t)t * q_heads + h) * key_dim;
        float max_score = -INFINITY;
        for (uint32_t k = 0; k <= t; k++) {
            float dot = 0.0f;
            const float *ek = expanded_k.data() + ((uint64_t)k * kv_heads + kh) * qk_nope;
            for (uint32_t d = 0; d < qk_nope; d++) dot += q[d] * ek[d];
            for (uint32_t d = 0; d < rope_dim; d++)
                dot += q[qk_nope + d] * k_pe[(uint64_t)k * rope_dim + d];
            scores[k] = dot * scale;
            if (scores[k] > max_score) max_score = scores[k];
        }
        float denom = 0.0f;
        for (uint32_t k = 0; k <= t; k++) {
            probs[k] = std::exp(scores[k] - max_score);
            denom += probs[k];
        }
        float *dst = heads_want.data() + ((uint64_t)t * q_heads + h) * value_dim;
        for (uint32_t k = 0; k <= t; k++) {
            const float p = probs[k] / denom;
            const float *ev = expanded_v.data() + ((uint64_t)k * kv_heads + kh) * value_dim;
            for (uint32_t d = 0; d < value_dim; d++) dst[d] += p * ev[d];
        }
    }
    auto heads_got = download_f32(heads, heads_want.size());
    assert_close("CUDA latent GDLA vs expanded identity",
                 heads_got.data(), heads_want.data(), heads_want.size(),
                 2e-4f, 2e-4f);
    gpu_tensor heads_round(heads_want.size() * sizeof(float));
    gpu_tensor heads_fused(heads_want.size() * sizeof(float));
    if (!ds4_gpu_motif3_round_bf16_tensor(heads_round.p, heads.p,
                                           heads_want.size()) ||
        !ds4_gpu_motif3_value_project_q8_0_tensor(
                heads_fused.p, latent_out.p, model.data(), model.size(), 0,
                rows, q_heads, kv_heads, group, latent_dim,
                qk_nope, value_dim, 1)) {
        fprintf(stderr, "latent V BF16 fuse launch failed\n");
        std::exit(1);
    }
    const auto want_round = download_f32(heads_round, heads_want.size());
    const auto got_fused = download_f32(heads_fused, heads_want.size());
    assert_bits_equal("value_project BF16 fuse", got_fused.data(),
                      want_round.data(), heads_want.size());
}

static void test_latent_decode_split() {
    constexpr uint32_t cache_cap = 1024u;
    constexpr uint32_t q_heads = 80u;
    constexpr uint32_t latent_dim = 512u;
    constexpr uint32_t qk_nope = 128u;
    constexpr uint32_t rope_dim = 64u;
    constexpr uint32_t key_dim = qk_nope + rope_dim;
    const float scale = 1.0f / std::sqrt((float)key_dim);

    std::vector<uint16_t> latent_bits((size_t)cache_cap * latent_dim);
    std::vector<uint16_t> k_pe_bits((size_t)cache_cap * rope_dim);
    std::vector<float> latent(latent_bits.size());
    std::vector<float> k_pe(k_pe_bits.size());
    std::vector<float> q((size_t)q_heads * key_dim);
    std::vector<float> q_absorbed((size_t)q_heads * latent_dim);
    for (size_t i = 0; i < latent.size(); i++) {
        const float value =
            (float)((int32_t)((i * 17u + 13u) % 257u) - 128) / 1024.0f;
        latent_bits[i] = f32_to_bf16_bits(value);
        latent[i] = bf16_bits_to_f32(latent_bits[i]);
    }
    for (size_t i = 0; i < k_pe.size(); i++) {
        const float value =
            (float)((int32_t)((i * 19u + 7u) % 193u) - 96) / 1024.0f;
        k_pe_bits[i] = f32_to_bf16_bits(value);
        k_pe[i] = bf16_bits_to_f32(k_pe_bits[i]);
    }
    for (size_t i = 0; i < q.size(); i++)
        q[i] = (float)((int32_t)((i * 23u + 5u) % 251u) - 125) / 512.0f;
    for (size_t i = 0; i < q_absorbed.size(); i++)
        q_absorbed[i] =
            (float)((int32_t)((i * 29u + 11u) % 263u) - 131) / 1024.0f;

    gpu_tensor latent_gpu(latent_bits.size() * sizeof(uint16_t));
    gpu_tensor k_pe_gpu(k_pe_bits.size() * sizeof(uint16_t));
    gpu_tensor q_gpu(q.size() * sizeof(float));
    gpu_tensor q_absorbed_gpu(q_absorbed.size() * sizeof(float));
    gpu_tensor out_gpu((uint64_t)q_heads * latent_dim * sizeof(float));
    if (!ds4_gpu_tensor_write(latent_gpu.p, 0, latent_bits.data(),
                              latent_bits.size() * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_write(k_pe_gpu.p, 0, k_pe_bits.data(),
                              k_pe_bits.size() * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_write(q_gpu.p, 0, q.data(), q.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(q_absorbed_gpu.p, 0, q_absorbed.data(),
                              q_absorbed.size() * sizeof(float)) ||
        !ds4_gpu_motif3_latent_attention_bf16_tensor(
                out_gpu.p, q_gpu.p, q_absorbed_gpu.p,
                latent_gpu.p, k_pe_gpu.p,
                1u, cache_cap - 1u, cache_cap, 0u,
                q_heads, latent_dim, qk_nope, rope_dim, scale)) {
        fprintf(stderr, "latent decode split dispatch failed\n");
        std::exit(1);
    }

    std::vector<float> want((size_t)q_heads * latent_dim, 0.0f);
    for (uint32_t h = 0; h < q_heads; h++) {
        const float *qh = q.data() + (uint64_t)h * key_dim;
        const float *qa = q_absorbed.data() + (uint64_t)h * latent_dim;
        float *dst = want.data() + (uint64_t)h * latent_dim;
        float M = -INFINITY;
        float S = 0.0f;
        for (uint32_t k = 0; k < cache_cap; k++) {
            const float *c = latent.data() + (uint64_t)k * latent_dim;
            const float *kp = k_pe.data() + (uint64_t)k * rope_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < latent_dim; d++) dot += qa[d] * c[d];
            for (uint32_t d = 0; d < rope_dim; d++)
                dot += qh[qk_nope + d] * kp[d];
            const float score = dot * scale;
            const float new_m = std::max(M, score);
            const float old_scale = std::exp(M - new_m);
            const float row_scale = std::exp(score - new_m);
            for (uint32_t d = 0; d < latent_dim; d++)
                dst[d] = dst[d] * old_scale + c[d] * row_scale;
            S = S * old_scale + row_scale;
            M = new_m;
        }
        for (uint32_t d = 0; d < latent_dim; d++) dst[d] /= S;
    }
    auto got = download_f32(out_gpu, want.size());
    assert_close("CUDA latent GDLA split decode", got.data(), want.data(),
                 want.size(), 3e-5f, 3e-4f);
}

/* Safe NCU harness for the production pair kernels.  Full-model
 * kernel replay checkpoints the 86 GiB VMM import and can exhaust GB10 UMA;
 * these modes cover exact Motif and dots3-note prefill geometries. */
static void test_q8_pair_profile(const char *mode) {
    uint64_t in_dim = 4096u;
    uint64_t n_tok = 512u;
    uint64_t out0_dim = 0;
    uint64_t out1_dim = 0;
    const char *name0 = nullptr;
    const char *name1 = nullptr;
    if (!std::strcmp(mode, "qkv")) {
        out0_dim = 1024u;
        out1_dim = 576u;
        name0 = "blk.0.attn_q_a.weight";
        name1 = "blk.0.attn_kv_a.weight";
    } else if (!std::strcmp(mode, "shared")) {
        out0_dim = 1280u;
        out1_dim = 1280u;
        name0 = "blk.2.ffn_gate_shexp.weight";
        name1 = "blk.2.ffn_up_shexp.weight";
    } else if (!std::strcmp(mode, "dense")) {
        /* The two leading dense Motif blocks dominate pair-kernel GPU time:
         * nsys reports grid=(1536,256), i.e. M=12288 and n_tok=256. */
        n_tok = 256u;
        out0_dim = 12288u;
        out1_dim = 12288u;
        name0 = "blk.0.ffn_gate.weight";
        name1 = "blk.0.ffn_up.weight";
    } else if (!std::strcmp(mode, "dots-shared")) {
        in_dim = 5120u;
        n_tok = 1600u;
        out0_dim = 1536u;
        out1_dim = 1536u;
        name0 = "blk.2.ffn_gate_shexp.weight";
        name1 = "blk.2.ffn_up_shexp.weight";
    } else if (!std::strcmp(mode, "dots-dense")) {
        in_dim = 5120u;
        n_tok = 1600u;
        out0_dim = 13824u;
        out1_dim = 13824u;
        name0 = "blk.0.ffn_gate.weight";
        name1 = "blk.0.ffn_up.weight";
    } else if (!std::strcmp(mode, "qwen-ple")) {
        in_dim = 2560u;
        n_tok = 256u;
        out0_dim = 10240u;
        out1_dim = 2560u;
        name0 = "blk.1.ple.key.weight";
        name1 = "blk.1.ple.value.weight";
    } else if (!std::strcmp(mode, "qwen-shared")) {
        in_dim = 2560u;
        n_tok = 8192u;
        out0_dim = 640u;
        out1_dim = 640u;
        name0 = "blk.1.ffn_gate_shexp.weight";
        name1 = "blk.1.ffn_up_shexp.weight";
    } else {
        fprintf(stderr, "invalid DS4_MOTIF3_PROFILE_PAIR=%s\n", mode);
        std::exit(2);
    }

    const uint64_t blocks = in_dim / 32u;
    const uint64_t bytes0 = out0_dim * blocks * 34u;
    const uint64_t bytes1 = out1_dim * blocks * 34u;
    std::vector<uint8_t> model((size_t)(bytes0 + bytes1));
    auto fill_weight = [&](uint64_t offset, uint64_t out_dim) {
        constexpr uint16_t scale_bits = 0x2400u; /* IEEE fp16 1/64 */
        for (uint64_t row = 0; row < out_dim; row++) {
            for (uint64_t b = 0; b < blocks; b++) {
                uint8_t *blk = model.data() + offset + (row * blocks + b) * 34u;
                std::memcpy(blk, &scale_bits, sizeof(scale_bits));
                for (uint32_t k = 0; k < 32u; k++) {
                    blk[2u + k] = (uint8_t)(int8_t)
                        ((int)((row * 13u + b * 7u + k * 3u) % 15u) - 7);
                }
            }
        }
    };
    fill_weight(0u, out0_dim);
    fill_weight(bytes0, out1_dim);

    ds4_gpu_tensor_record records[2] = {};
    records[0].name = name0;
    records[0].name_len = std::strlen(name0);
    records[0].type = 8u; /* GGML_TYPE_Q8_0 */
    records[0].ndim = 2u;
    records[0].dims[0] = in_dim;
    records[0].dims[1] = out0_dim;
    records[0].offset = 0u;
    records[0].bytes = bytes0;
    records[1].name = name1;
    records[1].name_len = std::strlen(name1);
    records[1].type = 8u;
    records[1].ndim = 2u;
    records[1].dims[0] = in_dim;
    records[1].dims[1] = out1_dim;
    records[1].offset = bytes0;
    records[1].bytes = bytes1;
    const bool qwen_ple_mode = !std::strcmp(mode, "qwen-ple");
    if (!qwen_ple_mode && ds4_gpu_build_derived_artifacts_from_records(
            model.data(), model.size(), records, 2u) != 2) {
        fprintf(stderr, "could not build Motif Q8 pair artifacts\n");
        std::exit(1);
    }
    if (!std::strncmp(mode, "dots-", 5u) || qwen_ple_mode ||
        !std::strcmp(mode, "qwen-shared"))
        setenv("DS4_CUDA_NO_DERIVED_WEIGHTS", "1", 1);
    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    if (!ds4_gpu_set_model_map(model.data(), model.size())) {
        fprintf(stderr, "could not install Motif Q8 pair profile map\n");
        std::exit(1);
    }

    std::vector<float> input((size_t)(n_tok * in_dim));
    for (uint64_t i = 0; i < input.size(); i++)
        input[(size_t)i] =
            (float)((int)((i * 17u + 11u) % 257u) - 128) / 256.0f;
    gpu_tensor x(input.size() * sizeof(float));
    gpu_tensor out0(n_tok * out0_dim * sizeof(float));
    gpu_tensor out1(n_tok * out1_dim * sizeof(float));
    if (!ds4_gpu_tensor_write(
            x.p, 0, input.data(), input.size() * sizeof(float))) {
        fprintf(stderr, "Motif Q8 pair profile input upload failed\n");
        std::exit(1);
    }

    /* The old kernel is a local oracle: the cooperative kernel must retain
     * every lane's block order and therefore produce byte-identical floats. */
    setenv("DS4_CUDA_NO_Q8_PAIR_MMQ_SPLIT", "1", 1);
    setenv("DS4_CUDA_NO_Q8_PAIR_COALESCED", "1", 1);
    const bool asym_profile = qwen_ple_mode;
    if (asym_profile)
        setenv("DS4_CUDA_NO_Q8_PAIR_ASYM_SPLIT", "1", 1);
    if (!ds4_gpu_matmul_q8_0_pair_tensor(
            out0.p, out1.p, model.data(), model.size(), 0u, bytes0,
            in_dim, out0_dim, out1_dim, x.p, n_tok)) {
        fprintf(stderr, "Motif Q8 pair oracle launch failed\n");
        std::exit(1);
    }
    auto oracle0 = download_f32(out0, (size_t)(n_tok * out0_dim));
    auto oracle1 = download_f32(out1, (size_t)(n_tok * out1_dim));
    unsetenv("DS4_CUDA_NO_Q8_PAIR_COALESCED");
    if (asym_profile)
        unsetenv("DS4_CUDA_NO_Q8_PAIR_ASYM_SPLIT");
    const bool split_profile = !std::strcmp(mode, "dense") ||
        !std::strcmp(mode, "dots-shared") ||
        !std::strcmp(mode, "dots-dense");
    if (split_profile)
        setenv("DS4_CUDA_FORCE_Q8_PAIR_COALESCED", "1", 1);
    if (
        !ds4_gpu_matmul_q8_0_pair_tensor(
            out0.p, out1.p, model.data(), model.size(), 0u, bytes0,
            in_dim, out0_dim, out1_dim, x.p, n_tok)) {
        fprintf(stderr, "Motif Q8 pair profile launch failed\n");
        std::exit(1);
    }
    auto got0 = download_f32(out0, oracle0.size());
    auto got1 = download_f32(out1, oracle1.size());
    if (std::memcmp(got0.data(), oracle0.data(), got0.size() * sizeof(float)) ||
        std::memcmp(got1.data(), oracle1.data(), got1.size() * sizeof(float))) {
        fprintf(stderr, "Q8 pair profile output is not bit-identical\n");
        std::exit(1);
    }
    unsetenv("DS4_CUDA_FORCE_Q8_PAIR_COALESCED");
    unsetenv("DS4_CUDA_NO_Q8_PAIR_MMQ_SPLIT");
    const float sample0 = got0[0];
    const float sample1 = got1[0];
    if (!std::isfinite(sample0) || !std::isfinite(sample1)) std::exit(1);
    if (split_profile) {
        if (!ds4_gpu_matmul_q8_0_pair_tensor(
                out0.p, out1.p, model.data(), model.size(), 0u, bytes0,
                in_dim, out0_dim, out1_dim, x.p, n_tok)) {
            fprintf(stderr, "Motif Q8 dense MMQ split dispatch failed\n");
            std::exit(1);
        }
        auto single0 = download_f32(out0, oracle0.size());
        auto single1 = download_f32(out1, oracle1.size());
        double dot = 0.0;
        double ref2 = 0.0;
        double got2 = 0.0;
        double err2 = 0.0;
        auto measure = [&](const std::vector<float> &ref,
                           const std::vector<float> &got) {
            for (size_t i = 0; i < ref.size(); i++) {
                if (!std::isfinite(got[i])) {
                    fprintf(stderr, "Motif Q8 single output is not finite\n");
                    std::exit(1);
                }
                dot += (double)ref[i] * got[i];
                ref2 += (double)ref[i] * ref[i];
                got2 += (double)got[i] * got[i];
                const double e = (double)got[i] - ref[i];
                err2 += e * e;
            }
        };
        measure(oracle0, single0);
        measure(oracle1, single1);
        const double cosine = dot / std::sqrt(ref2 * got2);
        const double nrmse = std::sqrt(err2 / ref2);
        if (cosine < 0.999999 || nrmse > 1.0e-5) {
            fprintf(stderr,
                    "Motif Q8 dense MMQ split parity failed: cosine=%.9f nrmse=%.6g\n",
                    cosine, nrmse);
            std::exit(1);
        }
        printf("Q8 dense MMQ split: cosine=%.9f nrmse=%.6g\n",
               cosine, nrmse);
    }
    printf("NCU Q8 pair profile: mode=%s, n_tok=%llu, K=%llu, "
           "M=%llu/%llu, bit-identical output %.6g/%.6g\n",
           mode, (unsigned long long)n_tok, (unsigned long long)in_dim,
           (unsigned long long)out0_dim, (unsigned long long)out1_dim,
           sample0, sample1);
}

/* Safe production-shape NCU harness for Motif-3's current top kernel.  It
 * preserves the real M/K, 384 experts, 256 tokens and top-8 assignment
 * count, but uses zero-filled synthetic IQ2/Q8 payloads so profiler replay
 * never checkpoints the 86 GiB shared VMM model. */
static void test_iq2_d2r_pair_profile() {
    constexpr int M = 1280;
    constexpr int K = 4096;
    constexpr int n_tokens = 256;
    constexpr int n_experts = 384;
    constexpr int n_expert_used = 8;
    constexpr int n_assign = n_tokens * n_expert_used;
    const int64_t soa_blocks =
        (int64_t)n_experts * M * (K / 256);
    const size_t dq_bytes =
        ((size_t)soa_blocks * sizeof(uint16_t) + 63u) & ~(size_t)63u;
    const size_t artifact_bytes = dq_bytes + (size_t)soa_blocks * 64u;
    const size_t q8_bytes =
        (size_t)n_assign * (K / 128) * 144u;
    const size_t out_bytes = (size_t)n_assign * M * sizeof(float);
    const size_t work_bytes =
        ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(
            n_assign, n_experts);

    void *gate = nullptr;
    void *up = nullptr;
    void *q8 = nullptr;
    int32_t *ids = nullptr;
    int32_t *bounds = nullptr;
    float *out_gate = nullptr;
    float *out_up = nullptr;
    void *work = nullptr;
    auto check = [](cudaError_t err, const char *what) {
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
            std::exit(1);
        }
    };
    check(cudaMalloc(&gate, artifact_bytes), "IQ2 gate allocation");
    check(cudaMalloc(&up, artifact_bytes), "IQ2 up allocation");
    check(cudaMalloc(&q8, q8_bytes), "IQ2 activation allocation");
    check(cudaMalloc((void **)&ids, n_assign * sizeof(int32_t)),
          "IQ2 ids allocation");
    check(cudaMalloc((void **)&bounds,
                     (n_experts + 1u) * sizeof(int32_t)),
          "IQ2 bounds allocation");
    check(cudaMalloc((void **)&out_gate, out_bytes),
          "IQ2 gate output allocation");
    check(cudaMalloc((void **)&out_up, out_bytes),
          "IQ2 up output allocation");
    check(cudaMalloc(&work, work_bytes), "IQ2 work allocation");
    check(cudaMemset(gate, 0, artifact_bytes), "IQ2 gate clear");
    check(cudaMemset(up, 0, artifact_bytes), "IQ2 up clear");
    check(cudaMemset(q8, 0, q8_bytes), "IQ2 activation clear");

    std::vector<int32_t> host_ids(n_assign);
    std::vector<int32_t> host_bounds(n_experts + 1u);
    int col = 0;
    for (int expert = 0; expert < n_experts; expert++) {
        host_bounds[(size_t)expert] = col;
        const int count = n_assign / n_experts +
                          (expert < n_assign % n_experts ? 1 : 0);
        for (int i = 0; i < count; i++) host_ids[(size_t)col] = col++;
    }
    host_bounds[n_experts] = col;
    check(cudaMemcpy(ids, host_ids.data(), n_assign * sizeof(int32_t),
                     cudaMemcpyHostToDevice), "IQ2 ids upload");
    check(cudaMemcpy(bounds, host_bounds.data(),
                     (n_experts + 1u) * sizeof(int32_t),
                     cudaMemcpyHostToDevice), "IQ2 bounds upload");

    if (ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
            gate, up, soa_blocks, q8, ids, bounds,
            out_gate, out_up, M, K, n_assign, n_experts,
            work, work_bytes, 0) != 0) {
        fprintf(stderr, "IQ2 D2R pair profile launch failed\n");
        std::exit(1);
    }
    float sample[2] = {};
    check(cudaMemcpy(&sample[0], out_gate, sizeof(float),
                     cudaMemcpyDeviceToHost), "IQ2 gate read");
    check(cudaMemcpy(&sample[1], out_up, sizeof(float),
                     cudaMemcpyDeviceToHost), "IQ2 up read");
    if (!std::isfinite(sample[0]) || !std::isfinite(sample[1])) std::exit(1);
    printf("Motif-3 NCU IQ2 D2R pair: M=%d K=%d tokens=%d assignments=%d "
           "experts=%d, finite output %.6g/%.6g\n",
           M, K, n_tokens, n_assign, n_experts, sample[0], sample[1]);

    cudaFree(work);
    cudaFree(out_up);
    cudaFree(out_gate);
    cudaFree(bounds);
    cudaFree(ids);
    cudaFree(q8);
    cudaFree(up);
    cudaFree(gate);
}

static void test_fattn_profile() {
    constexpr int n_query = 4096;
    constexpr int n_kv = 4096;
    constexpr int n_head = 80;
    constexpr int n_head_kv = 16;
    constexpr int qk_dim = 192;
    constexpr int v_dim = 128;
    const size_t q_bytes =
        (size_t)n_query * n_head * qk_dim * sizeof(float);
    const size_t k_bytes =
        (size_t)n_kv * n_head_kv * qk_dim * sizeof(float);
    const size_t v_bytes =
        (size_t)n_kv * n_head_kv * v_dim * sizeof(float);
    const size_t out_bytes =
        (size_t)n_query * n_head * v_dim * sizeof(float);
    const size_t lse_bytes =
        (size_t)n_query * n_head * sizeof(float);
    float *q = nullptr, *k = nullptr, *v = nullptr;
    float *out = nullptr, *lse = nullptr;
    auto check = [](cudaError_t err, const char *what) {
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
            std::exit(1);
        }
    };
    check(cudaMalloc((void **)&q, q_bytes), "FATTN q allocation");
    check(cudaMalloc((void **)&k, k_bytes), "FATTN k allocation");
    check(cudaMalloc((void **)&v, v_bytes), "FATTN v allocation");
    check(cudaMalloc((void **)&out, out_bytes), "FATTN output allocation");
    check(cudaMalloc((void **)&lse, lse_bytes), "FATTN LSE allocation");
    check(cudaMemset(q, 0, q_bytes), "FATTN q clear");
    check(cudaMemset(k, 0, k_bytes), "FATTN k clear");
    check(cudaMemset(v, 0, v_bytes), "FATTN v clear");
    auto launch = [&]() {
        if (ds4_mmq_motif3_prefill_attn_hmma(
                out, lse, q, k, v,
                n_query, n_kv, n_kv, 0,
                n_head, n_head_kv, qk_dim, v_dim,
                1.0f / std::sqrt((float)qk_dim), 0, 0) != 0) {
            fprintf(stderr, "Motif FATTN profile launch failed\n");
            std::exit(1);
        }
    };
    launch();
    check(cudaDeviceSynchronize(), "FATTN warmup sync");
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0), "FATTN event0");
    check(cudaEventCreate(&e1), "FATTN event1");
    constexpr int kReps = 8;
    check(cudaEventRecord(e0), "FATTN t0");
    for (int i = 0; i < kReps; ++i) launch();
    check(cudaEventRecord(e1), "FATTN t1");
    check(cudaEventSynchronize(e1), "FATTN timed sync");
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, e0, e1), "FATTN elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    float sample[2] = {};
    check(cudaMemcpy(&sample[0], out, sizeof(float), cudaMemcpyDeviceToHost),
          "FATTN output read");
    check(cudaMemcpy(&sample[1], lse, sizeof(float), cudaMemcpyDeviceToHost),
          "FATTN LSE read");
    if (!std::isfinite(sample[0]) || !std::isfinite(sample[1])) std::exit(1);
    printf("Motif-3 NCU FATTN: query=%d kv=%d heads=%d/%d, "
           "finite output %.6g LSE %.6g, %.3f ms/launch\n",
           n_query, n_kv, n_head, n_head_kv, sample[0], sample[1],
           elapsed_ms / (float)kReps);
    cudaFree(lse);
    cudaFree(out);
    cudaFree(v);
    cudaFree(k);
    cudaFree(q);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE_DIR\n", argv[0]);
        return 2;
    }
    if (!ds4_gpu_init()) { fprintf(stderr, "CUDA init failed\n"); return 1; }
    if (std::getenv("DS4_DOTS3_PROFILE_ATTN")) {
        test_latent_gdla();
        ds4_gpu_cleanup();
        return 0;
    }
    if (std::getenv("DS4_MOTIF3_PROFILE_FATTN")) {
        test_fattn_profile();
        ds4_gpu_cleanup();
        return 0;
    }
    if (std::getenv("DS4_MOTIF3_PROFILE_IQ2_D2R")) {
        test_iq2_d2r_pair_profile();
        ds4_gpu_cleanup();
        return 0;
    }
    const char *pair_profile = std::getenv("DS4_MOTIF3_PROFILE_PAIR");
    if (pair_profile && pair_profile[0]) {
        test_q8_pair_profile(pair_profile);
        ds4_gpu_cleanup();
        return 0;
    }
    test_round_bf16();
    test_prepare_q_bf16_fuse();
    test_differential_bf16_fuse();
    test_router(argv[1]);
    test_polynorm(argv[1]);
    test_mhc(argv[1]);
    test_gdla(argv[1]);
    test_latent_gdla();
    test_latent_decode_split();
    test_bf16_projection();
    ds4_gpu_cleanup();
    printf("Motif-3 H200 CUDA fixtures: BF16, router, PolyNorm, mHC, expanded/latent GDLA valid\n");
    return 0;
}
