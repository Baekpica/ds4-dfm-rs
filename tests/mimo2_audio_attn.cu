// Actual full-causal audio wrapper parity and separate-Q/K/V indexing.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static bool m2_hmma_available = false;
static unsigned stream_calls = 0;
static const char *launch_label = nullptr;
static int cuda_ok(cudaError_t rc, const char *label) {
    launch_label = label;
    return rc == cudaSuccess;
}
static cudaStream_t ds4_current_stream(void) { ++stream_calls; return 0; }
static int ds4_capture_active(void) { return 0; }
static const char *cuda_model_range_ptr(
        const void *map, uint64_t offset, uint64_t, const char *) {
    return (const char *)map + offset;
}
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int owner; int memc; };
#include "../ds4_mimo2_gpu.cuh"

namespace {
constexpr size_t GUARD = 64;
constexpr uint32_t SENTINEL = 0x7fc0abcd;
constexpr const char *SWITCH = "DS4_MIMO2_AUDIO_ATTN";
enum class Input { Random, Flat, Sharp, Cancellation, Mixed, InvalidMax, InvalidSum, Rows, Dims };
enum class Path { Scalar, Candidate };
enum class Storage { Shared, Separate };

struct Shape {
    int n;
    int qh = 16, kv = 16, hd = 64, stride = 1024;
    int qo = 0, ko = 0, vo = 0;
    int window = -1, causal = 1, group = 0, sink = 0;
};

void check(cudaError_t rc) {
    if (rc == cudaSuccess) { return; }
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(rc));
    std::exit(1);
}

void require(bool ok, const char *why) {
    if (ok) { return; }
    std::fprintf(stderr, "%s\n", why);
    std::exit(2);
}

// NaN padding exposes stray writes and most accidental reads without changing
// the actual tensor's reported bounds or alignment.
struct Buffer {
    float *base = nullptr, *ptr = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) {
        check(cudaMalloc(&base, (n + 2 * GUARD) * sizeof(float)));
        ptr = base + GUARD;
        std::vector<uint32_t> init(n + 2 * GUARD, SENTINEL);
        check(cudaMemcpy(base, init.data(), init.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    ~Buffer() { cudaFree(base); }
    ds4_gpu_tensor tensor() { return {ptr, count * sizeof(float), 0, 0}; }
    void put(const std::vector<float> &value) {
        require(value.size() == count, "input size mismatch");
        check(cudaMemcpy(ptr, value.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<float> get() const {
        std::vector<float> value(count);
        check(cudaMemcpy(value.data(), ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
        return value;
    }
    void guards() const {
        uint32_t lo[GUARD], hi[GUARD];
        check(cudaMemcpy(lo, base, sizeof(lo), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hi, ptr + count, sizeof(hi), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < GUARD; ++i) {
            require(lo[i] == SENTINEL && hi[i] == SENTINEL, "tensor guard changed");
        }
    }
};

uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

struct Inputs { std::vector<float> q, k, v; };

Inputs input(Shape s, Input mode) {
    const size_t count = (size_t)s.n * s.stride;
    Inputs x{std::vector<float>(count), std::vector<float>(count), std::vector<float>(count)};
    for (size_t i = 0; i < count; ++i) {
        x.q[i] = ((int)(mix((uint32_t)i ^ 0x6287c159u) & 0xffffu) - 32768) / 32768.f;
        x.k[i] = ((int)(mix((uint32_t)i ^ 0x6287c15au) & 0xffffu) - 32768) / 32768.f;
        x.v[i] = ((int)(mix((uint32_t)i ^ 0x6287c15bu) & 0xffffu) - 32768) / 32768.f;
    }
    for (int row = 0; row < s.n; ++row) {
        for (int d = 0; d < s.qh * s.hd; ++d) {
            float &q = x.q[(size_t)row * s.stride + s.qo + d];
            if (mode == Input::Flat) { q = 0.f; }
            if (mode == Input::Sharp) { q = 8.f; }
            if (mode == Input::Cancellation) { q = d % 2 ? -64.f : 64.f; }
            if (mode == Input::Mixed) {
                const int part = d % s.hd % 3;
                q *= part == 0 ? 1024.f : part == 1 ? 1.f / 13.f : 0.0031f;
            }
            if (mode == Input::InvalidMax) { q = INFINITY; }
            if (mode == Input::InvalidSum) { q = 1.f; }
        }
        for (int d = 0; d < s.kv * s.hd; ++d) {
            float &k = x.k[(size_t)row * s.stride + s.ko + d];
            if (mode == Input::Sharp) { k = row == s.n - 1 ? 64.f : -64.f; }
            if (mode == Input::Cancellation) { k = 8.f + ((row + d) % 5) / 65536.f; }
            if (mode == Input::Mixed) {
                const int part = d % s.hd % 3;
                k *= part == 0 ? 1.f / 1023.f : part == 1 ? 13.37f : 321.7f;
            }
            if (mode == Input::InvalidMax) { k = 1.f; }
            if (mode == Input::InvalidSum) { k = row == 0 ? NAN : 1.f; }
        }
    }
    // Distinct Q/K/V values expose accidental pointer reuse. Permute each
    // tensor's rows or head dimensions to exercise stride and tail indices.
    if (mode == Input::Rows || mode == Input::Dims) {
        for (auto *value : {&x.q, &x.k, &x.v}) {
            const auto original = *value;
            for (int row = 0; row < s.n; ++row) {
                for (int d = 0; d < s.stride; ++d) {
                    const int sr = mode == Input::Rows ? s.n - row - 1 : row;
                    const int sd = mode == Input::Dims ? d / s.hd * s.hd + (17 * (d % s.hd) + 5) % s.hd : d;
                    (*value)[(size_t)row * s.stride + d] = original[(size_t)sr * s.stride + sd];
                }
            }
        }
    }
    return x;
}

int call(Shape s, ds4_gpu_tensor *out, ds4_gpu_tensor *q,
         ds4_gpu_tensor *k, ds4_gpu_tensor *v, const Buffer &sink) {
    return ds4_gpu_mimo2_attn(out, q, k, v, sink.ptr, sink.count * sizeof(float), 0, s.sink,
        s.n, s.qh, s.kv, s.hd, s.stride, s.stride, s.stride,
        s.qo, s.ko, s.vo, s.window, s.causal, s.group);
}

void launch(Shape s, Buffer &out, Buffer &q, Buffer &k, Buffer &v,
            const Buffer &sink, const char *env, Path want) {
    if (env) { setenv(SWITCH, env, 1); } else { unsetenv(SWITCH); }
    auto o = out.tensor(), qt = q.tensor(), kt = k.tensor(), vt = v.tensor();
    stream_calls = 0;
    launch_label = nullptr;
    require(call(s, &o, &qt, &kt, &vt, sink) == 1, "wrapper failed");
    check(cudaDeviceSynchronize());
    const char *label = want == Path::Candidate ? "MiMo audio attention" : "MiMo media attention";
    require(stream_calls == 1 && launch_label && !std::strcmp(launch_label, label), "unexpected wrapper path");
}

void exact(const std::vector<float> &a, const std::vector<float> &b) {
    require(a.size() == b.size(), "output size mismatch");
    if (!std::memcmp(a.data(), b.data(), a.size() * sizeof(float))) { return; }
    double squared = 0.;
    float maximum = 0.f;
    size_t mismatches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float))) { ++mismatches; }
        const float delta = std::fabs(a[i] - b[i]);
        maximum = std::fmax(maximum, delta);
        squared += (double)delta * delta;
    }
    std::fprintf(stderr, "byte parity failed: mismatches=%zu max_abs=%.9g rms=%.9g\n",
                 mismatches, maximum, std::sqrt(squared / a.size()));
    std::exit(3);
}

// Independent small-case formula: full causal keys are exactly [0, row].
void oracle(Shape s, const Inputs &x, const std::vector<float> &got) {
    double max_error = 0.;
    std::vector<double> weight(s.n);
    for (int row = 0; row < s.n; ++row) {
        for (int head = 0; head < s.qh; ++head) {
            double largest = -INFINITY, sum = 0.;
            const int kv = head / (s.qh / s.kv);
            for (int key = 0; key <= row; ++key) {
                double dot = 0.;
                for (int d = 0; d < s.hd; ++d) {
                    dot += (double)x.q[(size_t)row * s.stride + s.qo + head * s.hd + d] *
                           x.k[(size_t)key * s.stride + s.ko + kv * s.hd + d];
                }
                weight[key] = dot / std::sqrt((double)s.hd);
                largest = std::max(largest, weight[key]);
            }
            for (int key = 0; key <= row; ++key) {
                weight[key] = std::exp(weight[key] - largest);
                sum += weight[key];
            }
            for (int d = 0; d < s.hd; ++d) {
                double value = 0.;
                for (int key = 0; key <= row; ++key) {
                    value += weight[key] / sum * x.v[(size_t)key * s.stride + s.vo + kv * s.hd + d];
                }
                max_error = std::max(max_error, std::fabs(value - got[((size_t)row * s.qh + head) * s.hd + d]));
            }
        }
    }
    std::printf("oracle_max_abs=%.9g ", max_error);
    require(max_error <= 2e-5, "small-case double oracle failed");
}

void run(Shape s, Input mode, Storage storage, Path path, const char *name) {
    const auto source = input(s, mode);
    const size_t outputs = (size_t)s.n * s.qh * s.hd;
    Buffer q(source.q.size()), kb(source.k.size()), vb(source.v.size());
    Buffer sink(s.qh), original(outputs), candidate(outputs);
    q.put(source.q); kb.put(source.k); vb.put(source.v);
    std::vector<float> sinks(s.qh);
    for (int h = 0; h < s.qh; ++h) { sinks[h] = (h % 7 - 3) / 4.f; }
    sink.put(sinks);
    Buffer &k = storage == Storage::Shared ? q : kb;
    Buffer &v = storage == Storage::Shared ? q : vb;
    launch(s, original, q, k, v, sink, "0", Path::Scalar);
    const auto expected = original.get();
    for (const char *env : {static_cast<const char *>(nullptr), "1"}) {
        launch(s, candidate, q, k, v, sink, env, path);
        const auto actual = candidate.get();
        exact(expected, actual);
        for (size_t i = 0; i < actual.size(); ++i) {
            const float value = actual[i];
            const size_t row = i / (s.qh * s.hd);
            if (mode == Input::InvalidMax || (mode == Input::InvalidSum && row == 0)) {
                require(std::isnan(value), "nonfinite maximum must produce NaN");
                continue;
            }
            require(std::isfinite(value), "finite case produced nonfinite output");
            if (mode == Input::InvalidSum) { require(value == 0.f, "invalid denominator must produce zero"); }
        }
        candidate.guards();
    }
    exact(source.q, q.get()); exact(source.k, kb.get()); exact(source.v, vb.get());
    exact(sinks, sink.get());
    for (Buffer *buffer : {&q, &kb, &vb, &sink, &original}) { buffer->guards(); }
    if (path == Path::Candidate && s.n <= 33 && mode != Input::InvalidMax && mode != Input::InvalidSum) {
        oracle(s, source, expected);
    }
    std::printf("case=%s rows=%d path=%s off_default_on_exact=true guards=true PASS\n",
                name, s.n, path == Path::Candidate ? "candidate" : "fallback");
    std::fflush(stdout);
}

void refusals() {
    Shape s{1};
    Buffer q(s.stride), o(s.qh * s.hd), sink(s.qh);
    auto qt = q.tensor(), ot = o.tensor();
    auto reject = [&](Shape shape, ds4_gpu_tensor *out, ds4_gpu_tensor *query) {
        stream_calls = 0;
        require(call(shape, out, query, &qt, &qt, sink) == 0, "invalid input accepted");
        require(stream_calls == 0, "invalid input launched GPU work");
    };
    reject(s, nullptr, &qt);
    reject(s, &ot, nullptr);
    Shape zero = s; zero.n = 0; reject(zero, &ot, &qt);
    Shape heads = s; heads.qh = 15; reject(heads, &ot, &qt);
    auto short_out = ot; --short_out.bytes; reject(s, &short_out, &qt);
    auto short_q = qt; --short_q.bytes; reject(s, &ot, &short_q);
    q.guards(); o.guards(); sink.guards();
    std::puts("invalid-wrapper-inputs no_launch=true PASS");
}
}

int main() {
    refusals();
    for (int n : {1, 31, 32, 33, 127, 128, 129, 557, 727, 1024, 8192}) {
        run(Shape{n}, Input::Random, Storage::Separate, Path::Candidate, "causal-tail");
    }
    for (Input mode : {Input::Flat, Input::Sharp, Input::Cancellation, Input::Mixed,
                       Input::InvalidMax, Input::InvalidSum, Input::Rows, Input::Dims}) {
        run(Shape{33}, mode, Storage::Separate, Path::Candidate, "numeric-layout");
    }
    Shape s{33}; s.causal = 0;
    run(s, Input::Random, Storage::Separate, Path::Scalar, "noncausal");
    s = Shape{129}; s.window = 128;
    run(s, Input::Random, Storage::Separate, Path::Scalar, "causal-window128");
    s = Shape{33}; s.group = 4;
    run(s, Input::Random, Storage::Separate, Path::Scalar, "group4");
    s = Shape{33}; s.sink = 1;
    run(s, Input::Random, Storage::Separate, Path::Scalar, "with-sink");
    s = Shape{33}; s.qh = 8; s.kv = 8;
    run(s, Input::Random, Storage::Separate, Path::Scalar, "heads");
    s = Shape{33}; s.hd = 32;
    run(s, Input::Random, Storage::Separate, Path::Scalar, "head-dimension");
    s = Shape{33}; s.stride += 16;
    run(s, Input::Random, Storage::Separate, Path::Scalar, "stride");
    s = Shape{33}; s.stride += 16; s.qo = 8;
    run(s, Input::Random, Storage::Separate, Path::Scalar, "offset");
    run(Shape{33}, Input::Random, Storage::Shared, Path::Scalar, "aliased-qkv");
    unsetenv(SWITCH);
    return 0;
}
