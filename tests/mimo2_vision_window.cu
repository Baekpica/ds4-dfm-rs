// Actual window-64 wrapper parity, sink semantics, and unsupported layouts.
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
constexpr const char *SWITCH = "DS4_MIMO2_VISION_WINDOW";
enum class Input { Random, Flat, Sharp, Cancellation, Mixed, InvalidMax, InvalidSum, Rows, Dims };
enum class Sink { Normal, Large, Small, Nonfinite };
enum class Path { Scalar, Candidate };
enum class Storage { Shared, Separate };

struct Shape {
    int n;
    int qh = 32, kv = 8, hd = 64, stride = 3072;
    int qo = 0, ko = 2048, vo = 2560;
    int window = 64, causal = 0, group = 0, sink = 1;
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

std::vector<float> input(Shape s, Input mode) {
    std::vector<float> value((size_t)s.n * s.stride);
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] = ((int)(mix((uint32_t)i ^ 0x6287c159u) & 0xffffu) - 32768) / 32768.f;
    }
    // Permute complete rows or the dimensions within every Q/K/V head. This
    // exercises layout indices independently of the sequential input seed.
    if (mode == Input::Rows || mode == Input::Dims) {
        const auto original = value;
        for (int row = 0; row < s.n; ++row) {
            for (int column = 0; column < s.stride; ++column) {
                const int src_row = mode == Input::Rows ? s.n - row - 1 : row;
                const int src_col = mode == Input::Dims ?
                    column / s.hd * s.hd + (17 * (column % s.hd) + 5) % s.hd : column;
                value[(size_t)row * s.stride + column] = original[(size_t)src_row * s.stride + src_col];
            }
        }
    }
    for (int row = 0; row < s.n; ++row) {
        float *base = value.data() + (size_t)row * s.stride;
        for (int d = 0; d < s.qh * s.hd; ++d) {
            if (mode == Input::Flat) { base[s.qo + d] = 0.f; }
            if (mode == Input::Sharp) { base[s.qo + d] = 8.f; }
            if (mode == Input::Cancellation) { base[s.qo + d] = d % 2 ? -64.f : 64.f; }
            if (mode == Input::Mixed) {
                const int part = (d % s.hd) % 3;
                base[s.qo + d] *= part == 0 ? 1024.f : part == 1 ? 1.f / 13.f : 0.0031f;
            }
            if (mode == Input::InvalidMax) { base[s.qo + d] = INFINITY; }
            if (mode == Input::InvalidSum) { base[s.qo + d] = 1.f; }
        }
        for (int d = 0; d < s.kv * s.hd; ++d) {
            if (mode == Input::Sharp) { base[s.ko + d] = row == s.n - 1 ? 64.f : -64.f; }
            if (mode == Input::Cancellation) { base[s.ko + d] = 8.f + ((row + d) % 5) / 65536.f; }
            if (mode == Input::Mixed) {
                const int part = (d % s.hd) % 3;
                base[s.ko + d] *= part == 0 ? 1.f / 1023.f : part == 1 ? 13.37f : 321.7f;
            }
            if (mode == Input::InvalidMax) { base[s.ko + d] = 1.f; }
            if (mode == Input::InvalidSum) { base[s.ko + d] = row == 0 ? NAN : 1.f; }
        }
    }
    return value;
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
    const char *label = want == Path::Candidate ? "MiMo vision window" : "MiMo media attention";
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

// A double oracle checks the finite window/sink formula independently.
void oracle(Shape s, const std::vector<float> &x, const std::vector<float> &sinks,
            const std::vector<float> &got) {
    double max_error = 0.;
    std::vector<double> weight(s.n);
    for (int row = 0; row < s.n; ++row) {
        const int first = std::max(0, row - s.window);
        const int end = std::min(s.n, row + s.window + 1);
        for (int head = 0; head < s.qh; ++head) {
            double largest = sinks[head], sum = 0.;
            const int kv = head / (s.qh / s.kv);
            for (int key = first; key < end; ++key) {
                double dot = 0.;
                for (int d = 0; d < s.hd; ++d) {
                    dot += (double)x[(size_t)row * s.stride + s.qo + head * s.hd + d] *
                           x[(size_t)key * s.stride + s.ko + kv * s.hd + d];
                }
                weight[key] = dot / std::sqrt((double)s.hd);
                largest = std::max(largest, weight[key]);
            }
            for (int key = first; key < end; ++key) {
                weight[key] = std::exp(weight[key] - largest);
                sum += weight[key];
            }
            sum += std::exp((double)sinks[head] - largest);
            for (int d = 0; d < s.hd; ++d) {
                double value = 0.;
                for (int key = first; key < end; ++key) {
                    value += weight[key] / sum * x[(size_t)key * s.stride + s.vo + kv * s.hd + d];
                }
                const double delta = std::fabs(value - got[((size_t)row * s.qh + head) * s.hd + d]);
                max_error = std::max(max_error, delta);
            }
        }
    }
    std::printf("oracle_max_abs=%.9g ", max_error);
    require(max_error <= 2e-5, "small-case double oracle failed");
}

void run(Shape s, Input mode, Storage storage, Path path, Sink sink_mode, const char *name) {
    const auto source = input(s, mode);
    const size_t outputs = (size_t)s.n * s.qh * s.hd;
    Buffer q(source.size()), sink(s.qh), original(outputs), candidate(outputs);
    q.put(source);
    std::vector<float> sinks(s.qh);
    for (int h = 0; h < s.qh; ++h) {
        sinks[h] = (h % 7 - 3) / 4.f;
        if (sink_mode == Sink::Large) { sinks[h] = 1000.f + h; }
        if (sink_mode == Sink::Small) { sinks[h] = -1000.f - h; }
        if (sink_mode == Sink::Nonfinite) {
            sinks[h] = h % 3 == 0 ? INFINITY : h % 3 == 1 ? -INFINITY : NAN;
        }
    }
    sink.put(sinks);
    Buffer *k = &q, *v = &q;
    if (storage == Storage::Separate) {
        k = new Buffer(source.size()); v = new Buffer(source.size());
        k->put(source); v->put(source);
    }
    launch(s, original, q, *k, *v, sink, "0", Path::Scalar);
    const auto expected = original.get();
    for (const char *env : {static_cast<const char *>(nullptr), "1"}) {
        launch(s, candidate, q, *k, *v, sink, env, path);
        const auto actual = candidate.get();
        exact(expected, actual);
        for (size_t i = 0; i < actual.size(); ++i) {
            const float value = actual[i];
            const int head = (i / s.hd) % s.qh;
            if (mode == Input::InvalidMax || (sink_mode == Sink::Nonfinite && head % 3 == 0)) {
                require(std::isnan(value), "nonfinite max must produce NaN");
                continue;
            }
            require(std::isfinite(value), "finite case produced nonfinite output");
            if (mode == Input::InvalidSum || (sink_mode == Sink::Nonfinite && head % 3 == 2)) {
                require(value == 0.f, "invalid sum must produce zero");
            }
            if (sink_mode == Sink::Large) { require(value == 0.f, "dominant sink must suppress finite values"); }
        }
        candidate.guards();
    }
    exact(source, q.get());
    exact(sinks, sink.get());
    for (Buffer *buffer : {&q, k, v, &sink, &original}) { buffer->guards(); }
    if (storage == Storage::Separate) {
        exact(source, k->get()); exact(source, v->get());
        delete k; delete v;
    }
    if (path == Path::Candidate && s.n <= 65 && mode != Input::InvalidMax &&
        mode != Input::InvalidSum && sink_mode != Sink::Nonfinite) {
        oracle(s, source, sinks, expected);
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
    Shape heads = s; heads.qh = 31; reject(heads, &ot, &qt);
    auto short_out = ot; --short_out.bytes; reject(s, &short_out, &qt);
    auto short_q = qt; --short_q.bytes; reject(s, &ot, &short_q);
    q.guards(); o.guards(); sink.guards();
    std::puts("invalid-wrapper-inputs no_launch=true PASS");
}
}

int main() {
    refusals();
    for (int n : {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 130, 3072, 6144, 8192}) {
        run(Shape{n}, Input::Random, Storage::Shared, Path::Candidate, Sink::Normal, "window-tail");
    }
    for (Input mode : {Input::Flat, Input::Sharp, Input::Cancellation, Input::Mixed,
                       Input::InvalidMax, Input::InvalidSum, Input::Rows, Input::Dims}) {
        run(Shape{33}, mode, Storage::Shared, Path::Candidate, Sink::Normal, "numeric-layout");
    }
    for (Sink sink : {Sink::Large, Sink::Small, Sink::Nonfinite}) {
        run(Shape{65}, Input::Random, Storage::Shared, Path::Candidate, sink, "sink-boundary");
    }
    for (Input mode : {Input::Rows, Input::Dims}) {
        run(Shape{130}, mode, Storage::Shared, Path::Candidate, Sink::Normal, "window-permutation");
    }
    Shape s{33}; s.window = 32;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "window32");
    s = Shape{33}; s.window = -1;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "full-with-sink");
    s = Shape{33}; s.sink = 0;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "window-no-sink");
    s = Shape{33}; s.causal = 1;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "causal");
    s = Shape{33}; s.group = 4;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "group");
    s = Shape{33}; s.qh = 16;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "heads");
    s = Shape{33}; s.hd = 32;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "head-dimension");
    s = Shape{33}; s.stride += 16;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "stride");
    s = Shape{33}; s.qo = 8;
    run(s, Input::Random, Storage::Shared, Path::Scalar, Sink::Normal, "offset");
    run(Shape{33}, Input::Random, Storage::Separate, Path::Scalar, Sink::Normal, "separate-qkv");
    unsetenv(SWITCH);
    return 0;
}
