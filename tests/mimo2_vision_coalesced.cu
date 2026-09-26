// Reuse test-only buffers, input generators, and the actual production wrapper.
// The retained test's main is separate; this gate exercises new dispatch only.
#define main retained_vision_test_main
#include "mimo2_vision_attn.cu"
#undef main

namespace {
constexpr const char *COALESCED = "DS4_MIMO2_VISION_COALESCED";
enum class Dispatch { Scalar, Retained, Coalesced, Window, Audio };

const char *label(Dispatch path) {
    switch (path) {
        case Dispatch::Scalar: return "MiMo media attention";
        case Dispatch::Retained: return "MiMo vision attention";
        case Dispatch::Coalesced: return "MiMo vision coalesced attention";
        case Dispatch::Window: return "MiMo vision window";
        case Dispatch::Audio: return "MiMo audio attention";
    }
    std::abort();
}

void dispatch(Shape s, Buffer &out, Buffer &q, Buffer &k, Buffer &v,
              const Buffer &sink, const char *setting, Dispatch want) {
    if (setting) { setenv(COALESCED, setting, 1); } else { unsetenv(COALESCED); }
    auto o = out.tensor(), qt = q.tensor(), kt = k.tensor(), vt = v.tensor();
    stream_calls = 0;
    launch_label = nullptr;
    require(call(s, &o, &qt, &kt, &vt, sink) == 1, "wrapper failed");
    check(cudaDeviceSynchronize());
    if (stream_calls != 1 || !launch_label || std::strcmp(launch_label, label(want))) {
        std::fprintf(stderr, "dispatch mismatch: rows=%d env=%s got=%s want=%s\n",
                     s.n, setting ? setting : "default", launch_label ? launch_label : "none", label(want));
        std::exit(4);
    }
}

void dispatch_case(Shape s, Input mode, Storage storage, Dispatch want, const char *name) {
    auto source = input(s, mode);
    if (mode == Input::Random) {
        // Exercise all mantissa bits without saturating every softmax to one key.
        for (size_t i = 0; i < source.size(); ++i) {
            const uint32_t bits = mix((uint32_t)i ^ 0x7c394b19u);
            const uint32_t fp = (bits & 0x807fffffu) | ((124u + ((bits >> 24) & 3u)) << 23);
            std::memcpy(&source[i], &fp, sizeof(fp));
        }
    }
    Buffer q(source.size()), sink(s.qh), before((size_t)s.n * s.qh * s.hd), after(before.count);
    q.put(source);
    std::vector<float> sinks(s.qh);
    for (int h = 0; h < s.qh; ++h) { sinks[h] = (h % 7 - 3) / 4.f; }
    sink.put(sinks);
    Buffer *k = &q, *v = &q;
    if (storage == Storage::Separate) {
        k = new Buffer(source.size()); v = new Buffer(source.size());
        k->put(source); v->put(source);
    }
    setenv(SWITCH, "1", 1);
    const Dispatch off = want == Dispatch::Coalesced ? Dispatch::Retained : want;
    dispatch(s, before, q, *k, *v, sink, "0", off);
    const auto expected = before.get();
    for (const char *setting : {static_cast<const char *>(nullptr), "1"}) {
        dispatch(s, after, q, *k, *v, sink, setting, want);
        const auto actual = after.get();
        exact(expected, actual);
        for (float value : actual) {
            if (mode == Input::InvalidMax) { require(std::isnan(value), "invalid max must produce NaN"); }
            else { require(std::isfinite(value), "finite case produced nonfinite output"); }
            if (mode == Input::InvalidSum) { require(value == 0.f, "invalid sum must produce zero"); }
        }
        after.guards();
    }
    exact(source, q.get()); exact(sinks, sink.get());
    for (Buffer *buffer : {&q, k, v, &sink, &before}) { buffer->guards(); }
    if (storage == Storage::Separate) {
        exact(source, k->get()); exact(source, v->get());
        delete k; delete v;
    }
    std::printf("case=%s rows=%d path=%s default_on_off_exact=true input_unchanged=true guards=true PASS\n",
                name, s.n, label(want));
    std::fflush(stdout);
}

void scalar_switch() {
    Shape s{M2V_K_MIN};
    const auto source = input(s, Input::Random);
    Buffer q(source.size()), sink(s.qh), before((size_t)s.n * s.qh * s.hd), after(before.count);
    q.put(source); sink.put(std::vector<float>(s.qh, 0.f));
    setenv(SWITCH, "0", 1);
    dispatch(s, before, q, q, q, sink, "0", Dispatch::Scalar);
    dispatch(s, after, q, q, q, sink, "1", Dispatch::Scalar);
    exact(before.get(), after.get());
    q.guards(); sink.guards(); before.guards(); after.guards();
    std::puts("vision-switch-scalar hierarchy=true PASS");
}
}

int main() {
    unsetenv("DS4_MIMO2_VISION_WINDOW");
    unsetenv("DS4_MIMO2_AUDIO_ATTN");
    setenv(COALESCED, "1", 1);
    refusals();
    std::vector<int> rows = {M2V_K_MIN - 1, M2V_K_MIN, M2V_K_MIN + 1,
                             879, 880, 881, M2V_K_MAX - 1, M2V_K_MAX, M2V_K_MAX + 1};
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (int n : rows) {
        const Dispatch path = n >= M2V_K_MIN && n <= M2V_K_MAX ? Dispatch::Coalesced : Dispatch::Retained;
        dispatch_case(Shape{n}, Input::Random, Storage::Shared, path, "range-boundary");
    }
    for (Input mode : {Input::Mixed, Input::InvalidMax, Input::InvalidSum}) {
        dispatch_case(Shape{880}, mode, Storage::Shared, Dispatch::Coalesced, "numeric-boundary");
    }
    // Keep rows inside the new range so another failed predicate explains fallback.
    Shape s{880}; s.window = 64; s.sink = 1;
    dispatch_case(s, Input::Random, Storage::Shared, Dispatch::Window, "window-sink");
    s = Shape{880}; s.sink = 1;
    dispatch_case(s, Input::Random, Storage::Shared, Dispatch::Scalar, "full-sink");
    s = Shape{880}; s.causal = 1;
    dispatch_case(s, Input::Random, Storage::Shared, Dispatch::Scalar, "causal");
    s = Shape{880}; s.group = 4;
    dispatch_case(s, Input::Random, Storage::Shared, Dispatch::Scalar, "group");
    s = Shape{880}; s.qh = 16;
    dispatch_case(s, Input::Random, Storage::Shared, Dispatch::Scalar, "heads");
    s = Shape{880}; s.hd = 32;
    dispatch_case(s, Input::Random, Storage::Shared, Dispatch::Scalar, "head-dimension");
    s = Shape{880}; s.stride += 16;
    dispatch_case(s, Input::Random, Storage::Shared, Dispatch::Scalar, "stride");
    s = Shape{880}; s.qo = 8;
    dispatch_case(s, Input::Random, Storage::Shared, Dispatch::Scalar, "offset");
    dispatch_case(Shape{880}, Input::Random, Storage::Separate, Dispatch::Scalar, "separate-qkv");
    s = Shape{880}; s.qh = s.kv = 16; s.stride = 1024; s.ko = s.vo = 0; s.causal = 1;
    dispatch_case(s, Input::Random, Storage::Separate, Dispatch::Audio, "audio-layout");
    scalar_switch();
    unsetenv(SWITCH); unsetenv(COALESCED);
    return 0;
}
