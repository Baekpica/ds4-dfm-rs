/* Compare every F32 recurrence state, before BF16 probabilities can hide drift. */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../cuda/naive_primitives.cuh"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)
#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); exit(1); \
} } while (0)

enum class Recurrence { Source, Walk, Unit };
enum class Seed { Empty, Sink };
enum class Direction { Down, Up };

struct Case {
    unsigned begin, count;
    float maximum, denominator;
};

struct State {
    uint32_t maximum, denominator;
};

struct Suite {
    std::vector<Case> cases;
    std::vector<float> scores;
    std::vector<const char *> names;
};

static constexpr unsigned THREADS = 128;
static constexpr uint16_t BF16_EXP = 0x7f80;
static constexpr uint16_t BF16_MAX = 0x7f7f;
static constexpr uint16_t BF16_SIGN = 0x8000;
static constexpr float LN2 = 0.6931471805599453f;

static float from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t bits(float value) {
    uint32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static float bf16(uint16_t code) {
    return from_bits((uint32_t)code << 16);
}

static uint16_t rounded(float value) {
    const uint32_t word = bits(value);
    return (uint16_t)((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}

static float neighbor(uint16_t code, Direction direction) {
    const uint16_t magnitude = code & ~BF16_SIGN;
    if (!magnitude) {
        return bf16(direction == Direction::Up ? 1 : BF16_SIGN | 1);
    }
    const int delta = (direction == Direction::Up) == !(code & BF16_SIGN) ? 1 : -1;
    if (magnitude == BF16_MAX && delta == 1) { return bf16(code); }
    return bf16((uint16_t)(code + delta));
}

static void add_case(Suite &suite, const char *name, const std::vector<float> &scores,
                     Seed seed, float sink = 0.0f) {
    CHECK(!scores.empty());
    CHECK(suite.scores.size() + scores.size() <= UINT32_MAX);
    // These are the attention kernel's only initial states; denominators evolve causally.
    const float maximum = seed == Seed::Sink ? sink : -INFINITY;
    const float denominator = seed == Seed::Sink ? 1.0f : 0.0f;
    suite.cases.push_back({(unsigned)suite.scores.size(), (unsigned)scores.size(),
                           maximum, denominator});
    suite.names.push_back(name);
    suite.scores.insert(suite.scores.end(), scores.begin(), scores.end());
}

static void finite_cases(Suite &suite) {
    // Every finite BF16 code, including both zero signs and all subnormal exponents.
    for (unsigned code = 0; code <= UINT16_MAX; code++) {
        if ((code & BF16_EXP) == BF16_EXP) { continue; }
        const float score = bf16((uint16_t)code);
        const std::vector<float> values = {
            score, score, neighbor((uint16_t)code, Direction::Down),
            neighbor((uint16_t)code, Direction::Up), 0.0f, -0.0f,
            bf16((uint16_t)(code ^ BF16_SIGN)), score
        };
        add_case(suite, "finite BF16 / empty", values, Seed::Empty);
        add_case(suite, "finite BF16 / sink", values, Seed::Sink);
    }

    const std::vector<float> scores = {
        -0.0f, 0.0f, bf16(1), bf16(BF16_SIGN | 1), -1.0f, 1.0f,
        bf16(BF16_MAX), bf16(BF16_MAX | BF16_SIGN), 1.0f, 1.0f
    };
    const float sinks[] = {
        -0.0f, 0.0f, 0.0314159265f, -0.0314159265f, 0.99999994f,
        1.00000012f, -1.00000012f, 80.0f, -80.0f,
        std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()
    };
    for (float sink : sinks) { add_case(suite, "F32 sink boundary", scores, Seed::Sink, sink); }
}

static void causal_cases(Suite &suite) {
    for (unsigned count : {1u, 2u, 3u, 31u, 127u, 128u, 2047u, (unsigned)N05_TOP_K}) {
        for (unsigned pattern = 0; pattern < 4; pattern++) {
            std::vector<float> scores(count);
            for (unsigned i = 0; i < count; i++) {
                float score = 0.0f;
                if (pattern == 1) { score = -80.0f + 0.25f * i; }
                if (pattern == 2) { score = 80.0f - 0.25f * i; }
                if (pattern == 3) { score = ((i * 73u) % 257u) * 0.03125f - 4.0f; }
                scores[i] = bf16(rounded(score));
            }
            add_case(suite, "causal ties/rise/fall/mixed", scores, Seed::Empty);
            if (count <= N05_WINDOW) {
                add_case(suite, "window ties/rise/fall/mixed", scores, Seed::Sink, 0.125f);
            }
        }
    }

    // A natural power-of-two denominator makes exp(score) straddle its half-ULP.
    // BF16 neighbors surround that RNE boundary without injecting an impossible state.
    for (unsigned exponent = 0; exponent <= 10; exponent++) {
        const uint16_t center = rounded(-(24.0f - exponent) * LN2);
        for (float score : {neighbor(center, Direction::Down), bf16(center),
                            neighbor(center, Direction::Up)}) {
            std::vector<float> scores(1u << exponent, 0.0f);
            scores.push_back(score);
            add_case(suite, "denominator half-ULP / empty", scores, Seed::Empty);
            if (exponent <= 6) {
                scores.erase(scores.begin());
                add_case(suite, "denominator half-ULP / sink", scores, Seed::Sink);
            }
        }
    }
}

static void exceptional_cases(Suite &suite) {
    // All BF16 infinities/NaN payloads must use the original exceptional-value equation.
    for (unsigned code = 0; code <= UINT16_MAX; code++) {
        if ((code & BF16_EXP) != BF16_EXP) { continue; }
        const float value = bf16((uint16_t)code);
        const std::vector<float> scores = {0.0f, value, -0.0f, 1.0f, value, -1.0f};
        add_case(suite, "exceptional score / empty", scores, Seed::Empty);
        add_case(suite, "exceptional score / sink", scores, Seed::Sink, 0.125f);
        add_case(suite, "exceptional sink", {value, 0.0f, -0.0f, 1.0f, -1.0f},
                 Seed::Sink, value);
    }
}

template<Recurrence MODE> __global__ static void recurrence(
        const Case *cases, const float *scores, State *states, unsigned count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const Case item = cases[index];
    float maximum = item.maximum, denominator = item.denominator;
    for (unsigned i = 0; i < item.count; i++) {
        const float score = scores[item.begin + i];
        if constexpr (MODE == Recurrence::Source) {
            // Standalone pre-optimization equation, compiled independently of either helper.
            const float next = fmaxf(maximum, score);
            denominator = denominator * expf(maximum - next) + expf(score - next);
            maximum = next;
        } else if constexpr (MODE == Recurrence::Walk) {
            naive_softmax_step<NaiveSoftmax::Walk>(score, maximum, denominator);
        } else {
            naive_softmax_step<NaiveSoftmax::Unit>(score, maximum, denominator);
        }
        states[item.begin + i] = {__float_as_uint(maximum), __float_as_uint(denominator)};
    }
}

template<class T> static T *device(size_t count) {
    T *pointer;
    CUDA(cudaMalloc(&pointer, count * sizeof(T)));
    return pointer;
}

template<Recurrence MODE> static std::vector<State> capture(
        const Suite &suite, const Case *cases, const float *scores, State *states) {
    const unsigned count = (unsigned)suite.cases.size();
    recurrence<MODE><<<(count + THREADS - 1) / THREADS, THREADS>>>(cases, scores, states, count);
    CUDA(cudaGetLastError());
    std::vector<State> result(suite.scores.size());
    CUDA(cudaMemcpy(result.data(), states, result.size() * sizeof(State), cudaMemcpyDeviceToHost));
    return result;
}

static void compare(const Suite &suite, const char *mode, const std::vector<State> &source,
                    const std::vector<State> &got) {
    CHECK(source.size() == got.size());
    for (unsigned index = 0; index < suite.cases.size(); index++) {
        const Case item = suite.cases[index];
        for (unsigned step = 0; step < item.count; step++) {
            const size_t offset = item.begin + step;
            const State a = source[offset], b = got[offset];
            if (a.maximum == b.maximum && a.denominator == b.denominator) { continue; }
            fprintf(stderr, "FAIL %s %s case=%u step=%u score=%08x seed=%08x/%08x "
                            "source=%08x/%08x got=%08x/%08x\n", mode, suite.names[index],
                    index, step, bits(suite.scores[offset]), bits(item.maximum),
                    bits(item.denominator), a.maximum, a.denominator, b.maximum, b.denominator);
            exit(1);
        }
    }
    printf("%s byte-exact max/den: %zu cases, %zu causal steps OK\n",
           mode, suite.cases.size(), suite.scores.size());
}

int main(void) {
    Suite suite;
    finite_cases(suite);
    causal_cases(suite);
    exceptional_cases(suite);

    Case *cases = device<Case>(suite.cases.size());
    float *scores = device<float>(suite.scores.size());
    State *states = device<State>(suite.scores.size());
    CUDA(cudaMemcpy(cases, suite.cases.data(), suite.cases.size() * sizeof(Case), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(scores, suite.scores.data(), suite.scores.size() * sizeof(float), cudaMemcpyHostToDevice));

    const std::vector<State> source = capture<Recurrence::Source>(suite, cases, scores, states);
    compare(suite, "Walk", source, capture<Recurrence::Walk>(suite, cases, scores, states));
    compare(suite, "Unit", source, capture<Recurrence::Unit>(suite, cases, scores, states));

    CUDA(cudaFree(states));
    CUDA(cudaFree(scores));
    CUDA(cudaFree(cases));
    return 0;
}
