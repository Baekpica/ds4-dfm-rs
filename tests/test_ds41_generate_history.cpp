/* The final output is a decision, not necessarily another forward input.
 * Every prompt row and every prior output consumed by decode must match. */
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

struct engram_layer {
    uint32_t il, cols;
    std::vector<int64_t> ref;
    uint64_t ref_lines;
    int64_t ref_first_bad;
};
struct gen_feed {
    uint32_t np;
    std::vector<int32_t> hist, gold_hist;
    uint64_t hist_bad;
    int64_t hist_first_bad;
    std::vector<engram_layer> layers;
};

#include "ds41_history_under_test.inc"

static int run_case(const char *name, std::vector<int32_t> hist, size_t emitted,
                    uint64_t bad, int64_t first) {
    gen_feed g = {2u, hist, {1, 2, 3, 4}, 0u, -1, {}};
    hist_vs_golden(&g, emitted);
    if (g.hist_bad != bad || g.hist_first_bad != first) {
        std::printf("FAIL %s: mismatches=%llu/%llu first=%lld/%lld\n", name,
                    (unsigned long long)g.hist_bad, (unsigned long long)bad,
                    (long long)g.hist_first_bad, (long long)first);
        return 1;
    }
    return 0;
}

static int ref_case(const char *prefix, int expected) {
    gen_feed g = {2u, {}, {}, 0u, -1, {{0u, 2u, {}, 0u, -1}}};
    const int result = load_refs(g, {3, 4}, prefix);
    if ((result == 0) != (expected == 0)) {
        std::printf("FAIL reference coverage: result=%d expected=%d\n", result, expected);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) { return 2; }
    int failures = 0;
    failures += run_case("last output unconsumed", {1, 2, 3, 0}, 2, 0, -1);
    failures += run_case("one output unconsumed", {1, 2, 0, 0}, 1, 0, -1);
    failures += run_case("prior output corrupt", {1, 2, 0, 4}, 2, 1, 2);
    failures += run_case("prompt corrupt", {1, 0, 3, 4}, 2, 1, 1);
    failures += run_case("no output prompt corrupt", {1, 0, 0, 0}, 0, 1, 1);
    failures += run_case("consumed prefix valid", {1, 2, 3, 4}, 2, 0, -1);
    failures += ref_case(argv[1], 0);
    failures += ref_case(argv[2], 2);
    std::printf("ds41 generate history gate: %s (%d/8 failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
