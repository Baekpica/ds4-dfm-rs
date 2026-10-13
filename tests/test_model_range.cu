/* Registry regression only: no model, CUDA initialization or allocation. */
#include <cstdint>
#include <cstdio>
#include <cstdlib>

enum ds4_test_range_tier { DS4_TEST_RANGE_DEVICE, DS4_TEST_RANGE_HOST };
extern "C" void ds4_test_range_reset(void);
extern "C" int ds4_test_range_add(const void *, uint64_t, uint64_t,
                                   ds4_test_range_tier);
extern "C" int ds4_test_range_covered(const void *, uint64_t, uint64_t, int *);

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "model range FAIL %d: %s\n", __LINE__, #x); \
    std::exit(1); \
} } while (0)

static char map_a, map_b;
static constexpr uint64_t UNIT_OFF = 1000, UNIT_BYTES = 500;
static constexpr uint64_t HALF_BYTES = UNIT_BYTES / 2;

static void check(const char *name, uint64_t off, uint64_t bytes,
                  int expected, int expected_exact) {
    int exact = -1;
    const int covered = ds4_test_range_covered(&map_a, off, bytes, &exact);
    if (covered == expected && exact == expected_exact) { return; }
    std::fprintf(stderr, "model range FAIL %s: covered=%d exact=%d\n",
                 name, covered, exact);
    std::exit(1);
}

static void add(const void *map, uint64_t off, uint64_t bytes,
                ds4_test_range_tier tier = DS4_TEST_RANGE_DEVICE) {
    CHECK(ds4_test_range_add(map, off, bytes, tier) == 1);
}

static void adjacent(ds4_test_range_tier tier) {
    add(&map_a, UNIT_OFF, HALF_BYTES, tier);
    add(&map_a, UNIT_OFF + HALF_BYTES, HALF_BYTES, tier);
}

int main(void) {
    /* Owner and canonical-unit boundaries differ; do not duplicate imports. */
    ds4_test_range_reset();
    adjacent(DS4_TEST_RANGE_DEVICE);
    check("adjacent imports", UNIT_OFF, UNIT_BYTES, 1, 0);
    CHECK(ds4_test_range_covered(&map_a, UNIT_OFF, UNIT_BYTES, nullptr) == 1);

    ds4_test_range_reset();
    add(&map_a, UNIT_OFF + HALF_BYTES, HALF_BYTES);
    add(&map_a, UNIT_OFF, HALF_BYTES);
    check("reversed imports", UNIT_OFF, UNIT_BYTES, 1, 0);
    check("interior across boundary", UNIT_OFF + 1, UNIT_BYTES - 2, 1, 0);
    check("exact first half", UNIT_OFF, HALF_BYTES, 1, 1);

    ds4_test_range_reset();
    add(&map_a, UNIT_OFF - 1, UNIT_BYTES + 2);
    check("wider import", UNIT_OFF, UNIT_BYTES, 1, 0);
    ds4_test_range_reset();
    add(&map_a, UNIT_OFF, UNIT_BYTES + 1);
    check("exact start", UNIT_OFF, UNIT_BYTES, 1, 1);

    /* A real hole, another map or host registration cannot fund a unit. */
    ds4_test_range_reset();
    add(&map_a, UNIT_OFF, HALF_BYTES);
    add(&map_a, UNIT_OFF + HALF_BYTES + 1, HALF_BYTES - 1);
    check("one-byte hole", UNIT_OFF, UNIT_BYTES, 0, 0);
    ds4_test_range_reset();
    adjacent(DS4_TEST_RANGE_HOST);
    add(&map_b, UNIT_OFF, UNIT_BYTES);
    check("host and other map", UNIT_OFF, UNIT_BYTES, 0, 0);
    ds4_test_range_reset();
    add(&map_a, UNIT_OFF, HALF_BYTES);
    add(&map_b, UNIT_OFF + HALF_BYTES, HALF_BYTES);
    check("cross-map halves", UNIT_OFF, UNIT_BYTES, 0, 0);
    check("empty request", UNIT_OFF, 0, 0, 0);
    check("overflow request", UINT64_MAX - 1, 2, 0, 0);

    /* Publication still rejects same-tier overlap and invalid intervals. */
    ds4_test_range_reset();
    add(&map_a, UNIT_OFF, UNIT_BYTES);
    CHECK(ds4_test_range_add(&map_a, UNIT_OFF + 1, 1,
                             DS4_TEST_RANGE_DEVICE) == 0);
    CHECK(ds4_test_range_add(&map_a, UNIT_OFF, 0,
                             DS4_TEST_RANGE_DEVICE) == 0);
    CHECK(ds4_test_range_add(&map_a, UINT64_MAX - 1, 2,
                             DS4_TEST_RANGE_DEVICE) == 0);
    add(&map_a, UNIT_OFF, UNIT_BYTES, DS4_TEST_RANGE_HOST);
    check("cross-tier overlay", UNIT_OFF, UNIT_BYTES, 1, 1);
    ds4_test_range_reset();
    add(&map_a, UINT64_MAX - HALF_BYTES, HALF_BYTES);
    check("highest valid end", UINT64_MAX - HALF_BYTES, HALF_BYTES, 1, 1);
    ds4_test_range_reset();
    check("empty registry", UNIT_OFF, UNIT_BYTES, 0, 0);

    std::puts("model range union PASS");
    return 0;
}
