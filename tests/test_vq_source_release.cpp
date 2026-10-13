/* A resident device copy must not retain a second file-cache copy on UMA. */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include "../ds4_mem_census.h"
#include "../ds4_model_catalog.h"

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "VQ source release FAIL %d: %s\n", __LINE__, #x); \
    std::exit(1); \
} } while (0)

using CUmemGenericAllocationHandle = uint64_t;
using CUdeviceptr = uint64_t;
#include "range.inc"

static const void *g_model_fd_host_base;
static int g_model_fd = -1, g_in_governed_materialize;
static uint64_t g_model_registered_size, g_model_range_bytes;
static struct { struct { uint64_t map_len; } v[DS4_MSRC_MAX]; } g_model_srcs;
static struct { int keep_model_pages, verbose; } env;
static const auto &cuda_weight_env() { return env; }
#include "release.inc"

enum class Copy { Aligned, Flat, Failed };
static Copy copy_mode;
static std::vector<char> device;
static cuda_model_range published;
static int publications, staged;
static int cuda_mem_src_index(const void *) { return 0; }
static char *cuda_vq_blob_build_aligned(const void *map, uint64_t, uint64_t offset,
                                      uint64_t bytes, uint64_t *, const char *, int *flat) {
    *flat = copy_mode == Copy::Flat;
    if (copy_mode != Copy::Aligned) { return nullptr; }
    const char *src = static_cast<const char *>(map) + offset;
    device.assign(src, src + bytes);
    return device.data();
}
static char *cuda_vmm_arena_alloc(const void *, uint64_t bytes, const char *) {
    device.resize(bytes);
    return device.data();
}
static char *cuda_model_arena_alloc(const void *, uint64_t, const char *) { return nullptr; }
static int cuda_stage_copy_to_dev(const void *, uint64_t, uint64_t, char *, int, const char *) {
    ++staged;
    return 1;
}
static int cuda_model_range_publish(cuda_model_range range) {
    CHECK(g_in_governed_materialize == 1);
    published = range;
    ++publications;
    return 1;
}
static void cuda_substrate_cover(const void *, uint64_t) {}
static void cuda_model_load_progress_note(uint64_t) {}
#include "materialize.inc"

struct Mapping {
    int fd;
    char *data;
    size_t page, bytes;
};

static Mapping mapping() {
    // The worktree's disk filesystem supports cache eviction; tmpfs does not.
    char path[] = ".vq-source-release-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(unlink(path) == 0);
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t bytes = 5 * page;
    CHECK(ftruncate(fd, bytes) == 0);
    void *ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    CHECK(ptr != MAP_FAILED);
    return {fd, static_cast<char *>(ptr), page, bytes};
}

static void fill(const Mapping &map) {
    for (size_t i = 0; i < map.bytes; ++i) { map.data[i] = static_cast<char>(i % 127); }
    CHECK(msync(map.data, map.bytes, MS_SYNC) == 0);
    CHECK(fdatasync(map.fd) == 0);
}

static int resident(const Mapping &map) {
    unsigned char pages[3] = {};
    CHECK(mincore(map.data + map.page, 3 * map.page, pages) == 0);
    return (pages[0] & 1) + (pages[1] & 1) + (pages[2] & 1);
}

static void reset(const Mapping &map) {
    fill(map);
    CHECK(resident(map) == 3);
    copy_mode = Copy::Aligned;
    env = {};
    g_model_fd = map.fd;
    g_model_fd_host_base = map.data;
    g_model_srcs.v[0].map_len = g_model_registered_size = map.bytes;
    g_model_range_bytes = 0;
    publications = staged = 0;
    device.clear();
}

static int promote(const Mapping &map, int discard) {
    ds4_phys_unit unit = {};
    unit.src_off = map.page;
    unit.src_bytes = 3 * map.page;
    return cuda_unit_materialize_copy(map.data, &unit, discard, "fixture");
}

int main() {
    Mapping primary = mapping(), aux = mapping();
    reset(primary);
    CHECK(promote(primary, 1) == DS4_RESUNIT_POPULATED);
    CHECK(resident(primary) == 0);  /* Actual mincore, not a mocked advice call. */
    CHECK(publications == 1 && published.device_ptr == device.data());
    CHECK(g_model_range_bytes == 3 * primary.page);
    for (size_t i = 0; i < device.size(); ++i) {
        CHECK(device[i] == static_cast<char>((i + primary.page) % 127));
    }

    reset(primary);
    CHECK(promote(primary, 0) == DS4_RESUNIT_POPULATED);
    CHECK(resident(primary) == 3);

    reset(primary);
    env.keep_model_pages = 1;
    CHECK(promote(primary, 1) == DS4_RESUNIT_POPULATED);
    CHECK(resident(primary) == 3);

    reset(primary);
    fill(aux);
    CHECK(promote(aux, 1) == DS4_RESUNIT_POPULATED);
    CHECK(resident(primary) == 3 && resident(aux) == 3);  /* Never advise another map's fd. */

    reset(primary);
    g_model_fd = -1;
    CHECK(promote(primary, 1) == DS4_RESUNIT_POPULATED);
    CHECK(resident(primary) == 3);

    reset(primary);
    copy_mode = Copy::Failed;
    CHECK(promote(primary, 1) == DS4_RESUNIT_FAILED);
    CHECK(resident(primary) == 3 && publications == 0 && g_model_range_bytes == 0);

    reset(primary);
    copy_mode = Copy::Flat;
    CHECK(promote(primary, 1) == DS4_RESUNIT_POPULATED);
    CHECK(staged == 1 && resident(primary) == 3);  /* Flat pipeline owns its release. */

    CHECK(munmap(primary.data, primary.bytes) == 0 && close(primary.fd) == 0);
    CHECK(munmap(aux.data, aux.bytes) == 0 && close(aux.fd) == 0);
    std::puts("VQ source release PASS: seven cache/copy policies");
}
