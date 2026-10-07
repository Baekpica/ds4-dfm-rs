/* Isolated canonical expert upload, using the production blocking tensor API.
 * Eight raw slots preserve unit/stride shape; no model execution is created.
 * This stable-file benchmark does not prove mmap's truncation error contract. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4_gpu.h"
#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <thread>
#include <unistd.h>
#include <vector>

enum { SLOTS = 8, GUARD = 64, CYCLES = 56, UNIT_COUNT = 54,
       MAX_UNIT = 2752512, SENTINEL = 0x5a };
enum class Arm { Pread, Mmap, Pread3 };
struct Unit { uint32_t layer, expert, part, type; uint64_t offset, bytes; };
struct ReadResult {
    int ok = 0, error = 0;
    uint64_t bytes = 0, calls = 0, interrupted = 0, partial = 0;
};
struct Plan {
    uint64_t bytes, device, inode, mtime, ctime, gate_stride, down_stride;
    uint32_t count, slots, staging;
    std::vector<Unit> units;
};

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "GLM stream FAIL %d: %s errno=%d\n", __LINE__, #x, errno); \
    std::exit(1); \
} } while (0)

static double now() {
    timespec t;
    CHECK(!clock_gettime(CLOCK_MONOTONIC, &t));
    return double(t.tv_sec) + double(t.tv_nsec) * 1.0e-9;
}

static uint64_t read_le(FILE *fp, unsigned width) {
    uint64_t value = 0;
    for (unsigned i = 0; i < width; ++i) {
        const int byte = std::fgetc(fp);
        CHECK(byte != EOF);
        value |= uint64_t(byte) << (i * 8);
    }
    return value;
}

static Plan load_plan(const char *path) {
    FILE *fp = std::fopen(path, "rb");
    CHECK(fp);
    char magic[8];
    CHECK(std::fread(magic, 1, sizeof(magic), fp) == sizeof(magic) &&
          !std::memcmp(magic, "GLMIO01\0", sizeof(magic)));
    Plan p;
    p.bytes = read_le(fp, 8); p.device = read_le(fp, 8); p.inode = read_le(fp, 8);
    p.mtime = read_le(fp, 8); p.ctime = read_le(fp, 8);
    p.gate_stride = read_le(fp, 8); p.down_stride = read_le(fp, 8);
    p.count = uint32_t(read_le(fp, 4)); p.slots = uint32_t(read_le(fp, 4));
    p.staging = uint32_t(read_le(fp, 4));
    CHECK(p.count == UNIT_COUNT && p.slots == SLOTS && p.staging == MAX_UNIT);
    CHECK(p.gate_stride == 2424906 && p.down_stride == 2753688);
    const uint32_t layers[6] = {3, 4, 5, 43, 44, 17}, experts[3] = {0, 143, 287};
    for (uint32_t i = 0; i < p.count; ++i) {
        Unit u;
        u.layer = uint32_t(read_le(fp, 4)); u.expert = uint32_t(read_le(fp, 4));
        u.part = uint32_t(read_le(fp, 4)); u.type = uint32_t(read_le(fp, 4));
        u.offset = read_le(fp, 8); u.bytes = read_le(fp, 8);
        CHECK(u.layer == layers[i / 9] && u.expert == experts[(i / 3) % 3] && u.part == i % 3);
        const uint32_t kind = u.layer == 17 ? (u.part < 2 ? 16 : 17) : (u.part < 2 ? 17 : 10);
        const uint64_t block = kind == 16 ? 66 : kind == 17 ? 74 : 84;
        CHECK(u.type == kind && u.bytes == uint64_t(4096) * 2048 / 256 * block &&
              u.bytes <= p.staging && u.offset <= p.bytes && u.bytes <= p.bytes - u.offset);
        p.units.push_back(u);
    }
    CHECK(std::fgetc(fp) == EOF && !std::ferror(fp) && !std::fclose(fp));
    return p;
}

static int same_stat(const Plan &p, const struct stat &s) {
    const uint64_t mtime = uint64_t(s.st_mtim.tv_sec) * 1000000000u + s.st_mtim.tv_nsec;
    const uint64_t ctime = uint64_t(s.st_ctim.tv_sec) * 1000000000u + s.st_ctim.tv_nsec;
    return uint64_t(s.st_size) == p.bytes && uint64_t(s.st_dev) == p.device &&
           uint64_t(s.st_ino) == p.inode && mtime == p.mtime && ctime == p.ctime;
}

static ReadResult pread_unit(int fd, unsigned char *buf, const Unit &u) {
    ReadResult result;
    while (result.bytes < u.bytes) {
        const uint64_t left = u.bytes - result.bytes;
        const ssize_t n = pread(fd, buf + result.bytes, size_t(left), off_t(u.offset + result.bytes));
        result.calls++;
        if (n < 0 && errno == EINTR) { result.interrupted++; continue; }
        if (n <= 0) { result.error = n ? errno : EIO; return result; }
        if (uint64_t(n) < left) { result.partial++; }
        result.bytes += uint64_t(n);
    }
    result.ok = 1;
    return result;
}

/* Three persistent reader threads approximate the native shared pool's
 * mechanics. They are independent threads, not ds4_parallel_for_min_rows;
 * the main thread waits and all GPU uploads remain serial and blocking. */
class ReadPool {
    struct Job { const Unit *unit; unsigned char *buffer; ReadResult result; };
    int fd;
    std::mutex mutex;
    std::condition_variable ready, complete;
    std::thread workers[3];
    Job jobs[3] = {};
    uint64_t epoch = 0;
    unsigned pending = 0;
    bool stop = false;

    void worker(unsigned part) {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            ready.wait(lock, [&] { return stop || epoch != seen; });
            if (stop) { return; }
            seen = epoch;
            const Unit *unit = jobs[part].unit;
            unsigned char *buffer = jobs[part].buffer;
            lock.unlock();
            const ReadResult result = pread_unit(fd, buffer, *unit);
            lock.lock();
            jobs[part].result = result;
            if (!--pending) { complete.notify_one(); }
        }
    }

public:
    explicit ReadPool(int source) : fd(source) {
        for (unsigned part = 0; part < 3; ++part) {
            workers[part] = std::thread(&ReadPool::worker, this, part);
        }
    }

    ~ReadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        ready.notify_all();
        for (auto &thread : workers) { thread.join(); }
    }

    void read(const Unit *units, unsigned char **buffers, ReadResult *results) {
        std::unique_lock<std::mutex> lock(mutex);
        CHECK(!pending);
        for (unsigned part = 0; part < 3; ++part) {
            jobs[part] = {units + part, buffers[part], {}};
        }
        pending = 3;
        epoch++;
        ready.notify_all();
        complete.wait(lock, [&] { return !pending; });
        for (unsigned part = 0; part < 3; ++part) { results[part] = jobs[part].result; }
    }
};

static void read_check(const ReadResult &result, const Unit &u) {
    if (!result.ok || result.bytes != u.bytes) {
        std::fprintf(stderr, "GLM stream read failed layer=%u expert=%u part=%u bytes=%" PRIu64
            "/%" PRIu64 " error=%d calls=%" PRIu64 " EINTR=%" PRIu64 " partial=%" PRIu64 "\n",
            u.layer, u.expert, u.part, result.bytes, u.bytes, result.error,
            result.calls, result.interrupted, result.partial);
        std::exit(1);
    }
}

static uint32_t crc32(const unsigned char *data, size_t bytes) {
    static uint32_t table[256];
    static int ready = 0;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (unsigned j = 0; j < 8; ++j) { c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u))); }
            table[i] = c;
        }
        ready = 1;
    }
    uint32_t c = UINT32_MAX;
    for (size_t i = 0; i < bytes; ++i) { c = table[(c ^ data[i]) & 255u] ^ (c >> 8); }
    return ~c;
}

static void sentinel(ds4_gpu_tensor *tensor) {
    const uint32_t bits = 0x5a5a5a5au;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    CHECK(ds4_gpu_tensor_fill_f32(tensor, value, ds4_gpu_tensor_bytes(tensor) / sizeof(float)));
}

static void check_gap(ds4_gpu_tensor *tensor, uint64_t off, uint64_t bytes,
                      unsigned char *buffer) {
    CHECK(bytes <= MAX_UNIT && ds4_gpu_tensor_read(tensor, off, buffer, bytes));
    for (uint64_t i = 0; i < bytes; ++i) { CHECK(buffer[i] == SENTINEL); }
}

int main(int argc, char **argv) {
    if ((argc != 4 && argc != 5) ||
        (std::strcmp(argv[3], "pread") && std::strcmp(argv[3], "mmap") &&
         std::strcmp(argv[3], "pread3"))) {
        std::fprintf(stderr, "usage: %s MODEL PLAN.bin pread|mmap|pread3 [CYCLES]\n", argv[0]);
        return 2;
    }
    const Arm arm = !std::strcmp(argv[3], "pread") ? Arm::Pread :
                    !std::strcmp(argv[3], "mmap") ? Arm::Mmap : Arm::Pread3;
    unsigned cycles = CYCLES;
    if (argc == 5) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(argv[4], &end, 10);
        CHECK(end && !*end && parsed > 0 && parsed <= 1024);
        cycles = unsigned(parsed);
    }
    const Plan p = load_plan(argv[2]);
    const int fd = open(argv[1], O_RDONLY);
    struct stat before;
    CHECK(fd >= 0 && !fstat(fd, &before) && same_stat(p, before));
    // Full virtual mapping, bounded touched bytes; no anonymous weight copy.
    const auto *map = static_cast<const unsigned char *>(mmap(nullptr, size_t(p.bytes),
                      PROT_READ, MAP_PRIVATE, fd, 0));
    CHECK(map != MAP_FAILED);
    auto *staging = static_cast<unsigned char *>(std::malloc(p.staging));
    CHECK(staging && ds4_gpu_init());
    unsigned char *parallel[3] = {nullptr, nullptr, staging};
    const uint64_t staging_bytes = arm == Arm::Pread3 ? UINT64_C(2424832) * 2 + p.staging : p.staging;
    std::unique_ptr<ReadPool> readers;
    if (arm == Arm::Pread3) {
        for (unsigned part = 0; part < 2; ++part) {
            parallel[part] = static_cast<unsigned char *>(std::malloc(2424832));
            CHECK(parallel[part]);
        }
        readers.reset(new ReadPool(fd));
    }
    const uint64_t strides[3] = {p.gate_stride, p.gate_stride, p.down_stride};
    ds4_gpu_tensor *pool[3];
    uint64_t pool_bytes = 0;
    for (unsigned part = 0; part < 3; ++part) {
        const uint64_t bytes = SLOTS * strides[part] + 2 * GUARD;
        pool[part] = ds4_gpu_tensor_alloc(bytes);
        CHECK(pool[part]);
        sentinel(pool[part]);
        pool_bytes += bytes;
    }
    CHECK(pool_bytes <= (UINT64_C(64) << 20) && ds4_gpu_synchronize());
    // Identical pread warmup makes both arms hot-page experiments.
    for (const Unit &u : p.units) { read_check(pread_unit(fd, staging, u), u); }
    const Unit *last[3][SLOTS] = {};
    double read_s = 0.0, copy_s = 0.0;
    uint64_t bytes = 0, calls = 0;
    uint64_t read_calls = 0, interrupted = 0, partial = 0;
    std::fprintf(stderr, "GLM stream begin arm=%s cycles=%u staging=%" PRIu64 " pool=%" PRIu64 "\n",
                 argv[3], cycles, staging_bytes, pool_bytes);
    const double start = now();
    for (unsigned cycle = 0; cycle < cycles; ++cycle) {
        for (size_t i = 0; i < p.units.size(); ++i) {
            const Unit &u = p.units[i];
            const unsigned slot = unsigned(i / 3) % SLOTS;
            const unsigned char *source = map + u.offset;
            if (arm == Arm::Pread3 && u.part == 0) {
                ReadResult results[3];
                const double at = now();
                readers->read(p.units.data() + i, parallel, results);
                read_s += now() - at;
                // Check every read before the first upload in the triplet.
                for (unsigned part = 0; part < 3; ++part) {
                    read_check(results[part], p.units[i + part]);
                    read_calls += results[part].calls;
                    interrupted += results[part].interrupted;
                    partial += results[part].partial;
                }
            }
            if (arm == Arm::Pread) {
                const double at = now();
                const ReadResult result = pread_unit(fd, staging, u);
                read_s += now() - at;
                read_check(result, u);
                read_calls += result.calls;
                interrupted += result.interrupted;
                partial += result.partial;
                source = staging;
            }
            if (arm == Arm::Pread3) { source = parallel[u.part]; }
            const double at = now();
            CHECK(ds4_gpu_tensor_write(pool[u.part], GUARD + slot * strides[u.part], source, u.bytes));
            copy_s += now() - at;
            bytes += u.bytes;
            calls++;
            last[u.part][slot] = &u;
        }
    }
    CHECK(ds4_gpu_synchronize());
    const double wall_s = now() - start;
    struct stat after;
    CHECK(!fstat(fd, &after) && same_stat(p, after));
    // Byte proof and CRC stay outside the measured upload interval.
    for (unsigned part = 0; part < 3; ++part) {
        check_gap(pool[part], 0, GUARD, staging);
        check_gap(pool[part], GUARD + SLOTS * strides[part], GUARD, staging);
        const uint64_t max_unit = part < 2 ? UINT64_C(2424832) : uint64_t(MAX_UNIT);
        for (unsigned slot = 0; slot < SLOTS; ++slot) {
            const Unit *u = last[part][slot];
            CHECK(u && ds4_gpu_tensor_read(pool[part], GUARD + slot * strides[part], staging, u->bytes));
            CHECK(!std::memcmp(staging, map + u->offset, size_t(u->bytes)));
            std::printf("unit part=%u slot=%u layer=%u expert=%u bytes=%" PRIu64 " crc32=%08x exact=1\n",
                part, slot, u->layer, u->expert, u->bytes, crc32(staging, size_t(u->bytes)));
            check_gap(pool[part], GUARD + slot * strides[part] + max_unit,
                      strides[part] - max_unit, staging);
        }
    }
    CHECK(!fstat(fd, &after) && same_stat(p, after));
    std::printf("{\"arm\":\"%s\",\"cycles\":%u,\"calls\":%" PRIu64 ",\"bytes\":%" PRIu64
        ",\"cpu_wall_s\":%.9f,\"pread_wall_s\":%.9f,\"blocking_copy_wall_s\":%.9f,"
        "\"pool_bytes\":%" PRIu64 ",\"staging_bytes\":%" PRIu64 ",\"read_syscalls\":%" PRIu64
        ",\"EINTR_retries\":%" PRIu64 ",\"partial_reads\":%" PRIu64 ",\"final_raw_byte_exact\":true,"
        "\"sentinel_gaps_exact\":true,\"hot_page_warmup\":true,\"math\":false}\n",
        argv[3], cycles, calls, bytes, wall_s, read_s, copy_s, pool_bytes, staging_bytes,
        read_calls, interrupted, partial);
    for (auto *tensor : pool) { ds4_gpu_tensor_free(tensor); }
    ds4_gpu_cleanup();
    readers.reset();
    for (unsigned part = 0; part < 2; ++part) { std::free(parallel[part]); }
    std::free(staging);
    CHECK(!munmap(const_cast<unsigned char *>(map), size_t(p.bytes)) && !close(fd));
    return 0;
}
