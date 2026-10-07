/* Bounded startup-upload scout. Exact production bodies are emitted by
 * test_glm53_upload.py; policy/census publication is mocked, CUDA copies real.
 * No model, kernels, full GGUF mapping, host registration or inference. */
#include <cuda_runtime.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "upload scout FAIL %d: %s\n", __LINE__, #x); \
    std::exit(1); \
} } while (0)
#define CUDA(x) do { cudaError_t error = (x); if (error != cudaSuccess) { \
    std::fprintf(stderr, "upload scout CUDA FAIL %d: %s\n", __LINE__, \
                 cudaGetErrorString(error)); std::exit(2); \
} } while (0)

static constexpr uint64_t CHUNK = 64ull * 1024 * 1024;
static constexpr uint64_t BODY = 4 * CHUNK;
static constexpr uint64_t TAIL = 4093;
static constexpr uint64_t OFFSET = 4097;
static constexpr uint64_t GUARD = 64;
static constexpr uint64_t FILE_BYTES = OFFSET + BODY + TAIL + GUARD;
static constexpr unsigned char SENTINEL = 0xa5;
using Clock = std::chrono::steady_clock;
static double millis(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

enum { DS4_GOVC_ENGINE_BOOT, DS4_MEMC_WEIGHT_SPAN, DS4_MEMD_UNIFIED_DEVICE,
       DS4_GOV_ADMIT, DS4_MEMC_STAGE_PIN, DS4_MEMD_PINNED_HOST, DS4_MSRC_MAX = 1 };
struct ds4_gov_claim {
    int requester, memc, domain;
    uint64_t proposed_outstanding, operation_transient;
};
struct cuda_model_range {
    const void *host_base;
    uint64_t offset, bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered, arena_allocated, imported_ipc, imported_vmm;
    uint64_t vmm_handle, vmm_va, vmm_alloc_bytes;
};
static uint64_t g_model_range_bytes;
static int g_model_plan_frozen, g_in_governed_materialize;
static cudaStream_t g_model_upload_stream;
static void *g_model_stage_raw[4], *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes, g_mem_stage_slot_bytes[4];
static uint64_t g_model_direct_align = 4096;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static uint64_t g_model_registered_size = FILE_BYTES;
static struct { struct { uint64_t map_len; } v[1]; } g_model_srcs = {{{FILE_BYTES}}};
static uint64_t pinned_live, pinned_peak;
static uint64_t span_live;
static void *allocation;
static char *device;
static uint64_t device_bytes;
static Clock::time_point allocation_end;
static double alloc_ms, pin_ms, copy_api_ms;
static unsigned copy_calls, stage_calls;

static bool cuda_model_source_host_mapped(const void *) { return false; }
static const char *cuda_model_direct_fallback_ptr(const void *map, uint64_t offset) {
    return static_cast<const char *>(map) + offset;
}
static int ds4_gov_governed_check(const char *, const ds4_gov_claim *, int) {
    return DS4_GOV_ADMIT;
}
static int cuda_model_range_publish(cuda_model_range range) {
    CHECK(g_in_governed_materialize && range.device_ptr == device);
    return 1;
}
static void cuda_mem_note_alloc_src(int, int, uint64_t bytes, uint64_t,
                                     const void *) { span_live += bytes; }
static void ds4_gov_publish_use(int, uint64_t, uint64_t) {}
static void cuda_substrate_cover(const void *, uint64_t) {}
struct weight_env { int verbose; };
static weight_env cuda_weight_env() { return {0}; }
static int cuda_mem_src_index(const void *) { return 0; }
static uint64_t cuda_model_copy_chunk_bytes() { return CHUNK; }
static int cuda_model_stage_read(void *, uint64_t, uint64_t, uint64_t,
                                  const char **) {
    CHECK(false); /* This scout preserves source mode zero: mapped memcpy. */
    return 0;
}
static void cuda_model_drop_file_pages(uint64_t, uint64_t) { CHECK(false); }
static void cuda_model_discard_source_pages(const void *, uint64_t, uint64_t,
                                             uint64_t) { CHECK(false); }
static void cuda_model_load_progress_note(uint64_t) {}
static void cuda_mem_note_alloc(int cls, int domain, uint64_t alloc, uint64_t commit) {
    CHECK(cls == DS4_MEMC_STAGE_PIN && domain == DS4_MEMD_PINNED_HOST && alloc == commit);
    pinned_live += alloc;
    if (pinned_live > pinned_peak) { pinned_peak = pinned_live; }
}
static void cuda_mem_note_free(int cls, int domain, uint64_t alloc, uint64_t commit) {
    CHECK(cls == DS4_MEMC_STAGE_PIN && domain == DS4_MEMD_PINNED_HOST && alloc == commit);
    CHECK(pinned_live >= alloc);
    pinned_live -= alloc;
}

static cudaError_t fixture_alloc(void **out, size_t bytes) {
    const auto start = Clock::now();
    CHECK(!allocation);
    cudaError_t error = cudaMalloc(&allocation, bytes + 2 * GUARD);
    if (error != cudaSuccess) { return error; }
    device = static_cast<char *>(allocation) + GUARD;
    device_bytes = bytes;
    *out = device;
    CUDA(cudaMemset(allocation, SENTINEL, GUARD));
    CUDA(cudaMemset(device + bytes, SENTINEL, GUARD));
    CUDA(cudaDeviceSynchronize());
    allocation_end = Clock::now();
    alloc_ms += millis(start, allocation_end);
    return cudaSuccess;
}
static cudaError_t fixture_free(void *ptr) {
    CHECK(ptr == device);
    const cudaError_t error = cudaFree(allocation);
    allocation = nullptr;
    device = nullptr;
    device_bytes = 0;
    return error;
}
static cudaError_t fixture_pin(void **out, size_t bytes) {
    const auto start = Clock::now();
    const cudaError_t error = cudaMallocHost(out, bytes);
    pin_ms += millis(start, Clock::now());
    return error;
}
static cudaError_t fixture_copy(void *dst, const void *src, size_t bytes,
                                 cudaMemcpyKind kind) {
    const auto start = Clock::now();
    const cudaError_t error = cudaMemcpy(dst, src, bytes, kind);
    copy_api_ms += millis(start, Clock::now());
    ++copy_calls;
    return error;
}
static cudaError_t fixture_async(void *dst, const void *src, size_t bytes,
                                  cudaMemcpyKind kind, cudaStream_t stream) {
    const auto start = Clock::now();
    const cudaError_t error = cudaMemcpyAsync(dst, src, bytes, kind, stream);
    copy_api_ms += millis(start, Clock::now());
    ++stage_calls;
    return error;
}

#define cudaMallocHost fixture_pin
#define cudaMemcpyAsync fixture_async
#include "glm53_upload_stage.inc"
#undef cudaMemcpyAsync
#undef cudaMallocHost
#define cudaMalloc fixture_alloc
#define cudaFree fixture_free
#define cudaMemcpy fixture_copy
#include "glm53_upload_range.inc"
#undef cudaMemcpy
#undef cudaFree
#undef cudaMalloc

static unsigned char pattern(uint64_t index) {
    return static_cast<unsigned char>((index * 17u) ^ (index >> 8) ^ 0x5du);
}

static void prepare(const char *path) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    CHECK(fd >= 0);
    std::vector<unsigned char> buffer(1024 * 1024);
    for (uint64_t offset = 0; offset < FILE_BYTES;) {
        const size_t bytes = FILE_BYTES - offset < buffer.size()
            ? size_t(FILE_BYTES - offset) : buffer.size();
        for (size_t i = 0; i < bytes; ++i) { buffer[i] = pattern(offset + i); }
        size_t done = 0;
        while (done < bytes) {
            const ssize_t wrote = write(fd, buffer.data() + done, bytes - done);
            if (wrote < 0 && errno == EINTR) { continue; }
            CHECK(wrote > 0);
            done += size_t(wrote);
        }
        offset += bytes;
    }
    CHECK(fsync(fd) == 0 && close(fd) == 0);
    std::printf("prepared bytes=%llu offset=%llu chunk=%llu tail=%llu\n",
                (unsigned long long)FILE_BYTES, (unsigned long long)OFFSET,
                (unsigned long long)CHUNK, (unsigned long long)TAIL);
}

static uint64_t resident_pages(void *map, uint64_t bytes, uint64_t page) {
    std::vector<unsigned char> state((bytes + page - 1) / page);
    CHECK(mincore(map, bytes, state.data()) == 0);
    uint64_t count = 0;
    for (unsigned char one : state) { count += (one & 1u); }
    return count;
}

static void verify(uint64_t bytes) {
    unsigned char guard[2 * GUARD];
    CUDA(cudaMemcpy(guard, allocation, GUARD, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(guard + GUARD, device + bytes, GUARD, cudaMemcpyDeviceToHost));
    for (unsigned char one : guard) { CHECK(one == SENTINEL); }
    std::vector<unsigned char> buffer(1024 * 1024);
    for (uint64_t offset = 0; offset < bytes;) {
        const size_t take = bytes - offset < buffer.size()
            ? size_t(bytes - offset) : buffer.size();
        CUDA(cudaMemcpy(buffer.data(), device + offset, take, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < take; ++i) { CHECK(buffer[i] == pattern(OFFSET + offset + i)); }
        offset += take;
    }
}

int main(int argc, char **argv) {
    CHECK(argc >= 3);
    if (std::strcmp(argv[1], "prepare") == 0) {
        CHECK(argc == 3);
        prepare(argv[2]);
        return 0; /* File preparation invokes no CUDA API. */
    }
    CHECK(argc == 6);
    const bool staged = std::strcmp(argv[1], "pinned") == 0;
    CHECK(staged || std::strcmp(argv[1], "pageable") == 0);
    const bool warm = std::strcmp(argv[3], "warm") == 0;
    CHECK(warm || std::strcmp(argv[3], "advised") == 0);
    CHECK(std::strcmp(argv[4], "0") == 0 || std::strcmp(argv[4], "4093") == 0);
    CHECK(std::strlen(argv[5]) == 1 && argv[5][0] >= '1' && argv[5][0] <= '5');
    const uint64_t bytes = BODY + (argv[4][0] == '0' ? 0 : TAIL);
    const unsigned samples = unsigned(argv[5][0] - '0');
    const int fd = open(argv[2], O_RDONLY);
    CHECK(fd >= 0);
    struct stat info;
    CHECK(fstat(fd, &info) == 0 && uint64_t(info.st_size) == FILE_BYTES);
    void *map = mmap(nullptr, FILE_BYTES, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(map != MAP_FAILED);
    const long raw_page = sysconf(_SC_PAGESIZE);
    CHECK(raw_page > 0);
    const uint64_t page = uint64_t(raw_page);
    CUDA(cudaSetDevice(0));
    CUDA(cudaFree(nullptr));
    cudaDeviceProp prop;
    CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("gpu=%s sm=%d%d file_dev=%llu file_ino=%llu bytes=%llu "
                "offset=%llu chunk=%llu source=mmap direct_discard=0 cache=%s "
                "policy=census_mocks\n", prop.name, prop.major, prop.minor,
                (unsigned long long)info.st_dev, (unsigned long long)info.st_ino,
                (unsigned long long)bytes, (unsigned long long)OFFSET,
                (unsigned long long)CHUNK, argv[3]);
    std::puts("mode,sample,total_ms,alloc_ms,upload_ms,pin_alloc_ms,copy_api_ms,"
              "copy_calls,async_calls,resident_pages_before,resident_pages_after,pinned_live,parity");

    for (unsigned sample = 0; sample < samples; ++sample) {
        if (warm) {
            volatile unsigned sum = 0;
            const auto *source = static_cast<const unsigned char *>(map);
            for (uint64_t i = 0; i < FILE_BYTES; i += page) { sum += source[i]; }
            (void)sum;
        } else {
            CHECK(madvise(map, FILE_BYTES, MADV_DONTNEED) == 0);
            CHECK(posix_fadvise(fd, 0, FILE_BYTES, POSIX_FADV_DONTNEED) == 0);
        }
        const uint64_t before = resident_pages(map, FILE_BYTES, page);
        alloc_ms = pin_ms = copy_api_ms = 0;
        copy_calls = stage_calls = 0;
        g_model_range_bytes = span_live = 0;
        const auto start = Clock::now();
        if (staged) {
            void *out = nullptr;
            CUDA(fixture_alloc(&out, bytes));
            CHECK(cuda_stage_copy_to_dev(map, OFFSET, bytes,
                                        static_cast<char *>(out), 0, "scout"));
        } else {
            CHECK(cuda_model_range_populate_device_copy(map, OFFSET, bytes, "scout") == device);
            CHECK(span_live == bytes && g_model_range_bytes == bytes);
        }
        const auto end = Clock::now();
        const uint64_t after = resident_pages(map, FILE_BYTES, page);
        CHECK(device_bytes == bytes);
        CHECK(copy_calls == (staged ? 0u : (bytes + CHUNK - 1) / CHUNK));
        CHECK(stage_calls == (staged ? (bytes + CHUNK - 1) / CHUNK : 0u));
        verify(bytes); /* Readback/reference work is outside the copy timing. */
        std::printf("%s,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%u,%u,%llu,%llu,%llu,PASS\n",
                    argv[1], sample, millis(start, end), alloc_ms,
                    millis(allocation_end, end), pin_ms, copy_api_ms,
                    copy_calls, stage_calls, (unsigned long long)before,
                    (unsigned long long)after, (unsigned long long)pinned_live);
        std::fflush(stdout);
        CUDA(fixture_free(device));
    }
    cuda_model_stage_pool_release();
    CHECK(pinned_live == 0);
    if (g_model_upload_stream) { CUDA(cudaStreamDestroy(g_model_upload_stream)); }
    CHECK(munmap(map, FILE_BYTES) == 0 && close(fd) == 0);
    std::printf("cleanup pinned_live=0 pinned_peak=%llu guards=PASS tail=PASS\n",
                (unsigned long long)pinned_peak);
}
