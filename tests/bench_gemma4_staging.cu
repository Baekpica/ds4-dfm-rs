/* UNIT 7 PHASE A, step A2: the expert-record upload ceiling.
 *
 * Measures host-to-device of one Gemma-4 VQ expert record, the way the cache
 * lane moves it today and the ways PHASE A considers:
 *
 *   pageable  ds4_gpu_tensor_write today: cudaMemcpy H2D from the pageable
 *             model mmap, one call per record, blocking (ds4_cuda.cu
 *             ds4_gpu_tensor_write).
 *   pinned-sync   the same blocking call from a cudaHostAlloc buffer.
 *   pinned-async  one cudaMemcpyAsync per record on a private non-blocking
 *             stream, the host free to memcpy the next record into the ring
 *             while the copy runs, one cudaStreamSynchronize per expert
 *             (six records), which is the PHASE A1 shape.
 *
 * The default record is the 1.87 MiB the handoff measured for one expert's
 * six records (gate/up/down plus their residual planes) divided by six.
 *
 * The engine's measured 5.4 GiB/s (71.46 GiB in 13.24 s, the 4 GiB pool at
 * ctx 8192, 2026-11-05) is well under the pageable ceiling here, so --mmap
 * adds the other candidate: the source is the model's file mapping, and a
 * record read from it may pay page faults rather than bandwidth. It then
 * repeats the two pinned modes with the mmap as the source.
 *
 * Build: nvcc -O3 -arch=sm_89 -o tests/bench_gemma4_staging \
 *            tests/bench_gemma4_staging.cu
 * Run:   ./tests/bench_gemma4_staging [--record BYTES] [--experts N]
 *                                     [--rounds R] [--ring MIB]
 *                                     [--mmap FILE]
 */
#include <cuda_runtime.h>
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
    std::fprintf(stderr, "staging bench FAIL %d: %s\n", __LINE__, #x); \
    std::exit(1); \
} } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "staging bench CUDA FAIL %d: %s\n", __LINE__, \
                 cudaGetErrorString(e_)); std::exit(2); \
} } while (0)

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char **argv) {
    uint64_t record = 327 * 1024;   /* 1.87 MiB expert / 6 records */
    uint32_t experts = 128;
    uint32_t rounds = 4;
    uint64_t ring_mib = 256;
    const char *mmap_path = nullptr;
    const char *pread_path = nullptr;
    for (int i = 1; i + 1 < argc; i += 2) {
        const char *k = argv[i];
        const long v = std::strtol(argv[i + 1], nullptr, 10);
        if (!std::strcmp(k, "--record") && v > 0) { record = (uint64_t)v; }
        else if (!std::strcmp(k, "--experts") && v > 0) { experts = (uint32_t)v; }
        else if (!std::strcmp(k, "--rounds") && v > 0) { rounds = (uint32_t)v; }
        else if (!std::strcmp(k, "--ring") && v > 0) { ring_mib = (uint64_t)v; }
        else if (!std::strcmp(k, "--mmap")) { mmap_path = argv[i + 1]; }
        else if (!std::strcmp(k, "--pread")) { pread_path = argv[i + 1]; }
        else { std::fprintf(stderr, "staging bench: bad option %s\n", k); return 1; }
    }
    const uint32_t per_expert = 6u;      /* gate, up, down + three residuals */
    const uint64_t total = (uint64_t)experts * per_expert * record;
    const uint64_t ring = ring_mib << 20;

    int dev = 0, major = 0, minor = 0;
    CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, dev));
    CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    printf("gpu %s sm_%d%d, record %.2f MiB x %u per expert x %u experts = %.2f GiB\n",
           prop.name, major, minor, (double)record / 1048576.0, per_expert,
           experts, (double)total / 1073741824.0);

    /* The device side: one slot region the size of the whole run so every
     * record lands somewhere distinct, exactly as a slot pool would. */
    char *dev_buf = nullptr;
    CUDA(cudaMalloc(&dev_buf, (size_t)(record * 8)));

    /* Pageable host source, as the model mmap is today. */
    std::vector<unsigned char> pageable((size_t)record);
    for (uint64_t i = 0; i < record; i += 4096) { pageable[i] = (unsigned char)(i >> 12); }

    /* Pinned host source: the A1 ring. */
    unsigned char *pinned = nullptr;
    CUDA(cudaHostAlloc((void **)&pinned, (size_t)ring, cudaHostAllocDefault));
    std::memset(pinned, 0x5a, (size_t)ring);
    const uint32_t ring_slots = (uint32_t)(ring / record);
    CHECK(ring_slots >= 2);

    cudaStream_t up = nullptr;
    CUDA(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking));

    /* Optional: the model's own file mapping as the copy source. The
     * offsets spread over the whole file, as the router's records do, so a
     * fault or a cold page is not hidden by one hot 300 KB window. */
    int pread_fd = -1;
    uint64_t pread_bytes = 0;
    if (pread_path) {
        pread_fd = open(pread_path, O_RDONLY);
        CHECK(pread_fd >= 0);
        struct stat st{};
        CHECK(fstat(pread_fd, &st) == 0);
        pread_bytes = (uint64_t)st.st_size;
        CHECK(pread_bytes > record * 2u);
        printf("pread %s: %.2f GiB\n", pread_path,
               (double)pread_bytes / 1073741824.0);
    }
    const unsigned char *map = nullptr;
    uint64_t map_bytes = 0, map_stride = record;
    if (mmap_path) {
        const int fd = open(mmap_path, O_RDONLY);
        CHECK(fd >= 0);
        struct stat st{};
        CHECK(fstat(fd, &st) == 0);
        map_bytes = (uint64_t)st.st_size;
        map = (const unsigned char *)mmap(nullptr, (size_t)map_bytes, PROT_READ,
                                          MAP_SHARED, fd, 0);
        CHECK(map != MAP_FAILED);
        close(fd);
        const uint64_t records = (uint64_t)experts * per_expert;
        CHECK(map_bytes > record * 2u);
        map_stride = (map_bytes - record) / records;
        printf("mmap %s: %.2f GiB, stride %.2f MiB\n", mmap_path,
               (double)map_bytes / 1073741824.0, (double)map_stride / 1048576.0);
    }

    /* Warm the device and the driver's pageable bounce path. */
    CUDA(cudaMemcpy(dev_buf, pageable.data(), (size_t)record, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dev_buf, pinned, (size_t)record, cudaMemcpyHostToDevice));
    CUDA(cudaStreamSynchronize(0));

    double best[6] = {1e30, 1e30, 1e30, 1e30, 1e30, 1e30};
    for (uint32_t r = 0; r < rounds; r++) {
        /* 1. pageable, blocking, one call per record (today's path). */
        auto t0 = Clock::now();
        for (uint32_t e = 0; e < experts; e++) {
            for (uint32_t k = 0; k < per_expert; k++) {
                CUDA(cudaMemcpy(dev_buf + (uint64_t)k * record, pageable.data(),
                                (size_t)record, cudaMemcpyHostToDevice));
            }
        }
        const double pageable_s = seconds(t0, Clock::now());
        if (pageable_s < best[0]) { best[0] = pageable_s; }

        /* 2. pinned, blocking, one call per record. */
        t0 = Clock::now();
        for (uint32_t e = 0; e < experts; e++) {
            for (uint32_t k = 0; k < per_expert; k++) {
                CUDA(cudaMemcpy(dev_buf + (uint64_t)k * record, pinned,
                                (size_t)record, cudaMemcpyHostToDevice));
            }
        }
        const double pinned_sync_s = seconds(t0, Clock::now());
        if (pinned_sync_s < best[1]) { best[1] = pinned_sync_s; }

        /* 3. pinned ring, async, one sync per expert (PHASE A1). */
        t0 = Clock::now();
        uint32_t slot = 0;
        for (uint32_t e = 0; e < experts; e++) {
            for (uint32_t k = 0; k < per_expert; k++) {
                unsigned char *stage = pinned + (uint64_t)slot * record;
                std::memcpy(stage, pageable.data(), (size_t)record);
                CUDA(cudaMemcpyAsync(dev_buf + (uint64_t)k * record, stage,
                                     (size_t)record, cudaMemcpyHostToDevice, up));
                slot = (slot + 1u) % ring_slots;
            }
            CUDA(cudaStreamSynchronize(up));
        }
        const double pinned_async_s = seconds(t0, Clock::now());
        if (pinned_async_s < best[2]) { best[2] = pinned_async_s; }

        if (map) {
            /* 4. pageable file mapping, blocking (the MOE_HOST source). */
            t0 = Clock::now();
            for (uint32_t e = 0; e < experts; e++) {
                for (uint32_t k = 0; k < per_expert; k++) {
                    const unsigned char *src =
                        map + ((uint64_t)e * per_expert + k) * map_stride;
                    CUDA(cudaMemcpy(dev_buf + (uint64_t)k * record, src,
                                    (size_t)record, cudaMemcpyHostToDevice));
                }
            }
            const double map_sync_s = seconds(t0, Clock::now());
            if (map_sync_s < best[3]) { best[3] = map_sync_s; }

            /* 5. file mapping through the pinned ring, async (PHASE A1). */
            t0 = Clock::now();
            uint32_t slot = 0;
            for (uint32_t e = 0; e < experts; e++) {
                for (uint32_t k = 0; k < per_expert; k++) {
                    const unsigned char *src =
                        map + ((uint64_t)e * per_expert + k) * map_stride;
                    unsigned char *stage = pinned + (uint64_t)slot * record;
                    std::memcpy(stage, src, (size_t)record);
                    CUDA(cudaMemcpyAsync(dev_buf + (uint64_t)k * record, stage,
                                         (size_t)record, cudaMemcpyHostToDevice, up));
                    slot = (slot + 1u) % ring_slots;
                }
                CUDA(cudaStreamSynchronize(up));
            }
            const double map_async_s = seconds(t0, Clock::now());
            if (map_async_s < best[4]) { best[4] = map_async_s; }
        }
        if (pread_fd >= 0) {
            t0 = Clock::now();
            uint32_t slot = 0;
            const uint64_t stride =
                (pread_bytes - record) / ((uint64_t)experts * per_expert);
            for (uint32_t e = 0; e < experts; e++) {
                for (uint32_t k = 0; k < per_expert; k++) {
                    unsigned char *stage = pinned + (uint64_t)slot * record;
                    const off_t off = (off_t)(((uint64_t)e * per_expert + k) * stride);
                    ssize_t got = pread(pread_fd, stage, (size_t)record, off);
                    CHECK(got == (ssize_t)record);
                    CUDA(cudaMemcpyAsync(dev_buf + (uint64_t)k * record, stage,
                                         (size_t)record, cudaMemcpyHostToDevice, up));
                    slot = (slot + 1u) % ring_slots;
                }
                CUDA(cudaStreamSynchronize(up));
            }
            const double pread_s = seconds(t0, Clock::now());
            if (pread_s < best[5]) { best[5] = pread_s; }
        }
    }

    const char *names[6] = {"pageable blocking", "pinned blocking",
                            "pinned async + sync/expert",
                            "mmap blocking", "mmap->ring async + sync/expert",
                            "pread->ring async + sync/expert"};
    const int modes = pread_fd >= 0 ? 6 : (map ? 5 : 3);
    for (int m = 0; m < modes; m++) {
        printf("%-28s %.3f s  %6.2f GiB/s\n", names[m], best[m],
               (double)total / best[m] / 1073741824.0);
    }
    printf("ring %llu MiB, %u slots of %.2f MiB\n",
           (unsigned long long)ring_mib, ring_slots, (double)record / 1048576.0);

    CUDA(cudaStreamDestroy(up));
    CUDA(cudaFreeHost(pinned));
    CUDA(cudaFree(dev_buf));
    return 0;
}
