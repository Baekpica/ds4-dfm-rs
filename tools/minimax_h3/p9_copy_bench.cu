/* P9 evidence harness: host-source cost of one H2D weight copy.
 *
 * The reference uploads every weight from a malloc'd staging buffer filled by
 * `pread` (`h3_gpu_cuda.cu:454-478`) and then calls `cudaMemcpy`. The port's
 * three loaders (`minimax_h3/{weights,text_weights,vae_store}.rs`) instead
 * copy straight out of a read-only `mmap` of the shard. This probe isolates
 * that difference on one file: the same range is copied H2D once from an
 * mmap and once from an anonymous buffer filled by `pread`, with the same
 * copy API, and both are reported as MB/s over the range size.
 *
 * Usage: p9_copy_bench FILE MIB [async]
 *        FILE  shard to read (only the first MIB range is touched)
 *        MIB   range size
 *        async copy with cudaMemcpyAsync on a stream plus one sync, the
 *              port's call shape, instead of the reference's cudaMemcpy
 *
 * The range is copied twice per source so the report shows the first (cold
 * for this range) and second (warm) pass; page cache state is whatever the
 * caller left it in. Nothing here is part of the runtime.
 */

#include <cuda_runtime.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void fail(const char *what) {
    fprintf(stderr, "p9_copy_bench: %s: %s\n", what, cudaGetErrorString(cudaGetLastError()));
    exit(1);
}

static int copy(float *device, const void *host, size_t bytes, int async) {
    if (!async) {
        return cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    cudaStream_t stream = NULL;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 0;
    cudaError_t status = cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return status == cudaSuccess;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s FILE MIB [async]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    size_t bytes = (size_t)strtoull(argv[2], NULL, 10) << 20;
    int async = argc > 3 && strcmp(argv[3], "async") == 0;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat info;
    if (fstat(fd, &info) != 0 || (size_t)info.st_size < bytes) {
        fprintf(stderr, "file is smaller than the requested range\n");
        return 1;
    }

    void *mapped = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) { perror("mmap"); return 1; }
    void *staged = malloc(bytes);
    if (!staged) { fprintf(stderr, "out of memory\n"); return 1; }
    size_t done = 0;
    while (done < bytes) {
        ssize_t got = pread(fd, (char *)staged + done, bytes - done, (off_t)done);
        if (got <= 0) { fprintf(stderr, "short read\n"); return 1; }
        done += (size_t)got;
    }

    float *device = NULL;
    if (cudaMalloc((void **)&device, bytes) != cudaSuccess) fail("cudaMalloc");

    for (int pass = 0; pass < 2; pass++) {
        double started = seconds();
        if (!copy(device, mapped, bytes, async)) fail("mmap copy");
        double mapped_seconds = seconds() - started;
        started = seconds();
        if (!copy(device, staged, bytes, async)) fail("staged copy");
        double staged_seconds = seconds() - started;
        printf("PASS %d api=%s mib=%zu mmap_ms=%.1f mmap_mbs=%.0f "
               "staged_ms=%.1f staged_mbs=%.0f\n",
               pass + 1, async ? "async" : "sync", bytes >> 20,
               mapped_seconds * 1e3, (double)bytes / mapped_seconds / 1e6,
               staged_seconds * 1e3, (double)bytes / staged_seconds / 1e6);
        fflush(stdout);
    }

    cudaFree(device);
    free(staged);
    munmap(mapped, bytes);
    close(fd);
    return 0;
}
