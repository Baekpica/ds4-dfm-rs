#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "../cuda/mmq/ds4_repack.h"
#include "../ds4_weight_identity.h" /* rider #48: manifest content fingerprint */

/* File mapping, the GGUF tensor catalog, and the aligned repack builders
 * live in the shared layout library (cuda/mmq/ds4_repack.{h,cu}) so the
 * engine's self-load boot builds bit-identical artifacts.  Local aliases
 * keep the historical names readable here. */
typedef ds4_repack_file mapped_file;
typedef ds4_repack_span tensor_span;
typedef ds4_repack_tensor tensor_record;

enum derived_kind {
    DERIVED_NONE = 0,
    DERIVED_Q8_0_ROW_GROUP_NORMS = 1,
    DERIVED_Q8_0_F16_COLMAJOR = 2,
    DERIVED_Q8_0_F32_COLMAJOR = 3,
    /* Aligned-SoA IQ2_XXS routed-expert repack (--repack-iq2-aligned).
     * Byte-neutral: [__half dq[nblk]][pad to 64B][uint2 qs[nblk*8]], block
     * order identical to the raw tensor byte order.  Layout contract shared
     * with ds4_mmq_iq2_xxs_aligned_moe_vec (cuda/mmq/ds4_mmq.h).  Unlike the
     * other derived kinds this REPLACES the raw upload of its source tensor,
     * so it does not count against --derive-budget-gb. */
    DERIVED_IQ2_XXS_ALIGNED_MOE = 4,
    DERIVED_IQ2_XS_ALIGNED_MOE = 8,
    /* Aligned-SoA Q8_0 dense repack (--repack-q8-aligned): [__half dq[nblk]]
     * [pad to 64B][int8 qs[nblk*32]], nblk = out_dim * (in_dim/32), block
     * order identical to the raw tensor byte order.  Layout contract shared
     * with ds4_mmq_q8_0_aligned_dense_vec (cuda/mmq/ds4_mmq.h).  Unlike the
     * IQ2 expert repack, the raw spans stay SERVED (dense candidates total
     * ~6 GiB, affordable to duplicate), so all other consumers (prefill MMQ,
     * cublas f16, batched vec) are untouched. */
    DERIVED_Q8_0_ALIGNED_DENSE = 5,
    /* Aligned row-pair-SoA Q2_K routed-expert repack (--repack-q2k-aligned):
     * the .ffn_down_exps stacks.  Byte-neutral: with npair = nblk/2 raw-order
     * row-pair blocks (rows 2p, 2p+1 of an expert),
     *   [uint2 dm2[npair]][pad64][int4 sc4[npair*2]][pad64][uint2 qs2[npair*16]]
     * Layout contract shared with ds4_mmq_q2_K_aligned_moe_vec
     * (cuda/mmq/ds4_mmq.h).  Like the IQ2 kind this REPLACES the raw upload
     * of its source tensor (~30 GiB would not fit twice), so it does not
     * count against --derive-budget-gb. */
    DERIVED_Q2_K_ALIGNED_MOE = 6,
    /* Motif-3 MLA W_UV Q8_0 transposed across value rows.  ADDITIVE: raw
     * kv_b remains served for the absorbed-key projection. */
    DERIVED_MOTIF3_KV_B_VALUE_Q8_0 = 7,
};

enum weight_backend {
    WEIGHT_BACKEND_IPC = 0,
    WEIGHT_BACKEND_VMM = 1,
};

struct owned_range {
    weight_backend backend = WEIGHT_BACKEND_IPC;
    bool derived = false;
    std::string model_id;
    uint64_t model_size = 0;
    uint64_t off = 0;
    uint64_t bytes = 0;
    uint64_t alloc_bytes = 0;
    uint32_t derived_kind = DERIVED_NONE;
    uint64_t source_off = 0;
    uint64_t source_bytes = 0;
    uint64_t in_dim = 0;
    uint64_t out_dim = 0;
    uint32_t group_count = 0;
    std::string source_name;
    void *dev = nullptr;
    cudaIpcMemHandle_t handle{};
    CUmemGenericAllocationHandle vmm_handle{};
    CUdeviceptr vmm_va = 0;
    int exported_fd = -1;
};

struct vmm_support {
    int vmm = 0;
    int posix_fd = 0;
    int uva = 0;
    size_t granularity_min = 0;
    size_t granularity_recommended = 0;
};

struct fd_broker {
    int listen_fd = -1;
    std::string path;
    uint64_t requests = 0;
};

static volatile sig_atomic_t g_stop;

static void on_signal(int) {
    g_stop = 1;
}

static bool parse_scope(const char *scope, bool &want_base, bool &want_mtp) {
    if (!scope || !scope[0] || !strcmp(scope, "both")) {
        want_base = true;
        want_mtp = true;
        return true;
    }
    if (!strcmp(scope, "base")) {
        want_base = true;
        want_mtp = false;
        return true;
    }
    if (!strcmp(scope, "mtp")) {
        want_base = false;
        want_mtp = true;
        return true;
    }
    return false;
}

static bool parse_backend(const char *backend, weight_backend &out) {
    if (!backend || !backend[0] || !strcmp(backend, "ipc")) {
        out = WEIGHT_BACKEND_IPC;
        return true;
    }
    if (!strcmp(backend, "vmm")) {
        out = WEIGHT_BACKEND_VMM;
        return true;
    }
    return false;
}

static const char *backend_name(weight_backend backend) {
    return backend == WEIGHT_BACKEND_VMM ? "vmm" : "ipc";
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static bool parent_pid_alive(pid_t pid) {
    if (pid <= 0) return true;
    if (kill(pid, 0) == 0) return true;
    return errno != ESRCH;
}

static void driver_error_name(CUresult result, const char **name, const char **text) {
    *name = nullptr;
    *text = nullptr;
    (void)cuGetErrorName(result, name);
    (void)cuGetErrorString(result, text);
}

static bool driver_ok(CUresult result, const char *what) {
    if (result == CUDA_SUCCESS) return true;
    const char *name = nullptr;
    const char *text = nullptr;
    driver_error_name(result, &name, &text);
    fprintf(stderr, "ds4_weight_server: CUDA driver %s failed: %s%s%s\n",
            what,
            name ? name : "unknown",
            text ? ": " : "",
            text ? text : "");
    return false;
}

static CUmemAllocationProp vmm_allocation_prop(int device) {
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    return prop;
}

static bool query_vmm_support(int device, vmm_support &support) {
    CUdevice cu_device;
    if (!driver_ok(cuDeviceGet(&cu_device, device), "device get")) return false;
    if (!driver_ok(cuDeviceGetAttribute(&support.vmm,
                                        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                        cu_device),
                   "query VMM support")) return false;
    if (!driver_ok(cuDeviceGetAttribute(&support.posix_fd,
                                        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
                                        cu_device),
                   "query POSIX FD handle support")) return false;
    if (!driver_ok(cuDeviceGetAttribute(&support.uva,
                                        CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                                        cu_device),
                   "query UVA support")) return false;
    CUmemAllocationProp prop = vmm_allocation_prop(device);
    if (!driver_ok(cuMemGetAllocationGranularity(&support.granularity_min,
                                                 &prop,
                                                 CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                   "query minimum VMM granularity")) return false;
    if (!driver_ok(cuMemGetAllocationGranularity(&support.granularity_recommended,
                                                 &prop,
                                                 CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                   "query recommended VMM granularity")) return false;
    fprintf(stderr,
            "ds4_weight_server: vmm support vmm=%d posix_fd=%d uva=%d gran_min=%zu gran_rec=%zu\n",
            support.vmm,
            support.posix_fd,
            support.uva,
            support.granularity_min,
            support.granularity_recommended);
    return support.vmm && support.posix_fd && support.uva && support.granularity_min != 0;
}

static int acquire_owner_lock(const char *path) {
    if (!path || !path[0]) return -1;
    int fd = open(path, O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4_weight_server: lock open failed %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        char buf[128] = {0};
        ssize_t nread = pread(fd, buf, sizeof(buf) - 1u, 0);
        if (nread < 0) buf[0] = '\0';
        fprintf(stderr,
                "ds4_weight_server: another weight server owns lock %s%s%s\n",
                path,
                buf[0] ? " pid=" : "",
                buf[0] ? buf : "");
        close(fd);
        return -1;
    }
    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4_weight_server: lock truncate failed %s: %s\n", path, strerror(errno));
        close(fd);
        return -1;
    }
    if (dprintf(fd, "%ld\n", (long)getpid()) < 0) {
        fprintf(stderr, "ds4_weight_server: lock write failed %s: %s\n", path, strerror(errno));
        close(fd);
        return -1;
    }
    fprintf(stderr, "ds4_weight_server: acquired lock %s\n", path);
    return fd;
}

static uint64_t align_up(uint64_t v, uint64_t a) {
    if (a <= 1) return v;
    const uint64_t r = v % a;
    return r ? v + (a - r) : v;
}

static void *align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    const uintptr_t p = (uintptr_t)ptr;
    const uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static uint64_t sum_rounded_ranges(const std::vector<tensor_span> &ranges, uint64_t granularity) {
    uint64_t total = 0;
    for (const tensor_span &r : ranges) {
        if (r.end < r.off) continue;
        const uint64_t logical = r.end - r.off;
        const uint64_t rounded = align_up(logical, granularity);
        if (UINT64_MAX - total < rounded) return UINT64_MAX;
        total += rounded;
    }
    return total;
}

__device__ static float warp_sum_f32(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__global__ static void q8_0_row_group_norms_warp_kernel(
        float *row_group_norms,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t group_count) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint32_t group = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || group >= group_count) return;

    const uint64_t group_start = ((uint64_t)group * in_dim) / group_count;
    const uint64_t group_end = ((uint64_t)(group + 1u) * in_dim) / group_count;
    const uint64_t block_start = group_start / 32u;
    const uint64_t block_end = (group_end + 31u) / 32u;
    const unsigned char *wr = w + row * blocks * 34u;
    float sum = 0.0f;
    for (uint64_t b = block_start; b < block_end; b++) {
        const uint64_t i0 = b * 32u;
        const uint64_t lo = group_start > i0 ? group_start - i0 : 0u;
        const uint64_t hi = group_end < i0 + 32u ? group_end - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const float scale = __half2float(*scale_h);
        for (uint64_t i = lo + lane; i < hi; i += 32u) {
            const float v = scale * (float)qs[i];
            sum += v * v;
        }
    }
    sum = warp_sum_f32(sum);
    if (lane == 0) row_group_norms[row * group_count + group] = sqrtf(sum);
}

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    const uint64_t row = gid / in_dim;
    const uint64_t i = gid - row * in_dim;
    const uint64_t b = i / 32u;
    const uint64_t j = i - b * 32u;
    const unsigned char *blk = w + (row * blocks + b) * 34u;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2u + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    const uint64_t row = gid / in_dim;
    const uint64_t i = gid - row * in_dim;
    const uint64_t b = i / 32u;
    const uint64_t j = i - b * 32u;
    const unsigned char *blk = w + (row * blocks + b) * 34u;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2u + j);
    out[gid] = scale * (float)q;
}

/* The aligned repack kernels (iq2/q2k/q8) live in cuda/mmq/ds4_repack.cu. */

static bool map_file(const char *path, mapped_file &m) {
    return ds4_repack_map_file("ds4_weight_server", path, m);
}

static void unmap_file(mapped_file &m) {
    ds4_repack_unmap_file(m);
}

static bool collect_tensor_catalog(const mapped_file &m,
                                   std::vector<tensor_span> &spans,
                                   std::vector<tensor_record> *records) {
    return ds4_repack_collect_catalog("ds4_weight_server", m, &spans, records);
}

static uint64_t parse_mib(const char *s, uint64_t fallback) {
    if (!s || !s[0]) return fallback;
    char *end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s || v == 0) return fallback;
    return (uint64_t)v * 1048576ull;
}

static uint64_t parse_gib(const char *s, uint64_t fallback) {
    if (!s || !s[0]) return fallback;
    char *end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s) return fallback;
    return (uint64_t)v * 1073741824ull;
}

static void build_range_plan(std::vector<tensor_span> &spans, uint64_t span_bytes,
                             std::vector<tensor_span> &ranges) {
    std::sort(spans.begin(), spans.end(), [](const tensor_span &a, const tensor_span &b) {
        if (a.off != b.off) return a.off < b.off;
        return a.end < b.end;
    });

    /* Grow each range by WHOLE tensor spans up to span_bytes; a single tensor
     * larger than span_bytes becomes its own oversized range.  Never cut
     * inside a tensor: the client resolves a tensor with one whole-tensor
     * containment lookup (cuda_model_range_ptr), so a mid-tensor split makes
     * that tensor unresolvable against the import table and it silently falls
     * back to the host-mmap tier (measured: the 10.7 GiB Q4_K drafter's
     * 2.11 GiB expert stacks split at the old 1 GiB chunk boundary read at
     * page-fault rate, ~1.7 s per DSpark block step vs ~11 ms served). */
    ranges.clear();
    for (size_t i = 0; i < spans.size();) {
        uint64_t off = spans[i].off;
        uint64_t end = spans[i].end;
        i++;
        while (i < spans.size() && spans[i].off <= end + 65536u &&
               spans[i].end - off <= span_bytes) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        ranges.push_back({off, end});
    }
}

static uint64_t range_plan_bytes(const std::vector<tensor_span> &ranges) {
    uint64_t total = 0;
    for (const tensor_span &r : ranges) {
        if (r.end >= r.off && UINT64_MAX - total >= r.end - r.off) total += r.end - r.off;
    }
    return total;
}

static bool inspect_model_plan(const char *id, const char *path, uint64_t span_bytes,
                               uint64_t *model_size_out, uint64_t *bytes_out,
                               uint64_t *ranges_out, uint64_t vmm_granularity,
                               uint64_t *vmm_alloc_bytes_out) {
    mapped_file m;
    if (!map_file(path, m)) return false;
    std::vector<tensor_span> spans;
    std::vector<tensor_record> records;
    if (!collect_tensor_catalog(m, spans, &records)) {
        unmap_file(m);
        return false;
    }
    std::vector<tensor_span> ranges;
    build_range_plan(spans, span_bytes, ranges);
    const uint64_t planned = range_plan_bytes(ranges);
    const uint64_t vmm_alloc = vmm_granularity ? sum_rounded_ranges(ranges, vmm_granularity) : 0;
    fprintf(stderr,
            "ds4_weight_server: %s plan model=%.2f GiB raw_tensor_ranges=%.2f GiB ranges=%zu\n",
            id,
            (double)m.size / 1073741824.0,
            (double)planned / 1073741824.0,
            ranges.size());
    fprintf(stderr,
            "ds4_weight_server: %s catalog tensors=%zu\n",
            id,
            records.size());
    if (vmm_granularity) {
        fprintf(stderr,
                "ds4_weight_server: %s vmm plan logical=%.2f GiB allocated=%.2f GiB granularity=%llu\n",
                id,
                (double)planned / 1073741824.0,
                (double)vmm_alloc / 1073741824.0,
                (unsigned long long)vmm_granularity);
    }
    if (model_size_out) *model_size_out = m.size;
    if (bytes_out) *bytes_out = planned;
    if (ranges_out) *ranges_out = (uint64_t)ranges.size();
    if (vmm_alloc_bytes_out) *vmm_alloc_bytes_out = vmm_alloc;
    unmap_file(m);
    return true;
}

/* Host MemAvailable (free + reclaimable page cache), in bytes; 0 if unreadable. */
static uint64_t host_mem_available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long v = 0;
        if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) { kb = (uint64_t)v; break; }
    }
    fclose(f);
    return kb * 1024ull;
}

static bool cuda_memory_preflight(const char *what, uint64_t planned_bytes, uint64_t reserve_bytes) {
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: cudaMemGetInfo failed for %s: %s\n",
                what, cudaGetErrorString(err));
        return false;
    }
    /* On integrated / unified-memory GPUs (e.g. DGX Spark GB10), cudaMemGetInfo's
     * `free` reports only physically-free pages (kernel MemFree) and ignores the
     * reclaimable page cache -- which on these hosts is dominated by the mmap'd
     * model file itself. A device allocation reclaims that clean, file-backed
     * cache, so the real allocatable budget is host MemAvailable, not MemFree.
     * Discrete-VRAM GPUs have no such host cache to reclaim, so MemFree stays the
     * correct ceiling there. Use the larger of the two only when integrated. */
    uint64_t budget = (uint64_t)free_b;
    int integrated = 0;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) == cudaSuccess &&
        integrated) {
        const uint64_t avail = host_mem_available_bytes();
        if (avail > budget) budget = avail;
    }
    const char *bypass = getenv("DS4_WEIGHT_SERVER_NO_PREFLIGHT");
    fprintf(stderr,
            "ds4_weight_server: memory preflight %s need=%.2f GiB reserve=%.2f GiB free=%.2f GiB budget=%.2f GiB total=%.2f GiB integrated=%d\n",
            what,
            (double)planned_bytes / 1073741824.0,
            (double)reserve_bytes / 1073741824.0,
            (double)free_b / 1073741824.0,
            (double)budget / 1073741824.0,
            (double)total_b / 1073741824.0,
            integrated);
    if (bypass && bypass[0]) return true;
    if (reserve_bytes > 0 &&
        planned_bytes <= UINT64_MAX - reserve_bytes &&
        budget < planned_bytes + reserve_bytes) {
        fprintf(stderr,
                "ds4_weight_server: refusing upload; not enough allocatable CUDA memory for %s plus reserve. "
                "Stop other model processes, lower --reserve-gb, or set DS4_WEIGHT_SERVER_NO_PREFLIGHT=1.\n",
                what);
        return false;
    }
    return true;
}

struct upload_stage_pool {
    cudaStream_t stream = nullptr;
    void *raw[4] = {nullptr, nullptr, nullptr, nullptr};
    void *stage[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaEvent_t event[4] = {nullptr, nullptr, nullptr, nullptr};
    uint64_t bytes = 0;
    uint64_t align = 1;
};

static void upload_stage_pool_destroy(upload_stage_pool &pool) {
    for (int i = 0; i < 4; i++) {
        if (pool.event[i]) cudaEventDestroy(pool.event[i]);
        if (pool.raw[i]) cudaFreeHost(pool.raw[i]);
        pool.event[i] = nullptr;
        pool.raw[i] = nullptr;
        pool.stage[i] = nullptr;
    }
    if (pool.stream) cudaStreamDestroy(pool.stream);
    pool.stream = nullptr;
    pool.bytes = 0;
    pool.align = 1;
}

static bool upload_stage_pool_init(upload_stage_pool &pool, uint64_t bytes, uint64_t align) {
    if (align < 1) align = 1;
    if (pool.bytes >= bytes && pool.align >= align) return true;
    upload_stage_pool_destroy(pool);
    cudaError_t err = cudaStreamCreateWithFlags(&pool.stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: upload stream create failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    for (int i = 0; i < 4; i++) {
        const uint64_t alloc_bytes = bytes + (align > 1 ? align : 1);
        err = cudaMallocHost(&pool.raw[i], (size_t)alloc_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: pinned staging alloc failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        pool.stage[i] = align_ptr(pool.raw[i], align);
        err = cudaEventCreateWithFlags(&pool.event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: staging event create failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }
    pool.bytes = bytes;
    pool.align = align;
    return true;
}

static bool read_stage(const mapped_file &m, void *stage, uint64_t stage_bytes,
                       uint64_t file_off, uint64_t bytes, const char **payload) {
    return ds4_repack_read_stage(m, stage, stage_bytes, file_off, bytes, payload);
}

static bool upload_range_chunked(const mapped_file &m, uint64_t file_off, void *dev, uint64_t bytes,
                                 upload_stage_pool &pool, uint64_t chunk_bytes) {
    const uint64_t stage_align = m.direct_fd >= 0 ? m.direct_align : 1;
    const uint64_t stage_bytes = chunk_bytes + (stage_align > 1 ? stage_align : 1);
    if (!upload_stage_pool_init(pool, stage_bytes, stage_align)) return false;
    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = bytes - copied < chunk_bytes ? bytes - copied : chunk_bytes;
        const int bi = (int)(chunk_idx % 4u);
        cudaError_t err = cudaSuccess;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(pool.event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4_weight_server: staging wait failed: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        const char *payload = nullptr;
        if (!read_stage(m, pool.stage[bi], pool.bytes, file_off + copied, n, &payload)) {
            fprintf(stderr, "ds4_weight_server: model read failed at off=%llu: %s\n",
                    (unsigned long long)(file_off + copied), strerror(errno));
            return false;
        }
        err = cudaMemcpyAsync((char *)dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, pool.stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: async upload failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        err = cudaEventRecord(pool.event[bi], pool.stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: staging record failed: %s\n", cudaGetErrorString(err));
            return false;
        }
#if defined(POSIX_FADV_DONTNEED)
        (void)posix_fadvise(m.fd, (off_t)(file_off + copied), (off_t)n, POSIX_FADV_DONTNEED);
#endif
        copied += n;
        chunk_idx++;
    }
    cudaError_t err = cudaStreamSynchronize(pool.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: upload sync failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

static void release_owned_range(owned_range &r) {
    if (r.exported_fd >= 0) {
        close(r.exported_fd);
        r.exported_fd = -1;
    }
    if (r.backend == WEIGHT_BACKEND_VMM) {
        if (r.vmm_va && r.alloc_bytes) {
            (void)cuMemUnmap(r.vmm_va, (size_t)r.alloc_bytes);
            (void)cuMemAddressFree(r.vmm_va, (size_t)r.alloc_bytes);
            r.vmm_va = 0;
        }
        if (r.vmm_handle) {
            (void)cuMemRelease(r.vmm_handle);
            r.vmm_handle = 0;
        }
        r.dev = nullptr;
        return;
    }
    if (r.dev) {
        (void)cudaFree(r.dev);
        r.dev = nullptr;
    }
}

static bool vmm_alloc_range(int device, uint64_t logical_bytes, uint64_t granularity, owned_range &r) {
    if (logical_bytes == 0 || granularity == 0) return false;
    const uint64_t alloc_bytes = align_up(logical_bytes, granularity);
    CUmemAllocationProp prop = vmm_allocation_prop(device);
    CUmemGenericAllocationHandle handle;
    memset(&handle, 0, sizeof(handle));
    if (!driver_ok(cuMemCreate(&handle, (size_t)alloc_bytes, &prop, 0), "VMM allocation create")) {
        return false;
    }
    CUdeviceptr va = 0;
    if (!driver_ok(cuMemAddressReserve(&va, (size_t)alloc_bytes, 0, 0, 0), "VMM address reserve")) {
        (void)cuMemRelease(handle);
        return false;
    }
    if (!driver_ok(cuMemMap(va, (size_t)alloc_bytes, 0, handle, 0), "VMM map")) {
        (void)cuMemAddressFree(va, (size_t)alloc_bytes);
        (void)cuMemRelease(handle);
        return false;
    }
    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!driver_ok(cuMemSetAccess(va, (size_t)alloc_bytes, &access, 1), "VMM set access")) {
        (void)cuMemUnmap(va, (size_t)alloc_bytes);
        (void)cuMemAddressFree(va, (size_t)alloc_bytes);
        (void)cuMemRelease(handle);
        return false;
    }
    r.backend = WEIGHT_BACKEND_VMM;
    r.alloc_bytes = alloc_bytes;
    r.vmm_handle = handle;
    r.vmm_va = va;
    r.dev = (void *)va;
    return true;
}

static void hex_encode(const void *data, size_t bytes, std::string &out) {
    static const char *hex = "0123456789abcdef";
    const uint8_t *p = (const uint8_t *)data;
    out.resize(bytes * 2u);
    for (size_t i = 0; i < bytes; i++) {
        out[2u * i] = hex[p[i] >> 4];
        out[2u * i + 1u] = hex[p[i] & 15u];
    }
}

static bool upload_model(const char *id, const char *path, uint64_t span_bytes,
                         uint64_t copy_chunk_bytes, std::vector<owned_range> &ranges,
                         weight_backend backend, int device, uint64_t vmm_granularity,
                         std::vector<tensor_record> *records_out,
                         bool exclude_iq2_aligned,
                         bool exclude_q2k_aligned) {
    mapped_file m;
    if (!map_file(path, m)) return false;
    std::vector<tensor_span> spans;
    std::vector<tensor_record> records;
    if (!collect_tensor_catalog(m, spans, &records)) {
        unmap_file(m);
        return false;
    }
    if (records_out) *records_out = records;
    if (exclude_iq2_aligned || exclude_q2k_aligned) {
        /* --repack-iq2-aligned / --repack-q2k-aligned: the candidate tensors
         * are re-served as aligned-SoA derived artifacts
         * (build_iq2_aligned_expert_artifacts /
         * build_q2k_aligned_expert_artifacts), so drop their raw spans from
         * the upload plan rather than holding both copies (~45 GiB gate/up +
         * ~30 GiB down would not fit).  collect_tensor_catalog pushes exactly
         * one span per tensor, so matching on the span start is
         * unambiguous. */
        std::unordered_set<uint64_t> excluded;
        uint64_t excluded_bytes = 0;
        size_t excluded_iq2 = 0, excluded_q2k = 0;
        for (const tensor_record &t : records) {
            if (exclude_iq2_aligned &&
                (ds4_repack_iq2_candidate(t) || ds4_repack_iq2_xs_candidate(t))) {
                excluded.insert(t.off);
                excluded_bytes += t.bytes;
                excluded_iq2++;
            } else if (exclude_q2k_aligned && ds4_repack_q2k_candidate(t)) {
                excluded.insert(t.off);
                excluded_bytes += t.bytes;
                excluded_q2k++;
            }
        }
        if (!excluded.empty()) {
            std::vector<tensor_span> kept;
            kept.reserve(spans.size());
            for (const tensor_span &s : spans) {
                if (excluded.find(s.off) == excluded.end()) kept.push_back(s);
            }
            spans.swap(kept);
            fprintf(stderr,
                    "ds4_weight_server: %s excluding %zu iq2 + %zu q2k expert tensors (%.2f GiB) from raw upload for aligned repack\n",
                    id,
                    excluded_iq2,
                    excluded_q2k,
                    (double)excluded_bytes / 1073741824.0);
        }
    }
    std::vector<tensor_span> plan;
    build_range_plan(spans, span_bytes, plan);
    if (m.data) {
        munmap((void *)m.data, (size_t)m.size);
        m.data = nullptr;
    }

    upload_stage_pool pool;
    uint64_t uploaded = 0;
    uint64_t count = 0;
    for (const tensor_span &planned_range : plan) {
        const uint64_t off = planned_range.off;
        const uint64_t bytes = planned_range.end - planned_range.off;
        owned_range r;
        r.backend = backend;
        r.model_id = id;
        r.model_size = m.size;
        r.off = off;
        r.bytes = bytes;
        r.alloc_bytes = bytes;
        if (backend == WEIGHT_BACKEND_VMM) {
            if (!vmm_alloc_range(device, bytes, vmm_granularity, r)) {
                fprintf(stderr, "ds4_weight_server: VMM allocation failed for %s %.2f MiB\n",
                        id, (double)bytes / 1048576.0);
                upload_stage_pool_destroy(pool);
                unmap_file(m);
                return false;
            }
        } else {
            void *dev = nullptr;
            cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4_weight_server: cudaMalloc failed for %s %.2f MiB: %s\n",
                        id, (double)bytes / 1048576.0, cudaGetErrorString(err));
                upload_stage_pool_destroy(pool);
                unmap_file(m);
                return false;
            }
            r.dev = dev;
        }
        if (!upload_range_chunked(m, off, r.dev, bytes, pool, copy_chunk_bytes)) {
            release_owned_range(r);
            upload_stage_pool_destroy(pool);
            unmap_file(m);
            return false;
        }
        if (backend == WEIGHT_BACKEND_IPC) {
            cudaIpcMemHandle_t handle;
            cudaError_t err = cudaIpcGetMemHandle(&handle, r.dev);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4_weight_server: cudaIpcGetMemHandle failed for %s: %s\n",
                        id, cudaGetErrorString(err));
                release_owned_range(r);
                upload_stage_pool_destroy(pool);
                unmap_file(m);
                return false;
            }
            r.handle = handle;
        }
        ranges.push_back(r);
        uploaded += bytes;
        count++;
        if (count == 1 || uploaded / (8ull * 1073741824ull) != (uploaded - bytes) / (8ull * 1073741824ull)) {
            fprintf(stderr, "ds4_weight_server: %s uploaded %.2f GiB\r", id, (double)uploaded / 1073741824.0);
            fflush(stderr);
        }
    }
    upload_stage_pool_destroy(pool);
    fprintf(stderr, "ds4_weight_server: %s uploaded %.2f GiB across %llu ranges\n",
            id, (double)uploaded / 1073741824.0, (unsigned long long)count);
    unmap_file(m);
    return true;
}

static const tensor_record *find_tensor_record(const std::vector<tensor_record> &records,
                                               const char *name) {
    if (!name) return nullptr;
    for (const tensor_record &t : records) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

static const unsigned char *owned_raw_range_ptr(const std::vector<owned_range> &ranges,
                                                const char *model_id,
                                                uint64_t off,
                                                uint64_t bytes) {
    const uint64_t end = off + bytes;
    if (!model_id || end < off) return nullptr;
    for (const owned_range &r : ranges) {
        if (r.derived || r.model_id != model_id) continue;
        const uint64_t rend = r.off + r.bytes;
        if (rend < r.off) continue;
        if (off >= r.off && end <= rend) {
            return (const unsigned char *)r.dev + (off - r.off);
        }
    }
    return nullptr;
}

static bool build_output_certifier_norms(const char *model_id,
                                         uint64_t model_size,
                                         const std::vector<tensor_record> &records,
                                         std::vector<owned_range> &ranges,
                                         weight_backend backend,
                                         int device,
                                         uint64_t vmm_granularity,
                                         uint32_t group_count,
                                         uint64_t *derived_bytes_used,
                                         uint64_t derived_budget_bytes) {
    if (!model_id || group_count == 0) return false;
    const tensor_record *t = find_tensor_record(records, "output.weight");
    if (!t) {
        fprintf(stderr, "ds4_weight_server: output certifier derivation skipped for %s: missing output.weight\n",
                model_id);
        return false;
    }
    if (t->type != 8 || t->ndim != 2 || t->dims[0] == 0 || t->dims[1] == 0) {
        fprintf(stderr,
                "ds4_weight_server: output certifier derivation skipped for %s: output.weight is not 2D Q8_0\n",
                model_id);
        return false;
    }
    if (t->dims[1] > UINT64_MAX / group_count / sizeof(float)) {
        fprintf(stderr, "ds4_weight_server: output certifier derived size overflow\n");
        return false;
    }
    const uint64_t in_dim = t->dims[0];
    const uint64_t out_dim = t->dims[1];
    const uint64_t bytes = out_dim * (uint64_t)group_count * sizeof(float);
    if (derived_bytes_used &&
        derived_budget_bytes != 0 &&
        (*derived_bytes_used > derived_budget_bytes ||
         bytes > derived_budget_bytes - *derived_bytes_used)) {
        fprintf(stderr,
                "ds4_weight_server: output certifier derivation exceeds derived budget "
                "(request=%.2f MiB used=%.2f MiB budget=%.2f MiB)\n",
                (double)bytes / 1048576.0,
                (double)*derived_bytes_used / 1048576.0,
                (double)derived_budget_bytes / 1048576.0);
        return false;
    }
    const unsigned char *src = owned_raw_range_ptr(ranges, model_id, t->off, t->bytes);
    if (!src) {
        fprintf(stderr,
                "ds4_weight_server: output certifier derivation skipped for %s: raw output.weight is not resident\n",
                model_id);
        return false;
    }

    owned_range r;
    r.backend = backend;
    r.derived = true;
    r.model_id = model_id;
    r.model_size = model_size;
    r.off = 0;
    r.bytes = bytes;
    r.alloc_bytes = bytes;
    r.derived_kind = DERIVED_Q8_0_ROW_GROUP_NORMS;
    r.source_off = t->off;
    r.source_bytes = t->bytes;
    r.in_dim = in_dim;
    r.out_dim = out_dim;
    r.group_count = group_count;
    r.source_name = t->name;
    if (backend == WEIGHT_BACKEND_VMM) {
        if (!vmm_alloc_range(device, bytes, vmm_granularity, r)) {
            fprintf(stderr, "ds4_weight_server: VMM derived allocation failed for output certifier %.2f MiB\n",
                    (double)bytes / 1048576.0);
            return false;
        }
    } else {
        void *dev = nullptr;
        cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: cudaMalloc failed for output certifier derived %.2f MiB: %s\n",
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            return false;
        }
        r.dev = dev;
    }

    const double t0 = now_sec();
    const uint64_t blocks = (in_dim + 31u) / 32u;
    q8_0_row_group_norms_warp_kernel<<<(unsigned)out_dim, 512>>>(
            (float *)r.dev,
            src,
            in_dim,
            out_dim,
            blocks,
            group_count);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: output certifier derived kernel failed: %s\n",
                cudaGetErrorString(err));
        release_owned_range(r);
        return false;
    }
    if (backend == WEIGHT_BACKEND_IPC) {
        cudaIpcMemHandle_t handle;
        err = cudaIpcGetMemHandle(&handle, r.dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: cudaIpcGetMemHandle failed for output certifier derived: %s\n",
                    cudaGetErrorString(err));
            release_owned_range(r);
            return false;
        }
        r.handle = handle;
    }
    const double t1 = now_sec();
    ranges.push_back(r);
    if (derived_bytes_used) *derived_bytes_used += bytes;
    fprintf(stderr,
            "ds4_weight_server: derived %s output certifier row norms %.2f MiB groups=%u in=%llu out=%llu built in %.3fs\n",
            model_id,
            (double)bytes / 1048576.0,
            group_count,
            (unsigned long long)in_dim,
            (unsigned long long)out_dim,
            t1 - t0);
    return true;
}

static bool build_q8_0_dequant_artifact(const char *model_id,
                                        uint64_t model_size,
                                        const std::vector<tensor_record> &records,
                                        std::vector<owned_range> &ranges,
                                        weight_backend backend,
                                        int device,
                                        uint64_t vmm_granularity,
                                        const char *tensor_name,
                                        uint32_t kind,
                                        uint64_t *derived_bytes_used,
                                        uint64_t derived_budget_bytes) {
    if (!model_id || !tensor_name ||
        (kind != DERIVED_Q8_0_F16_COLMAJOR && kind != DERIVED_Q8_0_F32_COLMAJOR)) {
        return false;
    }
    const tensor_record *t = find_tensor_record(records, tensor_name);
    if (!t) {
        fprintf(stderr, "ds4_weight_server: q8 derived artifact skipped for %s: missing %s\n",
                model_id, tensor_name);
        return false;
    }
    if (t->type != 8 || t->ndim != 2 || t->dims[0] == 0 || t->dims[1] == 0) {
        fprintf(stderr,
                "ds4_weight_server: q8 derived artifact skipped for %s: %s is not 2D Q8_0\n",
                model_id,
                tensor_name);
        return false;
    }
    const uint64_t in_dim = t->dims[0];
    const uint64_t out_dim = t->dims[1];
    const uint64_t elem_bytes = kind == DERIVED_Q8_0_F16_COLMAJOR ? sizeof(__half) : sizeof(float);
    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / elem_bytes) {
        fprintf(stderr, "ds4_weight_server: q8 derived artifact size overflow for %s\n", tensor_name);
        return false;
    }
    const uint64_t bytes = in_dim * out_dim * elem_bytes;
    if (derived_bytes_used &&
        derived_budget_bytes != 0 &&
        (*derived_bytes_used > derived_budget_bytes ||
         bytes > derived_budget_bytes - *derived_bytes_used)) {
        fprintf(stderr,
                "ds4_weight_server: q8 derived artifact %s exceeds derived budget "
                "(request=%.2f MiB used=%.2f MiB budget=%.2f MiB)\n",
                tensor_name,
                (double)bytes / 1048576.0,
                (double)*derived_bytes_used / 1048576.0,
                (double)derived_budget_bytes / 1048576.0);
        return false;
    }
    const unsigned char *src = owned_raw_range_ptr(ranges, model_id, t->off, t->bytes);
    if (!src) {
        fprintf(stderr,
                "ds4_weight_server: q8 derived artifact skipped for %s: raw %s is not resident\n",
                model_id,
                tensor_name);
        return false;
    }

    owned_range r;
    r.backend = backend;
    r.derived = true;
    r.model_id = model_id;
    r.model_size = model_size;
    r.bytes = bytes;
    r.alloc_bytes = bytes;
    r.derived_kind = kind;
    r.source_off = t->off;
    r.source_bytes = t->bytes;
    r.in_dim = in_dim;
    r.out_dim = out_dim;
    r.group_count = 0;
    r.source_name = t->name;
    if (backend == WEIGHT_BACKEND_VMM) {
        if (!vmm_alloc_range(device, bytes, vmm_granularity, r)) {
            fprintf(stderr, "ds4_weight_server: VMM derived allocation failed for %s %.2f MiB\n",
                    tensor_name,
                    (double)bytes / 1048576.0);
            return false;
        }
    } else {
        void *dev = nullptr;
        cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: cudaMalloc failed for %s derived %.2f MiB: %s\n",
                    tensor_name,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            return false;
        }
        r.dev = dev;
    }

    const double t0 = now_sec();
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t n = in_dim * out_dim;
    if (kind == DERIVED_Q8_0_F16_COLMAJOR) {
        dequant_q8_0_to_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                (__half *)r.dev,
                src,
                in_dim,
                out_dim,
                blocks);
    } else {
        dequant_q8_0_to_f32_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)r.dev,
                src,
                in_dim,
                out_dim,
                blocks);
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: q8 derived kernel failed for %s: %s\n",
                tensor_name,
                cudaGetErrorString(err));
        release_owned_range(r);
        return false;
    }
    if (backend == WEIGHT_BACKEND_IPC) {
        cudaIpcMemHandle_t handle;
        err = cudaIpcGetMemHandle(&handle, r.dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: cudaIpcGetMemHandle failed for %s derived: %s\n",
                    tensor_name,
                    cudaGetErrorString(err));
            release_owned_range(r);
            return false;
        }
        r.handle = handle;
    }
    const double t1 = now_sec();
    ranges.push_back(r);
    if (derived_bytes_used) *derived_bytes_used += bytes;
    fprintf(stderr,
            "ds4_weight_server: derived %s %s %s %.2f MiB in=%llu out=%llu built in %.3fs\n",
            model_id,
            tensor_name,
            kind == DERIVED_Q8_0_F16_COLMAJOR ? "q8_0_f16" : "q8_0_f32",
            (double)bytes / 1048576.0,
            (unsigned long long)in_dim,
            (unsigned long long)out_dim,
            t1 - t0);
    return true;
}

/* ---- aligned repack artifacts (shared layout library) --------------------
 * The candidate scan, S5 parallel driver, repack kernels, and byte layouts
 * live in cuda/mmq/ds4_repack.cu, shared with the engine's in-process
 * self-load build so both producers emit bit-identical artifacts.  This
 * adapter owns the weight-server side: VMM/IPC allocation, export handles,
 * and owned_range bookkeeping for the manifest.  The iq2/q2k artifacts
 * REPLACE the raw residency of their source tensors (upload_model excluded
 * those spans), so they are not charged to --derive-budget-gb; the q8 dense
 * artifacts are ADDITIVE (raw stays served). */
struct ws_repack_vmm_ctx {
    int device = 0;
    uint64_t granularity = 0;
    std::unordered_map<const void *, owned_range> allocs; /* dev -> vmm bookkeeping */
};

static bool ws_repack_vmm_alloc(void *ctx_p, ds4_repack_artifact *art) {
    ws_repack_vmm_ctx *ctx = (ws_repack_vmm_ctx *)ctx_p;
    owned_range r;
    if (!vmm_alloc_range(ctx->device, art->bytes, ctx->granularity, r)) {
        fprintf(stderr, "ds4_weight_server: VMM allocation failed for repack %s %.2f MiB\n",
                art->t->name.c_str(), (double)art->bytes / 1048576.0);
        return false;
    }
    art->dev = r.dev;
    ctx->allocs[r.dev] = r;
    return true;
}

static void ws_repack_vmm_free(void *ctx_p, ds4_repack_artifact *art) {
    ws_repack_vmm_ctx *ctx = (ws_repack_vmm_ctx *)ctx_p;
    auto it = ctx->allocs.find(art->dev);
    if (it == ctx->allocs.end()) return;
    release_owned_range(it->second);
    ctx->allocs.erase(it);
}

static bool ws_build_aligned_artifacts(uint32_t kind,
                                       const char *model_id,
                                       const char *path,
                                       uint64_t model_size,
                                       const std::vector<tensor_record> &records,
                                       std::vector<owned_range> &ranges,
                                       weight_backend backend,
                                       int device,
                                       uint64_t vmm_granularity,
                                       uint64_t copy_chunk_bytes,
                                       uint64_t *repacked_bytes_out) {
    ds4_repack_build_args a;
    a.log_prefix = "ds4_weight_server";
    a.model_id = model_id;
    a.path = path;
    a.records = &records;
    a.device = device;
    a.copy_chunk_bytes = copy_chunk_bytes;
    ws_repack_vmm_ctx vmm_ctx;
    if (backend == WEIGHT_BACKEND_VMM) {
        vmm_ctx.device = device;
        vmm_ctx.granularity = vmm_granularity;
        a.alloc_fn = ws_repack_vmm_alloc;
        a.free_fn = ws_repack_vmm_free;
        a.alloc_ctx = &vmm_ctx;
    }
    std::vector<ds4_repack_artifact> arts;
    const bool ok = kind == DERIVED_Q8_0_ALIGNED_DENSE
        ? ds4_repack_build_q8_aligned(a, arts, repacked_bytes_out)
        : kind == DERIVED_MOTIF3_KV_B_VALUE_Q8_0
        ? ds4_repack_build_motif3_kv_b_value(a, arts, repacked_bytes_out)
        : kind == DERIVED_IQ2_XXS_ALIGNED_MOE
        ? ds4_repack_build_iq2_aligned(a, arts, repacked_bytes_out)
        : kind == DERIVED_IQ2_XS_ALIGNED_MOE
        ? ds4_repack_build_iq2_xs_aligned(a, arts, repacked_bytes_out)
        : ds4_repack_build_q2k_aligned(a, arts, repacked_bytes_out);
    if (!ok) return false;
    for (size_t i = 0; i < arts.size(); i++) {
        const ds4_repack_artifact &art = arts[i];
        owned_range r;
        if (backend == WEIGHT_BACKEND_VMM) {
            auto it = vmm_ctx.allocs.find(art.dev);
            if (it == vmm_ctx.allocs.end()) return false; /* unreachable */
            r = it->second;
        } else {
            r.dev = art.dev;
            r.alloc_bytes = art.bytes;
        }
        r.backend = backend;
        r.derived = true;
        r.model_id = model_id;
        r.model_size = model_size;
        r.off = 0;
        r.bytes = art.bytes;
        r.derived_kind = art.kind;
        r.source_off = art.t->off;
        r.source_bytes = art.t->bytes;
        r.in_dim = art.in_dim;
        r.out_dim = art.out_dim;
        r.group_count = art.group_count;
        r.source_name = art.t->name;
        if (backend == WEIGHT_BACKEND_IPC) {
            cudaIpcMemHandle_t handle;
            cudaError_t err = cudaIpcGetMemHandle(&handle, r.dev);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4_weight_server: cudaIpcGetMemHandle failed for repack %s: %s\n",
                        art.t->name.c_str(), cudaGetErrorString(err));
                for (size_t k = i; k < arts.size(); k++) (void)cudaFree(arts[k].dev);
                return false;
            }
            r.handle = handle;
        }
        ranges.push_back(r);
    }
    return true;
}

/* Rider #48: one content record per served model file so importers can
 * refuse when this server is holding a superseded checkpoint (identity
 * used to be model_id + size only, and same-size checkpoint swaps passed).
 * The fingerprint runs over a fresh private mapping of the file taken
 * here, at manifest-write time: if the file was replaced after upload,
 * the record describes the NEW bytes, importers of the OLD uploaded
 * ranges mismatch, and the swap is caught instead of served. */
struct content_entry {
    std::string model_id;
    uint64_t model_size;
    uint64_t fp;
};

static bool fingerprint_model_file(const char *model_id, const char *path,
                                   uint64_t expect_size,
                                   std::vector<content_entry> &contents) {
    /* DFM split GGUFs: the identity the importer checks is the LOGICAL
     * model -- the page-spanned shard concatenation both repack_map_split
     * here and the engine's model_open_split lay out identically.  A raw
     * stat of the first shard would neither match the plan's logical size
     * nor the importer's mapped bytes, so take the same logical mapping
     * the plan pass used.  Single files map 1:1 and are unchanged. */
    mapped_file m;
    if (!map_file(path, m)) {
        fprintf(stderr, "ds4_weight_server: content fingerprint map failed %s\n",
                path);
        return false;
    }
    if (m.size != expect_size) {
        fprintf(stderr,
                "ds4_weight_server: %s changed size during startup "
                "(%llu at plan, %llu now): %s\n",
                model_id,
                (unsigned long long)expect_size,
                (unsigned long long)m.size,
                path);
        unmap_file(m);
        return false;
    }
    content_entry e;
    e.model_id = model_id;
    e.model_size = m.size;
    e.fp = ds4_weight_content_fingerprint(m.data, m.size);
    unmap_file(m);
    fprintf(stderr, "ds4_weight_server: content fingerprint %s %s %016llx\n",
            model_id, DS4_WFP_ALGO, (unsigned long long)e.fp);
    contents.push_back(e);
    return true;
}

static bool write_manifest(const char *path, const std::vector<owned_range> &ranges,
                           int device, const char *scope, const char *lock_file,
                           weight_backend backend, const char *broker_path,
                           const std::vector<content_entry> &contents) {
    std::string tmp = std::string(path) + ".tmp";
    FILE *fp = fopen(tmp.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ds4_weight_server: manifest open failed %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    bool has_derived = false;
    for (const owned_range &r : ranges) {
        if (r.derived) {
            has_derived = true;
            break;
        }
    }
    if (backend == WEIGHT_BACKEND_VMM) {
        fprintf(fp, "%s\n", has_derived ? "DS4_WEIGHT_SERVER_VMM_DERIVED_V1" : "DS4_WEIGHT_SERVER_VMM_V1");
    } else {
        fprintf(fp, "%s\n", has_derived ? "DS4_WEIGHT_SERVER_IPC_DERIVED_V1" : "DS4_WEIGHT_SERVER_IPC_V1");
    }
    fprintf(fp, "# owner <pid> <cuda-device> <scope> <lock-file-or-dash>\n");
    fprintf(fp, "owner %ld %d %s %s\n",
            (long)getpid(),
            device,
            scope ? scope : "both",
            (lock_file && lock_file[0]) ? lock_file : "-");
    if (backend == WEIGHT_BACKEND_VMM) {
        fprintf(fp, "# broker <unix-socket-path>\n");
        fprintf(fp, "broker %s\n", broker_path ? broker_path : "");
    }
    if (!contents.empty()) {
        fprintf(fp, "# content <model-id> <model-size> <algo> <fingerprint-hex>\n");
        for (const content_entry &c : contents) {
            fprintf(fp, "content %s %llu %s %016llx\n",
                    c.model_id.c_str(),
                    (unsigned long long)c.model_size,
                    DS4_WFP_ALGO,
                    (unsigned long long)c.fp);
        }
    }
    if (backend == WEIGHT_BACKEND_VMM) {
        fprintf(fp, "# alloc <alloc-id> <model-id> <model-size> <offset> <bytes> <alloc-bytes>\n");
        fprintf(fp, "# derived-alloc <alloc-id> <model-id> <model-size> <source-offset> <source-bytes> <kind> <in-dim> <out-dim> <group-count> <bytes> <alloc-bytes> <source-name>\n");
    } else {
        fprintf(fp, "# range <model-id> <model-size> <offset> <bytes> <cuda-ipc-handle-hex>\n");
        fprintf(fp, "# derived-range <model-id> <model-size> <source-offset> <source-bytes> <kind> <in-dim> <out-dim> <group-count> <bytes> <cuda-ipc-handle-hex> <source-name>\n");
    }
    uint64_t alloc_id = 0;
    for (const owned_range &r : ranges) {
        if (backend == WEIGHT_BACKEND_VMM) {
            if (r.derived) {
                fprintf(fp, "derived-alloc %llu %s %llu %llu %llu %u %llu %llu %u %llu %llu %s\n",
                        (unsigned long long)alloc_id++,
                        r.model_id.c_str(),
                        (unsigned long long)r.model_size,
                        (unsigned long long)r.source_off,
                        (unsigned long long)r.source_bytes,
                        r.derived_kind,
                        (unsigned long long)r.in_dim,
                        (unsigned long long)r.out_dim,
                        r.group_count,
                        (unsigned long long)r.bytes,
                        (unsigned long long)r.alloc_bytes,
                        r.source_name.c_str());
            } else {
                fprintf(fp, "alloc %llu %s %llu %llu %llu %llu\n",
                        (unsigned long long)alloc_id++,
                        r.model_id.c_str(),
                        (unsigned long long)r.model_size,
                        (unsigned long long)r.off,
                        (unsigned long long)r.bytes,
                        (unsigned long long)r.alloc_bytes);
            }
        } else {
            std::string hex;
            hex_encode(&r.handle, sizeof(r.handle), hex);
            if (r.derived) {
                fprintf(fp, "derived-range %s %llu %llu %llu %u %llu %llu %u %llu %s %s\n",
                        r.model_id.c_str(),
                        (unsigned long long)r.model_size,
                        (unsigned long long)r.source_off,
                        (unsigned long long)r.source_bytes,
                        r.derived_kind,
                        (unsigned long long)r.in_dim,
                        (unsigned long long)r.out_dim,
                        r.group_count,
                        (unsigned long long)r.bytes,
                        hex.c_str(),
                        r.source_name.c_str());
            } else {
                fprintf(fp, "range %s %llu %llu %llu %s\n",
                        r.model_id.c_str(),
                        (unsigned long long)r.model_size,
                        (unsigned long long)r.off,
                        (unsigned long long)r.bytes,
                        hex.c_str());
            }
        }
    }
    if (fclose(fp) != 0) return false;
    if (rename(tmp.c_str(), path) != 0) {
        fprintf(stderr, "ds4_weight_server: manifest rename failed: %s\n", strerror(errno));
        return false;
    }
    return true;
}

static bool export_vmm_fds(std::vector<owned_range> &ranges) {
    uint64_t count = 0;
    for (owned_range &r : ranges) {
        if (r.backend != WEIGHT_BACKEND_VMM) continue;
        int fd = -1;
        if (!driver_ok(cuMemExportToShareableHandle(&fd,
                                                    r.vmm_handle,
                                                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                                    0),
                       "VMM export POSIX FD")) {
            return false;
        }
        r.exported_fd = fd;
        count++;
    }
    fprintf(stderr, "ds4_weight_server: vmm exported %llu POSIX file descriptors\n",
            (unsigned long long)count);
    return true;
}

static bool fd_broker_start(fd_broker &broker, const char *path) {
    if (!path || !path[0]) return false;
    if (strlen(path) >= sizeof(sockaddr_un::sun_path)) {
        fprintf(stderr, "ds4_weight_server: broker socket path is too long: %s\n", path);
        return false;
    }
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "ds4_weight_server: broker socket create failed: %s\n", strerror(errno));
        return false;
    }
    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1u);
    (void)unlink(path);
    if (bind(fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "ds4_weight_server: broker bind failed %s: %s\n", path, strerror(errno));
        close(fd);
        return false;
    }
    (void)chmod(path, 0600);
    if (listen(fd, 16) != 0) {
        fprintf(stderr, "ds4_weight_server: broker listen failed %s: %s\n", path, strerror(errno));
        close(fd);
        unlink(path);
        return false;
    }
    broker.listen_fd = fd;
    broker.path = path;
    broker.requests = 0;
    fprintf(stderr, "ds4_weight_server: broker listening %s\n", path);
    return true;
}

static void fd_broker_stop(fd_broker &broker) {
    if (broker.listen_fd >= 0) {
        close(broker.listen_fd);
        broker.listen_fd = -1;
    }
    if (!broker.path.empty()) {
        unlink(broker.path.c_str());
    }
}

static bool send_status_fd(int client_fd, const char *status, int fd_to_send) {
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    struct iovec iov;
    iov.iov_base = (void *)status;
    iov.iov_len = strlen(status);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    char control[CMSG_SPACE(sizeof(int))];
    memset(control, 0, sizeof(control));
    if (fd_to_send >= 0) {
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));
    }
    return sendmsg(client_fd, &msg, 0) >= 0;
}

static void fd_broker_serve_once(fd_broker &broker,
                                 const std::vector<owned_range> &ranges,
                                 long wait_usec) {
    if (broker.listen_fd < 0) return;
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(broker.listen_fd, &rfds);
    timeval tv;
    tv.tv_sec = 0;
    if (wait_usec < 0) wait_usec = 0;
    tv.tv_sec = wait_usec / 1000000;
    tv.tv_usec = wait_usec % 1000000;
    int ready = select(broker.listen_fd + 1, &rfds, nullptr, nullptr, &tv);
    if (ready <= 0 || !FD_ISSET(broker.listen_fd, &rfds)) return;
    int client = accept(broker.listen_fd, nullptr, nullptr);
    if (client < 0) return;
    char buf[128];
    ssize_t n = read(client, buf, sizeof(buf) - 1u);
    if (n <= 0) {
        close(client);
        return;
    }
    buf[n] = '\0';
    unsigned long long alloc_id = 0;
    if (sscanf(buf, "GET %llu", &alloc_id) != 1 || alloc_id >= ranges.size() ||
        ranges[(size_t)alloc_id].exported_fd < 0) {
        (void)send_status_fd(client, "ERR invalid allocation\n", -1);
        close(client);
        return;
    }
    const owned_range &r = ranges[(size_t)alloc_id];
    char status[128];
    snprintf(status, sizeof(status), "OK %llu %llu\n",
             alloc_id,
             (unsigned long long)r.alloc_bytes);
    if (send_status_fd(client, status, r.exported_fd)) {
        broker.requests++;
        if (broker.requests == 1u || (broker.requests & 63u) == 0u ||
            alloc_id + 1u == ranges.size()) {
            fprintf(stderr,
                    "ds4_weight_server: broker served alloc=%llu bytes=%llu requests=%llu\n",
                    alloc_id,
                    (unsigned long long)r.alloc_bytes,
                    (unsigned long long)broker.requests);
        }
    }
    close(client);
}

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4_weight_server --base FILE [--mtp FILE] [--drafter FILE] --manifest FILE [options]\n"
            "\n"
            "Options:\n"
            "  --drafter FILE    Also serve a DSpark drafter GGUF (model id \"drafter\").\n"
            "                    Independent of --scope; raw spans only, no repacks.\n"
            "  --device N        CUDA device ordinal. Default: 0\n"
            "  --backend B       Weight sharing backend: ipc or vmm. Default: ipc\n"
            "  --scope S         Models to upload: both, base, or mtp. Default: both\n"
            "  --broker-socket FILE Unix socket for VMM FD transfer. Default: <manifest>.sock\n"
            "  --exit-on-parent-pid N Exit if parent/orchestrator PID disappears\n"
            "  --lock-file FILE  Single-owner lock file. Default: /tmp/ds4_weight_server_cudaN.lock\n"
            "  --no-lock         Disable the single-owner lock\n"
            "  --span-mb N       Raw range granularity target. Ranges grow by whole\n"
            "                    tensors up to this size; a single larger tensor gets\n"
            "                    its own oversized range (never split). Default: 1024\n"
            "  --copy-chunk-mb N Pinned staged upload chunk. Default: 256\n"
            "  --reserve-gb N    Free CUDA memory to keep unused. Default: 32\n"
            "  --derive-output-certifier Build base output Q8_0 row-group norms for exact verifier\n"
            "  --derive-group-count N Row groups for --derive-output-certifier. Default: 8\n"
            "  --derive-q8-f16 NAME Build a base Q8_0 tensor as imported F16 layout. Repeatable\n"
            "  --derive-q8-f32 NAME Build a base Q8_0 tensor as imported F32 layout. Repeatable\n"
            "  --derive-budget-gb N Maximum derived artifact memory. Default: 4\n"
            "  --repack-iq2-aligned Serve base IQ2_XXS routed-expert gate/up stacks in the\n"
            "                    64B-aligned SoA layout (replaces their raw ranges; byte-neutral).\n"
            "                    Default: ON. Disable with --no-repack-iq2-aligned.\n"
            "                    NOTE: needs clients >= 136cec7 (older MoE decode widths >=2\n"
            "                    fall to a catastrophic host-mmap tier on a repacked manifest).\n"
            "  --repack-q8-aligned Additionally serve large base Q8_0 dense tensors in the\n"
            "                    64B-aligned SoA layout (raw ranges stay served; ~6 GiB extra).\n"
            "                    Default: ON. Disable with --no-repack-q8-aligned.\n"
            "  --repack-q2k-aligned Serve base Q2_K routed-expert down stacks in the row-pair\n"
            "                    SoA layout (replaces their raw ranges; byte-neutral).\n"
            "                    Default: ON. Disable with --no-repack-q2k-aligned.\n"
            "                    NOTE: needs clients with ds4_mmq_q2_K_aligned_moe_vec (M2\n"
            "                    moe-down commit); older clients read the down stacks through\n"
            "                    a catastrophic host-mmap tier on a repacked manifest.\n"
            "  --repack-threads N Worker threads for the aligned repack builds (S5).\n"
            "                    Default: auto (min(6, cores)); DS4_WS_REPACK_THREADS also read.\n"
            "  --repack-hash     Print a per-artifact FNV-1a after each repack build (the\n"
            "                    thread-count bit-identity gate; costs a D2H pass).\n"
            "  --dry-run         Parse GGUFs, print upload plan, and exit before allocation\n");
}

int main(int argc, char **argv) {
    const char *base = nullptr;
    const char *mtp = nullptr;
    const char *drafter = nullptr;
    const char *manifest = nullptr;
    const char *scope = "both";
    const char *backend_s = "ipc";
    const char *broker_socket = nullptr;
    const char *lock_file = nullptr;
    pid_t exit_on_parent_pid = 0;
    int device = 0;
    uint64_t span_bytes = 1024ull * 1048576ull;
    uint64_t copy_chunk_bytes = 256ull * 1048576ull;
    uint64_t reserve_bytes = 32ull * 1073741824ull;
    bool derive_output_certifier = false;
    uint32_t derive_group_count = 8;
    std::vector<std::string> derive_q8_f16;
    std::vector<std::string> derive_q8_f32;
    uint64_t derive_budget_bytes = 4ull * 1073741824ull;
    /* Aligned-SoA repacks default ON (quality gate passed 2026-07-04:
     * gsm8k 97.4 / HumanEval 92.1 / MBPP 89.0 vs 97.6/88.4/90.0 baselines;
     * serial decode −1.4 ms/tok).  NOTE the iq2 repack REPLACES raw ranges:
     * clients older than 136cec7 fall to a host-mmap tier at MoE decode
     * widths >=2 — pair a repacked manifest with current clients only. */
    bool repack_iq2_aligned = true;
    bool repack_q8_aligned = true;
    /* M2 moe-down row-pair-SoA Q2_K repack.  Default ON since the 2026-07-05
     * quality gate (gsm8k 486/500=97.2 / HumanEval 150/164=91.5 / MBPP
     * 177/200=88.5, all within boot noise of the iq2/q8-flip 97.4/92.1/89.0;
     * the moe-down twin is bit-identical to the raw path).  Like the iq2
     * repack it REPLACES the raw .ffn_down_exps ranges, so it needs clients
     * >= e221241 (older ones fall to a host-mmap down tier on a repacked
     * manifest); disable with --no-repack-q2k-aligned. */
    bool repack_q2k_aligned = true;
    bool repack_explicit = false;
    bool dry_run = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--base") && i + 1 < argc) base = argv[++i];
        else if (!strcmp(argv[i], "--mtp") && i + 1 < argc) mtp = argv[++i];
        else if (!strcmp(argv[i], "--drafter") && i + 1 < argc) drafter = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
        else if (!strcmp(argv[i], "--scope") && i + 1 < argc) scope = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend_s = argv[++i];
        else if (!strcmp(argv[i], "--broker-socket") && i + 1 < argc) broker_socket = argv[++i];
        else if (!strcmp(argv[i], "--exit-on-parent-pid") && i + 1 < argc) exit_on_parent_pid = (pid_t)strtol(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--lock-file") && i + 1 < argc) lock_file = argv[++i];
        else if (!strcmp(argv[i], "--no-lock")) lock_file = "";
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--span-mb") && i + 1 < argc) span_bytes = parse_mib(argv[++i], span_bytes);
        else if (!strcmp(argv[i], "--copy-chunk-mb") && i + 1 < argc) copy_chunk_bytes = parse_mib(argv[++i], copy_chunk_bytes);
        else if (!strcmp(argv[i], "--reserve-gb") && i + 1 < argc) reserve_bytes = parse_gib(argv[++i], reserve_bytes);
        else if (!strcmp(argv[i], "--derive-output-certifier")) derive_output_certifier = true;
        else if (!strcmp(argv[i], "--derive-group-count") && i + 1 < argc) {
            unsigned long v = strtoul(argv[++i], nullptr, 10);
            if (v > 0 && v <= 16) derive_group_count = (uint32_t)v;
            else {
                fprintf(stderr, "ds4_weight_server: invalid --derive-group-count; expected 1..16\n");
                return 2;
            }
        }
        else if (!strcmp(argv[i], "--derive-q8-f16") && i + 1 < argc) derive_q8_f16.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--derive-q8-f32") && i + 1 < argc) derive_q8_f32.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--derive-budget-gb") && i + 1 < argc) derive_budget_bytes = parse_gib(argv[++i], derive_budget_bytes);
        else if (!strcmp(argv[i], "--repack-iq2-aligned")) { repack_iq2_aligned = true; repack_explicit = true; }
        else if (!strcmp(argv[i], "--repack-q8-aligned")) { repack_q8_aligned = true; repack_explicit = true; }
        else if (!strcmp(argv[i], "--repack-q2k-aligned")) { repack_q2k_aligned = true; repack_explicit = true; }
        else if (!strcmp(argv[i], "--repack-threads") && i + 1 < argc) ds4_repack_set_threads(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--repack-hash")) ds4_repack_set_hash(true);
        else if (!strcmp(argv[i], "--no-repack-iq2-aligned")) repack_iq2_aligned = false;
        else if (!strcmp(argv[i], "--no-repack-q8-aligned")) repack_q8_aligned = false;
        else if (!strcmp(argv[i], "--no-repack-q2k-aligned")) repack_q2k_aligned = false;
        else if (!strcmp(argv[i], "--dry-run")) dry_run = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "ds4_weight_server: unknown or incomplete option: %s\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }
    bool want_base = true;
    bool want_mtp = true;
    weight_backend backend = WEIGHT_BACKEND_IPC;
    if (!parse_scope(scope, want_base, want_mtp)) {
        fprintf(stderr, "ds4_weight_server: invalid --scope %s; expected both, base, or mtp\n", scope);
        return 2;
    }
    if (!parse_backend(backend_s, backend)) {
        fprintf(stderr, "ds4_weight_server: invalid --backend %s; expected ipc or vmm\n", backend_s);
        return 2;
    }
    if ((want_base && !base) || (want_mtp && !mtp) || (!manifest && !dry_run)) {
        usage(stderr);
        return 2;
    }
    if (span_bytes < 64ull * 1048576ull) span_bytes = 64ull * 1048576ull;
    if (span_bytes > 4096ull * 1048576ull) span_bytes = 4096ull * 1048576ull;
    if (copy_chunk_bytes < 16ull * 1048576ull) copy_chunk_bytes = 16ull * 1048576ull;
    if (copy_chunk_bytes > 1024ull * 1048576ull) copy_chunk_bytes = 1024ull * 1048576ull;
    if (!want_base && (repack_iq2_aligned || repack_q8_aligned || repack_q2k_aligned)) {
        if (repack_explicit) {
            fprintf(stderr, "ds4_weight_server: derived artifacts currently require base scope\n");
            return 2;
        }
        /* default-on repacks quietly drop out of a base-less scope */
        repack_iq2_aligned = false;
        repack_q8_aligned = false;
        repack_q2k_aligned = false;
    }
    if ((derive_output_certifier || !derive_q8_f16.empty() || !derive_q8_f32.empty()) && !want_base) {
        fprintf(stderr, "ds4_weight_server: derived artifacts currently require base scope\n");
        return 2;
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (!driver_ok(cuInit(0), "init")) return 1;

    vmm_support support;
    uint64_t vmm_granularity = 0;
    if (backend == WEIGHT_BACKEND_VMM) {
        if (!query_vmm_support(device, support)) {
            fprintf(stderr, "ds4_weight_server: VMM backend is not supported on CUDA device %d\n", device);
            return 1;
        }
        vmm_granularity = (uint64_t)support.granularity_recommended;
    }

    char default_lock[128];
    int lock_fd = -1;
    if (!dry_run) {
        if (!lock_file) {
            snprintf(default_lock, sizeof(default_lock), "/tmp/ds4_weight_server_cuda%d.lock", device);
            lock_file = default_lock;
        }
        if (lock_file[0]) {
            lock_fd = acquire_owner_lock(lock_file);
            if (lock_fd < 0) return 1;
        }
    }

    uint64_t total_upload_bytes = 0;
    uint64_t total_alloc_bytes = 0;
    uint64_t base_bytes = 0;
    uint64_t base_model_size = 0;
    uint64_t mtp_model_size = 0;
    if (want_base) {
        uint64_t base_alloc_bytes = 0;
        if (!inspect_model_plan("base", base, span_bytes, &base_model_size, &base_bytes, nullptr,
                                vmm_granularity, &base_alloc_bytes)) return 1;
        total_upload_bytes += base_bytes;
        total_alloc_bytes += backend == WEIGHT_BACKEND_VMM ? base_alloc_bytes : base_bytes;
    }
    if (want_mtp) {
        uint64_t mtp_bytes = 0;
        uint64_t mtp_alloc_bytes = 0;
        if (!inspect_model_plan("mtp", mtp, span_bytes, &mtp_model_size, &mtp_bytes, nullptr,
                                vmm_granularity, &mtp_alloc_bytes)) return 1;
        if (UINT64_MAX - total_upload_bytes < mtp_bytes) {
            fprintf(stderr, "ds4_weight_server: upload plan size overflow\n");
            return 1;
        }
        if (UINT64_MAX - total_alloc_bytes < (backend == WEIGHT_BACKEND_VMM ? mtp_alloc_bytes : mtp_bytes)) {
            fprintf(stderr, "ds4_weight_server: allocation plan size overflow\n");
            return 1;
        }
        total_upload_bytes += mtp_bytes;
        total_alloc_bytes += backend == WEIGHT_BACKEND_VMM ? mtp_alloc_bytes : mtp_bytes;
    }
    uint64_t drafter_model_size = 0;
    if (drafter) {
        uint64_t drafter_bytes = 0;
        uint64_t drafter_alloc_bytes = 0;
        if (!inspect_model_plan("drafter", drafter, span_bytes, &drafter_model_size, &drafter_bytes,
                                nullptr, vmm_granularity, &drafter_alloc_bytes)) return 1;
        if (UINT64_MAX - total_upload_bytes < drafter_bytes) {
            fprintf(stderr, "ds4_weight_server: upload plan size overflow\n");
            return 1;
        }
        if (UINT64_MAX - total_alloc_bytes < (backend == WEIGHT_BACKEND_VMM ? drafter_alloc_bytes : drafter_bytes)) {
            fprintf(stderr, "ds4_weight_server: allocation plan size overflow\n");
            return 1;
        }
        total_upload_bytes += drafter_bytes;
        total_alloc_bytes += backend == WEIGHT_BACKEND_VMM ? drafter_alloc_bytes : drafter_bytes;
    }
    fprintf(stderr,
            "ds4_weight_server: backend=%s logical_upload=%.2f GiB allocation_plan=%.2f GiB\n",
            backend_name(backend),
            (double)total_upload_bytes / 1073741824.0,
            (double)total_alloc_bytes / 1073741824.0);
    if (!cuda_memory_preflight("full upload plan", total_alloc_bytes, reserve_bytes)) return 1;
    if (dry_run) {
        fprintf(stderr, "ds4_weight_server: dry-run complete; no allocations or manifest were created\n");
        return 0;
    }

    std::vector<owned_range> ranges;
    std::vector<tensor_record> base_records;
    std::vector<tensor_record> mtp_records;
    if (want_base && !upload_model("base", base, span_bytes, copy_chunk_bytes, ranges,
                                   backend, device, vmm_granularity, &base_records,
                                   repack_iq2_aligned, repack_q2k_aligned)) return 1;
    if (want_mtp && !upload_model("mtp", mtp, span_bytes, copy_chunk_bytes, ranges,
                                  backend, device, vmm_granularity, &mtp_records,
                                  false, false)) return 1;
    if (drafter && !upload_model("drafter", drafter, span_bytes, copy_chunk_bytes, ranges,
                                 backend, device, vmm_granularity, nullptr,
                                 false, false)) return 1;
    if (repack_q2k_aligned) {
        uint64_t q2k_repacked_bytes = 0;
        if (!ws_build_aligned_artifacts(DERIVED_Q2_K_ALIGNED_MOE,
                                        "base",
                                        base,
                                        base_model_size,
                                        base_records,
                                        ranges,
                                        backend,
                                        device,
                                        vmm_granularity,
                                        copy_chunk_bytes,
                                        &q2k_repacked_bytes)) {
            return 1;
        }
    }
    if (repack_iq2_aligned) {
        uint64_t repacked_bytes = 0;
        if (!ws_build_aligned_artifacts(DERIVED_IQ2_XXS_ALIGNED_MOE,
                                        "base",
                                        base,
                                        base_model_size,
                                        base_records,
                                        ranges,
                                        backend,
                                        device,
                                        vmm_granularity,
                                        copy_chunk_bytes,
                                        &repacked_bytes)) {
            return 1;
        }
        uint64_t xs_bytes = 0;
        if (!ws_build_aligned_artifacts(DERIVED_IQ2_XS_ALIGNED_MOE,
                                        "base",
                                        base,
                                        base_model_size,
                                        base_records,
                                        ranges,
                                        backend,
                                        device,
                                        vmm_granularity,
                                        copy_chunk_bytes,
                                        &xs_bytes)) {
            return 1;
        }
    }
    if (repack_q8_aligned) {
        uint64_t q8_repacked_bytes = 0;
        if (!ws_build_aligned_artifacts(DERIVED_Q8_0_ALIGNED_DENSE,
                                        "base",
                                        base,
                                        base_model_size,
                                        base_records,
                                        ranges,
                                        backend,
                                        device,
                                        vmm_granularity,
                                        copy_chunk_bytes,
                                        &q8_repacked_bytes)) {
            return 1;
        }
        uint64_t motif3_kv_b_value_bytes = 0;
        if (!ws_build_aligned_artifacts(
                DERIVED_MOTIF3_KV_B_VALUE_Q8_0,
                "base",
                base,
                base_model_size,
                base_records,
                ranges,
                backend,
                device,
                vmm_granularity,
                copy_chunk_bytes,
                &motif3_kv_b_value_bytes)) {
            return 1;
        }
    }
    uint64_t derived_bytes_used = 0;
    if (derive_output_certifier &&
        !build_output_certifier_norms("base",
                                      base_model_size,
                                      base_records,
                                      ranges,
                                      backend,
                                      device,
                                      vmm_granularity,
                                      derive_group_count,
                                      &derived_bytes_used,
                                      derive_budget_bytes)) {
        return 1;
    }
    for (const std::string &name : derive_q8_f16) {
        if (!build_q8_0_dequant_artifact("base",
                                         base_model_size,
                                         base_records,
                                         ranges,
                                         backend,
                                         device,
                                         vmm_granularity,
                                         name.c_str(),
                                         DERIVED_Q8_0_F16_COLMAJOR,
                                         &derived_bytes_used,
                                         derive_budget_bytes)) {
            return 1;
        }
    }
    for (const std::string &name : derive_q8_f32) {
        if (!build_q8_0_dequant_artifact("base",
                                         base_model_size,
                                         base_records,
                                         ranges,
                                         backend,
                                         device,
                                         vmm_granularity,
                                         name.c_str(),
                                         DERIVED_Q8_0_F32_COLMAJOR,
                                         &derived_bytes_used,
                                         derive_budget_bytes)) {
            return 1;
        }
    }
    if (derived_bytes_used != 0) {
        fprintf(stderr,
                "ds4_weight_server: derived artifacts total %.2f MiB budget %.2f MiB\n",
                (double)derived_bytes_used / 1048576.0,
                (double)derive_budget_bytes / 1048576.0);
    }
    std::vector<content_entry> contents;
    if (want_base &&
        !fingerprint_model_file("base", base, base_model_size, contents)) return 1;
    if (want_mtp &&
        !fingerprint_model_file("mtp", mtp, mtp_model_size, contents)) return 1;
    if (drafter &&
        !fingerprint_model_file("drafter", drafter, drafter_model_size, contents)) return 1;

    std::string default_broker_socket;
    fd_broker broker;
    if (backend == WEIGHT_BACKEND_VMM) {
        if (!export_vmm_fds(ranges)) return 1;
        if (!broker_socket) {
            default_broker_socket = std::string(manifest) + ".sock";
            broker_socket = default_broker_socket.c_str();
        }
        if (!fd_broker_start(broker, broker_socket)) return 1;
    }
    if (!write_manifest(manifest, ranges, device, scope, lock_file, backend, broker_socket, contents)) return 1;

    fprintf(stderr,
            "ds4_weight_server: ready manifest=%s ranges=%zu. Keep this process alive while workers run.\n",
            manifest,
            ranges.size());
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    while (!g_stop) {
        if (exit_on_parent_pid > 0 && !parent_pid_alive(exit_on_parent_pid)) {
            fprintf(stderr, "ds4_weight_server: parent pid %ld disappeared; shutting down\n",
                    (long)exit_on_parent_pid);
            break;
        }
        if (backend == WEIGHT_BACKEND_VMM) {
            /* Wait on the socket instead of polling then sleeping.  Workers
             * import hundreds of VMM allocations sequentially; the old
             * unconditional 100 ms sleep charged almost 45 seconds to a
             * 454-allocation Solar worker restart. */
            fd_broker_serve_once(broker, ranges, 100000);
        } else {
            sleep(1);
        }
    }

    fprintf(stderr, "ds4_weight_server: shutting down broker_requests=%llu\n",
            (unsigned long long)broker.requests);
    fd_broker_stop(broker);
    for (owned_range &r : ranges) {
        release_owned_range(r);
    }
    if (lock_fd >= 0) close(lock_fd);
    return 0;
}
