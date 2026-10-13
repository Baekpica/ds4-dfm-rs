/* V4.1 (ds41) VQ expert decode gate: the device row probe against the host
 * oracle, bit-exact.
 *
 *   tests/test_ds41_vq <blob> <probes.txt> [<ref.f32>]
 *
 * The probe (ds4_gpu_v41_vq_row_probe, ds4_ds41_gpu.cuh) dots n consecutive
 * rows of one expert matrix against a one-hot activation; with a one-hot at
 * column c the row_dot has exactly one nonzero term, so the result IS the
 * decoded value at (row, c) times nothing else — the device arithmetic is
 * compared against a host oracle without replicating the kernel's per-lane /
 * per-round / shfl-tree accumulation order.
 *
 * Oracle by version:
 *   v3 — ref.f32, produced independently of the kernels (the fixture
 *        generator's own index/codebook/gain arrays; on the Spark, vq.rs).
 *   v2 — ds4vq_dequant_f32 (ds41_vq_fmt.h), the engine's own host decoder.
 *
 * probes.txt: one line per probe `e which row n col`; ref.f32 carries the n
 * values of each line in order. PASS is bit-exact except for the sign of
 * zero (a one-hot dot can flush -0 to +0; not a decode difference).
 *
 * Build: make tests/test_ds41_vq (or the test-ds41-vq runner). */

#include "ds4_gpu.h"
#include "ds41_vq_fmt.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Probe {
    int e;
    int which;
    uint32_t row;
    uint32_t n;
    uint32_t col;
};

struct PayloadInfo {
    uint32_t nc;
    uint32_t rows;
    uint32_t cols;
};

bool same_value(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    if (ua == ub) return true;
    return a == 0.0f && b == 0.0f;   /* +0 vs -0: not a decode difference */
}

std::vector<uint8_t> read_file(const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> out(sz > 0 ? (size_t)sz : 0);
    if (sz > 0 && std::fread(out.data(), 1, (size_t)sz, f) != (size_t)sz) {
        std::fprintf(stderr, "short read %s\n", path);
        std::exit(2);
    }
    std::fclose(f);
    return out;
}

bool load_probes(const char *path, std::vector<Probe> &out) {
    FILE *f = std::fopen(path, "r");
    if (!f) return false;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int e, which;
        unsigned row, n, col;
        if (std::sscanf(line, "%d %d %u %u %u", &e, &which, &row, &n, &col) == 5) {
            out.push_back({e, which, row, n, col});
        }
    }
    std::fclose(f);
    return !out.empty();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s <blob> <probes.txt> [<ref.f32>]\n", argv[0]);
        return 2;
    }
    const std::vector<uint8_t> blob = read_file(argv[1]);
    if (!ds4vq_blob_ok(blob.data(), blob.size())) {
        std::fprintf(stderr, "blob rejected: %s\n", argv[1]);
        return 2;
    }
    const uint32_t ver = ds4vq_blob_ver(blob.data());
    std::vector<Probe> probes;
    if (!load_probes(argv[2], probes)) {
        std::fprintf(stderr, "no probes in %s\n", argv[2]);
        return 2;
    }
    std::vector<float> ref;
    if (ver == 3) {
        if (argc < 4) {
            std::fprintf(stderr, "v3 needs ref.f32 (the independent oracle)\n");
            return 2;
        }
        const std::vector<uint8_t> raw = read_file(argv[3]);
        ref.resize(raw.size() / sizeof(float));
        std::memcpy(ref.data(), raw.data(), ref.size() * sizeof(float));
    }
    size_t ref_total = 0;
    for (const Probe &p : probes) ref_total += p.n;
    if (ver == 3 && ref.size() != ref_total) {
        std::fprintf(stderr, "ref.f32 has %zu values, probes need %zu\n", ref.size(), ref_total);
        return 2;
    }

    /* Per-payload geometry from the blob itself: the probe call must carry the
     * payload's own nc/rows/cols, exactly like the decode worker's caller. */
    std::map<std::pair<int, int>, PayloadInfo> geo;
    std::map<std::pair<int, int>, std::vector<float>> v2_oracle;
    uint32_t max_cols = 1, max_n = 1;
    for (const Probe &p : probes) {
        const auto key = std::make_pair(p.e, p.which);
        if (geo.find(key) == geo.end()) {
            const uint64_t off = ds4vq_slot(blob.data(), p.e, p.which);
            if (off == 0 || off + 32 > blob.size()) {
                std::fprintf(stderr, "probe e=%d which=%d: slot absent\n", p.e, p.which);
                return 2;
            }
            const uint8_t *pay = blob.data() + off;
            uint32_t magic;
            std::memcpy(&magic, pay, 4);
            const uint32_t want = (ver == 3) ? DS4VQ_MAT3_MAGIC : DS4VQ_MAT_MAGIC;
            if (magic != want) {
                std::fprintf(stderr, "probe e=%d which=%d: payload magic %08x, blob v%u expects %08x\n",
                             p.e, p.which, magic, ver, want);
                return 2;
            }
            PayloadInfo pi;
            uint16_t nc16;
            std::memcpy(&nc16, pay + 6, 2);
            std::memcpy(&pi.rows, pay + 8, 4);
            std::memcpy(&pi.cols, pay + 12, 4);
            pi.nc = nc16;
            geo[key] = pi;
            if (ver == 2) {
                /* The engine's own host decoder, whole matrix once per slot. */
                std::vector<float> m((size_t)pi.rows * pi.cols);
                if (ds4vq_dequant_f32(pay, m.data(), (int)pi.rows, (int)pi.cols) != 0) {
                    std::fprintf(stderr, "ds4vq_dequant_f32 refused e=%d which=%d\n", p.e, p.which);
                    return 2;
                }
                v2_oracle[key] = std::move(m);
            }
        }
        if (p.col >= geo[key].cols || p.row + p.n > geo[key].rows) {
            std::fprintf(stderr, "probe out of payload geometry: e=%d which=%d row=%u n=%u col=%u (rows %u cols %u)\n",
                         p.e, p.which, p.row, p.n, p.col, geo[key].rows, geo[key].cols);
            return 2;
        }
        max_cols = p.col + 1 > max_cols ? p.col + 1 : max_cols;
        max_n = p.n > max_n ? p.n : max_n;
    }

    uint8_t *d_blob = nullptr;
    float *d_x = nullptr;
    float *d_out = nullptr;
    if (cudaMalloc(&d_blob, blob.size()) != cudaSuccess ||
        cudaMalloc(&d_x, max_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, max_n * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 2;
    }
    if (cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "blob upload failed\n");
        return 2;
    }

    size_t passes = 0, fails = 0, values = 0, ref_at = 0;
    std::vector<float> x, got;
    for (const Probe &p : probes) {
        const PayloadInfo &pi = geo[std::make_pair(p.e, p.which)];
        x.assign(pi.cols, 0.0f);
        x[p.col] = 1.0f;
        got.assign(p.n, 0.0f);
        if (cudaMemcpy(d_x, x.data(), pi.cols * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "x upload failed\n");
            return 2;
        }
        if (!ds4_gpu_v41_vq_row_probe(d_out, d_blob, ver, pi.nc, p.e, p.which, p.row, pi.rows, pi.cols, d_x, p.n)) {
            std::printf("FAIL e=%d which=%d row=%u n=%u col=%u: probe refused\n", p.e, p.which, p.row, p.n, p.col);
            fails++;
            ref_at += p.n;
            continue;
        }
        if (cudaMemcpy(got.data(), d_out, p.n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "result readback failed\n");
            return 2;
        }
        size_t bad = 0;
        for (uint32_t i = 0; i < p.n; i++) {
            float want;
            if (ver == 3) {
                want = ref[ref_at + i];
            } else {
                want = v2_oracle[std::make_pair(p.e, p.which)][(size_t)(p.row + i) * pi.cols + p.col];
            }
            values++;
            if (same_value(got[i], want)) continue;
            bad++;
            if (bad <= 3) {
                uint32_t uw, ug;
                std::memcpy(&uw, &want, 4);
                std::memcpy(&ug, &got[i], 4);
                std::printf("  mismatch row=%u col=%u want=%08x (%.9g) got=%08x (%.9g)%s\n",
                            p.row + i, p.col, uw, want, ug, got[i],
                            (got[i] != got[i] && want == want) ? "  [device NaN: payload open failed?]" : "");
            }
        }
        ref_at += p.n;
        if (bad == 0) {
            passes++;
            std::printf("PASS e=%d which=%d row=%u n=%u col=%u\n", p.e, p.which, p.row, p.n, p.col);
        } else {
            fails++;
            std::printf("FAIL e=%d which=%d row=%u n=%u col=%u: %zu/%u values differ\n", p.e, p.which, p.row, p.n, p.col, bad, p.n);
        }
    }

    cudaFree(d_blob);
    cudaFree(d_x);
    cudaFree(d_out);
    if (fails == 0) {
        std::printf("DS41 VQ row probe: PASS (blob v%u, %zu probes, %zu values bit-exact)\n", ver, passes, values);
        return 0;
    }
    std::printf("DS41 VQ row probe: FAIL (%zu/%zu probes)\n", fails, passes + fails);
    return 1;
}
