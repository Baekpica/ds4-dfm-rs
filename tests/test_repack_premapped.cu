/* Hermetic regression for split-GGUF aligned-artifact input.
 *
 * The engine maps every shard into one virtual address range and hands the
 * repacker a merged tensor catalog.  Builders must therefore read source
 * bytes from that mapping rather than reopening shard 00001.  Exercise the
 * contract with one Q8_0 tensor at a non-zero merged offset and verify the
 * resulting aligned-SoA bytes exactly.
 */
#include "../cuda/mmq/ds4_repack.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>

static int fail(const char *what) {
    std::fprintf(stderr, "test_repack_premapped: %s\n", what);
    return 1;
}

static bool write_all(std::FILE *f, const void *p, size_t n) {
    return std::fwrite(p, 1, n, f) == n;
}

static bool write_u16(std::FILE *f, uint16_t v) {
    return write_all(f, &v, sizeof(v));
}

static bool write_u32(std::FILE *f, uint32_t v) {
    return write_all(f, &v, sizeof(v));
}

static bool write_u64(std::FILE *f, uint64_t v) {
    return write_all(f, &v, sizeof(v));
}

static bool write_string(std::FILE *f, const char *s) {
    const uint64_t n = std::strlen(s);
    return write_u64(f, n) && write_all(f, s, (size_t)n);
}

static bool write_split_shard(const char *path, const char *name,
                              float value, bool first) {
    std::FILE *f = std::fopen(path, "wb");
    if (!f) return false;
    bool ok = write_u32(f, 0x46554747u) && write_u32(f, 3u) &&
              write_u64(f, 1u) && write_u64(f, first ? 1u : 0u);
    if (ok && first) {
        ok = write_string(f, "split.count") && write_u32(f, 2u) &&
             write_u16(f, 2u);
    }
    if (ok) {
        ok = write_string(f, name) && write_u32(f, 1u) &&
             write_u64(f, 1u) && write_u32(f, 0u) && write_u64(f, 0u);
    }
    long pos = ok ? std::ftell(f) : -1;
    while (ok && pos >= 0 && (pos % 32) != 0) {
        ok = std::fputc(0, f) != EOF;
        pos++;
    }
    if (ok) ok = write_all(f, &value, sizeof(value));
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

static int test_split_source() {
    char dir[] = "/tmp/ds4-repack-split-XXXXXX";
    if (!mkdtemp(dir)) return fail("mkdtemp failed");
    char first[PATH_MAX];
    char second[PATH_MAX];
    if (std::snprintf(first, sizeof(first),
                      "%s/fixture-00001-of-00002.gguf", dir) <= 0 ||
        std::snprintf(second, sizeof(second),
                      "%s/fixture-00002-of-00002.gguf", dir) <= 0) {
        return fail("split fixture path failed");
    }
    if (!write_split_shard(first, "first.weight", 1.25f, true) ||
        !write_split_shard(second, "second.weight", -2.5f, false)) {
        return fail("split fixture write failed");
    }

    ds4_repack_file model;
    if (!ds4_repack_map_file("test_repack_premapped", first, model)) {
        return fail("split model map failed");
    }
    if (model.shards.size() != 2u) return fail("split shard count mismatch");
    std::vector<ds4_repack_span> spans;
    std::vector<ds4_repack_tensor> records;
    if (!ds4_repack_collect_catalog(
            "test_repack_premapped", model, &spans, &records)) {
        return fail("split catalog failed");
    }
    if (records.size() != 2u || spans.size() != 2u ||
        records[0].name != "first.weight" ||
        records[1].name != "second.weight" ||
        records[1].off <= records[0].off) {
        return fail("split catalog merge mismatch");
    }
    alignas(4096) unsigned char stage[4096] = {};
    const char *payload = nullptr;
    if (!ds4_repack_read_stage(model, stage, sizeof(stage),
                               records[1].off, records[1].bytes, &payload)) {
        return fail("second shard staged read failed");
    }
    float got = 0.0f;
    std::memcpy(&got, payload, sizeof(got));
    if (got != -2.5f) return fail("second shard payload mismatch");
    ds4_repack_unmap_file(model);
    if (unlink(first) != 0 || unlink(second) != 0 || rmdir(dir) != 0) {
        return fail("split fixture cleanup failed");
    }
    std::puts("split repack source: ok");
    return 0;
}

static int test_q8_candidate_shapes() {
    ds4_repack_tensor kda;
    kda.name = "blk.1.ssm_f_b.weight";
    kda.type = 8u;
    kda.ndim = 2u;
    kda.dims[0] = 128u;
    kda.dims[1] = 8192u;
    kda.elements = kda.dims[0] * kda.dims[1];
    kda.bytes = (kda.dims[0] / 32u) * kda.dims[1] * 34u;
    if (!ds4_repack_q8_candidate(kda)) {
        return fail("K=128 wide Q8 tensor was not an aligned candidate");
    }

    ds4_repack_tensor tiny = kda;
    tiny.name = "tiny.weight";
    tiny.dims[1] = 1024u;
    tiny.elements = tiny.dims[0] * tiny.dims[1];
    tiny.bytes = (tiny.dims[0] / 32u) * tiny.dims[1] * 34u;
    if (ds4_repack_q8_candidate(tiny)) {
        return fail("tiny K=128 Q8 tensor should not allocate an artifact");
    }

    ds4_repack_tensor qwen_shared = kda;
    qwen_shared.name = "blk.1.ffn_gate_shexp.weight";
    qwen_shared.dims[0] = 2560u;
    qwen_shared.dims[1] = 640u;
    qwen_shared.elements = qwen_shared.dims[0] * qwen_shared.dims[1];
    qwen_shared.bytes = (qwen_shared.dims[0] / 32u) *
                        qwen_shared.dims[1] * 34u;
    if (!ds4_repack_q8_candidate(qwen_shared)) {
        return fail("Qwen shared-expert Q8 pair was not an aligned candidate");
    }

    ds4_repack_tensor qwen_qkv = kda;
    qwen_qkv.name = "blk.0.linear_attn.qkv.weight";
    qwen_qkv.dims[0] = 2560u;
    qwen_qkv.dims[1] = 10240u;
    qwen_qkv.elements = qwen_qkv.dims[0] * qwen_qkv.dims[1];
    qwen_qkv.bytes = (qwen_qkv.dims[0] / 32u) * qwen_qkv.dims[1] * 34u;
    if (!ds4_repack_q8_candidate(qwen_qkv)) {
        return fail("Qwen GDN qkv was not an aligned candidate");
    }

    ds4_repack_tensor qwen_z = qwen_qkv;
    qwen_z.name = "blk.0.linear_attn.z.weight";
    qwen_z.dims[1] = 6144u;
    qwen_z.elements = qwen_z.dims[0] * qwen_z.dims[1];
    qwen_z.bytes = (qwen_z.dims[0] / 32u) * qwen_z.dims[1] * 34u;
    if (!ds4_repack_q8_candidate(qwen_z)) {
        return fail("Qwen GDN z was not an aligned candidate");
    }

    ds4_repack_tensor qwen_q = qwen_qkv;
    qwen_q.name = "blk.3.attn_q.weight";
    qwen_q.dims[1] = 12288u;
    qwen_q.elements = qwen_q.dims[0] * qwen_q.dims[1];
    qwen_q.bytes = (qwen_q.dims[0] / 32u) * qwen_q.dims[1] * 34u;
    if (!ds4_repack_q8_candidate(qwen_q)) {
        return fail("Qwen QSA q was not an aligned candidate");
    }

    ds4_repack_tensor qwen_in_a = qwen_qkv;
    qwen_in_a.name = "blk.0.linear_attn.in_a.weight";
    qwen_in_a.dims[1] = 48u;
    qwen_in_a.elements = qwen_in_a.dims[0] * qwen_in_a.dims[1];
    qwen_in_a.bytes = (qwen_in_a.dims[0] / 32u) * qwen_in_a.dims[1] * 34u;
    if (ds4_repack_q8_candidate(qwen_in_a)) {
        return fail("Qwen GDN in_a should not allocate an artifact");
    }

    ds4_repack_tensor kv_b;
    kv_b.name = "blk.1.attn_kv_b.weight";
    kv_b.type = 8u;
    kv_b.ndim = 2u;
    kv_b.dims[0] = 512u;
    kv_b.dims[1] = 32768u;
    kv_b.bytes = (kv_b.dims[0] / 32u) * kv_b.dims[1] * 34u;
    if (!ds4_repack_motif3_kv_b_value_candidate(kv_b)) {
        return fail("Dots3 full kv_b was not a transposed-value candidate");
    }
    kv_b.dims[0] = 1024u;
    kv_b.dims[1] = 20480u;
    kv_b.bytes = (kv_b.dims[0] / 32u) * kv_b.dims[1] * 34u;
    if (!ds4_repack_motif3_kv_b_value_candidate(kv_b)) {
        return fail("Dots3 SWA kv_b was not a transposed-value candidate");
    }
    kv_b.dims[1]++;
    if (ds4_repack_motif3_kv_b_value_candidate(kv_b)) {
        return fail("unknown kv_b geometry should not allocate an artifact");
    }
    std::puts("q8 aligned candidate shapes: ok");
    return 0;
}

int main() {
    if (test_split_source() != 0) return 1;
    if (test_q8_candidate_shapes() != 0) return 1;
    constexpr uint64_t kOffset = 4096;
    constexpr uint64_t kIn = 1024;
    constexpr uint64_t kOut = 2048;
    constexpr uint64_t kBlocks = (kIn / 32u) * kOut;
    constexpr uint64_t kRawBytes = kBlocks * 34u;
    constexpr uint64_t kScaleBytes = kBlocks * 2u;
    constexpr uint64_t kArtifactBytes = kScaleBytes + kBlocks * 32u;

    std::vector<uint8_t> source(kOffset + kRawBytes + 64u, 0xa5u);
    for (uint64_t block = 0; block < kBlocks; block++) {
        uint8_t *raw = source.data() + kOffset + block * 34u;
        raw[0] = 0x00u;
        raw[1] = 0x3cu;  // IEEE fp16 1.0
        for (uint32_t i = 0; i < 32u; i++) {
            raw[2u + i] = static_cast<uint8_t>((block + i * 17u) & 0xffu);
        }
    }

    ds4_repack_tensor tensor;
    tensor.name = "blk.0.attn_q.weight";
    tensor.type = 8u;
    tensor.ndim = 2u;
    tensor.dims[0] = kIn;
    tensor.dims[1] = kOut;
    tensor.elements = kIn * kOut;
    tensor.off = kOffset;
    tensor.bytes = kRawBytes;
    std::vector<ds4_repack_tensor> records{tensor};

    ds4_repack_build_args args;
    args.log_prefix = "test_repack_premapped";
    args.model_id = "fixture";
    args.records = &records;
    args.copy_chunk_bytes = kRawBytes;
    args.source_data = source.data();
    args.source_size = source.size();

    std::vector<ds4_repack_artifact> artifacts;
    uint64_t built = 0;
    if (!ds4_repack_build_q8_aligned(args, artifacts, &built)) {
        return fail("premapped Q8 build failed");
    }
    if (artifacts.size() != 1u || built != kArtifactBytes ||
        artifacts[0].bytes != kArtifactBytes) {
        return fail("unexpected artifact geometry");
    }

    std::vector<uint8_t> actual(kArtifactBytes);
    cudaError_t err = cudaMemcpy(actual.data(), artifacts[0].dev,
                                 actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return fail(cudaGetErrorString(err));

    for (uint64_t block = 0; block < kBlocks; block++) {
        if (actual[block * 2u] != 0x00u ||
            actual[block * 2u + 1u] != 0x3cu) {
            return fail("scale plane mismatch");
        }
        const uint8_t *want = source.data() + kOffset + block * 34u + 2u;
        const uint8_t *got = actual.data() + kScaleBytes + block * 32u;
        if (std::memcmp(got, want, 32u) != 0) {
            return fail("quant plane mismatch");
        }
    }

    (void)cudaFree(artifacts[0].dev);
    std::puts("premapped repack: ok");
    return 0;
}
