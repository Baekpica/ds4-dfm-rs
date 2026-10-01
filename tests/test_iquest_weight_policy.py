#!/usr/bin/env python3
"""Exercise the production weight-policy accessors across model lifetimes on CPU."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_weight_manifest_policy import REPO, source_block


KNOBS = (
    "DS4_CUDA_NO_DERIVED_WEIGHTS",
    "DS4_CUDA_NO_Q8_F16_CACHE",
    "DS4_CUDA_NO_Q8_F32_CACHE",
    "DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE",
    "DS4_CUDA_NO_ATTN_Q_B_F16_CACHE",
)


def harness():
    source = (REPO / "ds4_cuda.cu").read_text()
    family = (REPO / "ds4_iquest_gpu.cuh").read_text()
    pieces = [r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using CUmemGenericAllocationHandle = uint64_t;
using CUdeviceptr = uint64_t;
static int g_quality_mode = 0, g_q8_f16_disabled_after_oom = 0, g_cublas = 1;
static const char *g_derived_artifact_none_reason = nullptr;
constexpr int cudaSuccess = 0, cudaDevAttrIntegrated = 1;
static int cudaGetDevice(int *device) { *device = 0; return 0; }
static int cudaDeviceGetAttribute(int *value, int, int) { *value = 1; return 0; }
''']
    enum = "enum cuda_weight_policy {"
    if enum in source:
        pieces.append(source_block(source, enum) + ";")
        pieces.append(next(line for line in source.splitlines()
                           if line.startswith("static cuda_weight_policy g_weight_policy")))
    end = source.index("} cuda_weight_env_t;") + len("} cuda_weight_env_t;")
    start = source.rfind("typedef struct {", 0, end)
    pieces.append(source[start:end])
    for name in ("cuda_parse_mib_env", "cuda_weight_env_read", "cuda_weight_env",
                 "cuda_weight_policy_reset", "cuda_weight_no_derived"):
        marker = next((line for line in source.splitlines()
                       if line.startswith("static ") and
                       (f" {name}(" in line or f"&{name}(" in line)), None)
        if marker:
            pieces.append(source_block(source, marker))
    if "static void cuda_weight_policy_reset(" not in source:
        pieces.append("static void cuda_weight_policy_reset() {}")
    pieces.append(source_block(source, "struct cuda_derived_range {") + ";")
    pieces.append("static std::vector<cuda_derived_range> g_derived_ranges;")
    for marker in ("static char *cuda_derived_weight_ptr(",
                   "static uint64_t cuda_q8_f16_cache_limit_bytes(",
                   "static int cuda_q8_f16_cache_allowed(",
                   "static int cuda_q8_f32_cache_allowed(",
                   "static int cuda_derived_artifact_build_device(",
                   "static int manifest_weight_policy("):
        pieces.append(source_block(source, marker))
    pieces.append(source_block(family, 'extern "C" int ds4_gpu_iquest_policy('))
    pieces.append(r'''
static const char *keys[] = {
    "DS4_CUDA_NO_DERIVED_WEIGHTS", "DS4_CUDA_NO_Q8_F16_CACHE",
    "DS4_CUDA_NO_Q8_F32_CACHE", "DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE",
    "DS4_CUDA_NO_ATTN_Q_B_F16_CACHE"
};
static int failures = 0;
static void check(const char *name, int passed) {
    if (!passed) { fprintf(stderr, "FAIL %s\n", name); failures++; }
}
static int fields(const cuda_weight_env_t &e) {
    return e.no_derived | (e.no_q8_f16_cache << 1) | (e.no_q8_f32_cache << 2) |
        (e.no_attn_out_f16_cache << 3) | (e.no_attn_q_b_f16_cache << 4);
}
static void consumers(const cuda_weight_env_t &e) {
    check("derived resolver", (cuda_derived_weight_ptr(keys, 0, 32, 1, 4, 8, 1, 32, "test") == nullptr) == !!e.no_derived);
    check("F16 shared", cuda_q8_f16_cache_allowed("ffn_gate_shexp", 4, 8) == !e.no_q8_f16_cache);
    check("F16 output", cuda_q8_f16_cache_allowed("attn_output_a", 4, 8) == !(e.no_q8_f16_cache || e.no_attn_out_f16_cache));
    check("F16 q_b", cuda_q8_f16_cache_allowed("attn_q_b", 4, 8) == !(e.no_q8_f16_cache || e.no_attn_q_b_f16_cache));
    check("F32", cuda_q8_f32_cache_allowed("attn_q_b", 4, 8) == !e.no_q8_f32_cache);
    int device = -1;
    check("artifact build", cuda_derived_artifact_build_device(&device) == !e.no_derived);
    FILE *fp = tmpfile();
    fputs("alloc 1 base 64 0 32 32\nderived-alloc 2 base 64 0 32 1 4 8 1 64 64 tensor\n", fp);
    rewind(fp);
    check("manifest prescan", manifest_weight_policy(fp, "base") == !e.no_derived);
    fclose(fp);
}
int main(int argc, char **argv) {
    if (argc != 2) { return 2; }
    std::string saved[5]; int present = 0;
    for (unsigned i = 0; i < 5; i++) {
        const char *value = getenv(keys[i]);
        if (value) { present |= 1 << i; saved[i] = value; }
    }
    setenv("DS4_CUDA_ATTN_Q_B_F32_CACHE", "1", 1);
    const cuda_weight_env_t original = cuda_weight_env_read();
    cuda_derived_range range = {};
    range.host_base = keys; range.source_bytes = 32; range.kind = 1;
    range.in_dim = 4; range.out_dim = 8; range.group_count = 1;
    range.bytes = 32; range.device_ptr = (char *)keys;
    g_derived_ranges.push_back(range);
    if (strcmp(argv[1], "warm") == 0) {
        check("initial flags", fields(cuda_weight_env()) == present);
        consumers(original);
    }
    check("activate", ds4_gpu_iquest_policy());
    cuda_weight_env_t restricted = original;
    restricted.no_derived = restricted.no_q8_f16_cache = restricted.no_q8_f32_cache = 1;
    restricted.no_attn_out_f16_cache = restricted.no_attn_q_b_f16_cache = 1;
    check("active five flags", fields(cuda_weight_env()) == 31);
    check("only family fields change", !memcmp(&restricted, &cuda_weight_env(), sizeof(restricted)));
    consumers(restricted);
    setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1);
    cuda_weight_policy_reset();
    check("cleanup restores flags", fields(cuda_weight_env()) == present);
    check("cleanup restores common snapshot", !memcmp(&original, &cuda_weight_env(), sizeof(original)));
    consumers(original);
    for (unsigned i = 0; i < 5; i++) {
        const char *value = getenv(keys[i]);
        check("caller environment preserved", present & (1 << i)
            ? value && saved[i] == value : value == nullptr);
    }
    return failures ? 1 : 0;
}
''')
    return "\n".join(pieces)


class WeightPolicyTest(unittest.TestCase):
    def test_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix="iquest-weight-policy-") as directory:
            path = Path(directory)
            source, binary = path / "test.cc", path / "test"
            source.write_text(harness())
            subprocess.run(["c++", "-std=c++17", "-O0", str(source), "-o", str(binary)], check=True)
            cases = [(None, None)] + [(key, value) for key in KNOBS for value in ("", "0", "1")]
            for order in ("warm", "cold"):
                for key, value in cases:
                    with self.subTest(order=order, key=key, value=value):
                        env = {key: value for key, value in os.environ.items() if not key.startswith("DS4_")}
                        if key is not None:
                            env[key] = value
                        result = subprocess.run([str(binary), order], env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
            print("32 weight-policy lifecycle cases checked")

    def test_cleanup_wiring(self):
        source = (REPO / "ds4_cuda.cu").read_text()
        cleanup = source_block(source, 'extern "C" void ds4_gpu_cleanup(')
        self.assertTrue("cuda_weight_policy_reset();" in cleanup, "GPU cleanup must reset the family override")


if __name__ == "__main__":
    unittest.main()
