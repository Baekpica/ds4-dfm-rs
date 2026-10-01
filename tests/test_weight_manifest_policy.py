#!/usr/bin/env python3
"""Run the production manifest parser with CPU mocks for GPU imports/publication."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
NO_DERIVED = "DS4_CUDA_NO_DERIVED_WEIGHTS"


def source_block(source, marker):
    start = source.index(marker)
    cursor = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


def harness_source():
    source = (REPO / "ds4_cuda.cu").read_text()
    pieces = [r'''
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using CUmemGenericAllocationHandle = uint64_t;
using CUdeviceptr = uint64_t;
using cudaError_t = int;
struct cudaIpcMemHandle_t { unsigned char bytes[64]; };
constexpr int cudaSuccess = 0;
constexpr int cudaIpcMemLazyEnablePeerAccess = 1;
constexpr int DS4_MEMC_WEIGHT_IMPORT = 1;
constexpr int DS4_MEMD_UNIFIED_DEVICE = 1;
constexpr int CUDA_DERIVED_ARTIFACTS_NONE = 0;
constexpr int CUDA_DERIVED_ARTIFACTS_BUILT = 1;
constexpr int CUDA_DERIVED_ARTIFACTS_MIXED = 2;
constexpr int CUDA_DERIVED_ARTIFACTS_IMPORTED = 3;
constexpr const char *DS4_WFP_ALGO = "test";
static int canonical = 0, derived = 0, gpu_calls = 0;
static uint64_t g_model_range_bytes = 0, g_derived_range_bytes = 0;
static uint64_t g_derived_artifact_count = 0, g_derived_artifact_bytes = 0;
static int g_derived_artifact_source = CUDA_DERIVED_ARTIFACTS_NONE;
static const char *g_derived_artifact_none_reason = nullptr;
static uint64_t ds4_weight_content_fingerprint(const void *, uint64_t) { return 0; }
static int cuInit(unsigned int) { ++gpu_calls; return 0; }
static int driver_ok(int result, const char *) { return result == 0; }
static const char *cudaGetErrorString(int) { return "mock"; }
static int cudaGetLastError() { return 0; }
static int cudaIpcOpenMemHandle(void **ptr, cudaIpcMemHandle_t, unsigned int) {
    ++gpu_calls;
    *ptr = reinterpret_cast<void *>(1);
    return 0;
}
static int cudaIpcCloseMemHandle(void *) { return 0; }
static void cuda_mem_note_alloc_src(int, int, uint64_t, uint64_t, const void *) {}
''']
    for name in ("cuda_model_range", "cuda_derived_range"):
        pieces.append(source_block(source, f"struct {name} {{") + ";")
    pieces.append(r'''
static int cuda_model_range_publish(const cuda_model_range &) {
    ++canonical;
    return 1;
}
static int cuda_derived_range_publish(const cuda_derived_range &) {
    ++derived;
    return 1;
}
static int import_vmm_allocation(
        const void *, uint64_t, const char *, const char *,
        unsigned long long, unsigned long long, unsigned long long,
        unsigned long long bytes, unsigned long long,
        uint64_t *total, uint64_t *ranges) {
    ++gpu_calls;
    ++canonical;
    *total += bytes;
    ++*ranges;
    return 1;
}
static int import_vmm_derived_allocation(
        const void *, uint64_t, const char *, const char *,
        unsigned long long, unsigned long long, unsigned long long,
        unsigned long long, unsigned int, unsigned long long,
        unsigned long long, unsigned int, unsigned long long bytes,
        unsigned long long, uint64_t *total, uint64_t *ranges) {
    ++gpu_calls;
    ++derived;
    *total += bytes;
    ++*ranges;
    return 1;
}
''')
    for name in ("cuda_hex_value", "cuda_hex_decode", "manifest_content_identity_check"):
        pieces.append(source_block(source, f"static int {name}("))
    # Include the policy helper after the fix; the pre-fix parser must fail
    # these same behavioral assertions without substituting a test parser.
    family_policy = "enum cuda_weight_policy {"
    if family_policy in source:
        pieces.append(source_block(source, family_policy) + ";")
        pieces.append(next(line for line in source.splitlines()
                           if line.startswith("static cuda_weight_policy g_weight_policy")))
        pieces.append(source_block(source, "static int cuda_weight_no_derived("))
        pieces.append(source_block(source, "static void cuda_weight_policy_reset("))
    else:
        pieces.append("static void cuda_weight_policy_reset() {}")
    family = (REPO / "ds4_iquest_gpu.cuh").read_text()
    pieces.append(source_block(family, 'extern "C" int ds4_gpu_iquest_policy('))
    policy = "static int manifest_weight_policy("
    if policy in source:
        pieces.append(source_block(source, policy))
    pieces.append(source_block(source, 'extern "C" int ds4_gpu_import_model_ipc_manifest('))
    pieces.append(r'''
int main(int argc, char **argv) {
    if (argc != 3) { return 2; }
    if (strcmp(argv[2], "generic") != 0) {
        if (!ds4_gpu_iquest_policy()) { return 2; }
        if (strcmp(argv[2], "cleaned") == 0) { cuda_weight_policy_reset(); }
    }
    const char model[64] = {};
    int accepted = ds4_gpu_import_model_ipc_manifest(model, sizeof(model), argv[1], "base");
    printf("{\"accepted\":%d,\"canonical\":%d,\"derived\":%d,\"gpu_calls\":%d}\n",
           accepted, canonical, derived, gpu_calls);
    return 0;
}
''')
    return "\n".join(pieces)


class ManifestPolicyTest(unittest.TestCase):
    def test_manifest_policy(self):
        with tempfile.TemporaryDirectory(prefix="ds4-manifest-policy-") as directory:
            work = Path(directory)
            source = work / "test.cc"
            binary = work / "test"
            source.write_text(harness_source())
            subprocess.run(["c++", "-std=c++17", "-O0", str(source), "-o", str(binary)], check=True)
            checked = 0
            handle = "00" * 64
            transports = {
                "ipc": (
                    "DS4_WEIGHT_SERVER_IPC_DERIVED_V1\n",
                    f"range base 64 0 32 {handle}\n",
                    f"derived-range {{model}} 64 0 32 1 4 8 1 64 {handle} tensor\n",
                ),
                "vmm": (
                    "DS4_WEIGHT_SERVER_VMM_DERIVED_V1\nbroker /tmp/mock-broker\n",
                    "alloc 1 base 64 0 32 32\n",
                    "derived-alloc 2 {model} 64 0 32 1 4 8 1 64 64 tensor\n",
                ),
            }
            for mode, transport, (header, canonical, derived) in (
                (mode, transport, records)
                for mode in ("generic", "active", "cleaned")
                for transport, records in transports.items()
            ):
                target = derived.format(model="base")
                other = derived.format(model="base-extra")
                for knob in (None, "1", "0", ""):
                    cases = [
                        ("canonical", canonical, 1, 1, 0),
                        ("other-model", canonical + other, 1, 1, 0),
                        ("comment", "# " + target + canonical, 1, 1, 0),
                    ]
                    if knob is None and mode != "active":
                        cases.append(("derived-allowed", canonical + target, 1, 1, 1))
                    else:
                        cases.extend([
                            ("derived-first", target + canonical, 0, 0, 0),
                            ("derived-last", canonical + target, 0, 0, 0),
                            ("whitespace", canonical + "\t " + target, 0, 0, 0),
                        ])
                        if transport == "ipc":
                            # The retained IPC parser recognizes this prefix too.
                            cases.append(("legacy-prefix", canonical + target.replace(
                                "derived-range ", "derived-range-legacy "), 0, 0, 0))
                    for name, records, accepted, want_canonical, want_derived in cases:
                        checked += 1
                        with self.subTest(mode=mode, transport=transport, knob=knob, case=name):
                            manifest = work / "manifest.txt"
                            manifest.write_text(header + records)
                            env = os.environ.copy()
                            env.pop("DS4_WEIGHT_FP_CHECK", None)
                            env.pop(NO_DERIVED, None)
                            if knob is not None:
                                env[NO_DERIVED] = knob
                            result = subprocess.run([str(binary), str(manifest), mode], env=env,
                                                    capture_output=True, text=True, check=True)
                            actual = json.loads(result.stdout)
                            self.assertEqual(actual["accepted"], accepted, actual)
                            self.assertEqual(actual["canonical"], want_canonical, actual)
                            self.assertEqual(actual["derived"], want_derived, actual)
                            if not accepted:
                                self.assertEqual(actual["gpu_calls"], 0, actual)
            print(json.dumps({"cases": checked, "scope": "production parser with CPU GPU mocks"}))


if __name__ == "__main__":
    unittest.main()
