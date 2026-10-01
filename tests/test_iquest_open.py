#!/usr/bin/env python3
"""CPU admission regression: actual engine-open prefix, mocked metadata and policy.

Stops before weight binding, device allocation or import. No model or GPU is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

C = r'''
#include "ds4.h"
#include "ds4_iquest_ref.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
enum { DS4_MODEL_FAMILY_DEEPSEEK4, DS4_MODEL_FAMILY_QWEN35, DS4_MODEL_FAMILY_IQUEST };
typedef struct { int fd; } ds4_model;
struct ds4_engine {
    ds4_model model, mtp_model, vision_model;
    ds4_backend backend;
    bool quality;
    ds4_distributed_options distributed;
    int power_percent, mtp_draft_tokens;
    float mtp_margin, directional_steering_attn_scale, directional_steering_ffn_scale;
    char *directional_steering_file;
};
static int family, selected_family, g_requested_threads;
static const void *g_host_shape;
static unsigned policy_calls, closes, admitted, checks, failures;
#define DS4_MODEL_FAMILY family
#define DS4_N_LAYER IQ_LAYERS
#define DS4_MAX_LAYER IQ_LAYERS
#define DS4_MODEL_SHAPE_NAME "fixture"
void ds4_gov_modes_init(void) {}
static void *xcalloc(size_t n, size_t size) { return calloc(n, size); }
static char *ds4_strdup(const char *s) { return strdup(s); }
static void ds4_acquire_instance_lock(void) {}
bool ds4_backend_uses_graph(ds4_backend b) { return b != DS4_BACKEND_CPU; }
static void model_open(ds4_model *m, const char *path, bool graph, bool prefetch) {
    (void)path; (void)graph; (void)prefetch; m->fd = -1;
}
static void model_apply_host_shape(void) { family = selected_family; }
static void config_validate_model(ds4_model *m) { (void)m; family = selected_family; }
static void config_apply_qwen35_runtime(ds4_model *m) { (void)m; }
static int ds4_gpu_iquest_policy(void) { policy_calls++; return 1; }
void ds4_engine_close(ds4_engine *e) {
    closes++; free(e->directional_steering_file); free(e);
}
OPEN_PREFIX
    admitted++;
    *out = e;
    return 0;
}
int main(void) {
    const int widths[] = {-1, 0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, INT_MAX};
    printf("{\"cases\":[");
    for (int f = 0; f < 2; f++) {
        for (int host = 0; host < 2; host++) {
            for (int inspect = 0; inspect < 2; inspect++) {
                for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
                    const int width = widths[i];
                    selected_family = f ? DS4_MODEL_FAMILY_IQUEST : DS4_MODEL_FAMILY_DEEPSEEK4;
                    family = -1;
                    g_host_shape = host ? &selected_family : NULL;
                    policy_calls = closes = admitted = 0;
                    ds4_engine_options opt = {.backend=DS4_BACKEND_CUDA,
                        .mtp_draft_tokens=width, .inspect_only=inspect};
                    ds4_engine *e = NULL;
                    const int rc = ds4_engine_open(&e, &opt);
                    const bool reject = f && width > IQ_DRAFT_SLOTS;
                    const int expected = width <= 0 ? 1 : width > 16 ? 16 : width;
                    const bool passed = reject
                        ? rc != 0 && e == NULL && closes == 1 && admitted == 0 && policy_calls == 0
                        : rc == 0 && e != NULL && e->mtp_draft_tokens == expected &&
                          closes == 0 && admitted == 1 && policy_calls == (unsigned)f;
                    if (checks++) { printf(","); }
                    printf("{\"iquest\":%d,\"host_shape\":%d,\"inspect\":%d,"
                           "\"requested\":%d,\"rejected\":%s,\"passed\":%s}",
                           f, host, inspect, width, rc ? "true" : "false", passed ? "true" : "false");
                    failures += !passed;
                    if (e) { ds4_engine_close(e); }
                }
            }
        }
    }
    printf("],\"checks\":%u,\"failed\":%u}\n", checks, failures);
    return failures ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = (ROOT / "ds4.c").read_text()
    start = source.index("int ds4_engine_open(")
    end = source.index(
        "    if (!opt->inspect_only && DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_NAIVE", start
    )
    prefix = source[start:end]
    with tempfile.TemporaryDirectory(prefix="iquest-open-") as tmp:
        tmp = Path(tmp)
        code, binary = tmp / "test.c", tmp / "test"
        code.write_text(C.replace("OPEN_PREFIX\n", prefix))
        subprocess.run([
            "cc", "-D_GNU_SOURCE", "-std=c11", "-O0", "-Wall", "-Wextra",
            "-I", str(ROOT), str(code), "-lm", "-o", str(binary),
        ], check=True)
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    report = json.loads(run.stdout)
    report.update({
        "scope": "Production engine-open prefix through IQuest admission; metadata/policy stubs, no weight binding, device allocation or import.",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
