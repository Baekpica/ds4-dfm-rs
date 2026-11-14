#!/usr/bin/env python3
"""Exercise the production one-row projection dispatch over recording stubs."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ds4_inkling2_graph.inc").read_text()
start = source.index("static bool inkling2_mm(")
body = source[start:source.index("static bool inkling2_mm1(", start)]
code = r"""
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { int unused; } ds4_gpu_tensor;
typedef struct { uint32_t index; } ds4_tensor;
typedef struct { int unused; } ds4_model;
typedef struct { void *gvq; } ds4_inkling2_graph;
enum { DS4_GVQ_BF16 = 0 };
static int grouped_calls, linear_calls, accepts, fallback;
static uint32_t want_count, want_rows;
static ds4_gpu_tensor input, outputs[4], *outs[4];
static uint32_t inkling2_index(const ds4_model *m, const ds4_tensor *w) {
    (void)m; return w->index;
}
static void check(void *ctx, ds4_gpu_tensor *const *out, const uint32_t *ids,
        uint32_t count, const ds4_gpu_tensor *x, uint32_t rows) {
    assert(ctx == &input && x == &input && count == want_count && rows == want_rows);
    for (uint32_t i = 0; i < count; i++) { assert(out[i] == &outputs[i] && ids[i] == 100 + i); }
}
static int ds4_gpu_gvq_q4_decode_many(void *ctx, ds4_gpu_tensor *const *out,
        const uint32_t *ids, uint32_t count, const ds4_gpu_tensor *x, uint32_t rows) {
    check(ctx, out, ids, count, x, rows); grouped_calls++; return accepts;
}
static int ds4_gpu_gvq_linear_many(void *ctx, ds4_gpu_tensor *const *out,
        const uint32_t *ids, uint32_t count, const ds4_gpu_tensor *x, uint32_t rows, int math) {
    check(ctx, out, ids, count, x, rows); assert(math == DS4_GVQ_BF16); linear_calls++; return fallback;
}
@BODY@
int main(int argc, char **argv) {
    assert(argc == 8);
    want_count = (uint32_t)atoi(argv[1]); want_rows = (uint32_t)atoi(argv[2]);
    accepts = atoi(argv[3]); fallback = atoi(argv[4]);
    ds4_inkling2_graph graph = {&input}; ds4_model model = {0};
    ds4_tensor weights[4]; const ds4_tensor *ws[4];
    for (uint32_t i = 0; i < 4; i++) { weights[i].index = 100 + i; ws[i] = &weights[i]; outs[i] = &outputs[i]; }
    assert(inkling2_mm(&graph, &model, outs, ws, want_count, &input, want_rows) == atoi(argv[7]));
    assert(grouped_calls == atoi(argv[5]) && linear_calls == atoi(argv[6]));
}
""".replace("@BODY@", body)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    src, exe = tmp / "projection.c", tmp / "projection"
    src.write_text(code)
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(src), "-o", str(exe)], check=True)
    cases = 0
    for mode in (None, "0"):
        env = os.environ.copy()
        env.pop("DS4_INKLING2_Q4_GROUP", None)
        if mode is not None:
            env["DS4_INKLING2_Q4_GROUP"] = mode
        for count in (1, 2, 3, 4):
            for rows in (1, 2, 4, 8, 9, 2048):
                for accepts, fallback in ((1, 1), (0, 1), (0, 0)):
                    grouped = int(rows == 1 and mode is None)
                    linear = int(not (grouped and accepts))
                    ok = int(bool((grouped and accepts) or fallback))
                    args = (count, rows, accepts, fallback, grouped, linear, ok)
                    subprocess.run([str(exe), *map(str, args)], check=True, env=env, stderr=subprocess.DEVNULL)
                    cases += 1
    print(f"Inkling-2 projection dispatch: {cases} cases passed")
