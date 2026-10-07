#!/usr/bin/env python3
"""Compile the production boot active-set filter with model-free spans."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


class UnitPlanTests(unittest.TestCase):
    def test_boot_active_set(self):
        source = (Path(__file__).resolve().parents[1] / "ds4.c").read_text()
        start = source.index("ds4_model_map_span_vec sl_spans = {NULL, 0, 0, 0};")
        end = source.index("free(sl_spans.v);", start) + len("free(sl_spans.v);")
        block = source[start:end]
        fixture = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { uint64_t off, end; } span;
typedef struct { span *v; uint32_t len, cap; uint64_t max_tensor_bytes; } ds4_model_map_span_vec;
typedef struct { uint64_t abs_offset, bytes; } ds4_tensor;
typedef struct { ds4_tensor *tensors; uint8_t *tensor_traits; } ds4_model;
typedef struct { uint64_t off, bytes; uint8_t traits, active; } ds4_unit_tensor_in;
static unsigned mandatory_calls, slice_calls;
static bool glm53_stream_spans(const void *weights, ds4_model_map_span_vec *s) {
    (void)weights;
    mandatory_calls++;
    s->v = malloc(2 * sizeof(*s->v));
    s->v[0] = (span){0, 16};
    s->v[1] = (span){48, 64};
    s->len = 2;
    return true;
}
static bool weights_model_map_spans(const void *weights, unsigned lo,
                                    unsigned hi, bool output,
                                    ds4_model_map_span_vec *s) {
    (void)weights; (void)lo; (void)hi; (void)output;
    slice_calls++;
    s->v = malloc(sizeof(*s->v));
    s->v[0] = (span){16, 32};
    s->len = 1;
    return true;
}
static void run(unsigned si, bool sliced, bool ssd, const unsigned expected[4]) {
    ds4_tensor tensors[] = {{0,16}, {16,16}, {32,16}, {48,16}};
    uint8_t traits[] = {0,0,1,0};
    ds4_model model = {tensors,traits};
    const ds4_model *mm = &model;
    const uint32_t n = 4;
    ds4_unit_tensor_in tin[4] = {0};
    const struct { bool sliced; } uts[] = {{sliced},{sliced},{sliced}};
    struct { int weights; } engine = {0};
    const struct { int weights; } *unused_engine = NULL;
    (void)unused_engine;
    __typeof__(engine) *e = &engine;
    struct { bool ssd_streaming; } options = {ssd};
    __typeof__(options) *opt = &options;
    unsigned load_layer_start = 0, load_layer_end = 1;
    bool load_output = false;
    (void)si; (void)opt;
    /* PRODUCTION */
    for (unsigned i = 0; i < n; i++) {
        if (tin[i].active != expected[i]) {
            fprintf(stderr,"source=%u sliced=%u ssd=%u tensor=%u active=%u expected=%u\n",
                    si,sliced,ssd,i,tin[i].active,expected[i]);
            exit(1);
        }
        if (tin[i].off != tensors[i].abs_offset || tin[i].bytes != tensors[i].bytes ||
            tin[i].traits != traits[i]) { exit(2); }
    }
}
int main(void) {
    const unsigned mandatory[] = {1,0,0,1};
    const unsigned all[] = {1,1,1,1};
    const unsigned slice[] = {0,1,0,0};
    run(0,false,true,mandatory);
    if (mandatory_calls != 1 || slice_calls != 0) { return 3; }
    run(0,false,false,all);
    run(0,true,false,slice);
    run(1,false,true,all);
    run(2,false,true,all);
    if (mandatory_calls != 1 || slice_calls != 1) { return 4; }
    puts("GLM SSD boot active set: PASS");
    return 0;
}
'''.replace("/* PRODUCTION */", block)
        with tempfile.TemporaryDirectory() as directory:
            c = Path(directory) / "unit_plan.c"
            binary = Path(directory) / "unit_plan"
            c.write_text(fixture)
            subprocess.run([os.environ.get("CC", "cc"), "-std=gnu11", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                            str(c), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
