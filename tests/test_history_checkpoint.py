"""Execute the native history-cut admission and prefill block without a model.

Only GPU forward/capture calls are simulated. The actual ds4.c block must
split before the generation suffix, publish matching tokens, then capture.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


class HistoryCheckpoint(unittest.TestCase):
    def test_family_history_cut(self):
        source = (Path(__file__).resolve().parents[1] / "ds4.c").read_text()
        function = source[source.index("static int family_banked_engine_continuous_generate("):]
        admission = re.search(r"cb->checkpoint_at = .*?;", function, re.S).group()
        start = function.index("const uint32_t pos = cb->prefill_base + cb->prefill_off;")
        end = function.index("ds4_metric_add(", start)
        chunk = function[start:end]
        program = r'''#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
#define FCG_ERR(...) ((void)0)
typedef struct {
    bool step37, motif3;
    uint32_t bank_gen[1], bank_hist_valid[1], kv_len, hist_len, capture_len;
    int kv[700], history[700], callbacks, captures, fault_at;
    bool logits_valid;
} context;
typedef struct {
    uint32_t prefill_base, prefill_len, prefill_off, checkpoint_at;
    const int *prefill;
    void *user;
    void (*on_checkpoint)(void *, void *, int, int);
} bank;
static int failed;
static bool family_banked_prefill(context *ctx, uint32_t bank_id, const int *tokens,
                                  uint32_t n, uint32_t pos, bool final,
                                  const int *next, uint32_t next_n) {
    (void)bank_id; (void)next; (void)next_n;
    if (ctx->fault_at >= 0 && pos + n >= (uint32_t)ctx->fault_at) return false;
    if (pos != ctx->kv_len) failed++;
    memcpy(ctx->kv + pos, tokens, n * sizeof(int));
    ctx->kv_len = pos + n;
    ctx->logits_valid = final;
    return true;
}
static void bank_hist_append_n(context *ctx, uint32_t bank_id, const int *tokens, uint32_t n) {
    (void)bank_id;
    memcpy(ctx->history + ctx->hist_len, tokens, n * sizeof(int));
    ctx->hist_len += n;
}
static void family_banked_capture_checkpoint(context *ctx, uint32_t bank_id, uint32_t pos, bool logits) {
    (void)bank_id;
    if (pos != ctx->kv_len || pos != ctx->hist_len || !logits || !ctx->logits_valid ||
        memcmp(ctx->kv, ctx->history, pos * sizeof(int))) failed++;
    ctx->capture_len = pos;
    ctx->captures++;
}
static void checkpoint(void *ud, void *user, int bank_id, int pos) {
    (void)user; (void)bank_id;
    context *ctx = ud;
    if (ctx->capture_len != (uint32_t)pos || ctx->captures != 1) failed++;
    ctx->callbacks++;
}
static int run(int family, int cut, uint32_t cached, uint32_t cap, int fault_at) {
    context state = {.step37 = family == 1, .motif3 = family == 2,
        .kv_len = cached, .hist_len = cached, .fault_at = fault_at};
    context *ctx = &state;
    int tokens[620];
    for (unsigned i = 0; i < 620; i++) tokens[i] = (int)i + 17;
    memcpy(ctx->kv, tokens, cached * sizeof(int));
    memcpy(ctx->history, tokens, cached * sizeof(int));
    struct {int checkpoint_at, n;} req = {cut, 620};
    bank b = {.prefill_base = cached, .prefill_len = 620 - cached,
        .prefill = tokens + cached, .on_checkpoint = checkpoint};
    bank *cb = &b;
    ADMISSION
    bool expected = family != 0 && cut > (int)cached && cut < req.n;
    CHECK(cb->checkpoint_at == (expected ? (uint32_t)cut : 0));
    uint32_t pb = 0;
    bool ok = true;
    void *ud = ctx;
    while (ok && cb->prefill_off < cb->prefill_len) {
        const uint32_t remain = cb->prefill_len - cb->prefill_off;
        uint32_t n = remain < cap ? remain : cap;
        CHUNK
    }
    CHECK(!failed);
    if (fault_at >= 0) {
        CHECK(!ok && !ctx->callbacks && !ctx->captures && !ctx->bank_hist_valid[0]);
        CHECK(ctx->kv_len == ctx->hist_len && ctx->hist_len < (uint32_t)fault_at);
    } else {
        CHECK(ok && ctx->kv_len == 620 && ctx->hist_len == 620 && ctx->logits_valid);
        CHECK(!memcmp(tokens, ctx->kv, sizeof(tokens)));
        CHECK(ctx->callbacks == (int)expected && ctx->captures == (int)expected);
        CHECK(ctx->capture_len == (expected ? (uint32_t)cut : 0));
    }
    return 0;
}
int main(void) {
    const uint32_t caps[] = {48, 64, 512};
    for (int family = 0; family <= 2; family++) {
        for (unsigned c = 0; c < sizeof(caps) / sizeof(*caps); c++) {
            CHECK(!run(family, 593, 0, caps[c], -1));
            CHECK(!run(family, 593, 576, caps[c], -1));
            CHECK(!run(family, 593, 593, caps[c], -1));
            CHECK(!run(family, 0, 0, caps[c], -1));
            CHECK(!run(family, 620, 0, caps[c], -1));
            CHECK(!run(family, 593, 0, caps[c], 593));
        }
    }
    return 0;
}
'''.replace("ADMISSION", admission).replace("CHUNK", chunk)
        with tempfile.TemporaryDirectory(prefix="ds4-history-cut-") as tmp:
            path = Path(tmp) / "cut.c"
            path.write_text(program)
            binary = Path(tmp) / "cut"
            subprocess.run([*shlex.split(os.environ.get("CC", "cc")), "-std=c11",
                            "-Wall", "-Wextra", "-Werror", str(path), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
