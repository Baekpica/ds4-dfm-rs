#!/usr/bin/env python3
"""Run Gemma's native MTP enable/trial logic with deterministic GPU stubs.

One draft means two verify rows. Exercise that default as well as wider
rounds and the token/context limits, without loading a model.
"""
import subprocess
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
engine = (repo / "ds4.c").read_text()
start = engine.index("bool ds4_engine_has_mtp(")
start = engine.index("    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GEMMA4)", start)
end = engine.index("    return e->mtp_ready;", start)
enable = engine[start:end]
source = (repo / "ds4_gemma4_mtp.inc").read_text()
start = source.index("static int ds4_session_gemma4_trial(")
trial = source[start:source.index("\n}\n", start) + 3]
code = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_gemma4_plan.h"

enum { DS4_MODEL_FAMILY_GEMMA4, DS4_BACKEND_CUDA };
#define DS4_MODEL_FAMILY DS4_MODEL_FAMILY_GEMMA4
#define DS4_N_VOCAB G4_VOCAB
typedef struct { int unused; } ds4_model;
typedef struct { int unused; } ds4_weights;
typedef struct { int unused; } ds4_gemma4_mtp;
typedef struct {
    int backend, mtp_draft_tokens;
    bool mtp_ready;
    struct { ds4_gemma4_mtp *gemma4_mtp; } mtp_weights;
    ds4_model model;
    ds4_weights weights;
} ds4_engine;
typedef struct { uint32_t position, ctx; bool failed; } ds4_gemma4_graph;
typedef struct {
    uint32_t drafts, trial_n, trial_start;
    bool hidden_valid;
    int trial[G4_MTP_DRAFTS + 1];
    void *positions, *verify;
    float *host;
} ds4_gemma4_mtp_runtime;
typedef struct {
    ds4_engine *engine;
    ds4_gemma4_graph *gemma4_graph;
    ds4_gemma4_mtp_runtime *gemma4_mtp;
    bool gemma4_graph_ready, checkpoint_valid;
    struct { int len; } checkpoint;
    int spec;
} ds4_session;

static bool enabled(ds4_engine *e) {
ENABLE
    return false;
}
static bool ds4_session_is_gemma4(ds4_session *s) { return s != NULL; }
static void payload_set_err(char *err, size_t len, const char *msg) {
    snprintf(err, len, "%s", msg);
}
static int gemma4_session_fail(ds4_session *s, char *err, size_t len) { return 1; }
static int ds4_gpu_tensor_write(void *dst, uint64_t off, const void *src, uint64_t n) {
    memcpy((char *)dst + off, src, n);
    return 1;
}
static int ds4_gpu_tensor_read(void *src, uint64_t off, void *dst, uint64_t n) {
    assert(false); /* Greedy trials do not read back full verify logits. */
    return 0;
}
static unsigned steps, verified;
static bool gemma4_mtp_step(ds4_gemma4_mtp_runtime *r, ds4_model *m,
        ds4_weights *w, const ds4_gemma4_mtp *b, int token, uint32_t pos, int *next) {
    assert(pos == *(uint32_t *)r->positions);
    steps++;
    *next = token + 1;
    return true;
}
static bool gemma4_mtp_verify(ds4_gemma4_mtp_runtime *r, ds4_model *m,
        ds4_weights *w, const int *tokens, uint32_t n, uint32_t start, int *target) {
    verified = n;
    for (uint32_t i = 0; i < n; i++) { target[i] = tokens[i] + 1; }
    return true;
}
static bool spec_sampling(void *spec) { return false; }
static void spec_sample_rows(void *spec, const float *rows, uint32_t stride,
        uint32_t vocab, const int *tokens, uint32_t n, int *target) { assert(false); }
typedef struct { uint64_t spec_drafts; } metrics_t;
static metrics_t metrics;
static metrics_t *ds4_metrics_get(void) { return &metrics; }
static void ds4_metric_add(uint64_t *n, uint64_t delta) { *n += delta; }
TRIAL

static bool check(unsigned drafts, int limit, int capacity, uint32_t remaining,
                  bool hidden, unsigned expected) {
    ds4_gemma4_mtp weights = {0};
    ds4_engine e = {.backend = DS4_BACKEND_CUDA, .mtp_draft_tokens = (int)drafts,
                    .mtp_ready = true, .mtp_weights.gemma4_mtp = &weights};
    const bool armed = enabled(&e) == (drafts > 0);
    if (!armed) {
        fprintf(stderr, "draft %u: engine enable mismatch\n", drafts);
    }
    uint32_t pos = 0;
    ds4_gemma4_graph g = {.position = 10u, .ctx = 10u + remaining};
    ds4_gemma4_mtp_runtime r = {.drafts = drafts, .hidden_valid = hidden, .positions = &pos};
    ds4_session s = {.engine = &e, .gemma4_graph = &g, .gemma4_mtp = &r,
                     .gemma4_graph_ready = true, .checkpoint_valid = true,
                     .checkpoint.len = 10};
    int tokens[G4_MTP_DRAFTS + 1] = {0}, target[G4_MTP_DRAFTS + 1] = {0};
    char err[128] = {0};
    steps = verified = 0;
    metrics.spec_drafts = 0;
    const int n = ds4_session_gemma4_trial(&s, 42, limit, tokens, target, capacity,
                                         err, sizeof(err));
    if (n != (int)expected || verified != expected ||
        steps != (expected ? expected - 1u : 0u) || metrics.spec_drafts != steps ||
        r.trial_n != expected || g.position != 10u + expected) {
        fprintf(stderr, "draft %u: trial=%d steps=%u verified=%u expected=%u: %s\n",
                drafts, n, steps, verified, expected, err);
        return false;
    }
    for (unsigned i = 0; i < expected; i++) {
        assert(tokens[i] == 42 + (int)i && target[i] == tokens[i] + 1);
        assert(r.trial[i] == tokens[i] && r.trial_start == 10u);
    }
    return armed;
}
int main(void) {
    unsetenv("DS4_MTP_SPEC_DISABLE");
    bool ok = true;
    for (unsigned d = 0; d <= G4_MTP_DRAFTS; d++) {
        ok &= check(d, 7, 7, 20u, true, d ? d + 1u : 0u);
    }
    ok &= check(6, 2, 7, 20u, true, 2u);
    ok &= check(6, 7, 3, 20u, true, 3u);
    ok &= check(6, 7, 7, 2u, true, 2u);
    ok &= check(1, 1, 7, 20u, true, 0u);
    ok &= check(1, 7, 1, 20u, true, 0u);
    ok &= check(1, 7, 7, 1u, true, 0u);
    ok &= check(1, 7, 7, 20u, false, 0u);
    ds4_gemma4_mtp weights = {0};
    ds4_engine e = {.backend = DS4_BACKEND_CUDA, .mtp_draft_tokens = 1,
                    .mtp_ready = true, .mtp_weights.gemma4_mtp = &weights};
    setenv("DS4_MTP_SPEC_DISABLE", "1", 1);
    assert(!enabled(&e));
    unsetenv("DS4_MTP_SPEC_DISABLE");
    e.mtp_ready = false;
    assert(!enabled(&e));
    e.mtp_ready = true;
    e.mtp_weights.gemma4_mtp = NULL;
    assert(!enabled(&e));
    if (ok) { puts("gemma4 MTP controls: ok"); }
    return ok ? 0 : 1;
}
'''.replace("ENABLE", enable).replace("TRIAL", trial)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)
    (path / "mtp.c").write_text(code)
    subprocess.run([
        "cc", "-std=c99", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra",
        "-Werror", "-Wno-unused-parameter", "-I", str(repo),
        str(path / "mtp.c"), "-o", str(path / "mtp"),
    ], check=True)
    raise SystemExit(subprocess.run([str(path / "mtp")]).returncode)
