#!/usr/bin/env python3
"""Compile actual IQuest trial/commit with tiny CPU tensors and controlled logits.

This tests confidence policy and state plumbing, not model math or GPU kernels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[1]


def function(source, name, declaration="int "):
    start = source.index(declaration + name + "(")
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


C = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef enum { IQ_NO_LOGITS, IQ_HAS_LOGITS } iquest_logits_mode;
typedef enum { IQ_MTP_OFF, IQ_MTP_ON } iquest_mtp_mode;
#define IQ_VOCAB 4
#define IQ_EMBED 2
#define IQ_LAYERS 2
#define IQ_DRAFT_SLOTS 7
#define IQ_DRAFT_WINDOW 4
#define IQ_Q8_ROW_BLOCKS 1
#define DS4_NEG_INF (-1.0e30f)
#define DS4_MAYBE_UNUSED
typedef struct { unsigned char bytes[8]; } iquest_q8;
typedef struct { unsigned char data[256]; uint64_t bytes; } ds4_gpu_tensor;
typedef struct { int unused; } ds4_model;
typedef struct { int unused; } ds4_weights;
typedef struct {
    bool failed; unsigned trial_n, position, mtp_position, context, trial_base;
    ds4_gpu_tensor *mtp_backup, *mtp_kv, *carry, *draft_carry, *mtp_logits;
    ds4_gpu_tensor *logits, *trial_kv, *trial_mtp, *trial_hidden, *kv[IQ_LAYERS];
    unsigned kv_cap[IQ_LAYERS]; float *draft_logits, *trial_logits;
    int trial_tokens[IQ_DRAFT_SLOTS + 1];
} ds4_iquest_graph;
typedef struct { int len, cap, v[32]; } token_vec;
typedef struct { ds4_model model; ds4_weights weights; int mtp_draft_tokens; float mtp_margin; } ds4_engine;
typedef struct {
    ds4_engine *engine; bool checkpoint_valid, iquest_graph_ready;
    token_vec checkpoint; ds4_iquest_graph iquest_graph; float logits[IQ_VOCAB];
} ds4_session;
typedef struct { uint64_t spec_drafts, spec_hits; } metrics;
static metrics stats;
static ds4_engine eng; static ds4_session session;
static ds4_gpu_tensor pool[32]; static unsigned pool_n;
static float draft[IQ_VOCAB], trial[(IQ_DRAFT_SLOTS + 1) * IQ_VOCAB];
static unsigned draft_calls, target_calls, copies, failures, checks;
static int read_fail, forward_fail, nonfinite_at, ties;
static float gaps[IQ_DRAFT_SLOTS];
static ds4_gpu_tensor *tensor(uint64_t bytes) {
    ds4_gpu_tensor *t = &pool[pool_n++]; t->bytes = bytes;
    for (unsigned i = 0; i < bytes; i++) { t->data[i] = (unsigned char)(pool_n * 7 + i); }
    return t;
}
static bool ds4_session_is_iquest(ds4_session *s) { return s != NULL; }
static bool ds4_engine_has_mtp(ds4_engine *e) { return e != NULL; }
static void payload_set_err(char *err, size_t n, const char *msg) { snprintf(err, n, "%s", msg); }
static uint64_t ds4_gpu_tensor_bytes(ds4_gpu_tensor *t) { return t->bytes; }
static bool ds4_gpu_tensor_copy(ds4_gpu_tensor *d, uint64_t doff, ds4_gpu_tensor *s, uint64_t soff, uint64_t n) {
    if (doff + n > d->bytes || soff + n > s->bytes) { abort(); }
    memmove(d->data + doff, s->data + soff, (size_t)n); copies++; return true;
}
static bool ds4_gpu_tensor_read(ds4_gpu_tensor *t, uint64_t off, void *out, uint64_t n) {
    if (read_fail && t == session.iquest_graph.mtp_logits) { return false; }
    if (off + n > t->bytes) { abort(); } memcpy(out, t->data + off, (size_t)n); return true;
}
static bool ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *src, uint64_t n) {
    if (off + n > t->bytes) { abort(); } memcpy(t->data + off, src, (size_t)n); return true;
}
static bool ds4_gpu_synchronize(void) { return true; }
static metrics *ds4_metrics_get(void) { return &stats; }
static void ds4_metric_add(uint64_t *x, uint64_t n) { *x += n; }
static int sample_argmax(const float *x, unsigned n) {
    unsigned id = 0; for (unsigned i = 1; i < n; i++) { if (x[i] > x[id]) { id = i; } } return (int)id;
}
TOP2
static int iquest_session_fail(ds4_session *s, char *err, size_t n) {
    s->iquest_graph.failed = true; s->checkpoint_valid = false; s->checkpoint.len = 0;
    payload_set_err(err, n, "controlled failure"); return 1;
}
static void token_vec_push(token_vec *v, int t) { v->v[v->len++] = t; }
static bool iquest_mtp_forward(ds4_iquest_graph *g, const ds4_model *m, const ds4_weights *w,
    const int *tokens, const ds4_gpu_tensor *prior, unsigned n, unsigned pos, iquest_logits_mode need_logits) {
    (void)m; (void)w; (void)tokens; (void)prior; (void)need_logits;
    if (n != 1 || pos != g->mtp_position) { abort(); }
    unsigned k = draft_calls++; if (forward_fail) { return false; }
    memset(g->mtp_kv->data + (pos % (IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS)) * 8, 180 + k, 8);
    memset(g->draft_carry->data, 70 + k, g->draft_carry->bytes);
    float x[IQ_VOCAB] = {-20, 10, 10 - gaps[k], -30};
    if (ties) { x[0] = 10; x[1] = 10; x[2] = -20; }
    if (nonfinite_at == (int)k) { x[3] = NAN; }
    memcpy(g->mtp_logits->data, x, sizeof(x)); g->mtp_position = pos + 1; return true;
}
static bool iquest_forward(ds4_iquest_graph *g, const ds4_model *m, const ds4_weights *w,
    const int *tokens, unsigned n, unsigned pos, iquest_mtp_mode warm) {
    (void)m; (void)w; (void)tokens; (void)warm;
    if (n != 1 || pos != g->position || g->mtp_position != pos - 1) { abort(); }
    for (unsigned i = 0; i < IQ_LAYERS; i++) {
        memset(g->kv[i]->data + (pos % g->kv_cap[i]) * 8, 100 + target_calls + i, 8);
    }
    memset(g->mtp_kv->data + ((pos - 1) % (IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS)) * 8, 90 + target_calls, 8);
    memset(g->carry->data, 30 + target_calls, g->carry->bytes);
    float x[IQ_VOCAB] = {-20, 10, 0, -30}; memcpy(g->logits->data, x, sizeof(x));
    target_calls++; g->position = pos + 1; g->mtp_position = pos; return true;
}
TRIAL
COMMIT
static void init(float margin) {
    memset(&session, 0, sizeof(session)); memset(&eng, 0, sizeof(eng)); memset(pool, 0, sizeof(pool));
    memset(&stats, 0, sizeof(stats)); pool_n = 0; draft_calls = target_calls = copies = 0;
    read_fail = forward_fail = ties = 0; nonfinite_at = -1;
    for (unsigned i = 0; i < IQ_DRAFT_SLOTS; i++) { gaps[i] = 5; }
    eng.mtp_margin = margin; eng.mtp_draft_tokens = 7;
    session.engine = &eng; session.checkpoint_valid = session.iquest_graph_ready = true;
    session.checkpoint.len = 4; session.checkpoint.cap = 32;
    ds4_iquest_graph *g = &session.iquest_graph;
    g->position = 4; g->mtp_position = 3; g->context = 16; g->trial_base = 123;
    g->mtp_kv = tensor((IQ_DRAFT_WINDOW + IQ_DRAFT_SLOTS) * 8); g->mtp_backup = tensor(g->mtp_kv->bytes);
    g->carry = tensor(IQ_EMBED * sizeof(float)); g->draft_carry = tensor(IQ_EMBED * sizeof(float));
    g->mtp_logits = tensor(IQ_VOCAB * sizeof(float)); g->logits = tensor(IQ_VOCAB * sizeof(float));
    g->trial_kv = tensor((IQ_DRAFT_SLOTS + 1) * IQ_LAYERS * 8);
    g->trial_mtp = tensor((IQ_DRAFT_SLOTS + 1) * 8); g->trial_hidden = tensor((IQ_DRAFT_SLOTS + 1) * IQ_EMBED * sizeof(float));
    for (unsigned i = 0; i < IQ_LAYERS; i++) { g->kv[i] = tensor(8 * 8); g->kv_cap[i] = 8; }
    g->draft_logits = draft; g->trial_logits = trial;
}
static void check(const char *name, bool pass) {
    if (checks++) { printf(","); } printf("{\"name\":\"%s\",\"passed\":%s}", name, pass ? "true" : "false"); failures += !pass;
}
int main(void) {
    int tokens[8], targets[8]; char err[128]; int n; unsigned char original[256], carry[8], kv[IQ_LAYERS][64];
    printf("{\"cases\":[");
    init(3); gaps[0] = 2.999f;
    memcpy(original, session.iquest_graph.mtp_kv->data, session.iquest_graph.mtp_kv->bytes);
    memcpy(carry, session.iquest_graph.carry->data, sizeof(carry));
    for (unsigned i = 0; i < IQ_LAYERS; i++) { memcpy(kv[i], session.iquest_graph.kv[i]->data, 64); }
    n = ds4_session_iquest_trial(&session, 0, 4, tokens, targets, 8, err, sizeof(err));
    check("below_threshold_falls_back", n == 0 && draft_calls == 1 && target_calls == 0 && stats.spec_drafts == 0);
    bool restored = !memcmp(original, session.iquest_graph.mtp_kv->data, session.iquest_graph.mtp_kv->bytes) &&
        !memcmp(carry, session.iquest_graph.carry->data, sizeof(carry));
    for (unsigned i = 0; i < IQ_LAYERS; i++) { restored &= !memcmp(kv[i], session.iquest_graph.kv[i]->data, 64); }
    check("first_cutoff_restores_bytes_and_frontiers", restored && session.iquest_graph.position == 4 &&
        session.iquest_graph.mtp_position == 3 && session.iquest_graph.trial_n == 0 &&
        session.iquest_graph.trial_base == 123 && session.checkpoint.len == 4 && session.checkpoint_valid);
    init(3); gaps[0] = 3; n = ds4_session_iquest_trial(&session, 0, 2, tokens, targets, 8, err, sizeof(err));
    check("equal_threshold_admitted", n == 2 && tokens[1] == 1 && target_calls == 2 && stats.spec_drafts == 1);
    init(3); gaps[0] = 3.01f; n = ds4_session_iquest_trial(&session, 0, 2, tokens, targets, 8, err, sizeof(err));
    check("above_threshold_admitted", n == 2 && target_calls == 2);
    init(0); ties = 1; n = ds4_session_iquest_trial(&session, 0, 2, tokens, targets, 8, err, sizeof(err));
    check("zero_margin_tie_admitted_stable_id", n == 2 && tokens[1] == 0 && target_calls == 2);
    init(0); gaps[0] = 0.01f; n = ds4_session_iquest_trial(&session, 0, 4, tokens, targets, 8, err, sizeof(err));
    check("zero_margin_full_prefix", n == 4 && draft_calls == 3 && target_calls == 4);
    init(3); gaps[2] = 2; n = ds4_session_iquest_trial(&session, 0, 5, tokens, targets, 8, err, sizeof(err));
    check("later_cutoff_contiguous_prefix", n == 3 && draft_calls == 3 && target_calls == 3 && stats.spec_drafts == 2 &&
        session.iquest_graph.trial_n == 3 && session.iquest_graph.position == 7 && session.iquest_graph.mtp_position == 6);
    init(3); n = ds4_session_iquest_trial(&session, 0, 5, tokens, targets, 2, err, sizeof(err));
    check("capacity_limit", n == 2 && draft_calls == 1 && target_calls == 2);
    init(3); eng.mtp_draft_tokens = 2; n = ds4_session_iquest_trial(&session, 0, 5, tokens, targets, 8, err, sizeof(err));
    check("depth_limit", n == 3 && draft_calls == 2 && target_calls == 3);
    init(3); session.iquest_graph.context = 6; n = ds4_session_iquest_trial(&session, 0, 5, tokens, targets, 8, err, sizeof(err));
    check("context_limit", n == 2 && draft_calls == 1 && target_calls == 2);
    init(3); nonfinite_at = 0; n = ds4_session_iquest_trial(&session, 0, 3, tokens, targets, 8, err, sizeof(err));
    check("nonfinite_draft_invalidates", n < 0 && target_calls == 0 && session.iquest_graph.failed && !session.checkpoint_valid);
    init(3); read_fail = 1; n = ds4_session_iquest_trial(&session, 0, 3, tokens, targets, 8, err, sizeof(err));
    check("read_failure_invalidates", n < 0 && target_calls == 0 && session.iquest_graph.failed && !session.checkpoint_valid);
    init(3); forward_fail = 1; n = ds4_session_iquest_trial(&session, 0, 3, tokens, targets, 8, err, sizeof(err));
    check("draft_forward_failure_invalidates", n < 0 && target_calls == 0 && session.iquest_graph.failed && !session.checkpoint_valid);
    init(3); gaps[2] = 2;
    for (unsigned i = 0; i < IQ_LAYERS; i++) { memcpy(kv[i], session.iquest_graph.kv[i]->data, 64); }
    memcpy(original, session.iquest_graph.mtp_kv->data, session.iquest_graph.mtp_kv->bytes);
    n = ds4_session_iquest_trial(&session, 0, 5, tokens, targets, 8, err, sizeof(err));
    int rc = n > 0 ? ds4_session_iquest_commit(&session, 1, err, sizeof(err)) : -1;
    bool rejected_restored = true;
    for (unsigned k = 1; k < (unsigned)(n > 0 ? n : 0); k++) {
        unsigned p = 4 + k;
        for (unsigned i = 0; i < IQ_LAYERS; i++) {
            rejected_restored &= !memcmp(kv[i] + (p % 8) * 8, session.iquest_graph.kv[i]->data + (p % 8) * 8, 8);
        }
        rejected_restored &= !memcmp(original + ((p - 1) % 11) * 8, session.iquest_graph.mtp_kv->data + ((p - 1) % 11) * 8, 8);
    }
    check("short_commit_restores_rejected_rows", n == 3 && rc == 0 && rejected_restored &&
        session.iquest_graph.position == 5 && session.iquest_graph.mtp_position == 4 &&
        session.iquest_graph.trial_n == 0 && session.checkpoint.len == 5 && stats.spec_hits == 0);
    for (unsigned i = 0; i < IQ_DRAFT_SLOTS; i++) { gaps[i] = 5; }
    draft_calls = target_calls = 0;
    n = ds4_session_iquest_trial(&session, 1, 2, tokens, targets, 8, err, sizeof(err));
    check("resume_after_short_commit", n == 2 && draft_calls == 1 && target_calls == 2 && session.iquest_graph.trial_base == 5);
    printf("],\"failed\":%u,\"passed\":%s}\n", failures, failures ? "false" : "true");
    return failures ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    path = REPO / "ds4_iquest_session.inc"
    source = path.read_text()
    trial = function(source, "ds4_session_iquest_trial")
    commit = function(source, "ds4_session_iquest_commit")
    common = (REPO / "ds4.c").read_text()
    top2 = function(common, "logits_top2", "static DS4_MAYBE_UNUSED void ")
    code = (C.replace("TRIAL\n", trial + "\n").replace("COMMIT\n", commit + "\n")
             .replace("TOP2\n", top2 + "\n"))
    with tempfile.TemporaryDirectory(prefix="iquest-margin-") as tmp:
        tmp = Path(tmp)
        (tmp / "test.c").write_text(code)
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wno-unused-function",
                        str(tmp / "test.c"), "-lm", "-o", str(tmp / "test")], check=True)
        run = subprocess.run([str(tmp / "test")], capture_output=True, text=True)
    report = json.loads(run.stdout)
    report.update({
        "scope": "Actual extracted production trial and commit; 4-token vocabulary, 2 mocked layers, CPU tensor-copy stubs. No model inference or GPU math.",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "trial_function_sha256": hashlib.sha256(trial.encode()).hexdigest(),
        "commit_function_sha256": hashlib.sha256(commit.encode()).hexdigest(),
        "top2_function_sha256": hashlib.sha256(top2.encode()).hexdigest(),
        "common_source_sha256": hashlib.sha256(common.encode()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
