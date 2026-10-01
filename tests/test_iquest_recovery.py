#!/usr/bin/env python3
"""Actual IQuest commit/failure/reset/sync control flow with CPU tensor stubs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from test_iquest_sync import function

ROOT = Path(__file__).resolve().parents[1]

C = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
enum { IQ_VOCAB=4, IQ_EMBED=2, IQ_LAYERS=1, IQ_Q8_ROW_BLOCKS=1,
       IQ_DRAFT_WINDOW=4, IQ_DRAFT_SLOTS=7, BASE=4, TRIAL=2, GENERATION=9 };
typedef enum { IQ_MTP_OFF, IQ_MTP_ON } iquest_mtp_mode;
typedef enum { NO_FAULT, COPY_FAULT, WRITE_FAULT, SYNC_FAULT } fault;
typedef uint64_t iquest_q8;
typedef struct { uint8_t data[256]; } ds4_gpu_tensor;
typedef struct { int unused; } ds4_model;
typedef struct { int unused; } ds4_weights;
typedef struct { int *v, len, cap; } ds4_tokens;
typedef struct {
    bool failed; unsigned trial_n, trial_base, position, mtp_position, context, cap;
    ds4_gpu_tensor *kv[IQ_LAYERS], *trial_kv, *mtp_kv, *trial_mtp, *carry, *trial_hidden, *logits;
    unsigned kv_cap[IQ_LAYERS];
    float *trial_logits; int trial_tokens[IQ_DRAFT_SLOTS + 1];
} ds4_iquest_graph;
typedef struct { ds4_model model; ds4_weights weights; float mtp_margin; } ds4_engine;
typedef struct {
    ds4_engine *engine; ds4_iquest_graph iquest_graph;
    bool checkpoint_valid, mtp_draft_valid, iquest_graph_ready;
    ds4_tokens checkpoint; uint64_t generation; float logits[IQ_VOCAB];
    void *qwen35_graph, *qwen35_ref;
    void (*progress)(void *, const char *, int, int); void *progress_ud;
} ds4_session;
typedef struct { uint64_t spec_hits; } metrics;
static metrics stats;
static ds4_engine engine;
static ds4_session session;
static ds4_gpu_tensor tensors[7];
static float trial_logits[TRIAL * IQ_VOCAB];
static int kept[16];
static fault injected;
static unsigned checks, failures, ensures, reads, forwards, syncs;
static char events[32]; static unsigned event_count;
#define CHECK(name, expression) do { \
    const bool pass = (expression); \
    if (checks++) { printf(","); } \
    printf("{\"name\":\"%s\",\"passed\":%s}", name, pass ? "true" : "false"); \
    failures += !pass; \
} while (0)
static void event(char c) { events[event_count++] = c; events[event_count] = 0; }
static bool ds4_gpu_synchronize(void) { syncs++; event('S'); return injected != SYNC_FAULT; }
static bool ds4_gpu_tensor_copy(ds4_gpu_tensor *d, uint64_t doff, ds4_gpu_tensor *s, uint64_t soff, uint64_t n) {
    if (injected == COPY_FAULT) { return false; }
    if (doff + n > sizeof(d->data) || soff + n > sizeof(s->data)) { abort(); }
    memcpy(d->data + doff, s->data + soff, n); return true;
}
static bool ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *src, uint64_t n) {
    if (injected == WRITE_FAULT) { return false; }
    if (off + n > sizeof(t->data)) { abort(); }
    memcpy(t->data + off, src, n); return true;
}
static bool ds4_gpu_tensor_read(ds4_gpu_tensor *t, uint64_t off, void *dst, uint64_t n) {
    if (off + n > sizeof(t->data)) { abort(); }
    reads++; event('R'); memcpy(dst, t->data + off, n); return true;
}
static void payload_set_err(char *e, size_t n, const char *s) { snprintf(e, n, "%s", s); }
static bool ds4_session_is_iquest(ds4_session *s) { return s != NULL; }
static bool ds4_engine_has_mtp(ds4_engine *e) { return e != NULL; }
static int ds4_session_ensure_graph(ds4_session *s, char *e, size_t n) {
    (void)s; (void)e; (void)n; ensures++; return 0;
}
static void qwen35_session_reset(ds4_session *s) { (void)s; abort(); }
static metrics *ds4_metrics_get(void) { return &stats; }
static void ds4_metric_add(uint64_t *x, uint64_t n) { *x += n; }
static void token_vec_push(ds4_tokens *t, int token) { t->v[t->len++] = token; }
static bool ds4_tokens_starts_with(const ds4_tokens *p, const ds4_tokens *prefix) {
    return p->len >= prefix->len && !memcmp(p->v, prefix->v, prefix->len * sizeof(int));
}
static void ds4_tokens_copy(ds4_tokens *d, const ds4_tokens *s) {
    memcpy(d->v, s->v, s->len * sizeof(int)); d->len = s->len;
}
static bool iquest_forward(ds4_iquest_graph *g, const ds4_model *m, const ds4_weights *w,
    const int *tokens, unsigned n, unsigned pos, iquest_mtp_mode mtp) {
    (void)m; (void)w; (void)tokens; (void)mtp;
    if (g->failed || g->trial_n || g->position != pos || (!pos && g->mtp_position)) { abort(); }
    forwards++; event('F'); g->position = pos + n; g->mtp_position = pos + n - 1;
    const float logits[IQ_VOCAB] = {1, 2, 3, 4};
    memcpy(g->logits->data, logits, sizeof(logits)); return true;
}
RESET
FAIL
READ_LOGITS
INVALIDATE
SYNC
COMMIT
static void counters(void) {
    ensures = reads = forwards = syncs = event_count = 0; events[0] = 0;
}
static void init(void) {
    memset(&session, 0, sizeof(session)); memset(tensors, 0, sizeof(tensors));
    memset(kept, 0, sizeof(kept)); memset(&stats, 0, sizeof(stats));
    session.engine = &engine; session.generation = GENERATION;
    session.checkpoint = (ds4_tokens){.v=kept, .len=BASE, .cap=16};
    session.checkpoint_valid = session.mtp_draft_valid = session.iquest_graph_ready = true;
    ds4_iquest_graph *g = &session.iquest_graph;
    g->cap = 2; g->context = 16; g->position = BASE + TRIAL; g->mtp_position = g->position - 1;
    g->trial_base = BASE; g->trial_n = TRIAL; g->trial_logits = trial_logits;
    g->kv[0] = &tensors[0]; g->kv_cap[0] = 8; g->trial_kv = &tensors[1];
    g->mtp_kv = &tensors[2]; g->trial_mtp = &tensors[3];
    g->carry = &tensors[4]; g->trial_hidden = &tensors[5]; g->logits = &tensors[6];
    injected = NO_FAULT; counters();
}
int main(void) {
    char err[128]; int tokens[] = {1, 2, 3};
    const ds4_tokens prompt = {.v=tokens, .len=3, .cap=3};
    printf("{\"cases\":[");
    for (fault f = COPY_FAULT; f <= SYNC_FAULT; f++) {
        init(); injected = f;
        int rc = ds4_session_iquest_commit(&session, 1, err, sizeof(err));
        CHECK("failed_commit_discards_trial", rc != 0 && session.iquest_graph.failed &&
            !session.checkpoint_valid && session.checkpoint.len == 0 && !session.mtp_draft_valid &&
            session.generation == GENERATION + 1 && session.iquest_graph.trial_n == 0);
        injected = NO_FAULT; counters();
        rc = iquest_session_sync(&session, &prompt, err, sizeof(err));
        CHECK("failed_commit_rebuilds", rc == 0 && !strcmp(events, "SFFR") &&
            session.checkpoint_valid && session.checkpoint.len == prompt.len &&
            !memcmp(session.checkpoint.v, tokens, sizeof(tokens)) &&
            session.generation == GENERATION + 2 && !session.iquest_graph.failed &&
            session.iquest_graph.position == 3 && session.iquest_graph.mtp_position == 2);
    }
    init(); injected = WRITE_FAULT;
    ds4_session_iquest_commit(&session, 1, err, sizeof(err));
    injected = SYNC_FAULT; counters();
    int rc = iquest_session_sync(&session, &prompt, err, sizeof(err));
    CHECK("failed_reset_does_not_forward", rc != 0 && !strcmp(events, "S") &&
        forwards == 0 && reads == 0 && session.iquest_graph.failed &&
        !session.checkpoint_valid && session.checkpoint.len == 0);
    injected = NO_FAULT; counters();
    rc = iquest_session_sync(&session, &prompt, err, sizeof(err));
    CHECK("reset_retry_rebuilds", rc == 0 && !strcmp(events, "SFFR") &&
        session.checkpoint_valid && !session.iquest_graph.failed);
    init(); ds4_session before = session;
    rc = iquest_session_sync(&session, &prompt, err, sizeof(err));
    CHECK("healthy_trial_still_blocks_sync", rc != 0 && event_count == 0 &&
        !memcmp(&before, &session, sizeof(session)));
    int invalid = IQ_VOCAB;
    const ds4_tokens bad = {.v=&invalid, .len=1, .cap=1};
    counters(); rc = iquest_session_sync(&session, &bad, err, sizeof(err));
    CHECK("invalid_input_preserves_trial", rc != 0 && ensures == 0 && event_count == 0 &&
        !memcmp(&before, &session, sizeof(session)));
    rc = ds4_session_iquest_commit(&session, 0, err, sizeof(err));
    CHECK("invalid_commit_preserves_trial", rc != 0 && event_count == 0 &&
        !memcmp(&before, &session, sizeof(session)));
    printf("],\"checks\":%u,\"failed\":%u}\n", checks, failures);
    return failures ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = (ROOT / "ds4_iquest_session.inc").read_text()
    native = (ROOT / "ds4.c").read_text()
    start = native.index("void ds4_session_invalidate(")
    end = native.index("    } else if (ds4_session_is_naive(s)", start)
    bodies = {
        "RESET": function(source, "iquest_reset", "static bool "),
        "FAIL": function(source, "iquest_session_fail", "static int "),
        "READ_LOGITS": function(source, "iquest_read_logits", "static bool "),
        "INVALIDATE": native[start:end] + "    }\n#endif\n}\n",
        "SYNC": function(source, "iquest_session_sync", "static int "),
        "COMMIT": function(source, "ds4_session_iquest_commit", "int "),
    }
    code = C
    for key, body in bodies.items():
        code = code.replace(key + "\n", body + "\n")
    with tempfile.TemporaryDirectory(prefix="iquest-recovery-") as tmp:
        tmp = Path(tmp)
        path, binary = tmp / "test.c", tmp / "test"
        path.write_text(code)
        subprocess.run([
            "cc", "-std=c11", "-O0", "-Wall", "-Wextra", str(path), "-lm", "-o", str(binary),
        ], check=True)
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    report = json.loads(run.stdout)
    report.update({
        "scope": "Actual failure/reset/readback/sync/commit bodies and native IQuest invalidation branch; tiny CPU tensor stubs, no GPU/model math.",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "native_sha256": hashlib.sha256(native.encode()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "functions_sha256": {key: hashlib.sha256(body.encode()).hexdigest() for key, body in bodies.items()},
    })
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
