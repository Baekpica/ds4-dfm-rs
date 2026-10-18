#!/usr/bin/env python3
"""Compile family_cont_spec's accepted-prefix loop and commit over stub sessions.

A request that finishes inside the accepted prefix must leave its last token
pending, as ordinary decode does: the server keys the bank by every generated
token but the last. Every accepted draft still counts as a hit."""
import subprocess, tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
s = (repo / 'ds4.c').read_text()
fn = s.index('static int family_cont_spec(')
a = s.index('    int emitted = 1;', fn)
b = s.index('    if (gr) {\n        gr->graph[bank] = s.glm53_graph;', a)
code = r'''
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#define DS4_SAMPLE_OVERRIDE_NONE (-1)
#define DS4_SAMPLE_OVERRIDE_IS_TOKEN(v) ((v) >= 0)
typedef struct { uint64_t tokens_decoded, spec_hits; } metrics_t;
static metrics_t metrics;
static metrics_t *ds4_metrics_get(void) { return &metrics; }
static void ds4_metric_add(uint64_t *v, uint64_t n) { *v += n; }
static void ds4_metrics_window_add(int a, int b, int c) { (void)a; (void)b; (void)c; }
typedef struct { int unused; } ds4_session;
static int landed;
/* Every family commit lands `keep` trial rows and counts keep - 1 hits. */
static int land(ds4_session *s, int keep, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    landed = keep;
    if (keep > 1) { ds4_metric_add(&metrics.spec_hits, (uint64_t)keep - 1u); }
    return 0;
}
#define ds4_session_glm53_commit land
#define ds4_session_iquest_commit land
#define ds4_session_ling31_commit land
#define ds4_session_naive_commit land
#define ds4_session_step37_commit land
typedef struct {
    int (*sample_exclude)(void *, void *);
    int (*sample_override)(void *, void *);
    void *user;
    int *generated;
    uint32_t generated_len, max_new;
    int current, eos;
} bank_t;
static int abort_at;
static int deliver(void *ud, void *user, int token) { (void)ud; (void)user; return token != abort_at; }

/* Trial [10, 11, 12, 13] after 10 was emitted; returns the rows landed. */
static int step(int keep, uint32_t max_new, int eos, int abort_token, uint64_t *hits) {
    int generated[8] = {10};
    bank_t bank = {.generated = generated, .generated_len = 1, .max_new = max_new, .current = 10, .eos = eos};
    bank_t *cb = &bank;
    const int tokens[4] = {10, 11, 12, 13};
    int graph = 0;
    void *gr = NULL, *ir = NULL, *lr = &graph, *nr = NULL, *ud = NULL;
    int (*on_token)(void *, void *, int) = deliver;
    const int first_override = DS4_SAMPLE_OVERRIDE_NONE;
    ds4_session s = {0};
    char err[16];
    const size_t errlen = sizeof(err);
    abort_at = abort_token;
    metrics.spec_hits = 0;
    landed = 0;
BODY
    (void)finished;
    *hits = metrics.spec_hits;
    return landed;
fail:
    return -1;
}

int main(void) {
    static const struct { const char *name; int keep; uint32_t max_new; int eos, abort_token, rows; uint64_t hits; } cases[] = {
        {"length_inside", 3, 3, -1, -1, 2, 2},
        {"eos_inside", 3, 8, 12, -1, 2, 2},
        {"abort_inside", 3, 8, -1, 11, 1, 1},
        {"running", 3, 8, -1, -1, 3, 2},
    };
    int ok = 1;
    printf("{");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint64_t hits = 0;
        const int rows = step(cases[i].keep, cases[i].max_new, cases[i].eos, cases[i].abort_token, &hits);
        const int pass = rows == cases[i].rows && hits == cases[i].hits;
        ok &= pass;
        printf("%s\"%s\":{\"rows\":%d,\"hits\":%llu,\"pass\":%s}", i ? "," : "", cases[i].name, rows,
               (unsigned long long)hits, pass ? "true" : "false");
    }
    printf("}\n");
    return ok ? 0 : 1;
}
'''.replace('BODY', s[a:b])
with tempfile.TemporaryDirectory() as t:
    p = Path(t)
    (p / 'finish.c').write_text(code)
    subprocess.run(['cc', '-std=gnu11', '-O2', '-Wall', '-Werror', str(p / 'finish.c'), '-o', str(p / 'finish')], check=True)
    raise SystemExit(subprocess.run([str(p / 'finish')]).returncode)
