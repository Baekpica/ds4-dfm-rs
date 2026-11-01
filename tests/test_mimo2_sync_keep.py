#!/usr/bin/env python3
"""Compile MiMo's serial rewind rule (mimo2_sync_keep) over stub sessions.

A rewind needs the SWA window before the row it resumes at. The ring keeps
it only while counted from the newest row written, and a restored snapshot
holds only the window before its frontier, however large the ring."""
import subprocess, tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
session = (repo / 'ds4_mimo2_session.inc').read_text()
payload = (repo / 'ds4_mimo2_payload.inc').read_text()


def cut(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a)]


code = r'''
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
enum { M2_WINDOW = 128 };
typedef struct { int *v; int len, cap; } ds4_tokens;
typedef struct { unsigned context, cap, position, high, restored; bool failed; } ds4_mimo2_graph;
typedef struct { ds4_tokens checkpoint; bool checkpoint_valid; } ds4_session;
static bool ds4_gpu_synchronize(void) { return true; }
RESET
KEEP
enum { N = 20000, OTHER = 1 << 30 };
static int toks[N + 8], prompt_v[N + 8];
static ds4_session s;
static ds4_mimo2_graph graph;

static void fresh(unsigned context, unsigned cap) {
    memset(&graph, 0, sizeof graph);
    graph.context = context;
    graph.cap = cap;
    mimo2_reset(&graph);
}

/* Rows 0..end-1 committed, as mimo2_forward leaves them. */
static void write_to(unsigned end) {
    graph.position = end;
    if (graph.high < end) { graph.high = end; }
    s.checkpoint = (ds4_tokens){toks, (int)end, (int)end};
    s.checkpoint_valid = true;
}

static void restore(unsigned n) {
    ds4_mimo2_graph *g = &graph;
RESTORED
    s.checkpoint = (ds4_tokens){toks, (int)n, (int)n};
    s.checkpoint_valid = true;
}

/* Keep for a prompt sharing `common` live tokens, `len` tokens long. */
static unsigned keep_for(unsigned common, unsigned len) {
    for (unsigned i = 0; i < len; i++) { prompt_v[i] = i < common ? toks[i] : OTHER; }
    const ds4_tokens prompt = {prompt_v, (int)len, (int)len};
    return mimo2_sync_keep(&s, &graph, &prompt);
}

static int check(const char *name, unsigned got, unsigned want) {
    printf("%s: keep %u, want %u%s\n", name, got, want, got == want ? "" : "  FAIL");
    return got == want;
}

int main(void) {
    for (int i = 0; i < N + 8; i++) { toks[i] = i; }
    int ok = 1;

    /* The ring covers a 4,096-row context, but the snapshot holds only the
     * window before 3,500. */
    fresh(4096, 4096);
    restore(3500);
    ok &= check("restored_resend", keep_for(3000, 3000), 0);
    ok &= check("restored_extends", keep_for(3500, 3600), 3500);

    /* Rows decoded after the restore are whole again. */
    fresh(1u << 20, 4096);
    restore(3500);
    write_to(3700);
    ok &= check("restored_rewind_above", keep_for(3600, 3610), 3600);

    /* The ring is counted from the newest row written: a second rewind
     * cannot rely on the first one's shorter state. */
    fresh(1u << 20, 4096);
    write_to(20000);
    ok &= check("first_rewind", keep_for(16500, 16500), 16499);
    graph.position = 16499;
    write_to(16500);
    ok &= check("second_rewind", keep_for(13000, 13000), 0);

    /* A re-rendered answer: the drafter windows re-run below the prefix. */
    fresh(1u << 20, 4096);
    write_to(20000);
    ok &= check("redraft", keep_for(19500, 19503), 19500 - 1024);
    return ok ? 0 : 1;
}
'''.replace('RESET', cut(session, 'static bool mimo2_reset(', 'static int mimo2_session_fail(')) \
   .replace('KEEP', cut(session, 'enum { M2_REDRAFT', 'static int mimo2_session_sync(')) \
   .replace('RESTORED', cut(payload, '    g->position = n;\n', '    mimo2_mtp_rebase(g, n);'))
with tempfile.TemporaryDirectory() as t:
    p = Path(t)
    (p / 'keep.c').write_text(code)
    subprocess.run(['cc', '-std=gnu11', '-O2', '-Wall', '-Werror', str(p / 'keep.c'), '-o', str(p / 'keep')], check=True)
    raise SystemExit(subprocess.run([str(p / 'keep')]).returncode)
