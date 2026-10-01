#!/usr/bin/env python3
"""CPU regression for the public sync admission that guards IQuest dispatch.

Compile the actual public admission and bridge functions with an instrumented
IQuest handler. Removing the length guard must make the same tests fail.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name, declaration):
    start = source.index(declaration + name + "(")
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


C = r'''
#include "ds4.h"
#include "native/bridge/ds4_bridge.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
enum { TEST_CONTEXT = 8 };
typedef enum { DIRECT, BRIDGE, CALLBACK } entry;
typedef enum { FRESH, POPULATED } timeline;
struct ds4_session {
    int ctx_size;
    ds4_tokens checkpoint;
    bool checkpoint_valid, graph_ready;
    uint64_t generation;
    unsigned position, mtp_position;
    float logits[4];
    ds4_session_progress_fn progress;
    void *progress_ud;
};
struct ds4_bridge_session { ds4_session *session; };
typedef struct { ds4_bridge_prefill_fn progress; void *ud; } sync_tramp;
static unsigned dispatches, checks, failed;
static bool ds4_session_is_iquest(ds4_session *s) { return s != NULL; }
static int iquest_session_sync(ds4_session *s, const ds4_tokens *p, char *err, size_t n) {
    (void)s; (void)p; (void)err; (void)n;
    dispatches++;
    return 0;
}
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    s->progress = fn; s->progress_ud = ud;
}
static void progress(void *ud, int32_t current, int32_t total) {
    (void)ud; (void)current; (void)total; abort();
}
SET_ERR
SYNC_TRAMP
PUBLIC_SYNC
BRIDGE_SYNC
BRIDGE_ENTRY
BRIDGE_CALLBACK
static void run_case(entry mode, timeline state, const int *tokens, int count) {
    int kept[] = {1, 2, 3};
    ds4_session s = {.ctx_size=TEST_CONTEXT, .generation=42};
    if (state == POPULATED) {
        s.checkpoint = (ds4_tokens){.v=kept, .len=3, .cap=3};
        s.checkpoint_valid = s.graph_ready = true;
        s.position = 3; s.mtp_position = 2; s.logits[0] = 7;
    }
    unsigned char before[sizeof(s)];
    memcpy(before, &s, sizeof(s));
    ds4_tokens prompt = {.v=(int *)tokens, .len=count, .cap=count};
    ds4_bridge_session bridge = {.session=&s};
    char err[128] = {0};
    dispatches = 0;
    int rc;
    if (mode == DIRECT) {
        rc = ds4_session_sync(&s, &prompt, err, sizeof(err));
    } else if (mode == BRIDGE) {
        rc = ds4_bridge_session_sync(&bridge, tokens, count, err, sizeof(err));
    } else {
        rc = ds4_bridge_session_sync_cb(&bridge, tokens, count, progress, NULL, err, sizeof(err));
    }
    const bool reject = count <= 0 || count > TEST_CONTEXT;
    const bool passed = reject
        ? rc != 0 && dispatches == 0 && err[0] && !memcmp(before, &s, sizeof(s))
        : rc == 0 && dispatches == 1;
    checks++; failed += !passed;
}
int main(void) {
    int tokens[TEST_CONTEXT + 1] = {1};
    const int lengths[] = {INT_MIN, -1, 0, 1, TEST_CONTEXT, TEST_CONTEXT + 1};
    for (entry mode = DIRECT; mode <= CALLBACK; mode++) {
        for (timeline state = FRESH; state <= POPULATED; state++) {
            for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) {
                run_case(mode, state, tokens, lengths[i]);
            }
            run_case(mode, state, NULL, 0);
        }
    }
    ds4_session s = {.ctx_size=TEST_CONTEXT};
    ds4_tokens prompt = {.v=tokens, .len=1, .cap=1};
    char err[128];
    dispatches = 0;
    checks++;
    failed += ds4_session_sync(NULL, &prompt, err, sizeof(err)) == 0 || dispatches != 0;
    checks++;
    failed += ds4_session_sync(&s, NULL, err, sizeof(err)) == 0 || dispatches != 0;
    printf("{\"checks\":%u,\"failed\":%u}\n", checks, failed);
    return failed ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    native = (ROOT / "ds4.c").read_text()
    bridge = (ROOT / "native/bridge/ds4_bridge.c").read_text()
    start = native.index("int ds4_session_sync(")
    end = native.index("    if (ds4_session_is_naive(s))", start)
    public = native[start:end] + "#endif\n    abort();\n}\n"
    replacements = {
        "SET_ERR": function(bridge, "set_err", "static void "),
        "SYNC_TRAMP": function(bridge, "sync_tramp_progress", "static void "),
        "PUBLIC_SYNC": public,
        "BRIDGE_SYNC": function(bridge, "session_sync", "static int "),
        "BRIDGE_ENTRY": function(bridge, "ds4_bridge_session_sync", "int "),
        "BRIDGE_CALLBACK": function(bridge, "ds4_bridge_session_sync_cb", "int "),
    }
    code = C
    for key, value in replacements.items():
        code = code.replace(key + "\n", value + "\n")
    guard = "prompt->len <= 0 || "
    assert code.count(guard) == 1, "public sync guard changed"
    reports = {}
    with tempfile.TemporaryDirectory(prefix="iquest-sync-") as tmp:
        tmp = Path(tmp)
        for name, text in [("production", code), ("removed_guard", code.replace(guard, ""))]:
            path, binary = tmp / f"{name}.c", tmp / name
            path.write_text(text)
            subprocess.run([
                "cc", "-std=c11", "-O0", "-Wall", "-Wextra", "-I", str(ROOT),
                str(path), "-o", str(binary),
            ], check=True)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            reports[name] = json.loads(run.stdout)
            reports[name]["exit_code"] = run.returncode
    passed = reports["production"]["exit_code"] == 0 and reports["removed_guard"]["exit_code"] == 1
    report = {
        "passed": passed,
        "scope": "Actual public sync admission and bridge; instrumented IQuest dispatch, no graph/model/GPU. Rejected input preserves fresh and populated native state.",
        "results": reports,
        "native_sha256": hashlib.sha256(native.encode()).hexdigest(),
        "bridge_sha256": hashlib.sha256(bridge.encode()).hexdigest(),
        "public_prefix_sha256": hashlib.sha256(public.encode()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
