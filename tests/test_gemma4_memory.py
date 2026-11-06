#!/usr/bin/env python3
"""Compare the Gemma activation and MTP quotes with allocations without a GPU.

Compile the native allocation blocks over a counting allocator, so adding or
resizing a tensor must also update the admission estimate.
"""
import re
import subprocess
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
source = (repo / "ds4_gemma4_graph.inc").read_text()
start = source.index("    const uint64_t act =", source.index("gemma4_graph_create("))
end = source.index("    float inv[", start)
allocations = source[start:end]
fields = re.findall(r"^    G4_ALLOC\((\w+),", allocations, re.MULTILINE)
assert fields, "graph allocation block missing"
mtp = (repo / "ds4_gemma4_mtp.inc").read_text()
start = mtp.index("    const uint32_t rows = drafts + 1u;")
mtp_allocations = mtp[start:mtp.index("    r->host =", start)]
mtp_fields = re.findall(r"r->(\w+) = ds4_gpu_tensor_alloc", mtp_allocations)
engine = (repo / "ds4.c").read_text()
start = engine.index("static uint64_t gemma4_session_bytes(const ds4_engine *e, uint32_t ctx, uint32_t cap) {")
session_bytes = engine[start:engine.index("\n}\n", start) + 3]
code = r'''
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include "ds4_gemma4_plan.h"

typedef struct { FIELDS } graph;
static uint64_t allocated;
static void *ds4_gpu_tensor_alloc(uint64_t bytes) {
    allocated += bytes;
    return &allocated;
}
static void gemma4_graph_release(graph *g) { (void)g; }
static graph *allocate(unsigned cap) {
    static graph storage;
    graph *g = &storage;
    const uint32_t rows = gemma4_token_rows(cap);
ALLOCATIONS
    return g;
}
typedef struct { MTP_FIELDS } mtp_runtime;
static void allocate_mtp(unsigned drafts) {
    mtp_runtime storage;
    mtp_runtime *r = &storage;
MTP_ALLOCATIONS
}
typedef struct { struct { void *gemma4_mtp; } mtp_weights; } ds4_engine;
typedef struct { uint64_t total_bytes; } ds4_context_memory;
static ds4_context_memory gemma4_session_memory(uint32_t ctx, uint32_t cap) {
    (void)ctx; (void)cap;
    return (ds4_context_memory){0}; /* Isolate the extra MTP allocation. */
}
static uint64_t ds4_gpu_gvq_workspace_linear(uint32_t rows) {
    return 32ull * 1024 * 1024 + (uint64_t)rows * 8u * sizeof(uint32_t);
}
SESSION_BYTES
int main(void) {
    const unsigned caps[] = {1u, 7u, 512u, 4096u, G4_PREFILL_MAX};
    int failed = 0;
    for (size_t i = 0; i < sizeof(caps) / sizeof(*caps); i++) {
        allocated = 0;
        if (!allocate(caps[i])) { return 1; }
        const uint64_t quoted = gemma4_activation_bytes(caps[i]);
        if (quoted != allocated) {
            fprintf(stderr, "chunk %u: quoted %" PRIu64 ", allocated %" PRIu64 "\n",
                    caps[i], quoted, allocated);
            failed = 1;
        }
    }
    allocated = 0;
    allocate_mtp(G4_MTP_DRAFTS);
    if (gemma4_mtp_scratch_bytes() != allocated) {
        fprintf(stderr, "MTP tensors: quoted %" PRIu64 ", allocated %" PRIu64 "\n",
                gemma4_mtp_scratch_bytes(), allocated);
        failed = 1;
    }
    allocated += ds4_gpu_gvq_workspace_linear(1u);
    const ds4_engine e = {.mtp_weights.gemma4_mtp = &allocated};
    if (gemma4_session_bytes(&e, 1024u, 512u) != allocated) {
        fprintf(stderr, "MTP session: quoted %" PRIu64 ", allocated %" PRIu64 "\n",
                gemma4_session_bytes(&e, 1024u, 512u), allocated);
        failed = 1;
    }
    if (!failed) { puts("gemma4 memory: ok"); }
    return failed;
}
'''.replace("MTP_FIELDS", " ".join(f"void *{field};" for field in mtp_fields))
code = code.replace("FIELDS", " ".join(f"void *{field};" for field in fields))
code = code.replace("MTP_ALLOCATIONS", mtp_allocations).replace("ALLOCATIONS", allocations)
code = code.replace("SESSION_BYTES", session_bytes)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)
    (path / "memory.c").write_text(code)
    subprocess.run([
        "cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-I", str(repo),
        str(path / "memory.c"), "-o", str(path / "memory"),
    ], check=True)
    raise SystemExit(subprocess.run([str(path / "memory")]).returncode)
