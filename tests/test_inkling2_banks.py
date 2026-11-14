#!/usr/bin/env python3
"""Compile Inkling-2's bank checkpoint rules (inkling2_ckpt_find/inherit/drop,
inkling2_ring_spans) and its snapshot sizes (inkling2_payload_body,
inkling2_draft_kept) over stub runtimes.

A fork below a bank's frontier restores the deepest checkpoint at or below
the shared prefix that the source bank holds; a fork shares the checkpoints
below its cut and a rewind in place drops those above. A snapshot keeps the
drafter's rows the next trial can read: at or above lo, within the ring less
one trial block."""
import subprocess, tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
ds4 = (repo / 'ds4.c').read_text()
bind = (repo / 'ds4_inkling2_bind.inc').read_text()
batch = (repo / 'ds4_inkling2_batch.inc').read_text()
payload = (repo / 'ds4_inkling2_payload.inc').read_text()


def cut(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a)]


code = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "ds4_inkling2.h"
enum { INKLING2_LAYERS = 48, DS4_MULTISEQ_MAX_SEQ = 128, DS4_N_VOCAB = 201024, DS4_SESSION_PAYLOAD_U32_FIELDS = 13 };
@ROWS@
@FULL@
@DRAFT@
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
typedef struct ds4_inkling2_graph ds4_inkling2_graph;
typedef struct { uint32_t ring[DS4_INK2_DRAFT_LAYERS], lo[DS4_INK2_DRAFT_LAYERS]; } inkling2_draft_rt;
static bool inkling2_draft_local(uint32_t l) { return l != INKLING2_DRAFT_GLOBAL_LAYER; }
@PARTIAL@
@RUNTIME@
@SPANS@
@CKPT@
@SIZES@
@KEPT@
static int check(const char *name, uint64_t got, uint64_t want) {
    printf("%s: %llu, want %llu%s\n", name, (unsigned long long)got, (unsigned long long)want,
           got == want ? "" : "  FAIL");
    return got == want;
}

static ds4_inkling2_batch_runtime rt;

static void slot(uint32_t i, uint32_t pos, uint32_t bank, uint8_t logits) {
    rt.checkpoint[i] = (ds4_partial_checkpoint){0};
    rt.checkpoint[i].pos = pos;
    rt.checkpoint[i].logits_valid = logits;
    partial_checkpoint_set_ref(&rt.checkpoint[i], bank);
}

int main(void) {
    int ok = 1;

    /* Bank 0 holds checkpoints at 4,096 and 8,192 (prompt end, logits). */
    slot(0, 4096, 0, 0);
    slot(1, 8192, 0, 1);
    ok &= check("find_below_frontier", (uint64_t)inkling2_ckpt_find(&rt, 0, 6144, 9000), 0);
    ok &= check("find_deepest", (uint64_t)inkling2_ckpt_find(&rt, 0, 8192, 9000), 1);
    ok &= check("find_other_bank", (uint64_t)(inkling2_ckpt_find(&rt, 1, 8192, 9000) < 0), 1);
    ok &= check("find_none_below", (uint64_t)(inkling2_ckpt_find(&rt, 0, 4095, 9000) < 0), 1);
    /* A request ending at a checkpoint needs its logits. */
    ok &= check("request_end_needs_logits", (uint64_t)(inkling2_ckpt_find(&rt, 0, 4096, 4096) < 0), 1);
    ok &= check("request_end_with_logits", (uint64_t)inkling2_ckpt_find(&rt, 0, 8192, 8192), 1);

    /* A fork at exactly a checkpoint's position shares it. */
    inkling2_ckpt_inherit(&rt, 0, 2, 4096);
    ok &= check("fork_shares_at_cut", (uint64_t)inkling2_ckpt_find(&rt, 2, 4096, 9000), 0);
    inkling2_ckpt_drop(&rt, 2);

    /* A fork to bank 1 at 6,144 shares only the checkpoint below its cut. */
    inkling2_ckpt_inherit(&rt, 0, 1, 6144);
    ok &= check("fork_shares_below", (uint64_t)inkling2_ckpt_find(&rt, 1, 8192, 9000), 0);
    ok &= check("source_keeps_both", (uint64_t)inkling2_ckpt_find(&rt, 0, 8192, 9000), 1);
    /* Rewinding bank 0 in place to 6,144 drops the one above. */
    inkling2_ckpt_inherit(&rt, 0, 0, 6144);
    ok &= check("rewind_drops_above", (uint64_t)inkling2_ckpt_find(&rt, 0, 8192, 9000), 0);
    ok &= check("slot_freed", rt.checkpoint[1].pos, 0);
    inkling2_ckpt_drop(&rt, 0);
    ok &= check("drop_keeps_other_bank", (uint64_t)inkling2_ckpt_find(&rt, 1, 8192, 9000), 0);
    inkling2_ckpt_drop(&rt, 1);
    ok &= check("last_ref_frees", rt.checkpoint[0].pos, 0);

    /* Ring spans: the 1,023 slots ending at (pos - 1) % cap. */
    const uint32_t caps[] = {512 + DS4_INK2_WINDOW - 1u, 2048 + DS4_INK2_WINDOW - 1u};
    const uint32_t ats[] = {1, 1023, 1024, 3071, 3072, 5000, 1048575};
    for (int c = 0; c < 2; c++) {
        for (int a = 0; a < (int)(sizeof ats / sizeof ats[0]); a++) {
            inkling2_span s[2];
            const uint32_t n = inkling2_ring_spans(caps[c], ats[a], s);
            const uint64_t first = (ats[a] % caps[c] + caps[c] - (DS4_INK2_WINDOW - 1u)) % caps[c];
            ok &= check("span_rows", s[0].rows + (n > 1 ? s[1].rows : 0), DS4_INK2_WINDOW - 1u);
            ok &= check("span_first", s[0].slot, first);
            if (n > 1) { ok &= check("span_wraps", s[0].slot + s[0].rows, caps[c]); }
        }
    }

    /* Draft rows a snapshot keeps: lo bounds them, the ring less a block caps them. */
    inkling2_draft_rt d = {{0}, {0}};
    for (uint32_t l = 0; l < DS4_INK2_DRAFT_LAYERS; l++) {
        d.ring[l] = inkling2_draft_local(l) ? INKLING2_DRAFT_LOCAL_RING : INKLING2_DRAFT_GLOBAL + 9u;
    }
    ok &= check("kept_short", inkling2_draft_kept(&d, 0, 500), 500);
    ok &= check("kept_local_cap", inkling2_draft_kept(&d, 0, 100000), INKLING2_DRAFT_LOCAL_RING - 8u);
    ok &= check("kept_global_cap", inkling2_draft_kept(&d, INKLING2_DRAFT_GLOBAL_LAYER, 100000), INKLING2_DRAFT_GLOBAL + 1u);
    d.lo[0] = 99700;
    ok &= check("kept_from_lo", inkling2_draft_kept(&d, 0, 100000), 300);
    d.lo[0] = 100000;
    ok &= check("kept_none_at_lo", inkling2_draft_kept(&d, 0, 100000), 0);

    /* Payload body: tokens, logits, 8 full layers' blocks, 40 rings' window, 48 histories. */
    const uint32_t none[2] = {0, 0}, rows[2] = {1032, 4000};
    const uint64_t n = 5000, blocks = (n + 127) / 128;
    const uint64_t base = n * 4 + (uint64_t)DS4_N_VOCAB * 4 + 8 * blocks * 147456 + 40 * 1023 * 4096 +
        48 * 3 * (2 * 1024 + 2 * 4608) * 4;
    ok &= check("body_ik21", inkling2_payload_body(5000, INKLING2_PAYLOAD_MAGIC, none), base);
    ok &= check("body_ik22", inkling2_payload_body(5000, INKLING2_PAYLOAD_DRAFT, rows),
        base + (4 * 1032 + 4000) * 2 * 1024 * 2 + 4608 * 4);
    ok &= check("body_short_window", inkling2_payload_body(300, INKLING2_PAYLOAD_MAGIC, none),
        300 * 4 + (uint64_t)DS4_N_VOCAB * 4 + 8 * 3 * 147456 + 40 * 300 * 4096 + 48 * 3 * (2 * 1024 + 2 * 4608) * 4);
    return ok ? 0 : 1;
}
'''
subs = {
    '@ROWS@': cut(bind, 'enum {\n    INKLING2_CONTEXT', 'static uint32_t inkling2_prefill_cap('),
    '@FULL@': 'enum { INKLING2_FULL_LAYERS = INKLING2_LAYERS / DS4_INK2_BLOCK };\n',
    '@DRAFT@': cut(bind, 'enum {\n    INKLING2_DRAFT_LOCAL_RING', '\n};\n') + '\n};\n',
    '@PARTIAL@': cut(ds4, 'enum {\n    DS4_PARTIAL_CHECKPOINT_SLOTS', 'static uint64_t ds4_mem_usable_beyond('),
    '@RUNTIME@': cut(batch, 'enum { INKLING2_BANK_MAX', '// What one more bank adds'),
    '@SPANS@': cut(batch, 'typedef struct { uint64_t slot, rows; } inkling2_span;', 'static bool inkling2_ring_copy('),
    '@CKPT@': cut(batch, 'static void inkling2_ckpt_drop(', 'static bool inkling2_ckpt_capture('),
    '@SIZES@': cut(payload, 'enum { INKLING2_PAYLOAD_MAGIC', '#ifndef __APPLE__') +
               cut(payload, 'static uint32_t inkling2_payload_window(', '/* A draft layer'),
    '@KEPT@': cut(payload, '/* A draft layer', '// Words 8 and 9'),
}
for key, value in subs.items():
    assert code.count(key) == 1, key
    code = code.replace(key, value)
with tempfile.TemporaryDirectory() as t:
    p = Path(t)
    (p / 'banks.c').write_text(code)
    subprocess.run(['cc', '-std=gnu11', '-O2', '-Wall', '-Werror', '-Wno-unused-function', f'-I{repo}',
                    str(p / 'banks.c'), '-o', str(p / 'banks')], check=True)
    raise SystemExit(subprocess.run([str(p / 'banks')]).returncode)
