#!/usr/bin/env python3
"""Compile Inkling-2's serial resume rule (inkling2_sync_keep with its image
cut) and the ring spans a resume copies (inkling2_resume_parts) over stub
sessions.

The convolution histories have no positional index, so a prompt resumes
where it extends the live state or at a copy at or below its common prefix;
a copy holds the histories and each sliding ring's 1,023 rows below it.
Placeholder tokens do not name pixels: the prefix also ends at the first
image the live state and the prompt do not share (run and pixel tag)."""
import subprocess, tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
ds4 = (repo / 'ds4.c').read_text()
bind = (repo / 'ds4_inkling2_bind.inc').read_text()
session = (repo / 'ds4_inkling2_session.inc').read_text()


def cut(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a)]


code = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "ds4_inkling2.h"
enum { INKLING2_LAYERS = 48, SERIAL_CKPTS = 2 };
@ROWS@
@FULL@
@GLOBAL@
typedef struct { int *v; int len, cap; } ds4_tokens;
typedef struct ds4_gpu_tensor { uint64_t off, bytes; } ds4_gpu_tensor;
typedef struct {
    uint32_t pos[SERIAL_CKPTS];
    uint64_t gen[SERIAL_CKPTS];
} ds4_serial_ckpt;
typedef struct { uint32_t start, count; uint64_t tag; } inkling2_image;
typedef struct {
    uint32_t ring_cap, position;
    bool failed;
    ds4_gpu_tensor *kv[INKLING2_LAYERS], *hist[INKLING2_LAYERS];
    const unsigned *span_start, *span_count;
    const uint64_t *span_tag;
    uint32_t span_n;
    inkling2_image *live;
    uint32_t live_n;
} ds4_inkling2_graph;
typedef struct {
    ds4_tokens checkpoint;
    bool checkpoint_valid, mtp_draft_valid;
    uint64_t generation;
    ds4_serial_ckpt rewind;
    ds4_inkling2_graph inkling2_graph;
} ds4_session;
typedef struct { ds4_gpu_tensor *tensor; uint64_t bytes; } serial_part;

/* Views record their byte range; an empty one fails, as a device view does. */
static ds4_gpu_tensor views[2 * INKLING2_LAYERS];
static int nviews;
static ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t off, uint64_t bytes) {
    (void)base;
    if (!bytes) { return NULL; }
    views[nviews] = (ds4_gpu_tensor){off, bytes};
    return &views[nviews++];
}
@COMMON@
@PREFIX@
@CKPT@
@REWOUND@
@KEEP@
enum { N = 8192, OTHER = 1 << 30 };
static int toks[N], prompt_v[N];
static ds4_session s;

/* Live state of `len` rows with copies at `a` and `b` (0: none). */
static void live(uint32_t len, uint32_t a, uint32_t b) {
    s = (ds4_session){0};
    s.checkpoint = (ds4_tokens){toks, (int)len, N};
    s.checkpoint_valid = true;
    s.inkling2_graph.position = len;
    s.rewind.pos[0] = a;
    s.rewind.pos[1] = b;
}

/* The live state's images and the prompt's (runs and tags). */
static inkling2_image held[4];
static unsigned asked_start[4], asked_count[4];
static uint64_t asked_tag[4];
static void images(const inkling2_image *live_images, uint32_t nlive, const inkling2_image *asked, uint32_t nasked) {
    ds4_inkling2_graph *g = &s.inkling2_graph;
    for (uint32_t i = 0; i < nlive; i++) { held[i] = live_images[i]; }
    for (uint32_t i = 0; i < nasked; i++) {
        asked_start[i] = asked[i].start;
        asked_count[i] = asked[i].count;
        asked_tag[i] = asked[i].tag;
    }
    g->live = held;
    g->live_n = nlive;
    g->span_start = asked_start;
    g->span_count = asked_count;
    g->span_tag = asked_tag;
    g->span_n = nasked;
}

/* Keep for a prompt sharing `common` live tokens, `len` tokens long. */
static uint32_t keep_for(uint32_t common, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) { prompt_v[i] = i < common ? toks[i] : OTHER; }
    const ds4_tokens prompt = {prompt_v, (int)len, N};
    return inkling2_sync_keep(&s, &prompt);
}

static int check(const char *name, uint32_t got, uint32_t want) {
    printf("%s: %u, want %u%s\n", name, got, want, got == want ? "" : "  FAIL");
    return got == want;
}

/* The ring spans a copy at `at` takes: 1,023 rows ending at slot (at - 1) % ring. */
static int spans(uint32_t ring, uint32_t at) {
    static ds4_gpu_tensor kv, hist;
    ds4_inkling2_graph g = {.ring_cap = ring};
    for (int il = 0; il < INKLING2_LAYERS; il++) { g.kv[il] = &kv; g.hist[il] = &hist; }
    serial_part parts[INKLING2_RESUME_PARTS] = {{0}};
    nviews = 0;
    if (!inkling2_resume_parts(&g, at, parts)) {
        printf("spans ring %u at %u: a view failed  FAIL\n", ring, at);
        return 0;
    }
    int ok = 1;
    for (int il = 0; il < INKLING2_LAYERS; il++) {
        ok &= parts[il].tensor == &hist && parts[il].bytes == (uint64_t)INKLING2_HIST * sizeof(float);
    }
    for (int i = INKLING2_LAYERS; i < INKLING2_RESUME_PARTS; i += 2) {
        const ds4_gpu_tensor *a = parts[i].tensor, *b = parts[i + 1].tensor;
        const uint64_t row = INKLING2_KV_ROW, first = (at % ring + ring - (DS4_INK2_WINDOW - 1u)) % ring;
        ok &= a->off == first * row && a->bytes + b->bytes == (uint64_t)(DS4_INK2_WINDOW - 1u) * row;
        ok &= b->off == (a->off + a->bytes) % ((uint64_t)ring * row);
        ok &= parts[i].bytes == a->bytes && parts[i + 1].bytes == b->bytes;
    }
    printf("spans ring %u at %u%s\n", ring, at, ok ? "" : "  FAIL");
    return ok;
}

int main(void) {
    for (int i = 0; i < N; i++) { toks[i] = i; }
    int ok = 1;

    /* A sync copies where its final chunk starts and at its end. */
    live(5000, 4096, 5000);
    ok &= check("extension", keep_for(5000, 5100), 5000);
    ok &= check("same_prompt", keep_for(5000, 5000), 5000);
    ok &= check("below_every_copy", keep_for(3000, 3100), 0);

    /* An answer decoded past the prompt: the turn is resent (its last
     * token re-runs for the logits), or the client re-renders the answer. */
    live(5200, 4096, 5000);
    ok &= check("resent_turn", keep_for(5000, 5000), 4096);
    ok &= check("rerendered_continuation", keep_for(5100, 5300), 5000);

    /* A rewind to 4,500 keeps the copy below it; the one above described
     * the abandoned branch, also once the new branch passes it. */
    live(6000, 4000, 5000);
    session_rewound(&s, 4500);
    s.checkpoint.len = 6000;
    s.inkling2_graph.position = 6000;
    ok &= check("stale_copy", keep_for(5500, 5600), 4000);

    /* Images: the live state holds A at 100 and B at 3,000 (copies at
     * 2,048 and 5,000). */
    const inkling2_image A = {100, 50, 11}, B = {3000, 20, 22}, other_b = {3000, 20, 23};
    const inkling2_image short_b = {3000, 19, 22}, extra = {4000, 8, 33}, C = {4800, 9, 44}, D = {4800, 9, 45};
    const inkling2_image both[] = {A, B}, changed[] = {A, other_b}, shorter[] = {A, short_b};
    const inkling2_image added[] = {A, B, extra}, late_live[] = {A, C}, late_asked[] = {A, D};
    live(5000, 2048, 5000);
    images(both, 2, both, 2);
    ok &= check("same_images", keep_for(5000, 5100), 5000);
    images(both, 2, changed, 2);
    ok &= check("other_pixels", keep_for(5000, 5100), 2048);
    images(both, 2, shorter, 2);
    ok &= check("other_run", keep_for(5000, 5100), 2048);
    images(both, 2, both, 1);
    ok &= check("image_dropped", keep_for(5000, 5100), 2048);
    images(both, 2, added, 3);
    ok &= check("image_added_below", keep_for(5000, 5100), 2048);
    /* An image at or past the common prefix does not cut it. */
    live(5000, 4096, 5000);
    images(late_live, 2, late_asked, 2);
    ok &= check("image_past_prefix", keep_for(4500, 4600), 4096);
    images(both, 0, both, 0);

    live(5000, 4096, 5000);
    s.inkling2_graph.failed = true;
    ok &= check("failed_graph", keep_for(5000, 5100), 0);
    live(5000, 4096, 5000);
    s.inkling2_graph.position = 5003;
    ok &= check("trial_in_flight", keep_for(5000, 5100), 0);

    /* Rings of 512- and 2,048-row chunks: unwrapped, wrapped, and copies
     * below the first window (slots under position 0 hold nothing read). */
    const uint32_t rings[] = {512 + DS4_INK2_WINDOW - 1u, 2048 + DS4_INK2_WINDOW - 1u};
    const uint32_t ats[] = {1, 1023, 1024, 1535, 1536, 2047, 3071, 3072, 4095, 5000, 1048575};
    for (int r = 0; r < 2; r++) {
        for (int a = 0; a < (int)(sizeof ats / sizeof ats[0]); a++) { ok &= spans(rings[r], ats[a]); }
    }
    return ok ? 0 : 1;
}
'''.replace('@ROWS@', cut(bind, 'enum {\n    INKLING2_CONTEXT', 'static uint32_t inkling2_prefill_cap(')) \
   .replace('@FULL@', 'enum { INKLING2_FULL_LAYERS = INKLING2_LAYERS / DS4_INK2_BLOCK };\n') \
   .replace('@GLOBAL@', cut(bind, 'static bool inkling2_global(', 'static bool inkling2_boundary(')) \
   .replace('@COMMON@', cut(ds4, 'int ds4_session_common_prefix(', 'static bool session_logits_unready(')) \
   .replace('@PREFIX@', cut(ds4, 'static uint32_t session_keep_prefix(', '/* Whether a sliding ring')) \
   .replace('@CKPT@', cut(ds4, 'static bool serial_ckpt_live(', '/* One tensor span')) \
   .replace('@REWOUND@', cut(ds4, 'static void session_rewound(', '/* GLM-5.3:')) \
   .replace('@KEEP@', cut(session, "// A media sync's placeholder", 'static void inkling2_resume_free('))
with tempfile.TemporaryDirectory() as t:
    p = Path(t)
    (p / 'keep.c').write_text(code)
    subprocess.run(['cc', '-std=gnu11', '-O2', '-Wall', '-Werror', f'-I{repo}', str(p / 'keep.c'),
                    '-o', str(p / 'keep')], check=True)
    raise SystemExit(subprocess.run([str(p / 'keep')]).returncode)
