/* tests/test_ds41_generate.cu — the greedy-generate gate harness (unit E, E0).
 *
 * Runs the port's generate entry (ds4_v41_generate_argmax, ds4_ds41_forward.inc)
 * over a golden prompt's exact ids and compares the produced token ids against
 * the engine's own --emit-trace capture:
 *
 *   test_ds41_generate <gguf> <ids_file> <engine_log> <n_predict>
 *                      [--engram-dir <shard dir>] [--no-engram]
 *                      [--rows-ref <score dump prefix>]
 *
 * The engine log is the stderr of
 *   ds4 --cuda -m <gguf> --engram-dir <dir> --gen-ids <ids_file> -n N
 *       --temp 0 --no-dspark --emit-trace
 * whose `[emit] <absolute position> <token id>` lines are the golden sequence
 * (core_v41_api.c:246/387; the ids are the exact instrument — the printed text
 * loses the token boundaries).  The port must reproduce the sequence token for
 * token, positions included (a position mismatch = the state advanced wrong).
 *
 * --rows-ref <prefix>: an INDEPENDENT reference for the engram rows the feed
 * delivers at every position — the score dump's `<prefix>.erows_Lnn.txt` over
 * the full prompt+emitted sequence (n <= 64, one block: core_v41_score.c:36).
 * The gate pin is N=56 with the 8-token prompt, so positions 8..63 cover all
 * 56 emitted steps; a longer run's tail would be silently ungated, and the
 * harness refuses that (the coverage print + the verdict).  The comparison is
 * only meaningful while the port still matches the golden (the reference rows
 * are for the golden history), so the check stops at the first id divergence
 * and says so.
 *
 * The feed protocol: the harness derives each step's position from its own
 * token index, declares it in feed->pos0 and the PORT asserts it against the
 * state at the point of use (ds4_ds41_forward.inc) — the E0 class (a drifted
 * counter feeding shifted rows, invisible to every device instrument) now
 * refuses loudly.  With spec decode the per-emit refresh cannot work (a
 * verify batch's rows must exist before the batch, and only the forward knows
 * its tokens), so the harness implements ds41_engram_feed.prepare: the port
 * hands over every block it is about to run and the provider hashes its rows
 * (the engine's full-block hist memcpy, core_v41_forward.c:377).  The final
 * history is checked against the golden sequence — a maintenance bug there is
 * exactly the E0 class.  Two deliberate corruptions keep the gate falsifiable:
 *   DS41_EMIT_OFFSET=n  shifts the declared position itself (the E0 shape):
 *                       the port must refuse the block;
 *   DS41_ROW_SHIFT=n    keeps the declaration right but hashes rows for the
 *                       wrong position: the row reference must MISMATCH while
 *                       the port's assertion stays quiet (the two defenses are
 *                       independent).
 * The spec mode (DS41_VERIFY_K=<k>, the engine's --dspark-verify) compares the
 * per-round [dspark] lines against the engine's capture (g_ds4_v41_prof turns
 * them on); DS41_NO_DSPARK disables speculation (the engine's --no-dspark).
 * The runner (tests/ds41_generate_gate.sh, make test-ds41-generate) requires
 * the positive runs to PASS with full coverage and both controls to FAIL.
 *
 * The engine's generate path has NO no-engram switch (only the score path
 * takes --v41-no-engram), so the gate runs with the engram layers live: the
 * harness hashes each step's rows itself — the C mirror of the Rust host's
 * engram.rs (itself verified against the golden erows), fed through the
 * ds41_engram_feed the port's forward consumes.  --no-engram is a debug-only
 * port-side mode (no engine golden exists for it). */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>

extern "C" {
#include "../ds4.h"
#include "../ds4_gpu.h"
#include "../ds41_forward.h"
}

/* ---- shared helpers (the score harness's, kept local) ---- */

static bool parse_ids(const char *path, std::vector<int> *ids) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "harness: cannot read %s\n", path); return false; }
    long v;
    while (fscanf(f, "%ld", &v) == 1) ids->push_back((int)v);
    fclose(f);
    return !ids->empty();
}

static bool parse_engine_emit(const char *path, std::vector<int> *pos, std::vector<int> *ids) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "harness: cannot read %s\n", path); return false; }
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        /* the CLI's token text goes to stdout and [emit] to stderr, both into
         * one file: "[emit] 8 455\nThe[emit] 9 8397" — find the marker anywhere */
        const char *p = strstr(line, "[emit]");
        unsigned pp; int t;
        if (p && sscanf(p, "[emit] %u %d", &pp, &t) == 2) {
            pos->push_back((int)pp);
            ids->push_back(t);
        }
    }
    fclose(f);
    return !ids->empty();
}

static bool pread_exact(int fd, void *dst, size_t len, uint64_t off) {
    uint8_t *p = (uint8_t *)dst; size_t left = len; uint64_t o = off;
    while (left) {
        const ssize_t r = pread(fd, p, left, (off_t)o);
        if (r <= 0) return false;
        p += r; left -= (size_t)r; o += (uint64_t)r;
    }
    return true;
}

/* ---- the live engram feed (unit E) ----
 * The engine hashes rows per step inside its forward; the port consumes a
 * caller-built feed, so the harness owns the hash here — a line-for-line C
 * mirror of crates/ds4-core/src/engram.rs rows(), which mirrors the engine's
 * v41_engram_hash (core_v41_engram.c:94-116) and was verified against the
 * golden erows in P2/P4-3.  The table layouts are the engine's (layer-major),
 * NOT the row-major order the GGUF dims suggest: reading them by dims
 * silently produces different rows, not an error. */

struct engram_layer {
    uint32_t il, cols, hd, stride;
    uint64_t nrows, woff, soff;
    int fd;
    void *raw;      /* pinned [np][cols][stride]; slots 0..n-1 hold the current block's rows */
    /* --rows-ref: the engine's own rows for every position of the reference
     * sequence (the score dump's erows file), compared against every row set
     * this harness hashes; ref_first_bad = -1 until the first mismatch.
     * ref_skip counts positions the comparison had to skip (the port's token
     * there left the golden sequence — a rejected draft is EXPECTED to, so a
     * skip is not a failure; the ids comparison is the failure signal). */
    std::vector<int64_t> ref;
    uint64_t ref_lines;
    uint64_t ref_prompt, ref_emit, ref_skip;   /* positions checked / skipped */
    uint64_t ref_bad;
    int64_t ref_first_bad;
    uint32_t ref_first_col;
};

struct gen_feed {
    ds41_engram_hash_ref ref;
    ds41_engram_feed *feed;         /* pos0 declared per block by feed_prepare (the port asserts it) */
    std::vector<int> *gids;         /* the golden ids */
    std::vector<int32_t> token_map;
    std::vector<int64_t> mult, prim, offs;
    std::vector<int32_t> hist;      /* absolute position -> token id */
    std::vector<int32_t> gold_hist; /* prompt + golden ids: the reference history */
    std::vector<engram_layer> layers;
    uint32_t np;
    uint32_t emitted;
    uint32_t emit_offset;           /* DS41_EMIT_OFFSET: shifts the declared position (the E0 shape) */
    int row_shift;                  /* DS41_ROW_SHIFT: hashes the wrong position, declares the right one */
    uint64_t hist_bad;              /* final history vs golden mismatches */
    int64_t hist_first_bad;
    int32_t eos;
    std::vector<int> *ids;          /* emitted ids for the comparison */
};

/* The model map is the host mmap (registered for device access); the same
 * cudaMemcpyDefault read the engine's loader uses handles either side. */
static bool read_map(const void *map, uint64_t off, uint64_t bytes, void *dst) {
    return cudaMemcpy(dst, (const char *)map + off, (size_t)bytes, cudaMemcpyDefault) == cudaSuccess;
}

static bool hash_tables(gen_feed *g) {
    const ds41_engram_hash_ref *r = &g->ref;
    g->token_map.resize(r->token_map_bytes / 4);
    g->mult.resize(r->mult_bytes / 8);
    g->prim.resize(r->prim_bytes / 8);
    g->offs.resize(r->off_bytes / 8);
    if (!read_map(r->map, r->token_map_off, r->token_map_bytes, g->token_map.data()) ||
        !read_map(r->map, r->mult_off, r->mult_bytes, g->mult.data()) ||
        !read_map(r->map, r->prim_off, r->prim_bytes, g->prim.data()) ||
        !read_map(r->map, r->off_off, r->off_bytes, g->offs.data())) return false;
    return true;
}

/* The rows for position `p` of engram layer index `ei` (engram.rs rows()). */
static void ehash_rows(const gen_feed *g, int64_t p, uint32_t ei, std::vector<int64_t> *rows) {
    const ds41_engram_hash_ref *r = &g->ref;
    const int ng = (int)r->max_ngram, nh = (int)r->heads;
    int64_t prod[16];
    for (int k = 0; k < ng; k++) {
        const int64_t back = p - k;
        int64_t cid;
        if (back < 0) {
            cid = r->pad;
        } else {
            const int32_t tok = g->hist[(size_t)back];
            cid = (tok >= 0 && (uint32_t)tok < r->n_vocab) ? (int64_t)g->token_map[(size_t)tok] : r->pad;
        }
        const int64_t m = g->mult[(size_t)ei * ng + k];
        prod[k] = (int64_t)((uint64_t)cid * (uint64_t)m);   /* the engine's wrap-around multiply */
    }
    rows->clear();
    int64_t rolling = prod[0];
    for (int i = 1; i < ng; i++) {
        rolling ^= prod[i];
        for (int head = 0; head < nh; head++) {
            const int64_t pr = g->prim[((size_t)ei * (ng - 1) + (i - 1)) * nh + head];
            int64_t rr = rolling % pr;
            if (rr < 0) rr += pr;
            rows->push_back(rr + g->offs[(size_t)ei * (ng - 1) * nh + (size_t)(i - 1) * nh + head]);
        }
    }
}

/* Hash the rows for `hash_pos` into `slot`; with `check` on, compare them
 * against the reference line for `declared` — the position the feed claims.
 * DS41_ROW_SHIFT makes the two differ: this check is what proves it (the
 * port's own assertion cannot see a wrong hash target under a right
 * declaration). */
static bool feed_fill(gen_feed *g, uint32_t slot, int64_t declared, int64_t hash_pos, int check) {
    std::vector<int64_t> rows;
    for (uint32_t k = 0; k < g->layers.size(); k++) {
        engram_layer &L = g->layers[k];
        ehash_rows(g, hash_pos, k, &rows);
        if (rows.size() != L.cols) { fprintf(stderr, "harness: hash cols %zu != meta %u\n", rows.size(), L.cols); return false; }
        if (check && !L.ref.empty() && declared >= 0 && (uint64_t)declared < L.ref_lines) {
            const int64_t *want = &L.ref[(size_t)declared * L.cols];
            if (declared < (int64_t)g->np) { L.ref_prompt++; } else { L.ref_emit++; }
            for (uint32_t c = 0; c < L.cols; c++) {
                if (rows[c] == want[c]) { continue; }
                L.ref_bad++;
                if (L.ref_first_bad < 0) {
                    L.ref_first_bad = declared; L.ref_first_col = c;
                    fprintf(stderr, "harness: row reference L%02u: MISMATCH at declared pos %lld col %u (harness row %lld != engine row %lld)\n",
                            L.il, (long long)declared, c, (long long)rows[c], (long long)want[c]);
                }
                break;
            }
        }
        uint8_t *dst = (uint8_t *)L.raw + (size_t)slot * L.cols * L.stride;
        for (uint32_t c = 0; c < L.cols; c++) {
            const int64_t r = rows[c];
            if (r < 0 || (uint64_t)r >= L.nrows) { fprintf(stderr, "harness: engram row %lld out of range\n", (long long)r); return false; }
            uint8_t *d = dst + (size_t)c * L.stride;
            if (!pread_exact(L.fd, d, L.hd, L.woff + (uint64_t)r * L.hd) ||
                !pread_exact(L.fd, d + L.hd, L.hd / 32u, L.soff + (uint64_t)r * (L.hd / 32u))) return false;
        }
    }
    return true;
}

/* FNV-1a 64 over a byte range (the digest both sides print for the payload
 * check: DS41_FEED_DIGEST). */
static uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

static FILE *feed_digest_file(void) {
    static FILE *f = NULL;
    static int tried = 0;
    if (!tried) { tried = 1; const char *p = getenv("DS41_FEED_DIGEST"); if (p) f = fopen(p, "a"); }
    return f;
}

/* The block provider (ds41_engram_feed.prepare): the forward hands over the
 * EXACT block it is about to run — for a spec verify batch that is [round
 * token, draft ids...], which only the forward knows.  The provider writes the
 * block's tokens into the history (the engine's full-block hist memcpy,
 * core_v41_forward.c:377: rejected draft tails are overwritten by the next
 * block's write before any read), hashes every slot's rows and declares the
 * position (the port asserts it at the point of use).  DS41_EMIT_OFFSET shifts
 * the declared position itself (the E0 shape); DS41_ROW_SHIFT hashes the wrong
 * position under a right declaration. */
static int feed_prepare(uint32_t pos0, const int *tokens, int n, void *ud) {
    gen_feed *g = (gen_feed *)ud;
    const int64_t declared = (int64_t)pos0 + (int64_t)g->emit_offset;
    if (n <= 0 || declared < 0 || (size_t)(declared + n) > g->hist.size() ||
        (size_t)(declared + n + g->row_shift) > g->hist.size()) {
        fprintf(stderr, "harness: prepare block %u+%d (declared %lld) outside the history window (%zu)\n",
                pos0, n, (long long)declared, g->hist.size());
        return 1;
    }
    for (int i = 0; i < n; i++) g->hist[(size_t)declared + (size_t)i] = tokens[i];
    /* The reference comparison is valid only where the history still equals
     * the golden sequence (a rejected draft is EXPECTED to differ, so a
     * mismatch skips, never fails; the ids comparison is the failure signal).
     * The in-block prefix decides per slot; the whole history's consistency is
     * re-checked at the end of the run. */
    for (int i = 0; i < n; i++) {
        const int64_t p = declared + (int64_t)i;
        int comparable = 0;
        if (!g->layers.empty() && !g->layers[0].ref.empty()) {
            comparable = 1;
            for (int j = 0; j <= i && comparable; j++) {
                const int64_t q = declared + (int64_t)j;
                comparable = q >= 0 && (size_t)q < g->gold_hist.size() && tokens[j] == g->gold_hist[(size_t)q];
            }
            if (!comparable) {
                for (uint32_t k = 0; k < g->layers.size(); k++) g->layers[k].ref_skip++;
            }
        }
        if (!feed_fill(g, (uint32_t)i, p, p + g->row_shift, comparable)) {
            fprintf(stderr, "harness: feed fill failed at pos %lld\n", (long long)p);
            return 1;
        }
    }
    if (g->feed) { g->feed->pos0 = (uint32_t)declared; }   /* the port asserts this at the point of use */
    FILE *df = feed_digest_file();
    if (df) {
        for (uint32_t k = 0; k < g->layers.size(); k++) {
            const engram_layer &L = g->layers[k];
            fprintf(df, "host pos0 %lld n %d k %u fnv %016llx\n", (long long)declared, n, k,
                    (unsigned long long)fnv1a(L.raw, (size_t)n * L.cols * L.stride));
        }
        fflush(df);
    }
    return 0;
}

/* The emit hook: record the token (the ids are the gate's primary
 * instrument).  The feed itself is the block provider's job now — the verify
 * batch's rows must exist before the batch runs, which only prepare can do. */
static int emit_feed(int token, void *ud) {
    gen_feed *g = (gen_feed *)ud;
    g->ids->push_back(token);
    g->emitted++;
    return 0;
}

/* The row-reference accounting, printed on every path: a check that does not
 * print its coverage is how a dead check passes for a live one. */
static void print_ref_summary(const gen_feed *g) {
    int any = 0;
    for (uint32_t k = 0; k < g->layers.size(); k++) {
        const engram_layer &L = g->layers[k];
        if (L.ref.empty()) { continue; }
        any = 1;
        printf("row reference L%02u: prompt %llu/%u, emitted %llu checked, %llu skipped (draft/golden), %llu mismatches\n",
               L.il, (unsigned long long)L.ref_prompt, g->np, (unsigned long long)L.ref_emit,
               (unsigned long long)L.ref_skip, (unsigned long long)L.ref_bad);
    }
    if (!any) { printf("row reference: none (the emitted positions' rows are ungated)\n"); }
    printf("harness history vs golden: %llu mismatches (first at %s)\n",
           (unsigned long long)g->hist_bad, g->hist_first_bad < 0 ? "none" : std::to_string((long)g->hist_first_bad).c_str());
}

/* The history's final state must equal the accepted (golden) sequence: the
 * rows are hashed from this history, so a maintenance bug here is exactly the
 * E0 class — invisible to every device instrument. */
static void hist_vs_golden(gen_feed *g, size_t emitted) {
    for (size_t i = 0; i < (size_t)g->np + emitted && i < g->gold_hist.size(); i++) {
        if (g->hist[i] == g->gold_hist[i]) { continue; }
        g->hist_bad++;
        if (g->hist_first_bad < 0) { g->hist_first_bad = (int64_t)i; }
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <gguf> <ids_file> <engine_log> <n_predict> [--engram-dir <dir>] [--no-engram] [--rows-ref <score dump prefix>]\n", argv[0]);
        return 2;
    }
    const char *gguf = argv[1], *ids_path = argv[2], *englog = argv[3];
    const int n_predict = atoi(argv[4]);
    const char *engram_dir = NULL, *rows_ref = NULL;
    int no_engram = 0;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--engram-dir") == 0 && i + 1 < argc) { engram_dir = argv[++i]; continue; }
        if (strcmp(argv[i], "--no-engram") == 0) { no_engram = 1; continue; }
        if (strcmp(argv[i], "--rows-ref") == 0 && i + 1 < argc) { rows_ref = argv[++i]; continue; }
        fprintf(stderr, "harness: unknown argument %s\n", argv[i]);
        return 2;
    }

    std::vector<int> prompt;
    if (!parse_ids(ids_path, &prompt)) return 2;
    std::vector<int> gpos, gids;
    if (!parse_engine_emit(englog, &gpos, &gids)) {
        fprintf(stderr, "harness: no [emit] lines in %s (was the engine run with --emit-trace?)\n", englog);
        return 2;
    }
    printf("engine golden: %zu emitted tokens (positions %d..%d)\n", gids.size(),
           gpos.empty() ? -1 : gpos.front(), gpos.empty() ? -1 : gpos.back());

    ds4_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = gguf;
    opt.backend = DS4_BACKEND_CUDA;
    opt.defer_boot_prewarm = true;   /* the V4.1 boot prewarm has no session graph in the port yet */
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) { fprintf(stderr, "harness: engine open failed\n"); return 1; }

    /* The engram feed: the tables from the model map, the rows hashed per
     * position, the raw bytes pread from the shards (the score harness's
     * --engram-dir rule: keep the metadata basename, prefix the directory). */
    gen_feed g;
    memset(&g.ref, 0, sizeof g.ref);
    g.np = (uint32_t)prompt.size();
    /* THE E0 gate bug: the emit hook's row position is np + emitted, and
     * emitted was never initialized — uninitialized stack garbage that is
     * deterministic per binary (the 485e9c9 harness diverged at the first
     * step, the b36a865 one passed; identical addresses, identical payload
     * digests, so the garbage was invisible to every instrument).  Zero it;
     * the port also asserts the declared position against its own state at
     * the point of use now, and DS41_EMIT_OFFSET is the deliberate
     * reproduction: it shifts the declared position the same way the garbage
     * did (the row reference must MISMATCH and the port must refuse). */
    g.emitted = 0;
    g.eos = 0;
    g.ids = NULL;
    g.feed = NULL;
    g.gids = NULL;
    g.emit_offset = 0;
    g.row_shift = 0;
    g.hist_bad = 0;
    g.hist_first_bad = -1;
    if (const char *eo = getenv("DS41_EMIT_OFFSET")) {
        g.emit_offset = (uint32_t)atoi(eo);
        printf("emit offset: %u (deliberate)\n", g.emit_offset);
    }
    if (const char *rs = getenv("DS41_ROW_SHIFT")) {
        g.row_shift = atoi(rs);
        printf("row shift: %d (deliberate)\n", g.row_shift);
    }
    const int n_eng = ds4_v41_engram_count(e);
    if (!no_engram && n_eng > 0) {
        if (!engram_dir) { fprintf(stderr, "harness: the model has engram layers; pass --engram-dir <dir> (or --no-engram for the port-only debug mode)\n"); return 2; }
        if (!ds4_v41_engram_hash_ref(e, &g.ref) || !hash_tables(&g)) { fprintf(stderr, "harness: engram hash tables unavailable\n"); return 1; }
        /* The history window must hold a verify batch past n_predict (block+1
         * rows; DS4_MTP_MAX_BLOCK = 8, ds4_ds41_forward.inc:36). */
        g.hist.assign((size_t)g.np + (size_t)n_predict + 16u + g.emit_offset + (g.row_shift > 0 ? (size_t)g.row_shift : 0u), 0);
        for (uint32_t i = 0; i < g.np; i++) g.hist[i] = prompt[i];
        /* The reference history: prompt + the golden ids, -1 where the golden
         * sequence does not reach (those positions skip the row comparison). */
        g.gold_hist.assign(g.hist.size(), -1);
        for (uint32_t i = 0; i < g.np; i++) g.gold_hist[i] = prompt[i];
        for (size_t i = 0; i < gids.size(); i++) g.gold_hist[(size_t)g.np + i] = gids[i];
        for (int k = 0; k < n_eng; k++) {
            engram_layer L;
            char tpath[4096];
            uint64_t nrows = 0, woff = 0, soff = 0;
            uint32_t hd = 0, cols = 0;
            if (!ds4_v41_engram_meta(e, k, &L.il, tpath, sizeof tpath, &nrows, &woff, &soff, &hd, &cols)) { fprintf(stderr, "harness: engram meta %d failed\n", k); return 1; }
            std::string shard = tpath;
            const size_t slash = shard.find_last_of('/');
            shard = std::string(engram_dir) + "/" + (slash == std::string::npos ? shard : shard.substr(slash + 1));
            L.cols = cols; L.hd = hd; L.stride = hd + hd / 32u; L.nrows = nrows; L.woff = woff; L.soff = soff;
            L.fd = open(shard.c_str(), O_RDONLY);
            if (L.fd < 0) { fprintf(stderr, "harness: cannot open engram shard %s\n", shard.c_str()); return 1; }
            L.raw = ds4_gpu_host_alloc((uint64_t)g.np * cols * L.stride);
            if (!L.raw) { fprintf(stderr, "harness: pinned feed allocation failed\n"); return 1; }
            if (getenv("DS41_ADDR_DUMP")) {   /* the A/B layout discriminator: identical addresses rule the layout class out */
                void *dva = NULL;
                (void)cudaHostGetDevicePointer(&dva, L.raw, 0);
                printf("addr feed k %u host %p dev %p\n", k, L.raw, dva);
            }
            g.layers.push_back(L);
        }
        /* The independent row reference (--rows-ref): the engine's own rows
         * for every position of the full prompt+emitted sequence, from a score
         * dump over those exact ids (single block, n <= 64: core_v41_score.c:36).
         * Loaded per layer; a missing file is a hard stop (the caller asked
         * for the check) and a short file is named SHORT — the tail would
         * otherwise be silently ungated. */
        if (rows_ref) {
            const uint64_t need = (uint64_t)g.np + (uint64_t)gids.size();
            for (uint32_t k = 0; k < g.layers.size(); k++) {
                engram_layer &L = g.layers[k];
                char rp[4600];
                snprintf(rp, sizeof rp, "%s.erows_L%02u.txt", rows_ref, L.il);
                FILE *rf = fopen(rp, "r");
                if (!rf) { fprintf(stderr, "harness: --rows-ref file missing: %s\n", rp); return 2; }
                long long v;
                while (fscanf(rf, "%lld", &v) == 1) { L.ref.push_back((int64_t)v); }
                fclose(rf);
                if (L.ref.empty() || L.ref.size() % L.cols != 0) { fprintf(stderr, "harness: %s: %zu values are not a multiple of cols %u\n", rp, L.ref.size(), L.cols); return 2; }
                L.ref_lines = L.ref.size() / L.cols;
                L.ref_first_bad = -1;
                printf("row reference L%02u: %s (%llu lines; the run needs %llu = %u prompt + %zu emitted)%s\n",
                       L.il, rp, (unsigned long long)L.ref_lines, (unsigned long long)need, g.np, gids.size(),
                       L.ref_lines < need ? "  -- SHORT: the tail is NOT covered" : "");
            }
        }
        /* No initial prompt fill: the prefill block's prepare does it (and
         * every later block), so the multi-chunk case is correct too. */
        printf("engram feed: %zu layers, cols %u, head_dim %u, %u prompt positions\n",
               g.layers.size(), g.layers.empty() ? 0 : g.layers[0].cols, g.layers.empty() ? 0 : g.layers[0].hd, g.np);
        /* Self-check: the engine's own captured rows for this prompt (the
         * golden erows next to the ids file, written by the score capture)
         * must equal what this harness's hash produces — the P2 instrument,
         * re-run here so a hash bug can never masquerade as a port bug.
         * DS41_SKIP_SELFCHECK=1 disables it (the A/B discrimination test:
         * binary A differed from B only by this block). */
        const int skip_sc = getenv("DS41_SKIP_SELFCHECK") != NULL;
        for (uint32_t k = 0; !skip_sc && k < g.layers.size(); k++) {
            std::string ids_path_s = ids_path;
            const size_t dot = ids_path_s.rfind(".ids");
            const std::string base = (dot == std::string::npos) ? ids_path_s : ids_path_s.substr(0, dot);
            char ep[4600];
            snprintf(ep, sizeof ep, "%s.logits.bin.erows_L%02u.txt", base.c_str(), g.layers[k].il);
            FILE *ef = fopen(ep, "r");
            if (!ef) { printf("hash self-check L%02u: %s absent, skipped\n", g.layers[k].il, ep); continue; }
            std::vector<int64_t> erows;
            long long v;
            while (fscanf(ef, "%lld", &v) == 1) erows.push_back(v);
            fclose(ef);
            size_t bad = 0;
            const size_t want = (size_t)g.np * g.layers[k].cols;
            if (erows.size() != want) { printf("hash self-check L%02u: golden %zu rows != %zu expected\n", g.layers[k].il, erows.size(), want); continue; }
            std::vector<int64_t> rows;
            for (uint32_t i = 0; i < g.np; i++) {
                ehash_rows(&g, (int64_t)i, k, &rows);
                for (uint32_t c = 0; c < g.layers[k].cols; c++) if (rows[c] != erows[(size_t)i * g.layers[k].cols + c]) bad++;
            }
            printf("hash self-check L%02u: %zu/%zu rows differ vs the golden erows\n", g.layers[k].il, bad, want);
        }
    }

    std::vector<int> oids;
    g.ids = &oids;
    ds41_engram_feed feed;
    memset(&feed, 0, sizeof feed);
    feed.pos0 = 0;   /* the prefill's prepare declares it too; the port asserts at the point of use */
    feed.prepare = feed_prepare;   /* every block: prefill chunks, decode steps, verify batches */
    feed.ud = &g;
    g.feed = &feed;
    g.gids = &gids;  /* the golden sequence (the reference history's tail) */
    for (uint32_t k = 0; k < g.layers.size(); k++) feed.raw[k] = g.layers[k].raw;   /* k == the engram index (ds41_forward.h) */
    /* DS41_GEN_DELAY_MS=<n>: a bare delay before the generate call (the
     * timing-vs-content discriminator; no reads, no allocations). */
    if (const char *dm = getenv("DS41_GEN_DELAY_MS")) {
        const int ms = atoi(dm);
        if (ms > 0) {
            struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
            nanosleep(&ts, NULL);
            printf("pre-generate delay: %d ms\n", ms);
        }
    }
    /* Spec controls: DS41_VERIFY_K pins the round's k (the engine's
     * --dspark-verify; the trace gate needs it because the scheduler's k
     * follows measured wall-clock costs), DS41_NO_DSPARK disables speculation
     * (the engine's --no-dspark).  g_ds4_v41_prof turns on the per-round
     * [dspark] trace lines (the engine's --v41-prof/--emit-trace). */
    int verify_k = 0;
    if (const char *vk = getenv("DS41_VERIFY_K")) { verify_k = atoi(vk); printf("verify k pinned: %d (--dspark-verify)\n", verify_k); }
    if (getenv("DS41_NO_DSPARK")) { printf("spec decode disabled (DS41_NO_DSPARK; the engine's --no-dspark)\n"); }
    g_ds4_v41_emit_trace = 1;   /* the [dspark] lines (the engine's --emit-trace capture; --v41-prof would add the draft/main sections) */
    const int rc = ds4_v41_generate_argmax(e, prompt.data(), (int)prompt.size(), n_predict,
                                           no_engram || n_eng == 0 ? 1 : 0, verify_k,
                                           g.layers.empty() ? NULL : &feed, emit_feed, &g);
    if (rc != 0) {
        hist_vs_golden(&g, oids.size());
        print_ref_summary(&g);
        fprintf(stderr, "harness: generate entry failed\n");
        return 1;
    }

    size_t diffs = 0, first = SIZE_MAX;
    const size_t n = gids.size() < oids.size() ? gids.size() : oids.size();
    for (size_t i = 0; i < n; i++) {
        if (gids[i] != oids[i]) { diffs++; if (first == SIZE_MAX) first = i; }
    }
    int pos_bad = 0;   /* the engine's own accounting: the first emitted token sits at position np */
    for (size_t i = 0; i < gids.size(); i++) if (gpos[i] != (int)prompt.size() + (int)i) pos_bad++;
    printf("port: %zu emitted tokens; ids match %zu/%zu, first divergence %s; engine positions contiguous from %zu: %s\n",
           oids.size(), n - diffs, n, first == SIZE_MAX ? "none" : std::to_string((long)first).c_str(),
           prompt.size(), pos_bad ? "NO" : "yes");
    if (first != SIZE_MAX) {
        const size_t lo = first >= 4 ? first - 4 : 0;
        for (size_t i = lo; i < n && i < first + 4; i++)
            printf("  %5zu  engine %d  port %d%s\n", i, gids[i], oids[i], i == first ? "  <-- first" : "");
    }
    printf("engine ids:");
    for (size_t i = 0; i < gids.size(); i++) printf(" %d", gids[i]);
    printf("\nport   ids:");
    for (size_t i = 0; i < oids.size(); i++) printf(" %d", oids[i]);
    printf("\n");
    hist_vs_golden(&g, oids.size());
    print_ref_summary(&g);
    /* The verdict: ids match, positions contiguous, the row reference shows no
     * mismatch over everything comparable (its prompt coverage must be full),
     * and the final history equals the golden sequence.  A check narrower than
     * the claim is how the next wrong-index bug survives it. */
    uint64_t ref_bad = 0;
    int ref_short = 0;
    for (uint32_t k = 0; k < g.layers.size(); k++) {
        const engram_layer &L = g.layers[k];
        if (L.ref.empty()) { continue; }
        ref_bad += L.ref_bad;
        if (L.ref_prompt != g.np) { ref_short = 1; }   /* the prompt is always comparable */
    }
    const bool pass = (diffs == 0) && (gids.size() == oids.size()) && !pos_bad && !ref_bad && !ref_short && !g.hist_bad;
    printf("DS41 generate gate: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
