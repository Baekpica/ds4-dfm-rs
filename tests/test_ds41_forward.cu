/* tests/test_ds41_forward.cu — the P4-4 trace gate harness.
 *
 * Runs the port's V4.1 score entry (ds4_v41_score_ids, ds4_ds41_forward.inc)
 * over a golden prompt's exact ids and compares the per-layer dumps and the
 * logits against the golden directory:
 *
 *   test_ds41_forward <gguf> <golden_dir> <out_dir> [--engram-dir <dir>]
 *                     [--no-engram] [--zchain <dir>] <name>...
 *
 * The golden set (tests/capture_ds41_golden.sh) holds per prompt:
 *   <name>.ids                  the exact token ids (the engine tokenized them)
 *   <name>.logits.bin           header {n, V} + n x V f32
 *   <name>.logits.bin.x_Lnn.bin MoE input per layer (n x E f32)
 *   <name>.logits.bin.y_Lnn.bin MoE output per layer (n x E f32)
 *   <name>.logits.bin.hce_Lnn.bin hc after the engram layer (n x HC x E f32)
 *   <name>.logits.bin.erows_Lnn.txt the engram row ids (n lines x cols)
 *
 * Two golden variants exist (tests/capture_ds41_golden.sh): the default
 * capture ran the engine WITH --zchain (the sidecar: gr=39, rb=27 layers in
 * the real directory), NO_ZCHAIN=1 the bare variant.  --zchain <dir> makes
 * this harness mount that directory through the two GPU stores
 * (ds4_gpu_v41_set_gr_override / ds4_gpu_v41_set_rb_override) before
 * scoring, so the with-sidecar golden gates the port's own sidecar
 * application; without the flag the bare golden gates the bare path.
 *
 * Engram rows: the harness reads the golden erows txt (the engine's own row
 * ids) and preads the raw row bytes from the official shards, exactly the
 * host half the Rust host will own in production.  --engram-dir keeps the
 * metadata's basename and prefixes the given directory (the engine's rule).
 *
 * The comparison is a diagnostic report: per-layer bit diffs, max abs/rel,
 * argmax agreement and the recomputed PPL for both files.  The gate the
 * operator records is read from these numbers (the scalar-vs-mma attention
 * deviation means the traces are NOT bit-equal; see ds41_attn.cuh).  The
 * harness fails on: score error, non-finite values in any port dump, or a
 * malformed logits file. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_fp16.h>

extern "C" {
#include "../ds4.h"
#include "../ds4_gpu.h"
#include "../ds41_forward.h"
}

static bool file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static std::vector<uint8_t> slurp(const char *path) {
    std::vector<uint8_t> out;
    FILE *f = fopen(path, "rb");
    if (!f) return out;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n > 0) {
        out.resize((size_t)n);
        if (fread(out.data(), 1, (size_t)n, f) != (size_t)n) out.clear();
    }
    fclose(f);
    return out;
}

static bool parse_ids(const char *path, std::vector<int> *ids) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "harness: cannot read %s\n", path); return false; }
    long v;
    while (fscanf(f, "%ld", &v) == 1) ids->push_back((int)v);
    fclose(f);
    return !ids->empty();
}

/* ---- the zchain sidecar (unit D) ----
 * Gate-side mirror of the engine's loader, core_v41_amp.c:44-117, for the
 * single (2) directory: the capture ran --zchain without --posttrain, so
 * there is no (3) to merge.  Production reads the same files on the Rust host
 * (crates/ds4-core/src/sidecar.rs) and calls the two GPU stores; this C copy
 * exists because the harness is the gate driver.  A present-but-broken file
 * is a hard stop, never a skip (half a plugin is a fake reading). */

/* ds4_fp8.h:39-44 (E8M0: e=0 is the 0x00400000 bit pattern) and
 * ds4_fp8.h:121-126 (the E2M1 nibble table); ds4_quantfmt.c:28-40 is the
 * 17-byte block decode the fp4x32 gr files use (s-1 on disk). */
static float e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float f; memcpy(&f, &bits, 4); return f;
}

static const float kFp4Nibble[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static void deq_fp4x32(const uint8_t *src, size_t nblk, float *out) {
    for (size_t b = 0; b < nblk; b++) {
        const uint8_t *blk = src + b * 17u;
        const float s = e8m0_to_f32(blk[16]);
        float *o = out + b * 32u;
        for (int j = 0; j < 16; j++) {
            o[2 * j] = kFp4Nibble[blk[j] & 0x0Fu] * s;
            o[2 * j + 1] = kFp4Nibble[blk[j] >> 4] * s;
        }
    }
}

/* gr_Lnn.bin (core_v41_amp.c:44-84): header {n_expert, D, type}; type 1 = f32
 * s, 2 = f16 s, 43 = fp4x32 storing s-1 (restored as 1+raw).  Mounts the
 * layer's gain table; no file = the layer stays bare. */
static bool zchain_gr_mount(const char *dir, uint32_t il, uint32_t n_expert, uint32_t n_embd) {
    char path[4600];
    snprintf(path, sizeof path, "%s/gr_L%02u.bin", dir, il);
    FILE *f = fopen(path, "rb");
    if (!f) return true;
    int32_t hd[3];
    if (fread(hd, 4, 3, f) != 3 || hd[0] != (int32_t)n_expert || hd[1] != (int32_t)n_embd ||
        (hd[2] != 1 && hd[2] != 2 && hd[2] != 43)) {
        fprintf(stderr, "harness: %s header bad (expert %d d %d type %d; want %u/%u/1|2|43)\n",
                path, hd[0], hd[1], hd[2], n_expert, n_embd);
        fclose(f);
        return false;
    }
    const size_t nel = (size_t)n_expert * n_embd;
    std::vector<float> acc(nel);
    bool ok = true;
    if (hd[2] == 43) {
        if (nel % 32u) { ok = false; }
        else {
            std::vector<uint8_t> pk(nel / 32u * 17u);
            ok = fread(pk.data(), 1, pk.size(), f) == pk.size();
            if (ok) {
                deq_fp4x32(pk.data(), pk.size() / 17u, acc.data());
                for (size_t t = 0; t < nel; t++) acc[t] += 1.0f;
            }
        }
    } else if (hd[2] == 2) {
        std::vector<uint16_t> raw(nel);
        ok = fread(raw.data(), 2, nel, f) == nel;
        if (ok) for (size_t t = 0; t < nel; t++) acc[t] = __half2float(*(__half *)&raw[t]);
    } else {
        ok = fread(acc.data(), 4, nel, f) == nel;
    }
    fclose(f);
    if (!ok) { fprintf(stderr, "harness: %s truncated\n", path); return false; }
    if (!ds4_gpu_v41_set_gr_override(il, acc.data(), n_expert, n_embd)) {
        fprintf(stderr, "harness: L%02u gain override upload failed\n", il);
        return false;
    }
    return true;
}

/* rb_Lnn.bin (core_v41_amp.c:86-117): header {n_expert, 1=f32} + delta f32.
 * The store adds it to the on-disk exp_probs_b (selection score only). */
static bool zchain_rb_mount(void *engine, const char *dir, uint32_t il, uint32_t n_expert) {
    char path[4600];
    snprintf(path, sizeof path, "%s/rb_L%02u.bin", dir, il);
    FILE *f = fopen(path, "rb");
    if (!f) return true;
    int32_t hd[2];
    if (fread(hd, 4, 2, f) != 2 || hd[0] != (int32_t)n_expert || hd[1] != 1) {
        fprintf(stderr, "harness: %s header bad (expert %d type %d; want %u/1=f32)\n", path, hd[0], hd[1], n_expert);
        fclose(f);
        return false;
    }
    std::vector<float> delta(n_expert);
    const bool ok = fread(delta.data(), 4, n_expert, f) == n_expert;
    fclose(f);
    if (!ok) { fprintf(stderr, "harness: %s truncated\n", path); return false; }
    const void *map = NULL;
    uint64_t size = 0, off = 0;
    if (!ds4_v41_router_bias_ref(engine, il, &map, &size, &off)) {
        fprintf(stderr, "harness: L%02u has no exp_probs_b tensor for the rb sidecar\n", il);
        return false;
    }
    if (!ds4_gpu_v41_set_rb_override(map, size, off, delta.data(), n_expert)) {
        fprintf(stderr, "harness: L%02u route-bias override upload failed\n", il);
        return false;
    }
    return true;
}

/* Mount the whole directory in the engine's order (v41_amp_load,
 * core_v41_amp.c:227-229): rb unloads all first, then per-layer mounts. */
static bool zchain_load(void *engine, const char *dir) {
    uint32_t n_layer = 0, n_expert = 0, n_embd = 0;
    if (!ds4_v41_shape(engine, &n_layer, &n_expert, &n_embd)) {
        fprintf(stderr, "harness: shape query failed\n");
        return false;
    }
    (void)ds4_gpu_v41_set_rb_override(NULL, 0, 0, NULL, 0);
    uint32_t n_gr = 0, n_rb = 0;
    for (uint32_t il = 0; il < n_layer; il++) {
        char path[4600];
        snprintf(path, sizeof path, "%s/gr_L%02u.bin", dir, il);
        if (file_exists(path)) { if (!zchain_gr_mount(dir, il, n_expert, n_embd)) return false; n_gr++; }
        snprintf(path, sizeof path, "%s/rb_L%02u.bin", dir, il);
        if (file_exists(path)) { if (!zchain_rb_mount(engine, dir, il, n_expert)) return false; n_rb++; }
    }
    printf("zchain: gr=%u rb=%u layers mounted from %s\n", n_gr, n_rb, dir);
    return true;
}

static bool parse_erows(const char *path, uint32_t n, uint32_t cols, std::vector<int64_t> *rows) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "harness: cannot read %s\n", path); return false; }
    rows->assign((size_t)n * cols, 0);
    for (size_t i = 0; i < rows->size(); i++) {
        long long v;
        if (fscanf(f, "%lld", &v) != 1) { fclose(f); fprintf(stderr, "harness: %s short at %zu\n", path, i); return false; }
        (*rows)[i] = (int64_t)v;
    }
    fclose(f);
    return true;
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

struct stats { uint64_t diffs, nonfinite; double max_abs, max_rel; };

static stats cmp_f32(const float *a, const float *b, size_t n) {
    stats s = {0, 0, 0.0, 0.0};
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(a[i])) s.nonfinite++;
        if (memcmp(&a[i], &b[i], 4) == 0) continue;
        s.diffs++;
        const double d = fabs((double)a[i] - (double)b[i]);
        const double scale = fmax(fmax(fabs((double)a[i]), fabs((double)b[i])), 1e-30);
        if (d > s.max_abs) s.max_abs = d;
        if (d / scale > s.max_rel) s.max_rel = d / scale;
    }
    return s;
}

static const float *as_f32(const std::vector<uint8_t> &v) { return (const float *)v.data(); }

/* Teacher-forced PPL over the rows of a logits file (the engine's formula). */
static double ppl_of(const float *lg, uint32_t n, uint32_t V, const std::vector<int> &ids) {
    double nll = 0.0;
    for (uint32_t i = 0; i + 1u < n; i++) {
        const float *row = lg + (size_t)i * V;
        const int tgt = ids[i + 1u];
        float mx = row[0];
        for (uint32_t v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
        double se = 0.0;
        for (uint32_t v = 0; v < V; v++) se += exp((double)row[v] - mx);
        nll += -((double)row[tgt] - mx - log(se));
    }
    return n > 1u ? exp(nll / (double)(n - 1u)) : 0.0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <gguf> <golden_dir> <out_dir> [--engram-dir <dir>] [--no-engram] [--zchain <dir>] <name>...\n", argv[0]);
        return 2;
    }
    const char *gguf = argv[1], *golden = argv[2], *out_dir = argv[3];
    const char *engram_dir = NULL;
    const char *zchain_dir = NULL;
    int no_engram = 0;
    std::vector<std::string> names;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--engram-dir") == 0 && i + 1 < argc) { engram_dir = argv[++i]; continue; }
        if (strcmp(argv[i], "--no-engram") == 0) { no_engram = 1; continue; }
        if (strcmp(argv[i], "--zchain") == 0 && i + 1 < argc) { zchain_dir = argv[++i]; continue; }
        names.push_back(argv[i]);
    }
    if (names.empty()) { fprintf(stderr, "harness: no prompt names\n"); return 2; }
    mkdir(out_dir, 0777);

    ds4_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = gguf;
    opt.backend = DS4_BACKEND_CUDA;
    /* The boot prewarm runs a throwaway session sync; V4.1 has no session
     * graph in the port yet (the serving unit), so it segfaults there.  The
     * engine's own option is exactly for callers that do not want it: the
     * first request pays the one-time driver costs instead. */
    opt.defer_boot_prewarm = true;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) { fprintf(stderr, "harness: engine open failed\n"); return 1; }
    if (zchain_dir && !zchain_load(e, zchain_dir)) {
        fprintf(stderr, "harness: zchain load failed -- hard stop (no silent bare run)\n");
        return 1;
    }

    int failures = 0;
    for (const std::string &name : names) {
        char path[4600];
        snprintf(path, sizeof path, "%s/%s.ids", golden, name.c_str());
        std::vector<int> ids;
        if (!parse_ids(path, &ids)) { failures++; continue; }
        const uint32_t n = (uint32_t)ids.size();

        /* Build the engram feed: golden row ids + raw rows pread from the shards. */
        ds41_engram_feed feed;
        memset(&feed, 0, sizeof feed);
        feed.pos0 = 0;   /* one block from position 0 (every golden name here is <= DS4_V41_CHUNK, so the whole sequence fits the feed and no per-chunk refill is needed) */
        std::vector<std::vector<uint8_t>> raw_keep;   /* pins the pinned pointers' lifetime */
        std::vector<std::vector<int64_t>> rows_keep;
        const int n_eng = no_engram ? 0 : ds4_v41_engram_count(e);
        for (int k = 0; k < n_eng; k++) {
            uint32_t il = 0, hd = 0, cols = 0;
            uint64_t nrows = 0, woff = 0, soff = 0;
            char tpath[4096];
            if (!ds4_v41_engram_meta(e, k, &il, tpath, sizeof tpath, &nrows, &woff, &soff, &hd, &cols)) { failures++; break; }
            snprintf(path, sizeof path, "%s/%s.logits.bin.erows_L%02u.txt", golden, name.c_str(), il);
            std::vector<int64_t> rows;
            if (!parse_erows(path, n, cols, &rows)) { failures++; break; }
            const uint32_t stride = hd + hd / 32u;
            void *raw = ds4_gpu_host_alloc((uint64_t)n * cols * stride);
            if (!raw) { fprintf(stderr, "harness: pinned row buffer allocation failed\n"); failures++; break; }
            /* --engram-dir: keep the metadata basename, prefix the directory (the engine's rule) */
            std::string shard = tpath;
            if (engram_dir) {
                const size_t slash = shard.find_last_of('/');
                shard = std::string(engram_dir) + "/" + (slash == std::string::npos ? shard : shard.substr(slash + 1));
            }
            const int fd = open(shard.c_str(), O_RDONLY);
            if (fd < 0) { fprintf(stderr, "harness: cannot open engram shard %s\n", shard.c_str()); failures++; break; }
            bool ok = true;
            for (uint32_t p = 0; p < n && ok; p++) {
                for (uint32_t c = 0; c < cols && ok; c++) {
                    const int64_t r = rows[(size_t)p * cols + c];
                    if (r < 0 || (uint64_t)r >= nrows) { fprintf(stderr, "harness: row id %lld out of range\n", (long long)r); ok = false; break; }
                    uint8_t *dst = (uint8_t *)raw + ((size_t)p * cols + c) * stride;
                    ok = pread_exact(fd, dst, hd, woff + (uint64_t)r * hd) &&
                         pread_exact(fd, dst + hd, hd / 32u, soff + (uint64_t)r * (hd / 32u));
                }
            }
            close(fd);
            if (!ok) { fprintf(stderr, "harness: engram pread failed (L%02u)\n", il); failures++; break; }
            feed.raw[k] = raw;
            raw_keep.emplace_back();   /* keep the vector alive; raw itself is pinned memory owned by ds4_gpu_host_alloc */
            rows_keep.push_back(std::move(rows));
            feed.rows[k] = rows_keep.back().data();
        }
        if (failures) continue;

        snprintf(path, sizeof path, "%s/%s.logits.bin", out_dir, name.c_str());
        const std::string out_logits = path;
        printf("== %s: n=%u, running the score entry -> %s\n", name.c_str(), n, out_logits.c_str());
        fflush(stdout);
        const int rc = ds4_v41_score_ids(e, ids.data(), (int)n, out_logits.c_str(), no_engram, 0, n_eng ? &feed : NULL);
        if (rc != 0) { fprintf(stderr, "harness: score entry failed for %s\n", name.c_str()); failures++; continue; }

        /* Per-layer dumps: golden vs ours. */
        uint32_t E = 0, HC = 0;
        int layers = 0;
        for (uint32_t il = 0; il < 128u; il++) {
            char gx[4600], ox[4600];
            snprintf(gx, sizeof gx, "%s/%s.logits.bin.x_L%02u.bin", golden, name.c_str(), il);
            if (!file_exists(gx)) break;
            snprintf(ox, sizeof ox, "%s/%s.logits.bin.x_L%02u.bin", out_dir, name.c_str(), il);
            std::vector<uint8_t> gxv = slurp(gx), oxv = slurp(ox);
            if (gxv.empty() || oxv.size() != gxv.size()) { fprintf(stderr, "harness: x_L%02u size mismatch (golden %zu ours %zu)\n", il, gxv.size(), oxv.size()); failures++; break; }
            if (E == 0) E = (uint32_t)(gxv.size() / 4u / n);
            char gy[4600], oy[4600];
            snprintf(gy, sizeof gy, "%s/%s.logits.bin.y_L%02u.bin", golden, name.c_str(), il);
            snprintf(oy, sizeof oy, "%s/%s.logits.bin.y_L%02u.bin", out_dir, name.c_str(), il);
            std::vector<uint8_t> gyv = slurp(gy), oyv = slurp(oy);
            const stats sx = cmp_f32(as_f32(oxv), as_f32(gxv), gxv.size() / 4u);
            const stats sy = (oyv.size() == gyv.size() && !gyv.empty()) ? cmp_f32(as_f32(oyv), as_f32(gyv), gyv.size() / 4u) : stats{0, 0, -1.0, -1.0};
            printf("L%02u x: diffs %llu/%zu max_abs %.3e max_rel %.3e%s | y: diffs %llu/%zu max_abs %.3e max_rel %.3e%s\n",
                   il,
                   (unsigned long long)sx.diffs, gxv.size() / 4u, sx.max_abs, sx.max_rel, sx.nonfinite ? " NONFINITE" : "",
                   (unsigned long long)sy.diffs, gyv.size() / 4u, sy.max_abs, sy.max_rel, sy.nonfinite ? " NONFINITE" : "");
            if (sx.nonfinite || sy.nonfinite) failures++;
            layers++;
        }
        if (!layers) { fprintf(stderr, "harness: no golden x_L files under %s for %s\n", golden, name.c_str()); failures++; continue; }

        /* Engram hc dumps. */
        for (int k = 0; k < n_eng; k++) {
            uint32_t il = 0;
            if (!ds4_v41_engram_meta(e, k, &il, NULL, 0, NULL, NULL, NULL, NULL, NULL)) continue;
            char gh[4600], oh[4600];
            snprintf(gh, sizeof gh, "%s/%s.logits.bin.hce_L%02u.bin", golden, name.c_str(), il);
            snprintf(oh, sizeof oh, "%s/%s.logits.bin.hce_L%02u.bin", out_dir, name.c_str(), il);
            if (!file_exists(gh)) { printf("hce L%02u: golden absent, skipped\n", il); continue; }
            std::vector<uint8_t> ghv = slurp(gh), ohv = slurp(oh);
            if (ghv.empty() || ohv.size() != ghv.size()) { fprintf(stderr, "harness: hce_L%02u size mismatch\n", il); failures++; continue; }
            if (HC == 0) HC = (uint32_t)(ghv.size() / 4u / n / (E ? E : 1u));
            const stats s = cmp_f32(as_f32(ohv), as_f32(ghv), ghv.size() / 4u);
            printf("hce L%02u: diffs %llu/%zu max_abs %.3e max_rel %.3e%s\n", il,
                   (unsigned long long)s.diffs, ghv.size() / 4u, s.max_abs, s.max_rel, s.nonfinite ? " NONFINITE" : "");
            if (s.nonfinite) failures++;
        }

        /* Logits: argmax agreement + both PPLs. */
        char gl_path[4600];
        snprintf(gl_path, sizeof gl_path, "%s/%s.logits.bin", golden, name.c_str());
        std::vector<uint8_t> glv = slurp(gl_path), olv = slurp(out_logits.c_str());
        if (glv.size() < 8 || olv.size() < 8) { fprintf(stderr, "harness: logits file missing/short\n"); failures++; continue; }
        int32_t ghd[2], ohd[2];
        memcpy(ghd, glv.data(), 8); memcpy(ohd, olv.data(), 8);
        const uint32_t V = (uint32_t)ghd[1];
        if (ghd[0] != (int32_t)n || ohd[0] != (int32_t)n || ohd[1] != ghd[1]) { fprintf(stderr, "harness: logits header mismatch\n"); failures++; continue; }
        const float *gl = as_f32(glv) + 2, *ol = as_f32(olv) + 2;
        const stats sl = cmp_f32(ol, gl, (size_t)n * V);
        uint32_t am_bad = 0;
        for (uint32_t i = 0; i < n; i++) {
            const float *gr = gl + (size_t)i * V, *orow = ol + (size_t)i * V;
            int ga = 0, oa = 0;
            for (uint32_t v = 1; v < V; v++) { if (gr[v] > gr[ga]) ga = (int)v; if (orow[v] > orow[oa]) oa = (int)v; }
            if (ga != oa) am_bad++;
        }
        const double gppl = ppl_of(gl, n, V, ids), oppl = ppl_of(ol, n, V, ids);
        printf("logits: diffs %llu/%zu max_abs %.3e max_rel %.3e | argmax %u/%u rows | PPL golden %.4f ours %.4f (delta %.2f%%)\n",
               (unsigned long long)sl.diffs, (size_t)n * V, sl.max_abs, sl.max_rel, n - am_bad, n, gppl, oppl,
               gppl > 0.0 ? 100.0 * (oppl - gppl) / gppl : 0.0);
        if (sl.nonfinite) failures++;
    }
    printf("DS41 forward gate: %s\n", failures ? "FAIL" : "PASS (report above; the operator records the criterion)");
    return failures ? 1 : 0;
}
