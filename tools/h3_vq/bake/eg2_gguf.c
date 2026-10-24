/* eg2_gguf.c — the baseline EmbeddingGemma-2 GGUF writer (phase 1a, phase-doc
 * rows X.5 / X.5a): the pinned checkpoint re-expressed as a GGUF that the
 * reference loader resolves, so llama.cpp's embedding example (phase 2's anchor)
 * and our own C reference path can read the same file.
 *
 * Every key, type, name and dim below is READ from the file cited next to it,
 * per project rule 0.2; nothing here is invented. The sources:
 *
 *   /data/llama.cpp @ 31288c9c (the form authority, read-only)
 *     src/models/gemma-embedding2.cpp:3-13    the hparams keys the loader resolves
 *     src/models/gemma-embedding2.cpp:34-72   the 413 text-tower tensors and their
 *                                             declared shape, which is what the
 *                                             file's tensor table must match
 *     src/llama-arch.cpp:63-64                the arch string "gemma-embedding2"
 *     src/llama-arch.cpp:202-448              the KV key spellings (LLM_KV_*)
 *     src/llama-model.cpp:1382-1394           context/embedding/out/block_count
 *     src/llama-model.cpp:1462-1471           feed_forward_length, head counts
 *                                             (head_count_kv read as a 24-array)
 *     src/llama-model.cpp:1514-1541           key/value lengths and their SWA pair
 *     src/llama-model.cpp:3902-3912           sliding_window_pattern: array first,
 *                                             scalar fallback
 *     src/llama-hparams.cpp:8-17              set_swa_pattern(6, dense_first=false):
 *                                             is_swa = (il % 6 < 5)
 *     src/llama-hparams.h:524-533             the symmetric mask the pattern feeds
 *     src/llama-vocab.cpp:2236-2276           tokenizer.ggml.model "gemma4" is BPE
 *                                             and REQUIRES tokenizer.ggml.merges
 *     src/llama-vocab.cpp:2595-2637           tokens (required), scores, token_type
 *                                             (optional, length-checked)
 *     conversion/gemma.py:634-641             Gemma4Model.norm_shift = 0.0, so the
 *                                             norm weights are written VERBATIM (the
 *                                             +1.0 shift at :131 is Gemma3's)
 *     conversion/gemma.py:676-733             Gemma4Model.set_gguf_parameters: the
 *                                             exact key set for this arch, emitted
 *                                             as an array sliding_window_pattern,
 *                                             key/value lengths + SWA pair, and the
 *                                             per-layer head_count_kv array
 *     conversion/gemma.py:815-832             EmbeddingGemma2Model.set_gguf_parameters:
 *                                             sliding_window = 2 * sliding_window,
 *                                             embedding_length_out, causal=false,
 *                                             pooling from 1_Pooling/config.json
 *     conversion/gemma.py:641-676             Gemma4Model.set_vocab: model "gemma4",
 *                                             token list + scores + types, merges,
 *                                             add_space_prefix(false), add_bos(true)
 *     gguf-py/gguf/vocab.py:191-330           SpecialVocab: the special ids by
 *                                             content, the chat template, and the
 *                                             add_bos/eos/sep flags derived from
 *                                             tokenizer.json's post_processor
 *                                             (TemplateProcessing<single> here)
 *     gguf-py/gguf/vocab.py:563-684           LlamaHfVocab: tokens are the model.vocab
 *                                             keys by id, score -1000.0, type BYTE for
 *                                             <0xNN>, CONTROL for a special id, else
 *                                             NORMAL; added tokens are those with
 *                                             id >= len(model.vocab)
 *     gguf-py/gguf/vocab.py:191-222           merges taken verbatim when they are a
 *                                             list of plain strings
 *     gguf-py/gguf/constants.py:6274          general.file_type = 32 (MOSTLY_BF16)
 *     gguf-py/gguf/constants.py:6184-6187     PoolingType.MEAN = 1
 *
 *   /data/YoungAi (the writer and payload precedent, revision a4e6d31)
 *     src/common/ds4_gguf_write.h             vendored byte-identical at
 *                                             tools/ds4_gguf_write.h, sha256
 *                                             5bfc5dd4fc8646709d521ad9a212aa0c6cba83cce2f0016baf0ff698364f1fff
 *                                             — the two-phase GGUF v3 writer every
 *                                             ds4 VQ family ships through; used
 *                                             exactly as gguf-tools/quantize/
 *                                             v41_to_gguf.c uses it (head KVs at
 *                                             :357-363, plan/write split at :460-490)
 *     src/common/ds4_st41.h:133-169           v41_st_open / _find / _data, the
 *                                             multi-shard safetensors index
 *
 *   The checkpoint: .models/EmbeddingGemma-2-914f7f8 @ 914f7f89142e, whose
 *   config.json supplies every value (text_config.*) and whose safetensors supply
 *   every tensor byte. Norm weights ship verbatim (Gemma4Model.norm_shift = 0.0).
 *
 * The baseline is the checkpoint FAITHFULLY: all 413 text-tower tensors BF16,
 * including the token table. The artifact (phase 1b) is the one that ships
 * token=q8_0 and the VQ towers, so phase 5's delta against this file is the whole
 * cost of the recipe, and phase 2's anchor is the unquantized original.
 *
 * The artifact mode writes the second file from the DQVD container
 * (tools/eg2_pack.c) rather than from the checkpoint: phase 0 measured the
 * container's payloads byte-identical to a fresh encode
 * (docs/ds4-gemma-payload-contract.md section 2), so the artifact step is a
 * re-wrap — the bytes are copied, only the addressing changes. Sources for it:
 *
 *   /data/YoungAi (the writer and payload precedent, revision a4e6d31)
 *     gguf-tools/quantize/v41_to_gguf.c:137-142   plan_vqblob: the VQ tensor is
 *         registered nd=1, ne={total bytes}, type DS4_GGT_VQBLOB
 *     gguf-tools/quantize/v41_to_gguf.c:285-313   gen_vqblob: the tensor's bytes
 *         ARE the payloads, each beginning with its own 'DQVQ' + dim/nc/rows/cols
 *     gguf-tools/quantize/v41_to_gguf.c:357-363   the VQ family's own header:
 *         architecture, name, alignment 32, recipe — and NO general.file_type
 *     gguf-tools/quantize/vq_qc_bytes.h:15,20      the DQVQ byte account that the
 *         writer and the size allocator share; the artifact checks it per slot
 *     src/common/ds4_quantfmt.c:17                 q8_0 = 32 values in 34 bytes
 *     src/core/core_gguf.c:94-97                   type 42 vqblob: 1 element = 1 byte
 *   ds4-rs/ds41_vq_fmt.h:4-12 and :32               the payload layout, vendored
 *     (the same bytes the container's checker validates, eg2_pack.c:1044)
 *   ds4-rs/crates/ds4-core/src/tensors.rs:106-112   our own reader's type table:
 *         42 vqblob (1 element / 1 byte), 8 q8_0 (32 / 34), 30 bf16 (1 / 2)
 *   The container's own slot table (tools/eg2_pack.c:50-72 for the layout,
 *   :575-577 for the rank convention rows/cols follow for rank 1 and rank 0).
 *
 * Shape and rank come from the checkpoint itself (the v41_st index the baseline
 * already opens), cross-checked against the container's rows/cols: the file
 * declares the checkpoint's true rank, and every payload's own 16-byte header is
 * required to agree with both. Nothing is inferred from a name.
 *
 * Usage:
 *   eg2_gguf baseline <hf-dir> <out.gguf> [--limit N] [--only SUBSTR] [--no-tokenizer] [--manifest FILE]
 *   eg2_gguf artifact <container.bin> <hf-dir> <out.gguf> [--manifest FILE]
 *
 * --limit and --only write a SMOKE file: general.name carries the marker, so a
 * partial artifact cannot be mistaken for the baseline. The artifact mode takes
 * the whole container and refuses a subset.
 *
 * Build (the reference toolchain's include paths, as RECIPE.md section 4 does):
 *   cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L -Itools \
 *      -I/data/YoungAi/src/common -I/data/YoungAi/gguf-tools/quantize \
 *      -o bin/eg2_gguf tools/eg2_gguf.c -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>

#include "ds4_gguf_write.h"    /* vendored: the reference's own GGUF v3 writer */
#include "ds4_st41.h"          /* the reference's safetensors index (locate only) */
#include "vq_qc_bytes.h"       /* the reference's DQVQ byte account (one copy, shared) */

/* GGUF tensor type numbers (ggml), the same table the engine uses
 * (/data/YoungAi/src/core/core_gguf.c:94-97, our crates/ds4-core/src/tensors.rs:60-112).
 * The baseline writes f32 (0) and bf16 (30); the artifact adds q8_0 (8) for its
 * token table and vqblob (42) for its 461 VQ payloads. */
#define GGT_F32    0u
#define GGT_Q8_0   8u
#define GGT_BF16   30u
#define GGT_VQBLOB 42u

/* q8_0's geometry, from the reference's own table (ds4_quantfmt.c:17:
 * DS4_GGT_Q8_0 blk 32, tsz 34). */
#define Q8_0_BLK 32u
#define Q8_0_TSZ 34u

/* gguf-py/gguf/constants.py TokenType (the vocabulary's token_type values) */
#define TKN_NORMAL       1
#define TKN_UNKNOWN      2
#define TKN_CONTROL      3
#define TKN_USER_DEFINED 4
#define TKN_BYTE         6

/* the arch string the loader routes on (src/llama-arch.cpp:64) */
static const char *const ARCH = "gemma-embedding2";

/* the VQ family's own arch string (docs/ds4-gemma-payload-contract.md section 0:
 * the family gate is gemma_vq.format_version, in the shape of glm_vq's) */
static const char *const ARCH_VQ = "gemma_vq";

static void die(const char *m) { fprintf(stderr, "eg2_gguf: %s\n", m); exit(1); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory"); return p; }

/* ------------------------------------------------------------------ JSON ---- */
/* A hand-rolled scanner, the house style (cf. ds4_st41.h:52-90). It needs only
 * what this tool reads: object keys, values, arrays of strings, and numbers. */

static const char *jw(const char *p, const char *e) {
    while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++;
    return p;
}

/* past a string literal; p at the opening quote */
static const char *jstr_end(const char *p, const char *e) {
    if (p >= e || *p != '"') die("json: string expected");
    for (p++; p < e; p++) {
        if (*p == '\\') { p++; if (p >= e) break; continue; }
        if (*p == '"') return p + 1;
    }
    die("json: unterminated string");
    return NULL;
}

/* past one value */
static const char *jskip(const char *p, const char *e) {
    p = jw(p, e);
    if (p >= e) die("json: value expected");
    if (*p == '"') return jstr_end(p, e);
    if (*p == '{' || *p == '[') {
        int depth = 0;
        for (; p < e; p++) {
            if (*p == '"') { p = jstr_end(p, e) - 1; continue; }
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') { if (--depth == 0) return p + 1; }
        }
        die("json: unterminated container");
    }
    while (p < e && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') p++;
    return p;
}

/* value of `key` at the top level of the object starting at obj; NULL if absent.
 * vstart/vend delimit the value. */
static int jobj_get(const char *obj, const char *e, const char *key,
                    const char **vstart, const char **vend) {
    const char *p = jw(obj, e);
    if (p >= e || *p != '{') die("json: object expected");
    size_t klen = strlen(key);
    for (p = jw(p + 1, e); p < e && *p != '}'; p = jw(p, e)) {
        const char *ks = p, *ke = jstr_end(p, e);
        int hit = ((size_t)(ke - ks) == klen + 2) && !strncmp(ks + 1, key, klen);
        p = jw(ke, e);
        if (p >= e || *p != ':') die("json: ':' expected");
        const char *vs = jw(p + 1, e), *ve = jskip(vs, e);
        if (hit) { *vstart = vs; *vend = ve; return 1; }
        p = jw(ve, e);
        if (p < e && *p == ',') p++;
    }
    return 0;
}

/* decode a string literal into fresh bytes (escapes and \uXXXX to UTF-8) */
static char *jstr_dec(const char *p, const char *e, size_t *out_len) {
    if (p >= e || *p != '"') die("json: string expected");
    size_t cap = (size_t)(e - p) + 8, n = 0;
    char *o = xmalloc(cap);
    for (p++; p < e; p++) {
        if (*p == '\\') {
            p++;
            if (p >= e) die("json: bad escape");
            char c = *p;
            switch (c) {
                case 'n': o[n++] = '\n'; break;
                case 't': o[n++] = '\t'; break;
                case 'r': o[n++] = '\r'; break;
                case 'b': o[n++] = '\b'; break;
                case 'f': o[n++] = '\f'; break;
                case '/': o[n++] = '/';  break;
                case '"': o[n++] = '"';  break;
                case '\\': o[n++] = '\\'; break;
                case 'u': {
                    if (p + 4 >= e) die("json: short \\u escape");
                    char h[5] = {p[1], p[2], p[3], p[4], 0};
                    unsigned cp = (unsigned)strtoul(h, NULL, 16);
                    p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 < e && p[1] == '\\' && p[2] == 'u') {
                        char h2[5] = {p[3], p[4], p[5], p[6], 0};
                        unsigned lo = (unsigned)strtoul(h2, NULL, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                            p += 6;
                        }
                    }
                    if (cp < 0x80) o[n++] = (char)cp;
                    else if (cp < 0x800) { o[n++] = (char)(0xC0 | (cp >> 6)); o[n++] = (char)(0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000) { o[n++] = (char)(0xE0 | (cp >> 12)); o[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[n++] = (char)(0x80 | (cp & 0x3F)); }
                    else { o[n++] = (char)(0xF0 | (cp >> 18)); o[n++] = (char)(0x80 | ((cp >> 12) & 0x3F)); o[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[n++] = (char)(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: die("json: unknown escape");
            }
        } else if (*p == '"') {
            o[n] = 0; *out_len = n; return o;
        } else {
            o[n++] = *p;
        }
        if (n + 8 > cap) { cap *= 2; o = realloc(o, cap); if (!o) die("out of memory"); }
    }
    die("json: unterminated string");
    return NULL;
}

static uint64_t jnum_u64(const char *v, const char *e) {
    char buf[64]; size_t n = (size_t)(e - v); if (n >= sizeof buf) die("json: number too long");
    memcpy(buf, v, n); buf[n] = 0;
    return strtoull(buf, NULL, 10);
}
static double jnum_f64(const char *v, const char *e) {
    char buf[64]; size_t n = (size_t)(e - v); if (n >= sizeof buf) die("json: number too long");
    memcpy(buf, v, n); buf[n] = 0;
    return strtod(buf, NULL);
}
static int jis_num(const char *v, const char *e) { return v < e && (*v == '-' || (*v >= '0' && *v <= '9')); }

/* ------------------------------------------------------------------ model --- */

#define MAX_LAYER 64

typedef struct {
    /* text_config, read from the checkpoint's config.json */
    uint32_t n_layer, n_embd, n_head, n_ff, n_embd_out, n_embd_pl, n_vocab;
    uint32_t head_dim_swa, head_dim_full;
    uint32_t n_head_kv_swa, n_head_kv_full;
    uint32_t n_swa_hf;          /* the checkpoint's sliding_window (HF semantics) */
    uint32_t n_ctx;
    float    rms_eps, rope_base, rope_base_swa;
    uint8_t  is_swa[MAX_LAYER];
} hp_t;

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "eg2_gguf: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)sz + 1);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) die("short read");
    b[sz] = 0;
    fclose(f);
    *len = (size_t)sz;
    return b;
}

/* the value at the dotted `path` inside `root` (a top-level object) */
static int jget(const char *root, const char *end, const char *path,
                const char **vs, const char **ve) {
    const char *s = root, *e = end;
    const char *p = path;
    for (;;) {
        const char *dot = strchr(p, '.');
        char key[128];
        size_t n = dot ? (size_t)(dot - p) : strlen(p);
        if (n >= sizeof key) die("json: path segment too long");
        memcpy(key, p, n); key[n] = 0;
        if (!jobj_get(s, e, key, vs, ve)) return 0;
        if (!dot) return 1;
        s = *vs; e = *ve; p = dot + 1;
    }
}

static uint64_t get_u64(const char *root, const char *end, const char *path) {
    const char *vs, *ve;
    if (!jget(root, end, path, &vs, &ve)) { fprintf(stderr, "eg2_gguf: config.json has no %s\n", path); exit(1); }
    if (!jis_num(vs, ve)) { fprintf(stderr, "eg2_gguf: config.json %s is not a number\n", path); exit(1); }
    return jnum_u64(vs, ve);
}
static double get_f64(const char *root, const char *end, const char *path) {
    const char *vs, *ve;
    if (!jget(root, end, path, &vs, &ve)) { fprintf(stderr, "eg2_gguf: config.json has no %s\n", path); exit(1); }
    if (!jis_num(vs, ve)) { fprintf(stderr, "eg2_gguf: config.json %s is not a number\n", path); exit(1); }
    return jnum_f64(vs, ve);
}

/* The hparams, each from the checkpoint's own config.json, cross-checked against
 * the arch's SWA rule (llama-hparams.cpp:8-17: is_swa = il % 6 < 5 for
 * sliding_window_pattern 6 with dense_first false — what LLAMA_SWA_TYPE_SYMMETRIC
 * plus load_swa_pattern(ml, 6) at gemma-embedding2.cpp:4-5 produce). */
static void load_hparams(const char *hfdir, hp_t *H) {
    char path[4300];
    snprintf(path, sizeof path, "%s/config.json", hfdir);
    size_t len;
    char *txt = read_file(path, &len);
    const char *end = txt + len;
    const char *ts, *te;
    if (!jget(txt, end, "text_config", &ts, &te)) die("config.json has no text_config");

    memset(H, 0, sizeof *H);
    H->n_layer        = (uint32_t)get_u64(ts, te, "num_hidden_layers");
    H->n_embd         = (uint32_t)get_u64(ts, te, "hidden_size");
    H->n_head         = (uint32_t)get_u64(ts, te, "num_attention_heads");
    H->n_ff           = (uint32_t)get_u64(ts, te, "intermediate_size");
    H->n_embd_out     = (uint32_t)get_u64(ts, te, "embedding_dim");
    H->n_embd_pl      = (uint32_t)get_u64(ts, te, "hidden_size_per_layer_input");
    H->n_vocab        = (uint32_t)get_u64(ts, te, "vocab_size");
    H->head_dim_swa   = (uint32_t)get_u64(ts, te, "head_dim");
    H->n_head_kv_swa  = (uint32_t)get_u64(ts, te, "num_key_value_heads");
    H->n_swa_hf       = (uint32_t)get_u64(ts, te, "sliding_window");
    H->n_ctx          = (uint32_t)get_u64(ts, te, "max_position_embeddings");
    H->rms_eps        = (float)get_f64(ts, te, "rms_norm_eps");
    H->rope_base      = (float)get_f64(ts, te, "rope_parameters.full_attention.rope_theta");
    H->rope_base_swa  = (float)get_f64(ts, te, "rope_parameters.sliding_attention.rope_theta");
    if (H->n_layer > MAX_LAYER) die("config.json: more layers than MAX_LAYER");

    /* layer_types: the SWA pattern, from the checkpoint itself. The converter
     * writes exactly this array (conversion/gemma.py:681-682). */
    const char *as, *ae, *p;
    if (!jget(ts, te, "layer_types", &as, &ae)) die("config.json has no text_config.layer_types");
    if (*as != '[') die("layer_types is not an array");
    p = jw(as + 1, ae);
    int il = 0;
    while (p < ae && *p != ']') {
        size_t sl;
        char *s = jstr_dec(p, ae, &sl);
        if (il >= (int)H->n_layer) die("layer_types longer than the layer count");
        H->is_swa[il] = (strcmp(s, "sliding_attention") == 0);
        if (!H->is_swa[il] && strcmp(s, "full_attention") != 0) die("unknown layer_types entry");
        free(s);
        il++;
        p = jw(jskip(p, ae), ae);
        if (p < ae && *p == ',') p = jw(p + 1, ae);
    }
    if (il != (int)H->n_layer) die("layer_types shorter than the layer count");

    /* per_layer_config: the full-attention layers' head_dim and KV head count
     * (config.json:99-116). Its keys are zero-padded two-digit strings. */
    H->head_dim_full = H->head_dim_swa;
    H->n_head_kv_full = H->n_head_kv_swa;
    const char *cs, *ce;
    if (jget(ts, te, "per_layer_config", &cs, &ce)) {
        char keybuf[16];
        for (int i = 0; i < (int)H->n_layer; i++) {
            if (H->is_swa[i]) continue;
            snprintf(keybuf, sizeof keybuf, "%02d", i);
            const char *ls, *le;
            if (!jobj_get(cs, ce, keybuf, &ls, &le))
                snprintf(keybuf, sizeof keybuf, "%d", i);      /* unpadded fallback */
            if (!jobj_get(cs, ce, keybuf, &ls, &le))
                { fprintf(stderr, "per_layer_config has no %s\n", keybuf); exit(1); }
            H->head_dim_full  = (uint32_t)get_u64(ls, le, "head_dim");
            H->n_head_kv_full = (uint32_t)get_u64(ls, le, "num_key_value_heads");
        }
    }
    free(txt);

    /* cross-check the SWA rule against the arch's own formula */
    for (int i = 0; i < (int)H->n_layer; i++) {
        int want = (i % 6 < 5);
        if ((int)H->is_swa[i] != want) {
            fprintf(stderr, "eg2_gguf: layer_types[%d] contradicts set_swa_pattern(6,false): "
                            "config says swa=%d, llama-hparams.cpp:8-17 says %d\n",
                    i, H->is_swa[i], want);
            exit(1);
        }
    }
    fprintf(stderr, "[hparams] %u layers, embd %u, heads %u (kv %u swa / %u full), ff %u, "
                    "out %u, per-layer %u, vocab %u, head_dim %u swa / %u full, swa window %u (file %u), ctx %u\n",
            H->n_layer, H->n_embd, H->n_head, H->n_head_kv_swa, H->n_head_kv_full, H->n_ff,
            H->n_embd_out, H->n_embd_pl, H->n_vocab, H->head_dim_swa, H->head_dim_full,
            H->n_swa_hf, 2 * H->n_swa_hf, H->n_ctx);
}

/* the per-layer head geometry, hparams' own helpers (llama-hparams.h:463-472) */
static void layer_geom(const hp_t *H, int il, uint32_t *h_k, uint32_t *h_v, uint32_t *kv) {
    *h_k = H->is_swa[il] ? H->head_dim_swa : H->head_dim_full;
    *h_v = *h_k;
    *kv  = H->is_swa[il] ? H->n_head_kv_swa : H->n_head_kv_full;
}

/* the tensor map: checkpoint suffix (after "language_model.") -> GGUF name, and
 * the dims the loader declares for it (gemma-embedding2.cpp:34-72). GGUF order is
 * inner dimension first, so a checkpoint row [rows, cols] becomes ne=[cols, rows]. */
typedef struct { const char *ckpt, *gguf; int per_layer; } tmap_t;

static const tmap_t TMAP[] = {
    { "embed_tokens.weight",                        "token_embd.weight",            0 },
    { "norm.weight",                                "output_norm.weight",           0 },
    { "embedding_projection.weight",                "output.weight",                0 },
    { "ple.per_layer_model_projection.weight",      "per_layer_model_proj.weight",  0 },
    { "ple.per_layer_projection_norm.weight",       "per_layer_proj_norm.weight",   0 },
    { "layers.%d.input_layernorm.weight",           "blk.%d.attn_norm.weight",            1 },
    { "layers.%d.self_attn.q_proj.weight",          "blk.%d.attn_q.weight",               1 },
    { "layers.%d.self_attn.k_proj.weight",          "blk.%d.attn_k.weight",               1 },
    { "layers.%d.self_attn.v_proj.weight",          "blk.%d.attn_v.weight",               1 },
    { "layers.%d.self_attn.o_proj.weight",          "blk.%d.attn_output.weight",          1 },
    { "layers.%d.self_attn.q_norm.weight",          "blk.%d.attn_q_norm.weight",          1 },
    { "layers.%d.self_attn.k_norm.weight",          "blk.%d.attn_k_norm.weight",          1 },
    { "layers.%d.post_attention_layernorm.weight",  "blk.%d.post_attention_norm.weight",  1 },
    { "layers.%d.pre_feedforward_layernorm.weight", "blk.%d.ffn_norm.weight",             1 },
    { "layers.%d.post_feedforward_layernorm.weight","blk.%d.post_ffw_norm.weight",        1 },
    { "layers.%d.mlp.gate_proj.weight",             "blk.%d.ffn_gate.weight",             1 },
    { "layers.%d.mlp.up_proj.weight",               "blk.%d.ffn_up.weight",               1 },
    { "layers.%d.mlp.down_proj.weight",             "blk.%d.ffn_down.weight",             1 },
    { "layers.%d.ple_block.per_layer_input_gate.weight", "blk.%d.inp_gate.weight",        1 },
    { "layers.%d.ple_block.per_layer_projection.weight", "blk.%d.proj.weight",            1 },
    { "layers.%d.ple_block.post_per_layer_input_norm.weight", "blk.%d.post_norm.weight",  1 },
    { "layers.%d.layer_scalar",                     "blk.%d.layer_output_scale.weight",   1 },
};
#define NTMAP ((int)(sizeof TMAP / sizeof TMAP[0]))

/* the shape the loader declares, per gemma-embedding2.cpp:34-72 */
static uint32_t expect_dims(const hp_t *H, int kind, int il, uint64_t ne[4]) {
    uint32_t h_k, h_v, kv;
    layer_geom(H, il, &h_k, &h_v, &kv);
    ne[0] = ne[1] = ne[2] = ne[3] = 0;
    switch (kind) {
        case 0:  ne[0] = H->n_embd; ne[1] = H->n_vocab; return 2;                  /* token_embd */
        case 1:  ne[0] = H->n_embd; return 1;                                       /* output_norm */
        case 2:  ne[0] = H->n_embd; ne[1] = H->n_embd_out; return 2;                /* output */
        case 3:  ne[0] = H->n_embd; ne[1] = H->n_embd_pl * H->n_layer; return 2;    /* per_layer_model_proj */
        case 4:  ne[0] = H->n_embd_pl; return 1;                                    /* per_layer_proj_norm */
        case 5:  ne[0] = H->n_embd; return 1;                                       /* attn_norm */
        case 6:  ne[0] = H->n_embd; ne[1] = h_k * H->n_head; return 2;               /* attn_q */
        case 7:  ne[0] = H->n_embd; ne[1] = h_k * kv; return 2;                      /* attn_k */
        case 8:  ne[0] = H->n_embd; ne[1] = h_v * kv; return 2;                      /* attn_v */
        case 9:  ne[0] = h_k * H->n_head; ne[1] = H->n_embd; return 2;               /* attn_output */
        case 10: ne[0] = h_k; return 1;                                              /* attn_q_norm */
        case 11: ne[0] = h_k; return 1;                                              /* attn_k_norm */
        case 12: ne[0] = H->n_embd; return 1;                                        /* attn_post_norm */
        case 13: ne[0] = H->n_embd; return 1;                                        /* ffn_norm */
        case 14: ne[0] = H->n_embd; return 1;                                        /* ffn_post_norm */
        case 15: ne[0] = H->n_embd; ne[1] = H->n_ff; return 2;                       /* ffn_gate */
        case 16: ne[0] = H->n_embd; ne[1] = H->n_ff; return 2;                       /* ffn_up */
        case 17: ne[0] = H->n_ff; ne[1] = H->n_embd; return 2;                       /* ffn_down */
        case 18: ne[0] = H->n_embd; ne[1] = H->n_embd_pl; return 2;                  /* inp_gate */
        case 19: ne[0] = H->n_embd_pl; ne[1] = H->n_embd; return 2;                  /* proj */
        case 20: ne[0] = H->n_embd; return 1;                                        /* post_norm */
        case 21: ne[0] = 1; return 1;                                                /* layer_output_scale */
        default: die("expect_dims: unknown kind");
    }
    return 0;
}

/* --------------------------------------------------------------- tokenizer -- */
/* LlamaHfVocab (gguf-py/gguf/vocab.py:563-684), reproduced: tokens are the
 * model.vocab keys by id, score -1000.0, token_type BYTE for <0xNN>, CONTROL for
 * a special id, NORMAL otherwise; any added token with id >= len(model.vocab) is
 * appended in id order as USER_DEFINED (CONTROL if it is one of the specials). */
typedef struct {
    uint32_t n;                 /* tokens in the file (base + appended) */
    char  **tok; size_t *toklen;
    float *score; int32_t *type;
    char **merge; uint32_t n_merge;
    uint32_t bos, eos, unk, pad, mask; int have_bos, have_eos, have_unk, have_pad, have_mask;
    char *chat_template;
} tok_t;

static int is_byte_token(const char *s, size_t n) {
    if (n != 6) return 0;
    if (s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return 0;
    return isxdigit((unsigned char)s[3]) && isxdigit((unsigned char)s[4]);
}

static void tok_load(const char *hfdir, tok_t *T) {
    char path[4300];
    snprintf(path, sizeof path, "%s/tokenizer.json", hfdir);
    size_t len;
    char *txt = read_file(path, &len);
    const char *end = txt + len, *ms, *me, *vs, *ve, *p;

    if (!jget(txt, end, "model", &ms, &me)) die("tokenizer.json has no model");
    { const char *ts, *te;
      if (!jget(ms, me, "type", &ts, &te)) die("tokenizer.json: model.type missing");
      size_t sl; char *s = jstr_dec(ts, te, &sl); int bpe = !strcmp(s, "BPE"); free(s);
      if (!bpe) die("tokenizer.json: model.type is not BPE (LlamaHfVocab requires it)"); }
    if (!jget(ms, me, "vocab", &vs, &ve)) die("tokenizer.json has no model.vocab");

    /* count the entries */
    uint32_t nv = 0;
    for (p = jw(vs + 1, ve); p < ve && *p != '}'; ) {
        p = jstr_end(p, ve); p = jw(p, ve);
        if (p >= ve || *p != ':') die("vocab: ':' expected");
        p = jw(jskip(p + 1, ve), ve);
        if (p < ve && *p == ',') p = jw(p + 1, ve);
        nv++;
    }
    T->n = nv;
    T->tok = xmalloc(sizeof(char *) * nv); T->toklen = xmalloc(sizeof(size_t) * nv);
    T->score = xmalloc(sizeof(float) * nv); T->type = xmalloc(sizeof(int32_t) * nv);
    uint8_t *seen = xmalloc(nv);

    /* the special ids, so token_type can mark CONTROL (SparkVocab's special_ids) */
    /* first the added-token entries of tokenizer.json flagged special */
    uint8_t *is_special = xmalloc(nv);
    memset(seen, 0, nv); memset(is_special, 0, nv);

    const char *as, *ae;
    if (jget(txt, end, "added_tokens", &as, &ae) && *as == '[') {
        for (p = jw(as + 1, ae); p < ae && *p != ']'; ) {
            const char *os, *oe, *ks, *ke;
            os = p; oe = jskip(p, ae);
            if (jobj_get(os, oe, "id", &ks, &ke)) {
                uint64_t id = jnum_u64(ks, ke);
                const char *ss, *se;
                int sp = jobj_get(os, oe, "special", &ss, &se) && *ss == 't';
                const char *cs, *ce;
                if (sp && id < nv) is_special[id] = 1;
                if (id >= nv) {
                    /* appended token: mirror get_added_vocab() + the >= vocab_size filter */
                    if (!jobj_get(os, oe, "content", &cs, &ce)) die("added_tokens: content missing");
                    size_t sl; char *s = jstr_dec(cs, ce, &sl);
                    T->tok = realloc(T->tok, sizeof(char *) * (T->n + 1));
                    T->toklen = realloc(T->toklen, sizeof(size_t) * (T->n + 1));
                    T->score = realloc(T->score, sizeof(float) * (T->n + 1));
                    T->type = realloc(T->type, sizeof(int32_t) * (T->n + 1));
                    T->tok[T->n] = s; T->toklen[T->n] = sl;
                    T->score[T->n] = -1000.0f;
                    T->type[T->n] = sp ? TKN_CONTROL : TKN_USER_DEFINED;
                    T->n++;
                }
            }
            p = jw(oe, ae); if (p < ae && *p == ',') p = jw(p + 1, ae);
        }
    }

    /* the base vocab by id */
    for (p = jw(vs + 1, ve); p < ve && *p != '}'; ) {
        const char *ke = jstr_end(p, ve);
        size_t klen; char *k = jstr_dec(p, ve, &klen);
        (void)klen;
        p = jw(ke, ve);
        if (p >= ve || *p != ':') die("vocab: ':' expected");
        uint64_t id = jnum_u64(jw(p + 1, ve), jskip(p + 1, ve));
        if (id >= nv) die("vocab: id out of range");
        if (seen[id]) die("vocab: duplicate id");
        seen[id] = 1;
        T->tok[id] = k; T->toklen[id] = klen;
        T->score[id] = -1000.0f;
        T->type[id] = is_byte_token(k, klen) ? TKN_BYTE
                    : (is_special[id] ? TKN_CONTROL : TKN_NORMAL);
        p = jw(jskip(p + 1, ve), ve);
        if (p < ve && *p == ',') p = jw(p + 1, ve);
    }
    for (uint32_t i = 0; i < nv; i++) if (!seen[i]) die("vocab: missing id (not a bijection)");
    free(seen); free(is_special);

    /* merges. Two shapes are possible (vocab.py:197-222): a list of plain strings,
     * and — since transformers 4.45 — a list of [a, b] pairs whose parts may
     * contain spaces, which the reference escapes as U+0101 before joining with a
     * single space. Both are implemented; the pair form is what this checkpoint
     * carries (measured: 514906 pairs, no part contains a space). */
    T->n_merge = 0;
    {
        const char *rs, *re;
        if (jget(ms, me, "merges", &rs, &re) && *rs == '[') {
            uint32_t cap = 1024;
            T->merge = xmalloc(sizeof(char *) * cap);
            int escaped = 0;
            for (p = jw(rs + 1, re); p < re && *p != ']'; ) {
                if (T->n_merge == cap) { cap *= 2; T->merge = realloc(T->merge, sizeof(char *) * cap); }
                if (*p == '"') {                       /* the plain-string form */
                    size_t sl; T->merge[T->n_merge++] = jstr_dec(p, re, &sl);
                } else if (*p == '[') {                /* the pair form */
                    const char *pe = jskip(p, re);
                    const char *q = jw(p + 1, pe);
                    size_t l1, l2;
                    char *a = jstr_dec(q, pe, &l1);
                    q = jw(jskip(q, pe), pe);
                    if (q < pe && *q == ',') q = jw(q + 1, pe);
                    char *b = jstr_dec(q, pe, &l2);
                    /* escape spaces in both parts as U+0101, then join with ' ' */
                    size_t need = (l1 + l2) * 2 + 4, n = 0;
                    char *o = xmalloc(need);
                    for (int side = 0; side < 2; side++) {
                        const char *s = side ? b : a; size_t sl = side ? l2 : l1;
                        if (side) o[n++] = ' ';
                        for (size_t i = 0; i < sl; i++) {
                            if (s[i] == ' ') { o[n++] = (char)0xC4; o[n++] = (char)0x81; escaped++; }
                            else o[n++] = s[i];
                        }
                    }
                    o[n] = 0;
                    free(a); free(b);
                    T->merge[T->n_merge++] = o;
                } else {
                    die("merges: neither a string nor a pair");
                }
                p = jw(jskip(p, re), re);
                if (p < re && *p == ',') p = jw(p + 1, re);
            }
            if (escaped) fprintf(stderr, "[tokenizer] %d merge parts carried a space (escaped as U+0101)\n", escaped);
        }
    }

    /* the specials the loader needs by id, from tokenizer_config.json */
    T->have_bos = T->have_eos = T->have_unk = T->have_pad = T->have_mask = 0;
    {
        char tp[4300];
        snprintf(tp, sizeof tp, "%s/tokenizer_config.json", hfdir);
        size_t tl; char *t2 = read_file(tp, &tl);
        const char *tend = t2 + tl;
        static const char *names[5] = { "bos_token", "eos_token", "unk_token", "pad_token", "mask_token" };
        uint32_t *slot[5] = { &T->bos, &T->eos, &T->unk, &T->pad, &T->mask };
        int *have[5] = { &T->have_bos, &T->have_eos, &T->have_unk, &T->have_pad, &T->have_mask };
        for (int i = 0; i < 5; i++) {
            const char *ss, *se;
            if (!jget(t2, tend, names[i], &ss, &se) || *ss != '"') continue;
            size_t sl; char *s = jstr_dec(ss, se, &sl);
            for (uint32_t id = 0; id < T->n; id++)
                if (T->toklen[id] == sl && !memcmp(T->tok[id], s, sl)) { *slot[i] = id; *have[i] = 1; break; }
            free(s);
        }
        /* chat_template.jinja, as SpecialVocab.add_to_gguf writes it */
        char cp[4300];
        snprintf(cp, sizeof cp, "%s/chat_template.jinja", hfdir);
        FILE *f = fopen(cp, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            T->chat_template = xmalloc((size_t)sz + 1);
            if (fread(T->chat_template, 1, (size_t)sz, f) != (size_t)sz) die("short read of chat_template.jinja");
            T->chat_template[sz] = 0;
            fclose(f);
        }
        free(t2);
    }
    free(txt);
    fprintf(stderr, "[tokenizer] %u tokens (%u merges), bos %d eos %d unk %d pad %d mask %d%s\n",
            T->n, T->n_merge, T->have_bos ? (int)T->bos : -1, T->have_eos ? (int)T->eos : -1,
            T->have_unk ? (int)T->unk : -1, T->have_pad ? (int)T->pad : -1,
            T->have_mask ? (int)T->mask : -1,
            T->chat_template ? ", chat template" : "");
}

/* the bool-array key the converter writes (conversion/gemma.py:681-682); the
 * vendored helper has no bool-array writer, and the loader prefers this form
 * over the scalar (llama-model.cpp:3902-3907), so it is written here. */
static void kv_arr_bool(ds4gw *W, const char *k, const uint8_t *v, uint64_t n) {
    ds4gw_put_str(W, k);
    ds4gw_put_u32(W, DS4GW_ARR);
    ds4gw_put_u32(W, DS4GW_BOOL);
    ds4gw_put_u64(W, n);
    ds4gw_put(W, v, n);
    W->nkv++;
}

/* ---- the two metadata blocks, written identically by both modes -------------
 * The arch prefix is the only thing the artifact changes: the hparams and the
 * tokenizer carry this model's geometry and vocabulary either way
 * (docs/ds4-gemma-payload-contract.md section 4). Extracted so that the two
 * files cannot drift in a key value; the emission order of the baseline is
 * unchanged, which its digest still confirms. */

static void emit_hparams(ds4gw *W, const char *arch, const hp_t *H) {
    char k[128];
#define KVK(name) (snprintf(k, sizeof k, "%s." name, arch), k)
    ds4gw_kv_u32(W, KVK("block_count"), H->n_layer);
    ds4gw_kv_u32(W, KVK("context_length"), H->n_ctx);
    ds4gw_kv_u32(W, KVK("embedding_length"), H->n_embd);
    ds4gw_kv_u32(W, KVK("embedding_length_out"), H->n_embd_out);
    ds4gw_kv_u32(W, KVK("embedding_length_per_layer_input"), H->n_embd_pl);
    ds4gw_kv_u32(W, KVK("feed_forward_length"), H->n_ff);
    ds4gw_kv_u32(W, KVK("attention.head_count"), H->n_head);
    ds4gw_kv_u32(W, KVK("attention.key_length"), H->head_dim_full);
    ds4gw_kv_u32(W, KVK("attention.value_length"), H->head_dim_full);
    ds4gw_kv_u32(W, KVK("attention.key_length_swa"), H->head_dim_swa);
    ds4gw_kv_u32(W, KVK("attention.value_length_swa"), H->head_dim_swa);
    ds4gw_kv_f32(W, KVK("attention.layer_norm_rms_epsilon"), H->rms_eps);
    /* HF's sliding_window is one-sided; the converter doubles it so the loader's
     * symmetric mask (llama-hparams.h:524-533, |p1-p0| > n_swa/2) reproduces it
     * (conversion/gemma.py:824-825). */
    ds4gw_kv_u32(W, KVK("attention.sliding_window"), 2 * H->n_swa_hf);
    kv_arr_bool(W, KVK("attention.sliding_window_pattern"), H->is_swa, H->n_layer);
    ds4gw_kv_bool(W, KVK("attention.causal"), 0);
    ds4gw_kv_f32(W, KVK("rope.freq_base"), H->rope_base);
    ds4gw_kv_f32(W, KVK("rope.freq_base_swa"), H->rope_base_swa);
    ds4gw_kv_u32(W, KVK("rope.dimension_count"), H->head_dim_full);
    ds4gw_kv_u32(W, KVK("rope.dimension_count_swa"), H->head_dim_swa);
    ds4gw_kv_u32(W, KVK("vocab_size"), H->n_vocab);
    { uint32_t *kvarr = xmalloc(sizeof(uint32_t) * H->n_layer);
      for (uint32_t i = 0; i < H->n_layer; i++) kvarr[i] = H->is_swa[i] ? H->n_head_kv_swa : H->n_head_kv_full;
      ds4gw_kv_arr_u32(W, KVK("attention.head_count_kv"), kvarr, H->n_layer);
      free(kvarr); }
    ds4gw_kv_u32(W, KVK("pooling_type"), 1);   /* PoolingType.MEAN */
#undef KVK
}

static void emit_tokenizer(ds4gw *W, const tok_t *T) {
    /* The converter's own key set for this arch (conversion/gemma.py:641-676):
     * model "gemma4", the token list, scores and types, then merges and the
     * special ids. It writes no tokenizer.ggml.pre — the loader pins
     * tokenizer_pre = "gemma4" for this model type itself
     * (src/llama-vocab.cpp:2275), so the file does not carry one. */
    ds4gw_kv_str(W, "tokenizer.ggml.model", "gemma4");
    ds4gw_kv_arr_str(W, "tokenizer.ggml.tokens", (const char *const *)T->tok, T->n);
    ds4gw_kv_arr_f32(W, "tokenizer.ggml.scores", T->score, T->n);
    ds4gw_kv_arr_i32(W, "tokenizer.ggml.token_type", T->type, T->n);
    if (T->n_merge) ds4gw_kv_arr_str(W, "tokenizer.ggml.merges", (const char *const *)T->merge, T->n_merge);
    if (T->have_bos) ds4gw_kv_u32(W, "tokenizer.ggml.bos_token_id", T->bos);
    if (T->have_eos) ds4gw_kv_u32(W, "tokenizer.ggml.eos_token_id", T->eos);
    if (T->have_unk) ds4gw_kv_u32(W, "tokenizer.ggml.unknown_token_id", T->unk);
    if (T->have_pad) ds4gw_kv_u32(W, "tokenizer.ggml.padding_token_id", T->pad);
    if (T->have_mask) ds4gw_kv_u32(W, "tokenizer.ggml.mask_token_id", T->mask);
    /* The add_* flags are SpecialVocab's, not the converter's: it derives them
     * from tokenizer.json's post_processor (vocab.py:191-330). This
     * checkpoint's post_processor is TemplateProcessing<single> =
     * [<bos>, A, <eos>], so add_special_token = {bos: true, eos: true,
     * sep: false} — measured, not assumed: tools/ref_vocab_probe.py runs the
     * reference's own SpecialVocab over this checkpoint and prints exactly
     * that (out/ref-specialvocab.gguf carries the keys it writes). Without
     * add_eos the reference tokenizer drops the trailing <eos> the
     * checkpoint's own tokenizer appends (src/llama-vocab.cpp:3625). */
    ds4gw_kv_bool(W, "tokenizer.ggml.add_bos_token", 1);
    ds4gw_kv_bool(W, "tokenizer.ggml.add_eos_token", 1);
    ds4gw_kv_bool(W, "tokenizer.ggml.add_sep_token", 0);
    ds4gw_kv_bool(W, "tokenizer.ggml.add_space_prefix", 0);
    if (T->chat_template) ds4gw_kv_str(W, "tokenizer.chat_template", T->chat_template);
}

/* ------------------------------------------------- the DQVD container (read) --
 * Read-only, for the artifact re-wrap. The container's own structure — magic,
 * version, kind, the name blob, the hashes, ordering, alignment, payload ranges
 * and every byte account — is `bin/eg2_pack_cpu check`'s job (tools/eg2_pack.c:
 * 980-1077), which phase 1b runs first; this reader does not re-implement it. It
 * re-establishes only what the wrap itself stands on — a DQVD v1 kind-0 header,
 * a file whose size the header accounts for, and per slot a name inside the name
 * blob and a payload inside the region — and refuses to write anything if one of
 * those fails. */

enum { CT_VQ = 0, CT_Q4K = 1, CT_VERBATIM = 2, CT_Q8_0 = 3 };
enum { CT_BF16 = 1, CT_F32 = 2, CT_F16 = 3 };

/* the source-dtype code the container records per slot (tools/eg2_pack.c:410-416),
 * derived from the checkpoint's own safetensors dtype */
static int dtype_code(const char *d) {
    if (!strcmp(d, "BF16")) return CT_BF16;
    if (!strcmp(d, "F32"))  return CT_F32;
    if (!strcmp(d, "F16"))  return CT_F16;
    return 0;
}

#define DQVD_MAGIC      0x44565144u   /* 'DQVD' LE  (tools/eg2_pack.c:221) */
#define DQVD_HDR_BYTES  64
#define DQVD_SLOT_BYTES 48
#define DQVQ_MAT_MAGIC  0x51565144u   /* 'DQVQ' LE  (ds4-rs/ds41_vq_fmt.h:32) */

typedef struct {
    char name[192];
    uint32_t rows, cols, payload_len, payload_off;   /* payload_off: from the region start */
    uint16_t dim, nc, flags;
    uint8_t kind, src_dtype;
} cont_slot;

typedef struct {
    FILE *f;
    char path[4300];
    char recipe[512];
    uint64_t payload_off, payload_bytes;
    uint32_t n;
    cont_slot *slot;
} cont_t;

static uint16_t ct_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t ct_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t ct_u64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

static void cont_open(const char *path, cont_t *C) {
    memset(C, 0, sizeof *C);
    snprintf(C->path, sizeof C->path, "%s", path);
    C->f = fopen(path, "rb");
    if (!C->f) { fprintf(stderr, "eg2_gguf: cannot read %s\n", path); exit(1); }
    uint8_t h[DQVD_HDR_BYTES];
    if (fread(h, 1, sizeof h, C->f) != sizeof h) die("container: short header");
    if (ct_u32(h + 0) != DQVD_MAGIC) die("container: magic is not DQVD (hard stop)");
    if (ct_u16(h + 4) != 1) die("container: version is not 1 (hard stop)");
    if (ct_u16(h + 6) != 0) die("container: kind is not 0 (hard stop)");
    C->n = ct_u32(h + 8);
    const uint32_t recipe_len = ct_u32(h + 12);
    const uint64_t recipe_off = ct_u64(h + 16), name_off = ct_u64(h + 24);
    const uint64_t name_bytes = ct_u64(h + 32), slot_off = ct_u64(h + 40);
    C->payload_off = ct_u64(h + 48);
    C->payload_bytes = ct_u64(h + 56);
    fseeko(C->f, 0, SEEK_END);
    const off_t fsz = ftello(C->f);
    if ((uint64_t)fsz != C->payload_off + C->payload_bytes)
        die("container: the file size disagrees with the header");
    if (recipe_len == 0 || recipe_len >= sizeof C->recipe) die("container: recipe length out of range");
    fseeko(C->f, (off_t)recipe_off, SEEK_SET);
    if (fread(C->recipe, 1, recipe_len, C->f) != recipe_len) die("container: short recipe read");
    C->recipe[recipe_len] = 0;
    char *blob = xmalloc((size_t)name_bytes + 1);
    fseeko(C->f, (off_t)name_off, SEEK_SET);
    if (fread(blob, 1, (size_t)name_bytes, C->f) != name_bytes) die("container: short name blob read");
    blob[name_bytes] = 0;
    uint8_t *raw = xmalloc((size_t)C->n * DQVD_SLOT_BYTES);
    fseeko(C->f, (off_t)slot_off, SEEK_SET);
    if (fread(raw, DQVD_SLOT_BYTES, C->n, C->f) != C->n) die("container: short slot table read");
    C->slot = xmalloc(sizeof(cont_slot) * (C->n ? C->n : 1));
    for (uint32_t i = 0; i < C->n; i++) {
        const uint8_t *p = raw + (size_t)i * DQVD_SLOT_BYTES;
        cont_slot *s = &C->slot[i];
        const uint64_t noff = ct_u64(p + 8);
        const uint32_t nlen = ct_u32(p + 16);
        s->payload_off = ct_u32(p + 20); s->payload_len = ct_u32(p + 24);
        s->rows = ct_u32(p + 28); s->cols = ct_u32(p + 32);
        s->dim = ct_u16(p + 36);  s->nc = ct_u16(p + 38);
        s->kind = p[40]; s->src_dtype = p[41]; s->flags = ct_u16(p + 42);
        if (noff + nlen + 1 > name_bytes) die("container: a slot name lies outside the name blob");
        if ((uint64_t)s->payload_off + s->payload_len > C->payload_bytes) die("container: a payload lies outside the region");
        snprintf(s->name, sizeof s->name, "%s", blob + noff);
        if (strlen(s->name) != nlen) die("container: a slot name does not match its recorded length");
        if (s->kind > CT_Q8_0) die("container: unknown slot kind");
        if (i && strcmp(C->slot[i - 1].name, s->name) >= 0) die("container: slot names are not in ascending order");
    }
    free(blob); free(raw);
    fprintf(stderr, "[container] %s: kind=0 slots=%u payloads=%llu+%llu\n            recipe: %s\n",
            path, C->n, (unsigned long long)C->payload_off,
            (unsigned long long)C->payload_bytes, C->recipe);
}

/* the checkpoint name a scheduled GGUF tensor came from (the inverse of TMAP) */
static int ckpt_name_for(const hp_t *H, const char *gguf, char *out) {
    (void)H;
    if (!strncmp(gguf, "blk.", 4)) {
        int il; char suffix[128];
        if (sscanf(gguf, "blk.%d.%127s", &il, suffix) != 2) return 0;
        for (int kind = 0; kind < NTMAP; kind++) {
            if (!TMAP[kind].per_layer) continue;
            char gn[256];
            snprintf(gn, sizeof gn, TMAP[kind].gguf, il);
            if (!strcmp(gn, gguf)) {
                char cc[200];
                snprintf(cc, sizeof cc, TMAP[kind].ckpt, il);
                snprintf(out, 320, "language_model.%s", cc);
                return 1;
            }
        }
        return 0;
    }
    for (int kind = 0; kind < NTMAP; kind++) {
        if (TMAP[kind].per_layer) continue;
        if (!strcmp(TMAP[kind].gguf, gguf)) { snprintf(out, 320, "language_model.%s", TMAP[kind].ckpt); return 1; }
    }
    return 0;
}

/* ---------------------------------------------------- the artifact (phase 1b) */

/* What a tensor is and where its bytes came from: the manifest's and gate 1b's
 * per-tensor record (role, the container interval it was copied from, and the
 * checkpoint tensor it came from, for the independent byte comparison). */
enum { R_VQ = 0, R_Q8_0, R_VERBATIM, R_NROLE };
static const char *const ROLE_NAME[R_NROLE] = { "vq", "q8_0", "verbatim" };

typedef struct {
    int role;
    uint64_t cont_abs;
    const char *st_file;
    uint64_t st_abs;
    const char *st_dtype;
} art_rec;

static int artifact_run(const char *container, const char *hfdir, const char *out, const char *manifest) {
    hp_t H; load_hparams(hfdir, &H);
    tok_t T; memset(&T, 0, sizeof T); tok_load(hfdir, &T);
    cont_t C; cont_open(container, &C);

    /* The checkpoint is opened for SHAPE and for the independent byte
     * comparison, not for payload bytes: every byte the file ships comes from the
     * container. Its tensor count must equal the container's slot count, so a
     * partial container cannot be wrapped into something that looks whole. */
    v41_st S;
    if (v41_st_open(&S, hfdir)) return 1;
    if ((uint64_t)S.ne != C.n) {
        fprintf(stderr, "eg2_gguf: the checkpoint holds %d tensors and the container %u slots; "
                        "the artifact is a re-wrap of the whole plan, never of a subset (hard stop)\n", S.ne, C.n);
        return 1;
    }

    ds4gw W;
    if (ds4gw_begin(&W, out)) return 1;

    /* ---- metadata: docs/ds4-gemma-payload-contract.md section 4. The family
     * writes its own arch, its gate key and its recipe, and writes NO
     * general.file_type, mirroring the one shipping VQ-to-GGUF tool
     * (v41_to_gguf.c:357-363). The hparams and the tokenizer are this model's and
     * come from the same two functions the baseline uses, so the two files can
     * never disagree on a geometry value or a vocabulary entry. ---- */
    ds4gw_kv_str(&W, "general.architecture", ARCH_VQ);
    ds4gw_kv_str(&W, "general.name", "EmbeddingGemma-2 VQ");
    ds4gw_kv_u32(&W, "general.alignment", 32);
    ds4gw_kv_u32(&W, "gemma_vq.format_version", 1);
    ds4gw_kv_str(&W, "gemma_vq.recipe", C.recipe);
    emit_hparams(&W, ARCH_VQ, &H);
    emit_tokenizer(&W, &T);

    /* ---- plan: one GGUF tensor per container slot, in the container's order ---- */
    uint64_t total_bytes = 0, role_bytes[R_NROLE] = {0};
    int role_n[R_NROLE] = {0};
    int nt = 0, cap = 0;
    art_rec *rec = NULL;
    uint8_t ph[16];
    for (uint32_t i = 0; i < C.n; i++) {
        const cont_slot *s = &C.slot[i];
        const v41_st_ent *e = v41_st_find(&S, s->name);
        if (!e) {
            fprintf(stderr, "eg2_gguf: the checkpoint holds no %s (is this container from another checkpoint?)\n", s->name);
            return 1;
        }
        if (s->src_dtype != (uint8_t)dtype_code(e->dtype)) {
            fprintf(stderr, "eg2_gguf: %s: the container records src_dtype %u, the checkpoint says %s\n",
                    s->name, s->src_dtype, e->dtype);
            return 1;
        }
        const uint32_t cnd = (uint32_t)e->nd;
        uint64_t elems = 1;
        for (uint32_t d = 0; d < cnd; d++) elems *= (uint64_t)e->shape[d];
        /* the container's own rank convention (tools/eg2_pack.c:575-577): rank >= 2
         * keeps shape[0] x shape[1], rank 1 is rows = count, cols = 1, rank 0 is 1x1 */
        const uint64_t want_rows = cnd >= 2 ? (uint64_t)e->shape[0] : (cnd == 1 ? (uint64_t)e->shape[0] : 1);
        const uint64_t want_cols = cnd >= 2 ? (uint64_t)e->shape[1] : 1;
        if (s->rows != want_rows || s->cols != want_cols) {
            fprintf(stderr, "eg2_gguf: %s: the container says %ux%u, the checkpoint's shape says %llux%llu\n",
                    s->name, s->rows, s->cols, (unsigned long long)want_rows, (unsigned long long)want_cols);
            return 1;
        }

        uint32_t type, ship_nd = cnd;
        uint64_t ship_ne[4] = {1, 1, 1, 1};
        int role;
        if (s->kind == CT_VQ) {
            if (cnd != 2) { fprintf(stderr, "eg2_gguf: %s: a VQ payload comes from a rank-%u tensor\n", s->name, cnd); return 1; }
            fseeko(C.f, (off_t)(C.payload_off + s->payload_off), SEEK_SET);
            if (fread(ph, 1, sizeof ph, C.f) != sizeof ph) die("short read of a payload header");
            const uint16_t dim = ct_u16(ph + 4), nc = ct_u16(ph + 6);
            const uint32_t rows = ct_u32(ph + 8), cols = ct_u32(ph + 12);
            if (ct_u32(ph) != DQVQ_MAT_MAGIC) {
                fprintf(stderr, "eg2_gguf: %s: the payload magic is not DQVQ\n", s->name);
                return 1;
            }
            if (dim != s->dim || nc != s->nc || rows != s->rows || cols != s->cols) {
                fprintf(stderr, "eg2_gguf: %s: the payload's own header (dim %u nc %u %ux%u) disagrees "
                                "with the slot's (dim %u nc %u %ux%u)\n",
                        s->name, dim, nc, rows, cols, s->dim, s->nc, s->rows, s->cols);
                return 1;
            }
            /* the reference's own byte account, the single copy its writer and its
             * size allocator share (vq_qc_bytes.h:20) */
            const size_t want = vq_payload_bytes((int)s->rows, (int)s->cols, (int)dim, (int)nc);
            if (want != s->payload_len) {
                fprintf(stderr, "eg2_gguf: %s: payload_len %u, the DQVQ byte account says %zu\n",
                        s->name, s->payload_len, want);
                return 1;
            }
            /* A vqblob is opaque bytes and type 42 is one element per byte, so the
             * declaration is nd 1, ne = {bytes} (contract section 4); the precedent
             * registers ne[1] = {total} the same way (v41_to_gguf.c:137-142). The
             * geometry is in the payload itself and in the family layout, never in
             * the container's geometry — a [cols, rows] declaration would make a
             * generic reader compute cols*rows bytes, which is not the payload. */
            type = GGT_VQBLOB; ship_nd = 1; ship_ne[0] = s->payload_len;
            role = R_VQ;
        } else if (s->kind == CT_Q8_0) {
            if (cnd != 2) { fprintf(stderr, "eg2_gguf: %s: a q8_0 token table comes from a rank-%u tensor\n", s->name, cnd); return 1; }
            if (s->cols % Q8_0_BLK) { fprintf(stderr, "eg2_gguf: %s: %u columns are not a whole number of q8_0 blocks\n", s->name, s->cols); return 1; }
            const uint64_t want = (uint64_t)s->rows * (s->cols / Q8_0_BLK) * Q8_0_TSZ;
            if (want != s->payload_len) {
                fprintf(stderr, "eg2_gguf: %s: payload_len %u, the q8_0 account says %llu\n",
                        s->name, s->payload_len, (unsigned long long)want);
                return 1;
            }
            type = GGT_Q8_0;
            for (uint32_t d = 0; d < cnd; d++) ship_ne[cnd - 1 - d] = (uint64_t)e->shape[d];
            role = R_Q8_0;
        } else if (s->kind == CT_VERBATIM) {
            /* Verbatim slots carry the checkpoint's own bytes, and every one of the
             * 914 is BF16 (contract section 4, re-measured on this container): two
             * bytes per element, so the shape is the byte count. */
            if (s->src_dtype != CT_BF16) {
                fprintf(stderr, "eg2_gguf: %s: verbatim slots ship BF16, this one is dtype %u\n", s->name, s->src_dtype);
                return 1;
            }
            if (elems * 2 != s->payload_len) {
                fprintf(stderr, "eg2_gguf: %s: payload_len %u, the checkpoint's shape needs %llu\n",
                        s->name, s->payload_len, (unsigned long long)(elems * 2));
                return 1;
            }
            type = GGT_BF16;
            for (uint32_t d = 0; d < cnd; d++) ship_ne[cnd - 1 - d] = (uint64_t)e->shape[d];
            /* The checkpoint stores 480 of these 914 as rank-0 scalars (shape [],
             * e.g. audio_tower.layers.0.feed_forward1.ffw_layer_1.input_max), and
             * BOTH readers of this file reject nd 0: the engine dies "tensor has an
             * unsupported number of dimensions" (ds4.c:3353, mirrored at
             * crates/ds4-core/src/tensors.rs:266) and ggml asserts n_dims >= 1
             * (ggml.c:1786). So a scalar ships as the one-element vector it is:
             * rank 1, ne = {1} — the same shape the baseline gives its 24
             * layer_output_scale tensors. */
            if (cnd == 0) { ship_nd = 1; ship_ne[0] = 1; }
            role = R_VERBATIM;
        } else {
            fprintf(stderr, "eg2_gguf: %s: the container carries a q4_K slot; this artifact writes none (hard stop)\n", s->name);
            return 1;
        }

        const int ti = ds4gw_tensor(&W, s->name, type, ship_nd, ship_ne, s->payload_len);
        if (ti + 1 > cap) {
            cap = cap ? cap * 2 : 2048;
            rec = realloc(rec, sizeof(art_rec) * (size_t)cap);
            if (!rec) die("out of memory");
        }
        rec[ti].role = role;
        rec[ti].cont_abs = C.payload_off + s->payload_off;
        rec[ti].st_file = S.sh[e->shard].file;
        rec[ti].st_abs = S.sh[e->shard].data0 + e->off;
        rec[ti].st_dtype = e->dtype;
        role_n[role]++;
        role_bytes[role] += s->payload_len;
        total_bytes += s->payload_len;
        nt++;
    }
    fprintf(stderr, "[plan] %d tensors, %.3f GB of tensor data\n", nt, (double)total_bytes / 1e9);
    for (int r = 0; r < R_NROLE; r++)
        fprintf(stderr, "[plan]   %-8s %4d tensors  %10.3f MB\n", ROLE_NAME[r], role_n[r], (double)role_bytes[r] / 1e6);
    if (ds4gw_header(&W)) return 1;

    /* ---- the manifest: what the writer put where, plus the container interval
     * each tensor was copied from and the checkpoint tensor it should equal, so
     * gate 1b can compare bytes from both sides without trusting this tool. ---- */
    if (manifest) {
        FILE *mf = fopen(manifest, "w");
        if (!mf) { fprintf(stderr, "eg2_gguf: cannot write %s\n", manifest); return 1; }
        fprintf(mf, "#gguf_name\ttype\tnd\tdims\tbytes\tgguf_abs\trole\tcont_abs\tst_file\tst_abs\tst_dtype\n");
        for (int i = 0; i < nt; i++) {
            const ds4gw_tinfo *t = &W.t[i];
            fprintf(mf, "%s\t%u\t%u\t", t->name, t->type, t->nd);
            for (uint32_t d = 0; d < t->nd; d++) fprintf(mf, "%s%llu", d ? "," : "", (unsigned long long)t->ne[d]);
            fprintf(mf, "\t%llu\t%llu\t%s\t%llu\t%s\t%llu\t%s\n",
                    (unsigned long long)t->nbytes, (unsigned long long)(W.data0 + t->off),
                    ROLE_NAME[rec[i].role], (unsigned long long)rec[i].cont_abs,
                    rec[i].st_file, (unsigned long long)rec[i].st_abs, rec[i].st_dtype);
        }
        fclose(mf);
        fprintf(stderr, "[manifest] %s\n", manifest);
    }

    /* ---- write the bytes: the container's payload, unchanged, in order ---- */
    uint64_t max_pay = 0;
    for (int i = 0; i < nt; i++) if (W.t[i].nbytes > max_pay) max_pay = W.t[i].nbytes;
    uint8_t *buf = xmalloc((size_t)(max_pay ? max_pay : 1));
    for (int i = 0; i < nt; i++) {
        const ds4gw_tinfo *t = &W.t[i];
        if (fseeko(C.f, (off_t)rec[i].cont_abs, SEEK_SET)) { perror("fseeko"); return 1; }
        if (fread(buf, 1, (size_t)t->nbytes, C.f) != t->nbytes) {
            fprintf(stderr, "eg2_gguf: short read of %s from the container\n", t->name);
            return 1;
        }
        if (ds4gw_write(&W, i, buf, t->nbytes)) return 1;
        if ((i + 1) % 128 == 0) { fprintf(stderr, "[write] %d/%d\n", i + 1, nt); fflush(stderr); }
    }
    if (ds4gw_end(&W)) return 1;
    free(buf); free(rec);
    v41_st_close(&S);
    fclose(C.f);
    fprintf(stderr, "[done] %s: %d tensors, %.3f GB of tensor data\n", out, nt, (double)total_bytes / 1e9);
    return 0;
}

/* -------------------------------------------------------------------- main -- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: eg2_gguf baseline <hf-dir> <out.gguf> [--limit N] [--only SUBSTR] [--no-tokenizer] [--manifest FILE]\n"
                        "       eg2_gguf artifact <container.bin> <hf-dir> <out.gguf> [--manifest FILE]\n");
        return 2;
    }
    const char *mode = argv[1];

    if (!strcmp(mode, "artifact")) {
        if (argc < 5) {
            fprintf(stderr, "eg2_gguf: artifact <container.bin> <hf-dir> <out.gguf> [--manifest FILE]\n");
            return 2;
        }
        const char *manifest = NULL;
        for (int i = 5; i < argc; i++) {
            if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
            else { fprintf(stderr, "eg2_gguf: unknown argument %s\n", argv[i]); return 2; }
        }
        return artifact_run(argv[2], argv[3], argv[4], manifest);
    }

    const char *hfdir = argv[2], *out = argc > 3 ? argv[3] : NULL;
    if (strcmp(mode, "baseline")) { fprintf(stderr, "eg2_gguf: unknown mode %s\n", mode); return 2; }
    if (!out) { fprintf(stderr, "eg2_gguf: output path required\n"); return 2; }

    long limit = 0; const char *only = NULL, *manifest = NULL; int want_tok = 1;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
        else if (!strcmp(argv[i], "--no-tokenizer")) want_tok = 0;
        else { fprintf(stderr, "eg2_gguf: unknown argument %s\n", argv[i]); return 2; }
    }
    const int smoke = (limit > 0 || only != NULL);

    hp_t H; load_hparams(hfdir, &H);
    tok_t T; memset(&T, 0, sizeof T);
    if (want_tok) tok_load(hfdir, &T);
    else fprintf(stderr, "[tokenizer] skipped on request\n");

    v41_st S;
    if (v41_st_open(&S, hfdir)) return 1;

    ds4gw W;
    if (ds4gw_begin(&W, out)) return 1;

    /* ---- metadata: the key set of conversion/gemma.py:676-733 as the loader
     * reads it (llama-model.cpp:1382-1541). Name and file_type as the converter
     * writes them (base.py; gguf-py/gguf/constants.py:6274). ---- */
    ds4gw_kv_str(&W, "general.architecture", "gemma-embedding2");
    ds4gw_kv_str(&W, "general.name", smoke ? "EmbeddingGemma-2 (SMOKE-TEST-NOT-A-RELEASE)" : "EmbeddingGemma-2");
    ds4gw_kv_u32(&W, "general.file_type", 32);            /* MOSTLY_BF16 */
    ds4gw_kv_u32(&W, "general.alignment", 32);            /* the writer's default */

    emit_hparams(&W, ARCH, &H);
    if (want_tok) emit_tokenizer(&W, &T);

    /* ---- plan the tensor table ---- */
    uint64_t total_bytes = 0;
    int nt = 0, nt_cap = 0;
    uint8_t *widen_flags = NULL;
    for (int kind = 0; kind < NTMAP; kind++) {
        int reps = TMAP[kind].per_layer ? (int)H.n_layer : 1;
        for (int il = 0; il < reps; il++) {
            char ck[256], gn[256];
            if (TMAP[kind].per_layer) { snprintf(ck, sizeof ck, TMAP[kind].ckpt, il); snprintf(gn, sizeof gn, TMAP[kind].gguf, il); }
            else { snprintf(ck, sizeof ck, "%s", TMAP[kind].ckpt); snprintf(gn, sizeof gn, "%s", TMAP[kind].gguf); }
            char full[320];
            snprintf(full, sizeof full, "language_model.%s", ck);
            if (only && !strstr(full, only)) continue;
            if (limit && nt >= limit) break;

            const v41_st_ent *e = v41_st_find(&S, full);
            if (!e) { fprintf(stderr, "eg2_gguf: checkpoint has no %s\n", full); return 1; }
            if (strcmp(e->dtype, "BF16")) {
                fprintf(stderr, "eg2_gguf: %s is %s, the baseline writes BF16 only\n", full, e->dtype);
                return 1;
            }
            uint64_t ne[4]; uint32_t nd = expect_dims(&H, kind, il, ne);
            /* the checkpoint's own shape must be the declared shape, reversed */
            if ((uint32_t)e->nd != nd) {
                fprintf(stderr, "eg2_gguf: %s rank %d != declared %u\n", full, e->nd, nd);
                return 1;
            }
            for (uint32_t d = 0; d < nd; d++) {
                int64_t want = (int64_t)ne[nd - 1 - d];
                if (e->shape[d] != want) {
                    fprintf(stderr, "eg2_gguf: %s shape[%u] %lld != declared %lld\n",
                            full, d, (long long)e->shape[d], (long long)want);
                    return 1;
                }
            }
            uint64_t elems = 1;
            for (uint32_t d = 0; d < nd; d++) elems *= ne[d];
            uint64_t src_bytes = elems * 2;                       /* the checkpoint is BF16 */
            if (src_bytes != e->nbytes) {
                fprintf(stderr, "eg2_gguf: %s carries %llu B, the declared shape needs %llu\n",
                        full, (unsigned long long)e->nbytes, (unsigned long long)src_bytes);
                return 1;
            }
            /* The converter's own rule (conversion/base.py:1244-1246): 1-D tensors
             * and every *_norm.weight are written F32 even in a BF16 conversion,
             * "most of the codebase that takes in 1D tensors or norms only handles
             * F32". llama.cpp's CPU backend enforces it the hard way — a BF16 1-D
             * operand in a binary op aborts (ggml binary-ops: "unsupported types:
             * dst: f32, src0: f32, src1: bf16", measured this session on the
             * all-BF16 variant, which is why this rule exists here). */
            size_t glen = strlen(gn);
            int widen = (nd <= 1) || (glen > 11 && !strcmp(gn + glen - 11, "_norm.weight"));
            uint64_t bytes = widen ? elems * 4 : src_bytes;
            int ti = ds4gw_tensor(&W, gn, widen ? GGT_F32 : GGT_BF16, nd, ne, bytes);
            if (ti + 1 > nt_cap) { nt_cap = nt_cap ? nt_cap * 2 : 512; widen_flags = realloc(widen_flags, nt_cap); if (!widen_flags) die("out of memory"); }
            widen_flags[ti] = (uint8_t)widen;
            total_bytes += bytes;
            nt++;
        }
        if (limit && nt >= limit) break;
    }
    fprintf(stderr, "[plan] %d tensors, %.3f GB of tensor data\n", nt, (double)total_bytes / 1e9);
    if (ds4gw_header(&W)) return 1;

    /* ---- the manifest: what the writer put where, per tensor, so the reader's
     * dump and the safetensors source can be joined against it field by field.
     * Absolutes are the header's data start plus the registered offset. ---- */
    if (manifest) {
        FILE *mf = fopen(manifest, "w");
        if (!mf) { fprintf(stderr, "eg2_gguf: cannot write %s\n", manifest); return 1; }
        fprintf(mf, "#gguf_name\ttype\tnd\tdims\tbytes\tgguf_abs\tckpt_name\tst_file\tst_abs\tst_dtype\n");
        for (int i = 0; i < nt; i++) {
            const ds4gw_tinfo *t = &W.t[i];
            char ck[256] = {0};
            if (!ckpt_name_for(&H, t->name, ck)) { fprintf(stderr, "eg2_gguf: no checkpoint name for %s\n", t->name); return 1; }
            const v41_st_ent *e = v41_st_find(&S, ck);
            if (!e) { fprintf(stderr, "eg2_gguf: checkpoint has no %s\n", ck); return 1; }
            fprintf(mf, "%s\t%u\t%u\t", t->name, t->type, t->nd);
            for (uint32_t d = 0; d < t->nd; d++) fprintf(mf, "%s%llu", d ? "," : "", (unsigned long long)t->ne[d]);
            fprintf(mf, "\t%llu\t%llu\t%s\t%s\t%llu\t%s\n",
                    (unsigned long long)t->nbytes, (unsigned long long)(W.data0 + t->off), ck,
                    S.sh[e->shard].file, (unsigned long long)(S.sh[e->shard].data0 + e->off),
                    widen_flags[i] ? "bf16->f32" : "bf16");
        }
        fclose(mf);
        fprintf(stderr, "[manifest] %s\n", manifest);
    }

    /* ---- write the bytes, in registration order, straight from the mapping ---- */
    for (int i = 0; i < nt; i++) {
        const ds4gw_tinfo *t = &W.t[i];
        char ck[320];
        if (!ckpt_name_for(&H, t->name, ck)) { fprintf(stderr, "eg2_gguf: no checkpoint name for %s\n", t->name); return 1; }
        const v41_st_ent *e = v41_st_find(&S, ck);
        if (!e) { fprintf(stderr, "eg2_gguf: checkpoint has no %s\n", ck); return 1; }
        const uint8_t *d = v41_st_data(&S, e);
        if (!d) return 1;
        if (widen_flags[i]) {
            /* BF16 -> F32, the widening the reference applies: the BF16 bit
             * pattern becomes the F32 word's high half (ds4_bf16_to_f32,
             * /data/YoungAi/src/common/ds4_float.h:34). */
            uint64_t n = t->nbytes / 4;
            uint8_t *buf = xmalloc(t->nbytes);
            for (uint64_t k = 0; k < n; k++) {
                uint32_t v = ((uint32_t)d[2 * k] | ((uint32_t)d[2 * k + 1] << 8)) << 16;
                memcpy(buf + 4 * k, &v, 4);
            }
            int rc = ds4gw_write(&W, i, buf, t->nbytes);
            free(buf);
            if (rc) return 1;
        } else if (ds4gw_write(&W, i, d, t->nbytes)) {
            return 1;
        }
        if ((i + 1) % 64 == 0) { fprintf(stderr, "[write] %d/%d\n", i + 1, nt); fflush(stderr); }
    }
    if (ds4gw_end(&W)) return 1;
    v41_st_close(&S);
    fprintf(stderr, "[done] %s: %d tensors, %.3f GB\n", out, nt, (double)total_bytes / 1e9);
    free(widen_flags);
    return 0;
}
