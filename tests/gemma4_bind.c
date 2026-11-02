/* Exercise the production Gemma-4 pointer adapter against the artifact's own
 * tensor directory (`vq-gguf list`: one `name type dim0 [dim1 ...]` line per
 * tensor, GGUF dim order). Descriptor stubs carry the stored type and dims so
 * the bind's own layout checks run; no payload is read and nothing is loaded.
 *
 * The stub ds4_die exits 4, so a type/dim mismatch is a test failure rather
 * than a process abort. Every tensor must be resolved exactly once.
 */
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GEMMA4_LAYERS = 30,
    GEMMA4_EMBED = 2816,
    GEMMA4_VOCAB = 262144,
    GEMMA4_HEADS = 16,
    GEMMA4_HEAD_DIM = 256,
    GEMMA4_GLOBAL_HEAD_DIM = 512,
    GEMMA4_KV_HEADS = 8,
    GEMMA4_GLOBAL_KV_HEADS = 2,
    GEMMA4_DENSE = 2112,
    GEMMA4_EXPERTS = 128,
    GEMMA4_FF = 704,
    GEMMA4_VECTOR = 8,
    GEMMA4_WORDS = 8192,
    GEMMA4_TENSORS = 12177,
};

typedef struct {
    char name[160];
    uint32_t type;
    uint32_t ndim;
    uint64_t dim[4];
    unsigned visits;
} ds4_tensor;

typedef struct {
    ds4_tensor tensors[GEMMA4_TENSORS];
} ds4_model;

static void ds4_die(const char *message) {
    fprintf(stderr, "gemma4 bind: %s\n", message);
    exit(4);
}

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    for (unsigned i = 0; i < GEMMA4_TENSORS; i++) {
        if (!strcmp(m->tensors[i].name, name)) {
            ds4_tensor *t = (ds4_tensor *)&m->tensors[i];
            if (t->visits++) {
                fprintf(stderr, "duplicate: %s\n", name);
                exit(2);
            }
            return t;
        }
    }
    fprintf(stderr, "missing: %s\n", name);
    exit(3);
}

static void tensor_expect_layout(const ds4_tensor *t, uint32_t type,
                                 uint32_t ndim, uint64_t d0, uint64_t d1,
                                 uint64_t d2) {
    const uint64_t want[4] = {d0, d1, d2, 0};
    if (!t) { ds4_die("missing tensor"); }
    if (t->type != type) {
        fprintf(stderr, "type of %s is %u, expected %u\n", t->name, t->type, type);
        exit(4);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "rank of %s is %u, expected %u\n", t->name, t->ndim, ndim);
        exit(4);
    }
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != want[i]) {
            fprintf(stderr, "dim[%u] of %s is %llu, expected %llu\n", i, t->name,
                    (unsigned long long)t->dim[i], (unsigned long long)want[i]);
            exit(4);
        }
    }
}

/* The native type ids the engine's enum carries (ds4.c):
 * F32 0, Q8_0 8, Q4_K 12, BF16 30, and the GVQ1 storage types GVBC/GVQ8. */
enum {
    DS4_TENSOR_BF16 = 30,
    DS4_TENSOR_Q4_K = 12,
    DS4_TENSOR_Q8_0 = 8,
    DS4_TENSOR_GLM_BOOK = 0x47564342,
    DS4_TENSOR_GLM_VQ = 0x47565138,
};

#include "../ds4_gemma4_plan.h"

typedef struct {
    ds4_tensor *norm[7];
    ds4_tensor *layer_scalar;
    ds4_tensor *q, *k, *v, *o;
    ds4_tensor *attn_norm[2];
    ds4_tensor *dense_gate, *dense_up, *dense_down;
    ds4_tensor *router, *router_scale, *router_per_expert;
    ds4_tensor *book_down, *book_gate_up;
    ds4_tensor *gate[G4_EXPERTS], *up[G4_EXPERTS], *down[G4_EXPERTS];
} ds4_gemma4_layer;

typedef struct {
    ds4_tensor *token_embd, *output, *output_norm;
    ds4_gemma4_layer *gemma4;
} ds4_weights;

#include "../ds4_gemma4_bind.inc"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <inventory>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    static ds4_model m;
    unsigned n = 0;
    char name[160];
    while (n < GEMMA4_TENSORS && fscanf(f, "%159s", name) == 1) {
        ds4_tensor *t = &m.tensors[n++];
        snprintf(t->name, sizeof(t->name), "%s", name);
        if (fscanf(f, "%u", &t->type) != 1) { return 1; }
        unsigned long long dims[4] = {0};
        unsigned ndim = 0;
        while (ndim < 4) {
            const int c = fgetc(f);
            if (c == ' ' || c == '\t') { continue; }
            if (c == '\n' || c == EOF) { break; }
            ungetc(c, f);
            if (fscanf(f, "%llu", &dims[ndim]) != 1) { return 1; }
            ndim++;
        }
        t->ndim = ndim;
        for (unsigned i = 0; i < ndim; i++) { t->dim[i] = dims[i]; }
    }
    fclose(f);
    if (n != GEMMA4_TENSORS) {
        fprintf(stderr, "inventory has %u tensors, expected %u\n", n, GEMMA4_TENSORS);
        return 1;
    }

    ds4_weights w;
    memset(&w, 0, sizeof(w));
    gemma4_bind(&w, &m);
    if (!w.token_embd || !w.output || !w.output_norm || !w.gemma4) { return 5; }
    if (w.output != w.token_embd) {
        fprintf(stderr, "the head must borrow the tied token table\n");
        return 5;
    }
    for (unsigned il = 0; il < GEMMA4_LAYERS; il++) {
        const ds4_gemma4_layer *l = &w.gemma4[il];
        for (unsigned i = 0; i < 7; i++) { if (!l->norm[i]) { return 5; } }
        if (!l->q || !l->k || !l->o || !l->dense_gate || !l->dense_up ||
            !l->dense_down || !l->router || !l->router_scale ||
            !l->router_per_expert || !l->book_down || !l->book_gate_up ||
            !l->layer_scalar || !l->attn_norm[0] || !l->attn_norm[1]) {
            return 5;
        }
        /* attention_k_eq_v: a full layer binds v to its own k. */
        if (gemma4_is_full(il) != (l->v == l->k)) {
            fprintf(stderr, "layer %u value projection is wrong\n", il);
            return 5;
        }
        for (unsigned e = 0; e < GEMMA4_EXPERTS; e++) {
            if (!l->gate[e] || !l->up[e] || !l->down[e]) { return 5; }
        }
    }
    unsigned seen = 0;
    for (unsigned i = 0; i < GEMMA4_TENSORS; i++) { seen += m.tensors[i].visits != 0; }
    if (seen != GEMMA4_TENSORS) {
        fprintf(stderr, "%u of %u tensors resolved\n", seen, GEMMA4_TENSORS);
        return 1;
    }
    printf("GEMMA4_BIND_PASS tensors=%u layers=%u experts=%u\n", seen,
           GEMMA4_LAYERS, GEMMA4_EXPERTS);
    free(w.gemma4);
    return 0;
}
