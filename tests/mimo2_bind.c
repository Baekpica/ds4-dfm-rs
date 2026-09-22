/* Exercise the production pointer adapter against the independent quantizer
 * inventory. These descriptor stubs never load or validate tensor payloads. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include "../ds4.h"

enum { MIMO2_LAYERS = 48, MIMO2_DRAFT_LAYERS = 3, TENSORS = 508, DS4_N_VOCAB = 152576 };
typedef struct { char name[128]; unsigned visits; } ds4_tensor;
typedef struct { ds4_tensor tensors[TENSORS]; } ds4_model;
typedef struct {
    ds4_tensor *attn_norm, *attn_qkv, *attn_output, *ffn_norm, *attn_sinks;
    ds4_tensor *ffn_gate, *ffn_up, *ffn_down, *ffn_gate_inp, *ffn_exp_probs_b;
    ds4_tensor *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;
    ds4_tensor *nextn_eh_proj, *nextn_enorm, *nextn_hnorm, *layer_output_norm;
} ds4_layer_weights;
typedef struct {
    ds4_tensor *token_embd, *output, *output_norm;
    ds4_layer_weights layer[MIMO2_LAYERS + MIMO2_DRAFT_LAYERS];
} ds4_weights;

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    for (unsigned i = 0; i < TENSORS; i++) {
        if (!strcmp(m->tensors[i].name, name)) {
            ds4_tensor *t = (ds4_tensor *)&m->tensors[i];
            if (t->visits++) { fprintf(stderr, "duplicate: %s\n", name); exit(2); }
            return t;
        }
    }
    fprintf(stderr, "missing: %s\n", name);
    exit(3);
}

static ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, ...) {
    char name[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    return required_tensor(m, name);
}
#include "../ds4_mimo2_plan.h"
#include "../ds4_mimo2_bind.inc"

int main(int argc, char **argv) {
    if (argc != 2) { return 1; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { return 1; }
    ds4_model m = {0};
    ds4_weights w;
    for (unsigned i = 0; i < TENSORS; i++) {
        if (fscanf(f, "%127s", m.tensors[i].name) != 1) { return 1; }
    }
    char extra[128];
    if (fscanf(f, "%127s", extra) != EOF) { return 1; }
    fclose(f);
    mimo2_bind(&w, &m);
    for (unsigned i = 0; i < TENSORS; i++) {
        if (m.tensors[i].visits != 1) { fprintf(stderr, "unbound: %s\n", m.tensors[i].name); return 4; }
    }
    for (unsigned il = 48; il < 51; il++) {
        if (!w.layer[il].layer_output_norm || w.layer[il].ffn_gate_exps) { return 5; }
    }
    puts("508 independent artifact names bound exactly once; three dense MTP blocks retained");
    return 0;
}
