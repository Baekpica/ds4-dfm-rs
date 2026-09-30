/* Feed the independently parsed draft directory to the native binder. */
#include "../ds4.c"
#include <assert.h>

int main(void) {
    ds4_model m = {0};
    m.tensors = xcalloc(63, sizeof(ds4_tensor));
    char name[128];
    unsigned type, ndim, d0, d1;
    while (scanf("%127s %u %u %u %u", name, &type, &ndim, &d0, &d1) == 5) {
        assert(m.n_tensors < 63);
        ds4_tensor *t = &m.tensors[m.n_tensors++];
        t->name = (ds4_str){strdup(name), strlen(name)};
        t->type = type; t->ndim = ndim; t->dim[0] = d0; t->dim[1] = d1;
    }
    assert(m.n_tensors == 63);
    ds4_naive_draft d;
    assert(naive_draft_bind(&d, &m));
    assert(d.model == &m && d.fc && d.mask && d.markov1 && d.markov2 && d.conf && d.bias);
    for (unsigned il = 0; il < 5; il++) {
        assert(d.q[il] && d.k[il] && d.v[il] && d.o[il]);
        assert(d.q_norm[il] && d.k_norm[il] && d.gate[il] && d.up[il] && d.down[il]);
    }
    for (unsigned i = 0; i < 63; i++) {
        const unsigned saved = m.tensors[i].dim[0];
        m.tensors[i].dim[0] = 32;
        assert(!naive_draft_bind(&d, &m));
        m.tensors[i].dim[0] = saved;
        m.tensors[i].type ^= 8;
        assert(!naive_draft_bind(&d, &m));
        m.tensors[i].type ^= 8;
    }
    for (unsigned i = 0; i < 63; i++) { free((void *)m.tensors[i].name.ptr); }
    free(m.tensors);
    puts("63 native draft bindings; source dimensions and formats");
    return 0;
}
