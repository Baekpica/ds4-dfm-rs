/* Source-derived cache sizes and causal SWA retention, without a model. */
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include "../ds4_naive_plan.h"

int main(void) {
    const unsigned contexts[] = {1, 127, 128, 129, 2048, 2049, 262144, 524288, 1048576};
    const unsigned chunks[] = {1, 7, 128, 2048, 8192};
    const unsigned dsa_layers[] = {0, 5, 11, 17, 23, 29, 35, 41, 47};
    unsigned cases = 0;
    for (unsigned ci = 0; ci < sizeof(contexts) / sizeof(contexts[0]); ci++) {
        for (unsigned bi = 0; bi < sizeof(chunks) / sizeof(chunks[0]); bi++) {
            const unsigned ctx = contexts[ci], cap = chunks[bi];
            const ds4_naive_memory m = naive_memory(ctx, cap);
            if (cap > ctx) { assert(m.dsa == 0 && m.scratch == 0); continue; }
            uint64_t kv = 0;
            for (unsigned il = 0; il < N05_LAYERS; il++) {
                bool sparse = false;
                for (unsigned j = 0; j < N05_DSA_LAYERS; j++) { sparse |= il == dsa_layers[j]; }
                assert(naive_is_dsa(il) == sparse);
                const unsigned capacity = naive_kv_capacity(il, ctx, cap);
                kv += (uint64_t)capacity * (sparse ? 4 : 8) * (192 + 128) * 2;
                for (unsigned pos = 0; pos < ctx; pos += cap) {
                    const unsigned n = ctx - pos < cap ? ctx - pos : cap;
                    const unsigned start = sparse || pos < 127 ? 0 : pos - 127;
                    assert(pos + n - start <= capacity);
                }
            }
            assert(kv == m.dsa + m.swa);
            assert(m.index == (uint64_t)9 * ctx * (128 + 4));
            assert(m.scratch > 0);
            cases++;
        }
    }
    const ds4_naive_memory million = naive_memory(1048576, 2048);
    assert(million.dsa == UINT64_C(24159191040));
    assert(million.swa == UINT64_C(434304000));
    assert(million.index == UINT64_C(1245708288));
    assert(million.scratch == UINT64_C(1837994240));
    assert(naive_memory(0, 1).scratch == 0);
    assert(naive_memory(1048577, 1).scratch == 0);
    assert(naive_memory(16384, 8193).scratch == 0);
    printf("%u context/chunk cases; 1M cache=%" PRIu64 " scratch=%" PRIu64 "\n",
           cases, million.dsa + million.swa + million.index, million.scratch);
    return 0;
}
