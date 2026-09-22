/* Admission regression: every query must retain its full causal key span
 * when an entire prefill batch is stored before attention. */
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include "../ds4.h"
enum { MIMO2_LAYERS = 48, DS4_N_VOCAB = 152576 };
#include "../ds4_mimo2_plan.h"

int main(void) {
    const unsigned contexts[] = {1, 127, 128, 129, 512, 8192, 1048576};
    const unsigned chunks[] = {1, 7, 128, 512, 4096};
    const unsigned full_layers[] = {0, 5, 11, 17, 23, 29, 35, 41, 47};
    unsigned cases = 0;
    for (unsigned ci = 0; ci < sizeof(contexts) / sizeof(contexts[0]); ci++) {
        for (unsigned bi = 0; bi < sizeof(chunks) / sizeof(chunks[0]); bi++) {
            const unsigned ctx = contexts[ci], batch = chunks[bi];
            ds4_context_memory m = mimo2_memory(ctx, batch);
            if (batch > ctx) { assert(m.total_bytes == 0); continue; }
            uint64_t expected_kv = 0;
            for (unsigned il = 0; il < 48; il++) {
                bool full = false;
                for (unsigned j = 0; j < 9; j++) { full |= il == full_layers[j]; }
                assert(mimo2_is_full(il) == full);
                const unsigned cap = mimo2_kv_capacity(il, ctx, batch);
                expected_kv += (uint64_t)cap * (full ? 4 : 8) * (192 + 128) * 2;
                for (unsigned pos = 0; pos < ctx; pos += batch) {
                    unsigned n = ctx - pos < batch ? ctx - pos : batch;
                    const unsigned first = full || pos < 127 ? 0 : pos - 127;
                    // Later queries start no earlier; this bounds the whole batch.
                    assert(pos + n - first <= cap);
                }
            }
            assert(expected_kv == m.raw_bytes);
            assert(m.total_bytes == m.raw_bytes + m.scratch_bytes);
            cases++;
        }
    }
    assert(mimo2_memory(0, 1).total_bytes == 0);
    assert(mimo2_memory(1048577, 1).total_bytes == 0);
    assert(mimo2_memory(8192, 4097).total_bytes == 0);
    printf("%u context/chunk cases: causal KV retention verified\n", cases);
    return 0;
}
