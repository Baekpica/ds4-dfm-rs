/* Gemma-4 sliding-ring resume rule (gemma4_ring_keeps), model-free.
 *
 * A sliding layer keeps G4_WINDOW plus one block of rows and writes slot
 * position % rows, so every slot holds the newest position written to it. A
 * serial rewind or a bank fork at k is exact only while the ring still holds
 * the window under k. */
#include <stdio.h>

#include "../ds4_gemma4_plan.h"

static int failures;

static void expect(const char *name, bool got, bool want) {
    if (got == want) { return; }
    fprintf(stderr, "FAIL %s: got %d want %d\n", name, got, want);
    failures++;
}

int main(void) {
    /* The 512-row default chunk: 1,536 ring rows, a 512-row margin. */
    const uint32_t rows = G4_WINDOW + 512u;

    expect("extension", gemma4_ring_keeps(rows, 4000u, 0u, 4000u), true);
    expect("ring never wrapped", gemma4_ring_keeps(rows, rows, 0u, 10u), true);
    expect("within the margin", gemma4_ring_keeps(rows, 4000u, 0u, 3488u), true);
    expect("past the margin", gemma4_ring_keeps(rows, 4000u, 0u, 3487u), false);
    /* A thinking turn dropped from history: 2,000 rows behind the frontier. */
    expect("deep rewind", gemma4_ring_keeps(rows, 6000u, 0u, 4000u), false);
    expect("below a restore", gemma4_ring_keeps(rows, 3000u, 3000u, 2999u), false);
    expect("at a restore", gemma4_ring_keeps(rows, 3000u, 3000u, 3000u), true);
    /* A context below the ring clamps the ring to ctx: nothing wraps. */
    expect("short context", gemma4_ring_keeps(1024u, 1024u, 0u, 1u), true);

    if (failures) { return 1; }
    printf("gemma4 ring: ok\n");
    return 0;
}
