/* V4.1 tool-syntax greedy pick gate (tests/test_ds41_pick, model-free): the
 * engine's units_test_sample_pick_greedy_hook (t_units.c:230) -- the c6e2706
 * contract.  Under sampling, a position whose next token starts tool-call
 * syntax takes the raw argmax (the sample kernel's slot [3]) instead of the
 * sample; the ask walks only the accepted draft prefix and stops once the
 * draft is not accepted; a greedy position accepts the draft only when it
 * equals the argmax; without a hook the original accept/residual pick stands.
 *
 * The engine's test builds a ds4_v41_state directly (core_internal.h); the
 * port's state is private to the ds4.c TU, so the same vectors run through
 * ds4_test_v41_sample_pick (the DS4_TEST_HOOKS entry), which sets the face
 * the way the server does.
 *
 * Build: make test-ds41-pick */

#include "ds41_forward.h"

#include <stdint.h>
#include <stdio.h>

extern "C" int ds4_test_v41_sample_pick(const int32_t *slot, const int32_t *batch,
                                        uint32_t n, ds4_greedy_fn fn, void *ud,
                                        int32_t *want);

static uint32_t greedy_calls, greedy_last_n;
static const int32_t *greedy_ahead;
static int greedy_at1(void *ud, const int32_t *ahead, uint32_t n) {
    (void)ud;
    greedy_calls++; greedy_last_n = n; greedy_ahead = ahead;
    return n == 1;
}

static int check(const char *what, int ok) {
    printf("  %-52s %s\n", what, ok ? "ok" : "RED");
    return !ok;
}

int main(void) {
    const int32_t batch[4] = {100, 5, 6, 7};
    int32_t want[4];
    int fail = 0;

    /* The greedy argmax at position 1 equals the draft (6): keep asking; the
     * next ask (position 2) answers no, the sample rejects (slot[1]=0) and the
     * residual (slot[2]=4) is the pick; no ask after a non-accepted draft. */
    const int32_t a[16] = {9, 1, 8, 99,   20, 0, 21, 6,   3, 0, 4, 55,   1, 1, 2, 3};
    greedy_calls = 0;
    fail |= check("case a: call ok", ds4_test_v41_sample_pick(a, batch, 4, greedy_at1, NULL, want) == 0);
    fail |= check("case a: want == {5,6,4}", want[0] == 5 && want[1] == 6 && want[2] == 4);
    fail |= check("case a: 3 asks, last n 2, ahead == batch+1",
                  greedy_calls == 3 && greedy_last_n == 2 && greedy_ahead == batch + 1);

    /* The greedy argmax at position 1 (42) differs from the draft (6): take
     * the argmax even though the kernel accepted (slot[1]=1); the draft ends. */
    const int32_t b[16] = {9, 1, 8, 99,   20, 1, 21, 42,   3, 1, 4, 55,   1, 1, 2, 3};
    greedy_calls = 0;
    fail |= check("case b: call ok", ds4_test_v41_sample_pick(b, batch, 4, greedy_at1, NULL, want) == 0);
    fail |= check("case b: want[0..1] == {5,42}, 2 asks", want[0] == 5 && want[1] == 42 && greedy_calls == 2);

    /* No hook: the original accept/residual pick. */
    fail |= check("case c: call ok", ds4_test_v41_sample_pick(b, batch, 4, NULL, NULL, want) == 0);
    fail |= check("case c: want == {5,6,7,1}", want[0] == 5 && want[1] == 6 && want[2] == 7 && want[3] == 1);

    printf(fail ? "ds41 sample pick gate: RED\n" : "ds41 sample pick gate: all green\n");
    return fail ? 1 : 0;
}
