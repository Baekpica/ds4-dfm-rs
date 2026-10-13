/* Model-free teardown gate. Compile the actual bridge with section GC:
 * cc -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -I. \
 *   tests/test_ds41_bridge_lifetime.c -Wl,--gc-sections -o /tmp/ds41-feed-life
 * An aborted forward leaves the provider writing native-owned row buffers.
 * The bridge must drain that provider before releasing those buffers. */
#include <assert.h>

#include "../native/bridge/ds4_bridge.c"

static int pending;
static int closed;
static int drain_calls;
static int run_rc;
static int drain_rc;
static ds4_v41_wait_fn provider_wait;
static void *provider_ud;

int ds4_v41_engram_count(void *engine) {
    (void)engine;
    return 2;
}

ds4_v41_feed *ds4_v41_feed_open(void *engine, int n_prompt,
                               ds4_v41_rows_fn rows, void *rows_ud,
                               ds4_v41_wait_fn wait, void *wait_ud,
                               ds4_v41_next_fn next, void *next_ud) {
    (void)engine; (void)n_prompt; (void)rows; (void)rows_ud;
    (void)next; (void)next_ud;
    provider_wait = wait;
    provider_ud = wait_ud;
    return (ds4_v41_feed *)1;
}

const ds41_engram_feed *ds4_v41_feed_handle(const ds4_v41_feed *feed) {
    return (const ds41_engram_feed *)feed;
}

void ds4_v41_feed_close(ds4_v41_feed *feed) {
    assert(feed);
    assert(!pending && "native rows freed while provider writes are pending");
    closed++;
}

void ds4_engine_v41_set_progress(int (*cb)(void *, const char *, int, int), void *ud) {
    (void)cb; (void)ud;
}

int ds4_v41_generate_argmax(void *engine, const int *prompt, int n_prompt,
                           int n_predict, int no_engram, int verify_k,
                           const ds41_engram_feed *feed,
                           int (*emit)(int, void *), void *emit_ud) {
    (void)engine; (void)prompt; (void)n_prompt; (void)n_predict;
    (void)no_engram; (void)verify_k; (void)feed; (void)emit; (void)emit_ud;
    pending = 1;
    /* The failed layer wait deliberately leaves an in-flight row writer. */
    assert(provider_wait(0, provider_ud) != 0);
    return run_rc;
}

static int wait_rows(int layer, void *ud) {
    (void)layer; (void)ud;
    return 1;
}

static int drain_rows(void *ud) {
    (void)ud;
    pending = 0;
    drain_calls++;
    return drain_rc;
}

int main(void) {
    ds4_bridge_model model = { .engine = (ds4_engine *)1 };
    const int32_t prompt[] = {1};
    char err[128];
    for (run_rc = 0; run_rc <= 1; run_rc++) {
        for (drain_rc = 0; drain_rc <= 1; drain_rc++) {
            closed = drain_calls = 0;
            const int rc = ds4_bridge_v41_generate(&model, prompt, 1, 1,
                0, 0, NULL, NULL, wait_rows, NULL, drain_rows, NULL, NULL, NULL, NULL,
                err, sizeof err);
            assert(closed == 1);
            assert(drain_calls == 1);
            assert((rc != 0) == (run_rc != 0 || drain_rc != 0));
        }
    }
    closed = drain_calls = 0;
    const int rc = ds4_bridge_v41_generate(&model, prompt, 1, 1,
        0, 0, NULL, NULL, wait_rows, NULL, NULL, NULL, NULL, NULL, NULL,
        err, sizeof err);
    assert(rc != 0 && "an asynchronous provider requires a drain hook");
    assert(closed == 0);
    puts("V4.1 bridge teardown gate passed");
    return 0;
}
