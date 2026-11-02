/* Model-free policy check: fake tensor handles are never dereferenced. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "K2 rewind FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

int main(void) {
    enum { N = 1000 };
    int live[N], edited[N];
    for (int i = 0; i < N; i++) { live[i] = edited[i] = i; }
    ds4_engine engine = {.backend = DS4_BACKEND_CUDA};
    ds4_session session = {.engine = &engine, .exaone_graph_ready = true,
                          .checkpoint_valid = true,
                          .checkpoint = {.v = live, .len = 28, .cap = N}};
    ds4_tokens prompt = {.v = edited, .len = 12, .cap = N};

    /* K2 has no sliding ring: an edit keeps its common prefix. */
    g_ds4_shape = DS4_SHAPE_K2_HORIZON_375B;
    session.exaone_graph.ctx_size = 1024;
    session.exaone_graph.high = 28;
    edited[3] = -1;
    CHECK(exaone_sync_keep(&session, &prompt) == 3);
    edited[3] = 3;

    /* K-EXAONE LLLG: a 640-row ring holds the 127 rows under k while the
     * newest row written is at most 513 rows above k. */
    g_ds4_shape = DS4_SHAPE_KEXAONE_236B;
    session.exaone_graph.ctx_size = 4096;
    session.exaone_graph.layer_kv[0] = (ds4_gpu_tensor *)(uintptr_t)1;
    session.exaone_graph.layer_kv_cap[0] = 640;
    session.checkpoint.len = N;
    session.exaone_graph.high = N;
    prompt.len = 950;
    edited[487] = -1;
    CHECK(exaone_sync_keep(&session, &prompt) == 487);
    edited[487] = 487;
    edited[486] = -1;
    CHECK(exaone_sync_keep(&session, &prompt) == 0);
    edited[486] = 486;

    /* A longer abandoned branch wrapped the ring: the newest row written,
     * not the live length, bounds the rewind. */
    edited[900] = -1;
    CHECK(exaone_sync_keep(&session, &prompt) == 900);
    session.exaone_graph.high = 1500;
    CHECK(exaone_sync_keep(&session, &prompt) == 0);

    /* A restore holds no row under its floor. */
    session.exaone_graph.high = session.exaone_graph.restored = N;
    CHECK(exaone_sync_keep(&session, &prompt) == 0);
    puts("K2 rewind: K2 keeps any prefix; K-EXAONE ring bounded by high-water and restore PASS");
    return 0;
}
