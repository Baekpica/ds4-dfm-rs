/* Model-free policy check: fake tensor handles are never dereferenced. */
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "K2 rewind FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)

int main(void) {
    ds4_engine engine = {.backend = DS4_BACKEND_CUDA};
    ds4_session session = {.engine = &engine, .exaone_graph_ready = true,
                          .checkpoint_valid = true};
    session.checkpoint.len = 28;
    g_ds4_shape = DS4_SHAPE_K2_HORIZON_375B;
    for (unsigned ctx = 1024; ctx <= 32768; ctx *= 32) {
        session.exaone_graph.ctx_size = ctx;
        CHECK(ds4_session_exaone_rewind_span(&session) == 0);
    }

    /* The family-specific restriction must leave K-EXAONE LLLG intact. */
    g_ds4_shape = DS4_SHAPE_KEXAONE_236B;
    session.exaone_graph.ctx_size = 4096;
    session.checkpoint.len = 1000;
    session.exaone_graph.layer_kv[0] = (ds4_gpu_tensor *)(uintptr_t)1;
    session.exaone_graph.layer_kv_cap[0] = 640;
    CHECK(ds4_session_exaone_rewind_span(&session) == 513);
    puts("K2 rewind: exact-only span; K-EXAONE LLLG unchanged PASS");
    return 0;
}
