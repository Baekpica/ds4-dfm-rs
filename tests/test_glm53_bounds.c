/* Host-only boundary gate linked against CUDA, without GPU initialization.
 * Lazy admission and a disabled memory-fit check must still enforce DSA's cap. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "GLM bounds FAIL %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

int main(void) {
    g_ds4_shape = DS4_SHAPE_GLM53_FLASH;
    CHECK(setenv("DS4_SESSION_LAZY_GRAPH", "1", 1) == 0);
    CHECK(setenv("DS4_SESSION_GRAPH_FIT", "0", 1) == 0);
    CHECK(glm53_graph_bytes_estimate(1) > 0);
    CHECK(glm53_graph_bytes_estimate(2048) > glm53_graph_bytes_estimate(1));
    CHECK(glm53_graph_bytes_estimate(0) == 0);
    CHECK(glm53_graph_bytes_estimate(2049) == 0);
    CHECK(glm53_graph_bytes_estimate(UINT32_MAX) == 0);
    ds4_session_graph_fit_quote quote = {0};
    CHECK(glm53_graph_session_fit_check(DS4_BACKEND_CUDA, 2048, false, &quote));
    CHECK(quote.fits && quote.fail_open);
    for (unsigned i = 0; i < 3; i++) {
        const uint32_t invalid[] = {0, 2049, UINT32_MAX};
        CHECK(!glm53_graph_session_fit_check(DS4_BACKEND_CUDA, invalid[i], false, &quote));
        CHECK(!quote.fits && !quote.fail_open);
    }
    CHECK(!glm53_graph_session_fit_check(DS4_BACKEND_CPU, 2048, false, &quote));
    CHECK(!quote.fits && !quote.fail_open);

    ds4_engine engine = {.backend = DS4_BACKEND_CUDA, .metal_ready = true};
    ds4_session *session = NULL;
    CHECK(ds4_session_create(&session, &engine, 2048) == 0);
    CHECK(session && ds4_session_graph_pending(session));
    CHECK(ds4_session_prefill_cap(session) == 1);
    /* A nonempty checkpoint still has no GLM wire format. */
    ds4_tokens_push(&session->checkpoint, 0);
    session->checkpoint_valid = true;
    CHECK(ds4_session_payload_bytes(session) == 0);
    FILE *fp = tmpfile();
    char err[256] = {0};
    CHECK(fp);
    CHECK(ds4_session_save_payload(session, fp, err, sizeof(err)) != 0);
    CHECK(strstr(err, "snapshots are not supported"));
    CHECK(ftello(fp) == 0);
    CHECK(ds4_session_load_payload(session, fp, 0, err, sizeof(err)) != 0);
    CHECK(strstr(err, "snapshots are not supported"));
    CHECK(!session->checkpoint_valid && session->checkpoint.len == 0);
    CHECK(fclose(fp) == 0);
    ds4_session_free(session);
    session = NULL;
    const int invalid[] = {-1, 0, 2049, INT_MAX};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        CHECK(ds4_session_create(&session, &engine, invalid[i]) != 0);
        CHECK(!session);
    }
    engine.backend = DS4_BACKEND_CPU;
    CHECK(ds4_session_create(&session, &engine, 2048) != 0 && !session);
    engine.backend = DS4_BACKEND_CUDA;
    engine.mtp_ready = true;
    CHECK(ds4_session_create(&session, &engine, 2048) != 0 && !session);
    engine.mtp_ready = false;
    engine.dspark_ready = true;
    CHECK(ds4_session_create(&session, &engine, 2048) != 0 && !session);
    engine.dspark_ready = false;
    engine.metal_ready = false;
    CHECK(ds4_session_create(&session, &engine, 2048) != 0 && !session);
    puts("GLM bounds: ctx=2048 accepted; invalid contexts/backends/sidecars and snapshots rejected");
    return 0;
}
