/* The Python driver supplies the actual engine-close and vision-bias bodies.
 * Reuse the bounded tensor seam; no model or GPU is loaded. */
#define main kv_import_main
#include "test_ds41_kv_import.c"
#undef main

typedef struct { uint64_t bytes; const void *data; } ds4_tensor;
typedef struct { void *map; } ds4_model;
typedef struct {
    struct { const ds4_tensor *ffn_exp_probs_b_vl; } layer[DS4_MAX_LAYER];
} ds4_weights;
typedef struct {
    void *qwen_ple_cuda, *qwen_ple_store;
    int glm53_stream, vocab, mimo2_media, mtp_ready, dspark_ready;
    ds4_weights weights;
    ds4_model model, mtp_model, dspark_model, vision_model, audio_model;
    void *directional_steering_dirs, *directional_steering_file;
} ds4_engine;

static ds4_gpu_tensor *g_v41_bias_vl[DS4_MAX_LAYER];
static uint32_t tensors_at_cleanup;

static const ds4_model *v41_vision_model(const ds4_engine *e) { return &e->vision_model; }
static const void *tensor_data(const ds4_model *m, const ds4_tensor *t) {
    (void)m;
    return t->data;
}
static void ds4_gpu_cleanup(void) { tensors_at_cleanup += g_live_tensors; }

/* Unrelated model resources are empty in this fixture. */
#define qwen4exp_report_ple_stats(...) ((void)0)
#define ds4_qwen38_ple_cuda_destroy(...) ((void)0)
#define ds4_ple_store_close(...) ((void)0)
#define glm53_stream_cancel(...) ((void)0)
#define weights_free(...) ((void)0)
#define vocab_free(...) ((void)0)
#define ds4_threads_shutdown(...) ((void)0)
#define model_close(...) ((void)0)
#define mimo2_media_free(...) ((void)0)
#define glm53_stream_report(...) ((void)0)
#define glm53_stream_free(...) ((void)0)
#define ds4_release_instance_lock(...) ((void)0)
#include DS41_CLOSE_FIXTURE

static ds4_engine *fixture_model(const ds4_tensor *bias) {
    ds4_engine *e = calloc(1, sizeof *e);
    if (!e) { abort(); }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        e->weights.layer[il].ffn_exp_probs_b_vl = bias;
    }
    return e;
}

static void save_checkpoint(const int32_t *tokens, uint32_t n) {
    ds4_v41_state st = {0};
    const uint64_t rowb = DS4_N_HEAD_DIM * sizeof(float);
    bool ok = true;
    st.ctx = g_ds4_v41.ctx;
    st.himg = calloc(st.ctx, 1);
    if (!st.himg) { abort(); }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        st.win[il] = v41_alloc(DS4_N_SWA * rowb, &ok);
    }
    st.comp_kv[0] = v41_alloc(2u * DS4_V41_CKV_BYTES, &ok);
    st.index_k[0] = v41_alloc(2u * DS4_V41_IDXK_BYTES, &ok);
    st.cpre_kv[0] = v41_alloc(rowb, &ok);
    st.cpre_sc[0] = v41_alloc(rowb, &ok);
    st.mainh = v41_alloc(DS4_N_SWA * DS4_N_EMBD * sizeof(float), &ok);
    st.logits = v41_alloc(DS4_N_VOCAB * sizeof(float), &ok);
    st.ng_src[0] = 2u;
    st.cpend[0] = 1u;
    st.mainh_end = n - 1u;
    st.mainh_n = DS4_N_SWA;
    v41_kvreuse_note_mount(1, UINT64_C(0x0123456789abcdef));
    v41_kvreuse_save(&st, tokens, n, NULL);
    v41_kvreuse_save_turn(&st, tokens, n, tokens, 2u, NULL);   /* the port addition: close must release this slot too */
    if (v41_kvreuse_restore(&st, tokens, n, NULL) != n) { abort(); }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(st.win[il]);
    }
    ds4_gpu_tensor_free(st.comp_kv[0]);
    ds4_gpu_tensor_free(st.index_k[0]);
    ds4_gpu_tensor_free(st.cpre_kv[0]);
    ds4_gpu_tensor_free(st.cpre_sc[0]);
    ds4_gpu_tensor_free(st.mainh);
    ds4_gpu_tensor_free(st.logits);
    free(st.himg);
}

int main(void) {
    const int32_t tokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    /* K's key for `tokens`, T's key for the same run (the prompt then the two
     * ids the state committed, save_checkpoint), and the next turn's prompt:
     * the same block plus one new token, as a tool result extends it. */
    const int32_t committed[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2};
    const int32_t next_turn[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 4};
    const float first_bias[] = {1.f, 2.f}, next_bias[] = {3.f, 4.f};
    const ds4_tensor first = {sizeof first_bias, first_bias};
    const ds4_tensor next = {sizeof next_bias, next_bias};
    ds4_engine *e = fixture_model(&first);
    int failures = check_result("initial vision bias upload", v41_vis_bias_vl_upload(e));
    save_checkpoint(tokens, sizeof tokens / sizeof tokens[0]);
    (void)v41_kv_span_buf_get(64u);
    failures += check_result("turn checkpoint serves the longer prompt",
        v41_kvreuse_peek(next_turn, sizeof next_turn / sizeof next_turn[0], NULL) ==
        sizeof committed / sizeof committed[0]);
    failures += check_result("equal length stays the prompt-end checkpoint",
        v41_kvreuse_peek(committed, sizeof committed / sizeof committed[0], NULL) ==
        sizeof tokens / sizeof tokens[0]);
    ds4_engine_close(e);

    failures += check_result("close releases tensors before GPU teardown", tensors_at_cleanup == 0u);
    failures += check_result("close invalidates old-model checkpoint",
        v41_kvreuse_peek(tokens, sizeof tokens / sizeof tokens[0], NULL) == 0u &&
        v41_kvreuse_peek(next_turn, sizeof next_turn / sizeof next_turn[0], NULL) == 0u &&
        ds4_engine_v41_last_reused() == 0);
    failures += check_result("close clears host checkpoint storage",
        !K.tok && !K.sig && !K.tok_cap && !T.tok && !T.sig && !T.tok_cap &&
        !g_v41_kv_mount_has && !g_v41_kv_mount_hash &&
        !g_v41_kv_span_buf && !g_v41_kv_span_cap);

    e = fixture_model(&next);
    int uploaded = v41_vis_bias_vl_upload(e);
    for (uint32_t il = 0; uploaded && il < DS4_N_LAYER; il++) {
        uploaded = g_v41_bias_vl[il] &&
            !memcmp(g_v41_bias_vl[il]->data, next_bias, sizeof next_bias);
    }
    failures += check_result("reopen uploads the new model bias", uploaded);
    ds4_engine_close(e);
    ds4_engine_close(fixture_model(&next));
    ds4_engine_close(NULL);
    failures += check_result("repeated and empty closes release all tensors", !g_live_tensors);
    printf("ds41 model lifetime gate: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
