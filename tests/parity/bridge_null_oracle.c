/* NULL-handle FFI checks for the Phase 7 generation ABI. Links
 * ds4_bridge.o against stubs so it does not load CUDA or a GGUF. */

#include "ds4.h"
#include "ds4_distributed.h"
#include "native/bridge/ds4_bridge.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern unsigned bridge_payload_load_calls;
extern int64_t bridge_payload_load_offset;
extern uint64_t bridge_payload_load_bytes;
extern int bridge_routed_quant_bits;
extern unsigned bridge_boot_prewarm_calls;
extern int bridge_sync_rc;
extern unsigned bridge_sync_calls;
extern unsigned bridge_progress_sets;
extern unsigned bridge_progress_clears;
extern int bridge_progress_active;
extern int bridge_batch_max_seq;
extern int bridge_static_batch_calls;
extern int bridge_static_batch_n;
extern int bridge_static_batch_prompt_lens[8];
extern int bridge_static_batch_max_new[8];
extern int bridge_static_batch_eos[8];
extern int bridge_bank_committed;
extern int bridge_bank_tokens[8];
extern uint64_t bridge_bank_generation;
extern unsigned bridge_bank_save_calls;
extern int bridge_bank_save_result;
extern unsigned bridge_bank_load_calls;
extern int64_t bridge_bank_load_offset;
extern uint64_t bridge_bank_load_bytes;
extern int bridge_cont_run;
extern ds4_cont_seq_stats bridge_cont_stats;
extern int bridge_dist_open_enabled;
extern int bridge_dist_prepare_calls;
extern int bridge_dist_run_calls;
extern int bridge_dist_run_ctx;
extern int bridge_dist_run_result;
extern int bridge_dist_route_ready;
extern ds4_engine_options bridge_dist_engine_options;
extern ds4_dist_options bridge_dist_run_options;

struct ds4_bridge_model {
    ds4_engine *engine;
};

struct ds4_bridge_session {
    ds4_bridge_model *model;
    ds4_session *session;
};

struct ds4_bridge_batch_ctx {
    ds4_batch_ctx *ctx;
};

static void fail(const char *m) {
    fprintf(stderr, "bridge_null_oracle: %s\n", m);
    exit(1);
}

static int progress_current[8];
static int progress_total[8];
static int progress_n;

static void record_progress(void *ud, int32_t current, int32_t total) {
    int *cookie = ud;
    if (!cookie || *cookie != 42) {
        fail("progress userdata");
    }
    if (progress_n >= (int)(sizeof(progress_current) / sizeof(progress_current[0]))) {
        fail("progress overflow");
    }
    progress_current[progress_n] = current;
    progress_total[progress_n] = total;
    progress_n++;
}

static int no_admit(void *ud, ds4_bridge_cont_request *req) {
    (void)ud;
    (void)req;
    return 0;
}

static ds4_bridge_cont_stats done_stats;

static void record_done(void *ud, void *user, const int32_t *tokens, int32_t n,
                        int32_t finish, const ds4_bridge_cont_stats *stats) {
    (void)ud;
    if (user != (void *)42 || !tokens || n != 5 || finish != 1 || !stats) {
        fail("cont done callback");
    }
    done_stats = *stats;
}

int main(void) {
    char err[64];
    int32_t toks[4];
    int n = -1;
    char text[8];
    size_t tn = 99;
    uint64_t rng = 1;

    {
        const ds4_bridge_step37_pixels crop = {0};
        ds4_bridge_session fake = {.session = (ds4_session *)1};
        if (!ds4_bridge_sync_step37(NULL, toks, 1, &crop, 1, err, sizeof(err)) ||
            !ds4_bridge_sync_step37(&fake, toks, 1, NULL, 1, err, sizeof(err)) ||
            !ds4_bridge_sync_step37(&fake, toks, 1, &crop, 0, err, sizeof(err)) ||
            !ds4_bridge_sync_step37(&fake, toks, 1, &crop, 102, err, sizeof(err))) {
            fail("Step invalid crop bridge input accepted");
        }
    }

    {
        const uint8_t image[] = {1, 2, 3};
        ds4_bridge_qwen_image_info info;
        uint64_t hash = 0;
        memset(&info, 0, sizeof(info));
        memset(err, 0, sizeof(err));
        if (ds4_bridge_qwen_image_probe(image, sizeof(image), &info,
                                        err, sizeof(err)) != 0 ||
            info.source_width != 1 || info.source_height != 2 ||
            info.grid_h != 32 || info.grid_w != 16 ||
            info.token_count != 128)
            fail("qwen image probe bridge");
        if (ds4_bridge_qwen_image_pixel_hash(image, sizeof(image), &hash,
                                             err, sizeof(err)) != 0 ||
            hash != UINT64_C(0x0123456789abcdef))
            fail("qwen image hash bridge");
        memset(err, 0, sizeof(err));
        if (ds4_bridge_qwen_image_probe(image, sizeof(image), NULL,
                                        err, sizeof(err)) == 0 ||
            !strstr(err, "NULL"))
            fail("qwen image probe NULL info");
    }

    memset(err, 0, sizeof(err));
    if (ds4_bridge_tokenize_text(NULL, "hi", toks, 4, &n, err, sizeof(err)) == 0)
        fail("tokenize_text NULL model");
    if (n != 0) fail("tokenize_text n_out");
    if (!strstr(err, "NULL")) fail("tokenize_text err");

    memset(err, 0, sizeof(err));
    n = -1;
    if (ds4_bridge_tokenize_rendered_chat(NULL, "hi", toks, 4, &n, err, sizeof(err)) == 0)
        fail("tokenize_chat NULL model");
    if (n != 0) fail("tokenize_chat n_out");

    memset(err, 0, sizeof(err));
    if (ds4_bridge_token_text(NULL, 1, text, sizeof(text), &tn, err, sizeof(err)) == 0)
        fail("token_text NULL model");
    if (tn != 0) fail("token_text n_out");

    if (ds4_bridge_token_eos(NULL) != -1) fail("token_eos");
    if (ds4_bridge_token_is_stop(NULL, 1) != 0) fail("token_is_stop");
    if (ds4_bridge_model_id(NULL) != -1) fail("model_id");
    if (ds4_bridge_model_routed_quant_bits(NULL) != 0) fail("routed_quant_bits NULL");
    memset(err, 0, sizeof(err));
    if (ds4_bridge_model_run_distributed_worker(NULL, 4096,
                                                err, sizeof(err)) == 0) {
        fail("distributed worker NULL model");
    }
    if (!strstr(err, "NULL")) {
        fail("distributed worker NULL err");
    }
    memset(err, 0, sizeof(err));
    if (ds4_bridge_session_distributed_route_ready(NULL,
                                                   err, sizeof(err)) != -1) {
        fail("distributed route NULL session");
    }
    if (!strstr(err, "NULL")) {
        fail("distributed route NULL err");
    }
    bridge_boot_prewarm_calls = 0;
    ds4_bridge_model_boot_prewarm(NULL);
    if (bridge_boot_prewarm_calls != 0) fail("boot_prewarm NULL");
    {
        ds4_engine *fake_native = malloc(1);
        struct ds4_bridge_model fake = {fake_native};
        if (!fake_native) fail("routed_quant_bits fake model");
        bridge_routed_quant_bits = 4;
        if (ds4_bridge_model_routed_quant_bits(&fake) != 4)
            fail("routed_quant_bits delegation");
        ds4_bridge_model_boot_prewarm(&fake);
        if (bridge_boot_prewarm_calls != 1)
            fail("boot_prewarm delegation");
        free(fake_native);
    }
    {
        ds4_bridge_model *model = NULL;
        ds4_bridge_model_open_options open_opt;
        ds4_bridge_distributed_options dist;
        memset(&open_opt, 0, sizeof(open_opt));
        memset(&dist, 0, sizeof(dist));
        open_opt.model_path = "fixture.gguf";
        open_opt.backend = DS4_BRIDGE_BACKEND_CPU;
        open_opt.power_percent = 37;
        open_opt.warm_weights = 1;
        open_opt.quality = 1;
        dist.role = DS4_BRIDGE_DISTRIBUTED_WORKER;
        dist.layer_start = 21;
        dist.layer_end = UINT32_MAX;
        dist.has_output = 1;
        dist.listen_host = "0.0.0.0";
        dist.listen_port = 7100;
        dist.coordinator_host = "127.0.0.1";
        dist.coordinator_port = 7000;

        bridge_dist_open_enabled = 1;
        bridge_dist_prepare_calls = 0;
        bridge_dist_run_calls = 0;
        bridge_dist_run_result = 0;
        memset(err, 0, sizeof(err));
        if (ds4_bridge_model_open_distributed(&model, &open_opt, &dist,
                                              err, sizeof(err)) != 0 || !model) {
            fail("distributed model open");
        }
        if (bridge_dist_prepare_calls != 1) {
            fail("distributed prepare call count");
        }
        if (!bridge_dist_engine_options.load_slice ||
            bridge_dist_engine_options.load_layer_start != 21 ||
            bridge_dist_engine_options.load_layer_end != UINT32_MAX ||
            !bridge_dist_engine_options.load_output ||
            bridge_dist_engine_options.distributed.role != DS4_DISTRIBUTED_WORKER) {
            fail("distributed engine option mapping");
        }
        if (bridge_dist_engine_options.power_percent != 37 ||
            !bridge_dist_engine_options.warm_weights ||
            !bridge_dist_engine_options.quality) {
            fail("engine tuning option mapping");
        }
        if (ds4_bridge_model_run_distributed_worker(model, 4096,
                                                    err, sizeof(err)) != 0) {
            fail("distributed worker run");
        }
        if (bridge_dist_run_calls != 1 || bridge_dist_run_ctx != 4096 ||
            bridge_dist_run_options.layers.start != 21 ||
            !bridge_dist_run_options.layers.has_output) {
            fail("distributed worker run mapping");
        }
        ds4_bridge_model_free(model);
        bridge_dist_open_enabled = 0;
    }
    if (ds4_bridge_session_sample(NULL, 1.0f, 0, 1.0f, 0.05f, &rng) != -1)
        fail("sample");
    if (ds4_bridge_session_ctx(NULL) != -1) fail("ctx");
    if (ds4_bridge_session_argmax(NULL) != -1) fail("argmax");
    if (ds4_bridge_step37_trial(NULL, 1, 4, NULL, NULL, 4, err, sizeof(err)) != -1) {
        fail("Step trial NULL session");
    }
    if (ds4_bridge_step37_commit(NULL, 1, err, sizeof(err)) == 0) {
        fail("Step commit NULL session");
    }
    if (ds4_bridge_session_argmax_excluding(NULL, 7) != -1)
        fail("argmax_excluding");
    if (ds4_bridge_session_pos(NULL) != -1) fail("pos");

    {
        static int32_t prompt[6800];
        ds4_session *fake_native = malloc(1);
        struct ds4_bridge_session fake = {NULL, fake_native};
        int cookie = 42;
        if (!fake_native) {
            fail("sync callback fake session");
        }

        bridge_dist_route_ready = 0;
        if (ds4_bridge_session_distributed_route_ready(&fake,
                                                       err, sizeof(err)) != 0) {
            fail("distributed route pending delegation");
        }
        bridge_dist_route_ready = 1;
        if (ds4_bridge_session_distributed_route_ready(&fake,
                                                       err, sizeof(err)) != 1) {
            fail("distributed route ready delegation");
        }

        memset(err, 0, sizeof(err));
        if (ds4_bridge_session_sync_cb(NULL, prompt, 6800, record_progress,
                                       &cookie, err, sizeof(err)) == 0) {
            fail("sync callback NULL session");
        }
        if (!strstr(err, "NULL")) {
            fail("sync callback NULL err");
        }

        bridge_sync_rc = 0;
        bridge_sync_calls = 0;
        bridge_progress_sets = 0;
        bridge_progress_clears = 0;
        progress_n = 0;
        if (ds4_bridge_session_sync_cb(&fake, prompt, 6800, record_progress,
                                       &cookie, err, sizeof(err)) != 0) {
            fail("sync callback success");
        }
        if (bridge_sync_calls != 1) {
            fail("sync callback call count");
        }
        if (bridge_progress_sets != 1 || bridge_progress_clears != 1) {
            fail("sync callback install lifecycle");
        }
        if (bridge_progress_active) {
            fail("sync callback left installed");
        }
        if (progress_n != 3) {
            fail("sync callback event count");
        }
        if (progress_current[0] != 0 || progress_current[1] != 4096 ||
            progress_current[2] != 6800) {
            fail("sync callback cadence");
        }
        for (int i = 0; i < progress_n; i++) {
            if (progress_total[i] != 6800) {
                fail("sync callback total");
            }
        }

        bridge_sync_rc = 7;
        progress_n = 0;
        if (ds4_bridge_session_sync_cb(&fake, prompt, 6800, record_progress,
                                       &cookie, err, sizeof(err)) != 7) {
            fail("sync callback failure rc");
        }
        if (bridge_progress_sets != 2 || bridge_progress_clears != 2) {
            fail("sync callback failure cleanup");
        }
        if (bridge_progress_active) {
            fail("sync callback failure left installed");
        }

        bridge_sync_rc = 0;
        progress_n = 0;
        if (ds4_bridge_session_sync(&fake, prompt, 6800,
                                    err, sizeof(err)) != 0) {
            fail("legacy sync success");
        }
        if (bridge_progress_sets != 2 || bridge_progress_clears != 2) {
            fail("legacy sync installed callback");
        }
        if (progress_n != 0) {
            fail("legacy sync forwarded progress");
        }
        free(fake_native);
    }

    memset(err, 0, sizeof(err));
    if (ds4_bridge_session_load_payload_range(NULL, "/tmp/missing", 0, 1,
                                              err, sizeof(err)) == 0)
        fail("payload range NULL session");
    if (!strstr(err, "NULL")) fail("payload range err");

    {
        ds4_session *fake_native = malloc(1);
        struct ds4_bridge_session fake = {NULL, fake_native};
        ds4_bridge_session *session = &fake;
        char path[] = "/tmp/ds4-bridge-range-XXXXXX";
        unsigned char data[16] = {0};
        if (!fake_native) fail("payload range fake session");
        int fd = mkstemp(path);
        if (fd < 0)
            fail("payload range fixture");
        if (write(fd, data, sizeof(data)) != (ssize_t)sizeof(data)) {
            close(fd);
            unlink(path);
            fail("payload range fixture write");
        }
        if (close(fd) != 0) {
            unlink(path);
            fail("payload range fixture close");
        }

        bridge_payload_load_calls = 0;
        bridge_payload_load_offset = -1;
        bridge_payload_load_bytes = 0;
        if (ds4_bridge_session_load_payload_range(session, path, 3, 5,
                                                  err, sizeof(err)) != 0)
            fail("payload range load");
        if (bridge_payload_load_calls != 1) fail("payload range call count");
        if (bridge_payload_load_offset != 3) fail("payload range seek");
        if (bridge_payload_load_bytes != 5) fail("payload range length");

        if (ds4_bridge_session_load_payload_range(session, path, 12, 5,
                                                  err, sizeof(err)) == 0)
            fail("payload range EOF");
        if (bridge_payload_load_calls != 1) fail("payload range EOF called native");

        if (ds4_bridge_session_load_payload_range(session, path, UINT64_MAX, 2,
                                                  err, sizeof(err)) == 0)
            fail("payload range overflow");
        if (bridge_payload_load_calls != 1) fail("payload range overflow called native");
        if (unlink(path) != 0) fail("payload range unlink");
        free(fake_native);
    }

    {
        ds4_bridge_snapshot *snap = NULL;
        memset(err, 0, sizeof(err));
        if (ds4_bridge_snapshot_create(&snap, err, sizeof(err)) != 0 || !snap)
            fail("snapshot create");
        if (ds4_bridge_snapshot_len(snap) != 0) fail("snapshot initial len");
        if (ds4_bridge_session_save_snapshot(NULL, snap, err, sizeof(err)) == 0)
            fail("snapshot save NULL session");
        if (ds4_bridge_session_load_snapshot(NULL, snap, err, sizeof(err)) == 0)
            fail("snapshot load NULL session");
        ds4_bridge_snapshot_free(snap);
    }

    if (ds4_bridge_mem_census_snap(NULL) == 0) fail("census NULL");
    if (ds4_bridge_mem_observe_snap(NULL) == 0) fail("observe NULL");
    {
        ds4_bridge_mem_census img;
        ds4_bridge_mem_observe o;
        memset(&img, 0xff, sizeof(img));
        memset(&o, 0xff, sizeof(o));
        if (ds4_bridge_mem_census_snap(&img) != 0) fail("census rc");
        if (img.supported != 0) fail("census supported");
        if (img.faults != 0) fail("census faults");
        if (img.epoch != 0) fail("census epoch");
        if (img.torn_fallbacks != 0) fail("census torn");
        if (img.cells[0][0].requested != 0) fail("census cell");
        if (sizeof(ds4_bridge_mem_cell) != 48) fail("cell sizeof");
        if (ds4_bridge_mem_observe_snap(&o) != 0) fail("observe rc");
        if (o.status != 1) fail("observe unsupported");
        if (o.source != 0) fail("observe source");
        if (o.free_bytes != 0) fail("observe free");
        if (ds4_bridge_mem_substrate_outstanding() != 0) fail("substrate");
    }

    {
        ds4_bridge_token_score sc[2];
        memset(err, 0, sizeof(err));
        n = -1;
        if (ds4_bridge_encode_chat_prompt(NULL, NULL, "hi", 0, toks, 4, &n,
                                          err, sizeof(err)) == 0)
            fail("encode_chat NULL model");
        if (n != 0) fail("encode_chat n_out");
        if (ds4_bridge_encode_chat_prompt(NULL, NULL, "hi", 9, toks, 4, &n,
                                          err, sizeof(err)) == 0)
            fail("encode_chat think range");
        if (ds4_bridge_session_top_logprobs(NULL, sc, 2) != -1)
            fail("top_logprobs NULL");
    }

    {
        ds4_bridge_batch_ctx *bc = (ds4_bridge_batch_ctx *)0;
        memset(err, 0, sizeof(err));
        if (ds4_bridge_batch_ctx_create_fit(NULL, 2048, 4, 8192, &bc,
                                            err, sizeof(err)) == 0)
            fail("batch_ctx NULL model");
        if (bc != NULL) fail("batch_ctx out");
        if (!strstr(err, "NULL")) fail("batch_ctx err");
        ds4_bridge_batch_ctx_destroy(NULL);
        if (ds4_bridge_batch_ctx_max_seq(NULL) != 0) fail("batch max_seq");
        if (ds4_bridge_batch_ctx_raw_cap(NULL) != 0) fail("batch raw_cap");
        if (ds4_bridge_batch_ctx_seq_cap(NULL) != 0) fail("batch seq_cap");
        memset(err, 0, sizeof(err));
        if (ds4_bridge_continuous_generate(NULL, NULL, NULL, NULL, NULL,
                                           err, sizeof(err)) == 0)
            fail("cont NULL ctx");
        if (!strstr(err, "NULL")) fail("cont err");
        {
            const int32_t *prompt_ptrs[1] = {toks};
            const int32_t prompt_lens[1] = {1};
            const int32_t max_new[1] = {1};
            const int32_t eos[1] = {-1};
            int32_t out_lens[1] = {-1};
            int32_t out_finish[1] = {-1};
            if (ds4_bridge_batch_ctx_generate_static(
                    NULL, prompt_ptrs, prompt_lens, max_new, eos, 1,
                    toks, 4, out_lens, out_finish, err, sizeof(err)) == 0)
                fail("static batch NULL ctx");
            if (out_lens[0] != -1 || out_finish[0] != -1)
                fail("static batch NULL touched outputs");
        }

        n = -1;
        memset(err, 0, sizeof(err));
        if (ds4_bridge_batch_ctx_bank_snapshot(NULL, 0, toks, 4, &n,
                                               NULL, err, sizeof(err)) == 0)
            fail("bank snapshot NULL ctx");
        if (n != 0) fail("bank snapshot NULL n");
        if (!strstr(err, "NULL")) fail("bank snapshot NULL err");
        if (ds4_bridge_batch_ctx_bank_save_payload(NULL, 0, "/tmp/nope",
                                                   err, sizeof(err)) == 0)
            fail("bank save NULL ctx");
        if (ds4_bridge_batch_ctx_bank_load_payload_range(
                NULL, 0, "/tmp/nope", 0, 1, err, sizeof(err)) == 0)
            fail("bank load NULL ctx");

        {
            ds4_batch_ctx *fake_native = malloc(1);
            struct ds4_bridge_batch_ctx fake = {fake_native};
            uint64_t generation = 0;
            char path[] = "/tmp/ds4-bridge-bank-XXXXXX";
            unsigned char data[16] = {0};
            if (!fake_native) fail("bank fake ctx");
            bridge_batch_max_seq = 2;
            bridge_bank_committed = 3;
            bridge_bank_tokens[0] = 11;
            bridge_bank_tokens[1] = 22;
            bridge_bank_tokens[2] = 33;
            bridge_bank_generation = 9;
            n = -1;
            memset(toks, 0, sizeof(toks));
            if (ds4_bridge_batch_ctx_bank_snapshot(&fake, 1, toks, 4, &n,
                                                   &generation, err,
                                                   sizeof(err)) != 0)
                fail("bank snapshot success");
            if (n != 3 || generation != 9 || toks[0] != 11 ||
                toks[1] != 22 || toks[2] != 33)
                fail("bank snapshot values");
            n = -1;
            generation = 99;
            if (ds4_bridge_batch_ctx_bank_snapshot(&fake, 2, toks, 4, &n,
                                                   &generation, err,
                                                   sizeof(err)) == 0)
                fail("bank snapshot OOB");
            if (n != 0 || generation != 0) fail("bank snapshot OOB outputs");
            n = -1;
            if (ds4_bridge_batch_ctx_bank_snapshot(&fake, 1, toks, 2, &n,
                                                   &generation, err,
                                                   sizeof(err)) == 0)
                fail("bank snapshot short buffer");
            if (n != 3) fail("bank snapshot required length");

            bridge_cont_run = 1;
            memset(&bridge_cont_stats, 0, sizeof(bridge_cont_stats));
            bridge_cont_stats.first_token_sec = 10.0;
            bridge_cont_stats.done_sec = 10.333;
            /* The native decode includes EOS: five tokens over four steps. */
            bridge_cont_stats.decode_tokens = 5;
            bridge_cont_stats.decode_steps = 4;
            memset(&done_stats, 0, sizeof(done_stats));
            if (ds4_bridge_continuous_generate(
                    &fake, no_admit, NULL, record_done, NULL,
                    err, sizeof(err)) != 0) {
                fail("cont stats generate");
            }
            if (done_stats.decode_tokens != 5 || done_stats.decode_steps != 4 ||
                done_stats.decode_ms < 332.9 || done_stats.decode_ms > 333.1) {
                fail("cont stats values");
            }
            bridge_cont_run = 0;

            {
                const int32_t first[] = {10, 11};
                const int32_t second[] = {20, 21, 22};
                const int32_t *prompt_ptrs[] = {first, second};
                const int32_t prompt_lens[] = {2, 3};
                const int32_t max_new[] = {2, 3};
                const int32_t eos[] = {99, -1};
                int32_t generated[5] = {0};
                int32_t out_lens[2] = {-1, -1};
                int32_t out_finish[2] = {-1, -1};

                bridge_static_batch_calls = 0;
                if (ds4_bridge_batch_ctx_generate_static(
                        &fake, prompt_ptrs, prompt_lens, max_new, eos, 2,
                        generated, 5, out_lens, out_finish,
                        err, sizeof(err)) != 0)
                    fail("static batch generate");
                if (bridge_static_batch_calls != 1 ||
                    bridge_static_batch_n != 2 ||
                    bridge_static_batch_prompt_lens[0] != 2 ||
                    bridge_static_batch_prompt_lens[1] != 3 ||
                    bridge_static_batch_max_new[0] != 2 ||
                    bridge_static_batch_max_new[1] != 3 ||
                    bridge_static_batch_eos[0] != 99 ||
                    bridge_static_batch_eos[1] != -1)
                    fail("static batch native inputs");
                if (out_lens[0] != 2 || out_lens[1] != 3 ||
                    out_finish[0] != 1 || out_finish[1] != 0 ||
                    generated[0] != 101 || generated[1] != 102 ||
                    generated[2] != 201 || generated[3] != 202 ||
                    generated[4] != 203)
                    fail("static batch copied outputs");

                memset(err, 0, sizeof(err));
                out_lens[0] = out_lens[1] = -1;
                out_finish[0] = out_finish[1] = -1;
                if (ds4_bridge_batch_ctx_generate_static(
                        &fake, prompt_ptrs, prompt_lens, max_new, eos, 2,
                        generated, 4, out_lens, out_finish,
                        err, sizeof(err)) == 0)
                    fail("static batch short output");
                if (!strstr(err, "small") || out_lens[0] != 0 ||
                    out_lens[1] != 0 || out_finish[0] != 0 ||
                    out_finish[1] != 0)
                    fail("static batch short output contract");
            }

            int fd = mkstemp(path);
            if (fd < 0) fail("bank payload fixture");
            if (write(fd, "KEEP", 4) != 4) fail("bank sentinel write");
            if (close(fd) != 0) fail("bank payload fixture close");
            bridge_bank_save_calls = 0;
            bridge_bank_save_result = 1;
            if (ds4_bridge_batch_ctx_bank_save_payload(&fake, 1, path,
                                                       err, sizeof(err)) == 0)
                fail("bank payload failed save");
            fd = open(path, O_RDONLY);
            if (fd < 0) fail("bank sentinel open");
            char saved[4];
            if (read(fd, saved, sizeof(saved)) != (ssize_t)sizeof(saved) ||
                memcmp(saved, "KEEP", sizeof(saved)))
                fail("bank failed save replaced destination");
            close(fd);
            bridge_bank_save_result = 0;
            if (ds4_bridge_batch_ctx_bank_save_payload(&fake, 1, path,
                                                       err, sizeof(err)) != 0)
                fail("bank payload save");
            if (bridge_bank_save_calls != 2) fail("bank payload save call");
            fd = open(path, O_RDONLY);
            if (fd < 0) fail("bank payload saved open");
            if (read(fd, saved, sizeof(saved)) != (ssize_t)sizeof(saved) ||
                memcmp(saved, "BANK", sizeof(saved)))
                fail("bank payload saved bytes");
            close(fd);

            fd = open(path, O_WRONLY | O_TRUNC);
            if (fd < 0 || write(fd, data, sizeof(data)) != (ssize_t)sizeof(data))
                fail("bank payload range fixture");
            close(fd);
            bridge_bank_load_calls = 0;
            bridge_bank_load_offset = -1;
            bridge_bank_load_bytes = 0;
            if (ds4_bridge_batch_ctx_bank_load_payload_range(
                    &fake, 1, path, 3, 5, err, sizeof(err)) != 0)
                fail("bank payload range load");
            if (bridge_bank_load_calls != 1 || bridge_bank_load_offset != 3 ||
                bridge_bank_load_bytes != 5)
                fail("bank payload range delegation");
            if (ds4_bridge_batch_ctx_bank_load_payload_range(
                    &fake, 1, path, 12, 5, err, sizeof(err)) == 0)
                fail("bank payload range truncated");
            if (ds4_bridge_batch_ctx_bank_load_payload_range(
                    &fake, 1, path, UINT64_MAX, 2, err, sizeof(err)) == 0)
                fail("bank payload range overflow");
            if (bridge_bank_load_calls != 1)
                fail("bank invalid range called native");
            if (unlink(path) != 0) fail("bank payload unlink");
            free(fake_native);
        }
    }

    printf("ok\n");
    return 0;
}
