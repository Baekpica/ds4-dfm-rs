#include "ds4_bridge.h"

#include "ds4.h"
#include "ds4_distributed.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

/* Thin wrappers over the existing engine/session API.  No inference logic
 * lives here; this file exists so Rust never includes ds4.h. */

struct ds4_bridge_model {
    ds4_engine *engine;
    ds4_dist_options distributed;
    int distributed_set;
};

struct ds4_bridge_session {
    ds4_bridge_model *model;
    ds4_session *session;
};

struct ds4_bridge_snapshot {
    ds4_session_snapshot snapshot;
};

static void set_err(char *err, size_t errlen, const char *msg)
{
    if (!err || errlen == 0) return;
    snprintf(err, errlen, "%s", msg ? msg : "unknown error");
}

static int map_backend(int bridge_backend, ds4_backend *out, char *err, size_t errlen)
{
    switch (bridge_backend) {
    case DS4_BRIDGE_BACKEND_CUDA:
        *out = DS4_BACKEND_CUDA;
        return 0;
    case DS4_BRIDGE_BACKEND_METAL:
        *out = DS4_BACKEND_METAL;
        return 0;
    case DS4_BRIDGE_BACKEND_CPU:
        *out = DS4_BACKEND_CPU;
        return 0;
    default:
        set_err(err, errlen, "unknown backend");
        return 1;
    }
}

int ds4_bridge_bind_plan_check(const ds4_bridge_bind_plan *plan,
                               char *err, size_t errlen)
{
    uint32_t i;

    if (!plan) {
        set_err(err, errlen, "plan-null");
        return 1;
    }
    if (plan->n_slots > 0 && !plan->slots) {
        set_err(err, errlen, "slots-null");
        return 1;
    }
    if (plan->n_shards > 0 && !plan->shards) {
        set_err(err, errlen, "shards-null");
        return 1;
    }
    for (i = 0; i < plan->n_slots; i++) {
        const ds4_bridge_bind_slot *s = &plan->slots[i];
        if (!s->name || !s->name[0]) {
            set_err(err, errlen, "name-empty");
            return 1;
        }
        if (s->required && !s->found) {
            if (err && errlen) snprintf(err, errlen, "missing %s", s->name);
            return 1;
        }
        if (s->found && (s->ndim == 0 || s->ndim > DS4_BRIDGE_MAX_DIMS)) {
            set_err(err, errlen, "bad-ndim");
            return 1;
        }
    }
    return 0;
}

int ds4_bridge_bind_plan_match(const ds4_bridge_bind_plan *host,
                               const ds4_bridge_bind_plan *native,
                               char *err, size_t errlen)
{
    uint32_t i, d;

    if (!host || !native) {
        set_err(err, errlen, "plan-null");
        return 1;
    }
    if (host->n_slots != native->n_slots) {
        set_err(err, errlen, "count-mismatch");
        return 1;
    }
    for (i = 0; i < host->n_slots; i++) {
        const ds4_bridge_bind_slot *h = &host->slots[i];
        const ds4_bridge_bind_slot *n = &native->slots[i];
        if (!h->name || !n->name || strcmp(h->name, n->name) != 0) {
            set_err(err, errlen, "name-mismatch");
            return 1;
        }
        if (h->required != n->required) {
            set_err(err, errlen, "need-mismatch");
            return 1;
        }
        if (h->found != n->found) {
            set_err(err, errlen, "found-mismatch");
            return 1;
        }
        if (!h->found) continue;
        if (h->type != n->type) {
            set_err(err, errlen, "type-mismatch");
            return 1;
        }
        if (h->ndim != n->ndim) {
            set_err(err, errlen, "dim-mismatch");
            return 1;
        }
        for (d = 0; d < h->ndim && d < DS4_BRIDGE_MAX_DIMS; d++) {
            if (h->dim[d] != n->dim[d]) {
                set_err(err, errlen, "dim-mismatch");
                return 1;
            }
        }
        if (h->rel_offset != n->rel_offset || h->abs_offset != n->abs_offset) {
            set_err(err, errlen, "offset-mismatch");
            return 1;
        }
        if (h->bytes != n->bytes) {
            set_err(err, errlen, "bytes-mismatch");
            return 1;
        }
        if (h->shard != n->shard) {
            set_err(err, errlen, "shard-mismatch");
            return 1;
        }
    }
    if (host->n_shards != native->n_shards ||
        host->data_pos != native->data_pos ||
        host->alignment != native->alignment ||
        host->page != native->page) {
        set_err(err, errlen, "data-mismatch");
        return 1;
    }
    return 0;
}

static int map_distributed_options(
        const ds4_bridge_distributed_options *src,
        ds4_dist_options *dst,
        char *err,
        size_t errlen)
{
    if (!src || !dst) {
        set_err(err, errlen, "distributed options are NULL");
        return 1;
    }
    memset(dst, 0, sizeof(*dst));
    switch (src->role) {
    case DS4_BRIDGE_DISTRIBUTED_NONE:
        dst->role = DS4_DISTRIBUTED_NONE;
        break;
    case DS4_BRIDGE_DISTRIBUTED_COORDINATOR:
        dst->role = DS4_DISTRIBUTED_COORDINATOR;
        break;
    case DS4_BRIDGE_DISTRIBUTED_WORKER:
        dst->role = DS4_DISTRIBUTED_WORKER;
        break;
    default:
        set_err(err, errlen, "invalid distributed role");
        return 1;
    }
    dst->layers.start = src->layer_start;
    dst->layers.end = src->layer_end;
    dst->layers.has_output = src->has_output != 0;
    dst->layers.set = src->role != DS4_BRIDGE_DISTRIBUTED_NONE;
    dst->listen_host = src->listen_host;
    dst->listen_port = src->listen_port;
    dst->coordinator_host = src->coordinator_host;
    dst->coordinator_port = src->coordinator_port;
    dst->prefill_chunk = src->prefill_chunk;
    dst->prefill_window = src->prefill_window;
    dst->activation_bits = src->activation_bits;
    dst->replay_check = src->replay_check != 0;
    dst->debug = src->debug != 0;
    return 0;
}

static int model_open_impl(ds4_bridge_model **out,
                           const ds4_bridge_model_open_options *opt,
                           const ds4_bridge_distributed_options *distributed,
                           char *err, size_t errlen)
{
    ds4_engine_options eopt;
    ds4_dist_options dist;
    ds4_engine *engine = NULL;
    ds4_bridge_model *m;
    int rc;

    if (out) *out = NULL;
    if (!out) {
        set_err(err, errlen, "out is NULL");
        return 1;
    }
    if (!opt || !opt->model_path || !opt->model_path[0]) {
        set_err(err, errlen, "model_path is required");
        return 1;
    }
    if (opt->plan && ds4_bridge_bind_plan_check(opt->plan, err, errlen) != 0) {
        return 1;
    }

    memset(&eopt, 0, sizeof(eopt));
    eopt.model_path = opt->model_path;
    eopt.vision_path = opt->vision_path;
    eopt.n_threads = opt->n_threads;
    eopt.defer_boot_prewarm = opt->defer_boot_prewarm != 0;
    eopt.power_percent = opt->power_percent;
    eopt.warm_weights = opt->warm_weights != 0;
    eopt.quality = opt->quality != 0;
    eopt.mtp_path = opt->mtp_path;
    eopt.dspark_path = opt->dspark_path;
    eopt.mtp_draft_tokens = opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    eopt.mtp_margin = opt->mtp_margin;
    eopt.directional_steering_file = opt->directional_steering_file;
    eopt.directional_steering_attn = opt->directional_steering_attn;
    eopt.directional_steering_ffn = opt->directional_steering_ffn;
    if (map_backend(opt->backend, &eopt.backend, err, errlen) != 0) return 1;
    if (distributed) {
        if (map_distributed_options(distributed, &dist, err, errlen) != 0 ||
            ds4_dist_prepare_engine_options(&dist, &eopt, err, errlen) != 0) {
            return 1;
        }
    }

    if (opt->tensors) ds4_host_tensor_dir_install(opt->tensors);
    if (opt->shape) ds4_host_shape_install(opt->shape);
    if (opt->vocab) ds4_host_vocab_install(opt->vocab);
    if (opt->bind) ds4_host_bind_map_install(opt->bind);
    /* Sibling maps index the sibling's own tensor dir; only meaningful
     * when the matching path rides the same open. */
    if (opt->mtp_bind && opt->mtp_path && opt->mtp_path[0])
        ds4_host_mtp_bind_map_install(opt->mtp_bind);
    if (opt->dspark_bind && opt->dspark_path && opt->dspark_path[0])
        ds4_host_dspark_bind_map_install(opt->dspark_bind);
    rc = ds4_engine_open(&engine, &eopt);
    ds4_host_dspark_bind_map_clear();
    ds4_host_mtp_bind_map_clear();
    ds4_host_bind_map_clear();
    ds4_host_vocab_clear();
    ds4_host_shape_clear();
    ds4_host_tensor_dir_clear();
    if (rc != 0 || !engine) {
        set_err(err, errlen, "ds4_engine_open failed");
        return rc != 0 ? rc : 1;
    }

    m = calloc(1, sizeof(*m));
    if (!m) {
        ds4_engine_close(engine);
        set_err(err, errlen, "out of memory");
        return 1;
    }
    m->engine = engine;
    if (distributed) {
        m->distributed = dist;
        m->distributed_set = 1;
    }
    *out = m;
    return 0;
}

int ds4_bridge_model_open(ds4_bridge_model **out,
                          const ds4_bridge_model_open_options *opt,
                          char *err, size_t errlen)
{
    return model_open_impl(out, opt, NULL, err, errlen);
}

int ds4_bridge_model_open_distributed(
        ds4_bridge_model **out,
        const ds4_bridge_model_open_options *opt,
        const ds4_bridge_distributed_options *distributed,
        char *err, size_t errlen)
{
    return model_open_impl(out, opt, distributed, err, errlen);
}

int ds4_bridge_model_run_distributed_worker(ds4_bridge_model *m,
                                            int32_t ctx_size,
                                            char *err, size_t errlen)
{
    ds4_dist_generation_options gen;
    int rc;
    if (!m || !m->engine) {
        set_err(err, errlen, "model is NULL");
        return 1;
    }
    if (!m->distributed_set ||
        m->distributed.role != DS4_DISTRIBUTED_WORKER) {
        set_err(err, errlen, "model is not a distributed worker");
        return 1;
    }
    if (ctx_size <= 0) {
        set_err(err, errlen, "ctx_size must be positive");
        return 1;
    }
    memset(&gen, 0, sizeof(gen));
    gen.ctx_size = ctx_size;
    rc = ds4_dist_run(m->engine, &m->distributed, &gen);
    if (rc != 0) {
        set_err(err, errlen, "distributed worker stopped with an error");
    }
    return rc;
}

void ds4_bridge_model_boot_prewarm(ds4_bridge_model *m)
{
    if (!m) return;
    ds4_engine_boot_prewarm(m->engine);
}

void ds4_bridge_model_free(ds4_bridge_model *m)
{
    if (!m) return;
    ds4_engine_close(m->engine);
    free(m);
}

int ds4_bridge_session_create(ds4_bridge_session **out,
                              ds4_bridge_model *m,
                              int ctx_size,
                              char *err, size_t errlen)
{
    ds4_session *session = NULL;
    ds4_bridge_session *s;
    int rc;

    if (out) *out = NULL;
    if (!out) {
        set_err(err, errlen, "out is NULL");
        return 1;
    }
    if (!m || !m->engine) {
        set_err(err, errlen, "model is NULL");
        return 1;
    }
    if (ctx_size <= 0) {
        set_err(err, errlen, "ctx_size must be positive");
        return 1;
    }

    rc = ds4_session_create(&session, m->engine, ctx_size);
    if (rc != 0 || !session) {
        set_err(err, errlen, "ds4_session_create failed");
        return rc != 0 ? rc : 1;
    }

    s = calloc(1, sizeof(*s));
    if (!s) {
        ds4_session_free(session);
        set_err(err, errlen, "out of memory");
        return 1;
    }
    s->model = m;
    s->session = session;
    *out = s;
    return 0;
}

void ds4_bridge_session_free(ds4_bridge_session *s)
{
    if (!s) return;
    ds4_session_free(s->session);
    free(s);
}

typedef struct {
    ds4_bridge_prefill_fn progress;
    void *ud;
} sync_tramp;

static void sync_tramp_progress(void *ud, const char *event,
                                int current, int total)
{
    sync_tramp *t = ud;

    if (!t || !t->progress || !event) {
        return;
    }
    if (strcmp(event, "prefill_chunk") != 0) {
        return;
    }
    t->progress(t->ud, (int32_t)current, (int32_t)total);
}

static int session_sync(ds4_bridge_session *s,
                        const int32_t *tokens, int n_tokens,
                        ds4_bridge_prefill_fn progress, void *ud,
                        char *err, size_t errlen)
{
    ds4_tokens prompt;
    sync_tramp t;
    int rc;

    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (n_tokens < 0) {
        set_err(err, errlen, "n_tokens is negative");
        return 1;
    }
    if (n_tokens > 0 && !tokens) {
        set_err(err, errlen, "tokens is NULL");
        return 1;
    }

    memset(&prompt, 0, sizeof(prompt));
    /* Borrowed view: ds4_session_sync must not retain prompt.v. */
    prompt.v = (int *)(void *)tokens;
    prompt.len = n_tokens;
    prompt.cap = n_tokens;
    if (!progress) {
        return ds4_session_sync(s->session, &prompt, err, errlen);
    }

    t.progress = progress;
    t.ud = ud;
    ds4_session_set_progress(s->session, sync_tramp_progress, &t);
    rc = ds4_session_sync(s->session, &prompt, err, errlen);
    ds4_session_set_progress(s->session, NULL, NULL);
    return rc;
}

int ds4_bridge_session_sync(ds4_bridge_session *s,
                            const int32_t *tokens, int n_tokens,
                            char *err, size_t errlen)
{
    return session_sync(s, tokens, n_tokens, NULL, NULL, err, errlen);
}

int ds4_bridge_model_vision_probe(ds4_bridge_model *m,
                                  const uint8_t *data, size_t data_len,
                                  ds4_bridge_vision_info *info,
                                  char *err, size_t errlen)
{
    ds4_vision_image_info native;

    if (!m || !m->engine || !info) {
        set_err(err, errlen, "model or image info is NULL");
        return 1;
    }
    if (!ds4_engine_vision_probe(m->engine, data, data_len,
                                 &native, err, errlen)) return 1;
    info->source_width = native.source_width;
    info->source_height = native.source_height;
    info->content_width = native.content_width;
    info->content_height = native.content_height;
    info->padded_width = native.padded_width;
    info->padded_height = native.padded_height;
    info->grid_height = native.grid_height;
    info->grid_width = native.grid_width;
    info->token_count = native.token_count;
    return 0;
}

int ds4_bridge_session_sync_vision(ds4_bridge_session *s,
                                   const int32_t *tokens, int n_tokens,
                                   const ds4_bridge_vision_input *images,
                                   uint32_t image_count,
                                   char *err, size_t errlen)
{
    ds4_tokens prompt;
    ds4_vision_span spans[4] = {0};
    int rc = 1;

    if (!s || !s->session || !s->model || !s->model->engine ||
        !tokens || n_tokens <= 0 || !images || image_count == 0u ||
        image_count > 4u) {
        set_err(err, errlen, "invalid vision sync input");
        return 1;
    }
    prompt.v = (int *)(void *)tokens;
    prompt.len = n_tokens;
    prompt.cap = n_tokens;
    for (uint32_t i = 0; i < image_count; i++) {
        if (!images[i].data || images[i].data_len == 0u ||
            !ds4_engine_vision_encode_memory(
                s->model->engine, images[i].data, images[i].data_len,
                &spans[i].embedding, err, errlen)) goto cleanup;
        spans[i].token_start = images[i].token_offset;
    }
    rc = ds4_session_sync_multimodal(
        s->session, &prompt, spans, image_count, err, errlen);

cleanup:
    for (uint32_t i = 0; i < image_count; i++)
        ds4_vision_embedding_free(&spans[i].embedding);
    return rc;
}

int ds4_bridge_session_sync_cb(ds4_bridge_session *s,
                               const int32_t *tokens, int n_tokens,
                               ds4_bridge_prefill_fn progress, void *ud,
                               char *err, size_t errlen)
{
    return session_sync(s, tokens, n_tokens, progress, ud, err, errlen);
}

int ds4_bridge_eval(ds4_bridge_session *s, int32_t token,
                    char *err, size_t errlen)
{
    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    return ds4_session_eval(s->session, (int)token, err, errlen);
}

int ds4_bridge_session_eval_layer_slice(ds4_bridge_session *s,
                                        const int32_t *tokens,
                                        uint32_t n_tokens,
                                        uint32_t pos0,
                                        uint32_t layer_start,
                                        uint32_t layer_end,
                                        const float *input_hc,
                                        float *output_hc,
                                        int32_t output_logits,
                                        float *logits,
                                        char *err, size_t errlen)
{
    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (!tokens && n_tokens != 0) {
        set_err(err, errlen, "tokens is NULL");
        return 1;
    }
    return ds4_session_eval_layer_slice(s->session,
                                        tokens,
                                        n_tokens,
                                        pos0,
                                        layer_start,
                                        layer_end,
                                        input_hc,
                                        output_hc,
                                        output_logits != 0,
                                        logits,
                                        err,
                                        errlen);
}

int ds4_bridge_session_layer_slice_reset(ds4_bridge_session *s,
                                         char *err, size_t errlen)
{
    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    return ds4_session_layer_slice_reset(s->session, err, errlen);
}

int ds4_bridge_eval_speculative_argmax(ds4_bridge_session *s,
                                       int32_t first_token,
                                       int32_t max_tokens,
                                       int32_t eos_token,
                                       int32_t *accepted,
                                       int32_t accepted_cap,
                                       char *err, size_t errlen)
{
    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return -1;
    }
    if (!accepted && accepted_cap != 0) {
        set_err(err, errlen, "accepted is NULL");
        return -1;
    }
    return ds4_session_eval_speculative_argmax(s->session,
                                               (int)first_token,
                                               (int)max_tokens,
                                               (int)eos_token,
                                               accepted,
                                               (int)accepted_cap,
                                               err,
                                               errlen);
}

int ds4_bridge_session_argmax(ds4_bridge_session *s)
{
    if (!s || !s->session) return -1;
    return ds4_session_argmax(s->session);
}

int ds4_bridge_session_argmax_excluding(ds4_bridge_session *s,
                                        int32_t excluded_id)
{
    if (!s || !s->session) return -1;
    return ds4_session_argmax_excluding(s->session, (int)excluded_id);
}

int ds4_bridge_session_pos(ds4_bridge_session *s)
{
    if (!s || !s->session) return -1;
    return ds4_session_pos(s->session);
}

int ds4_bridge_session_ctx(ds4_bridge_session *s)
{
    if (!s || !s->session) return -1;
    return ds4_session_ctx(s->session);
}

int ds4_bridge_session_graph_pending(ds4_bridge_session *s)
{
    if (!s || !s->session) return 0;
    return ds4_session_graph_pending(s->session);
}

int ds4_bridge_session_power(ds4_bridge_session *s)
{
    if (!s || !s->session) return 100;
    return ds4_session_power(s->session);
}

int ds4_bridge_session_set_power(ds4_bridge_session *s, int power_percent)
{
    if (!s || !s->session) return 1;
    return ds4_session_set_power(s->session, power_percent);
}

void ds4_bridge_session_rewind(ds4_bridge_session *s, int pos)
{
    if (!s || !s->session) return;
    ds4_session_rewind(s->session, pos);
}

void ds4_bridge_session_invalidate(ds4_bridge_session *s)
{
    if (!s || !s->session) return;
    ds4_session_invalidate(s->session);
}

uint64_t ds4_bridge_session_generation(ds4_bridge_session *s)
{
    if (!s || !s->session) return 0;
    return ds4_session_generation(s->session);
}

int ds4_bridge_session_prefill_cap(ds4_bridge_session *s)
{
    if (!s || !s->session) return 0;
    return ds4_session_prefill_cap(s->session);
}

int ds4_bridge_session_exaone_rewind_span(ds4_bridge_session *s)
{
    if (!s || !s->session) return 0;
    return ds4_session_exaone_rewind_span(s->session);
}

int ds4_bridge_session_distributed_route_ready(ds4_bridge_session *s,
                                               char *err, size_t errlen)
{
    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return -1;
    }
    return ds4_session_distributed_route_ready(s->session, err, errlen);
}

int ds4_bridge_session_sample(ds4_bridge_session *s,
                              float temperature, int top_k, float top_p, float min_p,
                              uint64_t *rng)
{
    if (!s || !s->session) return -1;
    return ds4_session_sample(s->session, temperature, top_k, top_p, min_p, rng);
}

int ds4_bridge_session_save_payload(ds4_bridge_session *s, const char *path,
                                    char *err, size_t errlen)
{
    FILE *fp;
    int rc;

    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (!path || !path[0]) {
        set_err(err, errlen, "payload path is required");
        return 1;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        set_err(err, errlen, "failed to open session payload for write");
        return 1;
    }
    rc = ds4_session_save_payload(s->session, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        set_err(err, errlen, "failed to close session payload");
        return 1;
    }
    return rc;
}

int ds4_bridge_session_load_payload(ds4_bridge_session *s, const char *path,
                                    char *err, size_t errlen)
{
    FILE *fp;
    int rc;
    off_t sz;
    uint64_t payload_bytes;

    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (!path || !path[0]) {
        set_err(err, errlen, "payload path is required");
        return 1;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        set_err(err, errlen, "failed to open session payload for read");
        return 1;
    }
    if (fseeko(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        set_err(err, errlen, "failed to measure session payload");
        return 1;
    }
    sz = ftello(fp);
    if (sz < 0) {
        fclose(fp);
        set_err(err, errlen, "failed to measure session payload");
        return 1;
    }
    payload_bytes = (uint64_t)sz;
    if (fseeko(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        set_err(err, errlen, "failed to rewind session payload");
        return 1;
    }
    rc = ds4_session_load_payload(s->session, fp, payload_bytes, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        set_err(err, errlen, "failed to close session payload");
        return 1;
    }
    return rc;
}

int ds4_bridge_session_load_payload_range(ds4_bridge_session *s, const char *path,
                                          uint64_t offset, uint64_t length,
                                          char *err, size_t errlen)
{
    FILE *fp;
    int rc;
    off_t sz;
    uint64_t file_bytes;

    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (!path || !path[0]) {
        set_err(err, errlen, "payload path is required");
        return 1;
    }
    if (offset > UINT64_MAX - length) {
        set_err(err, errlen, "session payload range overflows");
        return 1;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        set_err(err, errlen, "failed to open session payload for read");
        return 1;
    }
    if (fseeko(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        set_err(err, errlen, "failed to measure session payload");
        return 1;
    }
    sz = ftello(fp);
    if (sz < 0) {
        fclose(fp);
        set_err(err, errlen, "failed to measure session payload");
        return 1;
    }
    file_bytes = (uint64_t)sz;
    if (offset > file_bytes || length > file_bytes - offset) {
        fclose(fp);
        set_err(err, errlen, "truncated session payload range");
        return 1;
    }
    if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) {
        fclose(fp);
        set_err(err, errlen, "failed to seek session payload range");
        return 1;
    }
    rc = ds4_session_load_payload(s->session, fp, length, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        set_err(err, errlen, "failed to close session payload");
        return 1;
    }
    return rc;
}

int ds4_bridge_session_save_layer_payload(ds4_bridge_session *s, const char *path,
                                          uint32_t layer_start, uint32_t layer_end,
                                          char *err, size_t errlen)
{
    FILE *fp;
    int rc;

    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (!path || !path[0]) {
        set_err(err, errlen, "payload path is required");
        return 1;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        set_err(err, errlen, "failed to open session payload for write");
        return 1;
    }
    rc = ds4_session_save_layer_payload(s->session, fp, layer_start, layer_end,
                                        err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        set_err(err, errlen, "failed to close session payload");
        return 1;
    }
    return rc;
}

int ds4_bridge_session_load_layer_payload(ds4_bridge_session *s, const char *path,
                                          uint64_t payload_bytes,
                                          const int32_t *tokens, uint32_t n_tokens,
                                          uint32_t layer_start, uint32_t layer_end,
                                          char *err, size_t errlen)
{
    FILE *fp;
    int rc;

    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (!path || !path[0]) {
        set_err(err, errlen, "payload path is required");
        return 1;
    }
    if (!tokens && n_tokens != 0) {
        set_err(err, errlen, "tokens is NULL");
        return 1;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        set_err(err, errlen, "failed to open session payload for read");
        return 1;
    }
    rc = ds4_session_load_layer_payload(s->session, fp, payload_bytes, tokens,
                                        n_tokens, layer_start, layer_end,
                                        err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        set_err(err, errlen, "failed to close session payload");
        return 1;
    }
    return rc;
}

int ds4_bridge_snapshot_create(ds4_bridge_snapshot **out,
                               char *err, size_t errlen)
{
    ds4_bridge_snapshot *snap;

    if (out) *out = NULL;
    if (!out) {
        set_err(err, errlen, "out is NULL");
        return 1;
    }
    snap = calloc(1, sizeof(*snap));
    if (!snap) {
        set_err(err, errlen, "out of memory");
        return 1;
    }
    *out = snap;
    return 0;
}

void ds4_bridge_snapshot_free(ds4_bridge_snapshot *snap)
{
    if (!snap) return;
    ds4_session_snapshot_free(&snap->snapshot);
    free(snap);
}

uint64_t ds4_bridge_snapshot_len(const ds4_bridge_snapshot *snap)
{
    return snap ? snap->snapshot.len : 0;
}

int ds4_bridge_session_save_snapshot(ds4_bridge_session *s,
                                     ds4_bridge_snapshot *snap,
                                     char *err, size_t errlen)
{
    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (!snap) {
        set_err(err, errlen, "snapshot is NULL");
        return 1;
    }
    return ds4_session_save_snapshot(s->session, &snap->snapshot, err, errlen);
}

int ds4_bridge_session_load_snapshot(ds4_bridge_session *s,
                                     const ds4_bridge_snapshot *snap,
                                     char *err, size_t errlen)
{
    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (!snap) {
        set_err(err, errlen, "snapshot is NULL");
        return 1;
    }
    return ds4_session_load_snapshot(s->session, &snap->snapshot, err, errlen);
}

static int copy_tokens(ds4_tokens *tv, int32_t *out, int cap, int *n_out,
                       char *err, size_t errlen)
{
    if (!n_out) {
        ds4_tokens_free(tv);
        set_err(err, errlen, "n_out is NULL");
        return 1;
    }
    *n_out = tv->len;
    if (tv->len > cap) {
        ds4_tokens_free(tv);
        set_err(err, errlen, "token buffer too small");
        return 1;
    }
    if (tv->len > 0 && !out) {
        ds4_tokens_free(tv);
        set_err(err, errlen, "tokens is NULL");
        return 1;
    }
    if (tv->len > 0) {
        memcpy(out, tv->v, (size_t)tv->len * sizeof(int32_t));
    }
    ds4_tokens_free(tv);
    return 0;
}

int ds4_bridge_tokenize_text(ds4_bridge_model *m, const char *text,
                             int32_t *out, int cap, int *n_out,
                             char *err, size_t errlen)
{
    ds4_tokens tv;

    if (!m || !m->engine) {
        if (n_out) *n_out = 0;
        set_err(err, errlen, "model is NULL");
        return 1;
    }
    if (cap < 0) {
        if (n_out) *n_out = 0;
        set_err(err, errlen, "cap is negative");
        return 1;
    }
    memset(&tv, 0, sizeof(tv));
    ds4_tokenize_text(m->engine, text ? text : "", &tv);
    return copy_tokens(&tv, out, cap, n_out, err, errlen);
}

int ds4_bridge_tokenize_rendered_chat(ds4_bridge_model *m, const char *text,
                                      int32_t *out, int cap, int *n_out,
                                      char *err, size_t errlen)
{
    ds4_tokens tv;

    if (!m || !m->engine) {
        if (n_out) *n_out = 0;
        set_err(err, errlen, "model is NULL");
        return 1;
    }
    if (cap < 0) {
        if (n_out) *n_out = 0;
        set_err(err, errlen, "cap is negative");
        return 1;
    }
    memset(&tv, 0, sizeof(tv));
    ds4_tokenize_rendered_chat(m->engine, text ? text : "", &tv);
    return copy_tokens(&tv, out, cap, n_out, err, errlen);
}

int ds4_bridge_token_text(ds4_bridge_model *m, int32_t token,
                          char *out, size_t cap, size_t *n_out,
                          char *err, size_t errlen)
{
    size_t len = 0;
    char *piece;

    if (!m || !m->engine) {
        if (n_out) *n_out = 0;
        set_err(err, errlen, "model is NULL");
        return 1;
    }
    piece = ds4_token_text(m->engine, (int)token, &len);
    if (!piece) {
        if (n_out) *n_out = 0;
        if (out && cap > 0) out[0] = '\0';
        return 0;
    }
    if (n_out) *n_out = len;
    if (len >= cap) {
        free(piece);
        set_err(err, errlen, "token text buffer too small");
        return 1;
    }
    if (out) {
        memcpy(out, piece, len);
        out[len] = '\0';
    }
    free(piece);
    return 0;
}

int ds4_bridge_token_eos(ds4_bridge_model *m)
{
    if (!m || !m->engine) return -1;
    return ds4_token_eos(m->engine);
}

int ds4_bridge_token_is_stop(ds4_bridge_model *m, int32_t token)
{
    if (!m || !m->engine) return 0;
    return ds4_token_is_stop(m->engine, (int)token) ? 1 : 0;
}

int ds4_bridge_model_id(ds4_bridge_model *m)
{
    if (!m || !m->engine) return -1;
    return ds4_engine_model_id(m->engine);
}

int ds4_bridge_model_routed_quant_bits(ds4_bridge_model *m)
{
    if (!m || !m->engine) return 0;
    return ds4_engine_routed_quant_bits(m->engine);
}

int ds4_bridge_session_graph_fit_quote(ds4_bridge_model *m, int ctx_size,
                                       ds4_bridge_graph_fit_quote *q)
{
    ds4_session_graph_fit_quote native;
    int fits;

    if (!q) return 0;
    memset(q, 0, sizeof(*q));
    if (!m || !m->engine || ctx_size <= 0) return 0;
    fits = ds4_engine_session_graph_fit_quote(m->engine, ctx_size, &native);
    q->fits = native.fits;
    q->fail_open = native.fail_open;
    q->need_bytes = native.need_bytes;
    q->headroom_bytes = native.headroom_bytes;
    q->avail_bytes = native.avail_bytes;
    q->deficit_bytes = native.deficit_bytes;
    return fits;
}

int ds4_bridge_encode_chat_prompt(ds4_bridge_model *m, const char *system,
                                  const char *prompt, int think_mode,
                                  int32_t *out, int cap, int *n_out,
                                  char *err, size_t errlen)
{
    ds4_tokens tv;

    if (!m || !m->engine) {
        if (n_out) *n_out = 0;
        set_err(err, errlen, "model is NULL");
        return 1;
    }
    if (think_mode < DS4_THINK_NONE || think_mode > DS4_THINK_MAX) {
        if (n_out) *n_out = 0;
        set_err(err, errlen, "think_mode out of range");
        return 1;
    }
    if (cap < 0) {
        if (n_out) *n_out = 0;
        set_err(err, errlen, "cap is negative");
        return 1;
    }
    memset(&tv, 0, sizeof(tv));
    ds4_encode_chat_prompt(m->engine, system, prompt ? prompt : "",
                           (ds4_think_mode)think_mode, &tv);
    return copy_tokens(&tv, out, cap, n_out, err, errlen);
}

int ds4_bridge_session_top_logprobs(ds4_bridge_session *s,
                                    ds4_bridge_token_score *out, int k)
{
    enum { SCORE_CAP = 128 };   /* matches the C CLI clamp */
    ds4_token_score scores[SCORE_CAP];
    int n, i;

    if (!s || !s->session || !out || k <= 0) return -1;
    if (k > SCORE_CAP) k = SCORE_CAP;
    n = ds4_session_top_logprobs(s->session, scores, k);
    if (n < 0) return -1;
    if (n > k) n = k;

    for (i = 0; i < n; i++) {
        out[i].id = scores[i].id;
        out[i].logit = scores[i].logit;
        out[i].logprob = scores[i].logprob;
    }
    return n;
}

int ds4_bridge_session_copy_logits(ds4_bridge_session *s, float *out, int cap)
{
    if (!s || !s->session || !out || cap <= 0) return -1;
    return ds4_session_copy_logits(s->session, out, cap);
}

int ds4_bridge_session_output_head_bench(ds4_bridge_session *s,
                                         int iters, const char *path,
                                         char *err, size_t errlen)
{
    FILE *fp;
    int rc;

    if (!s || !s->session) {
        set_err(err, errlen, "session is NULL");
        return 1;
    }
    if (iters <= 0) {
        set_err(err, errlen, "output-head bench iters must be positive");
        return 1;
    }
    if (!path || !path[0]) {
        fp = stdout;
    } else {
        fp = fopen(path, "wb");
        if (!fp) {
            set_err(err, errlen, "failed to open output-head bench path");
            return 1;
        }
    }
    rc = ds4_session_output_head_bench(s->session, iters, fp, err, errlen);
    if (fp != stdout && fclose(fp) != 0 && rc == 0) {
        set_err(err, errlen, "failed to close output-head bench path");
        return 1;
    }
    return rc;
}

/* Declared in ds4_gpu.h; the bridge does not include that header. */
uint64_t ds4_gpu_substrate_outstanding(void);

/* Seqlock snapshot + last-stable cache, copied from ds4_server.c
 * mem_census_snapshot.  Do not include ds4_mem_census.h from Rust. */
typedef char ds4_bridge_memc_count_ok[
    (DS4_MEMC__COUNT == DS4_BRIDGE_MEMC_COUNT) ? 1 : -1];
typedef char ds4_bridge_memd_count_ok[
    (DS4_MEMD__COUNT == DS4_BRIDGE_MEMD_COUNT) ? 1 : -1];

static pthread_mutex_t mem_census_snap_mu = PTHREAD_MUTEX_INITIALIZER;
static ds4_bridge_mem_census mem_census_last_stable;
static int mem_census_last_stable_valid;
static uint64_t mem_census_torn_fallbacks;

static void copy_mem_cell(ds4_bridge_mem_cell *dst, const ds4_mem_cell *src)
{
    dst->requested = src->requested;
    dst->committed = src->committed;
    dst->freed_requested = src->freed_requested;
    dst->freed_committed = src->freed_committed;
    dst->alloc_calls = src->alloc_calls;
    dst->free_calls = src->free_calls;
}

int ds4_bridge_mem_census_snap(ds4_bridge_mem_census *out)
{
    int c, d, attempt;

    if (!out) return 1;
    memset(out, 0, sizeof(*out));
    for (attempt = 0; attempt < 8; attempt++) {
        const uint64_t began = ds4_gpu_mem_census_epoch_begin();
        if (began & 1u)
            continue;
        for (c = 0; c < DS4_BRIDGE_MEMC_COUNT; c++) {
            for (d = 0; d < DS4_BRIDGE_MEMD_COUNT; d++) {
                ds4_mem_cell cell;
                memset(&cell, 0, sizeof(cell));
                if (ds4_gpu_mem_census_read(c, d, &cell) != 0)
                    return 0;      /* backend keeps no census: supported=0 */
                copy_mem_cell(&out->cells[c][d], &cell);
            }
        }
        out->faults = ds4_gpu_mem_census_faults();
        if (ds4_gpu_mem_census_epoch_verify(began)) {
            out->epoch = began;
            out->supported = 1;
            pthread_mutex_lock(&mem_census_snap_mu);
            mem_census_last_stable = *out;
            mem_census_last_stable_valid = 1;
            out->torn_fallbacks = mem_census_torn_fallbacks;
            pthread_mutex_unlock(&mem_census_snap_mu);
            return 0;
        }
    }
    pthread_mutex_lock(&mem_census_snap_mu);
    mem_census_torn_fallbacks++;
    if (mem_census_last_stable_valid)
        *out = mem_census_last_stable;
    out->torn_fallbacks = mem_census_torn_fallbacks;
    pthread_mutex_unlock(&mem_census_snap_mu);
    return 0;
}

int ds4_bridge_mem_observe_snap(ds4_bridge_mem_observe *out)
{
    ds4_mem_observation o;

    if (!out) return 1;
    memset(&o, 0, sizeof(o));
    o.status = DS4_MEMOBS_UNSUPPORTED;
    (void)ds4_gpu_mem_observe(&o);
    if (o.status < 0 || o.status > DS4_MEMOBS_QUERY_ERROR)
        o.status = DS4_MEMOBS_QUERY_ERROR;
    if (o.source < 0 || o.source > DS4_MEMOBS_SRC_MEMINFO_AVAILABLE)
        o.source = DS4_MEMOBS_SRC_NONE;
    out->status = (int32_t)o.status;
    out->source = (int32_t)o.source;
    out->free_bytes = o.free_bytes;
    out->total_bytes = o.total_bytes;
    out->cuda_free_bytes = o.cuda_free_bytes;
    out->meminfo_avail_bytes = o.meminfo_avail_bytes;
    return 0;
}

uint64_t ds4_bridge_mem_substrate_outstanding(void)
{
    return ds4_gpu_substrate_outstanding();
}

int ds4_bridge_qwen_image_probe(const uint8_t *data, size_t data_len,
                                ds4_bridge_qwen_image_info *info,
                                char *err, size_t errlen)
{
    ds4_qwen_image_info native;
    int rc;

    if (!info) {
        set_err(err, errlen, "image info is NULL");
        return 1;
    }
    memset(&native, 0, sizeof(native));
    rc = ds4_qwen_image_probe(data, data_len, &native, err, errlen);
    if (rc != 0) return rc;
    info->source_width = native.source_width;
    info->source_height = native.source_height;
    info->resized_width = native.resized_width;
    info->resized_height = native.resized_height;
    info->grid_h = native.grid_h;
    info->grid_w = native.grid_w;
    info->token_count = native.token_count;
    return 0;
}

int ds4_bridge_qwen_image_pixel_hash(const uint8_t *data, size_t data_len,
                                     uint64_t *hash,
                                     char *err, size_t errlen)
{
    if (!hash) {
        set_err(err, errlen, "image hash is NULL");
        return 1;
    }
    return ds4_qwen_image_pixel_hash(data, data_len, hash, err, errlen);
}

struct ds4_bridge_batch_ctx {
    ds4_batch_ctx *ctx;
};

int ds4_bridge_batch_ctx_create_fit(ds4_bridge_model *m, int ctx_size,
                                    int max_seq, int max_total_tokens,
                                    ds4_bridge_batch_ctx **out,
                                    char *err, size_t errlen)
{
    ds4_batch_ctx *ctx = NULL;
    ds4_bridge_batch_ctx *c;
    int rc;

    if (out) *out = NULL;
    if (!out) {
        set_err(err, errlen, "out is NULL");
        return 1;
    }
    if (!m || !m->engine) {
        set_err(err, errlen, "model is NULL");
        return 1;
    }
    rc = ds4_batch_ctx_create_fit(m->engine, ctx_size, max_seq,
                                  max_total_tokens, &ctx, err, errlen);
    if (rc != 0 || !ctx) return rc != 0 ? rc : 1;
    c = calloc(1, sizeof(*c));
    if (!c) {
        ds4_batch_ctx_destroy(ctx);
        set_err(err, errlen, "out of memory");
        return 1;
    }
    c->ctx = ctx;
    *out = c;
    return 0;
}

void ds4_bridge_batch_ctx_destroy(ds4_bridge_batch_ctx *c)
{
    if (!c) return;
    ds4_batch_ctx_destroy(c->ctx);
    free(c);
}

int ds4_bridge_batch_ctx_max_seq(ds4_bridge_batch_ctx *c)
{
    if (!c || !c->ctx) return 0;
    return ds4_batch_ctx_max_seq(c->ctx);
}

int ds4_bridge_batch_ctx_raw_cap(ds4_bridge_batch_ctx *c)
{
    if (!c || !c->ctx) return 0;
    return ds4_batch_ctx_raw_cap(c->ctx);
}

int ds4_bridge_batch_ctx_seq_cap(ds4_bridge_batch_ctx *c)
{
    if (!c || !c->ctx) return 0;
    return ds4_batch_ctx_seq_cap(c->ctx);
}

int ds4_bridge_batch_ctx_supports_partial_reuse(ds4_bridge_batch_ctx *c)
{
    if (!c || !c->ctx) return 0;
    return ds4_batch_ctx_supports_partial_reuse(c->ctx);
}

uint64_t ds4_bridge_batch_ctx_trim_free(ds4_bridge_batch_ctx *c, uint64_t want_bytes)
{
    if (!c || !c->ctx || want_bytes == 0) return 0;
    return ds4_batch_ctx_trim_free(c->ctx, want_bytes);
}

int ds4_bridge_batch_ctx_generate_static(
        ds4_bridge_batch_ctx *c,
        const int32_t *const *prompt_tokens,
        const int32_t *prompt_lengths,
        const int32_t *max_new_tokens,
        const int32_t *eos_ids,
        int32_t n,
        int32_t *out_tokens,
        int32_t out_tokens_cap,
        int32_t *out_lengths,
        int32_t *out_finish,
        char *err, size_t errlen)
{
    ds4_tokens *prompts = NULL;
    ds4_batch_gen_result *results = NULL;
    size_t total = 0;
    int rc = 1;

    if (!c || !c->ctx) {
        set_err(err, errlen, "static batch ctx is NULL");
        return 1;
    }
    if (n <= 0 || n > ds4_batch_ctx_max_seq(c->ctx)) {
        set_err(err, errlen, "static batch request count is out of range");
        return 1;
    }
    if (out_lengths)
        memset(out_lengths, 0, (size_t)n * sizeof(*out_lengths));
    if (out_finish)
        memset(out_finish, 0, (size_t)n * sizeof(*out_finish));
    if (!prompt_tokens || !prompt_lengths || !max_new_tokens || !eos_ids) {
        set_err(err, errlen, "static batch request arrays are NULL");
        return 1;
    }
    if (!out_lengths || !out_finish || out_tokens_cap < 0 ||
        (out_tokens_cap > 0 && !out_tokens)) {
        set_err(err, errlen, "static batch output is invalid");
        return 1;
    }

    prompts = calloc((size_t)n, sizeof(*prompts));
    results = calloc((size_t)n, sizeof(*results));
    if (!prompts || !results) {
        set_err(err, errlen, "out of memory");
        goto done;
    }
    for (int32_t i = 0; i < n; i++) {
        if (prompt_lengths[i] > 0 && !prompt_tokens[i]) {
            set_err(err, errlen, "static batch prompt tokens are NULL");
            goto done;
        }
        prompts[i].v = (int *)prompt_tokens[i];
        prompts[i].len = prompt_lengths[i];
        prompts[i].cap = prompt_lengths[i];
    }

    rc = ds4_engine_batched_generate_ctx(c->ctx, prompts, n,
                                         max_new_tokens, eos_ids,
                                         results, err, errlen);
    if (rc != 0) goto done;

    for (int32_t i = 0; i < n; i++) {
        const int32_t len = results[i].n_tokens;
        const int32_t limit = max_new_tokens[i] > 0 ? max_new_tokens[i] : 1;
        if (len < 0 || (len > 0 && !results[i].tokens) ||
            len > limit ||
            (results[i].finish != 0 && results[i].finish != 1)) {
            set_err(err, errlen, "static batch native result is invalid");
            rc = 1;
            goto done;
        }
        if ((size_t)len > (size_t)out_tokens_cap - total) {
            set_err(err, errlen, "static batch output buffer is too small");
            rc = 1;
            goto done;
        }
        total += (size_t)len;
    }

    total = 0;
    for (int32_t i = 0; i < n; i++) {
        const int32_t len = results[i].n_tokens;
        if (len > 0) {
            memcpy(out_tokens + total, results[i].tokens,
                   (size_t)len * sizeof(*out_tokens));
        }
        out_lengths[i] = len;
        out_finish[i] = results[i].finish;
        total += (size_t)len;
    }
    rc = 0;

done:
    if (results) {
        for (int32_t i = 0; i < n; i++) free(results[i].tokens);
    }
    free(results);
    free(prompts);
    return rc;
}

int ds4_bridge_batch_ctx_bank_snapshot(ds4_bridge_batch_ctx *c, int32_t bank,
                                       int32_t *tokens, int32_t cap,
                                       int32_t *n_tokens,
                                       uint64_t *generation,
                                       char *err, size_t errlen)
{
    const int *committed = NULL;
    int n;

    if (n_tokens) *n_tokens = 0;
    if (generation) *generation = 0;
    if (!c || !c->ctx) {
        set_err(err, errlen, "batch ctx is NULL");
        return 1;
    }
    if (!n_tokens) {
        set_err(err, errlen, "bank snapshot output is NULL");
        return 1;
    }
    if (bank < 0 || bank >= ds4_batch_ctx_max_seq(c->ctx)) {
        set_err(err, errlen, "bank is out of range");
        return 1;
    }
    n = ds4_batch_ctx_bank_committed(c->ctx, bank, &committed);
    if (generation)
        *generation = ds4_batch_ctx_bank_generation(c->ctx, bank);
    *n_tokens = n;
    if (n <= 0) return 0;
    if (!committed || !tokens || cap < n) {
        set_err(err, errlen, "bank snapshot buffer is too small");
        return 1;
    }
    memcpy(tokens, committed, (size_t)n * sizeof(*tokens));
    return 0;
}

int ds4_bridge_batch_ctx_bank_save_payload(ds4_bridge_batch_ctx *c,
                                           int32_t bank, const char *path,
                                           char *err, size_t errlen)
{
    char *tmp;
    FILE *fp;
    int fd;
    int rc;
    size_t path_len;

    if (!c || !c->ctx) {
        set_err(err, errlen, "batch ctx is NULL");
        return 1;
    }
    if (bank < 0 || bank >= ds4_batch_ctx_max_seq(c->ctx)) {
        set_err(err, errlen, "bank is out of range");
        return 1;
    }
    if (!path || !path[0]) {
        set_err(err, errlen, "payload path is required");
        return 1;
    }
    path_len = strlen(path);
    if (path_len > SIZE_MAX - sizeof(".tmp.XXXXXX")) {
        set_err(err, errlen, "bank payload path is too long");
        return 1;
    }
    tmp = malloc(path_len + sizeof(".tmp.XXXXXX"));
    if (!tmp) {
        set_err(err, errlen, "failed to allocate bank payload path");
        return 1;
    }
    snprintf(tmp, path_len + sizeof(".tmp.XXXXXX"), "%s.tmp.XXXXXX", path);
    fd = mkstemp(tmp);
    if (fd < 0) {
        free(tmp);
        set_err(err, errlen, "failed to create bank payload staging file");
        return 1;
    }
    fp = fdopen(fd, "wb");
    if (!fp) {
        close(fd);
        unlink(tmp);
        free(tmp);
        set_err(err, errlen, "failed to open bank payload staging file");
        return 1;
    }
    rc = ds4_cont_bank_save_payload(c->ctx, (uint32_t)bank, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        set_err(err, errlen, "failed to close bank payload");
        rc = 1;
    }
    if (rc == 0 && rename(tmp, path) != 0) {
        set_err(err, errlen, "failed to install bank payload");
        rc = 1;
    }
    if (rc != 0) unlink(tmp);
    free(tmp);
    return rc;
}

int ds4_bridge_batch_ctx_bank_load_payload_range(ds4_bridge_batch_ctx *c,
                                                 int32_t bank,
                                                 const char *path,
                                                 uint64_t offset,
                                                 uint64_t length,
                                                 char *err, size_t errlen)
{
    FILE *fp;
    int rc;
    off_t sz;
    uint64_t file_bytes;

    if (!c || !c->ctx) {
        set_err(err, errlen, "batch ctx is NULL");
        return 1;
    }
    if (bank < 0 || bank >= ds4_batch_ctx_max_seq(c->ctx)) {
        set_err(err, errlen, "bank is out of range");
        return 1;
    }
    if (!path || !path[0]) {
        set_err(err, errlen, "payload path is required");
        return 1;
    }
    if (offset > UINT64_MAX - length) {
        set_err(err, errlen, "bank payload range overflows");
        return 1;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        set_err(err, errlen, "failed to open bank payload for read");
        return 1;
    }
    if (fseeko(fp, 0, SEEK_END) != 0 || (sz = ftello(fp)) < 0) {
        fclose(fp);
        set_err(err, errlen, "failed to measure bank payload");
        return 1;
    }
    file_bytes = (uint64_t)sz;
    if (offset > file_bytes || length > file_bytes - offset) {
        fclose(fp);
        set_err(err, errlen, "truncated bank payload range");
        return 1;
    }
    if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) {
        fclose(fp);
        set_err(err, errlen, "failed to seek bank payload range");
        return 1;
    }
    rc = ds4_cont_bank_restore_payload(c->ctx, (uint32_t)bank, fp, length,
                                       err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        set_err(err, errlen, "failed to close bank payload");
        return 1;
    }
    return rc;
}

/* Trampolines: the engine receives bridge-owned callbacks whose ud is this
 * frame struct, and every caller-visible callback is forwarded with the
 * caller's own ud.  Per-request callbacks are one shared set (the caller
 * dispatches per `user`), installed only when the caller set them. */
typedef struct {
    int (*admit)(void *ud, ds4_bridge_cont_request *req);
    int (*on_token)(void *ud, void *user, int32_t token);
    void (*on_done)(void *ud, void *user, const int32_t *tokens, int32_t n,
                    int32_t finish, const ds4_bridge_cont_stats *stats);
    int (*sample_override)(void *ud, void *user);
    int (*alive)(void *ud, void *user);
    int (*on_admitted)(void *ud, void *user, int n_cached, int n_computed,
                       int bank);
    ds4_batch_ctx *ctx;
    void *ud;
} cont_tramp;

static int cont_tramp_sample_override(void *ud, void *user)
{
    cont_tramp *t = ud;
    return t->sample_override ? t->sample_override(t->ud, user) : 0;
}

static int cont_tramp_alive(void *ud, void *user)
{
    cont_tramp *t = ud;
    return t->alive ? t->alive(t->ud, user) : 1;
}

static int cont_tramp_on_admitted(void *ud, void *user, int n_cached,
                                  int n_computed, int bank)
{
    cont_tramp *t = ud;
    return t->on_admitted ? t->on_admitted(t->ud, user, n_cached,
                                           n_computed, bank)
                          : 1;
}

static int cont_tramp_admit(void *ud, ds4_cont_request *req)
{
    cont_tramp *t = ud;
    ds4_bridge_cont_request br;

    memset(&br, 0, sizeof(br));
    if (!t->admit || t->admit(t->ud, &br) == 0) return 0;
    memset(req, 0, sizeof(*req));
    req->tokens = br.tokens;
    req->n = br.n;
    if (br.image_count > 4u) return 0;
    for (uint32_t i = 0; i < br.image_count; i++) {
        req->images[i].data = br.images[i].data;
        req->images[i].data_len = br.images[i].data_len;
        req->images[i].token_offset = br.images[i].token_offset;
        req->images[i].grid_h = br.images[i].grid_h;
        req->images[i].grid_w = br.images[i].grid_w;
    }
    req->image_count = br.image_count;
    req->max_new = br.max_new;
    req->eos = br.eos;
    req->user = br.user;
    req->temperature = br.temperature;
    req->top_k = br.top_k;
    req->top_p = br.top_p;
    req->min_p = br.min_p;
    req->seed = br.seed;
    /* The engine calls these with ITS ud (this frame), so the request
     * carries the shared trampolines; the per-call fns land in t->*. */
    t->sample_override = br.sample_override;
    t->alive = br.alive;
    t->on_admitted = br.on_admitted;
    if (br.sample_override) req->sample_override = cont_tramp_sample_override;
    if (br.alive) req->alive = cont_tramp_alive;
    if (br.on_admitted) req->on_admitted = cont_tramp_on_admitted;
    req->place_bank = br.place_bank;
    req->n_cached = br.n_cached;
    req->bank_used = br.bank_used;
    req->fork_bank = br.fork_bank;
    return 1;
}

static int cont_tramp_on_token(void *ud, void *user, int token)
{
    cont_tramp *t = ud;
    return t->on_token ? t->on_token(t->ud, user, (int32_t)token) : 1;
}

static void cont_tramp_on_done(void *ud, void *user, const int *tokens,
                               int n, int finish)
{
    cont_tramp *t = ud;
    ds4_bridge_cont_stats stats = {0};
    ds4_cont_seq_stats native;

    if (tokens && ds4_cont_last_done_stats(t->ctx, &native)) {
        stats.decode_ms = (native.done_sec - native.first_token_sec) * 1e3;
        stats.decode_tokens = native.decode_tokens;
        stats.decode_steps = native.decode_steps;
    }
    if (t->on_done) {
        t->on_done(t->ud, user, (const int32_t *)tokens, (int32_t)n,
                   (int32_t)finish, &stats);
    }
}

int ds4_bridge_continuous_generate(
    ds4_bridge_batch_ctx *c,
    int (*admit)(void *ud, ds4_bridge_cont_request *req),
    int (*on_token)(void *ud, void *user, int32_t token),
    void (*on_done)(void *ud, void *user, const int32_t *tokens, int32_t n,
                    int32_t finish, const ds4_bridge_cont_stats *stats),
    void *ud, char *err, size_t errlen)
{
    cont_tramp t;

    if (!c || !c->ctx) {
        set_err(err, errlen, "batch ctx is NULL");
        return 1;
    }
    if (!admit) {
        set_err(err, errlen, "admit is NULL");
        return 1;
    }
    memset(&t, 0, sizeof(t));
    t.admit = admit;
    t.on_token = on_token;
    t.on_done = on_done;
    t.ctx = c->ctx;
    t.ud = ud;
    return ds4_engine_continuous_generate(c->ctx, cont_tramp_admit,
                                          on_token ? cont_tramp_on_token : NULL,
                                          cont_tramp_on_done, &t, err, errlen);
}
