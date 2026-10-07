/* Checkpoint I/O exercises real files and byte-exact mutable graph state. */
#define DS4_NO_GPU
#include "../ds4.c"
#include "../ds4_glm53_compact.h"

struct ds4_gpu_tensor { uint8_t *data; uint64_t bytes; };
typedef struct {
    bool ready, expanded_diag, mtp_ready;
    uint32_t ctx_cap, cache_len, row_cap, pool_cap, mtp_len, mtp_min, mtp_trial_n;
    uint64_t state_bytes;
    ds4_gpu_tensor *state_pool;
    ds4_gpu_tensor *mtp_kv;
    ds4_gpu_tensor *layer_kv[DS4_MAX_LAYER], *layer_pool[DS4_MAX_LAYER];
} ds4_glm53_gpu_graph;

static int write_fail, sync_fail;
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0; }
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off, void *p, uint64_t n) {
    if (!t || off > t->bytes || n > t->bytes - off) { return 0; }
    memcpy(p, t->data + off, n);
    return 1;
}
int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *p, uint64_t n) {
    if (!t || off > t->bytes || n > t->bytes - off || (write_fail && --write_fail == 0)) { return 0; }
    memcpy(t->data + off, p, n);
    return 1;
}
int ds4_gpu_synchronize(void) { return !sync_fail; }
/* Small fixture spans fit in one production I/O chunk. The fake device
 * adapter still uses the real binary file codec and bounds every transfer. */
static int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *t,
        uint64_t off, uint64_t n, uint8_t *buf, size_t cap, char *err, size_t len) {
    if (n > cap || !ds4_gpu_tensor_read(t, off, buf, n)) { return 1; }
    return payload_write_bytes(fp, buf, n, err, len);
}
static int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *t,
        uint64_t off, uint64_t n, uint8_t *buf, size_t cap, uint64_t *left,
        char *err, size_t len) {
    if (n > cap || payload_read_bytes(fp, buf, n, left, err, len)) { return 1; }
    return !ds4_gpu_tensor_write(t, off, buf, n);
}
static void glm53_graph_layers(uint32_t *kda, uint32_t *dsa) {
    *kda = *dsa = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        if (ds4_glm53_layer_is_kda(il)) { (*kda)++; }
        else { (*dsa)++; }
    }
}
static bool glm53_graph_reset(ds4_glm53_gpu_graph *g) {
    memset(g->state_pool->data, 0, g->state_bytes);
    g->cache_len = 0;
    g->mtp_len = g->mtp_min = g->mtp_trial_n = 0;
    return true;
}
#include "../ds4_glm53_payload.inc"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "GLM payload FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
enum { CAP = 2056, POISON = 0xa5 };
typedef enum { FIXTURE_COMPACT, FIXTURE_EXPANDED } fixture_layout;
typedef enum { FIXTURE_MTP_OFF, FIXTURE_MTP_ON } fixture_mtp;

static ds4_gpu_tensor *tensor(uint64_t bytes) {
    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    CHECK(t);
    t->data = malloc(bytes);
    CHECK(t->data);
    t->bytes = bytes;
    for (uint64_t i = 0; i < bytes; i++) { t->data[i] = (uint8_t)(i * 29u + 7u); }
    return t;
}
static void init(ds4_glm53_gpu_graph *g, fixture_layout layout, fixture_mtp mtp) {
    const bool diag = layout == FIXTURE_EXPANDED;
    *g = (ds4_glm53_gpu_graph){.ready = true, .expanded_diag = diag,
        .ctx_cap = CAP, .row_cap = 128, .pool_cap = CAP / DS4_GLM53_POOL_SIZE,
        .state_bytes = 4096};
    g->state_pool = tensor(g->state_bytes);
    g->mtp_ready = mtp == FIXTURE_MTP_ON;
    if (g->mtp_ready) { g->mtp_kv = tensor(CAP * DS4_N_KV_LORA * sizeof(uint16_t)); }
    for (uint32_t il = 0; il < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; il++) {
        if (ds4_glm53_layer_is_kda(il)) { continue; }
        uint64_t row = diag ? DS4_N_HEAD * DS4_N_KEY_MLA * 2u : DS4_N_KV_LORA;
        g->layer_kv[il] = tensor(CAP * row * sizeof(uint16_t));
        if (!diag) { g->layer_pool[il] = tensor(g->pool_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t)); }
    }
}
static void release(ds4_glm53_gpu_graph *g) {
    free(g->state_pool->data); free(g->state_pool);
    if (g->mtp_kv) { free(g->mtp_kv->data); free(g->mtp_kv); }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        if (g->layer_kv[il]) { free(g->layer_kv[il]->data); free(g->layer_kv[il]); }
        if (g->layer_pool[il]) { free(g->layer_pool[il]->data); free(g->layer_pool[il]); }
    }
}
static void roundtrip(uint32_t n, fixture_layout layout, fixture_mtp mtp) {
    const bool diag = layout == FIXTURE_EXPANDED;
    ds4_glm53_gpu_graph src, dst;
    init(&src, layout, mtp); init(&dst, layout, mtp);
    src.cache_len = n;
    if (src.mtp_ready) { src.mtp_min = n > 2u ? n - 2u : 0u; src.mtp_len = n - 1u; }
    dst.row_cap = 1; /* Scratch width does not change the durable layout. */
    memset(dst.state_pool->data, POISON, dst.state_bytes);
    if (dst.mtp_ready) { memset(dst.mtp_kv->data, POISON, dst.mtp_kv->bytes); }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        if (dst.layer_kv[il]) { memset(dst.layer_kv[il]->data, POISON, dst.layer_kv[il]->bytes); }
        if (dst.layer_pool[il]) { memset(dst.layer_pool[il]->data, POISON, dst.layer_pool[il]->bytes); }
    }
    int *tokens = malloc(n * sizeof(*tokens));
    float logits[32], restored[32];
    for (uint32_t i = 0; i < n; i++) { tokens[i] = i % DS4_N_VOCAB; }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) { logits[i] = (float)i / 7; }
    FILE *fp = tmpfile(); CHECK(fp);
    char err[256] = {0};
    CHECK(glm53_payload_save(&src, tokens, n, logits, fp, err, sizeof(err)) == 0);
    uint64_t total = (uint64_t)ftell(fp);
    CHECK(total == glm53_payload_bytes(&src, n));
    if (src.mtp_ready) {
        src.mtp_trial_n = 1u;
        CHECK(glm53_payload_bytes(&src, n) == 0u);
        CHECK(glm53_payload_save(&src, tokens, n, logits, fp, err, sizeof(err)) != 0);
        CHECK((uint64_t)ftell(fp) == total);
        src.mtp_trial_n = 0u;
    }
    rewind(fp);
    uint64_t left = total;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (unsigned i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) { CHECK(!payload_read_u32(fp, &h[i], &left, err, sizeof(err))); }
    int *got = NULL;
    CHECK(!glm53_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)));
    CHECK(left == 0 && dst.cache_len == n);
    CHECK(!memcmp(tokens, got, n * sizeof(*tokens)) && !memcmp(logits, restored, sizeof(logits)));
    CHECK(!memcmp(src.state_pool->data, dst.state_pool->data, src.state_bytes));
    if (src.mtp_ready) {
        const uint64_t row = DS4_N_KV_LORA * sizeof(uint16_t);
        CHECK(dst.mtp_min == src.mtp_min && dst.mtp_len == src.mtp_len);
        CHECK(!memcmp(src.mtp_kv->data + src.mtp_min * row,
            dst.mtp_kv->data + dst.mtp_min * row, (src.mtp_len - src.mtp_min) * row));
        CHECK(dst.mtp_kv->data[src.mtp_len * row] == POISON);
    }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        if (!src.layer_kv[il]) { continue; }
        uint64_t row = diag ? DS4_N_HEAD * DS4_N_KEY_MLA * 2u : DS4_N_KV_LORA;
        uint64_t bytes = n * row * sizeof(uint16_t);
        CHECK(!memcmp(src.layer_kv[il]->data, dst.layer_kv[il]->data, bytes));
        CHECK(dst.layer_kv[il]->data[bytes] == POISON);
        if (!diag) {
            bytes = (n / DS4_GLM53_POOL_SIZE) * DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t);
            CHECK(!memcmp(src.layer_pool[il]->data, dst.layer_pool[il]->data, bytes));
            CHECK(dst.layer_pool[il]->data[bytes] == POISON);
        }
    }
    free(got);
    const long body = DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    const uint64_t body_bytes = total - (uint64_t)body;
    if (src.mtp_ready) {
        const long frontier = body + n * sizeof(uint32_t) + DS4_N_VOCAB * sizeof(float);
        const uint32_t invalid[][2] = {{n, n - 2u}, {n - 1u, n}, {0u, 1u}};
        for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
            fseek(fp, frontier, SEEK_SET);
            CHECK(!payload_write_u32(fp, invalid[i][0], err, sizeof(err)));
            CHECK(!payload_write_u32(fp, invalid[i][1], err, sizeof(err)) && !fflush(fp));
            fseek(fp, body, SEEK_SET); left = body_bytes; got = NULL;
            CHECK(glm53_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)) != 0);
            CHECK(!got && dst.cache_len == 0u && dst.mtp_len == 0u && dst.mtp_min == 0u);
        }
        fseek(fp, frontier, SEEK_SET);
        CHECK(!payload_write_u32(fp, src.mtp_len, err, sizeof(err)));
        CHECK(!payload_write_u32(fp, src.mtp_min, err, sizeof(err)) && !fflush(fp));
    }
    const unsigned altered[] = {0, 1, 4, 5, 6, 8, 9, 10, 11, 12};
    for (unsigned i = 0; i < sizeof(altered) / sizeof(altered[0]); i++) {
        uint32_t old = h[altered[i]]; h[altered[i]]++;
        fseek(fp, body, SEEK_SET); left = body_bytes; got = NULL;
        CHECK(glm53_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)) != 0 && !got);
        h[altered[i]] = old;
    }
    fseek(fp, body, SEEK_SET); left = body_bytes - 1; got = NULL;
    CHECK(glm53_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)) != 0 && !got);
    fseek(fp, body, SEEK_SET); left = body_bytes; write_fail = 1;
    CHECK(glm53_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)) != 0 && !got && dst.cache_len == 0);
    write_fail = 0;
    fseek(fp, body, SEEK_SET); left = body_bytes; sync_fail = 1;
    CHECK(glm53_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)) != 0 && !got && dst.cache_len == 0);
    sync_fail = 0;
    fseek(fp, body, SEEK_SET);
    CHECK(!payload_write_u32(fp, DS4_N_VOCAB, err, sizeof(err)) && !fflush(fp));
    fseek(fp, body, SEEK_SET); left = body_bytes;
    CHECK(glm53_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)) != 0 && !got);
    fseek(fp, body, SEEK_SET);
    CHECK(!payload_write_u32(fp, (uint32_t)tokens[0], err, sizeof(err)) && !fflush(fp));
    CHECK(!ftruncate(fileno(fp), (off_t)total - 1));
    fseek(fp, body, SEEK_SET); left = body_bytes;
    CHECK(glm53_payload_restore(&dst, fp, &left, h, &got, restored, err, sizeof(err)) != 0 && !got && dst.cache_len == 0);
    fclose(fp); free(tokens); release(&dst); release(&src);
}
int main(void) {
    g_ds4_shape = DS4_SHAPE_GLM53_FLASH;
    g_ds4_shape.n_layer = 9; g_ds4_shape.n_nextn_predict = 1;
    g_ds4_shape.n_head = 2; g_ds4_shape.n_key_mla = 16;
    g_ds4_shape.n_kv_lora = 32; g_ds4_shape.n_vocab = 32;
    roundtrip(3, FIXTURE_COMPACT, FIXTURE_MTP_OFF);
    roundtrip(5, FIXTURE_COMPACT, FIXTURE_MTP_OFF);
    roundtrip(2049, FIXTURE_COMPACT, FIXTURE_MTP_OFF);
    roundtrip(5, FIXTURE_EXPANDED, FIXTURE_MTP_OFF);
    roundtrip(5, FIXTURE_COMPACT, FIXTURE_MTP_ON);
    puts("GLM payload: byte-exact state/latent/pools, split tail, layout and failure gates PASS");
    return 0;
}
