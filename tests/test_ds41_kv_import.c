/* Execute the native checkpoint parser with bounded host tensors.
 * This checks import admission and stale-buffer reuse without a GPU/model. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DS4_MAX_LAYER 2u
#define DS4_N_LAYER 2u
#define DS4_N_SWA 8u
#define DS4_N_HEAD_DIM 8u
#define DS4_N_EMBD 8u
#define DS4_N_VOCAB 16u
#define DS4_V41_CKV_BYTES 288u
#define DS4_V41_IDXK_BYTES 72u

typedef struct {
    uint8_t *data;
    uint64_t bytes;
} ds4_gpu_tensor;

typedef struct {
    ds4_gpu_tensor *win[DS4_MAX_LAYER], *comp_kv[DS4_MAX_LAYER], *index_k[DS4_MAX_LAYER];
    ds4_gpu_tensor *cpre_kv[DS4_MAX_LAYER], *cpre_sc[DS4_MAX_LAYER], *mainh, *logits;
    uint32_t ng_src[DS4_MAX_LAYER], cpend[DS4_MAX_LAYER], win_from[DS4_MAX_LAYER];
    uint32_t ctx, n_past, mainh_n, last_logit_row;
    int64_t mainh_end;
    int dev_sample;
    uint8_t *himg;
} ds4_v41_state;

static struct {
    uint32_t ctx, n_mtp_target, mtp_towers;
    uint8_t is_kv_source[DS4_MAX_LAYER];
} g_ds4_v41 = {64u, 1u, 1u, {1u, 0u}};
static uint32_t g_ratio[DS4_MAX_LAYER] = {4u, 4u};
static uint32_t g_live_tensors;

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) { abort(); }
    return p;
}

static ds4_gpu_tensor *v41_alloc(uint64_t bytes, bool *ok) {
    ds4_gpu_tensor *t = xmalloc(sizeof *t);
    t->data = xmalloc((size_t)bytes);
    t->bytes = bytes;
    memset(t->data, 0x5a, (size_t)bytes);
    g_live_tensors++;
    *ok = true;
    return t;
}

static uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t->bytes; }
static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    free(t->data); free(t);
    g_live_tensors--;
}
static int ds4_gpu_synchronize(void) { return 1; }
static uint32_t ds4_layer_compress_ratio(uint32_t il) { return g_ratio[il]; }

static int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t doff,
                               const ds4_gpu_tensor *src, uint64_t soff, uint64_t bytes) {
    if (!dst || !src || doff > dst->bytes || bytes > dst->bytes - doff ||
        soff > src->bytes || bytes > src->bytes - soff) { return 0; }
    memmove(dst->data + doff, src->data + soff, (size_t)bytes);
    return 1;
}

static int ds4_gpu_tensor_read(const ds4_gpu_tensor *src, uint64_t off, void *dst, uint64_t bytes) {
    if (!src || off > src->bytes || bytes > src->bytes - off) { return 0; }
    memcpy(dst, src->data + off, (size_t)bytes);
    return 1;
}

static int ds4_gpu_tensor_write(ds4_gpu_tensor *dst, uint64_t off, const void *src, uint64_t bytes) {
    if (!dst || off > dst->bytes || bytes > dst->bytes - off) { return 0; }
    memcpy(dst->data + off, src, (size_t)bytes);
    return 1;
}

#include "../ds4_ds41_kvreuse.inc"

static int write_blob(const char *path, const uint8_t *blob, size_t bytes) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { return 0; }
    const int ok = fwrite(blob, 1, bytes, fp) == bytes;
    return fclose(fp) == 0 && ok;
}

static int rejected(const char *path, const uint8_t *blob, size_t bytes,
                     const int32_t *tokens, uint32_t n, const char *name) {
    if (!write_blob(path, blob, bytes)) { return 1; }
    const int accepted = v41_kvreuse_import(path, 0, bytes, tokens, n);
    printf("%s: %s\n", name, accepted ? "FAIL (accepted)" : "PASS");
    return accepted;
}

static int filled(const ds4_gpu_tensor *t, uint8_t value) {
    for (uint64_t i = 0; i < t->bytes; i++) {
        if (t->data[i] != value) { return 0; }
    }
    return 1;
}

/* Only `bytes` are the checkpoint's live span: an allocation can be larger. */
static int filled_prefix(const ds4_gpu_tensor *t, uint64_t bytes, uint8_t value) {
    if (bytes > t->bytes) { return 0; }
    for (uint64_t i = 0; i < bytes; i++) {
        if (t->data[i] != value) { return 0; }
    }
    return 1;
}

/* Damage a checkpoint's own copies, so a later restore that shows the saved
 * bytes proves they came from the file and not from the in-memory slot. */
static void fill_slot_spans(v41_kv_slot *s, uint8_t value) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor *spans[] = {s->win[il], s->ckv[il], s->idk[il], s->cpk[il], s->cps[il]};
        for (size_t i = 0; i < sizeof spans / sizeof spans[0]; i++) {
            if (spans[i]) { memset(spans[i]->data, value, (size_t)spans[i]->bytes); }
        }
    }
    if (s->mainh) { memset(s->mainh->data, value, (size_t)s->mainh->bytes); }
}

static int reject_counter(const char *path, const uint8_t *blob, size_t bytes,
                           size_t new_bytes, size_t offset, uint32_t value,
                           const int32_t *tokens, uint32_t n, const char *name) {
    /* Keep the payload length consistent with the altered counter. A size
     * mismatch must not mask the missing frontier validation. */
    uint8_t *mut = xmalloc(new_bytes);
    memset(mut, 0x5a, new_bytes);
    memcpy(mut, blob, bytes < new_bytes ? bytes : new_bytes);
    v41_kv_put_le32(mut + offset, value);
    const int failed = rejected(path, mut, new_bytes, tokens, n, name);
    free(mut);
    return failed;
}

static int check_result(const char *name, int ok) {
    printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    return !ok;
}

/* YoungAi core_v41_kvreuse.c:91-131 compares image signatures at each
 * checkpoint position and restores image history for the engram lookback. */
static int vision_cases(ds4_v41_state *st, const char *path) {
    enum { PREFIX = 9, EXTENDED = 12 };
    const int32_t prompt[EXTENDED] = {1, 2, 3, 7, 7, 6, 7, 7, 9, 10, 7, 7};
    const uint64_t sig[EXTENDED] = {
        0, 0, 0, UINT64_C(0x0123456789abcdef), UINT64_C(0x0123456789abcdef),
        0, UINT64_C(0xfedcba9876543210), UINT64_C(0xfedcba9876543210), 0,
        0, UINT64_C(0x1122334455667788), UINT64_C(0x1122334455667788),
    };
    uint64_t changed[EXTENDED], zeros[EXTENDED] = {0};
    memcpy(changed, sig, sizeof changed);
    changed[3] = changed[4] = UINT64_C(0x8877665544332211);
    int failures = 0;

    v41_kvreuse_save(st, prompt, PREFIX, sig);
    memset(st->himg, 0xa5, st->ctx);
    int same = v41_kvreuse_peek(prompt, PREFIX, sig) == PREFIX &&
        v41_kvreuse_restore(st, prompt, PREFIX, sig) == PREFIX;
    for (uint32_t i = 0; same && i < PREFIX; i++) {
        same = st->himg[i] == (sig[i] != 0);
    }
    failures += check_result("same image restores image history", same && st->himg[PREFIX] == 0xa5);

    /* Matching placeholder IDs do not identify image content. A miss must
     * leave the fresh state's buffers and history available for cold fill. */
    st->n_past = 0;
    memset(st->himg, 0xa5, st->ctx);
    memset(st->logits->data, 0x3c, (size_t)st->logits->bytes);
    int different = v41_kvreuse_peek(prompt, PREFIX, changed) == 0u &&
        v41_kvreuse_restore(st, prompt, PREFIX, changed) == 0u &&
        st->n_past == 0u && g_v41_kvreuse_last_reused == 0 &&
        filled(st->logits, 0x3c);
    for (uint32_t i = 0; different && i < st->ctx; i++) {
        different = st->himg[i] == 0xa5;
    }
    failures += check_result("changed image with same tokens misses", different);
    failures += check_result("image checkpoint rejects text signatures",
        v41_kvreuse_peek(prompt, PREFIX, NULL) == 0u &&
        v41_kvreuse_restore(st, prompt, PREFIX, NULL) == 0u &&
        v41_kvreuse_peek(prompt, PREFIX, zeros) == 0u &&
        v41_kvreuse_restore(st, prompt, PREFIX, zeros) == 0u);

    /* A later image has no influence on the saved causal prefix. */
    failures += check_result("appended image retains matching prefix",
        v41_kvreuse_peek(prompt, EXTENDED, sig) == PREFIX &&
        v41_kvreuse_restore(st, prompt, EXTENDED, sig) == PREFIX);
    failures += check_result("changed image rejects extended prompt",
        v41_kvreuse_peek(prompt, EXTENDED, changed) == 0u &&
        v41_kvreuse_restore(st, prompt, EXTENDED, changed) == 0u);

    v41_kvreuse_save(st, prompt, PREFIX, NULL);
    uint64_t appended[EXTENDED] = {0};
    appended[10] = appended[11] = sig[10];
    memset(st->himg, 0xa5, st->ctx);
    int text = v41_kvreuse_peek(prompt, EXTENDED, appended) == PREFIX &&
        v41_kvreuse_restore(st, prompt, EXTENDED, appended) == PREFIX;
    for (uint32_t i = 0; text && i < PREFIX; i++) {
        text = st->himg[i] == 0;
    }
    failures += check_result("text prefix before first image reuses", text);
    failures += check_result("null and zero signatures are equivalent",
        v41_kvreuse_peek(prompt, PREFIX, zeros) == PREFIX &&
        v41_kvreuse_restore(st, prompt, PREFIX, zeros) == PREFIX);
    failures += check_result("image cannot replace saved text prefix",
        v41_kvreuse_peek(prompt, PREFIX, sig) == 0u &&
        v41_kvreuse_restore(st, prompt, PREFIX, sig) == 0u);

    v41_kvreuse_save(st, prompt, PREFIX, sig);
    struct stat sb;
    if (!v41_kvreuse_export(path, V41_KV_CLASS_PROMPT) || stat(path, &sb) != 0) {
        return failures + check_result("image checkpoint export", 0);
    }
    const uint64_t bytes = (uint64_t)sb.st_size;
    const uint64_t sig_bytes = (uint64_t)PREFIX * sizeof(uint64_t);
    FILE *fp = fopen(path, "rb");
    uint8_t encoded[PREFIX * sizeof(uint64_t)];
    int encoded_ok = fp && bytes >= sig_bytes &&
        fseeko(fp, (off_t)(bytes - sig_bytes), SEEK_SET) == 0 &&
        fread(encoded, 1, sizeof encoded, fp) == sizeof encoded;
    if (fp) { fclose(fp); }
    for (uint32_t i = 0; encoded_ok && i < PREFIX; i++) {
        for (uint32_t byte = 0; byte < sizeof(uint64_t); byte++) {
            if (encoded[i * sizeof(uint64_t) + byte] != (uint8_t)(sig[i] >> (byte * 8u))) {
                encoded_ok = 0;
            }
        }
    }
    failures += check_result("disk signatures use little endian", encoded_ok);

    /* Overwrite the resident identity before import so a passing check
     * proves the disk payload restored signatures, not stale host memory. */
    v41_kvreuse_save(st, prompt, PREFIX, changed);
    memset(st->himg, 0xa5, st->ctx);
    int loaded = v41_kvreuse_import(path, 0, bytes, prompt, PREFIX) &&
        v41_kvreuse_peek(prompt, PREFIX, sig) == PREFIX &&
        v41_kvreuse_peek(prompt, PREFIX, changed) == 0u &&
        v41_kvreuse_restore(st, prompt, PREFIX, sig) == PREFIX;
    for (uint32_t i = 0; loaded && i < PREFIX; i++) {
        loaded = st->himg[i] == (sig[i] != 0);
    }
    failures += check_result("disk round trip restores image identity", loaded);
    failures += check_result("disk prefix allows a later image",
        v41_kvreuse_peek(prompt, EXTENDED, sig) == PREFIX &&
        v41_kvreuse_restore(st, prompt, EXTENDED, sig) == PREFIX);

    v41_kvreuse_drop();
    failures += check_result("legacy payload without signatures misses",
        !v41_kvreuse_import(path, 0, bytes - sig_bytes, prompt, PREFIX) &&
        v41_kvreuse_peek(prompt, PREFIX, sig) == 0u);
    failures += check_result("truncated signature payload misses",
        !v41_kvreuse_import(path, 0, bytes - 1u, prompt, PREFIX) &&
        v41_kvreuse_peek(prompt, PREFIX, sig) == 0u);
    return failures;
}

/* The port addition: the post-generation checkpoint. Its key is the prompt
 * followed by the ids the run committed, so it serves a prompt that continues
 * that position, never one of equal length (it carries no logits row), and a
 * miss on it must leave the prompt-end checkpoint's own decision untouched. */
static int turn_cases(ds4_v41_state *st) {
    enum { PROMPT = 6, COMMITTED = 4, KEY = PROMPT + COMMITTED, NEXT = KEY + 2 };
    const int32_t prompt[NEXT] = {1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 40, 41};
    const int32_t *committed = prompt + PROMPT;   /* what the state committed after the prompt */
    int32_t mismatch[NEXT];
    uint64_t image[NEXT] = {0}, changed[NEXT], claimed[NEXT];
    v41_turn_ids key = {NULL, 0, 0};
    bool ok = true;
    const uint64_t rowb = DS4_N_HEAD_DIM * 4u;
    struct carry { ds4_gpu_tensor *t; uint64_t bytes; } spans[8];
    size_t nspans = 0;

    memcpy(mismatch, prompt, sizeof prompt);
    mismatch[PROMPT + 2] = 99;   /* one committed id differs */
    image[2] = UINT64_C(0x0123456789abcdef);
    memcpy(changed, image, sizeof image);
    changed[2] = UINT64_C(0x8877665544332211);
    memcpy(claimed, image, sizeof image);
    claimed[PROMPT + 1] = image[2];   /* a sampled position never carries an image */
    /* The span set a run holds at the end: main() leaves mainh NULL after the
     * no-MTP case and a ratio-one source's counters behind, so set the shapes
     * this run had.  Each span carries its live size -- an allocation can hold
     * an older checkpoint's tail, and only the live span is copied. */
    g_ratio[0] = 4u;
    st->ng_src[0] = 2u; st->cpend[0] = 1u;
    st->mainh = v41_alloc(DS4_N_SWA * DS4_N_EMBD * 4u, &ok);
    st->mainh_end = KEY - 1u; st->mainh_n = 1u;
    spans[nspans++] = (struct carry){st->win[0], DS4_N_SWA * rowb};
    spans[nspans++] = (struct carry){st->win[1], DS4_N_SWA * rowb};
    spans[nspans++] = (struct carry){st->comp_kv[0], (uint64_t)st->ng_src[0] * DS4_V41_CKV_BYTES};
    spans[nspans++] = (struct carry){st->index_k[0], (uint64_t)st->ng_src[0] * DS4_V41_IDXK_BYTES};
    spans[nspans++] = (struct carry){st->cpre_kv[0], rowb};
    spans[nspans++] = (struct carry){st->cpre_sc[0], rowb};
    spans[nspans++] = (struct carry){st->mainh, DS4_N_SWA * DS4_N_EMBD * 4u};
    for (size_t i = 0; i < nspans; i++) { memset(spans[i].t->data, 0x5a, (size_t)spans[i].t->bytes); }

    int failures = 0;
    /* The buffer the run fills, in the state's own order: the prompt then what
     * the run committed. */
    v41_turn_ids_add(&key, prompt, PROMPT);
    v41_turn_ids_add(&key, committed, COMMITTED);
    int assembled = key.n == KEY && key.cap >= KEY &&
        !memcmp(key.ids, prompt, sizeof prompt[0] * KEY);
    v41_turn_ids_free(&key);
    failures += check_result("the committed-id buffer keeps the state's order",
        assembled && !key.ids && !key.n && !key.cap);
    v41_kvreuse_drop();
    v41_kvreuse_save_turn(st, prompt, PROMPT, committed, COMMITTED, NULL);
    failures += check_result("turn checkpoint serves the continuing prompt",
        v41_kvreuse_peek(prompt, NEXT, NULL) == KEY);
    for (size_t i = 0; i < nspans; i++) { memset(spans[i].t->data, 0x3c, (size_t)spans[i].t->bytes); }
    memset(st->logits->data, 0x3c, (size_t)st->logits->bytes);
    st->n_past = 0;
    int restored = v41_kvreuse_restore(st, prompt, NEXT, NULL) == KEY &&
        st->n_past == KEY && g_v41_kvreuse_last_reused == (int)KEY &&
        st->ng_src[0] == 2u && st->cpend[0] == 1u && st->win_from[1] == 8u &&
        st->mainh_n == 1u && st->mainh_end == (int64_t)KEY - 1;
    for (size_t i = 0; restored && i < nspans; i++) { restored = filled_prefix(spans[i].t, spans[i].bytes, 0x5a); }
    failures += check_result("turn checkpoint restores the state's spans", restored);
    /* Its position is never the prompt's end, so it carries no logits row. */
    failures += check_result("turn checkpoint writes no logits row", filled(st->logits, 0x3c));

    /* No prompt-end checkpoint here: the miss must be a cold start, not a
     * half-copied state. */
    v41_kvreuse_drop();
    v41_kvreuse_save_turn(st, prompt, PROMPT, committed, COMMITTED, NULL);
    st->n_past = 0;
    memset(st->himg, 0xa5, st->ctx);
    failures += check_result("committed id mismatch misses",
        v41_kvreuse_peek(mismatch, NEXT, NULL) == 0u &&
        v41_kvreuse_restore(st, mismatch, NEXT, NULL) == 0u && st->n_past == 0u);

    /* The same run's prompt-end checkpoint still serves the prompt the turn
     * checkpoint refused. */
    v41_kvreuse_save(st, prompt, PROMPT, NULL);
    failures += check_result("a miss on the turn checkpoint keeps the prompt end",
        v41_kvreuse_peek(mismatch, NEXT, NULL) == PROMPT &&
        v41_kvreuse_restore(st, mismatch, NEXT, NULL) == PROMPT && st->n_past == PROMPT);

    v41_kvreuse_drop();
    v41_kvreuse_save_turn(st, prompt, PROMPT, committed, COMMITTED, NULL);
    st->n_past = 0;
    failures += check_result("turn checkpoint never serves an equal length prompt",
        v41_kvreuse_peek(prompt, KEY, NULL) == 0u &&
        v41_kvreuse_restore(st, prompt, KEY, NULL) == 0u && st->n_past == 0u);
    failures += check_result("turn checkpoint never serves a shorter prompt",
        v41_kvreuse_peek(prompt, PROMPT, NULL) == 0u &&
        v41_kvreuse_restore(st, prompt, PROMPT, NULL) == 0u);

    v41_kvreuse_save_turn(st, prompt, PROMPT, committed, COMMITTED, image);
    failures += check_result("turn checkpoint restores its image history",
        v41_kvreuse_peek(prompt, NEXT, image) == KEY &&
        v41_kvreuse_restore(st, prompt, NEXT, image) == KEY && st->himg[2] != 0);
    st->n_past = 0;
    memset(st->himg, 0xa5, st->ctx);
    int intolerant = v41_kvreuse_peek(prompt, NEXT, changed) == 0u &&
        v41_kvreuse_restore(st, prompt, NEXT, changed) == 0u;
    for (uint32_t i = 0; intolerant && i < st->ctx; i++) { intolerant = st->himg[i] == 0xa5; }
    failures += check_result("changed image misses the turn checkpoint", intolerant);
    failures += check_result("a sampled position carries no image identity",
        v41_kvreuse_peek(prompt, NEXT, claimed) == 0u &&
        v41_kvreuse_restore(st, prompt, NEXT, claimed) == 0u);

    st->dev_sample = 1;
    st->n_past = 0;
    failures += check_result("sampled turn checkpoint still reuses its prefix",
        v41_kvreuse_restore(st, prompt, NEXT, image) == KEY);
    st->dev_sample = 0;

    v41_kvreuse_drop();
    failures += check_result("drop invalidates the turn checkpoint",
        v41_kvreuse_peek(prompt, NEXT, image) == 0u &&
        v41_kvreuse_restore(st, prompt, NEXT, image) == 0u);
    ds4_gpu_tensor_free(st->mainh);
    st->mainh = NULL;
    return failures;
}

/* The turn class on disk.  The payload carries T's span set and no logits
 * row, the meta names its class, and the import lands it in that class's own
 * slot: the store selects a candidate by token prefix alone and cannot know
 * which shape it holds, so a class this build does not know, or one whose
 * shape does not fit the size fence, is refused.  The store's trailer carries
 * the mount digest, so a stale amplifier's payload never reaches this import
 * (the Rust gate); here the amplifier change must drop what was loaded. */
static int disk_turn_cases(ds4_v41_state *st, const char *path) {
    enum { PROMPT = 6, COMMITTED = 3, KEY = PROMPT + COMMITTED, NEXT = KEY + 2 };
    const int32_t prompt[NEXT] = {1, 2, 3, 4, 5, 6, 20, 21, 22, 40, 41};
    const int32_t *committed = prompt + PROMPT;
    uint64_t image[NEXT] = {0}, changed[NEXT], claimed[NEXT];
    struct carry { ds4_gpu_tensor *t; uint64_t bytes; } spans[7];
    size_t nspans = 0;
    const uint64_t rowb = DS4_N_HEAD_DIM * 4u;
    /* The meta's trailing four bytes: the class, then three reserved. */
    const size_t class_off =
        (size_t)DS4_N_LAYER * V41_KVREUSE_META_ROW_BYTES + V41_KVREUSE_META_TAIL_BYTES;
    struct stat sb;
    bool ok = true;
    int failures = 0;

    image[2] = UINT64_C(0x0123456789abcdef);
    memcpy(changed, image, sizeof image);
    changed[2] = UINT64_C(0x8877665544332211);
    memcpy(claimed, image, sizeof image);
    claimed[PROMPT + 1] = image[2];   /* a sampled position never carries an image */

    /* The state a run leaves at its turn end (turn_cases' shape).  KEY is
     * divisible by the ratio the way the source counters expect (9/4 = 2
     * groups, 9%4 = 1 remainder row). */
    g_ratio[0] = 4u;
    g_ds4_v41.mtp_towers = 1u; g_ds4_v41.n_mtp_target = 1u;
    st->ng_src[0] = 2u; st->cpend[0] = 1u;
    st->win_from[0] = 0u; st->win_from[1] = 8u;
    st->mainh = v41_alloc(DS4_N_SWA * DS4_N_EMBD * 4u, &ok);
    st->mainh_end = KEY - 1u; st->mainh_n = 1u;
    spans[nspans++] = (struct carry){st->win[0], DS4_N_SWA * rowb};
    spans[nspans++] = (struct carry){st->win[1], DS4_N_SWA * rowb};
    spans[nspans++] = (struct carry){st->comp_kv[0], (uint64_t)st->ng_src[0] * DS4_V41_CKV_BYTES};
    spans[nspans++] = (struct carry){st->index_k[0], (uint64_t)st->ng_src[0] * DS4_V41_IDXK_BYTES};
    spans[nspans++] = (struct carry){st->cpre_kv[0], (uint64_t)st->cpend[0] * rowb};
    spans[nspans++] = (struct carry){st->cpre_sc[0], (uint64_t)st->cpend[0] * rowb};
    spans[nspans++] = (struct carry){st->mainh, DS4_N_SWA * DS4_N_EMBD * 4u};
    for (size_t i = 0; i < nspans; i++) { memset(spans[i].t->data, 0x5a, (size_t)spans[i].t->bytes); }

    failures += check_result("the class ABI is the one Rust names",
        V41_KV_CLASS_PROMPT == 0 && V41_KV_CLASS_TURN == 1);
    v41_kvreuse_drop();
    v41_kvreuse_save_turn(st, prompt, PROMPT, committed, COMMITTED, image);

    /* The key the host stores the entry under: the prompt then the committed
     * ids, and a cap below the count writes nothing. */
    int32_t key[NEXT] = {0};
    const uint32_t count = v41_kvreuse_turn_key(key, NEXT);
    int key_ok = count == KEY && !memcmp(key, prompt, sizeof prompt[0] * KEY) &&
        key[KEY] == 0 && v41_kvreuse_turn_key(NULL, 0) == KEY;
    memset(key, 0x7f, sizeof key);
    key_ok = key_ok && v41_kvreuse_turn_key(key, KEY - 1u) == KEY && key[0] == 0x7f7f7f7f;
    failures += check_result("the turn key is the prompt plus the committed ids", key_ok);

    if (!v41_kvreuse_export(path, V41_KV_CLASS_TURN) || stat(path, &sb) != 0) {
        return failures + check_result("turn checkpoint export", 0);
    }
    const uint64_t bytes = (uint64_t)sb.st_size;

    /* The meta names the class; the round trip below proves the shape the
     * export left (the import computes it from the class, so a stray logits
     * row would fail there). */
    uint8_t cls_raw[4] = {0xff, 0xff, 0xff, 0xff};
    FILE *cf = fopen(path, "rb");
    int cls_ok = cf && fseeko(cf, (off_t)class_off, SEEK_SET) == 0 &&
        fread(cls_raw, 1, sizeof cls_raw, cf) == sizeof cls_raw;
    if (cf) { fclose(cf); }
    cls_ok = cls_ok && cls_raw[0] == V41_KV_CLASS_TURN &&
        !cls_raw[1] && !cls_raw[2] && !cls_raw[3];
    failures += check_result("the payload meta names the turn class", cls_ok);

    /* Damage both copies, so a passing restore proves the bytes came back
     * from the file. */
    fill_slot_spans(&T, 0x3c);
    for (size_t i = 0; i < nspans; i++) { memset(spans[i].t->data, 0x3c, (size_t)spans[i].t->bytes); }
    memset(st->logits->data, 0x3c, (size_t)st->logits->bytes);
    st->n_past = 0;
    int loaded = v41_kvreuse_import(path, 0, bytes, prompt, KEY) &&
        v41_kvreuse_peek(prompt, NEXT, image) == KEY &&
        v41_kvreuse_restore(st, prompt, NEXT, image) == KEY && st->n_past == KEY;
    for (size_t i = 0; loaded && i < nspans; i++) {
        loaded = filled_prefix(spans[i].t, spans[i].bytes, 0x5a);
    }
    loaded = loaded && filled(st->logits, 0x3c);
    failures += check_result("turn payload round-trips from disk", loaded);

    /* The payload's own class is what decides the slot: the turn's KV must
     * not have landed in the prompt slot, whose restore may serve a logits
     * row it never wrote. */
    failures += check_result("the payload landed in the turn slot",
        T.valid && T.n == KEY && !K.valid);

    /* The payload's own signature tail is what the restore compares. */
    failures += check_result("a changed image misses the turn payload",
        v41_kvreuse_peek(prompt, NEXT, changed) == 0u &&
        v41_kvreuse_restore(st, prompt, NEXT, changed) == 0u);
    failures += check_result("an image at a sampled position misses it",
        v41_kvreuse_peek(prompt, NEXT, claimed) == 0u);

    /* The amplifier change is the native drop; the store's trailer carries
     * the digest that refuses a stale payload before it reaches this path. */
    const int before_mount = v41_kvreuse_peek(prompt, NEXT, image) == KEY;
    v41_kvreuse_note_mount(1, UINT64_C(0x5555555555555555));
    failures += check_result("an amplifier change drops the loaded turn checkpoint",
        before_mount && v41_kvreuse_peek(prompt, NEXT, image) == 0u);

    /* The class byte is read from the payload and validated: a class this
     * build does not know is refused, and a payload whose class byte claims
     * the prompt class cannot pass the size fence (its shape is a turn's, so
     * it carries no logits row). */
    uint8_t *raw = xmalloc((size_t)bytes);
    FILE *rf = fopen(path, "rb");
    const int have_raw = rf && fread(raw, 1, (size_t)bytes, rf) == (size_t)bytes;
    if (rf) { fclose(rf); }
    const uint8_t others[2] = {7u, V41_KV_CLASS_PROMPT};
    for (int v = 0; v < 2; v++) {
        raw[class_off] = others[v];
        v41_kvreuse_drop();
        const int refused = !have_raw || !write_blob(path, raw, (size_t)bytes) ||
            !v41_kvreuse_import(path, 0, bytes, prompt, KEY);
        failures += check_result(v ? "a class byte the shape does not match is refused"
                                   : "an unknown class byte is refused",
            refused && v41_kvreuse_peek(prompt, NEXT, image) == 0u && !K.valid && !T.valid);
    }
    free(raw);
    failures += check_result("no turn checkpoint answers no key",
        v41_kvreuse_turn_key(NULL, 0) == 0u);
    ds4_gpu_tensor_free(st->mainh);
    st->mainh = NULL;
    return failures;
}

int main(void) {
    const uint32_t n = 9u;
    const int32_t tokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    ds4_v41_state st = {0};
    bool ok = true;
    const uint64_t rowb = DS4_N_HEAD_DIM * 4u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        st.win[il] = v41_alloc(DS4_N_SWA * rowb, &ok);
    }
    st.comp_kv[0] = v41_alloc(2u * DS4_V41_CKV_BYTES, &ok);
    st.index_k[0] = v41_alloc(2u * DS4_V41_IDXK_BYTES, &ok);
    st.cpre_kv[0] = v41_alloc(rowb, &ok);
    st.cpre_sc[0] = v41_alloc(rowb, &ok);
    st.mainh = v41_alloc(DS4_N_SWA * DS4_N_EMBD * 4u, &ok);
    st.logits = v41_alloc(DS4_N_VOCAB * 4u, &ok);
    st.ng_src[0] = 2u; st.cpend[0] = 1u; st.ctx = g_ds4_v41.ctx;
    st.himg = xmalloc(st.ctx);
    memset(st.himg, 0, st.ctx);
    st.win_from[1] = 8u;   /* a CED decoder layer retains a nonzero window floor */
    st.mainh_end = n - 1u; st.mainh_n = DS4_N_SWA;
    v41_kvreuse_save(&st, tokens, n, NULL);

    char path[] = "/tmp/ds41-kv-import-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) { return 1; }
    close(fd);
    if (!v41_kvreuse_export(path, V41_KV_CLASS_PROMPT)) { unlink(path); return 1; }
    FILE *fp = fopen(path, "rb");
    if (!fp || fseeko(fp, 0, SEEK_END) != 0) { unlink(path); return 1; }
    const size_t bytes = (size_t)ftello(fp);
    uint8_t *blob = xmalloc(bytes), *mut = xmalloc(bytes);
    rewind(fp);
    if (fread(blob, 1, bytes, fp) != bytes) { fclose(fp); unlink(path); return 1; }
    fclose(fp);

    ds4_gpu_tensor *spans[] = {st.win[0], st.win[1], st.comp_kv[0], st.index_k[0],
                              st.cpre_kv[0], st.cpre_sc[0], st.mainh, st.logits};
    for (size_t i = 0; i < sizeof spans / sizeof spans[0]; i++) {
        memset(spans[i]->data, 0x3c, (size_t)spans[i]->bytes);
    }
    st.ng_src[0] = 0; st.cpend[0] = 0; st.win_from[1] = 0; st.mainh_n = 0; st.mainh_end = -1;
    int failures = 0;
    if (!v41_kvreuse_import(path, 0, bytes, tokens, n) ||
        v41_kvreuse_restore(&st, tokens, n, NULL) != n) {
        fprintf(stderr, "valid round trip: FAIL\n");
        failures++;
    } else {
        puts("valid round trip: PASS");
    }
    int restored = st.ng_src[0] == 2u && st.cpend[0] == 1u && st.win_from[1] == 8u && st.n_past == n &&
        st.mainh_n == DS4_N_SWA && st.mainh_end == (int64_t)n - 1;
    for (size_t i = 0; restored && i < sizeof spans / sizeof spans[0]; i++) {
        restored = filled(spans[i], 0x5a);
    }
    printf("restored state and bytes: %s\n", restored ? "PASS" : "FAIL");
    failures += !restored;

    /* The checkpoint has logits but no sampler position. A full hit must
     * prefill again; a suffix still writes its own positions before sampling. */
    st.dev_sample = 1;
    st.n_past = 0;
    memset(st.logits->data, 0x3c, (size_t)st.logits->bytes);
    int sample_cold = v41_kvreuse_restore(&st, tokens, n, NULL) == 0u &&
        st.n_past == 0u && g_v41_kvreuse_last_reused == 0 && filled(st.logits, 0x3c);
    printf("sampled full hit is cold: %s\n", sample_cold ? "PASS" : "FAIL");
    failures += !sample_cold;
    const int32_t extended[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int sample_suffix = v41_kvreuse_restore(&st, extended, n + 1u, NULL) == n;
    printf("sampled suffix retains prefix: %s\n", sample_suffix ? "PASS" : "FAIL");
    failures += !sample_suffix;
    st.dev_sample = 0;
    const size_t tail = DS4_N_LAYER * V41_KVREUSE_META_ROW_BYTES;
    memcpy(mut, blob, bytes);
    v41_kv_put_le64(mut + tail, UINT64_MAX);
    v41_kv_put_le32(mut + tail + 8u, 0u);
    int empty = write_blob(path, mut, bytes) &&
        v41_kvreuse_import(path, 0, bytes, tokens, n) &&
        v41_kvreuse_restore(&st, tokens, n, NULL) == n && st.mainh_n == 0u && st.mainh_end == -1;
    printf("empty mainh run: %s\n", empty ? "PASS" : "FAIL");
    failures += !empty;
    memcpy(mut, blob, bytes);
    v41_kv_put_le32(mut + tail + 8u, DS4_N_SWA + 1u);
    failures += rejected(path, mut, bytes, tokens, n, "mainh count beyond ring");
    memcpy(mut, blob, bytes);
    v41_kv_put_le64(mut + tail, n);
    failures += rejected(path, mut, bytes, tokens, n, "mainh frontier beyond prompt");
    memcpy(mut, blob, bytes);
    v41_kv_put_le32(mut + 4u, ds4_layer_compress_ratio(0) + 1u);
    failures += rejected(path, mut, bytes, tokens, n, "remainder beyond allocation");

    const size_t group_bytes = DS4_V41_CKV_BYTES + DS4_V41_IDXK_BYTES;
    failures += reject_counter(path, blob, bytes, bytes + 6u * rowb, 4u, 4u,
                                tokens, n, "complete group left pending");
    failures += reject_counter(path, blob, bytes, bytes - 2u * rowb, 4u, 0u,
                                tokens, n, "remainder below prompt frontier");
    failures += reject_counter(path, blob, bytes, bytes - group_bytes, 0u, 1u,
                                tokens, n, "group below prompt frontier");
    failures += reject_counter(path, blob, bytes, bytes + group_bytes, 0u, 3u,
                                tokens, n, "group beyond prompt frontier");
    failures += reject_counter(path, blob, bytes, bytes, 8u, n + 1u,
                                tokens, n, "source window floor beyond prompt");
    const size_t reused = V41_KVREUSE_META_ROW_BYTES;
    failures += reject_counter(path, blob, bytes, bytes + group_bytes, reused, 1u,
                                tokens, n, "reused layer carries private groups");
    failures += reject_counter(path, blob, bytes, bytes + 2u * rowb, reused + 4u, 1u,
                                tokens, n, "reused layer carries private remainder");
    failures += reject_counter(path, blob, bytes, bytes, reused + 8u, n + 1u,
                                tokens, n, "reused window floor beyond prompt");

    /* Reuse the larger K.mainh allocation, then import half the declared
     * ring. Its remaining bytes belong to an older checkpoint, not this file. */
    const uint64_t mainh_bytes = ds4_gpu_tensor_bytes(st.mainh), half = mainh_bytes / 2u;
    const size_t tail_bytes = DS4_N_VOCAB * 4u + (size_t)n * sizeof(uint64_t);
    const size_t mainh_off = bytes - (size_t)mainh_bytes - tail_bytes;
    memcpy(mut, blob, bytes);
    v41_kv_put_le64(mut + tail + 12u, half);
    memmove(mut + mainh_off + half, blob + mainh_off + mainh_bytes, tail_bytes);
    const size_t short_bytes = bytes - (size_t)half;
    if (!write_blob(path, mut, short_bytes)) { failures++; }
    else {
        const int loaded =
            v41_kvreuse_import(path, 0, short_bytes, tokens, n);
        const int used = loaded && v41_kvreuse_restore(&st, tokens, n, NULL) != 0u;
        printf("short mainh after larger allocation: %s%s\n", loaded ? "FAIL (accepted)" : "PASS",
               used ? " (used stale bytes)" : "");
        failures += loaded;
    }
    ds4_gpu_tensor_free(K.mainh); K.mainh = NULL;
    failures += rejected(path, mut, short_bytes, tokens, n, "short mainh with fresh allocation");

    /* Artifacts without towers never allocate a ring and export an empty
     * span with zero counters; import must preserve that valid shape. */
    memcpy(mut, blob, bytes);
    v41_kv_put_le64(mut + tail, 0u);
    v41_kv_put_le32(mut + tail + 8u, 0u);
    v41_kv_put_le64(mut + tail + 12u, 0u);
    memmove(mut + mainh_off, blob + mainh_off + mainh_bytes, tail_bytes);
    g_ds4_v41.mtp_towers = 0u; g_ds4_v41.n_mtp_target = 0u;
    st.mainh = NULL; st.mainh_n = 0u; st.mainh_end = 0;
    const size_t no_mtp_bytes = bytes - (size_t)mainh_bytes;
    int no_mtp = write_blob(path, mut, no_mtp_bytes) &&
        v41_kvreuse_import(path, 0, no_mtp_bytes, tokens, n) &&
        v41_kvreuse_restore(&st, tokens, n, NULL) == n && K.mainh_bytes == 0u;
    printf("no-MTP checkpoint: %s\n", no_mtp ? "PASS" : "FAIL");
    failures += !no_mtp;

    /* A ratio-one source stores one group per token and has no pending
     * compressor rows. It remains distinct from the positive-ratio borrower. */
    g_ratio[0] = 1u;
    ds4_gpu_tensor_free(st.comp_kv[0]); ds4_gpu_tensor_free(st.index_k[0]);
    st.comp_kv[0] = v41_alloc((uint64_t)n * DS4_V41_CKV_BYTES, &ok);
    st.index_k[0] = v41_alloc((uint64_t)n * DS4_V41_IDXK_BYTES, &ok);
    st.ng_src[0] = n; st.cpend[0] = 0u;
    v41_kvreuse_save(&st, tokens, n, NULL);
    struct stat sb;
    int ratio_one = v41_kvreuse_export(path, V41_KV_CLASS_PROMPT) && stat(path, &sb) == 0;
    st.ng_src[0] = 0u;
    memset(st.comp_kv[0]->data, 0x3c, (size_t)st.comp_kv[0]->bytes);
    if (ratio_one) {
        ratio_one =
            v41_kvreuse_import(path, 0, (uint64_t)sb.st_size, tokens, n) &&
            v41_kvreuse_restore(&st, tokens, n, NULL) == n && st.ng_src[0] == n &&
            st.cpend[0] == 0u && filled(st.comp_kv[0], 0x5a);
    }
    printf("ratio-one source checkpoint: %s\n", ratio_one ? "PASS" : "FAIL");
    failures += !ratio_one;
    failures += vision_cases(&st, path);
    failures += turn_cases(&st);
    failures += disk_turn_cases(&st, path);
    unlink(path);
    v41_kvreuse_cleanup();
    free(mut); free(blob); free(st.himg);
    printf("ds41 KV import gate: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
