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
} ds4_v41_state;

static struct {
    uint32_t ctx, n_mtp_target, mtp_towers;
    uint8_t is_kv_source[DS4_MAX_LAYER];
} g_ds4_v41 = {64u, 1u, 1u, {1u, 0u}};
static uint32_t g_ratio[DS4_MAX_LAYER] = {4u, 4u};

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
    *ok = true;
    return t;
}

static uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t->bytes; }
static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) { free(t->data); free(t); }
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
    st.win_from[1] = 8u;   /* a CED decoder layer retains a nonzero window floor */
    st.mainh_end = n - 1u; st.mainh_n = DS4_N_SWA;
    v41_kvreuse_save(&st, tokens, n);

    char path[] = "/tmp/ds41-kv-import-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) { return 1; }
    close(fd);
    if (!v41_kvreuse_export(path)) { unlink(path); return 1; }
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
        v41_kvreuse_restore(&st, tokens, n) != n) {
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
    const size_t tail = DS4_N_LAYER * V41_KVREUSE_META_ROW_BYTES;
    memcpy(mut, blob, bytes);
    v41_kv_put_le64(mut + tail, UINT64_MAX);
    v41_kv_put_le32(mut + tail + 8u, 0u);
    int empty = write_blob(path, mut, bytes) && v41_kvreuse_import(path, 0, bytes, tokens, n) &&
        v41_kvreuse_restore(&st, tokens, n) == n && st.mainh_n == 0u && st.mainh_end == -1;
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
    const size_t mainh_off = bytes - (size_t)mainh_bytes - DS4_N_VOCAB * 4u;
    memcpy(mut, blob, bytes);
    v41_kv_put_le64(mut + tail + 12u, half);
    memmove(mut + mainh_off + half, blob + mainh_off + mainh_bytes, DS4_N_VOCAB * 4u);
    const size_t short_bytes = bytes - (size_t)half;
    if (!write_blob(path, mut, short_bytes)) { failures++; }
    else {
        const int loaded = v41_kvreuse_import(path, 0, short_bytes, tokens, n);
        const int used = loaded && v41_kvreuse_restore(&st, tokens, n) != 0u;
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
    memmove(mut + mainh_off, blob + mainh_off + mainh_bytes, DS4_N_VOCAB * 4u);
    g_ds4_v41.mtp_towers = 0u; g_ds4_v41.n_mtp_target = 0u;
    st.mainh = NULL; st.mainh_n = 0u; st.mainh_end = 0;
    const size_t no_mtp_bytes = bytes - (size_t)mainh_bytes;
    int no_mtp = write_blob(path, mut, no_mtp_bytes) &&
        v41_kvreuse_import(path, 0, no_mtp_bytes, tokens, n) &&
        v41_kvreuse_restore(&st, tokens, n) == n && K.mainh_bytes == 0u;
    printf("no-MTP checkpoint: %s\n", no_mtp ? "PASS" : "FAIL");
    failures += !no_mtp;

    /* A ratio-one source stores one group per token and has no pending
     * compressor rows. It remains distinct from the positive-ratio borrower. */
    g_ratio[0] = 1u;
    ds4_gpu_tensor_free(st.comp_kv[0]); ds4_gpu_tensor_free(st.index_k[0]);
    st.comp_kv[0] = v41_alloc((uint64_t)n * DS4_V41_CKV_BYTES, &ok);
    st.index_k[0] = v41_alloc((uint64_t)n * DS4_V41_IDXK_BYTES, &ok);
    st.ng_src[0] = n; st.cpend[0] = 0u;
    v41_kvreuse_save(&st, tokens, n);
    struct stat sb;
    int ratio_one = v41_kvreuse_export(path) && stat(path, &sb) == 0;
    st.ng_src[0] = 0u;
    memset(st.comp_kv[0]->data, 0x3c, (size_t)st.comp_kv[0]->bytes);
    if (ratio_one) {
        ratio_one = v41_kvreuse_import(path, 0, (uint64_t)sb.st_size, tokens, n) &&
            v41_kvreuse_restore(&st, tokens, n) == n && st.ng_src[0] == n &&
            st.cpend[0] == 0u && filled(st.comp_kv[0], 0x5a);
    }
    printf("ratio-one source checkpoint: %s\n", ratio_one ? "PASS" : "FAIL");
    failures += !ratio_one;
    unlink(path);
    free(mut); free(blob);
    printf("ds41 KV import gate: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
