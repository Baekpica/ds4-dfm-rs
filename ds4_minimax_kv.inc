/* MiniMax K/V tiers. Index keys (72 bytes per token and layer) stay
 * resident in every tier: the indexer scores all blocks on the device. K/V
 * pages (73,728 bytes per block and layer) stay resident up to
 * minimax_resident_ctx() tokens. A longer context keeps them in an unlinked
 * O_DIRECT file, [layer][block], and a pinned pool the GPU reads in place
 * holds what a forward touches:
 *
 *   - at forward start, every layer's written blocks are pinned in the pool
 *     (a block that already holds tokens is read back first);
 *   - per layer, once the indexer has selected, the selected blocks missing
 *     from the pool are read into least-recently-used slots (file-contiguous
 *     runs in one preadv), a dirty victim written back first.
 *
 * The pool holds one layer's every block plus a forward's pins, so the
 * union of a prefill chunk's selections always fits. A slot table
 * [layer][block] -> slot, also pinned, steers the store and attention
 * kernels; the resident tier passes none. Both tiers run the same kernels
 * on the same bytes, so logits do not depend on the tier. */
#include <sys/uio.h>

enum {
    // Default largest context with resident K/V pages: 60 x 1,024 blocks
    // x 73,728 bytes = 4.5 GB beside 0.55 GB of index keys.
    MINIMAX_RESIDENT_CTX = 131072,
    // Decode keeps up to 128 blocks per layer when the history is shorter
    // than that.
    MINIMAX_HOT_PAGES = MINIMAX_LAYERS * 128,
    // One forward pins the blocks its rows write: at most 33 per layer.
    MINIMAX_PIN_PAGES = MINIMAX_LAYERS * (MINIMAX_PREFILL_MAX / MINIMAX_BLOCK + 1),
    MINIMAX_IO_THREADS = 8,
    // File-contiguous blocks per preadv/pwritev.
    MINIMAX_IO_RUN = 32,
};

/* DS4_MINIMAX_KV_RESIDENT_CTX: the largest context whose K/V pages stay
 * resident; 0 sends every context to the SSD tier (tier equivalence
 * tests). */
static uint32_t minimax_resident_ctx(void) {
    static uint32_t value;
    static bool read;
    if (!read) {
        const char *s = getenv("DS4_MINIMAX_KV_RESIDENT_CTX");
        char *end = NULL;
        const unsigned long v = s && *s ? strtoul(s, &end, 10) : 0;
        value = s && *s && end != s && !*end && v <= MINIMAX_CONTEXT ? (uint32_t)v : MINIMAX_RESIDENT_CTX;
        read = true;
    }
    return value;
}

static bool minimax_spill(uint32_t ctx) { return ctx > minimax_resident_ctx(); }

static uint64_t minimax_pool_slots(uint32_t ctx) {
    const uint64_t blocks = minimax_blocks(ctx);
    return (blocks > MINIMAX_HOT_PAGES ? blocks : MINIMAX_HOT_PAGES) + MINIMAX_PIN_PAGES;
}

// One layer's K/V pages in the resident tier.
static uint64_t minimax_page_bytes(uint32_t ctx) {
    return minimax_blocks(ctx) * MINIMAX_KV_BLOCK;
}

// Index keys, then resident pages or the SSD tier's pool and slot table.
static uint64_t minimax_kv_bytes(uint32_t ctx) {
    const uint64_t index = (uint64_t)MINIMAX_LAYERS * minimax_index_bytes(ctx);
    if (!minimax_spill(ctx)) { return index + (uint64_t)MINIMAX_LAYERS * minimax_page_bytes(ctx); }
    return index + minimax_pool_slots(ctx) * MINIMAX_KV_BLOCK +
        (uint64_t)MINIMAX_LAYERS * minimax_blocks(ctx) * sizeof(int32_t);
}

/* DS4_KV_OFFLOAD_DIR: where SSD-tier files live (unlinked: nothing stays
 * behind); default ~/.cache/ds4/kv-offload. */
static int minimax_kv_dir(char *out, size_t cap) {
    const char *dir = getenv("DS4_KV_OFFLOAD_DIR");
    if (dir && *dir) { return snprintf(out, cap, "%s", dir) < (int)cap ? 0 : -1; }
    const char *home = getenv("HOME");
    if (!home || !*home) { return -1; }
    char parent[PATH_MAX];
    if (snprintf(parent, sizeof(parent), "%s/.cache/ds4", home) >= (int)sizeof(parent)) { return -1; }
    (void)mkdir(parent, 0700);
    if (snprintf(out, cap, "%s/kv-offload", parent) >= (int)cap) { return -1; }
    (void)mkdir(out, 0700);
    return 0;
}

typedef struct {
    uint32_t layer, block;   // owner; UINT32_MAX when free
    int32_t prev, next;      // LRU list, head most recent
    uint64_t stamp;          // last resolve that needed it
    bool dirty, pinned;
} minimax_slot;

typedef struct {
    bool spill;
    // Every block below the frontier is pooled (nothing evicted since the
    // sequence started): resolves have nothing to read and are skipped.
    bool complete;
    uint32_t blocks, slots;
    uint64_t stamp;
    uint8_t *pool;           // pinned pool, host address
    int32_t *table;          // pinned [layer][block] slot ids, host address
    ds4_gpu_tensor *pool_gpu, *table_gpu, *rows[MINIMAX_LAYERS];
    minimax_slot *slot;
    int32_t head, tail;
    uint64_t *seen;
    uint32_t *miss;
    int fd;
    uint64_t hits, misses, read_bytes, write_bytes;
    double io_s;
} minimax_kv;

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
/* ---------------------------------------------------------------- I/O pool */

typedef struct {
    bool write;
    uint32_t layer, block, count;
    int32_t slots[MINIMAX_IO_RUN];
} minimax_io_job;

static struct {
    pthread_mutex_t mu;
    pthread_cond_t go, done;
    bool started, failed;
    uint64_t gen;
    const minimax_kv *kv;
    const minimax_io_job *jobs;
    uint32_t count, next, finished;
} g_minimax_io = {.mu = PTHREAD_MUTEX_INITIALIZER, .go = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER};

static uint64_t minimax_kv_offset(const minimax_kv *kv, uint32_t il, uint32_t block) {
    return ((uint64_t)il * kv->blocks + block) * MINIMAX_KV_BLOCK;
}

static bool minimax_io_job_run(const minimax_kv *kv, const minimax_io_job *job) {
    struct iovec iov[MINIMAX_IO_RUN];
    for (uint32_t i = 0; i < job->count; i++) {
        iov[i].iov_base = kv->pool + (uint64_t)job->slots[i] * MINIMAX_KV_BLOCK;
        iov[i].iov_len = MINIMAX_KV_BLOCK;
    }
    uint64_t off = minimax_kv_offset(kv, job->layer, job->block);
    uint64_t left = (uint64_t)job->count * MINIMAX_KV_BLOCK;
    struct iovec *v = iov;
    int n = (int)job->count;
    while (left) {
        const ssize_t got = job->write ? pwritev(kv->fd, v, n, (off_t)off) : preadv(kv->fd, v, n, (off_t)off);
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) {
            fprintf(stderr, "ds4: MiniMax K/V %s failed at layer %u block %u: %s\n", job->write ? "write" : "read",
                job->layer, job->block, got < 0 ? strerror(errno) : "short transfer");
            return false;
        }
        off += (uint64_t)got;
        left -= (uint64_t)got;
        // A short O_DIRECT transfer ends on a 512-byte boundary: advance the iovecs.
        size_t done = (size_t)got;
        while (n && done >= v->iov_len) { done -= v->iov_len; v++; n--; }
        if (n && done) { v->iov_base = (uint8_t *)v->iov_base + done; v->iov_len -= done; }
    }
    return true;
}

// Takes jobs until the batch is empty; called with the lock held.
static void minimax_io_drain(void) {
    while (g_minimax_io.next < g_minimax_io.count) {
        const minimax_io_job *job = &g_minimax_io.jobs[g_minimax_io.next++];
        const minimax_kv *kv = g_minimax_io.kv;
        pthread_mutex_unlock(&g_minimax_io.mu);
        const bool ok = minimax_io_job_run(kv, job);
        pthread_mutex_lock(&g_minimax_io.mu);
        if (!ok) { g_minimax_io.failed = true; }
        if (++g_minimax_io.finished == g_minimax_io.count) { pthread_cond_broadcast(&g_minimax_io.done); }
    }
}

static void *minimax_io_main(void *arg) {
    (void)arg;
    uint64_t seen = 0;
    pthread_mutex_lock(&g_minimax_io.mu);
    for (;;) {
        while (g_minimax_io.gen == seen) { pthread_cond_wait(&g_minimax_io.go, &g_minimax_io.mu); }
        seen = g_minimax_io.gen;
        minimax_io_drain();
    }
    return NULL;
}

// Runs the jobs on the pool's threads and the caller; true when all succeed.
static bool minimax_io_run(minimax_kv *kv, const minimax_io_job *jobs, uint32_t count) {
    if (!count) { return true; }
    const double t0 = now_sec();
    pthread_mutex_lock(&g_minimax_io.mu);
    for (uint32_t i = 0; !g_minimax_io.started && i < MINIMAX_IO_THREADS; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, minimax_io_main, NULL) == 0) { pthread_detach(t); }
    }
    g_minimax_io.started = true;
    g_minimax_io.kv = kv;
    g_minimax_io.jobs = jobs;
    g_minimax_io.count = count;
    g_minimax_io.next = g_minimax_io.finished = 0;
    g_minimax_io.failed = false;
    g_minimax_io.gen++;
    pthread_cond_broadcast(&g_minimax_io.go);
    minimax_io_drain();
    while (g_minimax_io.finished < count) { pthread_cond_wait(&g_minimax_io.done, &g_minimax_io.mu); }
    const bool ok = !g_minimax_io.failed;
    g_minimax_io.count = 0;
    pthread_mutex_unlock(&g_minimax_io.mu);
    kv->io_s += now_sec() - t0;
    return ok;
}

/* -------------------------------------------------------------- slot pool */

// Every slot is on the list: unlink and front keep it whole.
static void minimax_lru_unlink(minimax_kv *kv, int32_t s) {
    minimax_slot *x = &kv->slot[s];
    if (x->prev >= 0) { kv->slot[x->prev].next = x->next; } else { kv->head = x->next; }
    if (x->next >= 0) { kv->slot[x->next].prev = x->prev; } else { kv->tail = x->prev; }
    x->prev = x->next = -1;
}

static void minimax_lru_front(minimax_kv *kv, int32_t s) {
    minimax_lru_unlink(kv, s);
    kv->slot[s].next = kv->head;
    if (kv->head >= 0) { kv->slot[kv->head].prev = s; }
    kv->head = s;
    if (kv->tail < 0) { kv->tail = s; }
}

static void minimax_kv_close(minimax_kv *kv) {
    if (!kv->spill) { return; }
    for (uint32_t il = 0; il < MINIMAX_LAYERS; il++) { free(kv->rows[il]); }
    ds4_gpu_minimax_host_free(kv->pool, kv->pool_gpu);
    ds4_gpu_minimax_host_free(kv->table, kv->table_gpu);
    if (kv->fd >= 0) { close(kv->fd); }
    free(kv->slot);
    free(kv->seen);
    free(kv->miss);
    memset(kv, 0, sizeof(*kv));
}

static bool minimax_kv_open(minimax_kv *kv, uint32_t ctx) {
    memset(kv, 0, sizeof(*kv));
    kv->fd = -1;
    if (!minimax_spill(ctx)) { return true; }
    kv->spill = true;
    kv->blocks = (uint32_t)minimax_blocks(ctx);
    kv->slots = (uint32_t)minimax_pool_slots(ctx);
    // DS4_MINIMAX_KV_POOL_SLOTS (diagnostic): a smaller pool, to make tests
    // evict and read back; a forward that cannot pin or resolve fails.
    const char *forced = getenv("DS4_MINIMAX_KV_POOL_SLOTS");
    if (forced && *forced && strtoul(forced, NULL, 10) > 0 && strtoul(forced, NULL, 10) < kv->slots) {
        kv->slots = (uint32_t)strtoul(forced, NULL, 10);
    }
    const uint64_t table = (uint64_t)MINIMAX_LAYERS * kv->blocks;
    char dir[PATH_MAX];
    kv->pool = ds4_gpu_minimax_host_alloc((uint64_t)kv->slots * MINIMAX_KV_BLOCK, &kv->pool_gpu);
    kv->table = ds4_gpu_minimax_host_alloc(table * sizeof(int32_t), &kv->table_gpu);
    kv->slot = calloc(kv->slots, sizeof(*kv->slot));
    kv->seen = calloc(kv->blocks, sizeof(*kv->seen));
    kv->miss = malloc((size_t)kv->blocks * sizeof(*kv->miss));
    if (!kv->pool || !kv->table || !kv->slot || !kv->seen || !kv->miss || minimax_kv_dir(dir, sizeof(dir))) {
        fprintf(stderr, "ds4: MiniMax K/V tier allocation failed\n");
        minimax_kv_close(kv);
        return false;
    }
    for (uint32_t il = 0; il < MINIMAX_LAYERS; il++) {
        kv->rows[il] = ds4_gpu_tensor_view(kv->table_gpu, (uint64_t)il * kv->blocks * sizeof(int32_t),
            (uint64_t)kv->blocks * sizeof(int32_t));
        if (!kv->rows[il]) { minimax_kv_close(kv); return false; }
    }
    // Reserve the whole file now: a full disk fails here, not mid-forward.
    const uint64_t bytes = table * MINIMAX_KV_BLOCK;
    kv->fd = open(dir, O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    if (kv->fd < 0 || fallocate(kv->fd, 0, 0, (off_t)bytes) != 0) {
        fprintf(stderr, "ds4: MiniMax K/V file in %s (%.2f GiB) failed: %s (DS4_KV_OFFLOAD_DIR sets the "
            "directory)\n", dir, (double)bytes / 1073741824.0, strerror(errno));
        minimax_kv_close(kv);
        return false;
    }
    for (uint64_t i = 0; i < table; i++) { kv->table[i] = -1; }
    // Every slot is always on the LRU list; free ones start in slot order.
    for (uint32_t s = 0; s < kv->slots; s++) {
        kv->slot[s] = (minimax_slot){UINT32_MAX, UINT32_MAX, (int32_t)s - 1,
            s + 1u < kv->slots ? (int32_t)s + 1 : -1, 0, false, false};
    }
    kv->head = 0;
    kv->tail = (int32_t)kv->slots - 1;
    kv->complete = true;
    fprintf(stderr, "ds4: MiniMax K/V on SSD: %s, %.2f GiB file, %.2f GiB pinned pool (%u slots)\n", dir,
        (double)bytes / 1073741824.0, (double)kv->slots * MINIMAX_KV_BLOCK / 1073741824.0, kv->slots);
    return true;
}

/* A slot for (il, block): the least-recently-used slot that is neither
 * pinned nor needed by this resolve. A dirty victim queues its write-back
 * (which must land before the slot is reused). */
static int32_t minimax_kv_take(minimax_kv *kv, uint32_t il, uint32_t block, minimax_io_job *writes,
        uint32_t *nwrites) {
    int32_t s = kv->tail;
    while (s >= 0 && (kv->slot[s].pinned || kv->slot[s].stamp == kv->stamp)) { s = kv->slot[s].prev; }
    if (s < 0) { return -1; }
    minimax_slot *x = &kv->slot[s];
    if (x->layer != UINT32_MAX) {
        kv->complete = false;
        if (x->dirty) {
            writes[(*nwrites)++] = (minimax_io_job){true, x->layer, x->block, 1u, {s}};
            kv->write_bytes += MINIMAX_KV_BLOCK;
        }
        kv->table[(uint64_t)x->layer * kv->blocks + x->block] = -1;
    }
    *x = (minimax_slot){il, block, x->prev, x->next, kv->stamp, false, false};
    kv->table[(uint64_t)il * kv->blocks + block] = s;
    minimax_lru_front(kv, s);
    return s;
}

// Coalesces reads of file-contiguous blocks (sorted by layer, block) into runs.
static uint32_t minimax_kv_runs(minimax_kv *kv, const uint32_t *layers, const uint32_t *blocks, uint32_t n,
        minimax_io_job *jobs) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < n; i++) {
        const int32_t s = kv->table[(uint64_t)layers[i] * kv->blocks + blocks[i]];
        minimax_io_job *last = count ? &jobs[count - 1] : NULL;
        if (last && last->layer == layers[i] && last->block + last->count == blocks[i] &&
            last->count < MINIMAX_IO_RUN) {
            last->slots[last->count++] = s;
        } else {
            jobs[count++] = (minimax_io_job){false, layers[i], blocks[i], 1u, {s}};
        }
        kv->read_bytes += MINIMAX_KV_BLOCK;
    }
    return count;
}

static int minimax_u32_cmp(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/* Pins every layer's blocks written by rows pos..pos+rows-1. A block that
 * already holds tokens before pos and is not pooled is read back. The
 * previous forward's last kernels may still read the pool: drain first. */
static bool minimax_kv_begin(minimax_kv *kv, uint32_t rows, uint32_t pos) {
    if (!kv->spill) { return true; }
    if (!ds4_gpu_synchronize()) { return false; }
    kv->stamp++;
    const uint32_t first = pos / MINIMAX_BLOCK, last = (pos + rows - 1u) / MINIMAX_BLOCK;
    const uint32_t per = last - first + 1u, n = per * MINIMAX_LAYERS;
    minimax_io_job *writes = malloc((size_t)n * sizeof(*writes)), *reads = malloc((size_t)n * sizeof(*reads));
    uint32_t *layers = malloc((size_t)n * sizeof(*layers)), *blocks = malloc((size_t)n * sizeof(*blocks));
    uint32_t nwrites = 0, nreads = 0;
    bool ok = writes && reads && layers && blocks;
    for (uint32_t il = 0; ok && il < MINIMAX_LAYERS; il++) {
        for (uint32_t b = first; ok && b <= last; b++) {
            int32_t s = kv->table[(uint64_t)il * kv->blocks + b];
            if (s < 0) {
                s = minimax_kv_take(kv, il, b, writes, &nwrites);
                ok = s >= 0;
                if (ok && (uint64_t)b * MINIMAX_BLOCK < pos) { layers[nreads] = il; blocks[nreads++] = b; }
            } else {
                minimax_lru_front(kv, s);
            }
            if (ok) {
                kv->slot[s].stamp = kv->stamp;
                kv->slot[s].pinned = kv->slot[s].dirty = true;
            }
        }
    }
    // Write-backs land before any read reuses their slots.
    ok = ok && minimax_io_run(kv, writes, nwrites) &&
        minimax_io_run(kv, reads, minimax_kv_runs(kv, layers, blocks, nreads, reads));
    if (!ok) { fprintf(stderr, "ds4: MiniMax K/V pinning failed (rows=%u pos=%u)\n", rows, pos); }
    free(writes);
    free(reads);
    free(layers);
    free(blocks);
    return ok;
}

/* The forward is done with its pins (its pages stay dirty until evicted).
 * DS4_MINIMAX_KV_TRACE prints the tier's running counters. */
static void minimax_kv_end(minimax_kv *kv) {
    if (!kv->spill) { return; }
    for (uint32_t s = 0; s < kv->slots; s++) { kv->slot[s].pinned = false; }
    if (getenv("DS4_MINIMAX_KV_TRACE")) {
        fprintf(stderr, "ds4: MiniMax K/V tier: hits %" PRIu64 " misses %" PRIu64 " read %.1f MiB written %.1f MiB "
            "I/O %.3f s\n", kv->hits, kv->misses, (double)kv->read_bytes / 1048576.0,
            (double)kv->write_bytes / 1048576.0, kv->io_s);
    }
}

/* Layer il's selections (`sel`, rows x 4 x 16 block ids, -1 unused) into
 * the pool: hits move to the LRU front, misses are read in file order. */
static bool minimax_kv_needs_resolve(const minimax_kv *kv) { return kv->spill && !kv->complete; }

static bool minimax_kv_resolve(minimax_kv *kv, uint32_t il, const int32_t *sel, uint32_t count) {
    kv->stamp++;
    uint32_t nmiss = 0;
    for (uint32_t i = 0; i < count; i++) {
        const int32_t b = sel[i];
        if (b < 0 || (uint32_t)b >= kv->blocks || kv->seen[b] == kv->stamp) { continue; }
        kv->seen[b] = kv->stamp;
        const int32_t s = kv->table[(uint64_t)il * kv->blocks + (uint32_t)b];
        if (s >= 0) {
            kv->slot[s].stamp = kv->stamp;
            minimax_lru_front(kv, s);
            kv->hits++;
        } else {
            kv->miss[nmiss++] = (uint32_t)b;
        }
    }
    if (!nmiss) { return true; }
    kv->misses += nmiss;
    qsort(kv->miss, nmiss, sizeof(*kv->miss), minimax_u32_cmp);
    minimax_io_job *writes = malloc((size_t)nmiss * sizeof(*writes)), *reads = malloc((size_t)nmiss * sizeof(*reads));
    uint32_t *layers = malloc((size_t)nmiss * sizeof(*layers));
    uint32_t nwrites = 0;
    bool ok = writes && reads && layers;
    for (uint32_t i = 0; ok && i < nmiss; i++) {
        layers[i] = il;
        ok = minimax_kv_take(kv, il, kv->miss[i], writes, &nwrites) >= 0;
    }
    ok = ok && minimax_io_run(kv, writes, nwrites) &&
        minimax_io_run(kv, reads, minimax_kv_runs(kv, layers, kv->miss, nmiss, reads));
    if (!ok) { fprintf(stderr, "ds4: MiniMax K/V read of layer %u failed (%u blocks)\n", il, nmiss); }
    free(writes);
    free(reads);
    free(layers);
    return ok;
}

// Every dirty page to the file (before the file is read as a whole).
static bool minimax_kv_flush(minimax_kv *kv) {
    if (!kv->spill) { return true; }
    minimax_io_job *jobs = malloc((size_t)kv->slots * sizeof(*jobs));
    uint32_t n = 0;
    for (uint32_t s = 0; jobs && s < kv->slots; s++) {
        minimax_slot *x = &kv->slot[s];
        if (x->layer == UINT32_MAX || !x->dirty) { continue; }
        jobs[n++] = (minimax_io_job){true, x->layer, x->block, 1u, {(int32_t)s}};
        kv->write_bytes += MINIMAX_KV_BLOCK;
        x->dirty = false;
    }
    const bool ok = jobs && minimax_io_run(kv, jobs, n);
    free(jobs);
    return ok;
}

// Forgets every pooled page (a restore rewrote their file copies, or the
// sequence ended).
static void minimax_kv_drop(minimax_kv *kv) {
    kv->complete = false;
    for (uint32_t s = 0; kv->spill && s < kv->slots; s++) {
        minimax_slot *x = &kv->slot[s];
        if (x->layer != UINT32_MAX) { kv->table[(uint64_t)x->layer * kv->blocks + x->block] = -1; }
        x->layer = x->block = UINT32_MAX;
        x->dirty = x->pinned = false;
    }
}

// A new sequence: nothing of the old one is kept, and the new one's blocks
// are pinned as it writes them.
static void minimax_kv_restart(minimax_kv *kv) {
    if (!kv->spill) { return; }
    minimax_kv_drop(kv);
    kv->complete = true;
}

/* Pages [first, first + count) of layer il through `buf` (count x 73,728
 * bytes, 4 KiB aligned): pooled pages are copied, the rest move with the
 * file. Callers flush before reading and drop after writing. */
static bool minimax_kv_pages(minimax_kv *kv, bool write, uint32_t il, uint32_t first, uint32_t count, uint8_t *buf) {
    const uint64_t bytes = (uint64_t)count * MINIMAX_KV_BLOCK;
    const uint64_t off = minimax_kv_offset(kv, il, first);
    for (uint64_t done = 0; done < bytes;) {
        const ssize_t got = write ? pwrite(kv->fd, buf + done, bytes - done, (off_t)(off + done))
                                  : pread(kv->fd, buf + done, bytes - done, (off_t)(off + done));
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) { return false; }
        done += (uint64_t)got;
    }
    if (write) { kv->write_bytes += bytes; } else { kv->read_bytes += bytes; }
    return true;
}
#elif !defined(DS4_NO_GPU)
// MiniMax is CUDA-only: Metal refuses the graph before any of these run.
static inline bool minimax_kv_open(minimax_kv *kv, uint32_t ctx) {
    memset(kv, 0, sizeof(*kv));
    return !minimax_spill(ctx);
}
static inline void minimax_kv_close(minimax_kv *kv) { (void)kv; }
static inline bool minimax_kv_begin(minimax_kv *kv, uint32_t rows, uint32_t pos) {
    (void)rows; (void)pos;
    return !kv->spill;
}
static inline void minimax_kv_end(minimax_kv *kv) { (void)kv; }
static inline bool minimax_kv_resolve(minimax_kv *kv, uint32_t il, const int32_t *sel, uint32_t count) {
    (void)il; (void)sel; (void)count;
    return !kv->spill;
}
static inline bool minimax_kv_needs_resolve(const minimax_kv *kv) { return kv->spill; }
static inline void minimax_kv_restart(minimax_kv *kv) { (void)kv; }
static inline bool minimax_kv_flush(minimax_kv *kv) { return !kv->spill; }
static inline void minimax_kv_drop(minimax_kv *kv) { (void)kv; }
static inline bool minimax_kv_pages(minimax_kv *kv, bool write, uint32_t il, uint32_t first, uint32_t count,
        uint8_t *buf) {
    (void)kv; (void)write; (void)il; (void)first; (void)count; (void)buf;
    return false;
}
#endif
