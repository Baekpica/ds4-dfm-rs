/* Inkling-2 K/V tier for the eight full-attention layers. Their NVFP4
 * blocks (128 positions, 147,456 bytes) stay resident up to
 * inkling2_resident_ctx() positions; a longer context appends the blocks
 * past that to an unlinked O_DIRECT file, [layer][block]:
 *
 *   resident  blocks [0, resident)             device store, as without a tier
 *   file      blocks [resident, window_block0) written once, read per forward
 *   window    blocks [window_block0, ...)      pinned memory the device reads
 *                                              and writes in place: the blocks
 *                                              a forward writes
 *
 * These layers attend densely, so every forward past the resident context
 * reads the layer's whole file region, INKLING2_RING blocks at a time, into
 * two pinned buffers; the attention state (m, l, acc) carries across the key
 * segments (ds4_gpu_ink2_attn_seg). Sliding layers keep their rings
 * resident.
 *
 * Reads run on an I/O thread, ahead of the graph: the forward queues its
 * reads in the order the graph takes them, and read n fills buffer n % 2
 * once the attention that took read n - 2 has run. Large single reads
 * already reach the drive's rate (O_DIRECT on GB10: 4 MB 4.3 GB/s, 36 MB
 * 5.9 GB/s; two or three concurrent streams 5.4-5.7 GB/s in all); reading
 * ahead keeps it busy while the device computes. */

typedef struct { uint32_t layer, first, count; } inkling2_read;  // full layer, file blocks

typedef struct {
    bool spill;
    uint32_t resident;        // resident blocks per full layer
    uint32_t blocks;          // blocks of the whole context
    uint32_t window_blocks, window_block0;
    uint8_t *window;          // pinned [full layer][window block][block bytes]
    ds4_gpu_tensor *window_all, *window_gpu[INKLING2_FULL_LAYERS];
    uint8_t *ring[2];         // pinned read buffers of INKLING2_RING blocks
    ds4_gpu_tensor *ring_gpu[2];
    int fd;
    // This forward's reads; counters number reads over the session (under mu).
    inkling2_read *reads;
    uint32_t max_reads;
    uint64_t base, planned, done, taken, released;
    bool busy, failed, quit, io_live;
    pthread_t io;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint64_t read_bytes, write_bytes;
    double io_s, wait_s;
} inkling2_kv;

#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
static uint64_t inkling2_file_offset(const inkling2_kv *kv, uint32_t layer, uint32_t block) {
    return ((uint64_t)layer * (kv->blocks - kv->resident) + (block - kv->resident)) * DS4_INK2_BLOCK_BYTES;
}

static bool inkling2_kv_io(inkling2_kv *kv, bool write, uint8_t *buf, uint64_t bytes, uint64_t off) {
    const double t0 = now_sec();
    for (uint64_t done = 0; done < bytes;) {
        const ssize_t got = write ? pwrite(kv->fd, buf + done, bytes - done, (off_t)(off + done))
                                  : pread(kv->fd, buf + done, bytes - done, (off_t)(off + done));
        if (got < 0 && errno == EINTR) { continue; }
        if (got <= 0) {
            fprintf(stderr, "ds4: Inkling-2 K/V %s failed: %s\n", write ? "write" : "read",
                got < 0 ? strerror(errno) : "short transfer");
            return false;
        }
        done += (uint64_t)got;
    }
    if (write) { kv->write_bytes += bytes; } else { kv->read_bytes += bytes; }
    kv->io_s += now_sec() - t0;
    return true;
}

// The I/O thread: each queued read, in order, once its buffer is free.
static void *inkling2_io_main(void *arg) {
    inkling2_kv *kv = arg;
    pthread_mutex_lock(&kv->mu);
    for (;;) {
        while (!kv->quit && (kv->done == kv->planned || kv->released + 2u <= kv->done)) {
            pthread_cond_wait(&kv->cv, &kv->mu);
        }
        if (kv->quit) { break; }
        const uint64_t seq = kv->done;
        const inkling2_read r = kv->reads[seq - kv->base];
        kv->busy = true;
        pthread_mutex_unlock(&kv->mu);
        const uint32_t slot = (uint32_t)(seq % 2u);
        const bool ok = ds4_gpu_ink2_mark_wait(slot) &&
            inkling2_kv_io(kv, false, kv->ring[slot], (uint64_t)r.count * DS4_INK2_BLOCK_BYTES,
                inkling2_file_offset(kv, r.layer, r.first));
        pthread_mutex_lock(&kv->mu);
        if (!ok) { kv->failed = true; }
        kv->busy = false;
        kv->done = seq + 1u;
        pthread_cond_broadcast(&kv->cv);
    }
    pthread_mutex_unlock(&kv->mu);
    return NULL;
}

/* Drops reads a failed forward queued but never took, once none is in
 * flight: the buffers they filled carry no attention to wait for. */
static void inkling2_io_idle(inkling2_kv *kv) {
    pthread_mutex_lock(&kv->mu);
    while (kv->busy) { pthread_cond_wait(&kv->cv, &kv->mu); }
    kv->planned = kv->taken = kv->released = kv->done;
    kv->failed = false;
    pthread_mutex_unlock(&kv->mu);
}

static void inkling2_kv_close(inkling2_kv *kv) {
    if (!kv->spill) { return; }
    if (kv->io_live) {
        pthread_mutex_lock(&kv->mu);
        kv->quit = true;
        pthread_cond_broadcast(&kv->cv);
        pthread_mutex_unlock(&kv->mu);
        pthread_join(kv->io, NULL);
        pthread_cond_destroy(&kv->cv);
        pthread_mutex_destroy(&kv->mu);
    }
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: Inkling-2 K/V tier read %.2f GiB, wrote %.2f GiB, %.2f s in I/O, "
        "%.2f s waited\n", (double)kv->read_bytes / 1073741824.0, (double)kv->write_bytes / 1073741824.0, kv->io_s,
        kv->wait_s);
    free(kv->reads);
    for (uint32_t i = 0; i < INKLING2_FULL_LAYERS; i++) { ds4_gpu_tensor_free(kv->window_gpu[i]); }
    ds4_gpu_minimax_host_free(kv->window, kv->window_all);
    for (uint32_t i = 0; i < 2; i++) { ds4_gpu_minimax_host_free(kv->ring[i], kv->ring_gpu[i]); }
    if (kv->fd >= 0) { close(kv->fd); }
    memset(kv, 0, sizeof(*kv));
}

static bool inkling2_kv_open(inkling2_kv *kv, uint32_t ctx, uint32_t cap) {
    memset(kv, 0, sizeof(*kv));
    kv->fd = -1;
    if (!inkling2_spill(ctx)) { return true; }
    kv->spill = true;
    kv->blocks = inkling2_ctx_blocks(ctx);
    kv->resident = inkling2_resident_ctx() / DS4_INK2_KV_BLOCK;
    kv->window_blocks = inkling2_window_blocks(cap);
    kv->window_block0 = kv->resident;
    const uint64_t window_layer = (uint64_t)kv->window_blocks * DS4_INK2_BLOCK_BYTES;
    kv->window = ds4_gpu_minimax_host_alloc(window_layer * INKLING2_FULL_LAYERS, &kv->window_all);
    for (uint32_t i = 0; i < 2; i++) {
        kv->ring[i] = ds4_gpu_minimax_host_alloc((uint64_t)INKLING2_RING * DS4_INK2_BLOCK_BYTES, &kv->ring_gpu[i]);
    }
    // Reads per forward: each full layer's file blocks, INKLING2_RING at a time.
    kv->max_reads = INKLING2_FULL_LAYERS * ((kv->blocks - kv->resident + INKLING2_RING - 1u) / INKLING2_RING);
    kv->reads = xcalloc(kv->max_reads ? kv->max_reads : 1u, sizeof(*kv->reads));
    char dir[PATH_MAX];
    bool ok = kv->window && kv->ring[0] && kv->ring[1] && !minimax_kv_dir(dir, sizeof(dir));
    for (uint32_t i = 0; ok && i < INKLING2_FULL_LAYERS; i++) {
        kv->window_gpu[i] = ds4_gpu_tensor_view(kv->window_all, i * window_layer, window_layer);
        ok = kv->window_gpu[i] != NULL;
    }
    if (!ok) {
        fprintf(stderr, "ds4: Inkling-2 K/V tier allocation failed\n");
        inkling2_kv_close(kv);
        return false;
    }
    // Reserve the whole file now: a full disk fails here, not mid-forward.
    const uint64_t bytes = (uint64_t)INKLING2_FULL_LAYERS * (kv->blocks - kv->resident) * DS4_INK2_BLOCK_BYTES;
    kv->fd = open(dir, O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    if (kv->fd < 0 || fallocate(kv->fd, 0, 0, (off_t)bytes) != 0) {
        fprintf(stderr, "ds4: Inkling-2 K/V file in %s (%.2f GiB) failed: %s (DS4_KV_OFFLOAD_DIR sets the "
            "directory)\n", dir, (double)bytes / 1073741824.0, strerror(errno));
        inkling2_kv_close(kv);
        return false;
    }
    pthread_mutex_init(&kv->mu, NULL);
    pthread_cond_init(&kv->cv, NULL);
    kv->io_live = pthread_create(&kv->io, NULL, inkling2_io_main, kv) == 0;
    if (!kv->io_live) {
        pthread_cond_destroy(&kv->cv);
        pthread_mutex_destroy(&kv->mu);
        fprintf(stderr, "ds4: Inkling-2 K/V I/O thread failed to start\n");
        inkling2_kv_close(kv);
        return false;
    }
    fprintf(stderr, "ds4: Inkling-2 K/V past %u positions on SSD: %s, %.2f GiB file, %.2f GiB pinned\n",
        kv->resident * DS4_INK2_KV_BLOCK, dir, (double)bytes / 1073741824.0,
        (double)(window_layer * INKLING2_FULL_LAYERS + 2ull * INKLING2_RING * DS4_INK2_BLOCK_BYTES) / 1073741824.0);
    return true;
}

// A new sequence: the window starts at the first file block again.
static void inkling2_kv_restart(inkling2_kv *kv) {
    if (kv->spill) { kv->window_block0 = kv->resident; }
}

/* This forward's reads, in the order inkling2_tier_attn takes them. */
static void inkling2_kv_plan(inkling2_kv *kv) {
    pthread_mutex_lock(&kv->mu);
    kv->base = kv->planned;
    uint32_t n = 0;
    for (uint32_t l = 0; l < INKLING2_FULL_LAYERS; l++) {
        for (uint32_t b = kv->resident; b < kv->window_block0 && n < kv->max_reads; b += INKLING2_RING) {
            const uint32_t count = kv->window_block0 - b < INKLING2_RING ? kv->window_block0 - b : INKLING2_RING;
            kv->reads[n++] = (inkling2_read){l, b, count};
        }
    }
    kv->planned += n;
    pthread_cond_broadcast(&kv->cv);
    pthread_mutex_unlock(&kv->mu);
}

/* Before a forward of rows pos..: the window slides to start at the block
 * holding `pos` (never below the resident blocks), the complete blocks it
 * leaves going to the file, and the forward's reads are queued. Positions
 * only grow; a rewind restarts. The previous forward's kernels may still
 * write the window: drain first. */
static bool inkling2_kv_begin(inkling2_kv *kv, uint32_t rows, uint32_t pos) {
    if (!kv->spill) { return true; }
    inkling2_io_idle(kv);
    const uint32_t first = pos / DS4_INK2_KV_BLOCK, last = (pos + rows - 1u) / DS4_INK2_KV_BLOCK;
    if (last < kv->resident) { return true; }
    const uint32_t start = first > kv->resident ? first : kv->resident;
    if (start < kv->window_block0 || last - start >= kv->window_blocks) { return false; }
    if (start == kv->window_block0) {
        inkling2_kv_plan(kv);
        return true;
    }
    if (!ds4_gpu_synchronize()) { return false; }
    const uint32_t moved = start - kv->window_block0;
    const uint32_t flushed = moved < kv->window_blocks ? moved : kv->window_blocks;
    const uint64_t window_layer = (uint64_t)kv->window_blocks * DS4_INK2_BLOCK_BYTES;
    for (uint32_t l = 0; l < INKLING2_FULL_LAYERS; l++) {
        uint8_t *w = kv->window + l * window_layer;
        if (!inkling2_kv_io(kv, true, w, (uint64_t)flushed * DS4_INK2_BLOCK_BYTES,
                inkling2_file_offset(kv, l, kv->window_block0))) { return false; }
        if (moved < kv->window_blocks) {
            memmove(w, w + (uint64_t)moved * DS4_INK2_BLOCK_BYTES,
                (uint64_t)(kv->window_blocks - moved) * DS4_INK2_BLOCK_BYTES);
        }
    }
    kv->window_block0 = start;
    inkling2_kv_plan(kv);
    return true;
}

/* A rewind to `pos`: the window restarts at the block holding it (never
 * below the resident blocks); a file block's rows below pos are read back.
 * Rows at or above pos are rewritten before any read. */
static bool inkling2_kv_rewind(inkling2_kv *kv, uint32_t pos) {
    if (!kv->spill) { return true; }
    inkling2_io_idle(kv);
    const uint32_t block = pos / DS4_INK2_KV_BLOCK, start = block > kv->resident ? block : kv->resident;
    if (start >= kv->window_block0) { return true; }
    if (!ds4_gpu_synchronize()) { return false; }
    kv->window_block0 = start;
    if (block < kv->resident || pos % DS4_INK2_KV_BLOCK == 0) { return true; }
    const uint64_t window_layer = (uint64_t)kv->window_blocks * DS4_INK2_BLOCK_BYTES;
    for (uint32_t l = 0; l < INKLING2_FULL_LAYERS; l++) {
        if (!inkling2_kv_io(kv, false, kv->window + l * window_layer, DS4_INK2_BLOCK_BYTES,
                inkling2_file_offset(kv, l, start))) { return false; }
    }
    return true;
}

// Where full layer `il`'s blocks live for the store kernel.
static bool inkling2_kv_tier(const inkling2_kv *kv, uint32_t il, ds4_ink2_tier *tier) {
    if (!kv->spill) { return false; }
    *tier = (ds4_ink2_tier){kv->window_gpu[il / DS4_INK2_BLOCK], kv->resident, kv->window_block0,
        kv->window_blocks};
    return true;
}

/* The buffer holding full layer `il`'s file blocks [first, first + count),
 * the forward's next queued read, once the I/O thread has read it. */
static ds4_gpu_tensor *inkling2_kv_read(inkling2_kv *kv, uint32_t il, uint32_t first, uint32_t count) {
    pthread_mutex_lock(&kv->mu);
    const uint64_t seq = kv->taken;
    const inkling2_read *r = seq < kv->planned ? &kv->reads[seq - kv->base] : NULL;
    bool ok = r && r->layer == il / DS4_INK2_BLOCK && r->first == first && r->count == count;
    const double t0 = now_sec();
    while (ok && kv->done <= seq) { pthread_cond_wait(&kv->cv, &kv->mu); }
    kv->wait_s += now_sec() - t0;
    ok = ok && !kv->failed;
    if (ok) { kv->taken = seq + 1u; }
    pthread_mutex_unlock(&kv->mu);
    if (!ok) { fprintf(stderr, "ds4: Inkling-2 K/V read of layer %u blocks %u+%u failed\n", il, first, count); }
    return ok ? kv->ring_gpu[seq % 2u] : NULL;
}

// After the attention over the buffer inkling2_kv_read returned last.
static bool inkling2_kv_release(inkling2_kv *kv) {
    if (!ds4_gpu_ink2_mark((uint32_t)((kv->taken - 1u) % 2u))) { return false; }
    pthread_mutex_lock(&kv->mu);
    kv->released = kv->taken;
    pthread_cond_broadcast(&kv->cv);
    pthread_mutex_unlock(&kv->mu);
    return true;
}
#else
static inline bool inkling2_kv_open(inkling2_kv *kv, uint32_t ctx, uint32_t cap) {
    (void)cap;
    memset(kv, 0, sizeof(*kv));
    return !inkling2_spill(ctx);
}
static inline void inkling2_kv_close(inkling2_kv *kv) { (void)kv; }
static inline void inkling2_kv_restart(inkling2_kv *kv) { (void)kv; }
static inline bool inkling2_kv_rewind(inkling2_kv *kv, uint32_t pos) {
    (void)pos;
    return !kv->spill;
}
#endif
