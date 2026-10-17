/* Rust validates the named-part manifest and owns the mapping plan. Native
 * only realizes that plan in one reserved range; no weight copy is made. */
static void model_open_named(ds4_model *m, int mmap_flags) {
    const ds4_host_tensor_dir *dir = g_host_tensor_dir;
    const long page_l = sysconf(_SC_PAGESIZE);
    if (page_l <= 0 || !dir->shards || !dir->n || !dir->v) {
        ds4_die("invalid named shard plan");
    }
    const uint64_t page = (uint64_t)page_l;
    uint64_t total = 0;
    for (uint32_t k = 0; k < dir->n_shards; k++) {
        const ds4_host_shard *s = &dir->shards[k];
        if (!s->path || !s->path[0] || s->size < 32u ||
            s->base != total || s->size > UINT64_MAX - (page - 1u)) {
            ds4_die("invalid named shard geometry");
        }
        const uint64_t span = align_up(s->size, page);
        if (span > SIZE_MAX - total) { ds4_die("named shard range overflow"); }
        total += span;
    }
    // A tensor must live entirely inside a file, never inside page padding
    // or across two independently mapped parts.
    for (uint32_t i = 0; i < dir->n; i++) {
        const ds4_host_tensor *t = &dir->v[i];
        bool found = false;
        for (uint32_t k = 0; k < dir->n_shards; k++) {
            const ds4_host_shard *s = &dir->shards[k];
            if (t->abs_offset < s->base || t->abs_offset - s->base >= s->size) { continue; }
            found = t->bytes > 0u && t->bytes <= s->size - (t->abs_offset - s->base);
            break;
        }
        if (!found) { ds4_die("tensor outside named shard"); }
    }
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *region = mmap(NULL, (size_t)total, PROT_NONE, flags, -1, 0);
    if (region == MAP_FAILED) { ds4_die_errno("cannot reserve named shards", ""); }
    m->map = region;
    m->size = total;
    m->split_count = dir->n_shards;
    if (dir->n_shards > 1u) {
        m->split_fds = xcalloc(dir->n_shards - 1u, sizeof(*m->split_fds));
        for (uint32_t k = 1; k < dir->n_shards; k++) { m->split_fds[k - 1u] = -1; }
    }
    for (uint32_t k = 0; k < dir->n_shards; k++) {
        const ds4_host_shard *s = &dir->shards[k];
        const int fd = open(s->path, O_RDONLY);
        if (fd < 0) { ds4_die_errno("cannot open named shard", s->path); }
        if (k == 0) { m->fd = fd; }
        else { m->split_fds[k - 1u] = fd; }
        struct stat st;
        if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
            (uint64_t)st.st_size != s->size) { ds4_die("named shard size changed"); }
        void *want = (uint8_t *)region + s->base;
        if (mmap(want, (size_t)s->size, PROT_READ, mmap_flags | MAP_FIXED, fd, 0) == MAP_FAILED) {
            ds4_die_errno("cannot mmap named shard", s->path);
        }
    }
    // Metadata belongs to the first file; the tensor directory belongs to
    // the complete Rust plan. The first GGUF tensor count is not the total.
    ds4_model first = {0};
    first.map = m->map;
    first.size = dir->shards[0].size;
    ds4_cursor c = cursor_at(&first, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic) || magic != DS4_GGUF_MAGIC ||
        !cursor_u32(&c, &first.version) || first.version != 3u ||
        !cursor_u64(&c, &first.n_tensors) || !cursor_u64(&c, &first.n_kv)) {
        ds4_die("invalid named shard GGUF header");
    }
    parse_metadata(&first, &c);
    m->version = first.version;
    m->n_kv = first.n_kv;
    m->kv = first.kv;
    m->alignment = first.alignment;
    model_apply_host_tensors(m);
}
