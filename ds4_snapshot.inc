/* SSD-tier snapshots must not turn the serialized K/V file into a heap copy.
 * The backing is private and unlinked, in the operator's K/V directory. */
typedef enum { SNAPSHOT_IO_READ, SNAPSHOT_IO_WRITE } snapshot_io_mode;

static void snapshot_release(ds4_session_snapshot *snap) {
    free(snap->ptr);
    if (snap->file) { fclose(snap->file); }
    memset(snap, 0, sizeof(*snap));
}

#ifdef __linux__
static int snapshot_file_open(ds4_session_snapshot *snap, const char *dir,
        uint64_t bytes, char *err, size_t errlen) {
    snapshot_release(snap);
    if (!bytes || bytes > INT64_MAX) {
        payload_set_err(err, errlen, "invalid SSD session snapshot size");
        return 1;
    }
    const int fd = open(dir, O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create SSD session snapshot");
        return 1;
    }
    // Reserve before reading any GPU state; a full SSD fails at admission.
    if (fallocate(fd, 0, 0, (off_t)bytes) != 0) {
        close(fd);
        payload_set_err(err, errlen, "failed to reserve SSD session snapshot space");
        return 1;
    }
    snap->file = fdopen(fd, "w+b");
    if (!snap->file) {
        close(fd);
        payload_set_err(err, errlen, "failed to open SSD session snapshot stream");
        return 1;
    }
    return 0;
}
#endif

/* Flush and discard completed layer spans. Buffered file I/O alone would
 * exchange a large heap allocation for an equally large dirty page cache.
 * DONTNEED is advisory: live long-context gates still measure host pressure. */
static int snapshot_io_trim(FILE *fp, uint64_t start, snapshot_io_mode mode,
        char *err, size_t errlen) {
#ifdef __linux__
    const int fd = fileno(fp);
    if (fd < 0) { return 0; } // fmemopen snapshots have no page cache.
    const off_t end = ftello(fp);
    if (end < 0 || (uint64_t)end < start ||
            (mode == SNAPSHOT_IO_WRITE && (fflush(fp) != 0 || fdatasync(fd) != 0))) {
        payload_set_err(err, errlen, "failed to flush SSD session snapshot span");
        return 1;
    }
    (void)posix_fadvise(fd, (off_t)start, end - (off_t)start, POSIX_FADV_DONTNEED);
#else
    (void)fp; (void)start; (void)mode; (void)err; (void)errlen;
#endif
    return 0;
}

static int snapshot_file_finish(ds4_session_snapshot *snap, uint64_t bytes,
        char *err, size_t errlen) {
    const off_t end = ftello(snap->file);
    struct stat st;
    if (end < 0 || (uint64_t)end != bytes ||
            snapshot_io_trim(snap->file, 0, SNAPSHOT_IO_WRITE, err, errlen) ||
            fstat(fileno(snap->file), &st) != 0 || st.st_size < 0 || (uint64_t)st.st_size != bytes) {
        payload_set_err(err, errlen, "failed to finalize SSD session snapshot");
        snapshot_release(snap);
        return 1;
    }
    snap->len = bytes;
    return 0;
}
