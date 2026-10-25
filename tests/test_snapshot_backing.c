/* Real-file ownership/failure tests; no model or CUDA runtime. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../ds4.h"

static void payload_set_err(char *err, size_t len, const char *message) {
    if (len) { snprintf(err, len, "%s", message); }
}
static int fail_sync, fail_reserve;
static int test_sync(int fd) {
    if (fail_sync) { errno = EIO; return -1; }
    return fdatasync(fd);
}
static int test_reserve(int fd, int mode, off_t off, off_t bytes) {
    if (fail_reserve) { errno = ENOSPC; return -1; }
    return fallocate(fd, mode, off, bytes);
}
#define fdatasync test_sync
#define fallocate test_reserve
#include "../ds4_snapshot.inc"
#undef fdatasync
#undef fallocate

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "snapshot FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
enum { BLOCK = 65536, LARGE_BLOCKS = 1024 };

static void empty_dir(const char *path) {
    DIR *dir = opendir(path);
    CHECK(dir);
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        CHECK(!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."));
    }
    CHECK(closedir(dir) == 0);
}

static void roundtrip(ds4_session_snapshot *snap, const char *dir, unsigned blocks) {
    char err[256];
    unsigned char data[BLOCK], readback[BLOCK];
    const uint64_t bytes = (uint64_t)blocks * BLOCK;
    CHECK(!snapshot_file_open(snap, dir, bytes, err, sizeof(err)));
    CHECK(!snap->ptr && !snap->cap && !snap->len && snap->file);
    const int fd = fileno(snap->file);
    struct stat st;
    CHECK(fstat(fd, &st) == 0 && st.st_nlink == 0);
    CHECK(fcntl(fd, F_GETFD) & FD_CLOEXEC);
    for (unsigned i = 0; i < blocks; i++) {
        memset(data, i % 251, sizeof(data));
        CHECK(fwrite(data, 1, sizeof(data), snap->file) == sizeof(data));
        if ((i + 1) % 64 == 0) {
            CHECK(!snapshot_io_trim(snap->file, (uint64_t)(i - 63) * BLOCK,
                SNAPSHOT_IO_WRITE, err, sizeof(err)));
        }
    }
    CHECK(!snapshot_file_finish(snap, bytes, err, sizeof(err)) && snap->len == bytes);
    for (unsigned pass = 0; pass < 2; pass++) {
        CHECK(fseeko(snap->file, 0, SEEK_SET) == 0);
        for (unsigned i = 0; i < blocks; i++) {
            memset(data, i % 251, sizeof(data));
            CHECK(fread(readback, 1, sizeof(readback), snap->file) == sizeof(readback));
            CHECK(!memcmp(data, readback, sizeof(data)));
        }
        CHECK(!snapshot_io_trim(snap->file, 0, SNAPSHOT_IO_READ, err, sizeof(err)));
    }
    empty_dir(dir);
}

int main(void) {
    char dir[] = "/tmp/ds4-snapshot-XXXXXX", err[256];
    CHECK(mkdtemp(dir));
    ds4_session_snapshot snap = {0};
    // A prior resident snapshot is released on the first SSD save.
    snap.ptr = malloc(BLOCK); snap.cap = snap.len = BLOCK;
    CHECK(snap.ptr);
    roundtrip(&snap, dir, LARGE_BLOCKS);
    roundtrip(&snap, dir, 3);
    roundtrip(&snap, dir, 5);
    const int old_fd = fileno(snap.file);
    snapshot_release(&snap);
    CHECK(!snap.file && !snap.ptr && !snap.len && !snap.cap);
    CHECK(fcntl(old_fd, F_GETFD) == -1 && errno == EBADF);
    snapshot_release(&snap);

    CHECK(snapshot_file_open(&snap, "/no/such/ds4-directory", BLOCK, err, sizeof(err)));
    CHECK(snapshot_file_open(&snap, dir, UINT64_MAX, err, sizeof(err)));
    fail_reserve = 1;
    CHECK(snapshot_file_open(&snap, dir, BLOCK, err, sizeof(err)));
    fail_reserve = 0;
    CHECK(!snap.file && !snap.len);
    CHECK(!snapshot_file_open(&snap, dir, BLOCK, err, sizeof(err)));
    CHECK(snapshot_file_finish(&snap, BLOCK, err, sizeof(err))); // Incomplete write.
    CHECK(!snap.file && !snap.len);

    CHECK(!snapshot_file_open(&snap, dir, 1, err, sizeof(err)));
    CHECK(fputc(17, snap.file) != EOF);
    fail_sync = 1;
    CHECK(snapshot_file_finish(&snap, 1, err, sizeof(err)));
    fail_sync = 0;
    CHECK(!snap.file && !snap.len);
    CHECK(!snapshot_file_open(&snap, dir, BLOCK, err, sizeof(err)));
    CHECK(fseeko(snap.file, BLOCK, SEEK_SET) == 0);
    CHECK(ftruncate(fileno(snap.file), BLOCK - 1) == 0);
    CHECK(snapshot_file_finish(&snap, BLOCK, err, sizeof(err))); // Truncated backing.

    // Memory streams keep the existing trailing-byte contract and need no fd.
    char memory[2] = {0};
    FILE *fp = fmemopen(memory, sizeof(memory), "wb");
    CHECK(fp && fputc(0x7f, fp) != EOF);
    CHECK(!snapshot_io_trim(fp, 0, SNAPSHOT_IO_WRITE, err, sizeof(err)));
    CHECK(fclose(fp) == 0 && memory[0] == 0x7f);
    empty_dir(dir);
    CHECK(rmdir(dir) == 0);
    puts("snapshot backing: PASS");
    return 0;
}
