/* Hermetic split-GGUF loader test.
 *
 * Builds two tiny GGUF v3 shards with one F32 tensor each, opens shard 0
 * through the real loader, and verifies that both tensor payloads remain
 * addressable through the engine's existing map + abs_offset contract.
 */
#include "../ds4.c"
#include <sys/wait.h>

static int test_failures;

static void test_assert(bool ok, const char *expr, int line) {
    if (ok) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, line, expr);
    test_failures++;
}

#define TEST_ASSERT(expr) test_assert((expr), #expr, __LINE__)

static void write_u16(FILE *f, uint16_t v) {
    TEST_ASSERT(fwrite(&v, sizeof(v), 1, f) == 1);
}

static void write_u32(FILE *f, uint32_t v) {
    TEST_ASSERT(fwrite(&v, sizeof(v), 1, f) == 1);
}

static void write_u64(FILE *f, uint64_t v) {
    TEST_ASSERT(fwrite(&v, sizeof(v), 1, f) == 1);
}

static void write_string(FILE *f, const char *s) {
    const uint64_t n = (uint64_t)strlen(s);
    write_u64(f, n);
    TEST_ASSERT(fwrite(s, 1, (size_t)n, f) == n);
}

static void write_kv_u16(FILE *f, const char *key, uint16_t value) {
    write_string(f, key);
    write_u32(f, GGUF_VALUE_UINT16);
    write_u16(f, value);
}

static void write_kv_i32(FILE *f, const char *key, int32_t value) {
    write_string(f, key);
    write_u32(f, GGUF_VALUE_INT32);
    TEST_ASSERT(fwrite(&value, sizeof(value), 1, f) == 1);
}

static void write_fixture_shard(const char *path, const char *tensor_name,
                                float value, bool first) {
    FILE *f = fopen(path, "wb");
    TEST_ASSERT(f != NULL);
    if (!f) return;

    write_u32(f, DS4_GGUF_MAGIC);
    write_u32(f, 3);
    write_u64(f, 1);
    write_u64(f, first ? 2 : 0);
    if (first) {
        write_kv_u16(f, "split.count", 2);
        write_kv_i32(f, "split.tensors.count", 2);
    }

    write_string(f, tensor_name);
    write_u32(f, 1);
    write_u64(f, 1);
    write_u32(f, DS4_TENSOR_F32);
    write_u64(f, 0);

    long pos = ftell(f);
    TEST_ASSERT(pos >= 0);
    while (pos >= 0 && (pos % 32) != 0) {
        TEST_ASSERT(fputc(0, f) != EOF);
        pos++;
    }
    TEST_ASSERT(fwrite(&value, sizeof(value), 1, f) == 1);
    TEST_ASSERT(fclose(f) == 0);
}

static float tensor_scalar(const ds4_model *m, const char *name) {
    const ds4_tensor *t = model_find_tensor(m, name);
    TEST_ASSERT(t != NULL);
    float value = 0.0f;
    if (t) memcpy(&value, m->map + t->abs_offset, sizeof(value));
    return value;
}

static void reject_named_plan(ds4_host_tensor_dir *dir) {
    const pid_t child = fork();
    TEST_ASSERT(child >= 0);
    if (child < 0) { return; }
    if (child == 0) {
        (void)freopen("/dev/null", "w", stderr);
        ds4_host_tensor_dir_install(dir);
        ds4_model m;
        model_open(&m, "/not/a/model/file", false, false);
        model_close(&m);
        _exit(0);
    }
    int status = 0;
    TEST_ASSERT(waitpid(child, &status, 0) == child);
    TEST_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) != 0);
}

static void test_named_shards(const char *a, const char *b) {
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    ds4_host_shard shards[2] = {{.path = a}, {.path = b}};
    ds4_host_tensor tensors[2] = {0};
    const char *names[] = {"second.weight", "single.weight"};
    uint64_t base = 0, data_pos = 0;
    for (uint32_t k = 0; k < 2; k++) {
        ds4_model part;
        model_open(&part, shards[k].path, false, false);
        const ds4_tensor *t = model_find_tensor(&part, names[k]);
        TEST_ASSERT(t != NULL);
        if (!t) { model_close(&part); return; }
        shards[k].size = part.size;
        shards[k].base = base;
        tensors[k] = (ds4_host_tensor){.name = names[k], .ndim = 1, .dim = {1},
            .type = DS4_TENSOR_F32, .abs_offset = base + t->abs_offset,
            .rel_offset = t->rel_offset, .bytes = sizeof(float)};
        if (k == 0) { data_pos = part.tensor_data_pos; }
        base += align_up(part.size, page);
        model_close(&part);
    }
    ds4_host_tensor_dir dir = {.n = 2, .v = tensors, .data_pos = data_pos,
        .alignment = 32, .n_shards = 2, .shards = shards};
    ds4_host_tensor_dir_install(&dir);
    ds4_model model;
    model_open(&model, "/not/a/model/file", false, false);
    ds4_host_tensor_dir_clear();
    TEST_ASSERT(model.n_tensors == 2 && model.split_count == 2);
    TEST_ASSERT(tensor_scalar(&model, "second.weight") == -2.5f);
    TEST_ASSERT(tensor_scalar(&model, "single.weight") == 3.75f);
    TEST_ASSERT(model.tensor_traits != NULL && model.size == base);
    const int fd0 = model.fd, fd1 = model.split_fds[0];
    model_close(&model);
    TEST_ASSERT(fcntl(fd0, F_GETFD) == -1 && errno == EBADF);
    TEST_ASSERT(fcntl(fd1, F_GETFD) == -1 && errno == EBADF);

    shards[1].base++;
    reject_named_plan(&dir);
    shards[1].base--;
    shards[1].size++;
    reject_named_plan(&dir);
    shards[1].size--;
    tensors[0].bytes++;
    reject_named_plan(&dir);
    tensors[0].bytes--;
    dir.shards = NULL;
    reject_named_plan(&dir);
}

int main(int argc, char **argv) {
    if (argc > 2) {
        fprintf(stderr, "usage: %s [split-model-00001-of-N.gguf]\n", argv[0]);
        return 2;
    }
    char dir[] = "/tmp/ds4-split-gguf-XXXXXX";
    TEST_ASSERT(mkdtemp(dir) != NULL);

    char first[PATH_MAX];
    char second[PATH_MAX];
    char single[PATH_MAX];
    TEST_ASSERT(snprintf(first, sizeof(first),
                         "%s/fixture-00001-of-00002.gguf", dir) > 0);
    TEST_ASSERT(snprintf(second, sizeof(second),
                         "%s/fixture-00002-of-00002.gguf", dir) > 0);
    TEST_ASSERT(snprintf(single, sizeof(single),
                         "%s/single.gguf", dir) > 0);
    write_fixture_shard(first, "first.weight", 1.25f, true);
    write_fixture_shard(second, "second.weight", -2.5f, false);
    write_fixture_shard(single, "single.weight", 3.75f, false);

    char sibling[PATH_MAX];
    TEST_ASSERT(model_split_sibling_path(first, 1, 2,
                                          sibling, sizeof(sibling)));
    TEST_ASSERT(strcmp(sibling, second) == 0);

    ds4_model model;
    model_open(&model, first, false, false);
    TEST_ASSERT(model.split_count == 2);
    TEST_ASSERT(model.n_tensors == 2);
    TEST_ASSERT(fabsf(tensor_scalar(&model, "first.weight") - 1.25f) < 1e-7f);
    TEST_ASSERT(fabsf(tensor_scalar(&model, "second.weight") + 2.5f) < 1e-7f);
    model_close(&model);

    model_open(&model, single, false, false);
    TEST_ASSERT(model.split_count == 0);
    TEST_ASSERT(model.n_tensors == 1);
    TEST_ASSERT(fabsf(tensor_scalar(&model, "single.weight") - 3.75f) < 1e-7f);
    model_close(&model);

    test_named_shards(second, single);

    TEST_ASSERT(unlink(first) == 0);
    TEST_ASSERT(unlink(second) == 0);
    TEST_ASSERT(unlink(single) == 0);
    TEST_ASSERT(rmdir(dir) == 0);

    if (argc == 2) {
        ds4_model real;
        model_open(&real, argv[1], false, false);
        TEST_ASSERT(real.split_count > 1);
        TEST_ASSERT(real.n_tensors > 1);
        printf("real split GGUF: %u shards, %" PRIu64 " tensors\n",
               real.split_count, real.n_tensors);
        model_close(&real);
    }
    if (test_failures) {
        fprintf(stderr, "split GGUF tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("split GGUF tests: ok");
    return 0;
}
