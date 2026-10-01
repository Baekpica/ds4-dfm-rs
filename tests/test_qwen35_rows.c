/* Bonsai (qwen35) weight-row read against the exporter's own dequantizer.
 *
 *   ./tests/test_qwen35_rows <bonsai.gguf> [fixture]
 *
 * tests/pq2_0/reference_checksums.txt holds, for every one of the 851 tensors
 * in the artifact and the first 64 rows of each, an FNV-1a checksum over the
 * f32 row the PrismML fork's ggml dequantizer produces, plus the row sum and
 * four leading values.  This test reads the same rows through this tree's own
 * loader and reference reader and compares.
 *
 * That matters because the other parity gate available here compares against
 * the sibling tree's implementation, which is itself derived from the same
 * upstream code: a shared mistake would agree with itself.  The fixture comes
 * from the runtime the weights were exported for.
 *
 * Build: make test-qwen35-rows
 */
#include "../ds4.c"

#define PQ2_0_FIXTURE "tests/pq2_0/reference_checksums.txt"
#define ROWS_PER_TENSOR 64u

typedef struct {
    char     name[160];
    int      type;
    long long ne0;
    long long rows;
    long long rows_total;
    size_t   row_bytes;
    unsigned long long checksum;
    double   sum;
    double   head[4];
} row_expectation;

static uint64_t fnv1a_f32(uint64_t h, const float *x, size_t n) {
    const uint8_t *b = (const uint8_t *)x;
    for (size_t i = 0; i < n * sizeof(float); i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

/* nrows() of the tensor: the product of every dimension above the row width. */
static uint64_t tensor_n_rows(const ds4_tensor *t) {
    uint64_t rows = 1;
    for (uint32_t i = 1; i < t->ndim; i++) rows *= t->dim[i];
    return rows;
}

static bool parse_expectation(const char *line, row_expectation *out) {
    char name[160] = {0};
    int type = 0;
    long long ne0 = 0, rows = 0, rows_total = 0;
    unsigned long long row_bytes = 0, checksum = 0;
    double sum = 0.0, h0 = 0.0, h1 = 0.0, h2 = 0.0, h3 = 0.0;
    if (sscanf(line,
               "%159s type=%d ne0=%lld rows=%lld/%lld row_bytes=%llu "
               "checksum=%llx sum=%lf head=[%lf %lf %lf %lf]",
               name, &type, &ne0, &rows, &rows_total, &row_bytes,
               &checksum, &sum, &h0, &h1, &h2, &h3) != 12) {
        return false;
    }
    snprintf(out->name, sizeof(out->name), "%s", name);
    out->type = type;
    out->ne0 = ne0;
    out->rows = rows;
    out->rows_total = rows_total;
    out->row_bytes = (size_t)row_bytes;
    out->checksum = checksum;
    out->sum = sum;
    out->head[0] = h0;
    out->head[1] = h1;
    out->head[2] = h2;
    out->head[3] = h3;
    return true;
}

/* The fixture lists the tensors in GGUF order, so a single pass compares them
 * positionally; a name mismatch means the file drifted from the fixture. */
static int check_tensor(const ds4_model *m, const ds4_tensor *t, const row_expectation *e) {
    if (t->name.len != strlen(e->name) ||
        memcmp(t->name.ptr, e->name, t->name.len) != 0) {
        printf("FAIL tensor order: got %.*s, fixture says %s\n",
               (int)t->name.len, t->name.ptr, e->name);
        return 1;
    }
    if ((int)t->type != e->type || (long long)t->dim[0] != e->ne0) {
        printf("FAIL %s: type %d ne0 %llu, fixture says type %d ne0 %lld\n",
               e->name, t->type, (unsigned long long)t->dim[0], e->type, e->ne0);
        return 1;
    }
    const gguf_type_info *info = tensor_type(t->type);
    if (!info) {
        printf("FAIL %s: unknown tensor type %d\n", e->name, t->type);
        return 1;
    }
    const size_t row_bytes = (size_t)(t->dim[0] / info->block_elems) * info->block_bytes;
    const uint64_t n_rows = tensor_n_rows(t);
    if (row_bytes != e->row_bytes || (long long)n_rows != e->rows_total) {
        printf("FAIL %s: row_bytes %zu rows %llu, fixture says %zu / %lld\n",
               e->name, row_bytes, (unsigned long long)n_rows, e->row_bytes, e->rows_total);
        return 1;
    }

    const uint64_t rows = n_rows < ROWS_PER_TENSOR ? n_rows : ROWS_PER_TENSOR;
    float *row = xmalloc((size_t)t->dim[0] * sizeof(float));
    uint64_t checksum = 0xcbf29ce484222325ull;
    double sum = 0.0;
    for (uint64_t r = 0; r < rows; r++) {
        ds4_ref_row(m, t, r, row);
        checksum = fnv1a_f32(checksum, row, (size_t)t->dim[0]);
        for (uint64_t i = 0; i < t->dim[0]; i++) sum += row[i];
    }

    int failures = 0;
    if (checksum != e->checksum) {
        printf("FAIL %s: row checksum %016llx, reference %016llx\n",
               e->name, (unsigned long long)checksum, e->checksum);
        failures++;
    }
    if (fabs(sum - e->sum) > 1e-3 + 1e-5 * fabs(e->sum)) {
        printf("FAIL %s: row sum %.6f, reference %.6f\n", e->name, sum, e->sum);
        failures++;
    }
    for (uint32_t i = 0; i < 4; i++) {
        const uint64_t idx = i % t->dim[0];
        const double got = row[idx];
        if (fabs(got - e->head[i]) > 1e-6 * fabs(e->head[i]) + 1e-9) {
            printf("FAIL %s: head[%u] %.9g, reference %.9g\n", e->name, i, got, e->head[i]);
            failures++;
        }
    }
    free(row);
    return failures;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <bonsai.gguf> [fixture]\n", argv[0]);
        return 2;
    }
    const char *fixture = argc == 3 ? argv[2] : PQ2_0_FIXTURE;

    ds4_model model;
    model_open(&model, argv[1], false, false);
    config_validate_model(&model);
    if (!ds4_model_is_qwen35()) {
        fprintf(stderr, "%s is not a qwen35 (Bonsai) artifact\n", argv[1]);
        return 1;
    }

    FILE *f = fopen(fixture, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s; run from the repository root\n", fixture);
        return 2;
    }

    char line[512];
    uint64_t compared = 0, skipped = 0;
    int failures = 0;
    for (uint64_t i = 0; i < model.n_tensors; i++) {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "FAIL fixture ends before tensor %llu\n",
                    (unsigned long long)i);
            failures++;
            break;
        }
        row_expectation e;
        if (!parse_expectation(line, &e)) {
            printf("FAIL could not parse fixture line %llu\n", (unsigned long long)i + 1u);
            failures++;
            continue;
        }
        failures += check_tensor(&model, &model.tensors[i], &e);
        compared++;
    }
    while (fgets(line, sizeof(line), f)) skipped++;
    fclose(f);

    if (failures) {
        printf("qwen35 rows: %d failure(s)\n", failures);
        return 1;
    }
    if (skipped) {
        printf("qwen35 rows: fixture has %llu more line(s) than the model has tensors\n",
               (unsigned long long)skipped);
        return 1;
    }
    printf("qwen35 rows: %llu tensors matched the reference dequantizer "
           "(first %u rows each)\n", (unsigned long long)compared, ROWS_PER_TENSOR);
    return 0;
}
