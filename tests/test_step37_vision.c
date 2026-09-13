/* Actual F16 projector, independent PyTorch equations at both crop sizes. */
static void vision_trace(const char *, const void *, unsigned, unsigned);
#define STEP37_VISION_TRACE vision_trace
#include "../ds4.c"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "vision FAIL %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static const char *reference_dir;
static bool local_replay;
static void read_ref(const char *name, float *out, size_t count) {
    char path[1024]; CHECK(snprintf(path, sizeof(path), "%s/%s.f32", reference_dir, name) < (int)sizeof(path));
    FILE *fp = fopen(path, "rb"); CHECK(fp && fread(out, sizeof(float), count, fp) == count);
    CHECK(fgetc(fp) == EOF && !fclose(fp));
}
static void vision_trace(const char *name, const void *tensor, unsigned width, unsigned rows) {
    const size_t count = (size_t)width * rows, bytes = count * sizeof(float);
    float *got = xmalloc(bytes), *want = xmalloc(bytes);
    read_ref(name, want, count); CHECK(ds4_gpu_tensor_read(tensor, 0, got, bytes));
    double error = 0, reference = 0, maximum = 0;
    for (size_t i = 0; i < count; i++) {
        CHECK(isfinite(got[i]) && isfinite(want[i]));
        const double d = (double)got[i] - want[i];
        error += d * d; reference += (double)want[i] * want[i]; maximum = fmax(maximum, fabs(d));
    }
    CHECK(reference > 0);
    double relative = sqrt(error / reference);
    printf("%s %zu rel_rms=%.8g max_abs=%.8g\n", name, count, relative, maximum); fflush(stdout);
    CHECK(relative < (local_replay ? 0.001 : 0.01));
    if (local_replay) { CHECK(ds4_gpu_tensor_write((ds4_gpu_tensor *)tensor, 0, want, bytes)); }
    free(got); free(want);
}
int main(int argc, char **argv) {
    CHECK(argc == 4 || (argc == 5 && (!strcmp(argv[4], "--local") || !strcmp(argv[4], "--reuse"))));
    const unsigned edge = (unsigned)atoi(argv[3]); CHECK(edge == 504 || edge == 728);
    reference_dir = argv[2]; local_replay = argc == 5 && !strcmp(argv[4], "--local");
    const bool reuse = argc == 5 && !strcmp(argv[4], "--reuse");
    ds4_model m; model_open(&m, argv[1], true, false);
    ds4_step37_vision_weights weights; step37_vision_bind(&weights, &m);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(m.map, m.size));
    ds4_step37_vision graph;
    CHECK(!step37_vision_alloc(&graph, 727));
    CHECK(step37_vision_alloc(&graph, reuse ? 728 : edge));
    if (reuse) {
        CHECK(step37_vision_shape(&graph, 504));
        CHECK(step37_vision_shape(&graph, 728));
        CHECK(step37_vision_shape(&graph, edge));
    }
    CHECK(!ds4_gpu_step37_columns(graph.columns, graph.pixels, 729, 3));
    CHECK(!ds4_gpu_step37_columns(graph.columns, graph.pixels, 728, 4));
    CHECK(!ds4_gpu_step37_position(graph.hidden, m.map, m.size, m.size, edge / 14));
    CHECK(!ds4_gpu_step37_vqkv(graph.qkv, m.map, m.size, m.size, edge / 14));
    CHECK(!ds4_gpu_step37_vgelu(graph.mlp, m.map, m.size, 0, graph.rows - 1));
    CHECK(!ds4_gpu_step37_vresidual(graph.hidden, graph.temp, m.map, m.size, 0, m.size, graph.rows));
    const size_t count = (size_t)edge * edge * 3;
    float *pixels = xmalloc(count * sizeof(float)); read_ref("pixels", pixels, count);
    CHECK(step37_vision_forward(&graph, &m, &weights, pixels));
    CHECK(graph.rows == (edge / 14) * (edge / 14));
    CHECK(graph.output_rows == (edge / 56) * (edge / 56));
    printf("Step vision edge=%u features=%u scratch=%" PRIu64 " PASS\n", edge, graph.output_rows, graph.bytes);
    step37_vision_free(&graph); ds4_gpu_cleanup(); model_close(&m); free(pixels);
}
