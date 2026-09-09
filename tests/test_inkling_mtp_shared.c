/* Actual shared MQ85GB embedding/head with all eight imported BF16 depths. */
#include "../ds4.c"

enum { ROWS = 7 };
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static int read_top(ds4_inkling_graph *g, float *values) {
    CHECK(ds4_gpu_tensor_read(g->logits, 0, values, DS4_N_VOCAB * sizeof(float)));
    int top = 0;
    for (unsigned i = 0; i < INKLING_VALID_VOCAB; i++) {
        CHECK(isfinite(values[i]));
        if (values[i] > values[top]) { top = (int)i; }
    }
    for (unsigned i = INKLING_VALID_VOCAB; i < DS4_N_VOCAB; i++) {
        CHECK(values[i] == -INFINITY);
    }
    return top;
}

typedef struct { unsigned char *data; size_t bytes; } layer_snapshot;
typedef enum { STATE_PREFIX, STATE_FULL } state_view;

static layer_snapshot read_layer(const ds4_inkling_mtp_graph *d, unsigned depth, state_view view) {
    const inkling_layer_state *s = &d->graph.layer[depth];
    unsigned valid = d->positions[depth] < s->capacity ? d->positions[depth] : s->capacity;
    if (view == STATE_FULL) {
        valid = s->capacity;
    }
    const size_t kv_bytes = (size_t)valid * 2 * IK_KV * sizeof(uint16_t);
    const size_t bytes = kv_bytes + IK_HISTORY * (2 * IK_KV + 2 * IK_HIDDEN) * sizeof(float);
    layer_snapshot out = {xmalloc(bytes), bytes};
    if (kv_bytes) {
        CHECK(ds4_gpu_tensor_read(s->kv, 0, out.data, kv_bytes));
    }
    size_t offset = kv_bytes;
    for (unsigned i = 0; i < IK_CONV_STREAMS; i++) {
        const size_t len = IK_HISTORY * (i < 2 ? IK_KV : IK_HIDDEN) * sizeof(float);
        CHECK(ds4_gpu_tensor_read(s->conv[i], 0, out.data + offset, len));
        offset += len;
    }
    return out;
}

/* Assemble the source rotation over the complete prefix. Preserve each
 * depth's stable state before evaluating its last eight speculative rows. */
static void rotated_reference(ds4_inkling_mtp_graph *d, const ds4_model *main,
                               const ds4_weights *shared, const ds4_model *mtp,
                               const ds4_inkling_draft *weights,
                               ds4_gpu_tensor *hidden, const int *prefix, unsigned n,
                               int bonus, int *draft, layer_snapshot *states) {
    CHECK(inkling_draft_reset(d));
    int *ids = xmalloc(n * sizeof(*ids));
    memcpy(ids, prefix + 1, (n - 1) * sizeof(*ids));
    ids[n - 1] = bonus;
    const unsigned stable = n > INKLING_DRAFT_LAYERS ? n - INKLING_DRAFT_LAYERS : 0;
    const size_t row_bytes = IK_HIDDEN * sizeof(float);
    float *logits = xmalloc(DS4_N_VOCAB * sizeof(float));
    for (unsigned depth = 0; depth < INKLING_DRAFT_LAYERS; depth++) {
        if (stable) {
            ds4_gpu_tensor *h = ds4_gpu_tensor_view(hidden, 0, stable * row_bytes);
            CHECK(h && inkling_draft_tokens(d, main, shared, mtp, weights, depth, h, ids, stable));
            CHECK(ds4_gpu_tensor_copy(hidden, 0, d->graph.buf[IK_X], 0, stable * row_bytes));
            ds4_gpu_tensor_free(h);
        }
        states[depth] = read_layer(d, depth, STATE_PREFIX);
        ds4_gpu_tensor *tail = ds4_gpu_tensor_view(hidden, stable * row_bytes, (n - stable) * row_bytes);
        CHECK(tail && inkling_draft_tokens(d, main, shared, mtp, weights, depth, tail, ids + stable, n - stable));
        CHECK(ds4_gpu_tensor_copy(hidden, stable * row_bytes, d->graph.buf[IK_X], 0, (n - stable) * row_bytes));
        ds4_gpu_tensor_free(tail);
        draft[depth] = read_top(&d->graph, logits);
        memmove(ids, ids + 1, (n - 1) * sizeof(*ids));
        ids[n - 1] = draft[depth];
    }
    free(ids); free(logits);
}

static void check_boundary(const ds4_model *main, const ds4_weights *shared,
                            const ds4_model *mtp, const ds4_inkling_draft *weights) {
    enum { TOKENS = 41, CONTEXT = 64, EXTEND_CAP = 25 };
    const unsigned points[] = {1, 7, 8, 9, 16, 25, TOKENS}, chunks[] = {1, 3, 13};
    const size_t row_bytes = IK_HIDDEN * sizeof(float), bytes = TOKENS * row_bytes;
    int tokens[TOKENS + 1], wanted[INKLING_DRAFT_LAYERS], got[INKLING_DRAFT_LAYERS];
    float *input = xmalloc(bytes), *raw = xmalloc(bytes), *actual = xmalloc(bytes);
    for (unsigned i = 0; i <= TOKENS; i++) {
        tokens[i] = (int)((i * 173 + 976) % INKLING_VALID_VOCAB);
    }
    for (size_t i = 0; i < TOKENS * IK_HIDDEN; i++) {
        input[i] = ((int)(i * 37 % 1021) - 510) / 256.0f;
    }
    ds4_gpu_tensor *seeds = ds4_gpu_tensor_alloc(bytes), *reference_hidden = ds4_gpu_tensor_alloc(bytes);
    CHECK(seeds && reference_hidden && ds4_gpu_tensor_write(seeds, 0, input, bytes));
    ds4_inkling_spec spec;
    ds4_inkling_mtp_graph reference;
    CHECK(inkling_spec_alloc(&spec, mtp, weights, CONTEXT, EXTEND_CAP));
    CHECK(inkling_draft_alloc(&reference, mtp, weights, CONTEXT, TOKENS));
    /* Canary every physical KV slot, including masked future rows. KEEP0
     * must restore those bytes too, independently of prefix visibility. */
    for (unsigned depth = 0; depth < INKLING_DRAFT_LAYERS; depth++) {
        inkling_layer_state *s = &spec.draft.graph.layer[depth];
        CHECK(ds4_gpu_tensor_fill_f32(s->kv, depth + 0.25f, (uint64_t)s->capacity * IK_KV));
    }
    for (unsigned chunk_i = 0; chunk_i < sizeof(chunks) / sizeof(chunks[0]); chunk_i++) {
        CHECK(inkling_spec_reset(&spec));
        for (unsigned point = 0; point < sizeof(points) / sizeof(points[0]); point++) {
            const unsigned n = points[point];
            while (spec.position < n) {
                const unsigned left = n - spec.position;
                const unsigned step = left < chunks[chunk_i] ? left : chunks[chunk_i];
                ds4_gpu_tensor *h = ds4_gpu_tensor_view(seeds, spec.position * row_bytes, step * row_bytes);
                CHECK(h && inkling_spec_extend(&spec, main, shared, mtp, weights,
                                               h, tokens, spec.position + step, step));
                ds4_gpu_tensor_free(h);
            }
            const unsigned stable = n > INKLING_DRAFT_LAYERS ? n - INKLING_DRAFT_LAYERS : 0;
            CHECK(spec.tail_rows == n - stable);
            CHECK(ds4_gpu_tensor_read(spec.tail, 0, actual, spec.tail_rows * row_bytes));
            CHECK(memcmp(actual, input + (size_t)stable * IK_HIDDEN, spec.tail_rows * row_bytes) == 0);
            for (unsigned repeat = 0; repeat < 2; repeat++) {
                const int bonus = repeat ? 12650 : 382;
                layer_snapshot expected[INKLING_DRAFT_LAYERS], before[INKLING_DRAFT_LAYERS];
                CHECK(ds4_gpu_tensor_write(reference_hidden, 0, input, n * row_bytes));
                rotated_reference(&reference, main, shared, mtp, weights,
                                    reference_hidden, tokens, n, bonus, wanted, expected);
                for (unsigned depth = 0; depth < INKLING_DRAFT_LAYERS; depth++) {
                    CHECK(spec.draft.positions[depth] == stable);
                    layer_snapshot prefix = read_layer(&spec.draft, depth, STATE_PREFIX);
                    CHECK(prefix.bytes == expected[depth].bytes);
                    CHECK(memcmp(prefix.data, expected[depth].data, expected[depth].bytes) == 0);
                    free(prefix.data);
                    before[depth] = read_layer(&spec.draft, depth, STATE_FULL);
                }
                CHECK(inkling_spec_propose(&spec, main, shared, mtp, weights,
                                            tokens, bonus, got, INKLING_DRAFT_LAYERS));
                CHECK(memcmp(got, wanted, sizeof(got)) == 0);
                CHECK(spec.position == n && !spec.draft.graph.failed);
                CHECK(ds4_gpu_tensor_read(reference_hidden, stable * row_bytes, raw, spec.tail_rows * row_bytes));
                CHECK(ds4_gpu_tensor_read(spec.draft.graph.buf[IK_X], 0, actual, spec.tail_rows * row_bytes));
                CHECK(memcmp(raw, actual, spec.tail_rows * row_bytes) == 0);
                for (unsigned depth = 0; depth < INKLING_DRAFT_LAYERS; depth++) {
                    layer_snapshot after = read_layer(&spec.draft, depth, STATE_FULL);
                    CHECK(spec.draft.positions[depth] == stable && after.bytes == before[depth].bytes);
                    CHECK(memcmp(after.data, before[depth].data, after.bytes) == 0);
                    free(after.data); free(before[depth].data); free(expected[depth].data);
                }
            }
            printf("Inkling boundary prefix=%u chunk=%u: all8 proposals/hidden/stable state exact\n", n, chunks[chunk_i]);
        }
    }
    layer_snapshot before[INKLING_DRAFT_LAYERS];
    for (unsigned depth = 0; depth < INKLING_DRAFT_LAYERS; depth++) {
        before[depth] = read_layer(&spec.draft, depth, STATE_FULL);
    }
    for (unsigned count = 1; count <= INKLING_DRAFT_LAYERS; count++) {
        for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
            got[i] = -1;
        }
        CHECK(inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 12650, got, count));
        CHECK(memcmp(got, wanted, count * sizeof(*got)) == 0);
        for (unsigned i = count; i < INKLING_DRAFT_LAYERS; i++) {
            CHECK(got[i] == -1);
        }
    }
    for (unsigned i = 0; i < INKLING_DRAFT_LAYERS; i++) {
        got[i] = -1;
    }
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 0));
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, INKLING_DRAFT_LAYERS + 1));
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, INKLING_VALID_VOCAB, got, 1));
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, NULL, 1));
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, NULL, 382, got, 1));
    const int saved = tokens[TOKENS - 1];
    tokens[TOKENS - 1] = INKLING_VALID_VOCAB;
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 1));
    tokens[TOKENS - 1] = saved;
    tokens[TOKENS] = INKLING_VALID_VOCAB;
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, TOKENS + 1, 1));
    tokens[TOKENS] = 382;
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, TOKENS + 1, 0));
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, TOKENS + 2, 1));
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, NULL, tokens, TOKENS + 1, 1));
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, NULL, TOKENS + 1, 1));
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, CONTEXT + 1, CONTEXT + 1 - TOKENS));
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, TOKENS + EXTEND_CAP + 1, EXTEND_CAP + 1));
    spec.draft.positions[INKLING_DRAFT_LAYERS - 1]++;
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 1));
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, TOKENS + 1, 1));
    spec.draft.positions[INKLING_DRAFT_LAYERS - 1]--;
    spec.draft.graph.context = TOKENS + 1;
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 1));
    spec.draft.graph.context = TOKENS;
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 1));
    spec.draft.graph.context = CONTEXT;
    CHECK(spec.position == TOKENS && !spec.draft.graph.failed);
    CHECK(ds4_gpu_tensor_read(spec.tail, 0, actual, spec.tail_rows * row_bytes));
    CHECK(memcmp(actual, input + (TOKENS - INKLING_DRAFT_LAYERS) * IK_HIDDEN, spec.tail_rows * row_bytes) == 0);
    for (unsigned depth = 0; depth < INKLING_DRAFT_LAYERS; depth++) {
        layer_snapshot after = read_layer(&spec.draft, depth, STATE_FULL);
        CHECK(spec.draft.positions[depth] == TOKENS - INKLING_DRAFT_LAYERS);
        CHECK(after.bytes == before[depth].bytes && memcmp(after.data, before[depth].data, after.bytes) == 0);
        CHECK(got[depth] == -1);
        free(after.data); free(before[depth].data);
    }
    inkling_spec_free(&spec);
    CHECK(!inkling_spec_alloc(&spec, mtp, weights, CONTEXT, INKLING_DRAFT_LAYERS));
    CHECK(inkling_spec_alloc(&spec, mtp, weights, 3, 3));
    CHECK(inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, 1, 1));
    CHECK(inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 1));
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 2));
    CHECK(inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, 2, 1));
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 1));
    CHECK(inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, 3, 1));
    CHECK(!inkling_spec_extend(&spec, main, shared, mtp, weights, seeds, tokens, 4, 1));
    CHECK(!inkling_spec_propose(&spec, main, shared, mtp, weights, tokens, 382, got, 1));
    CHECK(spec.position == 3 && !spec.draft.graph.failed);
    puts("Inkling partial-depth proposals, reset and invalid boundary/context inputs passed");
    inkling_spec_free(&spec); inkling_draft_free(&reference);
    ds4_gpu_tensor_free(seeds); ds4_gpu_tensor_free(reference_hidden);
    free(input); free(raw); free(actual);
}

int main(int argc, char **argv) {
    CHECK(argc == 4);
    const ds4_host_shape host = {.variant = DS4_VARIANT_INKLING_SMALL};
    ds4_host_shape_install(&host); model_apply_host_shape(); ds4_host_shape_clear();
    ds4_model main_model, mtp_model;
    model_open(&main_model, argv[1], false, false);
    model_open(&mtp_model, argv[2], false, false);
    ds4_weights main_weights;
    ds4_inkling_draft mtp_weights;
    weights_bind(&main_weights, &main_model, false, 0, UINT32_MAX, true, false);
    inkling_bind_draft(&mtp_weights, &mtp_model);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(main_model.map, main_model.size));
    /* Import publishes the sidecar ranges; an auxiliary copy would overlap. */
    CHECK(ds4_gpu_import_model_ipc_manifest(mtp_model.map, mtp_model.size, argv[3], "mtp"));
    ds4_inkling_mtp_graph actual, reference;
    CHECK(inkling_draft_alloc(&actual, &mtp_model, &mtp_weights, 32, ROWS));
    CHECK(inkling_draft_alloc(&reference, &mtp_model, &mtp_weights, 32, ROWS));
    const size_t count = ROWS * IK_HIDDEN, bytes = count * sizeof(float);
    float *input = xmalloc(bytes), *raw = xmalloc(bytes), *got = xmalloc(bytes);
    for (size_t i = 0; i < count; i++) { input[i] = ((int)(i * 37 % 1021) - 510) / 256.0f; }
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_alloc(bytes), *embed = ds4_gpu_tensor_alloc(bytes);
    CHECK(hidden && embed && ds4_gpu_tensor_write(hidden, 0, input, bytes));
    const size_t logit_bytes = DS4_N_VOCAB * sizeof(float);
    float *logits = xmalloc(logit_bytes), *want = xmalloc(logit_bytes);
    int tokens[ROWS] = {200000, 976, 9029, 328, 10128, 382, 12650};
    for (unsigned depth = 0; depth < INKLING_DRAFT_LAYERS; depth++) {
        const ds4_gpu_tensor *h = depth ? actual.graph.buf[IK_X] : hidden;
        CHECK(inkling_draft_tokens(&actual, &main_model, &main_weights,
              &mtp_model, &mtp_weights, depth, h, tokens, ROWS));
        const int top = read_top(&actual.graph, logits);
        CHECK(ds4_gpu_tensor_read(actual.graph.buf[IK_X], 0, raw, bytes));
        /* Independently assemble the existing primitives to catch ordering,
         * wrong source-map and buffer-aliasing errors in the token wrapper. */
        CHECK(ds4_gpu_tensor_write(reference.graph.buf[IK_TOKENS], 0, tokens, sizeof(tokens)));
        CHECK(ds4_gpu_embed_tokens_q8_0_tensor(embed, reference.graph.buf[IK_TOKENS],
              main_model.map, main_model.size, main_weights.token_embd->abs_offset,
              DS4_N_VOCAB, ROWS, IK_HIDDEN));
        CHECK(ds4_gpu_inkling_norm(embed, embed, main_model.map, main_model.size,
              main_weights.inkling.embed_norm->abs_offset, IK_HIDDEN, ROWS));
        CHECK(inkling_draft_forward(&reference, &mtp_model, &mtp_weights, depth, hidden, embed, ROWS));
        CHECK(ds4_gpu_tensor_read(reference.graph.buf[IK_X], 0, got, bytes));
        CHECK(memcmp(raw, got, bytes) == 0);
        ds4_gpu_tensor *last = ds4_gpu_tensor_view(reference.graph.buf[IK_X],
                                                 (ROWS - 1) * IK_HIDDEN * sizeof(float), IK_HIDDEN * sizeof(float));
        CHECK(last && ds4_gpu_inkling_add_scale(reference.graph.buf[IK_NORM], last, NULL,
                                               1.0f / IK_LOGIT_DIVISOR, IK_HIDDEN));
        CHECK(plain_graph_matmul_tensor(reference.graph.logits, &main_model, main_weights.output,
                                        IK_HIDDEN, DS4_N_VOCAB, reference.graph.buf[IK_NORM], 1));
        CHECK(ds4_gpu_tensor_read(reference.graph.logits, 0, want, logit_bytes));
        CHECK(memcmp(logits, want, INKLING_VALID_VOCAB * sizeof(float)) == 0);
        ds4_gpu_tensor_free(last);
        CHECK(ds4_gpu_tensor_write(hidden, 0, got, bytes));
        memmove(tokens, tokens + 1, (ROWS - 1) * sizeof(int)); tokens[ROWS - 1] = top;
        printf("shared draft %u: top=%d, all hidden/logits exact\n", depth, top);
    }
    const unsigned position = actual.positions[0];
    tokens[ROWS - 1] = INKLING_VALID_VOCAB;
    CHECK(!inkling_draft_tokens(&actual, &main_model, &main_weights,
          &mtp_model, &mtp_weights, 0, hidden, tokens, ROWS));
    CHECK(actual.positions[0] == position && !actual.graph.failed);
    CHECK(ds4_gpu_tensor_read(actual.graph.buf[IK_X], 0, got, bytes));
    CHECK(memcmp(raw, got, bytes) == 0);
    CHECK(ds4_gpu_tensor_read(actual.graph.logits, 0, want, logit_bytes));
    CHECK(memcmp(logits, want, logit_bytes) == 0);
    inkling_draft_free(&actual); inkling_draft_free(&reference);
    ds4_gpu_tensor_free(hidden); ds4_gpu_tensor_free(embed);
    check_boundary(&main_model, &main_weights, &mtp_model, &mtp_weights);
    ds4_gpu_cleanup(); model_close(&main_model); model_close(&mtp_model);
    free(input); free(raw); free(got); free(logits); free(want);
    puts("Inkling all8 shared embedding/head gate passed");
    return 0;
}
