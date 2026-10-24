/*
 * h3_dit_probe — run one reference FL2VA DiT forward and hash everything the
 * port must reproduce.
 *
 * The engine compiled here is `/data/h3.c` at eae5d54, unmodified; only this
 * driver is ours. The prompt encoder is the same one the E4 gate uses, so the
 * probe also pins that the two units chain.
 *
 * Usage:
 *   h3_dit_probe <text_encoder_dir> <tokenizer.json> <FL2VA/transformer>
 *                <prompt> [--width W] [--height H] [--frames F]
 *                [--steps N] [--step S] [--seed N] [--layers N]
 *                [--rope-scale X]
 *   h3_dit_probe <FL2VA/transformer> --adaln-only [--steps N]
 *   h3_dit_probe --features [--steps N]
 *
 * Output of the forward mode, in order:
 *   TOKENS ...                       prompt ids from the reference BPE
 *   TEXT ...                         the encoder's BF16 output hash (E4)
 *   LAYOUT rows= segments= positions=<fnv> targets=...   packed sequence
 *   SIGMAS ...                       the serving sigma grid, as bits
 *   NOISE video=<fnv> audio=<fnv>    seed-derived latents fed to the forward
 *   FORWARD video=<fnv> audio=<fnv>  the velocity the port must match
 *   LOAD / TIME with the wall clock per phase
 *
 * `--adaln-only` skips the prompt encoder and the core: it precomputes the
 * AdaLN schedule alone and prints
 *   ADALN rows= block0=<fnv> final=<fnv>
 * which is the cheapest gate that covers the timestep embedding, every
 * `adaln_proj` and the final projection, and it isolates a modulation defect
 * from a block defect. The DiT owns its own schedule and exposes no accessor,
 * so this mode runs the reference's own precompute a second time
 * (`h3_dit_schedule.c:235-331`); the cost is the 26 GiB of `adaln_proj`
 * weights, not the core.
 *
 * `--features` prints the timestep sinusoid table alone, `FEATURES rows= fnv=
 * times= first=`, with no weights and no device: it separates host feature
 * arithmetic from the device chain.
 *
 * The hashes cover the exact F32 bytes the reference hands to the caller:
 * the noise latents before the forward and the velocities after it, both in
 * the [C,T,H,W] / [32,2,T] layouts of h3_dit.h:93-98.
 */

#include "h3_dit.h"
#include "h3_dit_schedule.h"
#include "h3_host.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* `h3_dit.c:15-30`, the geometry the reference compiles in. */
enum {
    DIT_HIDDEN = 5376,
    DIT_BLOCKS = 50,
    DIT_VIDEO_CHANNELS = 24,
    DIT_AUDIO_CHANNELS = 32,
    DIT_AUDIO_STREAMS = 2,
    DIT_VIDEO_PATCH = 96
};

/* `h3_dit_schedule.c:11-16`: 3 modalities x 6 slots x HIDDEN, and 2 x HIDDEN. */
enum {
    DIT_TIME_INPUT = 256,
    DIT_TIME_HIDDEN = 5376,
    DIT_TIME_DIM = 2688,
    DIT_BLOCK_OUTPUT = 3 * 6 * DIT_HIDDEN,
    DIT_FINAL_OUTPUT = 2 * DIT_HIDDEN
};

/* Same seed and prime as tools/minimax_h3/text_probe.c:37-38, which records
 * why: the probe family hashes with the reference's nonstandard seed. */
static const uint64_t FNV_SEED = UINT64_C(1469598103934665603);
static const uint64_t FNV_PRIME = UINT64_C(1099511628211);

static uint64_t fnv(const void *data, size_t bytes) {
    const unsigned char *values = data;
    uint64_t hash = FNV_SEED;

    for (size_t index = 0; index < bytes; index++) {
        hash ^= values[index];
        hash *= FNV_PRIME;
    }

    return hash;
}

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static int parse_int(const char *text, int *value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (!end || *end || parsed < INT32_MIN || parsed > INT32_MAX) return 0;
    *value = (int)parsed;
    return 1;
}

static void usage(void) {
    fprintf(stderr,
            "usage: h3_dit_probe <text_encoder_dir> <tokenizer.json> "
            "<FL2VA/transformer> <prompt>\n"
            "       [--width W] [--height H] [--frames F] [--steps N] "
            "[--step S]\n"
            "       [--seed N] [--layers N] [--rope-scale X]\n"
            "       h3_dit_probe <FL2VA/transformer> --adaln-only [--steps N]\n"
            "       h3_dit_probe --features [--steps N]\n");
}

/* Hash a BF16 tensor the way the reference reads it back: BF16 word -> F32
 * with the low half zero (`h3_dit.c:2474-2483`). */
static uint64_t fnv_bf16_to_f32(const h3_gpu_tensor *tensor, size_t elements) {
    uint16_t *words = malloc(elements * sizeof(*words));
    if (!words) return 0;

    uint64_t hash = 0;
    if (h3_gpu_tensor_read_bf16(tensor, words, elements)) {
        float *values = malloc(elements * sizeof(*values));
        if (values) {
            for (size_t index = 0; index < elements; index++) {
                uint32_t bits = (uint32_t)words[index] << 16;
                memcpy(&values[index], &bits, sizeof(bits));
            }
            hash = fnv(values, elements * sizeof(*values));
            free(values);
        }
    }

    free(words);
    return hash;
}

/* The AdaLN gate: timestep features -> time_embedder -> every
 * `blocks.N.adaln_proj` -> `final_layer.adaln_proj`, hashed as the reference
 * stores them (`h3_dit_schedule.c:235-331`). */
static int adaln_only(int count, char **args, const char *directory) {
    int steps = 2;
    for (int index = 0; index + 1 < count; index += 2) {
        if (!strcmp(args[index], "--steps")) {
            if (!parse_int(args[index + 1], &steps)) return 2;
        } else {
            usage();
            return 2;
        }
    }

    char error[1024];
    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(steps, &sigmas)) {
        fprintf(stderr, "h3_dit_probe: steps %d is not a legal schedule\n",
                steps);
        return 1;
    }

    double started = seconds();
    h3_weight_store *weights = h3_weight_store_open(directory, error,
                                                    sizeof(error));
    h3_gpu *gpu = weights ? h3_gpu_create("h3_shaders.metal", error,
                                          sizeof(error)) : NULL;
    h3_dit_schedule *schedule = gpu ? h3_dit_schedule_precompute(
        weights, gpu, &sigmas, 0, 0, NULL, NULL, error, sizeof(error)) : NULL;
    if (!schedule) {
        fprintf(stderr, "%s\n", error);
        if (gpu) h3_gpu_free(gpu);
        if (weights) h3_weight_store_free(weights);
        return 1;
    }

    size_t time_rows = h3_dit_schedule_time_rows(schedule);
    uint64_t block0 = fnv_bf16_to_f32(
        h3_dit_schedule_block(schedule, 0), time_rows * DIT_BLOCK_OUTPUT);
    uint64_t final = fnv_bf16_to_f32(h3_dit_schedule_final(schedule),
                                     time_rows * DIT_FINAL_OUTPUT);
    printf("ADALN rows=%zu block0=%016llx final=%016llx seconds=%.3f\n",
           time_rows, (unsigned long long)block0,
           (unsigned long long)final, seconds() - started);

    h3_dit_schedule_free(schedule);
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    return 0;
}

/* `h3_dit_schedule.c:79-160` `prepare_rows` for a T2VA schedule: the timestep
 * row assignment and the time each row carries. Returns the row count. */
static uint32_t probe_rows(const h3_sigma_schedule *sigmas, uint32_t *video,
                           uint32_t *audio, float *times) {
    uint32_t count = 0;
    for (int step = 0; step < sigmas->steps; step++) {
        float video_time = 1.0f - sigmas->video[step];
        float audio_time = 1.0f - sigmas->audio[step];
        if (video_time == audio_time) {
            video[step] = count;
            audio[step] = count++;
        } else if (video_time < audio_time) {
            video[step] = count++;
            audio[step] = count++;
        } else {
            audio[step] = count++;
            video[step] = count++;
        }
    }
    for (int step = 0; step < sigmas->steps; step++) {
        times[video[step]] = 1.0f - sigmas->video[step];
        times[audio[step]] = 1.0f - sigmas->audio[step];
    }
    return count;
}

/* `h3_dit_schedule.c:147-157`: the sinusoid table the timestep embedder reads.
 * This mode exists to separate host feature arithmetic from the device chain,
 * so it prints bits rather than decimals. */
static int features_only(int count, char **args) {
    int steps = 2;
    for (int index = 0; index + 1 < count; index += 2) {
        if (!strcmp(args[index], "--steps")) {
            if (!parse_int(args[index + 1], &steps)) return 2;
        } else {
            usage();
            return 2;
        }
    }

    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(steps, &sigmas)) {
        fprintf(stderr, "h3_dit_probe: steps %d is not a legal schedule\n",
                steps);
        return 1;
    }

    uint32_t video[H3_MAX_STEPS], audio[H3_MAX_STEPS];
    float times[H3_MAX_STEPS * 2 + 2];
    memset(times, 0, sizeof(times));
    uint32_t rows = probe_rows(&sigmas, video, audio, times);

    float *features = calloc((size_t)rows * DIT_TIME_INPUT, sizeof(*features));
    if (!features) return 1;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t index = 0; index < DIT_TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) *
                                   (float)index / (float)(DIT_TIME_INPUT / 2));
            float angle = times[row] * frequency;
            features[(size_t)row * DIT_TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * DIT_TIME_INPUT + DIT_TIME_INPUT / 2 + index] =
                sinf(angle);
        }
    }

    printf("FEATURES rows=%u fnv=%016llx", rows,
           (unsigned long long)fnv(features,
                                   (size_t)rows * DIT_TIME_INPUT *
                                       sizeof(*features)));
    printf(" times=");
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t bits;
        memcpy(&bits, &times[row], sizeof(bits));
        printf("%s%08x", row ? "." : "", bits);
    }
    uint32_t first;
    memcpy(&first, &features[0], sizeof(first));
    printf(" first=%08x\n", first);

    free(features);
    return 0;
}

/* The same chain `h3_dit_schedule_precompute` runs, rebuilt here through the
 * public GPU API so each stage can be hashed: the schedule keeps only the
 * final projections. Any disagreement between `--adaln-only` and these
 * numbers would mean this transcription is wrong, not the port. */
static int adaln_stages(const char *directory, int steps) {
    char error[1024];
    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(steps, &sigmas)) {
        fprintf(stderr, "h3_dit_probe: steps %d is not a legal schedule\n",
                steps);
        return 1;
    }

    uint32_t video[H3_MAX_STEPS], audio[H3_MAX_STEPS];
    float times[H3_MAX_STEPS * 2 + 2];
    memset(times, 0, sizeof(times));
    uint32_t rows = probe_rows(&sigmas, video, audio, times);
    size_t feature_count = (size_t)rows * DIT_TIME_INPUT;
    float *features = calloc(feature_count, sizeof(*features));
    if (!features) return 1;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t index = 0; index < DIT_TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) *
                                   (float)index / (float)(DIT_TIME_INPUT / 2));
            float angle = times[row] * frequency;
            features[(size_t)row * DIT_TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * DIT_TIME_INPUT + DIT_TIME_INPUT / 2 + index] =
                sinf(angle);
        }
    }
    printf("FEATURES rows=%u fnv=%016llx\n", rows,
           (unsigned long long)fnv(features,
                                   feature_count * sizeof(*features)));

    h3_weight_store *weights = h3_weight_store_open(directory, error,
                                                    sizeof(error));
    h3_gpu *gpu = weights ? h3_gpu_create("h3_shaders.metal", error,
                                          sizeof(error)) : NULL;
    if (!gpu) {
        fprintf(stderr, "%s\n", error);
        if (weights) h3_weight_store_free(weights);
        return 1;
    }

    uint64_t in_shape[] = {DIT_TIME_HIDDEN, DIT_TIME_INPUT};
    uint64_t out_shape[] = {DIT_TIME_DIM, DIT_TIME_HIDDEN};
    h3_gpu_tensor *in_w = h3_weight_load_f32(weights, gpu,
        "time_embedder.proj_in.weight", 2, in_shape, error, sizeof(error));
    h3_gpu_tensor *in_b = NULL, *out_w = NULL, *out_b = NULL;
    if (in_w) in_b = h3_weight_load_f32(weights, gpu,
        "time_embedder.proj_in.bias", 1, (uint64_t[]){DIT_TIME_HIDDEN},
        error, sizeof(error));
    if (in_b) out_w = h3_weight_load_f32(weights, gpu,
        "time_embedder.proj_out.weight", 2, out_shape, error, sizeof(error));
    if (out_w) out_b = h3_weight_load_f32(weights, gpu,
        "time_embedder.proj_out.bias", 1, (uint64_t[]){DIT_TIME_DIM},
        error, sizeof(error));
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, features, feature_count);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * DIT_TIME_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * DIT_TIME_HIDDEN);
    h3_gpu_tensor *projected = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * DIT_TIME_DIM);
    h3_gpu_tensor *rounded = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * DIT_TIME_DIM);
    h3_gpu_tensor *embedding = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * DIT_TIME_DIM);

    int ok = input && hidden && activated && projected && rounded &&
        embedding && in_w && in_b && out_w && out_b &&
        h3_gpu_begin(gpu) &&
        h3_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                          DIT_TIME_INPUT, DIT_TIME_HIDDEN) &&
        h3_gpu_silu_f32(gpu, activated, hidden,
                        rows * DIT_TIME_HIDDEN) &&
        h3_gpu_linear_f32(gpu, projected, activated, out_w, out_b, rows,
                          DIT_TIME_HIDDEN, DIT_TIME_DIM) &&
        h3_gpu_cast_f32_to_bf16(gpu, rounded, projected,
                                rows * DIT_TIME_DIM) &&
        h3_gpu_silu_bf16(gpu, embedding, rounded, rows * DIT_TIME_DIM) &&
        h3_gpu_submit(gpu);
    if (!ok) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    printf("EMBED rows=%u fnv=%016llx first=%016llx\n", rows,
           (unsigned long long)fnv_bf16_to_f32(embedding,
                                               (size_t)rows * DIT_TIME_DIM),
           (unsigned long long)fnv_bf16_to_f32(embedding, 1));

    uint64_t block_shape[] = {DIT_BLOCK_OUTPUT, DIT_TIME_DIM};
    h3_gpu_tensor *block_w = h3_weight_load_bf16(weights, gpu,
        "blocks.0.adaln_proj.linear.weight", 2, block_shape, error,
        sizeof(error));
    h3_gpu_tensor *block_b = block_w ? h3_weight_load_bf16(weights, gpu,
        "blocks.0.adaln_proj.linear.bias", 1, (uint64_t[]){DIT_BLOCK_OUTPUT},
        error, sizeof(error)) : NULL;
    h3_gpu_tensor *modulation = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * DIT_BLOCK_OUTPUT);
    ok = block_w && block_b && modulation && h3_gpu_begin(gpu) &&
        h3_gpu_linear_bf16(gpu, modulation, embedding, block_w, block_b, rows,
                           DIT_TIME_DIM, DIT_BLOCK_OUTPUT) &&
        h3_gpu_submit(gpu);
    if (!ok) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    printf("BLOCK rows=%u fnv=%016llx\n", rows,
           (unsigned long long)fnv_bf16_to_f32(
               modulation, (size_t)rows * DIT_BLOCK_OUTPUT));

    uint64_t final_shape[] = {DIT_FINAL_OUTPUT, DIT_TIME_DIM};
    h3_gpu_tensor *final_w = h3_weight_load_bf16(weights, gpu,
        "final_layer.adaln_proj.linear.weight", 2, final_shape, error,
        sizeof(error));
    h3_gpu_tensor *final_b = final_w ? h3_weight_load_bf16(weights, gpu,
        "final_layer.adaln_proj.linear.bias", 1, (uint64_t[]){DIT_FINAL_OUTPUT},
        error, sizeof(error)) : NULL;
    h3_gpu_tensor *final_modulation = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * DIT_FINAL_OUTPUT);
    ok = final_w && final_b && final_modulation && h3_gpu_begin(gpu) &&
        h3_gpu_linear_bf16(gpu, final_modulation, embedding, final_w, final_b,
                           rows, DIT_TIME_DIM, DIT_FINAL_OUTPUT) &&
        h3_gpu_submit(gpu);
    if (!ok) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    printf("FINAL rows=%u fnv=%016llx\n", rows,
           (unsigned long long)fnv_bf16_to_f32(
               final_modulation, (size_t)rows * DIT_FINAL_OUTPUT));

    free(features);
    h3_gpu_tensor_free(final_modulation);
    h3_gpu_tensor_free(final_b);
    h3_gpu_tensor_free(final_w);
    h3_gpu_tensor_free(modulation);
    h3_gpu_tensor_free(block_b);
    h3_gpu_tensor_free(block_w);
    h3_gpu_tensor_free(embedding);
    h3_gpu_tensor_free(rounded);
    h3_gpu_tensor_free(projected);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(out_b);
    h3_gpu_tensor_free(out_w);
    h3_gpu_tensor_free(in_b);
    h3_gpu_tensor_free(in_w);
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--features") == 0) {
        return features_only(argc - 2, argv + 2);
    }
    if (argc >= 3 && strcmp(argv[1], "--adaln-stages") == 0) {
        int steps = 2;
        if (argc >= 5 && strcmp(argv[3], "--steps") == 0 &&
            !parse_int(argv[4], &steps)) return 2;
        return adaln_stages(argv[2], steps);
    }
    if (argc >= 3 && strcmp(argv[2], "--adaln-only") == 0) {
        return adaln_only(argc - 3, argv + 3, argv[1]);
    }

    if (argc < 5) {
        usage();
        return 2;
    }

    int width = 256, height = 256, frames = 5;
    int steps = 2, step = 0, layers = DIT_BLOCKS;
    unsigned long long seed = 1;
    float rope_scale = 1.0f;

    for (int index = 5; index + 1 < argc; index += 2) {
        if (!strcmp(argv[index], "--width")) {
            if (!parse_int(argv[index + 1], &width)) return 2;
        } else if (!strcmp(argv[index], "--height")) {
            if (!parse_int(argv[index + 1], &height)) return 2;
        } else if (!strcmp(argv[index], "--frames")) {
            if (!parse_int(argv[index + 1], &frames)) return 2;
        } else if (!strcmp(argv[index], "--steps")) {
            if (!parse_int(argv[index + 1], &steps)) return 2;
        } else if (!strcmp(argv[index], "--step")) {
            if (!parse_int(argv[index + 1], &step)) return 2;
        } else if (!strcmp(argv[index], "--layers")) {
            if (!parse_int(argv[index + 1], &layers)) return 2;
        } else if (!strcmp(argv[index], "--seed")) {
            char *end = NULL;
            seed = strtoull(argv[index + 1], &end, 10);
            if (!end || *end) return 2;
        } else if (!strcmp(argv[index], "--rope-scale")) {
            char *end = NULL;
            rope_scale = strtof(argv[index + 1], &end);
            if (!end || *end || !(rope_scale > 0.0f)) return 2;
        } else {
            usage();
            return 2;
        }
    }

    char error[1024];
    uint32_t *ids = NULL;
    size_t token_count = 0;
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[2], error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (!h3_tokenizer_encode(tokenizer, argv[4], 1, &ids, &token_count, error,
                             sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        h3_tokenizer_free(tokenizer);
        return 1;
    }
    h3_tokenizer_free(tokenizer);

    printf("TOKENS");
    for (size_t index = 0; index < token_count; index++)
        printf(" %u", ids[index]);
    putchar('\n');

    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    double started = seconds();
    if (!h3_text_encode_bf16(argv[1], "h3_shaders.metal", ids, token_count,
                             NULL, NULL, &text, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        free(ids);
        return 1;
    }
    double text_seconds = seconds() - started;
    size_t text_values = text.tokens * text.width;
    printf("TEXT tokens=%zu width=%zu fnv=%016llx first=%04x.%04x "
           "last=%04x.%04x seconds=%.3f\n",
           text.tokens, text.width,
           (unsigned long long)fnv(text.values,
                                   text_values * sizeof(*text.values)),
           text.values[0], text.values[1],
           text.values[text_values - 2], text.values[text_values - 1],
           text_seconds);
    h3_tokenizer_ids_free(ids);
    ids = NULL;

    h3_temporal_shape temporal = h3_temporal(frames);
    if (temporal.frame_count != frames) {
        fprintf(stderr, "h3_dit_probe: frames %d align to %d\n", frames,
                temporal.frame_count);
    }
    int latent_w = 0, latent_h = 0;
    h3_latent_canvas(width, height, &latent_w, &latent_h);

    h3_layout_spec spec = {(int)text.tokens, temporal.video_t, latent_h,
                           latent_w, temporal.audio_t, temporal.frame_count,
                           NULL, 0, NULL, 0};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        h3_text_embedding_free(&text);
        return 1;
    }
    printf("LAYOUT rows=%zu segments=%zu positions=%016llx cond=%zu+%zu "
           "target=%zu+%zu latent=%dx%dx%d audio_t=%d\n",
           layout.seq_len, layout.segment_count,
           (unsigned long long)fnv(layout.positions,
                                   layout.seq_len * sizeof(*layout.positions)),
           layout.img_cond_rows, layout.audio_cond_rows,
           layout.img_target_rows, layout.audio_target_rows,
           temporal.video_t, latent_h, latent_w, temporal.audio_t);

    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(steps, &sigmas)) {
        fprintf(stderr, "h3_dit_probe: steps %d is not a legal schedule\n",
                steps);
        h3_layout_free(&layout);
        h3_text_embedding_free(&text);
        return 1;
    }
    /* Bit patterns, not decimals: the port gates the schedule exactly, and a
     * last-bit difference here moves every timestep embedding. */
    printf("SIGMAS steps=%d", steps);
    printf(" video=");
    for (int index = 0; index < steps; index++) {
        uint32_t bits;
        memcpy(&bits, &sigmas.video[index], sizeof(bits));
        printf("%s%08x", index ? "." : "", bits);
    }
    printf(" audio=");
    for (int index = 0; index < steps; index++) {
        uint32_t bits;
        memcpy(&bits, &sigmas.audio[index], sizeof(bits));
        printf("%s%08x", index ? "." : "", bits);
    }
    putchar('\n');

    started = seconds();
    h3_dit *dit = h3_dit_load_t2va(
        argv[3], "h3_shaders.metal", &text, &layout, &sigmas,
        (unsigned)layers, 1, 0, 0, rope_scale,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        NULL, NULL, error, sizeof(error));
    if (!dit) {
        fprintf(stderr, "%s\n", error);
        h3_layout_free(&layout);
        h3_text_embedding_free(&text);
        return 1;
    }
    printf("LOAD seconds=%.3f blocks=%d step=%d of %d sigma_v=%.9g "
           "sigma_a=%.9g\n",
           seconds() - started, layers, step, steps, sigmas.video[step],
           sigmas.audio[step]);
    h3_text_embedding_free(&text);

    size_t video_elements = h3_dit_video_elements(dit);
    size_t audio_elements = h3_dit_audio_elements(dit);
    float *video = malloc(video_elements * sizeof(*video));
    float *audio = malloc(audio_elements * sizeof(*audio));
    float *video_velocity = malloc(video_elements * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_elements * sizeof(*audio_velocity));
    if (!video || !audio || !video_velocity || !audio_velocity) {
        fprintf(stderr, "h3_dit_probe: out of memory\n");
        return 1;
    }

    /* The released server seeds each modality from its own generator with the
     * same value (`h3.c:1574-1581`). */
    h3_rng video_rng, audio_rng;
    h3_rng_seed(&video_rng, seed);
    h3_rng_seed(&audio_rng, seed);
    h3_rng_fill_normal(&video_rng, video, video_elements);
    h3_rng_fill_normal(&audio_rng, audio, audio_elements);
    printf("NOISE seed=%llu video=%zu=%016llx audio=%zu=%016llx "
           "video_edge=%.9g/%.9g audio_edge=%.9g/%.9g\n",
           seed, video_elements,
           (unsigned long long)fnv(video, video_elements * sizeof(*video)),
           audio_elements,
           (unsigned long long)fnv(audio, audio_elements * sizeof(*audio)),
           video[0], video[video_elements - 1], audio[0],
           audio[audio_elements - 1]);

    started = seconds();
    if (!h3_dit_forward(dit, step, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    double forward_seconds = seconds() - started;

    printf("FORWARD video=%016llx audio=%016llx "
           "video_edge=%.9g/%.9g audio_edge=%.9g/%.9g\n",
           (unsigned long long)fnv(video_velocity,
                                   video_elements * sizeof(*video_velocity)),
           (unsigned long long)fnv(audio_velocity,
                                   audio_elements * sizeof(*audio_velocity)),
           video_velocity[0], video_velocity[video_elements - 1],
           audio_velocity[0], audio_velocity[audio_elements - 1]);
    printf("TIME seconds=%.3f\n", forward_seconds);

    free(video);
    free(audio);
    free(video_velocity);
    free(audio_velocity);
    h3_dit_free(dit);
    h3_layout_free(&layout);
    return 0;
}
