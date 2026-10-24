/*
 * h3_video_encoder_probe — run one reference VideoVAE encode and hash the
 * latent the port must reproduce.
 *
 * The engine compiled here is `/data/h3.c` at eae5d54, unmodified; only this
 * driver is ours (`h3_video_encoder.h`).
 *
 * Usage:
 *   h3_video_encoder_probe <FL2VA/video_vae/source> <media>
 *                          [--height H] [--width W] [--frames N]
 *                          [--shader PATH]
 *
 * `<media>` goes through the reference's own reader, the same one the port
 * calls: one image at `--frames 1` (`h3_ffmpeg.c:143-283`) and a 24 fps clip
 * at five frames and up (`:285-393`, trimmed to the released 5+17k cadence).
 * `--shader` defaults to `h3_shaders.metal`; the CUDA backend ignores it
 * (`h3_video_encoder.c:745`, the Metal graph alone reads it).
 *
 * Output, in order:
 *   PIXELS frames= height= width= fnv= first= last=
 *   LATENT t= h= w= values= fnv= first= last=
 *   SECONDS total= encode=
 *
 * The hash covers `h3_video_latent.values`, channel-major `[24,t,h,w]` after
 * the released normalization (`h3_video_encoder.c:722-741`), so it is the
 * same bytes `h3_video_encoder_gate` hashes. A tile count is printed last so
 * the tiling path a run took is visible in the log.
 */

#include "h3_ffmpeg.h"
#include "h3_video_encoder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

/* One line per tile keeps the tiling path visible without drowning the log. */
static void progress(int completed, int total, void *opaque) {
    int *seen = opaque;

    *seen = completed;
    if (completed == total || completed % 8 == 0)
        fprintf(stderr, "tile %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    int height = 192, width = 512, frames = 1;
    const char *shader = "h3_shaders.metal";
    const char *positional[2] = {NULL, NULL};
    int count = 0;

    for (int index = 1; index < argc; index++) {
        const char *argument = argv[index];
        const char *value = index + 1 < argc ? argv[index + 1] : NULL;

        if (!strcmp(argument, "--height")) {
            if (!value || !parse_int(value, &height)) return 2;
            index++;
        } else if (!strcmp(argument, "--width")) {
            if (!value || !parse_int(value, &width)) return 2;
            index++;
        } else if (!strcmp(argument, "--frames")) {
            if (!value || !parse_int(value, &frames)) return 2;
            index++;
        } else if (!strcmp(argument, "--shader")) {
            if (!value) return 2;
            shader = value;
            index++;
        } else if (!strncmp(argument, "--", 2)) {
            fprintf(stderr, "unknown option %s\n", argument);
            return 2;
        } else if (count < 2) {
            positional[count++] = argument;
        } else {
            return 2;
        }
    }

    if (count != 2) {
        fprintf(stderr,
                "usage: h3_video_encoder_probe <FL2VA/video_vae/source> "
                "<media> [--height H] [--width W] [--frames N] "
                "[--shader PATH]\n");
        return 2;
    }

    char error[512];
    float *pixels = NULL;
    if (frames == 1) {
        if (!h3_ffmpeg_read_image_f32(positional[1], width, height,
                                      H3_IMAGE_FIT_STRETCH, &pixels,
                                      error, sizeof(error))) {
            fprintf(stderr, "read image: %s\n", error);
            return 1;
        }
    } else {
        if (frames < 5) {
            fprintf(stderr, "read video: --frames must be 1 or at least 5\n");
            return 2;
        }
        if (!h3_ffmpeg_read_video_f32(positional[1], width, height, frames,
                                      &pixels, &frames, error,
                                      sizeof(error))) {
            fprintf(stderr, "read video: %s\n", error);
            return 1;
        }
    }

    size_t pixel_elements = (size_t)frames * (size_t)height * (size_t)width * 3;
    printf("PIXELS frames=%d height=%d width=%d fnv=%016llx first=%.9f last=%.9f\n",
           frames, height, width,
           (unsigned long long)fnv(pixels, pixel_elements * sizeof(*pixels)),
           pixels[0], pixels[pixel_elements - 1]);

    h3_video_latent latent;
    memset(&latent, 0, sizeof(latent));
    int seen = 0;
    double started = seconds();
    if (!h3_video_vae_encode(positional[0], shader, pixels, frames, height,
                             width, progress, &seen, &latent, error,
                             sizeof(error))) {
        fprintf(stderr, "encode: %s\n", error);
        free(pixels);
        return 1;
    }
    double encode_seconds = seconds() - started;

    size_t latent_elements = (size_t)latent.time * (size_t)latent.height *
                             (size_t)latent.width * 24;
    printf("LATENT t=%d h=%d w=%d values=%zu fnv=%016llx first=%.9f last=%.9f\n",
           latent.time, latent.height, latent.width, latent_elements,
           (unsigned long long)fnv(latent.values,
                                   latent_elements * sizeof(*latent.values)),
           latent.values[0], latent.values[latent_elements - 1]);

    /* tile_axis_build is static in the reference, so the plan is restated from
     * its own rule (h3_video_encoder.c:392-424) rather than read out. */
    int y_tiles = height <= 256 ? 1 : (height + 255) / 256;
    while (256 * y_tiles - 64 * (y_tiles - 1) < height) y_tiles++;
    int x_tiles = width <= 256 ? 1 : (width + 255) / 256;
    while (256 * x_tiles - 64 * (x_tiles - 1) < width) x_tiles++;

    printf("SECONDS encode=%.3f tiles=%d\n", encode_seconds, y_tiles * x_tiles);
    if (seen != y_tiles * x_tiles)
        printf("WARNING last_progress=%d expected=%d\n", seen, y_tiles * x_tiles);

    h3_video_latent_free(&latent);
    free(pixels);
    return 0;
}
