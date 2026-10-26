/*
 * h3_audio_encoder_probe — run one reference AudioVAE encode and hash the
 * latent the port must reproduce.
 *
 * The engine compiled here is `/data/h3.c` at eae5d54, unmodified; only this
 * driver is ours (`h3_audio_vae.h`, `h3_ffmpeg.h`).
 *
 * Usage:
 *   h3_audio_encoder_probe <FL2VA/audio_vae> <media>
 *                          [--samples N] [--shader PATH]
 *
 * `<media>` goes through the reference's own reader
 * (`h3_ffmpeg.c:392-521`): first audio stream, stereo 32 kHz F32, truncated
 * at `--samples` (default 64000, the reference test's own length,
 * `tests/test_real_audio_encoder.c:20`). `--samples` must be 64000..=480000.
 * `--shader` defaults to `h3_shaders.metal`; the CUDA backend ignores it.
 *
 * Output, in order:
 *   PCM samples= values= fnv= first= last=
 *   LATENT channels= stereo= t= values= fnv= first= last=
 *   SECONDS total= encode=
 *
 * The latent hash covers `h3_audio_latent.values`, channel-major
 * `[32,2,t]` after the released inverse normalization
 * (`h3_audio_vae.c:1253-1268`), the same bytes `h3_audio_encoder_gate`
 * hashes.
 */

#include "h3_audio_vae.h"
#include "h3_ffmpeg.h"

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

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "audio encoder stage %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    const char *shader = "h3_shaders.metal";
    int max_samples = 64000;
    const char *weights = NULL;
    const char *media = NULL;

    for (int index = 1; index < argc; index++) {
        if (!strcmp(argv[index], "--samples") && index + 1 < argc) {
            max_samples = atoi(argv[++index]);
        } else if (!strcmp(argv[index], "--shader") && index + 1 < argc) {
            shader = argv[++index];
        } else if (!weights) {
            weights = argv[index];
        } else if (!media) {
            media = argv[index];
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[index]);
            return 2;
        }
    }

    if (!weights || !media || max_samples < 64000 || max_samples > 480000) {
        fprintf(stderr,
                "usage: %s <FL2VA/audio_vae> <media> [--samples N] "
                "[--shader PATH]\n",
                argv[0]);
        return 2;
    }

    char error[512];
    float *pcm = NULL;
    int samples = 0;

    if (!h3_ffmpeg_read_audio_f32(media, max_samples, 1, &pcm, &samples,
                                  error, sizeof(error))) {
        fprintf(stderr, "read audio: %s\n", error);
        return 1;
    }

    size_t pcm_values = (size_t)samples * 2;
    printf("PCM samples=%d values=%zu fnv=%016llx first=%.9f last=%.9f\n",
           samples, pcm_values,
           (unsigned long long)fnv(pcm, pcm_values * sizeof(*pcm)), pcm[0],
           pcm[pcm_values - 1]);

    h3_audio_latent latent;
    memset(&latent, 0, sizeof(latent));
    double started = seconds();
    int ok = h3_audio_vae_encode(weights, shader, pcm, samples, progress, NULL,
                                 &latent, error, sizeof(error));
    double encode_seconds = seconds() - started;

    if (!ok) {
        fprintf(stderr, "encode: %s\n", error);
        free(pcm);
        return 1;
    }

    size_t latent_values = (size_t)latent.channels * (size_t)latent.stereo *
                           (size_t)latent.length;
    printf("LATENT channels=%d stereo=%d t=%d values=%zu fnv=%016llx "
           "first=%.9f last=%.9f\n",
           latent.channels, latent.stereo, latent.length, latent_values,
           (unsigned long long)fnv(latent.values,
                                   latent_values * sizeof(*latent.values)),
           latent.values[0], latent.values[latent_values - 1]);
    printf("SECONDS encode=%.3f\n", encode_seconds);

    h3_audio_latent_free(&latent);
    free(pcm);
    return 0;
}
