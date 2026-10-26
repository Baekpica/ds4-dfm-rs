/*
 * h3_reference_canvas_probe — print the reference's reference-media canvas
 * policy over a fixed sweep.
 *
 * The P14.3 gate replays this table with the port's own
 * `minimax_h3::host::{adapt_canvas, reference_image_canvas,
 * reference_video_canvas}`. The policy compiled here is `/data/h3.c` at
 * eae5d54 (`h3_host.c:49-124`), unmodified; only this driver is ours.
 *
 * Usage: h3_reference_canvas_probe
 * Output: three TSV row kinds over the sweep below, then TOTAL.
 *
 *   ADAPT <src_w> <src_h> <out_w> <out_h>
 *   IMAGE <src_w> <src_h> <tgt_w> <tgt_h> <cap> <out_w> <out_h>
 *   VIDEO <src_w> <src_h> <out_w> <out_h>
 *
 * A refusal prints the same row with `-` in place of the output pair.
 */

#include "h3_host.h"

#include <stdio.h>

/* Sources: the platform's own shapes, portrait mirrors, tiny and extreme
 * aspects, and one odd pair to exercise the rounding. */
static const int SOURCES[][2] = {
    {1344, 768}, {768, 1344}, {1920, 1080}, {1080, 1920}, {1280, 720},
    {720, 1280}, {640, 480},  {480, 640},  {512, 512},  {256, 256},
    {3000, 1000}, {1000, 3000}, {3840, 2160}, {2160, 3840}, {100, 100},
    {17, 9993}, {999, 101},
};

/* Targets: the two 768P canvases, the tier canvas and a small one. */
static const int TARGETS[][2] = {
    {1344, 768}, {768, 1344}, {512, 512}, {256, 256},
};

/* `match` is cap 0, `max` is the reference's 2048 short-edge cap. */
static const int CAPS[] = {0, 2048};

int main(void) {
    const int sources = (int)(sizeof(SOURCES) / sizeof(SOURCES[0]));
    const int targets = (int)(sizeof(TARGETS) / sizeof(TARGETS[0]));
    const int caps = (int)(sizeof(CAPS) / sizeof(CAPS[0]));
    int rows = 0;

    for (int index = 0; index < sources; index++) {
        int width = 0, height = 0;
        if (h3_adapt_canvas(SOURCES[index][0], SOURCES[index][1],
                            &width, &height)) {
            printf("ADAPT\t%d\t%d\t%d\t%d\n", SOURCES[index][0],
                   SOURCES[index][1], width, height);
        } else {
            printf("ADAPT\t%d\t%d\t-\t-\n", SOURCES[index][0],
                   SOURCES[index][1]);
        }
        if (h3_reference_video_canvas(SOURCES[index][0], SOURCES[index][1],
                                      &width, &height)) {
            printf("VIDEO\t%d\t%d\t%d\t%d\n", SOURCES[index][0],
                   SOURCES[index][1], width, height);
        } else {
            printf("VIDEO\t%d\t%d\t-\t-\n", SOURCES[index][0],
                   SOURCES[index][1]);
        }
        rows += 2;
    }

    for (int index = 0; index < sources; index++) {
        for (int target = 0; target < targets; target++) {
            for (int cap = 0; cap < caps; cap++) {
                int width = 0, height = 0;
                int ok = h3_reference_image_canvas(
                    SOURCES[index][0], SOURCES[index][1], TARGETS[target][0],
                    TARGETS[target][1], CAPS[cap], &width, &height);
                if (ok) {
                    printf("IMAGE\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
                           SOURCES[index][0], SOURCES[index][1],
                           TARGETS[target][0], TARGETS[target][1], CAPS[cap],
                           width, height);
                } else {
                    printf("IMAGE\t%d\t%d\t%d\t%d\t%d\t-\t-\n",
                           SOURCES[index][0], SOURCES[index][1],
                           TARGETS[target][0], TARGETS[target][1], CAPS[cap]);
                }
                rows++;
            }
        }
    }

    printf("TOTAL\trows=%d\n", rows);
    return 0;
}
