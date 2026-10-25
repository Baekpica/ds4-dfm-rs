/*
 * h3_ppm_diff — compare two RGB24 PPM files pixel by pixel.
 *
 * The reference CLI writes generated frames and step previews as PPM
 * (`main.c:162-176`, `:196-245`); this driver says exactly how two of those
 * files differ, so a render difference can be attributed to a region instead
 * of only to a whole-frame hash.
 *
 * Usage:
 *   h3_ppm_diff <a.ppm> <b.ppm>
 *
 * Output:
 *   SIZE width height                  both files agree on the geometry
 *   PIXELS different=<n> of <total> max=<n> rms=<x>
 *   PSNR decibels=<x>                  against an 8-bit full scale
 *   BBOX x0=<n> y0=<n> x1=<n> y1=<n>   every differing pixel (inclusive)
 *   BAND axis=x start=<n> different=<n>   one line per 128-pixel band
 *
 * Exit 0 when the files are identical, 1 when they differ, 2 on a read or
 * geometry error.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BAND 128
#define MAX_BANDS 64

static unsigned char *read_ppm(const char *path, int *width, int *height,
                               char *error, size_t error_size) {
    FILE *input = fopen(path, "rb");
    if (!input) {
        snprintf(error, error_size, "cannot open %s", path);
        return NULL;
    }
    int maximum = 0;
    if (fscanf(input, "P6 %d %d %d", width, height, &maximum) != 3 ||
        *width < 1 || *height < 1 || maximum != 255) {
        snprintf(error, error_size, "%s is not an 8-bit binary PPM", path);
        fclose(input);
        return NULL;
    }
    /* Exactly one whitespace character separates the header from the data. */
    if (fgetc(input) == EOF) {
        snprintf(error, error_size, "%s ends inside its header", path);
        fclose(input);
        return NULL;
    }
    size_t count = (size_t)*width * (size_t)*height * 3;
    unsigned char *pixels = malloc(count);
    if (!pixels) {
        snprintf(error, error_size, "out of memory reading %s", path);
        fclose(input);
        return NULL;
    }
    if (fread(pixels, 1, count, input) != count) {
        snprintf(error, error_size, "%s holds fewer than %zu data bytes", path,
                 count);
        free(pixels);
        fclose(input);
        return NULL;
    }
    fclose(input);
    return pixels;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: h3_ppm_diff <a.ppm> <b.ppm>\n");
        return 2;
    }

    char error[512];
    int width_a = 0, height_a = 0, width_b = 0, height_b = 0;
    unsigned char *a = read_ppm(argv[1], &width_a, &height_a, error,
                                sizeof(error));
    if (!a) {
        fprintf(stderr, "%s\n", error);
        return 2;
    }
    unsigned char *b = read_ppm(argv[2], &width_b, &height_b, error,
                                sizeof(error));
    if (!b) {
        fprintf(stderr, "%s\n", error);
        free(a);
        return 2;
    }
    if (width_a != width_b || height_a != height_b) {
        fprintf(stderr, "geometry differs: %dx%d against %dx%d\n",
                width_a, height_a, width_b, height_b);
        free(a);
        free(b);
        return 2;
    }

    printf("SIZE width=%d height=%d\n", width_a, height_a);

    size_t total = (size_t)width_a * (size_t)height_a;
    size_t different = 0;
    int maximum = 0;
    double sum = 0.0;
    int x0 = width_a, y0 = height_a, x1 = -1, y1 = -1;
    size_t columns[MAX_BANDS] = {0};
    size_t rows[MAX_BANDS] = {0};
    int column_bands = (width_a + BAND - 1) / BAND;
    int row_bands = (height_a + BAND - 1) / BAND;

    for (int y = 0; y < height_a; y++) {
        for (int x = 0; x < width_a; x++) {
            size_t offset = ((size_t)y * (size_t)width_a + (size_t)x) * 3;
            int pixel_differs = 0;
            for (int channel = 0; channel < 3; channel++) {
                int delta = (int)a[offset + (size_t)channel] -
                            (int)b[offset + (size_t)channel];
                int magnitude = delta < 0 ? -delta : delta;
                if (magnitude) {
                    pixel_differs = 1;
                }
                if (magnitude > maximum) {
                    maximum = magnitude;
                }
                sum += (double)delta * (double)delta;
            }
            if (!pixel_differs) {
                continue;
            }
            different++;
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            if (y < y0) y0 = y;
            if (y > y1) y1 = y;
            columns[x / BAND]++;
            rows[y / BAND]++;
        }
    }

    double values = (double)total * 3.0;
    double mse = sum / values;
    double rms = sqrt(mse);
    printf("PIXELS different=%zu of %zu max=%d rms=%.6f\n", different, total,
           maximum, rms);
    if (different) {
        double psnr = mse > 0.0 ? 10.0 * log10(255.0 * 255.0 / mse) : INFINITY;
        printf("PSNR decibels=%.3f\n", psnr);
        printf("BBOX x0=%d y0=%d x1=%d y1=%d\n", x0, y0, x1, y1);
    } else {
        printf("PSNR decibels=inf\n");
    }
    for (int band = 0; band < column_bands; band++) {
        printf("BAND axis=x start=%d different=%zu\n", band * BAND,
               columns[band]);
    }
    for (int band = 0; band < row_bands; band++) {
        printf("BAND axis=y start=%d different=%zu\n", band * BAND, rows[band]);
    }

    free(a);
    free(b);
    return different ? 1 : 0;
}
