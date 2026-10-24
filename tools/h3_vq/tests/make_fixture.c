/* make_fixture.c — write a miniature safetensors checkpoint with H3's names.
 *
 * Purpose: exercise the packer's classification, plan, encode, check and merge
 * paths without a 144 GB download. The tensor *names* and dtypes are H3's own,
 * read from FL2VA/transformer/model.safetensors.index.json and the shard
 * headers; the shapes are shrunk (columns stay multiples of 64 so every VQ
 * path is reachable, rows are 8 or fewer so the encode finishes instantly).
 *
 * It covers each branch of classify(): the five block classes, the head class,
 * rank-1 tensors, and all six F32-by-design matrices that stay native.
 *
 * Usage: make_fixture <out-dir>      writes <out-dir>/fixture.safetensors
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct { const char *name; const char *dtype; int nd; uint64_t d0, d1; } ent;

static const ent TENSORS[] = {
    /* the five VQ classes, one matrix each */
    {"blocks.0.attn.qkv_proj.weight",          "BF16", 2, 8, 64},
    {"blocks.0.attn.out_proj.weight",          "BF16", 2, 8, 64},
    {"blocks.0.mlp.fc1.weight",                "BF16", 2, 8, 64},
    {"blocks.0.mlp.fc2.weight",                "BF16", 2, 8, 64},
    {"blocks.0.adaln_proj.linear.weight",      "BF16", 2, 8, 64},
    /* kept: rank 1 */
    {"blocks.0.adaln_proj.linear.bias",        "BF16", 1, 8, 0},
    {"blocks.0.norm1.weight",                  "BF16", 1, 64, 0},
    {"blocks.0.attn.q_norm.weight",            "BF16", 1, 16, 0},
    {"token_refiner.final_norm.weight",        "BF16", 1, 64, 0},
    {"rope.inv_freq",                          "F32",  1, 16, 0},
    /* the token refiner, classified by matrix kind */
    {"token_refiner.blocks.0.attn.qkv_proj.weight", "BF16", 2, 8, 64},
    {"token_refiner.blocks.0.mlp.fc2.weight",       "BF16", 2, 8, 64},
    /* the head class */
    {"condition_proj.weight",                       "BF16", 2, 8, 64},
    {"final_layer.adaln_proj.linear.weight",        "BF16", 2, 8, 64},
    /* kept: F32 by design, and one with columns that divide the vector width
     * (audio_patch_proj, 32) so the name list is what keeps it, not the shape */
    {"video_patch_proj.weight",                "F32",  2, 8, 96},
    {"audio_patch_proj.weight",                "F32",  2, 8, 32},
    {"final_layer.video_out.weight",           "F32",  2, 4, 64},
    {"final_layer.audio_out.weight",           "F32",  2, 4, 64},
    {"time_embedder.proj_in.weight",           "F32",  2, 8, 64},
    {"time_embedder.proj_out.weight",          "F32",  2, 8, 64},
};
#define NT (int)(sizeof TENSORS / sizeof TENSORS[0])

/* The table above is the read-only default; main() copies it here so the
 * optional rows/cols override can be applied without writing to .rodata. */
static ent ENT[NT];

static uint64_t nbytes(const ent *e) {
    uint64_t n = e->nd == 2 ? e->d0 * e->d1 : e->d0;
    return n * (strcmp(e->dtype, "F32") == 0 ? 4 : 2);
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 4) {
        fprintf(stderr, "usage: make_fixture <out-dir> [rows cols]\n"
                        "  rows/cols override the VQ matrices' shape; the default (8x64)\n"
                        "  is degenerate on purpose (128 training vectors against a 4096-entry\n"
                        "  codebook), so pass e.g. '512 256' when testing determinism.\n");
        return 2;
    }
    memcpy(ENT, TENSORS, sizeof ENT);
    if (argc == 4) {
        /* rank-2 VQ entries only: the kept matrices keep their own shapes */
        for (int i = 0; i < NT; i++) {
            ent *e = &ENT[i];
            if (e->nd == 2 && (strstr(e->name, ".attn.") || strstr(e->name, ".mlp.") ||
                               strstr(e->name, "adaln_proj") || strstr(e->name, "condition_proj")))
                { e->d0 = (uint64_t)atoll(argv[2]); e->d1 = (uint64_t)atoll(argv[3]); }
        }
    }
    char path[1024];
    snprintf(path, sizeof path, "%s/fixture.safetensors", argv[1]);

    /* header */
    char *h = malloc(1 << 20);
    size_t o = 0;
    o += (size_t)sprintf(h + o, "{");
    uint64_t off = 0;
    for (int i = 0; i < NT; i++) {
        const ent *e = &ENT[i];
        uint64_t b = nbytes(e);
        char shp[64];
        if (e->nd == 2) snprintf(shp, sizeof shp, "[%llu,%llu]",
                                 (unsigned long long)e->d0, (unsigned long long)e->d1);
        else            snprintf(shp, sizeof shp, "[%llu]", (unsigned long long)e->d0);
        o += (size_t)sprintf(h + o,
            "\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%llu,%llu]}%s",
            e->name, e->dtype, shp,
            (unsigned long long)off, (unsigned long long)(off + b),
            i + 1 < NT ? "," : "");
        off += b;
    }
    o += (size_t)sprintf(h + o, "}");
    while (o % 8) h[o++] = ' ';                       /* the format pads the header */

    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    uint64_t hl = o;
    fwrite(&hl, 8, 1, f);
    fwrite(h, 1, o, f);
    /* data: a deterministic ramp, so two runs give the same bytes */
    for (int i = 0; i < NT; i++) {
        const ent *e = &ENT[i];
        uint64_t b = nbytes(e);
        if (strcmp(e->dtype, "F32") == 0) {
            for (uint64_t k = 0; k < b / 4; k++) { float v = (float)((int)(k % 37) - 18) / 32.0f; fwrite(&v, 4, 1, f); }
        } else {
            for (uint64_t k = 0; k < b / 2; k++) { uint16_t v = (uint16_t)(0x3c00 + (k % 29)); fwrite(&v, 2, 1, f); }
        }
    }
    fclose(f);
    printf("wrote %s: %d tensors, %llu bytes of data, header %llu bytes\n",
           path, NT, (unsigned long long)off, (unsigned long long)hl);
    free(h);
    return 0;
}
