/* gen_engram_ref.c — P2 parity harness for the V4.1 engram hash.
 *
 * The hash below is `v41_engram_hash` from the engine
 * (/data/YoungAi src/core/core_v41_engram.c:94-116) with one change: the four
 * constant tensors and the scalar parameters arrive as arguments instead of
 * being read out of `ds4_engine`/`g_ds4_v41`. Everything inside the function -
 * the layer-major flat indexing `mult[ei*G + k]`,
 * `prim[(ei*(G-1) + i-1)*H + h]`, `offs[...]`, the unsigned wrap-around
 * multiply, the modulo correction - is verbatim, because that indexing is
 * what a dims-following reader gets wrong silently.
 *
 *   cc -O2 gen_engram_ref.c -o gen_engram_ref && ./gen_engram_ref
 *
 * Writes engram_ref.txt: the constants, the history, and the rows for both
 * engram indices over every position. The Rust `EngramHash::rows` must
 * reproduce every row.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Engine: core_v41_engram.c:94-116, parameter list only. */
static void v41_engram_hash(const int32_t *tmap, const int64_t *mult,
                            const int64_t *prim, const int64_t *offs,
                            const int32_t *hist, uint32_t G, uint32_t H,
                            int32_t pad, int32_t n_vocab, uint32_t ei,
                            uint32_t p, int64_t *rows /*[cols]*/) {
    int64_t prod[8];
    for (uint32_t k = 0; k < G; k++) {
        int64_t cid;
        if ((int64_t)p - (int64_t)k < 0) cid = (int64_t)pad;
        else {
            int32_t tok = hist[p - k];
            cid = (tok >= 0 && (uint32_t)tok < (uint32_t)n_vocab) ? tmap[tok] : (int64_t)pad;
        }
        prod[k] = (int64_t)((uint64_t)cid * (uint64_t)mult[(uint64_t)ei * G + k]);
    }
    int64_t rolling = prod[0];
    for (uint32_t i = 1; i < G; i++) {
        rolling ^= prod[i];
        for (uint32_t h = 0; h < H; h++) {
            const int64_t pr = prim[((uint64_t)ei * (G - 1) + (i - 1)) * H + h];
            int64_t r = rolling % pr;
            if (r < 0) r += pr;
            rows[(i - 1) * H + h] = r + offs[(uint64_t)ei * (G - 1) * H + (i - 1) * H + h];
        }
    }
}

#define NVOCAB 64u
#define G 4u
#define H 8u
#define N_ENG 2u
#define PAD 2
#define S 12u

static void print_i64(FILE *f, const char *tag, const int64_t *v, size_t n) {
    fprintf(f, "%s %zu", tag, n);
    for (size_t i = 0; i < n; i++) fprintf(f, " %lld", (long long)v[i]);
    fprintf(f, "\n");
}

int main(void) {
    int32_t tmap[NVOCAB];
    int64_t mult[N_ENG * G], prim[N_ENG * (G - 1) * H], offs[N_ENG * (G - 1) * H];
    int32_t hist[S] = {5, 9, 33, 1, 60, 0, 12, 7, 63, 2, -1, 40};

    for (uint32_t i = 0; i < NVOCAB; i++) tmap[i] = (int32_t)(i * 7u + 3u);
    for (size_t j = 0; j < N_ENG * G; j++) mult[j] = (int64_t)(j * 13u + 5u);
    for (size_t j = 0; j < N_ENG * (G - 1) * H; j++) prim[j] = (int64_t)(101u + j * 17u);
    for (size_t j = 0; j < N_ENG * (G - 1) * H; j++) offs[j] = (int64_t)(j * 1000u);

    FILE *f = fopen("engram_ref.txt", "w");
    if (!f) { perror("engram_ref.txt"); return 1; }
    fprintf(f, "NGRAM %u\nHEADS %u\nPAD %d\nNVOCAB %u\n", G, H, PAD, NVOCAB);
    /* tmap is i32; print it explicitly so the fixture keeps the C types. */
    fprintf(f, "TMAPI32 %u", NVOCAB);
    for (uint32_t i = 0; i < NVOCAB; i++) fprintf(f, " %d", tmap[i]);
    fprintf(f, "\n");
    print_i64(f, "MULT", mult, N_ENG * G);
    print_i64(f, "PRIM", prim, N_ENG * (G - 1) * H);
    print_i64(f, "OFFS", offs, N_ENG * (G - 1) * H);
    fprintf(f, "HIST %u", S);
    for (uint32_t i = 0; i < S; i++) fprintf(f, " %d", hist[i]);
    fprintf(f, "\n");

    int64_t rows[(G - 1) * H];
    for (uint32_t ei = 0; ei < N_ENG; ei++) {
        for (uint32_t p = 0; p < S; p++) {
            v41_engram_hash(tmap, mult, prim, offs, hist, G, H, PAD, (int32_t)NVOCAB, ei, p, rows);
            fprintf(f, "ROW %u %u", ei, p);
            for (uint32_t c = 0; c < (G - 1) * H; c++) fprintf(f, " %lld", (long long)rows[c]);
            fprintf(f, "\n");
        }
    }
    fclose(f);
    return 0;
}
