/* YoungAi 846c929c src/core/core_v41_engram.c:76-93, unchanged arithmetic.
 * State/tensors are supplied directly; braces and whitespace are expanded.
 * cc -O2 gen_engram_mask_ref.c -o /tmp/engram-mask-ref
 * /tmp/engram-mask-ref > engram_mask_ref.txt
 */
#include <stdint.h>
#include <stdio.h>

static void hash(const int32_t *tmap, const int64_t *mult,
                 const int64_t *prim, const int64_t *offs, const int32_t *hist,
                 const uint8_t *himg, uint32_t G, uint32_t H, int32_t pad,
                 uint32_t n_vocab, uint32_t ei, uint32_t p, int64_t *rows) {
    int64_t prod[8];
    int blocked = 0;
    for (uint32_t k = 0; k < G; k++) {
        int64_t cid;
        if ((int64_t)p - (int64_t)k < 0 || blocked) {
            cid = (int64_t)pad;
        } else if (himg && himg[p - k]) {
            blocked = 1;
            cid = (int64_t)pad;
        } else {
            int32_t tok = hist[p - k];
            cid = (tok >= 0 && (uint32_t)tok < n_vocab) ? tmap[tok] : (int64_t)pad;
        }
        prod[k] = (int64_t)((uint64_t)cid * (uint64_t)mult[(uint64_t)ei * G + k]);
    }
    int64_t rolling = prod[0];
    for (uint32_t i = 1; i < G; i++) {
        rolling ^= prod[i];
        for (uint32_t h = 0; h < H; h++) {
            const int64_t pr = prim[((uint64_t)ei * (G - 1) + (i - 1)) * H + h];
            int64_t r = rolling % pr;
            if (r < 0) {
                r += pr;
            }
            rows[(i - 1) * H + h] = r + offs[(uint64_t)ei * (G - 1) * H + (i - 1) * H + h];
        }
    }
}

int main(void) {
    enum { N_VOCAB = 8, G = 4, H = 2, N_ENG = 2, PAD = 2, S = 8 };
    const int32_t hist[S] = {5, 6, 1, 3, 4, 0, 2, 7};
    const uint8_t himg[S] = {0, 0, 1, 1, 0, 0, 0, 0};
    int32_t tmap[N_VOCAB];
    int64_t mult[N_ENG * G], prim[N_ENG * (G - 1) * H], offs[N_ENG * (G - 1) * H];

    for (uint32_t j = 0; j < N_VOCAB; j++) {
        tmap[j] = j * 7 + 3;
    }
    for (uint32_t j = 0; j < N_ENG * G; j++) {
        mult[j] = j * 13 + 5;
    }
    for (uint32_t j = 0; j < N_ENG * (G - 1) * H; j++) {
        prim[j] = 101 + j * 17;
        offs[j] = j * 1000;
    }

    for (uint32_t ei = 0; ei < N_ENG; ei++) {
        for (uint32_t p = 0; p < S; p++) {
            int64_t rows[(G - 1) * H];
            hash(tmap, mult, prim, offs, hist, himg, G, H, PAD, N_VOCAB, ei, p, rows);
            printf("%u %u", ei, p);
            for (uint32_t c = 0; c < (G - 1) * H; c++) {
                printf(" %lld", (long long)rows[c]);
            }
            printf("\n");
        }
    }
}
