"""Execute native drain-loop guards with simulated chunks and disconnects.

Only the GPU forward is replaced: compile the actual loop conditions from
ds4.c so removing a between-chunk alive check fails without loading a model.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


class PrefillCancel(unittest.TestCase):
    def test_idle_yield_obeys_boot(self):
        source = (Path(__file__).resolve().parents[1] / "ds4.c").read_text()
        start = source.index("static uint32_t bg_prefill_yield(")
        function = source[start:source.index("\n}\n", start) + 3]
        program = r"""#include <stdint.h>
#include <stdio.h>
static uint32_t boot, live;
static uint32_t bg_prefill_chunk_tokens(void) { return boot; }
static uint32_t bg_prefill_chunk_live_tokens(void) { return live; }
FUNCTION
int main(void) {
    const uint32_t cases[][6] = {
        {9000, 4096, 512, 256, 0, 512},
        {9000, 4096, 512, 256, 1, 256},
        {73, 4096, 512, 256, 0, 73},
        {9000, 256, 512, 128, 0, 256},
        {9000, 4096, 0, 512, 1, 4096},
        {9000, 4096, 512, 0, 1, 512},
        {9000, 4096, 512, 1024, 1, 512}
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const uint32_t *c = cases[i];
        boot = c[2]; live = c[3];
        uint32_t got = bg_prefill_yield(c[0], c[1], c[4]);
        if (got != c[5]) {
            fprintf(stderr, "case %u: yield=%u expected=%u\n", i, got, c[5]);
            return 1;
        }
    }
    return 0;
}
""".replace("FUNCTION", function)
        with tempfile.TemporaryDirectory(prefix="ds4-prefill-yield-") as tmp:
            path = Path(tmp) / "yield.c"
            path.write_text(program)
            binary = Path(tmp) / "yield"
            subprocess.run(
                [*shlex.split(os.environ.get("CC", "cc")), "-std=c11",
                 str(path), "-o", str(binary)], check=True
            )
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_drain_stops_at_disconnect(self):
        source = (Path(__file__).resolve().parents[1] / "ds4.c").read_text()
        for function, bank in [
            ("solar_engine_continuous_generate", "sb"),
            ("family_banked_engine_continuous_generate", "cb"),
        ]:
            with self.subTest(function=function):
                body = source[source.index("static int " + function):]
                loop = re.search(r"while\s*\((ok && .*?)\)\s*\{", body, re.S)
                self.assertIsNotNone(loop)
                condition = loop.group(1)
                program = r"""#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
typedef struct bank {
    uint32_t prefill_off, prefill_len;
    bool (*alive)(void *, void *);
    void *user;
} bank;
static bool alive(void *ud, void *user) {
    return ((bank *)user)->prefill_off < *(uint32_t *)ud;
}
int main(void) {
    for (uint32_t stop = 1; stop <= 8; stop++) {
        for (int callback = 0; callback <= 1; callback++) {
            bank state = {0, 8, callback ? alive : NULL, NULL};
            state.user = &state;
            bank *BANK = &state;
            void *ud = &stop;
            bool ok = true;
            while (CONDITION) {
                /* One completed native chunk; no decode interleaving. */
                state.prefill_off++;
            }
            uint32_t expected = callback ? stop : 8;
            if (state.prefill_off != expected) {
                fprintf(stderr, "stop=%u callback=%d: processed=%u expected=%u\n",
                        stop, callback, state.prefill_off, expected);
                return 1;
            }
        }
    }
    return 0;
}
""".replace("CONDITION", condition).replace("BANK", bank)
                with tempfile.TemporaryDirectory(prefix="ds4-prefill-cancel-") as tmp:
                    path = Path(tmp) / "cancel.c"
                    path.write_text(program)
                    binary = Path(tmp) / "cancel"
                    subprocess.run(
                        [*shlex.split(os.environ.get("CC", "cc")), "-std=c11",
                         str(path), "-o", str(binary)], check=True
                    )
                    result = subprocess.run([str(binary)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
