# tools/minimax_h3 — MiniMax-H3 tooling

All h3 tooling lives under this directory; nothing h3-named sits loose in
`tools/`.

| Path | What it is |
|---|---|
| `st_probe.c` | The reference safetensors parser driven over a checkpoint directory, for the two-sided diff against `minimax_h3::safetensors` |
| `tok_probe.c` | The reference tokenizer driven over stdin, for the two-sided diff against `minimax_h3::tokenizer` |
| `text_probe.c` | The reference prompt encoder run on the Spark, printing the BF16 output hash the port's forward must reproduce |
| `dit_probe.c` | The reference DiT run on the Spark: one forward, plus `--adaln-only` and `--adaln-stages` for localization and `--features` for the timestep table |
| [vq/](vq/README.md) | The GVQ1 v1 converter (rung A): build, selftest, encode, check, pack |

All four are development oracles and converters. None is linked into the
production hosts.

```sh
cc -O2 -o bin/st_probe tools/minimax_h3/st_probe.c   # from a checkout
tools/minimax_h3/vq/build.sh                        # the VQ converter
```

`tok_probe.c` and `st_probe.c` include the reference headers, so they build
against an unpacked `/data/h3.c` (or `~/h3-oracle` on the Spark):

```sh
cc -O2 -I ~/h3-oracle tools/minimax_h3/tok_probe.c ~/h3-oracle/libh3.a \
   -licui18n -licuuc -lm -o ~/tok_probe
```

`text_probe.c` and `dit_probe.c` reach the CUDA backend, so they also link the
CUDA runtime and C++ runtime the reference uses, and keep the reference's
`_POSIX_C_SOURCE` define for `clock_gettime`:

```sh
cc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I ~/h3-oracle \
   tools/minimax_h3/dit_probe.c ~/h3-oracle/libh3.a \
   -L/usr/local/cuda/lib64 -lcudart -lcublasLt -lstdc++ -licui18n -licuuc -lm \
   -o ~/dit_probe
```

The model root is `/mnt/h3-artifacts/MiniMax-H3` on the Spark. See
[docs/minimax-h3/](../../docs/minimax-h3/README.md) for the plan, the cheat
sheet and the VQ recipe.
