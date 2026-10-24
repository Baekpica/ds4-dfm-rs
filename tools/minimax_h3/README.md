# tools/minimax_h3 — MiniMax-H3 tooling

All h3 tooling lives under this directory; nothing h3-named sits loose in
`tools/`.

| Path | What it is |
|---|---|
| `st_probe.c` | The reference safetensors parser driven over a checkpoint directory, for the two-sided diff against `minimax_h3::safetensors` |
| `tok_probe.c` | The reference tokenizer driven over stdin, for the two-sided diff against `minimax_h3::tokenizer` |
| [vq/](vq/README.md) | The GVQ1 v1 converter (rung A): build, selftest, encode, check, pack |

All three are development oracles and converters. None is linked into the
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

The model root is `/mnt/h3-artifacts/MiniMax-H3` on the Spark. See
[docs/minimax-h3/](../../docs/minimax-h3/README.md) for the plan, the cheat
sheet and the VQ recipe.
