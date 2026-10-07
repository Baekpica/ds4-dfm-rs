The contract runs one engine per fresh process, sequentially. Use the final
central CUDA build and inspect the live owner/memory before running. The hash
argument reuses a verified download identity; the harness does not rehash94GB.

```
uv run --with numpy python tests/glm53_contract.py run \
  --model /path/to/GLM-5.3-Flash-Uncensored-Mixed-IQ2XXS-IQ2XS-Q2K.gguf \
  --model-sha256 7f6f96df758b5d651561c2f06ffdd0d1075a24f10654320e24e7d66c48db6017 \
  --vision /path/to/GLM-5.3-Flash-Uncensored-BF16-Vision.gguf \
  --output scratch/glm-uncensored/contract
uv run --with numpy python tests/glm53_contract.py compare \
  --output scratch/glm-uncensored/contract
```

Default arms are resident/SSD × rows1/128, ctx2048, >256 prompt tokens,
24 greedy and24 seeded sampled tokens, two simultaneous decode banks,
source-preserving fork, checkpoint-floor partial replay, disk roundtrip,
MTP abort/keep1 state and pending-operation rejection. Each arm emits full
vocabulary logits, all serialized state, token streams and partial answers.
`--arms ssd-r128` runs one arm; remaining arms may use the same output directory
only when binary/artifact/fixture identities match. `--arms expanded-r1` adds
an optional <=2K expanded-KV arithmetic reference. `--no-mtp` is partial scope.

The comparator records KDA/conv/tail/hidden/latent/pool/MTP numeric differences.
It does not approve unexplained drift or score truncated answers. Review the
numeric report and text before qualification. Configured1M is not a1M gate.
`plan-long` writes the separate pending8K selective and completed1M workload.
