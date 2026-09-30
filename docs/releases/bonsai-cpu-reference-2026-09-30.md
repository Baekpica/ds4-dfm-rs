# Bonsai (qwen35) on the CPU reference path — 2026-09-30

Records the first unit in which this tree runs the local Prism Bonsai
artifact.  Artifacts, commands and outputs are as observed on this host
(RTX 4070 SUPER, x86_64, 28 cores, 31 GiB RAM) at commits
`789eada`, `1f5ff5c` and `97c64a3` on `feature/qwen35-port`.

Model: `/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`, 7.2 GB, 851 tensors
(402 pq2_0, 353 f32, 96 bf16), `general.architecture = qwen35`.

Build the CPU host binary:

    make cpu -j4

## 1. The artifact loads and its fold metadata validates

    ./ds4-c -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --inspect

    ds4: prism.hadamard folding: block 1024, 3 sign vector(s), gdn_v_grouped 1

## 2. The folded-activation transform matches the explicit matrix

    DS4_QWEN35_FOLD_SELFTEST=1 ./ds4-c -m .../Ternary-Bonsai-2-27B-PQ2_0.gguf \
        --cpu --first-token-test -p "x"

    fold selftest: blocks 2, 4 and 1024 match the explicit Hadamard matrix,
    blocks stay independent, forward/inverse round-trips, and the gdn
    permutation follows the tiled-to-grouped index map

## 3. Every weight row reads exactly like the exporter's dequantizer

`tests/pq2_0/reference_checksums.txt` carries, per tensor, the FNV-1a
checksum of the first 64 dequantized rows as the PrismML fork's own ggml
dequantizer produces them.

    DS4_BONSAI_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
        make test-qwen35-rows

    qwen35 rows: 851 tensors matched the reference dequantizer (first 64 rows each)

## 4. Reference forward parity with the sibling tree

The oracle is the C implementation in `/data/ds4` (branch `bonsai`, tip
`bbaf298`) that already runs this family.  Both trees were given the same
five raw prompt tokens (`760 6511 314 9338 369`, "The capital of France is")
through `DS4_QWEN35_TOKENS`, and both dumped the full `[5][248320]` f32 logits.

    cd /data/ds4
    DS4_QWEN35_TOKENS="760,6511,314,9338,369" DS4_QWEN35_STEPS=0 \
        DS4_QWEN35_LOGITS=ref_logits.bin \
        ./ds4 -m .../Ternary-Bonsai-2-27B-PQ2_0.gguf --cpu --first-token-test \
              --raw -p "x"

    cd /data/ds4-dfm-rs
    DS4_QWEN35_TOKENS="760,6511,314,9338,369" DS4_QWEN35_STEPS=0 \
        DS4_QWEN35_LOGITS=our_logits.bin \
        ./ds4-c -m .../Ternary-Bonsai-2-27B-PQ2_0.gguf --cpu --first-token-test \
               -p "x"

Comparison of the two dumps (1 241 600 values each):

    max abs difference 0
    elements over 1e-3: 0
    pos 0..4 argmax: 220, 314, 279, 369, 11751 in both

The two streams are bit-identical.  The observed next-token distribution is
also the coherent one: `11751 " Paris"` at 14.2724 against 10.8312 for the
runner-up.

## 5. Greedy decode parity

    DS4_QWEN35_TOKENS="760,6511,314,9338,369" DS4_QWEN35_STEPS=8 ... 
    prompt: 760 6511 314 9338 369
    token 5: 11751  Paris
    token 6: 13 .
    token 7: 198
    token 8: 760 The
    token 9: 6511  capital
    token 10: 314  of
    token 11: 9564  Germany
    token 12: 369  is

Both trees print this identical stream, so the recurrent gated delta-net
state and the attention cache follow the reference over eight decode steps.

## Cost

About 6 s per token on 28 cores (34 s for a five-token prompt pass), single
token attention, no batching.  This path is the correctness oracle, not a
serving path.

## Not covered by this unit

The CUDA graph, sessions and serving.  The engine refuses everything but
`--cpu --first-token-test` for this family by name.
