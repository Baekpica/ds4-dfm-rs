# Step 3.7 MQ83 contract fixtures

`metadata.json` records official GGUF metadata at
`stepfun-ai/Step-3.7-Flash-GGUF@0b69336d2fd2adfdef9c66e425f7778196c31482`,
plus the mixed artifact's source revision marker. The test constructs a
metadata-only file with placeholder vocabulary strings; it is not a tokenizer
or inference fixture.

`mq83.tsv` records the independently inventoried 754 tensor names and GGUF
dimensions, with the owner's locked mixed recipe: Q8 critical/shared matrices,
Q4_K routed edges/down, IQ2_XXS interior gate/up, F32 controls. Payload size is
83,001,512,448 bytes. No weight data is included.

Original model revision: `5f6244077ac62e04eec3f320501ff8c2b293373a`.

`mtp.tsv`, `vision.tsv` and their metadata JSON files come from the handoff's
`sidecar-inventory.json`. They preserve the official Q8 MTP and F16 projector
layouts without payload data. Tests use them independently of runtime specs.
