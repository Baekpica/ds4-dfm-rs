# DeepSeek V4.1 warm-path campaign: the asset-open cache

One DGX Spark / GB10, the author's five-case HTTP suite, three successive
passes against one server process (this campaign) against the archived
pre-change pass. Every sample is retained; medians carry the min-max band.

[Conditions and evidence](README.md).

## Warm-prefix TTFT

Later requests of a session that reuse the prompt's KV checkpoint.

| Case | Prompt tokens | Before, ms | After, ms | Before/after |
| --- | ---: | ---: | ---: | ---: |
| prose_short | 54 | 190.9 (181.8-200.0) | 15.1 (11.1-20.8) | 12.6x |
| code_short | 54 | 196.7 (196.2-197.1) | 17.3 (11.4-18.9) | 11.4x |
| table_short | 43 | 194.1 (191.2-197.0) | 14.1 (12.6-19.1) | 13.8x |
| prose_long | 10,356 | 265.7 (259.8-271.6) | 99.1 (85.6-103.3) | 2.7x |
| table_long | 10,377 | 273.4 (268.3-278.5) | 99.5 (89.3-108.4) | 2.7x |

## First request of a case (first-touch TTFT)

The first request that sends those bytes carries the prefill, so this is the
cold cell, one sample per pass.

| Case | Prompt tokens | Before, s | After, s | After cold prefill, tok/s |
| --- | ---: | ---: | ---: | ---: |
| prose_short | 54 | 0.657 (0.657-0.657) | 0.427 (0.427-0.451) | 126.4 |
| code_short | 54 | 0.591 (0.591-0.591) | 0.409 (0.407-0.409) | 131.9 |
| table_short | 43 | 0.544 (0.544-0.544) | 0.372 (0.359-0.373) | 115.6 |
| prose_long | 10,356 | 8.053 (8.053-8.053) | 7.957 (7.949-7.988) | 1,301.5 |
| table_long | 10,377 | 8.086 (8.086-8.086) | 8.001 (8.001-8.010) | 1,297.0 |

## Decode throughput (automatic embedded MTP)

| Case | Before, tok/s | After, tok/s |
| --- | ---: | ---: |
| prose_short | 36.26 (36.19-36.27) | 35.87 (35.50-36.12) |
| code_short | 54.33 (54.02-54.45) | 53.42 (53.31-54.36) |
| table_short | 65.41 (65.36-65.57) | 65.28 (64.92-65.41) |
| prose_long | 31.08 (30.97-31.15) | 32.50 (32.34-32.83) |
| table_long | 59.14 (59.01-59.42) | 58.61 (58.45-58.78) |
