# GLM-5.3 values, live: the artifact's sampler on the Spark

Run of 2026-10-17 23:26-23:29 (+02:00), the first after the values unit.
Harness `tests/glm53_http_suite.py` (all five cases, `temperature: 0` as it
always sends), against the deployed server.

| | |
| --- | --- |
| Build | fix/glm53-agent-args `c9d325b`, `ds4-server` 70,708,352 bytes, sha256 `468560528448d17c...`, built 23:17:56 |
| Start | `GLM53UNC_RUN_FLOOR_MB=8000 ~/install-glm53-uncensored-vq.sh start --daemon`; pid 403583, watchdog 403584 |
| Startup line | `ds4-server-rs: GLM values: temperature=1 top_p=0.95 min_p=0 effort=Max` (log line 801 of ~/glm53unc.log) |
| Platform | GB10 sm_121, 0.0.0.0:8009, ctx 1048576, banks 2, MTP on with two drafts, continuous lane |

## The five cases against the 7a medians (f738059, also temperature 0)

| case | prompt tok | ttft now | ttft 7a | prefill tok/s now | now 7a | decode tok/s now | 7a | tok/step now | 7a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prose_short | 58 | 271.5 ms | 271.4 | 209.6 | 213.2 | 37.4 | 24.4 | 2.46 | 1.01 |
| tool_short | 230 | 433.3 ms | 434.3 | 527.9 | 529.7 | 40.6 | 24.5 | 2.68 | 1.02 |
| prose_64k | 65,565 | 48,132.6 ms | 47,955.6 | 1,367.4 | 1,371.3 | 24.9 | 22.5 | 2.00 | 1.03 |
| interleave_a | 65 | 287.4 ms | 265.1 | 245.6 | 250.3 | 37.3 | 24.1 | 2.42 | 1.04 |
| interleave_b | 81 | 286.0 ms | 285.7 | 285.4 | 286.5 | 37.8 | 24.2 | 2.61 | 1.01 |
| prose_short_repeat | 58 | 273.4 ms | 272.7 | 213.1 | 213.1 | 37.2 | 24.3 | 2.46 | 1.01 |

Prefill and time-to-first-token are unchanged inside noise (the largest gap,
interleave_a's 265 -> 287 ms, is the harness's own a/b interleaving, and the
64k case repeats 7a's 48 s at 1,367 tok/s against 1,371). Cached tokens are 0
in all 17 rows, as in 7a: the reuse item (7c) is untouched by this unit.

`tool_short` still returns the parseable call with numeric arguments --
`{"code":"QZ5813","limit":3,"max_results":15}` -- under the artifact's top_p
0.95 / min_p 0, so the values did not disturb the tool path.

## The decode column moved, and the sampler is why

Decode rose from 22.5-24.5 to 24.9-40.6 tok/s and `tok_per_step` from 1.01-1.04
to 2.00-2.68, because the harness asks for `temperature: 0` and the server now
honours it: before this unit the thinking regime overwrote that 0 with 1.0.
One A/B on the same server, same prompt and cap, only the sampler differing:

| request | tok_per_step | decode tok/s |
| --- | --- | --- |
| `temperature: 0` | 2.13 | 17.7 |
| `temperature: 1.0` | 1.01 | 9.3 |

Two more requests with `temperature: 0`, byte-compared:

| | reasoning sha256 | content sha256 | tokens | finish | answer |
| --- | --- | --- | --- | --- | --- |
| run 1 | `8c66f9c6c1f15924` | `34ff51c7b6320bbd` | 261 | stop | `2, 3, 5` |
| run 2 | `8c66f9c6c1f15924` | `34ff51c7b6320bbd` | 261 | stop | `2, 3, 5` |

## Findings

1. Reading 7a's `tok_per_step 1.01` as "MTP accepts almost nothing" was wrong.
   Acceptance follows the sampler: at 1.0 it is ~0 (1.01), at 0 it is ~1.1-1.7
   accepted drafts per step (2.13-2.68). MTP itself is working; what 7a measured
   was a greedy-capable request being served sampled. 7d remains open for the
   sampled regime, where acceptance near zero is still unexplained.
2. Honouring the request is worth 1.9-2.5x decode throughput for greedy
   requests, at unchanged prefill. Nothing about the sampler changed the warm
   path's fixed cost.
3. The tool path is unaffected: the same call parses with numeric arguments.
4. Loops are NOT measured here. This run is latency and path correctness; the
   loop probe (the 2026-10-15 byte-identical repetition) is the next unit and
   needs its own counter, not this suite.

## Limits

- One run, two reps per long case; a single session on a shared box.
- Greedy rows cannot be compared with sampled ones as if the harness were the
  same workload; the 7a comparison holds only because both ran temperature 0.
- `prose_64k`'s 48 s is dominated by prefill and was measured twice.
