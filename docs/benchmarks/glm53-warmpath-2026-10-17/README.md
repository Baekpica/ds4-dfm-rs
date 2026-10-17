GLM-5.3 VQ warm-path suite, 2026-10-17 (phase 7a)

What this is

The first measurement pass of the warm-path unit: five cases over the live
GLM-5.3-Flash-Uncensored-VQ serve on the Spark (gx10-e1d2, 192.168.1.91:8009),
one row per sample in raw.csv. The harness is tests/glm53_http_suite.py; its
header carries the protocol and the one deviation from the ds41 shape (this wire
is non-streamed, because the Rust server emits its `timings` block -- including
tok_per_step, the MTP column -- only on the non-streamed final body).

Identity of what was measured

- Server: fix/glm53-agent-args = f738059, built 22:02 by `make ds4-server`
  (the binary the launcher runs), launched with GLM53UNC_RUN_FLOOR_MB=8000.
- Effective serving plan, from the server's own log: reuse=exact, max_seqs=2,
  mtp=on, ctx=1048576, floor=4G, disk=false, and the plan's own warning
  "GLM-5.3-Flash-VQ partial reuse is implemented but not qualified
  (partial_unqualified)".
- The continuous lane is live: "ds4-server-rs: continuous lane ready (width=2
  seq_cap=1048576)".
- Client: the dev box, over the LAN. One run, no interleaved A/B.

The rows (medians of the samples in raw.csv)

| case               | prompt tok | server ttft | prefill tok/s | decode tok/s | cached tok |
| prose_short        | 58         | 271.4 ms    | 213.2         | 24.4         | 0          |
| tool_short         | 230        | 434.3 ms    | 529.7         | 24.5         | 0          |
| prose_64k          | 65565      | 47955.6 ms  | 1371.3        | 22.5         | 0          |
| interleave_a       | 65         | 265.1 ms    | 250.3         | 24.1         | 0          |
| interleave_b       | 81         | 285.7 ms    | 286.5         | 24.2         | 0          |
| prose_short_repeat | 58         | 272.7 ms    | 213.1         | 24.3         | 0          |

What the rows say

1. Reuse never happens, in any shape measured. Every sample reports
   cached_tokens 0 with prefill_tokens equal to the whole prompt and
   cache_write_tokens equal to the whole prompt -- 17 of 17 rows, including
   byte-identical repeats (prose_short reps 1-2, prose_short_repeat rep 0 whose
   bytes are prose_short's) and the interleaved conversations returning to a
   bank they had just used.

2. Three further shapes probed outside the harness, same result. A 2219-token
   prompt repeated byte-identically, the same prompt with one user turn appended,
   and the same prompt with an assistant turn carrying the server's own reply
   (content only, and again with its reasoning_content) all reported cached 0 and
   prefill_tokens = the full prompt. So the miss is not a property of the append
   form or of the reasoning field.

3. One structural reason is already visible in the source and is certain: the
   live-bank candidate picker accepts a stored record as a prefix only when the
   record is STRICTLY SHORTER than the new request's rendered prompt
   (serve_cont.rs:881-884, `key.len() < request_key.len()`), so a repeat of the
   same bytes can never be a candidate by construction. Whether a well-formed
   growing conversation is refused for a second reason is NOT yet established --
   that is the open question 7c's instrumentation has to answer.

4. The per-request cost decomposes as a small fixed part plus a marginal prefill
   rate, both read off the rows arithmetically:
   - 58 -> 230 prompt tokens: (434.3-271.4)/(230-58) = 0.95 ms/token, i.e.
     ~1056 tok/s marginal, with a ~217 ms intercept.
   - 230 -> 65565 tokens: (47955.6-434.3)/(65565-230) = 0.73 ms/token, i.e.
     ~1375 tok/s marginal, with a ~267 ms intercept.
   So the 64k turn's 48 s is prefill work, not a fixed per-request stall; the
   agent-turn complaint is the absence of reuse (item 1), not a hidden constant.

5. MTP accepts almost nothing at two drafts: tok_per_step 1.01-1.04 across every
   row, decode 22.5-24.5 tok/s. That is 7d's target and the handoff's open item 4,
   now with rows behind it.

6. One outlier, single observation, recorded for honesty: in an earlier manual
   probe a 25-token prompt sent right after a 2219-token one cost ttft 1269 ms
   with tok_per_step 1.33, while the suite's short prompts cost 265-435 ms. It may
   be bank placement after a large prompt, or graph recapture; one sample is not
   evidence either way.

Limits

- One Spark, one build, one time window; no interleaved A/B pairs, so these are
  not adoption-grade numbers (the AGENTS.md bar) -- they are the reconnaissance
  the bar's first step requires.
- prefill_tok_s is the server's own account (prefill_tokens / prefill_ms); the
  client wall clock in total_s agrees with it to within ~1%, so the two are not
  independent.
- The tool case's schema is deliberately the unit-1 shape; its rows also prove
  the argument typing on live traffic: every tool row returned
  {"code":"QZ5813","limit":3,"max_results":15}, numbers for the u8-spelled and
  the type-list property.

Next step this record sets up (7c)

Expose the warm plan's own miss reason. ReuseMiss already carries
NoCheckpoint / RenderedPrefix / BelowThreshold / PayloadMismatch / StateReplay
with as_str() (ds4-core/src/serving.rs:405-425) and serve_cont already computes
it per request (serve_cont.rs:3184, and the only-consumer check at :3401); one
diagnostic print behind a switch turns "no hit" into a named refusal and decides whether 7c is a one-line
candidate-picker question or a bank-record question.
