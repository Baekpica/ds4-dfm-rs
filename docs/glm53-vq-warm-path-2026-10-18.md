GLM-5.3 VQ warm path — why prefix reuse never fires, and what fires

Phase 7c of the GLM-5.3-VQ serving line, 2026-10-18. The 7a suite measured
cached_tokens 0 in 17 of 17 rows (docs/benchmarks/glm53-warmpath-2026-10-17/);
this pass names the refusal site for each request shape and measures the one
configuration in which a live bank does serve a growing conversation.

Identity of what was measured

- Server: fix/glm53-vq-penalties at 432fcf1 (extension 567664c), built with
  `make ds4-server` on the Spark (gx10-e1d2, 192.168.1.91:8009).
- Launcher: `install-glm53-uncensored-vq.sh start --daemon`, environment
  GLM53UNC_RUN_FLOOR_MB=7900 DS4_SERVER_GLM_TEMP_MAX=0
  DS4_SERVER_DEFAULT_DRY_MULTIPLIER=0.8, plus DS4_SERVER_WARM_TRACE=1 for the
  trace. The plan's own line: requested reuse=auto -> effective reuse=exact,
  max_seqs=2, ctx=1048576, disk=false, warning partial_unqualified.
- Client: plain curl bodies from the dev box, non-streamed (this wire reports
  the `timings` block only on the final non-streamed body).

The instrument

DS4_SERVER_WARM_TRACE=1 (432fcf1, sized in 567664c) prints one header and one
line per bank per admission, from inside the live-bank plan:

  warm trace key=<request key bytes> cache_key=<bool> reuse=<Exact|Partial> \
             full=<cached or -1> partial=<cached or -1>
  warm trace   bank=<n> text=<bytes> lcp=<common prefix with the key> \
             partial_only=<bool> cache_text=<..> exact=<..> exact_cache=<..> \
             gen=<record generation> live_gen=<bank generation> \
             committed=<bank tokens> live_tokens=<bank tokens>
  warm trace   full plan_tokens=<n> request_tokens=<n> cached=<n> \
             first_diff=<index or none>

`reuse_miss` in /v1/stats cannot answer this question on its own: with no disk
store the store-backed producer assigns NoCheckpoint unconditionally
(serve_cont.rs, the block that runs when no plan was found and the miss is still
None), so a cold request reports "no checkpoint at or below LCP" whatever the
live-bank picker thought.

The three refusals, each measured

1. A byte-identical repeat is refused by construction. `warm_match_pick`
   accepts a record as a prefix only when it is STRICTLY shorter than the
   request key (serve_cont.rs, `key.len() < request_key.len()`). Measured
   (probe 2, key 273 bytes): the repeat sees its own record at lcp=273, no
   candidate, plan -1, and /v1/stats then reports the store's NoCheckpoint.

2. A growing conversation is refused when the client drops the reasoning
   (probe 2: record 273 bytes, key 347, lcp=273). The picker finds the
   candidate, builds a plan, and the plan is dropped as "rendered prefix
   changed" because the plan's tokens are not the request's. Sized (probe 3):
   plan_tokens=574 against request_tokens=319 with cached=305 and
   first_diff=305 — the bank's frontier is a true prefix and the rebuilt suffix
   is 255 tokens too long. Cause: the bank's committed frontier is the previous
   prompt plus the generated answer (305 tokens), and the record's key text
   covers only the previous prompt (224 bytes), so `warm_admit_tokens` rebuilds
   the suffix starting at the prompt end and re-encodes what the bank already
   holds. The same shape measured again under partial reuse (probe 7):
   plan_tokens=1243, request_tokens=1236, cached=1230, first_diff=1223.

3. A growing conversation that echoes reasoning is refused at the text key
   (probe 4: record 117 bytes, key 514, lcp=108). The record's key is a
   visible-only rendering; the client's next prompt re-renders the assistant
   turn inside the think block, so the two diverge before the previous prompt
   even ends. Every record this pass observed carried exact=None: the
   reasoning-bearing exact key, the field that exists for exactly this client
   shape, is never populated on the GLM path.

What the operator's client sends

The harness transcript items are (type, content, tool_calls, model_id) — no
reasoning field (session 01a14aaa, chat_history.jsonl). The open-grok transcript
never echoes the model's reasoning, so refusal 3 is the shape of an operator
turn, and after it refusal 2 refuses the same request on token grounds. Exact
prefix reuse therefore cannot fire for an agent turn on this artifact, which is
the operator-visible cost: every turn re-prefills its whole context.

What does fire

DS4_SERVER_FORK_PARTIAL=1 makes the requested plan partial (the plan then
reports requested=partial, effective=partial; the partial_unqualified warning
stays). The partial path matches on the text LCP and cuts the reuse to the
common TOKEN prefix, so the bank's frontier may extend past what the client
sent. Measured on a 1223-token conversation (probe 7):

  case  prompt tok  cached tok  prefill tok  ttft ms   prefill tok/s  outcome
  A     1223        0           1223         1040.7    1217.1         cold
  A2    1223        0           1223         1001.5    1222.4         cold (plan found, see below)
  B     1236        1223        13           758.5     17.2           partial
  C     1249        1236        13           792.4     16.5           partial

So the reusable unit is the whole prior conversation, and each turn re-prefills
only its own 13 tokens. Two costs belong with the row: the per-request fixed
part is not small (ttft 758 ms for 13 computed tokens against 1040 ms for the
full 1223), and the partial floor is a token count, DS4_SERVER_FORK_PARTIAL_MIN
default 192 — a 61-token conversation was refused "below minimum token
threshold" in the same configuration.

Open, recorded not resolved

- The equal-length repeat (A2) finds a partial plan (cached=1222 in the trace)
  and still lands cold with no miss recorded: a plan that was found and not
  placed, which the code path calls a bank-budget refusal rather than a cache
  answer. One row, unowned.
- Partial reuse is unqualified for this artifact by its own plan
  (partial_unqualified) and needs its own correctness gate before adoption:
  under one build and one box this pass measured speed, not KV correctness.
- The operator's real case is a much longer context (the 13,335-token harness
  request; the 65,565-token 7a row costs 48 s of prefill). Partial reuse was
  measured here at 1.2k tokens and not at that size.
- Deployment left after this pass: the recorded exact configuration restored,
  trace switch off.
