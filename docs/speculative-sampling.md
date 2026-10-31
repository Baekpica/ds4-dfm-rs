# Sampled speculation

Drafts (embedded MTP blocks, MTP sidecars, DFlash2 and DSpark drafters) verify
against the sampler the request decodes with. Temperature, thinking and forced
control prefixes never turn drafting off for a request: a position the plain
loop would force ends one prefix, and the next step drafts again.

## Policy

Sample-and-match. The target draws each verify row in order with the request's
sampler and random stream; a draft is kept while it equals the draw. The first
mismatch ends the prefix, and that draw is the next token. A draft `d` is kept
with probability `p(d)`, the target's probability after temperature, top-k,
top-p and min-p: the acceptance and the output distribution of vLLM's and
SGLang's target-only rejection of argmax drafts. Temperature 0 keeps argmax.

| Engine (October 2026) | Verification of argmax drafts |
|---|---|
| vLLM | rejection sampler (`standard`); argmax drafts leave the target's draw |
| SGLang | greedy only when every request is greedy, else target-only tree sampling (thresholds 1.0, exact) |
| TensorRT-LLM | MTP samples target tokens with the request's parameters and matches; relaxed acceptance is opt-in and lossy |
| llama.cpp | `--spec-draft-sampling greedy` (default): sample and accept |

The last draw is rewound: the next ordinary `sample` of the committed row
repeats it. Each emitted token takes one draw, as in plain sampling, so a fixed
seed reproduces the plain stream until the verify forward's batched arithmetic
moves a draw across a near-tie.

## Contract

```
 host (server, CLI, agent)        core: Session::eval_speculative      native trial
 -------------------------        -------------------------------      ------------
 sampler + rng          ------->  lend sample::draw as the trial's ->  forward first + drafts;
                                  row draw                             spec_sample_rows calls the
                                                                       draw per row, in order
                                  keep = matched prefix, ending   <-   (argmax at temperature 0)
                                  after EOS
 limit(prefix): stop token,  <->  keep = limit(prefix)
 forced position, closed call     commit(keep)                   ->   keep rows, roll back the rest
```

1. A family's trial leaves each verify row's logits readable (a host buffer, or
   a device tensor read with `spec_host_rows`) and fills its targets with
   `spec_sample_rows` when `spec_sampling(&s->spec)`, argmax otherwise. That
   helper hands each row to the host's draw (Rust's `sample::draw`; a bank's
   own sampler on the native bank lane) and rewinds the last draw. Rows after
   the first mismatch repeat its draw; acceptance never reads them.
2. Rust keeps the prefix whose drafts match their targets, ending after EOS,
   and cuts it with the host's `limit` before commit. Every serial family,
   Ling-3.1 included, goes through this path.
3. The host's `limit` ends the prefix after a stop token and before a position
   its own loop would decide: a forced token, structural tool syntax for a
   sampled request (its argmax position), or anything past a closed DSML call.
   The server replays the committed text on a copy of its accumulator to find
   that position. Banks apply the same rule natively (`cont_spec_forced`).

New families implement step 1 in their trial and call the shared step 2.

## Coverage

| Family | Lanes | Drafter |
|---|---|---|
| MiMo-V2.6 (MOPD, VQ) | serial | embedded MTP blocks |
| GLM-5.3, GLM-5.3 VQ | serial, banks | embedded MTP |
| Ling-3.1 VQ | serial, banks | embedded MTP |
| IQuest-Q1 | serial, banks | embedded MTP |
| dots3 | serial | embedded MTP |
| Step-3.7, Inkling | serial (Step also banks) | MTP sidecar |
| Naive N0.5 | serial, banks | DSpark sidecar |
| Beam VQ | serial | DFlash2 |
| MiniMax VQ | serial | DSpark |
| DeepSeek V4, Qwen4Exp | banks | MTP (bank lanes sampled before this change) |
| DeepSeek V4.1 | serial | in-kernel sampling of its own engine |

The DeepSeek V4 and Qwen4Exp serial verifiers (CLI `--mtp` only) compare
argmax rows; a sampling session there decodes one target token per step.

## Cost

Greedy-only drafting kept host sampling off the measured path. Sampling is
host work, so it moved to Rust (`ds4-core` `sample.rs`): plain draws and the
trials' verify-row draws go through one function, and native trials call it
back for each row. MiMo-V2.6 VQ on GB10, serial lane, two MTP drafts,
temperature 1.0, top-p 0.95:

- The C sampler sorts every logit for top-p: 13 ms per draw over 152,576
  tokens on one Grace core. A sampled verify step draws up to three rows and
  the plain loop draws once more: 127 ms per step against 79 greedy.
- The Rust sampler sorts the head first. Tokens at least 1e-3 as likely as
  the top one outrank every other token, so their order is the full order's
  prefix; a wider head is sorted only when the nucleus reaches past it. A
  sampled step took 83 ms, a plain sampled token 42 ms against 41 greedy.

| tok/s, short prompt included | Plain decode | Two drafts | Tokens per step |
|---|---:|---:|---:|
| Code, sampled | 23.7 | **32.7** | 2.70 |
| Thinking, sampled | 22.8 | **31.6** | 2.81 |
| Tool call, sampled | 20.3 | **25.7** | 2.47 |
| Story, sampled | 23.0 | 22.1 | 1.81 |
| Code, greedy | 24.6 | 35.7 | 2.82 |

Before this change the code request decoded without drafts through the C
sampler: 18.4 tok/s. Low acceptance (the story, 1.81 tokens per step) still
loses to plain decode at two drafts; matching the draft count to acceptance
is open. The banks still sample in C with the full sort (MiMo's bank lane: 55
ms per sampled token against 41 greedy); moving their draws to the Rust
sampler is the next step.

## Validation

Gates follow the change, not one number:

- Sampler: argmax, exclusion, top-k, top-p and min-p semantics; frequencies
  within 4σ of the probabilities; the head-first walk against a full sort.
- Distribution: 40 seeded requests for 30 random digits per run, drafting
  against plain decode on MiMo: chi-square 0.07, against 0.04 and 0.07
  between the halves of each run (9 degrees of freedom, p = 0.01 at 21.7).
- Acceptance, speed and output: the table above, finish reasons and tool
  calls on both runs.

Each A/B ran the same server twice, with its drafter and with `--mtp-mode
off` on the same lane. `--mtp-mode on` removes `DS4_MTP_SPEC_DISABLE` from
the environment, so an off run drops the drafting flags instead; MiMo's off
plan picks the bank lane unless `DS4_SERVER_CONTINUOUS=0` keeps it serial.
