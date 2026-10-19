# GLM-5.3 Flash VQ prefix reuse and disk KV, one DGX Spark: October 19

[GLM-5.3-Flash-VQ-GGUF](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF)
`71c73e7e` (`files.json` SHA-256 `11198c9a…`), GB10, GPU clock range
300–2200 MHz. Before: `main` `e99161ba` (`ds4-server` SHA-256 `c3002715…`).
After: this change. The gates ran `b6817288…` with explicit partial reuse;
the final build `ec2b25a5…` only adds the qualification that lets `auto`
choose it.

## What was wrong

GLM's template replays each earlier reply as `<think>`, the reasoning (only
when the client returns `reasoning_content`), `</think>` and the stripped
content. Three host defects kept a retired bank's text key off that history,
so no new user turn found its bank:

| Defect | Key | Next render |
|---|---|---|
| `<|user|>`/`<|observation|>` end a reply as host stop tokens, not the engine EOS, so retirement saw an unfinished turn | no key with the reasoning | `<think>R</think>A` |
| A thinking-off key stripped the prompt's trailing pair | `<|assistant|>A` | `<|assistant|><think></think>A` |
| A visible key used DeepSeek's lone close | `<|assistant|></think>A` | `<|assistant|><think></think>A` |

Two more kept the remaining paths off. `auto` resolved to exact reuse
because VQ partial reuse was unqualified, and that cannot serve a client
that drops the reasoning the bank holds. `routed_quant_bits` returned 0 for
the HF expert names, so every disk save and restore was skipped without a
message, as Ling-3.1's were before `c43f4152`.

## Gates

`tests/serving_reuse_live.py`: context 2048, two banks, chunks 64, partial
reuse, 2 GiB disk with one-token thresholds. Seed and warm share a process,
`restored` starts a second on the same disk and `cold` a third with reuse
and disk off. Receipts: [gate evidence](gate-evidence.json).

| Request | Thinking off, MTP off | Thinking off, MTP 2 | High, MTP 2 |
|---|---|---|---|
| Seed | 539 / 0 cold | 539 / 0 cold | 538 / 0 cold |
| Append | 558 / 540 fork | 558 / 540 fork | 573 / 556 fork |
| Edit | 558 / 539 partial | 558 / 539 partial | 573 / 538 partial |
| Branch | 577 / 559 exact | 577 / 559 exact | 593 / 538 partial |
| Restart (disk) | 596 / 578 fork | 596 / 578 fork | 613 / 596 partial |

Cells are input / cached tokens and the reported reuse. Every phase passes:
the five cold replays match their messages, stop reasons and completion
counts byte for byte, and both MTP campaigns pass the digit probe. Append
keeps the whole previous turn, including its reasoning in high mode. There
the edit rewinds the branch's bank in place (the seed bank's primary key is
the visible text, so neither bank counts as superseded) and the branch
restores the seed checkpoint; MTP committed the restart's source stop row,
so token-LCP reuse reports partial for the complete frontier.

## Production shape

1,048,576 context, one bank, two MTP drafts, 2 GiB floor and guard trip,
32 GiB disk, temperature 0, no `--prefix-reuse` flag. A 1,374-token system
prompt opens the short conversations; the long one is the 30K story fixture.
Rows: [production evidence](production.json).

| Next turn | Before: input / cached / TTFT | After: input / cached / TTFT |
|---|---|---|
| Thinking, reasoning dropped | 1,434 / 0 / 1.32 s | 1,434 / 1,374 partial / 0.82 s |
| Thinking, reasoning returned | 1,607 / 0 / 1.46 s | 1,652 / 1,639 exact / 1.40 s |
| Thinking off | 1,399 / 0 / 1.26 s | 1,399 / 1,389 exact / 1.03 s |
| Tool result, reasoning returned | 1,573 / 1,554 exact / 1.72 s | 1,573 / 1,554 exact / 1.60 s |
| Tool result, reasoning dropped | 1,552 / 0 / 1.42 s | 1,552 / 1,522 partial / 2.05 s |
| 30K conversation | 30,628 / 0 / 25.0 s | 30,628 / 30,613 exact / 1.29 s |
| 30K after a restart | 30,687 / 0 / 25.5 s | 30,687 / 30,676 exact / 1.92 s |
| 30K after another conversation | 30,651 / 0 / 25.0 s | 30,651 / 0 / 25.2 s |
| Same, `DS4_SERVER_PIN_MIN_TOKENS=8192` | — | 30,651 / 30,637 exact / 1.39 s |

Inputs differ where the before run generated a different reply: a cold
4,096-row prefill and a restored prefix take different reduction orders.

- **Short suffixes stay slow.** A 10–30-token suffix takes 1.0–2.0 s to the
  first token (one token: 47 ms; 60 tokens: 0.82 s), so a reused 1.5K prompt
  with a 30-token suffix is slower than its cold prefill. This is a
  prefill-width cost below 32 rows, not a cache miss.
- **Pin depth.** A disk record shallower than `DS4_SERVER_PIN_MIN_TOKENS`
  (default 65,536) never evicts a live bank. With one bank, set it to the
  8,192-token persistence threshold so a stored conversation can displace a
  short live one.
- **Memory.** Minimum host availability was 3.80 GiB (before 4.02 GiB); the
  quote prices the eight lazy checkpoint slots at 1.25 GB.
- **Disk.** A record holds the 149 MiB recurrent state plus about 13 KiB per
  token: 555 MB for 30K tokens.
