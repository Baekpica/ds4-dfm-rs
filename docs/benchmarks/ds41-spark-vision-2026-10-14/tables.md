# DeepSeek V4.1 Flash vision on the feat/optimize head

One DGX Spark / GB10, one vision server process, three attach passes.
Image rows send the repo fixtures with one fixed prompt; text rows
bracket the image spans. Tower seconds and the per-image plan come
from the server's own log lines (tower-lines.txt). Every sample is
retained in the run CSVs; medians carry the min-max band.

[Conditions and evidence](README.md).

## Rows (median of three passes)

| Row | Kind | Prompt tokens | Span | Patch -> token | Tower, s | TTFT, ms | Prefill, tok/s | Decode, tok/s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| review-market-panel.png | image | 502 | 482 | 48x86 -> 16x29 | 4.00 (3.90-4.08) | 5,016 (4,890-5,088) | 100.3 | 43.10 |
| bug.jpeg | image | 967 | 947 | 134x60 -> 45x20 | 6.06 (5.67-6.07) | 7,725 (7,321-7,748) | 126.2 | 32.50 |
| homer.png | image | 231 | 211 | 56x28 -> 19x10 | 3.43 (3.34-3.46) | 4,029 (3,964-4,081) | 57.4 | 38.50 |
| text 556 tok | text | 556 | - | - | - | 1,060 (1,048-1,060) | 525.0 | 44.50 |
| text 1038 tok | text | 1,038 | - | - | - | 1,803 (1,800-1,809) | 576.2 | 30.10 |

Image-row TTFT includes the tower seconds; the prefill rate is the
server's own prefill timer over the whole prompt build, so it also
includes the tower. homer.png ends early (finish=stop, 59 tokens)
in every pass.

