# MiMo-V2.6-Flash VQ serving on one DGX Spark: October 30

[UNCENSORED VQ release](https://huggingface.co/Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF)
`f1cd8523` (`files.json` as the artifact manifest), GB10, CUDA 13.4, driver
615.71.10, clock range 300–2200 MHz, host memory guard with a 2 GiB reserve.

## Reuse and disk restore

`tests/serving_reuse_live.py --family mimo2`: context 2048, two banks,
chunks 64, partial reuse, 2 GiB disk with one-token thresholds, MTP off.
Seed and warm share a process; `restored` starts a second on the same disk
and `cold` a third with reuse and disk off. Receipts:
[gate evidence](gate-evidence.json).

| Request | Input / cached tokens | Reuse | Answer |
|---|---|---|---|
| Seed | 589 / 0 | cold | 4 |
| Append | 614 / 590 | fork | 5 |
| Edit | 614 / 589 | partial | 6 |
| Branch | 639 / 615 | exact | 8 |
| Restart (disk) | 664 / 640 | fork | 9 |

Every phase passes; the five cold replays match their messages, stop
reasons and completion counts byte for byte.

## Default plan, media and drafting

`ds4-server -m <release>` with no other option plans the native
1,048,576-token context with two embedded MTP drafts on the serial lane:
109.79 of 117.62 GiB with the vision and audio parts and the media
workspace (`--check-config`). Against that server: text answered 42 and 10,
`homer.png` "Homer Simpson.", `photo.jpg` a satellite view of Earth over
Africa, a Homer clip "Homer Simpson", a 440 Hz tone "a continuous,
high-pitched electronic beep", a rising sweep "sweeps upwards in pitch",
white noise "loud hissing or static noise"; the digit probe copied
`1234567890` with speculation active and no fallback. Minimum available
host memory 13.6 GiB.

| Other plans (`--check-config`) | Lane | GiB |
|---|---|---:|
| `--mtp <release>/dflash-q8_0.gguf --mtp-mode on --mtp-draft 7` | serial | 110.27 |
| `--mtp-mode off` with `DS4_SERVER_CONTINUOUS=0` | serial | 108.88 |
| `-c 524288 --mtp-mode off` | one text bank + serial media | 111.77 |
| `-c 262144 --max-seqs 2 --prefix-reuse partial --mtp-mode off` | two text banks + serial media | 109.89 |
| `--mtp-mode off` at 1M | one text bank + serial media | 123.16, refused |
