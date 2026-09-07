## Benchmarking

This directory collects manual prefill/generation sweeps and historical
hardware CSVs. Record the exact artifact, build, backend and workload when
adding results; old CSVs are not current release qualification.

For CUDA phase attribution and diagnosis, use the
[ds4-perf workflow](../docs/prefill-decode-optimization-playbook.md#local-scout-with-ds4-perf):
`make ds4-bench-perf`, then `ds4-perf doctor` and `scout`.

Example Metal sweep (add `--cuda` and use an appropriate filename for CUDA):

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv speed-bench/m3_max.csv
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.
