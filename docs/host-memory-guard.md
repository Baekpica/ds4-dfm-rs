# Guarding model jobs on a unified-memory host

Run resident owners and their workers through `tools/host_memory_guard.py`.
Separate fixed limits can overcommit one host: an owner allowed 108 GiB and
a worker allowed 24 GiB exceed the 121.6 GiB installed on this GB10.

The guard reads host `MemAvailable` before launch and limits the new job to
the smaller of the requested ceiling and available memory minus a 12 GiB
reserve. It installs `MemoryMax`, `MemoryHigh`, `MemorySwapMax=0`, and an OOM
preference of 800 in a unique user systemd scope. The watchdog stays outside
that scope. Launch the worker only after the owner is ready and stable.

```sh
python3 tools/host_memory_guard.py \
  --max-gib 24 --high-gib 21 --reserve-gib 12 \
  --log scratch/bench.memory.jsonl -- ./ds4-bench <benchmark arguments>
```

For a persistent owner, use its measured ceiling and `--timeout 0`; the
default job deadline is 1,800 seconds. The guard samples the whole host
every 100 ms, including memory that the CUDA driver may not charge to the
scope. It sends SIGTERM when the reserve is crossed, or when memory PSI
full stalls reach 20% near the reserve, then SIGKILL after one second.
Only that job's scope is targeted, including profiler descendants that
create a separate process session. A filesystem cgroup kill is the fallback
if the user service manager does not respond. A worker trip leaves the
separate owner intact. Exit 75 means the job was refused or interrupted by
the guard; do not use its partial benchmark outputs.

Timestamped JSONL samples and stop events are synced to disk. Stop the
campaign on any guard failure, inspect the log and remaining PIDs, then
reclaim page cache only after the affected processes have exited. A
userspace watchdog cannot guarantee recovery from a kernel or GPU-driver
lockup; this guard addresses memory admission and observed memory pressure.

`python3 tests/test_host_memory_guard.py` checks the installed cgroup limits,
owner-aware admission, launch refusal, escaped descendants, user-manager
failure and repeated termination signals using small processes. It does
not deliberately exhaust host memory.
