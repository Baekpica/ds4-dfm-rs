#!/usr/bin/env python3
"""Run a Linux model job with a cgroup ceiling and a whole-host watchdog.

An existing weight owner consumes the same physical memory as its worker.
Derive the new job's ceiling from MemAvailable, then keep watching the host:
CUDA allocations are not necessarily all charged to the job's cgroup.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
GUARD_EXIT = 75


def memory_snapshot():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024 / GIB
    pressure = Path("/proc/pressure/memory").read_text().splitlines()
    full = next(line for line in pressure if line.startswith("full "))
    psi = float(dict(item.split("=") for item in full.split()[1:])["avg10"])
    return {"available_gib": available, "psi_full_avg10": psi}


def memory_limits(available, requested_max, requested_high, reserve):
    maximum = min(requested_max, math.floor((available - reserve) * 10) / 10)
    if maximum <= 0:
        raise ValueError(f"only {available:.2f} GiB available; reserve is {reserve:.2f} GiB")
    high = min(requested_high, maximum - min(1.0, maximum * 0.1))
    return maximum, high


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-gib", type=float, default=24)
    parser.add_argument("--high-gib", type=float, default=21)
    parser.add_argument("--reserve-gib", type=float, default=12)
    parser.add_argument("--trip-gib", type=float, help="watchdog floor; defaults to reserve, use a lower floor for a stable owner")
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--grace-seconds", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=1800, help="seconds; 0 keeps monitoring a server indefinitely")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if args.trip_gib is None:
        args.trip_gib = args.reserve_gib
    for name in ("max_gib", "high_gib", "reserve_gib", "poll_seconds", "grace_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.trip_gib) or not 0 < args.trip_gib <= args.reserve_gib:
        parser.error("trip-gib must be positive and no greater than reserve-gib")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("timeout must be finite and nonnegative")
    if not args.command:
        parser.error("a command is required after --")
    return args


def signal_scope(unit, sig):
    # Kill every descendant, including a profiler that started a new session.
    try:
        return subprocess.run(
            ["systemctl", "--user", "kill", "--kill-whom=all", f"--signal={sig}", unit],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.5,
            check=False).returncode
    except (OSError, subprocess.TimeoutExpired):
        return -1


@contextmanager
def stop_signals():
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield stopped
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def scope_path(unit):
    manager = subprocess.check_output(
        ["systemctl", "--user", "show", "-p", "ControlGroup", "--value"],
        text=True, timeout=3).strip()
    if not manager.startswith("/") or ".." in manager.split("/"):
        raise ValueError("user manager has no valid cgroup")
    return Path("/sys/fs/cgroup" + manager) / "ds4guard.slice" / unit


def scope_populated(path):
    try:
        return "populated 1" in (path / "cgroup.events").read_text()
    except FileNotFoundError:
        return False
    except OSError:
        return True  # Unknown cleanup state must fail closed.


def kill_scope_files(path):
    """Backstop if the user service manager is stalled or unreachable."""
    try:
        (path / "cgroup.kill").write_text("1\n")
        return
    except FileNotFoundError:
        return
    except OSError:
        pass
    # Some systemd versions delegate cgroup.procs but not cgroup.kill.
    for procs in path.rglob("cgroup.procs"):
        try:
            pids = procs.read_text().split()
        except OSError:
            continue
        for item in pids:
            try:
                pid = int(item)
                fd = os.pidfd_open(pid)
                try:
                    membership = Path(f"/proc/{pid}/cgroup").read_text()
                    relative = str(path).removeprefix("/sys/fs/cgroup")
                    if any(line == "0::" + relative or line.startswith("0::" + relative + "/")
                           for line in membership.splitlines()):
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)
            except OSError:
                pass


def run_guard(args, read_memory=None):
    read_memory = read_memory or memory_snapshot
    args.log.parent.mkdir(parents=True, exist_ok=True)
    unit = f"ds4-hostguard-{os.getpid()}-{time.time_ns()}.scope"
    with args.log.open("a", buffering=1) as log, stop_signals() as stopped:
        proc = None
        log_failed = False
        def record(event, **data):
            nonlocal log_failed
            try:
                log.write(json.dumps({"time": time.time(), "event": event, "unit": unit, **data}) + "\n")
                log.flush()
                os.fsync(log.fileno())
            except OSError:
                if proc is None:
                    raise
                log_failed = True  # Logging failure must never interrupt cleanup.

        try:
            before = read_memory()
            maximum, high = memory_limits(before["available_gib"], args.max_gib,
                                          args.high_gib, args.reserve_gib)
            cgroup = scope_path(unit)
            if stopped[0]:
                raise ValueError("interrupted before launch")
        except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as error:
            record("refused", reason=str(error))
            print(f"host-memory-guard: refused: {error}", file=sys.stderr, flush=True)
            return GUARD_EXIT
        record("start", **before, max_gib=maximum, high_gib=high,
               reserve_gib=args.reserve_gib, trip_gib=args.trip_gib, command=args.command)
        print(f"host-memory-guard: available={before['available_gib']:.2f} GiB, "
              f"scope max/high={maximum:.2f}/{high:.2f}, reserve={args.reserve_gib:.2f}, "
              f"trip={args.trip_gib:.2f}", flush=True)
        command = ["systemd-run", "--user", "--scope", "--slice=ds4guard.slice", "--quiet", f"--unit={unit}",
                   "-p", f"MemoryMax={int(maximum * GIB)}",
                   "-p", f"MemoryHigh={int(high * GIB)}", "-p", "MemorySwapMax=0",
                   "bash", "-c", 'echo 800 > /proc/self/oom_score_adj; exec "$@"',
                   "_", *args.command]
        # The watchdog stays outside the payload's cgroup and OOM preference.
        proc = subprocess.Popen(command, start_new_session=True)
        start = last_log = time.monotonic()
        reason = None
        try:
            while proc.poll() is None:
                snapshot = read_memory()
                now = time.monotonic()
                if log_failed:
                    reason = "watchdog telemetry could not be saved"
                elif stopped[0]:
                    reason = "watchdog received a stop signal"
                elif snapshot["available_gib"] < args.trip_gib:
                    reason = "host memory watchdog floor crossed"
                elif (snapshot["psi_full_avg10"] >= 20 and
                      snapshot["available_gib"] < args.trip_gib + 4):
                    reason = "sustained host memory stalls near reserve"
                elif args.timeout and now - start >= args.timeout:
                    reason = "job deadline exceeded"
                if reason:
                    record("terminate", reason=reason, **snapshot)
                    break
                if now - last_log >= 1:
                    record("sample", **snapshot)
                    last_log = now
                time.sleep(args.poll_seconds)
        except (OSError, ValueError, KeyError, StopIteration, KeyboardInterrupt) as error:
            reason = f"watchdog interrupted: {error}"
            record("terminate", reason=reason)
        finally:
            if reason:
                try:
                    print(f"host-memory-guard: {reason}; terminating {unit}", file=sys.stderr, flush=True)
                except OSError:
                    pass
                if signal_scope(unit, "SIGTERM") == 0:
                    grace_deadline = time.monotonic() + args.grace_seconds
                    while time.monotonic() < grace_deadline:
                        try:
                            if read_memory()["available_gib"] < args.trip_gib * 2 / 3:
                                break
                        except (OSError, ValueError, KeyError, StopIteration):
                            break
                        time.sleep(min(args.poll_seconds, max(0, grace_deadline - time.monotonic())))
                record("kill", reason=reason)
            # Also remove descendants if their launcher exited on its own.
            if reason or scope_populated(cgroup):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                kill_scope_files(cgroup)
            signal_scope(unit, "SIGKILL")
            cleanup_deadline = time.monotonic() + 3
            while scope_populated(cgroup) and time.monotonic() < cleanup_deadline:
                kill_scope_files(cgroup)
                time.sleep(0.05)
            if scope_populated(cgroup):
                reason = "scope still has live processes after SIGKILL"
                record("cleanup_failed", reason=reason, cgroup=str(cgroup))
            try:
                rc = proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                reason = "launcher did not exit after SIGKILL"
                record("cleanup_failed", reason=reason)
                rc = -signal.SIGKILL
        result = GUARD_EXIT if reason else (rc if rc >= 0 else 128 - rc)
        record("exit", status=result, payload_status=rc)
        return GUARD_EXIT if log_failed else result


if __name__ == "__main__":
    sys.exit(run_guard(parse_args()))
