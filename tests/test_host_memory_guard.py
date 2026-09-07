#!/usr/bin/env python3
"""Small process tests; never create actual host memory pressure."""
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("guard", ROOT / "tools/host_memory_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class HostMemoryGuardTest(unittest.TestCase):
    def test_resident_owner_reduces_worker_budget(self):
        self.assertEqual(guard.memory_limits(20, 24, 21, 12), (8, 7.2))
        self.assertEqual(guard.memory_limits(118, 108, 102, 12), (106, 102))
        with self.assertRaises(ValueError):
            guard.memory_limits(12, 24, 21, 12)

    def test_refuse_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = pathlib.Path(tmp) / "launched"
            args = guard.parse_args(["--log", str(pathlib.Path(tmp)/"guard.jsonl"),
                "--", sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])
            rc = guard.run_guard(args, lambda: {"available_gib": 11, "psi_full_avg10": 0})
            self.assertEqual(rc, guard.GUARD_EXIT)
            self.assertFalse(marker.exists())

    def test_low_memory_kills_scope_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = pathlib.Path(tmp) / "pids.json"
            log = pathlib.Path(tmp) / "guard.jsonl"
            payload = (
                "import os,signal,subprocess,sys,time,json; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],start_new_session=True); "
                f"open({str(marker)!r},'w').write(json.dumps([os.getpid(),p.pid])); "
                "time.sleep(60)"
            )
            args = guard.parse_args(["--max-gib", "0.25", "--high-gib", "0.2",
                "--grace-seconds", "0.2", "--timeout", "10", "--log", str(log),
                "--", sys.executable, "-c", payload])
            def sample():
                return {"available_gib": 11 if marker.exists() else 20,
                        "psi_full_avg10": 0}
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            fsync = os.fsync
            output = print
            def failing_log(fd):
                if marker.exists():
                    raise OSError('simulated log filesystem failure during cleanup')
                return fsync(fd)
            def failing_stderr(*args, **kwargs):
                if marker.exists():
                    raise BrokenPipeError('simulated closed watchdog stderr')
                return output(*args, **kwargs)
            try:
                # Force the cgroup filesystem backstop instead of systemctl.
                with mock.patch.object(guard, 'signal_scope', return_value=-1), \
                     mock.patch.object(guard.os, 'fsync', side_effect=failing_log), \
                     mock.patch('builtins.print', side_effect=failing_stderr):
                    self.assertEqual(guard.run_guard(args, sample), guard.GUARD_EXIT)
                self.assertIsNone(unrelated.poll(), "guard must leave unrelated owner/session alone")
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)
            self.assertTrue(marker.exists())
            for pid in json.loads(marker.read_text()):
                stat = pathlib.Path(f"/proc/{pid}/stat")
                self.assertTrue(not stat.exists() or stat.read_text().split()[2] == "Z")
            events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
            self.assertIn("terminate", events)
            self.assertIn("kill", events)

    def test_systemctl_errors_are_contained(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('systemctl', 3)):
            with mock.patch.object(guard.subprocess, 'run', side_effect=error):
                self.assertEqual(guard.signal_scope('unused.scope', 'SIGKILL'), -1)

    def check_stopped_guard(self, stop_signal):
        with tempfile.TemporaryDirectory() as tmp:
            marker = pathlib.Path(tmp) / 'pid'
            payload = (
                "import os,signal,time,pathlib; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
            )
            proc = subprocess.Popen([sys.executable, str(ROOT/'tools/host_memory_guard.py'),
                '--max-gib', '0.25', '--high-gib', '0.2', '--grace-seconds', '0.3',
                '--timeout', '10', '--log', str(pathlib.Path(tmp)/'guard.jsonl'),
                '--', sys.executable, '-c', payload])
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists())
            proc.send_signal(stop_signal)
            time.sleep(0.15)
            proc.send_signal(stop_signal)
            self.assertEqual(proc.wait(timeout=5), guard.GUARD_EXIT)
            stat = pathlib.Path(f'/proc/{marker.read_text()}/stat')
            self.assertTrue(not stat.exists() or stat.read_text().split()[2] == 'Z')

    def test_repeated_guard_sigterm(self):
        self.check_stopped_guard(signal.SIGTERM)

    def test_guard_sighup(self):
        self.check_stopped_guard(signal.SIGHUP)

    def test_cgroup_limits_are_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            limits = pathlib.Path(tmp) / "limits.json"
            payload = (
                "import json,pathlib,sys; "
                "cg=pathlib.Path('/sys/fs/cgroup'+pathlib.Path('/proc/self/cgroup').read_text().strip().split('::')[1]); "
                "data={n:(cg/n).read_text().strip() for n in ['memory.max','memory.high','memory.swap.max']}; "
                "data['oom_score_adj']=pathlib.Path('/proc/self/oom_score_adj').read_text().strip(); "
                f"pathlib.Path({str(limits)!r}).write_text(json.dumps(data)); sys.exit(7)"
            )
            args = guard.parse_args(["--max-gib", "0.0625", "--high-gib", "0.05",
                "--timeout", "10", "--log", str(pathlib.Path(tmp)/"guard.jsonl"),
                "--", sys.executable, "-c", payload])
            self.assertEqual(guard.run_guard(args), 7)
            actual = json.loads(limits.read_text())
            self.assertEqual(int(actual['memory.max']), 64 * 1024 * 1024)
            self.assertLess(int(actual['memory.high']), int(actual['memory.max']))
            self.assertEqual(actual['memory.swap.max'], '0')
            self.assertEqual(actual['oom_score_adj'], '800')


if __name__ == "__main__":
    unittest.main()
