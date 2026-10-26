#!/usr/bin/env python3
"""Linux manual gate: python3 tests/agent_acl_window.py <ds4-cli lib-test binary>.

Uses strace to pause the existing ACL fixture before it removes the inherited
staging ACL. No other user may gain access while replacement is prepared.
"""

import errno
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time


def check(binary):
    observed = 0
    exposed = False
    with tempfile.TemporaryDirectory(prefix="ds4-acl-window-") as scratch:
        trace = Path(scratch) / "trace"
        process = subprocess.Popen(
            [
                "strace", "-f", "-o", str(trace),
                "-e", "trace=fremovexattr",
                "-e", "inject=fremovexattr:delay_enter=1s:when=2",
                binary, "--exact", "agent::write_parity::audit_access_acl",
                "--nocapture",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
        while process.poll() is None:
            try:
                pids = children.read_text().split()
            except FileNotFoundError:
                break
            for pid in pids:
                for path in Path("/tmp").glob(f"ds4_agent_write_{pid}_*_acl/.ds4-write-*.tmp"):
                    try:
                        value = os.getxattr(path, "system.posix_acl_access")
                    except OSError as error:
                        if error.errno in (errno.ENOENT, errno.ENODATA):
                            continue
                        raise
                    entries = list(struct.iter_unpack("<HHI", value[4:]))
                    mask = next(perm for tag, perm, _ in entries if tag == 16)
                    inherited = [perm for tag, perm, uid in entries if tag == 2 and uid == 65534]
                    if inherited:
                        observed += 1
                        exposed |= bool(inherited[0] & mask)
            time.sleep(0.005)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        assert "(DELAYED)" in trace.read_text(), "ACL removal was not delayed"
        assert observed, "did not observe an inherited staging ACL"
        assert not exposed, "staging ACL exposed new file contents to inherited user 65534"
    print(f"staging ACL remained inaccessible ({observed} observations)")


if __name__ == "__main__":
    check(sys.argv[1])
