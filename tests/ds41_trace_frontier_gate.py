#!/usr/bin/env python3
"""Run each live runner's exact trace-selection pipelines on frontier fixtures."""

import os
from pathlib import Path
import re
import subprocess
import tempfile


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    common = "[dspark] pos 8 k 5 val 0.000 acc 0 conf 0.500\n[dspark] pos 9 k 5 val 0.000 acc 0 conf 0.500\n"
    final = "[dspark] pos 10 k 5 val 0.000 acc 0 conf 0.500\n"
    emits = "[emit] 8 1\n[emit] 9 2\n[emit] 10 3\n"
    cases = [("unused final round", common + final, common, True),
             ("contributing round changed", common + final, common.replace("pos 9 k 5", "pos 9 k 4"), False),
             ("same contributing rounds", common, common, True)]
    failures = 0
    with tempfile.TemporaryDirectory(prefix="ds41-trace-frontier-") as directory:
        temp = Path(directory)
        for runner in ["ds41_generate_gate.sh", "ds41_graph_gate.sh", "ds41_serving_gate.sh"]:
            source = (root / "tests" / runner).read_text()
            pipelines = [line.strip() for line in source.splitlines() if "grep -o '\\[dspark\\] .*'" in line]
            if len(pipelines) != 2:
                raise RuntimeError(f"{runner}: expected two trace-selection pipelines")
            assignment = re.search(r"^\s*LAST_EMIT=.*$", source, re.M)
            commands = ([assignment.group().strip()] if assignment else []) + pipelines
            for name, engine, port, expected in cases:
                engine_path = temp / "engine.log"
                engine_path.write_text(emits + engine)
                log = temp / "port"
                Path(str(log) + ".spec.log").write_text(emits + port)
                environment = dict(os.environ, ENGLOG_SPEC=str(engine_path), LOG=str(log))
                subprocess.run(["sh", "-c", "\n".join(commands)], env=environment, check=True)
                lhs = Path(str(log) + ".eng.dspark").read_text()
                rhs = Path(str(log) + ".port.dspark").read_text()
                result = bool(lhs and rhs) and lhs == rhs
                if result != expected:
                    print(f"FAIL {runner}: {name}, matched={result}, expected={expected}")
                    failures += 1
    print(f"ds41 trace frontier gate: {'FAIL' if failures else 'PASS'} ({failures}/9 failures)")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
