#!/usr/bin/env python3
"""Regression check: the V4.1 (ds41) run shape the proof harness builds.

Model-free: the command builder, the plan fingerprint and the prompt
tokenizer are exercised with a stub binary and literal ids, so this runs
wherever Python does (the real runs need the Spark and the artifact).
"""

import importlib.util
import json
import stat
import sys
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ds4_proof", Path(__file__).with_name("ds4_proof.py")
)
assert spec and spec.loader
proof = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = proof
spec.loader.exec_module(proof)

profile = proof.EngineProfile(
    name="cuda-eager",
    env={"DS4_CUDA_LAYER_GRAPHS": "0"},
    args=[],
    backend="cuda",
)

# --gen-ids replaces the prompt, and the engram feed is passed explicitly:
# the artifact records the converting host's shard paths, so a run without
# it dies at engram-open.
cmd = proof.build_command(
    bin_path="/tmp/ds4",
    base_model="/mnt/model.gguf",
    mtp_model=None,
    profile=profile,
    prompt=None,
    prompt_file=None,
    tokens=1024,
    temperature=0.0,
    nothink=True,
    dump_logprobs_path=Path("/tmp/cell.logprobs.json"),
    v41=proof.V41Run(
        ids_path="/tmp/prompt.v41.ids",
        engram_dir="/mnt/engram",
        zchain="/mnt/sidecar",
        no_graph=True,
    ),
)
assert "--gen-ids" in cmd and "/tmp/prompt.v41.ids" in cmd, cmd
assert "--engram-dir" in cmd and "/mnt/engram" in cmd, cmd
assert "--zchain" in cmd and "/mnt/sidecar" in cmd, cmd
assert "--no-graph" in cmd, cmd
# The profile env cannot deliver the switch on this route (ds4 sets the v41
# graph global explicitly), so the eager arm must carry the flag, and the
# capture arm (no flag) must not.
assert "--prompt-file" not in cmd and "-p" not in cmd, cmd
capture = proof.build_command(
    bin_path="/tmp/ds4", base_model="/mnt/model.gguf", mtp_model=None,
    profile=profile, prompt=None, prompt_file=None, tokens=1024,
    temperature=0.0, nothink=True,
    v41=proof.V41Run(ids_path="/tmp/prompt.v41.ids", engram_dir="/mnt/engram"),
)
assert "--no-graph" not in capture, capture

# The prompt requirement stays for every other family.
try:
    proof.build_command(
        bin_path="/tmp/ds4", base_model="/mnt/model.gguf", mtp_model=None,
        profile=profile, prompt=None, prompt_file=None, tokens=8,
        temperature=0.0, nothink=True,
    )
    raise AssertionError("a prompt-less non-V4.1 run must be refused")
except ValueError as e:
    assert "prompt" in str(e), e

# The tokenizer step: the stub prints the id list the way --dump-tokens does.
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    stub = work / "ds4-stub"
    stub.write_text('#!/bin/sh\necho "[1, 2, 3]"\necho "     1  a"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    case = proof.PromptCase(id="probe", prompt="hello")
    ids_path = proof.tokenize_v41_prompt(
        bin_path=str(stub), base_model="/mnt/model.gguf", prompt_case=case, work_dir=work
    )
    assert Path(ids_path).read_text() == "1 2 3\n", Path(ids_path).read_text()
    # Cached: a second call must not re-run the binary.
    assert proof.tokenize_v41_prompt(
        bin_path=str(work / "missing"), base_model="/mnt/model.gguf",
        prompt_case=case, work_dir=work,
    ) == ids_path

# The plan keeps its committed fingerprint for every other scenario, and the
# V4.1 shape is recorded only when it is in use.
base_plan = dict(
    scenario_name="cuda-long-context-full", suite="argmax_generation", tokens=1024,
    budget="default-on", profiles=[profile], prompts=[proof.PromptCase(id="p", prompt="x")],
    contracts=[], expected_gen_tokens_min=1000, bin_path="/tmp/ds4",
    base_model="/mnt/model.gguf", mtp_model=None, temperature=0.0, nothink=True,
)
plain = proof.build_expanded_plan(**base_plan)
assert "v41" not in plain, plain
v41_plan = proof.build_expanded_plan(**base_plan, v41_engram="/mnt/engram", v41_zchain="/mnt/sidecar")
assert v41_plan["v41"] == {"engram_dir": "/mnt/engram", "zchain": "/mnt/sidecar"}
assert json.dumps(plain, sort_keys=True) == json.dumps(
    {k: v for k, v in v41_plan.items() if k != "v41"}, sort_keys=True
)

print("ds4 proof v41 smoke: PASS")
