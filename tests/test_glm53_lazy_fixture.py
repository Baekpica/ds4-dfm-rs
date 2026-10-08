#!/usr/bin/env python3
"""Exercise actual GLM lazy session capture, fit and graph allocation.

Uses toy tensor dimensions and GPU/state mocks. Production row-buffer schema,
budget arithmetic and GLM create/allocate branches are extracted unchanged.
This checks policy lifetime, not model arithmetic or real GPU memory fitting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

from test_glm53_close_fixture import ROOT, run
from test_glm53_prefill_fixture import body


def branch(source, function, condition):
    code = body(source, function)
    start = code.index("    if (" + condition + ") {")
    # Reuse the balanced function extractor for the family branch itself.
    selected = "static int selected(void) {" + code[start:].split("{", 1)[1]
    return body(selected, "selected").split("{", 1)[1].rsplit("}", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs="*", default=[
        "rows", "window-on", "window-off", "retry", "eager"])
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["ds4.c", "ds4_glm53_graph.inc", "tests/test_glm53_lazy.c"]
    contents = {name: (ROOT / name).read_bytes() for name in names}
    native, graph = (contents[name].decode() for name in names[:2])
    schema = graph[graph.index("#define GLM53_ROW_BUFFERS(X)"):
                   graph.index("static void glm53_graph_layers")]
    schema_path = output / "rows.inc"
    schema_path.write_text(schema)
    helpers = ["glm53_graph_row_cap", "glm53_window_cap", "glm53_prefill_cap",
               "glm53_window_bytes", "glm53_graph_diag", "glm53_graph_ffn_cols",
               "glm53_graph_layers", "glm53_graph_state_bytes"]
    if "static uint64_t glm53_graph_bytes_plan(" in graph:
        helpers.append("glm53_graph_bytes_plan")
    helpers += ["glm53_graph_bytes_for"]
    code = "\n".join(body(graph, name) for name in helpers)
    code += "typedef enum { GLM53_MTP_OFF, GLM53_MTP_ON } glm53_mtp_mode;\n"
    code += body(graph, "glm53_session_bytes")
    allocation_helper = "glm53_graph_alloc_plan" \
        if "static bool glm53_graph_alloc_plan(" in graph else "glm53_graph_alloc"
    code += body(graph, allocation_helper)
    code += body(native, "glm53_graph_session_fit_check")
    allocation = branch(native, "ds4_session_alloc_graph", "ds4_session_is_glm53(s)")
    creation = branch(native, "ds4_session_create", "DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM53")
    code += "static int ds4_session_alloc_graph(ds4_session *s) {\n"
    code += "    ds4_engine *e = s->engine;\n" + allocation + "}\n"
    code += "static int create(ds4_session **out, ds4_engine *e, int ctx_size) {\n"
    code += creation + "}\n"
    header = output / "session.inc"
    header.write_text(code)
    for name, raw in contents.items():
        saved = output / "source" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(raw)
    flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] \
        if args.sanitize else ["-O2"]
    binary = output / "test-session"
    command = shlex.split(args.cc) + flags + ["-std=gnu11", "-Wall", "-Wextra", "-Werror",
        f'-DGLM53_ROW_FIXTURE="{schema_path}"', f'-DGLM53_SESSION_FIXTURE="{header}"',
        "-o", str(binary), str(ROOT / names[2])]
    built = run(command, output, "build")
    receipt = dict(sources={name: hashlib.sha256(raw).hexdigest()
                           for name, raw in contents.items()},
                   unchanged=all((ROOT / name).read_bytes() == raw
                                 for name, raw in contents.items()),
                   coordinator_sha256=hashlib.sha256(code.encode()).hexdigest(),
                   scope="actual GLM session/fit/row allocation; toy shapes; GPU/state mocks")
    (output / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["unchanged"]:
        raise ValueError("source changed during fixture compilation")
    if built:
        return 1
    env = os.environ.copy()
    if args.sanitize:
        env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1")
    failed = False
    for case in args.cases:
        failed |= run([str(binary), case], output, case, env) != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
