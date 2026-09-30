#!/usr/bin/env bash
# tests/qa-gate.sh — rule 19 guardrail mount for ds4-dfm-rs.
#
# Copied from the canonical template ~/.opengrok/templates/qa-gate.sh and
# parameterized for this project: the surface classes are the ones a family
# unit actually adds here (C ABI entries in ds4_gpu.h, DS4_* environment
# knobs, Makefile targets, and the public items of the Rust host), because the
# template's DB/endpoint/client-method classes match nothing in an inference
# engine.  The contract is unchanged: before a push, an AI QA-tester (model
# selected per project, never hardcoded) must write a FRESH
# qa-evidence/qa-report.md whose LAST non-empty line is `verdict: overall PASS`
# and which covers every surface added relative to the base ref.  The mount
# FAILS the local gate (exit 1) until that holds; it SKIPS in real CI, where
# the QA subagent cannot run.
#
# Env knobs:
#   QA_BASE           ref the new surfaces are diffed against
#                     (default origin/<current branch>, falling back to
#                     origin/main on a branch that is not pushed yet)
#   QA_MODEL          QA-tester model name, used in messages only; the project's
#                     recorded choice is deepseek-v4.1-CC-flash
#   QA_SURFACE_PATHS  newline list of surface paths (default: below)
#   QA_REPORT         path to the report (default qa-evidence/qa-report.md)
#
# Exit 0 = green (no new surfaces, or report fresh+pass+covering). Exit 1 red.
cd "$(dirname "$0")/.." || exit 1

if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  echo "SKIP tests/qa-gate.sh (AI QA-tester is a local-session step, not provisioned in CI)"
  exit 0
fi

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
BASE=${QA_BASE:-origin/${BRANCH}}
MODEL=${QA_MODEL:-deepseek-v4.1-CC-flash}
REPORT=${QA_REPORT:-qa-evidence/qa-report.md}

# --verify --quiet is required: a plain `git rev-parse <unresolvable-ref>`
# echoes the argument on stdout and exits non-zero, which would leave BASE_SHA
# looking set and silently turn every diff below into an empty one.
BASE_SHA=$(git rev-parse --verify --quiet "${BASE}^{commit}" 2>/dev/null)
if [[ -z "$BASE_SHA" ]]; then
  BASE=origin/main
  BASE_SHA=$(git rev-parse --verify --quiet "${BASE}^{commit}" 2>/dev/null)
fi
HEAD_SHA=$(git rev-parse --verify --quiet "HEAD^{commit}" 2>/dev/null)
if [[ -z "$BASE_SHA" || -z "$HEAD_SHA" || "$BASE_SHA" == "$HEAD_SHA" ]]; then
  echo "QA GATE: ALL PASS (no unpushed commits yet; nothing new to QA-tester)"
  exit 0
fi

# ---- surface paths: project default -------------------------------------
if [[ -n "${QA_SURFACE_PATHS:-}" ]]; then
  mapfile -t PATHS <<< "$QA_SURFACE_PATHS"
else
  PATHS=(ds4_gpu.h ds4.c Makefile crates/ds4-core/src crates/ds4-cli/src)
fi

abi=""; knobs=""; targets=""; rust=""
for path in "${PATHS[@]}"; do
  # Only paths that exist at HEAD and were touched are surface sources.
  if ! git cat-file -e "${HEAD_SHA}:${path}" 2>/dev/null; then continue; fi
  added=$(git diff "${BASE}" "${HEAD_SHA}" -- "$path" 2>/dev/null | grep -E '^\+[^+]')
  [[ -z "$added" ]] && continue
  case "$path" in
    ds4_gpu.h)
      abi+="$(grep -oE 'ds4_gpu_[a-z0-9_]+[[:space:]]*\(' <<< "$added" \
              | sed -E 's/[[:space:]]*\($//' | sort -u)"$'\n';;
    ds4.c)
      knobs+="$(grep -oE 'getenv\("DS4_[A-Z0-9_]+"' <<< "$added" \
              | sed -E 's/getenv\("//; s/"$//' | sort -u)"$'\n';;
    Makefile)
      targets+="$(grep -oE '^\+[a-z0-9-]+:' <<< "$added" \
              | sed -E 's/^\+//; s/:$//' | sort -u)"$'\n';;
    crates/*)
      rust+="$(grep -oE 'pub (fn|const|struct|enum) [a-zA-Z0-9_]+' <<< "$added" \
              | sort -u)"$'\n';;
  esac
done

all_surfaces=$(printf '%s\n%s\n%s\n%s' "$abi" "$knobs" "$targets" "$rust" \
               | sed '/^$/d' | sort -u)
if [[ -z "$all_surfaces" ]]; then
  echo "QA GATE: ALL PASS (no new C ABI entry, DS4_* knob, make target or Rust item relative to ${BASE})"
  exit 0
fi
echo "New surfaces to be QA'd (${MODEL}):"
echo "$all_surfaces" | sed 's/^/  /'

failures=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; failures=$((failures+1)); fi; }

check "QA report exists ($(basename "$REPORT"))" "[[ -f '$REPORT' ]]"
base_sec=$(git log -1 --format=%ct "$BASE" 2>/dev/null || echo 0)
check "QA report is fresh (>= last pushed commit at ${BASE})" \
  "[[ -f '$REPORT' ]] && [[ '$(stat -c %Y "$REPORT" 2>/dev/null || echo 0)' -ge '$base_sec' ]]"
# The operative verdict is the report's LAST non-empty line and nothing else: a
# FAILed report, or a report with trailing prose after the verdict, fails.
last_verdict=$(grep -v '^[[:space:]]*$' "$REPORT" 2>/dev/null | tail -1 | tr 'A-Z' 'a-z' | tr -s ' ' | sed 's/[[:space:]]*$//')
check "operative verdict is PASS (report's last non-empty line: ${last_verdict:-none})" \
  "[[ '$last_verdict' == 'verdict: overall pass' ]]"

while IFS= read -r surf; do
  [[ -z "$surf" ]] && continue
  check "report covers surface: $surf" "grep -qF '$surf' '$REPORT'"
done <<< "$all_surfaces"

if (( failures == 0 )); then
  echo "QA GATE: overall PASS (${MODEL} evidence fresh and covering)"
  exit 0
else
  echo "QA GATE: FAIL (${failures} check(s) red). Run the AI QA-tester on ${MODEL},"
  echo "have it verify the surfaces above live, and update $REPORT with"
  echo "'verdict: overall PASS' before committing/pushing."
  exit 1
fi
