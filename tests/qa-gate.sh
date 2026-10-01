#!/usr/bin/env bash
# tests/qa-gate.sh - rule 19 QA guardrail mount for ds4-dfm-rs.
#
# Adapted from the canonical generic template
# (~/.opengrok/templates/qa-gate.sh).  This project's deliverables are C/CUDA
# kernels and make targets, not DB functions, API endpoints and web client
# methods, so the template's db|ep|cm surface extraction is replaced by an
# explicit surface list; the checks that make the evidence operative are the
# template's, unchanged:
#   1. the QA report exists;
#   2. it is at least as new as the base the unit is diffed against;
#   3. its LAST non-empty line reads "verdict: overall PASS" (nothing after it,
#      so a superseded PASS line cannot go green over a later FAIL);
#   4. it names every surface the unit touches.
# The mount SKIPS in real CI, where the AI QA-tester is not provisioned.
#
# Env knobs:
#   QA_BASE      ref the unit is diffed against (default origin/<branch>)
#   QA_MODEL     QA-tester model, used in messages only; the operator picks it
#                per project, so there is no default here
#   QA_SURFACES  newline list of surface names the report must cover
#                (default: every file the unit changed)
#   QA_REPORT    report path (default qa-evidence/qa-report.md)
#
# Exit 0 = green, exit 1 = red.
cd "$(dirname "$0")/.." || exit 1

# Real CI: the QA subagent is a local-session capability, not provisioned there.
if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  echo "SKIP tests/qa-gate.sh (AI QA-tester is a local-session step, not provisioned in CI)"
  exit 0
fi

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo main)
# The base is whatever is already published for this branch: the first remote
# that carries it.  This workspace pushes to the "fork" remote, because our
# account has read-only access to the upstream origin; a gate that insisted on
# origin/<branch> there would find no surfaces and pass everything.
BASE=${QA_BASE:-}
if [[ -z "$BASE" ]]; then
  BASE="origin/${BRANCH}"
  for cand in "origin/${BRANCH}" "fork/${BRANCH}"; do
    if git rev-parse --verify --quiet "refs/remotes/${cand}" >/dev/null 2>&1; then
      BASE="$cand"
      break
    fi
  done
fi
MODEL=${QA_MODEL:-"the per-project QA model"}
REPORT=${QA_REPORT:-qa-evidence/qa-report.md}
echo "QA gate: branch ${BRANCH}, base ${BASE}"

BASE_SHA=$(git rev-parse "${BASE}" 2>/dev/null || echo "")
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo "")
base_sec=0
if [[ -n "$BASE_SHA" && "$BASE_SHA" != "$HEAD_SHA" ]]; then
  base_sec=$(git log -1 --format=%ct "$BASE" 2>/dev/null || echo 0)
fi

# Surfaces: the files this unit changes, committed or not, unless the caller
# names them explicitly.  Machine-local and regenerable files are not review
# surfaces: the QA evidence itself, the knowledge-graph and callgraph outputs
# (rule 12), object files, built test binaries and the session handoff.
DEFAULT_EXCLUDE='^(qa-evidence/|graphify-out/|\.callgraph-index\.bin|tests/cuda_long_context_smoke|.*-handoff\.md$|.*\.o$|.*\.bin$)'
if [[ -n "${QA_SURFACES:-}" ]]; then
  mapfile -t SURFACES <<< "$QA_SURFACES"
else
  mapfile -t SURFACES < <(
    {
      if [[ -n "$BASE_SHA" && "$BASE_SHA" != "$HEAD_SHA" ]]; then
        git diff --name-only "$BASE"...HEAD 2>/dev/null
      fi
      git status --porcelain 2>/dev/null | awk '{print $NF}'
    } | sed '/^$/d' | grep -vE "${QA_SURFACE_EXCLUDE:-$DEFAULT_EXCLUDE}" | sort -u
  )
fi

if (( ${#SURFACES[@]} == 0 )); then
  echo "QA GATE: ALL PASS (no changed surfaces relative to ${BASE})"
  exit 0
fi

echo "Surfaces this unit touches (QA model: ${MODEL}):"
printf '  %s\n' "${SURFACES[@]}"

failures=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; failures=$((failures+1)); fi; }

check "QA report exists (${REPORT})" "[[ -f '$REPORT' ]]"
if [[ -z "$BASE_SHA" ]]; then
  echo "SKIP  freshness (${BASE} does not exist yet: the branch is not published)"
elif [[ "$BASE_SHA" == "$HEAD_SHA" ]]; then
  echo "SKIP  freshness (HEAD is level with ${BASE}; nothing published is behind it)"
else
  check "QA report is fresh (>= last pushed commit at ${BASE})" \
    "[[ -f '$REPORT' ]] && [[ '$(stat -c %Y "$REPORT" 2>/dev/null || echo 0)' -ge '$base_sec' ]]"
fi

# The operative verdict is the report's LAST non-empty line, and nothing else:
# an appended line after a PASS verdict, or a FAIL below it, must both go red.
last_verdict=$(grep -v '^[[:space:]]*$' "$REPORT" 2>/dev/null | tail -1 | tr 'A-Z' 'a-z' | tr -s ' ' | sed 's/[[:space:]]*$//')
check "operative verdict is PASS (last non-empty line: ${last_verdict:-none})" \
  "[[ '$last_verdict' == 'verdict: overall pass' ]]"

while IFS= read -r surf; do
  [[ -z "$surf" ]] && continue
  check "report covers surface: $surf" "grep -qF '$surf' '$REPORT'"
done <<< "$(printf '%s\n' "${SURFACES[@]}")"

if (( failures == 0 )); then
  echo "QA GATE: overall PASS (${MODEL} evidence present, operative, covering)"
  exit 0
fi
echo "QA GATE: FAIL (${failures} check(s) red). Run the AI QA-tester on ${MODEL}, have it"
echo "verify the surfaces above live, and write ${REPORT} ending in"
echo "'verdict: overall PASS' before committing for delivery or pushing."
exit 1
