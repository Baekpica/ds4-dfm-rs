#!/usr/bin/env bash
# Make one MiniMax-H3 clip through the local v1 contract:
#
#   POST /v1/videos               the five-field body -> {"id": "..."}
#   GET  /v1/videos/{id}          poll "status" until completed or failed
#   GET  /v1/videos/{id}/content  the MP4
#
# The body is exactly the five fields of docs/minimax-h3/local-v1-contract.md
# (crates/ds4-core/src/minimax_h3/local_api.rs:58). Nothing else is sent,
# because the server refuses unknown fields by name.
#
#   h3-video.sh "a red kite over a grey sea at dusk"
#   h3-video.sh --seconds 5 --ratio 9:16 --keyframe first.png "the kite turns"
#   h3-video.sh --ref-image chair.png --ref-audio room.wav "the chair drifts"
#
# Options:
#   --url URL            server base (default http://127.0.0.1:8090)
#   --out FILE           output MP4 (default h3-<seed>-<epoch>.mp4)
#   --ratio R            auto | 16:9 | 9:16 | 4:3 | 3:4 | 1:1 (default 16:9)
#   --seconds N          4 through 15 (default 4)
#   --seed N             non-negative integer (default 1)
#   --short-edge N       must be 768; the one tier the checkpoint provides
#   --keyframe FILE      first-frame anchor, role keyframe, frame_index 0
#   --last-frame FILE    last-frame anchor, role keyframe, frame_index 1
#   --ref-image FILE     ordered reference, role reference
#   --ref-video FILE     ordered reference, role reference_video
#   --ref-audio FILE     ordered reference, role reference_audio
#   --task T             t2va | fl2va | ref2va (default: inferred)
#   --prompt-file FILE   read the prompt from a file (up to 32000 characters)
#   --poll N             poll interval seconds (default 5)
#   --timeout N          give up after N seconds (default 7200)
#   --dry-run            print the JSON body and exit
#
# The prompt is the first positional argument; with none, stdin is read when
# it is not a terminal. Reference flags keep command-line order.
set -uo pipefail

BASE_URL=${H3_URL:-http://127.0.0.1:8090}
OUT=""
RATIO="16:9"
SECONDS_LEN=4
SEED=1
SHORT_EDGE=768
TASK=""
PROMPT=""
PROMPT_FILE=""
POLL=5
TIMEOUT=7200
DRY_RUN=0
FIRST=""
LAST=""
REF_KIND=()
REF_ROLE=()
REF_PATH=()

usage() {
  sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'
}

fail() {
  echo "h3-video: $*" >&2
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --url) BASE_URL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --ratio) RATIO=$2; shift 2 ;;
    --seconds) SECONDS_LEN=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --short-edge) SHORT_EDGE=$2; shift 2 ;;
    --keyframe) FIRST=$2; shift 2 ;;
    --last-frame) LAST=$2; shift 2 ;;
    --ref-image) REF_KIND+=(image); REF_ROLE+=(reference); REF_PATH+=("$2"); shift 2 ;;
    --ref-video) REF_KIND+=(video); REF_ROLE+=(reference_video); REF_PATH+=("$2"); shift 2 ;;
    --ref-audio) REF_KIND+=(audio); REF_ROLE+=(reference_audio); REF_PATH+=("$2"); shift 2 ;;
    --task) TASK=$2; shift 2 ;;
    --prompt-file) PROMPT_FILE=$2; shift 2 ;;
    --poll) POLL=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    -*) fail "unknown option $1 (try --help)" ;;
    *) PROMPT=$1; shift ;;
  esac
done

[ -n "$PROMPT_FILE" ] && PROMPT=$(cat "$PROMPT_FILE")
if [ -z "$PROMPT" ] && [ ! -t 0 ] && [ -z "$PROMPT_FILE" ]; then
  PROMPT=$(cat)
fi
[ -n "$PROMPT" ] || fail "no prompt: pass it as the first argument or --prompt-file"

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v base64 >/dev/null 2>&1 || fail "base64 is required"

# JSON string escaping, in bash parameter expansion. Newlines, tabs, quotes
# and backslashes are the four a prompt actually carries.
json_string() {
  local text=$1
  text=${text//\\/\\\\}
  text=${text//\"/\\\"}
  text=${text//$'\n'/\\n}
  text=${text//$'\r'/\\r}
  text=${text//$'\t'/\\t}
  printf '"%s"' "$text"
}

media_type() {
  case "${1##*.}" in
    png|PNG) echo image/png ;;
    jpg|jpeg|JPG|JPEG) echo image/jpeg ;;
    webp|WEBP) echo image/webp ;;
    mp4|MP4|mov|MOV|mkv|MKV|webm|WEBM) echo video/mp4 ;;
    wav|WAV) echo audio/wav ;;
    mp3|MP3) echo audio/mpeg ;;
    m4a|M4A|aac|AAC) echo audio/mp4 ;;
    flac|FLAC) echo audio/flac ;;
    ogg|OGG) echo audio/ogg ;;
    *) echo "" ;;
  esac
}

# A condition object: data URI only. The server never reads a path or fetches
# a URL from a request body (host memory guard, contract page).
condition() {
  local type=$1 role=$2 path=$3 index=${4:-} mime payload object
  mime=$(media_type "$path")
  payload=$(base64 -w0 <"$path")
  object=$(printf '{"type":"%s","role":"%s",' "$type" "$role")
  [ -n "$index" ] && object="$object\"frame_index\":$index,"
  printf '%s"uri":"data:%s;base64,%s"}' "$object" "$mime" "$payload"
}

# Validated here, in the main shell: a failure inside a command substitution
# would only end the subshell and ship a truncated condition list.
check_file() {
  [ -f "$1" ] || fail "no such file: $1"
  [ -n "$(media_type "$1")" ] || fail "unknown media type for $1"
}

[ -n "$FIRST" ] && check_file "$FIRST"
[ -n "$LAST" ] && check_file "$LAST"
for path in ${REF_PATH[@]+"${REF_PATH[@]}"}; do
  check_file "$path"
done

BUILT=()
[ -n "$FIRST" ] && BUILT+=("$(condition image keyframe "$FIRST" 0)")
[ -n "$LAST" ] && BUILT+=("$(condition image keyframe "$LAST" 1)")
for i in ${REF_KIND[@]+"${!REF_KIND[@]}"}; do
  BUILT+=("$(condition "${REF_KIND[$i]}" "${REF_ROLE[$i]}" "${REF_PATH[$i]}")")
done

# The task is the request's own shape: references condition, keyframes anchor,
# an empty list is text to video.
if [ -z "$TASK" ]; then
  TASK=t2va
  if [ ${#REF_KIND[@]} -gt 0 ]; then
    TASK=ref2va
  elif [ -n "$FIRST" ] || [ -n "$LAST" ]; then
    TASK=fl2va
  fi
fi

joined=$(
  IFS=,
  echo "${BUILT[*]-}"
)
BODY=$(printf '{"task":"%s","prompt":%s,"conditions":[%s],"target":{"short_edge":%s,"aspect_ratio":"%s","duration_seconds":%s},"seed":%s}' \
  "$TASK" "$(json_string "$PROMPT")" "$joined" "$SHORT_EDGE" "$RATIO" "$SECONDS_LEN" "$SEED")

if [ "$DRY_RUN" = 1 ]; then
  printf '%s\n' "$BODY"
  exit 0
fi

[ -n "$OUT" ] || OUT="h3-${SEED}-$(date +%s).mp4"

# The body goes through a file: a 16 MiB keyframe is a 21 MiB data URI, well
# past the kernel's per-argument limit (128 KiB), so it cannot be an argv.
body_file=$(mktemp "${TMPDIR:-/tmp}/h3-body.XXXXXX")
trap 'rm -f "$body_file"' EXIT
printf '%s' "$BODY" >"$body_file"

# One POST. A refusal carries {"error":{"code":..,"message":..}}; print it and
# stop rather than retrying, because the body will not become valid.
create=$(curl -sS -X POST -H 'Content-Type: application/json' \
  --data-binary @"$body_file" "$BASE_URL/v1/videos") || fail "POST $BASE_URL/v1/videos failed"
id=$(printf '%s' "$create" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
if [ -z "$id" ]; then
  echo "$create" >&2
  code=$(printf '%s' "$create" | sed -n 's/.*"code"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
  fail "create refused${code:+ ($code)}"
fi
echo "id=$id task=$TASK ratio=$RATIO seconds=$SECONDS_LEN seed=$SEED"

# Poll. The official client polls status and downloads on "completed"
# (full-2k-t2va-h3-base.sh:26-31); in_progress covers the first-step wait.
started=$SECONDS
last=""
while :; do
  record=$(curl -sS "$BASE_URL/v1/videos/$id") || fail "GET /v1/videos/$id failed"
  status=$(printf '%s' "$record" | sed -n 's/.*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
  size=$(printf '%s' "$record" | sed -n 's/.*"size"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
  frames=$(printf '%s' "$record" | sed -n 's/.*"frames"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p')

  if [ "$status" != "$last" ]; then
    echo "status=$status size=$size frames=$frames elapsed=$((SECONDS - started))s"
    last=$status
  fi

  case "$status" in
    completed) break ;;
    failed)
      echo "$record" >&2
      fail "render failed"
      ;;
    queued|in_progress) ;;
    "")
      echo "$record" >&2
      fail "no status in the record"
      ;;
    *) fail "unknown status $status" ;;
  esac

  [ $((SECONDS - started)) -lt "$TIMEOUT" ] || fail "timed out after ${TIMEOUT}s (job $id may still finish)"
  sleep "$POLL"
done

curl -sS --fail -o "$OUT" "$BASE_URL/v1/videos/$id/content" \
  || fail "GET /v1/videos/$id/content failed"
bytes=$(wc -c <"$OUT")
echo "wrote $OUT ($bytes bytes)"

if command -v ffprobe >/dev/null 2>&1; then
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,r_frame_rate,codec_name \
    -of default=nw=1 "$OUT" 2>/dev/null | sed 's/^/  /'
  ffprobe -v error -select_streams a:0 \
    -show_entries stream=codec_name,channels,sample_rate -of default=nw=1 "$OUT" 2>/dev/null |
    sed 's/^/  /'
fi
