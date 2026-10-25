#!/usr/bin/env bash
# Stand-in renderer for wire tests: reads the argv the bridge builds, records
# it, and writes a real MP4 of the requested geometry. It lets the local v1
# contract be driven without the checkpoint (P15.5 gate, skill tests); it is
# never a quality path and never touches the model.
set -euo pipefail

out=""
w=""; h=""; s=""; seed=""; first=""; last=""
while [ $# -gt 0 ]; do
  case "$1" in
    -p) shift 2 ;;
    --width) w=$2; shift 2 ;;
    --height) h=$2; shift 2 ;;
    --seconds) s=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --first-frame) first=$2; shift 2 ;;
    --last-frame) last=$2; shift 2 ;;
    -o) out=$2; shift 2 ;;
    *) shift ;;
  esac
done

{
  printf 'argv: width=%s height=%s seconds=%s seed=%s first=%s last=%s out=%s\n' \
    "$w" "$h" "$s" "$seed" "${first:--}" "${last:--}" "$out"
  if [ -n "$first" ]; then
    printf 'first-frame: exists=%s bytes=%s header=%s\n' \
      "$([ -f "$first" ] && echo yes || echo no)" "$(wc -c <"$first")" \
      "$(head -c 8 "$first" | od -An -tx1 | tr -d ' \n')"
  fi
} >>"${H3_STANDIN_LOG:-/dev/null}"

ffmpeg -v error -y \
  -f lavfi -i "testsrc=size=${w}x${h}:rate=24:duration=${s}" \
  -f lavfi -i "anullsrc=r=32000:cl=mono" \
  -t "$s" -c:v libx264 -preset ultrafast -pix_fmt yuv420p -c:a aac -shortest \
  "$out"
