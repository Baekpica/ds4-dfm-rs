#!/bin/sh
# Composite the two speech balloons onto an H3 clip.
#
# Why a composite: a prompt-only render has no anchor, so the checkpoint
# invents glyph shapes instead of the gag's dialogue (docs/minimax-h3/
# post-port-phases.md, P12.9). The balloons here are authored text: white
# antialiased ellipses with a 6 px black rim, DejaVu Sans Bold 26/23 px, at
# 12,14 and 560,8 on a 1344x576 frame, clear of both faces and of the
# crocodile's head. Audio is stream-copied; the model's track is untouched.
#
# Usage: letterballoons.sh <in.mp4> <out.mp4>
set -e

IN=$1
OUT=$2
[ -n "$IN" ] && [ -n "$OUT" ] || { echo "usage: $0 <in.mp4> <out.mp4>" >&2; exit 2; }

FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
[ -f "$FONT" ] || { echo "missing $FONT" >&2; exit 2; }

WORK=${TMPDIR:-/tmp}/h3-balloons.$$
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"

# Ellipse = clip(1 - hypot((x-cx)/rx, (y-cy)/ry)) with a one-pixel ramp, so the
# rim and the white fill share one geometry and stay concentric.
ellipse() { # W H rim out
  W=$1; H=$2; rim=$3; out=$4
  iw=$((W - 2*rim)); ih=$((H - 2*rim))
  rx=$((W / 2)); ry=$((H / 2)); irx=$((iw / 2)); iry=$((ih / 2))
  ffmpeg -v error -y -f lavfi -i "color=c=black:s=${W}x${H}:d=1" \
    -f lavfi -i "color=c=white:s=${iw}x${ih}:d=1" -filter_complex \
    "[0]format=rgba,geq=r='0':g='0':b='0':a='255*clip((1-hypot((X-${rx})/${rx},(Y-${ry})/${ry}))*${rx}/2,0,1)'[b];\
     [1]format=rgba,geq=r='255':g='255':b='255':a='255*clip((1-hypot((X-${irx})/${irx},(Y-${iry})/${iry}))*${irx}/2,0,1)'[w];\
     [b][w]overlay=${rim}:${rim}[o]" -map "[o]" -frames:v 1 "$WORK/$out-base.png"
}

# line <name> <text>: text files keep the apostrophe and the dash out of
# filtergraph escaping.
line() { printf '%s' "$2" > "$WORK/$1.txt"; }

# letter <base> <out> <size> <y1> <name1> [<y2> <name2> ...]
letter() {
  base=$1; out=$2; size=$3; shift 3
  filter=
  while [ $# -gt 0 ]; do
    filter="$filter${filter:+,}drawtext=fontfile=$FONT:textfile=$WORK/$2.txt:fontsize=$size:fontcolor=black:x=(w-text_w)/2:y=$1"
    shift 2
  done
  ffmpeg -v error -y -i "$WORK/$base-base.png" -vf "$filter" "$WORK/$out.png"
}

line b1-1 "IT ISN'T A LOG, COSTELLO!"
line b1-2 "CAN YOU SWIM?"
line b2-1 "WHY, SURE! I'M"
line b2-2 "THE HUNDRED METER"
line b2-3 "CHAMP! ER- MEET"
line b2-4 "THE NEW CHAMP!"

ellipse 500 180 6 balloon1
letter balloon1 balloon1 26 54 b1-1 96 b1-2

ellipse 380 190 6 balloon2
letter balloon2 balloon2 23 38 b2-1 70 b2-2 102 b2-3 134 b2-4

ffmpeg -v error -y -i "$IN" -i "$WORK/balloon1.png" -i "$WORK/balloon2.png" \
  -filter_complex "[0:v][1:v]overlay=12:14[a];[a][2:v]overlay=560:8[v]" \
  -map "[v]" -map 0:a -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p \
  -c:a copy "$OUT"
