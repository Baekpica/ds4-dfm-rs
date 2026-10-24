#!/bin/sh
# Reference-media gate for MiniMax-H3 (P12).
#
# Runs the reference engine's own conditioning spellings
# (`/data/h3.c/main.c:47-56`, option table `:333-338`) against one binary and
# records what came out, so the same manifest can be produced for the port and
# for the reference and compared.
#
# Two families, and they need different artifacts:
#
#   FL2VA  --first-frame / --last-frame     works with the FL2VA checkpoint
#   Ref2VA --ref-image / --ref-video / --ref-silent-video / --ref-video-audio
#          --ref-audio                      needs Ref2VA/ in the model root
#                                           (`h3.c:872-874`: "ordered references
#                                           require the Ref2VA checkpoint")
#
# Rules the cases obey, read from the source:
#   - references cannot be combined with frame anchors (`h3.c:594-597`);
#   - at most 12 references, 9 images, 3 videos, 3 audio inputs
#     (`h3.c:589`, `h3.c:633-636`);
#   - audio alone is refused, it needs an image or video reference
#     (`h3.c:638-640`);
#   - every input is a quality render: 20 evaluations (`--steps 20`); the
#     operator reviews the picture, so nothing here uses a short-circuit.
#
# Usage:
#   ref_gate.sh --binary BIN --model DIR --out DIR --clip MP4 --image PNG
#               [--case NAME] [--compare OTHER_DIR]
#
# The fixtures are derived from --clip and --image so no large binaries are
# stored in the tree. Each reference clip must last 2-15 s (model card,
# `README.md:75`), so the fixture is looped to 2.5 s.
set -eu

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

BINARY=""
MODEL=""
OUT=""
CLIP=""
IMAGE=""
ONLY=""
COMPARE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --binary) BINARY=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --clip) CLIP=$2; shift 2 ;;
        --image) IMAGE=$2; shift 2 ;;
        --case) ONLY=$2; shift 2 ;;
        --compare) COMPARE=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "ref_gate: unknown option $1" >&2; usage ;;
    esac
done

[ -n "$BINARY" ] && [ -n "$MODEL" ] && [ -n "$OUT" ] && [ -n "$CLIP" ] && [ -n "$IMAGE" ] || usage
[ -x "$BINARY" ] || { echo "ref_gate: $BINARY is not executable" >&2; exit 2; }
[ -d "$MODEL" ] || { echo "ref_gate: no model root at $MODEL" >&2; exit 2; }
[ -f "$CLIP" ] || { echo "ref_gate: no clip at $CLIP" >&2; exit 2; }
[ -f "$IMAGE" ] || { echo "ref_gate: no image at $IMAGE" >&2; exit 2; }

mkdir -p "$OUT"

PROMPT_IMAGE="Animate the scene in the reference."
PROMPT_VIDEO="Continue the motion in this clip."
PROMPT_AUDIO="Use the animal and music from the references."

# Two fixtures from one clip: with its soundtrack, without it, and the
# soundtrack alone at the reference's own 32 kHz stereo.
REF_CLIP="$OUT/fixtures/ref-clip.mp4"
REF_SILENT="$OUT/fixtures/ref-silent.mp4"
REF_AUDIO="$OUT/fixtures/ref-audio.wav"
LAST_FRAME="$OUT/fixtures/last-frame.png"

if [ ! -f "$REF_CLIP" ]; then
    mkdir -p "$OUT/fixtures"
    ffmpeg -v error -stream_loop -1 -i "$CLIP" -t 2.5 -r 24 \
        -vf "scale=512:512:flags=bilinear" -pix_fmt yuv420p \
        -c:v libx264 -preset medium -crf 18 -c:a aac -b:a 192k \
        -y "$REF_CLIP"
    ffmpeg -v error -i "$REF_CLIP" -an -c copy -y "$REF_SILENT"
    ffmpeg -v error -i "$REF_CLIP" -vn -ar 32000 -ac 2 -c:a pcm_f32le \
        -y "$REF_AUDIO"
    ffmpeg -v error -i "$IMAGE" -vf hflip -y "$LAST_FRAME"
fi

HAVE_REF2VA=0
[ -d "$MODEL/Ref2VA/transformer" ] && HAVE_REF2VA=1

MANIFEST="$OUT/manifest.txt"
: > "$MANIFEST"

# name|frames|prompt|args...|family
case_selected() {
    [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]
}

run_case() {
    name=$1
    frames=$2
    prompt=$3
    family=$4
    shift 4

    case_selected "$name" || return 0

    if [ "$family" = "ref2va" ] && [ "$HAVE_REF2VA" -eq 0 ]; then
        echo "SKIP $name (no $MODEL/Ref2VA/transformer; h3.c:872-874)" | tee -a "$MANIFEST"
        return 0
    fi

    output="$OUT/$name.mp4"
    log="$OUT/$name.log"
    started=$(date +%s)
    if ! "$BINARY" -d "$MODEL" -p "$prompt" \
            --width 512 --height 512 --frames "$frames" \
            --steps 20 --seed 42 --reuse 1 \
            -o "$output" "$@" > "$log" 2>&1; then
        echo "FAIL $name exit=$? log=$log" | tee -a "$MANIFEST"
        return 1
    fi
    elapsed=$(( $(date +%s) - started ))

    if [ ! -s "$output" ]; then
        echo "FAIL $name wrote no output" | tee -a "$MANIFEST"
        return 1
    fi

    # Bytes for the container, framemd5 for the picture and md5 for the
    # samples: the first catches everything, the other two survive a muxer
    # difference.
    frames_md5=$(ffmpeg -v error -i "$output" -map 0:v:0 -f framemd5 - | md5sum | cut -d' ' -f1)
    audio_md5=$(ffmpeg -v error -i "$output" -map 0:a:0 -f md5 - | cut -d' ' -f1)
    bytes=$(stat -c %s "$output")
    sha=$(sha256sum "$output" | cut -d' ' -f1)
    geometry=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,nb_frames -of csv=p=0 "$output")

    echo "PASS $name frames=$frames geometry=$geometry bytes=$bytes seconds=$elapsed" | tee -a "$MANIFEST"
    echo "  mp4=$sha frames_md5=$frames_md5 audio_md5=$audio_md5" | tee -a "$MANIFEST"
    return 0
}

status=0
run_case fl2va-first 22 "$PROMPT_IMAGE" fl2va --first-frame "$IMAGE" || status=1
run_case fl2va-first-last 22 "$PROMPT_IMAGE" fl2va --first-frame "$IMAGE" --last-frame "$LAST_FRAME" || status=1
run_case ref2va-image 22 "$PROMPT_IMAGE" ref2va --ref-image "$IMAGE" || status=1
run_case ref2va-video 22 "$PROMPT_VIDEO" ref2va --ref-video "$REF_CLIP" || status=1
run_case ref2va-silent 22 "$PROMPT_VIDEO" ref2va --ref-silent-video "$REF_CLIP" || status=1
run_case ref2va-video-audio 56 "$PROMPT_VIDEO" ref2va --ref-video-audio "$REF_SILENT" "$REF_AUDIO" || status=1
run_case ref2va-image-audio 56 "$PROMPT_AUDIO" ref2va --ref-image "$IMAGE" --ref-audio "$REF_AUDIO" || status=1
run_case ref2va-mixed 56 "$PROMPT_AUDIO" ref2va --ref-image "$IMAGE" --ref-silent-video "$REF_CLIP" --ref-audio "$REF_AUDIO" || status=1

if [ -n "$COMPARE" ]; then
    other="$COMPARE/manifest.txt"
    [ -f "$other" ] || { echo "ref_gate: no manifest at $other" >&2; exit 2; }
    echo "COMPARE $other"
    if diff -u "$other" "$MANIFEST" > "$OUT/compare.diff"; then
        echo "MANIFEST identical"
    else
        echo "MANIFEST differs; see $OUT/compare.diff"
        status=1
    fi
fi

exit $status
