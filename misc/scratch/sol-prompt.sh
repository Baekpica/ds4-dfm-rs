#!/bin/bash
# sol-prompt.sh - your prompt, three clips, one draft.
#
#   ./sol-prompt.sh "A red fox trots through fresh snow at dawn, side profile,
#                    one tail, eye level, soft warm light on the fur."
#
# What runs, in order:
#
#   1. stage 1   h3-rs (FastH3 4-step, VSA 0.9, NVFP4 prompt encoder) draws the
#                draft at 672x384, 124 frames - latent and soundtrack only, no
#                H3 video decode. This is the ONLY stage your prompt touches.
#
#   2. stage 2   three arms, all on that same draft, so the difference between
#                them is the refinement and nothing else:
#                  draft    --transfer-only     upscaler + adapter + LTX VAE decode
#                  port     --stage2-anchor 0.7 our refiner, the shipped route
#                  truth    /data/Sol-h3-Spark's own Python stage 2, release sigmas
#
#   3. measure   normalized detail, std(3x3 Laplacian) / std(luma), plus a 1:1
#                frame sheet at 2.5 s.
#
# Why the prompt drives stage 1 only: both refiners are conditioned on the
# reference's fixed prompt ("4K, refined, high quality, cinematic detail, clean
# textures, natural motion."). That is the reference's own contract - its
# runtime/prompt_cache.py refuses any other prompt for stage 2 - so a prompt
# that changed the refiner's text cue would make the comparison unfair. Stage 1
# is where subject, motion and composition come from, which is exactly the
# question "does the refiner enhance or replace".
#
# Cost on this Spark, one run: ~80 s stage 1 + ~15 s draft + ~80 s port +
# ~125 s reference, about six minutes end to end. One GPU, no clock changes.
#
# Options:
#   -s, --seed N        stage-1 seed                       (default 42)
#   -n, --steps N       stage-1 denoise steps              (default 4)
#   -a, --anchor F      port's stage-2 anchor              (default 0.7)
#   -o, --out DIR       where the run goes                 (default outputs/prompt-<stamp>)
#   -i, --image PATH    a Ref2VA reference image, repeatable, in order; the
#                       prompt addresses them as <Picture 1>, <Picture 2>, ...
#   -p, --prompt-file F read the prompt from a file instead of the command line
#       --name SLUG     name this run explicitly
#       --skip-truth    skip the reference arm (then --size/--frames are free)
#       --skip-port     skip our refiner arm
#       --skip-draft    skip the transfer-only control
#       --size WxH      stage-1 canvas                       (default 672x384)
#       --frames N      stage-1 frames                       (default 124)
#       --port N        torchrun master port                 (default 29511)
#       --dry-run       print every command, run nothing
#   -f, --force         run even if another GPU job looks active
#
# Env: H3_DIR (default $HOME/ds4-sol-h3), ARTIFACTS (default /mnt/h3-artifacts),
#      REF (default $HOME/solh3ref/Sol-h3-Spark/models/minimax_h3/Sol-H3-Spark),
#      TORCHPY (default $HOME/torchenv/bin/python)
set -u -o pipefail

H3_DIR=${H3_DIR:-$HOME/ds4-sol-h3}
A=${ARTIFACTS:-/mnt/h3-artifacts}
REF=${REF:-$HOME/solh3ref/Sol-h3-Spark/models/minimax_h3/Sol-H3-Spark}
TORCHPY=${TORCHPY:-$HOME/torchenv/bin/python}

SEED=42
STEPS=4
ANCHOR=0.7
WIDTH=672
HEIGHT=384
FRAMES=124
PORT=29511
OUT=""
NAME=""
IMAGES=()
PROMPTFILE=""
SKIP_TRUTH=0
SKIP_PORT=0
SKIP_DRAFT=0
DRY=0
FORCE=0
PROMPT=""

die() { printf 'sol-prompt: %s\n' "$*" >&2; exit 1; }

usage() { awk 'NR > 1 && /^set / { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    -s|--seed)     SEED=$2; shift 2 ;;
    -n|--steps)    STEPS=$2; shift 2 ;;
    -a|--anchor)   ANCHOR=$2; shift 2 ;;
    -o|--out)      OUT=$2; shift 2 ;;
    -i|--image)    IMAGES+=("$2"); shift 2 ;;
    -p|--prompt-file) PROMPTFILE=$2; shift 2 ;;
    --name)        NAME=$2; shift 2 ;;
    --size)        WIDTH=${2%x*}; HEIGHT=${2#*x}; shift 2 ;;
    --frames)      FRAMES=$2; shift 2 ;;
    --port)        PORT=$2; shift 2 ;;
    --skip-truth)  SKIP_TRUTH=1; shift ;;
    --skip-port)   SKIP_PORT=1; shift ;;
    --skip-draft)  SKIP_DRAFT=1; shift ;;
    --dry-run)     DRY=1; shift ;;
    -f|--force)    FORCE=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    --)            shift; PROMPT=$*; break ;;
    -*)            die "unknown option $1 (--help)" ;;
    *)             PROMPT=$1; shift ;;
  esac
done

[ -n "$PROMPTFILE" ] && {
  [ -f "$PROMPTFILE" ] || die "no such prompt file: $PROMPTFILE"
  PROMPT=$(cat "$PROMPTFILE")
}
[ -n "$PROMPT" ] || { usage; die "no prompt"; }

for img in ${IMAGES[@]+"${IMAGES[@]}"}; do
  [ -f "$img" ] || die "no such reference image: $img"
done

# The reference's wire contract is fixed: it refines a 24x37x24x42 latent whose
# audio is exactly 161333 frames (121 frames at 24 fps, 32 kHz). A different
# canvas would be silently mis-shaped, so it is refused rather than truncated.
if [ "$SKIP_TRUTH" = 0 ] && { [ "$WIDTH" != 672 ] || [ "$HEIGHT" != 384 ] || [ "$FRAMES" != 124 ]; }; then
  die "the reference arm needs 672x384 and 124 frames; use --skip-truth to change the canvas"
fi

STAMP=$(date +%Y%m%d-%H%M%S)
if [ -z "$NAME" ]; then
  NAME=$(printf '%s' "$PROMPT" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | cut -c1-28 | sed 's/-*$//')
fi
WORK=${OUT:-$H3_DIR/outputs/prompt-$STAMP-$NAME}
mkdir -p "$WORK" || die "cannot create $WORK"
LOG=$WORK/run.log

export H3_DIT_SOL=0.9
export H3_DIT_GEMM=fp8

NVFP4=$A/Comfy-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
H3MODEL=$A/MiniMax-H3-FastH3-4step
GEMMA=$A/Gemma4/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
REFINER=$A/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors
AVAE=$A/LTX-2.5/ltx-2.5-audio-vae-bf16.safetensors
VAE=$A/LTX-2.5/ltx-2.5-video-vae-conv-bf16.safetensors
LORA=$A/LTX-2.5/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
UPS=$A/H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors
ADP=$A/H3-to-LTX-Latent-Adapter
CACHE=$A/prompt-cache.pt

need() { [ -e "$1" ] || die "missing $1"; }
need "$H3_DIR/h3-rs"
need "$H3_DIR/h3-sol-rs"
need "$NVFP4"
need "$H3MODEL"
need "$CACHE"
[ "$SKIP_PORT" = 1 ] || { need "$REFINER"; need "$VAE"; need "$LORA"; need "$UPS"; need "$ADP"; }
if [ "$SKIP_TRUTH" = 0 ]; then
  need "$TORCHPY"
  need "$REF/paths-stage2.json"
  need "$HOME/solh3ref/prompt-cache.ref.pt"
fi

# One GPU job at a time: a stage-1 or stage-2 already resident would put two
# 30-40 GiB processes on one GB10 and both results would be wrong.
if [ "$FORCE" = 0 ] && [ "$DRY" = 0 ]; then
  busy=$(pgrep -a -f 'h3-rs|h3-sol-rs|runtime\.worker' | grep -v "sol-prompt" || true)
  if [ -n "$busy" ]; then
    printf 'sol-prompt: another GPU job is running:\n%s\n' "$busy" >&2
    die "wait for it, or pass --force"
  fi
fi

LAT=$WORK/stage1-latent.bin
PCM=$WORK/stage1-audio.pcm
CAP=$WORK/capture
PORT_MP4=""
DRAFT_MP4=""
TRUTH_MP4=""

guard() { # log-name, then the command
  local log=$1; shift
  python3 "$H3_DIR/tools/host_memory_guard.py" --max-gib 120 --high-gib 120 \
    --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 2400 \
    --log "$WORK/$log.memory.jsonl" -- "$@"
}

say() { printf '\n== %s\n' "$*" | tee -a "$LOG"; }

{
  printf '%s\n' "$PROMPT" > "$WORK/prompt.txt"
  printf 'prompt     %s\n' "$(printf '%s' "$PROMPT" | cut -c1-68)$( [ "${#PROMPT}" -gt 68 ] && printf ' [...]' )"
  printf 'prompt.txt %s words, full text beside the clips\n' "$(printf '%s' "$PROMPT" | wc -w)"
  printf 'seed       %s\n' "$SEED"
  printf 'steps      %s\n' "$STEPS"
  printf 'canvas     %sx%s x %s frames\n' "$WIDTH" "$HEIGHT" "$FRAMES"
  printf 'anchor     %s\n' "$ANCHOR"
  if [ "${#IMAGES[@]}" -gt 0 ]; then
    n=0
    for img in "${IMAGES[@]}"; do
      n=$((n + 1))
      printf 'image %-2s   %s (%s)\n' "$n" "$img" "$(sha256sum "$img" | cut -c1-16)"
    done
  fi
  printf 'work       %s\n' "$WORK"
  printf 'started    %s\n' "$STAMP"
} | tee "$LOG"

cd "$H3_DIR" || die "no $H3_DIR"

# ---------------------------------------------------------------- stage 1 ----
say "stage 1: h3-rs, FastH3 $STEPS steps, latent only"
S1=(./h3-rs -d "$H3MODEL" --text-nvfp4 "$NVFP4" -p "$PROMPT"
    --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" --steps "$STEPS"
    --seed "$SEED" --save-latent "$LAT" --save-audio "$PCM" --latent-only)
for img in ${IMAGES[@]+"${IMAGES[@]}"}; do
  S1+=(--ref-image "$img")
done
if [ "$DRY" = 1 ]; then printf '  %q ' "${S1[@]}"; printf '\n'; else
  t0=$(date +%s)
  guard stage1 "${S1[@]}" >>"$LOG" 2>&1 || die "stage 1 failed, see $LOG"
  t1=$(date +%s)
  printf '  stage 1 wall   %s s\n' "$((t1 - t0))" | tee -a "$LOG"
  [ -s "$LAT" ] || die "stage 1 wrote no latent"
fi

# ------------------------------------------------------- capture for truth ---
say "capture: our draft in the reference's container"
cat > "$WORK/.capture.py" <<'PY'
import hashlib, json, struct, sys
from pathlib import Path
import numpy as np, torch

latent, pcm_path, out, prompt, seed, frames = sys.argv[1:7]
seed, frames = int(seed), int(frames)
emitted = 121
shape = (1, 24, 37, 24, 42)              # the reference's H3_SHAPE
audio_shape = (1, 2, 161333)             # 121 frames at 24 fps, 32 kHz, stereo

raw = np.fromfile(latent, dtype=np.uint8)
want = int(np.prod(shape)) * 2           # bf16
if raw.size != want:
    raise SystemExit(f"latent is {raw.size} B, expected {want} for {shape}")
# bf16 -> f32 by moving each uint16 into the high half: exact for every bf16.
h3 = torch.from_numpy((raw.view(np.uint16).astype(np.uint32) << 16)
                      .view(np.float32).reshape(shape).copy()).to(torch.bfloat16)

pcm = np.fromfile(pcm_path, dtype=np.float32).reshape(-1, 2)
if pcm.shape[0] < audio_shape[2]:
    raise SystemExit(f"pcm has {pcm.shape[0]} frames, need {audio_shape[2]}")
audio = torch.from_numpy(np.ascontiguousarray(pcm[:audio_shape[2]].T.reshape(audio_shape)))

out = Path(out); out.mkdir(parents=True, exist_ok=True)
payload = out / "stage1_direct_tensors.pt"
torch.save({"h3_normalized": h3.contiguous(), "audio": audio.contiguous()}, payload)

capture = {
    "schema_version": 1, "status": "PASS", "request_id": "promptrun",
    "case_id": "promptrun", "same_request": True, "external_anchor_used": False,
    "task": "t2va", "latent_only_transfer": True, "h3_decoder_calls": 0,
    "prompt": prompt, "seed": seed, "source_index": 0,
    "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    "pcm_frames_dropped": int(pcm.shape[0] - audio_shape[2]),
    "input_variant": "h3_adapter", "primary_kind": "h3_normalized_latent",
    "generation_frames": frames, "emitted_frames": emitted,
    "pixel_height": 384, "pixel_width": 672, "fps": 24.0, "audio_sample_rate": 32000,
}
(out / "capture.json").write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
print(f"  latent {tuple(h3.shape)} {h3.dtype}  audio {tuple(audio.shape)} "
      f"(dropped {capture['pcm_frames_dropped']} padded frames)")
PY
if [ "$DRY" = 1 ]; then printf '  %q ' "$TORCHPY" "$WORK/.capture.py" "$LAT" "$PCM" "$CAP"; printf '\n'; else
  "$TORCHPY" "$WORK/.capture.py" "$LAT" "$PCM" "$CAP" "$PROMPT" "$SEED" "$FRAMES" \
    | tee -a "$LOG" || die "capture failed"
fi

# --------------------------------------------------------- port arm ----------
if [ "$SKIP_PORT" = 0 ]; then
  say "stage 2 port: our refiner, anchor $ANCHOR, tensor-core arms"
  PORT_MP4=$WORK/port-anchor$ANCHOR.mp4
  S2=(./h3-sol-rs --latent "$LAT" --audio "$PCM" --upscaler "$UPS" --adapter "$ADP"
      --refiner "$REFINER" --vae "$VAE" --context "$CACHE" --lora "$LORA"
      --lora-strength 0.8 --audio-vae "$AVAE" --sol --stage2-anchor "$ANCHOR"
      -o "$PORT_MP4")
  if [ "$DRY" = 1 ]; then printf '  %q ' "${S2[@]}"; printf '\n'; else
    t0=$(date +%s)
    H3_SOL_MMA=1 H3_LTX_ATTN_MMA=1 guard port2 \
      env H3_SOL_MMA=1 H3_LTX_ATTN_MMA=1 "${S2[@]}" >>"$LOG" 2>&1 \
      || die "port arm failed, see $LOG"
    printf '  port wall      %s s\n' "$(( $(date +%s) - t0 ))" | tee -a "$LOG"
  fi
fi

# -------------------------------------------------------- draft control ------
if [ "$SKIP_DRAFT" = 0 ]; then
  say "stage 2 control: --transfer-only, the draft with no refinement"
  DRAFT_MP4=$WORK/draft-transfer.mp4
  S3=(./h3-sol-rs --latent "$LAT" --audio "$PCM" --upscaler "$UPS" --adapter "$ADP"
      --refiner "$REFINER" --vae "$VAE" --context "$CACHE" --lora "$LORA"
      --lora-strength 0.8 --audio-vae "$AVAE" --transfer-only -o "$DRAFT_MP4")
  if [ "$DRY" = 1 ]; then printf '  %q ' "${S3[@]}"; printf '\n'; else
    t0=$(date +%s)
    guard draft2 "${S3[@]}" >>"$LOG" 2>&1 || die "draft control failed, see $LOG"
    printf '  draft wall     %s s\n' "$(( $(date +%s) - t0 ))" | tee -a "$LOG"
  fi
fi

# ------------------------------------------------------- truth arm -----------
if [ "$SKIP_TRUTH" = 0 ]; then
  say "stage 2 truth: /data/Sol-h3-Spark (Python), release sigmas"
  TRUTH_MP4=$WORK/truth-reference.mp4
  cat > "$WORK/.refstage2.py" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path

ref, capture, work, port = sys.argv[1:5]
ref = Path(ref); work = Path(work); work.mkdir(parents=True, exist_ok=True)
py = Path.home() / "torchenv" / "bin"
paths = json.loads((ref / "paths-stage2.json").read_text())

sys.path.insert(0, str(ref))
from runtime.config import load_recipe

config = load_recipe("t2va")
out_root = work / "out"
ready, options, log = work / "ready.json", work / "options.json", work / "stage2.log"
options.write_text(json.dumps({"paths": paths, "work_dir": str(work), "config": config}))

env = dict(os.environ); env.pop("PYTHONPATH", None); env["PYTHONUNBUFFERED"] = "1"
cmd = [str(py / "torchrun"), "--nproc-per-node=1", "--master-port", port,
       "-m", "runtime.worker", "--module", "stage2",
       "--options", str(options), "--ready", str(ready)]

with log.open("wb") as sink:
    proc = subprocess.Popen(cmd, cwd=str(ref), stdin=subprocess.PIPE, stdout=sink,
                            stderr=subprocess.STDOUT, env=env, text=True)

    def wait_for(path, what, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                return json.loads(path.read_text())
            if proc.poll() is not None:
                raise SystemExit(f"worker exited rc={proc.returncode} before {what}; see {log}")
            time.sleep(2)
        raise SystemExit(f"timeout waiting for {what}; see {log}")

    try:
        print("  waiting for session READY (model load takes minutes) ...", flush=True)
        state = wait_for(ready, "ready", 1800)
        print(f"  ready: {state.get('status')}", flush=True)
        if state.get("status") != "READY":
            raise SystemExit(json.dumps(state)[:2000])

        prepared = work / "prepare.json"
        proc.stdin.write(json.dumps({"op": "prepare", "response": str(prepared),
                                     "kwargs": {"request_id": "promptrun",
                                                "output_root": str(out_root)}}) + "\n")
        proc.stdin.flush()
        reply = wait_for(prepared, "prepare", 600)
        print(f"  prepare: {reply.get('status')} {reply.get('error', '')}", flush=True)
        if reply.get("status") != "PASS":
            raise SystemExit(1)

        result = work / "result.json"
        proc.stdin.write(json.dumps({"op": "run", "response": str(result),
                                     "kwargs": {"capture_dir": str(capture),
                                                "output_root": str(out_root),
                                                "request_id": "promptrun"}}) + "\n")
        proc.stdin.flush()
        started = time.monotonic()
        done = wait_for(result, "run", 5400)
        print(f"  run: {done.get('status')} in {time.monotonic() - started:.1f}s "
              f"{done.get('error', '')}", flush=True)
        if done.get("status") != "PASS":
            raise SystemExit(json.dumps(done)[:3000])
        body = done.get("result", {})
        print(f"  phases: {json.dumps(body.get('result', {}).get('phases_s', {}))}", flush=True)
    finally:
        try:
            proc.stdin.write(json.dumps({"op": "close"}) + "\n"); proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
PY
  if [ "$DRY" = 1 ]; then
    printf '  %q ' "$TORCHPY" "$WORK/.refstage2.py" "$REF" "$CAP" "$WORK/reference" "$PORT"; printf '\n'
  else
    t0=$(date +%s)
    "$TORCHPY" "$WORK/.refstage2.py" "$REF" "$CAP" "$WORK/reference" "$PORT" \
      >>"$LOG" 2>&1 || die "reference arm failed, see $LOG"
    printf '  truth wall     %s s\n' "$(( $(date +%s) - t0 ))" | tee -a "$LOG"
    found=$(find "$WORK/reference/out" -name '*.mp4' -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$found" ] || die "reference wrote no mp4, see $WORK/reference/stage2.log"
    cp -f "$found" "$TRUTH_MP4" || die "cannot copy $found"
    printf '  copied %s\n' "$found" | tee -a "$LOG"
  fi
fi

# --------------------------------------------------------- measurement ------
say "measure"
if [ "$DRY" = 1 ]; then
  printf '  norm detail table + sheet-2.5s.png\n'
else
  "$TORCHPY" - "$WORK" "$DRAFT_MP4" "$PORT_MP4" "$TRUTH_MP4" <<'PY' | tee -a "$LOG"
import subprocess, sys
from pathlib import Path
import numpy as np

work = Path(sys.argv[1])
clips = [("draft  (transfer only)", sys.argv[2]),
         ("port   (our refiner)",   sys.argv[3]),
         ("truth  (reference py)",  sys.argv[4])]
clips = [(n, p) for n, p in clips if p and Path(p).is_file()]
stamps = (0.5, 1.5, 2.5, 3.5, 4.5)

def gray(path, t):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(v) for v in out.split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path,
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w).astype(np.float32)

def norm_detail(frame):
    lap = (frame[1:-1, :-2] + frame[1:-1, 2:] + frame[:-2, 1:-1] + frame[2:, 1:-1]
           - 4.0 * frame[1:-1, 1:-1])
    return float(lap.std() / max(frame.std(), 1e-6)), float(frame.std()), float(lap.std())

print(f"{'clip':24s} {'luma_std':>9s} {'hp_std':>8s} {'norm detail':>12s}")
rows = []
for name, path in clips:
    vals = [norm_detail(gray(path, t)) for t in stamps]
    nd = sum(v[0] for v in vals) / len(vals)
    ls = sum(v[1] for v in vals) / len(vals)
    hs = sum(v[2] for v in vals) / len(vals)
    rows.append((name, ls, hs, nd))
    print(f"{name:24s} {ls:9.2f} {hs:8.3f} {nd:12.4f}")
if rows:
    base = rows[0][3]
    for name, _, _, nd in rows[1:]:
        print(f"  {name:22s} detail x{nd / base:.3f} vs the draft")
PY
  if [ "${#WORK}" -gt 0 ]; then
    ffmpeg -v error -y \
      -ss 2.5 -i "$WORK/draft-transfer.mp4" -frames:v 1 "$WORK/.a.png" 2>/dev/null
    ffmpeg -v error -y \
      -ss 2.5 -i "$WORK/port-anchor$ANCHOR.mp4" -frames:v 1 "$WORK/.b.png" 2>/dev/null
    ffmpeg -v error -y \
      -ss 2.5 -i "$WORK/truth-reference.mp4" -frames:v 1 "$WORK/.c.png" 2>/dev/null
    if [ -f "$WORK/.a.png" ] && [ -f "$WORK/.b.png" ] && [ -f "$WORK/.c.png" ]; then
      ffmpeg -v error -y -i "$WORK/.a.png" -i "$WORK/.b.png" -i "$WORK/.c.png" \
        -filter_complex hstack=inputs=3 "$WORK/sheet-2.5s.png" || true
      printf '  sheet: %s (draft | port | truth, 2.5 s)\n' "$WORK/sheet-2.5s.png" | tee -a "$LOG"
    fi
    rm -f "$WORK/.a.png" "$WORK/.b.png" "$WORK/.c.png"
  fi
fi

say "done"
{
  ls -l "$WORK"/*.mp4 2>/dev/null
  printf '\nprompt: %s\n' "$PROMPT"
  printf 'work:   %s\n' "$WORK"
} | tee -a "$LOG"
