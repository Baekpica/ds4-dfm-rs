#!/bin/sh
# eg2_artifact_check.sh — gate 1b: the three-sided check of the artifact GGUF
# written by `eg2_gguf artifact` (the re-wrap of the DQVD container).
#
#   1. the writer's manifest (tools/eg2_gguf.c --manifest) and the ds4-rs
#      reader's own dump (crates/ds4-core/examples/gguf_inventory.rs) must agree
#      on every tensor's name, type, rank, dims, byte count and absolute offset;
#   2. every tensor's bytes in the GGUF must equal the container payload the
#      writer copied them from — the re-wrap moves bytes, it does not rewrite
#      them (phase 0 measured those payloads byte-identical to a fresh encode,
#      docs/ds4-gemma-payload-contract.md section 2);
#   3. every verbatim tensor must ALSO equal the checkpoint's own bytes
#      (safetensors, bf16), so the file is tied to the model and not only to the
#      container;
#   4. every VQ tensor must begin with the payload magic 'DQVQ' (ds41_vq_fmt.h:32).
#
#   tools/eg2_artifact_check.sh <artifact.gguf> <manifest.tsv> <container.bin> <hf-dir> <inventory.txt>
#
# The per-tensor verdict goes to stdout; diagnostics go to stderr. Exit non-zero
# on any disagreement.
set -eu
GGUF=$1; MAN=$2; CONT=$3; HF=$4; INV=$5
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
TAB=$(printf '\t')

# manifest: name type nd dims bytes gguf_abs role cont_abs st_file st_abs st_dtype
awk -F'\t' '!/^#/ { print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11 }' "$MAN" | sort > "$TMP/man.tsv"
# reader dump: name, and the fields of the T line
awk '
  /^T / {
    t = $5; sub(/^type=/, "", t); sub(/\(.*/, "", t);
    print $2"\t"t"\t"substr($3, 6)"\t"substr($4, 6)"\t"substr($7, 7)"\t"substr($9, 5);
  }' "$INV" | sort > "$TMP/inv.tsv"

n_man=$(wc -l < "$TMP/man.tsv")
n_inv=$(wc -l < "$TMP/inv.tsv")
if [ "$n_man" != "$n_inv" ]; then
    echo "the tensor counts differ: writer $n_man, reader $n_inv" >&2
    exit 1
fi
# every name must appear once on each side: a join would silently drop a
# mismatch, so the duplicate check and the join count are both required
for f in "$TMP/man.tsv" "$TMP/inv.tsv"; do
    dup=$(cut -f 1 "$f" | sort | uniq -d | head -3)
    if [ -n "$dup" ]; then
        echo "duplicate tensor names in $(basename "$f"): $dup" >&2
        exit 1
    fi
done

join -t "$TAB" -j 1 "$TMP/man.tsv" "$TMP/inv.tsv" > "$TMP/joined.tsv"
n_join=$(wc -l < "$TMP/joined.tsv")
if [ "$n_join" != "$n_man" ]; then
    cut -f 1 "$TMP/man.tsv" > "$TMP/mannames"
    cut -f 1 "$TMP/inv.tsv" > "$TMP/invnames"
    echo "$n_man tensors in the manifest but $n_join matched in the reader's dump; first differences:" >&2
    comm -3 "$TMP/mannames" "$TMP/invnames" | head -5 >&2
    exit 1
fi

# joined: 1 name 2 type 3 nd 4 dims 5 bytes 6 gguf_abs 7 role 8 cont_abs
#         9 st_file 10 st_abs 11 st_dtype 12 reader_type 13 reader_nd
#         14 reader_dims 15 reader_bytes 16 reader_abs
awk -F'\t' '
  {
    bad = 0
    if ($2  != $12) { printf "type  %s: writer %s, reader %s\n", $1, $2, $12  > "/dev/stderr"; bad = 1 }
    if ($3  != $13) { printf "rank  %s: writer %s, reader %s\n", $1, $3, $13  > "/dev/stderr"; bad = 1 }
    if ($4  != $14) { printf "dims  %s: writer %s, reader %s\n", $1, $4, $14  > "/dev/stderr"; bad = 1 }
    if ($5  != $15) { printf "bytes %s: writer %s, reader %s\n", $1, $5, $15  > "/dev/stderr"; bad = 1 }
    if ($6  != $16) { printf "abs   %s: writer %s, reader %s\n", $1, $6, $16  > "/dev/stderr"; bad = 1 }
    if (bad) { nbad++ }
    else { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $1, $6, $5, $7, $8, $9, $10 }
  }
  END { if (nbad) { printf "%d tensors disagree between the writer and the reader\n", nbad > "/dev/stderr"; exit 1 } }
' "$TMP/joined.tsv" > "$TMP/checked.tsv"

# the safetensors file every manifest row names must be the same one, or the
# checkpoint comparison below would silently read the wrong shard
sf=$(awk -F'\t' '{ print $6 }' "$TMP/checked.tsv" | sort -u)
if [ "$(printf '%s\n' "$sf" | wc -l)" != 1 ]; then
    echo "the manifest names more than one source shard: $(printf '%s ' "$sf")" >&2
    exit 1
fi
ST="$HF/$sf"

copied=0; fromckpt=0; magic=0; mismatched=0
while IFS="$TAB" read -r name gg bytes role cont st_file st_abs; do
    if cmp -s -i "$gg:$cont" -n "$bytes" "$GGUF" "$CONT"; then
        copied=$((copied + 1))
    else
        echo "BYTES DIFFER $name: the GGUF tensor at $gg does not equal the container payload at $cont ($bytes bytes)" >&2
        mismatched=$((mismatched + 1))
        continue
    fi
    case "$role" in
    verbatim)
        if cmp -s -i "$gg:$st_abs" -n "$bytes" "$GGUF" "$ST"; then
            fromckpt=$((fromckpt + 1))
        else
            echo "SOURCE DIFFERS $name: the GGUF tensor at $gg does not equal the checkpoint's bytes at $st_abs" >&2
            mismatched=$((mismatched + 1))
        fi
        ;;
    vq)
        m=$(od -An -tx1 -j "$gg" -N 4 -v "$GGUF" | tr -d ' \n')
        if [ "$m" = "44515651" ]; then
            magic=$((magic + 1))
        else
            echo "NOT A DQVQ PAYLOAD $name: the tensor at $gg begins with $m" >&2
            mismatched=$((mismatched + 1))
        fi
        ;;
    q8_0)
        ;;   # byte identity to the container is the whole claim for this one;
        #      the container's own q8_0 payload was verified against a fresh
        #      encode by `eg2_pack_cpu verify` (docs/artifact-verify-summary.txt)
    *)
        echo "unknown role '$role' for $name" >&2
        mismatched=$((mismatched + 1))
        ;;
    esac
done < "$TMP/checked.tsv"

echo "checked $copied tensors: writer and ds4-rs reader agree on name/type/rank/dims/bytes/offset;"
echo "  $copied copied byte-for-byte from the container,"
echo "  $fromckpt verbatim tensors equal the checkpoint's own bytes,"
echo "  $magic VQ tensors begin with the DQVQ payload magic"
[ "$mismatched" = 0 ] || { echo "FAILED: $mismatched tensors differ" >&2; exit 1; }
