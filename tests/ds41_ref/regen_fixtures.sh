#!/usr/bin/env bash
# Regenerate every DS41 reference fixture from the oracle.
#
# The fixtures under crates/ds4-server/tests/fixtures/ds41-ref/ are the
# oracle's own output: the prompt bytes for the encode cases and the parsed
# message for the decode cases.  Regenerating is an explicit act, because a
# fixture that silently tracks the oracle does not pin anything; the oracle
# file hash is printed on every run, and ref_drive.py refuses a revision
# other than the pinned one.
#
# Usage: bash tests/ds41_ref/regen_fixtures.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out="$root/crates/ds4-server/tests/fixtures/ds41-ref"

mkdir -p "$out"

# Encode cases: the prompt our route must render byte for byte.
for case in "$here"/cases/*.json; do
    name=$(basename "$case" .json)
    mode=chat
    driver=encode
    # A case that states its own thinking mode is encoded in it.
    if grep -q '"thinking_mode": "thinking"' "$case"; then
        mode=thinking
    fi
    # A case that carries an image block needs the media driver: the text-only
    # one refuses a non-empty image record on purpose.
    if grep -q '"image_url"' "$case"; then
        driver=encode-media
    fi
    python3 "$here/ref_drive.py" "$driver" "$case" "$mode" > "$out/$name.prompt.txt"
    printf 'encode %-24s mode=%-8s %s bytes\n' "$name" "$mode" "$(wc -c < "$out/$name.prompt.txt")"
done

# Decode cases: the completion bytes are inputs, the oracle's parse of them is
# the fixture.  Both land in the fixture directory so the Rust test reads one
# self-contained place; the inputs stay authored under tests/ds41_ref/decode/
# because mk_decode_input.py rebuilds them from the oracle's own templates.
for case in "$here"/decode/*.txt; do
    name=$(basename "$case" .txt)
    mode=chat
    if [ "$name" = "dec-thinking" ]; then
        mode=thinking
    fi
    python3 "$here/ref_drive.py" parse "$case" "$mode" > "$out/$name.calls.json"
    cp "$case" "$out/$name.txt"
    printf 'decode %-24s mode=%-8s %s bytes\n' "$name" "$mode" "$(wc -c < "$out/$name.calls.json")"
done

echo "fixtures written to $out"
