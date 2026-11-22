#!/usr/bin/env bash
# Run one example driver by name.
#
#   pixi run example imhd_orszag_tang_2d_vtu
#   pixi run example                 # list the available examples
#
# VTU-writing examples put their frames in ./output/<name>/.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if (( $# == 0 )); then
    echo "Usage: pixi run example <name>"
    echo "Available examples:"
    for e in examples/*.mojo; do echo "  $(basename "$e" .mojo)"; done
    exit 0
fi

name="$(basename "$1" .mojo)"
shift
if [[ ! -f "examples/$name.mojo" ]]; then
    echo "error: no such example: examples/$name.mojo" >&2
    exit 1
fi
exec mojo run -I src "examples/$name.mojo" "$@"
