#!/usr/bin/env python3
"""Rescale a Stim circuit's QUBIT_COORDS so it reads nicely in Crumble.

These circuits place *everything* on integer coordinates: data qubits on even
integers and ancilla qubits on odd integers (an overall scale factor of 2).
Crumble expects data qubits on integers and ancilla qubits on half-integers
(x.5). Dividing every QUBIT_COORDS by 2 achieves exactly that:

    data   (0,0) (2,2) (28,18)  ->  (0,0) (1,1) (14,9)      # integers
    ancilla (-1,-1) (1,1) (29,19) -> (-0.5,-0.5) ... (14.5,9.5)  # halves

Only QUBIT_COORDS lines carry spatial coordinates in these files, so every
other instruction (R, H, CX, M, DETECTOR, OBSERVABLE_INCLUDE, ...) is copied
through untouched.

Usage:
    python to_crumble.py input.txt                 # -> input_crumble.txt
    python to_crumble.py input.txt output.txt      # explicit output path
    python to_crumble.py input.txt --scale 2       # override the divisor
"""

import argparse
import re
from pathlib import Path

QUBIT_COORDS_RE = re.compile(r"^(\s*QUBIT_COORDS)\s*\(([^)]*)\)(.*)$")


def fmt(value: float) -> str:
    """Format a number without a needless '.0' (0.0 -> '0', -0.5 -> '-0.5')."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def convert_line(line: str, scale: float) -> str:
    match = QUBIT_COORDS_RE.match(line)
    if not match:
        return line
    head, coords, tail = match.groups()
    scaled = [fmt(float(c) / scale) for c in coords.split(",")]
    return f"{head}({', '.join(scaled)}){tail}"


def convert(text: str, scale: float) -> str:
    return "\n".join(convert_line(l, scale) for l in text.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Stim circuit file to convert")
    parser.add_argument("output", type=Path, nargs="?", default=None,
                        help="output path (default: <input>_crumble.txt)")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="divide all QUBIT_COORDS by this (default: 2)")
    args = parser.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_crumble{args.input.suffix}")
    text = args.input.read_text()
    result = convert(text, args.scale)
    if not result.endswith("\n"):
        result += "\n"
    output.write_text(result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
