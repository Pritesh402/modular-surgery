#!/usr/bin/env python3
"""Strip every DETECTOR line from a Stim circuit, writing <input>_noDec<ext>.

DETECTOR instructions each live on their own line in these circuits, so removal is
purely line-based: any line whose first token is ``DETECTOR`` is dropped, everything
else (QUBIT_COORDS, R, H, CX, M, TICK, OBSERVABLE_INCLUDE, REPEAT, ...) is copied
through verbatim.  This is the inverse of annotating -- it produces the "noDec"
input that annotator.py expects.

Usage:
    python detector_remover.py input.txt                 # -> input_noDec.txt
    python detector_remover.py input.txt output.txt      # explicit output path
"""

import argparse
from pathlib import Path


def is_detector_line(line: str) -> bool:
    return line.lstrip().startswith("DETECTOR")


def strip_detectors(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not is_detector_line(l))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Stim circuit file to strip")
    parser.add_argument("output", type=Path, nargs="?", default=None,
                        help="output path (default: <input>_noDec<ext>)")
    args = parser.parse_args()

    output = args.output or args.input.with_name(
        f"{args.input.stem}_noDec{args.input.suffix}")

    text = args.input.read_text()
    n_before = sum(1 for l in text.splitlines() if is_detector_line(l))
    result = strip_detectors(text)
    if not result.endswith("\n"):
        result += "\n"
    output.write_text(result)
    print(f"wrote {output}  (removed {n_before} DETECTOR line(s))")


if __name__ == "__main__":
    main()
