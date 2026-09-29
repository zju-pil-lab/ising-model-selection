"""Keep generated calculations outside the immutable reference snapshot."""

from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "data/reference"


def output_directory(path, parser):
    result = Path(path).resolve()
    if result == REFERENCE or REFERENCE in result.parents:
        parser.error("data/reference is immutable; choose an output directory outside it.")
    return result
