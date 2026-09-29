"""Verify the bundled reference files without changing them."""

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main():
    root = BASE / "data/reference"
    manifest = json.loads((root / "MANIFEST.json").read_text())
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json"
    }
    if actual != set(manifest):
        raise RuntimeError("Reference file inventory differs from MANIFEST.json.")
    for name, expected in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Reference checksum mismatch: {name}")
    print(f"All {len(manifest)} reference files match SHA-256 checksums.")


if __name__ == "__main__":
    main()
