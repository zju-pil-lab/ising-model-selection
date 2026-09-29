"""Create a GitHub-ready source/data ZIP with no machine-specific artifacts."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

BASE = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(BASE / "tools/verify_reference.py")], check=True)
    fixed = [
        "LICENSE",
        "CITATION.cff",
        "CITATION.bib",
        ".gitignore",
        ".gitattributes",
        ".clang-format",
        "pyproject.toml",
    ]
    files = [BASE / name for name in fixed]
    for pattern in ["*.py", "*.cpp", "*.md", "requirements*.txt"]:
        files.extend(BASE.glob(pattern))
    for directory in ["docs", "data/reference", ".github", "tools"]:
        files.extend(
            p
            for p in (BASE / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    files = sorted(set(files))
    manifest = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out = BASE / "dist"
    out.mkdir(exist_ok=True)
    name = "ising-model-selection-v1.0.0"
    archive = out / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, name + "/" + str(p.relative_to(BASE)))
        z.writestr(name + "/RELEASE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed.")
        for path, digest in manifest.items():
            if hashlib.sha256(z.read(name + "/" + path)).hexdigest() != digest:
                raise RuntimeError(f"ZIP member checksum mismatch: {path}")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / f"{name}.zip.sha256").write_text(f"{checksum}  {archive.name}\n")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "files": len(files) + 1,
                "sha256": checksum,
                "verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
