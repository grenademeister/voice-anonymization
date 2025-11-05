"""Helper script to bundle the application into a Windows executable."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(dist_path: Path, onefile: bool = True) -> None:
    spec_args = [
        "pyinstaller",
        "--name=voice-anonymizer",
        "--clean",
        f"--distpath={dist_path}",
        "--add-data=models/avg_model.json;models",
        "src/voice_anonymizer/main.py",
    ]
    if onefile:
        spec_args.insert(1, "--onefile")
    subprocess.run(spec_args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Windows executable with PyInstaller")
    parser.add_argument("--dist", type=Path, default=ROOT / "dist", help="Destination directory for artifacts")
    parser.add_argument("--no-onefile", action="store_true", help="Disable PyInstaller one-file mode")
    args = parser.parse_args()
    build(args.dist, onefile=not args.no_onefile)


if __name__ == "__main__":
    main()
