"""Module entry-point for `python -m voice_anonymizer` and PyInstaller."""

from __future__ import annotations

from .main import run


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
