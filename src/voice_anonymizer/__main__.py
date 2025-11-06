"""Module entry-point for `python -m voice_anonymizer` and PyInstaller."""

from __future__ import annotations

try:  # Prefer relative import when executed as a package module.
    from .main import run
except ImportError:  # Fallback for frozen/standalone environments.
    from voice_anonymizer.main import run  # type: ignore[no-redef]


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
