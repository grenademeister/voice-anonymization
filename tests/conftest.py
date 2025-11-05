import sys
import types
from pathlib import Path


def _ensure_sounddevice_stub() -> None:
    if "sounddevice" in sys.modules:
        return

    class _StubStream:
        def __init__(self, *args, **kwargs):  # noqa: D401 - mimic sounddevice API
            raise RuntimeError(
                "sounddevice is not installed. Install dependencies via 'uv sync' before using audio IO."
            )

    stub = types.SimpleNamespace(InputStream=_StubStream, OutputStream=_StubStream, CallbackFlags=int)
    sys.modules["sounddevice"] = stub


_ensure_sounddevice_stub()

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
