"""Real-time voice anonymization package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AppConfig",
    "load_averaged_voice_model",
    "TransformationEngine",
    "AudioCapture",
    "AudioOutput",
    "run",
]


def __getattr__(name: str) -> Any:
    if name == "AppConfig":
        from .config import AppConfig as _AppConfig

        return _AppConfig
    if name == "load_averaged_voice_model":
        from .model_loader import load_averaged_voice_model as _load

        return _load
    if name == "TransformationEngine":
        from .transformation_engine import TransformationEngine as _Engine

        return _Engine
    if name == "AudioCapture":
        from .audio_capture import AudioCapture as _Capture

        return _Capture
    if name == "AudioOutput":
        from .audio_output import AudioOutput as _Output

        return _Output
    if name == "run":
        from .main import run as _run

        return _run
    raise AttributeError(f"module 'voice_anonymizer' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
