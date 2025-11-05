"""Helpers for loading the averaged voice model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class AveragedVoiceModel:
    """Container for averaged spectral and pitch statistics."""

    spectral_envelope: np.ndarray
    average_f0: float

    def validate(self, fft_bins: int) -> None:
        """Validate the model dimensions against the processing configuration."""

        if self.spectral_envelope.ndim != 1:
            raise ValueError("Spectral envelope must be one-dimensional")
        if self.spectral_envelope.shape[0] != fft_bins:
            raise ValueError(
                f"Spectral envelope length {self.spectral_envelope.shape[0]} does not match fft bins {fft_bins}"
            )
        if not np.isfinite(self.average_f0) or self.average_f0 <= 0:
            raise ValueError("Average F0 must be a positive finite number")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        npz = np.load(path)
        return {key: npz[key] for key in npz.files}
    if path.suffix in {".json", ".jsn"}:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix in {".pkl", ".pickle"}:
        import pickle

        with path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError(f"Unsupported model format: {path.suffix}")


def load_averaged_voice_model(path: Path) -> AveragedVoiceModel:
    """Load the averaged voice model from ``path``."""

    expanded = path.expanduser().resolve()
    if not expanded.exists():
        raise FileNotFoundError(f"Averaged model not found at {expanded}")

    payload = _load_payload(expanded)
    spectral_envelope = np.asarray(payload.get("spectral_envelope"), dtype=np.float32)
    average_f0 = float(np.asarray(payload.get("average_f0"), dtype=np.float64))

    model = AveragedVoiceModel(spectral_envelope=spectral_envelope, average_f0=average_f0)
    return model
