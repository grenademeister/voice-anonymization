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

    if "spectral_envelope" not in payload or "average_f0" not in payload:
        raise ValueError("Model payload must contain 'spectral_envelope' and 'average_f0' fields")

    spectral_envelope = np.asarray(payload["spectral_envelope"], dtype=np.float32)
    if spectral_envelope.ndim != 1 or spectral_envelope.size == 0:
        raise ValueError("Spectral envelope must be a non-empty one-dimensional array")

    average_f0_array = np.asarray(payload["average_f0"], dtype=np.float64)
    if average_f0_array.size == 0:
        raise ValueError("Average F0 value missing from model payload")
    average_f0 = float(average_f0_array.reshape(-1)[0])
    if not np.isfinite(average_f0) or average_f0 <= 0:
        raise ValueError("Average F0 must be a positive finite number")

    model = AveragedVoiceModel(spectral_envelope=spectral_envelope, average_f0=average_f0)
    return model
