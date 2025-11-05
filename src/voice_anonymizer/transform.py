"""Lightweight audio frame transforms for anonymization."""

from __future__ import annotations

import numpy as np
import librosa

from .config import AppConfig


class FrameTransformer:
    """Applies configured anonymization transforms to audio frames."""

    def __init__(self, config: AppConfig) -> None:
        self._pitch_steps = float(config.pitch_shift_semitones)
        self._gain = float(config.amplitude_scale)
        self._noise = float(config.noise_level)
        self._rng = np.random.default_rng()
        self._sr = int(config.sample_rate)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        data = np.asarray(frame, dtype=np.float32)
        original_shape = data.shape
        if data.ndim == 1:
            data = data[:, None]

        if (
            np.isclose(self._pitch_steps, 0.0)
            and np.isclose(self._gain, 1.0)
            and self._noise <= 0.0
        ):
            return data.reshape(original_shape)

        if not np.isclose(self._pitch_steps, 0.0):
            data = self._shift_pitch(data, self._sr, self._pitch_steps)

        if not np.isclose(self._gain, 1.0):
            data = data * self._gain

        if self._noise > 0.0:
            data = data + self._rng.normal(0.0, self._noise, size=data.shape).astype(
                np.float32
            )

        return np.clip(data, -1.0, 1.0).reshape(original_shape)

    @staticmethod
    def _shift_pitch(data: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        length = data.shape[0]
        if length < 2:
            return data

        n_fft = 1 << max(1, length.bit_length() - 1)
        n_fft = min(n_fft, length)
        hop_length = max(1, n_fft // 4)

        out = np.empty_like(data)
        for c in range(data.shape[1]):
            shifted = librosa.effects.pitch_shift(
                data[:, c],
                sr=sr,
                n_steps=semitones,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            if shifted.shape[0] != length:
                shifted = librosa.util.fix_length(shifted, length)
            out[:, c] = shifted.astype(np.float32, copy=False)
        return out
