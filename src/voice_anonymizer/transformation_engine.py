"""Real-time spectral and pitch normalization utilities."""

from __future__ import annotations

import logging

import numpy as np

from .config import AppConfig
from .model_loader import AveragedVoiceModel

logger = logging.getLogger(__name__)


def _hann_window(length: int) -> np.ndarray:
    return np.hanning(length).astype(np.float32)


def _estimate_f0(frame: np.ndarray, sample_rate: int) -> float:
    """Estimate the fundamental frequency using an autocorrelation method."""

    centered = frame - np.mean(frame)
    if np.allclose(centered, 0.0):
        return 0.0
    autocorr = np.correlate(centered, centered, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    min_lag = max(int(sample_rate / 400), 1)
    max_lag = max(int(sample_rate / 80), min_lag + 1)
    autocorr[:min_lag] = 0
    segment = autocorr[min_lag:max_lag]
    if segment.size == 0:
        return 0.0
    peak_index = int(np.argmax(segment)) + min_lag
    if autocorr[peak_index] <= 0:
        return 0.0
    return float(sample_rate / peak_index)


def _apply_pitch_shift(frame: np.ndarray, source_f0: float, target_f0: float) -> np.ndarray:
    """Apply a simplistic pitch shift via resampling."""

    if source_f0 <= 0 or target_f0 <= 0:
        return frame
    scale = float(np.clip(target_f0 / source_f0, 0.5, 2.0))
    indices = np.linspace(0.0, frame.size - 1, num=max(int(frame.size / scale), 2))
    resampled = np.interp(indices, np.arange(frame.size), frame)
    stretched = np.interp(
        np.linspace(0.0, resampled.size - 1, num=frame.size),
        np.arange(resampled.size),
        resampled,
    )
    return stretched.astype(frame.dtype, copy=False)


class TransformationEngine:
    """Transforms frames to anonymize speaker characteristics."""

    def __init__(self, config: AppConfig, model: AveragedVoiceModel) -> None:
        self._config = config
        fft_bins = config.frame_length_samples // 2 + 1
        model.validate(fft_bins)
        self._model = model
        self._window = _hann_window(config.frame_length_samples)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame of PCM audio and return the anonymized result."""

        frame_mono = frame[:, 0] if frame.ndim == 2 else frame
        padded = self._pad_frame(frame_mono)
        windowed = padded * self._window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        avg_env = self._model.spectral_envelope
        if avg_env.shape[0] != magnitude.shape[0]:
            avg_env = np.interp(
                np.linspace(0, avg_env.shape[0] - 1, num=magnitude.shape[0]),
                np.arange(avg_env.shape[0]),
                avg_env,
            )
        blended_magnitude = (1.0 - self._config.blend_coefficient) * magnitude + self._config.blend_coefficient * avg_env
        anonymized_spectrum = blended_magnitude * np.exp(1j * phase)
        reconstructed = np.fft.irfft(anonymized_spectrum, n=self._config.frame_length_samples)

        source_f0 = _estimate_f0(reconstructed, self._config.sample_rate)
        target_f0 = (1.0 - self._config.blend_coefficient) * source_f0 + self._config.blend_coefficient * self._model.average_f0
        pitch_adjusted = _apply_pitch_shift(reconstructed, source_f0, target_f0)

        normalized = pitch_adjusted / max(np.max(np.abs(pitch_adjusted)), 1e-6)
        output = normalized[: self._config.frame_hop_samples]
        return output.reshape((-1, 1))

    def _pad_frame(self, frame: np.ndarray) -> np.ndarray:
        length = self._config.frame_length_samples
        if frame.size >= length:
            segment = frame[-length:]
        else:
            pad_width = length - frame.size
            segment = np.pad(frame, (pad_width, 0), mode="constant")
        return segment.astype(np.float32, copy=False)
