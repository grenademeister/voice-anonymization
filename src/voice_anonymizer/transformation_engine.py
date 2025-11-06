"""Real-time pitch shifting and loudness normalization utilities."""

from __future__ import annotations

import numpy as np

from .config import AppConfig
from .model_loader import AveragedVoiceModel


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
    """Apply a pitch shift by warping the frequency spectrum."""

    if source_f0 <= 0 or target_f0 <= 0:
        return frame

    scale = float(np.clip(target_f0 / source_f0, 0.5, 2.0))
    if np.isclose(scale, 1.0, atol=1e-3):
        return frame

    spectrum = np.fft.rfft(frame.astype(np.float32, copy=False))
    bins = spectrum.size
    if bins == 0:
        return frame

    target_bins = np.arange(bins, dtype=np.float32)
    source_bins = target_bins / scale
    valid = source_bins < bins - 1
    lower = np.floor(source_bins[valid]).astype(int)
    upper = np.clip(lower + 1, 0, bins - 1)
    weight = source_bins[valid] - lower.astype(np.float32)

    shifted = np.zeros_like(spectrum)
    shifted[valid] = (1.0 - weight) * spectrum[lower] + weight * spectrum[upper]

    return np.fft.irfft(shifted, n=frame.size).astype(frame.dtype, copy=False)


class TransformationEngine:
    """Transforms frames to anonymize speaker characteristics."""

    def __init__(self, config: AppConfig, model: AveragedVoiceModel) -> None:
        self._config = config
        fft_bins = config.frame_length_samples // 2 + 1
        model.validate(fft_bins)
        self._model = model
        envelope = model.spectral_envelope.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(envelope), dtype=np.float64)))
        if not np.isfinite(rms) or rms <= 1e-6:
            rms = 1.0
        self._target_rms = rms
        history_length = max(config.frame_length_samples - config.frame_hop_samples, 0)
        self._history = np.zeros(history_length, dtype=np.float32)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame of PCM audio and return the anonymized result."""

        mono = frame[:, 0] if frame.ndim == 2 else frame
        hop = self._config.frame_hop_samples
        mono = np.asarray(mono, dtype=np.float32)
        if mono.size == 0:
            return np.zeros((0, self._config.channels), dtype=np.float32)
        if mono.size > hop:
            mono = mono[-hop:]
        elif mono.size < hop:
            mono = np.pad(mono, (hop - mono.size, 0), mode="constant")

        analysis_frame = self._prepare_analysis_frame(mono)

        if self._config.blend_coefficient <= 1e-6:
            return self._expand_channels(mono.astype(np.float32, copy=False))

        source_f0 = _estimate_f0(analysis_frame, self._config.sample_rate)
        target_f0 = (1.0 - self._config.blend_coefficient) * source_f0 + self._config.blend_coefficient * self._model.average_f0
        pitch_adjusted = _apply_pitch_shift(analysis_frame.astype(np.float32, copy=False), source_f0, target_f0)
        normalized = self._normalize_amplitude(pitch_adjusted, analysis_frame)

        output = normalized[-mono.size :].astype(np.float32, copy=False)
        return self._expand_channels(output)

    def _prepare_analysis_frame(self, chunk: np.ndarray) -> np.ndarray:
        hop = self._config.frame_hop_samples
        frame_length = self._config.frame_length_samples
        if chunk.size != hop:
            raise ValueError(
                f"Expected chunk of {hop} samples, received {chunk.size}"
            )
        if self._history.size == 0:
            frame = chunk
        else:
            frame = np.concatenate([self._history, chunk])
        if frame.size < frame_length:
            frame = np.pad(frame, (frame_length - frame.size, 0), mode="constant")
        elif frame.size > frame_length:
            frame = frame[-frame_length:]
        if self._history.size:
            self._history = frame[-self._history.size :].astype(np.float32, copy=False)
        return frame.astype(np.float32, copy=False)

    def _expand_channels(self, data: np.ndarray) -> np.ndarray:
        channels = self._config.channels
        column = data.reshape((-1, 1))
        if channels == 1:
            return column
        return np.repeat(column, channels, axis=1)

    def _normalize_amplitude(self, transformed: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match output loudness to a normalized target while avoiding clipping."""

        ref_rms = float(np.sqrt(np.mean(np.square(reference), dtype=np.float64)))
        target_rms = (1.0 - self._config.blend_coefficient) * ref_rms + self._config.blend_coefficient * self._target_rms
        transformed_rms = float(np.sqrt(np.mean(np.square(transformed), dtype=np.float64)))

        if transformed_rms <= 1e-6 or target_rms <= 0.0:
            return np.zeros_like(transformed, dtype=np.float32)

        gain = target_rms / transformed_rms
        normalized = transformed * gain
        normalized -= float(np.mean(normalized, dtype=np.float64))
        max_abs = float(np.max(np.abs(normalized)))
        if max_abs > 1.0:
            normalized = normalized / max_abs
        return normalized.astype(np.float32, copy=False)
