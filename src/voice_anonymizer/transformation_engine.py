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


def _apply_pitch_shift(
    frame: np.ndarray, source_f0: float, target_f0: float
) -> np.ndarray:
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
        self._avg_envelope = model.spectral_envelope.astype(np.float32, copy=False)
        self._window = _hann_window(config.frame_length_samples)
        history_length = max(config.frame_length_samples - config.frame_hop_samples, 0)
        self._history = np.zeros(history_length, dtype=np.float32)
        self._frame_count = 0

        logger.info(
            "TransformationEngine initialized: blend=%.2f, avg_f0=%.1f Hz, envelope_shape=%s",
            config.blend_coefficient,
            model.average_f0,
            model.spectral_envelope.shape,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame of PCM audio and return the anonymized result."""

        self._frame_count += 1
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
            return self._expand_channels(mono)

        windowed = analysis_frame * self._window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Blend the magnitude spectrum with the averaged model
        blended_magnitude = (
            1.0 - self._config.blend_coefficient
        ) * magnitude + self._config.blend_coefficient * self._avg_envelope

        anonymized_spectrum = blended_magnitude * np.exp(1j * phase)
        reconstructed = np.fft.irfft(
            anonymized_spectrum, n=self._config.frame_length_samples
        )

        source_f0 = _estimate_f0(analysis_frame, self._config.sample_rate)
        target_f0 = (
            1.0 - self._config.blend_coefficient
        ) * source_f0 + self._config.blend_coefficient * self._model.average_f0

        # Pitch shifting often introduces artifacts, so only apply if difference is significant
        # and blend coefficient is high enough to warrant it
        if self._config.blend_coefficient > 0.6 and abs(target_f0 - source_f0) > 20:
            pitch_adjusted = _apply_pitch_shift(
                reconstructed.astype(np.float32, copy=False), source_f0, target_f0
            )
        else:
            # Skip pitch shifting to preserve audio quality
            pitch_adjusted = reconstructed.astype(np.float32, copy=False)

        # Normalize output to match input RMS (energy preservation)
        input_rms = float(np.sqrt(np.mean(mono**2)))
        if input_rms > 1e-6:
            output_section = pitch_adjusted[-mono.size :]
            output_rms = float(np.sqrt(np.mean(output_section**2)))
            if output_rms > 1e-6:
                pitch_adjusted = pitch_adjusted * (input_rms / output_rms)

        if self._frame_count % 100 == 0:
            input_rms_log = float(np.sqrt(np.mean(mono**2)))
            output_rms_log = float(np.sqrt(np.mean(pitch_adjusted[-mono.size :] ** 2)))
            logger.debug(
                "Transform frame %d: input_rms=%.4f, output_rms=%.4f, source_f0=%.1f Hz, target_f0=%.1f Hz",
                self._frame_count,
                input_rms_log,
                output_rms_log,
                source_f0,
                target_f0,
            )

        output = pitch_adjusted[-mono.size :].astype(np.float32, copy=False)
        return self._expand_channels(output)

    def _prepare_analysis_frame(self, chunk: np.ndarray) -> np.ndarray:
        hop = self._config.frame_hop_samples
        frame_length = self._config.frame_length_samples
        if chunk.size != hop:
            raise ValueError(f"Expected chunk of {hop} samples, received {chunk.size}")
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
