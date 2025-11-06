"""WORLD-based formant and pitch shifting for anonymization."""

from __future__ import annotations

from typing import Final

import numpy as np
import pyworld as pw

from .config import AppConfig


class FrameTransformer:
    """Applies WORLD-based speech modification to audio frames."""

    _EPS: Final[float] = 1e-8

    def __init__(self, config: AppConfig) -> None:
        self._sr = int(config.sample_rate)
        self._formant_ratio = float(config.formant_shift_ratio)
        self._pitch_steps = float(config.pitch_shift_semitones)
        self._spectral_exp = float(config.spectral_envelope_exponent)
        self._frame_period_ms = (
            min(5.0, float(config.frame_hop_ms)) if config.frame_hop_ms > 0 else 5.0
        )
        self._hop_samples = max(1, config.frame_hop_samples)
        self._channels = max(1, int(config.channels))
        window_ms = max(
            float(config.world_analysis_window_ms),
            float(config.frame_hop_ms) if config.frame_hop_ms > 0 else 10.0,
        )
        hops_for_window = max(
            4,
            int(np.ceil(window_ms / max(1e-6, float(config.frame_hop_ms))))
            if config.frame_hop_ms > 0
            else int(np.ceil((window_ms / 1_000.0) * self._sr / self._hop_samples)),
        )
        self._analysis_samples = hops_for_window * self._hop_samples
        self._buffers: list[np.ndarray] = [
            np.zeros(0, dtype=np.float32) for _ in range(self._channels)
        ]
        self._wet_mix = float(np.clip(config.world_wet_mix, 0.0, 1.0))
        self._needs_processing = not (
            np.isclose(self._formant_ratio, 1.0)
            and np.isclose(self._pitch_steps, 0.0)
            and np.isclose(self._spectral_exp, 1.0)
        )

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        data = np.asarray(frame, dtype=np.float32)
        original_shape = data.shape
        if data.ndim == 1:
            data = data[:, None]

        if data.size == 0 or not self._needs_processing:
            return data.reshape(original_shape)

        channels = data.shape[1]
        self._ensure_buffer_count(channels)

        out = np.empty_like(data)
        for idx in range(channels):
            out[:, idx] = self._process_stream(data[:, idx], idx)

        return np.clip(out, -1.0, 1.0).reshape(original_shape)

    def _ensure_buffer_count(self, count: int) -> None:
        if len(self._buffers) < count:
            for _ in range(count - len(self._buffers)):
                self._buffers.append(np.zeros(0, dtype=np.float32))

    def _process_stream(self, samples: np.ndarray, index: int) -> np.ndarray:
        if samples.size == 0:
            return samples

        buffer = np.concatenate((self._buffers[index], samples))
        output = np.empty_like(samples)
        produced = 0

        while produced < samples.size:
            if buffer.size < self._analysis_samples:
                output[produced:] = samples[produced:]
                break

            chunk = buffer[: self._analysis_samples]
            processed_chunk = self._process_channel(chunk)

            take = min(self._hop_samples, samples.size - produced)
            wet = processed_chunk[:take]
            if self._wet_mix >= 1.0:
                mixed = wet
            elif self._wet_mix <= 0.0:
                mixed = chunk[:take]
            else:
                mixed = (
                    self._wet_mix * wet
                    + (1.0 - self._wet_mix) * chunk[:take]
                )
            output[produced : produced + take] = mixed
            produced += take
            buffer = buffer[self._hop_samples :]

        self._buffers[index] = buffer
        return output

    def _process_channel(self, channel: np.ndarray) -> np.ndarray:
        signal = np.asarray(channel, dtype=np.float64, order="C")
        if signal.size < 4:
            return channel

        frame_period = max(1.0, self._frame_period_ms)
        try:
            f0, time_axis = pw.harvest(signal, self._sr, frame_period=frame_period)
            f0 = pw.stonemask(signal, f0, time_axis, self._sr)
            spectrogram = pw.cheaptrick(signal, f0, time_axis, self._sr)
            aperiodicity = pw.d4c(signal, f0, time_axis, self._sr)
            spectrogram = np.nan_to_num(
                spectrogram, nan=self._EPS, posinf=1.0, neginf=self._EPS
            )
            aperiodicity = np.nan_to_num(aperiodicity, nan=0.0, posinf=0.0, neginf=0.0)
            spectrogram = np.maximum(spectrogram, self._EPS)
            aperiodicity = np.clip(aperiodicity, 0.0, 1.0)
        except Exception:
            return channel

        if not np.isclose(self._formant_ratio, 1.0):
            spectrogram = self._warp_spectral_envelope(spectrogram, self._formant_ratio)
        if not np.isclose(self._spectral_exp, 1.0):
            spectrogram = np.power(
                np.clip(spectrogram, self._EPS, None), self._spectral_exp
            )
        if not np.isclose(self._pitch_steps, 0.0):
            f0 = self._shift_f0(f0, self._pitch_steps)

        try:
            synthesized = pw.synthesize(f0, spectrogram, aperiodicity, self._sr, frame_period=frame_period)
        except Exception:
            return channel

        if synthesized.shape[0] != signal.shape[0]:
            synthesized = self._match_length(synthesized, signal.shape[0])

        synthesized = self._match_rms(signal, synthesized)
        return synthesized.astype(np.float32, copy=False)

    @staticmethod
    def _shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
        factor = 2.0 ** (semitones / 12.0)
        return f0 * factor

    @staticmethod
    def _warp_spectral_envelope(
        spectrogram: np.ndarray, ratio: float
    ) -> np.ndarray:
        if ratio <= 0:
            return spectrogram
        warped = np.empty_like(spectrogram)
        bins = spectrogram.shape[1]
        src_bins = np.arange(bins)
        dst = np.linspace(0.0, 1.0, bins)
        remapped = np.clip(dst / ratio, 0.0, 1.0) * (bins - 1)
        for frame_idx in range(spectrogram.shape[0]):
            warped[frame_idx] = np.interp(
                remapped,
                src_bins,
                spectrogram[frame_idx],
                left=spectrogram[frame_idx, 0],
                right=spectrogram[frame_idx, -1],
            )
        return warped

    @staticmethod
    def _match_length(signal: np.ndarray, length: int) -> np.ndarray:
        if length <= 0 or signal.size == 0:
            return np.zeros((length,), dtype=signal.dtype)
        if signal.shape[0] == length:
            return signal
        x_old = np.linspace(0.0, 1.0, signal.shape[0])
        x_new = np.linspace(0.0, 1.0, length)
        return np.interp(x_new, x_old, signal)

    @classmethod
    def _match_rms(cls, reference: np.ndarray, signal: np.ndarray) -> np.ndarray:
        ref_rms = np.sqrt(np.mean(reference**2))
        sig_rms = np.sqrt(np.mean(signal**2))
        if sig_rms < cls._EPS or not np.isfinite(sig_rms) or ref_rms <= cls._EPS:
            return signal
        gain = ref_rms / sig_rms
        return signal * gain
