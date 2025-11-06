"""WORLD-based audio anonymization transforms."""

from __future__ import annotations

from typing import List

import numpy as np
import pyworld as pw

from .config import AppConfig


class FrameTransformer:
    """Applies WORLD vocoder based anonymization to incoming audio frames."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._channel_processors: list[_WorldChannelProcessor] = []

    def reset(self) -> None:
        """Clear internal buffers."""

        for processor in self._channel_processors:
            processor.reset()

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        data = np.asarray(frame, dtype=np.float32)
        original_shape = data.shape
        if data.ndim == 1:
            data = data[:, None]

        # Initialize processors for each channel lazily to match runtime topology.
        if not self._channel_processors:
            self._channel_processors = [
                _WorldChannelProcessor(self._config) for _ in range(data.shape[1])
            ]

        if len(self._channel_processors) != data.shape[1]:
            raise ValueError(
                f"Expected {len(self._channel_processors)} channels but received {data.shape[1]}"
            )

        processed: List[np.ndarray] = []
        for channel, processor in enumerate(self._channel_processors):
            processed_channel = processor.process(data[:, channel])
            processed.append(processed_channel[:, None])

        result = np.concatenate(processed, axis=1)
        return result.reshape(original_shape)


class _WorldChannelProcessor:
    """Streaming WORLD vocoder processor for a single channel."""

    _EPSILON = 1e-8

    def __init__(self, config: AppConfig) -> None:
        self._sr = int(config.sample_rate)
        self._frame_period = float(config.frame_hop_ms)
        self._block = max(config.world_block_samples, config.frame_hop_samples)
        self._pitch_ratio = float(config.pitch_shift_ratio)
        self._formant_ratio = float(config.formant_shift_ratio)
        self._gain = float(config.amplitude_scale)
        self._noise = float(config.noise_level)
        self._min_f0 = float(config.min_f0)
        self._max_f0 = float(config.max_f0)
        self._rng = np.random.default_rng()
        self._input_buffer = np.zeros(0, dtype=np.float32)
        self._output_buffer = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._input_buffer = np.zeros(0, dtype=np.float32)
        self._output_buffer = np.zeros(0, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        self._input_buffer = np.concatenate((self._input_buffer, samples))

        produced: list[np.ndarray] = []
        while self._input_buffer.size >= self._block:
            chunk = self._input_buffer[: self._block]
            self._input_buffer = self._input_buffer[self._block :]
            produced.append(self._transform_block(chunk))

        if produced:
            combined = np.concatenate(produced)
            if self._output_buffer.size:
                self._output_buffer = np.concatenate((self._output_buffer, combined))
            else:
                self._output_buffer = combined

        if self._output_buffer.size >= samples.size:
            out = self._output_buffer[: samples.size]
            self._output_buffer = self._output_buffer[samples.size :]
        else:
            out = np.zeros_like(samples)
            if self._output_buffer.size:
                out[: self._output_buffer.size] = self._output_buffer
                self._output_buffer = np.zeros(0, dtype=np.float32)
        return out

    def _transform_block(self, chunk: np.ndarray) -> np.ndarray:
        if not np.any(chunk):
            return chunk.copy()

        signal = chunk.astype(np.float64, copy=False)
        f0, time_axis = pw.dio(
            signal, fs=self._sr, frame_period=self._frame_period, f0_floor=self._min_f0
        )
        f0 = pw.stonemask(signal, f0, time_axis, self._sr)

        # Apply pitch scaling only to voiced frames.
        voiced = f0 > 0.0
        if np.any(voiced):
            scaled = f0[voiced] * self._pitch_ratio
            scaled = np.clip(scaled, self._min_f0, self._max_f0)
            f0[voiced] = scaled

        sp = pw.cheaptrick(signal, f0, time_axis, self._sr)
        if not np.isclose(self._formant_ratio, 1.0):
            sp = self._shift_formants(sp, self._formant_ratio)

        ap = pw.d4c(signal, f0, time_axis, self._sr)
        synthesized = pw.synthesize(
            f0, sp, ap, self._sr, frame_period=self._frame_period
        )

        output = synthesized.astype(np.float32, copy=False)
        if output.size > chunk.size:
            output = output[: chunk.size]
        elif output.size < chunk.size:
            output = np.pad(output, (0, chunk.size - output.size))

        if not np.isclose(self._gain, 1.0):
            output *= self._gain

        if self._noise > 0.0:
            noise = self._rng.normal(0.0, self._noise, size=output.shape).astype(
                np.float32
            )
            output += noise

        return np.clip(output, -1.0, 1.0)

    def _shift_formants(self, spectrum: np.ndarray, ratio: float) -> np.ndarray:
        freq_bins = spectrum.shape[1]
        freq_axis = np.linspace(0.0, self._sr / 2.0, freq_bins)
        source_axis = np.minimum(freq_axis / ratio, freq_axis[-1])

        shifted = np.empty_like(spectrum)
        log_spectrum = np.log(spectrum + self._EPSILON)
        for idx in range(spectrum.shape[0]):
            shifted_log = np.interp(
                source_axis, freq_axis, log_spectrum[idx], left=log_spectrum[idx, 0]
            )
            shifted[idx] = np.exp(shifted_log)
        return shifted
