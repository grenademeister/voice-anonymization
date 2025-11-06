"""Application configuration models and helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    """Configuration parameters that control the audio passthrough pipeline."""

    sample_rate: int = 16_000
    frame_length_ms: float = 40.0
    frame_hop_ms: float = 10.0
    channels: int = 1
    input_device: int = 1
    output_device: int = 3
    formant_shift_ratio: float = 1.0
    pitch_shift_semitones: float = 0.01
    spectral_envelope_exponent: float = 1.0
    world_analysis_window_ms: float = 120.0
    world_wet_mix: float = 0.7

    @property
    def frame_length_samples(self) -> int:
        """Number of samples contained in a single analysis frame."""

        return int(self.sample_rate * self.frame_length_ms / 1_000.0)

    @property
    def frame_hop_samples(self) -> int:
        """Number of samples between consecutive frames."""

        return int(self.sample_rate * self.frame_hop_ms / 1_000.0)


DEFAULT_CONFIG = AppConfig()
"""Default configuration used when no overrides are supplied."""
