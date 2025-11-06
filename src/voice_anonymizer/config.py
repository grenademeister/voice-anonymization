"""Application configuration models and helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(slots=True)
class AppConfig:
    """Configuration parameters that control the audio anonymization pipeline."""

    sample_rate: int = 16_000
    frame_length_ms: float = 40.0
    frame_hop_ms: float = 10.0
    channels: int = 1
    input_device: int | str | None = None
    output_device: int | str | None = None
    pitch_shift_semitones: float = -4.0
    formant_shift_ratio: float = 1.15
    amplitude_scale: float = 1.0
    noise_level: float = 0.005
    world_block_ms: float = 120.0
    min_f0: float = 75.0
    max_f0: float = 450.0

    @property
    def frame_length_samples(self) -> int:
        """Number of samples contained in a single analysis frame."""

        return int(self.sample_rate * self.frame_length_ms / 1_000.0)

    @property
    def frame_hop_samples(self) -> int:
        """Number of samples between consecutive frames."""

        return int(self.sample_rate * self.frame_hop_ms / 1_000.0)

    @property
    def world_block_samples(self) -> int:
        """Chunk size processed by the WORLD vocoder."""

        return max(1, int(self.sample_rate * self.world_block_ms / 1_000.0))

    @property
    def pitch_shift_ratio(self) -> float:
        """Pitch scaling factor derived from semitone shift."""

        return 2.0 ** (self.pitch_shift_semitones / 12.0)

    def with_preset(self, preset: str) -> "AppConfig":
        """Return a copy configured for a named anonymization preset."""

        try:
            overrides = ANONYMIZATION_PRESETS[preset]
        except KeyError as exc:
            raise ValueError(f"Unknown anonymization preset: {preset}") from exc
        return replace(self, **overrides)


ANONYMIZATION_PRESETS: dict[str, dict[str, float]] = {
    "off": {
        "pitch_shift_semitones": 0.0,
        "formant_shift_ratio": 1.0,
        "amplitude_scale": 1.0,
        "noise_level": 0.0,
    },
    "light": {
        "pitch_shift_semitones": -2.5,
        "formant_shift_ratio": 1.08,
        "amplitude_scale": 1.0,
        "noise_level": 0.003,
    },
    "medium": {
        "pitch_shift_semitones": -5.0,
        "formant_shift_ratio": 1.18,
        "amplitude_scale": 1.05,
        "noise_level": 0.007,
    },
    "strong": {
        "pitch_shift_semitones": -8.0,
        "formant_shift_ratio": 1.3,
        "amplitude_scale": 1.1,
        "noise_level": 0.015,
    },
}
DEFAULT_PRESET_NAME = "medium"


DEFAULT_CONFIG = AppConfig()
"""Default configuration used when no overrides are supplied."""
