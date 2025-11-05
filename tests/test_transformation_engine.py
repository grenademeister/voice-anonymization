import pytest

np = pytest.importorskip("numpy")

from voice_anonymizer.config import AppConfig
from voice_anonymizer.model_loader import AveragedVoiceModel
from voice_anonymizer.transformation_engine import TransformationEngine


def _make_engine(blend: float = 0.0, channels: int = 1) -> tuple[AppConfig, TransformationEngine]:
    config = AppConfig(blend_coefficient=blend, channels=channels)
    fft_bins = config.frame_length_samples // 2 + 1
    spectral = np.linspace(0.2, 1.0, num=fft_bins, dtype=np.float32)
    model = AveragedVoiceModel(spectral_envelope=spectral, average_f0=190.0)
    return config, TransformationEngine(config, model)


def _sine_chunk(config: AppConfig, frequency: float = 220.0) -> np.ndarray:
    hop = config.frame_hop_samples
    t = np.arange(hop) / config.sample_rate
    chunk = np.sin(2.0 * np.pi * frequency * t).astype(np.float32)
    return chunk


def test_identity_preserved_when_blend_is_zero():
    config, engine = _make_engine(blend=0.0)
    chunk = _sine_chunk(config)
    outputs = []
    for _ in range(8):
        outputs.append(engine.process(chunk.reshape((-1, 1))))
    recovered = outputs[-1][:, 0]
    np.testing.assert_allclose(recovered, chunk, rtol=1e-3, atol=1e-4)


def test_transform_differs_from_input_with_blending():
    blend = 0.85
    config, engine = _make_engine(blend=blend)
    chunk = _sine_chunk(config, frequency=140.0)
    outputs = []
    for _ in range(8):
        outputs.append(engine.process(chunk.reshape((-1, 1))))
    anonymized = outputs[-1][:, 0]
    assert not np.allclose(anonymized, chunk, atol=5e-3)
    assert np.max(np.abs(anonymized)) <= 1.5


def test_multichannel_output_is_broadcast():
    config, engine = _make_engine(blend=0.5, channels=2)
    chunk = np.stack([_sine_chunk(config), _sine_chunk(config)], axis=1)
    processed = engine.process(chunk)
    assert processed.shape == (config.frame_hop_samples, 2)
    np.testing.assert_allclose(processed[:, 0], processed[:, 1])
