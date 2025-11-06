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


def _sine_stream(config: AppConfig, frequency: float, frames: int) -> list[np.ndarray]:
    hop = config.frame_hop_samples
    sample_rate = config.sample_rate
    offset = 0
    output: list[np.ndarray] = []
    for _ in range(frames):
        t = (np.arange(hop) + offset) / sample_rate
        output.append(np.sin(2.0 * np.pi * frequency * t).astype(np.float32))
        offset += hop
    return output


def _dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(samples.size)
    spectrum = np.fft.rfft(samples * window)
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(np.abs(spectrum)))])


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


def test_pitch_shifts_toward_model_average():
    config, engine = _make_engine(blend=1.0)
    frames = _sine_stream(config, frequency=120.0, frames=12)
    outputs = [engine.process(frame.reshape((-1, 1)))[:, 0] for frame in frames]
    tail = np.concatenate(outputs[-4:])
    dominant = _dominant_frequency(tail, config.sample_rate)
    assert dominant > 140.0
    assert dominant == pytest.approx(engine._model.average_f0, abs=40.0)


def test_output_rms_tracks_model_target():
    config, engine = _make_engine(blend=1.0)
    frames = _sine_stream(config, frequency=150.0, frames=12)
    outputs = [engine.process(frame.reshape((-1, 1)))[:, 0] for frame in frames]
    tail = np.concatenate(outputs[-4:])
    rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float64))))
    envelope = engine._model.spectral_envelope.astype(np.float64, copy=False)
    target_rms = float(np.sqrt(np.mean(np.square(envelope))))
    assert rms == pytest.approx(target_rms, rel=0.15, abs=0.05)
