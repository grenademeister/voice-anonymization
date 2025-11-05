import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from voice_anonymizer.config import AppConfig
from voice_anonymizer.model_loader import AveragedVoiceModel, load_averaged_voice_model


@pytest.fixture()
def sample_model_path(tmp_path: Path) -> Path:
    config = AppConfig()
    fft_bins = config.frame_length_samples // 2 + 1
    payload = {
        "spectral_envelope": np.linspace(0.1, 1.0, num=fft_bins, dtype=np.float32).tolist(),
        "average_f0": 180.0,
    }
    path = tmp_path / "avg_model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_averaged_voice_model_success(sample_model_path: Path):
    model = load_averaged_voice_model(sample_model_path)
    assert isinstance(model, AveragedVoiceModel)
    assert model.spectral_envelope.ndim == 1
    assert model.spectral_envelope.size > 0
    assert model.average_f0 == pytest.approx(180.0)


def test_load_averaged_voice_model_requires_fields(tmp_path: Path):
    path = tmp_path / "invalid_model.json"
    path.write_text(json.dumps({"spectral_envelope": [1.0, 2.0]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_averaged_voice_model(path)
