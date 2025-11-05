from voice_anonymizer.config import AppConfig


def test_frame_length_and_hop_conversion():
    config = AppConfig(sample_rate=16000, frame_length_ms=40, frame_hop_ms=10)
    assert config.frame_length_samples == 640
    assert config.frame_hop_samples == 160


def test_channels_default_to_mono():
    assert AppConfig().channels == 1
