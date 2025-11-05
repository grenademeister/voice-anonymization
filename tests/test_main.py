from pathlib import Path

import pytest

pytest.importorskip("numpy")

from voice_anonymizer.main import parse_args


def test_parse_args_overrides_defaults(tmp_path):
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")
    args = parse_args([
        "--input-device",
        "mic",
        "--output-device",
        "sink",
        "--model",
        str(model),
        "--blend",
        "0.5",
        "--sample-rate",
        "22050",
        "--frame-length",
        "30",
        "--frame-hop",
        "15",
    ])
    assert args.input_device == "mic"
    assert args.output_device == "sink"
    assert args.model_path == Path(model)
    assert args.blend_coefficient == 0.5
    assert args.sample_rate == 22050
    assert args.frame_length_ms == 30
    assert args.frame_hop_ms == 15
