"""Quick test of very low blend coefficients."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice_anonymizer.config import AppConfig
from voice_anonymizer.model_loader import load_averaged_voice_model
from voice_anonymizer.transformation_engine import TransformationEngine

for blend in [0.0, 0.001, 0.01, 0.1, 0.3, 0.5]:
    config = AppConfig(blend_coefficient=blend)
    model = load_averaged_voice_model(Path("models/avg_model.json"))
    engine = TransformationEngine(config, model)

    # Create test signal
    frame_size = config.frame_hop_samples
    t = np.linspace(0, frame_size / config.sample_rate, frame_size, endpoint=False)
    test_frame = (0.1 * np.sin(2 * np.pi * 200 * t)).reshape(-1, 1).astype(np.float32)

    transformed = engine.process(test_frame)

    input_fft = np.fft.rfft(test_frame.flatten())
    output_fft = np.fft.rfft(transformed.flatten())

    input_power = np.abs(input_fft)
    output_power = np.abs(output_fft)

    low_freq_power = np.sum(output_power[:20])
    high_freq_power = np.sum(output_power[20:])

    ratio = low_freq_power / high_freq_power if high_freq_power > 0 else float("inf")

    print(
        f"Blend {blend:5.3f}: Low/High ratio = {ratio:8.2f}x, Output RMS = {np.sqrt(np.mean(transformed**2)):.6f}"
    )
