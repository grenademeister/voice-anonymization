"""Detailed test script to debug transformation quality."""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice_anonymizer.config import AppConfig
from voice_anonymizer.model_loader import load_averaged_voice_model
from voice_anonymizer.transformation_engine import TransformationEngine

# Test with different blend coefficients
blend_coefficients = [0.0, 0.3, 0.5, 0.7, 0.95]

for blend in blend_coefficients:
    print(f"\n{'='*60}")
    print(f"Testing with blend coefficient: {blend}")
    print(f"{'='*60}")

    # Create config
    config = AppConfig(
        sample_rate=16000,
        frame_length_ms=40.0,
        frame_hop_ms=10.0,
        channels=1,
        blend_coefficient=blend,
    )

    # Load model
    model = load_averaged_voice_model(Path("models/avg_model.json"))

    # Create engine
    engine = TransformationEngine(config, model)

    # Create a test frame with a sine wave (simulated tone)
    frame_size = config.frame_hop_samples  # 160 samples
    t = np.linspace(0, frame_size / config.sample_rate, frame_size, endpoint=False)

    # Create a 200 Hz tone (typical male voice fundamental)
    test_frame = (0.1 * np.sin(2 * np.pi * 200 * t)).reshape(-1, 1).astype(np.float32)

    print(f"Input: 200 Hz sine wave, RMS={np.sqrt(np.mean(test_frame**2)):.6f}")

    # Transform it
    transformed = engine.process(test_frame)

    print(f"Output: RMS={np.sqrt(np.mean(transformed**2)):.6f}")

    # Check spectral content
    input_fft = np.fft.rfft(test_frame.flatten())
    output_fft = np.fft.rfft(transformed.flatten())

    input_power = np.abs(input_fft)
    output_power = np.abs(output_fft)

    input_peak_freq = np.argmax(input_power) * config.sample_rate / frame_size
    output_peak_freq = np.argmax(output_power) * config.sample_rate / frame_size

    print(f"Input peak frequency: {input_peak_freq:.1f} Hz")
    print(f"Output peak frequency: {output_peak_freq:.1f} Hz")

    # Check if output is noisy (high frequency content)
    low_freq_power = np.sum(output_power[:20])  # 0-1250 Hz
    high_freq_power = np.sum(output_power[20:])  # 1250+ Hz

    if high_freq_power > low_freq_power:
        print(
            "⚠️  WARNING: Output has more high-frequency than low-frequency content (noisy)"
        )
    else:
        print(
            f"✓ Output has {low_freq_power/high_freq_power:.2f}x more low-freq than high-freq"
        )

print(f"\n{'='*60}")
print("Recommendation:")
print("- If blend=0.0 sounds good, the problem is in the spectral blending")
print("- If even blend=0.0 sounds noisy, the problem is in pitch shifting")
print("- Try reducing blend coefficient (e.g., 0.3-0.5) for better quality")
print(f"{'='*60}")
