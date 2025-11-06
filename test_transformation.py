"""Test script to debug transformation engine."""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice_anonymizer.config import AppConfig
from voice_anonymizer.model_loader import load_averaged_voice_model
from voice_anonymizer.transformation_engine import TransformationEngine

# Create config
config = AppConfig(
    sample_rate=16000,
    frame_length_ms=40.0,
    frame_hop_ms=10.0,
    channels=1,
    blend_coefficient=0.95,
)

# Load model
model = load_averaged_voice_model(Path("models/avg_model.json"))

# Create engine
engine = TransformationEngine(config, model)

# Create a test frame with some audio (simulated voice)
np.random.seed(42)
frame_size = config.frame_hop_samples  # 160 samples
test_frame = (
    np.random.randn(frame_size, 1).astype(np.float32) * 0.1
)  # Mono frame with noise

print(f"Test frame shape: {test_frame.shape}")
print(f"Test frame RMS: {np.sqrt(np.mean(test_frame**2)):.6f}")
print(f"Test frame range: [{np.min(test_frame):.6f}, {np.max(test_frame):.6f}]")

# Transform it
transformed = engine.process(test_frame)

print(f"\nTransformed shape: {transformed.shape}")
print(f"Transformed RMS: {np.sqrt(np.mean(transformed**2)):.6f}")
print(f"Transformed range: [{np.min(transformed):.6f}, {np.max(transformed):.6f}]")

# Check if output is basically zero
if np.sqrt(np.mean(transformed**2)) < 1e-6:
    print("\n⚠️  WARNING: Output is essentially zero!")
    print("The transformation is producing silent audio.")
else:
    print("\n✓ Output has non-zero audio")

# Try with a louder signal
print("\n" + "=" * 60)
print("Testing with louder input (RMS=0.3)...")
print("=" * 60)
loud_frame = np.random.randn(frame_size, 1).astype(np.float32) * 0.3
print(f"Loud frame RMS: {np.sqrt(np.mean(loud_frame**2)):.6f}")

transformed_loud = engine.process(loud_frame)
print(f"Transformed loud RMS: {np.sqrt(np.mean(transformed_loud**2)):.6f}")

if np.sqrt(np.mean(transformed_loud**2)) < 1e-3:
    print("\n⚠️  WARNING: Even loud input produces near-zero output!")
else:
    ratio = np.sqrt(np.mean(transformed_loud**2)) / np.sqrt(np.mean(loud_frame**2))
    print(f"\n✓ Output/Input ratio: {ratio:.4f}")
