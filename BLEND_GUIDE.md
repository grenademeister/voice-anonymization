# Voice Anonymizer - Blend Coefficient Guide

## Summary of Changes

The voice anonymizer now works correctly with proper blend coefficient control and energy preservation.

## Blend Coefficient Guide

The `--blend` parameter controls how much your voice is anonymized:

| Blend Value | Effect | Speech Quality | Use Case |
|-------------|--------|----------------|----------|
| **0.0** | No anonymization (pass-through) | Perfect clarity | Testing input/output |
| **0.001** | Minimal anonymization | Near-perfect clarity | Very subtle changes |
| **0.01** | Very light anonymization | Excellent clarity | Slight voice modification |
| **0.1** | Light anonymization (default) | Good clarity | Balanced anonymization |
| **0.2** | Moderate anonymization | Fair clarity | Noticeable voice change |
| **0.3** | Strong anonymization | Reduced clarity | Heavy disguise |
| **0.5+** | Very strong anonymization | Poor clarity (noisy) | Maximum disguise |

## Usage Examples

```bash
# Default settings (blend=0.1)
uv run python -m voice_anonymizer.main --input-device 1 --output-device 3

# With visualization
uv run python -m voice_anonymizer.main --input-device 1 --output-device 3 --visualize

# Light anonymization with good quality
uv run python -m voice_anonymizer.main --input-device 1 --output-device 3 --blend 0.05

# Moderate anonymization
uv run python -m voice_anonymizer.main --input-device 1 --output-device 3 --blend 0.2

# Strong anonymization (may sound noisy)
uv run python -m voice_anonymizer.main --input-device 1 --output-device 3 --blend 0.4
```

## Recommendations

- **Start with `--blend 0.1`** for a good balance of anonymization and clarity
- **Use `--blend 0.05` to `--blend 0.15`** for the sweet spot of voice modification with good intelligibility
- **Avoid values above 0.3** unless you need maximum anonymization (audio will be degraded)
- **Use `--visualize`** to see real-time input/output levels

## Technical Details

- **Pitch shifting** is only applied when blend > 0.6 and F0 difference > 20 Hz (to preserve quality)
- **RMS normalization** ensures output volume matches input volume
- **Spectral envelope blending** modifies voice timbre/character
- Lower blend values preserve more of your original voice characteristics
