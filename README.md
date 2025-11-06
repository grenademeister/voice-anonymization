# Audio Passthrough (Base Branch)

This is the **base** branch containing a minimal audio passthrough implementation.

> **Note:** For the full voice anonymization engine, switch to the `main` branch.

## What This Branch Does

This branch implements a simple audio pipeline:
1. **Captures audio** from your microphone
2. **Applies identity transform** (passthrough - no modification)
3. **Outputs the audio** directly to your speaker

## Purpose

This branch serves as:
- A baseline for testing audio I/O setup
- A debugging tool for audio device configuration
- A starting point for developing custom audio transformations

## Getting Started

### 1. Install Dependencies

```bash
pip install -e .
```

Or with uv:

```bash
uv sync
```

### 2. Run the Passthrough

```bash
python -m voice_anonymizer.main
```

Or with uv:

```bash
uv run python -m voice_anonymizer.main
```

### 3. List Available Audio Devices

```bash
python -m sounddevice
```

### 4. Run with Specific Devices

```bash
python -m voice_anonymizer.main --input-device 0 --output-device 1
```

## Command-Line Options

- `--input-device <id>` - Microphone device index/name
- `--output-device <id>` - Speaker device index/name
- `--sample-rate <rate>` - Sample rate (default: 16000)
- `--frame-length <ms>` - Frame length in milliseconds (default: 40.0)
- `--frame-hop <ms>` - Frame hop in milliseconds (default: 10.0)
- `--debug` - Enable debug logging

## Project Structure

```
voice-anonymization/ (base branch)
├── src/
│   └── voice_anonymizer/
│       ├── __init__.py          # Package exports
│       ├── audio_capture.py     # Microphone input
│       ├── audio_output.py      # Speaker output
│       ├── config.py            # Configuration
│       └── main.py              # Main entry point
├── pyproject.toml               # Project metadata
├── BASE_README.md               # Detailed base branch docs
└── README.md                    # This file
```

## Architecture

```
Microphone → AudioCapture → [Identity Transform] → AudioOutput → Speaker
                                      ↓
                              (No modification)
```

## For Full Voice Anonymization

Switch to the `main` branch:

```bash
git checkout main
```

The main branch includes:
- Voice transformation engine
- Model loading
- Spectral envelope blending
- Pitch normalization
- Visualization tools

```bash
uv run python scripts/build_windows_exe.py --dist dist/windows
```

This command produces a standalone executable in `dist/windows`, bundling the anonymizer code and `avg_model.json`. For custom packaging, edit `packaging/voice_anonymizer.spec` or adjust the build script flags.

## License

This project is released under the MIT License. See `LICENSE` for details.
