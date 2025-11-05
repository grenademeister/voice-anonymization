# Real-Time Voice Anonymization Engine

This repository contains a local-first voice anonymizer that captures audio from a microphone, transforms it to obscure speaker identity, and plays the anonymized stream through a selected output device. The processing pipeline performs frame-based spectral envelope blending and pitch normalization using precomputed averaged voice statistics.

## Features

- **Real-time capture and playback** via [SoundDevice](https://python-sounddevice.readthedocs.io/)
- **Spectral envelope normalization** blending the incoming magnitude spectrum with an averaged reference
- **Pitch normalization** toward the averaged fundamental frequency using lightweight resampling
- **Configurable anonymization strength** through the blend coefficient
- **Packaged averaged model** (`models/avg_model.json`) containing spectral and pitch statistics
- **Packaging workflow** for creating a standalone Windows executable with PyInstaller

## Project Layout

```
voice-anonymization/
├── models/
│   └── avg_model.json
├── packaging/
│   └── voice_anonymizer.spec
├── scripts/
│   └── build_windows_exe.py
├── src/
│   └── voice_anonymizer/
│       ├── __init__.py
│       ├── audio_capture.py
│       ├── audio_output.py
│       ├── config.py
│       ├── main.py
│       ├── model_loader.py
│       └── transformation_engine.py
└── pyproject.toml
```

## Getting Started

1. **Synchronize dependencies with [uv](https://docs.astral.sh/uv/)**

   ```bash
   uv sync
   ```

   This command creates a local `.venv/` with the runtime dependencies and development tooling (pytest, etc.).

2. **Run the anonymizer**

   ```bash
   uv run python -m voice_anonymizer.main --input-device <mic> --output-device <sink>
   ```

   Use `uv run python -m sounddevice` to list device indices/names. The anonymizer runs until interrupted (Ctrl+C) and maintains a mono, 16 kHz signal path with ~40 ms frames.

3. **Execute the test suite**

   ```bash
   uv run pytest
   ```

### Configuration Flags

- `--model`: Path to an averaged model JSON/NPZ/PKL file
- `--blend`: Blend coefficient (0.0 retains original, 1.0 fully averaged)
- `--sample-rate`: Processing sample rate (default 16000)
- `--frame-length`: Frame analysis window in milliseconds (default 40)
- `--frame-hop`: Frame hop in milliseconds (default 10)

## Averaged Model Format

The bundled `models/avg_model.json` stores two fields:

- `spectral_envelope`: Array of length `frame_length_samples // 2 + 1` representing the averaged magnitude spectrum
- `average_f0`: Scalar value with the averaged fundamental frequency in Hz

To generate your own model, compute averaged spectral envelopes and mean F0 statistics from multiple speakers and export them as JSON (or NPZ/PKL). Update the `--model` flag to point to the new file.

## Packaging for Windows

The project ships with a PyInstaller workflow:

```bash
uv run python scripts/build_windows_exe.py --dist dist/windows
```

This command produces a standalone executable in `dist/windows`, bundling the anonymizer code and `avg_model.json`. For custom packaging, edit `packaging/voice_anonymizer.spec` or adjust the build script flags.

## License

This project is released under the MIT License. See `LICENSE` for details.
