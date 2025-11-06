# Voice Anonymizer

Realtime voice anonymization pipeline that reshapes pitch, spectral envelope, and voicing statistics using the WORLD vocoder. The application captures audio from a microphone, anonymizes it in-flight, and streams the transformed signal to your chosen playback device.

## Features
- WORLD-based analysis/resynthesis with configurable pitch shifting and formant warping
- Optional amplitude scaling and noise injection for additional masking
- Interactive device picker (and anonymization intensity selector) when input/output devices are not supplied on the command line
- `--list-devices` helper to inspect the current audio topology
- PyInstaller spec for producing a standalone Windows executable

## Setup
```bash
pip install -e .
# or: uv sync
```

## Running
- `python -m voice_anonymizer` – launch with defaults, choose devices interactively if running in a TTY
- `python -m voice_anonymizer --list-devices` – print available devices and exit
- `python -m voice_anonymizer --input-device 2 --output-device 5` – run headless with explicit devices

### Key Options
- `--pitch-shift <semitones>` – scale F0 (`-4.0` by default)
- `--formant-ratio <ratio>` – warp spectral envelope (`1.15` raises formants)
- `--preset <name>` – override anonymization settings via `off`, `light`, `medium`, or `strong`
- `--world-block <ms>` – WORLD processing window (latency control)
- `--amplitude-scale <gain>` – post-resynthesis gain
- `--noise-level <stddev>` – Gaussian noise injected after resynthesis
- `--min-f0` / `--max-f0` – bounds for voiced pitch tracking

Combine options to match the level of anonymization you need, e.g.:
```bash
python -m voice_anonymizer --pitch-shift -6 --formant-ratio 1.25 --noise-level 0.01
```
When running without explicit devices/preset, the CLI prompts you to choose both, with `medium` anonymization selected by default.

## Building a Standalone EXE
PyInstaller is configured via `voice_anonymizer.spec`. Generate a single-file executable with:
```bash
pyinstaller --clean --dist dist --work build voice_anonymizer.spec
```
The resulting binary (`dist/VoiceAnonymizer.exe`) embeds the anonymizer and exposes the same CLI options, including interactive device selection.

Run the executable from a terminal to supply options or use the interactive prompts:
```powershell
.\dist\VoiceAnonymizer.exe --list-devices
```

## Project Layout
- `src/voice_anonymizer/config.py` – runtime configuration model
- `src/voice_anonymizer/audio_capture.py` – microphone capture worker
- `src/voice_anonymizer/transform.py` – WORLD-based anonymization
- `src/voice_anonymizer/audio_output.py` – playback stream handler
- `src/voice_anonymizer/main.py` – CLI entry point and wiring
- `voice_anonymizer.spec` – PyInstaller build recipe

## Notes
- The anonymizer introduces latency equal to `world_block_ms`; lower values reduce latency but require more CPU headroom.
- WORLD processing expects clean input audio. Excessive clipping or noise reduces the quality of the anonymized output.
- When packaging with PyInstaller, ensure the environment includes the correct versions of `numpy`, `pyworld`, and `sounddevice` before building.
