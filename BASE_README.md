# Base Branch - Simple Audio Passthrough

This is the **base** branch that contains a minimal audio passthrough implementation.

## What it does

This branch implements a simple audio pipeline that:
1. **Captures audio** from the microphone
2. **Applies identity transform** (dummy/passthrough - no modification)
3. **Outputs the audio** directly to the speaker

## Key Components

### Essential Files Copied from Main Branch

- `src/voice_anonymizer/audio_capture.py` - Handles microphone input
- `src/voice_anonymizer/audio_output.py` - Handles speaker output
- `src/voice_anonymizer/config.py` - Configuration (simplified)
- `src/voice_anonymizer/main.py` - Main application loop (simplified)

### Removed Components

The following components from the main branch are NOT included in base:
- `transformation_engine.py` - Complex voice transformation logic
- `model_loader.py` - Model loading functionality
- `visualizer.py` - Visualization and detailed logging
- Model files and transformation logic

## Usage

Run the passthrough audio:

```bash
python -m voice_anonymizer.main
```

### Command-line Options

- `--input-device <id>` - Select input device (microphone)
- `--output-device <id>` - Select output device (speaker)
- `--sample-rate <rate>` - Sample rate (default: 16000)
- `--frame-length <ms>` - Frame length in milliseconds (default: 40.0)
- `--frame-hop <ms>` - Frame hop in milliseconds (default: 10.0)
- `--debug` - Enable debug logging

## Purpose

This branch serves as a baseline/testing branch to:
- Verify audio input/output setup works correctly
- Test audio device configuration
- Measure baseline latency
- Debug audio pipeline issues without transformation complexity

## Architecture

```
Microphone → AudioCapture → Identity Transform → AudioOutput → Speaker
                                    ↓
                            (No modification - just pass through)
```
