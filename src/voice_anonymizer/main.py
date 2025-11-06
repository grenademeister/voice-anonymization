"""Application entry point coordinating capture, transformation, and output."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from contextlib import contextmanager
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .audio_capture import AudioCapture
from .audio_output import AudioOutput
from .config import AppConfig, DEFAULT_CONFIG
from .transform import FrameTransformer

logger = logging.getLogger(__name__)


def _device_type(value: str) -> int | str:
    """Convert device argument to int if numeric, otherwise keep as string."""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="Simple audio passthrough test")
    parser.add_argument(
        "--input-device",
        type=_device_type,
        default=None,
        help="SoundDevice input identifier (device index or name)",
    )
    parser.add_argument(
        "--output-device",
        type=_device_type,
        default=None,
        help="SoundDevice output identifier (device index or name)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_CONFIG.sample_rate,
        help="Processing sample rate",
    )
    parser.add_argument(
        "--frame-length",
        type=float,
        default=DEFAULT_CONFIG.frame_length_ms,
        help="Frame length (ms)",
    )
    parser.add_argument(
        "--frame-hop",
        type=float,
        default=DEFAULT_CONFIG.frame_hop_ms,
        help="Frame hop (ms)",
    )
    parser.add_argument(
        "--formant-shift",
        type=float,
        default=DEFAULT_CONFIG.formant_shift_ratio,
        help="WORLD spectral envelope scaling ratio",
    )
    parser.add_argument(
        "--pitch-shift",
        type=float,
        default=DEFAULT_CONFIG.pitch_shift_semitones,
        help="WORLD pitch shift (semitones)",
    )
    parser.add_argument(
        "--spectral-exp",
        type=float,
        default=DEFAULT_CONFIG.spectral_envelope_exponent,
        help="WORLD spectral envelope exponent",
    )
    parser.add_argument(
        "--world-window",
        type=float,
        default=DEFAULT_CONFIG.world_analysis_window_ms,
        help="WORLD analysis window (ms) controlling smoothness vs latency",
    )
    parser.add_argument(
        "--world-mix",
        type=float,
        default=DEFAULT_CONFIG.world_wet_mix,
        help="WORLD wet mix ratio (1.0 = fully processed, 0.0 = dry)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)
    return AppConfig(
        input_device=args.input_device,
        output_device=args.output_device,
        sample_rate=args.sample_rate,
        frame_length_ms=args.frame_length,
        frame_hop_ms=args.frame_hop,
        formant_shift_ratio=args.formant_shift,
        pitch_shift_semitones=args.pitch_shift,
        spectral_envelope_exponent=args.spectral_exp,
        world_analysis_window_ms=args.world_window,
        world_wet_mix=args.world_mix,
    )


@contextmanager
def _graceful_shutdown() -> Iterator[None]:
    previous_handlers: dict[int, signal.Handlers] = {}
    stop = False

    def _handler(signum, frame):  # type: ignore[override]
        nonlocal stop
        logger.info("Received signal %s; shutting down", signum)
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, _handler)
    try:
        yield lambda: stop
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def run(argv: list[str] | None = None) -> int:
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simple audio passthrough test")
    parser.add_argument(
        "--input-device",
        type=_device_type,
        default=None,
        help="SoundDevice input identifier (device index or name)",
    )
    parser.add_argument(
        "--output-device",
        type=_device_type,
        default=None,
        help="SoundDevice output identifier (device index or name)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_CONFIG.sample_rate,
        help="Processing sample rate",
    )
    parser.add_argument(
        "--frame-length",
        type=float,
        default=DEFAULT_CONFIG.frame_length_ms,
        help="Frame length (ms)",
    )
    parser.add_argument(
        "--frame-hop",
        type=float,
        default=DEFAULT_CONFIG.frame_hop_ms,
        help="Frame hop (ms)",
    )
    parser.add_argument(
        "--formant-shift",
        type=float,
        default=DEFAULT_CONFIG.formant_shift_ratio,
        help="WORLD spectral envelope scaling ratio",
    )
    parser.add_argument(
        "--pitch-shift",
        type=float,
        default=DEFAULT_CONFIG.pitch_shift_semitones,
        help="WORLD pitch shift (semitones)",
    )
    parser.add_argument(
        "--spectral-exp",
        type=float,
        default=DEFAULT_CONFIG.spectral_envelope_exponent,
        help="WORLD spectral envelope exponent",
    )
    parser.add_argument(
        "--world-window",
        type=float,
        default=DEFAULT_CONFIG.world_analysis_window_ms,
        help="WORLD analysis window (ms) controlling smoothness vs latency",
    )
    parser.add_argument(
        "--world-mix",
        type=float,
        default=DEFAULT_CONFIG.world_wet_mix,
        help="WORLD wet mix ratio (1.0 = fully processed, 0.0 = dry)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args_namespace = parser.parse_args(argv)

    # Configure logging based on debug flag
    log_level = logging.DEBUG if args_namespace.debug else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(name)s: %(message)s")

    config = AppConfig(
        input_device=args_namespace.input_device,
        output_device=args_namespace.output_device,
        sample_rate=args_namespace.sample_rate,
        frame_length_ms=args_namespace.frame_length,
        frame_hop_ms=args_namespace.frame_hop,
        formant_shift_ratio=args_namespace.formant_shift,
        pitch_shift_semitones=args_namespace.pitch_shift,
        spectral_envelope_exponent=args_namespace.spectral_exp,
        world_analysis_window_ms=args_namespace.world_window,
        world_wet_mix=args_namespace.world_mix,
    )

    logger.info("Starting audio passthrough with config: %s", config)

    # Initialize audio I/O
    capture = AudioCapture(config)
    output = AudioOutput(config)
    transformer = FrameTransformer(config)

    output.start()

    with _graceful_shutdown() as should_stop:

        def _process_frame(frame: "np.ndarray") -> None:
            """Apply WORLD transform to the captured frame and stream it out."""
            processed = transformer(frame)
            output.enqueue([processed])
            if should_stop():
                capture.stop()

        capture.start(_process_frame)

        logger.info("Audio passthrough running. Press Ctrl+C to exit.")
        logger.info("Input audio will be WORLD-processed before playback.")
        while not should_stop():
            time.sleep(0.1)

    output.stop()
    return 0


if __name__ == "__main__":
    sys.exit(run())
