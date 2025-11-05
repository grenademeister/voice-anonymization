"""Application entry point coordinating capture, transformation, and output."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from contextlib import contextmanager
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .audio_capture import AudioCapture
from .audio_output import AudioOutput
from .config import AppConfig, DEFAULT_CONFIG
from .model_loader import load_averaged_voice_model
from .transformation_engine import TransformationEngine

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="Real-time voice anonymizer")
    parser.add_argument("--input-device", type=str, default=None, help="SoundDevice input identifier")
    parser.add_argument("--output-device", type=str, default=None, help="SoundDevice output identifier")
    parser.add_argument("--model", type=str, default=str(DEFAULT_CONFIG.model_path), help="Path to averaged model")
    parser.add_argument("--blend", type=float, default=DEFAULT_CONFIG.blend_coefficient, help="Blend coefficient")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_CONFIG.sample_rate, help="Processing sample rate")
    parser.add_argument("--frame-length", type=float, default=DEFAULT_CONFIG.frame_length_ms, help="Frame length (ms)")
    parser.add_argument("--frame-hop", type=float, default=DEFAULT_CONFIG.frame_hop_ms, help="Frame hop (ms)")
    args = parser.parse_args(argv)
    return AppConfig(
        input_device=args.input_device,
        output_device=args.output_device,
        model_path=DEFAULT_CONFIG.model_path.__class__(args.model),
        blend_coefficient=args.blend,
        sample_rate=args.sample_rate,
        frame_length_ms=args.frame_length,
        frame_hop_ms=args.frame_hop,
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
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    config = parse_args(argv)
    logger.info("Starting voice anonymizer with config: %s", config)

    model = load_averaged_voice_model(config.model_path)
    engine = TransformationEngine(config, model)
    capture = AudioCapture(config)
    output = AudioOutput(config)

    output.start()

    with _graceful_shutdown() as should_stop:
        def _process_frame(frame: "np.ndarray") -> None:
            transformed = engine.process(frame)
            output.enqueue([transformed])
            if should_stop():
                capture.stop()

        capture.start(_process_frame)
        logger.info("Voice anonymizer running. Press Ctrl+C to exit.")
        while not should_stop():
            signal.pause()

    output.stop()
    return 0


if __name__ == "__main__":
    sys.exit(run())
