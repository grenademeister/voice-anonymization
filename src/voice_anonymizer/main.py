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

import sounddevice as sd

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime voice anonymizer")
    parser.add_argument(
        "--input-device",
        type=_device_type,
        default=None,
        help="SoundDevice input identifier (device index or name). Leave empty to choose interactively.",
    )
    parser.add_argument(
        "--output-device",
        type=_device_type,
        default=None,
        help="SoundDevice output identifier (device index or name). Leave empty to choose interactively.",
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
        "--pitch-shift",
        type=float,
        default=DEFAULT_CONFIG.pitch_shift_semitones,
        help="Pitch shift applied via WORLD vocoder (in semitones)",
    )
    parser.add_argument(
        "--formant-ratio",
        type=float,
        default=DEFAULT_CONFIG.formant_shift_ratio,
        help="Spectral envelope warp ratio (>1 raises formants, <1 lowers them)",
    )
    parser.add_argument(
        "--amplitude-scale",
        type=float,
        default=DEFAULT_CONFIG.amplitude_scale,
        help="Overall gain applied after resynthesis",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=DEFAULT_CONFIG.noise_level,
        help="Gaussian noise (std-dev) injected to mask artifacts",
    )
    parser.add_argument(
        "--world-block",
        type=float,
        default=DEFAULT_CONFIG.world_block_ms,
        help="WORLD processing block size in milliseconds (controls latency)",
    )
    parser.add_argument(
        "--min-f0",
        type=float,
        default=DEFAULT_CONFIG.min_f0,
        help="Lower bound for voiced F0 tracking",
    )
    parser.add_argument(
        "--max-f0",
        type=float,
        default=DEFAULT_CONFIG.max_f0,
        help="Upper bound for voiced F0 tracking",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


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


def _audio_devices_table() -> list[tuple[int, dict]]:
    devices = []
    for idx, info in enumerate(sd.query_devices()):
        devices.append((idx, info))
    return devices


def _print_device_table(kind: str) -> None:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    header = f"Available {kind} devices:"
    print(header)
    print("-" * len(header))
    for idx, info in _audio_devices_table():
        if info.get(key, 0) <= 0:
            continue
        print(
            f"{idx:>3} | {info['name']} (inputs={info['max_input_channels']}, outputs={info['max_output_channels']})"
        )
    print()


def _prompt_for_device(kind: str) -> int | str | None:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    candidates = [
        (idx, info)
        for idx, info in _audio_devices_table()
        if info.get(key, 0) > 0
    ]
    if not candidates:
        logger.warning("No %s devices found; falling back to defaults", kind)
        return None

    _print_device_table(kind)
    default_device = sd.default.device[0 if kind == "input" else 1]

    while True:
        prompt = (
            f"Select {kind} device index (blank for default {default_device!r}): "
        )
        try:
            selected = input(prompt).strip()
        except EOFError:
            logger.info("No input available; using default device")
            return None

        if not selected:
            return None
        if selected.isdigit():
            idx = int(selected)
            if any(idx == candidate_idx for candidate_idx, _ in candidates):
                return idx
            print("Invalid index. Please choose one from the list.")
            continue
        # Allow partial name matching
        matches = [
            info["name"]
            for _, info in candidates
            if selected.lower() in info["name"].lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print("Multiple matches found; please specify by index instead.")
            continue
        print("No device matched that input; try again.")


def _prepare_config(namespace: argparse.Namespace) -> AppConfig:
    config = AppConfig(
        input_device=namespace.input_device,
        output_device=namespace.output_device,
        sample_rate=namespace.sample_rate,
        frame_length_ms=namespace.frame_length,
        frame_hop_ms=namespace.frame_hop,
        pitch_shift_semitones=namespace.pitch_shift,
        formant_shift_ratio=namespace.formant_ratio,
        amplitude_scale=namespace.amplitude_scale,
        noise_level=namespace.noise_level,
        world_block_ms=namespace.world_block,
        min_f0=namespace.min_f0,
        max_f0=namespace.max_f0,
    )

    if sys.stdin.isatty():
        if config.input_device is None:
            config.input_device = _prompt_for_device("input")
        if config.output_device is None:
            config.output_device = _prompt_for_device("output")

    return config


def run(argv: list[str] | None = None) -> int:
    args_namespace = parse_args(argv)

    if args_namespace.list_devices:
        _print_device_table("input")
        _print_device_table("output")
        return 0

    log_level = logging.DEBUG if args_namespace.debug else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(name)s: %(message)s")

    config = _prepare_config(args_namespace)

    logger.info("Starting voice anonymizer with config: %s", config)

    # Initialize audio I/O
    capture = AudioCapture(config)
    output = AudioOutput(config)
    transformer = FrameTransformer(config)

    output.start()

    with _graceful_shutdown() as should_stop:

        def _process_frame(frame: "np.ndarray") -> None:
            processed = transformer(frame)
            output.enqueue([processed])
            if should_stop():
                capture.stop()

        capture.start(_process_frame)

        logger.info("Audio anonymization running. Press Ctrl+C to exit.")
        logger.info("Input will be routed through WORLD-based anonymizer.")
        while not should_stop():
            time.sleep(0.1)

    output.stop()
    return 0


if __name__ == "__main__":
    sys.exit(run())
