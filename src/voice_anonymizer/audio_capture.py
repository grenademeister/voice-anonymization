"""Input audio capture utilities."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

import sounddevice as sd

from .config import AppConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """Continuously captures frames from the configured microphone device."""

    def __init__(self, config: AppConfig, sd_module: Any | None = None) -> None:
        self._config = config
        self._sd = sd_module or sd
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        """Start capturing microphone audio and invoke *on_frame* for each chunk."""

        if self._running.is_set():
            raise RuntimeError("Audio capture already running")

        self._running.set()

        def _callback(indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:  # type: ignore[override]
            if status:
                logger.warning("Input stream status: %s", status)
            try:
                self._queue.put_nowait(indata.copy())
            except queue.Full:
                logger.warning("Input queue overflow; dropping audio frame")

        self._stream = self._sd.InputStream(
            samplerate=self._config.sample_rate,
            blocksize=self._config.frame_hop_samples,
            channels=self._config.channels,
            dtype="float32",
            callback=_callback,
            device=self._config.input_device,
        )
        self._stream.start()

        def _worker() -> None:
            while self._running.is_set():
                try:
                    frame = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                on_frame(frame)

        self._thread = threading.Thread(target=_worker, name="AudioCaptureThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing audio and clean up resources."""

        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue = queue.Queue(maxsize=8)
