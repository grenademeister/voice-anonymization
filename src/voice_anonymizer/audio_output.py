"""Output audio streaming utilities."""

from __future__ import annotations

import logging
import queue
from collections.abc import Iterable

import numpy as np
import sounddevice as sd

from .config import AppConfig

logger = logging.getLogger(__name__)


class AudioOutput:
    """Streams anonymized frames to the selected playback device."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        self._stream: sd.OutputStream | None = None
        self._running = False

    def start(self) -> None:
        """Start the audio output stream."""

        if self._running:
            raise RuntimeError("Audio output already running")

        self._running = True

        def _callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:  # type: ignore[override]
            if status:
                logger.warning("Output stream status: %s", status)
            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                outdata.fill(0)
                return
            if chunk.shape[0] < frames:
                padded = np.zeros((frames, self._config.channels), dtype=chunk.dtype)
                padded[: chunk.shape[0], :] = chunk
                chunk = padded
            outdata[:] = chunk

        self._stream = sd.OutputStream(
            samplerate=self._config.sample_rate,
            blocksize=self._config.frame_hop_samples,
            channels=self._config.channels,
            dtype="float32",
            callback=_callback,
            device=self._config.output_device,
        )
        self._stream.start()

    def enqueue(self, frames: Iterable[np.ndarray]) -> None:
        """Add frames to the playback queue."""

        for frame in frames:
            try:
                self._queue.put_nowait(frame.astype("float32", copy=False))
            except queue.Full:
                logger.warning("Output queue overflow; dropping audio frame")

    def stop(self) -> None:
        """Stop playback and release resources."""

        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue = queue.Queue(maxsize=8)
