"""Output audio streaming utilities."""

from __future__ import annotations

import logging
import queue
from collections.abc import Iterable
from typing import Any

import numpy as np
import sounddevice as sd

from .config import AppConfig

logger = logging.getLogger(__name__)


class AudioOutput:
    """Streams anonymized frames to the selected playback device."""

    def __init__(self, config: AppConfig, sd_module: Any | None = None) -> None:
        self._config = config
        self._sd = sd_module or sd
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        self._stream: sd.OutputStream | None = None
        self._running = False

    def start(self) -> None:
        """Start the audio output stream."""

        if self._running:
            raise RuntimeError("Audio output already running")

        self._running = True

        output_frame_count = 0
        underrun_count = 0

        def _callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:  # type: ignore[override]
            nonlocal output_frame_count, underrun_count
            output_frame_count += 1
            if status:
                logger.warning("Output stream status: %s", status)
            try:
                chunk = self._queue.get_nowait()
                if output_frame_count % 100 == 0:
                    logger.debug(
                        "Output frame %d: shape=%s, rms=%.4f",
                        output_frame_count,
                        chunk.shape,
                        np.sqrt(np.mean(chunk**2)),
                    )
            except queue.Empty:
                underrun_count += 1
                if underrun_count % 50 == 0:
                    logger.debug(
                        "Output queue underrun (count=%d); filling with silence",
                        underrun_count,
                    )
                outdata.fill(0)
                return
            if chunk.shape[0] < frames:
                padded = np.zeros((frames, self._config.channels), dtype=chunk.dtype)
                padded[: chunk.shape[0], :] = chunk
                chunk = padded
            outdata[:] = chunk

        self._stream = self._sd.OutputStream(
            samplerate=self._config.sample_rate,
            blocksize=self._config.frame_hop_samples,
            channels=self._config.channels,
            dtype="float32",
            callback=_callback,
            device=self._config.output_device,
        )
        logger.info(
            "Starting output stream: device=%s, sample_rate=%d, channels=%d, blocksize=%d",
            self._config.output_device,
            self._config.sample_rate,
            self._config.channels,
            self._config.frame_hop_samples,
        )
        self._stream.start()

    def enqueue(self, frames: Iterable[np.ndarray]) -> None:
        """Add frames to the playback queue."""

        for frame in frames:
            array = np.asarray(frame, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape((-1, self._config.channels))
            if array.shape[1] != self._config.channels:
                raise ValueError(
                    f"Frame has {array.shape[1]} channels but output expects {self._config.channels}"
                )
            try:
                self._queue.put_nowait(array)
            except queue.Full:
                logger.warning(
                    "Output queue overflow; dropping audio frame (queue size=%d)",
                    self._queue.qsize(),
                )

    def stop(self) -> None:
        """Stop playback and release resources."""

        logger.info("Stopping audio output")
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue = queue.Queue(maxsize=8)
        logger.info("Audio output stopped")
