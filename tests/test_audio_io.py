import threading

import pytest

np = pytest.importorskip("numpy")

from voice_anonymizer.audio_capture import AudioCapture
from voice_anonymizer.audio_output import AudioOutput
from voice_anonymizer.config import AppConfig


class FakeInputStream:
    def __init__(self, *, blocksize: int, channels: int, callback, frames: list[np.ndarray]):
        self.blocksize = blocksize
        self.channels = channels
        self.callback = callback
        self.frames = [frame.copy() for frame in frames]
        self.started = False

    def start(self):
        self.started = True
        for frame in self.frames:
            self.callback(frame, frame.shape[0], None, 0)

    def stop(self):
        self.started = False

    def close(self):
        pass


class FakeOutputStream:
    def __init__(self, *, blocksize: int, channels: int, callback):
        self.blocksize = blocksize
        self.channels = channels
        self.callback = callback
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass

    def pull(self, frames: int | None = None) -> np.ndarray:
        frames = frames or self.blocksize
        buffer = np.zeros((frames, self.channels), dtype=np.float32)
        self.callback(buffer, frames, None, 0)
        return buffer


class FakeSoundDevice:
    def __init__(self, input_frames: list[np.ndarray] | None = None):
        self._input_frames = input_frames or []
        self.output_streams: list[FakeOutputStream] = []

    def InputStream(self, *, samplerate, blocksize, channels, dtype, callback, device=None):  # noqa: N802 - mimic sounddevice API
        frames = self._input_frames or [np.zeros((blocksize, channels), dtype=np.float32)]
        return FakeInputStream(blocksize=blocksize, channels=channels, callback=callback, frames=frames)

    def OutputStream(self, *, samplerate, blocksize, channels, dtype, callback, device=None):  # noqa: N802 - mimic sounddevice API
        stream = FakeOutputStream(blocksize=blocksize, channels=channels, callback=callback)
        self.output_streams.append(stream)
        return stream


def test_audio_capture_emits_frames():
    config = AppConfig()
    hop = config.frame_hop_samples
    sample_frame = np.linspace(0.0, 1.0, num=hop * config.channels, dtype=np.float32).reshape(hop, config.channels)
    fake_sd = FakeSoundDevice(input_frames=[sample_frame])
    capture = AudioCapture(config, sd_module=fake_sd)

    collected: list[np.ndarray] = []
    event = threading.Event()

    def _on_frame(frame: np.ndarray):
        collected.append(frame)
        event.set()

    capture.start(_on_frame)
    assert event.wait(timeout=0.5)
    capture.stop()

    assert len(collected) == 1
    np.testing.assert_allclose(collected[0], sample_frame)


def test_audio_capture_rejects_second_start():
    config = AppConfig()
    fake_sd = FakeSoundDevice()
    capture = AudioCapture(config, sd_module=fake_sd)
    capture.start(lambda frame: None)
    with pytest.raises(RuntimeError):
        capture.start(lambda frame: None)
    capture.stop()


def test_audio_output_streams_enqueued_frames():
    config = AppConfig()
    fake_sd = FakeSoundDevice()
    output = AudioOutput(config, sd_module=fake_sd)
    output.start()
    assert fake_sd.output_streams, "Output stream was not created"
    stream = fake_sd.output_streams[0]

    silence = stream.pull()
    assert np.allclose(silence, 0.0)

    frame = np.linspace(0.0, 1.0, num=config.frame_hop_samples, dtype=np.float32).reshape((-1, 1))
    output.enqueue([frame])

    rendered = stream.pull()
    np.testing.assert_allclose(rendered[: frame.shape[0], 0], frame[:, 0])

    output.stop()
