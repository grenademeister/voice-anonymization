"""Real-time audio visualization utilities."""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class AudioVisualizer:
    """Visualizes input and output audio in the console."""

    def __init__(self, window_size: int = 50, bar_width: int = 40) -> None:
        self._window_size = window_size
        self._bar_width = bar_width
        self._input_levels: deque[float] = deque(maxlen=window_size)
        self._output_levels: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._frame_count = 0

    def update_input(self, frame: "np.ndarray") -> None:
        """Update visualization with new input audio frame."""
        import numpy as np

        rms = float(np.sqrt(np.mean(frame**2)))
        with self._lock:
            self._input_levels.append(rms)
            self._frame_count += 1

            # Display every 10 frames to avoid console spam
            if self._frame_count % 10 == 0:
                self._display()

    def update_output(self, frame: "np.ndarray") -> None:
        """Update visualization with new output audio frame."""
        import numpy as np

        rms = float(np.sqrt(np.mean(frame**2)))
        with self._lock:
            self._output_levels.append(rms)

    def _display(self) -> None:
        """Display current audio levels in console."""
        if not self._input_levels and not self._output_levels:
            return

        # Calculate average levels
        avg_input = (
            sum(self._input_levels) / len(self._input_levels)
            if self._input_levels
            else 0.0
        )
        avg_output = (
            sum(self._output_levels) / len(self._output_levels)
            if self._output_levels
            else 0.0
        )

        # Calculate peak levels
        peak_input = max(self._input_levels) if self._input_levels else 0.0
        peak_output = max(self._output_levels) if self._output_levels else 0.0

        # Create bar visualization
        input_bar = self._create_bar(avg_input, peak_input)
        output_bar = self._create_bar(avg_output, peak_output)

        # Print visualization
        print(f"\r{'=' * 60}", end="")
        print(
            f"\rFrame {self._frame_count:6d} | "
            f"INPUT:  {input_bar} {avg_input:6.4f} (peak: {peak_input:6.4f}) | "
            f"OUTPUT: {output_bar} {avg_output:6.4f} (peak: {peak_output:6.4f})",
            end="",
            flush=True,
        )

    def _create_bar(self, avg: float, peak: float) -> str:
        """Create a visual bar representing audio level."""
        # Scale to bar width (assuming max RMS of 0.5 for reasonable speech)
        avg_scaled = min(int(avg * self._bar_width / 0.5), self._bar_width)
        peak_scaled = min(int(peak * self._bar_width / 0.5), self._bar_width)

        bar = ["░"] * self._bar_width
        for i in range(avg_scaled):
            bar[i] = "█"
        if peak_scaled < self._bar_width and peak_scaled > avg_scaled:
            bar[peak_scaled] = "▓"

        return "".join(bar)

    def print_comparison(
        self, input_frame: "np.ndarray", output_frame: "np.ndarray"
    ) -> None:
        """Print detailed comparison between input and output frames."""
        import numpy as np

        input_rms = float(np.sqrt(np.mean(input_frame**2)))
        output_rms = float(np.sqrt(np.mean(output_frame**2)))

        input_peak = float(np.max(np.abs(input_frame)))
        output_peak = float(np.max(np.abs(output_frame)))

        input_mean = float(np.mean(input_frame))
        output_mean = float(np.mean(output_frame))

        print("\n" + "=" * 80)
        print("AUDIO COMPARISON:")
        print(
            f"  Input  - RMS: {input_rms:.6f}, Peak: {input_peak:.6f}, Mean: {input_mean:.6f}, Shape: {input_frame.shape}"
        )
        print(
            f"  Output - RMS: {output_rms:.6f}, Peak: {output_peak:.6f}, Mean: {output_mean:.6f}, Shape: {output_frame.shape}"
        )
        print(
            f"  Ratio  - RMS: {output_rms/input_rms if input_rms > 0 else 0:.2f}x, Peak: {output_peak/input_peak if input_peak > 0 else 0:.2f}x"
        )
        print("=" * 80)


class DetailedAudioLogger:
    """Logs detailed audio statistics periodically."""

    def __init__(self, log_interval: int = 100) -> None:
        self._log_interval = log_interval
        self._input_count = 0
        self._output_count = 0
        self._input_stats: list[tuple[float, float, float]] = []  # (rms, peak, mean)
        self._output_stats: list[tuple[float, float, float]] = []

    def log_input(self, frame: "np.ndarray") -> None:
        """Log input frame statistics."""
        import numpy as np

        self._input_count += 1
        rms = float(np.sqrt(np.mean(frame**2)))
        peak = float(np.max(np.abs(frame)))
        mean = float(np.mean(frame))

        self._input_stats.append((rms, peak, mean))

        if self._input_count % self._log_interval == 0:
            self._print_stats("INPUT", self._input_stats, self._input_count)
            self._input_stats.clear()

    def log_output(self, frame: "np.ndarray") -> None:
        """Log output frame statistics."""
        import numpy as np

        self._output_count += 1
        rms = float(np.sqrt(np.mean(frame**2)))
        peak = float(np.max(np.abs(frame)))
        mean = float(np.mean(frame))

        self._output_stats.append((rms, peak, mean))

        if self._output_count % self._log_interval == 0:
            self._print_stats("OUTPUT", self._output_stats, self._output_count)
            self._output_stats.clear()

    def _print_stats(
        self, label: str, stats: list[tuple[float, float, float]], count: int
    ) -> None:
        """Print aggregated statistics."""
        if not stats:
            return

        import numpy as np

        rms_values = [s[0] for s in stats]
        peak_values = [s[1] for s in stats]
        mean_values = [s[2] for s in stats]

        logger.info(
            "%s [frames %d-%d]: RMS(avg=%.4f, min=%.4f, max=%.4f), "
            "Peak(avg=%.4f, min=%.4f, max=%.4f), Mean(avg=%.6f)",
            label,
            count - len(stats) + 1,
            count,
            np.mean(rms_values),
            np.min(rms_values),
            np.max(rms_values),
            np.mean(peak_values),
            np.min(peak_values),
            np.max(peak_values),
            np.mean(mean_values),
        )
