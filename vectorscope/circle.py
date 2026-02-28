"""Circle display with optional frequency sweep."""

import numpy as np
from .base import VectorScopePlayer


class CirclePlayer(VectorScopePlayer):
    """
    Real-time circle with optional frequency sweep.

    Great for testing your scope setup or debugging issues like
    ghosting (spoiler: if the ghost offset doesn't change with
    frequency, it's a constant delay, not a frequency-dependent issue).
    """

    def __init__(self, freq_min=None, freq_max=None, sweep_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.direction = -1 if kwargs.get('freq_sign', 1) < 0 else 1
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sweep_rate = sweep_rate
        self.sweep_mode = freq_min is not None and freq_max is not None
        self._last_phase = 0

    def audio_callback(self, outdata, frames, time, status):
        """Generate circle in real-time with optional frequency sweep."""
        self._check_status(status)

        if self.sweep_mode:
            t = (self.global_sample + np.arange(frames)) / self.sample_rate
            freq = self.freq_min + (self.freq_max - self.freq_min) * \
                   (0.5 + 0.5 * np.sin(2 * np.pi * self.sweep_rate * t))
            phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
            phase += self._last_phase
            self._last_phase = phase[-1]
        else:
            phase = 2 * np.pi * self._compute_trace_phase(frames)

        phase *= self.direction

        outdata[:, 0] = np.sin(phase) * self.amp
        outdata[:, 1] = np.cos(phase) * self.amp

        self._apply_noise(outdata, frames)
        self.global_sample += frames

    def _on_start(self):
        if self.sweep_mode:
            print(f"◯ Circle sweep: {self.freq_min}-{self.freq_max} Hz "
                  f"(rate: {self.sweep_rate} Hz)")
        else:
            print(f"◯ Circle at {self.freq} Hz")
        dir_str = "↺ counter-clockwise" if self.direction < 0 else "↻ clockwise"
        print(f"  {dir_str}")
        print("  Press Ctrl+C to stop.")
