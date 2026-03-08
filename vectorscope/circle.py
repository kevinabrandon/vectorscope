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

    def audio_callback(self, outdata, frames, time_info, status):
        """Generate circle in real-time with optional frequency sweep."""
        import time as _time
        t_start = _time.perf_counter()
        t_compute_start = _time.perf_counter()
        
        has_stats = hasattr(self, 'stats')
        if has_stats and self.stats['last_callback_end'] is not None:
            self.stats['wait_time'] += (t_start - self.stats['last_callback_end'])
            self.stats['wait_count'] += 1

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

        xy = np.empty((frames, 2), dtype=np.float32)
        xy[:, 0] = np.sin(phase) * self.amp
        xy[:, 1] = np.cos(phase) * self.amp

        # Prepare signals and swap buffers
        self._prepare_output(xy)

        # Attribute stats (circle is one continuous curve, so vectors=1)
        effective_samples = int(self.sample_rate / abs(self.freq)) if self.freq != 0 else frames
        self._increment_compute_stats(_time.perf_counter() - t_compute_start, 1, 0, effective_samples)

        with self._lock:
            self._fill_buffer(outdata, frames)

        self._apply_noise(outdata, frames)

        # Zero spare channel
        if self.channels >= 4:
            outdata[:, 3] = 0.0

        self._push_web_output(outdata, frames)

        self.global_sample += frames

        if has_stats:
            tend = _time.perf_counter()
            self.stats['callback_time'] += (tend - t_start)
            self.stats['callback_count'] += 1
            self.stats['last_callback_end'] = tend

    def _on_start(self):
        if self.sweep_mode:
            print(f"◯ Circle sweep: {self.freq_min}-{self.freq_max} Hz "
                  f"(rate: {self.sweep_rate} Hz)")
        else:
            print(f"◯ Circle at {self.freq} Hz")
        dir_str = "↺ counter-clockwise" if self.direction < 0 else "↻ clockwise"
        print(f"  {dir_str}")
        print("  Press Ctrl+C to stop.")
