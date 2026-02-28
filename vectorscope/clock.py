"""Real-time oscilloscope clock display."""

import numpy as np
from datetime import datetime

from .base import VectorScopePlayer
from .text import build_xy_from_text


def format_time(use_24h=False):
    """Return current time as '12:55pm' or '14:55' format."""
    now = datetime.now()
    if use_24h:
        return now.strftime("%H:%M")
    else:
        return now.strftime("%I:%M%p").lstrip("0").lower()


class ClockPlayer(VectorScopePlayer):
    """Real-time digital clock - updates each minute."""

    def __init__(self, use_24h=False, **kwargs):
        super().__init__(**kwargs)
        self.use_24h = use_24h
        self.current_time_str = None
        self._update_time()

    def _update_time(self):
        """Regenerate XY data if time changed."""
        time_str = format_time(self.use_24h)
        if time_str != self.current_time_str:
            self.current_time_str = time_str
            xy = build_xy_from_text(
                time_str,
                curve_pts=20,
                samples=self.samples,
                pen_lift_samples=0
            )
            self.xy_data = np.clip(xy * self.amp, -1.0, 1.0).astype(np.float32)
            self.position = 0
            print(f"  {time_str}")

    def audio_callback(self, outdata, frames, time, status):
        """Custom callback that checks for time updates."""
        self._check_status(status)

        self._update_time()
        self._fill_buffer(outdata, frames)
        self._apply_noise(outdata, frames)
        self.global_sample += frames

    def _on_start(self):
        fmt = "24h" if self.use_24h else "12h"
        print(f"🕐 Clock ({fmt} format)")
        print("  Press Ctrl+C to stop.")
