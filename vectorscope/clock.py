"""Real-time oscilloscope clock display."""

import threading
from datetime import datetime

from .base import VectorScopePlayer
from .hershey_player import is_hershey_font, build_xy_from_hershey
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

    def __init__(self, use_24h=False, font="futural", penlift=20,
                 curve_pts=30, **kwargs):
        super().__init__(**kwargs)
        self.use_24h = use_24h
        self.penlift_samples = penlift
        self.curve_pts = curve_pts
        self.current_time_str = None
        self.font_name = font

        self._is_hershey = is_hershey_font(font)
        if self._is_hershey:
            from HersheyFonts import HersheyFonts
            self.hf = HersheyFonts()
            self.hf.load_default_font(font)
            self.hf.normalize_rendering(1.0)

        # Background update thread (used in interactive mode)
        self._update_stop_event = threading.Event()
        self._update_thread = None

        self._update_time()

    def _start_background(self):
        """Start the clock update loop in a background thread."""
        self._update_stop_event.clear()
        def _loop():
            while not self._update_stop_event.is_set():
                self._update_time()
                self._update_stop_event.wait(0.5)
        self._update_thread = threading.Thread(target=_loop, daemon=True)
        self._update_thread.start()

    def _stop_background(self):
        """Stop the background clock update loop."""
        self._update_stop_event.set()
        if self._update_thread is not None:
            self._update_thread.join(timeout=2.0)
            self._update_thread = None

    def _update_time(self):
        """Regenerate XY data if time changed."""
        time_str = format_time(self.use_24h)
        if time_str != self.current_time_str:
            self.current_time_str = time_str
            if self._is_hershey:
                xy, blanking, intensity, n_lifts, n_samples = build_xy_from_hershey(
                    self.hf, time_str, self.samples, self.amp,
                    self.penlift_samples
                )
            else:
                xy, blanking, intensity, n_lifts, n_samples = build_xy_from_text(
                    time_str,
                    font_family=self.font_name,
                    curve_pts=self.curve_pts,
                    samples=self.samples,
                    pen_lift_samples=self.pen_lift_samples,
                    amp=self.amp
                )
            self._prepare_output(xy, blanking, intensity)
            self.position = 0
            print(f"  {time_str}")

    def _on_start(self):
        fmt = "24h" if self.use_24h else "12h"
        print(f"Clock ({fmt} format, font: {self.font_name})")
        print("  Press Ctrl+C to stop.")
