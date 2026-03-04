"""Real-time oscilloscope clock display."""

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

        self._update_time()

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

    def run(self):
        """Start the audio stream, checking for time updates in main loop."""
        import sounddevice as sd
        import time
        
        self._stream_start_time = time.monotonic()
        self._last_status_time = 0.0

        # Start web server if requested
        if self._web_port is not None:
            from .web import VectorscopeWebServer
            self._web_server = VectorscopeWebServer(self._web_port)
            self._web_server.set_z_amp(self.z_amp)
            self._web_server.set_web_scale_factor(self._web_scale_factor)
            self._web_server.start()
            self._web_server.push_metadata({
                'command': 'clock',
                'channels': self.channels,
            })

        # Use standard optimized callback
        callback = self.audio_callback

        if self.device == 'demo':
            stream = NullOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=callback,
            )
        else:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=callback,
                device=self.device,
                latency='high',
            )
        stream.start()
        self._on_start()
        try:
            while True:
                # Update time if needed (once per minute)
                self._update_time()

                # Push data to web server from main thread
                if self._web_server is not None:
                    web_data_to_push = None
                    with self._lock:
                        if self._web_data is not None:
                            web_data_to_push = self._web_data
                            self._web_data = None
                    
                    if web_data_to_push is not None:
                        self._web_server.push_frame(web_data_to_push)

                # Periodically log performance stats
                self._check_perf_log()

                if self.device == 'demo':
                    stream.sleep(100)
                else:
                    sd.sleep(100)
        except KeyboardInterrupt:
            pass
        finally:
            self.perf_log.close()
            stream.stop()
            stream.close()
            if self._web_server:
                self._web_server.stop()
            self._on_stop()

    def _on_start(self):
        fmt = "24h" if self.use_24h else "12h"
        print(f"Clock ({fmt} format, font: {self.font_name})")
        print("  Press Ctrl+C to stop.")
