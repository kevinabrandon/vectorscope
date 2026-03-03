"""Asteroids game player for vectorscope."""

import time
import sys
import termios
import tty
import select
import threading
import numpy as np
from HersheyFonts import HersheyFonts

from .base import VectorScopePlayer, NullOutputStream
from .polyline import polylines_to_xy
from .asteroids.asteroids import Asteroids


class PolylineBuilder:
    def __init__(self, hf=None):
        self.polylines = []
        self.current_polyline = []
        self.hf = hf
        if self.hf is None:
            self.hf = HersheyFonts()
            self.hf.load_default_font("futural")
            self.hf.normalize_rendering(1.0)

    def move_to(self, x, y):
        if len(self.current_polyline) > 1:
            self.polylines.append(np.array(self.current_polyline, dtype=np.float64))
        self.current_polyline = [[x, y]]

    def line_to(self, x, y):
        self.current_polyline.append([x, y])

    def text_at(self, x, y, text, size=1, rot=0):
        if not text:
            return
        if len(self.current_polyline) > 1:
            self.polylines.append(np.array(self.current_polyline, dtype=np.float64))
        self.current_polyline = []

        # HersheyFonts rendering
        # We need to scale and position the text
        # size=1 in David's code seems to be about 20-30 units high in a 2048 space.
        # HersheyFonts normalize_rendering(1.0) makes it roughly 1.0 units high.
        scale = size * 25.0
        
        # hf.strokes_for_text returns a list of strokes
        # Each stroke is a list of (x, y)
        strokes = self.hf.strokes_for_text(text)
        
        cos_val = np.cos(np.radians(rot))
        sin_val = np.sin(np.radians(rot))

        for stroke in strokes:
            pts = np.array(stroke, dtype=np.float64)
            # Scale
            pts *= scale
            
            # Rotate if needed
            if rot != 0:
                # Using the same rotation logic as VectorSprite if possible, 
                # but standard CCW rotation is usually better for fonts.
                # However, David's code uses rot=0 always.
                rx = pts[:, 0] * cos_val - pts[:, 1] * sin_val
                ry = pts[:, 0] * sin_val + pts[:, 1] * cos_val
                pts[:, 0] = rx
                pts[:, 1] = ry

            # Translate to (x, y)
            pts[:, 0] += x
            pts[:, 1] += y
            self.polylines.append(pts)

    def get_polylines(self):
        if len(self.current_polyline) > 1:
            self.polylines.append(np.array(self.current_polyline, dtype=np.float64))
            self.current_polyline = []
        return self.polylines


class AsteroidsPlayer(VectorScopePlayer):
    def __init__(self, max_vectors=800, aspect_x=0.75, penlift=20, 
                 dynamic_refresh=False, optimize_order=True, 
                 initial_rocks=3, **kwargs):
        # In dynamic refresh mode, we use kwargs['freq'] as the "target"
        # for a screen of average complexity.
        super().__init__(**kwargs)
        # Treat 0 as unlimited
        self.max_vectors = max_vectors if max_vectors > 0 else None
        self.aspect_x = aspect_x
        self.penlift_samples = penlift
        self.dynamic_refresh = dynamic_refresh
        self.optimize_order = optimize_order
        self.last_t = time.monotonic()
        
        # Calculate a "units per sample" constant based on the initial freq.
        # A circle of radius 1 (circumference ~6.28) at 60Hz and 96k sr
        # takes ~1600 samples. Speed = 6.28 / 1600 = 0.0039.
        # We'll use a slightly different heuristic based on the user's freq.
        target_samples = self.samples if self.samples > 0 else 1600
        # Average "screen units" complexity (arbitrary units)
        # 10.0 units is about a circle + some text.
        self.target_speed = 10.0 / target_samples
        
        # Initialize the game
        self.game = Asteroids(maxc=2048, aspect_x=aspect_x, num_rocks=initial_rocks)
        self.game.attract_mode = True # Start in attract mode
        
        self.hf = HersheyFonts()
        self.hf.load_default_font("futural")
        self.hf.normalize_rendering(1.0)
        
        self.builder = PolylineBuilder(self.hf)
        
        # Threading support
        self._lock = threading.Lock()
        
        # Initial frame
        self._update_frame()

    def _update_frame(self):
        now = time.monotonic()
        dt = now - self.last_t
        self.last_t = now
        
        # Limit dt to avoid massive jumps
        if dt > 0.1:
            dt = 0.1
            
        self.builder.polylines = []
        self.game.step(dt, self.builder, self.max_vectors)
        
        polys = self.builder.get_polylines()
        
        # Fixed-space normalization
        fixed_polys = []
        for p in polys:
            fixed_polys.append((p / 1024.0 - 1.0) * 0.9)
        polys = fixed_polys

        # At high sample rates, we need more points per segment to make them visible
        # and to prevent truncation during rounding.
        min_pts = 2
        if self.sample_rate >= 192000:
            min_pts = 6
        elif self.sample_rate >= 96000:
            min_pts = 4

        # Determine how many samples to use for this frame.
        if self.dynamic_refresh:
            # Sum the distance (arc length), accounting for min samples per segment
            total_samples = 0
            for p in polys:
                if len(p) < 2: continue
                d = np.diff(p, axis=0)
                L = np.sqrt((d**2).sum(axis=1)).sum()
                # Ensure each segment gets enough samples at its speed,
                # plus its required minimum for visibility.
                total_samples += max(min_pts, L / self.target_speed)
            
            # Add pen lifts and a safety margin (extra 50 samples)
            current_samples = int(np.ceil(total_samples)) + self.penlift_samples * len(polys) + 50
            current_samples = max(200, current_samples)
        else:
            current_samples = self.samples

        # Convert to XY
        new_xy, new_blanking = polylines_to_xy(
            polys, current_samples, amp=self.amp,
            pen_lift_samples=self.penlift_samples,
            margin=0.9,
            optimize_order=self.optimize_order,
            min_points_per_segment=min_pts,
            normalize=False
        )
        
        # Thread-safe swap
        with self._lock:
            self.xy_data = new_xy
            self.xy_blanking = new_blanking

    def audio_callback(self, outdata, frames, time, status):
        self._check_status(status)
        
        # audio_callback must return IMMEDIATELY. 
        # We only copy existing data here.
        with self._lock:
            self._fill_buffer(outdata, frames)
            
        self._apply_noise(outdata, frames)
        self._apply_z_channel(outdata, frames)
        self.global_sample += frames

    def _on_start(self):
        print("Asteroids on Oscilloscope!")
        print("  W, A, S, D or Arrows to move/fire")
        print("  Space to fire, H for Hyperspace")
        print("  1 to Start Game, 0 for Attract Mode")
        print("  Press Ctrl+C to stop.")

    def run(self):
        """Override run to handle keyboard input."""
        import sounddevice as sd
        
        self._stream_start_time = time.monotonic()
        if self.device == 'demo':
            stream = NullOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=self.audio_callback,
                blocksize=2048,
            )
        else:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=self.audio_callback,
                device=self.device,
                latency='high',
                blocksize=2048,
            )
        stream.start()
        self._on_start()
        
        # Setup TTY for non-blocking input
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setraw(fd)
        else:
            old_settings = None

        try:
            while True:
                # Update frame logic here (NOT in callback)
                self._update_frame()

                if old_settings:
                    # Non-blocking check for input
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.005) # Lower timeout for more responsive game
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch == '\x03': # Ctrl+C
                            break
                        self._handle_input(ch)
                else:
                    time.sleep(0.016) # ~60 FPS update rate
        except KeyboardInterrupt:
            pass
        finally:
            if old_settings:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            stream.stop()
            stream.close()
            self._on_stop()

    def _handle_input(self, ch):
        # Handle arrow keys (escape sequences)
        if ch == '\x1b':
            # Potentially an escape sequence
            rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
            if rlist:
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                    if rlist:
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A': # Up
                            ch = 'w'
                        elif ch3 == 'B': # Down
                            ch = 's'
                        elif ch3 == 'C': # Right
                            ch = 'd'
                        elif ch3 == 'D': # Left
                            ch = 'a'

        ship = self.game.ship
        if not ship:
            if ch == '1':
                self.game.attract_mode = False
                self.game.reset_attract()
                self.game.lives = 3
            return

        step = 1.0 # arbitrary speed for single keypress
        
        if ch in ('w', 'W'): # Up
            ship.increaseThrust(step * 5)
            ship.thrustJet.accelerating = True
        elif ch in ('a', 'A'): # Left
            ship.rotateRight(step * 5)
        elif ch in ('d', 'D'): # Right
            ship.rotateLeft(step * 5)
        elif ch in ('s', 'S'): # Down / Fire?
            ship.fireBullet()
        elif ch == ' ':
            ship.fireBullet()
        elif ch in ('h', 'H'):
            ship.enterHyperSpace()
        elif ch == '1':
            if self.game.attract_mode:
                self.game.attract_mode = False
                self.game.reset_attract()
                self.game.lives = 3
        elif ch == '0':
            self.game.attract_mode = True
            self.game.reset_attract()
