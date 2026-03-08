"""Asteroids game player for vectorscope."""

import time
import sys
import termios
import tty
import select
import logging
import threading
import numpy as np
from HersheyFonts import HersheyFonts

from .base import VectorScopePlayer
from .polyline import path_to_xy
from .asteroids.asteroids import Asteroids


class PolylineBuilder:
    """Builds a single flat XY path with embedded blanked pen-lift segments.

    move_to() starts a new visible shape.  If a previous shape exists, a
    single blanked hop point is inserted so that arc-length resampling in
    path_to_xy() automatically allocates samples proportional to the travel
    distance — short hops get few blanked samples (beam barely moves, Z-blank
    leakage invisible), long hops get many (beam moves slowly, leakage short).

    Blanking convention (trailing-edge): blanking[i] = True means the segment
    from point i-1 to point i is blanked.
    """

    def __init__(self, hf=None):
        self._pts = []
        self._blk = []
        self._itn = []
        self._cur_intensity = 1.0
        self.hf = hf
        if self.hf is None:
            self.hf = HersheyFonts()
            self.hf.load_default_font("futural")
            self.hf.normalize_rendering(1.0)

    def move_to(self, x, y, intensity=1.0):
        self._cur_intensity = intensity
        if self._pts:
            # Blanked hop: insert destination as a blanked arrival point.
            # The segment from the previous visible point to here is blanked.
            self._pts.append([x, y])
            self._blk.append(True)
            self._itn.append(0.0)
        else:
            # Very first point — no preceding segment, start visible.
            self._pts.append([x, y])
            self._blk.append(False)
            self._itn.append(intensity)

    def line_to(self, x, y):
        self._pts.append([x, y])
        self._blk.append(False)
        self._itn.append(self._cur_intensity)

    def text_at(self, x, y, text, size=1, rot=0, intensity=1.0):
        if not text:
            return
        scale = size * 25.0
        strokes = self.hf.strokes_for_text(text)
        cos_val = np.cos(np.radians(rot))
        sin_val = np.sin(np.radians(rot))
        for stroke in strokes:
            pts = np.array(stroke, dtype=np.float64)
            if len(pts) == 0:
                continue
            pts *= scale
            if rot != 0:
                rx = pts[:, 0] * cos_val - pts[:, 1] * sin_val
                ry = pts[:, 0] * sin_val + pts[:, 1] * cos_val
                pts[:, 0] = rx
                pts[:, 1] = ry
            pts[:, 0] += x
            pts[:, 1] += y
            self.move_to(pts[0, 0], pts[0, 1], intensity=intensity)
            for pt in pts[1:]:
                self.line_to(pt[0], pt[1])

    def text_at_centered(self, x, y, text, size=1, rot=0, intensity=1.0):
        if not text:
            return
        strokes = self.hf.strokes_for_text(text)
        if not strokes:
            return
        all_x = [p[0] for s in strokes for p in s]
        if not all_x:
            return
        width = max(all_x) - min(all_x)
        scale = size * 25.0
        offset_x = (width * scale) / 2.0
        self.text_at(x - offset_x, y, text, size=size, rot=rot, intensity=intensity)

    def get_path(self):
        if not self._pts:
            return (np.zeros((0, 2), dtype=np.float64),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=np.float32))
        return (np.array(self._pts, dtype=np.float64),
                np.array(self._blk, dtype=bool),
                np.array(self._itn, dtype=np.float32))

    def clear(self):
        self._pts = []
        self._blk = []
        self._itn = []
        self._cur_intensity = 1.0


class AsteroidsPlayer(VectorScopePlayer):
    def __init__(self, max_vectors=800, aspect_x=0.75, penlift=20,
                 dynamic_refresh=False, optimize_order=False,
                 initial_rocks=3, friendly_fire=False,
                 ship_bullet_speed=None, ship_bullet_ttl=None, ship_max_bullets=None,
                 saucer_bullet_speed=None, saucer_bullet_ttl=None, saucer_max_bullets=None,
                 difficulty=None, max_hop_speed=0.02, **kwargs):
        # In dynamic refresh mode, we use kwargs['freq'] as the "target"
        # for a screen of average complexity.
        super().__init__(**kwargs)
        # Treat 0 as unlimited
        self.max_vectors = max_vectors if max_vectors > 0 else None
        self.aspect_x = aspect_x
        self.penlift_samples = penlift
        self.max_hop_speed = max_hop_speed
        self.dynamic_refresh = dynamic_refresh
        self.optimize_order = optimize_order
        self.last_t = time.monotonic()
        self.last_thrust_time = 0.0
        
        # Calculate a "units per sample" constant based on the initial freq.
        # A circle of radius 1 (circumference ~6.28) at 60Hz and 96k sr
        # takes ~1600 samples. Speed = 6.28 / 1600 = 0.0039.
        # We'll use a slightly different heuristic based on the user's freq.
        target_samples = self.samples if self.samples > 0 else 1600
        # Average "screen units" complexity (arbitrary units)
        # 10.0 units is about a circle + some text.
        self.target_speed = 10.0 / target_samples
        
        # Initialize the game
        self.game = Asteroids(maxc=2048, aspect_x=aspect_x, num_rocks=initial_rocks,
                              friendly_fire=friendly_fire,
                              ship_bullet_speed=ship_bullet_speed,
                              ship_bullet_ttl=ship_bullet_ttl,
                              ship_max_bullets=ship_max_bullets,
                              saucer_bullet_speed=saucer_bullet_speed,
                              saucer_bullet_ttl=saucer_bullet_ttl,
                              saucer_max_bullets=saucer_max_bullets)
        if difficulty:
            self.game.start_game(difficulty)
        else:
            self.game.attract_mode = True
        # Hershey font support
        self.hf = HersheyFonts()
        self.hf.load_default_font("futural")
        self.hf.normalize_rendering(1.0)

        self.builder = PolylineBuilder(self.hf)

        # Background update thread (used in interactive mode)
        self._update_stop_event = threading.Event()
        self._update_thread = None

        # TTY state saved by _setup_input, restored by _teardown_input
        self._tty_fd = None
        self._tty_old_settings = None

        # Request a fixed blocksize for consistent game timing
        self._stream_blocksize = 2048

        # Initial frame
        self._update_frame()

    def _update_frame(self):
        t0 = time.perf_counter()
        now = time.monotonic()
        dt = now - self.last_t
        self.last_t = now
        
        # Auto-off thruster if no heartbeat for 100ms
        if self.game.ship and self.game.ship.thrustJet.accelerating:
            if now - self.last_thrust_time > 0.1:
                self.game.ship.decreaseThrust()

        # Limit dt to avoid massive jumps
        if dt > 0.1:
            dt = 0.1
            
        self.builder.clear()
        self.game.step(dt, self.builder, self.max_vectors)

        xy, blanking, intensity = self.builder.get_path()

        if len(xy) == 0:
            self._prepare_output(
                np.zeros((self.samples, 2), dtype=np.float32),
                np.zeros(self.samples, dtype=bool),
                np.zeros(self.samples, dtype=np.float32),
            )
            self._increment_compute_stats(time.perf_counter() - t0, 0, 0, 0)
            return

        # Normalize: game coords [0, 2048] → [-0.9, 0.9] in one vectorized op
        xy = (xy / 1024.0 - 1.0) * 0.9

        if self.dynamic_refresh:
            diffs = np.diff(xy, axis=0)
            seglen = np.sqrt((diffs ** 2).sum(axis=1))
            hop_mask = blanking[1:]  # trailing-edge: segment i blanked iff blanking[i+1]=True
            L_visible = float(seglen[~hop_mask].sum())

            # Per-hop sample allocation: distance-proportional with floor
            hop_seglens = seglen[hop_mask]
            hop_alloc = np.maximum(
                self.penlift_samples,
                np.ceil(hop_seglens / self.max_hop_speed).astype(int)
            )
            # Estimate closing loop hop that path_to_xy will add
            closing_dist = float(np.sqrt(((xy[-1] - xy[0]) ** 2).sum()))
            closing_alloc = max(self.penlift_samples,
                                int(np.ceil(closing_dist / self.max_hop_speed)))
            n_hop_total = int(hop_alloc.sum()) + closing_alloc

            current_samples = max(200, int(np.ceil(L_visible / self.target_speed)) + n_hop_total + 50)
        else:
            current_samples = self.samples

        new_xy, new_blanking, new_intensities = path_to_xy(
            xy, blanking, intensity, current_samples, amp=self.amp,
            min_hop_samples=self.penlift_samples,
            max_hop_speed=self.max_hop_speed,
        )

        self._prepare_output(new_xy, new_blanking, new_intensities)

        n_lifts = int(np.sum(blanking))
        self._increment_compute_stats(time.perf_counter() - t0, n_lifts + 1, n_lifts, current_samples)

    def _start_background(self):
        """Start the game update loop in a background thread."""
        self._update_stop_event.clear()
        def _loop():
            while not self._update_stop_event.is_set():
                self._update_frame()
                self._update_stop_event.wait(0.001)
        self._update_thread = threading.Thread(target=_loop, daemon=True)
        self._update_thread.start()

    def _stop_background(self):
        """Stop the background game update loop."""
        self._update_stop_event.set()
        if self._update_thread is not None:
            self._update_thread.join(timeout=2.0)
            self._update_thread = None

    def _on_start(self):
        print("Asteroids on Oscilloscope!")
        print("  1 Easy, 2 Medium, 3 Hard")
        print("  ? for Help, Ctrl+C to quit")

    def _setup_input(self):
        """Set up raw TTY and start the keyboard input thread."""
        if not sys.stdin.isatty():
            return
        self._tty_fd = sys.stdin.fileno()
        self._tty_old_settings = termios.tcgetattr(self._tty_fd)
        tty.setraw(self._tty_fd)

        def input_thread_func():
            try:
                while True:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch == '\x03':  # Ctrl+C
                            import os, signal
                            os.kill(os.getpid(), signal.SIGINT)
                            break
                        if ch == '\x1b':
                            ch2 = sys.stdin.read(1)
                            if ch2 == '[':
                                ch3 = sys.stdin.read(1)
                                if ch3 == 'A': ch = 'w'
                                elif ch3 == 'B': ch = 's'
                                elif ch3 == 'C': ch = 'd'
                                elif ch3 == 'D': ch = 'a'
                        with self._lock:
                            self._handle_input(ch)
            except (EOFError, OSError):
                pass

        threading.Thread(target=input_thread_func, daemon=True).start()

    def _teardown_input(self):
        """Restore TTY to its original settings."""
        if self._tty_old_settings is not None:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._tty_old_settings)
            self._tty_old_settings = None

    def _handle_input(self, ch):
        # Help toggle — works in any state
        if ch == '?':
            self.game.show_help = not self.game.show_help
            return

        # Difficulty start — works when not actively playing
        if ch in ('1', '2', '3'):
            difficulties = {'1': 'easy', '2': 'medium', '3': 'hard'}
            self.game.start_game(difficulties[ch])
            return

        if ch in ('c', 'C') and self.game.gameState == "gameover":
            self.game.continue_game()
            return

        if ch == '0':
            self.game.show_help = False
            self.game.attract_mode = True
            self.game.reset_attract()
            return

        ship = self.game.ship
        if not ship or self.game.gameState != "playing" or self.game.attract_mode:
            return

        step = 1.0 # arbitrary speed for single keypress

        if ch in ('w', 'W'): # Up
            ship.increaseThrust(step * 5)
            ship.thrustJet.accelerating = True
            self.last_thrust_time = time.monotonic()
        elif ch in ('a', 'A'): # Left
            ship.rotateRight(step * 5)
        elif ch in ('d', 'D'): # Right
            ship.rotateLeft(step * 5)
        elif ch in ('s', 'S'): # Down / Fire?
            if ship.fireBullet():
                self.game._log("[bullet]", "Player Ship fired bullet.", level=logging.DEBUG)
        elif ch == ' ':
            if ship.fireBullet():
                self.game._log("[bullet]", "Player Ship fired bullet.", level=logging.DEBUG)
        elif ch in ('h', 'H'):
            ship.hyperspace_exit_pos = self.game.find_safe_position()
            ship.enterHyperSpace()
