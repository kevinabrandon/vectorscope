"""Spirograph pattern generator."""

import numpy as np
import math
from .base import VectorScopePlayer

class SpirographPlayer(VectorScopePlayer):
    """
    Generates spirograph patterns.
    """

    def __init__(self, R=5, r=3, d=0.8, rot_freq=0.0,
                 animate_d_range=None, **kwargs):
        super().__init__(**kwargs)
        self.R = R
        self.r = r
        self.d_default = d
        self.rot_freq = rot_freq
        self.animate_d_range = animate_d_range
        
        if self.animate_d_range:
            self.d_min = self.animate_d_range[0]
            self.d_max = self.animate_d_range[1]
        else:
            self.d_min = self.d_default
            self.d_max = self.d_default

        self._update_params()

    def _update_params(self):
        # Calculate the number of revolutions needed to close the loop
        self.revolutions = self.r // math.gcd(self.R, self.r) if self.r != 0 else 1
        self.lcm = (self.R * self.r) // math.gcd(self.R, self.r) if self.r != 0 else self.R
        print(f"Spirograph R={self.R}, r={self.r}, d={self.d_default}, freq={self.freq}, rot_freq={self.rot_freq}")

    def audio_callback(self, outdata, frames, time, status):
        self._check_status(status)

        R, r, rot_freq, revolutions = self.R, self.r, self.rot_freq, self.revolutions
        d_min, d_max = self.d_min, self.d_max

        t_global = (self.global_sample + np.arange(frames)) / self.sample_rate

        if r == 0: # Avoid division by zero
            outdata.fill(0)
            self.global_sample += frames
            return

        trace_phase = self._compute_trace_phase(frames)
        t = trace_phase * 2 * np.pi * revolutions

        if self.animate_d_range:
            lfo = 0.5 - 0.5 * np.cos(2 * np.pi * t_global / self.fade_period)
            d = d_min + lfo * (d_max - d_min)
        else:
            d = self.d_default

        x = (R - r) * np.cos(t) + d * np.cos((R - r) / r * t)
        y = (R - r) * np.sin(t) - d * np.sin((R - r) / r * t)
        
        # Normalize
        norm_factor = np.abs(R - r) + np.abs(d)
        if isinstance(norm_factor, np.ndarray):
            norm_factor[norm_factor <= 1e-8] = 1
        elif norm_factor <= 1e-8:
            norm_factor = 1

        x /= norm_factor
        y /= norm_factor

        # Apply rotation if spinning
        if rot_freq != 0:
            angles = 2 * np.pi * rot_freq * t_global
            cos_a = np.cos(angles)
            sin_a = np.sin(angles)
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            x, y = rx, ry

        outdata[:, 0] = x * self.amp
        outdata[:, 1] = y * self.amp

        self._apply_noise(outdata, frames)
        self.global_sample += frames

    def _on_start(self):
        self._update_params()
        if self.rot_freq != 0:
            dir_str = "counter-clockwise" if self.rot_freq < 0 else "clockwise"
            print(f"  Rotating {dir_str} at {abs(self.rot_freq)} Hz")
        if self.animate_d_range:
            print(f"  Animating 'd' from {self.d_min} to {self.d_max} over {self.fade_period}s")
        print("  Press Ctrl+C to stop.")
