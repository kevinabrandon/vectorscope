"""Regular N-gon display with optional rotation."""

import numpy as np
from .base import VectorScopePlayer

POLYGON_NAMES = {
    3: 'Triangle',
    4: 'Square',
    5: 'Pentagon',
    6: 'Hexagon',
    7: 'Heptagon',
    8: 'Octagon',
}


class NgonPlayer(VectorScopePlayer):
    """
    Real-time regular N-gon with optional rotation.

    Traces the outline of an N-sided regular polygon. With rotation
    enabled, the shape spins smoothly.
    """

    def __init__(self, sides=4, rot_freq=0.0, **kwargs):
        super().__init__(**kwargs)
        if sides > 1024:
            raise ValueError(f"Max 1024 sides (got {sides}). Use 'circle' for round shapes.")
        self.sides = max(3, sides)
        self.rot_freq = rot_freq
        self._build_vertices()

    def _build_vertices(self):
        """Pre-compute the unit polygon vertices (closed loop)."""
        n = self.sides
        # N+1 vertices so the last == the first (closed path)
        angles = np.linspace(0, 2 * np.pi, n + 1)
        self._vx = np.sin(angles).astype(np.float32)
        self._vy = np.cos(angles).astype(np.float32)

    def audio_callback(self, outdata, frames, time, status):
        self._check_status(status)

        # Phase: 0 → 1 representing one full trip around the polygon
        phase = self._compute_trace_phase(frames) % 1.0

        # Which edge and how far along it
        edge_pos = phase * self.sides
        edge_idx = edge_pos.astype(int) % self.sides
        frac = (edge_pos - np.floor(edge_pos)).astype(np.float32)

        # Interpolate between consecutive vertices
        x = self._vx[edge_idx] * (1 - frac) + self._vx[edge_idx + 1] * frac
        y = self._vy[edge_idx] * (1 - frac) + self._vy[edge_idx + 1] * frac

        # Apply rotation if spinning
        if self.rot_freq != 0:
            t = (self.global_sample + np.arange(frames)) / self.sample_rate
            angles = 2 * np.pi * self.rot_freq * t
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
        name = POLYGON_NAMES.get(self.sides, f'{self.sides}-gon')
        print(f"{name} at {self.freq} Hz")
        if self.rot_freq != 0:
            dir_str = "counter-clockwise" if self.rot_freq < 0 else "clockwise"
            print(f"  Rotating {dir_str} at {abs(self.rot_freq)} Hz")
        print("  Press Ctrl+C to stop.")
