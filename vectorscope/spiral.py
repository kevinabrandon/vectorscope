"""Real-time hypnotic spiral display for oscilloscope."""

import numpy as np

from .base import VectorScopePlayer


class SpiralPlayer(VectorScopePlayer):
    """Hypnotic rotating spiral - stare at it and get sleepy..."""

    def __init__(self, arms=3, turns=5, rot_freq=0.5, **kwargs):
        super().__init__(**kwargs)
        self.arms = arms
        self.turns = turns
        self.rot_freq = rot_freq
        self._generate_spiral()

    def _generate_spiral(self):
        """Generate base spiral path (one frame, no rotation)."""
        points_per_arm = self.samples // self.arms

        all_points = []
        for arm in range(self.arms):
            theta = np.linspace(0, self.turns * 2 * np.pi, points_per_arm)
            r = theta / (self.turns * 2 * np.pi) * 0.95

            arm_offset = arm * (2 * np.pi / self.arms)
            x = r * np.cos(theta + arm_offset)
            y = r * np.sin(theta + arm_offset)

            all_points.append(np.column_stack([x, y]))

        self.base_spiral = np.vstack(all_points).astype(np.float32)
        self.xy_data = self.base_spiral

    def audio_callback(self, outdata, frames, time, status):
        """Fill from xy_data (supports animate-freq) then apply rotation."""
        self._check_status(status)

        self._fill_buffer(outdata, frames)

        # Rotation angles for each sample
        t_samples = (self.global_sample + np.arange(frames)) / self.sample_rate
        angles = 2 * np.pi * self.rot_freq * t_samples
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        x, y = outdata[:, 0].copy(), outdata[:, 1].copy()
        outdata[:, 0] = (x * cos_a - y * sin_a) * self.amp
        outdata[:, 1] = (x * sin_a + y * cos_a) * self.amp

        self._apply_noise(outdata, frames)
        self.global_sample += frames

    def _on_start(self):
        print(f"🌀 Spiral: {self.arms} arms, {self.turns} turns, {self.rot_freq} Hz rotation")
        dir_str = "↺ counter-clockwise" if self.rot_freq < 0 else "↻ clockwise"
        print(f"  {dir_str}")
        print("  You are getting sleepy... Press Ctrl+C to wake up.")

    def _on_stop(self):
        print("\nWide awake!")
