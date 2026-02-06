"""Real-time hypnotic spiral display for oscilloscope."""

import numpy as np

from .base import VectorScopePlayer


class SpiralPlayer(VectorScopePlayer):
    """Hypnotic rotating spiral - stare at it and get sleepy..."""

    def __init__(self, arms=3, turns=5, speed=0.5, direction=1, **kwargs):
        super().__init__(**kwargs)
        self.arms = arms
        self.turns = turns
        self.speed = speed
        self.direction = direction
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

    def audio_callback(self, outdata, frames, time, status):
        """Custom callback that applies rotation to the spiral."""
        if status:
            print(f"Audio status: {status}")

        spiral_len = len(self.base_spiral)

        # Get spiral points for this chunk (vectorized)
        indices = (self.global_sample + np.arange(frames)) % spiral_len
        xy = self.base_spiral[indices]
        x, y = xy[:, 0], xy[:, 1]

        # Rotation angles for each sample (vectorized)
        t_samples = (self.global_sample + np.arange(frames)) / self.sample_rate
        angles = self.direction * 2 * np.pi * self.speed * t_samples
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        # Apply rotation (vectorized)
        outdata[:, 0] = (x * cos_a - y * sin_a) * self.amp
        outdata[:, 1] = (x * sin_a + y * cos_a) * self.amp

        self._apply_noise(outdata, frames)
        self.global_sample += frames

    def _on_start(self):
        print(f"🌀 Spiral: {self.arms} arms, {self.turns} turns, {self.speed} Hz rotation")
        dir_str = "↺ counter-clockwise" if self.direction < 0 else "↻ clockwise"
        print(f"  {dir_str}")
        print("  You are getting sleepy... Press Ctrl+C to wake up.")

    def _on_stop(self):
        print("\nWide awake!")
