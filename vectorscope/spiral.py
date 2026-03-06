"""Real-time hypnotic spiral display for oscilloscope."""

import numpy as np
import time as _time

from .base import VectorScopePlayer


class SpiralPlayer(VectorScopePlayer):
    """Hypnotic rotating spiral - stare at it and get sleepy..."""

    def __init__(self, arms=3, turns=5, rot_freq=0.5, **kwargs):
        super().__init__(**kwargs)
        self.arms = arms
        self.turns = turns
        self.rot_freq = rot_freq
        self._path_pos = 0
        self._generate_spiral()

    def _generate_spiral(self):
        """Generate base spiral path (one frame, no rotation)."""
        # Ensure we use the full 'samples' budget for all arms combined
        points_per_arm = self.samples // self.arms
        total_samples = points_per_arm * self.arms

        all_points = []
        blanking = np.zeros(total_samples, dtype=bool)
        
        for arm in range(self.arms):
            # theta goes from 0 to turns*2pi
            theta = np.linspace(0, self.turns * 2 * np.pi, points_per_arm)
            # r goes from 0 to 0.95
            r = (theta / (self.turns * 2 * np.pi)) * 0.95

            arm_offset = arm * (2 * np.pi / self.arms)
            x = r * np.cos(theta + arm_offset)
            y = r * np.sin(theta + arm_offset)

            all_points.append(np.column_stack([x, y]))
            # Blank the jump from the end of the previous arm to the start of this one
            blanking[arm * points_per_arm] = True

        self._base_spiral = np.vstack(all_points).astype(np.float32)
        self._base_blanking = blanking

    def audio_callback(self, outdata, frames, time_info, status):
        """Fill from xy_data (supports animate-freq) then apply rotation."""
        t_start = _time.perf_counter()
        t_compute_start = _time.perf_counter()
        
        has_stats = hasattr(self, 'stats')
        if has_stats and self.stats['last_callback_end'] is not None:
            self.stats['wait_time'] += (t_start - self.stats['last_callback_end'])
            self.stats['wait_count'] += 1

        self._check_status(status)

        # 1. Slice raw static samples from base spiral
        xy_raw = np.empty((frames, 2), dtype=np.float32)
        blank_raw = np.empty(frames, dtype=bool)
        
        # Determine indices for this block manually (since we need both XY and Blanking)
        path_len = len(self._base_spiral)
        self._path_pos %= path_len
        out_idx = 0
        while out_idx < frames:
            chunk = min(frames - out_idx, path_len - self._path_pos)
            xy_raw[out_idx:out_idx+chunk] = self._base_spiral[self._path_pos:self._path_pos+chunk]
            blank_raw[out_idx:out_idx+chunk] = self._base_blanking[self._path_pos:self._path_pos+chunk]
            self._path_pos = (self._path_pos + chunk) % path_len
            out_idx += chunk

        # 2. Apply real-time rotation math
        t_samples = (self.global_sample + np.arange(frames)) / self.sample_rate
        angles = 2 * np.pi * self.rot_freq * t_samples
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        x, y = xy_raw[:, 0], xy_raw[:, 1]
        out_x = (x * cos_a - y * sin_a) * self.amp
        out_y = (x * sin_a + y * cos_a) * self.amp
        xy_rotated = np.column_stack([out_x, out_y]).astype(np.float32)

        # 3. Prepare signals (handles Z voltage mapping, delay, and Web packing)
        self._prepare_output(xy_rotated, blank_raw)
        
        # 4. Copy prepared data to outdata
        with self._lock:
            self._fill_buffer(outdata, frames)

        # 5. Post-processing
        self._apply_noise(outdata, frames)
        if self.channels >= 4:
            outdata[:, 3] = 0.0

        # Attribute stats
        self._increment_compute_stats(_time.perf_counter() - t_compute_start, self.arms, self.arms, self.samples)

        self.global_sample += frames

        if has_stats:
            tend = _time.perf_counter()
            self.stats['callback_time'] += (tend - t_start)
            self.stats['callback_count'] += 1
            self.stats['last_callback_end'] = tend

    def _on_start(self):
        print(f"🌀 Spiral: {self.arms} arms, {self.turns} turns, {self.rot_freq} Hz rotation")
        dir_str = "↺ counter-clockwise" if self.rot_freq < 0 else "↻ clockwise"
        print(f"  {dir_str}")
        print("  You are getting sleepy... Press Ctrl+C to wake up.")

    def _on_stop(self):
        print("\nWide awake!")
