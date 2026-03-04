"""Animated 3D wireframe sinc surface: sin(r)/r with traveling wave."""

import numpy as np
import time as _time

from .base import VectorScopePlayer


class SincPlayer(VectorScopePlayer):
    """Animated wireframe sinc surface rendered as isometric grid lines.

    The path stores one sample per grid intersection. The audio callback
    computes z at each vertex, projects to 2D, then interpolates the
    projected positions — giving straight lines between vertices rather
    than curves following the sinc surface.
    """

    def __init__(self, cells=8, cycles=2, speed=0.5,
                 zscale=0.4, pen_lift=4,
                 elevation=30, azimuth=45, rot_freq=0, **kwargs):
        super().__init__(**kwargs)
        self.cells = cells
        self.cycles = cycles
        self.speed = speed
        self.zscale = zscale
        self.pen_lift = pen_lift
        self.elevation = elevation
        self.azimuth_deg = azimuth
        self.rot_freq = rot_freq
        el_rad = np.radians(elevation)
        self._cos_el = np.cos(el_rad)
        self._sin_el = np.sin(el_rad)
        self._azimuth_rad = np.radians(azimuth)
        self._build_path()

    def _build_path(self):
        """Build grid vertex path with one sample per intersection.

        Generates zigzag row and column polylines over a (cells+1)^2 grid.
        Between polylines, pen_lift blanked samples bridge the gap.
        """
        n = self.cells + 1
        half = self.cycles * 2 * np.pi
        axis = np.linspace(-half, half, n, dtype=np.float64)

        gx_grid, gy_grid = np.meshgrid(axis, axis)  # [n, n]
        r_grid = np.hypot(gx_grid, gy_grid)

        # Build zigzag polylines: rows then columns
        polylines = []

        for j in range(n):
            row = np.column_stack([gx_grid[j, :], gy_grid[j, :], r_grid[j, :]])
            if j % 2 == 1:
                row = row[::-1]
            polylines.append(row)

        for k, i in enumerate(range(n - 1, -1, -1)):
            col = np.column_stack([gx_grid[:, i], gy_grid[:, i], r_grid[:, i]])
            if k % 2 == 0:
                col = col[::-1]
            polylines.append(col)

        # Assemble: vertex samples + pen-lift samples between polylines
        path_points = []
        blanking_flags = []

        for k, poly in enumerate(polylines):
            if k > 0 and self.pen_lift > 0:
                prev_end = polylines[k - 1][-1]
                cur_start = poly[0]
                # Intermediate blanked points (exclude endpoints already in polylines)
                pts = np.linspace(prev_end, cur_start,
                                  self.pen_lift + 2)[1:-1]
                path_points.append(pts)
                blanking_flags.append(np.full(len(pts), True, dtype=bool))

            path_points.append(poly)
            blanking_flags.append(np.full(len(poly), False, dtype=bool))

        # Wrap-around pen lift
        if self.pen_lift > 0:
            prev_end = polylines[-1][-1]
            cur_start = polylines[0][0]
            pts = np.linspace(prev_end, cur_start,
                              self.pen_lift + 2)[1:-1]
            if len(pts) > 0:
                path_points.append(pts)
                blanking_flags.append(np.full(len(pts), True, dtype=bool))

        self.base_path = np.vstack(path_points).astype(np.float64)
        self.base_blanking = np.concatenate(blanking_flags)
        
        # Performance metadata
        self._n_draw_vectors = len(polylines)
        self._n_lifts = len(polylines) # Between each zig-zag segment plus wrap

        # Precompute safe_r for audio_callback
        r_v = self.base_path[:, 2]
        self._r_small = np.abs(r_v) < 1e-8
        self._safe_r = np.where(self._r_small, 1.0, r_v)

        # Cache grid coordinates for projection
        self._gx = self.base_path[:, 0]
        self._gy = self.base_path[:, 1]

        # Projection scale from worst-case bounds (any azimuth, z in [-1, 1])
        self._max_r_xy = np.max(np.hypot(self._gx, self._gy))
        max_px = self._max_r_xy * self._cos_el
        max_py = self._max_r_xy * self._sin_el + self.zscale
        max_abs = max(max_px, max_py, 1e-8)
        self._proj_scale = 0.95 / max_abs

    def audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback. Minimal work mode."""
        t_start = _time.perf_counter()
        t_compute_start = _time.perf_counter()
        has_stats = hasattr(self, 'stats')
        if has_stats and self.stats['last_callback_end'] is not None:
            self.stats['wait_time'] += (t_start - self.stats['last_callback_end'])
            self.stats['wait_count'] += 1

        self._check_status(status)

        path_len = len(self.base_path)

        # Phase-based indexing with linear interpolation
        phase = self._compute_trace_phase(frames) % 1.0
        frac_idx = phase * path_len
        idx0 = frac_idx.astype(int) % path_len
        idx1 = (idx0 + 1) % path_len
        frac = (frac_idx - np.floor(frac_idx)).astype(np.float64)

        # Per-cycle locked time
        t = (self.global_sample + np.arange(frames, dtype=np.float64)) / self.sample_rate
        t_cycle = np.floor(self.freq * t) / self.freq

        # Compute z and projection at all vertices for each unique t_cycle
        # (usually 1, at most 2 at cycle boundaries)
        unique_t, t_inv = np.unique(t_cycle, return_inverse=True)

        px_table = np.empty((len(unique_t), path_len), dtype=np.float64)
        py_table = np.empty((len(unique_t), path_len), dtype=np.float64)
        depth_table = (np.empty((len(unique_t), path_len), dtype=np.float64)
                       if self.z_enabled else None)

        r_v = self.base_path[:, 2]
        gx, gy = self._gx, self._gy
        cos_el, sin_el = self._cos_el, self._sin_el
        for ti, tc in enumerate(unique_t):
            wt = self.speed * 2 * np.pi * tc
            z = np.where(self._r_small, np.cos(wt),
                         np.sin(r_v - wt) / self._safe_r)
            # Azimuth rotation
            az = self._azimuth_rad + 2 * np.pi * self.rot_freq * tc
            cos_az = np.cos(az)
            sin_az = np.sin(az)
            u = gx * cos_az - gy * sin_az
            v = gx * sin_az + gy * cos_az
            px_table[ti] = u * cos_el
            py_table[ti] = v * sin_el - z * self.zscale
            if depth_table is not None:
                depth_table[ti] = v

        # Interpolate projected positions (straight lines between vertices)
        px0 = px_table[t_inv, idx0]
        px1 = px_table[t_inv, idx1]
        out_x = px0 * (1 - frac) + px1 * frac

        py0 = py_table[t_inv, idx0]
        py1 = py_table[t_inv, idx1]
        out_y = py0 * (1 - frac) + py1 * frac

        xy = np.empty((frames, 2), dtype=np.float32)
        xy[:, 0] = (out_x * self._proj_scale * self.amp).astype(np.float32)
        xy[:, 1] = (out_y * self._proj_scale * self.amp).astype(np.float32)

        # Z-channel: depth shading + blanking
        blanking = None
        intensity = None
        if self.z_enabled:
            blanking = self.base_blanking[idx0]
            # Depth shading: v (into-screen coord) → intensity (near=bright)
            d0 = depth_table[t_inv, idx0]
            d1 = depth_table[t_inv, idx1]
            depth = d0 * (1 - frac) + d1 * frac
            norm = (self._max_r_xy - depth) / (2.0 * self._max_r_xy + 1e-8)
            intensity = (0.15 + 0.85 * norm).astype(np.float32)

        # Pre-calculate voltages/delays and swap buffers
        self._prepare_output(xy, blanking, intensity)
        
        # Attribute stats
        effective_samples = int(self.sample_rate / abs(self.freq)) if self.freq != 0 else frames
        self._increment_compute_stats(_time.perf_counter() - t_compute_start, 
                                      self._n_draw_vectors, 
                                      self._n_lifts, 
                                      effective_samples)

        with self._lock:
            self._fill_buffer(outdata, frames)

        self._apply_noise(outdata, frames)
        
        # Zero spare channel
        if self.channels >= 4:
            outdata[:, 3] = 0.0

        self.global_sample += frames

        if has_stats:
            tend = _time.perf_counter()
            self.stats['callback_time'] += (tend - t_start)
            self.stats['callback_count'] += 1
            self.stats['last_callback_end'] = tend

    def _on_start(self):
        n_verts = int((~self.base_blanking).sum())
        n_lifts = (int(self.base_blanking.sum()) // max(1, self.pen_lift)
                   if self.pen_lift > 0 else 0)
        print(f"Sinc surface: {self.cells}x{self.cells} grid ({n_verts} vertices)")
        print(f"  cycles: {self.cycles}, speed: {self.speed}, zscale: {self.zscale}")
        print(f"  elevation: {self.elevation}°, azimuth: {self.azimuth_deg}°, rot_freq: {self.rot_freq} Hz")
        print(f"  trace: {self.freq} Hz")
        print(f"  path: {len(self.base_path)} samples, {n_lifts} pen lifts")
        print("  Press Ctrl+C to stop.")
