"""3D wireframe platonic solids with smooth rotation and perspective projection."""

import numpy as np
from collections import defaultdict
import time as _time

from .base import VectorScopePlayer

# Golden ratio
_PHI = (1 + np.sqrt(5)) / 2

# ---------------------------------------------------------------------------
# Geometry: vertices (unit-sphere-inscribed) and edge lists
# ---------------------------------------------------------------------------

def _normalize_to_unit_sphere(verts):
    """Scale vertices so the furthest sits on the unit sphere."""
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms.max()


def _tetrahedron():
    verts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ], dtype=np.float32) / np.sqrt(3)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return verts, edges


def _cube():
    verts = np.array([
        [s1, s2, s3]
        for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)
    ], dtype=np.float32) / np.sqrt(3)
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            # Connected if they differ in exactly one coordinate
            if np.sum(verts[i] != verts[j]) == 1:
                edges.append((i, j))
    return verts, edges


def _octahedron():
    verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=np.float32)
    edges = []
    for i in range(6):
        for j in range(i + 1, 6):
            # Connected if not on the same axis (not antipodal)
            if np.dot(verts[i], verts[j]) == 0:
                edges.append((i, j))
    return verts, edges


def _dodecahedron():
    p, ip = _PHI, 1.0 / _PHI
    verts = []
    # Cube vertices
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for s3 in (-1, 1):
                verts.append([s1, s2, s3])
    # Rectangle vertices on each axis-aligned plane
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts.append([0, s1 * ip, s2 * p])
            verts.append([s1 * ip, s2 * p, 0])
            verts.append([s1 * p, 0, s2 * ip])
    verts = np.array(verts, dtype=np.float32)
    verts = _normalize_to_unit_sphere(verts)

    # Connect vertices that are closest neighbours.
    # Each vertex has degree 3 -> 20*3/2 = 30 edges.
    from itertools import combinations
    dists = {}
    for i, j in combinations(range(len(verts)), 2):
        dists[(i, j)] = np.linalg.norm(verts[i] - verts[j])
    sorted_edges = sorted(dists, key=dists.get)
    edges = sorted_edges[:30]
    return verts, edges


def _icosahedron():
    verts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts.append([0, s1, s2 * _PHI])
            verts.append([s1, s2 * _PHI, 0])
            verts.append([s2 * _PHI, 0, s1])
    verts = np.array(verts, dtype=np.float32)
    verts = _normalize_to_unit_sphere(verts)

    from itertools import combinations
    dists = {}
    for i, j in combinations(range(len(verts)), 2):
        dists[(i, j)] = np.linalg.norm(verts[i] - verts[j])
    sorted_edges = sorted(dists, key=dists.get)
    edges = sorted_edges[:30]
    return verts, edges


SOLIDS = {
    'tetrahedron': _tetrahedron,
    'cube':        _cube,
    'octahedron':  _octahedron,
    'dodecahedron': _dodecahedron,
    'icosahedron': _icosahedron,
}

SOLID_NAMES = list(SOLIDS.keys())

# ---------------------------------------------------------------------------
# Edge ordering
# ---------------------------------------------------------------------------

def _find_euler_trail(edges):
    """Find an Euler trail or circuit using Hierholzer's algorithm.

    Returns a list of directed (start, end) edges, or None if no Euler
    trail exists (requires 0 or 2 odd-degree vertices).
    """
    if not edges:
        return []

    adj = defaultdict(list)
    for idx, (a, b) in enumerate(edges):
        adj[a].append((b, idx))
        adj[b].append((a, idx))

    odd_verts = [v for v in adj if len(adj[v]) % 2 != 0]
    if len(odd_verts) not in (0, 2):
        return None

    # Start from an odd-degree vertex if any, else any vertex
    start = odd_verts[0] if odd_verts else next(iter(adj))

    # Hierholzer's: walk edges, backtrack when stuck
    used = set()
    adj_idx = {v: 0 for v in adj}
    stack = [start]
    trail_verts = []

    while stack:
        v = stack[-1]
        found = False
        while adj_idx[v] < len(adj[v]):
            w, idx = adj[v][adj_idx[v]]
            adj_idx[v] += 1
            if idx not in used:
                used.add(idx)
                stack.append(w)
                found = True
                break
        if not found:
            trail_verts.append(stack.pop())

    if len(used) != len(edges):
        return None

    trail_verts.reverse()
    return [(trail_verts[i], trail_verts[i + 1])
            for i in range(len(trail_verts) - 1)]


def _greedy_order(edges):
    """Order edges greedily to minimise pen lifts."""
    remaining = list(edges)
    if not remaining:
        return []

    ordered = [remaining.pop(0)]
    while remaining:
        last_end = ordered[-1][1]
        best = None
        for idx, (a, b) in enumerate(remaining):
            if a == last_end:
                best = (idx, (a, b))
                break
            elif b == last_end:
                best = (idx, (b, a))
                break
        if best is not None:
            idx, edge = best
            remaining.pop(idx)
            ordered.append(edge)
        else:
            ordered.append(remaining.pop(0))
    return ordered


def _order_edges(edges):
    """Order edges optimally: Euler trail if possible, else greedy."""
    trail = _find_euler_trail(edges)
    if trail is not None:
        return trail
    return _greedy_order(edges)


# ---------------------------------------------------------------------------
# PlatonicPlayer
# ---------------------------------------------------------------------------

class PlatonicPlayer(VectorScopePlayer):
    """3D wireframe platonic solid with smooth tumbling rotation."""

    def __init__(self, solid='cube', rot_freq=0.15, rx=None, ry=None, rz=None,
                 perspective=3.0, pen_lift=4, **kwargs):
        super().__init__(**kwargs)
        if solid not in SOLIDS:
            raise ValueError(f"Unknown solid '{solid}'. Choose from: {', '.join(SOLID_NAMES)}")
        self.solid = solid
        self.rot_freq = rot_freq
        # Irrational-ratio multipliers so the tumble never repeats exactly
        self.rx = rx if rx is not None else rot_freq * 1.0
        self.ry = ry if ry is not None else rot_freq * np.sqrt(2)
        self.rz = rz if rz is not None else rot_freq * (1 + np.sqrt(5)) / 2
        self.perspective = perspective
        self.pen_lift = pen_lift
        self._build_path()

    def _build_path(self):
        """Build the base 3D path and blanking mask.

        Path resolution matches the audio cycle (sample_rate / freq) so
        each path sample = one audio sample.
        """
        verts, edges = SOLIDS[self.solid]()
        ordered = _order_edges(edges)

        # Scale vertices to fit in [-0.95, 0.95]
        verts = verts * 0.95

        num_edges = len(ordered)
        if num_edges == 0:
            self.base_path = np.zeros((1, 3), dtype=np.float32)
            self.base_blanking = np.zeros(1, dtype=bool)
            return

        # Path length = one audio cycle so path samples = audio samples
        path_samples = int(self.sample_rate / self.freq)

        # Precompute connectivity
        connects_prev = [False] * num_edges
        for i in range(num_edges - 1):
            if ordered[i][1] == ordered[i + 1][0]:
                connects_prev[i + 1] = True
        needs_wrap = ordered[-1][1] != ordered[0][0]

        # Build segment list: (start_vertex, end_vertex, is_pen_lift)
        # and compute distances for proportional sample allocation
        segments = []
        for i, (a, b) in enumerate(ordered):
            if i > 0 and not connects_prev[i]:
                prev_b = ordered[i - 1][1]
                segments.append((prev_b, a, True))   # pen lift
            segments.append((a, b, False))            # edge
        if needs_wrap:
            segments.append((ordered[-1][1], ordered[0][0], True))

        # Allocate samples proportional to distance so beam velocity
        # is uniform.
        distances = [max(np.linalg.norm(verts[b] - verts[a]), 1e-8)
                     for a, b, _ in segments]
        total_dist = sum(distances)
        seg_samples = [max(2, round(d / total_dist * path_samples))
                       for d in distances]

        # Enforce minimum pen lift samples
        for idx, (_, _, is_lift) in enumerate(segments):
            if is_lift:
                seg_samples[idx] = max(seg_samples[idx], self.pen_lift)

        # Build path with sharp corners
        path_points = []
        blanking_flags = []

        for (a, b, is_lift), n in zip(segments, seg_samples):
            tc = np.linspace(0, 1, n, dtype=np.float32)
            seg = verts[a] * (1 - tc[:, np.newaxis]) + \
                  verts[b] * tc[:, np.newaxis]
            path_points.append(seg)
            blanking_flags.append(np.full(n, is_lift, dtype=bool))

        self.base_path = np.vstack(path_points).astype(np.float32)
        self.base_blanking = np.concatenate(blanking_flags)
        
        # Performance metadata
        self._n_draw_vectors = len([s for s in segments if not s[2]])
        self._n_lifts = len([s for s in segments if s[2]])

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
        frac = (frac_idx - np.floor(frac_idx)).astype(np.float32)
        xyz = (self.base_path[idx0] * (1 - frac[:, np.newaxis]) +
               self.base_path[idx1] * frac[:, np.newaxis])
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Rotation is locked per trace cycle so every visit to the same
        # vertex within one cycle gets the exact same rotation angle.
        t = (self.global_sample + np.arange(frames, dtype=np.float64)) / self.sample_rate
        t_rot = np.floor(self.freq * t) / self.freq

        # Rx rotation (around X axis)
        a = 2 * np.pi * self.rx * t_rot
        cos_a, sin_a = np.cos(a), np.sin(a)
        x1 = x
        y1 = y * cos_a - z * sin_a
        z1 = y * sin_a + z * cos_a

        # Ry rotation (around Y axis)
        a = 2 * np.pi * self.ry * t_rot
        cos_a, sin_a = np.cos(a), np.sin(a)
        x2 = x1 * cos_a + z1 * sin_a
        z2 = -x1 * sin_a + z1 * cos_a
        y2 = y1

        # Rz rotation (around Z axis)
        a = 2 * np.pi * self.rz * t_rot
        cos_a, sin_a = np.cos(a), np.sin(a)
        x3 = x2 * cos_a - y2 * sin_a
        y3 = x2 * sin_a + y2 * cos_a

        # Perspective projection
        scale = self.perspective / (self.perspective + z2)
        
        xy = np.empty((frames, 2), dtype=np.float32)
        xy[:, 0] = x3 * scale * self.amp
        xy[:, 1] = y3 * scale * self.amp

        # Z-channel: blanking from pen-lift segments
        blanking = None
        intensity = None
        if self.z_enabled:
            blanking = self.base_blanking[idx0]
            # Depth intensity: near faces bright, far faces dim
            # Lower z2 = closer to camera (larger perspective scale)
            z_range = z2.max() - z2.min()
            if z_range > 1e-8:
                intensity = (0.3 + 0.7 * (z2.max() - z2) / z_range).astype(np.float32)
            else:
                intensity = np.ones(frames, dtype=np.float32)

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
        self._push_web_output(outdata, frames)

        self.global_sample += frames

        if has_stats:
            tend = _time.perf_counter()
            self.stats['callback_time'] += (tend - t_start)
            self.stats['callback_count'] += 1
            self.stats['last_callback_end'] = tend

    def _on_start(self):
        pen_lifts = int(self.base_blanking.sum()) // max(1, self.pen_lift)
        print(f"Platonic solid: {self.solid}")
        print(f"  trace: {self.freq} Hz, rotation: rx={self.rx:.3f} ry={self.ry:.3f} rz={self.rz:.3f} Hz")
        print(f"  perspective: {self.perspective}")
        print(f"  path: {len(self.base_path)} samples, {pen_lifts} pen lifts")
        print("  Press Ctrl+C to stop.")
