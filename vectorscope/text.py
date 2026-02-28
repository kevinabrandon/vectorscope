"""Text rendering for oscilloscope XY display."""

import numpy as np
import soundfile as sf

from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path

from .base import VectorScopePlayer


def path_to_polylines(mpl_path: Path, points_per_curve: int = 30):
    """
    Convert a Matplotlib Path (with CURVE3/CURVE4) into a list of polylines.
    Returns list[np.ndarray] where each polyline is Nx2 float array.
    """
    polys = []
    cur = []

    verts = mpl_path.vertices
    codes = mpl_path.codes
    if codes is None:
        return [verts.copy()]

    i = 0
    last = None
    while i < len(verts):
        code = codes[i]
        v = verts[i]

        if code == Path.MOVETO:
            if len(cur) > 1:
                polys.append(np.array(cur, dtype=np.float64))
            cur = [v]
            last = v
            i += 1

        elif code == Path.LINETO:
            cur.append(v)
            last = v
            i += 1

        elif code == Path.CLOSEPOLY:
            if len(cur) > 1:
                cur.append(cur[0])
                polys.append(np.array(cur, dtype=np.float64))
            cur = []
            last = None
            i += 1

        elif code == Path.CURVE3:
            if last is None:
                i += 1
                continue
            p0 = last
            p1 = v
            p2 = verts[i + 1]
            ts = np.linspace(0.0, 1.0, points_per_curve, endpoint=True)
            curve = (1-ts)[:,None]**2 * p0 + 2*(1-ts)[:,None]*ts[:,None]*p1 + ts[:,None]**2 * p2
            cur.extend(curve[1:].tolist())
            last = p2
            i += 2

        elif code == Path.CURVE4:
            if last is None:
                i += 1
                continue
            p0 = last
            p1 = v
            p2 = verts[i + 1]
            p3 = verts[i + 2]
            ts = np.linspace(0.0, 1.0, points_per_curve, endpoint=True)
            curve = ((1-ts)[:,None]**3 * p0 +
                     3*(1-ts)[:,None]**2 * ts[:,None] * p1 +
                     3*(1-ts)[:,None] * ts[:,None]**2 * p2 +
                     ts[:,None]**3 * p3)
            cur.extend(curve[1:].tolist())
            last = p3
            i += 3

        else:
            i += 1

    if len(cur) > 1:
        polys.append(np.array(cur, dtype=np.float64))

    return polys


def normalize_polylines(polys):
    """Center and scale polylines into [-1, 1] range (preserving aspect)."""
    all_pts = np.vstack([p for p in polys if len(p) > 0])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    span = (max_xy - min_xy).max()
    if span <= 0:
        span = 1.0
    out = []
    for p in polys:
        q = (p - center) / (span / 2.0)
        out.append(q)
    return out


def resample_polyline(p, n):
    """Resample a polyline p (Nx2) to n points at approximately constant speed."""
    if len(p) < 2:
        return np.repeat(p[:1], n, axis=0) if len(p) == 1 else np.zeros((n,2), dtype=np.float64)
    diffs = np.diff(p, axis=0)
    seglen = np.sqrt((diffs**2).sum(axis=1))
    s = np.concatenate(([0.0], np.cumsum(seglen)))
    total = s[-1]
    if total <= 0:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, total, n)
    x = np.interp(t, s, p[:,0])
    y = np.interp(t, s, p[:,1])
    return np.column_stack((x,y))


def _optimize_contour_order(polys):
    """Reorder contours and rotate closed ones to minimize beam travel.

    For each contour after the first, rotates the polyline so tracing
    begins at the point closest to where the previous contour ended.
    Then reorders contours greedily by nearest start point.
    """
    if len(polys) <= 1:
        return polys

    def _rotate_to_nearest(poly, target):
        """Rotate a closed polyline so it starts nearest to target."""
        if not np.allclose(poly[0], poly[-1], atol=1e-10):
            return poly  # open contour, can't rotate
        # Exclude duplicate closing point for distance calc
        dists = ((poly[:-1] - target) ** 2).sum(axis=1)
        best = int(np.argmin(dists))
        if best == 0:
            return poly
        return np.vstack([poly[best:-1], poly[:best + 1]])

    # Greedy nearest-neighbor ordering
    remaining = list(range(len(polys)))
    ordered = [remaining.pop(0)]
    while remaining:
        prev_end = polys[ordered[-1]][-1]
        best_idx = min(remaining,
                       key=lambda j: ((polys[j][0] - prev_end) ** 2).sum())
        remaining.remove(best_idx)
        ordered.append(best_idx)

    # Rotate each contour to start near previous end
    result = [polys[ordered[0]]]
    for k in range(1, len(ordered)):
        prev_end = result[-1][-1]
        result.append(_rotate_to_nearest(polys[ordered[k]], prev_end))

    # Rotate the first contour to start near where the last one ends,
    # so the overall loop also closes cleanly.
    last_end = result[-1][-1]
    result[0] = _rotate_to_nearest(result[0], last_end)

    return result


def build_xy_from_text(text, font_size=1.0, font_family="DejaVu Sans", curve_pts=30,
                       samples=48000*5, pen_lift_samples=300):
    """Convert text to XY coordinate array for oscilloscope display."""
    # Escape special characters that trigger matplotlib mathtext
    text = text.replace('$', r'\$')

    fp = FontProperties(family=font_family)
    tp = TextPath((0, 0), text, size=font_size, prop=fp)

    polys = path_to_polylines(tp, points_per_curve=curve_pts)
    polys = [p for p in polys if len(p) >= 2]

    if not polys:
        # Empty text - return silence
        return np.zeros((samples, 2), dtype=np.float64)

    polys = normalize_polylines(polys)
    polys = _optimize_contour_order(polys)

    # Allocate samples across contours proportionally to arc length
    pen_lift_total = pen_lift_samples * max(0, len(polys) - 1)
    contour_budget = samples - pen_lift_total

    lengths = []
    for p in polys:
        d = np.diff(p, axis=0)
        lengths.append(float(np.sqrt((d**2).sum(axis=1)).sum()))
    total_len = sum(lengths) if lengths else 1.0

    # Build XY sequence with pen lifts between contours
    out = []
    for i, (p, L) in enumerate(zip(polys, lengths)):
        n = int(max(10, contour_budget * (L / total_len)))
        pts = resample_polyline(p, n)
        out.append(pts)
        if pen_lift_samples > 0 and i < len(polys) - 1:
            out.append(np.zeros((pen_lift_samples, 2), dtype=np.float64))

    xy = np.vstack(out) if out else np.zeros((samples, 2), dtype=np.float64)

    if len(xy) < samples:
        pad = np.tile(xy[-1], (samples - len(xy), 1))
        xy = np.vstack([xy, pad])
    else:
        xy = xy[:samples]

    # Force seamless loop wrap
    xy[-1] = xy[0]

    return xy


class TextPlayer(VectorScopePlayer):
    """Real-time text display with interactive input."""

    def __init__(self, text="Hello", font="DejaVu Sans", curve_pts=30,
                 pen_lift_samples=0, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.font = font
        self.curve_pts = curve_pts
        self.pen_lift_samples = pen_lift_samples
        self._update_text(text)

    def _update_text(self, text):
        """Regenerate XY data for new text."""
        self.text = text
        xy = build_xy_from_text(
            text,
            font_family=self.font,
            curve_pts=self.curve_pts,
            samples=self.samples,
            pen_lift_samples=self.pen_lift_samples
        )
        self.xy_data = np.clip(xy * self.amp, -1.0, 1.0).astype(np.float32)
        self.position = 0

    def _on_start(self):
        print(f"Displaying: {self.text}")
        print("Press Ctrl+C to stop.")


def generate_wav(text, output, rate, freq, amp, font, curve_pts, penlift):
    """Generate a WAV file."""
    samples = int(rate / abs(freq))
    xy = build_xy_from_text(
        text,
        font_size=1.0,
        font_family=font,
        curve_pts=curve_pts,
        samples=samples,
        pen_lift_samples=penlift
    )

    xy = np.clip(xy * float(amp), -1.0, 1.0)
    xy[-1] = xy[0]

    sf.write(output, xy.astype(np.float32), rate)
    print(f"Wrote {output} ({rate} Hz, {samples} frames). L=X, R=Y")
