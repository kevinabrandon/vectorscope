"""Text rendering for oscilloscope XY display."""

import threading
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

    # Allocate samples across contours proportionally to contour length
    lengths = []
    for p in polys:
        d = np.diff(p, axis=0)
        lengths.append(float(np.sqrt((d**2).sum(axis=1)).sum()))
    total_len = sum(lengths) if lengths else 1.0

    # Build XY sequence with pen lifts
    out = []
    for p, L in zip(polys, lengths):
        n = int(max(10, (samples - pen_lift_samples*len(polys)) * (L / total_len)))
        pts = resample_polyline(p, n)
        out.append(pts)
        out.append(np.zeros((pen_lift_samples, 2), dtype=np.float64))

    xy = np.vstack(out) if out else np.zeros((samples,2), dtype=np.float64)

    # Seamless loop: smoothly return to the starting point
    return_samples = max(1, int(0.05 * samples))
    start = xy[0].copy()
    end = xy[-1].copy()

    u = np.linspace(0.0, 1.0, return_samples, endpoint=False)
    w = 0.5 - 0.5 * np.cos(np.pi * u)
    ret = (1.0 - w)[:, None] * end + w[:, None] * start

    xy = np.vstack([xy, ret])

    if len(xy) < samples:
        pad = np.zeros((samples - len(xy), 2), dtype=np.float64)
        xy = np.vstack([xy, pad])
    else:
        xy = xy[:samples]

    xy[-1] = xy[0]

    return xy


class TextPlayer(VectorScopePlayer):
    """Real-time text display with interactive input."""

    def __init__(self, text="Hello", font="DejaVu Sans", curve_pts=30,
                 pen_lift_samples=0, interactive=False, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.font = font
        self.curve_pts = curve_pts
        self.pen_lift_samples = pen_lift_samples
        self.interactive = interactive
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

    def _input_loop(self):
        """Background thread that reads input and updates text."""
        while True:
            try:
                line = input()
                if line:
                    print(f"Text: {line}")
                    self._update_text(line)
            except EOFError:
                break

    def _on_start(self):
        if self.interactive:
            print(f"Displaying: {self.text}")
            print("Type new text and press Enter to update (Ctrl+C to quit)")
            self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self._input_thread.start()
        else:
            print(f"Displaying: {self.text}")
            print("Press Ctrl+C to stop.")


def generate_wav(text, output, rate, secs, amp, font, curve_pts, penlift):
    """Generate a WAV file."""
    samples = int(rate * secs)
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
